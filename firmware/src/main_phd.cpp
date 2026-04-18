// rfs_on_teensy firmware: USB-serial loop driving brew RFS filters on Teensy 4.1.
//
// Wire framing:  [u8 type][u16 seq][u16 len][protobuf payload]
// Each payload is a serialized rfs.<MessageType> from rfs.proto. Empty payloads
// are used for PING / PONG / RESET.

#include <Arduino.h>

// Arduino macros that collide with Eigen / local identifiers.
#undef B0
#undef B1
#undef B2
#undef B3
#undef B4
#undef B5
#undef B6
#undef B7
#undef F

#include <Eigen/Dense>

#include "brew/advanced/multi_target/phd.hpp"
#include "brew/core/filters/ekf.hpp"
#include "brew/core/filters/ggiw_ekf.hpp"
#include "brew/core/filters/trajectory_gaussian_ekf.hpp"
#include "brew/core/filters/trajectory_ggiw_ekf.hpp"
#include "brew/core/dynamics/single_integrator.hpp"
#include "brew/core/models/gaussian.hpp"
#include "brew/core/models/ggiw.hpp"
#include "brew/core/models/trajectory.hpp"
#include "brew/core/models/mixture.hpp"

#include <pb.h>
#include <pb_encode.h>
#include <pb_decode.h>
#include "rfs.pb.h"

#include <memory>

// Application-layer message types (mirrors the comment in rfs.proto).
enum : uint8_t {
    MSG_PING   = 1,
    MSG_PONG   = 2,
    MSG_RESET  = 3,
    MSG_CONFIG = 4,
    MSG_STEP   = 5,
    MSG_TRACKS = 6,
    MSG_LOG    = 7,
    MSG_ERROR  = 0x7F
};

constexpr size_t MAX_PAYLOAD = 8192;

#pragma pack(push, 1)
struct WireHeader {
    uint8_t  type;
    uint16_t seq;
    uint16_t len;
};
#pragma pack(pop)
static_assert(sizeof(WireHeader) == 5, "WireHeader must be 5 bytes packed");

// Compile-time sizes. Fixing these means Eigen uses stack-allocated matrices
// for every Gaussian/GGIW operation instead of heap-allocated dynamic ones.
// Trajectory's MaxHistory stays Eigen::Dynamic so the runtime cfg.window_size
// is actually honored — brew's advance_window ignores the soft-cap argument
// whenever MaxHistory is a compile-time constant.
constexpr int STATE_DIM   = 4;    // [x, y, vx, vy]
constexpr int SPATIAL_DIM = 2;    // [x, y]
constexpr int MEAS_DIM    = 2;    // position-only

using GaussD      = brew::models::Gaussian<double, STATE_DIM>;
using EKFD        = brew::filters::EKF<double, STATE_DIM>;
using PHDGauss    = brew::multi_target::PHD<GaussD>;
using MixGauss    = brew::models::Mixture<GaussD>;

using TrajGauss   = brew::models::Trajectory<GaussD>; 
using TrajEKFD    = brew::filters::TrajectoryGaussianEKF<double, STATE_DIM>;
using PHDTraj     = brew::multi_target::PHD<TrajGauss>;
using MixTraj     = brew::models::Mixture<TrajGauss>;

using GGIWD       = brew::models::GGIW<double, STATE_DIM, MEAS_DIM>;
using GGIWEKFD    = brew::filters::GGIWEKF<double, STATE_DIM, MEAS_DIM>;
using PHDGGIW     = brew::multi_target::PHD<GGIWD>;
using MixGGIW     = brew::models::Mixture<GGIWD>;

using TrajGGIW    = brew::models::Trajectory<GGIWD>; 
using TrajGGIWEKF = brew::filters::TrajectoryGGIWEKF<double, STATE_DIM, MEAS_DIM>;
using PHDGGIWTraj = brew::multi_target::PHD<TrajGGIW>;
using MixGGIWTraj = brew::models::Mixture<TrajGGIW>;

using DynaD      = brew::dynamics::SingleIntegrator<double, SPATIAL_DIM>;

namespace {

std::unique_ptr<PHDGauss>     g_tracker_gauss;
std::unique_ptr<PHDGGIW>      g_tracker_ggiw;
std::unique_ptr<PHDTraj>      g_tracker_tst;
std::unique_ptr<PHDGGIWTraj>  g_tracker_tst_ggiw;
rfs_Config                    g_cfg = rfs_Config_init_zero;
bool                          g_have_cfg = false;

uint8_t payload_buf[MAX_PAYLOAD];
uint8_t out_buf[MAX_PAYLOAD];

// ---------- Frame I/O ----------

void send_raw(uint8_t type, uint16_t seq, const void* payload, uint16_t len) {
    WireHeader h{type, seq, len};
    Serial.write(reinterpret_cast<const uint8_t*>(&h), sizeof(h));
    if (len) Serial.write(reinterpret_cast<const uint8_t*>(payload), len);
}

template <typename T>
bool send_pb(uint8_t type, uint16_t seq, const pb_msgdesc_t* desc, const T& msg) {
    pb_ostream_t os = pb_ostream_from_buffer(out_buf, sizeof(out_buf));
    if (!pb_encode(&os, desc, &msg)) return false;
    send_raw(type, seq, out_buf, (uint16_t)os.bytes_written);
    return true;
}

void send_log(uint16_t seq, const char* msg) {
    rfs_Log m = rfs_Log_init_zero;
    strncpy(m.msg, msg, sizeof(m.msg) - 1);
    send_pb(MSG_LOG, seq, rfs_Log_fields, m);
}

void send_error(uint16_t seq, const char* msg) {
    rfs_Error m = rfs_Error_init_zero;
    strncpy(m.msg, msg, sizeof(m.msg) - 1);
    send_pb(MSG_ERROR, seq, rfs_Error_fields, m);
}

// ---------- Config helpers ----------

template <size_t N>
void copy_repeated(double* dst, int dst_n, const float (&src)[N], int src_n, double fill = 0.0) {
    for (int i = 0; i < dst_n; ++i) dst[i] = (i < src_n) ? (double)src[i] : fill;
}

// ---------- Tracker rebuild ----------

void rebuild_tracker_gauss(const rfs_Config& cfg) {
    auto ekf = std::make_unique<EKFD>();
    auto dyn = std::make_shared<DynaD>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(MEAS_DIM, STATE_DIM);
    for (int i = 0; i < MEAS_DIM; ++i) H(i, i) = 1.0;
    ekf->set_measurement_jacobian(H);

    Eigen::MatrixXd Q_small = Eigen::MatrixXd::Zero(SPATIAL_DIM, SPATIAL_DIM);
    for (int i = 0; i < SPATIAL_DIM && i < (int)cfg.process_noise_diag_count; ++i)
        Q_small(i, i) = cfg.process_noise_diag[i];
    Eigen::MatrixXd G = dyn->get_input_mat(1.0, DynaD::Vector::Zero());
    ekf->set_process_noise(G * Q_small * G.transpose());

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(MEAS_DIM, MEAS_DIM);
    for (int i = 0; i < MEAS_DIM && i < (int)cfg.measurement_noise_diag_count; ++i)
        R(i, i) = cfg.measurement_noise_diag[i];
    ekf->set_measurement_noise(R);

    auto birth = std::make_unique<MixGauss>();
    GaussD::Vector bm = GaussD::Vector::Zero();
    for (int i = 0; i < STATE_DIM; ++i)
        bm(i) = (i < (int)cfg.birth_mean_count) ? cfg.birth_mean[i] : 0.0;
    GaussD::Matrix bc = GaussD::Matrix::Zero();
    for (int i = 0; i < STATE_DIM; ++i)
        bc(i, i) = (i < (int)cfg.birth_cov_diag_count) ? cfg.birth_cov_diag[i] : 1.0;
    birth->add_component(std::make_unique<GaussD>(bm, bc), cfg.birth_weight);
    auto intensity = birth->clone();

    auto phd = std::make_unique<PHDGauss>();
    phd->set_filter(std::move(ekf));
    phd->set_birth_model(std::move(birth));
    phd->set_intensity(std::move(intensity));
    phd->set_prob_detection(cfg.prob_detection);
    phd->set_prob_survive(cfg.prob_survive);
    phd->set_clutter_rate(cfg.clutter_rate);
    phd->set_clutter_density(cfg.clutter_density);
    phd->set_prune_threshold(cfg.prune_threshold);
    phd->set_merge_threshold(cfg.merge_threshold);
    phd->set_max_components(cfg.max_components);
    phd->set_extract_threshold(cfg.extract_threshold);
    phd->set_gate_threshold(cfg.gate_threshold);
    phd->set_max_history(cfg.max_history);

    g_tracker_gauss = std::move(phd);
    g_tracker_ggiw.reset();
    g_tracker_tst.reset();
    g_tracker_tst_ggiw.reset();
}

void rebuild_tracker_ggiw(const rfs_Config& cfg) {
    auto ekf = std::make_unique<GGIWEKFD>();
    auto dyn = std::make_shared<DynaD>();
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(MEAS_DIM, STATE_DIM);
    for (int i = 0; i < MEAS_DIM; ++i) H(i, i) = 1.0;
    ekf->set_measurement_jacobian(H);

    Eigen::MatrixXd Q_small = Eigen::MatrixXd::Zero(SPATIAL_DIM, SPATIAL_DIM);
    for (int i = 0; i < SPATIAL_DIM && i < (int)cfg.process_noise_diag_count; ++i)
        Q_small(i, i) = cfg.process_noise_diag[i];
    Eigen::MatrixXd G = dyn->get_input_mat(1.0, DynaD::Vector::Zero());
    ekf->set_process_noise(G * Q_small * G.transpose());

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(MEAS_DIM, MEAS_DIM);
    for (int i = 0; i < MEAS_DIM && i < (int)cfg.measurement_noise_diag_count; ++i)
        R(i, i) = cfg.measurement_noise_diag[i];
    ekf->set_measurement_noise(R);

    ekf->set_temporal_decay(cfg.eta);
    ekf->set_forgetting_factor(cfg.tau);
    ekf->set_scaling_parameter(cfg.rho);

    auto birth = std::make_unique<MixGGIW>();
    GaussD::Vector bm = GaussD::Vector::Zero();
    for (int i = 0; i < STATE_DIM; ++i)
        bm(i) = (i < (int)cfg.birth_mean_count) ? cfg.birth_mean[i] : 0.0;
    GaussD::Matrix bc = GaussD::Matrix::Zero();
    for (int i = 0; i < STATE_DIM; ++i)
        bc(i, i) = (i < (int)cfg.birth_cov_diag_count) ? cfg.birth_cov_diag[i] : 1.0;

    GGIWD::ExtentMatrix V0 = GGIWD::ExtentMatrix::Zero();
    V0(0, 0) = (cfg.birth_extent_count > 0) ? cfg.birth_extent[0] : 1.0;
    V0(1, 1) = (cfg.birth_extent_count > 1) ? cfg.birth_extent[1] : 1.0;
    V0(0, 1) = (cfg.birth_extent_count > 2) ? cfg.birth_extent[2] : 0.0;
    V0(1, 0) = V0(0, 1);

    birth->add_component(
        std::make_unique<GGIWD>(cfg.birth_alpha, cfg.birth_beta,
                                bm, bc, cfg.birth_v, V0),
        cfg.birth_weight);
    auto intensity = birth->clone();

    auto phd = std::make_unique<PHDGGIW>();
    phd->set_filter(std::move(ekf));
    phd->set_birth_model(std::move(birth));
    phd->set_intensity(std::move(intensity));
    phd->set_prob_detection(cfg.prob_detection);
    phd->set_prob_survive(cfg.prob_survive);
    phd->set_clutter_rate(cfg.clutter_rate);
    phd->set_clutter_density(cfg.clutter_density);
    phd->set_prune_threshold(cfg.prune_threshold);
    phd->set_merge_threshold(cfg.merge_threshold);
    phd->set_max_components(cfg.max_components);
    phd->set_extract_threshold(cfg.extract_threshold);
    phd->set_gate_threshold(cfg.gate_threshold);
    phd->set_max_history(cfg.max_history);

    g_tracker_ggiw = std::move(phd);
    g_tracker_gauss.reset();
    g_tracker_tst.reset();
    g_tracker_tst_ggiw.reset();
}

void rebuild_tracker_tst(const rfs_Config& cfg) {
    auto ekf = std::make_unique<TrajEKFD>();
    auto dyn = std::make_shared<DynaD>();
    ekf->set_dynamics(dyn);
    const int win = cfg.window_size > 0 ? (int)cfg.window_size : 3;
    ekf->set_window_size(win);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(MEAS_DIM, STATE_DIM);
    for (int i = 0; i < MEAS_DIM; ++i) H(i, i) = 1.0;
    ekf->set_measurement_jacobian(H);

    Eigen::MatrixXd Q_small = Eigen::MatrixXd::Zero(SPATIAL_DIM, SPATIAL_DIM);
    for (int i = 0; i < SPATIAL_DIM && i < (int)cfg.process_noise_diag_count; ++i)
        Q_small(i, i) = cfg.process_noise_diag[i];
    Eigen::MatrixXd G = dyn->get_input_mat(1.0, DynaD::Vector::Zero());
    ekf->set_process_noise(G * Q_small * G.transpose());

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(MEAS_DIM, MEAS_DIM);
    for (int i = 0; i < MEAS_DIM && i < (int)cfg.measurement_noise_diag_count; ++i)
        R(i, i) = cfg.measurement_noise_diag[i];
    ekf->set_measurement_noise(R);

    GaussD::Vector bm = GaussD::Vector::Zero();
    for (int i = 0; i < STATE_DIM; ++i)
        bm(i) = (i < (int)cfg.birth_mean_count) ? cfg.birth_mean[i] : 0.0;
    GaussD::Matrix bc = GaussD::Matrix::Zero();
    for (int i = 0; i < STATE_DIM; ++i)
        bc(i, i) = (i < (int)cfg.birth_cov_diag_count) ? cfg.birth_cov_diag[i] : 1.0;

    auto birth_traj = std::make_unique<TrajGauss>(STATE_DIM, GaussD(bm, bc));
    // Per-trajectory state_history cap. 0 disables recording entirely.
    // Every trajectory PHD clones from this birth inherits the same cap.
    birth_traj->set_max_history(cfg.trajectory_history);
    auto birth = std::make_unique<MixTraj>();
    birth->add_component(std::move(birth_traj), cfg.birth_weight);
    auto intensity = birth->clone();

    auto phd = std::make_unique<PHDTraj>();
    phd->set_filter(std::move(ekf));
    phd->set_birth_model(std::move(birth));
    phd->set_intensity(std::move(intensity));
    phd->set_prob_detection(cfg.prob_detection);
    phd->set_prob_survive(cfg.prob_survive);
    phd->set_clutter_rate(cfg.clutter_rate);
    phd->set_clutter_density(cfg.clutter_density);
    phd->set_prune_threshold(cfg.prune_threshold);
    phd->set_merge_threshold(cfg.merge_threshold);
    phd->set_max_components(cfg.max_components);
    phd->set_extract_threshold(cfg.extract_threshold);
    phd->set_gate_threshold(cfg.gate_threshold);
    phd->set_max_history(cfg.max_history);

    g_tracker_tst = std::move(phd);
    g_tracker_gauss.reset();
    g_tracker_ggiw.reset();
    g_tracker_tst_ggiw.reset();
}

void rebuild_tracker_tst_ggiw(const rfs_Config& cfg) {
    auto ekf = std::make_unique<TrajGGIWEKF>();
    auto dyn = std::make_shared<DynaD>();
    ekf->set_dynamics(dyn);
    const int win = cfg.window_size > 0 ? (int)cfg.window_size : 3;
    ekf->set_window_size(win);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(MEAS_DIM, STATE_DIM);
    for (int i = 0; i < MEAS_DIM; ++i) H(i, i) = 1.0;
    ekf->set_measurement_jacobian(H);

    Eigen::MatrixXd Q_small = Eigen::MatrixXd::Zero(SPATIAL_DIM, SPATIAL_DIM);
    for (int i = 0; i < SPATIAL_DIM && i < (int)cfg.process_noise_diag_count; ++i)
        Q_small(i, i) = cfg.process_noise_diag[i];
    Eigen::MatrixXd G = dyn->get_input_mat(1.0, DynaD::Vector::Zero());
    ekf->set_process_noise(G * Q_small * G.transpose());

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(MEAS_DIM, MEAS_DIM);
    for (int i = 0; i < MEAS_DIM && i < (int)cfg.measurement_noise_diag_count; ++i)
        R(i, i) = cfg.measurement_noise_diag[i];
    ekf->set_measurement_noise(R);

    ekf->set_temporal_decay(cfg.eta);
    ekf->set_forgetting_factor(cfg.tau);
    ekf->set_scaling_parameter(cfg.rho);

    GaussD::Vector bm = GaussD::Vector::Zero();
    for (int i = 0; i < STATE_DIM; ++i)
        bm(i) = (i < (int)cfg.birth_mean_count) ? cfg.birth_mean[i] : 0.0;
    GaussD::Matrix bc = GaussD::Matrix::Zero();
    for (int i = 0; i < STATE_DIM; ++i)
        bc(i, i) = (i < (int)cfg.birth_cov_diag_count) ? cfg.birth_cov_diag[i] : 1.0;

    GGIWD::ExtentMatrix V0 = GGIWD::ExtentMatrix::Zero();
    V0(0, 0) = (cfg.birth_extent_count > 0) ? cfg.birth_extent[0] : 1.0;
    V0(1, 1) = (cfg.birth_extent_count > 1) ? cfg.birth_extent[1] : 1.0;
    V0(0, 1) = (cfg.birth_extent_count > 2) ? cfg.birth_extent[2] : 0.0;
    V0(1, 0) = V0(0, 1);

    GGIWD initial(cfg.birth_alpha, cfg.birth_beta, bm, bc, cfg.birth_v, V0);
    auto birth_traj = std::make_unique<TrajGGIW>(STATE_DIM, std::move(initial));
    birth_traj->set_max_history(cfg.trajectory_history);
    auto birth = std::make_unique<MixGGIWTraj>();
    birth->add_component(std::move(birth_traj), cfg.birth_weight);
    auto intensity = birth->clone();

    auto phd = std::make_unique<PHDGGIWTraj>();
    phd->set_filter(std::move(ekf));
    phd->set_birth_model(std::move(birth));
    phd->set_intensity(std::move(intensity));
    phd->set_prob_detection(cfg.prob_detection);
    phd->set_prob_survive(cfg.prob_survive);
    phd->set_clutter_rate(cfg.clutter_rate);
    phd->set_clutter_density(cfg.clutter_density);
    phd->set_prune_threshold(cfg.prune_threshold);
    phd->set_merge_threshold(cfg.merge_threshold);
    phd->set_max_components(cfg.max_components);
    phd->set_extract_threshold(cfg.extract_threshold);
    phd->set_gate_threshold(cfg.gate_threshold);
    phd->set_max_history(cfg.max_history);

    g_tracker_tst_ggiw = std::move(phd);
    g_tracker_gauss.reset();
    g_tracker_ggiw.reset();
    g_tracker_tst.reset();
}

void rebuild_tracker(const rfs_Config& cfg) {
    g_cfg = cfg;
    g_have_cfg = true;
    switch (cfg.filter_kind) {
        case rfs_FilterKind_FILTER_GGIW_PHD:     rebuild_tracker_ggiw(cfg);     break;
        case rfs_FilterKind_FILTER_TST_GM_PHD:   rebuild_tracker_tst(cfg);      break;
        case rfs_FilterKind_FILTER_TST_GGIW_PHD: rebuild_tracker_tst_ggiw(cfg); break;
        default:                                 rebuild_tracker_gauss(cfg);    break;
    }
}

// ---------- Numeric hygiene ----------

template <typename Mix>
int purge_nan_gauss(Mix& mix) {
    std::vector<std::size_t> bad;
    for (std::size_t i = 0; i < mix.size(); ++i) {
        const auto& c = mix.component(i);
        const bool finite =
            std::isfinite(mix.weight(i)) &&
            c.mean().array().isFinite().all() &&
            c.covariance().array().isFinite().all();
        if (!finite) bad.push_back(i);
    }
    if (!bad.empty()) mix.remove_components(bad);
    return (int)bad.size();
}

template <typename Mix>
int purge_nan_ggiw(Mix& mix) {
    std::vector<std::size_t> bad;
    for (std::size_t i = 0; i < mix.size(); ++i) {
        const auto& c = mix.component(i);
        const bool finite =
            std::isfinite(mix.weight(i)) &&
            c.mean().array().isFinite().all() &&
            c.covariance().array().isFinite().all() &&
            std::isfinite(c.alpha()) && std::isfinite(c.beta()) &&
            std::isfinite(c.v()) && c.V().array().isFinite().all();
        if (!finite) bad.push_back(i);
    }
    if (!bad.empty()) mix.remove_components(bad);
    return (int)bad.size();
}

template <typename Mix>
int purge_nan_tst_ggiw(Mix& mix) {
    // Mix = Mixture<Trajectory<GGIW>>. The stacked mean/cov are the trajectory
    // blocks; GGIW shape parameters live on the current() inner dist.
    std::vector<std::size_t> bad;
    for (std::size_t i = 0; i < mix.size(); ++i) {
        const auto& traj = mix.component(i);
        const auto& g = traj.current();
        const bool finite =
            std::isfinite(mix.weight(i)) &&
            traj.mean().array().isFinite().all() &&
            traj.covariance().array().isFinite().all() &&
            std::isfinite(g.alpha()) && std::isfinite(g.beta()) &&
            std::isfinite(g.v()) && g.V().array().isFinite().all();
        if (!finite) bad.push_back(i);
    }
    if (!bad.empty()) mix.remove_components(bad);
    return (int)bad.size();
}

// ---------- Step ----------

Eigen::MatrixXd load_meas(const rfs_Step& step, int md) {
    const int n = step.meas_count;
    Eigen::MatrixXd Z(md, n);
    for (int j = 0; j < n; ++j) {
        const auto& d = step.meas[j];
        for (int i = 0; i < md; ++i) {
            Z(i, j) = (i < (int)d.coords_count) ? (double)d.coords[i] : 0.0;
        }
    }
    return Z;
}

void fill_track_kinematics(rfs_Track& tk, const Eigen::VectorXd& mean,
                           const Eigen::MatrixXd& cov, int sd) {
    const int n = std::min(sd, 8);
    tk.mean_count = n;
    tk.cov_diag_count = n;
    for (int k = 0; k < n; ++k) {
        tk.mean[k] = (float)mean(k);
        tk.cov_diag[k] = (float)cov(k, k);
    }
}

void emit_tracks_gauss(uint16_t seq, uint32_t timestep) {
    const auto& extracted = g_tracker_gauss->extracted_mixtures();
    const MixGauss* est = extracted.empty() ? nullptr : extracted.back().get();
    rfs_Tracks msg = rfs_Tracks_init_zero;
    msg.timestep = timestep;
    const int sd = g_cfg.state_dim;
    if (est) {
        const int n = std::min<int>((int)est->size(),
                                    (int)(sizeof(msg.tracks) / sizeof(msg.tracks[0])));
        msg.tracks_count = n;
        for (int i = 0; i < n; ++i) {
            msg.tracks[i] = rfs_Track_init_zero;
            msg.tracks[i].weight = (float)est->weight(i);
            const auto& g = est->component(i);
            fill_track_kinematics(msg.tracks[i], g.mean(), g.covariance(), sd);
        }
    }
    send_pb(MSG_TRACKS, seq, rfs_Tracks_fields, msg);
}

void emit_tracks_ggiw(uint16_t seq, uint32_t timestep) {
    const auto& extracted = g_tracker_ggiw->extracted_mixtures();
    const MixGGIW* est = extracted.empty() ? nullptr : extracted.back().get();
    rfs_Tracks msg = rfs_Tracks_init_zero;
    msg.timestep = timestep;
    const int sd = g_cfg.state_dim;
    if (est) {
        const int n = std::min<int>((int)est->size(),
                                    (int)(sizeof(msg.tracks) / sizeof(msg.tracks[0])));
        msg.tracks_count = n;
        for (int i = 0; i < n; ++i) {
            msg.tracks[i] = rfs_Track_init_zero;
            msg.tracks[i].weight = (float)est->weight(i);
            const auto& g = est->component(i);
            fill_track_kinematics(msg.tracks[i], g.mean(), g.covariance(), sd);
            msg.tracks[i].alpha = (float)g.alpha();
            msg.tracks[i].beta  = (float)g.beta();
            msg.tracks[i].v     = (float)g.v();
            const auto& V = g.V();
            if (V.rows() >= 2 && V.cols() >= 2) {
                msg.tracks[i].extent_count = 3;
                msg.tracks[i].extent[0] = (float)V(0, 0);
                msg.tracks[i].extent[1] = (float)V(1, 1);
                msg.tracks[i].extent[2] = (float)V(0, 1);
            }
        }
    }
    send_pb(MSG_TRACKS, seq, rfs_Tracks_fields, msg);
}

void emit_tracks_tst(uint16_t seq, uint32_t timestep) {
    const auto& extracted = g_tracker_tst->extracted_mixtures();
    const MixTraj* est = extracted.empty() ? nullptr : extracted.back().get();
    rfs_Tracks msg = rfs_Tracks_init_zero;
    msg.timestep = timestep;
    const int sd = g_cfg.state_dim;
    if (est) {
        const int n = std::min<int>((int)est->size(),
                                    (int)(sizeof(msg.tracks) / sizeof(msg.tracks[0])));
        msg.tracks_count = n;
        const int hist_cap = (int)(sizeof(msg.tracks[0].history)
                                   / sizeof(msg.tracks[0].history[0]));
        const int kdim = std::min(sd, 8);
        for (int i = 0; i < n; ++i) {
            msg.tracks[i] = rfs_Track_init_zero;
            msg.tracks[i].weight = (float)est->weight(i);
            const auto& traj = est->component(i);
            fill_track_kinematics(msg.tracks[i], traj.get_last_state(), traj.get_last_cov(), sd);

            // history: past finalized states from Trajectory::state_history
            // (oldest first). Bounded by cfg.trajectory_history, not by
            // window_size — so this can stretch far beyond the filter's
            // working window. Capped again by the protobuf buffer.
            //
            // Skip entry 0: the Trajectory ctor seeds state_history with the
            // birth/initial state, which for PHD is always at the birth mean
            // (usually origin) and makes every polyline render a bogus leg
            // from (0,0) to wherever the track actually formed.
            const auto& sh = traj.state_history();
            int out = 0;
            for (std::size_t h = 1; h < sh.size() && out + kdim <= hist_cap; ++h) {
                const auto& g = sh[h];
                for (int k = 0; k < kdim; ++k) {
                    msg.tracks[i].history[out++] = (float)g.mean()(k);
                }
            }
            msg.tracks[i].history_count = out;
        }
    }
    send_pb(MSG_TRACKS, seq, rfs_Tracks_fields, msg);
}

void emit_tracks_tst_ggiw(uint16_t seq, uint32_t timestep) {
    const auto& extracted = g_tracker_tst_ggiw->extracted_mixtures();
    const MixGGIWTraj* est = extracted.empty() ? nullptr : extracted.back().get();
    rfs_Tracks msg = rfs_Tracks_init_zero;
    msg.timestep = timestep;
    const int sd = g_cfg.state_dim;
    if (est) {
        const int n = std::min<int>((int)est->size(),
                                    (int)(sizeof(msg.tracks) / sizeof(msg.tracks[0])));
        msg.tracks_count = n;
        const int hist_cap = (int)(sizeof(msg.tracks[0].history)
                                   / sizeof(msg.tracks[0].history[0]));
        const int kdim = std::min(sd, 8);
        for (int i = 0; i < n; ++i) {
            msg.tracks[i] = rfs_Track_init_zero;
            msg.tracks[i].weight = (float)est->weight(i);
            const auto& traj = est->component(i);
            fill_track_kinematics(msg.tracks[i], traj.get_last_state(), traj.get_last_cov(), sd);

            // GGIW shape from the trajectory's current inner marginal.
            const auto& g = traj.current();
            msg.tracks[i].alpha = (float)g.alpha();
            msg.tracks[i].beta  = (float)g.beta();
            msg.tracks[i].v     = (float)g.v();
            const auto& V = g.V();
            if (V.rows() >= 2 && V.cols() >= 2) {
                msg.tracks[i].extent_count = 3;
                msg.tracks[i].extent[0] = (float)V(0, 0);
                msg.tracks[i].extent[1] = (float)V(1, 1);
                msg.tracks[i].extent[2] = (float)V(0, 1);
            }

            // Past kinematic states, oldest first, skipping the birth seed
            // entry for the same reason as emit_tracks_tst (birth mean polyline
            // artifact).
            const auto& sh = traj.state_history();
            int out = 0;
            for (std::size_t h = 1; h < sh.size() && out + kdim <= hist_cap; ++h) {
                const auto& sg = sh[h];
                for (int k = 0; k < kdim; ++k) {
                    msg.tracks[i].history[out++] = (float)sg.mean()(k);
                }
            }
            msg.tracks[i].history_count = out;
        }
    }
    send_pb(MSG_TRACKS, seq, rfs_Tracks_fields, msg);
}

void handle_step(uint16_t seq, const uint8_t* payload, uint16_t plen) {
    rfs_Step step = rfs_Step_init_zero;
    pb_istream_t is = pb_istream_from_buffer(payload, plen);
    if (!pb_decode(&is, rfs_Step_fields, &step)) {
        send_error(seq, "STEP decode failed");
        return;
    }
    const int md = g_cfg.meas_dim;

    const uint32_t t0 = micros();
    int dropped = 0;
    if (g_tracker_gauss) {
        Eigen::MatrixXd Z = load_meas(step, md);
        g_tracker_gauss->predict((int)step.timestep, (double)g_cfg.dt);
        if (step.meas_count > 0) g_tracker_gauss->correct(Z);
        dropped = purge_nan_gauss(g_tracker_gauss->intensity());
        g_tracker_gauss->cleanup();
        emit_tracks_gauss(seq, step.timestep);
    } else if (g_tracker_ggiw) {
        Eigen::MatrixXd Z = load_meas(step, md);
        g_tracker_ggiw->predict((int)step.timestep, (double)g_cfg.dt);
        if (step.meas_count > 0) g_tracker_ggiw->correct(Z);
        dropped = purge_nan_ggiw(g_tracker_ggiw->intensity());
        g_tracker_ggiw->cleanup();
        emit_tracks_ggiw(seq, step.timestep);
    } else if (g_tracker_tst) {
        Eigen::MatrixXd Z = load_meas(step, md);
        g_tracker_tst->predict((int)step.timestep, (double)g_cfg.dt);
        if (step.meas_count > 0) g_tracker_tst->correct(Z);
        dropped = purge_nan_gauss(g_tracker_tst->intensity());
        g_tracker_tst->cleanup();
        emit_tracks_tst(seq, step.timestep);
    } else if (g_tracker_tst_ggiw) {
        Eigen::MatrixXd Z = load_meas(step, md);
        g_tracker_tst_ggiw->predict((int)step.timestep, (double)g_cfg.dt);
        if (step.meas_count > 0) g_tracker_tst_ggiw->correct(Z);
        dropped = purge_nan_tst_ggiw(g_tracker_tst_ggiw->intensity());
        g_tracker_tst_ggiw->cleanup();
        emit_tracks_tst_ggiw(seq, step.timestep);
    } else {
        send_error(seq, "no tracker; send CONFIG first");
        return;
    }
    const uint32_t dt_us = micros() - t0;
    if (dt_us > 50000 || dropped > 0) {
        char buf[96];
        snprintf(buf, sizeof(buf), "step %lu: %lu us, dropped=%d",
                 (unsigned long)step.timestep, (unsigned long)dt_us, dropped);
        send_log(seq, buf);
    }
}

void dispatch(uint8_t type, uint16_t seq, const uint8_t* payload, uint16_t plen) {
    switch (type) {
        case MSG_PING:
            send_raw(MSG_PONG, seq, nullptr, 0);
            break;
        case MSG_RESET:
            g_tracker_gauss.reset();
            g_tracker_ggiw.reset();
            g_tracker_tst.reset();
            g_tracker_tst_ggiw.reset();
            send_log(seq, "tracker reset");
            break;
        case MSG_CONFIG: {
            rfs_Config cfg = rfs_Config_init_zero;
            pb_istream_t is = pb_istream_from_buffer(payload, plen);
            if (!pb_decode(&is, rfs_Config_fields, &cfg)) {
                send_error(seq, "CONFIG decode failed");
                break;
            }
            rebuild_tracker(cfg);
            const char* tag = g_tracker_ggiw     ? "tracker built (GGIW-PHD)"
                            : g_tracker_tst      ? "tracker built (TST-GM-PHD)"
                            : g_tracker_tst_ggiw ? "tracker built (TST-GGIW-PHD)"
                                                 : "tracker built (GM-PHD)";
            send_log(seq, tag);
            break;
        }
        case MSG_STEP:
            handle_step(seq, payload, plen);
            break;
        default:
            send_error(seq, "unknown msg type");
            break;
    }
}

} // namespace

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);
    send_log(0, "rfs_on_teensy ready");
}

void loop() {
    WireHeader h;
    if (Serial.readBytes(reinterpret_cast<char*>(&h), sizeof(h)) != sizeof(h)) return;
    if (h.len > MAX_PAYLOAD) {
        send_error(h.seq, "payload too large");
        return;
    }
    if (h.len && Serial.readBytes(reinterpret_cast<char*>(payload_buf), h.len) != h.len) return;
    dispatch(h.type, h.seq, payload_buf, h.len);
}
