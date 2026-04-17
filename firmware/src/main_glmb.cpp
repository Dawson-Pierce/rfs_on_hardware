// GLMB (Generalized Labeled Multi-Bernoulli) sketch for Teensy 4.1.
// Same wire protocol as main_phd.cpp — host sends CONFIG + STEP, we reply
// with TRACKS. For now this sketch only supports filter_kind=FILTER_GM_PHD
// in the Config (re-interpreted as "GLMB with Gaussian components"); other
// filter_kinds are rejected.
//
// Config field mapping for GLMB (first pass):
//   prune_threshold    -> prune_threshold_bernoulli (and hypothesis threshold)
//   merge_threshold    -> unused (GLMB doesn't moment-match mixtures)
//   max_components     -> max_hypotheses
//   extract_threshold  -> Bernoulli existence-prob cutoff for extraction
//   gate_threshold     -> per-measurement gate

#include <Arduino.h>

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

#include "brew/advanced/multi_target/glmb.hpp"
#include "brew/core/filters/ekf.hpp"
#include "brew/core/dynamics/single_integrator.hpp"
#include "brew/core/models/gaussian.hpp"
#include "brew/core/models/mixture.hpp"

#include <pb.h>
#include <pb_encode.h>
#include <pb_decode.h>
#include "rfs.pb.h"

#include <memory>

enum : uint8_t {
    MSG_PING   = 1, MSG_PONG   = 2, MSG_RESET  = 3, MSG_CONFIG = 4,
    MSG_STEP   = 5, MSG_TRACKS = 6, MSG_LOG    = 7, MSG_ERROR  = 0x7F
};

constexpr size_t MAX_PAYLOAD = 8192;

#pragma pack(push, 1)
struct WireHeader { uint8_t type; uint16_t seq; uint16_t len; };
#pragma pack(pop)
static_assert(sizeof(WireHeader) == 5, "");

using GaussD    = brew::models::Gaussian<>;
using MixGauss  = brew::models::Mixture<GaussD>;
using GLMBGauss = brew::multi_target::GLMB<GaussD>;

namespace {

std::unique_ptr<GLMBGauss> g_tracker;
rfs_Config                 g_cfg = rfs_Config_init_zero;

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

// ---------- Tracker rebuild ----------

void rebuild_tracker(const rfs_Config& cfg) {
    g_cfg = cfg;
    const int sd = cfg.state_dim;
    const int md = cfg.meas_dim;
    const int spatial = sd / 2;

    auto ekf = std::make_unique<brew::filters::EKF<>>();
    auto dyn = std::make_shared<brew::dynamics::SingleIntegrator<>>(spatial);
    ekf->set_dynamics(dyn);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(md, sd);
    for (int i = 0; i < md; ++i) H(i, i) = 1.0;
    ekf->set_measurement_jacobian(H);

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(spatial, spatial);
    for (int i = 0; i < spatial && i < (int)cfg.process_noise_diag_count; ++i)
        Q(i, i) = cfg.process_noise_diag[i];
    ekf->set_process_noise(Q);

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(md, md);
    for (int i = 0; i < md && i < (int)cfg.measurement_noise_diag_count; ++i)
        R(i, i) = cfg.measurement_noise_diag[i];
    ekf->set_measurement_noise(R);

    auto birth = std::make_unique<MixGauss>();
    Eigen::VectorXd bm(sd);
    for (int i = 0; i < sd; ++i)
        bm(i) = (i < (int)cfg.birth_mean_count) ? cfg.birth_mean[i] : 0.0;
    Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(sd, sd);
    for (int i = 0; i < sd; ++i)
        bc(i, i) = (i < (int)cfg.birth_cov_diag_count) ? cfg.birth_cov_diag[i] : 1.0;
    birth->add_component(std::make_unique<GaussD>(bm, bc), cfg.birth_weight);

    auto glmb = std::make_unique<GLMBGauss>();
    glmb->set_filter(std::move(ekf));
    // Second arg is the per-step birth probability for this term — we reuse
    // Config.birth_weight for that.
    glmb->set_birth_model(std::move(birth), cfg.birth_weight);
    glmb->set_prob_detection(cfg.prob_detection);
    glmb->set_prob_survive(cfg.prob_survive);
    glmb->set_clutter_rate(cfg.clutter_rate);
    glmb->set_clutter_density(cfg.clutter_density);
    glmb->set_prune_threshold(cfg.prune_threshold);
    glmb->set_max_hypotheses(cfg.max_components);
    glmb->set_extract_threshold(cfg.extract_threshold);
    glmb->set_gate_threshold(cfg.gate_threshold);
    glmb->set_max_history(cfg.max_history);
    glmb->set_k_best(5);

    g_tracker = std::move(glmb);
}

// ---------- Step ----------
// (No NaN purge yet — GLMB state is Bernoullis + hypotheses, not a single
// intensity mixture. Revisit if numerical faults show up in testing.)

Eigen::MatrixXd load_meas(const rfs_Step& step, int md) {
    Eigen::MatrixXd Z(md, step.meas_count);
    for (int j = 0; j < (int)step.meas_count; ++j) {
        const auto& d = step.meas[j];
        for (int i = 0; i < md; ++i)
            Z(i, j) = (i < (int)d.coords_count) ? (double)d.coords[i] : 0.0;
    }
    return Z;
}

void emit_tracks(uint16_t seq, uint32_t timestep) {
    // Use extracted_tracks() (labeled, history-aware) rather than the unlabeled
    // extracted_mixtures() snapshot — lets us emit a stable per-track id so the
    // host can stitch trajectories together across steps.
    const auto& tracks = g_tracker->extracted_tracks();
    rfs_Tracks msg = rfs_Tracks_init_zero;
    msg.timestep = timestep;
    const int sd = g_cfg.state_dim;
    const int cap = (int)(sizeof(msg.tracks) / sizeof(msg.tracks[0]));
    const int n = std::min<int>((int)tracks.size(), cap);
    msg.tracks_count = n;
    for (int i = 0; i < n; ++i) {
        msg.tracks[i] = rfs_Track_init_zero;
        const auto& trk = *tracks[i];
        if (trk.states.empty()) continue;
        // Same hash brew uses internally in track_histories().
        msg.tracks[i].id = (uint32_t)(trk.label.first * 100000 + trk.label.second);
        msg.tracks[i].weight = 1.0f;
        const auto& g = *trk.states.back();
        const int kdim = std::min(sd, 8);
        msg.tracks[i].mean_count = kdim;
        msg.tracks[i].cov_diag_count = kdim;
        for (int k = 0; k < kdim; ++k) {
            msg.tracks[i].mean[k] = (float)g.mean()(k);
            msg.tracks[i].cov_diag[k] = (float)g.covariance()(k, k);
        }
    }
    send_pb(MSG_TRACKS, seq, rfs_Tracks_fields, msg);
}

void handle_step(uint16_t seq, const uint8_t* payload, uint16_t plen) {
    if (!g_tracker) { send_error(seq, "no tracker; send CONFIG first"); return; }
    rfs_Step step = rfs_Step_init_zero;
    pb_istream_t is = pb_istream_from_buffer(payload, plen);
    if (!pb_decode(&is, rfs_Step_fields, &step)) {
        send_error(seq, "STEP decode failed");
        return;
    }
    const int md = g_cfg.meas_dim;
    const uint32_t t0 = micros();
    Eigen::MatrixXd Z = load_meas(step, md);
    g_tracker->predict((int)step.timestep, (double)g_cfg.dt);
    if (step.meas_count > 0) g_tracker->correct(Z);
    g_tracker->cleanup();
    emit_tracks(seq, step.timestep);

    const uint32_t dt_us = micros() - t0;
    if (dt_us > 50000) {
        char buf[96];
        snprintf(buf, sizeof(buf), "step %lu: %lu us", (unsigned long)step.timestep,
                 (unsigned long)dt_us);
        send_log(seq, buf);
    }
}

void dispatch(uint8_t type, uint16_t seq, const uint8_t* payload, uint16_t plen) {
    switch (type) {
        case MSG_PING:  send_raw(MSG_PONG, seq, nullptr, 0); break;
        case MSG_RESET: g_tracker.reset(); send_log(seq, "tracker reset"); break;
        case MSG_CONFIG: {
            rfs_Config cfg = rfs_Config_init_zero;
            pb_istream_t is = pb_istream_from_buffer(payload, plen);
            if (!pb_decode(&is, rfs_Config_fields, &cfg)) {
                send_error(seq, "CONFIG decode failed"); break;
            }
            if (cfg.filter_kind != rfs_FilterKind_FILTER_GM_PHD) {
                send_error(seq, "GLMB sketch: only GM (Gaussian) filter_kind supported");
                break;
            }
            rebuild_tracker(cfg);
            send_log(seq, "tracker built (GLMB-Gaussian)");
            break;
        }
        case MSG_STEP: handle_step(seq, payload, plen); break;
        default: send_error(seq, "unknown msg type"); break;
    }
}

} // namespace

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);
    send_log(0, "rfs_on_teensy GLMB ready");
}

void loop() {
    static uint8_t in[MAX_PAYLOAD];
    WireHeader h;
    if (Serial.readBytes(reinterpret_cast<char*>(&h), sizeof(h)) != sizeof(h)) return;
    if (h.len > MAX_PAYLOAD) { send_error(h.seq, "payload too large"); return; }
    if (h.len && Serial.readBytes(reinterpret_cast<char*>(in), h.len) != h.len) return;
    dispatch(h.type, h.seq, in, h.len);
}
