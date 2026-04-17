"""Animated sim harness with Chebyshev-polynomial truth trajectories.

Each target's x(t) and y(t) are independent random Chebyshev series with
coefficients that decay as 1/(k+1), producing smooth but non-trivial paths
the constant-velocity EKF has to chase recursively. Great for stress-testing
the tracker with motion it can't linearly extrapolate.

Usage:
    python sim_runner_chebyshev.py --preview-truth
    python sim_runner_chebyshev.py --port COM6 --degree 8 --targets 4
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy.polynomial import chebyshev as cheb

from link import Link, pb, PING, PONG, RESET, CONFIG, STEP, TRACKS


# ---- Scenario + detection generation ----

def make_scenario(n_targets, n_steps, dt, fov, rng, degree=6):
    """Each target: x(t), y(t) are random Chebyshev polynomials on t in [-1, 1].

    Coefficient decay 1/(k+1) keeps low-order (long-wavelength) terms dominant.
    Per-axis normalization uses the L1 norm of the coefficients, which is a
    hard upper bound on |x(t)| for Chebyshev polynomials (since |T_k(t)| <= 1
    on [-1, 1]). That bound is *invariant* to how coefficients interfere, so
    max speed is consistent run-to-run (unlike a max|x(t)| post-hoc rescale,
    which could produce huge scale factors on unlucky draws).
    """
    truth = np.zeros((n_steps, n_targets, 4))
    t_norm = np.linspace(-1.0, 1.0, n_steps)
    T = max((n_steps - 1) * dt, 1e-9)
    amp_target = fov / 3.0
    decay = 1.0 / (np.arange(degree + 1) + 1.0)
    for i in range(n_targets):
        cx = rng.normal(size=degree + 1) * decay
        cy = rng.normal(size=degree + 1) * decay

        # L1 normalization: guarantees |x(t)| <= amp_target and |y(t)| <= amp_target
        # without depending on where the peak happens to land.
        cx *= amp_target / (np.sum(np.abs(cx)) + 1e-9)
        cy *= amp_target / (np.sum(np.abs(cy)) + 1e-9)

        x = cheb.chebval(t_norm, cx)
        y = cheb.chebval(t_norm, cy)

        # Analytic velocity via Chebyshev derivative. Chain rule: t_norm = -1 + 2t/T
        # so dt_norm/dt = 2/T and vx(t) = dx/dt_norm * 2/T.
        dcx = cheb.chebder(cx)
        dcy = cheb.chebder(cy)
        vx = cheb.chebval(t_norm, dcx) * (2.0 / T)
        vy = cheb.chebval(t_norm, dcy) * (2.0 / T)

        truth[:, i, 0] = x
        truth[:, i, 1] = y
        truth[:, i, 2] = vx
        truth[:, i, 3] = vy
    return truth


def detections_point(truth_step, pd, clutter_lambda, meas_noise_std, fov, rng):
    out = []
    for i in range(truth_step.shape[0]):
        if rng.uniform() < pd:
            z = truth_step[i, 0:2] + rng.normal(0.0, meas_noise_std, size=2)
            out.append([float(z[0]), float(z[1])])
    for _ in range(rng.poisson(clutter_lambda)):
        z = rng.uniform(-fov / 2, fov / 2, size=2)
        out.append([float(z[0]), float(z[1])])
    return out


def detections_extended(truth_step, pd, meas_rate, clutter_lambda, extent_std,
                        meas_noise_std, fov, rng):
    out = []
    sigma = float(np.hypot(extent_std, meas_noise_std))
    for i in range(truth_step.shape[0]):
        if rng.uniform() >= pd:
            continue
        for _ in range(max(1, int(rng.poisson(meas_rate)))):
            z = truth_step[i, 0:2] + rng.normal(0.0, sigma, size=2)
            out.append([float(z[0]), float(z[1])])
    for _ in range(rng.poisson(clutter_lambda)):
        z = rng.uniform(-fov / 2, fov / 2, size=2)
        out.append([float(z[0]), float(z[1])])
    return out


def build_config(args):
    kind = {"gm": pb.FILTER_GM_PHD, "ggiw": pb.FILTER_GGIW_PHD,
            "tst": pb.FILTER_TST_GM_PHD}[args.filter]
    cfg = pb.Config(
        filter_kind=kind, state_dim=4, meas_dim=2,
        dt=args.dt, prob_detection=args.pd, prob_survive=0.99,
        clutter_rate=max(args.clutter, 1e-6),
        clutter_density=1.0 / (args.fov * args.fov),
        prune_threshold=1e-3, merge_threshold=4.0,
        extract_threshold=0.4, gate_threshold=16.0,
        max_components=args.max_components, birth_weight=0.03,
        eta=1.0, tau=15.0, rho=1.0,
        birth_alpha=args.meas_rate, birth_beta=1.0, birth_v=20.0,
        window_size=args.window,
        max_history=args.max_history,
        trajectory_history=args.traj_history,
    )
    cfg.birth_mean.extend([0.0, 0.0, 0.0, 0.0])
    cfg.birth_cov_diag.extend([(args.fov / 2)**2.0, (args.fov / 2)**2.0, 1.0, 1.0])
    cfg.process_noise_diag.extend([args.q, args.q])
    cfg.measurement_noise_diag.extend([args.r, args.r])
    cfg.birth_extent.extend([args.extent_std ** 2, args.extent_std ** 2, 0.0])
    return cfg

def run(args):
    rng = np.random.default_rng(args.seed)
    truth = make_scenario(args.targets, args.steps, args.dt, args.fov, rng,
                          degree=args.degree)
    cfg = build_config(args)
    is_ggiw = (args.filter == "ggiw")

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-args.fov / 2, args.fov / 2)
    ax.set_ylim(-args.fov / 2, args.fov / 2)
    ax.set_aspect("equal")

    stop = {"flag": False}
    def on_key(event):
        if event.key == args.stop_key:
            stop["flag"] = True
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(f"[host] press '{args.stop_key}' in the figure window to stop early")

    for i in range(truth.shape[1]):
        ax.plot(truth[:, i, 0], truth[:, i, 1], "k-", alpha=0.2, lw=1)
    truth_now = ax.scatter(truth[0, :, 0], truth[0, :, 1], c="k", marker="x",
                           s=40, label="truth (current)")
    det_scat = ax.scatter([], [], c="b", s=6, alpha=0.6, label="detections (step)")
    track_scat = ax.scatter([], [], c="r", s=30, label="tracks (step)")
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize="small")

    track_lines: dict[int, tuple] = {}
    extent_patches: list[Ellipse] = []
    traj_lines: list = []   # TST per-step trajectory polylines (cleared each step)
    cmap = plt.get_cmap("tab20")
    is_tst = (args.filter == "tst")
    state_dim = 4

    link = Link(args.port, baud=args.baud)
    try:
        link.send(PING); link.recv(link.seq, PONG)
        print("[host] ping/pong ok")
        link.send(RESET)
        link.send(CONFIG, cfg)

        for k in range(args.steps):
            if not plt.fignum_exists(fig.number):
                print("[host] figure closed, stopping early")
                break
            if stop["flag"]:
                print(f"[host] '{args.stop_key}' pressed, stopping early")
                break

            dets = (detections_extended(truth[k], args.pd, args.meas_rate, args.clutter,
                                        args.extent_std, args.meas_std, args.fov, rng)
                    if is_ggiw else
                    detections_point(truth[k], args.pd, args.clutter,
                                     args.meas_std, args.fov, rng))

            step = pb.Step(timestep=k)
            for d in dets:
                step.meas.add().coords.extend(d)
            seq = link.send(STEP, step)
            tracks_msg = pb.Tracks(); tracks_msg.ParseFromString(link.recv(seq, TRACKS))
            tracks = list(tracks_msg.tracks)

            truth_now.set_offsets(truth[k, :, :2])
            det_scat.set_offsets(np.array(dets) if dets else np.empty((0, 2)))

            # For TST we draw each track as a polyline (see below); suppress
            # the per-step red dot scatter to avoid hiding the polyline ends.
            if is_tst:
                track_scat.set_offsets(np.empty((0, 2)))
            else:
                now_pts = np.array([[t.mean[0], t.mean[1]]
                                    for t in tracks if len(t.mean) >= 2])
                track_scat.set_offsets(now_pts if len(now_pts) else np.empty((0, 2)))

            for t in tracks:
                tid = int(getattr(t, "id", 0))
                if tid <= 0 or len(t.mean) < 2:
                    continue
                if tid not in track_lines:
                    color = cmap(len(track_lines) % cmap.N)
                    ln, = ax.plot([], [], "-", color=color, lw=1.5, alpha=0.85)
                    track_lines[tid] = (ln, [], [])
                ln, xs, ys = track_lines[tid]
                xs.append(float(t.mean[0])); ys.append(float(t.mean[1]))
                ln.set_data(xs, ys)

            # TST: draw each extracted trajectory as history+current polyline.
            # NOT cleared between steps so each step's short window-polyline
            # overlaps with the previous ones, accumulating into the full
            # trajectory since birth. The wire only carries the window; the
            # full history lives in this accumulated drawing.
            if is_tst:
                for t in tracks:
                    if len(t.mean) < 2:
                        continue
                    hist = np.array(list(t.history), dtype=float)
                    pts_x, pts_y = [], []
                    if hist.size and hist.size % state_dim == 0:
                        hist = hist.reshape(-1, state_dim)
                        pts_x.extend(hist[:, 0].tolist())
                        pts_y.extend(hist[:, 1].tolist())
                    pts_x.append(float(t.mean[0]))
                    pts_y.append(float(t.mean[1]))
                    if len(pts_x) >= 2:
                        ax.plot(pts_x, pts_y, "-",
                                color="#ff0000", lw=1.2,
                                alpha=0.35, zorder=10)

            for p in extent_patches:
                p.remove()
            extent_patches.clear()
            if is_ggiw:
                for t in tracks:
                    if t.v > 4.0 and len(t.extent) >= 3 and len(t.mean) >= 2:
                        denom = max(t.v - 6.0, 1e-3)
                        Vxx, Vyy, Vxy = t.extent
                        X = np.array([[Vxx / denom, Vxy / denom],
                                      [Vxy / denom, Vyy / denom]])
                        w, vec = np.linalg.eigh(X)
                        w = np.clip(w, 1e-6, None)
                        angle = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
                        ell = Ellipse((t.mean[0], t.mean[1]),
                                      2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]),
                                      angle=angle, fill=False,
                                      edgecolor="r", alpha=0.35)
                        ax.add_patch(ell)
                        extent_patches.append(ell)

            title.set_text(f"step {k:4d}  dets={len(dets)}  tracks={len(tracks)}")
            fig.canvas.draw_idle()
            plt.pause(args.frame_delay)
    finally:
        link.close()

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.show()


def main():
    p = argparse.ArgumentParser(description="rfs_on_teensy Chebyshev sim harness")
    p.add_argument("--port", help="Serial port (required unless --preview-truth)") 
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--targets", type=int, default=3)
    p.add_argument("--clutter", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--pd", type=float, default=0.95)
    p.add_argument("--q", type=float, default=1,
                   help="Process noise std^2 per axis (bump vs circles: paths are wilder)")
    p.add_argument("--r", type=float, default=0.5,
                   help="Filter's measurement noise variance per axis "
                        "(passed into Config.measurement_noise_diag). Bigger = "
                        "filter trusts predictions more = smoother tracks. "
                        "Does NOT control scenario noise (see --meas-std).")
    p.add_argument("--meas-std", type=float, default=0.1,
                   help="Scenario noise std used when sampling detections "
                        "around truth. Independent of --r.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--filter", choices=["gm", "ggiw", "tst"], default="gm")
    p.add_argument("--meas-rate", type=float, default=8.0)
    p.add_argument("--extent-std", type=float, default=2.0)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--max-history", type=int, default=10,
                   help="Cap on PHD/GLMB extracted-mixtures deque. 0 = no "
                        "recording (zero-growth), N = keep N latest.")
    p.add_argument("--traj-history", type=int, default=10,
                   help="TST-only: per-Trajectory state_history cap "
                        "(independent of --window). 0 disables, N keeps N "
                        "most-recent states per trajectory.")
    p.add_argument("--max_components", type=int, default=10)
    p.add_argument("--degree", type=int, default=2,
                   help="Chebyshev polynomial degree per axis per target")
    p.add_argument("--frame-delay", type=float, default=0.05)
    p.add_argument("--stop-key", default="q")
    args = p.parse_args() 
    if not args.port:
        p.error("--port is required unless --preview-truth is set")
    run(args)


if __name__ == "__main__":
    main()
