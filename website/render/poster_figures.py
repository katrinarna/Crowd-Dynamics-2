"""
Polished poster figures for the STRESS website. Imports Filip's sfm_smooth.py
(unchanged) and runs the same sweep that evac_time_plot.py uses, then renders:

  Figure 1: Mean evac time vs desired speed v0
            Title: "Faster-Is-Slower: Optimal Desired Speed for Crowd Evacuation"
            Filled optimal-v0* marker.

  Figure 3: Mean evac time vs initial collective stress Sigma_0
            Using poster's linear link v0 = V_min + Sigma * (V_max - V_min)
            with V_min = 1.0, V_max = 5.0. Same data, x-axis transformed.

Outputs:
  evac_time_plot.png         (Figure 1)
  evac_time_vs_sigma.png     (Figure 3)
  sweep_data.csv             (raw data, for reproducibility)
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from sfm_smooth import (
    step, check_exit, init_agents,
    ROOM_W, ROOM_H, EXIT_W, DT,
)

# Match evac_time_plot.py exactly
N     = 60
T_END = 160.0
SEEDS = 5
V0S   = np.round(np.arange(1.0, 5.01, 0.2), 2)
EY_LO = ROOM_H / 2 - EXIT_W / 2
EY_HI = ROOM_H / 2 + EXIT_W / 2

# Poster linear link (Figure 3): V_MIN = 1.0, V_MAX = 5.0 -> Sigma in [0, 1]
V_MIN_POSTER = 1.0
V_MAX_POSTER = 5.0


def run_one(v0, seed):
    pos, vel, radii = init_agents(N, ROOM_W, ROOM_H, seed)
    t = 0.0
    last_exit = 0.0
    while t < T_END and len(pos) > 0:
        pos, vel = step(pos, vel, radii, v0, ROOM_W, ROOM_H, EY_LO, EY_HI, DT)
        t += DT
        ex = check_exit(pos)
        if ex.any():
            last_exit = t
            pos = pos[~ex]; vel = vel[~ex]; radii = radii[~ex]
    return last_exit if len(pos) == 0 else T_END


# ── run sweep ───────────────────────────────────────────────────────────────
print(f"Simulating {len(V0S)} speeds x {SEEDS} seeds (N={N}) ...", flush=True)
data = {}
for v0 in V0S:
    times = [run_one(v0, s) for s in range(SEEDS)]
    data[v0] = times
    print(f"  v0={v0:.1f}  mean={np.mean(times):.1f}s", flush=True)

means = np.array([np.mean(data[v]) for v in V0S])
stds  = np.array([np.std(data[v], ddof=1) for v in V0S])
ci95  = 1.96 * stds / np.sqrt(SEEDS)

# save CSV
csv_path = os.path.join(HERE, "sweep_data.csv")
with open(csv_path, "w") as f:
    f.write("v0,mean_evac,ci95\n")
    for v0, m, c in zip(V0S, means, ci95):
        f.write(f"{v0:.2f},{m:.3f},{c:.3f}\n")
print(f"  -> wrote {csv_path}", flush=True)

# locate optimum
i_opt = int(np.argmin(means))
v0_opt = float(V0S[i_opt])
t_opt = float(means[i_opt])
print(f"  optimum: v0* = {v0_opt:.2f} m/s, t* = {t_opt:.1f} s", flush=True)


# ── shared style ───────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "figure.dpi":        130,
})


def render_evac_plot(xs, xlabel, title, optimal_x_label, out_path, shaded_zone=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    # mean line + 95% CI
    ax.errorbar(
        xs, means, yerr=ci95,
        fmt="o-",
        color="#C0392B",
        markerfacecolor="white",
        markeredgecolor="#C0392B",
        markeredgewidth=1.4,
        markersize=6.5,
        linewidth=1.6,
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3,
        label="Mean evacuation time (95% CI)",
    )
    # optimal filled marker
    ax.plot(
        xs[i_opt], means[i_opt],
        "o",
        color="#7F1F12",
        markersize=10.5,
        markeredgecolor="#7F1F12",
        zorder=5,
        label=optimal_x_label,
    )
    # optional clogging-zone shading
    if shaded_zone is not None:
        ax.axvspan(*shaded_zone, color="#C0392B", alpha=0.08, zorder=1,
                   label="Arch-dominated clogging")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Evacuation time  (s)")
    ax.set_title(title)
    ax.set_xlim(xs[0] - (xs[1] - xs[0]), xs[-1] + (xs[1] - xs[0]))
    ax.set_ylim(0, T_END + 20)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", color="black", width=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}", flush=True)


# ── Figure 1: vs v0 ────────────────────────────────────────────────────────
fig1_path = os.path.join(HERE, "..", "public", "figures", "evac_time_plot.png")
fig1_path = os.path.abspath(fig1_path)
render_evac_plot(
    xs=V0S,
    xlabel=r"Desired speed  $v_0$  (m s$^{-1}$)",
    title="Faster-Is-Slower: Optimal Desired Speed for Crowd Evacuation",
    optimal_x_label=fr"Optimal $v_0^* = {v0_opt:.1f}\ \mathrm{{m\,s^{{-1}}}}$",
    out_path=fig1_path,
    shaded_zone=None,
)

# ── Figure 3: vs Sigma_0 (poster linear link V_min=1.0, V_max=5.0) ─────────
sigmas = (V0S - V_MIN_POSTER) / (V_MAX_POSTER - V_MIN_POSTER)
sigma_opt = float(sigmas[i_opt])

# Clogging zone (right of optimum, where evac time noticeably rises)
clog_start_idx = i_opt + 1
# walk forward until evac time is ~50% above the optimum, that's the start of the clogging zone
for k in range(i_opt, len(means)):
    if means[k] > 1.5 * t_opt:
        clog_start_idx = k
        break
clog_zone = (float(sigmas[clog_start_idx]), float(sigmas[-1] + 0.05))

fig3_path = os.path.join(HERE, "..", "public", "figures", "evac_time_vs_sigma.png")
fig3_path = os.path.abspath(fig3_path)
render_evac_plot(
    xs=sigmas,
    xlabel=r"Initial collective stress  $\Sigma_0$",
    title="Evacuation Time vs Initial Collective Stress",
    optimal_x_label=(fr"Optimal $\Sigma_0^* = {sigma_opt:.2f}$ "
                     fr"($v_0 \approx {v0_opt:.1f}\ \mathrm{{m\,s^{{-1}}}}$)"),
    out_path=fig3_path,
    shaded_zone=clog_zone,
)

print("\nDone.")
