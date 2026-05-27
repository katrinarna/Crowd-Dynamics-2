"""
Evacuation time vs. desired speed — ggplot-style mean ± 95% CI plot.
v0 from 1.0 to 5.0 in 0.2 steps (21 speeds × 5 seeds = 105 runs).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_smooth import (
    step, check_exit, init_agents,
    ROOM_W, ROOM_H, EXIT_W, EXIT_Y_LO, EXIT_Y_HI, DT,
)

# ── simulation parameters ────────────────────────────────────────────────────
N     = 60
T_END = 160.0
SEEDS = 5
V0S   = np.round(np.arange(1.0, 5.01, 0.2), 2)   # 21 speeds
EY_LO = ROOM_H / 2 - EXIT_W / 2
EY_HI = ROOM_H / 2 + EXIT_W / 2

# ── run ──────────────────────────────────────────────────────────────────────
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

print(f"Simulating {len(V0S)} speeds × {SEEDS} seeds (N={N})…")
data = {}
for v0 in V0S:
    times = [run_one(v0, s) for s in range(SEEDS)]
    data[v0] = times
    print(f"  v0={v0:.1f}  mean={np.mean(times):.0f}s  [{min(times):.0f}–{max(times):.0f}]",
          flush=True)

# ── stats ─────────────────────────────────────────────────────────────────────
means = np.array([np.mean(data[v]) for v in V0S])
stds  = np.array([np.std(data[v], ddof=1) for v in V0S])
ci95  = 1.96 * stds / np.sqrt(SEEDS)

# ── ggplot-style rcParams ────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
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

fig, ax = plt.subplots(figsize=(8, 5))

# ── error bars + line + open circles ─────────────────────────────────────────
ax.errorbar(
    V0S, means, yerr=ci95,
    fmt="o",
    color="#C0392B",
    markerfacecolor="white",
    markeredgecolor="#C0392B",
    markeredgewidth=1.2,
    markersize=6,
    linewidth=1.2,
    elinewidth=1.1,
    capsize=4,
    capthick=1.1,
    zorder=3,
)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Desired speed  $v_0$  (m s⁻¹)")
ax.set_ylabel("Evacuation time  (s)")
ax.set_title(
    f"Mean evacuation time depending on desired speed.\n"
    f"Error bars represent 95% Confidence Intervals",
    loc="left",
)

ax.set_xlim(V0S[0] - 0.2, V0S[-1] + 0.2)
ax.set_ylim(0, T_END + 15)

ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")
ax.tick_params(axis="both", color="black", width=0.8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "evac_time_plot.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {out}")
plt.show()
