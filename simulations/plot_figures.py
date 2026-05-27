"""
Generate all four stress-propagation figures.

Run this first; then compile stress_report.tex.

Figures saved to ../stress_visualizations/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from stress_sim import run, run_snapshots, make_positions, spatial_kernel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR    = os.path.join(SCRIPT_DIR, "..", "stress_visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#333333",
    "axes.linewidth":      1.1,
    "axes.grid":           True,
    "grid.color":          "#e8e8e8",
    "grid.linewidth":      0.7,
    "font.family":         "serif",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "legend.fontsize":     9.5,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#bbbbbb",
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "lines.linewidth":     2.2,
})

# ── shared model parameters ───────────────────────────────────────────────
N_BASE = 300          # number of agents
L_BASE = 10.0         # box side  (m)
ELL    = 1.5          # kernel length scale  (m)
MU     = 0.10         # recovery rate  (s⁻¹)
DT     = 0.05         # time step  (s)

# Compute the *actual* mean contact strength from the W matrix.
# The infinite-space formula  C = ρ·2πℓ²  overestimates contacts because
# boundary agents have fewer neighbours.  Using the empirical mean row-sum
# of W ensures that R₀ = κ·C_actual/μ is the same R₀ that the simulation
# actually experiences, so the phase-diagram theory curve and the simulation
# dots are consistent.
_pos_ref = make_positions(N_BASE, L_BASE, seed=1)
_W_ref   = spatial_kernel(_pos_ref, ELL)
C_ACTUAL = float(_W_ref.sum()) / N_BASE   # mean contacts per agent

def kappa_for_R0(R0, C=C_ACTUAL):
    """κ such that R₀ = κ·C/μ in the actual simulation."""
    return R0 * MU / C


# ══════════════════════════════════════════════════════════════════════════
#  FIG 1 — Temporal evolution: three regimes
# ══════════════════════════════════════════════════════════════════════════

def fig1_temporal():
    R0_specs = [
        (0.30, "#3a6ea5", "solid"),
        (0.70, "#56a0d3", "solid"),
        (1.00, "#888888", "dashed"),
        (1.80, "#e08c2a", "solid"),
        (3.00, "#c0392b", "solid"),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    for R0, col, ls in R0_specs:
        kappa = kappa_for_R0(R0)
        t, bm, _, _ = run(N_BASE, L_BASE, MU, kappa, ELL,
                          beta0=0.15, dt=DT, t_end=250.0, seed=1)
        ax.plot(t, bm, color=col, ls=ls,
                label=f"$R_0 = {R0}$")
        if R0 > 1:
            bstar = 1 - 1 / R0
            ax.axhline(bstar, color=col, lw=0.9, ls=":", alpha=0.6)

    ax.set_xlabel("Time  (s)")
    ax.set_ylabel(r"Mean stress  $\bar{\beta}(t)$")
    ax.set_title("Temporal evolution of collective stress — three regimes")
    ax.set_xlim(0, 250)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.legend(loc="center right")

    # annotations
    ax.text(230, 0.02, "subcritical\n$R_0<1$: decay",
            ha="right", fontsize=8.5, color="#3a6ea5", style="italic")
    ax.text(230, 0.70, r"supercritical: $\beta^*=1-1/R_0$",
            ha="right", fontsize=8.5, color="#c0392b", style="italic")
    ax.text(90, 0.115, "$R_0=1$: critical",
            fontsize=8.5, color="#666", style="italic")

    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "fig1_temporal_regimes.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════
#  FIG 2 — Phase diagram: endemic β* vs R₀
# ══════════════════════════════════════════════════════════════════════════

def fig2_phase():
    R0_arr = np.linspace(0.1, 4.0, 50)
    beta_sim = []

    for R0 in R0_arr:
        kappa = kappa_for_R0(R0)
        _, bm, _, _ = run(N_BASE, L_BASE, MU, kappa, ELL,
                          beta0=0.15, dt=DT, t_end=400.0, seed=1)
        beta_sim.append(bm[-1])

    beta_sim    = np.array(beta_sim)
    beta_theory = np.where(R0_arr > 1, 1 - 1 / R0_arr, 0.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.plot(R0_arr, beta_theory, "k--", lw=1.6,
            label=r"Theory: $\beta^* = 1 - 1/R_0$")
    ax.scatter(R0_arr, beta_sim, s=18, color="#c0392b", zorder=4,
               label="Simulation (spatial)")

    ax.axvline(1.0, color="#888", lw=1.0, ls=":")
    ax.fill_between(R0_arr, 0, np.maximum(beta_sim, 0),
                    where=(R0_arr > 1), color="#c0392b", alpha=0.08)

    ax.set_xlabel(r"Reproduction number  $R_0$")
    ax.set_ylabel(r"Endemic stress  $\beta^*$")
    ax.set_title(r"Phase transition at $R_0 = 1$")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")

    ax.text(0.50, 0.88, "stress\nextinguishes",
            transform=ax.transAxes, ha="center",
            fontsize=9, color="#3a6ea5", style="italic")
    ax.text(0.80, 0.40, "collective stress\npersists",
            transform=ax.transAxes, ha="center",
            fontsize=9, color="#c0392b", style="italic")
    ax.annotate("$R_0=1$", xy=(1.0, 0.0), xytext=(1.25, 0.12),
                fontsize=9, color="#555",
                arrowprops=dict(arrowstyle="->", color="#777", lw=1.0))

    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "fig2_phase_diagram.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════
#  FIG 3 — Spatial snapshots (uniform initial condition)
# ══════════════════════════════════════════════════════════════════════════

def fig3_spatial():
    N2, L2 = 400, 12.0
    R0     = 2.5
    # compute C_actual for this specific population
    _p2 = make_positions(N2, L2, seed=7)
    _W2 = spatial_kernel(_p2, ELL)
    C2  = float(_W2.sum()) / N2
    kappa2 = R0 * MU / C2

    snap_times = [0, 25, 80, 250]
    pos, snaps = run_snapshots(N2, L2, MU, kappa2, ELL,
                               beta0=0.05, dt=DT,
                               snap_times=snap_times, seed=7)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    cmap = plt.cm.RdYlGn_r

    for ax, t in zip(axes, snap_times):
        sc = ax.scatter(pos[:, 0], pos[:, 1],
                        c=snaps[t], cmap=cmap, vmin=0, vmax=1,
                        s=14, linewidths=0)
        mean_b = snaps[t].mean()
        ax.set_title(f"$t = {t}$ s\n" + r"$\bar{\beta}$" + f" = {mean_b:.2f}",
                     fontsize=10)
        ax.set_xlim(0, L2); ax.set_ylim(0, L2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f5f5f5")

    cbar = plt.colorbar(sc, ax=axes[-1], label=r"stress $\beta_i$",
                        shrink=0.88, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(r"Spatial stress field  ($R_0 = 2.5$,  uniform $\beta_0 = 0.05$)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "fig3_spatial_snapshots.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════
#  FIG 4 — Wave propagation from a single stressed seed
# ══════════════════════════════════════════════════════════════════════════

def fig4_wave():
    from stress_sim import stress_rhs

    N2, L2 = 500, 15.0
    R0     = 2.5
    rng = np.random.default_rng(42)
    pos = rng.uniform(0, L2, (N2, 2))

    # seed: single agent closest to the centre
    centre   = np.array([L2 / 2, L2 / 2])
    idx_seed = int(np.argmin(np.linalg.norm(pos - centre, axis=1)))
    beta0    = np.zeros(N2)
    beta0[idx_seed] = 1.0

    W      = spatial_kernel(pos, ELL)
    C_w    = float(W.sum()) / N2
    kappa2 = R0 * MU / C_w

    snap_steps = {0: 0, round(15 / DT): 15,
                  round(40 / DT): 40, round(110 / DT): 110}
    max_step = max(snap_steps)
    beta     = beta0.copy()
    snaps    = {}

    for k in range(max_step + 1):
        if k in snap_steps:
            snaps[snap_steps[k]] = beta.copy()
        if k < max_step:
            beta = np.clip(beta + stress_rhs(beta, W, MU, kappa2) * DT, 0.0, 1.0)

    snap_times = [0, 15, 40, 110]
    fig, axes  = plt.subplots(1, 4, figsize=(13, 3.6))
    cmap = plt.cm.RdYlGn_r

    for ax, t in zip(axes, snap_times):
        sc = ax.scatter(pos[:, 0], pos[:, 1],
                        c=snaps[t], cmap=cmap, vmin=0, vmax=1,
                        s=10, linewidths=0)
        ax.plot(*centre, "k*", ms=11, zorder=6, label="seed" if t == 0 else "")
        mean_b = snaps[t].mean()
        ax.set_title(f"$t = {t}$ s\n" + r"$\bar{\beta}$" + f" = {mean_b:.2f}",
                     fontsize=10)
        ax.set_xlim(0, L2); ax.set_ylim(0, L2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f5f5f5")

    cbar = plt.colorbar(sc, ax=axes[-1], label=r"stress $\beta_i$",
                        shrink=0.88, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(r"Stress wave from a single seed  ($\bigstar$)  —  $R_0 = 2.5$",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    out = os.path.join(VIZ_DIR, "fig4_wave_propagation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating stress-propagation figures …")
    fig1_temporal()
    fig2_phase()
    fig3_spatial()
    fig4_wave()
    print("Done.")
