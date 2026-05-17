"""
evacuation_vs_stress.py
=======================
Sweep contagion strength κ, run a full SSFM + stress evacuation for each value,
record the evacuation time T(κ).

Story:
  κ ↑  →  β̄ ↑  →  v₀(β) ↑  →  α = Aτ/(mv₀) ↓  →  clogging  →  T ↑
                                                           ↑
                                               Faster-is-Slower regime

Three-panel figure:
  Left   – T(κ)  simulation ± std, κ_c marked, optimal κ* starred
  Centre – mean-field β̄∞(κ) and desired speed v₀(κ) showing the chain
  Right  – α(κ) = Aτ/(mv₀(β̄∞)) showing when drive overwhelms repulsion
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Parameters ───────────────────────────────────────────────────────────────
MASS  = 80.0;   TAU = 0.5;   R = 0.25
A     = 2000.0; B   = 0.08
AW    = 2000.0; BW  = 0.08
V_MIN = 1.0;    V_MAX = 4.0
MU    = 0.3;    ELL   = 3.0

DT    = 0.04    # time step  (s)  — smaller for contact-force stability
T_MAX = 120.0   # max sim time per run (s)

ROOM_X = (0.0, 10.0)
ROOM_Y = (0.0,  8.0)
EXIT_Y = (3.6,  4.4)    # 0.8 m wide exit — narrow enough for FIS with contact forces
TARGET = np.array([ROOM_X[1] + 0.5, (EXIT_Y[0] + EXIT_Y[1]) / 2.0])

N_AGENTS   = 60
SEED_FRAC  = 0.10        # fraction of agents seeded as panickers at t=0
BETA_SEED  = 0.80        # initial stress of seed panickers

N_KAPPA    = 30          # κ sweep resolution
N_SIM_SEEDS = 8          # independent runs per κ (for averaging)

K_BODY  = 120000.0       # body compression stiffness (N/m)  — full SFM contact
K_FRIC  = 240000.0       # sliding friction coefficient (N·s/m)


# ── Vectorised force functions ────────────────────────────────────────────────
def self_driving(pos, vel, betas):
    diff  = TARGET - pos                                    # (N, 2)
    dists = np.linalg.norm(diff, axis=1, keepdims=True)
    dirs  = diff / np.maximum(dists, 1e-9)
    v0    = (V_MIN + betas * (V_MAX - V_MIN))[:, None]     # (N, 1)
    return MASS * (v0 * dirs - vel) / TAU                  # (N, 2)


def repulsion(pos):
    """Fully vectorised O(N²) pairwise exponential repulsion."""
    d_vec = pos[:, None, :] - pos[None, :, :]              # (N, N, 2)
    d     = np.linalg.norm(d_vec, axis=-1)                 # (N, N)
    np.fill_diagonal(d, np.inf)
    n_hat = d_vec / d[:, :, None]                          # (N, N, 2)  — diag=0
    mag   = A * np.exp((2 * R - d) / B)                    # (N, N)  — diag=0
    return (mag[:, :, None] * n_hat).sum(axis=1)           # (N, 2)


def wall_forces(pos):
    N = len(pos)
    F = np.zeros((N, 2))
    # left wall  (+x)
    F[:, 0] += AW * np.exp((R - np.maximum(pos[:, 0] - ROOM_X[0], 1e-9)) / BW)
    # bottom wall (+y)
    F[:, 1] += AW * np.exp((R - np.maximum(pos[:, 1] - ROOM_Y[0], 1e-9)) / BW)
    # top wall   (-y)
    F[:, 1] -= AW * np.exp((R - np.maximum(ROOM_Y[1] - pos[:, 1], 1e-9)) / BW)
    # right wall (solid only outside exit gap)
    solid = (pos[:, 1] < EXIT_Y[0]) | (pos[:, 1] > EXIT_Y[1])
    d_right = np.maximum(ROOM_X[1] - pos[solid, 0], 1e-9)
    F[solid, 0] -= AW * np.exp((R - d_right) / BW)
    return F


def stress_deriv(beta, pos, kappa):
    """β̇ᵢ = −µβᵢ + κ Σⱼ w_{ij} βⱼ (1 − βᵢ),  w_{ij} = exp(−d_{ij}/ℓ)."""
    d_vec = pos[:, None, :] - pos[None, :, :]
    d     = np.linalg.norm(d_vec, axis=-1)
    np.fill_diagonal(d, np.inf)
    W     = np.exp(-d / ELL)                               # (N, N), diag=0
    return -MU * beta + kappa * (W @ beta) * (1.0 - beta)


# ── Initial grid placement ────────────────────────────────────────────────────
def place_grid(N, rng, jitter=0.12):
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    xs   = np.linspace(ROOM_X[0] + 1.0, ROOM_X[1] - 3.5, cols)
    ys   = np.linspace(ROOM_Y[0] + 1.0, ROOM_Y[1] - 1.0, rows)
    gx, gy = np.meshgrid(xs, ys)
    pos  = np.column_stack([gx.ravel(), gy.ravel()])[:N].copy()
    pos += rng.uniform(-jitter, jitter, pos.shape)
    return pos


# ── Estimate κ_c from a typical initial configuration ────────────────────────
_rng0 = np.random.default_rng(0)
_pos0 = place_grid(N_AGENTS, _rng0)
_d0   = np.linalg.norm(_pos0[:, None, :] - _pos0[None, :, :], axis=-1)
np.fill_diagonal(_d0, np.inf)
_W0   = np.exp(-_d0 / ELL)
W_ROW_MEAN = _W0.sum(axis=1).mean()     # Λ = κ × W_ROW_MEAN  →  κ_c = µ/W_ROW_MEAN
KAPPA_C    = MU / W_ROW_MEAN
print(f"κ_c (mean-field estimate) = {KAPPA_C:.4f}")


# ── Single evacuation run ─────────────────────────────────────────────────────
def run_evacuation(kappa, sim_seed):
    rng  = np.random.default_rng(sim_seed)
    pos  = place_grid(N_AGENTS, rng)
    vel  = np.zeros((N_AGENTS, 2))
    beta = np.zeros(N_AGENTS)

    n_panic          = max(1, int(SEED_FRAC * N_AGENTS))
    panic_idx        = rng.choice(N_AGENTS, n_panic, replace=False)
    beta[panic_idx]  = BETA_SEED

    t = 0.0
    while t < T_MAX and len(pos) > 0:
        F    = self_driving(pos, vel, beta) + repulsion(pos) + wall_forces(pos)
        vel += (F / MASS) * DT
        pos += vel * DT
        beta = np.clip(beta + stress_deriv(beta, pos, kappa) * DT, 0.0, 1.0)

        exited = ((pos[:, 0] > ROOM_X[1])
                  & (pos[:, 1] > EXIT_Y[0])
                  & (pos[:, 1] < EXIT_Y[1]))
        pos  = pos[~exited]
        vel  = vel[~exited]
        beta = beta[~exited]
        t   += DT

    return t if len(pos) == 0 else T_MAX     # T_MAX flags "didn't finish"


# ── κ sweep ───────────────────────────────────────────────────────────────────
kappa_arr = np.linspace(0.0, 3.0 * KAPPA_C, N_KAPPA)
T_all     = np.full((N_KAPPA, N_SIM_SEEDS), np.nan)

total = N_KAPPA * N_SIM_SEEDS
done  = 0
print(f"Running {total} evacuations  (N={N_AGENTS}, exit={EXIT_Y[1]-EXIT_Y[0]:.1f} m) …")

for i, kappa in enumerate(kappa_arr):
    for s in range(N_SIM_SEEDS):
        T_all[i, s] = run_evacuation(kappa, sim_seed=s * 1000 + i)
        done += 1
        print(f"  [{done:3d}/{total}]  κ={kappa:.4f}  T={T_all[i,s]:.1f}s", end="\r")

print(f"\nDone.  {(T_all >= T_MAX).sum()} runs hit T_MAX (didn't fully evacuate).")

T_mean = T_all.mean(axis=1)
T_std  = T_all.std(axis=1)
i_opt  = np.argmin(T_mean)
kappa_opt = kappa_arr[i_opt]
T_opt     = T_mean[i_opt]
print(f"Optimal κ* = {kappa_opt:.4f}  (T* = {T_opt:.1f} s,  κ*/κ_c = {kappa_opt/KAPPA_C:.2f})")


# ── Mean-field theory curves ──────────────────────────────────────────────────
kappa_fine  = np.linspace(1e-4, kappa_arr[-1], 600)
Lambda_fine = kappa_fine * W_ROW_MEAN
beta_mf     = np.where(Lambda_fine > MU, 1.0 - MU / Lambda_fine, 0.0)
v0_mf       = V_MIN + beta_mf * (V_MAX - V_MIN)
alpha_mf    = (A * TAU) / (MASS * v0_mf)

# α at κ=0 baseline (no contagion)
alpha_baseline = (A * TAU) / (MASS * V_MIN)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.patch.set_facecolor("#f8f8f8")
C_EVAC = "#2176ae"
C_BETA = "#c1121f"
C_V0   = "#7b2d8b"
C_ALP  = "#1b5e20"

# ─── Panel 1 : T(κ) ──────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("white")

# Shade FIS regime (above κ_c)
ax.axvspan(KAPPA_C, kappa_arr[-1], alpha=0.07, color="firebrick", label="FIS regime")
# Shade optimal zone
if kappa_opt > 0:
    ax.axvspan(0, kappa_opt, alpha=0.06, color="limegreen")

# ± std band
ax.fill_between(kappa_arr, T_mean - T_std, T_mean + T_std,
                color=C_EVAC, alpha=0.22, label=r"$\pm 1\,\sigma$")
# mean curve
ax.plot(kappa_arr, T_mean, color=C_EVAC, lw=2.8, zorder=4,
        label=r"$\bar T(\kappa)$")

# κ_c line
ax.axvline(KAPPA_C, color="gray", ls="--", lw=1.6, zorder=3)
ax.text(KAPPA_C * 1.02, T_mean.max() * 0.96,
        r"$\kappa_c$", fontsize=12, color="gray", ha="left")

# optimal star
ax.plot(kappa_opt, T_opt, "*", ms=16, color="gold",
        markeredgecolor="darkorange", markeredgewidth=1.5,
        zorder=6, label=fr"$\kappa^* = {kappa_opt:.3f}$  (min $T$)")

# FIS annotation arrow
fis_x = KAPPA_C * 1.6
fis_i = np.argmin(np.abs(kappa_arr - fis_x))
ax.annotate("Faster-is-Slower\n(panic → clogging → slower exit)",
            xy=(fis_x, T_mean[fis_i]),
            xytext=(fis_x - KAPPA_C * 0.5, T_mean[fis_i] + (T_mean.max() - T_mean.min()) * 0.25),
            fontsize=8.5, color="firebrick", ha="center",
            arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.3))

ax.set_xlabel(r"Contagion strength  $\kappa$  (s$^{-1}$)", fontsize=11)
ax.set_ylabel(r"Evacuation time  $T$  (s)", fontsize=11)
ax.set_title("Evacuation Time vs\nStress Contagion Strength", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.25)
ax.set_xlim(kappa_arr[0], kappa_arr[-1])


# ─── Panel 2 : β̄∞(κ) and v₀(κ) ─────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("white")
ax2r = ax2.twinx()

l1, = ax2.plot(kappa_fine, beta_mf, color=C_BETA, lw=2.5,
               label=r"$\bar\beta_\infty(\kappa)$  [mean-field]")
ax2.axvline(KAPPA_C, color="gray", ls="--", lw=1.6)
ax2.text(KAPPA_C * 1.02, 0.95, r"$\kappa_c$", fontsize=12, color="gray")
ax2.axvline(kappa_opt, color="darkorange", ls=":", lw=1.5, alpha=0.8)
ax2.set_ylim(-0.05, 1.12)
ax2.set_ylabel(r"Equilibrium mean stress  $\bar\beta_\infty$",
               fontsize=11, color=C_BETA)
ax2.tick_params(axis="y", colors=C_BETA)

l2, = ax2r.plot(kappa_fine, v0_mf, color=C_V0, lw=2.5, ls="-.",
                label=r"$v_0(\bar\beta)$  (m/s)")
ax2r.axhline(V_MIN, color=C_V0, ls=":", lw=1.0, alpha=0.5)
ax2r.axhline(V_MAX, color=C_V0, ls=":", lw=1.0, alpha=0.5)
ax2r.text(kappa_fine[-1] * 0.98, V_MIN + 0.05, r"$v_{\min}$", color=C_V0,
          fontsize=9, ha="right")
ax2r.text(kappa_fine[-1] * 0.98, V_MAX + 0.05, r"$v_{\max}$", color=C_V0,
          fontsize=9, ha="right")
ax2r.set_ylim(V_MIN - 0.3, V_MAX + 0.6)
ax2r.set_ylabel(r"Desired speed  $v_0$  (m/s)", fontsize=11, color=C_V0)
ax2r.tick_params(axis="y", colors=C_V0)

ax2.set_xlabel(r"Contagion strength  $\kappa$  (s$^{-1}$)", fontsize=11)
ax2.set_title("Mechanism: Stress Raises\nDesired Speed", fontsize=11, fontweight="bold")
ax2.legend(handles=[l1, l2], fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.25)
ax2.set_xlim(kappa_fine[0], kappa_fine[-1])

# Chain annotation
ax2.annotate("",
             xy=(KAPPA_C * 1.5, 0.35), xytext=(KAPPA_C * 2.2, 0.55),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
ax2.text(KAPPA_C * 2.25, 0.57,
         r"$\kappa\uparrow\;\Rightarrow\;\bar\beta\uparrow$"
         "\n" r"$\Rightarrow\;v_0\uparrow\;\Rightarrow\;\alpha\downarrow$",
         fontsize=8.5, ha="left")


# ─── Panel 3 : α(κ) ─────────────────────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor("white")

ax3.plot(kappa_fine, alpha_mf, color=C_ALP, lw=2.8,
         label=r"$\alpha(\kappa) = A\tau\,/\,(m\,v_0(\bar\beta_\infty))$")

# α=1 critical line
ax3.axhline(1.0, color="firebrick", ls="--", lw=1.8,
            label=r"$\alpha = 1$  (drive = repulsion)")

# shade regimes
ax3.fill_between(kappa_fine, alpha_mf, 1.0,
                 where=(alpha_mf > 1.0), alpha=0.10, color="steelblue",
                 label="Repulsion-dominated\n(free flow)")
ax3.fill_between(kappa_fine, alpha_mf, 1.0,
                 where=(alpha_mf < 1.0), alpha=0.12, color="firebrick",
                 label="Drive-dominated\n(clogging / FIS)")

ax3.axvline(KAPPA_C, color="gray", ls="--", lw=1.6)
ax3.text(KAPPA_C * 1.02, alpha_mf.max() * 0.96,
         r"$\kappa_c$", fontsize=12, color="gray")
ax3.axvline(kappa_opt, color="darkorange", ls=":", lw=1.5,
            label=fr"$\kappa^* = {kappa_opt:.3f}$")

# α at calm baseline
ax3.axhline(alpha_baseline, color=C_ALP, ls=":", lw=1.2, alpha=0.6)
ax3.text(kappa_fine[-1] * 0.98, alpha_baseline + alpha_mf.max() * 0.02,
         fr"$\alpha(\kappa{{=}}0) = {alpha_baseline:.1f}$",
         fontsize=8.5, color=C_ALP, ha="right")

ax3.set_xlabel(r"Contagion strength  $\kappa$  (s$^{-1}$)", fontsize=11)
ax3.set_ylabel(r"Repulsion-to-drive ratio  $\alpha = A\tau/(mv_0)$", fontsize=11)
ax3.set_title(r"$\alpha(\kappa)$: When Does Clogging Begin?", fontsize=11,
              fontweight="bold")
ax3.legend(fontsize=8.5, loc="upper right")
ax3.grid(True, alpha=0.25)
ax3.set_ylim(0, alpha_mf.max() * 1.12)
ax3.set_xlim(kappa_fine[0], kappa_fine[-1])

# Super-title
fig.suptitle(
    fr"Evacuation Time vs Stress Contagion  —  SSFM + Stress  "
    fr"($N={N_AGENTS}$, exit={EXIT_Y[1]-EXIT_Y[0]:.1f} m, "
    fr"{N_SIM_SEEDS} seeds, $\mu={MU}$, $\ell={ELL}$ m)"
    "\n"
    r"Chain: $\kappa\uparrow\;\Rightarrow\;\bar\beta\uparrow\;"
    r"\Rightarrow\;v_0\uparrow\;\Rightarrow\;\alpha\downarrow\;"
    r"\Rightarrow\;$ clogging  (Faster-is-Slower)",
    fontsize=11, fontweight="bold", y=1.02,
)

plt.tight_layout()
out = "evacuation_vs_stress.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
