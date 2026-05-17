"""
Smooth Social Force Model with Stress Contagion
================================================
Equations (all in SI units):

  Position:   x_dot_i = v_i
  Velocity:   m v_dot_i = (m/tau)(v0(beta_i) e_i - v_i)
                         + A * sum_j exp((2r - d_ij)/B) n_ij
                         + wall repulsion
  Stress:     beta_dot_i = -mu beta_i
                           + kappa * sum_j exp(-d_ij/ell) beta_j (1 - beta_i)
                           + sigma_i(t)
  Speed law:  v0(beta_i) = v_min + beta_i * (v_max - v_min)

Mean-field reduction (Section 4, stress_contagion_crowd.pdf):
  Assume all beta_i ≈ beta(t) and define Lambda = kappa * rho * w_bar.
  Quadratic:  beta_dot = -mu beta + Lambda beta (1 - beta)
  Cubic:      beta_dot = -mu beta + Lambda beta (1 - beta) + gamma beta^2 (1 - beta)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ── Physical parameters ──────────────────────────────────────────────────────
MASS   = 80.0    # kg
TAU    = 0.5     # relaxation time (s)
R      = 0.25    # agent radius (m)
A      = 2000.0  # repulsion strength (N)
B      = 0.08    # repulsion range (m)
AW     = 2000.0  # wall repulsion strength (N)
BW     = 0.08    # wall repulsion range (m)

V_MIN  = 1.0     # calm desired speed (m/s)
V_MAX  = 4.0     # panic desired speed (m/s)

# Stress parameters (from paper Table 8.4)
MU     = 0.3     # natural calming rate (s^-1)
KAPPA  = 0.5     # contagion coupling (varied in bifurcation plot)
ELL    = 3.0     # spatial influence range (m)
GAMMA  = 0.5     # cubic self-reinforcement (s^-1)

DT     = 0.02    # timestep (s)


# ── Speed law ────────────────────────────────────────────────────────────────
def desired_speed(beta):
    return V_MIN + beta * (V_MAX - V_MIN)


# ── SSFM forces ──────────────────────────────────────────────────────────────
def self_driving_forces(pos, vel, betas, target):
    diff  = target - pos
    dists = np.linalg.norm(diff, axis=1, keepdims=True)
    dirs  = diff / np.maximum(dists, 1e-9)
    v0    = desired_speed(betas)[:, None]
    return MASS * (v0 * dirs - vel) / TAU


def repulsion_forces(pos):
    N = len(pos)
    F = np.zeros((N, 2))
    for i in range(N):
        d_vec = pos[i] - pos          # (N, 2)
        dists = np.linalg.norm(d_vec, axis=1)
        dists[i] = 1.0                # avoid self
        mask  = dists > 1e-6
        n_hat = d_vec / dists[:, None]
        mag   = A * np.exp((2 * R - dists) / B)
        F[i] += (mag[:, None] * n_hat * mask[:, None]).sum(axis=0)
    return F


def wall_forces(pos, room_x, room_y, exit_y):
    N = len(pos)
    F = np.zeros((N, 2))
    # left wall  (normal = +x)
    d = np.maximum(pos[:, 0] - room_x[0], 1e-9)
    F[:, 0] += AW * np.exp((R - d) / BW)
    # bottom wall (normal = +y)
    d = np.maximum(pos[:, 1] - room_y[0], 1e-9)
    F[:, 1] += AW * np.exp((R - d) / BW)
    # top wall (normal = -y)
    d = np.maximum(room_y[1] - pos[:, 1], 1e-9)
    F[:, 1] -= AW * np.exp((R - d) / BW)
    # right wall: solid except at exit gap
    for i in range(N):
        if pos[i, 1] < exit_y[0] or pos[i, 1] > exit_y[1]:
            d = max(room_x[1] - pos[i, 0], 1e-9)
            F[i, 0] -= AW * np.exp((R - d) / BW)
    return F


# ── Stress update ────────────────────────────────────────────────────────────
def stress_update(betas, pos, kappa, sigma=None):
    N = len(betas)
    dbeta = -MU * betas
    for i in range(N):
        d_vec = pos[i] - pos
        dists = np.linalg.norm(d_vec, axis=1)
        dists[i] = np.inf
        w_ij  = np.exp(-dists / ELL)
        dbeta[i] += kappa * np.sum(w_ij * betas * (1.0 - betas[i]))
    if sigma is not None:
        dbeta += sigma
    return dbeta


# ── Full simulation ──────────────────────────────────────────────────────────
def run_simulation(N=40, kappa=KAPPA, t_end=30.0, seed_fraction=0.1,
                   room_x=(0, 10), room_y=(0, 8), exit_y=(3.5, 4.5),
                   seed=42):
    rng = np.random.default_rng(seed)
    target = np.array([room_x[1] + 0.5, (exit_y[0] + exit_y[1]) / 2])

    # Initial positions on a grid
    cols  = int(np.ceil(np.sqrt(N)))
    rows  = int(np.ceil(N / cols))
    xs    = np.linspace(room_x[0] + 1, room_x[1] - 3, cols)
    ys    = np.linspace(room_y[0] + 1, room_y[1] - 1, rows)
    gx, gy = np.meshgrid(xs, ys)
    pos   = np.column_stack([gx.ravel(), gy.ravel()])[:N].astype(float)
    vel   = np.zeros((N, 2))
    betas = np.zeros(N)

    # Seed a fraction as panicked
    n_seed = max(1, int(seed_fraction * N))
    idx    = rng.choice(N, n_seed, replace=False)
    betas[idx] = 0.8

    t          = 0.0
    history    = {"t": [], "beta_mean": [], "speed_mean": [], "n_agents": []}

    while t < t_end and len(pos) > 0:
        F = (self_driving_forces(pos, vel, betas, target)
             + repulsion_forces(pos)
             + wall_forces(pos, room_x, room_y, exit_y))

        vel    += F / MASS * DT
        pos    += vel * DT
        dbeta   = stress_update(betas, pos, kappa)
        betas   = np.clip(betas + dbeta * DT, 0.0, 1.0)

        # Remove agents that crossed the exit
        exited = (pos[:, 0] > room_x[1]) & (pos[:, 1] > exit_y[0]) & (pos[:, 1] < exit_y[1])
        pos    = pos[~exited]
        vel    = vel[~exited]
        betas  = betas[~exited]

        history["t"].append(t)
        history["beta_mean"].append(betas.mean() if len(betas) > 0 else 0.0)
        history["speed_mean"].append(np.linalg.norm(vel, axis=1).mean() if len(vel) > 0 else 0.0)
        history["n_agents"].append(len(pos))

        t += DT

    return history


# ── Mean-field bifurcation analysis ─────────────────────────────────────────
def mean_field_quadratic(beta, Lambda):
    """β̇ / β  for the quadratic model (β ≠ 0)."""
    return -MU + Lambda * (1 - beta)


def mean_field_cubic(beta, Lambda):
    """β̇ / β  for the cubic model (β ≠ 0)."""
    return -MU + Lambda * (1 - beta) + GAMMA * beta * (1 - beta)


def find_nontrivial_roots(g_func, Lambda, n_scan=4000):
    """Find roots of g(beta) = 0 in (0, 1) by sign-change scanning + brentq."""
    betas = np.linspace(1e-6, 1 - 1e-6, n_scan)
    vals  = np.array([g_func(b, Lambda) for b in betas])
    roots = []
    for i in range(n_scan - 1):
        if vals[i] * vals[i + 1] < 0:
            try:
                r = brentq(g_func, betas[i], betas[i + 1], args=(Lambda,))
                roots.append(r)
            except ValueError:
                pass
    return roots


def stability(g_func, beta, Lambda, eps=1e-5):
    """Sign of dg/dbeta at beta (negative = stable non-trivial equilibrium)."""
    return (g_func(beta + eps, Lambda) - g_func(beta - eps, Lambda)) / (2 * eps)


def build_bifurcation_data(g_func, Lambda_arr):
    stable   = []  # (Lambda, beta*)
    unstable = []
    for L in Lambda_arr:
        roots = find_nontrivial_roots(g_func, L)
        for r in roots:
            s = stability(g_func, r, L)
            if s < 0:
                stable.append((L, r))
            else:
                unstable.append((L, r))
    return np.array(stable) if stable else np.empty((0, 2)), \
           np.array(unstable) if unstable else np.empty((0, 2))


# ── Bifurcation diagram ──────────────────────────────────────────────────────
def plot_bifurcation():
    Lambda_arr = np.linspace(0.001, 1.4, 1400)

    # Quadratic model
    q_stable, q_unstable = build_bifurcation_data(mean_field_quadratic, Lambda_arr)

    # Cubic model
    c_stable, c_unstable = build_bifurcation_data(mean_field_cubic, Lambda_arr)

    # Saddle-node location for cubic
    Lambda_SN = 2 * np.sqrt(GAMMA * MU) - GAMMA
    beta_SN   = (GAMMA - Lambda_SN) / (2 * GAMMA)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#f8f8f8")

    colors = dict(calm="#2176ae", panic="#c1121f", unstable="#888888",
                  bifpt="black", sn="#e07b00", bistable="#9b59b6")

    # ── Left panel: Transcritical / Quadratic ────────────────────
    ax = axes[0]
    ax.set_facecolor("white")

    # β*=0 branch
    ax.plot([0, MU], [0, 0], color=colors["calm"], lw=2.5,
            label=r"$\beta^* = 0$  (stable)")
    ax.plot([MU, Lambda_arr[-1]], [0, 0], color=colors["calm"], lw=2.5,
            ls="--", label=r"$\beta^* = 0$  (unstable)")

    # β*+ = 1 − µ/Λ branch
    Lp = Lambda_arr[Lambda_arr > MU]
    ax.plot(Lp, 1 - MU / Lp, color=colors["panic"], lw=2.5,
            label=r"$\beta^*_+ = 1-\mu/\Lambda$  (stable)")

    # non-trivial roots from scan (should match above; shown as a check)
    if len(q_stable):
        ax.scatter(q_stable[:, 0], q_stable[:, 1],
                   color=colors["panic"], s=3, zorder=4, alpha=0.4)
    if len(q_unstable):
        ax.scatter(q_unstable[:, 0], q_unstable[:, 1],
                   color=colors["unstable"], s=3, zorder=4, alpha=0.3)

    # Bifurcation point
    ax.axvline(MU, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax.plot(MU, 0, "o", color=colors["bifpt"], ms=9, zorder=6,
            label=fr"Bifurcation  $\Lambda_c = \mu = {MU}$")

    ax.annotate(r"$\Lambda_c = \mu$", xy=(MU, 0), xytext=(MU + 0.06, -0.09),
                fontsize=11, ha="left",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax.text(0.12, 0.75, "CALM\n" r"($\beta^*=0$ attracts)", fontsize=11,
            ha="center", color=colors["calm"], style="italic")
    ax.text(1.1, 0.55, "PANIC\n" r"($\beta^*_+$ attracts)", fontsize=11,
            ha="center", color=colors["panic"], style="italic")

    ax.set_xlim(0, Lambda_arr[-1])
    ax.set_ylim(-0.18, 1.12)
    ax.set_xlabel(r"Effective contagion rate  $\Lambda = \kappa\rho\bar{w}$",
                  fontsize=12)
    ax.set_ylabel(r"Equilibrium stress  $\beta^*$", fontsize=12)
    ax.set_title("Transcritical Bifurcation\n"
                 r"(Quadratic model,  $\dot\beta = -\mu\beta + \Lambda\beta(1-\beta)$)",
                 fontsize=11)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # ── Right panel: Saddle-node / Hysteresis / Cubic ────────────
    ax = axes[1]
    ax.set_facecolor("white")

    # β*=0 branch
    ax.plot([0, MU], [0, 0], color=colors["calm"], lw=2.5,
            label=r"$\beta^* = 0$  (stable)")
    ax.plot([MU, Lambda_arr[-1]], [0, 0], color=colors["calm"], lw=2.5,
            ls="--", label=r"$\beta^* = 0$  (unstable)")

    # Non-trivial branches from numerical scan
    if len(c_stable):
        ax.plot(c_stable[:, 0], c_stable[:, 1],
                color=colors["panic"], lw=2.5, label="Upper branch (stable)")
    if len(c_unstable):
        ax.plot(c_unstable[:, 0], c_unstable[:, 1],
                color=colors["unstable"], lw=2.0, ls="--",
                label="Middle branch (unstable)")

    # Bistable shading
    if Lambda_SN > 0:
        ax.axvspan(Lambda_SN, MU, alpha=0.12, color=colors["bistable"],
                   label="Bistable region")

    # Key bifurcation points
    ax.axvline(MU, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax.axvline(Lambda_SN, color=colors["sn"], ls=":", lw=1.5, alpha=0.8)

    ax.plot(MU, 0, "o", color=colors["bifpt"], ms=9, zorder=6)
    ax.plot(Lambda_SN, beta_SN, "s", color=colors["sn"], ms=9, zorder=6,
            label=fr"Saddle-node  $\Lambda_{{SN}} = {Lambda_SN:.3f}$")

    ax.annotate(r"$\Lambda_c = \mu$", xy=(MU, 0),
                xytext=(MU + 0.06, -0.09), fontsize=10, ha="left",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.annotate(r"$\Lambda_{SN}$", xy=(Lambda_SN, beta_SN),
                xytext=(Lambda_SN - 0.18, beta_SN + 0.1), fontsize=10, ha="right",
                color=colors["sn"],
                arrowprops=dict(arrowstyle="->", color=colors["sn"], lw=1.2))

    # Hysteresis arrows
    mid_L  = (Lambda_SN + MU) / 2
    y_calm = -0.13
    y_panic = 0.55
    ax.annotate("", xy=(Lambda_SN + 0.01, y_calm),
                xytext=(MU - 0.01, y_calm),
                arrowprops=dict(arrowstyle="<-", color=colors["sn"], lw=2))
    ax.annotate("", xy=(MU - 0.01, y_panic),
                xytext=(Lambda_SN + 0.01, y_panic),
                arrowprops=dict(arrowstyle="<-", color=colors["panic"], lw=2))
    ax.text(mid_L, y_calm - 0.04, "ramp down  ↓ calm", fontsize=8.5,
            ha="center", color=colors["sn"])
    ax.text(mid_L, y_panic + 0.03, "ramp up  ↑ panic", fontsize=8.5,
            ha="center", color=colors["panic"])

    ax.set_xlim(0, Lambda_arr[-1])
    ax.set_ylim(-0.22, 1.12)
    ax.set_xlabel(r"Effective contagion rate  $\Lambda = \kappa\rho\bar{w}$",
                  fontsize=12)
    ax.set_ylabel(r"Equilibrium stress  $\beta^*$", fontsize=12)
    ax.set_title("Saddle-Node Bifurcation  /  Hysteresis\n"
                 r"(Cubic model,  $\dot\beta = -\mu\beta + \Lambda\beta(1-\beta) + \gamma\beta^2(1-\beta)$,"
                 f"  $\\gamma={GAMMA}$)",
                 fontsize=10)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Shared super-title
    fig.suptitle(
        "Stress Contagion Bifurcation Diagrams — Smooth Social Force Model\n"
        fr"Parameters: $\mu = {MU}$,  $\gamma = {GAMMA}$  (from stress_contagion_crowd.pdf §8.4)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    out = "bifurcation_stress.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    return out


if __name__ == "__main__":
    plot_bifurcation()
