"""
Evacuation time vs urgency rate λ — exit-proximity stress + KAPPA friction.

No seed agents.  Each agent's stress β grows proportionally to how close
they are to the exit (x > 6 m).  λ sets how strongly proximity translates
into urgency.

Expected U-shape:
  λ=0        → all agents at V_MIN=1.0 m/s → slow (no urgency, low desired speed)
  λ_optimal  → moderate urgency at exit zone → v0 ≈ 4 m/s → fastest throughput
  λ too high → agents near exit immediately panic → v0 → 5.5 m/s → excess crowd
               pressure → KAPPA friction arches → FIS → slower evacuation

Stress model:
  β̇_i = −µ·β_i  +  λ · clip((x_i − 6)/4, 0, 1) · (1 − β_i)
  v0_i  = V_MIN + β_i · (V_MAX − V_MIN)

FIS mechanism: v0 > ABS_V_MAX → self-driving force remains non-zero at cap
  → extra crowd pressure → KAPPA tangential friction → arch formation.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── room ─────────────────────────────────────────────────────────
ROOM_X = (0.0, 10.0)
ROOM_Y = (0.0, 6.0)
EXIT_Y = (2.55, 3.45)   # 0.9 m door
target = np.array([10.5, 3.0])

# ── physics ───────────────────────────────────────────────────────
MASS      = 70.0
TAU       = 0.5
DT        = 0.01
T_END     = 70.0
R_AGENT   = 0.3
R_WALL    = 0.3
A_SOC     = 2000.0
B_SOC     = 0.08
KAPPA     = 240_000.0   # tangential friction — arch formation at high crowd pressure
ABS_V_MAX = 3.5         # speed cap; v0 > 3.5 → excess force → FIS mechanism

# ── stress model ─────────────────────────────────────────────────
V_MIN     = 1.0    # calm desired speed — clearly suboptimal, left side of U-curve
V_MAX     = 5.5    # panic desired speed (exceeds ABS_V_MAX → excess crowd force)
MU        = 0.5    # stress decay rate — β decays in ~2 s when not near exit
X_STRESS  = 6.0    # stress zone boundary: agents with x > X_STRESS feel urgency
# proximity signal: 0 at x=6, 1 at x=10 (exit wall)

N        = 55
N_SEEDS  = 20     # Monte Carlo runs (random initial positions)

lambda_range = np.unique(np.concatenate([
    np.linspace(0.0, 0.5,  4),
    np.linspace(0.5, 2.0,  9),
    np.linspace(2.0, 6.0,  6),
]))


# ── forces ───────────────────────────────────────────────────────
def self_driving(pos, vel, v0_arr):
    diff = target - pos
    d    = np.linalg.norm(diff, axis=1, keepdims=True)
    e    = diff / np.maximum(d, 1e-6)
    return MASS * (v0_arr[:, None] * e - vel) / TAU


def agent_repulsion(pos, vel):
    """Exponential social repulsion + KAPPA tangential friction."""
    n = len(pos)
    if n < 2:
        return np.zeros((n, 2))
    diff  = pos[:, None, :] - pos[None, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, 1.0)
    dist  = np.sqrt(dist2)
    np.fill_diagonal(dist, 1.0)
    n_vec   = diff / dist[:, :, None]
    t_vec   = np.stack([-n_vec[:, :, 1], n_vec[:, :, 0]], axis=-1)
    overlap = 2 * R_AGENT - dist
    g       = np.maximum(overlap, 0.0)
    dv_t    = np.einsum('ijk,ijk->ij', vel[None, :, :] - vel[:, None, :], t_vec)
    mask    = ~np.eye(n, dtype=bool)
    soc_mag = A_SOC * np.exp(overlap / B_SOC)
    F_norm  = np.einsum('ij,ijk->ik', soc_mag * mask, n_vec)
    F_fric  = np.einsum('ij,ijk->ik', (KAPPA * g * dv_t) * mask, t_vec)
    return F_norm + F_fric


def wall_forces(pos):
    F = np.zeros_like(pos)

    def push(dist_vec, normal):
        d   = np.maximum(dist_vec, 1e-6)
        mag = A_SOC * np.exp((R_WALL - d) / B_SOC)
        return mag[:, None] * normal

    F += push(pos[:, 0] - ROOM_X[0], np.array([ 1.0,  0.0]))
    F += push(pos[:, 1] - ROOM_Y[0], np.array([ 0.0,  1.0]))
    F += push(ROOM_Y[1] - pos[:, 1], np.array([ 0.0, -1.0]))
    not_in_exit = (pos[:, 1] < EXIT_Y[0]) | (pos[:, 1] > EXIT_Y[1])
    if np.any(not_in_exit):
        sub = pos[not_in_exit]
        F[not_in_exit] += push(ROOM_X[1] - sub[:, 0], np.array([-1.0, 0.0]))
    return F


def stress_step(beta, pos, lam):
    """Exit-proximity urgency: β grows when agent is in the exit zone (x > X_STRESS)."""
    proximity = np.clip((pos[:, 0] - X_STRESS) / (ROOM_X[1] - X_STRESS), 0.0, 1.0)
    if lam == 0.0:
        return np.clip(beta - MU * beta * DT, 0.0, 1.0)
    contagion = lam * proximity * (1.0 - beta)
    return np.clip(beta + (-MU * beta + contagion) * DT, 0.0, 1.0)


# ── initial layout ────────────────────────────────────────────────
def initial_positions(n, seed=0):
    spacing  = 2 * R_AGENT + 0.08
    y_lo     = ROOM_Y[0] + spacing
    y_hi     = ROOM_Y[1] - spacing
    max_rows = max(1, int((y_hi - y_lo) / spacing) + 1)
    cols     = max(5, int(np.ceil(n / max_rows)))
    rows     = int(np.ceil(n / cols))
    x_hi     = min(ROOM_X[0] + cols * spacing + 0.5, ROOM_X[1] - 0.5)
    xs       = np.linspace(ROOM_X[0] + spacing, x_hi, cols)
    ys       = np.linspace(y_lo, y_hi, rows)
    gx, gy   = np.meshgrid(xs, ys)
    base     = np.column_stack([gx.ravel(), gy.ravel()])[:n].copy()
    rng      = np.random.default_rng(seed)
    base    += rng.uniform(-min(0.12, spacing * 0.15), min(0.12, spacing * 0.15), base.shape)
    return base


# ── single run ────────────────────────────────────────────────────
def run(lam, seed=0):
    pos  = initial_positions(N, seed=seed)
    vel  = np.zeros((N, 2))
    beta = np.zeros(N)   # all calm at t=0; stress builds from exit proximity

    stuck_counter = np.zeros(N)
    rng_kick      = np.random.default_rng(seed + 42)
    evac_time     = T_END
    t             = 0.0

    for step in range(int(T_END / DT)):
        if len(pos) == 0:
            evac_time = t
            break
        v0_arr = V_MIN + beta * (V_MAX - V_MIN)
        F      = self_driving(pos, vel, v0_arr) + agent_repulsion(pos, vel) + wall_forces(pos)
        vel   += F / MASS * DT
        spds   = np.linalg.norm(vel, axis=1, keepdims=True)
        vel    = np.where(spds > ABS_V_MAX, vel / spds * ABS_V_MAX, vel)
        beta   = stress_step(beta, pos, lam)
        pos   += vel * DT
        t     += DT
        exited = pos[:, 0] > ROOM_X[1] - 0.1
        pos  = pos[~exited];  vel  = vel[~exited]
        beta = beta[~exited]; stuck_counter = stuck_counter[~exited]
        if 0 < len(pos) <= 5:
            spds_now = np.linalg.norm(vel, axis=1)
            stuck_counter = np.where(spds_now < 0.05, stuck_counter + DT, 0.0)
            need_kick = stuck_counter > 8.0
            if np.any(need_kick):
                to_exit = target - pos[need_kick]
                to_exit /= np.maximum(np.linalg.norm(to_exit, axis=1, keepdims=True), 1e-6)
                speed = 1.0 + rng_kick.uniform(0.0, 0.5, need_kick.sum())
                vel[need_kick] = to_exit * speed[:, None]
                vel[need_kick] += rng_kick.uniform(-0.2, 0.2, vel[need_kick].shape)
                stuck_counter[need_kick] = 0.0
        else:
            stuck_counter[:] = 0.0
    return evac_time


# ── sweep ─────────────────────────────────────────────────────────
print(f"Sweeping λ over {len(lambda_range)} values × {N_SEEDS} seeds…")
means, stds, medians, q25s, q75s, deadlocks = [], [], [], [], [], []

for lam in lambda_range:
    ets = np.array([run(lam, seed=s) for s in range(N_SEEDS)])
    means.append(ets.mean())
    stds.append(ets.std())
    medians.append(np.median(ets))
    q25s.append(np.percentile(ets, 25))
    q75s.append(np.percentile(ets, 75))
    deadlocks.append(int(np.sum(ets >= T_END)))
    print(f"  λ={lam:.2f}  mean={means[-1]:.1f}±{stds[-1]:.1f}s  "
          f"median={medians[-1]:.1f}s  deadlocks={deadlocks[-1]}/{N_SEEDS}")

means     = np.array(means)
stds      = np.array(stds)
medians   = np.array(medians)
q25s      = np.array(q25s)
q75s      = np.array(q75s)
deadlocks = np.array(deadlocks)

opt_idx  = int(np.argmin(means))
lam_opt  = lambda_range[opt_idx]
t_opt    = means[opt_idx]


# ── plot ──────────────────────────────────────────────────────────
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                               gridspec_kw={'height_ratios': [3, 1]},
                               sharex=True)
fig.patch.set_facecolor('#1a1a2e')
for a in (ax, ax2):
    a.set_facecolor('#16213e')

# ── main panel: mean ± std ────────────────────────────────────────
ax.fill_between(lambda_range, means - stds, means + stds,
                color='#88ccff', alpha=0.15, label='±1 std')
ax.fill_between(lambda_range, q25s, q75s,
                color='#88ccff', alpha=0.25, label='IQR')
ax.plot(lambda_range, means, 'o-', color='#88ccff', lw=2, ms=5,
        label='mean evacuation time')
ax.plot(lambda_range, medians, 's--', color='#aaddff', lw=1.2, ms=4,
        alpha=0.7, label='median')

# optimal point
ax.axvline(lam_opt, color='#ff8844', lw=1.5, ls='--',
           label=f'optimal λ = {lam_opt:.2f}')
ax.scatter([lam_opt], [t_opt], color='#ff8844', s=120, zorder=6)

# T_END reference
ax.axhline(T_END, color='#666688', lw=1, ls=':', alpha=0.7)
ax.text(lambda_range[-1] - 0.05, T_END + 0.8,
        f'T_end={T_END:.0f}s (deadlock cap)',
        color='#888899', fontsize=8, ha='right')

# U-shape labels
ax.text(lam_opt * 0.35, means[0] * 0.88,
        '← too slow\n   (low urgency)',
        color='#88ccff', fontsize=9, ha='center')
ax.axvspan(lam_opt, lambda_range[-1], alpha=0.06, color='red')
ax.text(lam_opt + 0.15, means[means > t_opt * 1.05][0] if any(means > t_opt * 1.05) else T_END * 0.7,
        'panic → FIS →', color='#ff6666', fontsize=9)

ax.set_ylabel('Evacuation time (s)', color='white', fontsize=12)
ax.set_title(
    f'U-shaped evacuation time: optimal urgency rate λ\n'
    f'(N={N}, exit-proximity stress, KAPPA={int(KAPPA/1000)}k friction, '
    f'V_min={V_MIN} m/s, V_max={V_MAX} m/s, {N_SEEDS} runs)',
    color='white', fontsize=11, fontweight='bold'
)
ax.legend(framealpha=0.2, labelcolor='white', fontsize=9, loc='upper right')
ax.tick_params(colors='white')
for s in ax.spines.values():
    s.set_edgecolor('#444466')
ax.grid(True, alpha=0.15, color='white')

# ── lower panel: deadlock fraction ───────────────────────────────
dl_frac = deadlocks / N_SEEDS
ax2.bar(lambda_range, dl_frac, width=(lambda_range[1] - lambda_range[0]) * 0.8,
        color='#ff6666', alpha=0.7, label='deadlock fraction')
ax2.axvline(lam_opt, color='#ff8844', lw=1.5, ls='--')
ax2.set_xlabel('Urgency rate  λ', color='white', fontsize=12)
ax2.set_ylabel('P(deadlock)', color='white', fontsize=10)
ax2.set_ylim(0, 1)
ax2.tick_params(colors='white')
for s in ax2.spines.values():
    s.set_edgecolor('#444466')
ax2.grid(True, alpha=0.15, color='white')
ax2.legend(framealpha=0.2, labelcolor='white', fontsize=9)

plt.tight_layout()
plt.savefig('stress_lambda_sweep.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()
