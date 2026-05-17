"""
Faster-is-slower via stress contagion — smooth SFM (SSFM) + per-agent stress β_i.

4 panels with increasing contagion strength λ.  Each drives the crowd to a
different effective panic level and average desired speed.

Force model: smooth SFM (k=0, κ=0 — exponential repulsion only, no body
contact spring, no tangential friction).

FIS mechanism: ABS_V_MAX = 3.5 m/s caps actual speed, but self-driving force
∝ (v0 − 3.5) remains non-zero.  Panicked agents push harder without moving
faster → crowd pressure builds → congestion at bottleneck.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── room ─────────────────────────────────────────────────────────
ROOM_X = (0.0, 10.0)
ROOM_Y = (0.0, 6.0)
EXIT_Y = (2.55, 3.45)   # 0.9 m door — narrow enough for arch formation
target = np.array([10.5, 3.0])

# ── full SFM physics ─────────────────────────────────────────────
MASS      = 70.0
TAU       = 0.5
DT        = 0.01
T_END     = 70.0
R_AGENT   = 0.3
R_WALL    = 0.3
A_SOC     = 2000.0
B_SOC     = 0.08
# smooth SFM: k=0, κ=0 — no body contact spring, no tangential friction
ABS_V_MAX = 3.5          # speed ceiling — agents above this push harder (FIS)

# ── stress model ─────────────────────────────────────────────────
V_MIN    = 1.5    # calm walking speed (m/s)
V_MAX    = 6.5    # full-panic desired speed — well above cap → 420 N excess force
MU       = 0.1    # stress decay rate (1/s)
R_STRESS = 3.0    # contagion radius (m)
# contagion normalised by observable speed range [V_MIN, ABS_V_MAX]
# so an agent at the speed cap always gives maximum contagion signal
_SPD_RANGE = max(ABS_V_MAX - V_MIN, 1e-6)

N    = 55    # FIS-sensitive density (validated in fis_analysis.py)
SEED = 6     # seed=6 gives clear FIS without permanent deadlock

FRAME_SKIP = 5

LAMBDA_VALS = [0.0, 2.0, 4.0, 6.0]
LABELS      = ['No contagion  (λ=0)', 'Weak  (λ=2)',
               'Moderate  (λ=4)', 'Strong  (λ=6)']
HDR_COLORS  = ['#88aaff', '#88ddaa', '#ffcc44', '#ff6666']


# ── forces ───────────────────────────────────────────────────────
def self_driving(pos, vel, v0_arr):
    diff = target - pos
    d    = np.linalg.norm(diff, axis=1, keepdims=True)
    e    = diff / np.maximum(d, 1e-6)
    return MASS * (v0_arr[:, None] * e - vel) / TAU


def agent_repulsion(pos):
    """Smooth SFM: exponential social repulsion only (k=0, κ=0)."""
    n = len(pos)
    if n < 2:
        return np.zeros((n, 2))
    diff  = pos[:, None, :] - pos[None, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, 1.0)
    dist  = np.sqrt(dist2)
    np.fill_diagonal(dist, 1.0)
    n_vec   = diff / dist[:, :, None]
    overlap = 2 * R_AGENT - dist
    soc_mag = A_SOC * np.exp(overlap / B_SOC)
    mask    = ~np.eye(n, dtype=bool)
    return np.einsum('ij,ijk->ik', soc_mag * mask, n_vec)


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


def stress_step(beta, pos, vel, lam):
    if lam == 0.0:
        return np.clip(beta - MU * beta * DT, 0.0, 1.0)
    spds   = np.linalg.norm(vel, axis=1)
    # normalise by observable speed range so agents at cap give signal = 1
    excess = np.maximum(spds - V_MIN, 0.0) / _SPD_RANGE
    diff   = pos[:, None, :] - pos[None, :, :]
    dist2  = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, np.inf)
    in_r   = np.sqrt(dist2) < R_STRESS
    n_nbr  = in_r.sum(axis=1)
    mean_exc = np.where(
        n_nbr > 0,
        (in_r * excess[None, :]).sum(axis=1) / np.maximum(n_nbr, 1),
        0.0
    )
    contagion = lam * mean_exc * (1.0 - beta)
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
    jitter   = min(0.12, spacing * 0.15)
    base    += rng.uniform(-jitter, jitter, base.shape)
    return base


# ── run one simulation ────────────────────────────────────────────
def run_sim(lam, seed=SEED, record_frames=True):
    pos  = initial_positions(N, seed=seed)
    vel  = np.zeros((N, 2))
    beta = np.zeros(N)

    seed_idx = int(np.argmin(np.linalg.norm(pos - np.array([4.0, 3.0]), axis=1)))
    beta[seed_idx] = 1.0

    fps        = ([pos.copy()], [beta.copy()], [0.0]) if record_frames else None
    evac_time  = T_END

    stuck_counter = np.zeros(N)
    rng_kick      = np.random.default_rng(seed + 42)

    t = 0.0
    for step in range(int(T_END / DT)):
        if len(pos) == 0:
            evac_time = t
            break

        v0_arr = V_MIN + beta * (V_MAX - V_MIN)
        F      = self_driving(pos, vel, v0_arr) + agent_repulsion(pos) + wall_forces(pos)
        vel   += F / MASS * DT

        spds = np.linalg.norm(vel, axis=1, keepdims=True)
        spds = np.maximum(spds, 1e-6)
        vel  = np.where(spds > ABS_V_MAX, vel / spds * ABS_V_MAX, vel)

        beta = stress_step(beta, pos, vel, lam)
        pos += vel * DT
        t   += DT

        exited = pos[:, 0] > ROOM_X[1] - 0.1
        pos   = pos[~exited];  vel  = vel[~exited]
        beta  = beta[~exited]; stuck_counter = stuck_counter[~exited]

        # end-of-evacuation rescue only (same rule as fis_analysis.py)
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

        if record_frames and step % FRAME_SKIP == 0:
            fps[0].append(pos.copy())
            fps[1].append(beta.copy())
            fps[2].append(t)

    if record_frames:
        return fps[0], fps[1], fps[2], evac_time
    return evac_time


# ── run all 4 ─────────────────────────────────────────────────────
print("Running 4 FIS + stress simulations…")
all_data = []
for lam, lab in zip(LAMBDA_VALS, LABELS):
    print(f"  {lab}…", end=' ', flush=True)
    fps, fbs, fts, evac_t = run_sim(lam)
    all_data.append((fps, fbs, fts, evac_t))
    print(f"evacuated in {evac_t:.1f} s")

# ── multi-seed sweep → bar chart ─────────────────────────────────
N_SEEDS = 15
print(f"\nMulti-seed sweep ({N_SEEDS} seeds per λ)…")
sweep_ets = []
for lam in LAMBDA_VALS:
    ets = np.array([run_sim(lam, seed=s, record_frames=False) for s in range(N_SEEDS)])
    sweep_ets.append(ets)
    dl = int(np.sum(ets >= T_END))
    print(f"  λ={lam}: mean={np.mean(ets):.1f}s  median={np.median(ets):.1f}s  "
          f"deadlocks={dl}/{N_SEEDS}")

sweep_ets       = [np.array(e) for e in sweep_ets]
sweep_means     = np.array([e.mean()                      for e in sweep_ets])
sweep_stds      = np.array([e.std()                       for e in sweep_ets])
sweep_medians   = np.array([np.median(e)                  for e in sweep_ets])
sweep_deadlocks = np.array([int(np.sum(e >= T_END))       for e in sweep_ets])

max_f = max(len(d[0]) for d in all_data)
for fps, fbs, fts, _ in all_data:
    while len(fps) < max_f:
        fps.append(np.empty((0, 2)))
        fbs.append(np.array([]))
        fts.append(fts[-1])


# ── figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle(
    'Faster-is-slower via stress contagion  —  smooth SFM (exponential repulsion only)\n'
    'colour = stress β:  blue → calm (v₀=1.5 m/s)   red → panic (v₀=6.5 m/s, capped at 3.5)',
    color='white', fontsize=12, fontweight='bold', y=0.98
)

cmap_beta = plt.cm.coolwarm
norm_beta = Normalize(vmin=0, vmax=1)


def setup_room(ax, label, hdr_color, evac_t):
    ax.set_facecolor('#16213e')
    ax.set_title(label, color=hdr_color, fontsize=11, fontweight='bold', pad=6)
    ax.add_patch(patches.Rectangle((0, 0), 10, 6, facecolor='#0f3460', edgecolor='none'))
    wc, lw = '#e0e0e0', 2.5
    ax.plot([0, 10], [0, 0], color=wc, lw=lw)
    ax.plot([0, 10], [6, 6], color=wc, lw=lw)
    ax.plot([0, 0],  [0, 6], color=wc, lw=lw)
    ax.plot([10, 10], [0, EXIT_Y[0]], color=wc, lw=lw)
    ax.plot([10, 10], [EXIT_Y[1], 6], color=wc, lw=lw)
    ax.add_patch(patches.Rectangle(
        (9.85, EXIT_Y[0]), 0.15, EXIT_Y[1] - EXIT_Y[0],
        facecolor='#00ff88', alpha=0.3
    ))
    ax.annotate('', xy=(11.2, 3.0), xytext=(10.1, 3.0),
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=1.5))
    ax.set_xlim(-0.5, 12.0);  ax.set_ylim(-0.5, 6.8)
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor('#444466')
    done_txt = ax.text(5, 3, f'Evacuated\nin {evac_t:.1f} s',
                       color='#00ff88', fontsize=13, fontweight='bold',
                       ha='center', va='center', alpha=0,
                       bbox=dict(facecolor='#0f3460', edgecolor='#00ff88',
                                 alpha=0.85, boxstyle='round,pad=0.4'))
    info_txt = ax.text(0.03, 0.97, '', transform=ax.transAxes,
                       color='white', fontsize=9, va='top')
    return done_txt, info_txt


scats, done_txts, info_txts = [], [], []
for ax, (fps, fbs, fts, evac_t), lab, col in zip(
        axes.ravel(), all_data, LABELS, HDR_COLORS):
    done_t, info_t = setup_room(ax, lab, col, evac_t)
    scat = ax.scatter([], [], c=[], s=140, cmap=cmap_beta, norm=norm_beta,
                      edgecolors='white', linewidths=0.3, zorder=5)
    scats.append(scat)
    done_txts.append(done_t)
    info_txts.append(info_t)

sm = ScalarMappable(cmap=cmap_beta, norm=norm_beta)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
cbar.set_label('stress  β', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(['0  calm', '0.5', '1  panic'])


def update(frame):
    artists = []
    for scat, done_t, info_t, (fps, fbs, fts, evac_t) in zip(
            scats, done_txts, info_txts, all_data):
        f      = min(frame, len(fps) - 1)
        pos_f  = fps[f]
        beta_f = fbs[f]
        t_f    = fts[f]

        if len(pos_f) > 0:
            scat.set_offsets(pos_f)
            scat.set_array(beta_f)
        else:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array(np.array([]))

        info_t.set_text(f't={t_f:.1f}s   {len(pos_f)}/{N}')
        if len(pos_f) == 0:
            done_t.set_alpha(1)

        artists += [scat, done_t, info_t]
    return artists


ani = animation.FuncAnimation(
    fig, update, frames=max_f,
    interval=25, blit=False, repeat=False
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# ── Figure 2: bar chart ───────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor('#1a1a2e')
ax2.set_facecolor('#16213e')

x = np.arange(len(LAMBDA_VALS))

bars = ax2.bar(x, sweep_means, color=HDR_COLORS, width=0.55,
               edgecolor='white', linewidth=0.8, zorder=3, alpha=0.85)

# ±1 std error bars
ax2.errorbar(x, sweep_means, yerr=sweep_stds,
             fmt='none', color='white', capsize=6, capthick=1.5,
             linewidth=1.5, zorder=4)

# median as a white dash so both statistics are visible
ax2.scatter(x, sweep_medians, color='white', marker='_',
            s=300, linewidths=2.5, zorder=5, label='median')

# deadlock count on each bar
for i, (bar, dl) in enumerate(zip(bars, sweep_deadlocks)):
    top = sweep_means[i] + sweep_stds[i] + 1.5
    if dl > 0:
        ax2.text(bar.get_x() + bar.get_width() / 2, top,
                 f'{dl}/{N_SEEDS} deadlocked',
                 color='#ff9999', fontsize=8.5, ha='center', va='bottom',
                 fontweight='bold')
    else:
        ax2.text(bar.get_x() + bar.get_width() / 2, top,
                 '0 deadlocked',
                 color='#99ff99', fontsize=8.5, ha='center', va='bottom')

# mark the fastest mean
opt_i = int(np.argmin(sweep_means))
ax2.annotate('fastest', xy=(x[opt_i], sweep_means[opt_i]),
             xytext=(x[opt_i], sweep_means[opt_i] - 6),
             color='#ffdd44', fontsize=9, ha='center', va='top',
             arrowprops=dict(arrowstyle='->', color='#ffdd44', lw=1.5))

ax2.axhline(T_END, color='#666688', lw=1.2, ls=':', alpha=0.7)
ax2.text(x[-1] + 0.32, T_END - 1.5, f'T_end={T_END:.0f}s (deadlock cap)',
         color='#888899', fontsize=8, ha='right', va='top')

ax2.legend(handles=[plt.Line2D([0], [0], color='white', lw=2.5, ls='--',
                                label='median')],
           framealpha=0.2, labelcolor='white', fontsize=9)

ax2.set_xticks(x)
ax2.set_xticklabels([f'λ={v}' for v in LAMBDA_VALS], color='white', fontsize=11)
ax2.set_ylabel('Evacuation time (s)', color='white', fontsize=11)
ax2.set_title(
    f'FIS via stress contagion  —  mean ± std  |  white dash = median  |  {N_SEEDS} seeds\n'
    f'(N={N}, ABS_V_MAX={ABS_V_MAX} m/s, V_max={V_MAX} m/s)',
    color='white', fontsize=11, fontweight='bold'
)
ax2.set_ylim(0, T_END + 14)
ax2.tick_params(colors='white')
for s in ax2.spines.values():
    s.set_edgecolor('#444466')
ax2.grid(True, axis='y', alpha=0.15, color='white')

plt.tight_layout()
plt.show()
