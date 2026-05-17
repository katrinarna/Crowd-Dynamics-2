"""
Stress contagion comparison — 4 panels with varying contagion strength λ.

Same crowd, same panicked seed agent, same physics.
Only λ (how strongly fast movement stresses nearby agents) changes.

λ=0  → seed panics alone, everyone else stays calm
λ=1.5 → weak cascade, partial spread
λ=3.0 → moderate cascade
λ=4.0 → full cascade (current stress_contagion_sfm.py setting)
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
EXIT_Y = (2.55, 3.45)
target = np.array([10.5, 3.0])

# ── physics (smooth SFM) ─────────────────────────────────────────
MASS      = 70.0
TAU       = 0.5
DT        = 0.01
T_END     = 60.0
R_AGENT   = 0.3
R_WALL    = 0.3
A_SOC     = 2000.0
B_SOC     = 0.08
ABS_V_MAX = 5.0

# ── stress model ─────────────────────────────────────────────────
V_MIN    = 1.5
V_MAX    = 4.5
MU       = 0.1
R_STRESS = 3.0

N          = 45
SEED       = 0
FRAME_SKIP = 5   # record every 5th step → manageable frame count

LAMBDA_VALS = [0.0, 1.5, 3.0, 4.0]
LABELS      = ['No contagion  (λ=0)', 'Weak  (λ=1.5)',
               'Moderate  (λ=3)', 'Strong  (λ=4)']
HDR_COLORS  = ['#88aaff', '#88ddaa', '#ffcc44', '#ff6666']


# ── forces ───────────────────────────────────────────────────────
def self_driving(pos, vel, v0_arr):
    diff = target - pos
    d    = np.linalg.norm(diff, axis=1, keepdims=True)
    e    = diff / np.maximum(d, 1e-6)
    return MASS * (v0_arr[:, None] * e - vel) / TAU


def agent_repulsion(pos):
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
        return np.clip(beta - MU * beta * DT, 0.0, 1.0)   # pure decay, no contagion
    spds   = np.linalg.norm(vel, axis=1)
    excess = np.maximum(spds - V_MIN, 0.0) / (V_MAX - V_MIN)
    diff   = pos[:, None, :] - pos[None, :, :]
    dist2  = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, np.inf)
    dist   = np.sqrt(dist2)
    in_r   = dist < R_STRESS
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
def run_sim(lam):
    pos  = initial_positions(N, seed=SEED)
    vel  = np.zeros((N, 2))
    beta = np.zeros(N)

    seed_idx = int(np.argmin(np.linalg.norm(pos - np.array([4.0, 3.0]), axis=1)))
    beta[seed_idx] = 1.0

    fps   = [pos.copy()]
    fbs   = [beta.copy()]
    fts   = [0.0]
    evac_time = T_END

    t = 0.0
    for step in range(int(T_END / DT)):
        if len(pos) == 0:
            evac_time = t
            break
        v0_arr = V_MIN + beta * (V_MAX - V_MIN)
        F      = self_driving(pos, vel, v0_arr) + agent_repulsion(pos) + wall_forces(pos)
        vel   += F / MASS * DT
        spds   = np.linalg.norm(vel, axis=1, keepdims=True)
        vel    = np.where(spds > ABS_V_MAX, vel / spds * ABS_V_MAX, vel)
        beta   = stress_step(beta, pos, vel, lam)
        pos   += vel * DT
        t     += DT
        exited = pos[:, 0] > ROOM_X[1] - 0.1
        pos  = pos[~exited];  vel  = vel[~exited];  beta = beta[~exited]
        if step % FRAME_SKIP == 0:
            fps.append(pos.copy())
            fbs.append(beta.copy())
            fts.append(t)

    return fps, fbs, fts, evac_time


# ── run all 4 ─────────────────────────────────────────────────────
print("Running 4 simulations…")
all_data = []
for lam, lab in zip(LAMBDA_VALS, LABELS):
    print(f"  {lab}…", end=' ', flush=True)
    fps, fbs, fts, evac_t = run_sim(lam)
    all_data.append((fps, fbs, fts, evac_t))
    print(f"evacuated in {evac_t:.1f} s")

# pad all to same frame count
max_f = max(len(d[0]) for d in all_data)
for fps, fbs, fts, _ in all_data:
    while len(fps) < max_f:
        fps.append(np.empty((0, 2)))
        fbs.append(np.array([]))
        fts.append(fts[-1])


# ── figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Stress contagion — how strongly panic spreads changes evacuation dynamics\n'
             'colour = stress β:  blue → calm   red → panicked',
             color='white', fontsize=13, fontweight='bold', y=0.98)

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
    # pre-built evacuation banner (hidden until complete)
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

# shared colorbar
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

        remaining = len(pos_f)
        info_t.set_text(f't={t_f:.1f}s   {remaining}/{N}')

        if remaining == 0:
            done_t.set_alpha(1)

        artists += [scat, done_t, info_t]
    return artists


ani = animation.FuncAnimation(
    fig, update, frames=max_f,
    interval=25, blit=False, repeat=False
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
