"""
Stress contagion in a crowd — smooth SFM with per-agent stress β_i.

One agent starts panicked (β=1); nearby agents who observe fast movement
become stressed, speed up, and propagate panic outward through the crowd.

Force model: smooth SFM (k=0, κ=0 — exponential social repulsion only,
no body-contact spring, no tangential friction).

Stress dynamics (Euler, per step):
  β̇_i = −µ·β_i  +  λ · mean_excess_speed(neighbours within r_s) · (1−β_i)
  excess_speed_j  = max(speed_j − V_MIN, 0) / (V_MAX − V_MIN)
  v₀_i = V_MIN + β_i · (V_MAX − V_MIN)

Calm walking (speed = V_MIN) causes zero contagion.
Only speeds above V_MIN propagate stress.
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
EXIT_Y = (2.55, 3.45)   # 0.9 m door
target = np.array([10.5, 3.0])

# ── physics (smooth SFM: k=0, κ=0) ──────────────────────────────
MASS    = 70.0
TAU     = 0.5
DT      = 0.01
T_END   = 80.0
R_AGENT = 0.3
R_WALL  = 0.3
A_SOC   = 2000.0
B_SOC   = 0.08
ABS_V_MAX = 5.0   # safety ceiling — rarely triggers

# ── stress model ─────────────────────────────────────────────────
V_MIN    = 1.5    # desired speed when fully calm (m/s)
V_MAX    = 4.5    # desired speed when fully panicked (m/s)
MU       = 0.1    # stress decay rate (1/s) — slow decay so stress persists ~10 s
LAMBDA   = 4.0    # contagion strength — high enough for self-sustaining cascade
R_STRESS = 3.0    # observation radius (m)

N    = 45
SEED = 0


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
        mag = A_SOC * np.exp((R_WALL - d) / B_SOC)   # pure exponential, no K_BODY
        return mag[:, None] * normal

    F += push(pos[:, 0] - ROOM_X[0], np.array([ 1.0,  0.0]))
    F += push(pos[:, 1] - ROOM_Y[0], np.array([ 0.0,  1.0]))
    F += push(ROOM_Y[1] - pos[:, 1], np.array([ 0.0, -1.0]))
    not_in_exit = (pos[:, 1] < EXIT_Y[0]) | (pos[:, 1] > EXIT_Y[1])
    if np.any(not_in_exit):
        sub = pos[not_in_exit]
        F[not_in_exit] += push(ROOM_X[1] - sub[:, 0], np.array([-1.0, 0.0]))
    return F


def stress_step(beta, pos, vel):
    """Euler update of β_i driven by excess speed of neighbours."""
    spds   = np.linalg.norm(vel, axis=1)
    excess = np.maximum(spds - V_MIN, 0.0) / (V_MAX - V_MIN)   # 0→1

    diff  = pos[:, None, :] - pos[None, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, np.inf)
    dist  = np.sqrt(dist2)
    in_r  = dist < R_STRESS                         # (n, n) — j in radius of i
    n_nbr = in_r.sum(axis=1)

    # mean excess speed of neighbours within radius
    mean_exc = np.where(
        n_nbr > 0,
        (in_r * excess[None, :]).sum(axis=1) / np.maximum(n_nbr, 1),
        0.0
    )
    contagion = LAMBDA * mean_exc * (1.0 - beta)
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


# ── simulation ────────────────────────────────────────────────────
print("Running stress contagion simulation…")
pos  = initial_positions(N, seed=SEED)
vel  = np.zeros((N, 2))
beta = np.zeros(N)

# seed: agent closest to crowd centre — panic radiates outward in all directions
seed_pos = np.array([4.0, 3.0])
seed_idx = int(np.argmin(np.linalg.norm(pos - seed_pos, axis=1)))
beta[seed_idx] = 1.0
print(f"  Seed agent #{seed_idx} at {pos[seed_idx].round(2)} — starting at full panic (β=1)")

FRAME_SKIP   = 2
frames_pos   = [pos.copy()]
frames_vel   = [vel.copy()]
frames_beta  = [beta.copy()]
frame_times  = [0.0]

t = 0.0
for step in range(int(T_END / DT)):
    if len(pos) == 0:
        break

    v0_arr = V_MIN + beta * (V_MAX - V_MIN)
    F      = self_driving(pos, vel, v0_arr) + agent_repulsion(pos) + wall_forces(pos)
    vel   += F / MASS * DT
    spds   = np.linalg.norm(vel, axis=1, keepdims=True)
    vel    = np.where(spds > ABS_V_MAX, vel / spds * ABS_V_MAX, vel)
    beta   = stress_step(beta, pos, vel)
    pos   += vel * DT
    t     += DT

    exited = pos[:, 0] > ROOM_X[1] - 0.1
    pos  = pos[~exited]
    vel  = vel[~exited]
    beta = beta[~exited]

    if step % FRAME_SKIP == 0:
        frames_pos.append(pos.copy())
        frames_vel.append(vel.copy())
        frames_beta.append(beta.copy())
        frame_times.append(t)

mean_beta_ts = [np.mean(b) if len(b) > 0 else 0.0 for b in frames_beta]
print(f"  Done — {len(frames_pos)} frames, peak mean β = {max(mean_beta_ts):.2f}")


# ── figure ────────────────────────────────────────────────────────
fig, (ax_room, ax_ts) = plt.subplots(
    1, 2, figsize=(15, 6.5),
    gridspec_kw={'width_ratios': [2.2, 1]}
)
fig.patch.set_facecolor('#1a1a2e')

# room panel
ax_room.set_facecolor('#16213e')
ax_room.set_title(
    f'Stress contagion — smooth SFM   (N={N},  λ={LAMBDA},  µ={MU},  r_s={R_STRESS} m)\n'
    f'colour = stress β:  blue → calm (v₀={V_MIN} m/s)   red → panic (v₀={V_MAX} m/s)',
    color='white', fontsize=10, fontweight='bold', pad=8
)
ax_room.add_patch(patches.Rectangle((0, 0), 10, 6, facecolor='#0f3460', edgecolor='none'))
wc, lw = '#e0e0e0', 3
ax_room.plot([0, 10], [0, 0], color=wc, lw=lw)
ax_room.plot([0, 10], [6, 6], color=wc, lw=lw)
ax_room.plot([0, 0],  [0, 6], color=wc, lw=lw)
ax_room.plot([10, 10], [0, EXIT_Y[0]], color=wc, lw=lw)
ax_room.plot([10, 10], [EXIT_Y[1], 6], color=wc, lw=lw)
ax_room.add_patch(patches.Rectangle(
    (9.85, EXIT_Y[0]), 0.15, EXIT_Y[1] - EXIT_Y[0],
    facecolor='#00ff88', alpha=0.3
))
ax_room.annotate('', xy=(11.0, 3.0), xytext=(10.1, 3.0),
                 arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2))
ax_room.text(11.2, 3.0, 'EXIT', color='#00ff88',
             fontsize=10, fontweight='bold', va='center')
ax_room.set_xlim(-0.5, 12.5)
ax_room.set_ylim(-0.5, 7.0)
ax_room.set_aspect('equal')
ax_room.set_xlabel('x (m)', color='white')
ax_room.set_ylabel('y (m)', color='white')
ax_room.tick_params(colors='white')
for s in ax_room.spines.values():
    s.set_edgecolor('#444466')

cmap_beta = plt.cm.coolwarm          # blue = calm, red = panic
norm_beta = Normalize(vmin=0, vmax=1)
scat = ax_room.scatter([], [], c=[], s=180, cmap=cmap_beta, norm=norm_beta,
                       edgecolors='white', linewidths=0.4, zorder=5)
sm = ScalarMappable(cmap=cmap_beta, norm=norm_beta)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_room, fraction=0.025, pad=0.01)
cbar.set_label('stress  β', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(['0 (calm)', '0.25', '0.5', '0.75', '1 (panic)'])

txt_info = ax_room.text(0.03, 0.97, '', transform=ax_room.transAxes,
                        color='white', fontsize=10, va='top')

# time series panel
ax_ts.set_facecolor('#16213e')
ax_ts.set_title('Mean crowd stress  β̄(t)', color='white', fontsize=11, fontweight='bold')
ax_ts.set_xlabel('time (s)', color='white', fontsize=10)
ax_ts.set_ylabel('mean β', color='white', fontsize=10)
ax_ts.set_xlim(0, T_END)
ax_ts.set_ylim(0, 1)
ax_ts.tick_params(colors='white')
for s in ax_ts.spines.values():
    s.set_edgecolor('#444466')
ax_ts.grid(True, alpha=0.15, color='white')
ax_ts.axhline(0.5, color='#ff8844', lw=1, ls='--', alpha=0.5)
ax_ts.text(T_END * 0.02, 0.52, 'half-panic', color='#ff8844', fontsize=8)

ts_line, = ax_ts.plot([], [], color='#ff6666', lw=2, label='mean β')
ts_dot   = ax_ts.scatter([], [], color='white', s=30, zorder=5)


def update(frame):
    pos_f  = frames_pos[frame]
    beta_f = frames_beta[frame]
    t_f    = frame_times[frame]
    mb     = mean_beta_ts[frame]

    if len(pos_f) > 0:
        scat.set_offsets(pos_f)
        scat.set_array(beta_f)
    else:
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))

    txt_info.set_text(
        f't = {t_f:.1f} s\n'
        f'agents: {len(pos_f)}/{N}\n'
        f'mean β = {mb:.2f}'
    )
    ts_line.set_data(frame_times[:frame + 1], mean_beta_ts[:frame + 1])
    if frame > 0:
        ts_dot.set_offsets([[t_f, mb]])

    return scat, txt_info, ts_line, ts_dot


ani = animation.FuncAnimation(
    fig, update, frames=len(frames_pos),
    interval=30, blit=False, repeat=False
)
plt.tight_layout()
plt.show()
