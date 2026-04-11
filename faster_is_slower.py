import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─── parameters ───────────────────────────────────────────────
N     = 50
mass  = 70.0
tau   = 0.5
dt    = 0.05
t_end = 25.0

ROOM_X = (0.0, 10.0)
ROOM_Y = (0.0, 6.0)
EXIT_Y = (2.8, 3.2)    # 0.4m exit — very narrow
target = np.array([10.5, 3.0])

# ─── forces ───────────────────────────────────────────────────
def self_driving(positions, velocities, v0):
    diff       = target - positions
    dists      = np.linalg.norm(diff, axis=1, keepdims=True)
    directions = diff / np.maximum(dists, 1e-6)
    return mass * (v0 * directions - velocities) / tau

def repulsion(positions, velocities):
    n_agents = len(positions)
    F = np.zeros((n_agents, 2))
    A, B, k, r = 2000.0, 0.08, 120000.0, 0.5
    kappa = 240000.0          # ← sliding friction coefficient

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            diff = positions[i] - positions[j]
            d    = np.linalg.norm(diff)
            if d < 1e-6:
                continue
            n_vec   = diff / d
            t_vec   = np.array([-n_vec[1], n_vec[0]])   # perpendicular
            overlap = r - d
            g       = max(overlap, 0.0)

            # relative tangential velocity
            delta_vt = np.dot(velocities[j] - velocities[i], t_vec)

            F[i] += (A * np.exp(overlap / B) + k * g) * n_vec
            F[i] += kappa * g * delta_vt * t_vec        # ← friction term

    return F

def wall_force_segment(positions, wall_point, wall_normal):
    n = len(positions)
    F = np.zeros((n, 2))
    AW, BW, k, r = 2000.0, 0.08, 120000.0, 0.25
    for i in range(n):
        d = np.dot(positions[i] - wall_point, wall_normal)
        d = max(d, 1e-6)
        overlap = r - d
        g = max(overlap, 0.0)
        F[i] += (AW * np.exp(overlap / BW) + k * g) * wall_normal
    return F

def all_wall_forces(positions):
    F = np.zeros((len(positions), 2))
    A, B, k, r = 2000.0, 0.08, 120000.0, 0.25
    F += wall_force_segment(positions,
                            np.array([ROOM_X[0], 0.0]), np.array([1.0, 0.0]))
    F += wall_force_segment(positions,
                            np.array([0.0, ROOM_Y[0]]), np.array([0.0, 1.0]))
    F += wall_force_segment(positions,
                            np.array([0.0, ROOM_Y[1]]), np.array([0.0, -1.0]))
    for i in range(len(positions)):
        if positions[i, 1] < EXIT_Y[0] or positions[i, 1] > EXIT_Y[1]:
            d = max(ROOM_X[1] - positions[i, 0], 1e-6)
            overlap = r - d
            g = max(overlap, 0.0)
            F[i] += (A * np.exp(overlap / B) + k * g) * np.array([-1.0, 0.0])
    return F

# ─── run one simulation ───────────────────────────────────────
def run(v0):
    cols = 5
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.5, 6.0, cols)
    ys = np.linspace(0.5, 5.5, rows)
    gx, gy = np.meshgrid(xs, ys)
    pos = np.column_stack([gx.ravel(), gy.ravel()])[:N]
    vel = np.zeros((N, 2))

    pos_frames = [pos.copy()]
    vel_frames = [vel.copy()]

    t = 0.0
    evac_time = t_end
    while t < t_end:
        if len(pos) == 0:
            evac_time = t
            break
        f = self_driving(pos, vel, v0) + repulsion(pos, vel) + all_wall_forces(pos)
        vel += f / mass * dt
        spds = np.linalg.norm(vel, axis=1, keepdims=True)
        spds = np.maximum(spds, 1e-6)
        vel  = np.where(spds > v0 * 2, vel / spds * v0 * 2, vel)
        pos += vel * dt
        exited = np.linalg.norm(pos - target, axis=1) < 1.0
        pos = pos[~exited]
        vel = vel[~exited]
        pos_frames.append(pos.copy())
        vel_frames.append(vel.copy())
        t += dt

    return pos_frames, vel_frames, evac_time

# ─── run both ─────────────────────────────────────────────────

print("Running normal (v0=1.5)...")
frames_n, vel_n, t_normal = run(v0=1.5)
print(f"  evacuated in {t_normal:.1f}s")

print("Running panic  (v0=3.5)...")
frames_p, vel_p, t_panic = run(v0=3.5)
print(f"  evacuated in {t_panic:.1f}s")

# pad shorter simulation with empty frames
max_frames = max(len(frames_n), len(frames_p))
while len(frames_n) < max_frames:
    frames_n.append(np.empty((0, 2)))
    vel_n.append(np.empty((0, 2)))
while len(frames_p) < max_frames:
    frames_p.append(np.empty((0, 2)))
    vel_p.append(np.empty((0, 2)))

# ─── colormap ─────────────────────────────────────────────────
all_spds = []
for v in vel_n + vel_p:
    if len(v) > 0:
        all_spds.extend(np.linalg.norm(v, axis=1).tolist())
all_spds = np.array(all_spds)
cmap = plt.cm.coolwarm
norm = Normalize(vmin=np.percentile(all_spds, 5),
                 vmax=np.percentile(all_spds, 95))

# ─── figure ───────────────────────────────────────────────────
fig, (ax_n, ax_p) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Normal walking  vs  Panic — does speed help or hurt?',
             color='white', fontsize=14, fontweight='bold', y=0.97)

def setup_room(ax, title, color):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color=color, fontsize=12, fontweight='bold', pad=10)

    # room floor
    ax.add_patch(patches.Rectangle((0, 0), 10, 6,
                                   facecolor='#0f3460', edgecolor='none'))
    # walls
    wc, lw = '#e0e0e0', 3
    ax.plot([0, 10], [0, 0], color=wc, lw=lw)
    ax.plot([0, 10], [6, 6], color=wc, lw=lw)
    ax.plot([0, 0],  [0, 6], color=wc, lw=lw)
    ax.plot([10, 10], [0, EXIT_Y[0]], color=wc, lw=lw)
    ax.plot([10, 10], [EXIT_Y[1], 6], color=wc, lw=lw)

    # exit
    ax.add_patch(patches.Rectangle((9.85, EXIT_Y[0]), 0.15,
                                   EXIT_Y[1] - EXIT_Y[0],
                                   facecolor='#00ff88', alpha=0.3))
    ax.annotate('', xy=(11.0, 3.0), xytext=(10.1, 3.0),
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2))
    ax.text(11.1, 3.0, 'EXIT', color='#00ff88', fontsize=10,
            fontweight='bold', va='center')

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 7.0)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', color='white', fontsize=10)
    ax.set_ylabel('y (m)', color='white', fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

setup_room(ax_n, f'Normal  v⁰ = 1.5 m/s', '#88ccff')
setup_room(ax_p, f'Panic    v⁰ = 4.5 m/s', '#ff8888')

# scatter plots
scat_n = ax_n.scatter([], [], c=[], s=180, cmap=cmap, norm=norm,
                      edgecolors='white', linewidths=0.4, zorder=5)
scat_p = ax_p.scatter([], [], c=[], s=180, cmap=cmap, norm=norm,
                      edgecolors='white', linewidths=0.4, zorder=5)

# colorbar on right panel only
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_p, fraction=0.03, pad=0.02)
cbar.set_label('speed (m/s)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# countdown text in each panel
time_n = ax_n.text(0.05, 0.95, '', transform=ax_n.transAxes,
                   color='white', fontsize=11, va='top')
time_p = ax_p.text(0.05, 0.95, '', transform=ax_p.transAxes,
                   color='white', fontsize=11, va='top')

# evacuation time banner (shown when done)
done_n = ax_n.text(5, 3, '', color='#00ff88', fontsize=14,
                   fontweight='bold', ha='center', va='center', alpha=0)
done_p = ax_p.text(5, 3, '', color='#00ff88', fontsize=14,
                   fontweight='bold', ha='center', va='center', alpha=0)

# ─── update ───────────────────────────────────────────────────
def update(frame):
    t = frame * dt

    for scat, frames, vels, txt, done, evac_t in [
        (scat_n, frames_n, vel_n, time_n, done_n, t_normal),
        (scat_p, frames_p, vel_p, time_p, done_p, t_panic)
    ]:
        pos = frames[frame]
        vel = vels[frame]

        if len(pos) > 0:
            spds = np.linalg.norm(vel, axis=1)
            scat.set_offsets(pos)
            scat.set_array(spds)
        else:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array(np.array([]))

        remaining = len(pos)
        txt.set_text(f't = {t:.1f}s\nagents: {remaining}/{N}')

        # show evacuation time once everyone is out
        if remaining == 0:
            done.set_text(f'✓ evacuated\nin {evac_t:.1f}s')
            done.set_alpha(1)

    return scat_n, scat_p, time_n, time_p, done_n, done_p

ani = animation.FuncAnimation(
    fig,
    update,
    frames=max_frames,
    interval=40,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()