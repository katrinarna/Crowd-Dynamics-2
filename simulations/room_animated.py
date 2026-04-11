import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─── parameters ───────────────────────────────────────────────
N     = 40
mass  = 70.0
v0    = 1.5
tau   = 0.5
dt    = 0.05
t_end = 30.0

ROOM_X = (0.0, 10.0)
ROOM_Y = (0.0, 6.0)
EXIT_Y = (2.5, 3.5)
target = np.array([10.5, 3.0])

# ─── starting positions ───────────────────────────────────────
cols = 5
rows = int(np.ceil(N / cols))
xs = np.linspace(0.5, 6.0, cols)
ys = np.linspace(0.5, 5.5, rows)
grid_x, grid_y = np.meshgrid(xs, ys)
positions  = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
velocities = np.zeros((N, 2))

# ─── forces ───────────────────────────────────────────────────
def self_driving(positions, velocities):
    diff       = target - positions
    dists      = np.linalg.norm(diff, axis=1, keepdims=True)
    directions = diff / np.maximum(dists, 1e-6)
    return mass * (v0 * directions - velocities) / tau

def repulsion(positions):
    n_agents = len(positions)
    F = np.zeros((n_agents, 2))
    A, B, k, r = 2000.0, 0.08, 120000.0, 0.5
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            diff = positions[i] - positions[j]
            d    = np.linalg.norm(diff)
            if d < 1e-6:
                continue
            n_vec   = diff / d
            overlap = r - d
            g       = max(overlap, 0.0)
            F[i]   += (A * np.exp(overlap / B) + k * g) * n_vec
    return F

def wall_force_segment(positions, wall_point, wall_normal):
    n_agents = len(positions)
    F = np.zeros((n_agents, 2))
    AW, BW, k, r = 2000.0, 0.08, 120000.0, 0.25
    for i in range(n_agents):
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

# ─── run simulation ───────────────────────────────────────────
print("Running simulation...")
frames     = [positions.copy()]
vel_frames = [velocities.copy()]

t = 0.0
while t < t_end:
    if len(positions) == 0:
        break

    f = (self_driving(positions, velocities)
       + repulsion(positions)
       + all_wall_forces(positions))

    velocities += f / mass * dt
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds = np.maximum(speeds, 1e-6)
    velocities = np.where(speeds > 3.0, velocities / speeds * 3.0, velocities)
    positions  += velocities * dt

    exited     = np.linalg.norm(positions - target, axis=1) < 1.0
    positions  = positions[~exited]
    velocities = velocities[~exited]

    frames.append(positions.copy())
    vel_frames.append(velocities.copy())
    t += dt

print(f"Done — {len(frames)} frames")

# ─── figure setup ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('#1a1a2e')

# main simulation panel
ax = fig.add_axes([0.05, 0.1, 0.55, 0.82])
ax.set_facecolor('#16213e')

# speed over time panel
ax2 = fig.add_axes([0.67, 0.55, 0.30, 0.35])
ax2.set_facecolor('#16213e')
ax2.set_title('avg speed over time', color='white', fontsize=10)
ax2.set_xlabel('time (s)', color='white', fontsize=9)
ax2.set_ylabel('avg speed (m/s)', color='white', fontsize=9)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444466')

# agents remaining panel
ax3 = fig.add_axes([0.67, 0.10, 0.30, 0.35])
ax3.set_facecolor('#16213e')
ax3.set_title('agents remaining', color='white', fontsize=10)
ax3.set_xlabel('time (s)', color='white', fontsize=9)
ax3.set_ylabel('count', color='white', fontsize=9)
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_edgecolor('#444466')

# ─── draw static room elements ────────────────────────────────
# room floor
room_bg = patches.Rectangle((0, 0), 10, 6,
                             facecolor='#0f3460', edgecolor='none')
ax.add_patch(room_bg)

# walls (thick lines)
wall_color = '#e0e0e0'
lw = 3
# bottom wall
ax.plot([0, 10], [0, 0], color=wall_color, lw=lw)
# top wall
ax.plot([0, 10], [6, 6], color=wall_color, lw=lw)
# left wall
ax.plot([0, 0], [0, 6], color=wall_color, lw=lw)
# right wall bottom segment
ax.plot([10, 10], [0, EXIT_Y[0]], color=wall_color, lw=lw)
# right wall top segment
ax.plot([10, 10], [EXIT_Y[1], 6], color=wall_color, lw=lw)

# exit arrow
ax.annotate('', xy=(11.2, 3.0), xytext=(10.2, 3.0),
            arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2))
ax.text(11.3, 3.0, 'EXIT', color='#00ff88', fontsize=11,
        fontweight='bold', va='center')

# exit gap highlight
exit_glow = patches.Rectangle((9.85, EXIT_Y[0]),
                               0.15, EXIT_Y[1] - EXIT_Y[0],
                               facecolor='#00ff88', alpha=0.3)
ax.add_patch(exit_glow)

ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.5, 7.0)
ax.set_aspect('equal')
ax.set_xlabel('x (m)', color='white', fontsize=11)
ax.set_ylabel('y (m)', color='white', fontsize=11)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444466')

# title
title_text = ax.set_title('', color='white', fontsize=13, fontweight='bold')



# ─── colormap setup ───────────────────────────────────────────
all_speeds_flat = []
for vel in vel_frames:
    if len(vel) > 0:
        all_speeds_flat.extend(np.linalg.norm(vel, axis=1).tolist())

all_speeds_flat = np.array(all_speeds_flat)
cmap     = plt.cm.coolwarm
norm     = Normalize(vmin=np.percentile(all_speeds_flat, 5),
                     vmax=np.percentile(all_speeds_flat, 95))
max_speed = np.percentile(all_speeds_flat, 95)

print(f"speed range: min={all_speeds_flat.min():.3f}  max={all_speeds_flat.max():.3f}")
print(f"5th pct={np.percentile(all_speeds_flat,5):.3f}  95th pct={np.percentile(all_speeds_flat,95):.3f}")
print(f"sample speeds frame 50: {np.linalg.norm(vel_frames[50], axis=1)}")

# agent scatter (colored by speed)
scat = ax.scatter([], [], c=[], s=180, zorder=5, cmap=cmap, norm=norm,
                  edgecolors='white', linewidths=0.4)

# colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label(f'speed (m/s)  max={max_speed:.1f}', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# ─── precompute stats ─────────────────────────────────────────

# ─── precompute stats ─────────────────────────────────────────
time_axis    = np.arange(len(frames)) * dt
avg_speeds   = []
agent_counts = []

for i, (pos, vel) in enumerate(zip(frames, vel_frames)):
    if len(vel) > 0:
        avg_speeds.append(np.mean(np.linalg.norm(vel, axis=1)))
    else:
        avg_speeds.append(0.0)
    agent_counts.append(len(pos))

# plot full lines in background (faint)
ax2.plot(time_axis, avg_speeds, color='#553377', lw=1, alpha=0.3)
ax3.plot(time_axis, agent_counts, color='#335577', lw=1, alpha=0.3)

# live lines that grow each frame
speed_line, = ax2.plot([], [], color='#cc88ff', lw=1.5)
count_line, = ax3.plot([], [], color='#88ccff', lw=1.5)

ax2.set_xlim(0, time_axis[-1])
ax2.set_ylim(0, max(avg_speeds) * 1.2 + 0.1)
ax3.set_xlim(0, time_axis[-1])
ax3.set_ylim(0, N * 1.1)

def update(frame):
    pos = frames[frame]
    vel = vel_frames[frame]
    t   = frame * dt

    # clear and redraw scatter each frame — fixes color update
    scat.set_offsets(np.empty((0, 2)))

    if len(pos) > 0:
        spds = np.linalg.norm(vel, axis=1)
        scat.set_offsets(pos)
        scat.set_array(spds)

    # update live graphs
    speed_line.set_data(time_axis[:frame+1], avg_speeds[:frame+1])
    count_line.set_data(time_axis[:frame+1], agent_counts[:frame+1])

    # title
    title_text.set_text(
        f'Social Force Model — t = {t:.1f}s    '
        f'agents remaining: {len(pos)}/{N}'
    )

    return scat, speed_line, count_line, title_text

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=40,
    blit=False,       # ← changed from True to False
    repeat=False
)

plt.show()