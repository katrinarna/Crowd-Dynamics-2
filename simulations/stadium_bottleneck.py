import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─── parameters ───────────────────────────────────────────────
N     = 60
mass  = 70.0
v0    = 1.5
tau   = 0.5
dt    = 0.05
t_end = 60.0

# ─── venue geometry ───────────────────────────────────────────
# seating area
SEAT_X  = (0.0, 10.0)
SEAT_Y  = (3.0, 11.0)   # 10m x 8m

# corridor (centered on seating area)
COR_W   = 2.0            # corridor width — we will vary this
COR_X   = (5.0 - COR_W/2, 5.0 + COR_W/2)
COR_Y   = (0.0, 3.0)     # 3m long corridor below seating area

# exit door at bottom of corridor
EXIT_W  = 1.0
EXIT_X  = (5.0 - EXIT_W/2, 5.0 + EXIT_W/2)
target  = np.array([5.0, -0.5])   # just below exit

# ─── starting positions: random in seating area ───────────────
np.random.seed(42)
positions = np.column_stack([
    np.random.uniform(SEAT_X[0] + 0.5, SEAT_X[1] - 0.5, N),
    np.random.uniform(SEAT_Y[0] + 0.5, SEAT_Y[1] - 0.5, N)
])
velocities = np.zeros((N, 2))

# ─── forces ───────────────────────────────────────────────────
def self_driving(positions, velocities):
    diff       = target - positions
    dists      = np.linalg.norm(diff, axis=1, keepdims=True)
    directions = diff / np.maximum(dists, 1e-6)
    return mass * (v0 * directions - velocities) / tau

def repulsion(positions, velocities):
    n = len(positions)
    F = np.zeros((n, 2))
    A, B, k, r = 2000.0, 0.08, 120000.0, 0.5
    kappa = 240000.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = positions[i] - positions[j]
            d    = np.linalg.norm(diff)
            if d < 1e-6:
                continue
            n_vec    = diff / d
            t_vec    = np.array([-n_vec[1], n_vec[0]])
            overlap  = r - d
            g        = max(overlap, 0.0)
            delta_vt = np.dot(velocities[j] - velocities[i], t_vec)
            F[i] += (A * np.exp(overlap / B) + k * g) * n_vec
            F[i] += kappa * g * delta_vt * t_vec
    return F

def push_from_wall(pos, d, normal, A=2000., B=0.08, k=120000., r=0.25):
    d    = max(d, 1e-6)
    ov   = r - d
    g    = max(ov, 0.0)
    return (A * np.exp(ov / B) + k * g) * normal

def all_wall_forces(positions):
    n = len(positions)
    F = np.zeros((n, 2))

    for i in range(n):
        x, y = positions[i]

        # ── figure out which zone agent is in ──
        in_corridor = (COR_X[0] <= x <= COR_X[1]) and (COR_Y[0] <= y <= COR_Y[1])
        in_seating  = (SEAT_X[0] <= x <= SEAT_X[1]) and (SEAT_Y[0] <= y <= SEAT_Y[1])

        if in_seating:
            # seating area walls
            F[i] += push_from_wall(positions[i], x - SEAT_X[0],  np.array([ 1., 0.]))
            F[i] += push_from_wall(positions[i], SEAT_X[1] - x,  np.array([-1., 0.]))
            F[i] += push_from_wall(positions[i], SEAT_Y[1] - y,  np.array([ 0.,-1.]))
            # bottom wall of seating — but has a gap where corridor entrance is
            if x < COR_X[0]:
                F[i] += push_from_wall(positions[i], y - SEAT_Y[0], np.array([0., 1.]))
            elif x > COR_X[1]:
                F[i] += push_from_wall(positions[i], y - SEAT_Y[0], np.array([0., 1.]))
            # else: agent is above corridor entrance — no bottom wall force

        elif in_corridor:
            # corridor side walls
            F[i] += push_from_wall(positions[i], x - COR_X[0],  np.array([ 1., 0.]))
            F[i] += push_from_wall(positions[i], COR_X[1] - x,  np.array([-1., 0.]))
            # corridor bottom — has exit gap
            if x < EXIT_X[0] or x > EXIT_X[1]:
                F[i] += push_from_wall(positions[i], y - COR_Y[0], np.array([0., 1.]))

        else:
            # agent is in a corner or transition zone —
            # push gently toward corridor entrance
            if y < SEAT_Y[0] and y > COR_Y[1]:
                F[i] += push_from_wall(positions[i], y - COR_Y[1], np.array([0., 1.]))
            # left wall of overall venue
            if x < COR_X[0]:
                F[i] += push_from_wall(positions[i], x - SEAT_X[0], np.array([1., 0.]))
            if x > COR_X[1]:
                F[i] += push_from_wall(positions[i], SEAT_X[1] - x, np.array([-1., 0.]))

    return F

# ─── run simulation ───────────────────────────────────────────
print("Running simulation...")
frames     = [positions.copy()]
vel_frames = [velocities.copy()]

t = 0.0
while t < t_end:
    if len(positions) == 0:
        print(f"All evacuated at t={t:.1f}s")
        break

    f = (self_driving(positions, velocities)
       + repulsion(positions, velocities)
       + all_wall_forces(positions))

    velocities += f / mass * dt
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds = np.maximum(speeds, 1e-6)
    velocities = np.where(speeds > 3.0, velocities / speeds * 3.0, velocities)
    positions  += velocities * dt

    exited     = np.linalg.norm(positions - target, axis=1) < 0.6
    positions  = positions[~exited]
    velocities = velocities[~exited]

    frames.append(positions.copy())
    vel_frames.append(velocities.copy())
    t += dt

print(f"Done — {len(frames)} frames")

# ─── colormap ─────────────────────────────────────────────────
all_spds = []
for v in vel_frames:
    if len(v) > 0:
        all_spds.extend(np.linalg.norm(v, axis=1).tolist())
all_spds = np.array(all_spds)
cmap = plt.cm.coolwarm
norm = Normalize(vmin=np.percentile(all_spds, 5),
                 vmax=np.percentile(all_spds, 95))

# ─── figure ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 12))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')
ax.set_title('Stadium evacuation — two-stage bottleneck',
             color='white', fontsize=13, fontweight='bold')

# ── draw venue ────────────────────────────────────────────────
wc, lw = '#e0e0e0', 2.5

# seating area fill
ax.add_patch(patches.Rectangle(
    (SEAT_X[0], SEAT_Y[0]), SEAT_X[1]-SEAT_X[0], SEAT_Y[1]-SEAT_Y[0],
    facecolor='#0f3460', edgecolor='none'))

# corridor fill
ax.add_patch(patches.Rectangle(
    (COR_X[0], COR_Y[0]), COR_X[1]-COR_X[0], COR_Y[1]-COR_Y[0],
    facecolor='#0a2040', edgecolor='none'))

# seating walls
ax.plot([SEAT_X[0], SEAT_X[1]], [SEAT_Y[1], SEAT_Y[1]], color=wc, lw=lw)  # top
ax.plot([SEAT_X[0], SEAT_X[0]], [SEAT_Y[0], SEAT_Y[1]], color=wc, lw=lw)  # left
ax.plot([SEAT_X[1], SEAT_X[1]], [SEAT_Y[0], SEAT_Y[1]], color=wc, lw=lw)  # right
# bottom wall left of corridor
ax.plot([SEAT_X[0], COR_X[0]], [SEAT_Y[0], SEAT_Y[0]], color=wc, lw=lw)
# bottom wall right of corridor
ax.plot([COR_X[1], SEAT_X[1]], [SEAT_Y[0], SEAT_Y[0]], color=wc, lw=lw)

# corridor walls
ax.plot([COR_X[0], COR_X[0]], [COR_Y[0], COR_Y[1]], color=wc, lw=lw)  # left
ax.plot([COR_X[1], COR_X[1]], [COR_Y[0], COR_Y[1]], color=wc, lw=lw)  # right
# corridor bottom left of exit
ax.plot([COR_X[0], EXIT_X[0]], [COR_Y[0], COR_Y[0]], color=wc, lw=lw)
# corridor bottom right of exit
ax.plot([EXIT_X[1], COR_X[1]], [COR_Y[0], COR_Y[0]], color=wc, lw=lw)

# exit glow
ax.add_patch(patches.Rectangle(
    (EXIT_X[0], -0.3), EXIT_W, 0.3,
    facecolor='#00ff88', alpha=0.4))
ax.text(5.0, -0.6, 'EXIT', color='#00ff88', fontsize=11,
        fontweight='bold', ha='center')

# stage label
ax.text(5.0, 10.3, '🎵 STAGE', color='#ffcc44', fontsize=12,
        fontweight='bold', ha='center')
ax.add_patch(patches.Rectangle(
    (1.5, 9.8), 7.0, 0.8,
    facecolor='#2a1a00', edgecolor='#ffcc44', lw=1.5))

# corridor label
ax.text(5.0, 1.5, 'corridor', color='#aaaacc', fontsize=9,
        ha='center', style='italic')

# bottleneck markers
ax.annotate('bottleneck 1', xy=(COR_X[0], SEAT_Y[0]),
            xytext=(-1.5, 2.8), color='#ff8888', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#ff8888', lw=1))
ax.annotate('bottleneck 2', xy=(EXIT_X[0], COR_Y[0]),
            xytext=(-1.5, 0.5), color='#ff8888', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#ff8888', lw=1))

ax.set_xlim(-2.5, 12)
ax.set_ylim(-1.2, 12)
ax.set_aspect('equal')
ax.set_xlabel('x (m)', color='white', fontsize=10)
ax.set_ylabel('y (m)', color='white', fontsize=10)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444466')

# scatter
scat = ax.scatter([], [], c=[], s=120, cmap=cmap, norm=norm,
                  edgecolors='white', linewidths=0.3, zorder=5)

# colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('speed (m/s)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# info text
info = ax.text(0.02, 0.02, '', transform=ax.transAxes,
               color='white', fontsize=10, va='bottom')

# ─── update ───────────────────────────────────────────────────
def update(frame):
    pos = frames[frame]
    vel = vel_frames[frame]
    t   = frame * dt

    if len(pos) > 0:
        spds = np.linalg.norm(vel, axis=1)
        scat.set_offsets(pos)
        scat.set_array(spds)
    else:
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))

    info.set_text(f't = {t:.1f}s    agents: {len(pos)}/{N}'
                  f'\ncorridor width: {COR_W:.1f}m    exit width: {EXIT_W:.1f}m')
    return scat, info

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=40,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()