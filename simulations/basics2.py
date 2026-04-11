import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ─── parameters ───────────────────────────────────────────────
N     = 20
mass  = 70.0
v0    = 1.5
tau   = 0.5
dt    = 0.05
t_end = 15.0

# ─── starting positions: grid ─────────────────────────────────
cols = 5
rows = 4
xs = np.linspace(1, 7, cols)
ys = np.linspace(1, 5, rows)
grid_x, grid_y = np.meshgrid(xs, ys)
positions  = np.column_stack([grid_x.ravel(), grid_y.ravel()])
velocities = np.zeros((N, 2))

target = np.array([10.0, 3.0])

# ─── forces ───────────────────────────────────────────────────
def self_driving(positions, velocities):
    diff       = target - positions
    dists      = np.linalg.norm(diff, axis=1, keepdims=True)
    directions = diff / dists
    return mass * (v0 * directions - velocities) / tau

def repulsion(positions):
    N = len(positions)
    F = np.zeros((N, 2))
    A, B, k, r = 2000.0, 0.08, 120000.0, 0.5
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = positions[i] - positions[j]
            d    = np.linalg.norm(diff)
            if d < 1e-6:
                continue
            n       = diff / d
            overlap = r - d
            g       = max(overlap, 0.0)
            F[i]   += (A * np.exp(overlap / B) + k * g) * n
    return F

# ─── run simulation, store frames ─────────────────────────────
frames = [positions.copy()]

t = 0.0
while t < t_end:
    f = self_driving(positions, velocities) + repulsion(positions)
    velocities += f / mass * dt
    positions  += velocities * dt

    reached    = np.linalg.norm(positions - target, axis=1) < 0.5
    positions  = positions[~reached]
    velocities = velocities[~reached]

    frames.append(positions.copy())
    t += dt

# ─── animate ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
scat = ax.scatter([], [], s=150, color='steelblue', zorder=3)
ax.plot(*target, 'g*', markersize=16, label='exit')
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 7)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('20 agents — Social Force Model')
ax.grid(True)
ax.legend()

def update(frame):
    step = frames[frame]
    if len(step) > 0:
        scat.set_offsets(step)
    else:
        scat.set_offsets(np.empty((0, 2)))
    return scat,

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=50,
    blit=True,
    repeat=False
)

plt.show()