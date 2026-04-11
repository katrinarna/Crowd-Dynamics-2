import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ─── parameters ───────────────────────────────────────────────
N     = 30
mass  = 70.0
v0    = 1.5
tau   = 0.5
dt    = 0.05
t_end = 30.0

# room dimensions
ROOM_X  = (0.0, 10.0)
ROOM_Y  = (0.0, 6.0)
EXIT_Y  = (2.5, 3.5)    # gap in the right wall
target  = np.array([10.5, 3.0])   # just outside the exit

# ─── starting positions: grid on left side of room ────────────
cols = 5
rows = 6
xs = np.linspace(0.5, 6.0, cols)
ys = np.linspace(0.5, 5.5, rows)
grid_x, grid_y = np.meshgrid(xs, ys)
positions  = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
velocities = np.zeros((N, 2))

# ─── forces ───────────────────────────────────────────────────
def self_driving(positions, velocities):
    diff       = target - positions
    dists      = np.linalg.norm(diff, axis=1, keepdims=True)
    directions = diff / dists
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

def wall_force_segment(positions, wall_point, wall_normal, agent_radius=0.25):
    """
    Push agents away from an infinite wall.
    wall_point:  any point on the wall (numpy array)
    wall_normal: unit vector pointing INTO the room (away from wall)
    """
    n_agents = len(positions)
    F = np.zeros((n_agents, 2))
    AW, BW, k = 2000.0, 0.08, 120000.0

    for i in range(n_agents):
        # distance from agent to the wall
        d = np.dot(positions[i] - wall_point, wall_normal)
        d = max(d, 1e-6)

        overlap = agent_radius - d
        g       = max(overlap, 0.0)
        F[i]   += (AW * np.exp(overlap / BW) + k * g) * wall_normal
    return F

def all_wall_forces(positions):
    F = np.zeros((len(positions), 2))

    # left wall:   x = 0,  normal points right  (+x)
    F += wall_force_segment(positions,
                            np.array([ROOM_X[0], 0.0]),
                            np.array([1.0, 0.0]))

    # bottom wall: y = 0,  normal points up      (+y)
    F += wall_force_segment(positions,
                            np.array([0.0, ROOM_Y[0]]),
                            np.array([0.0, 1.0]))

    # top wall:    y = 6,  normal points down    (-y)
    F += wall_force_segment(positions,
                            np.array([0.0, ROOM_Y[1]]),
                            np.array([0.0, -1.0]))

    # right wall has a gap — split into two segments
    # bottom segment: x=10, y = 0 to EXIT_Y[0]
    for i in range(len(positions)):
        if positions[i, 1] < EXIT_Y[0]:   # agent is below the gap
            d = max(ROOM_X[1] - positions[i, 0], 1e-6)
            overlap = 0.25 - d
            g = max(overlap, 0.0)
            F[i] += (2000.0 * np.exp(overlap / 0.08) + 120000.0 * g) * np.array([-1.0, 0.0])

        elif positions[i, 1] > EXIT_Y[1]: # agent is above the gap
            d = max(ROOM_X[1] - positions[i, 0], 1e-6)
            overlap = 0.25 - d
            g = max(overlap, 0.0)
            F[i] += (2000.0 * np.exp(overlap / 0.08) + 120000.0 * g) * np.array([-1.0, 0.0])

        # agents aligned with the gap feel no right-wall force — they can exit

    return F

# ─── run simulation ───────────────────────────────────────────
frames = [positions.copy()]

t = 0.0
while t < t_end:
    if len(positions) == 0:
        break

    f = (self_driving(positions, velocities)
       + repulsion(positions)
       + all_wall_forces(positions))

    velocities += f / mass * dt

    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds = np.maximum(speeds, 1e-6)   # avoid division by zero
    velocities = np.where(speeds > 3.0, velocities / speeds * 3.0, velocities)

    positions += velocities * dt

    # remove agents who reached the exit
    exited     = np.linalg.norm(positions - target, axis=1) < 1.0
    positions  = positions[~exited]
    velocities = velocities[~exited]

    frames.append(positions.copy())
    t += dt

print(f"Simulation done — {len(frames)} frames")

# ─── animate ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# draw room walls
room = patches.Rectangle((0, 0), 10, 6,
                          linewidth=2, edgecolor='black',
                          facecolor='lightyellow')
ax.add_patch(room)

# draw exit gap (white it out)
gap = patches.Rectangle((9.8, EXIT_Y[0]),
                         0.4, EXIT_Y[1] - EXIT_Y[0],
                         linewidth=0, facecolor='white')
ax.add_patch(gap)

# exit label
ax.annotate('exit', xy=(10.0, 3.0), fontsize=11, color='green',
            va='center', ha='left')

scat = ax.scatter([], [], s=150, color='steelblue', zorder=3)

ax.set_xlim(-0.5, 12)
ax.set_ylim(-0.5, 7)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('30 agents evacuating a room')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# agent counter text
counter = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11)

def update(frame):
    step = frames[frame]
    if len(step) > 0:
        scat.set_offsets(step)
    else:
        scat.set_offsets(np.empty((0, 2)))
    counter.set_text(f'agents remaining: {len(step)}')
    return scat, counter

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=40,
    blit=True,
    repeat=False
)

plt.tight_layout()
plt.show()

def run_simulation(N):

    # ── starting positions ────────────────────────────────────
    cols = 5
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.5, 6.0, cols)
    ys = np.linspace(0.5, 5.5, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions  = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    velocities = np.zeros((N, 2))

    # ── time loop ─────────────────────────────────────────────
    t = 0.0
    while t < t_end:
        if len(positions) == 0:
            return t          # ← everyone exited, return current time

        f = (self_driving(positions, velocities)
           + repulsion(positions)
           + all_wall_forces(positions))

        velocities += f / mass * dt

        # clamp speed
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)
        velocities = np.where(speeds > 3.0, velocities / speeds * 3.0, velocities)

        positions  += velocities * dt

        exited     = np.linalg.norm(positions - target, axis=1) < 1.0
        positions  = positions[~exited]
        velocities = velocities[~exited]

        t += dt

    return t_end   # ← if time ran out before everyone exited

crowd_sizes = [5, 10, 20, 30, 40, 50]
evac_times  = []

for N in crowd_sizes:
    t = run_simulation(N)
    evac_times.append(t)
    print(f"N={N:3d}  →  evacuation time: {t:.1f} s")

plt.figure(figsize=(8, 5))
plt.plot(crowd_sizes, evac_times, 'o-', linewidth=2, markersize=8)
plt.xlabel('number of agents')
plt.ylabel('evacuation time (s)')
plt.title('Evacuation time vs crowd size')
plt.grid(True)
plt.show()

def run_simulation_exitwidth(N, exit_width):

    # redefine exit based on width
    ey_low  = 3.0 - exit_width / 2
    ey_high = 3.0 + exit_width / 2

    # starting positions
    cols = 5
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.5, 6.0, cols)
    ys = np.linspace(0.5, 5.5, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions  = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    velocities = np.zeros((N, 2))

    t = 0.0
    while t < t_end:
        if len(positions) == 0:
            return t

        f = (self_driving(positions, velocities)
           + repulsion(positions)
           + wall_forces_custom_exit(positions, ey_low, ey_high))

        velocities += f / mass * dt
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)
        velocities = np.where(speeds > 3.0, velocities / speeds * 3.0, velocities)
        positions  += velocities * dt

        exited     = np.linalg.norm(positions - target, axis=1) < 1.0
        positions  = positions[~exited]
        velocities = velocities[~exited]

        t += dt

    return t_end

def wall_forces_custom_exit(positions, ey_low, ey_high):
    F = np.zeros((len(positions), 2))
    A, B, k = 2000.0, 0.08, 120000.0
    r = 0.25

    # left wall
    F += wall_force_segment(positions,
                            np.array([ROOM_X[0], 0.0]),
                            np.array([1.0, 0.0]))
    # bottom wall
    F += wall_force_segment(positions,
                            np.array([0.0, ROOM_Y[0]]),
                            np.array([0.0, 1.0]))
    # top wall
    F += wall_force_segment(positions,
                            np.array([0.0, ROOM_Y[1]]),
                            np.array([0.0, -1.0]))

    # right wall with custom exit gap
    for i in range(len(positions)):
        if positions[i, 1] < ey_low or positions[i, 1] > ey_high:
            d = max(ROOM_X[1] - positions[i, 0], 1e-6)
            overlap = r - d
            g = max(overlap, 0.0)
            F[i] += (A * np.exp(overlap / B) + k * g) * np.array([-1.0, 0.0])

    return F

N = 30
exit_widths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
evac_times  = []

for w in exit_widths:
    t = run_simulation_exitwidth(N, w)
    evac_times.append(t)
    print(f"exit width={w:.1f}m  →  evacuation time: {t:.1f} s")

plt.figure(figsize=(8, 5))
plt.plot(exit_widths, evac_times, 'o-', linewidth=2, markersize=8, color='darkorange')
plt.xlabel('exit width (m)')
plt.ylabel('evacuation time (s)')
plt.title(f'Evacuation time vs exit width  (N={N})')
plt.grid(True)
plt.show()


def run_simulation_panic(N, desired_speed):

    cols = 5
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.5, 6.0, cols)
    ys = np.linspace(0.5, 5.5, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions  = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    velocities = np.zeros((N, 2))

    t = 0.0
    while t < t_end:
        if len(positions) == 0:
            return t

        # use desired_speed instead of global v0
        diff       = target - positions
        dists      = np.linalg.norm(diff, axis=1, keepdims=True)
        directions = diff / dists
        f_drive    = mass * (desired_speed * directions - velocities) / tau

        f = f_drive + repulsion(positions) + all_wall_forces(positions)

        velocities += f / mass * dt
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)
        velocities = np.where(speeds > desired_speed * 2, 
                              velocities / speeds * desired_speed * 2, 
                              velocities)
        positions  += velocities * dt

        exited     = np.linalg.norm(positions - target, axis=1) < 1.0
        positions  = positions[~exited]
        velocities = velocities[~exited]

        t += dt

    return t_end


N = 30
speeds     = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
evac_times = []

for s in speeds:
    t = run_simulation_panic(N, s)
    evac_times.append(t)
    print(f"v0={s:.1f} m/s  →  evacuation time: {t:.1f} s")

plt.figure(figsize=(8, 5))
plt.plot(speeds, evac_times, 'o-', linewidth=2, markersize=8, color='crimson')
plt.xlabel('desired speed v⁰ (m/s)')
plt.ylabel('evacuation time (s)')
plt.title(f'Faster is slower effect  (N={N})')
plt.grid(True)
plt.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, label='normal walking speed')
plt.legend()
plt.show()


def run_simulation_two_exits(N, exit_width):
    # two exits: one on right wall, one on bottom wall
    # each has half the total width
    half_w = exit_width / 2

    # right exit: centered at y=3.0
    ey_low  = 3.0 - half_w / 2
    ey_high = 3.0 + half_w / 2

    # bottom exit: centered at x=5.0
    ex_low  = 5.0 - half_w / 2
    ex_high = 5.0 + half_w / 2

    # two targets — each agent goes to the nearest one
    target_right  = np.array([10.5, 3.0])
    target_bottom = np.array([5.0, -0.5])

    cols = 5
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.5, 6.0, cols)
    ys = np.linspace(0.5, 5.5, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions  = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    velocities = np.zeros((N, 2))

    t = 0.0
    while t < t_end:
        if len(positions) == 0:
            return t

        # each agent heads toward whichever exit is closer
        dist_right  = np.linalg.norm(positions - target_right,  axis=1)
        dist_bottom = np.linalg.norm(positions - target_bottom, axis=1)
        go_right    = dist_right <= dist_bottom   # boolean mask

        # desired directions
        targets_i  = np.where(go_right[:, None], target_right, target_bottom)
        diff       = targets_i - positions
        dists      = np.linalg.norm(diff, axis=1, keepdims=True)
        directions = diff / np.maximum(dists, 1e-6)
        f_drive    = mass * (v0 * directions - velocities) / tau

        f = (f_drive
           + repulsion(positions)
           + wall_forces_two_exits(positions, ey_low, ey_high, ex_low, ex_high))

        velocities += f / mass * dt
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)
        velocities = np.where(speeds > 3.0, velocities / speeds * 3.0, velocities)
        positions  += velocities * dt

        # remove agents who reached either exit
        exited = ((np.linalg.norm(positions - target_right,  axis=1) < 1.0) |
                  (np.linalg.norm(positions - target_bottom, axis=1) < 1.0))
        positions  = positions[~exited]
        velocities = velocities[~exited]

        t += dt

    return t_end


def wall_forces_two_exits(positions, ey_low, ey_high, ex_low, ex_high):
    F = np.zeros((len(positions), 2))
    A, B, k, r = 2000.0, 0.08, 120000.0, 0.25

    # left wall — full, no gap
    F += wall_force_segment(positions,
                            np.array([ROOM_X[0], 0.0]),
                            np.array([1.0, 0.0]))

    # top wall — full, no gap
    F += wall_force_segment(positions,
                            np.array([0.0, ROOM_Y[1]]),
                            np.array([0.0, -1.0]))

    # right wall with gap ey_low to ey_high
    for i in range(len(positions)):
        if positions[i, 1] < ey_low or positions[i, 1] > ey_high:
            d = max(ROOM_X[1] - positions[i, 0], 1e-6)
            overlap = r - d
            g = max(overlap, 0.0)
            F[i] += (A * np.exp(overlap / B) + k * g) * np.array([-1.0, 0.0])

    # bottom wall with gap ex_low to ex_high
    for i in range(len(positions)):
        if positions[i, 0] < ex_low or positions[i, 0] > ex_high:
            d = max(positions[i, 1] - ROOM_Y[0], 1e-6)
            overlap = r - d
            g = max(overlap, 0.0)
            F[i] += (A * np.exp(overlap / B) + k * g) * np.array([0.0, 1.0])

    return F

N = 30
total_width = 2.0   # same total exit width for fair comparison
widths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

times_one = []
times_two = []

for w in widths:
    t1 = run_simulation_exitwidth(N, w)
    t2 = run_simulation_two_exits(N, w)
    times_one.append(t1)
    times_two.append(t2)
    print(f"width={w:.1f}m  one exit: {t1:.1f}s   two exits: {t2:.1f}s")

plt.figure(figsize=(8, 5))
plt.plot(widths, times_one, 'o-', linewidth=2, markersize=8,
         color='steelblue', label='one exit (right wall)')
plt.plot(widths, times_two, 's-', linewidth=2, markersize=8,
         color='darkorange', label='two exits (right + bottom)')
plt.xlabel('exit width per exit (m)')
plt.ylabel('evacuation time (s)')
plt.title(f'One exit vs two exits  (N={N})')
plt.legend()
plt.grid(True)
plt.show()

# -----
# exit position.
# -----
def run_simulation_exit_position(N, exit_center_y, exit_width=1.0):

    ey_low  = exit_center_y - exit_width / 2
    ey_high = exit_center_y + exit_width / 2
    target_pos = np.array([10.5, exit_center_y])

    cols = 5
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.5, 6.0, cols)
    ys = np.linspace(0.5, 5.5, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions  = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    velocities = np.zeros((N, 2))

    t = 0.0
    while t < t_end:
        if len(positions) == 0:
            return t

        # self driving toward the exit position
        diff       = target_pos - positions
        dists      = np.linalg.norm(diff, axis=1, keepdims=True)
        directions = diff / np.maximum(dists, 1e-6)
        f_drive    = mass * (v0 * directions - velocities) / tau

        f = (f_drive
           + repulsion(positions)
           + wall_forces_custom_exit(positions, ey_low, ey_high))

        velocities += f / mass * dt
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)
        velocities = np.where(speeds > 3.0, velocities / speeds * 3.0, velocities)
        positions  += velocities * dt

        exited     = np.linalg.norm(positions - target_pos, axis=1) < 1.0
        positions  = positions[~exited]
        velocities = velocities[~exited]

        t += dt

    return t_end

N = 30
# slide exit from near bottom to near top
# room is 6m tall, exit width 1.0m
# so valid centers run from 0.5 to 5.5
exit_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
evac_times     = []

for y in exit_positions:
    t = run_simulation_exit_position(N, exit_center_y=y)
    evac_times.append(t)
    print(f"exit center y={y:.1f}m  →  {t:.1f} s")

plt.figure(figsize=(8, 5))
plt.plot(exit_positions, evac_times, 'o-', linewidth=2, markersize=8, color='purple')
plt.axvline(x=3.0, color='gray', linestyle='--', alpha=0.6, label='room center (y=3)')
plt.axhline(y=min(evac_times), color='green', linestyle='--', alpha=0.6, label='fastest time')
plt.xlabel('exit center position (m from bottom)')
plt.ylabel('evacuation time (s)')
plt.title(f'Evacuation time vs exit position  (N={N}, width=1.0m)')
plt.legend()
plt.grid(True)
plt.show()