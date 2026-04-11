import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle

# ─── parameters ───────────────────────────────────────────────
N          = 30
mass       = 70.0
v0         = 1.5
tau        = 0.3
dt         = 0.05
t_end      = 120.0
FIRE_RATE  = 0.20  # slower spread
FIRE_MAX_R = 5.0   # cap — fire stops growing at 4m radius

# ─── building geometry ────────────────────────────────────────
# three offices on top, corridor below
OFFICE_W = 6.0
OFFICE_H = 5.0
COR_H    = 3.0
TOTAL_W  = OFFICE_W * 3   # 18m wide

# y ranges
COR_Y    = (0.0, COR_H)
OFFICE_Y = (COR_H, COR_H + OFFICE_H)

# x ranges for each office
OFF1_X = (0.0,       OFFICE_W)
OFF2_X = (OFFICE_W,  OFFICE_W * 2)
OFF3_X = (OFFICE_W * 2, TOTAL_W)

# door positions (openings in office bottom walls)
DOOR_W   = 3.0
DOOR1_X  = (OFF1_X[0] + OFFICE_W/2 - DOOR_W/2,
             OFF1_X[0] + OFFICE_W/2 + DOOR_W/2)
DOOR2_X  = (OFF2_X[0] + OFFICE_W/2 - DOOR_W/2,
             OFF2_X[0] + OFFICE_W/2 + DOOR_W/2)
DOOR3_X  = (OFF3_X[0] + OFFICE_W/2 - DOOR_W/2,
             OFF3_X[0] + OFFICE_W/2 + DOOR_W/2)

# exit at bottom center of corridor
EXIT_W   = 1.5
EXIT_X   = (TOTAL_W/2 - EXIT_W/2, TOTAL_W/2 + EXIT_W/2)
target   = np.array([TOTAL_W/2, -0.5])

# fire starts near office 3 door — blocks escape route
FIRE_START = np.array([OFF3_X[0] + OFFICE_W/2, OFFICE_Y[0] + 1.0])

# ─── starting positions ───────────────────────────────────────
np.random.seed(7)

def make_agents(ox_low, ox_high, n=15):
    return np.column_stack([
        np.random.uniform(ox_low + 0.5, ox_high - 0.5, n),
        np.random.uniform(OFFICE_Y[0] + 0.5, OFFICE_Y[1] - 0.5, n)
    ])

positions = np.vstack([
    make_agents(*OFF1_X, n=10),
    make_agents(*OFF2_X, n=10),
    make_agents(*OFF3_X, n=10)
])
velocities = np.zeros((N, 2))

# ─── forces ───────────────────────────────────────────────────
def get_waypoint(pos):
    """Return the next target for each agent based on their position."""
    x, y = pos
    in_office = (0 <= x <= TOTAL_W) and (OFFICE_Y[0] <= y <= OFFICE_Y[1])
    
    if in_office:
        # figure out which office and send them to that door center
        if x < OFF1_X[1]:
            return np.array([OFF1_X[0] + OFFICE_W/2, OFFICE_Y[0] - 0.5])
        elif x < OFF2_X[1]:
            return np.array([OFF2_X[0] + OFFICE_W/2, OFFICE_Y[0] - 0.5])
        else:
            return np.array([OFF3_X[0] + OFFICE_W/2, OFFICE_Y[0] - 0.5])
    else:
        # in corridor — head for exit
        return target

def self_driving(positions, velocities):
    n = len(positions)
    F = np.zeros((n, 2))
    for i in range(n):
        wp   = get_waypoint(positions[i])
        diff = wp - positions[i]
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            continue
        direction = diff / dist
        F[i] = mass * (v0 * direction - velocities[i]) / tau
    return F

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

def fire_force(positions, fire_center, fire_radius):
    n = len(positions)
    F = np.zeros((n, 2))
    A_fire = 5000.0    # stronger
    B_fire = 0.8       # longer range

    for i in range(n):
        diff = positions[i] - fire_center
        d    = np.linalg.norm(diff)
        if d < 1e-6:
            continue
        n_vec       = diff / d
        effective_d = max(d - fire_radius, 0.01)
        if effective_d > 4.0:
            continue
        F[i] += A_fire * np.exp(-effective_d / B_fire) * n_vec
    return F

def push_from_wall(d, normal, A=1500., B=0.08, k=120000., r=0.25):
    d  = max(d, 1e-6)
    ov = r - d
    g  = max(ov, 0.0)
    return (A * np.exp(ov / B) + k * g) * normal

def clamp_positions(positions, velocities):
    """Zero out velocity component pointing into a wall if agent is too close."""
    margin = 0.26
    for i in range(len(positions)):
        x, y = positions[i]
        
        # left wall
        if x < margin:
            positions[i, 0] = margin
            if velocities[i, 0] < 0:
                velocities[i, 0] = 0.0

        # right wall
        if x > TOTAL_W - margin:
            positions[i, 0] = TOTAL_W - margin
            if velocities[i, 0] > 0:
                velocities[i, 0] = 0.0

        # top wall (offices)
        if y > OFFICE_Y[1] - margin:
            positions[i, 1] = OFFICE_Y[1] - margin
            if velocities[i, 1] > 0:
                velocities[i, 1] = 0.0

        # bottom wall (corridor) — only outside exit gap
        if y < COR_Y[0] + margin:
            if x < EXIT_X[0] or x > EXIT_X[1]:
                positions[i, 1] = COR_Y[0] + margin
                if velocities[i, 1] < 0:
                    velocities[i, 1] = 0.0

    return positions, velocities

def all_wall_forces(positions):
    n = len(positions)
    F = np.zeros((n, 2))

    for i in range(n):
        x, y = positions[i]
        in_office   = (0 <= x <= TOTAL_W) and (OFFICE_Y[0] <= y <= OFFICE_Y[1])
        in_corridor = (0 <= x <= TOTAL_W) and (COR_Y[0] <= y <= COR_Y[1])

        if in_office:
            F[i] += push_from_wall(x,                np.array([ 1., 0.]))
            F[i] += push_from_wall(TOTAL_W - x,      np.array([-1., 0.]))
            F[i] += push_from_wall(OFFICE_Y[1] - y,  np.array([ 0.,-1.]))
            # internal vertical divider between office 1 and 2
            if abs(x - OFF1_X[1]) < 2.0:
                direction = 1.0 if x < OFF1_X[1] else -1.0
                F[i] += push_from_wall(abs(x - OFF1_X[1]),
                                       np.array([direction, 0.]),
                                       A=3000., k=200000.)
            # internal vertical divider between office 2 and 3
            if abs(x - OFF2_X[1]) < 2.0:
                direction = 1.0 if x < OFF2_X[1] else -1.0
                F[i] += push_from_wall(abs(x - OFF2_X[1]),
                                       np.array([direction, 0.]),
                                       A=3000., k=200000.)

            # bottom wall with door gap
            if x < OFF1_X[1]:
                if x < DOOR1_X[0] or x > DOOR1_X[1]:
                    F[i] += push_from_wall(y - OFFICE_Y[0], np.array([0., 1.]))
            elif x < OFF2_X[1]:
                if x < DOOR2_X[0] or x > DOOR2_X[1]:
                    F[i] += push_from_wall(y - OFFICE_Y[0], np.array([0., 1.]))
            else:
                if x < DOOR3_X[0] or x > DOOR3_X[1]:
                    F[i] += push_from_wall(y - OFFICE_Y[0], np.array([0., 1.]))

        elif in_corridor:
            F[i] += push_from_wall(x,                np.array([ 1., 0.]))
            F[i] += push_from_wall(TOTAL_W - x,      np.array([-1., 0.]))
            F[i] += push_from_wall(COR_Y[1] - y,     np.array([ 0.,-1.]))
            if x < EXIT_X[0] or x > EXIT_X[1]:
                F[i] += push_from_wall(y - COR_Y[0], np.array([ 0., 1.]),
                                       A=3000., k=200000.)

    return F

def clamp_positions(positions, velocities):
    margin = 0.26
    for i in range(len(positions)):
        x, y = positions[i]

        # outer left/right walls
        if x < margin:
            positions[i, 0] = margin
            if velocities[i, 0] < 0: velocities[i, 0] = 0.0
        if x > TOTAL_W - margin:
            positions[i, 0] = TOTAL_W - margin
            if velocities[i, 0] > 0: velocities[i, 0] = 0.0

        # top wall
        if y > OFFICE_Y[1] - margin:
            positions[i, 1] = OFFICE_Y[1] - margin
            if velocities[i, 1] > 0: velocities[i, 1] = 0.0

        # bottom corridor wall — outside exit gap
        if y < COR_Y[0] + margin:
            if x < EXIT_X[0] or x > EXIT_X[1]:
                positions[i, 1] = COR_Y[0] + margin
                if velocities[i, 1] < 0: velocities[i, 1] = 0.0

        # internal vertical wall office 1 | office 2
        in_office = (0 <= x <= TOTAL_W) and (OFFICE_Y[0] <= y <= OFFICE_Y[1])
        if in_office:
            if abs(x - OFF1_X[1]) < margin:
                if x < OFF1_X[1]:
                    positions[i, 0] = OFF1_X[1] - margin
                    if velocities[i, 0] > 0: velocities[i, 0] = 0.0
                else:
                    positions[i, 0] = OFF1_X[1] + margin
                    if velocities[i, 0] < 0: velocities[i, 0] = 0.0

            # internal vertical wall office 2 | office 3
            if abs(x - OFF2_X[1]) < margin:
                if x < OFF2_X[1]:
                    positions[i, 0] = OFF2_X[1] - margin
                    if velocities[i, 0] > 0: velocities[i, 0] = 0.0
                else:
                    positions[i, 0] = OFF2_X[1] + margin
                    if velocities[i, 0] < 0: velocities[i, 0] = 0.0

    return positions, velocities

# ─── run simulation ───────────────────────────────────────────
print("Running simulation...")
frames      = [positions.copy()]
vel_frames  = [velocities.copy()]
fire_frames = [0.0]   # fire radius at each frame

t            = 0.0
fire_radius  = 0.1

while t < t_end:
    if len(positions) == 0:
        print(f"All evacuated at t={t:.1f}s")
        break

    # grow fire
    fire_radius = min(fire_radius + FIRE_RATE * dt, FIRE_MAX_R)

    f = (self_driving(positions, velocities)
       + repulsion(positions, velocities)
       + all_wall_forces(positions)
       + fire_force(positions, FIRE_START, fire_radius))

    velocities += f / mass * dt
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds = np.maximum(speeds, 1e-6)
    velocities = np.where(speeds > 3.0, velocities / speeds * 3.0, velocities)
    positions  = positions + velocities * dt
    positions, velocities = clamp_positions(positions, velocities)

    # remove agents who reached exit OR are caught by fire
    exited  = np.linalg.norm(positions - target, axis=1) < 1.5
    caught  = np.linalg.norm(positions - FIRE_START, axis=1) < fire_radius
    remove  = exited | caught
    positions  = positions[~remove]
    velocities = velocities[~remove]

    frames.append(positions.copy())
    vel_frames.append(velocities.copy())
    fire_frames.append(fire_radius)
    t += dt

print(f"Done — {len(frames)} frames")

# ─── experiment: evacuation time vs fire spread rate ──────────
def run_experiment(fire_rate, n_agents=30, seed=7):
    np.random.seed(seed)
    pos = np.vstack([make_agents(*OFF1_X, n=10),
                     make_agents(*OFF2_X, n=10),
                     make_agents(*OFF3_X, n=10)])
    vel  = np.zeros((n_agents, 2))
    fr   = 0.1
    t    = 0.0
    escaped = 0

    while t < t_end:
        fr   = min(fr + fire_rate * dt, FIRE_MAX_R)
        f    = (self_driving(pos, vel)
               + repulsion(pos, vel)
               + all_wall_forces(pos)
               + fire_force(pos, FIRE_START, fr))
        vel += f / mass * dt
        spds = np.linalg.norm(vel, axis=1, keepdims=True)
        spds = np.maximum(spds, 1e-6)
        vel  = np.where(spds > 3.0, vel / spds * 3.0, vel)
        pos = pos + vel * dt
        pos, vel = clamp_positions(pos, vel) 

        exited  = np.linalg.norm(pos - target, axis=1) < 1.5
        caught  = np.linalg.norm(pos - FIRE_START, axis=1) < fr
        escaped += int(np.sum(exited))
        remove  = exited | caught
        pos = pos[~remove]
        vel = vel[~remove]
        t  += dt

        if len(pos) == 0:
            return t, escaped

    return t_end, escaped



fire_rates   = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
evac_times   = []
escaped_list = []

for rate in fire_rates:
    times_seeds   = []
    escaped_seeds = []
    for seed in [7, 42, 123]:
        et, esc = run_experiment(rate, seed=seed)
        times_seeds.append(et)
        escaped_seeds.append(esc)
    evac_times.append(np.mean(times_seeds))
    escaped_list.append(np.mean(escaped_seeds))
    print(f"  rate={rate:.2f}  avg_t={evac_times[-1]:.1f}s  avg_escaped={escaped_list[-1]:.1f}")

# ─── plot experiment results ───────────────────────────────────
fig_exp, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(12, 5))
fig_exp.patch.set_facecolor('#1a1a2e')

for ax in (ax_e1, ax_e2):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

ax_e1.plot(fire_rates, evac_times, 'o-', color='#ff6644',
           linewidth=2, markersize=8)
ax_e1.set_xlabel('fire spread rate (m/s)', color='white', fontsize=11)
ax_e1.set_ylabel('evacuation time (s)',    color='white', fontsize=11)
ax_e1.set_title('Evacuation time vs fire spread rate',
                color='white', fontsize=12, fontweight='bold')
ax_e1.grid(True, alpha=0.2)

ax_e2.plot(fire_rates, escaped_list, 's-', color='#44cc88',
           linewidth=2, markersize=8)
ax_e2.axhline(y=N, color='white', linestyle='--', alpha=0.4, label=f'total agents ({N})')
ax_e2.set_xlabel('fire spread rate (m/s)', color='white', fontsize=11)
ax_e2.set_ylabel('agents escaped',         color='white', fontsize=11)
ax_e2.set_title('Agents escaped vs fire spread rate',
                color='white', fontsize=12, fontweight='bold')
ax_e2.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
ax_e2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('fire_experiment.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
print("Experiment plot saved as fire_experiment.png")
plt.show()

# ─── colormap for animation ────────────────────────────────────
all_spds = []
for v in vel_frames:
    if len(v) > 0:
        all_spds.extend(np.linalg.norm(v, axis=1).tolist())
all_spds = np.array(all_spds) if len(all_spds) > 0 else np.array([0, 1])
cmap_agents = plt.cm.coolwarm
norm_agents = Normalize(vmin=np.percentile(all_spds, 5),
                        vmax=np.percentile(all_spds, 95))

# ─── animation figure ─────────────────────────────────────────
fig_ani, ax_ani = plt.subplots(figsize=(12, 9))
fig_ani.patch.set_facecolor('#1a1a2e')
ax_ani.set_facecolor('#111122')
ax_ani.set_title('Office building fire evacuation',
                 color='white', fontsize=13, fontweight='bold')

# ── static building elements ──────────────────────────────────
wc, lw = '#ccccdd', 2.5

# office fills
for ox, label in [(OFF1_X, 'Office 1'), (OFF2_X, 'Office 2'), (OFF3_X, 'Office 3')]:
    ax_ani.add_patch(patches.Rectangle(
        (ox[0], OFFICE_Y[0]), OFFICE_W, OFFICE_H,
        facecolor='#1a2a4a', edgecolor='none'))
    ax_ani.text(ox[0] + OFFICE_W/2, OFFICE_Y[1] - 0.4,
                label, color='#aaaacc', fontsize=9, ha='center')

# corridor fill
ax_ani.add_patch(patches.Rectangle(
    (0, COR_Y[0]), TOTAL_W, COR_H,
    facecolor='#0f1a2a', edgecolor='none'))
ax_ani.text(TOTAL_W/2, COR_H/2, 'corridor',
            color='#667788', fontsize=9, ha='center', style='italic')

# office walls
for ox, door in [(OFF1_X, DOOR1_X), (OFF2_X, DOOR2_X), (OFF3_X, DOOR3_X)]:
    ax_ani.plot([ox[0], ox[1]], [OFFICE_Y[1], OFFICE_Y[1]], color=wc, lw=lw)
    ax_ani.plot([ox[0], ox[0]], [OFFICE_Y[0], OFFICE_Y[1]], color=wc, lw=lw)
    ax_ani.plot([ox[1], ox[1]], [OFFICE_Y[0], OFFICE_Y[1]], color=wc, lw=lw)
    ax_ani.plot([ox[0], door[0]], [OFFICE_Y[0], OFFICE_Y[0]], color=wc, lw=lw)
    ax_ani.plot([door[1], ox[1]], [OFFICE_Y[0], OFFICE_Y[0]], color=wc, lw=lw)

# corridor walls
ax_ani.plot([0, TOTAL_W], [COR_Y[1], COR_Y[1]], color=wc, lw=lw, alpha=0.3)
ax_ani.plot([0, 0],       [COR_Y[0], COR_Y[1]], color=wc, lw=lw)
ax_ani.plot([TOTAL_W, TOTAL_W], [COR_Y[0], COR_Y[1]], color=wc, lw=lw)
ax_ani.plot([0, EXIT_X[0]],     [COR_Y[0], COR_Y[0]], color=wc, lw=lw)
ax_ani.plot([EXIT_X[1], TOTAL_W],[COR_Y[0], COR_Y[0]], color=wc, lw=lw)

# exit
ax_ani.add_patch(patches.Rectangle(
    (EXIT_X[0], -0.4), EXIT_W, 0.4,
    facecolor='#00ff88', alpha=0.5))
ax_ani.text(TOTAL_W/2, -0.7, 'EXIT', color='#00ff88',
            fontsize=11, fontweight='bold', ha='center')

# desk decorations
for ox in [OFF1_X, OFF2_X, OFF3_X]:
    for dx in np.linspace(ox[0]+1, ox[1]-1, 3):
        for dy in [OFFICE_Y[0]+1.2, OFFICE_Y[0]+2.8, OFFICE_Y[0]+4.2]:
            ax_ani.add_patch(patches.Rectangle(
                (dx-0.3, dy-0.2), 0.6, 0.35,
                facecolor='#2a3a5a', edgecolor='#445566', lw=0.5))

ax_ani.set_xlim(-1, TOTAL_W + 1)
ax_ani.set_ylim(-1.2, OFFICE_Y[1] + 0.5)
ax_ani.set_aspect('equal')
ax_ani.set_xlabel('x (m)', color='white', fontsize=10)
ax_ani.set_ylabel('y (m)', color='white', fontsize=10)
ax_ani.tick_params(colors='white')
for spine in ax_ani.spines.values():
    spine.set_edgecolor('#444466')

# fire visual — three concentric circles for glow effect
fire_core  = Circle(FIRE_START, 0.1, color='#ffffff', zorder=4)
fire_mid   = Circle(FIRE_START, 0.1, color='#ff6600', alpha=0.6, zorder=3)
fire_outer = Circle(FIRE_START, 0.1, color='#ff2200', alpha=0.25, zorder=2)
fire_glow  = Circle(FIRE_START, 0.1, color='#ffaa00', alpha=0.08, zorder=1)
ax_ani.add_patch(fire_glow)
ax_ani.add_patch(fire_outer)
ax_ani.add_patch(fire_mid)
ax_ani.add_patch(fire_core)

# agent scatter
scat = ax_ani.scatter([], [], c=[], s=120, cmap=cmap_agents, norm=norm_agents,
                      edgecolors='white', linewidths=0.3, zorder=5)

# colorbar
sm = ScalarMappable(cmap=cmap_agents, norm=norm_agents)
sm.set_array([])
cbar = fig_ani.colorbar(sm, ax=ax_ani, fraction=0.02, pad=0.02)
cbar.set_label('speed (m/s)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# info
info_text = ax_ani.text(0.01, 0.01, '', transform=ax_ani.transAxes,
                        color='white', fontsize=10, va='bottom')
fire_text = ax_ani.text(0.99, 0.01, '', transform=ax_ani.transAxes,
                        color='#ff6644', fontsize=10, va='bottom', ha='right')

# ─── animation update ─────────────────────────────────────────
def update(frame):
    pos = frames[frame]
    vel = vel_frames[frame]
    fr  = fire_frames[frame]
    t   = frame * dt

    # update agent scatter
    if len(pos) > 0:
        spds = np.linalg.norm(vel, axis=1)
        scat.set_offsets(pos)
        scat.set_array(spds)
    else:
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))

    # update fire circles
    fire_core.set_radius(fr * 0.4)
    fire_mid.set_radius(fr * 0.7)
    fire_outer.set_radius(fr)
    fire_glow.set_radius(fr * 1.5)

    info_text.set_text(f't = {t:.1f}s    agents remaining: {len(pos)}/{N}')
    fire_text.set_text(f'fire radius: {fr:.1f}m')

    return scat, fire_core, fire_mid, fire_outer, fire_glow, info_text, fire_text

ani = animation.FuncAnimation(
    fig_ani,
    update,
    frames=len(frames),
    interval=40,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()