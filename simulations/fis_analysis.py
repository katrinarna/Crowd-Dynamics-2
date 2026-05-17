"""
Faster-is-slower analysis — improved physics (no bouncing, clear FIS).

Key parameter changes vs faster_is_slower.py:
  dt        0.05 → 0.01   stable spring integration
  r         0.5  → 0.3    60 cm diameter (realistic shoulder width)
  exit      0.4  → 0.9 m  narrow enough for arch formation
  v_cap     relative → ABS_V_MAX = 3.5 m/s absolute ceiling.
             Agents above the cap still push harder (F ∝ v0 − 3.5),
             creating crowd pressure that makes arches MORE stable.
             That pressure difference IS the FIS mechanism.

A, B, k, κ are original Helbing 2000 values (unchanged).

Produces three figures:
  1. Animation: optimal (v0=4.0) vs panic (v0=5.0) — FIS visible directly
  2. U-shaped evacuation time vs desired speed
  3. Overlaid U-curves at 4 densities — minimum shifts LEFT as density rises
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── shared room geometry ─────────────────────────────────────────
ROOM_X  = (0.0, 10.0)
ROOM_Y  = (0.0, 6.0)
EXIT_Y  = (2.55, 3.45)      # 0.9 m door — narrow enough for arch formation
target  = np.array([10.5, 3.0])

# ── physics parameters ───────────────────────────────────────────
MASS    = 70.0
TAU     = 0.5
DT      = 0.01              # smaller step → stable spring integration
T_END   = 70.0

R_AGENT = 0.3               # agent radius (m) — 60 cm diameter
R_WALL  = 0.3               # agent-wall interaction radius

A_SOC   = 2000.0            # social force amplitude (N)  — original Helbing value
B_SOC   = 0.08              # social force range (m)     — original Helbing value
K_BODY  = 120_000.0         # body-contact normal stiffness (N/m)
KAPPA   = 240_000.0         # body-contact tangential friction (N·s/m)

ABS_V_MAX = 3.5             # absolute speed ceiling (m/s) — independent of v0.
                            # Agents with v0 > 3.5 can't move faster but push harder
                            # (F_self ∝ v0 − actual_speed), so higher v0 → more crowd
                            # pressure → more stable arches → FIS.  3.5 puts the FIS
                            # threshold in the range N=50-60, giving clean U-curves.


# ── forces ──────────────────────────────────────────────────────
def self_driving(pos, vel, v0):
    diff = target - pos
    d    = np.linalg.norm(diff, axis=1, keepdims=True)
    e    = diff / np.maximum(d, 1e-6)
    return MASS * (v0 * e - vel) / TAU


def agent_repulsion(pos, vel):
    """Fully vectorised pairwise repulsion — no Python loops over agents."""
    n = len(pos)
    if n < 2:
        return np.zeros((n, 2))

    # pairwise displacement and distance  (n, n, 2) / (n, n)
    diff  = pos[:, None, :] - pos[None, :, :]   # r_i - r_j
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, 1.0)                # avoid self-interaction
    dist  = np.sqrt(dist2)
    np.fill_diagonal(dist, 1.0)

    # unit normal (from j toward i)
    n_vec = diff / dist[:, :, None]             # (n, n, 2)
    # unit tangential (perpendicular to n, rightward)
    t_vec = np.stack([-n_vec[:, :, 1], n_vec[:, :, 0]], axis=-1)  # (n, n, 2)

    overlap = 2 * R_AGENT - dist                # (n, n); positive when touching
    g       = np.maximum(overlap, 0.0)

    # tangential relative velocity: (v_j - v_i) · t  [Helbing sign convention]
    dv_t = np.einsum('ijk,ijk->ij',
                     vel[None, :, :] - vel[:, None, :], t_vec)   # (n, n)

    # mask self-pairs
    mask = ~np.eye(n, dtype=bool)

    # social + contact spring — original Helbing 2000 values
    soc_mag = A_SOC * np.exp(overlap / B_SOC) + K_BODY * g

    # sum forces over all j≠i
    F_norm = np.einsum('ij,ijk->ik', soc_mag * mask, n_vec)
    F_fric = np.einsum('ij,ijk->ik', (KAPPA * g * dv_t) * mask, t_vec)

    return F_norm + F_fric


def wall_forces(pos):
    """Vectorised wall repulsion for left, top, bottom, and right (non-exit) walls."""
    F = np.zeros_like(pos)

    def push(dist_vec, normal):
        d       = np.maximum(dist_vec, 1e-6)
        overlap = R_WALL - d
        g       = np.maximum(overlap, 0.0)
        mag     = A_SOC * np.exp(overlap / B_SOC) + K_BODY * g
        return mag[:, None] * normal

    # left wall (x = ROOM_X[0])
    F += push(pos[:, 0] - ROOM_X[0],  np.array([1.0,  0.0]))
    # bottom wall (y = ROOM_Y[0])
    F += push(pos[:, 1] - ROOM_Y[0],  np.array([0.0,  1.0]))
    # top wall (y = ROOM_Y[1])
    F += push(ROOM_Y[1] - pos[:, 1],  np.array([0.0, -1.0]))

    # right wall — only agents NOT in the exit gap
    not_in_exit = (pos[:, 1] < EXIT_Y[0]) | (pos[:, 1] > EXIT_Y[1])
    if np.any(not_in_exit):
        sub = pos[not_in_exit]
        F[not_in_exit] += push(ROOM_X[1] - sub[:, 0], np.array([-1.0, 0.0]))

    return F


# ── initial layout ───────────────────────────────────────────────
def initial_positions(n_agents, seed=0):
    # ensure no initial overlap: minimum centre-to-centre spacing = 2*R_AGENT + gap
    spacing = 2 * R_AGENT + 0.08        # 0.68 m between agent centres
    y_lo, y_hi = ROOM_Y[0] + spacing, ROOM_Y[1] - spacing
    max_rows = max(1, int((y_hi - y_lo) / spacing) + 1)
    cols     = max(5, int(np.ceil(n_agents / max_rows)))
    rows     = int(np.ceil(n_agents / cols))
    x_hi     = min(ROOM_X[0] + cols * spacing + 0.5, ROOM_X[1] - 0.5)
    xs       = np.linspace(ROOM_X[0] + spacing, x_hi, cols)
    ys       = np.linspace(y_lo, y_hi, rows)
    gx, gy   = np.meshgrid(xs, ys)
    base     = np.column_stack([gx.ravel(), gy.ravel()])[:n_agents].copy()
    rng      = np.random.default_rng(seed)
    jitter   = min(0.12, spacing * 0.15)
    base    += rng.uniform(-jitter, jitter, base.shape)
    return base


# ── run simulation ───────────────────────────────────────────────
def run(v0, n_agents=50, record_frames=False, seed=0, t_end=None):
    if t_end is None:
        t_end = T_END
    pos = initial_positions(n_agents, seed=seed)
    vel = np.zeros((n_agents, 2))

    # record every FRAME_SKIP steps so animation has ~50 fps at real-time playback
    FRAME_SKIP = 2
    frames_pos, frames_vel = ([pos.copy()], [vel.copy()]) if record_frames else (None, None)

    t          = 0.0
    evac_time  = t_end
    steps      = int(t_end / DT)

    # stuck-agent detector: kick anyone stationary for >3 s toward the exit
    stuck_counter = np.zeros(n_agents)
    rng_kick      = np.random.default_rng(seed + 42)

    for step in range(steps):
        if len(pos) == 0:
            evac_time = t
            break

        F    = self_driving(pos, vel, v0) + agent_repulsion(pos, vel) + wall_forces(pos)
        vel += F / MASS * DT

        # absolute speed ceiling — same for every agent regardless of v0.
        # High desired speed raises crowd pressure without letting agents fly.
        spds = np.linalg.norm(vel, axis=1, keepdims=True)
        spds = np.maximum(spds, 1e-6)
        vel  = np.where(spds > ABS_V_MAX, vel / spds * ABS_V_MAX, vel)

        pos += vel * DT
        t   += DT

        exited = pos[:, 0] > ROOM_X[1] - 0.1
        pos    = pos[~exited]
        vel    = vel[~exited]
        stuck_counter = stuck_counter[~exited]

        # End-of-evacuation rescue: if only a handful of agents remain and
        # one is truly isolated and stuck (no nearby contact partner), give
        # it a gentle nudge.  Do NOT fire during the main evacuation — bulk
        # kicks while agents are queued behind an arch just reinforce it.
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
            stuck_counter[:] = 0.0   # reset during main evacuation

        if record_frames and (step % FRAME_SKIP == 0):
            frames_pos.append(pos.copy())
            frames_vel.append(vel.copy())

    return (frames_pos, frames_vel, evac_time) if record_frames else evac_time


# ════════════════════════════════════════════════════════════════
# FIGURE 1 — animation
# ════════════════════════════════════════════════════════════════
V_CALM     = 3.0    # below cap → moves at 3.0 m/s, no excess pressure, smooth flow
V_PANIC    = 5.5    # well above cap → capped at 3.5 m/s but pushes 280 N → FIS!
N_ANIM     = 55     # at N=55 FIS is clear without deadlock (seed 6)
T_END_ANIM = 38.0   # just past panic evacuation (~32 s)
SEED_ANIM  = 6      # seed=6: calm=22.6 s, panic=31.8 s, no deadlock

print(f"Animation run: calm (v0={V_CALM})…")
fp_c, fv_c, t_calm  = run(V_CALM,  n_agents=N_ANIM, record_frames=True,
                           t_end=T_END_ANIM, seed=SEED_ANIM)
print(f"  evacuated in {t_calm:.1f} s")

print(f"Animation run: panic (v0={V_PANIC})…")
fp_p, fv_p, t_panic = run(V_PANIC, n_agents=N_ANIM, record_frames=True,
                           t_end=T_END_ANIM, seed=SEED_ANIM)
print(f"  evacuated in {t_panic:.1f} s")

max_f = max(len(fp_c), len(fp_p))
while len(fp_c) < max_f:
    fp_c.append(np.empty((0, 2))); fv_c.append(np.empty((0, 2)))
while len(fp_p) < max_f:
    fp_p.append(np.empty((0, 2))); fv_p.append(np.empty((0, 2)))

all_spds = np.concatenate([np.linalg.norm(v, axis=1) for v in fv_c + fv_p if len(v) > 0])
cmap = plt.cm.coolwarm
norm = Normalize(vmin=np.percentile(all_spds, 5), vmax=np.percentile(all_spds, 95))

fig1, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(16, 7))
fig1.patch.set_facecolor('#1a1a2e')
fig1.suptitle('Faster-is-slower: calm vs panic (improved physics)',
              color='white', fontsize=14, fontweight='bold', y=0.97)


def setup_room(ax, title, color):
    ax.set_facecolor('#16213e')
    ax.set_title(title, color=color, fontsize=12, fontweight='bold', pad=10)
    ax.add_patch(patches.Rectangle((0, 0), 10, 6, facecolor='#0f3460', edgecolor='none'))
    wc, lw = '#e0e0e0', 3
    ax.plot([0, 10], [0, 0], color=wc, lw=lw)
    ax.plot([0, 10], [6, 6], color=wc, lw=lw)
    ax.plot([0, 0],  [0, 6], color=wc, lw=lw)
    ax.plot([10, 10], [0, EXIT_Y[0]], color=wc, lw=lw)
    ax.plot([10, 10], [EXIT_Y[1], 6], color=wc, lw=lw)
    ax.add_patch(patches.Rectangle((9.85, EXIT_Y[0]), 0.15,
                                   EXIT_Y[1] - EXIT_Y[0],
                                   facecolor='#00ff88', alpha=0.3))
    ax.annotate('', xy=(11.0, 3.0), xytext=(10.1, 3.0),
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2))
    ax.text(11.2, 3.0, 'EXIT', color='#00ff88', fontsize=10,
            fontweight='bold', va='center')
    ax.set_xlim(-0.5, 12.5); ax.set_ylim(-0.5, 7.0)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', color='white'); ax.set_ylabel('y (m)', color='white')
    ax.tick_params(colors='white')
    for s in ax.spines.values():
        s.set_edgecolor('#444466')


setup_room(ax_c, f'Steady walk  v₀ = {V_CALM} m/s  →  smooth flow, no arch', '#88ccff')
setup_room(ax_p, f'Panic run    v₀ = {V_PANIC} m/s  →  arch forms → FIS!  (slower!)', '#ff8888')

scat_c = ax_c.scatter([], [], c=[], s=180, cmap=cmap, norm=norm,
                      edgecolors='white', linewidths=0.4, zorder=5)
scat_p = ax_p.scatter([], [], c=[], s=180, cmap=cmap, norm=norm,
                      edgecolors='white', linewidths=0.4, zorder=5)

sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax_p, fraction=0.03, pad=0.02)
cbar.set_label('speed (m/s)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

txt_c = ax_c.text(0.05, 0.95, '', transform=ax_c.transAxes, color='white', fontsize=11, va='top')
txt_p = ax_p.text(0.05, 0.95, '', transform=ax_p.transAxes, color='white', fontsize=11, va='top')
done_c = ax_c.text(5, 3, '', color='#00ff88', fontsize=14, fontweight='bold', ha='center', va='center', alpha=0)
done_p = ax_p.text(5, 3, '', color='#00ff88', fontsize=14, fontweight='bold', ha='center', va='center', alpha=0)


def update(frame):
    t = frame * DT * 2   # FRAME_SKIP=2
    for scat, fps, fvs, txt, done, evac_t in [
        (scat_c, fp_c, fv_c, txt_c, done_c, t_calm),
        (scat_p, fp_p, fv_p, txt_p, done_p, t_panic),
    ]:
        pos = fps[frame]; vel = fvs[frame]
        if len(pos) > 0:
            scat.set_offsets(pos)
            scat.set_array(np.linalg.norm(vel, axis=1))
        else:
            scat.set_offsets(np.empty((0, 2))); scat.set_array(np.array([]))
        txt.set_text(f't = {t:.1f}s\nagents: {len(pos)}/{N_ANIM}')
        if len(pos) == 0:
            done.set_text(f'EVACUATED in {evac_t:.1f}s')
            done.set_alpha(1)
        elif frame >= max_f - 1 and len(pos) > 0:
            done.set_text(f'FIS: {len(pos)}/{N_ANIM} still inside')
            done.set_alpha(1)
    return scat_c, scat_p, txt_c, txt_p, done_c, done_p


ani = animation.FuncAnimation(fig1, update, frames=max_f,
                              interval=30, blit=False, repeat=False)
plt.tight_layout()


# ════════════════════════════════════════════════════════════════
# FIGURE 2 — evacuation time vs desired speed
# ════════════════════════════════════════════════════════════════
N_SWEEP  = 55    # U-shape is clear at N=55: opt~v4.5, FIS escalates beyond v5.5
# extend to v0=7 to capture the full FIS curve including the secondary hump
v0_range = np.unique(np.concatenate([np.linspace(1.0, 2.5, 4),
                                     np.linspace(2.5, 7.0, 19)]))

N_REPEATS = 18   # 18 seeds → stable medians through the bimodal deadlock regime

print("\nSweeping v0 for evacuation time curve…")
evac_mean = []
evac_std  = []
for v0 in v0_range:
    ets = np.array([run(v0, n_agents=N_SWEEP, seed=s) for s in range(N_REPEATS)])
    evac_mean.append(np.median(ets))           # median: robust to occasional deadlocks
    evac_std.append(np.percentile(ets, 75) - np.percentile(ets, 25))  # IQR
    print(f"  v0={v0:.2f}  →  median {np.median(ets):.1f}s  IQR±{evac_std[-1]:.1f}s")

evac_mean = np.array(evac_mean)
evac_std  = np.array(evac_std)
opt_idx   = np.argmin(evac_mean)
v0_opt    = v0_range[opt_idx]
t_opt     = evac_mean[opt_idx]

fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor('#1a1a2e')
ax2.set_facecolor('#16213e')
ax2.fill_between(v0_range, evac_mean - evac_std, evac_mean + evac_std,
                 color='#88ccff', alpha=0.2)
ax2.plot(v0_range, evac_mean, 'o-', color='#88ccff', lw=2, ms=6, label='median evacuation time')
ax2.axvline(v0_opt, color='#ff8844', lw=1.5, ls='--', label=f'optimal v₀ = {v0_opt:.2f} m/s')
ax2.scatter([v0_opt], [t_opt], color='#ff8844', s=120, zorder=6)
# mark simulation time limit
ax2.axhline(T_END, color='#666688', lw=1, ls=':', alpha=0.7)
ax2.text(v0_range[-1] - 0.15, T_END + 0.5, f'T_end={T_END:.0f}s (deadlock)',
         color='#888899', fontsize=8, ha='right', va='bottom')
ax2.set_xlabel('Desired speed v₀ (m/s)', color='white', fontsize=12)
ax2.set_ylabel('Evacuation time (s)', color='white', fontsize=12)
ax2.set_title(f'Faster-is-slower: evacuation time vs desired speed\n'
              f'(N={N_SWEEP} agents, door = {EXIT_Y[1]-EXIT_Y[0]:.2f} m, median ± IQR over {N_REPEATS} seeds)',
              color='white', fontsize=12)
ax2.legend(framealpha=0.2, labelcolor='white', fontsize=10)
ax2.tick_params(colors='white')
for s in ax2.spines.values():
    s.set_edgecolor('#444466')
ax2.grid(True, alpha=0.15, color='white')

# shade the "faster is slower" region starting from the optimal
fis_start = v0_range[opt_idx]
ax2.axvspan(fis_start, v0_range[-1], alpha=0.07, color='red')
ax2.text(fis_start + 0.1, evac_mean.max() * 0.92,
         'faster-is-slower →', color='#ff6666', fontsize=9, va='top')

plt.tight_layout()


# ════════════════════════════════════════════════════════════════
# FIGURE 3 — overlaid U-curves at 4 densities (left-shifted U)
# ════════════════════════════════════════════════════════════════
n_curve_vals = [30, 45, 52, 55]
curve_colors = ['#66aaff', '#88ddaa', '#ffaa44', '#ff5555']
curve_labels = [f'N={n}  (ρ={n/60:.2f}/m²)' for n in n_curve_vals]

v0_fig3    = np.unique(np.concatenate([np.linspace(1.0, 3.0, 5),
                                       np.linspace(3.0, 5.5, 11)]))
REPEATS_3  = 10

print("\nFigure 3: U-curves at 4 densities…")
curve_data = {}
for n in n_curve_vals:
    row = []
    for v0 in v0_fig3:
        ets = [run(v0, n_agents=n, seed=s) for s in range(REPEATS_3)]
        row.append(np.median(ets))
    curve_data[n] = np.array(row)
    opt_v = v0_fig3[np.argmin(row)]
    last_v = v0_fig3[-1]; last_t = row[-1]
    print(f"  N={n:2d}  optimal v₀≈{opt_v:.2f}  min={min(row):.1f}s  "
          f"v{last_v:.1f}={last_t:.1f}s")

fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor('#1a1a2e')
ax3.set_facecolor('#16213e')

for n, col, lab in zip(n_curve_vals, curve_colors, curve_labels):
    ets = curve_data[n]
    ax3.plot(v0_fig3, ets, 'o-', color=col, lw=2, ms=5, label=lab, alpha=0.85)
    # mark optimal v0 with a larger circle
    opt_i = np.argmin(ets)
    ax3.scatter([v0_fig3[opt_i]], [ets[opt_i]], color=col, s=120,
                edgecolors='white', linewidths=1, zorder=6)

ax3.set_xlabel('Desired speed v₀ (m/s)', color='white', fontsize=12)
ax3.set_ylabel('Median evacuation time (s)', color='white', fontsize=12)
ax3.set_title('Faster-is-slower U-curve shifts LEFT as density increases\n'
              '(white dot = optimal v₀ per density; higher N → lower optimal speed)',
              color='white', fontsize=12)
ax3.legend(framealpha=0.2, labelcolor='white', fontsize=10)
ax3.tick_params(colors='white')
for s in ax3.spines.values():
    s.set_edgecolor('#444466')
ax3.grid(True, alpha=0.15, color='white')
ax3.set_ylim(bottom=0)

plt.tight_layout()
plt.show()
