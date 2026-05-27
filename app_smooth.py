"""
SFM with v0-dependent personal space — interactive simulation.
Iteration 7: one parameter (v0) drives orderly → clogged transition.

Physics driven entirely by B_eff(v0) = B_LO + (B_HI-B_LO)*exp(-v0/V_B):

  Low  v0 → B_eff large → equilibrium gap 1.2 m >> agent size 0.6 m
             Agents CANNOT touch. Priority yielding creates wave-like queue.

  High v0 → B_eff small → equilibrium gap 0.15 m << agent size 0.6 m
             Agents always in body contact. Arch forms and locks.

Three sliders only: desired speed | door width | N agents
Run:  python app_smooth.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.widgets as widgets
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_smooth import (
    step, check_exit, init_agents, _settle,
    ROOM_W, ROOM_H, EXIT_W, EXIT_Y_LO, EXIT_Y_HI,
    R_LO, R_HI, MASS, DT, V0, V_CAP,
    B_eff, d_eq, kappa_scale, patience_val, shape_sq,
)

# ─── mutable state ────────────────────────────────────────────
N      = 60
v0     = 1.0          # default: orderly regime
door_w = EXIT_W
t_end  = 120.0

# ─── grid builder ─────────────────────────────────────────────
def make_grid(n, dw):
    rng     = np.random.default_rng(42)
    radii   = rng.uniform(R_LO, R_HI, n)
    spacing = 0.62
    margin  = 0.45
    xs = np.arange(margin, ROOM_W - margin + 1e-6, spacing)
    ys = np.arange(margin, ROOM_H - margin + 1e-6, spacing)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    if len(grid) < n:
        n = len(grid)
    idx = rng.permutation(len(grid))[:n]
    pos = grid[idx] + rng.uniform(-0.02, 0.02, (n, 2))
    pos[:, 0] = np.clip(pos[:, 0], radii + 0.01, ROOM_W - radii - 0.01)
    pos[:, 1] = np.clip(pos[:, 1], radii + 0.01, ROOM_H - radii - 0.01)
    ey_lo = ROOM_H / 2 - dw / 2
    ey_hi = ROOM_H / 2 + dw / 2
    pos = _settle(pos, radii, ROOM_W, ROOM_H, ey_lo, ey_hi, n_steps=300)
    return pos, np.zeros((n, 2)), radii


def fresh_state():
    global positions, velocities, radii, ey_lo, ey_hi
    global t, avg_speeds, agent_counts, time_axis, n_out_total, N
    ey_lo = ROOM_H / 2 - door_w / 2
    ey_hi = ROOM_H / 2 + door_w / 2
    positions, velocities, radii = make_grid(N, door_w)
    N = len(positions)
    t = 0.0
    avg_speeds = []; agent_counts = []; time_axis = []; n_out_total = 0

fresh_state()

# ─── figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#1a1a2e')

ax = fig.add_axes([0.05, 0.22, 0.55, 0.70])
ax.set_facecolor('#16213e')

ax2 = fig.add_axes([0.67, 0.55, 0.30, 0.35])
ax2.set_facecolor('#16213e')
ax2.set_title('avg speed over time', color='white', fontsize=10)
ax2.set_xlabel('time (s)', color='white', fontsize=9)
ax2.set_ylabel('avg speed (m/s)', color='white', fontsize=9)
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_edgecolor('#444466')

ax3 = fig.add_axes([0.67, 0.10, 0.30, 0.35])
ax3.set_facecolor('#16213e')
ax3.set_title('agents remaining', color='white', fontsize=10)
ax3.set_xlabel('time (s)', color='white', fontsize=9)
ax3.set_ylabel('count', color='white', fontsize=9)
ax3.tick_params(colors='white')
for sp in ax3.spines.values(): sp.set_edgecolor('#444466')

# ─── static room ──────────────────────────────────────────────
ax.add_patch(patches.Rectangle((0, 0), ROOM_W, ROOM_H,
                                facecolor='#0f3460', edgecolor='none'))
wc, lw = '#e0e0e0', 3
ax.plot([0, ROOM_W], [0, 0],           color=wc, lw=lw)
ax.plot([0, ROOM_W], [ROOM_H, ROOM_H], color=wc, lw=lw)
ax.plot([0, 0],      [0, ROOM_H],      color=wc, lw=lw)
wall_lo, = ax.plot([ROOM_W, ROOM_W], [0,      ey_lo],  color=wc, lw=lw)
wall_hi, = ax.plot([ROOM_W, ROOM_W], [ey_hi,  ROOM_H], color=wc, lw=lw)
ax.annotate('', xy=(ROOM_W + 1.2, ROOM_H / 2),
            xytext=(ROOM_W + 0.2, ROOM_H / 2),
            arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2))
ax.text(ROOM_W + 1.3, ROOM_H / 2, 'EXIT',
        color='#00ff88', fontsize=11, fontweight='bold', va='center')
exit_glow = patches.Rectangle((ROOM_W - 0.15, ey_lo), 0.15, ey_hi - ey_lo,
                               facecolor='#00ff88', alpha=0.3)
ax.add_patch(exit_glow)
ax.set_xlim(-0.5, ROOM_W + 3.0)
ax.set_ylim(-0.5, ROOM_H + 0.5)
ax.set_aspect('equal')
ax.set_xlabel('x (m)', color='white', fontsize=11)
ax.set_ylabel('y (m)', color='white', fontsize=11)
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor('#444466')

# ─── physics readout (bottom-left) ────────────────────────────
phys_ann = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                   color='#aaaacc', fontsize=8, va='bottom', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='#0a0a1e', alpha=0.55))

def phys_text(v):
    be  = B_eff(v);   de  = d_eq(v)
    ks  = kappa_scale(v)
    sq  = shape_sq(v)
    touch = 'body contact!' if de < 0.60 else f'gap {de:.2f} m  (no touch)'
    model = f'kinematic×{sq:.2f} + SFM×{1-sq:.2f}'
    return (f'B_eff={be:.3f} m  →  eq.gap={de:.2f} m  [{touch}]\n'
            f'friction×{ks:.2f}   [{model}]')

# ─── regime label ─────────────────────────────────────────────
regime_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                      fontsize=11, va='top', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='#0a0a1e', alpha=0.7))

def regime_info(v):
    de = d_eq(v)
    if de >= 0.60:
        return 'ORDERLY  —  agents never touch', '#00ff88'
    elif de >= 0.35:
        return 'TRANSITION  —  occasional contact', '#ffcc44'
    else:
        return 'JAMMING  —  arch forms at exit', '#ff4466'

def refresh_regime():
    label, color = regime_info(v0)
    regime_text.set_text(label)
    regime_text.set_color(color)
    phys_ann.set_text(phys_text(v0))


def update_walls():
    wall_lo.set_ydata([0, ey_lo])
    wall_hi.set_ydata([ey_hi, ROOM_H])
    exit_glow.set_y(ey_lo)
    exit_glow.set_height(ey_hi - ey_lo)


title_text = ax.set_title('', color='white', fontsize=12, fontweight='bold')

# ─── scatter ──────────────────────────────────────────────────
cmap = plt.cm.coolwarm
norm = Normalize(vmin=0.0, vmax=4.0)
MARKER_S = 480
scat = ax.scatter([], [], c=[], s=MARKER_S, zorder=5, cmap=cmap, norm=norm,
                  edgecolors='white', linewidths=0.3)
sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('speed (m/s)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

speed_line, = ax2.plot([], [], color='#cc88ff', lw=1.5)
count_line, = ax3.plot([], [], color='#88ccff', lw=1.5)
ax2.set_xlim(0, t_end); ax2.set_ylim(0, 5.0)
ax3.set_xlim(0, t_end); ax3.set_ylim(0, N * 1.1)

# ─── THREE sliders ────────────────────────────────────────────
sl_y0, sl_h = 0.145, 0.030
ax_v0 = fig.add_axes([0.07, sl_y0,         0.45, sl_h])
ax_dw = fig.add_axes([0.07, sl_y0 - 0.052, 0.20, sl_h])
ax_nn = fig.add_axes([0.36, sl_y0 - 0.052, 0.20, sl_h])
for aax in [ax_v0, ax_dw, ax_nn]: aax.set_facecolor('#222244')

sl_v0 = widgets.Slider(ax_v0, r'Desired speed  $v^0$ (m/s)',
                        0.3, 4.0, valinit=v0,     valstep=0.1,  color='#5577cc')
sl_dw = widgets.Slider(ax_dw, 'Door (m)',
                        0.5, 3.0, valinit=door_w, valstep=0.1,  color='#aa5577')
sl_nn = widgets.Slider(ax_nn, 'N agents',
                        10,  120, valinit=N,      valstep=5,    color='#aa8833')
for sl in [sl_v0, sl_dw, sl_nn]:
    sl.label.set_color('white'); sl.valtext.set_color('white')

def on_v0(val):
    global v0
    v0 = sl_v0.val
    refresh_regime()

def on_dw(val):
    global door_w
    door_w = sl_dw.val
    do_restart()

def on_nn(val):
    global N
    N = int(sl_nn.val)
    do_restart()

sl_v0.on_changed(on_v0)
sl_dw.on_changed(on_dw)
sl_nn.on_changed(on_nn)

# ─── restart button ───────────────────────────────────────────
ax_btn = fig.add_axes([0.73, 0.13, 0.09, 0.05])
btn = widgets.Button(ax_btn, 'Restart', color='#2a2a4a', hovercolor='#4444aa')
btn.label.set_color('white'); btn.label.set_fontsize(10)

def do_restart():
    fresh_state(); update_walls()
    scat.set_offsets(positions); scat.set_array(np.zeros(len(positions)))
    speed_line.set_data([], []); count_line.set_data([], [])
    ax3.set_ylim(0, N * 1.1)
    refresh_regime()
    title_text.set_text(f'Restarted  —  N={N}  door={door_w:.1f}m')
    fig.canvas.draw_idle()

btn.on_clicked(lambda e: do_restart())
refresh_regime()

# ─── animation ────────────────────────────────────────────────
def update(frame):
    global positions, velocities, radii, t, avg_speeds, agent_counts, time_axis, n_out_total

    if len(positions) == 0 or t >= t_end:
        return scat, speed_line, count_line, title_text

    positions, velocities = step(
        positions, velocities, radii, v0,
        ROOM_W, ROOM_H, ey_lo, ey_hi, DT
    )
    t += DT

    ex = check_exit(positions)
    if ex.any():
        n_out_total += int(ex.sum())
        positions = positions[~ex]; velocities = velocities[~ex]; radii = radii[~ex]

    spds = np.linalg.norm(velocities, axis=1) if len(velocities) > 0 else np.array([0.])
    avg_speeds.append(float(np.mean(spds)))
    agent_counts.append(len(positions))
    time_axis.append(t)

    if len(positions) > 0:
        scat.set_offsets(positions); scat.set_array(spds)
    else:
        scat.set_offsets(np.empty((0, 2))); scat.set_array(np.array([]))

    speed_line.set_data(time_axis, avg_speeds)
    count_line.set_data(time_axis, agent_counts)

    label, _ = regime_info(v0)
    title_text.set_text(
        f'Smooth SFM  —  t={t:.1f}s    in:{len(positions)}  out:{n_out_total}    '
        f'v0={v0:.1f} m/s  B_eff={B_eff(v0):.3f} m  [{label.split("—")[0].strip()}]'
    )
    return scat, speed_line, count_line, title_text

ani = animation.FuncAnimation(fig, update, interval=30, blit=False,
                               cache_frame_data=False, repeat=False)
plt.show()
