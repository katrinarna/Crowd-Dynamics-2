"""
Stress-Coupled Evacuation — same physics as app_smooth.py.
Slider "Collective stress σ" (0→1):
  - Sets initial β for ALL agents uniformly (β_i = σ at t=0)
  - Maps to v0 for physics: v0 = 0.3 + σ * 3.7
  - Scales contagion: high σ → stress spreads faster between nearby agents

SIS stress dynamics (COLORING ONLY, no effect on physics):
  dβ_i/dt = −μ β_i + κ(σ) Σ_j W_ij β_j (1−β_i),  W_ij = exp(−d_ij/ℓ)
  Jammed agents (close together) turn red; open-room agents recover green.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.widgets as widgets
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfm_smooth import (
    step, check_exit, _settle,
    ROOM_W, ROOM_H, EXIT_W,
    R_LO, R_HI, DT,
    B_eff, d_eq, kappa_scale, shape_sq,
)

# ── stress → v0 ───────────────────────────────────────────────
V0_MIN = 0.3
V0_MAX = 4.0

def stress_to_v0(s):
    return V0_MIN + s * (V0_MAX - V0_MIN)

# ── SIS parameters ────────────────────────────────────────────
MU_S  = 0.05    # recovery rate (fixed)
ELL_S = 1.5     # contagion range in metres (fixed)

def kappa_of_sigma(s):
    """Contagion coefficient scales with collective stress."""
    return 0.005 + s * 0.045   # 0.005 at calm, 0.05 at full panic

def sis_step(beta, pos, kappa_s, dt):
    n = len(pos)
    if n < 2:
        return np.clip(beta * (1.0 - MU_S * dt), 0.0, 1.0)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    W = np.exp(-d / ELL_S)
    np.fill_diagonal(W, 0.0)
    dbeta = -MU_S * beta + kappa_s * (W @ beta) * (1.0 - beta)
    return np.clip(beta + dbeta * dt, 0.0, 1.0)

# ── mutable state ─────────────────────────────────────────────
N      = 60
sigma  = 0.2
door_w = EXIT_W
t_end  = 120.0

def make_grid(n, dw):
    rng     = np.random.default_rng(42)
    radii   = rng.uniform(R_LO, R_HI, n)
    spacing = 0.62; margin = 0.45
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
    global positions, velocities, radii, beta_all, ey_lo, ey_hi
    global t, n_out_total, N
    pass  # rng is created fresh inside make_grid
    ey_lo = ROOM_H / 2 - door_w / 2
    ey_hi = ROOM_H / 2 + door_w / 2
    positions, velocities, radii = make_grid(N, door_w)
    N = len(positions)
    # All agents start at σ with tiny spatial noise to seed heterogeneity
    noise    = np.random.default_rng(7).normal(0, 0.03, N)
    beta_all = np.clip(sigma + noise, 0.0, 1.0)
    t = 0.0; n_out_total = 0

fresh_state()

# ── figure (no side plots — room only) ───────────────────────
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a2e')

ax = fig.add_axes([0.06, 0.18, 0.78, 0.78])
ax.set_facecolor('#16213e')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_edgecolor('#444466')
ax.set_xlabel('x (m)', color='white', fontsize=11)
ax.set_ylabel('y (m)', color='white', fontsize=11)

# Room walls
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

# Regime label
regime_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                      fontsize=11, va='top', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='#0a0a1e', alpha=0.7))

def regime_info(s):
    de = d_eq(stress_to_v0(s))
    if de >= 0.60:
        return 'LOW STRESS  —  orderly queue', '#00ff88'
    elif de >= 0.35:
        return 'MODERATE STRESS  —  some contact', '#ffcc44'
    else:
        return 'HIGH STRESS  —  arch jams at exit', '#ff4466'

title_text = ax.set_title('', color='white', fontsize=12, fontweight='bold')

# Agent scatter — colour by individual SIS stress β_i
stress_cmap = LinearSegmentedColormap.from_list(
    'stress', ['#22dd88', '#ffcc00', '#ff2244'])
norm     = Normalize(vmin=0.0, vmax=1.0)
MARKER_S = 520
scat = ax.scatter([], [], c=[], s=MARKER_S, zorder=5,
                  cmap=stress_cmap, norm=norm,
                  edgecolors='white', linewidths=0.3)
sm = ScalarMappable(cmap=stress_cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('stress  β', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# ── sliders (same layout as app_smooth) ──────────────────────
sl_y0, sl_h = 0.10, 0.030
ax_sg = fig.add_axes([0.07, sl_y0,         0.60, sl_h])
ax_dw = fig.add_axes([0.07, sl_y0 - 0.052, 0.27, sl_h])
ax_nn = fig.add_axes([0.43, sl_y0 - 0.052, 0.27, sl_h])
for aax in [ax_sg, ax_dw, ax_nn]: aax.set_facecolor('#222244')

sl_sg = widgets.Slider(ax_sg, 'Collective stress  σ',
                        0.0, 1.0, valinit=sigma,  valstep=0.02, color='#cc4455')
sl_dw = widgets.Slider(ax_dw, 'Door (m)',
                        0.5, 3.0, valinit=door_w, valstep=0.1,  color='#aa5577')
sl_nn = widgets.Slider(ax_nn, 'N agents',
                        10,  120, valinit=N,       valstep=5,    color='#aa8833')
for sl in [sl_sg, sl_dw, sl_nn]:
    sl.label.set_color('white'); sl.valtext.set_color('white')

def on_sg(val):
    global sigma
    sigma = sl_sg.val
    label, color = regime_info(sigma)
    regime_text.set_text(label); regime_text.set_color(color)

def on_dw(val):
    global door_w
    door_w = sl_dw.val; do_restart()

def on_nn(val):
    global N
    N = int(sl_nn.val); do_restart()

sl_sg.on_changed(on_sg)
sl_dw.on_changed(on_dw)
sl_nn.on_changed(on_nn)

# ── restart button ────────────────────────────────────────────
ax_btn = fig.add_axes([0.80, 0.05, 0.10, 0.05])
btn = widgets.Button(ax_btn, 'Restart', color='#2a2a4a', hovercolor='#4444aa')
btn.label.set_color('white'); btn.label.set_fontsize(10)

def do_restart():
    fresh_state()
    wall_lo.set_ydata([0, ey_lo]); wall_hi.set_ydata([ey_hi, ROOM_H])
    exit_glow.set_y(ey_lo); exit_glow.set_height(ey_hi - ey_lo)
    scat.set_offsets(positions); scat.set_array(beta_all)
    label, color = regime_info(sigma)
    regime_text.set_text(label); regime_text.set_color(color)
    title_text.set_text(f'Restarted  —  N={N}  door={door_w:.1f}m  sigma={sigma:.2f}')
    fig.canvas.draw_idle()

btn.on_clicked(lambda e: do_restart())

# ── animation ─────────────────────────────────────────────────
def update(frame):
    global positions, velocities, radii, beta_all, t, n_out_total

    if len(positions) == 0 or t >= t_end:
        return scat, title_text

    # 1. SIS stress update — coloring only
    kappa_s  = kappa_of_sigma(sigma)
    beta_all = sis_step(beta_all, positions, kappa_s, DT)

    # 2. Physics — identical to app_smooth
    v0 = stress_to_v0(sigma)
    positions, velocities = step(
        positions, velocities, radii, v0,
        ROOM_W, ROOM_H, ey_lo, ey_hi, DT)
    t += DT

    # 3. Remove exited agents; mask beta too
    ex = check_exit(positions)
    if ex.any():
        n_out_total += int(ex.sum())
        stay       = ~ex
        positions  = positions[stay]
        velocities = velocities[stay]
        radii      = radii[stay]
        beta_all   = beta_all[stay]

    if len(positions) > 0:
        scat.set_offsets(positions)
        scat.set_array(beta_all)
    else:
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))

    mean_b = float(beta_all.mean()) if len(beta_all) > 0 else 0.0
    label, _ = regime_info(sigma)
    title_text.set_text(
        f't={t:.1f}s   in:{len(positions)}  out:{n_out_total}   '
        f'sigma={sigma:.2f}  beta_mean={mean_b:.2f}   {label}')
    return scat, title_text

ani = animation.FuncAnimation(fig, update, interval=30, blit=False,
                               cache_frame_data=False, repeat=False)
do_restart()
plt.show()
