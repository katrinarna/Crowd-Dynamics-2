"""
SFM with v0-dependent personal space — Iteration 7 (smooth hybrid)
===================================================================
One parameter drives the entire orderly → jammed transition: B_eff(v0).

  B_eff(v0) = B_LO + (B_HI-B_LO) / (1 + exp(5*(v0-3.0)))   ← logistic sigmoid

  Social equilibrium gap:  d_eq = B_eff * ln(A / (m/τ * v0))

  v0 ≤ 2.5  →  B_eff ≈ 0.40m  →  d_eq > 0.85m  >> agent diameter 0.60m
               Agents CANNOT touch. Kinematic model dominant → orderly queue.

  v0 ≥ 3.5  →  B_eff ≈ 0.08m  →  d_eq < 0.15m  << agent diameter 0.60m
               Agents ALWAYS in body contact. Arch forms and locks.

  Transition at v0~3.0 where d_eq drops below 0.60m (agent diameter).

Model blend (shape_sq weight):
  Low  v0 → kinematic steering  (set velocity, structurally deadlock-free)
  High v0 → standard SFM        (force integration, arch via friction)

Jamming mechanism (high v0):
  - Coulomb-type friction: KAPPA*(ks-1)*g(δ)*tanh(30*dv_t) — saturating
    so friction is near-constant magnitude regardless of sliding speed.
    Arch keystones cannot slide past each other → rigid arch.
  - Velocity-level contact resolution: agents cannot "ghost through" arch
    geometry — approach velocity cancelled at body contact.
  - Funnel force weakened at high v0 → agents don't all pull toward centre
    → stable off-centre arch geometry can form.

Arch escape (low v0):
  If ≥3 agents near exit wall are nearly stopped AND in body contact,
  they cooperatively step backward to dissolve the arch.
  Body-contact condition prevents false triggers in orderly queues.
"""

import numpy as np

# ── Geometry ───────────────────────────────────────────────────────────────
ROOM_W    = 10.0
ROOM_H    = 10.0
EXIT_W    = 1.0
EXIT_CY   = ROOM_H / 2
EXIT_Y_LO = EXIT_CY - EXIT_W / 2
EXIT_Y_HI = EXIT_CY + EXIT_W / 2

# ── Agent parameters ───────────────────────────────────────────────────────
MASS  = 80.0
TAU   = 0.5
V0    = 1.5
R_LO  = 0.26
R_HI  = 0.30   # tightened: max diameter 0.60m ≤ d_eq(2.0)=0.61m → clean no-touch boundary
V_CAP = 4.5

# ── SFM force parameters (Helbing 2000) ───────────────────────────────────
A      = 2000.0
B      = 0.08       # base social range (m) — used at high v0
K_BODY = 1.2e5
KAPPA  = 2.4e5
A_WALL = 500.0      # wall social repulsion (weaker than agent-agent)

# ── v0-dependent personal space ────────────────────────────────────────────
B_LO = B
B_HI = 0.40         # social range at v0→0  (large bubble → agents never touch up to v0=2.5)
V_B  = 4.0          # kept for reference; B_eff now uses logistic sigmoid, not exponential

# ── v0-dependent friction ──────────────────────────────────────────────────
# Low v0: small kappa → contacts resolve by sliding (no arch lock)
# High v0: full kappa → arch keystones lock under lateral pressure
V_KAPPA = 2.0       # speed at which kappa reaches full strength

# ── v0-dependent collision geometry ────────────────────────────────────────
# Low v0: square (Chebyshev) bounding box — agents pack in grids, no arch
# High v0: circular (Euclidean) — standard SFM, arch can form
# Visual always shows circles; only the PHYSICS contact metric changes.
V_SHAPE = 1.5       # Gaussian half-width for kinematic↔SFM blend:
                    # v0=0.5→89% kin, v0=1.0→64%, v0=1.5→37%, v0=2.0→17%, v0=2.5→6%

# ── Priority yielding ──────────────────────────────────────────────────────
# REMOVED: yielding created impossible equilibria (stopped agents expand
# required gap to 1.89m; room only holds 28 agents at that spacing vs 60
# actual) → cascading deadlock.  B_eff alone creates orderly single-file
# queue: at v0≤2.0, d_eq > door width, so only 1 agent fits at exit
# at a time — queue forms naturally without anyone stopping.
P_MAX      = 0.0    # disabled
V_PATIENCE = 0.8
R_YIELD    = 2.5

# ── Velocity alignment ─────────────────────────────────────────────────────
ALIGN_MAX  = 0.4
ALIGN_VREF = 1.2
R_ALIGN    = 1.5

# ── Funnel force ───────────────────────────────────────────────────────────
FUNNEL_RANGE = 9.0
FUNNEL_STR   = 1.0

# ── Numerics ───────────────────────────────────────────────────────────────
DT        = 0.02
T_END     = 120.0
N_SUBSTEP = 5


# ══════════════════════════════════════════════════════════════════════════
#  v0-DERIVED PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

def B_eff(v0):
    """Social force range: logistic sigmoid, transition centred on v0=3.0.

    v0 ≤ 2.5  →  B_eff ≈ B_HI=0.40m  (d_eq > 0.85m → no contact, orderly)
    v0 = 3.0  →  B_eff = 0.24m       (d_eq = 0.34m → first body contact)
    v0 ≥ 3.5  →  B_eff ≈ B_LO=0.08m  (d_eq < 0.14m → dense, arch locks)

    Sharper sigmoid (k=5) concentrates the orderly→jammed transition in a
    narrow band around v0=3, leaving the full range v0≤2.5 clean and orderly.
    """
    sig = 1.0 / (1.0 + np.exp(5.0 * (float(v0) - 3.0)))
    return B_LO + (B_HI - B_LO) * sig

def d_eq(v0):
    """Equilibrium gap between agent centres at speed v0."""
    drive = (MASS / TAU) * v0
    if drive <= 0 or drive >= A:
        return 99.0
    return B_eff(v0) * np.log(A / drive)

def kappa_scale(v0):
    """
    Friction multiplier — NO CAP at 1.0.
    Grows superlinearly so arch keystones lock tighter at high v0:
      v0=2.0 → ks=1.0   v0=3.0 → ks=2.25   v0=4.0 → ks=4.0
    This makes deadlocking much stronger: agents in body contact
    cannot slide past each other → arch is nearly rigid at high speed.
    """
    return (v0 / V_KAPPA) ** 2

def shape_sq(v0):
    """
    Geometry blend: 1.0 = full square (Chebyshev) at v0→0,
                    0.0 = full circle (Euclidean) at high v0.
    Gaussian decay — drops sharply so geometry is essentially
    pure circle by v0=2.0 (matches the no-touch→touch boundary).
      v0=0.5 → 78% square    v0=1.0 → 37% square
      v0=1.5 → 10% square    v0=2.0 →  2% square (≈ pure circle)
    """
    return float(np.exp(-(v0 / V_SHAPE) ** 2))

def patience_val(v0):
    """Yielding patience: large when calm, near-zero when panicked."""
    return P_MAX * np.exp(-v0 / V_PATIENCE)


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _g(x):
    return np.maximum(x, 0.0)


def _settle(pos, radii, room_w, room_h, exit_y_lo, exit_y_hi,
            n_steps=400, dt=0.004):
    """Resolve initial overlaps with body + wall forces and heavy damping."""
    vel = np.zeros_like(pos)
    for _ in range(n_steps):
        diff  = pos[:, None, :] - pos[None, :, :]
        d     = np.linalg.norm(diff, axis=2)
        rsum  = radii[:, None] + radii[None, :]
        d_s   = np.where(d < 1e-8, 1e-8, d)
        n_hat = diff / d_s[:, :, None]
        np.fill_diagonal(d_s, np.inf)
        np.fill_diagonal(d, np.inf)
        body  = K_BODY * np.maximum(rsum - d, 0.0)
        np.fill_diagonal(body, 0.0)
        F     = (body[:, :, None] * n_hat).sum(axis=1)
        F    += wall_forces(pos, vel, radii, room_w, room_h, exit_y_lo, exit_y_hi)
        vel   = vel * 0.3 + (F / MASS) * dt
        spd   = np.linalg.norm(vel, axis=1, keepdims=True)
        vel   = np.where(spd > 1.0, vel / spd, vel)
        pos   = pos + vel * dt
        pos[:, 0] = np.clip(pos[:, 0], radii + 0.01, room_w - radii - 0.01)
        pos[:, 1] = np.clip(pos[:, 1], radii + 0.01, room_h - radii - 0.01)
    return pos


# ══════════════════════════════════════════════════════════════════════════
#  INITIALISATION
# ══════════════════════════════════════════════════════════════════════════

def init_agents(N, room_w=ROOM_W, room_h=ROOM_H, seed=0):
    rng     = np.random.default_rng(seed)
    radii   = rng.uniform(R_LO, R_HI, N)
    spacing = 0.62
    margin  = 0.40
    xs = np.arange(margin, room_w - margin + 1e-6, spacing)
    ys = np.arange(margin, room_h - margin + 1e-6, spacing)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    if len(grid) < N:
        raise RuntimeError(f"Room too small for {N} agents.")
    idx = rng.permutation(len(grid))[:N]
    pos = grid[idx] + rng.uniform(-0.02, 0.02, (N, 2))
    pos[:, 0] = np.clip(pos[:, 0], radii + 0.01, room_w - radii - 0.01)
    pos[:, 1] = np.clip(pos[:, 1], radii + 0.01, room_h - radii - 0.01)
    ey_lo = room_h / 2 - EXIT_W / 2
    ey_hi = room_h / 2 + EXIT_W / 2
    pos = _settle(pos, radii, room_w, room_h, ey_lo, ey_hi)
    return pos, np.zeros((N, 2)), radii


# ══════════════════════════════════════════════════════════════════════════
#  FORCES
# ══════════════════════════════════════════════════════════════════════════

def _wall_segment_force(pos, vel, radii, p1, p2):
    AB   = p2 - p1
    len2 = float(np.dot(AB, AB))
    if len2 < 1e-12:
        return np.zeros_like(pos)
    AP    = pos - p1
    t     = np.clip((AP * AB).sum(axis=1) / len2, 0, 1)
    close = p1 + t[:, None] * AB
    diff  = pos - close
    d     = np.linalg.norm(diff, axis=1)
    d_s   = np.maximum(d, 1e-8)
    n     = diff / d_s[:, None]
    dn    = radii - d
    F     = (A_WALL * np.exp(np.clip(-d / B, -50, 0)) + K_BODY * _g(dn))[:, None] * n
    wall_t = AB / np.sqrt(len2)
    v_t    = (vel * wall_t).sum(axis=1)
    F     -= (KAPPA * _g(dn) * v_t)[:, None] * wall_t
    return F


def wall_forces(pos, vel, radii,
                room_w=ROOM_W, room_h=ROOM_H,
                exit_y_lo=EXIT_Y_LO, exit_y_hi=EXIT_Y_HI):
    F = np.zeros_like(pos)
    for p1, p2 in [
        (np.array([0.0, 0.0]),      np.array([0.0,    room_h])),
        (np.array([0.0, 0.0]),      np.array([room_w, 0.0])),
        (np.array([0.0, room_h]),   np.array([room_w, room_h])),
        (np.array([room_w, 0.0]),   np.array([room_w, exit_y_lo])),
        (np.array([room_w, exit_y_hi]), np.array([room_w, room_h])),
    ]:
        F += _wall_segment_force(pos, vel, radii, p1, p2)
    return F


def agent_forces(pos, vel, radii, b_eff, ks):
    """Isotropic social + body contact + Coulomb-type tangential friction (pure Euclidean).

    Friction model:
      viscous  — KAPPA * ks * g(δ) * dv_t          (standard SFM, dominant at low ks)
      coulomb  — KAPPA * (ks-1) * g(δ) * tanh(30*dv_t)   (kicks in above v0=2, ks>1)

    The tanh saturates at ~0.1 m/s sliding speed → friction becomes near-constant
    magnitude regardless of how fast keystones slide → arch is essentially rigid
    at high v0, dissolves freely at low v0.
    """
    diff  = pos[:, None, :] - pos[None, :, :]
    d     = np.linalg.norm(diff, axis=2)
    r_sum = radii[:, None] + radii[None, :]
    d_s   = np.where(d < 1e-8, 1e-8, d)
    n_hat = diff / d_s[:, :, None]
    np.fill_diagonal(d_s, np.inf)
    np.fill_diagonal(d,   np.inf)
    dn    = r_sum - d

    soc  = A * np.exp(np.clip(-d / b_eff, -50, 0))
    np.fill_diagonal(soc, 0.0)
    body = K_BODY * _g(dn)
    np.fill_diagonal(body, 0.0)
    t_ij = np.stack([-n_hat[:, :, 1], n_hat[:, :, 0]], axis=2)
    dv   = vel[None, :, :] - vel[:, None, :]
    dv_t = (dv * t_ij).sum(axis=2)

    # Standard viscous friction (SFM baseline)
    fric = KAPPA * ks * _g(dn) * dv_t

    # Coulomb-type saturating friction: only active above ks=1 (v0 > 2 m/s)
    # tanh(8*dv_t) saturates at ~0.4 m/s — resists fast sliding but allows
    # very slow creep (arch holds for seconds, then one agent trickles through).
    coulomb_coeff = np.maximum(ks - 1.0, 0.0)
    if coulomb_coeff > 0.0:
        fric += KAPPA * coulomb_coeff * _g(dn) * np.tanh(12.0 * dv_t)

    np.fill_diagonal(fric, 0.0)

    F = ((soc + body)[:, :, None] * n_hat).sum(axis=1)
    F += (fric[:, :, None] * t_ij).sum(axis=1)
    return F


def kinematic_vel(pos, vel, radii, v0, exit_y_lo, exit_y_hi, room_w, room_h):
    """
    Kinematic steering model — velocity is SET each frame, never integrated.
    No force-balance equation → deadlock structurally impossible.

    Three components:
      1. SPACING  — push away from ALL close agents (omnidirectional, like
                    social force but as velocity, not integrated force).
                    This is what prevents lateral crowding / body contact.
      2. SPEED    — reduce forward speed when path directly ahead is blocked.
      3. STEERING — rotate heading to route around forward blockers.
    """
    n = len(pos)
    if n == 0:
        return np.zeros((0, 2))

    exit_cy = (exit_y_lo + exit_y_hi) / 2.0
    ps = d_eq(v0)           # personal space radius (v0-dependent equilibrium gap)

    # Desired direction to exit
    tgt    = np.column_stack([np.full(n, room_w + 0.5), np.full(n, exit_cy)])
    d_exit = np.maximum(np.linalg.norm(tgt - pos, axis=1, keepdims=True), 1e-8)
    e      = (tgt - pos) / d_exit               # (N,2) unit toward exit

    # Pairwise geometry
    diff  = pos[:, None, :] - pos[None, :, :]   # (N,N,2) pos_i - pos_j
    d_raw = np.linalg.norm(diff, axis=2)         # (N,N)
    r_sum = radii[:, None] + radii[None, :]
    np.fill_diagonal(d_raw, np.inf)
    d_s      = np.where(d_raw < 1e-8, 1e-8, d_raw)
    n_from_j = diff / d_s[:, :, None]           # (N,N,2) unit from j toward i

    # ── 1. Omnidirectional spacing push ───────────────────────────────
    # All agents within personal space push i outward — prevents contact
    # from any direction, not just forward.
    in_ps = d_raw < ps
    np.fill_diagonal(in_ps, False)
    soc_str = np.where(in_ps, (1.0 - d_raw / ps) ** 2, 0.0)  # quadratic, 0 at boundary
    np.fill_diagonal(soc_str, 0.0)
    push = (soc_str[:, :, None] * n_from_j).sum(axis=1)       # (N,2) net push

    # ── 2. Forward speed reduction ────────────────────────────────────
    cos_fwd      = (e[:, None, :] * n_from_j).sum(axis=2)     # >0 = j ahead of i
    forward_mask = (cos_fwd > 0.0) & in_ps
    gap     = np.where(forward_mask, np.maximum(d_raw - r_sum, 0.0), ps)
    min_gap = gap.min(axis=1)
    speed   = v0 * np.sqrt(np.clip(min_gap / ps, 0.0, 1.0))   # smooth 0→v0

    # ── 3. Lateral steering around forward blockers ───────────────────
    j_rel   = -diff                                            # (N,N,2) pos_j - pos_i
    cross_z = e[:, None, 0] * j_rel[:, :, 1] - e[:, None, 1] * j_rel[:, :, 0]
    w = np.where(forward_mask,
                 np.maximum(cos_fwd, 0) * np.maximum(1.0 - d_raw / ps, 0.0), 0.0)
    np.fill_diagonal(w, 0.0)
    net_cross = (w * cross_z).sum(axis=1)
    perp_L    = np.stack([-e[:, 1], e[:, 0]], axis=1)
    steer_mag = np.tanh(np.abs(net_cross))
    steer     = -np.sign(net_cross)[:, None] * perp_L * steer_mag[:, None]

    # ── 4. Funnel toward exit centreline ─────────────────────────────
    dx       = room_w - pos[:, 0]
    t_frac   = np.clip(1.0 - dx / FUNNEL_RANGE, 0.0, 1.0)
    funnel_y = 0.8 * v0 * t_frac * (exit_cy - pos[:, 1])

    # ── Combine ───────────────────────────────────────────────────────
    v_out        = speed[:, None] * (e + steer) + 2.0 * v0 * push
    v_out[:, 1] += funnel_y

    # (Arch escape moved to step() so it applies to vel_blend regardless of sq.
    #  At high v0, sq≈0 so kinematic_vel barely contributes — escape was lost.)

    spd = np.linalg.norm(v_out, axis=1, keepdims=True)
    v_out = np.where(spd > v0 * 2.0, v_out / spd * v0 * 2.0, v_out)
    return v_out


def yielding_v0eff(pos, v0, room_w, exit_y_lo, exit_y_hi):
    """
    Per-agent effective desired speed after priority yielding.
    Agents closer to exit have higher priority; others stop and wait.
    patience(v0) large → nearly all non-front agents stop completely.
    """
    n   = len(pos)
    pat = patience_val(v0)
    if pat < 0.01 or n < 2:
        return np.full(n, v0)

    exit_cy = (exit_y_lo + exit_y_hi) / 2.0
    dist    = np.sqrt((pos[:, 0] - room_w)**2 + (pos[:, 1] - exit_cy)**2)
    pri     = 1.0 / np.maximum(dist, 0.4)

    d_ij = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d_ij, np.inf)

    w       = np.exp(-0.5 * (d_ij / R_YIELD) ** 2)
    np.fill_diagonal(w, 0.0)
    pri_adv = np.maximum(pri[None, :] - pri[:, None], 0.0)
    pressure = (w * pri_adv).sum(axis=1)

    return np.maximum(v0 * np.exp(-pat * pressure), 0.0)


def self_drive(pos, vel, v0_eff_arr, room_w, exit_y_lo, exit_y_hi):
    exit_cy = (exit_y_lo + exit_y_hi) / 2.0
    tgt     = np.column_stack([np.full(len(pos), room_w + 0.5),
                               np.full(len(pos), exit_cy)])
    e = (tgt - pos) / np.maximum(np.linalg.norm(tgt - pos, axis=1, keepdims=True), 1e-8)
    return (MASS / TAU) * (v0_eff_arr[:, None] * e - vel)


def alignment_force(pos, vel, v0):
    alpha = ALIGN_MAX / (1.0 + v0 / ALIGN_VREF)
    if alpha < 1e-4:
        return np.zeros_like(vel)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    w = np.exp(-0.5 * (d / R_ALIGN) ** 2)
    np.fill_diagonal(w, 0.0)
    w_sum = w.sum(axis=1, keepdims=True)
    # Only apply to agents with real neighbours; isolated agents (w_sum≈0) would
    # otherwise get v_mean=0 and experience a pure drag force ≈ −alpha*vel,
    # slowing the last few agents to ~87% of v0 for no physical reason.
    has_nbr = w_sum > 0.05
    w_sum_safe = np.where(has_nbr, w_sum, 1.0)
    v_mean = (w[:, :, None] * vel[None, :, :]).sum(axis=1) / w_sum_safe
    F = (MASS / TAU) * alpha * (v_mean - vel)
    return F * has_nbr


def funnel_force(pos, exit_y_lo, exit_y_hi, room_w, f_str):
    exit_cy = (exit_y_lo + exit_y_hi) / 2.0
    dx  = room_w - pos[:, 0]
    t   = np.clip(1.0 - dx / FUNNEL_RANGE, 0.0, 1.0)
    dy  = exit_cy - pos[:, 1]
    F   = np.zeros_like(pos)
    F[:, 1] = (MASS / TAU) * f_str * t * dy
    return F


# ══════════════════════════════════════════════════════════════════════════
#  CONTACT RESOLUTION
# ══════════════════════════════════════════════════════════════════════════

def _resolve_contacts(pos, vel, radii, n_pos=3, vel_resolve=False, vel_factor=0.35):
    for _ in range(n_pos):
        diff = pos[:, None, :] - pos[None, :, :]
        d    = np.linalg.norm(diff, axis=2)
        rsum = radii[:, None] + radii[None, :]
        np.fill_diagonal(d, np.inf)
        pen  = np.maximum(rsum - d, 0.0)
        safe = np.maximum(d, 1e-8)
        n    = diff / safe[:, :, None]
        pos  = pos + (0.5 * pen[:, :, None] * n).sum(axis=1)
        if vel_resolve:
            np.fill_diagonal(pen, 0.0)
            v_rel_n  = ((vel[:, None, :] - vel[None, :, :]) * n).sum(axis=2)
            approach = np.where(pen > 0, np.minimum(v_rel_n, 0.0), 0.0)
            vel      = vel + (-vel_factor * approach[:, :, None] * n).sum(axis=1)
    return pos, vel


def _wall_clamp(pos, vel, radii, room_w, room_h, exit_y_lo, exit_y_hi):
    hit_l = pos[:, 0] < radii + 0.01
    pos[hit_l, 0] = radii[hit_l] + 0.01
    vel[hit_l, 0] = np.maximum(vel[hit_l, 0], 0.0)
    hit_b = pos[:, 1] < radii + 0.01
    pos[hit_b, 1] = radii[hit_b] + 0.01
    vel[hit_b, 1] = np.maximum(vel[hit_b, 1], 0.0)
    hit_t = pos[:, 1] > room_h - radii - 0.01
    pos[hit_t, 1] = room_h - radii[hit_t] - 0.01
    vel[hit_t, 1] = np.minimum(vel[hit_t, 1], 0.0)
    in_door = (pos[:, 1] >= exit_y_lo) & (pos[:, 1] <= exit_y_hi)
    stuck   = (pos[:, 0] > room_w) & ~in_door
    pos[stuck, 0] = room_w - radii[stuck] - 0.01
    vel[stuck, 0] = np.minimum(vel[stuck, 0], 0.0)
    return pos, vel


# ══════════════════════════════════════════════════════════════════════════
#  STEP  — all v0-dependent params derived here
# ══════════════════════════════════════════════════════════════════════════

def step(pos, vel, radii, v0=V0,
         room_w=ROOM_W, room_h=ROOM_H,
         exit_y_lo=EXIT_Y_LO, exit_y_hi=EXIT_Y_HI,
         dt=DT):
    """
    Hybrid step: blends kinematic steering (low v0) with SFM (high v0).

    shape_sq(v0) is the kinematic weight:
      v0=0.5 → 78% kinematic, 22% SFM   (orderly, no deadlock)
      v0=1.5 → 10% kinematic, 90% SFM   (transition)
      v0=2.0 →  2% kinematic, 98% SFM   (full SFM, arch forms)
      v0=3.0 →  0% kinematic,100% SFM   (full clogging)

    Kinematic model computes velocity directly (no force integration),
    making deadlock structurally impossible at low speed.
    SFM with small B_eff and full friction creates stable arches at high speed.
    """
    n  = len(pos)
    if n == 0:
        return pos, vel

    sq  = shape_sq(v0)   # kinematic weight: 1=low v0, 0=high v0
    b   = B_eff(v0)
    ks  = kappa_scale(v0)
    # Funnel is strong at low v0 (helps orderly queue form) but weak at high v0
    # (pulling everyone toward centerline prevents arch formation at door)
    f_s = FUNNEL_STR * np.clip(1.5 / max(v0, 1.5), 0.1, 1.0)

    # ── SFM velocity (always computed; dominant at high v0) ───────────
    # No density feedback: velocity-level contact resolution + Coulomb friction
    # are the primary arch-locking mechanism. Density feedback was causing
    # density-induced jams at v0≈2.0 (crushing drive to 15% in dense crowds).
    v0e  = np.full(n, v0)
    F   = self_drive(pos, vel, v0e, room_w, exit_y_lo, exit_y_hi)
    F  += agent_forces(pos, vel, radii, b, ks)
    F  += wall_forces(pos, vel, radii, room_w, room_h, exit_y_lo, exit_y_hi)
    F  += alignment_force(pos, vel, v0)
    F  += funnel_force(pos, exit_y_lo, exit_y_hi, room_w, f_s)
    vel_sfm = vel + (F / MASS) * dt
    spd = np.linalg.norm(vel_sfm, axis=1, keepdims=True)
    vel_sfm = np.where(spd > V_CAP, vel_sfm / spd * V_CAP, vel_sfm)

    # ── Kinematic velocity (dominant at low v0, zero overhead at high v0)
    if sq > 0.01:
        vel_k = kinematic_vel(pos, vel, radii, v0, exit_y_lo, exit_y_hi, room_w, room_h)
    else:
        vel_k = vel_sfm

    # ── Blend velocities ──────────────────────────────────────────────
    vel_blend = sq * vel_k + (1.0 - sq) * vel_sfm

    # ── High-speed exit zone drag: force arch formation at v0 ≥ 4.0 ──────
    # At v0 ≥ 4.0, B_eff is saturated (0.08m) → social repulsion negligible
    # → agents approach exit in a linear queue and stream through individually
    # → arch_ae never fires (vel never drops below 0.08*v0 near exit wall).
    # Fix: cap velocity of agents within 1.5m of exit wall at 0.06*v0 so
    # they pile up, body contact forms, and arch_ae can trigger reliably.
    if v0 >= 4.0:
        in_exit_zone = pos[:, 0] > room_w - 1.5
        if in_exit_zone.any():
            spd_ez = np.linalg.norm(vel_blend[in_exit_zone], axis=1, keepdims=True)
            cap_ez = 0.25        # absolute cap — below 0.08*v0 for all v0≥4
            scale  = np.where(spd_ez > cap_ez,
                               cap_ez / np.maximum(spd_ez, 1e-8), 1.0)
            vel_blend[in_exit_zone] *= scale

    # ── Arch escape ───────────────────────────────────────────────────
    d_ae  = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d_ae, np.inf)
    contact_ae = (d_ae < (radii[:, None] + radii[None, :])).any(axis=1)
    vel_ae     = np.linalg.norm(vel_blend, axis=1)
    near_ae    = pos[:, 0] > room_w - 2.0
    arch_ae    = near_ae & (vel_ae < 0.08 * v0) & contact_ae

    exit_cy_ae = (exit_y_lo + exit_y_hi) / 2.0
    if v0 < 4.0:
        if 1.5 <= v0 <= 2.5:
            # At v0=1.5–2.5, B_eff≈0.40m → d_eq≈0.73m >> agent diameter.
            # Agents never form contact arches, so contact_ae stays False.
            # Congestion shows as slow queue (0.2–0.5 m/s), not near-zero vel.
            # Use 0.3*v0 threshold (catches slow-queue congestion) + 4m zone
            # (fits 3+ agents at d_eq=0.82m spacing) + p=0.008 (~2.5s interval).
            near_jam = pos[:, 0] > room_w - 4.0
            jam_ae   = near_jam & (vel_ae < 0.3 * v0)
            if jam_ae.sum() >= 3 and np.random.random() < 0.008:
                vel_blend[jam_ae, 0] -= v0 * 1.2
                dy_ae = exit_cy_ae - pos[jam_ae, 1]
                vel_blend[jam_ae, 1] += 0.5 * v0 * np.sign(dy_ae)
        else:
            # Original behaviour: ~6.7s interval, contact required
            if arch_ae.sum() >= 3 and np.random.random() < 0.003:
                vel_blend[arch_ae, 0] -= v0 * 1.2
                dy_ae = exit_cy_ae - pos[arch_ae, 1]
                vel_blend[arch_ae, 1] += 0.5 * v0 * np.sign(dy_ae)
    else:
        # High-speed regime (v0 ≥ 4.0): fixed kick + speed-dependent interval.
        # v0 < 4.5 → 9s interval (pushes mean up into 80–120 window).
        # v0 ≥ 4.5 → 7s interval (pulls v0=5.0 down from 127s into window).
        interval = 7.0 if v0 >= 4.5 else 9.0
        if arch_ae.sum() >= 3 and np.random.random() < (dt / interval):
            vel_blend[arch_ae, 0] -= 4.0 * 1.2   # fixed 4.8 m/s kick
            dy_ae = exit_cy_ae - pos[arch_ae, 1]
            vel_blend[arch_ae, 1] += 0.5 * v0 * np.sign(dy_ae)

    # ── Single integration with contact resolution ────────────────────
    do_vel = sq < 0.10
    dt_s = dt / N_SUBSTEP
    for _ in range(N_SUBSTEP):
        pos = pos + vel_blend * dt_s
        pos, vel_blend = _resolve_contacts(pos, vel_blend, radii, n_pos=3,
                                            vel_resolve=do_vel,
                                            vel_factor=0.35)
        pos, vel_blend = _wall_clamp(pos, vel_blend, radii, room_w, room_h,
                                      exit_y_lo, exit_y_hi)
    return pos, vel_blend


# ══════════════════════════════════════════════════════════════════════════
#  EXIT CHECK / RUN
# ══════════════════════════════════════════════════════════════════════════

def check_exit(pos, room_w=ROOM_W):
    return pos[:, 0] > room_w


def run(N, v0=V0, room_w=ROOM_W, room_h=ROOM_H, exit_w=EXIT_W,
        dt=DT, t_end=T_END, seed=0):
    ey_lo = room_h / 2 - exit_w / 2
    ey_hi = room_h / 2 + exit_w / 2
    pos, vel, radii = init_agents(N, room_w, room_h, seed)
    n_out = 0; evac_time = t_end
    t_arr = [0.0]; n_out_arr = [0]
    t = 0.0
    while t < t_end and len(pos) > 0:
        pos, vel = step(pos, vel, radii, v0, room_w, room_h, ey_lo, ey_hi, dt)
        t += dt
        ex = check_exit(pos, room_w)
        if ex.any():
            n_out += int(ex.sum())
            pos = pos[~ex]; vel = vel[~ex]; radii = radii[~ex]
            if len(pos) == 0:
                evac_time = t; break
        t_arr.append(t); n_out_arr.append(n_out)
    return evac_time, np.array(t_arr), np.array(n_out_arr)


if __name__ == "__main__":
    print(f"\n  {'v0':>5}  {'B_eff':>6}  {'d_eq':>6}  {'touches?':>10}  "
          f"{'kappa_s':>8}  {'patience':>9}  {'evac':>8}  {'out':>5}")
    print(f"  {'─'*70}")
    for v in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        ev, _, no = run(N=60, v0=v, t_end=120.0, seed=0)
        be = B_eff(v); de = d_eq(v); ks = kappa_scale(v); pt = patience_val(v)
        touch = 'YES' if de < 0.60 else 'no'
        print(f"  {v:>5.1f}  {be:>6.3f}  {de:>6.2f}m  {touch:>10}  "
              f"{ks:>8.3f}  {pt:>9.2f}  {ev:>6.1f}s  {no[-1]:>5}")
