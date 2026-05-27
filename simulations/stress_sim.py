"""
Pure stress-propagation model
==============================
N agents at *fixed* 2-D positions.  No motion, no exits.

State variable: β_i(t) ∈ [0,1]  (individual stress level)

ODE
---
  β̇_i = −μ β_i  +  κ · [Σ_j K(d_ij) β_j] · (1 − β_i)
            ↑                   ↑
         recovery           contagion

Spatial kernel:   K(d) = exp(−d/ℓ)
Mean-field R₀:   R₀ = κ · ρ · 2πℓ² / μ   (ρ = N/L²)
Endemic level:   β* = 1 − 1/R₀  (when R₀ > 1, else β* = 0)
"""

import numpy as np


def make_positions(N, L, seed=0):
    """Uniform random positions in [0,L]²."""
    return np.random.default_rng(seed).uniform(0, L, (N, 2))


def spatial_kernel(pos, ell):
    """
    Symmetric N×N weight matrix  W_ij = exp(−d_ij/ℓ),  W_ii = 0.
    Pre-computing W once makes the time-stepping O(N²) per step.
    """
    diff = pos[:, None] - pos[None, :]          # (N, N, 2)
    d    = np.linalg.norm(diff, axis=2)         # (N, N)
    np.fill_diagonal(d, np.inf)
    W = np.exp(-d / ell)
    np.fill_diagonal(W, 0.0)
    return W


def stress_rhs(beta, W, mu, kappa):
    """Right-hand side: β̇ = −μβ + κ·(Wβ)·(1−β)."""
    return -mu * beta + kappa * (W @ beta) * (1.0 - beta)


# ── time integration ──────────────────────────────────────────────────────

def run(N, L, mu, kappa, ell, beta0=0.10, dt=0.05, t_end=300.0, seed=0):
    """
    Integrate the stress ODE and record mean and variance.

    Returns
    -------
    t_arr     : 1-D array (n_steps+1,)
    mean_beta : mean stress at each step
    var_beta  : variance of stress at each step
    beta_final: stress vector at t=t_end  (length N)
    """
    pos  = make_positions(N, L, seed)
    W    = spatial_kernel(pos, ell)

    beta = np.full(N, float(beta0)) if np.isscalar(beta0) else np.array(beta0, float)
    beta = np.clip(beta, 0.0, 1.0)

    n_steps   = int(t_end / dt)
    t_arr     = np.arange(n_steps + 1) * dt
    mean_beta = np.empty(n_steps + 1)
    var_beta  = np.empty(n_steps + 1)

    mean_beta[0] = beta.mean()
    var_beta[0]  = beta.var()

    for k in range(n_steps):
        beta = np.clip(beta + stress_rhs(beta, W, mu, kappa) * dt, 0.0, 1.0)
        mean_beta[k + 1] = beta.mean()
        var_beta[k + 1]  = beta.var()

    return t_arr, mean_beta, var_beta, beta


def run_snapshots(N, L, mu, kappa, ell, beta0=0.10, dt=0.05,
                  snap_times=(0,), seed=0):
    """
    Integrate and return β_i arrays at requested time points.

    Returns
    -------
    pos   : (N, 2) fixed positions
    snaps : dict  {t: beta_array}
    """
    pos  = make_positions(N, L, seed)
    W    = spatial_kernel(pos, ell)

    beta = np.full(N, float(beta0)) if np.isscalar(beta0) else np.array(beta0, float)
    beta = np.clip(beta, 0.0, 1.0)

    snap_steps = {round(t / dt): t for t in snap_times}
    max_step   = max(snap_steps.keys())
    snaps      = {}

    for k in range(max_step + 1):
        if k in snap_steps:
            snaps[snap_steps[k]] = beta.copy()
        if k < max_step:
            beta = np.clip(beta + stress_rhs(beta, W, mu, kappa) * dt, 0.0, 1.0)

    return pos, snaps
