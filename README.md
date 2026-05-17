# Crowd Dynamics and Evacuation Modeling

Agent-based evacuation simulations using the **Social Force Model** (Helbing et al.),
extended with per-agent stress contagion and faster-is-slower (FIS) analysis.
All simulations are implemented in Python with NumPy and Matplotlib.

---

## Simulation files

### Foundations

| File | What it does |
|------|-------------|
| `simulations/basics.ipynb` | Step-by-step introduction to the Social Force Model: single-agent motion, two-agent repulsion, small room evacuation |
| `simulations/basics2.py` | Early multi-agent prototype — room with several agents and basic force model |
| `simulations/room_experiments.py` | Parametric studies: crowd size, exit width, one vs two exits, exit position |
| `simulations/room_animated.py` | Polished animated room evacuation with live speed and agent-count panels |
| `simulations/stadium_bottleneck.py` | Concert-venue geometry with a two-stage bottleneck: seating → corridor → exit |
| `simulations/office_fire.py` | Three-office building with a spreading fire; measures survivor count vs fire spread rate |

### Faster-is-slower (FIS)

| File | What it does |
|------|-------------|
| `simulations/fis_analysis.py` | Clean FIS implementation (fixed physics: DT=0.01, R=0.3 m, 0.9 m exit, ABS_V_MAX=3.5). Produces two figures: (1) evacuation time vs desired speed — U-shaped curve with FIS onset; (2) density sweep showing the U-shape shifts left as the room gets more crowded |

### Stress contagion — mean-field theory

| File | What it does |
|------|-------------|
| `simulations/ssfm_stress.py` | Derives and plots the mean-field stress model: quadratic and cubic ODEs for crowd-average β(t), bifurcation diagrams, and equilibrium stress vs contagion strength κ |
| `simulations/evacuation_vs_stress.py` | Sweeps contagion strength κ, runs full SSFM simulations at each value, and plots the resulting chain: κ → β̄ → v₀ → clogging parameter α — tracing the FIS mechanism analytically |

### Stress contagion — agent-based simulations

| File | What it does |
|------|-------------|
| `simulations/stress_contagion_sfm.py` | Single-seed stress cascade animation. One agent starts panicked (β=1) at the crowd centre; nearby agents who observe fast movement become stressed, speed up, and propagate panic outward. Two-panel figure: room animation + mean β(t) time series |
| `simulations/stress_comparison_sfm.py` | Side-by-side comparison of four contagion rates (λ = 0, 1.5, 3, 4). Shows how the cascade strength and evacuation time change with λ — smooth SFM, same seed each panel |
| `simulations/fis_stress_comparison.py` | Four-panel animation comparing λ = 0, 2, 4, 6 with a multi-seed statistical sweep. Bar chart shows mean ± std evacuation time; higher λ raises deadlock probability, demonstrating FIS via stress |
| `simulations/stress_propagation.py` | **Stress wave visualization.** Seed agent (★) placed at the back of the room (far from exit). Shows how panic spreads as a wave toward the exit — agent colour = stress β, dashed ring = observation radius, two time-series panels track mean β and stressed-agent count |

### Optimal contagion rate

| File | What it does |
|------|-------------|
| `simulations/stress_lambda_sweep.py` | Sweeps urgency rate λ from 0 to 6 and plots the resulting **U-shaped evacuation time curve**. Too little urgency → slow (V_MIN = 1 m/s); optimal λ ≈ 0.9 → fastest (~21 s); too much → panic at exit → KAPPA arch formation → FIS → slower. Second panel shows deadlock probability rising on the right side of the U |

---

## Physics model

All simulations share the same core Social Force Model:

```
m·v̇_i = (m/τ)(v₀_i · ê_i − v_i)   ← self-driving toward target
        + Σ_j A·exp((2r−d_ij)/B)·n̂_ij   ← social/physical repulsion
        + κ·g(d_ij)·Δv_t·t̂_ij            ← tangential friction (FIS files)
        + wall forces
```

Stress dynamics (agent-based contagion files):

```
β̇_i = −µ·β_i  +  λ · signal_i · (1 − β_i)
v₀_i = V_MIN + β_i · (V_MAX − V_MIN)
```

The `signal_i` term varies by file:
- **Speed-based** (`stress_contagion_sfm.py`, `stress_propagation.py`): mean excess speed of neighbours within R_STRESS
- **Exit-proximity** (`stress_lambda_sweep.py`): how close agent i is to the exit wall

The **faster-is-slower** mechanism in `fis_analysis.py` and `stress_lambda_sweep.py`:
agents with `v₀ > ABS_V_MAX` (= 3.5 m/s) cannot move faster but push harder
→ excess crowd pressure at the bottleneck → tangential friction (κ = 240 000 N·s/m)
locks agents into stable arches → throughput drops.

---

## Requirements

```
numpy
matplotlib
scipy        # ssfm_stress.py only
jupyter      # basics.ipynb only
```

Install:
```bash
pip install -r requirements.txt
```

---

## Running a simulation

All `.py` files run stand-alone and open an interactive Matplotlib window:
```bash
cd simulations
python stress_propagation.py       # stress wave animation
python fis_analysis.py             # FIS U-curve + density sweep
python stress_lambda_sweep.py      # optimal λ sweep (takes ~5 min)
```

Files that produce long sweeps (`fis_analysis.py`, `stress_lambda_sweep.py`,
`fis_stress_comparison.py`, `evacuation_vs_stress.py`) print progress to the
terminal while running.

---

## Repository layout

```
.
├── simulations/          # all Python simulations and notebooks
├── sources/              # reference PDFs
├── stress_contagion_crowd.pdf   # mathematical writeup of the stress model
├── simplified_sfm.pdf           # SFM derivation notes
└── README.md
```
