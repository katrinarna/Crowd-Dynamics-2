# Crowd Dynamics and Evacuation Modeling

Agent-based evacuation simulations using the **Social Force Model** (Helbing et al.), implemented in Python with NumPy and Matplotlib.

## Simulations

| File | Description |
|------|-------------|
| `basics.ipynb` | Step-by-step introduction to the Social Force Model — single agent, two-agent repulsion, and a 20-person room |
| `basics2.py` | Early multi-agent prototype |
| `room_experiments.py` | Single room with parametric studies: crowd size, exit width, one vs two exits, exit position, and the "faster is slower" panic effect |
| `room_animated.py` | Polished animated visualization of room evacuation with live speed and agent-count panels |
| `faster_is_slower.py` | Side-by-side animation comparing normal walking vs panic, illustrating the faster-is-slower effect |
| `stadium_bottleneck.py` | Concert venue geometry with a two-stage bottleneck: seating area → corridor → exit |
| `office_fire.py` | Three-office building with a spreading fire; experiments on fire spread rate vs evacuation time and survivor count |

## Model

The simulations implement the Social Force Model with three force components:

- **Self-driving force** — each agent accelerates toward their desired speed and target
- **Agent repulsion** — psychological and physical (compression + sliding friction) forces between agents
- **Wall forces** — repulsion from walls and obstacles, with gaps for doors and exits

## Requirements

```
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

## Sources

Reference PDFs are in the `sources/` folder.