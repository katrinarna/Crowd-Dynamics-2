"""
Render matplotlib animations from the simulation scripts to MP4 for the website.

Usage:
    python render_animations.py stress_propagation
    python render_animations.py --all

Each named target points to a sim file in ../../simulations/ and writes to
../public/animations/<name>.mp4. The simulations themselves are not modified;
we run them in a controlled namespace with the Agg backend so plt.show() is a
no-op, then save the `ani` variable they define.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set non-interactive backend BEFORE anything imports pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SIM_DIR = (HERE / ".." / ".." / "simulations").resolve()
OUT_DIR = (HERE / ".." / "public" / "animations").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map render target -> (sim file, animation variable name, output filename, fps)
TARGETS = {
    "stress_propagation":      ("stress_propagation.py",      "ani", "stress_propagation.mp4",      30),
    "stress_contagion_sfm":    ("stress_contagion_sfm.py",    "ani", "stress_contagion_sfm.mp4",    30),
    "stress_comparison_sfm":   ("stress_comparison_sfm.py",   "ani", "stress_comparison_sfm.mp4",   30),
    "fis_stress_comparison":   ("fis_stress_comparison.py",   "ani", "fis_stress_comparison.mp4",   30),
    "fis_analysis":            ("fis_analysis.py",            "ani", "fis_analysis.mp4",            30),
}


def render(name: str) -> Path:
    if name not in TARGETS:
        raise SystemExit(f"Unknown target '{name}'. Known: {', '.join(TARGETS)}")
    sim_file, ani_var, out_name, fps = TARGETS[name]
    sim_path = SIM_DIR / sim_file
    if not sim_path.exists():
        raise SystemExit(f"Missing simulation file: {sim_path}")
    out_path = OUT_DIR / out_name

    # Run sim in a namespace where plt.show is a no-op (we already set Agg)
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    namespace: dict = {"__name__": "__main__", "__file__": str(sim_path)}

    print(f"[{name}] running simulation: {sim_path.name}")
    t0 = time.time()
    try:
        # Make the sim's own directory the cwd so any relative paths it
        # writes (e.g. PNGs) land next to it, not in this folder.
        cwd_before = os.getcwd()
        os.chdir(sim_path.parent)
        with sim_path.open("r") as fh:
            code = compile(fh.read(), str(sim_path), "exec")
        exec(code, namespace)
    finally:
        os.chdir(cwd_before)
        plt.show = original_show
    print(f"[{name}] simulation done in {time.time()-t0:.1f}s")

    if ani_var not in namespace:
        raise SystemExit(
            f"[{name}] expected variable '{ani_var}' in {sim_file} after exec. "
            f"Found: {[k for k in namespace if not k.startswith('_')][:30]}"
        )
    ani = namespace[ani_var]

    print(f"[{name}] saving MP4 -> {out_path}")
    t0 = time.time()
    ani.save(
        str(out_path),
        writer="ffmpeg",
        fps=fps,
        dpi=120,
        bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p"],  # Safari / iOS friendly
    )
    print(f"[{name}] wrote {out_path.name} in {time.time()-t0:.1f}s "
          f"({out_path.stat().st_size/1e6:.1f} MB)")

    # Close all figures the sim opened
    plt.close("all")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", nargs="?", help="Render target name (see TARGETS).")
    p.add_argument("--all", action="store_true", help="Render every target.")
    p.add_argument("--list", action="store_true", help="List available targets.")
    args = p.parse_args()

    if args.list:
        for k, (sim, _, out, fps) in TARGETS.items():
            print(f"  {k:28} sim={sim:32} out={out} fps={fps}")
        return

    targets = list(TARGETS) if args.all else [args.target] if args.target else None
    if not targets:
        p.error("Pass a target name, --all, or --list.")
    for t in targets:
        render(t)


if __name__ == "__main__":
    sys.exit(main())
