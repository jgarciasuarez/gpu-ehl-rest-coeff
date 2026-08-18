#!/usr/bin/env python3
"""Profile a single GPU run and capture a JAX XPlane trace."""
import json
import shutil
import time
from pathlib import Path

import jax

from gpu_ehl.elasticity import load_or_build_kernel
from gpu_ehl.geometry import make_radial_grid
from gpu_ehl.solver import simulate_single_setup

# Target case
STOKES = 1000.0
EXPONENT = 2.0
NR = 3000
LR = 6.0
DELTA_T = 0.5e-3
NT = 10000
NT_INI = 200
NUM_SNAPSHOTS = 100

OUT_DIR = Path("profiling")
OUT_DIR.mkdir(exist_ok=True, parents=True)
JAX_TRACE_DIR = OUT_DIR / "jax_trace_baseline"
SUMMARY_FILE = OUT_DIR / "baseline_summary.json"


def main() -> None:
    device = jax.devices()[0]
    print(f"Device: {device}")

    r = make_radial_grid(NR, LR)
    dr = float(r[1] - r[0])
    print("Loading/building elastic kernel...")
    kernel = load_or_build_kernel(NR, LR)

    # Clean any previous trace so start_trace does not complain
    if JAX_TRACE_DIR.exists():
        shutil.rmtree(JAX_TRACE_DIR)

    print("Starting JAX profiler trace...")
    jax.profiler.start_trace(str(JAX_TRACE_DIR))

    t0 = time.perf_counter()
    history = simulate_single_setup(
        stokes=STOKES,
        exponent=EXPONENT,
        KernelMatrix=kernel,
        r=r,
        delta_t=DELTA_T,
        dr=dr,
        nt=NT,
        nt_ini=NT_INI,
        num_snapshots=NUM_SNAPSHOTS,
    )
    history = jax.block_until_ready(history)
    wall_time = time.perf_counter() - t0

    jax.profiler.stop_trace()

    summary = {
        "case": {"Stokes": STOKES, "exponent": EXPONENT, "NR": NR, "NT": NT},
        "device": str(device),
        "wall_time_seconds": wall_time,
        "initial_velocity": float(history["V"][0]),
        "final_velocity": float(history["V"][-1]),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"Wall time: {wall_time:.3f}s")
    print(f"JAX trace dir: {JAX_TRACE_DIR}")
    print(f"Summary file: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
