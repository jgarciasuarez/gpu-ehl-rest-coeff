"""gpu_ehl: GPU-accelerated elastohydrodynamic lubrication solver.

Public API for running single simulations and parametric sweeps of power-law
elastic indenters in fluid-mediated impact.
"""

from gpu_ehl.config import SolverConfig
from gpu_ehl.elasticity import load_or_build_kernel
from gpu_ehl.geometry import make_radial_grid, make_time_grid
from gpu_ehl.io import load_results_h5, save_results_h5
from gpu_ehl.solver import simulate_single_setup
from gpu_ehl.sweep import make_parameter_grid, run_parametric_sweep

__all__ = [
    "SolverConfig",
    "load_or_build_kernel",
    "make_radial_grid",
    "make_time_grid",
    "simulate_single_setup",
    "run_parametric_sweep",
    "make_parameter_grid",
    "save_results_h5",
    "load_results_h5",
]
