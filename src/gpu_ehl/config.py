"""Configuration dataclass for the gpu_ehl solver."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfig:
    """Parameters for a single gpu_ehl simulation.

    Attributes
    ----------
    exponent : float
        Power-law shape exponent of the indenter profile.
    stokes : float
        Stokes number for the impact.
    nr : int, optional
        Number of radial grid points (default: 3000).
    lr : float, optional
        Outer radius of the computational domain (default: 6.0).
    nt : int, optional
        Number of time steps in the dynamic phase (default: 10000).
    nt_ini : int, optional
        Number of time steps in the prescribed initialization phase
        (default: 200).
    delta_t : float, optional
        Time step size (default: 0.5e-3).
    num_snapshots : int, optional
        Number of snapshots to store per simulation (default: 100).
    """

    exponent: float
    stokes: float
    nr: int = 3000
    lr: float = 6.0
    nt: int = 10000
    nt_ini: int = 200
    delta_t: float = 0.5e-3
    num_snapshots: int = 100
