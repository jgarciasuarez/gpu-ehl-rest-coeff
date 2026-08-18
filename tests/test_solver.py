"""Tests for the gpu_ehl solver on CPU with a coarse grid."""

import jax.numpy as jnp
import pytest

from gpu_ehl.elasticity import load_or_build_kernel
from gpu_ehl.geometry import make_radial_grid
from gpu_ehl.solver import simulate_single_setup


@pytest.fixture
def coarse_grid():
    """Return a coarse grid and kernel suitable for fast CPU tests."""
    nr = 64
    lr = 6.0
    r = make_radial_grid(nr, lr)
    dr = lr / nr
    kernel = load_or_build_kernel(nr, lr)
    return r, dr, kernel


def test_simulate_single_setup_runs(coarse_grid):
    """Check that a single simulation runs and returns expected shapes."""
    r, dr, kernel = coarse_grid
    nt = 50
    nt_ini = 10
    num_snapshots = 5

    history = simulate_single_setup(
        stokes=100.0,
        exponent=2.0,
        KernelMatrix=kernel,
        r=r,
        delta_t=0.5e-3,
        dr=dr,
        nt=nt,
        nt_ini=nt_ini,
        num_snapshots=num_snapshots,
    )

    assert history["Force"].shape == (num_snapshots,)
    assert history["V"].shape == (num_snapshots,)
    assert history["Central_h"].shape == (num_snapshots,)
    assert history["Min_h"].shape == (num_snapshots,)
    assert history["w"].shape == (num_snapshots, r.shape[0])
    assert history["p"].shape == (num_snapshots, r.shape[0])

    assert jnp.all(jnp.isfinite(history["Force"]))
    assert jnp.all(jnp.isfinite(history["Central_h"]))


def test_initial_approach_velocity(coarse_grid):
    """Smoke test for a different exponent value."""
    r, dr, kernel = coarse_grid
    history = simulate_single_setup(
        stokes=10.0,
        exponent=1.05,
        KernelMatrix=kernel,
        r=r,
        delta_t=0.5e-3,
        dr=dr,
        nt=20,
        nt_ini=5,
        num_snapshots=2,
    )

    assert history["Force"].shape == (2,)
    assert jnp.all(jnp.isfinite(history["Force"]))
