"""Tests for gpu_ehl HDF5 input/output."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpu_ehl.io import load_results_h5, save_results_h5


def _make_minimal_results(num_cases: int, num_snapshots: int, nr: int) -> dict:
    """Build a results dict matching the expected sweep layout."""
    shape = (num_cases, num_snapshots)
    w_shape = (num_cases, num_snapshots, nr)
    return {
        "Force": jnp.ones(shape, dtype=jnp.float32),
        "EnergyElastic": jnp.full(shape, 0.1, dtype=jnp.float32),
        "Central_h": jnp.linspace(1.0, 0.5, num_snapshots, dtype=jnp.float32)[
            None, :
        ].repeat(num_cases, axis=0),
        "Min_h": jnp.full(shape, 0.3, dtype=jnp.float32),
        "D": jnp.full(shape, 1.0, dtype=jnp.float32),
        "V": jnp.full(shape, -1.0, dtype=jnp.float32),
        "w": jnp.zeros(w_shape, dtype=jnp.float32),
        "p": jnp.zeros(w_shape, dtype=jnp.float32),
    }


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Save a minimal sweep result and reload it faithfully."""
    nr = 32
    num_snapshots = 3
    nt = 40
    num_cases = 2

    r_grid = jnp.linspace(0.0, 6.0, nr, dtype=jnp.float32)
    time_grid = jnp.linspace(0.0, 1.0, nt + 1, dtype=jnp.float32)
    results = _make_minimal_results(num_cases, num_snapshots, nr)

    S_flat = jnp.array([100.0, 1000.0], dtype=jnp.float32)
    E_flat = jnp.array([2.0, 2.0], dtype=jnp.float32)

    out_path = tmp_path / "roundtrip.h5"
    save_results_h5(
        out_path,
        results,
        r_grid,
        time_grid,
        S_flat,
        E_flat,
        num_snapshots=num_snapshots,
    )

    data = load_results_h5(out_path)

    np.testing.assert_array_equal(data["r_grid"], np.asarray(r_grid))
    np.testing.assert_array_equal(data["time_grid"], np.asarray(time_grid))
    assert "Stokes_100.0" in data["cases"]
    assert "Stokes_1000.0" in data["cases"]

    case = data["cases"]["Stokes_100.0"]["Shape_2.0"]
    np.testing.assert_array_almost_equal(
        case["Force"], np.asarray(results["Force"][0])
    )
    np.testing.assert_array_equal(case["w_snapshots"].shape, (num_snapshots, nr))


def test_save_and_load_single_run(tmp_path: Path) -> None:
    """Save a single-run result and reload it with the correct case dim."""
    nr = 32
    num_snapshots = 3
    nt = 40

    r_grid = jnp.linspace(0.0, 6.0, nr, dtype=jnp.float32)
    time_grid = jnp.linspace(0.0, 1.0, nt + 1, dtype=jnp.float32)

    # Single-run results lack the leading case dimension.
    results = {
        "Force": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        "EnergyElastic": jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32),
        "Central_h": jnp.array([1.0, 0.9, 0.8], dtype=jnp.float32),
        "Min_h": jnp.array([0.5, 0.4, 0.3], dtype=jnp.float32),
        "D": jnp.array([1.0, 0.95, 0.9], dtype=jnp.float32),
        "V": jnp.array([-1.0, -0.5, 0.0], dtype=jnp.float32),
        "w": jnp.zeros((num_snapshots, nr), dtype=jnp.float32),
        "p": jnp.zeros((num_snapshots, nr), dtype=jnp.float32),
    }

    S_flat = jnp.array([100.0], dtype=jnp.float32)
    E_flat = jnp.array([2.0], dtype=jnp.float32)

    out_path = tmp_path / "single_run.h5"
    save_results_h5(
        out_path,
        results,
        r_grid,
        time_grid,
        S_flat,
        E_flat,
        num_snapshots=num_snapshots,
    )

    data = load_results_h5(out_path)
    case = data["cases"]["Stokes_100.0"]["Shape_2.0"]
    np.testing.assert_array_almost_equal(
        case["Force"], np.asarray(results["Force"])
    )
    np.testing.assert_array_equal(case["w_snapshots"].shape, (num_snapshots, nr))
