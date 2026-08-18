"""HDF5 input/output for simulation results."""

from pathlib import Path
from typing import Optional, Union

import h5py
import jax
import jax.numpy as jnp
import numpy as np


def _snapshot_indices(nt: int, num_snapshots: int) -> np.ndarray:
    """Return approximate snapshot step indices matching the solver output."""
    return np.linspace(0, nt - 1, num_snapshots, dtype=int)


def _ensure_case_dimension(
    arr: np.ndarray, num_cases: int, num_snapshots: int
) -> np.ndarray:
    """Add a leading case dimension if ``arr`` was produced by a single run.

    A sweep result has shape ``(num_cases, num_snapshots, ...)`` while a
    single-run result has shape ``(num_snapshots, ...)``. This helper makes
    single-run arrays compatible with the sweep layout.
    """
    if arr.shape[0] != num_cases and arr.shape[0] == num_snapshots:
        return arr[np.newaxis, ...]
    return arr


def save_results_h5(
    filepath: Union[str, Path],
    results: dict[str, jnp.ndarray],
    r_grid: jnp.ndarray,
    time_grid: jnp.ndarray,
    S_flat: jnp.ndarray,
    E_flat: jnp.ndarray,
    num_snapshots: Optional[int] = None,
) -> None:
    """Save a parametric sweep or single run to an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Output HDF5 path.
    results : dict[str, jnp.ndarray]
        History dictionary from ``simulate_single_setup`` or ``run_parametric_sweep``.
    r_grid : jnp.ndarray
        Radial grid.
    time_grid : jnp.ndarray
        Dynamic-phase time grid.
    S_flat : jnp.ndarray
        Flat Stokes array.
    E_flat : jnp.ndarray
        Flat exponent array.
    num_snapshots : int or None, optional
        Number of snapshots. If ``None``, inferred from ``results``.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if num_snapshots is None:
        num_snapshots = int(results["Force"].shape[0])
    nt = int(time_grid.shape[0]) - 1
    snapshot_idx = _snapshot_indices(nt, num_snapshots)

    results_host = {
        k: _ensure_case_dimension(
            np.asarray(v), num_cases=len(S_flat), num_snapshots=num_snapshots
        )
        for k, v in jax.device_get(results).items()
    }
    r_host = jax.device_get(r_grid)
    Time_host = jax.device_get(time_grid)

    with h5py.File(filepath, "w") as f:
        f.create_dataset("r_grid", data=r_host)
        f.create_dataset("time_grid", data=Time_host)
        f.create_dataset("snapshot_indices", data=snapshot_idx)

        for i in range(len(S_flat)):
            s_val = float(S_flat[i])
            e_val = float(E_flat[i])
            group_name = f"Stokes_{s_val:.1f}/Shape_{e_val:.1f}"
            grp = f.create_group(group_name)

            grp.create_dataset("Force", data=results_host["Force"][i])
            grp.create_dataset("Central_h", data=results_host["Central_h"][i])
            grp.create_dataset("Min_h", data=results_host["Min_h"][i])
            grp.create_dataset("V", data=results_host["V"][i])
            grp.create_dataset("w_snapshots", data=results_host["w"][i])
            grp.create_dataset("p_snapshots", data=results_host["p"][i])
            grp.create_dataset("D_snapshots", data=results_host["D"][i])
            grp.create_dataset("E_elastic", data=results_host["EnergyElastic"][i])


def load_results_h5(filepath: Union[str, Path]) -> dict:
    """Load a results file into a nested dictionary.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        Nested dictionary with keys ``r_grid``, ``time_grid``,
        ``snapshot_indices``, and per-case groups.
    """
    filepath = Path(filepath)
    data: dict = {}

    with h5py.File(filepath, "r") as f:
        data["r_grid"] = f["r_grid"][:]
        data["time_grid"] = f["time_grid"][:]
        data["snapshot_indices"] = f["snapshot_indices"][:]
        data["cases"] = {}

        for group_name in f:
            if group_name in ("r_grid", "time_grid", "snapshot_indices"):
                continue
            grp = f[group_name]
            data["cases"][group_name] = {}
            for case_name in grp:
                case_grp = grp[case_name]
                data["cases"][group_name][case_name] = {
                    "Force": case_grp["Force"][:],
                    "Central_h": case_grp["Central_h"][:],
                    "Min_h": case_grp["Min_h"][:],
                    "V": case_grp["V"][:],
                    "w_snapshots": case_grp["w_snapshots"][:],
                    "p_snapshots": case_grp["p_snapshots"][:],
                    "D_snapshots": case_grp["D_snapshots"][:],
                    "E_elastic": case_grp["E_elastic"][:],
                }

    return data
