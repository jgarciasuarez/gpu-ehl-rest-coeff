"""Parametric sweep utilities: grid generation and batched GPU execution."""

import logging
import math
from typing import Optional

import jax
import jax.numpy as jnp

from gpu_ehl.solver import simulate_single_setup

logger = logging.getLogger(__name__)


def make_parameter_grid() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create the final (Stokes, exponent) parameter grid.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(S_flat, E_flat)`` flat arrays of all parameter combinations.
        Stokes ranges from ``10^1`` to ``10^4``; exponents are
        ``[1.05, 2.0, 3.0, ..., 10.0]``.
    """
    kk = 4
    nn = jnp.arange(1, 4)[:, None]
    jj = jnp.arange(kk)[None, :]

    vals = 10 ** (nn + jj / kk)
    vals = vals.reshape(-1)
    stokes_test = jnp.append(vals, 10 ** 4)
    shapes_test = jnp.array(
        [1.05, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )

    S_grid, E_grid = jnp.meshgrid(stokes_test, shapes_test)
    return S_grid.flatten(), E_grid.flatten()


def _batched_sim_factory(
    nt: int, nt_ini: int, num_snapshots: int
):
    """Build a JIT-compiled, vmapped version of ``simulate_single_setup``."""
    batched_sim = jax.vmap(
        simulate_single_setup,
        in_axes=(0, 0, None, None, None, None, None, None, None),
    )
    return jax.jit(
        batched_sim,
        static_argnums=(6, 7, 8),
    )


def run_batched_chunks(
    S_flat: jnp.ndarray,
    E_flat: jnp.ndarray,
    chunk_size: int,
    KernelMatrix: jnp.ndarray,
    r: jnp.ndarray,
    delta_t: float,
    dr: float,
    nt: int,
    nt_ini: int,
    num_snapshots: int = 100,
) -> dict[str, jnp.ndarray]:
    """Run a parameter sweep in GPU-batched chunks.

    Parameters
    ----------
    S_flat, E_flat : jnp.ndarray
        Flat Stokes and exponent arrays.
    chunk_size : int
        Number of simulations to run per GPU batch.
    KernelMatrix : jnp.ndarray
        Elastic influence kernel.
    r : jnp.ndarray
        Radial grid.
    delta_t : float
        Time step size.
    dr : float
        Radial grid spacing.
    nt : int
        Number of dynamic time steps.
    nt_ini : int
        Number of initialization time steps.
    num_snapshots : int, optional
        Number of snapshots per simulation (default: 100).

    Returns
    -------
    dict[str, jnp.ndarray]
        Concatenated history dictionary over all parameter combinations.
    """
    total_sims = len(S_flat)
    num_chunks = math.ceil(total_sims / chunk_size)
    jitted_batched_sim = _batched_sim_factory(nt, nt_ini, num_snapshots)

    results_list = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_sims)
        logger.info(
            "Running chunk %d/%d (simulations %d-%d)",
            i + 1,
            num_chunks,
            start,
            end,
        )

        chunk_res = jitted_batched_sim(
            S_flat[start:end],
            E_flat[start:end],
            KernelMatrix,
            r,
            delta_t,
            dr,
            nt,
            nt_ini,
            num_snapshots,
        )
        chunk_res = jax.block_until_ready(chunk_res)
        results_list.append(chunk_res)

    return jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *results_list)


def run_parametric_sweep(
    KernelMatrix: jnp.ndarray,
    r: jnp.ndarray,
    delta_t: float,
    dr: float,
    nt: int = 10000,
    nt_ini: int = 200,
    num_snapshots: int = 100,
    chunk_size: int = 10,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Run the full parameter sweep on the GPU.

    Parameters
    ----------
    KernelMatrix : jnp.ndarray
        Elastic influence kernel.
    r : jnp.ndarray
        Radial grid.
    delta_t : float
        Time step size.
    dr : float
        Radial grid spacing.
    nt : int, optional
        Number of dynamic time steps (default: 10000).
    nt_ini : int, optional
        Number of initialization time steps (default: 200).
    num_snapshots : int, optional
        Number of snapshots per simulation (default: 100).
    chunk_size : int, optional
        GPU batch size (default: 10).

    Returns
    -------
    tuple
        ``(results, S_flat, E_flat)`` where ``results`` is the concatenated
        history dictionary.
    """
    S_flat, E_flat = make_parameter_grid()
    S_flat = jnp.array(S_flat, dtype=jnp.float32)
    E_flat = jnp.array(E_flat, dtype=jnp.float32)

    logger.info(
        "Starting parametric sweep: %d simulations", S_flat.shape[0]
    )
    results = run_batched_chunks(
        S_flat,
        E_flat,
        chunk_size,
        KernelMatrix,
        r,
        delta_t,
        dr,
        nt,
        nt_ini,
        num_snapshots,
    )
    logger.info("Parametric sweep complete")
    return results, S_flat, E_flat
