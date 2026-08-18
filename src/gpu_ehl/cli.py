"""Command-line interface entry point for jax-ehl."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp

from gpu_ehl.elasticity import load_or_build_kernel
from gpu_ehl.geometry import make_radial_grid, make_time_grid
from gpu_ehl.io import save_results_h5
from gpu_ehl.solver import simulate_single_setup
from gpu_ehl.sweep import make_parameter_grid, run_parametric_sweep


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nr", type=int, default=3000, help="Number of radial grid points"
    )
    parser.add_argument(
        "--lr", type=float, default=6.0, help="Outer radial domain size"
    )
    parser.add_argument(
        "--nt", type=int, default=10000, help="Number of dynamic time steps"
    )
    parser.add_argument(
        "--nt-ini",
        type=int,
        default=200,
        help="Number of initialization time steps",
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=0.5e-3,
        help="Time step size",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=100,
        help="Number of snapshots per simulation",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for cached kernel matrices",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )


def _build_kernel_and_grid(args):
    r = make_radial_grid(args.nr, args.lr)
    dr = float(args.lr / args.nr)
    kernel = load_or_build_kernel(args.nr, args.lr, cache_dir=args.cache_dir)
    return kernel, r, dr


def cmd_single(args: argparse.Namespace) -> int:
    """Run a single simulation."""
    _setup_logging(args.verbose)
    logger = logging.getLogger("jax-ehl.single")

    logger.info(
        "Running single simulation: Stokes=%g, exponent=%g on %s",
        args.stokes,
        args.exponent,
        jax.devices()[0].platform.upper(),
    )

    kernel, r, dr = _build_kernel_and_grid(args)
    Time = make_time_grid(args.delta_t, args.nt, args.nt_ini)[1]

    # CPU fallback is allowed for single runs; JAX will use whatever device is
    # available.
    history = simulate_single_setup(
        stokes=args.stokes,
        exponent=args.exponent,
        KernelMatrix=kernel,
        r=r,
        delta_t=args.delta_t,
        dr=dr,
        nt=args.nt,
        nt_ini=args.nt_ini,
        num_snapshots=args.num_snapshots,
    )
    history = jax.block_until_ready(history)

    S_flat = jnp.array([args.stokes], dtype=jnp.float32)
    E_flat = jnp.array([args.exponent], dtype=jnp.float32)

    save_results_h5(
        args.output,
        history,
        r,
        Time,
        S_flat,
        E_flat,
        num_snapshots=args.num_snapshots,
    )
    logger.info("Results written to %s", args.output)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run a full parametric sweep on the GPU."""
    _setup_logging(args.verbose)
    logger = logging.getLogger("jax-ehl.sweep")

    device = jax.devices()[0]
    if device.platform == "cpu":
        logger.error(
            "Parametric sweep is only supported on GPU. "
            "Found device: %s",
            device,
        )
        return 1

    logger.info(
        "Running parametric sweep on %s (chunk_size=%d)",
        device.platform.upper(),
        args.chunk_size,
    )

    kernel, r, dr = _build_kernel_and_grid(args)
    Time = make_time_grid(args.delta_t, args.nt, args.nt_ini)[1]

    results, S_flat, E_flat = run_parametric_sweep(
        KernelMatrix=kernel,
        r=r,
        delta_t=args.delta_t,
        dr=dr,
        nt=args.nt,
        nt_ini=args.nt_ini,
        num_snapshots=args.num_snapshots,
        chunk_size=args.chunk_size,
    )

    save_results_h5(
        args.output,
        results,
        r,
        Time,
        S_flat,
        E_flat,
        num_snapshots=args.num_snapshots,
    )
    logger.info("Sweep results written to %s", args.output)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the ``jax-ehl`` command."""
    parser = argparse.ArgumentParser(
        prog="jax-ehl",
        description="GPU-accelerated elastohydrodynamic lubrication solver.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single", help="Run a single (Stokes, exponent) simulation"
    )
    single.add_argument(
        "--stokes", type=float, required=True, help="Stokes number"
    )
    single.add_argument(
        "--exponent", type=float, required=True, help="Shape exponent"
    )
    single.add_argument(
        "--output",
        type=Path,
        default=Path("result.h5"),
        help="Output HDF5 path",
    )
    _add_common_args(single)
    single.set_defaults(func=cmd_single)

    sweep = subparsers.add_parser(
        "sweep", help="Run the full GPU parametric sweep"
    )
    sweep.add_argument(
        "--output",
        type=Path,
        default=Path("parametric_study.h5"),
        help="Output HDF5 path",
    )
    sweep.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Number of simulations per GPU batch",
    )
    _add_common_args(sweep)
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
