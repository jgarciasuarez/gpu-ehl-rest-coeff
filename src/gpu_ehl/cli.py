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
from gpu_ehl.sweep import run_parametric_sweep


def _parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated string of floats into a list.

    Raises
    ------
    argparse.ArgumentTypeError
        If the string is empty or contains non-numeric entries.
    """
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise argparse.ArgumentTypeError("empty value list")
    try:
        return [float(v) for v in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float in list {value!r}") from exc


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
    dr = float(args.lr / (args.nr - 1))
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

    S_flat = jnp.array([args.stokes], dtype=jnp.float64)
    E_flat = jnp.array([args.exponent], dtype=jnp.float64)

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
            "Parametric sweep is only supported on GPU. Found device: %s",
            device,
        )
        return 1

    logger.info(
        "Running parametric sweep on %s (chunk_size=%d, stokes=%s, exponents=%s)",
        device.platform.upper(),
        args.chunk_size,
        "published grid" if args.stokes_list is None else args.stokes_list,
        "published set" if args.exponent_list is None else args.exponent_list,
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
        stokes_list=args.stokes_list,
        exponent_list=args.exponent_list,
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
    single.add_argument("--stokes", type=float, required=True, help="Stokes number")
    single.add_argument("--exponent", type=float, required=True, help="Shape exponent")
    single.add_argument(
        "--output",
        type=Path,
        default=Path("result.h5"),
        help="Output HDF5 path",
    )
    _add_common_args(single)
    single.set_defaults(func=cmd_single)

    sweep = subparsers.add_parser("sweep", help="Run the full GPU parametric sweep")
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
    sweep.add_argument(
        "--stokes-list",
        type=_parse_float_list,
        default=None,
        help=(
            "Comma-separated Stokes numbers to simulate (subset sweep); "
            "omitting it uses the published 13-value grid"
        ),
    )
    sweep.add_argument(
        "--exponent-list",
        type=_parse_float_list,
        default=None,
        help=(
            "Comma-separated shape exponents to simulate (subset sweep); "
            "omitting it uses the published 10-value set"
        ),
    )
    _add_common_args(sweep)
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
