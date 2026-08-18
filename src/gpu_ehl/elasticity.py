"""Elastic half-space kernel matrix construction and caching."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.integrate
import scipy.special
from jax import numpy as jnp
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    """Return the default cache directory for kernel matrices.

    Uses ``~/.cache/gpu-ehl/kernels`` on Unix-like systems.
    """
    cache_dir = Path.home() / ".cache" / "gpu-ehl" / "kernels"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def kernel_path(nr: int, lr: float, cache_dir: Optional[Path] = None) -> Path:
    """Return the cached kernel file path for a given grid.

    Parameters
    ----------
    nr : int
        Number of radial grid points.
    lr : float
        Outer radius of the domain.
    cache_dir : Path or None, optional
        Custom cache directory. If ``None``, the default cache dir is used.

    Returns
    -------
    Path
        Path to the ``.npy`` kernel file.
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir()
    return cache_dir / f"kernel_matrix_NR{nr}_LR{lr:.1f}_v2.npy"


def _k_integrand(x: float, ri: float) -> float:
    """Integrand for the elastic influence kernel."""
    m = 4.0 * ri * x / (ri + x) ** 2
    # Prevent m == 1.0 which causes an exact infinity.
    m = np.clip(m, 0.0, 1.0 - 1e-15)
    return (x / (ri + x)) * scipy.special.ellipk(m) * (8.0 / np.pi ** 2)


def build_kernel_matrix(nr: int, lr: float) -> np.ndarray:
    """Build the dense elastic influence kernel on the CPU.

    Parameters
    ----------
    nr : int
        Number of radial grid points.
    lr : float
        Outer radius of the domain.

    Returns
    -------
    np.ndarray
        Dense ``(nr, nr)`` kernel matrix in float64.
    """
    r_np = np.linspace(0.0, lr, nr)
    dr_np = lr / (nr - 1)
    Kernel_np = np.zeros((nr, nr), dtype=np.float64)

    for i in tqdm(range(nr), desc="Building kernel matrix"):
        ri = r_np[i]

        if i == 0:
            Kernel_np[0, 0] = scipy.integrate.quad(
                lambda _: 4.0 / np.pi, 0.0, dr_np / 2.0
            )[0]
            for j in range(1, nr):
                rj = r_np[j]
                Kernel_np[0, j] = scipy.integrate.quad(
                    lambda _: 4.0 / np.pi, rj - dr_np / 2.0, rj + dr_np / 2.0
                )[0]
        else:
            for j in range(nr):
                rj = r_np[j]
                a = 0.0 if j == 0 else rj - dr_np / 2.0
                b = rj + dr_np / 2.0

                if i == j:
                    val, _ = scipy.integrate.quad(
                        _k_integrand, a, b, args=(ri,), points=[ri], limit=100
                    )
                else:
                    val, _ = scipy.integrate.quad(
                        _k_integrand, a, b, args=(ri,), limit=100
                    )

                Kernel_np[i, j] = val

    return Kernel_np


def load_or_build_kernel(
    nr: int,
    lr: float,
    cache_dir: Optional[Path] = None,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Load a cached kernel matrix or build and cache it.

    Parameters
    ----------
    nr : int
        Number of radial grid points.
    lr : float
        Outer radius of the domain.
    cache_dir : Path or None, optional
        Custom cache directory.
    dtype
        JAX dtype for the returned array (default: ``jnp.float32``).

    Returns
    -------
    jnp.ndarray
        Kernel matrix on the default JAX device.
    """
    path = kernel_path(nr, lr, cache_dir)

    if path.exists():
        logger.info("Loading cached kernel matrix from %s", path)
        Kernel_np = np.load(path, allow_pickle=False)
    else:
        logger.info("Building kernel matrix for NR=%d LR=%.1f", nr, lr)
        Kernel_np = build_kernel_matrix(nr, lr)
        logger.info("Saving kernel matrix to %s", path)
        np.save(path, Kernel_np)

    nan_count = int(np.sum(np.isnan(Kernel_np)))
    if nan_count > 0:
        logger.warning("Kernel matrix contains %d NaN values", nan_count)
    else:
        logger.info("Kernel matrix validated: no NaN values")

    return jnp.array(Kernel_np, dtype=dtype)
