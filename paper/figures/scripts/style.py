"""Shared plotting style and helper utilities for paper figure scripts."""

import argparse
import colorsys
import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------
# 1. Font and style setup
# ------------------------------------------------------------------------------
def setup_fonts() -> None:
    """Configure DejaVu Sans + stix math font for all figures."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_spine_width(ax, lw: float = 1.5) -> None:
    """Set the width of all spines in an axes."""
    for spine in ax.spines.values():
        spine.set_linewidth(lw)


# ------------------------------------------------------------------------------
# 2. Colormaps
# ------------------------------------------------------------------------------
def get_plasma_capped() -> mcolors.LinearSegmentedColormap:
    """Return the plasma colormap capped at 0.8 (removes yellow/white shades)."""
    plasma_full = matplotlib.colormaps["plasma"]
    return mcolors.LinearSegmentedColormap.from_list(
        "plasma_capped", plasma_full(np.linspace(0, 0.8, 256))
    )


def get_twilight_half(second: bool = False) -> mcolors.LinearSegmentedColormap:
    """Return the first or second half of the twilight colormap."""
    twilight = matplotlib.colormaps["twilight"]
    if second:
        return mcolors.LinearSegmentedColormap.from_list(
            "twilight_second_half", twilight(np.flip(np.linspace(0.0, 0.5, 256)))
        )
    return mcolors.LinearSegmentedColormap.from_list(
        "twilight_first_half", twilight(np.linspace(0.0, 0.5, 256))
    )


# ------------------------------------------------------------------------------
# 3. Color shading
# ------------------------------------------------------------------------------
def shade_by_st(rgb: Tuple[float, ...], st_norm: float) -> Tuple[float, float, float, float]:
    """Return a Stokes-shaded version of a base RGB color.

    Parameters
    ----------
    rgb : tuple
        Base RGB(A) color.
    st_norm : float
        Normalized Stokes number in ``[0, 1]``; 0 = low St (lighter),
        1 = high St (darker).

    Returns
    -------
    tuple
        Shaded RGBA color.
    """
    r, g, b = rgb[:3]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    lmax = min(0.85, l + 0.35)
    lmin = max(0.15, l - 0.35)
    new_l = lmax + (lmin - lmax) * st_norm
    new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
    return (new_r, new_g, new_b, 1.0)


# ------------------------------------------------------------------------------
# 4. HDF5 path resolution
# ------------------------------------------------------------------------------
def resolve_h5_path(user_path: Optional[str] = None) -> Path:
    """Resolve the HDF5 dataset path.

    Resolution order:
    1. ``user_path`` if provided.
    2. ``GPU_EHL_DATA`` environment variable.
    3. Local ``data/sample/parametric_sample.h5``.

    Parameters
    ----------
    user_path : str or None, optional
        Explicit path passed by the user.

    Returns
    -------
    Path
        Resolved path to the HDF5 file.
    """
    if user_path is not None:
        return Path(user_path)

    env_path = os.environ.get("GPU_EHL_DATA")
    if env_path:
        return Path(env_path)

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "sample" / "parametric_sample.h5"


def load_common_grid(h5_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the radial grid, time grid, and snapshot times from an HDF5 file.

    Parameters
    ----------
    h5_file : Path
        Path to the HDF5 dataset.

    Returns
    -------
    tuple
        ``(r, time_grid, t_snaps)`` where ``t_snaps`` are the times at the
        stored snapshot indices.
    """
    with h5py.File(h5_file, "r") as f:
        r = np.array(f["r_grid"])
        time_grid = np.array(f["time_grid"])
        snapshot_indices = np.array(f["snapshot_indices"])
        t_snaps = time_grid[snapshot_indices]
    return r, time_grid, t_snaps


# ------------------------------------------------------------------------------
# 5. Key-time computation
# ------------------------------------------------------------------------------
def compute_key_times_for_shape(
    h5_file: Path,
    shape_key: str,
    allowed: np.ndarray,
    verbose: bool = False,
) -> dict:
    """Compute t1-t5 indices for a given shape and list of Stokes numbers.

    This is the algorithm shared across fig4, fig5, fig6, fig7, and fig8.

    Parameters
    ----------
    h5_file : Path
        Path to the HDF5 dataset.
    shape_key : str
        HDF5 shape group name, e.g. ``"Shape_2.0"``.
    allowed : np.ndarray
        Allowed Stokes values.
    verbose : bool, optional
        Print computed key times.

    Returns
    -------
    dict
        Mapping ``Stokes -> {t1_idx, t1, t2_idx, t2, ..., t5_idx, t5}``.
    """
    st_tol = 1e-6
    tol_zero = 1e-10
    tol_one = 1e-6
    curv_rel = 1e-3

    key_times = {}

    with h5py.File(h5_file, "r") as f:
        t = f["time_grid"][f["snapshot_indices"][:]]

        for stokes_key in f:
            if not stokes_key.startswith("Stokes_"):
                continue
            if shape_key not in f[stokes_key]:
                continue

            St_parsed = float(stokes_key.split("_")[1])
            if not np.any(np.isclose(St_parsed, allowed, rtol=0.0, atol=st_tol)):
                continue
            St = float(allowed[np.argmin(np.abs(allowed - St_parsed))])

            g = f[stokes_key][shape_key]
            if "D_snapshots" not in g or "Force" not in g:
                continue

            D = np.asarray(g["D_snapshots"][:], dtype=float)
            F = np.asarray(g["Force"][:], dtype=float)

            if len(D) != len(t) or len(F) != len(t):
                continue

            # t1: first time D reaches zero
            zero = np.where(np.abs(D) <= tol_zero)[0]
            if zero.size:
                t1_idx = int(zero[0])
            else:
                s = np.sign(D)
                cross = np.where(s[:-1] * s[1:] <= 0)[0]
                t1_idx = int(cross[0] + 1) if cross.size else int(np.argmin(np.abs(D)))
            t1 = float(t[t1_idx])

            # t2: time of maximum force
            t2_idx = int(np.argmax(F))
            t2 = float(t[t2_idx])

            # t3: force returns to zero after peak
            t3_idx = None
            if t2_idx < len(F) - 1:
                zeroF = np.where(np.abs(F[t2_idx + 1 :]) <= tol_zero)[0]
                if zeroF.size:
                    t3_idx = int(t2_idx + 1 + zeroF[0])
                else:
                    sF = np.sign(F)
                    crossF = np.where(sF[t2_idx:-1] * sF[t2_idx + 1 :] <= 0)[0]
                    if crossF.size:
                        t3_idx = int(t2_idx + crossF[0] + 1)
            t3 = float(t[t3_idx]) if t3_idx is not None else None

            # t4: inflection of total energy after relaxation
            t4_idx, t4 = None, None
            if "E_elastic" in g and "V" in g and t3_idx is not None:
                E = np.asarray(g["E_elastic"][:], dtype=float)
                V = np.asarray(g["V"][:], dtype=float)

                if len(E) == len(t) and len(V) == len(t):
                    Etot = E + 0.5 * V ** 2
                    s2 = np.diff(Etot, n=2)

                    if len(s2) >= 2 and np.any(np.isfinite(s2)):
                        curv_eps = curv_rel * np.nanmax(np.abs(s2))
                        s2_sign = np.sign(s2)
                        strong = np.abs(s2) > curv_eps
                        start_k = max(1, t3_idx - 1)

                        for k in range(start_k, len(s2)):
                            if (
                                strong[k - 1]
                                and strong[k]
                                and s2_sign[k - 1] != s2_sign[k]
                            ):
                                t4_idx = k + 1
                                t4 = float(t[t4_idx])
                                break

            # t5: D reaches one after relaxation
            start = t3_idx + 1 if t3_idx is not None else 0
            t5_idx, t5 = None, None
            if start < len(D):
                one = np.where(np.abs(D[start:] - 1.0) <= tol_one)[0]
                if one.size:
                    t5_idx = int(start + one[0])
                    t5 = float(t[t5_idx])

            key_times[St] = {
                "t1_idx": t1_idx,
                "t1": t1,
                "t2_idx": t2_idx,
                "t2": t2,
                "t3_idx": t3_idx,
                "t3": t3,
                "t4_idx": t4_idx,
                "t4": t4,
                "t5_idx": t5_idx,
                "t5": t5,
            }

    key_times = {St: key_times[St] for St in sorted(key_times)}

    if verbose:
        for St, kt in key_times.items():
            print(
                f"St={St:g}: t1={kt['t1']}, t2={kt['t2']}, "
                f"t3={kt['t3']}, t4={kt['t4']}, t5={kt['t5']}"
            )

    return key_times


# ------------------------------------------------------------------------------
# 6. Time-marker styles
# ------------------------------------------------------------------------------
time_markers = {
    "t1": {"marker": "o", "ms": 6, "label": r"$\tilde{t}_1$"},
    "t2": {"marker": "s", "ms": 6, "label": r"$\tilde{t}_2$"},
    "t3": {"marker": "^", "ms": 7, "label": r"$\tilde{t}_3$"},
    "t4": {"marker": "D", "ms": 6, "label": r"$\tilde{t}_4$"},
}


def marker_handles() -> list:
    """Return Line2D handles for the threshold-time legend."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=style["marker"],
            markersize=style["ms"],
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=style["label"],
        )
        for _, style in time_markers.items()
    ]


# ------------------------------------------------------------------------------
# 7. CLI and output helpers
# ------------------------------------------------------------------------------
def add_figure_cli(parser: argparse.ArgumentParser) -> None:
    """Add common figure-script CLI arguments."""
    parser.add_argument(
        "--h5-file",
        type=str,
        default=None,
        help="Path to HDF5 dataset (default: GPU_EHL_DATA env var or sample)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output PDF path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the figure (use for CI)",
    )


def output_path(default_name: str, user_output: Optional[str] = None) -> Path:
    """Resolve the output PDF path.

    Parameters
    ----------
    default_name : str
        Default filename, e.g. ``"fig1.pdf"``.
    user_output : str or None, optional
        User-specified output path.

    Returns
    -------
    Path
        Resolved output path under ``paper/figures/output/``.
    """
    if user_output is not None:
        return Path(user_output)

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "paper" / "figures" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / default_name


def save_or_show(fig, output: Path, no_show: bool = False) -> None:
    """Save the figure and optionally display it."""
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Saved: {output}")
    if not no_show:
        plt.show()


# ------------------------------------------------------------------------------
# 8. Flow-field helpers
# ------------------------------------------------------------------------------
def reconstruct_gap_velocity(r, t, h, n_eta=51):
    """Reconstruct lubrication velocity field from gap history h(r,t).

    Parameters
    ----------
    r : np.ndarray
        Radial grid.
    t : np.ndarray
        Time coordinates.
    h : np.ndarray
        Gap history with shape ``(t.size, r.size)``.
    n_eta : int, optional
        Number of points across the gap (default: 51).

    Returns
    -------
    tuple
        ``(eta, z, ur, uz, q, ht, hr)``.
    """
    from scipy.integrate import cumulative_trapezoid

    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)
    h = np.asarray(h, dtype=float)

    if h.shape != (t.size, r.size):
        raise ValueError(f"h must have shape {(t.size, r.size)}, got {h.shape}")
    if np.any(h <= 0):
        raise ValueError("The gap h must remain strictly positive.")
    if np.any(np.diff(r) <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("r and t must be strictly increasing.")

    ht = np.gradient(h, t, axis=0, edge_order=2)
    hr = np.gradient(h, r, axis=1, edge_order=2)

    integrand = r[None, :] * ht
    radial_integral = cumulative_trapezoid(integrand, r, axis=1, initial=0.0)

    q = np.zeros_like(h)
    nonzero = r != 0.0
    q[:, nonzero] = -radial_integral[:, nonzero] / r[nonzero]

    eta = np.linspace(0.0, 1.0, n_eta)
    E = eta[None, :, None]

    H = h[:, None, :]
    HT = ht[:, None, :]
    HR = hr[:, None, :]
    Q = q[:, None, :]

    z = E * H
    ur = 6.0 * Q / H * E * (1.0 - E)
    uz = HT * (3.0 * E ** 2 - 2.0 * E ** 3) + 6.0 * Q * HR / H * E ** 2 * (1.0 - E)

    return eta, z, ur, uz, q, ht, hr


def compute_flow_key_times(D: np.ndarray, F: np.ndarray) -> dict:
    """Minimal key-time finder for flow-field figures.

    Returns t1-t4 indices used to select representative snapshots.
    """
    tol_zero = 1e-10

    zero = np.where(np.abs(D) <= tol_zero)[0]
    if zero.size:
        t1_idx = int(zero[0])
    else:
        s = np.sign(D)
        cross = np.where(s[:-1] * s[1:] <= 0)[0]
        t1_idx = int(cross[0] + 1) if cross.size else int(np.argmin(np.abs(D)))

    t2_idx = int(np.argmax(F))

    t3_idx = None
    if t2_idx < len(F) - 1:
        zeroF = np.where(np.abs(F[t2_idx + 1 :]) <= tol_zero)[0]
        if zeroF.size:
            t3_idx = int(t2_idx + 1 + zeroF[0])
        else:
            sF = np.sign(F)
            crossF = np.where(sF[t2_idx:-1] * sF[t2_idx + 1 :] <= 0)[0]
            if crossF.size:
                t3_idx = int(t2_idx + crossF[0] + 1)

    t4_idx = None
    if t3_idx is not None:
        t4_idx = t3_idx + int(np.argmin(F[t3_idx:]))

    return {"t1_idx": t1_idx, "t2_idx": t2_idx, "t3_idx": t3_idx, "t4_idx": t4_idx}
