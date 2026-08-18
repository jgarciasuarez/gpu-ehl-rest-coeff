#!/usr/bin/env python3
"""Generate Fig. 1: restitution coefficient map for power-law impactors."""

import argparse

import h5py
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from style import (
    add_figure_cli,
    get_plasma_capped,
    output_path,
    save_or_show,
    setup_fonts,
    shade_by_st,
)


def load_restitution_data(h5_file):
    """Load restitution coefficients from the HDF5 dataset."""
    stokes_list, exp_list, restitution_list = [], [], []

    with h5py.File(h5_file, "r") as f:
        for stokes_key in f.keys():
            if not stokes_key.startswith("Stokes_"):
                continue
            stokes_val = float(stokes_key.split("_")[1])

            for shape_key in f[stokes_key].keys():
                if not shape_key.startswith("Shape_"):
                    continue
                exp_val = float(shape_key.split("_")[1])

                grp = f[stokes_key][shape_key]
                V = np.array(grp["V"])
                e = abs(V[-1] / V[0])

                stokes_list.append(stokes_val)
                exp_list.append(exp_val)
                restitution_list.append(e)

    return (
        np.array(stokes_list),
        np.array(exp_list),
        np.array(restitution_list),
    )


def build_regular_grid(stokes_arr, exp_arr, e_arr):
    """Build regular (St, n) grids for the 3D surface."""
    n_vals = np.unique(exp_arr)
    st_vals = np.unique(stokes_arr)
    E_grid = np.full((len(st_vals), len(n_vals)), np.nan)
    for i, St in enumerate(st_vals):
        for j, n in enumerate(n_vals):
            mask = (stokes_arr == St) & (exp_arr == n)
            if np.any(mask):
                E_grid[i, j] = e_arr[mask][0]
    N_grid, St_grid = np.meshgrid(n_vals, st_vals)
    return N_grid, St_grid, E_grid, n_vals, st_vals


def main():
    parser = argparse.ArgumentParser(
        description="Generate restitution coefficient map (Fig. 1)."
    )
    add_figure_cli(parser)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update(
        {
            "axes.labelsize": 25,
            "axes.titlesize": 25,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.dpi": 100,
        }
    )

    from style import resolve_h5_path

    h5_file = resolve_h5_path(args.h5_file)
    stokes_arr, exp_arr, e_arr = load_restitution_data(h5_file)
    N_grid, St_grid, E_grid, n_vals, st_vals = build_regular_grid(
        stokes_arr, exp_arr, e_arr
    )

    plasma_capped = get_plasma_capped()
    e_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(16.5, 5.5))

    # Panel (a): scatter map
    ax_a = fig.add_subplot(1, 3, 1)
    sc = ax_a.scatter(
        exp_arr,
        stokes_arr,
        c=e_arr,
        cmap=plasma_capped,
        norm=e_norm,
        s=80,
        edgecolor="k",
        linewidth=0.5,
    )
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"$n$")
    ax_a.set_ylabel(r"$\mathrm{St}$", rotation=0, labelpad=15)
    ax_a.set_box_aspect(1)
    ax_a.text(
        -0.25,
        0.97,
        "(a)",
        transform=ax_a.transAxes,
        fontsize=20,
        weight="bold",
        va="top",
        ha="left",
    )

    # Panel (b): 3D surface
    ax_b = fig.add_subplot(1, 3, 2, projection="3d")
    surf = ax_b.plot_surface(
        np.log10(N_grid),
        np.log10(St_grid),
        E_grid,
        cmap=plasma_capped,
        norm=e_norm,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax_b.set_xlabel(r"$\log n$", fontsize=20)
    ax_b.set_ylabel(r"$\log \mathrm{St}$", fontsize=20)
    ax_b.set_zlabel(r"$e$")
    ax_b.set_zlim(0.0, 1.0)
    ax_b.set_box_aspect([1, 1, 1])
    ax_b.text2D(
        0.06,
        0.97,
        "(b)",
        transform=ax_b.transAxes,
        fontsize=20,
        weight="bold",
        va="top",
        ha="left",
    )

    # Panel (c): overlapping curves e(St) per n
    ax_c = fig.add_subplot(1, 3, 3)
    n_colors = cm.plasma(np.linspace(0, 0.8, len(n_vals)))

    for j, n in enumerate(n_vals):
        mask = exp_arr == n
        st_subset = stokes_arr[mask]
        e_subset = e_arr[mask]
        order = np.argsort(st_subset)
        st_subset = st_subset[order]
        e_subset = e_subset[order]

        base_color = n_colors[j]
        ax_c.plot(
            st_subset,
            e_subset,
            "-",
            color=base_color,
            linewidth=1.5,
            label=f"$n={int(n)}$",
            zorder=2,
        )

        st_norm = (np.log10(st_subset) - np.log10(st_subset.min())) / (
            np.log10(st_subset.max()) - np.log10(st_subset.min()) + 1e-12
        )
        for k in range(len(st_subset)):
            ax_c.plot(
                st_subset[k],
                e_subset[k],
                "o",
                color=shade_by_st(base_color, st_norm[k]),
                markersize=5,
                markeredgecolor="k",
                markeredgewidth=0.3,
                zorder=3,
            )

    ax_c.set_xscale("log")
    ax_c.set_xlabel(r"$\mathrm{St}$")
    ax_c.set_ylabel(r"$e$", rotation=0, labelpad=15)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_box_aspect(1)
    ax_c.legend(frameon=True, ncol=2, loc="lower right", fontsize=11)
    ax_c.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_c.text(
        0.06,
        0.97,
        "(c)",
        transform=ax_c.transAxes,
        fontsize=20,
        weight="bold",
        va="top",
        ha="left",
    )

    fig.tight_layout(rect=[0, 0.18, 1, 1], w_pad=0.0)
    cbar_ax = fig.add_axes([0.15, 0.125, 0.7, 0.03])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Restitution coefficient $e$", fontsize=25)
    cbar.ax.tick_params(labelsize=20)

    output = output_path("restitution_map.pdf", args.output)
    save_or_show(fig, output, no_show=args.no_show)


if __name__ == "__main__":
    main()
