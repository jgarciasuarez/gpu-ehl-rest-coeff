#!/usr/bin/env python3
"""Generate Fig. 2: gap and pressure profiles during approach and spreading."""

import argparse

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from style import (
    add_figure_cli,
    get_twilight_half,
    load_common_grid,
    output_path,
    resolve_h5_path,
    save_or_show,
    set_spine_width,
    setup_fonts,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate approach/spreading profiles (Fig. 2)."
    )
    add_figure_cli(parser)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update(
        {
            "axes.labelsize": 24,
            "axes.titlesize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.dpi": 100,
        }
    )

    h5_file = resolve_h5_path(args.h5_file)
    r, time_grid, t_snaps = load_common_grid(h5_file)

    ST_TARGET = 1000.0
    N_COLS = [1.0, 2.0, 4.0, 8.0]
    SNAPSHOT_STRIDE = 2
    stokes_key = f"Stokes_{int(ST_TARGET)}.0"

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax in axes.flat:
        set_spine_width(ax)

    cmap = get_twilight_half(second=False)

    with h5py.File(h5_file, "r") as f:
        for j, n in enumerate(N_COLS):
            shape_key = f"Shape_{n}"
            if stokes_key not in f or shape_key not in f[stokes_key]:
                continue

            grp = f[stokes_key][shape_key]
            w_snaps = np.array(grp["w_snapshots"])
            p_snaps = np.array(grp["p_snapshots"])
            D_snaps = np.array(grp["D_snapshots"])
            V = np.array(grp["V"])

            h_all = D_snaps[:, None] + 0.5 * r ** n - w_snaps

            rebound_idx = np.argmax(V > 0)
            if rebound_idx == 0 and V[0] <= 0:
                rebound_idx = len(V)
            phase_indices = np.arange(0, rebound_idx, SNAPSHOT_STRIDE)
            if rebound_idx - 1 not in phase_indices and rebound_idx > 0:
                phase_indices = np.append(phase_indices, rebound_idx - 1)

            t_sel = t_snaps[phase_indices]
            norm = mcolors.Normalize(vmin=float(t_sel.min()), vmax=float(t_sel.max()))

            ax_top = axes[0, j]
            ax_bot = axes[1, j]

            for idx in phase_indices:
                color = cmap(norm(t_snaps[idx]))
                ax_top.plot(r, h_all[idx], color=color, linewidth=1.2)
                ax_bot.plot(r, p_snaps[idx], color=color, linewidth=1.2)

            ax_top.set_xlim(0.0, 1.3)
            ax_top.set_ylim(0.0, 0.5)
            ax_top.set_xticklabels([])
            ax_top.text(
                0.5,
                1.09,
                rf"$n = {int(n)}$",
                transform=ax_top.transAxes,
                ha="center",
                va="top",
                fontsize=22,
            )
            if j == 0:
                ax_top.set_ylabel(
                    r"$\frac{h}{\mathcal{H}}$", rotation=0, labelpad=15, fontsize=30
                )
            else:
                ax_top.set_yticklabels([])

            ax_bot.set_xlim(0.0, 1.3)
            ax_bot.set_ylim(-0.6, 1.0)
            ax_bot.set_xlabel(r"$r/\mathcal{L}$")
            if j == 0:
                ax_bot.set_ylabel(
                    r"$\frac{p}{\mathcal{P}}$", rotation=0, labelpad=15, fontsize=30
                )
            else:
                ax_bot.set_yticklabels([])

            ax_top.set_box_aspect(1)
            ax_bot.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.tight_layout(rect=[0, 0.12, 1, 1], w_pad=0.3, h_pad=3.3)

    cbar_ax = fig.add_axes([0.15, 0.075, 0.7, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$t$", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    output = output_path("profiles_approach_spread.pdf", args.output)
    save_or_show(fig, output, no_show=args.no_show)


if __name__ == "__main__":
    main()
