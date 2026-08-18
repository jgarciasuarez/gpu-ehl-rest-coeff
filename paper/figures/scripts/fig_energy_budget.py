#!/usr/bin/env python3
"""Generate Fig. 4: energy budget (E, K, D) vs time for selected n and St."""

import argparse

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from style import (
    add_figure_cli,
    compute_key_times_for_shape,
    get_plasma_capped,
    load_common_grid,
    marker_handles,
    output_path,
    resolve_h5_path,
    save_or_show,
    set_spine_width,
    setup_fonts,
    shade_by_st,
    time_markers,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate energy budget figure (Fig. 4)."
    )
    add_figure_cli(parser)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update(
        {
            "axes.labelsize": 18,
            "axes.titlesize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 18,
            "figure.dpi": 100,
        }
    )

    h5_file = resolve_h5_path(args.h5_file)
    r, time_grid, t_snaps = load_common_grid(h5_file)

    ST_TARGET_LIST = np.array([100.0, 316.2, 1778.3, 5623.3, 10000.0])
    N_COLS = [1.0, 2.0, 4.0, 8.0]

    all_key_times = {}
    for n in N_COLS:
        shape_key = f"Shape_{n}"
        all_key_times[shape_key] = compute_key_times_for_shape(
            h5_file, shape_key, ST_TARGET_LIST
        )

    st_log_min = np.log10(ST_TARGET_LIST.min())
    st_log_max = np.log10(ST_TARGET_LIST.max())

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex="col", sharey="row")
    for ax in axes.flat:
        set_spine_width(ax)

    n_colors = get_plasma_capped()(np.linspace(0, 1, len(N_COLS)))

    with h5py.File(h5_file, "r") as f:
        for j, n in enumerate(N_COLS):
            shape_key = f"Shape_{n}"
            base_color = n_colors[j]
            key_times = all_key_times[shape_key]
            stokes_handles_col = []

            stokes_pairs = []
            for stokes_key in f.keys():
                if not stokes_key.startswith("Stokes_"):
                    continue
                if shape_key not in f[stokes_key]:
                    continue
                St_parsed = float(stokes_key.split("_")[1])
                if not np.any(
                    np.isclose(St_parsed, ST_TARGET_LIST, rtol=0.0, atol=1e-1)
                ):
                    continue
                stokes_pairs.append((St_parsed, stokes_key))
            stokes_pairs.sort(key=lambda x: x[0])

            for St_parsed, stokes_key in stokes_pairs:
                St_snap = float(
                    ST_TARGET_LIST[np.argmin(np.abs(ST_TARGET_LIST - St_parsed))]
                )
                grp = f[stokes_key][shape_key]
                if "E_elastic" not in grp or "V" not in grp:
                    continue

                E = np.array(grp["E_elastic"])
                V = np.array(grp["V"])
                K = 0.5 * V ** 2
                D = 0.5 - K - E

                st_norm = (np.log10(St_parsed) - st_log_min) / (
                    st_log_max - st_log_min
                )
                color = shade_by_st(base_color, st_norm)

                axes[0, j].plot(t_snaps, E, color=color, lw=1.5)
                axes[1, j].plot(t_snaps, K, color=color, lw=1.5)
                axes[2, j].plot(t_snaps, D, color=color, lw=1.5)

                h, = axes[0, j].plot(
                    [], [], color=color, lw=1.5, label=f"{St_parsed:g}"
                )
                stokes_handles_col.append(h)

                kt = key_times.get(St_snap, None)
                if kt is not None:
                    for tk, style in time_markers.items():
                        idx = kt.get(f"{tk}_idx", None)
                        if idx is None or not (0 <= idx < len(t_snaps)):
                            continue
                        axes[0, j].plot(
                            t_snaps[idx],
                            E[idx],
                            linestyle="None",
                            marker=style["marker"],
                            markersize=style["ms"],
                            markerfacecolor=color,
                            markeredgecolor="black",
                            markeredgewidth=0.7,
                            zorder=5,
                        )
                        axes[1, j].plot(
                            t_snaps[idx],
                            K[idx],
                            linestyle="None",
                            marker=style["marker"],
                            markersize=style["ms"],
                            markerfacecolor=color,
                            markeredgecolor="black",
                            markeredgewidth=0.7,
                            zorder=5,
                        )
                        axes[2, j].plot(
                            t_snaps[idx],
                            D[idx],
                            linestyle="None",
                            marker=style["marker"],
                            markersize=style["ms"],
                            markerfacecolor=color,
                            markeredgecolor="black",
                            markeredgewidth=0.7,
                            zorder=5,
                        )

            axes[0, j].legend(
                handles=stokes_handles_col,
                title=r"$\mathrm{St}$",
                loc="best",
                frameon=True,
                ncol=1,
                fontsize=12,
            )

    for j, n in enumerate(N_COLS):
        axes[0, j].set_title(rf"$n = {int(n)}$", fontsize=25)
        axes[2, j].set_xlabel(r"$t / \tau$", fontsize=25)
        for row in range(3):
            axes[row, j].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("elastic", rotation=90, labelpad=15, fontsize=22)
    axes[1, 0].set_ylabel("kinetic", rotation=90, labelpad=15, fontsize=22)
    axes[2, 0].set_ylabel("dissipation", rotation=90, labelpad=15, fontsize=22)

    fig.tight_layout(rect=[0, 0.10, 1, 1], w_pad=0.0, h_pad=0.5)
    fig.legend(
        handles=marker_handles(),
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=20,
        frameon=True,
        title="Threshold times",
        title_fontsize=15,
    )

    output = output_path("energy_budget.pdf", args.output)
    save_or_show(fig, output, no_show=args.no_show)


if __name__ == "__main__":
    main()
