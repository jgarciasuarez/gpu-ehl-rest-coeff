#!/usr/bin/env python3
"""Generate Fig. 6: central-height scaling h(0,t) ~ St^alpha."""

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
    output_path,
    resolve_h5_path,
    save_or_show,
    set_spine_width,
    setup_fonts,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate central gap scaling figure (Fig. 6)."
    )
    add_figure_cli(parser)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update(
        {
            "axes.labelsize": 18,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 100,
        }
    )

    h5_file = resolve_h5_path(args.h5_file)
    r, time_grid, t_snaps = load_common_grid(h5_file)

    N_COLS = np.arange(1, 10, dtype=float)
    FRACTION_SPREAD = 0.5

    with h5py.File(h5_file, "r") as f:
        all_stokes = sorted(
            [float(k.split("_")[1]) for k in f.keys() if k.startswith("Stokes_")]
        )
    ST_TARGET_LIST = np.array(all_stokes)

    all_key_times = {}
    for n in N_COLS:
        shape_key = f"Shape_{n}"
        all_key_times[shape_key] = compute_key_times_for_shape(
            h5_file, shape_key, ST_TARGET_LIST, verbose=False
        )

    data = {n: {"St": [], "h0": []} for n in N_COLS}

    with h5py.File(h5_file, "r") as f:
        for n in N_COLS:
            shape_key = f"Shape_{n}"
            key_times = all_key_times[shape_key]

            for stokes_key in f:
                if not stokes_key.startswith("Stokes_"):
                    continue
                if shape_key not in f[stokes_key]:
                    continue

                St_parsed = float(stokes_key.split("_")[1])
                St_snap = float(
                    ST_TARGET_LIST[np.argmin(np.abs(ST_TARGET_LIST - St_parsed))]
                )
                kt = key_times.get(St_snap, None)

                if kt is None or kt["t1_idx"] is None or kt["t2_idx"] is None:
                    continue

                t_target = kt["t1"] + FRACTION_SPREAD * (kt["t2"] - kt["t1"])
                it = int(np.argmin(np.abs(t_snaps - t_target)))

                grp = f[stokes_key][shape_key]
                if "Central_h" not in grp:
                    continue
                h_central = float(grp["Central_h"][it])

                data[n]["St"].append(St_parsed)
                data[n]["h0"].append(h_central)

    fig, axes = plt.subplots(3, 3, figsize=(14, 14), sharex="col")
    for ax in axes.flat:
        set_spine_width(ax)

    n_colors = get_plasma_capped()(np.linspace(0, 1, len(N_COLS)))

    for idx, n in enumerate(N_COLS):
        ax = axes.flat[idx]
        color = n_colors[idx]

        St_arr = np.asarray(data[n]["St"])
        h0_arr = np.asarray(data[n]["h0"])

        order = np.argsort(St_arr)
        St_arr, h0_arr = St_arr[order], h0_arr[order]

        mask = (
            np.isfinite(St_arr)
            & np.isfinite(h0_arr)
            & (St_arr > 0)
            & (h0_arr > 0)
        )
        St_fit = St_arr[mask]
        h_fit = h0_arr[mask]

        ax.loglog(
            St_arr,
            h0_arr,
            "o",
            ms=6,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.7,
            zorder=3,
        )

        if len(St_fit) >= 2:
            alpha, lnC = np.polyfit(np.log(St_fit), np.log(h_fit), 1)
            C = np.exp(lnC)

            xs = np.logspace(
                np.log10(St_fit.min()), np.log10(St_fit.max()), 200
            )
            ys_fit = C * xs ** alpha

            St_ref = np.sqrt(St_fit.min() * St_fit.max())
            C_theory = C * St_ref ** (alpha + 0.5)
            ys_theory = C_theory * xs ** (-0.5)

            ax.loglog(xs, ys_fit, "k--", lw=2, zorder=2)
            ax.loglog(xs, ys_theory, ":", color="gray", lw=2, zorder=1)

            ax.text(
                0.5,
                0.95,
                fr"fit: $St^{{{alpha:.2f}}}$" + "\n" + r"theory: $St^{-0.5}$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=16,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none",
                ),
            )

        ax.set_title(rf"$n = {int(n)}$", fontsize=16)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_box_aspect(1)

    for j in range(3):
        axes[2, j].set_xlabel(r"$\mathrm{St}$", fontsize=25)

    for i in range(3):
        axes[i, 0].set_ylabel(
            r"$\frac{h}{\mathcal{H}}$", rotation=0, labelpad=15, fontsize=30
        )

    for i in range(3):
        for j in range(1, 3):
            axes[i, j].set_yticklabels([])

    fig.tight_layout(w_pad=0.0, h_pad=0.75)
    fig.subplots_adjust(wspace=0.02)

    output = output_path("central_gap_scaling.pdf", args.output)
    save_or_show(fig, output, no_show=args.no_show)


if __name__ == "__main__":
    main()
