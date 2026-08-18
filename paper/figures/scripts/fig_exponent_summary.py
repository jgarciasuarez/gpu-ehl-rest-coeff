#!/usr/bin/env python3
"""Generate fitted central-gap scaling exponent -alpha(n) vs n."""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from style import (
    add_figure_cli,
    compute_key_times_for_shape,
    load_common_grid,
    output_path,
    resolve_h5_path,
    save_or_show,
    set_spine_width,
    setup_fonts,
)


def fit_exponent(h5_file, t_snaps, all_key_times, ST_TARGET_LIST, n):
    """Fit a power-law exponent for the central gap at mid-spread."""
    shape_key = f"Shape_{n}"
    key_times = all_key_times[shape_key]

    St_list, h_list = [], []

    with h5py.File(h5_file, "r") as f:
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

            t_target = kt["t1"] + 0.5 * (kt["t2"] - kt["t1"])
            it = int(np.argmin(np.abs(t_snaps - t_target)))

            grp = f[stokes_key][shape_key]
            if "Central_h" not in grp:
                continue
            h = float(grp["Central_h"][it])

            St_list.append(St_parsed)
            h_list.append(h)

    St_arr = np.asarray(St_list)
    h_arr = np.asarray(h_list)

    mask = (
        np.isfinite(St_arr) & np.isfinite(h_arr) & (St_arr > 0) & (h_arr > 0)
    )
    St_fit = St_arr[mask]
    h_fit = h_arr[mask]

    if len(St_fit) < 2:
        return np.nan
    alpha, _ = np.polyfit(np.log(St_fit), np.log(h_fit), 1)
    return -alpha


def main():
    parser = argparse.ArgumentParser(
        description="Generate central-gap scaling exponent summary."
    )
    add_figure_cli(parser)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update(
        {
            "axes.labelsize": 24,
            "axes.titlesize": 16,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "figure.dpi": 100,
        }
    )

    h5_file = resolve_h5_path(args.h5_file)
    _, _, t_snaps = load_common_grid(h5_file)

    N_COLS = np.arange(1, 11, dtype=float)

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

    exponent_central = np.array(
        [fit_exponent(h5_file, t_snaps, all_key_times, ST_TARGET_LIST, n) for n in N_COLS]
    )

    theory_central = 0.5 * np.ones_like(N_COLS)

    # Central gap exponent
    fig_a, ax_a = plt.subplots(figsize=(6, 6))
    ax_a.plot(
        N_COLS,
        exponent_central,
        "o",
        ms=8,
        color="tab:blue",
        markeredgecolor="black",
        markeredgewidth=0.7,
        label="fit",
        zorder=3,
    )
    ax_a.plot(
        N_COLS,
        theory_central,
        "--",
        color="gray",
        lw=2,
        label="theory",
        zorder=2,
    )
    ax_a.set_xlabel(r"$n$", fontsize=20)
    ax_a.set_ylabel(r"$-\alpha_{\text{center}}$", rotation=0, labelpad=25, fontsize=20)
    ax_a.set_xlim(0.5, 10.5)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_xticks(N_COLS)
    ax_a.grid(True, which="both", ls=":", alpha=0.5)
    ax_a.set_box_aspect(1)
    set_spine_width(ax_a)
    ax_a.legend(frameon=True, loc="best")

    ax_in = ax_a.inset_axes([0.1, 0.75, 0.2, 0.2])
    ax_in.plot(
        N_COLS[0],
        exponent_central[0],
        "o",
        ms=8,
        color="tab:blue",
        markeredgecolor="black",
        markeredgewidth=0.7,
        zorder=3,
    )
    ax_in.set_xlim(0.5, 1.5)
    ax_in.set_ylim(3.5, 4.5)
    ax_in.set_xticks([1])
    ax_in.set_yticks([4])
    ax_in.tick_params(labelsize=12)
    ax_in.grid(True, which="both", ls=":", alpha=0.5)
    ax_in.set_box_aspect(1)
    for spine in ax_in.spines.values():
        spine.set_linewidth(1.0)

    fig_a.tight_layout()
    output = output_path("exponent_summary.pdf", args.output)
    save_or_show(fig_a, output, no_show=args.no_show)


if __name__ == "__main__":
    main()
