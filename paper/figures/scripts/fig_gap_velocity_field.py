#!/usr/bin/env python3
"""Generate gap velocity-field arrow figure (legacy fig_flow_v2.py)."""

import argparse

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np

from style import (
    add_figure_cli,
    compute_flow_key_times,
    load_common_grid,
    output_path,
    reconstruct_gap_velocity,
    resolve_h5_path,
    save_or_show,
    set_spine_width,
    setup_fonts,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate gap velocity field with arrows."
    )
    add_figure_cli(parser)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 100,
        }
    )

    h5_file = resolve_h5_path(args.h5_file)
    r, time_grid, t_snaps = load_common_grid(h5_file)

    N_FIXED = 2.0
    ST_FIXED = 1000.0
    stokes_key = f"Stokes_{ST_FIXED:.1f}"
    shape_key = f"Shape_{N_FIXED}"

    VERTICAL_EXAGGERATION = 20.0
    X_MAX = 0.5
    Z_MAX = 0.05
    N_ARROWS_R = 150
    N_ARROWS_Z = 52
    ARROW_WIDTH = 0.003
    ARROW_SCALE = 25.0
    CMAP = "viridis"
    SMOOTH = False
    N_ETA = 51

    with h5py.File(h5_file, "r") as f:
        if stokes_key not in f or shape_key not in f[stokes_key]:
            raise ValueError(
                f"Requested case {stokes_key}/{shape_key} not found in dataset."
            )
        grp = f[stokes_key][shape_key]
        w_snaps = np.array(grp["w_snapshots"])
        D_snaps = np.array(grp["D_snapshots"])
        F = np.array(grp["Force"])

    h_pos = D_snaps[:, None] + 0.5 * r ** N_FIXED - w_snaps
    h_pos = np.maximum(h_pos, 1e-6)

    if SMOOTH:
        from scipy.signal import savgol_filter

        h_pos = savgol_filter(h_pos, window_length=7, polyorder=3, axis=0)
        h_pos = savgol_filter(h_pos, window_length=7, polyorder=3, axis=1)
        h_pos = np.maximum(h_pos, 1e-6)

    eta, z_pos, ur_pos, uz_pos, q, ht, hr = reconstruct_gap_velocity(
        r=r, t=t_snaps, h=h_pos, n_eta=N_ETA
    )

    r_neg = -r[::-1][:-1]
    r_sym = np.concatenate([r_neg, r])

    h_sym = np.concatenate([h_pos[:, ::-1][:, 1:], h_pos], axis=1)
    uz_sym = np.concatenate([uz_pos[:, :, ::-1][:, :, 1:], uz_pos], axis=2)
    ur_x = np.concatenate([-ur_pos[:, :, ::-1][:, :, 1:], ur_pos], axis=2)
    speed_sym = np.sqrt(ur_x ** 2 + uz_sym ** 2)

    kt = compute_flow_key_times(D_snaps, F)
    t1_idx = kt["t1_idx"]
    t2_idx = kt["t2_idx"]
    t3_idx = kt["t3_idx"]
    t4_idx = kt["t4_idx"]

    idx_approach = max(0, t1_idx) // 2
    idx_spread = t1_idx + (t2_idx - t1_idx) // 2
    idx_retract = t2_idx + (t3_idx - t2_idx) // 2
    idx_adhesion = t4_idx if t4_idx is not None else len(t_snaps) - 2
    idx_last = len(t_snaps) - 1

    plot_indices = [idx_approach, idx_spread, idx_retract, idx_adhesion, idx_last]
    panel_titles = ["approach", "spreading", "retraction", "peak adhesion", "final"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    vmax = np.nanpercentile(speed_sym, 99.0)
    norm = Normalize(vmin=0.0, vmax=vmax)

    r_indices = np.linspace(0, r_sym.size - 1, N_ARROWS_R).astype(int)
    eta_indices = np.linspace(0, eta.size - 1, N_ARROWS_Z).astype(int)

    R_sym = np.broadcast_to(r_sym, (eta.size, r_sym.size))

    for ax, idx, title in zip(axes, plot_indices, panel_titles):
        Zplot = VERTICAL_EXAGGERATION * z_pos[idx]
        Z_mir = np.concatenate([Zplot[:, ::-1][:, 1:], Zplot], axis=1)

        ax.plot(r_sym, np.zeros_like(r_sym), "k-", linewidth=1.2)
        ax.plot(r_sym, VERTICAL_EXAGGERATION * h_sym[idx], "k-", linewidth=1.2)

        ur_mir = ur_x[idx][np.ix_(eta_indices, r_indices)]
        uz_mir = uz_sym[idx][np.ix_(eta_indices, r_indices)]
        arrow_speed = speed_sym[idx][np.ix_(eta_indices, r_indices)]

        arrow_norm = np.sqrt(ur_mir ** 2 + uz_mir ** 2)
        arrow_norm = np.where(arrow_norm > 0.0, arrow_norm, 1.0)

        ax.quiver(
            R_sym[np.ix_(eta_indices, r_indices)],
            Z_mir[np.ix_(eta_indices, r_indices)],
            ur_mir / arrow_norm,
            uz_mir / arrow_norm,
            arrow_speed,
            cmap=CMAP,
            norm=norm,
            width=ARROW_WIDTH,
            scale=ARROW_SCALE,
            scale_units="xy",
            angles="xy",
        )

        ax.set_xlim(-X_MAX, X_MAX)
        ax.set_xlabel(r"$r/\mathcal{L}$")
        ax.set_title(f"{title}\n" + r"$\tilde{t}=$" + f"{t_snaps[idx]:.3f}")

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    axes[0].set_ylim(0.0, VERTICAL_EXAGGERATION * Z_MAX)
    axes[0].set_ylabel(rf"${VERTICAL_EXAGGERATION:g}\,z/\mathcal{{H}}$")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    fig.tight_layout(rect=[0, 0.14, 1, 1], w_pad=0.3)

    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Speed $\sqrt{u_r^2+u_z^2}$", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    output = output_path(
        f"gap_velocity_field_n{int(N_FIXED)}_St{int(ST_FIXED)}.pdf", args.output
    )
    save_or_show(fig, output, no_show=args.no_show)


if __name__ == "__main__":
    main()
