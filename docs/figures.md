# Figures

This document maps each paper figure to its reproduction script in
`paper/figures/scripts/`.

## Legacy-to-new naming

| Legacy script | New script | Output PDF |
|---|---|---|
| `fig1.py` | `fig_restitution_map.py` | `restitution_map.pdf` |
| `fig2.py` | `fig_profiles_approach_spread.py` | `profiles_approach_spread.pdf` |
| `fig4.py` | `fig_energy_budget.py` | `energy_budget.pdf` |
| `fig5.py` | `fig_time_series.py` | `time_series.pdf` |
| `fig6.py` | `fig_central_gap_scaling.py` | `central_gap_scaling.pdf` |
| `fig8.py` | `fig_exponent_summary.py` | `exponent_summary.pdf` |
| `fig_flow_v2.py` | `fig_gap_velocity_field.py` | `gap_velocity_field_n*_St*.pdf` |

## Shared style

All figure scripts use `paper/figures/scripts/style.py`, which provides:

- DejaVu Sans + `stix` math font
- Plasma colormap capped at `0.8` for `n`
- `shade_by_st()` helper
- `compute_key_times_for_shape()` helper
- HDF5 path resolution via the `GPU_EHL_DATA` environment variable

## Running figures

From the repo root:

```bash
python paper/figures/scripts/fig_restitution_map.py
```

For CI:

```bash
MPLBACKEND=Agg python paper/figures/scripts/fig_restitution_map.py
```

Or use the Makefile:

```bash
make figures
```

## Output

Generated PDFs are written to `paper/figures/output/` and ignored by Git.
