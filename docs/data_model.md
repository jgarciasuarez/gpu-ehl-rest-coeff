# Data Model

The HDF5 output follows a hierarchical group structure that maps each
simulation to its `(Stokes, shape)` parameters.

## File: `parametric_study_float64.h5`

### Root-level datasets

| Dataset | Shape | Description |
|---|---|---|
| `r_grid` | `(NR,)` | Radial coordinates |
| `time_grid` | `(NT + 1,)` | Dynamic-phase time coordinates |
| `snapshot_indices` | `(num_snapshots,)` | Step indices of stored snapshots |

### Case groups

Groups are named according to the stored parameter labels:

```text
/Stokes_<stokes>.<decimals>/Shape_<shape>.<decimals>/
```

For example, the physical case `St = 100`, `n = 1.05` is stored as:

```text
/Stokes_100.0/Shape_1.0/
```

because the HDF5 uses the integer label `1.0` to represent `n = 1.05`.

### Per-case datasets

| Dataset | Shape | Description |
|---|---|---|
| `Force` | `(num_snapshots,)` | Hydrodynamic force |
| `V` | `(num_snapshots,)` | Indenter velocity |
| `Central_h` | `(num_snapshots,)` | Gap at the center |
| `Min_h` | `(num_snapshots,)` | Minimum gap over radius |
| `E_elastic` | `(num_snapshots,)` | Elastic strain energy |
| `w_snapshots` | `(num_snapshots, NR)` | Elastic deformation profiles |
| `p_snapshots` | `(num_snapshots, NR)` | Pressure profiles |
| `D_snapshots` | `(num_snapshots,)` | Rigid-body gap |

> **Note:** `Min_h` is written by the current version of `gpu_ehl.io.save_results_h5`.
> The initial Zenodo release (DOI: 10.5281/zenodo.21995894) was produced with an
> earlier output routine and therefore does not contain this field.

## Version 2: mixed-resolution dataset

Version 2 of the dataset (`parametric_study_float64_v2.h5`) replaces the
highest-Stokes cases with fine-resolution re-runs (larger `nr`, smaller
`delta_t`), because only high-Stokes impacts approach the spatial resolution
limit (neck width ~ a few `dr`).

Structure (extends the layout above):

- Root-level `r_grid`, `time_grid`, `snapshot_indices` describe the
  **coarse reference resolution** (as in version 1).
- Every case group carries provenance attributes: `resolution`
  (`"coarse"` or `"fine"`), `nr`, and `delta_t`.
- **Fine-resolution case groups additionally store their own**
  `r_grid`, `time_grid`, `snapshot_indices` **inside the group**, since the
  root-level grids do not describe their resolution.
- File-level attributes: `version`, `coarse_nr`, `coarse_delta_t`,
  `fine_nr`, `fine_delta_t`, and `fine_cases` (JSON list of the re-run
  group names).

### Loading rules

- Readers must use a case's own grid when the group contains
  `r_grid`/`time_grid`, and fall back to the root-level grids otherwise.
  This is implemented in:
  - `gpu_ehl.io.load_results_h5` (per-case grids exposed in the case dict),
  - `paper/figures/scripts/scaling_common.extract_scaling_data`,
  - `paper/figures/scripts/style.py` (`case_radial_grid`, `case_snap_times`,
    and `compute_key_times_for_shape`),
  - `paper/figures/scripts/verify_scalings.py`.
- Version-1 files are unaffected: they have no group-level grids, so all
  readers fall back to the root-level grids.

### Tools

- `scripts/merge_fine_subset.py` builds the version-2 file from the
  coarse archive and the fine-subset sweep, with built-in verification.
- `scripts/check_fine_coarse_convergence.py` quantifies coarse-vs-fine
  differences on the re-run subset.

## Sample dataset

`data/sample/parametric_sample.h5` contains a subset of the full dataset:

- For `n = 2.0`: `St = 10, 100, 1000, 10000`
- For all other shapes: `St = 100` only

Total: 13 simulations, ~27 MB.

## Loading data

Use the provided I/O helper:

```python
from gpu_ehl.io import load_results_h5

data = load_results_h5("data/sample/parametric_sample.h5")
r = data["r_grid"]
case = data["cases"]["Stokes_100.0"]["Shape_2.0"]
force = case["Force"]
```
