# Data Model

The HDF5 output follows a hierarchical group structure that maps each
simulation to its `(Stokes, shape)` parameters.

## File: `parametric_study_final.h5`

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
