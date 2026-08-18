# Research Pipeline

This document describes the end-to-end pipeline used to generate the results in
"Restitution of power-law elastic indenters in fluid-mediated impact".

## 1. Problem definition

A rigid-backed power-law elastic indenter approaches a flat substrate through a
thin incompressible viscous fluid film. During impact, the lubrication pressure
deforms the indenter, stores elastic energy, and eventually drives rebound. The
goal is to map the restitution coefficient as a function of:

- **Stokes number** `St` — ratio of impact inertia to viscous damping
- **Shape exponent** `n` — power-law profile `r^n / 2`

## 2. Governing equations

The solver integrates the coupled system:

- **Reynolds lubrication equation** for the thin-film pressure
- **Elastic half-space integral** relating pressure to surface deformation
- **Rigid-body force balance** updating indenter velocity and gap

See the solver modules in `src/gpu_ehl/` for the numerical implementation.

## 3. Parametric sweep

The full sweep covers 130 simulations:

```text
Stokes:  10, 17.8, 31.6, 56.2, 100, 177.8, 316.2, 562.3,
         1000, 1778.3, 3162.3, 5623.4, 10000
Shapes:  1.05, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
```

Simulations are batched on GPU via JAX `vmap` + `jit` and executed through
`scripts/slurm_sweep.sh` on the Kuma cluster.

## 4. Data export

Each simulation writes 100 snapshots of:

- `Force`, `V`, `Central_h`, `Min_h`, `EnergyElastic`
- Full radial profiles `w` (deformation) and `p` (pressure)
- Rigid-body gap `D`

The output is stored in `parametric_study_final.h5` with groups:

```text
/Stokes_X.X/Shape_Y.Y/
    Force
    V
    Central_h
    Min_h
    E_elastic
    w_snapshots
    p_snapshots
    D_snapshots
```

## 5. Post-processing and figures

The `paper/figures/scripts/` directory contains one script per paper figure.
Each script reads `parametric_study_final.h5` (or the sample dataset) and writes
a PDF to `paper/figures/output/`.

## 6. Reproducibility

- Dependencies are pinned in `environment.yml` and `pyproject.toml`.
- CI runs all figure scripts on the sample dataset with `MPLBACKEND=Agg`.
- The sample dataset (`data/sample/parametric_sample.h5`) is committed for
  offline testing and demos.
