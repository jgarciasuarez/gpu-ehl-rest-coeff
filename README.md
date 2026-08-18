# gpu-ehl-rest-coeff

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.6%2B-orange)](https://jax.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-blue)](https://numpy.org/)
[![HDF5](https://img.shields.io/badge/HDF5-3.8%2B-darkgreen)](https://www.h5py.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-blueviolet)](https://matplotlib.org/)
[![Slurm](https://img.shields.io/badge/Slurm-workload%20manager-blue)](https://slurm.schedmd.com/)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21995894.svg)](https://doi.org/10.5281/zenodo.21995894)

> **The problem we are addressing:** When a power-law elastic object moves through a thin viscous fluid film towards a rigid substrate, the coupled elastohydrodynamic process is governed by the Stokes number and the indenter shape exponent. This repository provides the data, solver, and figure-reproduction pipeline for our study of the restitution coefficient as a function of Stokes number and indenter shape.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Parameter  │ --> │  JAX/GPU    │ --> │  HDF5       │ --> │  Figure     │
│  grid       │     │  solver     │     │  dataset    │     │  scripts    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      ^                    |                  |                    |
      └──── Slurm sweep ───┘                  └──── notebooks ─────┘
```

- `src/gpu_ehl/` — modular JAX solver (geometry, elasticity, lubrication, dynamics, sweep, I/O)
- `data/` — sample dataset for CI/demos; full dataset can be fetched via `data/download_data.py`
- `paper/figures/scripts/` — scripts that reproduce paper figures
- `notebooks/` — interactive exploration and reproduction:
  - `01_reproduce_figures.ipynb` — regenerate the paper figures from the sample dataset.
  - `02_explore_restitution_map.ipynb` — interactive 3D restitution map.
  - `03_solver_demo_convergence.ipynb` — run the solver and check grid convergence.
  - `04_profiling_rationale.ipynb` — analyze GPU profiling data.

## Project scale

| Metric | Value |
|---|---|
| Simulations | 130 (13 Stokes numbers × 10 shape exponents) |
| Stokes numbers | `10^1` to `10^4` |
| Shape exponents | `1.05` to `10.0` |
| Radial grid | `NR = 3000`, domain `LR = 6.0` |
| GPU hardware | NVIDIA H100 on the Kuma cluster at EPFL |
| Final dataset | ~300 MB (`parametric_study_final.h5`, Zenodo) |
| Sample dataset | ~27 MB, 13 simulations (`data/sample/parametric_sample.h5`) |

## Quickstart

### 1. Install

```bash
git clone https://github.com/jgarciasuarez/gpu-ehl-rest-coeff.git
cd gpu-ehl-rest-coeff
conda env create -f environment.yml
conda activate gpu-ehl
pip install -e ".[dev]"
```

For GPU execution, ensure JAX is installed with the CUDA backend (`jax[cuda]`).

### 2. Download the full dataset

The full dataset (`parametric_study_final.h5`, ~0.3 GB) is archived on Zenodo:
[![DOI 10.5281/zenodo.21995894](https://zenodo.org/badge/DOI/10.5281/zenodo.21995894.svg)](https://doi.org/10.5281/zenodo.21995894)

```bash
python data/download_data.py
# or point to a local copy
export GPU_EHL_DATA=/path/to/parametric_study_final.h5
```

### 3. Run a single simulation

```bash
jax-ehl single --stokes 100 --exponent 2.0 --output result.h5
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Reproduce figures

```bash
make figures
```

## Solver release note

The modular solver in `src/gpu_ehl/` is included in this repository. The full
research pipeline — raw parametric sweep, post-processing, and figure scripts —
is available now. The dataset is archived on Zenodo with DOI
[10.5281/zenodo.21995894](https://doi.org/10.5281/zenodo.21995894).

## Citation

If you use this work, please cite:

> Joaquin Garcia-Suarez ... (TBD)

DOI links for the paper:  (will be added once the manuscript is published).

## License

This project is released under the [MIT License](LICENSE).
