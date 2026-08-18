# Architecture

`gpu_ehl` is organized as a small, modular JAX package.

## Module overview

| Module | Responsibility |
|---|---|
| `config.py` | `SolverConfig` dataclass |
| `geometry.py` | Radial/time grids, prescribed initial conditions |
| `elasticity.py` | Half-space kernel matrix construction and caching |
| `lubrication.py` | Schur-complement system and pressure solve |
| `dynamics.py` | Prescribed initialization step and dynamic step |
| `solver.py` | `simulate_single_setup` — full single-case simulation |
| `sweep.py` | Parameter grid, GPU-batched sweep, chunking |
| `io.py` | HDF5 read/write |
| `cli.py` | `jax-ehl` command-line interface |

## Data flow

```
SolverConfig
    |
    v
make_radial_grid  +  load_or_build_kernel
    |                     |
    +----------+----------+
               v
      simulate_single_setup
               |
      +--------+--------+
      |                 |
   prescribed_step   dynamic_step
      |                 |
   build_schur_system  build_schur_system
              |              |
              v              v
         jnp.linalg.solve
              |
              v
           history dict
              |
              v
         save_results_h5
```

## Design decisions

- **Plain dataclass** for configuration (no external validation library).
- **Kernel cache** stored in `~/.cache/gpu-ehl/kernels/` to avoid recomputation.
- **Logging** via Python’s standard `logging` module.
- **CPU fallback** supported for single runs; sweeps require GPU.
- **Type hints** throughout.

## CLI

After installation, two commands are available:

```bash
# Single simulation (works on CPU or GPU)
jax-ehl single --stokes 100 --exponent 2.0 --output result.h5

# Full parametric sweep (GPU only)
jax-ehl sweep --output parametric_study.h5
```
