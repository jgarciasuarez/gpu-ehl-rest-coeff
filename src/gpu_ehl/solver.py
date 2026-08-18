"""High-level solver: run a single simulation from initial approach to rebound."""

import jax
import jax.numpy as jnp

from gpu_ehl.dynamics import dynamic_step, prescribed_step
from gpu_ehl.geometry import initial_gap_velocity, make_time_grid


def simulate_single_setup(
    stokes: float,
    exponent: float,
    KernelMatrix: jnp.ndarray,
    r: jnp.ndarray,
    delta_t: float,
    dr: float,
    nt: int,
    nt_ini: int,
    num_snapshots: int = 100,
) -> dict[str, jnp.ndarray]:
    """Run one full simulation for a given (Stokes, exponent) pair.

    Parameters
    ----------
    stokes : float
        Stokes number.
    exponent : float
        Power-law shape exponent.
    KernelMatrix : jnp.ndarray
        Elastic influence kernel matrix.
    r : jnp.ndarray
        Radial grid.
    delta_t : float
        Time step size.
    dr : float
        Radial grid spacing.
    nt : int
        Number of dynamic time steps.
    nt_ini : int
        Number of initialization time steps.
    num_snapshots : int, optional
        Number of snapshots to store (default: 100).

    Returns
    -------
    dict[str, jnp.ndarray]
        History dictionary with keys ``Force``, ``EnergyElastic``, ``Central_h``,
        ``Min_h``, ``D``, ``V``, ``w``, ``p``. Each value has shape
        ``(num_snapshots, ...)``.
    """
    Time_ini, _ = make_time_grid(delta_t, nt, nt_ini)
    D_ini, V_ini = initial_gap_velocity(Time_ini, exponent)

    w0 = jnp.zeros(r.shape[0], dtype=jnp.float32)
    p0 = jnp.zeros(r.shape[0], dtype=jnp.float32)

    def init_step(carry, inputs):
        w, p = carry
        D_t, V_t = inputs
        w_new, p_new = prescribed_step(
            w, p, D_t, V_t, KernelMatrix, r, delta_t, dr, exponent, stokes
        )
        return (w_new, p_new), None

    (w, p), _ = jax.lax.scan(init_step, (w0, p0), (D_ini[:-1], V_ini[:-1]))

    D0_dyn = D_ini[-1]
    V0_dyn = V_ini[-1]

    steps_per_snapshot = nt // num_snapshots

    def snapshot_step(carry, _):
        def physics_step(i, state):
            w, p, D, V = state
            w_new, p_new, D_new, V_new, _, _, _, _ = dynamic_step(
                w, p, D, V, KernelMatrix, r, delta_t, dr, exponent, stokes
            )
            return (w_new, p_new, D_new, V_new)

        new_state = jax.lax.fori_loop(
            0, steps_per_snapshot, physics_step, carry
        )
        w_new, p_new, D_new, V_new = new_state

        F = 4.0 * jnp.sum(p_new * r * dr)
        E = -2.0 * jnp.sum(p_new * w_new * r * dr)
        hc = D_new - w_new[0]
        hmin = jnp.min(D_new + 0.5 * r ** exponent - w_new)

        history = {
            "Force": F,
            "EnergyElastic": E,
            "Central_h": hc,
            "Min_h": hmin,
            "D": D_new,
            "V": V_new,
            "w": w_new,
            "p": p_new,
        }
        return new_state, history

    init_carry = (w, p, D0_dyn, V0_dyn)
    _, history = jax.lax.scan(
        snapshot_step, init_carry, None, length=num_snapshots
    )

    return history
