"""Time-stepping functions: prescribed initialization and dynamic phases."""

import jax.numpy as jnp

from gpu_ehl.lubrication import build_schur_system, solve_pressure


def prescribed_step(
    w_old: jnp.ndarray,
    p_old: jnp.ndarray,
    D_t: float,
    V_t: float,
    KernelMatrix: jnp.ndarray,
    r: jnp.ndarray,
    delta_t: float,
    dr: float,
    exponent: float,
    stokes: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialization phase: ``D(t)`` and ``V(t)`` are prescribed.

    Parameters
    ----------
    w_old, p_old : jnp.ndarray
        Previous deformation and pressure.
    D_t, V_t : float
        Prescribed gap and velocity at this step.
    KernelMatrix : jnp.ndarray
        Elastic influence kernel.
    r : jnp.ndarray
        Radial grid.
    delta_t : float
        Time step size.
    dr : float
        Radial spacing.
    exponent : float
        Power-law shape exponent.
    stokes : float
        Stokes number.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(w_new, p_new)``.
    """
    S, RHS2 = build_schur_system(
        w_old, D_t, V_t, KernelMatrix, r, delta_t, dr, exponent, stokes
    )
    p_new = solve_pressure(S, RHS2)
    w_new = -KernelMatrix @ p_new
    return w_new, p_new


def dynamic_step(
    w_old: jnp.ndarray,
    p_old: jnp.ndarray,
    D_old: float,
    V_old: float,
    KernelMatrix: jnp.ndarray,
    r: jnp.ndarray,
    delta_t: float,
    dr: float,
    exponent: float,
    stokes: float,
) -> tuple[jnp.ndarray, jnp.ndarray, float, float, float, float, float, float]:
    """Main loop: solve pressure, update deformation, then update kinematics.

    Parameters
    ----------
    w_old, p_old : jnp.ndarray
        Previous deformation and pressure.
    D_old, V_old : float
        Previous gap and velocity.
    KernelMatrix : jnp.ndarray
        Elastic influence kernel.
    r : jnp.ndarray
        Radial grid.
    delta_t : float
        Time step size.
    dr : float
        Radial spacing.
    exponent : float
        Power-law shape exponent.
    stokes : float
        Stokes number.

    Returns
    -------
    tuple
        ``(w_new, p_new, D_new, V_new, Force, EnergyElastic, Central_h, Min_h)``.
    """
    S, RHS2 = build_schur_system(
        w_old, D_old, V_old, KernelMatrix, r, delta_t, dr, exponent, stokes
    )
    p_new = solve_pressure(S, RHS2)
    w_new = -KernelMatrix @ p_new

    Force = 4.0 * jnp.sum(p_new * r * dr)
    V_new = V_old + Force * delta_t
    D_new = D_old + V_new * delta_t

    EnergyElastic = -2.0 * jnp.sum(p_new * w_new * r * dr)
    Central_h = D_new - w_new[0]
    Min_h = jnp.min(D_new + 0.5 * r ** exponent - w_new)

    return w_new, p_new, D_new, V_new, Force, EnergyElastic, Central_h, Min_h
