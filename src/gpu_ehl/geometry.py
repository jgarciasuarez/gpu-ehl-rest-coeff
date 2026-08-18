"""Grid construction and initial conditions for the gpu_ehl solver."""

import jax.numpy as jnp


def make_radial_grid(nr: int, lr: float) -> jnp.ndarray:
    """Create the radial grid directly on the JAX device.

    Parameters
    ----------
    nr : int
        Number of radial grid points.
    lr : float
        Outer radius of the domain.

    Returns
    -------
    jnp.ndarray
        1-D array of ``nr`` points from ``0.0`` to ``lr``.
    """
    return jnp.linspace(0.0, lr, nr, dtype=jnp.float32)


def make_time_grid(
    delta_t: float, nt: int, nt_ini: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create the initialization and dynamic time grids.

    Parameters
    ----------
    delta_t : float
        Time step size.
    nt : int
        Number of dynamic time steps.
    nt_ini : int
        Number of initialization time steps.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(Time_ini, Time)`` where ``Time_ini`` covers ``[-nt_ini*delta_t, 0]``
        and ``Time`` covers ``[0, nt*delta_t]``.
    """
    t1 = -nt_ini * delta_t
    Time_ini = jnp.linspace(t1, 0.0, nt_ini + 1, dtype=jnp.float32)
    Time = jnp.linspace(0.0, delta_t * nt, nt + 1, dtype=jnp.float32)
    return Time_ini, Time


def initial_gap_velocity(
    Time_ini: jnp.ndarray, exponent: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Prescribed gap and velocity during the initialization phase.

    Parameters
    ----------
    Time_ini : jnp.ndarray
        Initialization time grid; its first element is ``t1 < 0``.
    exponent : float
        Power-law shape exponent.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(D_ini, V_ini)`` arrays describing the prescribed approach.
    """
    D0 = 1.0
    t1 = Time_ini[0]
    sign_factor = jnp.where(exponent % 2 == 1, jnp.sign(Time_ini), 1.0)
    time_pow = sign_factor * (jnp.abs(Time_ini) ** exponent)
    D_ini = D0 - (Time_ini - time_pow / (2.0 * t1))
    V_ini = -1.0 + Time_ini / t1
    return D_ini, V_ini
