"""Lubrication pressure solve via the Schur complement system."""

import jax.numpy as jnp


def build_schur_system(
    w_old: jnp.ndarray,
    D_old: float,
    V_old: float,
    KernelMatrix: jnp.ndarray,
    r: jnp.ndarray,
    delta_t: float,
    dr: float,
    exponent: float,
    stokes: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build dense Schur matrix ``S`` and RHS for the pressure solve.

    Parameters
    ----------
    w_old : jnp.ndarray
        Elastic deformation at the previous time step.
    D_old : float
        Rigid-body gap at the previous time step.
    V_old : float
        Indenter velocity at the previous time step.
    KernelMatrix : jnp.ndarray
        Elastic influence kernel matrix.
    r : jnp.ndarray
        Radial grid.
    delta_t : float
        Time step size.
    dr : float
        Radial grid spacing.
    exponent : float
        Power-law shape exponent.
    stokes : float
        Stokes number.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        ``(S, RHS2)`` ready for ``jnp.linalg.solve(S, RHS2)``.
    """
    h = D_old + 0.5 * r ** exponent - w_old

    r_in = r[1:-1]
    h_in = h[1:-1]

    dw = (w_old[2:] - w_old[1:-1]) / dr

    alpha = 3.0 * (0.5 * exponent * r_in ** (exponent - 1.0) - dw) + h_in / r_in

    diag_p = 2.0 * h_in ** 3 + (h_in ** 2) * alpha * dr
    over_p = -(h_in ** 3 + h_in ** 2 * alpha * dr)
    under_p = -h_in ** 3

    zero = jnp.array([0.0], dtype=w_old.dtype)
    diag_full = jnp.concatenate([zero, diag_p, zero])  # length NR
    over_full = jnp.concatenate([zero, over_p, zero])  # length NR
    under_full = jnp.concatenate([zero, under_p, zero])  # length NR

    coeff = dr ** 2 / (stokes * delta_t)
    S = coeff * KernelMatrix

    NR = r.shape[0]
    idx = jnp.arange(NR)

    # Build tridiagonal contribution plus dense kernel term.
    S = S.at[idx, idx].add(diag_full)
    S = S.at[idx[:-1], idx[1:]].add(over_full[:-1])
    S = S.at[idx[1:], idx[:-1]].add(under_full[1:])

    # Boundary conditions.
    S = S.at[0, :].set(0.0)
    S = S.at[0, 0].set(1.0)
    S = S.at[0, 1].set(-1.0)

    S = S.at[-1, :].set(0.0)
    S = S.at[-1, -1].set(1.0)

    RHS2 = jnp.zeros_like(r)
    RHS2 = RHS2.at[1:-1].set(-(w_old[1:-1] + V_old * delta_t) * coeff)

    return S, RHS2


def solve_pressure(S: jnp.ndarray, RHS2: jnp.ndarray) -> jnp.ndarray:
    """Solve the linear system for pressure.

    Parameters
    ----------
    S : jnp.ndarray
        Schur matrix.
    RHS2 : jnp.ndarray
        Right-hand side.

    Returns
    -------
    jnp.ndarray
        Pressure field.
    """
    return jnp.linalg.solve(S, RHS2)
