from __future__ import annotations

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp


def _matrix_element(x: ArrayLike, x_ref: ArrayLike, bond_indices: ArrayLike) -> Array:
    """matrix element for the RE matrix

    Computes a single entry (r_ref / r), given an array
    of coordinates x, a reference array of coordinates x_ref,
    and a pair of indices of atoms involved in the bond

    Args:
        x: coordinates, shape (n_atoms, 3)
        x_ref: reference coordinates, shape (n_atoms, 3)
        bond_indices: atom indices (starting from 0) forming
                      the bond, shape (2,)
    Returns:
        mat_element: matrix element, shape ()
    """
    idx1, idx2 = bond_indices
    r = jnp.linalg.norm(x[idx1] - x[idx2])
    r_ref = jnp.linalg.norm(x_ref[idx1] - x_ref[idx2])
    return r_ref / r


def _re_matrix(x: ArrayLike, x_ref: ArrayLike, indices: ArrayLike) -> Array:
    mat_func = jax.vmap(_matrix_element, in_axes=(None, None, 0))
    return mat_func(x, x_ref, indices)


@jax.jit
def re_matrix(x: ArrayLike, x_ref: ArrayLike, indices: ArrayLike) -> Array:
    r"""RE matrix descriptor

    Computes the Relative-to-Equilibrium (RE) matrix descriptor for
    atoms with coordinates `x` w.r.t. a reference geometry `x_ref`, for
    bonds listed in `indices`.

        M_{ij} = |x_{ref,bi} - x_{ref,bj}| / |x_{bi} - x_{bj}|
                 foreach bi, bj in indices

    Args:
        x: coordinates, shape (n_atoms, 3)
        x_ref: reference coordinates, shape (n_atoms, 3)
        indices: indices of the atoms forming the bond, shape (n_bonds, 2)
                 Ex., if you want to compute the bond lengths between atoms
                 0 and 1, and 0 and 2, then

                 >>> indices = jnp.array([[0, 1], [0, 2]])

    Returns:
        descriptor: RE matrix, shape (n_bonds,)
    """
    return _re_matrix(x, x_ref, indices)


_re_matrix.__doc__ = re_matrix.__doc__

batched_re_matrix = jax.jit(jax.vmap(re_matrix, in_axes=(0, None, None)))

batched_re_matrix.__doc__ = """RE matrix descriptor

Computes the RE matrix along a trajectory.

Args:
    x: coordinates, shape (n_frames, n_atoms, 3)
    x_ref: reference coordinates, shape (n_atoms, 3)
    indices: indices of the atoms forming the bond, shape (n_bonds, 2)

Returns:
    descriptor: RE matrix, shape (n_frames, n_bonds)
"""
