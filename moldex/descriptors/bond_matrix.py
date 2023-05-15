from __future__ import annotations

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp


def _matrix_element(x: ArrayLike, bond_indices: ArrayLike) -> Array:
    """matrix element for the bond matrix

    Computes a single entry (bond length), given an array
    of coordinates x and a pair of indices of atoms
    involved in the bond

    Args:
        x: coordinates, shape (n_atoms, 3)
        bond_indices: atom indices (starting from 0) forming
                      the bond, shape (2,)
    Returns:
        mat_element: matrix element, shape ()
    """
    idx1, idx2 = bond_indices
    norm = jnp.linalg.norm(x[idx1] - x[idx2])
    return norm


def _bond_matrix(x: ArrayLike, indices: ArrayLike) -> Array:
    mat_func = jax.vmap(_matrix_element, in_axes=(None, 0))
    return mat_func(x, indices)


@jax.jit
def bond_matrix(x: ArrayLike, indices: ArrayLike) -> Array:
    r"""bond matrix descriptor

    Computes the matrix (array) of bond lengths of atoms with
    coordinates `x` and bonds indexed by `indices`

        M_{i} = |x_{bi} - x_{bj}| foreach bi, bj in indices

    Args:
        x: coordinates, shape (n_atoms, 3)
        indices: indices of the atoms forming the bond, shape (n_bonds, 2)
                 Ex., if you want to compute the bond lengths between atoms
                 0 and 1, and 0 and 2, then

                 >>> indices = jnp.array([[0, 1], [0, 2]])

    Returns:
        descriptor: bond lengths, shape (n_bonds,)
    """
    return _bond_matrix(x, indices)


_bond_matrix.__doc__ = bond_matrix.__doc__

batched_bond_matrix = jax.jit(jax.vmap(bond_matrix, in_axes=(0, None)))

batched_bond_matrix.__doc__ = """bond matrix descriptor

Computes the bond matrix along a trajectory.

Args:
    x: coordinates, shape (n_frames, n_atoms, 3)
    indices: indices of the atoms forming the bond, shape (n_bonds, 2)

Returns:
    descriptor: bond_lengths, shape (n_frames, n_bonds)
"""
