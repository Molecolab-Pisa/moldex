from __future__ import annotations

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp


def _compute_dihedral(p1, p2, p3, p4):
    """compute a dihedral angle in radians

    It is assumed that the four points are bonded like:
        p1 -- p2 -- p3 -- p4
    """
    q1 = jnp.subtract(p2, p1)
    q1 = q1 / jnp.linalg.norm(q1)

    q2 = jnp.subtract(p3, p2)
    q2 = q2 / jnp.linalg.norm(q2)

    q3 = jnp.subtract(p4, p3)
    q3 = q3 / jnp.linalg.norm(q3)

    n1 = jnp.cross(q1, q2)
    n2 = jnp.cross(q2, q3)

    # define three orthogonal unit vectors
    u1 = n2
    u3 = q2
    u2 = jnp.cross(u3, u1)

    # projection along u1 and u2
    y = jnp.dot(n1, u2)
    x = jnp.dot(n1, u1)

    theta = -jnp.arctan2(y, x)
    return theta


def _matrix_element(x: ArrayLike, dihe_indices: ArrayLike) -> Array:
    """matrix element for the dihedral matrix

    Computes a single entry (dihedral) in radians, given
    an array of coordinate x and a quartet of indices of atoms
    involved in the dihedral.

    Args:
        x: coordinates, shape (n_atoms, 3)
        dihe_indices: atom indices (starting from 0) forming
                      the dihedral, shape (4,)

    Returns:
        mat_element: matrix element, shape ()
    """
    idx1, idx2, idx3, idx4 = dihe_indices
    p1, p2, p3, p4 = x[idx1], x[idx2], x[idx3], x[idx4]
    theta = _compute_dihedral(p1, p2, p3, p4)
    return theta


def _dihe_matrix(x: ArrayLike, indices: ArrayLike) -> Array:
    mat_func = jax.vmap(_matrix_element, in_axes=(None, 0))
    return mat_func(x, indices)


@jax.jit
def dihe_matrix(x: ArrayLike, indices: ArrayLike) -> Array:
    r"""dihedral matrix descriptor

    Computes the matrix (array) of dihedrals of atoms with coordinates
    `x` and dihedrals indexed by `indices`

        M_{i} = dihedral between x_{ai}, x_{aj}, x_{ak}, x_{al}
                foreach ai, aj, ak, al in indices

    Args:
        x: coordinates, shape (n_atoms, 3)
        indices: indices of the atoms forming the dihedral, shape (n_diheds, 4)
                 Ex. if you want to compute the dihedral between atoms 0 1 2 and 3,
                 and atoms 0 1 2 and 4, then

                 >>> indices = jnp.array([[0, 1, 2, 3], [0, 1, 2, 4]])

    Returns:
        descriptor: dihedrals, shape (n_diheds,)
    """
    return _dihe_matrix(x, indices)


_dihe_matrix.__doc__ = dihe_matrix.__doc__

batched_dihe_matrix = jax.jit(jax.vmap(dihe_matrix, in_axes=(0, None)))

batched_dihe_matrix.__doc__ = """dihedral matrix descriptor

Computes the dihedral matrix along a trajectory

Args:
    x: coordinates, shape (n_frames, n_atoms, 3)
    indices: indices of the atoms forming the dihedral, shape (n_diheds, 4)

Returns:
    descriptor: dihedrals, shape (n_frames, n_diheds)
"""
