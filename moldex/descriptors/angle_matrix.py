from __future__ import annotations

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp


def _matrix_element(x: ArrayLike, angle_indices: ArrayLike) -> Array:
    """matrix element for the angle matrix

    Computes a single entry (angle) in radians, given
    an array of coordinates x and a triplet of indices of atoms
    involved in the angle.

    Args:
        x: coordinates, shape (n_atoms, 3)
        angle_indices: atom indices (starting from 0) forming
                       the angle, shape (3,)

    Returns:
        mat_element: matrix element, shape ()
    """
    idx1, idx2, idx3 = angle_indices
    uvec1 = x[idx1] - x[idx2]
    uvec1 = uvec1 / jnp.linalg.norm(uvec1)
    uvec2 = x[idx3] - x[idx2]
    uvec2 = uvec2 / jnp.linalg.norm(uvec2)
    theta = jnp.arccos(jnp.dot(uvec1, uvec2))
    return theta


def _angle_matrix(x: ArrayLike, indices: ArrayLike) -> Array:
    mat_func = jax.vmap(_matrix_element, in_axes=(None, 0))
    return mat_func(x, indices)


@jax.jit
def angle_matrix(x: ArrayLike, indices: ArrayLike) -> Array:
    r"""angle matrix descriptor

    Computes the matrix (array) of angles of atoms with coordinates
    `x` and angles indexed by `indices`

        M_{i} = angle between (x_{ai} - x_{aj}) and (x_{ak} - x_{aj})
                foreach ai, aj, ak in indices

    Args:
        x: coordinates, shape (n_atoms, 3)
        indices: indices of the atoms forming the angle, shape (n_angles, 3)
                 Ex. if you want to compute the angle between atoms 0 1 and 2,
                 and atoms 0 1 and 3, then

                 >>> indices = jnp.array([[0, 1, 2], [0, 1, 3]])

    Returns:
        descriptor: angles, shape (n_angles,)
    """
    return _angle_matrix(x, indices)


_angle_matrix.__doc__ = angle_matrix.__doc__

batched_angle_matrix = jax.jit(jax.vmap(angle_matrix, in_axes=(0, None)))

batched_angle_matrix.__doc__ = """angle matrix descriptor

Computes the angle matrix along a trajectory

Args:
    x: coordinates, shape (n_frames, n_atoms, 3)
    indices: indices of the atoms forming the angle, shape (n_angles, 3)

Returns:
    descriptor: angles, shape (n_frames, n_angles)
"""
