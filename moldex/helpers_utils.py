"""
This is a collection of useful utilities common to the moldex
helpers (pytraj, mdtraj, ...).
"""
from __future__ import annotations
from typing import Callable, List, Any

import functools


from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp


def lexicographic_matrix_sort(mat: ArrayLike) -> Array:
    """Sort the rows of a matrix lexicographically

    Solution taken from:
    https://tinyurl.com/muha64vp

    Args:
        mat: matrix, shape (n_rows, n_cols)
    Returns:
        sorted_matrix: matrix sorted lexicographically,
                       shape (n_rows, n_cols)
    Example:
        >>> x = jnp.array([[0, 0, 0, 2, 3],
                           [2, 3, 2, 3, 2],
                           [3, 1, 3, 0, 0],
                           [3, 1, 1, 3, 1]])
        >>> lexicographic_matrix_sort(x)
        ... Array([[0, 0, 0, 2, 3],
        ...        [2, 3, 2, 3, 2],
        ...        [3, 1, 1, 3, 1],
        ...        [3, 1, 3, 0, 0]], dtype=int32)
    """
    return mat[jnp.lexsort(jnp.rot90(mat))]


def with_lexicographically_sorted_output(func: Callable) -> Callable:
    """
    Applies the lexicographic sorting to the output of func
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return lexicographic_matrix_sort(res)

    return wrapper


def maybe_add_indices(indices_list: List[Any], indices: Any) -> List[Any]:
    "add indices if not already collected in indices_list"
    sorted_indices = sorted(indices)
    sorted_indices_collected = [sorted(i) for i in indices_list]
    if sorted_indices not in sorted_indices_collected:
        indices_list.append(indices)
    return indices_list


def angle_indices_from_bonds(bond_indices: ArrayLike) -> Array:
    """
    Finds the 3-tuple of angle indices from an array of bond
    indices

    Args:
        bond_indices: indices of atoms forming the bonds, shape (n_bonds, 2)
    Returns:
        angle_indices: indices of atoms forming the angles, shape (n_angles, 3)
    """
    angle_indices = []
    for ai, aj in bond_indices:
        for bond in bond_indices:
            am, an = bond
            if ai in bond and aj not in bond:
                ak = am if ai != am else an
                angle_indices = maybe_add_indices(angle_indices, (ak, ai, aj))
            elif aj in bond and ai not in bond:
                ak = am if aj != am else an
                angle_indices = maybe_add_indices(angle_indices, (ai, aj, ak))
    return jnp.array(angle_indices, dtype=int)


def dihe_indices_from_bonds_angles(
    bond_indices: ArrayLike, angle_indices: ArrayLike
) -> Array:
    """
    Finds the 4-tuple of dihedral indices from an array of angle indices

    Args:
        angle_indices: indices of atoms forming the angles,
                       shape (n_angles, 3)
    Returns:
        dihe_indices: indices of atoms forming the dihedrals,
                      shape (n_diheds, 4)
    """
    dihe_indices = []
    for ai, aj, ak in angle_indices:
        for bond in bond_indices:
            am, an = bond
            if ai in bond and aj not in bond and ak not in bond:
                al = am if ai != am else an
                dihe_indices = maybe_add_indices(dihe_indices, (al, ai, aj, ak))
            elif ak in bond and ai not in bond and aj not in bond:
                al = am if ak != am else an
                dihe_indices = maybe_add_indices(dihe_indices, (ai, aj, ak, al))
    return jnp.array(dihe_indices, dtype=int)
