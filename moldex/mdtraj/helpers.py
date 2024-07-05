from __future__ import annotations
from typing import Any, Tuple, List, Optional

from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp

from ..helpers_utils import (
    with_lexicographically_sorted_output,
    angle_indices_from_bonds,
    dihe_indices_from_bonds_angles,
)

Topology = Any

# import warnings
#
# try:
#     import mdtraj as md
# except ImportError:
#     warnings.warn('MDTraj is not installed. MDTraj helpers not available.')


def _get_name(top: Topology, i: int) -> str:
    "string encoding the atom name"
    resname = top.atom(i).residue.name
    original_resid = top.atom(i).residue.index + 1
    atname = top.atom(i).name
    atom_name = resname + str(original_resid) + "@" + atname
    return atom_name


def _atom_names_from_indices(top: Topology, indices: ArrayLike) -> List[str]:
    """get the atom names from a matrix of indices

    Each row in the matrix corresponds to a tuple of atom indices
    linked together (e.g., atoms involved in a bond or an angle)
    """
    if indices.ndim != 2:
        raise ValueError(
            f"indices must be a 2D array, got indices.ndim = {indices.ndim}"
        )
    atom_names = []
    for tuple_indices in indices:
        atom_names.append([_get_name(top=top, i=i) for i in tuple_indices])
    return atom_names


@with_lexicographically_sorted_output
def _bond_indices_from_top(top: Topology) -> Array:
    """indices of atoms forming a bond

    Get the list of atom indices for those atoms that
    are directly bonded.

    Args:
        top: MDTraj topology

    Returns:
        bond_indices: array of bond indices, shape (n_bonds, 2)
    """
    bond_indices = []
    for bond in top.bonds:
        indices = (bond.atom1.index, bond.atom2.index)
        bond_indices.append(indices)
    bond_indices = jnp.array(bond_indices, dtype=int)
    # enforce a convention for bond ordering here:
    # first atom has smaller index
    bond_indices = jnp.sort(bond_indices, axis=1)
    return bond_indices


def bond_indices_from_top(top: Topology) -> Tuple[Array, List[str]]:
    """indices of atoms forming a bond

    Get the list of atom indices for those atoms that
    are directly bonded.
    Also returns the atom names involved in the bonds.

    Args:
        top: MDTraj topology

    Returns:
        bond_indices: array of bond indices, shape (n_bonds, 2)
        bond_atnames: list of atom names involved in the bond.
    """
    bond_indices = _bond_indices_from_top(top=top)
    bond_atnames = _atom_names_from_indices(top=top, indices=bond_indices)
    return bond_indices, bond_atnames


@with_lexicographically_sorted_output
def _angle_indices_from_top(top: Topology, legacy: Optional[bool] = False) -> Array:
    """indices of atoms forming an angle

    Get the list of atom indices for those triplets of
    atoms that are directly bonded.

    Args:
        top: MDTraj topology
        legacy: use the full Python (old) implementation instead of
                the Cython one.

    Returns:
        angle_indices: array of angle indices, shape (n_angles, 3)
    """
    bond_indices = _bond_indices_from_top(top)
    angle_indices = angle_indices_from_bonds(bond_indices, legacy=legacy)
    return angle_indices


def angle_indices_from_top(
    top: Topology, legacy: Optional[bool] = False
) -> Tuple[Array, List[str]]:
    """indices of atoms forming an angle

    Get the list of atom indices for those triplets of
    atoms that are directly bonded.
    Also returns the atom names involved in the angles.

    Args:
        top: MDTraj topology
        legacy: use the full Python (old) implementation instead of
                the Cython one.

    Returns:
        angle_indices: array of angle indices, shape (n_angles, 3)
        angle_atnames: list of atom names involved in the angles.
    """
    angle_indices = _angle_indices_from_top(top=top, legacy=legacy)
    angle_atnames = _atom_names_from_indices(top=top, indices=angle_indices)
    return angle_indices, angle_atnames


def _dihe_indices_from_top(top: Topology, legacy: Optional[bool] = False) -> Array:
    """indices of atoms forming a dihedral

    Get the list of atom indices for those quartets of
    atoms that are directly bonded.

    Args:
        top: MDTraj topology
        legacy: use the full Python (old) implementation instead of
                the Cython one.

    Returns:
        dihe_indices: array of dihedral indices, shape (n_diheds, 4)
    """
    bond_indices = _bond_indices_from_top(top)
    angle_indices = angle_indices_from_bonds(bond_indices, legacy=legacy)
    dihe_indices = dihe_indices_from_bonds_angles(
        bond_indices, angle_indices, legacy=legacy
    )
    return dihe_indices


def dihe_indices_from_top(
    top: Topology, legacy: Optional[bool] = False
) -> Tuple[Array, List[str]]:
    """indices of atoms forming a dihedral

    Get the list of atom indices for those quartets of
    atoms that are directly bonded.

    Args:
        top: MDTraj topology

    Returns:
        dihe_indices: array of dihedral indices, shape (n_diheds, 4)
    """
    dihe_indices = _dihe_indices_from_top(top=top, legacy=legacy)
    dihe_atnames = _atom_names_from_indices(top=top, indices=dihe_indices)
    return dihe_indices, dihe_atnames
