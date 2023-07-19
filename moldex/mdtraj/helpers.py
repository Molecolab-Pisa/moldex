from __future__ import annotations
from typing import Any

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


@with_lexicographically_sorted_output
def bond_indices_from_top(top: Topology) -> Array:
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


@with_lexicographically_sorted_output
def angle_indices_from_top(top: Topology) -> Array:
    """indices of atoms forming an angle

    Get the list of atom indices for those triplets of
    atoms that are directly bonded.

    Args:
        top: MDTraj topology

    Returns:
        angle_indices: array of angle indices, shape (n_angles, 3)
    """
    bond_indices = bond_indices_from_top(top)
    angle_indices = angle_indices_from_bonds(bond_indices)
    return angle_indices


def dihe_indices_from_top(top: Topology) -> Array:
    """indices of atoms forming a dihedral

    Get the list of atom indices for those quartets of
    atoms that are directly bonded.

    Args:
        top: MDTraj topology

    Returns:
        dihe_indices: array of dihedral indices, shape (n_diheds, 4)
    """
    bond_indices = bond_indices_from_top(top)
    angle_indices = angle_indices_from_bonds(bond_indices)
    dihe_indices = dihe_indices_from_bonds_angles(bond_indices, angle_indices)
    return dihe_indices
