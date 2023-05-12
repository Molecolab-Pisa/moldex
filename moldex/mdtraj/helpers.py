from __future__ import annotations
from typing import Any

from jax import Array
import jax.numpy as jnp

Trajectory = Any


def bond_indices_from_traj(traj: Trajectory) -> Array:
    """indices of atoms forming a bond

    Get the list of atom indices for those atoms that
    are directly bonded.

    Args:
        traj: MDTraj trajectory

    Returns:
        bond_indices: array of bond indices, shape (n_bonds, 2)
    """
    bond_indices = []
    for bond in traj.top.bonds:
        indices = (bond.atom1.index, bond.atom2.index)
        bond_indices.append(indices)
    return jnp.array(bond_indices, dtype=int)
