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


def _maybe_add_angle_indices(angle_indices, indices):
    "add indices if the tuple is not already collected"
    sorted_indices = sorted(indices)
    sorted_indices_collected = [sorted(i) for i in angle_indices]
    if sorted_indices not in sorted_indices_collected:
        angle_indices.append(indices)
    return angle_indices


def angle_indices_from_traj(traj):
    """indices of atoms forming an angle

    Get the list of atom indices for those triplets of
    atoms that are directly bonded.

    Args:
        traj: MDTraj trajectory

    Returns:
        angle_indices: array of angle indices, shape (n_angles, 3)
    """
    bond_indices = bond_indices_from_traj(traj)
    angle_indices = []
    for ai, aj in bond_indices:
        for bond in bond_indices:
            am, an = bond
            if ai in bond and aj not in bond:
                ak = am if ai != am else an
                angle_indices = _maybe_add_angle_indices(angle_indices, (ak, ai, aj))
            elif aj in bond and ai not in bond:
                ak = am if aj != am else an
                angle_indices = _maybe_add_angle_indices(angle_indices, (ai, aj, ak))
    return jnp.array(angle_indices, dtype=int)
