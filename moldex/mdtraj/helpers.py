from __future__ import annotations
from typing import Any, List

from jax import Array
import jax.numpy as jnp

Trajectory = Any

# import warnings
#
# try:
#     import mdtraj as md
# except ImportError:
#     warnings.warn('MDTraj is not installed. MDTraj helpers not available.')


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


def _maybe_add_indices(indices_list: List[Any], indices: Any) -> List[Any]:
    "add indices if the tuple is not already collected"
    sorted_indices = sorted(indices)
    sorted_indices_collected = [sorted(i) for i in indices_list]
    if sorted_indices not in sorted_indices_collected:
        indices_list.append(indices)
    return indices_list


def angle_indices_from_traj(traj: Trajectory) -> Array:
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
                angle_indices = _maybe_add_indices(angle_indices, (ak, ai, aj))
            elif aj in bond and ai not in bond:
                ak = am if aj != am else an
                angle_indices = _maybe_add_indices(angle_indices, (ai, aj, ak))
    return jnp.array(angle_indices, dtype=int)


def dihe_indices_from_traj(traj: Trajectory) -> Array:
    """indices of atoms forming a dihedral

    Get the list of atom indices for those quartets of
    atoms that are directly bonded.

    Args:
        traj: MDTraj trajectory

    Returns:
        dihe_indices: array of dihedral indices, shape (n_diheds, 4)
    """
    bond_indices = bond_indices_from_traj(traj)
    angle_indices = angle_indices_from_traj(traj)
    dihe_indices = []
    for ai, aj, ak in angle_indices:
        for bond in bond_indices:
            am, an = bond
            if ai in bond and aj not in bond and ak not in bond:
                al = am if ai != am else an
                dihe_indices = _maybe_add_indices(dihe_indices, (al, ai, aj, ak))
            elif ak in bond and ai not in bond and aj not in bond:
                al = am if ak != am else an
                dihe_indices = _maybe_add_indices(dihe_indices, (ai, aj, ak, al))
    return jnp.array(dihe_indices, dtype=int)
