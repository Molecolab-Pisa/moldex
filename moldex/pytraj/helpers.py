from __future__ import annotations
from typing import Any, Tuple

import warnings
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np

from ..helpers_utils import (
    with_lexicographically_sorted_output,
    angle_indices_from_bonds,
    dihe_indices_from_bonds_angles,
)

Trajectory = Any
Topology = Any
MMElectrostaticPotential = Any


@with_lexicographically_sorted_output
def bond_indices_from_traj(traj: Trajectory) -> Array:
    """indices of atoms forming a bond

    Get the list of atom indices for those atoms that
    are directly bonded.

    Args:
        traj: pytraj trajectory

    Returns:
        bond_indices: array of bond indices, shape (n_bonds, 2)
    """
    bond_indices = jnp.array(traj.top.bond_indices, dtype=int)
    # enforce a convention for bond ordering here:
    # the first atom has a smaller index
    bond_indices = jnp.sort(bond_indices, axis=1)
    return bond_indices


@with_lexicographically_sorted_output
def angle_indices_from_traj(traj: Trajectory) -> Array:
    """indices of atoms forming an angle

    Get the list of atom indices for those triplets of
    atoms that are directly bonded.

    Args:
        traj: pytraj trajectory

    Returns:
        angle_indices: array of angle indices, shape (n_angles, 3)
    """
    bond_indices = bond_indices_from_traj(traj)
    angle_indices = angle_indices_from_bonds(bond_indices)
    return angle_indices


def dihe_indices_from_traj(traj: Trajectory) -> Array:
    """indices of atoms forming a dihedral

    Get the list of atom indices for those quartets of
    atoms that are directly bonded.

    Args:
        traj: pytraj trajectory

    Returns:
        dihe_indices: array of dihedral indices, shape (n_diheds, 4)
    """
    bond_indices = bond_indices_from_traj(traj)
    angle_indices = angle_indices_from_bonds(bond_indices)
    dihe_indices = dihe_indices_from_bonds_angles(bond_indices, angle_indices)
    return dihe_indices


def _get_charges_db(charges_db: str) -> np.ndarray:
    """Reads a charges database from file

    Reads a database into a numpy array.
    The database is assumed to have at least three columns.
    Columns beyond the third will be discarded.

    The first column should contain the residue name.
    The second column should contain the atom name.
    The third column should contain the charge.

    The output is a numpy array with the first column
    with entries "residue_name atom_name" and the second
    column with the charges.
    """
    db_residues = np.loadtxt(charges_db, usecols=0, dtype=str)
    db_atnames = np.loadtxt(charges_db, usecols=1, dtype=str)
    db_charges = np.loadtxt(charges_db, usecols=2, dtype=float)
    # Array of 'RESIDUE_NAME ATOM_NAME' elements
    db = np.array(
        [
            (res + " " + name, q)
            for res, name, q in zip(db_residues, db_atnames, db_charges)
        ]
    )

    return db


def _get_charges(database: str, top: Topology, indices: ArrayLike) -> Array:
    """Get atom charges from a topology or a database

    Gets the atom charges from a database (if provided) or, alternatively,
    by reading them from a pytraj topology object. Only the charges
    corresponding to the given indices are retained.
    """
    if database is None:
        # MM charges taken from topology
        charges = jnp.asarray([a.charge for a in top.atoms])
        return charges[indices]
    else:
        # MM charges taken from charge database
        charges = []
        db = _get_charges_db(database)
        for atom in top.atoms:
            try:
                res = top.residue(atom.resid).name
                name = atom.name
                pattern = res.strip() + " " + name.strip()
                idx = np.where(db[:, 0] == pattern)[0][0]
                charge = float(db[idx][1])
            except IndexError:
                charge = atom.charge
                msg = f"Database charge for {res} {name} not found."
                msg += f" Taking from topology (q={charge:.6f})"
                warnings.warn(msg)
            charges.append(charge)

        return jnp.array(charges)[indices]


def _get_residues_array(top: Topology, mm_indices: ArrayLike) -> Array:
    """Gets an array with the starting position of each residue

    Builds an array storing the starting position of each residue in the
    pytraj topology. For example, for two residues, one of two atoms and one
    of four atoms, the residues_array would be [0, 2, 6], with the last number
    indicating the total number of atoms.
    """
    num_mm = jnp.array([len(mm_indices)])
    _, indices = jnp.unique(
        jnp.array([top.atom(i).resid for i in mm_indices]), return_index=True
    )
    residues_array = jnp.concatenate([indices, num_mm])

    return residues_array


def data_for_elecpot(
    top: Topology, qm_mask: str, charges_db: str = None, turnoff_mask: str = None
) -> Tuple[Array, Array, Array, Array]:
    """Helper for the electrostatic potential

    Provides the necessary arguments to instantiate the MMElectrostaticPotential
    class. In particular, given an Amber mask for the QM part, a (optional)
    charges database, and a (optional) mask for residues to be ignored (turnoff),
    provides the atom indices of the QM part, the atom indices of the MM part, the
    charges of the MM part, and the array indicating where each MM residue start.

    Args:
        top: pytraj topology
        qm_mask: Amber mask for selecting the QM part
        charges_db: path to the charges database
        turnoff_mask: Amber mask to ignore atoms in the MM part
    Returns:
        qm_indices: indices of the atoms in the QM part
        mm_indices: indices of the atoms in the MM part
        mm_charges: charges of the atoms in the MM part
        residues_array: starting positions (index of first atom)
                        of each MM residue.
    """
    mm_mask = "!" + qm_mask
    if turnoff_mask is not None:
        mm_mask = "(" + mm_mask + ")&(!" + turnoff_mask + ")"
    qm_indices = top.select(qm_mask)
    mm_indices = top.select(mm_mask)

    mm_charges = _get_charges(charges_db, top, mm_indices)
    residues_array = _get_residues_array(top, mm_indices)

    return qm_indices, mm_indices, mm_charges, residues_array


def visualize_cut_pdb(
    elecpot: MMElectrostaticPotential,
    coords: ArrayLike,
    top: Topology,
    outfile: str = "ml_env_cut.pdb",
) -> None:
    """Visualize the environment cut in the electrostatic potential

    Writes a PDB file to visualize the ML part and the environment part
    used in the calculation of the environment electrostatic potential on the
    ML atoms.

    The ML and environment parts are distinguished from their residue name:
    ML for the ML
    ENV for the environment

    Args:
        elecpot: MMElectrostaticPotential class
        coords: coordinates of the full system, shape (n_atoms, 3)
        outfile: name of the output file
    """
    _PDB_STD_FORMAT_ = (
        "ATOM  {:5d} {:<4s} {:3s} {:5d}    {:8.3f}{:8.3f}{:8.3f}  0.00  0.00  \n"
    )

    # True if a MM atom is included in the cutoff
    idx_cut = elecpot.cut_environment(coords)

    def mm_atom_is_included(i):
        return idx_cut[i]

    def get_data_for_pdb(i):
        atom_name = top.atom(i).name
        residue_id = top.atom(i).resid
        x, y, z = coords[i]
        return atom_name, residue_id, x, y, z

    with open(outfile, "w") as handle:
        # atom index
        atom_index = 1

        # QM part
        for i in elecpot.qm_indices:
            atom_name, residue_id, x, y, z = get_data_for_pdb(i)
            residue_name = "ML"
            handle.write(
                _PDB_STD_FORMAT_.format(
                    atom_index,
                    atom_name,
                    residue_name,
                    residue_id,
                    x,
                    y,
                    z,
                )
            )
            atom_index += 1

        # MM part
        for i_enum, i in enumerate(elecpot.mm_indices):
            if mm_atom_is_included(i_enum):
                atom_name, residue_id, x, y, z = get_data_for_pdb(i)
                residue_name = "ENV"
                handle.write(
                    _PDB_STD_FORMAT_.format(
                        atom_index,
                        atom_name,
                        residue_name,
                        residue_id,
                        x,
                        y,
                        z,
                    )
                )
                atom_index += 1
