from __future__ import annotations
from typing import Any, Tuple, List

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

Topology = Any
MMElectrostaticPotential = Any


def _get_name(top: Topology, i: int) -> str:
    "string encoding the atom name"
    resname = top.atom(i).resname
    resid = top.atom(i).resid
    original_resid = top.residue(resid).original_resid
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
        top: pytraj topology

    Returns:
        bond_indices: array of bond indices, shape (n_bonds, 2)
    """
    bond_indices = jnp.array(top.bond_indices, dtype=int)
    # enforce a convention for bond ordering here:
    # the first atom has a smaller index
    bond_indices = jnp.sort(bond_indices, axis=1)
    return bond_indices


def bond_indices_from_top(top: Topology) -> Tuple[Array, List[str]]:
    """indices of atoms forming a bond

    Get the list of atom indices for those atoms that
    are directly bonded.
    Also returns the atom names forming the bond.

    Args:
        top: pytraj topology

    Returns:
        bond_indices: array of bond indices, shape (n_bonds, 2)
        bond_atnames: list of atom names forming the bond
    """
    bond_indices = _bond_indices_from_top(top=top)
    bond_atnames = _atom_names_from_indices(top=top, indices=bond_indices)
    return bond_indices, bond_atnames


@with_lexicographically_sorted_output
def _angle_indices_from_top(top: Topology) -> Array:
    """indices of atoms forming an angle

    Get the list of atom indices for those triplets of
    atoms that are directly bonded.

    Args:
        top: pytraj topology

    Returns:
        angle_indices: array of angle indices, shape (n_angles, 3)
    """
    bond_indices = _bond_indices_from_top(top)
    angle_indices = angle_indices_from_bonds(bond_indices)
    return angle_indices


def angle_indices_from_top(top: Topology) -> Tuple[Array, List[str]]:
    """indices of atoms forming an angle

    Get the list of atom indices for those triplets of
    atoms that are directly bonded.
    Also returns the atom names forming the angle.

    Args:
        top: pytraj topology

    Returns:
        angle_indices: array of angle indices, shape (n_angles, 3)
        angle_atnames: list of atom names forming the angles.
    """
    angle_indices = _angle_indices_from_top(top=top)
    angle_atnames = _atom_names_from_indices(top=top, indices=angle_indices)
    return angle_indices, angle_atnames


def _dihe_indices_from_top(top: Topology) -> Array:
    """indices of atoms forming a dihedral

    Get the list of atom indices for those quartets of
    atoms that are directly bonded.

    Args:
        top: pytraj topology

    Returns:
        dihe_indices: array of dihedral indices, shape (n_diheds, 4)
    """
    bond_indices = _bond_indices_from_top(top)
    angle_indices = angle_indices_from_bonds(bond_indices)
    dihe_indices = dihe_indices_from_bonds_angles(bond_indices, angle_indices)
    return dihe_indices


def dihe_indices_from_top(top: Topology) -> Tuple[Array, List[str]]:
    """indices of atoms forming a dihedral

    Get the list of atom indices for those quartets of
    atoms that are directly bonded.
    Also returns the atom names forming the dihedrals.

    Args:
        top: pytraj topology

    Returns:
        dihe_indices: array of dihedral indices, shape (n_diheds, 4)
        dihe_atnames: list of atom names forming the dihedrals.
    """
    dihe_indices = _dihe_indices_from_top(top=top)
    dihe_atnames = _atom_names_from_indices(top=top, indices=dihe_indices)
    return dihe_indices, dihe_atnames


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
