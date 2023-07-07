from __future__ import annotations

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from ..cpp_extensions import retain_full_residues

# ===================================================================
# Basic operation
# ===================================================================


@jax.jit
def cut_box(coords1: ArrayLike, coords2: ArrayLike, cutoff: float) -> Array:
    """Cuts a cubic box around the atoms in coords1

    Cuts a cubic box around the atoms in coords1.
    The box extends a cutoff distance below/above the minimum/maximum
    position of atoms in coords1 in the three cartesian directions.

    Args:
        coords1: shape (n_atoms_1, 3)
        coords2: shape (n_atoms_2, 3)
        cutoff: cutoff value to cut the box
    Returns:
        idx0: True/False if an atom in coords2 is within/outside
              the cutted box, shape (n_atoms_2,)
    """
    # Select atoms in a box with length >= 2 * self.cutoff
    m1x = jnp.where(coords2[:, 0] < jnp.min(coords1[:, 0] - cutoff), 0, 1)
    m2x = jnp.where(coords2[:, 0] > jnp.max(coords1[:, 0] + cutoff), 0, 1)
    m1y = jnp.where(coords2[:, 1] < jnp.min(coords1[:, 1] - cutoff), 0, 1)
    m2y = jnp.where(coords2[:, 1] > jnp.max(coords1[:, 1] + cutoff), 0, 1)
    m1z = jnp.where(coords2[:, 2] < jnp.min(coords1[:, 2] - cutoff), 0, 1)
    m2z = jnp.where(coords2[:, 2] > jnp.max(coords1[:, 2] + cutoff), 0, 1)
    # True if an atom is within the box, False otherwise. We consider residues as a whole
    idx0 = jnp.min(jnp.row_stack([m1x, m2x, m1y, m2y, m1z, m2z]), axis=0).astype(bool)
    # Need to run a "same_residue_as" here.
    # UPDATE: we don't run it here and we computed the distances 2 times (less expensive)
    # idx0 = retain_full_residues(idx0, residues_array).astype(bool)
    return idx0


# ===================================================================
# Descriptor (function level)
# ===================================================================


@jax.jit
def distances(coords1: ArrayLike, coords2: ArrayLike) -> Array:
    """Pairwise distances between atoms in coords1 and coords2

    Args:
        coords1: shape (n_atoms_1, 3)
        coords2: shape (n_atoms_2, 3)
    Returns:
        distances: shape (n_atoms_1, n_atoms_2)
    """
    return jnp.sum((coords1[:, None] - coords2) ** 2, axis=-1) ** 0.5


@jax.jit
def compute_potential(charges2: ArrayLike, idx: ArrayLike, dd: ArrayLike) -> Array:
    """Electrostatic potential

    Args:
        charges2: set of charges, shape (n_atoms_2,)
        idx: 1/0 if an atom is included/excluded, shape (n_atoms_2,)
        dd: pairwise distances between 1 and 2, shape (n_atoms_1, n_atoms_2)
    Returns:
        potential: electrostatic potential on the atoms of 1
                   shape (n_atoms_1,)
    """
    return jnp.sum(charges2 * idx / dd, axis=1)


def _electrostatic_potential(
    coords1: ArrayLike,
    coords2: ArrayLike,
    charges2: ArrayLike,
    residues_array: ArrayLike,
    cutoff: float,
) -> Array:
    """Computes the electrostatic potential of 2 onto the atoms of 1

    Computes the electrostatic potential of the atoms in coords2 onto
    the atoms at coords1.

    Args:
        coords1: shape (n_atoms_1, 3)
        coords2: shape (n_atoms_2, 3)
        charges2: shape (n_atoms_2,)
        residues_array: starting position of each residue in 2, with
                        the last element equal to the number of atoms
                        in 2, shape (n_residues_2 + 1,)
        cutoff: residues in 2 within the cutoff are retained
                Note: residues are retained as a whole, i.e., none of
                      residues in 2 is truncated/cutted.
    Returns:
        potential: electrostatic potential on the atoms of 1,
                   shape (n_atoms_1,)
    """
    idx0 = cut_box(coords1, coords2, cutoff)
    mask = idx0.copy()
    dd = distances(coords1, coords2[mask])
    idx_cut = jnp.max(jnp.where(dd < cutoff, 1, 0), axis=0).astype(bool)
    idx0 = idx0.at[mask].set(idx_cut)
    idx = retain_full_residues(idx0.astype(int), residues_array)
    mask = idx.copy().astype(bool)
    dd = distances(coords1, coords2[mask])
    pot = compute_potential(charges2[mask], idx[mask], dd)
    return pot


def electrostatic_potential(
    coords1: ArrayLike,
    coords2: ArrayLike,
    charges2: ArrayLike,
    residues_array: ArrayLike,
    cutoff: float,
) -> Array:
    return _electrostatic_potential(coords1, coords2, charges2, residues_array, cutoff)


electrostatic_potential.__doc__ = _electrostatic_potential.__doc__


# ===================================================================
# Descriptor class
# ===================================================================


class MMElectrostaticPotential:
    "Electrostatic Potential of the MM atoms"

    def __init__(
        self,
        qm_indices: ArrayLike,
        mm_indices: ArrayLike,
        mm_charges: ArrayLike,
        residues_array: ArrayLike,
        cutoff: float,
    ) -> None:
        """
        Args:
            qm_indices: indices of the atoms in the QM part
            mm_indices: indices of the atoms in the MM part
            mm_charges: charges of the atoms in the MM part
            residues_array: starting position of each residue in 2, with
                            the last element equal to the number of atoms
                            in 2, shape (n_residues_2 + 1,)
            cutoff: residues in 2 within the cutoff are retained
                    Note: residues are retained as a whole, i.e., none of
                          residues in 2 is truncated/cutted.
        """
        self.qm_indices = qm_indices
        self.mm_indices = mm_indices
        self.mm_charges = mm_charges
        self.residues_array = residues_array
        self.cutoff = cutoff

    def _encode(self, coords: ArrayLike) -> Array:
        qm_coords = coords[self.qm_indices]  # Angstrom
        mm_coords = coords[self.mm_indices]  # Angstrom
        return electrostatic_potential(
            qm_coords, mm_coords, self.mm_charges, self.residues_array, self.cutoff
        )

    def compute(self, coords: ArrayLike) -> Array:
        """Computes the electrostatic potential on the QM part

        Args:
            coords: coordinates of the whole system, shape (n_atoms, 3)
        """
        return self._encode(coords)


# These functions represent efforts to write retain_full_residues in pure JAX
# (before we wrote our C++ function)

# @jax.jit
# def retain_full_residues(idx,residue_array):
#    new_idx = idx.copy()
#    n_resid = residue_array.shape[0]
#    indices = jnp.arange(idx.shape[0])
#
#    def func(i, new_idx):
#        start = residue_array[i]
#        stop = residue_array[i+1]
#        idx_sliced = jnp.where((indices >= start) & (indices < stop), idx, 0.0)
#        sum_ = jnp.sum(idx_sliced)
#        new_idx = jnp.where(sum_ > 0.5, jnp.where((indices >= start) & (indices < stop), 1, new_idx), new_idx)
#        return new_idx
#
#    new_idx = jax.lax.fori_loop(0, n_resid-1, func, new_idx)
#    return new_idx


# def retain_full_residues_new(idx,residues_array):
#    segment_ids = jnp.repeat(jnp.arange(len(residues_array)), jnp.diff(jnp.append(residues_array, len(idx))))
#    sum_by_segment = jax.ops.segment_sum(idx, segment_ids)
#    sum_by_segment = jnp.where(sum_by_segment > 0.5,1,0)
#    lengths_residues = jnp.diff(residues_array)
#    new_idx = sum_by_segment.repeat(lengths_residues)
#    return new_idx
#

# ===
# This function is a vectorized implementation
# Credits to David Yoshida
# https://stackoverflow.com/questions/76617821/selecting-all-elements-of-subsets-if-at-least-one-element-is-selected-jax
# This version is fast (I don't know how fast as compared to the C++ version) but very memory intensive

# @jax.jit
# def vectorized_select_subsets(arr, subsets):
#     l, = arr.shape
#
#     indices = jnp.arange(l)[None, :]
#
#     # Broadcast to mask of shape (n_subsets, input_length)
#     subset_masks = (
#         (indices >= subsets[:-1, None])
#         & (indices < subsets[1:, None])
#     )
#
#     # Shape (n_subsets,) array indicating whether each subset is included
#     include_subset = jnp.any(subset_masks & arr[None, :], axis=1)
#
#     # Reduce down columns
#     result = jnp.any(subset_masks & include_subset[:, None], axis=0).astype(jnp.int32)
#     return result
# ===
