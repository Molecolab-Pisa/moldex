from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import custom_jvp, jit, vmap
from jax.typing import ArrayLike
from jax import Array

# ===================================================================
# Basic operation
# ===================================================================

# A custom_jvp is used as to avoid NaN propagations when using jnp.where.


@custom_jvp
def _cm_matrix_element(
    x1: ArrayLike, z1: ArrayLike, x2: ArrayLike, z2: ArrayLike
) -> Array:
    """matrix element of coulomb matrix

    This function computes a single matrix element of
    the inverse distance matrix. x1 and x2 are intended
    to be single samples.

    Args:
        x1: first input, shape (n_features,)
        z1: charges of first input, shape (1,)
        x2: second input, shape (n_features,)
        z2: charges of second input, shape (1,)
    Returns:
        mat_element: matrix element, shape ()
    """
    norm = jnp.linalg.norm(x1 - x2)
    not_zero = norm > 1e-20
    return jnp.where(not_zero, (z1 * z2) / norm, 0.5 * jax.lax.abs(z1) ** 2.4)


def _cm_matrix_element_tangent_arg0(
    x1: ArrayLike, z1: ArrayLike, x2: ArrayLike, z2: ArrayLike
) -> Array:
    """tangent wrt x1 for a single matrix element

    Args:
        x1: first input, shape (n_features,)
        z1: charges of first input, shape (1,)
        x2: second input, shape (n_features,)
        z2: charges of second input, shape (1,)
    Returns:
        tanget: tangent wrt x1, shape (n_features,)
    """
    diff = x1 - x2
    norm3 = jnp.linalg.norm(diff) ** 3
    not_zero = norm3 > 1e-20
    return jnp.where(not_zero, -((z1 * z2) * diff) / norm3, 0.0)


def _cm_matrix_element_tangent_arg1(
    x1: ArrayLike, z1: ArrayLike, x2: ArrayLike, z2: ArrayLike
) -> Array:
    """tangent wrt z1 for a single matrix element

    Args:
        x1: first input, shape (n_features,)
        z1: charges of first input, shape (1,)
        x2: second input, shape (n_features,)
        z2: charges of second input, shape (1,)
    Returns:
        tanget: tangent wrt z1, shape ()
    """
    norm = jnp.linalg.norm(x1 - x2)
    not_zero = norm > 1e-20
    return jnp.where(
        not_zero,
        _cm_matrix_element(x1, z1, x2, z2) / z1,
        0.5 * 2.4 * jax.lax.abs(z1) ** (1.4),
    )


def _cm_matrix_element_tangent_arg2(
    x1: ArrayLike, z1: ArrayLike, x2: ArrayLike, z2: ArrayLike
) -> Array:
    """tangent wrt x2 for a single matrix element

    Args:
        x1: first input, shape (n_features,)
        z1: charges of first input, shape (1,)
        x2: second input, shape (n_features,)
        z2: charges of second input, shape (1,)
    Returns:
        tanget: tangent wrt x2, shape (n_features,)
    """
    return -_cm_matrix_element_tangent_arg0(x1, z1, x2, z2)


def _cm_matrix_element_tangent_arg3(
    x1: ArrayLike, z1: ArrayLike, x2: ArrayLike, z2: ArrayLike
) -> Array:
    """tangent wrt z2 for a single matrix element

    Args:
        x1: first input, shape (n_features,)
        z1: charges of first input, shape (1,)
        x2: second input, shape (n_features,)
        z2: charges of second input, shape (1,)
    Returns:
        tanget: tangent wrt z2, shape ()
    """
    norm = jnp.linalg.norm(x1 - x2)
    not_zero = norm > 1e-20
    return jnp.where(
        not_zero,
        _cm_matrix_element(x1, z1, x2, z2) / z2,
        0.5 * 2.4 * jax.lax.abs(z1) ** (1.4),
    )


_cm_matrix_element.defjvps(
    lambda x1_dot, primal_out, x1, z1, x2, z2: (
        _cm_matrix_element_tangent_arg0(x1, z1, x2, z2) @ x1_dot
    ).reshape(primal_out.shape),
    lambda z1_dot, primal_out, x1, z1, x2, z2: (
        jnp.atleast_1d(_cm_matrix_element_tangent_arg1(x1, z1, x2, z1))
        @ jnp.atleast_1d(z1_dot)
    ).reshape(primal_out.shape),
    lambda x2_dot, primal_out, x1, z1, x2, z2: (
        _cm_matrix_element_tangent_arg2(x1, z1, x2, z2) @ x2_dot
    ).reshape(primal_out.shape),
    lambda z2_dot, primal_out, x1, z1, x2, z2: (
        jnp.atleast_1d(_cm_matrix_element_tangent_arg3(x1, z1, x2, z2))
        @ jnp.atleast_1d(z2_dot)
    ).reshape(primal_out.shape),
)


# ===================================================================
# Descriptor (function level)
# ===================================================================


@jit
def _coulomb_matrix(
    x1: ArrayLike, z1: ArrayLike, x2: ArrayLike, z2: ArrayLike
) -> Array:
    # This is a vectorized version of the basic
    # function along the first dimension (samples)
    # of the third and fourth inputs
    row_func = vmap(_cm_matrix_element, in_axes=(None, None, 0, 0))

    def update_func(carry, pair):
        x1s, z1s = pair
        row = row_func(x1s, z1s, x2, z2)
        return carry, row

    # Coulomb matrix built one row at a time
    _, cm = jax.lax.scan(update_func, 0, (x1, z1))

    return cm


def coulomb_matrix(
    x1: ArrayLike, z1: ArrayLike, x2: ArrayLike = None, z2: ArrayLike = None
) -> Array:
    r"""coulomb matrix descriptor

    This descriptor cumputes the coulomb matrix of the input vectors x1 and x2,
    i.e. it computes the inverse distances matrix between x1 and x2, and
    adds the contrubution of the charges z1 and z2 corresponding to the elements of the
    inputs.

        CM_{ij} = z1_i * z2_j / |x1_i - x2_j|    i != j
                = 0.5 * z1_i ** 2.4              i == j and 1 == 2

    x1 and x2 should be of shape (n_samples1, n_features) and (n_samples2, n_features).
    Normally, when computing the coulomb matrix for a molecule, one would have only one
    input vector x and only one set of charges z.
    In that case the diagonal corresponds to the one reported for i == j. When the inputs
    are different vectors, the distances between their points should never be zero.

    x1 and x2 should be of shape (n_samples1, n_features) and (n_samples2, n_features),
    z1 and z2 should be of shape (n_samples1,) and (n_samples2,).

    Args:
        x1: first input coordinates, shape (n_features,)
        z1: charges of first input, shape (1,)
        x2: second input coordinates, shape (n_features,)
        z2: charges of second input, shape (1,)

    Note: the custom_jvp allows us to compute the first derivatives forward and
    backward avoiding the NaN propagation. The second dirivative matrix (Hessian)
    can be computed only forward. The reverse Hessian still suffers from the NaN problem.
    """
    x2 = x1 if x2 is None else x2
    z2 = z1 if z2 is None else z2
    return _coulomb_matrix(x1, z1, x2, z2)


_coulomb_matrix.__doc__ = coulomb_matrix.__doc__


def _coulomb_matrix_offdiag(x1: ArrayLike, z1: ArrayLike) -> Array:
    # This function computes only the off-diagonal elements of
    # the coulomb matrix. The output is a 1-D array with the
    # upper triangular part.
    n1, _ = x1.shape
    cm = jnp.zeros((int(n1 * (n1 - 1) / 2)))

    def row_scan(i, cm):
        def inner_func(j, cm):
            k = (n1 * (n1 - 1) / 2) - (n1 - i) * ((n1 - i) - 1) / 2 + j - i - 1
            cm = cm.at[k.astype(int)].set(
                _cm_matrix_element(x1[i], z1[i], x1[j], z1[j])
            )
            return cm

        return jax.lax.fori_loop(i + 1, n1, inner_func, cm)

    cm = jax.lax.fori_loop(0, n1 - 1, row_scan, cm)

    return cm


def coulomb_matrix_offdiag(
    x1: ArrayLike, z1: ArrayLike, x2: ArrayLike = None, z2: ArrayLike = None
) -> Array:
    r"""coulomb matrix descriptor

    This descriptor cumputes the off-diagonal elements of the coulomb matrix
    (this makes sense only when x1 == x2 ).
    i.e. it computes the inverse distances matrix among x1 elements, and
    adds the contrubution of the charges z1 corresponding to the elements of the
    inputs.

        CM_{ij} = z1_i * z1_j / |x1_i - x1_j|    i != j

    x1 should be of shape (n_samples, n_features).
    z1 should be of shape (n_samples,).

    Args:
        x1: first input coordinates, shape (n_features,)
        z1: charges of first input, shape (1,)
        x2: must be equal to x1
        z2: must be equal to z1

    Note: the custom_jvp allows us to compute the first derivatives forward and
    backward avoiding the NaN propagation. The second dirivative matrix (Hessian)
    can be computed only forward. The reverse Hessian still suffers from the NaN problem.
    """
    return _coulomb_matrix_offdiag(x1, z1)


_coulomb_matrix_offdiag.__doc__ = coulomb_matrix_offdiag.__doc__


# ===================================================================
# Descriptor Class
# ===================================================================


class CoulombMatrix:
    def __init__(self, offdiag=False):
        self.offdiag = offdiag

    @property
    def offdiag(self):
        return self._offdiag

    @offdiag.setter
    def offdiag(self, value):
        if value:
            self._compute_func = coulomb_matrix_offdiag
        else:
            self._compute_func = coulomb_matrix
        self._offdiag = value

    def compute(self, x1, z1, x2=None, z2=None):
        return self._compute_func(x1, z1, x2, z2)
