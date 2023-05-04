from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import custom_jvp, jit, vmap
from jax.typing import ArrayLike
from jax import Array

# ===================================================================
# Basic operation
# ===================================================================

# I am using a custom_jvp as to avoid NaN propagations when using
# jnp.where. An alternative would be to use the double where trick.
# Be careful with higher derivatives though, as derivatives beyond
# the first could present NaNs.


@custom_jvp
def _matrix_element(x1: ArrayLike, x2: ArrayLike) -> Array:
    """matrix element of inverse distance matrix

    This function computes a single matrix element of
    the inverse distance matrix. x1 and x2 are intended
    to be single samples.

    Args:
        x1: first input, shape (n_features,)
        x2: second input, shape (n_features,)
    Returns:
        mat_element: matrix element, shape ()
    """
    norm = jnp.linalg.norm(x1 - x2)
    not_zero = norm > 1e-20
    return jnp.where(not_zero, 1.0 / norm, 0.0)


def _matrix_element_tangent_arg0(x1: ArrayLike, x2: ArrayLike) -> Array:
    """tangent wrt the first argument for a single matrix element

    Args:
        x1: first input, shape (n_features,)
        x2: second input, shape (n_features,)
    Returns:
        tangent: tangent wrt first argument, shape (n_features,)
    """
    diff = x1 - x2
    norm3 = jnp.linalg.norm(diff) ** 3
    not_zero = norm3 > 1e-20
    return jnp.where(not_zero, -diff / norm3, 0.0)


def _matrix_element_tangent_arg1(x1, x2):
    """tangent wrt the second argument for a single matrix element

    Args:
        x1: first input, shape (n_features,)
        x2: second input, shape (n_features,)
    Returns:
        tangent: tangent wrt second argument, shape (n_features,)
    """
    return -_matrix_element_tangent_arg0(x1, x2)


_matrix_element.defjvps(
    lambda x1_dot, primal_out, x1, x2: (
        _matrix_element_tangent_arg0(x1, x2) @ x1_dot
    ).reshape(primal_out.shape),
    lambda x2_dot, primal_out, x1, x2: (
        _matrix_element_tangent_arg1(x1, x2) @ x2_dot
    ).reshape(primal_out.shape),
)


# ===================================================================
# Descriptor (function level)
# ===================================================================


@jit
def _inverse_distance_matrix(x1: ArrayLike, x2: ArrayLike) -> Array:
    # This is a vectorized version of the basic
    # function along the first dimension (samples)
    # of the second input
    row_func = vmap(_matrix_element, in_axes=(None, 0))

    def update_func(carry, x1s):
        row = row_func(x1s, x2)
        return carry, row

    # Inverse distance matrix built one row at a time
    _, mat = jax.lax.scan(update_func, 0, x1)

    return mat


def inverse_distance_matrix(x1: ArrayLike, x2: ArrayLike = None) -> Array:
    r"""inverse distance matrix descriptor

    This descriptor computes the matrix of inverse distances
    between inputs x1 and x2.

        M_{ij} = 1. / |x1_i - x2_j|    i != j
               = 0.                    i == j

    x1 and x2 should be of shape (n_samples1, n_features) and
    (n_samples2, n_features).

    Args:
        x1: first input, shape (n_samples1, n_features)
        x2: second input, shape (n_samples2, n_features)
    """
    x2 = x1 if x2 is None else x2
    return _inverse_distance_matrix(x1, x2)


_inverse_distance_matrix.__doc__ = inverse_distance_matrix.__doc__
