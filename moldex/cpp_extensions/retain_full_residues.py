# based on: https://github.com/dfm/extending-jax/tree/main
# extremely useful

__all__ = ["retain_full_residues"]

from functools import partial

import numpy as np
import jax
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import mlir, xla  # , batching  # ad
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# from jax._src.numpy.util import promote_dtypes_numeric
from .xla_helpers import (
    #    atleast_1d_arrays,
    #    size_arrays,
    default_layouts,
    #    array_shapes,
    #    reduce_array_shapes,
    mlir_dtype_and_shape,
)

# Register the CPU XLA custom calls
from . import cpu_ops


# ============================================================================
# Register the C++ functions in XLA
# ============================================================================

for _name, _value in cpu_ops.registrations().items():
    if _name.startswith("cpu_retain_full_residues"):
        # _value is a PyCapsule object containing the function pointer
        xla_client.register_cpu_custom_call_target(_name, _value)

# If the GPU versions exist, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

# ============================================================================
# Interface between the JAX primitive and the user
# ============================================================================


@jax.jit
def retain_full_residues(idx, residues_array):
    # ===
    # Uncommenting this results in 2x slower function
    #
    # idx, residues_array = atleast_1d_arrays(
    #     *promote_dtypes_numeric(idx, residues_array)
    # )
    # ===
    assert jnp.issubdtype(lax.dtype(idx), jnp.integer)
    assert jnp.issubdtype(lax.dtype(residues_array), jnp.integer)
    idx_max = idx.shape[0]
    n_resids = residues_array.shape[0] - 1
    return _retain_full_residues_prim.bind(idx, residues_array, idx_max, n_resids)


# ============================================================================
# Batching support (vmap)
# ============================================================================


# # easy because the C++ functions supports batching by design
# def _recursive_hermite_batching_rule(batched_args, batch_dims):
#     i, j, t, Qx, a, b, *_ = batched_args
#     return recursive_hermite_coefficient(i, j, t, Qx, a, b), 0


# slow version of batching, loops in python
# def _recursive_hermite_slow_batching_rule(batched_args, batch_dims):
#     i, j, t, Qx, a, b = atleast_1d_arrays(*batched_args)
#     out_shape = (
#         i.shape[0],
#         j.shape[0],
#         t.shape[0],
#         Qx.shape[0],
#         a.shape[0],
#         b.shape[0],
#     )
#     size = np.prod(out_shape)
#     out = jnp.zeros(size)
#     c = 0
#     for i_ in i:
#         for j_ in j:
#             for t_ in t:
#                 for Qx_ in Qx:
#                     for a_ in a:
#                         for b_ in b:
#                             out = out.at[c].set(
#                                 recursive_hermite_coefficient(i_, j_, t_, Qx_, a_, b_)[
#                                     0, 0, 0, 0, 0, 0
#                                 ]
#                             )
#                             c = c + 1
#     out = out.reshape(out_shape)
#     return out, 0


# ============================================================================
# JIT compilation support
# ============================================================================


def _retain_full_residues_abstract_eval(idx, residues_array, idx_max, n_resids):
    out_shape = idx.shape
    dtype = dtypes.canonicalize_dtype(idx.dtype)
    return ShapedArray(out_shape, dtype)


def _retain_full_residues_lowering(
    ctx,
    idx,
    residues_array,
    idx_max,
    n_resids,
    *,
    platform="cpu",
):
    idx_aval, residues_array_aval, idx_max_aval, n_resids_aval, *_ = ctx.avals_in
    out_shape = idx_aval.shape
    np_dtype = np.dtype(idx_aval.dtype)

    idx_dtype, idx_shape = mlir_dtype_and_shape(idx.type)
    residues_array_dtype, residues_array_shape = mlir_dtype_and_shape(
        residues_array.type
    )
    idx_max_dtype, idx_max_shape = mlir_dtype_and_shape(idx_max.type)
    n_resids_dtype, n_resids_shape = mlir_dtype_and_shape(n_resids.type)

    # We dispatch a different call depending on the dtype
    if np_dtype == np.int32:
        op_name = platform + "_retain_full_residues_i32"
    elif np_dtype == np.int64:
        op_name = platform + "_retain_full_residues_i64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    if platform == "cpu":
        out = custom_call(
            op_name,
            out_types=[mlir.ir.RankedTensorType.get(out_shape, idx_dtype.element_type)],
            operands=[
                idx,
                residues_array,
                idx_max,
                n_resids,
            ],
            operand_layouts=default_layouts(
                idx_shape,
                residues_array_shape,
                idx_max_shape,
                n_resids_shape,
            ),
            result_layouts=default_layouts(out_shape),
        )

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'retain_full_residues' module was not compiled with CUDA support"
            )
        raise ValueError("not implemented")
    #         # On the GPU, we do things a little differently and encapsulate the
    #         # dimension using the 'opaque' parameter
    #         opaque = gpu_ops.build_kepler_descriptor(size)
    #
    #         return custom_call(
    #             op_name,
    #             # Output types
    #             out_types=[dtype, dtype],
    #             # The inputs:
    #             operands=[mean_anom, ecc],
    #             # Layout specification:
    #             operand_layouts=[layout, layout],
    #             result_layouts=[layout, layout],
    #             # GPU specific additional data
    #             backend_config=opaque
    #         )

    else:
        ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")

    # This is very important... Even if the function returns a single output,
    # the lowering rule must return a tuple
    # https://github.com/google/jax/issues/15095
    return (out,)


# # **********************************
# # *  SUPPORT FOR FORWARD AUTODIFF  *
# # **********************************
#
# # Here we define the differentiation rules using a JVP derived using implicit
# # differentiation of Kepler's equation:
# #
# #  M = E - e * sin(E)
# #  -> dM = dE * (1 - e * cos(E)) - de * sin(E)
# #  -> dE/dM = 1 / (1 - e * cos(E))  and  de/dM = sin(E) / (1 - e * cos(E))
# #
# # In this case we don't need to define a transpose rule in order to support
# # reverse and higher order differentiation. This might not be true in other
# # applications, so check out the "How JAX primitives work" tutorial in the JAX
# # documentation for more info as necessary.
# def _kepler_jvp(args, tangents):
#     mean_anom, ecc = args
#     d_mean_anom, d_ecc = tangents
#
#     # We use "bind" here because we don't want to mod the mean anomaly again
#     sin_ecc_anom, cos_ecc_anom = _kepler_prim.bind(mean_anom, ecc)
#
#     def zero_tangent(tan, val):
#         return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan
#
#     # Propagate the derivatives
#     d_ecc_anom = (
#         zero_tangent(d_mean_anom, mean_anom)
#         + zero_tangent(d_ecc, ecc) * sin_ecc_anom
#     ) / (1 - ecc * cos_ecc_anom)
#
#     return (sin_ecc_anom, cos_ecc_anom), (
#         cos_ecc_anom * d_ecc_anom,
#         -sin_ecc_anom * d_ecc_anom,
#     )

# ============================================================================
# JAX primitive registration
# ============================================================================

_retain_full_residues_prim = core.Primitive("retain_full_residues")
_retain_full_residues_prim.multiple_results = False
_retain_full_residues_prim.def_impl(
    partial(xla.apply_primitive, _retain_full_residues_prim)
)
_retain_full_residues_prim.def_abstract_eval(_retain_full_residues_abstract_eval)

# Connect the XLA translation rules for JIT compilation
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _retain_full_residues_prim,
        partial(_retain_full_residues_lowering, platform=platform),
        platform=platform,
    )

# # Connect the JVP and batching rules
# ad.primitive_jvps[_kepler_prim] = _kepler_jvp
# batching.primitive_batchers[_recursive_hermite_prim] = _recursive_hermite_batching_rule
