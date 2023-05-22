# based on: https://github.com/dfm/extending-jax/tree/main
# extremely useful

__all__ = ["recursive_hermite_coefficient"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import mlir, xla, batching  # ad
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from jax._src.numpy.util import promote_dtypes_inexact, promote_dtypes_numeric

# Register the CPU XLA custom calls
from . import cpu_ops


# ============================================================================
# Some helper functions
# ============================================================================


def atleast_1d_arrays(*arrs):
    return [jnp.atleast_1d(arr) for arr in arrs]


def size_arrays(*arrs):
    return [np.prod(arr.shape) for arr in arrs]


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def array_shapes(*arrs):
    return [arr.shape for arr in arrs]


def reduce_array_shapes(*arrs):
    return sum(array_shapes(*arrs), ())


def mlir_dtype_and_shape(arg):
    dtype = mlir.ir.RankedTensorType(arg)
    shape = dtype.shape
    return dtype, shape


# ============================================================================
# Register the C++ functions in XLA
# ============================================================================

for _name, _value in cpu_ops.registrations().items():
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


def recursive_hermite_coefficient(i, j, t, Qx, a, b):
    i, j, t = atleast_1d_arrays(*promote_dtypes_numeric(i, j, t))
    assert jnp.issubdtype(lax.dtype(i), jnp.integer)
    assert jnp.issubdtype(lax.dtype(j), jnp.integer)
    assert jnp.issubdtype(lax.dtype(t), jnp.integer)
    Qx, a, b = atleast_1d_arrays(*promote_dtypes_inexact(Qx, a, b))
    i_size, j_size, t_size, Qx_size, a_size, b_size = size_arrays(i, j, t, Qx, a, b)
    return _recursive_hermite_prim.bind(
        i, j, t, Qx, a, b, i_size, j_size, t_size, Qx_size, a_size, b_size
    )


# ============================================================================
# Batching support (vmap)
# ============================================================================


# easy because the C++ functions supports batching by design
def _recursive_hermite_batching_rule(batched_args, batch_dims):
    i, j, t, Qx, a, b, *_ = batched_args
    return recursive_hermite_coefficient(i, j, t, Qx, a, b), 0


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


def _recursive_hermite_abstract_eval(
    i, j, t, Qx, a, b, i_size, j_size, t_size, Qx_size, a_size, b_size
):
    out_shape = reduce_array_shapes(i, j, t, Qx, a, b)
    # we take the dtype of Qx as representative, but in the interface
    # function Qx, a, and b are recasted with the same inexact type
    dtype = dtypes.canonicalize_dtype(Qx.dtype)
    return ShapedArray(out_shape, dtype)


def _recursive_hermite_lowering(
    ctx,
    i,
    j,
    t,
    Qx,
    a,
    b,
    i_size,
    j_size,
    t_size,
    Qx_size,
    a_size,
    b_size,
    *,
    platform="cpu",
):
    i_aval, j_aval, t_aval, Qx_aval, a_aval, b_aval, _, _, _, _, _, _ = ctx.avals_in

    out_shape = reduce_array_shapes(i_aval, j_aval, t_aval, Qx_aval, a_aval, b_aval)

    # only checking the dtype of Qx as Qx, a, b
    # are promoted together in the function interface to the
    # same canonical inxact type, and i, j, t are checked
    # to be integer in the same interface
    np_dtype = np.dtype(Qx_aval.dtype)

    i_dtype, i_shape = mlir_dtype_and_shape(i.type)
    j_dtype, j_shape = mlir_dtype_and_shape(j.type)
    t_dtype, t_shape = mlir_dtype_and_shape(t.type)
    Qx_dtype, Qx_shape = mlir_dtype_and_shape(Qx.type)
    a_dtype, a_shape = mlir_dtype_and_shape(a.type)
    b_dtype, b_shape = mlir_dtype_and_shape(b.type)
    i_size_dtype, i_size_shape = mlir_dtype_and_shape(i_size.type)
    j_size_dtype, j_size_shape = mlir_dtype_and_shape(j_size.type)
    t_size_dtype, t_size_shape = mlir_dtype_and_shape(t_size.type)
    Qx_size_dtype, Qx_size_shape = mlir_dtype_and_shape(Qx_size.type)
    a_size_dtype, a_size_shape = mlir_dtype_and_shape(a_size.type)
    b_size_dtype, b_size_shape = mlir_dtype_and_shape(b_size.type)

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_recursive_hermite_f32"
    elif np_dtype == np.float64:
        op_name = platform + "_recursive_hermite_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    if platform == "cpu":
        out = custom_call(
            op_name,
            out_types=[mlir.ir.RankedTensorType.get(out_shape, Qx_dtype.element_type)],
            operands=[
                i,
                j,
                t,
                Qx,
                a,
                b,
                i_size,
                j_size,
                t_size,
                Qx_size,
                a_size,
                b_size,
            ],
            operand_layouts=default_layouts(
                i_shape,
                j_shape,
                t_shape,
                Qx_shape,
                a_shape,
                b_shape,
                i_size_shape,
                j_size_shape,
                t_size_shape,
                Qx_size_shape,
                a_size_shape,
                b_size_shape,
            ),
            result_layouts=default_layouts(out_shape),
        )

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'kepler_jax' module was not compiled with CUDA support"
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

_recursive_hermite_prim = core.Primitive("recursive_hermite")
_recursive_hermite_prim.multiple_results = False
_recursive_hermite_prim.def_impl(partial(xla.apply_primitive, _recursive_hermite_prim))
_recursive_hermite_prim.def_abstract_eval(_recursive_hermite_abstract_eval)

# Connect the XLA translation rules for JIT compilation
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _recursive_hermite_prim,
        partial(_recursive_hermite_lowering, platform=platform),
        platform=platform,
    )

# # Connect the JVP and batching rules
# ad.primitive_jvps[_kepler_prim] = _kepler_jvp
batching.primitive_batchers[_recursive_hermite_prim] = _recursive_hermite_batching_rule
