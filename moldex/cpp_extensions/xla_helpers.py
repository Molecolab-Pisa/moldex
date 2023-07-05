from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np

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

