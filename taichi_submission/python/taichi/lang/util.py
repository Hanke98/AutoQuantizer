import functools
import os

import numpy as np
from taichi.core.util import ti_core as _ti_core
from taichi.lang import impl

import taichi as ti

_has_pytorch = False

_env_torch = os.environ.get('TI_ENABLE_TORCH', '1')
if not _env_torch or int(_env_torch):
    try:
        import torch
        _has_pytorch = True
    except:
        pass


def has_pytorch():
    """Whether has pytorch in the current Python environment.

    Returns:
        bool: True if has pytorch else False.

    """
    return _has_pytorch


from distutils.spawn import find_executable

# Taichi itself uses llvm-10.0.0 to compile.
# There will be some issues compiling CUDA with other clang++ version.
_clangpp_candidates = ['clang++-10']
_clangpp_presence = None
for c in _clangpp_candidates:
    if find_executable(c) is not None:
        _clangpp_presence = find_executable(c)


def has_clangpp():
    return _clangpp_presence is not None


def get_clangpp():
    return _clangpp_presence


def is_taichi_class(rhs):
    taichi_class = False
    try:
        if rhs.is_taichi_class:
            taichi_class = True
    except:
        pass
    return taichi_class


def to_numpy_type(dt):
    """Convert taichi data type to its counterpart in numpy.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in numpy.

    """
    if dt == ti.f32:
        return np.float32
    elif dt == ti.f64:
        return np.float64
    elif dt == ti.i32:
        return np.int32
    elif dt == ti.i64:
        return np.int64
    elif dt == ti.i8:
        return np.int8
    elif dt == ti.i16:
        return np.int16
    elif dt == ti.u8:
        return np.uint8
    elif dt == ti.u16:
        return np.uint16
    elif dt == ti.u32:
        return np.uint32
    elif dt == ti.u64:
        return np.uint64
    elif dt == ti.f16:
        return np.half
    else:
        assert False


def to_pytorch_type(dt):
    """Convert taichi data type to its counterpart in torch.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in torch.

    """
    if dt == ti.f32:
        return torch.float32
    elif dt == ti.f64:
        return torch.float64
    elif dt == ti.i32:
        return torch.int32
    elif dt == ti.i64:
        return torch.int64
    elif dt == ti.i8:
        return torch.int8
    elif dt == ti.i16:
        return torch.int16
    elif dt == ti.u8:
        return torch.uint8
    elif dt == ti.f16:
        return torch.float16
    elif dt in (ti.u16, ti.u32, ti.u64):
        raise RuntimeError(
            f'PyTorch doesn\'t support {dt.to_string()} data type.')
    else:
        assert False


def to_taichi_type(dt):
    """Convert numpy or torch data type to its counterpart in taichi.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in taichi.

    """
    if type(dt) == _ti_core.DataType:
        return dt

    if dt == np.float32:
        return ti.f32
    elif dt == np.float64:
        return ti.f64
    elif dt == np.int32:
        return ti.i32
    elif dt == np.int64:
        return ti.i64
    elif dt == np.int8:
        return ti.i8
    elif dt == np.int16:
        return ti.i16
    elif dt == np.uint8:
        return ti.u8
    elif dt == np.uint16:
        return ti.u16
    elif dt == np.uint32:
        return ti.u32
    elif dt == np.uint64:
        return ti.u64
    elif dt == np.half:
        return ti.f16

    if has_pytorch():
        if dt == torch.float32:
            return ti.f32
        elif dt == torch.float64:
            return ti.f64
        elif dt == torch.int32:
            return ti.i32
        elif dt == torch.int64:
            return ti.i64
        elif dt == torch.int8:
            return ti.i8
        elif dt == torch.int16:
            return ti.i16
        elif dt == torch.uint8:
            return ti.u8
        elif dt == torch.float16:
            return ti.f16
        elif dt in (ti.u16, ti.u32, ti.u64):
            raise RuntimeError(
                f'PyTorch doesn\'t support {dt.to_string()} data type.')

    raise AssertionError("Unknown type {}".format(dt))


def cook_dtype(dtype):
    _taichi_skip_traceback = 1
    if isinstance(dtype, _ti_core.DataType):
        return dtype
    elif isinstance(dtype, _ti_core.Type):
        return _ti_core.DataType(dtype)
    elif dtype is float:
        return impl.get_runtime().default_fp
    elif dtype is int:
        return impl.get_runtime().default_ip
    else:
        raise ValueError(f'Invalid data type {dtype}')


def in_taichi_scope():
    return impl.inside_kernel()


def in_python_scope():
    return not in_taichi_scope()


def taichi_scope(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        _taichi_skip_traceback = 1
        assert in_taichi_scope(), \
                f'{func.__name__} cannot be called in Python-scope'
        return func(*args, **kwargs)

    return wrapped


def python_scope(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        _taichi_skip_traceback = 1
        assert in_python_scope(), \
                f'{func.__name__} cannot be called in Taichi-scope'
        return func(*args, **kwargs)

    return wrapped
