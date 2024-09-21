#!/usr/bin/python
# -*- coding:utf-8 -*-
from pathlib import Path

import time
import pickle
import gzip

import torch
import torch.nn.functional as F

import contextlib
from functools import wraps, lru_cache
from filelock import FileLock

from einops import rearrange

__version__ = '0.4.0'''

# helper functions

def exists(val):
    return val is not None

def identity(t):
    return t

def default(val, d):
    return val if exists(val) else d

def to_order(degree):
    return 2 * degree + 1

def l2norm(t):
    return F.normalize(t, dim = -1)

def pad_for_centering_y_to_x(x, y):
    assert y <= x
    total_pad = x - y
    assert (total_pad % 2) == 0
    return total_pad // 2

def slice_for_centering_y_to_x(x, y):
    pad = pad_for_centering_y_to_x(x, y)
    if pad == 0:
        return slice(None)
    return slice(pad, -pad)

def safe_cat(arr, el, dim):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim = dim)

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def fast_split(arr, splits, dim=0):
    axis_len = arr.shape[dim]
    splits = min(axis_len, max(splits, 1))
    chunk_size = axis_len // splits
    remainder = axis_len - chunk_size * splits
    s = 0
    for i in range(splits):
        adjust, remainder = 1 if remainder > 0 else 0, remainder - 1
        yield torch.narrow(arr, dim, s, chunk_size + adjust)
        s += chunk_size + adjust

def masked_mean(tensor, mask, dim = -1):
    if not exists(mask):
        return tensor.mean(dim = dim)

    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim)
    mean = tensor.sum(dim = dim) / total_el.clamp(min = 1.)
    mean.masked_fill_(total_el == 0, 0.)
    return mean

def rand_uniform(size, min_val, max_val):
    return torch.empty(size).uniform_(min_val, max_val)

# default dtype context manager

@contextlib.contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype = torch.get_default_dtype())
        return fn(t)
    return inner

# benchmark tool

def benchmark(fn):
    def inner(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        diff = time.time() - start
        return diff, res
    return inner

# caching functions

def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner
    return cache_inner

# cache in directory

def cache_dir(dirname, maxsize=128):
    '''
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    '''
    def decorator(func):

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not exists(dirname):
                return func(*args, **kwargs)

            dirpath = Path(dirname)
            dirpath.mkdir(parents = True, exist_ok = True)

            indexfile = dirpath / 'index.pkl'
            lock = FileLock(str(dirpath / 'mutex'))

            with lock:
                index = {}
                if indexfile.exists():
                    with open(indexfile, "rb") as file:
                        index = pickle.load(file)

                key = (args, frozenset(kwargs), func.__defaults__)

                if key in index:
                    filename = index[key]
                else:
                    index[key] = filename = f"{len(index)}.pkl.gz"
                    with open(indexfile, "wb") as file:
                        pickle.dump(index, file)

            filepath = dirpath / filename

            if filepath.exists():
                with lock:
                    with gzip.open(filepath, "rb") as file:
                        result = pickle.load(file)
                return result

            print(f"compute {filename}... ", end="", flush = True)
            result = func(*args, **kwargs)
            print(f"save {filename}... ", end="", flush = True)

            with lock:
                with gzip.open(filepath, "wb") as file:
                    pickle.dump(result, file)

            print("done")

            return result
        return wrapper
    return decorator
