#!/usr/bin/env python3
"""FatWheels benchmarks: every kernel and the baseline, timed with pytest-benchmark.

File: bench.py
Author: Ash Vardanian
Date: August 9, 2024

This module defines benchmarks for reduction and matrix multiplication operations
using parameterized kernels. The available kernels (baseline, every CPU capability
this machine runs, and every GPU vendor whose module imported) are collected in lists,
and each benchmark is parameterized over the kernel, data type, and input size
(and tile size for matrix multiplication).

Usage:
    uv run pytest --benchmark-enable bench.py
"""

import numpy as np
import pytest

import fatwheels
from baseline import matmul as matmul_baseline, reduce as reduce_baseline

# Build lists of (name, kernel_function) for reduction and matrix multiplication.
REDUCTION_KERNELS = [("baseline", reduce_baseline)]
MATMUL_KERNELS = [("baseline", matmul_baseline)]
for name in fatwheels.capabilities():
    REDUCTION_KERNELS.append((f"cpu-{name}", lambda a, name=name: fatwheels.reduce(a, capability=name)))
    MATMUL_KERNELS.append((f"cpu-{name}", lambda a, b, name=name, **kw: fatwheels.matmul(a, b, capability=name, **kw)))
for vendor in ("cuda", "hip"):
    module = getattr(fatwheels, vendor)
    if module is not None:
        REDUCTION_KERNELS.append((vendor, module.reduce))
        MATMUL_KERNELS.append((vendor, module.matmul))


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("n", [2**i for i in range(10, 20)])
@pytest.mark.parametrize("kernel_name,kernel_func", REDUCTION_KERNELS)
def test_reduce(benchmark, dtype, n, kernel_name, kernel_func):
    """
    Benchmark a reduction kernel.

    Parameters:
        dtype (np.dtype): Data type for the input array.
        n (int): Size of the input array.
        kernel_name (str): Name of the kernel, like baseline, cpu-haswell, cuda or hip.
        kernel_func (function): The reduction function to benchmark.

    The test generates a random 1D array of size n, converts it to the given dtype,
    and then benchmarks the provided reduction kernel.
    """
    data = (np.random.rand(n) * 100).astype(dtype)

    # Wrap the kernel call in a lambda to delay execution until benchmarking.
    benchmark(lambda: kernel_func(data))


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
@pytest.mark.parametrize("n", [2**i for i in range(6, 11)])
@pytest.mark.parametrize("tile_size", [4, 8, 16, 32])
@pytest.mark.parametrize("kernel_name,kernel_func", MATMUL_KERNELS)
def test_matmul(benchmark, dtype, n, tile_size, kernel_name, kernel_func):
    """
    Benchmark a matrix multiplication kernel.

    Parameters:
        dtype (np.dtype): Data type for the input matrices.
        n (int): Dimension of the square matrices.
        tile_size (int): Tile size to use for the multiplication kernel.
        kernel_name (str): Name of the kernel, like baseline, cpu-haswell, cuda or hip.
        kernel_func (function): The matrix multiplication function to benchmark.

    The test generates a random n x n matrix (used as both input matrices) and benchmarks
    the provided kernel using the specified tile size.
    """
    a = (np.random.rand(n, n) * 100).astype(dtype)

    # Wrap the kernel call in a lambda.
    benchmark(lambda: kernel_func(a, a, tile_size=tile_size))
