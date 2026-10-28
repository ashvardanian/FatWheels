"""FatWheels: one Python wheel for every CPU and both GPU vendors.

File: fatwheels/__init__.py
Author: Ash Vardanian
Date: October 28, 2026

`reduce` and `matmul` run on the CPU, and on `cuda` and `hip`, either of which is `None` where its module
is missing, its runtime is absent, or it sees no device.
"""

from fatwheels._cpu import capabilities, matmul, reduce

__all__ = ["capabilities", "cuda", "hip", "matmul", "reduce"]

try:
    from fatwheels import _cuda as cuda
except ImportError:  # built without NVCC
    cuda = None
if cuda is not None and not cuda.device_count():
    cuda = None

try:
    from fatwheels import _hip as hip
except ImportError:  # built without ROCm, or this machine lacks `libamdhip64`
    hip = None
if hip is not None and not hip.device_count():
    hip = None
