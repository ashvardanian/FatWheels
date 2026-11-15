"""Every CPU capability this machine runs, and every GPU generation its driver loads, against serial.

Small whole numbers keep every product and sum exact in f32, bf16 and e4m3, so results must match exactly.
"""

import zlib

import fatwheels
import numpy as np
import pytest


def backends() -> list[tuple[str, str]]:
    """Every CPU capability and GPU generation this machine runs, best first."""
    pairs = [("cpu", name) for name in fatwheels.capabilities()]
    for device in ("cuda", "rocm"):
        if vendor := getattr(fatwheels, device):
            pairs += [(device, name) for name in vendor.capabilities()]
    return pairs


@pytest.fixture(params=backends(), ids="-".join)
def backend(request: pytest.FixtureRequest):
    """A `matmul` bound to one CPU capability or GPU generation."""
    device, capability = request.param
    module = fatwheels if device == "cpu" else getattr(fatwheels, device)
    return lambda a, b: module.matmul(a, b, capability=capability)


@pytest.fixture
def np_rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """A generator private to this test's shape, shared by every backend so they see the same inputs."""
    shape = {key: value for key, value in request.node.callspec.params.items() if key != "backend"}
    return np.random.default_rng(zlib.crc32(f"{request.node.originalname}{shape}".encode()))


@pytest.mark.parametrize("rows, columns, depth", [(1, 1, 1), (16, 16, 32), (37, 41, 67)])
def test_matmul(backend, np_rng: np.random.Generator, rows: int, columns: int, depth: int):
    """Matches the serial kernel at shapes that leave partial tiles on every edge."""
    a = np_rng.integers(-3, 4, (rows, depth)).astype(np.float32)
    b = np_rng.integers(-3, 4, (depth, columns)).astype(np.float32)
    np.testing.assert_array_equal(backend(a, b), fatwheels.matmul(a, b, "serial"))
