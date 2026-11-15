"""Every CPU capability this machine runs, and every GPU generation its driver loads, against serial.

Small whole numbers keep every product and sum exact in f32, bf16 and e4m3, so results must match exactly.
"""

import numpy as np
import pytest

import fatwheels

backends = [("cpu", name) for name in fatwheels.capabilities()]
backends += [("cuda", name) for name in (fatwheels.cuda.capabilities() if fatwheels.cuda else [])]
backends += [("rocm", name) for name in (fatwheels.rocm.capabilities() if fatwheels.rocm else [])]


@pytest.mark.parametrize("device, capability", backends)
@pytest.mark.parametrize("rows, columns, depth", [(1, 1, 1), (16, 16, 32), (37, 41, 67)])
def test_matmul(device, capability, rows, columns, depth):
    """Matches the serial kernel at shapes that leave partial tiles on every edge."""
    generator = np.random.default_rng(42)
    a = generator.integers(-3, 4, (rows, depth)).astype(np.float32)
    b = generator.integers(-3, 4, (depth, columns)).astype(np.float32)
    backend = fatwheels if device == "cpu" else getattr(fatwheels, device)
    np.testing.assert_array_equal(backend.matmul(a, b, capability=capability), fatwheels.matmul(a, b, "serial"))
