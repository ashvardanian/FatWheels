"""Pytest options shared by FatWheels' test modules."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-devices", action="store_true", help="also run kernels on this machine's GPUs")
