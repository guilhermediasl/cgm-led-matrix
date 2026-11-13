"""Local fork of python3-idotmatrix-library with a persistent BLE connection.

Derived from python3-idotmatrix-library 0.0.9 (GPLv3,
https://github.com/derkalle4/python3-idotmatrix-library), which is no longer
maintained.
"""
from .client import MatrixClient

__all__ = ["MatrixClient"]
