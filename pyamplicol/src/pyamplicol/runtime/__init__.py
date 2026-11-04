# SPDX-License-Identifier: 0BSD
"""Lazy adapter for the native Rusticol runtime extension."""

from .backend import RusticolRuntimeBackend, load_runtime_backend

__all__ = ["RusticolRuntimeBackend", "load_runtime_backend"]
