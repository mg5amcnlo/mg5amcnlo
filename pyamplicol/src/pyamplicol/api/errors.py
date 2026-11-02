# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol.config.errors import ConfigurationError, PyAmpliColError


class ModelError(PyAmpliColError):
    """A model source or compiled model is invalid."""


class GenerationError(PyAmpliColError):
    """Planning or artifact generation failed."""


class ArtifactError(PyAmpliColError):
    """An artifact is malformed, incomplete, or unavailable."""


class CompatibilityError(ArtifactError):
    """An artifact is incompatible with the active target or ABI."""


class EvaluationError(PyAmpliColError):
    """Runtime loading, selection, or evaluation failed."""


class DependencyError(PyAmpliColError):
    """A required optional or native dependency is unavailable."""


__all__ = [
    "ArtifactError",
    "CompatibilityError",
    "ConfigurationError",
    "DependencyError",
    "EvaluationError",
    "GenerationError",
    "ModelError",
    "PyAmpliColError",
]
