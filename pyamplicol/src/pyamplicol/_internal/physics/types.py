# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import dataclass

FourMomentum = tuple[float, float, float, float]
WaveFunction = tuple[complex, complex, complex, complex]
WeylWaveFunction = tuple[complex, complex]
TensorWaveFunction = tuple[complex, complex, complex, complex, complex, complex]


@dataclass(frozen=True)
class ExternalMomentum:
    pdg: int
    momentum: FourMomentum


@dataclass(frozen=True)
class HelicityContribution:
    helicities: tuple[int, ...]
    amplitude: complex
    squared: float


@dataclass(frozen=True)
class MatrixElementEvaluation:
    process: str
    particles: tuple[ExternalMomentum, ...]
    matrix_element: float
    raw_helicity_sum: float
    color_factor: int
    average_factor: int
    coupling_factor: float
    helicity_contributions: tuple[HelicityContribution, ...]
    identical_factor: int = 1


class NativeEvaluationError(ValueError):
    pass


__all__ = [
    "ExternalMomentum",
    "FourMomentum",
    "HelicityContribution",
    "MatrixElementEvaluation",
    "NativeEvaluationError",
    "TensorWaveFunction",
    "WaveFunction",
    "WeylWaveFunction",
]
