# SPDX-License-Identifier: 0BSD
"""Deterministic validation-point records for generated process artifacts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..models.base import Model
from .dag_types import GenericDAG
from .phase_space import massive_rambo_final_state

if TYPE_CHECKING:
    from .contracts import CanonicalProcessIR

FourMomentum = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ValidationPointRecord:
    process_id: str
    process: str
    seed: int
    particles: tuple[tuple[int, FourMomentum], ...] = ()
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None

    @property
    def four_vectors(self) -> tuple[FourMomentum, ...]:
        if not self.available:
            return ()
        return tuple(momentum for _pdg, momentum in self.particles)

    def to_mapping(self) -> dict[str, object]:
        points: list[object] = []
        if self.available:
            points.append(
                [
                    {
                        "pdg": pdg,
                        "momentum": [_decimal_string(value) for value in momentum],
                    }
                    for pdg, momentum in self.particles
                ]
            )
        return {
            "schema_version": 1,
            "kind": "pyamplicol-rusticol-validation-momenta",
            "process_id": self.process_id,
            "process": self.process,
            "seed": self.seed,
            "available": self.available,
            "error": self.error,
            "points": points,
        }


def build_validation_point(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str,
    seed: int,
) -> ValidationPointRecord:
    return build_process_validation_point(
        dag.process,
        model,
        process_id=process_id,
        seed=seed,
    )


def build_process_validation_point(
    process: CanonicalProcessIR,
    model: Model,
    *,
    process_id: str,
    seed: int,
) -> ValidationPointRecord:
    """Build validation momenta without requiring a materialized DAG."""

    try:
        particles = _build_particles(process, model, seed=seed)
    except (ArithmeticError, KeyError, RuntimeError, ValueError) as exc:
        return ValidationPointRecord(
            process_id=process_id,
            process=process.process,
            seed=seed,
            error=str(exc),
        )
    return ValidationPointRecord(
        process_id=process_id,
        process=process.process,
        seed=seed,
        particles=particles,
    )


def rotate_validation_point(
    point: ValidationPointRecord,
    *,
    rotation_seed: int,
) -> ValidationPointRecord:
    """Apply a deterministic proper rotation to one physical event."""

    if not point.available:
        return point
    digest = hashlib.sha256(
        f"{rotation_seed}:proper-spatial-rotation-v1".encode("ascii")
    ).digest()
    denominator = float(1 << 64)
    azimuth = 2.0 * math.pi * (
        (int.from_bytes(digest[:8], "big") + 0.5) / denominator
    )
    cosine_polar = (
        2.0 * ((int.from_bytes(digest[8:16], "big") + 0.5) / denominator)
        - 1.0
    )
    sine_polar = math.sqrt(max(0.0, 1.0 - cosine_polar * cosine_polar))
    cos_azimuth = math.cos(azimuth)
    sin_azimuth = math.sin(azimuth)
    # Rz(azimuth) Ry(polar), a proper orthogonal map taking +z to the
    # deterministic direction above.
    rotation = (
        (
            cos_azimuth * cosine_polar,
            -sin_azimuth,
            cos_azimuth * sine_polar,
        ),
        (
            sin_azimuth * cosine_polar,
            cos_azimuth,
            sin_azimuth * sine_polar,
        ),
        (-sine_polar, 0.0, cosine_polar),
    )

    def rotated(momentum: FourMomentum) -> FourMomentum:
        energy, x_component, y_component, z_component = momentum
        spatial = (x_component, y_component, z_component)
        return (
            energy,
            *tuple(
                sum(row[column] * spatial[column] for column in range(3))
                for row in rotation
            ),
        )

    return replace(
        point,
        particles=tuple(
            (pdg, rotated(momentum)) for pdg, momentum in point.particles
        ),
    )


def validation_point_map(
    records: tuple[ValidationPointRecord, ...],
) -> Mapping[str, Mapping[str, object]]:
    return {record.process_id: record.to_mapping() for record in records}


def _build_particles(
    process: CanonicalProcessIR,
    model: Model,
    *,
    seed: int,
) -> tuple[tuple[int, FourMomentum], ...]:
    initial_pdgs = tuple(int(pdg) for pdg in process.initial_pdgs)
    final_pdgs = tuple(int(pdg) for pdg in process.final_pdgs)
    if len(initial_pdgs) != 2:
        raise ValueError("validation momenta require a two-particle initial state")
    if not final_pdgs:
        raise ValueError("validation momenta require at least one final-state particle")
    final_masses = tuple(_mass(model, pdg) for pdg in final_pdgs)
    threshold = sum(final_masses)
    if len(final_pdgs) == 1:
        if threshold <= 0.0:
            raise ValueError(
                "no finite center-of-mass point exists for one massless final state"
            )
        sqrt_s = threshold
        final_momenta: tuple[FourMomentum, ...] = ((float(sqrt_s), 0.0, 0.0, 0.0),)
    else:
        sqrt_s = max(1000.0, threshold + 100.0)
        final_momenta = massive_rambo_final_state(
            len(final_pdgs),
            sqrt_s=sqrt_s,
            masses=final_masses,
            seed=seed,
        )
    beam_energy = 0.5 * sqrt_s
    particles = (
        (initial_pdgs[0], (beam_energy, 0.0, 0.0, beam_energy)),
        (initial_pdgs[1], (beam_energy, 0.0, 0.0, -beam_energy)),
        *tuple(zip(final_pdgs, final_momenta, strict=True)),
    )
    if not all(math.isfinite(value) for _, momentum in particles for value in momentum):
        raise ArithmeticError("validation momenta contain non-finite components")
    return particles


def _mass(model: Model, pdg: int) -> float:
    value = float(model.mass(pdg))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"particle {pdg} has invalid mass {value!r}")
    return value


def _decimal_string(value: float) -> str:
    return format(float(value), ".17g")


__all__ = [
    "ValidationPointRecord",
    "build_process_validation_point",
    "build_validation_point",
    "rotate_validation_point",
    "validation_point_map",
]
