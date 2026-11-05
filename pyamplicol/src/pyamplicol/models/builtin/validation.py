# SPDX-License-Identifier: 0BSD
"""Legacy built-in-SM validation kinematics."""

from __future__ import annotations

from ..._internal.physics.types import ExternalMomentum, NativeEvaluationError
from ...generation.phase_space import massive_rambo_final_state
from .model import BuiltinSMModel
from .process_ir import build_process_ir


def legacy_rambo_z_gluon_point(
    process: str,
    model: BuiltinSMModel,
    *,
    gluon_count: int,
    sqrt_s: float,
    seed: int,
) -> tuple[ExternalMomentum, ...]:
    """Generate a deterministic RAMBO point for q q~ -> Z + n g."""

    if gluon_count < 1:
        raise NativeEvaluationError("RAMBO Z-gluon points need at least one gluon")
    z_mass = model.mass(23)
    if sqrt_s <= z_mass:
        raise NativeEvaluationError("sqrt(s) must be above the Z mass")
    pdgs = _physical_pdgs(process)
    expected = gluon_count + 3
    if len(pdgs) != expected:
        raise NativeEvaluationError(
            f"expected {expected} external particles for q q~ -> Z + {gluon_count} g"
        )
    final_pdgs = (*((21,) * gluon_count), 23)
    final_masses = tuple(0.0 if pdg == 21 else model.mass(pdg) for pdg in final_pdgs)
    if sum(final_masses) >= sqrt_s:
        raise NativeEvaluationError("final-state masses exceed sqrt(s)")
    final_momenta = massive_rambo_final_state(
        len(final_masses),
        sqrt_s=sqrt_s,
        masses=final_masses,
        seed=seed,
    )
    beam_energy = 0.5 * sqrt_s
    return (
        ExternalMomentum(pdgs[0], (beam_energy, 0.0, 0.0, beam_energy)),
        ExternalMomentum(pdgs[1], (beam_energy, 0.0, 0.0, -beam_energy)),
        *(
            ExternalMomentum(pdg, momentum)
            for pdg, momentum in zip(final_pdgs, final_momenta, strict=True)
        ),
    )


def generic_validation_point(
    process: str,
    *,
    model: BuiltinSMModel | None = None,
    sqrt_s: float | None = None,
    seed: int = 101,
) -> tuple[ExternalMomentum, ...]:
    """Return deterministic two-beam kinematics for a built-in-SM process."""

    model = model or BuiltinSMModel()
    ir = build_process_ir(process)
    initial_pdgs = tuple(int(pdg) for pdg in ir.initial_pdgs)
    final_pdgs = tuple(int(pdg) for pdg in ir.final_pdgs)
    if len(initial_pdgs) != 2:
        raise NativeEvaluationError(
            "generic validation momenta currently require a two-body initial state"
        )
    if not final_pdgs:
        raise NativeEvaluationError(
            "generic validation momenta require at least one final-state particle"
        )
    final_masses = tuple(float(model.mass(pdg)) for pdg in final_pdgs)
    threshold = sum(final_masses)
    if sqrt_s is None:
        sqrt_s = threshold if len(final_pdgs) == 1 else max(1000.0, threshold + 100.0)
    if sqrt_s < threshold:
        raise NativeEvaluationError("sqrt(s) is below the final-state mass threshold")
    if len(final_pdgs) == 1:
        if threshold <= 0.0:
            raise NativeEvaluationError(
                "no finite centre-of-mass validation point exists for one "
                "massless final state"
            )
        final_momenta = ((float(sqrt_s), 0.0, 0.0, 0.0),)
    else:
        final_momenta = massive_rambo_final_state(
            len(final_pdgs),
            sqrt_s=float(sqrt_s),
            masses=final_masses,
            seed=seed,
        )
    beam_energy = 0.5 * float(sqrt_s)
    return (
        ExternalMomentum(initial_pdgs[0], (beam_energy, 0.0, 0.0, beam_energy)),
        ExternalMomentum(initial_pdgs[1], (beam_energy, 0.0, 0.0, -beam_energy)),
        *(
            ExternalMomentum(pdg, momentum)
            for pdg, momentum in zip(final_pdgs, final_momenta, strict=True)
        ),
    )


def _physical_pdgs(process: str) -> tuple[int, ...]:
    ir = build_process_ir(process)
    return (*ir.initial_pdgs, *ir.final_pdgs)


__all__ = ["generic_validation_point", "legacy_rambo_z_gluon_point"]
