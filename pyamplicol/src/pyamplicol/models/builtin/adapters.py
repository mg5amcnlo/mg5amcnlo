# SPDX-License-Identifier: 0BSD
"""Adapters from the hand-written built-in SM to canonical model records."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .model import BuiltinSMModel
from .process_catalog import PDGS


def build_model_payload() -> tuple[dict[str, object], dict[str, tuple[float, float]]]:
    """Return the canonical compiler payload for the built-in SM."""

    model = BuiltinSMModel()
    particles = []
    parameters: dict[str, tuple[float, float]] = {
        "alpha_s": (0.118, 0.0),
        "alpha_ew": (float(model.alpha_ew), 0.0),
    }
    for _key, particle in sorted(model.particles.items()):
        mass_name = (
            _parameter_name("mass", particle.pdg)
            if float(particle.mass) != 0.0
            else "ZERO"
        )
        width_name = (
            _parameter_name("width", particle.pdg)
            if float(particle.width) != 0.0
            else "ZERO"
        )
        if mass_name != "ZERO":
            parameters[mass_name] = (float(particle.mass), 0.0)
        if width_name != "ZERO":
            parameters[width_name] = (float(particle.width), 0.0)
        for pdg in dict.fromkeys((particle.pdg, particle.anti_pdg)):
            is_antiparticle = pdg != particle.pdg
            anti_pdg = particle.pdg if is_antiparticle else particle.anti_pdg
            color = particle.color_rep
            if is_antiparticle and abs(color) in {3, 6}:
                color = -color
            particles.append(
                {
                    "pdg_code": pdg,
                    "name": _particle_name(pdg),
                    "antiname": _particle_name(anti_pdg),
                    "spin": _ufo_spin(pdg, particle.spin),
                    "color": color,
                    "mass": mass_name,
                    "width": width_name,
                    "texname": _particle_name(pdg),
                    "antitexname": _particle_name(anti_pdg),
                    "charge": -particle.charge if is_antiparticle else particle.charge,
                    "quantum_numbers": [
                        list(item) for item in model.quantum_number_flow(pdg)
                    ],
                    "ghost_number": 0,
                    "lepton_number": 0,
                    "y_charge": 0,
                    "propagating": particle.spin >= 0,
                    "goldstoneboson": False,
                    "propagator": None,
                    "component_dimension": particle.dimension,
                    "auxiliary_kind": model.auxiliary_kind(pdg),
                }
            )
    return (
        {
            "name": "built-in-sm",
            "restriction": None,
            "orders": [
                {"name": "QCD", "expansion_order": 99, "hierarchy": 1},
                {"name": "QED", "expansion_order": 99, "hierarchy": 2},
            ],
            "parameters": [
                {
                    "name": name,
                    "nature": "external",
                    "parameter_type": "real",
                    "value": [value[0], value[1]],
                    "expression": None,
                    "lhablock": "PYAMPLICOL",
                    "lhacode": [index],
                }
                for index, (name, value) in enumerate(
                    sorted(parameters.items()),
                    start=1,
                )
            ],
            "particles": particles,
            "propagators": [],
            "lorentz_structures": [],
            "couplings": [],
            "vertex_rules": [
                {
                    "name": f"builtin_vertex_{index}",
                    "particles": [_particle_name(pdg) for pdg in vertex.particles],
                    "color_structures": ["1"],
                    "lorentz_structures": [f"builtin_kind_{vertex.kind}"],
                    "couplings": [[f"builtin_coupling_{index}"]],
                    "builtin_kind": vertex.kind,
                    "builtin_coupling": list(vertex.coupling),
                }
                for index, vertex in enumerate(model.vertices)
            ],
            "functions": [],
            "form_factors": [],
            "direct_contractions": [
                {
                    "left_particle": _particle_name(left_particle),
                    "left_chirality": left_chirality,
                    "right_particle": _particle_name(right_particle),
                    "right_chirality": right_chirality,
                    "contraction_ir": contraction_ir.to_json_dict(),
                }
                for (
                    left_particle,
                    left_chirality,
                    right_particle,
                    right_chirality,
                ), contraction_ir in sorted(
                    model._direct_contraction_ir_by_state.items()
                )
                if contraction_ir is not None
            ],
            "closure_contractions": [
                {
                    "particle": _particle_name(particle),
                    "chirality": chirality,
                    "contraction_ir": contraction_ir.to_json_dict(),
                }
                for (particle, chirality), contraction_ir in sorted(
                    model._closure_contraction_ir_by_state.items()
                )
                if contraction_ir is not None
            ],
            "builtin_model": True,
        },
        parameters,
    )


def source_digest() -> str:
    """Hash all implementation files behind the built-in model boundary."""

    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _parameter_name(field: str, pdg: int) -> str:
    pdg_token = f"m{abs(int(pdg))}" if int(pdg) < 0 else str(int(pdg))
    return f"{field}_{pdg_token}"


def _particle_name(pdg: int) -> str:
    candidates = [name for name, value in PDGS.items() if int(value) == pdg]
    if candidates:
        return min(candidates, key=lambda name: (len(name), name))
    return f"pdg_{pdg}"


def _ufo_spin(pdg: int, internal_spin: int) -> int:
    """Translate the legacy kernel spin tag to the UFO 2S+1 convention."""

    if internal_spin < 0:
        return -1
    absolute_pdg = abs(pdg)
    if 1 <= absolute_pdg <= 6 or 11 <= absolute_pdg <= 16:
        return 2
    if absolute_pdg in {21, 22, 23, 24}:
        return 3
    if absolute_pdg == 25:
        return 1
    raise ValueError(
        f"cannot map built-in particle {pdg} with spin tag {internal_spin} to UFO spin"
    )


__all__ = ["build_model_payload", "source_digest"]
