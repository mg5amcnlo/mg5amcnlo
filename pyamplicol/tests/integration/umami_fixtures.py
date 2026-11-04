# SPDX-License-Identifier: 0BSD
"""A small generic UFO model shared by UMAMI generation/parameter tests."""

from __future__ import annotations

import json
from pathlib import Path

from pyamplicol import ModelSource, ProcessSet


def equivalent_scalar_model(directory: Path) -> tuple[ModelSource, ProcessSet]:
    """Two different final flavours with exactly the same symbolic coupling.

    This is a restriction of the shipped scalar model, not an SM relabelling.
    All fields retain the normal UFO schema. The standard LHA declaration of
    ``aS`` exercises event alpha_s; ``G`` and the complex quartic coupling are
    derived from it. Both channels depend on the *same* mutable parameter, not
    two independent inputs that happen to have equal defaults.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "src/pyamplicol/assets/models/json/scalars/scalars.json"
    )
    data = json.loads(source.read_text(encoding="utf-8"))
    data["name"] = "umami_equivalent_scalars"
    data["particles"] = data["particles"][:3]
    for particle in data["particles"]:
        particle["mass"] = particle["width"] = "ZERO"
    data["parameters"] = [
        p for p in data["parameters"] if p["name"] in {"ZERO", "aS", "G"}
    ]
    data["propagators"] = data["propagators"][:3]
    for propagator in data["propagators"]:
        propagator["denominator"] = data["propagators"][0]["denominator"]
    data["lorentz_structures"] = [
        value
        for value in data["lorentz_structures"]
        if value["name"] == "SCALAR_4_LORENTZ_STRUCTURE"
    ]
    coupling = data["couplings"][0]
    coupling["expression"] = "1\U0001d456*UFO::{}::G^2"
    coupling["orders"] = [["QCD", 2]]
    g_default = next(p for p in data["parameters"] if p["name"] == "G")["value"][0]
    coupling["value"] = [0.0, g_default**2]
    data["vertex_rules"] = [
        {
            "name": f"V_QUARTIC_{index}",
            "particles": ["scalar_0", "scalar_0", f"scalar_{index}", f"scalar_{index}"],
            "color_structures": ["1"],
            "lorentz_structures": ["SCALAR_4_LORENTZ_STRUCTURE"],
            "couplings": [[coupling["name"]]],
        }
        for index in (1, 2)
    ]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "equivalent-scalars.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ModelSource.from_path(path), ProcessSet.from_expressions(
        [f"scalar_0 scalar_0 > scalar_{index} scalar_{index}" for index in (1, 2)],
        names=["flavour_one", "flavour_two"],
    )
