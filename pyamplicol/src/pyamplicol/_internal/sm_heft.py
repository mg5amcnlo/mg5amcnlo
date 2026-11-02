# SPDX-License-Identifier: 0BSD
"""Packaged scalar-HEFT extension of the Standard Model source.

The extension is deliberately stored as a small set of canonical model records
on top of the already packaged SM JSON model.  It adds the scalar Hgg, Hggg,
and Hgggg interactions from the FeynRules Higgs Effective Theory model without
duplicating the Standard Model payload or retaining a runtime dependency on an
external UFO checkout.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyamplicol.models.loading import CompiledModel

CANONICAL_SM_HEFT_SOURCE = "built-in-sm-heft"
SM_HEFT_ALIASES = frozenset(
    {
        CANONICAL_SM_HEFT_SOURCE,
        "builtin_sm_heft",
        "sm-heft",
        "sm_heft",
    }
)

_HEFT_ORDER = {
    "name": "HIG",
    "expansion_order": 1,
    "hierarchy": 1,
}

_HEFT_PARAMETER = {
    "name": "GH",
    "nature": "internal",
    "parameter_type": "real",
    "value": None,
    "expression": (
        "(1+1/168*UFO::{}::MH^4*UFO::{}::MT^(-4)"
        "+13/16800*UFO::{}::MH^6*UFO::{}::MT^(-6)"
        "+7/120*UFO::{}::MH^2*UFO::{}::MT^(-2))"
        "*-1/12*UFO::{}::G^2*UFO::{}::vev^(-1)*𝜋^(-2)"
    ),
    "lhablock": None,
    "lhacode": None,
}

_HEFT_COUPLINGS = (
    {
        "name": "GC_HEFT_HGG",
        "expression": "-1𝑖*UFO::{}::GH",  # noqa: RUF001
        "orders": [["HIG", 1]],
        "value": None,
    },
    {
        "name": "GC_HEFT_HGGG",
        "expression": "-1*UFO::{}::G*UFO::{}::GH",
        "orders": [["HIG", 1], ["QCD", 1]],
        "value": None,
    },
    {
        "name": "GC_HEFT_HGGGG",
        "expression": "1𝑖*UFO::{}::GH*UFO::{}::G^2",  # noqa: RUF001
        "orders": [["HIG", 1], ["QCD", 2]],
        "value": None,
    },
)

_HEFT_LORENTZ_STRUCTURES = (
    {
        "name": "HEFT_VVS",
        "spins": [3, 3, 1],
        "structure": (
            "-1*UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,2))"
            "*UFO::{}::P(UFO::{}::dummy(1),UFO::{}::idx(1,1))"
            "*UFO::{}::P(UFO::{}::dummy(1),UFO::{}::idx(1,2))"
            "+UFO::{}::P(UFO::{}::idx(1,1),UFO::{}::idx(1,2))"
            "*UFO::{}::P(UFO::{}::idx(1,2),UFO::{}::idx(1,1))"
        ),
    },
    {
        "name": "HEFT_VVVS",
        "spins": [3, 3, 3, 1],
        "structure": (
            "-1*UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,2))"
            "*UFO::{}::P(UFO::{}::idx(1,3),UFO::{}::idx(1,2))"
            "+-1*UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,3))"
            "*UFO::{}::P(UFO::{}::idx(1,2),UFO::{}::idx(1,1))"
            "+-1*UFO::{}::Metric(UFO::{}::idx(1,2),UFO::{}::idx(1,3))"
            "*UFO::{}::P(UFO::{}::idx(1,1),UFO::{}::idx(1,3))"
            "+UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,2))"
            "*UFO::{}::P(UFO::{}::idx(1,3),UFO::{}::idx(1,1))"
            "+UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,3))"
            "*UFO::{}::P(UFO::{}::idx(1,2),UFO::{}::idx(1,3))"
            "+UFO::{}::Metric(UFO::{}::idx(1,2),UFO::{}::idx(1,3))"
            "*UFO::{}::P(UFO::{}::idx(1,1),UFO::{}::idx(1,2))"
        ),
    },
    {
        "name": "HEFT_VVVVS1",
        "spins": [3, 3, 3, 3, 1],
        "structure": (
            "-1*UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,3))"
            "*UFO::{}::Metric(UFO::{}::idx(1,2),UFO::{}::idx(1,4))"
            "+UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,4))"
            "*UFO::{}::Metric(UFO::{}::idx(1,2),UFO::{}::idx(1,3))"
        ),
    },
    {
        "name": "HEFT_VVVVS2",
        "spins": [3, 3, 3, 3, 1],
        "structure": (
            "-1*UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,2))"
            "*UFO::{}::Metric(UFO::{}::idx(1,3),UFO::{}::idx(1,4))"
            "+UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,4))"
            "*UFO::{}::Metric(UFO::{}::idx(1,2),UFO::{}::idx(1,3))"
        ),
    },
    {
        "name": "HEFT_VVVVS3",
        "spins": [3, 3, 3, 3, 1],
        "structure": (
            "-1*UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,2))"
            "*UFO::{}::Metric(UFO::{}::idx(1,3),UFO::{}::idx(1,4))"
            "+UFO::{}::Metric(UFO::{}::idx(1,1),UFO::{}::idx(1,3))"
            "*UFO::{}::Metric(UFO::{}::idx(1,2),UFO::{}::idx(1,4))"
        ),
    },
)

_HEFT_VERTICES = (
    {
        "name": "V_HEFT_HGG",
        "particles": ["g", "g", "H"],
        "color_structures": ["UFO::{}::Identity(1,2)"],
        "lorentz_structures": ["HEFT_VVS"],
        "couplings": [["GC_HEFT_HGG"]],
    },
    {
        "name": "V_HEFT_HGGG",
        "particles": ["g", "g", "g", "H"],
        "color_structures": ["UFO::{}::f(1,2,3)"],
        "lorentz_structures": ["HEFT_VVVS"],
        "couplings": [["GC_HEFT_HGGG"]],
    },
    {
        "name": "V_HEFT_HGGGG",
        "particles": ["g", "g", "g", "g", "H"],
        "color_structures": [
            "UFO::{}::f(-1,1,2)*UFO::{}::f(3,4,-1)",
            "UFO::{}::f(-1,1,3)*UFO::{}::f(2,4,-1)",
            "UFO::{}::f(-1,1,4)*UFO::{}::f(2,3,-1)",
        ],
        "lorentz_structures": [
            "HEFT_VVVVS1",
            "HEFT_VVVVS2",
            "HEFT_VVVVS3",
        ],
        "couplings": [
            ["GC_HEFT_HGGGG", None, None],
            [None, "GC_HEFT_HGGGG", None],
            [None, None, "GC_HEFT_HGGGG"],
        ],
    },
)


def is_sm_heft_alias(value: object) -> bool:
    return str(value).lower() in SM_HEFT_ALIASES


def packaged_sm_source_path() -> Path:
    """Return the installed package's canonical SM JSON source."""

    path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "models"
        / "json"
        / "sm"
        / "sm.json"
    )
    if not path.is_file():
        raise RuntimeError(f"packaged Standard Model source is missing: {path}")
    return path


def extend_sm_model_payload(model: Mapping[str, object]) -> dict[str, object]:
    """Return an independent SM payload with the scalar HEFT records appended."""

    payload = deepcopy(dict(model))
    _require_base_particles(payload)
    additions: tuple[tuple[str, Sequence[Mapping[str, object]]], ...] = (
        ("orders", (_HEFT_ORDER,)),
        ("parameters", (_HEFT_PARAMETER,)),
        ("couplings", _HEFT_COUPLINGS),
        ("lorentz_structures", _HEFT_LORENTZ_STRUCTURES),
        ("vertex_rules", _HEFT_VERTICES),
    )
    for field, records in additions:
        raw_values = payload.get(field)
        if not isinstance(raw_values, list) or not all(
            isinstance(record, dict) for record in raw_values
        ):
            raise ValueError(f"packaged Standard Model {field} records are invalid")
        existing_names = {str(record.get("name")) for record in raw_values}
        added_names = {str(record["name"]) for record in records}
        conflicts = sorted(existing_names.intersection(added_names))
        if conflicts:
            raise ValueError(
                f"packaged Standard Model already defines HEFT {field}: {conflicts!r}"
            )
        raw_values.extend(deepcopy(list(records)))
    payload["name"] = CANONICAL_SM_HEFT_SOURCE
    return payload


def source_digest() -> str:
    """Identify only the packaged SM inputs and canonical HEFT physics records."""

    digest = hashlib.sha256()
    base = packaged_sm_source_path()
    restriction = base.with_name("restrict_default.json")
    for path in (base, restriction):
        if not path.is_file():
            raise RuntimeError(f"packaged Standard Model input is missing: {path}")
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(
        json.dumps(
            {
                "order": _HEFT_ORDER,
                "parameter": _HEFT_PARAMETER,
                "couplings": _HEFT_COUPLINGS,
                "lorentz_structures": _HEFT_LORENTZ_STRUCTURES,
                "vertices": _HEFT_VERTICES,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    return digest.hexdigest()


@contextmanager
def _materialized_source() -> Iterator[tuple[Path, Path]]:
    """Materialize the canonical overlay for the existing JSON compiler."""

    base = packaged_sm_source_path()
    packaged_restriction = base.with_name("restrict_default.json")
    with TemporaryDirectory(prefix="pyamplicol-sm-heft-") as temporary:
        source = Path(temporary) / "sm-heft.json"
        restriction = source.with_name("restrict_default.json")
        source.write_text(
            json.dumps(
                extend_sm_model_payload(json.loads(base.read_text(encoding="utf-8"))),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        restriction.write_bytes(packaged_restriction.read_bytes())
        yield source, restriction


def _with_packaged_identity(compiled: CompiledModel) -> CompiledModel:
    return replace(
        compiled,
        source={
            "kind": "built-in-sm-heft",
            "source_name": None,
            "digest": source_digest(),
            "options": {
                "restriction": {"kind": "name", "value": "default"},
                "simplify": True,
            },
        },
        _serialized_path=None,
    )


def compile_sm_heft_source(
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    require_supported: bool = True,
) -> CompiledModel:
    """Compile the packaged overlay through the canonical JSON model path."""

    from pyamplicol.models.loading import compile_model_source

    with _materialized_source() as (source, restriction):
        compiled = compile_model_source(
            source,
            restriction=str(restriction),
            simplify=True,
            cache_dir=cache_dir,
            use_cache=use_cache,
            require_supported=require_supported,
        )
    return _with_packaged_identity(compiled)


def load_cached_sm_heft_source(
    *,
    cache_dir: Path | None = None,
    require_supported: bool = True,
) -> CompiledModel | None:
    """Load the packaged overlay only when its canonical cache already exists."""

    from pyamplicol.models.loading import load_cached_model_source

    with _materialized_source() as (source, restriction):
        compiled = load_cached_model_source(
            source,
            restriction=str(restriction),
            simplify=True,
            cache_dir=cache_dir,
            require_supported=require_supported,
        )
    return None if compiled is None else _with_packaged_identity(compiled)


def _require_base_particles(payload: Mapping[str, object]) -> None:
    raw_particles = payload.get("particles")
    if not isinstance(raw_particles, list):
        raise ValueError("packaged Standard Model particle records are invalid")
    particles = {
        str(record.get("name")): record
        for record in raw_particles
        if isinstance(record, dict)
    }
    expected = {"g": (21, 3, 8), "H": (25, 1, 1)}
    for name, identity in expected.items():
        record = particles.get(name)
        observed = (
            None
            if record is None
            else (
                record.get("pdg_code"),
                record.get("spin"),
                record.get("color"),
            )
        )
        if observed != identity:
            raise ValueError(
                f"packaged Standard Model particle {name!r} changed: {observed!r}"
            )


__all__ = [
    "CANONICAL_SM_HEFT_SOURCE",
    "SM_HEFT_ALIASES",
    "compile_sm_heft_source",
    "extend_sm_model_payload",
    "is_sm_heft_alias",
    "load_cached_sm_heft_source",
    "packaged_sm_source_path",
    "source_digest",
]
