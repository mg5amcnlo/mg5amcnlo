# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from pyamplicol import CompiledModel, ModelSource
from pyamplicol._internal.sm_heft import (
    CANONICAL_SM_HEFT_SOURCE,
    SM_HEFT_ALIASES,
    extend_sm_model_payload,
    packaged_sm_source_path,
)
from pyamplicol.api.models import _compiled_model_payload


@pytest.fixture(scope="module")
def compiled_sm_heft() -> CompiledModel:
    return ModelSource.built_in_sm_heft().compile(use_cache=False)


def _named_records(
    payload: Mapping[str, object], field: str
) -> dict[str, Mapping[str, object]]:
    records = payload[field]
    assert isinstance(records, list)
    return {
        str(record["name"]): record for record in records if isinstance(record, Mapping)
    }


def test_packaged_sm_heft_has_canonical_aliases() -> None:
    assert CANONICAL_SM_HEFT_SOURCE in SM_HEFT_ALIASES


def test_sm_heft_extension_is_independent_and_has_constant_coupling() -> None:
    base = json.loads(packaged_sm_source_path().read_text(encoding="utf-8"))
    extended = extend_sm_model_payload(base)

    assert extended is not base
    assert extended["name"] == CANONICAL_SM_HEFT_SOURCE
    assert "GH" not in _named_records(base, "parameters")
    assert not any(
        name.startswith("V_HEFT") for name in _named_records(base, "vertex_rules")
    )

    parameters = _named_records(extended, "parameters")
    couplings = _named_records(extended, "couplings")
    gh_expression = str(parameters["GH"]["expression"])
    assert "P(" not in gh_expression
    assert "Momentum(" not in gh_expression
    assert {name for name in couplings if name.startswith("GC_HEFT_")} == {
        "GC_HEFT_HGG",
        "GC_HEFT_HGGG",
        "GC_HEFT_HGGGG",
    }
    assert all(
        "P(" not in str(couplings[name]["expression"])
        for name in ("GC_HEFT_HGG", "GC_HEFT_HGGG", "GC_HEFT_HGGGG")
    )


def test_compiled_sm_heft_contains_all_scalar_gluon_vertices(
    compiled_sm_heft: CompiledModel,
) -> None:
    private = _compiled_model_payload(compiled_sm_heft)
    heft_terms = tuple(
        term for term in private.ir.vertex_terms if term.vertex.startswith("V_HEFT_")
    )

    assert compiled_sm_heft.name == CANONICAL_SM_HEFT_SOURCE
    assert compiled_sm_heft.source.kind == "built-in-sm-heft"
    assert compiled_sm_heft.supported
    assert compiled_sm_heft.capabilities.form_factor_count == 0
    assert compiled_sm_heft.capabilities.maximum_valence == 5
    observed_terms = [
        (term.vertex, term.color_index, term.lorentz_index) for term in heft_terms
    ]
    assert observed_terms == [
        ("V_HEFT_HGG", 0, 0),
        ("V_HEFT_HGGG", 0, 0),
        ("V_HEFT_HGGGG", 0, 0),
        ("V_HEFT_HGGGG", 1, 1),
        ("V_HEFT_HGGGG", 2, 2),
    ]
    contact_ids = {term.id for term in heft_terms if len(term.particles) > 3}
    lowered_ids = {
        term_id
        for kernel in private.ir.oriented_kernels
        if "::contact-" in kernel.vertex and "final" in kernel.vertex
        for term_id in (kernel.term_ids or (kernel.term_id,))
    }
    assert contact_ids <= lowered_ids


def test_sm_heft_extension_rejects_record_collisions() -> None:
    base = json.loads(packaged_sm_source_path().read_text(encoding="utf-8"))
    base["parameters"].append({"name": "GH"})

    with pytest.raises(ValueError, match="already defines HEFT parameters"):
        extend_sm_model_payload(base)
