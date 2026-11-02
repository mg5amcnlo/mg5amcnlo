# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from pyamplicol.models.contracts import (
    CompiledLCColorTransitionTerm,
    CompiledModelIR,
    CompiledOrientedKernel,
)


def _term() -> CompiledLCColorTransitionTerm:
    return CompiledLCColorTransitionTerm(
        input_permutation=(1, 0),
        reverse_parent_mask=1,
        component_operation="concatenate-join",
        result_component_kind="open-string",
        result_component_role="active",
        input_shape_kinds=("adjoint-segment", "fundamental-open-string"),
        result_shape_kind="fundamental-open-string",
        exact_factor_expression="-1/2",
        proof_digest="a" * 64,
        provenance=(
            ("contact-split", "s-channel"),
            ("source", "exact-symbolica-projection-v1"),
        ),
    )


def _closure_term() -> CompiledLCColorTransitionTerm:
    return CompiledLCColorTransitionTerm(
        input_permutation=(0, 1),
        reverse_parent_mask=0,
        component_operation="close",
        result_component_kind="open-string",
        result_component_role="none",
        input_shape_kinds=(
            "fundamental-open-string",
            "antifundamental-open-string",
        ),
        result_shape_kind=None,
        exact_factor_expression="1",
        proof_digest="b" * 64,
        provenance=(("source", "exact-symbolica-closure-v1"),),
    )


def _model(
    term: CompiledLCColorTransitionTerm,
    closure_term: CompiledLCColorTransitionTerm | None = None,
) -> CompiledModelIR:
    kernel = CompiledOrientedKernel(
        kind=7,
        term_id=3,
        vertex="compiled-lc-color-test",
        particles=("left", "right", "result"),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="T(3,2,1)",
        color_expression="T(3,2,1)",
        term_ids=(3,),
        lc_color_transition_terms=(term,),
        lc_color_closure_terms=(() if closure_term is None else (closure_term,)),
    )
    return CompiledModelIR(
        name="compiled-lc-color-transition-term-test",
        orders=(),
        parameters=(),
        particles=(),
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(kernel,),
        direct_contractions=(),
        closure_contractions=(),
    )


def test_compiled_lc_color_transition_term_round_trips_through_model_ir() -> None:
    closure_term = _closure_term()
    model = _model(_term(), closure_term)

    payload = model.to_dict()
    encoded_term = payload["oriented_kernels"][0]["lc_color_transition_terms"][0]
    encoded_closure = payload["oriented_kernels"][0]["lc_color_closure_terms"][0]

    assert encoded_term == {
        "component_operation": "concatenate-join",
        "exact_factor_expression": "-1/2",
        "input_permutation": [1, 0],
        "input_shape_kinds": ["adjoint-segment", "fundamental-open-string"],
        "proof_digest": "a" * 64,
        "provenance": [
            ["contact-split", "s-channel"],
            ["source", "exact-symbolica-projection-v1"],
        ],
        "result_component_kind": "open-string",
        "result_component_role": "active",
        "result_shape_kind": "fundamental-open-string",
        "reverse_parent_mask": 1,
    }
    assert encoded_closure == closure_term.to_dict()
    assert CompiledModelIR.from_dict(payload) == model


def test_missing_kernel_transition_terms_preserve_compiled_ir_compatibility() -> None:
    payload = _model(_term()).to_dict()
    del payload["oriented_kernels"][0]["lc_color_transition_terms"]

    restored = CompiledModelIR.from_dict(payload)

    assert restored.oriented_kernels[0].lc_color_transition_terms == ()


def test_missing_kernel_closure_terms_preserve_nonrecurrence_compatibility() -> None:
    payload = _model(_term(), _closure_term()).to_dict()
    del payload["oriented_kernels"][0]["lc_color_closure_terms"]

    restored = CompiledModelIR.from_dict(payload)

    assert restored.oriented_kernels[0].lc_color_transition_terms == (_term(),)
    assert restored.oriented_kernels[0].lc_color_closure_terms == ()


def test_compiled_lc_color_transition_term_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        _term().proof_digest = "b" * 64  # type: ignore[misc]


def test_compiled_lc_color_transition_term_rejects_invalid_contracts() -> None:
    term = _term()

    with pytest.raises(ValueError, match="input permutation"):
        replace(term, input_permutation=(0, 0))
    with pytest.raises(ValueError, match="reverse-parent mask"):
        replace(term, reverse_parent_mask=4)
    with pytest.raises(ValueError, match="component operation"):
        replace(term, component_operation="append")
    with pytest.raises(ValueError, match="joins require a result kind"):
        replace(term, result_component_kind=None)
    with pytest.raises(ValueError, match="result component role"):
        replace(term, result_component_role="unknown")
    with pytest.raises(ValueError, match="supported input shape"):
        replace(term, input_shape_kinds=("unknown", "adjoint-segment"))
    with pytest.raises(ValueError, match="supported result shape"):
        replace(term, result_shape_kind="unknown")
    with pytest.raises(ValueError, match="nonempty canonical string"):
        replace(term, exact_factor_expression=" 1")
    with pytest.raises(ValueError, match="lowercase SHA256"):
        replace(term, proof_digest="A" * 64)
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(term, provenance=tuple(reversed(term.provenance)))
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(term, provenance=[])


def test_compiled_lc_color_transition_term_decoder_is_strict() -> None:
    payload = _term().to_dict()
    payload["unknown"] = "field"
    with pytest.raises(ValueError, match="unknown fields"):
        CompiledLCColorTransitionTerm.from_dict(payload)

    payload = _term().to_dict()
    payload["provenance"] = [["source"]]
    with pytest.raises(ValueError, match="must contain two entries"):
        CompiledLCColorTransitionTerm.from_dict(payload)

    model_payload = _model(_term()).to_dict()
    model_payload["oriented_kernels"][0]["lc_color_transition_terms"] = {}
    with pytest.raises(TypeError, match="transition terms must be an array"):
        CompiledModelIR.from_dict(model_payload)

    model_payload = _model(_term(), _closure_term()).to_dict()
    model_payload["oriented_kernels"][0]["lc_color_closure_terms"] = {}
    with pytest.raises(TypeError, match="closure terms must be an array"):
        CompiledModelIR.from_dict(model_payload)


def test_oriented_kernel_keeps_transition_and_closure_catalogs_disjoint() -> None:
    with pytest.raises(ValueError, match="transition terms cannot contain closure"):
        replace(
            _model(_term()).oriented_kernels[0],
            lc_color_transition_terms=(_closure_term(),),
        )
    with pytest.raises(ValueError, match="must contain only closure operations"):
        replace(
            _model(_term()).oriented_kernels[0],
            lc_color_closure_terms=(_term(),),
        )
