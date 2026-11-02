# SPDX-License-Identifier: 0BSD
"""Authenticated roots for model-generic recurrence closure obligations."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from pyamplicol.color.plan import GenericColorPlan, LCColorSector
from pyamplicol.generation.recurrence_columnar import (
    RecurrencePhysicalLCSectorV1,
)
from pyamplicol.generation.recurrence_fermion_pairing import (
    FermionPairingCatalogV1,
)
from pyamplicol.generation.recurrence_projection import (
    _project_closure_obligation_roots_digest,
)
from pyamplicol.models.recurrence_template import (
    ExactComplexRationalV1,
    LCColorTransitionWitnessV1,
    RecurrenceTemplateCatalog,
)
from pyamplicol.processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ProcessLegIR,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _factor(real: int, imag: int = 0) -> ExactComplexRationalV1:
    return ExactComplexRationalV1(real, 1, imag, 1)


def _process() -> CanonicalProcessIR:
    legs = tuple(
        ProcessLegIR(
            label,
            "initial" if label <= 2 else "final",
            "g",
            "g",
            21,
            21,
            "boson",
            "vector",
            "adjoint",
            "self-conjugate",
        )
        for label in range(1, 5)
    )
    return CanonicalProcessIR(
        process="g g > g g",
        key="g_g_to_g_g",
        color_accuracy="lc",
        legs=legs,
        color_endpoints=ColorEndpointSummary(0, 0, 0),
    )


def _color_plan(process: CanonicalProcessIR) -> GenericColorPlan:
    return GenericColorPlan(
        process=process,
        color_accuracy="lc",
        sectors=(
            LCColorSector(
                id=0,
                kind="single-trace",
                trace_labels=(1, 2, 3, 4),
                word_labels=(1, 2, 3, 4),
            ),
        ),
    )


def _template_catalog() -> RecurrenceTemplateCatalog:
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=_sha256("compiled-model"),
        prepared_kernel_pack_digest=_sha256("prepared-kernel-pack"),
    )


def _sector() -> RecurrencePhysicalLCSectorV1:
    return RecurrencePhysicalLCSectorV1(
        sector_id=0,
        public_id="flow:1,2,3,4",
        kind="single-trace",
        closure_source_slot=0,
        closure_proof_algorithm="canonical-lc-closure-anchor-v2",
        closure_proof_digest=_sha256("closure-proof"),
        trace_source_slots=(0, 1, 2, 3),
        word_source_slots=(0, 1, 2, 3),
        support_mask=1,
    )


def _pairing_catalog() -> FermionPairingCatalogV1:
    return FermionPairingCatalogV1(
        process_key="g_g_to_g_g",
        source_count=4,
        endpoints=(),
        pairing_classes=(),
        rules=(),
        topology_digest=_sha256("pairing-topology"),
        semantic_digest=_sha256("pairing-semantics"),
    )


def _obligation_roots_digest(
    *,
    sector: RecurrencePhysicalLCSectorV1 | None = None,
    pairing: FermionPairingCatalogV1 | None = None,
    reflection_contract: tuple[
        ExactComplexRationalV1 | None,
        str | None,
    ] = (None, None),
) -> str:
    process = _process()
    return _project_closure_obligation_roots_digest(
        process,
        _color_plan(process),
        _template_catalog(),
        (_sector() if sector is None else sector,),
        _pairing_catalog() if pairing is None else pairing,
        reflection_contract=reflection_contract,
    )


def test_reflected_pure_gluon_witness_preserves_exact_phase() -> None:
    witness = LCColorTransitionWitnessV1(
        input_shape_kinds=("adjoint-segment", "adjoint-segment"),
        input_permutation=(1, 0),
        reverse_parent_mask=0b11,
        component_operation="close",
        result_component_kind=None,
        result_component_role="none",
        result_shape_kind=None,
        exact_factor=_factor(-1),
        proof_digest=_sha256("pure-gluon-reflection"),
    )

    payload = witness.to_dict()
    assert payload["input_permutation"] == [1, 0]
    assert payload["reverse_parent_mask"] == 0b11
    assert payload["exact_factor"] == _factor(-1).to_dict()


def test_obligation_roots_digest_binds_exact_sector_closure_proof() -> None:
    sector = _sector()

    assert _obligation_roots_digest(sector=sector) != _obligation_roots_digest(
        sector=replace(
            sector,
            closure_proof_digest=_sha256("different-closure-proof"),
        )
    )


@pytest.mark.parametrize("digest_field", ["semantic_digest", "topology_digest"])
def test_obligation_roots_digest_binds_fermion_pairing_catalog(
    digest_field: str,
) -> None:
    pairing = _pairing_catalog()

    assert _obligation_roots_digest(pairing=pairing) != _obligation_roots_digest(
        pairing=replace(
            pairing,
            **{digest_field: _sha256(f"different-{digest_field}")},
        )
    )


def test_obligation_roots_digest_binds_exact_reflection_proof() -> None:
    phase = _factor(-1)

    assert _obligation_roots_digest(
        reflection_contract=(phase, _sha256("reflection-proof-a"))
    ) != _obligation_roots_digest(
        reflection_contract=(phase, _sha256("reflection-proof-b"))
    )
