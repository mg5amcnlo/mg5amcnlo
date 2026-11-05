# SPDX-License-Identifier: 0BSD
"""Compact source projection for the private on-the-fly lane.

This boundary deliberately stops before color-sector planning, process-DAG
construction, recurrence schedule lowering, and numerical relation discovery.
It retains only the canonical external source domain and the model-owned roots
needed by Rust to discover one requested recurrence at runtime.  Contracted
NLC/full generation additionally retains a Python-only color plan beside the
seed; it never enters the process-seed-v1 wire contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from ..color import GenericColorPlan, build_color_plan
from ..models.base import Model
from ..models.recurrence_template import RecurrenceTemplateCatalog
from ..processes.ir import CanonicalProcessIR
from .recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceCouplingLimitV1,
    RecurrenceExternalLegV1,
    RecurrenceNormalizationV1,
    RecurrenceParameterProjectionV1,
)
from .recurrence_fermion_pairing import (
    FermionPairingRootsV1,
    build_recurrence_fermion_pairing_roots_v1,
)
from .recurrence_physics import build_recurrence_normalization
from .recurrence_projection import (
    RecurrenceProjectionError,
    _digest,
    _project_coupling_limits,
    _project_external_legs,
    _project_parameters,
    _project_template_references,
)

ON_THE_FLY_SOURCE_PROJECTION_SCHEMA: Final = 1


@dataclass(frozen=True, slots=True)
class OnTheFlyProcessSeedProjectionV1:
    """Minimal deterministic input to the native process-seed builder."""

    process_digest: str
    external_sources: tuple[RecurrenceExternalLegV1, ...]
    external_permutation: tuple[int, ...]
    parameter_projection: tuple[RecurrenceParameterProjectionV1, ...]
    coupling_order_policy: Literal["minimal", "explicit"]
    coupling_hierarchies: tuple[tuple[str, int], ...]
    coupling_limits: tuple[RecurrenceCouplingLimitV1, ...]
    fermion_pairing: FermionPairingRootsV1 | None
    normalization: RecurrenceNormalizationV1

    def to_json_dict(self) -> dict[str, object]:
        """Return the exact private JSON root accepted by the Rust builder."""

        return {
            "schema_version": ON_THE_FLY_SOURCE_PROJECTION_SCHEMA,
            "process_digest": self.process_digest,
            "external_permutation": list(self.external_permutation),
            "external_sources": [
                {
                    "source_slot": leg.source_slot,
                    "public_label": leg.public_label,
                    "is_initial": leg.is_initial,
                    "states": [
                        {
                            "state_index": state.state_index,
                            "public_helicity": state.public_helicity,
                            # The direct source executor consumes the same
                            # crossed helicity as the public recurrence source
                            # state.  The source template retains the declared
                            # (outgoing-basis) helicity separately by ID.
                            "source_helicity": state.public_helicity,
                            "source_template_id": state.source_template_id,
                            "current_state_template_id": (
                                state.current_state_template_id
                            ),
                            "momentum_sign": state.momentum_sign,
                            "crossing_phase": _exact_json(state.crossing_phase),
                            "spin_state": state.spin_state,
                            "chirality": state.chirality,
                        }
                        for state in leg.source_states
                    ],
                }
                for leg in self.external_sources
            ],
            "parameter_projection": [
                {
                    "parameter_template_id": row.parameter_template_id,
                    "prepared_parameter_id": row.prepared_parameter_id,
                    "component": row.component,
                }
                for row in self.parameter_projection
            ],
            "coupling_order_policy": self.coupling_order_policy,
            "coupling_hierarchies": [
                {"name": name, "hierarchy": hierarchy}
                for name, hierarchy in self.coupling_hierarchies
            ],
            "coupling_limits": [
                {"name": row.name, "maximum": row.maximum}
                for row in self.coupling_limits
            ],
            "fermion_pairing": (
                None
                if self.fermion_pairing is None
                else {
                    "endpoints": [
                        {
                            "source_slot": endpoint.source_slot,
                            "color_orientation": endpoint.color_orientation,
                            "contract_digest": endpoint.contract_digest,
                        }
                        for endpoint in self.fermion_pairing.endpoints
                    ],
                    "classes": [
                        {
                            "species": pairing_class.species,
                            "proof_digest": pairing_class.proof_digest,
                            "fundamental_source_slots": list(
                                pairing_class.fundamental_source_slots
                            ),
                            "antifundamental_source_slots": list(
                                pairing_class.antifundamental_source_slots
                            ),
                        }
                        for pairing_class in self.fermion_pairing.classes
                    ],
                }
            ),
            "normalization": {
                "factor": _exact_json(self.normalization.factor),
                "convention": self.normalization.convention,
                "semantic_digest": self.normalization.semantic_digest,
            },
        }

    def to_json_bytes(self) -> bytes:
        """Encode canonical ASCII without adding a second artifact identity."""

        return json.dumps(
            self.to_json_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")


@dataclass(frozen=True, slots=True)
class OnTheFlyGenerationProjectionV1:
    """Source seed plus existing runtime-normalization metadata for the writer."""

    seed: OnTheFlyProcessSeedProjectionV1
    runtime_normalization: Mapping[str, object]
    color_plan: GenericColorPlan | None = None


def project_on_the_fly_process_seed_v1(
    process: CanonicalProcessIR,
    template_catalog: RecurrenceTemplateCatalog,
    model: Model,
    *,
    coupling_order_policy: Literal["minimal", "explicit"],
    coupling_order_limits: Mapping[str, int],
    reference_color_order: Sequence[int] | None = None,
    color_basis: Literal["trace", "adjoint"] = "trace",
) -> OnTheFlyGenerationProjectionV1:
    """Project one complete source domain without materializing a process DAG.

    The projection always retains every source state. Runtime selectors choose
    helicities after loading.  LC also chooses a color flow lazily; NLC/full
    carries a separate contracted-color payload over the same structural
    selector domain.
    """

    if not isinstance(process, CanonicalProcessIR):
        raise TypeError("on-the-fly projection requires a CanonicalProcessIR")
    if process.color_accuracy not in {"lc", "nlc", "full"}:
        raise RecurrenceProjectionError(
            "on-the-fly source projection requires LC, NLC, or full color"
        )
    if not isinstance(template_catalog, RecurrenceTemplateCatalog):
        raise TypeError(
            "on-the-fly projection requires a validated recurrence template catalog"
        )
    if coupling_order_policy not in {"minimal", "explicit"}:
        raise RecurrenceProjectionError(
            "on-the-fly coupling-order policy must be 'minimal' or 'explicit'"
        )

    template_ids, _template_references = _project_template_references(template_catalog)
    external_sources, selected_sources = _project_external_legs(
        process,
        template_catalog,
        template_ids,
        None,
        1,
    )
    if selected_sources is not None:  # pragma: no cover - fixed by this call
        raise AssertionError("complete on-the-fly source projection was trimmed")
    pairing = build_recurrence_fermion_pairing_roots_v1(
        process,
        template_catalog.current_states,
        quantum_flows=template_catalog.quantum_flows,
    )
    normalization, runtime_normalization = build_recurrence_normalization(
        process,
        model,
    )
    color_plan = None
    if process.color_accuracy in {"nlc", "full"}:
        color_plan = build_color_plan(
            process,
            basis=color_basis,
            color_accuracy=process.color_accuracy,
            reference_color_order=reference_color_order,
            fold_trace_reflections=False,
        )
        if color_plan.truncated or not color_plan.sectors:
            detail = "; ".join(color_plan.diagnostics) or "no color sectors"
            raise RecurrenceProjectionError(
                "on-the-fly contracted-color projection has no complete color "
                f"plan: {detail}"
            )
    coupling_order_names = sorted(
        {
            str(name).upper()
            for records in (
                template_catalog.quantum_flows,
                template_catalog.transitions,
                template_catalog.closures,
            )
            for record in records
            for name, _power in record.coupling_orders
        }
    )
    projected_coupling_limits = _project_coupling_limits(coupling_order_limits)
    model_hierarchies = {
        str(name).upper(): max(1, int(value))
        for name, value in model.coupling_order_hierarchies().items()
    }
    seed = OnTheFlyProcessSeedProjectionV1(
        process_digest=_digest(process.to_json_dict()),
        external_sources=external_sources,
        external_permutation=tuple(range(len(process.legs))),
        parameter_projection=_project_parameters(template_catalog, template_ids),
        coupling_order_policy=coupling_order_policy,
        coupling_hierarchies=tuple(
            (name, model_hierarchies.get(name, 1)) for name in coupling_order_names
        ),
        coupling_limits=tuple(
            row for row in projected_coupling_limits if row.name in coupling_order_names
        ),
        fermion_pairing=(None if not pairing.endpoints else pairing),
        normalization=normalization,
    )
    return OnTheFlyGenerationProjectionV1(
        seed=seed,
        runtime_normalization=runtime_normalization,
        color_plan=color_plan,
    )


def _exact_json(value: ExactComplexRationalV1) -> dict[str, str]:
    return {
        "real_numerator": str(value.real_numerator),
        "real_denominator": str(value.real_denominator),
        "imag_numerator": str(value.imag_numerator),
        "imag_denominator": str(value.imag_denominator),
    }


__all__ = [
    "ON_THE_FLY_SOURCE_PROJECTION_SCHEMA",
    "OnTheFlyGenerationProjectionV1",
    "OnTheFlyProcessSeedProjectionV1",
    "project_on_the_fly_process_seed_v1",
]
