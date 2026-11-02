# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501 - uninterrupted SHA-256 goldens are easier to audit.
"""Semantic plan-v2 goldens for the Rust eager-lowering transition.

The snapshots deliberately avoid hashing JSON text or packed table bytes.  They
project the Python lowerer's execution semantics into stable records so that a
future plan-v3 Rust lowerer can be compared without reproducing Python object
layout, selector-domain numbering, or serialization details.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import TypeAlias, cast

import pytest

import pyamplicol.generation.service as service_module
from pyamplicol.api import ProcessRequest
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    RunConfig,
)
from pyamplicol.generation.eager_lowering import (
    EagerExecutionTables,
    PreparedCatalogEagerKernelResolver,
    lower_fused_eager_execution,
)
from pyamplicol.generation.eager_tables import (
    EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY,
    EagerSelectorDomainIdRow,
)
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.prepared_catalog import (
    PreparedKernelCatalog,
    build_prepared_kernel_catalog,
)

_ComplexFraction: TypeAlias = tuple[Fraction, Fraction]

_LEGACY_PLAN_ABI = "pyamplicol-eager-plan-v2"
_LEGACY_RUNTIME_CAPABILITY = "rusticol.eager-dag.complex-f64.v1"


@dataclass(frozen=True, slots=True)
class _GoldenCase:
    name: str
    schema: Mapping[str, object]
    tables: EagerExecutionTables
    catalog: PreparedKernelCatalog


def _canonical(value: object) -> object:
    """Make semantic values JSON-canonical while preserving exact f64 bits."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"f64_hex": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    raise TypeError(f"unsupported semantic-golden value {type(value).__name__}")


def _digest(value: object) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{field} must be a mapping")
    return cast(Mapping[str, object], value)


def _records(value: object, field: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AssertionError(f"{field} must be a sequence")
    return tuple(_mapping(item, f"{field}[]") for item in value)


def _build_case(
    *,
    name: str,
    expression: str,
    accuracy: str,
    lc_flow_layout: str,
    catalog: PreparedKernelCatalog,
) -> _GoldenCase:
    model = BuiltinSMModel()
    backend = service_module.GenerationBackend(
        RunConfig(
            action="generate",
            # Fix the historical v2 optimization policy independently of
            # changing production defaults for numerical current discovery.
            generation=GenerationConfig(
                relation_discovery=GenerationRelationDiscoveryConfig(
                    mode="certified-reuse"
                )
            ),
            color=ColorConfig(
                accuracy=accuracy,
                lc_flow_layout=lc_flow_layout,
            ),
            evaluator=EvaluatorConfig(execution_mode="eager"),
        ),
        None,
    )
    process_ir = build_process_ir(expression, color_accuracy=accuracy)
    dag, coverage = backend._compile_concrete_process(process_ir, model)
    prepared = backend._prepare_warmup_process(
        service_module._DagProcess(
            expanded=service_module._ExpandedProcess(
                request=ProcessRequest.parse(expression, name=name),
                process_ir=process_ir,
            ),
            dag=dag,
            coverage=coverage,
        ),
        model,
        index=0,
        phase=PhaseHandle("semantic-golden", None, 1),
    )
    schema, tables = lower_fused_eager_execution(
        dag=prepared.dag,
        model=model,
        resolver=PreparedCatalogEagerKernelResolver(
            prepared.dag,
            catalog.resolver_manifest(),
        ),
        process_id=name,
    )
    return _GoldenCase(name=name, schema=schema, tables=tables, catalog=catalog)


@pytest.fixture(scope="module")
def golden_cases() -> tuple[_GoldenCase, ...]:
    catalog = build_prepared_kernel_catalog(BuiltinSMModel())
    return (
        _build_case(
            name="lc-topology-replay",
            expression="d d~ > z g g",
            accuracy="lc",
            lc_flow_layout="topology-replay",
            catalog=catalog,
        ),
        _build_case(
            name="lc-all-flow-union",
            expression="d d~ > z g g",
            accuracy="lc",
            lc_flow_layout="all-flow-union",
            catalog=catalog,
        ),
        _build_case(
            name="nlc-contracted",
            expression="g g > g g",
            accuracy="nlc",
            lc_flow_layout="topology-replay",
            catalog=catalog,
        ),
        _build_case(
            name="full-contracted",
            expression="g g > g g",
            accuracy="full",
            lc_flow_layout="topology-replay",
            catalog=catalog,
        ),
    )


def _layout_semantics(schema: Mapping[str, object]) -> object:
    current_storage = _mapping(schema["current_storage"], "current_storage")
    value_storage = _mapping(schema["value_storage"], "value_storage")
    stages = _records(schema["stages"], "stages")
    return {
        "parameter_layout": schema["parameter_layout"],
        "model_parameters": schema["model_parameters"],
        "current_component_count": current_storage["component_count"],
        "current_slots": sorted(
            _records(current_storage["current_slots"], "current_slots"),
            key=lambda row: int(row["current_id"]),
        ),
        "value_component_count": value_storage["component_count"],
        "value_slots": sorted(
            _records(value_storage["value_slots"], "value_slots"),
            key=lambda row: int(row["value_slot_id"]),
        ),
        "source_fill": schema["source_fill"],
        "momentum_slots": sorted(
            _records(schema["momentum_slots"], "momentum_slots"),
            key=lambda row: int(row["momentum_slot_id"]),
        ),
        "stages": [
            {
                key: stage[key]
                for key in (
                    "stage_index",
                    "stage_kind",
                    "subset_size",
                    "input_current_ids",
                    "output_current_ids",
                    "input_value_slot_ids",
                    "output_value_slot_ids",
                    "input_momentum_slot_ids",
                )
            }
            for stage in stages
        ],
    }


def _table_semantics(tables: EagerExecutionTables) -> object:
    couplings = tuple(asdict(row) for row in tables.couplings)

    def coupling(row_id: int) -> object:
        if row_id >= len(couplings):
            return {"missing": True}
        return couplings[row_id]

    return {
        "process_key": tables.process_key,
        "couplings": couplings,
        "stages": [
            {
                "stage_index": stage.stage_index,
                "subset_size": stage.subset_size,
                "invocations": [
                    {
                        **asdict(row),
                        "coupling": coupling(row.coupling_slot_id),
                    }
                    for row in stage.invocations
                ],
                "attachments": [asdict(row) for row in stage.attachments],
                "finalizations": [asdict(row) for row in stage.finalizations],
            }
            for stage in tables.stages
        ],
        "closures": [
            {
                **asdict(row),
                "coupling": coupling(row.coupling_slot_id),
            }
            for row in tables.closures
        ],
    }


def _selector_semantics(
    schema: Mapping[str, object], tables: EagerExecutionTables
) -> object:
    selector = tables.selector_closures
    if selector is None:
        return {"enabled": False}
    memberships = tuple(
        tuple(
            row.coherent_group_id
            for row in selector.domain_group_ids[
                domain.member_start : domain.member_start + domain.member_count
            ]
        )
        for domain in selector.domains
    )

    def resolve(rows: Sequence[EagerSelectorDomainIdRow]) -> list[tuple[int, ...]]:
        return [memberships[row.domain_id] for row in rows]

    physics = _mapping(schema["physics"], "physics")
    amplitude = _mapping(schema["amplitude_stage"], "amplitude_stage")
    extensions = _mapping(physics["extensions"], "physics.extensions")
    return {
        "enabled": True,
        "domain_memberships": sorted(set(memberships)),
        "stages": [
            {
                "stage_index": stage.stage_index,
                "invocations": resolve(stage.invocation_domains),
                "attachments": resolve(stage.attachment_domains),
                "unpropagated_finalizations": resolve(
                    stage.unpropagated_finalization_domains
                ),
                "propagated_finalizations": resolve(
                    stage.propagated_finalization_domains
                ),
            }
            for stage in selector.stages
        ],
        "closures": resolve(selector.closure_domains),
        "public_selectors": physics["selectors"],
        "runtime_selectors": extensions.get("runtime_selectors"),
        "lc_topology_replay": schema.get("lc_topology_replay"),
        "color_topology_replay": amplitude.get("color_topology_replay"),
        "helicity_recurrence": schema.get("helicity_recurrence"),
    }


def _reduction_semantics(schema: Mapping[str, object]) -> object:
    physics = _mapping(schema["physics"], "physics")
    amplitude = _mapping(schema["amplitude_stage"], "amplitude_stage")
    return {
        "roots": amplitude["roots"],
        "coherent_groups": amplitude["coherent_groups"],
        "final_reduction": amplitude["final_reduction"],
        "color_contraction": amplitude["color_contraction"],
        "physics_reduction": physics["reduction"],
        "color_components": physics["color_components"],
        "helicities": physics["helicities"],
        "normalization": schema["normalization"],
    }


def _logical_color_contraction_entries(
    contraction: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    entries = _records(contraction["entries"], "color_contraction.entries")
    repeated_raw = contraction.get("repeated_block")
    if repeated_raw is None:
        return entries
    assert not entries
    repeated = _mapping(repeated_raw, "color_contraction.repeated_block")
    component_count = int(repeated["component_count"])
    component_group_ids = tuple(
        int(value) for value in cast(Sequence[object], repeated["component_group_ids"])
    )
    templates = _records(
        repeated["entries"], "color_contraction.repeated_block.entries"
    )
    return tuple(
        {
            "left_group_id": component_group_ids[
                int(entry["left_group_index"]) * component_count + component_index
            ],
            "right_group_id": component_group_ids[
                int(entry["right_group_index"]) * component_count + component_index
            ],
            "weight": entry["weight"],
            "symmetry_factor": entry["symmetry_factor"],
        }
        for component_index in range(component_count)
        for entry in templates
    )


def _exact_semantics(case: _GoldenCase) -> object:
    referenced = case.tables.referenced_kernel_ids
    kernels = [case.catalog.by_id[kernel_id] for kernel_id in sorted(referenced)]
    amplitude = _mapping(case.schema["amplitude_stage"], "amplitude_stage")
    roots = _records(amplitude["roots"], "amplitude_stage.roots")
    return {
        "kernels": [
            {
                "kernel_id": kernel.kernel_id,
                "contract_kind": kernel.contract_kind,
                "canonical_signature": kernel.canonical_signature,
                "exact_expressions": kernel.exact_expressions,
                "inputs": [item.to_dict() for item in kernel.inputs],
                "output_layout": kernel.output_layout,
                "proof_classes": kernel.proof_classes,
            }
            for kernel in kernels
        ],
        "root_exact_factors": [
            {
                "output_index": root["output_index"],
                "color_weight": root["color_weight"],
                "coupling": root["coupling"],
                "helicity_weight": root["helicity_weight"],
                "all_sector_weight": root["all_sector_weight"],
                "contraction_coefficients": _mapping(
                    root["contraction_ir"], "root.contraction_ir"
                )["coefficients"],
            }
            for root in roots
        ],
    }


def _fraction(value: object) -> Fraction:
    return Fraction.from_float(float(value))


def _complex_mul(left: _ComplexFraction, right: _ComplexFraction) -> _ComplexFraction:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _fraction_record(value: Fraction) -> tuple[str, str]:
    return str(value.numerator), str(value.denominator)


def _complex_record(value: _ComplexFraction) -> object:
    return _fraction_record(value[0]), _fraction_record(value[1])


def _resolved_probe(
    schema: Mapping[str, object], tables: EagerExecutionTables
) -> object:
    """Evaluate reduction semantics with deterministic exact rational amplitudes.

    The probe is deliberately backend-independent.  Its synthetic closure
    values exercise closure factors, coherent grouping, diagonal LC weights,
    and sparse NLC/full contractions exactly, without compiling an evaluator.
    """

    amplitude = _mapping(schema["amplitude_stage"], "amplitude_stage")
    roots = _records(amplitude["roots"], "amplitude_stage.roots")
    root_by_output = {int(root["output_index"]): root for root in roots}
    group_amplitudes: dict[int, _ComplexFraction] = defaultdict(
        lambda: (Fraction(0), Fraction(0))
    )
    for closure in tables.closures:
        output = closure.amplitude_index
        root = root_by_output[output]
        seed = (
            Fraction((output * 17) % 23 + 1, 23),
            Fraction((output * 29) % 31 - 15, 31),
        )
        factor = (_fraction(closure.factor_real), _fraction(closure.factor_imag))
        value = _complex_mul(seed, factor)
        group_id = int(root["coherent_group_id"])
        previous = group_amplitudes[group_id]
        group_amplitudes[group_id] = (
            previous[0] + value[0],
            previous[1] + value[1],
        )

    physics = _mapping(schema["physics"], "physics")
    physical_groups: tuple[Mapping[str, object], ...] | None = None
    color_replay_raw = amplitude.get("color_topology_replay")
    if color_replay_raw is not None:
        color_replay = _mapping(
            color_replay_raw,
            "amplitude_stage.color_topology_replay",
        )
        physical_groups = _records(
            color_replay["physical_groups"],
            "amplitude_stage.color_topology_replay.physical_groups",
        )
        expanded_amplitudes: dict[int, _ComplexFraction] = {}
        for mapping in _records(
            color_replay["mappings"],
            "amplitude_stage.color_topology_replay.mappings",
        ):
            for route in _records(
                mapping["group_routes"],
                "amplitude_stage.color_topology_replay.group_routes",
            ):
                source_id = int(route["source_group_id"])
                target_id = int(route["target_group_id"])
                assert target_id not in expanded_amplitudes
                factor = cast(Sequence[object], route["factor"])
                expanded_amplitudes[target_id] = _complex_mul(
                    group_amplitudes[source_id],
                    (_fraction(factor[0]), _fraction(factor[1])),
                )
        assert set(expanded_amplitudes) == {
            int(group["group_id"]) for group in physical_groups
        }
        group_amplitudes = defaultdict(
            lambda: (Fraction(0), Fraction(0)),
            expanded_amplitudes,
        )

    reduction = _mapping(physics["reduction"], "physics.reduction")
    reduction_groups = _records(reduction["groups"], "physics.reduction.groups")
    if physical_groups is None:
        component_by_group = {}
        for record in reduction_groups:
            group_id = int(str(record["id"]).rsplit(":", maxsplit=1)[-1])
            component_by_group[group_id] = (
                f"{record['representative_color_id']}|"
                f"{record['representative_helicity_id']}"
            )
    else:
        color_components = _records(
            physics["color_components"],
            "physics.color_components",
        )
        assert len(color_components) == 1
        color_id = str(color_components[0]["id"])
        helicity_ids = {
            tuple(
                int(value) for value in cast(Sequence[object], record["values"])
            ): str(record["representative_id"])
            for record in _records(physics["helicities"], "physics.helicities")
        }
        component_by_group = {
            int(group["group_id"]): (
                f"{color_id}|"
                f"{helicity_ids[tuple(int(value) for value in cast(Sequence[object], group['helicities']))]}"
            )
            for group in physical_groups
        }

    resolved: dict[str, Fraction] = defaultdict(Fraction)
    contraction = amplitude["color_contraction"]
    if contraction is None:
        groups = _records(amplitude["coherent_groups"], "coherent_groups")
        for group in groups:
            group_id = int(group["group_id"])
            value = group_amplitudes[group_id]
            resolved[component_by_group[group_id]] += _fraction(
                group["all_sector_weight"]
            ) * (value[0] * value[0] + value[1] * value[1])
    else:
        entries = _logical_color_contraction_entries(
            _mapping(contraction, "color_contraction")
        )
        for entry in entries:
            left_id = int(entry["left_group_id"])
            right_id = int(entry["right_group_id"])
            left = group_amplitudes[left_id]
            right = group_amplitudes[right_id]
            product = (
                left[0] * right[0] + left[1] * right[1],
                left[1] * right[0] - left[0] * right[1],
            )
            weight = cast(Sequence[object], entry["weight"])
            contribution = _fraction(entry["symmetry_factor"]) * (
                _fraction(weight[0]) * product[0] - _fraction(weight[1]) * product[1]
            )
            # Sparse colour contractions only connect coherent groups for the
            # same physical helicity. Aggregate by the public resolved axis,
            # not by upper-triangular entry orientation.
            component = component_by_group[left_id]
            assert component == component_by_group[right_id]
            resolved[component] += contribution

    return {
        "coherent_amplitudes": [
            (group_id, _complex_record(value))
            for group_id, value in sorted(group_amplitudes.items())
        ],
        "resolved_group_totals": [
            (component, _fraction_record(value))
            for component, value in sorted(resolved.items())
        ],
        "total": _fraction_record(sum(resolved.values(), Fraction())),
    }


def _diagnostic_counts(case: _GoldenCase) -> dict[str, int]:
    schema = case.schema
    current_storage = _mapping(schema["current_storage"], "current_storage")
    value_storage = _mapping(schema["value_storage"], "value_storage")
    amplitude = _mapping(schema["amplitude_stage"], "amplitude_stage")
    physics = _mapping(schema["physics"], "physics")
    selector = case.tables.selector_closures
    contraction = amplitude["color_contraction"]
    contraction_entries = (
        ()
        if contraction is None
        else _logical_color_contraction_entries(
            _mapping(contraction, "color_contraction")
        )
    )
    return {
        "current_slots": len(_records(current_storage["current_slots"], "currents")),
        "value_slots": len(_records(value_storage["value_slots"], "values")),
        "momentum_slots": len(_records(schema["momentum_slots"], "momenta")),
        "stages": len(case.tables.stages),
        "invocations": case.tables.invocation_count,
        "attachments": case.tables.attachment_count,
        "finalizations": sum(len(stage.finalizations) for stage in case.tables.stages),
        "closures": len(case.tables.closures),
        "couplings": len(case.tables.couplings),
        "selector_domains": 0 if selector is None else len(selector.domains),
        "selector_memberships": (
            0 if selector is None else len(selector.domain_group_ids)
        ),
        "reduction_groups": len(
            _records(
                _mapping(physics["reduction"], "physics.reduction")["groups"],
                "physics.reduction.groups",
            )
        ),
        "color_contraction_entries": len(contraction_entries),
        "amplitude_roots": len(_records(amplitude["roots"], "roots")),
        "referenced_exact_kernels": len(case.tables.referenced_kernel_ids),
    }


def _snapshot(case: _GoldenCase) -> dict[str, object]:
    sections = {
        "layout": _layout_semantics(case.schema),
        "tables": _table_semantics(case.tables),
        "selectors": _selector_semantics(case.schema, case.tables),
        "reductions": _reduction_semantics(case.schema),
        "exact": _exact_semantics(case),
        "resolved": _resolved_probe(case.schema, case.tables),
    }
    return {
        "counts": _diagnostic_counts(case),
        "digests": {name: _digest(value) for name, value in sections.items()},
        "semantic_sha256": _digest(sections),
    }


# Refreshed after 0c4fa415 retained exact i/sqrt(2) vertex normalization and
# 961f1b67 added rational colour weights. Exact expressions change content-derived
# kernel IDs and hence table digests; exact_weight changes NLC/full reductions.
# Replaying only the former normalization and omitting exact_weight reproduces
# every preceding golden. Counts, layout, selectors and resolved probes are
# unchanged. Section digests identify semantic drift without hashing packed bytes.
# Refreshed again for exact symbolic propagator phases in the Symbolica 3 port.
# The Symbolica 2.2 baseline reproduces the preceding goldens; all eight referenced
# kernels remain symbolically equivalent. Only two Weyl propagators replace float
# unit phases with exact units, renumbering content-derived IDs across the catalog.
# Remapping those IDs reproduces all four preceding table digests. Counts, layout,
# selectors, reductions and exact-rational resolved probes are unchanged.
# The published Symbolica 3.0 wheel changes canonical complex-number printing,
# again renumbering content-derived IDs. All 53 kernels are symbolically identical
# to the retained candidate pack. Restoring that pack's exact strings/signatures
# and remapping IDs reproduces every preceding exact and table digest; all other
# sections remain unchanged.
_EXPECTED: dict[str, dict[str, object]] = {
    "lc-topology-replay": {
        "counts": {
            "amplitude_roots": 24,
            "attachments": 126,
            "closures": 24,
            "color_contraction_entries": 0,
            "couplings": 2,
            "current_slots": 69,
            "finalizations": 58,
            "invocations": 126,
            "momentum_slots": 11,
            "reduction_groups": 24,
            "referenced_exact_kernels": 6,
            "selector_domains": 59,
            "selector_memberships": 144,
            "stages": 3,
            "value_slots": 69,
        },
        "digests": {
            "exact": "b1c2417ad5ef0283b529f4d347f7e174df526a191df598899e1737aa76065824",
            "layout": "b62b48a5e371e65041aff8f7f0f47ccc6b354a838360b1ba8480de104744d83b",
            "reductions": "3864a2185e4b6613de4cd650431a1cb6a42c224556e73ffb57d471dc5424b6f1",
            "resolved": "4347816ea9472a7c77bd4638259933ac67a562a9fa7478795233da6dc43cab7b",
            "selectors": "e05447b4632d2b06aed2ecc2ecec8526d3aa65e62583ae58f52f4881207435cb",
            "tables": "c636926e14178c9b2a833a1c976a1100db7a4796aac323d44abb6fcf2b68a1c3",
        },
        "semantic_sha256": "15068225c280e5bb16516b044556e60bdd1e175664da650a2d00da154c65a189",
    },
    "lc-all-flow-union": {
        "counts": {
            "amplitude_roots": 48,
            "attachments": 242,
            "closures": 48,
            "color_contraction_entries": 0,
            "couplings": 2,
            "current_slots": 117,
            "finalizations": 106,
            "invocations": 210,
            "momentum_slots": 13,
            "reduction_groups": 48,
            "referenced_exact_kernels": 6,
            "selector_domains": 139,
            "selector_memberships": 384,
            "stages": 3,
            "value_slots": 117,
        },
        "digests": {
            "exact": "907d6b5d5681b55224a7582b600236d63aceb4bbed8e2ca5ab3409f394eceed0",
            "layout": "24432cc2b2d22a0c83adaddfe58e0beb1ce8447c5df44417e4b6c7e00fffb690",
            "reductions": "b7d5c3c614f70ad6c4bf7f8afb1b3a0a8a5c33dcae0a0c12d271992928ec3a2a",
            "resolved": "f038d7b438a5b59baabc439fb712dbe3fd455f433f20e9da7490e574acc22927",
            "selectors": "7eafd41c8048e0c0d992c5078af4d67e94a9295688d8166a4962ff129a99b419",
            "tables": "ef1d7173787a43a39214419b103b7a96f6abba3efc54743b4e00ba938de6c9fb",
        },
        "semantic_sha256": "e19d09c7b176933d503d84a821fac830573a8ddb90ccc5fe40f0906bf37966ca",
    },
    "nlc-contracted": {
        "counts": {
            "amplitude_roots": 9,
            "attachments": 60,
            "closures": 9,
            "color_contraction_entries": 63,
            "couplings": 1,
            "current_slots": 42,
            "finalizations": 35,
            "invocations": 54,
            "momentum_slots": 10,
            "reduction_groups": 9,
            "referenced_exact_kernels": 4,
            "selector_domains": 16,
            "selector_memberships": 23,
            "stages": 2,
            "value_slots": 42,
        },
        "digests": {
            "exact": "d488c9d3052b05076ec2c05c9ab16d258fff27125769ed0e777371d2fe4bf065",
            "layout": "6c829c0734a995d06f4dcc63bc0967d77bd838c7c733c0ef204f68b4c80ac32a",
            "reductions": "2aac0b41a34bf54335d14337c4cc130f078b11c8a13dce0e5ed7ff2eb7c0bb4d",
            "resolved": "673928032c4063e5aacb8eab41076de0d6c49ea64e56b13c0fa43c149c2c2393",
            "selectors": "72534cf8b4737ba7bbf91913532f6240828386a1cd1d3146b253f44680af0a00",
            "tables": "0831eff6d1237197672f30cecbeb763edf2682803920f07e52e11b2bfa2002a3",
        },
        "semantic_sha256": "90ae984c8e55ef27182edd2e8b4bad27e3f39c8d054c1834b95e707d15e54da7",
    },
    "full-contracted": {
        "counts": {
            "amplitude_roots": 9,
            "attachments": 60,
            "closures": 9,
            "color_contraction_entries": 63,
            "couplings": 1,
            "current_slots": 42,
            "finalizations": 35,
            "invocations": 54,
            "momentum_slots": 10,
            "reduction_groups": 9,
            "referenced_exact_kernels": 4,
            "selector_domains": 16,
            "selector_memberships": 23,
            "stages": 2,
            "value_slots": 42,
        },
        "digests": {
            "exact": "d488c9d3052b05076ec2c05c9ab16d258fff27125769ed0e777371d2fe4bf065",
            "layout": "71ddc16d027bd79283f4861d26a53e3233faf17f2411db49c9607bd21fbb8318",
            "reductions": "0c27a7a5da29bf9569e5477991283edf65c1acd5e424af9a8a15d828d985dd82",
            "resolved": "19acacf5110579b2c157b48dc5c4aacac3da5b51d76f13658ce4a5cbd73754de",
            "selectors": "6f33e6fbd9d47db7a6f0bb8434606ef5b24eb2c644fd3d03006a35201b89a6d3",
            "tables": "387438ecdc15df79663950e05408d4cd675c1157946ff7a451d89cd48b2a222d",
        },
        "semantic_sha256": "d64112012b5cc7280936f2ec8b55ef14b213b226e93734c436954d94716ec545",
    },
}


def test_plan_v2_abi_and_capability_are_the_explicit_legacy_contract() -> None:
    assert EAGER_PLAN_ABI == _LEGACY_PLAN_ABI
    assert EAGER_RUNTIME_CAPABILITY == _LEGACY_RUNTIME_CAPABILITY


def test_plan_v2_semantic_goldens(golden_cases: tuple[_GoldenCase, ...]) -> None:
    actual = {case.name: _snapshot(case) for case in golden_cases}
    assert actual == _EXPECTED
