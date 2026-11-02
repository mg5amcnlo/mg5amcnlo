# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol._internal.versions import EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY
from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.artifacts import ArtifactBuilder, load_manifest
from pyamplicol.generation.eager_lowering import EAGER_RUNTIME_KIND
from pyamplicol.generation.eager_tables import (
    EAGER_KERNEL_ABI,
    EAGER_OUTPUT_FACTOR_COUPLING_REAL,
    EAGER_OUTPUT_FACTOR_NONE,
    EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY,
    MISSING_U32,
    EagerAttachmentRow,
    EagerClosureRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerInvocationRow,
    pack_rows,
)
from pyamplicol.models.prepared import PreparedKernelPack, PreparedKernelRecord
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime.eager_exact import (
    EagerExactExecutor as _EagerExactExecutor,
)
from pyamplicol.runtime.eager_exact import _plan_v3
from pyamplicol.runtime.eager_exact._contracts import (
    _ExactComplexProduct,
    _resolve_exact_factor,
)
from pyamplicol.runtime.eager_exact._execution import _gather_inputs
from pyamplicol.runtime.eager_exact._plan import (
    _EagerExactPlan,
    _prepared_parameter_projection,
)
from pyamplicol.runtime.eager_exact._plan_v3 import (
    EAGER_EXACT_SECTIONS_ABI,
    EAGER_PLAN_V3_ABI,
    EAGER_PLAN_V3_RUNTIME_CAPABILITY,
    EAGER_REDUCTION_GROUPS_ABI,
    EAGER_RUNTIME_LAYOUT_ABI,
)

eager_plan_v3 = _plan_v3

_ComplexDecimal = tuple[Decimal, Decimal]
_ExactCallable = Callable[[Sequence[_ComplexDecimal], int], Sequence[_ComplexDecimal]]
_COMPACT_SECTIONS_BY_ARTIFACT: dict[Path, dict[str, object]] = {}


def test_exact_factor_product_is_combined_at_requested_precision() -> None:
    factors = _ExactComplexProduct(
        numerators=(
            (Decimal.from_float(0.1), Decimal(0)),
            (Decimal.from_float(0.2), Decimal(0)),
        ),
        denominator=(Decimal.from_float(0.3), Decimal(0)),
    )

    with localcontext() as context:
        context.prec = 50
        resolved = _resolve_exact_factor(factors)
        expected = (
            Decimal.from_float(0.1) * Decimal.from_float(0.2) / Decimal.from_float(0.3)
        )

    assert resolved == (expected, Decimal(0))
    assert resolved[0] != Decimal.from_float((0.1 * 0.2) / 0.3)


class _NativeRuntime:
    def __init__(
        self,
        model_parameter_values: Sequence[float] = (),
        normalization_factor: float = 1.0,
        *,
        representative_process_id: str = "synthetic",
        representative_process_key: str = "synthetic",
        external_permutation: Sequence[int] = (0, 1, 2),
        physics: Mapping[str, object] | None = None,
    ) -> None:
        self._model_parameter_values = tuple(model_parameter_values)
        self._normalization_factor = normalization_factor
        self._representative_process_id = representative_process_id
        self._representative_process_key = representative_process_key
        self._external_permutation = tuple(external_permutation)
        self._public_physics = physics

    def _exact_runtime_state_json(self) -> str:
        return json.dumps(
            {
                "model_parameter_values": self._model_parameter_values,
                "normalization_factor": self._normalization_factor,
                "representative_process_id": self._representative_process_id,
                "representative_process_key": self._representative_process_key,
                "external_permutation": self._external_permutation,
            }
        )

    def physics_json(self) -> str:
        return json.dumps(self._public_physics or _physics())


def test_eager_diagnostic_projection_maps_alias_masses_to_caller_order() -> None:
    executor = object.__new__(_EagerExactExecutor)
    schema = {
        "model": {
            "particles": [
                {"pdg": 25, "mass": 125.0, "mass_parameter": "ufo.MH"},
                {"pdg": 21, "mass": 0.0, "mass_parameter": None},
            ]
        },
        "model_parameters": [
            {"name": "ufo.MH", "parameter_index": 0},
        ],
    }
    executor._native_runtime = _NativeRuntime((3.0,))
    executor._exact_execution = {"runtime_schema": schema}
    executor._physics = {
        "external_particles": [
            {"label": 1, "pdg": 21, "role": "final"},
            {"label": 2, "pdg": 25, "role": "final"},
        ]
    }
    # Native public physics and its PDG-derived bindings are already (g, H).
    # The retained representative permutation must not be applied a second time.
    executor._permutation = (1, 0)

    projected, metadata = executor._diagnostic_project_onshell(
        (((-4.0, 0.0, 0.0, 4.0), (4.9, 4.0, 0.0, 0.0)),),
        precision=200,
    )

    assert projected[0][0][0] == Decimal("-4.0")
    assert projected[0][1][0] == Decimal("5.0")
    gluon, higgs = metadata["points"][0]["legs"]
    assert (gluon["pdg"], gluon["mass"], gluon["projected_energy"]) == (
        21,
        "0.0",
        "-4.0",
    )
    assert higgs["pdg"] == 25
    assert higgs["mass"] == "3.0"
    assert higgs["mass_parameter"] == "ufo.MH"
    assert Decimal(higgs["shell_residual_after"]) == 0


def _contract(role: str, component: int) -> dict[str, object]:
    return {
        "role": role,
        "component": component,
        "symbol": f"test::{role}::{component}",
        "model_parameter_name": None,
        "model_parameter_index": None,
    }


def _parameter_contract(name: str, index: int) -> dict[str, object]:
    return {
        "role": "model-parameter",
        "component": 0,
        "symbol": f"test::model-parameter::{index}",
        "model_parameter_name": name,
        "model_parameter_index": index,
    }


def _kernel(
    kernel_id: int,
    kind: str,
    contracts: tuple[dict[str, object], ...],
    *,
    output_layout: tuple[str, ...] = ("scalar:0",),
) -> PreparedKernelRecord:
    exact_path = f"kernels/{kernel_id:06d}/exact.bin"
    return PreparedKernelRecord(
        kernel_id=kernel_id,
        contract_kind=kind,  # type: ignore[arg-type]
        canonical_signature=f"test:{kind}:{kernel_id}",
        input_arity=len(contracts),
        output_arity=len(output_layout),
        input_layout=tuple(
            f"{contract['role']}:{contract['component']}" for contract in contracts
        ),
        input_contracts=contracts,
        output_layout=output_layout,
        exact_expressions=tuple(
            f"test::output::{index}" for index in range(len(output_layout))
        ),
        exact_evaluator_state_path=exact_path,
        f64_evaluator_manifest={
            "kind": "symjit-application-evaluator",
            "input_len": len(contracts),
            "output_len": len(output_layout),
            "optimization_level": 2,
            "settings": {"jit_optimization_level": 2},
            "application_path": f"kernels/{kernel_id:06d}/application.symjit",
            "evaluator_state_path": exact_path,
        },
    )


def _source_record() -> dict[str, object]:
    crossing = {
        "momentum_transform": "identity",
        "helicity_factor": 1,
        "chirality_factor": 1,
        "spin_state_factor": 1,
        "phase": [1.0, 0.0],
    }
    return {
        "source_id": 0,
        "current_id": 0,
        "current_component_start": 0,
        "current_component_stop": 1,
        "value_slot": {
            "value_slot_id": 0,
            "current_id": 0,
            "variant": "source",
            "component_start": 0,
            "component_stop": 1,
            "dimension": 1,
        },
        "source_parameter_start": 0,
        "source_parameter_stop": 1,
        "leg_label": 1,
        "input_momentum_slot": 0,
        "side": "final",
        "crossing": "identity",
        "physical_pdg": 9000001,
        "outgoing_pdg": 9000001,
        "particle_id": 9000001,
        "anti_particle_id": 9000001,
        "source_kind": "external-wavefunction",
        "wavefunction_kind": "scalar",
        "source_orientation": "self-conjugate",
        "source_basis": "scalar",
        "source_ir": {
            "basis": "scalar",
            "component_dimension": 1,
            "crossing": crossing,
            "identity": {
                "pdg_label": 9000001,
                "anti_pdg_label": 9000001,
                "orientation": "self-conjugate",
            },
            "states": [{"helicity": 0, "chirality": 0, "spin_state": 0}],
            "wavefunction_family": "scalar",
        },
        "applied_crossing": crossing,
        "source_helicity": 0,
        "chirality": 0,
        "spin_state": 0,
        "dimension": 1,
        "helicity_ancestry": "1",
        "color_state": {},
    }


def _runtime_schema(*, direct_closure: bool) -> dict[str, object]:
    root: dict[str, object] = {
        "output_index": 0,
        "root_id": 0,
        "kind": "direct-contraction" if direct_closure else "prepared-closure",
        "left_current_id": 1,
        "right_current_id": 0,
        "left_value_slot": {"value_slot_id": 2},
        "right_value_slot": {"value_slot_id": 0},
        "coherent_group_id": 0,
        "color_sector_id": 0,
        "helicity_weight": 1.0,
        "all_sector_weight": 1.0,
    }
    if direct_closure:
        root["contraction_ir"] = {"coefficients": [[2.0, 0.0]]}
    return {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-execution-plan",
        "process_key": "synthetic",
        "process": "s > s",
        "external_particles": [{"label": 1, "role": "final"}],
        "model": {"particles": []},
        "model_parameters": [],
        "normalization": {},
        "parameter_layout": {
            "source_component_parameter_count": 1,
            "momentum_parameter_count": 4,
            "model_parameter_count": 0,
            "parameter_count_if_flattened": 5,
            "value_component_count": 3,
            "source_components_complex": True,
            "momentum_components_real": True,
            "real_valued_inputs": [1, 2, 3, 4],
        },
        "current_storage": {
            "component_count": 2,
            "number_type": "complex",
            "metadata_compacted": True,
            "current_slots": [
                {
                    "current_id": 0,
                    "component_start": 0,
                    "component_stop": 1,
                    "dimension": 1,
                    "is_source": True,
                },
                {
                    "current_id": 1,
                    "component_start": 1,
                    "component_stop": 2,
                    "dimension": 1,
                    "is_source": False,
                },
            ],
        },
        "value_storage": {
            "component_count": 3,
            "number_type": "complex",
            "metadata_compacted": True,
            "value_slots": [
                {
                    "value_slot_id": 0,
                    "current_id": 0,
                    "variant": "source",
                    "component_start": 0,
                    "component_stop": 1,
                    "dimension": 1,
                },
                {
                    "value_slot_id": 1,
                    "current_id": 1,
                    "variant": "unpropagated",
                    "component_start": 1,
                    "component_stop": 2,
                    "dimension": 1,
                },
                {
                    "value_slot_id": 2,
                    "current_id": 1,
                    "variant": "propagated",
                    "component_start": 2,
                    "component_stop": 3,
                    "dimension": 1,
                },
            ],
        },
        "source_fill": {"source_count": 1, "sources": [_source_record()]},
        "momentum_slots": [
            {
                "momentum_slot_id": 0,
                "momentum_mask": 1,
                "external_labels": [1],
                "component_start": 0,
                "component_stop": 4,
                "real_valued": True,
            }
        ],
        "stages": [
            {
                "stage_index": 1,
                "stage_kind": "current-combine",
                "subset_size": 1,
                "input_current_ids": [0],
                "output_current_ids": [1],
                "input_value_slot_ids": [0],
                "output_value_slot_ids": [1, 2],
                "interaction_count": 2,
                "interactions_compacted": True,
                "interaction_ids": [0, 1],
                "interactions": [],
            }
        ],
        "amplitude_stage": {
            "stage_kind": "amplitude",
            "output_count": 1,
            "color_contraction": None,
            "roots": [root],
        },
    }


def _physics() -> dict[str, object]:
    return {
        "extensions": {"normalization": {}},
        "schema_version": 1,
        "kind": "pyamplicol-runtime-physics",
        "process_id": "synthetic",
        "process": "s > s s",
        "color_accuracy": "lc",
        "external_particles": [{}, {}, {}],
        "helicities": [{"id": "h:0", "values": [0, 0, 0], "coefficient": 1.0}],
        "color_components": [
            {
                "id": "flow:1",
                "kind": "lc-flow",
                "word": [1],
                "coefficient": 1.0,
            }
        ],
        "reduction": {
            "kind": "symmetry-reduction",
            "groups": [
                {
                    "id": "reduction:0",
                    "physical_helicity_ids": ["h:0"],
                    "physical_color_ids": ["flow:1"],
                    "representative_helicity_id": "h:0",
                    "representative_color_id": "flow:1",
                }
            ],
        },
    }


def _momenta(*energies: int) -> list[list[tuple[int, int, int, int]]]:
    return [
        [(energy, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
        for energy in energies
    ]


def _producer() -> dict[str, object]:
    return {
        "distribution": "pyamplicol",
        "version": "0.1.0",
        "versions": {
            "python_api": 1,
            "toml": 1,
            "compiled_model": 9,
            "process_artifact": 3,
            "runtime_physics": 1,
            "symbolica_serialization": "test",
            "c_abi": 1,
        },
        "target": {"triple": "test-target", "cpu_features": []},
    }


def _build_artifact(
    root: Path,
    *,
    direct_closure: bool = False,
    invocation_kernel_id: int = 10,
    invocation_row_size: int | None = None,
    plan_abi: str = EAGER_PLAN_ABI,
    kernel_abi: str | None = EAGER_KERNEL_ABI,
    prepared_backend: str = "jit",
    vertex_contracts: tuple[dict[str, object], ...] | None = None,
    parameter_kernel: PreparedKernelRecord | None = None,
    extra_kernels: Sequence[PreparedKernelRecord] = (),
    model_parameters: Sequence[dict[str, object]] = (),
    coupling_row: EagerCouplingRow | None = None,
    invocation_output_factor_source: int = EAGER_OUTPUT_FACTOR_NONE,
    closure_output_factor_source: int = EAGER_OUTPUT_FACTOR_NONE,
    compact_v3: bool = True,
) -> dict[str, object] | None:
    payload_target = {"triple": "test-target", "cpu_features": []}
    base_kernels = (
        _kernel(
            10,
            "vertex",
            vertex_contracts
            or (
                _contract("left-current", 0),
                _contract("right-current", 0),
                _contract("coupling-real", 0),
            ),
        ),
        _kernel(
            11,
            "propagator",
            (_contract("current", 0), _contract("momentum", 0)),
        ),
        _kernel(
            12,
            "closure",
            (
                _contract("left-current", 0),
                _contract("right-current", 0),
                _contract("coupling-real", 0),
            ),
        ),
    )
    kernels = base_kernels + tuple(extra_kernels)
    if parameter_kernel is not None:
        kernels += (parameter_kernel,)
    pack = PreparedKernelPack(
        backend="jit",
        optimization_settings={"jit_optimization_level": 2},
        producer={"distribution": "pyamplicol", "version": "test"},
        dependency_abis={
            "symbolica_serialization": "test",
            "symjit_plane_application": "pyamplicol-symjit-plane-application-v2",
        },
        provenance={"compiled_model": "test"},
        target={
            "portable": True,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "symjit-storage-v3-portable",
            "cpu_features": [],
        },
        resolver_manifest={
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "synthetic",
        },
        kernels=kernels,
    )
    pack_payload = pack.to_dict()
    if kernel_abi is not None:
        pack_payload["eager_kernel_abi"] = kernel_abi
    pack_payload["backend"] = prepared_backend
    coupling_rows = (
        coupling_row or EagerCouplingRow(MISSING_U32, MISSING_U32, 1.0, 0.0),
    )
    invocations = (
        EagerInvocationRow(
            invocation_kernel_id,
            0,
            0,
            0,
            0,
            0,
            invocation_output_factor_source,
            0,
            1,
        ),
        EagerInvocationRow(
            invocation_kernel_id,
            0,
            0,
            0,
            0,
            0,
            invocation_output_factor_source,
            1,
            1,
        ),
    )
    attachments = (
        EagerAttachmentRow(1, 1.0, 0.0),
        EagerAttachmentRow(1, 2.0, 0.0),
    )
    finalizations = (EagerFinalizationRow(11, 1, 1, 2, 0),)
    closure = EagerClosureRow(
        MISSING_U32 if direct_closure else 12,
        2,
        0,
        0,
        MISSING_U32 if direct_closure else 0,
        closure_output_factor_source,
        0.5 if direct_closure else 1.0,
        0.0,
    )
    tables = {
        "eager/couplings.bin": pack_rows(coupling_rows),
        "eager/stage-1-invocations.bin": pack_rows(invocations),
        "eager/stage-1-attachments.bin": pack_rows(attachments),
        "eager/stage-1-finalizations.bin": pack_rows(finalizations),
        "eager/closures.bin": pack_rows((closure,)),
    }
    plan = {
        "kind": EAGER_RUNTIME_KIND,
        "eager_plan_abi": plan_abi,
        "required_runtime_capabilities": [EAGER_RUNTIME_CAPABILITY],
        "process_key": "synthetic",
        "couplings": {
            "path": "eager/couplings.bin",
            "count": 1,
            "row_size": EagerCouplingRow._STRUCT.size,
        },
        "stages": [
            {
                "stage_index": 1,
                "subset_size": 1,
                "invocations": {
                    "path": "eager/stage-1-invocations.bin",
                    "count": 2,
                    "row_size": (
                        invocation_row_size
                        if invocation_row_size is not None
                        else EagerInvocationRow._STRUCT.size
                    ),
                },
                "attachments": {
                    "path": "eager/stage-1-attachments.bin",
                    "count": 2,
                    "row_size": EagerAttachmentRow._STRUCT.size,
                },
                "finalizations": {
                    "path": "eager/stage-1-finalizations.bin",
                    "count": 1,
                    "row_size": EagerFinalizationRow._STRUCT.size,
                },
            }
        ],
        "closures": {
            "path": "eager/closures.bin",
            "count": 1,
            "row_size": EagerClosureRow._STRUCT.size,
        },
    }
    runtime_schema = _runtime_schema(direct_closure=direct_closure)
    runtime_schema["model_parameters"] = list(model_parameters)
    parameter_layout = runtime_schema["parameter_layout"]
    assert isinstance(parameter_layout, dict)
    parameter_layout["model_parameter_count"] = len(model_parameters)
    parameter_layout["parameter_count_if_flattened"] = (
        int(parameter_layout["source_component_parameter_count"])
        + int(parameter_layout["momentum_parameter_count"])
        + len(model_parameters)
    )
    execution = {
        "schema_version": 3,
        "kind": EAGER_RUNTIME_KIND,
        "required_runtime_capabilities": [EAGER_RUNTIME_CAPABILITY],
        "process": "s > s",
        "key": "synthetic",
        "color_accuracy": "lc",
        "external_pdg_order": [9000001, 9000001, 9000001],
        "eager_plan_abi": plan_abi,
        "kernel_pack": {
            "manifest_path": "model/eager-kernel-pack.json",
            "payload_root": "model/eager-kernels",
        },
        "runtime_options": {"point_tile_size": 1024, "workspace_mib": 256},
        "plan": plan,
        "dag_summary": {},
        "runtime_schema": runtime_schema,
    }
    capabilities = [EAGER_RUNTIME_CAPABILITY]
    compact_sections: dict[str, object] | None = None
    if compact_v3:
        capabilities = sorted(
            (
                EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
                EAGER_PLAN_V3_RUNTIME_CAPABILITY,
            )
        )
        execution = {
            "schema_version": 3,
            "kind": EAGER_RUNTIME_KIND,
            "required_runtime_capabilities": capabilities,
            "process": "s > s",
            "key": "synthetic",
            "color_accuracy": "lc",
            "external_pdg_order": [9000001, 9000001, 9000001],
            "eager_plan_abi": EAGER_PLAN_V3_ABI,
            "kernel_pack": {
                "manifest_path": "model/eager-kernel-pack.json",
                "payload_root": "model/eager-kernels",
            },
            "runtime_options": {"point_tile_size": 1024, "workspace_mib": 256},
            "plan": {
                "kind": EAGER_RUNTIME_KIND,
                "eager_plan_abi": EAGER_PLAN_V3_ABI,
                "lowering_input_abi": "pyamplicol-eager-lowering-input-v1",
                "lowering_input_sha256": "1" * 64,
                "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
                "required_runtime_capabilities": capabilities,
                "runtime_container": {
                    "kind": "pyamplicol-eager-runtime-container",
                    "schema_version": 1,
                    "storage_abi": "pacbin-v1",
                    "path": "eager-runtime.pacbin",
                    "size_bytes": 1,
                    "sha256": "2" * 64,
                    "member_count": 1,
                    "unpacked_size_bytes": 1,
                    "index_sha256": "3" * 64,
                },
                "inspection_summary": {},
            },
            "dag_summary": {},
        }
        compact_sections = {
            "abi": EAGER_EXACT_SECTIONS_ABI,
            "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
            "process_id": "synthetic",
            "exact_schema": runtime_schema,
            "reduction_groups": _physics()["reduction"]["groups"],
            "selector_group_ids": [0],
            "selector_domains": [[0]],
            "couplings": [
                [
                    row.real_parameter_id,
                    row.imag_parameter_id,
                    str(row.constant_real),
                    str(row.constant_imag),
                ]
                for row in coupling_rows
            ],
            "stages": [[1, 0, len(invocations), 0, len(attachments), 0, 1]],
            "invocations": [[*row._values(), 0] for row in invocations],
            "attachments": [
                [
                    row.result_current_id,
                    [[str(row.factor_real), str(row.factor_imag)]],
                    None,
                    0,
                ]
                for row in attachments
            ],
            "finalizations": [[*row._values(), 0, 0] for row in finalizations],
            "closures": [
                [
                    closure.kernel_id,
                    closure.left_value_slot_id,
                    closure.right_value_slot_id,
                    closure.amplitude_index,
                    closure.coupling_slot_id,
                    closure.output_factor_source,
                    [[str(closure.factor_real), str(closure.factor_imag)]],
                    None,
                    [["2.0", "0.0"]] if direct_closure else None,
                    0,
                    0,
                ]
            ],
        }

    with ArtifactBuilder(root) as builder:
        builder.add_json(
            "model/eager-kernel-pack.json",
            pack_payload,
            role="evaluator-manifest",
        )
        for kernel in kernels:
            builder.add_bytes(
                f"model/eager-kernels/{kernel.exact_evaluator_state_path}",
                b"mock exact evaluator",
                role="evaluator-state",
                media_type="application/octet-stream",
                target=payload_target,
            )
        builder.add_json(
            "processes/synthetic/execution.json",
            execution,
            role="evaluator-manifest",
            process_id="synthetic",
        )
        if compact_v3:
            builder.add_bytes(
                "processes/synthetic/eager-runtime.pacbin",
                b"x",
                role="evaluator-state",
                media_type="application/octet-stream",
                target=payload_target,
                process_id="synthetic",
            )
        builder.add_json(
            "processes/synthetic/physics.json",
            _physics(),
            role="runtime-physics",
            process_id="synthetic",
        )
        if not compact_v3:
            for path, content in tables.items():
                builder.add_bytes(
                    f"processes/synthetic/{path}",
                    content,
                    role="evaluator-state",
                    media_type="application/octet-stream",
                    target=payload_target,
                    process_id="synthetic",
                )
        builder.finalize(
            kind="pyamplicol-process",
            producer=_producer(),
            model={
                "name": "synthetic",
                "source_kind": "built-in-sm",
                "content_sha256": "1" * 64,
                "compiled_schema_version": 9,
            },
            configuration={
                "toml_schema_version": 1,
                "requested_path": "config/requested.toml",
                "effective_path": "config/effective.toml",
                "adjustments": [],
            },
            processes=[
                {
                    "id": "synthetic",
                    "expression": "s > s s",
                    "color_accuracy": "lc",
                    "external_pdgs": [9000001, 9000001, 9000001],
                    "physics_path": "processes/synthetic/physics.json",
                    "required_runtime_capabilities": capabilities,
                    "aliases": [],
                }
            ],
            default_process_id="synthetic",
            runtime={
                "engine": "rusticol",
                "engine_version": "0.1.0",
                "evaluator_manifest_path": "processes/evaluators.json",
                "api_bundle_path": None,
                "required_runtime_capabilities": capabilities,
            },
        )
    resolved_root = root.resolve()
    if compact_sections is None:
        _COMPACT_SECTIONS_BY_ARTIFACT.pop(resolved_root, None)
    else:
        _COMPACT_SECTIONS_BY_ARTIFACT[resolved_root] = compact_sections
    return compact_sections


class EagerExactExecutor(_EagerExactExecutor):
    """Test adapter that supplies the synthetic compact native sections."""

    def __init__(
        self,
        artifact: Path,
        process_id: str,
        native_runtime: object,
        **options: object,
    ) -> None:
        if options.get("native_sections_loader") is None:
            sections = _COMPACT_SECTIONS_BY_ARTIFACT.get(artifact.resolve())
            if sections is not None:
                options["native_sections_loader"] = (
                    lambda _root, _process, payload=sections: payload
                )
        super().__init__(
            artifact,
            process_id,
            native_runtime,
            **options,  # type: ignore[arg-type]
        )


def _load_legacy_plan(
    artifact: Path,
    *,
    kernel_loader: Callable[[PreparedKernelRecord, Path], _ExactCallable],
) -> _EagerExactPlan:
    manifest = load_manifest(artifact)
    execution = json.loads(
        (artifact / "processes/synthetic/execution.json").read_text(encoding="utf-8")
    )
    return _EagerExactPlan.load(
        artifact_root=artifact,
        process_root=artifact / "processes/synthetic",
        process_id="synthetic",
        execution=execution,
        manifest=manifest,
        kernel_loader=kernel_loader,
        exact_payloads=ExactEvaluatorPayloadResolver(manifest),
    )


def _loader(
    finalization_inputs: list[tuple[_ComplexDecimal, ...]],
) -> Callable[[PreparedKernelRecord, Path], _ExactCallable]:
    def load(record: PreparedKernelRecord, _root: Path) -> _ExactCallable:
        if record.kernel_id == 10:
            return lambda values, _precision: (
                (
                    values[0][0] + values[1][0] + values[2][0],
                    Decimal(0),
                ),
            )
        if record.kernel_id == 11:

            def finalize(
                values: Sequence[_ComplexDecimal], _precision: int
            ) -> tuple[_ComplexDecimal, ...]:
                finalization_inputs.append(tuple(values))
                return ((values[0][0] * 10 + values[1][0], Decimal(0)),)

            return finalize
        if record.kernel_id == 12:
            return lambda values, _precision: (
                (
                    values[0][0] + values[1][0] + values[2][0],
                    Decimal(0),
                ),
            )
        raise AssertionError(f"unexpected kernel {record.kernel_id}")

    return load


def test_eager_exact_accumulates_then_finalizes_once(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _build_artifact(artifact)
    finalization_inputs: list[tuple[_ComplexDecimal, ...]] = []
    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime(),
        kernel_loader=_loader(finalization_inputs),
    )

    result = executor.evaluate_resolved(
        _momenta(5),
        helicities=None,
        color_flows=None,
        precision=50,
    )

    assert finalization_inputs == [((Decimal(9), Decimal(0)), (Decimal(5), Decimal(0)))]
    assert result.values == (((Decimal(9409),),),)
    assert result.helicity_ids == ("h:0",)
    assert result.color_ids == ("flow:1",)


def test_eager_exact_uses_native_representative_for_inferred_expression(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _build_artifact(artifact)
    executor = EagerExactExecutor(
        artifact,
        "s > x s",
        _NativeRuntime(external_permutation=(0, 2, 1)),
        kernel_loader=_loader([]),
    )

    assert executor._permutation == (0, 2, 1)


def test_eager_exact_plan_v3_uses_native_compact_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    sections = _build_artifact(artifact, compact_v3=True)
    assert sections is not None
    calls: list[tuple[str, str]] = []

    def load_sections(artifact_root: str, process_id: str) -> object:
        calls.append((artifact_root, process_id))
        return sections

    module = SimpleNamespace(_load_eager_exact_sections_v1=load_sections)
    monkeypatch.setattr(
        eager_plan_v3.importlib,
        "import_module",
        lambda name: (
            module
            if name == "pyamplicol._rusticol"
            else pytest.fail(f"unexpected native import {name}")
        ),
    )
    finalization_inputs: list[tuple[_ComplexDecimal, ...]] = []
    executor = _EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime(),
        kernel_loader=_loader(finalization_inputs),
    )

    result = executor.evaluate_resolved(
        _momenta(5),
        helicities=None,
        color_flows=None,
        precision=50,
    )

    assert calls == [(str(artifact.resolve()), "synthetic")]
    assert finalization_inputs == [((Decimal(9), Decimal(0)), (Decimal(5), Decimal(0)))]
    assert result.values == (((Decimal(9409),),),)


def test_eager_exact_plan_v3_rejects_wrong_native_section_abi(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    sections = _build_artifact(artifact, compact_v3=True)
    assert sections is not None
    sections["abi"] = "pyamplicol-eager-exact-sections-v0"

    with pytest.raises(CompatibilityError, match="exact-sections ABI"):
        EagerExactExecutor(
            artifact,
            "synthetic",
            _NativeRuntime(),
            kernel_loader=_loader([]),
            native_sections_loader=lambda _root, _process: sections,
        )


def test_eager_exact_plan_v3_preserves_attachment_start_diagnostic(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    sections = _build_artifact(artifact, compact_v3=True)
    assert sections is not None
    stages = sections["stages"]
    assert isinstance(stages, list)
    stage = stages[0]
    assert isinstance(stage, list)
    stage[3] = "invalid"

    with pytest.raises(
        ArtifactError,
        match=r"compact eager stage 0 start must be an integer",
    ):
        eager_plan_v3._load_eager_exact_sections_v1(
            artifact,
            "synthetic",
            loader=lambda _root, _process: sections,
        )


def test_eager_reduction_groups_use_bounded_native_bridge(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    groups = (
        {
            "id": "reduction:0",
            "physical_color_ids": ["flow:1"],
            "physical_helicity_ids": ["h:0"],
        },
    )
    calls: list[tuple[Path, str]] = []

    def load_groups(artifact_root: Path, process_id: str) -> object:
        calls.append((artifact_root, process_id))
        return {
            "abi": EAGER_REDUCTION_GROUPS_ABI,
            "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
            "process_id": process_id,
            "reduction_groups": groups,
        }

    loaded = eager_plan_v3._load_eager_reduction_groups_v1(
        artifact,
        "synthetic",
        loader=load_groups,
    )

    assert calls == [(artifact, "synthetic")]
    assert loaded == groups


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("abi", "pyamplicol-eager-reduction-groups-v0", "reduction-groups ABI"),
        ("runtime_layout_abi", "wrong-layout", "runtime-layout ABI"),
        ("process_id", "wrong-process", "wrong process"),
        ("reduction_groups", None, "must be an array"),
    ),
)
def test_eager_reduction_groups_reject_malformed_native_payload(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    payload: dict[str, object] = {
        "abi": EAGER_REDUCTION_GROUPS_ABI,
        "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
        "process_id": "synthetic",
        "reduction_groups": (),
    }
    payload[field] = value

    with pytest.raises((ArtifactError, CompatibilityError), match=match):
        eager_plan_v3._load_eager_reduction_groups_v1(
            tmp_path / "artifact",
            "synthetic",
            loader=lambda _root, _process: payload,
        )


def test_eager_exact_applies_dynamic_output_factor_after_kernel(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    _build_artifact(
        artifact,
        direct_closure=True,
        vertex_contracts=(
            _contract("left-current", 0),
            _contract("right-current", 0),
        ),
        coupling_row=EagerCouplingRow(MISSING_U32, MISSING_U32, 2.0, 7.0),
        invocation_output_factor_source=EAGER_OUTPUT_FACTOR_COUPLING_REAL,
    )
    finalization_inputs: list[tuple[_ComplexDecimal, ...]] = []

    def load(record: PreparedKernelRecord, _root: Path) -> _ExactCallable:
        if record.kernel_id == 10:
            return lambda values, _precision: (
                (values[0][0] + values[1][0], Decimal(0)),
            )
        if record.kernel_id == 11:

            def finalize(
                values: Sequence[_ComplexDecimal], _precision: int
            ) -> tuple[_ComplexDecimal, ...]:
                finalization_inputs.append(tuple(values))
                return (values[0],)

            return finalize
        raise AssertionError(f"unexpected kernel {record.kernel_id}")

    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime(),
        kernel_loader=load,
    )
    executor.evaluate_resolved(
        _momenta(5),
        helicities=None,
        color_flows=None,
        precision=50,
    )

    assert finalization_inputs[0][0] == (Decimal(12), Decimal(0))


def test_eager_exact_projects_sparse_complex_prepared_parameters() -> None:
    kernel = _kernel(
        20,
        "vertex",
        (
            _parameter_contract("alpha", 7),
            _parameter_contract("derived", 157),
        ),
    )
    runtime_schema = {
        "model_parameters": [
            {
                "name": "alpha",
                "kind": "external_parameter",
                "parameter_index": 0,
                "default": 2.0,
            },
            {
                "name": "derived.real",
                "kind": "derived_parameter_component",
                "parameter_index": 1,
                "runtime_name": "derived",
                "complex_component": "real",
                "default": 3.0,
            },
            {
                "name": "derived.imag",
                "kind": "derived_parameter_component",
                "parameter_index": 2,
                "runtime_name": "derived",
                "complex_component": "imag",
                "default": 4.0,
            },
        ]
    }

    projection = _prepared_parameter_projection((kernel,), runtime_schema, 3)
    projected = projection.project((Decimal(2), Decimal(3), Decimal(4)))

    assert projection.parameter_count == 158
    assert projected[7] == (Decimal(2), Decimal(0))
    assert projected[157] == (Decimal(3), Decimal(4))
    assert _gather_inputs(
        kernel,
        first_current=(),
        second_current=(),
        first_momentum=(),
        second_momentum=(),
        coupling=None,
        prepared_parameters=projected,
    ) == (projected[7], projected[157])


def test_eager_exact_derives_complex_parameters_at_requested_precision(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    parameter_kernel = _kernel(
        20,
        "model-parameter",
        (_parameter_contract("alpha", 7),),
        output_layout=("model-parameter:derived",),
    )
    model_parameters = (
        {
            "name": "alpha.real",
            "kind": "external_parameter_component",
            "parameter_index": 0,
            "runtime_name": "alpha",
            "complex_component": "real",
            "default": 2.0,
        },
        {
            "name": "alpha.imag",
            "kind": "external_parameter_component",
            "parameter_index": 1,
            "runtime_name": "alpha",
            "complex_component": "imag",
            "default": 0.25,
        },
        {
            "name": "derived.real",
            "kind": "derived_parameter_component",
            "parameter_index": 2,
            "runtime_name": "derived",
            "complex_component": "real",
            "default": 5.351713562373095,
        },
        {
            "name": "derived.imag",
            "kind": "derived_parameter_component",
            "parameter_index": 3,
            "runtime_name": "derived",
            "complex_component": "imag",
            "default": 1.3333333333333333,
        },
    )
    _build_artifact(
        artifact,
        parameter_kernel=parameter_kernel,
        model_parameters=model_parameters,
        vertex_contracts=(
            _contract("left-current", 0),
            _contract("right-current", 0),
            _contract("coupling-real", 0),
            _contract("coupling-imag", 0),
            _parameter_contract("derived", 157),
        ),
        coupling_row=EagerCouplingRow(2, 3, 0.0, 0.0),
    )
    loaded_kernel_ids: list[int] = []
    parameter_calls: list[tuple[tuple[_ComplexDecimal, ...], int]] = []
    vertex_inputs: list[tuple[_ComplexDecimal, ...]] = []
    finalization_inputs: list[tuple[_ComplexDecimal, ...]] = []

    def load(record: PreparedKernelRecord, root: Path) -> _ExactCallable:
        loaded_kernel_ids.append(record.kernel_id)
        if record.kernel_id == 20:

            def derive(
                values: Sequence[_ComplexDecimal], precision: int
            ) -> tuple[_ComplexDecimal, ...]:
                parameter_calls.append((tuple(values), precision))
                real, imaginary = values[0]
                return (
                    (
                        real * real - imaginary * imaginary + Decimal(2).sqrt(),
                        Decimal(2) * real * imaginary + Decimal(1) / Decimal(3),
                    ),
                )

            return derive
        if record.kernel_id == 10:

            def vertex(
                values: Sequence[_ComplexDecimal], _precision: int
            ) -> tuple[_ComplexDecimal, ...]:
                vertex_inputs.append(tuple(values))
                return (
                    (
                        values[0][0]
                        + values[1][0]
                        + values[2][0]
                        + values[3][0]
                        + values[4][0]
                        + values[4][1],
                        Decimal(0),
                    ),
                )

            return vertex
        return _loader(finalization_inputs)(record, root)

    native_derived = (
        2.0 * 2.0 - 0.25 * 0.25 + math.sqrt(2.0),
        2.0 * 2.0 * 0.25 + 1.0 / 3.0,
    )
    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime((2.0, 0.25, *native_derived)),
        kernel_loader=load,
    )
    assert loaded_kernel_ids == []

    result = executor.evaluate_resolved(
        _momenta(5, 7),
        helicities=None,
        color_flows=None,
        precision=60,
    )

    assert result.values
    assert len(parameter_calls) == 1
    parameter_inputs, working_precision = parameter_calls[0]
    assert loaded_kernel_ids[0] == 20
    assert loaded_kernel_ids.count(20) == 1
    assert parameter_inputs == ((Decimal(2), Decimal("0.25")),)
    assert working_precision > 60
    with localcontext() as context:
        context.prec = working_precision
        exact_derived = (
            Decimal(2) ** 2 - Decimal("0.25") ** 2 + Decimal(2).sqrt(),
            Decimal(2) * Decimal(2) * Decimal("0.25") + Decimal(1) / Decimal(3),
        )
    assert exact_derived[0] != Decimal(str(native_derived[0]))
    assert exact_derived[1] != Decimal(str(native_derived[1]))
    assert len(vertex_inputs) == 4
    for values in vertex_inputs:
        assert values[2] == (exact_derived[0], Decimal(0))
        assert values[3] == (exact_derived[1], Decimal(0))
        assert values[4] == exact_derived

    def expected_values(derived: _ComplexDecimal) -> tuple[Decimal, ...]:
        with localcontext() as context:
            context.prec = working_precision
            vertex = Decimal(2) + Decimal(2) * (derived[0] + derived[1])
            raw = tuple(
                (Decimal(30) * vertex + energy + Decimal(1) + derived[0]) ** 2
                for energy in (Decimal(5), Decimal(7))
            )
        with localcontext() as context:
            context.prec = 60
            return tuple(+value for value in raw)

    exact_values = expected_values(exact_derived)
    native_decimal: _ComplexDecimal = (
        Decimal(str(native_derived[0])),
        Decimal(str(native_derived[1])),
    )
    native_values = expected_values(native_decimal)
    assert result.values == tuple(((value,),) for value in exact_values)
    assert result.values != tuple(((value,),) for value in native_values)


def test_eager_exact_ignores_parameters_of_inactive_shared_pack_kernel(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    beta_kernel = _kernel(
        99,
        "vertex",
        (
            _contract("left-current", 0),
            _contract("right-current", 0),
            _parameter_contract("beta", 199),
        ),
    )
    _build_artifact(artifact, extra_kernels=(beta_kernel,))
    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime(),
        kernel_loader=_loader([]),
    )

    result = executor.evaluate_resolved(
        _momenta(5),
        helicities=None,
        color_flows=None,
        precision=50,
    )

    assert result.values
    assert set(executor._plan.kernels) == {10, 11, 12}


def test_eager_exact_ignores_derivation_kernel_without_derived_schema(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    parameter_kernel = _kernel(
        20,
        "model-parameter",
        (_parameter_contract("beta", 199),),
        output_layout=("model-parameter:unused",),
    )
    _build_artifact(artifact, parameter_kernel=parameter_kernel)

    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime(),
        kernel_loader=_loader([]),
    )

    assert executor._plan.parameter_derivation is None
    assert executor._plan.parameter_projection.parameter_count == 0


def test_eager_exact_rejects_multiple_inactive_derivation_kernels(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    first = _kernel(
        20,
        "model-parameter",
        (_parameter_contract("alpha", 7),),
        output_layout=("model-parameter:first",),
    )
    second = _kernel(
        21,
        "model-parameter",
        (_parameter_contract("beta", 9),),
        output_layout=("model-parameter:second",),
    )
    _build_artifact(
        artifact,
        parameter_kernel=first,
        extra_kernels=(second,),
    )

    with pytest.raises(
        ArtifactError,
        match=r"eager kernel pack declares multiple model-parameter kernels",
    ):
        EagerExactExecutor(
            artifact,
            "synthetic",
            _NativeRuntime(),
            kernel_loader=_loader([]),
        )


def test_eager_exact_requires_parameters_of_active_shared_pack_kernel(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    beta_kernel = _kernel(
        99,
        "vertex",
        (
            _contract("left-current", 0),
            _contract("right-current", 0),
            _parameter_contract("beta", 199),
        ),
    )
    _build_artifact(
        artifact,
        invocation_kernel_id=beta_kernel.kernel_id,
        extra_kernels=(beta_kernel,),
    )

    with pytest.raises(
        ArtifactError,
        match=r"prepared parameter 'beta' is absent from the process runtime schema",
    ):
        EagerExactExecutor(
            artifact,
            "synthetic",
            _NativeRuntime(),
            kernel_loader=_loader([]),
        )


def test_eager_exact_validates_inactive_pack_parameter_identity() -> None:
    beta_kernel = _kernel(
        98,
        "vertex",
        (_parameter_contract("beta", 199),),
    )
    gamma_kernel = _kernel(
        99,
        "vertex",
        (_parameter_contract("gamma", 199),),
    )

    with pytest.raises(
        ArtifactError,
        match=r"prepared parameter index 199 names multiple parameters",
    ):
        _prepared_parameter_projection(
            (beta_kernel, gamma_kernel),
            {"model_parameters": []},
            0,
            active_kernel_ids=frozenset(),
        )


def test_eager_exact_preserves_native_derived_values_without_kernel(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    model_parameters = (
        {
            "name": "derived.real",
            "kind": "derived_parameter_component",
            "parameter_index": 0,
            "runtime_name": "derived",
            "complex_component": "real",
            "default": 1.25,
        },
        {
            "name": "derived.imag",
            "kind": "derived_parameter_component",
            "parameter_index": 1,
            "runtime_name": "derived",
            "complex_component": "imag",
            "default": -0.5,
        },
    )
    _build_artifact(
        artifact,
        model_parameters=model_parameters,
        vertex_contracts=(
            _contract("left-current", 0),
            _contract("right-current", 0),
            _contract("coupling-real", 0),
            _contract("coupling-imag", 0),
            _parameter_contract("derived", 3),
        ),
        coupling_row=EagerCouplingRow(0, 1, 0.0, 0.0),
    )
    vertex_inputs: list[tuple[_ComplexDecimal, ...]] = []
    fallback = _loader([])

    def load(record: PreparedKernelRecord, root: Path) -> _ExactCallable:
        if record.kernel_id != 10:
            return fallback(record, root)

        def vertex(
            values: Sequence[_ComplexDecimal], _precision: int
        ) -> tuple[_ComplexDecimal, ...]:
            vertex_inputs.append(tuple(values))
            total = sum((value[0] for value in values), start=Decimal(0))
            return ((total, Decimal(0)),)

        return vertex

    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime((1.25, -0.5)),
        kernel_loader=load,
    )
    executor.evaluate_resolved(
        _momenta(5),
        helicities=None,
        color_flows=None,
        precision=50,
    )

    assert vertex_inputs
    for values in vertex_inputs:
        assert values[2] == (Decimal("1.25"), Decimal(0))
        assert values[3] == (Decimal("-0.5"), Decimal(0))
        assert values[4] == (Decimal("1.25"), Decimal("-0.5"))


@pytest.mark.parametrize(
    "output_layout, match",
    [
        ((), "declares no outputs"),
        (("scalar:0",), "invalid layout"),
        (("model-parameter: derived",), "invalid layout"),
        (
            ("model-parameter:derived", "model-parameter:derived"),
            "repeats output parameter",
        ),
    ],
)
def test_eager_exact_rejects_malformed_parameter_output_layout(
    tmp_path: Path,
    output_layout: tuple[str, ...],
    match: str,
) -> None:
    artifact = tmp_path / "artifact"
    parameter_kernel = _kernel(
        20,
        "model-parameter",
        (_parameter_contract("alpha", 7),),
        output_layout=output_layout,
    )
    _build_artifact(
        artifact,
        parameter_kernel=parameter_kernel,
        model_parameters=(
            {
                "name": "alpha",
                "kind": "external_parameter",
                "parameter_index": 0,
                "default": 2.0,
            },
            {
                "name": "derived.real",
                "kind": "derived_parameter_component",
                "parameter_index": 1,
                "runtime_name": "derived",
                "complex_component": "real",
                "default": 0.0,
            },
            {
                "name": "derived.imag",
                "kind": "derived_parameter_component",
                "parameter_index": 2,
                "runtime_name": "derived",
                "complex_component": "imag",
                "default": 0.0,
            },
        ),
    )
    with pytest.raises(ArtifactError, match=match):
        EagerExactExecutor(
            artifact,
            "synthetic",
            _NativeRuntime((2.0, 0.0, 0.0)),
            kernel_loader=_loader([]),
        )


def test_eager_exact_rejects_missing_runtime_derived_output(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    parameter_kernel = _kernel(
        20,
        "model-parameter",
        (_parameter_contract("alpha", 7),),
        output_layout=("model-parameter:unused",),
    )
    _build_artifact(
        artifact,
        parameter_kernel=parameter_kernel,
        model_parameters=(
            {
                "name": "alpha",
                "kind": "external_parameter",
                "parameter_index": 0,
                "default": 2.0,
            },
            {
                "name": "derived.real",
                "kind": "derived_parameter_component",
                "parameter_index": 1,
                "runtime_name": "derived",
                "complex_component": "real",
                "default": 0.0,
            },
            {
                "name": "derived.imag",
                "kind": "derived_parameter_component",
                "parameter_index": 2,
                "runtime_name": "derived",
                "complex_component": "imag",
                "default": 0.0,
            },
        ),
    )

    with pytest.raises(
        ArtifactError,
        match="does not output runtime derived parameters: 'derived'",
    ):
        EagerExactExecutor(
            artifact,
            "synthetic",
            _NativeRuntime((2.0, 0.0, 0.0)),
            kernel_loader=_loader([]),
        )


def test_eager_exact_rejects_malformed_parameter_input_contract(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    input_contract = _parameter_contract("alpha", 7)
    input_contract["component"] = 1
    parameter_kernel = _kernel(
        20,
        "model-parameter",
        (input_contract,),
        output_layout=("model-parameter:derived",),
    )
    _build_artifact(
        artifact,
        parameter_kernel=parameter_kernel,
        model_parameters=(
            {
                "name": "alpha",
                "kind": "external_parameter",
                "parameter_index": 0,
                "default": 2.0,
            },
            {
                "name": "derived.real",
                "kind": "derived_parameter_component",
                "parameter_index": 1,
                "runtime_name": "derived",
                "complex_component": "real",
                "default": 0.0,
            },
            {
                "name": "derived.imag",
                "kind": "derived_parameter_component",
                "parameter_index": 2,
                "runtime_name": "derived",
                "complex_component": "imag",
                "default": 0.0,
            },
        ),
    )

    with pytest.raises(ArtifactError, match="input 0 has invalid component 1"):
        EagerExactExecutor(
            artifact,
            "synthetic",
            _NativeRuntime((2.0, 0.0, 0.0)),
            kernel_loader=_loader([]),
        )


def test_eager_exact_rejects_parameter_evaluator_output_arity(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    parameter_kernel = _kernel(
        20,
        "model-parameter",
        (_parameter_contract("alpha", 7),),
        output_layout=("model-parameter:derived",),
    )
    _build_artifact(
        artifact,
        parameter_kernel=parameter_kernel,
        model_parameters=(
            {
                "name": "alpha",
                "kind": "external_parameter",
                "parameter_index": 0,
                "default": 2.0,
            },
            {
                "name": "derived.real",
                "kind": "derived_parameter_component",
                "parameter_index": 1,
                "runtime_name": "derived",
                "complex_component": "real",
                "default": 0.0,
            },
            {
                "name": "derived.imag",
                "kind": "derived_parameter_component",
                "parameter_index": 2,
                "runtime_name": "derived",
                "complex_component": "imag",
                "default": 0.0,
            },
        ),
    )
    fallback = _loader([])

    def load(record: PreparedKernelRecord, root: Path) -> _ExactCallable:
        if record.kernel_id == 20:
            return lambda _values, _precision: ()
        return fallback(record, root)

    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime((2.0, 0.0, 0.0)),
        kernel_loader=load,
    )
    with pytest.raises(EvaluationError, match="produced 0 outputs, expected 1"):
        executor.evaluate_resolved(
            _momenta(5),
            helicities=None,
            color_flows=None,
            precision=40,
        )


def test_eager_exact_executes_direct_contraction(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _build_artifact(artifact, direct_closure=True)
    finalization_inputs: list[tuple[_ComplexDecimal, ...]] = []
    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime(),
        kernel_loader=_loader(finalization_inputs),
    )

    result = executor.evaluate_resolved(
        _momenta(5),
        helicities=("h:0",),
        color_flows=("flow:1",),
        precision=40,
    )

    assert result.values == (((Decimal(9025),),),)
    assert len(finalization_inputs) == 1


def test_eager_exact_rejects_malformed_table_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _build_artifact(
        artifact,
        invocation_row_size=EagerInvocationRow._STRUCT.size + 1,
        compact_v3=False,
    )

    with pytest.raises(ArtifactError, match="row size"):
        _load_legacy_plan(artifact, kernel_loader=_loader([]))


def test_eager_exact_rejects_missing_kernel_and_plan_abi(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    _build_artifact(missing, invocation_kernel_id=99)
    with pytest.raises(ArtifactError, match="missing prepared kernel 99"):
        EagerExactExecutor(
            missing,
            "synthetic",
            _NativeRuntime(),
            kernel_loader=_loader([]),
        )

    retired = tmp_path / "retired-v2"
    _build_artifact(retired, compact_v3=False)
    with pytest.raises(
        CompatibilityError,
        match=r"requires a compact plan-v3 artifact.*[Rr]egenerate",
    ):
        EagerExactExecutor(
            retired,
            "synthetic",
            _NativeRuntime(),
            kernel_loader=_loader([]),
        )

    incompatible = tmp_path / "incompatible"
    _build_artifact(
        incompatible,
        plan_abi="future-eager-plan",
        compact_v3=False,
    )
    with pytest.raises(CompatibilityError, match="requires a compact plan-v3 artifact"):
        EagerExactExecutor(
            incompatible,
            "synthetic",
            _NativeRuntime(),
            kernel_loader=_loader([]),
        )

    backend = tmp_path / "backend"
    _build_artifact(backend, prepared_backend="future-backend")
    with pytest.raises(CompatibilityError, match="prepared eager backend"):
        EagerExactExecutor(
            backend,
            "synthetic",
            _NativeRuntime(),
            kernel_loader=_loader([]),
        )

    missing_kernel_abi = tmp_path / "missing-kernel-abi"
    _build_artifact(missing_kernel_abi, kernel_abi=None)
    with pytest.raises(CompatibilityError, match="unsupported eager kernel ABI None"):
        EagerExactExecutor(
            missing_kernel_abi,
            "synthetic",
            _NativeRuntime(),
            kernel_loader=_loader([]),
        )


class _WrongArityEvaluator:
    input_len = 999

    def evaluate(
        self,
        _values: Sequence[_ComplexDecimal],
        _precision: int,
    ) -> tuple[_ComplexDecimal, ...]:
        return ((Decimal(0), Decimal(0)),)


def test_eager_exact_rejects_loaded_evaluator_arity_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _build_artifact(artifact)
    executor = EagerExactExecutor(
        artifact,
        "synthetic",
        _NativeRuntime(),
        kernel_loader=lambda _record, _root: _WrongArityEvaluator(),
    )

    with pytest.raises(ArtifactError, match="input arity 999"):
        executor.evaluate_resolved(
            _momenta(5),
            helicities=None,
            color_flows=None,
            precision=40,
        )
