# SPDX-License-Identifier: 0BSD
"""Conservative, model-independent equality keys for UMAMI representatives.

These keys describe the executed recurrence circuit, not its particle names or
its values at sample points.  Local callable expressions are interned after
binding their inputs; mutable parameter names remain free symbols.  Different
arena layouts can miss an equality, which is preferable to guessing one.
This export-only analysis never modifies the numerical schedule.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .._internal.physics.symbols import ModelSymbolRegistry
from ..api.errors import ArtifactError

if TYPE_CHECKING:
    from ..models.contracts import CompiledModelIR
    from ..models.prepared import PreparedKernelPack, PreparedKernelRecord
    from .recurrence_schedule_sharing import RecurrenceProcessRemap

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:::[A-Za-z_][A-Za-z_0-9]*)*")
_NONE = (1 << 32) - 1


def _key(value: object) -> str:
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), allow_nan=False
    ).iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


class ParameterSemantics:
    """Resolve definitions, never substitute the defaults of external inputs."""

    def __init__(
        self,
        ir: CompiledModelIR | Mapping[str, object],
        catalog_parameters: Sequence[object] = (),
    ):
        payload = ir if isinstance(ir, Mapping) else ir.to_dict()
        self._parameters = {str(p["name"]): p for p in _objects(payload["parameters"])}
        self._couplings = {str(c["name"]): c for c in _objects(payload["couplings"])}
        parameter_names = {
            p.template_id: p.name
            for p in catalog_parameters
            if hasattr(p, "template_id")
        }
        # Contact decomposition can introduce derived couplings not present in
        # the original UFO IR. The catalog already certifies their exact
        # expression and external dependencies; do not compare their defaults.
        self._derived = {
            p.name: (
                p.value_type,
                p.exact_expression_digest,
                tuple(parameter_names[i] for i in p.dependency_parameter_ids),
            )
            for p in catalog_parameters
            if getattr(p, "parameter_kind", None) == "derived"
        }
        registry = ModelSymbolRegistry(str(payload["name"]))
        self._names = {
            spelling: name
            for name in (*self._parameters, *self._couplings, *self._derived)
            for spelling in (name, registry.qualified_name(name))
        }
        self._cache: dict[str, object] = {}
        self._active: set[str] = set()

    def parameter(self, name: str) -> object:
        if name in self._cache:
            return self._cache[name]
        if name in self._active:
            raise ValueError("cyclic symbolic parameter definition")
        self._active.add(name)
        try:
            parameter = self._parameters.get(name)
            coupling = self._couplings.get(name)
            if parameter is not None:
                if parameter["nature"] == "external":
                    value: object = ("external", name, parameter["parameter_type"])
                elif parameter.get("expression") is not None:
                    value = self.expression(
                        str(
                            parameter.get("resolved_expression")
                            or parameter["expression"]
                        )
                    )
                else:
                    value = (
                        "constant",
                        parameter["parameter_type"],
                        parameter.get("value"),
                    )
            elif coupling is not None:
                value = self.expression(
                    str(coupling.get("resolved_expression") or coupling["expression"])
                )
            elif name in self._derived:
                value_type, expression_digest, dependencies = self._derived[name]
                if not expression_digest:
                    raise ValueError("derived parameter has no exact expression")
                value = (
                    "derived-catalog-parameter",
                    value_type,
                    expression_digest,
                    tuple(self.parameter(dependency) for dependency in dependencies),
                )
            else:
                # Runtime-owned parameters retain their actual names.  There
                # is no model-specific interpretation or default-value match.
                value = ("runtime-parameter", name)
            self._cache[name] = value
            return value
        finally:
            self._active.remove(name)

    def expression(
        self, expression: str, bindings: Mapping[str, object] | None = None
    ) -> object:
        local = bindings or {}
        # A prepared input can be a Symbolica function atom such as p(0),
        # rather than an identifier. Match complete declared inputs before
        # ordinary tokens, so their roles/components cannot be lost.
        scanner = (
            re.compile(
                r"(?<![A-Za-z_0-9:])(?:"
                + "|".join(
                    re.escape(name) for name in sorted(local, key=len, reverse=True)
                )
                + r")(?![A-Za-z_0-9:])|"
                + _TOKEN.pattern
            )
            if local
            else _TOKEN
        )
        pieces: list[object] = []
        end = 0
        for match in scanner.finditer(expression):
            pieces.append(expression[end : match.start()])
            token = match.group()
            if token in local:
                pieces.append(("input", local[token]))
            elif token in self._names:
                pieces.append(("parameter", self.parameter(self._names[token])))
            else:
                pieces.append(token)
            end = match.end()
        pieces.append(expression[end:])
        return ("expression", tuple(pieces))


def _kernel_key(
    kernel: PreparedKernelRecord,
    parameters: ParameterSemantics,
    parameter_names: Mapping[int, str],
    coupling: object,
    parent_permutation: tuple[int, int],
) -> str:
    bindings: dict[str, object] = {}
    for contract in kernel.input_contracts:
        role = str(contract["role"])
        component = int(contract["component"])
        if role == "model-parameter":
            name = parameter_names[int(contract["model_parameter_index"])]
            value: object = (role, parameters.parameter(name), component)
        elif role in {"coupling-real", "coupling-imag"}:
            value = (role, coupling, component)
        else:
            value = (role, component)
        bindings[str(contract["symbol"])] = value
    return _key(
        (
            kernel.contract_kind,
            # The ordered expressions define the output coordinates. Display
            # labels may contain auxiliary/contact IDs, which are not algebra.
            len(kernel.output_layout),
            parent_permutation,
            [
                parameters.expression(expr, bindings)
                for expr in kernel.exact_expressions
            ],
        )
    )


def _exact_executor_couplings(pack, exact_ids: Mapping[int, int]) -> dict[int, object]:
    """Bind exact rationals, without Decimal rounding at ambient precision."""
    catalog = pack.recurrence_template_catalog
    records = {
        record.template_id: record
        for record in (*catalog.transitions, *catalog.closures)
    }
    kernels = {record.kernel_id: record for record in pack.kernels}
    result = {}
    for direct in pack.recurrence_direct_template_catalog.templates:
        kernel_id = exact_ids.get(direct.direct_executor_id)
        if kernel_id is None or not any(
            contract["role"] in {"coupling-real", "coupling-imag"}
            for contract in kernels[kernel_id].input_contracts
        ):
            continue
        values = set()
        for template_id in direct.semantic_template_ids:
            value = records[template_id].binding_coupling
            values.add(
                (
                    int(value.real_numerator),
                    int(value.real_denominator),
                    int(value.imag_numerator),
                    int(value.imag_denominator),
                )
            )
        if len(values) != 1:
            raise ValueError("executor has no uniform exact coupling")
        result[direct.direct_executor_id] = next(iter(values))
    return result


def recurrence_structural_keys(
    *,
    exact_sections: Mapping[str, object],
    runtime_metadata: Mapping[str, object],
    kernel_pack: PreparedKernelPack,
    ir: CompiledModelIR | Mapping[str, object],
    physics: Mapping[str, object],
    remap: RecurrenceProcessRemap,
    color_contraction_payload: bytes | None,
) -> dict[str, str]:
    """Prove equality only for fully comparable, unremapped exact circuits.

    The same ordered instruction graph and fully bound local operations imply the
    same amplitudes for every momentum/parameter value.  Colour contraction and
    normalization are included, so equality is of the complete squared observable.
    Unknown plans return no key; ordinary generation/evaluation stays available.
    """
    if exact_sections.get("abi") != "pyamplicol-recurrence-exact-sections-v1":
        return {}
    if physics.get("color_accuracy") != "lc" and color_contraction_payload is None:
        return {}
    if (
        remap.source_slots != tuple(range(len(remap.source_slots)))
        or any(
            v != 1 for v in (*remap.source_momentum_signs, *remap.source_helicity_signs)
        )
        or remap.state_template_changes
        or remap.source_template_changes
        or remap.direct_executor_changes
        or remap.parameter_slot_changes
        or any(
            remap.source_state_indices[start:end] != tuple(range(end - start))
            for start, end in zip(
                remap.source_state_offsets[:-1],
                remap.source_state_offsets[1:],
                strict=True,
            )
        )
        or remap.public_flow_ids != tuple(range(len(remap.public_flow_ids)))
        or remap.physical_sector_ids != tuple(range(len(remap.physical_sector_ids)))
    ):
        return {}
    catalog = kernel_pack.recurrence_template_catalog
    direct_catalog = kernel_pack.recurrence_direct_template_catalog
    if catalog is None or direct_catalog is None:
        return {}
    # Reuse the exact executor's authoritative bindings, including intrinsics
    # whose algebraic kernel is distinct from their native callable payload.
    from ..runtime.recurrence_exact._plan import (
        _executor_exact_kernel_ids,
        _executor_parent_permutations,
    )

    try:
        exact_ids = _executor_exact_kernel_ids(kernel_pack)
        couplings = _exact_executor_couplings(kernel_pack, exact_ids)
        parent_orders = _executor_parent_permutations(kernel_pack)
        parameter_names = {
            p.prepared_parameter_id: p.name
            for p in catalog.parameters
            if p.prepared_parameter_id is not None
        }
        parameters = ParameterSemantics(ir, catalog.parameters)
        kernels = {k.kernel_id: k for k in kernel_pack.kernels}
        # The exported executor catalog can cover the entire prepared model;
        # only referenced operations are part of this process's circuit.
        used_executors = {int(r[3]) for r in _rows(exact_sections["row_groups"])} | {
            int(r[10]) for r in _rows(exact_sections["source_dispatch_variants"])
        }
        used_executors.discard(_NONE)
        executor_keys: dict[int, str] = {}
        for row in _rows(exact_sections["executors"]):
            executor_id = int(row[0])
            if executor_id not in used_executors:
                continue
            kernel_id = exact_ids.get(executor_id)
            expression = (
                _kernel_key(
                    kernels[kernel_id],
                    parameters,
                    parameter_names,
                    couplings.get(executor_id, ()),
                    parent_orders.get(executor_id, (0, 1)),
                )
                if kernel_id is not None
                else ("intrinsic", row[7])
            )
            if kernel_id is None:
                # Source algorithms are specified by the source IR below;
                # diagonal closure reduction uses the retained exact factors
                # and closure rows. The identity finalizer is parameter-free.
                from ..models.recurrence_direct_template import (
                    RECURRENCE_DIRECT_IDENTITY_FINALIZER,
                )

                closure_reduction = row[1] == "closure" and re.fullmatch(
                    r"rusticol\.closure-reduce\.v1:[0-9a-f]{24}", str(row[7])
                )
                if row[7] is None or (
                    row[1] != "source"
                    and not closure_reduction
                    and row[7] != RECURRENCE_DIRECT_IDENTITY_FINALIZER
                ):
                    return {}
                if row[1] == "source":
                    # The suffix identifies a source's full catalog record
                    # (including flavour labels); its executable algorithm is
                    # the prefix and its bound physics is the SourceIR below.
                    # Strip only this documented digest format, not arbitrary
                    # text from an unfamiliar source implementation.
                    source_algorithm = re.fullmatch(
                        r"(rusticol\.source-fill\.[a-z0-9_-]+\.v1):[0-9a-f]{24}",
                        str(row[7]),
                    )
                    if source_algorithm is None:
                        return {}
                    expression = ("intrinsic", source_algorithm[1])
            executor_keys[executor_id] = _key((row[1:6], expression))
        source_keys: dict[int, str] = {}
        for item in _objects(runtime_metadata["source_templates"]):
            source_ir = _mapping(item["source_ir"])
            identity = _mapping(source_ir["identity"])
            mass_name = source_ir.get("mass_parameter")
            source_keys[int(item["source_template_id"])] = _key(
                (
                    item["dimension"],
                    item["helicity"],
                    item["chirality"],
                    item["spin_state"],
                    source_ir["wavefunction_family"],
                    identity["orientation"],
                    item["crossing"],
                    source_ir["statistics"],
                    source_ir["component_dimension"],
                    source_ir["basis"],
                    None
                    if source_ir.get("width_parameter") is None
                    else parameters.parameter(str(source_ir["width_parameter"])),
                    None if mass_name is None else parameters.parameter(str(mass_name)),
                )
            )
        program = {
            k: v
            for k, v in exact_sections.items()
            if k
            not in {
                "process_id",
                "semantic_digest",
                "runtime_layout_digest",
                "executors",
            }
        }
        # Current IDs and state-template IDs are diagnostic; the actual arena
        # ranges, stage clearing, bound source operations and kernels remain.
        program["currents"] = [
            [None, r[1], None, *r[3:]] for r in _rows(program["currents"])
        ]
        program["row_groups"] = [
            [*r[:3], None if r[3] == _NONE else executor_keys[int(r[3])], *r[4:]]
            for r in _rows(program["row_groups"])
        ]
        if program["strategy"] != "all-flow-union":
            program["sources"] = [
                [*r[:3], source_keys[int(r[3])], *r[4:]]
                for r in _rows(program["sources"])
            ]
        program["source_dispatch_variants"] = [
            [
                *r[:6],
                source_keys[int(r[6])],
                None,
                None,
                r[9],
                None if r[10] == _NONE else executor_keys[int(r[10])],
                *r[11:],
            ]
            for r in _rows(program["source_dispatch_variants"])
        ]
        # Keep all normalization dependencies, not just their default number.
        normalization = runtime_metadata["normalization"]
        external_layout = [
            (v["source_slot"], v["is_initial"])
            for v in _objects(runtime_metadata["external_legs"])
        ]
        parameter_projection = [
            (
                v["runtime_slot"],
                parameters.parameter(str(v["runtime_name"])),
                v["component"],
            )
            for v in _objects(runtime_metadata["parameter_projection"])
        ]
        color_key = (
            None
            if color_contraction_payload is None
            else hashlib.sha256(color_contraction_payload).hexdigest()
        )
        circuit = _key(
            (
                program,
                normalization,
                external_layout,
                parameter_projection,
                physics["color_accuracy"],
                color_key,
                [h["values"] for h in _objects(physics["helicities"])],
            )
        )
        return {
            str(c["id"]): _key((circuit, c["index"], c.get("coefficient", 1.0)))
            for c in _objects(physics["color_components"])
        }
    except (ArtifactError, KeyError, TypeError, ValueError, IndexError):
        # Unsupported/missing semantic data cannot authorize grouping.
        return {}


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic record must be a mapping")
    return value


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("semantic records must be a sequence")
    return value


def _rows(value: object) -> tuple[Sequence[object], ...]:
    return tuple(_sequence(v) for v in _sequence(value))


def _objects(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(v) for v in _sequence(value))
