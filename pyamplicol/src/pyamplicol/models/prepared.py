# SPDX-License-Identifier: 0BSD
"""Deterministic prepared-model bundle container contract.

This module owns the portable container format only. Kernel compilation and
backend-specific manifest interpretation belong to the model compiler and
runtime respectively.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

if TYPE_CHECKING:
    from .recurrence_direct_template import RecurrenceDirectTemplateCatalogV1
    from .recurrence_template import RecurrenceTemplateCatalog

PREPARED_MODEL_BUNDLE_KIND = "pyamplicol-prepared-model"
PREPARED_MODEL_BUNDLE_SCHEMA_VERSION = 1
PREPARED_MODEL_BUNDLE_SUFFIX = ".pyamplicol-model"
EAGER_KERNEL_ABI = "pyamplicol-eager-kernel-v1"
EAGER_KERNEL_ABI_VERSION = 1
PREPARED_KERNEL_VARIANT_ABI = "pyamplicol-prepared-kernel-variant-v2"
PREPARED_KERNEL_PACK_IDENTITY_ABI = "pyamplicol-prepared-kernel-pack-identity-v2"
PREPARED_INDEPENDENT_BLOCK_SIZE = 4
PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL = 2
PREPARED_JIT_PORTABLE_TARGET = "symjit-storage-v3-portable"
PREPARED_SYMJIT_PLANE_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"

PREPARED_MODEL_MANIFEST_PATH = "manifest.json"
PREPARED_MODEL_COMPILED_MODEL_PATH = "model/model.pyAmplicol-model.json"

PreparedBackend: TypeAlias = Literal["jit", "asm", "cpp"]
KernelContractKind: TypeAlias = Literal[
    "vertex",
    "propagator",
    "closure",
    "model-parameter",
]
PayloadSource: TypeAlias = bytes | bytearray | memoryview | Path
PreparedKernelVariantKind: TypeAlias = Literal["independent-block"]
PreparedKernelLaneLayout: TypeAlias = Literal["lane-major"]

_BACKENDS = frozenset(("jit", "asm", "cpp"))
_CONTRACT_KINDS = frozenset(("vertex", "propagator", "closure", "model-parameter"))
_PATH_FIELDS = frozenset(
    (
        "application_path",
        "descriptor_path",
        "evaluator_state_path",
        "library_path",
        "payload_path",
        "source_path",
    )
)
_PATH_LIST_FIELDS = frozenset(("payload_paths",))
_SHA256_LENGTH = 64
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = (stat.S_IFREG | 0o644) << 16


class PreparedModelBundleError(ValueError):
    """Raised when a prepared-model bundle violates its container contract."""


def _freeze_json(value: object, context: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PreparedModelBundleError(
                f"{context} must not contain NaN or infinity"
            )
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PreparedModelBundleError(f"{context} keys must be strings")
            frozen[key] = _freeze_json(item, f"{context}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise PreparedModelBundleError(
        f"{context} contains unsupported value {type(value).__name__}"
    )


def _freeze_mapping(value: Mapping[str, object], context: str) -> Mapping[str, object]:
    frozen = _freeze_json(value, context)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by annotation
        raise PreparedModelBundleError(f"{context} must be an object")
    return cast(Mapping[str, object], frozen)


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PreparedModelBundleError(f"{context} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PreparedModelBundleError(f"{context} must be an array")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    context: str,
    expected: frozenset[str],
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise PreparedModelBundleError(
            f"{context} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise PreparedModelBundleError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PreparedModelBundleError(f"{context} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreparedModelBundleError(f"{context} must be an integer >= 0")
    return value


def _positive_integer(value: object, context: str) -> int:
    result = _nonnegative_integer(value, context)
    if result == 0:
        raise PreparedModelBundleError(f"{context} must be an integer >= 1")
    return result


def _validate_portable_jit_manifest(
    value: Mapping[str, object],
    context: str,
) -> None:
    if value.get("kind") != "symjit-application-evaluator":
        raise PreparedModelBundleError(
            f"prepared JIT {context} must contain a SymJIT application evaluator"
        )
    level = value.get("optimization_level")
    if level != PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL:
        raise PreparedModelBundleError(
            f"prepared JIT {context} must use portable SymJIT optimization "
            f"level {PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL}, found {level!r}"
        )
    settings = _mapping(value.get("settings"), f"prepared JIT {context}.settings")
    if (
        settings.get("jit_optimization_level")
        != PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
    ):
        raise PreparedModelBundleError(
            f"prepared JIT {context} settings do not attest portable SymJIT "
            f"optimization level {PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL}"
        )


def _normalized_member_path(value: object, context: str) -> str:
    path = _nonempty_string(value, context)
    if "\\" in path or "\x00" in path:
        raise PreparedModelBundleError(
            f"{context} must be a normalized relative POSIX path"
        )
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or path.endswith("/")
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PreparedModelBundleError(
            f"{context} must be a normalized relative POSIX path"
        )
    return path


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    try:
        serialized = json.dumps(
            _thaw_json(payload),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PreparedModelBundleError(
            f"payload is not canonical JSON: {error}"
        ) from error
    return (serialized + "\n").encode("ascii")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: object, context: str) -> str:
    digest = _nonempty_string(value, context)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PreparedModelBundleError(f"{context} must be a lowercase SHA-256")
    return digest


def _mapping_digest(value: Mapping[str, object]) -> str:
    return _sha256(_canonical_json(value))


def prepared_kernel_pack_manifest_identity_sha256(
    manifest: Mapping[str, object],
) -> str:
    """Return the outer identity of one canonical prepared-bundle manifest.

    This is deliberately distinct from :func:`prepared_kernel_pack_identity`,
    which authenticates the callable kernel contract and referenced payloads.
    Artifact manifests use this identity to name the exact prepared bundle from
    which their copied evaluator pack was derived.
    """

    normalized = _freeze_mapping(manifest, "prepared bundle manifest identity")
    canonical_manifest = json.dumps(
        _thaw_json(normalized),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256(
        PREPARED_KERNEL_PACK_IDENTITY_ABI.encode("ascii")
        + b"\x00"
        + canonical_manifest
    )


def prepared_expression_digest(expressions: Sequence[str]) -> str:
    """Digest the ordered exact-expression contract of a scalar kernel."""

    return _mapping_digest({"exact_expressions": list(expressions)})


def prepared_input_contract_digest(
    input_layout: Sequence[str],
    input_contracts: Sequence[Mapping[str, object]],
) -> str:
    """Digest scalar input ordering, roles, symbols, and parameter metadata."""

    return _mapping_digest(
        {
            "input_arity": len(input_layout),
            "input_layout": list(input_layout),
            "input_contracts": [
                cast(dict[str, object], _thaw_json(contract))
                for contract in input_contracts
            ],
        }
    )


def prepared_output_contract_digest(output_layout: Sequence[str]) -> str:
    """Digest scalar output ordering and labels."""

    return _mapping_digest(
        {
            "output_arity": len(output_layout),
            "output_layout": list(output_layout),
        }
    )


def prepared_optimization_settings_digest(
    settings: Mapping[str, object],
) -> str:
    """Digest the backend optimization contract bound to a variant payload."""

    return _mapping_digest(settings)


def _recurrence_input_contract_layout(
    contracts: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return the canonical input contract used by recurrence bindings."""

    return tuple(
        json.dumps(
            cast(dict[str, object], _thaw_json(contract)),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for contract in contracts
    )


def _validate_recurrence_kernel_bindings(
    catalog: RecurrenceTemplateCatalog,
    kernels: Sequence[PreparedKernelRecord],
) -> None:
    """Authenticate every recurrence callable against its prepared kernel."""

    by_id = {kernel.kernel_id: kernel for kernel in kernels}
    for binding in catalog.evaluator_bindings:
        if binding.callable_kind != "prepared-kernel":
            continue
        kernel_id = binding.prepared_kernel_id
        assert kernel_id is not None  # EvaluatorBindingV1 invariant.
        kernel = by_id.get(kernel_id)
        if kernel is None:
            raise PreparedModelBundleError(
                "recurrence evaluator binding references unknown prepared kernel "
                f"ID {kernel_id}"
            )
        context = f"recurrence evaluator {binding.resolver_key!r}"
        expected = {
            "contract kind": kernel.contract_kind,
            "callable signature": kernel.canonical_signature,
            "input layout": _recurrence_input_contract_layout(kernel.input_contracts),
            "output layout": kernel.output_layout,
            "exact expression digests": tuple(
                hashlib.sha256(expression.encode("utf-8")).hexdigest()
                for expression in kernel.exact_expressions
            ),
        }
        actual = {
            "contract kind": binding.contract_kind,
            "callable signature": binding.callable_signature,
            "input layout": binding.input_layout,
            "output layout": binding.output_layout,
            "exact expression digests": binding.exact_expression_digests,
        }
        for name, expected_value in expected.items():
            if actual[name] != expected_value:
                raise PreparedModelBundleError(
                    f"{context} {name} does not match prepared kernel {kernel_id}"
                )


def _validate_recurrence_direct_bindings(
    direct_catalog: RecurrenceDirectTemplateCatalogV1,
    semantic_catalog: RecurrenceTemplateCatalog,
    kernels: Sequence[PreparedKernelRecord],
    *,
    backend: PreparedBackend,
    target: Mapping[str, object],
    optimization_settings: Mapping[str, object],
) -> None:
    """Authenticate stable semantic-binding to direct-executor mappings."""

    if (
        direct_catalog.compiled_model_digest
        != semantic_catalog.header.compiled_model_digest
        or direct_catalog.recurrence_template_catalog_digest
        != semantic_catalog.catalog_digest
        or direct_catalog.prepared_kernel_pack_digest
        != semantic_catalog.header.prepared_kernel_pack_digest
    ):
        raise PreparedModelBundleError(
            "direct recurrence companion does not match its semantic catalog"
        )
    if (
        direct_catalog.backend != backend
        or direct_catalog.target_triple != target.get("target_triple")
        or direct_catalog.portable != target.get("portable")
        or direct_catalog.optimization_settings_digest
        != prepared_optimization_settings_digest(optimization_settings)
    ):
        raise PreparedModelBundleError(
            "direct recurrence companion backend/target provenance does not "
            "match its prepared pack"
        )

    role_for_contract = {
        "source": "source",
        "vertex": "contribution",
        "propagator": "finalization",
        "closure": "closure",
    }
    by_kernel_id = {kernel.kernel_id: kernel for kernel in kernels}
    required: set[tuple[str, int]] = set()
    for evaluator_binding_id, binding in enumerate(semantic_catalog.evaluator_bindings):
        role = role_for_contract.get(binding.contract_kind)
        if role is None:
            continue
        required.add((role, evaluator_binding_id))
        try:
            executor_id = direct_catalog.direct_executor_id_for(  # type: ignore[arg-type]
                role,
                evaluator_binding_id,
            )
        except ValueError as exc:
            raise PreparedModelBundleError(str(exc)) from exc
        template = direct_catalog.templates[executor_id]
        if (
            template.evaluator_resolver_key != binding.resolver_key
            or template.semantic_template_ids != binding.semantic_template_ids
        ):
            raise PreparedModelBundleError(
                "direct recurrence executor mapping does not match semantic "
                f"evaluator binding {evaluator_binding_id}"
            )
        payload_kernel_id = template.payload_binding.prepared_kernel_id
        if binding.callable_kind == "prepared-kernel":
            prepared_payload_matches = (
                template.payload_binding.kind == "prepared-direct-call"
                and payload_kernel_id == binding.prepared_kernel_id
                and payload_kernel_id in by_kernel_id
            )
            certified_intrinsic_matches = (
                template.role == "contribution"
                and template.payload_binding.kind == "rusticol-intrinsic"
                and template.payload_binding.intrinsic_contract_digest is not None
                and payload_kernel_id is None
            )
            finalization_intrinsic_matches = (
                template.role == "finalization"
                and template.payload_binding.kind == "rusticol-intrinsic"
                and template.payload_binding.runtime_template is not None
                and payload_kernel_id is None
            )
            if not (
                prepared_payload_matches
                or certified_intrinsic_matches
                or finalization_intrinsic_matches
            ):
                raise PreparedModelBundleError(
                    "direct recurrence prepared payload does not match semantic "
                    f"evaluator binding {evaluator_binding_id}"
                )
        elif (
            template.payload_binding.kind != "rusticol-intrinsic"
            or template.payload_binding.runtime_template != binding.runtime_template
        ):
            raise PreparedModelBundleError(
                "direct recurrence intrinsic does not match semantic evaluator "
                f"binding {evaluator_binding_id}"
            )

    identity_propagators = {
        propagator.template_id
        for propagator in semantic_catalog.propagators
        if not propagator.applies_propagator
    }
    mapped_identity: set[str] = set()
    semantic_binding_count = len(semantic_catalog.evaluator_bindings)
    for template in direct_catalog.templates:
        key = (template.role, template.evaluator_binding_id)
        if template.evaluator_binding_id < semantic_binding_count:
            if key not in required:
                raise PreparedModelBundleError(
                    "direct recurrence companion exposes an unexpected semantic "
                    f"evaluator mapping {key!r}"
                )
            continue
        if (
            template.role != "finalization"
            or template.payload_binding.kind != "rusticol-intrinsic"
            or not template.payload_binding.runtime_template
            or template.payload_binding.runtime_template
            != "rusticol.identity-finalize-in-place.v1"
            or not set(template.semantic_template_ids) <= identity_propagators
        ):
            raise PreparedModelBundleError(
                "direct recurrence synthetic evaluator is not an authenticated "
                "identity finalizer"
            )
        mapped_identity.update(template.semantic_template_ids)
    if mapped_identity != identity_propagators:
        raise PreparedModelBundleError(
            "direct recurrence companion does not cover every identity finalizer"
        )


def prepared_compiled_model_digest(compiled_model: Mapping[str, object]) -> str:
    """Digest the exact canonical compiled-model member stored in a bundle."""

    frozen = _freeze_mapping(compiled_model, "compiled_model")
    return _sha256(_canonical_json(frozen))


def _collect_manifest_paths(value: object, *, context: str) -> tuple[str, ...]:
    paths: list[str] = []

    def visit(item: object, item_context: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in _PATH_FIELDS:
                    if child is not None:
                        paths.append(
                            _normalized_member_path(child, f"{item_context}.{key}")
                        )
                elif key in _PATH_LIST_FIELDS:
                    for index, path in enumerate(
                        _sequence(child, f"{item_context}.{key}")
                    ):
                        paths.append(
                            _normalized_member_path(
                                path,
                                f"{item_context}.{key}[{index}]",
                            )
                        )
                else:
                    visit(child, f"{item_context}.{key}")
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for index, child in enumerate(item):
                visit(child, f"{item_context}[{index}]")

    visit(value, context)
    return tuple(sorted(set(paths)))


@dataclass(frozen=True, slots=True)
class PreparedKernelVariantRecord:
    """One backend evaluator that packs independent scalar kernel calls."""

    variant_id: str
    variant_abi: str
    kind: PreparedKernelVariantKind
    block_size: int
    lane_layout: PreparedKernelLaneLayout
    base_kernel_id: int
    base_canonical_signature: str
    base_expression_digest: str
    base_input_contract_digest: str
    base_output_contract_digest: str
    backend: PreparedBackend
    optimization_settings_digest: str
    input_arity: int
    output_arity: int
    input_lane_stride: int
    output_lane_stride: int
    input_layout: tuple[str, ...]
    output_layout: tuple[str, ...]
    f64_evaluator_manifest: Mapping[str, object]
    _referenced_payload_paths: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _nonempty_string(self.variant_id, "kernel_variant.variant_id")
        if self.variant_abi != PREPARED_KERNEL_VARIANT_ABI:
            raise PreparedModelBundleError(
                f"unsupported prepared kernel variant ABI {self.variant_abi!r}; "
                "regenerate the prepared model with this pyAmpliCol version"
            )
        if self.kind != "independent-block":
            raise PreparedModelBundleError(
                f"unsupported prepared kernel variant kind {self.kind!r}"
            )
        block_size = _positive_integer(
            self.block_size,
            "kernel_variant.block_size",
        )
        if block_size != PREPARED_INDEPENDENT_BLOCK_SIZE:
            raise PreparedModelBundleError(
                "independent-block variants currently require block_size = 4"
            )
        if self.variant_id != f"independent-block-{block_size}":
            raise PreparedModelBundleError(
                "independent-block variant ID does not match its block size"
            )
        if self.lane_layout != "lane-major":
            raise PreparedModelBundleError(
                "independent-block variants require lane-major layout"
            )
        _nonnegative_integer(
            self.base_kernel_id,
            "kernel_variant.base_kernel_id",
        )
        _valid_sha256(
            self.base_canonical_signature,
            "kernel_variant.base_canonical_signature",
        )
        for name in (
            "base_expression_digest",
            "base_input_contract_digest",
            "base_output_contract_digest",
            "optimization_settings_digest",
        ):
            _valid_sha256(getattr(self, name), f"kernel_variant.{name}")
        if self.backend not in _BACKENDS:
            raise PreparedModelBundleError(
                f"unsupported prepared variant backend {self.backend!r}"
            )
        input_arity = _positive_integer(
            self.input_arity,
            "kernel_variant.input_arity",
        )
        output_arity = _positive_integer(
            self.output_arity,
            "kernel_variant.output_arity",
        )
        input_lane_stride = _positive_integer(
            self.input_lane_stride,
            "kernel_variant.input_lane_stride",
        )
        output_lane_stride = _positive_integer(
            self.output_lane_stride,
            "kernel_variant.output_lane_stride",
        )
        if input_arity != block_size * input_lane_stride:
            raise PreparedModelBundleError(
                "kernel variant input arity must equal block_size * input_lane_stride"
            )
        if output_arity != block_size * output_lane_stride:
            raise PreparedModelBundleError(
                "kernel variant output arity must equal block_size * output_lane_stride"
            )
        input_layout = tuple(
            _nonempty_string(item, f"kernel_variant.input_layout[{index}]")
            for index, item in enumerate(self.input_layout)
        )
        output_layout = tuple(
            _nonempty_string(item, f"kernel_variant.output_layout[{index}]")
            for index, item in enumerate(self.output_layout)
        )
        if len(input_layout) != input_arity:
            raise PreparedModelBundleError(
                "kernel variant input layout length must equal input arity"
            )
        if len(output_layout) != output_arity:
            raise PreparedModelBundleError(
                "kernel variant output layout length must equal output arity"
            )
        object.__setattr__(self, "input_layout", input_layout)
        object.__setattr__(self, "output_layout", output_layout)
        if not self.f64_evaluator_manifest:
            raise PreparedModelBundleError(
                "kernel_variant.f64_evaluator_manifest must not be empty"
            )
        manifest = _freeze_mapping(
            self.f64_evaluator_manifest,
            "kernel_variant.f64_evaluator_manifest",
        )
        if manifest.get("input_len") != input_arity:
            raise PreparedModelBundleError(
                "kernel variant evaluator input_len does not match input arity"
            )
        if manifest.get("output_len") != output_arity:
            raise PreparedModelBundleError(
                "kernel variant evaluator output_len does not match output arity"
            )
        object.__setattr__(self, "f64_evaluator_manifest", manifest)
        object.__setattr__(
            self,
            "_referenced_payload_paths",
            _collect_manifest_paths(
                manifest,
                context="kernel_variant.f64_evaluator_manifest",
            ),
        )

    @property
    def referenced_payload_paths(self) -> tuple[str, ...]:
        return self._referenced_payload_paths

    def validate_base_kernel(self, kernel: PreparedKernelRecord) -> None:
        """Validate every immutable binding back to the scalar kernel."""

        if self.base_kernel_id != kernel.kernel_id:
            raise PreparedModelBundleError(
                "kernel variant base_kernel_id does not match its scalar kernel"
            )
        if self.base_canonical_signature != kernel.canonical_signature:
            raise PreparedModelBundleError(
                "kernel variant canonical signature does not match its scalar kernel"
            )
        expected = (
            prepared_expression_digest(kernel.exact_expressions),
            prepared_input_contract_digest(
                kernel.input_layout,
                kernel.input_contracts,
            ),
            prepared_output_contract_digest(kernel.output_layout),
        )
        actual = (
            self.base_expression_digest,
            self.base_input_contract_digest,
            self.base_output_contract_digest,
        )
        if actual != expected:
            raise PreparedModelBundleError(
                "kernel variant scalar contract digest does not match its base kernel"
            )
        if self.input_lane_stride != kernel.input_arity:
            raise PreparedModelBundleError(
                "kernel variant input lane stride does not match its base kernel"
            )
        if self.output_lane_stride != kernel.output_arity:
            raise PreparedModelBundleError(
                "kernel variant output lane stride does not match its base kernel"
            )
        expected_inputs = tuple(
            f"lane:{lane}:{item}"
            for lane in range(self.block_size)
            for item in kernel.input_layout
        )
        expected_outputs = tuple(
            f"lane:{lane}:{item}"
            for lane in range(self.block_size)
            for item in kernel.output_layout
        )
        if (
            self.input_layout != expected_inputs
            or self.output_layout != expected_outputs
        ):
            raise PreparedModelBundleError(
                "kernel variant lane-major layout does not match its base kernel"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "variant_abi": self.variant_abi,
            "kind": self.kind,
            "block_size": self.block_size,
            "lane_layout": self.lane_layout,
            "base_kernel_id": self.base_kernel_id,
            "base_canonical_signature": self.base_canonical_signature,
            "base_expression_digest": self.base_expression_digest,
            "base_input_contract_digest": self.base_input_contract_digest,
            "base_output_contract_digest": self.base_output_contract_digest,
            "backend": self.backend,
            "optimization_settings_digest": self.optimization_settings_digest,
            "input_arity": self.input_arity,
            "output_arity": self.output_arity,
            "input_lane_stride": self.input_lane_stride,
            "output_lane_stride": self.output_lane_stride,
            "input_layout": list(self.input_layout),
            "output_layout": list(self.output_layout),
            "f64_evaluator_manifest": _thaw_json(self.f64_evaluator_manifest),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PreparedKernelVariantRecord:
        expected = frozenset(
            (
                "variant_id",
                "variant_abi",
                "kind",
                "block_size",
                "lane_layout",
                "base_kernel_id",
                "base_canonical_signature",
                "base_expression_digest",
                "base_input_contract_digest",
                "base_output_contract_digest",
                "backend",
                "optimization_settings_digest",
                "input_arity",
                "output_arity",
                "input_lane_stride",
                "output_lane_stride",
                "input_layout",
                "output_layout",
                "f64_evaluator_manifest",
            )
        )
        _require_exact_keys(value, "kernel_variant", expected)
        return cls(
            variant_id=_nonempty_string(
                value.get("variant_id"), "kernel_variant.variant_id"
            ),
            variant_abi=_nonempty_string(
                value.get("variant_abi"), "kernel_variant.variant_abi"
            ),
            kind=cast(
                PreparedKernelVariantKind,
                _nonempty_string(value.get("kind"), "kernel_variant.kind"),
            ),
            block_size=_positive_integer(
                value.get("block_size"), "kernel_variant.block_size"
            ),
            lane_layout=cast(
                PreparedKernelLaneLayout,
                _nonempty_string(
                    value.get("lane_layout"), "kernel_variant.lane_layout"
                ),
            ),
            base_kernel_id=_nonnegative_integer(
                value.get("base_kernel_id"), "kernel_variant.base_kernel_id"
            ),
            base_canonical_signature=_nonempty_string(
                value.get("base_canonical_signature"),
                "kernel_variant.base_canonical_signature",
            ),
            base_expression_digest=_nonempty_string(
                value.get("base_expression_digest"),
                "kernel_variant.base_expression_digest",
            ),
            base_input_contract_digest=_nonempty_string(
                value.get("base_input_contract_digest"),
                "kernel_variant.base_input_contract_digest",
            ),
            base_output_contract_digest=_nonempty_string(
                value.get("base_output_contract_digest"),
                "kernel_variant.base_output_contract_digest",
            ),
            backend=cast(
                PreparedBackend,
                _nonempty_string(value.get("backend"), "kernel_variant.backend"),
            ),
            optimization_settings_digest=_nonempty_string(
                value.get("optimization_settings_digest"),
                "kernel_variant.optimization_settings_digest",
            ),
            input_arity=_positive_integer(
                value.get("input_arity"), "kernel_variant.input_arity"
            ),
            output_arity=_positive_integer(
                value.get("output_arity"), "kernel_variant.output_arity"
            ),
            input_lane_stride=_positive_integer(
                value.get("input_lane_stride"),
                "kernel_variant.input_lane_stride",
            ),
            output_lane_stride=_positive_integer(
                value.get("output_lane_stride"),
                "kernel_variant.output_lane_stride",
            ),
            input_layout=tuple(
                _nonempty_string(item, f"kernel_variant.input_layout[{index}]")
                for index, item in enumerate(
                    _sequence(value.get("input_layout"), "kernel_variant.input_layout")
                )
            ),
            output_layout=tuple(
                _nonempty_string(item, f"kernel_variant.output_layout[{index}]")
                for index, item in enumerate(
                    _sequence(
                        value.get("output_layout"), "kernel_variant.output_layout"
                    )
                )
            ),
            f64_evaluator_manifest=_mapping(
                value.get("f64_evaluator_manifest"),
                "kernel_variant.f64_evaluator_manifest",
            ),
        )


@dataclass(frozen=True, slots=True)
class PreparedKernelRecord:
    """One canonical eager kernel and its exact/f64 evaluator contracts."""

    kernel_id: int
    contract_kind: KernelContractKind
    canonical_signature: str
    input_arity: int
    output_arity: int
    input_layout: tuple[str, ...]
    input_contracts: tuple[Mapping[str, object], ...]
    output_layout: tuple[str, ...]
    exact_expressions: tuple[str, ...]
    exact_evaluator_state_path: str
    f64_evaluator_manifest: Mapping[str, object]
    proof_classes: tuple[str, ...] = ()
    _referenced_payload_paths: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kernel_id",
            _nonnegative_integer(self.kernel_id, "kernel.kernel_id"),
        )
        if self.contract_kind not in _CONTRACT_KINDS:
            raise PreparedModelBundleError(
                f"unsupported kernel contract kind {self.contract_kind!r}"
            )
        _nonempty_string(self.canonical_signature, "kernel.canonical_signature")
        input_arity = _nonnegative_integer(self.input_arity, "kernel.input_arity")
        output_arity = _nonnegative_integer(self.output_arity, "kernel.output_arity")
        input_layout = tuple(
            _nonempty_string(item, f"kernel.input_layout[{index}]")
            for index, item in enumerate(self.input_layout)
        )
        output_layout = tuple(
            _nonempty_string(item, f"kernel.output_layout[{index}]")
            for index, item in enumerate(self.output_layout)
        )
        if len(input_layout) != input_arity:
            raise PreparedModelBundleError(
                "kernel input layout length must equal input arity"
            )
        if len(output_layout) != output_arity:
            raise PreparedModelBundleError(
                "kernel output layout length must equal output arity"
            )
        object.__setattr__(self, "input_layout", input_layout)
        object.__setattr__(self, "output_layout", output_layout)
        input_contracts = tuple(
            _freeze_mapping(item, f"kernel.input_contracts[{index}]")
            for index, item in enumerate(self.input_contracts)
        )
        if len(input_contracts) != input_arity:
            raise PreparedModelBundleError(
                "kernel input contract count must equal input arity"
            )
        for index, contract in enumerate(input_contracts):
            _require_exact_keys(
                contract,
                f"kernel.input_contracts[{index}]",
                frozenset(
                    (
                        "role",
                        "component",
                        "symbol",
                        "model_parameter_name",
                        "model_parameter_index",
                    )
                ),
            )
            _nonempty_string(
                contract.get("role"), f"kernel.input_contracts[{index}].role"
            )
            _nonnegative_integer(
                contract.get("component"),
                f"kernel.input_contracts[{index}].component",
            )
            _nonempty_string(
                contract.get("symbol"),
                f"kernel.input_contracts[{index}].symbol",
            )
        object.__setattr__(self, "input_contracts", input_contracts)
        exact_expressions = tuple(
            _nonempty_string(item, f"kernel.exact_expressions[{index}]")
            for index, item in enumerate(self.exact_expressions)
        )
        if len(exact_expressions) != output_arity:
            raise PreparedModelBundleError(
                "kernel exact expression count must equal output arity"
            )
        object.__setattr__(self, "exact_expressions", exact_expressions)
        object.__setattr__(
            self,
            "exact_evaluator_state_path",
            _normalized_member_path(
                self.exact_evaluator_state_path,
                "kernel.exact_evaluator_state_path",
            ),
        )
        if not self.f64_evaluator_manifest:
            raise PreparedModelBundleError(
                "kernel.f64_evaluator_manifest must not be empty"
            )
        object.__setattr__(
            self,
            "f64_evaluator_manifest",
            _freeze_mapping(
                self.f64_evaluator_manifest,
                "kernel.f64_evaluator_manifest",
            ),
        )
        proof_classes = tuple(
            _nonempty_string(item, f"kernel.proof_classes[{index}]")
            for index, item in enumerate(self.proof_classes)
        )
        if proof_classes != tuple(sorted(set(proof_classes))):
            raise PreparedModelBundleError(
                "kernel proof classes must be sorted and unique"
            )
        object.__setattr__(self, "proof_classes", proof_classes)
        object.__setattr__(
            self,
            "_referenced_payload_paths",
            tuple(
                sorted(
                    {
                        self.exact_evaluator_state_path,
                        *_collect_manifest_paths(
                            self.f64_evaluator_manifest,
                            context="kernel.f64_evaluator_manifest",
                        ),
                    }
                )
            ),
        )

    @property
    def referenced_payload_paths(self) -> tuple[str, ...]:
        return self._referenced_payload_paths

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "contract_kind": self.contract_kind,
            "canonical_signature": self.canonical_signature,
            "input_arity": self.input_arity,
            "output_arity": self.output_arity,
            "input_layout": list(self.input_layout),
            "input_contracts": [
                _thaw_json(contract) for contract in self.input_contracts
            ],
            "output_layout": list(self.output_layout),
            "exact_expressions": list(self.exact_expressions),
            "proof_classes": list(self.proof_classes),
            "exact_evaluator_state_path": self.exact_evaluator_state_path,
            "f64_evaluator_manifest": _thaw_json(self.f64_evaluator_manifest),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PreparedKernelRecord:
        normalized = dict(value)
        expected = frozenset(
            (
                "kernel_id",
                "contract_kind",
                "canonical_signature",
                "input_arity",
                "output_arity",
                "input_layout",
                "input_contracts",
                "output_layout",
                "exact_expressions",
                "proof_classes",
                "exact_evaluator_state_path",
                "f64_evaluator_manifest",
            )
        )
        _require_exact_keys(normalized, "kernel", expected)
        contract_kind = _nonempty_string(
            normalized.get("contract_kind"),
            "kernel.contract_kind",
        )
        return cls(
            kernel_id=_nonnegative_integer(
                normalized.get("kernel_id"), "kernel.kernel_id"
            ),
            contract_kind=cast(KernelContractKind, contract_kind),
            canonical_signature=_nonempty_string(
                normalized.get("canonical_signature"),
                "kernel.canonical_signature",
            ),
            input_arity=_nonnegative_integer(
                normalized.get("input_arity"), "kernel.input_arity"
            ),
            output_arity=_nonnegative_integer(
                normalized.get("output_arity"), "kernel.output_arity"
            ),
            input_layout=tuple(
                _nonempty_string(item, f"kernel.input_layout[{index}]")
                for index, item in enumerate(
                    _sequence(normalized.get("input_layout"), "kernel.input_layout")
                )
            ),
            input_contracts=tuple(
                _mapping(item, f"kernel.input_contracts[{index}]")
                for index, item in enumerate(
                    _sequence(
                        normalized.get("input_contracts"), "kernel.input_contracts"
                    )
                )
            ),
            output_layout=tuple(
                _nonempty_string(item, f"kernel.output_layout[{index}]")
                for index, item in enumerate(
                    _sequence(normalized.get("output_layout"), "kernel.output_layout")
                )
            ),
            exact_expressions=tuple(
                _nonempty_string(item, f"kernel.exact_expressions[{index}]")
                for index, item in enumerate(
                    _sequence(
                        normalized.get("exact_expressions"),
                        "kernel.exact_expressions",
                    )
                )
            ),
            proof_classes=tuple(
                _nonempty_string(item, f"kernel.proof_classes[{index}]")
                for index, item in enumerate(
                    _sequence(normalized.get("proof_classes"), "kernel.proof_classes")
                )
            ),
            exact_evaluator_state_path=_normalized_member_path(
                normalized.get("exact_evaluator_state_path"),
                "kernel.exact_evaluator_state_path",
            ),
            f64_evaluator_manifest=_mapping(
                normalized.get("f64_evaluator_manifest"),
                "kernel.f64_evaluator_manifest",
            ),
        )


@dataclass(frozen=True, slots=True)
class PreparedKernelPack:
    """The single prepared backend pack embedded in a model bundle."""

    backend: PreparedBackend
    optimization_settings: Mapping[str, object]
    producer: Mapping[str, object]
    dependency_abis: Mapping[str, object]
    provenance: Mapping[str, object]
    target: Mapping[str, object]
    resolver_manifest: Mapping[str, object]
    kernels: tuple[PreparedKernelRecord, ...]
    kernel_variants: tuple[PreparedKernelVariantRecord, ...] = ()
    recurrence_template: Mapping[str, object] | None = None
    recurrence_direct_template: Mapping[str, object] | None = None
    _referenced_payload_paths: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _recurrence_direct_template_catalog: RecurrenceDirectTemplateCatalogV1 | None = (
        field(
            init=False,
            repr=False,
            compare=False,
            default=None,
        )
    )

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise PreparedModelBundleError(
                f"unsupported prepared backend {self.backend!r}"
            )
        for name in (
            "optimization_settings",
            "producer",
            "dependency_abis",
            "provenance",
            "target",
            "resolver_manifest",
        ):
            value = cast(Mapping[str, object], getattr(self, name))
            if not value:
                raise PreparedModelBundleError(f"kernel_pack.{name} must not be empty")
            object.__setattr__(
                self,
                name,
                _freeze_mapping(value, f"kernel_pack.{name}"),
            )
        target = self.target
        target_fields = frozenset(
            ("portable", "word_bits", "endianness", "target_triple", "cpu_features")
        )
        _require_exact_keys(target, "kernel_pack.target", target_fields)
        if not isinstance(target.get("portable"), bool):
            raise PreparedModelBundleError(
                "kernel_pack.target.portable must be a boolean"
            )
        _positive_integer(target.get("word_bits"), "kernel_pack.target.word_bits")
        if target.get("endianness") not in {"little", "big"}:
            raise PreparedModelBundleError(
                "kernel_pack.target.endianness must be 'little' or 'big'"
            )
        _nonempty_string(
            target.get("target_triple"),
            "kernel_pack.target.target_triple",
        )
        cpu_features = tuple(
            _nonempty_string(item, f"kernel_pack.target.cpu_features[{index}]")
            for index, item in enumerate(
                _sequence(
                    target.get("cpu_features"),
                    "kernel_pack.target.cpu_features",
                )
            )
        )
        if cpu_features != tuple(sorted(set(cpu_features))):
            raise PreparedModelBundleError(
                "kernel_pack.target.cpu_features must be sorted and unique"
            )
        if self.backend == "jit":
            if (
                self.dependency_abis.get("symjit_plane_application")
                != PREPARED_SYMJIT_PLANE_APPLICATION_ABI
            ):
                raise PreparedModelBundleError(
                    "prepared JIT packs must authenticate SymJIT plane-application "
                    f"ABI {PREPARED_SYMJIT_PLANE_APPLICATION_ABI!r}; regenerate "
                    "the prepared model"
                )
            level = self.optimization_settings.get("jit_optimization_level")
            if level != PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL:
                raise PreparedModelBundleError(
                    "prepared JIT packs must use portable SymJIT optimization "
                    f"level {PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL}, found "
                    f"{level!r}"
                )
            if (
                target.get("portable") is not True
                or target.get("target_triple") != PREPARED_JIT_PORTABLE_TARGET
                or target.get("word_bits") != 64
                or target.get("endianness") != "little"
                or cpu_features
            ):
                raise PreparedModelBundleError(
                    "prepared JIT packs must use the portable 64-bit "
                    "little-endian SymJIT storage-v3 target"
                )
        kernels = tuple(sorted(self.kernels, key=lambda kernel: kernel.kernel_id))
        if not kernels:
            raise PreparedModelBundleError("kernel_pack.kernels must not be empty")
        kernel_ids = tuple(kernel.kernel_id for kernel in kernels)
        signatures = tuple(kernel.canonical_signature for kernel in kernels)
        if len(set(kernel_ids)) != len(kernel_ids):
            raise PreparedModelBundleError("kernel IDs must be unique")
        if len(set(signatures)) != len(signatures):
            raise PreparedModelBundleError("kernel canonical signatures must be unique")
        object.__setattr__(self, "kernels", kernels)
        if self.backend == "jit":
            for kernel in kernels:
                _validate_portable_jit_manifest(
                    kernel.f64_evaluator_manifest,
                    f"kernel {kernel.kernel_id}",
                )
        by_id = {kernel.kernel_id: kernel for kernel in kernels}
        variants = tuple(
            sorted(
                self.kernel_variants,
                key=lambda variant: (variant.base_kernel_id, variant.variant_id),
            )
        )
        variant_keys = tuple(
            (variant.base_kernel_id, variant.variant_id) for variant in variants
        )
        if len(set(variant_keys)) != len(variant_keys):
            raise PreparedModelBundleError(
                "prepared kernel variant identities must be unique"
            )
        optimization_digest = prepared_optimization_settings_digest(
            self.optimization_settings
        )
        for variant in variants:
            kernel = by_id.get(variant.base_kernel_id)
            if kernel is None:
                raise PreparedModelBundleError(
                    "prepared kernel variant references an unknown base kernel"
                )
            variant.validate_base_kernel(kernel)
            if variant.backend != self.backend:
                raise PreparedModelBundleError(
                    "prepared kernel variant backend does not match its pack"
                )
            if variant.optimization_settings_digest != optimization_digest:
                raise PreparedModelBundleError(
                    "prepared kernel variant optimization digest does not match "
                    "its pack"
                )
            if variant.kind == "independent-block" and self.backend != "jit":
                raise PreparedModelBundleError(
                    "independent-block variants are currently supported only "
                    "for JIT packs"
                )
        object.__setattr__(self, "kernel_variants", variants)
        if self.backend == "jit":
            for variant in variants:
                _validate_portable_jit_manifest(
                    variant.f64_evaluator_manifest,
                    f"kernel {variant.base_kernel_id} variant {variant.variant_id!r}",
                )
        recurrence_template = self.recurrence_template
        if recurrence_template is not None:
            # Keep the prepared-bundle container independent from recurrence
            # construction while still rejecting stale or incomplete semantic
            # catalogs at the publication boundary.
            from .recurrence_template import RecurrenceTemplateCatalog

            try:
                recurrence_payload = _thaw_json(recurrence_template)
                catalog = RecurrenceTemplateCatalog.from_dict(
                    _mapping(
                        recurrence_payload,
                        "kernel_pack.recurrence_template",
                    )
                )
            except ValueError as exc:
                raise PreparedModelBundleError(
                    f"invalid kernel_pack.recurrence_template: {exc}"
                ) from exc
            _validate_recurrence_kernel_bindings(catalog, kernels)
            object.__setattr__(
                self,
                "recurrence_template",
                _freeze_mapping(
                    catalog.to_dict(),
                    "kernel_pack.recurrence_template",
                ),
            )
        recurrence_direct_template = self.recurrence_direct_template
        if recurrence_direct_template is not None:
            if self.recurrence_template is None:
                raise PreparedModelBundleError(
                    "kernel_pack.recurrence_direct_template requires the "
                    "authenticated semantic recurrence companion"
                )
            from .recurrence_direct_template import (
                RecurrenceDirectTemplateCatalogV1,
                RecurrenceDirectTemplateError,
            )

            try:
                direct_catalog = RecurrenceDirectTemplateCatalogV1.from_dict(
                    _mapping(
                        _thaw_json(recurrence_direct_template),
                        "kernel_pack.recurrence_direct_template",
                    )
                )
            except RecurrenceDirectTemplateError as exc:
                raise PreparedModelBundleError(
                    f"invalid kernel_pack.recurrence_direct_template: {exc}"
                ) from exc
            semantic_catalog = self.recurrence_template_catalog
            assert semantic_catalog is not None
            _validate_recurrence_direct_bindings(
                direct_catalog,
                semantic_catalog,
                kernels,
                backend=self.backend,
                target=self.target,
                optimization_settings=self.optimization_settings,
            )
            expected_provenance = {
                "direct_template_catalog_digest": direct_catalog.catalog_digest,
                "prepared_kernel_contract_digest": (
                    direct_catalog.prepared_kernel_contract_digest
                ),
                "prepared_kernel_payload_digest": (
                    direct_catalog.prepared_kernel_payload_digest
                ),
                "prepared_kernel_pack_digest": (
                    direct_catalog.prepared_kernel_pack_digest
                ),
                "recurrence_direct_template_abi": direct_catalog.abi,
                "recurrence_direct_template_digest": direct_catalog.catalog_digest,
            }
            for name, expected in expected_provenance.items():
                if self.provenance.get(name) != expected:
                    raise PreparedModelBundleError(
                        "direct recurrence companion provenance field "
                        f"{name!r} does not match its authenticated catalog"
                    )
            object.__setattr__(
                self,
                "recurrence_direct_template",
                _freeze_mapping(
                    direct_catalog.to_dict(),
                    "kernel_pack.recurrence_direct_template",
                ),
            )
            object.__setattr__(
                self,
                "_recurrence_direct_template_catalog",
                direct_catalog,
            )
        object.__setattr__(
            self,
            "_referenced_payload_paths",
            tuple(
                sorted(
                    {
                        path
                        for record in (*kernels, *variants)
                        for path in record.referenced_payload_paths
                    }
                )
            ),
        )

    @property
    def referenced_payload_paths(self) -> tuple[str, ...]:
        return self._referenced_payload_paths

    @property
    def recurrence_template_catalog(self) -> RecurrenceTemplateCatalog | None:
        """Return the validated semantic companion as its typed catalog."""

        recurrence_template = self.recurrence_template
        if recurrence_template is None:
            return None
        from .recurrence_template import RecurrenceTemplateCatalog

        thawed = _thaw_json(recurrence_template)
        return RecurrenceTemplateCatalog.from_dict(
            _mapping(thawed, "kernel_pack.recurrence_template")
        )

    @property
    def recurrence_direct_template_catalog(
        self,
    ) -> RecurrenceDirectTemplateCatalogV1 | None:
        """Return the authenticated Direct-Arena executor companion."""

        return self._recurrence_direct_template_catalog

    @property
    def direct_template_catalog_digest(self) -> str | None:
        """Stable digest consumed by process plans and native handle binding."""

        catalog = self.recurrence_direct_template_catalog
        return None if catalog is None else catalog.catalog_digest

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend": self.backend,
            "optimization_settings": _thaw_json(self.optimization_settings),
            "producer": _thaw_json(self.producer),
            "dependency_abis": _thaw_json(self.dependency_abis),
            "provenance": _thaw_json(self.provenance),
            "target": _thaw_json(self.target),
            "resolver_manifest": _thaw_json(self.resolver_manifest),
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "kernel_variants": [variant.to_dict() for variant in self.kernel_variants],
        }
        # Omitting this optional companion preserves the byte-for-byte manifest
        # shape of prepared bundles produced before recurrence execution.
        if self.recurrence_template is not None:
            payload["recurrence_template"] = _thaw_json(self.recurrence_template)
        if self.recurrence_direct_template is not None:
            payload["recurrence_direct_template"] = _thaw_json(
                self.recurrence_direct_template
            )
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PreparedKernelPack:
        normalized = dict(value)
        normalized.setdefault("recurrence_template", None)
        normalized.setdefault("recurrence_direct_template", None)
        expected = frozenset(
            (
                "backend",
                "optimization_settings",
                "producer",
                "dependency_abis",
                "provenance",
                "target",
                "resolver_manifest",
                "kernels",
                "kernel_variants",
                "recurrence_template",
                "recurrence_direct_template",
            )
        )
        _require_exact_keys(normalized, "kernel_pack", expected)
        backend = _nonempty_string(normalized.get("backend"), "kernel_pack.backend")
        kernels = tuple(
            PreparedKernelRecord.from_dict(
                _mapping(item, f"kernel_pack.kernels[{index}]")
            )
            for index, item in enumerate(
                _sequence(normalized.get("kernels"), "kernel_pack.kernels")
            )
        )
        kernel_variants = tuple(
            PreparedKernelVariantRecord.from_dict(
                _mapping(item, f"kernel_pack.kernel_variants[{index}]")
            )
            for index, item in enumerate(
                _sequence(
                    normalized.get("kernel_variants"),
                    "kernel_pack.kernel_variants",
                )
            )
        )
        return cls(
            backend=cast(PreparedBackend, backend),
            optimization_settings=_mapping(
                normalized.get("optimization_settings"),
                "kernel_pack.optimization_settings",
            ),
            producer=_mapping(normalized.get("producer"), "kernel_pack.producer"),
            dependency_abis=_mapping(
                normalized.get("dependency_abis"),
                "kernel_pack.dependency_abis",
            ),
            provenance=_mapping(normalized.get("provenance"), "kernel_pack.provenance"),
            target=_mapping(normalized.get("target"), "kernel_pack.target"),
            resolver_manifest=_mapping(
                normalized.get("resolver_manifest"),
                "kernel_pack.resolver_manifest",
            ),
            kernels=kernels,
            kernel_variants=kernel_variants,
            recurrence_template=(
                None
                if normalized.get("recurrence_template") is None
                else _mapping(
                    normalized.get("recurrence_template"),
                    "kernel_pack.recurrence_template",
                )
            ),
            recurrence_direct_template=(
                None
                if normalized.get("recurrence_direct_template") is None
                else _mapping(
                    normalized.get("recurrence_direct_template"),
                    "kernel_pack.recurrence_direct_template",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class PreparedKernelPackIdentity:
    """Non-circular identity of callable contracts and payload contents."""

    contract_digest: str
    payload_digest: str
    pack_digest: str
    abi: str = PREPARED_KERNEL_PACK_IDENTITY_ABI

    def __post_init__(self) -> None:
        if self.abi != PREPARED_KERNEL_PACK_IDENTITY_ABI:
            raise PreparedModelBundleError(
                f"unsupported prepared pack identity ABI {self.abi!r}; "
                "regenerate the prepared model with this pyAmpliCol version"
            )
        for name in ("contract_digest", "payload_digest", "pack_digest"):
            _valid_sha256(getattr(self, name), f"prepared_pack_identity.{name}")


def prepared_kernel_pack_identity(
    kernel_pack: PreparedKernelPack,
    payload_records: Mapping[str, tuple[int, str]],
) -> PreparedKernelPackIdentity:
    """Authenticate one backend pack without including its recurrence companion."""

    normalized_records: dict[str, tuple[int, str]] = {}
    for raw_path, raw_record in payload_records.items():
        member_path = _normalized_member_path(
            raw_path, "prepared payload identity path"
        )
        if (
            not isinstance(raw_record, tuple)
            or len(raw_record) != 2
            or type(raw_record[0]) is not int
            or raw_record[0] < 0
        ):
            raise PreparedModelBundleError(
                f"prepared payload identity record for {member_path!r} is malformed"
            )
        normalized_records[member_path] = (
            raw_record[0],
            _valid_sha256(
                raw_record[1],
                f"prepared payload identity digest for {member_path!r}",
            ),
        )
    referenced = set(kernel_pack.referenced_payload_paths)
    supplied = set(normalized_records)
    if supplied != referenced:
        missing = sorted(referenced - supplied)
        unexpected = sorted(supplied - referenced)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unreferenced: " + ", ".join(unexpected))
        raise PreparedModelBundleError(
            "prepared payload identity does not match kernel references ("
            + "; ".join(details)
            + ")"
        )

    contract_payload = {
        "abi": PREPARED_KERNEL_PACK_IDENTITY_ABI,
        "eager_kernel_abi": EAGER_KERNEL_ABI,
        "backend": kernel_pack.backend,
        "optimization_settings": _thaw_json(kernel_pack.optimization_settings),
        "dependency_abis": _thaw_json(kernel_pack.dependency_abis),
        "target": _thaw_json(kernel_pack.target),
        "resolver_manifest": _thaw_json(kernel_pack.resolver_manifest),
        "kernels": [kernel.to_dict() for kernel in kernel_pack.kernels],
        "kernel_variants": [
            variant.to_dict() for variant in kernel_pack.kernel_variants
        ],
    }
    contract_digest = _mapping_digest(contract_payload)
    payload_digest = _mapping_digest(
        {
            "abi": PREPARED_KERNEL_PACK_IDENTITY_ABI,
            "payloads": [
                {"path": path, "size": size, "sha256": digest}
                for path, (size, digest) in sorted(normalized_records.items())
            ],
        }
    )
    pack_digest = _mapping_digest(
        {
            "abi": PREPARED_KERNEL_PACK_IDENTITY_ABI,
            "contract_digest": contract_digest,
            "payload_digest": payload_digest,
        }
    )
    return PreparedKernelPackIdentity(
        contract_digest=contract_digest,
        payload_digest=payload_digest,
        pack_digest=pack_digest,
    )


def _validate_recurrence_direct_pack_identity(
    kernel_pack: PreparedKernelPack,
    identity: PreparedKernelPackIdentity,
) -> None:
    direct_catalog = kernel_pack.recurrence_direct_template_catalog
    if direct_catalog is None:
        return
    expected = {
        "prepared kernel contract": (
            direct_catalog.prepared_kernel_contract_digest,
            identity.contract_digest,
        ),
        "prepared kernel payload": (
            direct_catalog.prepared_kernel_payload_digest,
            identity.payload_digest,
        ),
        "prepared kernel pack": (
            direct_catalog.prepared_kernel_pack_digest,
            identity.pack_digest,
        ),
    }
    for name, (catalog_digest, actual_digest) in expected.items():
        if catalog_digest != actual_digest:
            raise PreparedModelBundleError(
                f"direct recurrence {name} digest does not match the "
                "published evaluator payloads"
            )


@dataclass(frozen=True, slots=True)
class PreparedModelBundle:
    """A validated prepared-model archive and its immutable metadata."""

    path: Path
    compiled_model: Mapping[str, object]
    kernel_pack: PreparedKernelPack
    manifest: Mapping[str, object]
    _member_digests: Mapping[str, str] = field(repr=False)
    _referenced_payload_paths: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _referenced_payload_path_set: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.resolve())
        object.__setattr__(
            self,
            "compiled_model",
            _freeze_mapping(self.compiled_model, "compiled_model"),
        )
        object.__setattr__(
            self,
            "manifest",
            _freeze_mapping(self.manifest, "manifest"),
        )
        object.__setattr__(
            self,
            "_member_digests",
            MappingProxyType(dict(self._member_digests)),
        )
        recurrence_catalog = self.kernel_pack.recurrence_template_catalog
        if (
            recurrence_catalog is not None
            and recurrence_catalog.header.compiled_model_digest
            != self._member_digests[PREPARED_MODEL_COMPILED_MODEL_PATH]
        ):
            raise PreparedModelBundleError(
                "recurrence template compiled-model digest does not match the "
                "prepared bundle member"
            )
        direct_catalog = self.kernel_pack.recurrence_direct_template_catalog
        if (
            direct_catalog is not None
            and direct_catalog.compiled_model_digest
            != self._member_digests[PREPARED_MODEL_COMPILED_MODEL_PATH]
        ):
            raise PreparedModelBundleError(
                "direct recurrence template compiled-model digest does not "
                "match the prepared bundle member"
            )
        referenced_payload_paths = self.kernel_pack.referenced_payload_paths
        object.__setattr__(
            self,
            "_referenced_payload_paths",
            referenced_payload_paths,
        )
        object.__setattr__(
            self,
            "_referenced_payload_path_set",
            frozenset(referenced_payload_paths),
        )

    @property
    def backend(self) -> PreparedBackend:
        return self.kernel_pack.backend

    @property
    def direct_template_catalog_digest(self) -> str | None:
        """Expose the authenticated direct catalog identity without lowering."""

        return self.kernel_pack.direct_template_catalog_digest

    def compiled_model_payload(self) -> dict[str, object]:
        """Return a detached plain-JSON payload for the model deserializer."""

        payload = _thaw_json(self.compiled_model)
        if not isinstance(payload, dict):  # pragma: no cover - constructor invariant
            raise PreparedModelBundleError("compiled model root must be an object")
        return payload

    def read_payload(self, member_path: str) -> bytes:
        """Read and revalidate one kernel payload from the archive."""
        normalized = _normalized_member_path(member_path, "member_path")
        if normalized not in self._referenced_payload_path_set:
            raise PreparedModelBundleError(
                f"{normalized!r} is not referenced by the prepared kernel pack"
            )
        try:
            with zipfile.ZipFile(self.path, "r") as archive:
                data = archive.read(normalized)
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            raise PreparedModelBundleError(
                f"could not read prepared payload {normalized!r}: {error}"
            ) from error
        expected = self._member_digests[normalized]
        if _sha256(data) != expected:
            raise PreparedModelBundleError(
                f"prepared payload {normalized!r} has a SHA-256 mismatch"
            )
        return data

    def copy_referenced_payloads(self, destination: Path) -> tuple[Path, ...]:
        """Atomically copy all referenced backend payloads below *destination*."""
        destination.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise PreparedModelBundleError("payload destination must not be a symlink")
        root = destination.resolve()
        outputs: list[Path] = []
        for member_path in self._referenced_payload_paths:
            output = root.joinpath(*PurePosixPath(member_path).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                output.parent.resolve().relative_to(root)
            except ValueError as error:
                raise PreparedModelBundleError(
                    f"payload destination for {member_path!r} escapes its root"
                ) from error
            if output.is_symlink():
                raise PreparedModelBundleError(
                    f"payload destination {output} must not be a symlink"
                )
            data = self.read_payload(member_path)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
            try:
                temporary.write_bytes(data)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
            outputs.append(output)
        return tuple(outputs)


def _read_payload_source(source: PayloadSource, context: str) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, Path):
        if source.is_symlink():
            raise PreparedModelBundleError(f"{context} must not be a symlink")
        if not source.is_file():
            raise PreparedModelBundleError(f"{context} must be a regular file")
        return source.read_bytes()
    raise PreparedModelBundleError(f"{context} has an unsupported payload source")


def prepared_payload_identity_records(
    payloads: Mapping[str, PayloadSource],
) -> dict[str, tuple[int, str]]:
    """Read payload sources once into stable size/digest identity records."""

    records: dict[str, tuple[int, str]] = {}
    for raw_path, source in payloads.items():
        member_path = _normalized_member_path(raw_path, "payload identity path")
        if member_path in records:
            raise PreparedModelBundleError(
                f"duplicate payload identity path {member_path!r}"
            )
        data = _read_payload_source(source, f"payload {member_path!r}")
        records[member_path] = (len(data), _sha256(data))
    return records


def _bundle_output_path(path: Path) -> Path:
    text = str(path)
    if not text.endswith(PREPARED_MODEL_BUNDLE_SUFFIX):
        text += PREPARED_MODEL_BUNDLE_SUFFIX
    return Path(text).expanduser().resolve()


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = _ZIP_FILE_MODE
    info.extra = b""
    info.comment = b""
    return info


def _write_zip(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.comment = b""
            for member_path in (
                PREPARED_MODEL_MANIFEST_PATH,
                *sorted(
                    path for path in members if path != PREPARED_MODEL_MANIFEST_PATH
                ),
            ):
                archive.writestr(_zip_info(member_path), members[member_path])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_prepared_model_bundle(
    path: Path,
    *,
    compiled_model: Mapping[str, object],
    kernel_pack: PreparedKernelPack,
    payloads: Mapping[str, PayloadSource],
) -> Path:
    """Write one deterministic, self-contained prepared-model bundle."""
    frozen_model = _freeze_mapping(compiled_model, "compiled_model")
    recurrence_catalog = kernel_pack.recurrence_template_catalog
    if (
        recurrence_catalog is not None
        and recurrence_catalog.header.compiled_model_digest
        != _sha256(_canonical_json(frozen_model))
    ):
        raise PreparedModelBundleError(
            "recurrence template compiled-model digest does not match the "
            "prepared bundle member"
        )
    direct_catalog = kernel_pack.recurrence_direct_template_catalog
    if direct_catalog is not None and direct_catalog.compiled_model_digest != _sha256(
        _canonical_json(frozen_model)
    ):
        raise PreparedModelBundleError(
            "direct recurrence template compiled-model digest does not match "
            "the prepared bundle member"
        )
    payload_data: dict[str, bytes] = {}
    for raw_path, source in payloads.items():
        member_path = _normalized_member_path(raw_path, "payload path")
        if member_path in {
            PREPARED_MODEL_MANIFEST_PATH,
            PREPARED_MODEL_COMPILED_MODEL_PATH,
        }:
            raise PreparedModelBundleError(
                f"payload path {member_path!r} is reserved by the bundle"
            )
        if member_path in payload_data:
            raise PreparedModelBundleError(f"duplicate payload path {member_path!r}")
        payload_data[member_path] = _read_payload_source(
            source,
            f"payload {member_path!r}",
        )
    referenced = set(kernel_pack.referenced_payload_paths)
    supplied = set(payload_data)
    if referenced != supplied:
        missing = sorted(referenced - supplied)
        unexpected = sorted(supplied - referenced)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unreferenced: " + ", ".join(unexpected))
        raise PreparedModelBundleError(
            "prepared payload set does not match kernel references ("
            + "; ".join(details)
            + ")"
        )
    if recurrence_catalog is not None:
        identity = prepared_kernel_pack_identity(
            kernel_pack,
            {
                member_path: (len(data), _sha256(data))
                for member_path, data in payload_data.items()
            },
        )
        if (
            identity.pack_digest
            != recurrence_catalog.header.prepared_kernel_pack_digest
        ):
            raise PreparedModelBundleError(
                "recurrence template prepared-pack digest does not match the "
                "published evaluator payloads"
            )
        _validate_recurrence_direct_pack_identity(kernel_pack, identity)
    hashed_members = {
        PREPARED_MODEL_COMPILED_MODEL_PATH: _canonical_json(frozen_model),
        **payload_data,
    }
    member_records = [
        {
            "path": member_path,
            "size": len(data),
            "sha256": _sha256(data),
        }
        for member_path, data in sorted(hashed_members.items())
    ]
    manifest: dict[str, object] = {
        "kind": PREPARED_MODEL_BUNDLE_KIND,
        "schema_version": PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
        "eager_kernel_abi": EAGER_KERNEL_ABI,
        "kernel_pack": kernel_pack.to_dict(),
        "members": member_records,
    }
    if direct_catalog is not None:
        manifest["direct_template_catalog_digest"] = direct_catalog.catalog_digest
    members = {
        PREPARED_MODEL_MANIFEST_PATH: _canonical_json(manifest),
        **hashed_members,
    }
    output = _bundle_output_path(path)
    _write_zip(output, members)
    return output


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for index, info in enumerate(archive.infolist()):
        member_path = _normalized_member_path(info.filename, f"archive member {index}")
        if member_path in members:
            raise PreparedModelBundleError(
                f"prepared-model archive contains duplicate member {member_path!r}"
            )
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_IFMT(mode) == stat.S_IFLNK:
            raise PreparedModelBundleError(
                f"prepared-model member {member_path!r} must be a regular file"
            )
        members[member_path] = info
    return members


def _load_json_member(
    archive: zipfile.ZipFile,
    member_path: str,
    context: str,
) -> Mapping[str, object]:
    try:
        payload = json.loads(archive.read(member_path))
    except KeyError as error:
        raise PreparedModelBundleError(
            f"prepared-model archive is missing {member_path!r}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparedModelBundleError(
            f"{context} is not valid JSON: {error}"
        ) from error
    return _mapping(payload, context)


def _load_member_records(value: object) -> dict[str, tuple[int, str]]:
    records: dict[str, tuple[int, str]] = {}
    for index, item in enumerate(_sequence(value, "manifest.members")):
        record = _mapping(item, f"manifest.members[{index}]")
        _require_exact_keys(
            record,
            f"manifest.members[{index}]",
            frozenset(("path", "size", "sha256")),
        )
        path = _normalized_member_path(
            record.get("path"),
            f"manifest.members[{index}].path",
        )
        if path == PREPARED_MODEL_MANIFEST_PATH:
            raise PreparedModelBundleError("manifest.json must not hash itself")
        if path in records:
            raise PreparedModelBundleError(
                f"manifest contains duplicate member record {path!r}"
            )
        size = _nonnegative_integer(
            record.get("size"),
            f"manifest.members[{index}].size",
        )
        digest = _valid_sha256(
            record.get("sha256"),
            f"manifest.members[{index}].sha256",
        )
        records[path] = (size, digest)
    return records


def load_prepared_model_bundle(path: Path) -> PreparedModelBundle:
    """Load and validate a prepared-model bundle without extracting it."""
    requested_path = Path(path).expanduser()
    if requested_path.is_symlink():
        raise PreparedModelBundleError("prepared-model bundle must not be a symlink")
    archive_path = requested_path.resolve()
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_members = _archive_members(archive)
            if PREPARED_MODEL_MANIFEST_PATH not in archive_members:
                raise PreparedModelBundleError(
                    "prepared-model archive is missing 'manifest.json'"
                )
            manifest = _load_json_member(
                archive,
                PREPARED_MODEL_MANIFEST_PATH,
                "manifest",
            )
            base_manifest_fields = frozenset(
                (
                    "kind",
                    "schema_version",
                    "eager_kernel_abi",
                    "kernel_pack",
                    "members",
                )
            )
            manifest_fields = frozenset(manifest)
            expected_manifest_fields = (
                base_manifest_fields | {"direct_template_catalog_digest"}
                if "direct_template_catalog_digest" in manifest_fields
                else base_manifest_fields
            )
            _require_exact_keys(
                manifest,
                "manifest",
                expected_manifest_fields,
            )
            if manifest.get("kind") != PREPARED_MODEL_BUNDLE_KIND:
                raise PreparedModelBundleError("invalid prepared-model bundle kind")
            if manifest.get("schema_version") != PREPARED_MODEL_BUNDLE_SCHEMA_VERSION:
                raise PreparedModelBundleError(
                    "unsupported prepared-model bundle schema"
                )
            if manifest.get("eager_kernel_abi") != EAGER_KERNEL_ABI:
                raise PreparedModelBundleError("unsupported eager kernel ABI")
            kernel_pack = PreparedKernelPack.from_dict(
                _mapping(manifest.get("kernel_pack"), "manifest.kernel_pack")
            )
            direct_catalog = kernel_pack.recurrence_direct_template_catalog
            has_manifest_direct_digest = "direct_template_catalog_digest" in manifest
            if direct_catalog is None:
                if has_manifest_direct_digest:
                    raise PreparedModelBundleError(
                        "manifest advertises a direct template catalog that is absent"
                    )
            else:
                manifest_direct_digest = (
                    _valid_sha256(
                        manifest.get("direct_template_catalog_digest"),
                        "manifest.direct_template_catalog_digest",
                    )
                    if has_manifest_direct_digest
                    else None
                )
                if manifest_direct_digest != direct_catalog.catalog_digest:
                    raise PreparedModelBundleError(
                        "manifest direct template catalog digest does not match the "
                        "prepared companion"
                    )
            records = _load_member_records(manifest.get("members"))
            expected_paths = {PREPARED_MODEL_MANIFEST_PATH, *records}
            actual_paths = set(archive_members)
            if expected_paths != actual_paths:
                missing = sorted(expected_paths - actual_paths)
                unexpected = sorted(actual_paths - expected_paths)
                details: list[str] = []
                if missing:
                    details.append("missing: " + ", ".join(missing))
                if unexpected:
                    details.append("unexpected: " + ", ".join(unexpected))
                raise PreparedModelBundleError(
                    "prepared-model archive members do not match manifest ("
                    + "; ".join(details)
                    + ")"
                )
            if PREPARED_MODEL_COMPILED_MODEL_PATH not in records:
                raise PreparedModelBundleError(
                    "prepared-model archive is missing its compiled model"
                )
            for member_path, (expected_size, expected_digest) in records.items():
                data = archive.read(member_path)
                if len(data) != expected_size:
                    raise PreparedModelBundleError(
                        f"prepared-model member {member_path!r} has a size mismatch"
                    )
                if _sha256(data) != expected_digest:
                    raise PreparedModelBundleError(
                        f"prepared-model member {member_path!r} has a SHA-256 mismatch"
                    )
            payload_paths = set(records) - {PREPARED_MODEL_COMPILED_MODEL_PATH}
            referenced_paths = set(kernel_pack.referenced_payload_paths)
            if payload_paths != referenced_paths:
                missing = sorted(referenced_paths - payload_paths)
                unexpected = sorted(payload_paths - referenced_paths)
                details = []
                if missing:
                    details.append("missing: " + ", ".join(missing))
                if unexpected:
                    details.append("unreferenced: " + ", ".join(unexpected))
                raise PreparedModelBundleError(
                    "prepared payload members do not match kernel references ("
                    + "; ".join(details)
                    + ")"
                )
            recurrence_catalog = kernel_pack.recurrence_template_catalog
            if recurrence_catalog is not None:
                identity = prepared_kernel_pack_identity(
                    kernel_pack,
                    {
                        member_path: records[member_path]
                        for member_path in referenced_paths
                    },
                )
                if (
                    identity.pack_digest
                    != recurrence_catalog.header.prepared_kernel_pack_digest
                ):
                    raise PreparedModelBundleError(
                        "recurrence template prepared-pack digest does not match "
                        "the loaded evaluator payloads"
                    )
                _validate_recurrence_direct_pack_identity(kernel_pack, identity)
            compiled_model = _load_json_member(
                archive,
                PREPARED_MODEL_COMPILED_MODEL_PATH,
                "compiled_model",
            )
    except PreparedModelBundleError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise PreparedModelBundleError(
            f"could not open prepared-model bundle: {error}"
        ) from error
    return PreparedModelBundle(
        path=archive_path,
        compiled_model=compiled_model,
        kernel_pack=kernel_pack,
        manifest=manifest,
        _member_digests={
            member_path: digest for member_path, (_, digest) in records.items()
        },
    )


def read_prepared_model_bundle(path: Path) -> PreparedModelBundle:
    """Alias for :func:`load_prepared_model_bundle` for reader-style callers."""
    return load_prepared_model_bundle(path)


__all__ = [
    "EAGER_KERNEL_ABI",
    "EAGER_KERNEL_ABI_VERSION",
    "PREPARED_INDEPENDENT_BLOCK_SIZE",
    "PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL",
    "PREPARED_JIT_PORTABLE_TARGET",
    "PREPARED_KERNEL_PACK_IDENTITY_ABI",
    "PREPARED_KERNEL_VARIANT_ABI",
    "PREPARED_MODEL_BUNDLE_KIND",
    "PREPARED_MODEL_BUNDLE_SCHEMA_VERSION",
    "PREPARED_MODEL_BUNDLE_SUFFIX",
    "PREPARED_MODEL_COMPILED_MODEL_PATH",
    "PREPARED_MODEL_MANIFEST_PATH",
    "PreparedKernelPack",
    "PreparedKernelPackIdentity",
    "PreparedKernelRecord",
    "PreparedKernelVariantRecord",
    "PreparedModelBundle",
    "PreparedModelBundleError",
    "load_prepared_model_bundle",
    "prepared_compiled_model_digest",
    "prepared_expression_digest",
    "prepared_input_contract_digest",
    "prepared_kernel_pack_identity",
    "prepared_kernel_pack_manifest_identity_sha256",
    "prepared_optimization_settings_digest",
    "prepared_output_contract_digest",
    "prepared_payload_identity_records",
    "read_prepared_model_bundle",
    "write_prepared_model_bundle",
]
