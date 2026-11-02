# SPDX-License-Identifier: 0BSD
"""Process-independent exact kernel catalogs for prepared eager models.

The catalog is deliberately upstream of generation.  It converts the local
model contracts into content-addressed exact expressions and model-specific
lookup bindings, but it never constructs a backend evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias

from .contact_decomposition import (
    CompiledContactOrbitCertificate,
    CompiledContactOrbitStep,
)

if TYPE_CHECKING:
    from .base import Model

PreparedContractKind: TypeAlias = Literal[
    "vertex",
    "propagator",
    "closure",
    "model-parameter",
]
PreparedInputRole: TypeAlias = Literal[
    "left-current",
    "right-current",
    "left-momentum",
    "right-momentum",
    "current",
    "momentum",
    "coupling-real",
    "coupling-imag",
    "model-parameter",
]
PreparedOutputFactorSource: TypeAlias = Literal[
    "none",
    "coupling-real",
    "coupling-imag",
]

_CATALOG_ABI = "pyamplicol-prepared-kernel-catalog-v1"
PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF = (
    "prepared-kernel-homogeneous-complex-linear-current-v1"
)
PREPARED_INDEPENDENT_BLOCK_PROOF = (
    "prepared-kernel-independent-current-block-v1"
)
_CONTRACT_KINDS = frozenset(("vertex", "propagator", "closure", "model-parameter"))
_INPUT_ROLES = frozenset(
    (
        "left-current",
        "right-current",
        "left-momentum",
        "right-momentum",
        "current",
        "momentum",
        "coupling-real",
        "coupling-imag",
        "model-parameter",
    )
)
_OUTPUT_FACTOR_SOURCES = frozenset(("none", "coupling-real", "coupling-imag"))


class PreparedKernelCatalogError(ValueError):
    """Raised when a model cannot produce a complete exact eager catalog."""


@dataclass(frozen=True, order=True, slots=True)
class PreparedKernelInput:
    """One scalar input in a prepared evaluator's gather contract."""

    role: PreparedInputRole
    component: int
    symbol: str
    model_parameter_name: str | None = None
    model_parameter_index: int | None = None

    def __post_init__(self) -> None:
        if self.role not in _INPUT_ROLES:
            raise PreparedKernelCatalogError(
                f"unsupported prepared input role {self.role!r}"
            )
        if self.component < 0:
            raise PreparedKernelCatalogError(
                "prepared input component must be nonnegative"
            )
        if not self.symbol:
            raise PreparedKernelCatalogError("prepared input symbol must be nonempty")
        parameter_fields = (
            self.model_parameter_name,
            self.model_parameter_index,
        )
        if self.role == "model-parameter":
            if not self.model_parameter_name or self.model_parameter_index is None:
                raise PreparedKernelCatalogError(
                    "model-parameter inputs require a name and stable index"
                )
            if self.model_parameter_index < 0:
                raise PreparedKernelCatalogError(
                    "model-parameter input index must be nonnegative"
                )
        elif parameter_fields != (None, None):
            raise PreparedKernelCatalogError(
                "only model-parameter inputs may carry parameter metadata"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "component": self.component,
            "symbol": self.symbol,
            "model_parameter_name": self.model_parameter_name,
            "model_parameter_index": self.model_parameter_index,
        }


@dataclass(frozen=True, order=True, slots=True)
class PreparedParticleState:
    """Comparable model-owned orientation metadata for one current state."""

    particle_id: int
    identity: str
    orientation: str
    basis: str
    chirality: int
    dimension: int

    def __post_init__(self) -> None:
        if not self.identity or not self.orientation or not self.basis:
            raise PreparedKernelCatalogError("prepared particle metadata is incomplete")
        if self.dimension <= 0:
            raise PreparedKernelCatalogError(
                f"prepared current {self.identity!r} has invalid dimension "
                f"{self.dimension}"
            )

    def contract_dict(self) -> dict[str, object]:
        return {
            "orientation": self.orientation,
            "basis": self.basis,
            "chirality": self.chirality,
            "dimension": self.dimension,
        }


@dataclass(frozen=True, order=True, slots=True)
class VertexKernelKey:
    kind: int
    particles: tuple[int, int, int]
    left_chirality: int
    right_chirality: int
    result_chirality: int
    coupling: tuple[float, float]


@dataclass(frozen=True, order=True, slots=True)
class PropagatorKernelKey:
    particle_id: int
    chirality: int


@dataclass(frozen=True, order=True, slots=True)
class ClosureKernelKey:
    kind: int
    particles: tuple[int, int, int]
    left_chirality: int
    right_chirality: int
    coupling: tuple[float, float]


@dataclass(frozen=True, order=True, slots=True)
class PreparedVertexBinding:
    key: VertexKernelKey
    kernel_id: int
    canonical_input_order: tuple[int, int]
    equivalence_class: str
    equivalence_factor: tuple[float, float]
    input_exchange_factor: tuple[float, float] | None
    left_state: PreparedParticleState
    right_state: PreparedParticleState
    result_state: PreparedParticleState
    output_factor_source: PreparedOutputFactorSource = "none"
    contact_orbit_certificates: tuple[CompiledContactOrbitCertificate, ...] = ()
    contact_orbit_steps: tuple[CompiledContactOrbitStep, ...] = ()

    def __post_init__(self) -> None:
        if self.output_factor_source not in _OUTPUT_FACTOR_SOURCES:
            raise PreparedKernelCatalogError(
                f"unsupported prepared output factor source "
                f"{self.output_factor_source!r}"
            )
        if self.contact_orbit_certificates != tuple(
            sorted(set(self.contact_orbit_certificates))
        ) or self.contact_orbit_steps != tuple(sorted(set(self.contact_orbit_steps))):
            raise PreparedKernelCatalogError(
                "prepared contact-orbit contracts must be sorted and unique"
            )
        if any(
            not isinstance(item, CompiledContactOrbitCertificate)
            for item in self.contact_orbit_certificates
        ) or any(
            not isinstance(item, CompiledContactOrbitStep)
            for item in self.contact_orbit_steps
        ):
            raise PreparedKernelCatalogError(
                "prepared contact-orbit contract must contain typed records"
            )
        certificate_terms = {
            certificate.term_id for certificate in self.contact_orbit_certificates
        }
        step_terms = {step.term_id for step in self.contact_orbit_steps}
        if certificate_terms != step_terms:
            raise PreparedKernelCatalogError(
                "prepared contact-orbit certificates and steps disagree"
            )


@dataclass(frozen=True, order=True, slots=True)
class PreparedPropagatorBinding:
    key: PropagatorKernelKey
    kernel_id: int | None
    state: PreparedParticleState
    applies_propagator: bool
    propagator_kind: str
    mass_class: str
    gauge: str | None
    model_parameters: tuple[str, ...]


@dataclass(frozen=True, order=True, slots=True)
class PreparedClosureBinding:
    key: ClosureKernelKey
    kernel_id: int
    canonical_input_order: tuple[int, int]
    equivalence_class: str
    equivalence_factor: tuple[float, float]
    input_exchange_factor: tuple[float, float] | None
    left_state: PreparedParticleState
    right_state: PreparedParticleState
    result_state: PreparedParticleState
    projection: str
    output_factor_source: PreparedOutputFactorSource = "none"

    def __post_init__(self) -> None:
        if self.output_factor_source not in _OUTPUT_FACTOR_SOURCES:
            raise PreparedKernelCatalogError(
                f"unsupported prepared output factor source "
                f"{self.output_factor_source!r}"
            )


@dataclass(frozen=True, order=True, slots=True)
class PreparedKernelGap:
    """A model-admitted local variant lacking a constructive exact lowering."""

    contract_kind: PreparedContractKind
    context: str
    reason: str


@dataclass(frozen=True, slots=True)
class PreparedKernelSpec:
    """One process-blind exact evaluator contract, before backend compilation."""

    kernel_id: int
    contract_kind: PreparedContractKind
    canonical_signature: str
    exact_expressions: tuple[str, ...]
    inputs: tuple[PreparedKernelInput, ...]
    output_layout: tuple[str, ...]
    proof_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kernel_id < 0:
            raise PreparedKernelCatalogError("prepared kernel id must be nonnegative")
        if self.contract_kind not in _CONTRACT_KINDS:
            raise PreparedKernelCatalogError(
                f"unsupported prepared contract kind {self.contract_kind!r}"
            )
        if len(self.canonical_signature) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.canonical_signature
        ):
            raise PreparedKernelCatalogError(
                "prepared canonical signature must be a lowercase SHA-256"
            )
        if not self.exact_expressions or len(self.exact_expressions) != len(
            self.output_layout
        ):
            raise PreparedKernelCatalogError(
                "prepared exact outputs and output layout must be nonempty and aligned"
            )
        if any(not expression for expression in self.exact_expressions):
            raise PreparedKernelCatalogError(
                "prepared exact expressions must be nonempty"
            )
        if len(set(self.inputs)) != len(self.inputs):
            raise PreparedKernelCatalogError("prepared kernel inputs must be unique")
        if self.proof_classes != tuple(sorted(set(self.proof_classes))):
            raise PreparedKernelCatalogError(
                "prepared kernel proof classes must be sorted and unique"
            )

    @property
    def input_arity(self) -> int:
        return len(self.inputs)

    @property
    def output_dimension(self) -> int:
        return len(self.exact_expressions)


@dataclass(frozen=True, slots=True)
class PreparedKernelCatalog:
    """Exact kernels plus deterministic model-local resolver bindings."""

    model_name: str
    kernels: tuple[PreparedKernelSpec, ...]
    vertex_bindings: tuple[PreparedVertexBinding, ...]
    propagator_bindings: tuple[PreparedPropagatorBinding, ...]
    closure_bindings: tuple[PreparedClosureBinding, ...]
    model_parameter_kernel_id: int | None
    unsupported_variants: tuple[PreparedKernelGap, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_name:
            raise PreparedKernelCatalogError("prepared catalog model name is empty")
        expected_ids = tuple(range(len(self.kernels)))
        actual_ids = tuple(kernel.kernel_id for kernel in self.kernels)
        if actual_ids != expected_ids:
            raise PreparedKernelCatalogError(
                "prepared kernel ids must be contiguous in signature order"
            )
        signatures = tuple(kernel.canonical_signature for kernel in self.kernels)
        if signatures != tuple(sorted(signatures)) or len(set(signatures)) != len(
            signatures
        ):
            raise PreparedKernelCatalogError(
                "prepared kernels must have unique sorted canonical signatures"
            )
        referenced = (
            {binding.kernel_id for binding in self.vertex_bindings}
            | {
                binding.kernel_id
                for binding in self.propagator_bindings
                if binding.kernel_id is not None
            }
            | {binding.kernel_id for binding in self.closure_bindings}
        )
        if self.model_parameter_kernel_id is not None:
            referenced.add(self.model_parameter_kernel_id)
        if any(kernel_id not in expected_ids for kernel_id in referenced):
            raise PreparedKernelCatalogError(
                "prepared resolver binding references an unknown kernel"
            )
        for context, keys in (
            ("vertex", tuple(binding.key for binding in self.vertex_bindings)),
            (
                "propagator",
                tuple(binding.key for binding in self.propagator_bindings),
            ),
            ("closure", tuple(binding.key for binding in self.closure_bindings)),
        ):
            if len(keys) != len(set(keys)):
                raise PreparedKernelCatalogError(
                    f"prepared {context} resolver keys must be unique"
                )

    @property
    def by_id(self) -> Mapping[int, PreparedKernelSpec]:
        return MappingProxyType({kernel.kernel_id: kernel for kernel in self.kernels})

    def resolver_mappings(self) -> Mapping[str, Mapping[object, int | None]]:
        """Return neutral maps for a generation-layer resolver adapter."""

        return MappingProxyType(
            {
                "vertex": MappingProxyType(
                    {binding.key: binding.kernel_id for binding in self.vertex_bindings}
                ),
                "propagator": MappingProxyType(
                    {
                        binding.key: binding.kernel_id
                        for binding in self.propagator_bindings
                    }
                ),
                "closure": MappingProxyType(
                    {
                        binding.key: binding.kernel_id
                        for binding in self.closure_bindings
                    }
                ),
            }
        )

    def resolver_manifest(self) -> dict[str, object]:
        """Serialize the process-independent lookup and proof metadata.

        Exact expressions live in :class:`PreparedKernelRecord` objects.  This
        manifest contains only the model-owned keys and transformations needed
        to select those records while lowering a process DAG.
        """

        return {
            "abi": _CATALOG_ABI,
            "model_name": self.model_name,
            "vertex_bindings": [
                {
                    "key": {
                        "kind": binding.key.kind,
                        "particles": list(binding.key.particles),
                        "left_chirality": binding.key.left_chirality,
                        "right_chirality": binding.key.right_chirality,
                        "result_chirality": binding.key.result_chirality,
                        "coupling": list(binding.key.coupling),
                    },
                    "kernel_id": binding.kernel_id,
                    "canonical_input_order": list(binding.canonical_input_order),
                    "equivalence_class": binding.equivalence_class,
                    "equivalence_factor": list(binding.equivalence_factor),
                    "input_exchange_factor": (
                        None
                        if binding.input_exchange_factor is None
                        else list(binding.input_exchange_factor)
                    ),
                    "output_factor_source": binding.output_factor_source,
                    "contact_orbit_certificates": [
                        item.to_dict()
                        for item in binding.contact_orbit_certificates
                    ],
                    "contact_orbit_steps": [
                        item.to_dict() for item in binding.contact_orbit_steps
                    ],
                }
                for binding in self.vertex_bindings
            ],
            "propagator_bindings": [
                {
                    "key": {
                        "particle_id": binding.key.particle_id,
                        "chirality": binding.key.chirality,
                    },
                    "kernel_id": binding.kernel_id,
                    "applies_propagator": binding.applies_propagator,
                }
                for binding in self.propagator_bindings
            ],
            "closure_bindings": [
                {
                    "key": {
                        "kind": binding.key.kind,
                        "particles": list(binding.key.particles),
                        "left_chirality": binding.key.left_chirality,
                        "right_chirality": binding.key.right_chirality,
                        "coupling": list(binding.key.coupling),
                    },
                    "kernel_id": binding.kernel_id,
                    "canonical_input_order": list(binding.canonical_input_order),
                    "equivalence_class": binding.equivalence_class,
                    "equivalence_factor": list(binding.equivalence_factor),
                    "input_exchange_factor": (
                        None
                        if binding.input_exchange_factor is None
                        else list(binding.input_exchange_factor)
                    ),
                    "projection": binding.projection,
                    "output_factor_source": binding.output_factor_source,
                }
                for binding in self.closure_bindings
            ],
            "model_parameter_kernel_id": self.model_parameter_kernel_id,
            "unsupported_variants": [
                {
                    "contract_kind": gap.contract_kind,
                    "context": gap.context,
                    "reason": gap.reason,
                }
                for gap in self.unsupported_variants
            ],
        }


def build_prepared_kernel_catalog(model: Model) -> PreparedKernelCatalog:
    """Build the complete process-independent exact catalog for ``model``."""

    from .prepared_catalog_builder import build_prepared_kernel_catalog as build

    return build(model)


__all__ = [
    "PREPARED_HOMOGENEOUS_LINEAR_CURRENT_PROOF",
    "PREPARED_INDEPENDENT_BLOCK_PROOF",
    "ClosureKernelKey",
    "PreparedClosureBinding",
    "PreparedContractKind",
    "PreparedInputRole",
    "PreparedKernelCatalog",
    "PreparedKernelCatalogError",
    "PreparedKernelGap",
    "PreparedKernelInput",
    "PreparedKernelSpec",
    "PreparedOutputFactorSource",
    "PreparedParticleState",
    "PreparedPropagatorBinding",
    "PreparedVertexBinding",
    "PropagatorKernelKey",
    "VertexKernelKey",
    "build_prepared_kernel_catalog",
]
