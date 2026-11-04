# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal, cast

from ._physics_ir import (
    ContractionIR,
    CrossingIR,
    GoldstonePolicy,
    ParticleIdentityIR,
    ParticleOrientation,
    ParticleStatistics,
    PropagatorGauge,
    PropagatorIR,
    PropagatorKind,
    PropagatorMassClass,
    SourceIR,
    SourceStateIR,
    WavefunctionFamily,
)
from .contact_decomposition import (
    CompiledContactOrbitCertificate,
    CompiledContactOrbitStep,
)


@dataclass(frozen=True)
class Particle:
    pdg: int
    anti_pdg: int
    spin: int
    dimension: int
    color_rep: int
    mass: float = 0.0
    width: float = 0.0
    charge: float = 0.0
    weak_isospin: tuple[float, float] = (0.0, 0.0)
    weak_hypercharge: tuple[float, float] = (0.0, 0.0)


def _runtime_particle_parameter_name(
    declared: str | None,
    *,
    particle_pdg: int,
    kind: str,
    available_names: Collection[str],
) -> str | None:
    """Resolve a model-owned name or its prepared-catalog fallback."""

    if declared is not None:
        return declared
    fallback = f"particle.{particle_pdg}.{kind}"
    return fallback if fallback in available_names else None


@dataclass(frozen=True)
class Vertex:
    kind: int
    particles: tuple[int, int, int]
    coupling: tuple[float, float] = (1.0, 0.0)


CouplingOrders = tuple[tuple[str, int], ...]
QuantumNumberFlow = tuple[tuple[str, str], ...]
RecurrenceFlavourFlowOperation = Literal[
    "constant-result",
    "append-left-result",
    "append-right-result",
    "concat-left-right-result",
]
RecurrenceQuantumNumberFlowOperation = Literal["particle-static-result"]
RecurrenceLCColorShapeKind = Literal[
    "singlet-forest",
    "fundamental-open-string",
    "antifundamental-open-string",
    "adjoint-segment",
]
RecurrenceLCColorComponentKind = Literal["open-string", "adjoint-segment", "trace"]
RecurrenceLCColorComponentRole = Literal["active", "passive", "none"]
RecurrenceLCColorSourceSeedOperation = Literal["empty", "singleton"]
RecurrenceLCColorOperation = Literal[
    "concatenate-join",
    "concatenate-keep",
    "inherit-left",
    "inherit-right",
    "empty",
    "close",
]


@dataclass(frozen=True)
class SourceSpinState:
    helicity: int
    chirality: int
    spin_state: int | tuple[int, ...]


@dataclass(frozen=True)
class QuantumFlow:
    chirality: int
    spin_state: int | tuple[int, ...]
    flavour_flow: tuple[int, ...]
    quantum_number_flow: QuantumNumberFlow
    coupling: tuple[float, float]


@dataclass(frozen=True)
class RecurrenceQuantumFlowContract:
    """Declarative dynamic-flow contract consumed by recurrence generation."""

    flavour_flow_operation: RecurrenceFlavourFlowOperation
    quantum_number_flow_operation: RecurrenceQuantumNumberFlowOperation
    admission_scope: Literal["static-input-contract"] = "static-input-contract"

    def __post_init__(self) -> None:
        if self.flavour_flow_operation not in {
            "constant-result",
            "append-left-result",
            "append-right-result",
            "concat-left-right-result",
        }:
            raise ValueError(
                "unsupported recurrence flavour-flow operation "
                f"{self.flavour_flow_operation!r}"
            )
        if self.quantum_number_flow_operation != "particle-static-result":
            raise ValueError(
                "unsupported recurrence quantum-number-flow operation "
                f"{self.quantum_number_flow_operation!r}"
            )
        if self.admission_scope != "static-input-contract":
            raise ValueError(
                f"unsupported recurrence admission scope {self.admission_scope!r}"
            )

    def to_json_dict(self) -> dict[str, str]:
        return {
            "admission_scope": self.admission_scope,
            "flavour_flow_operation": self.flavour_flow_operation,
            "quantum_number_flow_operation": self.quantum_number_flow_operation,
        }


@dataclass(frozen=True)
class RecurrenceCurrentEmbeddingContract:
    """Exact embedding of one source current into a full runtime current."""

    input_embedding: tuple[int | None, ...]
    source_dimension: int
    full_dimension: int
    source_basis: str
    full_basis: str
    proof_kind: str
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.source_dimension <= 0 or self.full_dimension <= 0:
            raise ValueError("recurrence current dimensions must be positive")
        if len(self.input_embedding) != self.full_dimension:
            raise ValueError(
                "recurrence current embedding length must equal the full dimension"
            )
        populated = tuple(
            int(component)
            for component in self.input_embedding
            if component is not None
        )
        if tuple(sorted(populated)) != tuple(range(self.source_dimension)):
            raise ValueError(
                "recurrence current embedding must use every source component "
                "exactly once"
            )
        if not self.source_basis or not self.full_basis or not self.proof_kind:
            raise ValueError(
                "recurrence current embedding basis and proof labels must be nonempty"
            )
        if self.provenance != tuple(sorted(set(self.provenance))):
            raise ValueError(
                "recurrence current embedding provenance must be sorted and unique"
            )

    @property
    def source_projection(self) -> tuple[int, ...]:
        """Return full-component indices ordered by source component."""

        inverse = [-1] * self.source_dimension
        for full_component, source_component in enumerate(self.input_embedding):
            if source_component is not None:
                inverse[int(source_component)] = full_component
        return tuple(inverse)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "full_basis": self.full_basis,
            "full_dimension": self.full_dimension,
            "input_embedding": list(self.input_embedding),
            "proof_kind": self.proof_kind,
            "provenance": [list(item) for item in self.provenance],
            "source_basis": self.source_basis,
            "source_dimension": self.source_dimension,
            "source_projection": list(self.source_projection),
        }


@dataclass(frozen=True)
class RecurrenceLCColorSourceSeedContract:
    """Compiler-owned LC forest seed for one external source current."""

    operation: RecurrenceLCColorSourceSeedOperation
    output_shape_kind: RecurrenceLCColorShapeKind
    component_kind: RecurrenceLCColorComponentKind | None
    component_role: RecurrenceLCColorComponentRole
    proof_digest: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in {"empty", "singleton"}:
            raise ValueError(
                f"unsupported recurrence LC source operation {self.operation!r}"
            )
        if self.output_shape_kind not in {
            "singlet-forest",
            "fundamental-open-string",
            "antifundamental-open-string",
            "adjoint-segment",
        }:
            raise ValueError(
                f"unsupported recurrence LC source shape {self.output_shape_kind!r}"
            )
        if self.component_role not in {"active", "passive", "none"}:
            raise ValueError(
                f"unsupported recurrence LC source role {self.component_role!r}"
            )
        if self.operation == "empty":
            if self.component_kind is not None or self.component_role != "none":
                raise ValueError(
                    "an empty recurrence LC source cannot declare a component"
                )
        elif self.component_kind is None or self.component_role == "none":
            raise ValueError(
                "a singleton recurrence LC source requires component kind and role"
            )
        expected = {
            "singlet-forest": ("empty", None, "none"),
            "fundamental-open-string": ("singleton", "open-string", "active"),
            "antifundamental-open-string": (
                "singleton",
                "open-string",
                "active",
            ),
            "adjoint-segment": ("singleton", "adjoint-segment", "active"),
        }[self.output_shape_kind]
        if (self.operation, self.component_kind, self.component_role) != expected:
            raise ValueError(
                "recurrence LC source seed does not match its output shape"
            )
        if self.proof_digest is not None and (
            len(self.proof_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.proof_digest
            )
        ):
            raise ValueError(
                "recurrence LC source proof digest must be lowercase SHA-256"
            )
        if self.provenance != tuple(sorted(set(self.provenance))):
            raise ValueError(
                "recurrence LC source provenance must be sorted and unique"
            )


@dataclass(frozen=True)
class RecurrenceLCColorWitnessContract:
    """One exact ordered-word term in a model-owned LC transition contract."""

    input_permutation: tuple[int, int]
    reverse_parent_mask: int
    component_operation: RecurrenceLCColorOperation
    result_component_kind: RecurrenceLCColorComponentKind | None
    result_component_role: RecurrenceLCColorComponentRole
    exact_factor: tuple[float, float] = (1.0, 0.0)
    proof_digest: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.input_permutation not in {(0, 1), (1, 0)}:
            raise ValueError(
                "recurrence LC color input permutation must be (0, 1) or (1, 0)"
            )
        if self.reverse_parent_mask not in range(4):
            raise ValueError("recurrence LC color reversal mask must fit two inputs")
        if self.component_operation not in {
            "concatenate-join",
            "concatenate-keep",
            "inherit-left",
            "inherit-right",
            "empty",
            "close",
        }:
            raise ValueError(
                "unsupported recurrence LC color operation "
                f"{self.component_operation!r}"
            )
        if self.result_component_role not in {"active", "passive", "none"}:
            raise ValueError(
                f"unsupported recurrence LC result role {self.result_component_role!r}"
            )
        if self.component_operation == "concatenate-join":
            if (
                self.result_component_kind is None
                or self.result_component_role == "none"
            ):
                raise ValueError(
                    "an LC join requires a result component kind and "
                    "active/passive role"
                )
        elif self.component_operation == "close":
            if self.result_component_role != "none":
                raise ValueError(
                    "an LC closure cannot declare a recurrence result component"
                )
        elif (
            self.result_component_kind is not None
            or self.result_component_role
            != (
                "active"
                if self.component_operation in {"inherit-left", "inherit-right"}
                else "none"
            )
        ):
            raise ValueError(
                "LC non-join result roles must match their exact component operation"
            )
        if self.exact_factor == (0.0, 0.0):
            raise ValueError("recurrence LC color witness factor must be nonzero")
        if self.proof_digest is not None and (
            len(self.proof_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.proof_digest
            )
        ):
            raise ValueError(
                "recurrence LC color witness proof digest must be lowercase SHA-256"
            )
        if self.provenance != tuple(sorted(set(self.provenance))):
            raise ValueError(
                "recurrence LC color witness provenance must be sorted and unique"
            )


@dataclass(frozen=True)
class RecurrenceLCColorTransitionContract:
    """Closed, compiler-owned color-state contract for one local transition."""

    rule_kind: str
    witnesses: tuple[RecurrenceLCColorWitnessContract, ...]

    def __post_init__(self) -> None:
        if not self.rule_kind:
            raise ValueError("recurrence LC color rule kind must be nonempty")
        if not self.witnesses:
            raise ValueError("recurrence LC color transition requires exact witnesses")


@dataclass(frozen=True)
class VertexLoweringRule:
    kind: int
    backend: str
    tensor_names: tuple[str, ...] = ()
    expression_head: str = ""
    full_tensor_network_ready: bool = False
    description: str = ""
    kernel: str = ""
    input_roles: tuple[str, str] = ("", "")
    output_role: str = ""
    coupling_mode: str = "none"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "backend": self.backend,
            "tensor_names": list(self.tensor_names),
            "expression_head": self.expression_head,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "description": self.description,
            "kernel": self.kernel,
            "input_roles": list(self.input_roles),
            "output_role": self.output_role,
            "coupling_mode": self.coupling_mode,
        }


@dataclass(frozen=True)
class VertexEvaluationEquivalence:
    """Verified relation between oriented kernels used for evaluation reuse.

    ``factor`` states that the concrete oriented kernel equals ``factor`` times
    its equivalence-class representative after applying ``input_order``.
    ``input_exchange_factor``, when present, additionally proves that exchanging
    the representative's two inputs multiplies it by that factor. Keeping this
    model-owned lets compiled UFO models persist relations proven from their
    actual lowered Symbolica expressions.
    """

    class_id: str
    factor: tuple[float, float] = (1.0, 0.0)
    input_order: tuple[int, int] = (0, 1)
    input_exchange_factor: tuple[float, float] | None = None
    verified: bool = True

    def __post_init__(self) -> None:
        if not self.class_id:
            raise ValueError("vertex evaluation equivalence class must be non-empty")
        if self.input_order not in {(0, 1), (1, 0)}:
            raise ValueError("vertex evaluation input order must be (0, 1) or (1, 0)")
        if not all(math.isfinite(component) for component in self.factor):
            raise ValueError("vertex evaluation equivalence factor must be finite")
        if self.factor == (0.0, 0.0):
            raise ValueError("vertex evaluation equivalence factor must be nonzero")
        if self.input_exchange_factor is not None:
            if not all(
                math.isfinite(component) for component in self.input_exchange_factor
            ):
                raise ValueError(
                    "vertex evaluation input-exchange factor must be finite"
                )
            if self.input_exchange_factor == (0.0, 0.0):
                raise ValueError(
                    "vertex evaluation input-exchange factor must be nonzero"
                )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "factor": list(self.factor),
            "input_order": list(self.input_order),
            "input_exchange_factor": (
                None
                if self.input_exchange_factor is None
                else list(self.input_exchange_factor)
            ),
            "verified": self.verified,
        }


@dataclass(frozen=True)
class PropagatorLoweringRule:
    particle_id: int
    chirality: int
    backend: str
    full_tensor_network_ready: bool
    applies_propagator: bool
    kernel: str
    kind: PropagatorKind
    mass_class: PropagatorMassClass
    gauge: PropagatorGauge | None = None
    numerator: str | None = None
    denominator: str | None = None
    custom_source: str | None = None
    auxiliary_policy: str | None = None
    goldstone_policy: GoldstonePolicy = "not-applicable"
    description: str = ""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "particle_id": self.particle_id,
            "chirality": self.chirality,
            "backend": self.backend,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "applies_propagator": self.applies_propagator,
            "kernel": self.kernel,
            "kind": self.kind,
            "mass_class": self.mass_class,
            "gauge": self.gauge,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "custom_source": self.custom_source,
            "auxiliary_policy": self.auxiliary_policy,
            "goldstone_policy": self.goldstone_policy,
            "description": self.description,
        }


@dataclass
class Model:
    name: str
    compiled: Any | None = field(default=None, repr=False, compare=False)
    particles: dict[int, Particle] = field(default_factory=dict)
    vertices: tuple[Vertex, ...] = ()
    _source_ir_by_particle: dict[int, SourceIR] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _propagator_ir_by_state: dict[tuple[int, int], PropagatorIR] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _direct_contraction_ir_by_state: dict[
        tuple[int, int, int, int], ContractionIR | None
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _closure_contraction_ir_by_state: dict[tuple[int, int], ContractionIR | None] = (
        field(
            default_factory=dict,
            init=False,
            repr=False,
            compare=False,
        )
    )

    @cached_property
    def _species_by_pdg(self) -> dict[int, Particle]:
        species: dict[int, Particle] = {}
        for particle in self.particles.values():
            species.setdefault(particle.pdg, particle)
            species.setdefault(particle.anti_pdg, particle)
        return species

    @cached_property
    def _property_sign_by_pdg(self) -> dict[int, int]:
        signs: dict[int, int] = {}
        for particle in self.particles.values():
            signs.setdefault(particle.pdg, 1)
            if particle.anti_pdg != particle.pdg:
                signs.setdefault(particle.anti_pdg, -1)
            else:
                signs.setdefault(particle.anti_pdg, 1)
        return signs

    @cached_property
    def _vertices_by_input(self) -> dict[tuple[str, int, int], tuple[Vertex, ...]]:
        return {}

    def particle(self, pdg: int) -> Particle:
        species = self._species_by_pdg.get(pdg)
        if species is None:
            raise KeyError(f"particle not in model: {pdg}")
        return species

    def anti_particle(self, pdg: int) -> int:
        particle = self.particle(pdg)
        return particle.anti_pdg if particle.pdg == pdg else particle.pdg

    def mass(self, pdg: int) -> float:
        return self.particle(pdg).mass

    def width(self, pdg: int) -> float:
        return self.particle(pdg).width

    def spin(self, pdg: int) -> int:
        spin = self.particle(pdg).spin
        if spin < 0:
            raise ValueError(f"spin is ill-defined for particle {pdg}")
        return spin

    def dimension(self, pdg: int) -> int:
        return self.particle(pdg).dimension

    def charge(self, pdg: int) -> float:
        return self._property_sign(pdg) * self.particle(pdg).charge

    def weak_isospin_l(self, pdg: int) -> float:
        return self._property_sign(pdg) * self.particle(pdg).weak_isospin[0]

    def weak_isospin_r(self, pdg: int) -> float:
        return self._property_sign(pdg) * self.particle(pdg).weak_isospin[1]

    def color_rep(self, pdg: int) -> int:
        del pdg
        raise NotImplementedError("model does not define colour representations")

    def color_dim(self, pdg: int) -> int:
        return abs(self.color_rep(pdg))

    def is_fermion(self, pdg: int) -> bool:
        del pdg
        raise NotImplementedError("model does not define a fermion role")

    def is_chiral_eligible(self, pdg: int) -> bool:
        del pdg
        raise NotImplementedError("model does not define chiral source eligibility")

    def is_fundamental_colored_fermion(self, pdg: int) -> bool:
        del pdg
        raise NotImplementedError(
            "model does not define a fundamental colored-fermion role"
        )

    def is_massless_adjoint_vector(self, pdg: int) -> bool:
        del pdg
        raise NotImplementedError(
            "model does not define a massless adjoint-vector role"
        )

    def is_singlet(self, pdg: int) -> bool:
        return self.color_rep(pdg) == 1

    def vertex_lowering_rule(self, kind: int) -> VertexLoweringRule:
        raise NotImplementedError

    def vertex_evaluation_equivalence(
        self,
        kind: int,
    ) -> VertexEvaluationEquivalence:
        """Return a conservative, always-safe evaluation equivalence class.

        Models with compiled symbolic kernels may override this with broader
        classes.  A class unique to one model type and vertex kind still enables
        exact fan-out reuse when several DAG attachments use the same kernel and
        input currents.
        """

        model_type = f"{type(self).__module__}.{type(self).__qualname__}"
        return VertexEvaluationEquivalence(class_id=f"{model_type}:{int(kind)}")

    def vertex_contact_orbit_contracts(
        self,
        kind: int,
    ) -> tuple[
        tuple[CompiledContactOrbitCertificate, ...],
        tuple[CompiledContactOrbitStep, ...],
    ]:
        """Return compiler-owned contact-path orbit proofs when they exist."""

        del kind
        return (), ()

    def global_helicity_flip_equivalence_is_proven(
        self,
        vertices: Sequence[Vertex],
    ) -> bool:
        """Return whether the model proves parity pairing for these vertices.

        Coupling-order names and particle labels are not such a proof. External
        models therefore remain fail-closed until their compiler records an
        expression-level equivalence certificate.
        """

        del vertices
        return False

    def pure_massless_adjoint_helicity_zero_rule_is_proven(
        self,
        process: Any,
        vertices: Sequence[Vertex],
    ) -> bool:
        """Return whether all-equal and one-opposite helicities vanish.

        This tree-amplitude identity requires more than massless adjoint-vector
        external states. Generic and external models therefore remain
        fail-closed until their compiler can provide an expression-level proof.
        """

        del process, vertices
        return False

    def adjoint_current_reflection_phase(
        self,
        vertex: Vertex,
    ) -> tuple[float, float] | None:
        """Return a proven two-input adjoint-current reflection phase.

        A non-null result certifies the local off-shell identity obtained by
        exchanging the two inputs of ``vertex``.  This is deliberately
        separate from full-amplitude LC trace reflection: a pure gauge
        subcurrent may be reusable inside a mixed process even when the full
        process has no trace-reflection reduction.  Generic models remain
        fail-closed until their lowered component expressions prove the
        identity.
        """

        del vertex
        return None

    def lc_trace_reflection_equivalence_is_proven(self, process: Any) -> bool:
        """Return whether one LC trace may represent its reversed ordering.

        Reflection folding is an amplitude-level identity, not a consequence of
        adjoint colour alone.  External models therefore remain fail-closed
        unless their compiled model records a proof and overrides this hook.
        """

        del process
        return False

    def shared_single_trace_color_basis_is_proven(self, process: Any) -> bool:
        """Return whether NLC/full may use the shared single-trace recursion.

        Adjoint representation metadata alone does not prove that the model's
        color tensors obey the Yang--Mills trace relations assumed by that
        optimized recursion. Generic models therefore remain fail-closed.
        """

        del process
        return False

    def adjoint_tree_color_basis_is_proven(
        self,
        process: Any,
        *,
        max_coupling_orders: Mapping[str, int] | None = None,
    ) -> bool:
        """Prove connected DDM colour, independently of helicity identities.

        Singlet exchange can produce a colour forest even when every local
        adjoint tensor is a structure constant.  Implementations must account
        for the selected interactions and coupling-order limits.
        """

        del process, max_coupling_orders
        return False

    def propagator_lowering_rule(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> PropagatorLoweringRule:
        del particle_id, chirality
        raise NotImplementedError("model does not define propagator lowering")

    def source_spin_states(self, particle_id: int) -> tuple[SourceSpinState, ...]:
        if self.is_chiral_eligible(particle_id):
            return (
                SourceSpinState(helicity=-1, chirality=-1, spin_state=-1),
                SourceSpinState(helicity=1, chirality=1, spin_state=1),
            )
        spin = self.spin(particle_id)
        if spin == 1:
            return (SourceSpinState(helicity=0, chirality=0, spin_state=0),)
        if spin == 2:
            return (
                SourceSpinState(helicity=-1, chirality=0, spin_state=-1),
                SourceSpinState(helicity=1, chirality=0, spin_state=1),
            )
        if spin == 3:
            return (
                SourceSpinState(helicity=-1, chirality=0, spin_state=-1),
                SourceSpinState(helicity=0, chirality=0, spin_state=0),
                SourceSpinState(helicity=1, chirality=0, spin_state=1),
            )
        return (SourceSpinState(helicity=0, chirality=0, spin_state=0),)

    def source_wavefunction_kind(self, particle_id: int) -> str:
        dimension = self.current_dimension(particle_id)
        if self.is_fermion(particle_id):
            return "fermion"
        if dimension == 1:
            return "scalar"
        if dimension in {2, 4}:
            return "vector"
        if dimension == 16:
            return "spin2"
        return "unknown"

    def source_orientation(self, particle_id: int) -> str:
        """Return the source orientation from the model's anti-particle relation."""

        particle = self.particle(particle_id)
        antiparticle_id = self.anti_particle(particle_id)
        if self.anti_particle(antiparticle_id) != particle_id:
            raise ValueError(
                f"particle/antiparticle relation is not involutive for source "
                f"particle {particle_id}"
            )
        if antiparticle_id == particle_id:
            orientation = "self-conjugate"
        elif particle_id == particle.pdg and antiparticle_id == particle.anti_pdg:
            orientation = "particle"
        elif particle_id == particle.anti_pdg and antiparticle_id == particle.pdg:
            orientation = "antiparticle"
        else:
            raise ValueError(
                f"source particle {particle_id} is inconsistent with its model-owned "
                "particle/antiparticle relation"
            )
        if orientation == "self-conjugate" and self.is_fermion(particle_id):
            raise ValueError(
                f"unsupported self-conjugate fermion source {particle_id}: "
                "Majorana/FNV source wavefunctions are not implemented"
            )
        return orientation

    def _particle_identity_ir(self, particle_id: int) -> ParticleIdentityIR:
        """Return canonical oriented identity without assigning an SM role."""

        particle_id = int(particle_id)
        particle = self.particle(particle_id)
        anti_particle_id = int(self.anti_particle(particle_id))
        self_conjugate = particle_id == anti_particle_id
        canonical_id = f"model:{self.name}:state:{particle_id}"
        anti_canonical_id = f"model:{self.name}:state:{anti_particle_id}"
        return ParticleIdentityIR(
            canonical_id=canonical_id,
            species_id=f"model:{self.name}:species:{particle.pdg}",
            anti_canonical_id=anti_canonical_id,
            display_name=f"pdg_{particle_id}",
            anti_display_name=f"pdg_{anti_particle_id}",
            pdg_label=particle_id,
            anti_pdg_label=anti_particle_id,
            orientation=cast(ParticleOrientation, self.source_orientation(particle_id)),
            self_conjugate=self_conjugate,
        )

    def _source_crossing_ir(self, particle_id: int) -> CrossingIR:
        """Return the model-owned transform from outgoing to incoming source."""

        particle_id = int(particle_id)
        if self.is_chiral_eligible(particle_id):
            return CrossingIR(chirality_factor=-1, spin_state_factor=-1)
        if self.source_wavefunction_kind(particle_id) in {"vector", "spin2"}:
            return CrossingIR(helicity_factor=-1, spin_state_factor=-1)
        return CrossingIR()

    def _source_ir(self, particle_id: int) -> SourceIR:
        """Return the canonical source contract for one oriented particle.

        A compiled model's structural metadata is immutable during generation.
        Caching here ensures DAG construction, physics metadata, and execution
        schemas all consume the same contract object.
        """

        particle_id = int(particle_id)
        cache = self.__dict__.setdefault("_source_ir_by_particle", {})
        cached = cache.get(particle_id)
        if cached is not None:
            return cached
        source_ir = self._build_source_ir(particle_id)
        return cache.setdefault(particle_id, source_ir)

    def _build_source_ir(self, particle_id: int) -> SourceIR:
        """Build an uncached source contract for an oriented particle."""

        states = tuple(
            SourceStateIR(
                helicity=int(state.helicity),
                chirality=int(state.chirality),
                spin_state=state.spin_state,
            )
            for state in self.source_spin_states(particle_id)
        )
        dimensions = {
            int(self.current_dimension(particle_id, state.chirality))
            for state in states
        }
        if len(dimensions) != 1:
            raise ValueError(
                f"source {particle_id} has state-dependent component dimensions: "
                f"{sorted(dimensions)}"
            )
        wavefunction_family = self.source_wavefunction_kind(particle_id)
        if wavefunction_family not in {
            "scalar",
            "fermion",
            "vector",
            "spin2",
            "ghost",
            "auxiliary",
        }:
            raise ValueError(
                f"source {particle_id} has unsupported wavefunction family "
                f"{wavefunction_family!r}"
            )
        auxiliary = self.auxiliary_kind(particle_id)
        statistics: ParticleStatistics
        if auxiliary is not None:
            statistics = "auxiliary"
        elif self.spin(particle_id) < 0:
            statistics = "ghost"
        elif self.is_fermion(particle_id):
            statistics = "fermion"
        else:
            statistics = "boson"
        return SourceIR(
            identity=self._particle_identity_ir(particle_id),
            statistics=statistics,
            wavefunction_family=cast(WavefunctionFamily, wavefunction_family),
            component_dimension=dimensions.pop(),
            states=states,
            crossing=self._source_crossing_ir(particle_id),
            basis=self._current_basis(particle_id, states[0].chirality),
            mass_parameter=self._runtime_parameter_name(particle_id, "mass"),
            width_parameter=self._runtime_parameter_name(particle_id, "width"),
        )

    def _propagator_ir(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> PropagatorIR:
        """Project the active propagator lowering into a typed model contract."""

        particle_id = int(particle_id)
        chirality = int(chirality)
        key = (particle_id, chirality)
        cache = self.__dict__.setdefault("_propagator_ir_by_state", {})
        cached = cache.get(key)
        if cached is not None:
            return cached
        rule = self.propagator_lowering_rule(particle_id, chirality)
        auxiliary_policy = rule.auxiliary_policy
        if not rule.applies_propagator:
            auxiliary_policy = (
                auxiliary_policy or self.auxiliary_kind(particle_id) or rule.kernel
            )
        default_numerator, default_denominator = _propagator_formula_metadata(
            rule.kind, rule.gauge
        )
        result = PropagatorIR(
            identity=self._particle_identity_ir(particle_id),
            chirality=chirality,
            kind=rule.kind,
            backend=rule.backend,
            basis=self._current_basis(particle_id, chirality),
            applies_propagator=rule.applies_propagator,
            kernel=rule.kernel,
            full_tensor_network_ready=rule.full_tensor_network_ready,
            mass_class=rule.mass_class,
            gauge=rule.gauge,
            numerator=rule.numerator or default_numerator,
            denominator=rule.denominator or default_denominator,
            mass_parameter=self._runtime_parameter_name(particle_id, "mass"),
            width_parameter=self._runtime_parameter_name(particle_id, "width"),
            custom_source=rule.custom_source,
            auxiliary_policy=auxiliary_policy,
            goldstone_policy=rule.goldstone_policy,
            description=rule.description,
        )
        return cache.setdefault(key, result)

    def _current_basis(self, particle_id: int, chirality: int = 0) -> str:
        dimension = int(self.current_dimension(particle_id, chirality))
        auxiliary = self.auxiliary_kind(particle_id)
        if auxiliary is not None:
            return f"auxiliary:{auxiliary}"
        if self.is_fermion(particle_id):
            return "weyl-chiral" if dimension == 2 else "dirac"
        family = self.source_wavefunction_kind(particle_id)
        if family == "scalar" and dimension == 1:
            return "scalar"
        if family == "vector" and dimension == 4:
            return "lorentz-vector"
        if family == "spin2" and dimension == 16:
            return "lorentz-rank-2"
        if dimension == 6:
            return "antisymmetric-lorentz-pair"
        return f"components:{dimension}"

    def _runtime_parameter_name(self, particle_id: int, kind: str) -> str | None:
        provider = getattr(self, f"runtime_{kind}_parameter_name", None)
        if not callable(provider):
            return None
        name = provider(int(particle_id))
        return None if name is None else str(name)

    def runtime_normalization_payload(self, dag: Any) -> dict[str, object]:
        """Return model-owned averaging, symmetry, color, and coupling factors."""

        del dag
        raise NotImplementedError("model does not define runtime normalization")

    def runtime_normalization_parameter_defaults(self) -> Mapping[str, float]:
        """Return mutable runtime parameters used by normalization only."""

        return {}

    def recurrence_quantum_flow_contract(
        self,
        vertex: Vertex,
        left_particle_id: int,
        right_particle_id: int,
    ) -> RecurrenceQuantumFlowContract:
        """Declare the finite dynamic-flow law used by recurrence generation."""

        del vertex, left_particle_id, right_particle_id
        raise NotImplementedError(
            "model does not declare a recurrence quantum-flow contract"
        )

    def recurrence_current_embedding_contract(
        self,
        particle_id: int,
        source_chirality: int,
    ) -> RecurrenceCurrentEmbeddingContract:
        """Declare how a source state embeds into a runtime full current."""

        del particle_id, source_chirality
        raise NotImplementedError(
            "model does not declare recurrence current-embedding contracts"
        )

    def _standard_recurrence_current_embedding_contract(
        self,
        particle_id: int,
        source_chirality: int,
    ) -> RecurrenceCurrentEmbeddingContract:
        """Return the standard Weyl-to-Dirac current embedding."""

        particle = int(particle_id)
        chirality = int(source_chirality)
        if chirality not in {-1, 1} or not self.is_chiral_eligible(particle):
            raise NotImplementedError(
                "standard recurrence current embedding requires a massless "
                "chiral fermion source"
            )
        source_dimension = int(self.current_dimension(particle, chirality))
        full_dimension = int(self.current_dimension(particle, 0))
        source_basis = self._current_basis(particle, chirality)
        full_basis = self._current_basis(particle, 0)
        if (
            source_dimension != 2
            or full_dimension != 4
            or source_basis != "weyl-chiral"
            or full_basis != "dirac"
        ):
            raise NotImplementedError(
                "standard recurrence current embedding supports a two-component "
                "Weyl source in a four-component Dirac current"
            )
        embedding = (
            (0, 1, None, None)
            if chirality == 1
            else (None, None, 0, 1)
        )
        return RecurrenceCurrentEmbeddingContract(
            input_embedding=embedding,
            source_dimension=source_dimension,
            full_dimension=full_dimension,
            source_basis=source_basis,
            full_basis=full_basis,
            proof_kind="standard-weyl-dirac-current-ordering-v1",
            provenance=(
                ("chirality", f"{chirality:+d}"),
                ("contract", "standard-weyl-dirac-current-ordering-v1"),
                (
                    "particle-identity",
                    self._particle_identity_ir(particle).canonical_id,
                ),
            ),
        )

    def _standard_recurrence_quantum_flow_contract(
        self,
        vertex: Vertex,
        left_particle_id: int,
        right_particle_id: int,
    ) -> RecurrenceQuantumFlowContract:
        """Return the exact contract implemented by ``combine_flavour_flow``."""

        result_particle = int(vertex.particles[2])
        left_particle = int(left_particle_id)
        right_particle = int(right_particle_id)
        if self.is_fermion(result_particle):
            if self.is_fermion(left_particle):
                operation: RecurrenceFlavourFlowOperation = "append-left-result"
            elif self.is_fermion(right_particle):
                operation = "append-right-result"
            else:
                operation = "constant-result"
        else:
            operation = "constant-result"
        return RecurrenceQuantumFlowContract(
            flavour_flow_operation=operation,
            quantum_number_flow_operation="particle-static-result",
        )

    def recurrence_lc_color_shape_contract(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> RecurrenceLCColorShapeKind:
        """Declare the ordered partial-color shape of one current state."""

        del particle_id, chirality
        raise NotImplementedError(
            "model does not declare recurrence LC color-shape contracts"
        )

    def recurrence_lc_source_color_contract(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> RecurrenceLCColorSourceSeedContract:
        """Declare the exact LC forest seeded by one external source."""

        del particle_id, chirality
        raise NotImplementedError(
            "model does not declare recurrence LC source-color contracts"
        )

    def _standard_recurrence_lc_source_color_contract(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> RecurrenceLCColorSourceSeedContract:
        shape = self._standard_recurrence_lc_color_shape_contract(
            particle_id,
            chirality,
        )
        representation = int(self.color_rep(int(particle_id)))
        provenance = tuple(
            sorted(
                (
                    ("contract", "standard-lc-source-seed-v1"),
                    ("color-representation", str(representation)),
                    ("shape", shape),
                )
            )
        )
        if shape == "singlet-forest":
            return RecurrenceLCColorSourceSeedContract(
                "empty",
                shape,
                None,
                "none",
                provenance=provenance,
            )
        component_kind: RecurrenceLCColorComponentKind = (
            "adjoint-segment" if shape == "adjoint-segment" else "open-string"
        )
        return RecurrenceLCColorSourceSeedContract(
            "singleton",
            shape,
            component_kind,
            "active",
            provenance=provenance,
        )

    def _standard_recurrence_lc_color_shape_contract(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> RecurrenceLCColorShapeKind:
        del chirality
        representation = int(self.color_rep(int(particle_id)))
        if representation == 1:
            return "singlet-forest"
        if representation == 3:
            return "fundamental-open-string"
        if representation == -3:
            return "antifundamental-open-string"
        if representation == 8:
            return "adjoint-segment"
        raise NotImplementedError(
            "standard recurrence LC color shapes support only representations "
            f"1, 3, -3, and 8; got {representation}"
        )

    def recurrence_lc_color_transition_contract(
        self,
        vertex: Vertex,
        *,
        closure: bool,
    ) -> RecurrenceLCColorTransitionContract:
        """Declare exact ordered partial-color operations for one transition."""

        del vertex, closure
        raise NotImplementedError(
            "model does not declare recurrence LC color-transition contracts"
        )

    def _standard_recurrence_lc_color_transition_contract(
        self,
        vertex: Vertex,
        *,
        closure: bool,
    ) -> RecurrenceLCColorTransitionContract:
        representations = tuple(
            int(self.color_rep(particle)) for particle in vertex.particles
        )
        rule_kind = str(self.vertex_color_structure(vertex))
        if rule_kind in {"model-defined", "generic-tensor"}:
            absolute = tuple(abs(value) for value in representations)
            if all(value == 1 for value in absolute):
                rule_kind = "singlet"
            else:
                raise NotImplementedError(
                    "colored recurrence tensors require an exact model-owned "
                    "color projection; got an unresolved tensor for "
                    f"representations {representations!r}"
                )
        if closure:
            if any(abs(value) != 1 for value in representations):
                raise NotImplementedError(
                    "colored recurrence closures require compiler-owned LC "
                    "closure terms"
                )
            return RecurrenceLCColorTransitionContract(
                rule_kind,
                (
                    RecurrenceLCColorWitnessContract(
                        input_permutation=(0, 1),
                        reverse_parent_mask=0,
                        component_operation="close",
                        result_component_kind=None,
                        result_component_role="none",
                    ),
                ),
            )
        if rule_kind != "singlet":
            raise NotImplementedError(
                "colored recurrence transitions require compiler-owned ordered-word "
                f"terms; generic rule {rule_kind!r} is validation metadata only"
            )
        if rule_kind == "singlet":
            return RecurrenceLCColorTransitionContract(
                rule_kind,
                (
                    RecurrenceLCColorWitnessContract(
                        input_permutation=(0, 1),
                        reverse_parent_mask=0,
                        component_operation="empty",
                        result_component_kind=None,
                        result_component_role="none",
                    ),
                ),
            )
        raise NotImplementedError(f"unsupported recurrence LC color rule {rule_kind!r}")

    def allowed_quantum_flows(
        self,
        vertex: Vertex,
        left_index: Any,
        right_index: Any,
    ) -> tuple[QuantumFlow, ...]:
        result_particle = vertex.particles[2]
        chiralities = _model_vertex_result_chiralities(
            self,
            vertex,
            left_index,
            right_index,
        )
        flows: list[QuantumFlow] = []
        for chirality in chiralities:
            flows.append(
                QuantumFlow(
                    chirality=chirality,
                    spin_state=self.result_spin_state(result_particle, chirality),
                    flavour_flow=self.combine_flavour_flow(
                        result_particle,
                        left_index,
                        right_index,
                    ),
                    quantum_number_flow=self.quantum_number_flow(result_particle),
                    coupling=vertex.coupling,
                )
            )
        return tuple(flows)

    def combine_flavour_flow(
        self,
        result_particle: int,
        left_index: Any,
        right_index: Any,
    ) -> tuple[int, ...]:
        left_pdg = _index_particle_id(left_index)
        right_pdg = _index_particle_id(right_index)
        left_flow = _index_flavour_flow(left_index)
        right_flow = _index_flavour_flow(right_index)

        if self.is_fermion(result_particle):
            if self.is_fermion(left_pdg):
                return _append_flavour_transition(left_flow, result_particle)
            if self.is_fermion(right_pdg):
                return _append_flavour_transition(right_flow, result_particle)

        return (result_particle,)

    def result_spin_state(self, particle_id: int, chirality: int) -> int:
        if self.is_fermion(particle_id):
            return chirality
        return 0

    def current_dimension(self, particle_id: int, chirality: int = 0) -> int:
        if chirality != 0 and self.is_chiral_eligible(particle_id):
            return 2
        try:
            return self.dimension(particle_id)
        except KeyError:
            return 0

    def direct_contraction_ir(
        self,
        left_particle_id: int,
        right_particle_id: int,
        left_chirality: int = 0,
        right_chirality: int = 0,
    ) -> ContractionIR | None:
        """Return an explicitly installed contraction for two current states."""

        key = (
            int(left_particle_id),
            int(left_chirality),
            int(right_particle_id),
            int(right_chirality),
        )
        return self._direct_contraction_ir_by_state.get(key)

    def closure_contraction_ir(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> ContractionIR | None:
        """Return an explicitly installed one-current closure projection."""

        key = (int(particle_id), int(chirality))
        return self._closure_contraction_ir_by_state.get(key)

    def direct_contraction_possible(
        self,
        left_particle_id: int,
        right_particle_id: int,
    ) -> bool:
        """Over-approximate closure using only declared contraction records."""

        left = int(left_particle_id)
        right = int(right_particle_id)
        return any(
            value is not None and key[0] == left and key[2] == right
            for key, value in self._direct_contraction_ir_by_state.items()
        )

    def quantum_number_flow(self, particle_id: int) -> QuantumNumberFlow:
        del particle_id
        raise NotImplementedError("model does not define exact quantum-number flow")

    def auxiliary_kind(self, particle_id: int) -> str | None:
        del particle_id
        return None

    def vertex_component_expression(
        self,
        kind: int,
        left: Sequence[Any],
        right: Sequence[Any],
        *,
        result_particle_id: int,
        result_chirality: int,
        left_chirality: int = 0,
        right_chirality: int = 0,
        coupling: tuple[Any, Any] = (1.0, 0.0),
        left_momentum: Sequence[Any] | None = None,
        right_momentum: Sequence[Any] | None = None,
    ) -> tuple[Any, ...]:
        raise NotImplementedError

    def propagator_component_expression(
        self,
        particle_id: int,
        value: Sequence[Any],
        momentum: Sequence[Any],
        *,
        chirality: int = 0,
        propagator: PropagatorIR | None = None,
    ) -> tuple[Any, ...]:
        del propagator
        raise NotImplementedError

    def iter_vertices(self, *, color_accuracy: str = "lc") -> tuple[Vertex, ...]:
        if color_accuracy == "lc":
            return tuple(self.vertices)
        if color_accuracy in {"nlc", "full"}:
            return tuple(self.vertices)
        raise ValueError(f"unknown colour accuracy: {color_accuracy}")

    def vertices_for_inputs(
        self,
        left_pdg: int,
        right_pdg: int,
        *,
        color_accuracy: str = "lc",
    ) -> tuple[Vertex, ...]:
        key = (color_accuracy, int(left_pdg), int(right_pdg))
        if key not in self._vertices_by_input:
            self._vertices_by_input[key] = tuple(
                vertex
                for vertex in self.iter_vertices(color_accuracy=color_accuracy)
                if vertex.particles[0] == left_pdg and vertex.particles[1] == right_pdg
            )
        return self._vertices_by_input[key]

    def vertices_accepting(
        self,
        left_pdg: int,
        right_pdg: int,
        *,
        color_accuracy: str = "lc",
    ) -> tuple[Vertex, ...]:
        """Return model vertices for a local current-current combination.

        This is the process-generic name used by the DAG compiler.  It is a
        thin alias around the model table lookup, but keeping the name at the
        model boundary prevents production code from classifying whole process
        families before asking which local interactions are allowed.
        """

        return self.vertices_for_inputs(
            left_pdg,
            right_pdg,
            color_accuracy=color_accuracy,
        )

    def skip_duplicate_vertex_orientation(self, vertex: Vertex) -> bool:
        """Return whether a mirrored table entry should be skipped by DAG sweeps.

        Generic DAG generation asks the model because duplicated orientations are
        a model-table convention, not a process-family rule.
        """

        return False

    def vertex_closure_allowed(self, vertex: Vertex) -> bool:
        """Return whether a scalar vertex result represents a vacuum closure.

        The legacy built-in model uses auxiliary scalar result particles for a
        few contact interactions. Generic UFO particles are ordinary currents
        unless their compiled lowering explicitly introduces such an auxiliary.
        """

        return True

    def vertex_coupling_orders(self, vertex: Vertex) -> CouplingOrders:
        """Return model-generic coupling-order increments for one vertex.

        The keys intentionally mirror UFO-style coupling-order names.  They are
        used only as local model metadata, so DAG pruning can cap e.g. QCD or
        QED order without recognizing whole process families.
        """

        del vertex
        return ()

    def coupling_order_hierarchies(self) -> dict[str, int]:
        """Return UFO-style priorities used by minimal-order generation.

        A lower hierarchy value is preferred. Models without explicit order
        metadata conservatively assign equal priority to every observed order.
        """

        return {}

    def vertex_color_weight(
        self,
        vertex: Vertex,
        *,
        color_accuracy: str,
    ) -> tuple[float, float]:
        """Return the model-owned coefficient for one projected color vertex."""

        del vertex, color_accuracy
        return (1.0, 0.0)

    def vertex_color_structure(self, vertex: Vertex) -> str:
        """Return the model-level color tensor family for local projection.

        Built-in kernels already encode their projected color algebra. External
        tensor models override this hook so the generic color engine can apply
        flow-dependent identities without recognizing particles or processes.
        """

        del vertex
        return "model-defined"

    def vertex_is_internal_contact_fragment(self, vertex: Vertex) -> bool:
        del vertex
        return False

    def combine_coupling_orders(
        self,
        left_index: Any,
        right_index: Any,
        vertex: Vertex,
    ) -> CouplingOrders:
        totals: dict[str, int] = {}
        for orders in (
            _index_coupling_orders(left_index),
            _index_coupling_orders(right_index),
            self.vertex_coupling_orders(vertex),
        ):
            for name, value in orders:
                totals[str(name).upper()] = totals.get(str(name).upper(), 0) + int(
                    value
                )
        return tuple(sorted((name, value) for name, value in totals.items() if value))

    def current_allowed(self, index: Any) -> bool:
        """Return whether a generated current index is valid in this model."""

        try:
            particle_id = int(index.particle_id)
            chirality = int(getattr(index, "chirality", 0))
            if chirality != 0 and self.is_chiral_eligible(particle_id):
                return True
            return self.dimension(particle_id) > 0
        except (KeyError, TypeError, ValueError):
            return False

    def _species_particle(self, pdg: int) -> Particle | None:
        return self._species_by_pdg.get(pdg)

    def _property_sign(self, pdg: int) -> int:
        try:
            return self._property_sign_by_pdg[pdg]
        except KeyError as exc:
            raise KeyError(f"particle not in model: {pdg}") from exc


def _propagator_formula_metadata(
    kind: PropagatorKind,
    gauge: PropagatorGauge | None,
) -> tuple[str | None, str | None]:
    """Return descriptive formulas for the implemented propagator kernel."""

    if kind == "identity":
        return ("identity", "1")
    if kind == "custom":
        return (None, None)
    if kind == "weyl-fermion":
        return ("i*weyl_slash(momentum)", "momentum_squared")
    if kind == "dirac-fermion":
        return (
            "i*(dirac_slash(momentum)+oriented_mass)",
            "momentum_squared-mass_squared+i*mass*width",
        )
    if kind == "vector" and gauge == "feynman":
        return ("-i*metric", "momentum_squared")
    if kind == "vector" and gauge == "unitary":
        return (
            "-i*(metric-momentum_outer/mass_squared)",
            "momentum_squared-mass_squared+i*mass*width",
        )
    if kind == "spin2":
        return (
            "i*spin2_projector",
            "momentum_squared-mass_squared+i*mass*width",
        )
    if kind == "scalar":
        return ("i", "momentum_squared-mass_squared+i*mass*width")
    return (None, None)


from .expressions import (  # noqa: E402
    _append_flavour_transition,
    _index_coupling_orders,
    _index_flavour_flow,
    _index_particle_id,
    _model_vertex_result_chiralities,
)

__all__ = [
    "ContractionIR",
    "CouplingOrders",
    "Model",
    "Particle",
    "PropagatorLoweringRule",
    "QuantumFlow",
    "QuantumNumberFlow",
    "RecurrenceCurrentEmbeddingContract",
    "RecurrenceFlavourFlowOperation",
    "RecurrenceLCColorComponentKind",
    "RecurrenceLCColorComponentRole",
    "RecurrenceLCColorOperation",
    "RecurrenceLCColorShapeKind",
    "RecurrenceLCColorSourceSeedContract",
    "RecurrenceLCColorSourceSeedOperation",
    "RecurrenceLCColorTransitionContract",
    "RecurrenceLCColorWitnessContract",
    "RecurrenceQuantumFlowContract",
    "RecurrenceQuantumNumberFlowOperation",
    "SourceSpinState",
    "Vertex",
    "VertexEvaluationEquivalence",
    "VertexLoweringRule",
]
