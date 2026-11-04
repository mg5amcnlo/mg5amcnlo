# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ._physics_ir import ParticleIdentityIR, ParticleOrientation
from .base import (
    Particle,
    QuantumFlow,
    QuantumNumberFlow,
    RecurrenceCurrentEmbeddingContract,
    RecurrenceLCColorComponentKind,
    RecurrenceLCColorOperation,
    RecurrenceLCColorShapeKind,
    RecurrenceLCColorSourceSeedContract,
    RecurrenceLCColorTransitionContract,
    RecurrenceLCColorWitnessContract,
    RecurrenceQuantumFlowContract,
    SourceSpinState,
    Vertex,
    VertexEvaluationEquivalence,
    VertexLoweringRule,
)
from .contact_decomposition import (
    CompiledContactOrbitCertificate,
    CompiledContactOrbitStep,
)

if TYPE_CHECKING:
    pass

from .external_helpers import _spin_dimension


class ExternalModelCatalogMixin:
    def particle(self, pdg: int) -> Particle:
        try:
            return self.particles[int(pdg)]
        except KeyError as exc:
            raise KeyError(f"particle not in model: {pdg}") from exc

    def vertices_for_inputs(
        self,
        left_pdg: int,
        right_pdg: int,
        *,
        color_accuracy: str = "lc",
    ) -> tuple[Vertex, ...]:
        if color_accuracy not in {"lc", "nlc", "full"}:
            raise ValueError(f"unknown colour accuracy: {color_accuracy}")
        return self._compiled_vertices_by_input.get(
            (int(left_pdg), int(right_pdg)),
            (),
        )

    def anti_particle(self, pdg: int) -> int:
        return self.particle(pdg).anti_pdg

    def mass(self, pdg: int) -> Any:
        return self._real_parameter_value(
            self._particle_records_by_pdg[int(pdg)].mass,
            field="mass",
        )

    def width(self, pdg: int) -> Any:
        return self._real_parameter_value(
            self._particle_records_by_pdg[int(pdg)].width,
            field="width",
        )

    def spin(self, pdg: int) -> int:
        return self._particle_records_by_pdg[int(pdg)].spin

    def dimension(self, pdg: int) -> int:
        record = self._particle_records_by_pdg[int(pdg)]
        if record.component_dimension is not None:
            return record.component_dimension
        return _spin_dimension(record.spin)

    def current_dimension(self, particle_id: int, chirality: int = 0) -> int:
        if chirality != 0 and self.is_chiral_eligible(particle_id):
            return 2
        return self.dimension(particle_id)

    def color_rep(self, pdg: int) -> int:
        return self._particle_records_by_pdg[int(pdg)].color

    def color_dim(self, pdg: int) -> int:
        return abs(self.color_rep(pdg))

    def charge(self, pdg: int) -> float:
        return self._particle_records_by_pdg[int(pdg)].charge

    def quantum_number_flow(self, particle_id: int) -> QuantumNumberFlow:
        return self._particle_records_by_pdg[int(particle_id)].quantum_numbers

    def is_fermion(self, pdg: int) -> bool:
        return self._particle_records_by_pdg[int(pdg)].statistics == "fermion"

    def is_chiral_eligible(self, pdg: int) -> bool:
        if not self.is_fermion(pdg):
            return False
        propagator = self._propagator_record(pdg)
        if propagator is not None and propagator.custom:
            return False
        particle = self._particle_records_by_pdg[int(pdg)]
        if particle.mass.upper() == "ZERO":
            return True
        record = self._parameter_records.get(particle.mass)
        return (
            record is not None
            and record.nature != "external"
            and self._parameter_default(particle.mass) == 0.0
        )

    def is_fundamental_colored_fermion(self, pdg: int) -> bool:
        return self.is_fermion(pdg) and abs(self.color_rep(pdg)) == 3

    def is_massless_adjoint_vector(self, pdg: int) -> bool:
        return (
            self.spin(pdg) == 3
            and self.color_rep(pdg) == 8
            and complex(
                self._parameter_default(self._particle_records_by_pdg[int(pdg)].mass)
            ).real
            == 0.0
        )

    def is_singlet(self, pdg: int) -> bool:
        return self.color_rep(pdg) == 1

    def source_spin_states(self, particle_id: int) -> tuple[SourceSpinState, ...]:
        if self.is_chiral_eligible(particle_id):
            return super().source_spin_states(particle_id)
        spin = self.spin(particle_id)
        massive = (
            complex(
                self._parameter_default(
                    self._particle_records_by_pdg[int(particle_id)].mass
                )
            ).real
            != 0.0
        )
        if spin == 1:
            helicities = (0,)
        elif spin == 2:
            helicities = (-1, 1)
        elif spin == 3:
            helicities = (-1, 0, 1) if massive else (-1, 1)
        elif spin == 5:
            helicities = (-2, -1, 0, 1, 2) if massive else (-2, 2)
        else:
            raise ValueError(
                f"unsupported source spin {spin} for particle {particle_id}"
            )
        return tuple(
            SourceSpinState(helicity=helicity, chirality=0, spin_state=helicity)
            for helicity in helicities
        )

    def source_wavefunction_kind(self, particle_id: int) -> str:
        return self._particle_records_by_pdg[int(particle_id)].wavefunction_family

    def source_orientation(self, particle_id: int) -> str:
        record = self._particle_records_by_pdg[int(particle_id)]
        if record.statistics == "fermion" and record.self_conjugate:
            raise ValueError(
                f"unsupported self-conjugate fermion source {particle_id}: "
                "Majorana/FNV source wavefunctions are not implemented"
            )
        return record.source_orientation

    def _particle_identity_ir(self, particle_id: int) -> ParticleIdentityIR:
        record = self._particle_records_by_pdg[int(particle_id)]
        anti_record = self._particle_records_by_name[record.antiname]
        species_record = (
            record
            if record.source_orientation in {"particle", "self-conjugate"}
            else anti_record
        )
        return ParticleIdentityIR(
            canonical_id=f"model:{self.name}:state:{record.name}",
            species_id=f"model:{self.name}:species:{species_record.name}",
            anti_canonical_id=f"model:{self.name}:state:{anti_record.name}",
            display_name=record.name,
            anti_display_name=anti_record.name,
            pdg_label=int(record.pdg_code),
            anti_pdg_label=int(anti_record.pdg_code),
            orientation=cast(ParticleOrientation, record.source_orientation),
            self_conjugate=bool(record.self_conjugate),
        )

    def allowed_quantum_flows(
        self,
        vertex: Vertex,
        left_index: Any,
        right_index: Any,
    ) -> tuple[QuantumFlow, ...]:
        result_particle = vertex.particles[2]
        left_chirality = int(getattr(left_index, "chirality", 0))
        right_chirality = int(getattr(right_index, "chirality", 0))
        result_chiralities = (
            (0,)
            if (
                self.is_chiral_eligible(result_particle)
                and left_chirality == 0
                and right_chirality == 0
            )
            else (
                (-1, 1)
                if self.is_chiral_eligible(result_particle)
                else (0,)
            )
        )
        return tuple(
            QuantumFlow(
                chirality=result_chirality,
                spin_state=(
                    result_chirality if self.is_chiral_eligible(result_particle) else 0
                ),
                flavour_flow=self.combine_flavour_flow(
                    result_particle,
                    left_index,
                    right_index,
                ),
                quantum_number_flow=self.quantum_number_flow(result_particle),
                coupling=(1.0, 0.0),
            )
            for result_chirality in result_chiralities
            if self._weyl_projection_is_nonzero(
                vertex.kind,
                left_chirality,
                right_chirality,
                result_chirality,
            )
        )

    def recurrence_quantum_flow_contract(
        self,
        vertex: Vertex,
        left_particle_id: int,
        right_particle_id: int,
    ) -> RecurrenceQuantumFlowContract:
        return self._standard_recurrence_quantum_flow_contract(
            vertex,
            left_particle_id,
            right_particle_id,
        )

    def recurrence_current_embedding_contract(
        self,
        particle_id: int,
        source_chirality: int,
    ) -> RecurrenceCurrentEmbeddingContract:
        particle = self._particle_records_by_pdg[int(particle_id)]
        chirality = int(source_chirality)
        ordering = next(
            (
                record
                for record in self.compiled.ir.current_orderings
                if record.particle == particle.name
                and int(record.chirality) == chirality
            ),
            None,
        )
        if ordering is None:
            raise NotImplementedError(
                "compiled model has no current-ordering contract for "
                f"particle={particle.name!r}, chirality={chirality:+d}; "
                "re-run the original `pyamplicol model compile ...` command "
                "with the current pyAmpliCol compiler"
            )
        return RecurrenceCurrentEmbeddingContract(
            input_embedding=tuple(ordering.input_embedding),
            source_dimension=int(self.current_dimension(particle_id, chirality)),
            full_dimension=int(self.current_dimension(particle_id, 0)),
            source_basis=self._current_basis(particle_id, chirality),
            full_basis=self._current_basis(particle_id, 0),
            proof_kind="compiled-current-ordering-embedding-v1",
            provenance=tuple(
                sorted(
                    (
                        ("chirality", f"{chirality:+d}"),
                        ("kernel-ordering-id", ordering.kernel_ordering_id),
                        ("ordering-id", ordering.ordering_id),
                        ("particle", particle.name),
                    )
                )
            ),
        )

    def recurrence_lc_color_shape_contract(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> RecurrenceLCColorShapeKind:
        return self._standard_recurrence_lc_color_shape_contract(
            particle_id,
            chirality,
        )

    def recurrence_lc_source_color_contract(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> RecurrenceLCColorSourceSeedContract:
        return self._standard_recurrence_lc_source_color_contract(
            particle_id,
            chirality,
        )

    def recurrence_lc_color_transition_contract(
        self,
        vertex: Vertex,
        *,
        closure: bool,
    ) -> RecurrenceLCColorTransitionContract:
        kernel = self._kernel(vertex.kind)
        terms = (
            kernel.lc_color_closure_terms
            if closure
            else kernel.lc_color_transition_terms
        )
        contract_kind = "closure" if closure else "transition"
        if not terms:
            raise NotImplementedError(
                f"compiled kernel {kernel.kind} has no compiler-owned LC color "
                f"{contract_kind} terms; re-run the original `pyamplicol model "
                "compile ...` command with the current pyAmpliCol compiler"
            )
        input_shapes = tuple(
            self.recurrence_lc_color_shape_contract(particle)
            for particle in vertex.particles[:2]
        )
        result_shape = (
            None
            if closure
            else self.recurrence_lc_color_shape_contract(vertex.particles[2])
        )
        witnesses = []
        for term in terms:
            if (
                term.input_shape_kinds != input_shapes
                or term.result_shape_kind != result_shape
            ):
                raise ValueError(
                    f"compiled kernel {kernel.kind} LC color {contract_kind} term "
                    "does not match its live oriented particle-state shapes"
                )
            if (term.component_operation == "close") != closure:
                expectation = (
                    "a closure operation"
                    if closure
                    else "a non-closure operation"
                )
                raise ValueError(
                    f"compiled kernel {kernel.kind} LC color {contract_kind} term "
                    f"must contain {expectation}"
                )
            factor = {
                "1": (1.0, 0.0),
                "-1": (-1.0, 0.0),
            }.get(term.exact_factor_expression)
            if factor is None:
                raise NotImplementedError(
                    f"compiled kernel {kernel.kind} LC witness factor "
                    f"{term.exact_factor_expression!r} is not supported by "
                    "recurrence v1"
                )
            witnesses.append(
                RecurrenceLCColorWitnessContract(
                    input_permutation=term.input_permutation,
                    reverse_parent_mask=term.reverse_parent_mask,
                    component_operation=cast(
                        RecurrenceLCColorOperation,
                        term.component_operation,
                    ),
                    result_component_kind=cast(
                        RecurrenceLCColorComponentKind | None,
                        term.result_component_kind,
                    ),
                    result_component_role=term.result_component_role,
                    exact_factor=factor,
                    proof_digest=term.proof_digest,
                    provenance=term.provenance,
                )
            )
        return RecurrenceLCColorTransitionContract(
            str(
                kernel.lc_recurrence_color_rule_kind
                or kernel.color_projection_structure
            ),
            tuple(witnesses),
        )

    def vertex_lowering_rule(self, kind: int) -> VertexLoweringRule:
        kernel = self._kernel(kind)
        return VertexLoweringRule(
            kind=kind,
            backend="spenso-ufo",
            tensor_names=(kernel.vertex,),
            expression_head="compiled_ufo_kernel",
            full_tensor_network_ready=True,
            description="typed and oriented UFO tensor kernel",
            kernel="compiled_ufo_kernel",
            input_roles=(kernel.particles[0], kernel.particles[1]),
            output_role=kernel.particles[2],
            coupling_mode="external-model-parameters",
        )

    def vertex_evaluation_equivalence(
        self,
        kind: int,
    ) -> VertexEvaluationEquivalence:
        kernel = self._kernel(kind)
        if not kernel.evaluation_equivalence_verified or not kernel.evaluation_class:
            return super().vertex_evaluation_equivalence(kind)
        input_order = tuple(int(value) for value in kernel.evaluation_input_order)
        if input_order not in {(0, 1), (1, 0)}:
            raise ValueError(
                f"compiled UFO kernel {kind} has invalid evaluation input order "
                f"{input_order}"
            )
        return VertexEvaluationEquivalence(
            class_id=kernel.evaluation_class,
            factor=kernel.evaluation_factor,
            input_order=input_order,
            input_exchange_factor=(kernel.evaluation_input_exchange_factor),
            verified=True,
        )

    def vertex_contact_orbit_contracts(
        self,
        kind: int,
    ) -> tuple[
        tuple[CompiledContactOrbitCertificate, ...],
        tuple[CompiledContactOrbitStep, ...],
    ]:
        kernel = self._kernel(kind)
        steps = kernel.contact_orbit_steps
        certificates = tuple(
            self._vertex_terms[term_id].contact_orbit_certificate
            for term_id in sorted({step.term_id for step in steps})
        )
        if any(certificate is None for certificate in certificates):
            raise ValueError(
                f"compiled kernel {kind} has an unpaired contact-orbit step"
            )
        return (
            tuple(
                certificate
                for certificate in certificates
                if certificate is not None
            ),
            steps,
        )

    def vertex_coupling_orders(
        self,
        vertex: Vertex,
    ) -> tuple[tuple[str, int], ...]:
        return self._kernel(vertex.kind).coupling_orders

    def global_helicity_flip_equivalence_is_proven(
        self,
        vertices: Sequence[Vertex],
    ) -> bool:
        """Accept parity reduction only for algebraically certified kernels."""

        kinds = self._symmetry_certificates.parity_kernel_kinds
        return bool(vertices) and all(vertex.kind in kinds for vertex in vertices)

    def pure_massless_adjoint_helicity_zero_rule_is_proven(
        self,
        process: Any,
        vertices: Sequence[Vertex],
    ) -> bool:
        """Recognize the tree-level Yang--Mills helicity-zero theorem."""

        names = self._symmetry_certificates.yang_mills_adjoint_names
        kinds = self._symmetry_certificates.yang_mills_kernel_kinds
        legs = tuple(getattr(process, "legs", ()))
        return (
            bool(legs)
            and bool(vertices)
            and all(self._particle_name_for_leg(leg) in names for leg in legs)
            and all(vertex.kind in kinds for vertex in vertices)
        )

    def adjoint_current_reflection_phase(
        self,
        vertex: Vertex,
    ) -> tuple[float, float] | None:
        """Return an exact lowered-kernel reflection certificate, if present."""

        return dict(self._symmetry_certificates.adjoint_current_reflection_phases).get(
            vertex.kind
        )

    def lc_trace_reflection_equivalence_is_proven(self, process: Any) -> bool:
        """Recognize reflection folding for a certified Yang--Mills sector."""

        names = self._symmetry_certificates.yang_mills_adjoint_names
        legs = tuple(getattr(process, "legs", ()))
        return bool(legs) and all(
            self._particle_name_for_leg(leg) in names for leg in legs
        )

    def shared_single_trace_color_basis_is_proven(self, process: Any) -> bool:
        """Accept shared traces only for the certified Yang--Mills sector."""

        names = self._symmetry_certificates.yang_mills_adjoint_names
        legs = tuple(getattr(process, "legs", ()))
        return bool(legs) and all(
            self._particle_name_for_leg(leg) in names for leg in legs
        )

    def adjoint_tree_color_basis_is_proven(
        self,
        process: Any,
        *,
        max_coupling_orders: Mapping[str, int] | None = None,
    ) -> bool:
        """Certify actual colour tensors without assuming Yang--Mills helicities."""

        from .adjoint_color_certificate import (
            compiled_adjoint_tree_color_basis_is_proven,
        )

        return compiled_adjoint_tree_color_basis_is_proven(
            self.compiled.ir, process, max_coupling_orders=max_coupling_orders
        )

    def _particle_name_for_leg(self, leg: Any) -> str | None:
        outgoing_pdg = getattr(leg, "outgoing_pdg", None)
        if outgoing_pdg is None:
            return None
        record = self._particle_records_by_pdg.get(int(outgoing_pdg))
        return None if record is None else record.name

    def coupling_order_hierarchies(self) -> dict[str, int]:
        return {
            str(order.name).upper(): max(1, int(order.hierarchy))
            for order in self.compiled.ir.orders
        }

    def vertex_color_weight(
        self,
        vertex: Vertex,
        *,
        color_accuracy: str,
    ) -> tuple[float, float]:
        if color_accuracy not in {"lc", "nlc", "full"}:
            raise ValueError(f"unknown colour accuracy: {color_accuracy}")
        power = self._kernel(vertex.kind).lc_color_normalization_power
        structure, coefficient = self._vertex_color_projection(vertex)
        # The possibly irrational 2**(-power/2) belongs to the symbolic
        # kernel. Recurrence colour weights are exact complex rationals.
        if structure in {
            "adjoint-structure-constant",
            "adjoint-structure-constant-product",
        }:
            phase = (-1j) ** power
            weight = coefficient * phase
            return (weight.real, weight.imag)
        return (coefficient.real, coefficient.imag)

    def vertex_color_structure(self, vertex: Vertex) -> str:
        return self._vertex_color_projection(vertex)[0]

    def _vertex_color_projection(self, vertex: Vertex) -> tuple[str, complex]:
        cached = self._color_projection_cache.get(vertex.kind)
        if cached is not None:
            return cached
        kernel = self._kernel(vertex.kind)
        if (
            kernel.color_projection_structure is not None
            and kernel.color_projection_coefficient is not None
        ):
            projected = (
                kernel.color_projection_structure,
                complex(*kernel.color_projection_coefficient),
            )
        else:
            raise ValueError(
                f"compiled external-model kernel {kernel.kind} has no certified "
                "color-flow projection; recompile the UFO/JSON model with the "
                "current pyAmpliCol version"
            )
        self._color_projection_cache[vertex.kind] = projected
        return projected

    def vertex_is_internal_contact_fragment(self, vertex: Vertex) -> bool:
        return "::contact-" in self._kernel(vertex.kind).vertex

    def vertex_closure_allowed(self, vertex: Vertex) -> bool:
        del vertex
        return False
