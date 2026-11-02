# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, cast

from .._physics_ir import ContractionIR, ParticleIdentityIR, ParticleOrientation
from ..base import (
    CouplingOrders,
    Model,
    PropagatorLoweringRule,
    QuantumNumberFlow,
    RecurrenceCurrentEmbeddingContract,
    RecurrenceLCColorComponentKind,
    RecurrenceLCColorOperation,
    RecurrenceLCColorShapeKind,
    RecurrenceLCColorSourceSeedContract,
    RecurrenceLCColorTransitionContract,
    RecurrenceLCColorWitnessContract,
    RecurrenceQuantumFlowContract,
    Vertex,
)
from .definitions import BuiltinSMDefinitionMixin
from .expressions import _builtin_vertex_result_chiralities
from .lowering import BuiltinSMLoweringMixin
from .process_catalog import PDGS

_ELECTRIC_CHARGE_BY_PARTICLE = {
    1: "-1/3",
    2: "2/3",
    3: "-1/3",
    4: "2/3",
    5: "-1/3",
    6: "2/3",
    11: "-1",
    13: "-1",
    15: "-1",
    24: "1",
    26: "1",
}


class BuiltinModel(Model):
    """Legacy built-in-model conventions isolated from external UFO models."""

    @cached_property
    def _color_rep_by_pdg(self) -> dict[int, int]:
        reps: dict[int, int] = {}
        for pdg, particle in self._species_by_pdg.items():
            color = particle.color_rep
            if self._property_sign_by_pdg[pdg] < 0 and abs(color) == 3:
                reps[pdg] = -color
            else:
                reps[pdg] = color
        return reps

    def color_rep(self, pdg: int) -> int:
        try:
            return self._color_rep_by_pdg[pdg]
        except KeyError as exc:
            raise KeyError(f"particle not in model: {pdg}") from exc

    def is_quark(self, pdg: int) -> bool:
        return 1 <= pdg <= 6

    def is_antiquark(self, pdg: int) -> bool:
        return -6 <= pdg <= -1

    def is_lepton(self, pdg: int) -> bool:
        return 11 <= pdg <= 16

    def is_antilepton(self, pdg: int) -> bool:
        return -16 <= pdg <= -11

    def is_fermion(self, pdg: int) -> bool:
        return (
            self.is_quark(pdg)
            or self.is_antiquark(pdg)
            or self.is_lepton(pdg)
            or self.is_antilepton(pdg)
        )

    def is_chiral_eligible(self, pdg: int) -> bool:
        return self.is_fermion(pdg) and self.mass(pdg) == 0.0

    def _vertex_result_chiralities(
        self,
        vertex: Vertex,
        left_index: Any,
        right_index: Any,
    ) -> tuple[int, ...]:
        return _builtin_vertex_result_chiralities(
            self,
            vertex,
            left_index,
            right_index,
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
        return self._standard_recurrence_current_embedding_contract(
            particle_id,
            source_chirality,
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
        rule_kind = self.vertex_color_structure(vertex)
        representations = tuple(
            self.color_rep(particle) for particle in vertex.particles
        )
        if closure:
            if all(abs(value) == 1 for value in representations[:2]):
                closure_kind = None
            elif {representations[0], representations[1]} == {3, -3}:
                closure_kind = "open-string"
            elif representations[0] == representations[1] == 8:
                closure_kind = "trace"
            else:
                raise ValueError(
                    "built-in recurrence closure has unsupported color inputs "
                    f"{representations[:2]!r}"
                )
            return RecurrenceLCColorTransitionContract(
                rule_kind,
                (
                    RecurrenceLCColorWitnessContract(
                        input_permutation=(0, 1),
                        reverse_parent_mask=0,
                        component_operation="close",
                        result_component_kind=closure_kind,
                        result_component_role="none",
                    ),
                ),
            )

        def witness(
            permutation: tuple[int, int],
            operation: RecurrenceLCColorOperation,
            result_component: RecurrenceLCColorComponentKind | None,
            factor: tuple[float, float] = (1.0, 0.0),
        ) -> RecurrenceLCColorWitnessContract:
            result_role = (
                "passive"
                if operation == "concatenate-join" and abs(representations[2]) == 1
                else "active"
                if operation
                in {"concatenate-join", "inherit-left", "inherit-right"}
                else "none"
            )
            return RecurrenceLCColorWitnessContract(
                input_permutation=permutation,
                reverse_parent_mask=0,
                component_operation=operation,
                result_component_kind=result_component,
                result_component_role=result_role,
                exact_factor=factor,
            )

        if rule_kind == "adjoint-structure-constant":
            return RecurrenceLCColorTransitionContract(
                rule_kind,
                (
                    witness((0, 1), "concatenate-join", "adjoint-segment"),
                    witness(
                        (1, 0),
                        "concatenate-join",
                        "adjoint-segment",
                        (-1.0, 0.0),
                    ),
                ),
            )
        if rule_kind == "fundamental-generator":
            orientation = {
                # The open color word is always fundamental -> adjoints ->
                # antifundamental, independently of the Lorentz input order.
                4: ((1, 0), "open-string"),
                5: ((0, 1), "open-string"),
                6: ((0, 1), "open-string"),
                7: ((1, 0), "open-string"),
                9: ((1, 0), "adjoint-segment"),
            }.get(vertex.kind)
            if orientation is None:
                raise ValueError(
                    f"built-in fundamental-generator kernel {vertex.kind} has no "
                    "compiler-owned LC orientation"
                )
            permutation, result_component = orientation
            return RecurrenceLCColorTransitionContract(
                rule_kind,
                (witness(permutation, "concatenate-join", result_component),),
            )
        if rule_kind == "color-identity":
            colored_inputs = tuple(
                index
                for index, representation in enumerate(representations[:2])
                if abs(representation) != 1
            )
            if len(colored_inputs) == 0:
                operation = "empty"
                permutation = (0, 1)
                result_component = None
            elif len(colored_inputs) == 1:
                operation = (
                    "inherit-left" if colored_inputs[0] == 0 else "inherit-right"
                )
                permutation = (0, 1)
                result_component = None
            elif len(colored_inputs) == 2 and abs(representations[2]) == 1:
                operation = "concatenate-join"
                permutation = (
                    (0, 1)
                    if representations[0] == 3 or representations[1] == -3
                    else (1, 0)
                )
                result_component = "open-string"
            else:
                raise ValueError(
                    "built-in color-identity kernel has unsupported recurrence "
                    f"representations {representations!r}"
                )
            return RecurrenceLCColorTransitionContract(
                rule_kind,
                (witness(permutation, operation, result_component),),
            )
        if rule_kind == "singlet":
            return RecurrenceLCColorTransitionContract(
                rule_kind,
                (witness((0, 1), "empty", None),),
            )
        raise ValueError(
            f"built-in kernel {vertex.kind} has unsupported recurrence color rule "
            f"{rule_kind!r}"
        )

    def vertex_color_structure(self, vertex: Vertex) -> str:
        """Return the exact built-in tensor family for one oriented vertex."""

        representations = tuple(
            self.color_rep(particle) for particle in vertex.particles
        )
        absolute = tuple(abs(value) for value in representations)
        if all(value == 1 for value in absolute):
            return "singlet"
        if all(value == 8 for value in representations):
            if vertex.kind not in {0, 1, 2, 3}:
                raise ValueError(
                    f"built-in vertex kind {vertex.kind} has an unknown adjoint tensor"
                )
            return "adjoint-structure-constant"
        if sorted(absolute) == [1, 3, 3]:
            return "color-identity"
        if sorted(absolute) == [3, 3, 8]:
            if vertex.kind not in {4, 5, 6, 7, 9}:
                raise ValueError(
                    f"built-in vertex kind {vertex.kind} has an unknown "
                    "quark-gluon tensor"
                )
            return "fundamental-generator"
        raise ValueError(
            f"built-in vertex kind {vertex.kind} has unsupported color representations "
            f"{representations!r}"
        )

    def is_fundamental_colored_fermion(self, pdg: int) -> bool:
        return self.is_quark(pdg) or self.is_antiquark(pdg)

    def is_massless_adjoint_vector(self, pdg: int) -> bool:
        return pdg == 21

    def is_gluon(self, pdg: int) -> bool:
        return pdg in (21, 99)

    def is_singlet(self, pdg: int) -> bool:
        return not (abs(pdg) <= 6 or pdg == 21)

    def is_tensor(self, pdg: int) -> bool:
        return pdg in (-21, -23, 26, -26)

    def is_massive_boson(self, pdg: int) -> bool:
        return pdg == 23 or abs(pdg) == 24

    def is_photon(self, pdg: int) -> bool:
        return pdg == 22

    def is_higgs(self, pdg: int) -> bool:
        return pdg == 25

    def quantum_number_flow(self, particle_id: int) -> QuantumNumberFlow:
        particle = self.particle(particle_id)
        expression = _ELECTRIC_CHARGE_BY_PARTICLE.get(particle.pdg, "0")
        if expression != "0" and self._property_sign(particle_id) < 0:
            expression = (
                expression.removeprefix("-")
                if expression.startswith("-")
                else f"-{expression}"
            )
        return (("electric_charge", expression),)

    def _particle_identity_ir(self, particle_id: int) -> ParticleIdentityIR:
        particle_id = int(particle_id)
        anti_particle_id = int(self.anti_particle(particle_id))
        particle = self.particle(particle_id)
        display_name = _display_name(particle_id)
        anti_display_name = _display_name(anti_particle_id)
        species_name = _display_name(particle.pdg)
        return ParticleIdentityIR(
            canonical_id=f"model:{self.name}:state:{display_name}",
            species_id=f"model:{self.name}:species:{species_name}",
            anti_canonical_id=f"model:{self.name}:state:{anti_display_name}",
            display_name=display_name,
            anti_display_name=anti_display_name,
            pdg_label=particle_id,
            anti_pdg_label=anti_particle_id,
            orientation=cast(ParticleOrientation, self.source_orientation(particle_id)),
            self_conjugate=particle_id == anti_particle_id,
        )

    def propagator_lowering_rule(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> PropagatorLoweringRule:
        if self.is_tensor(particle_id):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="identity",
                full_tensor_network_ready=True,
                applies_propagator=False,
                kernel="auxiliary_tensor_embedded_propagator",
                kind="identity",
                mass_class="not-applicable",
                auxiliary_policy="antisymmetric-tensor",
                description=(
                    "auxiliary-tensor propagator factors are embedded in the "
                    "adjacent built-in-SM vertex kernels"
                ),
            )
        if particle_id in (125, 126, 127):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="identity",
                full_tensor_network_ready=True,
                applies_propagator=False,
                kernel="auxiliary_scalar_no_propagator",
                kind="identity",
                mass_class="not-applicable",
                auxiliary_policy="builtin-higgsor-scalar",
                description=(
                    "Higgsor auxiliary scalar currents are non-propagating in "
                    "the built-in-SM model"
                ),
            )
        if particle_id in (21, 22, 99):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="symbolica",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="massless_vector_feynman_gauge",
                kind="vector",
                mass_class="massless",
                gauge="feynman",
                description="massless vector propagator in mostly-minus metric",
            )
        if abs(particle_id) == 24 or particle_id == 23:
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="symbolica",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="massive_vector_unitary_gauge",
                kind="vector",
                mass_class="massive",
                gauge="unitary",
                goldstone_policy="absorbed",
                description="massive vector propagator with width",
            )
        if self.is_chiral_eligible(particle_id) and chirality != 0:
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="spenso",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="weyl_fermion",
                kind="weyl-fermion",
                mass_class="massless",
                description="massless Weyl fermion propagator",
            )
        if self.is_fermion(particle_id) and (
            self.mass(particle_id) != 0.0 or chirality == 0
        ):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="symbolica",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel=(
                    "massive_dirac_fermion"
                    if self.mass(particle_id) != 0.0
                    else "massless_dirac_fermion"
                ),
                kind="dirac-fermion",
                mass_class=("massive" if self.mass(particle_id) != 0.0 else "massless"),
                description=(
                    "massive Dirac fermion propagator"
                    if self.mass(particle_id) != 0.0
                    else "massless full-Dirac fermion propagator"
                ),
            )
        if self.is_higgs(particle_id):
            return PropagatorLoweringRule(
                particle_id=particle_id,
                chirality=chirality,
                backend="symbolica",
                full_tensor_network_ready=True,
                applies_propagator=True,
                kernel="scalar_with_width",
                kind="scalar",
                mass_class="massive",
                description="scalar propagator with optional width",
            )
        return PropagatorLoweringRule(
            particle_id=particle_id,
            chirality=chirality,
            backend="unimplemented",
            full_tensor_network_ready=False,
            applies_propagator=True,
            kernel="unknown",
            kind="unsupported",
            mass_class=("massive" if self.mass(particle_id) != 0.0 else "massless"),
            description="no built-in-model propagator lowering is registered",
        )

    def auxiliary_kind(self, particle_id: int) -> str | None:
        if particle_id == 99:
            return "u1-subtraction-color-flow-vector"
        if self.is_tensor(particle_id):
            return "antisymmetric-tensor"
        return None

    def _install_builtin_contractions(self) -> None:
        """Install the built-in model's explicit current bilinear forms."""

        direct: dict[tuple[int, int, int, int], ContractionIR | None] = {}
        closure: dict[tuple[int, int], ContractionIR | None] = {}
        particle_ids = tuple(sorted(self._species_by_pdg))
        for left_particle_id in particle_ids:
            right_particle_id = self.anti_particle(left_particle_id)
            for left_chirality in self._builtin_contraction_chiralities(
                left_particle_id
            ):
                for right_chirality in self._builtin_contraction_chiralities(
                    right_particle_id
                ):
                    contraction_ir = self._builtin_direct_contraction_ir(
                        left_particle_id,
                        right_particle_id,
                        left_chirality=left_chirality,
                        right_chirality=right_chirality,
                    )
                    if contraction_ir is not None:
                        direct[
                            (
                                left_particle_id,
                                left_chirality,
                                right_particle_id,
                                right_chirality,
                            )
                        ] = contraction_ir
            for chirality in self._builtin_contraction_chiralities(left_particle_id):
                if self.current_dimension(left_particle_id, chirality) == 1:
                    closure[(left_particle_id, chirality)] = ContractionIR(
                        name="scalar",
                        left_basis="scalar",
                        right_basis="scalar",
                        coefficients=((1.0, 0.0),),
                        chirality_relation="any",
                        metric_signature=None,
                    )
        self._direct_contraction_ir_by_state = direct
        self._closure_contraction_ir_by_state = closure

    def _builtin_contraction_chiralities(self, particle_id: int) -> tuple[int, ...]:
        if self.is_chiral_eligible(particle_id):
            return (-1, 0, 1)
        return (0,)

    def _builtin_direct_contraction_ir(
        self,
        left_particle_id: int,
        right_particle_id: int,
        *,
        left_chirality: int,
        right_chirality: int,
    ) -> ContractionIR | None:
        left_dimension = self.current_dimension(left_particle_id, left_chirality)
        right_dimension = self.current_dimension(right_particle_id, right_chirality)
        if left_dimension != right_dimension:
            return None

        chirality_relation: Literal["any", "opposite"] = "any"
        metric_signature: str | None = None
        if left_dimension == 1:
            name = "scalar"
            coefficients = ((1.0, 0.0),)
        elif left_dimension == 2:
            if left_chirality != -right_chirality:
                return None
            name = "weyl"
            coefficients = ((1.0, 0.0), (1.0, 0.0))
            chirality_relation = "opposite"
        elif left_dimension == 4:
            if self.is_fermion(left_particle_id) != self.is_fermion(right_particle_id):
                return None
            if self.is_fermion(left_particle_id):
                name = "dirac"
                coefficients = ((1.0, 0.0),) * 4
            else:
                name = "lorentz"
                coefficients = (
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                    (-1.0, 0.0),
                )
                metric_signature = "mostly-minus"
        elif left_dimension == 6:
            name = "antisymmetric-tensor"
            coefficients = ((1.0, 0.0),) * 6
        else:
            return None
        return ContractionIR(
            name=name,
            left_basis=self._current_basis(left_particle_id, left_chirality),
            right_basis=self._current_basis(right_particle_id, right_chirality),
            coefficients=coefficients,
            chirality_relation=chirality_relation,
            metric_signature=metric_signature,
        )


@dataclass
class BuiltinSMModel(BuiltinSMLoweringMixin, BuiltinSMDefinitionMixin, BuiltinModel):
    """Built-in Standard Model production path with pinned reference conventions."""

    name: str = "built-in-sm-leading-color"
    alpha_s_mz: float = 0.119
    alpha_s_me_check: float = 0.118
    alpha_ew: float = 1.0 / 132.507
    sin_weak: float = 0.47143025548407230
    sqrt_s: float = 14000.0

    def __post_init__(self) -> None:
        self.particles = {
            particle.pdg: particle for particle in self._build_particles()
        }
        self.vertices = tuple(self._build_vertices())
        self._install_builtin_contractions()

    @cached_property
    def cos_weak(self) -> float:
        return math.sqrt(1.0 - self.sin_weak**2)

    def weak_coupling(self) -> float:
        return 1.0 / self.sin_weak

    def neutral_gauge_coupling(self) -> float:
        return self.weak_coupling() * self.cos_weak

    def charged_current_coupling(self) -> float:
        return self.weak_coupling() / math.sqrt(2.0)

    def weak_coupling_over_cosine(self) -> float:
        return self.weak_coupling() / self.cos_weak

    def photon_fermion_coupling(self, pdg: int) -> tuple[float, float]:
        particle = self.particle(pdg)
        return particle.charge, particle.charge

    def z_fermion_coupling(self, pdg: int) -> tuple[float, float]:
        particle = self.particle(pdg)
        charge = particle.charge
        left = particle.weak_isospin[0]
        right = particle.weak_isospin[1]
        prefactor = self.weak_coupling_over_cosine()
        return (
            prefactor * (left - charge * self.sin_weak**2),
            prefactor * (right - charge * self.sin_weak**2),
        )

    def leading_color_factor(self, process: Iterable[int]) -> int:
        exponent_twice = 0
        for pdg in process:
            if pdg == 21:
                exponent_twice += 2
            elif 1 <= abs(pdg) <= 6:
                exponent_twice += 1
        if exponent_twice % 2:
            raise ValueError(f"non-integer leading-color exponent for {tuple(process)}")
        return 3 ** (exponent_twice // 2)

    def runtime_normalization_payload(self, dag: Any) -> dict[str, object]:
        """Preserve the established built-in-SM normalization convention."""

        initial = tuple(int(pdg) for pdg in dag.process.initial_pdgs)
        final = tuple(int(pdg) for pdg in dag.process.final_pdgs)
        average_factor = 1
        for pdg in initial:
            if pdg == 21:
                average_factor *= 16
            elif 1 <= abs(pdg) <= 6:
                average_factor *= 6
            else:
                average_factor *= 2
        identical_factor = math.prod(
            math.factorial(count) for count in Counter(final).values()
        )
        electroweak_power = (
            max(1, len(dag.process.singlet_labels)) if dag.process.singlet_labels else 0
        )
        qcd_power = max(0, len(dag.process.legs) - 2 - electroweak_power)
        return {
            "color_accuracy": dag.process.color_accuracy,
            "color_factor": int(self.leading_color_factor((*initial, *final))),
            "average_factor": average_factor,
            "identical_factor": identical_factor,
            "global_coupling_factor": (
                (4.0 * math.pi * self.alpha_s_me_check) ** qcd_power
                * (2.0 * 4.0 * math.pi * self.alpha_ew) ** electroweak_power
            ),
            "qcd_coupling_power": qcd_power,
            "electroweak_coupling_power": electroweak_power,
            "couplings_in_stage_evaluators": True,
            "coupling_policy": "stage evaluators include local vertex couplings",
        }

    def runtime_normalization_parameter_defaults(self) -> dict[str, float]:
        """Expose the built-in SM coupling normalization as runtime inputs."""

        return {
            "normalization.alpha_s_me_check": float(self.alpha_s_me_check),
            "normalization.alpha_ew": float(self.alpha_ew),
        }

    def skip_duplicate_vertex_orientation(self, vertex: Vertex) -> bool:
        """Skip mirrored model-table entries already covered by DAG sweeps."""

        return False

    def vertex_coupling_orders(self, vertex: Vertex) -> CouplingOrders:
        """Classify built-in SM vertices by UFO-style coupling order."""

        if vertex.kind in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
            return (("QCD", 1),)
        return (("QED", 1),)

    def global_helicity_flip_equivalence_is_proven(
        self,
        vertices: Iterable[Vertex],
    ) -> bool:
        """Use the pinned built-in-SM QCD kernel inventory as the proof source."""

        for vertex in vertices:
            orders = self.vertex_coupling_orders(vertex)
            if not orders and self.vertex_is_internal_contact_fragment(vertex):
                continue
            if not orders or any(name != "QCD" for name, _value in orders):
                return False
        return True

    def pure_massless_adjoint_helicity_zero_rule_is_proven(
        self,
        process: Any,
        vertices: Iterable[Vertex],
    ) -> bool:
        """Prove the tree-level Yang--Mills helicity-zero rule.

        The proof is deliberately tied to the built-in model's registered
        massless-adjoint and auxiliary-tensor kernels. A QCD coupling-order
        label alone would also admit quark vertices and is not sufficient.
        """

        legs = tuple(getattr(process, "legs", ()))
        if not legs or any(
            getattr(leg, "outgoing_pdg", None) is None
            or not self.is_massless_adjoint_vector(int(leg.outgoing_pdg))
            or self.mass(int(leg.outgoing_pdg)) != 0.0
            for leg in legs
        ):
            return False

        pure_gauge_inventory = {
            vertex
            for vertex in self.vertices
            if all(
                self.is_massless_adjoint_vector(particle_id)
                or self.auxiliary_kind(particle_id) == "antisymmetric-tensor"
                for particle_id in vertex.particles
            )
        }
        used_vertices = tuple(vertices)
        return bool(used_vertices) and all(
            vertex in pure_gauge_inventory for vertex in used_vertices
        )

    def adjoint_current_reflection_phase(
        self,
        vertex: Vertex,
    ) -> tuple[float, float] | None:
        """Expose the pinned three-gauge-boson current antisymmetry."""

        if vertex.kind != 0 or not all(
            self.is_massless_adjoint_vector(particle_id)
            for particle_id in vertex.particles
        ):
            return None
        return (-1.0, 0.0)

    def lc_trace_reflection_equivalence_is_proven(self, process: Any) -> bool:
        """Prove reflection folding for the pinned pure-gluon implementation."""

        legs = tuple(getattr(process, "legs", ()))
        return bool(legs) and all(
            getattr(leg, "outgoing_pdg", None) is not None
            and self.is_massless_adjoint_vector(int(leg.outgoing_pdg))
            for leg in legs
        )

    def shared_single_trace_color_basis_is_proven(self, process: Any) -> bool:
        """Prove the shared NLC/full trace basis for built-in Yang--Mills."""

        return self.lc_trace_reflection_equivalence_is_proven(process)

    def adjoint_tree_color_basis_is_proven(
        self,
        process: Any,
        *,
        max_coupling_orders: Mapping[str, int] | None = None,
    ) -> bool:
        """Certify only the pinned implementation's connected pure-gluon trees."""

        del max_coupling_orders
        # Subclasses may change the interaction inventory and must prove their
        # own certificate rather than inherit the pinned SM's reflection hook.
        return (
            type(self) is BuiltinSMModel
            and self.lc_trace_reflection_equivalence_is_proven(process)
        )

    def coupling_order_hierarchies(self) -> dict[str, int]:
        return {"QCD": 1, "QED": 2}


def _display_name(particle_id: int) -> str:
    candidates = [name for name, pdg in PDGS.items() if int(pdg) == particle_id]
    if not candidates:
        return f"pdg_{particle_id}"
    return min(candidates, key=lambda name: (len(name), name))


__all__ = ["BuiltinModel", "BuiltinSMModel"]
