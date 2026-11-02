# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._physics_ir import PropagatorIR
from ..base import (
    VertexEvaluationEquivalence,
    VertexLoweringRule,
)
from ..expressions import (
    _as_expression,
    _expr_antifermion_propagator_dirac,
    _expr_antifermion_propagator_weyl,
    _expr_fermion_propagator_dirac,
    _expr_fermion_propagator_weyl,
    _expr_minkowski_dot,
    _minkowski_square_expression,
    _number,
)
from .expressions import (
    _embed_weyl_current_in_dirac,
    _exact_imaginary_inverse_sqrt_two,
    _expr_fermion_antifermion_to_vector_dirac,
    _expr_fermion_antifermion_to_vector_weyl,
    _expr_fermion_scalar_to_fermion,
    _expr_fermion_vector_dirac,
    _expr_fermion_vector_weyl,
    _expr_tensor_vector_to_vector,
    _expr_three_vector_current,
    _expr_three_vector_current_coupled,
    _expr_two_vector_to_tensor,
    _expr_vector_tensor_to_vector,
    _flat_index,
    _gluon_tensor_to_gluon_data,
    _imaginary_inverse_sqrt_two,
    _quark_vector_weyl_data,
    _tensor_gluon_to_gluon_data,
    _two_gluon_to_tensor_data,
)
from .symbols import symbols


def _as_dirac_current(
    current: tuple[Any, ...],
    chirality: int,
) -> tuple[Any, ...]:
    if len(current) == 4:
        return current
    return _embed_weyl_current_in_dirac(current, chirality)


class BuiltinSMLoweringMixin:
    def build_tensor_library(self) -> Any:
        from symbolica.community.tensor import (
            Representation,
            Tensor,
            TensorLibrary,
            TensorName,
        )

        library = TensorLibrary.hep_lib_atom()
        mink = Representation.mink(4)
        antisym = Representation(symbols.antisymmetric_lorentz_pair_name, 6)
        weyl = Representation(symbols.weyl_spinor_name, 2)
        two_gluon_to_tensor = TensorName(symbols.display_name("two_gluon_to_tensor"))
        tensor_gluon_to_gluon = TensorName(
            symbols.display_name("tensor_gluon_to_gluon")
        )
        gluon_tensor_to_gluon = TensorName(
            symbols.display_name("gluon_tensor_to_gluon")
        )
        quark_vector_weyl_plus = TensorName(
            symbols.display_name("quark_vector_weyl_plus")
        )
        quark_vector_weyl_minus = TensorName(
            symbols.display_name("quark_vector_weyl_minus")
        )

        library.register(
            Tensor.dense(
                two_gluon_to_tensor(mink, mink, antisym),
                _two_gluon_to_tensor_data(),
            )
        )
        library.register(
            Tensor.dense(
                tensor_gluon_to_gluon(antisym, mink, mink),
                _tensor_gluon_to_gluon_data(),
            )
        )
        library.register(
            Tensor.dense(
                gluon_tensor_to_gluon(mink, antisym, mink),
                _gluon_tensor_to_gluon_data(),
            )
        )
        library.register(
            Tensor.dense(
                quark_vector_weyl_plus(weyl, mink, weyl),
                _quark_vector_weyl_data(chirality=1),
            )
        )
        library.register(
            Tensor.dense(
                quark_vector_weyl_minus(weyl, mink, weyl),
                _quark_vector_weyl_data(chirality=-1),
            )
        )
        return library

    def vertex_lowering_rule(self, kind: int) -> VertexLoweringRule:
        if kind == 0:
            return VertexLoweringRule(
                kind=kind,
                backend="spenso",
                tensor_names=("g", symbols.display_name("current_momentum")),
                expression_head="three_gluon_current",
                full_tensor_network_ready=True,
                description="color-ordered three-gluon current",
                kernel="three_vector_current",
                input_roles=("vector", "vector"),
                output_role="vector",
                coupling_mode="fixed",
            )
        if kind == 1:
            return VertexLoweringRule(
                kind=kind,
                backend="spenso",
                tensor_names=(symbols.display_name("two_gluon_to_tensor"),),
                expression_head=symbols.display_name("two_gluon_to_tensor"),
                full_tensor_network_ready=True,
                description="two gluons to auxiliary antisymmetric tensor",
                kernel="two_vector_to_tensor",
                input_roles=("vector", "vector"),
                output_role="antisymmetric_tensor",
                coupling_mode="fixed",
            )
        if kind == 2:
            return VertexLoweringRule(
                kind=kind,
                backend="spenso",
                tensor_names=(symbols.display_name("tensor_gluon_to_gluon"),),
                expression_head=symbols.display_name("tensor_gluon_to_gluon"),
                full_tensor_network_ready=True,
                description="auxiliary tensor and gluon to gluon current",
                kernel="tensor_vector_to_vector",
                input_roles=("antisymmetric_tensor", "vector"),
                output_role="vector",
                coupling_mode="fixed",
            )
        if kind == 3:
            return VertexLoweringRule(
                kind=kind,
                backend="spenso",
                tensor_names=(symbols.display_name("gluon_tensor_to_gluon"),),
                expression_head=symbols.display_name("gluon_tensor_to_gluon"),
                full_tensor_network_ready=True,
                description="gluon and auxiliary tensor to gluon current",
                kernel="vector_tensor_to_vector",
                input_roles=("vector", "antisymmetric_tensor"),
                output_role="vector",
                coupling_mode="fixed",
            )
        if kind == 6:
            return VertexLoweringRule(
                kind=kind,
                backend="spenso",
                tensor_names=(
                    symbols.display_name("quark_vector_weyl_minus"),
                    symbols.display_name("quark_vector_weyl_plus"),
                ),
                expression_head="quark_gluon_weyl_current",
                full_tensor_network_ready=True,
                description="Weyl quark-gluon current",
                kernel="fermion_vector_to_fermion",
                input_roles=("fermion", "vector"),
                output_role="fermion",
                coupling_mode="fixed",
            )
        if kind in {4, 5, 7, 9}:
            qcd_role_map: dict[int, tuple[tuple[str, str], str, str]] = {
                4: (("vector", "fermion"), "fermion", "vector-fermion current"),
                5: (
                    ("vector", "antifermion"),
                    "antifermion",
                    "vector-antifermion current",
                ),
                7: (
                    ("antifermion", "vector"),
                    "antifermion",
                    "antifermion-vector current",
                ),
                9: (
                    ("antifermion", "fermion"),
                    "vector",
                    "antifermion-fermion vector current",
                ),
            }
            input_roles, output_role, description = qcd_role_map[kind]
            return VertexLoweringRule(
                kind=kind,
                backend="symbolica",
                expression_head="quark_gluon_weyl_current",
                full_tensor_network_ready=True,
                description=f"Weyl QCD {description}",
                kernel=(
                    "fermion_pair_to_vector"
                    if kind == 9
                    else "fermion_vector_to_fermion"
                ),
                input_roles=input_roles,
                output_role=output_role,
                coupling_mode="fixed",
            )
        if kind == 10:
            return VertexLoweringRule(
                kind=kind,
                backend="spenso",
                tensor_names=(
                    symbols.display_name("quark_vector_weyl_minus"),
                    symbols.display_name("quark_vector_weyl_plus"),
                ),
                expression_head="fermion_gauge_weyl_current",
                full_tensor_network_ready=True,
                description="Weyl fermion electroweak current with graph coupling",
                kernel="fermion_vector_to_fermion",
                input_roles=("fermion", "vector"),
                output_role="fermion",
                coupling_mode="vertex",
            )
        if kind in {11, 21, 22, 23, 24}:
            fermion_gauge_role_map: dict[
                int,
                tuple[tuple[str, str], str, str, str],
            ] = {
                11: (
                    ("antifermion", "vector"),
                    "antifermion",
                    "antifermion electroweak current",
                    "fermion_vector_to_fermion",
                ),
                21: (
                    ("fermion", "antifermion"),
                    "vector",
                    "lepton-antilepton electroweak current",
                    "fermion_pair_to_vector",
                ),
                22: (
                    ("antifermion", "fermion"),
                    "vector",
                    "antilepton-lepton electroweak current",
                    "fermion_pair_to_vector",
                ),
                23: (
                    ("vector", "fermion"),
                    "fermion",
                    "vector-fermion electroweak current",
                    "fermion_vector_to_fermion",
                ),
                24: (
                    ("vector", "antifermion"),
                    "antifermion",
                    "vector-antifermion electroweak current",
                    "fermion_vector_to_fermion",
                ),
            }
            input_roles, output_role, description, kernel = fermion_gauge_role_map[kind]
            return VertexLoweringRule(
                kind=kind,
                backend="symbolica",
                expression_head="fermion_gauge_weyl_current",
                full_tensor_network_ready=True,
                description=f"Weyl {description} with runtime coupling",
                kernel=kernel,
                input_roles=input_roles,
                output_role=output_role,
                coupling_mode="vertex",
            )
        if kind == 8:
            return VertexLoweringRule(
                kind=kind,
                backend="symbolica",
                expression_head="qcd_u1_subtraction_current",
                full_tensor_network_ready=True,
                description="QCD U(1) subtraction current",
                kernel="fermion_pair_to_vector",
                input_roles=("fermion", "antifermion"),
                output_role="vector",
                coupling_mode="vertex",
            )
        if kind in {12, 13, 14, 15}:
            vector_role_map: dict[int, tuple[str, tuple[str, str], str, str]] = {
                12: (
                    "three_vector_current",
                    ("vector", "vector"),
                    "vector",
                    "electroweak three-vector current",
                ),
                13: (
                    "two_vector_to_tensor",
                    ("vector", "vector"),
                    "antisymmetric_tensor",
                    "two electroweak vectors to auxiliary tensor",
                ),
                14: (
                    "tensor_vector_to_vector",
                    ("antisymmetric_tensor", "vector"),
                    "vector",
                    "auxiliary tensor and vector to electroweak vector",
                ),
                15: (
                    "vector_tensor_to_vector",
                    ("vector", "antisymmetric_tensor"),
                    "vector",
                    "electroweak vector and auxiliary tensor to vector",
                ),
            }
            kernel, input_roles, output_role, description = vector_role_map[kind]
            return VertexLoweringRule(
                kind=kind,
                backend="symbolica",
                expression_head=kernel,
                full_tensor_network_ready=True,
                description=f"{description} with runtime coupling",
                kernel=kernel,
                input_roles=input_roles,
                output_role=output_role,
                coupling_mode="vertex",
            )
        if kind == 16:
            return VertexLoweringRule(
                kind=kind,
                backend="symbolica",
                expression_head="fermion_scalar_to_fermion",
                description=("massive Dirac fermion-scalar Yukawa current"),
                full_tensor_network_ready=True,
                kernel="fermion_scalar_to_fermion",
                input_roles=("fermion", "scalar"),
                output_role="fermion",
                coupling_mode="vertex",
            )
        if kind in {17, 18, 19, 20}:
            scalar_role_map: dict[int, tuple[str, tuple[str, str], str, str]] = {
                17: (
                    "two_vector_to_scalar",
                    ("vector", "vector"),
                    "scalar",
                    "two vectors to scalar current",
                ),
                18: (
                    "scalar_vector_to_vector",
                    ("scalar", "vector"),
                    "vector",
                    "scalar-vector to vector current",
                ),
                19: (
                    "vector_scalar_to_vector",
                    ("vector", "scalar"),
                    "vector",
                    "vector-scalar to vector current",
                ),
                20: (
                    "two_scalar_to_scalar",
                    ("scalar", "scalar"),
                    "scalar",
                    "scalar-scalar to scalar current",
                ),
            }
            kernel, input_roles, output_role, description = scalar_role_map[kind]
            return VertexLoweringRule(
                kind=kind,
                backend="symbolica",
                expression_head=kernel,
                full_tensor_network_ready=True,
                description=f"{description} with runtime coupling",
                kernel=kernel,
                input_roles=input_roles,
                output_role=output_role,
                coupling_mode="vertex",
            )
        return VertexLoweringRule(
            kind=kind,
            backend="unimplemented",
            description="no native pyamplicol lowering rule is registered yet",
            kernel="unknown",
        )

    def vertex_evaluation_equivalence(
        self,
        kind: int,
    ) -> VertexEvaluationEquivalence:
        """Expose exact input-orientation identities of built-in kernels."""

        antisymmetric_same_inputs = {
            0: "three-vector-current",
            1: "two-vector-to-tensor",
        }
        if kind in antisymmetric_same_inputs:
            return VertexEvaluationEquivalence(
                class_id=f"builtin-sm:{antisymmetric_same_inputs[kind]}",
                input_exchange_factor=(-1.0, 0.0),
            )

        pure_gauge_orientations = {
            2: ("tensor-vector-to-vector", (0, 1), (1.0, 0.0)),
            3: ("tensor-vector-to-vector", (1, 0), (-1.0, 0.0)),
        }
        relation = pure_gauge_orientations.get(kind)
        if relation is not None:
            class_name, input_order, factor = relation
            return VertexEvaluationEquivalence(
                class_id=f"builtin-sm:{class_name}",
                factor=factor,
                input_order=input_order,
            )

        quark_gluon_orientations = {
            4: ("quark-gluon-to-quark", (1, 0)),
            6: ("quark-gluon-to-quark", (0, 1)),
            5: ("antiquark-gluon-to-antiquark", (1, 0)),
            7: ("antiquark-gluon-to-antiquark", (0, 1)),
        }
        relation = quark_gluon_orientations.get(kind)
        if relation is not None:
            class_name, input_order = relation
            return VertexEvaluationEquivalence(
                class_id=f"builtin-sm:{class_name}",
                input_order=input_order,
            )

        fermion_gauge_orientations = {
            10: ("fermion-vector-to-fermion", (0, 1)),
            23: ("fermion-vector-to-fermion", (1, 0)),
            11: ("antifermion-vector-to-antifermion", (0, 1)),
            24: ("antifermion-vector-to-antifermion", (1, 0)),
        }
        relation = fermion_gauge_orientations.get(kind)
        if relation is not None:
            class_name, input_order = relation
            return VertexEvaluationEquivalence(
                class_id=f"builtin-sm:{class_name}",
                input_order=input_order,
            )

        fermion_pair_orientations = {
            21: (0, 1),
            22: (1, 0),
        }
        input_order = fermion_pair_orientations.get(kind)
        if input_order is not None:
            return VertexEvaluationEquivalence(
                class_id="builtin-sm:fermion-pair-to-vector",
                input_order=input_order,
            )

        scalar_vector_orientations = {
            18: (0, 1),
            19: (1, 0),
        }
        input_order = scalar_vector_orientations.get(kind)
        if input_order is not None:
            return VertexEvaluationEquivalence(
                class_id="builtin-sm:scalar-vector-to-vector",
                input_order=input_order,
            )

        return super().vertex_evaluation_equivalence(kind)

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
        """Lower a local model vertex into component expressions.

        The inputs are already component tuples for the two parent currents.
        This method is intentionally process-blind: all decisions are local to
        the vertex kind, chirality labels, particle id, coupling, and optional
        current momenta.
        """

        if kind == 0:
            if left_momentum is None or right_momentum is None:
                raise ValueError("three-vector current requires parent momenta")
            return _expr_three_vector_current(
                tuple(left),
                tuple(left_momentum),
                tuple(right),
                tuple(right_momentum),
            )
        if kind == 1:
            return _expr_two_vector_to_tensor(tuple(left), tuple(right))
        if kind == 2:
            return _expr_tensor_vector_to_vector(tuple(left), tuple(right))
        if kind == 3:
            return _expr_vector_tensor_to_vector(tuple(left), tuple(right))
        if kind == 12:
            if left_momentum is None or right_momentum is None:
                raise ValueError("three-vector current requires parent momenta")
            return _expr_three_vector_current_coupled(
                tuple(left),
                tuple(left_momentum),
                tuple(right),
                tuple(right_momentum),
                coupling,
            )
        if kind == 13:
            return tuple(
                coupling[0] * component
                for component in _expr_two_vector_to_tensor(tuple(left), tuple(right))
            )
        if kind == 14:
            return tuple(
                coupling[0] * component
                for component in _expr_tensor_vector_to_vector(
                    tuple(left),
                    tuple(right),
                )
            )
        if kind == 15:
            return tuple(
                coupling[0] * component
                for component in _expr_vector_tensor_to_vector(
                    tuple(left),
                    tuple(right),
                )
            )
        if kind == 16:
            return _expr_fermion_scalar_to_fermion(
                tuple(left),
                tuple(right),
                coupling,
            )
        if kind == 17:
            return (
                _imaginary_inverse_sqrt_two(*left, *right, *coupling)
                * coupling[0]
                * _expr_minkowski_dot(tuple(left), tuple(right)),
            )
        if kind == 18:
            prefactor = _imaginary_inverse_sqrt_two(*left, *right, *coupling)
            return tuple(
                prefactor * coupling[0] * left[0] * component
                for component in tuple(right)
            )
        if kind == 19:
            prefactor = _imaginary_inverse_sqrt_two(*left, *right, *coupling)
            return tuple(
                prefactor * coupling[0] * right[0] * component
                for component in tuple(left)
            )
        if kind == 20:
            phase = 1j if coupling[1] == -10.0 else 1.0
            prefactor = _imaginary_inverse_sqrt_two(*left, *right, *coupling)
            return (prefactor * phase * coupling[0] * left[0] * right[0],)
        if kind in {4, 6}:
            fermion, vector = (
                (tuple(right), tuple(left))
                if kind == 4
                else (tuple(left), tuple(right))
            )
            if len(fermion) == 4:
                return _expr_fermion_vector_dirac(
                    fermion,
                    vector,
                    antifermion=False,
                    coupling=None,
                )
            return _expr_fermion_vector_weyl(
                fermion,
                vector,
                result_chirality,
                antifermion=False,
                coupling=None,
            )
        if kind in {5, 7}:
            antifermion, vector = (
                (tuple(right), tuple(left))
                if kind == 5
                else (tuple(left), tuple(right))
            )
            if len(antifermion) == 4:
                return _expr_fermion_vector_dirac(
                    antifermion,
                    vector,
                    antifermion=True,
                    coupling=None,
                )
            return _expr_fermion_vector_weyl(
                antifermion,
                vector,
                result_chirality,
                antifermion=True,
                coupling=None,
            )
        if kind == 9:
            fermion = tuple(right)
            antifermion = tuple(left)
            if len(fermion) == 4 or len(antifermion) == 4:
                return _expr_fermion_antifermion_to_vector_dirac(
                    fermion=_as_dirac_current(fermion, right_chirality),
                    antifermion=_as_dirac_current(antifermion, left_chirality),
                    coupling=(1.0, 1.0),
                )
            return _expr_fermion_antifermion_to_vector_weyl(
                fermion=fermion,
                antifermion=antifermion,
                coupling=(1.0, 1.0),
                fermion_chirality=right_chirality,
                antifermion_chirality=left_chirality,
            )
        if kind == 8:
            qcd_coupling = (coupling[0], coupling[0])
            fermion = tuple(left)
            antifermion = tuple(right)
            if len(fermion) == 4 or len(antifermion) == 4:
                return _expr_fermion_antifermion_to_vector_dirac(
                    fermion=_as_dirac_current(fermion, left_chirality),
                    antifermion=_as_dirac_current(antifermion, right_chirality),
                    coupling=qcd_coupling,
                )
            return _expr_fermion_antifermion_to_vector_weyl(
                fermion=fermion,
                antifermion=antifermion,
                coupling=qcd_coupling,
                fermion_chirality=left_chirality,
                antifermion_chirality=right_chirality,
            )
        if kind in {10, 23}:
            fermion, vector = (
                (tuple(left), tuple(right))
                if kind == 10
                else (tuple(right), tuple(left))
            )
            input_chirality = left_chirality if kind == 10 else right_chirality
            if len(fermion) == 4:
                return _expr_fermion_vector_dirac(
                    fermion,
                    vector,
                    antifermion=False,
                    coupling=coupling,
                )
            current = _expr_fermion_vector_weyl(
                fermion,
                vector,
                input_chirality,
                antifermion=False,
                coupling=coupling,
            )
            if self.current_dimension(result_particle_id, result_chirality) == 4:
                return _embed_weyl_current_in_dirac(current, input_chirality)
            return current
        if kind in {11, 24}:
            antifermion, vector = (
                (tuple(left), tuple(right))
                if kind == 11
                else (tuple(right), tuple(left))
            )
            input_chirality = left_chirality if kind == 11 else right_chirality
            if len(antifermion) == 4:
                return _expr_fermion_vector_dirac(
                    antifermion,
                    vector,
                    antifermion=True,
                    coupling=coupling,
                )
            current = _expr_fermion_vector_weyl(
                antifermion,
                vector,
                input_chirality,
                antifermion=True,
                coupling=coupling,
            )
            if self.current_dimension(result_particle_id, result_chirality) == 4:
                return _embed_weyl_current_in_dirac(current, input_chirality)
            return current
        if kind == 21:
            fermion = tuple(left)
            antifermion = tuple(right)
            if len(fermion) == 4 or len(antifermion) == 4:
                return _expr_fermion_antifermion_to_vector_dirac(
                    fermion=_as_dirac_current(fermion, left_chirality),
                    antifermion=_as_dirac_current(antifermion, right_chirality),
                    coupling=coupling,
                )
            return _expr_fermion_antifermion_to_vector_weyl(
                fermion=fermion,
                antifermion=antifermion,
                coupling=coupling,
                fermion_chirality=left_chirality,
                antifermion_chirality=right_chirality,
            )
        if kind == 22:
            fermion = tuple(right)
            antifermion = tuple(left)
            if len(fermion) == 4 or len(antifermion) == 4:
                return _expr_fermion_antifermion_to_vector_dirac(
                    fermion=_as_dirac_current(fermion, right_chirality),
                    antifermion=_as_dirac_current(antifermion, left_chirality),
                    coupling=coupling,
                )
            return _expr_fermion_antifermion_to_vector_weyl(
                fermion=fermion,
                antifermion=antifermion,
                coupling=coupling,
                fermion_chirality=right_chirality,
                antifermion_chirality=left_chirality,
            )
        raise ValueError(f"vertex kind {kind} has no component expression lowering")

    def propagator_component_expression(
        self,
        particle_id: int,
        value: Sequence[Any],
        momentum: Sequence[Any],
        *,
        chirality: int = 0,
        propagator: PropagatorIR | None = None,
    ) -> tuple[Any, ...]:
        metadata = propagator or self._propagator_ir(particle_id, chirality)
        if metadata.identity.canonical_id != self._particle_identity_ir(
            particle_id
        ).canonical_id or metadata.chirality != int(chirality):
            raise ValueError("propagator metadata does not match the current")
        components = tuple(value)
        current_momentum = tuple(momentum)
        if not metadata.applies_propagator:
            return components
        if metadata.kind == "vector" and metadata.gauge == "feynman":
            denominator = _minkowski_square_expression(current_momentum)
            prefactor = -1j / denominator
            return tuple(component * prefactor for component in components)
        if metadata.kind == "vector" and metadata.gauge == "unitary":
            mass = self.mass(particle_id)
            width = self.width(particle_id)
            denominator = (
                _minkowski_square_expression(current_momentum)
                - mass * mass
                + 1j * mass * width
            )
            prefactor = -1j / denominator
            longitudinal = _expr_minkowski_dot(components, current_momentum) / (
                mass * mass
            )
            return tuple(
                (components[index] - current_momentum[index] * longitudinal) * prefactor
                for index in range(4)
            )
        if metadata.kind == "weyl-fermion":
            if metadata.identity.orientation == "antiparticle":
                return _expr_antifermion_propagator_weyl(
                    components,
                    current_momentum,
                    chirality,
                )
            return _expr_fermion_propagator_weyl(
                components,
                current_momentum,
                chirality,
            )
        if metadata.kind == "dirac-fermion":
            if metadata.identity.orientation == "antiparticle":
                return _expr_antifermion_propagator_dirac(
                    components,
                    current_momentum,
                    self.mass(particle_id),
                    self.width(particle_id),
                )
            return _expr_fermion_propagator_dirac(
                components,
                current_momentum,
                self.mass(particle_id),
                self.width(particle_id),
            )
        if metadata.kind == "scalar":
            mass = self.mass(particle_id)
            width = self.width(particle_id)
            denominator = (
                _minkowski_square_expression(current_momentum)
                - mass * mass
                + 1j * mass * width
            )
            prefactor = 1j / denominator
            return tuple(component * prefactor for component in components)
        raise ValueError(
            f"propagator kind {metadata.kind!r} in gauge {metadata.gauge!r} "
            f"is not lowered for particle "
            f"{particle_id}"
        )

    def three_gluon_current_expression(
        self,
        *,
        left_slot: Any,
        right_slot: Any,
        output_slot: Any,
        left_momentum_tensor_name: str,
        right_momentum_tensor_name: str,
        dummy_prefix: str,
    ) -> Any:
        from symbolica import Expression
        from symbolica.community.tensor import Representation, TensorName

        mink = Representation.mink(4)
        metric = TensorName.g()
        left_momentum = TensorName(left_momentum_tensor_name)
        right_momentum = TensorName(right_momentum_tensor_name)
        left_dot_slot = mink(f"{dummy_prefix}_left_dot")
        right_dot_slot = mink(f"{dummy_prefix}_right_dot")
        prefactor = _exact_imaginary_inverse_sqrt_two()

        return prefactor * (
            metric(left_slot, right_slot).to_expression()
            * (
                left_momentum(output_slot).to_expression()
                - right_momentum(output_slot).to_expression()
            )
            + Expression.num(2.0)
            * (
                metric(left_slot, left_dot_slot).to_expression()
                * right_momentum(left_dot_slot).to_expression()
                * metric(right_slot, output_slot).to_expression()
                - metric(right_slot, right_dot_slot).to_expression()
                * left_momentum(right_dot_slot).to_expression()
                * metric(left_slot, output_slot).to_expression()
            )
        )

    def gluon_propagator_tensor_data(
        self,
        momentum: Sequence[Any],
    ) -> list[Any]:
        prefactor = _number(-1j) / _minkowski_square_expression(momentum)
        metric = (1.0, -1.0, -1.0, -1.0)
        data = [_number(0.0)] * (4 * 4)
        for index, sign in enumerate(metric):
            data[_flat_index((index, index), (4, 4))] = _number(sign) * prefactor
        return data

    def quark_weyl_propagator_tensor_data(
        self,
        momentum: Sequence[Any],
        *,
        chirality: int,
    ) -> list[Any]:
        prefactor = _number(1j) / _minkowski_square_expression(momentum)
        p0, p1, p2, p3 = (_as_expression(value) for value in momentum)
        data = [_number(0.0)] * (2 * 2)

        def set_entry(q_in: int, q_out: int, value: Any) -> None:
            data[_flat_index((q_in, q_out), (2, 2))] = prefactor * value

        if chirality == 1:
            set_entry(0, 0, p0 + p3)
            set_entry(1, 0, p1 + _number(1j) * p2)
            set_entry(0, 1, p1 - _number(1j) * p2)
            set_entry(1, 1, p0 - p3)
            return data
        if chirality == -1:
            set_entry(0, 0, p0 - p3)
            set_entry(1, 0, -(p1 + _number(1j) * p2))
            set_entry(0, 1, -(p1 - _number(1j) * p2))
            set_entry(1, 1, p0 + p3)
            return data
        raise ValueError(f"unsupported Weyl chirality: {chirality}")
