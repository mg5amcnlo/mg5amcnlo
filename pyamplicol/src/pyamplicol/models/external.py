# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from typing import TYPE_CHECKING, Any

from .._internal.physics.symbols import symbols
from .base import (
    Model,
    Particle,
    Vertex,
)
from .contracts import validate_color_representation

if TYPE_CHECKING:
    from .loading import CompiledModel

from . import compiler_symbolica as _sym
from .external_catalog import ExternalModelCatalogMixin
from .external_evaluation import ExternalModelEvaluationMixin
from .external_helpers import _spin_dimension
from .external_kernels import ExternalModelKernelMixin
from .external_symmetries import derive_external_symmetry_certificates


class CompiledUFOModel(
    ExternalModelCatalogMixin,
    ExternalModelEvaluationMixin,
    ExternalModelKernelMixin,
    Model,
):
    """Compiled-UFO model boundary consumed by generic DAG generation."""

    def __init__(
        self,
        compiled: CompiledModel,
        runtime_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        _sym._ensure_symbolica()
        self.compiled = compiled
        self.name = compiled.name
        self._model_symbols = symbols.model(self.name)
        self._particle_records_by_name = {
            particle.name: particle for particle in compiled.ir.particles
        }
        self._particle_records_by_pdg = {
            particle.pdg_code: particle for particle in compiled.ir.particles
        }
        self._direct_contraction_ir_by_state = {
            (
                self._particle_records_by_name[record.left_particle].pdg_code,
                record.left_chirality,
                self._particle_records_by_name[record.right_particle].pdg_code,
                record.right_chirality,
            ): record.contraction_ir
            for record in compiled.ir.direct_contractions
        }
        self._closure_contraction_ir_by_state = {
            (
                self._particle_records_by_name[record.particle].pdg_code,
                record.chirality,
            ): record.contraction_ir
            for record in compiled.ir.closure_contractions
        }
        for particle in compiled.ir.particles:
            validate_color_representation(
                particle.color,
                context=f"particle {particle.name!r}",
            )
        self._parameter_records = {
            parameter.name: parameter for parameter in compiled.ir.parameters
        }
        self._propagator_records_by_particle_name = {
            propagator.particle: propagator for propagator in compiled.ir.propagators
        }
        self._vertex_terms = {term.id: term for term in compiled.ir.vertex_terms}
        self._coupling_records_by_name = {
            coupling.name: coupling for coupling in compiled.ir.couplings
        }
        defaults = {
            name: complex(value[0], value[1])
            for name, value in compiled.parameter_defaults.items()
        }
        self._runtime_parameters = {**defaults, **dict(runtime_parameters or {})}
        self.particles = {
            record.pdg_code: Particle(
                pdg=record.pdg_code,
                anti_pdg=self._particle_records_by_name[record.antiname].pdg_code,
                spin=record.spin,
                dimension=(
                    record.component_dimension
                    if record.component_dimension is not None
                    else _spin_dimension(record.spin)
                ),
                color_rep=record.color,
                mass=float(complex(self._parameter_default(record.mass)).real),
                width=float(complex(self._parameter_default(record.width)).real),
                charge=record.charge,
            )
            for record in compiled.ir.particles
        }
        self._kernels = {kernel.kind: kernel for kernel in compiled.ir.oriented_kernels}
        self._symmetry_certificates = derive_external_symmetry_certificates(compiled.ir)
        self._kernel_component_expression_cache: dict[
            int, tuple[_sym.Expression, ...]
        ] = {}
        self._kernel_coupling_expression_cache: dict[int, _sym.Expression] = {}
        self._kernel_function_specs: dict[
            tuple[int, int, int, int],
            tuple[tuple[_sym.Expression, tuple[int, ...]], ...],
        ] = {}
        self._symbolica_kernel_functions: dict[
            tuple[_sym.Expression, tuple[_sym.Expression, ...]], _sym.Expression
        ] = {}
        self._runtime_derived_definitions = {
            parameter.name: parameter.resolved_expression
            for parameter in compiled.ir.parameters
            if parameter.nature.lower() == "internal"
            and parameter.expression is not None
        }
        self._runtime_derived_definitions.update(
            {
                name: self._vertex_terms[
                    int(name.rsplit("_", 1)[1])
                ].coupling_expression
                for kernel in self._kernels.values()
                for name in kernel.runtime_parameters
                if name.startswith("derived_coupling_")
            }
        )
        self._runtime_derived_expression_cache: dict[str, _sym.Expression] = {}
        self._runtime_derived_default_cache: dict[str, complex] = {}
        self._runtime_derived_domain_cache: dict[str, str] = {}
        self._runtime_parameter_domain_cache: dict[str, str] = {}
        self._weyl_projection_support: dict[tuple[int, int, int, int], bool] = {}
        self._custom_propagator_expressions: dict[
            int,
            tuple[_sym.Expression, tuple[str, ...]],
        ] = {}
        self._custom_propagator_templates: dict[int, tuple[_sym.Expression, ...]] = {}
        self._color_projection_cache: dict[int, tuple[str, complex]] = {}
        self._goldstone_partner_records = {
            record.goldstone: record for record in compiled.ir.goldstone_partners
        }
        for record in compiled.ir.goldstone_partners:
            if record.vector is None:
                continue
            vector = self._particle_records_by_name[record.vector]
            propagator = self._propagator_ir(vector.pdg_code)
            if propagator.goldstone_policy != record.policy:
                raise ValueError(
                    f"compiled Goldstone {record.goldstone!r} policy does not "
                    f"match vector {record.vector!r} lowering"
                )
        self.inactive_goldstone_names = frozenset(
            record.goldstone
            for record in compiled.ir.goldstone_partners
            if record.policy == "absorbed"
        )
        self.vertices = tuple(
            Vertex(
                kind=kernel.kind,
                particles=(
                    self._particle_records_by_name[kernel.particles[0]].pdg_code,
                    self._particle_records_by_name[kernel.particles[1]].pdg_code,
                    self._particle_records_by_name[kernel.particles[2]].pdg_code,
                ),
                coupling=(1.0, 0.0),
            )
            for kernel in compiled.ir.oriented_kernels
            if not any(
                name in self.inactive_goldstone_names for name in kernel.particles
            )
        )
        vertices_by_input: dict[tuple[int, int], list[Vertex]] = {}
        for vertex in self.vertices:
            vertices_by_input.setdefault(
                (vertex.particles[0], vertex.particles[1]),
                [],
            ).append(vertex)
        self._compiled_vertices_by_input = {
            key: tuple(vertices) for key, vertices in vertices_by_input.items()
        }

    def with_runtime_parameters(
        self,
        parameters: Mapping[str, Any],
    ) -> CompiledUFOModel:
        model = copy(self)
        model._runtime_parameters = {
            **self._runtime_parameters,
            **dict(parameters),
        }
        model._runtime_derived_default_cache = {}
        return model


__all__ = ["CompiledUFOModel"]
