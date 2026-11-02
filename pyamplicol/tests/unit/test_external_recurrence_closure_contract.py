# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace

import pytest

from pyamplicol.models.base import RecurrenceLCColorShapeKind, Vertex
from pyamplicol.models.compiler_color_flow import (
    compile_lc_color_transition_terms,
)
from pyamplicol.models.contracts import CompiledOrientedKernel
from pyamplicol.models.external_catalog import ExternalModelCatalogMixin


class _ExternalCatalog(ExternalModelCatalogMixin):
    def __init__(
        self,
        kernel: CompiledOrientedKernel,
        shapes: dict[int, RecurrenceLCColorShapeKind],
    ) -> None:
        self._kernels = {kernel.kind: kernel}
        self._shapes = shapes

    def _kernel(self, kind: int) -> CompiledOrientedKernel:
        return self._kernels[kind]

    def recurrence_lc_color_shape_contract(
        self,
        particle_id: int,
        chirality: int = 0,
    ) -> RecurrenceLCColorShapeKind:
        del chirality
        return self._shapes[particle_id]


def _compiled_kernel(
    structure: str,
    representations: tuple[int, int, int],
) -> CompiledOrientedKernel:
    kernel = CompiledOrientedKernel(
        kind=7,
        term_id=3,
        vertex="V_1",
        particles=("left", "right", "result"),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="exact-color-source",
        color_expression="exact-color-expression",
        color_projection_structure=structure,
        color_projection_coefficient=(1.0, 0.0),
    )
    terms = compile_lc_color_transition_terms(
        kernel,
        representations,
        proof_source="test-exact-projection",
        provenance=(("model-compiler", "test"),),
    )
    return replace(kernel, lc_color_transition_terms=terms)


@pytest.mark.parametrize(
    ("structure", "representations", "shapes", "closure_kind"),
    (
        (
            "color-identity",
            (3, -3, 1),
            {
                1: "fundamental-open-string",
                2: "antifundamental-open-string",
                3: "singlet-forest",
            },
            "open-string",
        ),
        (
            "adjoint-structure-constant",
            (8, 8, 8),
            {1: "adjoint-segment", 2: "adjoint-segment", 3: "adjoint-segment"},
            "trace",
        ),
        (
            "singlet",
            (1, 1, 1),
            {1: "singlet-forest", 2: "singlet-forest", 3: "singlet-forest"},
            None,
        ),
    ),
)
def test_external_catalog_consumes_compiler_owned_lc_closure_terms(
    structure: str,
    representations: tuple[int, int, int],
    shapes: dict[int, RecurrenceLCColorShapeKind],
    closure_kind: str | None,
) -> None:
    kernel = _compiled_kernel(structure, representations)
    catalog = _ExternalCatalog(kernel, shapes)

    contract = catalog.recurrence_lc_color_transition_contract(
        Vertex(7, (1, 2, 3)),
        closure=True,
    )

    assert contract.rule_kind == structure
    assert len(contract.witnesses) == 1
    witness = contract.witnesses[0]
    assert witness.component_operation == "close"
    assert witness.result_component_kind == closure_kind
    assert witness.result_component_role == "none"
    assert witness.exact_factor == (1.0, 0.0)
    assert ("contract-kind", "closure") in witness.provenance
    assert ("model-compiler", "test") in witness.provenance


def test_old_external_bundle_fails_recurrence_closure_preflight_clearly() -> None:
    kernel = _compiled_kernel("color-identity", (3, -3, 1))
    old_kernel = replace(kernel, lc_color_closure_terms=())
    catalog = _ExternalCatalog(
        old_kernel,
        {
            1: "fundamental-open-string",
            2: "antifundamental-open-string",
            3: "singlet-forest",
        },
    )

    with pytest.raises(
        NotImplementedError,
        match=r"no compiler-owned LC color closure terms.*model compile",
    ):
        catalog.recurrence_lc_color_transition_contract(
            Vertex(7, (1, 2, 3)),
            closure=True,
        )
