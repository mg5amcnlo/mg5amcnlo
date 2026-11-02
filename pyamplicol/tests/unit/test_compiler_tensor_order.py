# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from pyamplicol.models import CompiledUFOModel, compile_model_source
from pyamplicol.models.compiler_kernels import _ordered_dense_tensor_components
from pyamplicol.models.compiler_tensor_ordering import (
    identity_ordering_for_materialized_axes,
)
from pyamplicol.models.contracts import CompiledOrientedKernel
from pyamplicol.models.loading import CompiledModel

SCALAR_GRAVITY_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "scalar_gravity"
)
_SPIN2_AXIS_PATTERN = re.compile(r"(?<![-0-9])(?P<axis>[12])00(?P<leg>[0-9]+)\b")
_DUMMY_INDEX_PATTERN = re.compile(r"(?P<prefix>[(,]\s*)-(?P<label>[0-9]+)(?=\s*[,)])")


class _Argument:
    def __init__(self, label: str) -> None:
        self.label = label

    def to_canonical_string(self) -> str:
        return f"spenso::mink(4,{self.label})"

    def to_expression(self) -> _Argument:
        return self


class _DenseTensor:
    def __init__(
        self,
        values: tuple[int, ...],
        axes: tuple[_Argument, ...],
        shape: tuple[int, ...],
    ) -> None:
        self.values = values
        self.axes = axes
        self.shape = shape

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        return self.values[index]


class _Tensor:
    def __init__(self) -> None:
        # Storage follows the actual (leg 4, leg 3) structure.
        self.values = (0, 10, 1, 11)
        self.coordinates = ((0, 0), (0, 1), (1, 0), (1, 1))
        self._axes = (_Argument("ufo_l_1_4"), _Argument("ufo_l_1_3"))
        self._shape = (2, 2)

    def to_dense(self) -> _DenseTensor:
        # Released Symbolica returns a new tensor in logical row-major order.
        values = tuple(
            value
            for _, value in sorted(zip(self.coordinates, self.values, strict=True))
        )
        return _DenseTensor(values, self._axes, self._shape)


def test_dense_tensor_components_follow_physical_ufo_axis_order() -> None:
    ordered = _ordered_dense_tensor_components(
        _Tensor(),
        ("ufo_l_1_3", "ufo_l_1_4"),
    )

    assert ordered.values == (0, 1, 10, 11)
    assert tuple(axis.extent for axis in ordered.ordering.axes) == (2, 2)
    assert ordered.ordering.component_basis == (0, 1, 2, 3)


def test_spenso_tensor_components_preserve_logical_axis_order() -> None:
    from symbolica import E
    from symbolica.community.tensor import Representation, Tensor, TensorName

    left = Representation.euc(2)("ufo_l_1_3")
    right = Representation.euc(3)("ufo_l_1_4")
    tensor = Tensor.dense(
        TensorName("tensor_order_actual")(right, left),
        [
            E(str(left_index * 10 + right_index))
            for right_index in range(3)
            for left_index in range(2)
        ],
    )
    ordered = _ordered_dense_tensor_components(tensor, ("ufo_l_1_3", "ufo_l_1_4"))

    assert ordered.values == tuple(E(str(value)) for value in (0, 1, 2, 10, 11, 12))
    assert tuple(axis.extent for axis in ordered.ordering.axes) == (2, 3)


class _RectangularTensor(_Tensor):
    def __init__(self) -> None:
        super().__init__()
        self.values = (12, 0, 1, 2, 10, 11)
        self.coordinates = ((2, 1), (0, 0), (1, 0), (2, 0), (0, 1), (1, 1))
        self._shape = (3, 2)


def test_dense_tensor_components_validate_non_square_cartesian_grid() -> None:
    ordered = _ordered_dense_tensor_components(
        _RectangularTensor(),
        ("ufo_l_1_3", "ufo_l_1_4"),
    )

    assert ordered.values == (0, 1, 2, 10, 11, 12)
    assert tuple(axis.extent for axis in ordered.ordering.axes) == (2, 3)
    assert ordered.ordering.canonical_size == 6


class _IncompleteTensor(_Tensor):
    def __init__(self) -> None:
        super().__init__()
        self.values = (0, 1, 10)
        self.coordinates = ((0, 0), (0, 1), (1, 0))


def test_dense_tensor_components_reject_incomplete_cartesian_grid() -> None:
    with pytest.raises(ValueError, match="complete Cartesian component grid"):
        _ordered_dense_tensor_components(
            _IncompleteTensor(),
            ("ufo_l_1_3", "ufo_l_1_4"),
        )


class _Spin2Tensor(_Tensor):
    def __init__(self) -> None:
        super().__init__()
        self.coordinates = tuple(
            reversed(tuple((right, left) for right in range(4) for left in range(4)))
        )
        self.values = tuple(
            left * 4 + right for right, left in self.coordinates
        )
        self._axes = (_Argument("ufo_l_2_3"), _Argument("ufo_l_1_3"))
        self._shape = (4, 4)


def test_spin2_axis_transpose_and_storage_permutation_canonicalize() -> None:
    labels = ("ufo_l_1_3", "ufo_l_2_3")
    ordered = _ordered_dense_tensor_components(_Spin2Tensor(), labels)

    assert ordered.values == tuple(range(16))
    assert ordered.ordering.basis == "lorentz-rank-2"
    assert (
        ordered.ordering.ordering_id
        == identity_ordering_for_materialized_axes(
            labels,
            (4, 4),
        ).ordering_id
    )


def _transpose_spin2_axes(source: str) -> str:
    return _SPIN2_AXIS_PATTERN.sub(
        lambda match: (
            ("2" if match.group("axis") == "1" else "1") + "00" + match.group("leg")
        ),
        source,
    )


def _rename_lorentz_dummies(source: str) -> str:
    return _DUMMY_INDEX_PATTERN.sub(
        lambda match: match.group("prefix") + f"-{7_000 + int(match.group('label'))}",
        source,
    )


def _kernel_numeric_values(
    model: CompiledUFOModel,
    kernel: CompiledOrientedKernel,
) -> tuple[complex, ...]:
    particles = {particle.name: particle for particle in model.compiled.ir.particles}

    def current(name: str, phase: float) -> tuple[complex, ...]:
        pdg = particles[name].pdg_code
        return tuple(
            complex(index + 1, phase * (index + 1))
            for index in range(model.current_dimension(pdg, 0))
        )

    left = current(kernel.particles[0], 0.125)
    right = current(kernel.particles[1], -0.25)
    values = model._projected_kernel_components(
        kernel,
        left,
        right,
        left_chirality=0,
        right_chirality=0,
        result_chirality=0,
        left_momentum=(3.0, 0.2, -0.4, 2.9),
        right_momentum=(2.0, -0.1, 0.5, -1.9),
        runtime_parameter_values={name: 1.0 for name in kernel.runtime_parameters},
    )
    return tuple(complex(value) for value in values)


def _minimal_scalar_gravity_vertex_model() -> dict[str, object]:
    source_path = SCALAR_GRAVITY_ROOT / "scalar_gravity.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["vertex_rules"] = [
        vertex for vertex in source["vertex_rules"] if vertex["name"] == "V_S0S0Gr"
    ]
    source["lorentz_structures"] = [
        lorentz
        for lorentz in source["lorentz_structures"]
        if lorentz["name"] in {"SSTmpart", "SST"}
    ]
    used_couplings = {
        name
        for row in source["vertex_rules"][0]["couplings"]
        for name in row
        if name is not None
    }
    source["couplings"] = [
        coupling
        for coupling in source["couplings"]
        if coupling["name"] in used_couplings
    ]
    return source


def test_external_spin2_axis_and_dummy_relabeling_preserve_numeric_kernels(
    tmp_path: Path,
) -> None:
    restriction_path = SCALAR_GRAVITY_ROOT / "restrict_default.json"
    source = _minimal_scalar_gravity_vertex_model()
    relabelled = deepcopy(source)
    changed = 0
    renamed_dummy = False
    for lorentz in relabelled["lorentz_structures"]:
        if lorentz["name"] not in {"SSTmpart", "SST"}:
            continue
        original = lorentz["structure"]
        transformed = _rename_lorentz_dummies(_transpose_spin2_axes(original))
        assert transformed != original
        renamed_dummy |= transformed != _transpose_spin2_axes(original)
        lorentz["structure"] = transformed
        changed += 1
    assert changed == 2
    assert renamed_dummy

    source_path = tmp_path / "scalar_gravity-original.json"
    relabelled_path = tmp_path / "scalar_gravity-relabelled.json"
    local_restriction_path = tmp_path / "restrict_default.json"
    source_path.write_text(
        json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    relabelled_path.write_text(
        json.dumps(relabelled, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    local_restriction_path.write_bytes(restriction_path.read_bytes())

    compiled = compile_model_source(
        source_path,
        restriction=str(local_restriction_path.resolve()),
        use_cache=False,
    )
    relabelled_compiled = compile_model_source(
        relabelled_path,
        restriction=str(local_restriction_path.resolve()),
        use_cache=False,
    )
    assert relabelled_compiled.ir.tensor_orderings == compiled.ir.tensor_orderings
    assert relabelled_compiled.ir.current_orderings == compiled.ir.current_orderings

    def kernels_by_orientation(
        model: CompiledModel,
    ) -> dict[tuple[tuple[str, ...], tuple[int, int, int]], CompiledOrientedKernel]:
        return {
            (kernel.particles, kernel.source_particle_legs): kernel
            for kernel in model.ir.oriented_kernels
            if kernel.vertex == "V_S0S0Gr"
        }

    kernels = kernels_by_orientation(compiled)
    relabelled_kernels = kernels_by_orientation(relabelled_compiled)
    assert kernels.keys() == relabelled_kernels.keys()
    assert len(kernels) == 3

    model = CompiledUFOModel(compiled)
    relabelled_model = CompiledUFOModel(relabelled_compiled)
    saw_nonzero = False
    for key, kernel in kernels.items():
        relabelled_kernel = relabelled_kernels[key]
        assert relabelled_kernel.input_ordering_ids == kernel.input_ordering_ids
        assert relabelled_kernel.output_ordering_id == kernel.output_ordering_id
        assert relabelled_kernel.component_expressions == kernel.component_expressions
        expected = _kernel_numeric_values(model, kernel)
        actual = _kernel_numeric_values(relabelled_model, relabelled_kernel)
        assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-14)
        saw_nonzero |= any(abs(value) > 1.0e-14 for value in expected)
    assert saw_nonzero
