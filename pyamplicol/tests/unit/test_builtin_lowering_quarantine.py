# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _assert_isolated_import_does_not_load_builtin(statement: str) -> None:
    script = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(SOURCE_ROOT)!r})",
            statement,
            "loaded = sorted(name for name in sys.modules ",
            "    if name == 'pyamplicol.models.builtin' ",
            "    or name.startswith('pyamplicol.models.builtin.'))",
            "assert loaded == [], loaded",
        )
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "statement",
    (
        "import pyamplicol.generation",
        "import pyamplicol.models.base",
        "import pyamplicol.models.expressions",
    ),
)
def test_generic_lowering_imports_do_not_load_builtin_physics(statement: str) -> None:
    _assert_isolated_import_does_not_load_builtin(statement)


def test_builtin_lowering_diagnostics_are_not_generic_model_contracts() -> None:
    from pyamplicol.models import base, expressions
    from pyamplicol.models.builtin import lowering_types

    assert hasattr(lowering_types, "SymbolicLoweringReport")
    assert not hasattr(base.Model, "build_tensor_library")
    assert not hasattr(base.Model, "vertex_lowering_coverage")
    assert not hasattr(base, "VertexLoweringCoverageEntry")
    assert not hasattr(base, "VertexLoweringCoverageReport")
    assert not hasattr(expressions, "_flat_index")
    assert not hasattr(expressions, "_index_chirality")
    assert not hasattr(expressions, "_expr_vector_slash_terms")


def test_builtin_auxiliary_tensor_probe_executes() -> None:
    from pyamplicol.models import BuiltinSMModel
    from pyamplicol.models.builtin.lowering_tensor import (
        _build_auxiliary_tensor_probe,
    )

    probe = _build_auxiliary_tensor_probe(BuiltinSMModel())

    assert probe.engine == "spenso"
    assert probe.output_rank == 2
    assert probe.output_size == 16
    assert probe.nonzero_entries == 4
    assert probe.max_abs_entry == pytest.approx(1.5)
    assert probe.weighted_checksum == pytest.approx((0.0, 48.0))


@pytest.mark.parametrize(
    ("vertex", "left_kind", "right_kind", "output_kind", "chirality"),
    (
        ("two_gluon_to_tensor", "vector", "vector", "tensor", 0),
        ("tensor_gluon_to_gluon", "tensor", "vector", "vector", 0),
        ("gluon_tensor_to_gluon", "vector", "tensor", "vector", 0),
        ("quark_vector_weyl_plus", "weyl", "vector", "weyl", 1),
        ("quark_vector_weyl_minus", "weyl", "vector", "weyl", -1),
    ),
)
def test_builtin_tensor_interface_order_matches_numeric_vertices(
    vertex: str,
    left_kind: str,
    right_kind: str,
    output_kind: str,
    chirality: int,
) -> None:
    from symbolica import Expression
    from symbolica.community.spenso import (
        Representation,
        Tensor,
        TensorName,
        TensorNetwork,
        as_tensor,
    )

    from pyamplicol.models import BuiltinSMModel
    from pyamplicol.models.builtin import expressions
    from pyamplicol.models.builtin.symbols import symbols

    representations = {
        "vector": Representation.mink(4),
        "tensor": Representation(symbols.antisymmetric_lorentz_pair_name, 6),
        "weyl": Representation(symbols.weyl_spinor_name, 2),
    }
    dimensions = {"vector": 4, "tensor": 6, "weyl": 2}
    left = tuple(complex(index + 1, 0.25) for index in range(dimensions[left_kind]))
    right = tuple(
        complex(2 * index - 1, -0.5) for index in range(dimensions[right_kind])
    )
    library = BuiltinSMModel().build_tensor_library()
    left_name = TensorName("pyamplicol::tensor_layout_probe_left")
    right_name = TensorName("pyamplicol::tensor_layout_probe_right")
    left_rep, right_rep = representations[left_kind], representations[right_kind]
    output_rep = representations[output_kind]
    library.register(Tensor.dense(left_name(left_rep), left))
    library.register(Tensor.dense(right_name(right_rep), right))
    expression = (
        TensorName(symbols.qualified_name(vertex))(
            left_rep("left"), right_rep("right"), output_rep("output")
        ).to_expression()
        * left_name(left_rep("left")).to_expression()
        * right_name(right_rep("right")).to_expression()
    )
    network = TensorNetwork(as_tensor(expression), library)
    network.execute(library=library)
    result = network.result_tensor(library)
    actual = tuple(
        complex(value.evaluate({}) if isinstance(value, Expression) else value)
        for value in (result[index] for index in range(len(result)))
    )
    if chirality:
        expected = expressions._expr_fermion_vector_weyl(
            left, right, chirality, antifermion=False, coupling=None
        )
    else:
        numeric_vertex = {
            "two_gluon_to_tensor": expressions._expr_two_vector_to_tensor,
            "tensor_gluon_to_gluon": expressions._expr_tensor_vector_to_vector,
            "gluon_tensor_to_gluon": expressions._expr_vector_tensor_to_vector,
        }[vertex]
        expected = numeric_vertex(left, right)
    assert actual == pytest.approx(expected)
