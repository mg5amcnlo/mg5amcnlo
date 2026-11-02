# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.api import ModelSource
from pyamplicol.api.errors import GenerationError
from pyamplicol.config import ColorConfig, EvaluatorConfig, RunConfig
from pyamplicol.generation.service import GenerationBackend
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _backend(*, accuracy: str = "lc") -> GenerationBackend:
    return GenerationBackend(
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy=accuracy),
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        ),
        None,
    )


@pytest.mark.parametrize("accuracy", ("nlc", "full"))
def test_recurrence_accepts_contracted_color_before_process_expansion(
    accuracy: str,
) -> None:
    _backend(accuracy=accuracy)._validate_recurrence_request()


def test_recurrence_lc_is_fail_closed_before_generic_dag_construction() -> None:
    backend = _backend()
    process = build_process_ir("d d~ > z g")

    with pytest.raises(GenerationError, match="recurrence-template-v1"):
        backend._compile_concrete_process(process, BuiltinSMModel())


def test_recurrence_requires_prepared_template_with_exact_rebuild_command() -> None:
    backend = _backend()
    source = ModelSource(kind="prepared", path=Path("model.pyamplicol-model"))
    resolved = SimpleNamespace(
        source=source,
        compiled=SimpleNamespace(
            prepared_bundle=SimpleNamespace(
                kernel_pack=SimpleNamespace(recurrence_template=None)
            )
        ),
    )

    with pytest.raises(
        GenerationError,
        match=(
            r"pyamplicol-recurrence-template-v1.*pyamplicol model compile "
            r".*model\.pyamplicol-model MODEL\.pyamplicol-model --backend jit"
        ),
    ):
        backend._require_eager_kernel_pack(resolved)  # type: ignore[arg-type]
