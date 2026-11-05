# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyamplicol import _rusticol
from pyamplicol.generation.recurrence_columnar import RecurrenceColumn
from pyamplicol.generation.recurrence_template_columnar import (
    build_recurrence_template_input_v1,
)
from pyamplicol.models.recurrence_template import (
    EvaluatorBindingV1,
    ParameterTemplateV1,
    RecurrenceTemplateCatalog,
)

_COMPILED_MODEL_DIGEST = "a" * 64
_PREPARED_PACK_DIGEST = "b" * 64
_EXPRESSION_DIGEST = "c" * 64
_CALLABLE_SIGNATURE = "d" * 64
_PREPARED_KERNEL_ID = 7


def _catalog() -> RecurrenceTemplateCatalog:
    parameter = ParameterTemplateV1(
        template_id="parameter:derived",
        name="derived",
        parameter_kind="derived",
        value_type="real",
        mutable=False,
        default_value=None,
        exact_expression_digest=_EXPRESSION_DIGEST,
        dependency_parameter_ids=(),
        prepared_parameter_id=0,
    )
    binding = EvaluatorBindingV1(
        resolver_key="evaluator:model-parameter:derived",
        prepared_kernel_id=_PREPARED_KERNEL_ID,
        contract_kind="model-parameter",
        callable_signature=_CALLABLE_SIGNATURE,
        input_state_template_ids=(),
        output_state_template_id=None,
        input_layout=("model-parameter-input:none",),
        output_layout=("derived",),
        exact_expression_digests=(_EXPRESSION_DIGEST,),
        semantic_template_ids=(parameter.template_id,),
    )
    return RecurrenceTemplateCatalog.create(
        compiled_model_digest=_COMPILED_MODEL_DIGEST,
        prepared_kernel_pack_digest=_PREPARED_PACK_DIGEST,
        parameters=(parameter,),
        evaluator_bindings=(binding,),
    )


def _input_with_stale_digest() -> SimpleNamespace:
    projected = build_recurrence_template_input_v1(_catalog())
    declared_digest = projected.digest
    tables = list(projected.tables)
    header_index = next(
        index for index, table in enumerate(tables) if table.name == "catalog_header"
    )
    header = tables[header_index]
    columns = list(header.columns)
    schema_index = next(
        index for index, column in enumerate(columns) if column.name == "schema_version"
    )
    values = np.array(columns[schema_index].values, copy=True, order="C")
    values[0] += 1
    values.flags.writeable = False
    columns[schema_index] = RecurrenceColumn(
        name=columns[schema_index].name,
        values=values,
    )
    tables[header_index] = replace(header, columns=tuple(columns))
    return SimpleNamespace(
        abi=projected.abi,
        catalog_digest=projected.catalog_digest,
        compiled_model_digest=projected.compiled_model_digest,
        prepared_kernel_pack_digest=projected.prepared_kernel_pack_digest,
        tables=tuple(tables),
        digest=declared_digest,
        canonical_digest=declared_digest,
    )


def test_native_recurrence_template_validation_authenticates_kernel_inventory() -> None:
    projected = build_recurrence_template_input_v1(_catalog())

    result = _rusticol._validate_recurrence_template_input_v1(
        projected,
        [_PREPARED_KERNEL_ID],
    )

    assert result["validation_status"] == "validated"
    assert result["template_input_sha256"] == projected.digest
    assert result["prepared_kernel_inventory_verified"] is True
    assert result["prepared_kernel_inventory_count"] == 1
    assert result["counts"]["prepared_kernels"] == 1
    assert result["counts"]["referenced_prepared_kernels"] == 1


def test_native_recurrence_template_validation_rejects_stale_digest() -> None:
    with pytest.raises(ValueError, match="template input digest mismatch"):
        _rusticol._validate_recurrence_template_input_v1(
            _input_with_stale_digest(),
            [_PREPARED_KERNEL_ID],
        )


def test_native_recurrence_template_validation_rejects_missing_kernel() -> None:
    projected = build_recurrence_template_input_v1(_catalog())

    with pytest.raises(
        ValueError,
        match=(
            r"prepared-kernel IDs absent from the authenticated pack inventory: \[7\]"
        ),
    ):
        _rusticol._validate_recurrence_template_input_v1(projected, [0])
