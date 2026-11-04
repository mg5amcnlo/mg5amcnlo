# SPDX-License-Identifier: 0BSD
"""Read retained rational colour weights at the active arithmetic precision."""

from collections.abc import Mapping, Sequence
from decimal import Decimal

from pyamplicol.api.errors import ArtifactError


def exact_color_weight(record: Mapping[str, object]) -> tuple[Decimal, Decimal] | None:
    raw = record.get("exact_weight")
    if raw is None:
        return None
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or len(raw) != 4
        or any(not isinstance(value, str) for value in raw)
    ):
        raise ArtifactError("exact colour weight must contain four integer strings")
    try:
        real_numerator, real_denominator, imag_numerator, imag_denominator = (
            int(value) for value in raw
        )
    except ValueError as exc:
        raise ArtifactError("exact colour weight contains an invalid integer") from exc
    if real_denominator <= 0 or imag_denominator <= 0:
        raise ArtifactError("exact colour weight denominators must be positive")
    return (
        Decimal(real_numerator) / Decimal(real_denominator),
        Decimal(imag_numerator) / Decimal(imag_denominator),
    )
