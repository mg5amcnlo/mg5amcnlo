# SPDX-License-Identifier: 0BSD
"""Evaluator runtime-capability metadata shared by artifact writers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .._internal.physics.types import NativeEvaluationError
from .._internal.versions import EVALUATOR_RUNTIME_CAPABILITIES


def evaluator_runtime_capabilities(
    manifest: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the primary f64 capabilities required by one evaluator tree."""

    kind = str(manifest.get("kind", ""))
    if kind == "chunked-symbolica-evaluator":
        chunks = manifest.get("chunks")
        if isinstance(chunks, str | bytes) or not isinstance(chunks, Sequence):
            raise NativeEvaluationError(
                "chunked evaluator capability metadata has no chunk list"
            )
        return aggregate_runtime_capabilities(
            _mapping(chunk, "chunked evaluator entry") for chunk in chunks
        )

    capability = manifest.get("runtime_capability")
    if (
        not isinstance(capability, str)
        or capability not in EVALUATOR_RUNTIME_CAPABILITIES
    ):
        raise NativeEvaluationError(
            f"evaluator {kind!r} has no supported runtime capability"
        )
    return (capability,)


def aggregate_runtime_capabilities(
    manifests: Iterable[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return a canonical union of the primary f64 evaluator capabilities."""

    capabilities: set[str] = set()
    for manifest in manifests:
        capabilities.update(evaluator_runtime_capabilities(manifest))
    return tuple(sorted(capabilities))


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NativeEvaluationError(f"{context} is not an object")
    return {str(key): item for key, item in value.items()}


__all__ = ["aggregate_runtime_capabilities", "evaluator_runtime_capabilities"]
