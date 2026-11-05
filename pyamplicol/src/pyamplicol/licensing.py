# SPDX-License-Identifier: 0BSD
"""Symbolica licensing and generation-resource policy."""

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TextIO

from pyamplicol.config import (
    Action,
    ClampRequest,
    ConfigResolution,
    GenerationConfig,
    RunConfig,
    config_to_dict,
    resolve_config,
)

_SUGGESTION_LOCK = threading.Lock()
_SUGGESTION_EMITTED = False
_RESOURCE_PATHS = frozenset(
    {
        "generation.workers",
        "evaluator.optimization.cores",
    }
)


@dataclass(frozen=True, slots=True)
class SymbolicaLicenseState:
    licensed: bool
    restricted: bool


def _load_symbolica() -> ModuleType:
    return importlib.import_module("symbolica")


def prepare_symbolica_environment(*, suppress_banner: bool) -> None:
    """Set banner policy before Symbolica's first import."""

    if suppress_banner:
        os.environ["SYMBOLICA_HIDE_BANNER"] = "1"


def detect_symbolica_license(
    *,
    suggest: bool = True,
    json_mode: bool = False,
    stream: TextIO | None = None,
    loader: Callable[[], ModuleType] = _load_symbolica,
) -> SymbolicaLicenseState:
    """Query Symbolica's actual package-or-personal license state.

    This call must stay within pyamplicol: OEM licenses are package-scoped.
    """

    prepare_symbolica_environment(suppress_banner=not suggest or json_mode)
    module = loader()
    checker = getattr(module, "is_licensed", None)
    if not callable(checker):
        raise RuntimeError("installed Symbolica does not provide is_licensed()")
    licensed = bool(checker())
    if not licensed and suggest and not json_mode:
        _suggest_license(sys.stderr if stream is None else stream)
    return SymbolicaLicenseState(licensed=licensed, restricted=not licensed)


def _suggest_license(stream: TextIO) -> None:
    global _SUGGESTION_EMITTED
    with _SUGGESTION_LOCK:
        if _SUGGESTION_EMITTED:
            return
        stream.write(
            "pyAmpliCol is using Symbolica restricted mode (one generation "
            "worker and one Symbolica core). Request a free license with "
            "'pyamplicol request-symbolica-trial-license' or "
            "'pyamplicol request-symbolica-hobbyist-license'.\n"
        )
        stream.flush()
        _SUGGESTION_EMITTED = True


def reset_suggestion_state_for_tests() -> None:
    global _SUGGESTION_EMITTED
    with _SUGGESTION_LOCK:
        _SUGGESTION_EMITTED = False


def _cpu_budget() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def symbolica_resource_clamps(
    config: RunConfig,
    state: SymbolicaLicenseState,
    *,
    cpu_budget: int | None = None,
    process_count: int | None = None,
) -> tuple[ClampRequest, ...]:
    """Share one CPU budget across concrete-process workers and evaluator cores.

    Licensed partitioning is deferred when the concrete process count is not yet
    known. Restricted mode does not need that count and is always clamped eagerly.
    """

    if cpu_budget is not None and (
        isinstance(cpu_budget, bool)
        or not isinstance(cpu_budget, int)
        or cpu_budget < 1
    ):
        raise ValueError("cpu_budget must be a positive integer")
    if process_count is not None and (
        isinstance(process_count, bool)
        or not isinstance(process_count, int)
        or process_count < 1
    ):
        raise ValueError("process_count must be a positive integer")
    requested_workers = config.generation.workers
    requested_cores = config.evaluator.optimization.cores
    clamps: list[ClampRequest] = []

    if state.restricted:
        if requested_workers != 1:
            clamps.append(
                ClampRequest(
                    "generation.workers",
                    1,
                    "Symbolica restricted mode permits one generation instance",
                )
            )
        if requested_cores != 1:
            clamps.append(
                ClampRequest(
                    "evaluator.optimization.cores",
                    1,
                    "Symbolica restricted mode permits one Symbolica core",
                )
            )
        return tuple(clamps)

    if process_count is None:
        return ()

    budget = _cpu_budget() if cpu_budget is None else cpu_budget
    assert budget is not None
    workers = (
        min(process_count, budget)
        if requested_workers == "auto"
        else min(requested_workers, process_count, budget)
    )
    if requested_workers != workers:
        clamps.append(
            ClampRequest(
                "generation.workers",
                workers,
                "shared affinity-aware CPU budget for concurrent process generation",
            )
        )
    cores = (
        max(1, budget // workers)
        if requested_cores == "auto"
        else min(requested_cores, max(1, budget // workers))
    )
    if requested_cores != cores:
        clamps.append(
            ClampRequest(
                "evaluator.optimization.cores",
                cores,
                "shared affinity-aware CPU budget for Symbolica evaluator work",
            )
        )
    return tuple(clamps)


def resolve_symbolica_resource_config(
    config: GenerationConfig | RunConfig | ConfigResolution | None,
    state: SymbolicaLicenseState,
    *,
    cpu_budget: int | None = None,
    process_count: int | None = None,
) -> ConfigResolution:
    """Apply current resource policy without losing original provenance.

    Resource clamps are recalculated from the original requested values. This
    removes provisional or stale process-count clamps while preserving unrelated
    adjustments and their ordering.
    """

    if isinstance(config, ConfigResolution):
        requested = config.requested
        inherited = config.clamps
    else:
        requested = (
            RunConfig(action=Action.GENERATE)
            if config is None
            else RunConfig(action=Action.GENERATE, generation=config)
            if isinstance(config, GenerationConfig)
            else config
        )
        inherited = ()

    preserved = tuple(
        ClampRequest(item.path, item.effective, item.reason)
        for item in inherited
        if item.path not in _RESOURCE_PATHS
    )
    base = resolve_config(config_to_dict(requested), clamps=preserved)
    policy = symbolica_resource_clamps(
        base.effective,
        state,
        cpu_budget=cpu_budget,
        process_count=process_count,
    )
    replacements = {item.path: item for item in policy}
    merged: list[ClampRequest] = []
    emitted: set[str] = set()
    for inherited_clamp in inherited:
        if inherited_clamp.path in _RESOURCE_PATHS:
            replacement = replacements.get(inherited_clamp.path)
            if replacement is not None and inherited_clamp.path not in emitted:
                merged.append(replacement)
                emitted.add(inherited_clamp.path)
            continue
        merged.append(
            ClampRequest(
                inherited_clamp.path,
                inherited_clamp.effective,
                inherited_clamp.reason,
            )
        )
    for policy_clamp in policy:
        if policy_clamp.path not in emitted:
            merged.append(policy_clamp)
            emitted.add(policy_clamp.path)

    return resolve_config(config_to_dict(requested), clamps=merged)


def request_trial_license(
    name: str,
    email: str,
    organization: str,
    *,
    loader: Callable[[], ModuleType] = _load_symbolica,
) -> None:
    _validate_identity((name, email, organization), requires_organization=True)
    function = getattr(loader(), "request_trial_license", None)
    if not callable(function):
        raise RuntimeError(
            "installed Symbolica does not provide request_trial_license()"
        )
    function(name, email, organization)


def request_hobbyist_license(
    name: str,
    email: str,
    *,
    loader: Callable[[], ModuleType] = _load_symbolica,
) -> None:
    _validate_identity((name, email), requires_organization=False)
    function = getattr(loader(), "request_hobbyist_license", None)
    if not callable(function):
        raise RuntimeError(
            "installed Symbolica does not provide request_hobbyist_license()"
        )
    function(name, email)


def _validate_identity(
    values: Sequence[str],
    *,
    requires_organization: bool,
) -> None:
    expected = 3 if requires_organization else 2
    if len(values) != expected or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        fields = (
            "name, email, and organization"
            if requires_organization
            else "name and email"
        )
        raise ValueError(f"Symbolica license requests require non-empty {fields}")
    email = values[1]
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError("Symbolica license requests require a valid email address")


__all__ = [
    "SymbolicaLicenseState",
    "detect_symbolica_license",
    "prepare_symbolica_environment",
    "request_hobbyist_license",
    "request_trial_license",
    "resolve_symbolica_resource_config",
    "symbolica_resource_clamps",
]
