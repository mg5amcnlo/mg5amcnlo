# SPDX-License-Identifier: 0BSD
"""Fast public-facade and native-adapter contracts for OTF cache persistence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pyamplicol import Runtime
from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.runtime.backend import RusticolRuntimeBackend


class _Backend:
    """The minimum runtime protocol, deliberately without optional cache APIs."""

    physics: Any = None

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cache operations must not evaluate")

    def evaluate_resolved(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("cache operations must not evaluate")

    def set_model_parameters(self, mapping: Any) -> None:
        raise AssertionError("cache operations must not change parameters")

    def clear(self) -> None:
        raise AssertionError("cache operations must not clear before validation")

    def mute_warnings(self) -> None:
        pass

    def unmute_warnings(self) -> None:
        pass


class _CacheBackend(_Backend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def save(self, path: Path) -> None:
        self.calls.append(("save", path))

    def load_cache(self, path: Path) -> None:
        self.calls.append(("load_cache", path))


@pytest.mark.parametrize("name", ("save", "load_cache"))
@pytest.mark.parametrize("as_string", (False, True))
def test_facade_normalizes_cache_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, as_string: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = _CacheBackend()
    runtime = Runtime(backend)
    relative = Path("cache with spaces") / ".." / "recursion.cache"
    assert getattr(runtime, name)(str(relative) if as_string else relative) is None
    assert backend.calls == [(name, tmp_path / "recursion.cache")]


@pytest.mark.parametrize("name", ("save", "load_cache"))
def test_facade_cache_support_does_not_expand_minimum_backend_protocol(
    name: str,
) -> None:
    runtime = Runtime(_Backend())
    with pytest.raises(CompatibilityError, match="does not support"):
        getattr(runtime, name)("recursion.cache")


@pytest.mark.parametrize("name", ("save", "load_cache"))
def test_facade_rejects_non_paths_before_calling_backend(name: str) -> None:
    backend = _CacheBackend()
    with pytest.raises(TypeError):
        getattr(Runtime(backend), name)(object())
    assert backend.calls == []


class _NativeArtifactError(Exception):
    pass


def _adapter(native: Any, *, mode: str = "on-the-fly") -> RusticolRuntimeBackend:
    # Constructor metadata validation is covered by test_runtime_adapter.py;
    # this seam isolates cache forwarding and native exception translation.
    backend = object.__new__(RusticolRuntimeBackend)
    backend._runtime = native
    backend._native_module = SimpleNamespace(ArtifactError=_NativeArtifactError)
    backend._execution_mode = mode  # type: ignore[assignment]
    return backend


@pytest.mark.parametrize("name", ("save", "load_cache"))
def test_native_adapter_forwards_path_without_opening_physics(
    tmp_path: Path, name: str
) -> None:
    paths: list[Path] = []
    backend = _adapter(SimpleNamespace(**{name: paths.append}))
    path = tmp_path / "recursion.cache"
    assert getattr(backend, name)(path) is None
    assert paths == [path]


@pytest.mark.parametrize("name", ("save", "load_cache"))
def test_native_adapter_translates_restore_and_save_errors(name: str) -> None:
    def fail(path: Path) -> None:
        raise _NativeArtifactError("cache does not match this process")

    backend = _adapter(SimpleNamespace(**{name: fail}))
    with pytest.raises(ArtifactError, match="does not match") as error:
        getattr(backend, name)(Path("cache"))
    assert isinstance(error.value.__cause__, _NativeArtifactError)


@pytest.mark.parametrize("name", ("save", "load_cache"))
@pytest.mark.parametrize("mode", ("recurrence", "compiled", "eager"))
def test_cache_operations_reject_non_otf_modes(name: str, mode: str) -> None:
    backend = _adapter(SimpleNamespace(), mode=mode)
    with pytest.raises(CompatibilityError, match="only for on-the-fly"):
        getattr(backend, name)(Path("cache"))


@pytest.mark.parametrize("name", ("save", "load_cache"))
def test_native_adapter_reports_missing_binding(name: str) -> None:
    with pytest.raises(CompatibilityError, match=f"no on-the-fly {name} binding"):
        getattr(_adapter(SimpleNamespace()), name)(Path("cache"))
