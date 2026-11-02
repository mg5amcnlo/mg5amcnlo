# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

import pytest

import pyamplicol._internal.physics.symbols as symbols_module
from pyamplicol.models.builtin.symbols import BuiltinSymbolRegistry


def test_builtin_lowering_vocabulary_is_not_on_shared_registry() -> None:
    shared = symbols_module.SymbolRegistry()
    builtin = BuiltinSymbolRegistry()

    for name in (
        "two_gluon_to_tensor_name",
        "tensor_gluon_to_gluon_name",
        "gluon_tensor_to_gluon_name",
        "quark_vector_weyl_plus_name",
        "quark_vector_weyl_minus_name",
        "matrix_element_plan",
    ):
        assert not hasattr(shared, name)
    assert builtin.two_gluon_to_tensor_name == "pyamplicol::two_gluon_to_tensor"
    assert builtin.quark_vector_weyl_plus_name == (
        "pyamplicol::quark_vector_weyl_plus"
    )


def test_tensor_canonization_indices_use_the_shared_namespace() -> None:
    index = symbols_module.SymbolRegistry().canonical_tensor_index(
        "c",
        "adjoint",
        3,
    )

    assert index.to_canonical_string() == (
        "pyamplicol::canonical_tensor_index::c::adjoint::{}::i3"
    )


def test_symbol_construction_is_serialized_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participants = 24
    start = Barrier(participants)
    observations_lock = Lock()
    calls: list[tuple[str, bool]] = []
    active = 0
    max_active = 0

    def fake_symbol(name: str, *, is_real: bool) -> object:
        nonlocal active, max_active
        with observations_lock:
            calls.append((name, is_real))
            active += 1
            max_active = max(max_active, active)
        sleep(0.01)
        value = object()
        with observations_lock:
            active -= 1
        return value

    monkeypatch.setattr(symbols_module, "_construct_symbol", fake_symbol)
    registry = symbols_module.SymbolRegistry(namespace="thread_safety_test")
    model_registry = registry.model("thread-safety-test")
    cache_keys = (
        (registry.qualified_name("shared_complex"), False),
        (registry.qualified_name("shared_real"), True),
        (model_registry.qualified_name("shared"), False),
    )
    for key in cache_keys:
        symbols_module._SYMBOL_CACHE.pop(key, None)

    def construct(index: int) -> object:
        start.wait(timeout=5)
        operation = index % 3
        if operation == 0:
            return registry.symbol("shared_complex")
        if operation == 1:
            return registry.real_symbol("shared_real")
        return model_registry.symbol("shared")

    with ThreadPoolExecutor(max_workers=participants) as executor:
        results = list(executor.map(construct, range(participants)))

    assert max_active == 1
    assert Counter(calls) == Counter(
        {
            (registry.qualified_name("shared_complex"), False): 1,
            (registry.qualified_name("shared_real"), True): 1,
            (model_registry.qualified_name("shared"), False): 1,
        }
    )
    assert len({id(result) for result in results}) == 3
    for key in cache_keys:
        symbols_module._SYMBOL_CACHE.pop(key, None)
