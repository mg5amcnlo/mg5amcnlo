# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from threading import Lock
from typing import Any

_UFO_CANONICAL_NAMESPACE = "UFO::{}::"
# Symbolica's process-wide symbol table cannot handle concurrent S(...) calls.
_SYMBOL_CONSTRUCTION_LOCK = Lock()
_SYMBOL_CACHE: dict[tuple[str, bool], Any] = {}


def _construct_symbol(qualified_name: str, *, is_real: bool) -> Any:
    from symbolica import S

    return S(qualified_name, is_real=True) if is_real else S(qualified_name)


def _symbol(qualified_name: str, *, is_real: bool = False) -> Any:
    with _SYMBOL_CONSTRUCTION_LOCK:
        key = (qualified_name, is_real)
        cached = _SYMBOL_CACHE.get(key)
        if cached is not None:
            return cached
        value = _construct_symbol(qualified_name, is_real=is_real)
        _SYMBOL_CACHE[key] = value
        return value


def _safe_namespace_part(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    readable = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_") or "unnamed"
    if readable[0].isdigit():
        readable = f"n_{readable}"
    digest = sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{readable}_{digest}"


def _safe_symbol_path(value: object) -> str:
    return "::".join(
        _safe_namespace_part(part) for part in str(value).split("::") if part
    )


@dataclass(frozen=True)
class ModelSymbolRegistry:
    """Symbolica names owned by one externally supplied model."""

    model_name: str

    @cached_property
    def namespace(self) -> str:
        return f"model_{_safe_namespace_part(self.model_name)}"

    def qualified_name(self, name: str) -> str:
        return f"{self.namespace}::{_safe_symbol_path(name)}"

    def symbol(self, name: str) -> Any:
        return _symbol(self.qualified_name(name))

    def expression(self, source: str) -> Any:
        """Parse a UFO expression after moving every UFO head into this model."""

        from symbolica import E

        namespaced = str(source).replace(
            _UFO_CANONICAL_NAMESPACE,
            f"{self.namespace}::",
        )
        namespaced = namespaced.replace("UFO::", f"{self.namespace}::")
        return E(namespaced)

    def expression_string(self, source: str) -> str:
        return self.expression(source).to_canonical_string()

    def kernel_tensor_name(self, kind: int, side: str) -> str:
        return self.qualified_name(f"compiler::kernel_{int(kind)}_{side}")

    def kernel_component(self, kind: int, side: str, index: int) -> Any:
        return self.symbol(f"compiler::kernel_{int(kind)}_{side}_{int(index)}")

    def kernel_momentum(self, kind: int, side: str, index: int) -> Any:
        return self.symbol(
            f"compiler::kernel_{int(kind)}_{side}_momentum_{int(index)}"
        )

    def contact_leg_tensor_name(self, kind: int, leg: int) -> str:
        return self.qualified_name(
            f"compiler::contact_{int(kind)}_leg_{int(leg)}"
        )

    def ufo_momentum_tensor_name(self, leg: int) -> str:
        return self.qualified_name(f"compiler::ufo_momentum_{int(leg)}")


@dataclass(frozen=True)
class SymbolRegistry:
    """Central registry for reusable Symbolica heads used by pyamplicol."""

    namespace: str = "pyamplicol"

    def qualified_name(self, name: str) -> str:
        return f"{self.namespace}::{name}"

    def display_name(self, name: str) -> str:
        return self.qualified_name(name)

    def symbol(self, name: str) -> Any:
        return _symbol(self.qualified_name(name))

    def real_symbol(self, name: str) -> Any:
        return _symbol(self.qualified_name(name), is_real=True)

    def parameter(self, name: str) -> Any:
        return self.symbol(f"param::{name}")

    def parameter_component(
        self,
        head: tuple[str, ...],
        index: int,
    ) -> Any:
        safe_head = "::".join(_safe_symbol_path(part) for part in head)
        return self.parameter(f"{safe_head}::c{int(index)}")

    def model(self, model_name: str) -> ModelSymbolRegistry:
        return ModelSymbolRegistry(str(model_name))

    def model_owned_name(self, model_name: str, name: str) -> str:
        model_tag = _safe_namespace_part(model_name)
        return self.qualified_name(f"model::{model_tag}::{name}")

    def model_owned(self, model_name: str, name: str) -> Any:
        return self.symbol(f"model::{_safe_namespace_part(model_name)}::{name}")

    def derived_coupling(self, model_name: str, term_id: int) -> Any:
        return self.model_owned(model_name, f"derived_coupling_{int(term_id)}")

    def runtime_model_parameter(self, model_name: str, name: str) -> Any:
        """Return the canonical head for one external-model runtime input."""

        match = re.fullmatch(r"derived_coupling_([0-9]+)", str(name))
        if match is not None:
            return self.derived_coupling(model_name, int(match.group(1)))
        return self.model(model_name).symbol(str(name))

    def kernel_component(self, kind: int, side: str, index: int) -> Any:
        return self.symbol(f"kernel_{int(kind)}_{side}_{int(index)}")

    def kernel_momentum(self, kind: int, side: str, index: int) -> Any:
        return self.symbol(f"kernel_{int(kind)}_{side}_momentum_{int(index)}")

    def kernel_tensor_name(self, kind: int, side: str) -> str:
        return self.qualified_name(f"kernel_{int(kind)}_{side}")

    def kernel_function_argument(
        self,
        model_name: str,
        key: str,
        role: str,
        index: int,
    ) -> Any:
        return self.model_owned(
            model_name,
            f"kernel_function::{key}::{role}::c{int(index)}",
        )

    def kernel_function_component(
        self,
        model_name: str,
        key: str,
        index: int,
    ) -> Any:
        return self.model_owned(
            model_name,
            f"kernel_function::{key}::component::c{int(index)}",
        )

    def weyl_probe(
        self,
        model_name: str,
        kind: int,
        side: str,
        role: str,
        index: int,
    ) -> Any:
        return self.model_owned(
            model_name,
            f"weyl_probe::kind_{int(kind)}::{side}::{role}::c{int(index)}",
        )

    def runtime_domain(
        self,
        model_name: str,
        parameter_name: str,
        domain: str,
    ) -> Any:
        name = f"runtime_domain::{domain}::{_safe_symbol_path(parameter_name)}"
        if domain == "complex":
            return self.model_owned(model_name, name)
        real_symbol = self.real_symbol(
            f"model::{_safe_namespace_part(model_name)}::{name}"
        )
        return 1j * real_symbol if domain == "imaginary" else real_symbol

    def custom_propagator_component(
        self,
        model_name: str,
        role: str,
        index: int,
    ) -> Any:
        return self.model_owned(
            model_name,
            f"custom_propagator::{role}::c{int(index)}",
        )

    def custom_propagator_tensor_name(self, model_name: str) -> str:
        return self.model_owned_name(model_name, "custom_propagator::input")

    def ufo_momentum_tensor_name(self, leg: int) -> str:
        return self.qualified_name(f"ufo_momentum_{int(leg)}")

    def contact_leg_tensor_name(self, kind: int, leg: int) -> str:
        return self.qualified_name(f"contact_{int(kind)}_leg_{int(leg)}")

    def canonical_tensor_index(
        self,
        family: str,
        representation: str,
        index: int,
    ) -> Any:
        """Return a reserved dummy-index candidate for tensor canonization."""

        return self.symbol(
            "canonical_tensor_index::"
            f"{_safe_symbol_path(family)}::"
            f"{_safe_symbol_path(representation)}::i{int(index)}"
        )

    @cached_property
    def color_projection_probe_name(self) -> str:
        return self.qualified_name("color_projection_probe")

    def inline_function_wildcard(
        self,
        definition_index: int,
        argument_index: int,
    ) -> Any:
        return self.symbol(
            f"inline_function_{int(definition_index)}_argument_{int(argument_index)}_"
        )

symbols = SymbolRegistry()


__all__ = ["ModelSymbolRegistry", "SymbolRegistry", "symbols"]
