# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from functools import cached_property
from typing import Any

from ..._internal.physics.symbols import SymbolRegistry


class BuiltinSymbolRegistry(SymbolRegistry):
    """Symbolica heads owned exclusively by built-in-SM lowering."""

    @cached_property
    def antisymmetric_lorentz_pair_name(self) -> str:
        return self.qualified_name("antisymmetric_lorentz_pair")

    @cached_property
    def weyl_spinor_name(self) -> str:
        return self.qualified_name("weyl_spinor")

    def vertex_kind_tensor_name(self, kind: int) -> str:
        return self.qualified_name(f"vertex_kind_{int(kind)}")

    def label(self, labels: tuple[int, ...]) -> Any:
        return self.symbol("label::" + "_".join(str(label) for label in labels))

    @cached_property
    def two_gluon_to_tensor_name(self) -> str:
        return self.qualified_name("two_gluon_to_tensor")

    @cached_property
    def tensor_gluon_to_gluon_name(self) -> str:
        return self.qualified_name("tensor_gluon_to_gluon")

    @cached_property
    def gluon_tensor_to_gluon_name(self) -> str:
        return self.qualified_name("gluon_tensor_to_gluon")

    @cached_property
    def quark_vector_weyl_plus_name(self) -> str:
        return self.qualified_name("quark_vector_weyl_plus")

    @cached_property
    def quark_vector_weyl_minus_name(self) -> str:
        return self.qualified_name("quark_vector_weyl_minus")

    @cached_property
    def current(self) -> Any:
        return self.symbol("current")

    @cached_property
    def vertex(self) -> Any:
        return self.symbol("vertex")

    @cached_property
    def assignment(self) -> Any:
        return self.symbol("assign")

    @cached_property
    def amplitude(self) -> Any:
        return self.symbol("amplitude")

    @cached_property
    def matrix_element_plan(self) -> Any:
        return self.symbol("matrix_element_plan")

    @cached_property
    def momentum(self) -> Any:
        return self.symbol("momentum")

    @cached_property
    def current_momentum(self) -> Any:
        return self.symbol("current_momentum")

    @cached_property
    def polarization(self) -> Any:
        return self.symbol("polarization")


symbols = BuiltinSymbolRegistry()


__all__ = ["BuiltinSymbolRegistry", "symbols"]
