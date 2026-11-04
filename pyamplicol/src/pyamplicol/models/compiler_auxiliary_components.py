# SPDX-License-Identifier: 0BSD
"""Remove unobservable components from compiler-owned contact-tree currents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .._internal.physics.symbols import ModelSymbolRegistry
from . import compiler_symbolica as _sym
from .compiler_kernels import _canonicalize_oriented_kernel_component
from .contracts import CompiledOrientedKernel, CompiledParticleRecord

# A reduced component is a signed sum of original producer components.  Only
# identical/opposite consumer columns are combined; no numerical rank decision
# or parameter-dependent change of basis is involved.
_ComponentGroup = tuple[tuple[int, int], ...]
_ComponentBasis = tuple[_ComponentGroup, ...]


def reduce_contact_auxiliary_components(
    particles: Sequence[CompiledParticleRecord],
    kernels: Sequence[CompiledOrientedKernel],
    *,
    auxiliary_particles: Sequence[CompiledParticleRecord],
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Reduce opaque auxiliaries before ordering/kernel preparation.

    The caller supplies only its newly lowered contact-tree auxiliaries.  Their
    sole consumers are oriented kernels: unlike physical particles and other
    auxiliary representations, they have no wavefunctions, propagators, direct
    contractions, or externally prescribed tensor basis to preserve.

    For every input orientation, certify ``F(v) = sum_i C_i v_i`` exactly.  A
    zero column is unused; columns ``C_i = +/- C_j`` need only the corresponding
    signed sum of producer components.  All kernels and both input sides are
    considered before changing any representation.  Repeat after a reduction
    so that discarded downstream components can expose upstream reductions.
    """
    candidates = {particle.name for particle in auxiliary_particles}
    particles, kernels = tuple(particles), tuple(kernels)
    if not candidates:
        return particles, kernels
    _sym._ensure_symbolica()
    while True:
        bases: dict[str, _ComponentBasis] = {}
        dimensions: dict[str, int] = {}
        exact_kernels: dict[int, tuple[_sym.Expression, ...] | None] = {}
        for particle in particles:
            if (
                particle.name not in candidates
                or particle.spin != -1
                or particle.propagating
                or particle.propagator is not None
                or particle.antiname != particle.name
                or particle.component_dimension is None
                or particle.component_dimension <= 1
            ):
                continue
            involved = tuple(
                kernel for kernel in kernels if particle.name in kernel.particles
            )
            producers = tuple(
                kernel for kernel in involved if kernel.particles[2] == particle.name
            )
            # Basis-sensitive metadata is derived later, not rewritten here.
            if not producers or any(
                kernel.input_ordering_ids
                or kernel.output_ordering_id
                or kernel.evaluation_equivalence_verified
                or any(
                    name == particle.name and leg != -1
                    for name, leg in zip(
                        kernel.particles, kernel.source_particle_legs, strict=True
                    )
                )
                for kernel in involved
            ):
                continue
            dimension = particle.component_dimension
            if any(
                len(kernel.component_expressions) != dimension for kernel in producers
            ):
                continue
            for kernel in involved:
                if kernel.kind not in exact_kernels:
                    exact_kernels[kernel.kind] = _exact_kernel_components(kernel)
            # Algebra on an unsupported floating coefficient could round away
            # a small difference. Leave that whole current basis untouched.
            if any(exact_kernels[kernel.kind] is None for kernel in involved):
                continue
            basis = _consumer_basis(
                particle.name,
                dimension,
                involved,
                exact_kernels=exact_kernels,
                model_symbols=model_symbols,
            )
            # Keep an entirely unobserved family unchanged: removing its last
            # component would require eliminating the family, not a basis change.
            if basis and len(basis) < dimension:
                bases[particle.name] = basis
                dimensions[particle.name] = dimension
        if not bases:
            return particles, kernels
        particles = tuple(
            replace(particle, component_dimension=len(bases[particle.name]))
            if particle.name in bases
            else particle
            for particle in particles
        )
        kernels = tuple(
            _reduce_kernel(
                kernel,
                bases,
                dimensions,
                exact_kernels=exact_kernels,
                model_symbols=model_symbols,
            )
            for kernel in kernels
        )


def _exact_kernel_components(
    kernel: CompiledOrientedKernel,
) -> tuple[_sym.Expression, ...] | None:
    """Admit only numeric atoms on which the change of basis is exact."""
    from symbolica import AtomType

    expressions = tuple(
        _sym._exact_binary64_coefficients(_sym.E(source))
        for source in kernel.component_expressions
    )
    pending = list(expressions)
    checked: list[_sym.Expression] = []
    exact_units = tuple(_sym.E(str(value)) for value in (0, 1, -1))
    unit_replacements: list[_sym.Replacement] = []
    imaginary_unit = _sym.E("1i")
    while pending:
        atom = pending.pop()
        kind = atom.get_type()
        if kind == AtomType.Var:
            continue
        if kind != AtomType.Num:
            pending.extend(atom)
            continue
        if any(bool(atom.matches(previous)) for previous in checked):
            continue
        checked.append(atom)
        # Numeric equality uses the exact stored rational value of Float atoms
        # in Symbolica, including arbitrary precision. It therefore recognizes
        # floating units without rounding a nearby coefficient onto that unit.
        unit = next((value for value in exact_units if bool(atom == value)), None)
        if unit is not None:
            if not bool(atom.matches(unit)):
                unit_replacements.append(_sym.Replacement(atom, unit))
            continue
        # Symbolica's rational-polynomial conversion rejects floats, but also
        # complex rational numbers. Split the latter and require a literal
        # roundtrip, so this check never promotes a rounded Float to a rational.
        try:
            real = ((atom + atom.conj()) / 2).to_rational_polynomial().to_expression()
            imag = (
                ((atom - atom.conj()) / (2 * imaginary_unit))
                .to_rational_polynomial()
                .to_expression()
            )
        except (TypeError, ValueError):
            return None
        if not bool(atom.matches(real + imaginary_unit * imag)):
            return None
    return tuple(
        expression.replace_multiple(unit_replacements)
        if unit_replacements
        else expression
        for expression in expressions
    )


def _consumer_basis(
    particle: str,
    dimension: int,
    kernels: Sequence[CompiledOrientedKernel],
    *,
    exact_kernels: Mapping[int, tuple[_sym.Expression, ...] | None],
    model_symbols: ModelSymbolRegistry,
) -> _ComponentBasis | None:
    zero = _sym.E("0")
    columns: list[list[_sym.Expression]] = [[] for _ in range(dimension)]
    for kernel in kernels:
        for side, name in zip(("left", "right"), kernel.particles[:2], strict=True):
            if name != particle:
                continue
            currents = tuple(
                model_symbols.kernel_component(kernel.kind, side, index)
                for index in range(dimension)
            )
            current_symbols = set(currents)
            expressions = exact_kernels[kernel.kind]
            assert expressions is not None
            for expression in expressions:
                expanded = expression.expand()
                present = set(expanded.get_all_symbols(False))
                try:
                    coefficients = tuple(
                        expanded.coefficient(current).expand()
                        if current in present
                        else zero
                        for current in currents
                    )
                except (TypeError, ValueError):
                    return None
                if any(
                    current_symbols.intersection(coefficient.get_all_symbols(False))
                    for coefficient in coefficients
                ):
                    return None
                reconstructed = sum(
                    (
                        coefficient * current
                        for coefficient, current in zip(
                            coefficients, currents, strict=True
                        )
                    ),
                    zero,
                )
                if (expanded - reconstructed).expand().to_canonical_string() != "0":
                    return None
                for column, coefficient in zip(columns, coefficients, strict=True):
                    column.append(coefficient)

    groups: list[list[tuple[int, int]]] = []
    known: dict[tuple[_sym.Expression, ...], tuple[int, int]] = {}
    for index, column in enumerate(columns):
        # Compare expanded atoms themselves, independently of text formatting
        # and the parentheses chosen for complex numeric coefficients.
        key = tuple(column)
        if not key or all(value == zero for value in key):
            continue
        existing = known.get(key)
        if existing is None:
            group = len(groups)
            groups.append([(index, 1)])
            known[key] = (group, 1)
            negative = tuple((-value).expand() for value in column)
            known[negative] = (group, -1)
        else:
            group, sign = existing
            groups[group].append((index, sign))
    return tuple(tuple(group) for group in groups)


def _reduce_kernel(
    kernel: CompiledOrientedKernel,
    bases: dict[str, _ComponentBasis],
    dimensions: dict[str, int],
    *,
    exact_kernels: Mapping[int, tuple[_sym.Expression, ...] | None],
    model_symbols: ModelSymbolRegistry,
) -> CompiledOrientedKernel:
    if not any(name in bases for name in kernel.particles):
        return kernel
    zero = _sym.E("0")
    expressions = exact_kernels[kernel.kind]
    assert expressions is not None
    output_basis = bases.get(kernel.particles[2])
    if output_basis is not None:
        expressions = tuple(
            sum((sign * expressions[index] for index, sign in group), zero)
            for group in output_basis
        )
    replacements: list[_sym.Replacement] = []
    for side, name in zip(("left", "right"), kernel.particles[:2], strict=True):
        if name not in bases:
            continue
        representatives = {
            group[0][0]: model_symbols.kernel_component(kernel.kind, side, index)
            for index, group in enumerate(bases[name])
        }
        replacements.extend(
            _sym.Replacement(
                model_symbols.kernel_component(kernel.kind, side, index),
                representatives.get(index, zero),
            )
            for index in range(dimensions[name])
        )
    return replace(
        kernel,
        component_expressions=tuple(
            _canonicalize_oriented_kernel_component(
                expression.replace_multiple(replacements)
                if replacements
                else expression
            ).to_canonical_string()
            for expression in expressions
        ),
    )
