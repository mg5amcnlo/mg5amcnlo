# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from functools import cache
from numbers import Number
from typing import Any

from ..base import Model, Vertex
from ..expressions import (
    _expr_minkowski_dot,
    _index_particle_id,
)


@cache
def _exact_imaginary_inverse_sqrt_two() -> Any:
    from symbolica import E

    return E("1i/sqrt(2)")


def _imaginary_inverse_sqrt_two(*values: Any) -> Any:
    """Keep symbolic vertices exact without changing direct numeric evaluation."""

    if all(isinstance(value, Number) for value in values):
        return 1j / math.sqrt(2.0)
    return _exact_imaginary_inverse_sqrt_two()


def _flat_index(indices: tuple[int, ...], dims: tuple[int, ...]) -> int:
    index = 0
    for value, dim in zip(indices, dims, strict=True):
        index = index * dim + value
    return index


def _index_chirality(index: Any) -> int:
    return int(getattr(index, "chirality", 0))


def _expr_vector_slash_terms(
    vector: tuple[Any, ...],
) -> tuple[Any, Any, Any, Any]:
    v0, v1, v2, v3 = vector
    return v0 + v3, v0 - v3, v1 + 1j * v2, v1 - 1j * v2


def _builtin_vertex_result_chiralities(
    model: Model,
    vertex: Vertex,
    left_index: Any,
    right_index: Any,
) -> tuple[int, ...]:
    result_pdg = vertex.particles[2]
    if model.is_chiral_eligible(result_pdg):
        input_chirality = _model_fermion_input_chirality(model, left_index, right_index)
        if input_chirality == 0:
            return (0,)
        if not _model_weyl_vertex_allowed(vertex, result_pdg, input_chirality):
            return ()
        return (input_chirality,)

    if _model_is_fermion_pair_to_vector_vertex(vertex.kind):
        left_pdg = _index_particle_id(left_index)
        right_pdg = _index_particle_id(right_index)
        left_chirality = (
            _index_chirality(left_index) if model.is_fermion(left_pdg) else 0
        )
        right_chirality = (
            _index_chirality(right_index) if model.is_fermion(right_pdg) else 0
        )
        if (
            left_chirality != 0
            and right_chirality != 0
            and left_chirality != -right_chirality
        ):
            return ()
        if not _model_fermion_pair_vector_coupling_allowed(
            vertex,
            left_chirality,
            right_chirality,
        ):
            return ()
    return (0,)


def _model_fermion_input_chirality(
    model: Model,
    left_index: Any,
    right_index: Any,
) -> int:
    left_pdg = _index_particle_id(left_index)
    right_pdg = _index_particle_id(right_index)
    left_chirality = _index_chirality(left_index)
    right_chirality = _index_chirality(right_index)
    if model.is_fermion(left_pdg) and left_chirality != 0:
        return left_chirality
    if model.is_fermion(right_pdg) and right_chirality != 0:
        return right_chirality
    return 0


def _model_weyl_vertex_allowed(
    vertex: Vertex,
    result_pdg: int,
    chirality: int,
) -> bool:
    if vertex.kind == 16:
        return False
    if vertex.kind in {10, 11, 23, 24}:
        index = _model_fermion_coupling_index(result_pdg, chirality)
        return vertex.coupling[index] != 0.0
    return True


def _model_fermion_coupling_index(pdg: int, chirality: int) -> int:
    if pdg > 0:
        return 0 if chirality == -1 else 1
    return 0 if chirality == 1 else 1


def _model_is_fermion_pair_to_vector_vertex(kind: int) -> bool:
    return kind in {8, 9, 21, 22}


def _model_fermion_pair_vector_coupling_allowed(
    vertex: Vertex,
    left_chirality: int,
    right_chirality: int,
) -> bool:
    if vertex.kind in {8, 9}:
        return True
    if left_chirality == 0 or right_chirality == 0:
        return any(component != 0.0 for component in vertex.coupling)
    if vertex.kind == 21:
        index = 0 if left_chirality == -1 and right_chirality == 1 else 1
        return vertex.coupling[index] != 0.0
    if vertex.kind == 22:
        index = 0 if left_chirality == 1 and right_chirality == -1 else 1
        return vertex.coupling[index] != 0.0
    return True


def _expr_three_vector_current(
    left: tuple[Any, ...],
    left_momentum: tuple[Any, ...],
    right: tuple[Any, ...],
    right_momentum: tuple[Any, ...],
) -> tuple[Any, ...]:
    dot = _expr_minkowski_dot(left, right)
    left_dot_right_momentum = _expr_minkowski_dot(left, right_momentum)
    right_dot_left_momentum = _expr_minkowski_dot(right, left_momentum)
    prefactor = _imaginary_inverse_sqrt_two(
        *left, *left_momentum, *right, *right_momentum
    )
    return tuple(
        prefactor
        * (
            dot * (left_momentum[index] - right_momentum[index])
            + 2.0
            * (
                left_dot_right_momentum * right[index]
                - right_dot_left_momentum * left[index]
            )
        )
        for index in range(4)
    )


def _expr_three_vector_current_coupled(
    left: tuple[Any, ...],
    left_momentum: tuple[Any, ...],
    right: tuple[Any, ...],
    right_momentum: tuple[Any, ...],
    coupling: tuple[Any, Any],
) -> tuple[Any, ...]:
    dot = _expr_minkowski_dot(left, right)
    tmp2_momentum = tuple(
        2.0 * right_momentum[index] + left_momentum[index] for index in range(4)
    )
    tmp3_momentum = tuple(
        -2.0 * left_momentum[index] - right_momentum[index] for index in range(4)
    )
    tmp2 = _expr_minkowski_dot(left, tmp2_momentum)
    tmp3 = _expr_minkowski_dot(right, tmp3_momentum)
    prefactor = (
        _imaginary_inverse_sqrt_two(
            *left, *left_momentum, *right, *right_momentum, *coupling
        )
        * coupling[0]
    )
    return tuple(
        prefactor
        * (
            dot * (left_momentum[index] - right_momentum[index])
            + tmp2 * right[index]
            + tmp3 * left[index]
        )
        for index in range(4)
    )


def _expr_two_vector_to_tensor(
    left: tuple[Any, ...],
    right: tuple[Any, ...],
) -> tuple[Any, ...]:
    return (
        left[0] * right[1] - left[1] * right[0],
        left[0] * right[2] - left[2] * right[0],
        left[0] * right[3] - left[3] * right[0],
        left[1] * right[2] - left[2] * right[1],
        left[1] * right[3] - left[3] * right[1],
        left[2] * right[3] - left[3] * right[2],
    )


def _expr_tensor_vector_to_vector(
    tensor: tuple[Any, ...],
    vector: tuple[Any, ...],
) -> tuple[Any, ...]:
    prefactor = 0.5j
    return (
        (tensor[0] * vector[1] + tensor[1] * vector[2] + tensor[2] * vector[3])
        * prefactor,
        (tensor[0] * vector[0] + tensor[3] * vector[2] + tensor[4] * vector[3])
        * prefactor,
        (tensor[1] * vector[0] - tensor[3] * vector[1] + tensor[5] * vector[3])
        * prefactor,
        (tensor[2] * vector[0] - tensor[4] * vector[1] - tensor[5] * vector[2])
        * prefactor,
    )


def _expr_vector_tensor_to_vector(
    vector: tuple[Any, ...],
    tensor: tuple[Any, ...],
) -> tuple[Any, ...]:
    prefactor = 0.5j
    return (
        (-vector[1] * tensor[0] - vector[2] * tensor[1] - vector[3] * tensor[2])
        * prefactor,
        (-vector[0] * tensor[0] - vector[2] * tensor[3] - vector[3] * tensor[4])
        * prefactor,
        (-vector[0] * tensor[1] + vector[1] * tensor[3] - vector[3] * tensor[5])
        * prefactor,
        (-vector[0] * tensor[2] + vector[1] * tensor[4] + vector[2] * tensor[5])
        * prefactor,
    )


def _expr_fermion_vector_weyl(
    fermion: tuple[Any, ...],
    vector: tuple[Any, ...],
    chirality: int,
    *,
    antifermion: bool,
    coupling: tuple[Any, Any] | None,
) -> tuple[Any, ...]:
    tmp1, tmp2, tmp3, tmp4 = _expr_vector_slash_terms(vector)
    prefactor = _imaginary_inverse_sqrt_two(*fermion, *vector, *(coupling or ()))
    f1, f2 = fermion
    if antifermion:
        if chirality == 1:
            factor = prefactor if coupling is None else prefactor * coupling[0]
            return (
                factor * (tmp1 * f1 + tmp4 * f2),
                factor * (tmp2 * f2 + tmp3 * f1),
            )
        if chirality == -1:
            factor = prefactor if coupling is None else prefactor * coupling[1]
            return (
                factor * (tmp2 * f1 - tmp4 * f2),
                factor * (tmp1 * f2 - tmp3 * f1),
            )
    else:
        if chirality == 1:
            factor = prefactor if coupling is None else prefactor * coupling[1]
            return (
                factor * (tmp2 * f1 - tmp3 * f2),
                factor * (tmp1 * f2 - tmp4 * f1),
            )
        if chirality == -1:
            factor = prefactor if coupling is None else prefactor * coupling[0]
            return (
                factor * (tmp1 * f1 + tmp3 * f2),
                factor * (tmp2 * f2 + tmp4 * f1),
            )
    raise ValueError("fermion-vector Weyl kernel needs nonzero chirality")


def _embed_weyl_current_in_dirac(
    current: tuple[Any, ...],
    chirality: int,
) -> tuple[Any, ...]:
    """Embed a massless chiral current in the built-in Dirac basis."""

    if len(current) != 2:
        raise ValueError("Weyl-to-Dirac embedding expects two components")
    zero = current[0] * 0
    if chirality == -1:
        return (*current, zero, zero)
    if chirality == 1:
        return (zero, zero, *current)
    raise ValueError("Weyl-to-Dirac embedding needs nonzero chirality")


def _expr_fermion_vector_dirac(
    fermion: tuple[Any, ...],
    vector: tuple[Any, ...],
    *,
    antifermion: bool,
    coupling: tuple[Any, Any] | None,
) -> tuple[Any, ...]:
    if len(fermion) != 4 or len(vector) != 4:
        raise ValueError("Dirac fermion-vector current expects dimensions 4 and 4")
    tmp1, tmp2, tmp3, tmp4 = _expr_vector_slash_terms(vector)
    prefactor = _imaginary_inverse_sqrt_two(*fermion, *vector, *(coupling or ()))
    f1, f2, f3, f4 = fermion
    if coupling is None:
        left_coupling = 1.0
        right_coupling = 1.0
    else:
        left_coupling, right_coupling = coupling
    if antifermion:
        upper = prefactor * right_coupling
        lower = prefactor * left_coupling
        return (
            upper * (tmp2 * f3 - tmp4 * f4),
            upper * (tmp1 * f4 - tmp3 * f3),
            lower * (tmp1 * f1 + tmp4 * f2),
            lower * (tmp2 * f2 + tmp3 * f1),
        )
    upper = prefactor * left_coupling
    lower = prefactor * right_coupling
    return (
        upper * (tmp1 * f3 + tmp3 * f4),
        upper * (tmp2 * f4 + tmp4 * f3),
        lower * (tmp2 * f1 - tmp3 * f2),
        lower * (tmp1 * f2 - tmp4 * f1),
    )


def _expr_fermion_antifermion_to_vector_weyl(
    *,
    fermion: tuple[Any, ...],
    antifermion: tuple[Any, ...],
    coupling: tuple[Any, Any],
    fermion_chirality: int,
    antifermion_chirality: int,
) -> tuple[Any, ...]:
    prefactor = _imaginary_inverse_sqrt_two(*fermion, *antifermion, *coupling)
    left, right = coupling
    f1, f2 = fermion
    a1, a2 = antifermion
    if fermion_chirality == -1 and antifermion_chirality == 1:
        factor = prefactor * left
        return (
            factor * (f1 * a1 + f2 * a2),
            -factor * (f2 * a1 + f1 * a2),
            1j * factor * (-f2 * a1 + f1 * a2),
            factor * (-f1 * a1 + f2 * a2),
        )
    if fermion_chirality == 1 and antifermion_chirality == -1:
        factor = prefactor * right
        return (
            factor * (f1 * a1 + f2 * a2),
            factor * (f1 * a2 + f2 * a1),
            1j * factor * (-f1 * a2 + f2 * a1),
            factor * (f1 * a1 - f2 * a2),
        )
    return (0j, 0j, 0j, 0j)


def _expr_fermion_antifermion_to_vector_dirac(
    *,
    fermion: tuple[Any, ...],
    antifermion: tuple[Any, ...],
    coupling: tuple[Any, Any],
) -> tuple[Any, ...]:
    if len(fermion) != 4 or len(antifermion) != 4:
        raise ValueError(
            "Dirac fermion-antifermion vector current expects dimensions 4 and 4"
        )
    prefactor = _imaginary_inverse_sqrt_two(*fermion, *antifermion, *coupling)
    left_coupling, right_coupling = coupling
    f1, f2, f3, f4 = fermion
    a1, a2, a3, a4 = antifermion
    left = (
        f3 * a1 + f4 * a2,
        -(f4 * a1 + f3 * a2),
        1j * (-f4 * a1 + f3 * a2),
        -f3 * a1 + f4 * a2,
    )
    right = (
        f1 * a3 + f2 * a4,
        f1 * a4 + f2 * a3,
        1j * (-f1 * a4 + f2 * a3),
        f1 * a3 - f2 * a4,
    )
    return tuple(
        prefactor * (left_coupling * left[index] + right_coupling * right[index])
        for index in range(4)
    )


def _expr_fermion_scalar_to_fermion(
    fermion: tuple[Any, ...],
    scalar: tuple[Any, ...],
    coupling: tuple[Any, Any],
) -> tuple[Any, ...]:
    if len(fermion) != 4 or len(scalar) != 1:
        raise ValueError("Dirac fermion-scalar current expects dimensions 4 and 1")
    prefactor = -_imaginary_inverse_sqrt_two(*fermion, *scalar, *coupling)
    return tuple(
        prefactor * coupling[0] * scalar[0] * component for component in fermion
    )


def _two_gluon_to_tensor_data() -> list[complex]:
    data = [0j] * (6 * 4 * 4)
    metric = (1.0, -1.0, -1.0, -1.0)
    for tensor_index, (i, j) in enumerate(_ANTISYM_PAIRS):
        data[_flat_index((i, j, tensor_index), (4, 4, 6))] = (
            1.0 / (metric[i] * metric[j])
        ) + 0j
        data[_flat_index((j, i, tensor_index), (4, 4, 6))] = (
            -1.0 / (metric[j] * metric[i])
        ) + 0j
    return data


def _tensor_gluon_to_gluon_data() -> list[complex]:
    data = [0j] * (6 * 4 * 4)
    prefactor = 0.5j
    metric = (1.0, -1.0, -1.0, -1.0)
    rows = {
        0: ((0, 1, 1), (1, 2, 1), (2, 3, 1)),
        1: ((0, 0, 1), (3, 2, 1), (4, 3, 1)),
        2: ((1, 0, 1), (3, 1, -1), (5, 3, 1)),
        3: ((2, 0, 1), (4, 1, -1), (5, 2, -1)),
    }
    for out, entries in rows.items():
        for tensor_index, gluon_index, sign in entries:
            data[_flat_index((tensor_index, gluon_index, out), (6, 4, 4))] = (
                sign * prefactor / metric[gluon_index]
            )
    return data


def _gluon_tensor_to_gluon_data() -> list[complex]:
    data = [0j] * (6 * 4 * 4)
    prefactor = 0.5j
    metric = (1.0, -1.0, -1.0, -1.0)
    rows = {
        0: ((1, 0, -1), (2, 1, -1), (3, 2, -1)),
        1: ((0, 0, -1), (2, 3, -1), (3, 4, -1)),
        2: ((0, 1, -1), (1, 3, 1), (3, 5, -1)),
        3: ((0, 2, -1), (1, 4, 1), (2, 5, 1)),
    }
    for out, entries in rows.items():
        for gluon_index, tensor_index, sign in entries:
            data[_flat_index((gluon_index, tensor_index, out), (4, 6, 4))] = (
                sign * prefactor / metric[gluon_index]
            )
    return data


def _quark_vector_weyl_data(*, chirality: int) -> list[Any]:
    prefactor = _exact_imaginary_inverse_sqrt_two()
    data: list[Any] = [prefactor * 0] * (2 * 4 * 2)
    metric = (1.0, -1.0, -1.0, -1.0)

    def add(q_in: int, vector: int, q_out: int, coefficient: Any) -> None:
        # Tensor.dense accepts logical interface order, not canonical storage order.
        data[_flat_index((q_in, vector, q_out), (2, 4, 2))] = (
            coefficient / metric[vector]
        )

    if chirality == 1:
        add(0, 0, 0, prefactor)
        add(0, 3, 0, -prefactor)
        add(1, 1, 0, -prefactor)
        add(1, 2, 0, -1j * prefactor)
        add(1, 0, 1, prefactor)
        add(1, 3, 1, prefactor)
        add(0, 1, 1, -prefactor)
        add(0, 2, 1, 1j * prefactor)
        return data
    if chirality == -1:
        add(0, 0, 0, prefactor)
        add(0, 3, 0, prefactor)
        add(1, 1, 0, prefactor)
        add(1, 2, 0, 1j * prefactor)
        add(1, 0, 1, prefactor)
        add(1, 3, 1, -prefactor)
        add(0, 1, 1, prefactor)
        add(0, 2, 1, -1j * prefactor)
        return data
    raise ValueError(f"unsupported Weyl chirality: {chirality}")


_ANTISYM_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
