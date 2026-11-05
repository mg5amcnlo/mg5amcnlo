# SPDX-License-Identifier: 0BSD
"""Generation-time external-fermion ordering signs.

Numerical current kernels commute their already-evaluated inputs.  Wick
contractions do not: every binary current merge must first restore the
canonical source order of its external fermions, and a two-fermion kernel has
one additional sign when its oriented inputs are particle--antiparticle rather
than antiparticle--particle.  Folding this exact ``+/-1`` into the generated
coefficient realizes the exterior algebra without adding runtime state or
splitting currents by factorially many Wick lineages.

Color-connected fundamental open lines already carry this ordering in their
authenticated color-flow construction.  This pass therefore owns only
color-singlet fermion sources; applying it to fundamental lines would count
the same exchange sign twice.
"""

from __future__ import annotations

from dataclasses import replace

from ..models.base import Model
from ..processes.ir import CanonicalProcessIR
from .dag_types import CurrentIndex, GenericDAG


def _external_fermion_requires_exterior_sign(
    statistics: str,
    color_role: str,
) -> bool:
    """Classify one canonical source without guessing unsupported color roles."""

    if statistics != "fermion":
        return False
    if color_role == "singlet":
        return True
    if color_role in {"fundamental", "antifundamental"}:
        return False
    raise ValueError(
        f"external-fermion ordering does not support fermion color role {color_role!r}"
    )


def _fermion_rep_requires_exterior_sign(color_representation: int) -> bool:
    """Return sign ownership for a model fermion in the supported UFO scope."""

    color_representation = int(color_representation)
    if color_representation == 1:
        return True
    if abs(color_representation) == 3:
        return False
    raise ValueError(
        "external-fermion ordering does not support fermion color "
        f"representation {color_representation}"
    )


def _external_fermion_support_sign(
    left_labels: tuple[int, ...],
    right_labels: tuple[int, ...],
    fermion_source_ranks: dict[int, int],
) -> int:
    """Return the Koszul sign for concatenating two disjoint supports."""

    left_ranks = tuple(
        fermion_source_ranks[label]
        for label in left_labels
        if label in fermion_source_ranks
    )
    right_ranks = tuple(
        fermion_source_ranks[label]
        for label in right_labels
        if label in fermion_source_ranks
    )
    inversions = sum(
        left_rank > right_rank for left_rank in left_ranks for right_rank in right_ranks
    )
    return -1 if inversions % 2 else 1


def _fermion_input_orientation_sign(
    left_is_fermion: bool,
    left_orientation: str,
    left_requires_exterior_sign: bool,
    right_is_fermion: bool,
    right_orientation: str,
    right_requires_exterior_sign: bool,
) -> int:
    """Return the field-order sign of one oriented two-fermion kernel."""

    if not (left_is_fermion and right_is_fermion):
        return 1
    if left_requires_exterior_sign != right_requires_exterior_sign:
        raise ValueError(
            "external-fermion ordering encountered mixed color-sign ownership"
        )
    if not left_requires_exterior_sign:
        return 1
    supported = {"particle", "antiparticle"}
    if left_orientation not in supported or right_orientation not in supported:
        raise ValueError(
            "external-fermion ordering requires non-self-conjugate model orientations"
        )
    pair = (left_orientation, right_orientation)
    if pair == ("antiparticle", "particle"):
        return 1
    if pair == ("particle", "antiparticle"):
        return -1
    raise ValueError("external-fermion ordering requires opposite Dirac orientations")


class FermionOrderingSign:
    """Cached model/process contract for cold DAG construction."""

    __slots__ = (
        "_merge_cache",
        "_model",
        "_particle_contract_cache",
        "_source_ranks",
    )

    def __init__(self, process: CanonicalProcessIR, model: Model) -> None:
        self._model = model
        source_ranks: dict[int, int] = {}
        for source_slot, leg in enumerate(process.legs):
            if _external_fermion_requires_exterior_sign(
                leg.statistics,
                leg.color_role,
            ):
                source_ranks[int(leg.label)] = source_slot
        self._source_ranks = source_ranks
        self._particle_contract_cache: dict[int, tuple[bool, str, bool]] = {}
        self._merge_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], int, int], int
        ] = {}

    @property
    def active(self) -> bool:
        """Whether this process has any source whose sign is owned here."""

        return bool(self._source_ranks)

    def _particle_contract(self, particle_id: int) -> tuple[bool, str, bool]:
        particle_id = int(particle_id)
        cached = self._particle_contract_cache.get(particle_id)
        if cached is not None:
            return cached
        is_fermion = bool(self._model.is_fermion(particle_id))
        orientation = (
            str(self._model.source_orientation(particle_id))
            if is_fermion
            else "self-conjugate"
        )
        requires_exterior_sign = (
            _fermion_rep_requires_exterior_sign(self._model.color_rep(particle_id))
            if is_fermion
            else False
        )
        result = (is_fermion, orientation, requires_exterior_sign)
        self._particle_contract_cache[particle_id] = result
        return result

    def merge_sign(self, left: CurrentIndex, right: CurrentIndex) -> int:
        key = (
            left.external_labels,
            right.external_labels,
            int(left.particle_id),
            int(right.particle_id),
        )
        cached = self._merge_cache.get(key)
        if cached is not None:
            return cached
        (
            left_is_fermion,
            left_orientation,
            left_requires_exterior_sign,
        ) = self._particle_contract(left.particle_id)
        (
            right_is_fermion,
            right_orientation,
            right_requires_exterior_sign,
        ) = self._particle_contract(right.particle_id)
        sign = _external_fermion_support_sign(
            left.external_labels,
            right.external_labels,
            self._source_ranks,
        )
        sign *= _fermion_input_orientation_sign(
            left_is_fermion,
            left_orientation,
            left_requires_exterior_sign,
            right_is_fermion,
            right_orientation,
            right_requires_exterior_sign,
        )
        self._merge_cache[key] = sign
        return sign


def apply_fermion_ordering_signs(dag: GenericDAG, model: Model) -> GenericDAG:
    """Fold every merge sign into existing DAG constants exactly once."""

    if not dag.interactions and not dag.amplitude_roots:
        return dag
    ordering = FermionOrderingSign(dag.process, model)
    if not ordering.active:
        return dag
    currents = dag.currents

    def signed_weight(
        weight: tuple[float, float], left_id: int, right_id: int
    ) -> tuple[float, float]:
        sign = ordering.merge_sign(
            currents[left_id].index,
            currents[right_id].index,
        )
        return weight if sign == 1 else (-weight[0], -weight[1])

    def signed_interaction(interaction):
        color_weight = signed_weight(
            interaction.color_weight,
            interaction.left_id,
            interaction.right_id,
        )
        if color_weight is interaction.color_weight:
            return interaction
        return replace(interaction, color_weight=color_weight)

    def signed_root(root):
        color_weight = signed_weight(
            root.color_weight,
            root.left_id,
            root.right_id,
        )
        if color_weight is root.color_weight:
            return root
        return replace(root, color_weight=color_weight)

    return replace(
        dag,
        interactions=tuple(signed_interaction(row) for row in dag.interactions),
        amplitude_roots=tuple(signed_root(row) for row in dag.amplitude_roots),
    )


__all__ = ["FermionOrderingSign", "apply_fermion_ordering_signs"]
