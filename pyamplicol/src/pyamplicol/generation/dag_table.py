# SPDX-License-Identifier: 0BSD
"""Deduplicated current storage used during DAG compilation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..models.base import Model
from .dag_types import CurrentIndex, CurrentNode


class _CurrentTable:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.currents: list[CurrentNode] = []
        self._ids: dict[CurrentIndex, int] = {}
        self._ids_by_mask: dict[int, list[int]] = {}
        self._ids_by_mask_particle: dict[tuple[int, int], list[int]] = {}
        self._ids_by_mask_particle_sector: dict[tuple[int, int, int], list[int]] = {}

    def add_or_get(
        self,
        index: CurrentIndex,
        *,
        is_source: bool,
        source_leg_label: int | None = None,
        source_helicity: int | None = None,
    ) -> CurrentNode:
        current_id = self._ids.get(index)
        if current_id is not None:
            return self.currents[current_id]
        current_id = len(self.currents)
        node = CurrentNode(
            id=current_id,
            index=index,
            dimension=self.model.current_dimension(
                index.particle_id,
                index.chirality,
            ),
            is_source=is_source,
            source_leg_label=source_leg_label,
            source_helicity=source_helicity,
        )
        self._ids[index] = current_id
        self.currents.append(node)
        self._ids_by_mask.setdefault(index.external_mask, []).append(current_id)
        self._ids_by_mask_particle.setdefault(
            (index.external_mask, index.particle_id),
            [],
        ).append(current_id)
        self._ids_by_mask_particle_sector.setdefault(
            (
                index.external_mask,
                index.particle_id,
                index.color_state.sector_id,
            ),
            [],
        ).append(current_id)
        return node

    def current(self, current_id: int) -> CurrentNode:
        return self.currents[current_id]

    def has_mask(self, mask: int) -> bool:
        return mask in self._ids_by_mask

    def ids_by_mask(self, mask: int) -> Sequence[int]:
        return self._ids_by_mask.get(mask, ())

    def ids_by_mask_and_particles(
        self,
        mask: int,
        particle_ids: Iterable[int],
        *,
        color_sector_id: int | None = None,
    ) -> Sequence[int]:
        try:
            particle_count = len(particle_ids)  # type: ignore[arg-type]
        except TypeError:
            particle_count = -1
        if particle_count == 1:
            particle_id = next(iter(particle_ids))
            if color_sector_id is None:
                return self._ids_by_mask_particle.get((mask, particle_id), ())
            return self._ids_by_mask_particle_sector.get(
                (mask, particle_id, color_sector_id),
                (),
            )
        ids: list[int] = []
        for particle_id in particle_ids:
            if color_sector_id is None:
                ids.extend(self._ids_by_mask_particle.get((mask, particle_id), ()))
            else:
                ids.extend(
                    self._ids_by_mask_particle_sector.get(
                        (mask, particle_id, color_sector_id),
                        (),
                    )
                )
        return ids
