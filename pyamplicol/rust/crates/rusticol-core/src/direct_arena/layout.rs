// SPDX-License-Identifier: 0BSD

//! Deterministic variable-width interval allocation for arena planes.

use std::collections::{BTreeMap, BTreeSet};

use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Inclusive liveness interval and required component-plane width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectArenaInterval {
    pub semantic_value_id: u32,
    pub first_use: u64,
    pub last_use: u64,
    pub component_count: u32,
}

impl DirectArenaInterval {
    pub fn new(
        semantic_value_id: u32,
        first_use: u64,
        last_use: u64,
        component_count: u32,
    ) -> RusticolResult<Self> {
        if first_use > last_use {
            return Err(invalid(format!(
                "direct-arena value {semantic_value_id} starts at {first_use} after its last use {last_use}"
            )));
        }
        if component_count == 0 {
            return Err(invalid(format!(
                "direct-arena value {semantic_value_id} has no components"
            )));
        }
        Ok(Self {
            semantic_value_id,
            first_use,
            last_use,
            component_count,
        })
    }
}

/// One semantic value's physical component range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectArenaAssignment {
    pub semantic_value_id: u32,
    pub component_base: u32,
    pub component_count: u32,
    pub first_use: u64,
    pub last_use: u64,
}

impl DirectArenaAssignment {
    pub fn component_stop(self) -> RusticolResult<u32> {
        self.component_base
            .checked_add(self.component_count)
            .ok_or_else(|| invalid("direct-arena component range overflows u32"))
    }
}

/// Deterministic interval-coloring result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectArenaLayout {
    assignments: Box<[DirectArenaAssignment]>,
    component_count: u32,
    total_semantic_components: u64,
    reused_semantic_components: u64,
}

impl DirectArenaLayout {
    pub fn assignments(&self) -> &[DirectArenaAssignment] {
        &self.assignments
    }

    pub const fn component_count(&self) -> u32 {
        self.component_count
    }

    pub const fn total_semantic_components(&self) -> u64 {
        self.total_semantic_components
    }

    pub const fn reused_semantic_components(&self) -> u64 {
        self.reused_semantic_components
    }

    pub fn assignment(&self, semantic_value_id: u32) -> Option<DirectArenaAssignment> {
        self.assignments
            .get(semantic_value_id as usize)
            .copied()
            .filter(|assignment| assignment.semantic_value_id == semantic_value_id)
    }

    pub fn validate(&self) -> RusticolResult<()> {
        if self.assignments.is_empty() {
            if self.component_count != 0
                || self.total_semantic_components != 0
                || self.reused_semantic_components != 0
            {
                return Err(invalid("empty direct-arena layout has nonempty accounting"));
            }
            return Ok(());
        }

        let mut semantic_components = 0_u64;
        for (index, assignment) in self.assignments.iter().copied().enumerate() {
            if assignment.semantic_value_id != index as u32 {
                return Err(invalid(format!(
                    "direct-arena assignment row {index} has semantic value ID {}",
                    assignment.semantic_value_id
                )));
            }
            if assignment.component_count == 0 || assignment.first_use > assignment.last_use {
                return Err(invalid(format!(
                    "direct-arena assignment {index} has invalid width or liveness"
                )));
            }
            if assignment.component_stop()? > self.component_count {
                return Err(invalid(format!(
                    "direct-arena assignment {index} exceeds the physical arena"
                )));
            }
            semantic_components = semantic_components
                .checked_add(u64::from(assignment.component_count))
                .ok_or_else(|| invalid("direct-arena semantic component count overflows u64"))?;
        }
        if semantic_components != self.total_semantic_components {
            return Err(invalid(format!(
                "direct-arena semantic component total {semantic_components} does not match recorded {}",
                self.total_semantic_components
            )));
        }
        if self
            .total_semantic_components
            .checked_sub(u64::from(self.component_count))
            != Some(self.reused_semantic_components)
        {
            return Err(invalid(
                "direct-arena reused component accounting is inconsistent",
            ));
        }

        for (left_index, left) in self.assignments.iter().copied().enumerate() {
            for right in self.assignments[left_index + 1..].iter().copied() {
                let lifetimes_overlap =
                    left.first_use <= right.last_use && right.first_use <= left.last_use;
                let ranges_overlap = left.component_base < right.component_stop()?
                    && right.component_base < left.component_stop()?;
                if lifetimes_overlap && ranges_overlap {
                    return Err(invalid(format!(
                        "live direct-arena values {} and {} overlap physical components",
                        left.semantic_value_id, right.semantic_value_id
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct FreeRanges {
    by_base: BTreeMap<u32, u32>,
    by_size: BTreeSet<(u32, u32)>,
}

impl FreeRanges {
    fn take_best_fit(&mut self, required: u32) -> RusticolResult<Option<(u32, u32)>> {
        let Some(&(component_count, component_base)) = self.by_size.range((required, 0)..).next()
        else {
            return Ok(None);
        };
        self.remove(component_base, component_count)?;
        Ok(Some((component_base, component_count)))
    }

    fn insert(&mut self, mut component_base: u32, mut component_count: u32) -> RusticolResult<()> {
        if component_count == 0 {
            return Err(invalid("cannot release an empty direct-arena range"));
        }

        if let Some((&previous_base, &previous_count)) =
            self.by_base.range(..component_base).next_back()
        {
            let previous_stop = previous_base
                .checked_add(previous_count)
                .ok_or_else(|| invalid("direct-arena free range overflows u32"))?;
            if previous_stop > component_base {
                return Err(invalid("released direct-arena ranges overlap"));
            }
            if previous_stop == component_base {
                self.remove(previous_base, previous_count)?;
                component_base = previous_base;
                component_count = component_count
                    .checked_add(previous_count)
                    .ok_or_else(|| invalid("direct-arena merged range overflows u32"))?;
            }
        }

        let component_stop = component_base
            .checked_add(component_count)
            .ok_or_else(|| invalid("direct-arena free range overflows u32"))?;
        if let Some((&next_base, &next_count)) = self.by_base.range(component_base..).next() {
            if component_stop > next_base {
                return Err(invalid("released direct-arena ranges overlap"));
            }
            if component_stop == next_base {
                self.remove(next_base, next_count)?;
                component_count = component_count
                    .checked_add(next_count)
                    .ok_or_else(|| invalid("direct-arena merged range overflows u32"))?;
            }
        }

        if self
            .by_base
            .insert(component_base, component_count)
            .is_some()
            || !self.by_size.insert((component_count, component_base))
        {
            return Err(invalid("released direct-arena range repeats its base"));
        }
        Ok(())
    }

    fn remove(&mut self, component_base: u32, component_count: u32) -> RusticolResult<()> {
        if self.by_base.remove(&component_base) != Some(component_count)
            || !self.by_size.remove(&(component_count, component_base))
        {
            return Err(invalid("direct-arena free-range indexes disagree"));
        }
        Ok(())
    }
}

/// Assign variable-width semantic intervals to reusable physical planes.
///
/// Inputs use dense semantic value IDs. Intervals are processed by
/// `(first_use, semantic_value_id)`. The allocator releases ranges whose
/// inclusive lifetime ended before the next value starts, then chooses the
/// smallest fitting range with the lowest base as a deterministic tie-breaker.
pub fn assign_direct_arena(intervals: &[DirectArenaInterval]) -> RusticolResult<DirectArenaLayout> {
    if intervals.is_empty() {
        return Ok(DirectArenaLayout {
            assignments: Box::new([]),
            component_count: 0,
            total_semantic_components: 0,
            reused_semantic_components: 0,
        });
    }

    let mut ordered = intervals.to_vec();
    ordered.sort_by_key(|interval| (interval.first_use, interval.semantic_value_id));
    let mut releases = ordered.clone();
    releases.sort_by_key(|interval| (interval.last_use, interval.semantic_value_id));
    let mut seen = vec![false; intervals.len()];
    for interval in &ordered {
        let id = interval.semantic_value_id as usize;
        if id >= seen.len() || seen[id] {
            return Err(invalid(
                "direct-arena semantic value IDs must be dense and unique",
            ));
        }
        seen[id] = true;
        if interval.component_count == 0 || interval.first_use > interval.last_use {
            return Err(invalid(format!(
                "direct-arena value {} has invalid width or liveness",
                interval.semantic_value_id
            )));
        }
    }
    if seen.contains(&false) {
        return Err(invalid(
            "direct-arena semantic value IDs must cover zero through count minus one",
        ));
    }

    let mut assignments = vec![None::<DirectArenaAssignment>; intervals.len()];
    let mut next_release = 0;
    let mut free = FreeRanges::default();
    let mut arena_stop = 0_u32;
    let mut total_semantic_components = 0_u64;

    for interval in ordered {
        while releases
            .get(next_release)
            .is_some_and(|released| released.last_use < interval.first_use)
        {
            let released = releases[next_release];
            let assignment = assignments[released.semantic_value_id as usize].ok_or_else(|| {
                invalid("direct-arena interval was released before it was assigned")
            })?;
            free.insert(assignment.component_base, assignment.component_count)?;
            next_release += 1;
        }

        let (component_base, available_count) = free
            .take_best_fit(interval.component_count)?
            .unwrap_or((arena_stop, 0));

        if available_count == 0 {
            arena_stop = arena_stop
                .checked_add(interval.component_count)
                .ok_or_else(|| invalid("direct-arena physical component count exceeds u32"))?;
        } else {
            let remaining = available_count - interval.component_count;
            if remaining != 0 {
                free.insert(
                    component_base
                        .checked_add(interval.component_count)
                        .ok_or_else(|| invalid("direct-arena free range overflows u32"))?,
                    remaining,
                )?;
            }
        }

        let assignment = DirectArenaAssignment {
            semantic_value_id: interval.semantic_value_id,
            component_base,
            component_count: interval.component_count,
            first_use: interval.first_use,
            last_use: interval.last_use,
        };
        assignments[interval.semantic_value_id as usize] = Some(assignment);
        total_semantic_components = total_semantic_components
            .checked_add(u64::from(interval.component_count))
            .ok_or_else(|| invalid("direct-arena semantic component count exceeds u64"))?;
    }

    let assignments = assignments
        .into_iter()
        .enumerate()
        .map(|(id, assignment)| {
            assignment.ok_or_else(|| {
                invalid(format!(
                    "direct-arena value {id} was not assigned a physical range"
                ))
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    Ok(DirectArenaLayout {
        assignments: assignments.into_boxed_slice(),
        component_count: arena_stop,
        total_semantic_components,
        reused_semantic_components: total_semantic_components - u64::from(arena_stop),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(id: u32, first: u64, last: u64, count: u32) -> DirectArenaInterval {
        DirectArenaInterval::new(id, first, last, count).unwrap()
    }

    fn assign_naively(intervals: &[DirectArenaInterval]) -> DirectArenaLayout {
        let mut ordered = intervals.to_vec();
        ordered.sort_by_key(|row| (row.first_use, row.semantic_value_id));
        let mut assignments = vec![None::<DirectArenaAssignment>; intervals.len()];
        let mut arena_stop = 0_u32;
        let mut total_semantic_components = 0_u64;
        for row in ordered {
            let mut occupied = vec![false; arena_stop as usize];
            for assignment in assignments.iter().flatten() {
                if assignment.last_use < row.first_use {
                    continue;
                }
                let stop = assignment.component_base + assignment.component_count;
                occupied[assignment.component_base as usize..stop as usize].fill(true);
            }
            let mut best = None::<(u32, u32)>;
            let mut cursor = 0usize;
            while cursor < occupied.len() {
                if occupied[cursor] {
                    cursor += 1;
                    continue;
                }
                let start = cursor;
                while cursor < occupied.len() && !occupied[cursor] {
                    cursor += 1;
                }
                let count = (cursor - start) as u32;
                let candidate = (count, start as u32);
                if count >= row.component_count && best.is_none_or(|current| candidate < current) {
                    best = Some(candidate);
                }
            }
            let component_base = best.map_or(arena_stop, |(_, base)| base);
            if best.is_none() {
                arena_stop += row.component_count;
            }
            assignments[row.semantic_value_id as usize] = Some(DirectArenaAssignment {
                semantic_value_id: row.semantic_value_id,
                component_base,
                component_count: row.component_count,
                first_use: row.first_use,
                last_use: row.last_use,
            });
            total_semantic_components += u64::from(row.component_count);
        }
        DirectArenaLayout {
            assignments: assignments.into_iter().map(Option::unwrap).collect(),
            component_count: arena_stop,
            total_semantic_components,
            reused_semantic_components: total_semantic_components - u64::from(arena_stop),
        }
    }

    #[test]
    fn non_overlapping_values_reuse_the_same_range() {
        let layout = assign_direct_arena(&[
            interval(0, 0, 2, 4),
            interval(1, 3, 5, 4),
            interval(2, 6, 8, 4),
        ])
        .unwrap();
        assert_eq!(layout.component_count(), 4);
        assert_eq!(layout.total_semantic_components(), 12);
        assert_eq!(layout.reused_semantic_components(), 8);
        assert!(
            layout
                .assignments()
                .iter()
                .all(|row| row.component_base == 0)
        );
    }

    #[test]
    fn inclusive_lifetimes_do_not_reuse_at_the_same_event() {
        let layout = assign_direct_arena(&[interval(0, 0, 3, 2), interval(1, 3, 4, 3)]).unwrap();
        assert_eq!(layout.component_count(), 5);
        assert_eq!(layout.assignment(0).unwrap().component_base, 0);
        assert_eq!(layout.assignment(1).unwrap().component_base, 2);
    }

    #[test]
    fn allocator_chooses_the_smallest_fitting_range_deterministically() {
        let layout = assign_direct_arena(&[
            interval(0, 0, 1, 8),
            interval(1, 0, 5, 2),
            interval(2, 0, 1, 3),
            interval(3, 2, 4, 2),
            interval(4, 2, 4, 7),
        ])
        .unwrap();
        assert_eq!(layout.assignment(3).unwrap().component_base, 10);
        assert_eq!(layout.assignment(4).unwrap().component_base, 0);
        assert_eq!(layout.component_count(), 13);
    }

    #[test]
    fn assignment_is_stable_for_unsorted_input() {
        let intervals = [
            interval(3, 8, 9, 2),
            interval(0, 0, 3, 4),
            interval(2, 4, 7, 3),
            interval(1, 1, 2, 1),
        ];
        let expected = assign_direct_arena(&intervals).unwrap();
        for _ in 0..32 {
            assert_eq!(assign_direct_arena(&intervals).unwrap(), expected);
        }
    }

    #[test]
    fn indexed_allocator_matches_naive_best_fit_oracle() {
        for seed in 0..48_u32 {
            let mut intervals = (0..32_u32)
                .map(|id| {
                    let first = u64::from((id * 7 + seed * 3) % 11);
                    interval(
                        id,
                        first,
                        first + u64::from((id * 5 + seed) % 6),
                        1 + (id * 3 + seed * 5) % 7,
                    )
                })
                .collect::<Vec<_>>();
            let length = intervals.len();
            intervals.rotate_left(seed as usize % length);
            if seed % 2 != 0 {
                intervals.reverse();
            }
            assert_eq!(
                assign_direct_arena(&intervals).unwrap(),
                assign_naively(&intervals)
            );
        }
    }

    #[test]
    fn malformed_ids_and_component_overflow_fail_closed() {
        assert!(assign_direct_arena(&[interval(1, 0, 1, 1)]).is_err());
        assert!(assign_direct_arena(&[interval(0, 0, 1, 1), interval(0, 2, 3, 1)]).is_err());
        assert!(assign_direct_arena(&[interval(0, 0, 1, u32::MAX), interval(1, 0, 1, 1)]).is_err());
    }
}
