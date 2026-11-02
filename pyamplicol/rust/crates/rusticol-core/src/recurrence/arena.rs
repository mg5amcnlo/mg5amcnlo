// SPDX-License-Identifier: 0BSD

//! Deterministic physical component-range assignment for direct recurrence arenas.
//!
//! Semantic current IDs remain stable in the authenticated plan. The direct
//! runtime stores only simultaneously live currents in its split-complex
//! workspace, so non-overlapping liveness intervals may share component
//! planes. This module is independent of model and process semantics.

use super::RecurrenceProgram;
use crate::direct_arena::{
    DirectArenaInterval as GenericDirectArenaInterval,
    assign_direct_arena as assign_generic_direct_arena,
};
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Inclusive liveness interval and required component-plane width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectArenaInterval {
    pub semantic_current_id: u32,
    pub first_use: u64,
    pub last_use: u64,
    pub component_count: u32,
}

impl DirectArenaInterval {
    pub fn new(
        semantic_current_id: u32,
        first_use: u64,
        last_use: u64,
        component_count: u32,
    ) -> RusticolResult<Self> {
        if first_use > last_use {
            return Err(invalid(format!(
                "direct-arena current {semantic_current_id} starts at {first_use} after its last use {last_use}"
            )));
        }
        if component_count == 0 {
            return Err(invalid(format!(
                "direct-arena current {semantic_current_id} has no components"
            )));
        }
        Ok(Self {
            semantic_current_id,
            first_use,
            last_use,
            component_count,
        })
    }
}

/// One semantic current's physical range in the split-complex arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectArenaAssignment {
    pub semantic_current_id: u32,
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

    pub fn assignment(&self, semantic_current_id: u32) -> Option<DirectArenaAssignment> {
        self.assignments
            .get(semantic_current_id as usize)
            .copied()
            .filter(|assignment| assignment.semantic_current_id == semantic_current_id)
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

        let mut semantic_components = 0u64;
        for (index, assignment) in self.assignments.iter().copied().enumerate() {
            if assignment.semantic_current_id != index as u32 {
                return Err(invalid(format!(
                    "direct-arena assignment row {index} has semantic current ID {}",
                    assignment.semantic_current_id
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
                "direct-arena semantic component total {} does not match recorded {}",
                semantic_components, self.total_semantic_components
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
                if !lifetimes_overlap {
                    continue;
                }
                let ranges_overlap = left.component_base < right.component_stop()?
                    && right.component_base < left.component_stop()?;
                if ranges_overlap {
                    return Err(invalid(format!(
                        "live direct-arena currents {} and {} overlap physical components",
                        left.semantic_current_id, right.semantic_current_id
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Assign variable-width current intervals to reusable physical component ranges.
///
/// Inputs must use dense semantic current IDs. The allocator processes
/// intervals by `(first_use, semantic_current_id)`, releases ranges whose
/// inclusive lifetime ended before the next current starts, and chooses the
/// smallest fitting free range with the lowest base as a deterministic tie
/// breaker.
pub fn assign_direct_arena(intervals: &[DirectArenaInterval]) -> RusticolResult<DirectArenaLayout> {
    let generic_intervals = intervals
        .iter()
        .map(|interval| {
            GenericDirectArenaInterval::new(
                interval.semantic_current_id,
                interval.first_use,
                interval.last_use,
                interval.component_count,
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let generic_layout = assign_generic_direct_arena(&generic_intervals)?;
    let assignments = generic_layout
        .assignments()
        .iter()
        .map(|assignment| DirectArenaAssignment {
            semantic_current_id: assignment.semantic_value_id,
            component_base: assignment.component_base,
            component_count: assignment.component_count,
            first_use: assignment.first_use,
            last_use: assignment.last_use,
        })
        .collect::<Vec<_>>();
    Ok(DirectArenaLayout {
        assignments: assignments.into_boxed_slice(),
        component_count: generic_layout.component_count(),
        total_semantic_components: generic_layout.total_semantic_components(),
        reused_semantic_components: generic_layout.reused_semantic_components(),
    })
}

/// Derive conservative stage-granular liveness for a semantic recurrence program.
///
/// Direct contribution packets may interleave destinations within one stage,
/// so every current produced in that stage is considered simultaneously live.
/// A current remains live through the latest child stage that reads it or
/// through the final closure event. This permits reuse between dependency
/// stages without relying on packet ordering.
pub fn recurrence_direct_arena_layout(
    program: &RecurrenceProgram,
    component_counts: &[u32],
) -> RusticolResult<DirectArenaLayout> {
    if component_counts.len() != program.currents().len() {
        return Err(invalid(format!(
            "direct-arena component-count catalog has {} rows for {} currents",
            component_counts.len(),
            program.currents().len()
        )));
    }

    let current_stages = program
        .currents()
        .iter()
        .map(|current| {
            u64::try_from(current.key().support_source_slots().len())
                .map_err(|_| invalid("direct-arena current stage exceeds u64"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let maximum_stage = current_stages.iter().copied().max().unwrap_or(0);
    let closure_event = maximum_stage
        .checked_add(1)
        .ok_or_else(|| invalid("direct-arena closure event overflows u64"))?;
    let mut last_uses = current_stages.clone();

    for contribution in program.contributions() {
        let result_stage = *current_stages
            .get(contribution.result_current_id() as usize)
            .ok_or_else(|| invalid("direct-arena contribution result current is absent"))?;
        for &parent_id in contribution.parent_current_ids() {
            let last_use = last_uses
                .get_mut(parent_id as usize)
                .ok_or_else(|| invalid("direct-arena contribution parent current is absent"))?;
            *last_use = (*last_use).max(result_stage);
        }
    }
    for closure in program.closure_terms() {
        for &parent_id in closure.parent_current_ids() {
            let last_use = last_uses
                .get_mut(parent_id as usize)
                .ok_or_else(|| invalid("direct-arena closure parent current is absent"))?;
            *last_use = closure_event;
        }
    }

    let intervals = program
        .currents()
        .iter()
        .zip(component_counts.iter().copied())
        .enumerate()
        .map(|(index, (current, component_count))| {
            if current.id() as usize != index {
                return Err(invalid(
                    "direct-arena lowering requires dense semantic current IDs",
                ));
            }
            DirectArenaInterval::new(
                current.id(),
                current_stages[index],
                last_uses[index],
                component_count,
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    assign_direct_arena(&intervals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(id: u32, first: u64, last: u64, count: u32) -> DirectArenaInterval {
        DirectArenaInterval::new(id, first, last, count).unwrap()
    }

    #[test]
    fn non_overlapping_currents_reuse_the_same_range() {
        let layout = assign_direct_arena(&[
            interval(0, 0, 2, 4),
            interval(1, 3, 5, 4),
            interval(2, 6, 8, 4),
        ])
        .unwrap();
        assert_eq!(layout.component_count(), 4);
        assert_eq!(layout.total_semantic_components(), 12);
        assert_eq!(layout.reused_semantic_components(), 8);
        assert_eq!(layout.assignment(0).unwrap().component_base, 0);
        assert_eq!(layout.assignment(1).unwrap().component_base, 0);
        assert_eq!(layout.assignment(2).unwrap().component_base, 0);
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
    fn allocator_rejects_sparse_or_duplicate_semantic_ids() {
        assert!(assign_direct_arena(&[interval(1, 0, 1, 1)]).is_err());
        assert!(assign_direct_arena(&[interval(0, 0, 1, 1), interval(0, 2, 3, 1)]).is_err());
    }

    #[test]
    fn validation_rejects_overlapping_live_ranges() {
        let layout = DirectArenaLayout {
            assignments: vec![
                DirectArenaAssignment {
                    semantic_current_id: 0,
                    component_base: 0,
                    component_count: 2,
                    first_use: 0,
                    last_use: 2,
                },
                DirectArenaAssignment {
                    semantic_current_id: 1,
                    component_base: 1,
                    component_count: 2,
                    first_use: 1,
                    last_use: 3,
                },
            ]
            .into_boxed_slice(),
            component_count: 3,
            total_semantic_components: 4,
            reused_semantic_components: 1,
        };
        assert!(layout.validate().is_err());
    }
}
