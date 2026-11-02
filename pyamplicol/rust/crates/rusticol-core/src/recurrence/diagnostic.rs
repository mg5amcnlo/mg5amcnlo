// SPDX-License-Identifier: 0BSD

//! Feature-only construction observer used to diagnose compact/established
//! recurrence disagreements.  Nothing in this module is compiled into default
//! or release builds and none of its records participate in artifact identity.

use std::cell::RefCell;
use std::collections::BTreeSet;

use super::{ExactComplexRational, SemanticDigest, SourceStateAssignment};
use crate::{RusticolError, RusticolResult};

#[derive(Clone, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct ConstructionTransitionDiagnosticRowV1 {
    pub materialized_sector_id: Option<u32>,
    pub output_current_digest: SemanticDigest,
    pub ordered_parent_digests: [SemanticDigest; 2],
    pub transition_template_id: u32,
    pub transition_semantic_digest: SemanticDigest,
    pub evaluator_binding_semantic_digest: SemanticDigest,
    pub result_state_template_id: u32,
    pub quantum_flow_witness_id: u32,
    pub quantum_semantic_digest: SemanticDigest,
    pub color_contraction_template_id: u32,
    pub color_witness_ordinal: u32,
    pub color_witness_proof_digest: SemanticDigest,
    pub output_projection_id: u32,
    pub transition_factor: ExactComplexRational,
    pub contraction_factor: ExactComplexRational,
    pub output_factor: ExactComplexRational,
    pub exchange_factor: ExactComplexRational,
    pub witness_factor: ExactComplexRational,
    pub reversal_mask: u8,
    pub reversal_factor: ExactComplexRational,
    pub candidate_factor: ExactComplexRational,
    pub aggregate_factor_after: ExactComplexRational,
    pub parent_reflection_proof_digests: [Option<SemanticDigest>; 2],
    pub parent_reflection_phases: [Option<ExactComplexRational>; 2],
    pub local_reflection_proof_digest: Option<SemanticDigest>,
    pub local_reflection_phase: Option<ExactComplexRational>,
    pub result_reflection_proof_digest: Option<SemanticDigest>,
    pub result_reflection_phase: Option<ExactComplexRational>,
    pub output_color_orientation: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ConstructionDiagnosticSelectionV1 {
    pub(crate) public_flow_id: u32,
    pub(crate) public_helicities: Vec<i32>,
}

pub(crate) fn representative_source_states_for_public_helicities(
    representative_to_public: &[u32],
    public_helicities: &[i32],
    mut state_index_for_public_helicity: impl FnMut(u32, i32) -> RusticolResult<u32>,
) -> RusticolResult<Vec<SourceStateAssignment>> {
    if representative_to_public.len() != public_helicities.len() {
        return Err(RusticolError::integrity(
            "transition diagnostic replay permutation and public-helicity axis have different sizes",
        ));
    }
    let mut seen_public_slots = vec![false; public_helicities.len()];
    representative_to_public
        .iter()
        .copied()
        .enumerate()
        .map(|(representative_slot, public_slot)| {
            let public_index = usize::try_from(public_slot).map_err(|_| {
                RusticolError::integrity("transition diagnostic public source slot exceeds usize")
            })?;
            let seen = seen_public_slots.get_mut(public_index).ok_or_else(|| {
                RusticolError::integrity(
                    "transition diagnostic replay permutation is out of bounds",
                )
            })?;
            if *seen {
                return Err(RusticolError::integrity(
                    "transition diagnostic replay permutation repeats a public source slot",
                ));
            }
            *seen = true;
            let representative_slot = u32::try_from(representative_slot).map_err(|_| {
                RusticolError::integrity(
                    "transition diagnostic representative source slot exceeds u32",
                )
            })?;
            let state_index = state_index_for_public_helicity(
                representative_slot,
                public_helicities[public_index],
            )?;
            Ok(SourceStateAssignment::new(representative_slot, state_index))
        })
        .collect()
}

#[derive(Debug)]
struct ConstructionDiagnosticObserverV1 {
    rows: Vec<ConstructionTransitionDiagnosticRowV1>,
    selection: Option<ConstructionDiagnosticSelectionV1>,
    live_current_digests: Option<BTreeSet<SemanticDigest>>,
    active_materialized_sector_id: Option<u32>,
}

thread_local! {
    static TRANSITION_OBSERVER: RefCell<Option<ConstructionDiagnosticObserverV1>> =
        const { RefCell::new(None) };
}

pub(crate) fn begin_transition_diagnostic_observation(
    selection: Option<ConstructionDiagnosticSelectionV1>,
) -> RusticolResult<()> {
    TRANSITION_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(RusticolError::integrity(
                "recurrence transition diagnostic observation is already active",
            ));
        }
        *slot = Some(ConstructionDiagnosticObserverV1 {
            rows: Vec::new(),
            selection,
            live_current_digests: None,
            active_materialized_sector_id: None,
        });
        Ok(())
    })
}

pub(crate) fn transition_diagnostic_observation_active() -> bool {
    TRANSITION_OBSERVER.with(|slot| slot.borrow().is_some())
}

#[allow(dead_code)]
pub(crate) fn observe_transition_diagnostic(mut row: ConstructionTransitionDiagnosticRowV1) {
    TRANSITION_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow_mut().as_mut() {
            row.materialized_sector_id = observer.active_materialized_sector_id;
            observer.rows.push(row);
        }
    });
}

pub(crate) fn with_transition_diagnostic_materialized_sector<T>(
    materialized_sector_id: Option<u32>,
    build: impl FnOnce() -> RusticolResult<T>,
) -> RusticolResult<T> {
    let activated = TRANSITION_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(observer) = slot.as_mut() else {
            return Ok(false);
        };
        let materialized_sector_id = materialized_sector_id.ok_or_else(|| {
            RusticolError::integrity(
                "transition diagnostic construction group is not one materialized sector",
            )
        })?;
        if observer
            .active_materialized_sector_id
            .replace(materialized_sector_id)
            .is_some()
        {
            return Err(RusticolError::integrity(
                "transition diagnostic materialized-sector context is already active",
            ));
        }
        Ok(true)
    })?;
    let result = build();
    if activated {
        TRANSITION_OBSERVER.with(|slot| {
            if let Some(observer) = slot.borrow_mut().as_mut() {
                observer.active_materialized_sector_id = None;
            }
        });
    }
    result
}

pub(crate) fn retain_materialized_sector_rows<T>(
    rows: Vec<T>,
    materialized_sector_id: u32,
    sector_of: impl Fn(&T) -> Option<u32>,
) -> Vec<T> {
    rows.into_iter()
        .filter(|row| sector_of(row) == Some(materialized_sector_id))
        .collect()
}

pub(crate) fn transition_diagnostic_selection() -> Option<ConstructionDiagnosticSelectionV1> {
    TRANSITION_OBSERVER.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|observer| observer.selection.clone())
    })
}

pub(crate) fn observe_transition_live_current_digests(
    digests: BTreeSet<SemanticDigest>,
) -> RusticolResult<()> {
    TRANSITION_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let observer = slot.as_mut().ok_or_else(|| {
            RusticolError::integrity("recurrence transition diagnostic observation was not active")
        })?;
        if observer.live_current_digests.replace(digests).is_some() {
            return Err(RusticolError::integrity(
                "recurrence transition diagnostic live slice was recorded twice",
            ));
        }
        Ok(())
    })
}

pub(crate) fn take_transition_diagnostic_observation() -> RusticolResult<(
    Vec<ConstructionTransitionDiagnosticRowV1>,
    Option<BTreeSet<SemanticDigest>>,
)> {
    TRANSITION_OBSERVER.with(|slot| {
        let observer = slot.borrow_mut().take().ok_or_else(|| {
            RusticolError::integrity("recurrence transition diagnostic observation was not active")
        })?;
        Ok((observer.rows, observer.live_current_digests))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_index_for_public_helicity(
        _source_slot: u32,
        public_helicity: i32,
    ) -> RusticolResult<u32> {
        match public_helicity {
            -1 => Ok(0),
            1 => Ok(1),
            value => Err(RusticolError::integrity(format!(
                "test source has no state for public helicity {value}"
            ))),
        }
    }

    #[test]
    fn diagnostic_identity_replay_keeps_public_helicity_assignments() {
        let assignments = representative_source_states_for_public_helicities(
            &[0, 1],
            &[-1, 1],
            state_index_for_public_helicity,
        )
        .unwrap();
        assert_eq!(
            assignments,
            [
                SourceStateAssignment::new(0, 0),
                SourceStateAssignment::new(1, 1)
            ]
        );
    }

    #[test]
    fn diagnostic_nonidentity_replay_transports_opposite_public_helicities() {
        // The two representative slots have the same state contract.  The
        // composed replay map swaps them, so representative state ancestry
        // must be selected from the opposite public coordinate.
        let assignments = representative_source_states_for_public_helicities(
            &[1, 0],
            &[-1, 1],
            state_index_for_public_helicity,
        )
        .unwrap();
        assert_eq!(
            assignments,
            [
                SourceStateAssignment::new(0, 1),
                SourceStateAssignment::new(1, 0)
            ]
        );
    }

    #[test]
    fn diagnostic_lane_filter_keeps_selected_duplicates_without_digest_leakage() {
        let shared_digest = "same-semantic-current";
        let rows = vec![
            (shared_digest, 6, 0),
            (shared_digest, 8, 0),
            (shared_digest, 8, 1),
        ];
        let selected = retain_materialized_sector_rows(rows, 8, |row| Some(row.1));
        assert_eq!(selected, [(shared_digest, 8, 0), (shared_digest, 8, 1)]);
    }

    #[test]
    fn diagnostic_observer_active_query_tracks_scope() {
        assert!(!transition_diagnostic_observation_active());
        begin_transition_diagnostic_observation(None).unwrap();
        assert!(transition_diagnostic_observation_active());
        take_transition_diagnostic_observation().unwrap();
        assert!(!transition_diagnostic_observation_active());
    }

    #[test]
    fn diagnostic_lane_context_fails_closed_and_resets_after_build_error() {
        begin_transition_diagnostic_observation(None).unwrap();
        let nonsingleton = with_transition_diagnostic_materialized_sector(None, || Ok(()));
        assert!(nonsingleton.is_err());

        let failed: RusticolResult<()> =
            with_transition_diagnostic_materialized_sector(Some(8), || {
                Err(RusticolError::integrity(
                    "injected diagnostic build failure",
                ))
            });
        assert!(failed.is_err());
        with_transition_diagnostic_materialized_sector(Some(6), || Ok(())).unwrap();
        take_transition_diagnostic_observation().unwrap();
    }
}
