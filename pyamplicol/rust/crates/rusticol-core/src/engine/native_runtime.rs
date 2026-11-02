// SPDX-License-Identifier: 0BSD

use super::*;

use std::borrow::Cow;
use std::sync::OnceLock;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[path = "on_the_fly_cache.rs"]
mod cache;

enum DeferredProcessPhysicsV1 {
    #[expect(
        dead_code,
        reason = "retained for the lazy dense-metadata compatibility path"
    )]
    Dense {
        artifact: Box<VerifiedArtifact>,
        selection: crate::ArtifactSelection,
    },
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    OnTheFly {
        metadata: super::on_the_fly_public_metadata::OnTheFlyPublicMetadataV1,
        selectors: super::on_the_fly_selectors::OnTheFlyCompactSelectorAdapterV1,
        selection: Box<crate::ArtifactSelection>,
    },
}

/// Dense public process metadata is deliberately not part of ordinary
/// on-the-fly load/evaluation. Existing explicit metadata accessors retain
/// their API and initialize this cell on first use.
pub(super) struct LazyProcessPhysicsV1 {
    value: OnceLock<RusticolResult<ProcessPhysicsV1>>,
    deferred: Option<DeferredProcessPhysicsV1>,
}

impl LazyProcessPhysicsV1 {
    pub(super) fn loaded(value: ProcessPhysicsV1) -> Self {
        let cell = OnceLock::new();
        cell.set(Ok(value))
            .expect("fresh process-physics cell must be empty");
        Self {
            value: cell,
            deferred: None,
        }
    }

    #[cfg(test)]
    pub(super) const fn unavailable() -> Self {
        Self {
            value: OnceLock::new(),
            deferred: None,
        }
    }

    #[expect(
        dead_code,
        reason = "retained for the lazy dense-metadata compatibility path"
    )]
    fn deferred(artifact: VerifiedArtifact, selection: crate::ArtifactSelection) -> Self {
        Self {
            value: OnceLock::new(),
            deferred: Some(DeferredProcessPhysicsV1::Dense {
                artifact: Box::new(artifact),
                selection,
            }),
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub(super) fn deferred_on_the_fly(
        metadata: super::on_the_fly_public_metadata::OnTheFlyPublicMetadataV1,
        selectors: super::on_the_fly_selectors::OnTheFlyCompactSelectorAdapterV1,
        selection: crate::ArtifactSelection,
    ) -> Self {
        Self {
            value: OnceLock::new(),
            deferred: Some(DeferredProcessPhysicsV1::OnTheFly {
                metadata,
                selectors,
                selection: Box::new(selection),
            }),
        }
    }

    pub(super) fn get(&self) -> RusticolResult<&ProcessPhysicsV1> {
        let result = self.value.get_or_init(|| {
            let deferred = self.deferred.as_ref().ok_or_else(|| {
                RusticolError::internal("process-physics cell has no value or deferred source")
            })?;
            match deferred {
                DeferredProcessPhysicsV1::Dense {
                    artifact,
                    selection,
                } => {
                    let bytes = artifact.read_payload(&selection.process.physics_path)?;
                    let physics =
                        ProcessPhysicsV1::from_json(&bytes, &selection.process.physics_path)?;
                    validate_representative_physics(&physics, selection)?;
                    public_process_physics(physics, selection)
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                DeferredProcessPhysicsV1::OnTheFly {
                    metadata,
                    selectors,
                    selection,
                } => {
                    let physics = metadata.synthesize(selectors)?;
                    validate_representative_physics(&physics, selection)?;
                    public_process_physics(physics, selection)
                }
            }
        });
        result.as_ref().map_err(Clone::clone)
    }
}

/// Public metadata of `selection`: the representative physics remapped to the
/// requested order and, for a process-group member, relabelled as the member.
fn public_process_physics(
    physics: ProcessPhysicsV1,
    selection: &crate::ArtifactSelection,
) -> RusticolResult<ProcessPhysicsV1> {
    if !selection.remaps_public_process() {
        return Ok(physics);
    }
    let physics = apply_process_permutation_metadata(physics, selection)?;
    if selection.process_group.is_some() {
        apply_process_group_metadata(&physics, selection)
    } else {
        Ok(physics)
    }
}

fn process_group_routing(
    selection: &crate::ArtifactSelection,
    representative_key: &str,
) -> Option<Box<ProcessGroupRouting>> {
    selection.process_group.as_ref().map(|group| {
        Box::new(ProcessGroupRouting {
            representative_process_key: representative_key.to_string(),
            factor: NativeRationalFactor {
                numerator: group.factor_numerator,
                denominator: group.factor_denominator,
            },
            conjugated: group.conjugated,
        })
    })
}

fn validate_representative_physics(
    physics: &ProcessPhysicsV1,
    selection: &crate::ArtifactSelection,
) -> RusticolResult<()> {
    if physics.process_id != selection.process.id
        || physics.process != selection.process.expression
        || physics.color_accuracy.as_str() != selection.process.color_accuracy
        || physics
            .external_particles
            .iter()
            .map(|particle| particle.pdg)
            .ne(selection.process.external_pdgs.iter().copied())
    {
        return Err(RusticolError::integrity(format!(
            "runtime physics payload {:?} does not match process {:?}",
            selection.process.physics_path, selection.process.id
        )));
    }
    Ok(())
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Clone, Debug, Eq, PartialEq, bincode::Encode, bincode::Decode)]
struct OnTheFlySelectionIdentityV1 {
    helicity_ordinals: Option<Box<[usize]>>,
    color_ordinals: Option<Box<[usize]>>,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
impl OnTheFlySelectionIdentityV1 {
    fn from_slices(helicity_ordinals: Option<&[usize]>, color_ordinals: Option<&[usize]>) -> Self {
        Self {
            helicity_ordinals: helicity_ordinals.map(|values| values.to_vec().into_boxed_slice()),
            color_ordinals: color_ordinals.map(|values| values.to_vec().into_boxed_slice()),
        }
    }

    fn selected_axis_matches(
        ordinals: Option<&[usize]>,
        prepared_ids: &[String],
        selected_ids: Option<&BTreeSet<String>>,
    ) -> bool {
        match (ordinals, selected_ids) {
            (None, None) => true,
            (Some(_), Some(selected)) => {
                prepared_ids.len() == selected.len()
                    && prepared_ids.iter().all(|id| selected.contains(id))
            }
            _ => false,
        }
    }

    fn matches_fixed_selection(
        &self,
        prepared_helicity_ids: &[String],
        prepared_color_ids: &[String],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> bool {
        Self::selected_axis_matches(
            self.helicity_ordinals.as_deref(),
            prepared_helicity_ids,
            selected_helicities,
        ) && Self::selected_axis_matches(
            self.color_ordinals.as_deref(),
            prepared_color_ids,
            selected_colors,
        )
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Debug)]
struct OnTheFlyPreparedSelectionV1 {
    identity: OnTheFlySelectionIdentityV1,
    helicity_indices: Box<[usize]>,
    color_indices: Box<[usize]>,
    helicity_ids: Box<[String]>,
    color_ids: Box<[String]>,
    query_count: usize,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
struct OnTheFlyWarmUpPreparedSelectionV1 {
    helicity_count: usize,
    color_count: usize,
    query_count: usize,
    already_warm: bool,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
enum OnTheFlyReductionV1 {
    Lc,
    Contracted(Box<super::on_the_fly_load::LoadedOnTheFlyColorContractionV1>),
}

type OnTheFlySelectedOrdinalsV1 = (Option<Box<[usize]>>, Option<Box<[usize]>>);

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub(super) struct OnTheFlyExecutionRuntime {
    lane: super::on_the_fly_lane::OnTheFlyNativeRuntime,
    selectors: super::on_the_fly_selectors::OnTheFlyCompactSelectorAdapterV1,
    reduction: OnTheFlyReductionV1,
    prepared_selections: Vec<OnTheFlyPreparedSelectionV1>,
    last_prepared_selection: Option<usize>,
    pending_prepared_selection: Option<OnTheFlyPreparedSelectionV1>,
    point_major_scratch: Vec<f64>,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
impl OnTheFlyExecutionRuntime {
    pub(super) fn new(
        lane: super::on_the_fly_lane::OnTheFlyNativeRuntime,
        selectors: super::on_the_fly_selectors::OnTheFlyCompactSelectorAdapterV1,
        color_contraction: Option<super::on_the_fly_load::LoadedOnTheFlyColorContractionV1>,
    ) -> Self {
        Self {
            lane,
            selectors,
            reduction: color_contraction.map_or(OnTheFlyReductionV1::Lc, |plan| {
                OnTheFlyReductionV1::Contracted(Box::new(plan))
            }),
            prepared_selections: Vec::new(),
            last_prepared_selection: None,
            pending_prepared_selection: None,
            point_major_scratch: Vec::new(),
        }
    }

    pub(super) fn clear(&mut self) -> RusticolResult<()> {
        self.lane.clear()?;
        self.pending_prepared_selection = None;
        self.prepared_selections.clear();
        self.last_prepared_selection = None;
        self.point_major_scratch = Vec::new();
        Ok(())
    }

    fn uses_contracted_color(&self) -> bool {
        matches!(&self.reduction, OnTheFlyReductionV1::Contracted(_))
    }

    fn public_selector_counts(&self) -> RusticolResult<(usize, usize)> {
        let selection = self.selectors.selection(self.lane.seed(), None, None)?;
        Ok((
            selection.helicity_count(),
            if self.uses_contracted_color() {
                1
            } else {
                selection.color_count()
            },
        ))
    }

    #[cfg(test)]
    pub(super) fn retained_family_count(&self) -> usize {
        self.lane.retained_family_count()
    }

    #[cfg(test)]
    pub(super) fn retained_selection_count(&self) -> usize {
        self.prepared_selections.len()
    }

    #[cfg(test)]
    pub(super) fn pending_family_count(&self) -> usize {
        self.lane.pending_family_count()
    }

    #[cfg(test)]
    pub(super) fn fail_contracted_execution_at_for_test(&mut self, attempt: Option<usize>) {
        self.lane.fail_contracted_execution_at_for_test(attempt);
    }

    #[cfg(test)]
    pub(super) const fn contracted_max_live_query_outcomes_for_test(&self) -> usize {
        self.lane.contracted_max_live_query_outcomes_for_test()
    }

    #[cfg(test)]
    pub(super) fn semantic_executor_binding_count(&self) -> RusticolResult<u32> {
        self.lane.semantic_executor_binding_count()
    }

    #[cfg(test)]
    pub(super) fn active_family_prepared_census(
        &self,
    ) -> Option<crate::recurrence::on_the_fly::OnTheFlyQueryFamilyCensusV1> {
        self.lane.prepared_census()
    }

    #[cfg(test)]
    pub(super) fn point_major_scratch_state(&self) -> (usize, usize) {
        (
            self.point_major_scratch.len(),
            self.point_major_scratch.capacity(),
        )
    }

    /// Private read-only production introspection of retained compact state.
    pub(super) fn state_census(&self, process_id: &str) -> RusticolResult<Value> {
        let retained = self.lane.retained_state_census();
        let active_family_union_census = self.lane.prepared_census().map(|census| {
            serde_json::json!({
                "basis": "shared-query-family-union-v1",
                "scope": "active-family-union",
                "query_count": census.query_count,
                "union_unique_current_count": census.union_unique_current_count,
                "union_unique_current_component_count": (
                    census.union_unique_current_component_count
                ),
                "union_source_rows": census.union_source_rows,
                "union_contribution_rows": census.union_contribution_rows,
                "union_finalization_rows": census.union_finalization_rows,
                "union_closure_rows": census.union_closure_rows,
                "union_amplitude_destination_count": (
                    census.union_amplitude_destination_count
                ),
                "union_source_executor_call_groups": (
                    census.union_source_executor_call_groups
                ),
                "union_contribution_executor_call_groups": (
                    census.union_contribution_executor_call_groups
                ),
                "union_finalization_executor_call_groups": (
                    census.union_finalization_executor_call_groups
                ),
                "union_closure_executor_call_groups": (
                    census.union_closure_executor_call_groups
                ),
            })
        });
        Ok(serde_json::json!({
            "kind": "rusticol-on-the-fly-runtime-state-census-v1",
            "process_id": process_id,
            "family_cache_policy": "last-family-only",
            "family_cache_limit": 1,
            "process_preparation_count": self.lane.process_preparation_count(),
            "retained_family_count": retained.family_count,
            "pending_family_count": self.lane.pending_family_count(),
            "retained_selection_count": self.prepared_selections.len(),
            "retained_request_count": retained.request_count,
            "retained_amplitude_destination_count": retained.amplitude_destination_count,
            "retained_executor_handle_count": retained.executor_handle_count,
            "retained_query_local_trace_count": retained.query_local_trace_count,
            "retained_embedded_lookup_key_count": retained.embedded_lookup_key_count,
            "semantic_executor_binding_count": self.lane.semantic_executor_binding_count()?,
            "active_family_union_census": active_family_union_census,
        }))
    }

    fn fill_point_major(&mut self, batch: F64MomentumBatchView<'_>) -> RusticolResult<usize> {
        let required = batch
            .point_count()
            .checked_mul(batch.external_count())
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                RusticolError::invalid_argument("on-the-fly momentum shape exceeds usize")
            })?;
        if self.point_major_scratch.len() < required {
            self.point_major_scratch
                .try_reserve_exact(required - self.point_major_scratch.len())
                .map_err(|error| {
                    RusticolError::invalid_argument(format!(
                        "on-the-fly point-major workspace allocation failed: {error}"
                    ))
                })?;
            self.point_major_scratch.resize(required, 0.0);
        }
        for point_index in 0..batch.point_count() {
            let point = batch.point(point_index);
            for external_index in 0..batch.external_count() {
                let momentum = point.momentum(external_index).ok_or_else(|| {
                    RusticolError::integrity("on-the-fly momentum view omits an external leg")
                })?;
                let start = (point_index * batch.external_count() + external_index) * 4;
                self.point_major_scratch[start..start + 4].copy_from_slice(&momentum);
            }
        }
        Ok(required)
    }

    fn prepare_selection(
        &mut self,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        point_count: usize,
    ) -> RusticolResult<(usize, usize)> {
        if matches!(&self.reduction, OnTheFlyReductionV1::Contracted(_))
            && selected_colors.is_some()
        {
            return Err(RusticolError::selector(
                "on-the-fly contracted color does not expose a color selector",
            ));
        }
        if let Some(shape) =
            self.reuse_fixed_selection(selected_helicities, selected_colors, point_count)?
        {
            return Ok(shape);
        }
        let (helicity_ordinals, color_ordinals) = match &self.reduction {
            OnTheFlyReductionV1::Lc => self
                .selectors
                .selected_ordinals(selected_helicities, selected_colors)?,
            OnTheFlyReductionV1::Contracted(_) => {
                let (helicity_ordinals, _) = self
                    .selectors
                    .selected_ordinals(selected_helicities, None)?;
                (helicity_ordinals, None)
            }
        };
        self.prepare_selection_identity(
            OnTheFlySelectionIdentityV1 {
                helicity_ordinals,
                color_ordinals,
            },
            point_count,
        )
    }

    /// Reuse the last committed fixed public selection before reparsing its
    /// string IDs.  The prepared public IDs already authenticate the exact
    /// selector set, while the identity's `Option` markers distinguish an
    /// omitted (sum-all) axis from an explicit complete-axis selection.
    /// Matching therefore borrows both sets and allocates nothing on the
    /// warmed ordinary-evaluation path.
    fn reuse_fixed_selection(
        &mut self,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        point_count: usize,
    ) -> RusticolResult<Option<(usize, usize)>> {
        let pending_matches = self
            .pending_prepared_selection
            .as_ref()
            .is_some_and(|prepared| {
                prepared.identity.matches_fixed_selection(
                    &prepared.helicity_ids,
                    &prepared.color_ids,
                    selected_helicities,
                    selected_colors,
                )
            });
        let committed_index = self
            .pending_prepared_selection
            .is_none()
            .then_some(self.last_prepared_selection)
            .flatten()
            .filter(|index| {
                self.prepared_selections
                    .get(*index)
                    .is_some_and(|prepared| {
                        prepared.identity.matches_fixed_selection(
                            &prepared.helicity_ids,
                            &prepared.color_ids,
                            selected_helicities,
                            selected_colors,
                        )
                    })
            });
        if !pending_matches && committed_index.is_none() {
            return Ok(None);
        }
        let family_point_capacity = self.family_point_capacity(point_count)?;
        self.reuse_current_queries(family_point_capacity)?;
        let prepared = if pending_matches {
            self.pending_prepared_selection
                .as_ref()
                .expect("matching pending on-the-fly selection disappeared")
        } else {
            &self.prepared_selections
                [committed_index.expect("matching committed selection disappeared")]
        };
        Ok(Some((
            prepared.helicity_indices.len(),
            prepared.color_indices.len(),
        )))
    }

    fn prepare_selection_from_ordinals(
        &mut self,
        helicity_ordinals: Option<&[usize]>,
        color_ordinals: Option<&[usize]>,
        point_count: usize,
    ) -> RusticolResult<(usize, usize)> {
        if matches!(&self.reduction, OnTheFlyReductionV1::Contracted(_)) && color_ordinals.is_some()
        {
            return Err(RusticolError::selector(
                "on-the-fly contracted color does not expose per-point color ordinals",
            ));
        }
        self.prepare_selection_identity(
            OnTheFlySelectionIdentityV1::from_slices(helicity_ordinals, color_ordinals),
            point_count,
        )
    }

    fn prepare_selection_for_warm_up(
        &mut self,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        query_construction_threads: usize,
        progress: &mut super::on_the_fly_warm_up::OnTheFlyWarmUpProgress<'_>,
    ) -> RusticolResult<OnTheFlyWarmUpPreparedSelectionV1> {
        if self.uses_contracted_color() && selected_colors.is_some() {
            return Err(RusticolError::selector(
                "on-the-fly contracted color does not expose a color selector",
            ));
        }
        let (helicity_ordinals, color_ordinals) = match &self.reduction {
            OnTheFlyReductionV1::Lc => self
                .selectors
                .selected_ordinals(selected_helicities, selected_colors)?,
            OnTheFlyReductionV1::Contracted(_) => {
                let (helicity_ordinals, _) = self
                    .selectors
                    .selected_ordinals(selected_helicities, None)?;
                (helicity_ordinals, None)
            }
        };
        self.prepare_selection_identity_impl(
            OnTheFlySelectionIdentityV1 {
                helicity_ordinals,
                color_ordinals,
            },
            1,
            Some(query_construction_threads),
            Some(progress),
        )
    }

    fn prepare_selection_identity(
        &mut self,
        identity: OnTheFlySelectionIdentityV1,
        point_count: usize,
    ) -> RusticolResult<(usize, usize)> {
        let prepared = self.prepare_selection_identity_impl(identity, point_count, None, None)?;
        Ok((prepared.helicity_count, prepared.color_count))
    }

    fn prepare_selection_identity_impl(
        &mut self,
        identity: OnTheFlySelectionIdentityV1,
        point_count: usize,
        query_construction_threads: Option<usize>,
        mut progress: Option<&mut super::on_the_fly_warm_up::OnTheFlyWarmUpProgress<'_>>,
    ) -> RusticolResult<OnTheFlyWarmUpPreparedSelectionV1> {
        let family_point_capacity = self.family_point_capacity(point_count)?;
        if self
            .pending_prepared_selection
            .as_ref()
            .is_some_and(|prepared| prepared.identity == identity)
        {
            self.reuse_current_queries(family_point_capacity)?;
            let prepared = self
                .pending_prepared_selection
                .as_ref()
                .expect("matching pending on-the-fly selection disappeared");
            if let Some(progress) = progress.as_deref_mut() {
                self.lane
                    .report_reused_family_for_warm_up(prepared.query_count, progress)?;
            }
            return Ok(OnTheFlyWarmUpPreparedSelectionV1 {
                helicity_count: prepared.helicity_indices.len(),
                color_count: prepared.color_indices.len(),
                query_count: prepared.query_count,
                already_warm: false,
            });
        }
        self.discard_pending_selection()?;
        if let Some(index) = self.last_prepared_selection.filter(|index| {
            self.prepared_selections
                .get(*index)
                .is_some_and(|prepared| prepared.identity == identity)
        }) {
            self.reuse_current_queries(family_point_capacity)?;
            let prepared = &self.prepared_selections[index];
            if let Some(progress) = progress.as_deref_mut() {
                self.lane
                    .report_reused_family_for_warm_up(prepared.query_count, progress)?;
            }
            return Ok(OnTheFlyWarmUpPreparedSelectionV1 {
                helicity_count: prepared.helicity_indices.len(),
                color_count: prepared.color_indices.len(),
                query_count: prepared.query_count,
                already_warm: true,
            });
        }

        if self.prepared_selections.capacity() == 0 {
            self.prepared_selections
                .try_reserve_exact(1)
                .map_err(|error| {
                    RusticolError::invalid_argument(format!(
                        "on-the-fly prepared-selection allocation failed: {error}"
                    ))
                })?;
        }
        let selection = self.selectors.selection_from_ordinals(
            self.lane.seed(),
            identity.helicity_ordinals.as_deref(),
            identity.color_ordinals.as_deref(),
        )?;
        let helicity_count = selection.helicity_count();
        let helicity_indices = (0..helicity_count)
            .map(|position| selection.helicity_ordinal_at(position))
            .collect::<RusticolResult<Vec<_>>>()?;
        let helicity_ids = (0..helicity_count)
            .map(|position| selection.helicity_id_at(position))
            .collect::<RusticolResult<Vec<_>>>()?;
        let (color_count, color_indices, color_ids, query_count) = match &self.reduction {
            OnTheFlyReductionV1::Lc => {
                let color_count = selection.color_count();
                let color_indices = (0..color_count)
                    .map(|position| selection.color_ordinal_at(position))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let color_ids = (0..color_count)
                    .map(|position| selection.color_id_at(position))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let requests = selection.iter().collect::<RusticolResult<Vec<_>>>()?;
                if let Some(progress) = progress {
                    self.lane.prepare_lc_queries_for_warm_up(
                        &requests,
                        family_point_capacity,
                        query_construction_threads.expect("warm-up worker count is absent"),
                        progress,
                    )?;
                } else {
                    self.lane
                        .prepare_lc_queries(&requests, family_point_capacity)?;
                }
                let query_count = requests.len();
                (color_count, color_indices, color_ids, query_count)
            }
            OnTheFlyReductionV1::Contracted(plan) => {
                if identity.color_ordinals.is_some() {
                    return Err(RusticolError::selector(
                        "on-the-fly contracted color selection is unavailable",
                    ));
                }
                let structural_color_count = plan.destination_by_owner_ordinal.len();
                if structural_color_count != self.selectors.color_count() {
                    return Err(RusticolError::integrity(
                        "on-the-fly contracted owner basis disagrees with compact structural selectors",
                    ));
                }
                if let Some(progress) = progress {
                    self.lane.prepare_contracted_queries_for_warm_up(
                        &self.selectors,
                        &helicity_indices,
                        structural_color_count,
                        &plan.destination_by_owner_ordinal,
                        family_point_capacity,
                        query_construction_threads.expect("warm-up worker count is absent"),
                        progress,
                    )?;
                } else {
                    self.lane.prepare_contracted_queries(
                        &self.selectors,
                        &helicity_indices,
                        structural_color_count,
                        &plan.destination_by_owner_ordinal,
                        family_point_capacity,
                    )?;
                }
                let query_count = helicity_indices
                    .len()
                    .checked_mul(structural_color_count)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "on-the-fly contracted query count exceeds usize",
                        )
                    })?;
                (
                    1,
                    vec![0],
                    vec!["color:contracted".to_string()],
                    query_count,
                )
            }
        };
        self.pending_prepared_selection = Some(OnTheFlyPreparedSelectionV1 {
            identity,
            helicity_indices: helicity_indices.into_boxed_slice(),
            color_indices: color_indices.into_boxed_slice(),
            helicity_ids: helicity_ids.into_boxed_slice(),
            color_ids: color_ids.into_boxed_slice(),
            query_count,
        });
        Ok(OnTheFlyWarmUpPreparedSelectionV1 {
            helicity_count,
            color_count,
            query_count,
            already_warm: false,
        })
    }

    fn family_point_capacity(&self, point_count: usize) -> RusticolResult<u32> {
        let point_capacity = u32::try_from(point_count)
            .map_err(|_| RusticolError::invalid_argument("on-the-fly point count exceeds u32"))?;
        match &self.reduction {
            OnTheFlyReductionV1::Lc => Ok(point_capacity),
            OnTheFlyReductionV1::Contracted(plan) => {
                u32::try_from(point_count.min(plan.point_tile_size)).map_err(|_| {
                    RusticolError::invalid_argument("on-the-fly contracted point tile exceeds u32")
                })
            }
        }
    }

    fn current_prepared_selection(&self) -> RusticolResult<&OnTheFlyPreparedSelectionV1> {
        self.pending_prepared_selection
            .as_ref()
            .or_else(|| {
                self.last_prepared_selection
                    .and_then(|index| self.prepared_selections.get(index))
            })
            .ok_or_else(|| RusticolError::internal("on-the-fly selection was not retained"))
    }

    fn discard_pending_selection(&mut self) -> RusticolResult<()> {
        let aborting_prepared_contracted_selection = self.pending_prepared_selection.is_some()
            && matches!(&self.reduction, OnTheFlyReductionV1::Contracted(_));
        if aborting_prepared_contracted_selection {
            // A successful earlier tile commits the executor family before
            // the wrapper commits its public selection.  If a later tile
            // fails, retaining either the old wrapper identity or the new
            // executor family would make the two caches disagree. Clear the
            // wrapper side even if native row-table invalidation itself fails.
            let rollback = self.lane.abort_contracted_selection();
            self.pending_prepared_selection = None;
            self.prepared_selections.clear();
            self.last_prepared_selection = None;
            return rollback;
        }
        match &self.reduction {
            OnTheFlyReductionV1::Lc => self.lane.discard_pending_lc_queries()?,
            OnTheFlyReductionV1::Contracted(_) => self.lane.discard_pending_contracted_queries()?,
        }
        self.pending_prepared_selection = None;
        Ok(())
    }

    fn reuse_current_queries(&mut self, point_capacity: u32) -> RusticolResult<bool> {
        match &self.reduction {
            OnTheFlyReductionV1::Lc => self.lane.reuse_current_lc_queries(point_capacity),
            OnTheFlyReductionV1::Contracted(_) => {
                self.lane.reuse_current_contracted_queries(point_capacity)
            }
        }
    }

    fn commit_pending_selection(&mut self) {
        let Some(selection) = self.pending_prepared_selection.take() else {
            return;
        };
        self.prepared_selections.clear();
        self.prepared_selections.push(selection);
        self.last_prepared_selection = Some(0);
    }

    fn selected_axis_ids(&self) -> RusticolResult<(Vec<String>, Vec<String>)> {
        let prepared = self.current_prepared_selection()?;
        Ok((prepared.helicity_ids.to_vec(), prepared.color_ids.to_vec()))
    }

    #[allow(clippy::too_many_arguments)]
    fn warm_up_f64<'observer>(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        n_cores: Option<usize>,
        observer: Option<
            &'observer mut super::on_the_fly_warm_up::NativeOnTheFlyWarmUpObserver<'observer>,
        >,
    ) -> RusticolResult<super::on_the_fly_warm_up::NativeOnTheFlyWarmUpResult> {
        if batch.point_count() != 1 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly warm-up requires exactly one binary64 phase-space point",
            ));
        }
        let query_construction_threads = match n_cores {
            Some(0) => {
                return Err(RusticolError::invalid_argument(
                    "on-the-fly warm-up n_cores must be positive",
                ));
            }
            Some(requested) => {
                super::on_the_fly_load::effective_query_construction_threads(requested)
            }
            None => self.lane.effective_query_construction_threads(),
        };
        let mut progress = super::on_the_fly_warm_up::OnTheFlyWarmUpProgress::new(
            observer,
            query_construction_threads,
        )?;
        let result = (|| {
            self.lane.prepare_process_for_warm_up(&mut progress)?;
            let prepared = self.prepare_selection_for_warm_up(
                selected_helicities,
                selected_colors,
                query_construction_threads,
                &mut progress,
            )?;
            let input_len = self.fill_point_major(batch)?;
            progress.emit(
                super::on_the_fly_warm_up::NativeOnTheFlyWarmUpEventKind::Start,
                super::on_the_fly_warm_up::NativeOnTheFlyWarmUpStage::FirstEvaluation,
                0,
                1,
                None,
            )?;
            let mut output = [0.0_f64];
            match &self.reduction {
                OnTheFlyReductionV1::Lc => self.lane.run_f64_into_unprofiled(
                    &self.point_major_scratch[..input_len],
                    1,
                    &common.model_parameter_values_f64,
                    common.normalization_factor,
                    &mut output,
                )?,
                OnTheFlyReductionV1::Contracted(plan) => {
                    self.lane.run_contracted_f64_into_unprofiled(
                        &self.point_major_scratch[..input_len],
                        1,
                        &common.model_parameter_values_f64,
                        common.normalization_factor,
                        &plan.plan,
                        plan.point_tile_size,
                        &mut output,
                    )?;
                }
            }
            Ok(prepared)
        })();
        let prepared = match result {
            Ok(prepared) => {
                // The lane commits its executable family only after the first
                // successful evaluation. Commit the matching public selector
                // before publishing the terminal event so both cache layers
                // remain coherent even if an embedding rejects that event.
                self.commit_pending_selection();
                prepared
            }
            Err(error) => {
                self.discard_pending_selection()?;
                return Err(error);
            }
        };
        progress.emit_terminal_notification(
            super::on_the_fly_warm_up::NativeOnTheFlyWarmUpStage::FirstEvaluation,
            1,
            1,
            None,
        );
        let query_count = u64::try_from(prepared.query_count)
            .map_err(|_| RusticolError::invalid_argument("warm-up query count exceeds u64"))?;
        Ok(super::on_the_fly_warm_up::NativeOnTheFlyWarmUpResult {
            schema_version: 1,
            elapsed_seconds: progress.elapsed_seconds(),
            query_count,
            warmed_query_count: if prepared.already_warm {
                0
            } else {
                query_count
            },
            current_rss_bytes: progress.current_rss_bytes(),
            peak_rss_bytes: progress.peak_rss_bytes(),
            already_warm: prepared.already_warm,
            first_evaluation_completed: true,
        })
    }

    pub(super) fn run_total_into_unprofiled(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let result = (|| {
            let point_count = batch.point_count();
            self.prepare_selection(selected_helicities, selected_colors, point_count)?;
            let input_len = self.fill_point_major(batch)?;
            let native_point_count = u32::try_from(point_count).map_err(|_| {
                RusticolError::invalid_argument("on-the-fly point count exceeds u32")
            })?;
            match &self.reduction {
                OnTheFlyReductionV1::Lc => self.lane.run_f64_into_unprofiled(
                    &self.point_major_scratch[..input_len],
                    native_point_count,
                    &common.model_parameter_values_f64,
                    common.normalization_factor,
                    output,
                ),
                OnTheFlyReductionV1::Contracted(plan) => {
                    self.lane.run_contracted_f64_into_unprofiled(
                        &self.point_major_scratch[..input_len],
                        native_point_count,
                        &common.model_parameter_values_f64,
                        common.normalization_factor,
                        &plan.plan,
                        plan.point_tile_size,
                        output,
                    )
                }
            }
        })();
        match result {
            Ok(()) => {
                self.commit_pending_selection();
                Ok(())
            }
            Err(error) => {
                self.discard_pending_selection()?;
                Err(error)
            }
        }
    }

    pub(super) fn run_total_into(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<RuntimeProfile> {
        let result = (|| {
            let point_count = batch.point_count();
            self.prepare_selection(selected_helicities, selected_colors, point_count)?;
            let input_len = self.fill_point_major(batch)?;
            let native_point_count = u32::try_from(point_count).map_err(|_| {
                RusticolError::invalid_argument("on-the-fly point count exceeds u32")
            })?;
            let (_, profile) = match &self.reduction {
                OnTheFlyReductionV1::Lc => self.lane.run_f64_into(
                    &self.point_major_scratch[..input_len],
                    native_point_count,
                    &common.model_parameter_values_f64,
                    common.normalization_factor,
                    output,
                )?,
                OnTheFlyReductionV1::Contracted(plan) => self.lane.run_contracted_f64_into(
                    &self.point_major_scratch[..input_len],
                    native_point_count,
                    &common.model_parameter_values_f64,
                    common.normalization_factor,
                    &plan.plan,
                    plan.point_tile_size,
                    output,
                )?,
            };
            Ok(profile)
        })();
        match result {
            Ok(profile) => {
                self.commit_pending_selection();
                Ok(profile)
            }
            Err(error) => {
                self.discard_pending_selection()?;
                Err(error)
            }
        }
    }

    fn run_total_into_by_ordinals(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        helicity_ordinals: Option<&[usize]>,
        color_ordinals: Option<&[usize]>,
        output: &mut [f64],
    ) -> RusticolResult<RuntimeProfile> {
        let result = (|| {
            let point_count = batch.point_count();
            self.prepare_selection_from_ordinals(helicity_ordinals, color_ordinals, point_count)?;
            let input_len = self.fill_point_major(batch)?;
            let native_point_count = u32::try_from(point_count).map_err(|_| {
                RusticolError::invalid_argument("on-the-fly point count exceeds u32")
            })?;
            let (_, profile) = match &self.reduction {
                OnTheFlyReductionV1::Lc => self.lane.run_f64_into(
                    &self.point_major_scratch[..input_len],
                    native_point_count,
                    &common.model_parameter_values_f64,
                    common.normalization_factor,
                    output,
                )?,
                OnTheFlyReductionV1::Contracted(plan) => self.lane.run_contracted_f64_into(
                    &self.point_major_scratch[..input_len],
                    native_point_count,
                    &common.model_parameter_values_f64,
                    common.normalization_factor,
                    &plan.plan,
                    plan.point_tile_size,
                    output,
                )?,
            };
            Ok(profile)
        })();
        match result {
            Ok(profile) => {
                self.commit_pending_selection();
                Ok(profile)
            }
            Err(error) => {
                self.discard_pending_selection()?;
                Err(error)
            }
        }
    }

    fn run_total_into_by_ordinals_unprofiled(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        helicity_ordinals: Option<&[usize]>,
        color_ordinals: Option<&[usize]>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let result = (|| {
            let point_count = batch.point_count();
            self.prepare_selection_from_ordinals(helicity_ordinals, color_ordinals, point_count)?;
            let input_len = self.fill_point_major(batch)?;
            let native_point_count = u32::try_from(point_count).map_err(|_| {
                RusticolError::invalid_argument("on-the-fly point count exceeds u32")
            })?;
            match &self.reduction {
                OnTheFlyReductionV1::Lc => self.lane.run_f64_into_unprofiled(
                    &self.point_major_scratch[..input_len],
                    native_point_count,
                    &common.model_parameter_values_f64,
                    common.normalization_factor,
                    output,
                ),
                OnTheFlyReductionV1::Contracted(plan) => {
                    self.lane.run_contracted_f64_into_unprofiled(
                        &self.point_major_scratch[..input_len],
                        native_point_count,
                        &common.model_parameter_values_f64,
                        common.normalization_factor,
                        &plan.plan,
                        plan.point_tile_size,
                        output,
                    )
                }
            }
        })();
        match result {
            Ok(()) => {
                self.commit_pending_selection();
                Ok(())
            }
            Err(error) => {
                self.discard_pending_selection()?;
                Err(error)
            }
        }
    }

    fn run_resolved(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        let result = (|| {
            let point_count = batch.point_count();
            let (helicity_count, color_count) =
                self.prepare_selection(selected_helicities, selected_colors, point_count)?;
            let prepared = self.current_prepared_selection()?;
            let helicity_indices = prepared.helicity_indices.to_vec();
            let color_indices = prepared.color_indices.to_vec();
            let value_count = point_count
                .checked_mul(helicity_count)
                .and_then(|value| value.checked_mul(color_count))
                .ok_or_else(|| {
                    RusticolError::invalid_argument("on-the-fly resolved shape exceeds usize")
                })?;
            let mut values = vec![0.0; value_count];
            let input_len = self.fill_point_major(batch)?;
            let native_point_count = u32::try_from(point_count).map_err(|_| {
                RusticolError::invalid_argument("on-the-fly point count exceeds u32")
            })?;
            let (_, profile) = match &self.reduction {
                OnTheFlyReductionV1::Lc => self.lane.run_resolved_f64_into(
                    &self.point_major_scratch[..input_len],
                    native_point_count,
                    &common.model_parameter_values_f64,
                    common.normalization_factor,
                    helicity_count,
                    color_count,
                    &mut values,
                )?,
                OnTheFlyReductionV1::Contracted(plan) => {
                    if color_count != 1 {
                        return Err(RusticolError::integrity(
                            "on-the-fly contracted resolved color axis is not singleton",
                        ));
                    }
                    self.lane.run_contracted_resolved_f64_into(
                        &self.point_major_scratch[..input_len],
                        native_point_count,
                        &common.model_parameter_values_f64,
                        common.normalization_factor,
                        &plan.plan,
                        plan.point_tile_size,
                        helicity_count,
                        &mut values,
                    )?
                }
            };
            Ok((
                ResolvedValues {
                    values,
                    point_count,
                    helicity_indices,
                    color_indices,
                },
                profile,
            ))
        })();
        match result {
            Ok(result) => {
                self.commit_pending_selection();
                Ok(result)
            }
            Err(error) => {
                self.discard_pending_selection()?;
                Err(error)
            }
        }
    }
}

struct PointSelectorProfileCounts {
    gather_point_count: usize,
    input_bytes_per_point: usize,
    scatter_value_count: usize,
}

type ProfiledPreparedF64Batch = (Vec<Vec<[f64; 4]>>, Duration, Duration);

fn attach_point_selector_profile(
    profile: &mut NativeRuntimeProfile,
    plan: &PointSelectorPlanProfile,
    planner: Duration,
    gather: Duration,
    scatter: Duration,
    counts: PointSelectorProfileCounts,
) {
    profile.selector_planner_s = profile_duration_seconds(planner);
    profile.selector_gather_s = profile_duration_seconds(gather);
    profile.selector_scatter_s = profile_duration_seconds(scatter);
    profile.selector_plan_kind = plan.kind.to_string();
    profile.selector_group_sizes.clone_from(&plan.group_sizes);
    profile.selector_reordered_point_count = plan.reordered_point_count;
    profile.selector_simd_lane_width = plan.simd_lane_width;
    profile.selector_simd_occupancy = plan.simd_occupancy;
    profile.selector_gather_point_count = counts.gather_point_count as u64;
    profile.selector_gather_bytes =
        (counts.gather_point_count * counts.input_bytes_per_point) as u64;
    profile.selector_scatter_value_count = counts.scatter_value_count as u64;
}

fn execution_uses_simd_jit(runtime: &ExecutionRuntime) -> bool {
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    if runtime.compiled_direct_runtime.is_some() {
        return true;
    }
    runtime
        .stages
        .as_ref()
        .is_some_and(|stages| stages.iter().any(|stage| stage.evaluator.uses_simd_jit()))
        || runtime
            .amplitude_stage
            .as_ref()
            .and_then(|stage| stage.evaluator.as_ref())
            .is_some_and(EvaluatorGroup::uses_simd_jit)
        || runtime
            .helicity_sum_runtime
            .as_deref()
            .is_some_and(execution_uses_simd_jit)
        || runtime
            .helicity_selector_runtimes
            .iter()
            .any(|lane| execution_uses_simd_jit(lane))
        || runtime
            .color_selector_runtimes
            .values()
            .any(|lane| execution_uses_simd_jit(lane))
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Clone, Copy, Debug, Default)]
struct CompiledDirectProfileSnapshot {
    engine_count: usize,
    minimum_effective_tile_capacity: usize,
    maximum_physical_scalar_values_per_point: usize,
    maximum_hot_scalar_values_per_point: usize,
    maximum_source_scalar_values_per_point: usize,
    maximum_reduction_scalar_values_per_point: usize,
    source_fill_bytes: u64,
    momentum_fill_bytes: u64,
    parameter_fill_bytes: u64,
    scalar_broadcast_fill_bytes: u64,
    amplitude_clear_bytes: u64,
    backend_call_count: u64,
    boundary_input_bytes: u64,
    boundary_current_output_bytes: u64,
    boundary_amplitude_output_bytes: u64,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
const fn minimum_nonzero(left: usize, right: usize) -> usize {
    match (left, right) {
        (0, value) | (value, 0) => value,
        (left, right) => {
            if left < right {
                left
            } else {
                right
            }
        }
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Clone, Copy, Debug, Default)]
struct CompiledDirectConfigurationSnapshot {
    engine_count: usize,
    minimum_effective_tile_capacity: usize,
    maximum_physical_scalar_values_per_point: usize,
    maximum_hot_scalar_values_per_point: usize,
    maximum_source_scalar_values_per_point: usize,
    maximum_reduction_scalar_values_per_point: usize,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn compiled_direct_configuration_snapshot(
    runtime: &ExecutionRuntime,
) -> CompiledDirectConfigurationSnapshot {
    let mut snapshot = CompiledDirectConfigurationSnapshot::default();
    if let Some(direct) = runtime.compiled_direct_runtime.as_ref() {
        let sizing = direct.tile_sizing();
        snapshot.engine_count = 1;
        snapshot.minimum_effective_tile_capacity = sizing.effective_tile_capacity;
        snapshot.maximum_physical_scalar_values_per_point = sizing.physical_scalar_values_per_point;
        snapshot.maximum_hot_scalar_values_per_point = sizing.hot_scalar_values_per_point;
        snapshot.maximum_source_scalar_values_per_point = sizing.source_scalar_values_per_point;
        snapshot.maximum_reduction_scalar_values_per_point =
            sizing.reduction_scalar_values_per_point;
    }
    let children = runtime
        .helicity_sum_runtime
        .iter()
        .map(Box::as_ref)
        .chain(runtime.helicity_selector_runtimes.iter().map(Box::as_ref))
        .chain(runtime.color_selector_runtimes.values().map(Box::as_ref));
    for child in children {
        let child = compiled_direct_configuration_snapshot(child);
        snapshot.engine_count += child.engine_count;
        snapshot.minimum_effective_tile_capacity = minimum_nonzero(
            snapshot.minimum_effective_tile_capacity,
            child.minimum_effective_tile_capacity,
        );
        snapshot.maximum_physical_scalar_values_per_point = snapshot
            .maximum_physical_scalar_values_per_point
            .max(child.maximum_physical_scalar_values_per_point);
        snapshot.maximum_hot_scalar_values_per_point = snapshot
            .maximum_hot_scalar_values_per_point
            .max(child.maximum_hot_scalar_values_per_point);
        snapshot.maximum_source_scalar_values_per_point = snapshot
            .maximum_source_scalar_values_per_point
            .max(child.maximum_source_scalar_values_per_point);
        snapshot.maximum_reduction_scalar_values_per_point = snapshot
            .maximum_reduction_scalar_values_per_point
            .max(child.maximum_reduction_scalar_values_per_point);
    }
    snapshot
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn compiled_direct_profile_snapshot(
    runtime: &ExecutionRuntime,
) -> RusticolResult<CompiledDirectProfileSnapshot> {
    let mut snapshot = CompiledDirectProfileSnapshot::default();
    if let Some(direct) = runtime.compiled_direct_runtime.as_ref() {
        let traffic = direct.traffic();
        traffic.leaf.validate_direct()?;
        if traffic.boundary_input_bytes
            | traffic.boundary_current_output_bytes
            | traffic.boundary_amplitude_output_bytes
            != 0
        {
            return Err(RusticolError::integrity(
                "production compiled Direct-Arena profile observed developer boundary traffic",
            ));
        }
        let sizing = direct.tile_sizing();
        snapshot.engine_count += 1;
        snapshot.minimum_effective_tile_capacity = sizing.effective_tile_capacity;
        snapshot.maximum_physical_scalar_values_per_point = sizing.physical_scalar_values_per_point;
        snapshot.maximum_hot_scalar_values_per_point = sizing.hot_scalar_values_per_point;
        snapshot.maximum_source_scalar_values_per_point = sizing.source_scalar_values_per_point;
        snapshot.maximum_reduction_scalar_values_per_point =
            sizing.reduction_scalar_values_per_point;
        snapshot.source_fill_bytes = traffic.source_fill_bytes;
        snapshot.momentum_fill_bytes = traffic.momentum_fill_bytes;
        snapshot.parameter_fill_bytes = traffic.parameter_fill_bytes;
        snapshot.scalar_broadcast_fill_bytes = traffic.scalar_broadcast_fill_bytes;
        snapshot.amplitude_clear_bytes = traffic.amplitude_clear_bytes;
        snapshot.backend_call_count = traffic.leaf.calls;
        snapshot.boundary_input_bytes = traffic.boundary_input_bytes;
        snapshot.boundary_current_output_bytes = traffic.boundary_current_output_bytes;
        snapshot.boundary_amplitude_output_bytes = traffic.boundary_amplitude_output_bytes;
    }
    let children = runtime
        .helicity_sum_runtime
        .iter()
        .map(Box::as_ref)
        .chain(runtime.helicity_selector_runtimes.iter().map(Box::as_ref))
        .chain(runtime.color_selector_runtimes.values().map(Box::as_ref));
    for child in children {
        let child = compiled_direct_profile_snapshot(child)?;
        snapshot.engine_count += child.engine_count;
        snapshot.minimum_effective_tile_capacity = minimum_nonzero(
            snapshot.minimum_effective_tile_capacity,
            child.minimum_effective_tile_capacity,
        );
        snapshot.maximum_physical_scalar_values_per_point = snapshot
            .maximum_physical_scalar_values_per_point
            .max(child.maximum_physical_scalar_values_per_point);
        snapshot.maximum_hot_scalar_values_per_point = snapshot
            .maximum_hot_scalar_values_per_point
            .max(child.maximum_hot_scalar_values_per_point);
        snapshot.maximum_source_scalar_values_per_point = snapshot
            .maximum_source_scalar_values_per_point
            .max(child.maximum_source_scalar_values_per_point);
        snapshot.maximum_reduction_scalar_values_per_point = snapshot
            .maximum_reduction_scalar_values_per_point
            .max(child.maximum_reduction_scalar_values_per_point);
        snapshot.source_fill_bytes = snapshot
            .source_fill_bytes
            .saturating_add(child.source_fill_bytes);
        snapshot.momentum_fill_bytes = snapshot
            .momentum_fill_bytes
            .saturating_add(child.momentum_fill_bytes);
        snapshot.parameter_fill_bytes = snapshot
            .parameter_fill_bytes
            .saturating_add(child.parameter_fill_bytes);
        snapshot.scalar_broadcast_fill_bytes = snapshot
            .scalar_broadcast_fill_bytes
            .saturating_add(child.scalar_broadcast_fill_bytes);
        snapshot.amplitude_clear_bytes = snapshot
            .amplitude_clear_bytes
            .saturating_add(child.amplitude_clear_bytes);
        snapshot.backend_call_count = snapshot
            .backend_call_count
            .saturating_add(child.backend_call_count);
        snapshot.boundary_input_bytes = snapshot
            .boundary_input_bytes
            .saturating_add(child.boundary_input_bytes);
        snapshot.boundary_current_output_bytes = snapshot
            .boundary_current_output_bytes
            .saturating_add(child.boundary_current_output_bytes);
        snapshot.boundary_amplitude_output_bytes = snapshot
            .boundary_amplitude_output_bytes
            .saturating_add(child.boundary_amplitude_output_bytes);
    }
    Ok(snapshot)
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn attach_compiled_direct_configuration(
    profile: &mut RuntimeProfile,
    snapshot: CompiledDirectProfileSnapshot,
) {
    profile.compiled_direct_arena_minimum_effective_tile_capacity =
        snapshot.minimum_effective_tile_capacity as u64;
    profile.compiled_direct_arena_maximum_physical_scalar_values_per_point =
        snapshot.maximum_physical_scalar_values_per_point as u64;
    profile.compiled_direct_arena_maximum_hot_scalar_values_per_point =
        snapshot.maximum_hot_scalar_values_per_point as u64;
    profile.compiled_direct_arena_maximum_source_scalar_values_per_point =
        snapshot.maximum_source_scalar_values_per_point as u64;
    profile.compiled_direct_arena_maximum_reduction_scalar_values_per_point =
        snapshot.maximum_reduction_scalar_values_per_point as u64;
}

fn ensure_selected_runtime_capabilities_supported(capabilities: &[String]) -> RusticolResult<()> {
    if capabilities.iter().any(|capability| {
        capability == EAGER_DAG_RUNTIME_CAPABILITY
            || capability == EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY
    }) {
        return Err(RusticolError::compatibility(
            "legacy eager plan-v2 artifacts are no longer executable; regenerate the artifact \
             with the current `pyamplicol generate` before loading it",
        ));
    }
    ensure_runtime_capabilities_supported(capabilities.iter().map(String::as_str))
}

impl NativeRuntime {
    pub const ABI_VERSION: u32 = crate::C_ABI_VERSION;

    /// Save the currently retained, completed on-the-fly cache to a file.
    /// The original process output is still required to load its kernels.
    /// Numeric momenta, parameters, and workspaces are not serialized.
    pub fn save(&self, path: impl AsRef<Path>) -> RusticolResult<()> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if let NativeExecutionLane::OnTheFly(runtime) = &self.execution_lane {
            return cache::save(self, runtime, path.as_ref());
        }
        let _ = path;
        Err(RusticolError::compatibility(
            "save() requires an on-the-fly runtime",
        ))
    }

    /// Restore completed on-the-fly structural work against this loaded
    /// process. The runner's current model parameters are kept. A failed load
    /// leaves the previously usable cache intact.
    pub fn load_cache(&mut self, path: impl AsRef<Path>) -> RusticolResult<()> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if matches!(self.execution_lane, NativeExecutionLane::OnTheFly(_)) {
            return cache::load(self, path.as_ref());
        }
        let _ = path;
        Err(RusticolError::compatibility(
            "load_cache() requires an on-the-fly runtime",
        ))
    }

    /// Content identity of the authenticated artifact manifest that supplied
    /// this in-memory runtime.
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// Load only the authenticated physical reduction groups of an eager
    /// plan-v3 artifact.
    ///
    /// Unlike the exact-section bridge, this does not materialize exact
    /// invocations, attachments, finalizations, or closures that are unrelated
    /// to reduction authentication.
    #[doc(hidden)]
    pub fn load_eager_reduction_groups(
        artifact_path: impl AsRef<Path>,
        process_id: &str,
    ) -> Result<Value, RusticolError> {
        let artifact = VerifiedArtifact::open_with_manifest_preflight(artifact_path, |manifest| {
            let selection = manifest.select_process(Some(process_id))?;
            ensure_selected_runtime_capabilities_supported(
                &selection.process.required_runtime_capabilities,
            )
        })?;
        let selection = artifact.select_process(Some(process_id))?;
        if selection.remaps_public_process() {
            return Err(RusticolError::invalid_argument(
                "compact eager reduction groups must be requested by representative process ID",
            ));
        }
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
        let mut physics =
            ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
        if physics.process_id != selection.process.id
            || physics.process != selection.process.expression
            || physics.color_accuracy.as_str() != selection.process.color_accuracy
            || physics
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .ne(selection.process.external_pdgs.iter().copied())
        {
            return Err(RusticolError::integrity(format!(
                "runtime physics payload {:?} does not match process {:?}",
                selection.process.physics_path, selection.process.id
            )));
        }
        match manifest {
            LoadedExecutionManifest::Compiled(_) => Err(RusticolError::compatibility(
                "compact reduction-group loading requires an eager plan-v3 artifact",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::EagerV3(manifest) => {
                super::eager_v3_load::load_eager_v3_reduction_groups(
                    &artifact,
                    &evaluator_root,
                    &manifest,
                    &mut physics,
                )
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::Recurrence(_) => Err(RusticolError::compatibility(
                "recurrence reduction-group loading is unavailable through the eager bridge",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::OnTheFly(_) => Err(RusticolError::compatibility(
                "on-the-fly reduction-group loading is unavailable through the eager bridge",
            )),
        }
    }

    /// Load only the exact-required sections of an eager plan-v3 artifact.
    ///
    /// This authenticates the artifact and compact runtime container but does
    /// not instantiate the prepared f64 backend. It is intentionally separate
    /// from the normal runtime hot path.
    pub fn load_eager_exact_sections(
        artifact_path: impl AsRef<Path>,
        process_id: &str,
    ) -> Result<NativeEagerExactSections, RusticolError> {
        let artifact = VerifiedArtifact::open_with_manifest_preflight(artifact_path, |manifest| {
            let selection = manifest.select_process(Some(process_id))?;
            ensure_selected_runtime_capabilities_supported(
                &selection.process.required_runtime_capabilities,
            )
        })?;
        let selection = artifact.select_process(Some(process_id))?;
        if selection.remaps_public_process() {
            return Err(RusticolError::invalid_argument(
                "compact eager exact sections must be requested by representative process ID",
            ));
        }
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
        let mut physics =
            ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
        if physics.process_id != selection.process.id
            || physics.process != selection.process.expression
            || physics.color_accuracy.as_str() != selection.process.color_accuracy
            || physics
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .ne(selection.process.external_pdgs.iter().copied())
        {
            return Err(RusticolError::integrity(format!(
                "runtime physics payload {:?} does not match process {:?}",
                selection.process.physics_path, selection.process.id
            )));
        }
        match manifest {
            LoadedExecutionManifest::Compiled(_) => Err(RusticolError::compatibility(
                "compact exact-section loading requires an eager plan-v3 artifact",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::EagerV3(manifest) => {
                super::eager_v3_load::load_eager_v3_exact_sections(
                    &artifact,
                    &evaluator_root,
                    &manifest,
                    &mut physics,
                )
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::Recurrence(_) => Err(RusticolError::compatibility(
                "recurrence exact-section loading is not available in the initial f64 runtime slice",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::OnTheFly(_) => Err(RusticolError::compatibility(
                "on-the-fly exact-section loading is unavailable through the eager bridge",
            )),
        }
    }

    /// Load the immutable exact-execution sections of an authenticated
    /// topology-replay recurrence artifact without instantiating its f64
    /// Direct-Arena backend.
    pub fn load_recurrence_exact_sections(
        artifact_path: impl AsRef<Path>,
        process_id: &str,
    ) -> Result<NativeRecurrenceExactSections, RusticolError> {
        let artifact = VerifiedArtifact::open_with_manifest_preflight(artifact_path, |manifest| {
            let selection = manifest.select_process(Some(process_id))?;
            ensure_selected_runtime_capabilities_supported(
                &selection.process.required_runtime_capabilities,
            )
        })?;
        let selection = artifact.select_process(Some(process_id))?;
        if selection.remaps_public_process() {
            return Err(RusticolError::invalid_argument(
                "compact recurrence exact sections must be requested by representative process ID",
            ));
        }
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
        let physics = ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
        if physics.process_id != selection.process.id
            || physics.process != selection.process.expression
            || physics.color_accuracy.as_str() != selection.process.color_accuracy
            || physics
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .ne(selection.process.external_pdgs.iter().copied())
        {
            return Err(RusticolError::integrity(format!(
                "runtime physics payload {:?} does not match process {:?}",
                selection.process.physics_path, selection.process.id
            )));
        }
        match manifest {
            LoadedExecutionManifest::Compiled(_) => Err(RusticolError::compatibility(
                "compact recurrence exact-section loading requires a recurrence artifact",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::EagerV3(_) => Err(RusticolError::compatibility(
                "compact recurrence exact-section loading requires a recurrence artifact",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::Recurrence(manifest) => {
                load_recurrence_exact_sections(&artifact, &evaluator_root, &manifest, &physics)
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::OnTheFly(_) => Err(RusticolError::compatibility(
                "compact recurrence exact-section loading is unavailable for on-the-fly artifacts",
            )),
        }
    }

    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    #[allow(clippy::too_many_arguments)]
    fn finish_dense_load(
        root: PathBuf,
        artifact_id: String,
        selection: crate::ArtifactSelection,
        mut physics_v1: ProcessPhysicsV1,
        representative_key: String,
        mut runtime: ExecutionRuntime,
        execution_lane: NativeExecutionLane,
        model_parameters_path: Option<&Path>,
    ) -> Result<Self, RusticolError> {
        let process = selection.public_expression.clone();
        let process_key = selection.requested_id.clone();
        let representative_process_id = selection.process.id.clone();
        let public_remap = selection.remaps_public_process();
        let nonidentity_permutation = selection
            .external_permutation
            .iter()
            .copied()
            .enumerate()
            .any(|(representative_index, public_index)| representative_index != public_index);
        let initial_count = physics_v1
            .external_particles
            .iter()
            .take_while(|particle| particle.role == crate::ParticleRole::Initial)
            .count();
        let final_state_only = public_remap
            && selection
                .external_permutation
                .iter()
                .take(initial_count)
                .copied()
                .enumerate()
                .all(|(index, public_index)| index == public_index);
        let input_crossing_map = if public_remap {
            let representative_physics = physics_v1.clone();
            let public_physics = apply_process_permutation_metadata(physics_v1, &selection)
                .map_err(|error| {
                    RusticolError::with_kind(
                        error.kind(),
                        format!("could not remap process physics metadata: {error}"),
                    )
                })?;
            let helicity_id_map = representative_physics
                .helicities
                .iter()
                .zip(&public_physics.helicities)
                .map(|(representative, public)| (representative.id.clone(), public.id.clone()))
                .collect::<BTreeMap<_, _>>();
            let color_id_map = representative_physics
                .color_components
                .iter()
                .zip(&public_physics.color_components)
                .map(|(representative, public)| {
                    (representative.id().to_string(), public.id().to_string())
                })
                .collect::<BTreeMap<_, _>>();
            runtime
                .remap_lc_topology_replay_public_labels(&selection.external_permutation)
                .map_err(|error| {
                    RusticolError::with_kind(
                        error.kind(),
                        format!("could not remap process execution metadata: {error}"),
                    )
                })?;
            runtime.remap_physics_reduction_overrides(&helicity_id_map, &color_id_map)?;
            physics_v1 = public_physics;
            runtime.set_external_pdg_order_recursive(&selection.external_pdgs);
            nonidentity_permutation.then(|| {
                selection
                    .external_permutation
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(target_index, source_index)| InputCrossingMapEntry {
                        target_index,
                        source_index,
                        sign: 1.0,
                    })
                    .collect()
            })
        } else {
            None
        };
        let input_crossing_map =
            prevalidate_input_crossing_lookup(runtime.external_count, input_crossing_map)?;
        runtime.attach_physics(Arc::new(PhysicsRuntime::new(physics_v1.clone())?))?;
        // Execution keeps the representative flavours and LC words; only the
        // exposed metadata names the selected process-group member.
        let physics_v1 = if selection.process_group.is_some() {
            apply_process_group_metadata(&physics_v1, &selection)?
        } else {
            physics_v1
        };
        if matches!(&execution_lane, NativeExecutionLane::Compiled) {
            runtime.initialize_compiled_helicity_execution_plan(
                public_remap.then_some(selection.external_permutation.as_slice()),
            )?;
        }
        let selector_simd_lane_width = {
            let uses_simd_jit = match &execution_lane {
                NativeExecutionLane::Compiled => execution_uses_simd_jit(&runtime),
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Eager(runtime) => runtime.backend_name() == "jit",
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Recurrence(runtime) => runtime.backend_name() == "jit",
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::OnTheFly(_) => false,
            };
            if uses_simd_jit {
                super::evaluator::native_f64_simd_lane_width()
            } else {
                1
            }
        };
        let mut loaded = Self {
            root,
            artifact_id,
            runtime,
            execution_lane,
            process,
            process_key,
            representative_process_id,
            external_permutation: selection.external_permutation.clone(),
            input_crossing_map,
            permutation_alias_of: public_remap.then(|| representative_key.clone()),
            process_group: process_group_routing(&selection, &representative_key),
            final_state_permutation_alias_of: final_state_only.then_some(representative_key),
            physics_v1: LazyProcessPhysicsV1::loaded(physics_v1),
            warnings_muted: false,
            warned_kinds: BTreeSet::new(),
            pending_warnings: Vec::new(),
            point_selector_scratch: PointSelectorExecutionScratch::default(),
            selector_simd_lane_width,
            correlated: None,
        };
        if let Some(path) = model_parameters_path {
            loaded.set_model_parameters_json(path)?;
        }
        Ok(loaded)
    }

    // Darwin reports each referenced clean executable page in process RSS.
    // Keep the authenticated process-ready load path and its load-only callees
    // in one Apple section; evaluation code remains in the ordinary text
    // section and non-Apple builds retain their normal placement.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    fn load_recurrence_bootstrap(
        bootstrap: crate::artifact::RecurrenceBootstrapArtifact,
        model_parameters_path: Option<&Path>,
    ) -> Result<Self, RusticolError> {
        let crate::artifact::RecurrenceBootstrapArtifact {
            mut artifact,
            selection,
            ready_execution,
            physics: physics_v1,
            evaluator_payload_container,
        } = bootstrap;
        ensure_selected_runtime_capabilities_supported(
            &selection.process.required_runtime_capabilities,
        )?;
        validate_capability_list_match(
            &artifact.manifest().runtime.required_runtime_capabilities,
            &ready_execution.required_runtime_capabilities,
            "outer runtime and recurrence process-ready recipe",
        )?;
        validate_representative_physics(&physics_v1, &selection)?;

        // The authenticated member uses the exact conventional execution path
        // above, and public process IDs are already confined by the outer
        // manifest validator. Avoid reopening the duplicate loose JSON merely
        // to recover its parent directory.
        let evaluator_root = artifact
            .root()
            .join("processes")
            .join(&selection.process.id);
        let representative_key = ready_execution.key.clone();
        let loaded = load_recurrence_ready_runtime(
            &mut artifact,
            &evaluator_root,
            &ready_execution,
            &physics_v1,
            &evaluator_payload_container,
        )?;
        Self::finish_dense_load(
            artifact.root().to_path_buf(),
            artifact.manifest().artifact_id.clone(),
            selection,
            physics_v1,
            representative_key,
            loaded.common,
            NativeExecutionLane::Recurrence(Box::new(loaded.lane)),
            model_parameters_path,
        )
    }

    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub fn load(
        artifact_path: impl AsRef<Path>,
        process_id: Option<&str>,
        model_parameters_path: Option<&Path>,
    ) -> Result<Self, RusticolError> {
        let artifact_path = artifact_path.as_ref();
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if let Some(bootstrap) =
            VerifiedArtifact::try_open_recurrence_bootstrap(artifact_path, process_id)?
        {
            return Self::load_recurrence_bootstrap(bootstrap, model_parameters_path);
        }
        Self::load_generic(artifact_path, process_id, model_parameters_path)
    }

    #[cold]
    #[inline(never)]
    fn load_generic(
        artifact_path: &Path,
        process_id: Option<&str>,
        model_parameters_path: Option<&Path>,
    ) -> Result<Self, RusticolError> {
        let artifact = VerifiedArtifact::open_with_manifest_preflight(artifact_path, |manifest| {
            match manifest.select_process(process_id) {
                Ok(selection) => ensure_selected_runtime_capabilities_supported(
                    &selection.process.required_runtime_capabilities,
                ),
                // A process-group member resolves only after the map payload
                // is read; its representative is checked below.
                Err(_)
                    if manifest
                        .payloads
                        .iter()
                        .any(|payload| payload.role == crate::PayloadRole::ProcessGroups) =>
                {
                    Ok(())
                }
                Err(error) => Err(error),
            }
        })?;
        let artifact_id = artifact.manifest().artifact_id.clone();
        let selection = artifact.select_process(process_id)?;
        if selection.process_group.is_some() {
            ensure_selected_runtime_capabilities_supported(
                &selection.process.required_runtime_capabilities,
            )?;
        }
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if let LoadedExecutionManifest::OnTheFly(on_the_fly) = &manifest {
            let public_remap = selection.remaps_public_process();
            let nonidentity_permutation = selection
                .external_permutation
                .iter()
                .copied()
                .enumerate()
                .any(|(representative_index, public_index)| representative_index != public_index);
            let initial_count = on_the_fly
                .runtime_metadata
                .external_legs
                .iter()
                .filter(|leg| leg.is_initial)
                .count();
            let final_state_only = public_remap
                && selection
                    .external_permutation
                    .iter()
                    .take(initial_count)
                    .copied()
                    .enumerate()
                    .all(|(index, public_index)| index == public_index);
            let loaded = super::on_the_fly_load::load_on_the_fly_native_runtime(
                &artifact,
                &evaluator_root,
                on_the_fly,
                &selection,
            )?;
            let super::on_the_fly_load::LoadedOnTheFlyRuntime {
                mut common,
                lane,
                selectors,
                metadata_selectors,
                public_metadata,
                color_contraction,
            } = loaded;
            if public_remap {
                common.set_external_pdg_order_recursive(&selection.external_pdgs);
            }
            let input_crossing_map = nonidentity_permutation.then(|| {
                selection
                    .external_permutation
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(target_index, source_index)| InputCrossingMapEntry {
                        target_index,
                        source_index,
                        sign: 1.0,
                    })
                    .collect::<Vec<_>>()
            });
            let input_crossing_map =
                prevalidate_input_crossing_lookup(common.external_count, input_crossing_map)?;
            let representative_key = on_the_fly.key.clone();
            let mut runtime = Self {
                root: artifact.root().to_path_buf(),
                artifact_id,
                runtime: common,
                execution_lane: NativeExecutionLane::OnTheFly(Box::new(
                    OnTheFlyExecutionRuntime::new(lane, selectors, color_contraction),
                )),
                process: selection.public_expression.clone(),
                process_key: selection.requested_id.clone(),
                representative_process_id: selection.process.id.clone(),
                external_permutation: selection.external_permutation.clone(),
                input_crossing_map,
                permutation_alias_of: public_remap.then(|| representative_key.clone()),
                process_group: process_group_routing(&selection, &representative_key),
                final_state_permutation_alias_of: final_state_only.then_some(representative_key),
                physics_v1: LazyProcessPhysicsV1::deferred_on_the_fly(
                    public_metadata,
                    metadata_selectors,
                    selection,
                ),
                warnings_muted: false,
                warned_kinds: BTreeSet::new(),
                pending_warnings: Vec::new(),
                point_selector_scratch: PointSelectorExecutionScratch::default(),
                selector_simd_lane_width: 1,
                correlated: None,
            };
            if let Some(path) = model_parameters_path {
                runtime.set_model_parameters_json(path)?;
            }
            return Ok(runtime);
        }
        let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
        let mut physics_v1 =
            ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
        if physics_v1.process_id != selection.process.id
            || physics_v1.process != selection.process.expression
            || physics_v1.color_accuracy.as_str() != selection.process.color_accuracy
            || physics_v1
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .ne(selection.process.external_pdgs.iter().copied())
        {
            return Err(RusticolError::integrity(format!(
                "runtime physics payload {:?} does not match process {:?}",
                selection.process.physics_path, selection.process.id
            )));
        }
        let (_representative_process, representative_key, runtime, execution_lane) = match manifest
        {
            LoadedExecutionManifest::Compiled(manifest) => {
                super::eager_v3_load::reject_native_reduction_groups_for_compiled(&physics_v1)?;
                let representative_process = manifest.process.clone();
                let representative_key = manifest.key.clone();
                let evaluator_payloads = artifact.evaluator_payload_store(&evaluator_root)?;
                // Tile sizing is bound to the representative physics
                // before public alias remapping. Aliases preserve these
                // component counts and the execution payload itself is
                // authenticated against the representative identifiers.
                let sizing_physics = PhysicsRuntime::new(physics_v1.clone())?;
                let runtime = load_execution_manifest_with_store(
                    *manifest,
                    &evaluator_payloads,
                    &sizing_physics,
                )?;
                (
                    representative_process,
                    representative_key,
                    runtime,
                    NativeExecutionLane::Compiled,
                )
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::EagerV3(manifest) => {
                let representative_process = manifest.process.clone();
                let representative_key = manifest.key.clone();
                let loaded = super::eager_v3_load::load_eager_v3_native_runtime(
                    &artifact,
                    &evaluator_root,
                    &manifest,
                    &mut physics_v1,
                )?;
                (
                    representative_process,
                    representative_key,
                    loaded.common,
                    NativeExecutionLane::Eager(Box::new(loaded.lane)),
                )
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::Recurrence(manifest) => {
                let representative_process = manifest.process.clone();
                let representative_key = manifest.key.clone();
                let loaded = load_recurrence_native_runtime(
                    &artifact,
                    &evaluator_root,
                    &manifest,
                    &physics_v1,
                    &selection,
                )?;
                (
                    representative_process,
                    representative_key,
                    loaded.common,
                    NativeExecutionLane::Recurrence(Box::new(loaded.lane)),
                )
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::OnTheFly(_) => {
                unreachable!("on-the-fly manifests are dispatched before dense physics loading")
            }
        };
        Self::finish_dense_load(
            artifact.root().to_path_buf(),
            artifact_id,
            selection,
            physics_v1,
            representative_key,
            runtime,
            execution_lane,
            model_parameters_path,
        )
    }

    pub fn metadata(&self) -> NativeRuntimeMetadata {
        let (
            execution_mode,
            prepared_backend,
            eager_effective_point_tile_size,
            eager_workspace_bytes,
            on_the_fly_requested_query_construction_threads,
            on_the_fly_effective_query_construction_threads,
        ) = match &self.execution_lane {
            NativeExecutionLane::Compiled => ("compiled", None, None, None, None, None),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => (
                "eager",
                Some(runtime.backend_name().to_string()),
                Some(runtime.effective_point_tile_size()),
                Some(runtime.workspace_bytes()),
                None,
                None,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => (
                "recurrence",
                Some(runtime.backend_name().to_string()),
                Some(runtime.effective_point_tile_size()),
                None,
                None,
                None,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::OnTheFly(runtime) => (
                "on-the-fly",
                None,
                None,
                None,
                Some(runtime.lane.requested_query_construction_threads()),
                Some(runtime.lane.effective_query_construction_threads()),
            ),
        };
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        let compiled_direct_configuration = compiled_direct_configuration_snapshot(&self.runtime);
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        let (
            compiled_direct_minimum_effective_tile_capacity,
            compiled_direct_maximum_physical_scalar_values_per_point,
            compiled_direct_maximum_hot_scalar_values_per_point,
            compiled_direct_maximum_source_scalar_values_per_point,
            compiled_direct_maximum_reduction_scalar_values_per_point,
        ) = if compiled_direct_configuration.engine_count == 0 {
            (None, None, None, None, None)
        } else {
            (
                Some(compiled_direct_configuration.minimum_effective_tile_capacity),
                Some(compiled_direct_configuration.maximum_physical_scalar_values_per_point),
                Some(compiled_direct_configuration.maximum_hot_scalar_values_per_point),
                Some(compiled_direct_configuration.maximum_source_scalar_values_per_point),
                Some(compiled_direct_configuration.maximum_reduction_scalar_values_per_point),
            )
        };
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        let (
            compiled_direct_minimum_effective_tile_capacity,
            compiled_direct_maximum_physical_scalar_values_per_point,
            compiled_direct_maximum_hot_scalar_values_per_point,
            compiled_direct_maximum_source_scalar_values_per_point,
            compiled_direct_maximum_reduction_scalar_values_per_point,
        ) = (None, None, None, None, None);
        NativeRuntimeMetadata {
            abi_version: Self::ABI_VERSION,
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            execution_mode: execution_mode.to_string(),
            prepared_backend,
            eager_effective_point_tile_size,
            eager_workspace_bytes,
            on_the_fly_requested_query_construction_threads,
            on_the_fly_effective_query_construction_threads,
            compiled_direct_minimum_effective_tile_capacity,
            compiled_direct_maximum_physical_scalar_values_per_point,
            compiled_direct_maximum_hot_scalar_values_per_point,
            compiled_direct_maximum_source_scalar_values_per_point,
            compiled_direct_maximum_reduction_scalar_values_per_point,
            process: self.process.clone(),
            process_key: self.process_key.clone(),
            representative_process: self.runtime.process.clone(),
            representative_process_key: self.runtime.key.clone(),
            external_permutation: self.external_permutation.clone(),
            permutation_alias_of: self.permutation_alias_of.clone(),
            final_state_permutation_alias_of: self.final_state_permutation_alias_of.clone(),
            process_group_member_of: self
                .process_group
                .as_ref()
                .map(|group| group.representative_process_key.clone()),
            process_group_factor: self.process_group_factor(),
            charge_conjugated: self.charge_conjugated(),
            color_accuracy: self.runtime.color_accuracy.clone(),
            external_pdg_order: self.runtime.external_pdg_order.clone(),
            external_count: self.runtime.external_count,
            current_count: self.runtime.current_count,
            source_count: self.runtime.source_count,
            interaction_count: self.runtime.interaction_count,
            stage_count: self.runtime.stage_count,
            amplitude_output_count: self.runtime.amplitude_output_count,
        }
    }

    fn selector_simd_lane_width(&self) -> usize {
        self.selector_simd_lane_width
    }

    pub fn metadata_json(&self) -> Result<String, RusticolError> {
        serde_json::to_string(&self.metadata()).map_err(|error| {
            RusticolError::serialization(format!("could not serialize runtime metadata: {error}"))
        })
    }

    pub fn physics_json(&self) -> Result<String, RusticolError> {
        serde_json::to_string(self.physics_v1.get()?).map_err(|error| {
            RusticolError::serialization(format!("could not serialize physics metadata: {error}"))
        })
    }

    /// Return the validated mutable state needed by the lazy Python
    /// high-precision executor.
    ///
    /// The values have already passed Rusticol's atomic parameter update and
    /// derived-parameter refresh logic. They are intentionally exposed only as
    /// an internal bridge; f64 evaluation continues to execute entirely in the
    /// Python-independent core.
    pub fn exact_runtime_state_json(&self) -> Result<String, RusticolError> {
        serde_json::to_string(&serde_json::json!({
            "model_parameter_values": self.runtime.model_parameter_values_f64,
            "normalization_factor": self.runtime.normalization_factor,
            "representative_process_id": self.representative_process_id,
            "representative_process_key": self.runtime.key,
            "external_permutation": self.external_permutation,
            "process_group_member_of": self
                .process_group
                .as_ref()
                .map(|group| group.representative_process_key.as_str()),
            "process_group_factor": self.process_group_factor(),
            "charge_conjugated": self.charge_conjugated(),
        }))
        .map_err(|error| {
            RusticolError::serialization(format!(
                "could not serialize exact-runtime state: {error}"
            ))
        })
    }

    pub fn process_physics(&self) -> RusticolResult<&ProcessPhysicsV1> {
        self.physics_v1.get()
    }

    pub fn external_count(&self) -> usize {
        self.runtime.external_count
    }

    pub fn representative_process_key(&self) -> &str {
        &self.runtime.key
    }

    /// Full representative-index to public-index external permutation.
    pub fn external_permutation(&self) -> &[usize] {
        &self.external_permutation
    }

    /// Exact factor multiplying every representative matrix element; one
    /// unless the selection is a process-group member.
    pub fn process_group_factor(&self) -> NativeRationalFactor {
        self.process_group.as_ref().map_or(
            NativeRationalFactor {
                numerator: 1,
                denominator: 1,
            },
            |group| group.factor,
        )
    }

    /// Whether the selection is the charge conjugate of its representative.
    pub fn charge_conjugated(&self) -> bool {
        self.process_group
            .as_ref()
            .is_some_and(|group| group.conjugated)
    }

    /// Scale representative matrix elements to the selected member.
    pub(super) fn apply_process_group_factor(&self, values: &mut [f64]) {
        let Some(group) = self.process_group.as_deref() else {
            return;
        };
        let NativeRationalFactor {
            numerator,
            denominator,
        } = group.factor;
        if numerator != denominator {
            let (numerator, denominator) = (numerator as f64, denominator as f64);
            for value in values {
                *value = *value * numerator / denominator;
            }
        }
    }

    /// Translate caller color IDs to the IDs the execution lanes use.
    pub(super) fn execution_color_ids<'a>(
        &self,
        ids: Option<&'a [String]>,
    ) -> Option<Cow<'a, [String]>> {
        let ids = ids?;
        Some(if self.charge_conjugated() {
            Cow::Owned(
                ids.iter()
                    .map(|id| charge_conjugate_color_id(id).into_owned())
                    .collect(),
            )
        } else {
            Cow::Borrowed(ids)
        })
    }

    /// Translate execution color IDs back to the caller's labels.
    pub(super) fn publish_color_ids(&self, ids: &mut [String]) {
        if self.charge_conjugated() {
            for id in ids {
                *id = charge_conjugate_color_id(id).into_owned();
            }
        }
    }

    /// Load one public-order kinematic point from JSON.
    ///
    /// The accepted shapes are `[external][4]` and a singleton batch
    /// `[[external][4]]`. Components may be JSON numbers or decimal strings.
    /// The returned flat values retain the caller's public process ordering;
    /// the ordinary evaluation entry points apply any active process
    /// permutation when packing runtime inputs.
    pub fn load_kinematics_json(&self, path: impl AsRef<Path>) -> Result<Vec<f64>, RusticolError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| {
            RusticolError::invalid_argument(format!(
                "could not read kinematics JSON {}: {error}",
                path.display()
            ))
        })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
            RusticolError::invalid_argument(format!(
                "could not parse kinematics JSON {}: {error}",
                path.display()
            ))
        })?;
        parse_public_kinematics_point(&value, self.external_count())
    }

    pub fn external_particles(&self) -> Result<Vec<NativeExternalParticle>, RusticolError> {
        Ok(self
            .physics_v1
            .get()?
            .external_particles
            .iter()
            .map(|item| NativeExternalParticle {
                label: item.label,
                index: item.index,
                side: match item.role {
                    crate::ParticleRole::Initial => "initial",
                    crate::ParticleRole::Final => "final",
                }
                .to_string(),
                role: match item.role {
                    crate::ParticleRole::Initial => "initial",
                    crate::ParticleRole::Final => "final",
                }
                .to_string(),
                particle: item.particle.clone(),
                outgoing_particle: item.particle.clone(),
                pdg: item.pdg,
                outgoing_pdg: item.pdg,
                particle_class: String::new(),
                momentum_slot: item.momentum_slot,
            })
            .collect())
    }

    pub fn helicities(&self) -> Result<Vec<NativeHelicityConfiguration>, RusticolError> {
        Ok(self
            .physics_v1
            .get()?
            .helicities
            .iter()
            .map(|item| NativeHelicityConfiguration {
                id: item.id.clone(),
                index: item.index,
                helicities: item.values.clone(),
                representative_id: item.representative_id.clone(),
                computed: item.computed,
                structural_zero: item.structural_zero,
                coefficient: item.coefficient,
            })
            .collect())
    }

    pub fn color_components(&self) -> Result<Vec<NativeColorComponent>, RusticolError> {
        Ok(self
            .physics_v1
            .get()?
            .color_components
            .iter()
            .map(|item| match item {
                PhysicsColorComponentV1::LcFlow(flow) => NativeColorComponent {
                    id: flow.id.clone(),
                    index: flow.index,
                    kind: "lc-flow".to_string(),
                    word: flow.word.clone(),
                    representative_id: flow.representative_id.clone(),
                    computed: flow.computed,
                    coefficient: flow.coefficient,
                },
                PhysicsColorComponentV1::ContractedColor(color) => NativeColorComponent {
                    id: color.id.clone(),
                    index: color.index,
                    kind: "contracted-color".to_string(),
                    word: Vec::new(),
                    representative_id: color.id.clone(),
                    computed: true,
                    coefficient: 1.0,
                },
            })
            .collect())
    }

    /// Return the current logical parameter value, including refreshed derived values.
    pub fn model_parameter_value(&self, name: &str) -> Result<(f64, f64), RusticolError> {
        if !self
            .physics_v1
            .get()?
            .model_parameters
            .iter()
            .any(|parameter| parameter.name == name)
        {
            return Err(RusticolError::model_parameter(format!(
                "model parameter {name:?} is not declared by process {}",
                self.process
            )));
        }
        self.runtime.current_model_parameter_value(name)
    }

    pub fn model_parameters(&self) -> Result<Vec<NativeModelParameter>, RusticolError> {
        Ok(self
            .physics_v1
            .get()?
            .model_parameters
            .iter()
            .enumerate()
            .map(|(parameter_index, item)| NativeModelParameter {
                name: item.name.clone(),
                kind: format!("{:?}", item.kind).to_ascii_lowercase(),
                parameter_index,
                default: item.default_real,
                default_imaginary: item.default_imaginary,
                mutable: item.mutable,
            })
            .collect())
    }

    pub fn helicity_ids(&self) -> Result<Vec<String>, RusticolError> {
        Ok(self
            .physics_v1
            .get()?
            .helicities
            .iter()
            .map(|item| item.id.clone())
            .collect())
    }

    pub fn color_ids(&self) -> Result<Vec<String>, RusticolError> {
        Ok(self
            .physics_v1
            .get()?
            .color_components
            .iter()
            .map(|item| item.id().to_string())
            .collect())
    }

    pub fn resolved_shape(
        &self,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<(usize, usize), RusticolError> {
        let color_ids = self.execution_color_ids(color_ids);
        let color_ids = color_ids.as_deref();
        self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
        let selected_helicities = selector_set(helicity_ids, "helicity")?;
        let selected_colors = selector_set(color_ids, "color component")?;
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if let NativeExecutionLane::OnTheFly(runtime) = &self.execution_lane {
            let selection = runtime.selectors.selection(
                runtime.lane.seed(),
                selected_helicities.as_ref(),
                if runtime.uses_contracted_color() {
                    None
                } else {
                    selected_colors.as_ref()
                },
            )?;
            return Ok((
                selection.helicity_count(),
                if runtime.uses_contracted_color() {
                    1
                } else {
                    selection.color_count()
                },
            ));
        }
        let physics = self.runtime.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let helicity_count = physics
            .selected_helicity_indices(selected_helicities.as_ref())
            .map_err(|error| RusticolError::selector(error.to_string()))?
            .len();
        let color_count = physics
            .selected_color_indices(selected_colors.as_ref())
            .map_err(|error| RusticolError::selector(error.to_string()))?
            .len();
        Ok((helicity_count, color_count))
    }

    /// Private compact context for the existing Python benchmark service.
    /// Non-OTF lanes return `None`; OTF resolves optional one-based color
    /// ordinals without opening process-wide physics metadata.
    pub fn on_the_fly_benchmark_context_json(
        &self,
        requested_color_ids: Option<&[String]>,
    ) -> RusticolResult<Option<String>> {
        let requested_color_ids = self.execution_color_ids(requested_color_ids);
        let requested_color_ids = requested_color_ids.as_deref();
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if let NativeExecutionLane::OnTheFly(runtime) = &self.execution_lane {
            let selection = runtime
                .selectors
                .selection(runtime.lane.seed(), None, None)?;
            let mut selected_color_ids = Vec::new();
            if let Some(requested) = requested_color_ids {
                if runtime.uses_contracted_color() {
                    return Err(RusticolError::selector(
                        "on-the-fly contracted color does not expose benchmark color selectors",
                    ));
                }
                selected_color_ids
                    .try_reserve_exact(requested.len())
                    .map_err(|error| {
                        RusticolError::invalid_argument(format!(
                            "on-the-fly benchmark selector allocation failed: {error}"
                        ))
                    })?;
                for value in requested {
                    let resolved = match value.parse::<usize>() {
                        Ok(ordinal) if ordinal.to_string() == value.trim() => {
                            if ordinal == 0 || ordinal > selection.color_count() {
                                return Err(RusticolError::selector(format!(
                                    "color-flow ordinal {value:?} is out of range; choose 1..{} or a stable color component ID",
                                    selection.color_count(),
                                )));
                            }
                            selection.color_id_at(ordinal - 1)?
                        }
                        _ => {
                            runtime.selectors.parse_color_id(value)?;
                            value.clone()
                        }
                    };
                    selected_color_ids.push(resolved);
                }
            }
            self.publish_color_ids(&mut selected_color_ids);
            return serde_json::to_string(&serde_json::json!({
                "process_id": self.process_key,
                "process_expression": self.process,
                "color_accuracy": self.runtime.color_accuracy,
                "helicity_count": selection.helicity_count(),
                "color_count": if runtime.uses_contracted_color() { 1 } else { selection.color_count() },
                "selected_color_ids": selected_color_ids,
            }))
            .map(Some)
            .map_err(|error| {
                RusticolError::serialization(format!(
                    "could not serialize on-the-fly benchmark context: {error}"
                ))
            });
        }
        Ok(None)
    }

    /// Resolve existing public per-point selector IDs to compact public-axis
    /// ordinals without materializing process-wide physics metadata.
    pub fn on_the_fly_selector_ordinals_json(
        &self,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> RusticolResult<Option<String>> {
        let color_ids = self.execution_color_ids(color_ids);
        let color_ids = color_ids.as_deref();
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if let NativeExecutionLane::OnTheFly(runtime) = &self.execution_lane {
            if runtime.uses_contracted_color() && color_ids.is_some() {
                return Err(RusticolError::selector(
                    "on-the-fly contracted color does not expose color selector ordinals",
                ));
            }
            let helicity_ordinals = helicity_ids
                .map(|ids| {
                    ids.iter()
                        .map(|id| {
                            let selected = BTreeSet::from([id.clone()]);
                            runtime
                                .selectors
                                .selection(runtime.lane.seed(), Some(&selected), None)?
                                .helicity_ordinal_at(0)
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                })
                .transpose()?;
            let color_ordinals = color_ids
                .map(|ids| {
                    ids.iter()
                        .map(|id| {
                            let selected = BTreeSet::from([id.clone()]);
                            runtime
                                .selectors
                                .selection(runtime.lane.seed(), None, Some(&selected))?
                                .color_ordinal_at(0)
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                })
                .transpose()?;
            return serde_json::to_string(&serde_json::json!({
                "helicity_ordinals": helicity_ordinals,
                "color_ordinals": color_ordinals,
            }))
            .map(Some)
            .map_err(|error| {
                RusticolError::serialization(format!(
                    "could not serialize on-the-fly selector ordinals: {error}"
                ))
            });
        }
        Ok(None)
    }

    /// Return private read-only production introspection for the loaded compact
    /// query-family lane.
    ///
    /// This observes the same retained selector/family caches used by public
    /// evaluation. It does not construct selectors or open dense physics
    /// metadata. A recurrence lane reports the same census only when it owns
    /// the authenticated singleton-helicity companion; other lanes return
    /// `None`.
    pub fn on_the_fly_runtime_state_census_json(&self) -> RusticolResult<Option<String>> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            let census = match &self.execution_lane {
                NativeExecutionLane::OnTheFly(runtime) => {
                    Some(runtime.state_census(&self.process_key)?)
                }
                NativeExecutionLane::Recurrence(runtime) => {
                    runtime.helicity_selector_companion_state_census(&self.process_key)?
                }
                _ => None,
            };
            let Some(census) = census else {
                return Ok(None);
            };
            serde_json::to_string(&census).map(Some).map_err(|error| {
                RusticolError::serialization(format!(
                    "could not serialize on-the-fly runtime state census: {error}"
                ))
            })
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        Ok(None)
    }

    /// Explicitly construct and retain one selected OTF family, then execute
    /// exactly one binary64 point before committing its selector identity.
    /// Progress callbacks run only on the coordinating caller thread.
    /// `n_cores` overrides cold query construction for this call only; `None`
    /// uses the process-output default. Positive overrides are capped by host
    /// availability and do not invalidate an already retained family.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn warm_up_f64<'observer>(
        &mut self,
        momenta: &[f64],
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        n_cores: Option<usize>,
        observer: Option<
            &'observer mut super::on_the_fly_warm_up::NativeOnTheFlyWarmUpObserver<'observer>,
        >,
    ) -> RusticolResult<super::on_the_fly_warm_up::NativeOnTheFlyWarmUpResult> {
        self.warm_up_on_the_fly_f64_with_selectors(
            momenta,
            helicity_ids,
            color_ids,
            n_cores,
            observer,
        )
    }

    /// Explicitly named alias for [`Self::warm_up_f64`], retained for bindings
    /// which make the on-the-fly-only capability visible in their method name.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn warm_up_on_the_fly_f64_with_selectors<'observer>(
        &mut self,
        momenta: &[f64],
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        n_cores: Option<usize>,
        observer: Option<
            &'observer mut super::on_the_fly_warm_up::NativeOnTheFlyWarmUpObserver<'observer>,
        >,
    ) -> RusticolResult<super::on_the_fly_warm_up::NativeOnTheFlyWarmUpResult> {
        let color_ids = self.execution_color_ids(color_ids);
        let color_ids = color_ids.as_deref();
        let expected = self.runtime.external_count.checked_mul(4).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly warm-up momentum shape exceeds usize")
        })?;
        if momenta.len() != expected {
            return Err(RusticolError::invalid_argument(format!(
                "on-the-fly warm-up requires exactly one point ({expected} binary64 momentum values), received {}",
                momenta.len()
            )));
        }
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let mut selector_scratch = std::mem::take(&mut self.point_selector_scratch);
        let result = (|| {
            let selected_helicities = selector_scratch
                .helicity_selector_sets
                .resolve(helicity_ids, "helicity")?;
            let selected_colors = selector_scratch
                .color_selector_sets
                .resolve(color_ids, "color component")?;
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                1,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            let NativeExecutionLane::OnTheFly(runtime) = &mut self.execution_lane else {
                return Err(RusticolError::unsupported_runtime_capability(
                    ON_THE_FLY_RUNTIME_CAPABILITY,
                    "the explicit warm-up API requires an on-the-fly artifact",
                ));
            };
            runtime.warm_up_f64(
                &self.runtime,
                batch,
                selected_helicities,
                selected_colors,
                n_cores,
                observer,
            )
        })();
        self.point_selector_scratch = selector_scratch;
        self.input_crossing_map = crossing_lookup;
        result
    }

    /// Resolve and retain batch-global recurrence selectors once.
    ///
    /// The returned handle is tied to this artifact and representative
    /// process. Handle creation may allocate; repeated evaluation through the
    /// prepared-plan API does not rebuild string selector sets.
    pub fn prepare_recurrence_selector_plan(
        &self,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> RusticolResult<NativeRecurrenceSelectorPlan> {
        let NativeExecutionLane::Recurrence(runtime) = &self.execution_lane else {
            return Err(RusticolError::invalid_argument(
                "recurrence selector plans require a recurrence artifact",
            ));
        };
        let color_ids = self.execution_color_ids(color_ids);
        let color_ids = color_ids.as_deref();
        self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
        let selected_helicities = selector_set(helicity_ids, "helicity")?;
        let selected_colors = selector_set(color_ids, "color component")?;
        let physics = self.runtime.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        runtime.validate_global_selectors(
            physics,
            selected_helicities.as_ref(),
            selected_colors.as_ref(),
        )?;
        Ok(NativeRecurrenceSelectorPlan {
            artifact_root: self.root.clone(),
            process_key: self.process_key.clone(),
            external_permutation: self.external_permutation.clone(),
            selected_helicities,
            selected_colors,
        })
    }

    pub fn evaluate_f64(
        &mut self,
        momenta: &[f64],
        point_count: usize,
    ) -> Result<Vec<f64>, RusticolError> {
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let mut output = vec![0.0; point_count];
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let result = (|| {
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            self.run_f64_batch_into(batch, &mut output)
        })();
        self.input_crossing_map = crossing_lookup;
        result?;
        self.apply_process_group_factor(&mut output);
        Ok(output)
    }

    pub fn evaluate_f64_into(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "evaluation output has length {}, expected {point_count}",
                output.len()
            )));
        }
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let result = (|| {
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            self.run_f64_batch_into(batch, output)
        })();
        self.input_crossing_map = crossing_lookup;
        result?;
        self.apply_process_group_factor(output);
        Ok(())
    }

    /// Evaluate recurrence totals with a previously resolved selector plan.
    pub fn evaluate_f64_into_with_recurrence_selector_plan(
        &mut self,
        plan: &NativeRecurrenceSelectorPlan,
        momenta: &[f64],
        point_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        if !matches!(&self.execution_lane, NativeExecutionLane::Recurrence(_)) {
            return Err(RusticolError::invalid_argument(
                "recurrence selector plans require a recurrence artifact",
            ));
        }
        plan.ensure_matches(self)?;
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "evaluation output has length {}, expected {point_count}",
                output.len()
            )));
        }
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let result = (|| {
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            if plan.selected_helicities.is_some() || plan.selected_colors.is_some() {
                self.run_selected_f64_batch_into(
                    batch,
                    plan.selected_helicities.as_ref(),
                    plan.selected_colors.as_ref(),
                    output,
                )
            } else {
                self.run_f64_batch_into(batch, output)
            }
        })();
        self.input_crossing_map = crossing_lookup;
        result?;
        self.apply_process_group_factor(output);
        Ok(())
    }

    /// Evaluate one total per point with optional global or per-point selectors.
    ///
    /// Global selectors retain the existing subset semantics. Per-point
    /// selectors are resolved physical-axis indices and contain exactly one
    /// selector for every input point. The two forms are mutually exclusive on
    /// the same axis. An omitted axis is summed over all components retained by
    /// the artifact.
    pub fn evaluate_f64_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<Vec<f64>, RusticolError> {
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let mut output = vec![0.0; point_count];
        self.evaluate_f64_into_with_selectors(
            momenta,
            point_count,
            helicity_ids,
            color_ids,
            helicity_by_point,
            color_by_point,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_f64_into_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let color_ids = self.execution_color_ids(color_ids);
        self.evaluate_representative_f64_into_with_selectors(
            momenta,
            point_count,
            helicity_ids,
            color_ids.as_deref(),
            helicity_by_point,
            color_by_point,
            output,
        )?;
        self.apply_process_group_factor(output);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_representative_f64_into_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "evaluation output has length {}, expected {point_count}",
                output.len()
            )));
        }
        if helicity_ids.is_some() && helicity_by_point.is_some() {
            return Err(RusticolError::selector(
                "helicities and helicity_by_point are mutually exclusive",
            ));
        }
        if color_ids.is_some() && color_by_point.is_some() {
            return Err(RusticolError::selector(
                "color_flows and color_flow_by_point are mutually exclusive",
            ));
        }
        self.validate_selector_capabilities(
            helicity_ids.is_some() || helicity_by_point.is_some(),
            color_ids.is_some() || color_by_point.is_some(),
        )?;
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if matches!(&self.execution_lane, NativeExecutionLane::OnTheFly(_)) {
            return self.evaluate_on_the_fly_f64_into_with_selectors(
                momenta,
                point_count,
                helicity_ids,
                color_ids,
                helicity_by_point,
                color_by_point,
                output,
            );
        }
        let physics = self.runtime.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let mut selector_scratch = std::mem::take(&mut self.point_selector_scratch);
        let result = (|| {
            if helicity_by_point.is_some() || color_by_point.is_some() {
                selector_scratch.prepare_singletons(&physics);
            }
            let PointSelectorExecutionScratch {
                planner,
                gathered_batch,
                partition_totals,
                output_totals,
                helicity_selector_sets,
                color_selector_sets,
                helicity_singletons,
                color_singletons,
            } = &mut selector_scratch;
            let selected_helicities = helicity_selector_sets.resolve(helicity_ids, "helicity")?;
            let selected_colors = color_selector_sets.resolve(color_ids, "color component")?;
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            let plan = planner.build(
                point_count,
                helicity_by_point,
                color_by_point,
                physics.manifest.helicities.len(),
                physics.manifest.color_components.len(),
            )?;
            if plan == PointSelectorPlan::None {
                if selected_helicities.is_some() || selected_colors.is_some() {
                    return self.run_selected_f64_batch_into(
                        batch,
                        selected_helicities,
                        selected_colors,
                        output,
                    );
                }
                return self.run_f64_batch_into(batch, output);
            }

            self.record_resolved_warnings(helicity_ids, color_ids)?;
            if let PointSelectorPlan::Homogeneous(key) = plan {
                // Selecting the sole retained value of an axis is identical
                // to leaving that axis unselected.  Generation-specialized
                // recurrence artifacts use exactly this shape in scalar
                // performance and event-loop calls; route it through the
                // ordinary borrowed-output path so we do not rescan singleton
                // string sets or stage through a second output buffer.
                let point_helicities = key.helicity_index.and_then(|index| {
                    (physics.manifest.helicities.len() != 1).then(|| &helicity_singletons[index])
                });
                let point_colors = key.color_index.and_then(|index| {
                    (physics.manifest.color_components.len() != 1).then(|| &color_singletons[index])
                });
                let effective_helicities = point_helicities.or(selected_helicities);
                let effective_colors = point_colors.or(selected_colors);
                if effective_helicities.is_none() && effective_colors.is_none() {
                    self.run_f64_batch_into(batch, output)?;
                } else {
                    self.run_selected_f64_batch_into(
                        batch,
                        effective_helicities,
                        effective_colors,
                        output,
                    )?;
                }
                return Ok(());
            }

            output_totals.resize(point_count, 0.0);
            output_totals.fill(0.0);
            let partition_count = planner.partitions().len();
            for partition_index in 0..partition_count {
                let partition = planner.partitions()[partition_index];
                let point_helicities = partition
                    .key
                    .helicity_index
                    .map(|index| &helicity_singletons[index]);
                let point_colors = partition
                    .key
                    .color_index
                    .map(|index| &color_singletons[index]);
                let effective_helicities = point_helicities.or(selected_helicities);
                let effective_colors = point_colors.or(selected_colors);
                match partition.rows {
                    PointSelectorRows::Contiguous { start, end } => {
                        self.run_selected_f64_batch_into(
                            batch.subview(start, end)?,
                            effective_helicities,
                            effective_colors,
                            &mut output_totals[start..end],
                        )?;
                    }
                    rows @ PointSelectorRows::Gathered { .. } => {
                        let point_indices = planner.gathered_rows(rows);
                        let gathered_batch =
                            fill_gathered_batch_from_view(gathered_batch, batch, point_indices)?;
                        partition_totals.resize(partition.rows.len(), 0.0);
                        partition_totals.fill(0.0);
                        self.run_selected_f64_batch_into(
                            F64MomentumBatchView::from_nested(
                                gathered_batch,
                                self.runtime.external_count,
                            )?,
                            effective_helicities,
                            effective_colors,
                            partition_totals,
                        )?;
                        scatter_partition_totals(
                            output_totals,
                            partition_totals,
                            partition.rows,
                            planner,
                        );
                    }
                }
            }
            output.copy_from_slice(output_totals);
            Ok(())
        })();
        self.point_selector_scratch = selector_scratch;
        self.input_crossing_map = crossing_lookup;
        result
    }

    /// Route ordinary public evaluation through the compact on-the-fly
    /// selector domain.  This must run before any dense `PhysicsRuntime`
    /// access: on-the-fly artifacts intentionally leave that metadata lazy.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate_on_the_fly_f64_into_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let mut selector_scratch = std::mem::take(&mut self.point_selector_scratch);
        let result = (|| {
            let PointSelectorExecutionScratch {
                planner,
                gathered_batch,
                partition_totals,
                output_totals,
                helicity_selector_sets,
                color_selector_sets,
                ..
            } = &mut selector_scratch;
            let selected_helicities = helicity_selector_sets.resolve(helicity_ids, "helicity")?;
            let selected_colors = color_selector_sets.resolve(color_ids, "color component")?;
            if helicity_by_point.is_none() && color_by_point.is_none() {
                let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                    momenta,
                    point_count,
                    self.runtime.external_count,
                    crossing_lookup.as_deref(),
                )?;
                return self.run_selected_f64_batch_into(
                    batch,
                    selected_helicities,
                    selected_colors,
                    output,
                );
            }
            // Per-point routing retains its ordinal planner.  Fixed global
            // selection is handled above so its IDs are parsed only by the
            // lane on a cold family and matched allocation-free once warm.
            let (selected_helicity_ordinals, selected_color_ordinals) =
                self.on_the_fly_selected_ordinals(selected_helicities, selected_colors)?;
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;

            let (helicity_count, color_count) = self.on_the_fly_selector_counts()?;
            let plan = planner.build(
                point_count,
                helicity_by_point,
                color_by_point,
                helicity_count,
                color_count,
            )?;
            self.record_resolved_warnings(helicity_ids, color_ids)?;
            output_totals.resize(point_count, 0.0);
            output_totals.fill(0.0);
            if let PointSelectorPlan::Homogeneous(key) = plan {
                let point_helicity = key.helicity_index.map(|index| [index]);
                let point_color = key.color_index.map(|index| [index]);
                self.run_on_the_fly_f64_batch_into_by_ordinals(
                    batch,
                    point_helicity
                        .as_ref()
                        .map(<[usize; 1]>::as_slice)
                        .or(selected_helicity_ordinals.as_deref()),
                    point_color
                        .as_ref()
                        .map(<[usize; 1]>::as_slice)
                        .or(selected_color_ordinals.as_deref()),
                    output_totals,
                )?;
                output.copy_from_slice(output_totals);
                return Ok(());
            }

            let partition_count = planner.partitions().len();
            for partition_index in 0..partition_count {
                let partition = planner.partitions()[partition_index];
                let point_helicity = partition.key.helicity_index.map(|index| [index]);
                let point_color = partition.key.color_index.map(|index| [index]);
                let effective_helicities = point_helicity
                    .as_ref()
                    .map(<[usize; 1]>::as_slice)
                    .or(selected_helicity_ordinals.as_deref());
                let effective_colors = point_color
                    .as_ref()
                    .map(<[usize; 1]>::as_slice)
                    .or(selected_color_ordinals.as_deref());
                match partition.rows {
                    PointSelectorRows::Contiguous { start, end } => {
                        self.run_on_the_fly_f64_batch_into_by_ordinals(
                            batch.subview(start, end)?,
                            effective_helicities,
                            effective_colors,
                            &mut output_totals[start..end],
                        )?;
                    }
                    rows @ PointSelectorRows::Gathered { .. } => {
                        let point_indices = planner.gathered_rows(rows);
                        let gathered_batch =
                            fill_gathered_batch_from_view(gathered_batch, batch, point_indices)?;
                        partition_totals.resize(partition.rows.len(), 0.0);
                        partition_totals.fill(0.0);
                        self.run_on_the_fly_f64_batch_into_by_ordinals(
                            F64MomentumBatchView::from_nested(
                                gathered_batch,
                                self.runtime.external_count,
                            )?,
                            effective_helicities,
                            effective_colors,
                            partition_totals,
                        )?;
                        scatter_partition_totals(
                            output_totals,
                            partition_totals,
                            partition.rows,
                            planner,
                        );
                    }
                }
            }
            output.copy_from_slice(output_totals);
            Ok(())
        })();
        self.point_selector_scratch = selector_scratch;
        self.input_crossing_map = crossing_lookup;
        result
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn on_the_fly_selector_counts(&self) -> RusticolResult<(usize, usize)> {
        let NativeExecutionLane::OnTheFly(runtime) = &self.execution_lane else {
            return Err(RusticolError::internal(
                "compact selector counts require an on-the-fly execution lane",
            ));
        };
        runtime.public_selector_counts()
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn on_the_fly_selected_ordinals(
        &self,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<OnTheFlySelectedOrdinalsV1> {
        let NativeExecutionLane::OnTheFly(runtime) = &self.execution_lane else {
            return Err(RusticolError::internal(
                "compact selector ordinals require an on-the-fly execution lane",
            ));
        };
        if runtime.uses_contracted_color() {
            if selected_colors.is_some() {
                return Err(RusticolError::selector(
                    "on-the-fly contracted color does not expose color selector ordinals",
                ));
            }
            let (helicity_ordinals, _) = runtime
                .selectors
                .selected_ordinals(selected_helicities, None)?;
            Ok((helicity_ordinals, None))
        } else {
            runtime
                .selectors
                .selected_ordinals(selected_helicities, selected_colors)
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn run_on_the_fly_f64_batch_into_by_ordinals(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        helicity_ordinals: Option<&[usize]>,
        color_ordinals: Option<&[usize]>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let NativeExecutionLane::OnTheFly(runtime) = &mut self.execution_lane else {
            return Err(RusticolError::internal(
                "compact selector ordinals require an on-the-fly execution lane",
            ));
        };
        runtime.run_total_into_by_ordinals_unprofiled(
            &self.runtime,
            batch,
            helicity_ordinals,
            color_ordinals,
            output,
        )
    }

    fn run_f64_batch_into(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        match &mut self.execution_lane {
            NativeExecutionLane::Compiled => self.runtime.run_f64_into_unprofiled(batch, output),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => {
                runtime.run_f64_view_into_unprofiled(&mut self.runtime, batch, output)
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => {
                runtime.run_f64_view_into_unprofiled(&mut self.runtime, batch, None, None, output)
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::OnTheFly(runtime) => {
                runtime.run_total_into_unprofiled(&self.runtime, batch, None, None, output)
            }
        }
    }

    fn run_selected_f64_batch_into(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        match &mut self.execution_lane {
            NativeExecutionLane::Compiled => self.runtime.run_f64_selected_into_unprofiled(
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => runtime.run_f64_view_selected_into_unprofiled(
                &mut self.runtime,
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => runtime.run_f64_view_into_unprofiled(
                &mut self.runtime,
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::OnTheFly(runtime) => runtime.run_total_into_unprofiled(
                &self.runtime,
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
        }
    }

    fn run_resolved_f64_batch(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> Result<ResolvedValues<f64>, RusticolError> {
        match &mut self.execution_lane {
            NativeExecutionLane::Compiled => self.runtime.run_resolved_f64_unprofiled(
                batch,
                selected_helicities,
                selected_colors,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => runtime.run_resolved_f64_view_unprofiled(
                &mut self.runtime,
                batch,
                selected_helicities,
                selected_colors,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => {
                let nested = batch.materialize_nested();
                runtime
                    .run_resolved_f64(
                        &mut self.runtime,
                        &nested,
                        selected_helicities,
                        selected_colors,
                    )
                    .map(|(resolved, _profile)| resolved)
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::OnTheFly(runtime) => runtime
                .run_resolved(&self.runtime, batch, selected_helicities, selected_colors)
                .map(|(resolved, _profile)| resolved),
        }
    }

    fn run_selected_f64_batch_profile(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> Result<(Vec<f64>, RuntimeProfile), RusticolError> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if matches!(&self.execution_lane, NativeExecutionLane::Compiled) {
            let before = compiled_direct_profile_snapshot(&self.runtime)?;
            if before.engine_count != 0 {
                let total_start = Instant::now();
                let mut values = vec![0.0; batch.len()];
                self.runtime.run_f64_selected_into_unprofiled(
                    F64MomentumBatchView::from_nested(batch, self.runtime.external_count)?,
                    selected_helicities,
                    selected_colors,
                    &mut values,
                )?;
                let total_s = total_start.elapsed().as_secs_f64();
                let after = compiled_direct_profile_snapshot(&self.runtime)?;
                let f64_bytes = std::mem::size_of::<f64>() as u64;
                let complex_bytes = 2 * f64_bytes;
                let mut profile = RuntimeProfile {
                    // The direct hot path is deliberately uninstrumented.
                    // Until its three coarse phase clocks are added, keep
                    // the complete measured envelope in orchestration so
                    // top-level accounting remains exact rather than
                    // manufacturing dense-stage timing fields.
                    orchestration_s: total_s,
                    total_s,
                    source_component_count: after
                        .source_fill_bytes
                        .saturating_sub(before.source_fill_bytes)
                        / complex_bytes,
                    momentum_component_count: after
                        .momentum_fill_bytes
                        .saturating_sub(before.momentum_fill_bytes)
                        / f64_bytes,
                    model_parameter_component_count: after
                        .parameter_fill_bytes
                        .saturating_sub(before.parameter_fill_bytes)
                        / complex_bytes,
                    state_clear_component_count: after
                        .amplitude_clear_bytes
                        .saturating_sub(before.amplitude_clear_bytes)
                        / complex_bytes,
                    evaluator_backend_call_count: after
                        .backend_call_count
                        .saturating_sub(before.backend_call_count),
                    compiled_direct_arena_engine_count: after.engine_count as u64,
                    compiled_direct_arena_call_count: after
                        .backend_call_count
                        .saturating_sub(before.backend_call_count),
                    compiled_direct_arena_boundary_input_bytes: after.boundary_input_bytes,
                    compiled_direct_arena_boundary_current_output_bytes: after
                        .boundary_current_output_bytes,
                    compiled_direct_arena_boundary_amplitude_output_bytes: after
                        .boundary_amplitude_output_bytes,
                    compiled_direct_arena_internal_broadcast_bytes: after
                        .scalar_broadcast_fill_bytes
                        .saturating_sub(before.scalar_broadcast_fill_bytes),
                    total_materialized_value_count: batch.len() as u64,
                    ..RuntimeProfile::default()
                };
                attach_compiled_direct_configuration(&mut profile, after);
                return Ok((values, profile));
            }
        }
        match &mut self.execution_lane {
            NativeExecutionLane::Compiled => {
                self.runtime
                    .run_f64_selected_totals(batch, selected_helicities, selected_colors)
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => {
                let (resolved, mut profile) = runtime.run_resolved_f64_profile(
                    &mut self.runtime,
                    batch,
                    selected_helicities,
                    selected_colors,
                )?;
                let materialization_start = Instant::now();
                let values = resolved_f64_totals(&resolved)?;
                profile.total_materialization_s += materialization_start.elapsed().as_secs_f64();
                profile.total_materialized_value_count += values.len() as u64;
                Ok((values, profile))
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => {
                let (resolved, mut profile) = runtime.run_resolved_f64(
                    &mut self.runtime,
                    batch,
                    selected_helicities,
                    selected_colors,
                )?;
                let materialization_start = Instant::now();
                let values = resolved_f64_totals(&resolved)?;
                profile.total_materialization_s += materialization_start.elapsed().as_secs_f64();
                profile.total_materialized_value_count += values.len() as u64;
                Ok((values, profile))
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::OnTheFly(runtime) => {
                let view = F64MomentumBatchView::from_nested(batch, self.runtime.external_count)?;
                let mut values = vec![0.0; batch.len()];
                let mut profile = runtime.run_total_into(
                    &self.runtime,
                    view,
                    selected_helicities,
                    selected_colors,
                    &mut values,
                )?;
                profile.total_materialized_value_count += values.len() as u64;
                Ok((values, profile))
            }
        }
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn run_on_the_fly_f64_batch_profile_by_ordinals(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        helicity_ordinals: Option<&[usize]>,
        color_ordinals: Option<&[usize]>,
    ) -> Result<(Vec<f64>, RuntimeProfile), RusticolError> {
        let view = F64MomentumBatchView::from_nested(batch, self.runtime.external_count)?;
        let mut values = vec![0.0; batch.len()];
        let NativeExecutionLane::OnTheFly(runtime) = &mut self.execution_lane else {
            return Err(RusticolError::internal(
                "compact selector ordinals require an on-the-fly execution lane",
            ));
        };
        let mut profile = runtime.run_total_into_by_ordinals(
            &self.runtime,
            view,
            helicity_ordinals,
            color_ordinals,
            &mut values,
        )?;
        profile.total_materialized_value_count += values.len() as u64;
        Ok((values, profile))
    }

    pub fn benchmark_f64_wall_time(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<f64, RusticolError> {
        self.benchmark_f64_wall_time_with_selectors(
            momenta,
            point_count,
            repetitions,
            helicity_ids,
            color_ids,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_f64_wall_time_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<f64, RusticolError> {
        if repetitions == 0 {
            return Err(RusticolError::invalid_argument(
                "benchmark repetitions must be positive",
            ));
        }
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let mut output = vec![0.0; point_count];
        let started = Instant::now();
        for _ in 0..repetitions {
            self.evaluate_f64_into_with_selectors(
                momenta,
                point_count,
                helicity_ids,
                color_ids,
                helicity_by_point,
                color_by_point,
                &mut output,
            )?;
            std::hint::black_box(&output);
        }
        Ok(started.elapsed().as_secs_f64())
    }

    /// Profile the warmed Direct-Arena timing boundary used by
    /// [`Self::benchmark_f64_wall_time`] for batch-global selectors.
    ///
    /// The borrowed flat input and the caller-sized output are prepared once
    /// outside the repeated region. Every measured repetition then uses the
    /// allocation-free `evaluate_f64_into_with_selectors` path. This is kept
    /// separate from the public diagnostic profiler below, whose materialized
    /// input and owned result remain useful when inspecting legacy lanes.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_f64_arena_profile_repeated(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        if repetitions == 0 {
            return Err(RusticolError::invalid_argument(
                "arena profile repetitions must be positive",
            ));
        }
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let compiled_lane = match &self.execution_lane {
            NativeExecutionLane::Compiled => {
                let snapshot = compiled_direct_profile_snapshot(&self.runtime)?;
                if snapshot.engine_count == 0 {
                    return Err(RusticolError::compatibility(
                        "warmed Arena profiling requires a compiled Direct-Arena artifact",
                    ));
                }
                true
            }
            NativeExecutionLane::Eager(_) => false,
            NativeExecutionLane::Recurrence(_) => {
                return Err(RusticolError::compatibility(
                    "warmed Arena profiling is available only for eager and compiled execution",
                ));
            }
            NativeExecutionLane::OnTheFly(_) => false,
        };
        let measured_points = point_count.checked_mul(repetitions).ok_or_else(|| {
            RusticolError::invalid_argument("arena profile point count overflowed")
        })?;
        let measured_input_components =
            momenta.len().checked_mul(repetitions).ok_or_else(|| {
                RusticolError::invalid_argument(
                    "arena profile native input component count overflowed",
                )
            })?;
        let measured_points_u64 = u64::try_from(measured_points).map_err(|_| {
            RusticolError::invalid_argument("arena profile point count does not fit in u64")
        })?;
        let measured_input_components_u64 =
            u64::try_from(measured_input_components).map_err(|_| {
                RusticolError::invalid_argument(
                    "arena profile native input component count does not fit in u64",
                )
            })?;
        let repetitions_u64 = u64::try_from(repetitions).map_err(|_| {
            RusticolError::invalid_argument("arena profile repetition count does not fit in u64")
        })?;

        // The result allocation is deliberately outside the measured repeated
        // boundary, matching the native headline timer. Prime the exact
        // selector and workspace route once before the counter snapshot so the
        // measured region is independently warmed even when this private
        // operation is called directly.
        let mut values = vec![0.0; point_count];
        self.evaluate_f64_into_with_selectors(
            momenta,
            point_count,
            helicity_ids,
            color_ids,
            None,
            None,
            &mut values,
        )?;
        let compiled_before = if compiled_lane {
            Some(compiled_direct_profile_snapshot(&self.runtime)?)
        } else {
            None
        };
        let started = Instant::now();
        for _ in 0..repetitions {
            std::hint::black_box(&mut values);
            self.evaluate_f64_into_with_selectors(
                momenta,
                point_count,
                helicity_ids,
                color_ids,
                None,
                None,
                &mut values,
            )?;
            std::hint::black_box(&values);
        }
        let total_s = started.elapsed().as_secs_f64();
        let mut runtime_profile = RuntimeProfile {
            // The warmed Direct-Arena boundary is deliberately free of phase
            // clocks. Keep its complete measured envelope in orchestration.
            // Callers must treat leaf-evaluator phase timing as unavailable,
            // rather than interpreting the zero-valued phase fields as
            // measurements below clock resolution.
            orchestration_s: total_s,
            total_s,
            total_materialized_value_count: measured_points_u64,
            ..RuntimeProfile::default()
        };
        if let Some(before) = compiled_before {
            let after = compiled_direct_profile_snapshot(&self.runtime)?;
            let f64_bytes = std::mem::size_of::<f64>() as u64;
            let complex_bytes = 2 * f64_bytes;
            runtime_profile.source_component_count = after
                .source_fill_bytes
                .saturating_sub(before.source_fill_bytes)
                / complex_bytes;
            runtime_profile.momentum_component_count = after
                .momentum_fill_bytes
                .saturating_sub(before.momentum_fill_bytes)
                / f64_bytes;
            runtime_profile.model_parameter_component_count = after
                .parameter_fill_bytes
                .saturating_sub(before.parameter_fill_bytes)
                / complex_bytes;
            runtime_profile.state_clear_component_count = after
                .amplitude_clear_bytes
                .saturating_sub(before.amplitude_clear_bytes)
                / complex_bytes;
            runtime_profile.evaluator_backend_call_count = after
                .backend_call_count
                .saturating_sub(before.backend_call_count);
            runtime_profile.compiled_direct_arena_engine_count = u64::try_from(before.engine_count)
                .ok()
                .and_then(|count| count.checked_mul(repetitions_u64))
                .ok_or_else(|| {
                    RusticolError::invalid_argument("compiled Direct-Arena engine count overflowed")
                })?;
            runtime_profile.compiled_direct_arena_call_count =
                runtime_profile.evaluator_backend_call_count;
            runtime_profile.compiled_direct_arena_boundary_input_bytes = after
                .boundary_input_bytes
                .saturating_sub(before.boundary_input_bytes);
            runtime_profile.compiled_direct_arena_boundary_current_output_bytes = after
                .boundary_current_output_bytes
                .saturating_sub(before.boundary_current_output_bytes);
            runtime_profile.compiled_direct_arena_boundary_amplitude_output_bytes = after
                .boundary_amplitude_output_bytes
                .saturating_sub(before.boundary_amplitude_output_bytes);
            runtime_profile.compiled_direct_arena_internal_broadcast_bytes = after
                .scalar_broadcast_fill_bytes
                .saturating_sub(before.scalar_broadcast_fill_bytes);
            attach_compiled_direct_configuration(&mut runtime_profile, after);
        }
        runtime_profile.validate_recurrence_direct_boundary_traffic()?;
        let mut profile: NativeRuntimeProfile = runtime_profile.into();
        profile.native_input_component_count = measured_input_components_u64;
        profile.native_input_pack_bytes = 0;
        profile.native_input_crossing_bytes = 0;
        profile.native_input_container_allocation_count = 0;
        profile.native_output_allocation_count = 0;
        self.validate_profile_accounting(&profile)?;
        Ok(NativeProfiledEvaluation { values, profile })
    }

    #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
    pub fn evaluate_f64_arena_profile_repeated(
        &mut self,
        _momenta: &[f64],
        _point_count: usize,
        _repetitions: usize,
        _helicity_ids: Option<&[String]>,
        _color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        Err(RusticolError::compatibility(
            "warmed Arena profiling requires the f64-compiled or f64-symjit feature",
        ))
    }

    pub fn evaluate_f64_profile(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        let color_ids = self.execution_color_ids(color_ids);
        let mut evaluation = self.evaluate_representative_f64_profile(
            momenta,
            point_count,
            helicity_ids,
            color_ids.as_deref(),
        )?;
        self.apply_process_group_factor(&mut evaluation.values);
        Ok(evaluation)
    }

    fn evaluate_representative_f64_profile(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        let total_start = Instant::now();
        let (batch, native_input_pack_elapsed, native_input_crossing_elapsed) =
            self.prepare_f64_batch_profile(momenta, point_count)?;
        let compiled_direct_profile = {
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            {
                matches!(&self.execution_lane, NativeExecutionLane::Compiled)
                    && compiled_direct_profile_snapshot(&self.runtime)?.engine_count != 0
            }
            #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
            {
                false
            }
        };
        let (values, profile) = if compiled_direct_profile {
            if helicity_ids.is_some() || color_ids.is_some() {
                self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
                self.record_resolved_warnings(helicity_ids, color_ids)?;
            }
            let selected_helicities = selector_set(helicity_ids, "helicity")?;
            let selected_colors = selector_set(color_ids, "color component")?;
            self.run_selected_f64_batch_profile(
                &batch,
                selected_helicities.as_ref(),
                selected_colors.as_ref(),
            )?
        } else if helicity_ids.is_some() || color_ids.is_some() {
            self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
            self.record_resolved_warnings(helicity_ids, color_ids)?;
            let selected_helicities = selector_set(helicity_ids, "helicity")?;
            let selected_colors = selector_set(color_ids, "color component")?;
            let (values, profile) = match &mut self.execution_lane {
                NativeExecutionLane::Compiled => self.runtime.run_f64_selected_totals(
                    &batch,
                    selected_helicities.as_ref(),
                    selected_colors.as_ref(),
                )?,
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Eager(runtime) => {
                    let (resolved, mut profile) = runtime.run_resolved_f64_profile(
                        &mut self.runtime,
                        &batch,
                        selected_helicities.as_ref(),
                        selected_colors.as_ref(),
                    )?;
                    let materialization_start = Instant::now();
                    let values = resolved_f64_totals(&resolved)?;
                    profile.total_materialization_s +=
                        materialization_start.elapsed().as_secs_f64();
                    profile.total_materialized_value_count += point_count as u64;
                    (values, profile)
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Recurrence(runtime) => {
                    let (resolved, mut profile) = runtime.run_resolved_f64(
                        &mut self.runtime,
                        &batch,
                        selected_helicities.as_ref(),
                        selected_colors.as_ref(),
                    )?;
                    let materialization_start = Instant::now();
                    let values = resolved_f64_totals(&resolved)?;
                    profile.total_materialization_s +=
                        materialization_start.elapsed().as_secs_f64();
                    profile.total_materialized_value_count += point_count as u64;
                    (values, profile)
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::OnTheFly(runtime) => {
                    let view =
                        F64MomentumBatchView::from_nested(&batch, self.runtime.external_count)?;
                    let mut values = vec![0.0; point_count];
                    let mut profile = runtime.run_total_into(
                        &self.runtime,
                        view,
                        selected_helicities.as_ref(),
                        selected_colors.as_ref(),
                        &mut values,
                    )?;
                    profile.total_materialized_value_count += point_count as u64;
                    (values, profile)
                }
            };
            (values, profile)
        } else {
            match &mut self.execution_lane {
                NativeExecutionLane::Compiled => self.runtime.run_f64(&batch)?,
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Eager(runtime) => {
                    runtime.run_f64_profile(&mut self.runtime, &batch)?
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Recurrence(runtime) => {
                    runtime.run_f64(&mut self.runtime, &batch)?
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::OnTheFly(runtime) => {
                    let view =
                        F64MomentumBatchView::from_nested(&batch, self.runtime.external_count)?;
                    let mut values = vec![0.0; point_count];
                    let profile =
                        runtime.run_total_into(&self.runtime, view, None, None, &mut values)?;
                    (values, profile)
                }
            }
        };
        profile.validate_recurrence_direct_boundary_traffic()?;
        let mut profile: NativeRuntimeProfile = profile.into();
        profile.native_input_pack_s = profile_duration_seconds(native_input_pack_elapsed);
        profile.native_input_crossing_s = profile_duration_seconds(native_input_crossing_elapsed);
        profile.native_input_component_count = momenta.len() as u64;
        profile.native_input_pack_bytes = std::mem::size_of_val(momenta) as u64;
        if self.input_crossing_map.is_some() {
            profile.native_input_crossing_bytes = profile.native_input_pack_bytes;
        }
        profile.native_input_container_allocation_count =
            (point_count + 1 + usize::from(self.input_crossing_map.is_some()) * (point_count + 2))
                as u64;
        profile.native_output_allocation_count = 1;
        profile.total_s = total_start.elapsed().as_secs_f64();
        self.validate_profile_accounting(&profile)?;
        Ok(NativeProfiledEvaluation { values, profile })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_f64_profile_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        let color_ids = self.execution_color_ids(color_ids);
        let mut evaluation = self.evaluate_representative_f64_profile_with_selectors(
            momenta,
            point_count,
            helicity_ids,
            color_ids.as_deref(),
            helicity_by_point,
            color_by_point,
        )?;
        self.apply_process_group_factor(&mut evaluation.values);
        Ok(evaluation)
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_representative_f64_profile_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        if helicity_by_point.is_none() && color_by_point.is_none() {
            return self.evaluate_representative_f64_profile(
                momenta,
                point_count,
                helicity_ids,
                color_ids,
            );
        }
        if helicity_ids.is_some() && helicity_by_point.is_some() {
            return Err(RusticolError::selector(
                "helicities and helicity_by_point are mutually exclusive",
            ));
        }
        if color_ids.is_some() && color_by_point.is_some() {
            return Err(RusticolError::selector(
                "color_flows and color_flow_by_point are mutually exclusive",
            ));
        }
        self.validate_selector_capabilities(
            helicity_ids.is_some() || helicity_by_point.is_some(),
            color_ids.is_some() || color_by_point.is_some(),
        )?;
        let total_start = Instant::now();
        let selected_helicities = selector_set(helicity_ids, "helicity")?;
        let selected_colors = selector_set(color_ids, "color component")?;
        let (batch, native_input_pack_elapsed, native_input_crossing_elapsed) =
            self.prepare_f64_batch_profile(momenta, point_count)?;
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        let on_the_fly = matches!(&self.execution_lane, NativeExecutionLane::OnTheFly(_));
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        let on_the_fly = false;
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        let on_the_fly_selected_ordinals = if on_the_fly {
            Some(self.on_the_fly_selected_ordinals(
                selected_helicities.as_ref(),
                selected_colors.as_ref(),
            )?)
        } else {
            None
        };
        let physics = if on_the_fly {
            None
        } else {
            Some(self.runtime.physics.clone().ok_or_else(|| {
                RusticolError::artifact(
                    "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
                )
            })?)
        };
        let (helicity_count, color_count) = if on_the_fly {
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            {
                self.on_the_fly_selector_counts()?
            }
            #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
            {
                unreachable!("on-the-fly lane requires an f64 evaluator feature")
            }
        } else {
            let physics = physics
                .as_ref()
                .expect("non-on-the-fly profile has physics metadata");
            (
                physics.manifest.helicities.len(),
                physics.manifest.color_components.len(),
            )
        };
        let selector_simd_lane_width = self.selector_simd_lane_width();
        let mut selector_scratch = std::mem::take(&mut self.point_selector_scratch);
        let result = (|| {
            let planner_started = Instant::now();
            let plan = selector_scratch.planner.build(
                point_count,
                helicity_by_point,
                color_by_point,
                helicity_count,
                color_count,
            )?;
            let plan_profile =
                selector_scratch
                    .planner
                    .profile(plan, point_count, selector_simd_lane_width);
            let planner = planner_started.elapsed();
            self.record_resolved_warnings(helicity_ids, color_ids)?;

            if let PointSelectorPlan::Homogeneous(key) = plan {
                let (values, runtime_profile) = if on_the_fly {
                    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                    {
                        let (selected_helicities, selected_colors) = on_the_fly_selected_ordinals
                            .as_ref()
                            .expect("on-the-fly selected ordinals are absent");
                        let point_helicity = key.helicity_index.map(|index| [index]);
                        let point_color = key.color_index.map(|index| [index]);
                        self.run_on_the_fly_f64_batch_profile_by_ordinals(
                            &batch,
                            point_helicity
                                .as_ref()
                                .map(<[usize; 1]>::as_slice)
                                .or(selected_helicities.as_deref()),
                            point_color
                                .as_ref()
                                .map(<[usize; 1]>::as_slice)
                                .or(selected_colors.as_deref()),
                        )?
                    }
                    #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
                    {
                        unreachable!("on-the-fly lane requires an f64 evaluator feature")
                    }
                } else {
                    let physics = physics
                        .as_ref()
                        .expect("non-on-the-fly profile has physics metadata");
                    let point_helicities = key.helicity_index.map(|index| {
                        BTreeSet::from([physics.manifest.helicities[index].id.clone()])
                    });
                    let point_colors = key.color_index.map(|index| {
                        BTreeSet::from([physics.manifest.color_components[index].id().to_string()])
                    });
                    let effective_helicities =
                        point_helicities.as_ref().or(selected_helicities.as_ref());
                    let effective_colors = point_colors.as_ref().or(selected_colors.as_ref());
                    self.run_selected_f64_batch_profile(
                        &batch,
                        effective_helicities,
                        effective_colors,
                    )?
                };
                runtime_profile.validate_recurrence_direct_boundary_traffic()?;
                let mut profile: NativeRuntimeProfile = runtime_profile.into();
                attach_point_selector_profile(
                    &mut profile,
                    &plan_profile,
                    planner,
                    Duration::ZERO,
                    Duration::ZERO,
                    PointSelectorProfileCounts {
                        gather_point_count: 0,
                        input_bytes_per_point: self.runtime.external_count
                            * 4
                            * std::mem::size_of::<f64>(),
                        scatter_value_count: 0,
                    },
                );
                profile.native_input_pack_s = profile_duration_seconds(native_input_pack_elapsed);
                profile.native_input_crossing_s =
                    profile_duration_seconds(native_input_crossing_elapsed);
                profile.native_input_component_count = momenta.len() as u64;
                profile.native_input_pack_bytes = std::mem::size_of_val(momenta) as u64;
                if self.input_crossing_map.is_some() {
                    profile.native_input_crossing_bytes = profile.native_input_pack_bytes;
                }
                profile.native_input_container_allocation_count = (point_count
                    + 1
                    + usize::from(self.input_crossing_map.is_some()) * (point_count + 2))
                    as u64;
                profile.native_output_allocation_count = 1;
                profile.total_s = total_start.elapsed().as_secs_f64();
                self.validate_profile_accounting(&profile)?;
                return Ok(NativeProfiledEvaluation { values, profile });
            }

            let mut values = vec![0.0; point_count];
            let mut partition_profiles = Vec::new();
            let mut gather = Duration::ZERO;
            let mut scatter = Duration::ZERO;
            let mut gather_point_count = 0usize;
            let partition_count = selector_scratch.planner.partitions().len();
            for partition_index in 0..partition_count {
                let partition = selector_scratch.planner.partitions()[partition_index];
                let (partition_totals, partition_profile) = if on_the_fly {
                    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                    {
                        let (selected_helicities, selected_colors) = on_the_fly_selected_ordinals
                            .as_ref()
                            .expect("on-the-fly selected ordinals are absent");
                        let point_helicity = partition.key.helicity_index.map(|index| [index]);
                        let point_color = partition.key.color_index.map(|index| [index]);
                        let effective_helicities = point_helicity
                            .as_ref()
                            .map(<[usize; 1]>::as_slice)
                            .or(selected_helicities.as_deref());
                        let effective_colors = point_color
                            .as_ref()
                            .map(<[usize; 1]>::as_slice)
                            .or(selected_colors.as_deref());
                        match partition.rows {
                            PointSelectorRows::Contiguous { start, end } => self
                                .run_on_the_fly_f64_batch_profile_by_ordinals(
                                    &batch[start..end],
                                    effective_helicities,
                                    effective_colors,
                                )?,
                            rows @ PointSelectorRows::Gathered { .. } => {
                                let gather_started = Instant::now();
                                let point_indices = selector_scratch.planner.gathered_rows(rows);
                                gather_point_count += point_indices.len();
                                let gathered_batch = fill_gathered_batch(
                                    &mut selector_scratch.gathered_batch,
                                    &batch,
                                    point_indices,
                                );
                                gather += gather_started.elapsed();
                                self.run_on_the_fly_f64_batch_profile_by_ordinals(
                                    gathered_batch,
                                    effective_helicities,
                                    effective_colors,
                                )?
                            }
                        }
                    }
                    #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
                    {
                        unreachable!("on-the-fly lane requires an f64 evaluator feature")
                    }
                } else {
                    let physics = physics
                        .as_ref()
                        .expect("non-on-the-fly profile has physics metadata");
                    let point_helicities = partition.key.helicity_index.map(|index| {
                        BTreeSet::from([physics.manifest.helicities[index].id.clone()])
                    });
                    let point_colors = partition.key.color_index.map(|index| {
                        BTreeSet::from([physics.manifest.color_components[index].id().to_string()])
                    });
                    let effective_helicities =
                        point_helicities.as_ref().or(selected_helicities.as_ref());
                    let effective_colors = point_colors.as_ref().or(selected_colors.as_ref());
                    match partition.rows {
                        PointSelectorRows::Contiguous { start, end } => self
                            .run_selected_f64_batch_profile(
                                &batch[start..end],
                                effective_helicities,
                                effective_colors,
                            )?,
                        rows @ PointSelectorRows::Gathered { .. } => {
                            let gather_started = Instant::now();
                            let point_indices = selector_scratch.planner.gathered_rows(rows);
                            gather_point_count += point_indices.len();
                            let gathered_batch = fill_gathered_batch(
                                &mut selector_scratch.gathered_batch,
                                &batch,
                                point_indices,
                            );
                            gather += gather_started.elapsed();
                            self.run_selected_f64_batch_profile(
                                gathered_batch,
                                effective_helicities,
                                effective_colors,
                            )?
                        }
                    }
                };
                if partition_totals.len() != partition.rows.len() {
                    return Err(RusticolError::integrity(
                        "per-point selector partition returned the wrong number of values",
                    ));
                }
                let scatter_started = Instant::now();
                scatter_partition_totals(
                    &mut values,
                    &partition_totals,
                    partition.rows,
                    &selector_scratch.planner,
                );
                scatter += scatter_started.elapsed();
                partition_profiles.push(partition_profile);
            }
            let mut runtime_profile = RuntimeProfile::default();
            for partition_profile in &partition_profiles {
                runtime_profile.add_sector(partition_profile);
            }
            runtime_profile.validate_recurrence_direct_boundary_traffic()?;
            let mut profile: NativeRuntimeProfile = runtime_profile.into();
            attach_point_selector_profile(
                &mut profile,
                &plan_profile,
                planner,
                gather,
                scatter,
                PointSelectorProfileCounts {
                    gather_point_count,
                    input_bytes_per_point: self.runtime.external_count
                        * 4
                        * std::mem::size_of::<f64>(),
                    scatter_value_count: point_count,
                },
            );
            profile.native_input_pack_s = profile_duration_seconds(native_input_pack_elapsed);
            profile.native_input_crossing_s =
                profile_duration_seconds(native_input_crossing_elapsed);
            profile.native_input_component_count = momenta.len() as u64;
            profile.native_input_pack_bytes = std::mem::size_of_val(momenta) as u64;
            if self.input_crossing_map.is_some() {
                profile.native_input_crossing_bytes = profile.native_input_pack_bytes;
            }
            profile.native_input_container_allocation_count = (point_count
                + 1
                + usize::from(self.input_crossing_map.is_some()) * (point_count + 2))
                as u64;
            profile.native_output_allocation_count = 1;
            profile.total_s = total_start.elapsed().as_secs_f64();
            self.validate_profile_accounting(&profile)?;
            Ok(NativeProfiledEvaluation { values, profile })
        })();
        self.point_selector_scratch = selector_scratch;
        result
    }

    pub fn evaluate_f64_profile_repeated(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        self.evaluate_f64_profile_repeated_with_selectors(
            momenta,
            point_count,
            repetitions,
            helicity_ids,
            color_ids,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_f64_profile_repeated_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        if repetitions == 0 {
            return Err(RusticolError::invalid_argument(
                "profile repetitions must be positive",
            ));
        }
        let started = Instant::now();
        let mut profile: Option<Box<NativeRuntimeProfile>> = None;
        let mut values = Vec::new();
        for _ in 0..repetitions {
            // The saved evaluator payload may use the complete AArch64
            // floating-point register file. Keep the aggregate behind an
            // opaque heap boundary so no accumulated f64 survives the next
            // generated call in a register.
            std::hint::black_box(&mut profile);
            let profiled = self.evaluate_f64_profile_with_selectors(
                momenta,
                point_count,
                helicity_ids,
                color_ids,
                helicity_by_point,
                color_by_point,
            )?;
            std::hint::black_box(&profiled.values);
            values = profiled.values;
            if let Some(aggregate) = profile.as_deref_mut() {
                aggregate.accumulate(&profiled.profile);
            } else {
                profile = Some(Box::new(profiled.profile));
            }
        }
        let mut profile = *profile.expect("positive repetitions checked");
        profile.total_s = started.elapsed().as_secs_f64();
        self.validate_profile_accounting(&profile)?;
        Ok(NativeProfiledEvaluation { values, profile })
    }

    fn validate_profile_accounting(&self, profile: &NativeRuntimeProfile) -> RusticolResult<()> {
        match &self.execution_lane {
            NativeExecutionLane::Compiled => profile.validate_compiled_top_level_accounting(),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(_) => profile.validate_eager_top_level_accounting(),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(_) => {
                profile.validate_recurrence_top_level_accounting()
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::OnTheFly(_) => profile.validate_recurrence_top_level_accounting(),
        }
    }

    pub fn evaluate_resolved_f64(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeResolvedEvaluation, RusticolError> {
        let color_ids = self.execution_color_ids(color_ids);
        let mut resolved = self.evaluate_representative_resolved_f64(
            momenta,
            point_count,
            helicity_ids,
            color_ids.as_deref(),
        )?;
        self.apply_process_group_factor(&mut resolved.values);
        self.publish_color_ids(&mut resolved.color_ids);
        Ok(resolved)
    }

    fn evaluate_representative_resolved_f64(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeResolvedEvaluation, RusticolError> {
        self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
        self.record_resolved_warnings(helicity_ids, color_ids)?;
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if let NativeExecutionLane::OnTheFly(runtime) = &mut self.execution_lane {
            let selected_helicities = selector_set(helicity_ids, "helicity")?;
            let selected_colors = selector_set(color_ids, "color component")?;
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                self.input_crossing_map.as_deref(),
            )?;
            let (resolved, _profile) = runtime.run_resolved(
                &self.runtime,
                batch,
                selected_helicities.as_ref(),
                selected_colors.as_ref(),
            )?;
            let (helicity_ids, color_ids) = runtime.selected_axis_ids()?;
            return Ok(NativeResolvedEvaluation {
                values: resolved.values,
                point_count: resolved.point_count,
                helicity_ids,
                color_ids,
            });
        }
        let physics = self.runtime.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let mut selector_scratch = std::mem::take(&mut self.point_selector_scratch);
        let result = (|| {
            physics.validate_helicity_id_slice(helicity_ids)?;
            physics.validate_color_id_slice(color_ids)?;
            let selected_helicities = selector_scratch
                .helicity_selector_sets
                .resolve(helicity_ids, "helicity")?;
            let selected_colors = selector_scratch
                .color_selector_sets
                .resolve(color_ids, "color component")?;
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            self.run_resolved_f64_batch(batch, selected_helicities, selected_colors)
        })();
        self.point_selector_scratch = selector_scratch;
        self.input_crossing_map = crossing_lookup;
        let resolved = result?;
        let helicity_ids = resolved
            .helicity_indices
            .iter()
            .map(|index| physics.manifest.helicities[*index].id.clone())
            .collect();
        let color_ids = resolved
            .color_indices
            .iter()
            .map(|index| physics.manifest.color_components[*index].id().to_string())
            .collect();
        Ok(NativeResolvedEvaluation {
            values: resolved.values,
            point_count: resolved.point_count,
            helicity_ids,
            color_ids,
        })
    }

    #[cfg(feature = "symbolica-runtime")]
    pub fn evaluate_with_precision(
        &mut self,
        momenta: &[String],
        point_count: usize,
        decimal_digits: u32,
    ) -> Result<NativeDecimalEvaluation, RusticolError> {
        if decimal_digits == 0 {
            return Err(RusticolError::unsupported_precision(
                "precision must be a positive number of decimal digits",
            ));
        }
        if decimal_digits == 16 {
            let values = momenta
                .iter()
                .map(|value| {
                    value.parse::<f64>().map_err(|error| {
                        RusticolError::invalid_argument(format!(
                            "could not parse f64 momentum component {value:?}: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let values = self.evaluate_f64(&values, point_count)?;
            return Ok(NativeDecimalEvaluation {
                values: format_decimal_values(values, decimal_digits),
                decimal_digits,
            });
        }
        if self.execution_lane.is_eager() {
            return Err(eager_parity_pending("higher-precision evaluation"));
        }
        if self.execution_lane.is_recurrence() {
            return Err(recurrence_parity_pending("higher-precision evaluation"));
        }
        if decimal_digits == 32 {
            let batch = self.prepare_double_batch(momenta, point_count)?;
            let (mut values, _profile) = self.runtime.run_double(&batch)?;
            self.apply_process_group_factor_high_precision(&mut values);
            return Ok(NativeDecimalEvaluation {
                values: format_decimal_values(values, decimal_digits),
                decimal_digits,
            });
        }
        let binary_precision = decimal_digits_to_bits(decimal_digits);
        let batch = self.prepare_float_batch(momenta, point_count, binary_precision)?;
        let (mut values, _profile) = self.runtime.run_float(&batch, binary_precision)?;
        self.apply_process_group_factor_high_precision(&mut values);
        Ok(NativeDecimalEvaluation {
            values: format_decimal_values(values, decimal_digits),
            decimal_digits,
        })
    }

    #[cfg(feature = "symbolica-runtime")]
    fn apply_process_group_factor_high_precision<T>(&self, values: &mut [T])
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let NativeRationalFactor {
            numerator,
            denominator,
        } = self.process_group_factor();
        if numerator != denominator {
            let numerator = T::from(numerator as f64);
            let denominator = T::from(denominator as f64);
            for value in values {
                *value = value.clone() * numerator.clone() / denominator.clone();
            }
        }
    }

    #[cfg(feature = "symbolica-runtime")]
    pub fn evaluate_resolved_with_precision(
        &mut self,
        momenta: &[String],
        point_count: usize,
        decimal_digits: u32,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeDecimalResolvedEvaluation, RusticolError> {
        if decimal_digits == 0 {
            return Err(RusticolError::unsupported_precision(
                "precision must be a positive number of decimal digits",
            ));
        }
        let color_ids = self.execution_color_ids(color_ids);
        let color_ids = color_ids.as_deref();
        self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
        if self.execution_lane.is_eager() {
            return Err(eager_parity_pending("resolved higher-precision evaluation"));
        }
        if self.execution_lane.is_recurrence() {
            return Err(recurrence_parity_pending(
                "resolved higher-precision evaluation",
            ));
        }
        self.record_resolved_warnings(helicity_ids, color_ids)?;
        let selected_helicities = selector_set(helicity_ids, "helicity")?;
        let selected_colors = selector_set(color_ids, "color component")?;
        if decimal_digits == 16 {
            let values = momenta
                .iter()
                .map(|value| {
                    value.parse::<f64>().map_err(|error| {
                        RusticolError::invalid_argument(format!(
                            "could not parse f64 momentum component {value:?}: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut resolved = self.evaluate_representative_resolved_f64(
                &values,
                point_count,
                helicity_ids,
                color_ids,
            )?;
            self.apply_process_group_factor(&mut resolved.values);
            self.publish_color_ids(&mut resolved.color_ids);
            let totals = resolved.totals();
            return Ok(NativeDecimalResolvedEvaluation {
                values: format_decimal_values(resolved.values, decimal_digits),
                totals: format_decimal_values(totals, decimal_digits),
                point_count: resolved.point_count,
                helicity_ids: resolved.helicity_ids,
                color_ids: resolved.color_ids,
                decimal_digits,
            });
        }
        let physics =
            self.runtime.physics.clone().ok_or_else(|| {
                RusticolError::artifact("resolved physics metadata is unavailable")
            })?;
        let mut evaluation = if decimal_digits == 32 {
            let batch = self.prepare_double_batch(momenta, point_count)?;
            let (mut resolved, _profile) = self.runtime.run_resolved_generic(
                &batch,
                None,
                selected_helicities.as_ref(),
                selected_colors.as_ref(),
            )?;
            self.apply_process_group_factor_high_precision(&mut resolved.values);
            decimal_resolved_evaluation(resolved, &physics.manifest, decimal_digits)?
        } else {
            let binary_precision = decimal_digits_to_bits(decimal_digits);
            let batch = self.prepare_float_batch(momenta, point_count, binary_precision)?;
            let (mut resolved, _profile) = self.runtime.run_resolved_generic(
                &batch,
                Some(binary_precision),
                selected_helicities.as_ref(),
                selected_colors.as_ref(),
            )?;
            self.apply_process_group_factor_high_precision(&mut resolved.values);
            decimal_resolved_evaluation(resolved, &physics.manifest, decimal_digits)?
        };
        self.publish_color_ids(&mut evaluation.color_ids);
        Ok(evaluation)
    }

    pub fn set_model_parameters(
        &mut self,
        values: &BTreeMap<String, (f64, f64)>,
    ) -> Result<(), RusticolError> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if matches!(self.execution_lane, NativeExecutionLane::OnTheFly(_)) {
            return self
                .runtime
                .apply_model_parameter_overrides(values)
                .map_err(|error| RusticolError::model_parameter(error.to_string()));
        }
        let physics = self.physics_v1.get()?;
        for name in values.keys() {
            let parameter = physics
                .model_parameters
                .iter()
                .find(|parameter| parameter.name == *name)
                .ok_or_else(|| {
                    RusticolError::model_parameter(format!(
                        "model parameter {name:?} is not declared by process {}",
                        self.process
                    ))
                })?;
            if !parameter.mutable {
                return Err(RusticolError::model_parameter(format!(
                    "model parameter {name:?} is derived or immutable"
                )));
            }
        }
        self.runtime
            .apply_model_parameter_overrides(values)
            .map_err(|error| RusticolError::model_parameter(error.to_string()))
    }

    /// Drop selector-local warmed state without unloading the artifact or
    /// changing runtime model parameters. Other execution modes have no
    /// corresponding query-family cache and therefore treat this as a no-op.
    pub fn clear(&mut self) -> RusticolResult<()> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        match &mut self.execution_lane {
            NativeExecutionLane::OnTheFly(runtime) => runtime.clear()?,
            NativeExecutionLane::Recurrence(runtime) => {
                runtime.clear_helicity_selector_companion()?
            }
            _ => {}
        }
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if matches!(
            &self.execution_lane,
            NativeExecutionLane::OnTheFly(_) | NativeExecutionLane::Recurrence(_)
        ) {
            self.point_selector_scratch = PointSelectorExecutionScratch::default();
        }
        Ok(())
    }

    pub fn set_model_parameter(
        &mut self,
        name: &str,
        real: f64,
        imaginary: f64,
    ) -> Result<(), RusticolError> {
        self.set_model_parameters(&BTreeMap::from([(name.to_string(), (real, imaginary))]))
    }

    pub fn set_model_parameters_json(&mut self, path: &Path) -> Result<(), RusticolError> {
        let text = fs::read_to_string(path).map_err(|error| {
            RusticolError::model_parameter(format!(
                "could not read model-parameter JSON {}: {error}",
                path.display()
            ))
        })?;
        let overrides = parse_complex_parameter_overrides(&text, path)
            .map_err(|error| RusticolError::model_parameter(error.to_string()))?;
        self.set_model_parameters(&overrides)
    }

    pub fn mute_warnings(&mut self) {
        self.warnings_muted = true;
    }

    pub fn unmute_warnings(&mut self) {
        self.warnings_muted = false;
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_warnings)
    }

    pub fn pending_warnings_json(&self) -> Result<String, RusticolError> {
        serde_json::to_string(&self.pending_warnings).map_err(|error| {
            RusticolError::serialization(format!("could not serialize warnings: {error}"))
        })
    }

    pub fn clear_pending_warnings(&mut self) {
        self.pending_warnings.clear();
    }

    pub fn take_warnings_json(&mut self) -> Result<String, RusticolError> {
        serde_json::to_string(&self.take_warnings()).map_err(|error| {
            RusticolError::serialization(format!("could not serialize warnings: {error}"))
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn prepare_f64_batch_profile(
        &self,
        momenta: &[f64],
        point_count: usize,
    ) -> Result<ProfiledPreparedF64Batch, RusticolError> {
        let pack_start = Instant::now();
        let batch = self.prepare_uncrossed_f64_batch(momenta, point_count)?;
        let pack_elapsed = pack_start.elapsed();
        if self.input_crossing_map.is_none() {
            return Ok((batch, pack_elapsed, Duration::ZERO));
        }
        let crossing_start = Instant::now();
        let batch = apply_input_crossing_map(
            batch,
            self.runtime.external_count,
            self.input_crossing_map.as_deref(),
        )?;
        Ok((batch, pack_elapsed, crossing_start.elapsed()))
    }

    fn prepare_uncrossed_f64_batch(
        &self,
        momenta: &[f64],
        point_count: usize,
    ) -> Result<Vec<Vec<[f64; 4]>>, RusticolError> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "point_count must be positive",
            ));
        }
        let values_per_point = self
            .runtime
            .external_count
            .checked_mul(4)
            .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
        let expected = point_count
            .checked_mul(values_per_point)
            .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
        if momenta.len() != expected {
            return Err(RusticolError::invalid_argument(format!(
                "momenta contain {} values, expected {expected} for shape ({point_count}, {}, 4)",
                momenta.len(),
                self.runtime.external_count
            )));
        }
        let mut batch = Vec::with_capacity(point_count);
        for point_values in momenta.chunks_exact(values_per_point) {
            let point = point_values
                .chunks_exact(4)
                .map(|components| [components[0], components[1], components[2], components[3]])
                .collect();
            batch.push(point);
        }
        Ok(batch)
    }

    #[cfg(feature = "symbolica-runtime")]
    fn prepare_double_batch(
        &self,
        momenta: &[String],
        point_count: usize,
    ) -> RusticolResult<Vec<Vec<[DoubleFloat; 4]>>> {
        let floats = self.prepare_float_batch(momenta, point_count, 106)?;
        Ok(floats
            .into_iter()
            .map(|point| {
                point
                    .into_iter()
                    .map(|leg| {
                        [
                            leg[0].to_double_float(),
                            leg[1].to_double_float(),
                            leg[2].to_double_float(),
                            leg[3].to_double_float(),
                        ]
                    })
                    .collect()
            })
            .collect())
    }

    #[cfg(feature = "symbolica-runtime")]
    fn prepare_float_batch(
        &self,
        momenta: &[String],
        point_count: usize,
        binary_precision: u32,
    ) -> RusticolResult<Vec<Vec<[Float; 4]>>> {
        let values_per_point =
            validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let mut batch = Vec::with_capacity(point_count);
        for point_values in momenta.chunks_exact(values_per_point) {
            let mut point = Vec::with_capacity(self.runtime.external_count);
            for components in point_values.chunks_exact(4) {
                let values = components
                    .iter()
                    .map(|value| {
                        Float::parse(value, Some(binary_precision)).map_err(|error| {
                            RusticolError::invalid_argument(format!(
                                "could not parse high-precision momentum component {value:?}: {error}"
                            ))
                        })
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                point.push([
                    values[0].clone(),
                    values[1].clone(),
                    values[2].clone(),
                    values[3].clone(),
                ]);
            }
            batch.push(point);
        }
        apply_input_crossing_map_generic(
            &batch,
            self.runtime.external_count,
            self.input_crossing_map.as_deref(),
        )
    }

    pub(super) fn record_resolved_warnings(
        &mut self,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<(), RusticolError> {
        if self.warnings_muted {
            return Ok(());
        }
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if matches!(self.execution_lane, NativeExecutionLane::OnTheFly(_)) {
            // Compact selector construction authenticates complete LC
            // coverage and materializes exact requested members directly;
            // there are no representative-only rows to warn about here.
            return Ok(());
        }
        let physics = self.runtime.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("resolved evaluation requires regenerated physics metadata")
        })?;
        let mut warnings = Vec::new();
        if physics.manifest.coverage.helicities != "complete" {
            warnings.push((
                "incomplete-helicity-coverage",
                "resolved evaluation contains only the helicities represented by this artifact",
            ));
        }
        if physics.manifest.color_accuracy == crate::ColorAccuracy::Lc
            && physics.manifest.coverage.color != "complete"
        {
            warnings.push((
                "incomplete-color-coverage",
                "resolved evaluation contains only the color components represented by this artifact",
            ));
        }
        let reduction_only_helicity = helicity_ids.is_some_and(|ids| {
            ids.iter().any(|id| {
                physics
                    .helicity_index_by_id
                    .get(id)
                    .and_then(|index| physics.manifest.helicities.get(*index))
                    .is_some_and(|item| !item.computed)
            })
        });
        let reduction_only_color = color_ids.is_some_and(|ids| {
            ids.iter().any(|id| {
                physics
                    .color_index_by_id
                    .get(id)
                    .is_some_and(|index| !physics.color_is_computed(*index))
            })
        });
        if reduction_only_helicity || reduction_only_color {
            warnings.push((
                "reduction-only-selection",
                "the selected resolved component reuses an exact symmetry representative",
            ));
        }
        for (kind, message) in warnings {
            if self.warned_kinds.insert(kind.to_string()) {
                self.pending_warnings.push(message.to_string());
            }
        }
        Ok(())
    }

    fn validate_selector_capabilities(
        &self,
        helicity_requested: bool,
        color_requested: bool,
    ) -> Result<(), RusticolError> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if matches!(self.execution_lane, NativeExecutionLane::OnTheFly(_)) {
            if color_requested && self.runtime.color_accuracy != "lc" {
                return Err(RusticolError::selector(
                    "on-the-fly color selection is available only for LC artifacts",
                ));
            }
            return Ok(());
        }
        let physics = self.physics_v1.get()?;
        if helicity_requested && !physics.selectors.helicity {
            return Err(RusticolError::selector(
                "this artifact does not support physical helicity selection",
            ));
        }
        if color_requested {
            if self.runtime.color_accuracy != "lc" {
                return Err(RusticolError::selector(
                    "LC color-flow selection is unavailable for NLC/full artifacts; their resolved color axis is contracted",
                ));
            }
            if !physics.selectors.color_flow {
                return Err(RusticolError::selector(
                    "this artifact does not support physical color-flow selection",
                ));
            }
        }
        Ok(())
    }
}

fn validate_flat_momentum_shape(
    value_count: usize,
    point_count: usize,
    external_count: usize,
) -> RusticolResult<usize> {
    if point_count == 0 {
        return Err(RusticolError::invalid_argument(
            "point_count must be positive",
        ));
    }
    let values_per_point = external_count
        .checked_mul(4)
        .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
    let expected = point_count
        .checked_mul(values_per_point)
        .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
    if value_count != expected {
        return Err(RusticolError::invalid_argument(format!(
            "momenta contain {value_count} values, expected {expected} for shape ({point_count}, {external_count}, 4)"
        )));
    }
    Ok(values_per_point)
}

#[cfg(feature = "symbolica-runtime")]
fn format_decimal_values<T: std::fmt::LowerExp>(
    values: Vec<T>,
    decimal_digits: u32,
) -> Vec<String> {
    let digits = decimal_digits as usize;
    values
        .into_iter()
        .map(|value| format!("{value:.digits$e}"))
        .collect()
}

#[cfg(feature = "symbolica-runtime")]
fn decimal_resolved_evaluation<T>(
    resolved: ResolvedValues<T>,
    physics: &ProcessPhysicsV1,
    decimal_digits: u32,
) -> RusticolResult<NativeDecimalResolvedEvaluation>
where
    T: RusticolHighPrecisionNumber + std::fmt::LowerExp,
    Complex<T>: Real + EvaluationDomain,
{
    let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
    if component_count == 0 {
        return Err(RusticolError::internal(
            "resolved evaluation produced an empty component axis",
        ));
    }
    let mut totals = Vec::with_capacity(resolved.point_count);
    for point in resolved.values.chunks(component_count) {
        let mut total = T::new_zero();
        for value in point {
            total += value.clone();
        }
        totals.push(total);
    }
    let helicity_ids = resolved
        .helicity_indices
        .iter()
        .map(|index| physics.helicities[*index].id.clone())
        .collect();
    let color_ids = resolved
        .color_indices
        .iter()
        .map(|index| physics.color_components[*index].id().to_string())
        .collect();
    Ok(NativeDecimalResolvedEvaluation {
        values: format_decimal_values(resolved.values, decimal_digits),
        totals: format_decimal_values(totals, decimal_digits),
        point_count: resolved.point_count,
        helicity_ids,
        color_ids,
        decimal_digits,
    })
}

pub(super) fn parse_public_kinematics_point(
    value: &Value,
    external_count: usize,
) -> RusticolResult<Vec<f64>> {
    let outer = value.as_array().ok_or_else(|| {
        RusticolError::invalid_argument(
            "kinematics JSON must be one [external][4] point or a singleton batch",
        )
    })?;
    let point = if looks_like_kinematics_point(outer) {
        outer.as_slice()
    } else if outer.len() == 1 {
        outer[0].as_array().map(Vec::as_slice).ok_or_else(|| {
            RusticolError::invalid_argument(
                "kinematics JSON singleton batch must contain one [external][4] point",
            )
        })?
    } else {
        return Err(RusticolError::invalid_argument(
            "kinematics JSON must contain exactly one point",
        ));
    };
    if point.len() != external_count {
        return Err(RusticolError::invalid_argument(format!(
            "kinematics point has {} external momenta, expected {external_count}",
            point.len()
        )));
    }
    let mut flat = Vec::with_capacity(external_count.saturating_mul(4));
    for (external_index, momentum) in point.iter().enumerate() {
        let components = momentum.as_array().ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "kinematics momentum {external_index} must be an array of four components"
            ))
        })?;
        if components.len() != 4 {
            return Err(RusticolError::invalid_argument(format!(
                "kinematics momentum {external_index} has {} components, expected 4",
                components.len()
            )));
        }
        for (component_index, component) in components.iter().enumerate() {
            let parsed = match component {
                Value::Number(number) => number.as_f64(),
                Value::String(number) => number.parse::<f64>().ok(),
                _ => None,
            }
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "kinematics component [{external_index}][{component_index}] must be a finite JSON number or decimal string"
                ))
            })?;
            flat.push(parsed);
        }
    }
    Ok(flat)
}

fn looks_like_kinematics_point(values: &[Value]) -> bool {
    values.iter().all(|momentum| {
        momentum.as_array().is_some_and(|components| {
            components.len() == 4
                && components
                    .iter()
                    .all(|component| matches!(component, Value::Number(_) | Value::String(_)))
        })
    })
}

#[cfg(feature = "symbolica-runtime")]
fn eager_parity_pending(feature: &str) -> RusticolError {
    RusticolError::unsupported_runtime_capability(
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        format!(
            "eager {feature} is not provided by the native f64 Direct-Arena ABI; \
             use the public Python Runtime exact-evaluation path for precision above 16 digits"
        ),
    )
}

#[cfg(feature = "symbolica-runtime")]
fn recurrence_parity_pending(feature: &str) -> RusticolError {
    RusticolError::unsupported_runtime_capability(
        RECURRENCE_RUNTIME_CAPABILITY,
        format!(
            "recurrence {feature} is not available in the initial f64 runtime slice; use compiled execution for this operation"
        ),
    )
}

#[cfg(test)]
mod runtime_capability_cutover_tests {
    use super::*;

    #[test]
    fn retired_eager_v2_capabilities_fail_with_regeneration_guidance() {
        for capability in [
            EAGER_DAG_RUNTIME_CAPABILITY,
            EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
        ] {
            let error = ensure_selected_runtime_capabilities_supported(&[capability.to_string()])
                .expect_err("retired eager capability must fail before generic preflight");
            assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
            assert!(error.to_string().contains("legacy eager plan-v2"));
            assert!(error.to_string().contains("regenerate"));
        }
    }
}
