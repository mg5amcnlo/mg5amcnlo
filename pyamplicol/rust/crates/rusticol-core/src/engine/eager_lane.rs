// SPDX-License-Identifier: 0BSD

use super::evaluation::accumulate_selected_lc_replay_resolved_f64;
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct EagerParameterProjectionEntry {
    pub(super) prepared_index: usize,
    pub(super) runtime_real_index: usize,
    pub(super) runtime_imaginary_index: Option<usize>,
}

#[derive(Debug)]
pub(super) struct EagerParameterProjection {
    pub(super) parameter_count: usize,
    pub(super) entries: Vec<EagerParameterProjectionEntry>,
}

#[allow(dead_code)] // Includes the order-preserving per-point benchmark reference buffers.
pub(super) struct EagerNativeRuntime {
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    direct_scheduler: Option<crate::eager_runtime::EagerDirectExecutionRuntime>,
    backend_name: String,
    amplitude_count: usize,
    has_selector_domains: bool,
    effective_point_tile_size: usize,
    workspace_bytes: usize,
    parameter_projection: EagerParameterProjection,
    raw_sum_groups: Vec<RawSumGroup>,
    lc_resolved_replay_plan: Option<Arc<LcResolvedReplayPlan>>,
    lc_resolved_replay_selection_cache:
        Vec<(LcResolvedReplaySelectionKey, Arc<LcResolvedReplaySelection>)>,
    lc_replay_expanded_batch: Vec<Vec<[f64; 4]>>,
    lc_replay_seen_labels: Vec<bool>,
    lc_replay_flat_momenta: Vec<f64>,
    lc_replay_materialized_values: Vec<f64>,
    lc_replay_target_components: Vec<f64>,
    color_topology_replay: Option<AmplitudeRuntime>,
    color_contraction: Option<ColorContractionRuntime>,
    initial_values: Vec<crate::EagerComplex64>,
    momenta: Vec<f64>,
    model_parameters: Vec<crate::EagerComplex64>,
    amplitudes: Vec<crate::EagerComplex64>,
    color_group_scratch: Vec<crate::EagerComplex64>,
    reduced: Vec<f64>,
    selected_groups: Vec<u32>,
    point_selector_offsets: Vec<usize>,
    point_selector_groups: Vec<u32>,
    point_selector_group_weights: Vec<f64>,
    point_selector_members: Option<Vec<Vec<(usize, usize, f64)>>>,
    point_selector_pairs: Vec<(u32, f64)>,
    selected_helicity_indices: Vec<usize>,
    selected_color_indices: Vec<usize>,
    helicity_position_scratch: Vec<Option<usize>>,
    color_position_scratch: Vec<Option<usize>>,
    helicity_weights_scratch: Vec<(usize, f64)>,
    right_helicity_weights_scratch: Vec<(usize, f64)>,
    member_weights_scratch: Vec<(usize, usize, f64)>,
    source_schedule: Option<EagerSourceSchedule>,
    source_wavefunction_scratch: Vec<Complex<f64>>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct EagerSourceKey {
    source_kind: String,
    leg_label: usize,
    family: u8,
    dimension: usize,
    pdg_label: i32,
    anti_pdg_label: i32,
    orientation: u8,
    momentum_transform: u8,
    phase_real: u64,
    phase_imaginary: u64,
    source_helicity: i32,
    chirality: i32,
}

#[derive(Debug)]
struct EagerSourceClass {
    representative_index: usize,
    target_component_starts: Vec<usize>,
    dimension: usize,
}

#[derive(Debug)]
struct EagerSourceSchedule {
    classes: Vec<EagerSourceClass>,
    maximum_dimension: usize,
}

impl EagerNativeRuntime {
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub(super) fn new_direct(
        scheduler: crate::eager_runtime::EagerDirectExecutionRuntime,
        backend_name: String,
        parameter_projection: EagerParameterProjection,
        raw_sum_groups: Vec<RawSumGroup>,
        color_contraction: Option<ColorContractionRuntime>,
        color_topology_replay: Option<AmplitudeRuntime>,
    ) -> Self {
        let amplitude_count = scheduler.plan().amplitude_count();
        let has_selector_domains = scheduler.plan().has_selector_domains();
        let effective_point_tile_size = scheduler.effective_point_tile_size();
        let workspace_bytes = scheduler.workspace_bytes();
        Self {
            direct_scheduler: Some(scheduler),
            backend_name,
            amplitude_count,
            has_selector_domains,
            effective_point_tile_size,
            workspace_bytes,
            parameter_projection,
            raw_sum_groups,
            lc_resolved_replay_plan: None,
            lc_resolved_replay_selection_cache: Vec::new(),
            lc_replay_expanded_batch: Vec::new(),
            lc_replay_seen_labels: Vec::new(),
            lc_replay_flat_momenta: Vec::new(),
            lc_replay_materialized_values: Vec::new(),
            lc_replay_target_components: Vec::new(),
            color_topology_replay,
            color_contraction,
            initial_values: Vec::new(),
            momenta: Vec::new(),
            model_parameters: Vec::new(),
            amplitudes: Vec::new(),
            color_group_scratch: Vec::new(),
            reduced: Vec::new(),
            selected_groups: Vec::new(),
            point_selector_offsets: Vec::new(),
            point_selector_groups: Vec::new(),
            point_selector_group_weights: Vec::new(),
            point_selector_members: None,
            point_selector_pairs: Vec::new(),
            selected_helicity_indices: Vec::new(),
            selected_color_indices: Vec::new(),
            helicity_position_scratch: Vec::new(),
            color_position_scratch: Vec::new(),
            helicity_weights_scratch: Vec::new(),
            right_helicity_weights_scratch: Vec::new(),
            member_weights_scratch: Vec::new(),
            source_schedule: None,
            source_wavefunction_scratch: Vec::new(),
        }
    }

    fn execute_full_scheduler(&mut self, point_count: usize) -> RusticolResult<()> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            self.direct_scheduler
                .as_mut()
                .ok_or_else(|| {
                    RusticolError::internal(
                        "production eager runtime has no Direct-Arena scheduler",
                    )
                })?
                .evaluate_into(
                    point_count,
                    &self.initial_values,
                    &self.momenta,
                    &self.model_parameters,
                    &mut self.amplitudes,
                    &mut self.reduced,
                )
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            let _ = point_count;
            Err(RusticolError::compatibility(
                "eager-direct-arena-v1 requires the f64-compiled or f64-symjit feature; regenerate or load \
                 this artifact with a Direct-Arena-capable runtime",
            ))
        }
    }

    fn execute_full_totals_scheduler(&mut self, point_count: usize) -> RusticolResult<()> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            self.direct_scheduler
                .as_mut()
                .ok_or_else(|| {
                    RusticolError::internal(
                        "production eager runtime has no Direct-Arena scheduler",
                    )
                })?
                .evaluate_totals_into(
                    point_count,
                    &self.initial_values,
                    &self.momenta,
                    &self.model_parameters,
                    &mut self.reduced,
                )
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            let _ = point_count;
            Err(RusticolError::compatibility(
                "eager-direct-arena-v1 requires the f64-compiled or f64-symjit feature; regenerate or load \
                 this artifact with a Direct-Arena-capable runtime",
            ))
        }
    }

    fn execute_selected_scheduler(
        &mut self,
        active_groups: &[u32],
        point_count: usize,
    ) -> RusticolResult<()> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            self.direct_scheduler
                .as_mut()
                .ok_or_else(|| {
                    RusticolError::internal(
                        "production eager runtime has no Direct-Arena scheduler",
                    )
                })?
                .evaluate_selected_active_amplitudes_into(
                    active_groups,
                    point_count,
                    &self.initial_values,
                    &self.momenta,
                    &self.model_parameters,
                    &mut self.amplitudes,
                )
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            let _ = (active_groups, point_count);
            Err(RusticolError::compatibility(
                "eager-direct-arena-v1 requires the f64-compiled or f64-symjit feature; regenerate or load \
                 this artifact with a Direct-Arena-capable runtime",
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_point_selected_scheduler(
        &mut self,
        group_offsets: &[usize],
        active_groups: &[u32],
        active_group_weights: &[f64],
        point_count: usize,
        reduced: &mut [f64],
    ) -> RusticolResult<()> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            self.direct_scheduler
                .as_mut()
                .ok_or_else(|| {
                    RusticolError::internal(
                        "production eager runtime has no Direct-Arena scheduler",
                    )
                })?
                .evaluate_point_selected_group_sets_into(
                    group_offsets,
                    active_groups,
                    active_group_weights,
                    point_count,
                    &self.initial_values,
                    &self.momenta,
                    &self.model_parameters,
                    reduced,
                )
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            let _ = (
                group_offsets,
                active_groups,
                active_group_weights,
                point_count,
                reduced,
            );
            Err(RusticolError::compatibility(
                "eager-direct-arena-v1 requires the f64-compiled or f64-symjit feature; regenerate or load \
                 this artifact with a Direct-Arena-capable runtime",
            ))
        }
    }

    fn execute_profile_scheduler(
        &mut self,
        active_groups: Option<&[u32]>,
        point_count: usize,
    ) -> RusticolResult<crate::eager_runtime::EagerExecutionProfile> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            let scheduler = self.direct_scheduler.as_mut().ok_or_else(|| {
                RusticolError::internal("production eager runtime has no Direct-Arena scheduler")
            })?;
            if let Some(active_groups) = active_groups {
                scheduler.evaluate_selected_profile_into(
                    active_groups,
                    point_count,
                    &self.initial_values,
                    &self.momenta,
                    &self.model_parameters,
                    &mut self.amplitudes,
                )
            } else {
                scheduler.evaluate_profile_into(
                    point_count,
                    &self.initial_values,
                    &self.momenta,
                    &self.model_parameters,
                    &mut self.amplitudes,
                    &mut self.reduced,
                )
            }
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            let _ = (active_groups, point_count);
            Err(RusticolError::compatibility(
                "eager-direct-arena-v1 requires the f64-compiled or f64-symjit feature; regenerate or load \
                 this artifact with a Direct-Arena-capable runtime",
            ))
        }
    }

    fn direct_internal_scratch_bytes(&self) -> u64 {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            self.direct_scheduler
                .as_ref()
                .map_or(0, |scheduler| scheduler.internal_scratch_bytes())
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            0
        }
    }

    fn direct_internal_broadcast_bytes(&self) -> u64 {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            self.direct_scheduler
                .as_ref()
                .map_or(0, |scheduler| scheduler.internal_broadcast_bytes())
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            0
        }
    }

    pub(super) fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub(super) fn effective_point_tile_size(&self) -> usize {
        self.effective_point_tile_size
    }

    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Execute the warmed eager Direct-Arena lane from a borrowed flat input.
    ///
    /// This is the ordinary native timing boundary: it performs no phase
    /// clocks, nested momentum materialization, resolved-value construction,
    /// or result allocation.
    pub(super) fn run_f64_view_into_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let point_count = batch.point_count();
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "eager evaluation output has length {}, expected {point_count}",
                output.len()
            )));
        }
        if common.color_topology_replay_enabled {
            return self.run_f64_view_with_color_topology_replay_into_unprofiled(
                common, batch, None, None, output,
            );
        }
        if common.lc_topology_replay_enabled {
            return self.run_f64_view_with_lc_topology_replay_into_unprofiled(
                common, batch, None, None, output,
            );
        }
        prepare_eager_inputs_view(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        self.reduced.resize(point_count, 0.0);
        self.execute_full_totals_scheduler(point_count)?;
        for (target, value) in output.iter_mut().zip(&self.reduced) {
            *target = common.normalization_factor * *value;
        }
        Ok(())
    }

    /// Execute public global selectors directly into caller-sized totals.
    ///
    /// The selector weights feed the plane-native eager reducer, so LC,
    /// compact NLC, and full-colour contractions avoid materializing the
    /// resolved `(point, helicity, colour)` tensor.
    pub(super) fn run_f64_view_selected_into_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let point_count = batch.point_count();
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "eager evaluation output has length {}, expected {point_count}",
                output.len()
            )));
        }
        if selected_helicity_ids.is_none() && selected_color_ids.is_none() {
            return self.run_f64_view_into_unprofiled(common, batch, output);
        }
        if common.color_topology_replay_enabled {
            return self.run_f64_view_with_color_topology_replay_into_unprofiled(
                common,
                batch,
                selected_helicity_ids,
                selected_color_ids,
                output,
            );
        }
        if common.lc_topology_replay_enabled {
            return self.run_f64_view_with_lc_topology_replay_into_unprofiled(
                common,
                batch,
                selected_helicity_ids,
                selected_color_ids,
                output,
            );
        }
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("selected eager evaluation requires physics metadata")
        })?;
        if self.point_selector_members.is_none() {
            self.point_selector_members = Some(build_eager_selector_members(
                &self.raw_sum_groups,
                self.color_contraction.is_some(),
                &physics,
            )?);
        }
        fill_eager_group_sets_by_point(
            &self.raw_sum_groups,
            self.point_selector_members
                .as_deref()
                .expect("selector members initialized"),
            &physics,
            point_count,
            selected_helicity_ids,
            selected_color_ids,
            None,
            None,
            &mut self.point_selector_offsets,
            &mut self.point_selector_groups,
            &mut self.point_selector_group_weights,
            &mut self.point_selector_pairs,
            &mut self.selected_helicity_indices,
            &mut self.selected_color_indices,
        )?;
        prepare_eager_inputs_view(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;

        // Keep caller output untouched if validation or execution fails.
        let offsets = std::mem::take(&mut self.point_selector_offsets);
        let groups = std::mem::take(&mut self.point_selector_groups);
        let group_weights = std::mem::take(&mut self.point_selector_group_weights);
        let mut reduced = std::mem::take(&mut self.reduced);
        reduced.resize(point_count, 0.0);
        let result = self.execute_point_selected_scheduler(
            &offsets,
            &groups,
            &group_weights,
            point_count,
            &mut reduced,
        );
        self.point_selector_offsets = offsets;
        self.point_selector_groups = groups;
        self.point_selector_group_weights = group_weights;
        self.reduced = reduced;
        result?;
        for (target, value) in output.iter_mut().zip(&self.reduced) {
            *target = common.normalization_factor * *value;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_resolved_f64_view_materialized_into_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut Vec<f64>,
    ) -> RusticolResult<()> {
        let point_count = batch.point_count();
        prepare_eager_inputs_view(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.amplitude_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved eager evaluation requires physics metadata")
        })?;
        let has_selected_groups = if self.has_selector_domains
            && (selected_helicity_ids.is_some() || selected_color_ids.is_some())
        {
            fill_selected_eager_group_ids(
                &self.raw_sum_groups,
                &physics,
                selected_helicity_ids,
                selected_color_ids,
                &mut self.selected_groups,
            )?;
            true
        } else {
            false
        };
        if has_selected_groups {
            let selected_groups = std::mem::take(&mut self.selected_groups);
            let result = self.execute_selected_scheduler(&selected_groups, point_count);
            self.selected_groups = selected_groups;
            result?;
        } else {
            self.execute_full_scheduler(point_count)?;
        }
        reduce_eager_amplitudes_resolved_into(
            &self.amplitudes,
            point_count,
            &self.raw_sum_groups,
            self.color_contraction.as_mut(),
            &mut self.color_group_scratch,
            &physics,
            common.normalization_factor,
            selected_helicity_ids,
            selected_color_ids,
            output,
            &mut self.selected_helicity_indices,
            &mut self.selected_color_indices,
            &mut self.helicity_position_scratch,
            &mut self.color_position_scratch,
            &mut self.helicity_weights_scratch,
            &mut self.right_helicity_weights_scratch,
            &mut self.member_weights_scratch,
        )
    }

    /// Resolve eager components directly from a borrowed momentum view.
    ///
    /// The returned value and public-axis vectors are the only result-owned
    /// allocations. Input crossing, eager source preparation, and topology
    /// replay all consume the borrowed flat view without constructing nested
    /// point or momentum containers.
    pub(super) fn run_resolved_f64_view_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let point_count = batch.point_count();
        let mut values = Vec::new();
        if common.color_topology_replay_enabled {
            return self.run_resolved_f64_view_with_color_topology_replay_unprofiled(
                common,
                batch,
                selected_helicity_ids,
                selected_color_ids,
            );
        }
        if common.lc_topology_replay_enabled {
            let selection = self.run_resolved_f64_view_with_lc_topology_replay_into_unprofiled(
                common,
                batch,
                selected_helicity_ids,
                selected_color_ids,
                &mut values,
            )?;
            return Ok(ResolvedValues {
                values,
                point_count,
                helicity_indices: selection.helicity_indices.clone(),
                color_indices: selection.color_indices.clone(),
            });
        }
        self.run_resolved_f64_view_materialized_into_unprofiled(
            common,
            batch,
            selected_helicity_ids,
            selected_color_ids,
            &mut values,
        )?;
        Ok(ResolvedValues {
            values,
            point_count,
            helicity_indices: self.selected_helicity_indices.clone(),
            color_indices: self.selected_color_indices.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_f64_view_with_color_topology_replay_into_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let resolved = self.run_resolved_f64_view_with_color_topology_replay_unprofiled(
            common,
            batch,
            selected_helicity_ids,
            selected_color_ids,
        )?;
        if output.len() != resolved.point_count {
            return Err(RusticolError::invalid_argument(
                "eager color replay output has an invalid point count",
            ));
        }
        let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
        if component_count == 0 {
            output.fill(0.0);
        } else {
            for (target, components) in output
                .iter_mut()
                .zip(resolved.values.chunks_exact(component_count))
            {
                *target = components.iter().sum();
            }
        }
        Ok(())
    }

    fn run_resolved_f64_view_with_color_topology_replay_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let point_count = batch.point_count();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "eager color topology replay requires resolved physics metadata",
            )
        })?;
        let mappings = Arc::clone(&common.color_topology_replay_mappings);
        let mut replay = self.color_topology_replay.take().ok_or_else(|| {
            RusticolError::integrity("eager color topology replay has no amplitude gather runtime")
        })?;
        let mut flat_momenta = std::mem::take(&mut self.lc_replay_flat_momenta);
        let result = (|| {
            replay.begin_color_topology_replay(point_count)?;
            for mapping_index in 0..mappings.len() {
                let evaluation_view = if mappings[mapping_index].is_empty() {
                    batch
                } else {
                    fill_lc_replay_expanded_flat_from_view(
                        &mut flat_momenta,
                        batch,
                        common.external_count,
                        &[mapping_index],
                        &mappings,
                    )?;
                    F64MomentumBatchView::from_contiguous_prevalidated(
                        &flat_momenta,
                        point_count,
                        common.external_count,
                        None,
                    )?
                };
                prepare_eager_inputs_view(
                    common,
                    evaluation_view,
                    &mut self.initial_values,
                    &mut self.momenta,
                    &mut self.source_schedule,
                    &mut self.source_wavefunction_scratch,
                )?;
                project_model_parameters(
                    &self.parameter_projection,
                    &common.model_parameter_values_f64,
                    &mut self.model_parameters,
                )?;
                self.amplitudes.resize(
                    point_count
                        .checked_mul(self.amplitude_count)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(
                                "eager color replay amplitudes overflow",
                            )
                        })?,
                    crate::EagerComplex64::new(0.0, 0.0),
                );
                self.reduced.resize(point_count, 0.0);
                self.execute_full_scheduler(point_count)?;
                replay.gather_color_topology_replay_eager_amplitudes(
                    &self.amplitudes,
                    point_count,
                    mapping_index,
                )?;
            }
            replay.reduce_color_topology_replay_f64_resolved(
                point_count,
                &physics,
                common.normalization_factor,
                selected_helicity_ids,
                selected_color_ids,
            )
        })();
        self.lc_replay_flat_momenta = flat_momenta;
        self.color_topology_replay = Some(replay);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_resolved_f64_view_with_lc_topology_replay_into_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut Vec<f64>,
    ) -> RusticolResult<Arc<LcResolvedReplaySelection>> {
        let point_count = batch.point_count();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("eager LC topology replay requires resolved physics metadata")
        })?;
        let replay_plan = if let Some(plan) = &self.lc_resolved_replay_plan {
            Arc::clone(plan)
        } else {
            let materialized_sector_by_id =
                common.lc_materialized_sectors_by_id_from_groups(&physics, &self.raw_sum_groups)?;
            let plan = Arc::new(physics.lc_resolved_replay_plan(
                &common.lc_topology_replay_public_mappings,
                &common.lc_topology_replay_routes,
                &materialized_sector_by_id,
            )?);
            self.lc_resolved_replay_plan = Some(Arc::clone(&plan));
            plan
        };
        let selection = if let Some((_, selection)) = self
            .lc_resolved_replay_selection_cache
            .iter()
            .find(|(key, _)| {
                lc_replay_selection_key_matches(
                    key,
                    &physics,
                    selected_helicity_ids,
                    selected_color_ids,
                )
            }) {
            Arc::clone(selection)
        } else {
            let selection_key = physics
                .lc_resolved_replay_selection_key(selected_helicity_ids, selected_color_ids)?;
            let selection = Arc::new(
                physics
                    .select_lc_resolved_replay_plan_for_key(replay_plan.as_ref(), &selection_key)?,
            );
            self.lc_resolved_replay_selection_cache
                .push((selection_key, Arc::clone(&selection)));
            selection
        };
        let target_component_count = selection
            .helicity_indices
            .len()
            .checked_mul(selection.color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved eager shape overflows"))?;
        let target_scalar_count = point_count
            .checked_mul(target_component_count)
            .ok_or_else(|| RusticolError::invalid_argument("resolved eager shape overflows"))?;
        output.resize(target_scalar_count, 0.0);
        output.fill(0.0);
        if target_component_count == 0 {
            return Ok(selection);
        }

        let mappings = Arc::clone(&common.lc_topology_replay_mappings);
        let mappings_per_chunk = replay_mappings_per_expanded_batch(point_count);
        let mut flat_momenta = std::mem::take(&mut self.lc_replay_flat_momenta);
        let mut materialized_values = std::mem::take(&mut self.lc_replay_materialized_values);
        let result = (|| {
            for source_group in &selection.source_groups {
                for chunk_start in
                    (0..source_group.mapping_indices.len()).step_by(mappings_per_chunk)
                {
                    let chunk_end = usize::min(
                        chunk_start + mappings_per_chunk,
                        source_group.mapping_indices.len(),
                    );
                    let mapping_indices = &source_group.mapping_indices[chunk_start..chunk_end];
                    let entry_chunk = &source_group.entries[chunk_start..chunk_end];
                    let identity_mapping =
                        mapping_indices.len() == 1 && mappings[mapping_indices[0]].is_empty();
                    let evaluation_view = if identity_mapping {
                        batch
                    } else {
                        fill_lc_replay_expanded_flat_from_view(
                            &mut flat_momenta,
                            batch,
                            common.external_count,
                            mapping_indices,
                            &mappings,
                        )?;
                        F64MomentumBatchView::from_contiguous_prevalidated(
                            &flat_momenta,
                            point_count * mapping_indices.len(),
                            common.external_count,
                            None,
                        )?
                    };
                    self.run_resolved_f64_view_materialized_into_unprofiled(
                        common,
                        evaluation_view,
                        Some(&source_group.helicity_ids),
                        Some(&source_group.color_ids),
                        &mut materialized_values,
                    )?;
                    accumulate_selected_lc_replay_values_f64(
                        output,
                        point_count,
                        &materialized_values,
                        entry_chunk,
                        source_group.source_component_count,
                        target_component_count,
                    )?;
                }
            }
            Ok(())
        })();
        self.lc_replay_flat_momenta = flat_momenta;
        self.lc_replay_materialized_values = materialized_values;
        result?;
        Ok(selection)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_f64_view_with_lc_topology_replay_into_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let point_count = batch.point_count();
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "eager topology-replay output has length {}, expected {point_count}",
                output.len()
            )));
        }
        let mut target_components = std::mem::take(&mut self.lc_replay_target_components);
        let result = self.run_resolved_f64_view_with_lc_topology_replay_into_unprofiled(
            common,
            batch,
            selected_helicity_ids,
            selected_color_ids,
            &mut target_components,
        );
        if let Ok(selection) = &result {
            let target_component_count =
                selection.helicity_indices.len() * selection.color_indices.len();
            if target_component_count == 0 {
                output.fill(0.0);
            } else {
                for (target, components) in output
                    .iter_mut()
                    .zip(target_components.chunks_exact(target_component_count))
                {
                    *target = components.iter().sum();
                }
            }
        }
        self.lc_replay_target_components = target_components;
        result.map(|_| ())
    }

    #[allow(dead_code)] // Retained as the row-major eager benchmark oracle.
    pub(super) fn run_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if common.color_topology_replay_enabled {
            let (resolved, profile) =
                self.run_resolved_f64_with_color_topology_replay(common, batch, None, None, false)?;
            return Ok((sum_eager_resolved(&resolved), profile));
        }
        if common.lc_topology_replay_enabled {
            let (resolved, profile) =
                self.run_resolved_f64_with_lc_topology_replay(common, batch, None, None, false)?;
            return Ok((sum_eager_resolved(&resolved), profile));
        }
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let (source_fill, momentum_setup) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.amplitude_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);

        let execute_start = Instant::now();
        self.execute_full_scheduler(point_count)?;
        let execute_s = execute_start.elapsed().as_secs_f64();
        for value in &mut self.reduced {
            *value *= common.normalization_factor;
        }
        let values = self.reduced.clone();
        Ok((
            values,
            RuntimeProfile {
                source_fill_s: profile_duration_seconds(source_fill),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup),
                momentum_setup_s: profile_duration_seconds(momentum_setup),
                stage_evaluator_call_s: execute_s,
                stage_evaluator_s: execute_s,
                total_s: total_start.elapsed().as_secs_f64(),
                ..RuntimeProfile::default()
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // Retained as the order-preserving benchmark reference lane.
    pub(super) fn run_f64_by_point_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if common.lc_topology_replay_enabled || common.color_topology_replay_enabled {
            return Err(RusticolError::integrity(
                "eager topology-replay per-point selectors must use the shared partition planner",
            ));
        }
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("per-point eager selection requires physics metadata")
        })?;
        if self.point_selector_members.is_none() {
            self.point_selector_members = Some(build_eager_selector_members(
                &self.raw_sum_groups,
                self.color_contraction.is_some(),
                &physics,
            )?);
        }
        fill_eager_group_sets_by_point(
            &self.raw_sum_groups,
            self.point_selector_members
                .as_deref()
                .expect("selector members initialized"),
            &physics,
            batch.len(),
            selected_helicity_ids,
            selected_color_ids,
            helicity_by_point,
            color_by_point,
            &mut self.point_selector_offsets,
            &mut self.point_selector_groups,
            &mut self.point_selector_group_weights,
            &mut self.point_selector_pairs,
            &mut self.selected_helicity_indices,
            &mut self.selected_color_indices,
        )?;
        let offsets = std::mem::take(&mut self.point_selector_offsets);
        let groups = std::mem::take(&mut self.point_selector_groups);
        let group_weights = std::mem::take(&mut self.point_selector_group_weights);
        let mut reduced = std::mem::take(&mut self.reduced);
        reduced.resize(batch.len(), 0.0);
        let result = self.run_coherent_group_sets_by_point_f64_into(
            common,
            batch,
            &offsets,
            &groups,
            &group_weights,
            &mut reduced,
        );
        self.point_selector_offsets = offsets;
        self.point_selector_groups = groups;
        self.point_selector_group_weights = group_weights;
        self.reduced = reduced;
        let profile = result?;
        Ok((self.reduced.clone(), profile))
    }

    #[allow(dead_code)] // Retained as the order-preserving benchmark reference lane.
    pub(super) fn run_coherent_group_sets_by_point_f64_into(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        group_offsets: &[usize],
        active_groups: &[u32],
        active_group_weights: &[f64],
        reduced: &mut [f64],
    ) -> RusticolResult<RuntimeProfile> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        if reduced.len() != batch.len() {
            return Err(RusticolError::invalid_argument(format!(
                "eager per-point selector output has length {}, expected {}",
                reduced.len(),
                batch.len()
            )));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let (source_fill, momentum_setup) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let execute_start = Instant::now();
        self.execute_point_selected_scheduler(
            group_offsets,
            active_groups,
            active_group_weights,
            point_count,
            reduced,
        )?;
        let execute_s = execute_start.elapsed().as_secs_f64();
        for value in reduced {
            *value *= common.normalization_factor;
        }
        Ok(RuntimeProfile {
            source_fill_s: profile_duration_seconds(source_fill),
            momentum_input_setup_s: profile_duration_seconds(momentum_setup),
            momentum_setup_s: profile_duration_seconds(momentum_setup),
            stage_evaluator_call_s: execute_s,
            stage_evaluator_s: execute_s,
            total_s: total_start.elapsed().as_secs_f64(),
            ..RuntimeProfile::default()
        })
    }

    pub(super) fn run_f64_profile(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if common.color_topology_replay_enabled {
            let (resolved, profile) =
                self.run_resolved_f64_with_color_topology_replay(common, batch, None, None, true)?;
            return Ok((sum_eager_resolved(&resolved), profile));
        }
        if common.lc_topology_replay_enabled {
            let (resolved, profile) =
                self.run_resolved_f64_with_lc_topology_replay(common, batch, None, None, true)?;
            return Ok((sum_eager_resolved(&resolved), profile));
        }
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let (source_fill, momentum_setup) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.amplitude_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);

        let eager = self.execute_profile_scheduler(None, point_count)?;
        validate_eager_profile_accounting(&eager)?;
        for value in &mut self.reduced {
            *value *= common.normalization_factor;
        }
        let values = self.reduced.clone();
        Ok((
            values,
            RuntimeProfile {
                source_fill_s: profile_duration_seconds(source_fill),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup),
                momentum_setup_s: profile_duration_seconds(momentum_setup),
                stage_evaluator_call_s: eager.total.as_secs_f64(),
                stage_evaluator_s: eager.total.as_secs_f64(),
                reduction_s: eager.reduction.as_secs_f64(),
                total_s: total_start.elapsed().as_secs_f64(),
                eager_initialize_s: eager.initialize.as_secs_f64(),
                eager_gather_s: eager.gather.as_secs_f64(),
                eager_kernel_call_s: eager.kernel_call.as_secs_f64(),
                eager_invocation_scatter_s: eager.invocation_scatter.as_secs_f64(),
                eager_finalization_s: eager.finalization.as_secs_f64(),
                eager_scatter_finalization_s: (eager.invocation_scatter + eager.finalization)
                    .as_secs_f64(),
                eager_closure_s: eager.closure.as_secs_f64(),
                eager_reduction_s: eager.reduction.as_secs_f64(),
                eager_copy_out_s: eager.copy_out.as_secs_f64(),
                eager_internal_scratch_bytes: eager.internal_scratch_bytes,
                eager_internal_broadcast_bytes: eager.internal_broadcast_bytes,
                evaluator_backend_call_count: eager.backend_call_count,
                ..RuntimeProfile::default()
            },
        ))
    }

    #[allow(dead_code)] // Retained as the row-major eager resolved-value oracle.
    pub(super) fn run_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if common.color_topology_replay_enabled {
            return self.run_resolved_f64_with_color_topology_replay(
                common,
                batch,
                selected_helicity_ids,
                selected_color_ids,
                false,
            );
        }
        if common.lc_topology_replay_enabled {
            return self.run_resolved_f64_with_lc_topology_replay(
                common,
                batch,
                selected_helicity_ids,
                selected_color_ids,
                false,
            );
        }
        self.run_resolved_f64_materialized(common, batch, selected_helicity_ids, selected_color_ids)
    }

    fn run_resolved_f64_materialized(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let (source_fill, momentum_setup) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.amplitude_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);

        let physics = common.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("resolved eager evaluation requires physics metadata")
        })?;
        let has_selected_groups = if self.has_selector_domains
            && (selected_helicity_ids.is_some() || selected_color_ids.is_some())
        {
            fill_selected_eager_group_ids(
                &self.raw_sum_groups,
                physics,
                selected_helicity_ids,
                selected_color_ids,
                &mut self.selected_groups,
            )?;
            true
        } else {
            false
        };
        let execute_start = Instant::now();
        if has_selected_groups {
            let selected_groups = std::mem::take(&mut self.selected_groups);
            let result = self.execute_selected_scheduler(&selected_groups, point_count);
            self.selected_groups = selected_groups;
            result?;
        } else {
            self.execute_full_scheduler(point_count)?;
        }
        let execute_s = execute_start.elapsed().as_secs_f64();
        let reduction_start = Instant::now();
        let resolved = reduce_eager_amplitudes_resolved(
            &self.amplitudes,
            point_count,
            &self.raw_sum_groups,
            self.color_contraction.as_mut(),
            &mut self.color_group_scratch,
            physics,
            common.normalization_factor,
            selected_helicity_ids,
            selected_color_ids,
        )?;
        let reduction_s = reduction_start.elapsed().as_secs_f64();
        Ok((
            resolved,
            RuntimeProfile {
                source_fill_s: profile_duration_seconds(source_fill),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup),
                momentum_setup_s: profile_duration_seconds(momentum_setup),
                stage_evaluator_call_s: execute_s,
                stage_evaluator_s: execute_s,
                reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                ..RuntimeProfile::default()
            },
        ))
    }

    pub(super) fn run_resolved_f64_profile(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if common.color_topology_replay_enabled {
            return self.run_resolved_f64_with_color_topology_replay(
                common,
                batch,
                selected_helicity_ids,
                selected_color_ids,
                true,
            );
        }
        if common.lc_topology_replay_enabled {
            return self.run_resolved_f64_with_lc_topology_replay(
                common,
                batch,
                selected_helicity_ids,
                selected_color_ids,
                true,
            );
        }
        self.run_resolved_f64_profile_materialized(
            common,
            batch,
            selected_helicity_ids,
            selected_color_ids,
        )
    }

    fn run_resolved_f64_profile_materialized(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let (source_fill, momentum_setup) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.amplitude_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);

        let physics = common.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("resolved eager evaluation requires physics metadata")
        })?;
        let has_selected_groups = if self.has_selector_domains
            && (selected_helicity_ids.is_some() || selected_color_ids.is_some())
        {
            fill_selected_eager_group_ids(
                &self.raw_sum_groups,
                physics,
                selected_helicity_ids,
                selected_color_ids,
                &mut self.selected_groups,
            )?;
            true
        } else {
            false
        };
        let eager = if has_selected_groups {
            let selected_groups = std::mem::take(&mut self.selected_groups);
            let result = self.execute_profile_scheduler(Some(&selected_groups), point_count);
            self.selected_groups = selected_groups;
            result?
        } else {
            self.execute_profile_scheduler(None, point_count)?
        };
        validate_eager_profile_accounting(&eager)?;
        let reduction_start = Instant::now();
        let resolved = reduce_eager_amplitudes_resolved(
            &self.amplitudes,
            point_count,
            &self.raw_sum_groups,
            self.color_contraction.as_mut(),
            &mut self.color_group_scratch,
            physics,
            common.normalization_factor,
            selected_helicity_ids,
            selected_color_ids,
        )?;
        let resolved_reduction = reduction_start.elapsed();
        let eager_execution = eager.total + resolved_reduction;
        let reduction = eager.reduction + resolved_reduction;
        Ok((
            resolved,
            RuntimeProfile {
                source_fill_s: profile_duration_seconds(source_fill),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup),
                momentum_setup_s: profile_duration_seconds(momentum_setup),
                stage_evaluator_call_s: eager_execution.as_secs_f64(),
                stage_evaluator_s: eager_execution.as_secs_f64(),
                reduction_s: reduction.as_secs_f64(),
                total_s: total_start.elapsed().as_secs_f64(),
                eager_initialize_s: eager.initialize.as_secs_f64(),
                eager_gather_s: eager.gather.as_secs_f64(),
                eager_kernel_call_s: eager.kernel_call.as_secs_f64(),
                eager_invocation_scatter_s: eager.invocation_scatter.as_secs_f64(),
                eager_finalization_s: eager.finalization.as_secs_f64(),
                eager_scatter_finalization_s: (eager.invocation_scatter + eager.finalization)
                    .as_secs_f64(),
                eager_closure_s: eager.closure.as_secs_f64(),
                eager_reduction_s: reduction.as_secs_f64(),
                eager_copy_out_s: eager.copy_out.as_secs_f64(),
                eager_internal_scratch_bytes: eager.internal_scratch_bytes,
                eager_internal_broadcast_bytes: eager.internal_broadcast_bytes,
                evaluator_backend_call_count: eager.backend_call_count,
                ..RuntimeProfile::default()
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_resolved_f64_with_color_topology_replay(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        _profiled: bool,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager color topology replay requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let eager_internal_scratch_before = self.direct_internal_scratch_bytes();
        let eager_internal_broadcast_before = self.direct_internal_broadcast_bytes();
        let point_count = batch.len();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "eager color topology replay requires resolved physics metadata",
            )
        })?;
        let mappings = Arc::clone(&common.color_topology_replay_mappings);
        let mut replay = self.color_topology_replay.take().ok_or_else(|| {
            RusticolError::integrity("eager color topology replay has no amplitude gather runtime")
        })?;
        let mut expanded_batch = std::mem::take(&mut self.lc_replay_expanded_batch);
        let mut seen_labels = std::mem::take(&mut self.lc_replay_seen_labels);
        let result = (|| {
            replay.begin_color_topology_replay(point_count)?;
            let mut source_fill_s = 0.0;
            let mut momentum_setup_s = 0.0;
            let mut evaluator_s = 0.0;
            for mapping_index in 0..mappings.len() {
                let evaluation_batch = if mappings[mapping_index].is_empty() {
                    batch
                } else {
                    let expanded_len = fill_lc_replay_expanded_batch(
                        &mut expanded_batch,
                        &mut seen_labels,
                        batch,
                        common.external_count,
                        &[mapping_index],
                        &mappings,
                    )?;
                    &expanded_batch[..expanded_len]
                };
                let (source_fill, momentum_setup) = prepare_eager_inputs(
                    common,
                    evaluation_batch,
                    &mut self.initial_values,
                    &mut self.momenta,
                    &mut self.source_schedule,
                    &mut self.source_wavefunction_scratch,
                )?;
                source_fill_s += profile_duration_seconds(source_fill);
                momentum_setup_s += profile_duration_seconds(momentum_setup);
                project_model_parameters(
                    &self.parameter_projection,
                    &common.model_parameter_values_f64,
                    &mut self.model_parameters,
                )?;
                self.amplitudes.resize(
                    point_count
                        .checked_mul(self.amplitude_count)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(
                                "eager color replay amplitudes overflow",
                            )
                        })?,
                    crate::EagerComplex64::new(0.0, 0.0),
                );
                self.reduced.resize(point_count, 0.0);
                let evaluator_start = Instant::now();
                self.execute_full_scheduler(point_count)?;
                evaluator_s += evaluator_start.elapsed().as_secs_f64();
                replay.gather_color_topology_replay_eager_amplitudes(
                    &self.amplitudes,
                    point_count,
                    mapping_index,
                )?;
            }
            let reduction_start = Instant::now();
            let resolved = replay.reduce_color_topology_replay_f64_resolved(
                point_count,
                &physics,
                common.normalization_factor,
                selected_helicity_ids,
                selected_color_ids,
            )?;
            let reduction_s = reduction_start.elapsed().as_secs_f64();
            let eager_internal_scratch_bytes = self
                .direct_internal_scratch_bytes()
                .saturating_sub(eager_internal_scratch_before);
            let eager_internal_broadcast_bytes = self
                .direct_internal_broadcast_bytes()
                .saturating_sub(eager_internal_broadcast_before);
            Ok((
                resolved,
                RuntimeProfile {
                    source_fill_s,
                    momentum_input_setup_s: momentum_setup_s,
                    momentum_setup_s,
                    stage_evaluator_call_s: evaluator_s,
                    stage_evaluator_s: evaluator_s,
                    reduction_s,
                    eager_reduction_s: reduction_s,
                    eager_internal_scratch_bytes,
                    eager_internal_broadcast_bytes,
                    total_s: total_start.elapsed().as_secs_f64(),
                    evaluator_backend_call_count: u64::try_from(mappings.len()).map_err(|_| {
                        RusticolError::invalid_argument(
                            "eager color replay mapping count exceeds u64",
                        )
                    })?,
                    ..RuntimeProfile::default()
                },
            ))
        })();
        self.lc_replay_expanded_batch = expanded_batch;
        self.lc_replay_seen_labels = seen_labels;
        self.color_topology_replay = Some(replay);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_resolved_f64_with_lc_topology_replay(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        profiled: bool,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("eager LC topology replay requires resolved physics metadata")
        })?;
        let replay_plan = if let Some(plan) = &self.lc_resolved_replay_plan {
            Arc::clone(plan)
        } else {
            let materialized_sector_by_id =
                common.lc_materialized_sectors_by_id_from_groups(&physics, &self.raw_sum_groups)?;
            let plan = Arc::new(physics.lc_resolved_replay_plan(
                &common.lc_topology_replay_public_mappings,
                &common.lc_topology_replay_routes,
                &materialized_sector_by_id,
            )?);
            self.lc_resolved_replay_plan = Some(Arc::clone(&plan));
            plan
        };
        let selection_key =
            physics.lc_resolved_replay_selection_key(selected_helicity_ids, selected_color_ids)?;
        let selection = if let Some((_, selection)) = self
            .lc_resolved_replay_selection_cache
            .iter()
            .find(|(key, _)| key == &selection_key)
        {
            Arc::clone(selection)
        } else {
            let selection = Arc::new(
                physics
                    .select_lc_resolved_replay_plan_for_key(replay_plan.as_ref(), &selection_key)?,
            );
            self.lc_resolved_replay_selection_cache
                .push((selection_key, Arc::clone(&selection)));
            selection
        };
        let total_start = Instant::now();
        let point_count = batch.len();
        let component_count = selection.helicity_indices.len() * selection.color_indices.len();
        let mut values = vec![0.0; point_count * component_count];
        // Saved SymJIT applications do not currently guarantee Rust's AArch64
        // callee-saved FP registers. Keep profile state produced by one call
        // in memory until all calls have completed instead of retaining f64
        // accumulators across the next kernel invocation.
        let mut sector_profiles = Vec::new();
        let mappings = common.lc_topology_replay_mappings.clone();
        let mappings_per_chunk = replay_mappings_per_expanded_batch(point_count);
        let mut expanded_batch = std::mem::take(&mut self.lc_replay_expanded_batch);
        let mut seen_labels = std::mem::take(&mut self.lc_replay_seen_labels);
        let result = (|| {
            for source_group in &selection.source_groups {
                for chunk_start in
                    (0..source_group.mapping_indices.len()).step_by(mappings_per_chunk)
                {
                    let chunk_end = usize::min(
                        chunk_start + mappings_per_chunk,
                        source_group.mapping_indices.len(),
                    );
                    let mapping_indices = &source_group.mapping_indices[chunk_start..chunk_end];
                    let entry_chunk = &source_group.entries[chunk_start..chunk_end];
                    let identity_mapping =
                        mapping_indices.len() == 1 && mappings[mapping_indices[0]].is_empty();
                    let evaluation_batch = if identity_mapping {
                        batch
                    } else {
                        let expanded_len = fill_lc_replay_expanded_batch(
                            &mut expanded_batch,
                            &mut seen_labels,
                            batch,
                            common.external_count,
                            mapping_indices,
                            &mappings,
                        )?;
                        &expanded_batch[..expanded_len]
                    };
                    let (materialized, sector_profile) = if profiled {
                        self.run_resolved_f64_profile_materialized(
                            common,
                            evaluation_batch,
                            Some(&source_group.helicity_ids),
                            Some(&source_group.color_ids),
                        )?
                    } else {
                        self.run_resolved_f64_materialized(
                            common,
                            evaluation_batch,
                            Some(&source_group.helicity_ids),
                            Some(&source_group.color_ids),
                        )?
                    };
                    let reduction_start = Instant::now();
                    accumulate_selected_lc_replay_resolved_f64(
                        &mut values,
                        point_count,
                        &materialized,
                        entry_chunk,
                        source_group.source_component_count,
                        component_count,
                    )?;
                    sector_profiles.push((sector_profile, reduction_start.elapsed()));
                }
            }
            let mut profile = RuntimeProfile::default();
            for (sector_profile, replay_reduction) in &sector_profiles {
                profile.add_sector(sector_profile);
                let replay_reduction_s = profile_duration_seconds(*replay_reduction);
                profile.stage_evaluator_call_s += replay_reduction_s;
                profile.stage_evaluator_s += replay_reduction_s;
                profile.reduction_s += replay_reduction_s;
                profile.eager_reduction_s += replay_reduction_s;
            }
            profile.total_s = total_start.elapsed().as_secs_f64();
            Ok((
                ResolvedValues {
                    values,
                    point_count,
                    helicity_indices: selection.helicity_indices.clone(),
                    color_indices: selection.color_indices.clone(),
                },
                profile,
            ))
        })();
        self.lc_replay_expanded_batch = expanded_batch;
        self.lc_replay_seen_labels = seen_labels;
        result
    }
}

fn fill_lc_replay_expanded_batch(
    target: &mut Vec<Vec<[f64; 4]>>,
    seen_labels: &mut Vec<bool>,
    batch: &[Vec<[f64; 4]>],
    expected_legs: usize,
    mapping_indices: &[usize],
    mappings: &[Vec<(usize, usize)>],
) -> RusticolResult<usize> {
    let required_rows = batch
        .len()
        .checked_mul(mapping_indices.len())
        .ok_or_else(|| RusticolError::invalid_argument("LC topology replay batch overflows"))?;
    if target.len() < required_rows {
        target.resize_with(required_rows, Vec::new);
    }
    seen_labels.resize(expected_legs, false);
    let mut target_row = 0;
    for mapping_index in mapping_indices {
        let mapping = mappings.get(*mapping_index).ok_or_else(|| {
            RusticolError::integrity("LC topology replay references an unknown mapping")
        })?;
        seen_labels.fill(false);
        for (representative_index, sector_index) in mapping {
            if *representative_index >= expected_legs || *sector_index >= expected_legs {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay label permutation references an out-of-range external leg",
                ));
            }
            if seen_labels[*representative_index] {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay label permutation contains a duplicate representative label",
                ));
            }
            seen_labels[*representative_index] = true;
        }
        for point in batch {
            if point.len() != expected_legs {
                return Err(RusticolError::invalid_argument(format!(
                    "LC topology replay point has {} external legs, expected {expected_legs}",
                    point.len(),
                )));
            }
            let mapped = &mut target[target_row];
            mapped.clear();
            mapped.extend_from_slice(point);
            for (representative_index, sector_index) in mapping {
                mapped[*representative_index] = point[*sector_index];
            }
            target_row += 1;
        }
    }
    Ok(required_rows)
}

fn sum_eager_resolved(resolved: &ResolvedValues<f64>) -> Vec<f64> {
    let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
    if component_count == 0 {
        return vec![0.0; resolved.point_count];
    }
    resolved
        .values
        .chunks_exact(component_count)
        .map(|components| components.iter().sum())
        .collect()
}

fn fill_selected_eager_group_ids(
    groups: &[RawSumGroup],
    physics: &PhysicsRuntime,
    selected_helicity_ids: Option<&BTreeSet<String>>,
    selected_color_ids: Option<&BTreeSet<String>>,
    active: &mut Vec<u32>,
) -> RusticolResult<()> {
    let intersects = |members: &[String], selected: Option<&BTreeSet<String>>| {
        selected.is_none_or(|selected| members.iter().any(|id| selected.contains(id)))
    };
    active.clear();
    active.try_reserve_exact(groups.len()).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "could not reserve eager active selector groups: {error}"
        ))
    })?;
    for group in groups {
        let reduction = physics
            .reduction_by_group_id
            .get(&group.id)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "resolved eager metadata is missing coherent group {}",
                    group.id
                ))
            })?;
        if intersects(&reduction.physical_helicity_ids, selected_helicity_ids)
            && intersects(&reduction.physical_color_ids, selected_color_ids)
        {
            active.push(u32::try_from(group.id).map_err(|_| {
                RusticolError::integrity(format!(
                    "eager coherent group {} does not fit the selector-domain ABI",
                    group.id
                ))
            })?);
        }
    }
    active.sort_unstable();
    active.dedup();
    Ok(())
}

#[allow(dead_code)] // Used by the order-preserving benchmark reference lane.
fn build_eager_selector_members(
    groups: &[RawSumGroup],
    contracted_color: bool,
    physics: &PhysicsRuntime,
) -> RusticolResult<Vec<Vec<(usize, usize, f64)>>> {
    groups
        .iter()
        .map(|group| {
            let reduction = physics
                .reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            if contracted_color {
                physics
                    .normalized_helicity_weights(reduction)
                    .map(|members| {
                        members
                            .into_iter()
                            .map(|(helicity, weight)| (helicity, 0, weight))
                            .collect()
                    })
            } else {
                physics.normalized_member_weights(reduction)
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // Used by the order-preserving benchmark reference lane.
fn fill_eager_group_sets_by_point(
    groups: &[RawSumGroup],
    selector_members: &[Vec<(usize, usize, f64)>],
    physics: &PhysicsRuntime,
    point_count: usize,
    selected_helicity_ids: Option<&BTreeSet<String>>,
    selected_color_ids: Option<&BTreeSet<String>>,
    helicity_by_point: Option<&[u32]>,
    color_by_point: Option<&[u32]>,
    offsets: &mut Vec<usize>,
    active_groups: &mut Vec<u32>,
    active_group_weights: &mut Vec<f64>,
    point_pairs: &mut Vec<(u32, f64)>,
    selected_helicity_indices: &mut Vec<usize>,
    selected_color_indices: &mut Vec<usize>,
) -> RusticolResult<()> {
    if selector_members.len() != groups.len() {
        return Err(RusticolError::integrity(
            "eager selector memberships do not match coherent groups",
        ));
    }
    physics.selected_helicity_indices_into(selected_helicity_ids, selected_helicity_indices)?;
    physics.selected_color_indices_into(selected_color_ids, selected_color_indices)?;
    let helicity_matches = |index: usize, point: usize| {
        helicity_by_point
            .map(|values| values[point] as usize == index)
            .unwrap_or_else(|| selected_helicity_indices.binary_search(&index).is_ok())
    };
    let color_matches = |index: usize, point: usize| {
        color_by_point
            .map(|values| values[point] as usize == index)
            .unwrap_or_else(|| selected_color_indices.binary_search(&index).is_ok())
    };

    offsets.clear();
    active_groups.clear();
    active_group_weights.clear();
    offsets.try_reserve(point_count + 1).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "could not reserve eager selector offsets: {error}"
        ))
    })?;
    offsets.push(0);
    point_pairs.clear();
    point_pairs.try_reserve(groups.len()).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "could not reserve eager selector group scratch: {error}"
        ))
    })?;
    for point in 0..point_count {
        point_pairs.clear();
        for (group, members) in groups.iter().zip(selector_members) {
            let weight = members
                .iter()
                .filter(|(helicity, color, _)| {
                    helicity_matches(*helicity, point) && color_matches(*color, point)
                })
                .map(|(_, _, weight)| *weight)
                .sum::<f64>();
            if weight == 0.0 {
                continue;
            }
            let group_id = u32::try_from(group.id).map_err(|_| {
                RusticolError::integrity(format!(
                    "eager coherent group {} does not fit the selector-domain ABI",
                    group.id
                ))
            })?;
            point_pairs.push((group_id, weight));
        }
        point_pairs.sort_unstable_by_key(|(group_id, _)| *group_id);
        if point_pairs.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(RusticolError::integrity(
                "eager physical selector maps duplicate coherent groups",
            ));
        }
        active_groups.extend(point_pairs.iter().map(|(group_id, _)| *group_id));
        active_group_weights.extend(point_pairs.iter().map(|(_, weight)| *weight));
        offsets.push(active_groups.len());
    }
    Ok(())
}

fn project_model_parameters(
    projection: &EagerParameterProjection,
    runtime_parameters: &[f64],
    output: &mut Vec<crate::EagerComplex64>,
) -> RusticolResult<()> {
    output.resize(
        projection.parameter_count,
        crate::EagerComplex64::new(0.0, 0.0),
    );
    output.fill(crate::EagerComplex64::new(0.0, 0.0));
    for entry in &projection.entries {
        let real = runtime_parameters
            .get(entry.runtime_real_index)
            .copied()
            .ok_or_else(|| {
                RusticolError::integrity("eager parameter projection has an invalid real slot")
            })?;
        let imaginary = entry
            .runtime_imaginary_index
            .map(|index| {
                runtime_parameters.get(index).copied().ok_or_else(|| {
                    RusticolError::integrity(
                        "eager parameter projection has an invalid imaginary slot",
                    )
                })
            })
            .transpose()?
            .unwrap_or(0.0);
        output[entry.prepared_index] = crate::EagerComplex64::new(real, imaginary);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reduce_eager_amplitudes_resolved_into(
    amplitudes: &[crate::EagerComplex64],
    point_count: usize,
    groups: &[RawSumGroup],
    contraction: Option<&mut ColorContractionRuntime>,
    color_group_scratch: &mut Vec<crate::EagerComplex64>,
    physics: &PhysicsRuntime,
    normalization_factor: f64,
    selected_helicity_ids: Option<&BTreeSet<String>>,
    selected_color_ids: Option<&BTreeSet<String>>,
    values: &mut Vec<f64>,
    helicity_indices: &mut Vec<usize>,
    color_indices: &mut Vec<usize>,
    helicity_position: &mut Vec<Option<usize>>,
    color_position: &mut Vec<Option<usize>>,
    helicity_weights: &mut Vec<(usize, f64)>,
    right_helicity_weights: &mut Vec<(usize, f64)>,
    member_weights: &mut Vec<(usize, usize, f64)>,
) -> RusticolResult<()> {
    physics.selected_helicity_indices_into(selected_helicity_ids, helicity_indices)?;
    physics.selected_color_indices_into(selected_color_ids, color_indices)?;
    let selected_component_count = helicity_indices
        .len()
        .checked_mul(color_indices.len())
        .ok_or_else(|| RusticolError::invalid_argument("resolved eager shape overflows"))?;
    let value_count = point_count
        .checked_mul(selected_component_count)
        .ok_or_else(|| RusticolError::invalid_argument("resolved eager shape overflows"))?;
    values.resize(value_count, 0.0);
    values.fill(0.0);

    let amplitude_count = groups
        .iter()
        .flat_map(|group| group.indices.iter().copied())
        .max()
        .map(|value| value + 1)
        .unwrap_or(0);
    let expected_amplitudes = point_count
        .checked_mul(amplitude_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager amplitude shape overflows"))?;
    if amplitudes.len() != expected_amplitudes {
        return Err(RusticolError::integrity(format!(
            "eager amplitude buffer has length {}, expected {expected_amplitudes}",
            amplitudes.len(),
        )));
    }

    helicity_position.resize(physics.manifest.helicities.len(), None);
    helicity_position.fill(None);
    for (position, index) in helicity_indices.iter().copied().enumerate() {
        let slot = helicity_position.get_mut(index).ok_or_else(|| {
            RusticolError::integrity("resolved eager helicity index is out of range")
        })?;
        *slot = Some(position);
    }
    color_position.resize(physics.manifest.color_components.len(), None);
    color_position.fill(None);
    for (position, index) in color_indices.iter().copied().enumerate() {
        let slot = color_position.get_mut(index).ok_or_else(|| {
            RusticolError::integrity("resolved eager color index is out of range")
        })?;
        *slot = Some(position);
    }

    if let Some(contraction) = contraction {
        if !physics.has_contracted_color_axis()
            || physics.manifest.color_components.len() != 1
            || contraction.group_count != groups.len()
        {
            return Err(RusticolError::integrity(
                "resolved eager color contraction does not match physics metadata",
            ));
        }
        let Some(selected_color_position) = color_position.first().copied().flatten() else {
            return Ok(());
        };
        color_group_scratch.resize(
            point_count * groups.len(),
            crate::EagerComplex64::new(0.0, 0.0),
        );
        for (group_index, group) in groups.iter().enumerate() {
            for point in 0..point_count {
                let sum = group
                    .indices
                    .iter()
                    .fold(crate::EagerComplex64::new(0.0, 0.0), |sum, index| {
                        sum + amplitudes[*index * point_count + point]
                    });
                color_group_scratch[group_index * point_count + point] = sum;
            }
        }
        for entry in contraction.logical_entries() {
            let left_group = &groups[entry.left_group_index];
            let right_group = &groups[entry.right_group_index];
            let left_reduction = physics
                .reduction_by_group_id
                .get(&left_group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        left_group.id
                    ))
                })?;
            let right_reduction = physics
                .reduction_by_group_id
                .get(&right_group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        right_group.id
                    ))
                })?;
            if left_reduction.physical_helicity_ids != right_reduction.physical_helicity_ids {
                return Err(RusticolError::integrity(
                    "resolved eager color contraction mixes distinct physical helicities",
                ));
            }
            physics.normalized_helicity_weights_into(left_reduction, helicity_weights)?;
            physics.normalized_helicity_weights_into(right_reduction, right_helicity_weights)?;
            if helicity_weights.len() != right_helicity_weights.len()
                || helicity_weights
                    .iter()
                    .zip(right_helicity_weights.iter())
                    .any(|((left_index, left_weight), (right_index, right_weight))| {
                        left_index != right_index || left_weight.to_bits() != right_weight.to_bits()
                    })
            {
                return Err(RusticolError::integrity(
                    "resolved eager color groups have inconsistent helicity weights",
                ));
            }
            for point in 0..point_count {
                let left = color_group_scratch[entry.left_group_index * point_count + point];
                let right = color_group_scratch[entry.right_group_index * point_count + point];
                let product = left * right.conj();
                let contribution = normalization_factor
                    * entry.symmetry_factor
                    * (entry.weight_re * product.re - entry.weight_im * product.im);
                for (helicity_index, weight) in helicity_weights.iter().copied() {
                    let Some(selected_helicity_position) = helicity_position[helicity_index] else {
                        continue;
                    };
                    let index = point * selected_component_count
                        + selected_helicity_position * color_indices.len()
                        + selected_color_position;
                    values[index] += contribution * weight;
                }
            }
        }
    } else {
        for group in groups {
            let reduction = physics
                .reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            physics.normalized_member_weights_into(reduction, member_weights)?;
            for point in 0..point_count {
                let sum = group
                    .indices
                    .iter()
                    .fold(crate::EagerComplex64::new(0.0, 0.0), |sum, index| {
                        sum + amplitudes[*index * point_count + point]
                    });
                let contribution = normalization_factor
                    * group.all_sector_weight
                    * (sum.re * sum.re + sum.im * sum.im);
                for (helicity_index, color_index, weight) in member_weights.iter().copied() {
                    let Some(selected_helicity_position) = helicity_position[helicity_index] else {
                        continue;
                    };
                    let Some(selected_color_position) = color_position[color_index] else {
                        continue;
                    };
                    let index = point * selected_component_count
                        + selected_helicity_position * color_indices.len()
                        + selected_color_position;
                    values[index] += contribution * weight;
                }
            }
        }
    }
    Ok(())
}

fn accumulate_selected_lc_replay_values_f64(
    target: &mut [f64],
    point_count: usize,
    materialized_values: &[f64],
    replay_entries: &[LcResolvedReplayEntry],
    source_component_count: usize,
    target_component_count: usize,
) -> RusticolResult<()> {
    let source_scalar_count = point_count
        .checked_mul(replay_entries.len())
        .and_then(|count| count.checked_mul(source_component_count))
        .ok_or_else(|| RusticolError::invalid_argument("LC replay source shape overflows"))?;
    if materialized_values.len() != source_scalar_count {
        return Err(RusticolError::integrity(format!(
            "LC topology replay materialized values have length {}, expected {source_scalar_count}",
            materialized_values.len()
        )));
    }
    let target_scalar_count = point_count
        .checked_mul(target_component_count)
        .ok_or_else(|| RusticolError::invalid_argument("LC replay target shape overflows"))?;
    if target.len() != target_scalar_count {
        return Err(RusticolError::integrity(format!(
            "LC topology replay target has length {}, expected {target_scalar_count}",
            target.len()
        )));
    }
    for (entry_index, entry) in replay_entries.iter().enumerate() {
        for point_index in 0..point_count {
            let source_row = (entry_index * point_count + point_index) * source_component_count;
            let target_row = point_index * target_component_count;
            for route in &entry.routes {
                if route.source_index >= source_component_count
                    || route.target_index >= target_component_count
                {
                    return Err(RusticolError::integrity(
                        "LC topology replay selected route is out of bounds",
                    ));
                }
                target[target_row + route.target_index] +=
                    route.weight * materialized_values[source_row + route.source_index];
            }
        }
    }
    Ok(())
}

fn lc_replay_selection_key_matches(
    key: &LcResolvedReplaySelectionKey,
    physics: &PhysicsRuntime,
    selected_helicity_ids: Option<&BTreeSet<String>>,
    selected_color_ids: Option<&BTreeSet<String>>,
) -> bool {
    let helicities_match = match (&key.helicity_indices, selected_helicity_ids) {
        (None, None) => true,
        (Some(indices), Some(ids)) if indices.len() == ids.len() => indices.iter().all(|index| {
            physics
                .manifest
                .helicities
                .get(*index)
                .is_some_and(|helicity| ids.contains(&helicity.id))
        }),
        _ => false,
    };
    let colors_match = match (&key.color_indices, selected_color_ids) {
        (None, None) => true,
        (Some(indices), Some(ids)) if indices.len() == ids.len() => indices.iter().all(|index| {
            physics
                .manifest
                .color_components
                .get(*index)
                .is_some_and(|color| ids.contains(color.id()))
        }),
        _ => false,
    };
    helicities_match && colors_match
}

fn fill_lc_replay_expanded_flat_from_view(
    target: &mut Vec<f64>,
    batch: F64MomentumBatchView<'_>,
    expected_legs: usize,
    mapping_indices: &[usize],
    mappings: &[Vec<(usize, usize)>],
) -> RusticolResult<()> {
    if batch.external_count() != expected_legs {
        return Err(RusticolError::invalid_argument(format!(
            "LC topology replay input has {} external legs, expected {expected_legs}",
            batch.external_count()
        )));
    }
    let values_per_point = expected_legs
        .checked_mul(4)
        .ok_or_else(|| RusticolError::invalid_argument("LC topology replay batch overflows"))?;
    let row_count = batch
        .point_count()
        .checked_mul(mapping_indices.len())
        .ok_or_else(|| RusticolError::invalid_argument("LC topology replay batch overflows"))?;
    let scalar_count = row_count
        .checked_mul(values_per_point)
        .ok_or_else(|| RusticolError::invalid_argument("LC topology replay batch overflows"))?;
    target.resize(scalar_count, 0.0);
    let mut target_row = 0usize;
    for mapping_index in mapping_indices {
        let mapping = mappings.get(*mapping_index).ok_or_else(|| {
            RusticolError::integrity("LC topology replay references an unknown mapping")
        })?;
        for (position, (representative_index, sector_index)) in mapping.iter().enumerate() {
            if *representative_index >= expected_legs || *sector_index >= expected_legs {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay label permutation references an out-of-range external leg",
                ));
            }
            if mapping[..position]
                .iter()
                .any(|(previous, _)| previous == representative_index)
            {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay label permutation contains a duplicate representative label",
                ));
            }
        }
        for point_index in 0..batch.point_count() {
            let point = batch.point(point_index);
            let row_start = target_row * values_per_point;
            for external_index in 0..expected_legs {
                let momentum = point.momentum(external_index).ok_or_else(|| {
                    RusticolError::integrity(
                        "validated momentum view is missing an external leg during topology replay",
                    )
                })?;
                let start = row_start + external_index * 4;
                target[start..start + 4].copy_from_slice(&momentum);
            }
            for (representative_index, sector_index) in mapping {
                let momentum = point.momentum(*sector_index).ok_or_else(|| {
                    RusticolError::integrity(
                        "validated momentum view is missing a permuted external leg",
                    )
                })?;
                let start = row_start + representative_index * 4;
                target[start..start + 4].copy_from_slice(&momentum);
            }
            target_row += 1;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reduce_eager_amplitudes_resolved(
    amplitudes: &[crate::EagerComplex64],
    point_count: usize,
    groups: &[RawSumGroup],
    contraction: Option<&mut ColorContractionRuntime>,
    color_group_scratch: &mut Vec<crate::EagerComplex64>,
    physics: &PhysicsRuntime,
    normalization_factor: f64,
    selected_helicity_ids: Option<&BTreeSet<String>>,
    selected_color_ids: Option<&BTreeSet<String>>,
) -> RusticolResult<ResolvedValues<f64>> {
    let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
    let color_indices = physics.selected_color_indices(selected_color_ids)?;
    let selected_component_count = helicity_indices
        .len()
        .checked_mul(color_indices.len())
        .ok_or_else(|| RusticolError::invalid_argument("resolved eager shape overflows"))?;
    let mut values = vec![0.0; point_count * selected_component_count];
    let amplitude_count = groups
        .iter()
        .flat_map(|group| group.indices.iter().copied())
        .max()
        .map(|value| value + 1)
        .unwrap_or(0);
    if amplitudes.len() != point_count * amplitude_count {
        return Err(RusticolError::integrity(format!(
            "eager amplitude buffer has length {}, expected {}",
            amplitudes.len(),
            point_count * amplitude_count,
        )));
    }
    let helicity_position = selected_position_map(
        physics.manifest.helicities.len(),
        &helicity_indices,
        "helicity",
    )?;
    let color_position = selected_position_map(
        physics.manifest.color_components.len(),
        &color_indices,
        "color",
    )?;

    if let Some(contraction) = contraction {
        if !physics.has_contracted_color_axis()
            || physics.manifest.color_components.len() != 1
            || contraction.group_count != groups.len()
        {
            return Err(RusticolError::integrity(
                "resolved eager color contraction does not match physics metadata",
            ));
        }
        let Some(selected_color_position) = color_position[0] else {
            return Ok(ResolvedValues {
                values,
                point_count,
                helicity_indices,
                color_indices,
            });
        };
        color_group_scratch.resize(
            point_count * groups.len(),
            crate::EagerComplex64::new(0.0, 0.0),
        );
        for (group_index, group) in groups.iter().enumerate() {
            for point in 0..point_count {
                let sum = group
                    .indices
                    .iter()
                    .fold(crate::EagerComplex64::new(0.0, 0.0), |sum, index| {
                        sum + amplitudes[*index * point_count + point]
                    });
                color_group_scratch[group_index * point_count + point] = sum;
            }
        }
        for entry in contraction.logical_entries() {
            let left_group = &groups[entry.left_group_index];
            let right_group = &groups[entry.right_group_index];
            let left_reduction = physics
                .reduction_by_group_id
                .get(&left_group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        left_group.id
                    ))
                })?;
            let right_reduction = physics
                .reduction_by_group_id
                .get(&right_group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        right_group.id
                    ))
                })?;
            if left_reduction.physical_helicity_ids != right_reduction.physical_helicity_ids {
                return Err(RusticolError::integrity(
                    "resolved eager color contraction mixes distinct physical helicities",
                ));
            }
            let left_weights = physics.normalized_helicity_weights(left_reduction)?;
            let right_weights = physics.normalized_helicity_weights(right_reduction)?;
            if left_weights.len() != right_weights.len()
                || left_weights.iter().zip(&right_weights).any(
                    |((left_index, left_weight), (right_index, right_weight))| {
                        left_index != right_index || left_weight.to_bits() != right_weight.to_bits()
                    },
                )
            {
                return Err(RusticolError::integrity(
                    "resolved eager color groups have inconsistent helicity weights",
                ));
            }
            let selected_weights = left_weights
                .into_iter()
                .filter_map(|(index, weight)| {
                    helicity_position[index].map(|position| (position, weight))
                })
                .collect::<Vec<_>>();
            if selected_weights.is_empty() {
                continue;
            }
            for point in 0..point_count {
                let left = color_group_scratch[entry.left_group_index * point_count + point];
                let right = color_group_scratch[entry.right_group_index * point_count + point];
                let product = left * right.conj();
                let contribution = normalization_factor
                    * entry.symmetry_factor
                    * (entry.weight_re * product.re - entry.weight_im * product.im);
                for (helicity_position, weight) in &selected_weights {
                    let index = point * selected_component_count
                        + *helicity_position * color_indices.len()
                        + selected_color_position;
                    values[index] += contribution * *weight;
                }
            }
        }
    } else {
        for group in groups {
            let reduction = physics
                .reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            let selected_members = physics
                .normalized_member_weights(reduction)?
                .into_iter()
                .filter_map(|(helicity_index, color_index, weight)| {
                    Some((
                        helicity_position[helicity_index]?,
                        color_position[color_index]?,
                        weight,
                    ))
                })
                .collect::<Vec<_>>();
            if selected_members.is_empty() {
                continue;
            }
            for point in 0..point_count {
                let sum = group
                    .indices
                    .iter()
                    .fold(crate::EagerComplex64::new(0.0, 0.0), |sum, index| {
                        sum + amplitudes[*index * point_count + point]
                    });
                let contribution = normalization_factor
                    * group.all_sector_weight
                    * (sum.re * sum.re + sum.im * sum.im);
                for (helicity_position, color_position, weight) in &selected_members {
                    let index = point * selected_component_count
                        + *helicity_position * color_indices.len()
                        + *color_position;
                    values[index] += contribution * *weight;
                }
            }
        }
    }

    Ok(ResolvedValues {
        values,
        point_count,
        helicity_indices,
        color_indices,
    })
}

fn selected_position_map(
    physical_count: usize,
    selected: &[usize],
    kind: &str,
) -> RusticolResult<Vec<Option<usize>>> {
    let mut positions = vec![None; physical_count];
    for (position, index) in selected.iter().copied().enumerate() {
        let slot = positions.get_mut(index).ok_or_else(|| {
            RusticolError::integrity(format!("resolved eager {kind} index is out of range"))
        })?;
        *slot = Some(position);
    }
    Ok(positions)
}

fn prepare_eager_inputs_view(
    common: &mut ExecutionRuntime,
    batch: F64MomentumBatchView<'_>,
    initial_values: &mut Vec<crate::EagerComplex64>,
    momenta: &mut Vec<f64>,
    source_schedule: &mut Option<EagerSourceSchedule>,
    source_wavefunction_scratch: &mut Vec<Complex<f64>>,
) -> RusticolResult<()> {
    if batch.external_count() != common.external_count {
        return Err(RusticolError::invalid_argument(format!(
            "eager momentum input has {} external legs, expected {}",
            batch.external_count(),
            common.external_count
        )));
    }
    let point_count = batch.point_count();
    let value_len = point_count
        .checked_mul(common.value_parameter_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager value input overflows"))?;
    let momentum_len = point_count
        .checked_mul(common.momentum_parameter_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager momentum input overflows"))?;
    initial_values.resize(value_len, crate::EagerComplex64::new(0.0, 0.0));
    momenta.resize(momentum_len, 0.0);

    if source_schedule.is_none() {
        *source_schedule = Some(build_eager_source_schedule(
            &common.sources,
            common.value_parameter_count,
        )?);
    }
    let source_schedule = source_schedule
        .as_ref()
        .expect("eager source schedule was initialized above");
    source_wavefunction_scratch.resize(source_schedule.maximum_dimension, Complex::new(0.0, 0.0));

    for point_index in 0..point_count {
        let point = batch.point(point_index);
        for class in &source_schedule.classes {
            let source = &common.sources[class.representative_index];
            let wavefunction = &mut source_wavefunction_scratch[..class.dimension];
            ExecutionRuntime::write_source_wavefunction(
                source,
                common.external_count,
                &common.particle_masses,
                &point,
                wavefunction,
            )?;
            for target_start in &class.target_component_starts {
                for (component, value) in wavefunction.iter().enumerate() {
                    initial_values[(target_start + component) * point_count + point_index] =
                        crate::EagerComplex64::new(value.re, value.im);
                }
            }
        }
    }

    for point_index in 0..point_count {
        let point = batch.point(point_index);
        for slot in &common.momentum_slots {
            let start = slot.component_start;
            let stop = slot.component_stop;
            if stop > common.momentum_parameter_count || stop < start || stop - start != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic momentum slot {} has an invalid component range",
                    slot.momentum_slot_id
                )));
            }
            let mut momentum = [0.0; 4];
            for label in &slot.external_labels {
                let index = label.checked_sub(1).ok_or_else(|| {
                    RusticolError::invalid_argument("generic momentum labels are one-based")
                })?;
                if index >= common.external_count || index >= common.external_is_initial.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "generic momentum slot {} refers to unknown external label {}",
                        slot.momentum_slot_id, label
                    )));
                }
                let source_momentum = point.momentum(index).ok_or_else(|| {
                    RusticolError::integrity(
                        "validated eager momentum view is missing an external leg",
                    )
                })?;
                let sign = if common.external_is_initial[index] {
                    -1.0
                } else {
                    1.0
                };
                for (output, input) in momentum.iter_mut().zip(source_momentum) {
                    *output += sign * input;
                }
            }
            for (component, value) in momentum.into_iter().enumerate() {
                momenta[(start + component) * point_count + point_index] = value;
            }
        }
    }
    Ok(())
}

fn prepare_eager_inputs(
    common: &mut ExecutionRuntime,
    batch: &[Vec<[f64; 4]>],
    initial_values: &mut Vec<crate::EagerComplex64>,
    momenta: &mut Vec<f64>,
    source_schedule: &mut Option<EagerSourceSchedule>,
    source_wavefunction_scratch: &mut Vec<Complex<f64>>,
) -> RusticolResult<(Duration, Duration)> {
    let point_count = batch.len();
    let value_len = point_count
        .checked_mul(common.value_parameter_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager value input overflows"))?;
    let momentum_len = point_count
        .checked_mul(common.momentum_parameter_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager momentum input overflows"))?;
    initial_values.resize(value_len, crate::EagerComplex64::new(0.0, 0.0));
    momenta.resize(momentum_len, 0.0);

    if source_schedule.is_none() {
        *source_schedule = Some(build_eager_source_schedule(
            &common.sources,
            common.value_parameter_count,
        )?);
    }
    let source_schedule = source_schedule
        .as_ref()
        .expect("eager source schedule was initialized above");
    source_wavefunction_scratch.resize(source_schedule.maximum_dimension, Complex::new(0.0, 0.0));

    let source_started = Instant::now();
    for (point_index, point) in batch.iter().enumerate() {
        for class in &source_schedule.classes {
            let source = &common.sources[class.representative_index];
            let wavefunction = &mut source_wavefunction_scratch[..class.dimension];
            ExecutionRuntime::write_source_wavefunction(
                source,
                common.external_count,
                &common.particle_masses,
                point,
                wavefunction,
            )?;
            for target_start in &class.target_component_starts {
                for (component, value) in wavefunction.iter().enumerate() {
                    initial_values[(target_start + component) * point_count + point_index] =
                        crate::EagerComplex64::new(value.re, value.im);
                }
            }
        }
    }
    let source_fill = source_started.elapsed();

    let momentum_started = Instant::now();
    for (point_index, point) in batch.iter().enumerate() {
        for slot in &common.momentum_slots {
            let start = slot.component_start;
            let stop = slot.component_stop;
            if stop > common.momentum_parameter_count || stop < start || stop - start != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic momentum slot {} has an invalid component range",
                    slot.momentum_slot_id
                )));
            }
            let mut momentum = [0.0; 4];
            for label in &slot.external_labels {
                let index = label.checked_sub(1).ok_or_else(|| {
                    RusticolError::invalid_argument("generic momentum labels are one-based")
                })?;
                if index >= common.external_count || index >= common.external_is_initial.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "generic momentum slot {} refers to unknown external label {}",
                        slot.momentum_slot_id, label
                    )));
                }
                let sign = if common.external_is_initial[index] {
                    -1.0
                } else {
                    1.0
                };
                for (output, input) in momentum.iter_mut().zip(&point[index]) {
                    *output += sign * input;
                }
            }
            for (component, value) in momentum.into_iter().enumerate() {
                momenta[(start + component) * point_count + point_index] = value;
            }
        }
    }
    Ok((source_fill, momentum_started.elapsed()))
}

fn build_eager_source_schedule(
    sources: &[GenericSourceRecordManifest],
    value_component_count: usize,
) -> RusticolResult<EagerSourceSchedule> {
    let mut class_by_key: BTreeMap<EagerSourceKey, usize> = BTreeMap::new();
    let mut classes: Vec<EagerSourceClass> = Vec::new();
    let mut maximum_dimension = 0usize;
    for (source_index, source) in sources.iter().enumerate() {
        let start = source.value_slot.component_start;
        let stop = source.value_slot.component_stop;
        if stop > value_component_count || stop < start || stop - start != source.dimension {
            return Err(RusticolError::artifact(format!(
                "eager source {} has an invalid value-slot range",
                source.source_id
            )));
        }
        let key = eager_source_key(source);
        if let Some(class_index) = class_by_key.get(&key).copied() {
            let class = &mut classes[class_index];
            if class.dimension != source.dimension {
                return Err(RusticolError::internal(
                    "equivalent eager sources have inconsistent dimensions",
                ));
            }
            class.target_component_starts.push(start);
        } else {
            let class_index = classes.len();
            class_by_key.insert(key, class_index);
            classes.push(EagerSourceClass {
                representative_index: source_index,
                target_component_starts: vec![start],
                dimension: source.dimension,
            });
        }
        maximum_dimension = maximum_dimension.max(source.dimension);
    }
    Ok(EagerSourceSchedule {
        classes,
        maximum_dimension,
    })
}

fn eager_source_key(source: &GenericSourceRecordManifest) -> EagerSourceKey {
    EagerSourceKey {
        source_kind: source.source_kind.clone(),
        leg_label: source.leg_label,
        family: match source.source_ir.wavefunction_family {
            GenericWavefunctionFamilyManifest::Scalar => 0,
            GenericWavefunctionFamilyManifest::Fermion => 1,
            GenericWavefunctionFamilyManifest::Vector => 2,
            GenericWavefunctionFamilyManifest::Spin2 => 3,
            GenericWavefunctionFamilyManifest::Ghost => 4,
            GenericWavefunctionFamilyManifest::Auxiliary => 5,
        },
        dimension: source.source_ir.component_dimension,
        pdg_label: source.source_ir.identity.pdg_label,
        anti_pdg_label: source.source_ir.identity.anti_pdg_label,
        orientation: match source.source_ir.identity.orientation {
            GenericSourceOrientationManifest::Particle => 0,
            GenericSourceOrientationManifest::Antiparticle => 1,
            GenericSourceOrientationManifest::SelfConjugate => 2,
        },
        momentum_transform: match source.applied_crossing.momentum_transform {
            GenericMomentumTransformManifest::Identity => 0,
            GenericMomentumTransformManifest::NegateFourMomentum => 1,
        },
        phase_real: source.applied_crossing.phase[0].to_bits(),
        phase_imaginary: source.applied_crossing.phase[1].to_bits(),
        source_helicity: source.source_helicity,
        chirality: source.chirality,
    }
}

fn validate_eager_profile_accounting(
    profile: &crate::eager_runtime::EagerExecutionProfile,
) -> RusticolResult<()> {
    let accounted = profile.accounted();
    let tolerance = Duration::from_nanos(1);
    if accounted > profile.total.saturating_add(tolerance) {
        return Err(RusticolError::internal(format!(
            "eager runtime profile phases account for {accounted_s:.9e}s, exceeding the aggregate {total_s:.9e}s: {profile:?}",
            accounted_s = accounted.as_secs_f64(),
            total_s = profile.total.as_secs_f64(),
        )));
    }
    Ok(())
}

#[inline(always)]
#[cfg(test)]
fn store_complex_row_component_major(
    row: &[Complex<f64>],
    point_index: usize,
    point_count: usize,
    output: &mut [crate::EagerComplex64],
) {
    debug_assert!(point_index < point_count);
    debug_assert_eq!(output.len(), row.len() * point_count);
    for (component, value) in row.iter().enumerate() {
        output[component * point_count + point_index] =
            crate::EagerComplex64::new(value.re, value.im);
    }
}

#[inline(always)]
#[cfg(test)]
fn store_real_row_component_major(
    row: &[Complex<f64>],
    point_index: usize,
    point_count: usize,
    output: &mut [f64],
) -> RusticolResult<()> {
    debug_assert!(point_index < point_count);
    debug_assert_eq!(output.len(), row.len() * point_count);
    for (component, value) in row.iter().enumerate() {
        if value.im != 0.0 {
            return Err(RusticolError::internal(
                "eager momentum setup produced an imaginary component",
            ));
        }
        output[component * point_count + point_index] = value.re;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eager_input_rows_are_stored_component_major() {
        let point_count = 2;
        let mut complex = vec![crate::EagerComplex64::new(0.0, 0.0); 6];
        let mut real = vec![0.0; 6];
        for (point, values) in [
            [
                Complex::new(1.0, -1.0),
                Complex::new(2.0, -2.0),
                Complex::new(3.0, -3.0),
            ],
            [
                Complex::new(4.0, -4.0),
                Complex::new(5.0, -5.0),
                Complex::new(6.0, -6.0),
            ],
        ]
        .iter()
        .enumerate()
        {
            store_complex_row_component_major(values, point, point_count, &mut complex);
            let real_values = values.map(|value| Complex::new(value.re, 0.0));
            store_real_row_component_major(&real_values, point, point_count, &mut real).unwrap();
        }
        assert_eq!(
            complex,
            [
                crate::EagerComplex64::new(1.0, -1.0),
                crate::EagerComplex64::new(4.0, -4.0),
                crate::EagerComplex64::new(2.0, -2.0),
                crate::EagerComplex64::new(5.0, -5.0),
                crate::EagerComplex64::new(3.0, -3.0),
                crate::EagerComplex64::new(6.0, -6.0),
            ]
        );
        assert_eq!(real, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn eager_real_input_rejects_imaginary_components() {
        let error = store_real_row_component_major(&[Complex::new(1.0, 1.0)], 0, 1, &mut [0.0])
            .unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
    }

    #[test]
    fn eager_profile_rejects_impossible_phase_totals() {
        let valid = crate::eager_runtime::EagerExecutionProfile {
            initialize: Duration::from_millis(2),
            gather: Duration::from_millis(3),
            kernel_call: Duration::from_millis(4),
            total: Duration::from_millis(10),
            ..crate::eager_runtime::EagerExecutionProfile::default()
        };
        validate_eager_profile_accounting(&valid).unwrap();

        let impossible = crate::eager_runtime::EagerExecutionProfile {
            invocation_scatter: Duration::from_millis(20),
            total: Duration::from_millis(10),
            ..crate::eager_runtime::EagerExecutionProfile::default()
        };
        let error = validate_eager_profile_accounting(&impossible).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
        assert!(error.to_string().contains("exceeding the aggregate"));
    }
}
