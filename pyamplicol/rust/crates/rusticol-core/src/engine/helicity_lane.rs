// SPDX-License-Identifier: 0BSD

use super::sources::RuntimeSourceState;
use super::*;

#[derive(Clone, Debug)]
pub(super) struct CompiledHelicitySelectorSchedule {
    pub(super) selector_domain_id: usize,
    pub(super) physical_helicity_index: usize,
    selected_helicity_ids: BTreeSet<String>,
    pub(super) source_states: Vec<RuntimeSourceState>,
    pub(super) active_stage_chunk_indices: Vec<Vec<usize>>,
    pub(super) active_amplitude_chunk_indices: Vec<usize>,
    pub(super) root_factors: Vec<Option<Complex<f64>>>,
    pub(super) structural_zero: bool,
}

#[derive(Clone, Debug)]
pub(super) struct CompiledHelicityExecutionPlan {
    schedules_by_physical_helicity: Vec<Option<Arc<CompiledHelicitySelectorSchedule>>>,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn bind_direct_helicity_color_schedule(
    direct: &mut compiled_direct_prototype::CompiledDirectEnginePrototype,
    helicity_schedules: &BTreeMap<
        usize,
        compiled_direct_prototype::CompiledDirectValidatedSchedule,
    >,
    color_schedules: &BTreeMap<i64, compiled_direct_prototype::CompiledDirectValidatedSchedule>,
    schedule: &CompiledHelicitySelectorSchedule,
    selected: Option<&BTreeSet<i64>>,
) -> RusticolResult<Option<Arc<compiled_direct_prototype::CompiledDirectValidatedSchedule>>> {
    let Some(selected) = selected else {
        return Ok(None);
    };
    if selected.is_empty() {
        return Err(RusticolError::invalid_argument(
            "compiled Direct-Arena materialized-sector selection cannot be empty",
        ));
    }
    let helicity = helicity_schedules
        .get(&schedule.selector_domain_id)
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "compiled Direct-Arena has no cold-bound helicity schedule for selector domain {}",
                schedule.selector_domain_id
            ))
        })?;
    direct
        .bind_helicity_color_schedule(
            schedule.selector_domain_id,
            helicity,
            selected,
            color_schedules,
        )
        .map(Some)
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn evaluate_direct_helicity_schedule(
    direct: &mut compiled_direct_prototype::CompiledDirectEnginePrototype,
    schedules: &BTreeMap<usize, compiled_direct_prototype::CompiledDirectValidatedSchedule>,
    schedule: &CompiledHelicitySelectorSchedule,
    color_schedule: Option<&compiled_direct_prototype::CompiledDirectValidatedSchedule>,
    execute_union: bool,
    point_count: usize,
) -> RusticolResult<()> {
    if execute_union {
        return direct.evaluate_all(point_count);
    }
    let direct_schedule = color_schedule
        .or_else(|| schedules.get(&schedule.selector_domain_id))
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "compiled Direct-Arena has no cold-bound helicity schedule for selector domain {}",
                schedule.selector_domain_id
            ))
        })?;
    direct.evaluate_validated_materialized_helicity(point_count, direct_schedule)
}

impl ExecutionRuntime {
    fn supports_materialized_helicity_totals(&self) -> bool {
        let Some(amplitude) = self.amplitude_stage.as_ref() else {
            return false;
        };
        if amplitude.color_contraction.is_none() {
            return true;
        }
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            self.compiled_direct_runtime.is_some()
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            false
        }
    }

    pub(super) fn initialize_compiled_helicity_execution_plan(
        &mut self,
        public_label_permutation: Option<&[usize]>,
    ) -> RusticolResult<()> {
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            sum_runtime.initialize_compiled_helicity_execution_plan(public_label_permutation)?;
        }
        for selector_runtime in &mut self.helicity_selector_runtimes {
            selector_runtime
                .initialize_compiled_helicity_execution_plan(public_label_permutation)?;
        }
        let Some(recurrence) = self.helicity_recurrence.as_ref() else {
            self.compiled_helicity_execution_plan = None;
            return Ok(());
        };
        let Some(materialization) = recurrence.materialization.as_ref() else {
            self.compiled_helicity_execution_plan = None;
            return Ok(());
        };
        let physics = self.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact(
                "helicity recurrence materialization requires resolved physics metadata",
            )
        })?;
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return Err(self.execution_unavailable_error());
        }

        let mut schedules_by_physical_helicity = vec![None; physics.manifest.helicities.len()];
        let mut schedule_by_domain = vec![None; recurrence.selector_domains.len()];
        let mut physical_helicity_by_domain = vec![None; recurrence.selector_domains.len()];
        for schedule in &materialization.selector_schedules {
            let domain = recurrence
                .selector_domains
                .get(schedule.selector_domain_id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "helicity schedule references unknown selector domain {}",
                        schedule.selector_domain_id
                    ))
                })?;
            if !domain.complete {
                return Err(RusticolError::integrity(format!(
                    "helicity schedule domain {} is not complete",
                    schedule.selector_domain_id
                )));
            }
            let physical_helicity_index =
                physical_helicity_index_for_domain(physics, domain, public_label_permutation)?;
            if physical_helicity_by_domain[schedule.selector_domain_id]
                .replace(physical_helicity_index)
                .is_some()
            {
                return Err(RusticolError::integrity(format!(
                    "helicity selector domain {} has multiple recurrence schedules",
                    schedule.selector_domain_id
                )));
            }

            let source_states = if schedule.structural_zero {
                Vec::new()
            } else {
                build_materialized_source_states(
                    &self.sources,
                    &recurrence.selector_domains,
                    &materialization.source_routes,
                    domain,
                    &schedule.active_current_ids,
                )?
            };
            let root_factors = materialized_root_factors_for_domain(
                self.amplitude_output_count,
                &materialization.amplitude_routes,
                schedule,
            )?;
            schedule_by_domain[schedule.selector_domain_id] =
                Some(Arc::new(CompiledHelicitySelectorSchedule {
                    selector_domain_id: schedule.selector_domain_id,
                    physical_helicity_index,
                    selected_helicity_ids: BTreeSet::from([physics.manifest.helicities
                        [physical_helicity_index]
                        .id
                        .clone()]),
                    source_states,
                    active_stage_chunk_indices: schedule.active_stage_chunk_indices.clone(),
                    active_amplitude_chunk_indices: schedule.active_amplitude_chunk_indices.clone(),
                    root_factors,
                    structural_zero: schedule.structural_zero,
                }));
        }
        let execution_domain_by_domain = helicity_execution_domain_by_domain(
            recurrence.selector_domains.len(),
            &materialization.amplitude_routes,
        )?;
        for (domain_id, execution_candidate) in schedule_by_domain.iter().enumerate() {
            if execution_candidate.is_none() {
                continue;
            }
            let physical_helicity_index =
                physical_helicity_by_domain[domain_id].ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "helicity selector domain {domain_id} has no physical-helicity mapping"
                    ))
                })?;
            let execution_domain_id = execution_domain_by_domain[domain_id];
            let execution_schedule = schedule_by_domain
                .get(execution_domain_id)
                .and_then(Option::as_ref)
                .cloned()
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "helicity selector domain {domain_id} maps to absent execution domain {execution_domain_id}"
                    ))
                })?;
            if schedules_by_physical_helicity[physical_helicity_index]
                .replace(execution_schedule)
                .is_some()
            {
                return Err(RusticolError::integrity(format!(
                    "physical helicity {} has multiple recurrence schedules",
                    physics.manifest.helicities[physical_helicity_index].id
                )));
            }
        }
        if schedules_by_physical_helicity.iter().any(Option::is_none) {
            return Err(RusticolError::integrity(
                "helicity recurrence schedules do not cover every physical helicity",
            ));
        }
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            let mut direct_schedules = BTreeMap::new();
            if let Some(direct) = self.compiled_direct_runtime.as_ref() {
                for schedule in schedules_by_physical_helicity.iter().flatten() {
                    if schedule.structural_zero
                        || direct_schedules.contains_key(&schedule.selector_domain_id)
                    {
                        continue;
                    }
                    direct_schedules.insert(
                        schedule.selector_domain_id,
                        direct.bind_helicity_schedule(schedule.as_ref())?,
                    );
                }
            }
            self.compiled_direct_helicity_schedules = direct_schedules;
        }
        self.compiled_helicity_execution_plan = Some(CompiledHelicityExecutionPlan {
            schedules_by_physical_helicity,
        });
        Ok(())
    }

    pub(super) fn has_compiled_helicity_execution_plan(&self) -> bool {
        self.compiled_helicity_execution_plan.is_some()
    }

    /// Whether a singleton physical-helicity request can bypass a compact
    /// helicity-sum/topology-replay execution without losing physical colour.
    ///
    /// A nested runtime owns a complete non-replay colour DAG. When a proper
    /// subset of LC flows is requested, prefer the compact topology replay:
    /// that path maps both the flow and its helicity to the representative.
    /// The complete-colour nested runtime remains useful for all-flow calls.
    /// A local materialized schedule is safe only when this runtime is itself
    /// not a replay representative. Parent closures are deliberately not
    /// considered here: replay-backed closures bypass the physical topology
    /// contraction.
    pub(super) fn has_safe_singleton_helicity_execution(
        &self,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<bool> {
        let Some(selected_helicity_ids) = selected_helicity_ids else {
            return Ok(false);
        };
        if selected_helicity_ids.len() != 1 {
            return Ok(false);
        }
        let Some(plan) = self.compiled_helicity_execution_plan.as_ref() else {
            return Ok(false);
        };
        let physics = self.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact(
                "helicity recurrence materialization requires resolved physics metadata",
            )
        })?;
        let selected_helicity_id = selected_helicity_ids
            .iter()
            .next()
            .expect("singleton helicity selection checked");
        let helicity_index = *physics
            .helicity_index_by_id
            .get(selected_helicity_id)
            .ok_or_else(|| {
                RusticolError::selector(format!(
                    "unknown resolved helicity id {selected_helicity_id:?}"
                ))
            })?;
        let schedule = plan
            .schedules_by_physical_helicity
            .get(helicity_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "physical helicity {} has no compiled recurrence schedule",
                    physics.manifest.helicities[helicity_index].id
                ))
            })?;
        if schedule.structural_zero {
            return Ok(true);
        }
        if self.lc_topology_replay_enabled
            && !physics.has_contracted_color_axis()
            && physics
                .canonical_total_color_selector(selected_color_ids)
                .is_some()
        {
            return Ok(false);
        }
        if self
            .helicity_selector_lane_by_domain
            .get(&schedule.selector_domain_id)
            .and_then(|lane_index| {
                self.helicity_selector_runtime_schedule_modes
                    .get(*lane_index)
            })
            == Some(&HelicitySelectorScheduleMode::NestedRuntime)
        {
            return Ok(true);
        }
        Ok(!self.color_topology_replay_enabled && !self.lc_topology_replay_enabled)
    }

    pub(super) fn compiled_color_selector_schedule(
        &self,
        materialized_sector_ids: &BTreeSet<i64>,
    ) -> RusticolResult<Option<CompiledColorSelectorSchedule>> {
        let Some(plan) = self.compiled_color_execution_plan.as_ref() else {
            return Ok(None);
        };
        if materialized_sector_ids.is_empty() {
            return Err(RusticolError::invalid_argument(
                "compiled color selection requires at least one materialized sector",
            ));
        }
        let mut active_stage_chunks = vec![BTreeSet::new(); self.stage_count];
        let mut active_amplitude_chunks = BTreeSet::new();
        for sector_id in materialized_sector_ids {
            let schedule = plan
                .schedules_by_materialized_sector
                .get(sector_id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "compiled color selector has no schedule for materialized sector {sector_id}"
                    ))
                })?;
            if schedule.active_stage_chunk_indices.len() != active_stage_chunks.len() {
                return Err(RusticolError::integrity(format!(
                    "compiled color selector sector {sector_id} has inconsistent stage coverage"
                )));
            }
            for (target, source) in active_stage_chunks
                .iter_mut()
                .zip(&schedule.active_stage_chunk_indices)
            {
                target.extend(source.iter().copied());
            }
            active_amplitude_chunks.extend(schedule.active_amplitude_chunk_indices.iter().copied());
        }
        Ok(Some(CompiledColorSelectorSchedule {
            active_stage_chunk_indices: active_stage_chunks
                .into_iter()
                .map(|indices| indices.into_iter().collect())
                .collect(),
            active_amplitude_chunk_indices: active_amplitude_chunks.into_iter().collect(),
        }))
    }

    /// Direct totals lane for the common runtime selector shape: one physical
    /// helicity and any manifest-ordered subset of diagonal LC colours.
    ///
    /// Broader helicity sums retain the resolved fallback because adding a
    /// pre-summed colour row to a previous helicity would change the flattened
    /// H-major/C-minor floating-point fold.
    pub(super) fn try_run_f64_with_helicity_recurrence_into_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<bool> {
        if output.len() != batch.point_count() {
            return Err(RusticolError::invalid_argument(format!(
                "evaluation output has length {}, expected {}",
                output.len(),
                batch.point_count()
            )));
        }
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "helicity recurrence materialization requires resolved physics metadata",
            )
        })?;
        let Some(selected_helicity_ids) = selected_helicity_ids else {
            return Ok(false);
        };
        if selected_helicity_ids.len() != 1 {
            return Ok(false);
        }
        let selected_helicity_id = selected_helicity_ids
            .iter()
            .next()
            .expect("singleton helicity selection checked");
        let helicity_index = *physics
            .helicity_index_by_id
            .get(selected_helicity_id)
            .ok_or_else(|| {
                RusticolError::selector(format!(
                    "unknown resolved helicity id {selected_helicity_id:?}"
                ))
            })?;
        if let Some(selected_color_ids) = selected_color_ids {
            for id in selected_color_ids {
                if !physics.color_index_by_id.contains_key(id) {
                    return Err(RusticolError::selector(format!(
                        "unknown resolved color component id {id:?}"
                    )));
                }
            }
        }
        let schedule = self
            .compiled_helicity_execution_plan
            .as_ref()
            .and_then(|plan| {
                plan.schedules_by_physical_helicity
                    .get(helicity_index)
                    .and_then(Option::as_ref)
            })
            .cloned()
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "physical helicity {} has no compiled recurrence schedule",
                    physics.manifest.helicities[helicity_index].id
                ))
            })?;
        output.fill(0.0);
        if schedule.structural_zero {
            return Ok(true);
        }

        let selector_lane_index = self
            .helicity_selector_lane_by_domain
            .get(&schedule.selector_domain_id)
            .copied()
            .filter(|lane_index| {
                self.helicity_selector_runtime_schedule_modes
                    .get(*lane_index)
                    .is_some_and(|schedule_mode| {
                        *schedule_mode == HelicitySelectorScheduleMode::NestedRuntime
                            || (*schedule_mode == HelicitySelectorScheduleMode::ParentClosure
                                && selected_color_ids.is_none()
                                && !physics.has_contracted_color_axis()
                                && !self.color_topology_replay_enabled
                                && !self.lc_topology_replay_enabled)
                    })
            });
        if let Some(lane_index) = selector_lane_index {
            let schedule_mode = self
                .helicity_selector_runtime_schedule_modes
                .get(lane_index)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "helicity selector domain {} maps to missing schedule mode for lane {lane_index}",
                        schedule.selector_domain_id
                    ))
                })?;
            let selector_runtime = self
                .helicity_selector_runtimes
                .get_mut(lane_index)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "helicity selector domain {} maps to missing execution lane {lane_index}",
                        schedule.selector_domain_id
                    ))
                })?;
            match schedule_mode {
                HelicitySelectorScheduleMode::ParentClosure => {
                    if !selector_runtime.supports_materialized_helicity_totals() {
                        return Ok(false);
                    }
                    selector_runtime.run_f64_materialized_helicity_schedule_add_into_unprofiled(
                        batch, &physics, None, None, &schedule, true, output,
                    )?;
                }
                HelicitySelectorScheduleMode::NestedRuntime => {
                    selector_runtime.run_f64_selected_into_unprofiled(
                        batch,
                        Some(&schedule.selected_helicity_ids),
                        selected_color_ids,
                        output,
                    )?;
                }
            }
            return Ok(true);
        }

        if self.color_topology_replay_enabled || self.lc_topology_replay_enabled {
            return Ok(false);
        }
        if !self.supports_materialized_helicity_totals() {
            return Ok(false);
        }
        self.run_f64_materialized_helicity_schedule_add_into_unprofiled(
            batch,
            &physics,
            selected_color_ids,
            None,
            &schedule,
            false,
            output,
        )?;
        Ok(true)
    }

    pub(super) fn run_resolved_f64_with_helicity_recurrence_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "helicity recurrence materialization requires resolved physics metadata",
            )
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let point_count = batch.point_count();
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        let mut values = vec![0.0; point_count * component_count];
        for (helicity_position, helicity_index) in helicity_indices.iter().copied().enumerate() {
            let schedule = self
                .compiled_helicity_execution_plan
                .as_ref()
                .and_then(|plan| {
                    plan.schedules_by_physical_helicity
                        .get(helicity_index)
                        .and_then(Option::as_ref)
                })
                .cloned()
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "physical helicity {} has no compiled recurrence schedule",
                        physics.manifest.helicities[helicity_index].id
                    ))
                })?;
            if schedule.structural_zero {
                continue;
            }
            let selector_lane_index = self
                .helicity_selector_lane_by_domain
                .get(&schedule.selector_domain_id)
                .copied()
                .filter(|lane_index| {
                    self.helicity_selector_runtime_schedule_modes
                        .get(*lane_index)
                        .is_some_and(|schedule_mode| {
                            (*schedule_mode == HelicitySelectorScheduleMode::NestedRuntime
                                && selected_materialized_sector_ids.is_none())
                                || (*schedule_mode == HelicitySelectorScheduleMode::ParentClosure
                                    && selected_color_ids.is_none()
                                    && selected_materialized_sector_ids.is_none()
                                    && !physics.has_contracted_color_axis()
                                    && !self.color_topology_replay_enabled
                                    && !self.lc_topology_replay_enabled)
                        })
                });
            let mut selected = if let Some(lane_index) = selector_lane_index {
                let schedule_mode = self
                    .helicity_selector_runtime_schedule_modes
                    .get(lane_index)
                    .copied()
                    .ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "helicity selector domain {} maps to missing schedule mode for lane {lane_index}",
                            schedule.selector_domain_id
                        ))
                    })?;
                let selector_runtime = self
                    .helicity_selector_runtimes
                    .get_mut(lane_index)
                    .ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "helicity selector domain {} maps to missing execution lane {lane_index}",
                            schedule.selector_domain_id
                        ))
                    })?;
                match schedule_mode {
                    HelicitySelectorScheduleMode::ParentClosure => selector_runtime
                        .run_f64_materialized_helicity_schedule_unprofiled(
                            batch, &physics, None, None, &schedule, true,
                        )?,
                    HelicitySelectorScheduleMode::NestedRuntime => {
                        let selected_helicity = BTreeSet::from([physics.manifest.helicities
                            [schedule.physical_helicity_index]
                            .id
                            .clone()]);
                        selector_runtime.run_resolved_f64_unprofiled(
                            batch,
                            Some(&selected_helicity),
                            selected_color_ids,
                        )?
                    }
                }
            } else {
                if self.color_topology_replay_enabled
                    || (self.lc_topology_replay_enabled
                        && selected_materialized_sector_ids.is_none())
                {
                    return Err(RusticolError::integrity(
                        "topology-replay runtime cannot execute its representative materialized helicity schedule directly",
                    ));
                }
                self.run_f64_materialized_helicity_schedule_unprofiled(
                    batch,
                    &physics,
                    selected_color_ids,
                    selected_materialized_sector_ids,
                    &schedule,
                    false,
                )?
            };
            if selected.helicity_indices != [schedule.physical_helicity_index]
                || selected.color_indices != color_indices
                || selected.point_count != point_count
            {
                return Err(RusticolError::integrity(
                    "materialized helicity schedule returned an inconsistent resolved shape",
                ));
            }
            selected.helicity_indices[0] = helicity_index;
            if helicity_indices.len() == 1 {
                return Ok(selected);
            }
            for point_index in 0..point_count {
                let source_start = point_index * color_indices.len();
                let target_start =
                    point_index * component_count + helicity_position * color_indices.len();
                values[target_start..target_start + color_indices.len()].copy_from_slice(
                    &selected.values[source_start..source_start + color_indices.len()],
                );
            }
        }
        Ok(ResolvedValues {
            values,
            point_count,
            helicity_indices,
            color_indices,
        })
    }

    fn run_f64_materialized_helicity_schedule_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        physics: &PhysicsRuntime,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
        schedule: &CompiledHelicitySelectorSchedule,
        execute_union: bool,
    ) -> RusticolResult<ResolvedValues<f64>> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if self.compiled_direct_runtime.is_some() {
            return self.run_f64_materialized_helicity_direct_resolved_unprofiled(
                batch,
                physics,
                selected_color_ids,
                selected_materialized_sector_ids,
                schedule,
                execute_union,
            );
        }
        self.evaluate_f64_materialized_helicity_schedule_unprofiled(
            batch,
            selected_materialized_sector_ids,
            schedule,
            execute_union,
        )?;
        self.amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .reduce_scratch_f64_for_materialized_helicity(
                batch.point_count(),
                physics,
                self.normalization_factor,
                schedule.physical_helicity_index,
                &schedule.root_factors,
                selected_color_ids,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_f64_materialized_helicity_schedule_add_into_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        physics: &PhysicsRuntime,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
        schedule: &CompiledHelicitySelectorSchedule,
        execute_union: bool,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if self.compiled_direct_runtime.is_some() {
            return self.run_f64_materialized_helicity_direct_add_into_unprofiled(
                batch,
                physics,
                selected_color_ids,
                selected_materialized_sector_ids,
                schedule,
                execute_union,
                output,
            );
        }
        self.evaluate_f64_materialized_helicity_schedule_unprofiled(
            batch,
            selected_materialized_sector_ids,
            schedule,
            execute_union,
        )?;
        self.amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .reduce_scratch_f64_for_materialized_helicity_add_into(
                batch.point_count(),
                physics,
                self.normalization_factor,
                schedule.physical_helicity_index,
                &schedule.root_factors,
                selected_color_ids,
                output,
            )
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[allow(clippy::too_many_arguments)]
    fn run_f64_materialized_helicity_direct_resolved_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        physics: &PhysicsRuntime,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
        schedule: &CompiledHelicitySelectorSchedule,
        execute_union: bool,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let point_count = batch.point_count();
        let mut values = vec![0.0; point_count * color_indices.len()];
        let Self {
            compiled_direct_runtime: Some(direct),
            compiled_direct_color_schedules,
            compiled_direct_helicity_schedules,
            sources,
            external_count,
            particle_masses,
            momentum_slots,
            external_is_initial,
            model_parameter_values_f64,
            amplitude_stage: Some(amplitude),
            normalization_factor,
            ..
        } = self
        else {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena helicity execution was selected without a complete runtime",
            ));
        };
        let color_schedule = bind_direct_helicity_color_schedule(
            direct,
            compiled_direct_helicity_schedules,
            compiled_direct_color_schedules,
            schedule,
            selected_materialized_sector_ids,
        )?;
        let tile_capacity = direct.reduction_tile_capacity();
        let mut point_start = 0usize;
        while point_start < point_count {
            let point_stop = (point_start + tile_capacity).min(point_count);
            direct.begin_tile_from_inputs(
                batch.subview(point_start, point_stop)?,
                sources,
                Some(&schedule.source_states),
                *external_count,
                particle_masses,
                momentum_slots,
                external_is_initial,
                model_parameter_values_f64,
            )?;
            evaluate_direct_helicity_schedule(
                direct,
                compiled_direct_helicity_schedules,
                schedule,
                color_schedule.as_deref(),
                execute_union,
                point_stop - point_start,
            )?;
            let resolved = amplitude.reduce_planes_f64_for_materialized_helicity(
                direct.amplitude_planes()?,
                physics,
                *normalization_factor,
                schedule.physical_helicity_index,
                &schedule.root_factors,
                selected_color_ids,
            )?;
            if resolved.helicity_indices != [schedule.physical_helicity_index]
                || resolved.color_indices != color_indices
                || resolved.point_count != point_stop - point_start
            {
                return Err(RusticolError::integrity(
                    "compiled Direct-Arena materialized-helicity reducer returned an inconsistent shape",
                ));
            }
            for tile_point in 0..resolved.point_count {
                let source = tile_point * color_indices.len();
                let target = (point_start + tile_point) * color_indices.len();
                values[target..target + color_indices.len()]
                    .copy_from_slice(&resolved.values[source..source + color_indices.len()]);
            }
            point_start = point_stop;
        }
        Ok(ResolvedValues {
            values,
            point_count,
            helicity_indices: vec![schedule.physical_helicity_index],
            color_indices,
        })
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[allow(clippy::too_many_arguments)]
    fn run_f64_materialized_helicity_direct_add_into_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        physics: &PhysicsRuntime,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
        schedule: &CompiledHelicitySelectorSchedule,
        execute_union: bool,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let point_count = batch.point_count();
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "materialized-helicity output has length {}, expected {point_count}",
                output.len()
            )));
        }
        let Self {
            compiled_direct_runtime: Some(direct),
            compiled_direct_color_schedules,
            compiled_direct_helicity_schedules,
            sources,
            external_count,
            particle_masses,
            momentum_slots,
            external_is_initial,
            model_parameter_values_f64,
            amplitude_stage: Some(amplitude),
            normalization_factor,
            ..
        } = self
        else {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena helicity totals were selected without a complete runtime",
            ));
        };
        let color_schedule = bind_direct_helicity_color_schedule(
            direct,
            compiled_direct_helicity_schedules,
            compiled_direct_color_schedules,
            schedule,
            selected_materialized_sector_ids,
        )?;
        let direct_total_plan = amplitude.bind_default_materialized_helicity_direct_total_plan(
            schedule.selector_domain_id,
            physics,
            schedule.physical_helicity_index,
            &schedule.root_factors,
            selected_color_ids,
        )?;
        let tile_capacity = direct.tile_capacity();
        let mut point_start = 0usize;
        while point_start < point_count {
            let point_stop = (point_start + tile_capacity).min(point_count);
            direct.begin_tile_from_inputs(
                batch.subview(point_start, point_stop)?,
                sources,
                Some(&schedule.source_states),
                *external_count,
                particle_masses,
                momentum_slots,
                external_is_initial,
                model_parameter_values_f64,
            )?;
            evaluate_direct_helicity_schedule(
                direct,
                compiled_direct_helicity_schedules,
                schedule,
                color_schedule.as_deref(),
                execute_union,
                point_stop - point_start,
            )?;
            amplitude.reduce_planes_f64_for_materialized_helicity_plan_add_into(
                direct.amplitude_planes()?,
                *normalization_factor,
                direct_total_plan,
                &mut output[point_start..point_stop],
            )?;
            point_start = point_stop;
        }
        Ok(())
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_f64_materialized_helicity_direct_routed_components_add_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        physics: &PhysicsRuntime,
        selected_helicity_ids: &BTreeSet<String>,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: &BTreeSet<i64>,
        replay_entry: &LcResolvedReplayEntry,
        source_component_count: usize,
        target_component_count: usize,
        target_point_count: usize,
        target_components: &mut [f64],
    ) -> RusticolResult<()> {
        if selected_helicity_ids.len() != 1 {
            return Err(RusticolError::integrity(
                "routed materialized-helicity execution requires one source helicity",
            ));
        }
        let selected_helicity_id = selected_helicity_ids
            .iter()
            .next()
            .expect("singleton source helicity checked");
        let helicity_index = *physics
            .helicity_index_by_id
            .get(selected_helicity_id)
            .ok_or_else(|| {
                RusticolError::selector(format!(
                    "unknown resolved helicity id {selected_helicity_id:?}"
                ))
            })?;
        let schedule = self
            .compiled_helicity_execution_plan
            .as_ref()
            .and_then(|plan| {
                plan.schedules_by_physical_helicity
                    .get(helicity_index)
                    .and_then(Option::as_ref)
            })
            .cloned()
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "physical helicity {} has no compiled recurrence schedule",
                    physics.manifest.helicities[helicity_index].id
                ))
            })?;
        if schedule.structural_zero {
            return Ok(());
        }
        let point_count = batch.point_count();
        let Self {
            compiled_direct_runtime: Some(direct),
            compiled_direct_color_schedules,
            compiled_direct_helicity_schedules,
            sources,
            external_count,
            particle_masses,
            momentum_slots,
            external_is_initial,
            model_parameter_values_f64,
            amplitude_stage: Some(amplitude),
            normalization_factor,
            ..
        } = self
        else {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena routed helicity execution was selected without a complete runtime",
            ));
        };
        let color_schedule = bind_direct_helicity_color_schedule(
            direct,
            compiled_direct_helicity_schedules,
            compiled_direct_color_schedules,
            schedule.as_ref(),
            Some(selected_materialized_sector_ids),
        )?;
        // A selected helicity/flow uses only these routed components. The
        // complete-axis footprint can otherwise shrink even a singleton to
        // one point per tile, despite the already allocated SIMD capacity.
        let reduction_footprint = amplitude.compiled_direct_reduction_footprint(
            0,
            CompiledDirectRoutedReductionFootprint {
                maximum_source_component_count: source_component_count,
                maximum_target_component_count: target_component_count,
            },
        )?;
        let tile_capacity = direct.reduction_tile_capacity_for_footprint(reduction_footprint)?;
        let mut point_start = 0usize;
        while point_start < point_count {
            let point_stop = (point_start + tile_capacity).min(point_count);
            direct.begin_tile_from_inputs(
                batch.subview(point_start, point_stop)?,
                sources,
                Some(&schedule.source_states),
                *external_count,
                particle_masses,
                momentum_slots,
                external_is_initial,
                model_parameter_values_f64,
            )?;
            evaluate_direct_helicity_schedule(
                direct,
                compiled_direct_helicity_schedules,
                schedule.as_ref(),
                color_schedule.as_deref(),
                false,
                point_stop - point_start,
            )?;
            amplitude.reduce_planes_f64_for_materialized_helicity_routed_components_add_into(
                direct.amplitude_planes()?,
                physics,
                *normalization_factor,
                schedule.physical_helicity_index,
                &schedule.root_factors,
                selected_color_ids,
                replay_entry,
                source_component_count,
                target_component_count,
                target_point_count,
                point_start,
                target_components,
            )?;
            point_start = point_stop;
        }
        Ok(())
    }

    fn evaluate_f64_materialized_helicity_schedule_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
        schedule: &CompiledHelicitySelectorSchedule,
        execute_union: bool,
    ) -> RusticolResult<()> {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return Err(self.execution_unavailable_error());
        }
        let point_count = batch.point_count();
        let color_schedule = selected_materialized_sector_ids
            .map(|sector_ids| self.compiled_color_selector_schedule(sector_ids))
            .transpose()?
            .flatten();
        let state_len = point_count
            .checked_mul(self.parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("runtime state length overflows"))?;
        self.state_scratch_f64.resize(state_len, c64(0.0, 0.0));
        self.state_scratch_f64_requires_clear = true;
        let state = &mut self.state_scratch_f64;
        let model_parameter_start = self.value_parameter_count + self.momentum_parameter_count;
        for row in 0..point_count {
            let point = batch.point(row);
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_sources_row_with_states(
                &self.sources,
                &schedule.source_states,
                self.external_count,
                &self.particle_masses,
                row_state,
                &point,
            )?;
        }
        for row in 0..point_count {
            let point = batch.point(row);
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_momenta_row(
                &self.momentum_slots,
                self.value_parameter_count,
                self.external_count,
                &self.external_is_initial,
                row_state,
                &point,
            )?;
        }
        for row in 0..point_count {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_model_parameters_row(
                model_parameter_start,
                &self.model_parameter_values_f64,
                row_state,
            )?;
        }
        let stages = self.stages.as_mut().expect("generic stages checked");
        if schedule.active_stage_chunk_indices.len() != stages.len() {
            return Err(RusticolError::integrity(format!(
                "helicity schedule {} has {} stage closures, expected {}",
                schedule.selector_domain_id,
                schedule.active_stage_chunk_indices.len(),
                stages.len()
            )));
        }
        for (stage_index, (stage, active_chunks)) in stages
            .iter_mut()
            .zip(&schedule.active_stage_chunk_indices)
            .enumerate()
        {
            if execute_union {
                stage.evaluate_f64_into_state(
                    point_count,
                    self.parameter_count,
                    state.as_mut_slice(),
                )?;
            } else {
                let combined_chunks;
                let active_chunks = if let Some(color_schedule) = color_schedule.as_ref() {
                    combined_chunks = intersect_sorted_chunk_indices(
                        active_chunks,
                        &color_schedule.active_stage_chunk_indices[stage_index],
                    );
                    &combined_chunks
                } else {
                    active_chunks
                };
                stage.evaluate_active_chunks_f64_into_state(
                    point_count,
                    self.parameter_count,
                    state.as_mut_slice(),
                    active_chunks,
                )?;
            }
        }
        let combined_amplitude_chunks;
        let active_amplitude_chunks = if let Some(color_schedule) = color_schedule.as_ref() {
            combined_amplitude_chunks = intersect_sorted_chunk_indices(
                &schedule.active_amplitude_chunk_indices,
                &color_schedule.active_amplitude_chunk_indices,
            );
            &combined_amplitude_chunks
        } else {
            &schedule.active_amplitude_chunk_indices
        };
        let amplitude = self
            .amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked");
        if execute_union {
            amplitude.evaluate_f64_into_scratch(point_count, state.as_slice())?;
        } else {
            amplitude.evaluate_active_chunks_f64_into_scratch(
                point_count,
                state.as_slice(),
                active_amplitude_chunks,
            )?;
        }
        Ok(())
    }

    pub(super) fn run_resolved_f64_with_helicity_recurrence(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        let total_start = Instant::now();
        let initial_orchestration_start = Instant::now();
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "helicity recurrence materialization requires resolved physics metadata",
            )
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let point_count = batch.len();
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        let mut values = vec![0.0; point_count * component_count];
        let mut schedule_profiles = Vec::with_capacity(helicity_indices.len());
        let mut orchestration_elapsed = initial_orchestration_start.elapsed();
        for (helicity_position, helicity_index) in helicity_indices.iter().copied().enumerate() {
            let orchestration_start = Instant::now();
            let schedule = self
                .compiled_helicity_execution_plan
                .as_ref()
                .and_then(|plan| {
                    plan.schedules_by_physical_helicity
                        .get(helicity_index)
                        .and_then(Option::as_ref)
                })
                .cloned()
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "physical helicity {} has no compiled recurrence schedule",
                        physics.manifest.helicities[helicity_index].id
                    ))
                })?;
            if schedule.structural_zero {
                orchestration_elapsed += orchestration_start.elapsed();
                continue;
            }
            let selector_lane_index = self
                .helicity_selector_lane_by_domain
                .get(&schedule.selector_domain_id)
                .copied()
                .filter(|lane_index| {
                    // A nested union understands public color IDs, but not a
                    // topology-replay group's parent materialized-sector
                    // restriction. Use the parent schedule for those groups.
                    self.helicity_selector_runtime_schedule_modes
                        .get(*lane_index)
                        .is_some_and(|schedule_mode| {
                            (*schedule_mode == HelicitySelectorScheduleMode::NestedRuntime
                                && selected_materialized_sector_ids.is_none())
                                || (*schedule_mode == HelicitySelectorScheduleMode::ParentClosure
                                    && selected_color_ids.is_none()
                                    && selected_materialized_sector_ids.is_none()
                                    && !physics.has_contracted_color_axis()
                                    && !self.color_topology_replay_enabled
                                    && !self.lc_topology_replay_enabled)
                        })
                });
            orchestration_elapsed += orchestration_start.elapsed();
            let (mut selected, schedule_profile) = if let Some(lane_index) = selector_lane_index {
                let schedule_mode = self
                    .helicity_selector_runtime_schedule_modes
                    .get(lane_index)
                    .copied()
                    .ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "helicity selector domain {} maps to missing schedule mode for lane {lane_index}",
                            schedule.selector_domain_id
                        ))
                    })?;
                let selector_runtime = self
                    .helicity_selector_runtimes
                    .get_mut(lane_index)
                    .ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "helicity selector domain {} maps to missing execution lane {lane_index}",
                            schedule.selector_domain_id
                        ))
                    })?;
                match schedule_mode {
                    HelicitySelectorScheduleMode::ParentClosure => selector_runtime
                        .run_f64_materialized_helicity_schedule(
                            batch, &physics, None, None, &schedule, true,
                        )?,
                    HelicitySelectorScheduleMode::NestedRuntime => {
                        let selected_helicity = BTreeSet::from([physics.manifest.helicities
                            [schedule.physical_helicity_index]
                            .id
                            .clone()]);
                        selector_runtime.run_resolved_f64(
                            batch,
                            Some(&selected_helicity),
                            selected_color_ids,
                        )?
                    }
                }
            } else {
                if self.color_topology_replay_enabled
                    || (self.lc_topology_replay_enabled
                        && selected_materialized_sector_ids.is_none())
                {
                    return Err(RusticolError::integrity(
                        "topology-replay runtime cannot execute its representative materialized helicity schedule directly",
                    ));
                }
                self.run_f64_materialized_helicity_schedule(
                    batch,
                    &physics,
                    selected_color_ids,
                    selected_materialized_sector_ids,
                    &schedule,
                    false,
                )?
            };
            let post_execution_start = Instant::now();
            if selected.helicity_indices != [schedule.physical_helicity_index]
                || selected.color_indices != color_indices
                || selected.point_count != point_count
            {
                return Err(RusticolError::integrity(
                    "materialized helicity schedule returned an inconsistent resolved shape",
                ));
            }
            selected.helicity_indices[0] = helicity_index;
            if helicity_indices.len() == 1 {
                let mut selected_profile = schedule_profile;
                orchestration_elapsed += post_execution_start.elapsed();
                selected_profile.orchestration_s += profile_duration_seconds(orchestration_elapsed);
                selected_profile.total_s = total_start.elapsed().as_secs_f64();
                return Ok((selected, selected_profile));
            }
            for point_index in 0..point_count {
                let source_start = point_index * color_indices.len();
                let target_start =
                    point_index * component_count + helicity_position * color_indices.len();
                values[target_start..target_start + color_indices.len()].copy_from_slice(
                    &selected.values[source_start..source_start + color_indices.len()],
                );
            }
            schedule_profiles.push(schedule_profile);
            orchestration_elapsed += post_execution_start.elapsed();
        }
        let mut profile = RuntimeProfile::default();
        for schedule_profile in &schedule_profiles {
            profile.add_sector(schedule_profile);
        }
        profile.orchestration_s += profile_duration_seconds(orchestration_elapsed);
        profile.total_s = total_start.elapsed().as_secs_f64();
        Ok((
            ResolvedValues {
                values,
                point_count,
                helicity_indices,
                color_indices,
            },
            profile,
        ))
    }

    fn run_f64_materialized_helicity_schedule(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        physics: &PhysicsRuntime,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
        schedule: &CompiledHelicitySelectorSchedule,
        execute_union: bool,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return Err(self.execution_unavailable_error());
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let state_prepare_start = Instant::now();
        let color_schedule = selected_materialized_sector_ids
            .map(|sector_ids| self.compiled_color_selector_schedule(sector_ids))
            .transpose()?
            .flatten();
        let state_len = point_count
            .checked_mul(self.parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("runtime state length overflows"))?;
        let state_capacity = self.state_scratch_f64.capacity();
        self.state_scratch_f64.resize(state_len, c64(0.0, 0.0));
        let state_prepare_elapsed = state_prepare_start.elapsed();
        let mut scratch_reallocation_count =
            u64::from(self.state_scratch_f64.capacity() != state_capacity);
        // Every source, momentum, model parameter, and recursive value read by
        // this exact closure is overwritten below.  Inactive parent-layout
        // slots are never read, so clearing the full reusable state would only
        // add a parent-size memory pass to a selector-specialized execution.
        // A later full execution still clears untouched slots once.
        self.state_scratch_f64_requires_clear = true;
        let state = &mut self.state_scratch_f64;
        let model_parameter_start = self.value_parameter_count + self.momentum_parameter_count;

        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_sources_row_with_states(
                &self.sources,
                &schedule.source_states,
                self.external_count,
                &self.particle_masses,
                row_state,
                point,
            )?;
        }
        let source_fill_elapsed = source_start.elapsed();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_momenta_row(
                &self.momentum_slots,
                self.value_parameter_count,
                self.external_count,
                &self.external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_elapsed = momentum_start.elapsed();

        let model_parameter_start_time = Instant::now();
        for row in 0..point_count {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_model_parameters_row(
                model_parameter_start,
                &self.model_parameter_values_f64,
                row_state,
            )?;
        }
        let model_parameter_setup_elapsed = model_parameter_start_time.elapsed();

        let stages = self.stages.as_mut().expect("generic stages checked");
        if schedule.active_stage_chunk_indices.len() != stages.len() {
            return Err(RusticolError::integrity(format!(
                "helicity schedule {} has {} stage closures, expected {}",
                schedule.selector_domain_id,
                schedule.active_stage_chunk_indices.len(),
                stages.len()
            )));
        }
        let mut stage_profiles = Vec::with_capacity(stages.len());
        let mut stage_input_pack_by_stage_s = Vec::with_capacity(stages.len());
        let mut stage_leaf_input_pack_by_stage_s = Vec::with_capacity(stages.len());
        let mut stage_evaluator_call_by_stage_s = Vec::with_capacity(stages.len());
        let mut stage_backend_call_by_stage_s = Vec::with_capacity(stages.len());
        let mut stage_evaluator_output_gather_by_stage_s = Vec::with_capacity(stages.len());
        let mut stage_output_assign_by_stage_s = Vec::with_capacity(stages.len());
        for (stage_index, (stage, active_chunks)) in stages
            .iter_mut()
            .zip(&schedule.active_stage_chunk_indices)
            .enumerate()
        {
            let stage_profile = if execute_union {
                stage.evaluate_f64_into_state_profile(
                    point_count,
                    self.parameter_count,
                    state.as_mut_slice(),
                )?
            } else {
                let combined_chunks;
                let active_chunks = if let Some(color_schedule) = color_schedule.as_ref() {
                    combined_chunks = intersect_sorted_chunk_indices(
                        active_chunks,
                        &color_schedule.active_stage_chunk_indices[stage_index],
                    );
                    &combined_chunks
                } else {
                    active_chunks
                };
                stage.evaluate_active_chunks_f64_into_state_profile(
                    point_count,
                    self.parameter_count,
                    state.as_mut_slice(),
                    active_chunks,
                )?
            };
            stage_input_pack_by_stage_s.push(stage_profile.input_pack_s);
            stage_leaf_input_pack_by_stage_s.push(stage_profile.evaluator.leaf_input_pack_s);
            stage_evaluator_call_by_stage_s.push(stage_profile.evaluator.legacy_evaluator_call_s);
            stage_backend_call_by_stage_s.push(stage_profile.evaluator.evaluator_call_s);
            stage_evaluator_output_gather_by_stage_s.push(stage_profile.evaluator.output_gather_s);
            stage_output_assign_by_stage_s.push(stage_profile.output_assign_s);
            stage_profiles.push(stage_profile);
        }

        let combined_amplitude_chunks;
        let active_amplitude_chunks = if let Some(color_schedule) = color_schedule.as_ref() {
            combined_amplitude_chunks = intersect_sorted_chunk_indices(
                &schedule.active_amplitude_chunk_indices,
                &color_schedule.active_amplitude_chunk_indices,
            );
            &combined_amplitude_chunks
        } else {
            &schedule.active_amplitude_chunk_indices
        };
        let amplitude = self
            .amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked");
        let amplitude_profile = if execute_union {
            amplitude.evaluate_f64_into_scratch_profile(point_count, state.as_slice())?
        } else {
            amplitude.evaluate_active_chunks_f64_into_scratch_profile(
                point_count,
                state.as_slice(),
                active_amplitude_chunks,
            )?
        };
        let amplitude_output_length = amplitude.output_length;
        let amplitude_input_pack_s = amplitude_profile.input_pack_s;
        let amplitude_evaluator_call_s =
            amplitude_profile.evaluator.legacy_evaluator_call_s + amplitude_profile.output_remap_s;
        let reduction_start = Instant::now();
        let resolved = self
            .amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .reduce_scratch_f64_for_materialized_helicity(
                point_count,
                physics,
                self.normalization_factor,
                schedule.physical_helicity_index,
                &schedule.root_factors,
                selected_color_ids,
            )?;
        let reduction_s = reduction_start.elapsed().as_secs_f64();
        let resolved_materialized_component_count = resolved.values.len() as u64;
        let stage_input_pack_s = stage_input_pack_by_stage_s.iter().sum::<f64>();
        let stage_evaluator_call_s = stage_evaluator_call_by_stage_s.iter().sum::<f64>();
        let output_assign_s = stage_output_assign_by_stage_s.iter().sum::<f64>();
        scratch_reallocation_count += stage_profiles
            .iter()
            .map(|profile| {
                profile.scratch_reallocation_count + profile.evaluator.scratch_reallocation_count
            })
            .sum::<u64>()
            + amplitude_profile.scratch_reallocation_count
            + amplitude_profile.evaluator.scratch_reallocation_count;
        Ok((
            resolved,
            RuntimeProfile {
                state_prepare_s: profile_duration_seconds(state_prepare_elapsed),
                source_fill_s: profile_duration_seconds(source_fill_elapsed),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup_elapsed),
                momentum_setup_s: profile_duration_seconds(
                    momentum_setup_elapsed + model_parameter_setup_elapsed,
                ),
                model_parameter_setup_s: profile_duration_seconds(model_parameter_setup_elapsed),
                stage_input_pack_s,
                stage_leaf_input_pack_s: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.leaf_input_pack_s)
                    .sum(),
                stage_evaluator_call_s,
                stage_backend_call_s: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.evaluator_call_s)
                    .sum(),
                stage_evaluator_output_gather_s: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.output_gather_s)
                    .sum(),
                stage_evaluator_s: stage_input_pack_s + stage_evaluator_call_s,
                output_assign_s,
                amplitude_input_pack_s,
                amplitude_leaf_input_pack_s: amplitude_profile.evaluator.leaf_input_pack_s,
                amplitude_evaluator_call_s,
                amplitude_backend_call_s: amplitude_profile.evaluator.evaluator_call_s,
                amplitude_evaluator_output_gather_s: amplitude_profile.evaluator.output_gather_s,
                amplitude_output_remap_s: amplitude_profile.output_remap_s,
                amplitude_evaluator_s: amplitude_input_pack_s + amplitude_evaluator_call_s,
                reduction_s,
                resolved_reduction_materialization_s: reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                stage_input_pack_by_stage_s,
                stage_leaf_input_pack_by_stage_s,
                stage_evaluator_call_by_stage_s,
                stage_backend_call_by_stage_s,
                stage_evaluator_output_gather_by_stage_s,
                stage_output_assign_by_stage_s,
                state_component_count: state_len as u64,
                source_component_count: (point_count * schedule.source_states.len()) as u64,
                momentum_component_count: (point_count * self.momentum_parameter_count) as u64,
                model_parameter_component_count: (point_count
                    * self.model_parameter_values_f64.len())
                    as u64,
                stage_input_copy_component_count: stage_profiles
                    .iter()
                    .map(|profile| profile.input_copy_component_count)
                    .sum(),
                stage_leaf_input_copy_component_count: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.leaf_input_copy_component_count)
                    .sum(),
                stage_evaluator_output_gather_component_count: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.output_gather_component_count)
                    .sum(),
                stage_output_assign_component_count: stage_profiles
                    .iter()
                    .map(|profile| profile.output_assign_component_count)
                    .sum(),
                amplitude_input_copy_component_count: amplitude_profile.input_copy_component_count,
                amplitude_leaf_input_copy_component_count: amplitude_profile
                    .evaluator
                    .leaf_input_copy_component_count,
                amplitude_evaluator_output_gather_component_count: amplitude_profile
                    .evaluator
                    .output_gather_component_count,
                amplitude_output_remap_component_count: amplitude_profile
                    .output_remap_component_count,
                evaluator_backend_call_count: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.backend_call_count)
                    .sum::<u64>()
                    + amplitude_profile.evaluator.backend_call_count,
                reduction_input_component_count: (point_count * amplitude_output_length) as u64,
                resolved_materialized_component_count,
                scratch_reallocation_count,
                ..RuntimeProfile::default()
            },
        ))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn run_resolved_generic_with_helicity_recurrence<T>(
        &mut self,
        batch: &[Vec<[T; 4]>],
        binary_precision: Option<u32>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<(ResolvedValues<T>, RuntimeProfile)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let total_start = Instant::now();
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "helicity recurrence materialization requires resolved physics metadata",
            )
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let point_count = batch.len();
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        let mut values = vec![T::new_zero(); point_count * component_count];
        let mut schedule_profiles = Vec::with_capacity(helicity_indices.len());

        for (helicity_position, helicity_index) in helicity_indices.iter().copied().enumerate() {
            let schedule = self
                .compiled_helicity_execution_plan
                .as_ref()
                .and_then(|plan| {
                    plan.schedules_by_physical_helicity
                        .get(helicity_index)
                        .and_then(Option::as_ref)
                })
                .cloned()
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "physical helicity {} has no compiled recurrence schedule",
                        physics.manifest.helicities[helicity_index].id
                    ))
                })?;
            if schedule.structural_zero {
                continue;
            }
            let nested_lane_index = self
                .helicity_selector_lane_by_domain
                .get(&schedule.selector_domain_id)
                .copied()
                .filter(|lane_index| {
                    selected_materialized_sector_ids.is_none()
                        && self
                            .helicity_selector_runtime_schedule_modes
                            .get(*lane_index)
                            == Some(&HelicitySelectorScheduleMode::NestedRuntime)
                });
            let (mut selected, schedule_profile) = if let Some(lane_index) = nested_lane_index {
                let selected_helicity = BTreeSet::from([physics.manifest.helicities
                    [schedule.physical_helicity_index]
                    .id
                    .clone()]);
                self.helicity_selector_runtimes
                    .get_mut(lane_index)
                    .ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "helicity selector domain {} maps to missing exact execution lane {lane_index}",
                            schedule.selector_domain_id
                        ))
                    })?
                    .run_resolved_generic(
                        batch,
                        binary_precision,
                        Some(&selected_helicity),
                        selected_color_ids,
                    )?
            } else {
                self.run_generic_materialized_helicity_schedule(
                    batch,
                    binary_precision,
                    &physics,
                    selected_color_ids,
                    selected_materialized_sector_ids,
                    &schedule,
                )?
            };
            if selected.helicity_indices != [schedule.physical_helicity_index]
                || selected.color_indices != color_indices
                || selected.point_count != point_count
            {
                return Err(RusticolError::integrity(
                    "materialized helicity schedule returned an inconsistent resolved shape",
                ));
            }
            selected.helicity_indices[0] = helicity_index;
            for point_index in 0..point_count {
                let source_start = point_index * color_indices.len();
                let target_start =
                    point_index * component_count + helicity_position * color_indices.len();
                values[target_start..target_start + color_indices.len()].clone_from_slice(
                    &selected.values[source_start..source_start + color_indices.len()],
                );
            }
            schedule_profiles.push(schedule_profile);
        }
        let mut profile = RuntimeProfile::default();
        for schedule_profile in &schedule_profiles {
            profile.add_sector(schedule_profile);
        }
        profile.total_s = total_start.elapsed().as_secs_f64();
        Ok((
            ResolvedValues {
                values,
                point_count,
                helicity_indices,
                color_indices,
            },
            profile,
        ))
    }

    #[cfg(feature = "symbolica-runtime")]
    fn run_generic_materialized_helicity_schedule<T>(
        &mut self,
        batch: &[Vec<[T; 4]>],
        binary_precision: Option<u32>,
        physics: &PhysicsRuntime,
        selected_color_ids: Option<&BTreeSet<String>>,
        selected_materialized_sector_ids: Option<&BTreeSet<i64>>,
        schedule: &CompiledHelicitySelectorSchedule,
    ) -> RusticolResult<(ResolvedValues<T>, RuntimeProfile)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return Err(self.execution_unavailable_error());
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let color_schedule = selected_materialized_sector_ids
            .map(|sector_ids| self.compiled_color_selector_schedule(sector_ids))
            .transpose()?
            .flatten();
        let state_len = point_count
            .checked_mul(self.parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("runtime state length overflows"))?;
        let mut state = vec![complex_zero::<T>(); state_len];
        let model_parameter_start = self.value_parameter_count + self.momentum_parameter_count;

        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_sources_row_generic_with_states(
                &self.sources,
                &schedule.source_states,
                self.external_count,
                &self.particle_masses,
                row_state,
                point,
            )?;
        }
        let source_fill_elapsed = source_start.elapsed();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_momenta_row_generic(
                &self.momentum_slots,
                self.value_parameter_count,
                self.external_count,
                &self.external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_elapsed = momentum_start.elapsed();

        let model_parameter_start_time = Instant::now();
        for row in 0..point_count {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_model_parameters_row_generic(
                model_parameter_start,
                &self.model_parameter_values_f64,
                row_state,
            )?;
        }
        let model_parameter_setup_elapsed = model_parameter_start_time.elapsed();

        let stages = self.stages.as_mut().expect("generic stages checked");
        if schedule.active_stage_chunk_indices.len() != stages.len() {
            return Err(RusticolError::integrity(format!(
                "helicity schedule {} has {} stage closures, expected {}",
                schedule.selector_domain_id,
                schedule.active_stage_chunk_indices.len(),
                stages.len()
            )));
        }
        // The same certified closure applies independently of precision.
        // Follow its chunks after intersecting the mapped colour selection.
        let mut stage_input_pack_by_stage_s = Vec::with_capacity(stages.len());
        let mut stage_evaluator_call_by_stage_s = Vec::with_capacity(stages.len());
        let mut stage_output_assign_by_stage_s = Vec::with_capacity(stages.len());
        for (stage_index, (stage, active_chunks)) in stages
            .iter_mut()
            .zip(&schedule.active_stage_chunk_indices)
            .enumerate()
        {
            let combined_chunks;
            let active_chunks = if let Some(color_schedule) = color_schedule.as_ref() {
                combined_chunks = intersect_sorted_chunk_indices(
                    active_chunks,
                    &color_schedule.active_stage_chunk_indices[stage_index],
                );
                &combined_chunks
            } else {
                active_chunks
            };
            let (pack_s, eval_s, assign_s) = stage.evaluate_selected_chunks_generic_into_state(
                point_count,
                self.parameter_count,
                state.as_mut_slice(),
                binary_precision,
                Some(active_chunks),
            )?;
            stage_input_pack_by_stage_s.push(pack_s);
            stage_evaluator_call_by_stage_s.push(eval_s);
            stage_output_assign_by_stage_s.push(assign_s);
        }

        let combined_amplitude_chunks;
        let active_amplitude_chunks = if let Some(color_schedule) = color_schedule.as_ref() {
            combined_amplitude_chunks = intersect_sorted_chunk_indices(
                &schedule.active_amplitude_chunk_indices,
                &color_schedule.active_amplitude_chunk_indices,
            );
            &combined_amplitude_chunks
        } else {
            &schedule.active_amplitude_chunk_indices
        };
        let (resolved, amplitude_input_pack_s, amplitude_evaluator_call_s, reduction_s) = self
            .amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .evaluate_materialized_helicity_generic(
                point_count,
                state.as_slice(),
                binary_precision,
                physics,
                self.normalization_factor,
                schedule.physical_helicity_index,
                &schedule.root_factors,
                selected_color_ids,
                active_amplitude_chunks,
            )?;
        let stage_input_pack_s = stage_input_pack_by_stage_s.iter().sum::<f64>();
        let stage_evaluator_call_s = stage_evaluator_call_by_stage_s.iter().sum::<f64>();
        let output_assign_s = stage_output_assign_by_stage_s.iter().sum::<f64>();
        Ok((
            resolved,
            RuntimeProfile {
                source_fill_s: profile_duration_seconds(source_fill_elapsed),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup_elapsed),
                momentum_setup_s: profile_duration_seconds(
                    momentum_setup_elapsed + model_parameter_setup_elapsed,
                ),
                model_parameter_setup_s: profile_duration_seconds(model_parameter_setup_elapsed),
                stage_input_pack_s,
                stage_evaluator_call_s,
                stage_evaluator_s: stage_input_pack_s + stage_evaluator_call_s,
                output_assign_s,
                amplitude_input_pack_s,
                amplitude_evaluator_call_s,
                amplitude_evaluator_s: amplitude_input_pack_s + amplitude_evaluator_call_s,
                reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                stage_input_pack_by_stage_s,
                stage_evaluator_call_by_stage_s,
                stage_output_assign_by_stage_s,
                ..RuntimeProfile::default()
            },
        ))
    }
}

fn intersect_sorted_chunk_indices(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(left.len().min(right.len()));
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                result.push(left[left_index]);
                left_index += 1;
                right_index += 1;
            }
        }
    }
    result
}

fn physical_helicity_index_for_domain(
    physics: &PhysicsRuntime,
    domain: &HelicitySelectorDomainRuntime,
    public_label_permutation: Option<&[usize]>,
) -> RusticolResult<usize> {
    let external_count = physics
        .manifest
        .helicities
        .first()
        .map(|helicity| helicity.values.len())
        .ok_or_else(|| RusticolError::artifact("physics metadata has no helicity axis"))?;
    let mut values = vec![None; external_count];
    for (external_index, helicity) in &domain.source_states {
        let value = values.get_mut(*external_index).ok_or_else(|| {
            RusticolError::integrity("helicity selector domain has a stale external label")
        })?;
        if value.replace(*helicity).is_some() {
            return Err(RusticolError::integrity(
                "helicity selector domain repeats an external label",
            ));
        }
    }
    let mut values = values
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| RusticolError::integrity("helicity selector domain is incomplete"))?;
    if let Some(permutation) = public_label_permutation {
        values = permute_representative_helicity_to_public_alias(&values, permutation)?;
    }
    physics
        .manifest
        .helicities
        .iter()
        .position(|helicity| helicity.values == values)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "helicity selector domain {values:?} has no physical metadata entry"
            ))
        })
}

fn permute_representative_helicity_to_public_alias(
    representative: &[i32],
    permutation: &[usize],
) -> RusticolResult<Vec<i32>> {
    if permutation.len() != representative.len()
        || permutation.iter().copied().collect::<BTreeSet<_>>()
            != (0..representative.len()).collect::<BTreeSet<_>>()
    {
        return Err(RusticolError::integrity(
            "process alias does not define a complete helicity-label permutation",
        ));
    }
    let mut public = vec![0; representative.len()];
    for (representative_index, public_index) in permutation.iter().copied().enumerate() {
        public[public_index] = representative[representative_index];
    }
    Ok(public)
}

fn build_materialized_source_states(
    sources: &[GenericSourceRecordManifest],
    selector_domains: &[HelicitySelectorDomainRuntime],
    source_routes: &[HelicityMaterializedSourceRouteRuntime],
    complete_domain: &HelicitySelectorDomainRuntime,
    active_current_ids: &[usize],
) -> RusticolResult<Vec<RuntimeSourceState>> {
    let active_current_ids = active_current_ids.iter().copied().collect::<BTreeSet<_>>();
    let state_by_external = complete_domain
        .source_states
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    let mut states = Vec::with_capacity(sources.len());
    for source in sources {
        if !active_current_ids.contains(&source.current_id) {
            states.push(ExecutionRuntime::inactive_runtime_source_state(source)?);
            continue;
        }
        let external_index = source
            .leg_label
            .checked_sub(1)
            .ok_or_else(|| RusticolError::integrity("runtime source labels must be one-based"))?;
        let selected_helicity = state_by_external.get(&external_index).ok_or_else(|| {
            RusticolError::integrity(format!(
                "active materialized source {} has no state in its complete helicity domain",
                source.source_id
            ))
        })?;
        let mut matching = source_routes.iter().filter(|route| {
            route.materialized_current_id == source.current_id
                && route.external_index == external_index
                && route.helicity == *selected_helicity
                && selector_domains
                    .get(route.selector_domain_id)
                    .is_some_and(|domain| {
                        domain
                            .source_states
                            .iter()
                            .all(|state| complete_domain.source_states.contains(state))
                    })
        });
        let route = matching.next();
        if matching.next().is_some() {
            return Err(RusticolError::artifact(format!(
                "active materialized source {} has multiple routes for helicity {}",
                source.source_id, selected_helicity
            )));
        }
        states.push(if let Some(route) = route {
            ExecutionRuntime::runtime_source_state(
                source,
                route.declared_state_index,
                c64(route.factor[0], route.factor[1]),
            )?
        } else {
            let declared_state_index =
                ExecutionRuntime::runtime_source_state_index_for_helicity(
                    source,
                    *selected_helicity,
                )?
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "active materialized source {} has neither a recurrence route nor a declared state for helicity {}",
                        source.source_id, selected_helicity
                    ))
                })?;
            ExecutionRuntime::runtime_source_state(
                source,
                declared_state_index,
                c64(1.0, 0.0),
            )?
        });
    }
    Ok(states)
}

fn materialized_root_factors_for_domain(
    root_count: usize,
    routes: &[HelicityMaterializedAmplitudeRouteRuntime],
    schedule: &HelicityMaterializedSelectorScheduleRuntime,
) -> RusticolResult<Vec<Option<Complex<f64>>>> {
    let active_roots = schedule
        .active_root_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut factors = vec![None; root_count];
    for route in routes {
        if !route
            .selector_domain_ids
            .contains(&schedule.selector_domain_id)
        {
            continue;
        }
        let factor = c64(route.factor[0], route.factor[1]);
        let target = factors.get_mut(route.materialized_root_id).ok_or_else(|| {
            RusticolError::integrity("helicity amplitude route has a stale materialized root")
        })?;
        if target.replace(factor).is_some() {
            return Err(RusticolError::artifact(format!(
                "materialized amplitude root {} has multiple routes for selector domain {}",
                route.materialized_root_id, schedule.selector_domain_id
            )));
        }
    }
    let routed_roots = factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| factor.as_ref().map(|_| index))
        .collect::<BTreeSet<_>>();
    if routed_roots != active_roots {
        return Err(RusticolError::artifact(format!(
            "selector domain {} amplitude routes do not match its active-root closure",
            schedule.selector_domain_id
        )));
    }
    Ok(factors)
}

fn helicity_execution_domain_by_domain(
    domain_count: usize,
    routes: &[HelicityMaterializedAmplitudeRouteRuntime],
) -> RusticolResult<Vec<usize>> {
    // The producer orders each proven route as its retained source state
    // followed by synthetic global-flip aliases. Execute aliases through the
    // retained source route and relabel the resolved result for the caller.
    let mut execution_domain_by_domain = (0..domain_count).collect::<Vec<_>>();
    let mut primary_domains = BTreeSet::new();
    for route in routes {
        let Some((&primary_domain, replay_domains)) = route.selector_domain_ids.split_first()
        else {
            return Err(RusticolError::artifact(
                "helicity materialized amplitude route has no selector domains",
            ));
        };
        if primary_domain >= domain_count {
            return Err(RusticolError::integrity(format!(
                "helicity amplitude route references stale primary domain {primary_domain}"
            )));
        }
        primary_domains.insert(primary_domain);
        for replay_domain in replay_domains {
            if *replay_domain >= domain_count {
                return Err(RusticolError::integrity(format!(
                    "helicity amplitude route references stale replay domain {replay_domain}"
                )));
            }
            let previous = execution_domain_by_domain[*replay_domain];
            if previous != *replay_domain && previous != primary_domain {
                return Err(RusticolError::artifact(format!(
                    "helicity replay domain {replay_domain} maps to inconsistent execution domains {previous} and {primary_domain}"
                )));
            }
            execution_domain_by_domain[*replay_domain] = primary_domain;
        }
    }
    for primary_domain in primary_domains {
        let execution_domain = execution_domain_by_domain[primary_domain];
        if execution_domain != primary_domain {
            return Err(RusticolError::artifact(format!(
                "helicity selector domain {primary_domain} is both a primary and replay-only domain"
            )));
        }
    }
    Ok(execution_domain_by_domain)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_with_singleton_helicity_schedule(
        structural_zero: bool,
    ) -> (ExecutionRuntime, BTreeSet<String>) {
        let mut runtime = super::super::tests::empty_generic_runtime();
        let physics = Arc::new(super::super::tests::test_physics_runtime("full"));
        let selected_helicity_id = physics.manifest.helicities[0].id.clone();
        let selected_helicity_ids = BTreeSet::from([selected_helicity_id.clone()]);
        runtime.physics = Some(physics);
        runtime.compiled_helicity_execution_plan = Some(CompiledHelicityExecutionPlan {
            schedules_by_physical_helicity: vec![
                Some(Arc::new(CompiledHelicitySelectorSchedule {
                    selector_domain_id: 0,
                    physical_helicity_index: 0,
                    selected_helicity_ids: BTreeSet::from([selected_helicity_id]),
                    source_states: Vec::new(),
                    active_stage_chunk_indices: Vec::new(),
                    active_amplitude_chunk_indices: Vec::new(),
                    root_factors: Vec::new(),
                    structural_zero,
                })),
                None,
            ],
        });
        (runtime, selected_helicity_ids)
    }

    #[test]
    fn representative_helicity_is_permuted_into_public_alias_order() {
        let values =
            permute_representative_helicity_to_public_alias(&[1, -1, 0, -1, 1], &[0, 1, 3, 4, 2])
                .unwrap();
        assert_eq!(values, vec![1, -1, 1, 0, -1]);
    }

    #[test]
    fn invalid_public_alias_permutation_is_rejected() {
        let error =
            permute_representative_helicity_to_public_alias(&[1, -1, 0], &[0, 1, 1]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("complete helicity-label permutation")
        );
    }

    #[test]
    fn global_flip_domains_execute_their_retained_partner_routes() {
        let routes = vec![
            HelicityMaterializedAmplitudeRouteRuntime {
                materialized_root_id: 0,
                selector_domain_ids: vec![2, 7],
                factor: [1.0, 0.0],
                residual: false,
            },
            HelicityMaterializedAmplitudeRouteRuntime {
                materialized_root_id: 1,
                selector_domain_ids: vec![3, 6],
                factor: [1.0, 0.0],
                residual: false,
            },
        ];

        let execution = helicity_execution_domain_by_domain(8, &routes).unwrap();

        assert_eq!(execution[2], 2);
        assert_eq!(execution[7], 2);
        assert_eq!(execution[3], 3);
        assert_eq!(execution[6], 3);
    }

    #[test]
    fn replay_owners_only_bypass_the_helicity_sum_through_nested_physical_runtimes() {
        let (mut runtime, selected_helicity_ids) = runtime_with_singleton_helicity_schedule(false);
        assert!(
            runtime
                .has_safe_singleton_helicity_execution(Some(&selected_helicity_ids), None)
                .unwrap()
        );

        runtime.color_topology_replay_enabled = true;
        assert!(
            !runtime
                .has_safe_singleton_helicity_execution(Some(&selected_helicity_ids), None)
                .unwrap()
        );

        runtime
            .helicity_selector_runtimes
            .push(Box::new(super::super::tests::empty_generic_runtime()));
        runtime
            .helicity_selector_runtime_schedule_modes
            .push(HelicitySelectorScheduleMode::NestedRuntime);
        runtime.helicity_selector_lane_by_domain.insert(0, 0);
        assert!(
            runtime
                .has_safe_singleton_helicity_execution(Some(&selected_helicity_ids), None)
                .unwrap()
        );

        runtime.helicity_selector_runtime_schedule_modes[0] =
            HelicitySelectorScheduleMode::ParentClosure;
        assert!(
            !runtime
                .has_safe_singleton_helicity_execution(Some(&selected_helicity_ids), None)
                .unwrap()
        );
    }

    #[test]
    fn structural_zero_singletons_are_safe_without_physical_replay() {
        let (mut runtime, selected_helicity_ids) = runtime_with_singleton_helicity_schedule(true);
        runtime.color_topology_replay_enabled = true;

        assert!(
            runtime
                .has_safe_singleton_helicity_execution(Some(&selected_helicity_ids), None)
                .unwrap()
        );
        assert!(
            !runtime
                .has_safe_singleton_helicity_execution(None, None)
                .unwrap()
        );
        assert!(
            !runtime
                .has_safe_singleton_helicity_execution(
                    Some(&BTreeSet::from([
                        "hel:+-".to_string(),
                        "hel:-+".to_string(),
                    ])),
                    None,
                )
                .unwrap()
        );
    }

    #[test]
    fn selected_lc_flows_keep_their_compact_replay_even_with_a_helicity_companion() {
        let (mut runtime, selected_helicity_ids) = runtime_with_singleton_helicity_schedule(false);
        runtime.physics = Some(Arc::new(super::super::tests::test_physics_runtime("lc")));
        runtime.lc_topology_replay_enabled = true;
        runtime
            .helicity_selector_runtimes
            .push(Box::new(super::super::tests::empty_generic_runtime()));
        runtime
            .helicity_selector_runtime_schedule_modes
            .push(HelicitySelectorScheduleMode::NestedRuntime);
        runtime.helicity_selector_lane_by_domain.insert(0, 0);

        // Both the computed flow and a nontrivial replay alias must be mapped
        // before selecting the helicity; neither belongs to the all-flow lane.
        for flow_id in ["flow:0", "flow:1"] {
            assert!(
                !runtime
                    .has_safe_singleton_helicity_execution(
                        Some(&selected_helicity_ids),
                        Some(&BTreeSet::from([flow_id.to_string()])),
                    )
                    .unwrap()
            );
        }
        assert!(
            runtime
                .has_safe_singleton_helicity_execution(Some(&selected_helicity_ids), None)
                .unwrap()
        );
        assert!(
            runtime
                .has_safe_singleton_helicity_execution(
                    Some(&selected_helicity_ids),
                    Some(&BTreeSet::from([
                        "flow:0".to_string(),
                        "flow:1".to_string()
                    ])),
                )
                .unwrap()
        );
        assert!(
            !runtime
                .has_safe_singleton_helicity_execution(
                    None,
                    Some(&BTreeSet::from(["flow:0".to_string()])),
                )
                .unwrap()
        );
    }

    #[test]
    fn contracted_color_and_structural_zero_requests_keep_the_singleton_shortcut() {
        for accuracy in ["full", "nlc"] {
            let (mut runtime, selected_helicity_ids) =
                runtime_with_singleton_helicity_schedule(false);
            runtime.physics = Some(Arc::new(super::super::tests::test_physics_runtime(
                accuracy,
            )));
            assert!(
                runtime
                    .has_safe_singleton_helicity_execution(
                        Some(&selected_helicity_ids),
                        Some(&BTreeSet::from(["contracted".to_string()])),
                    )
                    .unwrap()
            );
        }

        let (mut runtime, selected_helicity_ids) = runtime_with_singleton_helicity_schedule(true);
        runtime.physics = Some(Arc::new(super::super::tests::test_physics_runtime("lc")));
        runtime.lc_topology_replay_enabled = true;
        assert!(
            runtime
                .has_safe_singleton_helicity_execution(
                    Some(&selected_helicity_ids),
                    Some(&BTreeSet::from(["flow:1".to_string()])),
                )
                .unwrap()
        );
    }
}
