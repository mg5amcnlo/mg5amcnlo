// SPDX-License-Identifier: 0BSD

//! Private native LC lane for selector-local on-the-fly recurrence.
//!
//! This lane deliberately owns only the compact process seed, authenticated
//! model catalogs, and one warmed query-family executor.  It never accepts or
//! materializes a [`crate::recurrence::DirectRecurrencePlan`], a dense flow
//! catalog, or a process-wide recurrence schedule.  Public selectors are
//! decoded by the existing runtime adapter before they reach this module.
//! Likewise, load/evaluate/profile/benchmark need only this compact state.
//! The existing physics-metadata API can hang a separate lazy cache from
//! [`OnTheFlyNativeRuntime::seed`], deriving dense public axes only when a
//! caller explicitly requests introspection; that cache must not participate
//! in this lane's load or warmed-evaluation path.

use super::on_the_fly_selectors::OnTheFlyCompactSelectorAdapterV1;
use super::on_the_fly_warm_up::{
    NativeOnTheFlyWarmUpEventKind, NativeOnTheFlyWarmUpStage, OnTheFlyWarmUpProgress,
};
use super::recurrence_backend::NativeOnTheFlyPreparedExecutorResolver;
use super::recurrence_lane::{
    DirectProfileDelta, LcResolvedOutputLayout, PreparedParameterProjectionEntry,
    accumulate_lc_diagonal_amplitude, direct_profile_from_delta,
    projected_prepared_parameter_values,
};
use super::recurrence_load::on_the_fly_source_major_momenta_into;
use super::*;
use crate::direct_arena::DirectArenaTrafficCounters;
#[cfg(feature = "on-the-fly-test-support")]
use crate::recurrence::AuthenticatedRecurrenceBuilderInput;
use crate::recurrence::direct_backend::DirectExecutionCounters;
use crate::recurrence::direct_runtime::DirectRuntimeActivityCounters;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
use crate::recurrence::on_the_fly::OnTheFlyCouplingPolicyCensusV1;
#[cfg(feature = "on-the-fly-test-support")]
use crate::recurrence::on_the_fly::build_on_the_fly_selected_trace_against_seed_v1;
use crate::recurrence::on_the_fly::{
    DecodedLcQueryV1, OnTheFlyProcessSeedV1, OnTheFlyQueryConstructionPoolV1,
    OnTheFlyQueryFamilyCensusV1, OnTheFlyQueryFamilyExecutionReportV1,
    OnTheFlyQueryFamilyExecutorV1, OnTheFlyQueryFamilyHandleV1, OnTheFlyResolvedCouplingPolicyV1,
    OnTheFlySelectedQueryOutcomeV1, PreparedOnTheFlyGrammarV1, QueryFamilyTraceInput,
    build_selected_lc_query_family_v1, build_streamed_query_family_candidate_v1,
    prepare_on_the_fly_process_v1,
};
use crate::recurrence::template::ValidatedRecurrenceTemplateInput;
use crate::recurrence::{
    PreparedDirectExecutorCatalog, RecurrenceColorContraction, RuntimeColorContractionEntry,
    RuntimeColorContractionReducer, RuntimeSymmetricGroupColorWorkspace,
};
use std::time::{Duration, Instant};

#[path = "on_the_fly_lane_cache.rs"]
mod cache;
pub(super) use cache::OnTheFlyLaneSnapshotV1;

#[cfg(feature = "on-the-fly-test-support")]
#[derive(Clone, Debug, PartialEq)]
pub(super) struct OnTheFlyExecutionDiagnosticCurrentV1 {
    pub(super) semantic_digest: crate::recurrence::SemanticDigest,
    pub(super) stage: u32,
    pub(super) values: Vec<(f64, f64)>,
}

#[cfg(feature = "on-the-fly-test-support")]
#[derive(Clone, Debug, PartialEq)]
pub(super) struct OnTheFlyExecutionDiagnosticSnapshotV1 {
    pub(super) seed_digest: crate::recurrence::SemanticDigest,
    pub(super) query_digest: crate::recurrence::SemanticDigest,
    pub(super) trace_digest: crate::recurrence::SemanticDigest,
    pub(super) raw_amplitude: (f64, f64),
    pub(super) prepared_parameters: Vec<(f64, f64)>,
    pub(super) currents: Vec<OnTheFlyExecutionDiagnosticCurrentV1>,
}

/// One public resolved-output destination represented by a selector-local
/// trace.  The coefficient is the established helicity-orbit times LC color
/// weight; process normalization remains a call-local runtime input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct OnTheFlyLcReductionTargetV1 {
    helicity_position: usize,
    color_position: usize,
    coefficient: f64,
}

impl OnTheFlyLcReductionTargetV1 {
    pub(super) fn new(
        helicity_position: usize,
        color_position: usize,
        coefficient: f64,
    ) -> RusticolResult<Self> {
        if !coefficient.is_finite() || coefficient < 0.0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC reduction coefficient must be finite and nonnegative",
            ));
        }
        Ok(Self {
            helicity_position,
            color_position,
            coefficient,
        })
    }
}

/// Existing-runtime selector preparation expressed as one compact decoded LC
/// query plus its public resolved placements.  A representative helicity may
/// legitimately address several physical orbit members.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct OnTheFlyLcQueryRequestV1 {
    query: DecodedLcQueryV1,
    reduction_targets: Box<[OnTheFlyLcReductionTargetV1]>,
}

impl OnTheFlyLcQueryRequestV1 {
    pub(super) fn new(
        query: DecodedLcQueryV1,
        reduction_targets: Vec<OnTheFlyLcReductionTargetV1>,
    ) -> RusticolResult<Self> {
        if reduction_targets.is_empty() {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC query has no selected reduction target",
            ));
        }
        let mut coordinates = reduction_targets
            .iter()
            .map(|target| (target.helicity_position, target.color_position))
            .collect::<Vec<_>>();
        coordinates.sort_unstable();
        if coordinates.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC query repeats a resolved reduction target",
            ));
        }
        let total_coefficient = reduction_targets
            .iter()
            .try_fold(0.0_f64, |total, target| {
                let total = total + target.coefficient;
                total.is_finite().then_some(total).ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "on-the-fly LC reduction coefficient sum is not finite",
                    )
                })
            })?;
        if total_coefficient == 0.0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC query has zero total reduction weight",
            ));
        }
        Ok(Self {
            query,
            reduction_targets: reduction_targets.into_boxed_slice(),
        })
    }

    fn total_coefficient(&self) -> f64 {
        self.reduction_targets
            .iter()
            .map(|target| target.coefficient)
            .sum()
    }
}

struct PreparedOnTheFlyLcFamilyV1 {
    requests: Box<[OnTheFlyLcQueryRequestV1]>,
    amplitude_destinations: Box<[Option<usize>]>,
    executor_handle: Option<OnTheFlyQueryFamilyHandleV1>,
    census: Option<OnTheFlyQueryFamilyCensusV1>,
    logical_point_capacity: u32,
}

/// Retained contracted selection state.  Decoded structural queries and
/// their traces are deliberately absent: after the union family is prepared,
/// execution needs only the selected public-helicity identity and the compact
/// H x S projection from authenticated metric destinations to union-family
/// amplitude destinations.
struct PreparedOnTheFlyContractedFamilyV1 {
    helicity_ordinals: Box<[usize]>,
    structural_color_count: usize,
    amplitude_destinations: Box<[Option<usize>]>,
    executor_handle: Option<OnTheFlyQueryFamilyHandleV1>,
    census: Option<OnTheFlyQueryFamilyCensusV1>,
    logical_point_capacity: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct OnTheFlyRetainedStateCensusV1 {
    pub(super) family_count: usize,
    pub(super) request_count: usize,
    pub(super) amplitude_destination_count: usize,
    pub(super) executor_handle_count: usize,
    pub(super) query_local_trace_count: usize,
    pub(super) embedded_lookup_key_count: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct OnTheFlyProcessPreparationCensusV1 {
    pub(super) catalog_validation_count: u64,
    pub(super) grammar_preparation_count: u64,
}

/// Process-local production owner for the private on-the-fly lane.
///
/// Field order is intentional: the family executor (and therefore every row
/// table borrowed by a prepared kernel) is dropped before the catalogs and
/// compact process seed from which its traces were constructed.
pub(super) struct OnTheFlyNativeRuntime {
    executor: OnTheFlyQueryFamilyExecutorV1<NativeOnTheFlyPreparedExecutorResolver>,
    templates: ValidatedRecurrenceTemplateInput,
    direct_catalog: PreparedDirectExecutorCatalog,
    seed: OnTheFlyProcessSeedV1,
    requested_query_construction_threads: usize,
    effective_query_construction_threads: usize,
    prepared_grammar: Option<PreparedOnTheFlyGrammarV1>,
    coupling_policy: Option<OnTheFlyResolvedCouplingPolicyV1>,
    coupling_policy_resolution: Option<Duration>,
    process_preparation_count: u64,
    #[cfg(test)]
    process_preparation_census: OnTheFlyProcessPreparationCensusV1,
    parameter_defaults: Box<[crate::EagerComplex64]>,
    parameter_projection: Box<[PreparedParameterProjectionEntry]>,
    runtime_parameter_values: Box<[f64]>,
    families: Vec<PreparedOnTheFlyLcFamilyV1>,
    last_family: Option<usize>,
    pending_family: Option<PreparedOnTheFlyLcFamilyV1>,
    contracted_family: Option<PreparedOnTheFlyContractedFamilyV1>,
    pending_contracted_family: Option<PreparedOnTheFlyContractedFamilyV1>,
    #[cfg(test)]
    contracted_test_execution_attempt: usize,
    #[cfg(test)]
    contracted_test_fail_execution_at: Option<usize>,
    #[cfg(test)]
    contracted_max_live_query_outcomes: usize,
    source_momenta_scratch: Vec<f64>,
    amplitude_scratch: Vec<(f64, f64)>,
    symmetric_group_color_workspace: Option<RuntimeSymmetricGroupColorWorkspace>,
}

fn emit_progress(
    progress: Option<&mut OnTheFlyWarmUpProgress<'_>>,
    kind: NativeOnTheFlyWarmUpEventKind,
    stage: NativeOnTheFlyWarmUpStage,
    completed: usize,
    total: usize,
    message: Option<&str>,
) -> RusticolResult<()> {
    if let Some(progress) = progress {
        progress.emit(kind, stage, completed, total, message)?;
    }
    Ok(())
}

/// Map one owner-ordered selector request into the group-ordered amplitude
/// projection retained by contracted execution.  The inverse owner map is
/// authenticated at load, so every later reducer consumes this table directly
/// in normalized local-group order.
fn contracted_projection_index(
    helicity_position: usize,
    owner_ordinal: usize,
    structural_color_count: usize,
    destination_by_owner_ordinal: &[u32],
) -> RusticolResult<usize> {
    let group = usize::try_from(*destination_by_owner_ordinal.get(owner_ordinal).ok_or_else(
        || {
            RusticolError::integrity(
                "on-the-fly contracted owner is outside its destination mapping",
            )
        },
    )?)
    .map_err(|_| RusticolError::artifact("on-the-fly contracted destination exceeds usize"))?;
    if group >= structural_color_count {
        return Err(RusticolError::integrity(
            "on-the-fly contracted destination is outside the structural group domain",
        ));
    }
    helicity_position
        .checked_mul(structural_color_count)
        .and_then(|offset| offset.checked_add(group))
        .ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly contracted projection exceeds usize")
        })
}

fn emit_reused_family_progress(
    mut progress: Option<&mut OnTheFlyWarmUpProgress<'_>>,
    query_count: usize,
) -> RusticolResult<()> {
    emit_progress(
        progress.as_deref_mut(),
        NativeOnTheFlyWarmUpEventKind::Start,
        NativeOnTheFlyWarmUpStage::QueryFamily,
        query_count,
        query_count,
        Some("query family already retained"),
    )?;
    emit_progress(
        progress.as_deref_mut(),
        NativeOnTheFlyWarmUpEventKind::End,
        NativeOnTheFlyWarmUpStage::QueryFamily,
        query_count,
        query_count,
        Some("query family reused"),
    )?;
    emit_progress(
        progress.as_deref_mut(),
        NativeOnTheFlyWarmUpEventKind::Start,
        NativeOnTheFlyWarmUpStage::FamilyFinalization,
        1,
        1,
        Some("family finalization already retained"),
    )?;
    emit_progress(
        progress,
        NativeOnTheFlyWarmUpEventKind::End,
        NativeOnTheFlyWarmUpStage::FamilyFinalization,
        1,
        1,
        Some("family finalization reused"),
    )
}

impl OnTheFlyNativeRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        templates: ValidatedRecurrenceTemplateInput,
        direct_catalog: PreparedDirectExecutorCatalog,
        seed: OnTheFlyProcessSeedV1,
        resolver: NativeOnTheFlyPreparedExecutorResolver,
        requested_query_construction_threads: usize,
        effective_query_construction_threads: usize,
        parameter_defaults: Vec<crate::EagerComplex64>,
        parameter_projection: Vec<PreparedParameterProjectionEntry>,
        runtime_parameter_values: &[f64],
    ) -> RusticolResult<Self> {
        let summary = templates.summary();
        if seed.template_catalog_digest() != summary.catalog_digest
            || seed.model_digest() != summary.compiled_model_digest
            || seed.prepared_pack_digest() != summary.prepared_kernel_pack_digest
            || seed.direct_catalog_digest() != direct_catalog.direct_template_catalog_digest()
        {
            return Err(RusticolError::integrity(
                "on-the-fly native lane catalogs do not match its compact process seed",
            ));
        }
        if parameter_defaults.len()
            != usize::try_from(summary.parameter_count)
                .map_err(|_| RusticolError::artifact("on-the-fly parameter count exceeds usize"))?
        {
            return Err(RusticolError::integrity(
                "on-the-fly prepared parameter defaults have the wrong size",
            ));
        }
        if runtime_parameter_values
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(RusticolError::invalid_argument(
                "on-the-fly runtime parameter value is not finite",
            ));
        }
        if requested_query_construction_threads == 0
            || effective_query_construction_threads == 0
            || effective_query_construction_threads > requested_query_construction_threads
        {
            return Err(RusticolError::invalid_argument(
                "on-the-fly effective query construction threads must be in the positive requested domain",
            ));
        }

        let prepared_parameters = projected_prepared_parameter_values(
            &parameter_defaults,
            &parameter_projection,
            runtime_parameter_values,
        )?;
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        executor.set_parameters(&prepared_parameters)?;
        Ok(Self {
            executor,
            templates,
            direct_catalog,
            seed,
            requested_query_construction_threads,
            effective_query_construction_threads,
            prepared_grammar: None,
            coupling_policy: None,
            coupling_policy_resolution: None,
            process_preparation_count: 0,
            #[cfg(test)]
            process_preparation_census: OnTheFlyProcessPreparationCensusV1::default(),
            parameter_defaults: parameter_defaults.into_boxed_slice(),
            parameter_projection: parameter_projection.into_boxed_slice(),
            runtime_parameter_values: runtime_parameter_values.to_vec().into_boxed_slice(),
            families: Vec::new(),
            last_family: None,
            pending_family: None,
            contracted_family: None,
            pending_contracted_family: None,
            #[cfg(test)]
            contracted_test_execution_attempt: 0,
            #[cfg(test)]
            contracted_test_fail_execution_at: None,
            #[cfg(test)]
            contracted_max_live_query_outcomes: 0,
            source_momenta_scratch: Vec::new(),
            amplitude_scratch: Vec::new(),
            symmetric_group_color_workspace: None,
        })
    }

    pub(super) fn install_symmetric_group_color_workspace(
        &mut self,
        workspace: RuntimeSymmetricGroupColorWorkspace,
    ) -> RusticolResult<()> {
        if self.symmetric_group_color_workspace.is_some() {
            return Err(RusticolError::internal(
                "on-the-fly symmetric-group color workspace was installed twice",
            ));
        }
        self.symmetric_group_color_workspace = Some(workspace);
        Ok(())
    }

    pub(super) const fn seed(&self) -> &OnTheFlyProcessSeedV1 {
        &self.seed
    }

    pub(super) const fn requested_query_construction_threads(&self) -> usize {
        self.requested_query_construction_threads
    }

    pub(super) const fn effective_query_construction_threads(&self) -> usize {
        self.effective_query_construction_threads
    }

    /// Execute one retained public query through the production batched
    /// query-family path and expose point-zero current values for diagnostics.
    /// This remains absent from release builds and does not alter row grouping.
    #[cfg(feature = "on-the-fly-test-support")]
    pub(super) fn execution_diagnostic_v1(
        &mut self,
        authenticated: &AuthenticatedRecurrenceBuilderInput,
        selected_public_flow_id: u32,
        public_helicities: &[i32],
        point_major_momenta: &[f64],
        runtime_parameters: &[f64],
    ) -> RusticolResult<OnTheFlyExecutionDiagnosticSnapshotV1> {
        let selected = build_on_the_fly_selected_trace_against_seed_v1(
            authenticated,
            &self.direct_catalog,
            &self.seed,
            selected_public_flow_id,
            public_helicities,
            true,
            self.symmetric_group_color_workspace.is_some(),
        )?;
        let request = OnTheFlyLcQueryRequestV1::new(
            selected.query.clone(),
            vec![OnTheFlyLcReductionTargetV1::new(0, 0, 1.0)?],
        )?;
        self.prepare_lc_queries(std::slice::from_ref(&request), 1)?;
        let destination = self
            .current_family()
            .and_then(|family| family.amplitude_destinations.first())
            .copied()
            .flatten()
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly execution diagnostic selected a structural-zero query",
                )
            })?;
        let _ = self.execute_amplitudes(point_major_momenta, 1, runtime_parameters)?;
        let raw_amplitude = *self
            .amplitude_scratch
            .get(destination)
            .ok_or_else(|| RusticolError::integrity("on-the-fly diagnostic amplitude is absent"))?;
        let currents = self
            .executor
            .observed_currents(0)?
            .into_iter()
            .map(|current| OnTheFlyExecutionDiagnosticCurrentV1 {
                semantic_digest: current.semantic_digest,
                stage: current.stage,
                values: current.values,
            })
            .collect();
        let prepared_parameters = projected_prepared_parameter_values(
            &self.parameter_defaults,
            &self.parameter_projection,
            runtime_parameters,
        )?;
        Ok(OnTheFlyExecutionDiagnosticSnapshotV1 {
            seed_digest: selected.seed.semantic_digest(),
            query_digest: selected.query.semantic_digest(),
            trace_digest: selected.trace.semantic_digest(),
            raw_amplitude,
            prepared_parameters,
            currents,
        })
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) fn coupling_policy_census(&self) -> Option<OnTheFlyCouplingPolicyCensusV1> {
        self.coupling_policy.as_ref().map(|policy| policy.census())
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) const fn coupling_policy_resolution(&self) -> Option<Duration> {
        self.coupling_policy_resolution
    }

    /// Private read-only production introspection of the existing preparation count.
    pub(super) const fn process_preparation_count(&self) -> u64 {
        self.process_preparation_count
    }

    #[cfg(test)]
    pub(super) const fn process_preparation_census(&self) -> OnTheFlyProcessPreparationCensusV1 {
        self.process_preparation_census
    }

    fn ensure_process_prepared(&mut self) -> RusticolResult<()> {
        match (&self.prepared_grammar, &self.coupling_policy) {
            (Some(_), Some(_)) => return Ok(()),
            (None, None) => {}
            _ => {
                return Err(RusticolError::internal(
                    "on-the-fly process preparation is only partially initialized",
                ));
            }
        }
        let started = Instant::now();
        let (grammar, policy) = prepare_on_the_fly_process_v1(&self.templates, &self.seed)?;
        #[cfg(test)]
        {
            self.process_preparation_census.catalog_validation_count = self
                .process_preparation_census
                .catalog_validation_count
                .checked_add(1)
                .ok_or_else(|| {
                    RusticolError::internal("on-the-fly catalog-validation count exceeds u64")
                })?;
            self.process_preparation_census.grammar_preparation_count = self
                .process_preparation_census
                .grammar_preparation_count
                .checked_add(1)
                .ok_or_else(|| {
                    RusticolError::internal("on-the-fly grammar-preparation count exceeds u64")
                })?;
        }
        self.process_preparation_count =
            self.process_preparation_count
                .checked_add(1)
                .ok_or_else(|| {
                    RusticolError::internal("on-the-fly process preparation count exceeds u64")
                })?;
        self.prepared_grammar = Some(grammar);
        self.coupling_policy = Some(policy);
        self.coupling_policy_resolution = Some(started.elapsed());
        Ok(())
    }

    pub(super) fn prepare_process_for_warm_up(
        &mut self,
        progress: &mut OnTheFlyWarmUpProgress<'_>,
    ) -> RusticolResult<bool> {
        let already_prepared = self.prepared_grammar.is_some() && self.coupling_policy.is_some();
        progress.emit(
            NativeOnTheFlyWarmUpEventKind::Start,
            NativeOnTheFlyWarmUpStage::ProcessPreparation,
            usize::from(already_prepared),
            1,
            already_prepared.then_some("process preparation already retained"),
        )?;
        self.ensure_process_prepared()?;
        progress.emit(
            NativeOnTheFlyWarmUpEventKind::End,
            NativeOnTheFlyWarmUpStage::ProcessPreparation,
            1,
            1,
            already_prepared.then_some("process preparation reused"),
        )?;
        Ok(already_prepared)
    }

    pub(super) fn report_reused_family_for_warm_up(
        &self,
        query_count: usize,
        progress: &mut OnTheFlyWarmUpProgress<'_>,
    ) -> RusticolResult<()> {
        emit_reused_family_progress(Some(progress), query_count)
    }

    fn prepare_streamed_lc_candidate_for_warm_up(
        &mut self,
        requests: &[OnTheFlyLcQueryRequestV1],
        logical_point_capacity: u32,
        query_construction_threads: usize,
        progress: &mut OnTheFlyWarmUpProgress<'_>,
    ) -> RusticolResult<(bool, PreparedOnTheFlyLcFamilyV1)> {
        let enable_cyclic_trace_reflection = self.symmetric_group_color_workspace.is_some();
        let coupling_policy = self.coupling_policy.as_ref().ok_or_else(|| {
            RusticolError::internal("on-the-fly coupling policy disappeared after preparation")
        })?;
        let grammar = self.prepared_grammar.as_ref().ok_or_else(|| {
            RusticolError::internal("on-the-fly grammar disappeared after preparation")
        })?;
        let worker_count = query_construction_threads.min(requests.len()).max(1);
        let query_pool = OnTheFlyQueryConstructionPoolV1::new(worker_count)?;
        let mut amplitude_destinations = Vec::new();
        amplitude_destinations
            .try_reserve_exact(requests.len())
            .map_err(|error| {
                RusticolError::invalid_argument(format!(
                    "on-the-fly LC warm-up projection allocation failed: {error}"
                ))
            })?;
        amplitude_destinations.resize(requests.len(), None);

        let templates = &self.templates;
        let direct_catalog = &self.direct_catalog;
        let seed = &self.seed;
        let executor = &mut self.executor;
        let mut binding_started = false;
        let mut processed_query_count = 0_usize;
        let streamed = build_streamed_query_family_candidate_v1(direct_catalog, |consumer| {
            while processed_query_count < requests.len() {
                let chunk_end = processed_query_count
                    .saturating_add(query_pool.chunk_capacity())
                    .min(requests.len());
                let expected_queries = requests[processed_query_count..chunk_end]
                    .iter()
                    .map(|request| grammar.canonicalize_query(seed, request.query.clone()))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let outcomes = query_pool.build_chunk(
                    templates,
                    direct_catalog,
                    seed,
                    coupling_policy,
                    grammar,
                    enable_cyclic_trace_reflection,
                    expected_queries.clone(),
                )?;
                if outcomes.len() != expected_queries.len() {
                    return Err(RusticolError::integrity(
                        "on-the-fly LC warm-up trace builder changed a chunk count",
                    ));
                }

                let mut traces = Vec::new();
                let mut projection_slots = Vec::new();
                traces.try_reserve_exact(outcomes.len()).map_err(|error| {
                    RusticolError::invalid_argument(format!(
                        "on-the-fly LC warm-up trace-chunk allocation failed: {error}"
                    ))
                })?;
                projection_slots
                    .try_reserve_exact(outcomes.len())
                    .map_err(|error| {
                        RusticolError::invalid_argument(format!(
                            "on-the-fly LC warm-up projection-chunk allocation failed: {error}"
                        ))
                    })?;
                for (chunk_offset, (expected_query, outcome)) in
                    expected_queries.into_iter().zip(outcomes).enumerate()
                {
                    let request_index = processed_query_count + chunk_offset;
                    match outcome {
                        OnTheFlySelectedQueryOutcomeV1::Trace(selected) => {
                            if selected.query != expected_query {
                                return Err(RusticolError::integrity(
                                    "on-the-fly LC warm-up trace query changed during construction",
                                ));
                            }
                            projection_slots.push(request_index);
                            traces.push(selected);
                        }
                        OnTheFlySelectedQueryOutcomeV1::StructuralZero { query } => {
                            if query != expected_query {
                                return Err(RusticolError::integrity(
                                    "on-the-fly LC warm-up structural-zero query changed during construction",
                                ));
                            }
                        }
                    }
                }
                if !traces.is_empty() {
                    if !binding_started {
                        executor
                            .resolver_mut()
                            .begin_on_the_fly_family_binding(templates, direct_catalog)?;
                        binding_started = true;
                    }
                    executor.resolver_mut().extend_on_the_fly_family_binding(
                        templates,
                        direct_catalog,
                        seed,
                        traces.iter_mut().map(|selected| &mut selected.trace),
                    )?;
                    let inputs = traces
                        .iter()
                        .map(|selected| QueryFamilyTraceInput {
                            trace: &selected.trace,
                            projection: selected.projection,
                        })
                        .collect::<Vec<_>>();
                    let destinations = consumer.push_chunk(&inputs)?;
                    if destinations.len() != projection_slots.len() {
                        return Err(RusticolError::integrity(
                            "on-the-fly LC warm-up streamed destination range changed a chunk count",
                        ));
                    }
                    for (request_index, destination) in
                        projection_slots.into_iter().zip(destinations)
                    {
                        let slot =
                            amplitude_destinations
                                .get_mut(request_index)
                                .ok_or_else(|| {
                                    RusticolError::integrity(
                                        "on-the-fly LC warm-up projection slot is absent",
                                    )
                                })?;
                        if slot.replace(destination as usize).is_some() {
                            return Err(RusticolError::integrity(
                                "on-the-fly LC warm-up projection slot is repeated",
                            ));
                        }
                    }
                }
                processed_query_count = chunk_end;
                progress.emit(
                    NativeOnTheFlyWarmUpEventKind::Update,
                    NativeOnTheFlyWarmUpStage::QueryFamily,
                    processed_query_count,
                    requests.len(),
                    None,
                )?;
            }
            Ok(())
        })?;
        if processed_query_count != requests.len() {
            return Err(RusticolError::integrity(
                "on-the-fly LC warm-up streamed construction omitted a selected query",
            ));
        }
        progress.emit(
            NativeOnTheFlyWarmUpEventKind::End,
            NativeOnTheFlyWarmUpStage::QueryFamily,
            requests.len(),
            requests.len(),
            None,
        )?;
        progress.emit(
            NativeOnTheFlyWarmUpEventKind::Start,
            NativeOnTheFlyWarmUpStage::FamilyFinalization,
            0,
            1,
            None,
        )?;
        let (cache_hit, census) = if let Some(streamed) = streamed {
            if !binding_started {
                return Err(RusticolError::integrity(
                    "on-the-fly LC warm-up streamed family has no semantic binding transaction",
                ));
            }
            let cache_hit =
                executor.prepare_streamed_candidate(streamed, logical_point_capacity)?;
            let census = executor.prepared_census().ok_or_else(|| {
                RusticolError::integrity("on-the-fly LC warm-up family has no work census")
            })?;
            let projected_nonzero_count = amplitude_destinations
                .iter()
                .filter(|destination| destination.is_some())
                .count();
            if usize::try_from(census.union_amplitude_destination_count).ok()
                != Some(projected_nonzero_count)
            {
                return Err(RusticolError::integrity(
                    "on-the-fly LC warm-up family does not retain one destination per nonzero query",
                ));
            }
            (cache_hit, Some(census))
        } else {
            if binding_started {
                return Err(RusticolError::integrity(
                    "on-the-fly LC warm-up all-zero family retained semantic bindings",
                ));
            }
            (false, None)
        };
        progress.emit(
            NativeOnTheFlyWarmUpEventKind::End,
            NativeOnTheFlyWarmUpStage::FamilyFinalization,
            1,
            1,
            None,
        )?;
        Ok((
            cache_hit,
            PreparedOnTheFlyLcFamilyV1 {
                requests: requests.to_vec().into_boxed_slice(),
                amplitude_destinations: amplitude_destinations.into_boxed_slice(),
                executor_handle: None,
                census,
                logical_point_capacity,
            },
        ))
    }

    /// Cold selector/trace construction. Repeating an identical public
    /// selection reuses its retained requests, projections, and warmed row
    /// family; increasing only the point capacity replaces numeric workspace.
    pub(super) fn prepare_lc_queries(
        &mut self,
        requests: &[OnTheFlyLcQueryRequestV1],
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        self.prepare_lc_queries_impl(requests, logical_point_capacity, None, None)
    }

    pub(super) fn prepare_lc_queries_for_warm_up(
        &mut self,
        requests: &[OnTheFlyLcQueryRequestV1],
        logical_point_capacity: u32,
        query_construction_threads: usize,
        progress: &mut OnTheFlyWarmUpProgress<'_>,
    ) -> RusticolResult<bool> {
        self.prepare_lc_queries_impl(
            requests,
            logical_point_capacity,
            Some(query_construction_threads),
            Some(progress),
        )
    }

    fn prepare_lc_queries_impl(
        &mut self,
        requests: &[OnTheFlyLcQueryRequestV1],
        logical_point_capacity: u32,
        query_construction_threads: Option<usize>,
        mut progress: Option<&mut OnTheFlyWarmUpProgress<'_>>,
    ) -> RusticolResult<bool> {
        if requests.is_empty() || logical_point_capacity == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC preparation requires queries and a nonzero point capacity",
            ));
        }
        if self.contracted_family.is_some() || self.pending_contracted_family.is_some() {
            return Err(RusticolError::internal(
                "on-the-fly lane cannot mix LC and contracted retained families",
            ));
        }
        if requests
            .iter()
            .any(|request| request.query.process_seed_digest() != self.seed.semantic_digest())
        {
            return Err(RusticolError::integrity(
                "on-the-fly LC query belongs to a different compact process seed",
            ));
        }

        // The common same-selector path compares the sole retained family
        // directly. When the numeric
        // workspace is already large enough it also avoids a redundant second
        // identity walk inside the executor.
        if self.pending_family.is_none()
            && self.last_family.is_some_and(|index| {
                self.families.get(index).is_some_and(|family| {
                    family.requests.as_ref() == requests
                        && logical_point_capacity <= family.logical_point_capacity
                })
            })
        {
            emit_reused_family_progress(progress.as_deref_mut(), requests.len())?;
            return Ok(true);
        }
        if self
            .pending_family
            .as_ref()
            .is_some_and(|family| family.requests.as_ref() == requests)
        {
            let (current_capacity, has_executable_family) = self
                .pending_family
                .as_ref()
                .map(|family| (family.logical_point_capacity, family.census.is_some()))
                .expect("matching on-the-fly pending family disappeared");
            if logical_point_capacity > current_capacity
                && has_executable_family
                && let Err(error) = self.executor.resize_active_family(logical_point_capacity)
            {
                self.discard_pending_lc_queries()?;
                return Err(error);
            }
            let family = self
                .pending_family
                .as_mut()
                .expect("matching on-the-fly pending family disappeared after resize");
            family.logical_point_capacity =
                family.logical_point_capacity.max(logical_point_capacity);
            emit_reused_family_progress(progress.as_deref_mut(), requests.len())?;
            return Ok(false);
        }
        // A different uncommitted selector is a failed/superseded candidate,
        // not another cache entry.
        self.discard_pending_lc_queries()?;

        let retained = self.last_family.filter(|&index| {
            self.families
                .get(index)
                .is_some_and(|family| family.requests.as_ref() == requests)
        });
        if let Some(index) = retained {
            let family = self
                .families
                .get_mut(index)
                .expect("on-the-fly retained family index is absent");
            let cache_hit = if let Some(handle) = family.executor_handle {
                self.executor
                    .activate_retained_family(handle, logical_point_capacity)?
            } else {
                true
            };
            family.logical_point_capacity =
                family.logical_point_capacity.max(logical_point_capacity);
            self.pending_family = None;
            self.last_family = Some(index);
            emit_reused_family_progress(progress.as_deref_mut(), requests.len())?;
            return Ok(cache_hit);
        }

        if self.families.capacity() == 0 {
            self.families.try_reserve_exact(1).map_err(|error| {
                RusticolError::invalid_argument(format!(
                    "on-the-fly retained-family allocation failed: {error}"
                ))
            })?;
        }
        let candidate_result = (|| {
            emit_progress(
                progress.as_deref_mut(),
                NativeOnTheFlyWarmUpEventKind::Start,
                NativeOnTheFlyWarmUpStage::QueryFamily,
                0,
                requests.len(),
                None,
            )?;
            self.ensure_process_prepared()?;
            if let Some(progress) = progress.as_deref_mut() {
                return self.prepare_streamed_lc_candidate_for_warm_up(
                    requests,
                    logical_point_capacity,
                    query_construction_threads.expect("warm-up worker count is absent"),
                    progress,
                );
            }
            let enable_cyclic_trace_reflection = self.symmetric_group_color_workspace.is_some();
            let coupling_policy = self.coupling_policy.as_ref().ok_or_else(|| {
                RusticolError::internal("on-the-fly coupling policy disappeared after preparation")
            })?;
            let grammar = self.prepared_grammar.as_ref().ok_or_else(|| {
                RusticolError::internal("on-the-fly grammar disappeared after preparation")
            })?;
            let expected_queries = requests
                .iter()
                .map(|request| grammar.canonicalize_query(&self.seed, request.query.clone()))
                .collect::<RusticolResult<Vec<_>>>()?;
            let outcomes = build_selected_lc_query_family_v1(
                &self.templates,
                &self.direct_catalog,
                &self.seed,
                coupling_policy,
                grammar,
                enable_cyclic_trace_reflection,
                self.effective_query_construction_threads,
                expected_queries.iter().cloned(),
            )?;
            if outcomes.len() != requests.len() {
                return Err(RusticolError::integrity(
                    "on-the-fly batch trace builder changed the query count",
                ));
            }
            emit_progress(
                progress.as_deref_mut(),
                NativeOnTheFlyWarmUpEventKind::End,
                NativeOnTheFlyWarmUpStage::QueryFamily,
                requests.len(),
                requests.len(),
                None,
            )?;
            let mut traces = Vec::new();
            let mut amplitude_destinations = vec![None; requests.len()];
            traces.try_reserve_exact(requests.len()).map_err(|error| {
                RusticolError::invalid_argument(format!(
                    "on-the-fly trace-family allocation failed: {error}"
                ))
            })?;
            for (request_index, (expected_query, outcome)) in
                expected_queries.into_iter().zip(outcomes).enumerate()
            {
                match outcome {
                    OnTheFlySelectedQueryOutcomeV1::Trace(selected) => {
                        if selected.query != expected_query {
                            return Err(RusticolError::integrity(
                                "on-the-fly trace query changed during construction",
                            ));
                        }
                        amplitude_destinations[request_index] = Some(traces.len());
                        traces.push(selected);
                    }
                    OnTheFlySelectedQueryOutcomeV1::StructuralZero { query } => {
                        if query != expected_query {
                            return Err(RusticolError::integrity(
                                "on-the-fly structural-zero query changed during construction",
                            ));
                        }
                    }
                }
            }
            emit_progress(
                progress.as_deref_mut(),
                NativeOnTheFlyWarmUpEventKind::Start,
                NativeOnTheFlyWarmUpStage::FamilyFinalization,
                0,
                1,
                None,
            )?;
            let (cache_hit, census) = if traces.is_empty() {
                (false, None)
            } else {
                self.executor.resolver_mut().bind_on_the_fly_family(
                    &self.templates,
                    &self.direct_catalog,
                    &self.seed,
                    &mut traces,
                )?;
                let inputs = traces
                    .iter()
                    .map(|selected| QueryFamilyTraceInput {
                        trace: &selected.trace,
                        projection: selected.projection,
                    })
                    .collect::<Vec<_>>();
                let cache_hit =
                    self.executor
                        .prepare(&self.direct_catalog, &inputs, logical_point_capacity)?;
                let census = self.executor.prepared_census().ok_or_else(|| {
                    RusticolError::integrity("on-the-fly prepared family has no work census")
                })?;
                if usize::try_from(census.union_amplitude_destination_count).ok()
                    != Some(traces.len())
                {
                    return Err(RusticolError::integrity(
                        "on-the-fly family does not retain one destination per nonzero query",
                    ));
                }
                if cache_hit && self.executor.active_retained_handle().is_none() {
                    return Err(RusticolError::integrity(
                        "on-the-fly cache hit has no retained executor handle",
                    ));
                }
                (cache_hit, Some(census))
            };
            emit_progress(
                progress.as_deref_mut(),
                NativeOnTheFlyWarmUpEventKind::End,
                NativeOnTheFlyWarmUpStage::FamilyFinalization,
                1,
                1,
                None,
            )?;
            Ok((
                cache_hit,
                PreparedOnTheFlyLcFamilyV1 {
                    requests: requests.to_vec().into_boxed_slice(),
                    amplitude_destinations: amplitude_destinations.into_boxed_slice(),
                    executor_handle: None,
                    census,
                    logical_point_capacity,
                },
            ))
        })();
        let (cache_hit, candidate) = match candidate_result {
            Ok(candidate) => candidate,
            Err(error) => {
                self.discard_pending_lc_queries()?;
                return Err(error);
            }
        };
        // Selection-level state is committed only after the first successful
        // evaluation, including structural-zero and executor-cache-hit cases.
        self.pending_family = Some(candidate);
        Ok(cache_hit)
    }

    /// Prepare the transient H x S structural query family consumed by a
    /// contracted NLC/full color metric.  Only its compact identity and
    /// amplitude projection survive this cold boundary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_contracted_queries(
        &mut self,
        selectors: &OnTheFlyCompactSelectorAdapterV1,
        helicity_ordinals: &[usize],
        structural_color_count: usize,
        destination_by_owner_ordinal: &[u32],
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        self.prepare_contracted_queries_impl(
            selectors,
            helicity_ordinals,
            structural_color_count,
            destination_by_owner_ordinal,
            logical_point_capacity,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_contracted_queries_for_warm_up(
        &mut self,
        selectors: &OnTheFlyCompactSelectorAdapterV1,
        helicity_ordinals: &[usize],
        structural_color_count: usize,
        destination_by_owner_ordinal: &[u32],
        logical_point_capacity: u32,
        query_construction_threads: usize,
        progress: &mut OnTheFlyWarmUpProgress<'_>,
    ) -> RusticolResult<bool> {
        self.prepare_contracted_queries_impl(
            selectors,
            helicity_ordinals,
            structural_color_count,
            destination_by_owner_ordinal,
            logical_point_capacity,
            Some(query_construction_threads),
            Some(progress),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_contracted_queries_impl(
        &mut self,
        selectors: &OnTheFlyCompactSelectorAdapterV1,
        helicity_ordinals: &[usize],
        structural_color_count: usize,
        destination_by_owner_ordinal: &[u32],
        logical_point_capacity: u32,
        query_construction_threads: Option<usize>,
        mut progress: Option<&mut OnTheFlyWarmUpProgress<'_>>,
    ) -> RusticolResult<bool> {
        if helicity_ordinals.is_empty()
            || structural_color_count == 0
            || logical_point_capacity == 0
        {
            return Err(RusticolError::invalid_argument(
                "on-the-fly contracted preparation requires helicities, structural colors, and a nonzero point capacity",
            ));
        }
        if !self.families.is_empty() || self.pending_family.is_some() {
            return Err(RusticolError::internal(
                "on-the-fly lane cannot mix LC and contracted retained families",
            ));
        }
        if helicity_ordinals.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RusticolError::invalid_argument(
                "on-the-fly contracted helicity ordinals are not strictly increasing",
            ));
        }
        let query_count = helicity_ordinals
            .len()
            .checked_mul(structural_color_count)
            .ok_or_else(|| {
                RusticolError::invalid_argument("on-the-fly contracted query shape exceeds usize")
            })?;
        if destination_by_owner_ordinal.len() != structural_color_count
            || selectors.color_count() != structural_color_count
        {
            return Err(RusticolError::integrity(
                "on-the-fly contracted selector or destination shape is inconsistent",
            ));
        }
        let mut destination_seen = vec![false; structural_color_count];
        for destination in destination_by_owner_ordinal {
            let destination = usize::try_from(*destination).map_err(|_| {
                RusticolError::artifact("on-the-fly contracted destination exceeds usize")
            })?;
            let seen = destination_seen.get_mut(destination).ok_or_else(|| {
                RusticolError::integrity(
                    "on-the-fly contracted destination is outside the structural owner domain",
                )
            })?;
            if *seen {
                return Err(RusticolError::integrity(
                    "on-the-fly contracted destination mapping is not one-to-one",
                ));
            }
            *seen = true;
        }
        if destination_seen.contains(&false) {
            return Err(RusticolError::integrity(
                "on-the-fly contracted destination mapping is not complete",
            ));
        }
        let identity_matches = |family: &PreparedOnTheFlyContractedFamilyV1| {
            family.helicity_ordinals.as_ref() == helicity_ordinals
                && family.structural_color_count == structural_color_count
        };
        if self
            .pending_contracted_family
            .as_ref()
            .is_some_and(identity_matches)
        {
            let (current_capacity, has_executable_family) = self
                .pending_contracted_family
                .as_ref()
                .map(|family| (family.logical_point_capacity, family.census.is_some()))
                .expect("matching contracted pending family disappeared");
            if logical_point_capacity > current_capacity
                && has_executable_family
                && let Err(error) = self.executor.resize_active_family(logical_point_capacity)
            {
                self.discard_pending_contracted_queries()?;
                return Err(error);
            }
            let family = self
                .pending_contracted_family
                .as_mut()
                .expect("matching contracted pending family disappeared after resize");
            family.logical_point_capacity =
                family.logical_point_capacity.max(logical_point_capacity);
            emit_reused_family_progress(progress.as_deref_mut(), query_count)?;
            return Ok(false);
        }
        self.discard_pending_contracted_queries()?;
        if self
            .contracted_family
            .as_ref()
            .is_some_and(identity_matches)
        {
            let family = self
                .contracted_family
                .as_mut()
                .expect("matching retained contracted family disappeared");
            let cache_hit = if let Some(handle) = family.executor_handle {
                self.executor
                    .activate_retained_family(handle, logical_point_capacity)?
            } else {
                true
            };
            family.logical_point_capacity =
                family.logical_point_capacity.max(logical_point_capacity);
            emit_reused_family_progress(progress.as_deref_mut(), query_count)?;
            return Ok(cache_hit);
        }

        let candidate_result = (|| {
            emit_progress(
                progress.as_deref_mut(),
                NativeOnTheFlyWarmUpEventKind::Start,
                NativeOnTheFlyWarmUpStage::QueryFamily,
                0,
                query_count,
                None,
            )?;
            self.ensure_process_prepared()?;
            let enable_cyclic_trace_reflection = self.symmetric_group_color_workspace.is_some();
            let coupling_policy = self.coupling_policy.as_ref().ok_or_else(|| {
                RusticolError::internal("on-the-fly coupling policy disappeared after preparation")
            })?;
            let grammar = self.prepared_grammar.as_ref().ok_or_else(|| {
                RusticolError::internal("on-the-fly grammar disappeared after preparation")
            })?;
            let query_pool = OnTheFlyQueryConstructionPoolV1::new(
                query_construction_threads.unwrap_or(self.effective_query_construction_threads),
            )?;
            let mut amplitude_destinations = Vec::new();
            amplitude_destinations
                .try_reserve_exact(query_count)
                .map_err(|error| {
                    RusticolError::invalid_argument(format!(
                        "on-the-fly contracted projection allocation failed: {error}"
                    ))
                })?;
            amplitude_destinations.resize(query_count, None);

            let templates = &self.templates;
            let direct_catalog = &self.direct_catalog;
            let seed = &self.seed;
            let executor = &mut self.executor;
            let mut binding_started = false;
            let mut processed_query_count = 0_usize;
            #[cfg(test)]
            let mut max_live_query_outcomes = 0_usize;
            let streamed = build_streamed_query_family_candidate_v1(direct_catalog, |consumer| {
                while processed_query_count < query_count {
                    let chunk_end = processed_query_count
                        .saturating_add(query_pool.chunk_capacity())
                        .min(query_count);
                    let mut queries = Vec::new();
                    queries
                        .try_reserve_exact(chunk_end - processed_query_count)
                        .map_err(|error| {
                            RusticolError::invalid_argument(format!(
                                "on-the-fly contracted query-chunk allocation failed: {error}"
                            ))
                        })?;
                    for request_index in processed_query_count..chunk_end {
                        let helicity_position = request_index / structural_color_count;
                        let owner_ordinal = request_index % structural_color_count;
                        let query = selectors.decoded_query_at(
                            seed,
                            helicity_ordinals[helicity_position],
                            owner_ordinal,
                        )?;
                        queries.push(grammar.canonicalize_query(seed, query)?);
                    }
                    let expected_queries = queries.clone();
                    let outcomes = query_pool.build_chunk(
                        templates,
                        direct_catalog,
                        seed,
                        coupling_policy,
                        grammar,
                        enable_cyclic_trace_reflection,
                        queries,
                    )?;
                    if outcomes.len() != expected_queries.len() {
                        return Err(RusticolError::integrity(
                            "on-the-fly contracted trace builder changed a chunk count",
                        ));
                    }
                    #[cfg(test)]
                    {
                        max_live_query_outcomes = max_live_query_outcomes.max(outcomes.len());
                    }

                    let mut traces = Vec::new();
                    let mut projection_slots = Vec::new();
                    traces.try_reserve_exact(outcomes.len()).map_err(|error| {
                        RusticolError::invalid_argument(format!(
                            "on-the-fly contracted trace-chunk allocation failed: {error}"
                        ))
                    })?;
                    projection_slots
                        .try_reserve_exact(outcomes.len())
                        .map_err(|error| {
                            RusticolError::invalid_argument(format!(
                                "on-the-fly contracted projection-chunk allocation failed: {error}"
                            ))
                        })?;
                    for (chunk_offset, (expected_query, outcome)) in
                        expected_queries.into_iter().zip(outcomes).enumerate()
                    {
                        let request_index = processed_query_count + chunk_offset;
                        let helicity_position = request_index / structural_color_count;
                        let owner_ordinal = request_index % structural_color_count;
                        let projection_index = contracted_projection_index(
                            helicity_position,
                            owner_ordinal,
                            structural_color_count,
                            destination_by_owner_ordinal,
                        )?;
                        match outcome {
                            OnTheFlySelectedQueryOutcomeV1::Trace(selected) => {
                                if selected.query != expected_query {
                                    return Err(RusticolError::integrity(
                                        "on-the-fly contracted trace query changed during construction",
                                    ));
                                }
                                projection_slots.push(projection_index);
                                traces.push(selected);
                            }
                            OnTheFlySelectedQueryOutcomeV1::StructuralZero { query: selected } => {
                                if selected != expected_query {
                                    return Err(RusticolError::integrity(
                                        "on-the-fly contracted structural-zero query changed during construction",
                                    ));
                                }
                            }
                        }
                    }
                    if !traces.is_empty() {
                        if !binding_started {
                            executor
                                .resolver_mut()
                                .begin_on_the_fly_family_binding(templates, direct_catalog)?;
                            binding_started = true;
                        }
                        executor.resolver_mut().extend_on_the_fly_family_binding(
                            templates,
                            direct_catalog,
                            seed,
                            traces.iter_mut().map(|selected| &mut selected.trace),
                        )?;
                        let inputs = traces
                            .iter()
                            .map(|selected| QueryFamilyTraceInput {
                                trace: &selected.trace,
                                projection: selected.projection,
                            })
                            .collect::<Vec<_>>();
                        let destinations = consumer.push_chunk(&inputs)?;
                        if destinations.len() != projection_slots.len() {
                            return Err(RusticolError::integrity(
                                "on-the-fly streamed destination range changed a chunk count",
                            ));
                        }
                        for (projection_index, destination) in
                            projection_slots.into_iter().zip(destinations)
                        {
                            let slot = amplitude_destinations
                                .get_mut(projection_index)
                                .ok_or_else(|| {
                                    RusticolError::integrity(
                                        "on-the-fly contracted projection slot is absent",
                                    )
                                })?;
                            if slot.replace(destination as usize).is_some() {
                                return Err(RusticolError::integrity(
                                    "on-the-fly contracted projection slot is repeated",
                                ));
                            }
                        }
                    }
                    processed_query_count = chunk_end;
                    emit_progress(
                        progress.as_deref_mut(),
                        NativeOnTheFlyWarmUpEventKind::Update,
                        NativeOnTheFlyWarmUpStage::QueryFamily,
                        processed_query_count,
                        query_count,
                        None,
                    )?;
                }
                Ok(())
            })?;
            #[cfg(test)]
            {
                self.contracted_max_live_query_outcomes = max_live_query_outcomes;
            }
            if processed_query_count != query_count {
                return Err(RusticolError::integrity(
                    "on-the-fly contracted streamed construction omitted a structural query",
                ));
            }
            emit_progress(
                progress.as_deref_mut(),
                NativeOnTheFlyWarmUpEventKind::End,
                NativeOnTheFlyWarmUpStage::QueryFamily,
                query_count,
                query_count,
                None,
            )?;
            emit_progress(
                progress.as_deref_mut(),
                NativeOnTheFlyWarmUpEventKind::Start,
                NativeOnTheFlyWarmUpStage::FamilyFinalization,
                0,
                1,
                None,
            )?;
            let (cache_hit, census) = if let Some(streamed) = streamed {
                if !binding_started {
                    return Err(RusticolError::integrity(
                        "on-the-fly streamed family has no semantic binding transaction",
                    ));
                }
                let cache_hit =
                    executor.prepare_streamed_candidate(streamed, logical_point_capacity)?;
                let census = executor.prepared_census().ok_or_else(|| {
                    RusticolError::integrity("on-the-fly contracted family has no work census")
                })?;
                let projected_nonzero_count = amplitude_destinations
                    .iter()
                    .filter(|destination| destination.is_some())
                    .count();
                if usize::try_from(census.union_amplitude_destination_count).ok()
                    != Some(projected_nonzero_count)
                {
                    return Err(RusticolError::integrity(
                        "on-the-fly contracted family does not retain one destination per nonzero query",
                    ));
                }
                (cache_hit, Some(census))
            } else {
                if binding_started {
                    return Err(RusticolError::integrity(
                        "on-the-fly all-zero streamed family retained semantic bindings",
                    ));
                }
                (false, None)
            };
            emit_progress(
                progress.as_deref_mut(),
                NativeOnTheFlyWarmUpEventKind::End,
                NativeOnTheFlyWarmUpStage::FamilyFinalization,
                1,
                1,
                None,
            )?;
            Ok((
                cache_hit,
                PreparedOnTheFlyContractedFamilyV1 {
                    helicity_ordinals: helicity_ordinals.to_vec().into_boxed_slice(),
                    structural_color_count,
                    amplitude_destinations: amplitude_destinations.into_boxed_slice(),
                    executor_handle: None,
                    census,
                    logical_point_capacity,
                },
            ))
        })();
        let (cache_hit, candidate) = match candidate_result {
            Ok(candidate) => candidate,
            Err(error) => {
                self.discard_pending_contracted_queries()?;
                return Err(error);
            }
        };
        self.pending_contracted_family = Some(candidate);
        Ok(cache_hit)
    }

    /// Re-activate the family already selected by the runtime wrapper without
    /// rebuilding its public selector requests.  The common warmed call has
    /// sufficient point capacity and therefore performs no allocation or
    /// identity walk.  Capacity growth remains explicit and fallible.
    pub(super) fn reuse_current_lc_queries(
        &mut self,
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        if logical_point_capacity == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC preparation requires a nonzero point capacity",
            ));
        }
        if let Some(family) = self.pending_family.as_mut() {
            if logical_point_capacity <= family.logical_point_capacity {
                return Ok(false);
            }
            let has_executable_family = family.census.is_some();
            if has_executable_family
                && let Err(error) = self.executor.resize_active_family(logical_point_capacity)
            {
                self.discard_pending_lc_queries()?;
                return Err(error);
            }
            self.pending_family
                .as_mut()
                .expect("pending on-the-fly family disappeared after resize")
                .logical_point_capacity = logical_point_capacity;
            return Ok(false);
        }
        let index = self.last_family.ok_or_else(|| {
            RusticolError::internal(
                "on-the-fly selector cache has no corresponding prepared family",
            )
        })?;
        let family = self.families.get_mut(index).ok_or_else(|| {
            RusticolError::internal(
                "on-the-fly selector cache references an absent prepared family",
            )
        })?;
        if logical_point_capacity <= family.logical_point_capacity {
            return Ok(true);
        }
        let cache_hit = if let Some(handle) = family.executor_handle {
            self.executor
                .activate_retained_family(handle, logical_point_capacity)?
        } else {
            true
        };
        family.logical_point_capacity = logical_point_capacity;
        Ok(cache_hit)
    }

    pub(super) fn reuse_current_contracted_queries(
        &mut self,
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        if logical_point_capacity == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly contracted preparation requires a nonzero point capacity",
            ));
        }
        if let Some(family) = self.pending_contracted_family.as_mut() {
            if logical_point_capacity <= family.logical_point_capacity {
                return Ok(false);
            }
            if family.census.is_some()
                && let Err(error) = self.executor.resize_active_family(logical_point_capacity)
            {
                self.discard_pending_contracted_queries()?;
                return Err(error);
            }
            self.pending_contracted_family
                .as_mut()
                .expect("pending contracted family disappeared after resize")
                .logical_point_capacity = logical_point_capacity;
            return Ok(false);
        }
        let family = self.contracted_family.as_mut().ok_or_else(|| {
            RusticolError::internal("on-the-fly contracted selector cache has no prepared family")
        })?;
        if logical_point_capacity <= family.logical_point_capacity {
            return Ok(true);
        }
        let cache_hit = if let Some(handle) = family.executor_handle {
            self.executor
                .activate_retained_family(handle, logical_point_capacity)?
        } else {
            true
        };
        family.logical_point_capacity = logical_point_capacity;
        Ok(cache_hit)
    }

    pub(super) fn prepared_census(&self) -> Option<OnTheFlyQueryFamilyCensusV1> {
        self.current_family()
            .and_then(|family| family.census)
            .or_else(|| {
                self.current_contracted_family()
                    .and_then(|family| family.census)
            })
    }

    #[cfg(test)]
    pub(super) fn retained_family_count(&self) -> usize {
        self.families.len() + usize::from(self.contracted_family.is_some())
    }

    pub(super) fn pending_family_count(&self) -> usize {
        usize::from(self.pending_family.is_some())
            + usize::from(self.pending_contracted_family.is_some())
    }

    pub(super) fn semantic_executor_binding_count(&self) -> RusticolResult<u32> {
        self.executor.resolver().semantic_executor_binding_count()
    }

    /// Count retained production state without constructing or mutating it.
    pub(super) fn retained_state_census(&self) -> OnTheFlyRetainedStateCensusV1 {
        let mut census = OnTheFlyRetainedStateCensusV1::default();
        for family in &self.families {
            census.family_count += 1;
            census.request_count += family.requests.len();
            census.executor_handle_count += usize::from(family.executor_handle.is_some());
            for destination in &family.amplitude_destinations {
                if destination.is_some() {
                    census.amplitude_destination_count += 1;
                }
            }
        }
        if let Some(family) = &self.contracted_family {
            census.family_count += 1;
            census.request_count += family.amplitude_destinations.len();
            census.executor_handle_count += usize::from(family.executor_handle.is_some());
            census.amplitude_destination_count += family
                .amplitude_destinations
                .iter()
                .filter(|destination| destination.is_some())
                .count();
        }
        // Query-local traces and embedded lookup keys are deliberately absent
        // from the compact retained family type. Their zero counters keep that
        // invariant observable without exposing private family fields.
        census
    }

    pub(super) fn clear(&mut self) -> RusticolResult<()> {
        self.executor.clear_families()?;
        self.executor.resolver_mut().clear_resolved_bindings();
        self.pending_family = None;
        self.last_family = None;
        self.families.clear();
        self.pending_contracted_family = None;
        self.contracted_family = None;
        self.prepared_grammar = None;
        self.coupling_policy = None;
        self.coupling_policy_resolution = None;
        self.process_preparation_count = 0;
        #[cfg(test)]
        {
            self.process_preparation_census = OnTheFlyProcessPreparationCensusV1::default();
            self.contracted_max_live_query_outcomes = 0;
        }
        self.source_momenta_scratch = Vec::new();
        self.amplitude_scratch = Vec::new();
        Ok(())
    }

    fn current_family(&self) -> Option<&PreparedOnTheFlyLcFamilyV1> {
        self.pending_family
            .as_ref()
            .or_else(|| self.last_family.and_then(|index| self.families.get(index)))
    }

    fn current_contracted_family(&self) -> Option<&PreparedOnTheFlyContractedFamilyV1> {
        self.pending_contracted_family
            .as_ref()
            .or(self.contracted_family.as_ref())
    }

    pub(super) fn discard_pending_lc_queries(&mut self) -> RusticolResult<()> {
        self.executor.discard_pending_family()?;
        self.executor
            .resolver_mut()
            .discard_pending_resolved_bindings();
        self.pending_family = None;
        Ok(())
    }

    pub(super) fn discard_pending_contracted_queries(&mut self) -> RusticolResult<()> {
        self.executor.discard_pending_family()?;
        self.executor
            .resolver_mut()
            .discard_pending_resolved_bindings();
        self.pending_contracted_family = None;
        Ok(())
    }

    /// Roll back a contracted selection whose executor family may already
    /// have committed on an earlier tile.  The wrapper deliberately clears
    /// its public selection cache in the same error path, so both halves of
    /// the last-family-only identity remain coherent.
    pub(super) fn abort_contracted_selection(&mut self) -> RusticolResult<()> {
        self.executor.clear_families()?;
        self.executor.resolver_mut().clear_resolved_bindings();
        self.pending_contracted_family = None;
        self.contracted_family = None;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn fail_contracted_execution_at_for_test(&mut self, attempt: Option<usize>) {
        self.contracted_test_execution_attempt = 0;
        self.contracted_test_fail_execution_at = attempt;
    }

    #[cfg(test)]
    pub(super) const fn contracted_max_live_query_outcomes_for_test(&self) -> usize {
        self.contracted_max_live_query_outcomes
    }

    fn promote_pending_family(&mut self) -> RusticolResult<()> {
        let Some(family) = self.pending_family.as_ref() else {
            return Ok(());
        };
        let executor_handle = if family.census.is_some() {
            let handle = self.executor.active_retained_handle().ok_or_else(|| {
                RusticolError::internal(
                    "successful on-the-fly family has no retained executor handle",
                )
            })?;
            self.executor
                .resolver_mut()
                .commit_pending_resolved_bindings()?;
            Some(handle)
        } else {
            // A structural-zero family replaces executable state only after
            // its evaluation succeeds. Row-table owners disappear before the
            // semantic bindings to their prepared contexts.
            self.executor.clear_families()?;
            self.executor.resolver_mut().clear_resolved_bindings();
            None
        };
        let mut family = self
            .pending_family
            .take()
            .expect("validated on-the-fly pending family disappeared during promotion");
        family.executor_handle = executor_handle;
        self.retain_family(family);
        Ok(())
    }

    fn retain_family(&mut self, family: PreparedOnTheFlyLcFamilyV1) {
        self.families.clear();
        self.families.push(family);
        self.last_family = Some(0);
    }

    fn promote_pending_contracted_family(&mut self) -> RusticolResult<()> {
        let Some(family) = self.pending_contracted_family.as_ref() else {
            return Ok(());
        };
        let executor_handle = if family.census.is_some() {
            let handle = self.executor.active_retained_handle().ok_or_else(|| {
                RusticolError::internal(
                    "successful on-the-fly contracted family has no retained executor handle",
                )
            })?;
            self.executor
                .resolver_mut()
                .commit_pending_resolved_bindings()?;
            Some(handle)
        } else {
            self.executor.clear_families()?;
            self.executor.resolver_mut().clear_resolved_bindings();
            None
        };
        let mut family = self
            .pending_contracted_family
            .take()
            .expect("validated on-the-fly contracted pending family disappeared during promotion");
        family.executor_handle = executor_handle;
        self.contracted_family = Some(family);
        Ok(())
    }

    fn prepare_runtime_parameters(&mut self, runtime_values: &[f64]) -> RusticolResult<()> {
        if runtime_values.iter().any(|value| !value.is_finite()) {
            return Err(RusticolError::invalid_argument(
                "on-the-fly runtime parameter value is not finite",
            ));
        }
        if self.runtime_parameter_values.as_ref() == runtime_values {
            return Ok(());
        }
        let prepared = projected_prepared_parameter_values(
            &self.parameter_defaults,
            &self.parameter_projection,
            runtime_values,
        )?;
        self.executor.set_parameters(&prepared)?;
        self.runtime_parameter_values = runtime_values.to_vec().into_boxed_slice();
        Ok(())
    }

    fn ensure_point_capacity(&mut self, point_count: u32) -> RusticolResult<()> {
        let Some(prepared) = self.current_family() else {
            return Err(RusticolError::invalid_argument(
                "on-the-fly evaluation requires prepared LC queries",
            ));
        };
        if point_count <= prepared.logical_point_capacity {
            return Ok(());
        }
        self.reuse_current_lc_queries(point_count)?;
        Ok(())
    }

    fn ensure_amplitude_scratch(&mut self, point_count: u32) -> RusticolResult<usize> {
        let query_count = self
            .current_family()
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly evaluation requires prepared LC queries",
                )
            })?
            .amplitude_destinations
            .iter()
            .filter(|destination| destination.is_some())
            .count();
        let required = query_count
            .checked_mul(point_count as usize)
            .ok_or_else(|| {
                RusticolError::invalid_argument("on-the-fly amplitude shape exceeds usize")
            })?;
        if self.amplitude_scratch.len() < required {
            self.amplitude_scratch
                .try_reserve_exact(required - self.amplitude_scratch.len())
                .map_err(|error| {
                    RusticolError::invalid_argument(format!(
                        "on-the-fly amplitude workspace allocation failed: {error}"
                    ))
                })?;
            self.amplitude_scratch.resize(required, (0.0, 0.0));
        }
        Ok(required)
    }

    fn execute_amplitudes(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<(
        OnTheFlyQueryFamilyExecutionReportV1,
        Duration,
        Duration,
        Duration,
    )> {
        let candidate_pending = self.pending_family.is_some();
        let result =
            self.execute_amplitudes_inner(point_major_momenta, point_count, runtime_parameters);
        if result.is_err() && candidate_pending {
            self.discard_pending_lc_queries()?;
        }
        result
    }

    fn execute_amplitudes_inner(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<(
        OnTheFlyQueryFamilyExecutionReportV1,
        Duration,
        Duration,
        Duration,
    )> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly evaluation requires at least one point",
            ));
        }
        self.ensure_point_capacity(point_count)?;

        let parameter_started = Instant::now();
        self.prepare_runtime_parameters(runtime_parameters)?;
        let parameter_setup = parameter_started.elapsed();

        if self
            .current_family()
            .is_some_and(|family| family.census.is_none())
        {
            let cache_hit = self.pending_family.is_none();
            self.promote_pending_family()?;
            return Ok((
                OnTheFlyQueryFamilyExecutionReportV1 {
                    cache_hit,
                    ..OnTheFlyQueryFamilyExecutionReportV1::default()
                },
                parameter_setup,
                Duration::ZERO,
                Duration::ZERO,
            ));
        }

        let input_started = Instant::now();
        let source_major_len = on_the_fly_source_major_momenta_into(
            &self.seed,
            point_major_momenta,
            point_count,
            4,
            &mut self.source_momenta_scratch,
        )?;
        let input_setup = input_started.elapsed();

        let output_len = self.ensure_amplitude_scratch(point_count)?;
        let execution_started = Instant::now();
        let report = self.executor.execute_into(
            &self.source_momenta_scratch[..source_major_len],
            point_count,
            &mut self.amplitude_scratch[..output_len],
        )?;
        let execution = execution_started.elapsed();
        // A cold family becomes reusable only after its first successful
        // prepared-kernel execution.
        self.promote_pending_family()?;
        Ok((report, parameter_setup, input_setup, execution))
    }

    fn execute_amplitudes_unprofiled(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<()> {
        let candidate_pending = self.pending_family.is_some();
        let result = self.execute_amplitudes_unprofiled_inner(
            point_major_momenta,
            point_count,
            runtime_parameters,
        );
        if result.is_err() && candidate_pending {
            self.discard_pending_lc_queries()?;
        }
        result
    }

    fn execute_amplitudes_unprofiled_inner(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<()> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly evaluation requires at least one point",
            ));
        }
        self.ensure_point_capacity(point_count)?;
        self.prepare_runtime_parameters(runtime_parameters)?;

        if self
            .current_family()
            .is_some_and(|family| family.census.is_none())
        {
            self.promote_pending_family()?;
            return Ok(());
        }

        let source_major_len = on_the_fly_source_major_momenta_into(
            &self.seed,
            point_major_momenta,
            point_count,
            4,
            &mut self.source_momenta_scratch,
        )?;
        let output_len = self.ensure_amplitude_scratch(point_count)?;
        self.executor.execute_into_unprofiled(
            &self.source_momenta_scratch[..source_major_len],
            point_count,
            &mut self.amplitude_scratch[..output_len],
        )?;
        // A cold family becomes reusable only after its first successful
        // prepared-kernel execution.
        self.promote_pending_family()?;
        Ok(())
    }

    fn ensure_contracted_point_capacity(&mut self, point_count: u32) -> RusticolResult<()> {
        let Some(prepared) = self.current_contracted_family() else {
            return Err(RusticolError::invalid_argument(
                "on-the-fly contracted evaluation requires prepared queries",
            ));
        };
        if point_count > prepared.logical_point_capacity {
            self.reuse_current_contracted_queries(point_count)?;
        }
        Ok(())
    }

    fn ensure_contracted_amplitude_scratch(&mut self, point_count: u32) -> RusticolResult<usize> {
        let query_count = self
            .current_contracted_family()
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly contracted evaluation requires prepared queries",
                )
            })?
            .amplitude_destinations
            .iter()
            .filter(|destination| destination.is_some())
            .count();
        let required = query_count
            .checked_mul(point_count as usize)
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly contracted amplitude shape exceeds usize",
                )
            })?;
        if self.amplitude_scratch.len() < required {
            self.amplitude_scratch
                .try_reserve_exact(required - self.amplitude_scratch.len())
                .map_err(|error| {
                    RusticolError::invalid_argument(format!(
                        "on-the-fly contracted amplitude workspace allocation failed: {error}"
                    ))
                })?;
            self.amplitude_scratch.resize(required, (0.0, 0.0));
        }
        Ok(required)
    }

    fn execute_contracted_amplitudes(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<(
        OnTheFlyQueryFamilyExecutionReportV1,
        Duration,
        Duration,
        Duration,
    )> {
        #[cfg(test)]
        {
            let attempt = self.contracted_test_execution_attempt;
            self.contracted_test_execution_attempt = self
                .contracted_test_execution_attempt
                .checked_add(1)
                .expect("contracted test execution-attempt count exceeds usize");
            if self.contracted_test_fail_execution_at == Some(attempt) {
                return Err(RusticolError::internal(
                    "injected on-the-fly contracted tile failure",
                ));
            }
        }
        let candidate_pending = self.pending_contracted_family.is_some();
        let result = self.execute_contracted_amplitudes_inner(
            point_major_momenta,
            point_count,
            runtime_parameters,
        );
        if result.is_err() && candidate_pending {
            self.discard_pending_contracted_queries()?;
        }
        result
    }

    fn execute_contracted_amplitudes_inner(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<(
        OnTheFlyQueryFamilyExecutionReportV1,
        Duration,
        Duration,
        Duration,
    )> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly contracted evaluation requires at least one point",
            ));
        }
        self.ensure_contracted_point_capacity(point_count)?;

        let parameter_started = Instant::now();
        self.prepare_runtime_parameters(runtime_parameters)?;
        let parameter_setup = parameter_started.elapsed();
        if self
            .current_contracted_family()
            .is_some_and(|family| family.census.is_none())
        {
            let cache_hit = self.pending_contracted_family.is_none();
            self.promote_pending_contracted_family()?;
            return Ok((
                OnTheFlyQueryFamilyExecutionReportV1 {
                    cache_hit,
                    ..OnTheFlyQueryFamilyExecutionReportV1::default()
                },
                parameter_setup,
                Duration::ZERO,
                Duration::ZERO,
            ));
        }

        let input_started = Instant::now();
        let source_major_len = on_the_fly_source_major_momenta_into(
            &self.seed,
            point_major_momenta,
            point_count,
            4,
            &mut self.source_momenta_scratch,
        )?;
        let input_setup = input_started.elapsed();
        let output_len = self.ensure_contracted_amplitude_scratch(point_count)?;
        let execution_started = Instant::now();
        let report = self.executor.execute_into(
            &self.source_momenta_scratch[..source_major_len],
            point_count,
            &mut self.amplitude_scratch[..output_len],
        )?;
        let execution = execution_started.elapsed();
        self.promote_pending_contracted_family()?;
        Ok((report, parameter_setup, input_setup, execution))
    }

    fn execute_contracted_amplitudes_unprofiled(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<()> {
        #[cfg(test)]
        {
            let attempt = self.contracted_test_execution_attempt;
            self.contracted_test_execution_attempt = self
                .contracted_test_execution_attempt
                .checked_add(1)
                .expect("contracted test execution-attempt count exceeds usize");
            if self.contracted_test_fail_execution_at == Some(attempt) {
                return Err(RusticolError::internal(
                    "injected on-the-fly contracted tile failure",
                ));
            }
        }
        let candidate_pending = self.pending_contracted_family.is_some();
        let result = self.execute_contracted_amplitudes_unprofiled_inner(
            point_major_momenta,
            point_count,
            runtime_parameters,
        );
        if result.is_err() && candidate_pending {
            self.discard_pending_contracted_queries()?;
        }
        result
    }

    fn execute_contracted_amplitudes_unprofiled_inner(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<()> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly contracted evaluation requires at least one point",
            ));
        }
        self.ensure_contracted_point_capacity(point_count)?;
        self.prepare_runtime_parameters(runtime_parameters)?;
        if self
            .current_contracted_family()
            .is_some_and(|family| family.census.is_none())
        {
            self.promote_pending_contracted_family()?;
            return Ok(());
        }

        let source_major_len = on_the_fly_source_major_momenta_into(
            &self.seed,
            point_major_momenta,
            point_count,
            4,
            &mut self.source_momenta_scratch,
        )?;
        let output_len = self.ensure_contracted_amplitude_scratch(point_count)?;
        self.executor.execute_into_unprofiled(
            &self.source_momenta_scratch[..source_major_len],
            point_count,
            &mut self.amplitude_scratch[..output_len],
        )?;
        self.promote_pending_contracted_family()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_current_contracted_color(
        &mut self,
        contraction: &RecurrenceColorContraction,
        point_count: usize,
        helicity_count: usize,
        normalization_factor: f64,
        accumulate: impl FnMut(usize, usize, f64),
    ) -> RusticolResult<()> {
        let Self {
            contracted_family,
            pending_contracted_family,
            amplitude_scratch,
            symmetric_group_color_workspace,
            ..
        } = self;
        let family = pending_contracted_family
            .as_ref()
            .or(contracted_family.as_ref())
            .ok_or_else(|| {
                RusticolError::internal(
                    "successful on-the-fly contracted execution lost its prepared family",
                )
            })?;
        reduce_contracted_color(
            contraction,
            symmetric_group_color_workspace.as_mut(),
            &family.amplitude_destinations,
            amplitude_scratch,
            point_count,
            helicity_count,
            family.structural_color_count,
            normalization_factor,
            accumulate,
        )
    }

    /// Evaluate a contracted NLC/full color metric without constructing a
    /// diagnostic report or reading phase clocks.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_contracted_f64_into_unprofiled(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
        normalization_factor: f64,
        contraction: &RecurrenceColorContraction,
        point_tile_size: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        validate_contracted_run_shape(
            point_major_momenta,
            point_count,
            normalization_factor,
            contraction,
            point_tile_size,
            output.len(),
            point_count as usize,
        )?;
        output.fill(0.0);
        let point_stride = point_major_momenta.len() / point_count as usize;
        let helicity_count = self
            .current_contracted_family()
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly contracted total requires prepared queries",
                )
            })?
            .helicity_ordinals
            .len();
        for point_start in (0..point_count as usize).step_by(point_tile_size) {
            let tile_count = point_tile_size.min(point_count as usize - point_start);
            let input_start = point_start * point_stride;
            let input_end = input_start + tile_count * point_stride;
            self.execute_contracted_amplitudes_unprofiled(
                &point_major_momenta[input_start..input_end],
                u32::try_from(tile_count).map_err(|_| {
                    RusticolError::invalid_argument("on-the-fly contracted tile count exceeds u32")
                })?,
                runtime_parameters,
            )?;
            self.reduce_current_contracted_color(
                contraction,
                tile_count,
                helicity_count,
                normalization_factor,
                |point, _helicity, value| output[point_start + point] += value,
            )?;
        }
        Ok(())
    }

    /// Evaluate a contracted NLC/full color metric into one total per point.
    /// The manifest tile size bounds both the prepared numeric family and the
    /// H x S complex-amplitude scratch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_contracted_f64_into(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
        normalization_factor: f64,
        contraction: &RecurrenceColorContraction,
        point_tile_size: usize,
        output: &mut [f64],
    ) -> RusticolResult<(OnTheFlyQueryFamilyExecutionReportV1, RuntimeProfile)> {
        validate_contracted_run_shape(
            point_major_momenta,
            point_count,
            normalization_factor,
            contraction,
            point_tile_size,
            output.len(),
            point_count as usize,
        )?;
        output.fill(0.0);
        let total_started = Instant::now();
        let mut aggregate_profile = RuntimeProfile::default();
        let mut aggregate_report = None;
        let point_stride = point_major_momenta.len() / point_count as usize;
        let helicity_count = self
            .current_contracted_family()
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly contracted total requires prepared queries",
                )
            })?
            .helicity_ordinals
            .len();
        for point_start in (0..point_count as usize).step_by(point_tile_size) {
            let tile_count = point_tile_size.min(point_count as usize - point_start);
            let input_start = point_start.checked_mul(point_stride).ok_or_else(|| {
                RusticolError::invalid_argument("on-the-fly contracted tile offset exceeds usize")
            })?;
            let input_end = input_start
                .checked_add(tile_count.checked_mul(point_stride).ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "on-the-fly contracted tile shape exceeds usize",
                    )
                })?)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "on-the-fly contracted tile range exceeds usize",
                    )
                })?;
            let tile_started = Instant::now();
            let (report, parameter_setup, input_setup, execution) = self
                .execute_contracted_amplitudes(
                    &point_major_momenta[input_start..input_end],
                    u32::try_from(tile_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "on-the-fly contracted tile count exceeds u32",
                        )
                    })?,
                    runtime_parameters,
                )?;
            let reduction_started = Instant::now();
            self.reduce_current_contracted_color(
                contraction,
                tile_count,
                helicity_count,
                normalization_factor,
                |point, _helicity, value| output[point_start + point] += value,
            )?;
            let reduction = reduction_started.elapsed();
            let tile_profile = profile_from_report(
                tile_started.elapsed(),
                parameter_setup,
                input_setup,
                execution,
                reduction,
                input_end - input_start,
                u32::try_from(tile_count).expect("validated tile count"),
                report,
            );
            aggregate_profile.add_sector(&tile_profile);
            merge_execution_report(&mut aggregate_report, report)?;
        }
        aggregate_profile.total_s = profile_duration_seconds(total_started.elapsed());
        Ok((aggregate_report.unwrap_or_default(), aggregate_profile))
    }

    /// Evaluate contracted values in `[point][selected_helicity][1]` order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_contracted_resolved_f64_into(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
        normalization_factor: f64,
        contraction: &RecurrenceColorContraction,
        point_tile_size: usize,
        helicity_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<(OnTheFlyQueryFamilyExecutionReportV1, RuntimeProfile)> {
        let expected_output = (point_count as usize)
            .checked_mul(helicity_count)
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly contracted resolved shape exceeds usize",
                )
            })?;
        validate_contracted_run_shape(
            point_major_momenta,
            point_count,
            normalization_factor,
            contraction,
            point_tile_size,
            output.len(),
            expected_output,
        )?;
        let prepared_helicity_count = self
            .current_contracted_family()
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly contracted resolved evaluation requires prepared queries",
                )
            })?
            .helicity_ordinals
            .len();
        if helicity_count != prepared_helicity_count {
            return Err(RusticolError::integrity(
                "on-the-fly contracted resolved helicity shape disagrees with its prepared family",
            ));
        }
        output.fill(0.0);
        let total_started = Instant::now();
        let mut aggregate_profile = RuntimeProfile::default();
        let mut aggregate_report = None;
        let point_stride = point_major_momenta.len() / point_count as usize;
        for point_start in (0..point_count as usize).step_by(point_tile_size) {
            let tile_count = point_tile_size.min(point_count as usize - point_start);
            let input_start = point_start * point_stride;
            let input_end = input_start + tile_count * point_stride;
            let tile_started = Instant::now();
            let (report, parameter_setup, input_setup, execution) = self
                .execute_contracted_amplitudes(
                    &point_major_momenta[input_start..input_end],
                    u32::try_from(tile_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "on-the-fly contracted tile count exceeds u32",
                        )
                    })?,
                    runtime_parameters,
                )?;
            let reduction_started = Instant::now();
            self.reduce_current_contracted_color(
                contraction,
                tile_count,
                helicity_count,
                normalization_factor,
                |point, helicity, value| {
                    let destination = (point_start + point) * helicity_count + helicity;
                    output[destination] += value;
                },
            )?;
            let reduction = reduction_started.elapsed();
            let tile_profile = profile_from_report(
                tile_started.elapsed(),
                parameter_setup,
                input_setup,
                execution,
                reduction,
                input_end - input_start,
                u32::try_from(tile_count).expect("validated tile count"),
                report,
            );
            aggregate_profile.add_sector(&tile_profile);
            merge_execution_report(&mut aggregate_report, report)?;
        }
        aggregate_profile.total_s = profile_duration_seconds(total_started.elapsed());
        Ok((aggregate_report.unwrap_or_default(), aggregate_profile))
    }

    /// Evaluate total LC matrix elements without constructing a diagnostic
    /// report or reading phase clocks.
    pub(super) fn run_f64_into_unprofiled(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
        normalization_factor: f64,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        if output.len() != point_count as usize
            || !normalization_factor.is_finite()
            || normalization_factor < 0.0
        {
            return Err(RusticolError::invalid_argument(
                "on-the-fly total output shape or normalization is invalid",
            ));
        }
        self.execute_amplitudes_unprofiled(point_major_momenta, point_count, runtime_parameters)?;
        output.fill(0.0);
        let family = self
            .current_family()
            .expect("successful on-the-fly execution lost its prepared family");
        reduce_total_into(
            &family.requests,
            &family.amplitude_destinations,
            &self.amplitude_scratch,
            point_count as usize,
            normalization_factor,
            output,
        )?;
        Ok(())
    }

    /// Evaluate total LC matrix elements into the same one-value-per-point
    /// shape consumed by the existing `Runtime.evaluate_into` path.
    pub(super) fn run_f64_into(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
        normalization_factor: f64,
        output: &mut [f64],
    ) -> RusticolResult<(OnTheFlyQueryFamilyExecutionReportV1, RuntimeProfile)> {
        if output.len() != point_count as usize
            || !normalization_factor.is_finite()
            || normalization_factor < 0.0
        {
            return Err(RusticolError::invalid_argument(
                "on-the-fly total output shape or normalization is invalid",
            ));
        }
        let total_started = Instant::now();
        let (report, parameter_setup, input_setup, execution) =
            self.execute_amplitudes(point_major_momenta, point_count, runtime_parameters)?;
        output.fill(0.0);
        let reduction_started = Instant::now();
        let family = self
            .current_family()
            .expect("successful on-the-fly execution lost its prepared family");
        reduce_total_into(
            &family.requests,
            &family.amplitude_destinations,
            &self.amplitude_scratch,
            point_count as usize,
            normalization_factor,
            output,
        )?;
        let reduction = reduction_started.elapsed();
        let profile = profile_from_report(
            total_started.elapsed(),
            parameter_setup,
            input_setup,
            execution,
            reduction,
            point_major_momenta.len(),
            point_count,
            report,
        );
        Ok((report, profile))
    }

    /// Evaluate resolved LC values in the established
    /// `[point][helicity][color]` order without creating a dense flow table.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_resolved_f64_into(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
        normalization_factor: f64,
        helicity_count: usize,
        color_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<(OnTheFlyQueryFamilyExecutionReportV1, RuntimeProfile)> {
        if !normalization_factor.is_finite() || normalization_factor < 0.0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly resolved normalization is invalid",
            ));
        }
        let layout =
            LcResolvedOutputLayout::new(point_count as usize, helicity_count, color_count)?;
        if output.len() != layout.output_len()? {
            return Err(RusticolError::invalid_argument(
                "on-the-fly resolved output has the wrong size",
            ));
        }
        let requests = self.current_family().ok_or_else(|| {
            RusticolError::invalid_argument(
                "on-the-fly resolved evaluation requires prepared LC queries",
            )
        })?;
        for target in requests
            .requests
            .iter()
            .flat_map(|request| request.reduction_targets.iter())
        {
            if target.helicity_position >= helicity_count || target.color_position >= color_count {
                return Err(RusticolError::integrity(
                    "on-the-fly resolved target is outside the selected public axes",
                ));
            }
        }

        let total_started = Instant::now();
        let (report, parameter_setup, input_setup, execution) =
            self.execute_amplitudes(point_major_momenta, point_count, runtime_parameters)?;
        output.fill(0.0);
        let reduction_started = Instant::now();
        let family = self
            .current_family()
            .expect("successful on-the-fly execution lost its prepared family");
        reduce_resolved_into(
            &family.requests,
            &family.amplitude_destinations,
            &self.amplitude_scratch,
            point_count as usize,
            normalization_factor,
            layout,
            output,
        )?;
        let reduction = reduction_started.elapsed();
        let profile = profile_from_report(
            total_started.elapsed(),
            parameter_setup,
            input_setup,
            execution,
            reduction,
            point_major_momenta.len(),
            point_count,
            report,
        );
        Ok((report, profile))
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_contracted_run_shape(
    point_major_momenta: &[f64],
    point_count: u32,
    normalization_factor: f64,
    contraction: &RecurrenceColorContraction,
    point_tile_size: usize,
    actual_output_len: usize,
    expected_output_len: usize,
) -> RusticolResult<()> {
    if point_count == 0
        || point_tile_size == 0
        || !point_major_momenta
            .len()
            .is_multiple_of(point_count as usize)
        || !normalization_factor.is_finite()
        || normalization_factor < 0.0
        || actual_output_len != expected_output_len
    {
        return Err(RusticolError::invalid_argument(
            "on-the-fly contracted input, output, tile, or normalization shape is invalid",
        ));
    }
    let supported_reducer = matches!(
        contraction.runtime_reducer(),
        None | Some(RuntimeColorContractionReducer::SymmetricGroupFourier(_))
    );
    if contraction.component_count() != 1
        || contraction.destination_count() == 0
        || !supported_reducer
    {
        return Err(RusticolError::integrity(
            "on-the-fly contracted execution requires a supported one-component metric",
        ));
    }
    Ok(())
}

/// Apply A-dagger C A independently for each selected physical helicity.
/// Structural-zero amplitudes are represented by absent projection entries
/// and therefore make every touching bilinear term exactly zero.
#[allow(clippy::too_many_arguments)]
fn reduce_contracted_color(
    contraction: &RecurrenceColorContraction,
    symmetric_group_workspace: Option<&mut RuntimeSymmetricGroupColorWorkspace>,
    amplitude_destinations: &[Option<usize>],
    amplitudes: &[(f64, f64)],
    point_count: usize,
    helicity_count: usize,
    structural_color_count: usize,
    normalization_factor: f64,
    accumulate: impl FnMut(usize, usize, f64),
) -> RusticolResult<()> {
    let projection_len = helicity_count
        .checked_mul(structural_color_count)
        .ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly contracted projection shape exceeds usize")
        })?;
    if amplitude_destinations.len() != projection_len {
        return Err(RusticolError::integrity(
            "on-the-fly contracted amplitude projection has the wrong shape",
        ));
    }
    match contraction.runtime_reducer() {
        Some(RuntimeColorContractionReducer::SymmetricGroupFourier(reducer)) => {
            let workspace = symmetric_group_workspace.ok_or_else(|| {
                RusticolError::integrity("on-the-fly symmetric-group color workspace is absent")
            })?;
            return reduce_symmetric_group_contracted_color(
                contraction,
                reducer,
                workspace,
                amplitude_destinations,
                amplitudes,
                point_count,
                helicity_count,
                structural_color_count,
                normalization_factor,
                accumulate,
            );
        }
        Some(RuntimeColorContractionReducer::Walsh(_)) => {
            return Err(RusticolError::integrity(
                "on-the-fly contracted execution does not support Walsh color storage",
            ));
        }
        None => {}
    }
    reduce_direct_contracted_color(
        contraction.runtime_entries(),
        amplitude_destinations,
        amplitudes,
        point_count,
        helicity_count,
        structural_color_count,
        normalization_factor,
        accumulate,
    )
}

#[allow(clippy::too_many_arguments)]
fn reduce_direct_contracted_color(
    entries: impl IntoIterator<Item = RuntimeColorContractionEntry>,
    amplitude_destinations: &[Option<usize>],
    amplitudes: &[(f64, f64)],
    point_count: usize,
    helicity_count: usize,
    structural_color_count: usize,
    normalization_factor: f64,
    mut accumulate: impl FnMut(usize, usize, f64),
) -> RusticolResult<()> {
    if amplitude_destinations.len()
        != helicity_count
            .checked_mul(structural_color_count)
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly direct contracted projection shape exceeds usize",
                )
            })?
    {
        return Err(RusticolError::integrity(
            "on-the-fly direct contracted amplitude projection has the wrong shape",
        ));
    }
    for entry in entries {
        let left_color = usize::try_from(entry.left_destination_id).map_err(|_| {
            RusticolError::artifact("on-the-fly contracted left destination exceeds usize")
        })?;
        let right_color = usize::try_from(entry.right_destination_id).map_err(|_| {
            RusticolError::artifact("on-the-fly contracted right destination exceeds usize")
        })?;
        if left_color >= structural_color_count || right_color >= structural_color_count {
            return Err(RusticolError::integrity(
                "on-the-fly contracted metric destination is outside its structural basis",
            ));
        }
        for helicity in 0..helicity_count {
            let base = helicity * structural_color_count;
            let (Some(left_destination), Some(right_destination)) = (
                amplitude_destinations[base + left_color],
                amplitude_destinations[base + right_color],
            ) else {
                continue;
            };
            let left_base = left_destination.checked_mul(point_count).ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly contracted left amplitude offset exceeds usize",
                )
            })?;
            let right_base = right_destination.checked_mul(point_count).ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly contracted right amplitude offset exceeds usize",
                )
            })?;
            if left_base
                .checked_add(point_count)
                .is_none_or(|end| end > amplitudes.len())
                || right_base
                    .checked_add(point_count)
                    .is_none_or(|end| end > amplitudes.len())
            {
                return Err(RusticolError::integrity(
                    "on-the-fly contracted amplitude destination is absent",
                ));
            }
            for point in 0..point_count {
                let (left_re, left_im) = amplitudes[left_base + point];
                let (right_re, right_im) = amplitudes[right_base + point];
                let value = normalization_factor
                    * entry.contract_real_bilinear(left_re, left_im, right_re, right_im);
                accumulate(point, helicity, value);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reduce_symmetric_group_contracted_color(
    contraction: &RecurrenceColorContraction,
    reducer: &crate::recurrence::RuntimeSymmetricGroupColorContraction,
    workspace: &mut RuntimeSymmetricGroupColorWorkspace,
    amplitude_destinations: &[Option<usize>],
    amplitudes: &[(f64, f64)],
    point_count: usize,
    helicity_count: usize,
    structural_color_count: usize,
    normalization_factor: f64,
    mut accumulate: impl FnMut(usize, usize, f64),
) -> RusticolResult<()> {
    if reducer.local_group_count() != contraction.local_group_count() as usize
        || contraction.destination_by_group().len() != structural_color_count
        || reducer.local_group_count() != structural_color_count
    {
        return Err(RusticolError::integrity(
            "on-the-fly symmetric-group reducer disagrees with its structural owner basis",
        ));
    }
    for helicity in 0..helicity_count {
        let projection_base = helicity * structural_color_count;
        reducer.reduce_lanes(workspace, point_count, |local_group, point| {
            // Contracted query preparation already inverted the authenticated
            // owner projection and retained this table in normalized group
            // order.  Applying destination_by_group here a second time would
            // silently contract A(P^2 g) for a non-involutive permutation.
            let Some(amplitude_destination) = amplitude_destinations[projection_base + local_group]
            else {
                return Ok((0.0, 0.0));
            };
            let amplitude_index = amplitude_destination
                .checked_mul(point_count)
                .and_then(|base| base.checked_add(point))
                .ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "on-the-fly symmetric-group amplitude offset exceeds usize",
                    )
                })?;
            amplitudes.get(amplitude_index).copied().ok_or_else(|| {
                RusticolError::integrity(
                    "on-the-fly symmetric-group amplitude destination is absent",
                )
            })
        })?;
        for (point, value) in workspace.reduced(point_count)?.iter().copied().enumerate() {
            accumulate(point, helicity, normalization_factor * value);
        }
    }
    Ok(())
}

fn merge_execution_report(
    aggregate: &mut Option<OnTheFlyQueryFamilyExecutionReportV1>,
    report: OnTheFlyQueryFamilyExecutionReportV1,
) -> RusticolResult<()> {
    let Some(target) = aggregate.as_mut() else {
        *aggregate = Some(report);
        return Ok(());
    };
    target.cache_hit &= report.cache_hit;
    macro_rules! checked_add_report_field {
        ($field:ident) => {
            target.$field = target.$field.checked_add(report.$field).ok_or_else(|| {
                RusticolError::invalid_argument(concat!(
                    "on-the-fly contracted execution report ",
                    stringify!($field),
                    " exceeds u32"
                ))
            })?;
        };
    }
    checked_add_report_field!(source_calls);
    checked_add_report_field!(source_rows);
    checked_add_report_field!(contribution_calls);
    checked_add_report_field!(contribution_rows);
    checked_add_report_field!(finalization_calls);
    checked_add_report_field!(finalization_rows);
    checked_add_report_field!(closure_calls);
    checked_add_report_field!(closure_rows);
    Ok(())
}

fn reduce_total_into(
    requests: &[OnTheFlyLcQueryRequestV1],
    amplitude_destinations: &[Option<usize>],
    amplitudes: &[(f64, f64)],
    point_count: usize,
    normalization_factor: f64,
    output: &mut [f64],
) -> RusticolResult<()> {
    if amplitude_destinations.len() != requests.len() {
        return Err(RusticolError::integrity(
            "on-the-fly amplitude projection has the wrong request count",
        ));
    }
    for (request, destination) in requests.iter().zip(amplitude_destinations) {
        let Some(destination) = *destination else {
            continue;
        };
        let base = destination.checked_mul(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly amplitude destination offset exceeds usize")
        })?;
        let end = base.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly amplitude range exceeds usize")
        })?;
        if end > amplitudes.len() {
            return Err(RusticolError::integrity(
                "on-the-fly amplitude destination is absent",
            ));
        }
        let weight = request.total_coefficient() * normalization_factor;
        accumulate_lc_diagonal_amplitude(
            point_count,
            weight,
            |point| amplitudes[base + point],
            |point, value| output[point] += value,
        );
    }
    Ok(())
}

fn reduce_resolved_into(
    requests: &[OnTheFlyLcQueryRequestV1],
    amplitude_destinations: &[Option<usize>],
    amplitudes: &[(f64, f64)],
    point_count: usize,
    normalization_factor: f64,
    layout: LcResolvedOutputLayout,
    output: &mut [f64],
) -> RusticolResult<()> {
    if amplitude_destinations.len() != requests.len() {
        return Err(RusticolError::integrity(
            "on-the-fly amplitude projection has the wrong request count",
        ));
    }
    for (request, destination) in requests.iter().zip(amplitude_destinations) {
        let Some(destination) = *destination else {
            continue;
        };
        let base = destination.checked_mul(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly amplitude destination offset exceeds usize")
        })?;
        let end = base.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly amplitude range exceeds usize")
        })?;
        if end > amplitudes.len() {
            return Err(RusticolError::integrity(
                "on-the-fly amplitude destination is absent",
            ));
        }
        for target in &request.reduction_targets {
            let weight = target.coefficient * normalization_factor;
            accumulate_lc_diagonal_amplitude(
                point_count,
                weight,
                |point| amplitudes[base + point],
                |point, value| {
                    let target_index = layout
                        .index(point, target.helicity_position, target.color_position)
                        .expect("validated on-the-fly resolved coordinate");
                    output[target_index] += value;
                },
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn profile_from_report(
    total: Duration,
    parameter_setup: Duration,
    input_setup: Duration,
    execution: Duration,
    reduction: Duration,
    momentum_scalar_count: usize,
    point_count: u32,
    report: OnTheFlyQueryFamilyExecutionReportV1,
) -> RuntimeProfile {
    let calls = u64::from(report.source_calls)
        + u64::from(report.contribution_calls)
        + u64::from(report.finalization_calls)
        + u64::from(report.closure_calls);
    let rows = u64::from(report.source_rows)
        + u64::from(report.contribution_rows)
        + u64::from(report.finalization_rows)
        + u64::from(report.closure_rows);
    direct_profile_from_delta(
        total,
        parameter_setup,
        input_setup,
        reduction,
        DirectProfileDelta {
            direct_execution: execution,
            execution: DirectExecutionCounters {
                source_calls: u64::from(report.source_calls),
                source_rows: u64::from(report.source_rows),
                contribution_calls: u64::from(report.contribution_calls),
                contribution_rows: u64::from(report.contribution_rows),
                finalization_calls: u64::from(report.finalization_calls),
                finalization_rows: u64::from(report.finalization_rows),
                closure_calls: u64::from(report.closure_calls),
                closure_rows: u64::from(report.closure_rows),
                ..DirectExecutionCounters::default()
            },
            traffic: DirectArenaTrafficCounters {
                calls,
                rows,
                points: calls.saturating_mul(u64::from(point_count)),
                ..DirectArenaTrafficCounters::default()
            },
            activity: DirectRuntimeActivityCounters {
                momentum_fill_calls: 1,
                momentum_scalar_values_filled: momentum_scalar_count as u64,
                schedule_executions: 1,
                ..DirectRuntimeActivityCounters::default()
            },
            ..DirectProfileDelta::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::super::recurrence_backend::on_the_fly_adapter_tests::{
        direct_catalog, prepared_pool, source_domains,
    };
    use super::*;
    use crate::recurrence::CheckedTableRange;
    use crate::recurrence::SemanticDigest;
    use crate::recurrence::on_the_fly::{OnTheFlyLcSelectorV1, scalar_adapter_test_seed};
    use crate::recurrence::template::{CouplingOrderTermRow, IndexedRangeRow};
    use crate::recurrence::validated_template_fixture;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn scalar_query() -> DecodedLcQueryV1 {
        let seed = scalar_adapter_test_seed(digest(1), digest(2), digest(3), digest(4)).unwrap();
        DecodedLcQueryV1::new(&seed, vec![0, 1], &[0, 0], OnTheFlyLcSelectorV1::Singlet).unwrap()
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn scalar_lane_with_threads(query_construction_threads: usize) -> OnTheFlyNativeRuntime {
        let mut template_input = validated_template_fixture().into_input();
        template_input.coupling_order_ranges.push(IndexedRangeRow {
            id: 1,
            range: CheckedTableRange::new(0, 1),
        });
        template_input
            .coupling_order_terms
            .push(CouplingOrderTermRow {
                set_id: 1,
                name_string_id: 0,
                power: 1,
            });
        let templates = template_input.validate().unwrap();
        let summary = templates.summary();
        let direct_digest = digest(40);
        let direct = direct_catalog(direct_digest);
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            direct_digest,
        )
        .unwrap();
        let pool = prepared_pool(&templates, direct_digest);
        let sources = pool.bind_source_domains(source_domains()).unwrap();
        let resolver = pool.into_on_the_fly_resolver(sources);
        let defaults = vec![
            crate::EagerComplex64::new(0.0, 0.0);
            usize::try_from(summary.parameter_count).unwrap()
        ];
        OnTheFlyNativeRuntime::new(
            templates,
            direct,
            seed,
            resolver,
            query_construction_threads,
            query_construction_threads,
            defaults,
            Vec::new(),
            &[],
        )
        .unwrap()
    }

    fn scalar_lane() -> OnTheFlyNativeRuntime {
        scalar_lane_with_threads(4)
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    fn scalar_request(lane: &OnTheFlyNativeRuntime, coefficient: f64) -> OnTheFlyLcQueryRequestV1 {
        let query = DecodedLcQueryV1::new(
            lane.seed(),
            vec![0, 1],
            &[0, 0],
            OnTheFlyLcSelectorV1::Singlet,
        )
        .unwrap();
        OnTheFlyLcQueryRequestV1::new(
            query,
            vec![OnTheFlyLcReductionTargetV1::new(0, 0, coefficient).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn lc_reduction_targets_reject_nonfinite_or_negative_coefficients() {
        assert!(OnTheFlyLcReductionTargetV1::new(0, 0, -1.0).is_err());
        assert!(OnTheFlyLcReductionTargetV1::new(0, 0, f64::NAN).is_err());
        assert_eq!(
            OnTheFlyLcReductionTargetV1::new(3, 5, 2.0)
                .unwrap()
                .coefficient,
            2.0
        );
    }

    #[test]
    fn lc_query_rejects_duplicate_resolved_coordinates() {
        let duplicate = vec![
            OnTheFlyLcReductionTargetV1::new(2, 3, 1.0).unwrap(),
            OnTheFlyLcReductionTargetV1::new(2, 3, 2.0).unwrap(),
        ];
        assert!(OnTheFlyLcQueryRequestV1::new(scalar_query(), duplicate).is_err());
    }

    #[test]
    fn lc_query_rejects_zero_total_reduction_weight() {
        let zero = vec![OnTheFlyLcReductionTargetV1::new(0, 0, 0.0).unwrap()];
        assert!(OnTheFlyLcQueryRequestV1::new(scalar_query(), zero).is_err());
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn serial_and_bounded_parallel_query_construction_match() {
        let mut serial = scalar_lane_with_threads(1);
        let mut parallel = scalar_lane_with_threads(4);
        let serial_requests = [scalar_request(&serial, 1.0), scalar_request(&serial, 2.0)];
        let parallel_requests = [
            scalar_request(&parallel, 1.0),
            scalar_request(&parallel, 2.0),
        ];

        assert!(!serial.prepare_lc_queries(&serial_requests, 1).unwrap());
        assert!(!parallel.prepare_lc_queries(&parallel_requests, 1).unwrap());
        assert_eq!(
            serial.retained_state_census(),
            parallel.retained_state_census()
        );

        let mut serial_value = [f64::NAN];
        let mut parallel_value = [f64::NAN];
        serial
            .run_f64_into(&[0.0; 8], 1, &[], 1.0, &mut serial_value)
            .unwrap();
        parallel
            .run_f64_into(&[0.0; 8], 1, &[], 1.0, &mut parallel_value)
            .unwrap();
        assert_eq!(serial_value, parallel_value);
    }

    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[test]
    fn process_preparation_is_lazy_shared_compact_and_clearable() {
        let mut lane = scalar_lane();
        assert_eq!(lane.process_preparation_count(), 0);
        assert_eq!(
            lane.process_preparation_census(),
            OnTheFlyProcessPreparationCensusV1::default()
        );
        assert_eq!(lane.coupling_policy_census(), None);
        assert_eq!(lane.coupling_policy_resolution(), None);
        assert_eq!(lane.retained_family_count(), 0);

        let first = scalar_request(&lane, 1.0);
        assert!(
            !lane
                .prepare_lc_queries(std::slice::from_ref(&first), 1)
                .unwrap()
        );
        assert_eq!(lane.retained_state_census().family_count, 0);
        assert_eq!(lane.pending_family_count(), 1);
        assert_eq!(lane.process_preparation_count(), 1);
        assert_eq!(
            lane.process_preparation_census(),
            OnTheFlyProcessPreparationCensusV1 {
                catalog_validation_count: 1,
                grammar_preparation_count: 1,
            }
        );
        assert!(lane.coupling_policy_census().is_some());
        assert!(lane.coupling_policy_resolution().is_some());
        let mut first_value = [0.0];
        lane.run_f64_into(&[0.0; 8], 1, &[], 1.0, &mut first_value)
            .unwrap();
        // This compact fixture is an authenticated structural-zero query.  It
        // deliberately exercises the outer retained-family lifecycle without
        // manufacturing an executor binding that the fixture's prepared pool
        // does not own; nonzero A -> B -> A executor reuse is covered in the
        // family executor tests below this public lane.
        assert_eq!(first_value, [0.0]);
        assert_eq!(lane.retained_state_census().family_count, 1);
        assert_eq!(lane.pending_family_count(), 0);
        assert!(
            lane.prepare_lc_queries(std::slice::from_ref(&first), 1)
                .unwrap()
        );

        let second = scalar_request(&lane, 2.0);
        assert!(
            !lane
                .prepare_lc_queries(std::slice::from_ref(&second), 1)
                .unwrap()
        );
        let mut second_value = [0.0];
        lane.run_f64_into(&[0.0; 8], 1, &[], 1.0, &mut second_value)
            .unwrap();
        assert_eq!(second_value, [2.0 * first_value[0]]);

        assert!(
            !lane
                .prepare_lc_queries(std::slice::from_ref(&first), 1)
                .unwrap()
        );
        let mut repeated = [0.0];
        lane.run_f64_into(&[0.0; 8], 1, &[], 1.0, &mut repeated)
            .unwrap();
        assert_eq!(repeated, first_value);

        let batched = [first.clone(), second.clone()];
        assert!(!lane.prepare_lc_queries(&batched, 1).unwrap());
        let mut batched_value = [0.0];
        lane.run_f64_into(&[0.0; 8], 1, &[], 1.0, &mut batched_value)
            .unwrap();
        assert_eq!(batched_value, [3.0 * first_value[0]]);
        assert_eq!(lane.process_preparation_count(), 1);
        assert_eq!(
            lane.process_preparation_census(),
            OnTheFlyProcessPreparationCensusV1 {
                catalog_validation_count: 1,
                grammar_preparation_count: 1,
            }
        );

        let retained = lane.retained_state_census();
        assert_eq!(retained.family_count, 1);
        assert_eq!(retained.request_count, 2);
        assert_eq!(retained.amplitude_destination_count, 0);
        assert_eq!(retained.executor_handle_count, 0);
        assert_eq!(retained.query_local_trace_count, 0);
        assert_eq!(retained.embedded_lookup_key_count, 0);
        assert_eq!(lane.pending_family_count(), 0);

        lane.clear().unwrap();
        assert_eq!(lane.process_preparation_count(), 0);
        assert_eq!(
            lane.process_preparation_census(),
            OnTheFlyProcessPreparationCensusV1::default()
        );
        assert_eq!(lane.coupling_policy_census(), None);
        assert_eq!(lane.coupling_policy_resolution(), None);
        assert_eq!(
            lane.retained_state_census(),
            OnTheFlyRetainedStateCensusV1::default()
        );
        assert_eq!(lane.pending_family_count(), 0);
        assert!(lane.source_momenta_scratch.is_empty());
        assert_eq!(lane.source_momenta_scratch.capacity(), 0);
        assert!(lane.amplitude_scratch.is_empty());
        assert_eq!(lane.amplitude_scratch.capacity(), 0);

        assert!(!lane.prepare_lc_queries(&[first], 1).unwrap());
        assert_eq!(lane.process_preparation_count(), 1);
        assert_eq!(
            lane.process_preparation_census(),
            OnTheFlyProcessPreparationCensusV1 {
                catalog_validation_count: 1,
                grammar_preparation_count: 1,
            }
        );
        let mut rebuilt = [0.0];
        lane.run_f64_into(&[0.0; 8], 1, &[], 1.0, &mut rebuilt)
            .unwrap();
        assert_eq!(rebuilt, first_value);
    }

    #[test]
    fn profile_maps_grouped_family_role_counters_without_dense_plan_counters() {
        let profile = profile_from_report(
            Duration::from_millis(11),
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(7),
            Duration::from_millis(1),
            96,
            3,
            OnTheFlyQueryFamilyExecutionReportV1 {
                cache_hit: true,
                source_calls: 2,
                source_rows: 4,
                contribution_calls: 3,
                contribution_rows: 17,
                finalization_calls: 1,
                finalization_rows: 5,
                closure_calls: 2,
                closure_rows: 7,
            },
        );
        assert_eq!(profile.recurrence_schedule_execution_count, 1);
        assert_eq!(profile.recurrence_momentum_scalar_value_count, 96);
        assert_eq!(profile.recurrence_source_call_count, 2);
        assert_eq!(profile.recurrence_source_row_count, 4);
        assert_eq!(profile.recurrence_contribution_call_count, 3);
        assert_eq!(profile.recurrence_contribution_row_count, 17);
        assert_eq!(profile.recurrence_finalization_call_count, 1);
        assert_eq!(profile.recurrence_finalization_row_count, 5);
        assert_eq!(profile.recurrence_closure_call_count, 2);
        assert_eq!(profile.recurrence_closure_row_count, 7);
    }

    #[test]
    fn source_major_workspace_is_reused_after_warmup() {
        let seed = scalar_adapter_test_seed(digest(1), digest(2), digest(3), digest(4)).unwrap();
        let point_major = (0..16).map(f64::from).collect::<Vec<_>>();
        let mut scratch = Vec::new();
        let output_len =
            on_the_fly_source_major_momenta_into(&seed, &point_major, 2, 4, &mut scratch).unwrap();
        assert_eq!(output_len, 16);
        assert_eq!(scratch[..4], [0.0, 8.0, 1.0, 9.0]);
        let warmed_pointer = scratch.as_ptr();
        let warmed_capacity = scratch.capacity();
        on_the_fly_source_major_momenta_into(&seed, &point_major, 2, 4, &mut scratch).unwrap();
        assert_eq!(scratch.as_ptr(), warmed_pointer);
        assert_eq!(scratch.capacity(), warmed_capacity);
    }

    #[test]
    fn source_major_workspace_preserves_values_across_partial_batches() {
        use crate::recurrence::on_the_fly::{OnTheFlyExternalColorRoleV1, OnTheFlySourceAnchorV1};

        let base = scalar_adapter_test_seed(digest(1), digest(2), digest(3), digest(4)).unwrap();
        let anchors = (0..3)
            .map(|slot| {
                OnTheFlySourceAnchorV1::new(
                    slot,
                    slot,
                    false,
                    OnTheFlyExternalColorRoleV1::Singlet,
                    false,
                    None,
                    base.source_anchors()[0].states().to_vec(),
                )
                .unwrap()
            })
            .collect();
        let seed = OnTheFlyProcessSeedV1::new(
            base.process_digest(),
            base.model_digest(),
            base.template_catalog_digest(),
            base.prepared_pack_digest(),
            base.direct_catalog_digest(),
            digest(92),
            "raw-amplitude-test",
            crate::recurrence::ExactComplexRational::ONE,
            anchors,
            vec![1, 2, 0],
            base.coupling_order_policy(),
            base.coupling_hierarchies().to_vec(),
            base.explicit_coupling_limits().to_vec(),
            Vec::new(),
        )
        .unwrap();
        let mut scratch = vec![53.0; seed.external_permutation().len() * 4 * 17];
        let pointer = scratch.as_ptr();
        let capacity = scratch.capacity();
        for lorentz_count in [1, 3, 4] {
            for point_count in [1, 17, 3, 8, 1] {
                let point_width = seed.external_permutation().len() * lorentz_count;
                let input = (0..point_width * point_count)
                    .map(|index| {
                        if index % 7 == 0 {
                            -0.0
                        } else {
                            index as f64 / 11.0
                        }
                    })
                    .collect::<Vec<_>>();
                scratch.fill(53.0);
                let count = on_the_fly_source_major_momenta_into(
                    &seed,
                    &input,
                    point_count as u32,
                    lorentz_count as u16,
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(count, input.len());
                for (source, public) in seed.external_permutation().iter().copied().enumerate() {
                    for lorentz in 0..lorentz_count {
                        for point in 0..point_count {
                            assert_eq!(
                                scratch[(source * lorentz_count + lorentz) * point_count + point]
                                    .to_bits(),
                                input[point * point_width
                                    + public as usize * lorentz_count
                                    + lorentz]
                                    .to_bits(),
                            );
                        }
                    }
                }
                assert!(scratch[count..].iter().all(|value| *value == 53.0));
                assert_eq!(scratch.as_ptr(), pointer);
                assert_eq!(scratch.capacity(), capacity);
            }
        }
        assert!(on_the_fly_source_major_momenta_into(&seed, &[1.0], 1, 4, &mut scratch).is_err());
        assert!(on_the_fly_source_major_momenta_into(&seed, &[], 0, 4, &mut scratch).is_err());
    }

    #[test]
    fn mixed_structural_zero_and_trace_preserve_zero_resolved_slot() {
        let query = scalar_query();
        let requests = vec![
            OnTheFlyLcQueryRequestV1::new(
                query.clone(),
                vec![OnTheFlyLcReductionTargetV1::new(0, 0, 1.0).unwrap()],
            )
            .unwrap(),
            OnTheFlyLcQueryRequestV1::new(
                query,
                vec![OnTheFlyLcReductionTargetV1::new(1, 0, 2.0).unwrap()],
            )
            .unwrap(),
        ];
        // Request zero is a structural zero and therefore has no amplitude
        // destination. Request one is the sole nonzero trace destination.
        let amplitude_destinations = [None, Some(0)];
        let amplitudes = [(3.0, 4.0)];

        let mut total = [0.0];
        reduce_total_into(
            &requests,
            &amplitude_destinations,
            &amplitudes,
            1,
            1.0,
            &mut total,
        )
        .unwrap();
        assert_eq!(total, [50.0]);

        let layout = LcResolvedOutputLayout::new(1, 2, 1).unwrap();
        let mut resolved = [0.0; 2];
        reduce_resolved_into(
            &requests,
            &amplitude_destinations,
            &amplitudes,
            1,
            1.0,
            layout,
            &mut resolved,
        )
        .unwrap();
        assert_eq!(resolved, [0.0, 50.0]);
    }

    #[test]
    fn all_structural_zero_family_reduces_to_exact_zero_without_amplitudes() {
        let requests = [OnTheFlyLcQueryRequestV1::new(
            scalar_query(),
            vec![OnTheFlyLcReductionTargetV1::new(0, 0, 1.0).unwrap()],
        )
        .unwrap()];
        let mut total = [0.0];
        reduce_total_into(&requests, &[None], &[], 1, 1.0, &mut total).unwrap();
        assert_eq!(total, [0.0]);

        let layout = LcResolvedOutputLayout::new(1, 1, 1).unwrap();
        let mut resolved = [0.0];
        reduce_resolved_into(&requests, &[None], &[], 1, 1.0, layout, &mut resolved).unwrap();
        assert_eq!(resolved, [0.0]);
    }

    #[test]
    fn symmetric_group_contracted_query_projection_reaches_final_reducer_in_group_order() {
        let destination_by_group = (0..13)
            .map(|group| ((group * 5) % 13) as u32)
            .collect::<Vec<_>>();
        let mut destination_by_owner_ordinal = vec![0_u32; 13];
        for (group, owner) in destination_by_group.iter().copied().enumerate() {
            destination_by_owner_ordinal[owner as usize] = group as u32;
        }
        let contraction = RecurrenceColorContraction::symmetric_group_s3_for_runtime_test(
            destination_by_group.clone(),
            13,
        );
        let RuntimeColorContractionReducer::SymmetricGroupFourier(reducer) =
            contraction.runtime_reducer().unwrap()
        else {
            panic!("expected symmetric-group runtime reducer")
        };
        let covered_group_count = reducer.channel_count() * reducer.group_order();
        assert!(reducer.channel_count() > 0);
        assert!(reducer.local_group_count() > covered_group_count);
        assert!(reducer.residual_entries().iter().any(|entry| {
            entry.left_group_index < covered_group_count as u32
                && entry.right_group_index >= covered_group_count as u32
        }));
        let point_count = 3;
        let helicity_count = 2;
        let structural_color_count = 13;
        let normalization = 1.75;
        let mut projection = vec![None; helicity_count * structural_color_count];
        let mut amplitudes = Vec::new();
        let mut local_values =
            vec![(0.0, 0.0); helicity_count * structural_color_count * point_count];
        // Query construction enumerates owners.  Its production slot helper
        // stores every outcome in the inverse-mapped Fourier-group slot.  The
        // complex values deliberately depend on the owner so a second
        // application of this order-4 permutation cannot pass accidentally.
        for helicity in 0..helicity_count {
            for owner in 0..structural_color_count {
                let group = destination_by_owner_ordinal[owner] as usize;
                let structural_zero = helicity == 1 && group == 4;
                if structural_zero {
                    continue;
                }
                let destination = amplitudes.len() / point_count;
                let projection_index = contracted_projection_index(
                    helicity,
                    owner,
                    structural_color_count,
                    &destination_by_owner_ordinal,
                )
                .unwrap();
                projection[projection_index] = Some(destination);
                for point in 0..point_count {
                    let value = (
                        (helicity * 19 + owner * 7 + point * 3) as f64 / 11.0 - 1.0,
                        (helicity * 5 + owner * 13 + point) as f64 / 17.0 - 0.5,
                    );
                    amplitudes.push(value);
                    local_values
                        [(helicity * structural_color_count + group) * point_count + point] = value;
                }
            }
        }

        let mut expected = vec![0.0; helicity_count * point_count];
        let helicity_stride = structural_color_count * point_count;
        for helicity in 0..helicity_count {
            let start = helicity * helicity_stride;
            let dense = RecurrenceColorContraction::symmetric_group_s3_dense_for_runtime_test(
                &local_values[start..start + helicity_stride],
                point_count,
            );
            for (point, value) in dense.into_iter().enumerate() {
                expected[point * helicity_count + helicity] = normalization * value;
            }
        }

        // Install exactly the compact family state produced by query
        // preparation, then enter the native lane's real final-reduction seam.
        // No decoded queries or owner-indexed side table survives this point.
        let mut lane = scalar_lane();
        lane.pending_contracted_family = Some(PreparedOnTheFlyContractedFamilyV1 {
            helicity_ordinals: (0..helicity_count).collect::<Vec<_>>().into_boxed_slice(),
            structural_color_count,
            amplitude_destinations: projection.into_boxed_slice(),
            executor_handle: None,
            census: None,
            logical_point_capacity: point_count as u32,
        });
        lane.amplitude_scratch = amplitudes;
        lane.install_symmetric_group_color_workspace(reducer.workspace(point_count).unwrap())
            .unwrap();
        let mut actual = vec![0.0; expected.len()];
        lane.reduce_current_contracted_color(
            &contraction,
            point_count,
            helicity_count,
            normalization,
            |point, helicity, value| actual[point * helicity_count + helicity] += value,
        )
        .unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            let scale = expected.abs().max(1.0);
            assert!((actual - expected).abs() <= 2.0e-11 * scale);
        }
    }

    #[test]
    fn contracted_metric_preserves_imaginary_off_diagonal_terms_per_helicity() {
        let entries = [
            RuntimeColorContractionEntry {
                left_destination_id: 0,
                right_destination_id: 0,
                coefficient_re: 1.0,
                coefficient_im: 0.0,
            },
            RuntimeColorContractionEntry {
                left_destination_id: 0,
                right_destination_id: 1,
                coefficient_re: 2.0,
                coefficient_im: 3.0,
            },
            RuntimeColorContractionEntry {
                left_destination_id: 1,
                right_destination_id: 1,
                coefficient_re: 0.5,
                coefficient_im: 0.0,
            },
        ];
        // H0 owns both structural amplitudes. H1's second structural query is
        // an authenticated zero and every bilinear touching it must vanish.
        let projection = [Some(0), Some(1), Some(2), None];
        let amplitudes = [(1.0, 2.0), (3.0, 4.0), (2.0, 0.0)];
        let mut resolved = [0.0; 2];
        reduce_direct_contracted_color(
            entries,
            &projection,
            &amplitudes,
            1,
            2,
            2,
            2.0,
            |point, helicity, value| resolved[point * 2 + helicity] += value,
        )
        .unwrap();

        // Re((2+3i) * (1+2i) * conj(3+4i)) = 16. The complete
        // H0 metric is 5 + 16 + 12.5; H1 retains only |2|^2.
        assert_eq!(resolved, [67.0, 8.0]);
        assert_eq!(resolved.iter().sum::<f64>(), 75.0);
    }

    #[test]
    fn contracted_metric_rejects_projection_and_destination_shape_mismatches() {
        let entry = RuntimeColorContractionEntry {
            left_destination_id: 0,
            right_destination_id: 1,
            coefficient_re: 1.0,
            coefficient_im: 0.0,
        };
        assert!(
            reduce_direct_contracted_color(
                [entry],
                &[Some(0)],
                &[(1.0, 0.0)],
                1,
                1,
                2,
                1.0,
                |_, _, _| {},
            )
            .is_err()
        );
        assert!(
            reduce_direct_contracted_color(
                [RuntimeColorContractionEntry {
                    right_destination_id: 2,
                    ..entry
                }],
                &[Some(0), Some(1)],
                &[(1.0, 0.0), (1.0, 0.0)],
                1,
                1,
                2,
                1.0,
                |_, _, _| {},
            )
            .is_err()
        );
    }
}
