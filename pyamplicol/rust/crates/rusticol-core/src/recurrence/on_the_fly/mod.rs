// SPDX-License-Identifier: 0BSD

//! Private query-local LC recurrence construction and direct interpretation.
//!
//! This module deliberately consumes one decoded physical LC selector.  It
//! never constructs a process-wide color plan, selector projection, replay
//! catalog, feasibility index, or [`super::DirectRecurrencePlan`].

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::c_int;

use rayon::prelude::*;
use sha2::{Digest, Sha256};

use super::construct::{
    TemplateCatalog, aggregate_factor, combined_coupling_orders, merged_helicity_identity,
    merged_momentum, multiply_factors, output_factor_from_binding, quantum_parent_spin_matches,
    validate_crossed_source_state,
};
use super::contact_orbit_owner::{
    PreparedContactOrbitTransition, prepare_contact_orbit_transition,
    selected_contact_orbit_owner_tokens,
};
use super::direct_backend::{
    DirectExecutorHandle, DirectInteractionExecutorCapability, DirectInteractionGroupAction,
    DirectInteractionGroupView, DirectInteractionProgram, DirectInteractionScheduleView,
    DirectSingletonContributionFanoutProgram, DirectUnionSourceDispatchHandle,
    clear_direct_executor_error_detail,
};
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DirectClosureRow, DirectContributionRow,
    DirectExecutorRole, DirectFinalizationRow, DirectSourceRow,
};
use super::process::ProcessSourceStateRow;
use super::template::{
    ClosureRow, CurrentStateRow, EvaluatorContractKind, LCColorTransitionWitnessRow,
    QuantumFlowRow, SourceRow, TransitionRow, ValidatedRecurrenceTemplateInput,
};
use super::{
    CanonicalMomentumLinearForm, ContributionKey, CurrentCoreKey, CurrentHelicityIdentity,
    CurrentSourceBinding, DynamicLCColorState, DynamicLCColorStateInterner, ExactComplexRational,
    LCColorComponent, LCColorComponentKind, LCColorSourceSeed, LCColorWitnessTermId, MomentumTerm,
    PreparedDirectExecutorBinding, PreparedDirectExecutorCatalog, RecurrenceNodeKind,
    RecurrenceStrategy, SemanticDigest, SourceStateAssignment,
};
use crate::direct_arena::{
    AlignedF64Buffer, DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
    checked_aligned_point_stride, validate_direct_views,
};
use crate::{RusticolError, RusticolResult};

const MISSING_U32: u32 = u32::MAX;
const ON_THE_FLY_PROOF_DOMAIN: &[u8] = b"pyamplicol-on-the-fly-structural-proof-v1\0";

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("on-the-fly recurrence: {}", message.into()))
}

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(format!("on-the-fly recurrence: {}", message.into()))
}

#[allow(dead_code)] // Query-local compatibility and probe error constructor.
fn evaluation(message: impl Into<String>) -> RusticolError {
    RusticolError::evaluation(format!("on-the-fly recurrence: {}", message.into()))
}

fn checked_u32(value: usize, label: &str) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} exceeds u32")))
}

fn hash_len(hash: &mut Sha256, value: usize, label: &str) -> RusticolResult<()> {
    hash.update(
        u64::try_from(value)
            .map_err(|_| invalid(format!("{label} length exceeds u64")))?
            .to_le_bytes(),
    );
    Ok(())
}

fn hash_digest(hash: &mut Sha256, digest: SemanticDigest) {
    hash.update(digest.as_bytes());
}

fn hash_exact(hash: &mut Sha256, value: ExactComplexRational) {
    for part in [value.real(), value.imag()] {
        hash.update(part.numerator().to_le_bytes());
        hash.update(part.denominator().to_le_bytes());
    }
}

fn final_digest(hash: Sha256) -> RusticolResult<SemanticDigest> {
    SemanticDigest::new(hash.finalize().into())
}

mod coupling_policy;
mod family;
mod interpreter;
mod persisted;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
mod probe_guard;
mod projection;
mod public_query;
pub(crate) mod seed_codec;
mod source_builder;
mod source_seed;
mod sweep;
mod templates;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
mod test_support;
mod trace;

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) use coupling_policy::OnTheFlyCouplingPolicyCensusV1;
pub(crate) use coupling_policy::OnTheFlyResolvedCouplingPolicyV1;
pub use family::OnTheFlyQueryFamilyCensusV1;
#[cfg(feature = "on-the-fly-test-support")]
pub use family::on_the_fly_query_family_census_v1;
pub(crate) use family::snapshot::{OnTheFlyFamilySnapshotRefV1, OnTheFlyFamilySnapshotV1};
pub(crate) use family::{
    OnTheFlyQueryFamilyExecutionReportV1, OnTheFlyQueryFamilyExecutorV1,
    OnTheFlyQueryFamilyHandleV1, QueryFamilyTraceInput, build_streamed_query_family_candidate_v1,
};
#[allow(unused_imports)] // Query-local interpreter remains available to opt-in test support.
pub(crate) use interpreter::{
    OnTheFlyBoundUnionSourceProgram, OnTheFlyPreparedExecutorResolver,
    OnTheFlyStructuralInterpreter, OnTheFlyUnionSourceBindingView, OnTheFlyWorkspaceV1,
    ResolvedOnTheFlyExecutor,
};
pub(crate) use persisted::{
    PersistedHelicityAmplitudeTileV1, PersistedHelicityFamilyCacheCensusV1,
    PersistedHelicityFamilyExecutionReportV1, PersistedHelicityFamilyExecutorV1,
    PersistedHelicityFamilyInspectionCensusV1,
};
#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) use probe_guard::OnTheFlyForbiddenWorkGuardV1;
#[cfg(feature = "on-the-fly-test-support")]
pub(crate) use probe_guard::{OnTheFlyForbiddenWorkV1, reject_forbidden_work_if_probed};
use projection::*;
use public_query::OnTheFlyClosureAnchorPolicyV1;
pub(crate) use public_query::OnTheFlyLcSelectorV1;
pub(crate) use public_query::{DecodedLcQueryV1, OnTheFlySelectedSourceV1};
pub(crate) use seed_codec::decode_on_the_fly_process_seed_v1;
#[allow(unused_imports)] // Generation-only entries share this runtime validation seam.
pub(crate) use source_builder::{
    build_on_the_fly_process_seed_v1, build_on_the_fly_process_seeds_v1,
    parse_on_the_fly_process_seed_projection_v1, validate_on_the_fly_source_mass_bindings_v1,
};
#[cfg(test)]
pub(crate) use source_seed::scalar_adapter_test_seed;
#[allow(unused_imports)] // Pairing records are consumed by nested generation/test modules.
pub(crate) use source_seed::{
    ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI, OnTheFlyCouplingOrderPolicyV1,
    OnTheFlyExternalColorRoleV1, OnTheFlyPairingClassV1, OnTheFlyPairingEndpointV1,
    OnTheFlyProcessSeedIdentityV1, OnTheFlyProcessSeedV1, OnTheFlySourceAnchorV1,
    OnTheFlySourceExecutionSpecV1, OnTheFlySourceOrientationV1, OnTheFlySourceStateV1,
    OnTheFlySourceWavefunctionFamilyV1,
};
use sweep::*;
pub(crate) use templates::PreparedOnTheFlyGrammarV1;
use templates::*;
#[cfg(feature = "on-the-fly-test-support")]
pub use test_support::{OnTheFlyTestSupportReportV1, on_the_fly_test_support_probe_v1};
#[cfg(feature = "on-the-fly-test-support")]
pub(crate) use test_support::{
    build_on_the_fly_selected_trace_against_seed_v1, build_on_the_fly_selected_trace_v1,
};
#[cfg(feature = "on-the-fly-test-support")]
pub(crate) use trace::ON_THE_FLY_WORK_CENSUS_BASIS_V1;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) use trace::hash_current_key;
#[cfg(test)]
pub(crate) use trace::scalar_adapter_test_trace;
use trace::*;
pub(crate) use trace::{
    OnTheFlyExecutorKeyV1, OnTheFlyStructuralTraceV1, authenticated_prepared_executor_binding,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyProjectionProbeV1 {
    pub(crate) enabled: bool,
    pub(crate) applied: bool,
    pub(crate) pre: [u32; 3],
    pub(crate) post: [u32; 3],
}

/// One compact, selector-local LC trace ready for prepared-executor binding.
///
/// This production seam owns no global recurrence builder or direct plan. Its
/// inputs are precisely the compact process seed, one already-decoded public
/// query, and the authenticated prepared template catalogs.
pub(crate) struct OnTheFlySelectedQueryTraceV1 {
    pub(crate) query: DecodedLcQueryV1,
    pub(crate) trace: OnTheFlyStructuralTraceV1,
    pub(crate) projection: OnTheFlyProjectionProbeV1,
}

#[allow(clippy::large_enum_variant)] // Cold-path trace ownership avoids an extra allocation.
pub(crate) enum OnTheFlySelectedQueryOutcomeV1 {
    Trace(OnTheFlySelectedQueryTraceV1),
    StructuralZero { query: DecodedLcQueryV1 },
}

#[allow(dead_code)] // Compatibility entry point for the query-local probe path.
pub(crate) fn build_selected_lc_query_trace_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    seed: &OnTheFlyProcessSeedV1,
    coupling_policy: &OnTheFlyResolvedCouplingPolicyV1,
    query: DecodedLcQueryV1,
) -> RusticolResult<OnTheFlySelectedQueryOutcomeV1> {
    let catalog = validate_seed_against_templates(templates, seed)?;
    let grammar = prepare_on_the_fly_grammar_v1(templates, &catalog, seed)?;
    let query = grammar.canonicalize_query(seed, query)?;
    build_selected_lc_query_trace_impl(
        templates,
        direct,
        seed,
        coupling_policy,
        &grammar,
        query,
        true,
        false,
    )
}

/// Prepare the immutable process grammar and process-global coupling policy
/// in one pass over the validated template catalog.  Runtime load deliberately
/// does not call this; the first genuinely cold query family does.
pub(crate) fn prepare_on_the_fly_process_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<(PreparedOnTheFlyGrammarV1, OnTheFlyResolvedCouplingPolicyV1)> {
    let catalog = validate_seed_against_templates(templates, seed)?;
    let grammar = prepare_on_the_fly_grammar_v1(templates, &catalog, seed)?;
    let policy = coupling_policy::resolve_on_the_fly_coupling_policy_from_grammar_v1(
        templates, seed, &grammar,
    )?;
    Ok((grammar, policy))
}

/// Recreate model-only grammar while retaining the saved process-global
/// coupling decision; in particular, do not repeat the topology sweep.
pub(crate) fn restore_on_the_fly_grammar_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    policy: &OnTheFlyResolvedCouplingPolicyV1,
) -> RusticolResult<PreparedOnTheFlyGrammarV1> {
    if policy.seed_digest() != seed.semantic_digest()
        || policy.effective_limits().len() != seed.explicit_coupling_limits().len()
    {
        return Err(integrity(
            "saved on-the-fly coupling policy belongs to a different process",
        ));
    }
    let catalog = validate_seed_against_templates(templates, seed)?;
    prepare_on_the_fly_grammar_v1(templates, &catalog, seed)
}

fn resolve_indexed_query_results<T>(outcomes: Vec<RusticolResult<T>>) -> RusticolResult<Vec<T>> {
    outcomes.into_iter().collect()
}

/// One cold, bounded construction pool reused across every streamed query
/// chunk.  Keeping the pool outside the chunk loop avoids creating O(H x S)
/// thread pools while retaining deterministic indexed outcomes.
pub(crate) struct OnTheFlyQueryConstructionPoolV1 {
    pool: Option<rayon::ThreadPool>,
    thread_count: usize,
}

impl OnTheFlyQueryConstructionPoolV1 {
    pub(crate) fn new(thread_count: usize) -> RusticolResult<Self> {
        if thread_count == 0 {
            return Err(invalid(
                "on-the-fly query construction thread count must be positive",
            ));
        }
        let pool = (thread_count > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(thread_count)
                    .build()
                    .map_err(|error| {
                        RusticolError::internal(format!(
                            "could not create bounded on-the-fly query construction pool: {error}"
                        ))
                    })
            })
            .transpose()?;
        Ok(Self { pool, thread_count })
    }

    pub(crate) const fn chunk_capacity(&self) -> usize {
        self.thread_count
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_chunk(
        &self,
        templates: &ValidatedRecurrenceTemplateInput,
        direct: &PreparedDirectExecutorCatalog,
        seed: &OnTheFlyProcessSeedV1,
        coupling_policy: &OnTheFlyResolvedCouplingPolicyV1,
        grammar: &PreparedOnTheFlyGrammarV1,
        enable_cyclic_trace_reflection: bool,
        queries: Vec<DecodedLcQueryV1>,
    ) -> RusticolResult<Vec<OnTheFlySelectedQueryOutcomeV1>> {
        if queries.len() > self.thread_count {
            return Err(invalid(
                "streamed on-the-fly query chunk exceeds its bounded construction pool",
            ));
        }
        let build = |query| {
            build_selected_lc_query_trace_impl(
                templates,
                direct,
                seed,
                coupling_policy,
                grammar,
                query,
                true,
                enable_cyclic_trace_reflection,
            )
        };
        let outcomes = if let Some(pool) = &self.pool {
            pool.install(|| queries.into_par_iter().map(build).collect::<Vec<_>>())
        } else {
            queries.into_iter().map(build).collect::<Vec<_>>()
        };
        resolve_indexed_query_results(outcomes)
    }
}

fn build_selected_lc_query_family_serial_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    seed: &OnTheFlyProcessSeedV1,
    coupling_policy: &OnTheFlyResolvedCouplingPolicyV1,
    grammar: &PreparedOnTheFlyGrammarV1,
    enable_cyclic_trace_reflection: bool,
    queries: impl IntoIterator<Item = DecodedLcQueryV1>,
) -> RusticolResult<Vec<OnTheFlySelectedQueryOutcomeV1>> {
    queries
        .into_iter()
        .map(|query| {
            build_selected_lc_query_trace_impl(
                templates,
                direct,
                seed,
                coupling_policy,
                grammar,
                query,
                true,
                enable_cyclic_trace_reflection,
            )
        })
        .collect()
}

/// Build one exact public query family while sharing all immutable grammar
/// preparation.  Independent query-local sweeps run in parallel; their
/// outcomes and errors retain public selector order.
#[allow(clippy::too_many_arguments)] // Explicit cold-build context and execution policy.
pub(crate) fn build_selected_lc_query_family_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    seed: &OnTheFlyProcessSeedV1,
    coupling_policy: &OnTheFlyResolvedCouplingPolicyV1,
    grammar: &PreparedOnTheFlyGrammarV1,
    enable_cyclic_trace_reflection: bool,
    query_construction_threads: usize,
    queries: impl IntoIterator<Item = DecodedLcQueryV1>,
) -> RusticolResult<Vec<OnTheFlySelectedQueryOutcomeV1>> {
    if query_construction_threads == 0 {
        return Err(invalid(
            "on-the-fly query construction thread count must be positive",
        ));
    }
    if query_construction_threads == 1 {
        return build_selected_lc_query_family_serial_v1(
            templates,
            direct,
            seed,
            coupling_policy,
            grammar,
            enable_cyclic_trace_reflection,
            queries,
        );
    }

    let queries = queries.into_iter().collect::<Vec<_>>();
    if queries.len() < 2 {
        return build_selected_lc_query_family_serial_v1(
            templates,
            direct,
            seed,
            coupling_policy,
            grammar,
            enable_cyclic_trace_reflection,
            queries,
        );
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(query_construction_threads.min(queries.len()))
        .build()
        .map_err(|error| {
            RusticolError::internal(format!(
                "could not create bounded on-the-fly query construction pool: {error}"
            ))
        })?;
    let outcomes = pool.install(|| {
        queries
            .into_par_iter()
            .map(|query| {
                build_selected_lc_query_trace_impl(
                    templates,
                    direct,
                    seed,
                    coupling_policy,
                    grammar,
                    query,
                    true,
                    enable_cyclic_trace_reflection,
                )
            })
            .collect::<Vec<_>>()
    });
    resolve_indexed_query_results(outcomes)
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[allow(dead_code)] // Feature-only probe entry point.
pub(crate) fn build_selected_lc_query_trace_for_probe_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    seed: &OnTheFlyProcessSeedV1,
    query: DecodedLcQueryV1,
    enable_projection: bool,
    enable_cyclic_trace_reflection: bool,
) -> RusticolResult<OnTheFlySelectedQueryTraceV1> {
    let (grammar, coupling_policy) = prepare_on_the_fly_process_v1(templates, seed)?;
    let query = grammar.canonicalize_query(seed, query)?;
    match build_selected_lc_query_trace_impl(
        templates,
        direct,
        seed,
        &coupling_policy,
        &grammar,
        query,
        enable_projection,
        enable_cyclic_trace_reflection,
    )? {
        OnTheFlySelectedQueryOutcomeV1::Trace(selected) => Ok(selected),
        OnTheFlySelectedQueryOutcomeV1::StructuralZero { .. } => Err(evaluation(
            "selected probe query is an exact structural zero",
        )),
    }
}

#[allow(clippy::too_many_arguments)] // Explicit cold-build context and probe switches.
fn build_selected_lc_query_trace_impl(
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    seed: &OnTheFlyProcessSeedV1,
    coupling_policy: &OnTheFlyResolvedCouplingPolicyV1,
    grammar: &PreparedOnTheFlyGrammarV1,
    query: DecodedLcQueryV1,
    enable_projection: bool,
    enable_cyclic_trace_reflection: bool,
) -> RusticolResult<OnTheFlySelectedQueryOutcomeV1> {
    if seed.direct_catalog_digest() != direct.direct_template_catalog_digest() {
        return Err(integrity(
            "compact query direct-template catalog differs from its prepared catalog",
        ));
    }
    if coupling_policy.seed_digest() != seed.semantic_digest()
        || coupling_policy.effective_limits().len() != seed.explicit_coupling_limits().len()
    {
        return Err(integrity(
            "resolved coupling policy belongs to a different compact seed",
        ));
    }
    grammar.authenticate_canonical_query(seed, &query)?;
    let Some((trace, projection)) = build_selected_lc_trace_impl(
        templates,
        seed,
        grammar,
        coupling_policy.effective_limits(),
        &query,
        enable_projection,
        enable_cyclic_trace_reflection,
        true,
    )?
    else {
        return Ok(OnTheFlySelectedQueryOutcomeV1::StructuralZero { query });
    };
    Ok(OnTheFlySelectedQueryOutcomeV1::Trace(
        OnTheFlySelectedQueryTraceV1 {
            query,
            trace,
            projection: projection
                .ok_or_else(|| integrity("compact query projection proof was not collected"))?,
        },
    ))
}

fn selected_graph_counts(
    currents: &[PendingCurrent],
    closures: &[PendingClosure],
    live: &BTreeSet<u32>,
) -> RusticolResult<[u32; 3]> {
    let contributions = live.iter().try_fold(0usize, |total, id| {
        let count = currents
            .get(*id as usize)
            .ok_or_else(|| integrity("selected graph current is absent"))?
            .contributions
            .values()
            .filter(|factor| !factor.is_zero())
            .count();
        total
            .checked_add(count)
            .ok_or_else(|| invalid("selected graph contribution count exceeds usize"))
    })?;
    Ok([
        checked_u32(live.len(), "selected graph current count")?,
        checked_u32(contributions, "selected graph contribution count")?,
        checked_u32(closures.len(), "selected graph closure count")?,
    ])
}

/// Build one selected LC structural trace without constructing any global
/// recurrence/process projection.
#[allow(dead_code)] // Compatibility entry point for the query-local probe path.
pub(crate) fn build_selected_lc_trace(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
) -> RusticolResult<OnTheFlyStructuralTraceV1> {
    let (grammar, coupling_policy) = prepare_on_the_fly_process_v1(templates, seed)?;
    let query = grammar.canonicalize_query(seed, query.clone())?;
    build_selected_lc_trace_impl(
        templates,
        seed,
        &grammar,
        coupling_policy.effective_limits(),
        &query,
        true,
        false,
        false,
    )?
    .map(|(trace, _)| trace)
    .ok_or_else(|| evaluation("selected query is an exact structural zero"))
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[allow(dead_code)] // Feature-only probe entry point.
pub(crate) fn build_selected_lc_trace_for_probe(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
    enable_projection: bool,
) -> RusticolResult<(OnTheFlyStructuralTraceV1, OnTheFlyProjectionProbeV1)> {
    let (grammar, coupling_policy) = prepare_on_the_fly_process_v1(templates, seed)?;
    let query = grammar.canonicalize_query(seed, query.clone())?;
    let (trace, probe) = build_selected_lc_trace_impl(
        templates,
        seed,
        &grammar,
        coupling_policy.effective_limits(),
        &query,
        enable_projection,
        false,
        true,
    )?
    .ok_or_else(|| evaluation("selected probe query is an exact structural zero"))?;
    Ok((
        trace,
        probe.ok_or_else(|| integrity("projection probe was not collected"))?,
    ))
}

#[allow(clippy::too_many_arguments)] // Explicit cold-build context and probe switches.
fn build_selected_lc_trace_impl(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    grammar: &PreparedOnTheFlyGrammarV1,
    coupling_limits: &[Option<u32>],
    query: &DecodedLcQueryV1,
    enable_projection: bool,
    enable_cyclic_trace_reflection: bool,
    collect_projection_probe: bool,
) -> RusticolResult<Option<(OnTheFlyStructuralTraceV1, Option<OnTheFlyProjectionProbeV1>)>> {
    if query.seed_digest != seed.semantic_digest() {
        return Err(integrity(
            "decoded query belongs to a different compact seed",
        ));
    }
    let mut colors = DynamicLCColorStateInterner::default();
    let mut currents = Vec::new();
    let mut current_ids = BTreeMap::new();
    let enable_cyclic_trace_reflection = enable_cyclic_trace_reflection
        && grammar
            .closure_anchor_policy
            .certifies_cyclic_trace_reflection(seed, query);
    insert_selected_sources(
        grammar,
        seed,
        coupling_limits,
        query,
        enable_cyclic_trace_reflection,
        &mut colors,
        &mut currents,
        &mut current_ids,
    )?;
    build_forward_currents(
        templates,
        &grammar.transitions,
        &grammar.transition_reflections,
        &grammar.contact_orbits,
        enable_cyclic_trace_reflection,
        seed,
        &grammar.fermion_ordering,
        query.closure_anchor_slot,
        coupling_limits,
        &grammar.propagators,
        &mut colors,
        &mut currents,
        &mut current_ids,
    )?;
    let constructed_contribution_count = currents
        .iter()
        .map(|current| current.contributions.len())
        .try_fold(0usize, |total, count| {
            total
                .checked_add(count)
                .ok_or_else(|| invalid("constructed contribution count exceeds usize"))
        })?;
    let Some(selected_closures) = build_selected_closures(
        templates,
        &grammar.closures,
        seed,
        &grammar.fermion_ordering,
        query,
        enable_cyclic_trace_reflection,
        &colors,
        &currents,
    )?
    else {
        return Ok(None);
    };
    let live = live_current_ids(&currents, &selected_closures)?;
    let pre_counts = collect_projection_probe
        .then(|| selected_graph_counts(&currents, &selected_closures, &live))
        .transpose()?;
    let projected = enable_projection
        .then(|| project_query_local_color_aliases(&currents, &selected_closures, &live))
        .transpose()?
        .flatten();
    let projection_applied = projected.is_some();
    let (currents, selected_closures, live) = match projected {
        Some(projected) => {
            let live = (0..projected.currents.len())
                .map(|index| checked_u32(index, "projected query-local current ID"))
                .collect::<RusticolResult<BTreeSet<_>>>()?;
            (projected.currents, projected.closures, live)
        }
        None => (currents, selected_closures, live),
    };
    let post_counts = collect_projection_probe
        .then(|| selected_graph_counts(&currents, &selected_closures, &live))
        .transpose()?;
    let pairing_owners = resolve_projected_pairing_owners(seed, &selected_closures)?;
    let trace = lower_trace(
        templates,
        grammar,
        seed,
        query,
        &colors,
        &currents,
        &selected_closures,
        pairing_owners,
        &live,
        constructed_contribution_count,
    )?;
    Ok(Some((
        trace,
        pre_counts
            .zip(post_counts)
            .map(|(pre, post)| OnTheFlyProjectionProbeV1 {
                enabled: enable_projection,
                applied: projection_applied,
                pre,
                post,
            }),
    )))
}

#[cfg(test)]
mod structural_zero_tests {
    use super::*;
    use crate::recurrence::validated_template_fixture;

    #[test]
    fn indexed_query_results_keep_order_and_report_the_lowest_error() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let ordered = pool.install(|| {
            (0..32_u32)
                .into_par_iter()
                .map(Ok::<_, RusticolError>)
                .collect::<Vec<_>>()
        });
        assert_eq!(
            resolve_indexed_query_results(ordered).unwrap(),
            (0..32_u32).collect::<Vec<_>>()
        );

        let failures = pool.install(|| {
            (0..32_u32)
                .into_par_iter()
                .map(|index| match index {
                    5 | 19 => Err(invalid(format!("query index {index}"))),
                    _ => Ok(index),
                })
                .collect::<Vec<_>>()
        });
        let error = resolve_indexed_query_results(failures).unwrap_err();
        assert!(error.to_string().contains("query index 5"));
    }

    #[test]
    fn selected_closure_domain_fails_closed_when_missing_or_inconsistent() {
        let templates = validated_template_fixture();
        let summary = templates.summary();
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            SemanticDigest::new([0x7d; 32]).unwrap(),
        )
        .unwrap();
        let query =
            DecodedLcQueryV1::new(&seed, vec![0, 1], &[0, 0], OnTheFlyLcSelectorV1::Singlet)
                .unwrap();
        let catalog = TemplateCatalog::new(templates.input()).unwrap();
        let grammar = prepare_on_the_fly_grammar_v1(&templates, &catalog, &seed).unwrap();
        let mut colors = DynamicLCColorStateInterner::default();
        let mut currents = Vec::new();
        let mut current_ids = BTreeMap::new();
        insert_selected_sources(
            &grammar,
            &seed,
            seed.explicit_coupling_limits(),
            &query,
            false,
            &mut colors,
            &mut currents,
            &mut current_ids,
        )
        .unwrap();

        let absent = BTreeMap::new();
        let error = validate_prepared_closure_domain(&templates, &catalog, &absent).unwrap_err();
        assert!(error.to_string().contains("prepared closure domain"));

        let mut inconsistent = prepared_closures(&templates, &catalog).unwrap();
        let rows = inconsistent.values_mut().next().unwrap();
        rows[0].input_states = [4, 9];
        let error =
            validate_prepared_closure_domain(&templates, &catalog, &inconsistent).unwrap_err();
        assert!(error.to_string().contains("parent-state pair"));

        currents.clear();
        let complete = prepared_closures(&templates, &catalog).unwrap();
        let error = build_selected_closures(
            &templates,
            &complete,
            &seed,
            &grammar.fermion_ordering,
            &query,
            false,
            &colors,
            &currents,
        )
        .unwrap_err();
        assert!(error.to_string().contains("closure anchor"));
    }
}
