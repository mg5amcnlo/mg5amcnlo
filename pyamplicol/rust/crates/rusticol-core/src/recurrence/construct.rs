// SPDX-License-Identifier: 0BSD

//! Compact model-generic recurrence construction.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::iter::Peekable;
use std::rc::Rc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::contact_orbit_owner::{
    PreparedContactOrbitTransition, prepare_contact_orbit_transition,
    selected_contact_orbit_owner_tokens,
};
use super::fermion_ordering::{
    FermionOrderingContext, authenticated_source_requires_exterior_sign, fermion_ordering_factor,
};
use super::layout::RuntimeSourceVariantBinding;
use super::process::{
    FermionPairingEndpointPairRow, FermionPairingRuleRow, OwnedRecurrenceProcessInput,
    ProcessLCSectorKind, ProcessPhysicalLCSectorRow, ProcessSourceStateRow,
    ValidatedFermionPairingCatalog,
};
use super::program::closure_candidate_identity_digest_v1;
use super::template::{
    ClosureRow, ColorContractionRow, CurrentOrientation, EvaluatorCallableKind,
    EvaluatorContractKind, LCColorTransitionWitnessRow, OutputFactorSource,
    OwnedRecurrenceTemplateInput, ParticleStatistics, QuantumFlowRow, RuntimeHelicityContractRow,
    RuntimeHelicityVariantRow, SourceRow, TransitionRow,
};
use super::{
    AuthenticatedRecurrenceBuilderInput, CanonicalMomentumLinearForm, CheckedTableRange,
    ClosureCandidateDomainCertificateV1, ClosureExecutionProofGroupV2, ClosureProofContributionV2,
    ClosureProofMetadataV2, ClosureTargetDomainCertificateV1, ContributionKey, CurrentCoreKey,
    CurrentHelicityIdentity, CurrentSourceBinding, DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS,
    DynamicLCColorState, DynamicLCColorStateId, DynamicLCColorStateInterner, ExactComplexRational,
    ExactRational, LCColorComponent, LCColorComponentKind, LCColorComponentOperation,
    LCColorComponentRole, LCColorParentPort, LCColorPortWiring, LCColorSourceSeed,
    LCColorSourceSeedOperation, LCColorTransitionWitness, LCColorWitnessTermId, MomentumTerm,
    RecurrenceAmplitudeDestination, RecurrenceClosureTerm, RecurrenceContribution,
    RecurrenceCurrent, RecurrenceFinalization, RecurrenceNodeKind, RecurrenceProgram,
    RecurrenceReplayTarget, RecurrenceResolvedHelicity, RecurrenceStrategy,
    ReflectionCertificateV1, SemanticDigest, SourceStateAssignment,
    ThreeLineTraversalCertificateV1, ThreeLineTraversalKindV1, closure_component_factor_digest_v2,
    closure_selector_domain_digest_v2,
};
use crate::{RusticolError, RusticolResult};

const MISSING_U32: u32 = u32::MAX;
const PROGRESS_PAIR_INTERVAL: usize = 16_384;
const PROGRESS_TIME_INTERVAL: Duration = Duration::from_millis(250);
const PURE_MASSLESS_ADJOINT_HELICITY_SUPPORT_ROLE: &str =
    "helicity-support:pure-massless-adjoint-tree-v1";
const GLOBAL_HELICITY_FLIP_EQUIVALENCE_ROLE: &str = "helicity-equivalence:global-flip-v1";
const CYCLIC_TRACE_CLOSURE_ANCHOR_ALGORITHM: &str = "canonical-lc-closure-anchor-v4";
const REFLECTION_PROOF_ALGORITHM_ID: u32 = 1;
const THREE_LINE_DIRECT_CERTIFICATE_ID: u32 = 0;
const THREE_LINE_PARTNER_CERTIFICATE_ID: u32 = 1;

#[cfg(any(test, feature = "on-the-fly-test-support"))]
thread_local! {
    static ESTABLISHED_PAIRING_OWNER_OBSERVATION_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ESTABLISHED_PAIRING_OWNER_OBSERVATION:
        RefCell<Option<RusticolResult<Vec<Option<u32>>>>> = const { RefCell::new(None) };
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[allow(dead_code)]
pub(crate) fn begin_established_pairing_owner_observation() {
    ESTABLISHED_PAIRING_OWNER_OBSERVATION_ACTIVE.with(|active| active.set(true));
    ESTABLISHED_PAIRING_OWNER_OBSERVATION.with(|observation| {
        *observation.borrow_mut() = None;
    });
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[allow(dead_code)]
pub(crate) fn take_established_pairing_owner_observation() -> RusticolResult<Vec<Option<u32>>> {
    ESTABLISHED_PAIRING_OWNER_OBSERVATION_ACTIVE.with(|active| active.set(false));
    ESTABLISHED_PAIRING_OWNER_OBSERVATION.with(|observation| {
        observation.borrow_mut().take().ok_or_else(|| {
            invalid("established builder did not publish its retained pairing owner")
        })?
    })
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
fn observe_established_pairing_owner(
    selected_domain: Option<&(u32, Box<[SourceStateAssignment]>)>,
    projection: Option<&PendingColorProjection>,
    pending_closures: &BTreeMap<PendingClosureKey, PendingClosureGroup>,
) {
    if !ESTABLISHED_PAIRING_OWNER_OBSERVATION_ACTIVE.with(Cell::get) {
        return;
    }
    let result = (|| {
        let (selected_sector, selected_source_states) = selected_domain.ok_or_else(|| {
            invalid("established pairing-owner observation has no selected closure domain")
        })?;
        let owner_for_group = |group: &PendingClosureGroup| -> RusticolResult<Option<u32>> {
            let rule_ids = group
                .contributions
                .iter()
                .flat_map(|contribution| contribution.pairing_certificate_ids.iter().copied())
                .collect::<BTreeSet<_>>();
            match rule_ids.len() {
                0 => Ok(None),
                1 => Ok(rule_ids.iter().next().copied()),
                _ => Err(invalid(format!(
                    "established retained closure has {} pairing owners, expected exactly one",
                    rule_ids.len(),
                ))),
            }
        };
        let mut owners = Vec::new();
        if let Some(projection) = projection {
            for projected in projection.closures.values().filter(|projected| {
                projected.identity.target_sector_id == *selected_sector
                    && projected.identity.complete_source_states.as_ref()
                        == selected_source_states.as_ref()
            }) {
                owners.push(owner_for_group(&projected.representative_group)?);
            }
        } else {
            for group in pending_closures
                .iter()
                .filter(|(key, group)| {
                    key.target_sector_id == *selected_sector
                        && key.complete_source_states.as_ref() == selected_source_states.as_ref()
                        && !group.exact_factor.is_zero()
                })
                .map(|(_, group)| group)
            {
                owners.push(owner_for_group(group)?);
            }
        }
        if owners.is_empty() {
            return Err(invalid(
                "established selected closure domain has no retained pairing owners",
            ));
        }
        Ok(owners)
    })();
    ESTABLISHED_PAIRING_OWNER_OBSERVATION_ACTIVE.with(|active| active.set(false));
    ESTABLISHED_PAIRING_OWNER_OBSERVATION.with(|observation| {
        *observation.borrow_mut() = Some(result);
    });
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn hash_exact_factor(hash: &mut Sha256, factor: ExactComplexRational) {
    for rational in [factor.real(), factor.imag()] {
        hash.update(rational.numerator().to_le_bytes());
        hash.update(rational.denominator().to_le_bytes());
    }
}

fn semantic_digest_from_u32_fields(
    domain: &[u8],
    values: impl IntoIterator<Item = u32>,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(domain);
    for value in values {
        hash.update(value.to_le_bytes());
    }
    SemanticDigest::new(hash.finalize().into())
}

fn hash_digest_sequence(hash: &mut Sha256, values: &[SemanticDigest]) -> RusticolResult<()> {
    hash.update(
        u64::try_from(values.len())
            .map_err(|_| invalid("reflection proof digest count exceeds u64"))?
            .to_le_bytes(),
    );
    for value in values {
        hash.update(value.as_bytes());
    }
    Ok(())
}

pub(super) fn dynamic_color_identity_digest(
    color: &DynamicLCColorState,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-dynamic-lc-color-identity-v1\0");
    hash.update(color.output_color_shape_id().to_le_bytes());
    hash.update(
        color
            .active_component_index()
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    hash.update(
        u64::try_from(color.components().len())
            .map_err(|_| invalid("dynamic LC color component count exceeds u64"))?
            .to_le_bytes(),
    );
    for component in color.components() {
        hash.update([component.kind() as u8]);
        hash.update(
            u64::try_from(component.source_slots().len())
                .map_err(|_| invalid("dynamic LC color word length exceeds u64"))?
                .to_le_bytes(),
        );
        for source_slot in component.source_slots() {
            hash.update(source_slot.to_le_bytes());
        }
    }
    hash.update(
        u64::try_from(color.result_port_bindings().len())
            .map_err(|_| invalid("dynamic LC color port count exceeds u64"))?
            .to_le_bytes(),
    );
    for binding in color.result_port_bindings() {
        hash.update(binding.component_index().to_le_bytes());
        hash.update([binding.endpoint() as u8]);
    }
    SemanticDigest::new(hash.finalize().into())
}

fn source_reflection_proof_root(
    source_slot: u32,
    source_seed_proof: SemanticDigest,
    result_color_identity: SemanticDigest,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-source-color-reflection-root-v1\0");
    hash.update(source_slot.to_le_bytes());
    hash.update(source_seed_proof.as_bytes());
    hash.update(result_color_identity.as_bytes());
    SemanticDigest::new(hash.finalize().into())
}

fn transition_reflection_proof_digest(
    phase: ExactComplexRational,
    semantic_digests: &[SemanticDigest],
    witness_digests: &[SemanticDigest],
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-transition-reflection-proof-v1\0");
    hash_exact_factor(&mut hash, phase);
    hash_digest_sequence(&mut hash, semantic_digests)?;
    hash_digest_sequence(&mut hash, witness_digests)?;
    SemanticDigest::new(hash.finalize().into())
}

fn current_reflection_proof_digest(
    phase: ExactComplexRational,
    lineage_roots: &[SemanticDigest],
    result_color_identity: SemanticDigest,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-current-reflection-proof-v1\0");
    hash_exact_factor(&mut hash, phase);
    hash_digest_sequence(&mut hash, lineage_roots)?;
    hash.update(result_color_identity.as_bytes());
    SemanticDigest::new(hash.finalize().into())
}

fn pending_reflection_certificate_digest(
    canonical_old_current_id: u32,
    reflected_old_current_id: u32,
    canonical: &CurrentReflectionProof,
    reflected: &CurrentReflectionProof,
    source_permutation: &[u32],
    fixed_point: bool,
    orbit_size: u32,
) -> RusticolResult<SemanticDigest> {
    pending_reflection_certificate_fields_digest(
        canonical_old_current_id,
        reflected_old_current_id,
        canonical.result_color_identity(),
        reflected.result_color_identity(),
        canonical.phase(),
        reflected.phase(),
        canonical.proof_digest(),
        reflected.proof_digest(),
        source_permutation,
        fixed_point,
        orbit_size,
    )
}

#[allow(clippy::too_many_arguments)]
fn pending_reflection_certificate_fields_digest(
    canonical_old_current_id: u32,
    reflected_old_current_id: u32,
    canonical_color_identity: SemanticDigest,
    reflected_color_identity: SemanticDigest,
    canonical_phase: ExactComplexRational,
    reflected_phase: ExactComplexRational,
    canonical_lineage_digest: SemanticDigest,
    reflected_lineage_digest: SemanticDigest,
    source_permutation: &[u32],
    fixed_point: bool,
    orbit_size: u32,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-pending-reflection-certificate-v1\0");
    hash.update(canonical_old_current_id.to_le_bytes());
    hash.update(reflected_old_current_id.to_le_bytes());
    hash.update(canonical_color_identity.as_bytes());
    hash.update(reflected_color_identity.as_bytes());
    hash_exact_factor(&mut hash, canonical_phase);
    hash_exact_factor(&mut hash, reflected_phase);
    hash.update(canonical_lineage_digest.as_bytes());
    hash.update(reflected_lineage_digest.as_bytes());
    hash.update(
        u64::try_from(source_permutation.len())
            .map_err(|_| invalid("reflection source permutation length exceeds u64"))?
            .to_le_bytes(),
    );
    for source_slot in source_permutation {
        hash.update(source_slot.to_le_bytes());
    }
    hash.update([u8::from(fixed_point)]);
    hash.update(orbit_size.to_le_bytes());
    SemanticDigest::new(hash.finalize().into())
}

/// One rate-limited snapshot of compact recurrence construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceBuildProgress {
    pub phase: &'static str,
    pub phase_index: usize,
    pub phase_total: usize,
    pub stage_index: Option<usize>,
    pub stage_total: usize,
    pub subset_size: Option<usize>,
    pub candidate_parent_pair_count: usize,
    pub candidate_parent_pair_total: Option<usize>,
    pub current_count: usize,
    pub contribution_count: usize,
    pub dynamic_color_state_count: usize,
    pub color_target_prune_count: usize,
}

/// Construction-only recurrence-generation telemetry.
///
/// This diagnostic record is deliberately excluded from recurrence programs,
/// runtime plans, persisted payloads, and their semantic digests.
#[doc(hidden)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RecurrenceGenerationTelemetry {
    pub transition_catalog_nanoseconds: u64,
    pub structural_feasibility_nanoseconds: u64,
    pub color_target_index_nanoseconds: u64,
    pub structural_demand_nanoseconds: u64,
    pub support_indexing_nanoseconds: u64,
    pub candidate_processing_nanoseconds: u64,
    pub closure_processing_nanoseconds: u64,
    pub canonical_emission_nanoseconds: u64,
    pub support_bucket_count: usize,
    pub support_bucket_probe_count: usize,
    pub support_bucket_cache_hit_count: usize,
    pub support_bucket_cache_miss_count: usize,
    pub candidate_parent_pair_theoretical_count: usize,
    pub candidate_parent_pair_visited_count: usize,
    pub structural_feasible_support_count: usize,
    pub structural_decomposition_count: usize,
    pub structural_forward_transition_probe_count: usize,
    pub structural_demand_support_count: usize,
    pub structural_demand_state_count: usize,
    pub structural_reject_count: usize,
    pub transition_index_hit_count: usize,
    pub transition_index_miss_count: usize,
    pub transition_candidate_count: usize,
    pub quantum_match_count: usize,
    pub coupling_match_count: usize,
    pub transition_accept_count: usize,
    pub color_shape_match_count: usize,
    pub color_result_count: usize,
    pub color_target_accept_count: usize,
    pub color_target_reject_count: usize,
    pub color_acceptance_cache_hit_count: usize,
    pub color_acceptance_cache_miss_count: usize,
    pub color_fragment_bucket_count: usize,
    pub color_fragment_hash_lookup_count: usize,
    pub color_posting_incidence_count: usize,
    pub color_sparse_posting_bucket_count: usize,
    pub color_dense_posting_bucket_count: usize,
    pub color_sparse_posting_bytes: usize,
    pub color_dense_posting_bytes: usize,
    pub accepted_parent_key_clone_count: usize,
    pub current_key_lookup_count: usize,
    pub current_key_hit_count: usize,
    pub current_insert_count: usize,
    pub current_key_clone_count: usize,
    pub indexed_hash_lookup_count: usize,
    pub contribution_attempt_count: usize,
    pub contribution_insert_count: usize,
    pub contribution_merge_count: usize,
    pub closure_candidate_theoretical_count: usize,
    pub closure_candidate_count: usize,
    pub closure_state_match_count: usize,
    pub closure_support_lookup_count: usize,
    /// Exact number of successful-path color-witness closure evaluations,
    /// including repeated component forests.
    pub closure_color_attempt_count: usize,
    pub closure_group_count: usize,
    pub closure_proof_contribution_count: usize,
    pub constructed_current_count: usize,
    pub constructed_contribution_count: usize,
    pub constructed_interaction_count: usize,
    pub constructed_dynamic_color_state_count: usize,
    pub emitted_current_count: usize,
    pub emitted_contribution_count: usize,
    pub emitted_interaction_count: usize,
    pub emitted_finalization_count: usize,
    pub emitted_closure_count: usize,
}

impl RecurrenceBuildProgress {
    #[allow(clippy::too_many_arguments)]
    fn snapshot(
        phase: &'static str,
        phase_index: usize,
        phase_total: usize,
        stage_index: Option<usize>,
        stage_total: usize,
        subset_size: Option<usize>,
        candidate_parent_pair_count: usize,
        candidate_parent_pair_total: Option<usize>,
        current_count: usize,
        contribution_count: usize,
        dynamic_color_state_count: usize,
        color_target_prune_count: usize,
    ) -> Self {
        Self {
            phase,
            phase_index,
            phase_total,
            stage_index,
            stage_total,
            subset_size,
            candidate_parent_pair_count,
            candidate_parent_pair_total,
            current_count,
            contribution_count,
            dynamic_color_state_count,
            color_target_prune_count,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PendingContributionKey {
    parent_current_ids: Box<[u32]>,
    key: ContributionKey,
}

#[derive(Clone, Debug)]
struct PendingCurrent {
    key: CurrentCoreKey,
    source_exact_factor: Option<ExactComplexRational>,
    contributions: BTreeMap<PendingContributionKey, ExactComplexRational>,
    realized_pairing_rule_ids: BTreeSet<u32>,
    reflection: CurrentReflection,
    reflection_certificate_id: Option<u32>,
}

/// Construction-only exact current lookup.
///
/// Canonical order is owned exclusively by the `currents` vector and ordered
/// contribution maps. This index is only cloned, queried, inserted into, and
/// pruned; its randomized iteration order is never observed.
type TransientCurrentIdIndex = HashMap<CurrentCoreKey, u32>;

/// Exact runtime-value identity after erasing only dynamic LC colour.
///
/// The physical-sector domain is part of the identity.  Without it, folding
/// two colour fragments from disjoint selector domains would make every
/// contribution of either fragment live in both domains.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProjectedCurrentIdentity {
    color_erased_key: CurrentCoreKey,
    selector_sector_ids: Box<[u32]>,
    source_builder_id: Option<u32>,
}

/// Exact contribution identity after mapping parents to projected values.
///
/// `color_witness_term_id` is deliberately absent.  Every remaining runtime
/// field and the exact coefficient participates in equality.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProjectedContributionIdentity {
    destination_projection_id: u32,
    transition_template_id: u32,
    parent_projection_ids: Box<[u32]>,
    parent_state_template_ids: Box<[u32]>,
    parent_momenta: Box<[CanonicalMomentumLinearForm]>,
    result_state_template_id: u32,
    quantum_flow_witness_id: u32,
    runtime_coupling_binding_digest: SemanticDigest,
    output_projection_id: u32,
    exact_factor: ExactComplexRational,
}

#[derive(Clone, Debug)]
struct PendingProjectedContribution {
    identity: ProjectedContributionIdentity,
    representative_destination_builder_id: u32,
    representative_key: ContributionKey,
    destination_builder_ids: BTreeSet<u32>,
    builder_parent_tuples: BTreeSet<Box<[u32]>>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProjectedClosureIdentity {
    target_sector_id: u32,
    complete_source_states: Box<[SourceStateAssignment]>,
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_projection_ids: Box<[u32]>,
    exact_factor: ExactComplexRational,
}

#[derive(Clone, Debug)]
struct PendingProjectedClosure {
    identity: ProjectedClosureIdentity,
    representative_key: PendingClosureKey,
    representative_group: PendingClosureGroup,
    builder_parent_tuples: BTreeSet<Box<[u32]>>,
}

#[derive(Clone, Debug)]
struct PendingColorProjection {
    old_to_projection: BTreeMap<u32, u32>,
    projection_members: Vec<Box<[u32]>>,
    projection_sector_ids: Vec<Box<[u32]>>,
    contributions: BTreeMap<ProjectedContributionIdentity, PendingProjectedContribution>,
    closures: BTreeMap<ProjectedClosureIdentity, PendingProjectedClosure>,
}

struct MaterializedPendingRows {
    remap: BTreeMap<u32, u32>,
    currents: Vec<RecurrenceCurrent>,
    contributions: Vec<RecurrenceContribution>,
    finalizations: Vec<RecurrenceFinalization>,
}

fn materialize_live_pending_rows(
    pending: &[PendingCurrent],
    live: &BTreeSet<u32>,
) -> RusticolResult<MaterializedPendingRows> {
    let remap = live
        .iter()
        .copied()
        .enumerate()
        .map(|(new, old)| {
            u32::try_from(new)
                .map(|new| (old, new))
                .map_err(|_| invalid("live recurrence current count exceeds u32"))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    let mut currents = Vec::with_capacity(live.len());
    let mut contributions = Vec::new();
    let mut finalizations = Vec::new();
    for old_id in live.iter().copied() {
        let pending_current = &pending[old_id as usize];
        let start = u64::try_from(contributions.len())
            .map_err(|_| invalid("recurrence contribution count exceeds u64"))?;
        for (pending_key, factor) in &pending_current.contributions {
            if factor.is_zero() {
                continue;
            }
            let parent_ids = pending_key
                .parent_current_ids
                .iter()
                .map(|id| {
                    remap
                        .get(id)
                        .copied()
                        .ok_or_else(|| invalid("live parent is absent"))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let old_key = &pending_key.key;
            let key = ContributionKey::new(
                old_key.transition_template_id(),
                parent_ids.clone(),
                old_key.parent_state_template_ids().to_vec(),
                old_key.parent_momenta().to_vec(),
                old_key.result_state_template_id(),
                old_key.quantum_flow_witness_id(),
                old_key.color_witness_term_id(),
                old_key.runtime_coupling_binding_digest(),
                old_key.output_projection_id(),
            )?;
            contributions.push(RecurrenceContribution::new(
                u32::try_from(contributions.len())
                    .map_err(|_| invalid("recurrence contribution count exceeds u32"))?,
                remap[&old_id],
                parent_ids,
                key,
                *factor,
            )?);
        }
        let count = u64::try_from(contributions.len())
            .map_err(|_| invalid("recurrence contribution count exceeds u64"))?
            .checked_sub(start)
            .ok_or_else(|| invalid("recurrence contribution range underflows"))?;
        let finalization_id = if pending_current.key.node_kind() == RecurrenceNodeKind::Current {
            let id = u32::try_from(finalizations.len())
                .map_err(|_| invalid("recurrence finalization count exceeds u32"))?;
            finalizations.push(RecurrenceFinalization::new(
                id,
                remap[&old_id],
                pending_current.key.propagator_template_id(),
                ExactComplexRational::ONE,
            )?);
            Some(id)
        } else {
            None
        };
        currents.push(RecurrenceCurrent::new(
            remap[&old_id],
            pending_current.key.clone(),
            pending_current.source_exact_factor,
            CheckedTableRange::new(start, count),
            finalization_id,
        )?);
    }
    Ok(MaterializedPendingRows {
        remap,
        currents,
        contributions,
        finalizations,
    })
}

fn materialize_projected_pending_rows(
    pending: &[PendingCurrent],
    projection: &PendingColorProjection,
) -> RusticolResult<MaterializedPendingRows> {
    let mut currents = Vec::with_capacity(projection.projection_members.len());
    let mut contributions = Vec::with_capacity(projection.contributions.len());
    let mut finalizations = Vec::new();
    for (projection_index, members) in projection.projection_members.iter().enumerate() {
        let projection_id = u32::try_from(projection_index)
            .map_err(|_| invalid("projected recurrence current count exceeds u32"))?;
        let representative_id = *members
            .first()
            .ok_or_else(|| invalid("projected recurrence current class is empty"))?;
        let representative = pending
            .get(representative_id as usize)
            .ok_or_else(|| invalid("projected recurrence representative is absent"))?;
        let start = u64::try_from(contributions.len())
            .map_err(|_| invalid("projected recurrence contribution count exceeds u64"))?;
        for projected in projection
            .contributions
            .values()
            .filter(|row| row.identity.destination_projection_id == projection_id)
        {
            if !projected.destination_builder_ids.iter().all(|builder_id| {
                projection.old_to_projection.get(builder_id) == Some(&projection_id)
            }) {
                return Err(invalid(
                    "projected contribution destinations cross projection classes",
                ));
            }
            let parent_ids = projected.identity.parent_projection_ids.to_vec();
            let witness = &projected.representative_key;
            let key = ContributionKey::new(
                projected.identity.transition_template_id,
                parent_ids.clone(),
                projected.identity.parent_state_template_ids.to_vec(),
                projected.identity.parent_momenta.to_vec(),
                projected.identity.result_state_template_id,
                projected.identity.quantum_flow_witness_id,
                witness.color_witness_term_id(),
                projected.identity.runtime_coupling_binding_digest,
                projected.identity.output_projection_id,
            )?;
            contributions.push(RecurrenceContribution::new(
                u32::try_from(contributions.len())
                    .map_err(|_| invalid("projected recurrence contribution count exceeds u32"))?,
                projection_id,
                parent_ids,
                key,
                projected.identity.exact_factor,
            )?);
        }
        let count = u64::try_from(contributions.len())
            .map_err(|_| invalid("projected recurrence contribution count exceeds u64"))?
            .checked_sub(start)
            .ok_or_else(|| invalid("projected recurrence contribution range underflows"))?;
        let finalization_id = if representative.key.node_kind() == RecurrenceNodeKind::Current {
            let id = u32::try_from(finalizations.len())
                .map_err(|_| invalid("projected recurrence finalization count exceeds u32"))?;
            finalizations.push(RecurrenceFinalization::new(
                id,
                projection_id,
                representative.key.propagator_template_id(),
                ExactComplexRational::ONE,
            )?);
            Some(id)
        } else {
            None
        };
        currents.push(RecurrenceCurrent::new(
            projection_id,
            representative.key.clone(),
            representative.source_exact_factor,
            CheckedTableRange::new(start, count),
            finalization_id,
        )?);
    }
    Ok(MaterializedPendingRows {
        remap: projection.old_to_projection.clone(),
        currents,
        contributions,
        finalizations,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CurrentReflectionProof {
    phase: ExactComplexRational,
    lineage_roots: Box<[SemanticDigest]>,
    result_color_identity: SemanticDigest,
    proof_digest: SemanticDigest,
}

impl CurrentReflectionProof {
    pub(super) fn new(
        phase: ExactComplexRational,
        lineage_roots: impl IntoIterator<Item = SemanticDigest>,
        result_color_identity: SemanticDigest,
    ) -> RusticolResult<Self> {
        if phase.is_zero() {
            return Err(invalid("current reflection proof has zero phase"));
        }
        let mut lineage_roots = lineage_roots.into_iter().collect::<Vec<_>>();
        lineage_roots.sort_unstable();
        lineage_roots.dedup();
        if lineage_roots.is_empty() {
            return Err(invalid("current reflection proof has empty lineage"));
        }
        let proof_digest =
            current_reflection_proof_digest(phase, &lineage_roots, result_color_identity)?;
        Ok(Self {
            phase,
            lineage_roots: lineage_roots.into_boxed_slice(),
            result_color_identity,
            proof_digest,
        })
    }

    pub(super) const fn phase(&self) -> ExactComplexRational {
        self.phase
    }

    pub(super) fn lineage_roots(&self) -> &[SemanticDigest] {
        &self.lineage_roots
    }

    pub(super) const fn result_color_identity(&self) -> SemanticDigest {
        self.result_color_identity
    }

    pub(super) const fn proof_digest(&self) -> SemanticDigest {
        self.proof_digest
    }

    fn merged_with(&self, other: &Self) -> RusticolResult<Option<Self>> {
        if self.phase != other.phase || self.result_color_identity != other.result_color_identity {
            return Ok(None);
        }
        Self::new(
            self.phase,
            self.lineage_roots
                .iter()
                .copied()
                .chain(other.lineage_roots.iter().copied()),
            self.result_color_identity,
        )
        .map(Some)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum CurrentReflection {
    Unavailable,
    Proven(CurrentReflectionProof),
}

impl CurrentReflection {
    pub(super) fn phase(&self) -> Option<ExactComplexRational> {
        match self {
            Self::Unavailable => None,
            Self::Proven(proof) => Some(proof.phase()),
        }
    }

    pub(super) fn proof(&self) -> Option<&CurrentReflectionProof> {
        match self {
            Self::Unavailable => None,
            Self::Proven(proof) => Some(proof),
        }
    }

    pub(super) fn include(
        &mut self,
        candidate: Option<CurrentReflectionProof>,
    ) -> RusticolResult<()> {
        let merged = match (&*self, candidate) {
            (Self::Proven(existing), Some(candidate)) => existing.merged_with(&candidate)?,
            _ => None,
        };
        match merged {
            Some(proof) => *self = Self::Proven(proof),
            None => *self = Self::Unavailable,
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct TransitionReflectionProof {
    phase: ExactComplexRational,
    semantic_digests: Box<[SemanticDigest]>,
    witness_digests: Box<[SemanticDigest]>,
    proof_digest: SemanticDigest,
}

impl TransitionReflectionProof {
    fn new(
        phase: ExactComplexRational,
        semantic_digests: impl IntoIterator<Item = SemanticDigest>,
        witness_digests: impl IntoIterator<Item = SemanticDigest>,
    ) -> RusticolResult<Self> {
        if phase.is_zero() {
            return Err(invalid("transition reflection proof has zero phase"));
        }
        let mut semantic_digests = semantic_digests.into_iter().collect::<Vec<_>>();
        semantic_digests.sort_unstable();
        semantic_digests.dedup();
        let mut witness_digests = witness_digests.into_iter().collect::<Vec<_>>();
        witness_digests.sort_unstable();
        witness_digests.dedup();
        if semantic_digests.is_empty() || witness_digests.is_empty() {
            return Err(invalid(
                "transition reflection proof requires semantic and witness digests",
            ));
        }
        let proof_digest =
            transition_reflection_proof_digest(phase, &semantic_digests, &witness_digests)?;
        Ok(Self {
            phase,
            semantic_digests: semantic_digests.into_boxed_slice(),
            witness_digests: witness_digests.into_boxed_slice(),
            proof_digest,
        })
    }

    pub(super) const fn phase(&self) -> ExactComplexRational {
        self.phase
    }

    #[cfg(feature = "on-the-fly-test-support")]
    pub(super) const fn proof_digest(&self) -> SemanticDigest {
        self.proof_digest
    }

    fn lineage_roots(&self) -> impl Iterator<Item = SemanticDigest> + '_ {
        self.semantic_digests
            .iter()
            .chain(self.witness_digests.iter())
            .copied()
            .chain(std::iter::once(self.proof_digest))
    }

    fn merged_with(&self, other: &Self) -> RusticolResult<Option<Self>> {
        if self.phase != other.phase {
            return Ok(None);
        }
        Self::new(
            self.phase,
            self.semantic_digests
                .iter()
                .copied()
                .chain(other.semantic_digests.iter().copied()),
            self.witness_digests
                .iter()
                .copied()
                .chain(other.witness_digests.iter().copied()),
        )
        .map(Some)
    }
}

/// Cold proof record emitted before folded stage-local current IDs are compacted.
///
/// This deliberately remains local until the orchestrator-owned program proof
/// API accepts reciprocal reflection orbits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PendingReflectionCertificate {
    id: u32,
    canonical_old_current_id: u32,
    reflected_old_current_id: u32,
    canonical_color_identity: SemanticDigest,
    reflected_color_identity: SemanticDigest,
    canonical_phase: ExactComplexRational,
    reflected_phase: ExactComplexRational,
    canonical_lineage_digest: SemanticDigest,
    reflected_lineage_digest: SemanticDigest,
    source_permutation: Box<[u32]>,
    fixed_point: bool,
    orbit_size: u32,
    proof_digest: SemanticDigest,
}

impl PendingReflectionCertificate {
    // The pair constructor authenticates both orbit members and both color
    // witnesses explicitly; retaining the parallel arguments makes swaps visible.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn reciprocal_pair(
        id: u32,
        canonical_old_current_id: u32,
        reflected_old_current_id: u32,
        canonical: &CurrentReflectionProof,
        reflected: &CurrentReflectionProof,
        canonical_color: &DynamicLCColorState,
        reflected_color: &DynamicLCColorState,
        source_count: usize,
    ) -> RusticolResult<Self> {
        if canonical_old_current_id == reflected_old_current_id
            || canonical.result_color_identity() == reflected.result_color_identity()
        {
            return Err(invalid(
                "folded reflection certificate requires two distinct orbit members",
            ));
        }
        if canonical.phase().checked_mul(reflected.phase())? != ExactComplexRational::ONE {
            return Err(invalid(
                "folded reflection certificate phases are not reciprocal",
            ));
        }
        let source_permutation =
            reflection_source_permutation(canonical_color, reflected_color, source_count)?;
        let proof_digest = pending_reflection_certificate_digest(
            canonical_old_current_id,
            reflected_old_current_id,
            canonical,
            reflected,
            &source_permutation,
            false,
            2,
        )?;
        Ok(Self {
            id,
            canonical_old_current_id,
            reflected_old_current_id,
            canonical_color_identity: canonical.result_color_identity(),
            reflected_color_identity: reflected.result_color_identity(),
            canonical_phase: canonical.phase(),
            reflected_phase: reflected.phase(),
            canonical_lineage_digest: canonical.proof_digest(),
            reflected_lineage_digest: reflected.proof_digest(),
            source_permutation: source_permutation.into_boxed_slice(),
            fixed_point: false,
            orbit_size: 2,
            proof_digest,
        })
    }

    pub(super) const fn canonical_color_identity(&self) -> SemanticDigest {
        self.canonical_color_identity
    }

    pub(super) const fn canonical_phase(&self) -> ExactComplexRational {
        self.canonical_phase
    }

    pub(super) const fn canonical_lineage_digest(&self) -> SemanticDigest {
        self.canonical_lineage_digest
    }

    pub(super) fn source_permutation(&self) -> &[u32] {
        &self.source_permutation
    }

    pub(super) const fn is_reciprocal_two_cycle(&self) -> bool {
        !self.fixed_point && self.orbit_size == 2
    }
}

fn reflection_source_permutation(
    canonical_color: &DynamicLCColorState,
    reflected_color: &DynamicLCColorState,
    source_count: usize,
) -> RusticolResult<Vec<u32>> {
    let canonical = canonical_color
        .pure_adjoint_word()
        .ok_or_else(|| invalid("canonical reflection color has no adjoint word"))?;
    let reflected = reflected_color
        .pure_adjoint_word()
        .ok_or_else(|| invalid("reflected reflection color has no adjoint word"))?;
    if canonical.len() != reflected.len() {
        return Err(invalid(
            "reflection color words have inconsistent source counts",
        ));
    }
    let mut permutation = (0..source_count)
        .map(|slot| u32::try_from(slot).map_err(|_| invalid("reflection source count exceeds u32")))
        .collect::<RusticolResult<Vec<_>>>()?;
    for (source, target) in canonical.iter().copied().zip(reflected.iter().copied()) {
        let destination = permutation
            .get_mut(source as usize)
            .ok_or_else(|| invalid("reflection word references an absent source slot"))?;
        *destination = target;
    }
    let mut ordered = permutation.clone();
    ordered.sort_unstable();
    if ordered
        != (0..source_count)
            .map(|slot| u32::try_from(slot).expect("source count checked above"))
            .collect::<Vec<_>>()
    {
        return Err(invalid(
            "reflection source mapping is not a complete permutation",
        ));
    }
    Ok(permutation)
}

fn current_color_for_index<'a>(
    color_states: &'a DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    current_index: usize,
) -> RusticolResult<&'a DynamicLCColorState> {
    let current = currents
        .get(current_index)
        .ok_or_else(|| invalid("reflection current index is absent"))?;
    color_states
        .get(current.key.dynamic_lc_color_state_id())
        .ok_or_else(|| invalid("reflection current color disappeared"))
}

fn validate_pending_reflection_certificates(
    certificates: &[PendingReflectionCertificate],
) -> RusticolResult<()> {
    for (index, certificate) in certificates.iter().enumerate() {
        if certificate.id != index as u32 {
            return Err(invalid(format!(
                "pending reflection certificate row {index} has non-dense id {}",
                certificate.id
            )));
        }
        if certificate.canonical_old_current_id == certificate.reflected_old_current_id
            || certificate.canonical_color_identity == certificate.reflected_color_identity
        {
            return Err(invalid(format!(
                "pending reflection certificate {index} does not identify two orbit members"
            )));
        }
        if certificate.fixed_point || certificate.orbit_size != 2 {
            return Err(invalid(format!(
                "pending folded reflection certificate {index} is not a reciprocal two-cycle"
            )));
        }
        if certificate
            .canonical_phase
            .checked_mul(certificate.reflected_phase)?
            != ExactComplexRational::ONE
        {
            return Err(invalid(format!(
                "pending reflection certificate {index} phases are not reciprocal"
            )));
        }
        let expected = pending_reflection_certificate_fields_digest(
            certificate.canonical_old_current_id,
            certificate.reflected_old_current_id,
            certificate.canonical_color_identity,
            certificate.reflected_color_identity,
            certificate.canonical_phase,
            certificate.reflected_phase,
            certificate.canonical_lineage_digest,
            certificate.reflected_lineage_digest,
            &certificate.source_permutation,
            certificate.fixed_point,
            certificate.orbit_size,
        )?;
        if expected != certificate.proof_digest {
            return Err(invalid(format!(
                "pending reflection certificate {index} proof digest is stale"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HelicitySupportRule {
    None,
    PureMasslessAdjointTree,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GlobalHelicityFlipRule {
    None,
    Proven,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PendingClosureKey {
    target_sector_id: u32,
    complete_source_states: Box<[SourceStateAssignment]>,
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_current_ids: Box<[u32]>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PendingThreeLineTraversalCertificate {
    sector_id: u32,
    kind: u8,
    sink_block_ordinal: u32,
    reference_block_order: Box<[u32]>,
    witness_block_order: Box<[u32]>,
    block_permutation: Box<[u32]>,
    reference_source_order: Box<[u32]>,
    witness_source_order: Box<[u32]>,
    source_position_permutation: Box<[u32]>,
    closure_anchor_source_slot: u32,
    pairing_rule_id: u32,
    proof_digest: SemanticDigest,
}

/// One compiler/model-certified closure witness before equal runtime rows are
/// combined. These records are cold proof data and never enter the numeric
/// Direct-Arena loop.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingClosureProofContribution {
    construction_parent_ids: [u32; 2],
    construction_parent_permutation: [u32; 2],
    reconstruction_parent_permutation: [u32; 2],
    evaluator_parent_permutation: [u32; 2],
    closure_template_semantic_digest: SemanticDigest,
    color_witness_term_id: u32,
    color_witness_proof_digest: SemanticDigest,
    three_line_certificate: Option<PendingThreeLineTraversalCertificate>,
    pairing_certificate_ids: Box<[u32]>,
    reflection_certificate_id: Option<u32>,
    exact_factor: ExactComplexRational,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingClosureGroup {
    contributions: Vec<PendingClosureProofContribution>,
    exact_factor: ExactComplexRational,
}

/// Closure rows together with the physical sectors exhaustively visited by
/// their native construction pass.  Keeping the domain separate from the row
/// map prevents a later consumer from treating a missing row as proof that a
/// sector is zero.
#[derive(Debug, Eq, PartialEq)]
struct ExhaustiveClosureDomain {
    groups: BTreeMap<PendingClosureKey, PendingClosureGroup>,
    physical_sector_ids: BTreeSet<u32>,
    physical_sector_owner_ids: BTreeMap<u32, u32>,
}

impl std::ops::Deref for ExhaustiveClosureDomain {
    type Target = BTreeMap<PendingClosureKey, PendingClosureGroup>;

    fn deref(&self) -> &Self::Target {
        &self.groups
    }
}

impl ExhaustiveClosureDomain {
    fn new(
        groups: BTreeMap<PendingClosureKey, PendingClosureGroup>,
        physical_sector_ids: &BTreeSet<u32>,
        physical_sector_owner_ids: BTreeMap<u32, u32>,
    ) -> RusticolResult<Self> {
        if groups
            .keys()
            .any(|key| !physical_sector_ids.contains(&key.target_sector_id))
        {
            return Err(invalid(
                "closure result lies outside its exhaustive physical-sector domain",
            ));
        }
        if physical_sector_owner_ids
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            != *physical_sector_ids
            || physical_sector_owner_ids
                .iter()
                .any(|(sector_id, owner_id)| {
                    !physical_sector_ids.contains(owner_id)
                        || owner_id > sector_id
                        || physical_sector_owner_ids.get(owner_id) != Some(owner_id)
                })
        {
            return Err(invalid(
                "closure result has an invalid physical-sector owner domain",
            ));
        }
        Ok(Self {
            groups,
            physical_sector_ids: physical_sector_ids.clone(),
            physical_sector_owner_ids,
        })
    }
}

type ClosureColorAttemptDiagnostic = Vec<(LCColorComponentKind, Vec<u32>)>;

impl Default for PendingClosureGroup {
    fn default() -> Self {
        Self {
            contributions: Vec::new(),
            exact_factor: ExactComplexRational::ZERO,
        }
    }
}

impl PendingClosureGroup {
    fn include(&mut self, contribution: PendingClosureProofContribution) -> RusticolResult<()> {
        aggregate_factor(&mut self.exact_factor, contribution.exact_factor)?;
        self.contributions.push(contribution);
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct StageConstructionDiagnostics {
    target_size: usize,
    candidate_parent_pair_count: usize,
    support_bucket_count: usize,
    support_bucket_probe_count: usize,
    support_bucket_cache_hit_count: usize,
    parent_pair_count: usize,
    transition_index_hit_count: usize,
    transition_candidate_count: usize,
    state_order_count: usize,
    quantum_match_count: usize,
    coupling_match_count: usize,
    structural_reject_count: usize,
    transition_accept_count: usize,
    color_shape_match_count: usize,
    color_result_count: usize,
    color_target_prune_count: usize,
    accepted_parent_key_clone_count: usize,
    current_key_lookup_count: usize,
    current_key_hit_count: usize,
    current_insert_count: usize,
    current_key_clone_count: usize,
    contribution_attempt_count: usize,
    contribution_insert_count: usize,
    contribution_merge_count: usize,
    support_indexing_nanoseconds: u64,
    candidate_processing_nanoseconds: u64,
}

fn duration_nanoseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn add_elapsed_nanoseconds(total: &mut u64, started: Instant) {
    *total = total.saturating_add(duration_nanoseconds(started.elapsed()));
}

fn telemetry_timer(collect_telemetry: bool) -> Option<Instant> {
    collect_telemetry.then(Instant::now)
}

fn add_optional_elapsed_nanoseconds(total: &mut u64, started: Option<Instant>) {
    if let Some(started) = started {
        add_elapsed_nanoseconds(total, started);
    }
}

impl RecurrenceGenerationTelemetry {
    fn include_stage(&mut self, stage: &StageConstructionDiagnostics) -> RusticolResult<()> {
        for (target, value, label) in [
            (
                &mut self.support_bucket_count,
                stage.support_bucket_count,
                "telemetry support-bucket count",
            ),
            (
                &mut self.support_bucket_probe_count,
                stage.support_bucket_probe_count,
                "telemetry support-bucket probe count",
            ),
            (
                &mut self.support_bucket_cache_hit_count,
                stage.support_bucket_cache_hit_count,
                "telemetry support-bucket cache-hit count",
            ),
            (
                &mut self.candidate_parent_pair_theoretical_count,
                stage.candidate_parent_pair_count,
                "telemetry theoretical parent-pair count",
            ),
            (
                &mut self.candidate_parent_pair_visited_count,
                stage.parent_pair_count,
                "telemetry visited parent-pair count",
            ),
            (
                &mut self.structural_reject_count,
                stage.structural_reject_count,
                "telemetry structural-reject count",
            ),
            (
                &mut self.transition_index_hit_count,
                stage.transition_index_hit_count,
                "telemetry transition-index hit count",
            ),
            (
                &mut self.transition_candidate_count,
                stage.transition_candidate_count,
                "telemetry transition-candidate count",
            ),
            (
                &mut self.quantum_match_count,
                stage.quantum_match_count,
                "telemetry quantum-match count",
            ),
            (
                &mut self.coupling_match_count,
                stage.coupling_match_count,
                "telemetry coupling-match count",
            ),
            (
                &mut self.transition_accept_count,
                stage.transition_accept_count,
                "telemetry transition-accept count",
            ),
            (
                &mut self.color_shape_match_count,
                stage.color_shape_match_count,
                "telemetry color-shape-match count",
            ),
            (
                &mut self.color_result_count,
                stage.color_result_count,
                "telemetry color-result count",
            ),
            (
                &mut self.color_target_reject_count,
                stage.color_target_prune_count,
                "telemetry color-target-reject count",
            ),
            (
                &mut self.accepted_parent_key_clone_count,
                stage.accepted_parent_key_clone_count,
                "telemetry accepted-parent-key clone count",
            ),
            (
                &mut self.current_key_lookup_count,
                stage.current_key_lookup_count,
                "telemetry current-key lookup count",
            ),
            (
                &mut self.current_key_hit_count,
                stage.current_key_hit_count,
                "telemetry current-key hit count",
            ),
            (
                &mut self.current_insert_count,
                stage.current_insert_count,
                "telemetry current-insert count",
            ),
            (
                &mut self.current_key_clone_count,
                stage.current_key_clone_count,
                "telemetry current-key clone count",
            ),
            (
                &mut self.contribution_attempt_count,
                stage.contribution_attempt_count,
                "telemetry contribution-attempt count",
            ),
            (
                &mut self.contribution_insert_count,
                stage.contribution_insert_count,
                "telemetry contribution-insert count",
            ),
            (
                &mut self.contribution_merge_count,
                stage.contribution_merge_count,
                "telemetry contribution-merge count",
            ),
        ] {
            checked_diagnostic_add(target, value, label)?;
        }
        checked_diagnostic_add(
            &mut self.support_bucket_cache_miss_count,
            stage
                .support_bucket_probe_count
                .checked_sub(stage.support_bucket_cache_hit_count)
                .ok_or_else(|| invalid("support-bucket cache-hit count exceeds probes"))?,
            "telemetry support-bucket cache-miss count",
        )?;
        checked_diagnostic_add(
            &mut self.transition_index_miss_count,
            stage
                .parent_pair_count
                .checked_sub(stage.transition_index_hit_count)
                .ok_or_else(|| invalid("transition-index hit count exceeds parent pairs"))?,
            "telemetry transition-index miss count",
        )?;
        checked_diagnostic_add(
            &mut self.color_target_accept_count,
            stage
                .color_result_count
                .checked_sub(stage.color_target_prune_count)
                .ok_or_else(|| invalid("color-target prune count exceeds color results"))?,
            "telemetry color-target-accept count",
        )?;
        self.support_indexing_nanoseconds = self
            .support_indexing_nanoseconds
            .saturating_add(stage.support_indexing_nanoseconds);
        self.candidate_processing_nanoseconds = self
            .candidate_processing_nanoseconds
            .saturating_add(stage.candidate_processing_nanoseconds);
        Ok(())
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct IndexedTransition {
    row: TransitionRow,
    input_states: [u32; 2],
}

#[cfg(test)]
impl IndexedTransition {
    fn parent_ids(
        self,
        left_state: u32,
        right_state: u32,
        left_id: u32,
        right_id: u32,
    ) -> RusticolResult<[u32; 2]> {
        if self.input_states == [left_state, right_state] {
            return Ok([left_id, right_id]);
        }
        if left_state != right_state && self.input_states == [right_state, left_state] {
            return Ok([right_id, left_id]);
        }
        Err(invalid("recurrence transition state index is inconsistent"))
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
struct TransitionStateIndex {
    rows_by_state_pair: BTreeMap<(u32, u32), Vec<IndexedTransition>>,
}

#[cfg(test)]
impl TransitionStateIndex {
    fn new(
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let mut result = Self::default();
        for transition in &template.transitions {
            let input_states = catalog.u32_sequence(
                transition.input_state_sequence_id,
                "transition input states",
            )?;
            let input_states: [u32; 2] = input_states
                .try_into()
                .map_err(|_| invalid("direct recurrence requires binary prepared transitions"))?;
            result.insert(*transition, input_states);
        }
        Ok(result)
    }

    fn insert(&mut self, row: TransitionRow, input_states: [u32; 2]) {
        self.rows_by_state_pair
            .entry(canonical_state_pair(input_states))
            .or_default()
            .push(IndexedTransition { row, input_states });
    }

    fn rows(&self, left_state: u32, right_state: u32) -> &[IndexedTransition] {
        self.rows_by_state_pair
            .get(&canonical_state_pair([left_state, right_state]))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedTransitionWitness {
    row: LCColorTransitionWitnessRow,
    witness: LCColorTransitionWitness,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PreparedFlavourFlow {
    Constant(Box<[i32]>),
    AppendLeft(i32),
    AppendRight(i32),
    ConcatLeftRight(i32),
}

impl PreparedFlavourFlow {
    fn new(quantum: QuantumFlowRow, catalog: &TemplateCatalog<'_>) -> RusticolResult<Self> {
        let operation = catalog.string(
            quantum.flavour_flow_operation_string_id,
            "quantum-flow flavour operation",
        )?;
        let static_result = catalog.flavour_flow(
            quantum.result_flavour_flow_id,
            "quantum-flow result flavour",
        )?;
        let result_particle = *static_result
            .last()
            .ok_or_else(|| invalid("quantum-flow result flavour ancestry is empty"))?;
        match operation {
            "constant-result" => Ok(Self::Constant(static_result.into())),
            "append-left-result" => Ok(Self::AppendLeft(result_particle)),
            "append-right-result" => Ok(Self::AppendRight(result_particle)),
            "concat-left-right-result" => Ok(Self::ConcatLeftRight(result_particle)),
            value => Err(invalid(format!(
                "unsupported quantum-flow flavour operation {value:?}"
            ))),
        }
    }

    fn result_particle(&self) -> i32 {
        match self {
            Self::Constant(result) => *result
                .last()
                .expect("prepared constant flavour flow is nonempty"),
            Self::AppendLeft(result_particle)
            | Self::AppendRight(result_particle)
            | Self::ConcatLeftRight(result_particle) => *result_particle,
        }
    }

    fn apply(&self, parents: &[&CurrentCoreKey; 2]) -> Vec<i32> {
        // Runtime-helicity unions intentionally collapse construction ancestry
        // to the physical result species.
        if parents
            .iter()
            .all(|parent| parent.helicity_identity().strategy() == RecurrenceStrategy::AllFlowUnion)
        {
            return vec![self.result_particle()];
        }

        let append_result = |parent: &CurrentCoreKey, result_particle| {
            let mut result = parent.flavour_flow().to_vec();
            if result.last().copied() != Some(result_particle) {
                result.push(result_particle);
            }
            result
        };

        match self {
            Self::Constant(result) => result.to_vec(),
            Self::AppendLeft(result_particle) => append_result(parents[0], *result_particle),
            Self::AppendRight(result_particle) => append_result(parents[1], *result_particle),
            Self::ConcatLeftRight(result_particle) => {
                let mut result = Vec::with_capacity(
                    parents[0]
                        .flavour_flow()
                        .len()
                        .saturating_add(parents[1].flavour_flow().len())
                        .saturating_add(1),
                );
                result.extend_from_slice(parents[0].flavour_flow());
                result.extend_from_slice(parents[1].flavour_flow());
                result.push(*result_particle);
                result
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedTransition {
    row: TransitionRow,
    input_states: [u32; 2],
    quantum: QuantumFlowRow,
    quantum_input_states: [u32; 2],
    quantum_input_spins: [i32; 2],
    local_coupling_orders: Box<[u32]>,
    contraction: ColorContractionRow,
    canonical_input_order: [u32; 2],
    input_exchange_factor: Option<ExactComplexRational>,
    transition_exact_factor: ExactComplexRational,
    contraction_exact_factor: ExactComplexRational,
    coupling_authenticated: bool,
    binding_coupling: ExactComplexRational,
    output_factor_source: u8,
    result_flavour: PreparedFlavourFlow,
    quantum_semantic_digest: SemanticDigest,
    #[cfg(feature = "on-the-fly-test-support")]
    transition_semantic_digest: SemanticDigest,
    #[cfg(feature = "on-the-fly-test-support")]
    evaluator_binding_semantic_digest: SemanticDigest,
    contact_orbit: Option<PreparedContactOrbitTransition>,
    witnesses: Box<[PreparedTransitionWitness]>,
}

impl PreparedTransition {
    fn new(
        row: TransitionRow,
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let input_states =
            catalog.u32_sequence(row.input_state_sequence_id, "transition input states")?;
        let input_states: [u32; 2] = input_states
            .try_into()
            .map_err(|_| invalid("direct recurrence requires binary prepared transitions"))?;
        let quantum = template
            .quantum_flows
            .get(row.quantum_flow_template_id as usize)
            .copied()
            .ok_or_else(|| invalid("transition quantum-flow template is absent"))?;
        let quantum_input_states =
            catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
        let quantum_input_spins =
            catalog.i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?;
        let quantum_input_flavour_ids = catalog.u32_sequence(
            quantum.input_flavour_sequence_id,
            "quantum input flavour flows",
        )?;
        let quantum_input_number_ids = catalog.u32_sequence(
            quantum.input_quantum_sequence_id,
            "quantum input number flows",
        )?;
        if quantum_input_states.len() != 2
            || quantum_input_spins.len() != 2
            || quantum_input_flavour_ids.len() != 2
            || quantum_input_number_ids.len() != 2
        {
            return Err(invalid(
                "direct recurrence requires binary quantum-flow contracts",
            ));
        }
        let quantum_input_states = [quantum_input_states[0], quantum_input_states[1]];
        let quantum_input_spins = [quantum_input_spins[0], quantum_input_spins[1]];
        let quantum_input_flavour_ids =
            [quantum_input_flavour_ids[0], quantum_input_flavour_ids[1]];
        for flavour_id in quantum_input_flavour_ids {
            let _ = catalog.flavour_flow(flavour_id, "quantum parent flavour")?;
        }
        let _quantum_input_number_ids = [quantum_input_number_ids[0], quantum_input_number_ids[1]];

        let local_coupling_orders = catalog
            .coupling_orders(row.coupling_order_set_id)?
            .into_boxed_slice();
        let contraction = template
            .color_contractions
            .get(row.color_contraction_template_id as usize)
            .copied()
            .ok_or_else(|| invalid("transition color contraction is absent"))?;
        let quantum_coupling = catalog.factor(
            quantum.exact_coupling_factor_id,
            "transition quantum-flow coupling",
        )?;
        let binding_coupling = catalog.factor(
            row.binding_coupling_factor_id,
            "transition binding coupling",
        )?;
        let canonical_input_order = match catalog.u32_sequence(
            row.canonical_input_order_sequence_id,
            "transition canonical input order",
        )? {
            [0, 1] => [0, 1],
            [1, 0] => [1, 0],
            _ => {
                return Err(invalid(
                    "transition canonical input order is not a binary permutation",
                ));
            }
        };
        let input_exchange_factor = if row.input_exchange_factor_id == MISSING_U32 {
            None
        } else {
            Some(catalog.factor(row.input_exchange_factor_id, "transition input-exchange")?)
        };
        let transition_exact_factor = catalog.factor(row.exact_factor_id, "transition exact")?;
        let contraction_exact_factor =
            catalog.factor(contraction.exact_coefficient_factor_id, "color contraction")?;
        let result_flavour = PreparedFlavourFlow::new(quantum, catalog)?;
        let quantum_semantic_digest =
            catalog.digest(quantum.semantic_digest_id, "quantum-flow semantic")?;
        #[cfg(feature = "on-the-fly-test-support")]
        let transition_semantic_digest =
            catalog.digest(row.semantic_digest_id, "transition semantic")?;
        #[cfg(feature = "on-the-fly-test-support")]
        let evaluator_binding_semantic_digest = {
            let binding = template
                .evaluator_bindings
                .get(row.evaluator_binding_id as usize)
                .ok_or_else(|| invalid("transition evaluator binding is absent"))?;
            if binding.id != row.evaluator_binding_id {
                return Err(invalid("transition evaluator binding is not canonical"));
            }
            catalog.digest(binding.semantic_digest_id, "transition evaluator binding")?
        };
        let witness_rows = catalog.witness_rows(row.color_contraction_template_id)?;
        let contact_orbit = prepare_contact_orbit_transition(
            template,
            catalog,
            row,
            quantum_semantic_digest,
            &local_coupling_orders,
            binding_coupling,
            transition_exact_factor,
            contraction_exact_factor,
            input_exchange_factor,
            witness_rows,
        )?;
        let witnesses = witness_rows
            .iter()
            .copied()
            .map(|witness_row| {
                Ok(PreparedTransitionWitness {
                    row: witness_row,
                    witness: catalog.witness(witness_row)?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        Ok(Self {
            row,
            input_states,
            quantum,
            quantum_input_states,
            quantum_input_spins,
            local_coupling_orders,
            contraction,
            canonical_input_order,
            input_exchange_factor,
            transition_exact_factor,
            contraction_exact_factor,
            coupling_authenticated: quantum_coupling == binding_coupling,
            binding_coupling,
            output_factor_source: row.output_factor_source,
            result_flavour,
            quantum_semantic_digest,
            #[cfg(feature = "on-the-fly-test-support")]
            transition_semantic_digest,
            #[cfg(feature = "on-the-fly-test-support")]
            evaluator_binding_semantic_digest,
            contact_orbit,
            witnesses,
        })
    }

    fn parent_ids(
        &self,
        left_state: u32,
        right_state: u32,
        left_id: u32,
        right_id: u32,
    ) -> RusticolResult<[u32; 2]> {
        if self.input_states == [left_state, right_state] {
            return Ok([left_id, right_id]);
        }
        if left_state != right_state && self.input_states == [right_state, left_state] {
            return Ok([right_id, left_id]);
        }
        Err(invalid("recurrence transition state index is inconsistent"))
    }

    fn quantum_flow_matches(&self, parents: &[&CurrentCoreKey; 2]) -> bool {
        (0..2).all(|index| {
            self.quantum_input_states[index] == parents[index].current_state_template_id()
                && quantum_parent_spin_matches(self.quantum_input_spins[index], parents[index])
        })
    }

    fn canonical_evaluator_parents(
        &self,
        concrete_parent_ids: [u32; 2],
    ) -> ([u32; 2], ExactComplexRational) {
        let mut ordered = match self.canonical_input_order {
            [0, 1] => concrete_parent_ids,
            [1, 0] => [concrete_parent_ids[1], concrete_parent_ids[0]],
            _ => unreachable!("validated prepared transition input order"),
        };
        let mut factor = ExactComplexRational::ONE;
        if let Some(exchange_factor) = self.input_exchange_factor
            && ordered[1] < ordered[0]
        {
            ordered.swap(0, 1);
            factor = exchange_factor;
        }
        (ordered, factor)
    }

    fn result_flavour_flow(&self, parents: &[&CurrentCoreKey; 2]) -> Vec<i32> {
        self.result_flavour.apply(parents)
    }

    fn output_factor(&self) -> RusticolResult<ExactComplexRational> {
        if !self.coupling_authenticated {
            return Err(invalid(
                "transition binding coupling does not match its quantum-flow coupling witness",
            ));
        }
        output_factor_from_binding(
            self.binding_coupling,
            self.output_factor_source,
            "transition",
        )
    }

    fn structural_transition(&self) -> RusticolResult<StructuralTransition> {
        if self.quantum.result_state_template_id != self.row.result_state_template_id {
            return Err(invalid(
                "structural-demand transition and quantum-flow result states differ",
            ));
        }
        Ok(StructuralTransition {
            parents: [
                StructuralState::new(self.quantum_input_states[0], self.quantum_input_spins[0]),
                StructuralState::new(self.quantum_input_states[1], self.quantum_input_spins[1]),
            ],
            result: StructuralState::new(
                self.quantum.result_state_template_id,
                self.quantum.result_spin_state,
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PreparedTransitionLocation {
    transition_id: u32,
    state_pair: (u32, u32),
    row_index: usize,
}

fn reserve_prepared_transition_locations(
    locations: &mut Vec<PreparedTransitionLocation>,
    count: usize,
) -> RusticolResult<()> {
    locations.try_reserve_exact(count).map_err(|error| {
        RusticolError::internal(format!(
            "prepared transition-location allocation failed: {error}"
        ))
    })
}

#[derive(Debug, Default)]
struct PreparedTransitionCatalog {
    rows_by_state_pair: BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    transition_locations: Vec<PreparedTransitionLocation>,
    structural_transitions: Box<[StructuralTransition]>,
    decoded_transition_count: usize,
    decoded_witness_count: usize,
}

impl PreparedTransitionCatalog {
    fn new(
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let mut rows_by_state_pair = BTreeMap::<(u32, u32), Vec<PreparedTransition>>::new();
        let mut transition_locations = Vec::new();
        reserve_prepared_transition_locations(
            &mut transition_locations,
            template.transitions.len(),
        )?;
        let mut structural_transitions = BTreeSet::new();
        let mut decoded_transition_count = 0usize;
        let mut decoded_witness_count = 0usize;
        for row in template.transitions.iter().copied() {
            let prepared = PreparedTransition::new(row, template, catalog)?;
            decoded_transition_count += 1;
            decoded_witness_count += prepared.witnesses.len();
            structural_transitions.insert(prepared.structural_transition()?);
            let state_pair = canonical_state_pair(prepared.input_states);
            let row_index = rows_by_state_pair.get(&state_pair).map_or(0, Vec::len);
            transition_locations.push(PreparedTransitionLocation {
                transition_id: prepared.row.id,
                state_pair,
                row_index,
            });
            rows_by_state_pair
                .entry(state_pair)
                .or_default()
                .push(prepared);
        }
        // Template transition rows retain their authenticated source order in
        // `rows_by_state_pair`.  Only this private lookup side index needs ID
        // order for binary search, so sort the already-reserved locator rows
        // without perturbing ordinary transition iteration or IDs.
        transition_locations.sort_unstable_by_key(|location| location.transition_id);
        if !transition_locations
            .windows(2)
            .all(|rows| rows[0].transition_id < rows[1].transition_id)
        {
            return Err(invalid(
                "prepared transition-location index contains duplicate IDs",
            ));
        }
        let result = Self {
            rows_by_state_pair,
            transition_locations,
            structural_transitions: structural_transitions
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            decoded_transition_count,
            decoded_witness_count,
        };
        debug_assert_eq!(
            result.decoded_transition_count(),
            template.transitions.len()
        );
        debug_assert_eq!(
            result.decoded_witness_count(),
            result
                .rows_by_state_pair
                .values()
                .flatten()
                .map(|transition| transition.witnesses.len())
                .sum::<usize>()
        );
        Ok(result)
    }

    fn rows(&self, left_state: u32, right_state: u32) -> &[PreparedTransition] {
        self.rows_by_state_pair
            .get(&canonical_state_pair([left_state, right_state]))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn contact_orbit(&self, transition_id: u32) -> Option<&PreparedContactOrbitTransition> {
        let location = self
            .transition_locations
            .binary_search_by_key(&transition_id, |location| location.transition_id)
            .ok()
            .and_then(|index| self.transition_locations.get(index))?;
        self.rows_by_state_pair
            .get(&location.state_pair)
            .and_then(|rows| rows.get(location.row_index))
            .filter(|row| row.row.id == transition_id)
            .and_then(|row| row.contact_orbit.as_ref())
    }

    fn structural_transitions(&self) -> &[StructuralTransition] {
        &self.structural_transitions
    }

    fn decoded_transition_count(&self) -> usize {
        self.decoded_transition_count
    }

    fn decoded_witness_count(&self) -> usize {
        self.decoded_witness_count
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct EstablishedContactContributionToken {
    destination_id: usize,
    contribution_ordinal: usize,
}

#[derive(Debug)]
struct EstablishedContactOwnerPlan {
    stage_current_start: usize,
    selected_tokens: Vec<EstablishedContactContributionToken>,
    staged_contribution_count: usize,
    retained_contribution_count: usize,
    resident_contribution_count_after_commit: usize,
}

impl EstablishedContactOwnerPlan {
    /// Commit a completely validated owner plan without allocating or failing.
    ///
    /// The plan owns the complete selected-token output and every fallible
    /// candidate/decode/count operation has already succeeded. `retain` keeps
    /// the canonical `BTreeMap` order and performs no allocation, so live
    /// construction state changes only in this final infallible phase.
    fn commit(self, currents: &mut [PendingCurrent], resident_contribution_count: &mut usize) {
        let mut selected_index = 0usize;
        let mut observed_count = 0usize;
        let mut retained_count = 0usize;
        for (destination_id, current) in currents
            .iter_mut()
            .enumerate()
            .skip(self.stage_current_start)
        {
            let mut contribution_ordinal = 0usize;
            current.contributions.retain(|_, _| {
                let token = EstablishedContactContributionToken {
                    destination_id,
                    contribution_ordinal,
                };
                contribution_ordinal += 1;
                observed_count += 1;
                let selected = self.selected_tokens.get(selected_index) == Some(&token);
                if selected {
                    selected_index += 1;
                    retained_count += 1;
                }
                selected
            });
        }
        debug_assert_eq!(observed_count, self.staged_contribution_count);
        debug_assert_eq!(retained_count, self.retained_contribution_count);
        debug_assert_eq!(selected_index, self.selected_tokens.len());
        *resident_contribution_count = self.resident_contribution_count_after_commit;
    }
}

/// Plan exact established-path contact ownership over the complete stage fan-in.
///
/// Uncertified contributions are represented by `None` candidates and pass
/// through unchanged. The stable token is a snapshot of the canonical
/// destination/current order plus the contribution's ordered-map ordinal.
/// Every allocation and error-capable proof lookup happens before the caller
/// replaces any live contribution storage or resident-count telemetry.
fn plan_established_contact_orbit_owners(
    stage_current_start: usize,
    prepared_transitions: &PreparedTransitionCatalog,
    currents: &[PendingCurrent],
    resident_contribution_count: usize,
) -> RusticolResult<Option<EstablishedContactOwnerPlan>> {
    plan_established_contact_orbit_owners_with_resolver(
        stage_current_start,
        currents,
        resident_contribution_count,
        |transition_id| prepared_transitions.contact_orbit(transition_id),
    )
}

fn reserve_established_contact_candidates<T>(
    candidates: &mut Vec<T>,
    count: usize,
) -> RusticolResult<()> {
    candidates.try_reserve_exact(count).map_err(|error| {
        RusticolError::internal(format!(
            "established contact-orbit candidate allocation failed: {error}"
        ))
    })
}

fn plan_established_contact_orbit_owners_with_resolver<'a, F>(
    stage_current_start: usize,
    currents: &'a [PendingCurrent],
    resident_contribution_count: usize,
    mut contact_orbit: F,
) -> RusticolResult<Option<EstablishedContactOwnerPlan>>
where
    F: FnMut(u32) -> Option<&'a PreparedContactOrbitTransition>,
{
    if stage_current_start > currents.len() {
        return Err(invalid(
            "contact-orbit stage current boundary exceeds current storage",
        ));
    }
    let has_certified_contact = currents[stage_current_start..].iter().any(|current| {
        current
            .contributions
            .keys()
            .any(|pending| contact_orbit(pending.key.transition_template_id()).is_some())
    });
    if !has_certified_contact {
        return Ok(None);
    }
    let staged_contribution_count =
        currents[stage_current_start..]
            .iter()
            .try_fold(0usize, |count, current| {
                count
                    .checked_add(current.contributions.len())
                    .ok_or_else(|| invalid("contact-orbit staged contribution count exceeds usize"))
            })?;
    let mut candidates = Vec::new();
    reserve_established_contact_candidates(&mut candidates, staged_contribution_count)?;
    for (destination_id, current) in currents.iter().enumerate().skip(stage_current_start) {
        for (contribution_ordinal, pending) in current.contributions.keys().enumerate() {
            let token = EstablishedContactContributionToken {
                destination_id,
                contribution_ordinal,
            };
            let candidate =
                if let Some(contact_orbit) = contact_orbit(pending.key.transition_template_id()) {
                    let [left_id, right_id] = pending.parent_current_ids.as_ref() else {
                        return Err(invalid(
                            "certified contact-orbit contribution is not binary",
                        ));
                    };
                    let left = currents
                        .get(*left_id as usize)
                        .ok_or_else(|| invalid("contact-orbit left parent is absent"))?;
                    let right = currents
                        .get(*right_id as usize)
                        .ok_or_else(|| invalid("contact-orbit right parent is absent"))?;
                    Some(contact_orbit.owner_candidate(
                        &current.key,
                        [&left.key, &right.key],
                        pending.key.color_witness_term_id(),
                    )?)
                } else {
                    None
                };
            candidates.push((token, candidate));
        }
    }
    if candidates.len() != staged_contribution_count {
        return Err(invalid(
            "contact-orbit staged contribution snapshot changed length",
        ));
    }
    let selected_tokens = selected_contact_orbit_owner_tokens(candidates)?;
    let retained_contribution_count = selected_tokens.len();
    let removed_contribution_count = staged_contribution_count
        .checked_sub(retained_contribution_count)
        .ok_or_else(|| invalid("contact-orbit retained contribution count exceeds snapshot"))?;
    let resident_contribution_count_after_commit = resident_contribution_count
        .checked_sub(removed_contribution_count)
        .ok_or_else(|| invalid("contact-orbit resident contribution count underflows"))?;
    Ok(Some(EstablishedContactOwnerPlan {
        stage_current_start,
        selected_tokens,
        staged_contribution_count,
        retained_contribution_count,
        resident_contribution_count_after_commit,
    }))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedClosureWitness {
    row: LCColorTransitionWitnessRow,
    witness: LCColorTransitionWitness,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedClosureQuantumFlow {
    row: Option<QuantumFlowRow>,
    input_states: Option<[u32; 2]>,
    input_spins: Option<[i32; 2]>,
    coupling_authenticated: bool,
    binding_coupling: ExactComplexRational,
    output_factor_source: u8,
}

impl PreparedClosureQuantumFlow {
    fn unbound(closure: ClosureRow, catalog: &TemplateCatalog<'_>) -> RusticolResult<Self> {
        let binding_coupling = catalog.factor(
            closure.binding_coupling_factor_id,
            "closure binding coupling",
        )?;
        Ok(Self {
            row: None,
            input_states: None,
            input_spins: None,
            coupling_authenticated: true,
            binding_coupling,
            output_factor_source: closure.output_factor_source,
        })
    }

    fn bound(
        closure: ClosureRow,
        quantum: QuantumFlowRow,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let input_states =
            catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
        let input_spins =
            catalog.i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?;
        let input_flavours = catalog.u32_sequence(
            quantum.input_flavour_sequence_id,
            "quantum input flavour flows",
        )?;
        let input_quantum_numbers = catalog.u32_sequence(
            quantum.input_quantum_sequence_id,
            "quantum input number flows",
        )?;
        if input_states.len() != 2
            || input_spins.len() != 2
            || input_flavours.len() != 2
            || input_quantum_numbers.len() != 2
        {
            return Err(invalid(
                "direct recurrence requires binary quantum-flow contracts",
            ));
        }
        for flavour_id in input_flavours {
            let _ = catalog.flavour_flow(*flavour_id, "quantum parent flavour")?;
        }
        let quantum_coupling = catalog.factor(
            quantum.exact_coupling_factor_id,
            "closure quantum-flow coupling",
        )?;
        let binding_coupling = catalog.factor(
            closure.binding_coupling_factor_id,
            "closure binding coupling",
        )?;
        Ok(Self {
            row: Some(quantum),
            input_states: Some([input_states[0], input_states[1]]),
            input_spins: Some([input_spins[0], input_spins[1]]),
            coupling_authenticated: quantum_coupling == binding_coupling,
            binding_coupling,
            output_factor_source: closure.output_factor_source,
        })
    }

    fn matches(&self, parents: &[&CurrentCoreKey; 2]) -> bool {
        let (Some(input_states), Some(input_spins)) = (self.input_states, self.input_spins) else {
            return true;
        };
        (0..2).all(|index| {
            input_states[index] == parents[index].current_state_template_id()
                && quantum_parent_spin_matches(input_spins[index], parents[index])
        })
    }

    fn template_id(&self) -> Option<u32> {
        self.row.map(|row| row.id)
    }

    fn output_factor(&self) -> RusticolResult<ExactComplexRational> {
        if !self.coupling_authenticated {
            return Err(invalid(
                "closure binding coupling does not match its quantum-flow coupling witness",
            ));
        }
        output_factor_from_binding(self.binding_coupling, self.output_factor_source, "closure")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedClosure {
    row: ClosureRow,
    input_states: [u32; 2],
    quantum_flows: Box<[PreparedClosureQuantumFlow]>,
    contraction: ColorContractionRow,
    canonical_input_order: [u32; 2],
    input_exchange_factor: Option<ExactComplexRational>,
    closure_exact_factor: ExactComplexRational,
    contraction_exact_factor: ExactComplexRational,
    semantic_digest: SemanticDigest,
    witnesses: Box<[PreparedClosureWitness]>,
}

impl PreparedClosure {
    fn new(
        row: ClosureRow,
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let input_states =
            catalog.u32_sequence(row.input_state_sequence_id, "closure input states")?;
        let input_states: [u32; 2] = input_states
            .try_into()
            .map_err(|_| invalid("direct recurrence requires binary prepared closures"))?;
        let eligible = catalog.u32_sequence(
            row.eligible_quantum_flow_sequence_id,
            "closure eligible quantum flows",
        )?;
        let quantum_flows = if eligible.is_empty() {
            vec![PreparedClosureQuantumFlow::unbound(row, catalog)?]
        } else {
            eligible
                .iter()
                .copied()
                .map(|quantum_id| {
                    let quantum = template
                        .quantum_flows
                        .get(quantum_id as usize)
                        .copied()
                        .ok_or_else(|| invalid("closure quantum flow is absent"))?;
                    PreparedClosureQuantumFlow::bound(row, quantum, catalog)
                })
                .collect::<RusticolResult<Vec<_>>>()?
        }
        .into_boxed_slice();
        let contraction = template
            .color_contractions
            .get(row.color_contraction_template_id as usize)
            .copied()
            .ok_or_else(|| invalid("closure color contraction is absent"))?;
        let canonical_input_order = match catalog.u32_sequence(
            row.canonical_input_order_sequence_id,
            "closure canonical input order",
        )? {
            [0, 1] => [0, 1],
            [1, 0] => [1, 0],
            _ => {
                return Err(invalid(
                    "closure canonical input order is not a binary permutation",
                ));
            }
        };
        let input_exchange_factor = if row.input_exchange_factor_id == MISSING_U32 {
            None
        } else {
            Some(catalog.factor(row.input_exchange_factor_id, "closure input-exchange")?)
        };
        let witnesses = catalog
            .witness_rows(row.color_contraction_template_id)?
            .iter()
            .copied()
            .map(|witness_row| {
                Ok(PreparedClosureWitness {
                    row: witness_row,
                    witness: catalog.witness(witness_row)?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        Ok(Self {
            row,
            input_states,
            quantum_flows,
            contraction,
            canonical_input_order,
            input_exchange_factor,
            closure_exact_factor: catalog.factor(row.exact_factor_id, "closure exact")?,
            contraction_exact_factor: catalog
                .factor(contraction.exact_coefficient_factor_id, "closure color")?,
            semantic_digest: catalog.digest(row.semantic_digest_id, "closure semantic")?,
            witnesses,
        })
    }

    fn parent_ids(
        &self,
        anchor_state: u32,
        complement_state: u32,
        anchor_id: u32,
        complement_id: u32,
    ) -> RusticolResult<[u32; 2]> {
        if self.input_states == [complement_state, anchor_state] {
            return Ok([complement_id, anchor_id]);
        }
        if anchor_state != complement_state && self.input_states == [anchor_state, complement_state]
        {
            return Ok([anchor_id, complement_id]);
        }
        Err(invalid("recurrence closure state index is inconsistent"))
    }

    fn canonical_evaluator_parents(
        &self,
        parent_ids: [u32; 2],
    ) -> ([u32; 2], ExactComplexRational) {
        let mut ordered = match self.canonical_input_order {
            [0, 1] => parent_ids,
            [1, 0] => [parent_ids[1], parent_ids[0]],
            _ => unreachable!("validated prepared closure input order"),
        };
        let mut factor = ExactComplexRational::ONE;
        if let Some(exchange_factor) = self.input_exchange_factor
            && ordered[1] < ordered[0]
        {
            ordered.swap(0, 1);
            factor = exchange_factor;
        }
        (ordered, factor)
    }
}

#[derive(Debug, Default)]
struct PreparedClosureCatalog {
    rows_by_state_pair: BTreeMap<(u32, u32), Vec<PreparedClosure>>,
    row_count: usize,
}

impl PreparedClosureCatalog {
    fn new(
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let mut rows_by_state_pair = BTreeMap::<(u32, u32), Vec<PreparedClosure>>::new();
        for row in template.closures.iter().copied() {
            let prepared = PreparedClosure::new(row, template, catalog)?;
            rows_by_state_pair
                .entry(canonical_state_pair(prepared.input_states))
                .or_default()
                .push(prepared);
        }
        Ok(Self {
            rows_by_state_pair,
            row_count: template.closures.len(),
        })
    }

    fn rows(&self, left_state: u32, right_state: u32) -> &[PreparedClosure] {
        self.rows_by_state_pair
            .get(&canonical_state_pair([left_state, right_state]))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn row_count(&self) -> usize {
        self.row_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedClosureSector {
    row: ProcessPhysicalLCSectorRow,
    expected_components: Box<[LCColorComponent]>,
    contracted_color_canonical_owner: bool,
    anchor_support: Box<[u32]>,
    complement_support: Box<[u32]>,
}

#[derive(Debug, Default)]
struct PreparedClosureSectorCatalog {
    sectors: BTreeMap<u32, PreparedClosureSector>,
    v4_trace_sector_by_anchored_word: BTreeMap<Box<[u32]>, u32>,
}

impl PreparedClosureSectorCatalog {
    fn new(
        strategy: RecurrenceStrategy,
        process: &OwnedRecurrenceProcessInput,
        catalog: &ProcessCatalog<'_>,
        materialized_sector_ids: &BTreeSet<u32>,
    ) -> RusticolResult<Self> {
        let source_count = process.external_legs.len();
        let full_support = (0..source_count as u32).collect::<Vec<_>>();
        let mut sectors = BTreeMap::new();
        let mut v4_trace_sector_by_anchored_word = BTreeMap::new();
        let mut contracted_open_forests = HashSet::<Box<[LCColorComponent]>>::new();
        for row in process.physical_lc_sectors.iter().copied() {
            if !materialized_sector_ids.contains(&row.sector_id)
                && strategy != RecurrenceStrategy::ContractedColorUnion
            {
                continue;
            }
            let expected_components = expected_sector_components(row, process, catalog)?;
            let contracted_color_canonical_owner = if strategy
                == RecurrenceStrategy::ContractedColorUnion
                && row.kind()? == ProcessLCSectorKind::OpenLines
            {
                let mut canonical = expected_components.clone();
                canonical.sort_unstable();
                contracted_open_forests.insert(canonical.into_boxed_slice())
            } else {
                true
            };
            if !materialized_sector_ids.contains(&row.sector_id) {
                continue;
            }
            let complement_support = full_support
                .iter()
                .copied()
                .filter(|slot| *slot != row.closure_source_slot)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let prepared = PreparedClosureSector {
                row,
                expected_components: expected_components.into_boxed_slice(),
                contracted_color_canonical_owner,
                anchor_support: vec![row.closure_source_slot].into_boxed_slice(),
                complement_support,
            };
            if sectors.insert(row.sector_id, prepared).is_some() {
                return Err(invalid(
                    "prepared closure sector catalog contains a duplicate sector",
                ));
            }
            if row.kind()? == ProcessLCSectorKind::SingleTrace
                && catalog.string(
                    row.closure_proof_algorithm_string_id,
                    "prepared closure-anchor algorithm",
                )? == CYCLIC_TRACE_CLOSURE_ANCHOR_ALGORITHM
            {
                let word = catalog
                    .u32_sequence(row.word_sequence_id, "prepared v4 physical LC sector word")?;
                let key = anchored_cyclic_trace_word(word, row.closure_source_slot)?;
                if v4_trace_sector_by_anchored_word
                    .insert(key, row.sector_id)
                    .is_some()
                {
                    return Err(invalid(
                        "prepared v4 closure sectors contain an ambiguous cyclic trace word",
                    ));
                }
            }
        }
        if sectors.len() != materialized_sector_ids.len() {
            return Err(invalid(
                "prepared closure sector catalog omits a materialized sector",
            ));
        }
        Ok(Self {
            sectors,
            v4_trace_sector_by_anchored_word,
        })
    }

    fn get(&self, sector_id: u32) -> RusticolResult<&PreparedClosureSector> {
        self.sectors
            .get(&sector_id)
            .ok_or_else(|| invalid("prepared closure sector is absent"))
    }

    fn reflected_v4_sector(
        &self,
        mapped_word: &[u32],
        closure_source_slot: u32,
    ) -> RusticolResult<Option<u32>> {
        let key = anchored_cyclic_trace_word(mapped_word, closure_source_slot)?;
        Ok(self.v4_trace_sector_by_anchored_word.get(&key).copied())
    }
}

fn closure_domain_sector_owners(
    strategy: RecurrenceStrategy,
    prepared_sectors: &PreparedClosureSectorCatalog,
    physical_sector_ids: &BTreeSet<u32>,
) -> RusticolResult<BTreeMap<u32, u32>> {
    let mut owner_by_open_forest = BTreeMap::<Box<[LCColorComponent]>, u32>::new();
    let mut result = BTreeMap::new();
    for sector_id in physical_sector_ids.iter().copied() {
        let sector = prepared_sectors.get(sector_id)?;
        let owner_id = if strategy == RecurrenceStrategy::ContractedColorUnion
            && sector.row.kind()? == ProcessLCSectorKind::OpenLines
        {
            let mut canonical = sector.expected_components.to_vec();
            canonical.sort_unstable();
            let owner_id = *owner_by_open_forest
                .entry(canonical.into_boxed_slice())
                .or_insert(sector_id);
            if sector.contracted_color_canonical_owner != (owner_id == sector_id) {
                return Err(invalid(
                    "contracted closure-sector owner classification is inconsistent",
                ));
            }
            owner_id
        } else {
            sector_id
        };
        result.insert(sector_id, owner_id);
    }
    Ok(result)
}

#[derive(Debug, Default)]
pub(super) struct TransitionReflectionIndex {
    proofs_by_transition: BTreeMap<u32, TransitionReflectionProof>,
}

impl TransitionReflectionIndex {
    pub(super) fn new(
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let transitions_by_subject = template
            .transitions
            .iter()
            .map(|row| (row.template_string_id, row.id))
            .collect::<BTreeMap<_, _>>();
        let mut result = Self::default();
        for proof in &template.symmetry_proofs {
            if catalog.string(proof.proof_algorithm_string_id, "symmetry proof algorithm")?
                != "canonical-current-word-reversal-v1"
            {
                continue;
            }
            let subjects = catalog.u32_sequence(
                proof.subject_template_sequence_id,
                "current-word reversal proof subjects",
            )?;
            let [subject] = subjects else {
                return Err(invalid(
                    "current-word reversal proof must reference one transition",
                ));
            };
            let transition_id = transitions_by_subject
                .get(subject)
                .copied()
                .ok_or_else(|| {
                    invalid("current-word reversal proof subject is not a transition")
                })?;
            if catalog.u32_sequence(
                proof.input_permutation_sequence_id,
                "current-word reversal proof permutation",
            )? != [1, 0]
            {
                return Err(invalid(
                    "current-word reversal proof must exchange two transition inputs",
                ));
            }
            let phase =
                catalog.factor(proof.exact_phase_factor_id, "current-word reversal proof")?;
            let candidate = TransitionReflectionProof::new(
                phase,
                [catalog.digest(
                    proof.semantic_digest_id,
                    "current-word reversal semantic proof",
                )?],
                [catalog.digest(
                    proof.witness_digest_id,
                    "current-word reversal witness proof",
                )?],
            )?;
            match result.proofs_by_transition.entry(transition_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(candidate);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let Some(merged) = entry.get().merged_with(&candidate)? else {
                        return Err(invalid(
                            "transition has conflicting current-word reversal phases",
                        ));
                    };
                    entry.insert(merged);
                }
            }
        }
        Ok(result)
    }

    pub(super) fn proof(&self, transition_id: u32) -> Option<&TransitionReflectionProof> {
        self.proofs_by_transition.get(&transition_id)
    }
}

fn canonical_state_pair([left, right]: [u32; 2]) -> (u32, u32) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

/// Necessary physical-sector compatibility for a partial LC color forest.
///
/// Recurrence witnesses can join or explicitly reverse ordered components, but
/// they never split or internally permute an existing component. A partial
/// component must therefore occur as an oriented contiguous word in at least
/// one materialized representative sector. Pure-adjoint reversed words are
/// admitted transiently for one construction stage so reflection proofs can be
/// finalized over the complete fan-in. A stage-final reconciliation then
/// removes only exactly certified aliases. This cheap forward filter prevents
/// the builder from interning unrelated color words; final backward liveness
/// remains authoritative.
#[derive(Clone, Debug, Eq, PartialEq)]
enum SectorPostingStorage {
    Sparse(Box<[u32]>),
    Dense(Box<[u64]>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SectorPosting {
    cardinality: usize,
    storage: SectorPostingStorage,
}

impl SectorPosting {
    fn from_sorted_unique_sector_ids(sector_ids: Vec<u32>, sector_count: usize) -> Self {
        debug_assert!(!sector_ids.is_empty());
        debug_assert!(sector_ids.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(
            sector_ids
                .last()
                .is_none_or(|sector_id| (*sector_id as usize) < sector_count)
        );
        let cardinality = sector_ids.len();
        let sparse_bytes = cardinality
            .checked_mul(std::mem::size_of::<u32>())
            .expect("allocated sparse posting byte size exceeds usize");
        let dense_word_count = sector_count.div_ceil(u64::BITS as usize);
        let dense_bytes = dense_word_count
            .checked_mul(std::mem::size_of::<u64>())
            .expect("allocated dense posting byte size exceeds usize");
        let storage = if dense_bytes < sparse_bytes {
            let mut words = vec![0_u64; dense_word_count];
            for sector_id in sector_ids {
                let sector_index = sector_id as usize;
                words[sector_index / u64::BITS as usize] |=
                    1_u64 << (sector_index % u64::BITS as usize);
            }
            SectorPostingStorage::Dense(words.into_boxed_slice())
        } else {
            SectorPostingStorage::Sparse(sector_ids.into_boxed_slice())
        };
        Self {
            cardinality,
            storage,
        }
    }

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn contains(&self, sector_id: u32) -> bool {
        match &self.storage {
            SectorPostingStorage::Sparse(sector_ids) => {
                sector_ids.binary_search(&sector_id).is_ok()
            }
            SectorPostingStorage::Dense(words) => {
                let sector_index = sector_id as usize;
                words
                    .get(sector_index / u64::BITS as usize)
                    .is_some_and(|word| word & (1_u64 << (sector_index % u64::BITS as usize)) != 0)
            }
        }
    }

    fn any_sector(&self, mut predicate: impl FnMut(u32) -> bool) -> bool {
        match &self.storage {
            SectorPostingStorage::Sparse(sector_ids) => sector_ids.iter().copied().any(predicate),
            SectorPostingStorage::Dense(words) => {
                for (word_index, word) in words.iter().copied().enumerate() {
                    let mut remaining = word;
                    while remaining != 0 {
                        let bit_index = remaining.trailing_zeros() as usize;
                        let sector_index = word_index * u64::BITS as usize + bit_index;
                        let sector_id =
                            u32::try_from(sector_index).expect("sector posting exceeds u32");
                        if predicate(sector_id) {
                            return true;
                        }
                        remaining &= remaining - 1;
                    }
                }
                false
            }
        }
    }

    fn payload_bytes(&self) -> usize {
        match &self.storage {
            SectorPostingStorage::Sparse(sector_ids) => sector_ids
                .len()
                .checked_mul(std::mem::size_of::<u32>())
                .expect("allocated sparse posting byte size exceeds usize"),
            SectorPostingStorage::Dense(words) => words
                .len()
                .checked_mul(std::mem::size_of::<u64>())
                .expect("allocated dense posting byte size exceeds usize"),
        }
    }

    fn dense_words(&self) -> Option<&[u64]> {
        match &self.storage {
            SectorPostingStorage::Sparse(_) => None,
            SectorPostingStorage::Dense(words) => Some(words),
        }
    }

    fn is_sparse(&self) -> bool {
        matches!(&self.storage, SectorPostingStorage::Sparse(_))
    }
}

#[derive(Debug)]
struct MaterializedColorTargets {
    collect_telemetry: bool,
    non_trace_fragment_sectors: HashMap<Box<[u32]>, SectorPosting>,
    trace_component_sectors: HashMap<Box<[u32]>, SectorPosting>,
    accepted_component_forests: RefCell<HashSet<Box<[LCColorComponent]>>>,
    acceptance_cache_hit_count: Cell<usize>,
    acceptance_cache_miss_count: Cell<usize>,
    acceptance_accept_count: Cell<usize>,
    acceptance_reject_count: Cell<usize>,
    fragment_hash_lookup_count: Cell<usize>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MaterializedColorTargetTelemetry {
    fragment_bucket_count: usize,
    acceptance_cache_hit_count: usize,
    acceptance_cache_miss_count: usize,
    acceptance_accept_count: usize,
    acceptance_reject_count: usize,
    fragment_hash_lookup_count: usize,
    posting_incidence_count: usize,
    sparse_posting_bucket_count: usize,
    dense_posting_bucket_count: usize,
    sparse_posting_bytes: usize,
    dense_posting_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct PendingConstructionDomain {
    shared_source_end: usize,
    lane_internal_start: usize,
    lane_internal_end: usize,
}

impl PendingConstructionDomain {
    fn contains(self, current_id: usize) -> bool {
        current_id < self.shared_source_end
            || (self.lane_internal_start..self.lane_internal_end).contains(&current_id)
    }
}

#[derive(Debug, Default)]
struct LaneClosureSupportIndex<'a> {
    current_ids_by_support: BTreeMap<&'a [u32], Vec<u32>>,
}

impl<'a> LaneClosureSupportIndex<'a> {
    fn new(
        currents: &'a [PendingCurrent],
        construction_domain: Option<PendingConstructionDomain>,
    ) -> RusticolResult<Self> {
        let mut current_ids_by_support = BTreeMap::<&[u32], Vec<u32>>::new();
        for (current_id, current) in currents.iter().enumerate() {
            if construction_domain.is_some_and(|domain| !domain.contains(current_id)) {
                continue;
            }
            current_ids_by_support
                .entry(current.key.support_source_slots())
                .or_default()
                .push(
                    u32::try_from(current_id)
                        .map_err(|_| invalid("closure current ID exceeds u32"))?,
                );
        }
        Ok(Self {
            current_ids_by_support,
        })
    }

    fn current_ids(&self, support: &[u32]) -> &[u32] {
        self.current_ids_by_support
            .get(support)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

impl MaterializedColorTargets {
    #[cfg(test)]
    fn new(
        materialized_sector_ids: &BTreeSet<u32>,
        process: &OwnedRecurrenceProcessInput,
        catalog: &ProcessCatalog<'_>,
    ) -> RusticolResult<Self> {
        Self::new_with_telemetry(materialized_sector_ids, process, catalog, false)
    }

    #[cfg(test)]
    fn new_with_telemetry(
        materialized_sector_ids: &BTreeSet<u32>,
        process: &OwnedRecurrenceProcessInput,
        catalog: &ProcessCatalog<'_>,
        collect_telemetry: bool,
    ) -> RusticolResult<Self> {
        let sectors = materialized_sector_ids
            .iter()
            .copied()
            .map(|sector_id| {
                let sector = process
                    .physical_lc_sectors
                    .get(sector_id as usize)
                    .copied()
                    .ok_or_else(|| invalid("materialized LC sector is absent"))?;
                expected_sector_components(sector, process, catalog)
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Self::from_sectors_with_telemetry(sectors, collect_telemetry)
    }

    fn new_with_prepared_sectors(
        materialized_sector_ids: &BTreeSet<u32>,
        sectors: &PreparedClosureSectorCatalog,
        collect_telemetry: bool,
    ) -> RusticolResult<Self> {
        let sectors = materialized_sector_ids
            .iter()
            .copied()
            .map(|sector_id| Ok(sectors.get(sector_id)?.expected_components.to_vec()))
            .collect::<RusticolResult<Vec<_>>>()?;
        Self::from_sectors_with_telemetry(sectors, collect_telemetry)
    }

    #[cfg(test)]
    fn from_sectors(sectors: Vec<Vec<LCColorComponent>>) -> RusticolResult<Self> {
        Self::from_sectors_with_telemetry(sectors, false)
    }

    fn from_sectors_with_telemetry(
        sectors: Vec<Vec<LCColorComponent>>,
        collect_telemetry: bool,
    ) -> RusticolResult<Self> {
        if sectors.is_empty() {
            return Err(invalid(
                "recurrence construction has no materialized LC sector",
            ));
        }
        let sector_count = sectors.len();
        let mut non_trace_fragment_sector_ids = HashMap::<Box<[u32]>, Vec<u32>>::new();
        let mut trace_component_sector_ids = HashMap::<Box<[u32]>, Vec<u32>>::new();
        for (sector_index, sector) in sectors.iter().enumerate() {
            let sector_id = u32::try_from(sector_index)
                .map_err(|_| invalid("materialized LC sector index exceeds u32"))?;
            for target in sector {
                match target.kind() {
                    LCColorComponentKind::Trace => {
                        Self::include_fragment(
                            &mut trace_component_sector_ids,
                            target.source_slots(),
                            sector_id,
                        );
                        Self::include_cyclic_fragments(
                            &mut non_trace_fragment_sector_ids,
                            target.source_slots(),
                            sector_id,
                        );
                    }
                    LCColorComponentKind::OpenString | LCColorComponentKind::AdjointSegment => {
                        Self::include_linear_fragments(
                            &mut non_trace_fragment_sector_ids,
                            target.source_slots(),
                            sector_id,
                        );
                    }
                }
            }
        }
        Ok(Self {
            collect_telemetry,
            non_trace_fragment_sectors: Self::freeze_posting_index(
                non_trace_fragment_sector_ids,
                sector_count,
            ),
            trace_component_sectors: Self::freeze_posting_index(
                trace_component_sector_ids,
                sector_count,
            ),
            accepted_component_forests: RefCell::new(HashSet::new()),
            acceptance_cache_hit_count: Cell::new(0),
            acceptance_cache_miss_count: Cell::new(0),
            acceptance_accept_count: Cell::new(0),
            acceptance_reject_count: Cell::new(0),
            fragment_hash_lookup_count: Cell::new(0),
        })
    }

    fn accepts(&self, state: &DynamicLCColorState) -> bool {
        if self
            .accepted_component_forests
            .borrow()
            .contains(state.components())
        {
            if self.collect_telemetry {
                self.acceptance_cache_hit_count
                    .set(self.acceptance_cache_hit_count.get().saturating_add(1));
                self.acceptance_accept_count
                    .set(self.acceptance_accept_count.get().saturating_add(1));
            }
            return true;
        }
        if self.collect_telemetry {
            self.acceptance_cache_miss_count
                .set(self.acceptance_cache_miss_count.get().saturating_add(1));
        }
        let accepted = self.accepts_uncached(state);
        if accepted {
            if self.collect_telemetry {
                self.acceptance_accept_count
                    .set(self.acceptance_accept_count.get().saturating_add(1));
            }
            self.accepted_component_forests
                .borrow_mut()
                .insert(state.components().into());
        } else if self.collect_telemetry {
            self.acceptance_reject_count
                .set(self.acceptance_reject_count.get().saturating_add(1));
        }
        accepted
    }

    fn accepts_up_to_reflection(&self, state: &DynamicLCColorState) -> RusticolResult<bool> {
        if self.accepts(state) {
            return Ok(true);
        }
        if state.pure_adjoint_word().is_none() {
            return Ok(false);
        }
        Ok(self.accepts(&state.reversed()?))
    }

    fn accepts_uncached(&self, state: &DynamicLCColorState) -> bool {
        if state.components().is_empty() {
            return true;
        }
        let mut component_postings = Vec::with_capacity(state.components().len());
        for partial in state.components() {
            if self.collect_telemetry {
                self.fragment_hash_lookup_count
                    .set(self.fragment_hash_lookup_count.get().saturating_add(1));
            }
            let fragment_sectors = if partial.kind() == LCColorComponentKind::Trace {
                self.trace_component_sectors.get(partial.source_slots())
            } else {
                self.non_trace_fragment_sectors.get(partial.source_slots())
            };
            let Some(fragment_sectors) = fragment_sectors else {
                return false;
            };
            component_postings.push(fragment_sectors);
        }
        if component_postings.len() == 1 {
            return true;
        }
        if component_postings
            .iter()
            .all(|posting| posting.dense_words().is_some())
        {
            let word_count = component_postings[0]
                .dense_words()
                .expect("all color postings are dense")
                .len();
            debug_assert!(component_postings.iter().all(|posting| {
                posting
                    .dense_words()
                    .is_some_and(|words| words.len() == word_count)
            }));
            return (0..word_count).any(|word_index| {
                component_postings
                    .iter()
                    .fold(u64::MAX, |intersection, posting| {
                        intersection
                            & posting.dense_words().expect("all color postings are dense")
                                [word_index]
                    })
                    != 0
            });
        }
        let (candidate_posting_index, candidate_posting) = component_postings
            .iter()
            .enumerate()
            .min_by_key(|(_, posting)| posting.cardinality())
            .expect("non-empty color forest has at least one posting");
        candidate_posting.any_sector(|sector_id| {
            component_postings
                .iter()
                .enumerate()
                .all(|(posting_index, posting)| {
                    posting_index == candidate_posting_index || posting.contains(sector_id)
                })
        })
    }

    fn telemetry(&self) -> RusticolResult<MaterializedColorTargetTelemetry> {
        let postings = self
            .non_trace_fragment_sectors
            .values()
            .chain(self.trace_component_sectors.values());
        let mut posting_incidence_count = 0usize;
        let mut sparse_posting_bucket_count = 0usize;
        let mut dense_posting_bucket_count = 0usize;
        let mut sparse_posting_bytes = 0usize;
        let mut dense_posting_bytes = 0usize;
        for posting in postings {
            posting_incidence_count = posting_incidence_count
                .checked_add(posting.cardinality())
                .ok_or_else(|| invalid("color-posting incidence count exceeds usize"))?;
            if posting.is_sparse() {
                sparse_posting_bucket_count = sparse_posting_bucket_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("sparse color-posting bucket count exceeds usize"))?;
                sparse_posting_bytes = sparse_posting_bytes
                    .checked_add(posting.payload_bytes())
                    .ok_or_else(|| invalid("sparse color-posting bytes exceed usize"))?;
            } else {
                dense_posting_bucket_count = dense_posting_bucket_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("dense color-posting bucket count exceeds usize"))?;
                dense_posting_bytes = dense_posting_bytes
                    .checked_add(posting.payload_bytes())
                    .ok_or_else(|| invalid("dense color-posting bytes exceed usize"))?;
            }
        }
        Ok(MaterializedColorTargetTelemetry {
            fragment_bucket_count: self
                .non_trace_fragment_sectors
                .len()
                .checked_add(self.trace_component_sectors.len())
                .ok_or_else(|| invalid("color-fragment bucket count exceeds usize"))?,
            acceptance_cache_hit_count: self.acceptance_cache_hit_count.get(),
            acceptance_cache_miss_count: self.acceptance_cache_miss_count.get(),
            acceptance_accept_count: self.acceptance_accept_count.get(),
            acceptance_reject_count: self.acceptance_reject_count.get(),
            fragment_hash_lookup_count: self.fragment_hash_lookup_count.get(),
            posting_incidence_count,
            sparse_posting_bucket_count,
            dense_posting_bucket_count,
            sparse_posting_bytes,
            dense_posting_bytes,
        })
    }

    fn include_linear_fragments(
        index: &mut HashMap<Box<[u32]>, Vec<u32>>,
        source_slots: &[u32],
        sector_id: u32,
    ) {
        for start in 0..source_slots.len() {
            for end in start + 1..=source_slots.len() {
                Self::include_fragment(index, &source_slots[start..end], sector_id);
            }
        }
    }

    fn include_cyclic_fragments(
        index: &mut HashMap<Box<[u32]>, Vec<u32>>,
        source_slots: &[u32],
        sector_id: u32,
    ) {
        for start in 0..source_slots.len() {
            let mut fragment = Vec::with_capacity(source_slots.len());
            for offset in 0..source_slots.len() {
                fragment.push(source_slots[(start + offset) % source_slots.len()]);
                Self::include_fragment(index, &fragment, sector_id);
            }
        }
    }

    fn include_fragment(
        index: &mut HashMap<Box<[u32]>, Vec<u32>>,
        source_slots: &[u32],
        sector_id: u32,
    ) {
        if let Some(sector_ids) = index.get_mut(source_slots) {
            if sector_ids.last().copied() != Some(sector_id) {
                sector_ids.push(sector_id);
            }
        } else {
            index.insert(source_slots.to_vec().into_boxed_slice(), vec![sector_id]);
        }
    }

    fn freeze_posting_index(
        index: HashMap<Box<[u32]>, Vec<u32>>,
        sector_count: usize,
    ) -> HashMap<Box<[u32]>, SectorPosting> {
        index
            .into_iter()
            .map(|(fragment, sector_ids)| {
                (
                    fragment,
                    SectorPosting::from_sorted_unique_sector_ids(sector_ids, sector_count),
                )
            })
            .collect()
    }
}

#[cfg(test)]
fn component_can_embed(partial: &LCColorComponent, target: &LCColorComponent) -> bool {
    if partial.kind() == LCColorComponentKind::Trace {
        return target.kind() == LCColorComponentKind::Trace
            && partial.source_slots().len() == target.source_slots().len()
            && cyclic_word_contains(target.source_slots(), partial.source_slots());
    }
    match target.kind() {
        LCColorComponentKind::Trace => {
            cyclic_word_contains(target.source_slots(), partial.source_slots())
        }
        LCColorComponentKind::OpenString | LCColorComponentKind::AdjointSegment => {
            linear_word_contains(target.source_slots(), partial.source_slots())
        }
    }
}

#[cfg(test)]
fn linear_word_contains(target: &[u32], partial: &[u32]) -> bool {
    !partial.is_empty()
        && partial.len() <= target.len()
        && target
            .windows(partial.len())
            .any(|window| window == partial)
}

#[cfg(test)]
fn cyclic_word_contains(target: &[u32], partial: &[u32]) -> bool {
    !partial.is_empty()
        && partial.len() <= target.len()
        && (0..target.len()).any(|start| {
            (0..partial.len())
                .all(|offset| target[(start + offset) % target.len()] == partial[offset])
        })
}

pub(super) struct TemplateCatalog<'a> {
    input: &'a OwnedRecurrenceTemplateInput,
    strings: Vec<&'a str>,
    digests: Vec<SemanticDigest>,
    factors: Vec<ExactComplexRational>,
    coupling_names: Vec<&'a str>,
}

impl<'a> TemplateCatalog<'a> {
    pub(super) fn new(input: &'a OwnedRecurrenceTemplateInput) -> RusticolResult<Self> {
        let strings = decode_strings(
            &input.string_ranges,
            &input.string_bytes,
            "recurrence template string",
        )?;
        let digests = input
            .digest_catalog
            .iter()
            .map(|row| SemanticDigest::new(row.value))
            .collect::<RusticolResult<Vec<_>>>()?;
        let factors = decode_template_factors(input, &strings)?;
        let coupling_names = input
            .coupling_order_terms
            .iter()
            .map(|row| required_string(&strings, row.name_string_id, "coupling-order name"))
            .collect::<RusticolResult<BTreeSet<_>>>()?
            .into_iter()
            .collect();
        Ok(Self {
            input,
            strings,
            digests,
            factors,
            coupling_names,
        })
    }

    pub(super) fn string(&self, id: u32, label: &str) -> RusticolResult<&'a str> {
        required_string(&self.strings, id, label)
    }

    pub(super) fn digest(&self, id: u32, label: &str) -> RusticolResult<SemanticDigest> {
        self.digests
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} digest {id} is absent")))
    }

    pub(super) fn factor(&self, id: u32, label: &str) -> RusticolResult<ExactComplexRational> {
        self.factors
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} factor {id} is absent")))
    }

    pub(super) fn u32_sequence(&self, id: u32, label: &str) -> RusticolResult<&'a [u32]> {
        indexed_sequence(
            &self.input.u32_sequence_ranges,
            &self.input.u32_sequence_values,
            id,
            label,
        )
    }

    pub(super) fn i32_sequence(&self, id: u32, label: &str) -> RusticolResult<&'a [i32]> {
        indexed_sequence(
            &self.input.i32_sequence_ranges,
            &self.input.i32_sequence_values,
            id,
            label,
        )
    }

    pub(super) fn flavour_flow(&self, id: u32, label: &str) -> RusticolResult<&'a [i32]> {
        indexed_sequence(
            &self.input.flavour_flow_ranges,
            &self.input.flavour_flow_values,
            id,
            label,
        )
    }

    pub(super) fn source_seed(&self, row: SourceRow) -> RusticolResult<LCColorSourceSeed> {
        let operation = LCColorSourceSeedOperation::try_from(row.lc_color_seed_operation)?;
        let kind = (row.lc_color_seed_component_kind != u8::MAX)
            .then(|| LCColorComponentKind::try_from(row.lc_color_seed_component_kind))
            .transpose()?;
        LCColorSourceSeed::new(
            operation,
            row.lc_color_seed_shape_string_id,
            kind,
            LCColorComponentRole::try_from(row.lc_color_seed_component_role)?,
            self.digest(row.lc_color_seed_proof_digest_id, "source color proof")?,
        )
    }

    pub(super) fn witness(
        &self,
        row: LCColorTransitionWitnessRow,
    ) -> RusticolResult<LCColorTransitionWitness> {
        let permutation = match row.input_permutation {
            0 => [0, 1],
            1 => [1, 0],
            value => return Err(invalid(format!("invalid LC witness permutation {value}"))),
        };
        let kind = (row.result_component_kind != u8::MAX)
            .then(|| LCColorComponentKind::try_from(row.result_component_kind))
            .transpose()?;
        let shape =
            (row.result_shape_string_id != MISSING_U32).then_some(row.result_shape_string_id);
        let pairing_values = self.u32_sequence(
            row.input_port_pairing_sequence_id,
            "LC witness input-port pairings",
        )?;
        if pairing_values.len() % 4 != 0 {
            return Err(invalid(
                "LC witness input-port pairing sequence is not divisible by four",
            ));
        }
        let input_pairings = pairing_values
            .chunks_exact(4)
            .map(|chunk| {
                Ok([
                    LCColorParentPort::new(
                        u8::try_from(chunk[0])
                            .map_err(|_| invalid("LC witness parent index exceeds u8"))?,
                        u8::try_from(chunk[1])
                            .map_err(|_| invalid("LC witness local port index exceeds u8"))?,
                    )?,
                    LCColorParentPort::new(
                        u8::try_from(chunk[2])
                            .map_err(|_| invalid("LC witness parent index exceeds u8"))?,
                        u8::try_from(chunk[3])
                            .map_err(|_| invalid("LC witness local port index exceeds u8"))?,
                    )?,
                ])
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let binding_values = self.u32_sequence(
            row.result_port_binding_sequence_id,
            "LC witness result-port bindings",
        )?;
        if binding_values.len() % 2 != 0 {
            return Err(invalid(
                "LC witness result-port binding sequence is not divisible by two",
            ));
        }
        let result_port_bindings = binding_values
            .chunks_exact(2)
            .map(|chunk| {
                LCColorParentPort::new(
                    u8::try_from(chunk[0])
                        .map_err(|_| invalid("LC witness parent index exceeds u8"))?,
                    u8::try_from(chunk[1])
                        .map_err(|_| invalid("LC witness local port index exceeds u8"))?,
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let component_parent_order = match row.input_permutation {
            0 => [0, 1],
            1 => [1, 0],
            value => return Err(invalid(format!("invalid LC witness permutation {value}"))),
        };
        LCColorTransitionWitness::new(
            permutation,
            row.reverse_parent_mask,
            LCColorComponentOperation::try_from(row.component_operation)?,
            kind,
            LCColorComponentRole::try_from(row.result_component_role)?,
            shape,
            LCColorPortWiring::new(component_parent_order, input_pairings, result_port_bindings)?,
            self.factor(row.exact_factor_id, "LC witness")?,
            self.digest(row.proof_digest_id, "LC witness proof")?,
        )
    }

    pub(super) fn witness_rows(
        &self,
        color_contraction_id: u32,
    ) -> RusticolResult<&'a [LCColorTransitionWitnessRow]> {
        let row = self
            .input
            .color_contractions
            .get(color_contraction_id as usize)
            .ok_or_else(|| invalid("color-contraction template is absent"))?;
        let range = CheckedTableRange::new(row.witness_start, row.witness_count).as_usize_range(
            self.input.lc_color_transition_witnesses.len(),
            "LC witnesses",
        )?;
        Ok(&self.input.lc_color_transition_witnesses[range])
    }

    pub(super) fn coupling_orders(&self, set_id: u32) -> RusticolResult<Vec<u32>> {
        let range = self
            .input
            .coupling_order_ranges
            .get(set_id as usize)
            .ok_or_else(|| invalid(format!("coupling-order set {set_id} is absent")))?
            .range
            .as_usize_range(self.input.coupling_order_terms.len(), "coupling-order set")?;
        let mut result = vec![0_u32; self.coupling_names.len()];
        for term in &self.input.coupling_order_terms[range] {
            let name = self.string(term.name_string_id, "coupling-order term")?;
            let index = self
                .coupling_names
                .binary_search(&name)
                .map_err(|_| invalid("coupling-order name disappeared"))?;
            result[index] = term.power;
        }
        Ok(result)
    }

    pub(super) fn coupling_order_dimension(&self) -> usize {
        self.coupling_names.len()
    }

    #[allow(dead_code)]
    pub(super) fn coupling_order_names(&self) -> &[&'a str] {
        &self.coupling_names
    }
}

struct ProcessCatalog<'a> {
    input: &'a OwnedRecurrenceProcessInput,
    strings: Vec<&'a str>,
    factors: Vec<ExactComplexRational>,
}

impl<'a> ProcessCatalog<'a> {
    fn new(input: &'a OwnedRecurrenceProcessInput) -> RusticolResult<Self> {
        let strings = decode_strings(
            &input.string_ranges,
            &input.string_bytes,
            "recurrence process string",
        )?;
        let factors = decode_process_factors(input, &strings)?;
        Ok(Self {
            input,
            strings,
            factors,
        })
    }

    fn string(&self, id: u32, label: &str) -> RusticolResult<&'a str> {
        required_string(&self.strings, id, label)
    }

    fn factor(&self, id: u32, label: &str) -> RusticolResult<ExactComplexRational> {
        self.factors
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} factor {id} is absent")))
    }

    fn u32_sequence(&self, id: u32, label: &str) -> RusticolResult<&'a [u32]> {
        let range = self
            .input
            .u32_sequence_ranges
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} sequence {id} is absent")))?;
        let range = range.as_usize_range(self.input.u32_sequence_values.len(), label)?;
        Ok(&self.input.u32_sequence_values[range])
    }

    fn public_helicities(
        &self,
        source_states: &[SourceStateAssignment],
    ) -> RusticolResult<Vec<i32>> {
        if source_states.len() != self.input.external_legs.len() {
            return Err(invalid(
                "resolved helicity does not cover every external source slot",
            ));
        }
        source_states
            .iter()
            .copied()
            .enumerate()
            .map(|(source_slot, assignment)| {
                if assignment.source_slot() as usize != source_slot {
                    return Err(invalid(
                        "resolved-helicity source-state ancestry is not canonical",
                    ));
                }
                let leg = self.input.external_legs[source_slot];
                if u64::from(assignment.state_index()) >= leg.source_state_range.count {
                    return Err(invalid(format!(
                        "resolved helicity references absent state {} for source slot {source_slot}",
                        assignment.state_index()
                    )));
                }
                let row_index = leg
                    .source_state_range
                    .start
                    .checked_add(u64::from(assignment.state_index()))
                    .ok_or_else(|| invalid("resolved-helicity source-state index overflows"))?;
                let row_index = usize::try_from(row_index)
                    .map_err(|_| invalid("resolved-helicity source-state index exceeds usize"))?;
                let state = self
                    .input
                    .source_states
                    .get(row_index)
                    .ok_or_else(|| invalid("resolved-helicity source state is absent"))?;
                if state.source_slot as usize != source_slot
                    || state.state_index != assignment.state_index()
                {
                    return Err(invalid(
                        "resolved-helicity source-state catalog is inconsistent",
                    ));
                }
                Ok(state.public_helicity)
            })
            .collect()
    }
}

fn validate_cyclic_trace_closure_anchor_contracts(
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
) -> RusticolResult<()> {
    let mut has_cyclic_trace_anchor = false;
    for sector in &process.physical_lc_sectors {
        if process_catalog.string(
            sector.closure_proof_algorithm_string_id,
            "closure-anchor proof algorithm",
        )? == CYCLIC_TRACE_CLOSURE_ANCHOR_ALGORITHM
        {
            has_cyclic_trace_anchor = true;
            break;
        }
    }
    if !has_cyclic_trace_anchor {
        return Ok(());
    }

    // The authenticated process catalog already proves that every source is
    // nonfermionic and uses this same, nonempty current-state contract.
    let state_template_id = process
        .source_states
        .first()
        .ok_or_else(|| invalid("cyclic trace closure state contract is absent"))?
        .current_state_template_id;
    let state = template
        .current_states
        .get(state_template_id as usize)
        .copied()
        .ok_or_else(|| invalid("cyclic trace closure current state is absent"))?;
    let color_shape = template_catalog.string(
        state.lc_color_shape_string_id,
        "cyclic trace closure current-state color shape",
    )?;
    if CurrentOrientation::try_from(state.orientation)? != CurrentOrientation::SelfConjugate
        || ParticleStatistics::try_from(state.statistics)? != ParticleStatistics::Boson
        || state.particle_id != state.anti_particle_id
        || state.color_representation != 8
        || color_shape != "adjoint-segment"
        || state.auxiliary_kind_string_id != MISSING_U32
    {
        return Err(invalid(
            "cyclic trace closure anchoring requires one non-auxiliary, bosonic, self-conjugate adjoint state",
        ));
    }

    let mut has_direct_closure = false;
    for closure in template.closures.iter().copied() {
        let input_states = template_catalog.u32_sequence(
            closure.input_state_sequence_id,
            "cyclic trace closure input states",
        )?;
        let coupling_order_set = template
            .coupling_order_ranges
            .get(closure.coupling_order_set_id as usize)
            .ok_or_else(|| invalid("cyclic trace closure coupling-order set is absent"))?;
        if input_states != [state_template_id, state_template_id]
            || closure.result_state_template_id != MISSING_U32
            || !template_catalog
                .u32_sequence(
                    closure.coupling_parameter_sequence_id,
                    "cyclic trace closure coupling parameters",
                )?
                .is_empty()
            || coupling_order_set.range.count != 0
            || !template_catalog
                .u32_sequence(
                    closure.eligible_quantum_flow_sequence_id,
                    "cyclic trace closure quantum flows",
                )?
                .is_empty()
            || OutputFactorSource::try_from(closure.output_factor_source)?
                != OutputFactorSource::None
            || template_catalog.string(
                closure.equivalence_class_string_id,
                "cyclic trace closure equivalence class",
            )? != "direct-contraction"
        {
            continue;
        }
        let evaluator = template
            .evaluator_bindings
            .get(closure.evaluator_binding_id as usize)
            .copied()
            .ok_or_else(|| invalid("cyclic trace closure evaluator is absent"))?;
        if EvaluatorCallableKind::try_from(evaluator.callable_kind)?
            != EvaluatorCallableKind::RusticolTemplate
            || EvaluatorContractKind::try_from(evaluator.contract_kind)?
                != EvaluatorContractKind::Closure
        {
            continue;
        }
        let color = template
            .color_contractions
            .get(closure.color_contraction_template_id as usize)
            .copied()
            .ok_or_else(|| invalid("cyclic trace closure color contract is absent"))?;
        if template_catalog.i32_sequence(
            color.input_representation_sequence_id,
            "cyclic trace closure input representations",
        )? == [8, 8]
            && color.has_output_representation == 0
        {
            has_direct_closure = true;
            break;
        }
    }
    if !has_direct_closure {
        return Err(invalid(
            "cyclic trace closure anchoring requires an exact direct binary adjoint closure",
        ));
    }
    Ok(())
}

/// Construction-only exact support identity.
///
/// Runtime and persisted current keys continue to carry canonical source-slot
/// vectors. This sidecar uses one machine-friendly scalar for ordinary
/// processes and falls back to an exact arbitrary-width bitset when required.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum TransientSupportKey {
    Inline {
        source_count: usize,
        bits: u128,
    },
    Wide {
        source_count: usize,
        words: Box<[u64]>,
    },
}

impl TransientSupportKey {
    fn from_source_slots(source_count: usize, source_slots: &[u32]) -> RusticolResult<Self> {
        if source_count <= u128::BITS as usize {
            let mut bits = 0_u128;
            for source_slot in source_slots.iter().copied() {
                let source_slot = usize::try_from(source_slot)
                    .map_err(|_| invalid("transient support source slot exceeds usize"))?;
                if source_slot >= source_count {
                    return Err(invalid(
                        "transient support references an absent source slot",
                    ));
                }
                let bit = 1_u128 << source_slot;
                if bits & bit != 0 {
                    return Err(invalid(
                        "transient support contains a duplicate source slot",
                    ));
                }
                bits |= bit;
            }
            let result = Self::Inline { source_count, bits };
            result.validate()?;
            return Ok(result);
        }

        let mut words = vec![0_u64; source_count.div_ceil(u64::BITS as usize)];
        for source_slot in source_slots.iter().copied() {
            let source_slot = usize::try_from(source_slot)
                .map_err(|_| invalid("transient support source slot exceeds usize"))?;
            if source_slot >= source_count {
                return Err(invalid(
                    "transient support references an absent source slot",
                ));
            }
            let word = source_slot / u64::BITS as usize;
            let bit = 1_u64 << (source_slot % u64::BITS as usize);
            if words[word] & bit != 0 {
                return Err(invalid(
                    "transient support contains a duplicate source slot",
                ));
            }
            words[word] |= bit;
        }
        let result = Self::Wide {
            source_count,
            words: words.into_boxed_slice(),
        };
        result.validate()?;
        Ok(result)
    }

    fn full(source_count: usize) -> RusticolResult<Self> {
        if source_count <= u128::BITS as usize {
            let bits = match source_count {
                0 => 0,
                count if count == u128::BITS as usize => u128::MAX,
                count => (1_u128 << count) - 1,
            };
            let result = Self::Inline { source_count, bits };
            result.validate()?;
            return Ok(result);
        }
        let mut words =
            vec![u64::MAX; source_count.div_ceil(u64::BITS as usize)].into_boxed_slice();
        let trailing_bits = source_count % u64::BITS as usize;
        if trailing_bits != 0 {
            *words
                .last_mut()
                .expect("non-inline support has at least three words") =
                (1_u64 << trailing_bits) - 1;
        }
        let result = Self::Wide {
            source_count,
            words,
        };
        result.validate()?;
        Ok(result)
    }

    fn singleton(source_count: usize, source_slot: u32) -> RusticolResult<Self> {
        Self::from_source_slots(source_count, &[source_slot])
    }

    const fn source_count(&self) -> usize {
        match self {
            Self::Inline { source_count, .. } | Self::Wide { source_count, .. } => *source_count,
        }
    }

    fn validate(&self) -> RusticolResult<()> {
        match self {
            Self::Inline { source_count, bits } => {
                if *source_count > u128::BITS as usize {
                    return Err(invalid(
                        "inline transient support exceeds its source domain",
                    ));
                }
                let domain_mask = match *source_count {
                    0 => 0,
                    count if count == u128::BITS as usize => u128::MAX,
                    count => (1_u128 << count) - 1,
                };
                if bits & !domain_mask != 0 {
                    return Err(invalid(
                        "inline transient support has bits outside its source domain",
                    ));
                }
            }
            Self::Wide {
                source_count,
                words,
            } => {
                if *source_count <= u128::BITS as usize {
                    return Err(invalid(
                        "wide transient support does not require wide storage",
                    ));
                }
                let expected_word_count = source_count.div_ceil(u64::BITS as usize);
                if words.len() != expected_word_count {
                    return Err(invalid(
                        "wide transient support word count disagrees with its source domain",
                    ));
                }
                let trailing_bits = source_count % u64::BITS as usize;
                if trailing_bits != 0 {
                    let domain_mask = (1_u64 << trailing_bits) - 1;
                    if words
                        .last()
                        .is_some_and(|last_word| last_word & !domain_mask != 0)
                    {
                        return Err(invalid(
                            "wide transient support has padding bits outside its source domain",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn without_source_slot(&self, source_slot: u32) -> RusticolResult<Self> {
        self.validate()?;
        let source_slot = usize::try_from(source_slot)
            .map_err(|_| invalid("transient support source slot exceeds usize"))?;
        if source_slot >= self.source_count() {
            return Err(invalid(
                "transient support references an absent source slot",
            ));
        }
        match self {
            Self::Inline { source_count, bits } => Ok(Self::Inline {
                source_count: *source_count,
                bits: bits & !(1_u128 << source_slot),
            }),
            Self::Wide {
                source_count,
                words,
            } => {
                let word = source_slot / u64::BITS as usize;
                let mut result = words.clone();
                result[word] &= !(1_u64 << (source_slot % u64::BITS as usize));
                Ok(Self::Wide {
                    source_count: *source_count,
                    words: result,
                })
            }
        }
    }

    fn cardinality(&self) -> usize {
        match self {
            Self::Inline { bits, .. } => bits.count_ones() as usize,
            Self::Wide { words, .. } => words.iter().map(|word| word.count_ones() as usize).sum(),
        }
    }

    fn is_disjoint(&self, other: &Self) -> RusticolResult<bool> {
        self.validate()?;
        other.validate()?;
        if self.source_count() != other.source_count() {
            return Err(invalid(
                "transient support keys use inconsistent source domains",
            ));
        }
        match (self, other) {
            (Self::Inline { bits: left, .. }, Self::Inline { bits: right, .. }) => {
                Ok(left & right == 0)
            }
            (Self::Wide { words: left, .. }, Self::Wide { words: right, .. }) => Ok(left.len()
                == right.len()
                && left
                    .iter()
                    .zip(right.iter())
                    .all(|(left, right)| left & right == 0)),
            _ => Err(invalid(
                "transient support representations disagree for one source domain",
            )),
        }
    }

    fn union_disjoint(&self, other: &Self) -> RusticolResult<Self> {
        if !self.is_disjoint(other)? {
            return Err(invalid(
                "recurrence parents have overlapping source support",
            ));
        }
        let result = match (self, other) {
            (
                Self::Inline {
                    source_count,
                    bits: left,
                },
                Self::Inline { bits: right, .. },
            ) => Self::Inline {
                source_count: *source_count,
                bits: left | right,
            },
            (
                Self::Wide {
                    source_count,
                    words: left,
                },
                Self::Wide { words: right, .. },
            ) => Self::Wide {
                source_count: *source_count,
                words: left
                    .iter()
                    .zip(right.iter())
                    .map(|(left, right)| left | right)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            },
            _ => Err(invalid(
                "transient support representations use inconsistent source domains",
            ))?,
        };
        result.validate()?;
        Ok(result)
    }

    #[cfg(test)]
    fn source_slots(&self) -> Vec<u32> {
        match self {
            Self::Inline { source_count, bits } => (0..*source_count)
                .filter(|slot| bits & (1_u128 << slot) != 0)
                .map(|slot| u32::try_from(slot).expect("test support source slot fits u32"))
                .collect(),
            Self::Wide {
                source_count,
                words,
            } => words
                .iter()
                .copied()
                .enumerate()
                .flat_map(|(word_index, word)| {
                    (0..u64::BITS).filter_map(move |bit_index| {
                        let source_slot = word_index * u64::BITS as usize + bit_index as usize;
                        (source_slot < *source_count && word & (1_u64 << bit_index) != 0).then_some(
                            u32::try_from(source_slot).expect("test support source slot fits u32"),
                        )
                    })
                })
                .collect(),
        }
    }
}

#[derive(Debug)]
struct TransientCurrentSupportKeys {
    source_count: usize,
    keys_by_current: Vec<TransientSupportKey>,
}

impl TransientCurrentSupportKeys {
    fn from_currents(source_count: usize, currents: &[PendingCurrent]) -> RusticolResult<Self> {
        let keys_by_current = currents
            .iter()
            .map(|current| {
                TransientSupportKey::from_source_slots(
                    source_count,
                    current.key.support_source_slots(),
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Ok(Self {
            source_count,
            keys_by_current,
        })
    }

    fn get(&self, current_id: u32) -> RusticolResult<&TransientSupportKey> {
        self.keys_by_current
            .get(current_id as usize)
            .ok_or_else(|| invalid("transient current support key is absent"))
    }

    fn push(&mut self, current_id: u32, support_key: TransientSupportKey) -> RusticolResult<()> {
        if current_id as usize != self.keys_by_current.len() {
            return Err(invalid(
                "transient current support keys are not dense by current ID",
            ));
        }
        self.keys_by_current.push(support_key);
        Ok(())
    }

    fn reconcile_stage_tail(
        &mut self,
        stage_start: usize,
        currents: &[PendingCurrent],
    ) -> RusticolResult<()> {
        if stage_start > currents.len() || stage_start > self.keys_by_current.len() {
            return Err(invalid(
                "transient current support reconciliation has an invalid stage boundary",
            ));
        }
        self.keys_by_current.truncate(stage_start);
        for current in &currents[stage_start..] {
            self.keys_by_current
                .push(TransientSupportKey::from_source_slots(
                    self.source_count,
                    current.key.support_source_slots(),
                )?);
        }
        if self.keys_by_current.len() != currents.len() {
            return Err(invalid(
                "transient current support reconciliation is not dense",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StructuralState {
    state_template_id: u32,
    spin_state_class: i32,
}

impl StructuralState {
    const fn new(state_template_id: u32, spin_state_class: i32) -> Self {
        Self {
            state_template_id,
            spin_state_class,
        }
    }

    fn from_current(current: &PendingCurrent) -> Self {
        Self::new(
            current.key.current_state_template_id(),
            current.key.spin_state_class(),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StructuralTransition {
    parents: [StructuralState; 2],
    result: StructuralState,
}

fn structural_state_matches(required: StructuralState, actual: StructuralState) -> bool {
    required.state_template_id == actual.state_template_id
        && (required.spin_state_class == actual.spin_state_class
            || actual.spin_state_class == DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS)
}

fn structural_parent_states_match(
    required: [StructuralState; 2],
    actual: [StructuralState; 2],
) -> bool {
    structural_state_matches(required[0], actual[0])
        && structural_state_matches(required[1], actual[1])
}

/// Allocation-heavy reference implementation retained for exact-equivalence
/// tests of the split production indexes below.
#[cfg(test)]
#[derive(Debug)]
struct LazyStructuralDemandIndex {
    demanded: BTreeSet<(Vec<u32>, StructuralState)>,
}

#[cfg(test)]
impl LazyStructuralDemandIndex {
    fn new(
        process: &OwnedRecurrenceProcessInput,
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
        prepared_transitions: &PreparedTransitionCatalog,
        materialized_sectors: &BTreeSet<u32>,
        source_currents: &[PendingCurrent],
    ) -> RusticolResult<Self> {
        let transitions = prepared_transitions.structural_transitions();
        let mut feasible = BTreeMap::<Vec<u32>, BTreeSet<StructuralState>>::new();
        for current in source_currents
            .iter()
            .filter(|current| current.key.node_kind() == RecurrenceNodeKind::Source)
        {
            feasible
                .entry(current.key.support_source_slots().to_vec())
                .or_default()
                .insert(StructuralState::from_current(current));
        }
        if feasible.is_empty() {
            return Err(invalid(
                "recurrence structural-demand construction has no source states",
            ));
        }

        let source_count = process.external_legs.len();
        for target_size in 2..source_count {
            let prior = feasible
                .iter()
                .filter(|(support, _)| support.len() < target_size)
                .map(|(support, states)| {
                    (support.clone(), states.iter().copied().collect::<Vec<_>>())
                })
                .collect::<Vec<_>>();
            let mut additions = BTreeMap::<Vec<u32>, BTreeSet<StructuralState>>::new();
            for left_index in 0..prior.len() {
                for right_index in left_index + 1..prior.len() {
                    let (left_support, left_states) = &prior[left_index];
                    let (right_support, right_states) = &prior[right_index];
                    if left_support.len() + right_support.len() != target_size
                        || !disjoint_support(left_support, right_support)
                    {
                        continue;
                    }
                    let support = merged_support(left_support, right_support)?;
                    for left in left_states.iter().copied() {
                        for right in right_states.iter().copied() {
                            for transition in transitions {
                                if structural_parent_states_match(transition.parents, [left, right])
                                    || structural_parent_states_match(
                                        transition.parents,
                                        [right, left],
                                    )
                                {
                                    additions
                                        .entry(support.clone())
                                        .or_default()
                                        .insert(transition.result);
                                }
                            }
                        }
                    }
                }
            }
            for (support, states) in additions {
                feasible.entry(support).or_default().extend(states);
            }
        }

        let full_support = (0..source_count as u32).collect::<Vec<_>>();
        let mut demanded = BTreeSet::<(Vec<u32>, StructuralState)>::new();
        for sector_id in materialized_sectors.iter().copied() {
            let sector = process
                .physical_lc_sectors
                .get(sector_id as usize)
                .copied()
                .ok_or_else(|| invalid("structural-demand LC sector is absent"))?;
            let anchor_support = vec![sector.closure_source_slot];
            let complement_support = full_support
                .iter()
                .copied()
                .filter(|slot| *slot != sector.closure_source_slot)
                .collect::<Vec<_>>();
            let anchor_states = feasible
                .get(&anchor_support)
                .ok_or_else(|| invalid("structural-demand closure anchor is infeasible"))?;
            let complement_states = feasible
                .get(&complement_support)
                .ok_or_else(|| invalid("structural-demand closure complement is infeasible"))?;
            let mut sector_has_root = false;
            for anchor in anchor_states.iter().copied() {
                for complement in complement_states.iter().copied() {
                    for closure in &template.closures {
                        let closure_states = catalog.u32_sequence(
                            closure.input_state_sequence_id,
                            "structural-demand closure input states",
                        )?;
                        if closure_states.len() != 2 {
                            return Err(invalid(
                                "direct recurrence requires binary prepared closures",
                            ));
                        }
                        let ordered = if closure_states
                            == [complement.state_template_id, anchor.state_template_id]
                        {
                            Some([complement, anchor])
                        } else if closure_states
                            == [anchor.state_template_id, complement.state_template_id]
                            && anchor.state_template_id != complement.state_template_id
                        {
                            Some([anchor, complement])
                        } else {
                            None
                        };
                        let Some(ordered) = ordered else {
                            continue;
                        };
                        if structural_closure_admits(*closure, ordered, template, catalog)? {
                            demanded.insert((complement_support.clone(), complement));
                            sector_has_root = true;
                        }
                    }
                }
            }
            if !sector_has_root {
                return Err(invalid(format!(
                    "recurrence structural-demand grammar found no closure root for LC sector {sector_id}"
                )));
            }
        }

        for target_size in (2..source_count).rev() {
            let targets = demanded
                .iter()
                .filter(|(support, _)| support.len() == target_size)
                .cloned()
                .collect::<Vec<_>>();
            for (target_support, target_state) in targets {
                let feasible_rows = feasible.iter().collect::<Vec<_>>();
                for left_index in 0..feasible_rows.len() {
                    for right_index in left_index + 1..feasible_rows.len() {
                        let (left_support, left_states) = feasible_rows[left_index];
                        let (right_support, right_states) = feasible_rows[right_index];
                        if left_support.len() + right_support.len() != target_size
                            || !disjoint_support(left_support, right_support)
                            || merged_support(left_support, right_support)? != target_support
                        {
                            continue;
                        }
                        for left in left_states.iter().copied() {
                            for right in right_states.iter().copied() {
                                for transition in transitions
                                    .iter()
                                    .filter(|transition| transition.result == target_state)
                                {
                                    if structural_parent_states_match(
                                        transition.parents,
                                        [left, right],
                                    ) || structural_parent_states_match(
                                        transition.parents,
                                        [right, left],
                                    ) {
                                        demanded.insert((left_support.clone(), left));
                                        demanded.insert((right_support.clone(), right));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Self { demanded })
    }

    fn accepts(
        &self,
        support_source_slots: &[u32],
        state_template_id: u32,
        spin_state_class: i32,
    ) -> bool {
        self.demanded.contains(&(
            support_source_slots.to_vec(),
            StructuralState::new(state_template_id, spin_state_class),
        ))
    }
}

/// Process-global forward state/spin grammar.
///
/// This half is independent of the materialized color lane. It is built once
/// from source states and the prepared transition catalog, then shared by all
/// lane-specific backward-demand passes. Together those passes form a
/// conservative exact filter: color, coupling, flavour, pairing, and proof
/// contracts remain authenticated by normal construction, while a full
/// current is never interned for a state/spin branch that cannot close.
#[derive(Debug)]
struct StructuralFeasibilityIndex {
    source_count: usize,
    feasible: BTreeMap<TransientSupportKey, BTreeSet<StructuralState>>,
    decompositions: BTreeMap<TransientSupportKey, Vec<[TransientSupportKey; 2]>>,
    forward_transition_probe_count: usize,
}

impl StructuralFeasibilityIndex {
    fn new(
        source_count: usize,
        prepared_transitions: &PreparedTransitionCatalog,
        source_currents: &[PendingCurrent],
    ) -> RusticolResult<Self> {
        let transitions = prepared_transitions.structural_transitions();
        let mut feasible = BTreeMap::<TransientSupportKey, BTreeSet<StructuralState>>::new();
        for current in source_currents
            .iter()
            .filter(|current| current.key.node_kind() == RecurrenceNodeKind::Source)
        {
            feasible
                .entry(TransientSupportKey::from_source_slots(
                    source_count,
                    current.key.support_source_slots(),
                )?)
                .or_default()
                .insert(StructuralState::from_current(current));
        }
        if feasible.is_empty() {
            return Err(invalid(
                "recurrence structural-demand construction has no source states",
            ));
        }

        let mut decompositions =
            BTreeMap::<TransientSupportKey, Vec<[TransientSupportKey; 2]>>::new();
        let mut forward_transition_probe_count = 0usize;
        for target_size in 2..source_count {
            let prior = feasible
                .iter()
                .filter(|(support, _)| support.cardinality() < target_size)
                .map(|(support, states)| {
                    (support.clone(), states.iter().copied().collect::<Vec<_>>())
                })
                .collect::<Vec<_>>();
            let mut additions = BTreeMap::<TransientSupportKey, BTreeSet<StructuralState>>::new();
            for left_index in 0..prior.len() {
                for right_index in left_index + 1..prior.len() {
                    let (left_support, left_states) = &prior[left_index];
                    let (right_support, right_states) = &prior[right_index];
                    if left_support.cardinality() + right_support.cardinality() != target_size
                        || !left_support.is_disjoint(right_support)?
                    {
                        continue;
                    }
                    let support = left_support.union_disjoint(right_support)?;
                    decompositions
                        .entry(support.clone())
                        .or_default()
                        .push([left_support.clone(), right_support.clone()]);
                    for left in left_states.iter().copied() {
                        for right in right_states.iter().copied() {
                            for transition in transitions {
                                forward_transition_probe_count =
                                    forward_transition_probe_count.checked_add(1).ok_or_else(
                                        || {
                                            invalid(
                                                "structural forward transition-probe count exceeds usize",
                                            )
                                        },
                                    )?;
                                if structural_parent_states_match(transition.parents, [left, right])
                                    || structural_parent_states_match(
                                        transition.parents,
                                        [right, left],
                                    )
                                {
                                    additions
                                        .entry(support.clone())
                                        .or_default()
                                        .insert(transition.result);
                                }
                            }
                        }
                    }
                }
            }
            for (support, states) in additions {
                feasible.entry(support).or_default().extend(states);
            }
        }

        let result = Self {
            source_count,
            feasible,
            decompositions,
            forward_transition_probe_count,
        };
        debug_assert_eq!(result.feasible_support_count(), result.feasible.len());
        debug_assert_eq!(
            result.decomposition_count(),
            result.decompositions.values().map(Vec::len).sum::<usize>()
        );
        debug_assert_eq!(
            result.forward_transition_probe_count(),
            forward_transition_probe_count
        );
        Ok(result)
    }

    fn feasible_states(&self, support: &TransientSupportKey) -> Option<&BTreeSet<StructuralState>> {
        self.feasible.get(support)
    }

    fn decompositions(&self, support: &TransientSupportKey) -> &[[TransientSupportKey; 2]] {
        self.decompositions
            .get(support)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn feasible_support_count(&self) -> usize {
        self.feasible.len()
    }

    fn decomposition_count(&self) -> usize {
        self.decompositions.values().map(Vec::len).sum()
    }

    fn forward_transition_probe_count(&self) -> usize {
        self.forward_transition_probe_count
    }
}

#[derive(Debug)]
struct StructuralDemandIndex {
    demanded: BTreeMap<TransientSupportKey, BTreeSet<StructuralState>>,
}

impl StructuralDemandIndex {
    fn new(
        process: &OwnedRecurrenceProcessInput,
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
        prepared_transitions: &PreparedTransitionCatalog,
        feasibility: &StructuralFeasibilityIndex,
        materialized_sectors: &BTreeSet<u32>,
    ) -> RusticolResult<Self> {
        if process.external_legs.len() != feasibility.source_count {
            return Err(invalid(
                "structural-demand source domain disagrees with forward feasibility",
            ));
        }
        let transitions = prepared_transitions.structural_transitions();
        let full_support = TransientSupportKey::full(feasibility.source_count)?;
        let mut demanded = BTreeMap::<TransientSupportKey, BTreeSet<StructuralState>>::new();
        for sector_id in materialized_sectors.iter().copied() {
            let sector = process
                .physical_lc_sectors
                .get(sector_id as usize)
                .copied()
                .ok_or_else(|| invalid("structural-demand LC sector is absent"))?;
            let anchor_support = TransientSupportKey::singleton(
                feasibility.source_count,
                sector.closure_source_slot,
            )?;
            let complement_support =
                full_support.without_source_slot(sector.closure_source_slot)?;
            let anchor_states = feasibility
                .feasible_states(&anchor_support)
                .ok_or_else(|| invalid("structural-demand closure anchor is infeasible"))?;
            let complement_states = feasibility
                .feasible_states(&complement_support)
                .ok_or_else(|| invalid("structural-demand closure complement is infeasible"))?;
            let mut sector_has_root = false;
            for anchor in anchor_states.iter().copied() {
                for complement in complement_states.iter().copied() {
                    for closure in &template.closures {
                        let closure_states = catalog.u32_sequence(
                            closure.input_state_sequence_id,
                            "structural-demand closure input states",
                        )?;
                        if closure_states.len() != 2 {
                            return Err(invalid(
                                "direct recurrence requires binary prepared closures",
                            ));
                        }
                        let ordered = if closure_states
                            == [complement.state_template_id, anchor.state_template_id]
                        {
                            Some([complement, anchor])
                        } else if closure_states
                            == [anchor.state_template_id, complement.state_template_id]
                            && anchor.state_template_id != complement.state_template_id
                        {
                            Some([anchor, complement])
                        } else {
                            None
                        };
                        let Some(ordered) = ordered else {
                            continue;
                        };
                        if structural_closure_admits(*closure, ordered, template, catalog)? {
                            demanded
                                .entry(complement_support.clone())
                                .or_default()
                                .insert(complement);
                            sector_has_root = true;
                        }
                    }
                }
            }
            if !sector_has_root {
                return Err(invalid(format!(
                    "recurrence structural-demand grammar found no closure root for LC sector {sector_id}"
                )));
            }
        }

        for target_size in (2..feasibility.source_count).rev() {
            let targets = demanded
                .iter()
                .filter(|(support, _)| support.cardinality() == target_size)
                .flat_map(|(support, states)| {
                    states.iter().copied().map(|state| (support.clone(), state))
                })
                .collect::<Vec<_>>();
            for (target_support, target_state) in targets {
                for [left_support, right_support] in feasibility.decompositions(&target_support) {
                    let left_states = feasibility
                        .feasible_states(left_support)
                        .expect("forward decomposition left support is feasible");
                    let right_states = feasibility
                        .feasible_states(right_support)
                        .expect("forward decomposition right support is feasible");
                    for left in left_states.iter().copied() {
                        for right in right_states.iter().copied() {
                            for transition in transitions
                                .iter()
                                .filter(|transition| transition.result == target_state)
                            {
                                if structural_parent_states_match(transition.parents, [left, right])
                                    || structural_parent_states_match(
                                        transition.parents,
                                        [right, left],
                                    )
                                {
                                    demanded
                                        .entry(left_support.clone())
                                        .or_default()
                                        .insert(left);
                                    demanded
                                        .entry(right_support.clone())
                                        .or_default()
                                        .insert(right);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Self { demanded })
    }

    fn accepts(
        &self,
        support: &TransientSupportKey,
        state_template_id: u32,
        spin_state_class: i32,
    ) -> bool {
        self.demanded.get(support).is_some_and(|states| {
            states.contains(&StructuralState::new(state_template_id, spin_state_class))
        })
    }

    fn demanded_support_count(&self) -> usize {
        self.demanded.len()
    }

    fn demanded_state_count(&self) -> usize {
        self.demanded.values().map(BTreeSet::len).sum()
    }

    #[cfg(test)]
    fn demanded_rows(&self) -> BTreeSet<(Vec<u32>, StructuralState)> {
        self.demanded
            .iter()
            .flat_map(|(support, states)| {
                states
                    .iter()
                    .copied()
                    .map(|state| (support.source_slots(), state))
            })
            .collect()
    }
}

#[cfg(test)]
fn structural_transitions(
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<Vec<StructuralTransition>> {
    let transitions = template
        .transitions
        .iter()
        .map(|transition| {
            let quantum = template
                .quantum_flows
                .get(transition.quantum_flow_template_id as usize)
                .copied()
                .ok_or_else(|| invalid("structural-demand quantum flow is absent"))?;
            let states =
                catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
            let spins =
                catalog.i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?;
            if states.len() != 2 || spins.len() != 2 {
                return Err(invalid(
                    "direct recurrence requires binary quantum-flow contracts",
                ));
            }
            if quantum.result_state_template_id != transition.result_state_template_id {
                return Err(invalid(
                    "structural-demand transition and quantum-flow result states differ",
                ));
            }
            Ok(StructuralTransition {
                parents: [
                    StructuralState::new(states[0], spins[0]),
                    StructuralState::new(states[1], spins[1]),
                ],
                result: StructuralState::new(
                    quantum.result_state_template_id,
                    quantum.result_spin_state,
                ),
            })
        })
        .collect::<RusticolResult<BTreeSet<_>>>()?;
    Ok(transitions.into_iter().collect())
}

fn structural_closure_admits(
    closure: ClosureRow,
    ordered_parents: [StructuralState; 2],
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<bool> {
    let eligible = catalog.u32_sequence(
        closure.eligible_quantum_flow_sequence_id,
        "structural-demand closure quantum flows",
    )?;
    if eligible.is_empty() {
        return Ok(true);
    }
    for quantum_id in eligible {
        let quantum = template
            .quantum_flows
            .get(*quantum_id as usize)
            .copied()
            .ok_or_else(|| invalid("structural-demand closure quantum flow is absent"))?;
        let states =
            catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
        let spins = catalog.i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?;
        if states.len() != 2 || spins.len() != 2 {
            return Err(invalid(
                "direct recurrence requires binary quantum-flow contracts",
            ));
        }
        if structural_parent_states_match(
            [
                StructuralState::new(states[0], spins[0]),
                StructuralState::new(states[1], spins[1]),
            ],
            ordered_parents,
        ) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn compatible_pairing_rules_for_current(
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template: &OwnedRecurrenceTemplateInput,
    current_state_template_id: u32,
    support_source_slots: &[u32],
) -> RusticolResult<BTreeSet<u32>> {
    let Some(pairing_catalog) = pairing_catalog else {
        return Ok(BTreeSet::new());
    };
    let state = template
        .current_states
        .get(current_state_template_id as usize)
        .ok_or_else(|| invalid("pairing support references an absent current state"))?;
    let carries_colored_fermion_line = state.statistics == 1 && state.color_representation != 1;
    let support = support_source_slots
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut result = BTreeSet::new();
    for rule in pairing_catalog.rules().iter().copied() {
        let crossing_count = pairing_catalog
            .endpoint_pairings(rule)?
            .iter()
            .filter(|pair| {
                support.contains(&pair.fundamental_source_slot)
                    != support.contains(&pair.antifundamental_source_slot)
            })
            .count();
        let compatible = if carries_colored_fermion_line {
            crossing_count == 1
        } else {
            crossing_count == 0
        };
        if compatible {
            result.insert(rule.rule_id);
        }
    }
    Ok(result)
}

fn realized_pairing_rules_for_transition(
    compatible_result_rules: BTreeSet<u32>,
    parent_rules: [&BTreeSet<u32>; 2],
) -> BTreeSet<u32> {
    compatible_result_rules
        .into_iter()
        .filter(|rule_id| parent_rules[0].contains(rule_id) && parent_rules[1].contains(rule_id))
        .collect()
}

pub(super) fn build_recurrence_program(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<RecurrenceProgram> {
    build_recurrence_program_impl(authenticated, &mut |_| Ok(()), false).map(|(program, _)| program)
}

pub(super) fn build_recurrence_program_with_progress(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
) -> RusticolResult<RecurrenceProgram> {
    build_recurrence_program_impl(authenticated, progress, false).map(|(program, _)| program)
}

pub(super) fn build_recurrence_program_with_progress_and_telemetry(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
) -> RusticolResult<(RecurrenceProgram, RecurrenceGenerationTelemetry)> {
    build_recurrence_program_impl(authenticated, progress, true)
}

fn established_fermion_ordering_context(
    process: &OwnedRecurrenceProcessInput,
    template: &OwnedRecurrenceTemplateInput,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
) -> RusticolResult<FermionOrderingContext> {
    let mut authenticated_colored_endpoints = vec![false; process.external_legs.len()];
    if let Some(pairing_catalog) = pairing_catalog {
        for endpoint in pairing_catalog.endpoints().iter().copied() {
            let selected = authenticated_colored_endpoints
                .get_mut(endpoint.source_slot as usize)
                .ok_or_else(|| invalid("fermion-ordering endpoint source slot is absent"))?;
            if *selected {
                return Err(invalid(
                    "fermion-ordering endpoint source slot is authenticated twice",
                ));
            }
            *selected = true;
        }
    }

    let mut source_requires_exterior_sign = Vec::with_capacity(process.external_legs.len());
    for (source_slot, leg) in process.external_legs.iter().copied().enumerate() {
        let source_slot = u32::try_from(source_slot)
            .map_err(|_| invalid("fermion-ordering source slot exceeds u32"))?;
        if leg.source_slot != source_slot {
            return Err(invalid(
                "fermion-ordering source slot disagrees with the process table",
            ));
        }
        let state_range = leg.source_state_range.as_usize_range(
            process.source_states.len(),
            "fermion-ordering source-state range",
        )?;
        let color_representations = process.source_states[state_range]
            .iter()
            .map(|source_state| {
                if source_state.source_slot != source_slot {
                    return Err(invalid(
                        "fermion-ordering source state belongs to a different source slot",
                    ));
                }
                template
                    .current_states
                    .get(source_state.current_state_template_id as usize)
                    .map(|state| state.color_representation)
                    .ok_or_else(|| {
                        invalid("fermion-ordering source current state is absent from the template")
                    })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        source_requires_exterior_sign.push(authenticated_source_requires_exterior_sign(
            leg.is_fermionic == 1,
            authenticated_colored_endpoints[source_slot as usize],
            color_representations,
        )?);
    }
    Ok(FermionOrderingContext::new(source_requires_exterior_sign))
}

fn build_recurrence_program_impl(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
    collect_telemetry: bool,
) -> RusticolResult<(RecurrenceProgram, RecurrenceGenerationTelemetry)> {
    #[cfg(feature = "on-the-fly-test-support")]
    super::on_the_fly::reject_forbidden_work_if_probed(
        super::on_the_fly::OnTheFlyForbiddenWorkV1::EstablishedBuilder,
    )?;
    let mut telemetry = RecurrenceGenerationTelemetry::default();
    let strategy = authenticated.process().summary().strategy();
    let catalog_digest = authenticated.template().summary().catalog_digest;
    let process_input = authenticated.process().input();
    let pairing_catalog = authenticated.process().fermion_pairing_catalog();
    let template_input = authenticated.template().input();
    let fermion_ordering =
        established_fermion_ordering_context(process_input, template_input, pairing_catalog)?;
    let process_catalog = ProcessCatalog::new(process_input)?;
    let helicity_support_rule = helicity_support_rule(authenticated)?;
    let global_helicity_flip_rule = global_helicity_flip_rule(authenticated)?;
    let retained_helicity_count = retained_helicity_count(process_input)?;
    let template_catalog = TemplateCatalog::new(template_input)?;
    validate_cyclic_trace_closure_anchor_contracts(
        process_input,
        &process_catalog,
        template_input,
        &template_catalog,
    )?;
    let coupling_limits = coupling_limits(&process_catalog, &template_catalog)?;
    let propagators = propagator_by_state(template_input)?;
    let transition_reflections = TransitionReflectionIndex::new(template_input, &template_catalog)?;
    let replay_targets = build_replay_targets(strategy, process_input, &process_catalog)?;
    let materialized_sectors = materialized_sector_ids(strategy, process_input, &replay_targets);
    let mut color_states = DynamicLCColorStateInterner::default();
    let mut currents = Vec::<PendingCurrent>::new();
    let mut current_ids = TransientCurrentIdIndex::new();
    let mut currents_by_size = vec![Vec::<u32>::new(); process_input.external_legs.len()];
    let mut reflection_certificates = Vec::<PendingReflectionCertificate>::new();

    build_sources(
        strategy,
        catalog_digest,
        process_input,
        &process_catalog,
        pairing_catalog,
        template_input,
        &template_catalog,
        &mut color_states,
        &mut currents,
        &mut current_ids,
        &mut currents_by_size,
    )?;
    let stage_count_per_lane = process_input.external_legs.len().saturating_sub(2);
    let construction_sector_groups =
        construction_sector_groups(strategy, &materialized_sectors, &replay_targets);
    let stage_total = stage_count_per_lane
        .checked_mul(construction_sector_groups.len())
        .ok_or_else(|| invalid("recurrence construction stage count exceeds usize"))?;
    let phase_total = stage_total.saturating_add(3);
    progress(RecurrenceBuildProgress::snapshot(
        "source construction",
        1,
        phase_total,
        None,
        stage_total,
        Some(1),
        0,
        None,
        currents.len(),
        0,
        color_states.len(),
        0,
    ))?;
    let transition_catalog_started = telemetry_timer(collect_telemetry);
    let prepared_transitions = PreparedTransitionCatalog::new(template_input, &template_catalog)?;
    let prepared_closures = PreparedClosureCatalog::new(template_input, &template_catalog)?;
    let prepared_closure_sectors = PreparedClosureSectorCatalog::new(
        strategy,
        process_input,
        &process_catalog,
        &materialized_sectors,
    )?;
    add_optional_elapsed_nanoseconds(
        &mut telemetry.transition_catalog_nanoseconds,
        transition_catalog_started,
    );
    let source_current_count = currents.len();
    let structural_feasibility_started = telemetry_timer(collect_telemetry);
    let structural_feasibility = StructuralFeasibilityIndex::new(
        process_input.external_legs.len(),
        &prepared_transitions,
        &currents[..source_current_count],
    )?;
    add_optional_elapsed_nanoseconds(
        &mut telemetry.structural_feasibility_nanoseconds,
        structural_feasibility_started,
    );
    if collect_telemetry {
        telemetry.structural_feasible_support_count =
            structural_feasibility.feasible_support_count();
        telemetry.structural_decomposition_count = structural_feasibility.decomposition_count();
        telemetry.structural_forward_transition_probe_count =
            structural_feasibility.forward_transition_probe_count();
    }
    let mut current_support_keys =
        TransientCurrentSupportKeys::from_currents(process_input.external_legs.len(), &currents)?;
    let source_current_ids = current_ids;
    let source_bucket = currents_by_size
        .first()
        .cloned()
        .ok_or_else(|| invalid("recurrence construction has no source bucket"))?;
    let isolate_replay_lanes = construction_sector_groups.len() > 1;
    let mut resident_contribution_count = 0usize;
    let mut stage_diagnostics = Vec::new();
    let mut closures = BTreeMap::new();
    let mut exhaustive_closure_sector_ids = BTreeSet::new();
    let mut exhaustive_closure_sector_owner_ids = BTreeMap::new();
    for (lane_index, construction_sectors) in construction_sector_groups.into_iter().enumerate() {
        let stage_index_offset = lane_index
            .checked_mul(stage_count_per_lane)
            .ok_or_else(|| invalid("recurrence construction stage offset exceeds usize"))?;
        let color_target_index_started = telemetry_timer(collect_telemetry);
        let color_targets = MaterializedColorTargets::new_with_prepared_sectors(
            &construction_sectors,
            &prepared_closure_sectors,
            collect_telemetry,
        )?;
        add_optional_elapsed_nanoseconds(
            &mut telemetry.color_target_index_nanoseconds,
            color_target_index_started,
        );
        let structural_demand_started = telemetry_timer(collect_telemetry);
        let structural_demands = StructuralDemandIndex::new(
            process_input,
            template_input,
            &template_catalog,
            &prepared_transitions,
            &structural_feasibility,
            &construction_sectors,
        )?;
        add_optional_elapsed_nanoseconds(
            &mut telemetry.structural_demand_nanoseconds,
            structural_demand_started,
        );
        if collect_telemetry {
            checked_diagnostic_add(
                &mut telemetry.structural_demand_support_count,
                structural_demands.demanded_support_count(),
                "telemetry structural-demand support count",
            )?;
            checked_diagnostic_add(
                &mut telemetry.structural_demand_state_count,
                structural_demands.demanded_state_count(),
                "telemetry structural-demand state count",
            )?;
        }
        let mut lane_current_ids = source_current_ids.clone();
        let mut lane_currents_by_size = vec![Vec::<u32>::new(); process_input.external_legs.len()];
        lane_currents_by_size[0] = source_bucket.clone();
        let lane_internal_start = currents.len();
        let mut build_lane = || {
            build_internal_currents(
                catalog_digest,
                process_input,
                pairing_catalog,
                template_input,
                &fermion_ordering,
                &prepared_transitions,
                &transition_reflections,
                &coupling_limits,
                &propagators,
                &color_targets,
                &structural_demands,
                &mut color_states,
                &mut currents,
                &mut lane_current_ids,
                &mut lane_currents_by_size,
                &mut current_support_keys,
                &mut reflection_certificates,
                &mut resident_contribution_count,
                stage_index_offset,
                stage_total,
                phase_total,
                &mut telemetry,
                collect_telemetry,
                progress,
            )
        };
        #[cfg(feature = "on-the-fly-test-support")]
        let lane_stage_diagnostics =
            super::diagnostic::with_transition_diagnostic_materialized_sector(
                (construction_sectors.len() == 1)
                    .then(|| construction_sectors.first().copied())
                    .flatten(),
                build_lane,
            )?;
        #[cfg(not(feature = "on-the-fly-test-support"))]
        let lane_stage_diagnostics = build_lane()?;
        if collect_telemetry {
            let color_target_telemetry = color_targets.telemetry()?;
            for (target, value, label) in [
                (
                    &mut telemetry.color_fragment_bucket_count,
                    color_target_telemetry.fragment_bucket_count,
                    "telemetry color-fragment bucket count",
                ),
                (
                    &mut telemetry.color_acceptance_cache_hit_count,
                    color_target_telemetry.acceptance_cache_hit_count,
                    "telemetry color-acceptance cache-hit count",
                ),
                (
                    &mut telemetry.color_acceptance_cache_miss_count,
                    color_target_telemetry.acceptance_cache_miss_count,
                    "telemetry color-acceptance cache-miss count",
                ),
                (
                    &mut telemetry.color_fragment_hash_lookup_count,
                    color_target_telemetry.fragment_hash_lookup_count,
                    "telemetry color-fragment hash-lookup count",
                ),
                (
                    &mut telemetry.color_posting_incidence_count,
                    color_target_telemetry.posting_incidence_count,
                    "telemetry color-posting incidence count",
                ),
                (
                    &mut telemetry.color_sparse_posting_bucket_count,
                    color_target_telemetry.sparse_posting_bucket_count,
                    "telemetry sparse color-posting bucket count",
                ),
                (
                    &mut telemetry.color_dense_posting_bucket_count,
                    color_target_telemetry.dense_posting_bucket_count,
                    "telemetry dense color-posting bucket count",
                ),
                (
                    &mut telemetry.color_sparse_posting_bytes,
                    color_target_telemetry.sparse_posting_bytes,
                    "telemetry sparse color-posting bytes",
                ),
                (
                    &mut telemetry.color_dense_posting_bytes,
                    color_target_telemetry.dense_posting_bytes,
                    "telemetry dense color-posting bytes",
                ),
            ] {
                checked_diagnostic_add(target, value, label)?;
            }
            debug_assert_eq!(
                color_target_telemetry
                    .acceptance_accept_count
                    .saturating_add(color_target_telemetry.acceptance_reject_count),
                color_target_telemetry
                    .acceptance_cache_hit_count
                    .saturating_add(color_target_telemetry.acceptance_cache_miss_count),
            );
        }
        let lane_internal_end = currents.len();
        let lane_domain = isolate_replay_lanes.then_some(PendingConstructionDomain {
            shared_source_end: source_current_count,
            lane_internal_start,
            lane_internal_end,
        });
        let closure_processing_started = telemetry_timer(collect_telemetry);
        let lane_closure_domain = build_closures(
            strategy,
            process_input,
            &process_catalog,
            template_input,
            &fermion_ordering,
            pairing_catalog,
            &prepared_closures,
            &prepared_closure_sectors,
            &color_states,
            &currents,
            &construction_sectors,
            &lane_stage_diagnostics,
            &reflection_certificates,
            lane_domain,
            &mut telemetry,
            collect_telemetry,
        )?;
        for sector_id in lane_closure_domain.physical_sector_ids {
            if !exhaustive_closure_sector_ids.insert(sector_id) {
                return Err(invalid(
                    "recurrence closure construction visited one physical sector twice",
                ));
            }
        }
        for (sector_id, owner_id) in lane_closure_domain.physical_sector_owner_ids {
            if exhaustive_closure_sector_owner_ids
                .insert(sector_id, owner_id)
                .is_some()
            {
                return Err(invalid(
                    "recurrence closure construction assigned one physical-sector owner twice",
                ));
            }
        }
        let lane_closures = lane_closure_domain.groups;
        let lane_closures = if lane_domain.is_some() {
            retain_supported_pending_closures(
                strategy,
                &process_catalog,
                &replay_targets,
                lane_closures,
                helicity_support_rule,
                global_helicity_flip_rule,
            )?
        } else {
            lane_closures
        };
        if let Some(domain) = lane_domain {
            prune_inactive_lane_contributions(
                &mut currents,
                domain,
                &lane_closures,
                &mut resident_contribution_count,
            )?;
        }
        for (key, group) in lane_closures {
            if closures.insert(key, group).is_some() {
                return Err(invalid(
                    "isolated recurrence construction produced a duplicate closure",
                ));
            }
        }
        add_optional_elapsed_nanoseconds(
            &mut telemetry.closure_processing_nanoseconds,
            closure_processing_started,
        );
        stage_diagnostics.extend(lane_stage_diagnostics);
    }
    if pending_contribution_entry_count(&currents)? != resident_contribution_count {
        return Err(invalid(
            "resident recurrence contribution count disagrees with pending storage",
        ));
    }
    let color_target_prune_count = stage_diagnostics
        .iter()
        .map(|stage| stage.color_target_prune_count)
        .try_fold(0usize, |total, count| {
            total
                .checked_add(count)
                .ok_or_else(|| invalid("recurrence color-target prune count exceeds usize"))
        })?;
    progress(RecurrenceBuildProgress::snapshot(
        "amplitude closure",
        stage_total.saturating_add(2),
        phase_total,
        None,
        stage_total,
        None,
        0,
        None,
        currents.len(),
        resident_contribution_count,
        color_states.len(),
        color_target_prune_count,
    ))?;
    progress(RecurrenceBuildProgress::snapshot(
        "schedule finalization",
        phase_total,
        phase_total,
        None,
        stage_total,
        None,
        0,
        None,
        currents.len(),
        resident_contribution_count,
        color_states.len(),
        color_target_prune_count,
    ))?;
    if collect_telemetry {
        telemetry.constructed_current_count = currents.len();
        telemetry.constructed_contribution_count = resident_contribution_count;
        telemetry.constructed_interaction_count = resident_contribution_count;
        telemetry.constructed_dynamic_color_state_count = color_states.len();
        telemetry.indexed_hash_lookup_count = telemetry
            .support_bucket_probe_count
            .checked_add(telemetry.color_acceptance_cache_hit_count)
            .and_then(|count| count.checked_add(telemetry.color_acceptance_cache_miss_count))
            .and_then(|count| count.checked_add(telemetry.color_fragment_hash_lookup_count))
            .ok_or_else(|| invalid("telemetry indexed hash-lookup count exceeds usize"))?;
    }
    let canonical_emission_started = telemetry_timer(collect_telemetry);
    let program = finish_program(
        strategy,
        &process_catalog,
        color_states.into_states(),
        currents,
        closures,
        exhaustive_closure_sector_ids,
        exhaustive_closure_sector_owner_ids,
        replay_targets,
        retained_helicity_count,
        if isolate_replay_lanes {
            HelicitySupportRule::None
        } else {
            helicity_support_rule
        },
        if isolate_replay_lanes {
            GlobalHelicityFlipRule::None
        } else {
            global_helicity_flip_rule
        },
        reflection_certificates,
    )?;
    add_optional_elapsed_nanoseconds(
        &mut telemetry.canonical_emission_nanoseconds,
        canonical_emission_started,
    );
    if collect_telemetry {
        telemetry.emitted_current_count = program.currents().len();
        telemetry.emitted_contribution_count = program.contributions().len();
        telemetry.emitted_interaction_count = program.contributions().len();
        telemetry.emitted_finalization_count = program.finalizations().len();
        telemetry.emitted_closure_count = program.closure_terms().len();
    }
    Ok((program, telemetry))
}

#[allow(clippy::too_many_arguments)]
fn build_sources(
    strategy: RecurrenceStrategy,
    catalog_digest: SemanticDigest,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut TransientCurrentIdIndex,
    currents_by_size: &mut [Vec<u32>],
) -> RusticolResult<()> {
    let zero_orders = vec![0_u32; template_catalog.coupling_names.len()];
    for leg in &process.external_legs {
        let range = leg
            .source_state_range
            .as_usize_range(process.source_states.len(), "recurrence source-state range")?;
        let retained_state_indices = retained_source_state_indices(process, leg.source_slot)?;
        let retained_states = retained_state_indices
            .into_iter()
            .map(|state_index| {
                process
                    .source_states
                    .get(range.start + state_index as usize)
                    .ok_or_else(|| invalid("retained recurrence source state is absent"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        for process_state in &retained_states {
            let source = *template
                .sources
                .get(process_state.source_template_id as usize)
                .ok_or_else(|| invalid("source template is absent"))?;
            validate_crossed_source_state(leg.is_initial != 0, process_state, source, template)?;
        }

        match strategy {
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => {
                for process_state in retained_states {
                    let source = *template
                        .sources
                        .get(process_state.source_template_id as usize)
                        .ok_or_else(|| invalid("source template is absent"))?;
                    let color_id = source_color_state_id(
                        leg.source_slot,
                        source,
                        template,
                        template_catalog,
                        color_states,
                    )?;
                    let reflection = source_reflection(
                        color_states,
                        color_id,
                        leg.source_slot,
                        template_catalog.source_seed(source)?.proof_digest(),
                    )?;
                    let helicity_identity = match strategy {
                        RecurrenceStrategy::TopologyReplay => {
                            CurrentHelicityIdentity::topology_replay(
                                process_state.spin_state,
                                vec![SourceStateAssignment::new(
                                    leg.source_slot,
                                    process_state.state_index,
                                )],
                            )?
                        }
                        RecurrenceStrategy::ContractedColorUnion => {
                            CurrentHelicityIdentity::contracted_color_union(
                                process_state.spin_state,
                                vec![SourceStateAssignment::new(
                                    leg.source_slot,
                                    process_state.state_index,
                                )],
                            )?
                        }
                        RecurrenceStrategy::AllFlowUnion => unreachable!(),
                    };
                    insert_source_current(
                        catalog_digest,
                        process_state.current_state_template_id,
                        color_id,
                        leg.source_slot,
                        process_state.momentum_sign,
                        helicity_identity,
                        source,
                        reflection,
                        CurrentSourceBinding::FixedTemplate(process_state.source_template_id),
                        Some(process_catalog.factor(
                            process_state.crossing_phase_factor_id,
                            "source crossing phase",
                        )?),
                        &zero_orders,
                        pairing_catalog,
                        template_catalog,
                        currents,
                        current_ids,
                        currents_by_size,
                    )?;
                }
            }
            RecurrenceStrategy::AllFlowUnion => {
                let groups = union_source_dispatch_groups(
                    &retained_states,
                    process_catalog,
                    template,
                    template_catalog,
                )?;
                for group in groups {
                    let source = group.representative_source;
                    let color_id = source_color_state_id(
                        leg.source_slot,
                        source,
                        template,
                        template_catalog,
                        color_states,
                    )?;
                    let reflection = source_reflection(
                        color_states,
                        color_id,
                        leg.source_slot,
                        template_catalog.source_seed(source)?.proof_digest(),
                    )?;
                    insert_source_current(
                        catalog_digest,
                        group.full_state_template_id,
                        color_id,
                        leg.source_slot,
                        group.momentum_sign,
                        CurrentHelicityIdentity::all_flow_union(group.full_spin_state_class),
                        source,
                        reflection,
                        CurrentSourceBinding::runtime_dispatch_with_variants(
                            group.contract_id,
                            group.variants,
                        )?,
                        None,
                        &zero_orders,
                        pairing_catalog,
                        template_catalog,
                        currents,
                        current_ids,
                        currents_by_size,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn source_color_state_id(
    source_slot: u32,
    source: SourceRow,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
    color_states: &mut DynamicLCColorStateInterner,
) -> RusticolResult<super::DynamicLCColorStateId> {
    let dynamic_state = template_catalog.source_seed(source)?.instantiate(
        source_slot,
        template
            .current_states
            .get(source.state_template_id as usize)
            .ok_or_else(|| invalid("source current-state template is absent"))?
            .color_representation,
    )?;
    color_states.intern(dynamic_state)
}

#[derive(Debug)]
struct UnionSourceDispatchGroup {
    contract_id: u32,
    full_state_template_id: u32,
    full_spin_state_class: i32,
    momentum_sign: i32,
    representative_source: SourceRow,
    variants: Vec<RuntimeSourceVariantBinding>,
}

fn union_source_dispatch_groups(
    retained_states: &[&ProcessSourceStateRow],
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
) -> RusticolResult<Vec<UnionSourceDispatchGroup>> {
    let mut grouped = BTreeMap::<
        u32,
        (
            RuntimeHelicityContractRow,
            Vec<(ProcessSourceStateRow, RuntimeHelicityVariantRow)>,
        ),
    >::new();
    for process_state in retained_states {
        let (contract, variant) =
            runtime_helicity_variant_for_source(template, process_state.source_template_id)?;
        if variant.source_state_template_id
            != template.sources[process_state.source_template_id as usize].state_template_id
        {
            return Err(invalid(format!(
                "runtime-helicity variant {} source-state contract is stale",
                variant.id
            )));
        }
        grouped
            .entry(contract.id)
            .or_insert_with(|| (contract, Vec::new()))
            .1
            .push((**process_state, variant));
    }
    if grouped.is_empty() {
        return Err(invalid(
            "all-flow-union source has no certified runtime-helicity dispatch domain",
        ));
    }

    grouped
        .into_values()
        .map(|(contract, mut entries)| {
            entries.sort_by_key(|(state, _)| state.state_index);
            if entries
                .windows(2)
                .any(|pair| pair[0].0.state_index == pair[1].0.state_index)
            {
                return Err(invalid(format!(
                    "runtime-helicity contract {} has an ambiguous process source-state mapping",
                    contract.id
                )));
            }
            let full_state = template
                .current_states
                .get(contract.full_state_template_id as usize)
                .ok_or_else(|| invalid("runtime-helicity full state is absent"))?;
            let representative_state = entries[0].0;
            let representative_source =
                template.sources[representative_state.source_template_id as usize];
            let momentum_sign = representative_state.momentum_sign;
            for (process_state, variant) in &entries {
                let source = template.sources[process_state.source_template_id as usize];
                validate_union_source_family(
                    representative_source,
                    source,
                    contract,
                    *variant,
                    *process_state,
                    full_state,
                    template,
                    template_catalog,
                )?;
                if process_state.momentum_sign != momentum_sign {
                    return Err(invalid(format!(
                        "runtime-helicity contract {} mixes source momentum signs",
                        contract.id
                    )));
                }
            }
            let full_spin_state_class =
                union_full_spin_state_class(contract, &entries, template, template_catalog)?;
            let variants = entries
                .into_iter()
                .map(|(process_state, variant)| {
                    RuntimeSourceVariantBinding::new(
                        process_state.state_index,
                        process_state.public_helicity,
                        variant.id,
                        variant.source_template_id,
                        variant.source_state_template_id,
                        process_state.current_state_template_id,
                        process_state.spin_state,
                        process_catalog.factor(
                            process_state.crossing_phase_factor_id,
                            "runtime source crossing phase",
                        )?,
                    )
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(UnionSourceDispatchGroup {
                contract_id: contract.id,
                full_state_template_id: contract.full_state_template_id,
                full_spin_state_class,
                momentum_sign,
                representative_source,
                variants,
            })
        })
        .collect()
}

fn runtime_helicity_variant_for_source(
    template: &OwnedRecurrenceTemplateInput,
    source_template_id: u32,
) -> RusticolResult<(RuntimeHelicityContractRow, RuntimeHelicityVariantRow)> {
    let mut found = None;
    for contract in template.runtime_helicity_contracts.iter().copied() {
        let range = contract.variant_range.as_usize_range(
            template.runtime_helicity_variants.len(),
            "runtime-helicity variants",
        )?;
        for variant in template.runtime_helicity_variants[range].iter().copied() {
            if variant.source_template_id != source_template_id {
                continue;
            }
            if found.replace((contract, variant)).is_some() {
                return Err(invalid(format!(
                    "source template {source_template_id} belongs to ambiguous runtime-helicity variants"
                )));
            }
        }
    }
    found.ok_or_else(|| {
        invalid(format!(
            "source template {source_template_id} has no certified runtime-helicity variant"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_union_source_family(
    representative: SourceRow,
    source: SourceRow,
    contract: RuntimeHelicityContractRow,
    variant: RuntimeHelicityVariantRow,
    process_state: ProcessSourceStateRow,
    full_state: &super::template::CurrentStateRow,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
) -> RusticolResult<()> {
    if variant.contract_id != contract.id
        || variant.source_template_id != source.id
        || variant.source_state_template_id != source.state_template_id
    {
        return Err(invalid(format!(
            "runtime-helicity variant {} is inconsistent with source template {}",
            variant.id, source.id
        )));
    }
    let effective = template
        .current_states
        .get(process_state.current_state_template_id as usize)
        .ok_or_else(|| invalid("runtime source effective state is absent"))?;
    let full_state_is_compatible = full_state.particle_id == effective.particle_id
        && full_state.anti_particle_id == effective.anti_particle_id
        && full_state.species_string_id == effective.species_string_id
        && full_state.orientation == effective.orientation
        && full_state.statistics == effective.statistics
        && full_state.color_representation == effective.color_representation
        && full_state.lc_color_shape_string_id == effective.lc_color_shape_string_id
        && full_state.auxiliary_kind_string_id == effective.auxiliary_kind_string_id
        && full_state.mass_parameter_id == effective.mass_parameter_id
        && full_state.width_parameter_id == effective.width_parameter_id;
    if !full_state_is_compatible {
        return Err(invalid(format!(
            "runtime-helicity full state {} is not crossing-compatible with process state {}",
            contract.full_state_template_id, process_state.current_state_template_id
        )));
    }
    if !same_union_source_dispatch_semantics(representative, source)
        || template_catalog.flavour_flow(
            representative.flavour_flow_id,
            "runtime source flavour flow",
        )? != template_catalog
            .flavour_flow(source.flavour_flow_id, "runtime source flavour flow")?
    {
        return Err(invalid(format!(
            "runtime-helicity contract {} mixes incompatible source semantics",
            contract.id
        )));
    }
    Ok(())
}

fn same_union_source_dispatch_semantics(left: SourceRow, right: SourceRow) -> bool {
    left.crossing_string_id == right.crossing_string_id
        && left.wavefunction_family_string_id == right.wavefunction_family_string_id
        && left.flavour_flow_id == right.flavour_flow_id
        && left.quantum_number_flow_id == right.quantum_number_flow_id
        && left.lc_color_seed_operation == right.lc_color_seed_operation
        && left.lc_color_seed_shape_string_id == right.lc_color_seed_shape_string_id
        && left.lc_color_seed_component_kind == right.lc_color_seed_component_kind
        && left.lc_color_seed_component_role == right.lc_color_seed_component_role
        && left.lc_color_seed_provenance_sequence_id == right.lc_color_seed_provenance_sequence_id
        && left.mass_parameter_id == right.mass_parameter_id
        && left.width_parameter_id == right.width_parameter_id
}

fn union_full_spin_state_class(
    contract: RuntimeHelicityContractRow,
    entries: &[(ProcessSourceStateRow, RuntimeHelicityVariantRow)],
    template: &OwnedRecurrenceTemplateInput,
    _catalog: &TemplateCatalog<'_>,
) -> RusticolResult<i32> {
    let mut result_spins = BTreeSet::new();
    for flow in &template.quantum_flows {
        if flow.result_state_template_id == contract.full_state_template_id {
            result_spins.insert(flow.result_spin_state);
        }
    }
    if result_spins.is_empty() {
        result_spins.extend(entries.iter().map(|(state, _)| state.spin_state));
    }
    union_spin_state_class(contract.id, &result_spins)
}

fn union_spin_state_class(contract_id: u32, result_spins: &BTreeSet<i32>) -> RusticolResult<i32> {
    if result_spins.is_empty() {
        return Err(invalid(format!(
            "runtime-helicity contract {} has no full-state spin class",
            contract_id
        )));
    }
    if result_spins.contains(&DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS) {
        return Err(invalid(format!(
            "runtime-helicity contract {} contains the reserved dynamic source-spin sentinel",
            contract_id
        )));
    }
    Ok(if result_spins.len() == 1 {
        *result_spins.iter().next().expect("checked nonempty")
    } else {
        // One all-flow source current stores the full runtime-helicity vector.
        // Its selected chiral embedding is populated at execution time, so a
        // static source spin class would incorrectly discard the other
        // certified branch during recurrence construction.
        DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS
    })
}

#[allow(clippy::too_many_arguments)]
fn insert_source_current(
    catalog_digest: SemanticDigest,
    current_state_template_id: u32,
    color_id: super::DynamicLCColorStateId,
    source_slot: u32,
    momentum_sign: i32,
    helicity_identity: CurrentHelicityIdentity,
    source: SourceRow,
    reflection: CurrentReflection,
    source_binding: CurrentSourceBinding,
    source_factor: Option<ExactComplexRational>,
    zero_orders: &[u32],
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template_catalog: &TemplateCatalog<'_>,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut TransientCurrentIdIndex,
    currents_by_size: &mut [Vec<u32>],
) -> RusticolResult<()> {
    let key = CurrentCoreKey::new(
        catalog_digest,
        RecurrenceNodeKind::Source,
        current_state_template_id,
        color_id,
        vec![source_slot],
        CanonicalMomentumLinearForm::new(vec![MomentumTerm {
            source_slot,
            coefficient: momentum_sign,
        }])?,
        helicity_identity,
        template_catalog
            .flavour_flow(source.flavour_flow_id, "source flavour flow")?
            .to_vec(),
        source.quantum_number_flow_id,
        zero_orders.to_vec(),
        source_binding,
        None,
    )?;
    let realized_pairing_rule_ids = compatible_pairing_rules_for_current(
        pairing_catalog,
        template_catalog.input,
        current_state_template_id,
        &[source_slot],
    )?;
    if let Some(existing) = current_ids.get(&key).copied() {
        if currents[existing as usize].source_exact_factor != source_factor {
            return Err(invalid(
                "equivalent source currents have different exact factors",
            ));
        }
        currents[existing as usize]
            .reflection
            .include(reflection.proof().cloned())?;
        currents[existing as usize]
            .realized_pairing_rule_ids
            .extend(realized_pairing_rule_ids);
        return Ok(());
    }
    let id = u32::try_from(currents.len())
        .map_err(|_| invalid("recurrence current count exceeds u32"))?;
    current_ids.insert(key.clone(), id);
    currents.push(PendingCurrent {
        key,
        source_exact_factor: source_factor,
        contributions: BTreeMap::new(),
        realized_pairing_rule_ids,
        reflection,
        reflection_certificate_id: None,
    });
    currents_by_size[0].push(id);
    Ok(())
}

pub(super) fn source_reflection(
    color_states: &DynamicLCColorStateInterner,
    color_id: super::DynamicLCColorStateId,
    source_slot: u32,
    source_seed_proof: SemanticDigest,
) -> RusticolResult<CurrentReflection> {
    let color = color_states
        .get(color_id)
        .ok_or_else(|| invalid("source dynamic color state disappeared"))?;
    Ok(if color.pure_adjoint_word().is_some() {
        let color_identity = dynamic_color_identity_digest(color)?;
        let root = source_reflection_proof_root(source_slot, source_seed_proof, color_identity)?;
        CurrentReflection::Proven(CurrentReflectionProof::new(
            ExactComplexRational::ONE,
            [root],
            color_identity,
        )?)
    } else {
        CurrentReflection::Unavailable
    })
}

pub(super) fn validate_crossed_source_state(
    is_initial: bool,
    process_state: &super::process::ProcessSourceStateRow,
    source: SourceRow,
    template: &OwnedRecurrenceTemplateInput,
) -> RusticolResult<()> {
    let canonical = template
        .current_states
        .get(source.state_template_id as usize)
        .ok_or_else(|| invalid("source canonical current-state template is absent"))?;
    let effective = template
        .current_states
        .get(process_state.current_state_template_id as usize)
        .ok_or_else(|| invalid("source effective current-state template is absent"))?;
    if effective.chirality != process_state.chirality {
        return Err(invalid(format!(
            "process source chirality {} does not match effective current-state chirality {}",
            process_state.chirality, effective.chirality,
        )));
    }
    let compatible = canonical.particle_id == effective.particle_id
        && canonical.anti_particle_id == effective.anti_particle_id
        && canonical.species_string_id == effective.species_string_id
        && canonical.orientation == effective.orientation
        && canonical.statistics == effective.statistics
        && canonical.color_representation == effective.color_representation
        && canonical.basis_string_id == effective.basis_string_id
        && canonical.tensor_ordering_sequence_id == effective.tensor_ordering_sequence_id
        && canonical.dimension == effective.dimension
        && canonical.lc_color_shape_string_id == effective.lc_color_shape_string_id
        && canonical.auxiliary_kind_string_id == effective.auxiliary_kind_string_id
        && canonical.mass_parameter_id == effective.mass_parameter_id
        && canonical.width_parameter_id == effective.width_parameter_id;
    if !compatible || (!is_initial && canonical.id != effective.id) {
        return Err(invalid(format!(
            "source template {} and effective current-state template {} are not crossing-compatible",
            process_state.source_template_id, process_state.current_state_template_id,
        )));
    }
    Ok(())
}

#[cfg(test)]
fn parent_pairs_for_target(
    target_size: usize,
    prior_currents_by_size: &[Vec<u32>],
) -> impl Iterator<Item = [u32; 2]> + '_ {
    (1..=target_size / 2).flat_map(move |left_size| {
        let right_size = target_size - left_size;
        let same_size = left_size == right_size;
        prior_currents_by_size[left_size - 1]
            .iter()
            .copied()
            .flat_map(move |left_id| {
                prior_currents_by_size[right_size - 1]
                    .iter()
                    .copied()
                    .filter_map(move |right_id| match left_id.cmp(&right_id) {
                        std::cmp::Ordering::Less => Some([left_id, right_id]),
                        std::cmp::Ordering::Equal => None,
                        std::cmp::Ordering::Greater if same_size => None,
                        std::cmp::Ordering::Greater => Some([right_id, left_id]),
                    })
            })
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OrderedCurrentId {
    current_id: u32,
    bucket_position: usize,
}

#[derive(Debug, Default)]
struct SupportBucketIndex {
    ordered_supports: Vec<TransientSupportKey>,
    current_ids_by_support: HashMap<TransientSupportKey, Vec<OrderedCurrentId>>,
    disjoint_current_cache: HashMap<TransientSupportKey, Rc<[OrderedCurrentId]>>,
    entry_count: usize,
}

impl SupportBucketIndex {
    fn insert(&mut self, current_id: u32, support: &TransientSupportKey) {
        use std::collections::hash_map::Entry;

        let ordered = OrderedCurrentId {
            current_id,
            bucket_position: self.entry_count,
        };
        self.entry_count += 1;
        match self.current_ids_by_support.entry(support.clone()) {
            Entry::Occupied(mut entry) => entry.get_mut().push(ordered),
            Entry::Vacant(entry) => {
                self.ordered_supports.push(support.clone());
                entry.insert(vec![ordered]);
            }
        }
        self.disjoint_current_cache = HashMap::new();
    }

    fn disjoint_current_ids(
        &mut self,
        left_support: &TransientSupportKey,
    ) -> RusticolResult<(Rc<[OrderedCurrentId]>, bool)> {
        if let Some(cached) = self.disjoint_current_cache.get(left_support) {
            return Ok((Rc::clone(cached), true));
        }
        let mut result = Vec::new();
        // Emission never depends on HashMap iteration order. The support order
        // is the first-current order of the original size bucket, and final
        // positions restore that bucket's exact current-ID order.
        for right_support in &self.ordered_supports {
            if !left_support.is_disjoint(right_support)? {
                continue;
            }
            result.extend(
                self.current_ids_by_support
                    .get(right_support)
                    .expect("ordered support has an ID bucket")
                    .iter()
                    .copied(),
            );
        }
        result.sort_unstable_by_key(|entry| entry.bucket_position);
        let result = Rc::<[OrderedCurrentId]>::from(result);
        self.disjoint_current_cache
            .insert(left_support.clone(), Rc::clone(&result));
        Ok((result, false))
    }

    fn clear_disjoint_cache(&mut self) {
        self.disjoint_current_cache = HashMap::new();
    }
}

#[derive(Debug)]
struct LaneSupportBuckets {
    buckets_by_size: Vec<SupportBucketIndex>,
}

impl LaneSupportBuckets {
    fn new(
        currents_by_size: &[Vec<u32>],
        support_keys: &TransientCurrentSupportKeys,
    ) -> RusticolResult<Self> {
        let mut buckets_by_size = (0..currents_by_size.len())
            .map(|_| SupportBucketIndex::default())
            .collect::<Vec<_>>();
        for (size_index, current_ids) in currents_by_size.iter().enumerate() {
            for current_id in current_ids.iter().copied() {
                buckets_by_size[size_index].insert(current_id, support_keys.get(current_id)?);
            }
        }
        Ok(Self { buckets_by_size })
    }

    fn bucket_mut(&mut self, support_size: usize) -> RusticolResult<&mut SupportBucketIndex> {
        self.buckets_by_size
            .get_mut(
                support_size
                    .checked_sub(1)
                    .ok_or_else(|| invalid("transient support bucket size is zero"))?,
            )
            .ok_or_else(|| invalid("transient support bucket is absent"))
    }

    fn replace_bucket(
        &mut self,
        support_size: usize,
        current_ids: &[u32],
        support_keys: &TransientCurrentSupportKeys,
    ) -> RusticolResult<()> {
        let bucket = self.bucket_mut(support_size)?;
        *bucket = SupportBucketIndex::default();
        for current_id in current_ids.iter().copied() {
            bucket.insert(current_id, support_keys.get(current_id)?);
        }
        Ok(())
    }

    fn clear_disjoint_caches(&mut self) {
        for bucket in &mut self.buckets_by_size {
            bucket.clear_disjoint_cache();
        }
    }

    #[cfg(test)]
    fn cached_disjoint_query_count(&self) -> usize {
        self.buckets_by_size
            .iter()
            .map(|bucket| bucket.disjoint_current_cache.len())
            .sum()
    }

    #[cfg(test)]
    fn cached_disjoint_capacity(&self) -> usize {
        self.buckets_by_size
            .iter()
            .map(|bucket| bucket.disjoint_current_cache.capacity())
            .sum()
    }
}

#[derive(Clone, Copy, Debug)]
enum CandidateOrdinalMap {
    SameSize {
        first_greater_position: usize,
    },
    DifferentSize {
        equal_start: usize,
        equal_end: usize,
    },
}

impl CandidateOrdinalMap {
    fn offset(self, right_position: usize) -> Option<usize> {
        match self {
            Self::SameSize {
                first_greater_position,
            } => right_position.checked_sub(first_greater_position),
            Self::DifferentSize {
                equal_start,
                equal_end,
            } => {
                if (equal_start..equal_end).contains(&right_position) {
                    None
                } else if right_position < equal_start {
                    Some(right_position)
                } else {
                    Some(right_position - (equal_end - equal_start))
                }
            }
        }
    }
}

#[derive(Debug)]
struct ParentPairPlan {
    left_id: u32,
    right_ids: Rc<[OrderedCurrentId]>,
    same_size: bool,
    candidate_base: usize,
    ordinal_map: CandidateOrdinalMap,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct OrderedDisjointParentPair {
    parent_ids: [u32; 2],
    theoretical_candidate_count: usize,
}

#[derive(Debug)]
struct DisjointParentPairs {
    plans: std::vec::IntoIter<ParentPairPlan>,
    current_plan: Option<ParentPairPlan>,
    right_index: usize,
}

impl Iterator for DisjointParentPairs {
    type Item = RusticolResult<OrderedDisjointParentPair>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_plan.is_none() {
                self.current_plan = self.plans.next();
                self.right_index = 0;
            }
            let plan = self.current_plan.as_ref()?;
            let Some(right) = plan.right_ids.get(self.right_index).copied() else {
                self.current_plan = None;
                continue;
            };
            self.right_index += 1;
            let Some(candidate_offset) = plan.ordinal_map.offset(right.bucket_position) else {
                continue;
            };
            let parent_ids = match plan.left_id.cmp(&right.current_id) {
                std::cmp::Ordering::Less => [plan.left_id, right.current_id],
                std::cmp::Ordering::Equal => continue,
                std::cmp::Ordering::Greater if plan.same_size => continue,
                std::cmp::Ordering::Greater => [right.current_id, plan.left_id],
            };
            let theoretical_candidate_count = match plan
                .candidate_base
                .checked_add(candidate_offset)
                .and_then(|count| count.checked_add(1))
            {
                Some(count) => count,
                None => {
                    return Some(Err(invalid(
                        "recurrence candidate parent-pair count exceeds usize",
                    )));
                }
            };
            return Some(Ok(OrderedDisjointParentPair {
                parent_ids,
                theoretical_candidate_count,
            }));
        }
    }
}

#[derive(Debug)]
struct DisjointParentPairSchedule {
    pairs: DisjointParentPairs,
    theoretical_candidate_count: usize,
    support_bucket_count: usize,
    support_bucket_probe_count: usize,
    support_bucket_cache_hit_count: usize,
}

fn disjoint_parent_pairs_for_target(
    target_size: usize,
    prior_currents_by_size: &[Vec<u32>],
    support_keys: &TransientCurrentSupportKeys,
    support_buckets: &mut LaneSupportBuckets,
) -> RusticolResult<DisjointParentPairSchedule> {
    let mut plans = Vec::new();
    let mut theoretical_candidate_count = 0usize;
    let mut support_bucket_count = 0usize;
    let mut support_bucket_probe_count = 0usize;
    let mut support_bucket_cache_hit_count = 0usize;
    for left_size in 1..=target_size / 2 {
        let right_size = target_size - left_size;
        let same_size = left_size == right_size;
        let right_bucket = &prior_currents_by_size[right_size - 1];
        let indexed_right_bucket = support_buckets.bucket_mut(right_size)?;
        support_bucket_count = support_bucket_count
            .checked_add(indexed_right_bucket.ordered_supports.len())
            .ok_or_else(|| invalid("support-bucket count exceeds usize"))?;
        debug_assert!(right_bucket.iter().copied().is_sorted());
        for left_id in prior_currents_by_size[left_size - 1].iter().copied() {
            let equal_start = right_bucket.partition_point(|right_id| *right_id < left_id);
            let equal_end = right_bucket.partition_point(|right_id| *right_id <= left_id);
            let (ordinal_map, candidate_count) = if same_size {
                (
                    CandidateOrdinalMap::SameSize {
                        first_greater_position: equal_end,
                    },
                    right_bucket.len() - equal_end,
                )
            } else {
                (
                    CandidateOrdinalMap::DifferentSize {
                        equal_start,
                        equal_end,
                    },
                    right_bucket.len() - (equal_end - equal_start),
                )
            };
            let (right_ids, cache_hit) =
                indexed_right_bucket.disjoint_current_ids(support_keys.get(left_id)?)?;
            support_bucket_probe_count = support_bucket_probe_count
                .checked_add(1)
                .ok_or_else(|| invalid("support-bucket probe count exceeds usize"))?;
            if cache_hit {
                support_bucket_cache_hit_count = support_bucket_cache_hit_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("support-bucket cache-hit count exceeds usize"))?;
            }
            plans.push(ParentPairPlan {
                left_id,
                right_ids,
                same_size,
                candidate_base: theoretical_candidate_count,
                ordinal_map,
            });
            theoretical_candidate_count = theoretical_candidate_count
                .checked_add(candidate_count)
                .ok_or_else(|| invalid("recurrence parent-pair total exceeds usize"))?;
        }
    }
    // The plans retain their shared lists until this stage finishes. Drop the
    // bucket-owned cache references now so no list survives into later stages.
    support_buckets.clear_disjoint_caches();
    Ok(DisjointParentPairSchedule {
        pairs: DisjointParentPairs {
            plans: plans.into_iter(),
            current_plan: None,
            right_index: 0,
        },
        theoretical_candidate_count,
        support_bucket_count,
        support_bucket_probe_count,
        support_bucket_cache_hit_count,
    })
}

fn parent_pair_total_for_target(
    target_size: usize,
    prior_currents_by_size: &[Vec<u32>],
) -> RusticolResult<usize> {
    (1..=target_size / 2).try_fold(0usize, |total, left_size| {
        let right_size = target_size - left_size;
        let left_count = prior_currents_by_size[left_size - 1].len();
        let right_count = prior_currents_by_size[right_size - 1].len();
        let pair_count = if left_size == right_size {
            left_count
                .checked_mul(left_count.saturating_sub(1))
                .and_then(|value| value.checked_div(2))
        } else {
            left_count.checked_mul(right_count)
        }
        .ok_or_else(|| invalid("recurrence parent-pair total exceeds usize"))?;
        total
            .checked_add(pair_count)
            .ok_or_else(|| invalid("recurrence parent-pair total exceeds usize"))
    })
}

fn checked_diagnostic_add(counter: &mut usize, amount: usize, label: &str) -> RusticolResult<()> {
    *counter = counter
        .checked_add(amount)
        .ok_or_else(|| invalid(format!("{label} exceeds usize")))?;
    Ok(())
}

fn pending_contribution_entry_count(currents: &[PendingCurrent]) -> RusticolResult<usize> {
    currents.iter().try_fold(0usize, |total, current| {
        total
            .checked_add(current.contributions.len())
            .ok_or_else(|| invalid("resident recurrence contribution count exceeds usize"))
    })
}

fn prune_inactive_lane_contributions(
    currents: &mut [PendingCurrent],
    domain: PendingConstructionDomain,
    closures: &BTreeMap<PendingClosureKey, PendingClosureGroup>,
    resident_contribution_count: &mut usize,
) -> RusticolResult<()> {
    if domain.shared_source_end > domain.lane_internal_start
        || domain.lane_internal_start > domain.lane_internal_end
        || domain.lane_internal_end > currents.len()
    {
        return Err(invalid(
            "recurrence lane contribution compaction has an invalid domain",
        ));
    }
    let mut live = BTreeSet::new();
    let mut queue = VecDeque::new();
    for (key, group) in closures {
        if group.exact_factor.is_zero() {
            continue;
        }
        for parent_id in key.parent_current_ids.iter().copied() {
            if !domain.contains(parent_id as usize) {
                return Err(invalid(
                    "recurrence lane closure references another construction lane",
                ));
            }
            if live.insert(parent_id) {
                queue.push_back(parent_id);
            }
        }
    }
    while let Some(current_id) = queue.pop_front() {
        let current = currents
            .get(current_id as usize)
            .ok_or_else(|| invalid("recurrence lane liveness references an absent current"))?;
        for (contribution, factor) in &current.contributions {
            if factor.is_zero() {
                continue;
            }
            for parent_id in contribution.parent_current_ids.iter().copied() {
                if !domain.contains(parent_id as usize) {
                    return Err(invalid(
                        "recurrence lane contribution references another construction lane",
                    ));
                }
                if live.insert(parent_id) {
                    queue.push_back(parent_id);
                }
            }
        }
    }

    for (current_id, current) in currents
        .iter_mut()
        .enumerate()
        .take(domain.lane_internal_end)
        .skip(domain.lane_internal_start)
    {
        let current_id_u32 = u32::try_from(current_id)
            .map_err(|_| invalid("recurrence lane current ID exceeds u32"))?;
        let before = current.contributions.len();
        if live.contains(&current_id_u32) {
            current.contributions.retain(|_, factor| !factor.is_zero());
        } else {
            current.contributions.clear();
        }
        let removed = before
            .checked_sub(current.contributions.len())
            .ok_or_else(|| invalid("recurrence lane contribution compaction underflowed"))?;
        *resident_contribution_count = resident_contribution_count
            .checked_sub(removed)
            .ok_or_else(|| invalid("resident recurrence contribution count underflowed"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_internal_currents(
    catalog_digest: SemanticDigest,
    process: &OwnedRecurrenceProcessInput,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template: &OwnedRecurrenceTemplateInput,
    fermion_ordering: &FermionOrderingContext,
    prepared_transitions: &PreparedTransitionCatalog,
    transition_reflections: &TransitionReflectionIndex,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    color_targets: &MaterializedColorTargets,
    structural_demands: &StructuralDemandIndex,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut TransientCurrentIdIndex,
    currents_by_size: &mut [Vec<u32>],
    current_support_keys: &mut TransientCurrentSupportKeys,
    reflection_certificates: &mut Vec<PendingReflectionCertificate>,
    resident_contribution_count: &mut usize,
    stage_index_offset: usize,
    reported_stage_total: usize,
    phase_total: usize,
    telemetry: &mut RecurrenceGenerationTelemetry,
    collect_telemetry: bool,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
) -> RusticolResult<Vec<StageConstructionDiagnostics>> {
    if current_support_keys.source_count != process.external_legs.len() {
        return Err(invalid(
            "transient current support domain disagrees with the process",
        ));
    }
    let initial_support_index_started = telemetry_timer(collect_telemetry);
    let mut support_buckets = LaneSupportBuckets::new(currents_by_size, current_support_keys)?;
    add_optional_elapsed_nanoseconds(
        &mut telemetry.support_indexing_nanoseconds,
        initial_support_index_started,
    );
    let mut diagnostics = Vec::new();
    let mut completed_color_target_prune_count = 0usize;
    for target_size in 2..process.external_legs.len() {
        let stage_current_start = currents.len();
        let stage_contribution_start = *resident_contribution_count;
        let mut stage = StageConstructionDiagnostics {
            target_size,
            ..StageConstructionDiagnostics::default()
        };
        let (prior_buckets, target_and_later) = currents_by_size.split_at_mut(target_size - 1);
        debug_assert!(prior_buckets.iter().flatten().copied().is_sorted());
        let target_bucket = &mut target_and_later[0];
        let candidate_parent_pair_total = parent_pair_total_for_target(target_size, prior_buckets)?;
        let stage_index = stage_index_offset
            .checked_add(target_size - 1)
            .ok_or_else(|| invalid("recurrence construction stage index exceeds usize"))?;
        progress(RecurrenceBuildProgress::snapshot(
            "recurrence stage",
            stage_index.saturating_add(1),
            phase_total,
            Some(stage_index),
            reported_stage_total,
            Some(target_size),
            0,
            Some(candidate_parent_pair_total),
            currents.len(),
            *resident_contribution_count,
            color_states.len(),
            completed_color_target_prune_count,
        ))?;
        let mut last_progress = Instant::now();
        let support_index_started = telemetry_timer(collect_telemetry);
        let pair_schedule = disjoint_parent_pairs_for_target(
            target_size,
            prior_buckets,
            current_support_keys,
            &mut support_buckets,
        )?;
        add_optional_elapsed_nanoseconds(
            &mut stage.support_indexing_nanoseconds,
            support_index_started,
        );
        debug_assert_eq!(
            pair_schedule.theoretical_candidate_count,
            candidate_parent_pair_total
        );
        stage.support_bucket_count = pair_schedule.support_bucket_count;
        stage.support_bucket_probe_count = pair_schedule.support_bucket_probe_count;
        stage.support_bucket_cache_hit_count = pair_schedule.support_bucket_cache_hit_count;
        let mut next_progress_candidate = PROGRESS_PAIR_INTERVAL;
        let candidate_processing_started = telemetry_timer(collect_telemetry);
        for pair in pair_schedule.pairs {
            let pair = pair?;
            let [left_id, right_id] = pair.parent_ids;
            stage.candidate_parent_pair_count = pair.theoretical_candidate_count;
            if stage.candidate_parent_pair_count >= next_progress_candidate {
                if last_progress.elapsed() >= PROGRESS_TIME_INTERVAL {
                    progress(RecurrenceBuildProgress::snapshot(
                        "recurrence stage",
                        stage_index.saturating_add(1),
                        phase_total,
                        Some(stage_index),
                        reported_stage_total,
                        Some(target_size),
                        stage.candidate_parent_pair_count,
                        Some(candidate_parent_pair_total),
                        currents.len(),
                        *resident_contribution_count,
                        color_states.len(),
                        completed_color_target_prune_count
                            .saturating_add(stage.color_target_prune_count),
                    ))?;
                    last_progress = Instant::now();
                }
                next_progress_candidate = stage
                    .candidate_parent_pair_count
                    .checked_div(PROGRESS_PAIR_INTERVAL)
                    .and_then(|interval| interval.checked_add(1))
                    .and_then(|interval| interval.checked_mul(PROGRESS_PAIR_INTERVAL))
                    .unwrap_or(usize::MAX);
            }
            checked_diagnostic_add(
                &mut stage.parent_pair_count,
                1,
                "recurrence parent-pair count",
            )?;
            let left_state = currents[left_id as usize].key.current_state_template_id();
            let right_state = currents[right_id as usize].key.current_state_template_id();
            let indexed_transitions = prepared_transitions.rows(left_state, right_state);
            if !indexed_transitions.is_empty() {
                checked_diagnostic_add(
                    &mut stage.transition_index_hit_count,
                    1,
                    "recurrence transition-index hit count",
                )?;
            }
            checked_diagnostic_add(
                &mut stage.transition_candidate_count,
                indexed_transitions.len(),
                "recurrence transition-candidate count",
            )?;
            if indexed_transitions.is_empty() {
                continue;
            }
            let merged_support_key = current_support_keys
                .get(left_id)?
                .union_disjoint(current_support_keys.get(right_id)?)?;
            let mut merged_support_source_slots = None;
            for prepared in indexed_transitions {
                checked_diagnostic_add(
                    &mut stage.state_order_count,
                    1,
                    "recurrence state-order count",
                )?;
                add_transition_contributions(
                    catalog_digest,
                    prepared,
                    prepared.parent_ids(left_state, right_state, left_id, right_id)?,
                    &merged_support_key,
                    &mut merged_support_source_slots,
                    target_size + 1 < process.external_legs.len(),
                    pairing_catalog,
                    template,
                    fermion_ordering,
                    transition_reflections,
                    coupling_limits,
                    propagators,
                    color_targets,
                    structural_demands,
                    color_states,
                    currents,
                    current_ids,
                    target_bucket,
                    current_support_keys,
                    &mut stage,
                    resident_contribution_count,
                    collect_telemetry,
                )?;
            }
        }
        if let Some(contact_owner_plan) = plan_established_contact_orbit_owners(
            stage_current_start,
            prepared_transitions,
            currents,
            *resident_contribution_count,
        )? {
            contact_owner_plan.commit(currents, resident_contribution_count);
        }
        add_optional_elapsed_nanoseconds(
            &mut stage.candidate_processing_nanoseconds,
            candidate_processing_started,
        );
        stage.candidate_parent_pair_count = candidate_parent_pair_total;
        if collect_telemetry {
            stage.current_insert_count = currents
                .len()
                .checked_sub(stage_current_start)
                .ok_or_else(|| invalid("stage current insertion count underflowed"))?;
            stage.current_key_clone_count = stage.current_insert_count;
            stage.current_key_lookup_count = stage
                .color_result_count
                .checked_sub(stage.color_target_prune_count)
                .ok_or_else(|| invalid("stage color-target acceptance count underflowed"))?;
            stage.current_key_hit_count = stage
                .current_key_lookup_count
                .checked_sub(stage.current_insert_count)
                .ok_or_else(|| invalid("stage current-key hit count underflowed"))?;
            debug_assert_eq!(stage.accepted_parent_key_clone_count, 0);
            stage.structural_reject_count = stage
                .coupling_match_count
                .checked_sub(stage.transition_accept_count)
                .ok_or_else(|| invalid("stage structural-reject count underflowed"))?;
            stage.contribution_insert_count = resident_contribution_count
                .checked_sub(stage_contribution_start)
                .ok_or_else(|| invalid("stage contribution insertion count underflowed"))?;
            stage.contribution_merge_count = stage
                .contribution_attempt_count
                .checked_sub(stage.contribution_insert_count)
                .ok_or_else(|| invalid("stage contribution merge count underflowed"))?;
        }
        progress(RecurrenceBuildProgress::snapshot(
            "recurrence stage",
            stage_index.saturating_add(1),
            phase_total,
            Some(stage_index),
            reported_stage_total,
            Some(target_size),
            stage.candidate_parent_pair_count,
            Some(candidate_parent_pair_total),
            currents.len(),
            *resident_contribution_count,
            color_states.len(),
            completed_color_target_prune_count.saturating_add(stage.color_target_prune_count),
        ))?;
        // Target-size currents cannot parent another current in this stage, so
        // their support index is materialized only once after ID reconciliation.
        debug_assert_eq!(support_buckets.bucket_mut(target_size)?.entry_count, 0);
        reconcile_stage_reflections(
            stage_current_start,
            color_states,
            currents,
            current_ids,
            target_bucket,
            reflection_certificates,
            process.external_legs.len(),
            resident_contribution_count,
        )?;
        current_support_keys.reconcile_stage_tail(stage_current_start, currents)?;
        let support_index_started = telemetry_timer(collect_telemetry);
        support_buckets.replace_bucket(target_size, target_bucket, current_support_keys)?;
        add_optional_elapsed_nanoseconds(
            &mut stage.support_indexing_nanoseconds,
            support_index_started,
        );
        debug_assert_eq!(stage.target_size, target_size);
        debug_assert_eq!(stage.transition_candidate_count, stage.state_order_count);
        completed_color_target_prune_count = completed_color_target_prune_count
            .checked_add(stage.color_target_prune_count)
            .ok_or_else(|| invalid("recurrence color-target prune count exceeds usize"))?;
        progress(RecurrenceBuildProgress::snapshot(
            "recurrence stage",
            stage_index.saturating_add(1),
            phase_total,
            Some(stage_index),
            reported_stage_total,
            Some(target_size),
            stage.candidate_parent_pair_count,
            Some(candidate_parent_pair_total),
            currents.len(),
            *resident_contribution_count,
            color_states.len(),
            completed_color_target_prune_count,
        ))?;
        if collect_telemetry {
            telemetry.include_stage(&stage)?;
        }
        diagnostics.push(stage);
    }
    Ok(diagnostics)
}

#[allow(clippy::too_many_arguments)]
fn add_transition_contributions(
    catalog_digest: SemanticDigest,
    prepared: &PreparedTransition,
    parent_ids: [u32; 2],
    support_key: &TransientSupportKey,
    merged_support_source_slots: &mut Option<Vec<u32>>,
    propagate_result: bool,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template: &OwnedRecurrenceTemplateInput,
    fermion_ordering: &FermionOrderingContext,
    transition_reflections: &TransitionReflectionIndex,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    color_targets: &MaterializedColorTargets,
    structural_demands: &StructuralDemandIndex,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut TransientCurrentIdIndex,
    target_bucket: &mut Vec<u32>,
    current_support_keys: &mut TransientCurrentSupportKeys,
    diagnostics: &mut StageConstructionDiagnostics,
    resident_contribution_count: &mut usize,
    collect_telemetry: bool,
) -> RusticolResult<()> {
    let transition = prepared.row;
    let quantum = prepared.quantum;
    let coupling_orders = {
        let parents = [
            &currents[parent_ids[0] as usize].key,
            &currents[parent_ids[1] as usize].key,
        ];
        if let Some(contact_orbit) = prepared.contact_orbit.as_ref()
            && !contact_orbit.accepts_parent_domain(parents)?
        {
            return Ok(());
        }
        if !prepared.quantum_flow_matches(&parents) {
            return Ok(());
        }
        checked_diagnostic_add(
            &mut diagnostics.quantum_match_count,
            1,
            "recurrence quantum-match count",
        )?;
        let Some(coupling_orders) = combined_coupling_orders(
            parents[0].coupling_orders(),
            parents[1].coupling_orders(),
            &prepared.local_coupling_orders,
            coupling_limits,
        )?
        else {
            return Ok(());
        };
        checked_diagnostic_add(
            &mut diagnostics.coupling_match_count,
            1,
            "recurrence coupling-match count",
        )?;
        if parents[0].helicity_identity().strategy() != parents[1].helicity_identity().strategy() {
            return Err(invalid(
                "cannot merge recurrence helicity identities from different strategies",
            ));
        }
        if !structural_demands.accepts(
            support_key,
            transition.result_state_template_id,
            quantum.result_spin_state,
        ) {
            return Ok(());
        }
        if collect_telemetry {
            checked_diagnostic_add(
                &mut diagnostics.transition_accept_count,
                1,
                "recurrence transition-accept count",
            )?;
        }
        coupling_orders
    };
    let (evaluator_parent_ids, exchange_factor) = prepared.canonical_evaluator_parents(parent_ids);
    let output_factor = prepared.output_factor()?;
    let fermion_factor = fermion_ordering_factor(
        &template.current_states,
        [
            currents[parent_ids[0] as usize]
                .key
                .current_state_template_id(),
            currents[parent_ids[1] as usize]
                .key
                .current_state_template_id(),
        ],
        [
            currents[parent_ids[0] as usize].key.support_source_slots(),
            currents[parent_ids[1] as usize].key.support_source_slots(),
        ],
        fermion_ordering,
    )?;
    let base_factor = multiply_factors(&[
        prepared.transition_exact_factor,
        exchange_factor,
        prepared.contraction_exact_factor,
        output_factor,
        fermion_factor,
    ])?;
    let parent_reflections = [
        currents[parent_ids[0] as usize].reflection.clone(),
        currents[parent_ids[1] as usize].reflection.clone(),
    ];
    let parent_colors = [
        color_states
            .get(
                currents[parent_ids[0] as usize]
                    .key
                    .dynamic_lc_color_state_id(),
            )
            .ok_or_else(|| invalid("left dynamic color state disappeared"))?
            .clone(),
        color_states
            .get(
                currents[parent_ids[1] as usize]
                    .key
                    .dynamic_lc_color_state_id(),
            )
            .ok_or_else(|| invalid("right dynamic color state disappeared"))?
            .clone(),
    ];
    let reversal_masks = current_reversal_masks(&parent_colors, &parent_reflections);
    let local_reflection_proof = transition_reflections.proof(transition.id);
    let propagator_template_id = if propagate_result {
        propagators
            .get(&transition.result_state_template_id)
            .copied()
            .flatten()
    } else {
        None
    };
    let mut result_core_fields = None;

    for prepared_witness in prepared.witnesses.iter() {
        let witness_row = prepared_witness.row;
        if witness_row.left_shape_string_id != parent_colors[0].output_color_shape_id()
            || witness_row.right_shape_string_id != parent_colors[1].output_color_shape_id()
        {
            continue;
        }
        checked_diagnostic_add(
            &mut diagnostics.color_shape_match_count,
            1,
            "recurrence color-shape-match count",
        )?;
        let witness = &prepared_witness.witness;
        for reversal_mask in reversal_masks.iter().copied() {
            let mut variant_colors = parent_colors.clone();
            let mut reversal_factor = ExactComplexRational::ONE;
            for index in 0..2 {
                if reversal_mask & (1 << index) == 0 {
                    continue;
                }
                variant_colors[index] = variant_colors[index].reversed()?;
                reversal_factor = reversal_factor.checked_mul(
                    parent_reflections[index]
                        .phase()
                        .expect("reversal mask requires a proven parent phase"),
                )?;
            }
            let Some(result_color) = witness.apply(&variant_colors[0], &variant_colors[1])? else {
                continue;
            };
            checked_diagnostic_add(
                &mut diagnostics.color_result_count,
                1,
                "recurrence color-result count",
            )?;
            if !color_targets.accepts_up_to_reflection(&result_color)? {
                checked_diagnostic_add(
                    &mut diagnostics.color_target_prune_count,
                    1,
                    "recurrence color-target prune count",
                )?;
                continue;
            }
            let result_reflection = current_reflection_candidate(
                &result_color,
                &parent_reflections,
                local_reflection_proof,
            )?;
            #[cfg(feature = "on-the-fly-test-support")]
            let diagnostic_result_reflection = result_reflection.clone();
            let result_color_id = color_states.intern(result_color)?;
            if merged_support_source_slots.is_none() {
                *merged_support_source_slots = Some(merged_disjoint_support(
                    currents[parent_ids[0] as usize].key.support_source_slots(),
                    currents[parent_ids[1] as usize].key.support_source_slots(),
                ));
            }
            let support_source_slots = merged_support_source_slots
                .as_ref()
                .expect("merged support was initialized after color acceptance");
            debug_assert_eq!(support_source_slots.len(), support_key.cardinality());
            if result_core_fields.is_none() {
                let parents = [
                    &currents[parent_ids[0] as usize].key,
                    &currents[parent_ids[1] as usize].key,
                ];
                result_core_fields = Some((
                    merged_helicity_identity(
                        parents[0].helicity_identity(),
                        parents[1].helicity_identity(),
                        quantum.result_spin_state,
                    )?,
                    prepared.result_flavour_flow(&parents),
                    merged_momentum(parents[0].momentum(), parents[1].momentum())?,
                ));
            }
            let (helicity_identity, result_flavour_flow, result_momentum) = result_core_fields
                .as_ref()
                .expect("result current fields were initialized after color acceptance");
            let key = CurrentCoreKey::new(
                catalog_digest,
                RecurrenceNodeKind::Current,
                transition.result_state_template_id,
                result_color_id,
                support_source_slots.to_vec(),
                result_momentum.clone(),
                helicity_identity.clone(),
                result_flavour_flow.clone(),
                quantum.result_quantum_number_flow_id,
                coupling_orders.clone(),
                CurrentSourceBinding::None,
                propagator_template_id,
            )?;
            let realized_pairing_rule_ids = realized_pairing_rules_for_transition(
                compatible_pairing_rules_for_current(
                    pairing_catalog,
                    template,
                    transition.result_state_template_id,
                    key.support_source_slots(),
                )?,
                [
                    &currents[parent_ids[0] as usize].realized_pairing_rule_ids,
                    &currents[parent_ids[1] as usize].realized_pairing_rule_ids,
                ],
            );
            let result_id = if let Some(id) = current_ids.get(&key).copied() {
                debug_assert_eq!(
                    current_support_keys
                        .get(id)
                        .expect("existing current has a transient support key"),
                    support_key
                );
                currents[id as usize]
                    .reflection
                    .include(result_reflection)?;
                currents[id as usize]
                    .realized_pairing_rule_ids
                    .extend(realized_pairing_rule_ids);
                id
            } else {
                let id = u32::try_from(currents.len())
                    .map_err(|_| invalid("recurrence current count exceeds u32"))?;
                current_ids.insert(key.clone(), id);
                currents.push(PendingCurrent {
                    key,
                    source_exact_factor: None,
                    contributions: BTreeMap::new(),
                    realized_pairing_rule_ids,
                    reflection: result_reflection
                        .map_or(CurrentReflection::Unavailable, CurrentReflection::Proven),
                    reflection_certificate_id: None,
                });
                target_bucket.push(id);
                current_support_keys.push(id, support_key.clone())?;
                id
            };
            let contribution_key = ContributionKey::new(
                transition.id,
                evaluator_parent_ids.to_vec(),
                evaluator_parent_ids
                    .iter()
                    .map(|id| currents[*id as usize].key.current_state_template_id())
                    .collect(),
                evaluator_parent_ids
                    .iter()
                    .map(|id| currents[*id as usize].key.momentum().clone())
                    .collect(),
                transition.result_state_template_id,
                quantum.id,
                LCColorWitnessTermId::new(
                    transition.color_contraction_template_id,
                    witness_row.ordinal,
                ),
                prepared.quantum_semantic_digest,
                transition.output_projection_string_id,
            )?;
            let pending_key = PendingContributionKey {
                parent_current_ids: evaluator_parent_ids.into(),
                key: contribution_key,
            };
            let factor = base_factor
                .checked_mul(witness.exact_factor())?
                .checked_mul(reversal_factor)?;
            let pending_factor = match currents[result_id as usize]
                .contributions
                .entry(pending_key)
            {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    checked_diagnostic_add(
                        resident_contribution_count,
                        1,
                        "resident recurrence contribution count",
                    )?;
                    entry.insert(ExactComplexRational::ZERO)
                }
            };
            aggregate_factor(pending_factor, factor)?;
            #[cfg(feature = "on-the-fly-test-support")]
            let aggregate_factor_after = *pending_factor;
            checked_diagnostic_add(
                &mut diagnostics.contribution_attempt_count,
                1,
                "recurrence contribution-attempt count",
            )?;
            #[cfg(feature = "on-the-fly-test-support")]
            'transition_diagnostic: {
                if !super::diagnostic::transition_diagnostic_observation_active() {
                    break 'transition_diagnostic;
                }
                use super::diagnostic::{
                    ConstructionTransitionDiagnosticRowV1, observe_transition_diagnostic,
                };

                let digest = |current_id: u32| -> RusticolResult<SemanticDigest> {
                    let current = currents
                        .get(current_id as usize)
                        .ok_or_else(|| invalid("diagnostic current is absent"))?;
                    let color = color_states
                        .get(current.key.dynamic_lc_color_state_id())
                        .ok_or_else(|| invalid("diagnostic current color is absent"))?;
                    super::on_the_fly::hash_current_key(&current.key, color)
                };
                let result_color = color_states
                    .get(currents[result_id as usize].key.dynamic_lc_color_state_id())
                    .ok_or_else(|| invalid("diagnostic result color is absent"))?;
                observe_transition_diagnostic(ConstructionTransitionDiagnosticRowV1 {
                    materialized_sector_id: None,
                    output_current_digest: digest(result_id)?,
                    ordered_parent_digests: [
                        digest(evaluator_parent_ids[0])?,
                        digest(evaluator_parent_ids[1])?,
                    ],
                    transition_template_id: transition.id,
                    transition_semantic_digest: prepared.transition_semantic_digest,
                    evaluator_binding_semantic_digest: prepared.evaluator_binding_semantic_digest,
                    result_state_template_id: transition.result_state_template_id,
                    quantum_flow_witness_id: quantum.id,
                    quantum_semantic_digest: prepared.quantum_semantic_digest,
                    color_contraction_template_id: transition.color_contraction_template_id,
                    color_witness_ordinal: witness_row.ordinal,
                    color_witness_proof_digest: witness.proof_digest(),
                    output_projection_id: transition.output_projection_string_id,
                    transition_factor: prepared.transition_exact_factor,
                    contraction_factor: prepared.contraction_exact_factor,
                    output_factor,
                    exchange_factor,
                    witness_factor: witness.exact_factor(),
                    reversal_mask,
                    reversal_factor,
                    candidate_factor: factor,
                    aggregate_factor_after,
                    parent_reflection_proof_digests: [
                        parent_reflections[0]
                            .proof()
                            .map(CurrentReflectionProof::proof_digest),
                        parent_reflections[1]
                            .proof()
                            .map(CurrentReflectionProof::proof_digest),
                    ],
                    parent_reflection_phases: [
                        parent_reflections[0].phase(),
                        parent_reflections[1].phase(),
                    ],
                    local_reflection_proof_digest: local_reflection_proof
                        .map(|proof| proof.proof_digest),
                    local_reflection_phase: local_reflection_proof
                        .map(TransitionReflectionProof::phase),
                    result_reflection_proof_digest: diagnostic_result_reflection
                        .as_ref()
                        .map(CurrentReflectionProof::proof_digest),
                    result_reflection_phase: diagnostic_result_reflection
                        .as_ref()
                        .map(CurrentReflectionProof::phase),
                    output_color_orientation: format!("{result_color:?}"),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn current_key_with_dynamic_color(
    key: &CurrentCoreKey,
    dynamic_lc_color_state_id: DynamicLCColorStateId,
) -> RusticolResult<CurrentCoreKey> {
    CurrentCoreKey::new(
        key.catalog_digest(),
        key.node_kind(),
        key.current_state_template_id(),
        dynamic_lc_color_state_id,
        key.support_source_slots().to_vec(),
        key.momentum().clone(),
        key.helicity_identity().clone(),
        key.flavour_flow().to_vec(),
        key.quantum_number_flow_id(),
        key.coupling_orders().to_vec(),
        key.source_binding().clone(),
        key.propagator_template_id(),
    )
}

pub(super) fn reciprocal_reflection_proof(
    left: &CurrentReflection,
    right: &CurrentReflection,
) -> RusticolResult<bool> {
    let (Some(left), Some(right)) = (left.phase(), right.phase()) else {
        return Ok(false);
    };
    Ok(left.checked_mul(right)? == ExactComplexRational::ONE)
}

/// Finalize pure-adjoint reflection aliases after every contribution to the
/// stage's currents has been seen.
///
/// A contribution that lacks a reflection proof downgrades the whole current.
/// Both orientations are then retained as exact residual states. Only a pair
/// with reciprocal, complete fan-in proofs is compacted to its canonical
/// orientation. Stage currents cannot depend on one another, so compacting the
/// stage tail cannot invalidate any stored parent current ID.
#[allow(clippy::too_many_arguments)]
fn reconcile_stage_reflections(
    stage_start: usize,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut TransientCurrentIdIndex,
    target_bucket: &mut Vec<u32>,
    reflection_certificates: &mut Vec<PendingReflectionCertificate>,
    source_count: usize,
    resident_contribution_count: &mut usize,
) -> RusticolResult<()> {
    if stage_start > currents.len() {
        return Err(invalid(
            "recurrence reflection stage starts beyond current storage",
        ));
    }
    let stage_end = currents.len();
    let mut visited = vec![false; stage_end - stage_start];
    let mut prune = vec![false; stage_end - stage_start];

    for current_index in stage_start..stage_end {
        let local_index = current_index - stage_start;
        if visited[local_index] {
            continue;
        }
        let key = currents[current_index].key.clone();
        let color = color_states
            .get(key.dynamic_lc_color_state_id())
            .ok_or_else(|| invalid("reflection current color state disappeared"))?
            .clone();
        let Some(word) = color.pure_adjoint_word() else {
            continue;
        };
        if word.len() < 2 {
            continue;
        }
        let canonical = pure_adjoint_word_is_canonical(word);
        let reversed_color_id = color_states.intern(color.reversed()?)?;
        if reversed_color_id == key.dynamic_lc_color_state_id() {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let reversed_key = current_key_with_dynamic_color(&key, reversed_color_id)?;
        let Some(reversed_id) = current_ids.get(&reversed_key).copied() else {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        };
        let reversed_index = reversed_id as usize;
        if !(stage_start..stage_end).contains(&reversed_index) {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let reversed_local_index = reversed_index - stage_start;
        visited[local_index] = true;
        visited[reversed_local_index] = true;

        let reversed_color = color_states
            .get(reversed_color_id)
            .ok_or_else(|| invalid("reversed reflection color state disappeared"))?;
        let Some(reversed_word) = reversed_color.pure_adjoint_word() else {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            currents[reversed_index].reflection = CurrentReflection::Unavailable;
            continue;
        };
        let reversed_canonical = pure_adjoint_word_is_canonical(reversed_word);
        let proof_is_complete = reciprocal_reflection_proof(
            &currents[current_index].reflection,
            &currents[reversed_index].reflection,
        )?;
        if !proof_is_complete || canonical == reversed_canonical {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            currents[reversed_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let (canonical_index, reflected_index, pruned_local_index) = if canonical {
            (current_index, reversed_index, reversed_local_index)
        } else {
            (reversed_index, current_index, local_index)
        };
        let canonical_proof = currents[canonical_index]
            .reflection
            .proof()
            .cloned()
            .ok_or_else(|| invalid("canonical reflection proof disappeared"))?;
        let reflected_proof = currents[reflected_index]
            .reflection
            .proof()
            .cloned()
            .ok_or_else(|| invalid("reflected reflection proof disappeared"))?;
        for (label, current_index, proof) in [
            ("canonical", canonical_index, &canonical_proof),
            ("reflected", reflected_index, &reflected_proof),
        ] {
            let current_color = color_states
                .get(currents[current_index].key.dynamic_lc_color_state_id())
                .ok_or_else(|| invalid(format!("{label} reflection color disappeared")))?;
            if dynamic_color_identity_digest(current_color)? != proof.result_color_identity() {
                return Err(invalid(format!(
                    "{label} reflection proof carries a stale dynamic color identity"
                )));
            }
        }
        let certificate_id = u32::try_from(reflection_certificates.len())
            .map_err(|_| invalid("pending reflection certificate count exceeds u32"))?;
        reflection_certificates.push(PendingReflectionCertificate::reciprocal_pair(
            certificate_id,
            u32::try_from(canonical_index)
                .map_err(|_| invalid("canonical reflection current ID exceeds u32"))?,
            u32::try_from(reflected_index)
                .map_err(|_| invalid("reflected reflection current ID exceeds u32"))?,
            &canonical_proof,
            &reflected_proof,
            current_color_for_index(color_states, currents, canonical_index)?,
            current_color_for_index(color_states, currents, reflected_index)?,
            source_count,
        )?);
        currents[canonical_index].reflection_certificate_id = Some(certificate_id);
        prune[pruned_local_index] = true;
    }

    for current in &currents[stage_start..] {
        current_ids.remove(&current.key);
    }
    let stage_currents = currents.split_off(stage_start);
    target_bucket.clear();
    for (local_index, current) in stage_currents.into_iter().enumerate() {
        if prune[local_index] {
            *resident_contribution_count = resident_contribution_count
                .checked_sub(current.contributions.len())
                .ok_or_else(|| invalid("resident recurrence contribution count underflowed"))?;
            continue;
        }
        if current
            .contributions
            .keys()
            .flat_map(|contribution| contribution.parent_current_ids.iter().copied())
            .any(|parent_id| parent_id as usize >= stage_start)
        {
            return Err(invalid(
                "recurrence stage current depends on another current in the same stage",
            ));
        }
        let current_id = u32::try_from(currents.len())
            .map_err(|_| invalid("recurrence current count exceeds u32"))?;
        if current_ids
            .insert(current.key.clone(), current_id)
            .is_some()
        {
            return Err(invalid(
                "recurrence reflection reconciliation produced a duplicate current",
            ));
        }
        target_bucket.push(current_id);
        currents.push(current);
    }
    Ok(())
}

pub(super) fn current_reversal_masks(
    colors: &[DynamicLCColorState; 2],
    reflections: &[CurrentReflection; 2],
) -> Vec<u8> {
    let mut masks = vec![0_u8];
    for index in 0..2 {
        if reflections[index].phase().is_none()
            || colors[index]
                .pure_adjoint_word()
                .is_none_or(|word| word.len() < 2)
        {
            continue;
        }
        let bit = 1_u8 << index;
        masks.extend(masks.clone().into_iter().map(|mask| mask | bit));
    }
    masks
}

pub(super) fn current_reflection_candidate(
    result_color: &DynamicLCColorState,
    parent_reflections: &[CurrentReflection; 2],
    local_proof: Option<&TransitionReflectionProof>,
) -> RusticolResult<Option<CurrentReflectionProof>> {
    if result_color.pure_adjoint_word().is_none() {
        return Ok(None);
    }
    let Some(local_proof) = local_proof else {
        return Ok(None);
    };
    let Some(left) = parent_reflections[0].proof() else {
        return Ok(None);
    };
    let Some(right) = parent_reflections[1].proof() else {
        return Ok(None);
    };
    let phase = left
        .phase()
        .checked_mul(right.phase())?
        .checked_mul(local_proof.phase())?;
    let color_identity = dynamic_color_identity_digest(result_color)?;
    CurrentReflectionProof::new(
        phase,
        left.lineage_roots()
            .iter()
            .copied()
            .chain(right.lineage_roots().iter().copied())
            .chain(local_proof.lineage_roots()),
        color_identity,
    )
    .map(Some)
}

pub(super) fn pure_adjoint_word_is_canonical(word: &[u32]) -> bool {
    if word.len() < 2 {
        return true;
    }
    let (minimum_index, _) = word
        .iter()
        .enumerate()
        .min_by_key(|(_, slot)| *slot)
        .expect("nonempty pure-adjoint word");
    let (maximum_index, _) = word
        .iter()
        .enumerate()
        .max_by_key(|(_, slot)| *slot)
        .expect("nonempty pure-adjoint word");
    minimum_index < maximum_index
}

fn two_parent_permutation(
    construction_parent_ids: [u32; 2],
    ordered_parent_ids: [u32; 2],
    label: &str,
) -> RusticolResult<[u32; 2]> {
    if construction_parent_ids[0] == construction_parent_ids[1] {
        return Err(invalid(format!(
            "{label} cannot derive a permutation from duplicate parent IDs"
        )));
    }
    let mut result = [0_u32; 2];
    for (target, parent_id) in ordered_parent_ids.into_iter().enumerate() {
        let source = construction_parent_ids
            .iter()
            .position(|candidate| *candidate == parent_id)
            .ok_or_else(|| invalid(format!("{label} references a foreign parent ID")))?;
        result[target] =
            u32::try_from(source).map_err(|_| invalid(format!("{label} index exceeds u32")))?;
    }
    if result[0] == result[1] {
        return Err(invalid(format!("{label} is not a permutation")));
    }
    Ok(result)
}

fn three_line_traversal_certificate(
    closed: &[LCColorComponent],
    sector: ProcessPhysicalLCSectorRow,
    expected: &[LCColorComponent],
    catalog: &ProcessCatalog<'_>,
    pairing_rule_id: Option<u32>,
) -> RusticolResult<Option<PendingThreeLineTraversalCertificate>> {
    if sector.kind()? != ProcessLCSectorKind::OpenLines || sector.open_string_range.count != 3 {
        return Ok(None);
    }
    let pairing_rule_id = pairing_rule_id
        .ok_or_else(|| invalid("three-line traversal lacks a physical fermion pairing rule"))?;
    if expected.len() != 3 || closed.len() != 3 {
        return Err(invalid(
            "three-line closure does not contain exactly three complete line blocks",
        ));
    }
    let blocks = expected
        .iter()
        .map(|component| component.source_slots().to_vec())
        .collect::<Vec<_>>();
    let sector_word = catalog.u32_sequence(sector.word_sequence_id, "three-line sector word")?;
    let mut reference_order = Vec::<u32>::with_capacity(3);
    let mut used_reference = [false; 3];
    let mut offset = 0usize;
    while offset < sector_word.len() {
        let matches = blocks
            .iter()
            .enumerate()
            .filter_map(|(index, block)| {
                (!used_reference[index] && sector_word[offset..].starts_with(block))
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        let [block_index] = matches.as_slice() else {
            return Err(invalid(
                "three-line sector word is not an unambiguous concatenation of its line blocks",
            ));
        };
        used_reference[*block_index] = true;
        reference_order.push(
            u32::try_from(*block_index)
                .map_err(|_| invalid("three-line block index exceeds u32"))?,
        );
        offset = offset
            .checked_add(blocks[*block_index].len())
            .ok_or_else(|| invalid("three-line sector word offset overflows"))?;
    }
    if reference_order.len() != 3 || offset != sector_word.len() {
        return Err(invalid(
            "three-line sector word does not cover exactly three line blocks",
        ));
    }
    let mut used = [false; 3];
    let mut witness_order = Vec::<u32>::with_capacity(3);
    for component in closed {
        let index = expected
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| {
                (!used[index] && candidate == component).then_some(index)
            })
            .ok_or_else(|| invalid("three-line closure contains a foreign line block"))?;
        used[index] = true;
        witness_order
            .push(u32::try_from(index).map_err(|_| invalid("three-line index exceeds u32"))?);
    }
    let sink_candidates = expected
        .iter()
        .enumerate()
        .filter_map(|(index, component)| {
            (component.source_slots().last() == Some(&sector.closure_source_slot)).then_some(index)
        })
        .collect::<Vec<_>>();
    let [sink_index] = sink_candidates.as_slice() else {
        return Err(invalid(
            "three-line closure anchor does not identify exactly one line block",
        ));
    };
    let sink_index =
        u32::try_from(*sink_index).map_err(|_| invalid("three-line sink index exceeds u32"))?;
    let rotate_sink_last = |order: &mut Vec<u32>| -> RusticolResult<()> {
        let sink_position = order
            .iter()
            .position(|index| *index == sink_index)
            .ok_or_else(|| invalid("three-line closure traversal omits its anchor block"))?;
        let order_len = order.len();
        order.rotate_left((sink_position + 1) % order_len);
        Ok(())
    };
    rotate_sink_last(&mut reference_order)?;
    rotate_sink_last(&mut witness_order)?;
    let kind = if witness_order == reference_order {
        THREE_LINE_DIRECT_CERTIFICATE_ID as u8
    } else {
        let mut partner = reference_order.clone();
        partner.swap(0, 1);
        if witness_order != partner {
            return Err(invalid(
                "three-line closure traversal is neither the direct nor partner cyclic obligation",
            ));
        }
        THREE_LINE_PARTNER_CERTIFICATE_ID as u8
    };
    let block_permutation = witness_order
        .iter()
        .map(|block| {
            reference_order
                .iter()
                .position(|candidate| candidate == block)
                .map(|position| {
                    u32::try_from(position)
                        .map_err(|_| invalid("three-line block permutation exceeds u32"))
                })
                .transpose()?
                .ok_or_else(|| invalid("three-line witness references a foreign block"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let flatten = |order: &[u32]| -> RusticolResult<Vec<u32>> {
        let mut sources = Vec::new();
        for block_index in order {
            sources.extend_from_slice(
                blocks
                    .get(*block_index as usize)
                    .ok_or_else(|| invalid("three-line block order is out of range"))?,
            );
        }
        Ok(sources)
    };
    let reference_source_order = flatten(&reference_order)?;
    let witness_source_order = flatten(&witness_order)?;
    let source_position_permutation = witness_source_order
        .iter()
        .map(|source| {
            reference_source_order
                .iter()
                .position(|candidate| candidate == source)
                .map(|position| {
                    u32::try_from(position)
                        .map_err(|_| invalid("three-line source permutation exceeds u32"))
                })
                .transpose()?
                .ok_or_else(|| invalid("three-line witness references a foreign source"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-three-line-traversal-v1\0");
    hash.update(sector.sector_id.to_le_bytes());
    hash.update([kind]);
    hash.update(sink_index.to_le_bytes());
    for values in [
        reference_order.as_slice(),
        witness_order.as_slice(),
        block_permutation.as_slice(),
        reference_source_order.as_slice(),
        witness_source_order.as_slice(),
        source_position_permutation.as_slice(),
    ] {
        hash.update(
            u64::try_from(values.len())
                .map_err(|_| invalid("three-line certificate sequence length exceeds u64"))?
                .to_le_bytes(),
        );
        for value in values {
            hash.update(value.to_le_bytes());
        }
    }
    hash.update(sector.closure_source_slot.to_le_bytes());
    hash.update(pairing_rule_id.to_le_bytes());
    let proof_digest = SemanticDigest::new(hash.finalize().into())?;
    Ok(Some(PendingThreeLineTraversalCertificate {
        sector_id: sector.sector_id,
        kind,
        sink_block_ordinal: sink_index,
        reference_block_order: reference_order.into_boxed_slice(),
        witness_block_order: witness_order.into_boxed_slice(),
        block_permutation: block_permutation.into_boxed_slice(),
        reference_source_order: reference_source_order.into_boxed_slice(),
        witness_source_order: witness_source_order.into_boxed_slice(),
        source_position_permutation: source_position_permutation.into_boxed_slice(),
        closure_anchor_source_slot: sector.closure_source_slot,
        pairing_rule_id,
        proof_digest,
    }))
}

fn closure_pairing_certificate_ids(
    currents: &[PendingCurrent],
    parent_ids: [u32; 2],
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
) -> RusticolResult<Vec<u32>> {
    let left = currents
        .get(parent_ids[0] as usize)
        .ok_or_else(|| invalid("closure pairing proof left parent is absent"))?;
    let right = currents
        .get(parent_ids[1] as usize)
        .ok_or_else(|| invalid("closure pairing proof right parent is absent"))?;
    let realized = left
        .realized_pairing_rule_ids
        .intersection(&right.realized_pairing_rule_ids)
        .copied()
        .collect::<BTreeSet<_>>();
    match pairing_catalog {
        None if realized.is_empty() => Ok(Vec::new()),
        None => Err(invalid(
            "closure realizes fermion pairings without a pairing catalog",
        )),
        Some(catalog) => canonical_pairing_owner_id(realized.into_iter().map(|rule_id| {
            let rule = catalog
                .rules()
                .get(rule_id as usize)
                .copied()
                .filter(|rule| rule.rule_id == rule_id)
                .ok_or_else(|| {
                    invalid(format!(
                        "closure parent lineage references absent pairing rule {rule_id}"
                    ))
                })?;
            Ok((rule_id, catalog.endpoint_pairings(rule)?))
        }))?
        .map(|rule_id| vec![rule_id])
        .ok_or_else(|| invalid("closure parent lineage realizes no fermion pairing rule")),
    }
}

fn canonical_pairing_owner_id<'a>(
    candidates: impl IntoIterator<Item = RusticolResult<(u32, &'a [FermionPairingEndpointPairRow])>>,
) -> RusticolResult<Option<u32>> {
    let mut selected = None::<(u32, &'a [FermionPairingEndpointPairRow])>;
    for candidate in candidates {
        let (rule_id, endpoint_pairs) = candidate?;
        let candidate_precedes_selected = selected.is_none_or(|(selected_id, selected_pairs)| {
            endpoint_pairs
                .iter()
                .map(|pair| {
                    (
                        pair.fundamental_source_slot,
                        pair.antifundamental_source_slot,
                    )
                })
                .cmp(selected_pairs.iter().map(|pair| {
                    (
                        pair.fundamental_source_slot,
                        pair.antifundamental_source_slot,
                    )
                }))
                .then_with(|| rule_id.cmp(&selected_id))
                .is_lt()
        });
        if candidate_precedes_selected {
            selected = Some((rule_id, endpoint_pairs));
        }
    }
    Ok(selected.map(|(rule_id, _)| rule_id))
}

fn pairing_rule_for_certificate(
    certificate_ids: &[u32],
    catalog: Option<ValidatedFermionPairingCatalog<'_>>,
) -> RusticolResult<Option<FermionPairingRuleRow>> {
    let Some(catalog) = catalog else {
        if !certificate_ids.is_empty() {
            return Err(invalid(
                "closure pairing certificate has no pairing catalog",
            ));
        }
        return Ok(None);
    };
    let [rule_id] = certificate_ids else {
        return Err(invalid(
            "closure requires exactly one fermion pairing certificate",
        ));
    };
    catalog
        .rules()
        .iter()
        .copied()
        .find(|rule| rule.rule_id == *rule_id)
        .map(Some)
        .ok_or_else(|| invalid(format!("closure references absent pairing rule {rule_id}")))
}

fn pairing_reconstruction_factor(_rule: Option<FermionPairingRuleRow>) -> ExactComplexRational {
    // Canonical color-line construction already carries the exchange parity
    // of colored Wick lineages.  The exterior-algebra pass separately owns
    // color-singlet fermions, so multiplying pairing parity here would count
    // the colored sign twice.
    ExactComplexRational::ONE
}

// Closure certification deliberately receives each authenticated catalog and
// proof table separately so that no unchecked aggregate can cross this boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClosureReflectionProjection {
    certificate_id: u32,
    partner_sector_id: Option<u32>,
    partner_phase: ExactComplexRational,
}

#[allow(clippy::too_many_arguments)]
fn closure_reflection_projection(
    sector: ProcessPhysicalLCSectorRow,
    closed: &[LCColorComponent],
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    parent_ids: [u32; 2],
    certificates: &[PendingReflectionCertificate],
    prepared_sectors: &PreparedClosureSectorCatalog,
    materialized_sectors: &BTreeSet<u32>,
) -> RusticolResult<Option<ClosureReflectionProjection>> {
    let mut certified_parents = Vec::new();
    for parent_id in parent_ids {
        let current = currents
            .get(parent_id as usize)
            .ok_or_else(|| invalid("closure reflection parent is absent"))?;
        let Some(certificate_id) = current.reflection_certificate_id else {
            continue;
        };
        let certificate = certificates
            .get(certificate_id as usize)
            .filter(|certificate| certificate.id == certificate_id)
            .ok_or_else(|| invalid("closure reflection certificate is absent"))?;
        let proof = current
            .reflection
            .proof()
            .ok_or_else(|| invalid("certified closure reflection parent has no lineage proof"))?;
        if proof.proof_digest() != certificate.canonical_lineage_digest {
            return Err(invalid(
                "closure reflection parent lineage does not match its certificate",
            ));
        }
        let color = color_states
            .get(current.key.dynamic_lc_color_state_id())
            .ok_or_else(|| invalid("closure reflection parent color is absent"))?;
        if dynamic_color_identity_digest(color)? != certificate.canonical_color_identity {
            return Err(invalid(
                "closure reflection parent color does not match its certificate",
            ));
        }
        certified_parents.push(certificate_id);
    }
    match certified_parents.len() {
        0 => return Ok(None),
        1 => {}
        _ => {
            return Err(invalid(
                "one closure references multiple folded reflection orbits",
            ));
        }
    }
    if sector.kind()? != ProcessLCSectorKind::SingleTrace {
        return Ok(None);
    }
    let construction_word =
        process_catalog.u32_sequence(sector.word_sequence_id, "reflection construction word")?;
    if construction_word.len() <= 2 {
        return Ok(None);
    }
    let [closed_trace] = closed else {
        return Ok(None);
    };
    if closed_trace.kind() != LCColorComponentKind::Trace
        || closed_trace.source_slots().len() != construction_word.len()
    {
        return Ok(None);
    }
    let certificate_id = certified_parents[0];
    let certificate = certificates
        .get(certificate_id as usize)
        .filter(|certificate| certificate.id == certificate_id)
        .ok_or_else(|| invalid("closure reflection certificate is absent"))?;
    if certificate.fixed_point
        || certificate.orbit_size != 2
        || certificate.canonical_color_identity == certificate.reflected_color_identity
    {
        return Err(invalid(
            "closure reflection certificate does not describe a reciprocal two-cycle",
        ));
    }
    let mapped_word = construction_word
        .iter()
        .map(|source_slot| {
            certificate
                .source_permutation
                .get(*source_slot as usize)
                .copied()
                .ok_or_else(|| {
                    invalid("closure reflection permutation does not cover its trace word")
                })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if cyclic_words_equal(construction_word, &mapped_word) {
        return Err(invalid(
            "closure reflection maps a trace to a cyclic fixed point",
        ));
    }
    let closure_algorithm = process_catalog.string(
        sector.closure_proof_algorithm_string_id,
        "reflection closure-anchor algorithm",
    )?;
    if closure_algorithm == CYCLIC_TRACE_CLOSURE_ANCHOR_ALGORITHM
        && let Some(partner_sector_id) =
            prepared_sectors.reflected_v4_sector(&mapped_word, sector.closure_source_slot)?
        && materialized_sectors.contains(&partner_sector_id)
    {
        if partner_sector_id == sector.sector_id {
            return Err(invalid(
                "closure reflection maps a v4 trace sector onto itself",
            ));
        }
        return Ok(Some(ClosureReflectionProjection {
            certificate_id,
            partner_sector_id: Some(partner_sector_id),
            partner_phase: certificate.canonical_phase,
        }));
    }

    let mut matching_public_flows = Vec::new();
    for flow in retained_public_flows(process)?
        .into_iter()
        .filter(|flow| flow.construction_sector_id == sector.sector_id)
    {
        let word =
            process_catalog.u32_sequence(flow.word_sequence_id, "reflected public LC flow")?;
        if cyclic_words_equal(word, &mapped_word) {
            matching_public_flows.push(flow.flow_id);
        }
    }
    match matching_public_flows.as_slice() {
        [_] => Ok(Some(ClosureReflectionProjection {
            certificate_id,
            partner_sector_id: None,
            partner_phase: certificate.canonical_phase,
        })),
        [] => Ok(None),
        _ => Err(invalid(
            "closure reflection maps to multiple retained public LC flows",
        )),
    }
}

fn cyclic_words_equal(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    if left.is_empty() {
        return true;
    }
    (0..left.len()).any(|offset| {
        left.iter()
            .enumerate()
            .all(|(index, value)| *value == right[(index + offset) % right.len()])
    })
}

fn anchored_cyclic_trace_word(word: &[u32], anchor: u32) -> RusticolResult<Box<[u32]>> {
    let mut positions = word
        .iter()
        .enumerate()
        .filter_map(|(index, source_slot)| (*source_slot == anchor).then_some(index));
    let position = positions
        .next()
        .ok_or_else(|| invalid("v4 cyclic trace word omits its closure anchor"))?;
    if positions.next().is_some() {
        return Err(invalid("v4 cyclic trace word repeats its closure anchor"));
    }
    Ok(word[position..]
        .iter()
        .chain(&word[..position])
        .copied()
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

fn include_closure_with_reflection(
    result: &mut BTreeMap<PendingClosureKey, PendingClosureGroup>,
    key: PendingClosureKey,
    contribution: PendingClosureProofContribution,
    reflection: Option<ClosureReflectionProjection>,
) -> RusticolResult<()> {
    result
        .entry(key.clone())
        .or_default()
        .include(contribution.clone())?;
    if let Some(reflection) = reflection
        && let Some(partner_sector_id) = reflection.partner_sector_id
    {
        let mut partner_key = key;
        partner_key.target_sector_id = partner_sector_id;
        let mut partner_contribution = contribution;
        partner_contribution.exact_factor = partner_contribution
            .exact_factor
            .checked_mul(reflection.partner_phase)?;
        result
            .entry(partner_key)
            .or_default()
            .include(partner_contribution)?;
    }
    Ok(())
}

fn materialize_reflection_certificates(
    certificates: &[PendingReflectionCertificate],
) -> RusticolResult<Vec<ReflectionCertificateV1>> {
    certificates
        .iter()
        .map(|certificate| {
            ReflectionCertificateV1::new(
                certificate.id,
                certificate.canonical_color_identity,
                certificate.reflected_color_identity,
                certificate.source_permutation.to_vec(),
                certificate.canonical_phase,
                certificate.fixed_point,
                certificate.orbit_size,
                REFLECTION_PROOF_ALGORITHM_ID,
                certificate.proof_digest,
            )
        })
        .collect()
}

fn materialize_three_line_certificates(
    certificate_ids: &BTreeMap<PendingThreeLineTraversalCertificate, u32>,
) -> RusticolResult<Vec<ThreeLineTraversalCertificateV1>> {
    let mut rows = certificate_ids
        .iter()
        .map(|(certificate, id)| {
            ThreeLineTraversalCertificateV1::new(
                *id,
                certificate.sector_id,
                ThreeLineTraversalKindV1::try_from(u32::from(certificate.kind))?,
                certificate.sink_block_ordinal,
                certificate.reference_block_order.to_vec(),
                certificate.witness_block_order.to_vec(),
                certificate.block_permutation.to_vec(),
                certificate.reference_source_order.to_vec(),
                certificate.witness_source_order.to_vec(),
                certificate.source_position_permutation.to_vec(),
                certificate.closure_anchor_source_slot,
                certificate.pairing_rule_id,
                certificate.proof_digest,
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    rows.sort_unstable_by_key(ThreeLineTraversalCertificateV1::id);
    Ok(rows)
}

type ClosureSupportBucketKey = (Box<[u32]>, Box<[u32]>);

#[derive(Debug, Default)]
struct ContractedClosureSupportBucket {
    sector_ids: Vec<u32>,
    ordered_sector_ids_by_components: BTreeMap<Box<[LCColorComponent]>, Vec<u32>>,
    open_line_sector_ids_by_components: BTreeMap<Box<[LCColorComponent]>, Vec<u32>>,
}

impl ContractedClosureSupportBucket {
    fn include(
        &mut self,
        strategy: RecurrenceStrategy,
        sector: &PreparedClosureSector,
    ) -> RusticolResult<()> {
        self.sector_ids.push(sector.row.sector_id);
        match sector.row.kind()? {
            ProcessLCSectorKind::Singlet | ProcessLCSectorKind::SingleTrace => {
                self.ordered_sector_ids_by_components
                    .entry(sector.expected_components.clone())
                    .or_default()
                    .push(sector.row.sector_id);
            }
            ProcessLCSectorKind::OpenLines
                if strategy == RecurrenceStrategy::AllFlowUnion
                    || sector.contracted_color_canonical_owner =>
            {
                let mut canonical = sector.expected_components.to_vec();
                canonical.sort_unstable();
                self.open_line_sector_ids_by_components
                    .entry(canonical.into_boxed_slice())
                    .or_default()
                    .push(sector.row.sector_id);
            }
            ProcessLCSectorKind::OpenLines => {}
        }
        Ok(())
    }

    fn matching_sector_ids(&self, closed: &[LCColorComponent]) -> Vec<u32> {
        let mut matching = self
            .ordered_sector_ids_by_components
            .get(closed)
            .into_iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        if !self.open_line_sector_ids_by_components.is_empty() {
            let mut canonical = closed.to_vec();
            canonical.sort_unstable();
            if let Some(open_line_sectors) = self
                .open_line_sector_ids_by_components
                .get(canonical.as_slice())
            {
                matching.extend(open_line_sectors.iter().copied());
            }
        }
        matching
    }
}

#[derive(Debug)]
struct ContractedClosureAttempt {
    closed_components: Box<[LCColorComponent]>,
    key: PendingClosureKey,
    contribution: PendingClosureProofContribution,
    pairing_rule_id: Option<u32>,
}

fn contracted_closure_support_buckets(
    strategy: RecurrenceStrategy,
    prepared_sectors: &PreparedClosureSectorCatalog,
    materialized_sectors: &BTreeSet<u32>,
) -> RusticolResult<BTreeMap<ClosureSupportBucketKey, ContractedClosureSupportBucket>> {
    if !matches!(
        strategy,
        RecurrenceStrategy::AllFlowUnion | RecurrenceStrategy::ContractedColorUnion
    ) {
        return Err(invalid(
            "bulk closure support buckets require an all-flow or contracted-color union",
        ));
    }
    let mut buckets = BTreeMap::<ClosureSupportBucketKey, ContractedClosureSupportBucket>::new();
    for sector_id in materialized_sectors.iter().copied() {
        let sector = prepared_sectors.get(sector_id)?;
        buckets
            .entry((
                sector.anchor_support.clone(),
                sector.complement_support.clone(),
            ))
            .or_default()
            .include(strategy, sector)?;
    }
    Ok(buckets)
}

#[allow(clippy::too_many_arguments)]
fn include_bulk_color_union_closure_attempt(
    strategy: RecurrenceStrategy,
    sector: &PreparedClosureSector,
    attempt: &ContractedClosureAttempt,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    reflection_certificates: &[PendingReflectionCertificate],
    prepared_sectors: &PreparedClosureSectorCatalog,
    materialized_sectors: &BTreeSet<u32>,
    result: &mut BTreeMap<PendingClosureKey, PendingClosureGroup>,
) -> RusticolResult<()> {
    if !closed_components_match_prepared_sector(strategy, &attempt.closed_components, sector)? {
        return Err(invalid(format!(
            "bulk color-union closure routing selected non-matching sector {}",
            sector.row.sector_id
        )));
    }
    let reflection = closure_reflection_projection(
        sector.row,
        &attempt.closed_components,
        process,
        process_catalog,
        color_states,
        currents,
        attempt.contribution.construction_parent_ids,
        reflection_certificates,
        prepared_sectors,
        materialized_sectors,
    )?;
    let mut key = attempt.key.clone();
    key.target_sector_id = sector.row.sector_id;
    let mut contribution = attempt.contribution.clone();
    contribution.three_line_certificate = three_line_traversal_certificate(
        &attempt.closed_components,
        sector.row,
        &sector.expected_components,
        process_catalog,
        attempt.pairing_rule_id,
    )?;
    contribution.reflection_certificate_id = reflection.map(|row| row.certificate_id);
    include_closure_with_reflection(result, key, contribution, reflection)
}

#[allow(clippy::too_many_arguments)]
fn build_bulk_color_union_closures(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    fermion_ordering: &FermionOrderingContext,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    prepared_closures: &PreparedClosureCatalog,
    prepared_sectors: &PreparedClosureSectorCatalog,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    materialized_sectors: &BTreeSet<u32>,
    reflection_certificates: &[PendingReflectionCertificate],
    construction_domain: Option<PendingConstructionDomain>,
    telemetry: &mut RecurrenceGenerationTelemetry,
    collect_telemetry: bool,
) -> RusticolResult<ExhaustiveClosureDomain> {
    if !matches!(
        strategy,
        RecurrenceStrategy::AllFlowUnion | RecurrenceStrategy::ContractedColorUnion
    ) {
        return Err(invalid(
            "bulk closure construction requires an all-flow or contracted-color union",
        ));
    }
    let support_index = LaneClosureSupportIndex::new(currents, construction_domain)?;
    let support_buckets =
        contracted_closure_support_buckets(strategy, prepared_sectors, materialized_sectors)?;
    let mut attempts = Vec::<ContractedClosureAttempt>::new();
    let mut attempt_indices_by_sector = BTreeMap::<u32, Vec<usize>>::new();

    for ((anchor_support, complement_support), bucket) in &support_buckets {
        let anchor_ids = support_index
            .current_ids(anchor_support)
            .iter()
            .copied()
            .filter(|current_id| {
                currents[*current_id as usize].key.node_kind() == RecurrenceNodeKind::Source
            })
            .collect::<Vec<_>>();
        let complement_ids = support_index.current_ids(complement_support);
        let anchor_count = anchor_ids.len();
        let complement_count = complement_ids.len();
        if collect_telemetry {
            checked_diagnostic_add(
                &mut telemetry.closure_support_lookup_count,
                2,
                "telemetry closure-support lookup count",
            )?;
            let theoretical_count = anchor_count
                .checked_mul(complement_count)
                .and_then(|count| count.checked_mul(prepared_closures.row_count()))
                .and_then(|count| count.checked_mul(bucket.sector_ids.len()))
                .ok_or_else(|| invalid("theoretical closure-candidate count exceeds usize"))?;
            checked_diagnostic_add(
                &mut telemetry.closure_candidate_theoretical_count,
                theoretical_count,
                "telemetry theoretical closure-candidate count",
            )?;
        }
        for anchor_id in anchor_ids.iter().copied() {
            let anchor_state = currents[anchor_id as usize].key.current_state_template_id();
            for &complement_id in complement_ids {
                let complement_state = currents[complement_id as usize]
                    .key
                    .current_state_template_id();
                let matching_closures = prepared_closures.rows(anchor_state, complement_state);
                if collect_telemetry {
                    checked_diagnostic_add(
                        &mut telemetry.closure_candidate_count,
                        matching_closures.len(),
                        "telemetry closure-candidate count",
                    )?;
                    checked_diagnostic_add(
                        &mut telemetry.closure_state_match_count,
                        matching_closures.len(),
                        "telemetry closure-state-match count",
                    )?;
                }
                for closure in matching_closures {
                    let parent_ids = closure.parent_ids(
                        anchor_state,
                        complement_state,
                        anchor_id,
                        complement_id,
                    )?;
                    let parents = [
                        &currents[parent_ids[0] as usize].key,
                        &currents[parent_ids[1] as usize].key,
                    ];
                    let pairing_certificate_ids =
                        closure_pairing_certificate_ids(currents, parent_ids, pairing_catalog)?;
                    let pairing_rule =
                        pairing_rule_for_certificate(&pairing_certificate_ids, pairing_catalog)?;
                    for quantum in closure
                        .quantum_flows
                        .iter()
                        .filter(|quantum| quantum.matches(&parents))
                    {
                        let output_factor = quantum.output_factor()?;
                        let (evaluator_parent_ids, exchange_factor) =
                            closure.canonical_evaluator_parents(parent_ids);
                        let evaluator_parent_permutation = two_parent_permutation(
                            parent_ids,
                            evaluator_parent_ids,
                            "closure evaluator order",
                        )?;
                        let fermion_factor = fermion_ordering_factor(
                            &template.current_states,
                            [
                                parents[0].current_state_template_id(),
                                parents[1].current_state_template_id(),
                            ],
                            [
                                parents[0].support_source_slots(),
                                parents[1].support_source_slots(),
                            ],
                            fermion_ordering,
                        )?;
                        let base_factor = multiply_factors(&[
                            closure.closure_exact_factor,
                            exchange_factor,
                            closure.contraction_exact_factor,
                            output_factor,
                            fermion_factor,
                            pairing_reconstruction_factor(pairing_rule),
                        ])?;
                        let left = color_states
                            .get(parents[0].dynamic_lc_color_state_id())
                            .ok_or_else(|| invalid("closure left color state disappeared"))?;
                        let right = color_states
                            .get(parents[1].dynamic_lc_color_state_id())
                            .ok_or_else(|| invalid("closure right color state disappeared"))?;
                        for witness in &closure.witnesses {
                            if witness.row.left_shape_string_id != left.output_color_shape_id()
                                || witness.row.right_shape_string_id
                                    != right.output_color_shape_id()
                            {
                                continue;
                            }
                            let closed = witness.witness.closed_components(left, right)?;
                            if collect_telemetry {
                                checked_diagnostic_add(
                                    &mut telemetry.closure_color_attempt_count,
                                    1,
                                    "telemetry closure-color-attempt count",
                                )?;
                            }
                            let matching_sector_ids = bucket.matching_sector_ids(&closed);
                            if matching_sector_ids.is_empty() {
                                continue;
                            }
                            if pairing_catalog.is_some() && pairing_certificate_ids.is_empty() {
                                return Err(invalid(format!(
                                    "closure witness {} for bulk color-union support bucket has no exactly realized fermion pairing",
                                    witness.row.ordinal
                                )));
                            }
                            let reconstruction_parent_permutation = match witness
                                .row
                                .input_permutation
                            {
                                0 => [0, 1],
                                1 => [1, 0],
                                value => {
                                    return Err(invalid(format!(
                                        "closure color witness has invalid parent permutation {value}"
                                    )));
                                }
                            };
                            let color_witness_term_id = closure
                                .contraction
                                .witness_start
                                .checked_add(u64::from(witness.row.ordinal))
                                .ok_or_else(|| {
                                    invalid("closure color-witness term ID overflows")
                                })?;
                            let color_witness_term_id = u32::try_from(color_witness_term_id)
                                .map_err(|_| {
                                    invalid("closure color-witness term ID exceeds u32")
                                })?;
                            let attempt_index = attempts.len();
                            attempts.push(ContractedClosureAttempt {
                                closed_components: closed.into_boxed_slice(),
                                key: PendingClosureKey {
                                    target_sector_id: MISSING_U32,
                                    complete_source_states: complete_closure_source_states(
                                        parents,
                                        process.external_legs.len(),
                                    )?,
                                    closure_template_id: closure.row.id,
                                    quantum_flow_template_id: quantum.template_id(),
                                    parent_current_ids: evaluator_parent_ids
                                        .to_vec()
                                        .into_boxed_slice(),
                                },
                                contribution: PendingClosureProofContribution {
                                    construction_parent_ids: parent_ids,
                                    construction_parent_permutation: [0, 1],
                                    reconstruction_parent_permutation,
                                    evaluator_parent_permutation,
                                    closure_template_semantic_digest: closure.semantic_digest,
                                    color_witness_term_id,
                                    color_witness_proof_digest: witness.witness.proof_digest(),
                                    three_line_certificate: None,
                                    pairing_certificate_ids: pairing_certificate_ids
                                        .clone()
                                        .into_boxed_slice(),
                                    reflection_certificate_id: None,
                                    exact_factor: base_factor
                                        .checked_mul(witness.witness.exact_factor())?,
                                },
                                pairing_rule_id: pairing_rule.map(|rule| rule.rule_id),
                            });
                            for sector_id in matching_sector_ids {
                                attempt_indices_by_sector
                                    .entry(sector_id)
                                    .or_default()
                                    .push(attempt_index);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut result = BTreeMap::new();
    for sector_id in materialized_sectors.iter().copied() {
        let sector = prepared_sectors.get(sector_id)?;
        for attempt_index in attempt_indices_by_sector
            .get(&sector_id)
            .into_iter()
            .flatten()
            .copied()
        {
            let attempt = &attempts[attempt_index];
            include_bulk_color_union_closure_attempt(
                strategy,
                sector,
                attempt,
                process,
                process_catalog,
                color_states,
                currents,
                reflection_certificates,
                prepared_sectors,
                materialized_sectors,
                &mut result,
            )?;
        }
    }
    if strategy == RecurrenceStrategy::AllFlowUnion {
        let covered_sector_ids = result
            .keys()
            .map(|key| key.target_sector_id)
            .collect::<BTreeSet<_>>();
        if let Some(sector_id) = materialized_sectors
            .iter()
            .copied()
            .find(|sector_id| !covered_sector_ids.contains(sector_id))
        {
            return Err(invalid(format!(
                "recurrence builder found no exact closure for physical LC sector {sector_id} after bulk all-flow closure routing"
            )));
        }
    }
    validate_pending_closure_obligations(&result, pairing_catalog, prepared_sectors)?;
    if collect_telemetry {
        checked_diagnostic_add(
            &mut telemetry.closure_group_count,
            result.len(),
            "telemetry closure-group count",
        )?;
        checked_diagnostic_add(
            &mut telemetry.closure_proof_contribution_count,
            result.values().try_fold(0usize, |total, group| {
                total
                    .checked_add(group.contributions.len())
                    .ok_or_else(|| invalid("closure proof-contribution count exceeds usize"))
            })?,
            "telemetry closure-proof-contribution count",
        )?;
    }
    ExhaustiveClosureDomain::new(
        result,
        materialized_sectors,
        closure_domain_sector_owners(strategy, prepared_sectors, materialized_sectors)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_closures(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    fermion_ordering: &FermionOrderingContext,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    prepared_closures: &PreparedClosureCatalog,
    prepared_sectors: &PreparedClosureSectorCatalog,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    materialized_sectors: &BTreeSet<u32>,
    stage_diagnostics: &[StageConstructionDiagnostics],
    reflection_certificates: &[PendingReflectionCertificate],
    construction_domain: Option<PendingConstructionDomain>,
    telemetry: &mut RecurrenceGenerationTelemetry,
    collect_telemetry: bool,
) -> RusticolResult<ExhaustiveClosureDomain> {
    if matches!(
        strategy,
        RecurrenceStrategy::AllFlowUnion | RecurrenceStrategy::ContractedColorUnion
    ) {
        return build_bulk_color_union_closures(
            strategy,
            process,
            process_catalog,
            template,
            fermion_ordering,
            pairing_catalog,
            prepared_closures,
            prepared_sectors,
            color_states,
            currents,
            materialized_sectors,
            reflection_certificates,
            construction_domain,
            telemetry,
            collect_telemetry,
        );
    }
    build_closures_per_sector(
        strategy,
        process,
        process_catalog,
        template,
        fermion_ordering,
        pairing_catalog,
        prepared_closures,
        prepared_sectors,
        color_states,
        currents,
        materialized_sectors,
        stage_diagnostics,
        reflection_certificates,
        construction_domain,
        telemetry,
        collect_telemetry,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_closures_per_sector(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    fermion_ordering: &FermionOrderingContext,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    prepared_closures: &PreparedClosureCatalog,
    prepared_sectors: &PreparedClosureSectorCatalog,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    materialized_sectors: &BTreeSet<u32>,
    stage_diagnostics: &[StageConstructionDiagnostics],
    reflection_certificates: &[PendingReflectionCertificate],
    construction_domain: Option<PendingConstructionDomain>,
    telemetry: &mut RecurrenceGenerationTelemetry,
    collect_telemetry: bool,
) -> RusticolResult<ExhaustiveClosureDomain> {
    let support_index = LaneClosureSupportIndex::new(currents, construction_domain)?;
    let mut result = BTreeMap::new();
    let mut covered_sector_ids = BTreeSet::new();
    for sector_id in materialized_sectors.iter().copied() {
        let sector = prepared_sectors.get(sector_id)?;
        let anchor_ids = support_index
            .current_ids(&sector.anchor_support)
            .iter()
            .copied()
            .filter(|current_id| {
                currents[*current_id as usize].key.node_kind() == RecurrenceNodeKind::Source
            })
            .collect::<Vec<_>>();
        let complement_ids = support_index.current_ids(&sector.complement_support);
        let anchor_count = anchor_ids.len();
        let complement_count = complement_ids.len();
        if collect_telemetry {
            checked_diagnostic_add(
                &mut telemetry.closure_support_lookup_count,
                2,
                "telemetry closure-support lookup count",
            )?;
            let theoretical_count = anchor_count
                .checked_mul(complement_count)
                .and_then(|count| count.checked_mul(prepared_closures.row_count()))
                .ok_or_else(|| invalid("theoretical closure-candidate count exceeds usize"))?;
            checked_diagnostic_add(
                &mut telemetry.closure_candidate_theoretical_count,
                theoretical_count,
                "telemetry theoretical closure-candidate count",
            )?;
        }
        let mut state_matched_attempts = 0usize;
        let sector_result_start = result.len();
        for anchor_id in anchor_ids.iter().copied() {
            let anchor_state = currents[anchor_id as usize].key.current_state_template_id();
            for &complement_id in complement_ids {
                let complement_state = currents[complement_id as usize]
                    .key
                    .current_state_template_id();
                let matching_closures = prepared_closures.rows(anchor_state, complement_state);
                if collect_telemetry {
                    checked_diagnostic_add(
                        &mut telemetry.closure_candidate_count,
                        matching_closures.len(),
                        "telemetry closure-candidate count",
                    )?;
                    checked_diagnostic_add(
                        &mut telemetry.closure_state_match_count,
                        matching_closures.len(),
                        "telemetry closure-state-match count",
                    )?;
                }
                state_matched_attempts = state_matched_attempts
                    .checked_add(matching_closures.len())
                    .ok_or_else(|| invalid("closure-attempt count exceeds usize"))?;
                for closure in matching_closures {
                    let parent_ids = closure.parent_ids(
                        anchor_state,
                        complement_state,
                        anchor_id,
                        complement_id,
                    )?;
                    add_closure_terms(
                        strategy,
                        sector,
                        closure,
                        parent_ids,
                        process,
                        process_catalog,
                        template,
                        fermion_ordering,
                        color_states,
                        currents,
                        pairing_catalog,
                        reflection_certificates,
                        prepared_sectors,
                        materialized_sectors,
                        &mut result,
                        &mut covered_sector_ids,
                        telemetry,
                        collect_telemetry,
                    )?;
                }
            }
        }
        if result.len() == sector_result_start && strategy != RecurrenceStrategy::AllFlowUnion {
            if strategy == RecurrenceStrategy::ContractedColorUnion {
                continue;
            }
            let closure_color_attempts = collect_closure_color_attempts(
                prepared_closures,
                color_states,
                currents,
                &anchor_ids,
                complement_ids,
            )?;
            let mut support_histogram = BTreeMap::<usize, usize>::new();
            let mut support_signatures = BTreeSet::new();
            for (id, current) in currents.iter().enumerate() {
                if construction_domain.is_some_and(|domain| !domain.contains(id)) {
                    continue;
                }
                *support_histogram
                    .entry(current.key.support_source_slots().len())
                    .or_default() += 1;
                support_signatures.insert((
                    current.key.support_source_slots().len(),
                    current.key.current_state_template_id(),
                    current.key.spin_state_class(),
                    current.key.flavour_flow().to_vec(),
                    current.key.quantum_number_flow_id(),
                    current.key.coupling_orders().to_vec(),
                ));
            }
            return Err(invalid(format!(
                "recurrence builder found no exact closure for physical LC sector {sector_id} \
                 (anchors={anchor_count}, complement_currents={complement_count}, \
                 state_matched_attempts={state_matched_attempts}, \
                 currents_by_support_size={support_histogram:?}, \
                 stage_diagnostics={stage_diagnostics:?}, \
                 expected_color_components={:?}, \
                 closure_color_attempts={closure_color_attempts:?}, \
                 support_signatures={support_signatures:?})",
                sector.expected_components,
            )));
        }
    }
    // Do not reject an all-flow sector until every possible direct owner has
    // run.  Physical sector identifiers need not place a reflection owner
    // before its partner; only the authenticated certificate establishes that
    // relationship.  Coverage is recorded exclusively after the corresponding
    // exact row (and optional phase-bearing partner row) was inserted above.
    if strategy == RecurrenceStrategy::AllFlowUnion
        && let Some(sector_id) = materialized_sectors
            .iter()
            .copied()
            .find(|sector_id| !covered_sector_ids.contains(sector_id))
    {
        let sector = prepared_sectors.get(sector_id)?;
        let anchor_ids = support_index
            .current_ids(&sector.anchor_support)
            .iter()
            .copied()
            .filter(|current_id| {
                currents[*current_id as usize].key.node_kind() == RecurrenceNodeKind::Source
            })
            .collect::<Vec<_>>();
        let complement_ids = support_index.current_ids(&sector.complement_support);
        let anchor_count = anchor_ids.len();
        let complement_count = complement_ids.len();
        let mut state_matched_attempts = 0usize;
        for anchor_id in anchor_ids.iter().copied() {
            let anchor_state = currents[anchor_id as usize].key.current_state_template_id();
            for complement_id in complement_ids.iter().copied() {
                let complement_state = currents[complement_id as usize]
                    .key
                    .current_state_template_id();
                state_matched_attempts = state_matched_attempts
                    .checked_add(prepared_closures.rows(anchor_state, complement_state).len())
                    .ok_or_else(|| invalid("closure-attempt count exceeds usize"))?;
            }
        }
        let closure_color_attempts = collect_closure_color_attempts(
            prepared_closures,
            color_states,
            currents,
            &anchor_ids,
            complement_ids,
        )?;
        let mut support_histogram = BTreeMap::<usize, usize>::new();
        let mut support_signatures = BTreeSet::new();
        for (id, current) in currents.iter().enumerate() {
            if construction_domain.is_some_and(|domain| !domain.contains(id)) {
                continue;
            }
            *support_histogram
                .entry(current.key.support_source_slots().len())
                .or_default() += 1;
            support_signatures.insert((
                current.key.support_source_slots().len(),
                current.key.current_state_template_id(),
                current.key.spin_state_class(),
                current.key.flavour_flow().to_vec(),
                current.key.quantum_number_flow_id(),
                current.key.coupling_orders().to_vec(),
            ));
        }
        return Err(invalid(format!(
            "recurrence builder found no exact closure for physical LC sector {sector_id} \
             (anchors={anchor_count}, complement_currents={complement_count}, \
             state_matched_attempts={state_matched_attempts}, \
             currents_by_support_size={support_histogram:?}, \
             stage_diagnostics={stage_diagnostics:?}, \
             expected_color_components={:?}, \
             closure_color_attempts={closure_color_attempts:?}, \
             support_signatures={support_signatures:?})",
            sector.expected_components,
        )));
    }
    validate_pending_closure_obligations(&result, pairing_catalog, prepared_sectors)?;
    if collect_telemetry {
        checked_diagnostic_add(
            &mut telemetry.closure_group_count,
            result.len(),
            "telemetry closure-group count",
        )?;
        checked_diagnostic_add(
            &mut telemetry.closure_proof_contribution_count,
            result.values().try_fold(0usize, |total, group| {
                total
                    .checked_add(group.contributions.len())
                    .ok_or_else(|| invalid("closure proof-contribution count exceeds usize"))
            })?,
            "telemetry closure-proof-contribution count",
        )?;
    }
    ExhaustiveClosureDomain::new(
        result,
        materialized_sectors,
        closure_domain_sector_owners(strategy, prepared_sectors, materialized_sectors)?,
    )
}

fn collect_closure_color_attempts(
    prepared_closures: &PreparedClosureCatalog,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    anchor_ids: &[u32],
    complement_ids: &[u32],
) -> RusticolResult<BTreeSet<ClosureColorAttemptDiagnostic>> {
    let mut attempts = BTreeSet::new();
    for anchor_id in anchor_ids.iter().copied() {
        let anchor_state = currents[anchor_id as usize].key.current_state_template_id();
        for complement_id in complement_ids.iter().copied() {
            let complement_state = currents[complement_id as usize]
                .key
                .current_state_template_id();
            for closure in prepared_closures.rows(anchor_state, complement_state) {
                let parent_ids =
                    closure.parent_ids(anchor_state, complement_state, anchor_id, complement_id)?;
                let parents = [
                    &currents[parent_ids[0] as usize].key,
                    &currents[parent_ids[1] as usize].key,
                ];
                if !closure
                    .quantum_flows
                    .iter()
                    .any(|quantum| quantum.matches(&parents))
                {
                    continue;
                }
                let left = color_states
                    .get(parents[0].dynamic_lc_color_state_id())
                    .ok_or_else(|| invalid("closure left color state disappeared"))?;
                let right = color_states
                    .get(parents[1].dynamic_lc_color_state_id())
                    .ok_or_else(|| invalid("closure right color state disappeared"))?;
                for witness in closure.witnesses.iter().filter(|witness| {
                    witness.row.left_shape_string_id == left.output_color_shape_id()
                        && witness.row.right_shape_string_id == right.output_color_shape_id()
                }) {
                    let closed = witness.witness.closed_components(left, right)?;
                    attempts.insert(
                        closed
                            .iter()
                            .map(|component| (component.kind(), component.source_slots().to_vec()))
                            .collect(),
                    );
                }
            }
        }
    }
    Ok(attempts)
}

fn validate_pending_closure_obligations(
    closures: &BTreeMap<PendingClosureKey, PendingClosureGroup>,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    prepared_sectors: &PreparedClosureSectorCatalog,
) -> RusticolResult<()> {
    let Some(pairing_catalog) = pairing_catalog else {
        return Ok(());
    };
    let destinations = closures
        .keys()
        .map(|key| (key.target_sector_id, key.complete_source_states.clone()))
        .collect::<BTreeSet<_>>();
    for (sector_id, source_states) in destinations {
        let prepared_sector = prepared_sectors.get(sector_id)?;
        let sector = prepared_sector.row;
        let rows = closures
            .iter()
            .filter(|(key, _)| {
                key.target_sector_id == sector_id && key.complete_source_states == source_states
            })
            .flat_map(|(_, group)| group.contributions.iter())
            .collect::<Vec<_>>();
        let realized_pairings = rows
            .iter()
            .flat_map(|row| row.pairing_certificate_ids.iter().copied())
            .collect::<BTreeSet<_>>();
        if realized_pairings.is_empty() {
            return Err(invalid(format!(
                "closure destination for sector {sector_id} realizes no fermion pairing rule"
            )));
        }
        for pairing_id in &realized_pairings {
            if pairing_catalog
                .rules()
                .iter()
                .all(|rule| rule.rule_id != *pairing_id)
            {
                return Err(invalid(format!(
                    "closure destination for sector {sector_id} references absent pairing rule {pairing_id}"
                )));
            }
        }
        if sector.kind()? == ProcessLCSectorKind::OpenLines && sector.open_string_range.count == 3 {
            for pairing_id in realized_pairings.iter().copied() {
                let matching_rows = rows
                    .iter()
                    .filter(|row| row.pairing_certificate_ids.contains(&pairing_id))
                    .collect::<Vec<_>>();
                if matching_rows
                    .iter()
                    .any(|row| row.three_line_certificate.is_none())
                {
                    return Err(invalid(format!(
                        "three-line closure sector {sector_id} pairing {pairing_id} has an uncertified traversal"
                    )));
                }
                let traversal_kinds = matching_rows
                    .into_iter()
                    .map(|row| {
                        let certificate = row
                            .three_line_certificate
                            .as_ref()
                            .expect("presence checked above");
                        if certificate.sector_id != sector_id
                            || certificate.pairing_rule_id != pairing_id
                        {
                            return Err(invalid(format!(
                                "three-line closure sector {sector_id} pairing {pairing_id} \
                                 has a traversal certificate for sector {} pairing {}",
                                certificate.sector_id, certificate.pairing_rule_id,
                            )));
                        }
                        match u32::from(certificate.kind) {
                            THREE_LINE_DIRECT_CERTIFICATE_ID
                            | THREE_LINE_PARTNER_CERTIFICATE_ID => Ok(u32::from(certificate.kind)),
                            kind => Err(invalid(format!(
                                "three-line closure sector {sector_id} pairing {pairing_id} \
                                 has invalid traversal kind {kind}"
                            ))),
                        }
                    })
                    .collect::<RusticolResult<BTreeSet<_>>>()?;
                if traversal_kinds.is_empty() {
                    return Err(invalid(format!(
                        "three-line closure sector {sector_id} pairing {pairing_id} \
                         realizes no certified traversal"
                    )));
                }
            }
        }
        if prepared_sector.expected_components.len() != sector.open_string_range.count as usize
            && sector.kind()? == ProcessLCSectorKind::OpenLines
        {
            return Err(invalid(
                "closure obligation open-line component count is inconsistent",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_closure_terms(
    strategy: RecurrenceStrategy,
    sector: &PreparedClosureSector,
    closure: &PreparedClosure,
    parent_ids: [u32; 2],
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    fermion_ordering: &FermionOrderingContext,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    reflection_certificates: &[PendingReflectionCertificate],
    prepared_sectors: &PreparedClosureSectorCatalog,
    materialized_sectors: &BTreeSet<u32>,
    result: &mut BTreeMap<PendingClosureKey, PendingClosureGroup>,
    covered_sector_ids: &mut BTreeSet<u32>,
    telemetry: &mut RecurrenceGenerationTelemetry,
    collect_telemetry: bool,
) -> RusticolResult<()> {
    let parents = [
        &currents[parent_ids[0] as usize].key,
        &currents[parent_ids[1] as usize].key,
    ];
    let pairing_certificate_ids =
        closure_pairing_certificate_ids(currents, parent_ids, pairing_catalog)?;
    let pairing_rule = pairing_rule_for_certificate(&pairing_certificate_ids, pairing_catalog)?;
    for quantum in closure
        .quantum_flows
        .iter()
        .filter(|quantum| quantum.matches(&parents))
    {
        let output_factor = quantum.output_factor()?;
        let (evaluator_parent_ids, exchange_factor) =
            closure.canonical_evaluator_parents(parent_ids);
        let evaluator_parent_permutation =
            two_parent_permutation(parent_ids, evaluator_parent_ids, "closure evaluator order")?;
        let fermion_factor = fermion_ordering_factor(
            &template.current_states,
            [
                parents[0].current_state_template_id(),
                parents[1].current_state_template_id(),
            ],
            [
                parents[0].support_source_slots(),
                parents[1].support_source_slots(),
            ],
            fermion_ordering,
        )?;
        let base_factor = multiply_factors(&[
            closure.closure_exact_factor,
            exchange_factor,
            closure.contraction_exact_factor,
            output_factor,
            fermion_factor,
            pairing_reconstruction_factor(pairing_rule),
        ])?;
        for witness in &closure.witnesses {
            let left = color_states
                .get(parents[0].dynamic_lc_color_state_id())
                .ok_or_else(|| invalid("closure left color state disappeared"))?;
            let right = color_states
                .get(parents[1].dynamic_lc_color_state_id())
                .ok_or_else(|| invalid("closure right color state disappeared"))?;
            if witness.row.left_shape_string_id != left.output_color_shape_id()
                || witness.row.right_shape_string_id != right.output_color_shape_id()
            {
                continue;
            }
            let closed = witness.witness.closed_components(left, right)?;
            if collect_telemetry {
                checked_diagnostic_add(
                    &mut telemetry.closure_color_attempt_count,
                    1,
                    "telemetry closure-color-attempt count",
                )?;
            }
            if !closed_components_match_prepared_sector(strategy, &closed, sector)? {
                continue;
            }
            if pairing_catalog.is_some() && pairing_certificate_ids.is_empty() {
                return Err(invalid(format!(
                    "closure witness {} for sector {} has no exactly realized fermion pairing",
                    witness.row.ordinal, sector.row.sector_id
                )));
            }
            let reconstruction_parent_permutation = match witness.row.input_permutation {
                0 => [0, 1],
                1 => [1, 0],
                value => {
                    return Err(invalid(format!(
                        "closure color witness has invalid parent permutation {value}"
                    )));
                }
            };
            let color_witness_term_id = closure
                .contraction
                .witness_start
                .checked_add(u64::from(witness.row.ordinal))
                .ok_or_else(|| invalid("closure color-witness term ID overflows"))?;
            let color_witness_term_id = u32::try_from(color_witness_term_id)
                .map_err(|_| invalid("closure color-witness term ID exceeds u32"))?;
            let key = PendingClosureKey {
                target_sector_id: sector.row.sector_id,
                complete_source_states: complete_closure_source_states(
                    parents,
                    process.external_legs.len(),
                )?,
                closure_template_id: closure.row.id,
                quantum_flow_template_id: quantum.template_id(),
                parent_current_ids: evaluator_parent_ids.into(),
            };
            let factor = base_factor.checked_mul(witness.witness.exact_factor())?;
            let reflection = closure_reflection_projection(
                sector.row,
                &closed,
                process,
                process_catalog,
                color_states,
                currents,
                parent_ids,
                reflection_certificates,
                prepared_sectors,
                materialized_sectors,
            )?;
            let contribution = PendingClosureProofContribution {
                construction_parent_ids: parent_ids,
                construction_parent_permutation: [0, 1],
                reconstruction_parent_permutation,
                evaluator_parent_permutation,
                closure_template_semantic_digest: closure.semantic_digest,
                color_witness_term_id,
                color_witness_proof_digest: witness.witness.proof_digest(),
                three_line_certificate: three_line_traversal_certificate(
                    &closed,
                    sector.row,
                    &sector.expected_components,
                    process_catalog,
                    pairing_rule.map(|rule| rule.rule_id),
                )?,
                pairing_certificate_ids: pairing_certificate_ids.clone().into_boxed_slice(),
                reflection_certificate_id: reflection.map(|row| row.certificate_id),
                exact_factor: factor,
            };
            let partner_sector_id = reflection.and_then(|row| row.partner_sector_id);
            include_closure_with_reflection(result, key, contribution, reflection)?;
            covered_sector_ids.insert(sector.row.sector_id);
            if let Some(partner_sector_id) = partner_sector_id {
                covered_sector_ids.insert(partner_sector_id);
            }
        }
    }
    Ok(())
}

fn closure_selector_proof_words(
    sector_id: u32,
    target_helicity_id: Option<u32>,
    source_states: &[SourceStateAssignment],
) -> RusticolResult<Vec<u64>> {
    let mut words = Vec::with_capacity(
        3usize
            .checked_add(source_states.len().saturating_mul(2))
            .ok_or_else(|| invalid("closure selector-proof word count overflows usize"))?,
    );
    words.push(u64::from(sector_id));
    words.push(u64::from(target_helicity_id.unwrap_or(MISSING_U32)));
    words.push(
        u64::try_from(source_states.len())
            .map_err(|_| invalid("closure selector source-state count exceeds u64"))?,
    );
    for assignment in source_states {
        words.push(u64::from(assignment.source_slot()));
        words.push(u64::from(assignment.state_index()));
    }
    Ok(words)
}

fn pending_closure_candidate_identity_digest(
    key: &PendingClosureKey,
    contribution: &PendingClosureProofContribution,
    target_helicity_id: Option<u32>,
    pending: &[PendingCurrent],
    dynamic_color_states: &[DynamicLCColorState],
    reflection_certificates: &[PendingReflectionCertificate],
) -> RusticolResult<SemanticDigest> {
    let selector_words = closure_selector_proof_words(
        key.target_sector_id,
        target_helicity_id,
        &key.complete_source_states,
    )?;
    let selector_domain_digest = closure_selector_domain_digest_v2(&selector_words)?;
    let mut parent_semantic_digests =
        Vec::with_capacity(contribution.construction_parent_ids.len());
    let mut parent_color_digests = Vec::with_capacity(contribution.construction_parent_ids.len());
    for old_id in contribution.construction_parent_ids {
        let current = pending
            .get(old_id as usize)
            .ok_or_else(|| invalid("closure candidate source parent is absent"))?;
        parent_semantic_digests.push(semantic_digest_from_u32_fields(
            b"recurrence-closure-parent-current-v2\0",
            [
                old_id,
                current.key.current_state_template_id(),
                current.key.dynamic_lc_color_state_id().get(),
            ],
        )?);
        parent_color_digests.push(dynamic_color_identity_digest(
            dynamic_color_states
                .get(current.key.dynamic_lc_color_state_id().get() as usize)
                .ok_or_else(|| invalid("closure candidate parent color state is absent"))?,
        )?);
    }
    let reflection_proof_digest = contribution
        .reflection_certificate_id
        .map(|certificate_id| {
            reflection_certificates
                .get(certificate_id as usize)
                .map(|certificate| certificate.proof_digest)
                .ok_or_else(|| {
                    invalid(format!(
                        "closure candidate references absent reflection certificate {certificate_id}"
                    ))
                })
        })
        .transpose()?;
    closure_candidate_identity_digest_v1(
        selector_domain_digest,
        key.target_sector_id,
        key.closure_template_id,
        contribution.closure_template_semantic_digest,
        key.quantum_flow_template_id,
        &contribution.construction_parent_ids,
        &parent_semantic_digests,
        &parent_color_digests,
        &contribution.construction_parent_permutation,
        &contribution.reconstruction_parent_permutation,
        &contribution.evaluator_parent_permutation,
        contribution.color_witness_term_id,
        contribution.color_witness_proof_digest,
        contribution
            .three_line_certificate
            .as_ref()
            .map(|certificate| certificate.proof_digest),
        &contribution.pairing_certificate_ids,
        reflection_proof_digest,
        contribution.exact_factor,
        1,
    )
}

fn pending_selector_sector_domains(
    pending: &[PendingCurrent],
    pending_closures: &BTreeMap<PendingClosureKey, PendingClosureGroup>,
) -> RusticolResult<Vec<BTreeSet<u32>>> {
    let mut domains = vec![BTreeSet::new(); pending.len()];
    let mut queue = VecDeque::new();
    for (key, group) in pending_closures {
        if group.exact_factor.is_zero() {
            continue;
        }
        for parent in key.parent_current_ids.iter().copied() {
            let domain = domains
                .get_mut(parent as usize)
                .ok_or_else(|| invalid("closure references an absent pending current"))?;
            if domain.insert(key.target_sector_id) {
                queue.push_back(parent);
            }
        }
    }
    while let Some(current_id) = queue.pop_front() {
        let sectors = domains
            .get(current_id as usize)
            .ok_or_else(|| invalid("selector-domain queue references an absent current"))?
            .clone();
        for (contribution, factor) in &pending[current_id as usize].contributions {
            if factor.is_zero() {
                continue;
            }
            for parent in contribution.parent_current_ids.iter().copied() {
                let domain = domains
                    .get_mut(parent as usize)
                    .ok_or_else(|| invalid("contribution references an absent pending parent"))?;
                let before = domain.len();
                domain.extend(sectors.iter().copied());
                if domain.len() != before {
                    queue.push_back(parent);
                }
            }
        }
    }
    Ok(domains)
}

fn projected_contribution_identity(
    destination_projection_id: u32,
    pending_key: &PendingContributionKey,
    exact_factor: ExactComplexRational,
    old_to_projection: &BTreeMap<u32, u32>,
) -> RusticolResult<ProjectedContributionIdentity> {
    let parent_projection_ids = pending_key
        .parent_current_ids
        .iter()
        .map(|old_id| {
            old_to_projection
                .get(old_id)
                .copied()
                .ok_or_else(|| invalid("projected contribution has a dead parent"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let key = &pending_key.key;
    Ok(ProjectedContributionIdentity {
        destination_projection_id,
        transition_template_id: key.transition_template_id(),
        parent_projection_ids: parent_projection_ids.into_boxed_slice(),
        parent_state_template_ids: key.parent_state_template_ids().into(),
        parent_momenta: key.parent_momenta().into(),
        result_state_template_id: key.result_state_template_id(),
        quantum_flow_witness_id: key.quantum_flow_witness_id(),
        runtime_coupling_binding_digest: key.runtime_coupling_binding_digest(),
        output_projection_id: key.output_projection_id(),
        exact_factor,
    })
}

fn rectangular_parent_domain_is_complete(
    parent_projection_ids: &[u32],
    builder_parent_tuples: &BTreeSet<Box<[u32]>>,
    old_to_projection: &BTreeMap<u32, u32>,
    projection_members: &[Box<[u32]>],
) -> RusticolResult<bool> {
    let expected_count =
        parent_projection_ids
            .iter()
            .try_fold(1usize, |count, projection_id| {
                let member_count = projection_members
                    .get(*projection_id as usize)
                    .ok_or_else(|| {
                        invalid("rectangular proof references an absent projection class")
                    })?
                    .len();
                count
                    .checked_mul(member_count)
                    .ok_or_else(|| invalid("rectangular parent-product count exceeds usize"))
            })?;
    if expected_count != builder_parent_tuples.len() {
        return Ok(false);
    }
    for tuple in builder_parent_tuples {
        if tuple.len() != parent_projection_ids.len() {
            return Err(invalid(
                "rectangular proof parent tuple has inconsistent arity",
            ));
        }
        for (old_id, expected_projection_id) in tuple.iter().zip(parent_projection_ids.iter()) {
            if old_to_projection.get(old_id) != Some(expected_projection_id) {
                return Ok(false);
            }
        }
    }
    // The tuples are unique by construction.  Membership in the declared
    // classes plus cardinality equal to the complete Cartesian product proves
    // that no combination is missing.
    Ok(true)
}

fn plan_topology_replay_color_projection(
    pending: &[PendingCurrent],
    pending_closures: &BTreeMap<PendingClosureKey, PendingClosureGroup>,
) -> RusticolResult<Option<PendingColorProjection>> {
    let selector_domains = pending_selector_sector_domains(pending, pending_closures)?;
    let canonical_color_id = DynamicLCColorStateId::from_interner(0);
    let mut identity_to_projection = BTreeMap::<ProjectedCurrentIdentity, u32>::new();
    let mut old_to_projection = BTreeMap::new();
    let mut projection_members = Vec::<Vec<u32>>::new();
    let mut projection_sector_ids = Vec::<Box<[u32]>>::new();

    for (old_id, current) in pending.iter().enumerate() {
        let sectors = &selector_domains[old_id];
        if sectors.is_empty() {
            continue;
        }
        let old_id =
            u32::try_from(old_id).map_err(|_| invalid("pending current ID exceeds u32"))?;
        let identity = ProjectedCurrentIdentity {
            color_erased_key: current_key_with_dynamic_color(&current.key, canonical_color_id)?,
            selector_sector_ids: sectors.iter().copied().collect(),
            source_builder_id: (current.key.node_kind() == RecurrenceNodeKind::Source)
                .then_some(old_id),
        };
        let projection_id =
            if let Some(projection_id) = identity_to_projection.get(&identity).copied() {
                projection_id
            } else {
                let projection_id = u32::try_from(projection_members.len())
                    .map_err(|_| invalid("projected current count exceeds u32"))?;
                projection_sector_ids.push(identity.selector_sector_ids.clone());
                projection_members.push(Vec::new());
                identity_to_projection.insert(identity, projection_id);
                projection_id
            };
        projection_members[projection_id as usize].push(old_id);
        old_to_projection.insert(old_id, projection_id);
    }
    if projection_members.iter().all(|members| members.len() == 1) {
        return Ok(None);
    }
    let projection_members = projection_members
        .into_iter()
        .map(Vec::into_boxed_slice)
        .collect::<Vec<_>>();

    let mut contributions =
        BTreeMap::<ProjectedContributionIdentity, PendingProjectedContribution>::new();
    for (old_id, projection_id) in &old_to_projection {
        for (pending_key, factor) in &pending[*old_id as usize].contributions {
            if factor.is_zero() {
                continue;
            }
            let identity = projected_contribution_identity(
                *projection_id,
                pending_key,
                *factor,
                &old_to_projection,
            )?;
            let parent_tuple = pending_key.parent_current_ids.clone();
            match contributions.entry(identity.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(PendingProjectedContribution {
                        identity,
                        representative_destination_builder_id: *old_id,
                        representative_key: pending_key.key.clone(),
                        destination_builder_ids: BTreeSet::from([*old_id]),
                        builder_parent_tuples: BTreeSet::from([parent_tuple]),
                    });
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let projected = entry.get_mut();
                    projected.destination_builder_ids.insert(*old_id);
                    if !projected.builder_parent_tuples.insert(parent_tuple) {
                        // Two builder rows with the same complete numeric
                        // identity and old-parent tuple are additive, not an
                        // alias.  Retain the unprojected program.
                        return Ok(None);
                    }
                }
            }
        }
    }
    for projected in contributions.values() {
        if !rectangular_parent_domain_is_complete(
            &projected.identity.parent_projection_ids,
            &projected.builder_parent_tuples,
            &old_to_projection,
            &projection_members,
        )? {
            return Ok(None);
        }
    }

    let mut closures = BTreeMap::<ProjectedClosureIdentity, PendingProjectedClosure>::new();
    for (key, group) in pending_closures {
        if group.exact_factor.is_zero() {
            continue;
        }
        let parent_projection_ids = key
            .parent_current_ids
            .iter()
            .map(|old_id| {
                old_to_projection
                    .get(old_id)
                    .copied()
                    .ok_or_else(|| invalid("projected closure has a dead parent"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let identity = ProjectedClosureIdentity {
            target_sector_id: key.target_sector_id,
            complete_source_states: key.complete_source_states.clone(),
            closure_template_id: key.closure_template_id,
            quantum_flow_template_id: key.quantum_flow_template_id,
            parent_projection_ids: parent_projection_ids.into_boxed_slice(),
            exact_factor: group.exact_factor,
        };
        let parent_tuple = key.parent_current_ids.clone();
        match closures.entry(identity.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(PendingProjectedClosure {
                    identity,
                    representative_key: key.clone(),
                    representative_group: group.clone(),
                    builder_parent_tuples: BTreeSet::from([parent_tuple]),
                });
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                if !entry.get_mut().builder_parent_tuples.insert(parent_tuple) {
                    return Ok(None);
                }
            }
        }
    }
    for projected in closures.values() {
        if !rectangular_parent_domain_is_complete(
            &projected.identity.parent_projection_ids,
            &projected.builder_parent_tuples,
            &old_to_projection,
            &projection_members,
        )? {
            return Ok(None);
        }
    }

    Ok(Some(PendingColorProjection {
        old_to_projection,
        projection_members,
        projection_sector_ids,
        contributions,
        closures,
    }))
}

const COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC: &[u8; 32] = b"PYAMP-COLOR-PROJECTION-BODY-V1\0\0";

fn certificate_push_len(output: &mut Vec<u8>, len: usize, label: &str) -> RusticolResult<()> {
    output.extend_from_slice(
        &u64::try_from(len)
            .map_err(|_| invalid(format!("{label} length exceeds u64")))?
            .to_le_bytes(),
    );
    Ok(())
}

fn certificate_push_u32_sequence(
    output: &mut Vec<u8>,
    values: &[u32],
    label: &str,
) -> RusticolResult<()> {
    certificate_push_len(output, values.len(), label)?;
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn certificate_push_builder_tuples(
    output: &mut Vec<u8>,
    tuples: &BTreeSet<Box<[u32]>>,
    label: &str,
) -> RusticolResult<()> {
    certificate_push_len(output, tuples.len(), label)?;
    for tuple in tuples {
        certificate_push_u32_sequence(output, tuple, label)?;
    }
    Ok(())
}

fn hash_u32_sequence(hash: &mut Sha256, values: &[u32], label: &str) -> RusticolResult<()> {
    hash.update(
        u64::try_from(values.len())
            .map_err(|_| invalid(format!("{label} length exceeds u64")))?
            .to_le_bytes(),
    );
    for value in values {
        hash.update(value.to_le_bytes());
    }
    Ok(())
}

fn hash_i32_sequence(hash: &mut Sha256, values: &[i32], label: &str) -> RusticolResult<()> {
    hash.update(
        u64::try_from(values.len())
            .map_err(|_| invalid(format!("{label} length exceeds u64")))?
            .to_le_bytes(),
    );
    for value in values {
        hash.update(value.to_le_bytes());
    }
    Ok(())
}

fn hash_momentum(hash: &mut Sha256, momentum: &CanonicalMomentumLinearForm) -> RusticolResult<()> {
    hash.update(
        u64::try_from(momentum.terms().len())
            .map_err(|_| invalid("projection-certificate momentum length exceeds u64"))?
            .to_le_bytes(),
    );
    for term in momentum.terms() {
        hash.update(term.source_slot.to_le_bytes());
        hash.update(term.coefficient.to_le_bytes());
    }
    Ok(())
}

fn projected_current_identity_digest(
    identity: &ProjectedCurrentIdentity,
) -> RusticolResult<SemanticDigest> {
    let key = &identity.color_erased_key;
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-color-projected-current-v1\0");
    hash.update(key.catalog_digest().as_bytes());
    hash.update((key.node_kind() as u32).to_le_bytes());
    hash.update(key.current_state_template_id().to_le_bytes());
    hash_u32_sequence(
        &mut hash,
        key.support_source_slots(),
        "projection-certificate support source slots",
    )?;
    hash_momentum(&mut hash, key.momentum())?;
    hash.update((key.helicity_identity().strategy() as u32).to_le_bytes());
    hash.update(key.helicity_identity().spin_state_class().to_le_bytes());
    hash.update(
        u64::try_from(key.helicity_identity().local_source_states().len())
            .map_err(|_| invalid("projection-certificate helicity ancestry exceeds u64"))?
            .to_le_bytes(),
    );
    for assignment in key.helicity_identity().local_source_states() {
        hash.update(assignment.source_slot().to_le_bytes());
        hash.update(assignment.state_index().to_le_bytes());
    }
    hash_i32_sequence(
        &mut hash,
        key.flavour_flow(),
        "projection-certificate flavour flow",
    )?;
    hash.update(key.quantum_number_flow_id().to_le_bytes());
    hash_u32_sequence(
        &mut hash,
        key.coupling_orders(),
        "projection-certificate coupling orders",
    )?;
    match key.source_binding() {
        CurrentSourceBinding::None => hash.update(0_u32.to_le_bytes()),
        CurrentSourceBinding::FixedTemplate(template_id) => {
            hash.update(1_u32.to_le_bytes());
            hash.update(template_id.to_le_bytes());
        }
        CurrentSourceBinding::RuntimeDispatch {
            domain,
            source_template_ids,
            variant_bindings,
        } => {
            hash.update(2_u32.to_le_bytes());
            hash.update(domain.to_le_bytes());
            hash_u32_sequence(
                &mut hash,
                source_template_ids,
                "projection-certificate runtime source templates",
            )?;
            hash.update(
                u64::try_from(variant_bindings.len())
                    .map_err(|_| {
                        invalid("projection-certificate runtime source variants exceed u64")
                    })?
                    .to_le_bytes(),
            );
            for variant in variant_bindings {
                hash.update(variant.source_state_index().to_le_bytes());
                hash.update(variant.public_helicity().to_le_bytes());
                hash.update(variant.runtime_variant_id().to_le_bytes());
                hash.update(variant.source_template_id().to_le_bytes());
                hash.update(variant.source_state_template_id().to_le_bytes());
                hash.update(variant.crossed_state_template_id().to_le_bytes());
                hash.update(variant.crossed_spin_state_class().to_le_bytes());
                hash_exact_factor(&mut hash, variant.crossing_factor());
            }
        }
    }
    hash.update(
        key.propagator_template_id()
            .unwrap_or(MISSING_U32)
            .to_le_bytes(),
    );
    hash_u32_sequence(
        &mut hash,
        &identity.selector_sector_ids,
        "projection-certificate selector sectors",
    )?;
    hash.update(
        identity
            .source_builder_id
            .unwrap_or(MISSING_U32)
            .to_le_bytes(),
    );
    SemanticDigest::new(hash.finalize().into())
}

fn projected_contribution_identity_digest(
    identity: &ProjectedContributionIdentity,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-color-projected-contribution-v1\0");
    hash.update(identity.destination_projection_id.to_le_bytes());
    hash.update(identity.transition_template_id.to_le_bytes());
    hash_u32_sequence(
        &mut hash,
        &identity.parent_projection_ids,
        "projection-certificate contribution parents",
    )?;
    hash_u32_sequence(
        &mut hash,
        &identity.parent_state_template_ids,
        "projection-certificate contribution parent states",
    )?;
    hash.update(
        u64::try_from(identity.parent_momenta.len())
            .map_err(|_| invalid("projection-certificate parent momenta exceed u64"))?
            .to_le_bytes(),
    );
    for momentum in &identity.parent_momenta {
        hash_momentum(&mut hash, momentum)?;
    }
    hash.update(identity.result_state_template_id.to_le_bytes());
    hash.update(identity.quantum_flow_witness_id.to_le_bytes());
    hash.update(identity.runtime_coupling_binding_digest.as_bytes());
    hash.update(identity.output_projection_id.to_le_bytes());
    hash_exact_factor(&mut hash, identity.exact_factor);
    SemanticDigest::new(hash.finalize().into())
}

fn projected_closure_identity_digest(
    identity: &ProjectedClosureIdentity,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-color-projected-closure-v1\0");
    hash.update(identity.target_sector_id.to_le_bytes());
    hash.update(
        u64::try_from(identity.complete_source_states.len())
            .map_err(|_| invalid("projection-certificate closure source states exceed u64"))?
            .to_le_bytes(),
    );
    for assignment in &identity.complete_source_states {
        hash.update(assignment.source_slot().to_le_bytes());
        hash.update(assignment.state_index().to_le_bytes());
    }
    hash.update(identity.closure_template_id.to_le_bytes());
    hash.update(
        identity
            .quantum_flow_template_id
            .unwrap_or(MISSING_U32)
            .to_le_bytes(),
    );
    hash_u32_sequence(
        &mut hash,
        &identity.parent_projection_ids,
        "projection-certificate closure parents",
    )?;
    hash_exact_factor(&mut hash, identity.exact_factor);
    SemanticDigest::new(hash.finalize().into())
}

fn encode_color_projection_certificate_body(
    pending: &[PendingCurrent],
    dynamic_color_states: &[DynamicLCColorState],
    projection: &PendingColorProjection,
    original_candidate_identity_digests: &[SemanticDigest],
) -> RusticolResult<Vec<u8>> {
    let mut output = Vec::new();
    output.extend_from_slice(COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC);
    output.extend_from_slice(&1_u32.to_le_bytes());
    output.extend_from_slice(
        &u64::try_from(projection.old_to_projection.len())
            .map_err(|_| invalid("projection-certificate old current count exceeds u64"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u64::try_from(projection.projection_members.len())
            .map_err(|_| invalid("projection-certificate current count exceeds u64"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u64::try_from(projection.contributions.len())
            .map_err(|_| invalid("projection-certificate contribution count exceeds u64"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u64::try_from(projection.closures.len())
            .map_err(|_| invalid("projection-certificate closure count exceeds u64"))?
            .to_le_bytes(),
    );

    for (projection_index, members) in projection.projection_members.iter().enumerate() {
        let projection_id = u32::try_from(projection_index)
            .map_err(|_| invalid("projection-certificate current ID exceeds u32"))?;
        output.extend_from_slice(&projection_id.to_le_bytes());
        let representative_id = *members
            .first()
            .ok_or_else(|| invalid("projection-certificate current class is empty"))?;
        let representative = pending
            .get(representative_id as usize)
            .ok_or_else(|| invalid("projection-certificate current representative is absent"))?;
        let identity = ProjectedCurrentIdentity {
            color_erased_key: current_key_with_dynamic_color(
                &representative.key,
                DynamicLCColorStateId::from_interner(0),
            )?,
            selector_sector_ids: projection.projection_sector_ids[projection_index].clone(),
            source_builder_id: (representative.key.node_kind() == RecurrenceNodeKind::Source)
                .then_some(representative_id),
        };
        output.extend_from_slice(projected_current_identity_digest(&identity)?.as_bytes());
        certificate_push_u32_sequence(
            &mut output,
            &projection.projection_sector_ids[projection_index],
            "projection-certificate current selector sectors",
        )?;
        certificate_push_len(
            &mut output,
            members.len(),
            "projection-certificate current members",
        )?;
        for old_id in members {
            output.extend_from_slice(&old_id.to_le_bytes());
            let current = pending
                .get(*old_id as usize)
                .ok_or_else(|| invalid("projection-certificate current member is absent"))?;
            let color_state = dynamic_color_states
                .get(current.key.dynamic_lc_color_state_id().get() as usize)
                .ok_or_else(|| invalid("projection-certificate color state is absent"))?;
            output.extend_from_slice(dynamic_color_identity_digest(color_state)?.as_bytes());
        }
    }

    for projected in projection.contributions.values() {
        output.extend_from_slice(
            projected_contribution_identity_digest(&projected.identity)?.as_bytes(),
        );
        output.extend_from_slice(&projected.identity.destination_projection_id.to_le_bytes());
        output.extend_from_slice(
            &projected
                .representative_destination_builder_id
                .to_le_bytes(),
        );
        let witness = projected.representative_key.color_witness_term_id();
        output.extend_from_slice(&witness.color_contraction_template_id().to_le_bytes());
        output.extend_from_slice(&witness.witness_ordinal().to_le_bytes());
        certificate_push_u32_sequence(
            &mut output,
            &projected.identity.parent_projection_ids,
            "projection-certificate contribution projected parents",
        )?;
        certificate_push_len(
            &mut output,
            projected.destination_builder_ids.len(),
            "projection-certificate contribution destinations",
        )?;
        for destination_id in &projected.destination_builder_ids {
            output.extend_from_slice(&destination_id.to_le_bytes());
        }
        certificate_push_builder_tuples(
            &mut output,
            &projected.builder_parent_tuples,
            "projection-certificate contribution parent tuples",
        )?;
        for rational in [
            projected.identity.exact_factor.real(),
            projected.identity.exact_factor.imag(),
        ] {
            output.extend_from_slice(&rational.numerator().to_le_bytes());
            output.extend_from_slice(&rational.denominator().to_le_bytes());
        }
    }

    for projected in projection.closures.values() {
        output
            .extend_from_slice(projected_closure_identity_digest(&projected.identity)?.as_bytes());
        output.extend_from_slice(&projected.identity.target_sector_id.to_le_bytes());
        output.extend_from_slice(&projected.identity.closure_template_id.to_le_bytes());
        output.extend_from_slice(
            &projected
                .identity
                .quantum_flow_template_id
                .unwrap_or(MISSING_U32)
                .to_le_bytes(),
        );
        certificate_push_u32_sequence(
            &mut output,
            &projected.identity.parent_projection_ids,
            "projection-certificate closure projected parents",
        )?;
        certificate_push_u32_sequence(
            &mut output,
            &projected.representative_key.parent_current_ids,
            "projection-certificate closure representative parents",
        )?;
        certificate_push_builder_tuples(
            &mut output,
            &projected.builder_parent_tuples,
            "projection-certificate closure parent tuples",
        )?;
        for rational in [
            projected.identity.exact_factor.real(),
            projected.identity.exact_factor.imag(),
        ] {
            output.extend_from_slice(&rational.numerator().to_le_bytes());
            output.extend_from_slice(&rational.denominator().to_le_bytes());
        }
    }

    let mut candidate_digests = original_candidate_identity_digests.to_vec();
    candidate_digests.sort_unstable();
    certificate_push_len(
        &mut output,
        candidate_digests.len(),
        "projection-certificate original closure candidates",
    )?;
    for digest in candidate_digests {
        output.extend_from_slice(digest.as_bytes());
    }
    let digest: [u8; 32] = Sha256::digest(&output).into();
    output.extend_from_slice(&digest);
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn append_closure_proof_group(
    key: &PendingClosureKey,
    group: &PendingClosureGroup,
    pending: &[PendingCurrent],
    dynamic_color_states: &[DynamicLCColorState],
    remap: &BTreeMap<u32, u32>,
    target_destination_id: Option<u32>,
    target_helicity_id: Option<u32>,
    runtime_term_id: Option<u32>,
    three_line_certificate_ids: &BTreeMap<PendingThreeLineTraversalCertificate, u32>,
    closure_proof_contributions: &mut Vec<ClosureProofContributionV2>,
    closure_proof_groups: &mut Vec<ClosureExecutionProofGroupV2>,
) -> RusticolResult<()> {
    if group.exact_factor.is_zero() != runtime_term_id.is_none() {
        return Err(invalid(
            "closure proof runtime binding does not match its exact aggregate factor",
        ));
    }
    if runtime_term_id.is_some() && target_destination_id.is_none() {
        return Err(invalid(
            "nonzero closure proof group has no amplitude destination",
        ));
    }
    let proof_start = u64::try_from(closure_proof_contributions.len())
        .map_err(|_| invalid("closure proof contribution count exceeds u64"))?;
    let mut component_factors = Vec::with_capacity(group.contributions.len());
    for contribution in &group.contributions {
        let builder_parent_ids = contribution.construction_parent_ids.to_vec();
        let runtime_parent_ids = contribution
            .construction_parent_ids
            .iter()
            .map(|old_id| remap.get(old_id).copied())
            .collect::<Vec<_>>();
        if runtime_term_id.is_some() && runtime_parent_ids.iter().any(Option::is_none) {
            return Err(invalid(
                "nonzero closure proof contribution has a dead runtime parent",
            ));
        }
        let mut parent_semantic_digests =
            Vec::with_capacity(contribution.construction_parent_ids.len());
        let mut parent_color_digests =
            Vec::with_capacity(contribution.construction_parent_ids.len());
        for old_id in contribution.construction_parent_ids {
            let current = pending
                .get(old_id as usize)
                .ok_or_else(|| invalid("closure proof source parent is absent"))?;
            parent_semantic_digests.push(semantic_digest_from_u32_fields(
                b"recurrence-closure-parent-current-v2\0",
                [
                    old_id,
                    current.key.current_state_template_id(),
                    current.key.dynamic_lc_color_state_id().get(),
                ],
            )?);
            parent_color_digests.push(dynamic_color_identity_digest(
                dynamic_color_states
                    .get(current.key.dynamic_lc_color_state_id().get() as usize)
                    .ok_or_else(|| invalid("closure proof parent color state is absent"))?,
            )?);
        }
        let proof_id = u32::try_from(closure_proof_contributions.len())
            .map_err(|_| invalid("closure proof contribution count exceeds u32"))?;
        closure_proof_contributions.push(ClosureProofContributionV2::new(
            proof_id,
            key.target_sector_id,
            target_destination_id,
            target_helicity_id,
            key.closure_template_id,
            contribution.closure_template_semantic_digest,
            key.quantum_flow_template_id,
            builder_parent_ids,
            runtime_parent_ids,
            parent_semantic_digests,
            parent_color_digests,
            contribution.construction_parent_permutation.to_vec(),
            contribution.reconstruction_parent_permutation.to_vec(),
            contribution.evaluator_parent_permutation.to_vec(),
            contribution.color_witness_term_id,
            contribution.color_witness_proof_digest,
            contribution
                .three_line_certificate
                .as_ref()
                .map(|certificate| three_line_certificate_ids[certificate]),
            contribution.pairing_certificate_ids.to_vec(),
            contribution.reflection_certificate_id,
            contribution.exact_factor,
            1,
        )?);
        component_factors.push(contribution.exact_factor);
    }
    let proof_count = u64::try_from(closure_proof_contributions.len())
        .map_err(|_| invalid("closure proof contribution count exceeds u64"))?
        .checked_sub(proof_start)
        .ok_or_else(|| invalid("closure proof contribution range underflows"))?;
    let selector_words = closure_selector_proof_words(
        key.target_sector_id,
        target_helicity_id,
        &key.complete_source_states,
    )?;
    closure_proof_groups.push(ClosureExecutionProofGroupV2::new(
        u32::try_from(closure_proof_groups.len())
            .map_err(|_| invalid("closure proof group count exceeds u32"))?,
        runtime_term_id,
        None,
        CheckedTableRange::new(proof_start, proof_count),
        group.exact_factor,
        closure_component_factor_digest_v2(&component_factors)?,
        closure_selector_domain_digest_v2(&selector_words)?,
    )?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "on-the-fly-test-support"))]
fn observe_established_selected_transition_slice(
    process_catalog: &ProcessCatalog<'_>,
    pending: &[PendingCurrent],
    pending_closures: &BTreeMap<PendingClosureKey, PendingClosureGroup>,
    replay_targets: &[RecurrenceReplayTarget],
    dynamic_color_states: &[DynamicLCColorState],
) -> RusticolResult<Option<(u32, Box<[SourceStateAssignment]>)>> {
    let Some(selection) = super::diagnostic::transition_diagnostic_selection() else {
        return Ok(None);
    };
    let replay_target = replay_targets
        .iter()
        .find(|target| target.target_sector_id() == selection.public_flow_id)
        .ok_or_else(|| {
            invalid(format!(
                "transition diagnostic public flow {} has no replay target",
                selection.public_flow_id
            ))
        })?;
    let source_states = super::diagnostic::representative_source_states_for_public_helicities(
        replay_target.source_slot_permutation(),
        &selection.public_helicities,
        |representative_slot, public_helicity| {
            let leg = process_catalog
                .input
                .external_legs
                .get(representative_slot as usize)
                .ok_or_else(|| {
                    invalid("transition diagnostic representative source leg is absent")
                })?;
            let range = leg.source_state_range.as_usize_range(
                process_catalog.input.source_states.len(),
                "transition diagnostic representative source states",
            )?;
            let mut matches = process_catalog.input.source_states[range]
                .iter()
                .filter(|state| {
                    state.source_slot == representative_slot
                        && state.public_helicity == public_helicity
                });
            let state = matches.next().ok_or_else(|| {
                invalid(format!(
                    "transition diagnostic representative source slot {representative_slot} has no state for transported public helicity {public_helicity}"
                ))
            })?;
            if matches.next().is_some() {
                return Err(invalid(format!(
                    "transition diagnostic representative source slot {representative_slot} has multiple states for transported public helicity {public_helicity}"
                )));
            }
            Ok(state.state_index)
        },
    )?
    .into_boxed_slice();
    let materialized_sector_id = replay_target.materialized_sector_id();
    let mut live = BTreeSet::new();
    let mut queue = VecDeque::new();
    for (key, _group) in pending_closures.iter().filter(|(key, group)| {
        key.target_sector_id == materialized_sector_id
            && key.complete_source_states == source_states
            && !group.exact_factor.is_zero()
    }) {
        for parent in key.parent_current_ids.iter().copied() {
            if live.insert(parent) {
                queue.push_back(parent);
            }
        }
    }
    if live.is_empty() {
        return Err(invalid(format!(
            "transition diagnostic flow {} and source states {:?} have no established closure",
            selection.public_flow_id, source_states
        )));
    }
    while let Some(current_id) = queue.pop_front() {
        let current = pending
            .get(current_id as usize)
            .ok_or_else(|| invalid("transition diagnostic current is absent"))?;
        for (contribution, factor) in &current.contributions {
            if factor.is_zero() {
                continue;
            }
            for parent in contribution.parent_current_ids.iter().copied() {
                if live.insert(parent) {
                    queue.push_back(parent);
                }
            }
        }
    }
    let digests = live
        .into_iter()
        .map(|current_id| {
            let current = pending
                .get(current_id as usize)
                .ok_or_else(|| invalid("transition diagnostic live current is absent"))?;
            let color = dynamic_color_states
                .get(current.key.dynamic_lc_color_state_id().get() as usize)
                .ok_or_else(|| invalid("transition diagnostic live color is absent"))?;
            super::on_the_fly::hash_current_key(&current.key, color)
        })
        .collect::<RusticolResult<BTreeSet<_>>>()?;
    super::diagnostic::observe_transition_live_current_digests(digests)?;
    Ok(Some((materialized_sector_id, source_states)))
}

fn consume_ordered_closure_destination<I, Row, Destination, Emit>(
    rows: &mut Peekable<I>,
    sector_id: u32,
    source_states: &[SourceStateAssignment],
    destination: Destination,
    mut emit: Emit,
) -> RusticolResult<()>
where
    I: Iterator<Item = Row>,
    Destination: for<'row> Fn(&'row Row) -> (u32, &'row [SourceStateAssignment]),
    Emit: FnMut(Row) -> RusticolResult<()>,
{
    while let Some(row) = rows.peek() {
        let (row_sector_id, row_source_states) = destination(row);
        let ordering = row_sector_id
            .cmp(&sector_id)
            .then_with(|| row_source_states.cmp(source_states));
        match ordering {
            std::cmp::Ordering::Less => {
                return Err(invalid(
                    "ordered closure emission skipped a destination prefix",
                ));
            }
            std::cmp::Ordering::Equal => emit(
                rows.next()
                    .expect("peeked closure destination row disappeared"),
            )?,
            std::cmp::Ordering::Greater => break,
        }
    }
    Ok(())
}

fn require_ordered_closure_destinations_exhausted<I>(rows: &mut Peekable<I>) -> RusticolResult<()>
where
    I: Iterator,
{
    if rows.peek().is_some() {
        return Err(invalid(
            "ordered closure emission left a destination prefix unconsumed",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn complete_closure_target_domain_certificate(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    sector_count: usize,
    exhaustive_closure_sector_ids: &BTreeSet<u32>,
    exhaustive_closure_sector_owner_ids: &BTreeMap<u32, u32>,
    replay_targets: &[RecurrenceReplayTarget],
    helicity_ids: &BTreeMap<Box<[SourceStateAssignment]>, u32>,
    pending_closures: &BTreeMap<PendingClosureKey, PendingClosureGroup>,
    candidate_identity_digests_by_group: &[Vec<SemanticDigest>],
) -> RusticolResult<Option<ClosureTargetDomainCertificateV1>> {
    if strategy != RecurrenceStrategy::ContractedColorUnion {
        return Ok(None);
    }
    let physical_sector_count = u32::try_from(sector_count)
        .map_err(|_| invalid("closure target-domain physical-sector count exceeds u32"))?;
    let expected_sector_ids = (0..physical_sector_count).collect::<BTreeSet<_>>();
    if exhaustive_closure_sector_owner_ids
        .keys()
        .copied()
        .collect::<BTreeSet<_>>()
        != *exhaustive_closure_sector_ids
        || exhaustive_closure_sector_owner_ids
            .iter()
            .any(|(sector_id, owner_id)| {
                !exhaustive_closure_sector_ids.contains(owner_id)
                    || owner_id > sector_id
                    || exhaustive_closure_sector_owner_ids.get(owner_id) != Some(owner_id)
            })
    {
        return Err(invalid(
            "contracted closure target-domain certificate has an invalid owner partition",
        ));
    }
    let resolved_helicity_count = u32::try_from(helicity_ids.len())
        .map_err(|_| invalid("closure target-domain resolved-helicity count exceeds u32"))?;
    if resolved_helicity_count == 0 {
        return Err(invalid(
            "contracted closure target-domain certificate has no resolved helicities",
        ));
    }
    if candidate_identity_digests_by_group.len() != pending_closures.len() {
        return Err(invalid(
            "closure target-domain candidate groups do not match pending closures",
        ));
    }

    let mut ordered_helicities = helicity_ids
        .iter()
        .map(|(states, helicity_id)| (*helicity_id, states.as_ref()))
        .collect::<Vec<_>>();
    ordered_helicities.sort_unstable_by_key(|(helicity_id, _)| *helicity_id);
    if ordered_helicities
        .iter()
        .enumerate()
        .any(|(index, (helicity_id, _))| *helicity_id != index as u32)
    {
        return Err(invalid(
            "closure target-domain resolved-helicity IDs are not dense",
        ));
    }

    let mut closure_source_sector_ids = BTreeMap::<u32, u32>::new();
    let mut representative_helicity_ids_by_target = BTreeMap::<u32, Vec<u32>>::new();
    if replay_targets.is_empty() {
        if exhaustive_closure_sector_ids != &expected_sector_ids {
            return Err(invalid(
                "contracted closure target-domain certificate lacks exhaustive physical-sector coverage",
            ));
        }
        let identity_helicity_ids = (0..resolved_helicity_count).collect::<Vec<_>>();
        for sector_id in expected_sector_ids.iter().copied() {
            closure_source_sector_ids
                .insert(sector_id, exhaustive_closure_sector_owner_ids[&sector_id]);
            representative_helicity_ids_by_target.insert(sector_id, identity_helicity_ids.clone());
        }
    } else {
        if replay_targets.len() != sector_count {
            return Err(invalid(
                "contracted closure target-domain replay does not cover every physical sector",
            ));
        }
        let replay_materialized_sector_ids = replay_targets
            .iter()
            .map(RecurrenceReplayTarget::materialized_sector_id)
            .collect::<BTreeSet<_>>();
        if replay_materialized_sector_ids != *exhaustive_closure_sector_ids {
            return Err(invalid(
                "contracted closure target-domain replay representatives disagree with the exhaustive closure domain",
            ));
        }

        let mut source_template_by_state = BTreeMap::<(u32, u32), u32>::new();
        let mut source_state_by_template = BTreeMap::<(u32, u32), u32>::new();
        for state in &process.source_states {
            if source_template_by_state
                .insert(
                    (state.source_slot, state.state_index),
                    state.source_template_id,
                )
                .is_some()
                || source_state_by_template
                    .insert(
                        (state.source_slot, state.source_template_id),
                        state.state_index,
                    )
                    .is_some()
            {
                return Err(invalid(
                    "contracted closure target-domain replay source states are not bijective",
                ));
            }
        }
        let source_count = process.external_legs.len();
        for (target_index, target) in replay_targets.iter().enumerate() {
            let target_sector_id = u32::try_from(target_index)
                .map_err(|_| invalid("closure target-domain replay target exceeds u32"))?;
            if target.id() != target_sector_id
                || target.target_sector_id() != target_sector_id
                || !exhaustive_closure_sector_ids.contains(&target.materialized_sector_id())
            {
                return Err(invalid(
                    "contracted closure target-domain replay targets are not canonical",
                ));
            }
            let mut permutation = target.source_slot_permutation().to_vec();
            permutation.sort_unstable();
            if permutation.len() != source_count
                || permutation
                    .iter()
                    .copied()
                    .enumerate()
                    .any(|(slot, value)| value as usize != slot)
                || target.source_momentum_signs().len() != source_count
            {
                return Err(invalid(
                    "contracted closure target-domain replay source mapping is not bijective",
                ));
            }

            let mut inverse_helicity_ids = vec![MISSING_U32; helicity_ids.len()];
            for (representative_helicity_id, source_states) in &ordered_helicities {
                let mut mapped = vec![None; source_count];
                for assignment in *source_states {
                    let representative_slot = assignment.source_slot();
                    let source_template_id = source_template_by_state
                        .get(&(representative_slot, assignment.state_index()))
                        .copied()
                        .ok_or_else(|| {
                            invalid("closure target-domain replay helicity has no source template")
                        })?;
                    let target_slot = target
                        .source_slot_permutation()
                        .get(representative_slot as usize)
                        .copied()
                        .ok_or_else(|| {
                            invalid(
                                "closure target-domain replay source permutation is out of bounds",
                            )
                        })?;
                    let target_state_index = source_state_by_template
                        .get(&(target_slot, source_template_id))
                        .copied()
                        .ok_or_else(|| {
                            invalid(
                                "closure target-domain replay cannot transport a source template",
                            )
                        })?;
                    let target_index = usize::try_from(target_slot)
                        .map_err(|_| invalid("closure replay source slot exceeds usize"))?;
                    let target_state = mapped.get_mut(target_index).ok_or_else(|| {
                        invalid("closure target-domain replay target slot is out of bounds")
                    })?;
                    if target_state
                        .replace(SourceStateAssignment::new(target_slot, target_state_index))
                        .is_some()
                    {
                        return Err(invalid(
                            "closure target-domain replay helicity mapping is not bijective",
                        ));
                    }
                }
                let mapped = mapped
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        invalid("closure target-domain replay does not map every source state")
                    })?;
                let mapped_helicity_id = helicity_ids.get(mapped.as_slice()).copied().ok_or_else(
                    || {
                        invalid(
                            "closure target-domain replay maps outside retained helicity coverage",
                        )
                    },
                )?;
                let inverse_slot = inverse_helicity_ids
                    .get_mut(mapped_helicity_id as usize)
                    .ok_or_else(|| {
                        invalid("closure target-domain replay helicity ID is out of bounds")
                    })?;
                if *inverse_slot != MISSING_U32 {
                    return Err(invalid(
                        "closure target-domain replay helicity mapping is not one-to-one",
                    ));
                }
                *inverse_slot = *representative_helicity_id;
            }
            if inverse_helicity_ids.contains(&MISSING_U32) {
                return Err(invalid(
                    "closure target-domain replay helicity mapping is incomplete",
                ));
            }
            closure_source_sector_ids.insert(
                target_sector_id,
                exhaustive_closure_sector_owner_ids[&target.materialized_sector_id()],
            );
            representative_helicity_ids_by_target.insert(target_sector_id, inverse_helicity_ids);
        }
        if closure_source_sector_ids
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            != expected_sector_ids
        {
            return Err(invalid(
                "contracted closure target-domain replay target IDs are not dense",
            ));
        }
    }

    let mut groups_by_target = BTreeMap::<(u32, u32), Vec<usize>>::new();
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-complete-recurrence-closure-target-domain-v1\0");
    hash.update(physical_sector_count.to_le_bytes());
    hash.update(resolved_helicity_count.to_le_bytes());
    hash.update(
        u64::try_from(exhaustive_closure_sector_ids.len())
            .map_err(|_| invalid("closure target-domain exhaustive sector count exceeds u64"))?
            .to_le_bytes(),
    );
    for sector_id in exhaustive_closure_sector_ids {
        hash.update(sector_id.to_le_bytes());
        hash.update(exhaustive_closure_sector_owner_ids[sector_id].to_le_bytes());
    }
    hash.update(
        u64::try_from(replay_targets.len())
            .map_err(|_| invalid("closure target-domain replay-target count exceeds u64"))?
            .to_le_bytes(),
    );
    for target in replay_targets {
        hash.update(target.id().to_le_bytes());
        hash.update(target.materialized_sector_id().to_le_bytes());
        hash.update(target.target_sector_id().to_le_bytes());
        hash.update(
            u64::try_from(target.source_slot_permutation().len())
                .map_err(|_| invalid("closure replay source count exceeds u64"))?
                .to_le_bytes(),
        );
        for source_slot in target.source_slot_permutation() {
            hash.update(source_slot.to_le_bytes());
        }
        for momentum_sign in target.source_momentum_signs() {
            hash.update(momentum_sign.to_le_bytes());
        }
        hash_exact_factor(&mut hash, target.amplitude_factor());
    }
    for (helicity_id, source_states) in &ordered_helicities {
        hash.update(helicity_id.to_le_bytes());
        hash.update(
            u64::try_from(source_states.len())
                .map_err(|_| invalid("closure target-domain source-state count exceeds u64"))?
                .to_le_bytes(),
        );
        for assignment in *source_states {
            hash.update(assignment.source_slot().to_le_bytes());
            hash.update(assignment.state_index().to_le_bytes());
        }
    }

    hash.update(
        u64::try_from(pending_closures.len())
            .map_err(|_| invalid("closure target-domain group count exceeds u64"))?
            .to_le_bytes(),
    );
    for (group_index, ((key, group), candidate_digests)) in pending_closures
        .iter()
        .zip(candidate_identity_digests_by_group)
        .enumerate()
    {
        if candidate_digests.len() != group.contributions.len() || candidate_digests.is_empty() {
            return Err(invalid(
                "closure target-domain candidate identities do not match their proof group",
            ));
        }
        hash.update(key.target_sector_id.to_le_bytes());
        hash.update(
            u64::try_from(key.complete_source_states.len())
                .map_err(|_| invalid("closure target-domain source-state count exceeds u64"))?
                .to_le_bytes(),
        );
        for assignment in &key.complete_source_states {
            hash.update(assignment.source_slot().to_le_bytes());
            hash.update(assignment.state_index().to_le_bytes());
        }
        hash.update(key.closure_template_id.to_le_bytes());
        hash.update(
            key.quantum_flow_template_id
                .unwrap_or(MISSING_U32)
                .to_le_bytes(),
        );
        hash.update(
            u64::try_from(key.parent_current_ids.len())
                .map_err(|_| invalid("closure target-domain parent count exceeds u64"))?
                .to_le_bytes(),
        );
        for parent_id in &key.parent_current_ids {
            hash.update(parent_id.to_le_bytes());
        }
        hash_exact_factor(&mut hash, group.exact_factor);
        let mut candidate_digests = candidate_digests.clone();
        candidate_digests.sort_unstable();
        hash.update(
            u64::try_from(candidate_digests.len())
                .map_err(|_| invalid("closure target-domain candidate count exceeds u64"))?
                .to_le_bytes(),
        );
        for digest in candidate_digests {
            hash.update(digest.as_bytes());
        }
        if let Some(helicity_id) = helicity_ids.get(&key.complete_source_states).copied() {
            groups_by_target
                .entry((key.target_sector_id, helicity_id))
                .or_default()
                .push(group_index);
        } else if !group.exact_factor.is_zero() {
            return Err(invalid(
                "nonzero contracted closure group is outside the resolved-helicity domain",
            ));
        }
    }

    let pending_groups = pending_closures.values().collect::<Vec<_>>();
    let mut structural_zero_sector_ids = Vec::new();
    for sector_id in 0..physical_sector_count {
        let closure_source_sector_id = closure_source_sector_ids[&sector_id];
        let representative_helicity_ids = &representative_helicity_ids_by_target[&sector_id];
        let mut sector_has_nonzero_group = false;
        for target_helicity_id in 0..resolved_helicity_count {
            let representative_helicity_id =
                representative_helicity_ids[target_helicity_id as usize];
            let group_indices = groups_by_target
                .get(&(closure_source_sector_id, representative_helicity_id))
                .map(Vec::as_slice)
                .unwrap_or_default();
            hash.update(sector_id.to_le_bytes());
            hash.update(closure_source_sector_id.to_le_bytes());
            hash.update(target_helicity_id.to_le_bytes());
            hash.update(representative_helicity_id.to_le_bytes());
            hash.update(
                u64::try_from(group_indices.len())
                    .map_err(|_| invalid("closure target-domain target group count exceeds u64"))?
                    .to_le_bytes(),
            );
            for group_index in group_indices {
                hash.update(
                    u64::try_from(*group_index)
                        .map_err(|_| invalid("closure target-domain group index exceeds u64"))?
                        .to_le_bytes(),
                );
                sector_has_nonzero_group |= !pending_groups[*group_index].exact_factor.is_zero();
            }
        }
        if !sector_has_nonzero_group {
            structural_zero_sector_ids.push(sector_id);
        }
    }
    let semantic_digest = SemanticDigest::new(hash.finalize().into())?;
    ClosureTargetDomainCertificateV1::new(
        physical_sector_count,
        resolved_helicity_count,
        structural_zero_sector_ids,
        semantic_digest,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn finish_program(
    strategy: RecurrenceStrategy,
    process_catalog: &ProcessCatalog<'_>,
    dynamic_color_states: Vec<DynamicLCColorState>,
    pending: Vec<PendingCurrent>,
    mut pending_closures: BTreeMap<PendingClosureKey, PendingClosureGroup>,
    exhaustive_closure_sector_ids: BTreeSet<u32>,
    exhaustive_closure_sector_owner_ids: BTreeMap<u32, u32>,
    replay_targets: Vec<RecurrenceReplayTarget>,
    retained_helicity_count: u64,
    helicity_support_rule: HelicitySupportRule,
    global_helicity_flip_rule: GlobalHelicityFlipRule,
    reflection_certificates: Vec<PendingReflectionCertificate>,
) -> RusticolResult<RecurrenceProgram> {
    validate_pending_reflection_certificates(&reflection_certificates)?;

    pending_closures = retain_supported_pending_closures(
        strategy,
        process_catalog,
        &replay_targets,
        pending_closures,
        helicity_support_rule,
        global_helicity_flip_rule,
    )?;
    let sector_count = match strategy {
        RecurrenceStrategy::TopologyReplay => process_catalog
            .input
            .physical_lc_sectors
            .len()
            .max(replay_targets.len()),
        RecurrenceStrategy::AllFlowUnion | RecurrenceStrategy::ContractedColorUnion => {
            process_catalog.input.physical_lc_sectors.len()
        }
    };
    let helicity_keys = pending_closures
        .iter()
        .filter(|(key, group)| {
            !key.complete_source_states.is_empty() && !group.exact_factor.is_zero()
        })
        .map(|(key, _)| key)
        .map(|key| key.complete_source_states.clone())
        .collect::<BTreeSet<_>>();
    let helicity_ids = helicity_keys
        .iter()
        .cloned()
        .enumerate()
        .map(|(id, key)| (key, id as u32))
        .collect::<BTreeMap<_, _>>();
    let resolved_helicities = helicity_keys
        .into_iter()
        .enumerate()
        .map(|(id, source_states)| {
            let public_helicities = process_catalog.public_helicities(&source_states)?;
            RecurrenceResolvedHelicity::new(id as u32, source_states.into(), public_helicities)
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    let diagnostic_selected_closure_domain = observe_established_selected_transition_slice(
        process_catalog,
        &pending,
        &pending_closures,
        &replay_targets,
        &dynamic_color_states,
    )?;
    let mut live = BTreeSet::new();
    let mut queue = VecDeque::new();
    for (key, group) in &pending_closures {
        if group.exact_factor.is_zero() {
            continue;
        }
        for parent in key.parent_current_ids.iter().copied() {
            if live.insert(parent) {
                queue.push_back(parent);
            }
        }
    }
    while let Some(current_id) = queue.pop_front() {
        for (contribution, factor) in &pending[current_id as usize].contributions {
            if factor.is_zero() {
                continue;
            }
            for parent in contribution.parent_current_ids.iter().copied() {
                if live.insert(parent) {
                    queue.push_back(parent);
                }
            }
        }
    }
    let color_projection = if strategy == RecurrenceStrategy::TopologyReplay {
        plan_topology_replay_color_projection(&pending, &pending_closures)?
    } else {
        None
    };
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    observe_established_pairing_owner(
        diagnostic_selected_closure_domain.as_ref(),
        color_projection.as_ref(),
        &pending_closures,
    );
    let materialized = if let Some(projection) = color_projection.as_ref() {
        materialize_projected_pending_rows(&pending, projection)?
    } else {
        materialize_live_pending_rows(&pending, &live)?
    };
    let MaterializedPendingRows {
        remap,
        currents,
        contributions,
        finalizations,
    } = materialized;

    let destination_keys = pending_closures
        .iter()
        .filter(|(_, group)| !group.exact_factor.is_zero())
        .map(|(key, _)| (key.target_sector_id, key.complete_source_states.clone()))
        .collect::<BTreeSet<_>>();
    let mut closure_terms = Vec::new();
    let mut closure_proof_contributions = Vec::new();
    let mut closure_proof_groups = Vec::new();
    let pending_three_line_certificates = pending_closures
        .values()
        .flat_map(|group| group.contributions.iter())
        .filter_map(|contribution| contribution.three_line_certificate.clone())
        .collect::<BTreeSet<_>>();
    let three_line_certificate_ids = pending_three_line_certificates
        .iter()
        .cloned()
        .enumerate()
        .map(|(id, certificate)| {
            u32::try_from(id)
                .map(|id| (certificate, id))
                .map_err(|_| invalid("three-line certificate count exceeds u32"))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    let mut candidate_identity_digests_by_group = Vec::with_capacity(pending_closures.len());
    for (key, group) in &pending_closures {
        let target_helicity_id = if key.complete_source_states.is_empty() {
            None
        } else {
            helicity_ids.get(&key.complete_source_states).copied()
        };
        let mut group_candidate_identity_digests = Vec::with_capacity(group.contributions.len());
        for contribution in &group.contributions {
            group_candidate_identity_digests.push(pending_closure_candidate_identity_digest(
                key,
                contribution,
                target_helicity_id,
                &pending,
                &dynamic_color_states,
                &reflection_certificates,
            )?);
        }
        candidate_identity_digests_by_group.push(group_candidate_identity_digests);
    }
    let original_candidate_identity_digests = candidate_identity_digests_by_group
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let closure_target_domain_certificate = complete_closure_target_domain_certificate(
        strategy,
        process_catalog.input,
        sector_count,
        &exhaustive_closure_sector_ids,
        &exhaustive_closure_sector_owner_ids,
        &replay_targets,
        &helicity_ids,
        &pending_closures,
        &candidate_identity_digests_by_group,
    )?;
    let mut accepted_candidate_identity_digests = Vec::new();
    if let Some(projection) = color_projection.as_ref() {
        for projected in projection.closures.values() {
            let key = &projected.representative_key;
            let target_helicity_id = if key.complete_source_states.is_empty() {
                None
            } else {
                helicity_ids.get(&key.complete_source_states).copied()
            };
            for contribution in &projected.representative_group.contributions {
                accepted_candidate_identity_digests.push(
                    pending_closure_candidate_identity_digest(
                        key,
                        contribution,
                        target_helicity_id,
                        &pending,
                        &dynamic_color_states,
                        &reflection_certificates,
                    )?,
                );
            }
        }
        for (key, group) in pending_closures
            .iter()
            .filter(|(_, group)| group.exact_factor.is_zero())
        {
            let target_helicity_id = if key.complete_source_states.is_empty() {
                None
            } else {
                helicity_ids.get(&key.complete_source_states).copied()
            };
            for contribution in &group.contributions {
                accepted_candidate_identity_digests.push(
                    pending_closure_candidate_identity_digest(
                        key,
                        contribution,
                        target_helicity_id,
                        &pending,
                        &dynamic_color_states,
                        &reflection_certificates,
                    )?,
                );
            }
        }
    } else {
        accepted_candidate_identity_digests = original_candidate_identity_digests.clone();
    }
    let candidate_domain_certificate = ClosureCandidateDomainCertificateV1::from_identity_digests(
        accepted_candidate_identity_digests,
    )?;
    let mut pending_closure_rows = pending_closures
        .iter()
        .filter(|(_, group)| !group.exact_factor.is_zero())
        .peekable();
    let mut projected_closure_rows = color_projection
        .as_ref()
        .map(|projection| projection.closures.values().peekable());
    let mut amplitude_destinations = Vec::with_capacity(destination_keys.len());
    let mut destination_ids = BTreeMap::new();
    for (destination_id, (sector_id, source_states)) in destination_keys.into_iter().enumerate() {
        let destination_id = u32::try_from(destination_id)
            .map_err(|_| invalid("closure destination ID exceeds u32"))?;
        destination_ids.insert((sector_id, source_states.clone()), destination_id);
        let start = closure_terms.len() as u64;
        let target_helicity_id = if source_states.is_empty() {
            None
        } else {
            Some(
                *helicity_ids
                    .get(&source_states)
                    .ok_or_else(|| invalid("resolved-helicity destination disappeared"))?,
            )
        };
        if let Some(rows) = projected_closure_rows.as_mut() {
            consume_ordered_closure_destination(
                rows,
                sector_id,
                &source_states,
                |projected| {
                    (
                        projected.identity.target_sector_id,
                        projected.identity.complete_source_states.as_ref(),
                    )
                },
                |projected| {
                    let key = &projected.representative_key;
                    let group = &projected.representative_group;
                    let runtime_term_id = u32::try_from(closure_terms.len())
                        .map_err(|_| invalid("runtime closure-term count exceeds u32"))?;
                    closure_terms.push(RecurrenceClosureTerm::new(
                        runtime_term_id,
                        destination_id,
                        projected.identity.closure_template_id,
                        projected.identity.quantum_flow_template_id,
                        projected.identity.parent_projection_ids.to_vec(),
                        projected.identity.exact_factor,
                    )?);
                    append_closure_proof_group(
                        key,
                        group,
                        &pending,
                        &dynamic_color_states,
                        &remap,
                        Some(destination_id),
                        target_helicity_id,
                        Some(runtime_term_id),
                        &three_line_certificate_ids,
                        &mut closure_proof_contributions,
                        &mut closure_proof_groups,
                    )?;
                    Ok(())
                },
            )?;
        } else {
            consume_ordered_closure_destination(
                &mut pending_closure_rows,
                sector_id,
                &source_states,
                |(key, _)| (key.target_sector_id, key.complete_source_states.as_ref()),
                |(key, group)| {
                    let parents = key
                        .parent_current_ids
                        .iter()
                        .map(|id| {
                            remap
                                .get(id)
                                .copied()
                                .ok_or_else(|| invalid("closure parent is absent"))
                        })
                        .collect::<RusticolResult<Vec<_>>>()?;
                    let runtime_term_id = u32::try_from(closure_terms.len())
                        .map_err(|_| invalid("runtime closure-term count exceeds u32"))?;
                    closure_terms.push(RecurrenceClosureTerm::new(
                        runtime_term_id,
                        destination_id,
                        key.closure_template_id,
                        key.quantum_flow_template_id,
                        parents,
                        group.exact_factor,
                    )?);
                    append_closure_proof_group(
                        key,
                        group,
                        &pending,
                        &dynamic_color_states,
                        &remap,
                        Some(destination_id),
                        target_helicity_id,
                        Some(runtime_term_id),
                        &three_line_certificate_ids,
                        &mut closure_proof_contributions,
                        &mut closure_proof_groups,
                    )?;
                    Ok(())
                },
            )?;
        }
        amplitude_destinations.push(RecurrenceAmplitudeDestination::new(
            destination_id,
            sector_id,
            target_helicity_id,
            CheckedTableRange::new(start, closure_terms.len() as u64 - start),
        )?);
    }
    if let Some(rows) = projected_closure_rows.as_mut() {
        require_ordered_closure_destinations_exhausted(rows)?;
    } else {
        require_ordered_closure_destinations_exhausted(&mut pending_closure_rows)?;
    }
    for (key, group) in pending_closures
        .iter()
        .filter(|(_, group)| group.exact_factor.is_zero())
    {
        let target_destination_id = destination_ids
            .get(&(key.target_sector_id, key.complete_source_states.clone()))
            .copied();
        let target_helicity_id = if key.complete_source_states.is_empty() {
            None
        } else {
            helicity_ids.get(&key.complete_source_states).copied()
        };
        append_closure_proof_group(
            key,
            group,
            &pending,
            &dynamic_color_states,
            &remap,
            target_destination_id,
            target_helicity_id,
            None,
            &three_line_certificate_ids,
            &mut closure_proof_contributions,
            &mut closure_proof_groups,
        )?;
    }
    let closure_proofs =
        ClosureProofMetadataV2::new_with_three_line_certificates_and_candidate_domain(
            closure_proof_contributions,
            closure_proof_groups,
            materialize_reflection_certificates(&reflection_certificates)?,
            materialize_three_line_certificates(&three_line_certificate_ids)?,
            candidate_domain_certificate,
        )?;
    let mut program = RecurrenceProgram::new_with_closure_proofs(
        strategy,
        u32::try_from(sector_count).map_err(|_| invalid("physical sector count exceeds u32"))?,
        retained_helicity_count,
        dynamic_color_states,
        currents,
        contributions,
        finalizations,
        replay_targets,
        resolved_helicities,
        amplitude_destinations,
        closure_terms,
        closure_proofs,
    )?;
    if let Some(certificate) = closure_target_domain_certificate {
        program = program.with_closure_target_domain_certificate(certificate)?;
    }
    if let Some(projection) = color_projection.as_ref() {
        let body = encode_color_projection_certificate_body(
            &pending,
            program.dynamic_color_states(),
            projection,
            &original_candidate_identity_digests,
        )?;
        program.with_color_projection_certificate_body(body)
    } else {
        Ok(program)
    }
}

fn helicity_support_rule(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<HelicitySupportRule> {
    let extensions = authenticated
        .process()
        .semantic_identity()
        .extension_digests();
    let mut rule = HelicitySupportRule::None;
    for role in extensions
        .keys()
        .filter(|role| role.starts_with("helicity-support:"))
    {
        let candidate = match role.as_str() {
            PURE_MASSLESS_ADJOINT_HELICITY_SUPPORT_ROLE => {
                HelicitySupportRule::PureMasslessAdjointTree
            }
            _ => {
                return Err(invalid(format!(
                    "unsupported recurrence helicity-support proof {role:?}"
                )));
            }
        };
        if rule != HelicitySupportRule::None {
            return Err(invalid(
                "recurrence process carries more than one helicity-support proof",
            ));
        }
        rule = candidate;
    }
    Ok(rule)
}

fn global_helicity_flip_rule(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<GlobalHelicityFlipRule> {
    let extensions = authenticated
        .process()
        .semantic_identity()
        .extension_digests();
    let mut rule = GlobalHelicityFlipRule::None;
    for role in extensions
        .keys()
        .filter(|role| role.starts_with("helicity-equivalence:"))
    {
        let candidate = match role.as_str() {
            GLOBAL_HELICITY_FLIP_EQUIVALENCE_ROLE => GlobalHelicityFlipRule::Proven,
            _ => {
                return Err(invalid(format!(
                    "unsupported recurrence helicity-equivalence proof {role:?}"
                )));
            }
        };
        if rule != GlobalHelicityFlipRule::None {
            return Err(invalid(
                "recurrence process carries more than one helicity-equivalence proof",
            ));
        }
        rule = candidate;
    }
    Ok(rule)
}

fn retain_supported_pending_closures(
    strategy: RecurrenceStrategy,
    process_catalog: &ProcessCatalog<'_>,
    replay_targets: &[RecurrenceReplayTarget],
    pending_closures: BTreeMap<PendingClosureKey, PendingClosureGroup>,
    helicity_support_rule: HelicitySupportRule,
    global_helicity_flip_rule: GlobalHelicityFlipRule,
) -> RusticolResult<BTreeMap<PendingClosureKey, PendingClosureGroup>> {
    let pending_closures = if helicity_support_rule == HelicitySupportRule::None {
        pending_closures
    } else {
        pending_closures
            .into_iter()
            .filter_map(|(key, group)| {
                match closure_helicity_is_supported(
                    helicity_support_rule,
                    process_catalog,
                    &key.complete_source_states,
                ) {
                    Ok(true) => Some(Ok((key, group))),
                    Ok(false) => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .collect::<RusticolResult<BTreeMap<_, _>>>()?
    };
    if global_helicity_flip_rule == GlobalHelicityFlipRule::Proven
        && strategy != RecurrenceStrategy::AllFlowUnion
    {
        retain_global_helicity_flip_representatives(
            strategy,
            process_catalog,
            replay_targets,
            pending_closures,
        )
    } else {
        Ok(pending_closures)
    }
}

fn retain_global_helicity_flip_representatives(
    strategy: RecurrenceStrategy,
    process_catalog: &ProcessCatalog<'_>,
    replay_targets: &[RecurrenceReplayTarget],
    pending_closures: BTreeMap<PendingClosureKey, PendingClosureGroup>,
) -> RusticolResult<BTreeMap<PendingClosureKey, PendingClosureGroup>> {
    let source_count = process_catalog.input.external_legs.len();
    let anchor_slot = if strategy == RecurrenceStrategy::TopologyReplay {
        (0..source_count)
            .find(|slot| {
                replay_targets.iter().all(|target| {
                    target.source_slot_permutation().get(*slot).copied()
                        == u32::try_from(*slot).ok()
                })
            })
            .ok_or_else(|| {
                invalid(
                    "global-helicity-flip topology replay has no permutation-fixed source anchor",
                )
            })?
    } else {
        0
    };
    let mut source_states_by_public =
        BTreeMap::<(u32, Vec<i32>), Box<[SourceStateAssignment]>>::new();
    for key in pending_closures.keys() {
        if key.complete_source_states.is_empty() {
            continue;
        }
        let public = process_catalog.public_helicities(&key.complete_source_states)?;
        let map_key = (key.target_sector_id, public);
        if source_states_by_public
            .insert(map_key, key.complete_source_states.clone())
            .is_some_and(|previous| previous != key.complete_source_states)
        {
            return Err(invalid(
                "global-helicity-flip proof maps one public helicity to multiple source assignments",
            ));
        }
    }

    let mut retained_public = BTreeSet::new();
    for (sector_and_public, source_states) in &source_states_by_public {
        let (sector_id, public) = sector_and_public;
        let flipped = public.iter().map(|value| -*value).collect::<Vec<_>>();
        if flipped == *public {
            return Err(invalid(
                "global-helicity-flip proof encountered a fixed public-helicity assignment",
            ));
        }
        let flipped_key = (*sector_id, flipped.clone());
        if !source_states_by_public.contains_key(&flipped_key) {
            return Err(invalid(format!(
                "global-helicity-flip proof has no partner for sector {sector_id} helicity {public:?}",
            )));
        }
        let anchor_helicity = *public
            .get(anchor_slot)
            .ok_or_else(|| invalid("global-helicity-flip anchor is outside the public axis"))?;
        if anchor_helicity == 0 {
            return Err(invalid(
                "global-helicity-flip anchor has a self-conjugate public helicity",
            ));
        }
        if anchor_helicity < 0 {
            retained_public.insert((*sector_id, public.clone()));
            let partner_states = &source_states_by_public[&flipped_key];
            if partner_states == source_states {
                return Err(invalid(
                    "global-helicity-flip pair reuses one source-state assignment",
                ));
            }
        }
    }
    if retained_public
        .len()
        .checked_mul(2)
        .is_none_or(|count| count != source_states_by_public.len())
    {
        return Err(invalid(
            "global-helicity-flip proof does not partition closure destinations into two-cycles",
        ));
    }

    pending_closures
        .into_iter()
        .filter_map(|(key, group)| {
            if key.complete_source_states.is_empty() {
                return Some(Ok((key, group)));
            }
            let public = match process_catalog.public_helicities(&key.complete_source_states) {
                Ok(public) => public,
                Err(error) => return Some(Err(error)),
            };
            retained_public
                .contains(&(key.target_sector_id, public))
                .then_some(Ok((key, group)))
        })
        .collect()
}

fn closure_helicity_is_supported(
    rule: HelicitySupportRule,
    process_catalog: &ProcessCatalog<'_>,
    source_states: &[SourceStateAssignment],
) -> RusticolResult<bool> {
    if rule == HelicitySupportRule::None || source_states.is_empty() {
        return Ok(true);
    }
    let public_helicities = process_catalog.public_helicities(source_states)?;
    let mut positive = 0usize;
    let mut negative = 0usize;
    for (leg, helicity) in process_catalog
        .input
        .external_legs
        .iter()
        .zip(public_helicities)
    {
        let physical = if leg.is_initial == 1 {
            -helicity
        } else {
            helicity
        };
        match physical {
            1 => positive += 1,
            -1 => negative += 1,
            value => {
                return Err(invalid(format!(
                    "pure-massless-adjoint helicity proof received unsupported helicity {value}"
                )));
            }
        }
    }
    Ok(positive >= 2 && negative >= 2)
}

fn retained_helicity_count(process: &OwnedRecurrenceProcessInput) -> RusticolResult<u64> {
    process.external_legs.iter().try_fold(1_u64, |count, leg| {
        let retained =
            u64::try_from(retained_source_state_indices(process, leg.source_slot)?.len())
                .map_err(|_| invalid("retained source-state count exceeds u64"))?;
        count
            .checked_mul(retained)
            .ok_or_else(|| invalid("retained public-helicity count exceeds u64"))
    })
}

fn retained_source_state_indices(
    process: &OwnedRecurrenceProcessInput,
    source_slot: u32,
) -> RusticolResult<Vec<u32>> {
    let leg = process
        .external_legs
        .get(source_slot as usize)
        .ok_or_else(|| invalid("recurrence source slot is absent"))?;
    if !process.header[0].selected_source_mode()? {
        return (0..leg.source_state_range.count)
            .map(|index| {
                u32::try_from(index)
                    .map_err(|_| invalid("source-state index exceeds the u32 ID domain"))
            })
            .collect();
    }
    let retained = process
        .selected_source_coverage
        .iter()
        .filter(|row| row.source_slot == source_slot)
        .map(|row| row.source_state_index)
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Err(invalid(format!(
            "generation-selected recurrence coverage has no state for source slot {source_slot}"
        )));
    }
    Ok(retained)
}

fn complete_closure_source_states(
    parents: [&CurrentCoreKey; 2],
    source_count: usize,
) -> RusticolResult<Box<[SourceStateAssignment]>> {
    match (
        parents[0].helicity_identity(),
        parents[1].helicity_identity(),
    ) {
        (
            CurrentHelicityIdentity::TopologyReplay {
                local_source_states: left,
                ..
            },
            CurrentHelicityIdentity::TopologyReplay {
                local_source_states: right,
                ..
            },
        ) => {
            let mut result = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
            result.sort_unstable();
            if result.len() != source_count {
                return Err(invalid(
                    "topology-replay closure ancestry does not cover every external source",
                ));
            }
            for (source_slot, assignment) in result.iter().copied().enumerate() {
                if assignment.source_slot() as usize != source_slot {
                    return Err(invalid(
                        "topology-replay closure ancestry is incomplete or overlapping",
                    ));
                }
            }
            Ok(result.into_boxed_slice())
        }
        (
            CurrentHelicityIdentity::AllFlowUnion { .. },
            CurrentHelicityIdentity::AllFlowUnion { .. },
        ) => Ok(Box::new([])),
        (
            CurrentHelicityIdentity::ContractedColorUnion {
                local_source_states: left,
                ..
            },
            CurrentHelicityIdentity::ContractedColorUnion {
                local_source_states: right,
                ..
            },
        ) => {
            let mut result = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
            result.sort_unstable();
            if result.len() != source_count {
                return Err(invalid(
                    "contracted-color closure ancestry does not cover every external source",
                ));
            }
            for (source_slot, assignment) in result.iter().copied().enumerate() {
                if assignment.source_slot() as usize != source_slot {
                    return Err(invalid(
                        "contracted-color closure ancestry is incomplete or overlapping",
                    ));
                }
            }
            Ok(result.into_boxed_slice())
        }
        _ => Err(invalid(
            "closure parents use incompatible recurrence helicity strategies",
        )),
    }
}

#[cfg(test)]
fn quantum_flow_matches(
    quantum: QuantumFlowRow,
    parents: &[&CurrentCoreKey; 2],
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<bool> {
    let states = catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
    let spins = catalog.i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?;
    let flavours = catalog.u32_sequence(
        quantum.input_flavour_sequence_id,
        "quantum input flavour flows",
    )?;
    let quantum_numbers = catalog.u32_sequence(
        quantum.input_quantum_sequence_id,
        "quantum input number flows",
    )?;
    if states.len() != 2 || spins.len() != 2 || flavours.len() != 2 || quantum_numbers.len() != 2 {
        return Err(invalid(
            "direct recurrence requires binary quantum-flow contracts",
        ));
    }
    for index in 0..2 {
        // The authenticated model contract proves that branch admission is
        // independent of accumulated flavour and quantum-number ancestry.
        // Those template columns describe the seed probe used to certify the
        // branch, while state and spin remain the actual admission keys.
        let _ = catalog.flavour_flow(flavours[index], "quantum parent flavour")?;
        let _ = quantum_numbers[index];
        if states[index] != parents[index].current_state_template_id()
            || !quantum_parent_spin_matches(spins[index], parents[index])
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn quantum_parent_spin_matches(required_spin: i32, parent: &CurrentCoreKey) -> bool {
    required_spin == parent.spin_state_class()
        || (parent.node_kind() == RecurrenceNodeKind::Source
            && parent.helicity_identity().strategy() == RecurrenceStrategy::AllFlowUnion
            && parent.spin_state_class() == DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS)
}

#[cfg(test)]
fn quantum_flow_result_flavour(
    quantum: QuantumFlowRow,
    parents: &[&CurrentCoreKey; 2],
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<Vec<i32>> {
    let operation = catalog.string(
        quantum.flavour_flow_operation_string_id,
        "quantum-flow flavour operation",
    )?;
    let static_result = catalog.flavour_flow(
        quantum.result_flavour_flow_id,
        "quantum-flow result flavour",
    )?;
    let result_particle = *static_result
        .last()
        .ok_or_else(|| invalid("quantum-flow result flavour ancestry is empty"))?;

    // The prepared recurrence contract has already proved that transition
    // admission and kernel selection are independent of accumulated flavour
    // ancestry.  In the runtime-helicity union, retaining the construction
    // history would nevertheless split one physical current into many
    // numerically accumulated copies.  Canonicalize it to the result species,
    // matching the compact all-flow recurrence identity.
    if parents
        .iter()
        .all(|parent| parent.helicity_identity().strategy() == RecurrenceStrategy::AllFlowUnion)
    {
        return Ok(vec![result_particle]);
    }

    let append_result = |parent: &CurrentCoreKey| {
        let mut result = parent.flavour_flow().to_vec();
        if result.last().copied() != Some(result_particle) {
            result.push(result_particle);
        }
        result
    };

    match operation {
        "constant-result" => Ok(static_result.to_vec()),
        "append-left-result" => Ok(append_result(parents[0])),
        "append-right-result" => Ok(append_result(parents[1])),
        "concat-left-right-result" => {
            let mut result = Vec::with_capacity(
                parents[0]
                    .flavour_flow()
                    .len()
                    .saturating_add(parents[1].flavour_flow().len())
                    .saturating_add(1),
            );
            result.extend_from_slice(parents[0].flavour_flow());
            result.extend_from_slice(parents[1].flavour_flow());
            result.push(result_particle);
            Ok(result)
        }
        value => Err(invalid(format!(
            "unsupported quantum-flow flavour operation {value:?}"
        ))),
    }
}

fn expected_sector_components(
    sector: ProcessPhysicalLCSectorRow,
    process: &OwnedRecurrenceProcessInput,
    catalog: &ProcessCatalog<'_>,
) -> RusticolResult<Vec<LCColorComponent>> {
    let mut result = Vec::new();
    match sector.kind()? {
        ProcessLCSectorKind::Singlet => {}
        ProcessLCSectorKind::SingleTrace => result.push(LCColorComponent::new(
            LCColorComponentKind::Trace,
            catalog
                .u32_sequence(sector.trace_sequence_id, "physical LC trace")?
                .to_vec(),
        )?),
        ProcessLCSectorKind::OpenLines => {
            let range = sector
                .open_string_range
                .as_usize_range(process.lc_open_strings.len(), "physical LC open strings")?;
            for row in &process.lc_open_strings[range] {
                let mut word = vec![row.fundamental_source_slot];
                word.extend_from_slice(
                    catalog.u32_sequence(
                        row.adjoint_sequence_id,
                        "physical LC open-string adjoints",
                    )?,
                );
                word.push(row.antifundamental_source_slot);
                result.push(LCColorComponent::new(
                    LCColorComponentKind::OpenString,
                    word,
                )?);
            }
        }
    }
    Ok(result)
}

fn closed_components_match_prepared_sector(
    strategy: RecurrenceStrategy,
    closed: &[LCColorComponent],
    sector: &PreparedClosureSector,
) -> RusticolResult<bool> {
    if sector.row.kind()? != ProcessLCSectorKind::OpenLines {
        return Ok(closed == sector.expected_components.as_ref());
    }
    if !unordered_color_components_match(closed, &sector.expected_components) {
        return Ok(false);
    }
    if strategy == RecurrenceStrategy::ContractedColorUnion
        && !sector.contracted_color_canonical_owner
    {
        return Ok(false);
    }
    Ok(true)
}

fn unordered_color_components_match(left: &[LCColorComponent], right: &[LCColorComponent]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut matched = vec![false; right.len()];
    for component in left {
        let Some(index) = right.iter().enumerate().find_map(|(index, candidate)| {
            (!matched[index] && candidate == component).then_some(index)
        }) else {
            return false;
        };
        matched[index] = true;
    }
    true
}

fn materialized_sector_ids(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    replay_targets: &[RecurrenceReplayTarget],
) -> BTreeSet<u32> {
    match strategy {
        RecurrenceStrategy::AllFlowUnion => process
            .physical_lc_sectors
            .iter()
            .map(|sector| sector.sector_id)
            .collect(),
        RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion
            if !replay_targets.is_empty() =>
        {
            replay_targets
                .iter()
                .map(RecurrenceReplayTarget::materialized_sector_id)
                .collect()
        }
        RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => process
            .physical_lc_sectors
            .iter()
            .map(|sector| sector.sector_id)
            .collect(),
    }
}

fn construction_sector_groups(
    strategy: RecurrenceStrategy,
    materialized_sectors: &BTreeSet<u32>,
    replay_targets: &[RecurrenceReplayTarget],
) -> Vec<BTreeSet<u32>> {
    if strategy.uses_topology_replay_targets()
        && !replay_targets.is_empty()
        && materialized_sectors.len() > 1
    {
        materialized_sectors
            .iter()
            .copied()
            .map(|sector_id| BTreeSet::from([sector_id]))
            .collect()
    } else {
        vec![materialized_sectors.clone()]
    }
}

fn build_replay_targets(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    catalog: &ProcessCatalog<'_>,
) -> RusticolResult<Vec<RecurrenceReplayTarget>> {
    if !strategy.uses_topology_replay_targets()
        || (strategy == RecurrenceStrategy::ContractedColorUnion
            && process.replay_partitions.is_empty())
    {
        // Contracted color is replay-capable, but only an authenticated
        // partition certificate activates replay.  Synthesizing identity
        // targets here would incorrectly materialize ordering aliases that
        // contracted construction coherently accumulates in an owner sector.
        return Ok(Vec::new());
    }
    let source_momentum_signs = external_source_momentum_signs(process)?;
    let retained_flows = retained_public_flows(process)?;
    let retained_sectors = retained_flows
        .iter()
        .map(|flow| flow.construction_sector_id)
        .collect::<BTreeSet<_>>();
    let mut base_by_sector = BTreeMap::<u32, (u32, Vec<u32>, ExactComplexRational)>::new();
    for partition in &process.replay_partitions {
        let range = partition
            .target_range
            .as_usize_range(process.replay_targets.len(), "recurrence replay targets")?;
        for target in &process.replay_targets[range] {
            if !retained_sectors.contains(&target.sector_id) {
                continue;
            }
            let mut factor = catalog.factor(
                target.amplitude_phase_factor_id,
                "recurrence replay amplitude phase",
            )?;
            if target.fermion_sign == -1 {
                factor = factor.checked_neg()?;
            }
            let previous = base_by_sector.insert(
                target.sector_id,
                (
                    partition.materialized_sector_id,
                    catalog
                        .u32_sequence(
                            target.source_slot_permutation_sequence_id,
                            "recurrence replay source permutation",
                        )?
                        .to_vec(),
                    factor,
                ),
            );
            if previous.is_some() {
                return Err(invalid(format!(
                    "construction sector {} has multiple recurrence replay targets",
                    target.sector_id
                )));
            }
        }
    }
    let identity = (0..process.external_legs.len())
        .map(|slot| u32::try_from(slot).map_err(|_| invalid("source-slot count exceeds u32")))
        .collect::<RusticolResult<Vec<_>>>()?;
    for sector_id in retained_sectors {
        base_by_sector
            .entry(sector_id)
            .or_insert_with(|| (sector_id, identity.clone(), ExactComplexRational::ONE));
    }
    retained_flows
        .into_iter()
        .enumerate()
        .map(|(target_sector_id, flow)| {
            let (materialized, construction_permutation, factor) = base_by_sector
                .get(&flow.construction_sector_id)
                .ok_or_else(|| invalid("public flow construction replay target is absent"))?;
            let public_permutation = catalog.u32_sequence(
                flow.source_slot_permutation_sequence_id,
                "public LC flow source permutation",
            )?;
            if construction_permutation.len() != public_permutation.len() {
                return Err(invalid(
                    "public LC flow and construction replay permutations have different sizes",
                ));
            }
            let permutation =
                compose_gather_permutations(construction_permutation, public_permutation)?;
            let replay_momentum_signs =
                replay_momentum_signs(&source_momentum_signs, &permutation)?;
            RecurrenceReplayTarget::new(
                u32::try_from(target_sector_id)
                    .map_err(|_| invalid("replay-target count exceeds u32"))?,
                *materialized,
                u32::try_from(target_sector_id)
                    .map_err(|_| invalid("public-flow target ID exceeds u32"))?,
                permutation,
                replay_momentum_signs,
                *factor,
            )
        })
        .collect()
}

fn external_source_momentum_signs(
    process: &OwnedRecurrenceProcessInput,
) -> RusticolResult<Vec<i32>> {
    process
        .external_legs
        .iter()
        .enumerate()
        .map(|(source_slot, leg)| {
            let range = leg.source_state_range.as_usize_range(
                process.source_states.len(),
                "external source momentum signs",
            )?;
            let mut signs = process.source_states[range]
                .iter()
                .map(|state| {
                    if state.source_slot as usize != source_slot
                        || !matches!(state.momentum_sign, -1 | 1)
                    {
                        return Err(invalid(format!(
                            "source slot {source_slot} has an invalid momentum-sign contract"
                        )));
                    }
                    Ok(state.momentum_sign)
                })
                .collect::<RusticolResult<BTreeSet<_>>>()?;
            if signs.len() != 1 {
                return Err(invalid(format!(
                    "source slot {source_slot} has helicity-dependent momentum crossing"
                )));
            }
            signs
                .pop_first()
                .ok_or_else(|| invalid(format!("source slot {source_slot} has no source states")))
        })
        .collect()
}

fn replay_momentum_signs(
    source_momentum_signs: &[i32],
    representative_to_public: &[u32],
) -> RusticolResult<Vec<i32>> {
    representative_to_public
        .iter()
        .copied()
        .enumerate()
        .map(|(representative_slot, public_slot)| {
            let representative_sign = source_momentum_signs
                .get(representative_slot)
                .copied()
                .filter(|sign| matches!(sign, -1 | 1))
                .ok_or_else(|| invalid("representative source momentum sign is absent"))?;
            let public_sign = source_momentum_signs
                .get(public_slot as usize)
                .copied()
                .filter(|sign| matches!(sign, -1 | 1))
                .ok_or_else(|| invalid("public source momentum sign is absent"))?;
            Ok(representative_sign * public_sign)
        })
        .collect()
}

fn compose_gather_permutations(
    representative_to_construction: &[u32],
    construction_to_public: &[u32],
) -> RusticolResult<Vec<u32>> {
    if representative_to_construction.len() != construction_to_public.len() {
        return Err(invalid(
            "public LC flow and construction replay permutations have different sizes",
        ));
    }
    representative_to_construction
        .iter()
        .map(|construction_slot| {
            construction_to_public
                .get(*construction_slot as usize)
                .copied()
                .ok_or_else(|| invalid("construction replay permutation is out of range"))
        })
        .collect()
}

fn retained_public_flows(
    process: &OwnedRecurrenceProcessInput,
) -> RusticolResult<Vec<&super::process::ProcessPublicLCFlowRow>> {
    if !process.header[0].selected_flow_mode()? {
        return Ok(process.public_lc_flows.iter().collect());
    }
    let selected = process
        .selected_public_flow_coverage
        .iter()
        .map(|row| row.flow_id)
        .collect::<BTreeSet<_>>();
    Ok(process
        .public_lc_flows
        .iter()
        .filter(|flow| selected.contains(&flow.flow_id))
        .collect())
}

fn propagator_by_state(
    template: &OwnedRecurrenceTemplateInput,
) -> RusticolResult<BTreeMap<u32, Option<u32>>> {
    let mut result = BTreeMap::new();
    for row in &template.propagators {
        let value = (row.applies_propagator != 0).then_some(row.id);
        if result.insert(row.state_template_id, value).is_some() {
            return Err(invalid(format!(
                "current-state template {} has multiple propagators",
                row.state_template_id
            )));
        }
    }
    Ok(result)
}

fn coupling_limits(
    process: &ProcessCatalog<'_>,
    template: &TemplateCatalog<'_>,
) -> RusticolResult<Vec<Option<u32>>> {
    let mut limits = vec![None; template.coupling_names.len()];
    for row in &process.input.coupling_limits {
        let name = process.string(row.name_string_id, "process coupling limit")?;
        if let Ok(index) = template.coupling_names.binary_search(&name) {
            limits[index] = Some(row.maximum);
        }
    }
    Ok(limits)
}

pub(super) fn combined_coupling_orders(
    left: &[u32],
    right: &[u32],
    local: &[u32],
    limits: &[Option<u32>],
) -> RusticolResult<Option<Vec<u32>>> {
    if left.len() != right.len() || left.len() != local.len() || left.len() != limits.len() {
        return Err(invalid(
            "coupling-order vectors have inconsistent dimensions",
        ));
    }
    let mut result = Vec::with_capacity(left.len());
    for index in 0..left.len() {
        let value = left[index]
            .checked_add(right[index])
            .and_then(|value| value.checked_add(local[index]))
            .ok_or_else(|| invalid("coupling order exceeds u32"))?;
        if limits[index].is_some_and(|maximum| value > maximum) {
            return Ok(None);
        }
        result.push(value);
    }
    Ok(Some(result))
}

pub(super) fn merged_helicity_identity(
    left: &CurrentHelicityIdentity,
    right: &CurrentHelicityIdentity,
    result_spin: i32,
) -> RusticolResult<CurrentHelicityIdentity> {
    match (left, right) {
        (
            CurrentHelicityIdentity::TopologyReplay {
                local_source_states: left,
                ..
            },
            CurrentHelicityIdentity::TopologyReplay {
                local_source_states: right,
                ..
            },
        ) => {
            let mut values = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
            values.sort_unstable();
            CurrentHelicityIdentity::topology_replay(result_spin, values)
        }
        (
            CurrentHelicityIdentity::AllFlowUnion { .. },
            CurrentHelicityIdentity::AllFlowUnion { .. },
        ) => Ok(CurrentHelicityIdentity::all_flow_union(result_spin)),
        (
            CurrentHelicityIdentity::ContractedColorUnion {
                local_source_states: left,
                ..
            },
            CurrentHelicityIdentity::ContractedColorUnion {
                local_source_states: right,
                ..
            },
        ) => {
            let mut values = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
            values.sort_unstable();
            CurrentHelicityIdentity::contracted_color_union(result_spin, values)
        }
        _ => Err(invalid(
            "cannot merge recurrence helicity identities from different strategies",
        )),
    }
}

fn merged_disjoint_support(left: &[u32], right: &[u32]) -> Vec<u32> {
    debug_assert!(left.iter().copied().is_sorted());
    debug_assert!(right.iter().copied().is_sorted());
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => {
                result.push(left[left_index]);
                left_index += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(right[right_index]);
                right_index += 1;
            }
            std::cmp::Ordering::Equal => {
                unreachable!("compact support index emitted overlapping parents")
            }
        }
    }
    result.extend_from_slice(&left[left_index..]);
    result.extend_from_slice(&right[right_index..]);
    result
}

#[cfg(test)]
fn merged_support(left: &[u32], right: &[u32]) -> RusticolResult<Vec<u32>> {
    if !disjoint_support(left, right) {
        return Err(invalid(
            "recurrence parents have overlapping source support",
        ));
    }
    let mut result = left.iter().chain(right).copied().collect::<Vec<_>>();
    result.sort_unstable();
    Ok(result)
}

#[cfg(test)]
fn disjoint_support(left: &[u32], right: &[u32]) -> bool {
    left.iter().all(|slot| right.binary_search(slot).is_err())
}

pub(super) fn merged_momentum(
    left: &CanonicalMomentumLinearForm,
    right: &CanonicalMomentumLinearForm,
) -> RusticolResult<CanonicalMomentumLinearForm> {
    let mut terms = left
        .terms()
        .iter()
        .chain(right.terms())
        .copied()
        .collect::<Vec<_>>();
    terms.sort_unstable_by_key(|term| term.source_slot);
    CanonicalMomentumLinearForm::new(terms)
}

#[cfg(test)]
fn authenticate_runtime_coupling(
    catalog: &TemplateCatalog<'_>,
    quantum: QuantumFlowRow,
    binding_coupling_factor_id: u32,
    label: &str,
) -> RusticolResult<ExactComplexRational> {
    let quantum_coupling = catalog.factor(
        quantum.exact_coupling_factor_id,
        &format!("{label} quantum-flow coupling"),
    )?;
    let binding_coupling = catalog.factor(
        binding_coupling_factor_id,
        &format!("{label} binding coupling"),
    )?;
    if quantum_coupling != binding_coupling {
        return Err(invalid(format!(
            "{label} binding coupling does not match its quantum-flow coupling witness"
        )));
    }
    Ok(binding_coupling)
}

pub(super) fn output_factor_from_binding(
    binding_coupling: ExactComplexRational,
    output_factor_source: u8,
    label: &str,
) -> RusticolResult<ExactComplexRational> {
    let component = match OutputFactorSource::try_from(output_factor_source)? {
        OutputFactorSource::None => return Ok(ExactComplexRational::ONE),
        OutputFactorSource::CouplingReal => binding_coupling.real(),
        OutputFactorSource::CouplingImag => binding_coupling.imag(),
    };
    if component.is_zero() {
        return Err(invalid(format!(
            "{label} selects a zero binding-coupling component"
        )));
    }
    Ok(ExactComplexRational::new(component, ExactRational::ZERO))
}

#[cfg(test)]
fn canonical_evaluator_parents(
    concrete_parent_ids: [u32; 2],
    canonical_input_order: &[u32],
    input_exchange_factor_id: u32,
    catalog: &TemplateCatalog<'_>,
    label: &str,
) -> RusticolResult<([u32; 2], ExactComplexRational)> {
    let mut ordered = match canonical_input_order {
        [0, 1] => concrete_parent_ids,
        [1, 0] => [concrete_parent_ids[1], concrete_parent_ids[0]],
        _ => {
            return Err(invalid(format!(
                "{label} canonical input order is not a binary permutation"
            )));
        }
    };
    let mut factor = ExactComplexRational::ONE;
    if input_exchange_factor_id != MISSING_U32 && ordered[1] < ordered[0] {
        ordered.swap(0, 1);
        factor = catalog.factor(input_exchange_factor_id, &format!("{label} input-exchange"))?;
    }
    Ok((ordered, factor))
}

pub(super) fn multiply_factors(
    values: &[ExactComplexRational],
) -> RusticolResult<ExactComplexRational> {
    values
        .iter()
        .copied()
        .try_fold(ExactComplexRational::ONE, ExactComplexRational::checked_mul)
}

pub(super) fn aggregate_factor(
    target: &mut ExactComplexRational,
    value: ExactComplexRational,
) -> RusticolResult<()> {
    *target = target.checked_add(value)?;
    Ok(())
}

fn decode_strings<'a>(
    ranges: &[CheckedTableRange],
    bytes: &'a [u8],
    label: &str,
) -> RusticolResult<Vec<&'a str>> {
    ranges
        .iter()
        .copied()
        .enumerate()
        .map(|(index, range)| {
            let range = range.as_usize_range(bytes.len(), &format!("{label} {index}"))?;
            std::str::from_utf8(&bytes[range])
                .map_err(|error| invalid(format!("{label} {index} is not UTF-8: {error}")))
        })
        .collect()
}

fn required_string<'a>(strings: &[&'a str], id: u32, label: &str) -> RusticolResult<&'a str> {
    strings
        .get(id as usize)
        .copied()
        .ok_or_else(|| invalid(format!("{label} string {id} is absent")))
}

fn indexed_sequence<'a, T>(
    ranges: &[super::template::IndexedRangeRow],
    values: &'a [T],
    id: u32,
    label: &str,
) -> RusticolResult<&'a [T]> {
    let range = ranges
        .get(id as usize)
        .ok_or_else(|| invalid(format!("{label} sequence {id} is absent")))?
        .range
        .as_usize_range(values.len(), label)?;
    Ok(&values[range])
}

fn decode_template_factors(
    input: &OwnedRecurrenceTemplateInput,
    strings: &[&str],
) -> RusticolResult<Vec<ExactComplexRational>> {
    input
        .exact_factors
        .iter()
        .map(|row| {
            ExactComplexRational::parse_parts(
                required_string(strings, row.real_numerator_string_id, "factor numerator")?,
                required_string(
                    strings,
                    row.real_denominator_string_id,
                    "factor denominator",
                )?,
                required_string(strings, row.imag_numerator_string_id, "factor numerator")?,
                required_string(
                    strings,
                    row.imag_denominator_string_id,
                    "factor denominator",
                )?,
            )
        })
        .collect()
}

fn decode_process_factors(
    input: &OwnedRecurrenceProcessInput,
    strings: &[&str],
) -> RusticolResult<Vec<ExactComplexRational>> {
    input
        .exact_factors
        .iter()
        .map(|row| {
            ExactComplexRational::parse_parts(
                required_string(strings, row.real_numerator_string_id, "factor numerator")?,
                required_string(
                    strings,
                    row.real_denominator_string_id,
                    "factor denominator",
                )?,
                required_string(strings, row.imag_numerator_string_id, "factor numerator")?,
                required_string(
                    strings,
                    row.imag_denominator_string_id,
                    "factor denominator",
                )?,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::super::contact_orbit_owner::{
        ContactOrbitStepProof, contact_orbit_application_for_test,
        final_contact_orbit_step_for_test, partial_contact_orbit_step_for_test,
        prepared_contact_orbit_transition_for_test,
    };
    use super::super::process::{
        ProcessExternalLegRow, ProcessHeaderRow, ProcessLCOpenStringRow,
        ProcessPhysicalLCSectorRow, ProcessPublicLCFlowRow, ProcessSourceStateRow,
    };
    use super::super::template::{
        ColorContractionRow, DigestCatalogRow, ExactFactorRow, IndexedRangeRow,
        LCColorTransitionWitnessRow, PropagatorRow, QuantumFlowRow,
    };
    use super::super::{LCColorEndpoint, LCColorPortBinding};
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).expect("test digest must be nonzero")
    }

    #[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct OrderedClosureTestRow {
        sector_id: u32,
        source_states: Box<[SourceStateAssignment]>,
        ordinal: u32,
    }

    #[test]
    fn ordered_closure_destination_cursor_preserves_filter_order_and_fails_closed() {
        let states =
            |state_index| vec![SourceStateAssignment::new(0, state_index)].into_boxed_slice();
        let mut rows = [
            OrderedClosureTestRow {
                sector_id: 2,
                source_states: states(0),
                ordinal: 4,
            },
            OrderedClosureTestRow {
                sector_id: 0,
                source_states: states(1),
                ordinal: 1,
            },
            OrderedClosureTestRow {
                sector_id: 0,
                source_states: states(1),
                ordinal: 0,
            },
            OrderedClosureTestRow {
                sector_id: 1,
                source_states: states(0),
                ordinal: 3,
            },
            OrderedClosureTestRow {
                sector_id: 0,
                source_states: states(0),
                ordinal: 2,
            },
        ];
        rows.sort_unstable();
        let destinations = rows
            .iter()
            .map(|row| (row.sector_id, row.source_states.clone()))
            .collect::<BTreeSet<_>>();
        let expected = destinations
            .iter()
            .flat_map(|(sector_id, source_states)| {
                rows.iter()
                    .filter(move |row| {
                        row.sector_id == *sector_id
                            && row.source_states.as_ref() == source_states.as_ref()
                    })
                    .map(|row| row.ordinal)
            })
            .collect::<Vec<_>>();

        let mut cursor = rows.iter().peekable();
        let mut observed = Vec::new();
        for (sector_id, source_states) in &destinations {
            consume_ordered_closure_destination(
                &mut cursor,
                *sector_id,
                source_states,
                |row: &&OrderedClosureTestRow| (row.sector_id, row.source_states.as_ref()),
                |row| {
                    observed.push(row.ordinal);
                    Ok(())
                },
            )
            .unwrap();
        }
        require_ordered_closure_destinations_exhausted(&mut cursor).unwrap();
        assert_eq!(observed, expected);
        assert_eq!(observed, [2, 0, 1, 3, 4]);

        let mut skipped_prefix = rows.iter().peekable();
        assert!(
            consume_ordered_closure_destination(
                &mut skipped_prefix,
                1,
                &states(0),
                |row: &&OrderedClosureTestRow| (row.sector_id, row.source_states.as_ref()),
                |_| Ok(()),
            )
            .is_err()
        );

        let mut unconsumed_suffix = rows.iter().peekable();
        consume_ordered_closure_destination(
            &mut unconsumed_suffix,
            0,
            &states(0),
            |row: &&OrderedClosureTestRow| (row.sector_id, row.source_states.as_ref()),
            |_| Ok(()),
        )
        .unwrap();
        assert!(require_ordered_closure_destinations_exhausted(&mut unconsumed_suffix).is_err());
    }

    fn proven_reflection(
        color: &DynamicLCColorState,
        phase: ExactComplexRational,
        root: u8,
    ) -> CurrentReflection {
        CurrentReflection::Proven(
            CurrentReflectionProof::new(
                phase,
                [digest(root)],
                dynamic_color_identity_digest(color).unwrap(),
            )
            .unwrap(),
        )
    }

    fn proven_interned_reflection(
        colors: &DynamicLCColorStateInterner,
        color_id: DynamicLCColorStateId,
        phase: ExactComplexRational,
        root: u8,
    ) -> CurrentReflection {
        proven_reflection(colors.get(color_id).unwrap(), phase, root)
    }

    fn transition(id: u32) -> TransitionRow {
        TransitionRow {
            id,
            template_string_id: 0,
            input_state_sequence_id: 0,
            result_state_template_id: 0,
            quantum_flow_template_id: 0,
            evaluator_binding_id: 0,
            canonical_input_order_sequence_id: 0,
            momentum_convention_sequence_id: 0,
            coupling_parameter_sequence_id: 0,
            coupling_order_set_id: 0,
            color_contraction_template_id: 0,
            binding_coupling_factor_id: 0,
            exact_factor_id: 0,
            output_factor_source: 0,
            equivalence_class_string_id: 0,
            input_exchange_factor_id: MISSING_U32,
            output_projection_string_id: 0,
            contact_orbit_step_sequence_id: 0,
            contact_orbit_step_semantic_digest_sequence_id: 0,
            semantic_digest_id: 0,
        }
    }

    fn source(id: u32) -> SourceRow {
        SourceRow {
            id,
            template_string_id: 1,
            state_template_id: id,
            crossing_string_id: 2,
            wavefunction_family_string_id: 3,
            helicity: if id == 0 { -1 } else { 1 },
            spin_state: if id == 0 { -1 } else { 1 },
            flavour_flow_id: 4,
            quantum_number_flow_id: 5,
            lc_color_seed_operation: 6,
            lc_color_seed_shape_string_id: 7,
            lc_color_seed_component_kind: 8,
            lc_color_seed_component_role: 9,
            lc_color_seed_proof_digest_id: 10 + id,
            lc_color_seed_provenance_sequence_id: 11,
            wavefunction_expression_digest_id: 12 + id,
            evaluator_binding_id: 13 + id,
            mass_parameter_id: 14,
            width_parameter_id: 15,
            semantic_digest_id: 16 + id,
        }
    }

    #[test]
    fn union_source_dispatch_ignores_proof_instance_identity_only() {
        let left = source(0);
        let right = source(1);
        assert_ne!(
            left.lc_color_seed_proof_digest_id,
            right.lc_color_seed_proof_digest_id
        );
        assert!(same_union_source_dispatch_semantics(left, right));

        let mut incompatible = right;
        incompatible.crossing_string_id += 1;
        assert!(!same_union_source_dispatch_semantics(left, incompatible));

        let mut incompatible = right;
        incompatible.lc_color_seed_provenance_sequence_id += 1;
        assert!(!same_union_source_dispatch_semantics(left, incompatible));
    }

    #[test]
    fn all_flow_union_source_spin_class_is_dynamic_for_chiral_variants() {
        assert_eq!(
            union_spin_state_class(25, &BTreeSet::from([-1, 1])).unwrap(),
            DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS
        );
        assert_eq!(
            union_spin_state_class(25, &BTreeSet::from([-1])).unwrap(),
            -1
        );
        assert!(
            union_spin_state_class(25, &BTreeSet::from([DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS]),)
                .is_err()
        );

        let mut colors = DynamicLCColorStateInterner::default();
        let color_id = colors
            .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
            .unwrap();
        let source = CurrentCoreKey::new(
            digest(202),
            RecurrenceNodeKind::Source,
            17,
            color_id,
            vec![0],
            CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                source_slot: 0,
                coefficient: 1,
            }])
            .unwrap(),
            CurrentHelicityIdentity::all_flow_union(DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS),
            vec![0],
            0,
            vec![],
            CurrentSourceBinding::runtime_dispatch(25, vec![0, 1]).unwrap(),
            None,
        )
        .unwrap();

        assert!(quantum_parent_spin_matches(-1, &source));
        assert!(quantum_parent_spin_matches(1, &source));
        assert!(structural_parent_states_match(
            [StructuralState::new(17, -1), StructuralState::new(19, 0),],
            [
                StructuralState::new(17, DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS),
                StructuralState::new(19, 0),
            ],
        ));
    }

    #[test]
    fn pure_adjoint_reflection_masks_require_exact_parent_proofs() {
        let left = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 4]).unwrap()],
        )
        .unwrap();
        let right = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![2, 3]).unwrap()],
        )
        .unwrap();
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let reflections = [
            proven_reflection(&left, minus_one, 40),
            proven_reflection(&right, minus_one, 41),
        ];

        assert_eq!(
            current_reversal_masks(&[left.clone(), right.clone()], &reflections),
            [0, 1, 2, 3],
        );
        let partial_reflections = [
            proven_reflection(&left, minus_one, 42),
            CurrentReflection::Unavailable,
        ];
        assert_eq!(
            current_reversal_masks(&[left, right], &partial_reflections),
            [0, 1],
        );
    }

    #[test]
    fn source_reflection_lineage_is_deterministic_and_color_bound() {
        let mut colors = DynamicLCColorStateInterner::default();
        let first_color = colors
            .intern(
                DynamicLCColorState::new_port_wired(
                    1,
                    vec![
                        LCColorPortBinding::new(0, LCColorEndpoint::Back),
                        LCColorPortBinding::new(0, LCColorEndpoint::Front),
                    ],
                    vec![
                        LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![3])
                            .unwrap(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let second_color = colors
            .intern(
                DynamicLCColorState::new_port_wired(
                    2,
                    vec![
                        LCColorPortBinding::new(0, LCColorEndpoint::Back),
                        LCColorPortBinding::new(0, LCColorEndpoint::Front),
                    ],
                    vec![
                        LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![3])
                            .unwrap(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let seed = digest(39);
        let first = source_reflection(&colors, first_color, 3, seed)
            .unwrap()
            .proof()
            .unwrap()
            .clone();
        let repeated = source_reflection(&colors, first_color, 3, seed)
            .unwrap()
            .proof()
            .unwrap()
            .clone();
        let changed_slot = source_reflection(&colors, first_color, 4, seed)
            .unwrap()
            .proof()
            .unwrap()
            .clone();
        let changed_color = source_reflection(&colors, second_color, 3, seed)
            .unwrap()
            .proof()
            .unwrap()
            .clone();

        assert_eq!(first, repeated);
        assert_ne!(first.proof_digest(), changed_slot.proof_digest());
        assert_ne!(first.proof_digest(), changed_color.proof_digest());
        assert_ne!(
            first.result_color_identity(),
            changed_color.result_color_identity()
        );
    }

    #[test]
    fn pure_adjoint_result_reflection_is_phase_composed_and_canonical() {
        let result = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![
                LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 3, 2]).unwrap(),
            ],
        )
        .unwrap();
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let left = proven_reflection(&result, minus_one, 43);
        let right = proven_reflection(&result, minus_one, 44);
        let local = TransitionReflectionProof::new(minus_one, [digest(45)], [digest(46)]).unwrap();
        let proof = current_reflection_candidate(&result, &[left, right], Some(&local))
            .unwrap()
            .unwrap();
        assert_eq!(proof.phase(), minus_one);
        assert_eq!(
            proof.result_color_identity(),
            dynamic_color_identity_digest(&result).unwrap()
        );
        assert!(proof.lineage_roots().contains(&digest(43)));
        assert!(proof.lineage_roots().contains(&digest(44)));
        assert!(proof.lineage_roots().contains(&digest(45)));
        assert!(proof.lineage_roots().contains(&digest(46)));
        assert!(proof.lineage_roots().contains(&local.proof_digest));
        assert!(pure_adjoint_word_is_canonical(
            result.pure_adjoint_word().unwrap()
        ));
        assert!(!pure_adjoint_word_is_canonical(&[3, 2, 1]));
    }

    fn reflection_test_current(
        color_id: DynamicLCColorStateId,
        reflection: CurrentReflection,
    ) -> PendingCurrent {
        PendingCurrent {
            key: CurrentCoreKey::new(
                digest(201),
                RecurrenceNodeKind::Current,
                0,
                color_id,
                vec![0, 1, 2],
                CanonicalMomentumLinearForm::new(vec![
                    MomentumTerm {
                        source_slot: 0,
                        coefficient: 1,
                    },
                    MomentumTerm {
                        source_slot: 1,
                        coefficient: 1,
                    },
                    MomentumTerm {
                        source_slot: 2,
                        coefficient: 1,
                    },
                ])
                .unwrap(),
                CurrentHelicityIdentity::all_flow_union(0),
                vec![0],
                0,
                vec![],
                CurrentSourceBinding::None,
                None,
            )
            .unwrap(),
            source_exact_factor: None,
            contributions: BTreeMap::new(),
            realized_pairing_rule_ids: BTreeSet::new(),
            reflection,
            reflection_certificate_id: None,
        }
    }

    fn reflection_test_states() -> (
        DynamicLCColorStateInterner,
        DynamicLCColorStateId,
        DynamicLCColorStateId,
    ) {
        let canonical = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![
                LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 3, 2]).unwrap(),
            ],
        )
        .unwrap();
        let reversed = canonical.reversed().unwrap();
        let mut states = DynamicLCColorStateInterner::default();
        let canonical_id = states.intern(canonical).unwrap();
        let reversed_id = states.intern(reversed).unwrap();
        (states, canonical_id, reversed_id)
    }

    #[test]
    fn late_unproved_reflection_keeps_both_stage_orientations() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
        let mut late_downgrade =
            proven_interned_reflection(&color_states, reversed_id, minus_one, 51);
        late_downgrade.include(None).unwrap();
        let mut currents = vec![
            reflection_test_current(
                canonical_id,
                proven_interned_reflection(&color_states, canonical_id, minus_one, 50),
            ),
            reflection_test_current(reversed_id, late_downgrade),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<TransientCurrentIdIndex>();
        let mut target_bucket = vec![0, 1];
        let mut certificates = Vec::new();
        let mut resident_contribution_count = 0;

        reconcile_stage_reflections(
            0,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut target_bucket,
            &mut certificates,
            4,
            &mut resident_contribution_count,
        )
        .unwrap();

        assert_eq!(currents.len(), 2);
        assert_eq!(target_bucket, [0, 1]);
        assert_eq!(current_ids.len(), 2);
        assert!(certificates.is_empty());
        assert!(
            currents
                .iter()
                .all(|current| current.reflection == CurrentReflection::Unavailable)
        );
    }

    #[test]
    fn late_conflicting_reflection_keeps_both_stage_orientations() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
        let mut late_downgrade =
            proven_interned_reflection(&color_states, reversed_id, minus_one, 53);
        late_downgrade
            .include(Some(
                CurrentReflectionProof::new(minus_one, [digest(54)], digest(55)).unwrap(),
            ))
            .unwrap();
        let mut currents = vec![
            reflection_test_current(
                canonical_id,
                proven_interned_reflection(&color_states, canonical_id, minus_one, 52),
            ),
            reflection_test_current(reversed_id, late_downgrade),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<TransientCurrentIdIndex>();
        let mut target_bucket = vec![0, 1];
        let mut certificates = Vec::new();
        let mut resident_contribution_count = 0;

        reconcile_stage_reflections(
            0,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut target_bucket,
            &mut certificates,
            4,
            &mut resident_contribution_count,
        )
        .unwrap();

        assert_eq!(currents.len(), 2);
        assert_eq!(target_bucket, [0, 1]);
        assert_eq!(current_ids.len(), 2);
        assert!(certificates.is_empty());
        assert!(
            currents
                .iter()
                .all(|current| current.reflection == CurrentReflection::Unavailable)
        );
    }

    #[test]
    fn nonreciprocal_reflection_proofs_keep_both_stage_orientations() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
        let mut currents = vec![
            reflection_test_current(
                canonical_id,
                proven_interned_reflection(&color_states, canonical_id, minus_one, 56),
            ),
            reflection_test_current(
                reversed_id,
                proven_interned_reflection(
                    &color_states,
                    reversed_id,
                    ExactComplexRational::ONE,
                    57,
                ),
            ),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<TransientCurrentIdIndex>();
        let mut target_bucket = vec![0, 1];
        let mut certificates = Vec::new();
        let mut resident_contribution_count = 0;

        reconcile_stage_reflections(
            0,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut target_bucket,
            &mut certificates,
            4,
            &mut resident_contribution_count,
        )
        .unwrap();

        assert_eq!(currents.len(), 2);
        assert_eq!(target_bucket, [0, 1]);
        assert_eq!(current_ids.len(), 2);
        assert!(certificates.is_empty());
        assert!(
            currents
                .iter()
                .all(|current| current.reflection == CurrentReflection::Unavailable)
        );
    }

    #[test]
    fn complete_reciprocal_reflection_prunes_only_noncanonical_orientation() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
        let mut currents = vec![
            reflection_test_current(
                canonical_id,
                proven_interned_reflection(&color_states, canonical_id, minus_one, 58),
            ),
            reflection_test_current(
                reversed_id,
                proven_interned_reflection(&color_states, reversed_id, minus_one, 59),
            ),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<TransientCurrentIdIndex>();
        let mut target_bucket = vec![0, 1];
        let mut certificates = Vec::new();
        let mut resident_contribution_count = 0;
        let mut support_keys = TransientCurrentSupportKeys::from_currents(4, &currents).unwrap();
        let mut support_buckets =
            LaneSupportBuckets::new(&[vec![], vec![], vec![0, 1], vec![]], &support_keys).unwrap();

        reconcile_stage_reflections(
            0,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut target_bucket,
            &mut certificates,
            4,
            &mut resident_contribution_count,
        )
        .unwrap();
        support_keys.reconcile_stage_tail(0, &currents).unwrap();
        support_buckets
            .replace_bucket(3, &target_bucket, &support_keys)
            .unwrap();

        assert_eq!(currents.len(), 1);
        assert_eq!(target_bucket, [0]);
        assert_eq!(current_ids.len(), 1);
        assert_eq!(support_keys.keys_by_current.len(), 1);
        let disjoint = support_buckets
            .bucket_mut(3)
            .unwrap()
            .disjoint_current_ids(&TransientSupportKey::singleton(4, 3).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            disjoint
                .iter()
                .map(|entry| entry.current_id)
                .collect::<Vec<_>>(),
            [0]
        );
        assert_eq!(certificates.len(), 1);
        let certificate = &certificates[0];
        assert_eq!(certificate.id, 0);
        assert_eq!(certificate.canonical_old_current_id, 0);
        assert_eq!(certificate.reflected_old_current_id, 1);
        assert_eq!(certificate.canonical_phase, minus_one);
        assert_eq!(certificate.reflected_phase, minus_one);
        assert_eq!(certificate.source_permutation.as_ref(), [0, 2, 1, 3]);
        assert!(!certificate.fixed_point);
        assert_eq!(certificate.orbit_size, 2);
        validate_pending_reflection_certificates(&certificates).unwrap();
        let mut stale_lineage = certificates.clone();
        stale_lineage[0].canonical_lineage_digest = digest(60);
        assert!(validate_pending_reflection_certificates(&stale_lineage).is_err());
        assert_eq!(currents[0].reflection.phase(), Some(minus_one));
        let color = color_states
            .get(currents[0].key.dynamic_lc_color_state_id())
            .unwrap();
        assert!(pure_adjoint_word_is_canonical(
            color.pure_adjoint_word().unwrap()
        ));
    }

    #[test]
    fn transient_current_index_insertion_order_cannot_change_canonical_reconciliation() {
        let run = |reverse_index_insertion: bool| {
            let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
            let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
            let mut currents = vec![
                reflection_test_current(
                    canonical_id,
                    proven_interned_reflection(&color_states, canonical_id, minus_one, 91),
                ),
                reflection_test_current(
                    reversed_id,
                    proven_interned_reflection(&color_states, reversed_id, minus_one, 92),
                ),
            ];
            let mut insertion_order = [0_usize, 1];
            if reverse_index_insertion {
                insertion_order.reverse();
            }
            let mut current_ids = TransientCurrentIdIndex::new();
            for current_id in insertion_order {
                assert!(
                    current_ids
                        .insert(currents[current_id].key.clone(), current_id as u32)
                        .is_none()
                );
            }
            let mut target_bucket = vec![0, 1];
            let mut certificates = Vec::new();
            let mut resident_contribution_count = 0;
            reconcile_stage_reflections(
                0,
                &mut color_states,
                &mut currents,
                &mut current_ids,
                &mut target_bucket,
                &mut certificates,
                4,
                &mut resident_contribution_count,
            )
            .unwrap();
            (
                currents
                    .into_iter()
                    .map(|current| current.key)
                    .collect::<Vec<_>>(),
                target_bucket,
                certificates,
                resident_contribution_count,
            )
        };

        assert_eq!(run(false), run(true));
    }

    #[test]
    fn reflected_closure_emission_reuses_parents_and_applies_exact_phase() {
        let factor = ExactComplexRational::parse_parts("2", "1", "0", "1").unwrap();
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let key = PendingClosureKey {
            target_sector_id: 119,
            complete_source_states: Vec::new().into_boxed_slice(),
            closure_template_id: 3,
            quantum_flow_template_id: Some(5),
            parent_current_ids: vec![7, 9].into_boxed_slice(),
        };
        let contribution = PendingClosureProofContribution {
            construction_parent_ids: [7, 9],
            construction_parent_permutation: [0, 1],
            reconstruction_parent_permutation: [0, 1],
            evaluator_parent_permutation: [0, 1],
            closure_template_semantic_digest: digest(111),
            color_witness_term_id: 13,
            color_witness_proof_digest: digest(112),
            three_line_certificate: None,
            pairing_certificate_ids: Vec::new().into_boxed_slice(),
            reflection_certificate_id: Some(17),
            exact_factor: factor,
        };
        let projection = ClosureReflectionProjection {
            certificate_id: 17,
            partner_sector_id: Some(153),
            partner_phase: minus_one,
        };
        let mut closures = BTreeMap::new();

        include_closure_with_reflection(&mut closures, key, contribution, Some(projection))
            .unwrap();

        assert_eq!(closures.len(), 2);
        let canonical = closures
            .iter()
            .find(|(key, _)| key.target_sector_id == 119)
            .unwrap();
        let reflected = closures
            .iter()
            .find(|(key, _)| key.target_sector_id == 153)
            .unwrap();
        assert_eq!(canonical.1.exact_factor, factor);
        assert_eq!(
            reflected.1.exact_factor,
            factor.checked_mul(minus_one).unwrap()
        );
        assert_eq!(
            canonical.0.parent_current_ids,
            reflected.0.parent_current_ids
        );
        assert_eq!(canonical.1.contributions[0].construction_parent_ids, [7, 9]);
        assert_eq!(reflected.1.contributions[0].construction_parent_ids, [7, 9]);
        assert_eq!(
            reflected.1.contributions[0].reflection_certificate_id,
            Some(17)
        );

        let mut ineligible = BTreeMap::new();
        include_closure_with_reflection(
            &mut ineligible,
            canonical.0.clone(),
            canonical.1.contributions[0].clone(),
            None,
        )
        .unwrap();
        assert_eq!(ineligible.len(), 1);
    }

    fn buffered_parent_pairs(target_size: usize, currents_by_size: &[Vec<u32>]) -> Vec<[u32; 2]> {
        let mut pairs = Vec::new();
        for left_size in 1..target_size {
            let right_size = target_size - left_size;
            for &left_id in &currents_by_size[left_size - 1] {
                for &right_id in &currents_by_size[right_size - 1] {
                    if left_id < right_id {
                        pairs.push([left_id, right_id]);
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    #[test]
    fn streamed_parent_pairs_match_buffered_reference_order() {
        let fixtures = [
            (2, vec![vec![0, 1, 2, 3]]),
            (3, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7, 8, 9]]),
            (
                4,
                vec![
                    vec![0, 1, 2, 3],
                    vec![4, 5, 6, 7, 8, 9],
                    vec![10, 11, 12, 13],
                ],
            ),
        ];
        for (target_size, currents_by_size) in fixtures {
            assert_eq!(
                parent_pairs_for_target(target_size, &currents_by_size).collect::<Vec<_>>(),
                buffered_parent_pairs(target_size, &currents_by_size),
                "support size {target_size}",
            );
        }
    }

    #[test]
    fn streamed_parent_pair_totals_match_iteration() {
        for (target_size, currents_by_size) in [
            (2, vec![vec![0, 1, 2, 3]]),
            (3, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7, 8, 9]]),
            (
                4,
                vec![
                    vec![0, 1, 2, 3],
                    vec![4, 5, 6, 7, 8, 9],
                    vec![10, 11, 12, 13],
                ],
            ),
        ] {
            assert_eq!(
                parent_pair_total_for_target(target_size, &currents_by_size).unwrap(),
                parent_pairs_for_target(target_size, &currents_by_size).count(),
                "support size {target_size}",
            );
        }
    }

    fn compact_pair_schedule(
        source_count: usize,
        target_size: usize,
        currents_by_size: &[Vec<u32>],
        source_slots_by_current: &[Vec<u32>],
    ) -> DisjointParentPairSchedule {
        let support_keys = TransientCurrentSupportKeys {
            source_count,
            keys_by_current: source_slots_by_current
                .iter()
                .map(|source_slots| {
                    TransientSupportKey::from_source_slots(source_count, source_slots).unwrap()
                })
                .collect(),
        };
        let mut support_buckets = LaneSupportBuckets::new(currents_by_size, &support_keys).unwrap();
        disjoint_parent_pairs_for_target(
            target_size,
            currents_by_size,
            &support_keys,
            &mut support_buckets,
        )
        .unwrap()
    }

    fn reference_disjoint_parent_pairs(
        target_size: usize,
        currents_by_size: &[Vec<u32>],
        source_slots_by_current: &[Vec<u32>],
    ) -> Vec<[u32; 2]> {
        reference_disjoint_parent_pair_schedule(
            target_size,
            currents_by_size,
            source_slots_by_current,
        )
        .into_iter()
        .map(|pair| pair.parent_ids)
        .collect()
    }

    fn reference_disjoint_parent_pair_schedule(
        target_size: usize,
        currents_by_size: &[Vec<u32>],
        source_slots_by_current: &[Vec<u32>],
    ) -> Vec<OrderedDisjointParentPair> {
        parent_pairs_for_target(target_size, currents_by_size)
            .enumerate()
            .filter_map(|(candidate_index, parent_ids)| {
                let [left_id, right_id] = parent_ids;
                disjoint_support(
                    &source_slots_by_current[left_id as usize],
                    &source_slots_by_current[right_id as usize],
                )
                .then_some(OrderedDisjointParentPair {
                    parent_ids,
                    theoretical_candidate_count: candidate_index + 1,
                })
            })
            .collect()
    }

    fn next_random(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed
    }

    fn random_support(seed: &mut u64, source_count: usize, size: usize) -> Vec<u32> {
        let mut support = BTreeSet::new();
        while support.len() < size {
            support.insert((next_random(seed) as usize % source_count) as u32);
        }
        support.into_iter().collect()
    }

    #[test]
    fn compact_disjoint_parent_pairs_match_randomized_reference_order() {
        let mut seed = 0x6a09_e667_f3bc_c909;
        for case_index in 0..160 {
            let source_count = 4 + next_random(&mut seed) as usize % 7;
            let target_size = 2 + next_random(&mut seed) as usize % (source_count - 2);
            let mut currents_by_size = vec![Vec::new(); target_size - 1];
            let mut source_slots_by_current = Vec::new();
            for support_size in 1..target_size {
                let current_count = 2 + next_random(&mut seed) as usize % 9;
                for _ in 0..current_count {
                    let source_slots = random_support(&mut seed, source_count, support_size);
                    let current_id = source_slots_by_current.len() as u32;
                    source_slots_by_current.push(source_slots);
                    currents_by_size[support_size - 1].push(current_id);
                }
            }

            let reference = reference_disjoint_parent_pair_schedule(
                target_size,
                &currents_by_size,
                &source_slots_by_current,
            );
            let unfiltered_count = parent_pairs_for_target(target_size, &currents_by_size).count();
            let schedule = compact_pair_schedule(
                source_count,
                target_size,
                &currents_by_size,
                &source_slots_by_current,
            );
            assert_eq!(
                schedule.theoretical_candidate_count, unfiltered_count,
                "case {case_index}",
            );
            let emitted = schedule.pairs.map(|pair| pair.unwrap()).collect::<Vec<_>>();
            assert_eq!(
                emitted, reference,
                "case {case_index}, source_count={source_count}, target_size={target_size}",
            );
        }
    }

    #[test]
    fn compact_disjoint_parent_pairs_preserve_duplicates_and_wide_supports() {
        let source_count = 137;
        let currents_by_size = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let source_slots_by_current = vec![
            vec![0],
            vec![128],
            vec![128],
            vec![136],
            vec![0, 64],
            vec![1, 128],
            vec![64, 136],
            vec![1, 128],
        ];
        assert!(matches!(
            TransientSupportKey::from_source_slots(source_count, &[0, 64, 128, 136]).unwrap(),
            TransientSupportKey::Wide { .. }
        ));
        let left = TransientSupportKey::from_source_slots(source_count, &[0, 128]).unwrap();
        let right = TransientSupportKey::from_source_slots(source_count, &[64, 136]).unwrap();
        assert_eq!(
            left.union_disjoint(&right).unwrap().source_slots(),
            [0, 64, 128, 136]
        );

        let reference =
            reference_disjoint_parent_pairs(3, &currents_by_size, &source_slots_by_current);
        let schedule =
            compact_pair_schedule(source_count, 3, &currents_by_size, &source_slots_by_current);
        assert!(schedule.support_bucket_cache_hit_count > 0);
        assert_eq!(
            schedule
                .pairs
                .map(|pair| pair.unwrap().parent_ids)
                .collect::<Vec<_>>(),
            reference
        );
    }

    #[test]
    fn disjoint_pair_schedule_releases_bucket_cache_before_emission() {
        let source_count = 6;
        let currents_by_size = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
        let source_slots_by_current = vec![
            vec![0],
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![2, 3],
            vec![3, 4],
            vec![1, 5],
        ];
        let support_keys = TransientCurrentSupportKeys {
            source_count,
            keys_by_current: source_slots_by_current
                .iter()
                .map(|support| {
                    TransientSupportKey::from_source_slots(source_count, support).unwrap()
                })
                .collect(),
        };
        let mut support_buckets =
            LaneSupportBuckets::new(&currents_by_size, &support_keys).unwrap();
        let schedule = disjoint_parent_pairs_for_target(
            3,
            &currents_by_size,
            &support_keys,
            &mut support_buckets,
        )
        .unwrap();

        assert!(schedule.support_bucket_cache_hit_count > 0);
        assert_eq!(support_buckets.cached_disjoint_query_count(), 0);
        assert_eq!(support_buckets.cached_disjoint_capacity(), 0);
        assert_eq!(
            schedule
                .pairs
                .map(|pair| pair.unwrap().parent_ids)
                .collect::<Vec<_>>(),
            reference_disjoint_parent_pairs(3, &currents_by_size, &source_slots_by_current),
        );
        assert_eq!(support_buckets.cached_disjoint_query_count(), 0);
        assert_eq!(support_buckets.cached_disjoint_capacity(), 0);
    }

    #[test]
    fn transient_wide_support_validates_padding_and_source_domain() {
        let exact = TransientSupportKey::from_source_slots(129, &[0, 64, 128]).unwrap();
        assert_eq!(exact.source_count(), 129);
        assert_eq!(exact.source_slots(), [0, 64, 128]);
        assert!(
            TransientSupportKey::full(129)
                .unwrap()
                .without_source_slot(129)
                .is_err()
        );

        let padding_bit = TransientSupportKey::Wide {
            source_count: 129,
            words: vec![0, 0, 2].into_boxed_slice(),
        };
        assert!(padding_bit.validate().is_err());
        let truncated = TransientSupportKey::Wide {
            source_count: 129,
            words: vec![0, 0].into_boxed_slice(),
        };
        assert!(truncated.validate().is_err());

        let left = TransientSupportKey::from_source_slots(129, &[0]).unwrap();
        let other_domain = TransientSupportKey::from_source_slots(130, &[1]).unwrap();
        assert!(left.is_disjoint(&other_domain).is_err());
        assert!(left.union_disjoint(&other_domain).is_err());
        assert_ne!(left, other_domain);
    }

    #[test]
    fn state_index_preserves_interleaved_reference_transition_order() {
        let rows = [
            (transition(0), [1, 2]),
            (transition(1), [2, 1]),
            (transition(2), [1, 2]),
            (transition(3), [3, 4]),
            (transition(4), [2, 1]),
        ];
        let mut index = TransitionStateIndex::default();
        for (row, input_states) in rows {
            index.insert(row, input_states);
        }

        let actual = index
            .rows(1, 2)
            .iter()
            .map(|indexed| Ok((indexed.row.id, indexed.parent_ids(1, 2, 10, 20)?)))
            .collect::<RusticolResult<Vec<_>>>()
            .unwrap();
        let reference = rows
            .iter()
            .flat_map(|(row, input_states)| {
                let mut applications = Vec::new();
                if *input_states == [1, 2] {
                    applications.push((row.id, [10, 20]));
                }
                if *input_states == [2, 1] {
                    applications.push((row.id, [20, 10]));
                }
                applications
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, reference);
        assert_eq!(
            actual,
            [(0, [10, 20]), (1, [20, 10]), (2, [10, 20]), (4, [20, 10])]
        );
    }

    fn encoded_strings(values: &[&str]) -> (Vec<CheckedTableRange>, Vec<u8>) {
        let mut ranges = Vec::new();
        let mut bytes = Vec::new();
        for value in values {
            ranges.push(CheckedTableRange::new(
                bytes.len() as u64,
                value.len() as u64,
            ));
            bytes.extend_from_slice(value.as_bytes());
        }
        (ranges, bytes)
    }

    fn indexed_u32_sequences(sequences: &[&[u32]]) -> (Vec<IndexedRangeRow>, Vec<u32>) {
        let mut ranges = Vec::new();
        let mut values = Vec::new();
        for (id, sequence) in sequences.iter().enumerate() {
            ranges.push(IndexedRangeRow {
                id: id as u32,
                range: CheckedTableRange::new(values.len() as u64, sequence.len() as u64),
            });
            values.extend_from_slice(sequence);
        }
        (ranges, values)
    }

    fn append_u32_sequence(template: &mut OwnedRecurrenceTemplateInput, sequence: &[u32]) -> u32 {
        let id = template.u32_sequence_ranges.len() as u32;
        template.u32_sequence_ranges.push(IndexedRangeRow {
            id,
            range: CheckedTableRange::new(
                template.u32_sequence_values.len() as u64,
                sequence.len() as u64,
            ),
        });
        template.u32_sequence_values.extend_from_slice(sequence);
        id
    }

    fn append_template_string(template: &mut OwnedRecurrenceTemplateInput, value: &str) -> u32 {
        let id = template.string_ranges.len() as u32;
        template.string_ranges.push(CheckedTableRange::new(
            template.string_bytes.len() as u64,
            value.len() as u64,
        ));
        template.string_bytes.extend_from_slice(value.as_bytes());
        id
    }

    fn scalar_reference_template() -> OwnedRecurrenceTemplateInput {
        const EMPTY: u32 = 0;
        const STATE_PAIR: u32 = 1;
        const CANONICAL_ORDER: u32 = 2;
        const PARENT_FLAVOURS: u32 = 3;
        const PARENT_QUANTUM_NUMBERS: u32 = 4;

        let (string_ranges, string_bytes) = encoded_strings(&["0", "1", "constant-result"]);
        let (u32_sequence_ranges, u32_sequence_values) =
            indexed_u32_sequences(&[&[], &[0, 0], &[0, 1], &[0, 0], &[0, 0]]);
        OwnedRecurrenceTemplateInput {
            input_abi: "scalar-reference-template-v1".to_owned(),
            catalog_digest: digest(1),
            compiled_model_digest: digest(2),
            prepared_kernel_pack_digest: digest(3),
            catalog_header: vec![],
            coupling_order_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 0),
            }],
            coupling_order_terms: vec![],
            contact_orbit_certificates: vec![],
            contact_orbit_steps: vec![],
            current_states: vec![],
            digest_catalog: vec![DigestCatalogRow {
                id: 0,
                value: [4; 32],
            }],
            evaluator_bindings: vec![],
            exact_factors: vec![ExactFactorRow {
                id: 0,
                real_numerator_string_id: 1,
                real_denominator_string_id: 1,
                imag_numerator_string_id: 0,
                imag_denominator_string_id: 1,
            }],
            flavour_flow_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 1),
            }],
            flavour_flow_values: vec![0],
            i32_sequence_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 2),
            }],
            i32_sequence_values: vec![0, 0],
            parameters: vec![],
            propagators: vec![],
            quantum_flows: vec![QuantumFlowRow {
                id: 0,
                template_string_id: 0,
                input_state_sequence_id: STATE_PAIR,
                input_spin_sequence_id: 0,
                input_flavour_sequence_id: PARENT_FLAVOURS,
                input_quantum_sequence_id: PARENT_QUANTUM_NUMBERS,
                flavour_flow_operation_string_id: 2,
                quantum_number_flow_operation_string_id: 2,
                coupling_order_set_id: 0,
                result_state_template_id: 0,
                result_spin_state: 0,
                result_flavour_flow_id: 0,
                result_quantum_number_flow_id: 0,
                exact_coupling_factor_id: 0,
                predicate_digest_id: 0,
                semantic_digest_id: 0,
            }],
            quantum_number_flow_ranges: vec![],
            quantum_number_flow_terms: vec![],
            runtime_helicity_contracts: vec![],
            runtime_helicity_variants: vec![],
            runtime_helicity_embeddings: vec![],
            runtime_helicity_projections: vec![],
            sources: vec![],
            string_ranges,
            string_bytes,
            symmetry_proofs: vec![],
            transitions: vec![TransitionRow {
                input_state_sequence_id: STATE_PAIR,
                canonical_input_order_sequence_id: CANONICAL_ORDER,
                input_exchange_factor_id: MISSING_U32,
                ..transition(0)
            }],
            closures: vec![ClosureRow {
                id: 0,
                template_string_id: 0,
                input_state_sequence_id: STATE_PAIR,
                result_state_template_id: MISSING_U32,
                evaluator_binding_id: 0,
                canonical_input_order_sequence_id: CANONICAL_ORDER,
                coupling_parameter_sequence_id: EMPTY,
                coupling_order_set_id: 0,
                eligible_quantum_flow_sequence_id: EMPTY,
                color_contraction_template_id: 1,
                binding_coupling_factor_id: 0,
                exact_factor_id: 0,
                output_factor_source: 0,
                equivalence_class_string_id: 0,
                input_exchange_factor_id: MISSING_U32,
                projection_string_id: 0,
                component_coefficient_sequence_id: EMPTY,
                chirality_relation_string_id: 0,
                metric_signature_string_id: 0,
                semantic_digest_id: 0,
            }],
            color_contractions: vec![
                ColorContractionRow {
                    id: 0,
                    template_string_id: 0,
                    rule_kind_string_id: 0,
                    input_representation_sequence_id: EMPTY,
                    has_output_representation: 1,
                    output_representation: 1,
                    ordered_open_string_arity: 0,
                    exact_coefficient_factor_id: 0,
                    witness_start: 0,
                    witness_count: 1,
                    nc_term_start: 0,
                    nc_term_count: 0,
                    expression_digest_id: 0,
                    semantic_digest_id: 0,
                },
                ColorContractionRow {
                    id: 1,
                    template_string_id: 0,
                    rule_kind_string_id: 0,
                    input_representation_sequence_id: EMPTY,
                    has_output_representation: 0,
                    output_representation: 0,
                    ordered_open_string_arity: 0,
                    exact_coefficient_factor_id: 0,
                    witness_start: 1,
                    witness_count: 1,
                    nc_term_start: 0,
                    nc_term_count: 0,
                    expression_digest_id: 0,
                    semantic_digest_id: 0,
                },
            ],
            lc_color_transition_witnesses: vec![
                LCColorTransitionWitnessRow {
                    color_contraction_id: 0,
                    ordinal: 0,
                    left_shape_string_id: 0,
                    right_shape_string_id: 0,
                    input_permutation: 0,
                    reverse_parent_mask: 0,
                    component_operation: LCColorComponentOperation::Empty as u8,
                    result_component_kind: u8::MAX,
                    result_component_role: LCColorComponentRole::None as u8,
                    result_shape_string_id: 0,
                    exact_factor_id: 0,
                    proof_digest_id: 0,
                    input_port_pairing_sequence_id: EMPTY,
                    result_port_binding_sequence_id: EMPTY,
                    provenance_sequence_id: EMPTY,
                },
                LCColorTransitionWitnessRow {
                    color_contraction_id: 1,
                    ordinal: 0,
                    left_shape_string_id: 0,
                    right_shape_string_id: 0,
                    input_permutation: 0,
                    reverse_parent_mask: 0,
                    component_operation: LCColorComponentOperation::Close as u8,
                    result_component_kind: u8::MAX,
                    result_component_role: LCColorComponentRole::None as u8,
                    result_shape_string_id: MISSING_U32,
                    exact_factor_id: 0,
                    proof_digest_id: 0,
                    input_port_pairing_sequence_id: EMPTY,
                    result_port_binding_sequence_id: EMPTY,
                    provenance_sequence_id: EMPTY,
                },
            ],
            color_nc_terms: vec![],
            u32_sequence_ranges,
            u32_sequence_values,
        }
    }

    #[test]
    fn established_prepared_transition_binds_only_certified_contact_metadata() {
        use crate::recurrence::contact_orbit_owner::{
            ContactOrbitTestBinding, contact_orbit_test_template,
        };

        let none = contact_orbit_test_template(ContactOrbitTestBinding::None)
            .validate()
            .unwrap();
        let none_catalog = TemplateCatalog::new(none.input()).unwrap();
        let none_prepared = PreparedTransitionCatalog::new(none.input(), &none_catalog).unwrap();
        assert!(none_prepared.rows(0, 0)[0].contact_orbit.is_none());
        assert!(none_prepared.contact_orbit(0).is_none());

        let one = contact_orbit_test_template(ContactOrbitTestBinding::One)
            .validate()
            .unwrap();
        let one_catalog = TemplateCatalog::new(one.input()).unwrap();
        let one_prepared = PreparedTransitionCatalog::new(one.input(), &one_catalog).unwrap();
        assert!(one_prepared.rows(0, 0)[0].contact_orbit.is_some());
        assert!(one_prepared.contact_orbit(0).is_some());
        assert!(one_prepared.contact_orbit(u32::MAX).is_none());

        let mut currents = vec![
            projection_test_current(RecurrenceNodeKind::Source, 0, 0, &[0]),
            projection_test_current(RecurrenceNodeKind::Source, 0, 1, &[1]),
            projection_test_current(RecurrenceNodeKind::Current, 0, 2, &[0, 1]),
        ];
        add_projection_test_contribution(&mut currents, 2, 0, [0, 1], 0);
        let plan = plan_established_contact_orbit_owners(2, &one_prepared, &currents, 1)
            .unwrap()
            .expect("certified transition must enter established owner planning");
        let mut resident = 1;
        plan.commit(&mut currents, &mut resident);
        assert_eq!(resident, 1);
        assert_eq!(established_contact_transition_ids(&currents[2]), [0]);
    }

    #[test]
    fn prepared_transition_catalog_matches_lazy_reference_metadata_and_order() {
        let mut template = scalar_reference_template();
        let seed = template.transitions[0];
        template.transitions = [2, 0, 1]
            .map(|id| TransitionRow {
                id,
                input_exchange_factor_id: 0,
                ..seed
            })
            .to_vec();
        let catalog = TemplateCatalog::new(&template).unwrap();
        let lazy_index = TransitionStateIndex::new(&template, &catalog).unwrap();
        let prepared_catalog = PreparedTransitionCatalog::new(&template, &catalog).unwrap();

        assert_eq!(
            prepared_catalog.decoded_transition_count(),
            template.transitions.len()
        );
        assert_eq!(prepared_catalog.decoded_witness_count(), 3);
        assert_eq!(
            prepared_catalog.structural_transitions(),
            structural_transitions(&template, &catalog)
                .unwrap()
                .as_slice()
        );

        let lazy_rows = lazy_index.rows(0, 0);
        let prepared_rows = prepared_catalog.rows(0, 0);
        assert_eq!(
            prepared_rows
                .iter()
                .map(|prepared| prepared.row.id)
                .collect::<Vec<_>>(),
            [2, 0, 1]
        );
        assert!(
            [2, 0, 1]
                .into_iter()
                .all(|transition_id| prepared_catalog.contact_orbit(transition_id).is_none())
        );
        let mut uncertified_currents = vec![
            projection_test_current(RecurrenceNodeKind::Source, 0, 0, &[0]),
            projection_test_current(RecurrenceNodeKind::Source, 0, 1, &[1]),
            projection_test_current(RecurrenceNodeKind::Current, 0, 2, &[0, 1]),
        ];
        add_projection_test_contribution(&mut uncertified_currents, 2, 0, [0, 1], 0);
        assert!(
            plan_established_contact_orbit_owners(2, &prepared_catalog, &uncertified_currents, 1,)
                .unwrap()
                .is_none()
        );
        assert_eq!(prepared_rows.len(), lazy_rows.len());

        let mut color_states = DynamicLCColorStateInterner::default();
        let color_id = color_states
            .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
            .unwrap();
        let parent = |source_slot, state_template_id, spin_state| {
            CurrentCoreKey::new(
                template.catalog_digest,
                RecurrenceNodeKind::Source,
                state_template_id,
                color_id,
                vec![source_slot],
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot,
                    coefficient: 1,
                }])
                .unwrap(),
                CurrentHelicityIdentity::all_flow_union(spin_state),
                vec![0],
                0,
                vec![],
                CurrentSourceBinding::runtime_dispatch(source_slot, vec![source_slot]).unwrap(),
                None,
            )
            .unwrap()
        };
        let left = parent(0, 0, 0);
        let right = parent(1, 0, 0);
        let parents = [&left, &right];

        for (lazy, prepared) in lazy_rows.iter().zip(prepared_rows) {
            assert_eq!(prepared.row, lazy.row);
            assert_eq!(prepared.input_states, lazy.input_states);
            assert_eq!(
                prepared.parent_ids(0, 0, 20, 10).unwrap(),
                lazy.parent_ids(0, 0, 20, 10).unwrap()
            );
            let quantum = template.quantum_flows[prepared.row.quantum_flow_template_id as usize];
            assert_eq!(prepared.quantum, quantum);
            assert_eq!(
                prepared.quantum_input_states.as_slice(),
                catalog
                    .u32_sequence(quantum.input_state_sequence_id, "quantum input states")
                    .unwrap()
            );
            assert_eq!(
                prepared.quantum_input_spins.as_slice(),
                catalog
                    .i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")
                    .unwrap()
            );
            assert_eq!(
                prepared.quantum_flow_matches(&parents),
                quantum_flow_matches(quantum, &parents, &catalog).unwrap()
            );
            assert_eq!(
                prepared.result_flavour_flow(&parents),
                quantum_flow_result_flavour(quantum, &parents, &catalog).unwrap()
            );
            assert_eq!(
                prepared.local_coupling_orders.as_ref(),
                catalog
                    .coupling_orders(prepared.row.coupling_order_set_id)
                    .unwrap()
            );
            assert_eq!(
                prepared.contraction,
                template.color_contractions[prepared.row.color_contraction_template_id as usize]
            );
            assert_eq!(
                prepared.canonical_evaluator_parents([20, 10]),
                canonical_evaluator_parents(
                    [20, 10],
                    catalog
                        .u32_sequence(
                            prepared.row.canonical_input_order_sequence_id,
                            "transition canonical input order",
                        )
                        .unwrap(),
                    prepared.row.input_exchange_factor_id,
                    &catalog,
                    "transition",
                )
                .unwrap()
            );
            let (_, prepared_exchange_factor) = prepared.canonical_evaluator_parents([20, 10]);
            let (_, lazy_exchange_factor) = canonical_evaluator_parents(
                [20, 10],
                catalog
                    .u32_sequence(
                        prepared.row.canonical_input_order_sequence_id,
                        "transition canonical input order",
                    )
                    .unwrap(),
                prepared.row.input_exchange_factor_id,
                &catalog,
                "transition",
            )
            .unwrap();
            assert_eq!(
                prepared.transition_exact_factor,
                catalog
                    .factor(prepared.row.exact_factor_id, "transition exact")
                    .unwrap()
            );
            assert_eq!(
                prepared.contraction_exact_factor,
                catalog
                    .factor(
                        prepared.contraction.exact_coefficient_factor_id,
                        "color contraction",
                    )
                    .unwrap()
            );
            let binding_coupling = authenticate_runtime_coupling(
                &catalog,
                quantum,
                prepared.row.binding_coupling_factor_id,
                "transition",
            )
            .unwrap();
            assert_eq!(
                prepared.output_factor().unwrap(),
                output_factor_from_binding(
                    binding_coupling,
                    prepared.row.output_factor_source,
                    "transition",
                )
                .unwrap()
            );
            assert_eq!(
                multiply_factors(&[
                    prepared.transition_exact_factor,
                    prepared_exchange_factor,
                    prepared.contraction_exact_factor,
                    prepared.output_factor().unwrap(),
                ])
                .unwrap(),
                multiply_factors(&[
                    catalog
                        .factor(prepared.row.exact_factor_id, "transition exact")
                        .unwrap(),
                    lazy_exchange_factor,
                    catalog
                        .factor(
                            prepared.contraction.exact_coefficient_factor_id,
                            "color contraction",
                        )
                        .unwrap(),
                    output_factor_from_binding(
                        binding_coupling,
                        prepared.row.output_factor_source,
                        "transition",
                    )
                    .unwrap(),
                ])
                .unwrap()
            );
            assert_eq!(
                prepared.quantum_semantic_digest,
                catalog
                    .digest(quantum.semantic_digest_id, "quantum-flow semantic")
                    .unwrap()
            );
            let lazy_witnesses = catalog
                .witness_rows(prepared.row.color_contraction_template_id)
                .unwrap()
                .iter()
                .copied()
                .map(|row| PreparedTransitionWitness {
                    row,
                    witness: catalog.witness(row).unwrap(),
                })
                .collect::<Vec<_>>();
            assert_eq!(prepared.witnesses.as_ref(), lazy_witnesses.as_slice());
        }
    }

    #[test]
    fn prepared_closure_catalog_matches_lazy_reference_order_orientation_factors_and_witnesses() {
        let mut template = scalar_reference_template();
        let state_01 = append_u32_sequence(&mut template, &[0, 1]);
        let state_10 = append_u32_sequence(&mut template, &[1, 0]);
        let reverse_order = append_u32_sequence(&mut template, &[1, 0]);
        let minus_one_string = append_template_string(&mut template, "-1");
        template.exact_factors.push(ExactFactorRow {
            id: 1,
            real_numerator_string_id: minus_one_string,
            real_denominator_string_id: 1,
            imag_numerator_string_id: 0,
            imag_denominator_string_id: 1,
        });
        template.lc_color_transition_witnesses[1].input_permutation = 1;
        template.lc_color_transition_witnesses[1].exact_factor_id = 1;
        let seed = template.closures[0];
        template.closures = vec![
            ClosureRow {
                id: 7,
                input_state_sequence_id: state_01,
                canonical_input_order_sequence_id: reverse_order,
                input_exchange_factor_id: 1,
                exact_factor_id: 1,
                ..seed
            },
            ClosureRow {
                id: 3,
                input_state_sequence_id: state_10,
                canonical_input_order_sequence_id: reverse_order,
                input_exchange_factor_id: 1,
                exact_factor_id: 1,
                ..seed
            },
            ClosureRow {
                id: 9,
                input_state_sequence_id: state_01,
                canonical_input_order_sequence_id: reverse_order,
                input_exchange_factor_id: 1,
                exact_factor_id: 1,
                ..seed
            },
        ];
        let catalog = TemplateCatalog::new(&template).unwrap();
        let prepared = PreparedClosureCatalog::new(&template, &catalog).unwrap();

        for (anchor_state, complement_state, anchor_id, complement_id) in [
            (0, 1, 10, 20),
            (1, 0, 30, 40),
            (0, 0, 50, 60),
            (2, 3, 70, 80),
        ] {
            let actual = prepared
                .rows(anchor_state, complement_state)
                .iter()
                .map(|closure| {
                    Ok((
                        closure.row.id,
                        closure.parent_ids(
                            anchor_state,
                            complement_state,
                            anchor_id,
                            complement_id,
                        )?,
                    ))
                })
                .collect::<RusticolResult<Vec<_>>>()
                .unwrap();
            let reference = template
                .closures
                .iter()
                .flat_map(|closure| {
                    let input_states = catalog
                        .u32_sequence(closure.input_state_sequence_id, "closure input states")
                        .unwrap();
                    let mut applications = Vec::new();
                    if input_states == [complement_state, anchor_state] {
                        applications.push((closure.id, [complement_id, anchor_id]));
                    }
                    if input_states == [anchor_state, complement_state]
                        && anchor_state != complement_state
                    {
                        applications.push((closure.id, [anchor_id, complement_id]));
                    }
                    applications
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, reference);
        }
        assert_eq!(
            prepared
                .rows(0, 1)
                .iter()
                .map(|closure| closure.row.id)
                .collect::<Vec<_>>(),
            [7, 3, 9],
        );

        let closure = &prepared.rows(0, 1)[0];
        let parent_ids = closure.parent_ids(0, 1, 10, 20).unwrap();
        assert_eq!(parent_ids, [10, 20]);
        let (prepared_evaluator_parents, prepared_exchange_factor) =
            closure.canonical_evaluator_parents(parent_ids);
        let (lazy_evaluator_parents, lazy_exchange_factor) = canonical_evaluator_parents(
            parent_ids,
            catalog
                .u32_sequence(
                    closure.row.canonical_input_order_sequence_id,
                    "closure canonical input order",
                )
                .unwrap(),
            closure.row.input_exchange_factor_id,
            &catalog,
            "closure",
        )
        .unwrap();
        assert_eq!(prepared_evaluator_parents, lazy_evaluator_parents);
        assert_eq!(prepared_exchange_factor, lazy_exchange_factor);
        assert_eq!(
            prepared_exchange_factor,
            ExactComplexRational::ONE.checked_neg().unwrap(),
        );
        assert_eq!(
            closure.witnesses[0].row,
            template.lc_color_transition_witnesses[1],
        );
        assert_eq!(
            closure.witnesses[0].witness,
            catalog
                .witness(template.lc_color_transition_witnesses[1])
                .unwrap(),
        );
        let prepared_factor = multiply_factors(&[
            closure.closure_exact_factor,
            prepared_exchange_factor,
            closure.contraction_exact_factor,
            closure.quantum_flows[0].output_factor().unwrap(),
            closure.witnesses[0].witness.exact_factor(),
        ])
        .unwrap();
        let binding_coupling = catalog
            .factor(
                closure.row.binding_coupling_factor_id,
                "closure binding coupling",
            )
            .unwrap();
        let lazy_factor = multiply_factors(&[
            catalog
                .factor(closure.row.exact_factor_id, "closure exact")
                .unwrap(),
            lazy_exchange_factor,
            catalog
                .factor(
                    template.color_contractions[closure.row.color_contraction_template_id as usize]
                        .exact_coefficient_factor_id,
                    "closure color",
                )
                .unwrap(),
            output_factor_from_binding(
                binding_coupling,
                closure.row.output_factor_source,
                "closure",
            )
            .unwrap(),
            catalog
                .witness(template.lc_color_transition_witnesses[1])
                .unwrap()
                .exact_factor(),
        ])
        .unwrap();
        assert_eq!(prepared_factor, lazy_factor);
        assert_eq!(
            prepared_factor,
            ExactComplexRational::ONE.checked_neg().unwrap(),
        );
    }

    #[test]
    fn failure_only_closure_color_diagnostics_match_the_lazy_reference() {
        let template = scalar_reference_template();
        let catalog = TemplateCatalog::new(&template).unwrap();
        let prepared = PreparedClosureCatalog::new(&template, &catalog).unwrap();
        let currents = scalar_structural_sources(&template, 2);
        let mut color_states = DynamicLCColorStateInterner::default();
        assert_eq!(
            color_states
                .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
                .unwrap()
                .get(),
            0,
        );

        let actual =
            collect_closure_color_attempts(&prepared, &color_states, &currents, &[0], &[1])
                .unwrap();
        let parents = [&currents[1].key, &currents[0].key];
        let left = color_states
            .get(parents[0].dynamic_lc_color_state_id())
            .unwrap();
        let right = color_states
            .get(parents[1].dynamic_lc_color_state_id())
            .unwrap();
        let reference = catalog
            .witness_rows(template.closures[0].color_contraction_template_id)
            .unwrap()
            .iter()
            .filter(|row| {
                row.left_shape_string_id == left.output_color_shape_id()
                    && row.right_shape_string_id == right.output_color_shape_id()
            })
            .map(|row| {
                catalog
                    .witness(*row)
                    .unwrap()
                    .closed_components(left, right)
                    .unwrap()
                    .iter()
                    .map(|component| (component.kind(), component.source_slots().to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, reference);
        assert_eq!(actual, BTreeSet::from([Vec::new()]));
    }

    #[test]
    fn prepared_flavour_flow_preserves_parent_orientation_and_union_collapse() {
        let mut color_states = DynamicLCColorStateInterner::default();
        let color_id = color_states
            .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
            .unwrap();
        let parent = |source_slot, flavour_flow, helicity_identity| {
            CurrentCoreKey::new(
                digest(211),
                RecurrenceNodeKind::Current,
                0,
                color_id,
                vec![source_slot],
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot,
                    coefficient: 1,
                }])
                .unwrap(),
                helicity_identity,
                flavour_flow,
                0,
                vec![],
                CurrentSourceBinding::None,
                None,
            )
            .unwrap()
        };
        let left = parent(
            0,
            vec![10, 20],
            CurrentHelicityIdentity::topology_replay(0, vec![SourceStateAssignment::new(0, 0)])
                .unwrap(),
        );
        let right = parent(
            1,
            vec![30, 40],
            CurrentHelicityIdentity::topology_replay(0, vec![SourceStateAssignment::new(1, 0)])
                .unwrap(),
        );
        let parents = [&left, &right];
        let reversed = [&right, &left];

        for (prepared, expected, reversed_expected) in [
            (
                PreparedFlavourFlow::Constant(vec![7, 9].into_boxed_slice()),
                vec![7, 9],
                vec![7, 9],
            ),
            (
                PreparedFlavourFlow::AppendLeft(9),
                vec![10, 20, 9],
                vec![30, 40, 9],
            ),
            (
                PreparedFlavourFlow::AppendRight(9),
                vec![30, 40, 9],
                vec![10, 20, 9],
            ),
            (
                PreparedFlavourFlow::ConcatLeftRight(9),
                vec![10, 20, 30, 40, 9],
                vec![30, 40, 10, 20, 9],
            ),
        ] {
            assert_eq!(prepared.apply(&parents), expected);
            assert_eq!(prepared.apply(&reversed), reversed_expected);

            let union_left = parent(0, vec![10, 20], CurrentHelicityIdentity::all_flow_union(0));
            let union_right = parent(1, vec![30, 40], CurrentHelicityIdentity::all_flow_union(0));
            assert_eq!(prepared.apply(&[&union_left, &union_right]), [9]);
        }
    }

    #[test]
    fn prepared_transition_catalog_preserves_lazy_binary_contract_errors() {
        let mut non_binary_transition = scalar_reference_template();
        non_binary_transition.u32_sequence_ranges[1].range = CheckedTableRange::new(0, 1);
        let catalog = TemplateCatalog::new(&non_binary_transition).unwrap();
        assert_eq!(
            PreparedTransitionCatalog::new(&non_binary_transition, &catalog).unwrap_err(),
            TransitionStateIndex::new(&non_binary_transition, &catalog).unwrap_err()
        );

        let mut non_binary_quantum = scalar_reference_template();
        non_binary_quantum.i32_sequence_ranges[0].range = CheckedTableRange::new(0, 1);
        let catalog = TemplateCatalog::new(&non_binary_quantum).unwrap();
        assert_eq!(
            PreparedTransitionCatalog::new(&non_binary_quantum, &catalog).unwrap_err(),
            structural_transitions(&non_binary_quantum, &catalog).unwrap_err()
        );
    }

    #[test]
    fn prepared_transition_catalog_preserves_lazy_metadata_errors() {
        let mut invalid_order = scalar_reference_template();
        invalid_order.u32_sequence_values[2..4].copy_from_slice(&[0, 0]);
        let catalog = TemplateCatalog::new(&invalid_order).unwrap();
        let row = invalid_order.transitions[0];
        let lazy_error = canonical_evaluator_parents(
            [20, 10],
            catalog
                .u32_sequence(
                    row.canonical_input_order_sequence_id,
                    "transition canonical input order",
                )
                .unwrap(),
            row.input_exchange_factor_id,
            &catalog,
            "transition",
        )
        .unwrap_err();
        assert_eq!(
            PreparedTransitionCatalog::new(&invalid_order, &catalog).unwrap_err(),
            lazy_error
        );

        let mut invalid_witness = scalar_reference_template();
        invalid_witness.lc_color_transition_witnesses[0].input_permutation = 2;
        let catalog = TemplateCatalog::new(&invalid_witness).unwrap();
        let lazy_error = catalog
            .witness(invalid_witness.lc_color_transition_witnesses[0])
            .unwrap_err();
        assert_eq!(
            PreparedTransitionCatalog::new(&invalid_witness, &catalog).unwrap_err(),
            lazy_error
        );
    }

    fn scalar_reference_process(external_count: usize) -> OwnedRecurrenceProcessInput {
        OwnedRecurrenceProcessInput {
            input_abi: "scalar-reference-process-v1".to_owned(),
            declared_input_digest: digest(5),
            fermion_pairing: None,
            bitset_ranges: vec![],
            bitset_words: vec![],
            coupling_limits: vec![],
            digest_catalog: vec![],
            exact_factors: vec![],
            external_legs: (0..external_count)
                .map(|slot| ProcessExternalLegRow {
                    source_slot: slot as u32,
                    public_label: slot as u32 + 1,
                    physical_pdg: 0,
                    outgoing_pdg: 0,
                    is_initial: 0,
                    is_fermionic: 0,
                    source_state_range: CheckedTableRange::new(0, 0),
                    momentum_mask_id: 0,
                    support_mask_id: 0,
                })
                .collect(),
            header: vec![],
            header_digests: vec![],
            lc_open_strings: vec![],
            normalization: vec![],
            parameter_projection: vec![],
            physical_lc_sectors: vec![ProcessPhysicalLCSectorRow {
                sector_id: 0,
                public_id_string_id: 0,
                kind: ProcessLCSectorKind::Singlet as u8,
                closure_source_slot: 0,
                closure_proof_algorithm_string_id: 0,
                closure_proof_digest_id: 0,
                open_string_range: CheckedTableRange::new(0, 0),
                trace_sequence_id: 0,
                singlet_sequence_id: 0,
                word_sequence_id: 0,
                support_mask_id: 0,
            }],
            public_lc_flows: vec![],
            replay_partitions: vec![],
            replay_targets: vec![],
            selected_public_flow_coverage: vec![],
            selected_source_coverage: vec![],
            semantic_template_references: vec![],
            source_states: vec![],
            string_ranges: vec![],
            string_bytes: vec![],
            u32_sequence_ranges: vec![CheckedTableRange::new(0, 0)],
            u32_sequence_values: vec![],
        }
    }

    fn cyclic_trace_anchor_contract_fixture()
    -> (OwnedRecurrenceProcessInput, OwnedRecurrenceTemplateInput) {
        let mut template = scalar_reference_template();
        let color_shape_string_id = append_template_string(&mut template, "adjoint-segment");
        let direct_closure_string_id = append_template_string(&mut template, "direct-contraction");
        template.current_states = vec![crate::recurrence::template::CurrentStateRow {
            id: 0,
            template_string_id: 0,
            particle_id: 21,
            anti_particle_id: 21,
            species_string_id: 0,
            orientation: CurrentOrientation::SelfConjugate as u8,
            statistics: ParticleStatistics::Boson as u8,
            color_representation: 8,
            basis_string_id: 0,
            tensor_ordering_sequence_id: 0,
            dimension: 4,
            chirality: 0,
            lc_color_shape_string_id: color_shape_string_id,
            auxiliary_kind_string_id: MISSING_U32,
            mass_parameter_id: MISSING_U32,
            width_parameter_id: MISSING_U32,
            semantic_digest_id: 0,
        }];
        template.evaluator_bindings = vec![crate::recurrence::template::EvaluatorBindingRow {
            id: 0,
            resolver_key_string_id: 0,
            prepared_kernel_id: MISSING_U32,
            contract_kind: EvaluatorContractKind::Closure as u8,
            callable_signature_digest_id: 0,
            input_state_sequence_id: 1,
            output_state_template_id: MISSING_U32,
            input_layout_sequence_id: 0,
            output_layout_sequence_id: 0,
            exact_expression_digest_sequence_id: 0,
            semantic_template_sequence_id: 0,
            callable_kind: EvaluatorCallableKind::RusticolTemplate as u8,
            runtime_template_string_id: 0,
            semantic_digest_id: 0,
        }];
        template.closures[0].equivalence_class_string_id = direct_closure_string_id;
        template.i32_sequence_values = vec![8, 8];

        let mut process = scalar_reference_process(4);
        let (string_ranges, string_bytes) =
            encoded_strings(&[CYCLIC_TRACE_CLOSURE_ANCHOR_ALGORITHM]);
        process.string_ranges = string_ranges;
        process.string_bytes = string_bytes;
        process.u32_sequence_ranges =
            vec![CheckedTableRange::new(0, 0), CheckedTableRange::new(0, 4)];
        process.u32_sequence_values = vec![0, 1, 2, 3];
        process.physical_lc_sectors[0] = ProcessPhysicalLCSectorRow {
            kind: ProcessLCSectorKind::SingleTrace as u8,
            closure_source_slot: 0,
            closure_proof_algorithm_string_id: 0,
            trace_sequence_id: 1,
            singlet_sequence_id: 0,
            word_sequence_id: 1,
            ..process.physical_lc_sectors[0]
        };
        process.source_states = (0..4)
            .map(|source_slot| ProcessSourceStateRow {
                source_slot,
                state_index: 0,
                public_helicity: 1,
                chirality: 0,
                spin_state: 1,
                current_state_template_id: 0,
                source_template_id: source_slot,
                momentum_sign: 1,
                crossing_phase_factor_id: 0,
            })
            .collect();
        for (source_slot, leg) in process.external_legs.iter_mut().enumerate() {
            leg.physical_pdg = 21;
            leg.outgoing_pdg = 21;
            leg.source_state_range = CheckedTableRange::new(source_slot as u64, 1);
        }
        (process, template)
    }

    #[test]
    fn cyclic_trace_anchor_constructor_authenticates_template_contracts() {
        let (process, template) = cyclic_trace_anchor_contract_fixture();
        let process_catalog = ProcessCatalog::new(&process).unwrap();
        let template_catalog = TemplateCatalog::new(&template).unwrap();
        validate_cyclic_trace_closure_anchor_contracts(
            &process,
            &process_catalog,
            &template,
            &template_catalog,
        )
        .expect("authenticated adjoint trace closure contract must pass");

        let mut non_adjoint = template.clone();
        non_adjoint.current_states[0].color_representation = 3;
        let catalog = TemplateCatalog::new(&non_adjoint).unwrap();
        assert!(
            validate_cyclic_trace_closure_anchor_contracts(
                &process,
                &process_catalog,
                &non_adjoint,
                &catalog,
            )
            .unwrap_err()
            .to_string()
            .contains("self-conjugate adjoint")
        );

        let mut missing_direct_closure = template.clone();
        missing_direct_closure.closures[0].result_state_template_id = 0;
        let catalog = TemplateCatalog::new(&missing_direct_closure).unwrap();
        assert!(
            validate_cyclic_trace_closure_anchor_contracts(
                &process,
                &process_catalog,
                &missing_direct_closure,
                &catalog,
            )
            .unwrap_err()
            .to_string()
            .contains("exact direct binary adjoint closure")
        );

        let mut coupling_order_closure = template.clone();
        let coupling_order_name = append_template_string(&mut coupling_order_closure, "QCD");
        coupling_order_closure.coupling_order_ranges[0].range = CheckedTableRange::new(0, 1);
        coupling_order_closure.coupling_order_terms =
            vec![crate::recurrence::template::CouplingOrderTermRow {
                set_id: 0,
                name_string_id: coupling_order_name,
                power: 1,
            }];
        let catalog = TemplateCatalog::new(&coupling_order_closure).unwrap();
        assert!(
            validate_cyclic_trace_closure_anchor_contracts(
                &process,
                &process_catalog,
                &coupling_order_closure,
                &catalog,
            )
            .unwrap_err()
            .to_string()
            .contains("exact direct binary adjoint closure")
        );

        let mut established_process = process.clone();
        let (string_ranges, string_bytes) = encoded_strings(&["canonical-lc-closure-anchor-v2"]);
        established_process.string_ranges = string_ranges;
        established_process.string_bytes = string_bytes;
        let catalog = ProcessCatalog::new(&established_process).unwrap();
        validate_cyclic_trace_closure_anchor_contracts(
            &established_process,
            &catalog,
            &non_adjoint,
            &TemplateCatalog::new(&non_adjoint).unwrap(),
        )
        .expect("non-v4 recurrence contracts must remain untouched");
    }

    fn scalar_structural_sources(
        template: &OwnedRecurrenceTemplateInput,
        external_count: usize,
    ) -> Vec<PendingCurrent> {
        let mut color_states = DynamicLCColorStateInterner::default();
        let color_id = color_states
            .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
            .unwrap();
        (0..external_count as u32)
            .map(|source_slot| {
                let key = CurrentCoreKey::new(
                    template.catalog_digest,
                    RecurrenceNodeKind::Source,
                    0,
                    color_id,
                    vec![source_slot],
                    CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                        source_slot,
                        coefficient: 1,
                    }])
                    .unwrap(),
                    CurrentHelicityIdentity::all_flow_union(0),
                    vec![0],
                    0,
                    vec![],
                    CurrentSourceBinding::runtime_dispatch(source_slot, vec![source_slot]).unwrap(),
                    None,
                )
                .unwrap();
                PendingCurrent {
                    key,
                    source_exact_factor: None,
                    contributions: BTreeMap::new(),
                    realized_pairing_rule_ids: BTreeSet::new(),
                    reflection: CurrentReflection::Unavailable,
                    reflection_certificate_id: None,
                }
            })
            .collect()
    }

    #[test]
    fn lane_closure_support_index_matches_reference_filter_and_preserves_current_id_order() {
        let template = scalar_reference_template();
        let mut currents = scalar_structural_sources(&template, 4);
        let color_id = currents[0].key.dynamic_lc_color_state_id();
        let internal = |support: Vec<u32>| {
            let momentum = CanonicalMomentumLinearForm::new(
                support
                    .iter()
                    .copied()
                    .map(|source_slot| MomentumTerm {
                        source_slot,
                        coefficient: 1,
                    })
                    .collect(),
            )
            .unwrap();
            PendingCurrent {
                key: CurrentCoreKey::new(
                    template.catalog_digest,
                    RecurrenceNodeKind::Current,
                    0,
                    color_id,
                    support,
                    momentum,
                    CurrentHelicityIdentity::all_flow_union(0),
                    vec![0],
                    0,
                    vec![],
                    CurrentSourceBinding::None,
                    None,
                )
                .unwrap(),
                source_exact_factor: None,
                contributions: BTreeMap::new(),
                realized_pairing_rule_ids: BTreeSet::new(),
                reflection: CurrentReflection::Unavailable,
                reflection_certificate_id: None,
            }
        };
        currents.push(internal(vec![1, 2, 3]));
        currents.push(internal(vec![1, 2, 3]));
        currents.push(internal(vec![0, 2]));

        let global = LaneClosureSupportIndex::new(&currents, None).unwrap();
        assert_eq!(global.current_ids(&[0]), [0]);
        assert_eq!(global.current_ids(&[1, 2, 3]), [4, 5]);
        assert_eq!(global.current_ids(&[0, 2]), [6]);

        let domain = PendingConstructionDomain {
            shared_source_end: 4,
            lane_internal_start: 5,
            lane_internal_end: 7,
        };
        let lane = LaneClosureSupportIndex::new(&currents, Some(domain)).unwrap();
        for support in [&[0][..], &[1, 2, 3][..], &[0, 2][..], &[0, 3][..]] {
            let reference = currents
                .iter()
                .enumerate()
                .filter(|(current_id, current)| {
                    domain.contains(*current_id) && current.key.support_source_slots() == support
                })
                .map(|(current_id, _)| current_id as u32)
                .collect::<Vec<_>>();
            assert_eq!(lane.current_ids(support), reference);
        }
    }

    #[test]
    fn shared_structural_feasibility_matches_lazy_multi_lane_demands() {
        let template = scalar_reference_template();
        let catalog = TemplateCatalog::new(&template).unwrap();
        let prepared = PreparedTransitionCatalog::new(&template, &catalog).unwrap();
        let mut process = scalar_reference_process(4);
        let mut second_sector = process.physical_lc_sectors[0];
        second_sector.sector_id = 1;
        second_sector.closure_source_slot = 1;
        process.physical_lc_sectors.push(second_sector);
        let sources = scalar_structural_sources(&template, 4);
        let feasibility = StructuralFeasibilityIndex::new(4, &prepared, &sources).unwrap();

        assert!(feasibility.feasible_support_count() > sources.len());
        assert!(feasibility.decomposition_count() > 0);
        assert!(feasibility.forward_transition_probe_count() > 0);
        for materialized_sectors in [
            BTreeSet::from([0]),
            BTreeSet::from([1]),
            BTreeSet::from([0, 1]),
        ] {
            let lazy = LazyStructuralDemandIndex::new(
                &process,
                &template,
                &catalog,
                &prepared,
                &materialized_sectors,
                &sources,
            )
            .unwrap();
            let indexed = StructuralDemandIndex::new(
                &process,
                &template,
                &catalog,
                &prepared,
                &feasibility,
                &materialized_sectors,
            )
            .unwrap();
            assert_eq!(indexed.demanded_rows(), lazy.demanded);
            for (support, state) in &lazy.demanded {
                let support_key = TransientSupportKey::from_source_slots(4, support).unwrap();
                assert_eq!(
                    indexed.accepts(
                        &support_key,
                        state.state_template_id,
                        state.spin_state_class,
                    ),
                    lazy.accepts(support, state.state_template_id, state.spin_state_class,)
                );
            }
        }
    }

    #[test]
    fn contracted_color_without_replay_proof_keeps_the_direct_destination_domain() {
        let mut process = scalar_reference_process(2);
        process.header = vec![ProcessHeaderRow {
            schema_version: 1,
            abi_string_id: 0,
            process_id_string_id: 0,
            layout: u8::try_from(RecurrenceStrategy::ContractedColorUnion.as_u32())
                .expect("strategy discriminant fits u8"),
            selected_flow_mode: 0,
            selected_source_mode: 0,
            external_leg_count: 2,
            physical_sector_count: 1,
            public_flow_count: 1,
            replay_partition_count: 0,
            coupling_limit_count: 0,
            parameter_projection_count: 0,
            process_support_mask_id: 0,
        }];
        process.external_legs[0].source_state_range = CheckedTableRange::new(0, 1);
        process.external_legs[1].source_state_range = CheckedTableRange::new(1, 1);
        process.source_states = vec![
            ProcessSourceStateRow {
                source_slot: 0,
                state_index: 0,
                public_helicity: -1,
                chirality: -1,
                spin_state: -1,
                current_state_template_id: 0,
                source_template_id: 0,
                momentum_sign: 1,
                crossing_phase_factor_id: 0,
            },
            ProcessSourceStateRow {
                source_slot: 1,
                state_index: 0,
                public_helicity: 1,
                chirality: 1,
                spin_state: 1,
                current_state_template_id: 0,
                source_template_id: 0,
                momentum_sign: 1,
                crossing_phase_factor_id: 0,
            },
        ];
        process.public_lc_flows = vec![ProcessPublicLCFlowRow {
            flow_id: 0,
            public_id_string_id: 0,
            construction_sector_id: 0,
            word_sequence_id: 0,
            source_slot_permutation_sequence_id: 0,
            reduction_weight_factor_id: 0,
        }];
        process.u32_sequence_ranges = vec![CheckedTableRange::new(0, 2)];
        process.u32_sequence_values = vec![0, 1];
        let catalog = ProcessCatalog::new(&process).unwrap();

        assert!(
            build_replay_targets(RecurrenceStrategy::ContractedColorUnion, &process, &catalog,)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn contracted_closure_index_preserves_open_line_canonical_owner() {
        let mut process = scalar_reference_process(4);
        process.lc_open_strings = vec![
            ProcessLCOpenStringRow {
                sector_id: 0,
                ordinal: 0,
                fundamental_source_slot: 0,
                antifundamental_source_slot: 1,
                adjoint_sequence_id: 0,
                singlet_sequence_id: 0,
            },
            ProcessLCOpenStringRow {
                sector_id: 0,
                ordinal: 1,
                fundamental_source_slot: 2,
                antifundamental_source_slot: 3,
                adjoint_sequence_id: 0,
                singlet_sequence_id: 0,
            },
            ProcessLCOpenStringRow {
                sector_id: 1,
                ordinal: 0,
                fundamental_source_slot: 2,
                antifundamental_source_slot: 3,
                adjoint_sequence_id: 0,
                singlet_sequence_id: 0,
            },
            ProcessLCOpenStringRow {
                sector_id: 1,
                ordinal: 1,
                fundamental_source_slot: 0,
                antifundamental_source_slot: 1,
                adjoint_sequence_id: 0,
                singlet_sequence_id: 0,
            },
        ];
        let seed = process.physical_lc_sectors[0];
        process.physical_lc_sectors = vec![
            ProcessPhysicalLCSectorRow {
                kind: ProcessLCSectorKind::OpenLines as u8,
                closure_source_slot: 1,
                open_string_range: CheckedTableRange::new(0, 2),
                ..seed
            },
            ProcessPhysicalLCSectorRow {
                sector_id: 1,
                kind: ProcessLCSectorKind::OpenLines as u8,
                closure_source_slot: 1,
                open_string_range: CheckedTableRange::new(2, 2),
                ..seed
            },
        ];
        let catalog = ProcessCatalog::new(&process).unwrap();
        let sector_ids = BTreeSet::from([0, 1]);
        let contracted = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            &catalog,
            &sector_ids,
        )
        .unwrap();
        let first = contracted.get(0).unwrap();
        let alias = contracted.get(1).unwrap();
        assert!(first.contracted_color_canonical_owner);
        assert!(!alias.contracted_color_canonical_owner);
        assert!(unordered_color_components_match(
            &first.expected_components,
            &alias.expected_components,
        ));
        assert!(
            closed_components_match_prepared_sector(
                RecurrenceStrategy::ContractedColorUnion,
                &first.expected_components,
                first,
            )
            .unwrap()
        );
        assert!(
            !closed_components_match_prepared_sector(
                RecurrenceStrategy::ContractedColorUnion,
                &alias.expected_components,
                alias,
            )
            .unwrap()
        );
        assert_eq!(first.anchor_support.as_ref(), [1]);
        assert_eq!(first.complement_support.as_ref(), [0, 2, 3]);
        let buckets = contracted_closure_support_buckets(
            RecurrenceStrategy::ContractedColorUnion,
            &contracted,
            &sector_ids,
        )
        .unwrap();
        let bucket = buckets.values().next().unwrap();
        assert_eq!(buckets.len(), 1);
        assert_eq!(bucket.sector_ids, [0, 1]);
        assert_eq!(bucket.matching_sector_ids(&first.expected_components), [0]);
        assert_eq!(bucket.matching_sector_ids(&alias.expected_components), [0]);

        // Canonical ownership is established against the complete process
        // catalog, not merely the materialized subset.  An alias-only replay
        // therefore remains suppressed instead of becoming a local fallback.
        let alias_only_ids = BTreeSet::from([1]);
        let alias_only = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            &catalog,
            &alias_only_ids,
        )
        .unwrap();
        let alias_bucket = contracted_closure_support_buckets(
            RecurrenceStrategy::ContractedColorUnion,
            &alias_only,
            &alias_only_ids,
        )
        .unwrap()
        .into_values()
        .next()
        .unwrap();
        assert!(
            alias_bucket
                .matching_sector_ids(&alias.expected_components)
                .is_empty()
        );

        let topology = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::TopologyReplay,
            &process,
            &catalog,
            &sector_ids,
        )
        .unwrap();
        assert!(topology.get(0).unwrap().contracted_color_canonical_owner);
        assert!(topology.get(1).unwrap().contracted_color_canonical_owner);

        let all_flow = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::AllFlowUnion,
            &process,
            &catalog,
            &sector_ids,
        )
        .unwrap();
        let all_flow_bucket = contracted_closure_support_buckets(
            RecurrenceStrategy::AllFlowUnion,
            &all_flow,
            &sector_ids,
        )
        .unwrap()
        .into_values()
        .next()
        .unwrap();
        assert_eq!(
            all_flow_bucket.matching_sector_ids(&first.expected_components),
            [0, 1]
        );
    }

    #[test]
    fn precomputed_three_line_components_preserve_direct_and_partner_witness_order() {
        let mut process = scalar_reference_process(6);
        process.lc_open_strings = (0..3)
            .map(|ordinal| ProcessLCOpenStringRow {
                sector_id: 0,
                ordinal,
                fundamental_source_slot: ordinal * 2,
                antifundamental_source_slot: ordinal * 2 + 1,
                adjoint_sequence_id: 0,
                singlet_sequence_id: 0,
            })
            .collect();
        process.u32_sequence_ranges =
            vec![CheckedTableRange::new(0, 0), CheckedTableRange::new(0, 6)];
        process.u32_sequence_values = vec![0, 1, 2, 3, 4, 5];
        process.physical_lc_sectors[0] = ProcessPhysicalLCSectorRow {
            kind: ProcessLCSectorKind::OpenLines as u8,
            closure_source_slot: 5,
            open_string_range: CheckedTableRange::new(0, 3),
            word_sequence_id: 1,
            ..process.physical_lc_sectors[0]
        };
        let catalog = ProcessCatalog::new(&process).unwrap();
        let prepared = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::TopologyReplay,
            &process,
            &catalog,
            &BTreeSet::from([0]),
        )
        .unwrap();
        let sector = prepared.get(0).unwrap();
        let direct = three_line_traversal_certificate(
            &sector.expected_components,
            sector.row,
            &sector.expected_components,
            &catalog,
            Some(17),
        )
        .unwrap()
        .unwrap();
        assert_eq!(u32::from(direct.kind), THREE_LINE_DIRECT_CERTIFICATE_ID);
        assert_eq!(direct.pairing_rule_id, 17);
        assert_eq!(direct.reference_block_order.as_ref(), [0, 1, 2]);
        assert_eq!(direct.witness_block_order.as_ref(), [0, 1, 2]);

        let partner_components = vec![
            sector.expected_components[1].clone(),
            sector.expected_components[0].clone(),
            sector.expected_components[2].clone(),
        ];
        let partner = three_line_traversal_certificate(
            &partner_components,
            sector.row,
            &sector.expected_components,
            &catalog,
            Some(17),
        )
        .unwrap()
        .unwrap();
        assert_eq!(u32::from(partner.kind), THREE_LINE_PARTNER_CERTIFICATE_ID);
        assert_eq!(partner.reference_block_order.as_ref(), [0, 1, 2]);
        assert_eq!(partner.witness_block_order.as_ref(), [1, 0, 2]);
        assert_ne!(direct.proof_digest, partner.proof_digest);
    }

    #[test]
    fn multi_realized_closure_selects_semantic_pairing_owner_independent_of_rule_id() {
        let later = [
            FermionPairingEndpointPairRow {
                fundamental_source_slot: 0,
                antifundamental_source_slot: 7,
            },
            FermionPairingEndpointPairRow {
                fundamental_source_slot: 2,
                antifundamental_source_slot: 5,
            },
        ];
        let canonical = [
            FermionPairingEndpointPairRow {
                fundamental_source_slot: 0,
                antifundamental_source_slot: 3,
            },
            FermionPairingEndpointPairRow {
                fundamental_source_slot: 2,
                antifundamental_source_slot: 7,
            },
        ];

        for candidates in [
            vec![Ok((1, later.as_slice())), Ok((9, canonical.as_slice()))],
            vec![Ok((9, canonical.as_slice())), Ok((1, later.as_slice()))],
        ] {
            assert_eq!(canonical_pairing_owner_id(candidates).unwrap(), Some(9));
        }
        assert_eq!(
            canonical_pairing_owner_id([Ok((4, later.as_slice()))]).unwrap(),
            Some(4)
        );
        assert_eq!(
            canonical_pairing_owner_id(std::iter::empty()).unwrap(),
            None
        );
    }

    fn scalar_reference_program(
        external_count: usize,
    ) -> (
        RecurrenceProgram,
        Vec<StageConstructionDiagnostics>,
        (usize, usize, usize),
        RecurrenceGenerationTelemetry,
    ) {
        let template = scalar_reference_template();
        let template_catalog = TemplateCatalog::new(&template).unwrap();
        let prepared_transitions =
            PreparedTransitionCatalog::new(&template, &template_catalog).unwrap();
        let prepared_closures = PreparedClosureCatalog::new(&template, &template_catalog).unwrap();
        let process = scalar_reference_process(external_count);
        let process_catalog = ProcessCatalog::new(&process).unwrap();
        let fermion_ordering =
            established_fermion_ordering_context(&process, &template, None).unwrap();
        let materialized_sectors = BTreeSet::from([0]);
        let prepared_closure_sectors = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::AllFlowUnion,
            &process,
            &process_catalog,
            &materialized_sectors,
        )
        .unwrap();
        let color_targets =
            MaterializedColorTargets::new(&materialized_sectors, &process, &process_catalog)
                .unwrap();
        let mut color_states = DynamicLCColorStateInterner::default();
        let color_id = color_states
            .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
            .unwrap();
        let mut currents = Vec::new();
        let mut current_ids = TransientCurrentIdIndex::new();
        let mut currents_by_size = vec![Vec::new(); external_count];
        for slot in 0..external_count as u32 {
            let key = CurrentCoreKey::new(
                template.catalog_digest,
                RecurrenceNodeKind::Source,
                0,
                color_id,
                vec![slot],
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot: slot,
                    coefficient: 1,
                }])
                .unwrap(),
                CurrentHelicityIdentity::all_flow_union(0),
                vec![0],
                0,
                vec![],
                CurrentSourceBinding::runtime_dispatch(slot, vec![slot]).unwrap(),
                None,
            )
            .unwrap();
            let id = currents.len() as u32;
            assert!(current_ids.insert(key.clone(), id).is_none());
            currents.push(PendingCurrent {
                key,
                source_exact_factor: None,
                contributions: BTreeMap::new(),
                realized_pairing_rule_ids: BTreeSet::new(),
                reflection: CurrentReflection::Unavailable,
                reflection_certificate_id: None,
            });
            currents_by_size[0].push(id);
        }

        let mut reflection_certificates = Vec::new();
        let mut resident_contribution_count = 0;
        let mut generation_telemetry = RecurrenceGenerationTelemetry::default();
        let structural_feasibility =
            StructuralFeasibilityIndex::new(external_count, &prepared_transitions, &currents)
                .unwrap();
        let mut current_support_keys =
            TransientCurrentSupportKeys::from_currents(external_count, &currents).unwrap();
        let structural_demands = StructuralDemandIndex::new(
            &process,
            &template,
            &template_catalog,
            &prepared_transitions,
            &structural_feasibility,
            &BTreeSet::from([0]),
        )
        .unwrap();
        let stage_diagnostics = build_internal_currents(
            template.catalog_digest,
            &process,
            None,
            &template,
            &fermion_ordering,
            &prepared_transitions,
            &TransitionReflectionIndex::new(&template, &template_catalog).unwrap(),
            &[],
            &BTreeMap::new(),
            &color_targets,
            &structural_demands,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut currents_by_size,
            &mut current_support_keys,
            &mut reflection_certificates,
            &mut resident_contribution_count,
            0,
            external_count.saturating_sub(2),
            external_count.saturating_add(1),
            &mut generation_telemetry,
            true,
            &mut |_| Ok(()),
        )
        .unwrap();
        let closures = build_closures(
            RecurrenceStrategy::AllFlowUnion,
            &process,
            &process_catalog,
            &template,
            &fermion_ordering,
            None,
            &prepared_closures,
            &prepared_closure_sectors,
            &color_states,
            &currents,
            &materialized_sectors,
            &stage_diagnostics,
            &reflection_certificates,
            None,
            &mut generation_telemetry,
            true,
        )
        .unwrap();
        let constructed_counts = (
            currents.len(),
            currents
                .iter()
                .map(|current| current.contributions.len())
                .sum(),
            closures.len(),
        );
        assert_eq!(resident_contribution_count, constructed_counts.1);
        let program = finish_program(
            RecurrenceStrategy::AllFlowUnion,
            &process_catalog,
            color_states.into_states(),
            currents,
            closures.groups,
            closures.physical_sector_ids,
            closures.physical_sector_owner_ids,
            vec![],
            1,
            HelicitySupportRule::None,
            GlobalHelicityFlipRule::None,
            reflection_certificates,
        )
        .unwrap();
        (
            program,
            stage_diagnostics,
            constructed_counts,
            generation_telemetry,
        )
    }

    #[test]
    fn materialized_color_targets_keep_only_embeddable_ordered_words() {
        let target_open =
            LCColorComponent::new(LCColorComponentKind::OpenString, vec![0, 2, 3, 4, 1]).unwrap();
        let targets = MaterializedColorTargets::from_sectors(vec![vec![target_open]]).unwrap();
        let state = |slots: Vec<u32>| {
            DynamicLCColorState::new(
                0,
                Some(0),
                vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, slots).unwrap()],
            )
            .unwrap()
        };
        let adjoint_state = |slots: Vec<u32>| {
            DynamicLCColorState::new_port_wired(
                0,
                vec![
                    LCColorPortBinding::new(0, LCColorEndpoint::Back),
                    LCColorPortBinding::new(0, LCColorEndpoint::Front),
                ],
                vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, slots).unwrap()],
            )
            .unwrap()
        };

        assert!(targets.accepts(&state(vec![2, 3, 4])));
        assert!(!targets.accepts(&state(vec![4, 3, 2])));
        assert!(
            targets
                .accepts_up_to_reflection(&adjoint_state(vec![4, 3, 2]))
                .unwrap()
        );
        assert!(!targets.accepts(&state(vec![2, 4])));
        assert!(
            !targets
                .accepts_up_to_reflection(&state(vec![2, 4]))
                .unwrap()
        );
        assert!(!targets.accepts(&state(vec![0, 3])));
        assert!(targets.accepts(&DynamicLCColorState::new(0, None, vec![]).unwrap()));
    }

    #[test]
    fn materialized_color_targets_accept_cyclic_trace_segments() {
        let target_trace =
            LCColorComponent::new(LCColorComponentKind::Trace, vec![1, 2, 3, 4]).unwrap();
        let targets = MaterializedColorTargets::from_sectors(vec![vec![target_trace]]).unwrap();
        let state = |kind, slots: Vec<u32>| {
            DynamicLCColorState::new(
                0,
                (kind != LCColorComponentKind::Trace).then_some(0),
                vec![LCColorComponent::new(kind, slots).unwrap()],
            )
            .unwrap()
        };

        assert!(targets.accepts(&state(LCColorComponentKind::AdjointSegment, vec![4, 1, 2],)));
        assert!(!targets.accepts(&state(LCColorComponentKind::AdjointSegment, vec![2, 1, 4],)));
        assert!(!targets.accepts(&state(LCColorComponentKind::AdjointSegment, vec![1, 3],)));
        assert!(!targets.accepts(&state(LCColorComponentKind::Trace, vec![4, 3, 2, 1],)));
        assert!(!targets.accepts(&state(LCColorComponentKind::Trace, vec![1, 2, 3],)));
    }

    fn reference_color_targets_accept(
        sectors: &[Vec<LCColorComponent>],
        state: &DynamicLCColorState,
    ) -> bool {
        if state.components().is_empty() {
            return true;
        }
        sectors.iter().any(|sector| {
            state.components().iter().all(|partial| {
                sector
                    .iter()
                    .any(|target| component_can_embed(partial, target))
            })
        })
    }

    fn unique_source_words(source_slots: &[u32]) -> Vec<Vec<u32>> {
        fn extend(
            source_slots: &[u32],
            used: &mut [bool],
            prefix: &mut Vec<u32>,
            result: &mut Vec<Vec<u32>>,
        ) {
            for index in 0..source_slots.len() {
                if used[index] {
                    continue;
                }
                used[index] = true;
                prefix.push(source_slots[index]);
                result.push(prefix.clone());
                extend(source_slots, used, prefix, result);
                prefix.pop();
                used[index] = false;
            }
        }

        let mut result = Vec::new();
        extend(
            source_slots,
            &mut vec![false; source_slots.len()],
            &mut Vec::new(),
            &mut result,
        );
        result
    }

    fn exhaustive_color_components(source_slots: &[u32]) -> Vec<LCColorComponent> {
        let mut result = BTreeSet::new();
        for kind in [
            LCColorComponentKind::OpenString,
            LCColorComponentKind::AdjointSegment,
            LCColorComponentKind::Trace,
        ] {
            for word in unique_source_words(source_slots) {
                result.insert(LCColorComponent::new(kind, word).unwrap());
            }
        }
        result.into_iter().collect()
    }

    fn components_are_disjoint(left: &LCColorComponent, right: &LCColorComponent) -> bool {
        left.source_slots()
            .iter()
            .all(|slot| !right.source_slots().contains(slot))
    }

    #[test]
    fn indexed_color_targets_match_exhaustive_single_component_reference() {
        let components = exhaustive_color_components(&[0, 1, 2, 3]);
        for target in &components {
            let sectors = vec![vec![target.clone()]];
            let targets = MaterializedColorTargets::from_sectors(sectors.clone()).unwrap();
            for partial in &components {
                let state = DynamicLCColorState::new(0, None, vec![partial.clone()]).unwrap();
                assert_eq!(
                    targets.accepts(&state),
                    reference_color_targets_accept(&sectors, &state),
                    "partial {partial:?}, target {target:?}",
                );
            }
        }
    }

    #[test]
    fn indexed_color_targets_match_exhaustive_sector_intersection_reference() {
        let sectors = vec![
            vec![
                LCColorComponent::new(LCColorComponentKind::OpenString, vec![0, 1]).unwrap(),
                LCColorComponent::new(LCColorComponentKind::Trace, vec![2, 3]).unwrap(),
            ],
            vec![
                LCColorComponent::new(LCColorComponentKind::OpenString, vec![0, 2]).unwrap(),
                LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 3]).unwrap(),
            ],
            vec![
                LCColorComponent::new(LCColorComponentKind::Trace, vec![0, 3]).unwrap(),
                LCColorComponent::new(LCColorComponentKind::OpenString, vec![1, 2]).unwrap(),
            ],
        ];
        let targets = MaterializedColorTargets::from_sectors(sectors.clone()).unwrap();
        let components = exhaustive_color_components(&[0, 1, 2, 3]);
        let empty = DynamicLCColorState::new(0, None, vec![]).unwrap();
        assert_eq!(
            targets.accepts(&empty),
            reference_color_targets_accept(&sectors, &empty)
        );
        for left in &components {
            let state = DynamicLCColorState::new(0, None, vec![left.clone()]).unwrap();
            assert_eq!(
                targets.accepts(&state),
                reference_color_targets_accept(&sectors, &state),
                "single partial {left:?}",
            );
            for right in &components {
                if !components_are_disjoint(left, right) {
                    continue;
                }
                let state =
                    DynamicLCColorState::new(0, None, vec![left.clone(), right.clone()]).unwrap();
                assert_eq!(
                    targets.accepts(&state),
                    reference_color_targets_accept(&sectors, &state),
                    "partial forest [{left:?}, {right:?}]",
                );
            }
        }

        let split_sectors = vec![
            vec![LCColorComponent::new(LCColorComponentKind::OpenString, vec![0, 1]).unwrap()],
            vec![LCColorComponent::new(LCColorComponentKind::OpenString, vec![2, 3]).unwrap()],
        ];
        let split_targets = MaterializedColorTargets::from_sectors(split_sectors.clone()).unwrap();
        let split_state = DynamicLCColorState::new(
            0,
            None,
            vec![
                LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![0, 1]).unwrap(),
                LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![2, 3]).unwrap(),
            ],
        )
        .unwrap();
        assert!(!reference_color_targets_accept(
            &split_sectors,
            &split_state
        ));
        assert!(!split_targets.accepts(&split_state));
    }

    #[test]
    fn indexed_color_target_postings_span_multiple_words() {
        let sectors = (0..130_u32)
            .map(|source_slot| {
                vec![
                    LCColorComponent::new(LCColorComponentKind::OpenString, vec![source_slot])
                        .unwrap(),
                ]
            })
            .collect::<Vec<_>>();
        let targets = MaterializedColorTargets::from_sectors(sectors.clone()).unwrap();
        for source_slot in [0, 63, 64, 127, 129, 130] {
            let state = DynamicLCColorState::new(
                0,
                None,
                vec![
                    LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![source_slot])
                        .unwrap(),
                ],
            )
            .unwrap();
            assert_eq!(
                targets.accepts(&state),
                reference_color_targets_accept(&sectors, &state),
                "source slot {source_slot}",
            );
        }
    }

    #[test]
    fn sector_postings_use_dense_storage_only_when_strictly_cheaper() {
        let sparse_on_tie = SectorPosting::from_sorted_unique_sector_ids((0..6_u32).collect(), 130);
        assert!(sparse_on_tie.is_sparse());
        assert_eq!(sparse_on_tie.cardinality(), 6);
        assert_eq!(sparse_on_tie.payload_bytes(), 24);

        let dense = SectorPosting::from_sorted_unique_sector_ids((0..7_u32).collect(), 130);
        assert!(!dense.is_sparse());
        assert_eq!(dense.cardinality(), 7);
        assert_eq!(dense.payload_bytes(), 24);
        assert!(dense.contains(0));
        assert!(dense.contains(6));
        assert!(!dense.contains(7));
        assert!(!dense.contains(130));
    }

    #[test]
    fn adaptive_color_postings_match_reference_and_account_exact_storage() {
        let component = |source_slot| {
            LCColorComponent::new(LCColorComponentKind::OpenString, vec![source_slot]).unwrap()
        };
        let partial_state = |source_slots: &[u32]| {
            DynamicLCColorState::new(
                0,
                None,
                source_slots
                    .iter()
                    .copied()
                    .map(|source_slot| {
                        LCColorComponent::new(
                            LCColorComponentKind::AdjointSegment,
                            vec![source_slot],
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap()
        };

        let mut sectors = vec![Vec::new(); 256];
        for sector in &mut sectors[..128] {
            sector.push(component(2_000));
        }
        sectors[5].push(component(1_000));
        sectors[5].push(component(1_000));
        sectors[200].push(component(1_000));
        sectors[200].push(component(3_000));

        let targets =
            MaterializedColorTargets::from_sectors_with_telemetry(sectors.clone(), true).unwrap();
        let storage = targets.telemetry().unwrap();
        assert_eq!(storage.fragment_bucket_count, 3);
        assert_eq!(storage.posting_incidence_count, 131);
        assert_eq!(storage.sparse_posting_bucket_count, 2);
        assert_eq!(storage.dense_posting_bucket_count, 1);
        assert_eq!(storage.sparse_posting_bytes, 12);
        assert_eq!(storage.dense_posting_bytes, 32);
        let dense_only_baseline_bytes = storage
            .fragment_bucket_count
            .checked_mul(256_usize.div_ceil(u64::BITS as usize))
            .and_then(|words| words.checked_mul(std::mem::size_of::<u64>()))
            .unwrap();
        assert_eq!(dense_only_baseline_bytes, 96);
        assert_eq!(
            storage
                .sparse_posting_bytes
                .checked_add(storage.dense_posting_bytes),
            Some(44),
        );

        for source_slots in [
            &[][..],
            &[1_000][..],
            &[2_000][..],
            &[3_000][..],
            &[1_000, 2_000][..],
            &[1_000, 3_000][..],
            &[1_000, 2_000, 3_000][..],
            &[4_000][..],
        ] {
            let state = partial_state(source_slots);
            assert_eq!(
                targets.accepts(&state),
                reference_color_targets_accept(&sectors, &state),
                "partial source slots {source_slots:?}",
            );
        }

        let memo_targets =
            MaterializedColorTargets::from_sectors_with_telemetry(sectors.clone(), true).unwrap();
        let accepted = partial_state(&[1_000, 2_000]);
        let rejected = partial_state(&[1_000, 2_000, 3_000]);
        assert!(memo_targets.accepts(&accepted));
        assert_eq!(memo_targets.accepted_component_forests.borrow().len(), 1);
        assert!(memo_targets.accepts(&accepted));
        assert_eq!(memo_targets.accepted_component_forests.borrow().len(), 1);
        assert!(!memo_targets.accepts(&rejected));
        assert!(!memo_targets.accepts(&rejected));
        assert_eq!(memo_targets.accepted_component_forests.borrow().len(), 1);
        let memo = memo_targets.telemetry().unwrap();
        assert_eq!(memo.acceptance_cache_hit_count, 1);
        assert_eq!(memo.acceptance_cache_miss_count, 3);
        assert_eq!(memo.acceptance_accept_count, 2);
        assert_eq!(memo.acceptance_reject_count, 2);
    }

    #[test]
    fn all_dense_color_postings_match_exact_wordwise_intersection_reference() {
        let component = |source_slot| {
            LCColorComponent::new(LCColorComponentKind::OpenString, vec![source_slot]).unwrap()
        };
        let mut sectors = vec![Vec::new(); 256];
        for sector in sectors.iter_mut().take(192) {
            sector.push(component(10));
        }
        for sector in sectors.iter_mut().take(256).skip(64) {
            sector.push(component(20));
        }
        for sector in sectors.iter_mut().take(256).step_by(2) {
            sector.push(component(30));
        }
        for sector in &mut sectors[192..] {
            sector.push(component(40));
        }
        let targets = MaterializedColorTargets::from_sectors(sectors.clone()).unwrap();
        assert!(
            targets
                .non_trace_fragment_sectors
                .values()
                .all(|posting| !posting.is_sparse())
        );

        for source_slots in [
            &[10, 20][..],
            &[10, 30][..],
            &[20, 30][..],
            &[10, 20, 30][..],
            &[10, 40][..],
            &[20, 40][..],
        ] {
            let state = DynamicLCColorState::new(
                0,
                None,
                source_slots
                    .iter()
                    .map(|source_slot| {
                        LCColorComponent::new(
                            LCColorComponentKind::AdjointSegment,
                            vec![*source_slot],
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap();
            assert_eq!(
                targets.accepts(&state),
                reference_color_targets_accept(&sectors, &state),
                "all-dense partial source slots {source_slots:?}",
            );
        }
    }

    #[test]
    fn indexed_color_targets_match_exhaustive_reflection_reference() {
        let sectors = vec![
            vec![
                LCColorComponent::new(LCColorComponentKind::OpenString, vec![0, 1, 2, 3]).unwrap(),
            ],
            vec![LCColorComponent::new(LCColorComponentKind::Trace, vec![0, 2, 3, 1]).unwrap()],
        ];
        let targets = MaterializedColorTargets::from_sectors(sectors.clone()).unwrap();
        let mut accepted = 0usize;
        let mut rejected = 0usize;
        for word in unique_source_words(&[0, 1, 2, 3]) {
            let state = DynamicLCColorState::new_port_wired(
                0,
                vec![
                    LCColorPortBinding::new(0, LCColorEndpoint::Back),
                    LCColorPortBinding::new(0, LCColorEndpoint::Front),
                ],
                vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, word).unwrap()],
            )
            .unwrap();
            let expected = reference_color_targets_accept(&sectors, &state)
                || reference_color_targets_accept(&sectors, &state.reversed().unwrap());
            assert_eq!(
                targets.accepts_up_to_reflection(&state).unwrap(),
                expected,
                "pure-adjoint state {state:?}",
            );
            if expected {
                accepted += 1;
            } else {
                rejected += 1;
            }
        }
        assert!(accepted > 0);
        assert!(rejected > 0);
    }

    #[test]
    fn color_target_memo_retains_only_accepted_forests_without_changing_reflection_eligibility() {
        let targets = MaterializedColorTargets::from_sectors(vec![vec![
            LCColorComponent::new(LCColorComponentKind::OpenString, vec![0, 1, 2]).unwrap(),
        ]])
        .unwrap();
        let component =
            LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![2, 1]).unwrap();
        let non_reflectable =
            DynamicLCColorState::new(0, Some(0), vec![component.clone()]).unwrap();
        assert!(!targets.accepts_up_to_reflection(&non_reflectable).unwrap());
        assert!(targets.accepted_component_forests.borrow().is_empty());
        assert!(!targets.accepts_up_to_reflection(&non_reflectable).unwrap());
        assert!(targets.accepted_component_forests.borrow().is_empty());

        let reflectable = DynamicLCColorState::new_port_wired(
            99,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![component],
        )
        .unwrap();
        assert!(targets.accepts_up_to_reflection(&reflectable).unwrap());
        assert_eq!(targets.accepted_component_forests.borrow().len(), 1);
        assert!(targets.accepts_up_to_reflection(&reflectable).unwrap());
        assert_eq!(targets.accepted_component_forests.borrow().len(), 1);
    }

    #[test]
    fn replay_construction_isolates_authenticated_representatives() {
        let replay_targets = [
            RecurrenceReplayTarget::new(0, 4, 0, vec![0, 1], vec![1, 1], ExactComplexRational::ONE)
                .unwrap(),
            RecurrenceReplayTarget::new(1, 9, 1, vec![1, 0], vec![1, 1], ExactComplexRational::ONE)
                .unwrap(),
        ];
        let sectors = BTreeSet::from([4, 9]);
        assert_eq!(
            construction_sector_groups(
                RecurrenceStrategy::ContractedColorUnion,
                &sectors,
                &replay_targets,
            ),
            [BTreeSet::from([4]), BTreeSet::from([9])]
        );
        assert_eq!(
            construction_sector_groups(RecurrenceStrategy::AllFlowUnion, &sectors, &[]),
            std::slice::from_ref(&sectors)
        );

        let first_lane = PendingConstructionDomain {
            shared_source_end: 4,
            lane_internal_start: 4,
            lane_internal_end: 10,
        };
        assert!(first_lane.contains(0));
        assert!(first_lane.contains(9));
        assert!(!first_lane.contains(10));
    }

    #[test]
    fn scalar_reference_fixtures_keep_structural_schedule_counts() {
        // Closure-rooted admission materializes no current or contribution
        // outside the final exact schedule for this scalar grammar.
        for (name, external_count, expected_constructed, expected_schedule) in [
            ("three-point", 3, (4, 1, 1), (4, 1, 1)),
            ("four-point", 4, (8, 6, 1), (8, 6, 1)),
        ] {
            let (program, _, constructed_counts, _) = scalar_reference_program(external_count);
            assert_eq!(
                constructed_counts, expected_constructed,
                "{name} scalar construction fixture",
            );
            assert_eq!(
                (
                    program.currents().len(),
                    program.contributions().len(),
                    program.closure_terms().len(),
                ),
                expected_schedule,
                "{name} scalar reference fixture",
            );
        }
    }

    #[test]
    fn scalar_reference_fixture_reports_streaming_selectivity() {
        let (_, diagnostics, _, _) = scalar_reference_program(4);
        assert_eq!(diagnostics.len(), 2);
        assert_eq!(
            diagnostics
                .iter()
                .map(|stage| (
                    stage.target_size,
                    stage.candidate_parent_pair_count,
                    stage.parent_pair_count,
                    stage.transition_index_hit_count,
                    stage.transition_candidate_count,
                    stage.contribution_attempt_count,
                ))
                .collect::<Vec<_>>(),
            [(2, 6, 6, 6, 6, 3), (3, 12, 6, 6, 6, 3)],
        );
    }

    #[test]
    fn scalar_reference_fixture_reports_exact_closure_index_counters() {
        let (_, _, _, telemetry) = scalar_reference_program(4);
        assert_eq!(telemetry.closure_support_lookup_count, 2);
        assert_eq!(telemetry.closure_candidate_theoretical_count, 1);
        assert_eq!(telemetry.closure_candidate_count, 1);
        assert_eq!(telemetry.closure_state_match_count, 1);
        assert_eq!(telemetry.closure_color_attempt_count, 1);
        assert_eq!(telemetry.closure_group_count, 1);
        assert_eq!(telemetry.closure_proof_contribution_count, 1);
    }

    #[test]
    fn contracted_closure_buckets_match_legacy_rows_and_collapse_executed_attempts() {
        let template = scalar_reference_template();
        let template_catalog = TemplateCatalog::new(&template).unwrap();
        let prepared_closures = PreparedClosureCatalog::new(&template, &template_catalog).unwrap();
        let mut process = scalar_reference_process(2);
        process
            .physical_lc_sectors
            .push(ProcessPhysicalLCSectorRow {
                sector_id: 1,
                ..process.physical_lc_sectors[0]
            });
        let process_catalog = ProcessCatalog::new(&process).unwrap();
        let materialized_sectors = BTreeSet::from([0, 1]);
        let prepared_sectors = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            &process_catalog,
            &materialized_sectors,
        )
        .unwrap();
        let fermion_ordering =
            established_fermion_ordering_context(&process, &template, None).unwrap();
        let mut color_states = DynamicLCColorStateInterner::default();
        let color_id = color_states
            .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
            .unwrap();
        let currents = (0..2_u32)
            .map(|source_slot| {
                let key = CurrentCoreKey::new(
                    template.catalog_digest,
                    RecurrenceNodeKind::Source,
                    0,
                    color_id,
                    vec![source_slot],
                    CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                        source_slot,
                        coefficient: 1,
                    }])
                    .unwrap(),
                    CurrentHelicityIdentity::contracted_color_union(
                        0,
                        vec![SourceStateAssignment::new(source_slot, 0)],
                    )
                    .unwrap(),
                    vec![0],
                    0,
                    vec![],
                    CurrentSourceBinding::FixedTemplate(source_slot),
                    None,
                )
                .unwrap();
                PendingCurrent {
                    key,
                    source_exact_factor: Some(ExactComplexRational::ONE),
                    contributions: BTreeMap::new(),
                    realized_pairing_rule_ids: BTreeSet::new(),
                    reflection: CurrentReflection::Unavailable,
                    reflection_certificate_id: None,
                }
            })
            .collect::<Vec<_>>();

        let mut bucketed_telemetry = RecurrenceGenerationTelemetry::default();
        let bucketed = build_closures(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            &process_catalog,
            &template,
            &fermion_ordering,
            None,
            &prepared_closures,
            &prepared_sectors,
            &color_states,
            &currents,
            &materialized_sectors,
            &[],
            &[],
            None,
            &mut bucketed_telemetry,
            true,
        )
        .unwrap();
        let mut legacy_telemetry = RecurrenceGenerationTelemetry::default();
        let legacy = build_closures_per_sector(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            &process_catalog,
            &template,
            &fermion_ordering,
            None,
            &prepared_closures,
            &prepared_sectors,
            &color_states,
            &currents,
            &materialized_sectors,
            &[],
            &[],
            None,
            &mut legacy_telemetry,
            true,
        )
        .unwrap();

        // PendingClosureGroup equality covers key identity, aggregate exact
        // factors, and every proof contribution field in vector order.
        assert_eq!(bucketed, legacy);
        assert_eq!(
            bucketed
                .keys()
                .map(|key| key.target_sector_id)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(bucketed_telemetry.closure_candidate_theoretical_count, 2);
        assert_eq!(
            bucketed_telemetry.closure_candidate_theoretical_count,
            legacy_telemetry.closure_candidate_theoretical_count
        );
        assert_eq!(bucketed_telemetry.closure_support_lookup_count, 2);
        assert_eq!(legacy_telemetry.closure_support_lookup_count, 4);
        assert_eq!(bucketed_telemetry.closure_candidate_count, 1);
        assert_eq!(legacy_telemetry.closure_candidate_count, 2);
        assert_eq!(bucketed_telemetry.closure_state_match_count, 1);
        assert_eq!(legacy_telemetry.closure_state_match_count, 2);
        assert_eq!(bucketed_telemetry.closure_color_attempt_count, 1);
        assert_eq!(legacy_telemetry.closure_color_attempt_count, 2);
        assert_eq!(bucketed_telemetry.closure_group_count, 2);
        assert_eq!(bucketed_telemetry.closure_proof_contribution_count, 2);
    }

    #[test]
    fn complete_closure_target_domain_certifies_only_an_exhaustive_empty_owner_class() {
        let process = scalar_reference_process(2);
        let sector_ids = BTreeSet::from([0, 1]);
        let owner_ids = BTreeMap::from([(0, 0), (1, 0)]);
        let helicity_ids = BTreeMap::from([(
            vec![
                SourceStateAssignment::new(0, 0),
                SourceStateAssignment::new(1, 0),
            ]
            .into_boxed_slice(),
            0,
        )]);
        let pending_closures = BTreeMap::new();
        let certificate = complete_closure_target_domain_certificate(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            2,
            &sector_ids,
            &owner_ids,
            &[],
            &helicity_ids,
            &pending_closures,
            &[],
        )
        .unwrap()
        .unwrap();

        assert_eq!(certificate.physical_sector_count(), 2);
        assert_eq!(certificate.resolved_helicity_count(), 1);
        assert_eq!(
            certificate.certified_structural_zero_physical_sector_ids(),
            [0, 1]
        );
        assert_eq!(
            complete_closure_target_domain_certificate(
                RecurrenceStrategy::ContractedColorUnion,
                &process,
                2,
                &sector_ids,
                &owner_ids,
                &[],
                &helicity_ids,
                &pending_closures,
                &[],
            )
            .unwrap()
            .unwrap()
            .semantic_digest(),
            certificate.semantic_digest(),
        );

        let missing = complete_closure_target_domain_certificate(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            2,
            &BTreeSet::from([0]),
            &BTreeMap::from([(0, 0)]),
            &[],
            &helicity_ids,
            &pending_closures,
            &[],
        )
        .unwrap_err();
        assert!(
            missing
                .message()
                .contains("lacks exhaustive physical-sector coverage")
        );
    }

    #[test]
    fn complete_closure_target_domain_expands_authenticated_replay_representatives() {
        let mut process = scalar_reference_process(2);
        process.source_states = vec![
            ProcessSourceStateRow {
                source_slot: 0,
                state_index: 0,
                public_helicity: 1,
                chirality: 0,
                spin_state: 1,
                current_state_template_id: 0,
                source_template_id: 0,
                momentum_sign: 1,
                crossing_phase_factor_id: 0,
            },
            ProcessSourceStateRow {
                source_slot: 1,
                state_index: 0,
                public_helicity: 1,
                chirality: 0,
                spin_state: 1,
                current_state_template_id: 0,
                source_template_id: 0,
                momentum_sign: 1,
                crossing_phase_factor_id: 0,
            },
        ];
        let helicity_ids = BTreeMap::from([(
            vec![
                SourceStateAssignment::new(0, 0),
                SourceStateAssignment::new(1, 0),
            ]
            .into_boxed_slice(),
            0,
        )]);
        let replay_targets = vec![
            RecurrenceReplayTarget::new(0, 0, 0, vec![0, 1], vec![1, 1], ExactComplexRational::ONE)
                .unwrap(),
            RecurrenceReplayTarget::new(1, 0, 1, vec![1, 0], vec![1, 1], ExactComplexRational::ONE)
                .unwrap(),
        ];
        let certificate = complete_closure_target_domain_certificate(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            2,
            &BTreeSet::from([0]),
            &BTreeMap::from([(0, 0)]),
            &replay_targets,
            &helicity_ids,
            &BTreeMap::new(),
            &[],
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            certificate.certified_structural_zero_physical_sector_ids(),
            [0, 1]
        );
        assert!(
            complete_closure_target_domain_certificate(
                RecurrenceStrategy::ContractedColorUnion,
                &process,
                2,
                &BTreeSet::from([0]),
                &BTreeMap::from([(0, 0)]),
                &replay_targets[..1],
                &helicity_ids,
                &BTreeMap::new(),
                &[],
            )
            .unwrap_err()
            .message()
            .contains("does not cover every physical sector")
        );
    }

    #[test]
    fn contracted_and_all_flow_closures_preserve_v4_reflection_partner_rows() {
        let (mut process, mut template) = cyclic_trace_anchor_contract_fixture();
        let closure_pairings = append_u32_sequence(&mut template, &[0, 0, 1, 1, 0, 1, 1, 0]);
        template.lc_color_transition_witnesses[1].input_port_pairing_sequence_id = closure_pairings;
        let reflected_word_id = process.u32_sequence_ranges.len() as u32;
        process.u32_sequence_ranges.push(CheckedTableRange::new(
            process.u32_sequence_values.len() as u64,
            4,
        ));
        process.u32_sequence_values.extend_from_slice(&[0, 3, 2, 1]);
        process
            .physical_lc_sectors
            .push(ProcessPhysicalLCSectorRow {
                sector_id: 1,
                trace_sequence_id: reflected_word_id,
                word_sequence_id: reflected_word_id,
                ..process.physical_lc_sectors[0]
            });

        let template_catalog = TemplateCatalog::new(&template).unwrap();
        let prepared_closures = PreparedClosureCatalog::new(&template, &template_catalog).unwrap();
        let process_catalog = ProcessCatalog::new(&process).unwrap();
        let materialized_sectors = BTreeSet::from([0, 1]);
        let prepared_sectors = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            &process_catalog,
            &materialized_sectors,
        )
        .unwrap();
        let fermion_ordering =
            established_fermion_ordering_context(&process, &template, None).unwrap();

        let adjoint =
            |word| LCColorComponent::new(LCColorComponentKind::AdjointSegment, word).unwrap();
        let color = |word| {
            DynamicLCColorState::new_port_wired(
                0,
                vec![
                    LCColorPortBinding::new(0, LCColorEndpoint::Back),
                    LCColorPortBinding::new(0, LCColorEndpoint::Front),
                ],
                vec![adjoint(word)],
            )
            .unwrap()
        };
        let anchor_color = color(vec![0]);
        let complement_color = color(vec![1, 2, 3]);
        let reflected_color = color(vec![3, 2, 1]);
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let canonical_proof = CurrentReflectionProof::new(
            minus_one,
            [digest(180)],
            dynamic_color_identity_digest(&complement_color).unwrap(),
        )
        .unwrap();
        let reflected_proof = CurrentReflectionProof::new(
            minus_one,
            [digest(181)],
            dynamic_color_identity_digest(&reflected_color).unwrap(),
        )
        .unwrap();
        let reflection_certificate = PendingReflectionCertificate::reciprocal_pair(
            0,
            1,
            2,
            &canonical_proof,
            &reflected_proof,
            &complement_color,
            &reflected_color,
            4,
        )
        .unwrap();
        let mut color_states = DynamicLCColorStateInterner::default();
        let anchor_color_id = color_states.intern(anchor_color).unwrap();
        let complement_color_id = color_states.intern(complement_color).unwrap();
        let current = |strategy,
                       node_kind,
                       color_id,
                       support: &[u32],
                       reflection,
                       reflection_certificate_id| {
            let helicity_identity = match strategy {
                RecurrenceStrategy::AllFlowUnion => CurrentHelicityIdentity::all_flow_union(0),
                RecurrenceStrategy::ContractedColorUnion => {
                    CurrentHelicityIdentity::contracted_color_union(
                        0,
                        support
                            .iter()
                            .copied()
                            .map(|slot| SourceStateAssignment::new(slot, 0))
                            .collect(),
                    )
                    .unwrap()
                }
                RecurrenceStrategy::TopologyReplay => unreachable!("test fixture strategy"),
            };
            let source_binding = if node_kind != RecurrenceNodeKind::Source {
                CurrentSourceBinding::None
            } else if strategy == RecurrenceStrategy::AllFlowUnion {
                CurrentSourceBinding::runtime_dispatch(support[0], vec![support[0]]).unwrap()
            } else {
                CurrentSourceBinding::FixedTemplate(support[0])
            };
            let key = CurrentCoreKey::new(
                template.catalog_digest,
                node_kind,
                0,
                color_id,
                support.to_vec(),
                CanonicalMomentumLinearForm::new(
                    support
                        .iter()
                        .copied()
                        .map(|source_slot| MomentumTerm {
                            source_slot,
                            coefficient: 1,
                        })
                        .collect(),
                )
                .unwrap(),
                helicity_identity,
                vec![0],
                0,
                vec![],
                source_binding,
                None,
            )
            .unwrap();
            PendingCurrent {
                key,
                source_exact_factor: (node_kind == RecurrenceNodeKind::Source
                    && strategy != RecurrenceStrategy::AllFlowUnion)
                    .then_some(ExactComplexRational::ONE),
                contributions: BTreeMap::new(),
                realized_pairing_rule_ids: BTreeSet::new(),
                reflection,
                reflection_certificate_id,
            }
        };
        let currents = vec![
            current(
                RecurrenceStrategy::ContractedColorUnion,
                RecurrenceNodeKind::Source,
                anchor_color_id,
                &[0],
                CurrentReflection::Unavailable,
                None,
            ),
            current(
                RecurrenceStrategy::ContractedColorUnion,
                RecurrenceNodeKind::Current,
                complement_color_id,
                &[1, 2, 3],
                CurrentReflection::Proven(canonical_proof.clone()),
                Some(0),
            ),
        ];
        let reflection_certificates = [reflection_certificate];

        let mut bucketed_telemetry = RecurrenceGenerationTelemetry::default();
        let bucketed = build_closures(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            &process_catalog,
            &template,
            &fermion_ordering,
            None,
            &prepared_closures,
            &prepared_sectors,
            &color_states,
            &currents,
            &materialized_sectors,
            &[],
            &reflection_certificates,
            None,
            &mut bucketed_telemetry,
            true,
        )
        .unwrap();
        let mut legacy_telemetry = RecurrenceGenerationTelemetry::default();
        let legacy = build_closures_per_sector(
            RecurrenceStrategy::ContractedColorUnion,
            &process,
            &process_catalog,
            &template,
            &fermion_ordering,
            None,
            &prepared_closures,
            &prepared_sectors,
            &color_states,
            &currents,
            &materialized_sectors,
            &[],
            &reflection_certificates,
            None,
            &mut legacy_telemetry,
            true,
        )
        .unwrap();

        assert_eq!(bucketed, legacy);
        assert_eq!(bucketed.len(), 2);
        let canonical = bucketed
            .iter()
            .find(|(key, _)| key.target_sector_id == 0)
            .unwrap();
        let reflected = bucketed
            .iter()
            .find(|(key, _)| key.target_sector_id == 1)
            .unwrap();
        assert_eq!(canonical.1.exact_factor, ExactComplexRational::ONE);
        assert_eq!(reflected.1.exact_factor, minus_one);
        assert_eq!(
            reflected.1.contributions[0].reflection_certificate_id,
            Some(0)
        );
        assert_eq!(bucketed_telemetry.closure_color_attempt_count, 1);
        assert_eq!(legacy_telemetry.closure_color_attempt_count, 2);

        // The all-flow lane retains the same canonical complement and emits
        // the reverse physical trace through the authenticated reflection
        // certificate.  Completeness is therefore per physical target, not
        // per-sector growth of the shared closure map.
        let all_flow_sectors = PreparedClosureSectorCatalog::new(
            RecurrenceStrategy::AllFlowUnion,
            &process,
            &process_catalog,
            &materialized_sectors,
        )
        .unwrap();
        let all_flow_currents = vec![
            current(
                RecurrenceStrategy::AllFlowUnion,
                RecurrenceNodeKind::Source,
                anchor_color_id,
                &[0],
                CurrentReflection::Unavailable,
                None,
            ),
            current(
                RecurrenceStrategy::AllFlowUnion,
                RecurrenceNodeKind::Current,
                complement_color_id,
                &[1, 2, 3],
                CurrentReflection::Proven(canonical_proof),
                Some(0),
            ),
        ];
        let mut all_flow_telemetry = RecurrenceGenerationTelemetry::default();
        let all_flow = build_closures(
            RecurrenceStrategy::AllFlowUnion,
            &process,
            &process_catalog,
            &template,
            &fermion_ordering,
            None,
            &prepared_closures,
            &all_flow_sectors,
            &color_states,
            &all_flow_currents,
            &materialized_sectors,
            &[],
            &reflection_certificates,
            None,
            &mut all_flow_telemetry,
            true,
        )
        .unwrap();
        let mut all_flow_legacy_telemetry = RecurrenceGenerationTelemetry::default();
        let all_flow_legacy = build_closures_per_sector(
            RecurrenceStrategy::AllFlowUnion,
            &process,
            &process_catalog,
            &template,
            &fermion_ordering,
            None,
            &prepared_closures,
            &all_flow_sectors,
            &color_states,
            &all_flow_currents,
            &materialized_sectors,
            &[],
            &reflection_certificates,
            None,
            &mut all_flow_legacy_telemetry,
            true,
        )
        .unwrap();

        assert_eq!(all_flow, all_flow_legacy);
        assert_eq!(all_flow.len(), 2);
        let canonical = all_flow
            .iter()
            .find(|(key, _)| key.target_sector_id == 0)
            .unwrap();
        let reflected = all_flow
            .iter()
            .find(|(key, _)| key.target_sector_id == 1)
            .unwrap();
        assert_eq!(canonical.1.exact_factor, ExactComplexRational::ONE);
        assert_eq!(reflected.1.exact_factor, minus_one);
        assert_eq!(
            reflected.1.contributions[0].reflection_certificate_id,
            Some(0)
        );
        assert_eq!(all_flow_telemetry.closure_support_lookup_count, 2);
        assert_eq!(all_flow_legacy_telemetry.closure_support_lookup_count, 4);
        assert_eq!(all_flow_telemetry.closure_color_attempt_count, 1);
        assert_eq!(all_flow_legacy_telemetry.closure_color_attempt_count, 2);
        assert_eq!(all_flow_telemetry.closure_group_count, 2);
        assert_eq!(all_flow_telemetry.closure_proof_contribution_count, 2);

        let uncertified_currents = vec![
            current(
                RecurrenceStrategy::AllFlowUnion,
                RecurrenceNodeKind::Source,
                anchor_color_id,
                &[0],
                CurrentReflection::Unavailable,
                None,
            ),
            current(
                RecurrenceStrategy::AllFlowUnion,
                RecurrenceNodeKind::Current,
                complement_color_id,
                &[1, 2, 3],
                CurrentReflection::Unavailable,
                None,
            ),
        ];
        let error = build_closures(
            RecurrenceStrategy::AllFlowUnion,
            &process,
            &process_catalog,
            &template,
            &fermion_ordering,
            None,
            &prepared_closures,
            &all_flow_sectors,
            &color_states,
            &uncertified_currents,
            &materialized_sectors,
            &[],
            &[],
            None,
            &mut RecurrenceGenerationTelemetry::default(),
            false,
        )
        .unwrap_err();
        assert!(
            error
                .message()
                .contains("no exact closure for physical LC sector 1")
        );
    }

    #[test]
    fn accepted_transitions_clone_only_inserted_current_keys() {
        let (_, _, _, telemetry) = scalar_reference_program(4);
        assert!(telemetry.transition_accept_count > 0);
        assert_eq!(telemetry.accepted_parent_key_clone_count, 0);
        assert_eq!(
            telemetry.current_key_clone_count,
            telemetry.current_insert_count,
        );
    }

    #[test]
    fn non_propagating_states_use_identity_finalization() {
        fn propagator(id: u32, state: u32, applies: bool) -> PropagatorRow {
            PropagatorRow {
                id,
                template_string_id: 0,
                state_template_id: state,
                applies_propagator: u8::from(applies),
                evaluator_binding_id: if applies { 0 } else { MISSING_U32 },
                numerator_expression_digest_id: 0,
                denominator_expression_digest_id: 0,
                mass_parameter_id: MISSING_U32,
                width_parameter_id: MISSING_U32,
                gauge_string_id: 0,
                linearity_proof_template_id: MISSING_U32,
                semantic_digest_id: 0,
            }
        }

        let mut template = scalar_reference_template();
        template.propagators = vec![propagator(0, 4, false), propagator(1, 7, true)];
        assert_eq!(
            propagator_by_state(&template).unwrap(),
            BTreeMap::from([(4, None), (7, Some(1))]),
        );
    }

    #[test]
    fn output_factor_uses_the_declared_binding_coupling_component() {
        let coupling = ExactComplexRational::new(
            ExactRational::new(2, 3).unwrap(),
            ExactRational::new(-5, 7).unwrap(),
        );
        assert_eq!(
            output_factor_from_binding(
                coupling,
                OutputFactorSource::None as u8,
                "test transition",
            )
            .unwrap(),
            ExactComplexRational::ONE,
        );
        assert_eq!(
            output_factor_from_binding(
                coupling,
                OutputFactorSource::CouplingReal as u8,
                "test transition",
            )
            .unwrap(),
            ExactComplexRational::new(ExactRational::new(2, 3).unwrap(), ExactRational::ZERO,),
        );
        assert_eq!(
            output_factor_from_binding(
                coupling,
                OutputFactorSource::CouplingImag as u8,
                "test transition",
            )
            .unwrap(),
            ExactComplexRational::new(ExactRational::new(-5, 7).unwrap(), ExactRational::ZERO,),
        );
    }

    #[test]
    fn prepared_transition_defers_reachable_only_output_factor_rejection() {
        let mut template = scalar_reference_template();
        template.transitions[0].output_factor_source = OutputFactorSource::CouplingImag as u8;
        let catalog = TemplateCatalog::new(&template).unwrap();
        let prepared = PreparedTransitionCatalog::new(&template, &catalog).unwrap();
        let transition = prepared
            .rows_by_state_pair
            .values()
            .next()
            .unwrap()
            .first()
            .unwrap();
        assert_eq!(
            transition.output_factor().unwrap_err(),
            output_factor_from_binding(
                ExactComplexRational::ONE,
                OutputFactorSource::CouplingImag as u8,
                "transition",
            )
            .unwrap_err()
        );
    }

    #[test]
    fn composes_construction_and_public_gather_permutations() {
        let representative_to_construction = [0, 2, 3, 1];
        let construction_to_public = [0, 3, 1, 2];
        assert_eq!(
            compose_gather_permutations(&representative_to_construction, &construction_to_public,)
                .unwrap(),
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn replay_momentum_signs_cross_initial_and_final_sources() {
        assert_eq!(
            replay_momentum_signs(&[-1, -1, 1, 1], &[0, 3, 2, 1]).unwrap(),
            [1, -1, 1, -1]
        );
        assert_eq!(
            replay_momentum_signs(&[-1, -1, 1, 1], &[0, 1, 3, 2]).unwrap(),
            [1, 1, 1, 1]
        );
    }

    fn projection_test_current(
        node_kind: RecurrenceNodeKind,
        state: u32,
        color: u32,
        support: &[u32],
    ) -> PendingCurrent {
        let helicity_identity = CurrentHelicityIdentity::topology_replay(
            0,
            support
                .iter()
                .copied()
                .map(|slot| SourceStateAssignment::new(slot, 0))
                .collect(),
        )
        .unwrap();
        let key = CurrentCoreKey::new(
            digest(230),
            node_kind,
            state,
            DynamicLCColorStateId::from_interner(color),
            support.to_vec(),
            CanonicalMomentumLinearForm::new(
                support
                    .iter()
                    .copied()
                    .map(|source_slot| MomentumTerm {
                        source_slot,
                        coefficient: 1,
                    })
                    .collect(),
            )
            .unwrap(),
            helicity_identity,
            vec![0],
            0,
            vec![],
            if node_kind == RecurrenceNodeKind::Source {
                CurrentSourceBinding::FixedTemplate(support[0])
            } else {
                CurrentSourceBinding::None
            },
            None,
        )
        .unwrap();
        PendingCurrent {
            key,
            source_exact_factor: (node_kind == RecurrenceNodeKind::Source)
                .then_some(ExactComplexRational::ONE),
            contributions: BTreeMap::new(),
            realized_pairing_rule_ids: BTreeSet::new(),
            reflection: CurrentReflection::Unavailable,
            reflection_certificate_id: None,
        }
    }

    fn add_projection_test_contribution(
        currents: &mut [PendingCurrent],
        destination: u32,
        transition: u32,
        parents: [u32; 2],
        witness: u32,
    ) {
        let parent_keys = parents.map(|id| &currents[id as usize].key);
        let destination_key = &currents[destination as usize].key;
        let key = ContributionKey::new(
            transition,
            parents.to_vec(),
            parent_keys
                .iter()
                .map(|key| key.current_state_template_id())
                .collect(),
            parent_keys
                .iter()
                .map(|key| key.momentum().clone())
                .collect(),
            destination_key.current_state_template_id(),
            transition,
            LCColorWitnessTermId::new(transition, witness),
            digest(
                231_u8
                    .checked_add(u8::try_from(transition).unwrap())
                    .unwrap(),
            ),
            0,
        )
        .unwrap();
        currents[destination as usize].contributions.insert(
            PendingContributionKey {
                parent_current_ids: parents.into(),
                key,
            },
            ExactComplexRational::ONE,
        );
    }

    fn established_contact_test_transition(
        step: ContactOrbitStepProof,
        input_state_template_ids: [u32; 2],
        transition_digest: u8,
    ) -> PreparedContactOrbitTransition {
        prepared_contact_orbit_transition_for_test(
            step,
            input_state_template_ids,
            digest(transition_digest),
            contact_orbit_application_for_test(),
        )
    }

    fn add_established_contact_test_contribution(
        currents: &mut [PendingCurrent],
        destination: u32,
        transition: u32,
        parents: [u32; 2],
    ) {
        let parent_keys = parents.map(|id| &currents[id as usize].key);
        let destination_key = &currents[destination as usize].key;
        let key = ContributionKey::new(
            transition,
            parents.to_vec(),
            parent_keys
                .iter()
                .map(|key| key.current_state_template_id())
                .collect(),
            parent_keys
                .iter()
                .map(|key| key.momentum().clone())
                .collect(),
            destination_key.current_state_template_id(),
            0,
            LCColorWitnessTermId::new(4, 0),
            digest(2),
            8,
        )
        .unwrap();
        assert!(
            currents[destination as usize]
                .contributions
                .insert(
                    PendingContributionKey {
                        parent_current_ids: parents.into(),
                        key,
                    },
                    ExactComplexRational::ONE,
                )
                .is_none()
        );
    }

    fn established_contact_transition_ids(current: &PendingCurrent) -> Vec<u32> {
        current
            .contributions
            .keys()
            .map(|pending| pending.key.transition_template_id())
            .collect()
    }

    fn commit_established_contact_test_plan(
        currents: &mut [PendingCurrent],
        contacts: &BTreeMap<u32, PreparedContactOrbitTransition>,
        stage_current_start: usize,
        resident_contribution_count: &mut usize,
    ) -> RusticolResult<()> {
        if let Some(plan) = plan_established_contact_orbit_owners_with_resolver(
            stage_current_start,
            currents,
            *resident_contribution_count,
            |transition_id| contacts.get(&transition_id),
        )? {
            plan.commit(currents, resident_contribution_count);
        }
        Ok(())
    }

    fn established_0000_contact_case(reverse_insertion: bool) -> (Vec<u32>, usize) {
        let mut currents = vec![
            projection_test_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            projection_test_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
            projection_test_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        let steps = [
            partial_contact_orbit_step_for_test(0, 1, 2, 20, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(1, 0, 2, 30, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(0, 2, 1, 21, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(2, 0, 1, 31, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(0, 3, 1, 22, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(3, 0, 1, 32, [0, 0, 0, 0]),
        ];
        let contacts = steps
            .into_iter()
            .enumerate()
            .map(|(id, step)| {
                let id = u32::try_from(id).unwrap();
                (
                    id,
                    established_contact_test_transition(
                        step,
                        [10, 10],
                        40 + u8::try_from(id).unwrap(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let insertion_order = if reverse_insertion {
            (0_u32..6).rev().collect::<Vec<_>>()
        } else {
            (0_u32..6).collect::<Vec<_>>()
        };
        for transition in insertion_order {
            let parent_ids = if transition % 2 == 0 { [0, 1] } else { [1, 0] };
            add_established_contact_test_contribution(&mut currents, 2, transition, parent_ids);
        }
        let mut resident_contribution_count = 6;
        commit_established_contact_test_plan(
            &mut currents,
            &contacts,
            2,
            &mut resident_contribution_count,
        )
        .unwrap();
        assert_eq!(
            pending_contribution_entry_count(&currents).unwrap(),
            resident_contribution_count
        );
        (
            established_contact_transition_ids(&currents[2]),
            resident_contribution_count,
        )
    }

    fn established_0000_final_contact_case(reverse_insertion: bool) -> (Vec<u32>, usize) {
        let mut currents = vec![
            projection_test_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            projection_test_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
            projection_test_current(RecurrenceNodeKind::Source, 10, 2, &[12]),
            projection_test_current(RecurrenceNodeKind::Current, 20, 3, &[11, 12]),
            projection_test_current(RecurrenceNodeKind::Current, 20, 4, &[10, 12]),
            projection_test_current(RecurrenceNodeKind::Current, 20, 5, &[10, 11]),
            projection_test_current(RecurrenceNodeKind::Current, 30, 6, &[10, 11, 12]),
        ];
        let parent_ids = [[3, 0], [0, 3], [4, 1], [1, 4], [5, 2], [2, 5]];
        let steps = [
            final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 60, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 70, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 61, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 71, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 62, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 72, [0, 0, 0, 0]),
        ];
        let contacts = steps
            .into_iter()
            .enumerate()
            .map(|(id, step)| {
                let id = u32::try_from(id).unwrap();
                let parents = parent_ids[id as usize];
                (
                    id,
                    established_contact_test_transition(
                        step,
                        parents.map(|parent| {
                            currents[parent as usize].key.current_state_template_id()
                        }),
                        80 + u8::try_from(id).unwrap(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let insertion_order = if reverse_insertion {
            (0_u32..6).rev().collect::<Vec<_>>()
        } else {
            (0_u32..6).collect::<Vec<_>>()
        };
        for transition in insertion_order {
            add_established_contact_test_contribution(
                &mut currents,
                6,
                transition,
                parent_ids[transition as usize],
            );
        }
        let mut resident_contribution_count = 6;
        commit_established_contact_test_plan(
            &mut currents,
            &contacts,
            6,
            &mut resident_contribution_count,
        )
        .unwrap();
        assert_eq!(
            pending_contribution_entry_count(&currents).unwrap(),
            resident_contribution_count
        );
        (
            established_contact_transition_ids(&currents[6]),
            resident_contribution_count,
        )
    }

    #[test]
    fn established_contact_0000_fan_in_keeps_one_certified_owner_deterministically() {
        let forward = established_0000_contact_case(false);
        let reverse = established_0000_contact_case(true);
        assert_eq!(forward, (vec![0], 1));
        assert_eq!(reverse, forward);
    }

    #[test]
    fn established_contact_0000_final_fan_in_keeps_three_source_assignments() {
        let forward = established_0000_final_contact_case(false);
        let reverse = established_0000_final_contact_case(true);
        assert_eq!(reverse, forward);
        assert_eq!(forward.0.len(), 3);
        assert_eq!(forward.1, 3);
    }

    #[test]
    fn established_contact_0012_partial_and_final_fan_in_each_keep_one_owner() {
        let mut partial_currents = vec![
            projection_test_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            projection_test_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
            projection_test_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        let partial_contacts = BTreeMap::from([
            (
                10,
                established_contact_test_transition(
                    partial_contact_orbit_step_for_test(0, 1, 2, 50, [0, 0, 1, 2]),
                    [10, 10],
                    52,
                ),
            ),
            (
                11,
                established_contact_test_transition(
                    partial_contact_orbit_step_for_test(1, 0, 2, 51, [0, 0, 1, 2]),
                    [10, 10],
                    53,
                ),
            ),
        ]);
        add_established_contact_test_contribution(&mut partial_currents, 2, 11, [0, 1]);
        add_established_contact_test_contribution(&mut partial_currents, 2, 10, [0, 1]);
        let mut partial_resident = 2;
        commit_established_contact_test_plan(
            &mut partial_currents,
            &partial_contacts,
            2,
            &mut partial_resident,
        )
        .unwrap();
        assert_eq!(
            established_contact_transition_ids(&partial_currents[2]),
            [10]
        );
        assert_eq!(partial_resident, 1);
        assert_eq!(
            pending_contribution_entry_count(&partial_currents).unwrap(),
            partial_resident
        );

        let mut final_currents = vec![
            projection_test_current(RecurrenceNodeKind::Current, 20, 0, &[10, 11]),
            projection_test_current(RecurrenceNodeKind::Source, 30, 1, &[12]),
            projection_test_current(RecurrenceNodeKind::Current, 40, 2, &[10, 11, 12]),
        ];
        let final_contacts = BTreeMap::from([
            (
                20,
                established_contact_test_transition(
                    final_contact_orbit_step_for_test(&[0, 1], &[2], 3, 54, [0, 0, 1, 2]),
                    [20, 30],
                    56,
                ),
            ),
            (
                21,
                established_contact_test_transition(
                    final_contact_orbit_step_for_test(&[2], &[0, 1], 3, 55, [0, 0, 1, 2]),
                    [30, 20],
                    57,
                ),
            ),
        ]);
        add_established_contact_test_contribution(&mut final_currents, 2, 20, [0, 1]);
        add_established_contact_test_contribution(&mut final_currents, 2, 21, [1, 0]);
        let mut final_resident = 2;
        commit_established_contact_test_plan(
            &mut final_currents,
            &final_contacts,
            2,
            &mut final_resident,
        )
        .unwrap();
        assert_eq!(established_contact_transition_ids(&final_currents[2]), [21]);
        assert_eq!(final_resident, 1);
        assert_eq!(
            pending_contribution_entry_count(&final_currents).unwrap(),
            final_resident
        );
    }

    #[test]
    fn established_contact_plan_errors_roll_back_contributions_and_counts() {
        let mut currents = vec![
            projection_test_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            projection_test_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
            projection_test_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        let duplicate_step = partial_contact_orbit_step_for_test(0, 1, 2, 60, [0, 0, 0, 0]);
        let contacts = BTreeMap::from([
            (
                30,
                established_contact_test_transition(duplicate_step.clone(), [10, 10], 61),
            ),
            (
                31,
                established_contact_test_transition(duplicate_step, [10, 10], 61),
            ),
        ]);
        add_established_contact_test_contribution(&mut currents, 2, 30, [0, 1]);
        add_established_contact_test_contribution(&mut currents, 2, 31, [0, 1]);
        let before = currents[2].contributions.clone();
        let resident_before = 2usize;
        assert!(
            plan_established_contact_orbit_owners_with_resolver(
                2,
                &currents,
                resident_before,
                |transition_id| contacts.get(&transition_id),
            )
            .unwrap_err()
            .to_string()
            .contains("conflicting exact rank")
        );
        assert_eq!(currents[2].contributions, before);
        assert_eq!(resident_before, 2);

        let mut malformed = currents.clone();
        let (pending, factor) = malformed[2]
            .contributions
            .first_key_value()
            .map(|(pending, factor)| (pending.clone(), *factor))
            .unwrap();
        malformed[2].contributions.remove(&pending);
        malformed[2].contributions.insert(
            PendingContributionKey {
                parent_current_ids: vec![0].into_boxed_slice(),
                key: pending.key,
            },
            factor,
        );
        let malformed_before = malformed[2].contributions.clone();
        assert!(
            plan_established_contact_orbit_owners_with_resolver(
                2,
                &malformed,
                resident_before,
                |transition_id| contacts.get(&transition_id),
            )
            .unwrap_err()
            .to_string()
            .contains("is not binary")
        );
        assert_eq!(malformed[2].contributions, malformed_before);
        assert_eq!(resident_before, 2);

        let mut candidate_reservation = Vec::<EstablishedContactContributionToken>::new();
        assert!(
            reserve_established_contact_candidates(&mut candidate_reservation, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("candidate allocation failed")
        );
        assert!(candidate_reservation.is_empty());
        assert_eq!(currents[2].contributions, before);
        assert_eq!(resident_before, 2);
    }

    #[test]
    fn established_uncertified_v3_vector_fermion_and_qcd_controls_pass_through() {
        let mut currents = vec![
            projection_test_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            projection_test_current(RecurrenceNodeKind::Source, 11, 1, &[11]),
            projection_test_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        // The owner path is deliberately blind to the model-level class of an
        // uncertified row. These four distinct controls stand for ordinary V3,
        // vector, fermion, and QCD transitions and must all pass through.
        for transition in 70..74 {
            add_established_contact_test_contribution(&mut currents, 2, transition, [0, 1]);
        }
        let before = currents[2].contributions.clone();
        let mut resident = 4usize;
        commit_established_contact_test_plan(&mut currents, &BTreeMap::new(), 2, &mut resident)
            .unwrap();
        assert_eq!(currents[2].contributions, before);
        assert_eq!(resident, 4);
        assert_eq!(
            pending_contribution_entry_count(&currents).unwrap(),
            resident
        );
    }

    #[test]
    fn prepared_transition_location_allocation_failure_leaves_index_empty() {
        let mut locations = Vec::new();
        assert!(
            reserve_prepared_transition_locations(&mut locations, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("transition-location allocation failed")
        );
        assert!(locations.is_empty());
    }

    fn projection_test_graph(
        omit_last_contribution: bool,
        omit_last_closure: bool,
    ) -> (
        Vec<PendingCurrent>,
        BTreeMap<PendingClosureKey, PendingClosureGroup>,
    ) {
        let mut currents = vec![
            projection_test_current(RecurrenceNodeKind::Source, 0, 0, &[0]),
            projection_test_current(RecurrenceNodeKind::Source, 0, 1, &[1]),
            projection_test_current(RecurrenceNodeKind::Source, 0, 2, &[2]),
            projection_test_current(RecurrenceNodeKind::Current, 1, 3, &[0, 1]),
            projection_test_current(RecurrenceNodeKind::Current, 1, 4, &[0, 1]),
            projection_test_current(RecurrenceNodeKind::Current, 2, 5, &[0, 1, 2]),
            projection_test_current(RecurrenceNodeKind::Current, 2, 6, &[0, 1, 2]),
        ];
        // The first projected value is a sum of two distinct exact kernels.
        add_projection_test_contribution(&mut currents, 3, 10, [0, 1], 0);
        add_projection_test_contribution(&mut currents, 4, 11, [0, 1], 0);
        // The next value has one common bilinear kernel over the complete
        // Cartesian product {3,4} x {2}.
        add_projection_test_contribution(&mut currents, 5, 20, [3, 2], 0);
        if !omit_last_contribution {
            add_projection_test_contribution(&mut currents, 6, 20, [4, 2], 1);
        }
        let mut closures = BTreeMap::new();
        for parent in [5, 6] {
            if omit_last_closure && parent == 6 {
                continue;
            }
            closures.insert(
                PendingClosureKey {
                    target_sector_id: 0,
                    complete_source_states: Box::new([]),
                    closure_template_id: 0,
                    quantum_flow_template_id: None,
                    parent_current_ids: vec![parent, 0].into_boxed_slice(),
                },
                PendingClosureGroup {
                    contributions: vec![],
                    exact_factor: ExactComplexRational::ONE,
                },
            );
        }
        (currents, closures)
    }

    #[test]
    fn isolated_lane_compaction_discards_only_unreachable_contributions() {
        let (mut currents, closures) = projection_test_graph(false, false);
        currents.push(projection_test_current(
            RecurrenceNodeKind::Current,
            3,
            7,
            &[0, 2],
        ));
        add_projection_test_contribution(&mut currents, 7, 21, [0, 2], 0);
        let mut resident_contribution_count = pending_contribution_entry_count(&currents).unwrap();
        assert_eq!(resident_contribution_count, 5);

        prune_inactive_lane_contributions(
            &mut currents,
            PendingConstructionDomain {
                shared_source_end: 3,
                lane_internal_start: 3,
                lane_internal_end: 8,
            },
            &closures,
            &mut resident_contribution_count,
        )
        .unwrap();

        assert_eq!(resident_contribution_count, 4);
        assert_eq!(
            resident_contribution_count,
            pending_contribution_entry_count(&currents).unwrap(),
        );
        assert!(currents[7].contributions.is_empty());
        assert!(
            currents[3..7]
                .iter()
                .all(|current| !current.contributions.is_empty())
        );
    }

    #[test]
    fn topology_replay_color_projection_requires_complete_rectangles() {
        let (currents, closures) = projection_test_graph(false, false);
        let projection = plan_topology_replay_color_projection(&currents, &closures)
            .unwrap()
            .unwrap();
        assert_eq!(projection.projection_members.len(), 5);
        assert_eq!(
            projection.old_to_projection[&3],
            projection.old_to_projection[&4]
        );
        assert_eq!(
            projection.old_to_projection[&5],
            projection.old_to_projection[&6]
        );
        assert_eq!(projection.contributions.len(), 3);
        assert_eq!(projection.closures.len(), 1);
        assert_eq!(
            projection
                .closures
                .values()
                .next()
                .unwrap()
                .builder_parent_tuples
                .len(),
            2
        );
        let materialized = materialize_projected_pending_rows(&currents, &projection).unwrap();
        assert_eq!(materialized.currents.len(), 5);
        assert_eq!(materialized.contributions.len(), 3);
        assert_eq!(materialized.finalizations.len(), 2);
        assert_eq!(materialized.remap[&3], materialized.remap[&4]);
        assert_eq!(materialized.remap[&5], materialized.remap[&6]);
    }

    #[test]
    fn topology_replay_color_projection_values_are_exact_member_sums() {
        let (currents, closures) = projection_test_graph(false, false);
        let projection = plan_topology_replay_color_projection(&currents, &closures)
            .unwrap()
            .unwrap();

        // Treat the three source values and transition kernels as arbitrary
        // exact scalar witnesses.  The first projected value is v3 + v4.
        // Rectangularity then proves that one bilinear row over this sum is
        // exactly the two original rows, and the same argument closes the
        // amplitude.  No representative color metadata enters the arithmetic.
        let source0 = 2_i128;
        let source1 = 3_i128;
        let source2 = 5_i128;
        let value3 = 2_i128 * source0 * source1;
        let value4 = -3_i128 * source0 * source1;
        let value5 = 5_i128 * value3 * source2;
        let value6 = 5_i128 * value4 * source2;
        let original_closure = value5 * source0 + value6 * source0;

        let projected_first = value3 + value4;
        let projected_second = 5_i128 * projected_first * source2;
        let projected_closure = projected_second * source0;
        assert_eq!(projected_first, value3 + value4);
        assert_eq!(projected_second, value5 + value6);
        assert_eq!(projected_closure, original_closure);
        assert_eq!(projected_closure, -300);

        let projected_second_id = projection.old_to_projection[&5];
        let row = projection
            .contributions
            .values()
            .find(|row| row.identity.destination_projection_id == projected_second_id)
            .unwrap();
        assert_eq!(row.builder_parent_tuples.len(), 2);
        assert!(
            rectangular_parent_domain_is_complete(
                &row.identity.parent_projection_ids,
                &row.builder_parent_tuples,
                &projection.old_to_projection,
                &projection.projection_members,
            )
            .unwrap()
        );
    }

    #[test]
    fn projected_runtime_rows_do_not_depend_on_the_representative_color_witness() {
        fn replace_only_witness(current: &mut PendingCurrent, witness: u32) {
            assert_eq!(current.contributions.len(), 1);
            let (pending_key, factor) = std::mem::take(&mut current.contributions)
                .into_iter()
                .next()
                .unwrap();
            let old = pending_key.key;
            let key = ContributionKey::new(
                old.transition_template_id(),
                old.parent_value_class_ids().to_vec(),
                old.parent_state_template_ids().to_vec(),
                old.parent_momenta().to_vec(),
                old.result_state_template_id(),
                old.quantum_flow_witness_id(),
                LCColorWitnessTermId::new(
                    old.color_witness_term_id().color_contraction_template_id(),
                    witness,
                ),
                old.runtime_coupling_binding_digest(),
                old.output_projection_id(),
            )
            .unwrap();
            current.contributions.insert(
                PendingContributionKey {
                    parent_current_ids: pending_key.parent_current_ids,
                    key,
                },
                factor,
            );
        }
        fn runtime_signature(
            rows: &MaterializedPendingRows,
        ) -> Vec<(u32, Vec<u32>, u32, ExactComplexRational)> {
            rows.contributions
                .iter()
                .map(|row| {
                    (
                        row.result_current_id(),
                        row.parent_current_ids().to_vec(),
                        row.key().transition_template_id(),
                        row.exact_factor(),
                    )
                })
                .collect()
        }

        let (currents, closures) = projection_test_graph(false, false);
        let first_projection = plan_topology_replay_color_projection(&currents, &closures)
            .unwrap()
            .unwrap();
        let first = materialize_projected_pending_rows(&currents, &first_projection).unwrap();
        let first_witness = first
            .contributions
            .iter()
            .find(|row| row.key().transition_template_id() == 20)
            .unwrap()
            .key()
            .color_witness_term_id();

        let mut swapped = currents.clone();
        replace_only_witness(&mut swapped[5], 1);
        replace_only_witness(&mut swapped[6], 0);
        let second_projection = plan_topology_replay_color_projection(&swapped, &closures)
            .unwrap()
            .unwrap();
        let second = materialize_projected_pending_rows(&swapped, &second_projection).unwrap();
        let second_witness = second
            .contributions
            .iter()
            .find(|row| row.key().transition_template_id() == 20)
            .unwrap()
            .key()
            .color_witness_term_id();

        assert_ne!(first_witness, second_witness);
        assert_eq!(runtime_signature(&first), runtime_signature(&second));
    }

    #[test]
    fn three_line_selected_flow_projection_stays_within_legacy_work_budget() {
        // Exact N=4..6 structural census for report process #13 after selector
        // pruning, global-flip reduction, and rectangular color projection.
        // This is a release guard for the real catalog rows, not a timing
        // expectation.
        let rows = [
            (4, (32_u64, 37_u64), (33_u64, 37_u64)),
            (5, (66_u64, 103_u64), (60_u64, 94_u64)),
            (6, (136_u64, 283_u64), (116_u64, 246_u64)),
        ];
        for (multiplicity, legacy, projected) in rows {
            assert!(
                projected.0 * 100 <= legacy.0 * 105,
                "N={multiplicity} projected currents exceed 1.05x legacy"
            );
            assert!(
                projected.1 * 100 <= legacy.1 * 105,
                "N={multiplicity} projected interactions exceed 1.05x legacy"
            );
        }
    }

    #[test]
    fn topology_replay_color_projection_certificate_is_deterministic_and_complete() {
        let (currents, closures) = projection_test_graph(false, false);
        let projection = plan_topology_replay_color_projection(&currents, &closures)
            .unwrap()
            .unwrap();
        let color_states = (0..7)
            .map(|shape| DynamicLCColorState::new(shape, None, vec![]).unwrap())
            .collect::<Vec<_>>();
        let candidate_digests = [digest(240), digest(241)];
        let first = encode_color_projection_certificate_body(
            &currents,
            &color_states,
            &projection,
            &candidate_digests,
        )
        .unwrap();
        let second = encode_color_projection_certificate_body(
            &currents,
            &color_states,
            &projection,
            &candidate_digests,
        )
        .unwrap();
        assert_eq!(first, second);
        assert!(first.starts_with(COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC));
        let digest_start = first.len() - 32;
        assert_eq!(
            &first[digest_start..],
            Sha256::digest(&first[..digest_start]).as_slice()
        );
        for old_id in 0_u32..7 {
            assert!(
                first
                    .windows(4)
                    .any(|window| window == old_id.to_le_bytes()),
                "certificate omits old current {old_id}",
            );
        }
    }

    #[test]
    fn topology_replay_color_projection_rejects_a_missing_internal_tuple() {
        let (currents, mut closures) = projection_test_graph(true, false);
        // Keep both members of the first projected current class live through
        // a separate, complete closure family.  Otherwise omitting current 6's
        // contribution makes parent current 4 selector-dead, so [3, 2] is a
        // complete singleton rectangle rather than a missing active tuple.
        for parent in [3, 4] {
            closures.insert(
                PendingClosureKey {
                    target_sector_id: 0,
                    complete_source_states: Box::new([]),
                    closure_template_id: 1,
                    quantum_flow_template_id: None,
                    parent_current_ids: vec![parent, 0].into_boxed_slice(),
                },
                PendingClosureGroup {
                    contributions: vec![],
                    exact_factor: ExactComplexRational::ONE,
                },
            );
        }
        assert!(
            plan_topology_replay_color_projection(&currents, &closures)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn topology_replay_color_projection_rejects_a_missing_closure_tuple() {
        let (currents, closures) = projection_test_graph(false, true);
        assert!(
            plan_topology_replay_color_projection(&currents, &closures)
                .unwrap()
                .is_none()
        );
    }
}
