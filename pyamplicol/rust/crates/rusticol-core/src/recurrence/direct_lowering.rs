// SPDX-License-Identifier: 0BSD

//! Deterministic lowering from a validated semantic recurrence program to the
//! direct-arena layout.
//!
//! This is deliberately a lowering-only slice. It creates immutable row-group
//! descriptors and direct-arena rows; it does not construct or invoke a
//! prepared runtime backend.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use super::arena::{DirectArenaLayout, recurrence_direct_arena_layout};
use super::direct_helicity_dispatch::{
    DirectHelicityDispatch, DirectHelicityDispatchDescriptor, DirectHelicityDispatchParts,
    DirectHelicityRowGroupDescriptor, DirectHelicitySupportDomainDescriptor,
};
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
    DIRECT_NONE_U32, DirectAmplitudeDestinationDescriptor, DirectClosureRow, DirectContributionRow,
    DirectCurrentDescriptor, DirectDestinationOperation, DirectExecutorRole, DirectFinalizationRow,
    DirectMomentumFormDescriptor, DirectMomentumTerm, DirectNodeKind, DirectRecurrencePlan,
    DirectRecurrencePlanParts, DirectReplayTargetDescriptor, DirectResolvedHelicityDescriptor,
    DirectResolvedSourceSelection, DirectRowGroupDescriptor, DirectSelectorDomainDescriptor,
    DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow, DirectSourceProjectionRow,
    DirectSourceRow, DirectSourceStateAssignment,
};
use super::helicity_support::{
    AllFlowHelicityMask, AllFlowHelicitySupportProjection, project_all_flow_helicity_support,
};
use super::layout::RuntimeSourceVariantBinding;
use super::relation::{
    RecurrenceCurrentRelationCertificate, RecurrenceRelationDiscoveryMode,
    RecurrenceRelationDiscoveryOptions, RecurrenceRelationDiscoveryOutcome,
    RecurrenceRelationDiscoveryReport, RelationCurrentContract, RelationCurrentInput,
    RelationTermInput, discover_recurrence_current_relations, relation_certificate_algorithm,
};
use super::template::{
    EvaluatorContractKind, MISSING_U32, OwnedRecurrenceTemplateInput, RuntimeHelicityVariantRow,
    ValidatedRecurrenceTemplateInput,
};
use super::{
    CanonicalMomentumLinearForm, ClosureProofMetadataV2, CurrentSourceBinding,
    ExactComplexRational, RecurrenceCurrent, RecurrenceNodeKind, RecurrenceProgram,
    RecurrenceStrategy, SemanticDigest, SourceStateAssignment, closure_component_factor_digest_v2,
    closure_selector_domain_digest_v2,
};
use crate::{RusticolError, RusticolResult};

const UNIVERSAL_SELECTOR_DOMAIN_ID: u32 = 0;
const DIRECT_ROW_FLAGS_NONE: u32 = 0;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("recurrence direct lowering: {}", message.into()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|character| character.is_ascii_digit() || (b'a'..=b'f').contains(&character))
}

/// Runtime sizing recorded in the immutable direct plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectRecurrenceRuntimeOptions {
    pub point_tile_size: u32,
    pub workspace_mib: u32,
}

impl DirectRecurrenceRuntimeOptions {
    pub fn new(point_tile_size: u32, workspace_mib: u32) -> RusticolResult<Self> {
        if point_tile_size == 0 {
            return Err(invalid("point tile size must be positive"));
        }
        if workspace_mib == 0 {
            return Err(invalid("workspace MiB must be positive"));
        }
        Ok(Self {
            point_tile_size,
            workspace_mib,
        })
    }
}

/// Stable semantic key exported by prepared direct-template-v1.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PreparedDirectExecutorKey {
    Evaluator {
        role: DirectExecutorRole,
        evaluator_binding_id: u32,
    },
    IdentityFinalizer,
}

impl PreparedDirectExecutorKey {
    const fn role(self) -> DirectExecutorRole {
        match self {
            Self::Evaluator { role, .. } => role,
            Self::IdentityFinalizer => DirectExecutorRole::Finalization,
        }
    }
}

/// One authenticated prepared direct-template mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedDirectExecutorBinding {
    pub key: PreparedDirectExecutorKey,
    pub direct_executor_id: u32,
    pub parent_permutation: [u8; 2],
}

impl PreparedDirectExecutorBinding {
    pub const fn evaluator(
        role: DirectExecutorRole,
        evaluator_binding_id: u32,
        direct_executor_id: u32,
    ) -> Self {
        Self {
            key: PreparedDirectExecutorKey::Evaluator {
                role,
                evaluator_binding_id,
            },
            direct_executor_id,
            parent_permutation: [0, 1],
        }
    }

    pub const fn evaluator_with_parent_permutation(
        role: DirectExecutorRole,
        evaluator_binding_id: u32,
        direct_executor_id: u32,
        parent_permutation: [u8; 2],
    ) -> Self {
        Self {
            key: PreparedDirectExecutorKey::Evaluator {
                role,
                evaluator_binding_id,
            },
            direct_executor_id,
            parent_permutation,
        }
    }

    pub const fn identity_finalizer(direct_executor_id: u32) -> Self {
        Self {
            key: PreparedDirectExecutorKey::IdentityFinalizer,
            direct_executor_id,
            parent_permutation: [0, 1],
        }
    }
}

/// Authenticated model-level direct executor resolver.
///
/// IDs are the globally stable dense IDs owned by the complete loaded
/// direct-template catalog. Lowering never compacts or renumbers a
/// process-specific subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedDirectExecutorCatalog {
    direct_template_catalog_digest: SemanticDigest,
    bindings: BTreeMap<PreparedDirectExecutorKey, PreparedDirectExecutorBinding>,
    executor_roles: Box<[DirectExecutorRole]>,
}

impl PreparedDirectExecutorCatalog {
    pub fn new(
        direct_template_catalog_digest: SemanticDigest,
        bindings: Vec<PreparedDirectExecutorBinding>,
    ) -> RusticolResult<Self> {
        if bindings.is_empty() {
            return Err(invalid(
                "prepared direct-template executor catalog must not be empty",
            ));
        }
        let mut by_key = BTreeMap::new();
        let mut roles_by_id = BTreeMap::new();
        for binding in bindings {
            if matches!(
                binding.key,
                PreparedDirectExecutorKey::Evaluator {
                    evaluator_binding_id: MISSING_U32,
                    ..
                }
            ) {
                return Err(invalid(
                    "prepared direct-template mapping uses the missing evaluator-binding sentinel",
                ));
            }
            if !matches!(binding.parent_permutation, [0, 1] | [1, 0])
                || (binding.parent_permutation != [0, 1]
                    && binding.key.role() != DirectExecutorRole::Contribution)
            {
                return Err(invalid(
                    "prepared direct-template mapping has an invalid parent permutation",
                ));
            }
            if by_key.insert(binding.key, binding).is_some() {
                return Err(invalid(format!(
                    "prepared direct-template mapping repeats key {:?}",
                    binding.key
                )));
            }
            let role = binding.key.role();
            if let Some(previous) = roles_by_id.insert(binding.direct_executor_id, role)
                && previous != role
            {
                return Err(invalid(format!(
                    "prepared direct executor {} is assigned both {previous:?} and {role:?} roles",
                    binding.direct_executor_id
                )));
            }
        }
        for (expected, actual) in roles_by_id.keys().copied().enumerate() {
            if actual != u32_len("prepared direct executor", expected)? {
                return Err(invalid(format!(
                    "prepared direct executor IDs are not dense zero-based at {actual}"
                )));
            }
        }
        let executor_roles = roles_by_id
            .into_values()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            direct_template_catalog_digest,
            bindings: by_key,
            executor_roles,
        })
    }

    pub const fn direct_template_catalog_digest(&self) -> SemanticDigest {
        self.direct_template_catalog_digest
    }

    pub fn direct_executor_count(&self) -> u32 {
        self.executor_roles.len() as u32
    }

    pub fn resolve_evaluator(
        &self,
        role: DirectExecutorRole,
        evaluator_binding_id: u32,
    ) -> RusticolResult<u32> {
        let key = PreparedDirectExecutorKey::Evaluator {
            role,
            evaluator_binding_id,
        };
        if let Some(binding) = self.bindings.get(&key) {
            return Ok(binding.direct_executor_id);
        }
        if self.bindings.keys().any(|key| {
            matches!(
                key,
                PreparedDirectExecutorKey::Evaluator {
                    evaluator_binding_id: candidate,
                    ..
                } if *candidate == evaluator_binding_id
            )
        }) {
            return Err(invalid(format!(
                "prepared direct-template mapping for evaluator binding {evaluator_binding_id} has a role mismatch; expected {role:?}"
            )));
        }
        Err(invalid(format!(
            "prepared direct-template catalog has no {role:?} mapping for evaluator binding {evaluator_binding_id}"
        )))
    }

    pub fn resolve_contribution(
        &self,
        evaluator_binding_id: u32,
    ) -> RusticolResult<(u32, [u8; 2])> {
        let key = PreparedDirectExecutorKey::Evaluator {
            role: DirectExecutorRole::Contribution,
            evaluator_binding_id,
        };
        let binding = self.bindings.get(&key).ok_or_else(|| {
            invalid(format!(
                "prepared direct-template catalog has no Contribution mapping for evaluator binding {evaluator_binding_id}"
            ))
        })?;
        Ok((binding.direct_executor_id, binding.parent_permutation))
    }

    pub fn resolve_identity_finalizer(&self) -> RusticolResult<u32> {
        self.bindings
            .get(&PreparedDirectExecutorKey::IdentityFinalizer)
            .map(|binding| binding.direct_executor_id)
            .ok_or_else(|| {
                invalid(
                    "prepared direct-template catalog has no generic identity-finalizer binding",
                )
            })
    }
}

#[derive(Clone, Copy, Debug)]
struct SourceDraft {
    stage: u16,
    executor_id: u32,
    semantic_current_id: u32,
    row: DirectSourceRow,
}

#[derive(Clone, Copy, Debug)]
struct ContributionDraft {
    stage: u16,
    executor_id: u32,
    semantic_contribution_id: u32,
    semantic_result_current_id: u32,
    semantic_parent_current_ids: [u32; 2],
    semantic_parent_count: u8,
    row: DirectContributionRow,
}

/// Authenticated runtime inputs which determine one contribution callable's
/// kinematic result before its row-local destination/factor application.
///
/// The prepared executor ID fixes the callable, component layouts, coupling,
/// and model-parameter projections. Physical current/momentum coordinates fix
/// every remaining kinematic operand. Exact factors are ordered inside this
/// key rather than being part of it: intrinsic output-only executors may fan
/// out across factors, while prepared callables conservatively split the
/// adjacent run by factor at load time when the factor is a kernel input.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContributionFanoutOrderKey {
    stage: u32,
    executor_id: u32,
    selector_domain_id: u32,
    parent0_component_base: u32,
    parent1_component_base_or_sentinel: u32,
    parent0_momentum_form_id: u32,
    parent1_momentum_form_id_or_sentinel: u32,
}

fn contribution_fanout_order_key_parts(
    stage: u32,
    executor_id: u32,
    row: DirectContributionRow,
) -> ContributionFanoutOrderKey {
    ContributionFanoutOrderKey {
        stage,
        executor_id,
        selector_domain_id: row.selector_domain_id,
        parent0_component_base: row.parent0_component_base,
        parent1_component_base_or_sentinel: row.parent1_component_base_or_sentinel,
        parent0_momentum_form_id: row.parent0_momentum_form_id,
        parent1_momentum_form_id_or_sentinel: row.parent1_momentum_form_id_or_sentinel,
    }
}

#[cfg(test)]
fn contribution_fanout_order_key(draft: &ContributionDraft) -> ContributionFanoutOrderKey {
    contribution_fanout_order_key_parts(u32::from(draft.stage), draft.executor_id, draft.row)
}

/// Place repeated kinematic inputs before singleton inputs and make every
/// reusable class contiguous. This is a deliberate deterministic change to
/// floating-point accumulation order, not a physics approximation: each
/// destination still receives every exact-factor-scaled contribution once,
/// with its first row initialized and all later rows added. Exact/high-
/// precision execution remains driven by the semantic schedule.
/// Certified copies retain their incoming topological order within each
/// stage/executor/domain: recycled arena addresses are not dependency ranks.
pub(crate) fn order_contributions_for_runtime_fanout_by<T, Tie>(
    drafts: &mut [T],
    parts: impl Fn(&T) -> (u32, u32, DirectContributionRow, Tie),
) where
    Tie: Ord,
{
    let mut multiplicity = BTreeMap::<ContributionFanoutOrderKey, usize>::new();
    for draft in drafts.iter() {
        let (stage, executor_id, row, _) = parts(draft);
        if executor_id != DIRECT_NONE_U32
            && row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0
        {
            *multiplicity
                .entry(contribution_fanout_order_key_parts(stage, executor_id, row))
                .or_default() += 1;
        }
    }
    drafts.sort_by_key(|draft| {
        let (stage, executor_id, row, tie) = parts(draft);
        let key = contribution_fanout_order_key_parts(stage, executor_id, row);
        let certified_copy = row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0;
        let reusable = executor_id != DIRECT_NONE_U32
            && !certified_copy
            && multiplicity.get(&key).copied().unwrap_or(0) > 1;
        (
            stage,
            executor_id,
            row.selector_domain_id,
            !reusable,
            (!certified_copy).then_some((
                key.parent0_component_base,
                key.parent1_component_base_or_sentinel,
                key.parent0_momentum_form_id,
                key.parent1_momentum_form_id_or_sentinel,
                row.exact_factor_id,
                tie,
            )),
        )
    });
}

fn order_contributions_for_runtime_fanout(drafts: &mut [ContributionDraft]) {
    order_contributions_for_runtime_fanout_by(drafts, |draft| {
        (
            u32::from(draft.stage),
            draft.executor_id,
            draft.row,
            draft.semantic_contribution_id,
        )
    });
}

/// Temporary generation-time locality information. Only the resulting row
/// order is persisted: loading or selecting helicities does not run this pass.
#[derive(Debug)]
struct FixedHelicityDemands {
    word_count: usize,
    words: Vec<u64>,
}

impl FixedHelicityDemands {
    fn current(&self, current_id: u32) -> &[u64] {
        let start = current_id as usize * self.word_count;
        &self.words[start..start + self.word_count]
    }
}

fn fixed_helicity_demands(
    current_count: usize,
    helicity_count: usize,
    contributions: &[ContributionDraft],
    seeds: impl Iterator<Item = (u32, u32)>,
) -> RusticolResult<FixedHelicityDemands> {
    let word_count = helicity_count.div_ceil(64);
    let scalar_count = current_count
        .checked_mul(word_count)
        .ok_or_else(|| invalid("fixed-helicity locality mask size overflows usize"))?;
    let mut demands = FixedHelicityDemands {
        word_count,
        words: vec![0; scalar_count],
    };
    for (current_id, helicity_id) in seeds {
        if current_id as usize >= current_count || helicity_id as usize >= helicity_count {
            return Err(invalid(
                "fixed-helicity locality seed is outside the program",
            ));
        }
        demands.words[current_id as usize * word_count + helicity_id as usize / 64] |=
            1_u64 << (helicity_id % 64);
    }

    // Use the actual lowered edges, including certified numerical copies, not
    // the original source-state ancestry. Flat linked row lists avoid making
    // a second collection of all parent IDs or allocating one list per current.
    let mut first_contribution = vec![usize::MAX; current_count];
    let mut next_contribution = vec![usize::MAX; contributions.len()];
    for (row_id, draft) in contributions.iter().enumerate() {
        let result = draft.semantic_result_current_id as usize;
        if result >= current_count {
            return Err(invalid(
                "fixed-helicity locality result is outside the program",
            ));
        }
        let parents = draft
            .semantic_parent_current_ids
            .get(..usize::from(draft.semantic_parent_count))
            .ok_or_else(|| invalid("fixed-helicity locality parent count is invalid"))?;
        if parents.iter().any(|&parent| {
            parent as usize > result
                || (parent as usize == result
                    && draft.row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0)
        }) {
            return Err(invalid(
                "fixed-helicity locality requires topologically ordered current dependencies",
            ));
        }
        next_contribution[row_id] = first_contribution[result];
        first_contribution[result] = row_id;
    }
    for result in (0..current_count).rev() {
        let mut row_id = first_contribution[result];
        let (earlier, current_and_later) = demands.words.split_at_mut(result * word_count);
        let result_words = &current_and_later[..word_count];
        while row_id != usize::MAX {
            let draft = &contributions[row_id];
            for &parent in
                &draft.semantic_parent_current_ids[..usize::from(draft.semantic_parent_count)]
            {
                // A certified zero current may keep a self-copy row. It
                // introduces no additional dependency or demanded helicity.
                if parent as usize == result {
                    continue;
                }
                let start = parent as usize * word_count;
                for (word, demanded) in earlier[start..start + word_count]
                    .iter_mut()
                    .zip(result_words)
                {
                    *word |= *demanded;
                }
            }
            row_id = next_contribution[row_id];
        }
    }
    Ok(demands)
}

fn topology_replay_helicity_demands(
    program: &RecurrenceProgram,
    contributions: &[ContributionDraft],
) -> RusticolResult<Option<FixedHelicityDemands>> {
    if program.strategy() != RecurrenceStrategy::TopologyReplay
        || program.resolved_helicities().len() <= 1
        || program
            .amplitude_destinations()
            .iter()
            .any(|destination| destination.target_helicity_id().is_none())
    {
        return Ok(None);
    }
    let seeds = program.closure_terms().iter().flat_map(|closure| {
        let helicity = program.amplitude_destinations()[closure.target_destination_id() as usize]
            .target_helicity_id()
            .expect("validated topology-replay destinations specify their helicity");
        closure
            .parent_current_ids()
            .iter()
            .map(move |&current| (current, helicity))
    });
    fixed_helicity_demands(
        program.currents().len(),
        program.resolved_helicities().len(),
        contributions,
        seeds,
    )
    .map(Some)
}

/// Improve selected-helicity locality without splitting an existing reusable
/// parent-input class. This follows the ordinary fanout ordering and leaves
/// each class's exact-factor/destination order intact. Same-stage certified
/// copies are left in their existing order to preserve copy dependencies.
fn order_contributions_for_helicity_locality(
    drafts: &mut [ContributionDraft],
    demands: &FixedHelicityDemands,
) {
    let mut group_start = 0;
    while group_start < drafts.len() {
        let first = &drafts[group_start];
        let primary = (first.stage, first.executor_id, first.row.selector_domain_id);
        let mut group_end = group_start + 1;
        while group_end < drafts.len() {
            let next = &drafts[group_end];
            if (next.stage, next.executor_id, next.row.selector_domain_id) != primary {
                break;
            }
            group_end += 1;
        }
        let group = &mut drafts[group_start..group_end];
        if primary.1 != DIRECT_NONE_U32
            && !group
                .iter()
                .any(|draft| draft.row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0)
        {
            let mut blocks = Vec::new();
            let mut start = 0;
            while start < group.len() {
                let first = &group[start];
                let key = contribution_fanout_order_key_parts(
                    u32::from(first.stage),
                    first.executor_id,
                    first.row,
                );
                let mut end = start + 1;
                while end < group.len()
                    && contribution_fanout_order_key_parts(
                        u32::from(group[end].stage),
                        group[end].executor_id,
                        group[end].row,
                    ) == key
                {
                    end += 1;
                }
                let mut support = Cow::Borrowed(demands.current(first.semantic_result_current_id));
                for draft in &group[start + 1..end] {
                    let next = demands.current(draft.semantic_result_current_id);
                    if support.as_ref() != next {
                        for (word, demanded) in support.to_mut().iter_mut().zip(next) {
                            *word |= *demanded;
                        }
                    }
                }
                blocks.push((start..end, support));
                start = end;
            }
            // Preserve the original repeated-input-before-singleton policy.
            // Stable sorting retains the old order within equal support masks.
            if !blocks.is_sorted_by(|left, right| {
                (left.0.len() == 1, &left.1) <= (right.0.len() == 1, &right.1)
            }) {
                blocks.sort_by(|left, right| {
                    (left.0.len() == 1)
                        .cmp(&(right.0.len() == 1))
                        .then_with(|| left.1.cmp(&right.1))
                });
                let reordered = blocks
                    .iter()
                    .flat_map(|(range, _)| group[range.clone()].iter().copied())
                    .collect::<Vec<_>>();
                group.copy_from_slice(&reordered);
            }
        }
        group_start = group_end;
    }
}

#[derive(Clone, Copy, Debug)]
struct FinalizationDraft {
    stage: u16,
    executor_id: u32,
    semantic_current_id: u32,
    row: DirectFinalizationRow,
}

#[derive(Clone, Copy, Debug)]
struct ClosureDraft {
    stage: u16,
    executor_id: u32,
    semantic_closure_id: u32,
    amplitude_destination_id: u32,
    row: DirectClosureRow,
}

#[derive(Debug)]
struct LoweredHelicityRowSupport {
    resolved_helicity_count: u32,
    contributions: Vec<Vec<u64>>,
    finalizations: Vec<Vec<u64>>,
    closures: Vec<Vec<u64>>,
}

struct LoweredSelectorDomains {
    descriptors: Vec<DirectSelectorDomainDescriptor>,
    words: Vec<u64>,
    current_domain_ids: Vec<u32>,
    closure_domain_ids: Vec<u32>,
    destination_domain_ids: Vec<u32>,
    sector_domain_ids: Vec<Option<u32>>,
}

fn canonical_selector_words(domain: &[u64]) -> &[u64] {
    let mut retained = domain
        .iter()
        .rposition(|word| *word != 0)
        .map_or(0, |index| index + 1);
    // Direct-plan v2 reserves the one-word [u64::MAX] encoding for the
    // universal selector.  For a wider physical-sector domain, a legitimate
    // selector containing only sectors 0..63 must retain one trailing zero so
    // it cannot alias that sentinel.
    if retained == 1 && domain.len() > 1 && domain[0] == u64::MAX {
        retained = 2;
    }
    &domain[..retained]
}

fn lower_selector_domains(program: &RecurrenceProgram) -> RusticolResult<LoweredSelectorDomains> {
    // Keep the historical universal row first.  Besides making complete-color
    // plans byte-for-byte stable, this is the backwards-compatible
    // direct-plan-v2 encoding recognized by the runtime.
    let mut descriptors = vec![DirectSelectorDomainDescriptor {
        word_start: 0,
        word_count: 1,
    }];
    let mut words = vec![u64::MAX];
    if !program.strategy().uses_topology_replay_targets() || program.replay_targets().is_empty() {
        return Ok(LoweredSelectorDomains {
            descriptors,
            words,
            current_domain_ids: vec![UNIVERSAL_SELECTOR_DOMAIN_ID; program.currents().len()],
            closure_domain_ids: vec![UNIVERSAL_SELECTOR_DOMAIN_ID; program.closure_terms().len()],
            destination_domain_ids: vec![
                UNIVERSAL_SELECTOR_DOMAIN_ID;
                program.amplitude_destinations().len()
            ],
            sector_domain_ids: vec![
                Some(UNIVERSAL_SELECTOR_DOMAIN_ID);
                program.physical_sector_count() as usize
            ],
        });
    }

    let sector_count = program.physical_sector_count() as usize;
    let word_count = sector_count
        .checked_add(u64::BITS as usize - 1)
        .ok_or_else(|| invalid("selector-domain sector count overflows usize"))?
        / u64::BITS as usize;
    if word_count == 0 {
        return Err(invalid(
            "topology-replay selector domains require physical sectors",
        ));
    }
    let mut current_words = vec![vec![0_u64; word_count]; program.currents().len()];
    let mut closure_words = Vec::with_capacity(program.closure_terms().len());
    for closure in program.closure_terms() {
        let destination = program
            .amplitude_destinations()
            .get(closure.target_destination_id() as usize)
            .ok_or_else(|| invalid("closure selector destination is absent"))?;
        let sector_id = destination.target_sector_id() as usize;
        if sector_id >= sector_count {
            return Err(invalid("closure selector sector is out of bounds"));
        }
        let mut domain = vec![0_u64; word_count];
        domain[sector_id / u64::BITS as usize] |= 1_u64 << (sector_id % u64::BITS as usize);
        for parent_id in closure.parent_current_ids() {
            let parent = current_words
                .get_mut(*parent_id as usize)
                .ok_or_else(|| invalid("closure selector parent current is absent"))?;
            for (parent_word, sector_word) in parent.iter_mut().zip(&domain) {
                *parent_word |= *sector_word;
            }
        }
        closure_words.push(domain);
    }

    // Current IDs are topological.  Walking backwards propagates every
    // selected closure sector through exactly the parent currents needed to
    // construct it, retaining sharing as the union of dependent sectors.
    for current in program.currents().iter().rev() {
        let domain = current_words
            .get(current.id() as usize)
            .ok_or_else(|| invalid("selector current is absent"))?
            .clone();
        // Builder programs may retain structurally valid dead currents.  Keep
        // their fixed-width all-zero domain: full execution remains
        // byte-for-byte compatible, while selected replay can omit them.
        let contributions = current.contribution_range().as_usize_range(
            program.contributions().len(),
            "selector current contribution",
        )?;
        for contribution in &program.contributions()[contributions] {
            for parent_id in contribution.parent_current_ids() {
                let parent = current_words
                    .get_mut(*parent_id as usize)
                    .ok_or_else(|| invalid("selector contribution parent current is absent"))?;
                for (parent_word, current_word) in parent.iter_mut().zip(&domain) {
                    *parent_word |= *current_word;
                }
            }
        }
    }

    let mut interned = BTreeMap::<Vec<u64>, u32>::new();
    interned.insert(vec![u64::MAX], UNIVERSAL_SELECTOR_DOMAIN_ID);
    let mut intern = |domain: &[u64]| -> RusticolResult<u32> {
        let domain = canonical_selector_words(domain);
        if let Some(id) = interned.get(domain) {
            return Ok(*id);
        }
        let id = u32::try_from(descriptors.len())
            .map_err(|_| invalid("selector-domain count exceeds u32"))?;
        let word_start = u64::try_from(words.len())
            .map_err(|_| invalid("selector-domain word table exceeds u64"))?;
        let word_count = u32::try_from(domain.len())
            .map_err(|_| invalid("selector-domain word count exceeds u32"))?;
        words.extend_from_slice(domain);
        descriptors.push(DirectSelectorDomainDescriptor {
            word_start,
            word_count,
        });
        interned.insert(domain.to_vec(), id);
        Ok(id)
    };

    let referenced_sector_ids = program
        .amplitude_destinations()
        .iter()
        .map(|destination| destination.target_sector_id())
        .chain(
            program
                .replay_targets()
                .iter()
                .map(|target| target.materialized_sector_id()),
        )
        .collect::<BTreeSet<_>>();
    let mut sector_domain_ids = vec![None; sector_count];
    for sector_id in referenced_sector_ids {
        let sector_index = sector_id as usize;
        if sector_index >= sector_count {
            return Err(invalid("referenced selector sector is out of bounds"));
        }
        let mut domain = vec![0_u64; word_count];
        domain[sector_index / u64::BITS as usize] |= 1_u64 << (sector_index % u64::BITS as usize);
        sector_domain_ids[sector_index] = Some(intern(&domain)?);
    }
    let current_domain_ids = current_words
        .iter()
        .map(|domain| intern(domain))
        .collect::<RusticolResult<Vec<_>>>()?;
    let closure_domain_ids = closure_words
        .iter()
        .map(|domain| intern(domain))
        .collect::<RusticolResult<Vec<_>>>()?;
    let destination_domain_ids = program
        .amplitude_destinations()
        .iter()
        .map(|destination| {
            sector_domain_ids
                .get(destination.target_sector_id() as usize)
                .copied()
                .flatten()
                .ok_or_else(|| invalid("amplitude selector sector is out of bounds"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    Ok(LoweredSelectorDomains {
        descriptors,
        words,
        current_domain_ids,
        closure_domain_ids,
        destination_domain_ids,
        sector_domain_ids,
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DirectParents {
    pub(super) parent0_component_base: u32,
    pub(super) parent1_component_base_or_sentinel: u32,
    pub(super) parent0_momentum_form_id: u32,
    pub(super) parent1_momentum_form_id_or_sentinel: u32,
}

impl DirectParents {
    pub(super) fn permuted(self, permutation: [u8; 2]) -> RusticolResult<Self> {
        match permutation {
            [0, 1] => Ok(self),
            [1, 0] if self.parent1_component_base_or_sentinel != DIRECT_NONE_U32 => Ok(Self {
                parent0_component_base: self.parent1_component_base_or_sentinel,
                parent1_component_base_or_sentinel: self.parent0_component_base,
                parent0_momentum_form_id: self.parent1_momentum_form_id_or_sentinel,
                parent1_momentum_form_id_or_sentinel: self.parent0_momentum_form_id,
            }),
            [1, 0] => Err(invalid(
                "binary contribution parent permutation references an absent second parent",
            )),
            _ => Err(invalid(
                "direct contribution has an invalid parent permutation",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RuntimeSourceChoice {
    pub(super) source_state_index: u32,
    pub(super) public_helicity: i32,
    pub(super) dispatch_variant_id: u32,
}

#[derive(Debug, Default)]
struct LoweredRuntimeSources {
    variants: Vec<DirectSourceDispatchVariantDescriptor>,
    embeddings: Vec<DirectSourceEmbeddingRow>,
    projections: Vec<DirectSourceProjectionRow>,
    choices_by_source: Vec<Vec<RuntimeSourceChoice>>,
}

/// Lower one validated recurrence program to validated plan-v2 parts.
///
/// `semantic_digest` authenticates the process semantic plan.
/// `prepared_pack_digest` must authenticate the prepared pack whose evaluator
/// bindings are present in `templates`. `direct_template_catalog_digest`
/// authenticates `direct_executors`.
pub fn lower_recurrence_direct_v2(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    semantic_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
) -> RusticolResult<DirectRecurrencePlanParts> {
    let (parts, _, _) = build_direct_parts(
        program,
        templates,
        direct_executors,
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        runtime_options,
        &RecurrenceRelationDiscoveryOptions::off(),
        None,
    )?;
    DirectRecurrencePlan::new(parts).map(DirectRecurrencePlan::into_parts)
}

/// Lower one validated recurrence program directly to an immutable plan.
pub fn lower_recurrence_direct_plan_v2(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    semantic_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
) -> RusticolResult<DirectRecurrencePlan> {
    let (parts, _, _) = build_direct_parts(
        program,
        templates,
        direct_executors,
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        runtime_options,
        &RecurrenceRelationDiscoveryOptions::off(),
        None,
    )?;
    DirectRecurrencePlan::new(parts)
}

/// Lower one physical all-flow program together with its exact persisted
/// helicity-dispatch sidecar.
///
/// The ordinary DirectPlan remains unchanged and executable on its own.  The
/// companion sidecar selects dependency-closed non-source row runs for one
/// resolved helicity during cold schedule binding; warm execution uses only
/// that already-specialized schedule.
pub fn lower_recurrence_direct_plan_v2_with_helicity_dispatch(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    semantic_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
) -> RusticolResult<(DirectRecurrencePlan, DirectHelicityDispatch)> {
    let projection = project_all_flow_helicity_support(program, templates)?;
    let (parts, _, lowered_support) = build_direct_parts(
        program,
        templates,
        direct_executors,
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        runtime_options,
        &RecurrenceRelationDiscoveryOptions::off(),
        Some(&projection),
    )?;
    let lowered_support = lowered_support
        .ok_or_else(|| invalid("all-flow helicity support disappeared during direct lowering"))?;
    let plan = DirectRecurrencePlan::new(parts)?;
    let dispatch = lower_direct_helicity_dispatch(&plan, lowered_support)?;
    dispatch.validate_for_plan(&plan)?;
    Ok((plan, dispatch))
}

/// Lower recurrence with opt-in relation diagnostics or certified scale-copy
/// reuse. The returned report is absent only when the mode is `off`.
#[allow(clippy::too_many_arguments)]
pub fn lower_recurrence_direct_plan_v2_with_relation_discovery(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    semantic_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
    relation_options: &RecurrenceRelationDiscoveryOptions,
) -> RusticolResult<(
    DirectRecurrencePlan,
    Option<RecurrenceRelationDiscoveryReport>,
)> {
    let (parts, report, _) = build_direct_parts(
        program,
        templates,
        direct_executors,
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        runtime_options,
        relation_options,
        None,
    )?;
    Ok((DirectRecurrencePlan::new(parts)?, report))
}

#[allow(clippy::too_many_arguments)]
fn build_direct_parts(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    semantic_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
    relation_options: &RecurrenceRelationDiscoveryOptions,
    helicity_support: Option<&AllFlowHelicitySupportProjection>,
) -> RusticolResult<(
    DirectRecurrencePlanParts,
    Option<RecurrenceRelationDiscoveryReport>,
    Option<LoweredHelicityRowSupport>,
)> {
    validate_lowering_boundary(
        program,
        templates,
        direct_executors,
        prepared_pack_digest,
        direct_template_catalog_digest,
        runtime_options,
    )?;

    let input = templates.input();
    let external_source_count = external_source_count(program)?;
    let component_counts = component_counts(program, templates)?;
    let arena = recurrence_direct_arena_layout(program, &component_counts)?;
    let selector_domains = lower_selector_domains(program)?;

    let (momentum_ids, momentum_forms, momentum_terms) =
        intern_momentum_forms(program, external_source_count)?;
    let (mut exact_factors, mut exact_factor_ids) = intern_exact_factors(program)?;
    let mut closure_factor_blocks = BTreeMap::new();

    let mut source_drafts = Vec::new();
    let mut runtime_sources = LoweredRuntimeSources {
        choices_by_source: vec![Vec::new(); external_source_count as usize],
        ..LoweredRuntimeSources::default()
    };
    for current in program
        .currents()
        .iter()
        .filter(|current| current.is_source())
    {
        let assignment = arena_assignment(&arena, current.id())?;
        let source_slot = only_source_slot(current)?;
        match (program.strategy(), current.key().source_binding()) {
            (
                RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion,
                CurrentSourceBinding::FixedTemplate(source_template_id),
            ) => {
                let source_factor = current.source_exact_factor().ok_or_else(|| {
                    invalid(format!(
                        "{} source current {} has no exact source factor",
                        program.strategy(),
                        current.id(),
                    ))
                })?;
                let source = input
                    .sources
                    .get(*source_template_id as usize)
                    .ok_or_else(|| {
                        invalid(format!(
                            "source current {} references absent source template {source_template_id}",
                            current.id()
                        ))
                    })?;
                let current_state =
                    state_row(templates, current.key().current_state_template_id())?;
                let source_state = state_row(templates, source.state_template_id)?;
                if current_state.dimension != source_state.dimension {
                    return Err(invalid(format!(
                        "source current {} and source template {source_template_id} have different component dimensions",
                        current.id()
                    )));
                }
                let evaluator_binding_id = validated_evaluator_binding_id(
                    templates,
                    source.evaluator_binding_id,
                    EvaluatorContractKind::Source,
                    "source",
                )?;
                let executor_id = direct_executors
                    .resolve_evaluator(DirectExecutorRole::Source, evaluator_binding_id)?;
                source_drafts.push(SourceDraft {
                    stage: current_stage(current)?,
                    executor_id,
                    semantic_current_id: current.id(),
                    row: DirectSourceRow {
                        source_slot,
                        destination_component_base: assignment.component_base,
                        momentum_form_id: momentum_id(&momentum_ids, current.key().momentum())?,
                        source_template_or_dispatch_domain: *source_template_id,
                        spin_state_class: current.key().spin_state_class(),
                        exact_factor_id: exact_factor_id(&exact_factor_ids, source_factor)?,
                        selector_domain_id: selector_domains.current_domain_ids
                            [current.id() as usize],
                    },
                });
            }
            (
                RecurrenceStrategy::AllFlowUnion,
                CurrentSourceBinding::RuntimeDispatch {
                    domain,
                    source_template_ids,
                    variant_bindings,
                },
            ) => {
                lower_union_source(
                    current,
                    source_slot,
                    *domain,
                    source_template_ids,
                    variant_bindings,
                    templates,
                    direct_executors,
                    &arena,
                    &momentum_ids,
                    &mut exact_factors,
                    &mut exact_factor_ids,
                    &mut source_drafts,
                    &mut runtime_sources,
                )?;
            }
            (RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion, _) => {
                return Err(invalid(format!(
                    "{} source current {} does not use one fixed source template",
                    program.strategy(),
                    current.id(),
                )));
            }
            (RecurrenceStrategy::AllFlowUnion, _) => {
                return Err(invalid(format!(
                    "all-flow-union source current {} does not use runtime dispatch",
                    current.id()
                )));
            }
        }
    }

    let mut contribution_drafts = Vec::with_capacity(program.contributions().len());
    for contribution in program.contributions() {
        validate_direct_arity(
            "contribution",
            contribution.id(),
            contribution.parent_current_ids().len(),
        )?;
        let result = program
            .currents()
            .get(contribution.result_current_id() as usize)
            .ok_or_else(|| invalid("contribution result current is absent"))?;
        let transition_id = contribution.key().transition_template_id();
        let transition = input
            .transitions
            .get(transition_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "contribution {} references absent transition template {transition_id}",
                    contribution.id()
                ))
            })?;
        let expected_parent_states = template_u32_sequence(
            templates,
            transition.input_state_sequence_id,
            "transition input states",
        )?;
        let canonical_input_order = template_u32_sequence(
            templates,
            transition.canonical_input_order_sequence_id,
            "transition canonical input order",
        )?;
        let actual_parent_states =
            parent_state_template_ids(program, contribution.parent_current_ids())?;
        if !parent_state_contract_matches(
            expected_parent_states,
            canonical_input_order,
            transition.input_exchange_factor_id != MISSING_U32,
            &actual_parent_states,
        )? || contribution.key().parent_state_template_ids() != actual_parent_states
        {
            return Err(invalid(format!(
                "contribution {} parent-state contract does not match transition {transition_id}",
                contribution.id()
            )));
        }
        if transition.result_state_template_id != result.key().current_state_template_id()
            || transition.result_state_template_id != contribution.key().result_state_template_id()
        {
            return Err(invalid(format!(
                "contribution {} result-state contract does not match transition {transition_id}",
                contribution.id()
            )));
        }
        if transition.quantum_flow_template_id != contribution.key().quantum_flow_witness_id() {
            return Err(invalid(format!(
                "contribution {} quantum-flow witness does not match transition {transition_id}",
                contribution.id()
            )));
        }
        validate_coupling_digest(templates, contribution)?;
        if transition.output_projection_string_id != contribution.key().output_projection_id() {
            return Err(invalid(format!(
                "contribution {} output projection does not match transition {transition_id}",
                contribution.id()
            )));
        }
        let color_witness = contribution.key().color_witness_term_id();
        if color_witness.color_contraction_template_id() != transition.color_contraction_template_id
        {
            return Err(invalid(format!(
                "contribution {} color witness does not belong to transition {transition_id}",
                contribution.id()
            )));
        }
        let contraction = input
            .color_contractions
            .get(transition.color_contraction_template_id as usize)
            .ok_or_else(|| invalid("transition color contraction is absent"))?;
        if u64::from(color_witness.witness_ordinal()) >= contraction.witness_count {
            return Err(invalid(format!(
                "contribution {} color-witness ordinal is out of bounds",
                contribution.id()
            )));
        }
        validate_parent_momentum_contract(program, contribution)?;
        let parents = direct_parents(
            program,
            contribution.parent_current_ids(),
            &arena,
            &momentum_ids,
        )?;
        let evaluator_binding_id = validated_evaluator_binding_id(
            templates,
            transition.evaluator_binding_id,
            EvaluatorContractKind::Vertex,
            "contribution",
        )?;
        let (executor_id, parent_permutation) =
            direct_executors.resolve_contribution(evaluator_binding_id)?;
        let parents = parents.permuted(parent_permutation)?;
        let mut semantic_parent_current_ids = [DIRECT_NONE_U32; 2];
        for (slot, parent_id) in contribution
            .parent_current_ids()
            .iter()
            .copied()
            .enumerate()
        {
            semantic_parent_current_ids[slot] = parent_id;
        }
        if contribution.parent_current_ids().len() == 2 && parent_permutation == [1, 0] {
            semantic_parent_current_ids.swap(0, 1);
        }
        contribution_drafts.push(ContributionDraft {
            stage: current_stage(result)?,
            executor_id,
            semantic_contribution_id: contribution.id(),
            semantic_result_current_id: result.id(),
            semantic_parent_current_ids,
            semantic_parent_count: u8::try_from(contribution.parent_current_ids().len())
                .map_err(|_| invalid("contribution parent count exceeds u8"))?,
            row: DirectContributionRow {
                parent0_component_base: parents.parent0_component_base,
                parent1_component_base_or_sentinel: parents.parent1_component_base_or_sentinel,
                parent0_momentum_form_id: parents.parent0_momentum_form_id,
                parent1_momentum_form_id_or_sentinel: parents.parent1_momentum_form_id_or_sentinel,
                destination_component_base: arena_assignment(&arena, result.id())?.component_base,
                exact_factor_id: exact_factor_id(&exact_factor_ids, contribution.exact_factor())?,
                selector_domain_id: selector_domains.current_domain_ids[result.id() as usize],
                flags: DIRECT_ROW_FLAGS_NONE,
            },
        });
    }

    let mut finalization_drafts = Vec::with_capacity(program.finalizations().len());
    for finalization in program.finalizations() {
        let current = program
            .currents()
            .get(finalization.current_id() as usize)
            .ok_or_else(|| invalid("finalization current is absent"))?;
        if finalization.propagator_template_id().is_none()
            && finalization.exact_factor() == ExactComplexRational::ONE
        {
            continue;
        }
        let executor_id = if let Some(propagator_template_id) =
            finalization.propagator_template_id()
        {
            let propagator = input
                .propagators
                .get(propagator_template_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "finalization {} references absent propagator template {propagator_template_id}",
                        finalization.id()
                    ))
                })?;
            if propagator.applies_propagator == 0
                || propagator.state_template_id != current.key().current_state_template_id()
            {
                return Err(invalid(format!(
                    "finalization {} does not match an active propagator for current {}",
                    finalization.id(),
                    current.id()
                )));
            }
            let evaluator_binding_id = validated_evaluator_binding_id(
                templates,
                propagator.evaluator_binding_id,
                EvaluatorContractKind::Propagator,
                "finalization",
            )?;
            direct_executors
                .resolve_evaluator(DirectExecutorRole::Finalization, evaluator_binding_id)?
        } else {
            direct_executors.resolve_identity_finalizer()?
        };
        let assignment = arena_assignment(&arena, current.id())?;
        finalization_drafts.push(FinalizationDraft {
            stage: current_stage(current)?,
            executor_id,
            semantic_current_id: current.id(),
            row: DirectFinalizationRow {
                component_base: assignment.component_base,
                component_count: u16::try_from(assignment.component_count).map_err(|_| {
                    invalid(format!(
                        "current {} component count exceeds u16",
                        current.id()
                    ))
                })?,
                momentum_form_id: momentum_id(&momentum_ids, current.key().momentum())?,
                exact_factor_id: exact_factor_id(&exact_factor_ids, finalization.exact_factor())?,
                selector_domain_id: selector_domains.current_domain_ids[current.id() as usize],
                flags: DIRECT_ROW_FLAGS_NONE,
            },
        });
    }

    let closure_component_coefficients = templates.closure_component_coefficient_catalog()?;
    let closure_stage = program
        .currents()
        .iter()
        .map(current_stage)
        .collect::<RusticolResult<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| invalid("closure stage exceeds u16"))?;
    let mut closure_drafts = Vec::with_capacity(program.closure_terms().len());
    for closure in program.closure_terms() {
        validate_direct_arity("closure", closure.id(), closure.parent_current_ids().len())?;
        let closure_template_id = closure.closure_template_id();
        let template = input
            .closures
            .get(closure_template_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "closure {} references absent closure template {closure_template_id}",
                    closure.id()
                ))
            })?;
        let expected_parent_states = template_u32_sequence(
            templates,
            template.input_state_sequence_id,
            "closure input states",
        )?;
        let canonical_input_order = template_u32_sequence(
            templates,
            template.canonical_input_order_sequence_id,
            "closure canonical input order",
        )?;
        let actual_parent_states =
            parent_state_template_ids(program, closure.parent_current_ids())?;
        if !parent_state_contract_matches(
            expected_parent_states,
            canonical_input_order,
            template.input_exchange_factor_id != MISSING_U32,
            &actual_parent_states,
        )? {
            return Err(invalid(format!(
                "closure {} parent-state contract does not match template {closure_template_id}",
                closure.id()
            )));
        }
        let eligible_flows = template_u32_sequence(
            templates,
            template.eligible_quantum_flow_sequence_id,
            "closure eligible quantum flows",
        )?;
        match closure.quantum_flow_template_id() {
            Some(flow_id) if eligible_flows.binary_search(&flow_id).is_ok() => {}
            Some(flow_id) => {
                return Err(invalid(format!(
                    "closure {} quantum-flow witness {flow_id} is not eligible for template {closure_template_id}",
                    closure.id()
                )));
            }
            None if eligible_flows.is_empty() => {}
            None => {
                return Err(invalid(format!(
                    "closure {} omits the quantum-flow witness required by template {closure_template_id}",
                    closure.id()
                )));
            }
        }
        let evaluator_binding_id = validated_evaluator_binding_id(
            templates,
            template.evaluator_binding_id,
            EvaluatorContractKind::Closure,
            "closure",
        )?;
        let executor_id = direct_executors
            .resolve_evaluator(DirectExecutorRole::Closure, evaluator_binding_id)?;
        let parents = direct_parents(program, closure.parent_current_ids(), &arena, &momentum_ids)?;
        let component_coefficients = closure_component_coefficients
            .get(closure_template_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "closure {} references absent component coefficients for template {closure_template_id}",
                    closure.id()
                ))
            })?;
        let parent0_id = closure.parent_current_ids()[0] as usize;
        let component_count = *component_counts
            .get(parent0_id)
            .ok_or_else(|| invalid("closure parent 0 component count is absent"))?;
        if let Some(parent1_id) = closure.parent_current_ids().get(1) {
            let parent1_component_count = *component_counts
                .get(*parent1_id as usize)
                .ok_or_else(|| invalid("closure parent 1 component count is absent"))?;
            if parent1_component_count != component_count {
                return Err(invalid(format!(
                    "closure {} parent component counts differ",
                    closure.id()
                )));
            }
        }
        if component_coefficients.len() != component_count as usize {
            return Err(invalid(format!(
                "closure {} has {} component coefficients for {component_count} current components",
                closure.id(),
                component_coefficients.len()
            )));
        }
        let (component_factor_start, component_count) = intern_closure_factor_block(
            component_coefficients,
            &mut exact_factors,
            &mut closure_factor_blocks,
        )?;
        closure_drafts.push(ClosureDraft {
            stage: closure_stage,
            executor_id,
            semantic_closure_id: closure.id(),
            amplitude_destination_id: closure.target_destination_id(),
            row: DirectClosureRow {
                parent0_component_base: parents.parent0_component_base,
                parent1_component_base_or_sentinel: parents.parent1_component_base_or_sentinel,
                parent0_momentum_form_id: parents.parent0_momentum_form_id,
                parent1_momentum_form_id_or_sentinel: parents.parent1_momentum_form_id_or_sentinel,
                amplitude_destination_id: closure.target_destination_id(),
                exact_factor_id: exact_factor_id(&exact_factor_ids, closure.exact_factor())?,
                component_factor_start,
                component_count,
                selector_domain_id: selector_domains.closure_domain_ids[closure.id() as usize],
                flags: DIRECT_ROW_FLAGS_NONE,
            },
        });
    }

    let relation_report = apply_recurrence_relation_discovery(
        program,
        semantic_digest,
        &arena,
        &momentum_ids,
        &selector_domains.current_domain_ids,
        &mut contribution_drafts,
        &finalization_drafts,
        &mut exact_factors,
        &mut exact_factor_ids,
        relation_options,
    )?;

    let helicity_demands = topology_replay_helicity_demands(program, &contribution_drafts)?;
    source_drafts.sort_by(|left, right| {
        (left.stage, left.executor_id, left.row.selector_domain_id)
            .cmp(&(right.stage, right.executor_id, right.row.selector_domain_id))
            .then_with(|| {
                helicity_demands
                    .as_ref()
                    .map_or(std::cmp::Ordering::Equal, |demands| {
                        demands
                            .current(left.semantic_current_id)
                            .cmp(demands.current(right.semantic_current_id))
                    })
            })
            .then_with(|| left.semantic_current_id.cmp(&right.semantic_current_id))
    });
    order_contributions_for_runtime_fanout(&mut contribution_drafts);
    if let Some(demands) = &helicity_demands {
        order_contributions_for_helicity_locality(&mut contribution_drafts, demands);
    }
    let mut initialized_currents = BTreeSet::new();
    for draft in &mut contribution_drafts {
        if initialized_currents.insert(draft.semantic_result_current_id) {
            draft.row.flags |= DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
        }
    }
    finalization_drafts.sort_by(|left, right| {
        (left.stage, left.executor_id, left.row.selector_domain_id)
            .cmp(&(right.stage, right.executor_id, right.row.selector_domain_id))
            .then_with(|| {
                helicity_demands
                    .as_ref()
                    .map_or(std::cmp::Ordering::Equal, |demands| {
                        demands
                            .current(left.semantic_current_id)
                            .cmp(demands.current(right.semantic_current_id))
                    })
            })
            .then_with(|| left.semantic_current_id.cmp(&right.semantic_current_id))
    });
    drop(helicity_demands);
    // Destination ownership is primary. Each destination retains exactly one
    // contiguous closure range; executor changes create additional row groups.
    closure_drafts.sort_by_key(|draft| {
        (
            draft.amplitude_destination_id,
            draft.stage,
            draft.executor_id,
            draft.semantic_closure_id,
        )
    });
    let lowered_helicity_support = helicity_support
        .map(|projection| {
            lower_semantic_helicity_support(
                program,
                projection,
                &contribution_drafts,
                &finalization_drafts,
                &closure_drafts,
            )
        })
        .transpose()?;
    let mut closure_row_by_term = vec![DIRECT_NONE_U32; program.closure_terms().len()];
    let mut proof_group_by_term = vec![None; program.closure_terms().len()];
    for group in program.closure_proofs().groups() {
        if let Some(term_id) = group.emitted_runtime_closure_term_id() {
            let slot = proof_group_by_term
                .get_mut(term_id as usize)
                .ok_or_else(|| invalid("closure proof group references an absent runtime term"))?;
            *slot = Some(group.id());
        }
    }
    let mut lowered_proof_groups = program.closure_proofs().groups().to_vec();
    for (row_index, draft) in closure_drafts.iter_mut().enumerate() {
        let row_id = u32_len("closure row", row_index)?;
        let term_slot = closure_row_by_term
            .get_mut(draft.semantic_closure_id as usize)
            .ok_or_else(|| invalid("closure draft semantic term ID is out of bounds"))?;
        if *term_slot != DIRECT_NONE_U32 {
            return Err(invalid(format!(
                "semantic closure term {} lowered more than once",
                draft.semantic_closure_id
            )));
        }
        *term_slot = row_id;
        let group_id = proof_group_by_term
            .get(draft.semantic_closure_id as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                invalid(format!(
                    "semantic closure term {} has no proof group",
                    draft.semantic_closure_id
                ))
            })?;
        let group = program
            .closure_proofs()
            .groups()
            .get(group_id as usize)
            .filter(|group| group.id() == group_id)
            .ok_or_else(|| invalid("closure proof group ID is not dense"))?;
        draft.row.flags = group.id();
        let component_start = draft.row.component_factor_start as usize;
        let component_end = component_start
            .checked_add(draft.row.component_count as usize)
            .ok_or_else(|| invalid("closure component-factor range exceeds usize"))?;
        let component_digest = closure_component_factor_digest_v2(
            exact_factors
                .get(component_start..component_end)
                .ok_or_else(|| invalid("closure component-factor range is out of bounds"))?,
        )
        .map_err(|error| invalid(error.message()))?;
        let selector = selector_domains
            .descriptors
            .get(draft.row.selector_domain_id as usize)
            .ok_or_else(|| invalid("lowered closure selector domain is absent"))?;
        let selector_start = selector.word_start as usize;
        let selector_end = selector_start
            .checked_add(selector.word_count as usize)
            .ok_or_else(|| invalid("lowered closure selector range overflows usize"))?;
        let selector_digest = closure_selector_domain_digest_v2(
            selector_domains
                .words
                .get(selector_start..selector_end)
                .ok_or_else(|| invalid("lowered closure selector range is out of bounds"))?,
        )
        .map_err(|error| invalid(error.message()))?;
        lowered_proof_groups[group.id() as usize] =
            group.with_direct_closure_binding(Some(row_id), component_digest, selector_digest)?;
    }
    if closure_row_by_term.contains(&DIRECT_NONE_U32) {
        return Err(invalid(
            "not every semantic closure term was lowered to a direct row",
        ));
    }
    let closure_proofs =
        ClosureProofMetadataV2::new_with_three_line_certificates_and_candidate_domain(
            program.closure_proofs().contributions().to_vec(),
            lowered_proof_groups,
            program.closure_proofs().reflection_certificates().to_vec(),
            program
                .closure_proofs()
                .three_line_traversal_certificates()
                .to_vec(),
            program.closure_proofs().candidate_domain_certificate(),
        )
        .map_err(|error| invalid(error.message()))?;

    let mut source_row_by_current = vec![DIRECT_NONE_U32; program.currents().len()];
    for (row_index, draft) in source_drafts.iter().enumerate() {
        source_row_by_current[draft.semantic_current_id as usize] =
            u32_len("source row", row_index)?;
    }
    let mut finalization_row_by_current = vec![DIRECT_NONE_U32; program.currents().len()];
    for (row_index, draft) in finalization_drafts.iter().enumerate() {
        finalization_row_by_current[draft.semantic_current_id as usize] =
            u32_len("finalization row", row_index)?;
    }

    let currents = program
        .currents()
        .iter()
        .map(|current| {
            let assignment = arena_assignment(&arena, current.id())?;
            Ok(DirectCurrentDescriptor {
                semantic_current_id: current.id(),
                node_kind: match current.key().node_kind() {
                    RecurrenceNodeKind::Source => DirectNodeKind::Source,
                    RecurrenceNodeKind::Current => DirectNodeKind::Current,
                },
                state_template_id: current.key().current_state_template_id(),
                component_base: assignment.component_base,
                component_count: u16::try_from(assignment.component_count).map_err(|_| {
                    invalid(format!(
                        "current {} component count exceeds u16",
                        current.id()
                    ))
                })?,
                momentum_form_id: momentum_id(&momentum_ids, current.key().momentum())?,
                stage: current_stage(current)?,
                selector_domain_id: selector_domains.current_domain_ids[current.id() as usize],
                first_use: direct_event(assignment.first_use, "arena first use")?,
                last_use: direct_event(assignment.last_use, "arena last use")?,
                source_row_or_sentinel: source_row_by_current[current.id() as usize],
                finalization_row_or_sentinel: finalization_row_by_current[current.id() as usize],
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let mut row_groups = Vec::new();
    append_row_groups(
        source_drafts
            .iter()
            .map(|draft| (draft.stage, draft.executor_id)),
        DirectExecutorRole::Source,
        DirectDestinationOperation::Initialize,
        &mut row_groups,
    )?;
    append_row_groups(
        contribution_drafts
            .iter()
            .map(|draft| (draft.stage, draft.executor_id)),
        DirectExecutorRole::Contribution,
        DirectDestinationOperation::Add,
        &mut row_groups,
    )?;
    append_row_groups(
        finalization_drafts
            .iter()
            .map(|draft| (draft.stage, draft.executor_id)),
        DirectExecutorRole::Finalization,
        DirectDestinationOperation::FinalizeInPlace,
        &mut row_groups,
    )?;
    append_row_groups(
        closure_drafts
            .iter()
            .map(|draft| (draft.stage, draft.executor_id)),
        DirectExecutorRole::Closure,
        DirectDestinationOperation::ClosureAdd,
        &mut row_groups,
    )?;
    row_groups.sort_by_key(|row_group| (row_group.stage, row_group.role, row_group.row_start));

    let amplitude_destinations = lower_amplitude_destinations(
        program,
        &closure_drafts,
        &selector_domains.destination_domain_ids,
    )?;
    let (
        resolved_helicities,
        source_state_assignments,
        resolved_source_selections,
        public_helicities,
    ) = match program.strategy() {
        RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => {
            let (descriptors, source_states, public_helicities) =
                lower_resolved_helicities(program)?;
            (descriptors, source_states, Vec::new(), public_helicities)
        }
        RecurrenceStrategy::AllFlowUnion => lower_union_resolved_helicities(
            &runtime_sources.choices_by_source,
            program.retained_helicity_count(),
        )?,
    };
    let (replay_targets, source_permutations, replay_momentum_signs, replay_helicity_map) =
        lower_replay_targets(
            program,
            &exact_factor_ids,
            &selector_domains.sector_domain_ids,
        )?;

    let state_template_count = u32_len("state template", input.current_states.len())?;
    let source_template_count = u32_len("source template", input.sources.len())?;
    let direct_executor_count = direct_executors.direct_executor_count();
    if state_template_count == 0 || source_template_count == 0 || direct_executor_count == 0 {
        return Err(invalid(
            "direct lowering requires nonempty state, source, and executor catalogs",
        ));
    }

    let parts = DirectRecurrencePlanParts {
        strategy: program.strategy(),
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        point_tile_size: runtime_options.point_tile_size,
        workspace_mib: runtime_options.workspace_mib,
        current_arena_components: arena.component_count(),
        physical_sector_count: program.physical_sector_count(),
        retained_helicity_count: program.retained_helicity_count(),
        amplitude_destination_count: u32_len(
            "amplitude destination",
            program.amplitude_destinations().len(),
        )?,
        parameter_value_count: templates.summary().parameter_count,
        external_source_count,
        state_template_count,
        source_template_count,
        source_template_or_dispatch_count: match program.strategy() {
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => {
                source_template_count
            }
            RecurrenceStrategy::AllFlowUnion => templates.summary().runtime_helicity_contract_count,
        },
        runtime_helicity_contract_count: match program.strategy() {
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => 0,
            RecurrenceStrategy::AllFlowUnion => templates.summary().runtime_helicity_contract_count,
        },
        runtime_helicity_variant_count: match program.strategy() {
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => 0,
            RecurrenceStrategy::AllFlowUnion => u32_len(
                "runtime-helicity variant",
                input.runtime_helicity_variants.len(),
            )?,
        },
        direct_executor_count,
        currents,
        sources: source_drafts.iter().map(|draft| draft.row).collect(),
        contributions: contribution_drafts.iter().map(|draft| draft.row).collect(),
        finalizations: finalization_drafts.iter().map(|draft| draft.row).collect(),
        closures: closure_drafts.iter().map(|draft| draft.row).collect(),
        row_groups,
        momentum_forms,
        momentum_terms,
        // Topology replay persists one dependency-closed domain per physical
        // sector. Other strategies retain the historical universal domain.
        selector_domains: selector_domains.descriptors,
        selector_words: selector_domains.words,
        replay_targets,
        source_permutations,
        replay_momentum_signs,
        replay_helicity_map,
        amplitude_destinations,
        resolved_helicities,
        source_state_assignments,
        source_dispatch_variants: runtime_sources.variants,
        source_embeddings: runtime_sources.embeddings,
        source_projections: runtime_sources.projections,
        resolved_source_selections,
        public_helicities,
        exact_factors,
        closure_proofs,
    };
    Ok((parts, relation_report, lowered_helicity_support))
}

fn checked_projection_mask(
    mask: &AllFlowHelicityMask,
    expected_word_count: usize,
    label: &str,
) -> RusticolResult<Vec<u64>> {
    if mask.words().len() != expected_word_count {
        return Err(invalid(format!(
            "{label} support mask has {} words, expected {expected_word_count}",
            mask.words().len()
        )));
    }
    Ok(mask.words().to_vec())
}

fn lower_semantic_helicity_support(
    program: &RecurrenceProgram,
    projection: &AllFlowHelicitySupportProjection,
    contribution_drafts: &[ContributionDraft],
    finalization_drafts: &[FinalizationDraft],
    closure_drafts: &[ClosureDraft],
) -> RusticolResult<LoweredHelicityRowSupport> {
    if program.strategy() != RecurrenceStrategy::AllFlowUnion
        || u64::from(projection.resolved_helicity_count()) != program.retained_helicity_count()
    {
        return Err(invalid(
            "semantic helicity support does not match the all-flow program axis",
        ));
    }
    let word_count = projection.resolved_helicity_count().div_ceil(64) as usize;
    let mut contributions = Vec::with_capacity(contribution_drafts.len());
    for draft in contribution_drafts {
        let mask = if draft.row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0 {
            let result = projection
                .current_demand()
                .get(draft.semantic_result_current_id as usize)
                .ok_or_else(|| invalid("certified-reuse result current support is absent"))?;
            let representative_id = draft.semantic_parent_current_ids[0];
            if representative_id == DIRECT_NONE_U32 {
                return Err(invalid(
                    "certified-reuse contribution has no representative current",
                ));
            }
            let representative = projection
                .current_availability()
                .get(representative_id as usize)
                .ok_or_else(|| {
                    invalid("certified-reuse representative current support is absent")
                })?;
            if result.words().len() != word_count || representative.words().len() != word_count {
                return Err(invalid(
                    "certified-reuse current support mask has the wrong width",
                ));
            }
            result
                .words()
                .iter()
                .zip(representative.words())
                .map(|(result, representative)| result & representative)
                .collect()
        } else {
            let mask = projection
                .contribution_support()
                .get(draft.semantic_contribution_id as usize)
                .ok_or_else(|| invalid("semantic contribution support is absent"))?;
            checked_projection_mask(mask, word_count, "contribution")?
        };
        contributions.push(mask);
    }

    let finalizations = finalization_drafts
        .iter()
        .map(|draft| {
            let finalization_id = program
                .currents()
                .get(draft.semantic_current_id as usize)
                .ok_or_else(|| invalid("finalization current is absent"))?
                .finalization_id()
                .ok_or_else(|| invalid("finalization current has no semantic finalization"))?;
            let mask = projection
                .finalization_support()
                .get(finalization_id as usize)
                .ok_or_else(|| invalid("semantic finalization support is absent"))?;
            checked_projection_mask(mask, word_count, "finalization")
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let closures = closure_drafts
        .iter()
        .map(|draft| {
            let mask = projection
                .closure_support()
                .get(draft.semantic_closure_id as usize)
                .ok_or_else(|| invalid("semantic closure support is absent"))?;
            checked_projection_mask(mask, word_count, "closure")
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    Ok(LoweredHelicityRowSupport {
        resolved_helicity_count: projection.resolved_helicity_count(),
        contributions,
        finalizations,
        closures,
    })
}

#[derive(Default)]
struct HelicitySupportDomainInterner {
    ids: BTreeMap<Vec<u64>, u32>,
    descriptors: Vec<DirectHelicitySupportDomainDescriptor>,
    words: Vec<u64>,
}

impl HelicitySupportDomainInterner {
    fn intern(&mut self, words: &[u64]) -> RusticolResult<u32> {
        let retained = words
            .iter()
            .rposition(|word| *word != 0)
            .map_or(0, |index| index + 1);
        let canonical = &words[..retained];
        if let Some(domain_id) = self.ids.get(canonical) {
            return Ok(*domain_id);
        }
        let domain_id = u32_len("helicity support domain", self.descriptors.len())?;
        let word_start = u64::try_from(self.words.len())
            .map_err(|_| invalid("helicity support-word table exceeds u64"))?;
        let word_count = u32_len("helicity support word", canonical.len())?;
        self.words.extend_from_slice(canonical);
        self.descriptors
            .push(DirectHelicitySupportDomainDescriptor {
                word_start,
                word_count,
            });
        self.ids.insert(canonical.to_vec(), domain_id);
        Ok(domain_id)
    }
}

fn role_helicity_support(
    support: &LoweredHelicityRowSupport,
    role: DirectExecutorRole,
) -> RusticolResult<&[Vec<u64>]> {
    match role {
        DirectExecutorRole::Contribution => Ok(&support.contributions),
        DirectExecutorRole::Finalization => Ok(&support.finalizations),
        DirectExecutorRole::Closure => Ok(&support.closures),
        DirectExecutorRole::Source => Err(invalid(
            "source rows must not be lowered into helicity-dispatch groups",
        )),
    }
}

fn lower_direct_helicity_dispatch(
    plan: &DirectRecurrencePlan,
    support: LoweredHelicityRowSupport,
) -> RusticolResult<DirectHelicityDispatch> {
    if plan.strategy() != RecurrenceStrategy::AllFlowUnion
        || support.resolved_helicity_count as usize != plan.resolved_helicities().len()
    {
        return Err(invalid(
            "helicity dispatch lowering requires one matching all-flow resolved axis",
        ));
    }
    for (label, actual, expected) in [
        (
            "contribution",
            support.contributions.len(),
            plan.contributions().len(),
        ),
        (
            "finalization",
            support.finalizations.len(),
            plan.finalizations().len(),
        ),
        ("closure", support.closures.len(), plan.closures().len()),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "helicity {label} support has {actual} rows, expected {expected}"
            )));
        }
    }

    let mut interner = HelicitySupportDomainInterner::default();
    let mut row_groups = Vec::new();
    let mut group_masks = Vec::<Vec<u64>>::new();
    for base in plan
        .row_groups()
        .iter()
        .filter(|group| group.role != DirectExecutorRole::Source)
    {
        let role_support = role_helicity_support(&support, base.role)?;
        let start = usize::try_from(base.row_start)
            .map_err(|_| invalid("base helicity row-group start exceeds usize"))?;
        let end = start
            .checked_add(base.row_count as usize)
            .ok_or_else(|| invalid("base helicity row-group range overflows usize"))?;
        let rows = role_support
            .get(start..end)
            .ok_or_else(|| invalid("base helicity row-group range is out of bounds"))?;
        let mut run_start = 0usize;
        while run_start < rows.len() {
            let mut run_end = run_start + 1;
            while run_end < rows.len() && rows[run_end] == rows[run_start] {
                run_end += 1;
            }
            let support_domain_id = interner.intern(&rows[run_start])?;
            row_groups.push(DirectHelicityRowGroupDescriptor {
                stage: base.stage,
                role: base.role,
                direct_executor_id: base.direct_executor_id,
                row_start: base
                    .row_start
                    .checked_add(run_start as u64)
                    .ok_or_else(|| invalid("split helicity row-group start overflows u64"))?,
                row_count: u32_len("split helicity row-group row", run_end - run_start)?,
                support_domain_id,
            });
            group_masks.push(rows[run_start].clone());
            run_start = run_end;
        }
    }

    let resolved_count = support.resolved_helicity_count as usize;
    let mut groups_by_helicity = vec![Vec::<u32>::new(); resolved_count];
    for (group_id, words) in group_masks.iter().enumerate() {
        for (word_index, mut word) in words.iter().copied().enumerate() {
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let helicity_id = word_index
                    .checked_mul(u64::BITS as usize)
                    .and_then(|base| base.checked_add(bit))
                    .ok_or_else(|| invalid("helicity support bit index overflows usize"))?;
                let groups = groups_by_helicity
                    .get_mut(helicity_id)
                    .ok_or_else(|| invalid("helicity support mask exceeds the resolved axis"))?;
                groups.push(u32_len("split helicity row-group", group_id)?);
                word &= word - 1;
            }
        }
    }
    let mut dispatches = Vec::with_capacity(resolved_count);
    let mut dispatch_group_ids = Vec::new();
    for (helicity_id, group_ids) in groups_by_helicity.into_iter().enumerate() {
        let group_id_start = u64::try_from(dispatch_group_ids.len())
            .map_err(|_| invalid("helicity dispatch CSR exceeds u64"))?;
        let group_id_count = u32_len("helicity dispatch group ID", group_ids.len())?;
        dispatch_group_ids.extend(group_ids);
        dispatches.push(DirectHelicityDispatchDescriptor {
            resolved_helicity_id: u32_len("resolved helicity", helicity_id)?,
            group_id_start,
            group_id_count,
        });
    }

    DirectHelicityDispatch::new(DirectHelicityDispatchParts {
        runtime_layout_digest: plan.runtime_layout_digest(),
        source_row_count: plan.sources().len() as u64,
        contribution_row_count: plan.contributions().len() as u64,
        finalization_row_count: plan.finalizations().len() as u64,
        closure_row_count: plan.closures().len() as u64,
        amplitude_destination_count: plan.amplitude_destination_count(),
        direct_executor_count: plan.direct_executor_count(),
        resolved_helicity_count: support.resolved_helicity_count,
        support_domains: interner.descriptors,
        support_words: interner.words,
        row_groups,
        dispatches,
        dispatch_group_ids,
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_recurrence_relation_discovery(
    program: &RecurrenceProgram,
    semantic_digest: SemanticDigest,
    arena: &DirectArenaLayout,
    momentum_ids: &BTreeMap<CanonicalMomentumLinearForm, u32>,
    current_selector_domain_ids: &[u32],
    contribution_drafts: &mut Vec<ContributionDraft>,
    finalization_drafts: &[FinalizationDraft],
    exact_factors: &mut Vec<ExactComplexRational>,
    exact_factor_ids: &mut BTreeMap<ExactComplexRational, u32>,
    options: &RecurrenceRelationDiscoveryOptions,
) -> RusticolResult<Option<RecurrenceRelationDiscoveryReport>> {
    if options.mode == RecurrenceRelationDiscoveryMode::Off {
        return Ok(None);
    }
    let contribution_count_before = contribution_drafts.len();
    let mut contribution_by_current =
        vec![Vec::<RelationTermInput>::new(); program.currents().len()];
    for draft in contribution_drafts.iter() {
        let parent_count = usize::from(draft.semantic_parent_count);
        let parent_current_ids = draft
            .semantic_parent_current_ids
            .get(..parent_count)
            .ok_or_else(|| invalid("relation draft parent range is invalid"))?
            .to_vec();
        let parent_momentum_form_ids = match parent_count {
            1 => vec![draft.row.parent0_momentum_form_id],
            2 => vec![
                draft.row.parent0_momentum_form_id,
                draft.row.parent1_momentum_form_id_or_sentinel,
            ],
            _ => {
                return Err(invalid(format!(
                    "relation draft {} has unsupported arity {parent_count}",
                    draft.semantic_contribution_id
                )));
            }
        };
        contribution_by_current
            .get_mut(draft.semantic_result_current_id as usize)
            .ok_or_else(|| invalid("relation result current is absent"))?
            .push(RelationTermInput {
                executor_id: draft.executor_id,
                parent_current_ids: parent_current_ids.into_boxed_slice(),
                parent_momentum_form_ids: parent_momentum_form_ids.into_boxed_slice(),
                exact_factor: *exact_factors
                    .get(draft.row.exact_factor_id as usize)
                    .ok_or_else(|| invalid("relation contribution factor is absent"))?,
            });
    }
    let finalization_by_current = finalization_drafts
        .iter()
        .map(|draft| (draft.semantic_current_id, draft))
        .collect::<BTreeMap<_, _>>();
    let inputs = program
        .currents()
        .iter()
        .map(|current| {
            let assignment = arena_assignment(arena, current.id())?;
            let finalization = finalization_by_current.get(&current.id()).copied();
            Ok(RelationCurrentInput {
                current_id: current.id(),
                is_source: current.is_source(),
                contract: RelationCurrentContract {
                    stage: current_stage(current)?,
                    state_template_id: current.key().current_state_template_id(),
                    component_count: u16::try_from(assignment.component_count).map_err(|_| {
                        invalid(format!(
                            "relation current {} component count exceeds u16",
                            current.id()
                        ))
                    })?,
                    momentum_form_id: momentum_id(momentum_ids, current.key().momentum())?,
                    selector_domain_id: *current_selector_domain_ids
                        .get(current.id() as usize)
                        .ok_or_else(|| invalid("relation selector domain is absent"))?,
                    finalization_executor_id: finalization
                        .map(|draft| draft.executor_id)
                        .unwrap_or(DIRECT_NONE_U32),
                    finalization_momentum_form_id: finalization
                        .map(|draft| draft.row.momentum_form_id)
                        .unwrap_or(DIRECT_NONE_U32),
                    finalization_exact_factor: finalization
                        .map(|draft| {
                            exact_factors
                                .get(draft.row.exact_factor_id as usize)
                                .copied()
                                .ok_or_else(|| invalid("relation finalization factor is absent"))
                        })
                        .transpose()?
                        .unwrap_or(ExactComplexRational::ONE),
                },
                terms: std::mem::take(
                    contribution_by_current
                        .get_mut(current.id() as usize)
                        .ok_or_else(|| invalid("relation current term vector is absent"))?,
                )
                .into_boxed_slice(),
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let (outcome, certificate_algorithm, tested_hypothesis_count, verification_rejected_count) =
        if let Some(evidence) = options.numerical_evidence.as_ref() {
            if evidence.schedule_semantic_digest != semantic_digest.to_string() {
                return Err(invalid(
                    "numerical evidence selects a different semantic schedule",
                ));
            }
            let expected_candidate_count = evidence
                .mappings
                .len()
                .checked_add(evidence.verification_rejected_count)
                .ok_or_else(|| invalid("numerical evidence candidate count overflows usize"))?;
            if evidence.numerical_candidate_count != expected_candidate_count
                || evidence.tested_hypothesis_count < evidence.numerical_candidate_count
                || evidence.precision_digits != options.precision_digits
                || evidence.probe_count != options.probe_count
                || evidence.verification_probe_count != options.verification_probe_count
                || evidence.relative_tolerance.to_bits() != options.relative_tolerance.to_bits()
                || evidence.absolute_tolerance.to_bits() != options.absolute_tolerance.to_bits()
                || evidence.seed != options.seed
            {
                return Err(invalid(
                    "numerical evidence probe contract or counters are inconsistent",
                ));
            }
            for (label, value) in [
                (
                    "baseline runtime-layout digest",
                    evidence.baseline_runtime_layout_digest.as_str(),
                ),
                (
                    "source-semantics digest",
                    evidence.source_semantics_sha256.as_str(),
                ),
                (
                    "certificate-set digest",
                    evidence.certificate_set_sha256.as_str(),
                ),
                (
                    "runtime-parameter-schema digest",
                    evidence.runtime_parameter_schema_sha256.as_str(),
                ),
                (
                    "candidate observation-batch digest",
                    evidence.candidate_observation_batch_sha256.as_str(),
                ),
                (
                    "verification observation-batch digest",
                    evidence.verification_observation_batch_sha256.as_str(),
                ),
                ("decision digest", evidence.decision_sha256.as_str()),
                (
                    "rejection decision digest",
                    evidence.rejection_decision_sha256.as_str(),
                ),
            ] {
                if !is_sha256(value) {
                    return Err(invalid(format!(
                        "numerical evidence {label} is not a lowercase SHA-256"
                    )));
                }
            }
            if evidence
                .mappings
                .windows(2)
                .any(|pair| pair[0].current_id >= pair[1].current_id)
            {
                return Err(invalid(
                    "numerical evidence mappings are not in increasing current order",
                ));
            }
            let mut relations = Vec::with_capacity(evidence.mappings.len());
            for mapping in &evidence.mappings {
                let current = inputs
                    .get(mapping.current_id as usize)
                    .ok_or_else(|| invalid("numerical evidence current is absent"))?;
                let representative = inputs
                    .get(mapping.execution_representative_id as usize)
                    .ok_or_else(|| {
                        invalid("numerical evidence execution representative is absent")
                    })?;
                if current.current_id != mapping.current_id
                    || current.is_source
                    || representative.current_id != mapping.execution_representative_id
                    || current.contract != representative.contract
                    || mapping.current_dimension != current.contract.component_count
                    || !is_sha256(&mapping.certificate_proof_sha256)
                    || !is_sha256(&mapping.candidate_observations_sha256)
                    || !is_sha256(&mapping.verification_observations_sha256)
                {
                    return Err(invalid(
                        "numerical evidence violates its exact current contract",
                    ));
                }
                let expected_factor = match mapping.relation_kind.as_str() {
                    "equal" => ExactComplexRational::ONE,
                    "opposite" => ExactComplexRational::ONE.checked_neg()?,
                    "zero" => ExactComplexRational::ZERO,
                    _ => {
                        return Err(invalid("numerical evidence relation kind is unsupported"));
                    }
                };
                if mapping.factor != expected_factor {
                    return Err(invalid(
                        "numerical evidence relation factor is not exact ±1/zero",
                    ));
                }
                if mapping.relation_kind == "zero" {
                    if mapping.representative_id.is_some()
                        || mapping.execution_representative_id > mapping.current_id
                    {
                        return Err(invalid(
                            "zero-current evidence has an invalid representative",
                        ));
                    }
                } else if mapping.representative_id != Some(mapping.execution_representative_id)
                    || mapping.execution_representative_id >= mapping.current_id
                {
                    return Err(invalid(
                        "equal/opposite evidence is not topologically ordered",
                    ));
                }
                relations.push(RecurrenceCurrentRelationCertificate {
                    current_id: mapping.current_id,
                    representative_id: mapping.execution_representative_id,
                    factor: mapping.factor,
                    current_expression_sha256: mapping.candidate_observations_sha256.clone(),
                    representative_expression_sha256: mapping
                        .verification_observations_sha256
                        .clone(),
                    proof_sha256: mapping.certificate_proof_sha256.clone(),
                });
            }
            (
                RecurrenceRelationDiscoveryOutcome {
                    relations,
                    numerical_candidate_count: evidence.numerical_candidate_count,
                    uncertified_candidate_count: evidence.verification_rejected_count,
                    rejected_hypothesis_count: evidence.rejected_hypothesis_count,
                    rejected_candidates: Vec::new(),
                    rejected_decision_sha256: evidence.rejection_decision_sha256.clone(),
                    effective_projection_count: options
                        .probe_count
                        .checked_add(options.verification_probe_count)
                        .ok_or_else(|| invalid("numerical probe count overflows u32"))?,
                },
                evidence.certificate_algorithm.clone(),
                evidence.tested_hypothesis_count,
                evidence.verification_rejected_count,
            )
        } else {
            let outcome = discover_recurrence_current_relations(&inputs, options)?;
            (outcome, relation_certificate_algorithm().to_owned(), 0, 0)
        };

    // Keep the complete independently verified evidence and report, including
    // signed relations. Only their application is restricted: numerical
    // opposite-current reuse is not enabled for contracted binary64 colour
    // sums. Equal and zero currents do not require that signed rewrite, and
    // must not be disabled merely because another certificate is opposite.
    // Exact symbolic relations and diagnostic projections remain unchanged.
    let restrict_numerical_sign = options.mode.applies_reuse()
        && options.numerical_evidence.is_some()
        && program.strategy() == RecurrenceStrategy::ContractedColorUnion;
    let projected_relations = outcome.relations.iter().filter(|certificate| {
        !restrict_numerical_sign
            || certificate.factor == ExactComplexRational::ONE
            || certificate.factor == ExactComplexRational::ZERO
    });
    let projected_relation_count = projected_relations.clone().count();
    let applied_relation_count = if options.mode.applies_reuse() {
        projected_relation_count
    } else {
        0
    };
    let removed_contribution_count = if applied_relation_count != 0 {
        let relation_by_current = projected_relations
            .clone()
            .map(|certificate| (certificate.current_id, certificate))
            .collect::<BTreeMap<_, _>>();
        let original_len = contribution_drafts.len();
        contribution_drafts
            .retain(|draft| !relation_by_current.contains_key(&draft.semantic_result_current_id));
        let removed_contribution_count = original_len - contribution_drafts.len();
        for certificate in projected_relations.clone() {
            let current = program
                .currents()
                .get(certificate.current_id as usize)
                .ok_or_else(|| invalid("certified current is absent"))?;
            let representative = program
                .currents()
                .get(certificate.representative_id as usize)
                .ok_or_else(|| invalid("certified representative is absent"))?;
            let current_assignment = arena_assignment(arena, current.id())?;
            let representative_assignment = arena_assignment(arena, representative.id())?;
            if current_stage(current)? != current_stage(representative)?
                || current_assignment.component_count != representative_assignment.component_count
            {
                return Err(invalid(format!(
                    "certificate {} -> {} violates scale-copy layout",
                    current.id(),
                    representative.id()
                )));
            }
            let component_count = current_assignment.component_count;
            contribution_drafts.push(ContributionDraft {
                stage: current_stage(current)?,
                executor_id: DIRECT_NONE_U32,
                semantic_contribution_id: current.id(),
                semantic_result_current_id: current.id(),
                semantic_parent_current_ids: [representative.id(), DIRECT_NONE_U32],
                semantic_parent_count: 1,
                row: DirectContributionRow {
                    parent0_component_base: representative_assignment.component_base,
                    parent1_component_base_or_sentinel: DIRECT_NONE_U32,
                    // Certified-reuse rows interpret this field as the
                    // component count; no momentum plane is read.
                    parent0_momentum_form_id: component_count,
                    parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                    destination_component_base: current_assignment.component_base,
                    exact_factor_id: intern_direct_factor(
                        certificate.factor,
                        exact_factors,
                        exact_factor_ids,
                    )?,
                    selector_domain_id: *current_selector_domain_ids
                        .get(current.id() as usize)
                        .ok_or_else(|| invalid("certified selector domain is absent"))?,
                    flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
                        | DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE,
                },
            });
        }
        removed_contribution_count
    } else {
        projected_relations
            .clone()
            .try_fold(0usize, |total, certificate| {
                let count = usize::try_from(
                    program
                        .currents()
                        .get(certificate.current_id as usize)
                        .ok_or_else(|| invalid("diagnostic relation current is absent"))?
                        .fan_in(),
                )
                .map_err(|_| invalid("diagnostic current fan-in exceeds usize"))?;
                total
                    .checked_add(count)
                    .ok_or_else(|| invalid("diagnostic saved contribution count overflows"))
            })?
    };

    let projected_contribution_count = contribution_count_before
        .checked_sub(removed_contribution_count)
        .and_then(|count| count.checked_add(projected_relation_count))
        .ok_or_else(|| invalid("relation contribution count overflows usize"))?;
    let projected_interaction_count = contribution_count_before
        .checked_sub(removed_contribution_count)
        .ok_or_else(|| invalid("relation interaction count underflows"))?;
    let applied = applied_relation_count != 0;
    Ok(Some(RecurrenceRelationDiscoveryReport {
        requested_mode: options.mode,
        state: if applied {
            "exact-certified-applied"
        } else {
            "diagnostic-only"
        },
        strategy: program.strategy(),
        color_accuracy: options.color_accuracy.clone(),
        precision_digits: options.precision_digits,
        probe_count: options.probe_count,
        verification_probe_count: options.verification_probe_count,
        seed: options.seed,
        certificate_algorithm,
        authenticated_certificate_set_sha256: options
            .numerical_evidence
            .as_ref()
            .map(|evidence| evidence.certificate_set_sha256.clone()),
        authenticated_runtime_parameter_schema_sha256: options
            .numerical_evidence
            .as_ref()
            .map(|evidence| evidence.runtime_parameter_schema_sha256.clone()),
        authenticated_candidate_observation_batch_sha256: options
            .numerical_evidence
            .as_ref()
            .map(|evidence| evidence.candidate_observation_batch_sha256.clone()),
        authenticated_verification_observation_batch_sha256: options
            .numerical_evidence
            .as_ref()
            .map(|evidence| evidence.verification_observation_batch_sha256.clone()),
        authenticated_decision_sha256: options
            .numerical_evidence
            .as_ref()
            .map(|evidence| evidence.decision_sha256.clone()),
        rejected_decision_sha256: outcome.rejected_decision_sha256,
        tested_hypothesis_count,
        verification_rejected_count,
        rejected_hypothesis_count: outcome.rejected_hypothesis_count,
        effective_projection_count: outcome.effective_projection_count,
        numerical_candidate_count: outcome.numerical_candidate_count,
        uncertified_candidate_count: outcome.uncertified_candidate_count,
        exact_certified_relation_count: outcome.relations.len(),
        applied_relation_count,
        current_count_before: program.currents().len(),
        // Direct-plan-v2 requires dense semantic current IDs and exact
        // closure-proof bindings. Scale-copy reuse removes interaction work
        // while retaining those descriptors.
        current_count_after: program.currents().len(),
        contribution_count_before,
        contribution_count_after: projected_contribution_count,
        interaction_evaluation_count_before: contribution_count_before,
        interaction_evaluation_count_after: projected_interaction_count,
        scale_copy_row_count: applied_relation_count,
        certificates: outcome.relations,
        rejected_candidates: outcome.rejected_candidates,
    }))
}

fn validate_lowering_boundary(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
) -> RusticolResult<()> {
    if runtime_options.point_tile_size == 0 || runtime_options.workspace_mib == 0 {
        return Err(invalid(
            "point tile size and workspace MiB must both be positive",
        ));
    }
    let summary = templates.summary();
    if prepared_pack_digest != summary.prepared_kernel_pack_digest {
        return Err(invalid(format!(
            "prepared-pack digest {prepared_pack_digest} does not match validated template pack {}",
            summary.prepared_kernel_pack_digest
        )));
    }
    if direct_template_catalog_digest != direct_executors.direct_template_catalog_digest() {
        return Err(invalid(format!(
            "direct-template catalog digest {direct_template_catalog_digest} does not match resolver catalog {}",
            direct_executors.direct_template_catalog_digest()
        )));
    }
    validate_prepared_parameter_catalog(templates)?;
    validate_direct_executor_catalog(templates, direct_executors)?;
    for current in program.currents() {
        if current.key().catalog_digest() != summary.catalog_digest {
            return Err(invalid(format!(
                "current {} template-catalog digest does not match the validated catalog",
                current.id()
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_union_source(
    current: &RecurrenceCurrent,
    source_slot: u32,
    dispatch_domain: u32,
    advertised_source_templates: &[u32],
    variant_bindings: &[RuntimeSourceVariantBinding],
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    arena: &DirectArenaLayout,
    momentum_ids: &BTreeMap<CanonicalMomentumLinearForm, u32>,
    exact_factors: &mut Vec<ExactComplexRational>,
    exact_factor_ids: &mut BTreeMap<ExactComplexRational, u32>,
    source_drafts: &mut Vec<SourceDraft>,
    runtime_sources: &mut LoweredRuntimeSources,
) -> RusticolResult<()> {
    if current.source_exact_factor().is_some() {
        return Err(invalid(format!(
            "all-flow-union source current {} must take its factor from runtime dispatch",
            current.id()
        )));
    }
    if variant_bindings.is_empty() {
        return Err(invalid(format!(
            "all-flow-union source current {} has no runtime variants",
            current.id()
        )));
    }
    let input = templates.input();
    let contract = input
        .runtime_helicity_contracts
        .get(dispatch_domain as usize)
        .ok_or_else(|| {
            invalid(format!(
                "source current {} references absent runtime-helicity contract {dispatch_domain}",
                current.id()
            ))
        })?;
    if contract.id != dispatch_domain {
        return Err(invalid(format!(
            "runtime-helicity contract {dispatch_domain} has noncanonical ID {}",
            contract.id
        )));
    }
    if contract.full_state_template_id != current.key().current_state_template_id() {
        return Err(invalid(format!(
            "source current {} state does not match runtime-helicity contract {dispatch_domain}",
            current.id()
        )));
    }
    let contract_variant_range = contract.variant_range.as_usize_range(
        input.runtime_helicity_variants.len(),
        "runtime-helicity contract variants",
    )?;
    let assignment = arena_assignment(arena, current.id())?;
    let full_state = state_row(templates, contract.full_state_template_id)?;
    if assignment.component_count != full_state.dimension {
        return Err(invalid(format!(
            "source current {} arena width does not match runtime-helicity full state",
            current.id()
        )));
    }
    let source_row_id = u32_len("source row", source_drafts.len())?;
    let source_choices = runtime_sources
        .choices_by_source
        .get_mut(source_slot as usize)
        .ok_or_else(|| invalid(format!("source slot {source_slot} is out of range")))?;
    if !source_choices.is_empty() {
        return Err(invalid(format!(
            "all-flow-union source slot {source_slot} has more than one dispatch current"
        )));
    }
    source_drafts.push(SourceDraft {
        stage: current_stage(current)?,
        executor_id: DIRECT_NONE_U32,
        semantic_current_id: current.id(),
        row: DirectSourceRow {
            source_slot,
            destination_component_base: assignment.component_base,
            momentum_form_id: momentum_id(momentum_ids, current.key().momentum())?,
            source_template_or_dispatch_domain: dispatch_domain,
            spin_state_class: current.key().spin_state_class(),
            exact_factor_id: intern_direct_factor(
                ExactComplexRational::ONE,
                exact_factors,
                exact_factor_ids,
            )?,
            selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
        },
    });

    for binding in variant_bindings.iter().copied() {
        if advertised_source_templates
            .binary_search(&binding.source_template_id())
            .is_err()
        {
            return Err(invalid(format!(
                "source current {} runtime variant {} is absent from its advertised source-template catalog",
                current.id(),
                binding.runtime_variant_id()
            )));
        }
        let variant_index = usize::try_from(binding.runtime_variant_id())
            .map_err(|_| invalid("runtime-helicity variant ID exceeds usize"))?;
        if !contract_variant_range.contains(&variant_index) {
            return Err(invalid(format!(
                "source current {} runtime variant {} is outside contract {dispatch_domain}",
                current.id(),
                binding.runtime_variant_id()
            )));
        }
        let variant = input
            .runtime_helicity_variants
            .get(variant_index)
            .ok_or_else(|| invalid("runtime-helicity variant is absent"))?;
        if variant.id != binding.runtime_variant_id()
            || variant.contract_id != dispatch_domain
            || variant.source_template_id != binding.source_template_id()
            || variant.source_state_template_id != binding.source_state_template_id()
        {
            return Err(invalid(format!(
                "source current {} runtime variant {} disagrees with its prepared contract",
                current.id(),
                binding.runtime_variant_id()
            )));
        }
        let source = input
            .sources
            .get(binding.source_template_id() as usize)
            .ok_or_else(|| invalid("runtime source template is absent"))?;
        if source.id != binding.source_template_id()
            || source.state_template_id != binding.source_state_template_id()
        {
            return Err(invalid(format!(
                "runtime source template {} has a stale state contract",
                binding.source_template_id()
            )));
        }
        state_row(templates, binding.crossed_state_template_id())?;
        let evaluator_binding_id = validated_evaluator_binding_id(
            templates,
            source.evaluator_binding_id,
            EvaluatorContractKind::Source,
            "runtime source",
        )?;
        let direct_executor_id =
            direct_executors.resolve_evaluator(DirectExecutorRole::Source, evaluator_binding_id)?;

        let effective_variant = effective_runtime_helicity_variant(
            input,
            contract_variant_range.clone(),
            binding.crossed_state_template_id(),
        )?;
        let embedding_start = u64::try_from(runtime_sources.embeddings.len())
            .map_err(|_| invalid("runtime source embeddings exceed u64"))?;
        let embedding_range = effective_variant.embedding_range.as_usize_range(
            input.runtime_helicity_embeddings.len(),
            "runtime source embeddings",
        )?;
        for embedding in &input.runtime_helicity_embeddings[embedding_range] {
            if embedding.variant_id != effective_variant.id {
                return Err(invalid(format!(
                    "runtime variant {} owns a stale embedding row",
                    effective_variant.id
                )));
            }
            let factor = template_exact_factor(templates, embedding.factor_id)?;
            runtime_sources.embeddings.push(DirectSourceEmbeddingRow {
                full_component: embedding.full_component,
                source_component_or_sentinel: if embedding.source_component == MISSING_U32 {
                    DIRECT_NONE_U32
                } else {
                    embedding.source_component
                },
                exact_factor_id: intern_direct_factor(factor, exact_factors, exact_factor_ids)?,
            });
        }

        let projection_start = u64::try_from(runtime_sources.projections.len())
            .map_err(|_| invalid("runtime source projections exceed u64"))?;
        let projection_range = effective_variant.projection_range.as_usize_range(
            input.runtime_helicity_projections.len(),
            "runtime source projections",
        )?;
        for projection in &input.runtime_helicity_projections[projection_range] {
            if projection.variant_id != effective_variant.id {
                return Err(invalid(format!(
                    "runtime variant {} owns a stale projection row",
                    effective_variant.id
                )));
            }
            runtime_sources.projections.push(DirectSourceProjectionRow {
                source_component: projection.source_component,
                full_component: projection.full_component,
            });
        }

        let dispatch_variant_id = u32_len(
            "lowered source dispatch variant",
            runtime_sources.variants.len(),
        )?;
        runtime_sources
            .variants
            .push(DirectSourceDispatchVariantDescriptor {
                embedding_start,
                projection_start,
                source_row_id,
                dispatch_domain_id: dispatch_domain,
                runtime_variant_id: binding.runtime_variant_id(),
                source_state_index: binding.source_state_index(),
                source_template_id: binding.source_template_id(),
                source_state_template_id: binding.source_state_template_id(),
                crossed_state_template_id: binding.crossed_state_template_id(),
                crossed_spin_state_class: binding.crossed_spin_state_class(),
                direct_executor_id,
                crossing_exact_factor_id: intern_direct_factor(
                    binding.crossing_factor(),
                    exact_factors,
                    exact_factor_ids,
                )?,
                embedding_count: u32_len(
                    "runtime source embedding",
                    runtime_sources.embeddings.len()
                        - usize::try_from(embedding_start)
                            .map_err(|_| invalid("embedding start exceeds usize"))?,
                )?,
                projection_count: u32_len(
                    "runtime source projection",
                    runtime_sources.projections.len()
                        - usize::try_from(projection_start)
                            .map_err(|_| invalid("projection start exceeds usize"))?,
                )?,
            });
        source_choices.push(RuntimeSourceChoice {
            source_state_index: binding.source_state_index(),
            public_helicity: binding.public_helicity(),
            dispatch_variant_id,
        });
    }
    source_choices.sort_by_key(|choice| choice.source_state_index);
    if source_choices
        .windows(2)
        .any(|pair| pair[0].source_state_index == pair[1].source_state_index)
    {
        return Err(invalid(format!(
            "all-flow-union source slot {source_slot} repeats a source-state variant"
        )));
    }
    Ok(())
}

pub(super) fn effective_runtime_helicity_variant(
    input: &OwnedRecurrenceTemplateInput,
    contract_variant_range: std::ops::Range<usize>,
    crossed_state_template_id: u32,
) -> RusticolResult<RuntimeHelicityVariantRow> {
    let candidates = input.runtime_helicity_variants[contract_variant_range]
        .iter()
        .copied()
        .filter(|variant| variant.source_state_template_id == crossed_state_template_id)
        .collect::<Vec<_>>();
    let Some(representative) = candidates.first().copied() else {
        return Err(invalid(format!(
            "runtime-helicity contract has no embedding for crossed state {crossed_state_template_id}"
        )));
    };
    for candidate in candidates.iter().copied().skip(1) {
        if !runtime_helicity_variants_share_embedding(input, representative, candidate)? {
            return Err(invalid(format!(
                "runtime-helicity contract gives crossed state {crossed_state_template_id} incompatible embeddings"
            )));
        }
    }
    Ok(representative)
}

fn runtime_helicity_variants_share_embedding(
    input: &OwnedRecurrenceTemplateInput,
    left: RuntimeHelicityVariantRow,
    right: RuntimeHelicityVariantRow,
) -> RusticolResult<bool> {
    let left_embeddings = left.embedding_range.as_usize_range(
        input.runtime_helicity_embeddings.len(),
        "left runtime source embeddings",
    )?;
    let right_embeddings = right.embedding_range.as_usize_range(
        input.runtime_helicity_embeddings.len(),
        "right runtime source embeddings",
    )?;
    let left_embeddings = &input.runtime_helicity_embeddings[left_embeddings];
    let right_embeddings = &input.runtime_helicity_embeddings[right_embeddings];
    if left_embeddings.len() != right_embeddings.len()
        || !left_embeddings
            .iter()
            .zip(right_embeddings)
            .all(|(left, right)| {
                left.full_component == right.full_component
                    && left.source_component == right.source_component
                    && left.factor_id == right.factor_id
            })
    {
        return Ok(false);
    }

    let left_projections = left.projection_range.as_usize_range(
        input.runtime_helicity_projections.len(),
        "left runtime source projections",
    )?;
    let right_projections = right.projection_range.as_usize_range(
        input.runtime_helicity_projections.len(),
        "right runtime source projections",
    )?;
    let left_projections = &input.runtime_helicity_projections[left_projections];
    let right_projections = &input.runtime_helicity_projections[right_projections];
    Ok(left_projections.len() == right_projections.len()
        && left_projections
            .iter()
            .zip(right_projections)
            .all(|(left, right)| {
                left.source_component == right.source_component
                    && left.full_component == right.full_component
            }))
}

fn validate_prepared_parameter_catalog(
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<()> {
    let input = templates.input();
    // Prepared IDs address the model-wide parameter arena. A process may
    // reference only a sparse subset of those slots, especially for UFO
    // models, so only bounded uniqueness is required here.
    let prepared_parameter_count = input.parameters.len();
    let mut seen = vec![false; prepared_parameter_count];
    for parameter in &input.parameters {
        if parameter.prepared_parameter_id == MISSING_U32 {
            continue;
        }
        let slot = usize::try_from(parameter.prepared_parameter_id)
            .map_err(|_| invalid("prepared parameter slot exceeds usize"))?;
        let present = seen.get_mut(slot).ok_or_else(|| {
            invalid(format!(
                "parameter template {} prepared runtime slot {} is out of bounds {prepared_parameter_count}",
                parameter.id, parameter.prepared_parameter_id
            ))
        })?;
        if std::mem::replace(present, true) {
            return Err(invalid(format!(
                "prepared runtime parameter slot {} is assigned more than once",
                parameter.prepared_parameter_id
            )));
        }
    }
    Ok(())
}

fn validate_direct_executor_catalog(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &PreparedDirectExecutorCatalog,
) -> RusticolResult<()> {
    let evaluators = &templates.input().evaluator_bindings;
    for key in catalog.bindings.keys().copied() {
        let PreparedDirectExecutorKey::Evaluator {
            role,
            evaluator_binding_id,
        } = key
        else {
            continue;
        };
        let evaluator = evaluators
            .get(evaluator_binding_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "prepared direct-template mapping references absent evaluator binding {evaluator_binding_id}"
                ))
            })?;
        let expected_role = direct_role_for_contract(EvaluatorContractKind::try_from(
            evaluator.contract_kind,
        )?)?
        .ok_or_else(|| {
            invalid(format!(
                "model-parameter evaluator binding {evaluator_binding_id} cannot be a direct row executor"
            ))
        })?;
        if role != expected_role {
            return Err(invalid(format!(
                "prepared direct-template mapping for evaluator binding {evaluator_binding_id} has role {role:?}, expected {expected_role:?}"
            )));
        }
    }
    for evaluator in evaluators {
        let Some(role) =
            direct_role_for_contract(EvaluatorContractKind::try_from(evaluator.contract_kind)?)?
        else {
            continue;
        };
        catalog.resolve_evaluator(role, evaluator.id)?;
    }
    Ok(())
}

fn direct_role_for_contract(
    contract: EvaluatorContractKind,
) -> RusticolResult<Option<DirectExecutorRole>> {
    Ok(match contract {
        EvaluatorContractKind::Source => Some(DirectExecutorRole::Source),
        EvaluatorContractKind::Vertex => Some(DirectExecutorRole::Contribution),
        EvaluatorContractKind::Propagator => Some(DirectExecutorRole::Finalization),
        EvaluatorContractKind::Closure => Some(DirectExecutorRole::Closure),
        EvaluatorContractKind::ModelParameter => None,
    })
}

fn component_counts(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<Vec<u32>> {
    program
        .currents()
        .iter()
        .map(|current| {
            let dimension =
                state_row(templates, current.key().current_state_template_id())?.dimension;
            u16::try_from(dimension).map_err(|_| {
                invalid(format!(
                    "current {} state dimension {dimension} exceeds the direct-row u16 component domain",
                    current.id()
                ))
            })?;
            Ok(dimension)
        })
        .collect()
}

fn state_row(
    templates: &ValidatedRecurrenceTemplateInput,
    state_template_id: u32,
) -> RusticolResult<&super::template::CurrentStateRow> {
    templates
        .input()
        .current_states
        .get(state_template_id as usize)
        .ok_or_else(|| {
            invalid(format!(
                "state template {state_template_id} is absent from the validated catalog"
            ))
        })
}

fn validated_evaluator_binding_id(
    templates: &ValidatedRecurrenceTemplateInput,
    evaluator_binding_id: u32,
    expected_contract: EvaluatorContractKind,
    role: &str,
) -> RusticolResult<u32> {
    if evaluator_binding_id == MISSING_U32 {
        return Err(invalid(format!(
            "{role} has no prepared evaluator binding to use as a provisional direct executor"
        )));
    }
    let binding = templates
        .input()
        .evaluator_bindings
        .get(evaluator_binding_id as usize)
        .ok_or_else(|| {
            invalid(format!(
                "{role} references absent evaluator binding {evaluator_binding_id}"
            ))
        })?;
    if binding.id != evaluator_binding_id
        || EvaluatorContractKind::try_from(binding.contract_kind)? != expected_contract
    {
        return Err(invalid(format!(
            "{role} evaluator binding {evaluator_binding_id} has the wrong direct role"
        )));
    }

    Ok(evaluator_binding_id)
}

fn template_u32_sequence<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    sequence_id: u32,
    label: &str,
) -> RusticolResult<&'a [u32]> {
    let input = templates.input();
    let range = input
        .u32_sequence_ranges
        .get(sequence_id as usize)
        .ok_or_else(|| invalid(format!("{label} sequence {sequence_id} is absent")))?;
    if range.id != sequence_id {
        return Err(invalid(format!(
            "{label} sequence {sequence_id} has noncanonical ID {}",
            range.id
        )));
    }
    let range = range
        .range
        .as_usize_range(input.u32_sequence_values.len(), label)?;
    Ok(&input.u32_sequence_values[range])
}

fn validate_coupling_digest(
    templates: &ValidatedRecurrenceTemplateInput,
    contribution: &super::RecurrenceContribution,
) -> RusticolResult<()> {
    let flow_id = contribution.key().quantum_flow_witness_id();
    let flow = templates
        .input()
        .quantum_flows
        .get(flow_id as usize)
        .ok_or_else(|| {
            invalid(format!(
                "contribution {} references absent quantum-flow witness {flow_id}",
                contribution.id()
            ))
        })?;
    let digest = templates
        .input()
        .digest_catalog
        .get(flow.semantic_digest_id as usize)
        .ok_or_else(|| invalid("quantum-flow semantic digest is absent"))?;
    let digest = SemanticDigest::new(digest.value)?;
    if digest != contribution.key().runtime_coupling_binding_digest() {
        return Err(invalid(format!(
            "contribution {} runtime coupling digest does not match quantum flow {flow_id}",
            contribution.id()
        )));
    }
    Ok(())
}

fn parent_state_template_ids(
    program: &RecurrenceProgram,
    parent_ids: &[u32],
) -> RusticolResult<Vec<u32>> {
    parent_ids
        .iter()
        .map(|&parent_id| {
            program
                .currents()
                .get(parent_id as usize)
                .map(|parent| parent.key().current_state_template_id())
                .ok_or_else(|| invalid(format!("parent current {parent_id} is absent")))
        })
        .collect()
}

fn parent_state_contract_matches(
    concrete_states: &[u32],
    canonical_input_order: &[u32],
    exchange_is_proven: bool,
    actual_states: &[u32],
) -> RusticolResult<bool> {
    let [left, right] = concrete_states else {
        return Err(invalid("direct parent-state contract is not binary"));
    };
    let canonical = match canonical_input_order {
        [0, 1] => [*left, *right],
        [1, 0] => [*right, *left],
        _ => {
            return Err(invalid(
                "direct canonical input order is not a binary permutation",
            ));
        }
    };
    Ok(actual_states == canonical
        || (exchange_is_proven && actual_states == [canonical[1], canonical[0]]))
}

fn validate_parent_momentum_contract(
    program: &RecurrenceProgram,
    contribution: &super::RecurrenceContribution,
) -> RusticolResult<()> {
    if contribution.key().parent_momenta().len() != contribution.parent_current_ids().len() {
        return Err(invalid(format!(
            "contribution {} parent-momentum arity is inconsistent",
            contribution.id()
        )));
    }
    for (ordinal, (&parent_id, expected)) in contribution
        .parent_current_ids()
        .iter()
        .zip(contribution.key().parent_momenta())
        .enumerate()
    {
        let parent = program
            .currents()
            .get(parent_id as usize)
            .ok_or_else(|| invalid(format!("parent current {parent_id} is absent")))?;
        if parent.key().momentum() != expected {
            return Err(invalid(format!(
                "contribution {} parent {ordinal} momentum does not match current {parent_id}",
                contribution.id()
            )));
        }
    }
    Ok(())
}

fn validate_direct_arity(label: &str, id: u32, arity: usize) -> RusticolResult<()> {
    if !(1..=2).contains(&arity) {
        return Err(invalid(format!(
            "{label} {id} has unsupported parent arity {arity}; direct plan-v2 supports only one or two parents"
        )));
    }
    Ok(())
}

fn direct_parents(
    program: &RecurrenceProgram,
    parent_ids: &[u32],
    arena: &DirectArenaLayout,
    momentum_ids: &BTreeMap<CanonicalMomentumLinearForm, u32>,
) -> RusticolResult<DirectParents> {
    validate_direct_arity("direct row", 0, parent_ids.len())?;
    let parent0 = program
        .currents()
        .get(parent_ids[0] as usize)
        .ok_or_else(|| invalid("direct parent 0 current is absent"))?;
    let parent0_assignment = arena_assignment(arena, parent0.id())?;
    let (parent1_component_base_or_sentinel, parent1_momentum_form_id_or_sentinel) =
        if let Some(&parent1_id) = parent_ids.get(1) {
            let parent1 = program
                .currents()
                .get(parent1_id as usize)
                .ok_or_else(|| invalid("direct parent 1 current is absent"))?;
            (
                arena_assignment(arena, parent1.id())?.component_base,
                momentum_id(momentum_ids, parent1.key().momentum())?,
            )
        } else {
            (DIRECT_NONE_U32, DIRECT_NONE_U32)
        };
    Ok(DirectParents {
        parent0_component_base: parent0_assignment.component_base,
        parent1_component_base_or_sentinel,
        parent0_momentum_form_id: momentum_id(momentum_ids, parent0.key().momentum())?,
        parent1_momentum_form_id_or_sentinel,
    })
}

fn external_source_count(program: &RecurrenceProgram) -> RusticolResult<u32> {
    let slots = program
        .currents()
        .iter()
        .filter(|current| current.is_source())
        .map(only_source_slot)
        .collect::<RusticolResult<BTreeSet<_>>>()?;
    let count = slots
        .iter()
        .next_back()
        .copied()
        .and_then(|slot| slot.checked_add(1))
        .ok_or_else(|| invalid("semantic program contains no source currents"))?;
    if slots.iter().copied().ne(0..count) {
        return Err(invalid(
            "source currents do not cover a contiguous external-source domain",
        ));
    }
    Ok(count)
}

fn only_source_slot(current: &RecurrenceCurrent) -> RusticolResult<u32> {
    let [source_slot] = current.key().support_source_slots() else {
        return Err(invalid(format!(
            "source current {} does not cover exactly one source slot",
            current.id()
        )));
    };
    Ok(*source_slot)
}

fn current_stage(current: &RecurrenceCurrent) -> RusticolResult<u16> {
    let stage = current
        .key()
        .support_source_slots()
        .len()
        .checked_sub(1)
        .ok_or_else(|| invalid(format!("current {} has empty source support", current.id())))?;
    u16::try_from(stage).map_err(|_| invalid(format!("current {} stage exceeds u16", current.id())))
}

fn direct_event(event: u64, label: &str) -> RusticolResult<u32> {
    let event = event
        .checked_sub(1)
        .ok_or_else(|| invalid(format!("{label} precedes direct stage zero")))?;
    u32::try_from(event).map_err(|_| invalid(format!("{label} exceeds u32")))
}

fn arena_assignment(
    arena: &DirectArenaLayout,
    current_id: u32,
) -> RusticolResult<super::DirectArenaAssignment> {
    arena.assignment(current_id).ok_or_else(|| {
        invalid(format!(
            "arena assignment for semantic current {current_id} is absent"
        ))
    })
}

type DirectMomentumTables = (
    BTreeMap<CanonicalMomentumLinearForm, u32>,
    Vec<DirectMomentumFormDescriptor>,
    Vec<DirectMomentumTerm>,
);

fn intern_momentum_forms(
    program: &RecurrenceProgram,
    external_source_count: u32,
) -> RusticolResult<DirectMomentumTables> {
    let forms = program
        .currents()
        .iter()
        .map(|current| current.key().momentum().clone())
        .collect::<BTreeSet<_>>();
    let mut ids = BTreeMap::new();
    let mut descriptors = Vec::with_capacity(forms.len());
    let mut terms = Vec::new();
    for form in forms {
        for term in form.terms() {
            if term.source_slot >= external_source_count {
                return Err(invalid(format!(
                    "momentum form references source slot {} outside the source domain",
                    term.source_slot
                )));
            }
        }
        let id = u32_len("momentum form", descriptors.len())?;
        let term_start =
            u64::try_from(terms.len()).map_err(|_| invalid("momentum term table exceeds u64"))?;
        terms.extend(form.terms().iter().map(|term| DirectMomentumTerm {
            source_slot: term.source_slot,
            coefficient: term.coefficient,
        }));
        descriptors.push(DirectMomentumFormDescriptor {
            term_start,
            term_count: u32_len("momentum term", form.terms().len())?,
        });
        ids.insert(form, id);
    }
    Ok((ids, descriptors, terms))
}

fn momentum_id(
    ids: &BTreeMap<CanonicalMomentumLinearForm, u32>,
    form: &CanonicalMomentumLinearForm,
) -> RusticolResult<u32> {
    ids.get(form)
        .copied()
        .ok_or_else(|| invalid("semantic momentum form was not interned"))
}

fn intern_exact_factors(
    program: &RecurrenceProgram,
) -> RusticolResult<(
    Vec<ExactComplexRational>,
    BTreeMap<ExactComplexRational, u32>,
)> {
    let mut unique = BTreeSet::from([ExactComplexRational::ONE]);
    unique.extend(
        program
            .currents()
            .iter()
            .filter_map(RecurrenceCurrent::source_exact_factor),
    );
    unique.extend(
        program
            .contributions()
            .iter()
            .map(super::RecurrenceContribution::exact_factor),
    );
    unique.extend(
        program
            .finalizations()
            .iter()
            .map(super::RecurrenceFinalization::exact_factor),
    );
    unique.extend(
        program
            .closure_terms()
            .iter()
            .map(super::RecurrenceClosureTerm::exact_factor),
    );
    unique.extend(
        program
            .replay_targets()
            .iter()
            .map(super::RecurrenceReplayTarget::amplitude_factor),
    );

    let mut factors = vec![ExactComplexRational::ONE];
    factors.extend(
        unique
            .into_iter()
            .filter(|factor| *factor != ExactComplexRational::ONE),
    );
    let ids = factors
        .iter()
        .copied()
        .enumerate()
        .map(|(id, factor)| Ok((factor, u32_len("exact factor", id)?)))
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    Ok((factors, ids))
}

fn exact_factor_id(
    ids: &BTreeMap<ExactComplexRational, u32>,
    factor: ExactComplexRational,
) -> RusticolResult<u32> {
    ids.get(&factor)
        .copied()
        .ok_or_else(|| invalid("semantic exact factor was not interned"))
}

fn intern_direct_factor(
    factor: ExactComplexRational,
    factors: &mut Vec<ExactComplexRational>,
    ids: &mut BTreeMap<ExactComplexRational, u32>,
) -> RusticolResult<u32> {
    if let Some(id) = ids.get(&factor).copied() {
        return Ok(id);
    }
    let id = u32_len("exact factor", factors.len())?;
    factors.push(factor);
    ids.insert(factor, id);
    Ok(id)
}

fn template_exact_factor(
    templates: &ValidatedRecurrenceTemplateInput,
    factor_id: u32,
) -> RusticolResult<ExactComplexRational> {
    let input = templates.input();
    let row = input
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| invalid(format!("template exact factor {factor_id} is absent")))?;
    if row.id != factor_id {
        return Err(invalid(format!(
            "template exact factor {factor_id} has noncanonical ID {}",
            row.id
        )));
    }
    ExactComplexRational::parse_parts(
        template_string(input, row.real_numerator_string_id, "exact real numerator")?,
        template_string(
            input,
            row.real_denominator_string_id,
            "exact real denominator",
        )?,
        template_string(
            input,
            row.imag_numerator_string_id,
            "exact imaginary numerator",
        )?,
        template_string(
            input,
            row.imag_denominator_string_id,
            "exact imaginary denominator",
        )?,
    )
    .map_err(|error| {
        invalid(format!(
            "template exact factor {factor_id} is invalid: {error}"
        ))
    })
}

fn template_string<'a>(
    input: &'a super::template::OwnedRecurrenceTemplateInput,
    string_id: u32,
    label: &str,
) -> RusticolResult<&'a str> {
    let range = input
        .string_ranges
        .get(string_id as usize)
        .ok_or_else(|| invalid(format!("{label} string {string_id} is absent")))?;
    let bytes = &input.string_bytes[range.as_usize_range(input.string_bytes.len(), label)?];
    std::str::from_utf8(bytes)
        .map_err(|error| invalid(format!("{label} string {string_id} is not UTF-8: {error}")))
}

fn intern_closure_factor_block(
    coefficients: &[ExactComplexRational],
    factors: &mut Vec<ExactComplexRational>,
    blocks: &mut BTreeMap<Vec<ExactComplexRational>, (u32, u16)>,
) -> RusticolResult<(u32, u16)> {
    if let Some(block) = blocks.get(coefficients) {
        return Ok(*block);
    }
    let start = u32_len("closure component-factor start", factors.len())?;
    let count = u16::try_from(coefficients.len())
        .map_err(|_| invalid("closure component count exceeds u16"))?;
    if count == 0 {
        return Err(invalid("closure component-factor block is empty"));
    }
    factors.extend_from_slice(coefficients);
    let block = (start, count);
    blocks.insert(coefficients.to_vec(), block);
    Ok(block)
}

fn lower_amplitude_destinations(
    program: &RecurrenceProgram,
    closures: &[ClosureDraft],
    destination_domain_ids: &[u32],
) -> RusticolResult<Vec<DirectAmplitudeDestinationDescriptor>> {
    let mut next_closure = 0usize;
    let mut result = Vec::with_capacity(program.amplitude_destinations().len());
    for destination in program.amplitude_destinations().iter().copied() {
        let start = next_closure;
        while next_closure < closures.len()
            && closures[next_closure].amplitude_destination_id == destination.id()
        {
            next_closure += 1;
        }
        let count = next_closure - start;
        if count as u64 != destination.closure_range().count {
            return Err(invalid(format!(
                "amplitude destination {} lowered {count} closure rows, expected {}",
                destination.id(),
                destination.closure_range().count
            )));
        }
        result.push(DirectAmplitudeDestinationDescriptor {
            closure_row_start: u64::try_from(start)
                .map_err(|_| invalid("closure row start exceeds u64"))?,
            id: destination.id(),
            target_sector_id: destination.target_sector_id(),
            target_helicity_id_or_sentinel: destination
                .target_helicity_id()
                .unwrap_or(DIRECT_NONE_U32),
            closure_row_count: u32_len("amplitude-destination closure row", count)?,
            selector_domain_id: *destination_domain_ids
                .get(destination.id() as usize)
                .ok_or_else(|| invalid("amplitude destination selector domain is absent"))?,
        });
    }
    if next_closure != closures.len() {
        return Err(invalid(
            "closure rows are not fully owned by amplitude destinations",
        ));
    }
    Ok(result)
}

type DirectReplayTables = (
    Vec<DirectReplayTargetDescriptor>,
    Vec<u32>,
    Vec<i32>,
    Vec<u32>,
);

fn lower_replay_targets(
    program: &RecurrenceProgram,
    exact_factor_ids: &BTreeMap<ExactComplexRational, u32>,
    sector_domain_ids: &[Option<u32>],
) -> RusticolResult<DirectReplayTables> {
    if !program.strategy().uses_topology_replay_targets() {
        if !program.replay_targets().is_empty() {
            return Err(invalid(
                "non-replay recurrence program unexpectedly carries replay targets",
            ));
        }
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let mut source_template_by_state = BTreeMap::<(u32, u32), u32>::new();
    let mut source_state_by_template = BTreeMap::<(u32, u32), u32>::new();
    for current in program
        .currents()
        .iter()
        .filter(|current| current.is_source())
    {
        let source_slot = only_source_slot(current)?;
        let source_template_id = match current.key().source_binding() {
            CurrentSourceBinding::FixedTemplate(source_template_id) => *source_template_id,
            _ => {
                return Err(invalid(format!(
                    "topology-replay source current {} has no fixed source template",
                    current.id()
                )));
            }
        };
        let ancestry = current.key().helicity_identity().local_source_states();
        if ancestry.len() != 1 || ancestry[0].source_slot() != source_slot {
            return Err(invalid(format!(
                "topology-replay source current {} has invalid local source ancestry",
                current.id()
            )));
        }
        let state_index = ancestry[0].state_index();
        if source_template_by_state
            .insert((source_slot, state_index), source_template_id)
            .is_some_and(|previous| previous != source_template_id)
        {
            return Err(invalid(format!(
                "source slot {source_slot} state {state_index} has inconsistent source templates"
            )));
        }
        if source_state_by_template
            .insert((source_slot, source_template_id), state_index)
            .is_some_and(|previous| previous != state_index)
        {
            return Err(invalid(format!(
                "source slot {source_slot} template {source_template_id} is not a unique source state"
            )));
        }
    }

    let resolved_by_states = program
        .resolved_helicities()
        .iter()
        .map(|helicity| (helicity.source_states().to_vec(), helicity.id()))
        .collect::<BTreeMap<_, _>>();
    if resolved_by_states.len() != program.resolved_helicities().len() {
        return Err(invalid(
            "topology-replay program repeats a resolved source-state assignment",
        ));
    }

    let source_count = usize::try_from(external_source_count(program)?)
        .map_err(|_| invalid("external source count exceeds usize"))?;
    let mut source_permutations = Vec::new();
    let mut replay_momentum_signs = Vec::new();
    let mut replay_helicity_map = Vec::new();
    let mut replay_targets = Vec::with_capacity(program.replay_targets().len());
    for target in program.replay_targets() {
        let source_permutation_start = u64::try_from(source_permutations.len())
            .map_err(|_| invalid("source permutation table exceeds u64"))?;
        source_permutations.extend_from_slice(target.source_slot_permutation());
        replay_momentum_signs.extend_from_slice(target.source_momentum_signs());

        let helicity_map_start = u64::try_from(replay_helicity_map.len())
            .map_err(|_| invalid("replay helicity-map table exceeds u64"))?;
        for helicity in program.resolved_helicities() {
            let mut mapped = vec![None; source_count];
            for assignment in helicity.source_states().iter().copied() {
                let representative_slot = assignment.source_slot();
                let source_template_id = source_template_by_state
                    .get(&(representative_slot, assignment.state_index()))
                    .copied()
                    .ok_or_else(|| {
                        invalid(format!(
                            "resolved helicity {} source slot {representative_slot} state {} has no source template",
                            helicity.id(),
                            assignment.state_index()
                        ))
                    })?;
                let target_slot = target
                    .source_slot_permutation()
                    .get(representative_slot as usize)
                    .copied()
                    .ok_or_else(|| invalid("replay source permutation is out of bounds"))?;
                let target_state_index = source_state_by_template
                    .get(&(target_slot, source_template_id))
                    .copied()
                    .ok_or_else(|| {
                        invalid(format!(
                            "replay target {} cannot transport source template {source_template_id} from slot {representative_slot} to slot {target_slot}",
                            target.target_sector_id()
                        ))
                    })?;
                let target_index = usize::try_from(target_slot)
                    .map_err(|_| invalid("replay target source slot exceeds usize"))?;
                if target_index >= source_count || mapped[target_index].is_some() {
                    return Err(invalid(format!(
                        "replay target {} source mapping is not bijective",
                        target.target_sector_id()
                    )));
                }
                mapped[target_index] =
                    Some(SourceStateAssignment::new(target_slot, target_state_index));
            }
            let mapped = mapped
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    invalid(format!(
                        "replay target {} does not map every source state",
                        target.target_sector_id()
                    ))
                })?;
            let mapped_id = resolved_by_states.get(&mapped).copied().ok_or_else(|| {
                invalid(format!(
                    "replay target {} maps resolved helicity {} outside retained source-state coverage",
                    target.target_sector_id(),
                    helicity.id()
                ))
            })?;
            replay_helicity_map.push(mapped_id);
        }

        replay_targets.push(DirectReplayTargetDescriptor {
            public_flow_id: target.target_sector_id(),
            representative_id: target.materialized_sector_id(),
            source_permutation_start,
            source_permutation_count: u32_len(
                "replay source permutation",
                target.source_slot_permutation().len(),
            )?,
            helicity_map_start,
            helicity_map_count: u32_len(
                "replay helicity map",
                program.resolved_helicities().len(),
            )?,
            phase_exact_factor_id: exact_factor_id(exact_factor_ids, target.amplitude_factor())?,
            // RecurrenceProgram stores one exact row per public target and has
            // no separate orbit-multiplicity field.
            multiplicity: 1,
            selector_domain_id: *sector_domain_ids
                .get(target.materialized_sector_id() as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| invalid("replay representative selector domain is absent"))?,
        });
    }
    Ok((
        replay_targets,
        source_permutations,
        replay_momentum_signs,
        replay_helicity_map,
    ))
}

fn lower_resolved_helicities(
    program: &RecurrenceProgram,
) -> RusticolResult<(
    Vec<DirectResolvedHelicityDescriptor>,
    Vec<DirectSourceStateAssignment>,
    Vec<i32>,
)> {
    let mut descriptors = Vec::with_capacity(program.resolved_helicities().len());
    let mut source_states = Vec::new();
    let mut public_helicities = Vec::new();
    for helicity in program.resolved_helicities() {
        let source_state_start = u64::try_from(source_states.len())
            .map_err(|_| invalid("resolved-helicity source states exceed u64"))?;
        let public_helicity_start = u64::try_from(public_helicities.len())
            .map_err(|_| invalid("resolved-helicity public values exceed u64"))?;
        source_states.extend(helicity.source_states().iter().map(|assignment| {
            DirectSourceStateAssignment {
                source_slot: assignment.source_slot(),
                state_index: assignment.state_index(),
            }
        }));
        public_helicities.extend_from_slice(helicity.public_helicities());
        descriptors.push(DirectResolvedHelicityDescriptor {
            source_state_start,
            source_selection_start: 0,
            public_helicity_start,
            id: helicity.id(),
            source_state_count: u32_len(
                "resolved-helicity source state",
                helicity.source_states().len(),
            )?,
            source_selection_count: 0,
            public_helicity_count: u32_len(
                "resolved-helicity public value",
                helicity.public_helicities().len(),
            )?,
            selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
        });
    }
    Ok((descriptors, source_states, public_helicities))
}

type DirectUnionResolvedHelicityTables = (
    Vec<DirectResolvedHelicityDescriptor>,
    Vec<DirectSourceStateAssignment>,
    Vec<DirectResolvedSourceSelection>,
    Vec<i32>,
);

pub(super) fn lower_union_resolved_helicities(
    choices_by_source: &[Vec<RuntimeSourceChoice>],
    retained_helicity_count: u64,
) -> RusticolResult<DirectUnionResolvedHelicityTables> {
    if choices_by_source.is_empty() || choices_by_source.iter().any(Vec::is_empty) {
        return Err(invalid(
            "all-flow-union runtime source choices do not cover every external source",
        ));
    }
    let cartesian_count = choices_by_source.iter().try_fold(1_u64, |count, choices| {
        count
            .checked_mul(
                u64::try_from(choices.len())
                    .map_err(|_| invalid("runtime source-choice count exceeds u64"))?,
            )
            .ok_or_else(|| invalid("resolved-helicity Cartesian product exceeds u64"))
    })?;
    if cartesian_count != retained_helicity_count {
        return Err(invalid(format!(
            "all-flow-union runtime source choices span {cartesian_count} helicities, expected {retained_helicity_count}"
        )));
    }
    let descriptor_capacity = usize::try_from(cartesian_count)
        .map_err(|_| invalid("resolved-helicity count exceeds usize"))?;
    let mut descriptors = Vec::with_capacity(descriptor_capacity);
    let mut source_states = Vec::with_capacity(
        descriptor_capacity
            .checked_mul(choices_by_source.len())
            .ok_or_else(|| invalid("resolved source-state capacity exceeds usize"))?,
    );
    let mut source_selections = Vec::with_capacity(source_states.capacity());
    let mut public_helicities = Vec::with_capacity(source_states.capacity());
    let mut indices = vec![0usize; choices_by_source.len()];

    for id in 0..descriptor_capacity {
        let source_state_start = u64::try_from(source_states.len())
            .map_err(|_| invalid("resolved source states exceed u64"))?;
        let source_selection_start = u64::try_from(source_selections.len())
            .map_err(|_| invalid("resolved source selections exceed u64"))?;
        let public_helicity_start = u64::try_from(public_helicities.len())
            .map_err(|_| invalid("resolved public helicities exceed u64"))?;
        for (source_slot, (&choice_index, choices)) in
            indices.iter().zip(choices_by_source).enumerate()
        {
            let choice = choices[choice_index];
            let source_slot = u32_len("resolved source slot", source_slot)?;
            source_states.push(DirectSourceStateAssignment {
                source_slot,
                state_index: choice.source_state_index,
            });
            source_selections.push(DirectResolvedSourceSelection {
                source_slot,
                dispatch_variant_id: choice.dispatch_variant_id,
            });
            public_helicities.push(choice.public_helicity);
        }
        let source_count = u32_len("resolved source", choices_by_source.len())?;
        descriptors.push(DirectResolvedHelicityDescriptor {
            source_state_start,
            source_selection_start,
            public_helicity_start,
            id: u32_len("resolved helicity", id)?,
            source_state_count: source_count,
            source_selection_count: source_count,
            public_helicity_count: source_count,
            selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
        });

        for source_slot in (0..indices.len()).rev() {
            indices[source_slot] += 1;
            if indices[source_slot] < choices_by_source[source_slot].len() {
                break;
            }
            indices[source_slot] = 0;
        }
    }
    Ok((
        descriptors,
        source_states,
        source_selections,
        public_helicities,
    ))
}

fn append_row_groups(
    keys: impl IntoIterator<Item = (u16, u32)>,
    role: DirectExecutorRole,
    destination_operation: DirectDestinationOperation,
    row_groups: &mut Vec<DirectRowGroupDescriptor>,
) -> RusticolResult<()> {
    let keys = keys.into_iter().collect::<Vec<_>>();
    let mut start = 0usize;
    while start < keys.len() {
        let (stage, direct_executor_id) = keys[start];
        let mut stop = start + 1;
        while stop < keys.len() && keys[stop] == (stage, direct_executor_id) {
            stop += 1;
        }
        row_groups.push(DirectRowGroupDescriptor {
            stage,
            role,
            destination_operation,
            direct_executor_id,
            row_start: u64::try_from(start)
                .map_err(|_| invalid("direct row_group row start exceeds u64"))?,
            row_count: u32_len("direct row_group row", stop - start)?,
        });
        start = stop;
    }
    Ok(())
}

fn u32_len(label: &str, value: usize) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} count exceeds u32")))
}

#[cfg(test)]
#[path = "direct_lowering_tests.rs"]
mod tests;
#[cfg(test)]
pub(crate) use tests::validated_template as validated_template_fixture;
