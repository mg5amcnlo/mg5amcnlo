// SPDX-License-Identifier: 0BSD

//! Immutable direct-arena recurrence plan.
//!
//! Semantic current IDs are deliberately distinct from physical arena
//! component ranges. The latter may be reused after interval coloring.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::{
    ClosureProofMetadataV2, ExactComplexRational, RecurrenceStrategy, SemanticDigest,
    closure_component_factor_digest_v2, closure_selector_domain_digest_v2,
};
use crate::{RusticolError, RusticolResult};

pub const RECURRENCE_DIRECT_PLAN_ABI: &str = "pyamplicol-recurrence-plan-v2";
pub const RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI: &str = "pyamplicol-recurrence-runtime-layout-v2";
pub const RECURRENCE_DIRECT_TEMPLATE_ABI: &str = "pyamplicol-recurrence-direct-template-v1";
pub const RECURRENCE_DIRECT_RUNTIME_CAPABILITY: &str =
    "rusticol.recurrence-direct-arena.complex-f64.v1";
pub const DIRECT_NONE_U32: u32 = u32::MAX;
pub const DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION: u32 = 1 << 0;
/// The row scale-copies a certified representative current. For these rows
/// `parent0_momentum_form_id` stores the component count; no momentum is read.
pub const DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE: u32 = 1 << 1;
const DIRECT_CONTRIBUTION_FLAGS_KNOWN: u32 =
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION | DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("recurrence direct plan: {}", message.into()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum DirectNodeKind {
    Source = 0,
    Current = 1,
}

impl TryFrom<u16> for DirectNodeKind {
    type Error = RusticolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Source),
            1 => Ok(Self::Current),
            _ => Err(invalid(format!("unknown node-kind discriminant {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, bincode::Encode, bincode::Decode)]
#[repr(u16)]
pub enum DirectExecutorRole {
    Source = 0,
    Contribution = 1,
    Finalization = 2,
    Closure = 3,
}

impl TryFrom<u16> for DirectExecutorRole {
    type Error = RusticolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Source),
            1 => Ok(Self::Contribution),
            2 => Ok(Self::Finalization),
            3 => Ok(Self::Closure),
            _ => Err(invalid(format!(
                "unknown direct-executor-role discriminant {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum DirectDestinationOperation {
    Initialize = 0,
    Add = 1,
    FinalizeInPlace = 2,
    ClosureAdd = 3,
}

impl TryFrom<u16> for DirectDestinationOperation {
    type Error = RusticolError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Initialize),
            1 => Ok(Self::Add),
            2 => Ok(Self::FinalizeInPlace),
            3 => Ok(Self::ClosureAdd),
            _ => Err(invalid(format!(
                "unknown destination-operation discriminant {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectCurrentDescriptor {
    pub semantic_current_id: u32,
    pub node_kind: DirectNodeKind,
    pub state_template_id: u32,
    pub component_base: u32,
    pub component_count: u16,
    pub momentum_form_id: u32,
    pub stage: u16,
    pub selector_domain_id: u32,
    pub first_use: u32,
    pub last_use: u32,
    pub source_row_or_sentinel: u32,
    pub finalization_row_or_sentinel: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, bincode::Encode, bincode::Decode)]
#[repr(C)]
pub struct DirectSourceRow {
    pub source_slot: u32,
    pub destination_component_base: u32,
    pub momentum_form_id: u32,
    pub source_template_or_dispatch_domain: u32,
    pub spin_state_class: i32,
    pub exact_factor_id: u32,
    pub selector_domain_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, bincode::Encode, bincode::Decode)]
#[repr(C)]
pub struct DirectContributionRow {
    pub parent0_component_base: u32,
    pub parent1_component_base_or_sentinel: u32,
    pub parent0_momentum_form_id: u32,
    pub parent1_momentum_form_id_or_sentinel: u32,
    pub destination_component_base: u32,
    pub exact_factor_id: u32,
    pub selector_domain_id: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, bincode::Encode, bincode::Decode)]
#[repr(C)]
pub struct DirectFinalizationRow {
    pub component_base: u32,
    pub component_count: u16,
    pub momentum_form_id: u32,
    pub exact_factor_id: u32,
    pub selector_domain_id: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, bincode::Encode, bincode::Decode)]
#[repr(C)]
pub struct DirectClosureRow {
    pub parent0_component_base: u32,
    pub parent1_component_base_or_sentinel: u32,
    pub parent0_momentum_form_id: u32,
    pub parent1_momentum_form_id_or_sentinel: u32,
    pub amplitude_destination_id: u32,
    pub exact_factor_id: u32,
    pub component_factor_start: u32,
    pub component_count: u16,
    pub selector_domain_id: u32,
    /// Cold closure-proof group ID. Runtime executors deliberately ignore it.
    ///
    /// This occupies the former unused closure flags word, preserving the
    /// native 40-byte row layout.
    pub flags: u32,
}

impl DirectClosureRow {
    pub const fn proof_group_id(&self) -> u32 {
        self.flags
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectRowGroupDescriptor {
    pub stage: u16,
    pub role: DirectExecutorRole,
    pub destination_operation: DirectDestinationOperation,
    pub direct_executor_id: u32,
    pub row_start: u64,
    pub row_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectMomentumFormDescriptor {
    pub term_start: u64,
    pub term_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectMomentumTerm {
    pub source_slot: u32,
    pub coefficient: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectSelectorDomainDescriptor {
    pub word_start: u64,
    pub word_count: u32,
}

/// Auditable amount of persisted direct work active for one physical sector.
///
/// Topology-replay artifacts retain one union plan so public flows can share
/// currents.  These counters make the distinction between that persisted
/// union and the rows actually executed by a selected-flow call explicit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectSelectorWorkSummary {
    pub physical_sector_id: u32,
    pub current_count: u64,
    pub semantic_component_count: u64,
    pub source_row_count: u64,
    pub contribution_count: u64,
    pub finalization_count: u64,
    pub closure_count: u64,
    pub amplitude_destination_count: u64,
}

impl DirectSelectorWorkSummary {
    pub const fn row_count(&self) -> u64 {
        self.source_row_count
            + self.contribution_count
            + self.finalization_count
            + self.closure_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectReplayTargetDescriptor {
    pub public_flow_id: u32,
    pub representative_id: u32,
    pub source_permutation_start: u64,
    pub source_permutation_count: u32,
    pub helicity_map_start: u64,
    pub helicity_map_count: u32,
    pub phase_exact_factor_id: u32,
    pub multiplicity: u32,
    pub selector_domain_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectAmplitudeDestinationDescriptor {
    pub closure_row_start: u64,
    pub id: u32,
    pub target_sector_id: u32,
    pub target_helicity_id_or_sentinel: u32,
    pub closure_row_count: u32,
    pub selector_domain_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectResolvedHelicityDescriptor {
    pub source_state_start: u64,
    pub source_selection_start: u64,
    pub public_helicity_start: u64,
    pub id: u32,
    pub source_state_count: u32,
    pub source_selection_count: u32,
    pub public_helicity_count: u32,
    pub selector_domain_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectSourceStateAssignment {
    pub source_slot: u32,
    pub state_index: u32,
}

/// One concrete process-bound source variant referenced by resolved helicities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectSourceDispatchVariantDescriptor {
    pub embedding_start: u64,
    pub projection_start: u64,
    pub source_row_id: u32,
    pub dispatch_domain_id: u32,
    pub runtime_variant_id: u32,
    pub source_state_index: u32,
    pub source_template_id: u32,
    pub source_state_template_id: u32,
    pub crossed_state_template_id: u32,
    pub crossed_spin_state_class: i32,
    pub direct_executor_id: u32,
    pub crossing_exact_factor_id: u32,
    pub embedding_count: u32,
    pub projection_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectSourceEmbeddingRow {
    pub full_component: u32,
    pub source_component_or_sentinel: u32,
    pub exact_factor_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectSourceProjectionRow {
    pub source_component: u32,
    pub full_component: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectResolvedSourceSelection {
    pub source_slot: u32,
    pub dispatch_variant_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRecurrencePlanParts {
    pub strategy: RecurrenceStrategy,
    pub semantic_digest: SemanticDigest,
    pub prepared_pack_digest: SemanticDigest,
    pub direct_template_catalog_digest: SemanticDigest,
    pub point_tile_size: u32,
    pub workspace_mib: u32,
    pub current_arena_components: u32,
    pub physical_sector_count: u32,
    pub retained_helicity_count: u64,
    pub amplitude_destination_count: u32,
    pub parameter_value_count: u32,
    pub external_source_count: u32,
    pub state_template_count: u32,
    pub source_template_count: u32,
    pub source_template_or_dispatch_count: u32,
    pub runtime_helicity_contract_count: u32,
    pub runtime_helicity_variant_count: u32,
    pub direct_executor_count: u32,
    pub currents: Vec<DirectCurrentDescriptor>,
    pub sources: Vec<DirectSourceRow>,
    pub contributions: Vec<DirectContributionRow>,
    pub finalizations: Vec<DirectFinalizationRow>,
    pub closures: Vec<DirectClosureRow>,
    pub row_groups: Vec<DirectRowGroupDescriptor>,
    pub momentum_forms: Vec<DirectMomentumFormDescriptor>,
    pub momentum_terms: Vec<DirectMomentumTerm>,
    pub selector_domains: Vec<DirectSelectorDomainDescriptor>,
    pub selector_words: Vec<u64>,
    pub replay_targets: Vec<DirectReplayTargetDescriptor>,
    pub source_permutations: Vec<u32>,
    pub replay_momentum_signs: Vec<i32>,
    pub replay_helicity_map: Vec<u32>,
    pub amplitude_destinations: Vec<DirectAmplitudeDestinationDescriptor>,
    pub resolved_helicities: Vec<DirectResolvedHelicityDescriptor>,
    pub source_state_assignments: Vec<DirectSourceStateAssignment>,
    pub source_dispatch_variants: Vec<DirectSourceDispatchVariantDescriptor>,
    pub source_embeddings: Vec<DirectSourceEmbeddingRow>,
    pub source_projections: Vec<DirectSourceProjectionRow>,
    pub resolved_source_selections: Vec<DirectResolvedSourceSelection>,
    pub public_helicities: Vec<i32>,
    pub exact_factors: Vec<ExactComplexRational>,
    pub closure_proofs: ClosureProofMetadataV2,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRecurrencePlan {
    parts: DirectRecurrencePlanParts,
    runtime_layout_digest: SemanticDigest,
    closure_proof_rows_retained: bool,
}

fn selector_domain_contains_parts(
    parts: &DirectRecurrencePlanParts,
    selector_domain_id: u32,
    physical_sector_id: u32,
) -> RusticolResult<bool> {
    if physical_sector_id >= parts.physical_sector_count {
        return Err(invalid("physical selector sector is out of bounds"));
    }
    let descriptor = parts
        .selector_domains
        .get(selector_domain_id as usize)
        .ok_or_else(|| invalid("selector domain is absent"))?;
    let start = usize::try_from(descriptor.word_start)
        .map_err(|_| invalid("selector-domain word start exceeds usize"))?;
    let end = start
        .checked_add(descriptor.word_count as usize)
        .ok_or_else(|| invalid("selector-domain word range overflows usize"))?;
    let words = parts
        .selector_words
        .get(start..end)
        .ok_or_else(|| invalid("selector-domain word range is out of bounds"))?;
    if words == [u64::MAX] {
        return Ok(true);
    }
    let word = physical_sector_id as usize / u64::BITS as usize;
    let bit = physical_sector_id % u64::BITS;
    Ok(words
        .get(word)
        .is_some_and(|value| value & (1_u64 << bit) != 0))
}

impl DirectRecurrencePlan {
    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub fn new(mut parts: DirectRecurrencePlanParts) -> RusticolResult<Self> {
        #[cfg(feature = "on-the-fly-test-support")]
        super::on_the_fly::reject_forbidden_work_if_probed(
            super::on_the_fly::OnTheFlyForbiddenWorkV1::DirectPlanMaterialization,
        )?;
        canonicalize_contribution_initialization(&mut parts)?;
        validate_parts(&parts)?;
        let runtime_layout_digest = digest_parts(&parts)?;
        Ok(Self {
            parts,
            runtime_layout_digest,
            closure_proof_rows_retained: true,
        })
    }

    pub(crate) fn into_runtime_compacted(mut self) -> Self {
        self.parts.closure_proofs.discard_rows_for_runtime();
        self.closure_proof_rows_retained = false;
        self
    }

    pub(crate) fn ensure_serializable(&self) -> RusticolResult<()> {
        if !self.closure_proof_rows_retained {
            return Err(invalid(
                "runtime-compacted plan cannot be serialized without closure-proof rows",
            ));
        }
        Ok(())
    }

    pub const fn strategy(&self) -> RecurrenceStrategy {
        self.parts.strategy
    }

    pub const fn semantic_digest(&self) -> SemanticDigest {
        self.parts.semantic_digest
    }

    pub const fn prepared_pack_digest(&self) -> SemanticDigest {
        self.parts.prepared_pack_digest
    }

    pub const fn direct_template_catalog_digest(&self) -> SemanticDigest {
        self.parts.direct_template_catalog_digest
    }

    pub const fn runtime_layout_digest(&self) -> SemanticDigest {
        self.runtime_layout_digest
    }

    pub const fn point_tile_size(&self) -> u32 {
        self.parts.point_tile_size
    }

    pub const fn workspace_mib(&self) -> u32 {
        self.parts.workspace_mib
    }

    pub const fn current_arena_components(&self) -> u32 {
        self.parts.current_arena_components
    }

    pub const fn physical_sector_count(&self) -> u32 {
        self.parts.physical_sector_count
    }

    pub const fn retained_helicity_count(&self) -> u64 {
        self.parts.retained_helicity_count
    }

    pub const fn amplitude_destination_count(&self) -> u32 {
        self.parts.amplitude_destination_count
    }

    pub const fn parameter_value_count(&self) -> u32 {
        self.parts.parameter_value_count
    }

    pub const fn external_source_count(&self) -> u32 {
        self.parts.external_source_count
    }

    pub const fn state_template_count(&self) -> u32 {
        self.parts.state_template_count
    }

    pub const fn source_template_count(&self) -> u32 {
        self.parts.source_template_count
    }

    pub const fn source_template_or_dispatch_count(&self) -> u32 {
        self.parts.source_template_or_dispatch_count
    }

    pub const fn runtime_helicity_contract_count(&self) -> u32 {
        self.parts.runtime_helicity_contract_count
    }

    pub const fn runtime_helicity_variant_count(&self) -> u32 {
        self.parts.runtime_helicity_variant_count
    }

    pub const fn direct_executor_count(&self) -> u32 {
        self.parts.direct_executor_count
    }

    pub fn currents(&self) -> &[DirectCurrentDescriptor] {
        &self.parts.currents
    }

    pub fn sources(&self) -> &[DirectSourceRow] {
        &self.parts.sources
    }

    pub fn contributions(&self) -> &[DirectContributionRow] {
        &self.parts.contributions
    }

    pub fn finalizations(&self) -> &[DirectFinalizationRow] {
        &self.parts.finalizations
    }

    pub fn closures(&self) -> &[DirectClosureRow] {
        &self.parts.closures
    }

    pub fn row_groups(&self) -> &[DirectRowGroupDescriptor] {
        &self.parts.row_groups
    }

    pub fn momentum_forms(&self) -> &[DirectMomentumFormDescriptor] {
        &self.parts.momentum_forms
    }

    pub fn momentum_terms(&self) -> &[DirectMomentumTerm] {
        &self.parts.momentum_terms
    }

    pub fn selector_domains(&self) -> &[DirectSelectorDomainDescriptor] {
        &self.parts.selector_domains
    }

    pub fn selector_words(&self) -> &[u64] {
        &self.parts.selector_words
    }

    /// Return whether one physical selector bit belongs to a persisted domain.
    ///
    /// Direct-plan-v2 artifacts produced before selector-aware lowering used
    /// one literal `[u64::MAX]` domain for every row, even when a process had
    /// more than 64 physical sectors.  Preserve that encoding as the
    /// backwards-compatible universal domain while allowing newer plans to
    /// persist ordinary multiword sector masks.
    pub fn selector_domain_contains(
        &self,
        selector_domain_id: u32,
        physical_sector_id: u32,
    ) -> RusticolResult<bool> {
        selector_domain_contains_parts(&self.parts, selector_domain_id, physical_sector_id)
    }

    /// Count the persisted currents and executor rows active for one sector.
    ///
    /// This is derived only from authenticated plan tables and therefore forms
    /// a compact structural certificate for selected-flow pruning.
    pub fn selector_work_summary(
        &self,
        physical_sector_id: u32,
    ) -> RusticolResult<DirectSelectorWorkSummary> {
        if physical_sector_id >= self.parts.physical_sector_count {
            return Err(invalid("physical selector sector is out of bounds"));
        }
        let mut summary = DirectSelectorWorkSummary {
            physical_sector_id,
            ..DirectSelectorWorkSummary::default()
        };
        for current in &self.parts.currents {
            if self.selector_domain_contains(current.selector_domain_id, physical_sector_id)? {
                summary.current_count = summary
                    .current_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("selector current count overflows u64"))?;
                summary.semantic_component_count = summary
                    .semantic_component_count
                    .checked_add(u64::from(current.component_count))
                    .ok_or_else(|| invalid("selector semantic component count overflows u64"))?;
            }
        }
        for row in &self.parts.sources {
            if self.selector_domain_contains(row.selector_domain_id, physical_sector_id)? {
                summary.source_row_count += 1;
            }
        }
        for row in &self.parts.contributions {
            if self.selector_domain_contains(row.selector_domain_id, physical_sector_id)? {
                summary.contribution_count += 1;
            }
        }
        for row in &self.parts.finalizations {
            if self.selector_domain_contains(row.selector_domain_id, physical_sector_id)? {
                summary.finalization_count += 1;
            }
        }
        for row in &self.parts.closures {
            if self.selector_domain_contains(row.selector_domain_id, physical_sector_id)? {
                summary.closure_count += 1;
            }
        }
        for destination in &self.parts.amplitude_destinations {
            if self.selector_domain_contains(destination.selector_domain_id, physical_sector_id)? {
                summary.amplitude_destination_count += 1;
            }
        }
        Ok(summary)
    }

    pub fn replay_targets(&self) -> &[DirectReplayTargetDescriptor] {
        &self.parts.replay_targets
    }

    pub fn source_permutations(&self) -> &[u32] {
        &self.parts.source_permutations
    }

    pub fn replay_momentum_signs(&self) -> &[i32] {
        &self.parts.replay_momentum_signs
    }

    pub fn replay_helicity_map(&self) -> &[u32] {
        &self.parts.replay_helicity_map
    }

    pub fn amplitude_destinations(&self) -> &[DirectAmplitudeDestinationDescriptor] {
        &self.parts.amplitude_destinations
    }

    pub fn resolved_helicities(&self) -> &[DirectResolvedHelicityDescriptor] {
        &self.parts.resolved_helicities
    }

    pub fn source_state_assignments(&self) -> &[DirectSourceStateAssignment] {
        &self.parts.source_state_assignments
    }

    pub fn source_dispatch_variants(&self) -> &[DirectSourceDispatchVariantDescriptor] {
        &self.parts.source_dispatch_variants
    }

    pub fn source_embeddings(&self) -> &[DirectSourceEmbeddingRow] {
        &self.parts.source_embeddings
    }

    pub fn source_projections(&self) -> &[DirectSourceProjectionRow] {
        &self.parts.source_projections
    }

    pub fn resolved_source_selections(&self) -> &[DirectResolvedSourceSelection] {
        &self.parts.resolved_source_selections
    }

    pub fn public_helicities(&self) -> &[i32] {
        &self.parts.public_helicities
    }

    pub fn exact_factors(&self) -> &[ExactComplexRational] {
        &self.parts.exact_factors
    }

    pub const fn closure_proofs(&self) -> &ClosureProofMetadataV2 {
        &self.parts.closure_proofs
    }

    pub fn into_parts(self) -> DirectRecurrencePlanParts {
        assert!(
            self.closure_proof_rows_retained,
            "runtime-compacted DirectRecurrencePlan cannot be converted back into complete parts"
        );
        self.parts
    }
}

fn canonicalize_contribution_initialization(
    parts: &mut DirectRecurrencePlanParts,
) -> RusticolResult<()> {
    let current_by_stage_and_base = parts
        .currents
        .iter()
        .filter(|current| current.node_kind == DirectNodeKind::Current)
        .map(|current| {
            (
                (current.stage, current.component_base),
                current.semantic_current_id,
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut first_row_by_current = std::collections::BTreeMap::<u32, usize>::new();
    let mut initialization_count_by_current = std::collections::BTreeMap::<u32, u32>::new();

    for row_group in parts
        .row_groups
        .iter()
        .filter(|row_group| row_group.role == DirectExecutorRole::Contribution)
    {
        let start = usize::try_from(row_group.row_start)
            .map_err(|_| invalid("contribution row-group start exceeds usize"))?;
        let Some(end) = start.checked_add(row_group.row_count as usize) else {
            continue;
        };
        let Some(rows) = parts.contributions.get(start..end) else {
            continue;
        };
        for (offset, row) in rows.iter().enumerate() {
            let Some(current_id) = current_by_stage_and_base
                .get(&(row_group.stage, row.destination_component_base))
                .copied()
            else {
                continue;
            };
            first_row_by_current
                .entry(current_id)
                .or_insert(start + offset);
            if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
                *initialization_count_by_current
                    .entry(current_id)
                    .or_default() += 1;
            }
        }
    }

    for (current_id, row_index) in first_row_by_current {
        if initialization_count_by_current
            .get(&current_id)
            .copied()
            .unwrap_or(0)
            == 0
        {
            parts.contributions[row_index].flags |= DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
        }
    }
    Ok(())
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
#[cfg_attr(target_vendor = "apple", cold)]
fn validate_parts(parts: &DirectRecurrencePlanParts) -> RusticolResult<()> {
    if parts.point_tile_size == 0 {
        return Err(invalid("point tile size must be positive"));
    }
    if parts.workspace_mib == 0 {
        return Err(invalid("workspace MiB must be positive"));
    }
    if parts.current_arena_components == 0 {
        return Err(invalid("current arena must contain at least one component"));
    }
    if parts.physical_sector_count == 0 {
        return Err(invalid("plan must retain at least one physical LC sector"));
    }
    if parts.retained_helicity_count == 0 {
        return Err(invalid("plan must retain at least one physical helicity"));
    }
    if parts.amplitude_destination_count == 0 {
        return Err(invalid(
            "plan must contain at least one amplitude destination",
        ));
    }
    if usize::try_from(parts.amplitude_destination_count).ok()
        != Some(parts.amplitude_destinations.len())
    {
        return Err(invalid(
            "amplitude destination count does not match its descriptor catalog",
        ));
    }
    if parts.external_source_count == 0 {
        return Err(invalid("plan must contain at least one external source"));
    }
    for (label, count) in [
        ("state templates", parts.state_template_count),
        ("source templates", parts.source_template_count),
        (
            "source templates or dispatch domains",
            parts.source_template_or_dispatch_count,
        ),
        ("direct executors", parts.direct_executor_count),
    ] {
        if count == 0 {
            return Err(invalid(format!("{label} catalog must not be empty")));
        }
    }
    if parts.strategy == RecurrenceStrategy::AllFlowUnion
        && (parts.runtime_helicity_contract_count == 0 || parts.runtime_helicity_variant_count == 0)
    {
        return Err(invalid(
            "all-flow-union requires runtime-helicity contracts and variants",
        ));
    }
    if parts.momentum_forms.is_empty() {
        return Err(invalid("plan must contain at least one momentum form"));
    }
    if parts.selector_domains.is_empty() {
        return Err(invalid("plan must contain at least one selector domain"));
    }
    if parts.exact_factors.is_empty() {
        return Err(invalid("plan must contain at least one exact factor"));
    }
    if parts.replay_momentum_signs.len() != parts.source_permutations.len()
        || parts
            .replay_momentum_signs
            .iter()
            .any(|sign| !matches!(sign, -1 | 1))
    {
        return Err(invalid(
            "replay momentum signs must align one-to-one with source permutations",
        ));
    }

    validate_ranges(
        "momentum form",
        parts
            .momentum_forms
            .iter()
            .map(|row| (row.term_start, row.term_count)),
        parts.momentum_terms.len(),
        true,
    )?;
    validate_ranges(
        "selector domain",
        parts
            .selector_domains
            .iter()
            .map(|row| (row.word_start, row.word_count)),
        parts.selector_words.len(),
        true,
    )?;
    validate_ranges(
        "replay source permutation",
        parts
            .replay_targets
            .iter()
            .map(|row| (row.source_permutation_start, row.source_permutation_count)),
        parts.source_permutations.len(),
        true,
    )?;
    validate_ranges(
        "replay helicity map",
        parts
            .replay_targets
            .iter()
            .map(|row| (row.helicity_map_start, row.helicity_map_count)),
        parts.replay_helicity_map.len(),
        true,
    )?;
    validate_ranges(
        "amplitude destination closure",
        parts
            .amplitude_destinations
            .iter()
            .map(|row| (row.closure_row_start, row.closure_row_count)),
        parts.closures.len(),
        true,
    )?;
    validate_ranges(
        "resolved-helicity source state",
        parts
            .resolved_helicities
            .iter()
            .map(|row| (row.source_state_start, row.source_state_count)),
        parts.source_state_assignments.len(),
        true,
    )?;
    validate_ranges(
        "resolved-helicity source selection",
        parts
            .resolved_helicities
            .iter()
            .map(|row| (row.source_selection_start, row.source_selection_count)),
        parts.resolved_source_selections.len(),
        true,
    )?;
    validate_ranges(
        "resolved-helicity public value",
        parts
            .resolved_helicities
            .iter()
            .map(|row| (row.public_helicity_start, row.public_helicity_count)),
        parts.public_helicities.len(),
        true,
    )?;
    validate_ranges(
        "source dispatch embedding",
        parts
            .source_dispatch_variants
            .iter()
            .map(|row| (row.embedding_start, row.embedding_count)),
        parts.source_embeddings.len(),
        true,
    )?;
    validate_ranges(
        "source dispatch projection",
        parts
            .source_dispatch_variants
            .iter()
            .map(|row| (row.projection_start, row.projection_count)),
        parts.source_projections.len(),
        true,
    )?;

    let momentum_count = u32_len("momentum forms", parts.momentum_forms.len())?;
    let selector_count = u32_len("selector domains", parts.selector_domains.len())?;
    let exact_factor_count = u32_len("exact factors", parts.exact_factors.len())?;
    let finalization_count = u32_len("finalizations", parts.finalizations.len())?;
    let resolved_helicity_count = u32_len("resolved helicities", parts.resolved_helicities.len())?;
    let source_dispatch_variant_count = u32_len(
        "source dispatch variants",
        parts.source_dispatch_variants.len(),
    )?;

    let mut current_slots = BTreeSet::new();
    let mut current_by_source_row = vec![None; parts.sources.len()];
    let mut current_by_finalization_row = vec![None; parts.finalizations.len()];
    for (index, current) in parts.currents.iter().enumerate() {
        if current.semantic_current_id != index as u32 {
            return Err(invalid(format!(
                "current {index} has non-canonical semantic ID {}",
                current.semantic_current_id
            )));
        }
        if current.component_count == 0 {
            return Err(invalid(format!("current {index} has zero components")));
        }
        require_ref(
            "current state template",
            current.state_template_id,
            parts.state_template_count,
        )?;
        let end = current
            .component_base
            .checked_add(u32::from(current.component_count))
            .ok_or_else(|| invalid(format!("current {index} component range overflows u32")))?;
        if end > parts.current_arena_components {
            return Err(invalid(format!(
                "current {index} component range exceeds the arena"
            )));
        }
        require_ref(
            "current momentum form",
            current.momentum_form_id,
            momentum_count,
        )?;
        require_ref(
            "current selector domain",
            current.selector_domain_id,
            selector_count,
        )?;
        if current.first_use > current.last_use {
            return Err(invalid(format!(
                "current {index} has an inverted liveness interval"
            )));
        }
        if !current_slots.insert((current.stage, current.component_base)) {
            return Err(invalid(format!(
                "stage {} has duplicate current component base {}",
                current.stage, current.component_base
            )));
        }
        match current.node_kind {
            DirectNodeKind::Source => {
                if current.source_row_or_sentinel == DIRECT_NONE_U32 {
                    return Err(invalid(format!("source current {index} has no source row")));
                }
                if current.finalization_row_or_sentinel != DIRECT_NONE_U32 {
                    return Err(invalid(format!(
                        "source current {index} must not have a finalization row"
                    )));
                }
                let slot = current_by_source_row
                    .get_mut(current.source_row_or_sentinel as usize)
                    .ok_or_else(|| {
                        invalid(format!("source current {index} row is out of bounds"))
                    })?;
                if slot.replace(index).is_some() {
                    return Err(invalid(format!(
                        "source row {} is referenced more than once",
                        current.source_row_or_sentinel
                    )));
                }
            }
            DirectNodeKind::Current => {
                if current.source_row_or_sentinel != DIRECT_NONE_U32 {
                    return Err(invalid(format!(
                        "propagated current {index} must not have a source row"
                    )));
                }
                if current.finalization_row_or_sentinel != DIRECT_NONE_U32 {
                    require_ref(
                        "current finalization",
                        current.finalization_row_or_sentinel,
                        finalization_count,
                    )?;
                    let slot = current_by_finalization_row
                        .get_mut(current.finalization_row_or_sentinel as usize)
                        .expect("finalization reference was bounds checked");
                    if slot.replace(index).is_some() {
                        return Err(invalid(format!(
                            "finalization row {} is referenced more than once",
                            current.finalization_row_or_sentinel
                        )));
                    }
                }
            }
        }
    }
    if current_by_source_row.iter().any(Option::is_none) {
        return Err(invalid("not every source row is referenced exactly once"));
    }
    if current_by_finalization_row.iter().any(Option::is_none) {
        return Err(invalid(
            "not every finalization row is referenced exactly once",
        ));
    }
    validate_arena_liveness(parts)?;

    for (index, source) in parts.sources.iter().enumerate() {
        require_ref(
            "source external slot",
            source.source_slot,
            parts.external_source_count,
        )?;
        require_ref(
            "source momentum form",
            source.momentum_form_id,
            momentum_count,
        )?;
        require_ref(
            "source selector domain",
            source.selector_domain_id,
            selector_count,
        )?;
        require_ref(
            "source exact factor",
            source.exact_factor_id,
            exact_factor_count,
        )?;
        require_ref(
            "source template or dispatch domain",
            source.source_template_or_dispatch_domain,
            parts.source_template_or_dispatch_count,
        )?;
        let current = &parts.currents[current_by_source_row[index].expect("source row is bound")];
        if source.destination_component_base != current.component_base
            || source.momentum_form_id != current.momentum_form_id
            || source.selector_domain_id != current.selector_domain_id
        {
            return Err(invalid(format!(
                "source row {index} does not match its current descriptor"
            )));
        }
    }
    match parts.strategy {
        RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => {
            if !parts.source_dispatch_variants.is_empty()
                || !parts.source_embeddings.is_empty()
                || !parts.source_projections.is_empty()
                || !parts.resolved_source_selections.is_empty()
            {
                return Err(invalid(
                    "fixed-source recurrence must not carry runtime source-dispatch tables",
                ));
            }
        }
        RecurrenceStrategy::AllFlowUnion => {
            if parts.source_dispatch_variants.is_empty()
                || parts.resolved_source_selections.is_empty()
            {
                return Err(invalid(
                    "all-flow-union requires runtime source-dispatch tables",
                ));
            }
        }
    }
    let mut source_variant_keys = BTreeSet::new();
    let mut referenced_source_rows = BTreeSet::new();
    for (index, variant) in parts.source_dispatch_variants.iter().enumerate() {
        require_ref(
            "source-dispatch source row",
            variant.source_row_id,
            u32_len("sources", parts.sources.len())?,
        )?;
        require_ref(
            "source-dispatch domain",
            variant.dispatch_domain_id,
            parts.runtime_helicity_contract_count,
        )?;
        require_ref(
            "source-dispatch runtime variant",
            variant.runtime_variant_id,
            parts.runtime_helicity_variant_count,
        )?;
        require_ref(
            "source-dispatch concrete template",
            variant.source_template_id,
            parts.source_template_count,
        )?;
        require_ref(
            "source-dispatch source state",
            variant.source_state_template_id,
            parts.state_template_count,
        )?;
        require_ref(
            "source-dispatch crossed state",
            variant.crossed_state_template_id,
            parts.state_template_count,
        )?;
        require_ref(
            "source-dispatch executor",
            variant.direct_executor_id,
            parts.direct_executor_count,
        )?;
        require_ref(
            "source-dispatch crossing factor",
            variant.crossing_exact_factor_id,
            exact_factor_count,
        )?;
        if parts.exact_factors[variant.crossing_exact_factor_id as usize].is_zero() {
            return Err(invalid(format!(
                "source-dispatch variant {index} has a zero crossing factor"
            )));
        }
        let source = &parts.sources[variant.source_row_id as usize];
        if source.source_template_or_dispatch_domain != variant.dispatch_domain_id {
            return Err(invalid(format!(
                "source-dispatch variant {index} does not match source row {} domain",
                variant.source_row_id
            )));
        }
        let current = &parts.currents[current_by_source_row[variant.source_row_id as usize]
            .expect("source-dispatch row is bound")];
        if variant.embedding_count != u32::from(current.component_count) {
            return Err(invalid(format!(
                "source-dispatch variant {index} embedding does not cover its full current"
            )));
        }
        let embedding_start = usize::try_from(variant.embedding_start)
            .map_err(|_| invalid("source embedding start exceeds usize"))?;
        let embedding_end = embedding_start
            .checked_add(variant.embedding_count as usize)
            .ok_or_else(|| invalid("source embedding range overflows usize"))?;
        let projection_start = usize::try_from(variant.projection_start)
            .map_err(|_| invalid("source projection start exceeds usize"))?;
        let projection_end = projection_start
            .checked_add(variant.projection_count as usize)
            .ok_or_else(|| invalid("source projection range overflows usize"))?;
        let embeddings = &parts.source_embeddings[embedding_start..embedding_end];
        let projections = &parts.source_projections[projection_start..projection_end];
        for (full_component, embedding) in embeddings.iter().enumerate() {
            if embedding.full_component != full_component as u32 {
                return Err(invalid(format!(
                    "source-dispatch variant {index} embedding is not in full-component order"
                )));
            }
            require_ref(
                "source embedding factor",
                embedding.exact_factor_id,
                exact_factor_count,
            )?;
            let factor = parts.exact_factors[embedding.exact_factor_id as usize];
            if (embedding.source_component_or_sentinel == DIRECT_NONE_U32) != factor.is_zero() {
                return Err(invalid(format!(
                    "source-dispatch variant {index} has inconsistent zero embedding"
                )));
            }
            if embedding.source_component_or_sentinel != DIRECT_NONE_U32
                && embedding.source_component_or_sentinel >= variant.projection_count
            {
                return Err(invalid(format!(
                    "source-dispatch variant {index} embedding source component is out of bounds"
                )));
            }
        }
        for (source_component, projection) in projections.iter().enumerate() {
            if projection.source_component != source_component as u32
                || projection.full_component >= variant.embedding_count
                || embeddings[projection.full_component as usize].source_component_or_sentinel
                    != projection.source_component
            {
                return Err(invalid(format!(
                    "source-dispatch variant {index} projection does not invert its embedding"
                )));
            }
        }
        if !source_variant_keys.insert((
            source.source_slot,
            variant.source_state_index,
            variant.source_row_id,
        )) {
            return Err(invalid(format!(
                "source-dispatch variant {index} repeats a source-state mapping"
            )));
        }
        referenced_source_rows.insert(variant.source_row_id);
    }
    if parts.strategy == RecurrenceStrategy::AllFlowUnion
        && referenced_source_rows.len() != parts.sources.len()
    {
        return Err(invalid(
            "not every all-flow-union source row has dispatch variants",
        ));
    }
    for (index, term) in parts.momentum_terms.iter().enumerate() {
        if term.source_slot >= parts.external_source_count {
            return Err(invalid(format!(
                "momentum term {index} source slot is out of bounds"
            )));
        }
        if term.coefficient == 0 {
            return Err(invalid(format!(
                "momentum term {index} has a zero coefficient"
            )));
        }
    }

    let arena_bases = parts
        .currents
        .iter()
        .map(|current| current.component_base)
        .collect::<BTreeSet<_>>();
    for (index, row) in parts.contributions.iter().enumerate() {
        if row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0 {
            require_arena_base(
                &arena_bases,
                row.parent0_component_base,
                "certified-reuse source",
                index,
            )?;
            if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32
                || row.parent1_momentum_form_id_or_sentinel != DIRECT_NONE_U32
            {
                return Err(invalid(format!(
                    "certified-reuse contribution {index} must not carry a second parent"
                )));
            }
            if row.parent0_momentum_form_id == 0 {
                return Err(invalid(format!(
                    "certified-reuse contribution {index} has zero components"
                )));
            }
            require_arena_base(
                &arena_bases,
                row.destination_component_base,
                "certified-reuse destination",
                index,
            )?;
            require_ref(
                "certified-reuse exact factor",
                row.exact_factor_id,
                exact_factor_count,
            )?;
            require_ref(
                "certified-reuse selector domain",
                row.selector_domain_id,
                selector_count,
            )?;
            let component_count = u16::try_from(row.parent0_momentum_form_id).map_err(|_| {
                invalid(format!(
                    "certified-reuse contribution {index} component count exceeds u16"
                ))
            })?;
            if !parts.currents.iter().any(|current| {
                current.component_base == row.parent0_component_base
                    && current.component_count == component_count
            }) || !parts.currents.iter().any(|current| {
                current.component_base == row.destination_component_base
                    && current.component_count == component_count
            }) {
                return Err(invalid(format!(
                    "certified-reuse contribution {index} component count disagrees with its source or destination current"
                )));
            }
            continue;
        }
        require_arena_base(
            &arena_bases,
            row.parent0_component_base,
            "contribution parent 0",
            index,
        )?;
        require_optional_pair(
            row.parent1_component_base_or_sentinel,
            row.parent1_momentum_form_id_or_sentinel,
            "contribution parent 1",
            index,
        )?;
        if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 {
            require_arena_base(
                &arena_bases,
                row.parent1_component_base_or_sentinel,
                "contribution parent 1",
                index,
            )?;
            require_ref(
                "contribution parent 1 momentum form",
                row.parent1_momentum_form_id_or_sentinel,
                momentum_count,
            )?;
        }
        require_ref(
            "contribution parent 0 momentum form",
            row.parent0_momentum_form_id,
            momentum_count,
        )?;
        require_arena_base(
            &arena_bases,
            row.destination_component_base,
            "contribution destination",
            index,
        )?;
        require_ref(
            "contribution exact factor",
            row.exact_factor_id,
            exact_factor_count,
        )?;
        require_ref(
            "contribution selector domain",
            row.selector_domain_id,
            selector_count,
        )?;
    }

    for (index, row) in parts.finalizations.iter().enumerate() {
        require_arena_base(
            &arena_bases,
            row.component_base,
            "finalization component",
            index,
        )?;
        if row.component_count == 0 {
            return Err(invalid(format!("finalization {index} has zero components")));
        }
        require_ref(
            "finalization momentum form",
            row.momentum_form_id,
            momentum_count,
        )?;
        require_ref(
            "finalization exact factor",
            row.exact_factor_id,
            exact_factor_count,
        )?;
        require_ref(
            "finalization selector domain",
            row.selector_domain_id,
            selector_count,
        )?;
        let current =
            &parts.currents[current_by_finalization_row[index].expect("finalization row is bound")];
        if row.component_base != current.component_base
            || row.component_count != current.component_count
            || row.momentum_form_id != current.momentum_form_id
            || row.selector_domain_id != current.selector_domain_id
        {
            return Err(invalid(format!(
                "finalization {index} does not match its current descriptor"
            )));
        }
    }

    for (index, row) in parts.closures.iter().enumerate() {
        require_arena_base(
            &arena_bases,
            row.parent0_component_base,
            "closure parent 0",
            index,
        )?;
        require_optional_pair(
            row.parent1_component_base_or_sentinel,
            row.parent1_momentum_form_id_or_sentinel,
            "closure parent 1",
            index,
        )?;
        if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 {
            require_arena_base(
                &arena_bases,
                row.parent1_component_base_or_sentinel,
                "closure parent 1",
                index,
            )?;
            require_ref(
                "closure parent 1 momentum form",
                row.parent1_momentum_form_id_or_sentinel,
                momentum_count,
            )?;
        }
        require_ref(
            "closure parent 0 momentum form",
            row.parent0_momentum_form_id,
            momentum_count,
        )?;
        require_ref(
            "closure exact factor",
            row.exact_factor_id,
            exact_factor_count,
        )?;
        if row.component_count == 0 {
            return Err(invalid(format!(
                "closure {index} has zero contraction components"
            )));
        }
        let component_factor_end = row
            .component_factor_start
            .checked_add(u32::from(row.component_count))
            .ok_or_else(|| {
                invalid(format!(
                    "closure {index} component-factor range overflows u32"
                ))
            })?;
        if component_factor_end > exact_factor_count {
            return Err(invalid(format!(
                "closure {index} component-factor range is out of bounds"
            )));
        }
        require_ref(
            "closure selector domain",
            row.selector_domain_id,
            selector_count,
        )?;
        if row.amplitude_destination_id >= parts.amplitude_destination_count {
            return Err(invalid(format!(
                "closure {index} amplitude destination is out of bounds"
            )));
        }
    }

    for (index, target) in parts.replay_targets.iter().enumerate() {
        require_ref(
            "replay public flow",
            target.public_flow_id,
            parts.physical_sector_count,
        )?;
        require_ref(
            "replay representative flow",
            target.representative_id,
            parts.physical_sector_count,
        )?;
        require_ref(
            "replay phase exact factor",
            target.phase_exact_factor_id,
            exact_factor_count,
        )?;
        require_ref(
            "replay selector domain",
            target.selector_domain_id,
            selector_count,
        )?;
        if !selector_domain_contains_parts(
            parts,
            target.selector_domain_id,
            target.representative_id,
        )? {
            return Err(invalid(format!(
                "replay target {index} selector domain excludes its representative sector"
            )));
        }
        if target.multiplicity == 0 {
            return Err(invalid(format!(
                "replay target {index} has zero multiplicity"
            )));
        }
        if target.source_permutation_count != parts.external_source_count {
            return Err(invalid(format!(
                "replay target {index} does not map every external source"
            )));
        }
        let start = usize::try_from(target.source_permutation_start)
            .map_err(|_| invalid("replay permutation start exceeds usize"))?;
        let count = usize::try_from(target.source_permutation_count)
            .map_err(|_| invalid("replay permutation count exceeds usize"))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| invalid("replay permutation range overflows usize"))?;
        let values = &parts.source_permutations[start..end];
        let unique = values.iter().copied().collect::<BTreeSet<_>>();
        let external_source_count = usize::try_from(parts.external_source_count)
            .map_err(|_| invalid("external source count exceeds usize"))?;
        if unique.len() != external_source_count
            || unique.iter().copied().ne(0..parts.external_source_count)
        {
            return Err(invalid(format!(
                "replay target {index} source mapping is not a permutation"
            )));
        }
        match parts.strategy {
            RecurrenceStrategy::TopologyReplay => {
                if target.helicity_map_count != resolved_helicity_count {
                    return Err(invalid(format!(
                        "replay target {index} helicity map covers {} of {resolved_helicity_count} resolved helicities",
                        target.helicity_map_count
                    )));
                }
                let start = usize::try_from(target.helicity_map_start)
                    .map_err(|_| invalid("replay helicity-map start exceeds usize"))?;
                let count = usize::try_from(target.helicity_map_count)
                    .map_err(|_| invalid("replay helicity-map count exceeds usize"))?;
                let end = start
                    .checked_add(count)
                    .ok_or_else(|| invalid("replay helicity-map range overflows usize"))?;
                let values = &parts.replay_helicity_map[start..end];
                let unique = values.iter().copied().collect::<BTreeSet<_>>();
                if unique.len() != count || unique.iter().copied().ne(0..resolved_helicity_count) {
                    return Err(invalid(format!(
                        "replay target {index} helicity mapping is not a bijection"
                    )));
                }
            }
            RecurrenceStrategy::AllFlowUnion => {
                if target.helicity_map_count != 0 {
                    return Err(invalid(format!(
                        "all-flow-union replay target {index} carries a helicity map"
                    )));
                }
            }
            RecurrenceStrategy::ContractedColorUnion => {
                if target.helicity_map_count != resolved_helicity_count {
                    return Err(invalid(format!(
                        "contracted replay target {index} helicity map covers {} of {resolved_helicity_count} resolved helicities",
                        target.helicity_map_count
                    )));
                }
                let start = usize::try_from(target.helicity_map_start)
                    .map_err(|_| invalid("replay helicity-map start exceeds usize"))?;
                let count = usize::try_from(target.helicity_map_count)
                    .map_err(|_| invalid("replay helicity-map count exceeds usize"))?;
                let end = start
                    .checked_add(count)
                    .ok_or_else(|| invalid("replay helicity-map range overflows usize"))?;
                let values = &parts.replay_helicity_map[start..end];
                let unique = values.iter().copied().collect::<BTreeSet<_>>();
                if unique.len() != count || unique.iter().copied().ne(0..resolved_helicity_count) {
                    return Err(invalid(format!(
                        "contracted replay target {index} helicity mapping is not a bijection"
                    )));
                }
            }
        }
    }

    for (index, destination) in parts.amplitude_destinations.iter().enumerate() {
        if destination.id != index as u32 {
            return Err(invalid(format!(
                "amplitude destination {index} has non-canonical ID {}",
                destination.id
            )));
        }
        require_ref(
            "amplitude destination physical sector",
            destination.target_sector_id,
            parts.physical_sector_count,
        )?;
        if destination.target_helicity_id_or_sentinel != DIRECT_NONE_U32 {
            require_ref(
                "amplitude destination resolved helicity",
                destination.target_helicity_id_or_sentinel,
                resolved_helicity_count,
            )?;
        }
        require_ref(
            "amplitude destination selector domain",
            destination.selector_domain_id,
            selector_count,
        )?;
        if !selector_domain_contains_parts(
            parts,
            destination.selector_domain_id,
            destination.target_sector_id,
        )? {
            return Err(invalid(format!(
                "amplitude destination {index} selector domain excludes its target sector"
            )));
        }
        let start = usize::try_from(destination.closure_row_start)
            .map_err(|_| invalid("amplitude destination closure start exceeds usize"))?;
        let count = usize::try_from(destination.closure_row_count)
            .map_err(|_| invalid("amplitude destination closure count exceeds usize"))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| invalid("amplitude destination closure range overflows usize"))?;
        if parts.closures[start..end]
            .iter()
            .any(|row| row.amplitude_destination_id != destination.id)
        {
            return Err(invalid(format!(
                "amplitude destination {index} does not own every closure in its range"
            )));
        }
        if parts.closures[start..end]
            .iter()
            .any(|row| row.selector_domain_id != destination.selector_domain_id)
        {
            return Err(invalid(format!(
                "amplitude destination {index} disagrees with a closure selector domain"
            )));
        }
    }

    for (index, helicity) in parts.resolved_helicities.iter().enumerate() {
        if helicity.id != index as u32 {
            return Err(invalid(format!(
                "resolved helicity {index} has non-canonical ID {}",
                helicity.id
            )));
        }
        if helicity.source_state_count != parts.external_source_count
            || helicity.public_helicity_count != parts.external_source_count
        {
            return Err(invalid(format!(
                "resolved helicity {index} does not cover every external source"
            )));
        }
        match parts.strategy {
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion
                if helicity.source_selection_count != 0 =>
            {
                return Err(invalid(format!(
                    "fixed-source resolved helicity {index} carries source dispatch selections"
                )));
            }
            RecurrenceStrategy::AllFlowUnion
                if helicity.source_selection_count != parts.external_source_count =>
            {
                return Err(invalid(format!(
                    "all-flow-union resolved helicity {index} does not select every external source"
                )));
            }
            _ => {}
        }
        require_ref(
            "resolved-helicity selector domain",
            helicity.selector_domain_id,
            selector_count,
        )?;
        let start = usize::try_from(helicity.source_state_start)
            .map_err(|_| invalid("resolved-helicity source-state start exceeds usize"))?;
        let count = usize::try_from(helicity.source_state_count)
            .map_err(|_| invalid("resolved-helicity source-state count exceeds usize"))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| invalid("resolved-helicity source-state range overflows usize"))?;
        for (source_slot, assignment) in parts.source_state_assignments[start..end]
            .iter()
            .enumerate()
        {
            if assignment.source_slot != source_slot as u32 {
                return Err(invalid(format!(
                    "resolved helicity {index} source states are not in source-slot order"
                )));
            }
        }
        let selection_start = usize::try_from(helicity.source_selection_start)
            .map_err(|_| invalid("resolved-helicity source-selection start exceeds usize"))?;
        let selection_count = usize::try_from(helicity.source_selection_count)
            .map_err(|_| invalid("resolved-helicity source-selection count exceeds usize"))?;
        let selection_end = selection_start
            .checked_add(selection_count)
            .ok_or_else(|| invalid("resolved-helicity source-selection range overflows usize"))?;
        for (source_slot, selection) in parts.resolved_source_selections
            [selection_start..selection_end]
            .iter()
            .enumerate()
        {
            if selection.source_slot != source_slot as u32 {
                return Err(invalid(format!(
                    "resolved helicity {index} source selections are not in source-slot order"
                )));
            }
            require_ref(
                "resolved-helicity source dispatch variant",
                selection.dispatch_variant_id,
                source_dispatch_variant_count,
            )?;
            let variant = parts.source_dispatch_variants[selection.dispatch_variant_id as usize];
            let source = parts.sources[variant.source_row_id as usize];
            let assignment = parts.source_state_assignments[start + source_slot];
            if source.source_slot != selection.source_slot
                || variant.source_state_index != assignment.state_index
            {
                return Err(invalid(format!(
                    "resolved helicity {index} source selection does not match its public source-state assignment"
                )));
            }
        }
    }
    if parts.strategy == RecurrenceStrategy::AllFlowUnion
        && parts.resolved_helicities.len() as u64 != parts.retained_helicity_count
    {
        return Err(invalid(
            "all-flow-union resolved-helicity catalog does not cover retained helicities",
        ));
    }

    validate_row_groups(parts)?;
    validate_contribution_initialization(parts)?;
    validate_closure_proofs(parts)?;
    Ok(())
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn validate_closure_proofs(parts: &DirectRecurrencePlanParts) -> RusticolResult<()> {
    parts
        .closure_proofs
        .validate_tables()
        .map_err(|error| invalid(error.message()))?;
    let current_bases = parts
        .currents
        .iter()
        .map(|current| (current.semantic_current_id, current.component_base))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut row_owner = vec![None; parts.closures.len()];
    let mut term_owner = vec![None; parts.closures.len()];

    for group in parts.closure_proofs.groups() {
        let Some(term_id) = group.emitted_runtime_closure_term_id() else {
            if group.emitted_direct_closure_row_id().is_some() {
                return Err(invalid(format!(
                    "exact-zero closure proof group {} references a direct closure row",
                    group.id()
                )));
            }
            continue;
        };
        let term_slot = term_owner.get_mut(term_id as usize).ok_or_else(|| {
            invalid(format!(
                "closure proof group {} references out-of-range runtime term {term_id}",
                group.id()
            ))
        })?;
        if term_slot.replace(group.id()).is_some() {
            return Err(invalid(format!(
                "runtime closure term {term_id} belongs to multiple proof groups"
            )));
        }
        let row_id = group.emitted_direct_closure_row_id().ok_or_else(|| {
            invalid(format!(
                "nonzero closure proof group {} has no direct closure row",
                group.id()
            ))
        })?;
        let row = parts.closures.get(row_id as usize).ok_or_else(|| {
            invalid(format!(
                "closure proof group {} references out-of-range direct row {row_id}",
                group.id()
            ))
        })?;
        if row_owner[row_id as usize].replace(group.id()).is_some() {
            return Err(invalid(format!(
                "direct closure row {row_id} belongs to multiple proof groups"
            )));
        }
        if row.proof_group_id() != group.id() {
            return Err(invalid(format!(
                "direct closure row {row_id} names proof group {}, expected {}",
                row.proof_group_id(),
                group.id()
            )));
        }
        let runtime_factor = parts.exact_factors[row.exact_factor_id as usize];
        if runtime_factor != group.exact_summed_factor() {
            return Err(invalid(format!(
                "direct closure row {row_id} factor does not match proof group {}",
                group.id()
            )));
        }

        let component_start = row.component_factor_start as usize;
        let component_end = component_start
            .checked_add(row.component_count as usize)
            .ok_or_else(|| invalid("closure component-factor range overflows usize"))?;
        let component_factors = parts
            .exact_factors
            .get(component_start..component_end)
            .ok_or_else(|| invalid("closure component-factor range is out of bounds"))?;
        if closure_component_factor_digest_v2(component_factors)
            .map_err(|error| invalid(error.message()))?
            != group.component_factor_digest()
        {
            return Err(invalid(format!(
                "direct closure row {row_id} component-factor digest does not match proof group {}",
                group.id()
            )));
        }

        let selector = parts
            .selector_domains
            .get(row.selector_domain_id as usize)
            .ok_or_else(|| invalid("closure selector domain is absent"))?;
        let selector_start = selector.word_start as usize;
        let selector_end = selector_start
            .checked_add(selector.word_count as usize)
            .ok_or_else(|| invalid("closure selector-domain range overflows usize"))?;
        let selector_words = parts
            .selector_words
            .get(selector_start..selector_end)
            .ok_or_else(|| invalid("closure selector-domain range is out of bounds"))?;
        if closure_selector_domain_digest_v2(selector_words)
            .map_err(|error| invalid(error.message()))?
            != group.selector_domain_digest()
        {
            return Err(invalid(format!(
                "direct closure row {row_id} selector-domain digest does not match proof group {}",
                group.id()
            )));
        }

        let contribution_range = group
            .contribution_range()
            .as_usize_range(
                parts.closure_proofs.contributions().len(),
                &format!("closure proof group {} contribution", group.id()),
            )
            .map_err(|error| invalid(error.message()))?;
        let contributions = &parts.closure_proofs.contributions()[contribution_range];
        let representative = &contributions[0];
        if Some(row.amplitude_destination_id) != representative.target_destination_id() {
            return Err(invalid(format!(
                "direct closure row {row_id} destination does not match proof group {}",
                group.id()
            )));
        }
        let parent_ids = representative
            .evaluator_parent_runtime_ids()
            .ok_or_else(|| {
                invalid(format!(
                    "nonzero closure proof group {} has no complete runtime-parent binding",
                    group.id()
                ))
            })?;
        let expected_parent0 = *current_bases.get(&parent_ids[0]).ok_or_else(|| {
            invalid(format!(
                "closure proof group {} references absent semantic parent {}",
                group.id(),
                parent_ids[0]
            ))
        })?;
        let expected_parent1 = parent_ids
            .get(1)
            .map(|parent_id| {
                current_bases.get(parent_id).copied().ok_or_else(|| {
                    invalid(format!(
                        "closure proof group {} references absent semantic parent {parent_id}",
                        group.id()
                    ))
                })
            })
            .transpose()?
            .unwrap_or(DIRECT_NONE_U32);
        if parent_ids.len() > 2
            || row.parent0_component_base != expected_parent0
            || row.parent1_component_base_or_sentinel != expected_parent1
        {
            return Err(invalid(format!(
                "direct closure row {row_id} parents do not match proof group {}",
                group.id()
            )));
        }
    }

    if let Some((row_id, _)) = row_owner
        .iter()
        .enumerate()
        .find(|(_, owner)| owner.is_none())
    {
        return Err(invalid(format!(
            "direct closure row {row_id} has no closure proof group"
        )));
    }
    if let Some((term_id, _)) = term_owner
        .iter()
        .enumerate()
        .find(|(_, owner)| owner.is_none())
    {
        return Err(invalid(format!(
            "runtime closure term {term_id} has no closure proof group"
        )));
    }
    Ok(())
}

fn validate_row_groups(parts: &DirectRecurrencePlanParts) -> RusticolResult<()> {
    let mut next = [0_u64; 4];
    let mut previous = None;
    for (index, row_group) in parts.row_groups.iter().enumerate() {
        if row_group.row_count == 0 {
            return Err(invalid(format!("row group {index} has zero rows")));
        }
        if (row_group.role == DirectExecutorRole::Source
            && parts.strategy == RecurrenceStrategy::AllFlowUnion)
            || (row_group.role == DirectExecutorRole::Contribution
                && row_group.direct_executor_id == DIRECT_NONE_U32)
        {
            if row_group.direct_executor_id != DIRECT_NONE_U32 {
                return Err(invalid(format!(
                    "internal row group {index} must use the missing-executor sentinel"
                )));
            }
            if row_group.role == DirectExecutorRole::Contribution {
                let start = usize::try_from(row_group.row_start)
                    .map_err(|_| invalid("certified-reuse row-group start exceeds usize"))?;
                let end = start
                    .checked_add(row_group.row_count as usize)
                    .ok_or_else(|| invalid("certified-reuse row-group range overflows usize"))?;
                let rows = parts.contributions.get(start..end).ok_or_else(|| {
                    invalid(format!(
                        "certified-reuse row-group {index} range is out of bounds"
                    ))
                })?;
                if !rows
                    .iter()
                    .all(|row| row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0)
                {
                    return Err(invalid(format!(
                        "internal contribution row group {index} contains an uncertified row"
                    )));
                }
            }
        } else {
            require_ref(
                "row-group direct executor",
                row_group.direct_executor_id,
                parts.direct_executor_count,
            )?;
        }
        let order = (row_group.stage, row_group.role);
        if previous.is_some_and(|previous| order < previous) {
            return Err(invalid(
                "row groups are not in deterministic stage/role order",
            ));
        }
        previous = Some(order);
        let role_index = row_group.role as usize;
        if row_group.row_start != next[role_index] {
            return Err(invalid(format!(
                "row group {index} does not continue its role's row partition"
            )));
        }
        next[role_index] = row_group
            .row_start
            .checked_add(u64::from(row_group.row_count))
            .ok_or_else(|| invalid(format!("row group {index} row range overflows u64")))?;
        let start = usize::try_from(row_group.row_start)
            .map_err(|_| invalid(format!("row group {index} row start exceeds usize")))?;
        let end = start
            .checked_add(row_group.row_count as usize)
            .ok_or_else(|| invalid(format!("row group {index} row range overflows usize")))?;
        let expected_operation = match row_group.role {
            DirectExecutorRole::Source => DirectDestinationOperation::Initialize,
            DirectExecutorRole::Contribution => DirectDestinationOperation::Add,
            DirectExecutorRole::Finalization => DirectDestinationOperation::FinalizeInPlace,
            DirectExecutorRole::Closure => DirectDestinationOperation::ClosureAdd,
        };
        if row_group.destination_operation != expected_operation {
            return Err(invalid(format!(
                "row group {index} has an operation incompatible with its role"
            )));
        }
        let role_row_count = match row_group.role {
            DirectExecutorRole::Source => parts.sources.len(),
            DirectExecutorRole::Contribution => parts.contributions.len(),
            DirectExecutorRole::Finalization => parts.finalizations.len(),
            DirectExecutorRole::Closure => parts.closures.len(),
        };
        if end > role_row_count {
            return Err(invalid(format!(
                "row group {index} range is out of bounds for its role"
            )));
        }
    }
    let expected = [
        parts.sources.len(),
        parts.contributions.len(),
        parts.finalizations.len(),
        parts.closures.len(),
    ];
    for (index, expected) in expected.into_iter().enumerate() {
        if next[index] != expected as u64 {
            return Err(invalid(format!(
                "row groups do not partition role {index}: covered {}, expected {expected}",
                next[index]
            )));
        }
    }
    Ok(())
}

fn validate_contribution_initialization(parts: &DirectRecurrencePlanParts) -> RusticolResult<()> {
    let current_by_stage_and_base = parts
        .currents
        .iter()
        .filter(|current| current.node_kind == DirectNodeKind::Current)
        .map(|current| {
            (
                (current.stage, current.component_base),
                current.semantic_current_id,
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut contribution_counts = vec![0_u32; parts.currents.len()];
    let mut initialization_counts = vec![0_u32; parts.currents.len()];

    for row_group in parts
        .row_groups
        .iter()
        .filter(|row_group| row_group.role == DirectExecutorRole::Contribution)
    {
        let start = usize::try_from(row_group.row_start)
            .map_err(|_| invalid("contribution row-group start exceeds usize"))?;
        let end = start
            .checked_add(row_group.row_count as usize)
            .ok_or_else(|| invalid("contribution row-group range overflows usize"))?;
        for (offset, row) in parts.contributions[start..end].iter().enumerate() {
            if row.flags & !DIRECT_CONTRIBUTION_FLAGS_KNOWN != 0 {
                return Err(invalid(format!(
                    "contribution {} has unknown flags {:#x}",
                    start + offset,
                    row.flags
                )));
            }
            let current_id = current_by_stage_and_base
                .get(&(row_group.stage, row.destination_component_base))
                .copied()
                .ok_or_else(|| {
                    invalid(format!(
                        "contribution {} does not target a current in stage {}",
                        start + offset,
                        row_group.stage
                    ))
                })? as usize;
            contribution_counts[current_id] = contribution_counts[current_id]
                .checked_add(1)
                .ok_or_else(|| invalid("contribution count overflows u32"))?;
            if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
                initialization_counts[current_id] = initialization_counts[current_id]
                    .checked_add(1)
                    .ok_or_else(|| invalid("contribution initialization count overflows u32"))?;
            }
        }
    }

    for current in parts
        .currents
        .iter()
        .filter(|current| current.node_kind == DirectNodeKind::Current)
    {
        let current_id = current.semantic_current_id as usize;
        if contribution_counts[current_id] == 0 {
            return Err(invalid(format!(
                "current {} has no contribution rows",
                current.semantic_current_id
            )));
        }
        if initialization_counts[current_id] != 1 {
            return Err(invalid(format!(
                "current {} must have exactly one destination-initializing contribution, found {}",
                current.semantic_current_id, initialization_counts[current_id]
            )));
        }
    }
    Ok(())
}

fn validate_arena_liveness(parts: &DirectRecurrencePlanParts) -> RusticolResult<()> {
    let component_count = usize::try_from(parts.current_arena_components)
        .map_err(|_| invalid("current arena component count exceeds usize"))?;
    let mut intervals = vec![Vec::<(u32, u32, u32)>::new(); component_count];
    for current in &parts.currents {
        let end = current.component_base + u32::from(current.component_count);
        for component in current.component_base..end {
            let component = usize::try_from(component)
                .map_err(|_| invalid("current arena component ID exceeds usize"))?;
            intervals[component].push((
                current.first_use,
                current.last_use,
                current.semantic_current_id,
            ));
        }
    }
    for (component, intervals) in intervals.iter_mut().enumerate() {
        intervals.sort_unstable();
        for pair in intervals.windows(2) {
            let (_, left_last, left_id) = pair[0];
            let (right_first, _, right_id) = pair[1];
            if left_last >= right_first {
                return Err(invalid(format!(
                    "arena component {component} is shared by live currents {left_id} and {right_id}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_ranges(
    label: &str,
    ranges: impl IntoIterator<Item = (u64, u32)>,
    item_count: usize,
    require_partition: bool,
) -> RusticolResult<()> {
    let item_count = u64::try_from(item_count)
        .map_err(|_| invalid(format!("{label} item count exceeds u64")))?;
    let mut next = 0_u64;
    for (index, (start, count)) in ranges.into_iter().enumerate() {
        let end = start
            .checked_add(u64::from(count))
            .ok_or_else(|| invalid(format!("{label} range {index} overflows u64")))?;
        if end > item_count {
            return Err(invalid(format!("{label} range {index} is out of bounds")));
        }
        if require_partition && start != next {
            return Err(invalid(format!(
                "{label} range {index} does not continue the canonical partition"
            )));
        }
        if require_partition {
            next = end;
        }
    }
    if require_partition && next != item_count {
        return Err(invalid(format!(
            "{label} ranges cover {next} items, expected {item_count}"
        )));
    }
    Ok(())
}

fn require_ref(label: &str, value: u32, count: u32) -> RusticolResult<()> {
    if value >= count {
        return Err(invalid(format!("{label} {value} is out of bounds {count}")));
    }
    Ok(())
}

fn require_arena_base(
    bases: &BTreeSet<u32>,
    value: u32,
    label: &str,
    index: usize,
) -> RusticolResult<()> {
    if !bases.contains(&value) {
        return Err(invalid(format!(
            "{label} in row {index} does not name a current arena base"
        )));
    }
    Ok(())
}

fn require_optional_pair(left: u32, right: u32, label: &str, index: usize) -> RusticolResult<()> {
    if (left == DIRECT_NONE_U32) != (right == DIRECT_NONE_U32) {
        return Err(invalid(format!(
            "{label} in row {index} has mismatched optional fields"
        )));
    }
    Ok(())
}

fn u32_len(label: &str, length: usize) -> RusticolResult<u32> {
    u32::try_from(length).map_err(|_| invalid(format!("{label} length exceeds u32")))
}

fn digest_parts(parts: &DirectRecurrencePlanParts) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-runtime-layout-v2\0");
    hash.update(parts.strategy.as_u32().to_le_bytes());
    hash.update(parts.semantic_digest.as_bytes());
    hash.update(parts.prepared_pack_digest.as_bytes());
    hash.update(parts.direct_template_catalog_digest.as_bytes());
    for value in [
        parts.point_tile_size,
        parts.workspace_mib,
        parts.current_arena_components,
        parts.physical_sector_count,
        parts.amplitude_destination_count,
        parts.parameter_value_count,
        parts.external_source_count,
        parts.state_template_count,
        parts.source_template_count,
        parts.source_template_or_dispatch_count,
        parts.runtime_helicity_contract_count,
        parts.runtime_helicity_variant_count,
        parts.direct_executor_count,
    ] {
        hash.update(value.to_le_bytes());
    }
    hash.update(parts.retained_helicity_count.to_le_bytes());
    macro_rules! count {
        ($rows:expr) => {
            hash.update(
                u64::try_from($rows.len())
                    .map_err(|_| invalid("digest row count exceeds u64"))?
                    .to_le_bytes(),
            );
        };
    }
    count!(parts.currents);
    for row in &parts.currents {
        hash.update(row.semantic_current_id.to_le_bytes());
        hash.update((row.node_kind as u16).to_le_bytes());
        hash.update(row.state_template_id.to_le_bytes());
        hash.update(row.component_base.to_le_bytes());
        hash.update(row.component_count.to_le_bytes());
        hash.update(row.momentum_form_id.to_le_bytes());
        hash.update(row.stage.to_le_bytes());
        hash.update(row.selector_domain_id.to_le_bytes());
        hash.update(row.first_use.to_le_bytes());
        hash.update(row.last_use.to_le_bytes());
        hash.update(row.source_row_or_sentinel.to_le_bytes());
        hash.update(row.finalization_row_or_sentinel.to_le_bytes());
    }
    macro_rules! hash_u32s {
        ($($value:expr),+ $(,)?) => {
            $(hash.update($value.to_le_bytes());)+
        };
    }
    count!(parts.sources);
    for row in &parts.sources {
        hash_u32s!(
            row.source_slot,
            row.destination_component_base,
            row.momentum_form_id,
            row.source_template_or_dispatch_domain,
        );
        hash.update(row.spin_state_class.to_le_bytes());
        hash.update(row.exact_factor_id.to_le_bytes());
        hash.update(row.selector_domain_id.to_le_bytes());
    }
    count!(parts.contributions);
    for row in &parts.contributions {
        hash_u32s!(
            row.parent0_component_base,
            row.parent1_component_base_or_sentinel,
            row.parent0_momentum_form_id,
            row.parent1_momentum_form_id_or_sentinel,
            row.destination_component_base,
            row.exact_factor_id,
            row.selector_domain_id,
            row.flags,
        );
    }
    count!(parts.finalizations);
    for row in &parts.finalizations {
        hash.update(row.component_base.to_le_bytes());
        hash.update(row.component_count.to_le_bytes());
        hash_u32s!(
            row.momentum_form_id,
            row.exact_factor_id,
            row.selector_domain_id,
            row.flags,
        );
    }
    count!(parts.closures);
    for row in &parts.closures {
        hash_u32s!(
            row.parent0_component_base,
            row.parent1_component_base_or_sentinel,
            row.parent0_momentum_form_id,
            row.parent1_momentum_form_id_or_sentinel,
            row.amplitude_destination_id,
            row.exact_factor_id,
            row.component_factor_start,
        );
        hash.update(row.component_count.to_le_bytes());
        hash_u32s!(row.selector_domain_id, row.flags,);
    }
    count!(parts.row_groups);
    for row in &parts.row_groups {
        hash.update(row.stage.to_le_bytes());
        hash.update((row.role as u16).to_le_bytes());
        hash.update((row.destination_operation as u16).to_le_bytes());
        hash.update(row.direct_executor_id.to_le_bytes());
        hash.update(row.row_start.to_le_bytes());
        hash.update(row.row_count.to_le_bytes());
    }
    count!(parts.momentum_forms);
    for row in &parts.momentum_forms {
        hash.update(row.term_start.to_le_bytes());
        hash.update(row.term_count.to_le_bytes());
    }
    count!(parts.momentum_terms);
    for row in &parts.momentum_terms {
        hash.update(row.source_slot.to_le_bytes());
        hash.update(row.coefficient.to_le_bytes());
    }
    count!(parts.selector_domains);
    for row in &parts.selector_domains {
        hash.update(row.word_start.to_le_bytes());
        hash.update(row.word_count.to_le_bytes());
    }
    count!(parts.selector_words);
    for value in &parts.selector_words {
        hash.update(value.to_le_bytes());
    }
    count!(parts.replay_targets);
    for row in &parts.replay_targets {
        hash_u32s!(row.public_flow_id, row.representative_id);
        hash.update(row.source_permutation_start.to_le_bytes());
        hash.update(row.helicity_map_start.to_le_bytes());
        hash_u32s!(
            row.source_permutation_count,
            row.helicity_map_count,
            row.phase_exact_factor_id,
            row.multiplicity,
            row.selector_domain_id,
        );
    }
    count!(parts.source_permutations);
    for value in &parts.source_permutations {
        hash.update(value.to_le_bytes());
    }
    count!(parts.replay_momentum_signs);
    for value in &parts.replay_momentum_signs {
        hash.update(value.to_le_bytes());
    }
    count!(parts.replay_helicity_map);
    for value in &parts.replay_helicity_map {
        hash.update(value.to_le_bytes());
    }
    count!(parts.amplitude_destinations);
    for row in &parts.amplitude_destinations {
        hash.update(row.closure_row_start.to_le_bytes());
        hash_u32s!(
            row.id,
            row.target_sector_id,
            row.target_helicity_id_or_sentinel,
            row.closure_row_count,
            row.selector_domain_id,
        );
    }
    count!(parts.resolved_helicities);
    for row in &parts.resolved_helicities {
        hash.update(row.source_state_start.to_le_bytes());
        hash.update(row.source_selection_start.to_le_bytes());
        hash.update(row.public_helicity_start.to_le_bytes());
        hash_u32s!(
            row.id,
            row.source_state_count,
            row.source_selection_count,
            row.public_helicity_count,
            row.selector_domain_id,
        );
    }
    count!(parts.source_state_assignments);
    for row in &parts.source_state_assignments {
        hash_u32s!(row.source_slot, row.state_index);
    }
    count!(parts.source_dispatch_variants);
    for row in &parts.source_dispatch_variants {
        hash.update(row.embedding_start.to_le_bytes());
        hash.update(row.projection_start.to_le_bytes());
        hash_u32s!(
            row.source_row_id,
            row.dispatch_domain_id,
            row.runtime_variant_id,
            row.source_state_index,
            row.source_template_id,
            row.source_state_template_id,
            row.crossed_state_template_id,
        );
        hash.update(row.crossed_spin_state_class.to_le_bytes());
        hash_u32s!(
            row.direct_executor_id,
            row.crossing_exact_factor_id,
            row.embedding_count,
            row.projection_count,
        );
    }
    count!(parts.source_embeddings);
    for row in &parts.source_embeddings {
        hash_u32s!(
            row.full_component,
            row.source_component_or_sentinel,
            row.exact_factor_id,
        );
    }
    count!(parts.source_projections);
    for row in &parts.source_projections {
        hash_u32s!(row.source_component, row.full_component);
    }
    count!(parts.resolved_source_selections);
    for row in &parts.resolved_source_selections {
        hash_u32s!(row.source_slot, row.dispatch_variant_id);
    }
    count!(parts.public_helicities);
    for value in &parts.public_helicities {
        hash.update(value.to_le_bytes());
    }
    count!(parts.exact_factors);
    for factor in &parts.exact_factors {
        for rational in [factor.real(), factor.imag()] {
            hash.update(rational.numerator().to_le_bytes());
            hash.update(rational.denominator().to_le_bytes());
        }
    }
    hash.update(
        parts
            .closure_proofs
            .expected_semantic_completeness_digest()
            .as_bytes(),
    );
    SemanticDigest::new(hash.finalize().into())
}

#[cfg(test)]
#[path = "direct_plan_tests.rs"]
pub(crate) mod tests;
