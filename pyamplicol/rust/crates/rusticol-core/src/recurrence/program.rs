// SPDX-License-Identifier: 0BSD

//! Compact, immutable recurrence schedule produced by the recurrence builder.
//!
//! The owned vectors are build-time storage. Evaluation consumes only packed
//! slices and ranges, so traversing a validated program requires no maps or
//! heap allocation.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::{
    CheckedTableRange, ContributionKey, CurrentCoreKey, DynamicLCColorState, ExactComplexRational,
    ExactRational, RecurrenceNodeKind, RecurrenceStrategy, SemanticDigest, SourceStateAssignment,
};
use crate::{RusticolError, RusticolResult};

pub const CLOSURE_TARGET_DOMAIN_PROOF_KIND: &str = "complete-recurrence-closure-target-domain-v1";

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn checked_table_len(label: &str, length: usize) -> RusticolResult<u64> {
    u64::try_from(length).map_err(|_| invalid(format!("{label} length {length} exceeds u64")))
}

fn validate_permutation(label: &str, values: &[u32], expected_len: usize) -> RusticolResult<()> {
    if values.len() != expected_len {
        return Err(invalid(format!(
            "{label} has length {}, expected {expected_len}",
            values.len()
        )));
    }
    let mut seen = vec![false; expected_len];
    for (index, value) in values.iter().copied().enumerate() {
        let value = usize::try_from(value)
            .map_err(|_| invalid(format!("{label} row {index} exceeds usize")))?;
        let Some(slot) = seen.get_mut(value) else {
            return Err(invalid(format!(
                "{label} row {index} references out-of-range slot {value}"
            )));
        };
        if std::mem::replace(slot, true) {
            return Err(invalid(format!(
                "{label} repeats slot {value} at row {index}"
            )));
        }
    }
    Ok(())
}

fn hash_exact_factor(hash: &mut Sha256, factor: ExactComplexRational) {
    for rational in [factor.real(), factor.imag()] {
        hash.update(rational.numerator().to_le_bytes());
        hash.update(rational.denominator().to_le_bytes());
    }
}

fn hash_optional_u32(hash: &mut Sha256, value: Option<u32>) {
    hash.update(value.unwrap_or(u32::MAX).to_le_bytes());
}

fn hash_optional_digest(hash: &mut Sha256, value: Option<SemanticDigest>) {
    hash.update(value.map_or([0; 32], |digest| *digest.as_bytes()));
}

fn semantic_digest_from_fields(
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

/// One unaggregated exact closure obligation.
///
/// These rows are cold proof metadata. Runtime closure rows are formed only
/// after the complete contribution multiset has been validated and grouped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosureProofContributionV2 {
    id: u32,
    target_sector_id: u32,
    target_destination_id: Option<u32>,
    target_helicity_id: Option<u32>,
    closure_template_id: u32,
    closure_template_semantic_digest: SemanticDigest,
    quantum_flow_template_id: Option<u32>,
    construction_parent_builder_ids: Box<[u32]>,
    construction_parent_runtime_ids: Box<[Option<u32>]>,
    construction_parent_semantic_digests: Box<[SemanticDigest]>,
    construction_parent_color_digests: Box<[SemanticDigest]>,
    construction_parent_permutation: Box<[u32]>,
    reconstruction_parent_permutation: Box<[u32]>,
    evaluator_parent_permutation: Box<[u32]>,
    color_witness_term_id: u32,
    color_witness_proof_digest: SemanticDigest,
    three_line_certificate_id: Option<u32>,
    pairing_certificate_ids: Box<[u32]>,
    reflection_certificate_id: Option<u32>,
    exact_factor: ExactComplexRational,
    multiplicity: u32,
}

/// Compact commitment to the complete accepted closure-candidate multiset.
///
/// The builder computes this before aggregation. Loading recomputes it from the
/// persisted proof contributions and rejects omitted, duplicated, or altered
/// candidates without retaining a second expanded closure table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClosureCandidateDomainCertificateV1 {
    accepted_candidate_count: u64,
    accepted_candidate_digest: SemanticDigest,
}

/// Exact proof that complete physical colour sectors have no retained closure.
///
/// The construction pass produces this certificate only after enumerating the
/// complete physical-sector x resolved-helicity closure domain.  Its digest
/// binds every target row, including rows with an empty candidate list, plus
/// every original candidate identity and exact aggregate.  The compact sector
/// list is therefore a consequence of an exhaustive native proof, not an
/// inference from missing runtime destinations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosureTargetDomainCertificateV1 {
    physical_sector_count: u32,
    resolved_helicity_count: u32,
    certified_structural_zero_physical_sector_ids: Box<[u32]>,
    semantic_digest: SemanticDigest,
}

impl ClosureTargetDomainCertificateV1 {
    pub fn new(
        physical_sector_count: u32,
        resolved_helicity_count: u32,
        certified_structural_zero_physical_sector_ids: Vec<u32>,
        semantic_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        if physical_sector_count == 0 {
            return Err(invalid(
                "closure target-domain certificate requires physical sectors",
            ));
        }
        if resolved_helicity_count == 0 {
            return Err(invalid(
                "closure target-domain certificate requires resolved helicities",
            ));
        }
        if certified_structural_zero_physical_sector_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || certified_structural_zero_physical_sector_ids
                .iter()
                .any(|sector_id| *sector_id >= physical_sector_count)
        {
            return Err(invalid(
                "closure target-domain structural-zero sectors are not canonical",
            ));
        }
        Ok(Self {
            physical_sector_count,
            resolved_helicity_count,
            certified_structural_zero_physical_sector_ids:
                certified_structural_zero_physical_sector_ids.into_boxed_slice(),
            semantic_digest,
        })
    }

    pub const fn physical_sector_count(&self) -> u32 {
        self.physical_sector_count
    }

    pub const fn resolved_helicity_count(&self) -> u32 {
        self.resolved_helicity_count
    }

    pub fn certified_structural_zero_physical_sector_ids(&self) -> &[u32] {
        &self.certified_structural_zero_physical_sector_ids
    }

    pub const fn semantic_digest(&self) -> SemanticDigest {
        self.semantic_digest
    }
}

impl ClosureCandidateDomainCertificateV1 {
    pub const fn new(
        accepted_candidate_count: u64,
        accepted_candidate_digest: SemanticDigest,
    ) -> Self {
        Self {
            accepted_candidate_count,
            accepted_candidate_digest,
        }
    }

    pub(crate) fn from_identity_digests(
        mut identity_digests: Vec<SemanticDigest>,
    ) -> RusticolResult<Self> {
        identity_digests.sort_unstable();
        let accepted_candidate_count = u64::try_from(identity_digests.len())
            .map_err(|_| invalid("closure candidate-domain count exceeds u64"))?;
        let mut hash = Sha256::new();
        hash.update(b"pyamplicol-recurrence-closure-candidate-domain-v1\0");
        hash.update(accepted_candidate_count.to_le_bytes());
        for digest in identity_digests {
            hash.update(digest.as_bytes());
        }
        Ok(Self::new(
            accepted_candidate_count,
            SemanticDigest::new(hash.finalize().into())?,
        ))
    }

    pub const fn accepted_candidate_count(self) -> u64 {
        self.accepted_candidate_count
    }

    pub const fn accepted_candidate_digest(self) -> SemanticDigest {
        self.accepted_candidate_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn closure_candidate_identity_digest_v1(
    selector_domain_digest: SemanticDigest,
    target_sector_id: u32,
    closure_template_id: u32,
    closure_template_semantic_digest: SemanticDigest,
    quantum_flow_template_id: Option<u32>,
    construction_parent_builder_ids: &[u32],
    construction_parent_semantic_digests: &[SemanticDigest],
    construction_parent_color_digests: &[SemanticDigest],
    construction_parent_permutation: &[u32],
    reconstruction_parent_permutation: &[u32],
    evaluator_parent_permutation: &[u32],
    color_witness_term_id: u32,
    color_witness_proof_digest: SemanticDigest,
    three_line_proof_digest: Option<SemanticDigest>,
    pairing_certificate_ids: &[u32],
    reflection_proof_digest: Option<SemanticDigest>,
    exact_factor: ExactComplexRational,
    multiplicity: u32,
) -> RusticolResult<SemanticDigest> {
    let parent_count = construction_parent_builder_ids.len();
    if construction_parent_semantic_digests.len() != parent_count
        || construction_parent_color_digests.len() != parent_count
    {
        return Err(invalid(
            "closure candidate parent IDs and semantic/color digests have inconsistent lengths",
        ));
    }
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-closure-candidate-v1\0");
    hash.update(selector_domain_digest.as_bytes());
    hash.update(target_sector_id.to_le_bytes());
    hash.update(closure_template_id.to_le_bytes());
    hash.update(closure_template_semantic_digest.as_bytes());
    hash_optional_u32(&mut hash, quantum_flow_template_id);
    for values in [
        construction_parent_builder_ids,
        construction_parent_permutation,
        reconstruction_parent_permutation,
        evaluator_parent_permutation,
        pairing_certificate_ids,
    ] {
        hash.update(
            u64::try_from(values.len())
                .map_err(|_| invalid("closure candidate sequence length exceeds u64"))?
                .to_le_bytes(),
        );
        for value in values {
            hash.update(value.to_le_bytes());
        }
    }
    for digests in [
        construction_parent_semantic_digests,
        construction_parent_color_digests,
    ] {
        hash.update(
            u64::try_from(digests.len())
                .map_err(|_| invalid("closure candidate digest sequence length exceeds u64"))?
                .to_le_bytes(),
        );
        for digest in digests {
            hash.update(digest.as_bytes());
        }
    }
    hash.update(color_witness_term_id.to_le_bytes());
    hash.update(color_witness_proof_digest.as_bytes());
    hash_optional_digest(&mut hash, three_line_proof_digest);
    hash_optional_digest(&mut hash, reflection_proof_digest);
    hash_exact_factor(&mut hash, exact_factor);
    hash.update(multiplicity.to_le_bytes());
    SemanticDigest::new(hash.finalize().into())
}

impl ClosureProofContributionV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        target_sector_id: u32,
        target_destination_id: Option<u32>,
        target_helicity_id: Option<u32>,
        closure_template_id: u32,
        closure_template_semantic_digest: SemanticDigest,
        quantum_flow_template_id: Option<u32>,
        construction_parent_builder_ids: Vec<u32>,
        construction_parent_runtime_ids: Vec<Option<u32>>,
        construction_parent_semantic_digests: Vec<SemanticDigest>,
        construction_parent_color_digests: Vec<SemanticDigest>,
        construction_parent_permutation: Vec<u32>,
        reconstruction_parent_permutation: Vec<u32>,
        evaluator_parent_permutation: Vec<u32>,
        color_witness_term_id: u32,
        color_witness_proof_digest: SemanticDigest,
        three_line_certificate_id: Option<u32>,
        mut pairing_certificate_ids: Vec<u32>,
        reflection_certificate_id: Option<u32>,
        exact_factor: ExactComplexRational,
        multiplicity: u32,
    ) -> RusticolResult<Self> {
        let parent_count = construction_parent_builder_ids.len();
        if parent_count == 0 {
            return Err(invalid(
                "closure proof contribution requires at least one parent",
            ));
        }
        u32::try_from(parent_count)
            .map_err(|_| invalid("closure proof parent count exceeds u32"))?;
        if construction_parent_runtime_ids.len() != parent_count
            || construction_parent_semantic_digests.len() != parent_count
            || construction_parent_color_digests.len() != parent_count
        {
            return Err(invalid(
                "closure proof builder/runtime parent IDs and semantic/color digests have inconsistent lengths",
            ));
        }
        validate_permutation(
            "closure construction-parent permutation",
            &construction_parent_permutation,
            parent_count,
        )?;
        validate_permutation(
            "closure reconstruction-parent permutation",
            &reconstruction_parent_permutation,
            parent_count,
        )?;
        validate_permutation(
            "closure evaluator-parent permutation",
            &evaluator_parent_permutation,
            parent_count,
        )?;
        if exact_factor.is_zero() {
            return Err(invalid(
                "closure proof contribution factor must not be zero",
            ));
        }
        if multiplicity == 0 {
            return Err(invalid(
                "closure proof contribution multiplicity must be positive",
            ));
        }
        pairing_certificate_ids.sort_unstable();
        pairing_certificate_ids.dedup();
        Ok(Self {
            id,
            target_sector_id,
            target_destination_id,
            target_helicity_id,
            closure_template_id,
            closure_template_semantic_digest,
            quantum_flow_template_id,
            construction_parent_builder_ids: construction_parent_builder_ids.into_boxed_slice(),
            construction_parent_runtime_ids: construction_parent_runtime_ids.into_boxed_slice(),
            construction_parent_semantic_digests: construction_parent_semantic_digests
                .into_boxed_slice(),
            construction_parent_color_digests: construction_parent_color_digests.into_boxed_slice(),
            construction_parent_permutation: construction_parent_permutation.into_boxed_slice(),
            reconstruction_parent_permutation: reconstruction_parent_permutation.into_boxed_slice(),
            evaluator_parent_permutation: evaluator_parent_permutation.into_boxed_slice(),
            color_witness_term_id,
            color_witness_proof_digest,
            three_line_certificate_id,
            pairing_certificate_ids: pairing_certificate_ids.into_boxed_slice(),
            reflection_certificate_id,
            exact_factor,
            multiplicity,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn target_sector_id(&self) -> u32 {
        self.target_sector_id
    }

    pub const fn target_destination_id(&self) -> Option<u32> {
        self.target_destination_id
    }

    pub const fn target_helicity_id(&self) -> Option<u32> {
        self.target_helicity_id
    }

    pub const fn closure_template_id(&self) -> u32 {
        self.closure_template_id
    }

    pub const fn closure_template_semantic_digest(&self) -> SemanticDigest {
        self.closure_template_semantic_digest
    }

    pub const fn quantum_flow_template_id(&self) -> Option<u32> {
        self.quantum_flow_template_id
    }

    pub fn construction_parent_builder_ids(&self) -> &[u32] {
        &self.construction_parent_builder_ids
    }

    pub fn construction_parent_runtime_ids(&self) -> &[Option<u32>] {
        &self.construction_parent_runtime_ids
    }

    pub fn construction_parent_semantic_digests(&self) -> &[SemanticDigest] {
        &self.construction_parent_semantic_digests
    }

    pub fn construction_parent_color_digests(&self) -> &[SemanticDigest] {
        &self.construction_parent_color_digests
    }

    pub fn construction_parent_permutation(&self) -> &[u32] {
        &self.construction_parent_permutation
    }

    pub fn reconstruction_parent_permutation(&self) -> &[u32] {
        &self.reconstruction_parent_permutation
    }

    pub fn evaluator_parent_permutation(&self) -> &[u32] {
        &self.evaluator_parent_permutation
    }

    pub fn evaluator_parent_builder_ids(&self) -> Vec<u32> {
        self.evaluator_parent_permutation
            .iter()
            .map(|slot| self.construction_parent_builder_ids[*slot as usize])
            .collect()
    }

    pub fn evaluator_parent_runtime_ids(&self) -> Option<Vec<u32>> {
        self.evaluator_parent_permutation
            .iter()
            .map(|slot| self.construction_parent_runtime_ids[*slot as usize])
            .collect()
    }

    pub const fn color_witness_term_id(&self) -> u32 {
        self.color_witness_term_id
    }

    pub const fn color_witness_proof_digest(&self) -> SemanticDigest {
        self.color_witness_proof_digest
    }

    pub const fn three_line_certificate_id(&self) -> Option<u32> {
        self.three_line_certificate_id
    }

    pub fn pairing_certificate_ids(&self) -> &[u32] {
        &self.pairing_certificate_ids
    }

    pub const fn reflection_certificate_id(&self) -> Option<u32> {
        self.reflection_certificate_id
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }

    pub const fn multiplicity(&self) -> u32 {
        self.multiplicity
    }
}

/// One exact group of proof contributions represented by at most one runtime
/// closure row. Exact cancellation is retained with both runtime IDs absent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosureExecutionProofGroupV2 {
    id: u32,
    emitted_runtime_closure_term_id: Option<u32>,
    emitted_direct_closure_row_id: Option<u32>,
    contribution_range: CheckedTableRange,
    exact_summed_factor: ExactComplexRational,
    component_factor_digest: SemanticDigest,
    candidate_selector_domain_digest: SemanticDigest,
    selector_domain_digest: SemanticDigest,
}

impl ClosureExecutionProofGroupV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        emitted_runtime_closure_term_id: Option<u32>,
        emitted_direct_closure_row_id: Option<u32>,
        contribution_range: CheckedTableRange,
        exact_summed_factor: ExactComplexRational,
        component_factor_digest: SemanticDigest,
        selector_domain_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        Self::new_with_candidate_selector_domain(
            id,
            emitted_runtime_closure_term_id,
            emitted_direct_closure_row_id,
            contribution_range,
            exact_summed_factor,
            component_factor_digest,
            selector_domain_digest,
            selector_domain_digest,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_candidate_selector_domain(
        id: u32,
        emitted_runtime_closure_term_id: Option<u32>,
        emitted_direct_closure_row_id: Option<u32>,
        contribution_range: CheckedTableRange,
        exact_summed_factor: ExactComplexRational,
        component_factor_digest: SemanticDigest,
        candidate_selector_domain_digest: SemanticDigest,
        selector_domain_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        contribution_range.end("closure proof-group contribution")?;
        if exact_summed_factor.is_zero() {
            if emitted_runtime_closure_term_id.is_some() || emitted_direct_closure_row_id.is_some()
            {
                return Err(invalid(
                    "exact-zero closure proof group must not reference a runtime closure",
                ));
            }
        } else if emitted_runtime_closure_term_id.is_none() {
            return Err(invalid(
                "nonzero closure proof group requires a runtime closure term",
            ));
        }
        Ok(Self {
            id,
            emitted_runtime_closure_term_id,
            emitted_direct_closure_row_id,
            contribution_range,
            exact_summed_factor,
            component_factor_digest,
            candidate_selector_domain_digest,
            selector_domain_digest,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn emitted_runtime_closure_term_id(&self) -> Option<u32> {
        self.emitted_runtime_closure_term_id
    }

    pub const fn emitted_direct_closure_row_id(&self) -> Option<u32> {
        self.emitted_direct_closure_row_id
    }

    pub const fn contribution_range(&self) -> CheckedTableRange {
        self.contribution_range
    }

    pub const fn exact_summed_factor(&self) -> ExactComplexRational {
        self.exact_summed_factor
    }

    pub const fn component_factor_digest(&self) -> SemanticDigest {
        self.component_factor_digest
    }

    pub const fn candidate_selector_domain_digest(&self) -> SemanticDigest {
        self.candidate_selector_domain_digest
    }

    pub const fn selector_domain_digest(&self) -> SemanticDigest {
        self.selector_domain_digest
    }

    pub fn with_direct_closure_row_id(&self, row_id: Option<u32>) -> RusticolResult<Self> {
        self.with_direct_closure_binding(
            row_id,
            self.component_factor_digest,
            self.selector_domain_digest,
        )
    }

    pub fn with_direct_closure_binding(
        &self,
        row_id: Option<u32>,
        component_factor_digest: SemanticDigest,
        selector_domain_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        Self::new_with_candidate_selector_domain(
            self.id,
            self.emitted_runtime_closure_term_id,
            row_id,
            self.contribution_range,
            self.exact_summed_factor,
            component_factor_digest,
            self.candidate_selector_domain_digest,
            selector_domain_digest,
        )
    }
}

/// Exact reciprocal-orbit certificate used by folded pure-gluon closures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReflectionCertificateV1 {
    id: u32,
    orbit_identity: SemanticDigest,
    reciprocal_identity: SemanticDigest,
    source_permutation: Box<[u32]>,
    exact_phase: ExactComplexRational,
    fixed_point: bool,
    orbit_size: u32,
    proof_algorithm_id: u32,
    proof_digest: SemanticDigest,
}

impl ReflectionCertificateV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        orbit_identity: SemanticDigest,
        reciprocal_identity: SemanticDigest,
        source_permutation: Vec<u32>,
        exact_phase: ExactComplexRational,
        fixed_point: bool,
        orbit_size: u32,
        proof_algorithm_id: u32,
        proof_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        validate_permutation(
            "reflection source permutation",
            &source_permutation,
            source_permutation.len(),
        )?;
        if exact_phase.is_zero() {
            return Err(invalid("reflection phase must not be zero"));
        }
        match (
            fixed_point,
            orbit_size,
            orbit_identity == reciprocal_identity,
        ) {
            (true, 1, true) | (false, 2, false) => {}
            _ => {
                return Err(invalid(
                    "reflection fixed-point flag, orbit size, and reciprocal identities disagree",
                ));
            }
        }
        Ok(Self {
            id,
            orbit_identity,
            reciprocal_identity,
            source_permutation: source_permutation.into_boxed_slice(),
            exact_phase,
            fixed_point,
            orbit_size,
            proof_algorithm_id,
            proof_digest,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn orbit_identity(&self) -> SemanticDigest {
        self.orbit_identity
    }

    pub const fn reciprocal_identity(&self) -> SemanticDigest {
        self.reciprocal_identity
    }

    pub fn source_permutation(&self) -> &[u32] {
        &self.source_permutation
    }

    pub const fn exact_phase(&self) -> ExactComplexRational {
        self.exact_phase
    }

    pub const fn fixed_point(&self) -> bool {
        self.fixed_point
    }

    pub const fn orbit_size(&self) -> u32 {
        self.orbit_size
    }

    pub const fn proof_algorithm_id(&self) -> u32 {
        self.proof_algorithm_id
    }

    pub const fn proof_digest(&self) -> SemanticDigest {
        self.proof_digest
    }
}

/// Certified traversal class for a three-open-line closure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ThreeLineTraversalKindV1 {
    Direct = 0,
    Partner = 1,
}

impl ThreeLineTraversalKindV1 {
    pub const fn code(self) -> u32 {
        self as u32
    }
}

impl TryFrom<u32> for ThreeLineTraversalKindV1 {
    type Error = RusticolError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Direct),
            1 => Ok(Self::Partner),
            _ => Err(invalid(format!(
                "three-line traversal kind must be direct (0) or partner (1), found {value}"
            ))),
        }
    }
}

/// Cold exact witness for one direct or partner traversal of three open lines.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreeLineTraversalCertificateV1 {
    id: u32,
    sector_id: u32,
    kind: ThreeLineTraversalKindV1,
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

impl ThreeLineTraversalCertificateV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        sector_id: u32,
        kind: ThreeLineTraversalKindV1,
        sink_block_ordinal: u32,
        reference_block_order: Vec<u32>,
        witness_block_order: Vec<u32>,
        block_permutation: Vec<u32>,
        reference_source_order: Vec<u32>,
        witness_source_order: Vec<u32>,
        source_position_permutation: Vec<u32>,
        closure_anchor_source_slot: u32,
        pairing_rule_id: u32,
        proof_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        const BLOCK_COUNT: usize = 3;
        validate_permutation(
            "three-line reference block order",
            &reference_block_order,
            BLOCK_COUNT,
        )?;
        validate_permutation(
            "three-line witness block order",
            &witness_block_order,
            BLOCK_COUNT,
        )?;
        validate_permutation(
            "three-line block permutation",
            &block_permutation,
            BLOCK_COUNT,
        )?;
        if sink_block_ordinal >= BLOCK_COUNT as u32 {
            return Err(invalid(format!(
                "three-line sink block ordinal {sink_block_ordinal} is out of range"
            )));
        }
        if reference_block_order[BLOCK_COUNT - 1] != sink_block_ordinal
            || witness_block_order[BLOCK_COUNT - 1] != sink_block_ordinal
        {
            return Err(invalid(
                "three-line reference and witness traversals must end at the sink block",
            ));
        }
        for (witness_position, reference_position) in block_permutation.iter().copied().enumerate()
        {
            if witness_block_order[witness_position]
                != reference_block_order[reference_position as usize]
            {
                return Err(invalid(
                    "three-line block permutation does not transport reference order to witness order",
                ));
            }
        }
        let expected_witness_order = match kind {
            ThreeLineTraversalKindV1::Direct => reference_block_order.clone(),
            ThreeLineTraversalKindV1::Partner => {
                let mut partner = reference_block_order.clone();
                partner.swap(0, 1);
                partner
            }
        };
        if witness_block_order != expected_witness_order {
            return Err(invalid(format!(
                "three-line {:?} traversal has an inconsistent witness block order",
                kind
            )));
        }

        if reference_source_order.is_empty() {
            return Err(invalid(
                "three-line traversal requires a nonempty source order",
            ));
        }
        if witness_source_order.len() != reference_source_order.len() {
            return Err(invalid(
                "three-line reference and witness source orders have inconsistent lengths",
            ));
        }
        validate_permutation(
            "three-line source-position permutation",
            &source_position_permutation,
            reference_source_order.len(),
        )?;
        if reference_source_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != reference_source_order.len()
            || witness_source_order
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != witness_source_order.len()
        {
            return Err(invalid(
                "three-line source orders must contain unique source slots",
            ));
        }
        for (witness_position, reference_position) in
            source_position_permutation.iter().copied().enumerate()
        {
            if witness_source_order[witness_position]
                != reference_source_order[reference_position as usize]
            {
                return Err(invalid(
                    "three-line source-position permutation does not transport reference order to witness order",
                ));
            }
        }
        if reference_source_order.last() != Some(&closure_anchor_source_slot)
            || witness_source_order.last() != Some(&closure_anchor_source_slot)
        {
            return Err(invalid(
                "three-line reference and witness source orders must end at the closure anchor",
            ));
        }
        let expected_proof_digest = three_line_traversal_proof_digest_v1(
            sector_id,
            kind,
            sink_block_ordinal,
            &reference_block_order,
            &witness_block_order,
            &block_permutation,
            &reference_source_order,
            &witness_source_order,
            &source_position_permutation,
            closure_anchor_source_slot,
            pairing_rule_id,
        )?;
        if proof_digest != expected_proof_digest {
            return Err(invalid("three-line traversal proof digest mismatch"));
        }
        Ok(Self {
            id,
            sector_id,
            kind,
            sink_block_ordinal,
            reference_block_order: reference_block_order.into_boxed_slice(),
            witness_block_order: witness_block_order.into_boxed_slice(),
            block_permutation: block_permutation.into_boxed_slice(),
            reference_source_order: reference_source_order.into_boxed_slice(),
            witness_source_order: witness_source_order.into_boxed_slice(),
            source_position_permutation: source_position_permutation.into_boxed_slice(),
            closure_anchor_source_slot,
            pairing_rule_id,
            proof_digest,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn sector_id(&self) -> u32 {
        self.sector_id
    }

    pub const fn kind(&self) -> ThreeLineTraversalKindV1 {
        self.kind
    }

    pub const fn sink_block_ordinal(&self) -> u32 {
        self.sink_block_ordinal
    }

    pub fn reference_block_order(&self) -> &[u32] {
        &self.reference_block_order
    }

    pub fn witness_block_order(&self) -> &[u32] {
        &self.witness_block_order
    }

    pub fn block_permutation(&self) -> &[u32] {
        &self.block_permutation
    }

    pub fn reference_source_order(&self) -> &[u32] {
        &self.reference_source_order
    }

    pub fn witness_source_order(&self) -> &[u32] {
        &self.witness_source_order
    }

    pub fn source_position_permutation(&self) -> &[u32] {
        &self.source_position_permutation
    }

    pub const fn closure_anchor_source_slot(&self) -> u32 {
        self.closure_anchor_source_slot
    }

    pub const fn pairing_rule_id(&self) -> u32 {
        self.pairing_rule_id
    }

    pub const fn proof_digest(&self) -> SemanticDigest {
        self.proof_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn three_line_traversal_proof_digest_v1(
    sector_id: u32,
    kind: ThreeLineTraversalKindV1,
    sink_block_ordinal: u32,
    reference_block_order: &[u32],
    witness_block_order: &[u32],
    block_permutation: &[u32],
    reference_source_order: &[u32],
    witness_source_order: &[u32],
    source_position_permutation: &[u32],
    closure_anchor_source_slot: u32,
    pairing_rule_id: u32,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-three-line-traversal-v1\0");
    hash.update(sector_id.to_le_bytes());
    hash.update([kind as u8]);
    hash.update(sink_block_ordinal.to_le_bytes());
    for values in [
        reference_block_order,
        witness_block_order,
        block_permutation,
        reference_source_order,
        witness_source_order,
        source_position_permutation,
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
    hash.update(closure_anchor_source_slot.to_le_bytes());
    hash.update(pairing_rule_id.to_le_bytes());
    SemanticDigest::new(hash.finalize().into())
}

fn closure_candidate_domain_certificate_from_tables_v1(
    contributions: &[ClosureProofContributionV2],
    groups: &[ClosureExecutionProofGroupV2],
    reflection_certificates: &[ReflectionCertificateV1],
    three_line_traversal_certificates: &[ThreeLineTraversalCertificateV1],
) -> RusticolResult<ClosureCandidateDomainCertificateV1> {
    let mut identity_digests = Vec::with_capacity(contributions.len());
    for group in groups {
        let range = group
            .contribution_range
            .as_usize_range(contributions.len(), "closure candidate-domain contribution")?;
        for contribution in &contributions[range] {
            let three_line_proof_digest = contribution
                .three_line_certificate_id
                .map(|certificate_id| {
                    three_line_traversal_certificates
                        .get(certificate_id as usize)
                        .map(ThreeLineTraversalCertificateV1::proof_digest)
                        .ok_or_else(|| {
                            invalid(format!(
                                "closure candidate references absent three-line certificate {certificate_id}"
                            ))
                        })
                })
                .transpose()?;
            let reflection_proof_digest = contribution
                .reflection_certificate_id
                .map(|certificate_id| {
                    reflection_certificates
                        .get(certificate_id as usize)
                        .map(ReflectionCertificateV1::proof_digest)
                        .ok_or_else(|| {
                            invalid(format!(
                                "closure candidate references absent reflection certificate {certificate_id}"
                            ))
                        })
                })
                .transpose()?;
            identity_digests.push(closure_candidate_identity_digest_v1(
                group.candidate_selector_domain_digest,
                contribution.target_sector_id,
                contribution.closure_template_id,
                contribution.closure_template_semantic_digest,
                contribution.quantum_flow_template_id,
                &contribution.construction_parent_builder_ids,
                &contribution.construction_parent_semantic_digests,
                &contribution.construction_parent_color_digests,
                &contribution.construction_parent_permutation,
                &contribution.reconstruction_parent_permutation,
                &contribution.evaluator_parent_permutation,
                contribution.color_witness_term_id,
                contribution.color_witness_proof_digest,
                three_line_proof_digest,
                &contribution.pairing_certificate_ids,
                reflection_proof_digest,
                contribution.exact_factor,
                contribution.multiplicity,
            )?);
        }
    }
    ClosureCandidateDomainCertificateV1::from_identity_digests(identity_digests)
}

/// Cold exact closure-completeness metadata shared by program and direct plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClosureProofMetadataV2 {
    contributions: Box<[ClosureProofContributionV2]>,
    groups: Box<[ClosureExecutionProofGroupV2]>,
    reflection_certificates: Box<[ReflectionCertificateV1]>,
    three_line_traversal_certificates: Box<[ThreeLineTraversalCertificateV1]>,
    candidate_domain_certificate: ClosureCandidateDomainCertificateV1,
    expected_semantic_completeness_digest: SemanticDigest,
}

impl ClosureProofMetadataV2 {
    pub fn new(
        contributions: Vec<ClosureProofContributionV2>,
        groups: Vec<ClosureExecutionProofGroupV2>,
        reflection_certificates: Vec<ReflectionCertificateV1>,
    ) -> RusticolResult<Self> {
        Self::new_with_three_line_certificates(
            contributions,
            groups,
            reflection_certificates,
            Vec::new(),
        )
    }

    pub fn new_with_three_line_certificates(
        contributions: Vec<ClosureProofContributionV2>,
        groups: Vec<ClosureExecutionProofGroupV2>,
        reflection_certificates: Vec<ReflectionCertificateV1>,
        three_line_traversal_certificates: Vec<ThreeLineTraversalCertificateV1>,
    ) -> RusticolResult<Self> {
        validate_closure_proof_tables(
            &contributions,
            &groups,
            &reflection_certificates,
            &three_line_traversal_certificates,
        )?;
        let candidate_domain_certificate = closure_candidate_domain_certificate_from_tables_v1(
            &contributions,
            &groups,
            &reflection_certificates,
            &three_line_traversal_certificates,
        )?;
        Self::new_with_three_line_certificates_and_candidate_domain(
            contributions,
            groups,
            reflection_certificates,
            three_line_traversal_certificates,
            candidate_domain_certificate,
        )
    }

    pub fn new_with_three_line_certificates_and_candidate_domain(
        contributions: Vec<ClosureProofContributionV2>,
        groups: Vec<ClosureExecutionProofGroupV2>,
        reflection_certificates: Vec<ReflectionCertificateV1>,
        three_line_traversal_certificates: Vec<ThreeLineTraversalCertificateV1>,
        candidate_domain_certificate: ClosureCandidateDomainCertificateV1,
    ) -> RusticolResult<Self> {
        validate_closure_proof_tables(
            &contributions,
            &groups,
            &reflection_certificates,
            &three_line_traversal_certificates,
        )?;
        let actual_candidate_domain_certificate =
            closure_candidate_domain_certificate_from_tables_v1(
                &contributions,
                &groups,
                &reflection_certificates,
                &three_line_traversal_certificates,
            )?;
        if actual_candidate_domain_certificate != candidate_domain_certificate {
            return Err(invalid(
                "closure candidate-domain certificate does not match persisted proof rows",
            ));
        }
        let expected_semantic_completeness_digest =
            closure_proof_semantic_completeness_digest_with_three_line_v2(
                &contributions,
                &groups,
                &reflection_certificates,
                &three_line_traversal_certificates,
            )?;
        Ok(Self {
            contributions: contributions.into_boxed_slice(),
            groups: groups.into_boxed_slice(),
            reflection_certificates: reflection_certificates.into_boxed_slice(),
            three_line_traversal_certificates: three_line_traversal_certificates.into_boxed_slice(),
            candidate_domain_certificate,
            expected_semantic_completeness_digest,
        })
    }

    pub fn new_with_expected_digest(
        contributions: Vec<ClosureProofContributionV2>,
        groups: Vec<ClosureExecutionProofGroupV2>,
        reflection_certificates: Vec<ReflectionCertificateV1>,
        expected_semantic_completeness_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        let metadata = Self::new(contributions, groups, reflection_certificates)?;
        metadata.validate_expected_digest(expected_semantic_completeness_digest)
    }

    pub fn new_with_three_line_certificates_and_expected_digest(
        contributions: Vec<ClosureProofContributionV2>,
        groups: Vec<ClosureExecutionProofGroupV2>,
        reflection_certificates: Vec<ReflectionCertificateV1>,
        three_line_traversal_certificates: Vec<ThreeLineTraversalCertificateV1>,
        expected_semantic_completeness_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        let metadata = Self::new_with_three_line_certificates(
            contributions,
            groups,
            reflection_certificates,
            three_line_traversal_certificates,
        )?;
        metadata.validate_expected_digest(expected_semantic_completeness_digest)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_three_line_certificates_candidate_domain_and_expected_digest(
        contributions: Vec<ClosureProofContributionV2>,
        groups: Vec<ClosureExecutionProofGroupV2>,
        reflection_certificates: Vec<ReflectionCertificateV1>,
        three_line_traversal_certificates: Vec<ThreeLineTraversalCertificateV1>,
        candidate_domain_certificate: ClosureCandidateDomainCertificateV1,
        expected_semantic_completeness_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        let metadata = Self::new_with_three_line_certificates_and_candidate_domain(
            contributions,
            groups,
            reflection_certificates,
            three_line_traversal_certificates,
            candidate_domain_certificate,
        )?;
        metadata.validate_expected_digest(expected_semantic_completeness_digest)
    }

    fn validate_expected_digest(
        self,
        expected_semantic_completeness_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        if self.expected_semantic_completeness_digest != expected_semantic_completeness_digest {
            return Err(invalid(
                "closure proof semantic completeness digest mismatch",
            ));
        }
        Ok(self)
    }

    pub fn contributions(&self) -> &[ClosureProofContributionV2] {
        &self.contributions
    }

    pub fn groups(&self) -> &[ClosureExecutionProofGroupV2] {
        &self.groups
    }

    pub fn reflection_certificates(&self) -> &[ReflectionCertificateV1] {
        &self.reflection_certificates
    }

    pub fn three_line_traversal_certificates(&self) -> &[ThreeLineTraversalCertificateV1] {
        &self.three_line_traversal_certificates
    }

    pub const fn candidate_domain_certificate(&self) -> ClosureCandidateDomainCertificateV1 {
        self.candidate_domain_certificate
    }

    pub const fn expected_semantic_completeness_digest(&self) -> SemanticDigest {
        self.expected_semantic_completeness_digest
    }

    pub fn group_for_runtime_term(
        &self,
        closure_term_id: u32,
    ) -> Option<&ClosureExecutionProofGroupV2> {
        self.groups
            .iter()
            .find(|group| group.emitted_runtime_closure_term_id == Some(closure_term_id))
    }

    pub fn with_direct_closure_rows(&self, row_by_runtime_term: &[u32]) -> RusticolResult<Self> {
        let groups = self
            .groups
            .iter()
            .map(|group| {
                let row_id = group
                    .emitted_runtime_closure_term_id
                    .map(|term_id| {
                        row_by_runtime_term
                            .get(term_id as usize)
                            .copied()
                            .ok_or_else(|| {
                                invalid(format!(
                                    "closure proof group {} references absent runtime term {term_id}",
                                    group.id
                                ))
                            })
                    })
                    .transpose()?;
                group.with_direct_closure_row_id(row_id)
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Self::new_with_three_line_certificates_and_candidate_domain(
            self.contributions.to_vec(),
            groups,
            self.reflection_certificates.to_vec(),
            self.three_line_traversal_certificates.to_vec(),
            self.candidate_domain_certificate,
        )
    }

    pub(crate) fn validate_tables(&self) -> RusticolResult<()> {
        validate_closure_proof_tables(
            &self.contributions,
            &self.groups,
            &self.reflection_certificates,
            &self.three_line_traversal_certificates,
        )?;
        if closure_proof_semantic_completeness_digest_with_three_line_v2(
            &self.contributions,
            &self.groups,
            &self.reflection_certificates,
            &self.three_line_traversal_certificates,
        )? != self.expected_semantic_completeness_digest
        {
            return Err(invalid(
                "closure proof semantic completeness digest mismatch",
            ));
        }
        if closure_candidate_domain_certificate_from_tables_v1(
            &self.contributions,
            &self.groups,
            &self.reflection_certificates,
            &self.three_line_traversal_certificates,
        )? != self.candidate_domain_certificate
        {
            return Err(invalid(
                "closure candidate-domain certificate does not match persisted proof rows",
            ));
        }
        Ok(())
    }

    pub(crate) fn discard_rows_for_runtime(&mut self) {
        self.contributions = Box::default();
        self.groups = Box::default();
        self.reflection_certificates = Box::default();
        self.three_line_traversal_certificates = Box::default();
    }
}

pub fn closure_component_factor_digest_v2(
    factors: &[ExactComplexRational],
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-closure-component-factors-v2\0");
    hash.update(
        u64::try_from(factors.len())
            .map_err(|_| invalid("closure component-factor count exceeds u64"))?
            .to_le_bytes(),
    );
    for factor in factors {
        hash_exact_factor(&mut hash, *factor);
    }
    SemanticDigest::new(hash.finalize().into())
}

pub fn closure_selector_domain_digest_v2(words: &[u64]) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-closure-selector-domain-v2\0");
    hash.update(
        u64::try_from(words.len())
            .map_err(|_| invalid("closure selector-domain word count exceeds u64"))?
            .to_le_bytes(),
    );
    for word in words {
        hash.update(word.to_le_bytes());
    }
    SemanticDigest::new(hash.finalize().into())
}

pub fn closure_proof_semantic_completeness_digest_v2(
    contributions: &[ClosureProofContributionV2],
    groups: &[ClosureExecutionProofGroupV2],
    reflection_certificates: &[ReflectionCertificateV1],
) -> RusticolResult<SemanticDigest> {
    closure_proof_semantic_completeness_digest_with_three_line_v2(
        contributions,
        groups,
        reflection_certificates,
        &[],
    )
}

pub fn closure_proof_semantic_completeness_digest_with_three_line_v2(
    contributions: &[ClosureProofContributionV2],
    groups: &[ClosureExecutionProofGroupV2],
    reflection_certificates: &[ReflectionCertificateV1],
    three_line_traversal_certificates: &[ThreeLineTraversalCertificateV1],
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-closure-proof-v2\0");
    hash.update(
        u64::try_from(contributions.len())
            .map_err(|_| invalid("closure proof contribution count exceeds u64"))?
            .to_le_bytes(),
    );
    for row in contributions {
        hash.update(row.id.to_le_bytes());
        hash.update(row.target_sector_id.to_le_bytes());
        hash_optional_u32(&mut hash, row.target_destination_id);
        hash_optional_u32(&mut hash, row.target_helicity_id);
        hash.update(row.closure_template_id.to_le_bytes());
        hash.update(row.closure_template_semantic_digest.as_bytes());
        hash_optional_u32(&mut hash, row.quantum_flow_template_id);
        for values in [
            row.construction_parent_builder_ids.as_ref(),
            row.construction_parent_permutation.as_ref(),
            row.reconstruction_parent_permutation.as_ref(),
            row.evaluator_parent_permutation.as_ref(),
        ] {
            hash.update(
                u64::try_from(values.len())
                    .map_err(|_| invalid("closure proof sequence length exceeds u64"))?
                    .to_le_bytes(),
            );
            for value in values {
                hash.update(value.to_le_bytes());
            }
        }
        hash.update(
            u64::try_from(row.construction_parent_runtime_ids.len())
                .map_err(|_| invalid("closure proof runtime-parent sequence length exceeds u64"))?
                .to_le_bytes(),
        );
        for value in row.construction_parent_runtime_ids.iter().copied() {
            hash_optional_u32(&mut hash, value);
        }
        for digests in [
            row.construction_parent_semantic_digests.as_ref(),
            row.construction_parent_color_digests.as_ref(),
        ] {
            hash.update(
                u64::try_from(digests.len())
                    .map_err(|_| invalid("closure proof digest sequence length exceeds u64"))?
                    .to_le_bytes(),
            );
            for digest in digests {
                hash.update(digest.as_bytes());
            }
        }
        hash.update(row.color_witness_term_id.to_le_bytes());
        hash.update(row.color_witness_proof_digest.as_bytes());
        hash_optional_u32(&mut hash, row.three_line_certificate_id);
        hash.update(
            u64::try_from(row.pairing_certificate_ids.len())
                .map_err(|_| invalid("closure pairing-certificate count exceeds u64"))?
                .to_le_bytes(),
        );
        for certificate_id in row.pairing_certificate_ids.iter().copied() {
            hash.update(certificate_id.to_le_bytes());
        }
        hash_optional_u32(&mut hash, row.reflection_certificate_id);
        hash_exact_factor(&mut hash, row.exact_factor);
        hash.update(row.multiplicity.to_le_bytes());
    }
    hash.update(
        u64::try_from(groups.len())
            .map_err(|_| invalid("closure proof group count exceeds u64"))?
            .to_le_bytes(),
    );
    for row in groups {
        hash.update(row.id.to_le_bytes());
        hash_optional_u32(&mut hash, row.emitted_runtime_closure_term_id);
        hash_optional_u32(&mut hash, row.emitted_direct_closure_row_id);
        hash.update(row.contribution_range.start.to_le_bytes());
        hash.update(row.contribution_range.count.to_le_bytes());
        hash_exact_factor(&mut hash, row.exact_summed_factor);
        hash.update(row.component_factor_digest.as_bytes());
        hash.update(row.candidate_selector_domain_digest.as_bytes());
        hash.update(row.selector_domain_digest.as_bytes());
    }
    hash.update(
        u64::try_from(reflection_certificates.len())
            .map_err(|_| invalid("reflection certificate count exceeds u64"))?
            .to_le_bytes(),
    );
    for row in reflection_certificates {
        hash.update(row.id.to_le_bytes());
        hash.update(row.orbit_identity.as_bytes());
        hash.update(row.reciprocal_identity.as_bytes());
        hash.update(
            u64::try_from(row.source_permutation.len())
                .map_err(|_| invalid("reflection permutation length exceeds u64"))?
                .to_le_bytes(),
        );
        for value in row.source_permutation.iter() {
            hash.update(value.to_le_bytes());
        }
        hash_exact_factor(&mut hash, row.exact_phase);
        hash.update(u8::from(row.fixed_point).to_le_bytes());
        hash.update(row.orbit_size.to_le_bytes());
        hash.update(row.proof_algorithm_id.to_le_bytes());
        hash.update(row.proof_digest.as_bytes());
    }
    hash.update(
        u64::try_from(three_line_traversal_certificates.len())
            .map_err(|_| invalid("three-line traversal certificate count exceeds u64"))?
            .to_le_bytes(),
    );
    for row in three_line_traversal_certificates {
        hash.update(row.id.to_le_bytes());
        hash.update(row.sector_id.to_le_bytes());
        hash.update([row.kind as u8]);
        hash.update(row.sink_block_ordinal.to_le_bytes());
        for values in [
            row.reference_block_order.as_ref(),
            row.witness_block_order.as_ref(),
            row.block_permutation.as_ref(),
            row.reference_source_order.as_ref(),
            row.witness_source_order.as_ref(),
            row.source_position_permutation.as_ref(),
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
        hash.update(row.closure_anchor_source_slot.to_le_bytes());
        hash.update(row.pairing_rule_id.to_le_bytes());
        hash.update(row.proof_digest.as_bytes());
    }
    let candidate_domain_certificate = closure_candidate_domain_certificate_from_tables_v1(
        contributions,
        groups,
        reflection_certificates,
        three_line_traversal_certificates,
    )?;
    hash.update(
        candidate_domain_certificate
            .accepted_candidate_count()
            .to_le_bytes(),
    );
    hash.update(
        candidate_domain_certificate
            .accepted_candidate_digest()
            .as_bytes(),
    );
    SemanticDigest::new(hash.finalize().into())
}

fn validate_closure_proof_tables(
    contributions: &[ClosureProofContributionV2],
    groups: &[ClosureExecutionProofGroupV2],
    reflection_certificates: &[ReflectionCertificateV1],
    three_line_traversal_certificates: &[ThreeLineTraversalCertificateV1],
) -> RusticolResult<()> {
    u32::try_from(contributions.len())
        .map_err(|_| invalid("closure proof contribution count exceeds u32"))?;
    u32::try_from(groups.len()).map_err(|_| invalid("closure proof group count exceeds u32"))?;
    u32::try_from(reflection_certificates.len())
        .map_err(|_| invalid("reflection certificate count exceeds u32"))?;
    u32::try_from(three_line_traversal_certificates.len())
        .map_err(|_| invalid("three-line traversal certificate count exceeds u32"))?;

    for (index, certificate) in reflection_certificates.iter().enumerate() {
        if certificate.id != index as u32 {
            return Err(invalid(format!(
                "reflection certificate row {index} has non-dense id {}",
                certificate.id
            )));
        }
    }
    for (index, certificate) in three_line_traversal_certificates.iter().enumerate() {
        if certificate.id != index as u32 {
            return Err(invalid(format!(
                "three-line traversal certificate row {index} has non-dense id {}",
                certificate.id
            )));
        }
    }
    for (index, contribution) in contributions.iter().enumerate() {
        if contribution.id != index as u32 {
            return Err(invalid(format!(
                "closure proof contribution row {index} has non-dense id {}",
                contribution.id
            )));
        }
        if let Some(certificate_id) = contribution.reflection_certificate_id
            && certificate_id as usize >= reflection_certificates.len()
        {
            return Err(invalid(format!(
                "closure proof contribution {index} references absent reflection certificate {certificate_id}"
            )));
        }
        if let Some(certificate_id) = contribution.three_line_certificate_id {
            let Some(certificate) = three_line_traversal_certificates.get(certificate_id as usize)
            else {
                return Err(invalid(format!(
                    "closure proof contribution {index} references absent three-line traversal certificate {certificate_id}"
                )));
            };
            if certificate.sector_id != contribution.target_sector_id {
                return Err(invalid(format!(
                    "closure proof contribution {index} sector {} does not match three-line traversal certificate {certificate_id} sector {}",
                    contribution.target_sector_id, certificate.sector_id
                )));
            }
        }
    }

    let mut next_contribution = 0u64;
    for (index, group) in groups.iter().enumerate() {
        if group.id != index as u32 {
            return Err(invalid(format!(
                "closure proof group row {index} has non-dense id {}",
                group.id
            )));
        }
        if group.contribution_range.start != next_contribution {
            return Err(invalid(format!(
                "closure proof group {index} starts at {}, expected packed offset {next_contribution}",
                group.contribution_range.start
            )));
        }
        let range = group.contribution_range.as_usize_range(
            contributions.len(),
            &format!("closure proof group {index} contribution"),
        )?;
        if range.is_empty() {
            return Err(invalid(format!(
                "closure proof group {index} has no contributions"
            )));
        }
        let group_contributions = &contributions[range];
        next_contribution = group
            .contribution_range
            .end("closure proof-group contribution")?;
        let mut exact_sum = ExactComplexRational::ZERO;
        let mut execution_identity = None;
        for contribution in group_contributions {
            let multiplicity = ExactComplexRational::new(
                ExactRational::new(i128::from(contribution.multiplicity), 1)?,
                ExactRational::ZERO,
            );
            exact_sum =
                exact_sum.checked_add(contribution.exact_factor.checked_mul(multiplicity)?)?;
            let proof_identity = (
                contribution.target_sector_id,
                contribution.target_helicity_id,
                contribution.closure_template_id,
                contribution.quantum_flow_template_id,
                contribution.evaluator_parent_builder_ids(),
            );
            if let Some(expected) = &execution_identity {
                if expected != &proof_identity {
                    return Err(invalid(format!(
                        "closure proof group {index} mixes distinct construction identities"
                    )));
                }
            } else {
                execution_identity = Some(proof_identity);
            }
        }
        if exact_sum != group.exact_summed_factor {
            return Err(invalid(format!(
                "closure proof group {index} exact factor does not equal its contribution sum"
            )));
        }
        match (
            exact_sum.is_zero(),
            group.emitted_runtime_closure_term_id,
            group.emitted_direct_closure_row_id,
        ) {
            (true, None, None) | (false, Some(_), None | Some(_)) => {}
            (true, _, _) => {
                return Err(invalid(format!(
                    "exact-zero closure proof group {index} references a runtime closure"
                )));
            }
            (false, None, _) => {
                return Err(invalid(format!(
                    "nonzero closure proof group {index} has no runtime closure term"
                )));
            }
        }
        if !exact_sum.is_zero() {
            for contribution in group_contributions {
                if contribution.target_destination_id.is_none()
                    || contribution.evaluator_parent_runtime_ids().is_none()
                {
                    return Err(invalid(format!(
                        "nonzero closure proof group {index} lacks a complete runtime binding"
                    )));
                }
            }
        }
    }
    if next_contribution != contributions.len() as u64 {
        return Err(invalid(format!(
            "closure proof groups consume {next_contribution} of {} contributions",
            contributions.len()
        )));
    }
    Ok(())
}

/// One source or propagated current in topological execution order.
///
/// For a source, the key's dynamic color-state ID refers to the materialized
/// result of a compiler-owned source-color seed. This schedule never infers a
/// seed from particle identity, representation, or source position.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceCurrent {
    id: u32,
    key: CurrentCoreKey,
    source_exact_factor: Option<ExactComplexRational>,
    contribution_range: CheckedTableRange,
    finalization_id: Option<u32>,
}

impl RecurrenceCurrent {
    pub fn new(
        id: u32,
        key: CurrentCoreKey,
        source_exact_factor: Option<ExactComplexRational>,
        contribution_range: CheckedTableRange,
        finalization_id: Option<u32>,
    ) -> RusticolResult<Self> {
        contribution_range.end("recurrence current contribution")?;
        if source_exact_factor.is_some_and(ExactComplexRational::is_zero) {
            return Err(invalid("recurrence source factor must not be zero"));
        }
        Ok(Self {
            id,
            key,
            source_exact_factor,
            contribution_range,
            finalization_id,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn key(&self) -> &CurrentCoreKey {
        &self.key
    }

    pub const fn contribution_range(&self) -> CheckedTableRange {
        self.contribution_range
    }

    pub const fn source_exact_factor(&self) -> Option<ExactComplexRational> {
        self.source_exact_factor
    }

    pub const fn fan_in(&self) -> u64 {
        self.contribution_range.count
    }

    pub const fn finalization_id(&self) -> Option<u32> {
        self.finalization_id
    }

    pub const fn is_source(&self) -> bool {
        matches!(self.key.node_kind(), RecurrenceNodeKind::Source)
    }
}

/// One exact contribution accumulated into one result current.
///
/// The key's color-witness term owns the compiler-certified result-component
/// role (active, passive, or absent). The program only stores its materialized
/// result current and does not derive that role from parent ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceContribution {
    id: u32,
    result_current_id: u32,
    parent_current_ids: Box<[u32]>,
    key: ContributionKey,
    exact_factor: ExactComplexRational,
}

impl RecurrenceContribution {
    pub fn new(
        id: u32,
        result_current_id: u32,
        parent_current_ids: Vec<u32>,
        key: ContributionKey,
        exact_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if parent_current_ids.is_empty() {
            return Err(invalid(
                "recurrence contribution requires at least one parent current",
            ));
        }
        u32::try_from(parent_current_ids.len())
            .map_err(|_| invalid("recurrence contribution parent count exceeds u32"))?;
        if exact_factor.is_zero() {
            return Err(invalid("recurrence contribution factor must not be zero"));
        }
        Ok(Self {
            id,
            result_current_id,
            parent_current_ids: parent_current_ids.into_boxed_slice(),
            key,
            exact_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn result_current_id(&self) -> u32 {
        self.result_current_id
    }

    pub fn parent_current_ids(&self) -> &[u32] {
        &self.parent_current_ids
    }

    pub const fn key(&self) -> &ContributionKey {
        &self.key
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }
}

/// Exactly one propagation/finalization operation for one non-source current.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceFinalization {
    id: u32,
    current_id: u32,
    propagator_template_id: Option<u32>,
    exact_factor: ExactComplexRational,
}

/// One complete physical helicity retained by a topology-replay schedule.
///
/// Currents retain only their local source-state ancestry. This catalog names
/// the complete ancestry reached by closure terms so amplitudes from distinct
/// helicities are never added coherently before squaring.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceResolvedHelicity {
    id: u32,
    source_states: Box<[SourceStateAssignment]>,
    public_helicities: Box<[i32]>,
}

/// One live amplitude destination and its packed closure-term range.
///
/// Destinations are sparse: topology replay stores only live combinations of
/// a materialized representative sector and a resolved helicity, never their
/// Cartesian product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecurrenceAmplitudeDestination {
    id: u32,
    target_sector_id: u32,
    target_helicity_id: Option<u32>,
    closure_range: CheckedTableRange,
}

/// One exact topology-replay route from a materialized recurrence to a public sector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceReplayTarget {
    id: u32,
    materialized_sector_id: u32,
    target_sector_id: u32,
    source_slot_permutation: Box<[u32]>,
    source_momentum_signs: Box<[i32]>,
    amplitude_factor: ExactComplexRational,
}

impl RecurrenceReplayTarget {
    pub fn new(
        id: u32,
        materialized_sector_id: u32,
        target_sector_id: u32,
        source_slot_permutation: Vec<u32>,
        source_momentum_signs: Vec<i32>,
        amplitude_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if source_slot_permutation.is_empty() {
            return Err(invalid(
                "recurrence replay target requires a source permutation",
            ));
        }
        if source_momentum_signs.len() != source_slot_permutation.len()
            || source_momentum_signs
                .iter()
                .any(|sign| !matches!(sign, -1 | 1))
        {
            return Err(invalid(
                "recurrence replay target requires one signed momentum map per source",
            ));
        }
        if amplitude_factor.is_zero() {
            return Err(invalid("recurrence replay target factor must not be zero"));
        }
        Ok(Self {
            id,
            materialized_sector_id,
            target_sector_id,
            source_slot_permutation: source_slot_permutation.into_boxed_slice(),
            source_momentum_signs: source_momentum_signs.into_boxed_slice(),
            amplitude_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn materialized_sector_id(&self) -> u32 {
        self.materialized_sector_id
    }

    pub const fn target_sector_id(&self) -> u32 {
        self.target_sector_id
    }

    pub fn source_slot_permutation(&self) -> &[u32] {
        &self.source_slot_permutation
    }

    pub fn source_momentum_signs(&self) -> &[i32] {
        &self.source_momentum_signs
    }

    pub const fn amplitude_factor(&self) -> ExactComplexRational {
        self.amplitude_factor
    }
}

impl RecurrenceAmplitudeDestination {
    pub fn new(
        id: u32,
        target_sector_id: u32,
        target_helicity_id: Option<u32>,
        closure_range: CheckedTableRange,
    ) -> RusticolResult<Self> {
        closure_range.end("recurrence amplitude destination")?;
        if closure_range.count == 0 {
            return Err(invalid(
                "recurrence amplitude destination requires closure terms",
            ));
        }
        Ok(Self {
            id,
            target_sector_id,
            target_helicity_id,
            closure_range,
        })
    }

    pub const fn id(self) -> u32 {
        self.id
    }

    pub const fn target_sector_id(self) -> u32 {
        self.target_sector_id
    }

    pub const fn target_helicity_id(self) -> Option<u32> {
        self.target_helicity_id
    }

    pub const fn closure_range(self) -> CheckedTableRange {
        self.closure_range
    }
}

impl RecurrenceResolvedHelicity {
    pub fn new(
        id: u32,
        source_states: Vec<SourceStateAssignment>,
        public_helicities: Vec<i32>,
    ) -> RusticolResult<Self> {
        if source_states.is_empty() || source_states.len() != public_helicities.len() {
            return Err(invalid(
                "resolved helicity requires one public value per source-state assignment",
            ));
        }
        for (source_slot, assignment) in source_states.iter().copied().enumerate() {
            if assignment.source_slot() as usize != source_slot {
                return Err(invalid(
                    "resolved-helicity source states must cover every source slot in order",
                ));
            }
        }
        Ok(Self {
            id,
            source_states: source_states.into_boxed_slice(),
            public_helicities: public_helicities.into_boxed_slice(),
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub fn source_states(&self) -> &[SourceStateAssignment] {
        &self.source_states
    }

    pub fn public_helicities(&self) -> &[i32] {
        &self.public_helicities
    }
}

impl RecurrenceFinalization {
    pub fn new(
        id: u32,
        current_id: u32,
        propagator_template_id: Option<u32>,
        exact_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if exact_factor.is_zero() {
            return Err(invalid("recurrence finalization factor must not be zero"));
        }
        Ok(Self {
            id,
            current_id,
            propagator_template_id,
            exact_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn current_id(&self) -> u32 {
        self.current_id
    }

    pub const fn propagator_template_id(&self) -> Option<u32> {
        self.propagator_template_id
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }
}

/// One exact signed term contributing to one physical LC target sector.
///
/// The closure template owns the result-component kind. `target_sector_id` is
/// only the reduction destination selected after validating the physical
/// sector's compiler-owned closure anchor; it is not itself an anchor or an
/// instruction for reconstructing one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceClosureTerm {
    id: u32,
    target_destination_id: u32,
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_current_ids: Box<[u32]>,
    exact_factor: ExactComplexRational,
}

impl RecurrenceClosureTerm {
    pub fn new(
        id: u32,
        target_destination_id: u32,
        closure_template_id: u32,
        quantum_flow_template_id: Option<u32>,
        parent_current_ids: Vec<u32>,
        exact_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if parent_current_ids.is_empty() {
            return Err(invalid(
                "recurrence closure term requires at least one parent current",
            ));
        }
        u32::try_from(parent_current_ids.len())
            .map_err(|_| invalid("recurrence closure parent count exceeds u32"))?;
        if exact_factor.is_zero() {
            return Err(invalid("recurrence closure term factor must not be zero"));
        }
        Ok(Self {
            id,
            target_destination_id,
            closure_template_id,
            quantum_flow_template_id,
            parent_current_ids: parent_current_ids.into_boxed_slice(),
            exact_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn target_destination_id(&self) -> u32 {
        self.target_destination_id
    }

    pub const fn closure_template_id(&self) -> u32 {
        self.closure_template_id
    }

    /// Prepared quantum-flow witness for a kernel closure.
    ///
    /// Direct Rusticol closure templates have no vertex coupling and therefore
    /// carry `None`.
    pub const fn quantum_flow_template_id(&self) -> Option<u32> {
        self.quantum_flow_template_id
    }

    pub fn parent_current_ids(&self) -> &[u32] {
        &self.parent_current_ids
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }
}

/// Validated compact recurrence program ready for serialization or execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceProgram {
    strategy: RecurrenceStrategy,
    physical_sector_count: u32,
    retained_helicity_count: u64,
    dynamic_color_states: Box<[DynamicLCColorState]>,
    currents: Box<[RecurrenceCurrent]>,
    contributions: Box<[RecurrenceContribution]>,
    finalizations: Box<[RecurrenceFinalization]>,
    replay_targets: Box<[RecurrenceReplayTarget]>,
    resolved_helicities: Box<[RecurrenceResolvedHelicity]>,
    amplitude_destinations: Box<[RecurrenceAmplitudeDestination]>,
    closure_terms: Box<[RecurrenceClosureTerm]>,
    closure_proofs: ClosureProofMetadataV2,
    closure_target_domain_certificate: Option<ClosureTargetDomainCertificateV1>,
    color_projection_certificate_body: Option<Box<[u8]>>,
}

impl RecurrenceProgram {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        strategy: RecurrenceStrategy,
        physical_sector_count: u32,
        retained_helicity_count: u64,
        dynamic_color_states: Vec<DynamicLCColorState>,
        currents: Vec<RecurrenceCurrent>,
        contributions: Vec<RecurrenceContribution>,
        finalizations: Vec<RecurrenceFinalization>,
        replay_targets: Vec<RecurrenceReplayTarget>,
        resolved_helicities: Vec<RecurrenceResolvedHelicity>,
        amplitude_destinations: Vec<RecurrenceAmplitudeDestination>,
        closure_terms: Vec<RecurrenceClosureTerm>,
    ) -> RusticolResult<Self> {
        let closure_proofs = closure_proofs_from_materialized_terms(
            &currents,
            &amplitude_destinations,
            &closure_terms,
        )?;
        Self::new_with_closure_proofs(
            strategy,
            physical_sector_count,
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
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_closure_proofs(
        strategy: RecurrenceStrategy,
        physical_sector_count: u32,
        retained_helicity_count: u64,
        dynamic_color_states: Vec<DynamicLCColorState>,
        currents: Vec<RecurrenceCurrent>,
        contributions: Vec<RecurrenceContribution>,
        finalizations: Vec<RecurrenceFinalization>,
        replay_targets: Vec<RecurrenceReplayTarget>,
        resolved_helicities: Vec<RecurrenceResolvedHelicity>,
        amplitude_destinations: Vec<RecurrenceAmplitudeDestination>,
        closure_terms: Vec<RecurrenceClosureTerm>,
        closure_proofs: ClosureProofMetadataV2,
    ) -> RusticolResult<Self> {
        let program = Self {
            strategy,
            physical_sector_count,
            retained_helicity_count,
            dynamic_color_states: dynamic_color_states.into_boxed_slice(),
            currents: currents.into_boxed_slice(),
            contributions: contributions.into_boxed_slice(),
            finalizations: finalizations.into_boxed_slice(),
            replay_targets: replay_targets.into_boxed_slice(),
            resolved_helicities: resolved_helicities.into_boxed_slice(),
            amplitude_destinations: amplitude_destinations.into_boxed_slice(),
            closure_terms: closure_terms.into_boxed_slice(),
            closure_proofs,
            closure_target_domain_certificate: None,
            color_projection_certificate_body: None,
        };
        program.validate()?;
        Ok(program)
    }

    pub const fn strategy(&self) -> RecurrenceStrategy {
        self.strategy
    }

    pub const fn physical_sector_count(&self) -> u32 {
        self.physical_sector_count
    }

    pub const fn retained_helicity_count(&self) -> u64 {
        self.retained_helicity_count
    }

    pub fn dynamic_color_states(&self) -> &[DynamicLCColorState] {
        &self.dynamic_color_states
    }

    pub fn currents(&self) -> &[RecurrenceCurrent] {
        &self.currents
    }

    pub fn contributions(&self) -> &[RecurrenceContribution] {
        &self.contributions
    }

    pub fn finalizations(&self) -> &[RecurrenceFinalization] {
        &self.finalizations
    }

    pub fn replay_targets(&self) -> &[RecurrenceReplayTarget] {
        &self.replay_targets
    }

    pub fn resolved_helicities(&self) -> &[RecurrenceResolvedHelicity] {
        &self.resolved_helicities
    }

    pub fn amplitude_destinations(&self) -> &[RecurrenceAmplitudeDestination] {
        &self.amplitude_destinations
    }

    pub fn closure_terms(&self) -> &[RecurrenceClosureTerm] {
        &self.closure_terms
    }

    pub const fn closure_proofs(&self) -> &ClosureProofMetadataV2 {
        &self.closure_proofs
    }

    pub fn closure_target_domain_certificate(&self) -> Option<&ClosureTargetDomainCertificateV1> {
        self.closure_target_domain_certificate.as_ref()
    }

    pub(crate) fn with_closure_target_domain_certificate(
        mut self,
        certificate: ClosureTargetDomainCertificateV1,
    ) -> RusticolResult<Self> {
        if self.strategy != RecurrenceStrategy::ContractedColorUnion {
            return Err(invalid(
                "closure target-domain certificate is restricted to contracted color",
            ));
        }
        if certificate.physical_sector_count() != self.physical_sector_count
            || certificate.resolved_helicity_count() as usize != self.resolved_helicities.len()
        {
            return Err(invalid(
                "closure target-domain certificate disagrees with its recurrence domain",
            ));
        }
        let active_sector_ids = self
            .amplitude_destinations
            .iter()
            .map(|destination| destination.target_sector_id())
            .collect::<BTreeSet<_>>();
        if certificate
            .certified_structural_zero_physical_sector_ids()
            .iter()
            .any(|sector_id| active_sector_ids.contains(sector_id))
        {
            return Err(invalid(
                "closure target-domain certificate marks an active sector as structural zero",
            ));
        }
        self.closure_target_domain_certificate = Some(certificate);
        Ok(self)
    }

    pub(crate) fn with_color_projection_certificate_body(
        mut self,
        body: Vec<u8>,
    ) -> RusticolResult<Self> {
        if body.is_empty() {
            return Err(invalid("recurrence color-projection certificate is empty"));
        }
        if self.strategy != RecurrenceStrategy::TopologyReplay {
            return Err(invalid(
                "recurrence color projection is restricted to topology replay",
            ));
        }
        self.color_projection_certificate_body = Some(body.into_boxed_slice());
        Ok(self)
    }

    pub fn color_projection_certificate_body(&self) -> Option<&[u8]> {
        self.color_projection_certificate_body.as_deref()
    }

    pub fn current_range(&self) -> CheckedTableRange {
        CheckedTableRange::new(0, self.currents.len() as u64)
    }

    pub fn contribution_range(&self) -> CheckedTableRange {
        CheckedTableRange::new(0, self.contributions.len() as u64)
    }

    pub fn closure_term_range(&self) -> CheckedTableRange {
        CheckedTableRange::new(0, self.closure_terms.len() as u64)
    }

    pub fn closure_range_for_destination(&self, destination_id: u32) -> Option<CheckedTableRange> {
        self.amplitude_destinations
            .get(destination_id as usize)
            .map(|destination| destination.closure_range())
    }

    pub fn validate(&self) -> RusticolResult<()> {
        let current_count = checked_table_len("recurrence current", self.currents.len())?;
        let color_state_count = checked_table_len(
            "recurrence dynamic LC color state",
            self.dynamic_color_states.len(),
        )?;
        let contribution_count =
            checked_table_len("recurrence contribution", self.contributions.len())?;
        let finalization_count =
            checked_table_len("recurrence finalization", self.finalizations.len())?;
        let replay_target_count =
            checked_table_len("recurrence replay target", self.replay_targets.len())?;
        let closure_count = checked_table_len("recurrence closure term", self.closure_terms.len())?;
        let helicity_count = checked_table_len(
            "recurrence resolved helicity",
            self.resolved_helicities.len(),
        )?;
        let destination_count = checked_table_len(
            "recurrence amplitude destination",
            self.amplitude_destinations.len(),
        )?;

        if self.physical_sector_count == 0 {
            return Err(invalid("recurrence requires physical LC sectors"));
        }
        if self.retained_helicity_count == 0 {
            return Err(invalid("recurrence retains no public helicities"));
        }

        u32::try_from(current_count)
            .map_err(|_| invalid("recurrence current count exceeds the u32 ID domain"))?;
        u32::try_from(color_state_count)
            .map_err(|_| invalid("recurrence dynamic color-state count exceeds u32"))?;
        u32::try_from(contribution_count)
            .map_err(|_| invalid("recurrence contribution count exceeds the u32 ID domain"))?;
        u32::try_from(finalization_count)
            .map_err(|_| invalid("recurrence finalization count exceeds the u32 ID domain"))?;
        u32::try_from(replay_target_count)
            .map_err(|_| invalid("recurrence replay-target count exceeds the u32 ID domain"))?;
        u32::try_from(closure_count)
            .map_err(|_| invalid("recurrence closure count exceeds the u32 ID domain"))?;
        u32::try_from(helicity_count)
            .map_err(|_| invalid("recurrence resolved-helicity count exceeds u32"))?;
        u32::try_from(destination_count)
            .map_err(|_| invalid("recurrence amplitude-destination count exceeds u32"))?;

        let mut next_contribution = 0u64;
        let mut fan_in_sum = 0u64;
        let mut non_source_count = 0u64;
        for (index, current) in self.currents.iter().enumerate() {
            let expected_id = index as u32;
            if current.id != expected_id {
                return Err(invalid(format!(
                    "recurrence current row {index} has non-dense id {}",
                    current.id
                )));
            }
            if current.key.helicity_identity().strategy() != self.strategy {
                return Err(invalid(format!(
                    "recurrence current {expected_id} uses strategy {} in a {} program",
                    current.key.helicity_identity().strategy(),
                    self.strategy
                )));
            }
            if u64::from(current.key.dynamic_lc_color_state_id().get()) >= color_state_count {
                return Err(invalid(format!(
                    "recurrence current {expected_id} references unknown dynamic LC color state {}",
                    current.key.dynamic_lc_color_state_id().get()
                )));
            }
            if current.contribution_range.start != next_contribution {
                return Err(invalid(format!(
                    "recurrence current {expected_id} contribution range starts at {}, expected packed offset {next_contribution}",
                    current.contribution_range.start
                )));
            }
            let range = current.contribution_range.as_usize_range(
                self.contributions.len(),
                &format!("recurrence current {expected_id} contribution"),
            )?;
            next_contribution = current
                .contribution_range
                .end("recurrence current contribution")?;
            fan_in_sum = fan_in_sum
                .checked_add(current.fan_in())
                .ok_or_else(|| invalid("recurrence fan-in sum exceeds u64"))?;

            if current.is_source() {
                if !range.is_empty() || current.finalization_id.is_some() {
                    return Err(invalid(format!(
                        "source current {expected_id} must not have contributions or finalization"
                    )));
                }
                match self.strategy {
                    RecurrenceStrategy::TopologyReplay
                    | RecurrenceStrategy::ContractedColorUnion
                        if current.source_exact_factor.is_none() =>
                    {
                        return Err(invalid(format!(
                            "{} source current {expected_id} requires an exact source factor",
                            self.strategy
                        )));
                    }
                    RecurrenceStrategy::AllFlowUnion if current.source_exact_factor.is_some() => {
                        return Err(invalid(format!(
                            "all-flow-union source current {expected_id} must take its factor from runtime dispatch"
                        )));
                    }
                    _ => {}
                }
            } else {
                if current.source_exact_factor.is_some() {
                    return Err(invalid(format!(
                        "non-source current {expected_id} cannot carry a source factor"
                    )));
                }
                non_source_count += 1;
                if range.is_empty() {
                    return Err(invalid(format!(
                        "non-source current {expected_id} requires at least one contribution"
                    )));
                }
                let finalization_id = current.finalization_id.ok_or_else(|| {
                    invalid(format!(
                        "non-source current {expected_id} requires exactly one finalization"
                    ))
                })?;
                let finalization = self
                    .finalizations
                    .get(finalization_id as usize)
                    .ok_or_else(|| {
                        invalid(format!(
                            "current {expected_id} references unknown finalization {finalization_id}"
                        ))
                    })?;
                if finalization.current_id != expected_id {
                    return Err(invalid(format!(
                        "current {expected_id} references finalization {finalization_id} owned by current {}",
                        finalization.current_id
                    )));
                }
            }

            for contribution in &self.contributions[range] {
                if contribution.result_current_id != expected_id {
                    return Err(invalid(format!(
                        "contribution {} is packed under current {expected_id} but targets current {}",
                        contribution.id, contribution.result_current_id
                    )));
                }
            }
        }
        if next_contribution != contribution_count || fan_in_sum != contribution_count {
            return Err(invalid(format!(
                "stored contribution count {contribution_count} does not equal packed fan-in sum {fan_in_sum}"
            )));
        }

        for (index, contribution) in self.contributions.iter().enumerate() {
            let expected_id = index as u32;
            if contribution.id != expected_id {
                return Err(invalid(format!(
                    "recurrence contribution row {index} has non-dense id {}",
                    contribution.id
                )));
            }
            let result = self
                .currents
                .get(contribution.result_current_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "contribution {expected_id} references unknown result current {}",
                        contribution.result_current_id
                    ))
                })?;
            if contribution.parent_current_ids.len()
                != contribution.key.parent_state_template_ids().len()
            {
                return Err(invalid(format!(
                    "contribution {expected_id} has {} parent current IDs but {} semantic parent states",
                    contribution.parent_current_ids.len(),
                    contribution.key.parent_state_template_ids().len()
                )));
            }
            if contribution.key.result_state_template_id() != result.key.current_state_template_id()
            {
                return Err(invalid(format!(
                    "contribution {expected_id} result-state template {} does not match current {} state template {}",
                    contribution.key.result_state_template_id(),
                    contribution.result_current_id,
                    result.key.current_state_template_id()
                )));
            }
            for (parent_ordinal, parent_id) in
                contribution.parent_current_ids.iter().copied().enumerate()
            {
                if parent_id >= contribution.result_current_id {
                    return Err(invalid(format!(
                        "contribution {expected_id} parent {parent_ordinal} current {parent_id} does not precede result current {}",
                        contribution.result_current_id
                    )));
                }
                let parent = &self.currents[parent_id as usize];
                let expected_state = contribution.key.parent_state_template_ids()[parent_ordinal];
                if parent.key.current_state_template_id() != expected_state {
                    return Err(invalid(format!(
                        "contribution {expected_id} parent {parent_ordinal} current {parent_id} has state template {}, expected {expected_state}",
                        parent.key.current_state_template_id()
                    )));
                }
            }
            if contribution.exact_factor.is_zero() {
                return Err(invalid(format!(
                    "recurrence contribution {expected_id} has zero exact factor"
                )));
            }
        }

        if finalization_count != non_source_count {
            return Err(invalid(format!(
                "recurrence has {finalization_count} finalizations for {non_source_count} non-source currents"
            )));
        }
        for (index, finalization) in self.finalizations.iter().enumerate() {
            let expected_id = index as u32;
            if finalization.id != expected_id {
                return Err(invalid(format!(
                    "recurrence finalization row {index} has non-dense id {}",
                    finalization.id
                )));
            }
            let current = self
                .currents
                .get(finalization.current_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "finalization {expected_id} references unknown current {}",
                        finalization.current_id
                    ))
                })?;
            if current.is_source() {
                return Err(invalid(format!(
                    "finalization {expected_id} references source current {}",
                    finalization.current_id
                )));
            }
            if current.finalization_id != Some(expected_id) {
                return Err(invalid(format!(
                    "finalization {expected_id} is not the unique finalization recorded by current {}",
                    finalization.current_id
                )));
            }
            if finalization.propagator_template_id != current.key.propagator_template_id() {
                return Err(invalid(format!(
                    "finalization {expected_id} propagator template {:?} does not match current {} template {:?}",
                    finalization.propagator_template_id,
                    finalization.current_id,
                    current.key.propagator_template_id()
                )));
            }
            if finalization.exact_factor.is_zero() {
                return Err(invalid(format!(
                    "recurrence finalization {expected_id} has zero exact factor"
                )));
            }
        }

        let source_slot_count = self
            .currents
            .iter()
            .flat_map(|current| current.key.support_source_slots().iter().copied())
            .max()
            .map_or(0usize, |slot| slot as usize + 1);
        let mut replayed_sectors = BTreeSet::new();
        let mut materialized_sectors = BTreeSet::new();
        for (index, target) in self.replay_targets.iter().enumerate() {
            let expected_id = index as u32;
            if target.id != expected_id {
                return Err(invalid(format!(
                    "recurrence replay-target row {index} has non-dense id {}",
                    target.id
                )));
            }
            if target.materialized_sector_id >= self.physical_sector_count
                || target.target_sector_id >= self.physical_sector_count
            {
                return Err(invalid(format!(
                    "recurrence replay target {expected_id} references an unknown physical sector"
                )));
            }
            if !replayed_sectors.insert(target.target_sector_id) {
                return Err(invalid(format!(
                    "physical sector {} has multiple recurrence replay targets",
                    target.target_sector_id
                )));
            }
            materialized_sectors.insert(target.materialized_sector_id);
            if target.source_slot_permutation.len() != source_slot_count {
                return Err(invalid(format!(
                    "recurrence replay target {expected_id} source permutation has length {}, expected {source_slot_count}",
                    target.source_slot_permutation.len()
                )));
            }
            if target.source_momentum_signs.len() != source_slot_count
                || target
                    .source_momentum_signs
                    .iter()
                    .any(|sign| !matches!(sign, -1 | 1))
            {
                return Err(invalid(format!(
                    "recurrence replay target {expected_id} has an invalid momentum-sign mapping"
                )));
            }
            let mut permutation = target.source_slot_permutation.to_vec();
            permutation.sort_unstable();
            if permutation
                .iter()
                .copied()
                .enumerate()
                .any(|(slot, value)| value as usize != slot)
            {
                return Err(invalid(format!(
                    "recurrence replay target {expected_id} source mapping is not a permutation"
                )));
            }
        }

        match self.strategy {
            RecurrenceStrategy::TopologyReplay if self.replay_targets.is_empty() => {
                return Err(invalid(
                    "topology-replay recurrence requires replay targets",
                ));
            }
            RecurrenceStrategy::AllFlowUnion if !self.replay_targets.is_empty() => {
                return Err(invalid(
                    "all-flow-union recurrence must not carry topology-replay targets",
                ));
            }
            _ => {}
        }
        if self.strategy.uses_topology_replay_targets()
            && !self.replay_targets.is_empty()
            && replayed_sectors.len() != self.physical_sector_count as usize
        {
            return Err(invalid(format!(
                "{} recurrence replay covers {} of {} physical sectors",
                self.strategy,
                replayed_sectors.len(),
                self.physical_sector_count
            )));
        }
        let mut destination_sectors = BTreeSet::new();
        for destination in &self.amplitude_destinations {
            destination_sectors.insert(destination.target_sector_id);
            if self.strategy.uses_topology_replay_targets()
                && !self.replay_targets.is_empty()
                && !materialized_sectors.contains(&destination.target_sector_id)
            {
                return Err(invalid(format!(
                    "{} amplitude destination {} targets non-materialized sector {}",
                    self.strategy, destination.id, destination.target_sector_id
                )));
            }
        }
        for sector in materialized_sectors {
            if !destination_sectors.contains(&sector) {
                return Err(invalid(format!(
                    "recurrence replay materialized sector {sector} has no amplitude destination"
                )));
            }
        }

        match self.strategy {
            RecurrenceStrategy::TopologyReplay if self.resolved_helicities.is_empty() => {
                return Err(invalid(
                    "topology-replay recurrence requires resolved-helicity destinations",
                ));
            }
            RecurrenceStrategy::AllFlowUnion if !self.resolved_helicities.is_empty() => {
                return Err(invalid(
                    "all-flow-union recurrence must select source helicity at runtime",
                ));
            }
            RecurrenceStrategy::ContractedColorUnion if self.resolved_helicities.is_empty() => {
                return Err(invalid(
                    "contracted-color recurrence requires resolved-helicity destinations",
                ));
            }
            _ => {}
        }
        if helicity_count > self.retained_helicity_count {
            return Err(invalid(format!(
                "recurrence has {helicity_count} active helicities but retains only {} public assignments",
                self.retained_helicity_count
            )));
        }
        for (index, helicity) in self.resolved_helicities.iter().enumerate() {
            if helicity.id != index as u32 {
                return Err(invalid(format!(
                    "resolved-helicity row {index} has non-dense id {}",
                    helicity.id
                )));
            }
            if helicity.source_states.len() != helicity.public_helicities.len() {
                return Err(invalid(format!(
                    "resolved helicity {index} has inconsistent source-state and public-helicity dimensions"
                )));
            }
            for (source_slot, assignment) in helicity.source_states.iter().copied().enumerate() {
                if assignment.source_slot() as usize != source_slot {
                    return Err(invalid(format!(
                        "resolved helicity {index} does not cover source slot {source_slot} canonically"
                    )));
                }
            }
        }

        let mut next_closure = 0u64;
        for (destination_index, destination) in
            self.amplitude_destinations.iter().copied().enumerate()
        {
            let destination_id = destination_index as u32;
            if destination.id != destination_id {
                return Err(invalid(format!(
                    "amplitude destination row {destination_index} has non-dense id {}",
                    destination.id
                )));
            }
            if destination.target_sector_id >= self.physical_sector_count {
                return Err(invalid(format!(
                    "amplitude destination {destination_id} references unknown physical sector {}",
                    destination.target_sector_id
                )));
            }
            match (self.strategy, destination.target_helicity_id) {
                (RecurrenceStrategy::TopologyReplay, Some(helicity_id))
                    if u64::from(helicity_id) < helicity_count => {}
                (RecurrenceStrategy::TopologyReplay, Some(helicity_id)) => {
                    return Err(invalid(format!(
                        "amplitude destination {destination_id} references unknown resolved helicity {helicity_id}"
                    )));
                }
                (RecurrenceStrategy::TopologyReplay, None) => {
                    return Err(invalid(format!(
                        "topology-replay amplitude destination {destination_id} lacks a helicity"
                    )));
                }
                (RecurrenceStrategy::AllFlowUnion, None) => {}
                (RecurrenceStrategy::AllFlowUnion, Some(_)) => {
                    return Err(invalid(format!(
                        "all-flow-union amplitude destination {destination_id} fixes a helicity"
                    )));
                }
                (RecurrenceStrategy::ContractedColorUnion, Some(helicity_id))
                    if u64::from(helicity_id) < helicity_count => {}
                (RecurrenceStrategy::ContractedColorUnion, Some(helicity_id)) => {
                    return Err(invalid(format!(
                        "contracted-color amplitude destination {destination_id} references unknown resolved helicity {helicity_id}"
                    )));
                }
                (RecurrenceStrategy::ContractedColorUnion, None) => {
                    return Err(invalid(format!(
                        "contracted-color amplitude destination {destination_id} lacks a helicity"
                    )));
                }
            }
            let range = destination.closure_range;
            if range.start != next_closure {
                return Err(invalid(format!(
                    "amplitude destination {destination_id} closure range starts at {}, expected packed offset {next_closure}",
                    range.start
                )));
            }
            let term_range = range.as_usize_range(
                self.closure_terms.len(),
                &format!("amplitude destination {destination_id} closure"),
            )?;
            next_closure = range.end("amplitude-destination closure")?;
            for term in &self.closure_terms[term_range] {
                if term.target_destination_id != destination_id {
                    return Err(invalid(format!(
                        "closure term {} is packed under destination {destination_id} but points to destination {}",
                        term.id, term.target_destination_id
                    )));
                }
            }
        }
        if next_closure != closure_count {
            return Err(invalid(format!(
                "packed amplitude-destination ranges cover {next_closure} of {closure_count} terms"
            )));
        }

        for (index, term) in self.closure_terms.iter().enumerate() {
            let expected_id = index as u32;
            if term.id != expected_id {
                return Err(invalid(format!(
                    "recurrence closure row {index} has non-dense id {}",
                    term.id
                )));
            }
            let destination = self
                .amplitude_destinations
                .get(term.target_destination_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "closure term {expected_id} references unknown amplitude destination {}",
                        term.target_destination_id
                    ))
                })?;
            if term.exact_factor.is_zero() {
                return Err(invalid(format!(
                    "recurrence closure term {expected_id} has zero exact factor"
                )));
            }
            for parent_id in term.parent_current_ids.iter().copied() {
                if u64::from(parent_id) >= current_count {
                    return Err(invalid(format!(
                        "closure term {expected_id} references unknown parent current {parent_id}"
                    )));
                }
            }
            if let Some(helicity_id) = destination.target_helicity_id {
                let expected = self.resolved_helicities[helicity_id as usize].source_states();
                let actual = closure_parent_source_states(term, &self.currents)?;
                if actual != expected {
                    return Err(invalid(format!(
                        "closure term {expected_id} ancestry does not match amplitude destination {}",
                        destination.id
                    )));
                }
            }
        }

        validate_program_closure_proofs(self)?;

        Ok(())
    }
}

fn closure_proofs_from_materialized_terms(
    currents: &[RecurrenceCurrent],
    amplitude_destinations: &[RecurrenceAmplitudeDestination],
    closure_terms: &[RecurrenceClosureTerm],
) -> RusticolResult<ClosureProofMetadataV2> {
    let mut contributions = Vec::with_capacity(closure_terms.len());
    let mut groups = Vec::with_capacity(closure_terms.len());
    for term in closure_terms {
        let destination = amplitude_destinations
            .get(term.target_destination_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "closure term {} references absent destination {}",
                    term.id, term.target_destination_id
                ))
            })?;
        let parent_count = term.parent_current_ids.len();
        let identity_permutation = (0..parent_count)
            .map(|index| {
                u32::try_from(index).map_err(|_| invalid("closure proof parent index exceeds u32"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut parent_semantic_digests = Vec::with_capacity(parent_count);
        let mut parent_color_digests = Vec::with_capacity(parent_count);
        for parent_id in term.parent_current_ids.iter().copied() {
            let parent = currents.get(parent_id as usize).ok_or_else(|| {
                invalid(format!(
                    "closure term {} references absent parent {parent_id}",
                    term.id
                ))
            })?;
            parent_semantic_digests.push(parent.key.catalog_digest());
            parent_color_digests.push(semantic_digest_from_fields(
                b"recurrence-closure-parent-color-v2\0",
                [parent_id, parent.key.dynamic_lc_color_state_id().get()],
            )?);
        }
        let contribution_id = u32::try_from(contributions.len())
            .map_err(|_| invalid("closure proof contribution count exceeds u32"))?;
        contributions.push(ClosureProofContributionV2::new(
            contribution_id,
            destination.target_sector_id,
            Some(destination.id),
            destination.target_helicity_id,
            term.closure_template_id,
            semantic_digest_from_fields(
                b"recurrence-closure-template-v2\0",
                [term.closure_template_id],
            )?,
            term.quantum_flow_template_id,
            term.parent_current_ids.to_vec(),
            term.parent_current_ids.iter().copied().map(Some).collect(),
            parent_semantic_digests,
            parent_color_digests,
            identity_permutation.clone(),
            identity_permutation.clone(),
            identity_permutation,
            u32::MAX,
            semantic_digest_from_fields(
                b"recurrence-materialized-closure-witness-v2\0",
                [term.id],
            )?,
            None,
            vec![],
            None,
            term.exact_factor,
            1,
        )?);
        groups.push(ClosureExecutionProofGroupV2::new(
            term.id,
            Some(term.id),
            None,
            CheckedTableRange::new(u64::from(contribution_id), 1),
            term.exact_factor,
            semantic_digest_from_fields(
                b"recurrence-materialized-component-factor-v2\0",
                [term.closure_template_id],
            )?,
            semantic_digest_from_fields(
                b"recurrence-materialized-selector-domain-v2\0",
                [destination.id],
            )?,
        )?);
    }
    ClosureProofMetadataV2::new(contributions, groups, Vec::new())
}

fn validate_program_closure_proofs(program: &RecurrenceProgram) -> RusticolResult<()> {
    validate_closure_proof_tables(
        program.closure_proofs.contributions(),
        program.closure_proofs.groups(),
        program.closure_proofs.reflection_certificates(),
        program.closure_proofs.three_line_traversal_certificates(),
    )?;
    if closure_proof_semantic_completeness_digest_with_three_line_v2(
        program.closure_proofs.contributions(),
        program.closure_proofs.groups(),
        program.closure_proofs.reflection_certificates(),
        program.closure_proofs.three_line_traversal_certificates(),
    )? != program
        .closure_proofs
        .expected_semantic_completeness_digest()
    {
        return Err(invalid(
            "recurrence closure proof semantic completeness digest mismatch",
        ));
    }

    for (index, contribution) in program.closure_proofs.contributions().iter().enumerate() {
        if let Some(destination_id) = contribution.target_destination_id {
            let destination = program
                .amplitude_destinations
                .get(destination_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "closure proof contribution {index} references absent destination {destination_id}"
                    ))
                })?;
            if contribution.target_sector_id != destination.target_sector_id
                || contribution.target_helicity_id != destination.target_helicity_id
            {
                return Err(invalid(format!(
                    "closure proof contribution {index} target does not match destination {}",
                    destination.id
                )));
            }
        }
        for parent_id in contribution
            .construction_parent_runtime_ids
            .iter()
            .copied()
            .flatten()
        {
            if parent_id as usize >= program.currents.len() {
                return Err(invalid(format!(
                    "closure proof contribution {index} references absent runtime parent current {parent_id}"
                )));
            }
        }
    }

    let mut term_owner = vec![None; program.closure_terms.len()];
    for group in program.closure_proofs.groups() {
        if group.emitted_direct_closure_row_id.is_some() {
            return Err(invalid(format!(
                "program closure proof group {} must not contain a lowered direct row ID",
                group.id
            )));
        }
        let Some(term_id) = group.emitted_runtime_closure_term_id else {
            continue;
        };
        let term = program.closure_terms.get(term_id as usize).ok_or_else(|| {
            invalid(format!(
                "closure proof group {} references absent runtime closure term {term_id}",
                group.id
            ))
        })?;
        if term_owner[term_id as usize].replace(group.id).is_some() {
            return Err(invalid(format!(
                "runtime closure term {term_id} is owned by multiple proof groups"
            )));
        }
        if term.exact_factor != group.exact_summed_factor {
            return Err(invalid(format!(
                "closure proof group {} factor does not match runtime closure term {term_id}",
                group.id
            )));
        }
        let range = group.contribution_range.as_usize_range(
            program.closure_proofs.contributions.len(),
            &format!("closure proof group {} contribution", group.id),
        )?;
        for contribution in &program.closure_proofs.contributions[range] {
            if contribution.target_destination_id != Some(term.target_destination_id)
                || contribution.closure_template_id != term.closure_template_id
                || contribution.quantum_flow_template_id != term.quantum_flow_template_id
                || contribution.evaluator_parent_runtime_ids().as_deref()
                    != Some(term.parent_current_ids.as_ref())
            {
                return Err(invalid(format!(
                    "closure proof group {} does not describe runtime closure term {term_id}",
                    group.id
                )));
            }
        }
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

fn closure_parent_source_states(
    term: &RecurrenceClosureTerm,
    currents: &[RecurrenceCurrent],
) -> RusticolResult<Vec<SourceStateAssignment>> {
    let mut states = Vec::new();
    for parent_id in term.parent_current_ids.iter().copied() {
        let parent = currents
            .get(parent_id as usize)
            .ok_or_else(|| invalid("closure parent current is absent"))?;
        states.extend_from_slice(parent.key.helicity_identity().local_source_states());
    }
    states.sort_unstable();
    if states
        .windows(2)
        .any(|pair| pair[0].source_slot() == pair[1].source_slot())
    {
        return Err(invalid("closure parent helicity ancestries overlap"));
    }
    Ok(states)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{
        CanonicalMomentumLinearForm, CurrentHelicityIdentity, CurrentSourceBinding,
        DynamicLCColorStateId, LCColorWitnessTermId, MomentumTerm, SemanticDigest,
        SourceStateAssignment,
    };

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn momentum() -> CanonicalMomentumLinearForm {
        CanonicalMomentumLinearForm::new(vec![MomentumTerm {
            source_slot: 0,
            coefficient: 1,
        }])
        .unwrap()
    }

    fn source_key() -> CurrentCoreKey {
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Source,
            0,
            DynamicLCColorStateId::from_interner(0),
            vec![0],
            momentum(),
            CurrentHelicityIdentity::topology_replay(-1, vec![SourceStateAssignment::new(0, 0)])
                .unwrap(),
            vec![],
            0,
            vec![],
            CurrentSourceBinding::FixedTemplate(0),
            None,
        )
        .unwrap()
    }

    fn propagated_key() -> CurrentCoreKey {
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Current,
            1,
            DynamicLCColorStateId::from_interner(0),
            vec![0],
            momentum(),
            CurrentHelicityIdentity::topology_replay(-1, vec![SourceStateAssignment::new(0, 0)])
                .unwrap(),
            vec![],
            0,
            vec![],
            CurrentSourceBinding::None,
            Some(0),
        )
        .unwrap()
    }

    fn contribution(parent_id: u32) -> RecurrenceContribution {
        RecurrenceContribution::new(
            0,
            1,
            vec![parent_id],
            ContributionKey::new(
                0,
                vec![0],
                vec![0],
                vec![momentum()],
                1,
                0,
                LCColorWitnessTermId::new(0, 0),
                digest(2),
                0,
            )
            .unwrap(),
            ExactComplexRational::ONE,
        )
        .unwrap()
    }

    fn resolved_helicity() -> RecurrenceResolvedHelicity {
        RecurrenceResolvedHelicity::new(0, vec![SourceStateAssignment::new(0, 0)], vec![-1])
            .unwrap()
    }

    fn identity_replay_target() -> RecurrenceReplayTarget {
        RecurrenceReplayTarget::new(0, 0, 0, vec![0], vec![1], ExactComplexRational::ONE).unwrap()
    }

    fn minus_one() -> ExactComplexRational {
        ExactComplexRational::new(ExactRational::new(-1, 1).unwrap(), ExactRational::ZERO)
    }

    fn proof_contribution(
        id: u32,
        parent_id: u32,
        witness_id: u32,
        witness_digest: SemanticDigest,
        reflection_certificate_id: Option<u32>,
        exact_factor: ExactComplexRational,
    ) -> ClosureProofContributionV2 {
        proof_contribution_with_runtime_binding(
            id,
            parent_id,
            witness_id,
            witness_digest,
            reflection_certificate_id,
            exact_factor,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn proof_contribution_with_runtime_binding(
        id: u32,
        parent_id: u32,
        witness_id: u32,
        witness_digest: SemanticDigest,
        reflection_certificate_id: Option<u32>,
        exact_factor: ExactComplexRational,
        has_runtime_binding: bool,
    ) -> ClosureProofContributionV2 {
        ClosureProofContributionV2::new(
            id,
            0,
            has_runtime_binding.then_some(0),
            Some(0),
            0,
            digest(10),
            None,
            vec![parent_id],
            vec![has_runtime_binding.then_some(parent_id)],
            vec![digest(11)],
            vec![digest(12)],
            vec![0],
            vec![0],
            vec![0],
            witness_id,
            witness_digest,
            None,
            vec![],
            reflection_certificate_id,
            exact_factor,
            1,
        )
        .unwrap()
    }

    fn reflection_certificate(proof_digest: SemanticDigest) -> ReflectionCertificateV1 {
        ReflectionCertificateV1::new(
            0,
            digest(20),
            digest(21),
            vec![0],
            minus_one(),
            false,
            2,
            1,
            proof_digest,
        )
        .unwrap()
    }

    fn three_line_certificate(
        id: u32,
        sector_id: u32,
        pairing_rule_id: u32,
    ) -> ThreeLineTraversalCertificateV1 {
        let reference_block_order = vec![0, 1, 2];
        let witness_block_order = vec![1, 0, 2];
        let block_permutation = vec![1, 0, 2];
        let reference_source_order = vec![10, 11, 12];
        let witness_source_order = vec![11, 10, 12];
        let source_position_permutation = vec![1, 0, 2];
        let proof_digest = three_line_traversal_proof_digest_v1(
            sector_id,
            ThreeLineTraversalKindV1::Partner,
            2,
            &reference_block_order,
            &witness_block_order,
            &block_permutation,
            &reference_source_order,
            &witness_source_order,
            &source_position_permutation,
            12,
            pairing_rule_id,
        )
        .unwrap();
        ThreeLineTraversalCertificateV1::new(
            id,
            sector_id,
            ThreeLineTraversalKindV1::Partner,
            2,
            reference_block_order,
            witness_block_order,
            block_permutation,
            reference_source_order,
            witness_source_order,
            source_position_permutation,
            12,
            pairing_rule_id,
            proof_digest,
        )
        .unwrap()
    }

    fn three_line_proof_contribution(
        certificate_id: u32,
        sector_id: u32,
    ) -> ClosureProofContributionV2 {
        ClosureProofContributionV2::new(
            0,
            sector_id,
            Some(0),
            Some(0),
            0,
            digest(10),
            None,
            vec![1],
            vec![Some(1)],
            vec![digest(11)],
            vec![digest(12)],
            vec![0],
            vec![0],
            vec![0],
            0,
            digest(13),
            Some(certificate_id),
            vec![],
            None,
            ExactComplexRational::ONE,
            1,
        )
        .unwrap()
    }

    fn valid_program() -> RecurrenceProgram {
        RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
            1,
            1,
            vec![DynamicLCColorState::new(0, None, vec![]).unwrap()],
            vec![
                RecurrenceCurrent::new(
                    0,
                    source_key(),
                    Some(ExactComplexRational::ONE),
                    CheckedTableRange::new(0, 0),
                    None,
                )
                .unwrap(),
                RecurrenceCurrent::new(
                    1,
                    propagated_key(),
                    None,
                    CheckedTableRange::new(0, 1),
                    Some(0),
                )
                .unwrap(),
            ],
            vec![contribution(0)],
            vec![RecurrenceFinalization::new(0, 1, Some(0), ExactComplexRational::ONE).unwrap()],
            vec![identity_replay_target()],
            vec![resolved_helicity()],
            vec![
                RecurrenceAmplitudeDestination::new(0, 0, Some(0), CheckedTableRange::new(0, 1))
                    .unwrap(),
            ],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, None, vec![1], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn validates_one_compact_recurrence_program() {
        let program = valid_program();
        assert_eq!(program.current_range(), CheckedTableRange::new(0, 2));
        assert_eq!(program.contribution_range(), CheckedTableRange::new(0, 1));
        assert_eq!(program.closure_term_range(), CheckedTableRange::new(0, 1));
        assert_eq!(
            program.closure_range_for_destination(0),
            Some(CheckedTableRange::new(0, 1))
        );
        assert!(program.validate().is_ok());
    }

    #[test]
    fn rejects_a_parent_that_does_not_precede_its_result() {
        let error = RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
            1,
            1,
            vec![DynamicLCColorState::new(0, None, vec![]).unwrap()],
            vec![
                RecurrenceCurrent::new(
                    0,
                    source_key(),
                    Some(ExactComplexRational::ONE),
                    CheckedTableRange::new(0, 0),
                    None,
                )
                .unwrap(),
                RecurrenceCurrent::new(
                    1,
                    propagated_key(),
                    None,
                    CheckedTableRange::new(0, 1),
                    Some(0),
                )
                .unwrap(),
            ],
            vec![contribution(1)],
            vec![RecurrenceFinalization::new(0, 1, Some(0), ExactComplexRational::ONE).unwrap()],
            vec![identity_replay_target()],
            vec![resolved_helicity()],
            vec![
                RecurrenceAmplitudeDestination::new(0, 0, Some(0), CheckedTableRange::new(0, 1))
                    .unwrap(),
            ],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, None, vec![1], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap_err();
        assert!(error.message().contains("does not precede result current"));
    }

    #[test]
    fn rejects_a_closure_destination_with_mismatched_source_ancestry() {
        let error = RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
            1,
            2,
            vec![DynamicLCColorState::new(0, None, vec![]).unwrap()],
            vec![
                RecurrenceCurrent::new(
                    0,
                    source_key(),
                    Some(ExactComplexRational::ONE),
                    CheckedTableRange::new(0, 0),
                    None,
                )
                .unwrap(),
                RecurrenceCurrent::new(
                    1,
                    propagated_key(),
                    None,
                    CheckedTableRange::new(0, 1),
                    Some(0),
                )
                .unwrap(),
            ],
            vec![contribution(0)],
            vec![RecurrenceFinalization::new(0, 1, Some(0), ExactComplexRational::ONE).unwrap()],
            vec![identity_replay_target()],
            vec![
                RecurrenceResolvedHelicity::new(0, vec![SourceStateAssignment::new(0, 1)], vec![1])
                    .unwrap(),
            ],
            vec![
                RecurrenceAmplitudeDestination::new(0, 0, Some(0), CheckedTableRange::new(0, 1))
                    .unwrap(),
            ],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, None, vec![1], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap_err();
        assert!(error.message().contains("ancestry does not match"));
    }

    #[test]
    fn closure_proof_tables_reject_missing_contributions_and_wrong_group_sums() {
        let group = ClosureExecutionProofGroupV2::new(
            0,
            Some(0),
            None,
            CheckedTableRange::new(0, 1),
            ExactComplexRational::ONE,
            digest(30),
            digest(31),
        )
        .unwrap();
        let missing = ClosureProofMetadataV2::new(vec![], vec![group.clone()], vec![]).unwrap_err();
        assert!(missing.message().contains("exceeds table length"));

        let wrong_sum = ClosureExecutionProofGroupV2::new(
            0,
            Some(0),
            None,
            CheckedTableRange::new(0, 1),
            minus_one(),
            digest(30),
            digest(31),
        )
        .unwrap();
        let error = ClosureProofMetadataV2::new(
            vec![proof_contribution(
                0,
                1,
                0,
                digest(13),
                None,
                ExactComplexRational::ONE,
            )],
            vec![wrong_sum],
            vec![],
        )
        .unwrap_err();
        assert!(error.message().contains("does not equal"));
    }

    #[test]
    fn closure_proof_tables_reject_a_stale_candidate_domain_certificate() {
        let contribution = proof_contribution(0, 1, 0, digest(13), None, ExactComplexRational::ONE);
        let group = ClosureExecutionProofGroupV2::new(
            0,
            Some(0),
            None,
            CheckedTableRange::new(0, 1),
            ExactComplexRational::ONE,
            digest(30),
            digest(31),
        )
        .unwrap();
        let metadata =
            ClosureProofMetadataV2::new(vec![contribution.clone()], vec![group.clone()], vec![])
                .unwrap();
        let certificate = metadata.candidate_domain_certificate();
        let stale = ClosureCandidateDomainCertificateV1::new(
            certificate.accepted_candidate_count() + 1,
            certificate.accepted_candidate_digest(),
        );
        let error = ClosureProofMetadataV2::new_with_three_line_certificates_and_candidate_domain(
            vec![contribution],
            vec![group],
            vec![],
            vec![],
            stale,
        )
        .unwrap_err();
        assert!(error.message().contains("candidate-domain certificate"));
    }

    #[test]
    fn closure_proof_digest_rejects_parent_witness_proof_and_certificate_mutations() {
        let group = ClosureExecutionProofGroupV2::new(
            0,
            Some(0),
            None,
            CheckedTableRange::new(0, 1),
            ExactComplexRational::ONE,
            digest(30),
            digest(31),
        )
        .unwrap();
        let original = ClosureProofMetadataV2::new(
            vec![proof_contribution(
                0,
                1,
                7,
                digest(13),
                Some(0),
                ExactComplexRational::ONE,
            )],
            vec![group.clone()],
            vec![reflection_certificate(digest(22))],
        )
        .unwrap();
        let expected = original.expected_semantic_completeness_digest();

        for contribution in [
            proof_contribution(0, 2, 7, digest(13), Some(0), ExactComplexRational::ONE),
            proof_contribution(0, 1, 8, digest(13), Some(0), ExactComplexRational::ONE),
            proof_contribution(0, 1, 7, digest(14), Some(0), ExactComplexRational::ONE),
        ] {
            let error = ClosureProofMetadataV2::new_with_expected_digest(
                vec![contribution],
                vec![group.clone()],
                vec![reflection_certificate(digest(22))],
                expected,
            )
            .unwrap_err();
            assert!(error.message().contains("digest mismatch"));
        }

        let error = ClosureProofMetadataV2::new_with_expected_digest(
            original.contributions().to_vec(),
            original.groups().to_vec(),
            vec![reflection_certificate(digest(23))],
            expected,
        )
        .unwrap_err();
        assert!(error.message().contains("digest mismatch"));
    }

    #[test]
    fn closure_proof_tables_retain_exact_zero_groups_without_runtime_rows() {
        let metadata = ClosureProofMetadataV2::new(
            vec![
                proof_contribution_with_runtime_binding(
                    0,
                    91,
                    0,
                    digest(13),
                    None,
                    ExactComplexRational::ONE,
                    false,
                ),
                proof_contribution_with_runtime_binding(
                    1,
                    91,
                    0,
                    digest(13),
                    None,
                    minus_one(),
                    false,
                ),
            ],
            vec![
                ClosureExecutionProofGroupV2::new(
                    0,
                    None,
                    None,
                    CheckedTableRange::new(0, 2),
                    ExactComplexRational::ZERO,
                    digest(30),
                    digest(31),
                )
                .unwrap(),
            ],
            vec![],
        )
        .unwrap();
        assert_eq!(metadata.groups().len(), 1);
        assert!(metadata.groups()[0].exact_summed_factor().is_zero());
        assert_eq!(metadata.groups()[0].emitted_runtime_closure_term_id(), None);
        assert_eq!(metadata.groups()[0].emitted_direct_closure_row_id(), None);
    }

    #[test]
    fn closure_proof_tables_reject_nonzero_groups_without_runtime_bindings() {
        let contribution = proof_contribution_with_runtime_binding(
            0,
            91,
            0,
            digest(13),
            None,
            ExactComplexRational::ONE,
            false,
        );
        let group = ClosureExecutionProofGroupV2::new(
            0,
            Some(0),
            None,
            CheckedTableRange::new(0, 1),
            ExactComplexRational::ONE,
            digest(30),
            digest(31),
        )
        .unwrap();
        let error =
            ClosureProofMetadataV2::new(vec![contribution], vec![group], vec![]).unwrap_err();
        assert!(error.message().contains("lacks a complete runtime binding"));
    }

    #[test]
    fn three_line_traversal_certificate_authenticates_orders_anchor_and_kind() {
        let certificate = three_line_certificate(0, 4, 7);
        assert_eq!(certificate.kind(), ThreeLineTraversalKindV1::Partner);
        assert_eq!(certificate.sink_block_ordinal(), 2);
        assert_eq!(certificate.closure_anchor_source_slot(), 12);
        assert!(ThreeLineTraversalKindV1::try_from(2).is_err());

        let error = ThreeLineTraversalCertificateV1::new(
            0,
            4,
            ThreeLineTraversalKindV1::Partner,
            2,
            vec![0, 2],
            vec![1, 0, 2],
            vec![1, 0, 2],
            vec![10, 11, 12],
            vec![11, 10, 12],
            vec![1, 0, 2],
            12,
            7,
            digest(60),
        )
        .unwrap_err();
        assert!(error.message().contains("expected 3"));

        let error = ThreeLineTraversalCertificateV1::new(
            0,
            4,
            ThreeLineTraversalKindV1::Partner,
            2,
            vec![0, 1, 2],
            vec![1, 0, 2],
            vec![1, 0, 2],
            vec![10, 11, 12],
            vec![11, 10, 12],
            vec![0, 1, 2],
            12,
            7,
            digest(60),
        )
        .unwrap_err();
        assert!(error.message().contains("source-position permutation"));

        let original = three_line_certificate(0, 4, 7);
        let error = ThreeLineTraversalCertificateV1::new(
            original.id(),
            original.sector_id(),
            original.kind(),
            original.sink_block_ordinal(),
            original.reference_block_order().to_vec(),
            original.witness_block_order().to_vec(),
            original.block_permutation().to_vec(),
            original.reference_source_order().to_vec(),
            original.witness_source_order().to_vec(),
            original.source_position_permutation().to_vec(),
            original.closure_anchor_source_slot(),
            original.pairing_rule_id(),
            digest(61),
        )
        .unwrap_err();
        assert!(error.message().contains("proof digest mismatch"));
    }

    #[test]
    fn closure_proofs_authenticate_three_line_catalog_and_references() {
        let group = ClosureExecutionProofGroupV2::new(
            0,
            Some(0),
            None,
            CheckedTableRange::new(0, 1),
            ExactComplexRational::ONE,
            digest(30),
            digest(31),
        )
        .unwrap();
        let original = ClosureProofMetadataV2::new_with_three_line_certificates(
            vec![three_line_proof_contribution(0, 4)],
            vec![group.clone()],
            vec![],
            vec![three_line_certificate(0, 4, 7)],
        )
        .unwrap();
        let expected = original.expected_semantic_completeness_digest();

        let error = ClosureProofMetadataV2::new_with_three_line_certificates(
            original.contributions().to_vec(),
            original.groups().to_vec(),
            vec![],
            vec![],
        )
        .unwrap_err();
        assert!(error.message().contains("absent three-line"));

        let error = ClosureProofMetadataV2::new_with_three_line_certificates(
            vec![three_line_proof_contribution(1, 4)],
            vec![group.clone()],
            vec![],
            vec![three_line_certificate(1, 4, 7)],
        )
        .unwrap_err();
        assert!(error.message().contains("non-dense id"));

        let error = ClosureProofMetadataV2::new_with_three_line_certificates(
            vec![three_line_proof_contribution(0, 5)],
            vec![group.clone()],
            vec![],
            vec![three_line_certificate(0, 4, 7)],
        )
        .unwrap_err();
        assert!(error.message().contains("does not match"));

        let error = ClosureProofMetadataV2::new_with_three_line_certificates_and_expected_digest(
            original.contributions().to_vec(),
            original.groups().to_vec(),
            vec![],
            vec![three_line_certificate(0, 4, 8)],
            expected,
        )
        .unwrap_err();
        assert!(error.message().contains("digest mismatch"));
    }
}
