// SPDX-License-Identifier: 0BSD

//! Deterministic little-endian codec for recurrence direct-plan v2.

use super::direct_plan::{
    DirectAmplitudeDestinationDescriptor, DirectClosureRow, DirectContributionRow,
    DirectCurrentDescriptor, DirectDestinationOperation, DirectExecutorRole, DirectFinalizationRow,
    DirectMomentumFormDescriptor, DirectMomentumTerm, DirectNodeKind, DirectRecurrencePlan,
    DirectRecurrencePlanParts, DirectReplayTargetDescriptor, DirectResolvedHelicityDescriptor,
    DirectResolvedSourceSelection, DirectRowGroupDescriptor, DirectSelectorDomainDescriptor,
    DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow, DirectSourceProjectionRow,
    DirectSourceRow, DirectSourceStateAssignment,
};
use super::{
    CheckedTableRange, ClosureCandidateDomainCertificateV1, ClosureExecutionProofGroupV2,
    ClosureProofContributionV2, ClosureProofMetadataV2, ExactComplexRational, ExactRational,
    RecurrenceStrategy, ReflectionCertificateV1, SemanticDigest, ThreeLineTraversalCertificateV1,
    ThreeLineTraversalKindV1,
};
use crate::{RusticolError, RusticolResult};
use std::io::Write;

const MAGIC: &[u8; 8] = b"PACRDAP2";
const VERSION: u32 = 2;
const MAX_ROWS: u64 = u32::MAX as u64;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("recurrence direct-plan codec: {}", message.into()))
}

struct Writer<W> {
    destination: W,
    bytes_written: u64,
}

impl<W: Write> Writer<W> {
    fn new(destination: W) -> Self {
        Self {
            destination,
            bytes_written: 0,
        }
    }

    #[cfg(test)]
    fn with_bytes_written(destination: W, bytes_written: u64) -> Self {
        Self {
            destination,
            bytes_written,
        }
    }

    fn raw(&mut self, bytes: &[u8]) -> RusticolResult<()> {
        let byte_count =
            u64::try_from(bytes.len()).map_err(|_| invalid("payload write length exceeds u64"))?;
        let end = self
            .bytes_written
            .checked_add(byte_count)
            .ok_or_else(|| invalid("payload length overflows u64"))?;
        self.destination.write_all(bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "could not stream recurrence direct-plan payload: {error}"
            ))
        })?;
        self.bytes_written = end;
        Ok(())
    }

    fn u16(&mut self, value: u16) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn i32(&mut self, value: i32) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn i128(&mut self, value: i128) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn digest(&mut self, digest: SemanticDigest) -> RusticolResult<()> {
        self.raw(digest.as_bytes())
    }

    fn optional_u32(&mut self, value: Option<u32>) -> RusticolResult<()> {
        self.u32(value.unwrap_or(u32::MAX))
    }

    fn exact(&mut self, value: ExactComplexRational) -> RusticolResult<()> {
        for rational in [value.real(), value.imag()] {
            self.i128(rational.numerator())?;
            self.i128(rational.denominator())?;
        }
        Ok(())
    }

    fn u32_slice(&mut self, label: &str, values: &[u32]) -> RusticolResult<()> {
        self.count(label, values.len())?;
        for value in values {
            self.u32(*value)?;
        }
        Ok(())
    }

    fn optional_u32_slice(&mut self, label: &str, values: &[Option<u32>]) -> RusticolResult<()> {
        self.count(label, values.len())?;
        for value in values {
            self.optional_u32(*value)?;
        }
        Ok(())
    }

    fn digest_slice(&mut self, label: &str, values: &[SemanticDigest]) -> RusticolResult<()> {
        self.count(label, values.len())?;
        for value in values {
            self.digest(*value)?;
        }
        Ok(())
    }

    fn count(&mut self, label: &str, count: usize) -> RusticolResult<()> {
        let count =
            u64::try_from(count).map_err(|_| invalid(format!("{label} count exceeds u64")))?;
        if count > MAX_ROWS {
            return Err(invalid(format!("{label} count exceeds the u32 ID domain")));
        }
        self.u64(count)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> RusticolResult<Self> {
        Ok(Self { bytes, offset: 0 })
    }

    fn take(&mut self, count: usize, label: &str) -> RusticolResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| invalid(format!("{label} offset overflows usize")))?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(|| {
            invalid(format!(
                "truncated {label} at byte {}: need {count}, have {}",
                self.offset,
                self.bytes.len().saturating_sub(self.offset)
            ))
        })?;
        self.offset = end;
        Ok(bytes)
    }

    fn u16(&mut self, label: &str) -> RusticolResult<u16> {
        Ok(u16::from_le_bytes(
            self.take(2, label)?.try_into().expect("checked read"),
        ))
    }

    fn u32(&mut self, label: &str) -> RusticolResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4, label)?.try_into().expect("checked read"),
        ))
    }

    fn i32(&mut self, label: &str) -> RusticolResult<i32> {
        Ok(i32::from_le_bytes(
            self.take(4, label)?.try_into().expect("checked read"),
        ))
    }

    fn u64(&mut self, label: &str) -> RusticolResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8, label)?.try_into().expect("checked read"),
        ))
    }

    fn i128(&mut self, label: &str) -> RusticolResult<i128> {
        Ok(i128::from_le_bytes(
            self.take(16, label)?.try_into().expect("checked read"),
        ))
    }

    fn digest(&mut self, label: &str) -> RusticolResult<SemanticDigest> {
        let bytes: [u8; 32] = self.take(32, label)?.try_into().expect("checked read");
        SemanticDigest::new(bytes).map_err(|error| invalid(error.message()))
    }

    fn optional_u32(&mut self, label: &str) -> RusticolResult<Option<u32>> {
        let value = self.u32(label)?;
        Ok((value != u32::MAX).then_some(value))
    }

    fn exact(&mut self, label: &str) -> RusticolResult<ExactComplexRational> {
        Ok(ExactComplexRational::new(
            read_rational(self, &format!("{label} real"))?,
            read_rational(self, &format!("{label} imaginary"))?,
        ))
    }

    fn u32_vec(&mut self, label: &str) -> RusticolResult<Vec<u32>> {
        let count = self.count(label, 4)?;
        (0..count)
            .map(|_| self.u32(label))
            .collect::<RusticolResult<Vec<_>>>()
    }

    fn optional_u32_vec(&mut self, label: &str) -> RusticolResult<Vec<Option<u32>>> {
        let count = self.count(label, 4)?;
        (0..count)
            .map(|_| self.optional_u32(label))
            .collect::<RusticolResult<Vec<_>>>()
    }

    fn digest_vec(&mut self, label: &str) -> RusticolResult<Vec<SemanticDigest>> {
        let count = self.count(label, 32)?;
        (0..count)
            .map(|_| self.digest(label))
            .collect::<RusticolResult<Vec<_>>>()
    }

    fn count(&mut self, label: &str, row_bytes: usize) -> RusticolResult<usize> {
        let count = self.u64(&format!("{label} count"))?;
        if count > MAX_ROWS {
            return Err(invalid(format!("{label} count exceeds the u32 ID domain")));
        }
        let count =
            usize::try_from(count).map_err(|_| invalid(format!("{label} count exceeds usize")))?;
        if row_bytes != 0 && count > self.bytes.len().saturating_sub(self.offset) / row_bytes {
            return Err(invalid(format!(
                "{label} count cannot fit in the remaining payload"
            )));
        }
        Ok(count)
    }

    fn finish(self) -> RusticolResult<()> {
        if self.offset != self.bytes.len() {
            return Err(invalid(format!(
                "payload contains {} trailing bytes",
                self.bytes.len() - self.offset
            )));
        }
        Ok(())
    }
}

pub fn encode_recurrence_direct_plan_v2(plan: &DirectRecurrencePlan) -> RusticolResult<Vec<u8>> {
    let mut payload = Vec::new();
    encode_recurrence_direct_plan_v2_to_writer(plan, &mut payload)?;
    Ok(payload)
}

pub(crate) fn encode_recurrence_direct_plan_v2_to_writer<W: Write>(
    plan: &DirectRecurrencePlan,
    destination: W,
) -> RusticolResult<u64> {
    plan.ensure_serializable()?;
    let mut writer = Writer::new(destination);
    writer.raw(MAGIC)?;
    writer.u32(VERSION)?;
    writer.u32(0)?;
    writer.u32(plan.strategy().as_u32())?;
    writer.u32(plan.point_tile_size())?;
    writer.u32(plan.workspace_mib())?;
    writer.u32(plan.current_arena_components())?;
    writer.u32(plan.physical_sector_count())?;
    writer.u64(plan.retained_helicity_count())?;
    writer.u32(plan.amplitude_destination_count())?;
    writer.u32(plan.parameter_value_count())?;
    writer.u32(plan.external_source_count())?;
    writer.u32(plan.state_template_count())?;
    writer.u32(plan.source_template_count())?;
    writer.u32(plan.source_template_or_dispatch_count())?;
    writer.u32(plan.runtime_helicity_contract_count())?;
    writer.u32(plan.runtime_helicity_variant_count())?;
    writer.u32(plan.direct_executor_count())?;
    writer.digest(plan.semantic_digest())?;
    writer.digest(plan.prepared_pack_digest())?;
    writer.digest(plan.direct_template_catalog_digest())?;
    writer.digest(plan.runtime_layout_digest())?;
    writer.digest(
        plan.closure_proofs()
            .expected_semantic_completeness_digest(),
    )?;
    writer.u64(
        plan.closure_proofs()
            .candidate_domain_certificate()
            .accepted_candidate_count(),
    )?;
    writer.digest(
        plan.closure_proofs()
            .candidate_domain_certificate()
            .accepted_candidate_digest(),
    )?;

    writer.count("currents", plan.currents().len())?;
    writer.count("sources", plan.sources().len())?;
    writer.count("contributions", plan.contributions().len())?;
    writer.count("finalizations", plan.finalizations().len())?;
    writer.count("closures", plan.closures().len())?;
    writer.count("row groups", plan.row_groups().len())?;
    writer.count("momentum forms", plan.momentum_forms().len())?;
    writer.count("momentum terms", plan.momentum_terms().len())?;
    writer.count("selector domains", plan.selector_domains().len())?;
    writer.count("selector words", plan.selector_words().len())?;
    writer.count("replay targets", plan.replay_targets().len())?;
    writer.count("source permutations", plan.source_permutations().len())?;
    writer.count("replay momentum signs", plan.replay_momentum_signs().len())?;
    writer.count("replay helicity map", plan.replay_helicity_map().len())?;
    writer.count(
        "amplitude destinations",
        plan.amplitude_destinations().len(),
    )?;
    writer.count("resolved helicities", plan.resolved_helicities().len())?;
    writer.count(
        "source-state assignments",
        plan.source_state_assignments().len(),
    )?;
    writer.count(
        "source dispatch variants",
        plan.source_dispatch_variants().len(),
    )?;
    writer.count("source embeddings", plan.source_embeddings().len())?;
    writer.count("source projections", plan.source_projections().len())?;
    writer.count(
        "resolved source selections",
        plan.resolved_source_selections().len(),
    )?;
    writer.count("public helicities", plan.public_helicities().len())?;
    writer.count("exact factors", plan.exact_factors().len())?;
    writer.count(
        "closure proof contributions",
        plan.closure_proofs().contributions().len(),
    )?;
    writer.count("closure proof groups", plan.closure_proofs().groups().len())?;
    writer.count(
        "reflection certificates",
        plan.closure_proofs().reflection_certificates().len(),
    )?;
    writer.count(
        "three-line traversal certificates",
        plan.closure_proofs()
            .three_line_traversal_certificates()
            .len(),
    )?;

    for row in plan.currents() {
        writer.u32(row.semantic_current_id)?;
        writer.u16(row.node_kind as u16)?;
        writer.u16(row.stage)?;
        writer.u32(row.state_template_id)?;
        writer.u32(row.component_base)?;
        writer.u16(row.component_count)?;
        writer.u16(0)?;
        writer.u32(row.momentum_form_id)?;
        writer.u32(row.selector_domain_id)?;
        writer.u32(row.first_use)?;
        writer.u32(row.last_use)?;
        writer.u32(row.source_row_or_sentinel)?;
        writer.u32(row.finalization_row_or_sentinel)?;
    }
    for row in plan.sources() {
        writer.u32(row.source_slot)?;
        writer.u32(row.destination_component_base)?;
        writer.u32(row.momentum_form_id)?;
        writer.u32(row.source_template_or_dispatch_domain)?;
        writer.i32(row.spin_state_class)?;
        writer.u32(row.exact_factor_id)?;
        writer.u32(row.selector_domain_id)?;
    }
    for row in plan.contributions() {
        for value in [
            row.parent0_component_base,
            row.parent1_component_base_or_sentinel,
            row.parent0_momentum_form_id,
            row.parent1_momentum_form_id_or_sentinel,
            row.destination_component_base,
            row.exact_factor_id,
            row.selector_domain_id,
            row.flags,
        ] {
            writer.u32(value)?;
        }
    }
    for row in plan.finalizations() {
        writer.u32(row.component_base)?;
        writer.u16(row.component_count)?;
        writer.u16(0)?;
        for value in [
            row.momentum_form_id,
            row.exact_factor_id,
            row.selector_domain_id,
            row.flags,
        ] {
            writer.u32(value)?;
        }
    }
    for row in plan.closures() {
        for value in [
            row.parent0_component_base,
            row.parent1_component_base_or_sentinel,
            row.parent0_momentum_form_id,
            row.parent1_momentum_form_id_or_sentinel,
            row.amplitude_destination_id,
            row.exact_factor_id,
            row.component_factor_start,
        ] {
            writer.u32(value)?;
        }
        writer.u16(row.component_count)?;
        writer.u16(0)?;
        writer.u32(row.selector_domain_id)?;
        writer.u32(row.flags)?;
    }
    for row in plan.row_groups() {
        writer.u16(row.stage)?;
        writer.u16(row.role as u16)?;
        writer.u16(row.destination_operation as u16)?;
        writer.u16(0)?;
        writer.u32(row.direct_executor_id)?;
        writer.u64(row.row_start)?;
        writer.u32(row.row_count)?;
        writer.u32(0)?;
    }
    for row in plan.momentum_forms() {
        writer.u64(row.term_start)?;
        writer.u32(row.term_count)?;
        writer.u32(0)?;
    }
    for row in plan.momentum_terms() {
        writer.u32(row.source_slot)?;
        writer.i32(row.coefficient)?;
    }
    for row in plan.selector_domains() {
        writer.u64(row.word_start)?;
        writer.u32(row.word_count)?;
        writer.u32(0)?;
    }
    for value in plan.selector_words() {
        writer.u64(*value)?;
    }
    for row in plan.replay_targets() {
        writer.u32(row.public_flow_id)?;
        writer.u32(row.representative_id)?;
        writer.u64(row.source_permutation_start)?;
        writer.u32(row.source_permutation_count)?;
        writer.u64(row.helicity_map_start)?;
        writer.u32(row.helicity_map_count)?;
        writer.u32(row.phase_exact_factor_id)?;
        writer.u32(row.multiplicity)?;
        writer.u32(row.selector_domain_id)?;
    }
    for value in plan.source_permutations() {
        writer.u32(*value)?;
    }
    for value in plan.replay_momentum_signs() {
        writer.i32(*value)?;
    }
    for value in plan.replay_helicity_map() {
        writer.u32(*value)?;
    }
    for row in plan.amplitude_destinations() {
        writer.u64(row.closure_row_start)?;
        writer.u32(row.id)?;
        writer.u32(row.target_sector_id)?;
        writer.u32(row.target_helicity_id_or_sentinel)?;
        writer.u32(row.closure_row_count)?;
        writer.u32(row.selector_domain_id)?;
        writer.u32(0)?;
    }
    for row in plan.resolved_helicities() {
        writer.u64(row.source_state_start)?;
        writer.u64(row.source_selection_start)?;
        writer.u64(row.public_helicity_start)?;
        writer.u32(row.id)?;
        writer.u32(row.source_state_count)?;
        writer.u32(row.source_selection_count)?;
        writer.u32(row.public_helicity_count)?;
        writer.u32(row.selector_domain_id)?;
    }
    for row in plan.source_state_assignments() {
        writer.u32(row.source_slot)?;
        writer.u32(row.state_index)?;
    }
    for row in plan.source_dispatch_variants() {
        writer.u64(row.embedding_start)?;
        writer.u64(row.projection_start)?;
        for value in [
            row.source_row_id,
            row.dispatch_domain_id,
            row.runtime_variant_id,
            row.source_state_index,
            row.source_template_id,
            row.source_state_template_id,
            row.crossed_state_template_id,
        ] {
            writer.u32(value)?;
        }
        writer.i32(row.crossed_spin_state_class)?;
        for value in [
            row.direct_executor_id,
            row.crossing_exact_factor_id,
            row.embedding_count,
            row.projection_count,
        ] {
            writer.u32(value)?;
        }
    }
    for row in plan.source_embeddings() {
        writer.u32(row.full_component)?;
        writer.u32(row.source_component_or_sentinel)?;
        writer.u32(row.exact_factor_id)?;
    }
    for row in plan.source_projections() {
        writer.u32(row.source_component)?;
        writer.u32(row.full_component)?;
    }
    for row in plan.resolved_source_selections() {
        writer.u32(row.source_slot)?;
        writer.u32(row.dispatch_variant_id)?;
    }
    for value in plan.public_helicities() {
        writer.i32(*value)?;
    }
    for value in plan.exact_factors() {
        writer.exact(*value)?;
    }
    for row in plan.closure_proofs().contributions() {
        writer.u32(row.id())?;
        writer.u32(row.target_sector_id())?;
        writer.optional_u32(row.target_destination_id())?;
        writer.optional_u32(row.target_helicity_id())?;
        writer.u32(row.closure_template_id())?;
        writer.digest(row.closure_template_semantic_digest())?;
        writer.optional_u32(row.quantum_flow_template_id())?;
        writer.u32_slice(
            "closure construction builder-parent IDs",
            row.construction_parent_builder_ids(),
        )?;
        writer.optional_u32_slice(
            "closure construction runtime-parent IDs",
            row.construction_parent_runtime_ids(),
        )?;
        writer.digest_slice(
            "closure construction parent semantic digests",
            row.construction_parent_semantic_digests(),
        )?;
        writer.digest_slice(
            "closure construction parent color digests",
            row.construction_parent_color_digests(),
        )?;
        writer.u32_slice(
            "closure construction parent permutation",
            row.construction_parent_permutation(),
        )?;
        writer.u32_slice(
            "closure reconstruction parent permutation",
            row.reconstruction_parent_permutation(),
        )?;
        writer.u32_slice(
            "closure evaluator parent permutation",
            row.evaluator_parent_permutation(),
        )?;
        writer.u32(row.color_witness_term_id())?;
        writer.digest(row.color_witness_proof_digest())?;
        writer.optional_u32(row.three_line_certificate_id())?;
        writer.u32_slice(
            "closure pairing certificates",
            row.pairing_certificate_ids(),
        )?;
        writer.optional_u32(row.reflection_certificate_id())?;
        writer.exact(row.exact_factor())?;
        writer.u32(row.multiplicity())?;
    }
    for row in plan.closure_proofs().groups() {
        writer.u32(row.id())?;
        writer.optional_u32(row.emitted_runtime_closure_term_id())?;
        writer.optional_u32(row.emitted_direct_closure_row_id())?;
        writer.u64(row.contribution_range().start)?;
        writer.u64(row.contribution_range().count)?;
        writer.exact(row.exact_summed_factor())?;
        writer.digest(row.component_factor_digest())?;
        writer.digest(row.candidate_selector_domain_digest())?;
        writer.digest(row.selector_domain_digest())?;
    }
    for row in plan.closure_proofs().reflection_certificates() {
        writer.u32(row.id())?;
        writer.digest(row.orbit_identity())?;
        writer.digest(row.reciprocal_identity())?;
        writer.u32_slice("reflection source permutation", row.source_permutation())?;
        writer.exact(row.exact_phase())?;
        writer.u32(u32::from(row.fixed_point()))?;
        writer.u32(row.orbit_size())?;
        writer.u32(row.proof_algorithm_id())?;
        writer.digest(row.proof_digest())?;
    }
    for row in plan.closure_proofs().three_line_traversal_certificates() {
        writer.u32(row.id())?;
        writer.u32(row.sector_id())?;
        writer.u32(row.kind().code())?;
        writer.u32(row.sink_block_ordinal())?;
        writer.u32_slice(
            "three-line reference block order",
            row.reference_block_order(),
        )?;
        writer.u32_slice("three-line witness block order", row.witness_block_order())?;
        writer.u32_slice("three-line block permutation", row.block_permutation())?;
        writer.u32_slice(
            "three-line reference source order",
            row.reference_source_order(),
        )?;
        writer.u32_slice(
            "three-line witness source order",
            row.witness_source_order(),
        )?;
        writer.u32_slice(
            "three-line source-position permutation",
            row.source_position_permutation(),
        )?;
        writer.u32(row.closure_anchor_source_slot())?;
        writer.u32(row.pairing_rule_id())?;
        writer.digest(row.proof_digest())?;
    }
    Ok(writer.bytes_written)
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
#[cfg_attr(target_vendor = "apple", cold)]
pub fn decode_recurrence_direct_plan_v2(bytes: &[u8]) -> RusticolResult<DirectRecurrencePlan> {
    #[cfg(feature = "on-the-fly-test-support")]
    super::on_the_fly::reject_forbidden_work_if_probed(
        super::on_the_fly::OnTheFlyForbiddenWorkV1::DirectPlanDecode,
    )?;
    let mut reader = Reader::new(bytes)?;
    if reader.take(8, "magic")? != MAGIC {
        return Err(invalid(
            "unsupported recurrence payload; regenerate with direct-plan v2",
        ));
    }
    if reader.u32("version")? != VERSION {
        return Err(invalid(
            "unsupported recurrence version; regenerate with direct-plan v2",
        ));
    }
    if reader.u32("header flags")? != 0 {
        return Err(invalid("unsupported nonzero header flags"));
    }
    let strategy = RecurrenceStrategy::try_from(reader.u32("strategy")?)
        .map_err(|error| invalid(error.message()))?;
    let point_tile_size = reader.u32("point tile size")?;
    let workspace_mib = reader.u32("workspace MiB")?;
    let current_arena_components = reader.u32("current arena components")?;
    let physical_sector_count = reader.u32("physical LC sectors")?;
    let retained_helicity_count = reader.u64("retained helicities")?;
    let amplitude_destination_count = reader.u32("amplitude destinations")?;
    let parameter_value_count = reader.u32("parameter values")?;
    let external_source_count = reader.u32("external sources")?;
    let state_template_count = reader.u32("state templates")?;
    let source_template_count = reader.u32("source templates")?;
    let source_template_or_dispatch_count = reader.u32("source templates or dispatch domains")?;
    let runtime_helicity_contract_count = reader.u32("runtime-helicity contracts")?;
    let runtime_helicity_variant_count = reader.u32("runtime-helicity variants")?;
    let direct_executor_count = reader.u32("direct executors")?;
    let semantic_digest = reader.digest("semantic digest")?;
    let prepared_pack_digest = reader.digest("prepared-pack digest")?;
    let direct_template_catalog_digest = reader.digest("direct-template catalog digest")?;
    let expected_runtime_layout_digest = reader.digest("runtime-layout digest")?;
    let expected_closure_proof_digest =
        reader.digest("closure-proof semantic completeness digest")?;
    let closure_candidate_domain_certificate = ClosureCandidateDomainCertificateV1::new(
        reader.u64("closure candidate-domain count")?,
        reader.digest("closure candidate-domain digest")?,
    );

    let current_count = reader.count("currents", 44)?;
    let source_count = reader.count("sources", 28)?;
    let contribution_count = reader.count("contributions", 32)?;
    let finalization_count = reader.count("finalizations", 24)?;
    let closure_count = reader.count("closures", 40)?;
    let row_group_count = reader.count("row groups", 28)?;
    let momentum_form_count = reader.count("momentum forms", 16)?;
    let momentum_term_count = reader.count("momentum terms", 8)?;
    let selector_domain_count = reader.count("selector domains", 16)?;
    let selector_word_count = reader.count("selector words", 8)?;
    let replay_target_count = reader.count("replay targets", 44)?;
    let source_permutation_count = reader.count("source permutations", 4)?;
    let replay_momentum_sign_count = reader.count("replay momentum signs", 4)?;
    let replay_helicity_map_count = reader.count("replay helicity map", 4)?;
    let amplitude_destination_descriptor_count = reader.count("amplitude destinations", 32)?;
    let resolved_helicity_count = reader.count("resolved helicities", 44)?;
    let source_state_assignment_count = reader.count("source-state assignments", 8)?;
    let source_dispatch_variant_count = reader.count("source dispatch variants", 64)?;
    let source_embedding_count = reader.count("source embeddings", 12)?;
    let source_projection_count = reader.count("source projections", 8)?;
    let resolved_source_selection_count = reader.count("resolved source selections", 8)?;
    let public_helicity_count = reader.count("public helicities", 4)?;
    let exact_factor_count = reader.count("exact factors", 64)?;
    let closure_proof_contribution_count = reader.count("closure proof contributions", 0)?;
    let closure_proof_group_count = reader.count("closure proof groups", 0)?;
    let reflection_certificate_count = reader.count("reflection certificates", 0)?;
    let three_line_traversal_certificate_count =
        reader.count("three-line traversal certificates", 0)?;

    let mut currents = Vec::with_capacity(current_count);
    for _ in 0..current_count {
        let semantic_current_id = reader.u32("current semantic ID")?;
        let node_kind = DirectNodeKind::try_from(reader.u16("current node kind")?)?;
        let stage = reader.u16("current stage")?;
        let state_template_id = reader.u32("current state template")?;
        let component_base = reader.u32("current component base")?;
        let component_count = reader.u16("current component count")?;
        require_zero(reader.u16("current reserved")?, "current reserved")?;
        currents.push(DirectCurrentDescriptor {
            semantic_current_id,
            node_kind,
            state_template_id,
            component_base,
            component_count,
            momentum_form_id: reader.u32("current momentum form")?,
            stage,
            selector_domain_id: reader.u32("current selector domain")?,
            first_use: reader.u32("current first use")?,
            last_use: reader.u32("current last use")?,
            source_row_or_sentinel: reader.u32("current source row")?,
            finalization_row_or_sentinel: reader.u32("current finalization row")?,
        });
    }
    let mut sources = Vec::with_capacity(source_count);
    for _ in 0..source_count {
        let source_slot = reader.u32("source slot")?;
        let destination_component_base = reader.u32("source destination")?;
        let momentum_form_id = reader.u32("source momentum form")?;
        let source_template_or_dispatch_domain = reader.u32("source template")?;
        let spin_state_class = reader.i32("source spin state")?;
        sources.push(DirectSourceRow {
            source_slot,
            destination_component_base,
            momentum_form_id,
            source_template_or_dispatch_domain,
            spin_state_class,
            exact_factor_id: reader.u32("source exact factor")?,
            selector_domain_id: reader.u32("source selector domain")?,
        });
    }
    let mut contributions = Vec::with_capacity(contribution_count);
    for _ in 0..contribution_count {
        contributions.push(DirectContributionRow {
            parent0_component_base: reader.u32("contribution parent 0")?,
            parent1_component_base_or_sentinel: reader.u32("contribution parent 1")?,
            parent0_momentum_form_id: reader.u32("contribution momentum 0")?,
            parent1_momentum_form_id_or_sentinel: reader.u32("contribution momentum 1")?,
            destination_component_base: reader.u32("contribution destination")?,
            exact_factor_id: reader.u32("contribution exact factor")?,
            selector_domain_id: reader.u32("contribution selector domain")?,
            flags: reader.u32("contribution flags")?,
        });
    }
    let mut finalizations = Vec::with_capacity(finalization_count);
    for _ in 0..finalization_count {
        let component_base = reader.u32("finalization component base")?;
        let component_count = reader.u16("finalization component count")?;
        require_zero(
            reader.u16("finalization reserved")?,
            "finalization reserved",
        )?;
        finalizations.push(DirectFinalizationRow {
            component_base,
            component_count,
            momentum_form_id: reader.u32("finalization momentum form")?,
            exact_factor_id: reader.u32("finalization exact factor")?,
            selector_domain_id: reader.u32("finalization selector domain")?,
            flags: reader.u32("finalization flags")?,
        });
    }
    let mut closures = Vec::with_capacity(closure_count);
    for _ in 0..closure_count {
        closures.push(DirectClosureRow {
            parent0_component_base: reader.u32("closure parent 0")?,
            parent1_component_base_or_sentinel: reader.u32("closure parent 1")?,
            parent0_momentum_form_id: reader.u32("closure momentum 0")?,
            parent1_momentum_form_id_or_sentinel: reader.u32("closure momentum 1")?,
            amplitude_destination_id: reader.u32("closure destination")?,
            exact_factor_id: reader.u32("closure exact factor")?,
            component_factor_start: reader.u32("closure component-factor start")?,
            component_count: {
                let count = reader.u16("closure component count")?;
                require_zero(reader.u16("closure reserved")?, "closure reserved")?;
                count
            },
            selector_domain_id: reader.u32("closure selector domain")?,
            flags: reader.u32("closure flags")?,
        });
    }
    let mut row_groups = Vec::with_capacity(row_group_count);
    for _ in 0..row_group_count {
        let stage = reader.u16("row-group stage")?;
        let role = DirectExecutorRole::try_from(reader.u16("row-group role")?)?;
        let destination_operation =
            DirectDestinationOperation::try_from(reader.u16("row-group operation")?)?;
        require_zero(reader.u16("row-group reserved")?, "row-group reserved")?;
        let direct_executor_id = reader.u32("row-group executor")?;
        let row_start = reader.u64("row-group row start")?;
        let row_count = reader.u32("row-group row count")?;
        require_zero(
            reader.u32("row-group trailing reserved")?,
            "row-group reserved",
        )?;
        row_groups.push(DirectRowGroupDescriptor {
            stage,
            role,
            destination_operation,
            direct_executor_id,
            row_start,
            row_count,
        });
    }
    let mut momentum_forms = Vec::with_capacity(momentum_form_count);
    for _ in 0..momentum_form_count {
        let term_start = reader.u64("momentum-form term start")?;
        let term_count = reader.u32("momentum-form term count")?;
        require_zero(
            reader.u32("momentum-form reserved")?,
            "momentum-form reserved",
        )?;
        momentum_forms.push(DirectMomentumFormDescriptor {
            term_start,
            term_count,
        });
    }
    let mut momentum_terms = Vec::with_capacity(momentum_term_count);
    for _ in 0..momentum_term_count {
        momentum_terms.push(DirectMomentumTerm {
            source_slot: reader.u32("momentum-term source")?,
            coefficient: reader.i32("momentum-term coefficient")?,
        });
    }
    let mut selector_domains = Vec::with_capacity(selector_domain_count);
    for _ in 0..selector_domain_count {
        let word_start = reader.u64("selector-domain word start")?;
        let word_count = reader.u32("selector-domain word count")?;
        require_zero(
            reader.u32("selector-domain reserved")?,
            "selector-domain reserved",
        )?;
        selector_domains.push(DirectSelectorDomainDescriptor {
            word_start,
            word_count,
        });
    }
    let mut selector_words = Vec::with_capacity(selector_word_count);
    for _ in 0..selector_word_count {
        selector_words.push(reader.u64("selector word")?);
    }
    let mut replay_targets = Vec::with_capacity(replay_target_count);
    for _ in 0..replay_target_count {
        replay_targets.push(DirectReplayTargetDescriptor {
            public_flow_id: reader.u32("replay public flow")?,
            representative_id: reader.u32("replay representative")?,
            source_permutation_start: reader.u64("replay permutation start")?,
            source_permutation_count: reader.u32("replay permutation count")?,
            helicity_map_start: reader.u64("replay helicity-map start")?,
            helicity_map_count: reader.u32("replay helicity-map count")?,
            phase_exact_factor_id: reader.u32("replay phase factor")?,
            multiplicity: reader.u32("replay multiplicity")?,
            selector_domain_id: reader.u32("replay selector domain")?,
        });
    }
    let mut source_permutations = Vec::with_capacity(source_permutation_count);
    for _ in 0..source_permutation_count {
        source_permutations.push(reader.u32("source permutation")?);
    }
    let mut replay_momentum_signs = Vec::with_capacity(replay_momentum_sign_count);
    for _ in 0..replay_momentum_sign_count {
        replay_momentum_signs.push(reader.i32("replay momentum sign")?);
    }
    let mut replay_helicity_map = Vec::with_capacity(replay_helicity_map_count);
    for _ in 0..replay_helicity_map_count {
        replay_helicity_map.push(reader.u32("replay helicity map")?);
    }
    let mut amplitude_destinations = Vec::with_capacity(amplitude_destination_descriptor_count);
    for _ in 0..amplitude_destination_descriptor_count {
        let closure_row_start = reader.u64("amplitude destination closure start")?;
        let id = reader.u32("amplitude destination ID")?;
        let target_sector_id = reader.u32("amplitude destination sector")?;
        let target_helicity_id_or_sentinel = reader.u32("amplitude destination helicity")?;
        let closure_row_count = reader.u32("amplitude destination closure count")?;
        let selector_domain_id = reader.u32("amplitude destination selector domain")?;
        require_zero(
            reader.u32("amplitude destination reserved")?,
            "amplitude destination reserved",
        )?;
        amplitude_destinations.push(DirectAmplitudeDestinationDescriptor {
            closure_row_start,
            id,
            target_sector_id,
            target_helicity_id_or_sentinel,
            closure_row_count,
            selector_domain_id,
        });
    }
    let mut resolved_helicities = Vec::with_capacity(resolved_helicity_count);
    for _ in 0..resolved_helicity_count {
        resolved_helicities.push(DirectResolvedHelicityDescriptor {
            source_state_start: reader.u64("resolved-helicity source-state start")?,
            source_selection_start: reader.u64("resolved-helicity source-selection start")?,
            public_helicity_start: reader.u64("resolved-helicity public-value start")?,
            id: reader.u32("resolved-helicity ID")?,
            source_state_count: reader.u32("resolved-helicity source-state count")?,
            source_selection_count: reader.u32("resolved-helicity source-selection count")?,
            public_helicity_count: reader.u32("resolved-helicity public-value count")?,
            selector_domain_id: reader.u32("resolved-helicity selector domain")?,
        });
    }
    let mut source_state_assignments = Vec::with_capacity(source_state_assignment_count);
    for _ in 0..source_state_assignment_count {
        source_state_assignments.push(DirectSourceStateAssignment {
            source_slot: reader.u32("source-state source slot")?,
            state_index: reader.u32("source-state index")?,
        });
    }
    let mut source_dispatch_variants = Vec::with_capacity(source_dispatch_variant_count);
    for _ in 0..source_dispatch_variant_count {
        source_dispatch_variants.push(DirectSourceDispatchVariantDescriptor {
            embedding_start: reader.u64("source-dispatch embedding start")?,
            projection_start: reader.u64("source-dispatch projection start")?,
            source_row_id: reader.u32("source-dispatch source row")?,
            dispatch_domain_id: reader.u32("source-dispatch domain")?,
            runtime_variant_id: reader.u32("source-dispatch runtime variant")?,
            source_state_index: reader.u32("source-dispatch source-state index")?,
            source_template_id: reader.u32("source-dispatch source template")?,
            source_state_template_id: reader.u32("source-dispatch source state")?,
            crossed_state_template_id: reader.u32("source-dispatch crossed state")?,
            crossed_spin_state_class: reader.i32("source-dispatch crossed spin state")?,
            direct_executor_id: reader.u32("source-dispatch direct executor")?,
            crossing_exact_factor_id: reader.u32("source-dispatch crossing factor")?,
            embedding_count: reader.u32("source-dispatch embedding count")?,
            projection_count: reader.u32("source-dispatch projection count")?,
        });
    }
    let mut source_embeddings = Vec::with_capacity(source_embedding_count);
    for _ in 0..source_embedding_count {
        source_embeddings.push(DirectSourceEmbeddingRow {
            full_component: reader.u32("source embedding full component")?,
            source_component_or_sentinel: reader.u32("source embedding source component")?,
            exact_factor_id: reader.u32("source embedding factor")?,
        });
    }
    let mut source_projections = Vec::with_capacity(source_projection_count);
    for _ in 0..source_projection_count {
        source_projections.push(DirectSourceProjectionRow {
            source_component: reader.u32("source projection source component")?,
            full_component: reader.u32("source projection full component")?,
        });
    }
    let mut resolved_source_selections = Vec::with_capacity(resolved_source_selection_count);
    for _ in 0..resolved_source_selection_count {
        resolved_source_selections.push(DirectResolvedSourceSelection {
            source_slot: reader.u32("resolved source-selection slot")?,
            dispatch_variant_id: reader.u32("resolved source-selection variant")?,
        });
    }
    let mut public_helicities = Vec::with_capacity(public_helicity_count);
    for _ in 0..public_helicity_count {
        public_helicities.push(reader.i32("public helicity")?);
    }
    let mut exact_factors = Vec::with_capacity(exact_factor_count);
    for _ in 0..exact_factor_count {
        exact_factors.push(reader.exact("exact factor")?);
    }
    let mut closure_proof_contributions = Vec::with_capacity(closure_proof_contribution_count);
    for _ in 0..closure_proof_contribution_count {
        closure_proof_contributions.push(ClosureProofContributionV2::new(
            reader.u32("closure proof contribution ID")?,
            reader.u32("closure proof target sector")?,
            reader.optional_u32("closure proof target destination")?,
            reader.optional_u32("closure proof target helicity")?,
            reader.u32("closure proof template")?,
            reader.digest("closure proof template semantic digest")?,
            reader.optional_u32("closure proof quantum flow")?,
            reader.u32_vec("closure construction builder-parent IDs")?,
            reader.optional_u32_vec("closure construction runtime-parent IDs")?,
            reader.digest_vec("closure construction parent semantic digests")?,
            reader.digest_vec("closure construction parent color digests")?,
            reader.u32_vec("closure construction parent permutation")?,
            reader.u32_vec("closure reconstruction parent permutation")?,
            reader.u32_vec("closure evaluator parent permutation")?,
            reader.u32("closure color witness term")?,
            reader.digest("closure color witness proof digest")?,
            reader.optional_u32("closure three-line certificate")?,
            reader.u32_vec("closure pairing certificates")?,
            reader.optional_u32("closure reflection certificate")?,
            reader.exact("closure proof contribution factor")?,
            reader.u32("closure proof contribution multiplicity")?,
        )?);
    }
    let mut closure_proof_groups = Vec::with_capacity(closure_proof_group_count);
    for _ in 0..closure_proof_group_count {
        closure_proof_groups.push(
            ClosureExecutionProofGroupV2::new_with_candidate_selector_domain(
                reader.u32("closure proof group ID")?,
                reader.optional_u32("closure proof group runtime term")?,
                reader.optional_u32("closure proof group direct row")?,
                CheckedTableRange::new(
                    reader.u64("closure proof group contribution start")?,
                    reader.u64("closure proof group contribution count")?,
                ),
                reader.exact("closure proof group summed factor")?,
                reader.digest("closure proof group component-factor digest")?,
                reader.digest("closure proof group candidate-selector digest")?,
                reader.digest("closure proof group selector-domain digest")?,
            )?,
        );
    }
    let mut reflection_certificates = Vec::with_capacity(reflection_certificate_count);
    for _ in 0..reflection_certificate_count {
        let id = reader.u32("reflection certificate ID")?;
        let orbit_identity = reader.digest("reflection orbit identity")?;
        let reciprocal_identity = reader.digest("reflection reciprocal identity")?;
        let source_permutation = reader.u32_vec("reflection source permutation")?;
        let exact_phase = reader.exact("reflection exact phase")?;
        let fixed_point = match reader.u32("reflection fixed-point flag")? {
            0 => false,
            1 => true,
            value => {
                return Err(invalid(format!(
                    "reflection fixed-point flag must be 0 or 1, found {value}"
                )));
            }
        };
        reflection_certificates.push(ReflectionCertificateV1::new(
            id,
            orbit_identity,
            reciprocal_identity,
            source_permutation,
            exact_phase,
            fixed_point,
            reader.u32("reflection orbit size")?,
            reader.u32("reflection proof algorithm")?,
            reader.digest("reflection proof digest")?,
        )?);
    }
    let mut three_line_traversal_certificates =
        Vec::with_capacity(three_line_traversal_certificate_count);
    for _ in 0..three_line_traversal_certificate_count {
        let id = reader.u32("three-line traversal certificate ID")?;
        let sector_id = reader.u32("three-line traversal sector")?;
        let kind = ThreeLineTraversalKindV1::try_from(reader.u32("three-line traversal kind")?)
            .map_err(|error| invalid(error.message()))?;
        let sink_block_ordinal = reader.u32("three-line traversal sink block")?;
        let reference_block_order = reader.u32_vec("three-line reference block order")?;
        let witness_block_order = reader.u32_vec("three-line witness block order")?;
        let block_permutation = reader.u32_vec("three-line block permutation")?;
        let reference_source_order = reader.u32_vec("three-line reference source order")?;
        let witness_source_order = reader.u32_vec("three-line witness source order")?;
        let source_position_permutation =
            reader.u32_vec("three-line source-position permutation")?;
        three_line_traversal_certificates.push(ThreeLineTraversalCertificateV1::new(
            id,
            sector_id,
            kind,
            sink_block_ordinal,
            reference_block_order,
            witness_block_order,
            block_permutation,
            reference_source_order,
            witness_source_order,
            source_position_permutation,
            reader.u32("three-line closure anchor source slot")?,
            reader.u32("three-line pairing rule")?,
            reader.digest("three-line traversal proof digest")?,
        )?);
    }
    reader.finish()?;
    let closure_proofs =
        ClosureProofMetadataV2::new_with_three_line_certificates_candidate_domain_and_expected_digest(
            closure_proof_contributions,
            closure_proof_groups,
            reflection_certificates,
            three_line_traversal_certificates,
            closure_candidate_domain_certificate,
            expected_closure_proof_digest,
        )?;

    let plan = DirectRecurrencePlan::new(DirectRecurrencePlanParts {
        strategy,
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        point_tile_size,
        workspace_mib,
        current_arena_components,
        physical_sector_count,
        retained_helicity_count,
        amplitude_destination_count,
        parameter_value_count,
        external_source_count,
        state_template_count,
        source_template_count,
        source_template_or_dispatch_count,
        runtime_helicity_contract_count,
        runtime_helicity_variant_count,
        direct_executor_count,
        currents,
        sources,
        contributions,
        finalizations,
        closures,
        row_groups,
        momentum_forms,
        momentum_terms,
        selector_domains,
        selector_words,
        replay_targets,
        source_permutations,
        replay_momentum_signs,
        replay_helicity_map,
        amplitude_destinations,
        resolved_helicities,
        source_state_assignments,
        source_dispatch_variants,
        source_embeddings,
        source_projections,
        resolved_source_selections,
        public_helicities,
        exact_factors,
        closure_proofs,
    })?;
    if plan.runtime_layout_digest() != expected_runtime_layout_digest {
        return Err(invalid("runtime-layout digest mismatch"));
    }
    Ok(plan)
}

fn read_rational(reader: &mut Reader<'_>, label: &str) -> RusticolResult<ExactRational> {
    let numerator = reader.i128(&format!("{label} numerator"))?;
    let denominator = reader.i128(&format!("{label} denominator"))?;
    let value =
        ExactRational::new(numerator, denominator).map_err(|error| invalid(error.message()))?;
    if value.numerator() != numerator || value.denominator() != denominator {
        return Err(invalid(format!("{label} is not canonically reduced")));
    }
    Ok(value)
}

fn require_zero<T>(value: T, label: &str) -> RusticolResult<()>
where
    T: Default + PartialEq + std::fmt::Display,
{
    if value != T::default() {
        return Err(invalid(format!("{label} must be zero, found {value}")));
    }
    Ok(())
}

#[cfg(test)]
#[path = "direct_codec_tests.rs"]
mod tests;
