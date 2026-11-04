// SPDX-License-Identifier: 0BSD

use super::*;
use crate::recurrence::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DirectResolvedSourceSelection,
    DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow, DirectSourceProjectionRow,
};
use crate::recurrence::{
    CheckedTableRange, ClosureExecutionProofGroupV2, ClosureProofContributionV2, ExactRational,
    ThreeLineTraversalCertificateV1, ThreeLineTraversalKindV1,
    three_line_traversal_proof_digest_v1,
};

pub(crate) fn valid_three_line_traversal_certificate() -> ThreeLineTraversalCertificateV1 {
    let reference_block_order = vec![0, 1, 2];
    let witness_block_order = vec![1, 0, 2];
    let block_permutation = vec![1, 0, 2];
    let reference_source_order = vec![10, 11, 12];
    let witness_source_order = vec![11, 10, 12];
    let source_position_permutation = vec![1, 0, 2];
    let proof_digest = three_line_traversal_proof_digest_v1(
        0,
        ThreeLineTraversalKindV1::Partner,
        2,
        &reference_block_order,
        &witness_block_order,
        &block_permutation,
        &reference_source_order,
        &witness_source_order,
        &source_position_permutation,
        12,
        7,
    )
    .unwrap();
    ThreeLineTraversalCertificateV1::new(
        0,
        0,
        ThreeLineTraversalKindV1::Partner,
        2,
        reference_block_order,
        witness_block_order,
        block_permutation,
        reference_source_order,
        witness_source_order,
        source_position_permutation,
        12,
        7,
        proof_digest,
    )
    .unwrap()
}

pub(crate) fn valid_parts() -> DirectRecurrencePlanParts {
    let identity = vec![0];
    let proof_contribution = ClosureProofContributionV2::new(
        0,
        0,
        Some(0),
        Some(0),
        0,
        SemanticDigest::new([0x41; 32]).unwrap(),
        None,
        vec![1],
        vec![Some(1)],
        vec![SemanticDigest::new([0x42; 32]).unwrap()],
        vec![SemanticDigest::new([0x43; 32]).unwrap()],
        identity.clone(),
        identity.clone(),
        identity,
        0,
        SemanticDigest::new([0x44; 32]).unwrap(),
        Some(0),
        vec![],
        None,
        ExactComplexRational::ONE,
        1,
    )
    .unwrap();
    let proof_group = ClosureExecutionProofGroupV2::new(
        0,
        Some(0),
        Some(0),
        CheckedTableRange::new(0, 1),
        ExactComplexRational::ONE,
        closure_component_factor_digest_v2(&[ExactComplexRational::ONE]).unwrap(),
        closure_selector_domain_digest_v2(&[1]).unwrap(),
    )
    .unwrap();
    DirectRecurrencePlanParts {
        strategy: RecurrenceStrategy::TopologyReplay,
        semantic_digest: SemanticDigest::new([0x11; 32]).unwrap(),
        prepared_pack_digest: SemanticDigest::new([0x22; 32]).unwrap(),
        direct_template_catalog_digest: SemanticDigest::new([0x33; 32]).unwrap(),
        point_tile_size: 1024,
        workspace_mib: 256,
        current_arena_components: 3,
        physical_sector_count: 1,
        retained_helicity_count: 1,
        amplitude_destination_count: 1,
        parameter_value_count: 1,
        external_source_count: 1,
        state_template_count: 2,
        source_template_count: 8,
        source_template_or_dispatch_count: 8,
        runtime_helicity_contract_count: 0,
        runtime_helicity_variant_count: 0,
        direct_executor_count: 4,
        currents: vec![
            DirectCurrentDescriptor {
                semantic_current_id: 0,
                node_kind: DirectNodeKind::Source,
                state_template_id: 0,
                component_base: 0,
                component_count: 2,
                momentum_form_id: 0,
                stage: 0,
                selector_domain_id: 0,
                first_use: 0,
                last_use: 1,
                source_row_or_sentinel: 0,
                finalization_row_or_sentinel: DIRECT_NONE_U32,
            },
            DirectCurrentDescriptor {
                semantic_current_id: 1,
                node_kind: DirectNodeKind::Current,
                state_template_id: 1,
                component_base: 2,
                component_count: 1,
                momentum_form_id: 0,
                stage: 1,
                selector_domain_id: 0,
                first_use: 1,
                last_use: 2,
                source_row_or_sentinel: DIRECT_NONE_U32,
                finalization_row_or_sentinel: 0,
            },
        ],
        sources: vec![DirectSourceRow {
            source_slot: 0,
            destination_component_base: 0,
            momentum_form_id: 0,
            source_template_or_dispatch_domain: 7,
            spin_state_class: -1,
            exact_factor_id: 0,
            selector_domain_id: 0,
        }],
        contributions: vec![DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 2,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
        }],
        finalizations: vec![DirectFinalizationRow {
            component_base: 2,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        }],
        closures: vec![DirectClosureRow {
            parent0_component_base: 2,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            amplitude_destination_id: 0,
            exact_factor_id: 0,
            component_factor_start: 0,
            component_count: 1,
            selector_domain_id: 0,
            flags: 0,
        }],
        row_groups: vec![
            DirectRowGroupDescriptor {
                stage: 0,
                role: DirectExecutorRole::Source,
                destination_operation: DirectDestinationOperation::Initialize,
                direct_executor_id: 0,
                row_start: 0,
                row_count: 1,
            },
            DirectRowGroupDescriptor {
                stage: 1,
                role: DirectExecutorRole::Contribution,
                destination_operation: DirectDestinationOperation::Add,
                direct_executor_id: 1,
                row_start: 0,
                row_count: 1,
            },
            DirectRowGroupDescriptor {
                stage: 1,
                role: DirectExecutorRole::Finalization,
                destination_operation: DirectDestinationOperation::FinalizeInPlace,
                direct_executor_id: 2,
                row_start: 0,
                row_count: 1,
            },
            DirectRowGroupDescriptor {
                stage: 2,
                role: DirectExecutorRole::Closure,
                destination_operation: DirectDestinationOperation::ClosureAdd,
                direct_executor_id: 3,
                row_start: 0,
                row_count: 1,
            },
        ],
        momentum_forms: vec![DirectMomentumFormDescriptor {
            term_start: 0,
            term_count: 1,
        }],
        momentum_terms: vec![DirectMomentumTerm {
            source_slot: 0,
            coefficient: 1,
        }],
        selector_domains: vec![DirectSelectorDomainDescriptor {
            word_start: 0,
            word_count: 1,
        }],
        selector_words: vec![1],
        replay_targets: vec![DirectReplayTargetDescriptor {
            public_flow_id: 0,
            representative_id: 0,
            source_permutation_start: 0,
            source_permutation_count: 1,
            helicity_map_start: 0,
            helicity_map_count: 1,
            phase_exact_factor_id: 0,
            multiplicity: 1,
            selector_domain_id: 0,
        }],
        source_permutations: vec![0],
        replay_momentum_signs: vec![1],
        replay_helicity_map: vec![0],
        amplitude_destinations: vec![DirectAmplitudeDestinationDescriptor {
            closure_row_start: 0,
            id: 0,
            target_sector_id: 0,
            target_helicity_id_or_sentinel: 0,
            closure_row_count: 1,
            selector_domain_id: 0,
        }],
        resolved_helicities: vec![DirectResolvedHelicityDescriptor {
            source_state_start: 0,
            source_selection_start: 0,
            public_helicity_start: 0,
            id: 0,
            source_state_count: 1,
            source_selection_count: 0,
            public_helicity_count: 1,
            selector_domain_id: 0,
        }],
        source_state_assignments: vec![DirectSourceStateAssignment {
            source_slot: 0,
            state_index: 0,
        }],
        source_dispatch_variants: vec![],
        source_embeddings: vec![],
        source_projections: vec![],
        resolved_source_selections: vec![],
        public_helicities: vec![-1],
        exact_factors: vec![ExactComplexRational::ONE],
        closure_proofs: ClosureProofMetadataV2::new_with_three_line_certificates(
            vec![proof_contribution],
            vec![proof_group],
            vec![],
            vec![valid_three_line_traversal_certificate()],
        )
        .unwrap(),
    }
}

pub(crate) fn valid_plan() -> DirectRecurrencePlan {
    DirectRecurrencePlan::new(valid_parts()).unwrap()
}

#[cfg(feature = "on-the-fly-test-support")]
#[test]
fn genuine_probe_direct_plan_materialization_is_poisoned() {
    let guard = crate::recurrence::on_the_fly::OnTheFlyForbiddenWorkGuardV1::begin().unwrap();
    assert!(
        DirectRecurrencePlan::new(valid_parts())
            .unwrap_err()
            .to_string()
            .contains("DirectRecurrencePlan materialization is poisoned")
    );
    let counts = guard.finish().unwrap();
    assert_eq!(counts.direct_plan_load_attempts, 0);
    assert_eq!(counts.direct_plan_decode_attempts, 0);
    assert_eq!(counts.direct_plan_materialization_attempts, 1);
    assert_eq!(counts.established_builder_attempts, 0);
    assert!(DirectRecurrencePlan::new(valid_parts()).is_ok());
}

#[test]
fn direct_descriptor_native_layouts_are_fixed_width() {
    assert_eq!(std::mem::size_of::<DirectCurrentDescriptor>(), 48);
    assert_eq!(std::mem::size_of::<DirectSourceRow>(), 28);
    assert_eq!(std::mem::size_of::<DirectContributionRow>(), 32);
    assert_eq!(std::mem::size_of::<DirectFinalizationRow>(), 24);
    assert_eq!(std::mem::size_of::<DirectClosureRow>(), 40);
    assert_eq!(std::mem::size_of::<DirectRowGroupDescriptor>(), 32);
    assert_eq!(std::mem::size_of::<DirectMomentumFormDescriptor>(), 16);
    assert_eq!(std::mem::size_of::<DirectMomentumTerm>(), 8);
    assert_eq!(std::mem::size_of::<DirectSelectorDomainDescriptor>(), 16);
    assert_eq!(std::mem::size_of::<DirectReplayTargetDescriptor>(), 48);
    assert_eq!(
        std::mem::size_of::<DirectAmplitudeDestinationDescriptor>(),
        32
    );
    assert_eq!(std::mem::size_of::<DirectResolvedHelicityDescriptor>(), 48);
    assert_eq!(std::mem::size_of::<DirectSourceStateAssignment>(), 8);
    assert_eq!(
        std::mem::size_of::<DirectSourceDispatchVariantDescriptor>(),
        64
    );
    assert_eq!(std::mem::size_of::<DirectSourceEmbeddingRow>(), 12);
    assert_eq!(std::mem::size_of::<DirectSourceProjectionRow>(), 8);
    assert_eq!(std::mem::size_of::<DirectResolvedSourceSelection>(), 8);
}

#[test]
fn direct_plan_validates_and_has_a_stable_nonzero_layout_digest() {
    let first = valid_plan();
    let second = valid_plan();
    assert_eq!(first, second);
    assert_eq!(
        first.runtime_layout_digest(),
        second.runtime_layout_digest()
    );
    assert_ne!(first.runtime_layout_digest().as_bytes(), &[0; 32]);
}

#[test]
fn selector_work_summary_reports_active_rows_and_components() {
    let plan = valid_plan();
    let summary = plan.selector_work_summary(0).unwrap();
    assert_eq!(summary.physical_sector_id, 0);
    assert_eq!(summary.current_count, 2);
    assert_eq!(summary.semantic_component_count, 3);
    assert_eq!(summary.source_row_count, 1);
    assert_eq!(summary.contribution_count, 1);
    assert_eq!(summary.finalization_count, 1);
    assert_eq!(summary.closure_count, 1);
    assert_eq!(summary.amplitude_destination_count, 1);
    assert_eq!(summary.row_count(), 4);
}

#[test]
fn selector_domains_support_multiword_masks_and_legacy_universal_encoding() {
    let mut parts = valid_parts();
    parts.physical_sector_count = 71;
    parts.selector_domains.extend([
        DirectSelectorDomainDescriptor {
            word_start: 1,
            word_count: 2,
        },
        DirectSelectorDomainDescriptor {
            word_start: 3,
            word_count: 1,
        },
    ]);
    parts.selector_words.extend([0, 1_u64 << 6, u64::MAX]);
    let plan = DirectRecurrencePlan::new(parts).unwrap();

    assert!(!plan.selector_domain_contains(1, 63).unwrap());
    assert!(plan.selector_domain_contains(1, 70).unwrap());
    assert!(plan.selector_domain_contains(2, 70).unwrap());
    assert!(plan.selector_domain_contains(2, 0).unwrap());
    assert!(
        plan.selector_domain_contains(2, 71)
            .unwrap_err()
            .to_string()
            .contains("out of bounds")
    );
}

#[test]
fn direct_plan_rejects_selector_domains_inconsistent_with_bound_rows() {
    let mut parts = valid_parts();
    parts.selector_domains.push(DirectSelectorDomainDescriptor {
        word_start: 1,
        word_count: 1,
    });
    parts.selector_words.push(0);
    parts.sources[0].selector_domain_id = 1;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("does not match its current"));

    let mut parts = valid_parts();
    parts.selector_domains.push(DirectSelectorDomainDescriptor {
        word_start: 1,
        word_count: 1,
    });
    parts.selector_words.push(0);
    parts.amplitude_destinations[0].selector_domain_id = 1;
    parts.closures[0].selector_domain_id = 1;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("excludes its target sector"));
}

#[test]
fn direct_plan_accepts_an_elided_identity_finalization() {
    let mut parts = valid_parts();
    parts.currents[1].finalization_row_or_sentinel = DIRECT_NONE_U32;
    parts.finalizations.clear();
    parts
        .row_groups
        .retain(|group| group.role != DirectExecutorRole::Finalization);
    DirectRecurrencePlan::new(parts).unwrap();
}

#[test]
fn direct_plan_canonicalizes_a_missing_initialization_marker() {
    let mut parts = valid_parts();
    parts.contributions[0].flags = 0;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    assert_eq!(
        plan.contributions()[0].flags,
        DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
    );
}

#[test]
fn direct_plan_rejects_multiple_initialization_markers_for_one_current() {
    let mut parts = valid_parts();
    parts.contributions.push(parts.contributions[0]);
    parts.row_groups[1].row_count = 2;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("exactly one"));
}

#[test]
fn direct_plan_rejects_invalid_arena_and_optional_parent_references() {
    let mut parts = valid_parts();
    parts.contributions[0].destination_component_base = 99;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("arena base")
    );

    let mut parts = valid_parts();
    parts.contributions[0].parent1_component_base_or_sentinel = 0;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("mismatched optional")
    );
}

#[test]
fn direct_plan_rejects_overlapping_live_arena_ranges() {
    let mut parts = valid_parts();
    parts.currents[1].component_base = 1;
    parts.contributions[0].destination_component_base = 1;
    parts.finalizations[0].component_base = 1;
    parts.closures[0].parent0_component_base = 1;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("shared by live currents"));
}

#[test]
fn direct_plan_rejects_row_group_gaps_and_role_operation_mismatch() {
    let mut parts = valid_parts();
    parts.row_groups[1].row_start = 1;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("row partition")
    );

    let mut parts = valid_parts();
    parts.row_groups[1].destination_operation = DirectDestinationOperation::Initialize;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("incompatible")
    );
}

#[test]
fn direct_plan_rejects_non_permutation_replay_mappings() {
    let mut parts = valid_parts();
    parts.external_source_count = 2;
    parts.replay_targets[0].source_permutation_count = 2;
    parts.source_permutations = vec![0, 0];
    parts.replay_momentum_signs = vec![1, 1];
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("not a permutation"));
}

#[test]
fn direct_plan_rejects_incomplete_or_out_of_range_replay_helicity_maps() {
    let mut parts = valid_parts();
    parts.replay_targets[0].helicity_map_count = 0;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("helicity map ranges cover"));

    let mut parts = valid_parts();
    parts.replay_helicity_map[0] = 1;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("not a bijection"));
}

#[test]
fn direct_plan_rejects_invalid_replay_momentum_signs() {
    let mut parts = valid_parts();
    parts.replay_momentum_signs.clear();
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("momentum signs"));

    let mut parts = valid_parts();
    parts.replay_momentum_signs[0] = 0;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("momentum signs"));
}

#[test]
fn direct_plan_requires_packed_replay_tables() {
    let mut parts = valid_parts();
    parts.source_permutations.push(0);
    parts.replay_momentum_signs.push(1);
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("ranges cover"));

    let mut parts = valid_parts();
    parts.replay_helicity_map.push(0);
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("ranges cover"));

    let mut parts = valid_parts();
    let mut overlapping = parts.replay_targets[0];
    overlapping.public_flow_id = 1;
    overlapping.helicity_map_start = 1;
    parts.replay_targets.push(overlapping);
    parts.source_permutations.push(0);
    parts.replay_momentum_signs.push(1);
    parts.replay_helicity_map.push(0);
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("canonical partition"));

    let mut parts = valid_parts();
    let mut gapped = parts.replay_targets[0];
    gapped.public_flow_id = 1;
    gapped.source_permutation_start = 2;
    gapped.helicity_map_start = 1;
    parts.replay_targets.push(gapped);
    parts.source_permutations.extend_from_slice(&[0, 0]);
    parts.replay_momentum_signs.extend_from_slice(&[1, 1]);
    parts.replay_helicity_map.push(0);
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("canonical partition"));
}

#[test]
fn direct_plan_authenticates_resolved_helicity_and_destination_contracts() {
    let mut parts = valid_parts();
    parts.amplitude_destinations[0].target_sector_id = 1;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("physical sector")
    );

    let mut parts = valid_parts();
    parts.resolved_helicities[0].source_state_count = 0;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("source state")
    );

    let mut parts = valid_parts();
    parts.amplitude_destinations[0].target_helicity_id_or_sentinel = 1;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("resolved helicity")
    );
}

#[test]
fn changing_physical_layout_changes_only_the_layout_digest() {
    let first = valid_plan();
    let mut parts = valid_parts();
    parts.point_tile_size = 128;
    let second = DirectRecurrencePlan::new(parts).unwrap();
    assert_eq!(first.semantic_digest(), second.semantic_digest());
    assert_ne!(
        first.runtime_layout_digest(),
        second.runtime_layout_digest()
    );
}

fn rebuild_proof_group(
    parts: &DirectRecurrencePlanParts,
    direct_row_id: Option<u32>,
    component_factor_digest: SemanticDigest,
    selector_domain_digest: SemanticDigest,
) -> ClosureProofMetadataV2 {
    let group = &parts.closure_proofs.groups()[0];
    ClosureProofMetadataV2::new_with_three_line_certificates(
        parts.closure_proofs.contributions().to_vec(),
        vec![
            ClosureExecutionProofGroupV2::new(
                group.id(),
                group.emitted_runtime_closure_term_id(),
                direct_row_id,
                group.contribution_range(),
                group.exact_summed_factor(),
                component_factor_digest,
                selector_domain_digest,
            )
            .unwrap(),
        ],
        parts.closure_proofs.reflection_certificates().to_vec(),
        parts
            .closure_proofs
            .three_line_traversal_certificates()
            .to_vec(),
    )
    .unwrap()
}

#[test]
fn direct_plan_rejects_missing_or_misbound_runtime_closure_rows() {
    let mut parts = valid_parts();
    parts.closure_proofs = rebuild_proof_group(
        &parts,
        None,
        parts.closure_proofs.groups()[0].component_factor_digest(),
        parts.closure_proofs.groups()[0].selector_domain_digest(),
    );
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("no direct closure row"));

    let mut parts = valid_parts();
    parts.closures[0].flags = 1;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("names proof group"));
}

#[test]
fn direct_plan_rejects_mutated_runtime_factor_parent_and_cold_digests() {
    let mut parts = valid_parts();
    parts.exact_factors.push(ExactComplexRational::new(
        ExactRational::new(-1, 1).unwrap(),
        ExactRational::ZERO,
    ));
    parts.closures[0].exact_factor_id = 1;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("factor does not match"));

    let mut parts = valid_parts();
    let original = &parts.closure_proofs.contributions()[0];
    let parent_mutation = ClosureProofContributionV2::new(
        original.id(),
        original.target_sector_id(),
        original.target_destination_id(),
        original.target_helicity_id(),
        original.closure_template_id(),
        original.closure_template_semantic_digest(),
        original.quantum_flow_template_id(),
        vec![0],
        vec![Some(0)],
        original.construction_parent_semantic_digests().to_vec(),
        original.construction_parent_color_digests().to_vec(),
        original.construction_parent_permutation().to_vec(),
        original.reconstruction_parent_permutation().to_vec(),
        original.evaluator_parent_permutation().to_vec(),
        original.color_witness_term_id(),
        original.color_witness_proof_digest(),
        original.three_line_certificate_id(),
        original.pairing_certificate_ids().to_vec(),
        original.reflection_certificate_id(),
        original.exact_factor(),
        original.multiplicity(),
    )
    .unwrap();
    parts.closure_proofs = ClosureProofMetadataV2::new_with_three_line_certificates(
        vec![parent_mutation],
        parts.closure_proofs.groups().to_vec(),
        vec![],
        parts
            .closure_proofs
            .three_line_traversal_certificates()
            .to_vec(),
    )
    .unwrap();
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("parents do not match"));

    let mut parts = valid_parts();
    parts.closure_proofs = rebuild_proof_group(
        &parts,
        Some(0),
        SemanticDigest::new([0x51; 32]).unwrap(),
        parts.closure_proofs.groups()[0].selector_domain_digest(),
    );
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("component-factor digest"));

    let mut parts = valid_parts();
    parts.closure_proofs = rebuild_proof_group(
        &parts,
        Some(0),
        parts.closure_proofs.groups()[0].component_factor_digest(),
        SemanticDigest::new([0x52; 32]).unwrap(),
    );
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("selector-domain digest"));
}

#[test]
fn direct_plan_accepts_cold_exact_zero_groups_without_runtime_rows() {
    let mut parts = valid_parts();
    let original = parts.closure_proofs.contributions()[0].clone();
    let negative = ClosureProofContributionV2::new(
        2,
        original.target_sector_id(),
        None,
        original.target_helicity_id(),
        original.closure_template_id(),
        original.closure_template_semantic_digest(),
        original.quantum_flow_template_id(),
        original.construction_parent_builder_ids().to_vec(),
        vec![None],
        original.construction_parent_semantic_digests().to_vec(),
        original.construction_parent_color_digests().to_vec(),
        original.construction_parent_permutation().to_vec(),
        original.reconstruction_parent_permutation().to_vec(),
        original.evaluator_parent_permutation().to_vec(),
        original.color_witness_term_id(),
        original.color_witness_proof_digest(),
        original.three_line_certificate_id(),
        original.pairing_certificate_ids().to_vec(),
        original.reflection_certificate_id(),
        ExactComplexRational::new(ExactRational::new(-1, 1).unwrap(), ExactRational::ZERO),
        1,
    )
    .unwrap();
    let positive = ClosureProofContributionV2::new(
        1,
        original.target_sector_id(),
        None,
        original.target_helicity_id(),
        original.closure_template_id(),
        original.closure_template_semantic_digest(),
        original.quantum_flow_template_id(),
        original.construction_parent_builder_ids().to_vec(),
        vec![None],
        original.construction_parent_semantic_digests().to_vec(),
        original.construction_parent_color_digests().to_vec(),
        original.construction_parent_permutation().to_vec(),
        original.reconstruction_parent_permutation().to_vec(),
        original.evaluator_parent_permutation().to_vec(),
        original.color_witness_term_id(),
        original.color_witness_proof_digest(),
        original.three_line_certificate_id(),
        original.pairing_certificate_ids().to_vec(),
        original.reflection_certificate_id(),
        ExactComplexRational::ONE,
        1,
    )
    .unwrap();
    parts.closure_proofs = ClosureProofMetadataV2::new_with_three_line_certificates(
        vec![original, positive, negative],
        vec![
            ClosureExecutionProofGroupV2::new(
                0,
                Some(0),
                Some(0),
                CheckedTableRange::new(0, 1),
                ExactComplexRational::ONE,
                closure_component_factor_digest_v2(&[ExactComplexRational::ONE]).unwrap(),
                closure_selector_domain_digest_v2(&[1]).unwrap(),
            )
            .unwrap(),
            ClosureExecutionProofGroupV2::new(
                1,
                None,
                None,
                CheckedTableRange::new(1, 2),
                ExactComplexRational::ZERO,
                SemanticDigest::new([0x53; 32]).unwrap(),
                SemanticDigest::new([0x54; 32]).unwrap(),
            )
            .unwrap(),
        ],
        vec![],
        parts
            .closure_proofs
            .three_line_traversal_certificates()
            .to_vec(),
    )
    .unwrap();
    DirectRecurrencePlan::new(parts).unwrap();
}
