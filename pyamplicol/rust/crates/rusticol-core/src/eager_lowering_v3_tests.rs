// SPDX-License-Identifier: 0BSD

use super::*;
use crate::RusticolErrorKind;

#[derive(Clone)]
struct Fixture {
    current_id: Vec<u32>,
    current_dimension: Vec<u32>,
    current_is_source: Vec<u8>,
    current_momentum_bitset: Vec<u32>,
    current_propagator_ir: Vec<u32>,
    current_propagator_kernel: Vec<u32>,
    source_id: Vec<u32>,
    source_current: Vec<u32>,
    source_external_label: Vec<u32>,
    source_input_momentum: Vec<u32>,
    source_ir: Vec<u32>,
    source_crossing_ir: Vec<u32>,
    source_crossing_factor: Vec<u32>,
    source_state: Vec<u32>,
    interaction_id: Vec<u32>,
    interaction_stage_size: Vec<u32>,
    interaction_left: Vec<u32>,
    interaction_right: Vec<u32>,
    interaction_result: Vec<u32>,
    interaction_coupling: Vec<u32>,
    interaction_coupling_factor: Vec<u32>,
    interaction_color_factor: Vec<u32>,
    interaction_evaluation_factor: Vec<u32>,
    interaction_group: Vec<u32>,
    interaction_ready: Vec<u8>,
    interaction_kernel: Vec<u32>,
    interaction_order: Vec<[u8; 2]>,
    interaction_normalization: Vec<u32>,
    interaction_output_factor: Vec<u8>,
    root_id: Vec<u32>,
    root_left: Vec<u32>,
    root_right: Vec<u32>,
    root_color_factor: Vec<u32>,
    root_contraction_ir: Vec<u32>,
    root_coupling: Vec<u32>,
    root_coupling_factor: Vec<u32>,
    root_helicity_weight: Vec<u32>,
    root_group: Vec<u32>,
    root_kernel: Vec<u32>,
    root_order: Vec<[u8; 2]>,
    root_normalization: Vec<u32>,
    root_output_factor: Vec<u8>,
    group_representative: Vec<u32>,
    group_start: Vec<u64>,
    group_count: Vec<u64>,
    group_members: Vec<u32>,
    momentum_slot: Vec<u32>,
    momentum_bitset: Vec<u32>,
    bitset_start: Vec<u64>,
    bitset_count: Vec<u64>,
    bitset_population: Vec<u64>,
    bitset_words: Vec<u64>,
    u32_sequence_start: Vec<u64>,
    u32_sequence_count: Vec<u64>,
    u32_sequence_values: Vec<u32>,
    i32_sequence_start: Vec<u64>,
    i32_sequence_count: Vec<u64>,
    i32_sequence_values: Vec<i32>,
    factor_real: Vec<f64>,
    factor_imaginary: Vec<f64>,
    factor_string: Vec<u32>,
    factor_source: Vec<u8>,
    factor_ir: Vec<u32>,
    factor_source_ir: Vec<u32>,
    coupling_factor: Vec<u32>,
    coupling_names: Vec<u32>,
    parameter_name: Vec<u32>,
    parameter_kind: Vec<u32>,
    parameter_default: Vec<f64>,
    parameter_default_factor: Vec<u32>,
    parameter_runtime_name: Vec<u32>,
    parameter_component: Vec<i32>,
    parameter_derived: Vec<u8>,
    coefficient_ir: Vec<u32>,
    coefficient_component: Vec<u32>,
    coefficient_factor: Vec<u32>,
    coherent_id: Vec<u32>,
    coherent_helicity_weight: Vec<u32>,
    coherent_all_weight: Vec<u32>,
    helicity_values: Vec<u32>,
    helicity_representative: Vec<u32>,
    helicity_coefficient: Vec<u32>,
    helicity_computed: Vec<u8>,
    helicity_zero: Vec<u8>,
    color_word: Vec<u32>,
    color_representative: Vec<u32>,
    color_coefficient: Vec<u32>,
    color_computed: Vec<u8>,
    reduction_group: Vec<u32>,
    reduction_helicity: Vec<u32>,
    reduction_color: Vec<u32>,
    contraction_metadata: EagerColorContractionMetadata,
    contraction_left: Vec<u32>,
    contraction_right: Vec<u32>,
    contraction_weight: Vec<u32>,
    contraction_symmetry: Vec<u32>,
    string_catalog: Vec<&'static str>,
    ir_catalog: Vec<&'static str>,
    semantic_limitations: Vec<&'static str>,
}

impl Fixture {
    fn valid() -> Self {
        Self {
            current_id: vec![0, 1, 2, 3],
            current_dimension: vec![1; 4],
            current_is_source: vec![1, 1, 0, 0],
            current_momentum_bitset: vec![0, 1, 2, 2],
            current_propagator_ir: vec![0; 4],
            current_propagator_kernel: vec![MISSING_U32, MISSING_U32, 30, MISSING_U32],
            source_id: vec![0, 1],
            source_current: vec![0, 1],
            source_external_label: vec![1, 2],
            source_input_momentum: vec![0, 1],
            source_ir: vec![0, 0],
            source_crossing_ir: vec![0, 0],
            source_crossing_factor: vec![0, 0],
            source_state: vec![0, 0],
            interaction_id: vec![0, 1],
            interaction_stage_size: vec![2, 3],
            interaction_left: vec![0, 2],
            interaction_right: vec![1, 1],
            interaction_result: vec![2, 3],
            interaction_coupling: vec![0, 0],
            interaction_coupling_factor: vec![0, 0],
            interaction_color_factor: vec![1, 1],
            interaction_evaluation_factor: vec![0, 0],
            interaction_group: vec![0, 1],
            interaction_ready: vec![1, 1],
            interaction_kernel: vec![10, 11],
            interaction_order: vec![[0, 1], [1, 0]],
            interaction_normalization: vec![0, 0],
            interaction_output_factor: vec![0, 0],
            root_id: vec![0],
            root_left: vec![3],
            root_right: vec![0],
            root_color_factor: vec![1],
            root_contraction_ir: vec![0],
            root_coupling: vec![0],
            root_coupling_factor: vec![0],
            root_helicity_weight: vec![0],
            root_group: vec![0],
            root_kernel: vec![40],
            root_order: vec![[0, 1]],
            root_normalization: vec![0],
            root_output_factor: vec![0],
            group_representative: vec![0, 1],
            group_start: vec![0, 1],
            group_count: vec![1, 1],
            group_members: vec![0, 1],
            momentum_slot: vec![0, 1, 2],
            momentum_bitset: vec![0, 1, 2],
            bitset_start: vec![0, 1, 2],
            bitset_count: vec![1, 1, 1],
            bitset_population: vec![1, 1, 2],
            bitset_words: vec![1, 2, 3],
            u32_sequence_start: vec![0],
            u32_sequence_count: vec![0],
            u32_sequence_values: vec![],
            i32_sequence_start: vec![0],
            i32_sequence_count: vec![2],
            i32_sequence_values: vec![-1, 1],
            factor_real: vec![1.0, 2.0, -0.0],
            factor_imaginary: vec![0.0; 3],
            factor_string: vec![0, 1, 2],
            factor_source: vec![0; 3],
            factor_ir: vec![0, 1, 2],
            factor_source_ir: vec![MISSING_U32; 3],
            coupling_factor: vec![0],
            coupling_names: vec![0],
            parameter_name: vec![],
            parameter_kind: vec![],
            parameter_default: vec![],
            parameter_default_factor: vec![],
            parameter_runtime_name: vec![],
            parameter_component: vec![],
            parameter_derived: vec![],
            coefficient_ir: vec![],
            coefficient_component: vec![],
            coefficient_factor: vec![],
            coherent_id: vec![0],
            coherent_helicity_weight: vec![0],
            coherent_all_weight: vec![0],
            helicity_values: vec![0, 0],
            helicity_representative: vec![0, 0],
            helicity_coefficient: vec![0, 2],
            helicity_computed: vec![1, 0],
            helicity_zero: vec![0, 1],
            color_word: vec![0],
            color_representative: vec![0],
            color_coefficient: vec![0],
            color_computed: vec![1],
            reduction_group: vec![0],
            reduction_helicity: vec![0],
            reduction_color: vec![0],
            contraction_metadata: EagerColorContractionMetadata {
                present: 0,
                supported: 0,
                group_count: 0,
                includes_color_factor: 0,
            },
            contraction_left: vec![],
            contraction_right: vec![],
            contraction_weight: vec![],
            contraction_symmetry: vec![],
            string_catalog: vec!["factor-0", "factor-1", "factor-2"],
            ir_catalog: vec!["{\"id\":0}", "{\"id\":1}", "{\"id\":2}"],
            semantic_limitations: vec!["binary64 source factors retain exact payload bits"],
        }
    }

    fn retained_row_count(&self, name: &str) -> u64 {
        match name {
            "bitset_ranges" => self.bitset_start.len() as u64,
            "bitset_words" => self.bitset_words.len() as u64,
            "coherent_groups" => self.coherent_id.len() as u64,
            "color_contraction_entries" => self.contraction_left.len() as u64,
            "color_contraction_metadata"
            | "metadata"
            | "lc_replay_metadata"
            | "helicity_proof_metadata"
            | "helicity_materialization_metadata" => 1,
            "color_selectors" => self.color_word.len() as u64,
            "contraction_coefficients" => self.coefficient_ir.len() as u64,
            "couplings" => self.coupling_factor.len() as u64,
            "currents" => self.current_id.len() as u64,
            "exact_factors" => self.factor_real.len() as u64,
            "helicity_selectors" => self.helicity_values.len() as u64,
            "i32_sequence_ranges" => self.i32_sequence_start.len() as u64,
            "i32_sequence_values" => self.i32_sequence_values.len() as u64,
            "interaction_group_members" => self.group_members.len() as u64,
            "interaction_groups" => self.group_start.len() as u64,
            "interactions" => self.interaction_id.len() as u64,
            "model_parameters" => self.parameter_name.len() as u64,
            "momentum_masks" => self.momentum_slot.len() as u64,
            "reduction_members" => self.reduction_group.len() as u64,
            "roots" => self.root_id.len() as u64,
            "sources" => self.source_id.len() as u64,
            "u32_sequence_ranges" => self.u32_sequence_start.len() as u64,
            "u32_sequence_values" => self.u32_sequence_values.len() as u64,
            _ => 0,
        }
    }

    fn retained_views(&self) -> Vec<EagerRetainedTableView<'static>> {
        EAGER_LOWERING_V1_TABLE_NAMES
            .iter()
            .map(|name| EagerRetainedTableView {
                name,
                row_count: self.retained_row_count(name),
                columns: &[],
            })
            .collect()
    }

    fn view<'a>(
        &'a self,
        retained_tables: &'a [EagerRetainedTableView<'a>],
    ) -> EagerLoweringInputV1View<'a> {
        EagerLoweringInputV1View {
            abi: EAGER_LOWERING_INPUT_ABI,
            process_key: "fixture",
            model_name: "fixture-model",
            string_catalog: &self.string_catalog,
            canonical_ir_catalog: &self.ir_catalog,
            semantic_limitations: &self.semantic_limitations,
            retained_tables,
            unsupported_tables: &[],
            currents: EagerCurrentColumns {
                id: &self.current_id,
                dimension: &self.current_dimension,
                is_source: &self.current_is_source,
                momentum_mask_bitset_id: &self.current_momentum_bitset,
                propagator_ir_id: &self.current_propagator_ir,
                propagator_kernel_id: &self.current_propagator_kernel,
            },
            sources: EagerSourceColumns {
                source_id: &self.source_id,
                current_id: &self.source_current,
                external_label: &self.source_external_label,
                input_momentum_slot: &self.source_input_momentum,
                source_ir_id: &self.source_ir,
                crossing_ir_id: &self.source_crossing_ir,
                crossing_factor_id: &self.source_crossing_factor,
                declared_state_index: &self.source_state,
            },
            interactions: EagerInteractionColumns {
                id: &self.interaction_id,
                stage_subset_size: &self.interaction_stage_size,
                left_current_id: &self.interaction_left,
                right_current_id: &self.interaction_right,
                result_current_id: &self.interaction_result,
                coupling_id: &self.interaction_coupling,
                coupling_factor_id: &self.interaction_coupling_factor,
                color_factor_id: &self.interaction_color_factor,
                evaluation_factor_id: &self.interaction_evaluation_factor,
                evaluation_group_id: &self.interaction_group,
                full_tensor_network_ready: &self.interaction_ready,
                kernel_id: &self.interaction_kernel,
                canonical_input_order: &self.interaction_order,
                kernel_normalization_factor_id: &self.interaction_normalization,
                output_factor_source: &self.interaction_output_factor,
            },
            roots: EagerRootColumns {
                id: &self.root_id,
                left_current_id: &self.root_left,
                right_current_id: &self.root_right,
                color_factor_id: &self.root_color_factor,
                contraction_ir_id: &self.root_contraction_ir,
                coupling_id: &self.root_coupling,
                coupling_factor_id: &self.root_coupling_factor,
                helicity_weight_factor_id: &self.root_helicity_weight,
                coherent_group_id: &self.root_group,
                kernel_id: &self.root_kernel,
                canonical_input_order: &self.root_order,
                kernel_normalization_factor_id: &self.root_normalization,
                output_factor_source: &self.root_output_factor,
            },
            interaction_groups: EagerInteractionGroupColumns {
                representative_interaction_id: &self.group_representative,
                member_start: &self.group_start,
                member_count: &self.group_count,
                member_interaction_id: &self.group_members,
            },
            momentum_masks: EagerMomentumMaskColumns {
                slot_id: &self.momentum_slot,
                bitset_id: &self.momentum_bitset,
            },
            bitsets: EagerBitsetColumns {
                start: &self.bitset_start,
                count: &self.bitset_count,
                bit_count: &self.bitset_population,
                words: &self.bitset_words,
            },
            u32_sequences: EagerU32SequenceColumns {
                start: &self.u32_sequence_start,
                count: &self.u32_sequence_count,
                values: &self.u32_sequence_values,
            },
            i32_sequences: EagerI32SequenceColumns {
                start: &self.i32_sequence_start,
                count: &self.i32_sequence_count,
                values: &self.i32_sequence_values,
            },
            exact_factors: EagerExactFactorColumns {
                real: &self.factor_real,
                imaginary: &self.factor_imaginary,
                canonical_string_id: &self.factor_string,
                exact_source: &self.factor_source,
                exact_ir_id: &self.factor_ir,
                source_ir_id: &self.factor_source_ir,
            },
            couplings: EagerCouplingColumns {
                constant_factor_id: &self.coupling_factor,
                parameter_name_ids_sequence_id: &self.coupling_names,
            },
            model_parameters: EagerModelParameterColumns {
                name_string_id: &self.parameter_name,
                kind_string_id: &self.parameter_kind,
                default_value: &self.parameter_default,
                default_factor_id: &self.parameter_default_factor,
                runtime_name_string_id: &self.parameter_runtime_name,
                complex_component: &self.parameter_component,
                derived: &self.parameter_derived,
            },
            contraction_coefficients: EagerContractionCoefficientColumns {
                contraction_ir_id: &self.coefficient_ir,
                component_index: &self.coefficient_component,
                factor_id: &self.coefficient_factor,
            },
            coherent_groups: EagerCoherentGroupColumns {
                id: &self.coherent_id,
                helicity_weight_factor_id: &self.coherent_helicity_weight,
                all_sector_weight_factor_id: &self.coherent_all_weight,
            },
            helicity_selectors: EagerHelicitySelectorColumns {
                values_sequence_id: &self.helicity_values,
                representative_sequence_id: &self.helicity_representative,
                coefficient_factor_id: &self.helicity_coefficient,
                computed: &self.helicity_computed,
                structural_zero: &self.helicity_zero,
            },
            color_selectors: EagerColorSelectorColumns {
                word_sequence_id: &self.color_word,
                representative_word_sequence_id: &self.color_representative,
                coefficient_factor_id: &self.color_coefficient,
                computed: &self.color_computed,
            },
            reduction_members: EagerReductionMemberColumns {
                coherent_group_id: &self.reduction_group,
                helicity_selector_id: &self.reduction_helicity,
                color_selector_id: &self.reduction_color,
            },
            color_contraction: self.contraction_metadata,
            color_contraction_entries: EagerColorContractionEntryColumns {
                left_group_id: &self.contraction_left,
                right_group_id: &self.contraction_right,
                weight_factor_id: &self.contraction_weight,
                symmetry_factor_id: &self.contraction_symmetry,
            },
        }
    }

    fn input(&self) -> EagerLoweringInputV1 {
        let retained_tables = self.retained_views();
        EagerLoweringInputV1::try_from_view(self.view(&retained_tables)).unwrap()
    }
}

fn assert_invalid(fixture: &Fixture, message: &str) {
    let retained_tables = fixture.retained_views();
    let error = EagerLoweringInputV1::try_from_view(fixture.view(&retained_tables)).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
    assert!(
        error.to_string().contains(message),
        "expected {:?} to contain {message:?}",
        error.to_string()
    );
}

#[test]
fn owned_input_and_lowering_preserve_semantics_and_exact_factor_ids() {
    let input = Fixture::valid().input();
    assert_eq!(input.process_key(), "fixture");
    assert_eq!(input.model_name(), "fixture-model");
    assert_eq!(input.string_catalog().len(), 3);
    assert_eq!(input.canonical_ir_catalog().len(), 3);
    assert_eq!(input.semantic_limitations().len(), 1);
    assert_eq!(input.current_count(), 4);
    assert_eq!(input.interaction_count(), 2);
    assert_eq!(input.root_count(), 1);
    assert_eq!(input.exact_factor_count(), 3);

    let plan = lower_eager_plan_v3(input).unwrap();
    assert_eq!(plan.abi(), "pyamplicol-eager-plan-v3");
    assert_eq!(plan.current_component_count(), 4);
    assert_eq!(plan.value_component_count(), 4);
    assert_eq!(plan.momentum_component_count(), 12);
    assert_eq!(plan.stages().len(), 2);
    assert_eq!(plan.invocations().len(), 2);
    assert_eq!(plan.attachments().len(), 2);
    assert_eq!(plan.finalizations().len(), 2);
    assert_eq!(plan.closures().len(), 1);
    assert_eq!(
        plan.selector_work_summary().unwrap(),
        EagerPlanSelectorWorkSummary {
            selected_flow_selector_count: 1,
            selected_flow_worst: EagerPlanActiveWorkCounts {
                current_count: 4,
                evaluation_count: 2,
                attachment_count: 2,
            },
            all_flow_selector_count: 1,
            all_flow_worst: EagerPlanActiveWorkCounts {
                current_count: 4,
                evaluation_count: 2,
                attachment_count: 2,
            },
            contracted: EagerPlanActiveWorkCounts {
                current_count: 4,
                evaluation_count: 2,
                attachment_count: 2,
            },
        }
    );

    assert_eq!(
        plan.values()
            .iter()
            .map(|row| (row.current_id, row.kind))
            .collect::<Vec<_>>(),
        vec![
            (0, EagerValueSlotKind::Source),
            (1, EagerValueSlotKind::Source),
            (2, EagerValueSlotKind::Propagated),
            (3, EagerValueSlotKind::Unpropagated),
        ]
    );
    assert_eq!(plan.invocations()[1].left_value_slot_id, 1);
    assert_eq!(plan.invocations()[1].right_value_slot_id, 2);
    assert_eq!(plan.finalizations()[0].kernel_id, 30);
    assert_eq!(plan.finalizations()[0].propagated_value_slot_id, 2);
    assert_eq!(
        plan.finalizations()[0].unpropagated_value_slot_id,
        MISSING_U32
    );

    let attachment = plan.attachments()[0];
    assert_eq!(attachment.color_factor_id, 1);
    assert_eq!(attachment.evaluation_factor_id, 0);
    assert_eq!(attachment.normalization_factor_id, 0);
    assert_eq!(attachment.representative_evaluation_factor_id, 0);
    assert_eq!(plan.exact_factors()[2].real_bits, (-0.0_f64).to_bits());

    assert_eq!(
        plan.selector_domains(),
        &[
            EagerPlanSelectorDomainRow {
                member_start: 0,
                member_count: 0,
            },
            EagerPlanSelectorDomainRow {
                member_start: 0,
                member_count: 1,
            },
        ]
    );
    assert_eq!(plan.selector_memberships(), &[0]);
    assert_eq!(plan.closures()[0].selector_domain_id, 1);
    assert_eq!(plan.helicity_selectors()[1].structural_zero, 1);

    let group = plan.reduction_groups()[0];
    assert_eq!(group.amplitude_entry_count, 1);
    assert_eq!(group.selector_entry_count, 1);
    assert_eq!(plan.reduction_entries().len(), 3);
    assert_eq!(
        plan.reduction_entries()[2],
        EagerPlanReductionEntryRow {
            kind: EagerPlanReductionEntryKind::ColorContraction,
            owner_id: MISSING_U32,
            left_id: 0,
            right_id: 0,
            factor_id: 0,
            auxiliary_factor_id: MISSING_U32,
        }
    );
    assert_eq!(plan.color_contraction_entry_range(), (2, 1));

    let shapes = plan.section_shapes().unwrap();
    assert_eq!(shapes.len(), 17);
    for shape in shapes {
        let header = shape.header().unwrap();
        assert_eq!(header.kind(), shape.kind);
        assert_eq!(header.record_size(), shape.record_size);
        assert_eq!(header.record_count(), shape.record_count);
    }
}

#[test]
fn lowering_is_byte_order_and_hash_independent_deterministic() {
    let input = Fixture::valid().input();
    let first = lower_eager_plan_v3(input.clone()).unwrap();
    let second = lower_eager_plan_v3(input).unwrap();
    assert_eq!(first, second);
}

#[test]
fn authenticated_zero_evaluation_groups_emit_no_runtime_invocation() {
    let mut fixture = Fixture::valid();
    fixture.interaction_evaluation_factor[0] = 2;

    let plan = lower_eager_plan_v3(fixture.input()).unwrap();

    assert_eq!(plan.stages().len(), 2);
    assert_eq!(plan.invocations().len(), 1);
    assert_eq!(plan.attachments().len(), 1);
    assert_eq!(plan.finalizations().len(), 2);
    assert_eq!(plan.invocations()[0].evaluation_group_id, 1);
    assert_eq!(plan.attachments()[0].interaction_id, 1);
}

#[test]
fn malformed_dense_and_cross_table_ids_fail_closed() {
    let mut dense = Fixture::valid();
    dense.current_id[2] = 99;
    assert_invalid(&dense, "current IDs must be dense");

    let mut reference = Fixture::valid();
    reference.interaction_left[0] = 99;
    assert_invalid(&reference, "interaction left current ID 99");

    let mut factor = Fixture::valid();
    factor.interaction_color_factor[0] = 99;
    assert_invalid(&factor, "color ID 99");
}

#[test]
fn malformed_group_partition_and_kernel_permutation_fail_closed() {
    let mut group = Fixture::valid();
    group.group_members[1] = 0;
    assert_invalid(&group, "representative is not its first member");

    let mut order = Fixture::valid();
    order.interaction_order[0] = [0, 0];
    assert_invalid(&order, "not a permutation");
}

#[test]
fn equivalent_group_members_may_use_different_logical_current_ids() {
    let input = Fixture::valid().input();
    let representative = input.interactions[0];
    let mut equivalent = representative;
    equivalent.left_current_id = 2;
    equivalent.right_current_id = 0;
    validate_group_signature(&representative, &equivalent, 0).unwrap();

    equivalent.kernel_id += 1;
    equivalent.coupling_id += 1;
    equivalent.canonical_input_order = [1, 0];
    validate_group_signature(&representative, &equivalent, 0).unwrap();

    equivalent.stage_subset_size += 1;
    let error = validate_group_signature(&representative, &equivalent, 0).unwrap_err();
    assert!(error.to_string().contains("crosses recursion stages"));
}

#[test]
fn retained_table_inventory_and_payloads_are_fail_closed_and_bit_exact() {
    let fixture = Fixture::valid();
    let retained_value = [-0.0_f64];
    let retained_columns = [EagerRetainedColumnView {
        name: "probe",
        elements_per_row: 1,
        values: EagerPrimitiveColumnView::F64(&retained_value),
    }];
    let mut retained_tables = fixture.retained_views();
    let metadata_index = EAGER_LOWERING_V1_TABLE_NAMES
        .iter()
        .position(|name| *name == "metadata")
        .unwrap();
    retained_tables[metadata_index] = EagerRetainedTableView {
        name: "metadata",
        row_count: 1,
        columns: &retained_columns,
    };
    let input = EagerLoweringInputV1::try_from_view(fixture.view(&retained_tables)).unwrap();
    let metadata = input.retained_tables()[metadata_index].clone();
    assert_eq!(metadata.name(), "metadata");
    assert_eq!(metadata.row_count(), 1);
    assert_eq!(metadata.columns()[0].elements_per_row(), 1);
    assert_eq!(
        metadata.columns()[0].values(),
        &EagerOwnedPrimitiveColumn::F64Bits(vec![(-0.0_f64).to_bits()])
    );
    let plan = lower_eager_plan_v3(input).unwrap();
    assert_eq!(plan.retained_tables()[metadata_index], metadata);

    let missing = &retained_tables[..retained_tables.len() - 1];
    let error = EagerLoweringInputV1::try_from_view(fixture.view(missing)).unwrap_err();
    assert!(error.to_string().contains("retained table inventory"));

    let unsupported = ["future-proof-table"];
    let mut view = fixture.view(&retained_tables);
    view.unsupported_tables = &unsupported;
    let error = EagerLoweringInputV1::try_from_view(view).unwrap_err();
    assert!(error.to_string().contains("future-proof-table"));
}

#[test]
fn forward_stage_dependency_is_rejected_during_lowering() {
    let mut fixture = Fixture::valid();
    fixture.interaction_left[0] = 3;
    let input = fixture.input();
    let error = lower_eager_plan_v3(input).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
    assert!(error.to_string().contains("before it is finalized"));
}

#[test]
fn structural_zero_requires_an_exact_zero_factor() {
    let mut fixture = Fixture::valid();
    fixture.helicity_coefficient[1] = 1;
    assert_invalid(&fixture, "structural-zero helicity selector");
}

#[test]
fn checked_u32_and_u64_overflow_are_rejected() {
    let error = count_u32(u64::from(u32::MAX) + 1, "string catalog").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("string catalog count exceeds u32")
    );

    let mut ranges = Fixture::valid();
    ranges.bitset_start = vec![0, u64::MAX, 0];
    ranges.bitset_count = vec![u64::MAX, 1, 1];
    assert_invalid(&ranges, "bitset catalog range 1 exceeds u64");

    let error = checked_usize_range(u64::MAX, 1, 0, "overflow probe").unwrap_err();
    assert!(error.to_string().contains("exceeds u64"));

    let error = usize_present_u32(u32::MAX as usize, "selector domain ID").unwrap_err();
    assert!(error.to_string().contains("missing-ID sentinel"));
}
