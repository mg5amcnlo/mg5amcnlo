// SPDX-License-Identifier: 0BSD

use super::plan_v3::{EagerOwnedPlanV3Sections, EagerPlanV3Sections};
use super::{EagerExecutionPlan, EagerKernelInput, EagerKernelRole, EagerKernelSpec};
use crate::{
    EagerPlanAttachmentRow, EagerPlanClosureRow, EagerPlanCouplingRow, EagerPlanCurrentRow,
    EagerPlanDirectCoefficientRow, EagerPlanExactFactorRow, EagerPlanFinalizationRow,
    EagerPlanInvocationRow, EagerPlanMomentumRow, EagerPlanReductionEntryKind,
    EagerPlanReductionEntryRow, EagerPlanReductionGroupRow, EagerPlanSelectorDomainRow,
    EagerPlanStageRow, EagerPlanValueRow, EagerValueSlotKind, MISSING_U32,
};

#[derive(Clone)]
pub(super) struct Fixture {
    kernels: Vec<EagerKernelSpec>,
    pub(super) prepared_parameter_count: u32,
    pub(super) currents: Vec<EagerPlanCurrentRow>,
    pub(super) values: Vec<EagerPlanValueRow>,
    pub(super) momenta: Vec<EagerPlanMomentumRow>,
    pub(super) stages: Vec<EagerPlanStageRow>,
    couplings: Vec<EagerPlanCouplingRow>,
    pub(super) invocations: Vec<EagerPlanInvocationRow>,
    pub(super) attachments: Vec<EagerPlanAttachmentRow>,
    pub(super) finalizations: Vec<EagerPlanFinalizationRow>,
    pub(super) closures: Vec<EagerPlanClosureRow>,
    direct_coefficients: Vec<EagerPlanDirectCoefficientRow>,
    selector_domains: Vec<EagerPlanSelectorDomainRow>,
    selector_memberships: Vec<u32>,
    reduction_groups: Vec<EagerPlanReductionGroupRow>,
    reduction_entries: Vec<EagerPlanReductionEntryRow>,
    exact_factors: Vec<EagerPlanExactFactorRow>,
}

impl Fixture {
    pub(super) fn new() -> Self {
        let exact_factors = [
            (1.0, 0.0),
            (2.0, 3.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (5.0, 0.0),
            (2.0, 0.0),
            (7.0, 0.0),
            (11.0, 0.0),
            (13.0, 0.0),
            (17.0, 0.0),
            (19.0, 0.0),
        ]
        .into_iter()
        .enumerate()
        .map(|(factor_id, (real, imaginary))| EagerPlanExactFactorRow {
            factor_id: factor_id as u32,
            real_bits: f64::to_bits(real),
            imaginary_bits: f64::to_bits(imaginary),
            canonical_string_id: 0,
            exact_source: 0,
            exact_ir_id: MISSING_U32,
            source_ir_id: MISSING_U32,
        })
        .collect();
        Self {
            kernels: vec![
                EagerKernelSpec {
                    kernel_id: 10,
                    role: EagerKernelRole::Vertex,
                    inputs: vec![
                        EagerKernelInput::FirstCurrentComponent(0),
                        EagerKernelInput::SecondCurrentComponent(0),
                        EagerKernelInput::CouplingReal,
                    ],
                    output_component_count: 1,
                    homogeneous_linear_first_current: false,
                    independent_block_size: 1,
                },
                EagerKernelSpec {
                    kernel_id: 11,
                    role: EagerKernelRole::Closure,
                    inputs: vec![
                        EagerKernelInput::FirstCurrentComponent(0),
                        EagerKernelInput::SecondCurrentComponent(0),
                        EagerKernelInput::CouplingReal,
                    ],
                    output_component_count: 1,
                    homogeneous_linear_first_current: false,
                    independent_block_size: 1,
                },
            ],
            prepared_parameter_count: 0,
            currents: (0..3)
                .map(|current_id| EagerPlanCurrentRow {
                    current_id,
                    component_start: u64::from(current_id),
                    component_count: 1,
                    momentum_slot_id: current_id,
                    flags: u32::from(current_id < 2),
                })
                .collect(),
            values: vec![
                EagerPlanValueRow {
                    value_slot_id: 0,
                    current_id: 0,
                    component_start: 0,
                    component_count: 1,
                    kind: EagerValueSlotKind::Source,
                },
                EagerPlanValueRow {
                    value_slot_id: 1,
                    current_id: 1,
                    component_start: 1,
                    component_count: 1,
                    kind: EagerValueSlotKind::Source,
                },
                EagerPlanValueRow {
                    value_slot_id: 2,
                    current_id: 2,
                    component_start: 2,
                    component_count: 1,
                    kind: EagerValueSlotKind::Unpropagated,
                },
            ],
            momenta: (0..3)
                .map(|momentum_slot_id| EagerPlanMomentumRow {
                    momentum_slot_id,
                    bitset_id: momentum_slot_id,
                    component_start: u64::from(momentum_slot_id) * 4,
                    component_count: 4,
                })
                .collect(),
            stages: vec![EagerPlanStageRow {
                stage_index: 1,
                subset_size: 2,
                invocation_start: 0,
                invocation_count: 1,
                attachment_start: 0,
                attachment_count: 1,
                finalization_start: 0,
                finalization_count: 1,
            }],
            couplings: vec![EagerPlanCouplingRow {
                coupling_id: 0,
                real_parameter_id: MISSING_U32,
                imaginary_parameter_id: MISSING_U32,
                constant_factor_id: 1,
            }],
            invocations: vec![EagerPlanInvocationRow {
                evaluation_group_id: 0,
                kernel_id: 10,
                left_value_slot_id: 0,
                right_value_slot_id: 1,
                left_momentum_slot_id: 0,
                right_momentum_slot_id: 1,
                coupling_slot_id: 0,
                output_factor_source: 0,
                attachment_start: 0,
                attachment_count: 1,
                selector_domain_id: 1,
            }],
            attachments: vec![EagerPlanAttachmentRow {
                interaction_id: 0,
                result_current_id: 2,
                color_factor_id: 2,
                evaluation_factor_id: 3,
                normalization_factor_id: 4,
                representative_evaluation_factor_id: 5,
                selector_domain_id: 1,
            }],
            finalizations: vec![EagerPlanFinalizationRow {
                kernel_id: MISSING_U32,
                current_id: 2,
                unpropagated_value_slot_id: 2,
                propagated_value_slot_id: MISSING_U32,
                momentum_slot_id: 2,
                unpropagated_selector_domain_id: 1,
                propagated_selector_domain_id: 0,
            }],
            closures: vec![EagerPlanClosureRow {
                root_id: 0,
                kernel_id: 11,
                left_value_slot_id: 2,
                right_value_slot_id: 0,
                amplitude_index: 0,
                coherent_group_id: 7,
                coupling_slot_id: 0,
                coupling_factor_id: 1,
                output_factor_source: 0,
                color_factor_id: 6,
                normalization_factor_id: 7,
                direct_coefficient_start: 0,
                direct_coefficient_count: 0,
                selector_domain_id: 1,
            }],
            direct_coefficients: Vec::new(),
            selector_domains: vec![
                EagerPlanSelectorDomainRow {
                    member_start: 0,
                    member_count: 0,
                },
                EagerPlanSelectorDomainRow {
                    member_start: 0,
                    member_count: 1,
                },
            ],
            selector_memberships: vec![7],
            reduction_groups: vec![EagerPlanReductionGroupRow {
                coherent_group_id: 7,
                amplitude_entry_start: 0,
                amplitude_entry_count: 1,
                selector_entry_start: 1,
                selector_entry_count: 0,
                helicity_weight_factor_id: 0,
                all_sector_weight_factor_id: 0,
            }],
            reduction_entries: vec![
                EagerPlanReductionEntryRow {
                    kind: EagerPlanReductionEntryKind::AmplitudeMember,
                    owner_id: 7,
                    left_id: 0,
                    right_id: MISSING_U32,
                    factor_id: MISSING_U32,
                    auxiliary_factor_id: MISSING_U32,
                },
                EagerPlanReductionEntryRow {
                    kind: EagerPlanReductionEntryKind::ColorContraction,
                    owner_id: MISSING_U32,
                    left_id: 7,
                    right_id: 7,
                    factor_id: 8,
                    auxiliary_factor_id: 9,
                },
            ],
            exact_factors,
        }
    }

    pub(super) fn sections(&self) -> EagerPlanV3Sections<'_> {
        EagerPlanV3Sections {
            kernels: &self.kernels,
            prepared_parameter_count: self.prepared_parameter_count,
            currents: &self.currents,
            values: &self.values,
            momenta: &self.momenta,
            parameters: &[],
            stages: &self.stages,
            couplings: &self.couplings,
            invocations: &self.invocations,
            attachments: &self.attachments,
            finalizations: &self.finalizations,
            closures: &self.closures,
            direct_coefficients: &self.direct_coefficients,
            selector_domains: &self.selector_domains,
            selector_memberships: &self.selector_memberships,
            reduction_groups: &self.reduction_groups,
            reduction_entries: &self.reduction_entries,
            exact_factors: &self.exact_factors,
            color_contraction_entry_start: 1,
            color_contraction_entry_count: 1,
        }
    }
}

#[test]
fn builds_final_f64_semantics_without_plan_v2_payloads() {
    let fixture = Fixture::new();
    let owned: EagerOwnedPlanV3Sections = fixture.sections().to_owned();
    let plan = owned.into_execution_plan().unwrap();

    assert_eq!(plan.value_component_count(), 3);
    assert_eq!(plan.momentum_component_count(), 12);
    assert_eq!(plan.current_component_count(), 3);
    assert_eq!(plan.stage_indices().collect::<Vec<_>>(), vec![1]);
    assert!(plan.has_selector_domains());
    assert_eq!(plan.couplings[0].constant_real, 2.0);
    assert_eq!(plan.couplings[0].constant_imag, 3.0);
    assert_eq!(plan.stages[0].attachments[0].row.factor_real, 15.0);
    assert_eq!(plan.stages[0].attachments[0].row.factor_imag, 0.0);
    assert_eq!(plan.closures[0].row.factor_real, 77.0);
    assert_eq!(plan.reduction_groups[0].amplitude_indices, vec![0]);
    assert_eq!(plan.reduction_entries[0].coefficient.re, 221.0);
    assert_eq!(plan.initial_value_ranges.len(), 2);
}

#[test]
fn preserves_direct_closure_coefficients_and_color_factor() {
    let mut fixture = Fixture::new();
    fixture.closures[0].kernel_id = MISSING_U32;
    fixture.closures[0].coupling_slot_id = MISSING_U32;
    fixture.closures[0].direct_coefficient_count = 1;
    fixture.direct_coefficients = vec![EagerPlanDirectCoefficientRow {
        contraction_ir_id: 0,
        component_index: 0,
        factor_id: 10,
    }];

    let plan = EagerExecutionPlan::from_plan_v3_sections(fixture.sections()).unwrap();
    assert!(plan.closures.is_empty());
    assert_eq!(plan.direct_closures[0].row.factor_real, 7.0);
    assert_eq!(plan.direct_closures[0].coefficients[0].re, 19.0);
}

#[test]
fn prepared_parameter_domain_is_independent_of_decoded_parameter_rows() {
    let mut fixture = Fixture::new();
    fixture.prepared_parameter_count = 1;
    fixture.couplings[0].real_parameter_id = 0;
    fixture.kernels[0]
        .inputs
        .push(EagerKernelInput::ModelParameter(0));

    let plan = EagerExecutionPlan::from_plan_v3_sections(fixture.sections()).unwrap();
    assert_eq!(plan.parameter_count(), 1);
    assert_eq!(plan.couplings[0].real_parameter_id, 0);

    fixture.prepared_parameter_count = 0;
    let error = EagerExecutionPlan::from_plan_v3_sections(fixture.sections()).unwrap_err();
    assert!(error.to_string().contains("model parameter"));
}

#[test]
fn rejects_prepared_kernel_role_mismatches() {
    let mut fixture = Fixture::new();
    fixture.kernels[0].role = EagerKernelRole::Closure;

    let error = EagerExecutionPlan::from_plan_v3_sections(fixture.sections()).unwrap_err();
    assert!(error.to_string().contains("expected Vertex"));
}

#[test]
fn rejects_duplicate_selector_domains() {
    let mut fixture = Fixture::new();
    fixture.selector_domains.push(EagerPlanSelectorDomainRow {
        member_start: 1,
        member_count: 0,
    });

    let error = EagerExecutionPlan::from_plan_v3_sections(fixture.sections()).unwrap_err();
    assert!(error.to_string().contains("duplicates an earlier domain"));
}

#[test]
fn rejects_selector_union_missing_an_attachment_dependency() {
    let mut fixture = Fixture::new();
    fixture.invocations[0].selector_domain_id = 0;

    let error = EagerExecutionPlan::from_plan_v3_sections(fixture.sections()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("selector domain does not match its dependency closure")
    );
}
