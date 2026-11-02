// SPDX-License-Identifier: 0BSD

use super::direct_backend::*;
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
    DIRECT_NONE_U32, DirectClosureRow, DirectContributionRow, DirectExecutorRole,
    DirectFinalizationRow, DirectRecurrencePlan, DirectSourceRow,
};
use super::exact::{ExactComplexRational, ExactRational};
use crate::RusticolErrorKind;
use std::ffi::c_void;

const STATUS_BOUNDS: i32 = 2;

#[test]
fn executor_status_success_keeps_first_failure_detail_until_an_actual_failure() {
    clear_direct_executor_error_detail();
    record_direct_executor_error_detail(crate::RusticolError::integrity(
        "descriptor row 789823 is invalid",
    ));
    record_direct_executor_error_detail(crate::RusticolError::internal(
        "later detail must not replace the first failure",
    ));

    assert!(check_status(DirectExecutorRole::Source, 7, DIRECT_STATUS_OK).is_ok());

    let error = check_status(DirectExecutorRole::Contribution, 312, 4).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    assert_eq!(
        error.message(),
        "direct recurrence Contribution executor 312 failed: descriptor row 789823 is invalid"
    );

    let generic = check_status(DirectExecutorRole::Contribution, 312, 4).unwrap_err();
    assert_eq!(generic.kind(), RusticolErrorKind::Evaluation);
    assert_eq!(
        generic.message(),
        "direct recurrence Contribution executor 312 returned status 4"
    );
}

#[test]
fn executor_status_entry_clear_prevents_stale_detail_leakage() {
    record_direct_executor_error_detail(crate::RusticolError::integrity(
        "detail from an earlier schedule",
    ));
    clear_direct_executor_error_detail();

    let error = check_status(DirectExecutorRole::Closure, 41, 9).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Evaluation);
    assert_eq!(
        error.message(),
        "direct recurrence Closure executor 41 returned status 9"
    );
}

unsafe extern "C" fn fill_sources(
    _context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    _factors: DirectFactorView,
    rows: *const DirectSourceRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        for point in 0..point_count as usize {
            let source = (row.momentum_form_id as usize
                * momenta.lorentz_component_count as usize
                * momenta.point_stride as usize)
                + point;
            let destination =
                row.destination_component_base as usize * arena.point_stride as usize + point;
            if source >= momenta.scalar_len as usize
                || destination >= arena.current_scalar_len as usize
            {
                return STATUS_BOUNDS;
            }
            unsafe {
                *arena.current_re.add(destination) = *momenta.values.add(source);
                *arena.current_im.add(destination) = 0.0;
            }
        }
    }
    DIRECT_STATUS_OK
}

unsafe extern "C" fn accumulate_contributions(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        if row.exact_factor_id >= factors.value_count || parameters.value_count == 0 {
            return STATUS_BOUNDS;
        }
        let factor_re = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        let factor_im = unsafe { *factors.values_im.add(row.exact_factor_id as usize) };
        let parameter_re = unsafe { *parameters.values_re };
        let parameter_im = unsafe { *parameters.values_im };
        let scale_re = factor_re * parameter_re - factor_im * parameter_im;
        let scale_im = factor_re * parameter_im + factor_im * parameter_re;
        for point in 0..point_count as usize {
            let source = row.parent0_component_base as usize * arena.point_stride as usize + point;
            let destination =
                row.destination_component_base as usize * arena.point_stride as usize + point;
            if source >= arena.current_scalar_len as usize
                || destination >= arena.current_scalar_len as usize
            {
                return STATUS_BOUNDS;
            }
            let source_re = unsafe { *arena.current_re.add(source) };
            let source_im = unsafe { *arena.current_im.add(source) };
            unsafe {
                *arena.current_re.add(destination) += source_re * scale_re - source_im * scale_im;
                *arena.current_im.add(destination) += source_re * scale_im + source_im * scale_re;
            }
        }
    }
    DIRECT_STATUS_OK
}

unsafe extern "C" fn finalize_currents(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        if row.exact_factor_id >= factors.value_count {
            return STATUS_BOUNDS;
        }
        let scale = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        for component in 0..usize::from(row.component_count) {
            for point in 0..point_count as usize {
                let destination =
                    (row.component_base as usize + component) * arena.point_stride as usize + point;
                if destination >= arena.current_scalar_len as usize {
                    return STATUS_BOUNDS;
                }
                unsafe {
                    *arena.current_re.add(destination) *= scale;
                    *arena.current_im.add(destination) *= scale;
                }
            }
        }
    }
    DIRECT_STATUS_OK
}

unsafe extern "C" fn accumulate_closures(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectClosureRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        if row.exact_factor_id >= factors.value_count {
            return STATUS_BOUNDS;
        }
        let scale = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        for point in 0..point_count as usize {
            let source = row.parent0_component_base as usize * arena.point_stride as usize + point;
            let destination =
                row.amplitude_destination_id as usize * arena.point_stride as usize + point;
            if source >= arena.current_scalar_len as usize
                || destination >= arena.amplitude_scalar_len as usize
            {
                return STATUS_BOUNDS;
            }
            unsafe {
                *arena.amplitude_re.add(destination) += *arena.current_re.add(source) * scale;
                *arena.amplitude_im.add(destination) += *arena.current_im.add(source) * scale;
            }
        }
    }
    DIRECT_STATUS_OK
}

#[test]
fn synthetic_program_reads_rows_and_accumulates_directly_into_arenas() {
    let source_rows = [DirectSourceRow {
        source_slot: 0,
        destination_component_base: 0,
        momentum_form_id: 0,
        source_template_or_dispatch_domain: 0,
        spin_state_class: 1,
        exact_factor_id: 0,
        selector_domain_id: 0,
    }];
    let contribution_rows = [DirectContributionRow {
        parent0_component_base: 0,
        parent1_component_base_or_sentinel: DIRECT_NONE_U32,
        parent0_momentum_form_id: 0,
        parent1_momentum_form_id_or_sentinel: u32::MAX,
        destination_component_base: 1,
        exact_factor_id: 0,
        selector_domain_id: 0,
        flags: 0,
    }];
    let finalization_rows = [DirectFinalizationRow {
        component_base: 1,
        component_count: 1,
        momentum_form_id: 0,
        exact_factor_id: 1,
        selector_domain_id: 0,
        flags: 0,
    }];
    let closure_rows = [DirectClosureRow {
        parent0_component_base: 1,
        parent1_component_base_or_sentinel: DIRECT_NONE_U32,
        parent0_momentum_form_id: 0,
        parent1_momentum_form_id_or_sentinel: u32::MAX,
        amplitude_destination_id: 0,
        exact_factor_id: 2,
        component_factor_start: 0,
        component_count: 1,
        selector_domain_id: 0,
        flags: 0,
    }];
    let mut parts = super::direct_plan::tests::valid_parts();
    parts.current_arena_components = 2;
    parts.currents[0].component_count = 1;
    parts.currents[1].component_base = 1;
    parts.sources = source_rows.to_vec();
    parts.contributions = contribution_rows.to_vec();
    parts.finalizations = finalization_rows.to_vec();
    parts.closures = closure_rows.to_vec();
    parts.exact_factors = vec![ExactComplexRational::ONE; 3];
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        vec![
            DirectExecutorHandle::Source {
                call: fill_sources,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Contribution {
                call: accumulate_contributions,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Finalization {
                call: finalize_currents,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Closure {
                call: accumulate_closures,
                context: std::ptr::null(),
            },
        ],
    )
    .unwrap();

    // `DirectWorkspace` is the compatibility-facing public adapter. It keeps
    // accepting ordinary Rust slices and the pre-substrate stride contract;
    // the owned generic workspace used by the runtime enforces 64-byte planes.
    let mut current_re = [0.0; 8];
    let mut current_im = [0.0; 8];
    let mut amplitude_re = [0.0; 4];
    let mut amplitude_im = [0.0; 4];
    let momenta = [1.0, 2.0, 3.0, 4.0];
    let parameters_re = [3.0];
    let parameters_im = [0.0];
    let factors_re = [2.0, 0.5, -1.0];
    let factors_im = [0.0; 3];
    let mut workspace = DirectWorkspace {
        current_re: &mut current_re,
        current_im: &mut current_im,
        amplitude_re: &mut amplitude_re,
        amplitude_im: &mut amplitude_im,
        momenta: &momenta,
        momentum_form_count: 1,
        lorentz_component_count: 1,
        parameters_re: &parameters_re,
        parameters_im: &parameters_im,
        factors_re: &factors_re,
        factors_im: &factors_im,
        point_stride: 4,
    };
    let current_re_pointer = workspace.current_re.as_ptr();
    let amplitude_re_pointer = workspace.amplitude_re.as_ptr();
    let mut counters = DirectExecutionCounters::default();

    begin_direct_current_observation(&plan, None, 4).unwrap();
    execute_direct_plan(&plan, &executors, &mut workspace, 4, &mut counters).unwrap();
    let observation = take_direct_current_observation().unwrap();

    assert_eq!(workspace.current_re.as_ptr(), current_re_pointer);
    assert_eq!(workspace.amplitude_re.as_ptr(), amplitude_re_pointer);
    assert_eq!(workspace.current_re[0..4], momenta);
    assert_eq!(workspace.current_re[4..8], [3.0, 6.0, 9.0, 12.0]);
    assert_eq!(workspace.amplitude_re, &[-3.0, -6.0, -9.0, -12.0]);
    assert!(workspace.current_im.iter().all(|value| *value == 0.0));
    assert!(workspace.amplitude_im.iter().all(|value| *value == 0.0));
    assert_eq!(observation.plan_semantic_digest, plan.semantic_digest());
    assert_eq!(
        observation.runtime_layout_digest,
        plan.runtime_layout_digest()
    );
    assert_eq!(observation.selected_sector_id, None);
    assert_eq!(observation.point_count, 4);
    assert_eq!(
        observation.currents[&0].values,
        [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    );
    assert_eq!(
        observation.currents[&1].values,
        [(3.0, 0.0), (6.0, 0.0), (9.0, 0.0), (12.0, 0.0)]
    );
    assert_eq!(
        counters,
        DirectExecutionCounters {
            source_calls: 1,
            source_rows: 1,
            contribution_calls: 1,
            contribution_rows: 1,
            finalization_calls: 1,
            finalization_rows: 1,
            closure_calls: 1,
            closure_rows: 1,
            packed_input_bytes: 0,
            packed_output_bytes: 0,
            scatter_bytes: 0,
        }
    );
}

#[test]
fn certified_reuse_row_scale_copies_without_calling_an_interaction_executor() {
    let mut parts = super::direct_plan::tests::valid_parts();
    parts.currents[0].component_count = 1;
    parts.contributions[0] = DirectContributionRow {
        parent0_component_base: 0,
        parent1_component_base_or_sentinel: DIRECT_NONE_U32,
        // The certified-reuse encoding stores component count here.
        parent0_momentum_form_id: 1,
        parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
        destination_component_base: 2,
        exact_factor_id: 0,
        selector_domain_id: 0,
        flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
            | DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE,
    };
    parts.row_groups[1].direct_executor_id = DIRECT_NONE_U32;
    parts.finalizations[0].exact_factor_id = 1;
    parts.closures[0].exact_factor_id = 1;
    parts.closures[0].component_factor_start = 1;
    parts.exact_factors = vec![
        ExactComplexRational::new(
            ExactRational::new(2, 1).unwrap(),
            ExactRational::new(1, 1).unwrap(),
        ),
        ExactComplexRational::ONE,
    ];
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        vec![
            DirectExecutorHandle::Source {
                call: fill_sources,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Contribution {
                call: accumulate_contributions,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Finalization {
                call: finalize_currents,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Closure {
                call: accumulate_closures,
                context: std::ptr::null(),
            },
        ],
    )
    .unwrap();

    let mut current_re = [0.0; 12];
    let mut current_im = [0.0; 12];
    let mut amplitude_re = [0.0; 4];
    let mut amplitude_im = [0.0; 4];
    let momenta = [1.0, 2.0, 3.0, 4.0];
    let parameters_re = [1.0];
    let parameters_im = [0.0];
    let factors_re = [2.0, 1.0];
    let factors_im = [1.0, 0.0];
    let mut workspace = DirectWorkspace {
        current_re: &mut current_re,
        current_im: &mut current_im,
        amplitude_re: &mut amplitude_re,
        amplitude_im: &mut amplitude_im,
        momenta: &momenta,
        momentum_form_count: 1,
        lorentz_component_count: 1,
        parameters_re: &parameters_re,
        parameters_im: &parameters_im,
        factors_re: &factors_re,
        factors_im: &factors_im,
        point_stride: 4,
    };
    let mut counters = DirectExecutionCounters::default();
    execute_direct_plan(&plan, &executors, &mut workspace, 4, &mut counters).unwrap();

    assert_eq!(workspace.current_re[8..12], [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(workspace.current_im[8..12], [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(counters.contribution_calls, 1);
    assert_eq!(counters.contribution_rows, 1);
}

#[test]
fn zero_self_reuse_clears_stale_destination_before_scale_copy() {
    let mut parts = super::direct_plan::tests::valid_parts();
    parts.currents[0].component_count = 1;
    parts.contributions[0] = DirectContributionRow {
        // A certified zero current legitimately uses its own just-cleared
        // destination as the execution representative.
        parent0_component_base: 2,
        parent1_component_base_or_sentinel: DIRECT_NONE_U32,
        parent0_momentum_form_id: 1,
        parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
        destination_component_base: 2,
        exact_factor_id: 0,
        selector_domain_id: 0,
        flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
            | DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE,
    };
    parts.row_groups[1].direct_executor_id = DIRECT_NONE_U32;
    parts.finalizations[0].exact_factor_id = 1;
    parts.closures[0].exact_factor_id = 1;
    parts.closures[0].component_factor_start = 1;
    parts.exact_factors = vec![ExactComplexRational::ZERO, ExactComplexRational::ONE];
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        vec![
            DirectExecutorHandle::Source {
                call: fill_sources,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Contribution {
                call: accumulate_contributions,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Finalization {
                call: finalize_currents,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Closure {
                call: accumulate_closures,
                context: std::ptr::null(),
            },
        ],
    )
    .unwrap();

    let mut current_re = [f64::NAN; 12];
    let mut current_im = [f64::NAN; 12];
    let mut amplitude_re = [0.0; 4];
    let mut amplitude_im = [0.0; 4];
    let momenta = [1.0, 2.0, 3.0, 4.0];
    let parameters_re = [1.0];
    let parameters_im = [0.0];
    let factors_re = [0.0, 1.0];
    let factors_im = [0.0, 0.0];
    let mut workspace = DirectWorkspace {
        current_re: &mut current_re,
        current_im: &mut current_im,
        amplitude_re: &mut amplitude_re,
        amplitude_im: &mut amplitude_im,
        momenta: &momenta,
        momentum_form_count: 1,
        lorentz_component_count: 1,
        parameters_re: &parameters_re,
        parameters_im: &parameters_im,
        factors_re: &factors_re,
        factors_im: &factors_im,
        point_stride: 4,
    };
    let mut counters = DirectExecutionCounters::default();

    execute_direct_plan(&plan, &executors, &mut workspace, 4, &mut counters).unwrap();

    assert_eq!(workspace.current_re[8..12], [0.0; 4]);
    assert_eq!(workspace.current_im[8..12], [0.0; 4]);
    assert_eq!(counters.contribution_calls, 1);
    assert_eq!(counters.contribution_rows, 1);
    assert_eq!(
        plan.row_groups()[1].direct_executor_id,
        DIRECT_NONE_U32,
        "certified zero reuse must not call an interaction executor"
    );
}

#[test]
fn direct_backend_source_contains_no_eager_packing_or_batch_evaluator_route() {
    let source = include_str!("direct_backend.rs");
    for forbidden in [
        "EagerKernelInput",
        "EagerKernelBackend",
        "EagerKernelCall",
        "evaluate_batch",
    ] {
        assert!(
            !source.contains(forbidden),
            "direct backend must not contain {forbidden}"
        );
    }
}

#[test]
fn executor_catalog_authenticates_catalog_digest_and_stable_roles() {
    let plan = super::direct_plan::tests::valid_plan();
    let handles = || {
        vec![
            DirectExecutorHandle::Source {
                call: fill_sources,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Contribution {
                call: accumulate_contributions,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Finalization {
                call: finalize_currents,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Closure {
                call: accumulate_closures,
                context: std::ptr::null(),
            },
        ]
    };
    let wrong_digest = crate::recurrence::SemanticDigest::new([0x44; 32]).unwrap();
    let error = DirectExecutorCatalog::new(&plan, wrong_digest, handles())
        .err()
        .unwrap();
    assert!(error.to_string().contains("catalog digest"));

    let mut wrong_roles = handles();
    wrong_roles[1] = DirectExecutorHandle::Source {
        call: fill_sources,
        context: std::ptr::null(),
    };
    let error =
        DirectExecutorCatalog::new(&plan, plan.direct_template_catalog_digest(), wrong_roles)
            .err()
            .unwrap();
    assert!(error.to_string().contains("expected Contribution"));
}
