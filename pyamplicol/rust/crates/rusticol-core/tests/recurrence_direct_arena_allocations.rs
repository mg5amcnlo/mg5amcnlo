// SPDX-License-Identifier: 0BSD

use rusticol_core::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectExecutorCatalog, DirectExecutorHandle,
    DirectFactorView, DirectMomentumView, DirectParameterView, DirectUnionSourceDispatchHandle,
};
use rusticol_core::recurrence::direct_runtime::DirectRecurrenceExecutionRuntime;
use rusticol_core::recurrence::{
    CheckedTableRange, ClosureExecutionProofGroupV2, ClosureProofContributionV2,
    ClosureProofMetadataV2, DIRECT_NONE_U32, DirectAmplitudeDestinationDescriptor,
    DirectClosureRow, DirectContributionRow, DirectCurrentDescriptor, DirectDestinationOperation,
    DirectExecutorRole, DirectFinalizationRow, DirectMomentumFormDescriptor, DirectMomentumTerm,
    DirectNodeKind, DirectRecurrencePlan, DirectRecurrencePlanParts, DirectReplayTargetDescriptor,
    DirectResolvedHelicityDescriptor, DirectResolvedSourceSelection, DirectRowGroupDescriptor,
    DirectSelectorDomainDescriptor, DirectSourceDispatchVariantDescriptor,
    DirectSourceEmbeddingRow, DirectSourceProjectionRow, DirectSourceRow,
    DirectSourceStateAssignment, ExactComplexRational, RecurrenceStrategy, SemanticDigest,
    closure_component_factor_digest_v2, closure_selector_domain_digest_v2,
};
use rusticol_core::{NativeRuntime, RusticolError};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

fn count_allocation(bytes: usize) {
    let tracking = TRACK_ALLOCATIONS.try_with(Cell::get).unwrap_or(false);
    if tracking {
        let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
        let _ = ALLOCATED_BYTES.try_with(|total| total.set(total.get().saturating_add(bytes)));
    }
}

fn count_allocations<T>(function: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATION_COUNT.with(|count| count.set(0));
    ALLOCATED_BYTES.with(|total| total.set(0));
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    let result = function();
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    let count = ALLOCATION_COUNT.with(Cell::get);
    let bytes = ALLOCATED_BYTES.with(Cell::get);
    (result, count, bytes)
}

const STATUS_BOUNDS: i32 = 2;
const POINT_COUNT: u32 = 4;
const EXPECTED_RE: [f64; 4] = [3.0, 6.0, 9.0, 12.0];
const EXPECTED_IM: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
const REAL_ARTIFACT_REPETITIONS: usize = 8;
const TOPOLOGY_ARTIFACT_ENV: &str = "RUSTICOL_RECURRENCE_TOPOLOGY_ALLOCATION_ARTIFACT";
const UNION_ARTIFACT_ENV: &str = "RUSTICOL_RECURRENCE_UNION_ALLOCATION_ARTIFACT";
const CONTRACTED_ARTIFACT_ENV: &str = "RUSTICOL_RECURRENCE_CONTRACTED_ALLOCATION_ARTIFACT";

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
            let source = row.momentum_form_id as usize
                * momenta.lorentz_component_count as usize
                * momenta.point_stride as usize
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

unsafe extern "C" fn fill_union_sources(
    _context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectSourceRow,
    row_count: u32,
    variants: *const DirectSourceDispatchVariantDescriptor,
    variant_count: u32,
    embeddings: *const DirectSourceEmbeddingRow,
    embedding_count: u32,
    selections: *const DirectResolvedSourceSelection,
    selection_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    let variants = unsafe { std::slice::from_raw_parts(variants, variant_count as usize) };
    let embeddings = unsafe { std::slice::from_raw_parts(embeddings, embedding_count as usize) };
    let selections = unsafe { std::slice::from_raw_parts(selections, selection_count as usize) };
    for selection in selections {
        let Some(variant) = variants.get(selection.dispatch_variant_id as usize) else {
            return STATUS_BOUNDS;
        };
        let Some(row) = rows.get(variant.source_row_id as usize) else {
            return STATUS_BOUNDS;
        };
        if row.source_slot != selection.source_slot
            || variant.crossing_exact_factor_id >= factors.value_count
        {
            return STATUS_BOUNDS;
        }
        let embedding_start = variant.embedding_start as usize;
        let Some(variant_embeddings) =
            embeddings.get(embedding_start..embedding_start + variant.embedding_count as usize)
        else {
            return STATUS_BOUNDS;
        };
        let crossing = unsafe {
            *factors
                .values_re
                .add(variant.crossing_exact_factor_id as usize)
        };
        for embedding in variant_embeddings {
            if embedding.source_component_or_sentinel == DIRECT_NONE_U32
                || embedding.exact_factor_id >= factors.value_count
            {
                return STATUS_BOUNDS;
            }
            let embedding_factor =
                unsafe { *factors.values_re.add(embedding.exact_factor_id as usize) };
            for point in 0..point_count as usize {
                let source = row.momentum_form_id as usize
                    * momenta.lorentz_component_count as usize
                    * momenta.point_stride as usize
                    + point;
                let destination = (row.destination_component_base as usize
                    + embedding.full_component as usize)
                    * arena.point_stride as usize
                    + point;
                if source >= momenta.scalar_len as usize
                    || destination >= arena.current_scalar_len as usize
                {
                    return STATUS_BOUNDS;
                }
                unsafe {
                    *arena.current_re.add(destination) =
                        *momenta.values.add(source) * crossing * embedding_factor;
                    *arena.current_im.add(destination) = 0.0;
                }
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
        let factor = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        for component in 0..usize::from(row.component_count) {
            for point in 0..point_count as usize {
                let destination =
                    (row.component_base as usize + component) * arena.point_stride as usize + point;
                if destination >= arena.current_scalar_len as usize {
                    return STATUS_BOUNDS;
                }
                unsafe {
                    *arena.current_re.add(destination) *= factor;
                    *arena.current_im.add(destination) *= factor;
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
        let factor = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
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
                *arena.amplitude_re.add(destination) += *arena.current_re.add(source) * factor;
                *arena.amplitude_im.add(destination) += *arena.current_im.add(source) * factor;
            }
        }
    }
    DIRECT_STATUS_OK
}

fn direct_runtime(strategy: RecurrenceStrategy) -> DirectRecurrenceExecutionRuntime {
    let is_union = strategy == RecurrenceStrategy::AllFlowUnion;
    let catalog_digest = SemanticDigest::new([0x33; 32]).unwrap();
    let plan = DirectRecurrencePlan::new(DirectRecurrencePlanParts {
        strategy,
        semantic_digest: SemanticDigest::new([0x11; 32]).unwrap(),
        prepared_pack_digest: SemanticDigest::new([0x22; 32]).unwrap(),
        direct_template_catalog_digest: catalog_digest,
        point_tile_size: POINT_COUNT,
        workspace_mib: 1,
        current_arena_components: 2,
        physical_sector_count: 1,
        retained_helicity_count: 1,
        amplitude_destination_count: 1,
        parameter_value_count: 1,
        external_source_count: 1,
        state_template_count: 2,
        source_template_count: 1,
        source_template_or_dispatch_count: 1,
        runtime_helicity_contract_count: u32::from(is_union),
        runtime_helicity_variant_count: u32::from(is_union),
        direct_executor_count: 4,
        currents: vec![
            DirectCurrentDescriptor {
                semantic_current_id: 0,
                node_kind: DirectNodeKind::Source,
                state_template_id: 0,
                component_base: 0,
                component_count: 1,
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
                component_base: 1,
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
            source_template_or_dispatch_domain: 0,
            spin_state_class: -1,
            exact_factor_id: 0,
            selector_domain_id: 0,
        }],
        contributions: vec![DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 1,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        }],
        finalizations: vec![DirectFinalizationRow {
            component_base: 1,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        }],
        closures: vec![DirectClosureRow {
            parent0_component_base: 1,
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
                direct_executor_id: if is_union { DIRECT_NONE_U32 } else { 0 },
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
        replay_targets: if is_union {
            vec![]
        } else {
            vec![DirectReplayTargetDescriptor {
                public_flow_id: 0,
                representative_id: 0,
                source_permutation_start: 0,
                source_permutation_count: 1,
                helicity_map_start: 0,
                helicity_map_count: 1,
                phase_exact_factor_id: 0,
                multiplicity: 1,
                selector_domain_id: 0,
            }]
        },
        source_permutations: if is_union { vec![] } else { vec![0] },
        replay_momentum_signs: if is_union { vec![] } else { vec![1] },
        replay_helicity_map: if is_union { vec![] } else { vec![0] },
        amplitude_destinations: vec![DirectAmplitudeDestinationDescriptor {
            closure_row_start: 0,
            id: 0,
            target_sector_id: 0,
            target_helicity_id_or_sentinel: if is_union { DIRECT_NONE_U32 } else { 0 },
            closure_row_count: 1,
            selector_domain_id: 0,
        }],
        resolved_helicities: vec![DirectResolvedHelicityDescriptor {
            source_state_start: 0,
            source_selection_start: 0,
            public_helicity_start: 0,
            id: 0,
            source_state_count: 1,
            source_selection_count: u32::from(is_union),
            public_helicity_count: 1,
            selector_domain_id: 0,
        }],
        source_state_assignments: vec![DirectSourceStateAssignment {
            source_slot: 0,
            state_index: 0,
        }],
        source_dispatch_variants: if is_union {
            vec![DirectSourceDispatchVariantDescriptor {
                embedding_start: 0,
                projection_start: 0,
                source_row_id: 0,
                dispatch_domain_id: 0,
                runtime_variant_id: 0,
                source_state_index: 0,
                source_template_id: 0,
                source_state_template_id: 0,
                crossed_state_template_id: 0,
                crossed_spin_state_class: -1,
                direct_executor_id: 0,
                crossing_exact_factor_id: 0,
                embedding_count: 1,
                projection_count: 1,
            }]
        } else {
            vec![]
        },
        source_embeddings: if is_union {
            vec![DirectSourceEmbeddingRow {
                full_component: 0,
                source_component_or_sentinel: 0,
                exact_factor_id: 0,
            }]
        } else {
            vec![]
        },
        source_projections: if is_union {
            vec![DirectSourceProjectionRow {
                source_component: 0,
                full_component: 0,
            }]
        } else {
            vec![]
        },
        resolved_source_selections: if is_union {
            vec![DirectResolvedSourceSelection {
                source_slot: 0,
                dispatch_variant_id: 0,
            }]
        } else {
            vec![]
        },
        public_helicities: vec![-1],
        exact_factors: vec![ExactComplexRational::ONE],
        closure_proofs: ClosureProofMetadataV2::new(
            vec![
                ClosureProofContributionV2::new(
                    0,
                    0,
                    Some(0),
                    (!is_union).then_some(0),
                    0,
                    SemanticDigest::new([0x41; 32]).unwrap(),
                    None,
                    vec![1],
                    vec![Some(1)],
                    vec![SemanticDigest::new([0x42; 32]).unwrap()],
                    vec![SemanticDigest::new([0x43; 32]).unwrap()],
                    vec![0],
                    vec![0],
                    vec![0],
                    0,
                    SemanticDigest::new([0x44; 32]).unwrap(),
                    None,
                    vec![],
                    None,
                    ExactComplexRational::ONE,
                    1,
                )
                .unwrap(),
            ],
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
            ],
            vec![],
        )
        .unwrap(),
    })
    .unwrap();

    let executors = DirectExecutorCatalog::new(
        &plan,
        catalog_digest,
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
    let mut runtime = if is_union {
        DirectRecurrenceExecutionRuntime::new_with_union_source_dispatch(
            plan,
            executors,
            4,
            DirectUnionSourceDispatchHandle {
                call: fill_union_sources,
                context: std::ptr::null(),
            },
        )
        .unwrap()
    } else {
        DirectRecurrenceExecutionRuntime::new(plan, executors, 4).unwrap()
    };
    runtime.set_parameters(&[3.0], &[1.0]).unwrap();
    runtime
}

fn topology_replay_runtime() -> DirectRecurrenceExecutionRuntime {
    direct_runtime(RecurrenceStrategy::TopologyReplay)
}

fn all_flow_union_runtime() -> DirectRecurrenceExecutionRuntime {
    direct_runtime(RecurrenceStrategy::AllFlowUnion)
}

fn external_momenta() -> [f64; 16] {
    [
        1.0, 10.0, 100.0, 1000.0, 2.0, 20.0, 200.0, 2000.0, 3.0, 30.0, 300.0, 3000.0, 4.0, 40.0,
        400.0, 4000.0,
    ]
}

#[test]
fn warmed_topology_replay_tiles_allocate_zero_heap_bytes_and_remain_correct() {
    let mut runtime = topology_replay_runtime();
    let selector = runtime.prepare_replay_selector(0).unwrap();
    let momenta = external_momenta();

    let warm_output = runtime
        .execute_replay_tile_from_external(&selector, POINT_COUNT, &momenta)
        .unwrap();
    assert_eq!(warm_output.destination_re(0).unwrap(), EXPECTED_RE);
    assert_eq!(warm_output.destination_im(0).unwrap(), EXPECTED_IM);
    let persistent_allocations = runtime.allocation_counters();
    runtime.reset_counters();

    let (result, allocation_count, allocated_bytes) = count_allocations(|| {
        let mut observed_re = [0.0; POINT_COUNT as usize];
        let mut observed_im = [0.0; POINT_COUNT as usize];
        for _ in 0..32 {
            let output = runtime.execute_replay_tile_from_external_unprofiled(
                &selector,
                POINT_COUNT,
                &momenta,
            )?;
            observed_re.copy_from_slice(output.destination_re(0).ok_or_else(|| {
                RusticolError::internal("missing direct recurrence real destination")
            })?);
            observed_im.copy_from_slice(output.destination_im(0).ok_or_else(|| {
                RusticolError::internal("missing direct recurrence imaginary destination")
            })?);
        }
        Ok::<_, RusticolError>((observed_re, observed_im))
    });

    let (observed_re, observed_im) = result.unwrap();
    assert_eq!(observed_re, EXPECTED_RE);
    assert_eq!(observed_im, EXPECTED_IM);
    assert_eq!(allocation_count, 0, "warmed recurrence loop allocated");
    assert_eq!(allocated_bytes, 0, "warmed recurrence loop allocated bytes");
    assert_eq!(runtime.counters(), Default::default());
    assert_eq!(runtime.role_timings(), Default::default());
    assert_eq!(runtime.activity_counters(), Default::default());
    assert_eq!(runtime.phase_timings(), Default::default());
    assert_eq!(runtime.traffic_counters(), Default::default());
    assert_eq!(runtime.allocation_counters(), persistent_allocations);
}

#[test]
fn warmed_all_flow_union_tiles_allocate_zero_heap_bytes_and_remain_correct() {
    let mut runtime = all_flow_union_runtime();
    let selector = runtime.prepare_union_helicity_selector(0).unwrap();
    let momenta = external_momenta();

    let warm_output = runtime
        .execute_union_tile_from_external(&selector, POINT_COUNT, &momenta)
        .unwrap();
    assert_eq!(warm_output.destination_re(0).unwrap(), EXPECTED_RE);
    assert_eq!(warm_output.destination_im(0).unwrap(), EXPECTED_IM);
    let persistent_allocations = runtime.allocation_counters();
    runtime.reset_counters();

    let (result, allocation_count, allocated_bytes) = count_allocations(|| {
        let mut observed_re = [0.0; POINT_COUNT as usize];
        let mut observed_im = [0.0; POINT_COUNT as usize];
        for _ in 0..32 {
            let output = runtime.execute_union_tile_from_external_unprofiled(
                &selector,
                POINT_COUNT,
                &momenta,
            )?;
            observed_re.copy_from_slice(output.destination_re(0).ok_or_else(|| {
                RusticolError::internal("missing direct recurrence real destination")
            })?);
            observed_im.copy_from_slice(output.destination_im(0).ok_or_else(|| {
                RusticolError::internal("missing direct recurrence imaginary destination")
            })?);
        }
        Ok::<_, RusticolError>((observed_re, observed_im))
    });

    let (observed_re, observed_im) = result.unwrap();
    assert_eq!(observed_re, EXPECTED_RE);
    assert_eq!(observed_im, EXPECTED_IM);
    assert_eq!(
        allocation_count, 0,
        "warmed union recurrence loop allocated"
    );
    assert_eq!(
        allocated_bytes, 0,
        "warmed union recurrence loop allocated bytes"
    );
    assert_eq!(runtime.counters(), Default::default());
    assert_eq!(runtime.role_timings(), Default::default());
    assert_eq!(runtime.activity_counters(), Default::default());
    assert_eq!(runtime.phase_timings(), Default::default());
    assert_eq!(runtime.traffic_counters(), Default::default());
    assert_eq!(runtime.allocation_counters(), persistent_allocations);
}

fn genuine_artifact_path(environment_name: &str) -> Option<PathBuf> {
    let Some(path) = std::env::var_os(environment_name) else {
        eprintln!("skipping genuine recurrence allocation proof: {environment_name} is not set");
        return None;
    };
    let path = PathBuf::from(path);
    assert!(
        path.is_dir(),
        "{environment_name} does not name an artifact directory: {}",
        path.display()
    );
    Some(path)
}

fn validation_momenta(runtime: &NativeRuntime) -> Vec<f64> {
    let metadata = runtime.metadata();
    let path = runtime
        .root()
        .join("processes")
        .join(&metadata.representative_process_key)
        .join("validation-momenta.json");
    let payload: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "could not read recurrence validation momenta {}: {error}",
                path.display()
            )
        }))
        .unwrap_or_else(|error| {
            panic!(
                "could not parse recurrence validation momenta {}: {error}",
                path.display()
            )
        });
    let point = payload["points"][0]
        .as_array()
        .expect("recurrence validation payload must contain one point");
    assert_eq!(
        point.len(),
        metadata.external_count,
        "recurrence validation point has the wrong external multiplicity"
    );
    point
        .iter()
        .flat_map(|leg| {
            leg["momentum"]
                .as_array()
                .expect("validation leg must contain four momentum components")
                .iter()
                .map(|component| {
                    component
                        .as_str()
                        .map(str::parse::<f64>)
                        .transpose()
                        .expect("validation momentum string must be binary64")
                        .or_else(|| component.as_f64())
                        .expect("validation momentum component must be numeric")
                })
        })
        .collect()
}

fn assert_recurrence_layout(artifact: &Path, process_id: &str, expected_layout: &str) {
    let path = artifact
        .join("processes")
        .join(process_id)
        .join("execution.json");
    let execution: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "could not read recurrence execution manifest {}: {error}",
                path.display()
            )
        }))
        .unwrap_or_else(|error| {
            panic!(
                "could not parse recurrence execution manifest {}: {error}",
                path.display()
            )
        });
    assert_eq!(
        execution["recurrence_summary"]["lc_flow_layout"].as_str(),
        Some(expected_layout),
        "genuine allocation fixture has the wrong LC layout"
    );
    assert_eq!(
        execution["required_runtime_capabilities"]
            .as_array()
            .expect("recurrence capabilities must be an array")
            .iter()
            .filter_map(serde_json::Value::as_str)
            .filter(|capability| {
                *capability == "rusticol.recurrence-direct-arena.complex-f64.v1"
            })
            .count(),
        1,
        "genuine allocation fixture must require Direct-Arena"
    );
}

fn prove_genuine_artifact_warmed_loop_allocates_nothing(artifact: PathBuf, expected_layout: &str) {
    let mut runtime =
        NativeRuntime::load(&artifact, None, None).expect("load genuine recurrence artifact");
    let metadata = runtime.metadata();
    assert_eq!(metadata.execution_mode, "recurrence");
    assert_recurrence_layout(
        &artifact,
        &metadata.representative_process_key,
        expected_layout,
    );

    // Loading prepares every replay-flow or union-helicity selector. Momentum
    // parsing, output ownership, and SymJIT descriptor creation also stay
    // outside the counted region.
    let one_point_momenta = validation_momenta(&runtime);
    for point_count in [1, 3] {
        let momenta = one_point_momenta.repeat(point_count);
        let mut output = vec![f64::NAN; point_count];
        runtime
            .evaluate_f64_into(&momenta, point_count, &mut output)
            .expect("warm genuine recurrence evaluation");
        runtime
            .evaluate_f64_into(&momenta, point_count, &mut output)
            .expect("warm genuine recurrence descriptor caches");
        let expected = output.clone();
        assert!(
            expected.iter().all(|value| value.is_finite()),
            "genuine recurrence warmup returned a non-finite total"
        );

        let (result, allocation_count, allocated_bytes) = count_allocations(|| {
            for _ in 0..REAL_ARTIFACT_REPETITIONS {
                runtime.evaluate_f64_into(&momenta, point_count, &mut output)?;
                std::hint::black_box(output.as_slice());
            }
            Ok::<(), RusticolError>(())
        });
        result.expect("repeat genuine warmed recurrence evaluation");

        assert_eq!(output, expected, "genuine recurrence output changed");
        assert_eq!(
            allocation_count, 0,
            "genuine warmed recurrence loop allocated at {point_count} points"
        );
        assert_eq!(
            allocated_bytes, 0,
            "genuine warmed recurrence loop allocated bytes at {point_count} points"
        );
    }
}

fn prove_genuine_contracted_selector_plan_allocates_nothing(artifact: PathBuf) {
    let mut runtime =
        NativeRuntime::load(&artifact, None, None).expect("load genuine recurrence artifact");
    let metadata = runtime.metadata();
    assert_eq!(metadata.execution_mode, "recurrence");
    assert_recurrence_layout(
        &artifact,
        &metadata.representative_process_key,
        "contracted-color-union",
    );
    let helicity_ids = runtime
        .helicity_ids()
        .expect("load recurrence helicity IDs");
    assert_eq!(
        helicity_ids.len(),
        1,
        "contracted allocation artifact must retain exactly one helicity"
    );
    let helicity_id = helicity_ids[0].clone();
    let selected_helicities = [helicity_id];
    let selector_plan = runtime
        .prepare_recurrence_selector_plan(Some(&selected_helicities), None)
        .expect("prepare recurrence selector plan");
    let momenta = validation_momenta(&runtime);
    let mut output = [f64::NAN; 1];
    runtime
        .evaluate_f64_into_with_recurrence_selector_plan(&selector_plan, &momenta, 1, &mut output)
        .expect("warm genuine selected recurrence evaluation");
    let expected = output;
    let mut total = [f64::NAN; 1];
    runtime
        .evaluate_f64_into(&momenta, 1, &mut total)
        .expect("evaluate genuine recurrence through the no-selector total fast path");
    assert_eq!(
        total, expected,
        "no-selector total must equal the sole retained-helicity selector"
    );

    let (result, allocation_count, allocated_bytes) = count_allocations(|| {
        for _ in 0..REAL_ARTIFACT_REPETITIONS {
            runtime.evaluate_f64_into_with_recurrence_selector_plan(
                &selector_plan,
                &momenta,
                1,
                &mut output,
            )?;
            std::hint::black_box(output.as_slice());
        }
        Ok::<(), RusticolError>(())
    });
    result.expect("repeat genuine selected recurrence evaluation");

    assert_eq!(output, expected, "selected recurrence output changed");
    assert_eq!(
        allocation_count, 0,
        "prepared recurrence selector loop allocated"
    );
    assert_eq!(
        allocated_bytes, 0,
        "prepared recurrence selector loop allocated bytes"
    );
}

#[test]
fn genuine_topology_replay_artifact_warmed_loop_allocates_zero_heap_bytes() {
    let Some(artifact) = genuine_artifact_path(TOPOLOGY_ARTIFACT_ENV) else {
        return;
    };
    prove_genuine_artifact_warmed_loop_allocates_nothing(artifact, "topology-replay");
}

#[test]
fn genuine_all_flow_union_artifact_warmed_loop_allocates_zero_heap_bytes() {
    let Some(artifact) = genuine_artifact_path(UNION_ARTIFACT_ENV) else {
        return;
    };
    prove_genuine_artifact_warmed_loop_allocates_nothing(artifact, "all-flow-union");
}

#[test]
fn genuine_contracted_color_artifact_warmed_loop_allocates_zero_heap_bytes() {
    let Some(artifact) = genuine_artifact_path(CONTRACTED_ARTIFACT_ENV) else {
        return;
    };
    prove_genuine_artifact_warmed_loop_allocates_nothing(artifact, "contracted-color-union");
}

#[test]
fn genuine_contracted_color_prepared_selector_allocates_zero_heap_bytes() {
    let Some(artifact) = genuine_artifact_path(CONTRACTED_ARTIFACT_ENV) else {
        return;
    };
    prove_genuine_contracted_selector_plan_allocates_nothing(artifact);
}
