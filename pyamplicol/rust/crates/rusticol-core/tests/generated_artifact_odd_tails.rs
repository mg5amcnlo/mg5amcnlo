// SPDX-License-Identifier: 0BSD

#![cfg(all(feature = "f64-symjit", any(target_os = "linux", target_os = "macos")))]

use rusticol_core::{NativeResolvedEvaluation, NativeRuntime, NativeRuntimeProfile, RusticolError};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fs;
use std::path::PathBuf;

const ODD_TAIL_POINT_COUNTS: [usize; 4] = [127, 129, 1023, 1025];
const PROFILE_REPETITIONS: usize = 2;
const EXPECTED_PROCESS: &str = "d d~ > z g g g g g g";
const EXPECTED_EXTERNAL_PDGS: [i32; 9] = [1, -1, 23, 21, 21, 21, 21, 21, 21];
const INPUT_PREFIX_CANARY_BITS: u64 = 0x3fd1_2345_6789_abcd;
const INPUT_SUFFIX_CANARY_BITS: u64 = 0xbfc9_8765_4321_0fed;
const OUTPUT_PREFIX_CANARY_BITS: u64 = 0x4023_4567_89ab_cdef;
const OUTPUT_SUFFIX_CANARY_BITS: u64 = 0xc013_579b_dfb7_5311;

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
    if TRACK_ALLOCATIONS.try_with(Cell::get).unwrap_or(false) {
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

#[derive(Clone, Copy)]
enum ProfileLane {
    Compiled,
    Eager,
    RecurrenceTopology,
    RecurrenceUnion,
}

fn guarded_values(values: &[f64], prefix_bits: u64, suffix_bits: u64) -> Vec<f64> {
    let mut guarded = Vec::with_capacity(values.len() + 2);
    guarded.push(f64::from_bits(prefix_bits));
    guarded.extend_from_slice(values);
    guarded.push(f64::from_bits(suffix_bits));
    guarded
}

fn assert_guard_canaries(
    guarded: &[f64],
    logical_len: usize,
    prefix_bits: u64,
    suffix_bits: u64,
    context: &str,
) {
    assert_eq!(
        guarded.len(),
        logical_len + 2,
        "{context}: malformed guarded buffer"
    );
    assert_eq!(
        guarded[0].to_bits(),
        prefix_bits,
        "{context}: prefix canary changed"
    );
    assert_eq!(
        guarded[logical_len + 1].to_bits(),
        suffix_bits,
        "{context}: suffix canary changed"
    );
}

fn fixture_path(environment_name: &str) -> Option<PathBuf> {
    let Some(path) = std::env::var_os(environment_name) else {
        eprintln!("skipping genuine odd-tail gate: {environment_name} is not set");
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
                "could not read genuine odd-tail validation momenta {}: {error}",
                path.display()
            )
        }))
        .unwrap_or_else(|error| {
            panic!(
                "could not parse genuine odd-tail validation momenta {}: {error}",
                path.display()
            )
        });
    let point = payload["points"][0]
        .as_array()
        .expect("genuine odd-tail fixture must contain one validation point");
    assert_eq!(
        point.len(),
        metadata.external_count,
        "genuine odd-tail validation point has the wrong external multiplicity"
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

fn assert_close(actual: f64, expected: f64, context: &str) {
    let tolerance = 1.0e-15 + 1.0e-12 * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: {actual:.17e} != {expected:.17e} (tolerance {tolerance:.3e})"
    );
}

fn alternating_helicity_references(
    runtime: &mut NativeRuntime,
    point: &[f64],
    resolved: &NativeResolvedEvaluation,
) -> [(u32, f64); 2] {
    let helicities = runtime
        .helicities()
        .expect("load genuine odd-tail helicity metadata");
    let color_count = resolved.color_ids.len();
    assert!(
        color_count > 0,
        "genuine odd-tail resolved reference has no color components"
    );
    assert_eq!(
        resolved.values.len(),
        resolved.helicity_ids.len() * color_count,
        "genuine odd-tail one-point resolved shape"
    );

    let mut candidates = Vec::new();
    for (resolved_index, helicity_id) in resolved.helicity_ids.iter().enumerate() {
        let Some(helicity) = helicities.iter().find(|item| &item.id == helicity_id) else {
            panic!("resolved helicity {helicity_id:?} is absent from runtime metadata");
        };
        if helicity.structural_zero {
            continue;
        }
        let start = resolved_index * color_count;
        let expected = resolved.values[start..start + color_count]
            .iter()
            .copied()
            .sum::<f64>();
        if !expected.is_finite() || expected == 0.0 {
            continue;
        }
        candidates.push((
            u32::try_from(helicity.index)
                .expect("genuine odd-tail helicity index does not fit u32"),
            expected,
            helicity.id.clone(),
        ));
    }

    let mut selected_pair = None;
    'pairs: for (left_index, left) in candidates.iter().enumerate() {
        for right in candidates.iter().skip(left_index + 1) {
            let scale = left.1.abs().max(right.1.abs());
            let separation = (left.1 - right.1).abs();
            if separation > 64.0 * (1.0e-15 + 1.0e-12 * scale) {
                selected_pair = Some([left.clone(), right.clone()]);
                break 'pairs;
            }
        }
    }
    let selected_pair = selected_pair.unwrap_or_else(|| {
        panic!(concat!(
            "genuine odd-tail fixture has no two finite nonzero, numerically ",
            "distinguishable helicity references"
        ))
    });

    let mut verified = Vec::with_capacity(2);
    for (index, expected, helicity_id) in selected_pair {
        let selector = [helicity_id.clone()];
        let selected = runtime
            .evaluate_f64_with_selectors(point, 1, Some(&selector), None, None, None)
            .unwrap_or_else(|error| {
                panic!(
                    "evaluate genuine odd-tail scalar helicity reference {:?}: {error}",
                    helicity_id
                )
            })[0];
        assert_close(
            selected,
            expected,
            &format!(
                "genuine odd-tail selected/resolved helicity parity {:?}",
                helicity_id
            ),
        );
        verified.push((index, selected));
    }
    verified
        .try_into()
        .expect("genuine odd-tail helicity reference pair")
}

fn assert_recurrence_layout(runtime: &NativeRuntime, expected_layout: &str) {
    let metadata = runtime.metadata();
    let path = runtime
        .root()
        .join("processes")
        .join(&metadata.representative_process_key)
        .join("execution.json");
    let execution: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "could not read genuine odd-tail execution manifest {}: {error}",
                path.display()
            )
        }))
        .unwrap_or_else(|error| {
            panic!(
                "could not parse genuine odd-tail execution manifest {}: {error}",
                path.display()
            )
        });
    assert_eq!(
        execution["recurrence_summary"]["lc_flow_layout"].as_str(),
        Some(expected_layout),
        "genuine odd-tail recurrence fixture has the wrong LC layout"
    );
}

fn assert_jit_optimization_level(runtime: &NativeRuntime, expected_level: u32) {
    let path = runtime.root().join("config").join("effective.toml");
    let contents = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "could not read genuine odd-tail effective configuration {}: {error}",
            path.display()
        )
    });
    let mut section = "";
    let mut observed_backend = None;
    let mut observed_level = None;
    for line in contents.lines().map(str::trim) {
        if line.starts_with('[') {
            section = line;
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        match (section, name.trim()) {
            ("[evaluator]", "backend") => {
                observed_backend = Some(value.trim().trim_matches('"').to_string());
            }
            ("[evaluator.jit]", "optimization_level") => {
                observed_level = Some(
                    value
                        .trim()
                        .parse::<u32>()
                        .expect("JIT optimization level must be an unsigned integer"),
                );
            }
            _ => {}
        }
    }
    assert_eq!(
        observed_backend.as_deref(),
        Some("jit"),
        "genuine odd-tail fixture does not use the JIT evaluator backend"
    );
    assert_eq!(
        observed_level,
        Some(expected_level),
        "genuine odd-tail fixture has the wrong JIT optimization level"
    );
}

fn assert_zero_legacy_boundary_counters(profile: &NativeRuntimeProfile, context: &str) {
    // These are the legacy stage/amplitude and selector boundary adapters.
    // Eager DirectTable gather/fanout and P-kernel scratch traffic are internal
    // arena traffic and deliberately have separate attribution.
    for (label, observed) in [
        (
            "stage input copies",
            profile.stage_input_copy_component_count,
        ),
        (
            "stage leaf input copies",
            profile.stage_leaf_input_copy_component_count,
        ),
        (
            "stage evaluator output gathers",
            profile.stage_evaluator_output_gather_component_count,
        ),
        (
            "stage output assignments",
            profile.stage_output_assign_component_count,
        ),
        (
            "amplitude input copies",
            profile.amplitude_input_copy_component_count,
        ),
        (
            "amplitude leaf input copies",
            profile.amplitude_leaf_input_copy_component_count,
        ),
        (
            "amplitude evaluator output gathers",
            profile.amplitude_evaluator_output_gather_component_count,
        ),
        (
            "amplitude output remaps",
            profile.amplitude_output_remap_component_count,
        ),
        (
            "selector gather points",
            profile.selector_gather_point_count,
        ),
        ("selector gather bytes", profile.selector_gather_bytes),
        (
            "selector scatter values",
            profile.selector_scatter_value_count,
        ),
    ] {
        assert_eq!(observed, 0, "{context}: {label}");
    }
}

fn assert_zero_exposed_boundary_traffic(
    profile: &NativeRuntimeProfile,
    lane: ProfileLane,
    measured_input_components: u64,
    context: &str,
) {
    assert_eq!(
        profile.native_input_component_count, measured_input_components,
        "{context}: profiled input component count"
    );
    assert_eq!(
        profile.native_input_pack_bytes, 0,
        "{context}: native input packing traffic"
    );
    assert_eq!(
        profile.native_input_crossing_bytes, 0,
        "{context}: native input crossing traffic"
    );
    assert_eq!(
        profile.native_input_container_allocation_count, 0,
        "{context}: native input container allocations"
    );
    assert_eq!(
        profile.native_output_allocation_count, 0,
        "{context}: native output allocations"
    );
    assert_eq!(
        profile.observed_scratch_reallocation_count, 0,
        "{context}: warmed scratch reallocations"
    );
    assert_zero_legacy_boundary_counters(profile, context);

    match lane {
        ProfileLane::Compiled => {
            assert!(
                profile.compiled_direct_arena_engine_count > 0,
                "{context}: compiled profile did not authenticate an Arena engine"
            );
            assert!(
                profile.compiled_direct_arena_call_count > 0,
                "{context}: compiled profile observed no Arena calls"
            );
            assert_eq!(
                profile.compiled_direct_arena_boundary_input_bytes, 0,
                "{context}: compiled arena input boundary traffic"
            );
            assert_eq!(
                profile.compiled_direct_arena_boundary_current_output_bytes, 0,
                "{context}: compiled arena current-output boundary traffic"
            );
            assert_eq!(
                profile.compiled_direct_arena_boundary_amplitude_output_bytes, 0,
                "{context}: compiled arena amplitude-output boundary traffic"
            );
        }
        ProfileLane::Eager => {
            assert_eq!(
                profile.compiled_direct_arena_engine_count, 0,
                "{context}: eager profile reported a compiled Arena engine"
            );
        }
        ProfileLane::RecurrenceTopology | ProfileLane::RecurrenceUnion => {
            panic!("{context}: recurrence must not request the eager/compiled Arena profile")
        }
    }
}

fn assert_lane_activity_profile(runtime: &mut NativeRuntime, point: &[f64], lane: ProfileLane) {
    if matches!(lane, ProfileLane::Compiled) {
        return;
    }
    let point_count = *ODD_TAIL_POINT_COUNTS
        .last()
        .expect("genuine odd-tail point-count inventory");
    let momenta = point.repeat(point_count);
    let profiled = runtime
        .evaluate_f64_profile_repeated(&momenta, point_count, 1, None, None)
        .expect("profile genuine odd-tail lane activity");
    let profile = &profiled.profile;
    assert_eq!(
        profile.observed_scratch_reallocation_count, 0,
        "genuine odd-tail lane activity profile reallocated warmed scratch"
    );
    assert_zero_legacy_boundary_counters(profile, "genuine odd-tail lane activity");

    match lane {
        ProfileLane::Compiled => unreachable!("compiled lane returned above"),
        ProfileLane::Eager => {
            assert!(
                profile.evaluator_backend_call_count > 0,
                "genuine eager odd-tail profile observed no JIT backend calls"
            );
        }
        ProfileLane::RecurrenceTopology => {
            assert!(
                profile.recurrence_schedule_execution_count > 0,
                "genuine topology odd-tail profile observed no recurrence schedule"
            );
            assert!(
                profile.recurrence_replay_schedule_execution_count > 0,
                "genuine topology odd-tail profile observed no replay schedule"
            );
            assert_eq!(
                profile.recurrence_union_schedule_execution_count, 0,
                "genuine topology odd-tail profile unexpectedly used a union schedule"
            );
            assert!(
                profile.recurrence_replay_output_value_count > 0,
                "genuine topology odd-tail profile observed no replay outputs"
            );
            assert!(
                profile.recurrence_source_call_count > 0
                    && profile.recurrence_contribution_call_count > 0,
                "genuine topology odd-tail profile did not execute source and contribution kernels"
            );
        }
        ProfileLane::RecurrenceUnion => {
            assert!(
                profile.recurrence_schedule_execution_count > 0,
                "genuine union odd-tail profile observed no recurrence schedule"
            );
            assert!(
                profile.recurrence_union_schedule_execution_count > 0,
                "genuine union odd-tail profile observed no union schedule"
            );
            assert_eq!(
                profile.recurrence_replay_schedule_execution_count, 0,
                "genuine union odd-tail profile unexpectedly used a replay schedule"
            );
            assert!(
                profile.recurrence_union_source_row_count > 0,
                "genuine union odd-tail profile observed no union source rows"
            );
            assert!(
                profile.recurrence_contribution_call_count > 0,
                "genuine union odd-tail profile did not execute contribution kernels"
            );
        }
    }
}

fn prove_genuine_odd_tails(
    artifact: PathBuf,
    expected_execution_mode: &str,
    expected_jit_optimization_level: u32,
    recurrence_layout: Option<&str>,
    profile_lane: ProfileLane,
) {
    let mut runtime =
        NativeRuntime::load(&artifact, None, None).expect("load genuine odd-tail artifact");
    let metadata = runtime.metadata();
    assert_eq!(
        metadata.execution_mode, expected_execution_mode,
        "genuine odd-tail fixture has the wrong execution mode"
    );
    assert_eq!(
        metadata.process, EXPECTED_PROCESS,
        "genuine odd-tail fixture has the wrong process"
    );
    assert_eq!(
        metadata.external_pdg_order, EXPECTED_EXTERNAL_PDGS,
        "genuine odd-tail fixture has the wrong external PDG ordering"
    );
    assert!(
        metadata.current_count > 0 && metadata.interaction_count > 0,
        "genuine odd-tail fixture does not contain a generated current DAG"
    );
    match profile_lane {
        ProfileLane::Compiled => assert_eq!(
            metadata.prepared_backend, None,
            "genuine compiled odd-tail fixture unexpectedly reports a prepared backend"
        ),
        ProfileLane::Eager | ProfileLane::RecurrenceTopology | ProfileLane::RecurrenceUnion => {
            assert_eq!(
                metadata.prepared_backend.as_deref(),
                Some("jit"),
                "genuine prepared odd-tail fixture does not use the JIT backend"
            )
        }
    }
    assert_jit_optimization_level(&runtime, expected_jit_optimization_level);
    if let Some(expected_layout) = recurrence_layout {
        assert_recurrence_layout(&runtime, expected_layout);
    }

    let point = validation_momenta(&runtime);
    let owned_total = runtime
        .evaluate_f64(&point, 1)
        .expect("evaluate genuine odd-tail scalar reference")[0];
    let resolved = runtime
        .evaluate_resolved_f64(&point, 1, None, None)
        .expect("resolve genuine odd-tail scalar reference");
    let resolved_total = resolved.totals()[0];
    assert_close(
        owned_total,
        resolved_total,
        "genuine odd-tail scalar total/resolved parity",
    );
    assert!(
        owned_total.is_finite(),
        "genuine odd-tail scalar reference is non-finite"
    );
    let alternating_helicities = alternating_helicity_references(&mut runtime, &point, &resolved);

    for point_count in ODD_TAIL_POINT_COUNTS {
        let context = format!("{expected_execution_mode} point_count={point_count}");
        let logical_momenta = point.repeat(point_count);
        let guarded_momenta = guarded_values(
            &logical_momenta,
            INPUT_PREFIX_CANARY_BITS,
            INPUT_SUFFIX_CANARY_BITS,
        );
        let momenta = &guarded_momenta[1..logical_momenta.len() + 1];
        let logical_output = vec![f64::NAN; point_count];
        let mut guarded_output = guarded_values(
            &logical_output,
            OUTPUT_PREFIX_CANARY_BITS,
            OUTPUT_SUFFIX_CANARY_BITS,
        );

        for warmup_name in ["first", "descriptor"] {
            guarded_output[1..point_count + 1].fill(f64::NAN);
            runtime
                .evaluate_f64_into(
                    momenta,
                    point_count,
                    &mut guarded_output[1..point_count + 1],
                )
                .unwrap_or_else(|error| panic!("{context}: {warmup_name} warmup failed: {error}"));
            assert_guard_canaries(
                &guarded_momenta,
                logical_momenta.len(),
                INPUT_PREFIX_CANARY_BITS,
                INPUT_SUFFIX_CANARY_BITS,
                &format!("{context} {warmup_name} input"),
            );
            assert_guard_canaries(
                &guarded_output,
                point_count,
                OUTPUT_PREFIX_CANARY_BITS,
                OUTPUT_SUFFIX_CANARY_BITS,
                &format!("{context} {warmup_name} output"),
            );
            for (point_index, observed) in guarded_output[1..point_count + 1]
                .iter()
                .copied()
                .enumerate()
            {
                assert_close(
                    observed,
                    resolved_total,
                    &format!("{context} {warmup_name} numerical parity point={point_index}"),
                );
            }
        }

        guarded_output[1..point_count + 1].fill(f64::NAN);
        let (result, allocation_count, allocated_bytes) = count_allocations(|| {
            runtime.evaluate_f64_into(
                momenta,
                point_count,
                &mut guarded_output[1..point_count + 1],
            )
        });
        result.unwrap_or_else(|error: RusticolError| {
            panic!("{context}: counted caller-output evaluation failed: {error}")
        });
        assert_eq!(
            (allocation_count, allocated_bytes),
            (0, 0),
            "{context}: warmed caller-output evaluation allocated"
        );
        assert_guard_canaries(
            &guarded_momenta,
            logical_momenta.len(),
            INPUT_PREFIX_CANARY_BITS,
            INPUT_SUFFIX_CANARY_BITS,
            &format!("{context} counted input"),
        );
        assert_guard_canaries(
            &guarded_output,
            point_count,
            OUTPUT_PREFIX_CANARY_BITS,
            OUTPUT_SUFFIX_CANARY_BITS,
            &format!("{context} counted output"),
        );
        for (point_index, observed) in guarded_output[1..point_count + 1]
            .iter()
            .copied()
            .enumerate()
        {
            assert_close(
                observed,
                resolved_total,
                &format!("{context} counted numerical parity point={point_index}"),
            );
        }

        let helicity_by_point = (0..point_count)
            .map(|point_index| alternating_helicities[point_index % 2].0)
            .collect::<Vec<_>>();
        for warmup_name in ["first selector", "descriptor selector"] {
            guarded_output[1..point_count + 1].fill(f64::NAN);
            runtime
                .evaluate_f64_into_with_selectors(
                    momenta,
                    point_count,
                    None,
                    None,
                    Some(&helicity_by_point),
                    None,
                    &mut guarded_output[1..point_count + 1],
                )
                .unwrap_or_else(|error| panic!("{context}: {warmup_name} warmup failed: {error}"));
            assert_guard_canaries(
                &guarded_momenta,
                logical_momenta.len(),
                INPUT_PREFIX_CANARY_BITS,
                INPUT_SUFFIX_CANARY_BITS,
                &format!("{context} {warmup_name} input"),
            );
            assert_guard_canaries(
                &guarded_output,
                point_count,
                OUTPUT_PREFIX_CANARY_BITS,
                OUTPUT_SUFFIX_CANARY_BITS,
                &format!("{context} {warmup_name} output"),
            );
            for (point_index, observed) in guarded_output[1..point_count + 1]
                .iter()
                .copied()
                .enumerate()
            {
                assert_close(
                    observed,
                    alternating_helicities[point_index % 2].1,
                    &format!("{context} {warmup_name} numerical parity point={point_index}"),
                );
            }
        }

        guarded_output[1..point_count + 1].fill(f64::NAN);
        let (selector_result, selector_allocations, selector_bytes) = count_allocations(|| {
            runtime.evaluate_f64_into_with_selectors(
                momenta,
                point_count,
                None,
                None,
                Some(&helicity_by_point),
                None,
                &mut guarded_output[1..point_count + 1],
            )
        });
        selector_result.unwrap_or_else(|error: RusticolError| {
            panic!("{context}: counted selector evaluation failed: {error}")
        });
        assert_eq!(
            (selector_allocations, selector_bytes),
            (0, 0),
            "{context}: warmed caller-output selector evaluation allocated"
        );
        assert_guard_canaries(
            &guarded_momenta,
            logical_momenta.len(),
            INPUT_PREFIX_CANARY_BITS,
            INPUT_SUFFIX_CANARY_BITS,
            &format!("{context} counted selector input"),
        );
        assert_guard_canaries(
            &guarded_output,
            point_count,
            OUTPUT_PREFIX_CANARY_BITS,
            OUTPUT_SUFFIX_CANARY_BITS,
            &format!("{context} counted selector output"),
        );
        for (point_index, observed) in guarded_output[1..point_count + 1]
            .iter()
            .copied()
            .enumerate()
        {
            assert_close(
                observed,
                alternating_helicities[point_index % 2].1,
                &format!("{context} counted selector parity point={point_index}"),
            );
        }

        if matches!(profile_lane, ProfileLane::Compiled | ProfileLane::Eager) {
            let profiled = runtime
                .evaluate_f64_arena_profile_repeated(
                    momenta,
                    point_count,
                    PROFILE_REPETITIONS,
                    None,
                    None,
                )
                .unwrap_or_else(|error| panic!("{context}: warmed profile failed: {error}"));
            assert_eq!(
                profiled.values.len(),
                point_count,
                "{context}: profiled output length"
            );
            for (point_index, observed) in profiled.values.iter().copied().enumerate() {
                assert_close(
                    observed,
                    resolved_total,
                    &format!("{context} profiled numerical parity point={point_index}"),
                );
            }
            let measured_input_components = u64::try_from(
                momenta
                    .len()
                    .checked_mul(PROFILE_REPETITIONS)
                    .expect("profiled input component count overflowed"),
            )
            .expect("profiled input component count does not fit u64");
            assert_zero_exposed_boundary_traffic(
                &profiled.profile,
                profile_lane,
                measured_input_components,
                &context,
            );
        }
    }
    assert_lane_activity_profile(&mut runtime, &point, profile_lane);
}

#[test]
fn genuine_compiled_o3_odd_tails_preserve_numerics_allocations_and_boundary_traffic() {
    let Some(artifact) = fixture_path("RUSTICOL_COMPILED_DIRECT_ARTIFACT") else {
        return;
    };
    prove_genuine_odd_tails(artifact, "compiled", 3, None, ProfileLane::Compiled);
}

#[test]
fn genuine_eager_o2_odd_tails_preserve_numerics_allocations_and_boundary_traffic() {
    let Some(artifact) = fixture_path("RUSTICOL_EAGER_ARTIFACT") else {
        return;
    };
    prove_genuine_odd_tails(artifact, "eager", 2, None, ProfileLane::Eager);
}

#[test]
fn genuine_topology_recurrence_o2_odd_tails_preserve_numerics_and_allocations() {
    let Some(artifact) = fixture_path("RUSTICOL_RECURRENCE_TOPOLOGY_ALLOCATION_ARTIFACT") else {
        return;
    };
    prove_genuine_odd_tails(
        artifact,
        "recurrence",
        2,
        Some("topology-replay"),
        ProfileLane::RecurrenceTopology,
    );
}

#[test]
fn genuine_all_flow_union_recurrence_o2_odd_tails_preserve_numerics_and_allocations() {
    let Some(artifact) = fixture_path("RUSTICOL_RECURRENCE_UNION_ALLOCATION_ARTIFACT") else {
        return;
    };
    prove_genuine_odd_tails(
        artifact,
        "recurrence",
        2,
        Some("all-flow-union"),
        ProfileLane::RecurrenceUnion,
    );
}
