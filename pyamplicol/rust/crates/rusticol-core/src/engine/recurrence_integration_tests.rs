// SPDX-License-Identifier: 0BSD

//! Opt-in generated-artifact smoke tests for the recurrence execution lane.

use super::*;

fn assert_close(actual: f64, expected: f64, context: &str) {
    let tolerance = 1.0e-15 + 1.0e-12 * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: {actual:.17e} != {expected:.17e}"
    );
}

#[cfg(feature = "f64-symjit")]
fn generated_recurrence_validation_point(runtime: &NativeRuntime) -> Vec<[f64; 4]> {
    let validation_path = runtime
        .root()
        .join("processes")
        .join(&runtime.metadata().representative_process_key)
        .join("validation-momenta.json");
    let validation: Value = serde_json::from_slice(
        &fs::read(&validation_path).expect("read recurrence validation momenta"),
    )
    .expect("parse recurrence validation momenta");
    validation["points"][0]
        .as_array()
        .expect("one recurrence validation point")
        .iter()
        .map(|leg| {
            let components = leg["momentum"]
                .as_array()
                .expect("four momentum components");
            assert_eq!(
                components.len(),
                4,
                "recurrence momentum has four components"
            );
            std::array::from_fn(|index| {
                components[index]
                    .as_str()
                    .expect("decimal momentum string")
                    .parse::<f64>()
                    .expect("f64 validation momentum")
            })
        })
        .collect()
}

#[cfg(feature = "f64-symjit")]
fn assert_zero_recurrence_direct_boundary_traffic(profile: &RuntimeProfile, context: &str) {
    for (label, observed) in [
        (
            "legacy packed input bytes",
            profile.recurrence_direct_packed_input_bytes,
        ),
        (
            "legacy packed output bytes",
            profile.recurrence_direct_packed_output_bytes,
        ),
        (
            "legacy scatter bytes",
            profile.recurrence_direct_scatter_bytes,
        ),
        (
            "packet input bytes",
            profile.recurrence_direct_packet_input_bytes,
        ),
        (
            "packet output bytes",
            profile.recurrence_direct_packet_output_bytes,
        ),
        ("gather bytes", profile.recurrence_direct_gather_bytes),
        (
            "traffic scatter bytes",
            profile.recurrence_direct_traffic_scatter_bytes,
        ),
        ("remap bytes", profile.recurrence_direct_remap_bytes),
    ] {
        assert_eq!(observed, 0, "{context}: {label}");
    }
    profile
        .validate_recurrence_direct_boundary_traffic()
        .unwrap_or_else(|error| panic!("{context}: {error}"));
}

#[cfg(feature = "f64-symjit")]
#[test]
fn generated_recurrence_artifact_loads_when_fixture_is_supplied() {
    let Some(root) = std::env::var_os("RUSTICOL_RECURRENCE_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(PathBuf::from(root), None, None)
        .expect("load generated recurrence artifact through NativeRuntime");
    assert_eq!(runtime.metadata().execution_mode, "recurrence");

    let point = generated_recurrence_validation_point(&runtime);
    let momenta = point
        .iter()
        .flat_map(|momentum| momentum.iter().copied())
        .collect::<Vec<_>>();

    let values = runtime
        .evaluate_f64(&momenta, 1)
        .expect("evaluate generated recurrence artifact");
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());
    let mut direct_values = [f64::NAN];
    runtime
        .evaluate_f64_into(&momenta, 1, &mut direct_values)
        .expect("evaluate generated recurrence artifact into caller storage");
    assert_close(
        direct_values[0],
        values[0],
        "recurrence direct-output total",
    );

    let resolved = runtime
        .evaluate_resolved_f64(&momenta, 1, None, None)
        .expect("resolve generated recurrence artifact");
    assert_eq!(resolved.point_count, 1);
    assert_close(resolved.totals()[0], values[0], "recurrence resolved sum");

    let selected_color = runtime
        .color_ids()
        .expect("recurrence color metadata")
        .into_iter()
        .next()
        .expect("one recurrence color component");
    let selected = runtime
        .evaluate_resolved_f64(
            &momenta,
            1,
            None,
            Some(std::slice::from_ref(&selected_color)),
        )
        .expect("select recurrence color component");
    assert_eq!(selected.color_ids, [selected_color]);
    assert!(selected.values.iter().all(|value| value.is_finite()));
    let selected_total = runtime
        .evaluate_f64_with_selectors(&momenta, 1, None, Some(&selected.color_ids), None, None)
        .expect("evaluate selected recurrence color component");
    assert_close(
        selected_total[0],
        selected.totals()[0],
        "recurrence selected color total",
    );
    let mut selected_direct = [f64::NAN];
    runtime
        .evaluate_f64_into_with_selectors(
            &momenta,
            1,
            None,
            Some(&selected.color_ids),
            None,
            None,
            &mut selected_direct,
        )
        .expect("evaluate selected recurrence component into caller storage");
    assert_close(
        selected_direct[0],
        selected_total[0],
        "recurrence selected direct-output total",
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn generated_recurrence_odd_tails_report_zero_direct_boundary_traffic_when_fixtures_are_supplied() {
    const ODD_TAIL_POINT_COUNTS: [usize; 4] = [127, 129, 1023, 1025];
    const FIXTURES: [(&str, &str); 2] = [
        (
            "RUSTICOL_RECURRENCE_TOPOLOGY_ALLOCATION_ARTIFACT",
            "topology-replay",
        ),
        (
            "RUSTICOL_RECURRENCE_UNION_ALLOCATION_ARTIFACT",
            "all-flow-union",
        ),
    ];

    for (environment_name, layout) in FIXTURES {
        let Some(root) = std::env::var_os(environment_name) else {
            eprintln!(
                "skipping genuine recurrence {layout} boundary gate: \
                 {environment_name} is not set"
            );
            continue;
        };
        let mut runtime = NativeRuntime::load(PathBuf::from(root), None, None)
            .unwrap_or_else(|error| panic!("load genuine recurrence {layout} artifact: {error}"));
        let metadata = runtime.metadata();
        assert_eq!(
            metadata.execution_mode, "recurrence",
            "genuine {layout} fixture execution mode"
        );
        assert_eq!(
            metadata.prepared_backend.as_deref(),
            Some("jit"),
            "genuine {layout} fixture backend"
        );
        let point = generated_recurrence_validation_point(&runtime);

        for point_count in ODD_TAIL_POINT_COUNTS {
            let batch = vec![point.clone(); point_count];
            let flat_momenta = batch
                .iter()
                .flat_map(|momenta| momenta.iter().flat_map(|value| value.iter().copied()))
                .collect::<Vec<_>>();
            let mut caller_output = vec![f64::NAN; point_count];
            runtime
                .evaluate_f64_into(&flat_momenta, point_count, &mut caller_output)
                .unwrap_or_else(|error| {
                    panic!(
                        "warm genuine recurrence {layout} caller-output path at \
                         {point_count} points: {error}"
                    )
                });

            let (profiled_values, profile) = {
                let (common, execution_lane) = (&mut runtime.runtime, &mut runtime.execution_lane);
                let NativeExecutionLane::Recurrence(lane) = execution_lane else {
                    panic!("genuine {layout} fixture did not load the recurrence lane");
                };
                lane.run_f64(common, &batch).unwrap_or_else(|error| {
                    panic!(
                        "profile genuine recurrence {layout} Direct-Arena path at \
                         {point_count} points: {error}"
                    )
                })
            };
            let context = format!("genuine recurrence {layout} odd tail {point_count}");
            assert_zero_recurrence_direct_boundary_traffic(&profile, &context);
            assert_eq!(
                profiled_values.len(),
                caller_output.len(),
                "{context}: profiled output length"
            );
            for (index, (profiled, warmed)) in profiled_values
                .iter()
                .copied()
                .zip(caller_output.iter().copied())
                .enumerate()
            {
                assert_close(
                    profiled,
                    warmed,
                    &format!("{context}: caller-output parity point {index}"),
                );
            }
        }
    }
}
