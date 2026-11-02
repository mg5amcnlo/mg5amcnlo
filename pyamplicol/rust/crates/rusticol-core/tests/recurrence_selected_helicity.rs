// SPDX-License-Identifier: 0BSD

#![cfg(all(feature = "f64-symjit", any(target_os = "linux", target_os = "macos")))]

//! Reuse the genuine topology-replay fixture accepted by the allocation
//! tests. No generation is performed here; the ordinary all-helicity route
//! supplies the independent resolved-component reference.

use rusticol_core::NativeRuntime;
use std::fs;
use std::path::PathBuf;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let scale = actual.abs().max(expected.abs()).max(1.0e-200);
        assert!(
            (actual - expected).abs() <= 5.0e-12 * scale,
            "selected recurrence differs: {actual:.17e} versus {expected:.17e}"
        );
    }
}

fn momenta(runtime: &NativeRuntime) -> Vec<f64> {
    let path = runtime
        .root()
        .join("processes")
        .join(&runtime.metadata().representative_process_key)
        .join("validation-momenta.json");
    let payload: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let point = payload["points"][0].as_array().unwrap();
    let one_point = point
        .iter()
        .flat_map(|leg| {
            leg["momentum"].as_array().unwrap().iter().map(|value| {
                value
                    .as_str()
                    .map(|value| value.parse::<f64>().unwrap())
                    .or_else(|| value.as_f64())
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    let mut batch = Vec::new();
    // Include a partial SIMD tile and distinct, on-shell, rotated points.
    for angle in [0.0_f64, 0.21, 0.43, 0.76, 1.07] {
        for leg in one_point.chunks_exact(4) {
            batch.extend([
                leg[0],
                angle.cos() * leg[1] + angle.sin() * leg[3],
                leg[2],
                -angle.sin() * leg[1] + angle.cos() * leg[3],
            ]);
        }
    }
    batch
}

#[test]
fn genuine_selected_recurrence_matches_unrestricted_resolved_components() {
    let Some(path) = std::env::var_os("RUSTICOL_RECURRENCE_TOPOLOGY_ALLOCATION_ARTIFACT") else {
        return;
    };
    let path = PathBuf::from(path);
    assert!(path.is_dir(), "missing fixture {}", path.display());
    let mut runtime = NativeRuntime::load(&path, None, None).unwrap();
    assert_eq!(runtime.metadata().execution_mode, "recurrence");
    let helicities = runtime.process_physics().unwrap().helicities.clone();
    let color_ids = runtime.color_ids().unwrap();
    assert!(
        helicities.len() > 2,
        "fixture must retain generic helicities"
    );
    assert!(
        color_ids.len() > 1,
        "fixture must exercise flow relabelling"
    );
    let points = momenta(&runtime);
    let point_count = 5;
    let mut chosen = helicities
        .iter()
        .enumerate()
        .filter(|(_, helicity)| helicity.computed && !helicity.structural_zero)
        .take(2)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if let Some(index) = helicities
        .iter()
        .position(|helicity| !helicity.computed && !helicity.structural_zero)
    {
        chosen.push(index);
    }
    if let Some(index) = helicities
        .iter()
        .position(|helicity| helicity.structural_zero)
    {
        chosen.push(index);
    }
    assert!(chosen.len() >= 2);

    for color_index in [0, color_ids.len() - 1] {
        let colors = [color_ids[color_index].clone()];
        let baseline = runtime
            .evaluate_resolved_f64(&points, point_count, None, Some(&colors))
            .unwrap();
        let total = runtime
            .evaluate_f64_profile_with_selectors(
                &points,
                point_count,
                None,
                Some(&colors),
                None,
                None,
            )
            .unwrap();
        let complete_helicities = helicities
            .iter()
            .map(|helicity| helicity.id.clone())
            .collect::<Vec<_>>();
        let explicit_sum = runtime
            .evaluate_f64_profile_with_selectors(
                &points,
                point_count,
                Some(&complete_helicities),
                Some(&colors),
                None,
                None,
            )
            .unwrap();
        assert_close(&explicit_sum.values, &total.values);
        assert_eq!(
            explicit_sum.profile.recurrence_closure_row_count,
            total.profile.recurrence_closure_row_count,
        );
        for &index in &chosen {
            let selected = [helicities[index].id.clone()];
            let expected = (0..point_count)
                .map(|point| baseline.values[point * helicities.len() + index])
                .collect::<Vec<_>>();
            let actual = runtime
                .evaluate_f64_with_selectors(
                    &points,
                    point_count,
                    Some(&selected),
                    Some(&colors),
                    None,
                    None,
                )
                .unwrap();
            assert_close(&actual, &expected);
            let profiled = runtime
                .evaluate_f64_profile_with_selectors(
                    &points,
                    point_count,
                    Some(&selected),
                    Some(&colors),
                    None,
                    None,
                )
                .unwrap();
            assert_close(&profiled.values, &expected);
            if expected.iter().any(|value| *value != 0.0) {
                assert!(
                    profiled.profile.recurrence_closure_row_count
                        < total.profile.recurrence_closure_row_count,
                    "selected evaluator still executes all helicity closures"
                );
            }
            let resolved = runtime
                .evaluate_resolved_f64(&points, point_count, Some(&selected), Some(&colors))
                .unwrap();
            assert_close(&resolved.values, &expected);

            // Restore the unrestricted route, then select the same cached
            // route again. Reused arenas must not leak stale amplitudes.
            let sum_again = runtime
                .evaluate_f64_with_selectors(&points, point_count, None, Some(&colors), None, None)
                .unwrap();
            assert_close(&sum_again, &total.values);
            let selected_again = runtime
                .evaluate_f64_with_selectors(
                    &points,
                    point_count,
                    Some(&selected),
                    Some(&colors),
                    None,
                    None,
                )
                .unwrap();
            assert_close(&selected_again, &expected);
        }
        let mut indices = chosen.clone();
        indices.sort_unstable();
        indices.dedup();
        let selected = indices
            .iter()
            .map(|index| helicities[*index].id.clone())
            .collect::<Vec<_>>();
        let expected = (0..point_count)
            .map(|point| {
                indices
                    .iter()
                    .map(|index| baseline.values[point * helicities.len() + index])
                    .sum()
            })
            .collect::<Vec<_>>();
        let subset = runtime
            .evaluate_f64_with_selectors(
                &points,
                point_count,
                Some(&selected),
                Some(&colors),
                None,
                None,
            )
            .unwrap();
        assert_close(&subset, &expected);

        let by_point = (0..point_count)
            .map(|point| chosen[point % chosen.len()] as u32)
            .collect::<Vec<_>>();
        let expected = by_point
            .iter()
            .enumerate()
            .map(|(point, index)| baseline.values[point * helicities.len() + *index as usize])
            .collect::<Vec<_>>();
        let point_selected = runtime
            .evaluate_f64_with_selectors(
                &points,
                point_count,
                None,
                Some(&colors),
                Some(&by_point),
                None,
            )
            .unwrap();
        assert_close(&point_selected, &expected);
        assert!(
            runtime
                .evaluate_f64_with_selectors(
                    &points,
                    point_count,
                    Some(&[]),
                    Some(&colors),
                    None,
                    None
                )
                .is_err()
        );
        assert!(
            runtime
                .evaluate_f64_with_selectors(
                    &points,
                    point_count,
                    Some(&["missing-helicity".to_string()]),
                    Some(&colors),
                    None,
                    None,
                )
                .is_err()
        );
    }
}

#[cfg(feature = "symbolica-runtime")]
#[test]
fn selected_compiled_replay_preserves_double_double_and_arb_components() {
    let Some(path) = std::env::var_os("RUSTICOL_COMPILED_SELECTED_HELICITY_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(&PathBuf::from(path), None, None).unwrap();
    assert_eq!(runtime.metadata().execution_mode, "compiled");
    let helicities = runtime.process_physics().unwrap().helicities.clone();
    let colors = runtime.color_ids().unwrap();
    assert!(helicities.len() > 2 && colors.len() > 1);
    let colors = [colors[colors.len() - 1].clone()];
    let mut chosen = helicities
        .iter()
        .enumerate()
        .filter(|(_, helicity)| helicity.computed && !helicity.structural_zero)
        .take(2)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if let Some(index) = helicities
        .iter()
        .position(|helicity| !helicity.computed && !helicity.structural_zero)
    {
        chosen.push(index);
    }
    let points = momenta(&runtime)
        .into_iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>();
    for digits in [32, 64] {
        let baseline = runtime
            .evaluate_resolved_with_precision(&points, 5, digits, None, Some(&colors))
            .unwrap();
        for &index in &chosen {
            let selected = [helicities[index].id.clone()];
            let expected = (0..5)
                .map(|point| {
                    baseline.values[point * helicities.len() + index]
                        .parse()
                        .unwrap()
                })
                .collect::<Vec<f64>>();
            let actual = runtime
                .evaluate_resolved_with_precision(
                    &points,
                    5,
                    digits,
                    Some(&selected),
                    Some(&colors),
                )
                .unwrap();
            assert_close(
                &actual
                    .values
                    .iter()
                    .map(|value| value.parse().unwrap())
                    .collect::<Vec<_>>(),
                &expected,
            );
            let sum_again = runtime
                .evaluate_resolved_with_precision(&points, 5, digits, None, Some(&colors))
                .unwrap();
            assert_eq!(sum_again.values, baseline.values);
        }
    }
}
