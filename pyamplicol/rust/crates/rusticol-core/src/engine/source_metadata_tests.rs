// SPDX-License-Identifier: 0BSD

use super::*;
use serde_json::{Value, json};

#[test]
fn resolved_source_inputs_match_formula_bits_for_all_families_and_crossings() {
    let external_count = 3;
    let flat = (0..9)
        .flat_map(|index| [10.0 + index as f64, 1.0, -0.0, 3.0])
        .collect::<Vec<_>>();
    let crossing = [
        InputCrossingMapEntry {
            target_index: 0,
            source_index: 2,
            sign: -1.0,
        },
        InputCrossingMapEntry {
            target_index: 1,
            source_index: 0,
            sign: -1.0,
        },
        InputCrossingMapEntry {
            target_index: 2,
            source_index: 1,
            sign: 1.0,
        },
    ];
    for (family, dimension, orientation, mass) in [
        ("scalar", 1, "self-conjugate", 0.0),
        ("fermion", 2, "particle", 0.0),
        ("fermion", 2, "antiparticle", 0.0),
        ("fermion", 4, "particle", 2.0),
        ("fermion", 4, "antiparticle", 2.0),
        ("vector", 4, "self-conjugate", 0.0),
        ("vector", 4, "self-conjugate", 2.0),
        ("spin2", 16, "self-conjugate", 2.0),
    ] {
        let mut source = source_record(900, -900, orientation, family, dimension);
        source.leg_label = 2;
        for helicity in [-1, 1] {
            for chirality in [-1, 1] {
                let state = GenericSourceStateIrManifest {
                    helicity,
                    chirality,
                    spin_state: GenericSourceSpinStateManifest::Scalar(helicity),
                };
                for phase in [[1.0, 0.0], [0.0, 1.0], [0.6, -0.8]] {
                    source.applied_crossing.phase = phase;
                    for transform in [
                        GenericMomentumTransformManifest::Identity,
                        GenericMomentumTransformManifest::NegateFourMomentum,
                    ] {
                        source.applied_crossing.momentum_transform = transform;
                        // Re-resolve changed masses, just as each compiled tile does.
                        for mass in [mass, mass * 1.5] {
                            let masses = BTreeMap::from([(-900, mass)]);
                            let resolved = sources::ResolvedSourceWavefunction::new(
                                &source,
                                &state,
                                external_count,
                                &masses,
                                dimension,
                            )
                            .unwrap();
                            for lookup in [None, Some(crossing.as_slice())] {
                                let contiguous =
                                    F64MomentumBatchView::from_contiguous_prevalidated(
                                        &flat,
                                        3,
                                        external_count,
                                        lookup,
                                    )
                                    .unwrap();
                                let nested = contiguous.materialize_nested();
                                let nested_view =
                                    F64MomentumBatchView::from_nested(&nested, external_count)
                                        .unwrap();
                                for batch in [contiguous, nested_view] {
                                    for (start, stop) in [(0, 3), (1, 2)] {
                                        let batch = batch.subview(start, stop).unwrap();
                                        for point_index in 0..batch.point_count() {
                                            let point = batch.point(point_index);
                                            let mut momentum = point.momentum(1).unwrap();
                                            if transform == GenericMomentumTransformManifest::NegateFourMomentum {
                                                momentum = negate(momentum);
                                            }
                                            let mut expected =
                                                match (family, dimension, orientation) {
                                                    ("scalar", _, _) => vec![c64(1.0, 0.0)],
                                                    ("fermion", 2, "particle") => {
                                                        ext_quark_weyl_array(
                                                            momentum, helicity, chirality,
                                                        )
                                                        .to_vec()
                                                    }
                                                    ("fermion", 2, _) => ext_antiquark_weyl_array(
                                                        momentum, helicity, chirality,
                                                    )
                                                    .to_vec(),
                                                    ("fermion", 4, "particle") => {
                                                        ext_quark_dirac_massive(
                                                            momentum, helicity, mass,
                                                        )
                                                        .to_vec()
                                                    }
                                                    ("fermion", 4, _) => {
                                                        ext_antiquark_dirac_massive(
                                                            momentum, helicity, mass,
                                                        )
                                                        .to_vec()
                                                    }
                                                    ("vector", _, _) if mass == 0.0 => {
                                                        ext_gluon(momentum, helicity).to_vec()
                                                    }
                                                    ("vector", _, _) => {
                                                        ext_massive_vector(momentum, helicity, mass)
                                                            .to_vec()
                                                    }
                                                    ("spin2", _, _) => {
                                                        ext_spin2(momentum, helicity, mass)
                                                            .unwrap()
                                                            .to_vec()
                                                    }
                                                    _ => unreachable!(),
                                                };
                                            if phase != [1.0, 0.0] {
                                                for value in &mut expected {
                                                    *value *= c64(phase[0], phase[1]);
                                                }
                                            }
                                            let mut observed =
                                                vec![c64(17.0, -19.0); dimension + 2];
                                            resolved
                                                .write(&point, &mut observed[..dimension])
                                                .unwrap();
                                            for (observed, expected) in
                                                observed.iter().zip(expected)
                                            {
                                                assert_eq!(
                                                    observed.re.to_bits(),
                                                    expected.re.to_bits()
                                                );
                                                assert_eq!(
                                                    observed.im.to_bits(),
                                                    expected.im.to_bits()
                                                );
                                            }
                                            assert!(
                                                observed[dimension..]
                                                    .iter()
                                                    .all(|&v| v == c64(17.0, -19.0))
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn resolved_source_inputs_reject_invalid_metadata_and_absent_points() {
    let mut source = source_record(25, 25, "self-conjugate", "scalar", 1);
    let state = ExecutionRuntime::default_runtime_source_state(&source).unwrap();
    let masses = BTreeMap::new();
    assert!(sources::ResolvedSourceWavefunction::new(&source, &state, 1, &masses, 2).is_err());
    source.leg_label = 0;
    assert!(sources::ResolvedSourceWavefunction::new(&source, &state, 1, &masses, 1).is_err());
    source.leg_label = 2;
    assert!(sources::ResolvedSourceWavefunction::new(&source, &state, 1, &masses, 1).is_err());
    source.leg_label = 1;
    source.source_kind = "unsupported".to_owned();
    assert!(sources::ResolvedSourceWavefunction::new(&source, &state, 1, &masses, 1).is_err());
    source.source_kind = "external-wavefunction".to_owned();
    source.source_ir.component_dimension = 3;
    assert!(sources::ResolvedSourceWavefunction::new(&source, &state, 1, &masses, 3).is_err());
    source.source_ir.component_dimension = 1;
    let resolved =
        sources::ResolvedSourceWavefunction::new(&source, &state, 1, &masses, 1).unwrap();
    let mut output = [c64(17.0, -19.0)];
    assert!(
        resolved
            .write(&Vec::<[f64; 4]>::new(), &mut output)
            .is_err()
    );
    assert_eq!(output, [c64(17.0, -19.0)]);
}

fn crossing_value(
    momentum_transform: &str,
    helicity_factor: i32,
    chirality_factor: i32,
    spin_state_factor: i32,
) -> Value {
    json!({
        "momentum_transform": momentum_transform,
        "helicity_factor": helicity_factor,
        "chirality_factor": chirality_factor,
        "spin_state_factor": spin_state_factor,
        "phase": [1.0, 0.0],
    })
}

fn source_value(
    particle_id: i32,
    anti_particle_id: i32,
    source_orientation: &str,
    wavefunction_kind: &str,
    dimension: usize,
) -> Value {
    let basis = match (wavefunction_kind, dimension) {
        ("fermion", 2) => "weyl-chiral",
        ("fermion", 4) => "dirac",
        ("scalar", _) => "scalar",
        ("vector", _) => "lorentz-vector",
        ("spin2", _) => "lorentz-tensor-rank2",
        _ => "test-basis",
    };
    let statistics = if wavefunction_kind == "fermion" {
        "fermion"
    } else {
        "boson"
    };
    let chirality = if wavefunction_kind == "fermion" { 1 } else { 0 };
    let identity_crossing = crossing_value("identity", 1, 1, 1);
    json!({
        "source_id": 0,
        "current_id": 0,
        "current_component_start": 0,
        "current_component_stop": dimension,
        "value_slot": {
            "value_slot_id": 0,
            "current_id": 0,
            "variant": "source",
            "component_start": 0,
            "component_stop": dimension,
            "dimension": dimension,
        },
        "source_parameter_start": 0,
        "source_parameter_stop": dimension,
        "leg_label": 1,
        "input_momentum_slot": 0,
        "side": "final",
        "crossing": "identity",
        "physical_pdg": particle_id,
        "outgoing_pdg": particle_id,
        "particle_id": particle_id,
        "anti_particle_id": anti_particle_id,
        "source_kind": "external-wavefunction",
        "wavefunction_kind": wavefunction_kind,
        "source_orientation": source_orientation,
        "source_basis": basis,
        "source_ir": {
            "identity": {
                "canonical_id": format!("model:test:state:{particle_id}"),
                "species_id": format!("model:test:species:{}", particle_id.abs()),
                "anti_canonical_id": format!("model:test:state:{anti_particle_id}"),
                "display_name": format!("state_{particle_id}"),
                "anti_display_name": format!("state_{anti_particle_id}"),
                "pdg_label": particle_id,
                "anti_pdg_label": anti_particle_id,
                "orientation": source_orientation,
                "self_conjugate": particle_id == anti_particle_id,
            },
            "statistics": statistics,
            "wavefunction_family": wavefunction_kind,
            "component_dimension": dimension,
            "states": [{"helicity": 1, "chirality": chirality, "spin_state": 1}],
            "crossing": identity_crossing,
            "basis": basis,
            "mass_parameter": null,
            "width_parameter": null,
        },
        "applied_crossing": crossing_value("identity", 1, 1, 1),
        "source_helicity": 1,
        "chirality": chirality,
        "spin_state": 1,
        "dimension": dimension,
        "helicity_ancestry": 1,
        "color_state": {},
    })
}

fn initial_crossed_source_value(
    particle_id: i32,
    anti_particle_id: i32,
    source_orientation: &str,
) -> Value {
    let mut source = source_value(
        particle_id,
        anti_particle_id,
        source_orientation,
        "fermion",
        2,
    );
    let crossing = crossing_value("negate-four-momentum", 1, -1, -1);
    source["side"] = json!("initial");
    source["crossing"] = json!("negate-incoming-momentum");
    source["source_ir"]["crossing"] = crossing.clone();
    source["applied_crossing"] = crossing;
    source["chirality"] = json!(-1);
    source["spin_state"] = json!(-1);
    source
}

fn source_record(
    particle_id: i32,
    anti_particle_id: i32,
    source_orientation: &str,
    wavefunction_kind: &str,
    dimension: usize,
) -> GenericSourceRecordManifest {
    serde_json::from_value(source_value(
        particle_id,
        anti_particle_id,
        source_orientation,
        wavefunction_kind,
        dimension,
    ))
    .expect("source manifest record")
}

#[test]
fn runtime_source_state_selects_declared_helicity_and_applies_factor() {
    let mut value = source_value(21, 21, "self-conjugate", "vector", 4);
    value["source_ir"]["states"] = json!([
        {"helicity": 1, "chirality": 0, "spin_state": 1},
        {"helicity": -1, "chirality": 0, "spin_state": -1},
    ]);
    let source: GenericSourceRecordManifest =
        serde_json::from_value(value).expect("runtime-selectable vector source");
    let runtime_state = ExecutionRuntime::runtime_source_state(&source, 1, c64(0.0, 1.0))
        .expect("declared runtime source state");
    assert_eq!(runtime_state.state.helicity, -1);

    let point = [[10.0, 2.0, 1.0, 9.746_794_344_808_963]];
    let masses = BTreeMap::new();
    let mut raw = vec![c64(0.0, 0.0); 4];
    ExecutionRuntime::write_source_wavefunction_with_state(
        &source,
        &runtime_state.state,
        1,
        &masses,
        &point,
        &mut raw,
    )
    .expect("raw selected source");
    let mut factored = vec![c64(0.0, 0.0); 4];
    ExecutionRuntime::fill_sources_row_with_states(
        std::slice::from_ref(&source),
        std::slice::from_ref(&runtime_state),
        1,
        &masses,
        &mut factored,
        &point,
    )
    .expect("factored selected source");
    for (raw_component, factored_component) in raw.iter().zip(&factored) {
        assert_eq!(*factored_component, *raw_component * c64(0.0, 1.0));
    }
}

#[test]
fn inactive_runtime_source_state_zero_fills_its_slot() {
    let source = source_record(21, 21, "self-conjugate", "vector", 4);
    let runtime_state = ExecutionRuntime::inactive_runtime_source_state(&source)
        .expect("inactive runtime source state");
    let point = [[10.0, 2.0, 1.0, 9.746_794_344_808_963]];
    let masses = BTreeMap::new();
    let mut row = vec![c64(7.0, -3.0); 4];
    ExecutionRuntime::fill_sources_row_with_states(
        std::slice::from_ref(&source),
        std::slice::from_ref(&runtime_state),
        1,
        &masses,
        &mut row,
        &point,
    )
    .expect("zero-filled inactive source");
    assert!(row.iter().all(|value| *value == c64(0.0, 0.0)));
}

#[test]
fn runtime_source_helicity_lookup_is_exact_and_rejects_ambiguity() {
    let mut value = source_value(21, 21, "self-conjugate", "vector", 4);
    value["source_ir"]["states"] = json!([
        {"helicity": 1, "chirality": 0, "spin_state": 1},
        {"helicity": -1, "chirality": 0, "spin_state": -1},
    ]);
    let source: GenericSourceRecordManifest =
        serde_json::from_value(value.clone()).expect("runtime-selectable vector source");
    assert_eq!(
        ExecutionRuntime::runtime_source_state_index_for_helicity(&source, 1).unwrap(),
        Some(0)
    );
    assert_eq!(
        ExecutionRuntime::runtime_source_state_index_for_helicity(&source, -1).unwrap(),
        Some(1)
    );
    assert_eq!(
        ExecutionRuntime::runtime_source_state_index_for_helicity(&source, 0).unwrap(),
        None
    );

    value["source_ir"]["states"] = json!([
        {"helicity": 1, "chirality": -1, "spin_state": -1},
        {"helicity": 1, "chirality": 1, "spin_state": 1},
    ]);
    let ambiguous: GenericSourceRecordManifest =
        serde_json::from_value(value).expect("same-helicity source states");
    assert!(
        ExecutionRuntime::runtime_source_state_index_for_helicity(&ambiguous, 1)
            .unwrap_err()
            .to_string()
            .contains("multiple runtime states")
    );
}

#[test]
fn source_manifest_requires_typed_source_and_crossing_records() {
    for field in ["source_ir", "applied_crossing"] {
        let mut missing = source_value(810_001, -810_001, "particle", "fermion", 2);
        missing
            .as_object_mut()
            .expect("source object")
            .remove(field);

        let error = serde_json::from_value::<GenericSourceRecordManifest>(missing).unwrap_err();

        assert!(error.to_string().contains(field));
    }
}

#[test]
fn source_manifest_requires_complete_strict_source_ir() {
    let mut missing_orientation = source_value(810_001, -810_001, "particle", "fermion", 2);
    missing_orientation["source_ir"]["identity"]
        .as_object_mut()
        .expect("identity object")
        .remove("orientation");
    let error =
        serde_json::from_value::<GenericSourceRecordManifest>(missing_orientation).unwrap_err();
    assert!(error.to_string().contains("orientation"));

    let mut missing_nullable = source_value(810_001, -810_001, "particle", "fermion", 2);
    missing_nullable["source_ir"]
        .as_object_mut()
        .expect("SourceIR object")
        .remove("mass_parameter");
    let error =
        serde_json::from_value::<GenericSourceRecordManifest>(missing_nullable).unwrap_err();
    assert!(error.to_string().contains("mass_parameter"));

    let mut unknown = source_value(810_001, -810_001, "particle", "fermion", 2);
    unknown["source_ir"]["identity"]["taxonomy"] = json!("sm-fermion");
    let error = serde_json::from_value::<GenericSourceRecordManifest>(unknown).unwrap_err();
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn pre_source_ir_schema_v3_reports_actionable_regeneration_error() {
    let mut missing = source_value(810_001, -810_001, "particle", "fermion", 2);
    missing.as_object_mut().unwrap().remove("source_ir");
    let parse_error = serde_json::from_value::<GenericSourceRecordManifest>(missing).unwrap_err();

    let error = execution_manifest_parse_error("processes/example/execution.json", parse_error);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("typed source metadata"));
    assert!(error.to_string().contains("regenerate"));
}

#[test]
fn repeated_source_records_must_share_one_canonical_source_ir() {
    let first = source_record(810_001, -810_001, "particle", "fermion", 2);
    let mut second = source_record(810_001, -810_001, "particle", "fermion", 2);
    second.source_id = 1;
    second.current_id = 1;
    second.source_ir.crossing.phase = [0.0, 1.0];
    let mut canonical_by_slot = BTreeMap::new();
    let mut canonical_by_identity = BTreeMap::new();

    validate_consistent_source_ir(
        &mut canonical_by_slot,
        &mut canonical_by_identity,
        0,
        &first,
    )
    .expect("first source is canonical");
    let error = validate_consistent_source_ir(
        &mut canonical_by_slot,
        &mut canonical_by_identity,
        1,
        &second,
    )
    .unwrap_err();

    assert!(error.to_string().contains("canonical SourceIR"));
    assert!(error.to_string().contains("momentum slot"));
}

#[test]
fn separate_legs_share_one_source_ir_per_oriented_particle() {
    let first = source_record(810_001, -810_001, "particle", "fermion", 2);
    let mut second = source_record(810_001, -810_001, "particle", "fermion", 2);
    second.source_id = 1;
    second.current_id = 1;
    second.input_momentum_slot = 1;
    second.source_ir.crossing.phase = [0.0, 1.0];
    let mut canonical_by_slot = BTreeMap::new();
    let mut canonical_by_identity = BTreeMap::new();

    validate_consistent_source_ir(
        &mut canonical_by_slot,
        &mut canonical_by_identity,
        0,
        &first,
    )
    .expect("first source is canonical");
    let error = validate_consistent_source_ir(
        &mut canonical_by_slot,
        &mut canonical_by_identity,
        1,
        &second,
    )
    .unwrap_err();

    assert!(error.to_string().contains("canonical SourceIR"));
    assert!(error.to_string().contains("oriented particle"));
}

#[test]
fn represented_antiparticle_identities_must_form_an_involution() {
    let particle = source_record(810_001, -810_001, "particle", "fermion", 2);
    let mut antiparticle = source_record(-810_001, 810_001, "antiparticle", "fermion", 2);
    antiparticle.source_id = 1;
    antiparticle.current_id = 1;
    antiparticle.input_momentum_slot = 1;
    let mut canonical_by_slot = BTreeMap::new();
    let mut canonical_by_identity = BTreeMap::new();

    validate_consistent_source_ir(
        &mut canonical_by_slot,
        &mut canonical_by_identity,
        0,
        &particle,
    )
    .expect("particle source is valid");
    validate_consistent_source_ir(
        &mut canonical_by_slot,
        &mut canonical_by_identity,
        1,
        &antiparticle,
    )
    .expect("valid antiparticle relation is accepted");

    let mut inconsistent = antiparticle;
    inconsistent.source_ir.identity.species_id = "model:test:species:other".to_string();
    let mut canonical_by_slot = BTreeMap::new();
    let mut canonical_by_identity = BTreeMap::new();
    validate_consistent_source_ir(
        &mut canonical_by_slot,
        &mut canonical_by_identity,
        0,
        &particle,
    )
    .expect("particle source is valid");
    let error = validate_consistent_source_ir(
        &mut canonical_by_slot,
        &mut canonical_by_identity,
        1,
        &inconsistent,
    )
    .unwrap_err();

    assert!(error.to_string().contains("non-involutive"));
}

#[test]
fn source_manifest_validation_rejects_self_conjugate_fermions() {
    let source = source_record(810_001, 810_001, "self-conjugate", "fermion", 2);

    let error = validate_source_wavefunction_metadata(0, &source).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("unsupported self-conjugate fermion source")
    );
}

#[test]
fn source_manifest_validation_rejects_family_dimension_mismatches() {
    let source = source_record(910_101, 910_101, "self-conjugate", "vector", 2);

    let error = validate_source_wavefunction_metadata(0, &source).unwrap_err();

    assert!(error.to_string().contains("wavefunction kind \"vector\""));
}

#[test]
fn source_manifest_validation_checks_flattened_projections() {
    let mut value = source_value(810_001, -810_001, "particle", "fermion", 2);
    value["wavefunction_kind"] = json!("scalar");
    let source = serde_json::from_value(value).expect("source manifest record");

    let error = validate_source_wavefunction_metadata(0, &source).unwrap_err();

    assert!(error.to_string().contains("\"wavefunction_kind\""));
}

#[test]
fn source_manifest_validation_checks_statistics_against_family() {
    let mut value = source_value(810_001, -810_001, "particle", "fermion", 2);
    value["source_ir"]["statistics"] = json!("boson");
    let source = serde_json::from_value(value).expect("source manifest record");

    let error = validate_source_wavefunction_metadata(0, &source).unwrap_err();

    assert!(error.to_string().contains("statistics disagree"));
}

#[test]
fn source_manifest_validation_enforces_side_dependent_crossing() {
    let mut final_value = source_value(810_001, -810_001, "particle", "fermion", 2);
    final_value["crossing"] = json!("negate-incoming-momentum");
    final_value["applied_crossing"] = crossing_value("negate-four-momentum", 1, 1, 1);
    let final_source = serde_json::from_value(final_value).expect("final source record");
    let error = validate_source_wavefunction_metadata(0, &final_source).unwrap_err();
    assert!(error.to_string().contains("final-state applied crossing"));

    let mut initial_value = initial_crossed_source_value(810_001, -810_001, "particle");
    initial_value["crossing"] = json!("identity");
    initial_value["applied_crossing"] = crossing_value("identity", 1, 1, 1);
    let initial_source = serde_json::from_value(initial_value).expect("initial source record");
    let error = validate_source_wavefunction_metadata(0, &initial_source).unwrap_err();
    assert!(error.to_string().contains("declared SourceIR crossing"));
}

#[test]
fn source_manifest_validation_checks_crossed_current_state() {
    let mut value = initial_crossed_source_value(810_001, -810_001, "particle");
    value["chirality"] = json!(1);
    let source = serde_json::from_value(value).expect("initial source record");

    let error = validate_source_wavefunction_metadata(0, &source).unwrap_err();

    assert!(error.to_string().contains("current state is not declared"));
}

#[test]
fn source_manifest_validation_rejects_zero_and_nonfinite_crossing_phases() {
    for phase in [[0.0, 0.0], [f64::INFINITY, 0.0]] {
        let mut source: GenericSourceRecordManifest =
            serde_json::from_value(initial_crossed_source_value(810_001, -810_001, "particle"))
                .expect("initial source record");
        source.source_ir.crossing.phase = phase;
        source.applied_crossing.phase = phase;

        let error = validate_source_wavefunction_metadata(0, &source).unwrap_err();

        assert!(error.to_string().contains("invalid declared CrossingIR"));
    }
}

#[test]
fn f64_sources_use_typed_orientation_instead_of_flattened_metadata() {
    let mut source = source_record(810_001, -810_001, "antiparticle", "fermion", 2);
    source.source_orientation = GenericSourceOrientationManifest::Particle;
    let point = [[5.0, 3.0, 4.0, 0.0]];
    let mut output = [c64(0.0, 0.0); 2];

    ExecutionRuntime::write_source_wavefunction(&source, 1, &BTreeMap::new(), &point, &mut output)
        .expect("evaluate relabeled antiparticle source");

    assert_eq!(output, ext_antiquark_weyl_array(point[0], 1, 1));
    assert_ne!(output, ext_quark_weyl_array(point[0], 1, 1));
}

#[test]
fn f64_sources_use_typed_family_dimension_and_pdg_relation() {
    let mut source = source_record(810_001, 910_002, "particle", "vector", 4);
    source.wavefunction_kind = "scalar".to_owned();
    source.dimension = 1;
    source.particle_id = -810_001;
    source.anti_particle_id = -810_001;
    let point = [[13.0, 0.0, 0.0, 12.0]];
    let masses = BTreeMap::from([(-810_001, 99.0), (910_002, 5.0)]);
    let mut output = [c64(0.0, 0.0); 4];

    ExecutionRuntime::write_source_wavefunction(&source, 1, &masses, &point, &mut output)
        .expect("evaluate source with typed SourceIR metadata");

    assert_eq!(output, ext_massive_vector(point[0], 1, 5.0));
}

#[test]
fn f64_sources_use_applied_crossing_momentum_transform() {
    let mut source: GenericSourceRecordManifest =
        serde_json::from_value(initial_crossed_source_value(810_001, -810_001, "particle"))
            .expect("initial source record");
    source.crossing = "identity".to_owned();
    let point = [[5.0, 3.0, 4.0, 0.0]];
    let mut output = [c64(0.0, 0.0); 2];

    ExecutionRuntime::write_source_wavefunction(&source, 1, &BTreeMap::new(), &point, &mut output)
        .expect("evaluate crossed source");

    assert_eq!(output, ext_quark_weyl_array(negate(point[0]), 1, -1));
}

#[test]
fn f64_sources_apply_nontrivial_crossing_phase_to_every_component() {
    let mut value = initial_crossed_source_value(810_001, -810_001, "particle");
    value["source_ir"]["crossing"]["phase"] = json!([0.5, -0.25]);
    value["applied_crossing"]["phase"] = json!([0.5, -0.25]);
    let source: GenericSourceRecordManifest =
        serde_json::from_value(value).expect("phased initial source record");
    validate_source_wavefunction_metadata(0, &source).expect("valid nontrivial phase");
    let point = [[5.0, 3.0, 4.0, 0.0]];
    let mut output = [c64(0.0, 0.0); 2];

    ExecutionRuntime::write_source_wavefunction(&source, 1, &BTreeMap::new(), &point, &mut output)
        .expect("evaluate phased source");

    let phase = c64(0.5, -0.25);
    let expected = ext_quark_weyl_array(negate(point[0]), 1, -1).map(|component| component * phase);
    assert_eq!(output, expected);
}

#[cfg(feature = "symbolica-runtime")]
#[test]
fn high_precision_sources_use_typed_orientation_crossing_and_phase() {
    let mut source: GenericSourceRecordManifest = serde_json::from_value(
        initial_crossed_source_value(810_001, -810_001, "antiparticle"),
    )
    .expect("initial source record");
    source.source_orientation = GenericSourceOrientationManifest::Particle;
    source.crossing = "identity".to_owned();
    source.source_ir.crossing.phase = [0.5, -0.25];
    source.applied_crossing.phase = [0.5, -0.25];
    let point = [[
        DoubleFloat::from(5.0),
        DoubleFloat::from(3.0),
        DoubleFloat::from(4.0),
        DoubleFloat::from(0.0),
    ]];
    let zero = || Complex::new(DoubleFloat::from(0.0), DoubleFloat::from(0.0));
    let mut output = vec![zero(), zero()];

    ExecutionRuntime::write_source_wavefunction_generic(
        &source,
        1,
        &BTreeMap::new(),
        &point,
        &mut output,
    )
    .expect("evaluate high-precision crossed antiparticle source");

    let momentum = negate_generic(&point[0]);
    let phase = c_generic(DoubleFloat::from(0.5), DoubleFloat::from(-0.25));
    let expected = ext_antiquark_weyl_generic(&momentum, 1, -1)
        .into_iter()
        .map(|component| component * phase)
        .collect::<Vec<_>>();
    let unexpected = ext_quark_weyl_generic(&momentum, 1, -1)
        .into_iter()
        .map(|component| component * phase)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
    assert_ne!(output, unexpected);
}
