// SPDX-License-Identifier: 0BSD

use super::*;
use serde_json::{Value, json};

fn contraction_value() -> Value {
    json!({
        "name": "weyl",
        "left_basis": "weyl-chiral",
        "right_basis": "weyl-chiral",
        "coefficients": [[1.0, 0.0], [1.0, 0.0]],
        "chirality_relation": "opposite",
        "metric_signature": null,
    })
}

fn slot(current_id: usize, start: usize) -> Value {
    json!({
        "current_id": current_id,
        "component_start": start,
        "component_stop": start + 2,
        "dimension": 2,
    })
}

fn value_slot(value_slot_id: usize, current_id: usize, start: usize) -> Value {
    json!({
        "value_slot_id": value_slot_id,
        "current_id": current_id,
        "variant": "source",
        "component_start": start,
        "component_stop": start + 2,
        "dimension": 2,
    })
}

fn root_value() -> Value {
    json!({
        "output_index": 0,
        "root_id": 0,
        "kind": "direct-contraction",
        "left_current_id": 0,
        "right_current_id": 1,
        "left_slot": slot(0, 0),
        "right_slot": slot(1, 2),
        "left_value_slot": value_slot(0, 0, 0),
        "right_value_slot": value_slot(1, 1, 2),
        "vertex_kind": null,
        "vertex_particles": null,
        "coupling": [1.0, 0.0],
        "color_weight": [1.0, 0.0],
        "color_sector_id": 0,
        "contraction": "weyl",
        "contraction_ir": contraction_value(),
        "coherent_group_id": 0,
        "helicity_weight": 1.0,
        "all_sector_weight": 1.0,
    })
}

fn current_storage() -> GenericCurrentStorageManifest {
    serde_json::from_value(json!({
        "component_count": 4,
        "number_type": "complex",
        "metadata_compacted": true,
        "current_slots": [
            {
                "current_id": 0,
                "component_start": 0,
                "component_stop": 2,
                "dimension": 2,
                "is_source": true,
                "particle_id": 1,
                "external_mask": 1,
                "external_labels": [1],
                "helicity_ancestry": 1,
                "chirality": 1,
                "spin_state": 1,
                "flavour_flow": [],
                "color_state": {},
                "momentum_mask": 1,
                "auxiliary_kind": null
            },
            {
                "current_id": 1,
                "component_start": 2,
                "component_stop": 4,
                "dimension": 2,
                "is_source": true,
                "particle_id": -1,
                "external_mask": 2,
                "external_labels": [2],
                "helicity_ancestry": 1,
                "chirality": -1,
                "spin_state": -1,
                "flavour_flow": [],
                "color_state": {},
                "momentum_mask": 2,
                "auxiliary_kind": null
            }
        ]
    }))
    .expect("current storage")
}

fn propagator(particle_id: i32, anti_particle_id: i32, orientation: &str) -> Value {
    json!({
        "identity": {
            "canonical_id": format!("model:test:state:{particle_id}"),
            "species_id": "model:test:species:fermion",
            "anti_canonical_id": format!("model:test:state:{anti_particle_id}"),
            "display_name": format!("state_{particle_id}"),
            "anti_display_name": format!("state_{anti_particle_id}"),
            "pdg_label": particle_id,
            "anti_pdg_label": anti_particle_id,
            "orientation": orientation,
            "self_conjugate": false
        },
        "particle_id": particle_id,
        "chirality": if particle_id > 0 { 1 } else { -1 },
        "kind": "weyl-fermion",
        "backend": "spenso",
        "basis": "weyl-chiral",
        "applies_propagator": true,
        "kernel": "test-weyl",
        "full_tensor_network_ready": true,
        "mass_class": "massless",
        "gauge": null,
        "numerator": "test",
        "denominator": "test",
        "mass_parameter": null,
        "width_parameter": null,
        "custom_source": null,
        "auxiliary_policy": null,
        "goldstone_policy": "not-applicable",
        "description": "test"
    })
}

fn stored_value_slot(
    value_slot_id: usize,
    current_id: usize,
    start: usize,
    particle_id: i32,
) -> Value {
    json!({
        "value_slot_id": value_slot_id,
        "current_id": current_id,
        "variant": "source",
        "component_start": start,
        "component_stop": start + 2,
        "dimension": 2,
        "current_component_start": start,
        "current_component_stop": start + 2,
        "is_source": true,
        "applies_propagator": false,
        "particle_id": particle_id,
        "external_mask": 1_u64 << current_id,
        "external_labels": [current_id + 1],
        "momentum_mask": 1_u64 << current_id,
        "chirality": if particle_id > 0 { 1 } else { -1 },
        "propagator": propagator(
            particle_id,
            -particle_id,
            if particle_id > 0 { "particle" } else { "antiparticle" },
        )
    })
}

fn value_storage() -> GenericValueStorageManifest {
    serde_json::from_value(json!({
        "component_count": 4,
        "number_type": "complex",
        "metadata_compacted": true,
        "value_slots": [
            stored_value_slot(0, 0, 0, 1),
            stored_value_slot(1, 1, 2, -1)
        ]
    }))
    .expect("value storage")
}

fn root() -> GenericAmplitudeRootManifest {
    serde_json::from_value(root_value()).expect("amplitude root")
}

#[test]
fn contraction_ir_requires_strict_complete_metadata() {
    let parsed: GenericContractionIrManifest =
        serde_json::from_value(contraction_value()).expect("ContractionIR");
    assert_eq!(parsed.coefficients.len(), 2);

    let mut missing_nullable = contraction_value();
    missing_nullable
        .as_object_mut()
        .expect("contraction object")
        .remove("metric_signature");
    let error =
        serde_json::from_value::<GenericContractionIrManifest>(missing_nullable).unwrap_err();
    assert!(error.to_string().contains("metric_signature"));

    let mut malformed_pair = contraction_value();
    malformed_pair["coefficients"] = json!([[1.0], [1.0, 0.0]]);
    assert!(serde_json::from_value::<GenericContractionIrManifest>(malformed_pair).is_err());

    let mut unknown = contraction_value();
    unknown["inferred_from_dimension"] = json!(true);
    let error = serde_json::from_value::<GenericContractionIrManifest>(unknown).unwrap_err();
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn pre_contraction_ir_schema_v3_reports_actionable_regeneration_error() {
    let mut missing = root_value();
    missing
        .as_object_mut()
        .expect("root object")
        .remove("contraction_ir");
    let parse_error = serde_json::from_value::<GenericAmplitudeRootManifest>(missing).unwrap_err();

    let error = execution_manifest_parse_error("processes/example/execution.json", parse_error);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(
        error
            .to_string()
            .contains("typed amplitude-contraction metadata")
    );
    assert!(error.to_string().contains("regenerate"));
}

#[test]
fn direct_contraction_ir_matches_projection_slots_and_chirality() {
    let currents = current_storage();
    let values = value_storage();
    validate_amplitude_contraction(0, &root(), &currents, &values)
        .expect("valid typed contraction");

    let mut projection_mismatch = root();
    projection_mismatch.contraction = "display-only-mismatch".to_string();
    assert!(
        validate_amplitude_contraction(0, &projection_mismatch, &currents, &values)
            .unwrap_err()
            .to_string()
            .contains("projection")
    );

    let mut basis_mismatch = root();
    basis_mismatch.contraction_ir.left_basis = "dirac".to_string();
    assert!(
        validate_amplitude_contraction(0, &basis_mismatch, &currents, &values)
            .unwrap_err()
            .to_string()
            .contains("value slots")
    );

    let mut dimension_mismatch = root();
    dimension_mismatch.contraction_ir.coefficients.pop();
    assert!(
        validate_amplitude_contraction(0, &dimension_mismatch, &currents, &values)
            .unwrap_err()
            .to_string()
            .contains("value slots")
    );

    let mut chirality_mismatch = root();
    chirality_mismatch.contraction_ir.chirality_relation =
        GenericContractionChiralityRelationManifest::Equal;
    assert!(
        validate_amplitude_contraction(0, &chirality_mismatch, &currents, &values)
            .unwrap_err()
            .to_string()
            .contains("chirality relation")
    );
}

#[test]
fn contraction_ir_rejects_nonfinite_zero_and_invalid_vertex_projection() {
    let currents = current_storage();
    let values = value_storage();

    let mut nonfinite = root();
    nonfinite.contraction_ir.coefficients[0][0] = f64::NAN;
    assert!(
        validate_amplitude_contraction(0, &nonfinite, &currents, &values)
            .unwrap_err()
            .to_string()
            .contains("invalid typed contraction")
    );

    let mut zero = root();
    zero.contraction_ir.coefficients = vec![[0.0, 0.0], [0.0, 0.0]];
    assert!(
        validate_amplitude_contraction(0, &zero, &currents, &values)
            .unwrap_err()
            .to_string()
            .contains("invalid typed contraction")
    );

    let mut vertex = root();
    vertex.kind = "vertex-closure".to_string();
    vertex.vertex_kind = Some(1);
    vertex.vertex_particles = Some(vec![1, -1, 25]);
    vertex.contraction = "scalar".to_string();
    vertex.contraction_ir = serde_json::from_value(json!({
        "name": "scalar",
        "left_basis": "scalar",
        "right_basis": "scalar",
        "coefficients": [[1.0, 0.0]],
        "chirality_relation": "any",
        "metric_signature": null
    }))
    .expect("scalar projection");
    validate_amplitude_contraction(0, &vertex, &currents, &values)
        .expect("valid scalar projection");

    vertex.contraction_ir.right_basis = "lorentz-vector".to_string();
    assert!(
        validate_amplitude_contraction(0, &vertex, &currents, &values)
            .unwrap_err()
            .to_string()
            .contains("scalar projection")
    );
}

#[test]
fn amplitude_slots_match_storage_and_extreme_chiralities_are_safe() {
    let currents = current_storage();
    let valid_root = root();
    validate_slot_ref(&valid_root.left_slot, 0, &currents).expect("matching current slot");

    let mut mismatched = valid_root.left_slot;
    mismatched.component_start += 1;
    mismatched.component_stop += 1;
    assert!(
        validate_slot_ref(&mismatched, 0, &currents)
            .unwrap_err()
            .to_string()
            .contains("does not match current storage")
    );

    assert!(!valid_current_chirality(i32::MIN));
    assert!(!contraction_chiralities_match(
        GenericContractionChiralityRelationManifest::Opposite,
        0,
        i32::MIN,
    ));
}

#[test]
fn amplitude_root_kind_is_discriminated_and_matches_current_particles() {
    let currents = current_storage();
    let direct = root();
    validate_amplitude_root_kind(0, &direct, &currents).expect("direct contraction");

    let mut stale_direct = root();
    stale_direct.vertex_kind = Some(1);
    stale_direct.vertex_particles = Some(vec![1, -1, 25]);
    assert!(
        validate_amplitude_root_kind(0, &stale_direct, &currents)
            .unwrap_err()
            .to_string()
            .contains("carries vertex metadata")
    );

    let mut vertex = root();
    vertex.kind = "vertex-closure".to_string();
    vertex.vertex_kind = Some(1);
    vertex.vertex_particles = Some(vec![1, -1, 25]);
    validate_amplitude_root_kind(0, &vertex, &currents).expect("matching vertex closure");

    vertex.vertex_kind = Some(-1);
    assert!(
        validate_amplitude_root_kind(0, &vertex, &currents)
            .unwrap_err()
            .to_string()
            .contains("invalid vertex kind")
    );

    vertex.vertex_kind = Some(1);
    vertex.vertex_particles = Some(vec![-1, 1, 25]);
    assert!(
        validate_amplitude_root_kind(0, &vertex, &currents)
            .unwrap_err()
            .to_string()
            .contains("does not match its input currents")
    );
}
