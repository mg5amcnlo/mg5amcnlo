// SPDX-License-Identifier: 0BSD
use serde_json::{Value, json};

use super::GenericCurrentSlotManifest;

fn compact_current_slot() -> Value {
    json!({
        "current_id": 0,
        "component_start": 0,
        "component_stop": 1,
        "dimension": 1,
        "is_source": true,
        "particle_id": 101,
        "external_mask": 1,
        "chirality": 0,
        "momentum_mask": 1,
        "auxiliary_kind": null
    })
}

#[test]
fn compact_current_slot_has_no_quantum_number_payload() {
    serde_json::from_value::<GenericCurrentSlotManifest>(compact_current_slot())
        .expect("compact current slot should deserialize without quantum numbers");

    for (field, value) in [
        ("charge_flow", json!(1)),
        ("quantum_number_flow", json!([["electric_charge", "1/5"]])),
    ] {
        let mut payload = compact_current_slot();
        payload
            .as_object_mut()
            .expect("test current slot must be an object")
            .insert(field.to_owned(), value);
        let error = serde_json::from_value::<GenericCurrentSlotManifest>(payload)
            .expect_err("Rust execution current slots must reject quantum-number fields");
        assert!(error.to_string().contains("unknown field"));
    }
}
