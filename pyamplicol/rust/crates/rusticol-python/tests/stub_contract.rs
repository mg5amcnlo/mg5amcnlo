// SPDX-License-Identifier: 0BSD

const BINDING: &str = include_str!("../src/lib.rs");
const STUB: &str = include_str!("../stubs/pyamplicol/_rusticol.pyi");
const CONSUMER: &str = include_str!("typing_consumer.py");

#[test]
fn stub_covers_the_exported_binding_surface() {
    for name in [
        "Runtime",
        "ProcessPhysics",
        "ExternalParticle",
        "HelicityConfiguration",
        "ColorFlow",
        "ContractedColorComponent",
        "ModelParameter",
        "ResolvedEvaluation",
        "RusticolError",
        "ArtifactError",
        "CompatibilityError",
        "EvaluationError",
        "SelectorError",
        "ModelParameterError",
    ] {
        assert!(BINDING.contains(name), "binding is missing {name}");
        assert!(
            STUB.contains(&format!("class {name}")),
            "stub is missing {name}"
        );
    }
    for function in ["abi_version", "package_version"] {
        assert!(BINDING.contains(&format!("fn {function}")));
        assert!(STUB.contains(&format!("def {function}")));
    }
}

#[test]
fn f64_precision_and_typed_metadata_are_exercised_by_a_consumer() {
    assert!(BINDING.contains("fn artifact_id(&self) -> &str"));
    assert!(STUB.contains("def artifact_id(self) -> str"));
    assert!(CONSUMER.contains("assert_type(runtime.artifact_id, str)"));
    assert!(BINDING.contains("color_flows=None, precision=16"));
    assert!(STUB.contains("precision: Literal[16] = 16"));
    assert!(BINDING.contains("only precision=16"));
    assert!(CONSUMER.contains("evaluate(momenta, precision=16)"));
    assert!(CONSUMER.contains("evaluate_resolved(momenta, precision=16)"));
    assert!(!CONSUMER.contains("Decimal"));
    for field in [
        "external_particles",
        "helicities",
        "color_flows",
        "contracted_color_components",
        "model_parameters",
    ] {
        assert!(STUB.contains(&format!("def {field}")));
        assert!(CONSUMER.contains(field));
    }
}
