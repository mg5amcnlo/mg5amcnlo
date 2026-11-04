// SPDX-License-Identifier: 0BSD

use super::*;
use crate::pacbin::{PacbinWriteMember, PacbinWriteOptions, write_pacbin_atomic};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEST_ARTIFACT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn walsh_color_contraction_is_a_known_artifact_capability() {
    let capability = crate::engine::COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY.to_string();
    let validated =
        validate_runtime_capabilities(std::slice::from_ref(&capability), "test capabilities")
            .expect("Walsh color-contraction capability");

    assert_eq!(validated, BTreeSet::from([capability]));
}

#[test]
fn c2k_walsh_color_contraction_is_a_known_artifact_capability() {
    let capability = crate::engine::COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY.to_string();
    let validated =
        validate_runtime_capabilities(std::slice::from_ref(&capability), "test capabilities")
            .expect("C2^k Walsh color-contraction capability");

    assert_eq!(validated, BTreeSet::from([capability]));
}

#[test]
fn compiled_plane_arena_is_a_known_artifact_capability() {
    let capability = crate::engine::COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY.to_string();
    let validated =
        validate_runtime_capabilities(std::slice::from_ref(&capability), "test capabilities")
            .expect("compiled plane-arena capability");

    assert_eq!(validated, BTreeSet::from([capability]));
}

#[test]
fn typed_compiled_color_member_is_required_and_kind_checked() {
    let mut artifact = TestArtifact::new();
    let destination = artifact.root.join("evaluators.pacbin");
    let index = write_pacbin_atomic(
        &destination,
        vec![
            PacbinWriteMember::from_bytes(
                "processes/p0/compiled-color.pacrclr3",
                PacbinMemberKind::ColorContraction,
                b"color-payload",
            )
            .unwrap(),
            PacbinWriteMember::from_bytes(
                "processes/p0/wrong-kind.pacrclr3",
                PacbinMemberKind::SymbolicaExactState,
                b"wrong-kind",
            )
            .unwrap(),
        ],
        PacbinWriteOptions::default(),
    )
    .unwrap();
    let container = fs::read(&destination).expect("read evaluator container");
    add_test_payload(
        &mut artifact,
        "evaluators.pacbin",
        "evaluator-state",
        &container,
        None,
        true,
    );
    artifact.manifest["extensions"]["evaluator_payload_container"] = json!({
        "kind": "pyamplicol-evaluator-payload-container",
        "schema_version": 1,
        "storage_abi": "pacbin-v1",
        "path": "evaluators.pacbin",
        "member_count": index.members().len(),
        "unpacked_size_bytes": index.members().iter().map(|member| member.length()).sum::<u64>(),
        "index_sha256": hex_digest(index.index_sha256()),
    });
    artifact.write_manifest();

    let verified = VerifiedArtifact::open(&artifact.root).expect("typed color artifact");
    let store = verified
        .evaluator_payload_store(&artifact.root.join("processes/p0"))
        .expect("process evaluator payload store");

    assert_eq!(
        store
            .packed_member_bytes(
                "compiled-color.pacrclr3",
                PacbinMemberKind::ColorContraction,
            )
            .unwrap(),
        b"color-payload",
    );
    let wrong_kind = store
        .packed_member_bytes("wrong-kind.pacrclr3", PacbinMemberKind::ColorContraction)
        .unwrap_err();
    assert_eq!(wrong_kind.kind(), crate::RusticolErrorKind::Integrity);
    assert!(wrong_kind.message().contains("member kind"));
    let missing = store
        .packed_member_bytes("missing.pacrclr3", PacbinMemberKind::ColorContraction)
        .unwrap_err();
    assert_eq!(missing.kind(), crate::RusticolErrorKind::InvalidArgument);
    assert!(missing.message().contains("unknown pacbin member"));

    let unpacked = EvaluatorPayloadStore::directory(&artifact.root);
    let no_pack = unpacked
        .packed_member_bytes(
            "compiled-color.pacrclr3",
            PacbinMemberKind::ColorContraction,
        )
        .unwrap_err();
    assert_eq!(no_pack.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(no_pack.message().contains("requires evaluators.pacbin"));
}

#[cfg(feature = "f64-symjit")]
const PYTHON_PACBIN_GOLDEN_HEX: &str = concat!(
    "50414342494e000001004000000000004000000000000000c000000000000000",
    "0200000000000000000000000000000000000000000000000000000000000000",
    "6a69742d76310000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "65786163742d7374617465000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "5041434944580000010020000000000002000000000000000000000000000000",
    "0c0000000100000040000000000000000600000000000000e9a628134c31e4a5",
    "7771ecd7a1dd51c5c5bc7daca10a2553c83f8af1652d2587612f6a69742e7379",
    "6d6a6974000000000b0000000200000080000000000000000b00000000000000",
    "a9f46d72eb6f1c7e5d62e97d3d8e67b80605aeba2df022a8a25b51ca3fc89805",
    "7a2f65786163742e62696e0000000000504143454e4400000100400000000000",
    "c0000000000000000200000000000000263b30eb85e5b2a82600201822438767",
    "c5f66d9eecf2e08a0202383aafe0e3c1",
);

#[test]
fn distribution_version_normalization_is_narrow() {
    assert_eq!(
        canonical_distribution_version("0.1.0-dev.0+candidate.0123"),
        "0.1.0.dev0+candidate.0123"
    );
    assert_eq!(canonical_distribution_version("0.1.0"), "0.1.0");
    assert_eq!(
        canonical_distribution_version("0.2.1-dev.0+candidate.3112b2494ca7"),
        canonical_distribution_version("0.2.1.dev0+candidate.42ec2237ad19")
    );
    for other in [
        "0.2.1",
        "0.2.2.dev0+candidate.3112b2494ca7",
        "0.2.1.dev0+other.3112b2494ca7",
    ] {
        assert_ne!(
            canonical_distribution_version("0.2.1.dev0+candidate.3112b2494ca7"),
            canonical_distribution_version(other)
        );
    }
    assert_ne!(
        canonical_distribution_version("0.1.0-dev.0+candidate.0123"),
        canonical_distribution_version("0.1.0-dev.0+candidate.4567")
    );
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn compact_payload_identity_matches_the_python_canonical_contract() {
    let artifact = TestArtifact::new();
    let typed: ArtifactManifest =
        serde_json::from_value(artifact.manifest.clone()).expect("typed artifact manifest");
    assert_eq!(
        compute_payload_artifact_id(&typed.payloads).expect("typed payload identity"),
        artifact.manifest["artifact_id"]
            .as_str()
            .expect("manifest artifact ID")
    );
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn default_recurrence_bootstrap_discovery_requires_exactly_one_stable_process() {
    let artifact = TestArtifact::new();
    let processes = artifact.root.join("processes");
    fs::create_dir(&processes).expect("create process root");
    assert_eq!(
        discover_single_recurrence_bootstrap_process(&artifact.root).unwrap(),
        None
    );

    let first = processes.join("p0");
    fs::create_dir(&first).expect("create first process");
    fs::write(first.join(RECURRENCE_BOOTSTRAP_FILENAME), b"image").expect("write first bootstrap");
    assert_eq!(
        discover_single_recurrence_bootstrap_process(&artifact.root).unwrap(),
        Some("p0".to_string())
    );

    let second = processes.join("p1");
    fs::create_dir(&second).expect("create second process");
    fs::write(second.join(RECURRENCE_BOOTSTRAP_FILENAME), b"image")
        .expect("write second bootstrap");
    let error = discover_single_recurrence_bootstrap_process(&artifact.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("ambiguous"));
}

struct TestArtifact {
    root: PathBuf,
    manifest: Value,
}

impl TestArtifact {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let temp_root = std::env::temp_dir()
            .canonicalize()
            .expect("canonical temporary directory");
        let root = temp_root.join(format!(
            "pyamplicol-rusticol-manifest-{}-{nonce}-{}",
            std::process::id(),
            TEST_ARTIFACT_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).expect("create test artifact");
        let files = [
            ("requested.toml", b"action = \"generate\"\n".as_slice()),
            ("effective.toml", b"action = \"generate\"\n".as_slice()),
            ("evaluator.json", b"{}".as_slice()),
            ("physics.json", b"{}".as_slice()),
        ];
        let mut payloads = Vec::new();
        for (path, bytes) in files {
            fs::write(root.join(path), bytes).expect("write payload");
            let role = match path {
                "requested.toml" => "configuration-requested",
                "effective.toml" => "configuration-effective",
                "evaluator.json" => "evaluator-manifest",
                "physics.json" => "runtime-physics",
                _ => unreachable!(),
            };
            let mut payload = json!({
                "path": path,
                "role": role,
                "media_type": "application/octet-stream",
                "size_bytes": bytes.len(),
                "sha256": sha256(bytes),
                "executable": false,
            });
            if path == "physics.json" {
                payload["process_id"] = json!("p0");
            }
            payloads.push(payload);
        }
        let manifest = json!({
            "schema_version": 3,
            "kind": "pyamplicol-process",
            "artifact_id": "0".repeat(64),
            "created_utc": "2026-07-15T00:00:00Z",
            "producer": {
                "distribution": "pyamplicol",
                "version": env!("CARGO_PKG_VERSION"),
                "versions": {
                    "python_api": PYTHON_API_VERSION,
                    "toml": TOML_SCHEMA_VERSION,
                    "compiled_model": COMPILED_MODEL_SCHEMA_VERSION,
                    "process_artifact": PROCESS_ARTIFACT_SCHEMA_VERSION,
                    "runtime_physics": RUNTIME_PHYSICS_SCHEMA_VERSION,
                    "symbolica_serialization": crate::SYMBOLICA_SERIALIZATION_ABI,
                    "c_abi": C_ABI_VERSION,
                },
                "target": {"triple": current_target_triple(), "cpu_features": []},
            },
            "model": {
                "name": "test-model",
                "source_kind": "compiled-model",
                "content_sha256": "1".repeat(64),
                "compiled_schema_version": COMPILED_MODEL_SCHEMA_VERSION,
            },
            "configuration": {
                "toml_schema_version": TOML_SCHEMA_VERSION,
                "requested_path": "requested.toml",
                "effective_path": "effective.toml",
                "adjustments": [],
            },
            "processes": [{
                "id": "p0",
                "expression": "a b > c",
                "color_accuracy": "lc",
                "external_pdgs": [1, -1, 22],
                "physics_path": "physics.json",
                "required_runtime_capabilities": [
                    "symjit.application.complex-f64.v1"
                ],
                "aliases": [],
            }],
            "default_process_id": "p0",
            "runtime": {
                "engine": "rusticol",
                "engine_version": env!("CARGO_PKG_VERSION"),
                "evaluator_manifest_path": "evaluator.json",
                "required_runtime_capabilities": [
                    "symjit.application.complex-f64.v1"
                ],
                "api_bundle_path": null,
            },
            "payloads": payloads,
            "dependencies": [],
            "extensions": {
                "artifact_identity": {
                    "kind": ARTIFACT_IDENTITY_KIND,
                    "schema_version": ARTIFACT_IDENTITY_SCHEMA_VERSION,
                },
            },
        });
        let mut artifact = Self { root, manifest };
        artifact.write_manifest();
        artifact
    }

    fn write_manifest(&mut self) {
        if self.manifest["schema_version"] == json!(PROCESS_ARTIFACT_SCHEMA_VERSION) {
            self.manifest["artifact_id"] =
                json!(compute_artifact_id(&self.manifest).expect("compute artifact identity"));
        }
        let mut bytes = String::new();
        write_python_canonical_json(&self.manifest, &mut bytes)
            .expect("serialize canonical manifest");
        bytes.push('\n');
        fs::write(self.root.join(ARTIFACT_MANIFEST_FILE), bytes).expect("write manifest");
    }
}

impl Drop for TestArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn add_structural_source_proof(artifact: &mut TestArtifact) {
    let path = "processes/p0/structural-source-proof.json";
    let bytes = br#"{"schema":"pyamplicol-generation-structural-source-proof-v1"}"#;
    let output = artifact.root.join(path);
    fs::create_dir_all(output.parent().expect("structural proof parent"))
        .expect("create structural proof parent");
    fs::write(output, bytes).expect("write structural proof");
    artifact.manifest["payloads"]
        .as_array_mut()
        .expect("payload array")
        .push(json!({
            "path": path,
            "role": "structural-source-proof",
            "media_type": "application/json",
            "size_bytes": bytes.len(),
            "sha256": sha256(bytes),
            "executable": false,
            "process_id": "p0",
        }));
    artifact.write_manifest();
}

#[cfg(feature = "f64-symjit")]
fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0);
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("ASCII hex pair");
            u8::from_str_radix(text, 16).expect("valid hex pair")
        })
        .collect()
}

fn add_test_payload(
    artifact: &mut TestArtifact,
    path: &str,
    role: &str,
    bytes: &[u8],
    process_id: Option<&str>,
    targeted: bool,
) {
    let output = artifact.root.join(path);
    fs::create_dir_all(output.parent().expect("payload parent")).expect("create payload parent");
    fs::write(&output, bytes).expect("write mixed-runtime payload");
    let mut payload = json!({
        "path": path,
        "role": role,
        "media_type": "application/octet-stream",
        "size_bytes": bytes.len(),
        "sha256": sha256(bytes),
        "executable": false,
    });
    if let Some(process_id) = process_id {
        payload["process_id"] = json!(process_id);
    }
    if targeted {
        payload["target"] = json!({
            "triple": current_target_triple(),
            "cpu_features": [],
        });
    }
    artifact.manifest["payloads"]
        .as_array_mut()
        .expect("payload array")
        .push(payload);
}

#[cfg(feature = "f64-symjit")]
fn replace_test_payload(artifact: &mut TestArtifact, path: &str, bytes: &[u8]) {
    fs::write(artifact.root.join(path), bytes).expect("replace test payload");
    let payload = artifact.manifest["payloads"]
        .as_array_mut()
        .expect("payload array")
        .iter_mut()
        .find(|payload| payload["path"] == path)
        .expect("declared test payload");
    payload["size_bytes"] = json!(bytes.len());
    payload["sha256"] = json!(sha256(bytes));
}

#[cfg(feature = "f64-symjit")]
fn add_test_evaluator_container(artifact: &mut TestArtifact) {
    let bytes = decode_hex(PYTHON_PACBIN_GOLDEN_HEX);
    add_test_payload(
        artifact,
        "evaluators.pacbin",
        "evaluator-state",
        &bytes,
        None,
        true,
    );
    artifact.manifest["extensions"]["evaluator_payload_container"] = json!({
        "kind": "pyamplicol-evaluator-payload-container",
        "schema_version": 1,
        "storage_abi": "pacbin-v1",
        "path": "evaluators.pacbin",
        "member_count": 2,
        "unpacked_size_bytes": 17,
        "index_sha256": "263b30eb85e5b2a82600201822438767c5f66d9eecf2e08a0202383aafe0e3c1",
    });
    artifact.write_manifest();
}

#[cfg(feature = "f64-symjit")]
const DIRECT_SYMJIT_PROGRAM: &str = r#"[[{"Add":[{"Out":0},[{"Param":0},{"Param":1}],0]}],1,[]]"#;

#[cfg(feature = "f64-symjit")]
fn direct_symjit_application_bytes(input_len: usize) -> Vec<u8> {
    use symjit::{Compiler, CompilerType, Config, Storage};

    let mut config = Config::new(CompilerType::Native, 0).expect("native SymJIT config");
    config.set_complex(true);
    config.set_symbolica(true);
    config.set_opt_level(3);
    config.set_simd(true);
    let mut compiler = Compiler::with_config(config);
    let application = compiler
        .translate(DIRECT_SYMJIT_PROGRAM.to_string(), input_len)
        .expect("translate direct SymJIT test application");
    let mut bytes = Vec::new();
    application
        .save(&mut bytes)
        .expect("serialize direct SymJIT test application");
    bytes
}

#[cfg(feature = "f64-symjit")]
fn direct_symjit_plane_application_bytes(input_len: usize) -> Vec<u8> {
    crate::engine::compile_symbolica_program_to_plane_application_bytes(
        DIRECT_SYMJIT_PROGRAM,
        input_len,
        1,
        3,
        false,
    )
    .expect("translate direct SymJIT test plane application")
}

#[cfg(feature = "f64-symjit")]
pub(crate) fn minimal_runtime_physics(
    process_id: &str,
    process: &str,
    final_particle: &str,
) -> Value {
    json!({
        "schema_version": RUNTIME_PHYSICS_SCHEMA_VERSION,
        "kind": "pyamplicol-resolved-physics",
        "process_id": process_id,
        "process": process,
        "color_accuracy": "lc",
        "coverage": {
            "helicities": "complete",
            "color": "complete",
            "color_kind": "physical-lc-flows",
            "structural_zero_helicity_count": 0,
        },
        "external_particles": [
            {
                "index": 0,
                "label": 1,
                "particle": "a",
                "pdg": 1,
                "role": "initial",
                "momentum_slot": 0,
                "momentum_components": ["E", "px", "py", "pz"],
            },
            {
                "index": 1,
                "label": 2,
                "particle": "b",
                "pdg": -1,
                "role": "initial",
                "momentum_slot": 1,
                "momentum_components": ["E", "px", "py", "pz"],
            },
            {
                "index": 2,
                "label": 3,
                "particle": final_particle,
                "pdg": 22,
                "role": "final",
                "momentum_slot": 2,
                "momentum_components": ["E", "px", "py", "pz"],
            },
        ],
        "helicities": [{
            "id": "h:+0,+0,+0",
            "index": 0,
            "values": [0, 0, 0],
            "computed": true,
            "structural_zero": false,
            "representative_id": "h:+0,+0,+0",
            "coefficient": 1.0,
        }],
        "color_components": [{
            "kind": "lc-flow",
            "id": "flow:singlet",
            "index": 0,
            "word": [],
            "computed": true,
            "representative_id": "flow:singlet",
            "coefficient": 1.0,
        }],
        "reduction": {
            "kind": "lc-diagonal",
            "groups": [{
                "id": "group:7",
                "representative_helicity_id": "h:+0,+0,+0",
                "representative_color_id": "flow:singlet",
                "physical_helicity_ids": ["h:+0,+0,+0"],
                "physical_color_ids": ["flow:singlet"],
            }],
        },
        "model_parameters": [],
        "selectors": {
            "helicity": true,
            "color_flow": true,
            "contracted_color": false,
        },
        "extensions": {},
    })
}

#[cfg(feature = "f64-symjit")]
pub(crate) fn direct_evaluator_manifest(application_path: &str) -> Value {
    json!({
        "kind": "symjit-application-evaluator",
        "runtime_capability": "symjit.application.complex-f64.v1",
        "application_path": application_path,
        "application_abi": crate::engine::SYMJIT_APPLICATION_STORAGE_ABI,
        "input_len": 14,
        "output_len": 1,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": 3,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": null,
        "evaluator_state_runtime_capability": null,
    })
}

#[cfg(feature = "f64-symjit")]
fn compiled_evaluator_manifest(library_path: &str, runtime_capability: &str) -> Value {
    json!({
        "kind": "compiled-complex-evaluator",
        "runtime_capability": runtime_capability,
        "function_name": "rusticol_test_evaluator",
        "input_len": 14,
        "output_len": 1,
        "library_path": library_path,
        "evaluator_state_path": null,
        "number_type": "complex-f64",
    })
}

#[cfg(feature = "f64-symjit")]
pub(crate) fn minimal_execution_manifest(
    process_id: &str,
    process: &str,
    capability: &str,
    evaluator: Value,
) -> Value {
    let value_slot = |value_slot_id: usize, current_id: usize, component: usize| {
        json!({
            "value_slot_id": value_slot_id,
            "current_id": current_id,
            "variant": "source",
            "component_start": component,
            "component_stop": component + 1,
            "dimension": 1,
        })
    };
    let slot = |current_id: usize, component: usize| {
        json!({
            "current_id": current_id,
            "component_start": component,
            "component_stop": component + 1,
            "dimension": 1,
        })
    };
    let crossing_ir = || {
        json!({
            "momentum_transform": "negate-four-momentum",
            "helicity_factor": 1,
            "chirality_factor": 1,
            "spin_state_factor": 1,
            "phase": [1.0, 0.0],
        })
    };
    let source_ir = |particle_id: i32, anti_particle_id: i32, orientation: &str| {
        json!({
            "identity": {
                "canonical_id": format!("model:minimal:state:{particle_id}"),
                "species_id": "model:minimal:species:1",
                "anti_canonical_id": format!("model:minimal:state:{anti_particle_id}"),
                "display_name": format!("state_{particle_id}"),
                "anti_display_name": format!("state_{anti_particle_id}"),
                "pdg_label": particle_id,
                "anti_pdg_label": anti_particle_id,
                "orientation": orientation,
                "self_conjugate": false,
            },
            "statistics": "boson",
            "wavefunction_family": "scalar",
            "component_dimension": 1,
            "states": [{"helicity": 0, "chirality": 0, "spin_state": 1}],
            "crossing": crossing_ir(),
            "basis": "scalar",
            "mass_parameter": null,
            "width_parameter": null,
        })
    };
    let propagator_ir = |particle_id: i32, anti_particle_id: i32, orientation: &str| {
        json!({
            "identity": {
                "canonical_id": format!("model:minimal:state:{particle_id}"),
                "species_id": "model:minimal:species:1",
                "anti_canonical_id": format!(
                    "model:minimal:state:{anti_particle_id}"
                ),
                "display_name": format!("state_{particle_id}"),
                "anti_display_name": format!("state_{anti_particle_id}"),
                "pdg_label": particle_id,
                "anti_pdg_label": anti_particle_id,
                "orientation": orientation,
                "self_conjugate": false,
            },
            "particle_id": particle_id,
            "chirality": 0,
            "kind": "scalar",
            "backend": "test",
            "basis": "scalar",
            "applies_propagator": true,
            "kernel": "test_scalar",
            "full_tensor_network_ready": true,
            "mass_class": "massless",
            "gauge": null,
            "numerator": "i",
            "denominator": "momentum_squared-mass_squared+i*mass*width",
            "mass_parameter": null,
            "width_parameter": null,
            "custom_source": null,
            "auxiliary_policy": null,
            "goldstone_policy": "not-applicable",
            "description": "test scalar propagator",
        })
    };
    let real_inputs = (2..14).collect::<Vec<_>>();
    let current_storage = json!({
        "component_count": 2,
        "number_type": "complex",
        "metadata_compacted": true,
        "current_slots": [
            {
                "current_id": 0, "component_start": 0, "component_stop": 1,
                "dimension": 1, "is_source": true, "particle_id": 1,
                "external_mask": 1, "external_labels": [], "helicity_ancestry": null,
                "chirality": 0, "spin_state": null, "flavour_flow": [],
                "color_state": null, "momentum_mask": 1,
                "auxiliary_kind": null
            },
            {
                "current_id": 1, "component_start": 1, "component_stop": 2,
                "dimension": 1, "is_source": true, "particle_id": -1,
                "external_mask": 2, "external_labels": [], "helicity_ancestry": null,
                "chirality": 0, "spin_state": null, "flavour_flow": [],
                "color_state": null, "momentum_mask": 2,
                "auxiliary_kind": null
            }
        ],
    });
    let value_storage = json!({
        "component_count": 2,
        "number_type": "complex",
        "metadata_compacted": true,
        "value_slots": [
            {
                "value_slot_id": 0, "current_id": 0, "variant": "source",
                "component_start": 0, "component_stop": 1, "dimension": 1,
                "current_component_start": 0, "current_component_stop": 1,
                "is_source": true, "applies_propagator": false, "particle_id": 1,
                "external_mask": 1, "external_labels": [], "momentum_mask": 1,
                "chirality": 0,
                "propagator": propagator_ir(1, -1, "particle")
            },
            {
                "value_slot_id": 1, "current_id": 1, "variant": "source",
                "component_start": 1, "component_stop": 2, "dimension": 1,
                "current_component_start": 1, "current_component_stop": 2,
                "is_source": true, "applies_propagator": false, "particle_id": -1,
                "external_mask": 2, "external_labels": [], "momentum_mask": 2,
                "chirality": 0,
                "propagator": propagator_ir(-1, 1, "antiparticle")
            }
        ],
    });
    let source_fill = json!({
        "source_count": 2,
        "sources": [
            {
                "source_id": 0, "current_id": 0,
                "current_component_start": 0, "current_component_stop": 1,
                "value_slot": value_slot(0, 0, 0),
                "source_parameter_start": 0, "source_parameter_stop": 1,
                "leg_label": 1, "input_momentum_slot": 0, "side": "initial",
                "crossing": "negate-incoming-momentum", "physical_pdg": 1, "outgoing_pdg": 1,
                "particle_id": 1, "anti_particle_id": -1,
                "source_kind": "external-wavefunction",
                "wavefunction_kind": "scalar", "source_orientation": "particle",
                "source_basis": "scalar", "source_ir": source_ir(1, -1, "particle"),
                "applied_crossing": crossing_ir(),
                "source_helicity": 0,
                "chirality": 0, "spin_state": 1, "dimension": 1,
                "helicity_ancestry": 1, "color_state": {}
            },
            {
                "source_id": 1, "current_id": 1,
                "current_component_start": 1, "current_component_stop": 2,
                "value_slot": value_slot(1, 1, 1),
                "source_parameter_start": 1, "source_parameter_stop": 2,
                "leg_label": 2, "input_momentum_slot": 1, "side": "initial",
                "crossing": "negate-incoming-momentum", "physical_pdg": -1, "outgoing_pdg": -1,
                "particle_id": -1, "anti_particle_id": 1,
                "source_kind": "external-wavefunction",
                "wavefunction_kind": "scalar", "source_orientation": "antiparticle",
                "source_basis": "scalar", "source_ir": source_ir(-1, 1, "antiparticle"),
                "applied_crossing": crossing_ir(),
                "source_helicity": 0,
                "chirality": 0, "spin_state": 1, "dimension": 1,
                "helicity_ancestry": 1, "color_state": {}
            }
        ],
    });
    let amplitude_stage = json!({
        "stage_kind": "amplitude-roots",
        "output_count": 1,
        "color_contraction": null,
        "roots": [{
            "output_index": 0, "root_id": 0, "kind": "direct-contraction",
            "left_current_id": 0, "right_current_id": 1,
            "left_slot": slot(0, 0), "right_slot": slot(1, 1),
            "left_value_slot": value_slot(0, 0, 0),
            "right_value_slot": value_slot(1, 1, 1),
            "vertex_kind": null, "vertex_particles": null,
            "coupling": [1.0, 0.0], "color_weight": [1.0, 0.0],
            "color_sector_id": 0, "contraction": "minimal-test",
            "contraction_ir": {
                "name": "minimal-test",
                "left_basis": "scalar",
                "right_basis": "scalar",
                "coefficients": [[1.0, 0.0]],
                "chirality_relation": "any",
                "metric_signature": null
            },
            "coherent_group_id": 7, "helicity_weight": 1.0,
            "all_sector_weight": 1.0
        }],
    });
    let runtime_schema = json!({
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution-plan",
        "process_key": process_id,
        "process": process,
        "external_particles": [
            {"label": 1, "index": 0, "pdg": 1, "outgoing_pdg": 1, "role": "initial", "momentum_slot": 0},
            {"label": 2, "index": 1, "pdg": -1, "outgoing_pdg": -1, "role": "initial", "momentum_slot": 1},
            {"label": 3, "index": 2, "pdg": 22, "outgoing_pdg": 22, "role": "final", "momentum_slot": 2},
        ],
        "model": null,
        "model_parameters": [],
        "normalization": null,
        "parameter_layout": {
            "source_component_parameter_count": 2,
            "momentum_parameter_count": 12,
            "model_parameter_count": 0,
            "parameter_count_if_flattened": 14,
            "value_component_count": 2,
            "source_components_complex": true,
            "momentum_components_real": true,
            "real_valued_inputs": real_inputs,
        },
        "current_storage": current_storage,
        "value_storage": value_storage,
        "source_fill": source_fill,
        "momentum_slots": [
            {"momentum_slot_id": 0, "momentum_mask": 1, "external_labels": [1], "component_start": 0, "component_stop": 4, "real_valued": true},
            {"momentum_slot_id": 1, "momentum_mask": 2, "external_labels": [2], "component_start": 4, "component_stop": 8, "real_valued": true},
            {"momentum_slot_id": 2, "momentum_mask": 4, "external_labels": [3], "component_start": 8, "component_stop": 12, "real_valued": true},
        ],
        "stages": [],
        "amplitude_stage": amplitude_stage,
    });
    let serialized_amplitude = json!({
        "stage_index": 0,
        "stage_kind": "amplitude-roots",
        "subset_size": null,
        "evaluator_label": "minimal_amplitude",
        "parameter_layout": "global-value-momentum",
        "output_length": 1,
        "output_slots": [{
            "value_slot_id": -1,
            "current_id": -1,
            "variant": "amplitude-root",
            "component_start": 0,
            "component_stop": 1,
            "output_start": 0,
            "output_stop": 1,
        }],
        "input_value_slot_ids": [0, 1],
        "output_value_slot_ids": [],
        "interaction_ids": [],
        "input_components": [],
        "parameter_count": 14,
        "value_parameter_count": 2,
        "momentum_parameter_count": 12,
        "model_parameter_count": 0,
        "real_valued_inputs": (2..14).collect::<Vec<_>>(),
        "expression_ready": true,
        "blockers": [],
        "evaluator": evaluator,
    });
    let compiled = json!({
        "kind": "generic-dag-stage-blueprint",
        "runtime_available": true,
        "runtime_unavailable_message": null,
        "model_parameter_evaluator": null,
        "stage_evaluators": {
            "kind": "generic-dag-stage-evaluator-artifacts",
            "required_runtime_capabilities": [capability],
            "runtime_available": true,
            "runtime_unavailable_message": null,
            "parameter_count": 14,
            "value_parameter_count": 2,
            "momentum_parameter_count": 12,
            "model_parameter_count": 0,
            "real_valued_inputs": (2..14).collect::<Vec<_>>(),
            "parameter_layout": "global-value-momentum",
            "stage_count": 1,
            "stages": [],
            "amplitude_stage": serialized_amplitude,
        },
    });
    json!({
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution",
        "required_runtime_capabilities": [capability],
        "process": process,
        "key": process_id,
        "color_accuracy": "lc",
        "external_pdg_order": [1, -1, 22],
        "compiled": compiled,
        "dag_summary": {
            "current_count": 2,
            "source_count": 2,
            "interaction_count": 0,
            "interaction_evaluation_count": 0,
            "amplitude_root_count": 1,
            "truncated": false,
        },
        "materialization_census": {
            "abi": "pyamplicol-fully-resident-materialization-census-v1",
            "basis": "immutable-fully-resident-compiled-dag",
            "final": {
                "current_count": 2,
                "source_count": 2,
                "interaction_count": 0,
                "interaction_evaluation_count": 0,
                "amplitude_root_count": 1,
            },
            "peak": {
                "current_count": 2,
                "source_count": 2,
                "interaction_count": 0,
                "interaction_evaluation_count": 0,
                "amplitude_root_count": 1,
            },
            "final_equals_peak": true,
        },
        "runtime_schema": runtime_schema,
    })
}

#[cfg(feature = "f64-symjit")]
pub(crate) fn mock_helicity_materialization() -> Value {
    json!({
        "kind": "pyamplicol-helicity-recurrence",
        "contract_version": 1,
        "proof_algorithm": "canonical-source-transition-dependency-shape-v1",
        "current_count": 2,
        "amplitude_root_count": 1,
        "proof_counts": {
            "recurrence_class_count": 0,
            "optimized_recurrence_class_count": 0,
            "optimized_current_count": 0,
            "residual_current_count": 2,
            "amplitude_class_count": 0,
            "optimized_amplitude_class_count": 0,
            "residual_amplitude_count": 1,
            "source_state_mapping_count": 0,
            "physical_helicity_count": 2,
            "structural_zero_helicity_count": 0,
        },
        "selector_domains": [
            {"id": 0, "complete": true, "source_states": []},
            {"id": 1, "complete": true, "source_states": []},
        ],
        "source_state_mappings": [],
        "recurrence_classes": [],
        "amplitude_classes": [],
        "residual_current_ids": [0, 1],
        "residual_root_ids": [0],
        "structural_zero_selector_domain_ids": [],
        "diagnostics": [],
        "materialization": {
            "kind": "pyamplicol-helicity-recurrence-materialization",
            "contract_version": 1,
            "strategy": "quotient",
            "proof_current_count": 2,
            "proof_root_count": 1,
            "materialized_current_count": 2,
            "materialized_root_count": 1,
            "proof_to_materialized_current": [0, 1],
            "source_routes": [],
            "amplitude_routes": [],
            "selector_schedules": [],
        },
    })
}

#[cfg(feature = "f64-symjit")]
pub(crate) fn minimal_helicity_selector_lane_execution() -> Value {
    const DIRECT: &str = "symjit.application.complex-f64.v1";
    let mut execution = minimal_execution_manifest(
        "p0",
        "a b > c",
        DIRECT,
        direct_evaluator_manifest("evaluators/direct.symjit"),
    );
    let mut lane = execution.clone();
    lane["physics_reduction"] = minimal_runtime_physics("p0", "a b > c", "c")["reduction"].clone();
    execution["runtime_schema"]["helicity_recurrence"] = mock_helicity_materialization();
    execution["required_runtime_capabilities"] = json!([
        crate::engine::COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
        crate::engine::COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        crate::engine::COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        DIRECT,
    ]);
    execution["helicity_selector_executions"] = json!([{
        "selector_domain_ids": [0, 1],
        "execution": lane,
    }]);
    execution
}

#[cfg(feature = "f64-symjit")]
fn helicity_selector_lane_artifact(execution: Value) -> TestArtifact {
    const DIRECT: &str = "symjit.application.complex-f64.v1";
    let mut artifact = TestArtifact::new();
    let capabilities = json!([
        crate::engine::COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
        crate::engine::COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        crate::engine::COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        DIRECT,
    ]);
    artifact.manifest["processes"][0]["required_runtime_capabilities"] = capabilities.clone();
    artifact.manifest["runtime"]["required_runtime_capabilities"] = capabilities;
    replace_test_payload(
        &mut artifact,
        "evaluator.json",
        &serde_json::to_vec(&execution).expect("serialize execution fixture"),
    );
    replace_test_payload(
        &mut artifact,
        "physics.json",
        &serde_json::to_vec(&minimal_runtime_physics("p0", "a b > c", "c"))
            .expect("serialize physics fixture"),
    );
    add_test_payload(
        &mut artifact,
        "evaluators/direct.symjit",
        "evaluator-state",
        &direct_symjit_application_bytes(14),
        Some("p0"),
        true,
    );
    artifact.write_manifest();
    artifact
}

#[cfg(feature = "f64-symjit")]
fn helicity_selector_lane_load_error(execution: Value) -> crate::RusticolError {
    let artifact = helicity_selector_lane_artifact(execution);
    match crate::NativeRuntime::load(&artifact.root, Some("p0"), None) {
        Ok(_) => panic!("invalid compiled helicity selector lane unexpectedly loaded"),
        Err(error) => error,
    }
}

#[cfg(feature = "f64-symjit")]
#[test]
fn compiled_helicity_selector_lanes_reject_duplicate_domains() {
    let mut execution = minimal_helicity_selector_lane_execution();
    let mut duplicate = execution["helicity_selector_executions"][0].clone();
    duplicate["selector_domain_ids"] = json!([1]);
    execution["helicity_selector_executions"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);

    let error = helicity_selector_lane_load_error(execution);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
    assert!(
        error.to_string().contains("duplicate selector domain 1"),
        "{error}"
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn compiled_helicity_selector_lanes_reject_unknown_domains() {
    let mut execution = minimal_helicity_selector_lane_execution();
    execution["helicity_selector_executions"][0]["selector_domain_ids"] = json!([2]);

    let error = helicity_selector_lane_load_error(execution);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
    assert!(
        error.to_string().contains("unknown selector domain 2"),
        "{error}"
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn compiled_helicity_selector_lanes_reject_nested_auxiliaries() {
    for nested_field in ["helicity_sum_execution", "color_selector_executions"] {
        let mut execution = minimal_helicity_selector_lane_execution();
        let lane = &mut execution["helicity_selector_executions"][0]["execution"];
        let nested = lane.clone();
        lane[nested_field] = match nested_field {
            "helicity_sum_execution" => nested,
            _ => json!([{"materialized_sector_id": 0, "execution": nested}]),
        };

        let error = helicity_selector_lane_load_error(execution);

        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
        assert!(
            error.to_string().contains(
                "helicity-selector execution contains an unsupported auxiliary execution"
            ),
            "{nested_field}: {error}"
        );
    }
}

#[cfg(feature = "f64-symjit")]
#[test]
fn parent_closure_helicity_selector_lane_rejects_nested_closure() {
    let mut execution = minimal_helicity_selector_lane_execution();
    let lane = &mut execution["helicity_selector_executions"][0]["execution"];
    let nested = lane.clone();
    lane["helicity_selector_executions"] =
        json!([{"selector_domain_ids": [0], "execution": nested}]);

    let error = helicity_selector_lane_load_error(execution);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
    assert!(
        error
            .to_string()
            .contains("only a nested-runtime helicity-selector execution"),
        "{error}"
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn compiled_helicity_selector_lanes_reject_nested_materialization() {
    let mut execution = minimal_helicity_selector_lane_execution();
    execution["helicity_selector_executions"][0]["execution"]["runtime_schema"]["helicity_recurrence"] =
        mock_helicity_materialization();

    let error = helicity_selector_lane_load_error(execution);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
    assert!(
        error
            .to_string()
            .contains("cannot contain helicity materialization"),
        "{error}"
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn nested_runtime_helicity_selector_lane_requires_materialization() {
    let mut execution = minimal_helicity_selector_lane_execution();
    execution["helicity_selector_executions"][0]["schedule_mode"] = json!("nested-runtime");

    let error = helicity_selector_lane_load_error(execution);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
    assert!(
        error.to_string().contains(
            "nested-runtime helicity-selector execution requires helicity materialization"
        ),
        "{error}"
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn compiled_helicity_selector_lanes_verify_nested_payloads() {
    let mut execution = minimal_helicity_selector_lane_execution();
    execution["helicity_selector_executions"][0]["execution"]["compiled"]["stage_evaluators"]["amplitude_stage"]
        ["evaluator"]["application_path"] = json!("evaluators/undeclared-selector-lane.symjit");

    let error = helicity_selector_lane_load_error(execution);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Security, "{error}");
    assert!(
        error
            .to_string()
            .contains("undeclared-selector-lane.symjit"),
        "{error}"
    );
}

#[cfg(feature = "f64-symjit")]
fn minimal_color_topology_lane_execution() -> Value {
    const DIRECT: &str = "symjit.application.complex-f64.v1";
    let mut execution = minimal_execution_manifest(
        "p0",
        "a b > c",
        DIRECT,
        direct_evaluator_manifest("evaluators/direct.symjit"),
    );
    let mut lane = execution.clone();
    lane["physics_reduction"] = minimal_runtime_physics("p0", "a b > c", "c")["reduction"].clone();
    execution["required_runtime_capabilities"] = json!([
        crate::engine::COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        crate::engine::COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        DIRECT,
    ]);
    execution["color_selector_executions"] = json!([{
        "materialized_sector_id": 0,
        "execution": lane,
    }]);
    execution
}

#[cfg(feature = "f64-symjit")]
fn color_topology_lane_artifact(execution: Value) -> TestArtifact {
    const DIRECT: &str = "symjit.application.complex-f64.v1";
    let mut artifact = TestArtifact::new();
    let capabilities = json!([
        crate::engine::COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        crate::engine::COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        DIRECT,
    ]);
    artifact.manifest["processes"][0]["required_runtime_capabilities"] = capabilities.clone();
    artifact.manifest["runtime"]["required_runtime_capabilities"] = capabilities;
    replace_test_payload(
        &mut artifact,
        "evaluator.json",
        &serde_json::to_vec(&execution).expect("serialize execution fixture"),
    );
    replace_test_payload(
        &mut artifact,
        "physics.json",
        &serde_json::to_vec(&minimal_runtime_physics("p0", "a b > c", "c"))
            .expect("serialize physics fixture"),
    );
    add_test_payload(
        &mut artifact,
        "evaluators/direct.symjit",
        "evaluator-state",
        &direct_symjit_application_bytes(14),
        Some("p0"),
        true,
    );
    artifact.write_manifest();
    artifact
}

#[cfg(feature = "f64-symjit")]
fn color_topology_lane_load_error(execution: Value) -> crate::RusticolError {
    let artifact = color_topology_lane_artifact(execution);
    match crate::NativeRuntime::load(&artifact.root, Some("p0"), None) {
        Ok(_) => panic!("invalid compiled color topology lane unexpectedly loaded"),
        Err(error) => error,
    }
}

#[cfg(feature = "f64-symjit")]
#[test]
fn compiled_color_topology_lane_rejects_duplicate_materialized_sector_ids() {
    let mut execution = minimal_color_topology_lane_execution();
    let duplicate = execution["color_selector_executions"][0].clone();
    execution["color_selector_executions"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);

    let error = color_topology_lane_load_error(execution);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
    assert!(
        error
            .to_string()
            .contains("duplicate materialized sector 0")
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn compiled_color_topology_lane_rejects_recursive_auxiliary_executions() {
    for nested_field in [
        "helicity_sum_execution",
        "helicity_selector_executions",
        "color_selector_executions",
    ] {
        let mut execution = minimal_color_topology_lane_execution();
        let lane = &mut execution["color_selector_executions"][0]["execution"];
        let nested = lane.clone();
        lane[nested_field] = match nested_field {
            "helicity_sum_execution" => nested,
            "helicity_selector_executions" => {
                json!([{"selector_domain_ids": [0], "execution": nested}])
            }
            _ => json!([{"materialized_sector_id": 1, "execution": nested}]),
        };

        let error = color_topology_lane_load_error(execution);

        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
        assert!(
            error
                .to_string()
                .contains("color-selector execution cannot contain auxiliary executions"),
            "{nested_field}: {error}"
        );
    }
}

#[cfg(feature = "f64-symjit")]
#[test]
fn compiled_color_topology_lane_verifies_nested_evaluator_payloads() {
    let mut execution = minimal_color_topology_lane_execution();
    execution["color_selector_executions"][0]["execution"]["compiled"]["stage_evaluators"]["amplitude_stage"]
        ["evaluator"]["application_path"] = json!("evaluators/undeclared-lane.symjit");

    let error = color_topology_lane_load_error(execution);

    assert_eq!(error.kind(), crate::RusticolErrorKind::Security, "{error}");
    assert!(
        error.to_string().contains("undeclared-lane.symjit"),
        "{error}"
    );
}

#[cfg(feature = "f64-symjit")]
fn mixed_backend_runtime_artifact() -> TestArtifact {
    const DIRECT: &str = "symjit.application.complex-f64.v1";
    const ARENA: &str = crate::engine::COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY;
    const ASM: &str = "symbolica.compiled-asm.complex-f64.v1";
    const CPP: &str = "symbolica.compiled-cpp.complex-f64.v1";
    let mut artifact = TestArtifact::new();
    artifact.manifest["kind"] = json!("pyamplicol-process-set");
    artifact.manifest["default_process_id"] = json!("direct");
    artifact.manifest["processes"] = json!([
        {
            "id": "direct",
            "expression": "a b > c",
            "color_accuracy": "lc",
            "external_pdgs": [1, -1, 22],
            "physics_path": "processes/direct/physics.json",
            "required_runtime_capabilities": [ARENA, DIRECT],
            "aliases": [],
        },
        {
            "id": "asm",
            "expression": "a b > e",
            "color_accuracy": "lc",
            "external_pdgs": [1, -1, 22],
            "physics_path": "processes/asm/physics.json",
            "required_runtime_capabilities": [ASM],
            "aliases": [],
        },
        {
            "id": "cpp",
            "expression": "a b > d",
            "color_accuracy": "lc",
            "external_pdgs": [1, -1, 22],
            "physics_path": "processes/cpp/physics.json",
            "required_runtime_capabilities": [CPP],
            "aliases": [],
        }
    ]);
    artifact.manifest["runtime"]["evaluator_manifest_path"] = json!("processes/evaluators.json");
    artifact.manifest["runtime"]["required_runtime_capabilities"] =
        json!([ARENA, ASM, CPP, DIRECT]);

    let direct_application = direct_symjit_application_bytes(14);
    let direct_plane_application = direct_symjit_plane_application_bytes(14);
    let mut direct_execution = minimal_execution_manifest(
        "direct",
        "a b > c",
        DIRECT,
        direct_evaluator_manifest("evaluators/direct.symjit"),
    );
    direct_execution["compiled"]["momentum_slot_ids"] = json!([0, 1, 2]);
    let input_binding = |parameter_index: usize| {
        let (kind, source_id, component, global_component, real_valued) = if parameter_index < 2 {
            ("value", parameter_index, 0, parameter_index, false)
        } else {
            (
                "momentum",
                (parameter_index - 2) / 4,
                (parameter_index - 2) % 4,
                parameter_index,
                true,
            )
        };
        json!({
            "parameter_index": parameter_index,
            "kind": kind,
            "source_id": source_id,
            "component": component,
            "global_component": global_component,
            "real_valued": real_valued,
        })
    };
    let input_bindings = (0..14).map(input_binding).collect::<Vec<_>>();
    let direct_stage = &mut direct_execution["compiled"]["stage_evaluators"]["amplitude_stage"];
    direct_stage["parameter_layout"] = json!("stage-local-value-momentum");
    direct_stage["input_components"] = json!(input_bindings);
    direct_stage["evaluator"]["plane_application"] = json!({
        "application_path": "evaluators/direct.plane.symjit",
        "application_abi": crate::engine::SYMJIT_PLANE_APPLICATION_ABI,
        "storage_abi": crate::engine::SYMJIT_APPLICATION_STORAGE_ABI,
        "element_layout": "split-complex-plane-major",
        "descriptor_order": "inputs-re-im-then-outputs-re-im",
        "input_complex_count": 14,
        "output_complex_count": 1,
        "input_plane_count": 28,
        "output_plane_count": 2,
        "compiler_type": "native",
        "translation_mode": "symbolica-structured-instructions",
        "optimization_level": 3,
        "simd": true,
        "complex": true,
        "fast_math": true,
        "fast_complex": false,
        "compression": false,
        "threading": false,
        "direct_arena": true,
        "source_digest": sha256(DIRECT_SYMJIT_PROGRAM.as_bytes()),
        "target": {
            "word_bits": 64,
            "endianness": "little",
            "triple": "test-native",
            "cpu_features": [],
        },
    });
    direct_stage["evaluator"]["evaluator_state_path"] = json!("evaluators/direct.exact.symbolica");
    direct_stage["evaluator"]["evaluator_state_runtime_capability"] =
        json!(crate::engine::SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY);
    direct_stage["compiled_plane_arena"] = json!({
        "schema_version": 1,
        "kind": "compiled-plane-arena-stage",
        "application_abi": crate::engine::COMPILED_PLANE_DIRECT_APPLICATION_ABI,
        "source_application_abi": crate::engine::COMPILED_PLANE_SOURCE_APPLICATION_ABI,
        "element_layout": "split-complex-component-major",
        "output_operation": "overwrite",
        "output_factor": "identity",
        "input_output_aliasing": "forbidden",
        "output_output_aliasing": "forbidden",
        "input_bindings": (0..14).map(input_binding).collect::<Vec<_>>(),
        "output_bindings": [{
            "output_index": 0,
            "arena": "amplitude",
            "component": 0,
        }],
        "leaves": [{
            "application_path": "evaluators/direct.plane.symjit",
            "source_application_abi": crate::engine::COMPILED_PLANE_SOURCE_APPLICATION_ABI,
            "optimization_level": 3,
            "direct_codegen_optimization_level": 3,
            "input_len": 14,
            "output_len": 1,
            "input_indices": (0..14).collect::<Vec<_>>(),
            "output_start": 0,
            "output_stop": 1,
        }],
    });
    direct_execution["compiled"]["stage_evaluators"]["parameter_layout"] =
        json!("stage-local-value-momentum");
    direct_execution["compiled"]["stage_evaluators"]["parameter_count"] = json!(0);
    direct_execution["compiled"]["stage_evaluators"]["value_parameter_count"] = json!(0);
    direct_execution["compiled"]["stage_evaluators"]["momentum_parameter_count"] = json!(0);
    direct_execution["compiled"]["stage_evaluators"]["model_parameter_count"] = json!(0);
    direct_execution["compiled"]["stage_evaluators"]["real_valued_inputs"] = json!([]);
    direct_execution["compiled"]["stage_evaluators"]["required_runtime_capabilities"] =
        json!([ARENA, DIRECT]);
    direct_execution["required_runtime_capabilities"] = json!([ARENA, DIRECT]);
    let cpp_execution = minimal_execution_manifest(
        "cpp",
        "a b > d",
        CPP,
        compiled_evaluator_manifest("evaluators/fake-cpp.dylib", CPP),
    );
    let asm_execution = minimal_execution_manifest(
        "asm",
        "a b > e",
        ASM,
        compiled_evaluator_manifest("evaluators/fake-asm.dylib", ASM),
    );
    let execution_set = json!({
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution-set",
        "required_runtime_capabilities": [ARENA, ASM, CPP, DIRECT],
        "processes": [
            {
                "process_id": "direct",
                "manifest_path": "direct/execution.json",
                "required_runtime_capabilities": [ARENA, DIRECT]
            },
            {
                "process_id": "asm",
                "manifest_path": "asm/execution.json",
                "required_runtime_capabilities": [ASM]
            },
            {
                "process_id": "cpp",
                "manifest_path": "cpp/execution.json",
                "required_runtime_capabilities": [CPP]
            }
        ]
    });
    for (path, role, value, process_id) in [
        (
            "processes/evaluators.json",
            "evaluator-manifest",
            execution_set,
            None,
        ),
        (
            "processes/direct/execution.json",
            "evaluator-manifest",
            direct_execution,
            Some("direct"),
        ),
        (
            "processes/asm/execution.json",
            "evaluator-manifest",
            asm_execution,
            Some("asm"),
        ),
        (
            "processes/cpp/execution.json",
            "evaluator-manifest",
            cpp_execution,
            Some("cpp"),
        ),
        (
            "processes/direct/physics.json",
            "runtime-physics",
            minimal_runtime_physics("direct", "a b > c", "c"),
            Some("direct"),
        ),
        (
            "processes/asm/physics.json",
            "runtime-physics",
            minimal_runtime_physics("asm", "a b > e", "e"),
            Some("asm"),
        ),
        (
            "processes/cpp/physics.json",
            "runtime-physics",
            minimal_runtime_physics("cpp", "a b > d", "d"),
            Some("cpp"),
        ),
    ] {
        let bytes = serde_json::to_vec(&value).expect("serialize mixed-runtime JSON");
        add_test_payload(&mut artifact, path, role, &bytes, process_id, false);
    }
    add_test_payload(
        &mut artifact,
        "processes/direct/evaluators/direct.symjit",
        "evaluator-state",
        &direct_application,
        Some("direct"),
        true,
    );
    add_test_payload(
        &mut artifact,
        "processes/direct/evaluators/direct.plane.symjit",
        "evaluator-state",
        &direct_plane_application,
        Some("direct"),
        true,
    );
    add_test_payload(
        &mut artifact,
        "processes/direct/evaluators/direct.exact.symbolica",
        "evaluator-state",
        b"deferred exact-evaluator test state",
        Some("direct"),
        true,
    );
    add_test_payload(
        &mut artifact,
        "processes/asm/evaluators/fake-asm.dylib",
        "evaluator-state",
        b"unselected fake ASM evaluator",
        Some("asm"),
        true,
    );
    add_test_payload(
        &mut artifact,
        "processes/cpp/evaluators/fake-cpp.dylib",
        "evaluator-state",
        b"unselected fake C++ evaluator",
        Some("cpp"),
        true,
    );
    artifact.write_manifest();
    artifact
}

#[test]
fn artifact_identity_tracks_only_runtime_payload_records() {
    let mut value = json!({
        "artifact_id": "0".repeat(64),
        "created_utc": "2026-07-31T00:00:00Z",
        "payloads": [
            {
                "path": "config/effective.toml",
                "role": "configuration-effective",
                "media_type": "application/toml",
                "size_bytes": 1,
                "sha256": "1".repeat(64),
                "executable": false,
            },
            {
                "path": "processes/p0/validation-momenta.json",
                "role": "validation-momenta",
                "media_type": "application/json",
                "size_bytes": 1,
                "sha256": "2".repeat(64),
                "executable": false,
            },
            {
                "path": "processes/p0/physics.json",
                "role": "runtime-physics",
                "media_type": "application/json",
                "size_bytes": 1,
                "sha256": "3".repeat(64),
                "executable": false,
                "process_id": "p0",
            },
        ],
    });
    let identity = compute_artifact_id(&value).unwrap();

    value["created_utc"] = json!("2030-01-01T00:00:00Z");
    value["payloads"][0]["sha256"] = json!("a".repeat(64));
    value["payloads"][1]["sha256"] = json!("b".repeat(64));
    assert_eq!(compute_artifact_id(&value).unwrap(), identity);

    value["payloads"][2]["sha256"] = json!("c".repeat(64));
    assert_ne!(compute_artifact_id(&value).unwrap(), identity);
}

#[test]
fn valid_manifest_opens_trusted_payload_catalog() {
    let artifact = TestArtifact::new();
    let verified = VerifiedArtifact::open(&artifact.root).expect("valid artifact");

    assert_eq!(verified.manifest().schema_version, 3);
    assert_eq!(verified.select_process(None).unwrap().requested_id, "p0");
    assert_eq!(
        verified
            .select_process(Some("  A   b  >  C "))
            .expect("select process by case- and whitespace-normalized expression")
            .requested_id,
        "p0"
    );
}

#[test]
fn valid_manifest_larger_than_legacy_16_mib_guard_opens() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["extensions"]["large_valid_metadata"] = json!("x".repeat(16 * 1024 * 1024));
    artifact.write_manifest();
    let manifest_path = artifact.root.join(ARTIFACT_MANIFEST_FILE);
    assert!(
        fs::metadata(&manifest_path)
            .expect("large manifest metadata")
            .len()
            > 16 * 1024 * 1024
    );

    let verified = VerifiedArtifact::open(&artifact.root).expect("large valid artifact manifest");
    assert_eq!(
        verified
            .manifest()
            .extensions
            .get("large_valid_metadata")
            .and_then(Value::as_str)
            .map(str::len),
        Some(16 * 1024 * 1024)
    );
}

#[test]
fn artifact_without_current_identity_contract_fails_closed() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["extensions"]
        .as_object_mut()
        .unwrap()
        .remove(ARTIFACT_IDENTITY_EXTENSION);
    artifact.write_manifest();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("regenerate"));
}

#[cfg(feature = "f64-symjit")]
#[test]
fn verified_artifact_resolves_packed_evaluator_payloads() {
    let mut artifact = TestArtifact::new();
    add_test_evaluator_container(&mut artifact);

    let verified = VerifiedArtifact::open(&artifact.root).expect("packed artifact");
    assert!(verified.has_evaluator_payload("a/jit.symjit").unwrap());
    assert!(verified.has_evaluator_payload("z/exact.bin").unwrap());
    assert!(!verified.has_evaluator_payload("missing.symjit").unwrap());

    let store = verified
        .evaluator_payload_store(verified.root())
        .expect("root evaluator store");
    let application = store
        .source("a/jit.symjit")
        .expect("packed application source");
    assert_eq!(application.read().unwrap().as_ref(), b"jit-v1");
    assert_eq!(application.display_name(), "evaluators.pacbin:a/jit.symjit");
    assert_eq!(
        store
            .source("z/exact.bin")
            .expect("packed exact source")
            .read()
            .unwrap()
            .as_ref(),
        b"exact-state"
    );
}

#[test]
fn evaluator_payload_store_resolves_process_relative_paths_and_rejects_escape() {
    let mut artifact = TestArtifact::new();
    let declared = "processes/p0/on-the-fly-runtime.pacbin";
    add_test_payload(
        &mut artifact,
        declared,
        "evaluator-state",
        b"authenticated on-the-fly container",
        Some("p0"),
        true,
    );
    artifact.write_manifest();

    let verified = VerifiedArtifact::open(&artifact.root).expect("process payload artifact");
    let process_root = verified.root().join("processes/p0");
    let store = verified
        .evaluator_payload_store(&process_root)
        .expect("process evaluator store");
    let logical = store
        .logical_path("on-the-fly-runtime.pacbin")
        .expect("process-relative payload path");
    assert_eq!(logical, declared);
    assert_eq!(verified.payload(&logical).unwrap().path, declared);

    let error = store.logical_path("../escape.pacbin").unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);

    let missing = store
        .logical_path("missing.pacbin")
        .expect("confined undeclared path");
    let error = verified.payload(&missing).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);
    assert!(
        error.to_string().contains("not a declared payload"),
        "{error}"
    );
}

#[cfg(all(unix, feature = "f64-symjit"))]
#[test]
fn trusted_loose_evaluator_source_reads_replacement_bytes() {
    let mut artifact = TestArtifact::new();
    add_test_payload(
        &mut artifact,
        "loose.symjit",
        "evaluator-state",
        b"jit-v1",
        Some("p0"),
        true,
    );
    artifact.write_manifest();
    let verified = VerifiedArtifact::open(&artifact.root).expect("loose evaluator artifact");
    let source = verified
        .evaluator_payload_store(verified.root())
        .expect("verified evaluator store")
        .source("loose.symjit")
        .expect("loose evaluator source");
    let replacement_path = artifact.root.join("loose.replacement");
    fs::write(&replacement_path, b"bad-v1").expect("write replacement evaluator");
    fs::rename(&replacement_path, artifact.root.join("loose.symjit"))
        .expect("replace loose evaluator path");

    assert_eq!(
        source.read().expect("read trusted replacement").as_ref(),
        b"bad-v1"
    );
}

#[cfg(all(unix, feature = "f64-symjit"))]
#[test]
fn evaluator_container_replacement_after_validation_is_reauthenticated() {
    let mut artifact = TestArtifact::new();
    add_test_evaluator_container(&mut artifact);
    let verified = VerifiedArtifact::open(&artifact.root).expect("packed artifact");
    let container_path = artifact.root.join("evaluators.pacbin");
    let replacement_path = artifact.root.join("evaluators.replacement");
    let mut replacement = decode_hex(PYTHON_PACBIN_GOLDEN_HEX);
    let payload_offset = replacement
        .windows(b"jit-v1".len())
        .position(|window| window == b"jit-v1")
        .expect("golden JIT member");
    replacement[payload_offset..payload_offset + b"jit-v1".len()].copy_from_slice(b"bad-v1");
    fs::write(&replacement_path, replacement).expect("write replacement PACBIN");
    fs::rename(&replacement_path, &container_path).expect("replace validated PACBIN path");

    let error =
        load_evaluator_payload_container(&verified.root, &verified.manifest, &verified.payloads)
            .expect_err("replacement PACBIN bytes must not inherit prior validation");

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("payload digest mismatch"));
}

#[cfg(feature = "f64-symjit")]
#[test]
fn packed_evaluator_metadata_and_physical_collisions_are_rejected() {
    let mut count_mismatch = TestArtifact::new();
    add_test_evaluator_container(&mut count_mismatch);
    count_mismatch.manifest["extensions"]["evaluator_payload_container"]["member_count"] = json!(3);
    count_mismatch.write_manifest();
    let error = VerifiedArtifact::open(&count_mismatch.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("metadata disagrees"));

    let mut collision = TestArtifact::new();
    add_test_evaluator_container(&mut collision);
    add_test_payload(
        &mut collision,
        "a/jit.symjit",
        "evaluator-state",
        b"physical collision",
        Some("p0"),
        true,
    );
    collision.write_manifest();
    let error = VerifiedArtifact::open(&collision.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(
        error
            .to_string()
            .contains("invalid packed evaluator member")
    );
}

#[test]
fn ambiguous_process_expression_lists_stable_ids() {
    let artifact = TestArtifact::new();
    let verified = VerifiedArtifact::open(&artifact.root).expect("valid artifact");
    let mut manifest = verified.manifest().clone();
    let mut duplicate = manifest.processes[0].clone();
    duplicate.id = "p1".to_string();
    manifest.processes.push(duplicate);

    let error = manifest
        .select_process(Some("a b > c"))
        .expect_err("duplicate expressions must be ambiguous");

    assert_eq!(error.kind(), crate::RusticolErrorKind::Selector);
    assert!(error.to_string().contains("ambiguous"));
    assert!(error.to_string().contains("p0, p1"));
}

#[test]
fn direct_runtime_defers_symbolica_serialization_abi_validation() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["producer"]["versions"]["symbolica_serialization"] =
        json!("a-future-symbolica-serialization-abi");
    artifact.write_manifest();

    let verified = VerifiedArtifact::open(&artifact.root)
        .expect("outer artifact validation must not require an unused Symbolica serialization ABI");

    assert_eq!(
        verified
            .manifest()
            .producer
            .versions
            .symbolica_serialization,
        Some("a-future-symbolica-serialization-abi".to_string())
    );

    let mut no_symbolica_abi = TestArtifact::new();
    no_symbolica_abi.manifest["producer"]["versions"]
        .as_object_mut()
        .unwrap()
        .remove("symbolica_serialization");
    no_symbolica_abi.write_manifest();

    let verified = VerifiedArtifact::open(&no_symbolica_abi.root)
        .expect("direct f64 artifact must not require Symbolica ABI metadata");
    assert_eq!(
        verified
            .manifest()
            .producer
            .versions
            .symbolica_serialization,
        None
    );
}

#[test]
fn artifact_identity_is_trusted_during_normal_loading() {
    let artifact = TestArtifact::new();
    let path = artifact.root.join(ARTIFACT_MANIFEST_FILE);
    let mut manifest: Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).expect("parse fixture manifest");
    manifest["model"]["name"] = json!("tampered-model");
    fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let opened = VerifiedArtifact::open(&artifact.root).expect("open trusted manifest");
    assert_eq!(opened.manifest().model.name, "tampered-model");
}

#[test]
fn unknown_nested_manifest_keys_are_rejected() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["producer"]["target"]["unexpected"] = json!(true);
    artifact.write_manifest();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Serialization);
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn only_canonical_artifact_filename_is_accepted() {
    let artifact = TestArtifact::new();
    let legacy = artifact.root.join("manifest.json");
    fs::rename(artifact.root.join(ARTIFACT_MANIFEST_FILE), &legacy)
        .expect("rename canonical manifest");

    let directory_error = VerifiedArtifact::open(&artifact.root).unwrap_err();
    let direct_error = VerifiedArtifact::open(&legacy).unwrap_err();

    assert_eq!(directory_error.kind(), crate::RusticolErrorKind::Artifact);
    assert_eq!(direct_error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(directory_error.to_string().contains("artifact.json"));
    assert!(direct_error.to_string().contains("artifact.json"));
}

#[test]
fn legacy_schema_reports_actionable_regeneration_error() {
    let mut artifact = TestArtifact::new();
    artifact.manifest = json!({"schema_version": 2, "kind": "legacy"});
    artifact.write_manifest();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("regenerate"));
    assert!(error.to_string().contains("schema v3"));
}

#[test]
fn traversal_and_duplicate_payload_paths_are_rejected() {
    let mut traversal = TestArtifact::new();
    traversal.manifest["payloads"][3]["path"] = json!("../physics.json");
    traversal.write_manifest();
    let error = VerifiedArtifact::open(&traversal.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);

    let mut duplicate = TestArtifact::new();
    duplicate.manifest["payloads"][1]["path"] = json!("requested.toml");
    duplicate.write_manifest();
    let error = VerifiedArtifact::open(&duplicate.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);
    assert!(error.to_string().contains("duplicate"));
}

#[test]
fn payload_size_is_checked_when_payload_is_read() {
    let artifact = TestArtifact::new();
    fs::write(artifact.root.join("physics.json"), b"tampered").unwrap();

    let opened = VerifiedArtifact::open(&artifact.root).expect("open trusted manifest");
    let error = opened.read_payload("physics.json").unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("size"));
}

#[test]
fn manifest_preflight_runs_without_payload_verification() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    artifact.manifest["runtime"]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    artifact.write_manifest();
    fs::write(artifact.root.join("physics.json"), b"tampered").unwrap();

    let error = VerifiedArtifact::open_with_manifest_preflight(&artifact.root, |manifest| {
        assert_eq!(
            manifest.runtime.required_runtime_capabilities,
            ["symbolica.compiled-cpp.complex-f64.v1"]
        );
        Err(RusticolError::unsupported_runtime_capability(
            &manifest.runtime.required_runtime_capabilities[0],
            "test runtime supports direct SymJIT only",
        ))
    })
    .unwrap_err();

    assert_eq!(
        error.kind(),
        crate::RusticolErrorKind::UnsupportedRuntimeCapability
    );
}

#[test]
fn process_capabilities_are_strict_and_form_the_runtime_union() {
    for (capabilities, message) in [
        (json!([]), "at least one"),
        (
            json!([
                "symjit.application.complex-f64.v1",
                "symjit.application.complex-f64.v1"
            ]),
            "duplicates",
        ),
        (
            json!([
                "symjit.application.complex-f64.v1",
                "symbolica.compiled-cpp.complex-f64.v1"
            ]),
            "sorted",
        ),
    ] {
        let mut artifact = TestArtifact::new();
        artifact.manifest["processes"][0]["required_runtime_capabilities"] = capabilities.clone();
        artifact.manifest["runtime"]["required_runtime_capabilities"] = capabilities;
        artifact.write_manifest();

        let error = VerifiedArtifact::open(&artifact.root).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
        assert!(error.to_string().contains(message), "{error}");
    }

    let mut mismatch = TestArtifact::new();
    mismatch.manifest["runtime"]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    mismatch.write_manifest();
    let error = VerifiedArtifact::open(&mismatch.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("union"));

    let mut mixed = TestArtifact::new();
    mixed.manifest["kind"] = json!("pyamplicol-process-set");
    mixed.manifest["processes"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id": "p1",
            "expression": "a b > d",
            "color_accuracy": "lc",
            "external_pdgs": [1, -1, 23],
            "physics_path": "physics.json",
            "required_runtime_capabilities": [
                "symbolica.compiled-cpp.complex-f64.v1"
            ],
            "aliases": [],
        }));
    mixed.manifest["runtime"]["required_runtime_capabilities"] = json!([
        "symbolica.compiled-cpp.complex-f64.v1",
        "symjit.application.complex-f64.v1"
    ]);
    let parsed: ArtifactManifest =
        serde_json::from_value(mixed.manifest.clone()).expect("parse mixed manifest");
    validate_manifest(&parsed).expect("mixed process capability union validates");
}

#[test]
fn on_the_fly_fft_capabilities_are_known_and_the_catalog_remains_closed() {
    let capabilities = json!([
        crate::engine::SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
        "rusticol.on-the-fly.complex-f64.v1",
        "rusticol.on-the-fly.contracted-color.v1",
        "rusticol.on-the-fly.lc-color.v1"
    ]);
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["required_runtime_capabilities"] = capabilities.clone();
    artifact.manifest["runtime"]["required_runtime_capabilities"] = capabilities;
    artifact.write_manifest();

    VerifiedArtifact::open(&artifact.root)
        .expect("on-the-fly capabilities pass outer artifact validation");

    let unknown = json!(["rusticol.on-the-fly.unknown.v1"]);
    artifact.manifest["processes"][0]["required_runtime_capabilities"] = unknown.clone();
    artifact.manifest["runtime"]["required_runtime_capabilities"] = unknown;
    artifact.write_manifest();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(
        error.to_string().contains("unsupported capabilities"),
        "{error}"
    );
}

#[test]
fn recurrence_capabilities_are_part_of_the_outer_artifact_catalog() {
    let capabilities = json!([
        "rusticol.recurrence-color.lc.v1",
        "rusticol.recurrence-direct-arena.complex-f64.v1"
    ]);
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["required_runtime_capabilities"] = capabilities.clone();
    artifact.manifest["runtime"]["required_runtime_capabilities"] = capabilities;
    artifact.write_manifest();

    VerifiedArtifact::open(&artifact.root)
        .expect("recurrence capabilities pass outer artifact validation");
}

#[cfg(all(
    feature = "f64-symjit",
    not(feature = "f64-compiled"),
    not(feature = "symbolica-runtime")
))]
#[test]
fn selected_unsupported_process_capability_preflights_before_payloads() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    artifact.manifest["runtime"]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    artifact.write_manifest();
    fs::write(artifact.root.join("physics.json"), b"tampered").unwrap();

    let error = match crate::NativeRuntime::load(&artifact.root, Some("p0"), None) {
        Ok(_) => panic!("unsupported selected process unexpectedly loaded"),
        Err(error) => error,
    };

    assert_eq!(
        error.kind(),
        crate::RusticolErrorKind::UnsupportedRuntimeCapability
    );
    assert!(error.to_string().contains("compiled-cpp"));
}

#[cfg(feature = "f64-symjit")]
#[test]
fn mixed_backend_process_set_loads_selected_direct_symjit_process() {
    let artifact = mixed_backend_runtime_artifact();

    let runtime = crate::NativeRuntime::load(&artifact.root, Some("direct"), None)
        .expect("selected direct-SymJIT process loads despite unselected C++/ASM processes");

    assert_eq!(runtime.metadata().process_key, "direct");
    assert_eq!(runtime.metadata().external_pdg_order, vec![1, -1, 22]);
}

#[cfg(all(
    feature = "f64-symjit",
    not(feature = "f64-compiled"),
    not(feature = "symbolica-runtime")
))]
#[test]
fn mixed_backend_process_set_rejects_unsupported_selection_before_tampered_payload() {
    let artifact = mixed_backend_runtime_artifact();
    fs::write(
        artifact.root.join("processes/direct/physics.json"),
        b"tampered after manifest creation",
    )
    .unwrap();

    for (process_id, capability) in [("asm", "compiled-asm"), ("cpp", "compiled-cpp")] {
        let error = match crate::NativeRuntime::load(&artifact.root, Some(process_id), None) {
            Ok(_) => panic!("unsupported selected {process_id} process unexpectedly loaded"),
            Err(error) => error,
        };

        assert_eq!(
            error.kind(),
            crate::RusticolErrorKind::UnsupportedRuntimeCapability
        );
        assert!(error.to_string().contains(capability));
    }
}

#[test]
fn required_nullable_and_target_fields_remain_strict() {
    let mut missing_nullable = TestArtifact::new();
    missing_nullable.manifest["runtime"]
        .as_object_mut()
        .unwrap()
        .remove("api_bundle_path");
    missing_nullable.write_manifest();
    let error = VerifiedArtifact::open(&missing_nullable.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("api_bundle_path"));

    let mut missing_target = TestArtifact::new();
    let state = b"serialized state";
    fs::write(missing_target.root.join("state.bin"), state).unwrap();
    missing_target.manifest["payloads"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "path": "state.bin",
            "role": "evaluator-state",
            "media_type": "application/octet-stream",
            "size_bytes": state.len(),
            "sha256": sha256(state),
            "executable": false,
        }));
    missing_target.write_manifest();
    let error = VerifiedArtifact::open(&missing_target.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("target"));
}

#[test]
fn clean_source_structural_proof_payload_roundtrips_with_exact_declaration() {
    let mut valid = TestArtifact::new();
    valid.manifest["producer"]["git_revision"] = json!("a".repeat(40));
    valid.manifest["producer"]["native_build_inputs_sha256"] = json!("b".repeat(64));
    add_structural_source_proof(&mut valid);
    let verified = VerifiedArtifact::open(&valid.root).expect("valid structural proof artifact");
    assert_eq!(
        verified.manifest.producer.git_revision.as_deref(),
        Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    );
    assert_eq!(
        verified
            .manifest
            .producer
            .native_build_inputs_sha256
            .as_deref(),
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    );
    let payload = verified
        .payloads
        .get("processes/p0/structural-source-proof.json")
        .expect("verified structural proof payload");
    assert_eq!(payload.role, PayloadRole::StructuralSourceProof);
    assert_eq!(payload.media_type, "application/json");
    assert_eq!(payload.process_id.as_deref(), Some("p0"));
    assert!(payload.target.is_none());

    for (field, value, message) in [
        ("path", json!("structural-source-proof.json"), "exact path"),
        (
            "media_type",
            json!("application/octet-stream"),
            "media type",
        ),
        ("process_id", json!("missing"), "unknown process"),
        (
            "target",
            json!({"triple": current_target_triple(), "cpu_features": []}),
            "target metadata",
        ),
    ] {
        let mut malformed = TestArtifact::new();
        add_structural_source_proof(&mut malformed);
        {
            let payload = malformed.manifest["payloads"]
                .as_array_mut()
                .expect("payload array")
                .last_mut()
                .expect("structural proof payload");
            payload[field] = value;
            if field == "process_id" {
                payload["path"] = json!("processes/missing/structural-source-proof.json");
            }
        }
        if field == "process_id" {
            let destination = malformed
                .root
                .join("processes/missing/structural-source-proof.json");
            fs::create_dir_all(destination.parent().expect("structural proof parent"))
                .expect("create structural proof parent");
            fs::rename(
                malformed
                    .root
                    .join("processes/p0/structural-source-proof.json"),
                destination,
            )
            .expect("move structural proof to its exact declaration");
        }
        malformed.write_manifest();
        let error = VerifiedArtifact::open(&malformed.root).unwrap_err();
        assert!(
            error.to_string().contains(message),
            "{field} rejection did not mention {message:?}: {error}"
        );
    }

    let mut missing_process = TestArtifact::new();
    add_structural_source_proof(&mut missing_process);
    missing_process.manifest["payloads"]
        .as_array_mut()
        .expect("payload array")
        .last_mut()
        .expect("structural proof payload")
        .as_object_mut()
        .expect("payload object")
        .remove("process_id");
    missing_process.write_manifest();
    let error = VerifiedArtifact::open(&missing_process.root).unwrap_err();
    assert!(error.to_string().contains("required process id"));

    let mut malformed_native_identity = TestArtifact::new();
    malformed_native_identity.manifest["producer"]["git_revision"] = json!("a".repeat(40));
    malformed_native_identity.manifest["producer"]["native_build_inputs_sha256"] =
        json!("not-a-sha256");
    malformed_native_identity.write_manifest();
    let error = VerifiedArtifact::open(&malformed_native_identity.root).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("producer.native_build_inputs_sha256")
    );

    let mut native_identity_without_revision = TestArtifact::new();
    native_identity_without_revision.manifest["producer"]["native_build_inputs_sha256"] =
        json!("b".repeat(64));
    native_identity_without_revision.write_manifest();
    let verified = VerifiedArtifact::open(&native_identity_without_revision.root)
        .expect("native build-input identity is valid without a source revision");
    assert_eq!(verified.manifest.producer.git_revision, None);
    assert_eq!(
        verified
            .manifest
            .producer
            .native_build_inputs_sha256
            .as_deref(),
        Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    );
}

#[test]
fn producer_and_payload_cpu_features_are_checked() {
    let mut unavailable = TestArtifact::new();
    unavailable.manifest["producer"]["target"]["cpu_features"] =
        json!(["definitely-not-a-real-cpu-feature"]);
    unavailable.write_manifest();
    fs::write(unavailable.root.join("physics.json"), b"tampered payload").unwrap();
    let error = VerifiedArtifact::open(&unavailable.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("CPU feature"));

    let mut mismatch = TestArtifact::new();
    mismatch.manifest["payloads"][2]["target"] = json!({
        "triple": current_target_triple(),
        "cpu_features": ["baseline"],
    });
    mismatch.write_manifest();
    let error = VerifiedArtifact::open(&mismatch.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(
        error
            .to_string()
            .contains("do not match producer CPU features")
    );
}

#[test]
fn portable_64le_target_accepts_only_feature_free_symjit_artifacts() {
    let mut portable = TestArtifact::new();
    portable.manifest["producer"]["target"] = json!({
        "triple": PORTABLE_64LE_ARTIFACT_TARGET,
        "cpu_features": [],
    });
    portable.write_manifest();
    VerifiedArtifact::open(&portable.root).expect("portable O2 SymJIT target");

    let mut feature_bound = TestArtifact::new();
    feature_bound.manifest["producer"]["target"] = json!({
        "triple": PORTABLE_64LE_ARTIFACT_TARGET,
        "cpu_features": ["avx2"],
    });
    feature_bound.write_manifest();
    let error = VerifiedArtifact::open(&feature_bound.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("must not require CPU features"));

    for capability in [
        RuntimeCapability::SymbolicaLegacyJitContainerComplexF64V1.as_str(),
        RuntimeCapability::SymbolicaCompiledCppComplexF64V1.as_str(),
        RuntimeCapability::SymbolicaCompiledAsmComplexF64V1.as_str(),
    ] {
        let mut target_specific = TestArtifact::new();
        target_specific.manifest["producer"]["target"] = json!({
            "triple": PORTABLE_64LE_ARTIFACT_TARGET,
            "cpu_features": [],
        });
        target_specific.manifest["runtime"]["required_runtime_capabilities"] = json!([capability]);
        target_specific.manifest["processes"][0]["required_runtime_capabilities"] =
            json!([capability]);
        target_specific.write_manifest();
        let error = VerifiedArtifact::open(&target_specific.root).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.to_string().contains("remain target-specific"));
    }

    for capability in [
        RuntimeCapability::EagerDirectArenaV1.as_str(),
        RuntimeCapability::RecurrenceRuntimeComplexF64V1.as_str(),
    ] {
        let mut direct = TestArtifact::new();
        direct.manifest["producer"]["target"] = json!({
            "triple": PORTABLE_64LE_ARTIFACT_TARGET,
            "cpu_features": [],
        });
        direct.manifest["runtime"]["required_runtime_capabilities"] = json!([capability]);
        direct.manifest["processes"][0]["required_runtime_capabilities"] = json!([capability]);
        direct.write_manifest();
        VerifiedArtifact::open(&direct.root)
            .expect("portable eager/recurrence outer capability is pack-authenticated later");
    }
}

#[cfg(feature = "f64-symjit")]
#[test]
fn packaged_portable_64le_o2_artifact_loads_through_native_runtime() {
    let source = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../src/pyamplicol/assets/selftest/portable-64le/artifact"
    ));
    let mut staged = TestArtifact::new();
    fs::remove_dir_all(&staged.root).expect("remove synthetic test artifact");
    copy_test_artifact_tree(&source, &staged.root);
    staged.manifest = serde_json::from_slice(
        &fs::read(staged.root.join(ARTIFACT_MANIFEST_FILE)).expect("read portable manifest"),
    )
    .expect("parse portable manifest");
    let source_artifact_id = staged.manifest["artifact_id"].clone();
    staged.manifest["producer"]["version"] = json!(env!("CARGO_PKG_VERSION"));
    staged.manifest["runtime"]["engine_version"] = json!(env!("CARGO_PKG_VERSION"));
    staged.write_manifest();
    assert_eq!(staged.manifest["artifact_id"], source_artifact_id);

    let artifact =
        VerifiedArtifact::open(&staged.root).expect("open staged portable source artifact");
    assert_eq!(
        artifact.manifest().producer.target.triple,
        PORTABLE_64LE_ARTIFACT_TARGET
    );

    let runtime = crate::NativeRuntime::load(&staged.root, Some("d_dbar_to_z"), None)
        .expect("load portable O2 artifact through the authoritative runtime");
    assert_eq!(runtime.metadata().execution_mode, "compiled");
}

#[cfg(feature = "f64-symjit")]
fn copy_test_artifact_tree(source: &Path, destination: &Path) {
    fs::create_dir(destination).expect("create staged portable artifact");
    for entry in fs::read_dir(source).expect("read portable artifact tree") {
        let entry = entry.expect("read portable artifact entry");
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type().expect("read portable artifact file type");
        if file_type.is_dir() {
            copy_test_artifact_tree(&source_path, &destination_path);
        } else {
            assert!(file_type.is_file(), "portable artifact contains a symlink");
            fs::copy(&source_path, &destination_path).expect("copy portable artifact payload");
        }
    }
}

#[test]
fn target_metadata_requires_canonical_sorted_cpu_features() {
    let target = runtime_target_info();
    assert_eq!(target.triple, current_target_triple());
    assert!(target.cpu_features.windows(2).all(|pair| pair[0] < pair[1]));
    validate_target(&target, "runtime target").expect("runtime target validates");

    let mut unsorted = TestArtifact::new();
    unsorted.manifest["producer"]["target"]["cpu_features"] = json!(["z-feature", "a-feature"]);
    unsorted.write_manifest();
    let error = VerifiedArtifact::open(&unsorted.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("sorted and unique"));

    let mut noncanonical = TestArtifact::new();
    noncanonical.manifest["producer"]["target"]["cpu_features"] = json!(["+avx2"]);
    noncanonical.write_manifest();
    let error = VerifiedArtifact::open(&noncanonical.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("canonical feature ID"));
}

#[test]
fn malformed_timestamp_and_crossing_alias_are_rejected() {
    let mut timestamp = TestArtifact::new();
    timestamp.manifest["created_utc"] = json!("2026-02-30T25:00:00Z");
    timestamp.write_manifest();
    let error = VerifiedArtifact::open(&timestamp.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);

    let mut alias = TestArtifact::new();
    alias.manifest["processes"][0]["aliases"] = json!([{
        "id": "alias0",
        "expression": "b a > c",
        "external_pdgs": [-1, 1, 22],
        "external_permutation": [1, 0, 2],
    }]);
    alias.write_manifest();
    VerifiedArtifact::open(&alias.root)
        .expect("incoming particles may be permuted within their side");

    alias.manifest["processes"][0]["aliases"] = json!([{
        "id": "alias0",
        "expression": "a c > b",
        "external_pdgs": [1, 22, -1],
        "external_permutation": [0, 2, 1],
    }]);
    alias.write_manifest();
    let error = VerifiedArtifact::open(&alias.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("incoming and outgoing sides"));
}

#[test]
fn concrete_process_expression_infers_a_deterministic_side_preserving_permutation() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["expression"] = json!("d d~ > z g g");
    artifact.manifest["processes"][0]["external_pdgs"] = json!([1, -1, 23, 21, 21]);
    artifact.write_manifest();

    let verified = VerifiedArtifact::open(&artifact.root).unwrap();
    let selection = verified
        .select_process(Some(" d~   d > g z g "))
        .expect("unique permutation-equivalent expression");
    assert_eq!(selection.requested_id, "p0");
    assert!(selection.alias.is_none());
    assert!(selection.inferred_permutation);
    assert_eq!(selection.public_expression, "d~ d > g z g");
    assert_eq!(selection.external_permutation, [1, 0, 3, 2, 4]);
    assert_eq!(selection.external_pdgs, [-1, 1, 21, 23, 21]);

    let crossing = verified.select_process(Some("d z > d~ g g")).unwrap_err();
    assert_eq!(crossing.kind(), crate::RusticolErrorKind::Selector);
    assert!(crossing.to_string().contains("across the '>' boundary"));
}

#[test]
fn inferred_expression_prefers_the_least_reordered_representative() {
    let artifact = TestArtifact::new();
    let mut manifest: ArtifactManifest = serde_json::from_value(artifact.manifest.clone()).unwrap();
    manifest.processes[0].expression = "d d~ > z g g".to_string();
    manifest.processes[0].external_pdgs = vec![1, -1, 23, 21, 21];
    let mut reversed = manifest.processes[0].clone();
    reversed.id = "p1".to_string();
    reversed.expression = "d~ d > z g g".to_string();
    reversed.external_pdgs = vec![-1, 1, 23, 21, 21];
    manifest.processes.push(reversed);

    let forward = manifest.select_process(Some("d d~ > g z g")).unwrap();
    assert_eq!(forward.process.id, "p0");
    assert_eq!(forward.external_permutation, [0, 1, 3, 2, 4]);

    let reversed = manifest.select_process(Some("d~ d > g z g")).unwrap();
    assert_eq!(reversed.process.id, "p1");
    assert_eq!(reversed.external_permutation, [0, 1, 3, 2, 4]);
}

#[test]
fn exact_representative_expression_precedes_an_equal_alias_expression() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["aliases"] = json!([{
        "id": "same-expression-alias",
        "expression": "a b > c",
        "external_pdgs": [1, -1, 22],
        "external_permutation": [0, 1, 2],
    }]);
    artifact.write_manifest();

    let verified = VerifiedArtifact::open(&artifact.root).unwrap();
    let selection = verified.select_process(Some("a b > c")).unwrap();

    assert_eq!(selection.requested_id, "p0");
    assert!(selection.alias.is_none());
}

#[test]
fn inferred_process_expression_fails_on_multiple_representatives() {
    let artifact = TestArtifact::new();
    let mut manifest: ArtifactManifest = serde_json::from_value(artifact.manifest.clone()).unwrap();
    manifest.processes[0].expression = "d d~ > z g g".to_string();
    manifest.processes[0].external_pdgs = vec![1, -1, 23, 21, 21];
    let mut duplicate = manifest.processes[0].clone();
    duplicate.id = "p1".to_string();
    manifest.processes.push(duplicate);

    let error = manifest.select_process(Some("d d~ > g z g")).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Selector);
    assert!(error.to_string().contains("permutation-ambiguous"));
    assert!(error.to_string().contains("p0, p1"));
}

#[test]
fn non_self_inverse_alias_pdg_order_is_verified() {
    let mut valid = TestArtifact::new();
    valid.manifest["processes"][0]["expression"] = json!("d d~ > z g a");
    valid.manifest["processes"][0]["external_pdgs"] = json!([1, -1, 23, 21, 22]);
    valid.manifest["processes"][0]["aliases"] = json!([{
        "id": "cycled",
        "expression": "d d~ > a z g",
        "external_pdgs": [1, -1, 22, 23, 21],
        "external_permutation": [0, 1, 3, 4, 2],
    }]);
    valid.write_manifest();

    let verified = VerifiedArtifact::open(&valid.root).expect("valid three-cycle alias");
    let selected = verified
        .select_process(Some("cycled"))
        .expect("select three-cycle alias");
    assert_eq!(
        selected.process.required_runtime_capabilities,
        ["symjit.application.complex-f64.v1"]
    );
    assert_eq!(
        selected.alias.expect("alias metadata").external_pdgs,
        vec![1, -1, 22, 23, 21]
    );
    let selected_by_expression = verified
        .select_process(Some("d   d~ > a z g"))
        .expect("select three-cycle alias by expression");
    assert_eq!(selected_by_expression.requested_id, "cycled");
    assert_eq!(
        selected_by_expression
            .alias
            .expect("alias metadata selected by expression")
            .external_pdgs,
        vec![1, -1, 22, 23, 21]
    );

    let mut inconsistent = TestArtifact::new();
    inconsistent.manifest["processes"][0]["expression"] = json!("d d~ > z g a");
    inconsistent.manifest["processes"][0]["external_pdgs"] = json!([1, -1, 23, 21, 22]);
    inconsistent.manifest["processes"][0]["aliases"] = json!([{
        "id": "cycled",
        "expression": "d d~ > a z g",
        "external_pdgs": [1, -1, 21, 23, 22],
        "external_permutation": [0, 1, 3, 4, 2],
    }]);
    inconsistent.write_manifest();

    let error = VerifiedArtifact::open(&inconsistent.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(
        error
            .to_string()
            .contains("does not match external_permutation")
    );
}

#[cfg(unix)]
#[test]
fn referenced_symlink_payload_is_rejected_when_read() {
    use std::os::unix::fs::symlink;

    let artifact = TestArtifact::new();
    fs::rename(
        artifact.root.join("physics.json"),
        artifact.root.join("physics-target.json"),
    )
    .unwrap();
    symlink("physics-target.json", artifact.root.join("physics.json")).unwrap();

    let opened = VerifiedArtifact::open(&artifact.root).expect("open trusted manifest");
    let error = opened.read_payload("physics.json").unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);
    assert!(error.to_string().contains("symlink"));
}

#[cfg(unix)]
#[test]
fn undeclared_tree_entries_are_ignored_for_trusted_artifacts() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let symlink_artifact = TestArtifact::new();
    fs::create_dir(symlink_artifact.root.join("extra")).unwrap();
    symlink(
        "../physics.json",
        symlink_artifact.root.join("extra/undeclared-link"),
    )
    .unwrap();
    VerifiedArtifact::open(&symlink_artifact.root).expect("ignore undeclared symlink");

    let executable_artifact = TestArtifact::new();
    let executable = executable_artifact.root.join("undeclared-tool");
    fs::write(&executable, b"#!/bin/sh\n").unwrap();
    let mut permissions = fs::metadata(&executable).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&executable, permissions).unwrap();
    VerifiedArtifact::open(&executable_artifact.root).expect("ignore undeclared executable");
}

fn process_group_member(
    process_id: &str,
    process: &str,
    pdgs: Value,
    permutation: Value,
    factor: (i64, i64),
    conjugated: bool,
) -> Value {
    json!({
        "process_id": process_id,
        "process": process,
        "pdgs": pdgs,
        "initial_count": 2,
        "factor": {"numerator": factor.0, "denominator": factor.1},
        "permutation": permutation,
        "conjugated": conjugated,
        "basis": "flavor-blind",
        "generated": false,
    })
}

fn process_groups_document(members: Vec<Value>) -> Value {
    let mut all_members = vec![json!({
        "process_id": "p0",
        "process": "d d~ > a g",
        "pdgs": [1, -1, 22, 21],
        "initial_count": 2,
        "factor": {"numerator": 1, "denominator": 1},
        "permutation": [0, 1, 2, 3],
        "conjugated": false,
        "basis": "identity",
        "generated": true,
    })];
    all_members.extend(members);
    json!({
        "kind": "pyamplicol-process-groups",
        "schema_version": 1,
        "analyses": [{"kind": "future-analysis", "anything": [1, 2]}],
        "groups": [{
            "representative": {"process_id": "p0", "process": "d d~ > a g"},
            "members": all_members,
        }],
    })
}

fn process_group_artifact(document: &Value) -> TestArtifact {
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["expression"] = json!("d d~ > a g");
    artifact.manifest["processes"][0]["external_pdgs"] = json!([1, -1, 22, 21]);
    let bytes = serde_json::to_vec_pretty(document).expect("serialize process groups");
    add_test_payload(
        &mut artifact,
        "process-groups.json",
        "process-groups",
        &bytes,
        None,
        false,
    );
    artifact.write_manifest();
    artifact
}

fn default_process_groups() -> Value {
    process_groups_document(vec![
        process_group_member(
            "u_ubar_flipped",
            "u~ u > a g",
            json!([-2, 2, 22, 21]),
            json!([1, 0, 2, 3]),
            (8, 2),
            false,
        ),
        json!({
            "process_id": "s_sbar_conjugate",
            "process": "s s~ > a g",
            "pdgs": [3, -3, 22, 21],
            "initial_count": 2,
            "factor": {"numerator": 1, "denominator": 1},
            "permutation": [1, 0, 2, 3],
            "conjugated": true,
            "basis": "numerical",
            "generated": false,
            "warning": "assumed, not verified",
            "future_key": {"ignored": true},
        }),
    ])
}

#[test]
fn process_group_members_route_to_their_representative() {
    let artifact = process_group_artifact(&default_process_groups());
    let verified = VerifiedArtifact::open(&artifact.root).unwrap();

    let by_id = verified.select_process(Some("u_ubar_flipped")).unwrap();
    assert_eq!(by_id.process.id, "p0");
    assert_eq!(by_id.requested_id, "u_ubar_flipped");
    assert!(by_id.alias.is_none());
    assert!(!by_id.inferred_permutation);
    assert!(by_id.remaps_public_process());
    assert_eq!(by_id.public_expression, "u~ u > a g");
    assert_eq!(by_id.external_pdgs, [-2, 2, 22, 21]);
    assert_eq!(by_id.external_permutation, [1, 0, 2, 3]);
    let relation = by_id
        .process_group
        .as_ref()
        .expect("process-group relation");
    assert_eq!(
        (relation.factor_numerator, relation.factor_denominator),
        (4, 1)
    );
    assert!(!relation.conjugated);
    assert!(!relation.has_unit_factor());
    assert_eq!(relation.external_particles, ["u~", "u", "a", "g"]);

    let by_expression = verified.select_process(Some(" U~   u > A g ")).unwrap();
    assert_eq!(by_expression.requested_id, "u_ubar_flipped");
    assert_eq!(by_expression.external_permutation, [1, 0, 2, 3]);
    assert!(!by_expression.inferred_permutation);

    // Requested public order <- member order <- representative order.
    let permuted = verified.select_process(Some("u~ u > g a")).unwrap();
    assert_eq!(permuted.requested_id, "u_ubar_flipped");
    assert!(permuted.inferred_permutation);
    assert_eq!(permuted.public_expression, "u~ u > g a");
    assert_eq!(permuted.external_pdgs, [-2, 2, 21, 22]);
    assert_eq!(permuted.external_permutation, [1, 0, 3, 2]);
    assert_eq!(
        permuted.process_group.expect("relation").external_particles,
        ["u~", "u", "g", "a"]
    );

    let swapped = verified.select_process(Some("u u~ > a g")).unwrap();
    assert_eq!(swapped.external_permutation, [0, 1, 2, 3]);
    assert_eq!(swapped.external_pdgs, [2, -2, 22, 21]);

    let conjugate = verified.select_process(Some("s~ s > g a")).unwrap();
    assert_eq!(conjugate.requested_id, "s_sbar_conjugate");
    assert_eq!(conjugate.external_permutation, [0, 1, 3, 2]);
    assert_eq!(conjugate.external_pdgs, [-3, 3, 21, 22]);
    let relation = conjugate.process_group.expect("conjugate relation");
    assert!(relation.conjugated);
    assert!(relation.has_unit_factor());

    // Stored processes keep precedence and never carry a relation.
    let stored = verified.select_process(Some("d~ d > g a")).unwrap();
    assert_eq!(stored.requested_id, "p0");
    assert!(stored.process_group.is_none());
    assert_eq!(stored.external_permutation, [1, 0, 3, 2]);

    let unknown = verified.select_process(Some("c c~ > a g")).unwrap_err();
    assert_eq!(unknown.kind(), crate::RusticolErrorKind::Selector);
    let crossing = verified.select_process(Some("u~ a > u g")).unwrap_err();
    assert_eq!(crossing.kind(), crate::RusticolErrorKind::Selector);
}

#[test]
fn process_group_member_permutation_composes_through_member_order() {
    // Three final-state legs make side permutations non-commuting, so only
    // `public[member[r]]` puts the representative photon on the requested one.
    let mut document = process_groups_document(vec![process_group_member(
        "u_ubar_member",
        "u~ u > g a g",
        json!([-2, 2, 21, 22, 21]),
        json!([1, 0, 3, 2, 4]),
        (4, 1),
        false,
    )]);
    document["groups"][0]["representative"]["process"] = json!("d d~ > a g g");
    let representative = &mut document["groups"][0]["members"][0];
    representative["process"] = json!("d d~ > a g g");
    representative["pdgs"] = json!([1, -1, 22, 21, 21]);
    representative["permutation"] = json!([0, 1, 2, 3, 4]);
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["expression"] = json!("d d~ > a g g");
    artifact.manifest["processes"][0]["external_pdgs"] = json!([1, -1, 22, 21, 21]);
    let bytes = serde_json::to_vec_pretty(&document).expect("serialize process groups");
    add_test_payload(
        &mut artifact,
        "process-groups.json",
        "process-groups",
        &bytes,
        None,
        false,
    );
    artifact.write_manifest();
    let verified = VerifiedArtifact::open(&artifact.root).unwrap();

    let selection = verified.select_process(Some("u u~ > g g a")).unwrap();
    assert_eq!(selection.requested_id, "u_ubar_member");
    assert_eq!(selection.external_pdgs, [2, -2, 21, 21, 22]);
    // Representative legs d, d~, a, g, g sit at requested u, u~, a, g, g.
    assert_eq!(selection.external_permutation, [0, 1, 4, 2, 3]);
}

#[test]
fn process_group_map_is_read_only_after_stored_selection_fails() {
    let mut artifact = process_group_artifact(&default_process_groups());
    fs::write(
        artifact.root.join("process-groups.json"),
        b"{\"tampered\": true}",
    )
    .expect("tamper process groups");
    artifact.write_manifest();
    let verified = VerifiedArtifact::open(&artifact.root).unwrap();

    assert_eq!(
        verified.select_process(Some("p0")).unwrap().requested_id,
        "p0"
    );
    assert_eq!(
        verified
            .select_process(Some("d d~ > g a"))
            .unwrap()
            .requested_id,
        "p0"
    );
    assert_eq!(verified.select_process(None).unwrap().requested_id, "p0");
    let error = verified.select_process(Some("u_ubar_flipped")).unwrap_err();
    assert!(matches!(
        error.kind(),
        crate::RusticolErrorKind::Integrity | crate::RusticolErrorKind::Artifact
    ));
}

#[test]
fn process_group_payload_digest_is_verified() {
    let artifact = process_group_artifact(&default_process_groups());
    let mut document = default_process_groups();
    document["groups"][0]["members"][1]["factor"]["numerator"] = json!(9);
    let mut bytes = serde_json::to_vec_pretty(&document).unwrap();
    let declared = fs::metadata(artifact.root.join("process-groups.json"))
        .unwrap()
        .len() as usize;
    bytes.resize(declared, b' ');
    fs::write(artifact.root.join("process-groups.json"), &bytes).unwrap();

    let verified = VerifiedArtifact::open(&artifact.root).unwrap();
    let error = verified.select_process(Some("u_ubar_flipped")).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("sha256"));
}

#[test]
fn invalid_process_group_maps_are_rejected() {
    let member = |permutation: Value, factor: (i64, i64)| {
        process_groups_document(vec![process_group_member(
            "u_ubar",
            "u u~ > a g",
            json!([2, -2, 22, 21]),
            permutation,
            factor,
            false,
        )])
    };
    let mut wrong_representative = member(json!([0, 1, 2, 3]), (1, 1));
    wrong_representative["groups"][0]["representative"]["process_id"] = json!("missing");
    let mut representative_not_first = member(json!([0, 1, 2, 3]), (1, 1));
    representative_not_first["groups"][0]["members"]
        .as_array_mut()
        .unwrap()
        .reverse();
    let mut reused_id = member(json!([0, 1, 2, 3]), (1, 1));
    reused_id["groups"][0]["members"][1]["process_id"] = json!("p0");
    let mut wrong_kind = member(json!([0, 1, 2, 3]), (1, 1));
    wrong_kind["kind"] = json!("something-else");
    let mut wrong_count = member(json!([0, 1, 2, 3]), (1, 1));
    wrong_count["groups"][0]["members"][1]["process"] = json!("u > u~ a g");
    let cases = [
        (member(json!([2, 1, 0, 3]), (1, 1)), "'>' boundary"),
        (member(json!([0, 0, 2, 3]), (1, 1)), "complete zero-based"),
        (member(json!([0, 1, 2]), (1, 1)), "has length 3"),
        (member(json!([0, 1, 2, 3]), (1, 0)), "positive rational"),
        (member(json!([0, 1, 2, 3]), (-1, 2)), "positive rational"),
        (wrong_representative, "not a stored process"),
        (representative_not_first, "identity relation"),
        (reused_id, "listed more than once"),
        (wrong_kind, "unsupported process-groups payload"),
        (wrong_count, "particle counts"),
    ];
    for (document, expected) in cases {
        let artifact = process_group_artifact(&document);
        let verified = VerifiedArtifact::open(&artifact.root).unwrap();
        let error = verified.select_process(Some("u_ubar")).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "{expected:?} not in {error}"
        );
    }

    let mut collision = TestArtifact::new();
    collision.manifest["processes"][0]["expression"] = json!("d d~ > a g");
    collision.manifest["processes"][0]["external_pdgs"] = json!([1, -1, 22, 21]);
    collision.manifest["processes"][0]["aliases"] = json!([{
        "id": "u_ubar",
        "expression": "d~ d > a g",
        "external_pdgs": [-1, 1, 22, 21],
        "external_permutation": [1, 0, 2, 3],
    }]);
    let bytes = serde_json::to_vec(&member(json!([0, 1, 2, 3]), (1, 1))).unwrap();
    add_test_payload(
        &mut collision,
        "process-groups.json",
        "process-groups",
        &bytes,
        None,
        false,
    );
    collision.write_manifest();
    let verified = VerifiedArtifact::open(&collision.root).unwrap();
    let error = verified.select_process(Some("u u~ > a g")).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("reuses a stored process or alias id")
    );
}

#[test]
fn ambiguous_process_group_member_expressions_list_member_ids() {
    let document = process_groups_document(vec![
        process_group_member(
            "first",
            "u u~ > a g",
            json!([2, -2, 22, 21]),
            json!([0, 1, 2, 3]),
            (4, 1),
            false,
        ),
        process_group_member(
            "second",
            "u u~ > a g",
            json!([2, -2, 22, 21]),
            json!([0, 1, 2, 3]),
            (4, 1),
            false,
        ),
    ]);
    let artifact = process_group_artifact(&document);
    let verified = VerifiedArtifact::open(&artifact.root).unwrap();
    let error = verified.select_process(Some("u u~ > a g")).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Selector);
    assert!(error.to_string().contains("first, second"));
    assert_eq!(
        verified
            .select_process(Some("second"))
            .unwrap()
            .requested_id,
        "second"
    );
}
