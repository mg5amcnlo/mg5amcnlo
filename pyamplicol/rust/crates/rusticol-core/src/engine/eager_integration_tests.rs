// SPDX-License-Identifier: 0BSD

use super::*;
use crate::{
    EAGER_KERNEL_ABI, EagerClosureRow, EagerCouplingRow, EagerExecutionPlan, EagerExecutionRuntime,
    EagerKernelInput, EagerKernelRole, EagerKernelSpec, EagerPlanDefinition, EagerPlanDimensions,
    EagerPlanPayloads, EagerReductionEntry, EagerReductionGroup, EagerRuntimeOptions, MISSING_U32,
};
use serde_json::json;

const TEST_SYMJIT_APPLICATION_ABI: &str = "symjit-application-storage-v3";
const TEST_SYMJIT_PLANE_APPLICATION_ABI: &str = "pyamplicol-symjit-plane-application-v2";
const TEST_PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL: u64 = 2;
const TEST_PREPARED_JIT_PORTABLE_TARGET: &str = "symjit-storage-v3-portable";

#[test]
fn eager_color_topology_replay_preserves_amplitude_major_batches() {
    let materialized_groups = [
        RawSumGroup {
            id: 10,
            indices: vec![0, 1],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        },
        RawSumGroup {
            id: 20,
            indices: vec![2],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![1],
        },
    ];
    let replay = serde_json::from_value(json!({
        "contract_version": 1,
        "physical_group_count": 2,
        "physical_groups": [
            {"group_id": 0, "helicities": [1, -1], "color_sector_id": 0,
             "color_word": [1, 2], "helicity_weight": 1.0},
            {"group_id": 1, "helicities": [1, -1], "color_sector_id": 1,
             "color_word": [2, 1], "helicity_weight": 1.0},
        ],
        "mappings": [{"label_permutation": [], "group_routes": [
            {"source_group_id": 10, "target_group_id": 0, "factor": [1.0, 0.0]},
            {"source_group_id": 20, "target_group_id": 1, "factor": [0.0, 1.0]},
        ]}],
    }))
    .expect("color topology replay fixture");
    let contraction = serde_json::from_value(json!({
        "supported": true,
        "group_count": 2,
        "includes_color_factor": true,
        "entries": [
            {"left_group_id": 0, "right_group_id": 0, "weight": [2.0, 0.0]},
            {"left_group_id": 1, "right_group_id": 1, "weight": [3.0, 0.0]},
            {"left_group_id": 0, "right_group_id": 1, "weight": [1.0, 0.0],
             "symmetry_factor": 2.0},
        ],
    }))
    .expect("color contraction fixture");
    let mut reducer = AmplitudeRuntime::color_topology_replay_reducer(
        3,
        &materialized_groups,
        &replay,
        &contraction,
    )
    .expect("color topology replay reducer");

    // Include repeated and distinct points, and a batch crossing the normal
    // eager point-tile boundary. Each amplitude occupies one point plane.
    for point_count in [1, 2, 3, 65] {
        let value = |amplitude: usize, point: usize| {
            let point = (point / 2) as f64;
            crate::EagerComplex64::new(1.0 + amplitude as f64 * 3.0 + point, point - 2.0)
        };
        let amplitudes = (0..3)
            .flat_map(|amplitude| (0..point_count).map(move |point| value(amplitude, point)))
            .collect::<Vec<_>>();
        reducer.begin_color_topology_replay(point_count).unwrap();
        reducer
            .gather_color_topology_replay_eager_amplitudes(&amplitudes, point_count, 0)
            .unwrap();
        let mut totals = vec![0.0; point_count];
        reducer
            .reduce_color_topology_replay_f64_into(point_count, &mut totals)
            .unwrap();
        for (point, total) in totals.iter().copied().enumerate() {
            let left = value(0, point) + value(1, point);
            let right = value(2, point) * crate::EagerComplex64::new(0.0, 1.0);
            let expected =
                (2.0 * left * left.conj() + 3.0 * right * right.conj() + 2.0 * left * right.conj())
                    .re;
            assert_eq!(total, expected, "batch {point_count}, point {point}");
        }
    }
}

fn symjit_manifest(application_path: &str, exact_state_path: &str, input_len: usize) -> Value {
    let plane_application_path = application_path.strip_suffix(".symjit").map_or_else(
        || format!("{application_path}.plane.symjit"),
        |stem| format!("{stem}.plane.symjit"),
    );
    json!({
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
        "backend": "jit",
        "label": "prepared_test_kernel",
        "application_path": application_path,
        "application_abi": TEST_SYMJIT_APPLICATION_ABI,
        "plane_application": {
            "application_path": plane_application_path,
            "application_abi": TEST_SYMJIT_PLANE_APPLICATION_ABI,
            "storage_abi": TEST_SYMJIT_APPLICATION_ABI,
            "element_layout": "split-complex-plane-major",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "input_complex_count": input_len,
            "output_complex_count": 1,
            "input_plane_count": 2 * input_len,
            "output_plane_count": 2,
            "compiler_type": "native",
            "translation_mode": "symbolica-structured-instructions",
            "optimization_level": TEST_PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
            "simd": true,
            "complex": true,
            "fast_math": true,
            "fast_complex": false,
            "compression": false,
            "threading": false,
            "direct_arena": true,
            "source_digest": "0".repeat(64),
            "target": {
                "word_bits": 64,
                "endianness": "little",
            },
        },
        "input_len": input_len,
        "output_len": 1,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": TEST_PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": exact_state_path,
        "evaluator_state_runtime_capability": SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        "settings": {
            "jit_optimization_level": TEST_PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
        },
        "build_timing": {"jit_materialize_s": 0.0},
    })
}

fn compiled_manifest(runtime_capability: &str, exact_state_path: &str, input_len: usize) -> Value {
    json!({
        "kind": "compiled-complex-evaluator",
        "runtime_capability": runtime_capability,
        "backend": "compiled-complex",
        "function_name": "prepared_test_kernel",
        "library_path": "kernels/7/library-0",
        "source_path": "kernels/7/source-0.cpp",
        "input_len": input_len,
        "output_len": 1,
        "number_type": "complex-f64",
        "evaluator_state_path": exact_state_path,
        "settings": {"optimization_level": "o3"},
        "build_timing": {"compile_s": 0.0},
        "direct_table": {
            "capability": crate::eager_layout::EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
            "source_application_abi":
                crate::eager_layout::EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI,
            "descriptor_abi": crate::eager_layout::EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            "binding_abi": crate::eager_layout::EAGER_DIRECT_TABLE_BINDING_ABI,
            "library_path": "kernels/7/library-0",
            "function_name": "pyamplicol_eager_direct_table_k00000007_v1",
            "evaluator_state_sha256": "b".repeat(64),
            "input_complex_count": input_len,
            "output_complex_count": 1,
            "invocation_stride": (2 * input_len + 2) * 4,
            "attachment_stride": 16,
            "simd_lane_width": 2,
            "instruction_count": 1,
            "temporary_count": 0,
        },
    })
}

fn input(role: &str, component: u32) -> PreparedKernelInputManifest {
    PreparedKernelInputManifest {
        role: role.to_string(),
        component,
        symbol: format!("pyamplicol::test::{role}::{component}"),
        model_parameter_name: None,
        model_parameter_index: None,
    }
}

fn model_parameter_input(name: &str, index: u32) -> PreparedKernelInputManifest {
    PreparedKernelInputManifest {
        role: "model-parameter".to_string(),
        component: 0,
        symbol: format!("pyamplicol::test::model-parameter::{name}"),
        model_parameter_name: Some(name.to_string()),
        model_parameter_index: Some(index),
    }
}

fn kernel(
    kernel_id: u32,
    contract_kind: &str,
    inputs: Vec<PreparedKernelInputManifest>,
    application_path: &str,
) -> PreparedKernelManifest {
    let input_arity = inputs.len();
    let exact_evaluator_state_path = format!("kernels/{kernel_id}/exact.bin");
    PreparedKernelManifest {
        kernel_id,
        contract_kind: contract_kind.to_string(),
        canonical_signature: format!("test-kernel-{kernel_id}"),
        input_arity,
        output_arity: 1,
        input_layout: (0..input_arity)
            .map(|index| format!("input-{index}"))
            .collect(),
        input_contracts: inputs,
        output_layout: vec!["output-0".to_string()],
        exact_expressions: vec!["test-expression".to_string()],
        proof_classes: Vec::new(),
        f64_evaluator_manifest: symjit_manifest(
            application_path,
            &exact_evaluator_state_path,
            input_arity,
        ),
        exact_evaluator_state_path,
    }
}

fn add_direct_table_manifest(kernel: &mut PreparedKernelManifest) {
    kernel
        .f64_evaluator_manifest
        .as_object_mut()
        .expect("test evaluator object")
        .insert(
            "direct_table".to_string(),
            json!({
                "capability": crate::eager_layout::EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
                "source_application_abi":
                    crate::eager_layout::EAGER_DIRECT_SOURCE_APPLICATION_ABI,
                "descriptor_abi": crate::eager_layout::EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
                "binding_abi": crate::eager_layout::EAGER_DIRECT_TABLE_BINDING_ABI,
                "descriptor_path": format!(
                    "kernels/{}/eager-direct-table-descriptor-v1.bin",
                    kernel.kernel_id
                ),
                "descriptor_size_bytes": 128,
                "descriptor_sha256": "a".repeat(64),
                "input_complex_count": kernel.input_arity,
                "output_complex_count": kernel.output_arity,
            }),
        );
}

#[test]
fn eager_direct_table_manifest_is_explicit_and_fail_closed() {
    let mut prepared = kernel(
        50,
        "closure",
        vec![input("left-current", 0)],
        "kernels/50/application.symjit",
    );
    let missing = prepared
        .eager_direct_table_manifest()
        .expect_err("pre-arena kernel must fail");
    assert!(missing.to_string().contains("regenerate"));

    add_direct_table_manifest(&mut prepared);
    let direct = prepared
        .eager_direct_table_manifest()
        .expect("valid direct table metadata");
    assert_eq!(direct.input_complex_count, 1);
    assert_eq!(direct.output_complex_count, 1);
    assert_eq!(direct.descriptor_size_bytes, Some(128));

    let large_descriptor_size = (1_u64 << 30) + 1;
    prepared.f64_evaluator_manifest["direct_table"]["descriptor_size_bytes"] =
        json!(large_descriptor_size);
    let direct = prepared
        .eager_direct_table_manifest()
        .expect("large writer-representable DirectTable descriptor metadata must remain valid");
    assert_eq!(direct.descriptor_size_bytes, Some(large_descriptor_size));

    let mut predecessor = prepared.clone();
    predecessor.f64_evaluator_manifest["direct_table"]["descriptor_abi"] =
        json!(["symjit-direct-table-", "descriptor-v1"].concat());
    predecessor.f64_evaluator_manifest["direct_table"]["binding_abi"] =
        json!(["symjit-direct-table-", "binding-v1"].concat());
    let error = predecessor
        .eager_direct_table_manifest()
        .expect_err("predecessor DirectTable ABI must fail closed");
    assert!(error.to_string().contains("regenerate"));

    for (field, value) in [
        ("capability", json!("wrong-capability")),
        ("descriptor_abi", json!("wrong-descriptor-abi")),
        ("binding_abi", json!("wrong-binding-abi")),
        ("descriptor_size_bytes", json!(0)),
        ("descriptor_sha256", json!("A".repeat(64))),
        ("input_complex_count", json!(2)),
    ] {
        let mut malformed = prepared.clone();
        malformed.f64_evaluator_manifest["direct_table"][field] = value;
        assert!(
            malformed.eager_direct_table_manifest().is_err(),
            "malformed DirectTable field {field} must fail"
        );
    }
}

fn filtered_pack(kernels: Vec<PreparedKernelManifest>) -> PreparedKernelPackManifest {
    PreparedKernelPackManifest {
        eager_kernel_abi: EAGER_KERNEL_ABI.to_string(),
        backend: "jit".to_string(),
        optimization_settings: json!({
            "jit_optimization_level": TEST_PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
        }),
        producer: json!({"distribution": "pyamplicol", "version": "test"}),
        dependency_abis: json!({
            "symjit_application": TEST_SYMJIT_APPLICATION_ABI,
            "symjit_plane_application": TEST_SYMJIT_PLANE_APPLICATION_ABI,
        }),
        provenance: json!({"compiled_model_digest": "test"}),
        target: PreparedKernelTargetManifest {
            portable: true,
            word_bits: 64,
            endianness: "little".to_string(),
            target_triple: TEST_PREPARED_JIT_PORTABLE_TARGET.to_string(),
            cpu_features: Vec::new(),
        },
        resolver_manifest: json!({
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "test-model",
            "model_parameter_kernel_id": null,
            "propagator_bindings": [
                {"key": {"particle_id": 22, "chirality": 0}, "applies_propagator": false, "kernel_id": null}
            ],
            "vertex_bindings": [],
            "closure_bindings": []
        }),
        kernels,
        kernel_variants: Vec::new(),
        recurrence_template: None,
        recurrence_direct_template: None,
    }
}

#[test]
fn eager_parameter_projection_uses_active_and_derivation_dependencies_only() {
    let pack = filtered_pack(vec![
        kernel(
            7,
            "vertex",
            vec![model_parameter_input("alpha", 3)],
            "kernels/7/application.symjit",
        ),
        kernel(
            8,
            "propagator",
            vec![model_parameter_input("MW", 40)],
            "kernels/8/application.symjit",
        ),
        kernel(
            9,
            "model-parameter",
            vec![model_parameter_input("Gf", 100)],
            "kernels/9/application.symjit",
        ),
    ]);
    let indices = super::eager_load::prepared_parameter_indices(&pack, &BTreeSet::from([7]), true)
        .expect("project process-local prepared parameters");

    assert_eq!(
        indices,
        BTreeMap::from([("Gf".to_string(), 100), ("alpha".to_string(), 3)])
    );
    assert!(!indices.contains_key("MW"));

    let indices_without_derived =
        super::eager_load::prepared_parameter_indices(&pack, &BTreeSet::from([7]), false)
            .expect("omit unused derivation inputs");
    assert_eq!(
        indices_without_derived,
        BTreeMap::from([("alpha".to_string(), 3)])
    );

    let runtime_parameters = [GenericRuntimeModelParameterManifest {
        name: "alpha".to_string(),
        kind: "external_parameter".to_string(),
        parameter_index: 0,
        default: 1.0,
        pdg: None,
        runtime_name: None,
        complex_component: None,
    }];
    let payloads = EvaluatorPayloadStore::directory(Path::new("."));
    let (projection, _, evaluator) = super::eager_load::prepare_eager_parameter_state(
        &pack,
        &BTreeSet::from([7]),
        &runtime_parameters,
        &[],
        &payloads,
    )
    .expect("an unused derivation kernel must not load or impose its inputs");
    assert_eq!(projection.parameter_count, 4);
    assert!(evaluator.is_none());
}

#[test]
fn eager_parameter_projection_preserves_full_pack_parameter_integrity() {
    let pack = filtered_pack(vec![
        kernel(
            7,
            "vertex",
            vec![model_parameter_input("alpha", 3)],
            "kernels/7/application.symjit",
        ),
        kernel(
            8,
            "propagator",
            vec![model_parameter_input("MW", 40)],
            "kernels/8/application.symjit",
        ),
        kernel(
            9,
            "closure",
            vec![model_parameter_input("MW", 41)],
            "kernels/9/application.symjit",
        ),
    ]);

    let error = super::eager_load::prepared_parameter_indices(&pack, &BTreeSet::from([7]), false)
        .expect_err("inactive contracts must still share one authenticated namespace");
    assert!(error.to_string().contains("conflicting stable indices"));
}

#[test]
fn eager_parameter_projection_ignores_inactive_missing_runtime_dependencies_but_not_active_ones() {
    let pack = filtered_pack(vec![
        kernel(
            7,
            "vertex",
            vec![model_parameter_input("alpha", 3)],
            "kernels/7/application.symjit",
        ),
        kernel(
            8,
            "propagator",
            vec![model_parameter_input("MW", 40)],
            "kernels/8/application.symjit",
        ),
    ]);
    let runtime_parameters = vec![GenericRuntimeModelParameterManifest {
        name: "alpha".to_string(),
        kind: "external_parameter".to_string(),
        parameter_index: 0,
        default: 1.0,
        pdg: None,
        runtime_name: None,
        complex_component: None,
    }];
    let payloads = EvaluatorPayloadStore::directory(Path::new("."));

    let (projection, _, evaluator) = super::eager_load::prepare_eager_parameter_state(
        &pack,
        &BTreeSet::from([7]),
        &runtime_parameters,
        &[],
        &payloads,
    )
    .expect("inactive prepared dependencies do not belong to this process");
    assert_eq!(projection.parameter_count, 4);
    assert_eq!(projection.entries.len(), 1);
    assert_eq!(projection.entries[0].prepared_index, 3);
    assert_eq!(projection.entries[0].runtime_real_index, 0);
    assert!(evaluator.is_none());

    let error = match super::eager_load::prepare_eager_parameter_state(
        &pack,
        &BTreeSet::from([8]),
        &runtime_parameters,
        &[],
        &payloads,
    ) {
        Ok(_) => panic!("an active prepared dependency must exist in the runtime schema"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("prepared parameter \"MW\" is absent")
    );
}

fn portable_outer_target() -> crate::Target {
    crate::Target {
        triple: crate::artifact::PORTABLE_64LE_ARTIFACT_TARGET.to_string(),
        cpu_features: Vec::new(),
    }
}

fn portable_jit_pack_for_outer_target_test() -> PreparedKernelPackManifest {
    filtered_pack(vec![kernel(
        50,
        "closure",
        vec![input("left-current", 0)],
        "kernels/50/application.symjit",
    )])
}

fn assert_portable_outer_pack_boundary(
    validate: impl Fn(&crate::Target, &PreparedKernelPackManifest) -> RusticolResult<()>,
    context: &str,
) {
    let target = portable_outer_target();
    let jit = portable_jit_pack_for_outer_target_test();
    jit.validate().expect("valid portable O2 JIT pack");
    validate(&target, &jit).unwrap_or_else(|error| {
        panic!("portable O2 JIT pack must satisfy {context} outer target: {error}")
    });

    for (backend, capability) in [
        ("cpp", SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY),
        ("asm", SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY),
    ] {
        let native = compiled_pack(backend, capability);
        native
            .validate()
            .unwrap_or_else(|error| panic!("valid native {backend} pack: {error}"));
        let error = validate(&target, &native)
            .expect_err("portable outer artifact must reject a native prepared pack");
        assert!(error.to_string().contains("portable O2 SymJIT"));
        assert!(error.to_string().contains(context));
    }
}

#[test]
fn eager_portable_outer_target_requires_o2_jit_pack() {
    assert_portable_outer_pack_boundary(
        super::eager_v3_load::validate_eager_prepared_pack_outer_target,
        "eager",
    );
}

#[test]
fn recurrence_portable_outer_target_requires_o2_jit_pack() {
    assert_portable_outer_pack_boundary(
        super::recurrence_load::validate_recurrence_prepared_pack_outer_target,
        "recurrence",
    );
}

#[test]
fn prepared_jit_pack_rejects_nonportable_target_before_loading_payloads() {
    let mut pack = filtered_pack(vec![kernel(
        50,
        "closure",
        vec![input("left-current", 0)],
        "kernels/50/missing.symjit",
    )]);
    let other = if std::env::consts::ARCH == "aarch64" {
        "x86_64"
    } else {
        "aarch64"
    };
    pack.target.portable = false;
    pack.target.target_triple = format!("symjit-storage-v3-{other}");

    let error = pack
        .validate()
        .expect_err("nonportable prepared JIT pack must fail");
    assert!(error.to_string().contains("expected portable target"));
}

#[test]
fn prepared_jit_pack_requires_plane_application_dependency_abi() {
    let mut pack = filtered_pack(vec![kernel(
        50,
        "closure",
        vec![input("left-current", 0)],
        "kernels/50/missing.symjit",
    )]);
    pack.dependency_abis
        .as_object_mut()
        .expect("test dependency ABI object")
        .remove("symjit_plane_application");

    let error = pack
        .validate()
        .expect_err("prepared JIT pack without plane application ABI must fail");
    assert!(error.to_string().contains("plane-application ABI"));
}

#[test]
fn raw_prepared_jit_manifest_projects_plane_without_direct_table() {
    let mut pack = filtered_pack(vec![kernel(
        50,
        "closure",
        vec![input("left-current", 0)],
        "kernels/50/application.symjit",
    )]);
    pack.validate()
        .expect("raw prepared JIT pack with a plane application");
    assert!(
        pack.kernels[0]
            .extra_evaluator_payload_paths()
            .expect("raw JIT extra payload paths")
            .is_empty(),
        "a raw prepared JIT pack has no process-stage DirectTable payload"
    );

    let runtime = pack.kernels[0]
        .runtime_evaluator_manifest()
        .expect("project raw prepared evaluator");
    let EvaluatorManifest::SymjitApplication {
        plane_application: Some(plane),
        ..
    } = runtime
    else {
        panic!("raw prepared evaluator must retain its plane application");
    };
    assert_eq!(
        plane.application_path,
        "kernels/50/application.plane.symjit"
    );

    add_direct_table_manifest(&mut pack.kernels[0]);
    let runtime = pack.kernels[0]
        .runtime_evaluator_manifest()
        .expect("project process-stage prepared evaluator");
    assert!(matches!(
        runtime,
        EvaluatorManifest::SymjitApplication {
            plane_application: Some(_),
            ..
        }
    ));
    assert_eq!(
        pack.kernels[0]
            .extra_evaluator_payload_paths()
            .expect("process-stage extra payload paths"),
        ["kernels/50/eager-direct-table-descriptor-v1.bin"]
    );
}

#[test]
fn prepared_jit_plane_application_is_fail_closed() {
    let pack = filtered_pack(vec![kernel(
        50,
        "closure",
        vec![input("left-current", 0)],
        "kernels/50/application.symjit",
    )]);

    let mut missing = pack.clone();
    missing.kernels[0]
        .f64_evaluator_manifest
        .as_object_mut()
        .expect("test evaluator object")
        .remove("plane_application");
    let error = missing
        .validate()
        .expect_err("missing plane application must fail");
    assert!(error.to_string().contains("regenerate"));

    for (field, value) in [
        ("application_abi", json!("wrong-plane-abi")),
        ("input_plane_count", json!(3)),
        ("target", Value::Null),
    ] {
        let mut malformed = pack.clone();
        malformed.kernels[0].f64_evaluator_manifest["plane_application"][field] = value;
        let error = malformed
            .validate()
            .expect_err("malformed plane application must fail");
        assert!(
            error.to_string().contains("regenerate"),
            "malformed plane field {field} must include regeneration guidance: {error}"
        );
    }
}

fn compiled_pack(backend: &str, runtime_capability: &str) -> PreparedKernelPackManifest {
    let exact_evaluator_state_path = "kernels/7/exact.bin".to_string();
    let kernel = PreparedKernelManifest {
        kernel_id: 7,
        contract_kind: "vertex".to_string(),
        canonical_signature: format!("test-{backend}-kernel"),
        input_arity: 1,
        output_arity: 1,
        input_layout: vec!["input-0".to_string()],
        input_contracts: vec![input("left-current", 0)],
        output_layout: vec!["output-0".to_string()],
        exact_expressions: vec!["test-expression".to_string()],
        proof_classes: Vec::new(),
        f64_evaluator_manifest: compiled_manifest(
            runtime_capability,
            &exact_evaluator_state_path,
            1,
        ),
        exact_evaluator_state_path,
    };
    let target = crate::runtime_target_info();
    PreparedKernelPackManifest {
        eager_kernel_abi: EAGER_KERNEL_ABI.to_string(),
        backend: backend.to_string(),
        optimization_settings: json!({"optimization_level": "o3"}),
        producer: json!({"distribution": "pyamplicol", "version": "test"}),
        dependency_abis: json!({"compiled_complex": "test"}),
        provenance: json!({"compiled_model_digest": "test"}),
        target: PreparedKernelTargetManifest {
            portable: false,
            word_bits: 64,
            endianness: "little".to_string(),
            target_triple: target.triple,
            cpu_features: target.cpu_features,
        },
        resolver_manifest: json!({
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "test-model",
            "model_parameter_kernel_id": null,
            "propagator_bindings": [],
            "vertex_bindings": [],
            "closure_bindings": []
        }),
        kernels: vec![kernel],
        kernel_variants: Vec::new(),
        recurrence_template: None,
        recurrence_direct_template: None,
    }
}

#[test]
fn runtime_options_are_positive_and_use_checked_platform_sizes() {
    let options = EagerRuntimeOptionsManifest {
        point_tile_size: 1024,
        workspace_mib: 256,
    }
    .validate()
    .expect("valid eager runtime options");
    assert_eq!(options.point_tile_size, 1024);
    assert_eq!(options.workspace_bytes, 256 * 1024 * 1024);

    let above_old_reader_ceilings = EagerRuntimeOptionsManifest {
        point_tile_size: 1_048_577,
        workspace_mib: 4097,
    }
    .validate()
    .expect("writer-valid eager runtime options above old reader ceilings");
    assert_eq!(above_old_reader_ceilings.point_tile_size, 1_048_577);
    assert_eq!(
        above_old_reader_ceilings.workspace_bytes,
        4097 * 1024 * 1024
    );

    for invalid in [
        EagerRuntimeOptionsManifest {
            point_tile_size: 0,
            workspace_mib: 256,
        },
        EagerRuntimeOptionsManifest {
            point_tile_size: 1024,
            workspace_mib: 0,
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn filtered_pack_accepts_only_referenced_kernel_ids_and_preserves_input_order() {
    let pack = filtered_pack(vec![
        kernel(
            7,
            "propagator",
            vec![input("current", 0), input("momentum", 3)],
            "kernels/7/application.symjit",
        ),
        kernel(
            78,
            "vertex",
            vec![
                input("right-current", 1),
                input("left-current", 0),
                input("right-momentum", 2),
                input("coupling-imag", 0),
            ],
            "kernels/78/application.symjit",
        ),
    ]);
    pack.validate().expect("filtered prepared pack");

    let specs = pack.kernel_specs().expect("prepared kernel specs");
    assert_eq!(
        specs.iter().map(|spec| spec.kernel_id).collect::<Vec<_>>(),
        [7, 78]
    );
    assert_eq!(specs[0].role, EagerKernelRole::Finalization);
    assert_eq!(
        specs[1].inputs,
        vec![
            EagerKernelInput::SecondCurrentComponent(1),
            EagerKernelInput::FirstCurrentComponent(0),
            EagerKernelInput::SecondMomentumComponent(2),
            EagerKernelInput::CouplingImag,
        ]
    );
}

#[test]
fn prepared_pack_schema_requires_an_explicit_kernel_abi() {
    let missing = json!({
        "backend": "jit",
        "optimization_settings": {"jit_optimization_level": 3},
        "producer": {"distribution": "pyamplicol"},
        "dependency_abis": {"symjit_application": "test"},
        "provenance": {"compiled_model_digest": "test"},
        "target": {
            "portable": true,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "portable-symjit-mir",
            "cpu_features": []
        },
        "resolver_manifest": {"abi": "pyamplicol-prepared-kernel-catalog-v1"},
        "kernels": []
    });
    assert!(serde_json::from_value::<PreparedKernelPackManifest>(missing.clone()).is_err());

    let mut unknown = missing;
    unknown["eager_kernel_abi"] = json!(EAGER_KERNEL_ABI);
    unknown["unexpected"] = json!(true);
    assert!(serde_json::from_value::<PreparedKernelPackManifest>(unknown).is_err());
}

#[test]
fn compiled_prepared_backends_validate_runtime_identity_tuples() {
    compiled_pack("asm", SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY)
        .validate()
        .expect("ASM prepared evaluator tuple");
    compiled_pack("cpp", SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY)
        .validate()
        .expect("C++ prepared evaluator tuple");

    let mut wrong_backend = compiled_pack("asm", SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY);
    wrong_backend.kernels[0]
        .f64_evaluator_manifest
        .as_object_mut()
        .expect("compiled evaluator object")
        .insert("backend".to_string(), json!("asm"));
    assert!(wrong_backend.validate().is_err());

    let wrong_capability = compiled_pack("asm", SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY);
    assert!(wrong_capability.validate().is_err());
}

#[cfg(feature = "f64-symjit")]
#[test]
fn prepared_symjit_backend_executes_a_filtered_eager_plan() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use symjit::{Compiler, CompilerType, Config, Storage};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "rusticol-eager-backend-{}-{}",
        std::process::id(),
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(root.join("kernels/50")).expect("create prepared-kernel test root");

    let mut config = Config::new(CompilerType::Native, 0).expect("native SymJIT config");
    config.set_complex(true);
    config.set_symbolica(true);
    config.set_opt_level(TEST_PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL as u8);
    config.set_simd(true);
    let mut compiler = Compiler::with_config(config);
    let instructions = r#"[[{"Add":[{"Out":0},[{"Param":0},{"Param":1}],0]}],1,[]]"#;
    let application = compiler
        .translate(instructions.to_string(), 2)
        .expect("translate prepared test application");
    let mut application_bytes = Vec::new();
    application
        .save(&mut application_bytes)
        .expect("serialize prepared test application");
    fs::write(
        root.join("kernels/50/application.symjit"),
        application_bytes,
    )
    .expect("write prepared test application");
    let plane_application_bytes =
        crate::engine::compile_symbolica_program_to_plane_application_bytes(
            instructions,
            2,
            1,
            TEST_PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL as u8,
            false,
        )
        .expect("compile prepared test plane application");
    fs::write(
        root.join("kernels/50/application.plane.symjit"),
        plane_application_bytes,
    )
    .expect("write prepared test plane application");

    let pack = filtered_pack(vec![kernel(
        50,
        "closure",
        vec![input("left-current", 0), input("right-current", 0)],
        "kernels/50/application.symjit",
    )]);
    pack.validate().expect("valid filtered test pack");
    assert_eq!(
        PreparedEvaluatorBackend::preflight_all(&pack, &root)
            .expect("preflight raw prepared JIT pack without a DirectTable"),
        1
    );
    let mut backend = PreparedEvaluatorBackend::load(&pack, &root)
        .expect("load prepared SymJIT evaluator backend");
    let definition = EagerPlanDefinition {
        dimensions: EagerPlanDimensions {
            value_slot_component_counts: vec![1, 1],
            momentum_slot_component_counts: Vec::new(),
            current_component_counts: Vec::new(),
            parameter_count: 0,
            amplitude_count: 1,
        },
        kernels: vec![EagerKernelSpec {
            kernel_id: 50,
            role: EagerKernelRole::Closure,
            inputs: vec![
                EagerKernelInput::FirstCurrentComponent(0),
                EagerKernelInput::SecondCurrentComponent(0),
            ],
            output_component_count: 1,
            homogeneous_linear_first_current: false,
            independent_block_size: 1,
        }],
        direct_closures: Vec::new(),
        reduction_groups: vec![EagerReductionGroup {
            coherent_group_id: 0,
            amplitude_indices: vec![0],
        }],
        reduction_entries: vec![EagerReductionEntry {
            left_group_index: 0,
            right_group_index: 0,
            coefficient: crate::EagerComplex64::new(1.0, 0.0),
        }],
    };
    let coupling_bytes = EagerCouplingRow::encode_table(&[EagerCouplingRow {
        real_parameter_id: MISSING_U32,
        imag_parameter_id: MISSING_U32,
        constant_real: 1.0,
        constant_imag: 0.0,
    }])
    .expect("encode eager coupling table");
    let closure_bytes = EagerClosureRow::encode_table(&[EagerClosureRow {
        kernel_id: 50,
        left_value_slot_id: 0,
        right_value_slot_id: 1,
        amplitude_index: 0,
        coupling_slot_id: 0,
        output_factor_source: crate::EAGER_OUTPUT_FACTOR_NONE,
        factor_real: 1.0,
        factor_imag: 0.0,
    }])
    .expect("encode eager closure table");
    let plan = EagerExecutionPlan::from_payloads(
        definition,
        EagerPlanPayloads {
            couplings: &coupling_bytes,
            stages: &[],
            closures: &closure_bytes,
            selector_domains: None,
        },
    )
    .expect("build eager test plan");
    let mut runtime = EagerExecutionRuntime::new(
        plan,
        EagerRuntimeOptions {
            point_tile_size: 2,
            workspace_bytes: 4096,
        },
    )
    .expect("build eager test runtime");
    let values = [
        crate::EagerComplex64::new(1.0, 0.0),
        crate::EagerComplex64::new(2.0, 0.0),
        crate::EagerComplex64::new(3.0, 0.0),
        crate::EagerComplex64::new(10.0, 0.0),
        crate::EagerComplex64::new(20.0, 0.0),
        crate::EagerComplex64::new(30.0, 0.0),
    ];
    let mut amplitudes = [crate::EagerComplex64::new(0.0, 0.0); 3];
    let mut reduced = [0.0; 3];
    runtime
        .evaluate_into(
            &mut backend,
            3,
            &values,
            &[],
            &[],
            &mut amplitudes,
            &mut reduced,
        )
        .expect("evaluate prepared eager plan");
    assert_eq!(
        amplitudes,
        [
            crate::EagerComplex64::new(11.0, 0.0),
            crate::EagerComplex64::new(22.0, 0.0),
            crate::EagerComplex64::new(33.0, 0.0),
        ]
    );
    assert_eq!(reduced, [121.0, 484.0, 1089.0]);

    amplitudes.fill(crate::EagerComplex64::new(0.0, 0.0));
    reduced.fill(0.0);
    let profile = runtime
        .evaluate_profile_into(
            &mut backend,
            3,
            &values,
            &[],
            &[],
            &mut amplitudes,
            &mut reduced,
        )
        .expect("profile prepared eager plan");
    assert_eq!(
        amplitudes,
        [
            crate::EagerComplex64::new(11.0, 0.0),
            crate::EagerComplex64::new(22.0, 0.0),
            crate::EagerComplex64::new(33.0, 0.0),
        ]
    );
    assert_eq!(reduced, [121.0, 484.0, 1089.0]);
    assert!(!profile.total.is_zero());
    assert!(!profile.initialize.is_zero());
    assert!(!profile.kernel_call.is_zero());
    assert!(!profile.closure.is_zero());
    assert!(!profile.reduction.is_zero());
    assert!(!profile.copy_out.is_zero());
    assert!(profile.accounted() <= profile.total);

    let _ = fs::remove_dir_all(root);
}

#[cfg(all(feature = "f64-symjit", target_arch = "aarch64"))]
#[test]
#[ignore = "retained real-artifact Direct-Arena stage oracle and interleaved timing evidence"]
fn retained_ddbar_z3g_multistage_invocations_match_packet_execution() {
    use crate::eager_runtime::{
        EagerDirectPreparedKernel, EagerPlanV3Sections, run_retained_multistage_oracle,
        select_retained_multistage_candidate,
    };
    const ARTIFACT_ENV: &str = "PYAMPLICOL_EAGER_DIRECT_REAL_ARTIFACT";
    const PROCESS_ID: &str = "d_dbar_to_z_g_g_g";
    const POINTS: u32 = 129;
    const SAMPLES: usize = 7;
    const REPETITIONS: usize = 100;

    let root = std::env::var_os(ARTIFACT_ENV)
        .map(PathBuf::from)
        .expect("set PYAMPLICOL_EAGER_DIRECT_REAL_ARTIFACT to the retained eager artifact");
    let native = NativeRuntime::load(&root, Some(PROCESS_ID), None)
        .expect("load retained eager artifact through the production runtime");
    let artifact = VerifiedArtifact::open(&root).expect("verify retained eager artifact");
    let selection = artifact
        .select_process(Some(PROCESS_ID))
        .expect("select retained eager process");
    let (loaded, evaluator_root) =
        load_verified_evaluator(&artifact, &selection).expect("load eager execution manifest");
    let LoadedExecutionManifest::EagerV3(manifest) = loaded else {
        panic!("retained artifact is not eager plan-v3");
    };
    let pack = super::eager_v3_load::load_eager_v3_prepared_pack(&artifact, &manifest)
        .expect("load retained prepared kernel pack");
    let container = super::eager_v3_load::open_verified_eager_v3_runtime_container(
        &artifact,
        &evaluator_root,
        &manifest,
    )
    .expect("open retained eager runtime container");
    let decoded =
        super::eager_v3_decode::decode_eager_v3_runtime(&container, &manifest, &pack.manifest)
            .expect("decode retained eager plan-v3");
    let payloads = artifact
        .evaluator_payload_store(&pack.payload_root)
        .expect("open retained evaluator payload store");
    let (projection, couplings, _model_parameter_evaluator) =
        super::eager_v3_load::prepare_plan_v3_parameter_state(
            &pack.manifest,
            &decoded,
            &native.runtime.model_parameters,
            &payloads,
        )
        .expect("prepare retained eager parameter projection");
    let prepared_parameter_count =
        u32::try_from(projection.parameter_count).expect("prepared parameter count fits u32");
    let sections = EagerPlanV3Sections {
        kernels: &decoded.kernel_specs,
        prepared_parameter_count,
        currents: &decoded.currents,
        values: &decoded.values,
        momenta: &decoded.momenta,
        parameters: &decoded.parameters,
        stages: &decoded.stages,
        couplings: &couplings,
        invocations: &decoded.invocations,
        attachments: &decoded.attachments,
        finalizations: &decoded.finalizations,
        closures: &decoded.closures,
        direct_coefficients: &decoded.direct_coefficients,
        selector_domains: &decoded.selector_domains,
        selector_memberships: &decoded.selector_memberships,
        reduction_groups: &decoded.reduction_groups,
        reduction_entries: &decoded.reduction_entries,
        exact_factors: &decoded.exact_factors,
        color_contraction_entry_start: decoded.color_contraction_entry_start,
        color_contraction_entry_count: decoded.color_contraction_entry_count,
    };
    let candidate =
        select_retained_multistage_candidate(sections).expect("select genuine retained stage run");
    let kernel = decoded
        .kernel_specs
        .iter()
        .find(|kernel| kernel.kernel_id == candidate.kernel_id)
        .expect("candidate kernel spec");
    let kernel_manifest = pack
        .manifest
        .kernels
        .iter()
        .find(|kernel| kernel.kernel_id == candidate.kernel_id)
        .expect("candidate prepared kernel manifest");
    let plane_application = kernel_manifest
        .f64_evaluator_manifest
        .get("plane_application")
        .and_then(Value::as_object)
        .expect("candidate SymJIT plane application metadata");
    let application_path = plane_application
        .get("application_path")
        .and_then(Value::as_str)
        .expect("candidate SymJIT plane application path");
    let application_source = payloads
        .source(application_path)
        .expect("resolve candidate SymJIT application");
    let source_bytes = application_source
        .read()
        .expect("read candidate SymJIT application")
        .into_owned();
    let descriptor = symjit_eager_direct::eager_direct_descriptor_for_source_application_bytes(
        &source_bytes,
        u32::try_from(kernel.inputs.len()).expect("candidate input width fits u32"),
        kernel.output_component_count,
        Path::new(&application_source.display_name()),
    )
    .expect("encode retained eager table descriptor");
    let prepared = EagerDirectPreparedKernel {
        kernel_id: candidate.kernel_id,
        role: kernel.role,
        inputs: &kernel.inputs,
        output_component_count: kernel.output_component_count,
        application: crate::eager_runtime::EagerDirectPreparedApplication::Symjit {
            source_application: &source_bytes,
            descriptor: &descriptor,
            expected_optimization_level: 2,
            expected_compression: false,
        },
        display_path: PathBuf::from(application_source.display_name()),
    };

    let mut scalar_pack = pack.manifest.clone();
    scalar_pack
        .kernels
        .retain(|entry| entry.kernel_id == candidate.kernel_id);
    scalar_pack.kernel_variants.clear();
    let mut backend = PreparedEvaluatorBackend::load_from_store(&scalar_pack, &payloads)
        .expect("load retained scalar packet oracle");
    let mut model_parameters =
        vec![crate::EagerComplex64::new(0.0, 0.0); projection.parameter_count];
    for entry in &projection.entries {
        let real = native.runtime.model_parameter_values_f64[entry.runtime_real_index];
        let imaginary = entry
            .runtime_imaginary_index
            .map(|index| native.runtime.model_parameter_values_f64[index])
            .unwrap_or(0.0);
        model_parameters[entry.prepared_index] = crate::EagerComplex64::new(real, imaginary);
    }

    let evidence = run_retained_multistage_oracle(
        sections,
        &[prepared],
        candidate,
        &mut backend,
        &model_parameters,
        POINTS,
        SAMPLES,
        REPETITIONS,
    )
    .expect("retained Direct-Arena stage must match actual packet execution");
    for variant in [evidence.full, evidence.selected] {
        variant
            .direct_call_traffic
            .validate_direct()
            .expect("direct invocation slice has no forbidden materialization traffic");
        assert!(variant.comparison_count > 0);
    }
    eprintln!(
        "retained eager Direct-Arena stage oracle: artifact={} process={} \
         stage_position={} stage_index={} kernel_id={} selector_group={} points={} \
         full_rows={}/{} destinations={} comparisons={} bitwise={}/{} \
         full_max_abs={:.6e} full_max_rel={:.6e} \
         full_direct_ns={:.3} full_direct_mad_ns={:.3} \
         full_packet_ns={:.3} full_packet_mad_ns={:.3} full_ratio={:.6} \
         full_init_write_bytes={} full_factor_fill_bytes={} \
         full_packet_input_bytes/call={} full_packet_output_bytes/call={} \
         full_packet_scatter_bytes/call={} \
         selected_rows={}/{} comparisons={} bitwise={}/{} \
         selected_max_abs={:.6e} selected_max_rel={:.6e} \
         selected_direct_ns={:.3} selected_direct_mad_ns={:.3} \
         selected_packet_ns={:.3} selected_packet_mad_ns={:.3} selected_ratio={:.6} \
         selected_init_write_bytes={} selected_factor_fill_bytes={} \
         selected_packet_input_bytes/call={} selected_packet_output_bytes/call={} \
         selected_packet_scatter_bytes/call={} \
         direct_call_traffic_full={:?} direct_call_traffic_selected={:?}",
        root.display(),
        PROCESS_ID,
        evidence.candidate.stage_position,
        evidence.candidate.stage_index,
        evidence.candidate.kernel_id,
        evidence.candidate.selector_group_id,
        POINTS,
        evidence.full.invocation_count,
        evidence.full.attachment_count,
        evidence.candidate.distinct_destination_count,
        evidence.full.comparison_count,
        evidence.full.bitwise_comparison_count,
        evidence.full.comparison_count,
        evidence.full.maximum_absolute_error,
        evidence.full.maximum_relative_error,
        evidence.full.direct_median_ns,
        evidence.full.direct_mad_ns,
        evidence.full.packet_median_ns,
        evidence.full.packet_mad_ns,
        evidence.full.direct_over_packet,
        evidence.full.arena_initialization_write_bytes,
        evidence.full.factor_fill_write_bytes,
        evidence.full.packet_input_materialization_bytes_per_call,
        evidence.full.packet_output_materialization_bytes_per_call,
        evidence.full.packet_scatter_bytes_per_call,
        evidence.selected.invocation_count,
        evidence.selected.attachment_count,
        evidence.selected.comparison_count,
        evidence.selected.bitwise_comparison_count,
        evidence.selected.comparison_count,
        evidence.selected.maximum_absolute_error,
        evidence.selected.maximum_relative_error,
        evidence.selected.direct_median_ns,
        evidence.selected.direct_mad_ns,
        evidence.selected.packet_median_ns,
        evidence.selected.packet_mad_ns,
        evidence.selected.direct_over_packet,
        evidence.selected.arena_initialization_write_bytes,
        evidence.selected.factor_fill_write_bytes,
        evidence
            .selected
            .packet_input_materialization_bytes_per_call,
        evidence
            .selected
            .packet_output_materialization_bytes_per_call,
        evidence.selected.packet_scatter_bytes_per_call,
        evidence.full.direct_call_traffic,
        evidence.selected.direct_call_traffic,
    );
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn generated_eager_artifact_loads_when_fixture_is_supplied() {
    let Some(root) = std::env::var_os("RUSTICOL_EAGER_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(PathBuf::from(root), None, None)
        .expect("load generated eager artifact through NativeRuntime");
    assert_eq!(runtime.metadata().execution_mode, "eager");
    assert!(matches!(
        runtime.metadata().prepared_backend.as_deref(),
        Some("jit" | "asm" | "cpp")
    ));
    let validation_path = runtime
        .root()
        .join("processes")
        .join(&runtime.metadata().representative_process_key)
        .join("validation-momenta.json");
    let validation: Value =
        serde_json::from_slice(&fs::read(&validation_path).expect("read eager validation momenta"))
            .expect("parse eager validation momenta");
    let momenta = validation["points"][0]
        .as_array()
        .expect("one eager validation point")
        .iter()
        .flat_map(|leg| {
            leg["momentum"]
                .as_array()
                .expect("four momentum components")
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .expect("decimal momentum string")
                        .parse::<f64>()
                        .expect("f64 validation momentum")
                })
        })
        .collect::<Vec<_>>();
    let values = runtime
        .evaluate_f64(&momenta, 1)
        .expect("evaluate generated eager artifact");
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());

    let resolved = runtime
        .evaluate_resolved_f64(&momenta, 1, None, None)
        .expect("resolve generated eager artifact");
    let resolved_total = resolved.totals()[0];
    assert_close_f64(resolved_total, values[0], "eager resolved sum");

    let selected_helicity = runtime
        .helicities()
        .expect("eager helicity metadata")
        .into_iter()
        .find(|helicity| helicity.computed)
        .expect("one computed helicity")
        .id;
    let selected_color = runtime
        .color_ids()
        .expect("eager color metadata")
        .into_iter()
        .next()
        .expect("one color component");
    let selected_colors =
        (runtime.metadata().color_accuracy == "lc").then(|| std::slice::from_ref(&selected_color));
    let selected = runtime
        .evaluate_resolved_f64(
            &momenta,
            1,
            Some(std::slice::from_ref(&selected_helicity)),
            selected_colors,
        )
        .expect("select eager resolved component");
    assert_eq!(selected.shape(), (1, 1, 1));
    assert!(selected.values[0].is_finite());

    let selector_point_count = 4;
    let selector_momenta = momenta.repeat(selector_point_count);
    let all_components = runtime
        .evaluate_resolved_f64(&selector_momenta, selector_point_count, None, None)
        .expect("resolve eager selector fixture");
    let helicity_count = all_components.helicity_ids.len();
    let color_count = all_components.color_ids.len();
    assert!(helicity_count >= 2, "selector fixture needs two helicities");
    assert!(
        color_count >= 1,
        "selector fixture needs one color component"
    );
    let helicity_by_point = [0_u32, 1, 0, 1];
    let color_by_point = [0_u32; 4];
    let color_by_point =
        (runtime.metadata().color_accuracy == "lc").then_some(color_by_point.as_slice());
    let selected_by_point = runtime
        .evaluate_f64_with_selectors(
            &selector_momenta,
            selector_point_count,
            None,
            None,
            Some(&helicity_by_point),
            color_by_point,
        )
        .expect("evaluate eager per-point selectors");
    for point_index in 0..selector_point_count {
        let expected_index =
            (point_index * helicity_count + helicity_by_point[point_index] as usize) * color_count;
        assert_close_f64(
            selected_by_point[point_index],
            all_components.values[expected_index],
            "eager per-point selector",
        );
    }

    {
        for tail in [1_usize, 7, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025] {
            let tail_momenta = momenta.repeat(tail);
            let tail_values = runtime
                .evaluate_f64(&tail_momenta, tail)
                .expect("evaluate required eager Direct-Arena tail");
            for value in tail_values {
                assert_close_f64(value, values[0], "eager Direct-Arena repeated-point tail");
            }
        }

        for tail in [129_usize, 1025] {
            let tail_momenta = momenta.repeat(tail);
            let helicity_by_point = (0..tail)
                .map(|point| (point % helicity_count.min(2)) as u32)
                .collect::<Vec<_>>();
            let color_by_point =
                (runtime.metadata().color_accuracy == "lc").then(|| vec![0_u32; tail]);
            let selected_tail = runtime
                .evaluate_f64_with_selectors(
                    &tail_momenta,
                    tail,
                    None,
                    None,
                    Some(&helicity_by_point),
                    color_by_point.as_deref(),
                )
                .expect("evaluate required eager Direct-Arena selector tail");
            for (point, value) in selected_tail.into_iter().enumerate() {
                let expected_index = helicity_by_point[point] as usize * color_count;
                let expected = if color_by_point.is_some() {
                    resolved.values[expected_index]
                } else {
                    resolved.values[expected_index..expected_index + color_count]
                        .iter()
                        .sum()
                };
                assert_close_f64(value, expected, "eager Direct-Arena selector tail");
            }
        }
    }

    if let Some(compiled_root) = std::env::var_os("RUSTICOL_COMPILED_ARTIFACT") {
        let mut compiled = NativeRuntime::load(PathBuf::from(compiled_root), None, None)
            .expect("load matching compiled artifact");
        let compiled_values = compiled
            .evaluate_f64(&momenta, 1)
            .expect("evaluate matching compiled artifact");
        assert_close_f64(values[0], compiled_values[0], "eager/compiled total");
        let compiled_resolved = compiled
            .evaluate_resolved_f64(&momenta, 1, None, None)
            .expect("resolve matching compiled artifact");
        assert_eq!(resolved.helicity_ids, compiled_resolved.helicity_ids);
        assert_eq!(resolved.color_ids, compiled_resolved.color_ids);
        assert_eq!(resolved.values.len(), compiled_resolved.values.len());
        for (eager, compiled) in resolved.values.iter().zip(&compiled_resolved.values) {
            assert_close_f64(*eager, *compiled, "eager/compiled resolved component");
        }
    }

    let parameters = runtime.model_parameters().expect("eager model parameters");
    let candidates = if let Some(parameter) = parameters
        .iter()
        .find(|parameter| parameter.name == "aEWM1")
    {
        vec![(
            BTreeMap::from([(parameter.name.clone(), (parameter.default * 1.05, 0.0))]),
            BTreeMap::from([(
                parameter.name.clone(),
                (parameter.default, parameter.default_imaginary),
            )]),
        )]
    } else {
        let mut groups = BTreeMap::<String, Vec<(String, f64, f64)>>::new();
        for parameter in parameters
            .iter()
            .filter(|parameter| parameter.mutable && parameter.name.starts_with("coupling."))
        {
            let prefix = parameter
                .name
                .split_once(".component_")
                .map(|(prefix, _)| prefix.to_string())
                .expect("coupling component suffix");
            groups.entry(prefix).or_default().push((
                parameter.name.clone(),
                parameter.default,
                parameter.default_imaginary,
            ));
        }
        groups
            .into_values()
            .map(|selected| {
                let changed = selected
                    .iter()
                    .map(|(name, _, _)| (name.clone(), (0.0, 0.0)))
                    .collect();
                let restored = selected
                    .iter()
                    .map(|(name, real, imaginary)| (name.clone(), (*real, *imaginary)))
                    .collect();
                (changed, restored)
            })
            .collect::<Vec<_>>()
    };
    assert!(!candidates.is_empty(), "one mutable eager parameter group");
    let mut observed_parameter_effect = false;
    for (changed, restored) in candidates {
        runtime
            .set_model_parameters(&changed)
            .expect("update eager model parameters atomically");
        let changed_value = runtime
            .evaluate_f64(&momenta, 1)
            .expect("evaluate updated eager parameters")[0];
        runtime
            .set_model_parameters(&restored)
            .expect("restore eager model parameters atomically");
        let restored_value = runtime
            .evaluate_f64(&momenta, 1)
            .expect("evaluate restored eager parameters")[0];
        assert_eq!(restored_value.to_bits(), values[0].to_bits());
        if changed_value.to_bits() != values[0].to_bits() {
            observed_parameter_effect = true;
            break;
        }
    }
    assert!(
        observed_parameter_effect,
        "one eager parameter affects output"
    );

    if parameters.iter().any(|parameter| parameter.name == "aEWM1")
        && parameters.iter().any(|parameter| parameter.name == "MZ")
    {
        let before_failed_derivation = runtime
            .exact_runtime_state_json()
            .expect("eager state before failed derivation");
        assert!(runtime.set_model_parameter("MZ", 0.0, 0.0).is_err());
        assert_eq!(
            runtime
                .exact_runtime_state_json()
                .expect("eager state after failed derivation"),
            before_failed_derivation
        );
    }

    if let Some(parameter) = parameters.iter().find(|parameter| !parameter.mutable) {
        let before_derived_update = runtime
            .exact_runtime_state_json()
            .expect("eager parameter state before derived update");
        assert!(
            runtime
                .set_model_parameter(
                    &parameter.name,
                    parameter.default,
                    parameter.default_imaginary,
                )
                .is_err()
        );
        assert_eq!(
            runtime
                .exact_runtime_state_json()
                .expect("eager parameter state after derived update"),
            before_derived_update
        );
    }

    let before_failed_update = runtime
        .exact_runtime_state_json()
        .expect("eager parameter state before failed update");
    assert!(
        runtime
            .set_model_parameter("not-a-model-parameter", 1.0, 0.0)
            .is_err()
    );
    assert_eq!(
        runtime
            .exact_runtime_state_json()
            .expect("eager parameter state after failed update"),
        before_failed_update
    );
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn retained_sparse_group_eager_artifact_decodes_when_fixture_is_supplied() {
    let Some(root) = std::env::var_os("RUSTICOL_EAGER_SPARSE_GROUP_ARTIFACT") else {
        return;
    };
    let root = PathBuf::from(root);
    let _native = NativeRuntime::load(&root, None, None)
        .expect("load retained zero-first-group eager artifact through NativeRuntime");
    let artifact = VerifiedArtifact::open(&root).expect("verify retained eager artifact");
    let selection = artifact
        .select_process(None)
        .expect("select retained eager process");
    let (loaded, evaluator_root) =
        load_verified_evaluator(&artifact, &selection).expect("load eager execution manifest");
    let LoadedExecutionManifest::EagerV3(manifest) = loaded else {
        panic!("retained artifact is not eager plan-v3");
    };
    let pack = super::eager_v3_load::load_eager_v3_prepared_pack(&artifact, &manifest)
        .expect("load retained prepared kernel pack");
    let container = super::eager_v3_load::open_verified_eager_v3_runtime_container(
        &artifact,
        &evaluator_root,
        &manifest,
    )
    .expect("open retained eager runtime container");
    let decoded =
        super::eager_v3_decode::decode_eager_v3_runtime(&container, &manifest, &pack.manifest)
            .expect("decode retained zero-first-group eager plan-v3");
    let group_ids = decoded
        .invocations
        .iter()
        .map(|row| row.evaluation_group_id)
        .collect::<Vec<_>>();
    assert!(group_ids.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(
        group_ids
            .windows(2)
            .any(|pair| pair[1] > pair[0].saturating_add(1)),
        "retained eager failure fixture must contain a pruned group gap: {group_ids:?}"
    );
}

#[cfg(all(
    any(feature = "f64-compiled", feature = "f64-symjit"),
    any(target_os = "linux", target_os = "macos")
))]
#[test]
fn generated_eager_native_into_is_warmed_allocation_free_when_fixture_is_supplied() {
    let Some(root) = std::env::var_os("RUSTICOL_EAGER_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(PathBuf::from(root), None, None)
        .expect("load generated eager allocation fixture");
    let validation_path = runtime
        .root()
        .join("processes")
        .join(&runtime.metadata().representative_process_key)
        .join("validation-momenta.json");
    let validation: Value =
        serde_json::from_slice(&fs::read(&validation_path).expect("read eager validation momenta"))
            .expect("parse eager validation momenta");
    let point = validation["points"][0]
        .as_array()
        .expect("one eager validation point")
        .iter()
        .flat_map(|leg| {
            leg["momentum"]
                .as_array()
                .expect("four momentum components")
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .expect("decimal momentum string")
                        .parse::<f64>()
                        .expect("f64 validation momentum")
                })
        })
        .collect::<Vec<_>>();
    let point_count = 129usize;
    let momenta = point.repeat(point_count);
    let mut output = vec![-1.0; point_count];

    runtime
        .evaluate_f64_into(&momenta, point_count, &mut output)
        .expect("warm eager borrowed-input totals");
    let (result, allocations, bytes) =
        count_allocations(|| runtime.evaluate_f64_into(&momenta, point_count, &mut output));
    result.expect("repeat eager borrowed-input totals");
    assert_eq!(
        (allocations, bytes),
        (0, 0),
        "warmed eager borrowed-input totals allocated"
    );
    let arena_profile = runtime
        .evaluate_f64_arena_profile_repeated(&momenta, point_count, 3, None, None)
        .expect("profile warmed eager borrowed-input totals");
    assert_eq!(arena_profile.values, output);
    assert_eq!(
        arena_profile
            .profile
            .native_input_container_allocation_count,
        0
    );
    assert_eq!(arena_profile.profile.native_output_allocation_count, 0);
    assert_eq!(arena_profile.profile.observed_scratch_reallocation_count, 0);
    assert_eq!(
        arena_profile.profile.native_input_component_count,
        u64::try_from(momenta.len() * 3).expect("profile input count"),
    );

    let selected_helicity = runtime
        .helicities()
        .expect("eager helicity metadata")
        .into_iter()
        .find(|helicity| helicity.computed)
        .expect("one computed eager helicity")
        .id;
    let selected_helicities = [selected_helicity];
    let selected_colors = (runtime.metadata().color_accuracy == "lc").then(|| {
        [runtime
            .color_ids()
            .expect("eager color metadata")
            .into_iter()
            .next()
            .expect("one eager color")]
    });
    runtime
        .evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            Some(&selected_helicities),
            selected_colors.as_ref().map(|ids| ids.as_slice()),
            None,
            None,
            &mut output,
        )
        .expect("warm eager global selectors");
    let (result, allocations, bytes) = count_allocations(|| {
        runtime.evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            Some(&selected_helicities),
            selected_colors.as_ref().map(|ids| ids.as_slice()),
            None,
            None,
            &mut output,
        )
    });
    result.expect("repeat eager global selectors");
    assert_eq!(
        (allocations, bytes),
        (0, 0),
        "warmed eager global selectors allocated"
    );
    let mut profile_cases = vec![(
        "selected-helicity",
        Some(selected_helicities.as_slice()),
        None,
    )];
    if let Some(ids) = selected_colors.as_ref() {
        profile_cases.push(("selected-flow", None, Some(ids.as_slice())));
    }
    for (label, helicities, colors) in profile_cases {
        runtime
            .evaluate_f64_into_with_selectors(
                &momenta,
                point_count,
                helicities,
                colors,
                None,
                None,
                &mut output,
            )
            .unwrap_or_else(|error| panic!("evaluate eager {label} oracle: {error}"));
        let selected_profile = runtime
            .evaluate_f64_arena_profile_repeated(&momenta, point_count, 3, helicities, colors)
            .unwrap_or_else(|error| panic!("profile eager {label}: {error}"));
        assert_eq!(
            selected_profile.values, output,
            "profiled eager {label} values"
        );
        assert_eq!(
            selected_profile
                .profile
                .native_input_container_allocation_count,
            0,
            "profiled eager {label} input allocations"
        );
        assert_eq!(
            selected_profile.profile.native_output_allocation_count, 0,
            "profiled eager {label} output allocations"
        );
        assert_eq!(
            selected_profile.profile.observed_scratch_reallocation_count, 0,
            "profiled eager {label} scratch reallocations"
        );
    }

    let helicity_count = runtime
        .helicities()
        .expect("eager helicity metadata")
        .len()
        .min(2);
    let helicity_by_point = (0..point_count)
        .map(|point| (point % helicity_count) as u32)
        .collect::<Vec<_>>();
    let color_by_point =
        (runtime.metadata().color_accuracy == "lc").then(|| vec![0_u32; point_count]);
    runtime
        .evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            None,
            None,
            Some(&helicity_by_point),
            color_by_point.as_deref(),
            &mut output,
        )
        .expect("warm eager alternating selectors");
    let (result, allocations, bytes) = count_allocations(|| {
        runtime.evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            None,
            None,
            Some(&helicity_by_point),
            color_by_point.as_deref(),
            &mut output,
        )
    });
    result.expect("repeat eager alternating selectors");
    assert_eq!(
        (allocations, bytes),
        (0, 0),
        "warmed eager alternating selectors allocated"
    );

    runtime
        .evaluate_resolved_f64(&momenta, point_count, None, None)
        .expect("warm eager borrowed-input resolved values");
    let (resolved, allocations, bytes) =
        count_allocations(|| runtime.evaluate_resolved_f64(&momenta, point_count, None, None));
    let resolved = resolved.expect("repeat eager borrowed-input resolved values");
    let result_owned_allocation_bound = resolved.helicity_ids.len() + resolved.color_ids.len() + 12;
    assert!(
        allocations <= result_owned_allocation_bound,
        "warmed eager resolved path retained point-sized input containers: \
         allocations={allocations} bytes={bytes} result_bound={result_owned_allocation_bound}"
    );
    assert_eq!(resolved.point_count, point_count);
    assert_eq!(
        resolved.values.len(),
        point_count * resolved.helicity_ids.len() * resolved.color_ids.len()
    );

    runtime
        .evaluate_resolved_f64(
            &momenta,
            point_count,
            Some(&selected_helicities),
            selected_colors.as_ref().map(|ids| ids.as_slice()),
        )
        .expect("warm eager selected borrowed-input resolved values");
    let (selected_resolved, allocations, bytes) = count_allocations(|| {
        runtime.evaluate_resolved_f64(
            &momenta,
            point_count,
            Some(&selected_helicities),
            selected_colors.as_ref().map(|ids| ids.as_slice()),
        )
    });
    let selected_resolved =
        selected_resolved.expect("repeat eager selected borrowed-input resolved values");
    let result_owned_allocation_bound =
        selected_resolved.helicity_ids.len() + selected_resolved.color_ids.len() + 12;
    assert!(
        allocations <= result_owned_allocation_bound,
        "warmed selected eager resolved path retained point-sized input containers: \
         allocations={allocations} bytes={bytes} result_bound={result_owned_allocation_bound}"
    );
    assert_eq!(selected_resolved.point_count, point_count);
    assert_eq!(
        selected_resolved.values.len(),
        point_count * selected_resolved.helicity_ids.len() * selected_resolved.color_ids.len()
    );

    let mut sentinel = [37.0_f64, 41.0];
    assert!(
        runtime
            .evaluate_f64_into(&momenta, point_count, &mut sentinel)
            .is_err()
    );
    assert_eq!(sentinel, [37.0, 41.0]);
}

fn assert_close_f64(actual: f64, expected: f64, context: &str) {
    let tolerance = 1.0e-15 + 1.0e-12 * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: {actual:.17e} != {expected:.17e} (tolerance {tolerance:.3e})"
    );
}
