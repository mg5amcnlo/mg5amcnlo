// SPDX-License-Identifier: 0BSD

use super::evaluation::accumulate_selected_lc_replay_resolved_f64;
use super::physics::certifies_lc_direct_total_source;
use super::*;
use serde_json::json;

fn symjit_plane_application_manifest(target: Value) -> SymjitPlaneApplicationManifest {
    serde_json::from_value(json!({
        "application_path": "evaluators/kernel.plane.symjit",
        "application_abi": SYMJIT_PLANE_APPLICATION_ABI,
        "storage_abi": SYMJIT_APPLICATION_STORAGE_ABI,
        "element_layout": "split-complex-plane-major",
        "descriptor_order": "inputs-re-im-then-outputs-re-im",
        "input_complex_count": 2,
        "output_complex_count": 1,
        "input_plane_count": 4,
        "output_plane_count": 2,
        "compiler_type": "native",
        "translation_mode": "symbolica-structured-instructions",
        "optimization_level": 2,
        "simd": true,
        "complex": true,
        "fast_math": true,
        "fast_complex": false,
        "compression": false,
        "threading": false,
        "direct_arena": true,
        "source_digest": "a".repeat(64),
        "target": target,
    }))
    .unwrap()
}

#[test]
fn symjit_plane_application_manifest_authenticates_target_shape() {
    for target in [
        json!({"word_bits": 64, "endianness": "little"}),
        json!({
            "word_bits": 64,
            "endianness": "little",
            "triple": "aarch64-apple-darwin",
            "cpu_features": ["aes", "neon"],
        }),
    ] {
        symjit_plane_application_manifest(target)
            .validate(2, 1, 2)
            .unwrap();
    }

    for target in [
        json!({"word_bits": 32, "endianness": "little"}),
        json!({"word_bits": 64, "endianness": "big"}),
        json!({"word_bits": 64, "endianness": "little", "triple": ""}),
        json!({"word_bits": 64, "endianness": "little", "cpu_features": "neon"}),
        json!({
            "word_bits": 64,
            "endianness": "little",
            "cpu_features": ["neon", "neon"],
        }),
        json!({"word_bits": 64, "endianness": "little", "unexpected": true}),
    ] {
        let error = symjit_plane_application_manifest(target)
            .validate(2, 1, 2)
            .unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.message().contains("regenerate"));
    }
}

fn native_compiled_direct_application() -> NativeCompiledDirectApplicationManifest {
    NativeCompiledDirectApplicationManifest {
        application_abi: "pyamplicol-native-compiled-direct-application-v1".to_string(),
        function_name: "direct_leaf".to_string(),
        source_path: "compiled/direct_leaf.cpp".to_string(),
        library_path: "compiled/libdirect_leaf".to_string(),
        target: NativeCompiledDirectTargetManifest {
            triple: "aarch64-apple-darwin".to_string(),
            cpu_features: vec!["neon".to_string()],
        },
        evaluator_state_sha256: "a".repeat(64),
        instruction_count: 17,
        temporary_count: 3,
        input_plane_count: 4,
        scalar_input_count: 1,
        output_plane_count: 4,
        simd_lane_width: 2,
        logical_stack_bytes: 160,
        output_semantics: "factor-free-overwrite".to_string(),
    }
}

#[test]
fn native_compiled_direct_application_manifest_validates_shape() {
    let manifest = EvaluatorManifest::CompiledComplex {
        runtime_capability: "symbolica.compiled-cpp.complex-f64.v1".to_string(),
        function_name: "direct_leaf".to_string(),
        input_len: 3,
        output_len: 2,
        library_path: "compiled/liblegacy_leaf".to_string(),
        evaluator_state_path: Some("compiled/direct_leaf.evaluator.bin".to_string()),
        number_type: "complex".to_string(),
        native_direct_application: Some(native_compiled_direct_application()),
    };

    assert_eq!(manifest.io_len().unwrap(), (3, 2));
}

#[test]
fn native_compiled_direct_application_manifest_rejects_inconsistent_stack() {
    let mut application = native_compiled_direct_application();
    application.logical_stack_bytes += 32;
    let manifest = EvaluatorManifest::CompiledComplex {
        runtime_capability: "symbolica.compiled-cpp.complex-f64.v1".to_string(),
        function_name: "direct_leaf".to_string(),
        input_len: 3,
        output_len: 2,
        library_path: "compiled/liblegacy_leaf".to_string(),
        evaluator_state_path: Some("compiled/direct_leaf.evaluator.bin".to_string()),
        number_type: "complex".to_string(),
        native_direct_application: Some(application),
    };

    let error = manifest.io_len().unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("logical stack metadata"));
}

pub(super) fn test_physics_runtime(color_accuracy: &str) -> PhysicsRuntime {
    let contracted = color_accuracy != "lc";
    let color_components = if contracted {
        vec![crate::ColorComponent::ContractedColor(
            crate::ContractedColor {
                id: "contracted".to_string(),
                index: 0,
                description: "contracted color sum".to_string(),
            },
        )]
    } else {
        vec![
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:0".to_string(),
                index: 0,
                word: vec![1, 2],
                representative_id: "flow:0".to_string(),
                computed: true,
                coefficient: 1.0,
            }),
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:1".to_string(),
                index: 1,
                word: vec![2, 1],
                representative_id: "flow:0".to_string(),
                computed: false,
                coefficient: 1.0,
            }),
        ]
    };
    let physical_color_ids = color_components
        .iter()
        .map(|item| item.id().to_string())
        .collect();
    let mut helicities = vec![
        crate::Helicity {
            id: "hel:+-".to_string(),
            index: 0,
            values: vec![1, -1, 1],
            representative_id: "hel:+-".to_string(),
            computed: true,
            structural_zero: false,
            coefficient: 1.0,
        },
        crate::Helicity {
            id: "hel:-+".to_string(),
            index: 1,
            values: vec![-1, 1, 1],
            representative_id: "hel:+-".to_string(),
            computed: false,
            structural_zero: false,
            coefficient: 1.0,
        },
    ];
    if !contracted {
        helicities.push(crate::Helicity {
            id: "hel:zero".to_string(),
            index: 2,
            values: vec![1, 1, 1],
            representative_id: "hel:zero".to_string(),
            computed: false,
            structural_zero: true,
            coefficient: 0.0,
        });
    }
    PhysicsRuntime::new(ProcessPhysicsV1 {
        schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
        kind: "pyamplicol-resolved-physics".to_string(),
        process_id: "x_x_to_y".to_string(),
        process: "x x > y".to_string(),
        color_accuracy: if contracted {
            if color_accuracy == "nlc" {
                crate::ColorAccuracy::Nlc
            } else {
                crate::ColorAccuracy::Full
            }
        } else {
            crate::ColorAccuracy::Lc
        },
        coverage: crate::Coverage {
            helicities: "complete".to_string(),
            color: if contracted { "contracted" } else { "complete" }.to_string(),
            color_kind: if contracted {
                "contracted-color"
            } else {
                "physical-lc-flows"
            }
            .to_string(),
            structural_zero_helicity_count: usize::from(!contracted),
        },
        external_particles: vec![
            test_external_particle(0, "x", 1, crate::ParticleRole::Initial),
            test_external_particle(1, "x~", -1, crate::ParticleRole::Initial),
            test_external_particle(2, "y", 23, crate::ParticleRole::Final),
        ],
        helicities,
        color_components,
        reduction: crate::Reduction {
            kind: if contracted {
                crate::ReductionKind::ContractedColor
            } else {
                crate::ReductionKind::LcDiagonal
            },
            groups: vec![crate::ReductionGroup {
                id: "group:7".to_string(),
                representative_helicity_id: "hel:+-".to_string(),
                physical_helicity_ids: vec!["hel:+-".to_string(), "hel:-+".to_string()],
                representative_color_id: if contracted {
                    "contracted".to_string()
                } else {
                    "flow:0".to_string()
                },
                physical_color_ids,
            }],
        },
        model_parameters: Vec::new(),
        selectors: crate::SelectorCapabilities {
            helicity: true,
            color_flow: !contracted,
            contracted_color: false,
        },
        extensions: BTreeMap::new(),
    })
    .unwrap()
}

#[test]
fn total_selector_canonicalization_elides_only_exact_authenticated_axes() {
    let physics = test_physics_runtime("lc");
    let all_helicities = physics
        .manifest
        .helicities
        .iter()
        .map(|helicity| helicity.id.clone())
        .collect::<BTreeSet<_>>();
    let all_colors = physics
        .manifest
        .color_components
        .iter()
        .map(|color| color.id().to_string())
        .collect::<BTreeSet<_>>();
    assert!(
        physics
            .canonical_total_helicity_selector(Some(&all_helicities))
            .is_none()
    );
    assert!(
        physics
            .canonical_total_color_selector(Some(&all_colors))
            .is_none()
    );

    let helicity_subset = BTreeSet::from(["hel:+-".to_string()]);
    let color_subset = BTreeSet::from(["flow:0".to_string()]);
    assert_eq!(
        physics.canonical_total_helicity_selector(Some(&helicity_subset)),
        Some(&helicity_subset)
    );
    assert_eq!(
        physics.canonical_total_color_selector(Some(&color_subset)),
        Some(&color_subset)
    );

    let same_size_unknown_helicities = BTreeSet::from([
        "hel:+-".to_string(),
        "hel:-+".to_string(),
        "hel:unknown".to_string(),
    ]);
    let same_size_unknown_colors =
        BTreeSet::from(["flow:0".to_string(), "flow:unknown".to_string()]);
    let retained_helicities =
        physics.canonical_total_helicity_selector(Some(&same_size_unknown_helicities));
    let retained_colors = physics.canonical_total_color_selector(Some(&same_size_unknown_colors));
    assert_eq!(retained_helicities, Some(&same_size_unknown_helicities));
    assert_eq!(retained_colors, Some(&same_size_unknown_colors));
    assert_eq!(
        physics
            .selected_helicity_indices(retained_helicities)
            .unwrap_err()
            .kind(),
        crate::RusticolErrorKind::Selector
    );
    assert_eq!(
        physics
            .selected_color_indices(retained_colors)
            .unwrap_err()
            .kind(),
        crate::RusticolErrorKind::Selector
    );

    let mut selected_manifest = test_physics_runtime("full").manifest;
    selected_manifest.helicities.truncate(1);
    selected_manifest.reduction.groups[0]
        .physical_helicity_ids
        .truncate(1);
    selected_manifest.coverage.helicities = "selected".to_string();
    let selected_physics = PhysicsRuntime::new(selected_manifest).unwrap();
    let sole_helicity = BTreeSet::from([selected_physics.manifest.helicities[0].id.clone()]);
    assert!(
        selected_physics
            .canonical_total_helicity_selector(Some(&sole_helicity))
            .is_none()
    );
}

#[test]
fn compiled_symmetric_group_rejects_exhaustive_runtime_selector_before_canonicalization() {
    let physics = test_physics_runtime("full");
    let all_helicities = physics
        .manifest
        .helicities
        .iter()
        .map(|helicity| helicity.id.clone())
        .collect::<BTreeSet<_>>();
    let plan = crate::recurrence::RecurrenceColorContraction::symmetric_group_s3_for_runtime_test(
        vec![0; 13],
        13,
    );
    let contraction = ColorContractionRuntime {
        group_count: 0,
        entries: Vec::new(),
        repeated_block: None,
        symmetric_group: Some(CompiledSymmetricGroupColorContraction {
            plan,
            ordered_group_indices: Vec::new(),
            workspace: None,
        }),
        group_scratch_f64: Vec::new(),
    };
    let mut runtime = empty_generic_runtime();
    runtime.physics = Some(Arc::new(physics));
    runtime.amplitude_stage = Some(test_amplitude_runtime(Vec::new(), Some(contraction)));
    let batch = F64MomentumBatchView::from_contiguous_prevalidated(&[], 1, 0, None).unwrap();
    let error = runtime
        .run_f64_selected_into_unprofiled(batch, Some(&all_helicities), None, &mut [0.0])
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.message().contains("runtime selectors"));
}

fn test_external_particle(
    index: usize,
    particle: &str,
    pdg: i32,
    role: crate::ParticleRole,
) -> crate::ExternalParticle {
    crate::ExternalParticle {
        index,
        label: index + 1,
        particle: particle.to_string(),
        pdg,
        role,
        momentum_slot: index,
        momentum_components: [
            "E".to_string(),
            "px".to_string(),
            "py".to_string(),
            "pz".to_string(),
        ],
    }
}

fn replay_test_physics() -> PhysicsRuntime {
    PhysicsRuntime::new(ProcessPhysicsV1 {
        schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
        kind: "pyamplicol-resolved-physics".to_string(),
        process_id: "d_dbar_to_g_g".to_string(),
        process: "d d~ > g g".to_string(),
        color_accuracy: crate::ColorAccuracy::Lc,
        coverage: crate::Coverage {
            helicities: "complete".to_string(),
            color: "complete".to_string(),
            color_kind: "physical-lc-flows".to_string(),
            structural_zero_helicity_count: 0,
        },
        external_particles: vec![
            test_external_particle(0, "d", 1, crate::ParticleRole::Initial),
            test_external_particle(1, "d~", -1, crate::ParticleRole::Initial),
            test_external_particle(2, "g", 21, crate::ParticleRole::Final),
            test_external_particle(3, "g", 21, crate::ParticleRole::Final),
        ],
        helicities: vec![
            crate::Helicity {
                id: "h:+1,-1,+1,-1".to_string(),
                index: 0,
                values: vec![1, -1, 1, -1],
                computed: true,
                structural_zero: false,
                representative_id: "h:+1,-1,+1,-1".to_string(),
                coefficient: 1.0,
            },
            crate::Helicity {
                id: "h:+1,-1,-1,+1".to_string(),
                index: 1,
                values: vec![1, -1, -1, 1],
                computed: false,
                structural_zero: false,
                representative_id: "h:+1,-1,+1,-1".to_string(),
                coefficient: 1.0,
            },
        ],
        color_components: vec![
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:1,2,3,4".to_string(),
                index: 0,
                word: vec![1, 2, 3, 4],
                computed: true,
                representative_id: "flow:1,2,3,4".to_string(),
                coefficient: 1.0,
            }),
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:1,4,3,2".to_string(),
                index: 1,
                word: vec![1, 4, 3, 2],
                computed: false,
                representative_id: "flow:1,2,3,4".to_string(),
                coefficient: 1.0,
            }),
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:1,2,4,3".to_string(),
                index: 2,
                word: vec![1, 2, 4, 3],
                computed: false,
                representative_id: "flow:1,2,3,4".to_string(),
                coefficient: 1.0,
            }),
        ],
        reduction: crate::Reduction {
            kind: crate::ReductionKind::LcDiagonal,
            groups: vec![crate::ReductionGroup {
                id: "reduction:7".to_string(),
                representative_helicity_id: "h:+1,-1,+1,-1".to_string(),
                representative_color_id: "flow:1,2,3,4".to_string(),
                physical_helicity_ids: vec!["h:+1,-1,+1,-1".to_string()],
                physical_color_ids: vec!["flow:1,2,3,4".to_string()],
            }],
        },
        model_parameters: Vec::new(),
        selectors: crate::SelectorCapabilities {
            helicity: true,
            color_flow: true,
            contracted_color: false,
        },
        extensions: BTreeMap::new(),
    })
    .unwrap()
}

#[test]
fn lc_replay_routes_materialized_cells_to_public_axes_and_selectors() {
    let physics = replay_test_physics();
    let mappings = vec![Vec::new(), vec![(2, 3), (3, 2)]];
    let routes = vec![
        vec![LcTopologyReplaySectorRoute {
            physical_sector_id: 0,
            materialized_sector_id: 0,
            weight: 2.0,
            sign: 1,
            amplitude_factor: [2.0, 0.0],
            residual: false,
        }],
        vec![LcTopologyReplaySectorRoute {
            physical_sector_id: 1,
            materialized_sector_id: 0,
            weight: 1.0,
            sign: -1,
            amplitude_factor: [-1.0, 0.0],
            residual: false,
        }],
    ];
    let plan = physics
        .lc_resolved_replay_plan(
            &mappings,
            &routes,
            &BTreeMap::from([(
                0,
                LcMaterializedSector {
                    color_index: 0,
                    reduction_weight: 2.0,
                },
            )]),
        )
        .unwrap();
    assert_eq!(plan.color_count, 3);
    let replay_selection = physics
        .select_lc_resolved_replay_plan(
            &plan,
            Some(&BTreeSet::from(["h:+1,-1,-1,+1".to_string()])),
            Some(&BTreeSet::from(["flow:1,2,4,3".to_string()])),
        )
        .unwrap();
    assert_eq!(replay_selection.mapping_indices, vec![1]);
    assert_eq!(replay_selection.entries.len(), 1);
    assert_eq!(replay_selection.entries[0].routes.len(), 1);
    assert_eq!(replay_selection.entries[0].routes[0].source_index, 0);
    assert_eq!(replay_selection.entries[0].routes[0].target_index, 0);
    assert_eq!(replay_selection.source_helicity_indices, vec![vec![0]]);
    assert_eq!(replay_selection.source_color_indices, vec![vec![0]]);

    let materialized = ResolvedValues {
        values: vec![
            3.0, 0.0, 0.0, 0.0, 0.0, 0.0, // identity, point 0
            5.0, 0.0, 0.0, 0.0, 0.0, 0.0, // identity, point 1
            7.0, 0.0, 0.0, 0.0, 0.0, 0.0, // swap, point 0
            11.0, 0.0, 0.0, 0.0, 0.0, 0.0, // swap, point 1
        ],
        point_count: 4,
        helicity_indices: vec![0, 1],
        color_indices: vec![0, 1, 2],
    };
    let mut full = vec![0.0; 12];
    super::evaluation::accumulate_lc_replay_resolved_f64(
        &mut full,
        2,
        &materialized,
        &plan.entries,
        6,
    )
    .unwrap();

    assert_eq!(
        full,
        vec![3.0, 3.0, 0.0, 0.0, 0.0, 7.0, 5.0, 5.0, 0.0, 0.0, 0.0, 11.0]
    );
    assert_eq!(full[..6].iter().sum::<f64>(), 2.0 * 3.0 + 7.0);
    assert_eq!(full[6..].iter().sum::<f64>(), 2.0 * 5.0 + 11.0);

    let selected = super::evaluation::select_resolved_values(
        full,
        2,
        &physics,
        Some(&BTreeSet::from(["h:+1,-1,-1,+1".to_string()])),
        Some(&BTreeSet::from(["flow:1,2,4,3".to_string()])),
    )
    .unwrap();
    assert_eq!(selected.helicity_indices, vec![1]);
    assert_eq!(selected.color_indices, vec![2]);
    assert_eq!(selected.values, vec![7.0, 11.0]);
}

#[test]
fn lc_replay_requires_every_public_flow_reduction_member() {
    let mut manifest = replay_test_physics().manifest;
    manifest.color_components.remove(1);
    manifest.color_components[1] = match manifest.color_components[1].clone() {
        crate::ColorComponent::LcFlow(mut flow) => {
            flow.index = 1;
            crate::ColorComponent::LcFlow(flow)
        }
        value => value,
    };
    manifest.coverage.color = "selected".to_string();
    let physics = PhysicsRuntime::new(manifest).unwrap();

    let error = physics
        .lc_resolved_replay_plan(
            &vec![Vec::new()],
            &[vec![LcTopologyReplaySectorRoute {
                physical_sector_id: 0,
                materialized_sector_id: 0,
                weight: 2.0,
                sign: 1,
                amplitude_factor: [2.0, 0.0],
                residual: false,
            }]],
            &BTreeMap::from([(
                0,
                LcMaterializedSector {
                    color_index: 0,
                    reduction_weight: 2.0,
                },
            )]),
        )
        .unwrap_err();

    assert!(error.to_string().contains("missing replayed LC flow word"));
}

#[test]
fn lc_replay_expands_trace_reflection_for_residual_sectors() {
    fn flow(
        index: usize,
        word: &[usize],
        computed: bool,
        representative: &[usize],
    ) -> crate::ColorComponent {
        let id = format!(
            "flow:{}",
            word.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        let representative_id = format!(
            "flow:{}",
            representative
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        crate::ColorComponent::LcFlow(crate::LcColorFlow {
            id,
            index,
            word: word.to_vec(),
            computed,
            representative_id,
            coefficient: 1.0,
        })
    }

    let mut manifest = replay_test_physics().manifest;
    manifest.color_components = vec![
        flow(0, &[1, 2, 3, 4], true, &[1, 2, 3, 4]),
        flow(1, &[1, 2, 4, 3], false, &[1, 2, 3, 4]),
        flow(2, &[1, 3, 2, 4], true, &[1, 3, 2, 4]),
        flow(3, &[1, 3, 4, 2], false, &[1, 2, 3, 4]),
        flow(4, &[1, 4, 2, 3], false, &[1, 3, 2, 4]),
        flow(5, &[1, 4, 3, 2], false, &[1, 2, 3, 4]),
    ];
    manifest.reduction.groups = vec![
        crate::ReductionGroup {
            id: "reduction:0".to_string(),
            representative_helicity_id: "h:+1,-1,+1,-1".to_string(),
            representative_color_id: "flow:1,2,3,4".to_string(),
            physical_helicity_ids: vec!["h:+1,-1,+1,-1".to_string()],
            physical_color_ids: vec!["flow:1,2,3,4".to_string(), "flow:1,4,3,2".to_string()],
        },
        crate::ReductionGroup {
            id: "reduction:1".to_string(),
            representative_helicity_id: "h:+1,-1,+1,-1".to_string(),
            representative_color_id: "flow:1,3,2,4".to_string(),
            physical_helicity_ids: vec!["h:+1,-1,+1,-1".to_string()],
            physical_color_ids: vec!["flow:1,3,2,4".to_string(), "flow:1,4,2,3".to_string()],
        },
    ];
    let physics = PhysicsRuntime::new(manifest).unwrap();
    let mappings = vec![Vec::new(), vec![(2, 3), (3, 2)]];
    let routes = vec![
        vec![
            LcTopologyReplaySectorRoute {
                physical_sector_id: 0,
                materialized_sector_id: 0,
                weight: 2.0,
                sign: 1,
                amplitude_factor: [2.0, 0.0],
                residual: false,
            },
            LcTopologyReplaySectorRoute {
                physical_sector_id: 2,
                materialized_sector_id: 2,
                weight: 1.0,
                sign: 1,
                amplitude_factor: [1.0, 0.0],
                residual: true,
            },
        ],
        vec![LcTopologyReplaySectorRoute {
            physical_sector_id: 1,
            materialized_sector_id: 0,
            weight: 2.0,
            sign: 1,
            amplitude_factor: [2.0, 0.0],
            residual: false,
        }],
    ];
    let materialized = BTreeMap::from([
        (
            0,
            LcMaterializedSector {
                color_index: 0,
                reduction_weight: 2.0,
            },
        ),
        (
            2,
            LcMaterializedSector {
                color_index: 2,
                reduction_weight: 2.0,
            },
        ),
    ]);

    let plan = physics
        .lc_resolved_replay_plan(&mappings, &routes, &materialized)
        .unwrap();
    let target_colors = plan
        .entries
        .iter()
        .flat_map(|entry| entry.routes.iter())
        .map(|route| route.target_index % plan.color_count)
        .collect::<BTreeSet<_>>();

    assert_eq!(target_colors, (0..6).collect());
    assert!(
        plan.entries
            .iter()
            .flat_map(|entry| entry.routes.iter())
            .all(|route| route.weight.to_bits() == 1.0f64.to_bits())
    );
}

fn alias_test_physics(color_accuracy: &str) -> ProcessPhysicsV1 {
    let contracted = color_accuracy != "lc";
    let representative_helicity_id = "h:+1,-1,+0,+1,-1".to_string();
    let helicities = vec![
        crate::Helicity {
            id: representative_helicity_id.clone(),
            index: 0,
            values: vec![1, -1, 0, 1, -1],
            representative_id: representative_helicity_id.clone(),
            computed: true,
            structural_zero: false,
            coefficient: 1.0,
        },
        crate::Helicity {
            id: "h:-1,+1,+0,-1,+1".to_string(),
            index: 1,
            values: vec![-1, 1, 0, -1, 1],
            representative_id: representative_helicity_id.clone(),
            computed: false,
            structural_zero: false,
            coefficient: 1.0,
        },
    ];
    let (color_components, reduction_kind, representative_color_id, physical_color_ids) =
        if contracted {
            (
                vec![crate::ColorComponent::ContractedColor(
                    crate::ContractedColor {
                        id: "color:contracted".to_string(),
                        index: 0,
                        description: "contracted color sum".to_string(),
                    },
                )],
                crate::ReductionKind::ContractedColor,
                "color:contracted".to_string(),
                vec!["color:contracted".to_string()],
            )
        } else {
            (
                vec![
                    crate::ColorComponent::LcFlow(crate::LcColorFlow {
                        id: "flow:1,3,4,5,2".to_string(),
                        index: 0,
                        word: vec![1, 3, 4, 5, 2],
                        representative_id: "flow:1,3,4,5,2".to_string(),
                        computed: true,
                        coefficient: 1.0,
                    }),
                    crate::ColorComponent::LcFlow(crate::LcColorFlow {
                        id: "flow:1,5,4,3,2".to_string(),
                        index: 1,
                        word: vec![1, 5, 4, 3, 2],
                        representative_id: "flow:1,3,4,5,2".to_string(),
                        computed: false,
                        coefficient: 1.0,
                    }),
                ],
                crate::ReductionKind::LcDiagonal,
                "flow:1,3,4,5,2".to_string(),
                vec!["flow:1,3,4,5,2".to_string(), "flow:1,5,4,3,2".to_string()],
            )
        };
    ProcessPhysicsV1 {
        schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
        kind: "pyamplicol-resolved-physics".to_string(),
        process_id: "representative".to_string(),
        process: "d d~ > z g a".to_string(),
        color_accuracy: match color_accuracy {
            "lc" => crate::ColorAccuracy::Lc,
            "nlc" => crate::ColorAccuracy::Nlc,
            _ => crate::ColorAccuracy::Full,
        },
        coverage: crate::Coverage {
            helicities: "complete".to_string(),
            color: if contracted { "contracted" } else { "complete" }.to_string(),
            color_kind: if contracted {
                "contracted-color"
            } else {
                "physical-lc-flows"
            }
            .to_string(),
            structural_zero_helicity_count: 0,
        },
        external_particles: vec![
            test_external_particle(0, "d", 1, crate::ParticleRole::Initial),
            test_external_particle(1, "d~", -1, crate::ParticleRole::Initial),
            test_external_particle(2, "z", 23, crate::ParticleRole::Final),
            test_external_particle(3, "g", 21, crate::ParticleRole::Final),
            test_external_particle(4, "a", 22, crate::ParticleRole::Final),
        ],
        helicities,
        color_components,
        reduction: crate::Reduction {
            kind: reduction_kind,
            groups: vec![crate::ReductionGroup {
                id: "group:7".to_string(),
                representative_helicity_id,
                representative_color_id,
                physical_helicity_ids: vec![
                    "h:+1,-1,+0,+1,-1".to_string(),
                    "h:-1,+1,+0,-1,+1".to_string(),
                ],
                physical_color_ids,
            }],
        },
        model_parameters: Vec::new(),
        selectors: crate::SelectorCapabilities {
            helicity: true,
            color_flow: !contracted,
            contracted_color: false,
        },
        extensions: BTreeMap::new(),
    }
}

fn three_cycle_alias() -> crate::ProcessAlias {
    crate::ProcessAlias {
        id: "cycled".to_string(),
        expression: "d d~ > a z g".to_string(),
        external_pdgs: vec![1, -1, 22, 23, 21],
        external_permutation: vec![0, 1, 3, 4, 2],
    }
}

fn stored_alias_selection(
    physics: &ProcessPhysicsV1,
    alias: &crate::ProcessAlias,
) -> crate::ArtifactSelection {
    crate::ArtifactSelection {
        process: crate::ArtifactProcess {
            id: physics.process_id.clone(),
            expression: physics.process.clone(),
            color_accuracy: physics.color_accuracy.as_str().to_string(),
            external_pdgs: physics
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .collect(),
            physics_path: "processes/test/physics.json".to_string(),
            required_runtime_capabilities: vec!["symjit.application.complex-f64.v1".to_string()],
            aliases: vec![alias.clone()],
        },
        requested_id: alias.id.clone(),
        alias: Some(alias.clone()),
        public_expression: alias.expression.clone(),
        external_pdgs: alias.external_pdgs.clone(),
        external_permutation: alias.external_permutation.clone(),
        inferred_permutation: false,
        process_group: None,
    }
}

#[test]
fn final_state_alias_three_cycle_remaps_lc_metadata_and_selectors() {
    let representative_manifest = alias_test_physics("lc");
    representative_manifest.validate().unwrap();
    let representative_physics = PhysicsRuntime::new(representative_manifest.clone()).unwrap();
    let alias = three_cycle_alias();
    let alias_manifest = apply_process_permutation_metadata(
        representative_manifest.clone(),
        &stored_alias_selection(&representative_manifest, &alias),
    )
    .unwrap();

    let helicity_id_map = representative_manifest
        .helicities
        .iter()
        .zip(&alias_manifest.helicities)
        .map(|(representative, public)| (representative.id.clone(), public.id.clone()))
        .collect::<BTreeMap<_, _>>();
    let color_id_map = representative_manifest
        .color_components
        .iter()
        .zip(&alias_manifest.color_components)
        .map(|(representative, public)| (representative.id().to_string(), public.id().to_string()))
        .collect::<BTreeMap<_, _>>();
    let mut override_runtime = empty_generic_runtime();
    override_runtime.physics_reduction_override = Some(representative_manifest.reduction.clone());
    override_runtime
        .remap_physics_reduction_overrides(&helicity_id_map, &color_id_map)
        .unwrap();
    assert_eq!(
        override_runtime.physics_reduction_override,
        Some(alias_manifest.reduction.clone())
    );

    assert_eq!(alias_manifest.process_id, "cycled");
    assert_eq!(alias_manifest.process, "d d~ > a z g");
    assert_eq!(
        alias_manifest
            .external_particles
            .iter()
            .map(|particle| (
                particle.index,
                particle.label,
                particle.momentum_slot,
                particle.pdg
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 1, 0, 1),
            (1, 2, 1, -1),
            (2, 3, 2, 22),
            (3, 4, 3, 23),
            (4, 5, 4, 21),
        ]
    );
    assert_eq!(
        alias_manifest
            .external_particles
            .iter()
            .map(|particle| particle.particle.as_str())
            .collect::<Vec<_>>(),
        vec!["d", "d~", "a", "z", "g"]
    );
    assert_eq!(alias_manifest.helicities[0].values, vec![1, -1, -1, 0, 1]);
    assert_eq!(alias_manifest.helicities[0].id, "h:+1,-1,-1,+0,+1");
    assert_eq!(
        alias_manifest.helicities[1].representative_id,
        "h:+1,-1,-1,+0,+1"
    );
    let PhysicsColorComponentV1::LcFlow(first_flow) = &alias_manifest.color_components[0] else {
        panic!("expected LC flow");
    };
    let PhysicsColorComponentV1::LcFlow(second_flow) = &alias_manifest.color_components[1] else {
        panic!("expected LC flow");
    };
    assert_eq!(first_flow.word, vec![1, 4, 5, 3, 2]);
    assert_eq!(first_flow.id, "flow:1,4,5,3,2");
    assert_eq!(second_flow.id, "flow:1,3,5,4,2");
    assert_eq!(second_flow.representative_id, "flow:1,4,5,3,2");
    assert_eq!(
        alias_manifest.reduction.groups[0].physical_helicity_ids,
        vec![
            "h:+1,-1,-1,+0,+1".to_string(),
            "h:-1,+1,+1,+0,-1".to_string(),
        ]
    );
    assert_eq!(
        alias_manifest.reduction.groups[0].physical_color_ids,
        vec!["flow:1,4,5,3,2".to_string(), "flow:1,3,5,4,2".to_string(),]
    );

    let alias_physics = PhysicsRuntime::new(alias_manifest.clone()).unwrap();
    assert_eq!(
        alias_physics
            .selected_helicity_indices(Some(&BTreeSet::from(["h:-1,+1,+1,+0,-1".to_string(),])))
            .unwrap(),
        vec![1]
    );
    assert_eq!(
        alias_physics
            .selected_color_indices(Some(&BTreeSet::from(["flow:1,3,5,4,2".to_string(),])))
            .unwrap(),
        vec![1]
    );

    let representative_total = test_amplitude_runtime(vec![c64(2.0, 0.0)], None)
        .reduce_scratch_f64_resolved(1, &representative_physics, 4.0, None, None)
        .unwrap()
        .values
        .iter()
        .sum::<f64>();
    let alias_total = test_amplitude_runtime(vec![c64(2.0, 0.0)], None)
        .reduce_scratch_f64_resolved(1, &alias_physics, 4.0, None, None)
        .unwrap()
        .values
        .iter()
        .sum::<f64>();
    assert_eq!(alias_total, representative_total);

    let representative_point = (0..5)
        .map(|index| [index as f64, index as f64 + 0.1, 0.0, 0.0])
        .collect::<Vec<_>>();
    let mut alias_point = vec![[0.0; 4]; 5];
    for (representative_index, alias_index) in
        alias.external_permutation.iter().copied().enumerate()
    {
        alias_point[alias_index] = representative_point[representative_index];
    }
    let crossing_map = alias
        .external_permutation
        .iter()
        .copied()
        .enumerate()
        .map(|(target_index, source_index)| InputCrossingMapEntry {
            target_index,
            source_index,
            sign: 1.0,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        apply_input_crossing_map(vec![alias_point], 5, Some(&crossing_map)).unwrap(),
        vec![representative_point]
    );

    let mut execution = empty_generic_runtime();
    execution.external_pdg_order = alias.external_pdgs.clone();
    execution.external_count = 5;
    execution.physics = Some(Arc::new(alias_physics));
    let runtime = NativeRuntime {
        root: PathBuf::new(),
        artifact_id: "0".repeat(64),
        runtime: execution,
        execution_lane: NativeExecutionLane::Compiled,
        process: alias.expression,
        process_key: alias.id,
        representative_process_id: "representative".to_string(),
        external_permutation: alias.external_permutation,
        input_crossing_map: Some(crossing_map),
        permutation_alias_of: Some("representative".to_string()),
        final_state_permutation_alias_of: Some("representative".to_string()),
        physics_v1: native_runtime::LazyProcessPhysicsV1::loaded(alias_manifest),
        warnings_muted: false,
        warned_kinds: BTreeSet::new(),
        pending_warnings: Vec::new(),
        point_selector_scratch: PointSelectorExecutionScratch::default(),
        selector_simd_lane_width: 1,
        correlated: None,
        process_group: None,
    };
    let metadata = runtime.metadata();
    assert_eq!(metadata.external_pdg_order, vec![1, -1, 22, 23, 21]);
    assert_eq!(metadata.external_permutation, vec![0, 1, 3, 4, 2]);
    assert_eq!(
        metadata.permutation_alias_of.as_deref(),
        Some("representative")
    );
    assert_eq!(
        metadata.final_state_permutation_alias_of.as_deref(),
        Some("representative")
    );
    let exact_state: serde_json::Value =
        serde_json::from_str(&runtime.exact_runtime_state_json().unwrap()).unwrap();
    assert_eq!(exact_state["representative_process_id"], "representative");
    assert_eq!(exact_state["representative_process_key"], "p0");
    assert_eq!(
        exact_state["external_permutation"],
        serde_json::json!([0, 1, 3, 4, 2])
    );
}

fn conjugated_member_selection(physics: &ProcessPhysicsV1) -> crate::ArtifactSelection {
    let mut selection = stored_alias_selection(physics, &three_cycle_alias());
    selection.process.aliases.clear();
    selection.alias = None;
    selection.requested_id = "s_sbar_member".to_string();
    selection.public_expression = "s~ s > a z g".to_string();
    selection.external_pdgs = vec![-3, 3, 22, 23, 21];
    selection.external_permutation = vec![1, 0, 3, 4, 2];
    selection.process_group = Some(crate::ProcessGroupRelation {
        factor_numerator: 3,
        factor_denominator: 2,
        conjugated: true,
        external_particles: ["s~", "s", "a", "z", "g"].map(str::to_string).to_vec(),
    });
    selection
}

#[test]
fn process_group_member_relabels_flavours_and_reverses_lc_words() {
    let representative = alias_test_physics("lc");
    let selection = conjugated_member_selection(&representative);
    let execution = apply_process_permutation_metadata(representative, &selection).unwrap();
    let public = apply_process_group_metadata(&execution, &selection).unwrap();

    assert_eq!(public.process_id, "s_sbar_member");
    assert_eq!(public.process, "s~ s > a z g");
    assert_eq!(
        public
            .external_particles
            .iter()
            .map(|particle| (particle.particle.as_str(), particle.pdg))
            .collect::<Vec<_>>(),
        [("s~", -3), ("s", 3), ("a", 22), ("z", 23), ("g", 21)]
    );
    // Execution keeps the representative flavours in the public order.
    assert_eq!(
        execution
            .external_particles
            .iter()
            .map(|particle| particle.pdg)
            .collect::<Vec<_>>(),
        [-1, 1, 22, 23, 21]
    );
    assert_eq!(execution.helicities, public.helicities);
    for (execution_color, public_color) in execution
        .color_components
        .iter()
        .zip(&public.color_components)
    {
        let (
            PhysicsColorComponentV1::LcFlow(execution_flow),
            PhysicsColorComponentV1::LcFlow(public_flow),
        ) = (execution_color, public_color)
        else {
            panic!("expected LC flows");
        };
        let mut reversed = execution_flow.word.clone();
        reversed.reverse();
        assert_eq!(public_flow.word, reversed);
        assert_eq!(
            public_flow.id,
            format!(
                "flow:{}",
                reversed
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        );
        assert_eq!(
            charge_conjugate_color_id(&public_flow.id),
            execution_flow.id
        );
        assert_eq!(
            charge_conjugate_color_id(&execution_flow.representative_id),
            public_flow.representative_id
        );
    }
    assert_eq!(
        public.reduction.groups[0].physical_color_ids,
        execution.reduction.groups[0]
            .physical_color_ids
            .iter()
            .map(|id| charge_conjugate_color_id(id).into_owned())
            .collect::<Vec<_>>()
    );
    assert_eq!(charge_conjugate_color_id("flow:singlet"), "flow:singlet");
    assert_eq!(charge_conjugate_color_id("flow:7"), "flow:7");
    assert_eq!(
        charge_conjugate_color_id("color:contracted"),
        "color:contracted"
    );

    let contracted = alias_test_physics("full");
    let selection = conjugated_member_selection(&contracted);
    let execution = apply_process_permutation_metadata(contracted, &selection).unwrap();
    let public = apply_process_group_metadata(&execution, &selection).unwrap();
    assert_eq!(public.color_components, execution.color_components);
    assert_eq!(public.reduction, execution.reduction);
}

#[test]
fn process_group_runtime_scales_outputs_and_translates_color_ids() {
    let representative = alias_test_physics("lc");
    let selection = conjugated_member_selection(&representative);
    let execution_physics = apply_process_permutation_metadata(representative, &selection).unwrap();
    let public_physics = apply_process_group_metadata(&execution_physics, &selection).unwrap();
    let mut execution = empty_generic_runtime();
    execution.external_count = 5;
    execution.physics = Some(Arc::new(
        PhysicsRuntime::new(execution_physics.clone()).unwrap(),
    ));
    let mut runtime = NativeRuntime {
        root: PathBuf::new(),
        artifact_id: "0".repeat(64),
        runtime: execution,
        execution_lane: NativeExecutionLane::Compiled,
        process: selection.public_expression.clone(),
        process_key: selection.requested_id.clone(),
        representative_process_id: "representative".to_string(),
        external_permutation: selection.external_permutation.clone(),
        input_crossing_map: None,
        permutation_alias_of: Some("p0".to_string()),
        final_state_permutation_alias_of: None,
        process_group: None,
        physics_v1: native_runtime::LazyProcessPhysicsV1::loaded(public_physics.clone()),
        warnings_muted: false,
        warned_kinds: BTreeSet::new(),
        pending_warnings: Vec::new(),
        point_selector_scratch: PointSelectorExecutionScratch::default(),
        selector_simd_lane_width: 1,
        correlated: None,
    };

    // Without a relation, values and IDs pass through unchanged.
    let mut values = [2.0, 6.0];
    runtime.apply_process_group_factor(&mut values);
    assert_eq!(values, [2.0, 6.0]);
    let caller = ["flow:2,3,5,4,1".to_string()];
    assert_eq!(
        runtime.execution_color_ids(Some(&caller)).as_deref(),
        Some(&caller[..])
    );

    runtime.process_group = Some(Box::new(ProcessGroupRouting {
        representative_process_key: "p0".to_string(),
        factor: NativeRationalFactor {
            numerator: 3,
            denominator: 2,
        },
        conjugated: true,
    }));
    runtime.apply_process_group_factor(&mut values);
    assert_eq!(values, [3.0, 9.0]);
    let public_ids = runtime.color_ids().unwrap();
    let execution_ids = runtime
        .execution_color_ids(Some(&public_ids))
        .expect("translated IDs")
        .into_owned();
    assert_eq!(
        execution_ids,
        execution_physics
            .color_components
            .iter()
            .map(|color| color.id().to_string())
            .collect::<Vec<_>>()
    );
    let mut round_trip = execution_ids.clone();
    runtime.publish_color_ids(&mut round_trip);
    assert_eq!(round_trip, public_ids);

    let metadata = runtime.metadata();
    assert_eq!(metadata.process_group_member_of.as_deref(), Some("p0"));
    assert_eq!(
        metadata.process_group_factor,
        NativeRationalFactor {
            numerator: 3,
            denominator: 2
        }
    );
    assert!(metadata.charge_conjugated);
    let exact_state: serde_json::Value =
        serde_json::from_str(&runtime.exact_runtime_state_json().unwrap()).unwrap();
    assert_eq!(
        exact_state["process_group_factor"],
        serde_json::json!({"numerator": 3, "denominator": 2})
    );
    assert_eq!(exact_state["charge_conjugated"], true);
    assert_eq!(exact_state["process_group_member_of"], "p0");
}

#[test]
fn incoming_and_outgoing_process_permutation_remaps_all_resolved_metadata() {
    let representative = alias_test_physics("lc");
    let alias = crate::ProcessAlias {
        id: "both-sides".to_string(),
        expression: "d~ d > a z g".to_string(),
        external_pdgs: vec![-1, 1, 22, 23, 21],
        external_permutation: vec![1, 0, 3, 4, 2],
    };
    let selection = stored_alias_selection(&representative, &alias);
    let public = apply_process_permutation_metadata(representative, &selection).unwrap();

    assert_eq!(public.process_id, "both-sides");
    assert_eq!(public.process, "d~ d > a z g");
    assert_eq!(
        public
            .external_particles
            .iter()
            .map(|particle| (particle.index, particle.pdg, particle.role))
            .collect::<Vec<_>>(),
        vec![
            (0, -1, crate::ParticleRole::Initial),
            (1, 1, crate::ParticleRole::Initial),
            (2, 22, crate::ParticleRole::Final),
            (3, 23, crate::ParticleRole::Final),
            (4, 21, crate::ParticleRole::Final),
        ]
    );
    assert_eq!(public.helicities[0].values, [-1, 1, -1, 0, 1]);
    assert_eq!(public.helicities[0].id, "h:-1,+1,-1,+0,+1");
    let PhysicsColorComponentV1::LcFlow(flow) = &public.color_components[0] else {
        panic!("expected LC flow");
    };
    assert_eq!(flow.word, [2, 4, 5, 3, 1]);
    assert_eq!(flow.id, "flow:2,4,5,3,1");
    assert_eq!(
        public.reduction.groups[0].representative_helicity_id,
        "h:-1,+1,-1,+0,+1"
    );
    assert_eq!(
        public.reduction.groups[0].representative_color_id,
        "flow:2,4,5,3,1"
    );
}

#[test]
fn final_state_alias_three_cycle_preserves_contracted_color_reduction() {
    let representative_manifest = alias_test_physics("full");
    representative_manifest.validate().unwrap();
    let representative_physics = PhysicsRuntime::new(representative_manifest.clone()).unwrap();
    let alias = three_cycle_alias();
    let alias_manifest = apply_process_permutation_metadata(
        representative_manifest.clone(),
        &stored_alias_selection(&representative_manifest, &alias),
    )
    .unwrap();
    let alias_physics = PhysicsRuntime::new(alias_manifest.clone()).unwrap();

    assert_eq!(alias_manifest.color_components[0].id(), "color:contracted");
    assert_eq!(
        alias_manifest.reduction.groups[0].representative_color_id,
        "color:contracted"
    );
    assert_eq!(
        alias_physics
            .selected_helicity_indices(Some(&BTreeSet::from(["h:-1,+1,+1,+0,-1".to_string(),])))
            .unwrap(),
        vec![1]
    );
    assert_eq!(
        alias_physics
            .selected_color_indices(Some(&BTreeSet::from(["color:contracted".to_string()])))
            .unwrap(),
        vec![0]
    );

    let contraction = || {
        let groups = [RawSumGroup {
            id: 7,
            indices: vec![0],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        }];
        ColorContractionRuntime::new(
            &groups,
            vec![ColorContractionEntry {
                left_group_index: 0,
                right_group_index: 0,
                weight_re: 2.0,
                weight_im: 0.0,
                symmetry_factor: 1.0,
            }],
        )
    };
    let representative_total = test_amplitude_runtime(vec![c64(3.0, 0.0)], Some(contraction()))
        .reduce_scratch_f64_resolved(1, &representative_physics, 2.0, None, None)
        .unwrap()
        .values
        .iter()
        .sum::<f64>();
    let alias_total = test_amplitude_runtime(vec![c64(3.0, 0.0)], Some(contraction()))
        .reduce_scratch_f64_resolved(1, &alias_physics, 2.0, None, None)
        .unwrap()
        .values
        .iter()
        .sum::<f64>();
    assert_eq!(alias_total, representative_total);
}

fn empty_evaluator_group() -> EvaluatorGroup {
    EvaluatorGroup {
        evaluators: Vec::new(),
        input_len: 0,
        input_mappings: Vec::new(),
        input_mapping_spans: Vec::new(),
        output_len: 0,
        chunk_parameter_scratch_f64: Vec::new(),
        chunk_scratch_f64: Vec::new(),
        chunk_parameter_scratch_aosoa_f64: Vec::new(),
        chunk_scratch_aosoa_f64: Vec::new(),
        chunk_input_mapping_scratch: Vec::new(),
    }
}

fn test_amplitude_runtime(
    outputs: Vec<Complex<f64>>,
    color_contraction: Option<ColorContractionRuntime>,
) -> AmplitudeRuntime {
    let output_length = outputs.len();
    AmplitudeRuntime {
        output_length,
        raw_sum_weights: vec![1.0; output_length],
        raw_sum_all_sector_weights: vec![1.0; output_length],
        raw_sum_color_sector_ids: vec![None; output_length],
        raw_sum_groups: vec![RawSumGroup {
            id: 7,
            indices: (0..output_length).collect(),
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        }],
        has_coherent_groups: true,
        color_contraction,
        color_topology_replay: None,
        input_components: None,
        input_spans: Vec::new(),
        parameter_scratch_f64: Vec::new(),
        evaluator_output_scratch_f64: Vec::new(),
        output_scratch_f64: outputs,
        resolved_source_row_scratch_f64: Vec::new(),
        resolved_target_row_scratch_f64: Vec::new(),
        routed_reduction_scratch: RoutedReductionScratch::default(),
        materialized_helicity_direct_total_plans: Vec::new(),
        materialized_helicity_direct_total_plan_capacity: 0,
        materialized_helicity_direct_total_next_replacement: 0,
        materialized_helicity_direct_default_plan_by_selector_domain: Vec::new(),
        evaluator_output_order: None,
        evaluator: Some(empty_evaluator_group()),
    }
}

fn plane_native_totals(
    amplitude: &mut AmplitudeRuntime,
    outputs: &[Complex<f64>],
    batch_size: usize,
    selected_color_sector_ids: Option<&BTreeSet<i64>>,
) -> Vec<f64> {
    assert_eq!(outputs.len(), batch_size * amplitude.output_length);
    let mut workspace = crate::direct_arena::DirectArenaWorkspace::new(
        0,
        amplitude.output_length as u32,
        batch_size as u32,
    )
    .unwrap();
    workspace.begin_tile(batch_size as u32).unwrap();
    let stride = workspace.point_stride() as usize;
    {
        let (_, _, values_re, values_im) = workspace.split_slices_mut();
        values_re.fill(73.0);
        values_im.fill(-91.0);
        for point in 0..batch_size {
            for component in 0..amplitude.output_length {
                let value = outputs[point * amplitude.output_length + component];
                values_re[component * stride + point] = value.re;
                values_im[component * stride + point] = value.im;
            }
        }
    }
    let mut totals = vec![f64::from_bits(0x7ff8_0000_0000_0042); batch_size];
    {
        let (values_re, values_im) = workspace.amplitude_slices();
        let planes = crate::direct_arena::DirectAmplitudePlanes::new(
            values_re,
            values_im,
            stride as u32,
            batch_size as u32,
        )
        .unwrap();
        amplitude
            .reduce_planes_f64_into_selected_slice(planes, &mut totals, selected_color_sector_ids)
            .unwrap();
    }
    let (values_re, values_im) = workspace.amplitude_slices();
    for component in 0..amplitude.output_length {
        assert!(
            values_re[component * stride + batch_size..(component + 1) * stride]
                .iter()
                .all(|value| *value == 73.0)
        );
        assert!(
            values_im[component * stride + batch_size..(component + 1) * stride]
                .iter()
                .all(|value| *value == -91.0)
        );
    }
    totals
}

fn with_plane_native_amplitudes<T>(
    outputs: &[Complex<f64>],
    batch_size: usize,
    output_length: usize,
    reduce: impl FnOnce(crate::direct_arena::DirectAmplitudePlanes<'_>) -> T,
) -> T {
    assert_eq!(outputs.len(), batch_size * output_length);
    let mut workspace =
        crate::direct_arena::DirectArenaWorkspace::new(0, output_length as u32, batch_size as u32)
            .unwrap();
    workspace.begin_tile(batch_size as u32).unwrap();
    let stride = workspace.point_stride() as usize;
    const PAD_RE: f64 = 73.0;
    const PAD_IM: f64 = -91.0;
    {
        let (_, _, values_re, values_im) = workspace.split_slices_mut();
        values_re.fill(PAD_RE);
        values_im.fill(PAD_IM);
        for point in 0..batch_size {
            for component in 0..output_length {
                let value = outputs[point * output_length + component];
                values_re[component * stride + point] = value.re;
                values_im[component * stride + point] = value.im;
            }
        }
    }
    let reduced = {
        let (values_re, values_im) = workspace.amplitude_slices();
        let planes = crate::direct_arena::DirectAmplitudePlanes::new(
            values_re,
            values_im,
            stride as u32,
            batch_size as u32,
        )
        .unwrap();
        reduce(planes)
    };
    let (values_re, values_im) = workspace.amplitude_slices();
    for component in 0..output_length {
        assert!(
            values_re[component * stride + batch_size..(component + 1) * stride]
                .iter()
                .all(|value| *value == PAD_RE)
        );
        assert!(
            values_im[component * stride + batch_size..(component + 1) * stride]
                .iter()
                .all(|value| *value == PAD_IM)
        );
    }
    reduced
}

fn assert_resolved_values_equal(left: &ResolvedValues<f64>, right: &ResolvedValues<f64>) {
    assert_eq!(left.point_count, right.point_count);
    assert_eq!(left.helicity_indices, right.helicity_indices);
    assert_eq!(left.color_indices, right.color_indices);
    assert_eq!(left.values, right.values);
}

#[test]
fn plane_native_lc_totals_match_selected_row_major_odd_tail_and_structural_zero() {
    const POINT_COUNT: usize = 129;
    const OUTPUT_COUNT: usize = 5;
    let outputs = (0..POINT_COUNT)
        .flat_map(|point| {
            (0..OUTPUT_COUNT).map(move |output| {
                if output == 3 {
                    c64(0.0, 0.0)
                } else {
                    c64(
                        (point * 7 + output * 3) as f64 * 0.03125 - 2.0,
                        (point * 5 + output * 11) as f64 * -0.015625 + 1.0,
                    )
                }
            })
        })
        .collect::<Vec<_>>();
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    amplitude.output_length = OUTPUT_COUNT;
    amplitude.has_coherent_groups = false;
    amplitude.raw_sum_groups.clear();
    amplitude.raw_sum_weights = vec![1.0, 0.5, 2.0, 17.0, -0.25];
    amplitude.raw_sum_all_sector_weights = vec![1.25, 0.75, 3.0, 19.0, -0.5];
    amplitude.raw_sum_color_sector_ids = vec![Some(10), Some(20), Some(10), Some(30), Some(20)];

    for selected in [None, Some(BTreeSet::from([20_i64]))] {
        let mut row_major = vec![f64::NAN; POINT_COUNT];
        amplitude
            .reduce_scratch_f64_into_selected_slice(POINT_COUNT, &mut row_major, selected.as_ref())
            .unwrap();
        let plane_native =
            plane_native_totals(&mut amplitude, &outputs, POINT_COUNT, selected.as_ref());
        assert_eq!(plane_native, row_major);
    }
}

#[test]
fn row_major_reduction_dimensions_fail_closed_before_indexing() {
    let mut amplitude = test_amplitude_runtime(vec![c64(1.0, 0.0), c64(2.0, 0.0)], None);
    let error = amplitude
        .reduce_scratch_f64_into_selected_slice(usize::MAX, &mut [], None)
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
    assert!(error.to_string().contains("dimensions overflow"));
}

#[cfg(not(feature = "f64-symjit"))]
#[test]
fn warmed_plane_native_reduction_allocates_zero() {
    const POINT_COUNT: usize = 129;
    const OUTPUT_COUNT: usize = 5;
    let outputs = (0..POINT_COUNT * OUTPUT_COUNT)
        .map(|index| c64(index as f64 * 0.03125 - 3.0, index as f64 * -0.015625 + 2.0))
        .collect::<Vec<_>>();
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    amplitude.output_length = OUTPUT_COUNT;
    amplitude.has_coherent_groups = false;
    amplitude.raw_sum_groups.clear();
    amplitude.raw_sum_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_all_sector_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_color_sector_ids = vec![None; OUTPUT_COUNT];

    let mut workspace =
        crate::direct_arena::DirectArenaWorkspace::new(0, OUTPUT_COUNT as u32, POINT_COUNT as u32)
            .unwrap();
    workspace.begin_tile(POINT_COUNT as u32).unwrap();
    let stride = workspace.point_stride() as usize;
    {
        let (_, _, values_re, values_im) = workspace.split_slices_mut();
        for point in 0..POINT_COUNT {
            for component in 0..OUTPUT_COUNT {
                let value = outputs[point * OUTPUT_COUNT + component];
                values_re[component * stride + point] = value.re;
                values_im[component * stride + point] = value.im;
            }
        }
    }
    let mut totals = vec![0.0; POINT_COUNT];
    let (values_re, values_im) = workspace.amplitude_slices();
    let planes = crate::direct_arena::DirectAmplitudePlanes::new(
        values_re,
        values_im,
        stride as u32,
        POINT_COUNT as u32,
    )
    .unwrap();

    amplitude
        .reduce_planes_f64_into_selected_slice(planes, &mut totals, None)
        .unwrap();
    let (result, allocation_count, allocated_bytes) =
        super::evaluator::native_direct::tests::count_allocations(|| {
            amplitude.reduce_planes_f64_into_selected_slice(planes, &mut totals, None)
        });
    result.unwrap();
    assert_eq!(allocation_count, 0, "warmed plane reduction allocated");
    assert_eq!(allocated_bytes, 0, "warmed plane reduction allocated bytes");
}

#[cfg(not(feature = "f64-symjit"))]
#[test]
fn warmed_plane_native_repeated_contracted_reduction_allocates_zero() {
    const POINT_COUNT: usize = 129;
    const COMPONENT_COUNT: usize = 8;
    const OUTPUT_COUNT: usize = COMPONENT_COUNT * 2;
    let groups = (0..OUTPUT_COUNT)
        .map(|output_index| {
            repeated_test_group(
                10 + output_index as i64,
                output_index,
                if output_index < COMPONENT_COUNT {
                    100
                } else {
                    200
                },
            )
        })
        .collect::<Vec<_>>();
    let entries = (0..COMPONENT_COUNT)
        .map(|component| ColorContractionEntry {
            left_group_index: component,
            right_group_index: COMPONENT_COUNT + component,
            weight_re: 0.75,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        })
        .collect::<Vec<_>>();
    let outputs = (0..POINT_COUNT * OUTPUT_COUNT)
        .map(|index| {
            c64(
                (index * 13 % 101) as f64 * 0.03125 - 1.0,
                (index * 19 % 103) as f64 * -0.015625 + 0.5,
            )
        })
        .collect::<Vec<_>>();
    let mut amplitude = reduction_test_amplitude(OUTPUT_COUNT, outputs.clone(), groups, entries);
    let repeated = amplitude
        .color_contraction
        .as_ref()
        .and_then(|contraction| contraction.repeated_block.as_ref())
        .unwrap();
    assert_eq!(repeated.component_count, COMPONENT_COUNT);
    assert!(repeated.all_weights_real);

    let mut workspace =
        crate::direct_arena::DirectArenaWorkspace::new(0, OUTPUT_COUNT as u32, POINT_COUNT as u32)
            .unwrap();
    workspace.begin_tile(POINT_COUNT as u32).unwrap();
    let stride = workspace.point_stride() as usize;
    {
        let (_, _, values_re, values_im) = workspace.split_slices_mut();
        for point in 0..POINT_COUNT {
            for component in 0..OUTPUT_COUNT {
                let value = outputs[point * OUTPUT_COUNT + component];
                values_re[component * stride + point] = value.re;
                values_im[component * stride + point] = value.im;
            }
        }
    }
    let (values_re, values_im) = workspace.amplitude_slices();
    let planes = crate::direct_arena::DirectAmplitudePlanes::new(
        values_re,
        values_im,
        stride as u32,
        POINT_COUNT as u32,
    )
    .unwrap();
    let mut totals = vec![0.0; POINT_COUNT];
    amplitude
        .reduce_planes_f64_into_selected_slice(planes, &mut totals, None)
        .unwrap();
    let (result, allocation_count, allocated_bytes) =
        super::evaluator::native_direct::tests::count_allocations(|| {
            amplitude.reduce_planes_f64_into_selected_slice(planes, &mut totals, None)
        });
    result.unwrap();
    assert_eq!(
        allocation_count, 0,
        "warmed repeated NLC/full contraction allocated"
    );
    assert_eq!(
        allocated_bytes, 0,
        "warmed repeated NLC/full contraction allocated bytes"
    );
}

#[test]
fn resolved_lc_reduction_expands_symmetries_and_structural_zeros() {
    let physics = test_physics_runtime("lc");
    let mut amplitude = test_amplitude_runtime(vec![c64(2.0, 0.0)], None);

    let resolved = amplitude
        .reduce_scratch_f64_resolved(1, &physics, 4.0, None, None)
        .unwrap();

    assert_eq!(resolved.helicity_indices, vec![0, 1, 2]);
    assert_eq!(resolved.color_indices, vec![0, 1]);
    assert_eq!(resolved.values, vec![4.0, 4.0, 4.0, 4.0, 0.0, 0.0]);
    assert_eq!(resolved.values.iter().sum::<f64>(), 16.0);

    let helicities = BTreeSet::from(["hel:-+".to_string()]);
    let colors = BTreeSet::from(["flow:1".to_string()]);
    let selected = amplitude
        .reduce_scratch_f64_resolved(1, &physics, 4.0, Some(&helicities), Some(&colors))
        .unwrap();
    assert_eq!(selected.values, vec![4.0]);
}

#[test]
fn plane_native_lc_resolved_matches_full_and_selected_row_major_odd_tail() {
    const POINT_COUNT: usize = 129;
    let physics = test_physics_runtime("lc");
    let outputs = (0..POINT_COUNT)
        .map(|point| c64(point as f64 * 0.03125 - 2.0, point as f64 * -0.015625 + 1.0))
        .collect::<Vec<_>>();
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    amplitude.output_length = 1;
    amplitude.raw_sum_weights = vec![1.0];
    amplitude.raw_sum_all_sector_weights = vec![1.0];
    amplitude.raw_sum_color_sector_ids = vec![None];
    amplitude.raw_sum_groups[0].indices = vec![0];
    let selected_helicities = BTreeSet::from(["hel:-+".to_string()]);
    let selected_colors = BTreeSet::from(["flow:1".to_string()]);

    for (helicities, colors) in [
        (None, None),
        (Some(&selected_helicities), Some(&selected_colors)),
    ] {
        let row_major = amplitude
            .reduce_scratch_f64_resolved(POINT_COUNT, &physics, 1.75, helicities, colors)
            .unwrap();
        let plane_native = with_plane_native_amplitudes(&outputs, POINT_COUNT, 1, |planes| {
            amplitude
                .reduce_planes_f64_resolved(planes, &physics, 1.75, helicities, colors)
                .unwrap()
        });
        assert_resolved_values_equal(&plane_native, &row_major);
    }
}

#[test]
fn plane_native_lc_resolved_routes_preserve_row_major_order() {
    let physics = test_physics_runtime("lc");
    let outputs = vec![c64(2.0, -0.5)];
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    let replay = LcResolvedReplayEntry {
        routes: vec![
            LcResolvedReplayRoute {
                source_index: 1,
                target_index: 0,
                weight: 0.5,
            },
            LcResolvedReplayRoute {
                source_index: 0,
                target_index: 2,
                weight: 2.0,
            },
            LcResolvedReplayRoute {
                source_index: 2,
                target_index: 1,
                weight: -0.25,
            },
            LcResolvedReplayRoute {
                source_index: 3,
                target_index: 2,
                weight: 1.5,
            },
        ],
    };
    let mut row_major = [f64::NAN];
    amplitude
        .reduce_scratch_f64_routed_totals_into(
            1,
            &physics,
            4.0,
            None,
            None,
            &replay,
            6,
            3,
            &mut row_major,
        )
        .unwrap();

    let mut workspace = crate::direct_arena::DirectArenaWorkspace::new(0, 1, 1).unwrap();
    workspace.begin_tile(1).unwrap();
    let stride = workspace.point_stride();
    {
        let (_, _, values_re, values_im) = workspace.split_slices_mut();
        values_re[0] = outputs[0].re;
        values_im[0] = outputs[0].im;
    }
    let (values_re, values_im) = workspace.amplitude_slices();
    let planes =
        crate::direct_arena::DirectAmplitudePlanes::new(values_re, values_im, stride, 1).unwrap();
    let mut plane_native = [f64::NAN];
    amplitude
        .reduce_planes_f64_routed_totals_into(
            planes,
            &physics,
            4.0,
            None,
            None,
            &replay,
            6,
            3,
            &mut plane_native,
        )
        .unwrap();
    assert_eq!(plane_native, row_major);
}

#[test]
fn plane_native_lc_routed_components_match_resolved_multi_mapping_accumulation() {
    const POINT_COUNT: usize = 65;
    const TARGET_COMPONENT_COUNT: usize = 3;
    let physics = test_physics_runtime("lc");
    let mapping_outputs = [
        (0..POINT_COUNT)
            .map(|point| c64(point as f64 * 0.03125 - 1.0, point as f64 * -0.015625 + 0.5))
            .collect::<Vec<_>>(),
        (0..POINT_COUNT)
            .map(|point| {
                c64(
                    point as f64 * -0.0625 + 2.0,
                    point as f64 * 0.0078125 - 0.25,
                )
            })
            .collect::<Vec<_>>(),
    ];
    let replay_entries = [
        LcResolvedReplayEntry {
            routes: vec![
                LcResolvedReplayRoute {
                    source_index: 0,
                    target_index: 2,
                    weight: 1.25,
                },
                LcResolvedReplayRoute {
                    source_index: 3,
                    target_index: 0,
                    weight: -0.5,
                },
            ],
        },
        LcResolvedReplayEntry {
            routes: vec![
                LcResolvedReplayRoute {
                    source_index: 1,
                    target_index: 1,
                    weight: 0.75,
                },
                LcResolvedReplayRoute {
                    source_index: 4,
                    target_index: 2,
                    weight: 2.0,
                },
            ],
        },
    ];
    let mut amplitude = test_amplitude_runtime(mapping_outputs[0].clone(), None);
    amplitude.output_length = 1;
    amplitude.raw_sum_weights = vec![1.0];
    amplitude.raw_sum_all_sector_weights = vec![1.0];
    amplitude.raw_sum_color_sector_ids = vec![None];
    amplitude.raw_sum_groups[0].indices = vec![0];
    let mut expected = vec![0.0; POINT_COUNT * TARGET_COMPONENT_COUNT];
    let mut candidate = vec![0.0; POINT_COUNT * TARGET_COMPONENT_COUNT];

    for (outputs, replay_entry) in mapping_outputs.iter().zip(&replay_entries) {
        let materialized = with_plane_native_amplitudes(outputs, POINT_COUNT, 1, |planes| {
            amplitude
                .reduce_planes_f64_resolved(planes, &physics, 1.75, None, None)
                .unwrap()
        });
        accumulate_selected_lc_replay_resolved_f64(
            &mut expected,
            POINT_COUNT,
            &materialized,
            std::slice::from_ref(replay_entry),
            6,
            TARGET_COMPONENT_COUNT,
        )
        .unwrap();
        for (start, stop) in [(0usize, 32usize), (32, POINT_COUNT)] {
            with_plane_native_amplitudes(&outputs[start..stop], stop - start, 1, |planes| {
                amplitude
                    .reduce_planes_f64_routed_components_add_into(
                        planes,
                        &physics,
                        1.75,
                        None,
                        None,
                        replay_entry,
                        6,
                        TARGET_COMPONENT_COUNT,
                        POINT_COUNT,
                        start,
                        &mut candidate,
                    )
                    .unwrap()
            });
        }
    }
    assert_eq!(candidate, expected);
    let candidate_totals = candidate
        .chunks_exact(TARGET_COMPONENT_COUNT)
        .map(|components| components.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let expected_totals = expected
        .chunks_exact(TARGET_COMPONENT_COUNT)
        .map(|components| components.iter().sum::<f64>())
        .collect::<Vec<_>>();
    assert_eq!(candidate_totals, expected_totals);
}

#[cfg(not(feature = "f64-symjit"))]
#[test]
fn warmed_plane_native_selected_lc_routes_allocate_zero() {
    let physics = test_physics_runtime("lc");
    let outputs = vec![c64(2.0, -0.5)];
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    let replay = LcResolvedReplayEntry {
        routes: vec![LcResolvedReplayRoute {
            source_index: 0,
            target_index: 0,
            weight: 1.25,
        }],
    };
    let helicities = BTreeSet::from(["hel:-+".to_string()]);
    let colors = BTreeSet::from(["flow:1".to_string()]);
    let mut workspace = crate::direct_arena::DirectArenaWorkspace::new(0, 1, 1).unwrap();
    workspace.begin_tile(1).unwrap();
    let stride = workspace.point_stride();
    {
        let (_, _, values_re, values_im) = workspace.split_slices_mut();
        values_re[0] = outputs[0].re;
        values_im[0] = outputs[0].im;
    }
    let (values_re, values_im) = workspace.amplitude_slices();
    let planes =
        crate::direct_arena::DirectAmplitudePlanes::new(values_re, values_im, stride, 1).unwrap();
    let mut totals = [0.0];
    amplitude
        .reduce_planes_f64_routed_totals_into(
            planes,
            &physics,
            4.0,
            Some(&helicities),
            Some(&colors),
            &replay,
            1,
            1,
            &mut totals,
        )
        .unwrap();
    let (result, allocation_count, allocated_bytes) =
        super::evaluator::native_direct::tests::count_allocations(|| {
            amplitude.reduce_planes_f64_routed_totals_into(
                planes,
                &physics,
                4.0,
                Some(&helicities),
                Some(&colors),
                &replay,
                1,
                1,
                &mut totals,
            )
        });
    result.unwrap();
    assert_eq!(
        allocation_count, 0,
        "warmed selected LC routed reduction allocated"
    );
    assert_eq!(
        allocated_bytes, 0,
        "warmed selected LC routed reduction allocated bytes"
    );

    let mut target_components = [0.0];
    amplitude
        .reduce_planes_f64_routed_components_add_into(
            planes,
            &physics,
            4.0,
            Some(&helicities),
            Some(&colors),
            &replay,
            1,
            1,
            1,
            0,
            &mut target_components,
        )
        .unwrap();
    target_components.fill(0.0);
    let (result, allocation_count, allocated_bytes) =
        super::evaluator::native_direct::tests::count_allocations(|| {
            amplitude.reduce_planes_f64_routed_components_add_into(
                planes,
                &physics,
                4.0,
                Some(&helicities),
                Some(&colors),
                &replay,
                1,
                1,
                1,
                0,
                &mut target_components,
            )
        });
    result.unwrap();
    assert_eq!(
        allocation_count, 0,
        "warmed persistent selected LC routed reduction allocated"
    );
    assert_eq!(
        allocated_bytes, 0,
        "warmed persistent selected LC routed reduction allocated bytes"
    );
}

#[test]
fn contracted_reductions_have_one_color_component_and_sum_to_total() {
    for color_accuracy in ["nlc", "full"] {
        let physics = test_physics_runtime(color_accuracy);
        let groups = [RawSumGroup {
            id: 7,
            indices: vec![0],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        }];
        let contraction = ColorContractionRuntime::new(
            &groups,
            vec![ColorContractionEntry {
                left_group_index: 0,
                right_group_index: 0,
                weight_re: 2.0,
                weight_im: 0.0,
                symmetry_factor: 1.0,
            }],
        );
        let mut amplitude = test_amplitude_runtime(vec![c64(3.0, 0.0)], Some(contraction));

        let resolved = amplitude
            .reduce_scratch_f64_resolved(1, &physics, 2.0, None, None)
            .unwrap();

        assert_eq!(resolved.helicity_indices, vec![0, 1]);
        assert_eq!(resolved.color_indices, vec![0]);
        assert_eq!(resolved.values, vec![18.0, 18.0]);
        assert_eq!(resolved.values.iter().sum::<f64>(), 36.0);
    }
}

#[test]
fn plane_native_nlc_and_full_resolved_match_row_major_odd_tail() {
    const POINT_COUNT: usize = 129;
    let outputs = (0..POINT_COUNT)
        .map(|point| c64(point as f64 * 0.0625 - 3.0, point as f64 * -0.03125 + 0.75))
        .collect::<Vec<_>>();
    let selected_helicities = BTreeSet::from(["hel:-+".to_string()]);
    let root_factors = [Some(c64(0.75, -0.25))];

    for color_accuracy in ["nlc", "full"] {
        let physics = test_physics_runtime(color_accuracy);
        let groups = [RawSumGroup {
            id: 7,
            indices: vec![0],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        }];
        let contraction = || {
            ColorContractionRuntime::new(
                &groups,
                vec![ColorContractionEntry {
                    left_group_index: 0,
                    right_group_index: 0,
                    weight_re: 2.0,
                    weight_im: 0.0,
                    symmetry_factor: 1.0,
                }],
            )
        };
        let mut amplitude = test_amplitude_runtime(outputs.clone(), Some(contraction()));
        amplitude.output_length = 1;
        amplitude.raw_sum_weights = vec![1.0];
        amplitude.raw_sum_all_sector_weights = vec![1.0];
        amplitude.raw_sum_color_sector_ids = vec![None];
        amplitude.raw_sum_groups[0].indices = vec![0];
        let row_major = amplitude
            .reduce_scratch_f64_resolved(
                POINT_COUNT,
                &physics,
                2.0,
                Some(&selected_helicities),
                None,
            )
            .unwrap();
        let plane_native = with_plane_native_amplitudes(&outputs, POINT_COUNT, 1, |planes| {
            amplitude
                .reduce_planes_f64_resolved(planes, &physics, 2.0, Some(&selected_helicities), None)
                .unwrap()
        });
        assert_resolved_values_equal(&plane_native, &row_major);

        let row_major_materialized = amplitude
            .reduce_scratch_f64_for_materialized_helicity(
                POINT_COUNT,
                &physics,
                2.0,
                0,
                &root_factors,
                None,
            )
            .unwrap();
        let plane_native_materialized =
            with_plane_native_amplitudes(&outputs, POINT_COUNT, 1, |planes| {
                amplitude
                    .reduce_planes_f64_for_materialized_helicity(
                        planes,
                        &physics,
                        2.0,
                        0,
                        &root_factors,
                        None,
                    )
                    .unwrap()
            });
        assert_resolved_values_equal(&plane_native_materialized, &row_major_materialized);

        let initial = (0..POINT_COUNT)
            .map(|point| point as f64 * 0.125 - 4.0)
            .collect::<Vec<_>>();
        let mut row_major_totals = initial.clone();
        amplitude
            .reduce_scratch_f64_for_materialized_helicity_add_into(
                POINT_COUNT,
                &physics,
                2.0,
                0,
                &root_factors,
                None,
                &mut row_major_totals,
            )
            .unwrap();
        let mut plane_native_totals = initial;
        with_plane_native_amplitudes(&outputs, POINT_COUNT, 1, |planes| {
            amplitude
                .reduce_planes_f64_for_materialized_helicity_add_into(
                    planes,
                    &physics,
                    2.0,
                    0,
                    &root_factors,
                    None,
                    &mut plane_native_totals,
                )
                .unwrap()
        });
        assert_eq!(plane_native_totals, row_major_totals);
        assert_eq!(amplitude.materialized_helicity_direct_total_plans.len(), 1);

        let identity_root_factors = [Some(c64(1.0, 0.0))];
        let mut identity_row_major = vec![0.25; POINT_COUNT];
        amplitude
            .reduce_scratch_f64_for_materialized_helicity_add_into(
                POINT_COUNT,
                &physics,
                2.0,
                0,
                &identity_root_factors,
                None,
                &mut identity_row_major,
            )
            .unwrap();
        let mut identity_plane_native = vec![0.25; POINT_COUNT];
        with_plane_native_amplitudes(&outputs, POINT_COUNT, 1, |planes| {
            amplitude
                .reduce_planes_f64_for_materialized_helicity_add_into(
                    planes,
                    &physics,
                    2.0,
                    0,
                    &identity_root_factors,
                    None,
                    &mut identity_plane_native,
                )
                .unwrap()
        });
        assert_eq!(
            identity_plane_native
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            identity_row_major
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            amplitude.materialized_helicity_direct_total_plans[1].groups[0].identity_output_index,
            Some(0)
        );
    }
}

#[test]
fn materialized_contracted_plan_keeps_only_selected_helicity_entries_and_caches_domain() {
    let mut physics_manifest = test_physics_runtime("full").manifest.clone();
    physics_manifest.reduction.groups = vec![
        crate::ReductionGroup {
            id: "group:7".to_string(),
            representative_helicity_id: "hel:+-".to_string(),
            physical_helicity_ids: vec!["hel:+-".to_string()],
            representative_color_id: "contracted".to_string(),
            physical_color_ids: vec!["contracted".to_string()],
        },
        crate::ReductionGroup {
            id: "group:8".to_string(),
            representative_helicity_id: "hel:-+".to_string(),
            physical_helicity_ids: vec!["hel:-+".to_string()],
            representative_color_id: "contracted".to_string(),
            physical_color_ids: vec!["contracted".to_string()],
        },
    ];
    let physics = PhysicsRuntime::new(physics_manifest).unwrap();
    let groups = vec![
        RawSumGroup {
            id: 7,
            indices: vec![0],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        },
        RawSumGroup {
            id: 8,
            indices: vec![1],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        },
    ];
    let contraction = ColorContractionRuntime::new(
        &groups,
        vec![
            ColorContractionEntry {
                left_group_index: 0,
                right_group_index: 0,
                weight_re: 2.0,
                weight_im: 0.0,
                symmetry_factor: 1.0,
            },
            ColorContractionEntry {
                left_group_index: 1,
                right_group_index: 1,
                weight_re: 2.0,
                weight_im: 0.0,
                symmetry_factor: 1.0,
            },
        ],
    );
    let mut amplitude = test_amplitude_runtime(
        vec![c64(3.0, 0.0), c64(f64::NAN, f64::NAN)],
        Some(contraction),
    );
    amplitude.output_length = 2;
    amplitude.raw_sum_weights = vec![1.0; 2];
    amplitude.raw_sum_all_sector_weights = vec![1.0; 2];
    amplitude.raw_sum_color_sector_ids = vec![None; 2];
    amplitude.raw_sum_groups = groups;
    let root_factors = [Some(c64(1.0, 0.0)), None];

    let plan_index = amplitude
        .bind_default_materialized_helicity_direct_total_plan(4, &physics, 0, &root_factors, None)
        .unwrap();
    let MaterializedHelicityDirectTotalReduction::Contracted { entries } =
        &amplitude.materialized_helicity_direct_total_plans[plan_index].reduction
    else {
        panic!("FullColour materialized plan must use contracted reduction");
    };
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].left_group_index, 0);
    assert_eq!(entries[0].right_group_index, 0);

    assert_eq!(
        amplitude
            .bind_default_materialized_helicity_direct_total_plan(
                4,
                &physics,
                0,
                &root_factors,
                None,
            )
            .unwrap(),
        plan_index
    );
    assert_eq!(amplitude.materialized_helicity_direct_total_plans.len(), 1);
    assert_eq!(
        amplitude.materialized_helicity_direct_default_plan_by_selector_domain[4]
            .expect("selector-domain plan cache")
            .plan_index,
        plan_index
    );
}

#[test]
fn plane_native_materialized_helicity_resolved_and_add_into_match_lc_odd_tail() {
    const POINT_COUNT: usize = 129;
    const OUTPUT_COUNT: usize = 2;
    let physics = test_physics_runtime("lc");
    let outputs = (0..POINT_COUNT)
        .flat_map(|point| {
            (0..OUTPUT_COUNT).map(move |output| {
                c64(
                    (point * 7 + output * 3) as f64 * 0.03125 - 1.5,
                    (point * 5 + output * 11) as f64 * -0.015625 + 0.25,
                )
            })
        })
        .collect::<Vec<_>>();
    let root_factors = [Some(c64(0.75, -0.25)), Some(c64(-0.5, 0.125))];
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    amplitude.output_length = OUTPUT_COUNT;
    amplitude.raw_sum_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_all_sector_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_color_sector_ids = vec![None; OUTPUT_COUNT];
    amplitude.raw_sum_groups[0].indices = vec![0, 1];
    let selected_colors = BTreeSet::from(["flow:1".to_string()]);

    for colors in [None, Some(&selected_colors)] {
        let row_major = amplitude
            .reduce_scratch_f64_for_materialized_helicity(
                POINT_COUNT,
                &physics,
                1.25,
                0,
                &root_factors,
                colors,
            )
            .unwrap();
        let plane_native =
            with_plane_native_amplitudes(&outputs, POINT_COUNT, OUTPUT_COUNT, |planes| {
                amplitude
                    .reduce_planes_f64_for_materialized_helicity(
                        planes,
                        &physics,
                        1.25,
                        0,
                        &root_factors,
                        colors,
                    )
                    .unwrap()
            });
        assert_resolved_values_equal(&plane_native, &row_major);

        let initial = (0..POINT_COUNT)
            .map(|point| point as f64 * 0.125 - 4.0)
            .collect::<Vec<_>>();
        let mut row_major_totals = initial.clone();
        amplitude
            .reduce_scratch_f64_for_materialized_helicity_add_into(
                POINT_COUNT,
                &physics,
                1.25,
                0,
                &root_factors,
                colors,
                &mut row_major_totals,
            )
            .unwrap();
        let mut plane_native_totals = initial;
        with_plane_native_amplitudes(&outputs, POINT_COUNT, OUTPUT_COUNT, |planes| {
            amplitude
                .reduce_planes_f64_for_materialized_helicity_add_into(
                    planes,
                    &physics,
                    1.25,
                    0,
                    &root_factors,
                    colors,
                    &mut plane_native_totals,
                )
                .unwrap()
        });
        assert_eq!(plane_native_totals, row_major_totals);
        let expected_plan_index = usize::from(colors.is_some());
        assert_eq!(
            amplitude
                .bind_materialized_helicity_direct_total_plan(&physics, 0, &root_factors, colors)
                .unwrap(),
            expected_plan_index
        );
        assert_eq!(
            amplitude.materialized_helicity_direct_total_plans.len(),
            expected_plan_index + 1,
            "equivalent materialized-helicity reductions must reuse their cold binding"
        );
    }
}

#[test]
fn materialized_helicity_direct_total_plan_uses_actual_physics_recipient() {
    const POINT_COUNT: usize = 17;
    const OUTPUT_COUNT: usize = 2;
    let representative_physics = test_physics_runtime("lc");
    let mut weighted_manifest = representative_physics.manifest.clone();
    let crate::ColorComponent::LcFlow(weighted_flow) = &mut weighted_manifest.color_components[1]
    else {
        panic!("LC test physics must contain physical flows");
    };
    weighted_flow.coefficient = 3.0;
    let parent_physics =
        PhysicsRuntime::new(weighted_manifest).expect("valid parent-closure test physics");
    let outputs = (0..POINT_COUNT)
        .flat_map(|point| {
            (0..OUTPUT_COUNT).map(move |output| {
                c64(
                    (point * 5 + output * 7) as f64 * 0.03125 - 0.75,
                    (point * 11 + output * 3) as f64 * -0.015625 + 0.5,
                )
            })
        })
        .collect::<Vec<_>>();
    let root_factors = [Some(c64(0.75, -0.25)), Some(c64(-0.5, 0.125))];
    let selected_colors = BTreeSet::from(["flow:0".to_string()]);
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    amplitude.output_length = OUTPUT_COUNT;
    amplitude.raw_sum_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_all_sector_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_color_sector_ids = vec![None; OUTPUT_COUNT];
    amplitude.raw_sum_groups[0].indices = vec![0, 1];

    for (expected_plan_index, physics) in [&representative_physics, &parent_physics]
        .into_iter()
        .enumerate()
    {
        let mut expected = vec![0.0; POINT_COUNT];
        amplitude
            .reduce_scratch_f64_for_materialized_helicity_add_into(
                POINT_COUNT,
                physics,
                1.25,
                0,
                &root_factors,
                Some(&selected_colors),
                &mut expected,
            )
            .unwrap();
        let mut candidate = vec![0.0; POINT_COUNT];
        with_plane_native_amplitudes(&outputs, POINT_COUNT, OUTPUT_COUNT, |planes| {
            amplitude
                .reduce_planes_f64_for_materialized_helicity_add_into(
                    planes,
                    physics,
                    1.25,
                    0,
                    &root_factors,
                    Some(&selected_colors),
                    &mut candidate,
                )
                .unwrap()
        });
        assert_eq!(candidate, expected);
        assert_eq!(
            amplitude
                .bind_materialized_helicity_direct_total_plan(
                    physics,
                    0,
                    &root_factors,
                    Some(&selected_colors)
                )
                .unwrap(),
            expected_plan_index
        );
    }
    assert_eq!(amplitude.materialized_helicity_direct_total_plans.len(), 2);
}

#[test]
fn materialized_helicity_direct_total_identity_singleton_is_bitwise_exact() {
    let physics = test_physics_runtime("lc");
    let outputs = vec![
        c64(2.0, -3.0),
        c64(-0.0, 0.0),
        c64(0.0, -0.0),
        c64(-1.25, 4.5),
    ];
    let root_factors = [Some(c64(1.0, 0.0))];
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    amplitude.output_length = 1;
    amplitude.raw_sum_weights = vec![1.0];
    amplitude.raw_sum_all_sector_weights = vec![1.0];
    amplitude.raw_sum_color_sector_ids = vec![None];
    amplitude.raw_sum_groups[0].indices = vec![0];

    let mut expected = vec![-0.0, 0.25, -2.0, 7.0];
    amplitude
        .reduce_scratch_f64_for_materialized_helicity_add_into(
            outputs.len(),
            &physics,
            1.25,
            0,
            &root_factors,
            None,
            &mut expected,
        )
        .unwrap();
    let mut candidate = vec![-0.0, 0.25, -2.0, 7.0];
    with_plane_native_amplitudes(&outputs, outputs.len(), 1, |planes| {
        amplitude
            .reduce_planes_f64_for_materialized_helicity_add_into(
                planes,
                &physics,
                1.25,
                0,
                &root_factors,
                None,
                &mut candidate,
            )
            .unwrap()
    });
    assert_eq!(
        candidate
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        amplitude.materialized_helicity_direct_total_plans[0].groups[0].identity_output_index,
        Some(0)
    );
}

#[test]
fn materialized_helicity_direct_total_plan_cache_is_artifact_bounded() {
    let physics = test_physics_runtime("lc");
    let mut amplitude = test_amplitude_runtime(vec![c64(1.0, 0.0)], None);
    for index in 0..24 {
        amplitude
            .bind_materialized_helicity_direct_total_plan(
                &physics,
                0,
                &[Some(c64(index as f64 + 1.0, 0.0))],
                None,
            )
            .unwrap();
    }
    let expected_capacity = physics.manifest.helicities.len().clamp(16, 512);
    assert_eq!(
        amplitude.materialized_helicity_direct_total_plan_capacity,
        expected_capacity
    );
    assert_eq!(
        amplitude.materialized_helicity_direct_total_plans.len(),
        expected_capacity
    );
}

#[test]
fn plane_native_materialized_helicity_routed_components_match_resolved_accumulation() {
    const POINT_COUNT: usize = 65;
    const OUTPUT_COUNT: usize = 2;
    const TARGET_COMPONENT_COUNT: usize = 3;
    let physics = test_physics_runtime("lc");
    let outputs = (0..POINT_COUNT)
        .flat_map(|point| {
            (0..OUTPUT_COUNT).map(move |output| {
                c64(
                    (point * 7 + output * 3) as f64 * 0.03125 - 1.5,
                    (point * 5 + output * 11) as f64 * -0.015625 + 0.25,
                )
            })
        })
        .collect::<Vec<_>>();
    let root_factors = [Some(c64(0.75, -0.25)), Some(c64(-0.5, 0.125))];
    let replay = LcResolvedReplayEntry {
        routes: vec![
            LcResolvedReplayRoute {
                source_index: 0,
                target_index: 2,
                weight: 1.5,
            },
            LcResolvedReplayRoute {
                source_index: 1,
                target_index: 0,
                weight: -0.25,
            },
        ],
    };
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    amplitude.output_length = OUTPUT_COUNT;
    amplitude.raw_sum_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_all_sector_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_color_sector_ids = vec![None; OUTPUT_COUNT];
    amplitude.raw_sum_groups[0].indices = vec![0, 1];
    let materialized =
        with_plane_native_amplitudes(&outputs, POINT_COUNT, OUTPUT_COUNT, |planes| {
            amplitude
                .reduce_planes_f64_for_materialized_helicity(
                    planes,
                    &physics,
                    1.25,
                    0,
                    &root_factors,
                    None,
                )
                .unwrap()
        });
    let mut expected = vec![0.0; POINT_COUNT * TARGET_COMPONENT_COUNT];
    accumulate_selected_lc_replay_resolved_f64(
        &mut expected,
        POINT_COUNT,
        &materialized,
        std::slice::from_ref(&replay),
        2,
        TARGET_COMPONENT_COUNT,
    )
    .unwrap();
    let mut candidate = vec![0.0; POINT_COUNT * TARGET_COMPONENT_COUNT];
    for (start, stop) in [(0usize, 32usize), (32, POINT_COUNT)] {
        with_plane_native_amplitudes(
            &outputs[start * OUTPUT_COUNT..stop * OUTPUT_COUNT],
            stop - start,
            OUTPUT_COUNT,
            |planes| {
                amplitude
                    .reduce_planes_f64_for_materialized_helicity_routed_components_add_into(
                        planes,
                        &physics,
                        1.25,
                        0,
                        &root_factors,
                        None,
                        &replay,
                        2,
                        TARGET_COMPONENT_COUNT,
                        POINT_COUNT,
                        start,
                        &mut candidate,
                    )
                    .unwrap()
            },
        );
    }
    assert_eq!(candidate, expected);

    // Alternate proper subsets with implicit and explicit complete axes.
    // Sparse routing must neither read nor reset a previous dense position
    // table, and it must preserve the resolved reducer's accumulation order.
    for selected_colors in [
        Some(BTreeSet::from(["flow:1".to_string()])),
        None,
        Some(BTreeSet::from(["flow:0".to_string()])),
        Some(BTreeSet::from(["flow:0".to_string(), "flow:1".to_string()])),
        Some(BTreeSet::from(["flow:1".to_string()])),
    ] {
        let selected_colors = selected_colors.as_ref();
        let color_indices = physics.selected_color_indices(selected_colors).unwrap();
        let selected_replay = LcResolvedReplayEntry {
            routes: color_indices
                .iter()
                .enumerate()
                .map(|(position, &color)| LcResolvedReplayRoute {
                    source_index: position,
                    target_index: (color + 1) % TARGET_COMPONENT_COUNT,
                    weight: if position == 0 { 1.5 } else { -0.25 },
                })
                .collect(),
        };
        let materialized =
            with_plane_native_amplitudes(&outputs, POINT_COUNT, OUTPUT_COUNT, |planes| {
                amplitude
                    .reduce_planes_f64_for_materialized_helicity(
                        planes,
                        &physics,
                        1.25,
                        0,
                        &root_factors,
                        selected_colors,
                    )
                    .unwrap()
            });
        let mut expected = vec![0.0; POINT_COUNT * TARGET_COMPONENT_COUNT];
        accumulate_selected_lc_replay_resolved_f64(
            &mut expected,
            POINT_COUNT,
            &materialized,
            std::slice::from_ref(&selected_replay),
            color_indices.len(),
            TARGET_COMPONENT_COUNT,
        )
        .unwrap();
        amplitude.routed_reduction_scratch.color_positions = vec![Some(usize::MAX); 7];
        let mut candidate = vec![0.0; POINT_COUNT * TARGET_COMPONENT_COUNT];
        for (start, stop) in [(0usize, 32usize), (32, POINT_COUNT)] {
            with_plane_native_amplitudes(
                &outputs[start * OUTPUT_COUNT..stop * OUTPUT_COUNT],
                stop - start,
                OUTPUT_COUNT,
                |planes| {
                    amplitude
                        .reduce_planes_f64_for_materialized_helicity_routed_components_add_into(
                            planes,
                            &physics,
                            1.25,
                            0,
                            &root_factors,
                            selected_colors,
                            &selected_replay,
                            color_indices.len(),
                            TARGET_COMPONENT_COUNT,
                            POINT_COUNT,
                            start,
                            &mut candidate,
                        )
                        .unwrap()
                },
            );
        }
        assert_eq!(candidate, expected);
        if color_indices.len() < physics.manifest.color_components.len() {
            assert_eq!(
                amplitude.routed_reduction_scratch.color_positions,
                vec![Some(usize::MAX); 7],
                "restricted colours must not touch the complete-axis position table"
            );
        } else {
            assert_eq!(
                amplitude.routed_reduction_scratch.color_positions,
                vec![Some(0), Some(1)]
            );
        }
    }
}

#[cfg(not(feature = "f64-symjit"))]
#[test]
fn warmed_plane_native_materialized_helicity_add_into_allocates_zero() {
    const POINT_COUNT: usize = 129;
    const OUTPUT_COUNT: usize = 2;
    let physics = test_physics_runtime("lc");
    let outputs = (0..POINT_COUNT)
        .flat_map(|point| {
            (0..OUTPUT_COUNT).map(move |output| {
                c64(
                    (point * 7 + output * 3) as f64 * 0.03125 - 1.5,
                    (point * 5 + output * 11) as f64 * -0.015625 + 0.25,
                )
            })
        })
        .collect::<Vec<_>>();
    let root_factors = [Some(c64(0.75, -0.25)), Some(c64(-0.5, 0.125))];
    let selected_colors = BTreeSet::from(["flow:1".to_string()]);
    let mut amplitude = test_amplitude_runtime(outputs.clone(), None);
    amplitude.output_length = OUTPUT_COUNT;
    amplitude.raw_sum_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_all_sector_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_color_sector_ids = vec![None; OUTPUT_COUNT];
    amplitude.raw_sum_groups[0].indices = vec![0, 1];

    let mut workspace =
        crate::direct_arena::DirectArenaWorkspace::new(0, OUTPUT_COUNT as u32, POINT_COUNT as u32)
            .unwrap();
    workspace.begin_tile(POINT_COUNT as u32).unwrap();
    let stride = workspace.point_stride() as usize;
    {
        let (_, _, values_re, values_im) = workspace.split_slices_mut();
        for point in 0..POINT_COUNT {
            for component in 0..OUTPUT_COUNT {
                let value = outputs[point * OUTPUT_COUNT + component];
                values_re[component * stride + point] = value.re;
                values_im[component * stride + point] = value.im;
            }
        }
    }
    let (values_re, values_im) = workspace.amplitude_slices();
    let planes = crate::direct_arena::DirectAmplitudePlanes::new(
        values_re,
        values_im,
        stride as u32,
        POINT_COUNT as u32,
    )
    .unwrap();
    let mut totals = vec![0.0; POINT_COUNT];
    amplitude
        .reduce_planes_f64_for_materialized_helicity_add_into(
            planes,
            &physics,
            1.25,
            0,
            &root_factors,
            Some(&selected_colors),
            &mut totals,
        )
        .unwrap();
    totals.fill(0.0);
    let (result, allocation_count, allocated_bytes) =
        super::evaluator::native_direct::tests::count_allocations(|| {
            amplitude.reduce_planes_f64_for_materialized_helicity_add_into(
                planes,
                &physics,
                1.25,
                0,
                &root_factors,
                Some(&selected_colors),
                &mut totals,
            )
        });
    result.unwrap();
    assert_eq!(
        allocation_count, 0,
        "warmed materialized-helicity plane accumulator allocated"
    );
    assert_eq!(
        allocated_bytes, 0,
        "warmed materialized-helicity plane accumulator allocated bytes"
    );

    let replay = LcResolvedReplayEntry {
        routes: vec![LcResolvedReplayRoute {
            source_index: 0,
            target_index: 0,
            weight: 1.0,
        }],
    };
    let mut target_components = vec![0.0; POINT_COUNT];
    amplitude
        .reduce_planes_f64_for_materialized_helicity_routed_components_add_into(
            planes,
            &physics,
            1.25,
            0,
            &root_factors,
            Some(&selected_colors),
            &replay,
            1,
            1,
            POINT_COUNT,
            0,
            &mut target_components,
        )
        .unwrap();
    target_components.fill(0.0);
    let (result, allocation_count, allocated_bytes) =
        super::evaluator::native_direct::tests::count_allocations(|| {
            amplitude.reduce_planes_f64_for_materialized_helicity_routed_components_add_into(
                planes,
                &physics,
                1.25,
                0,
                &root_factors,
                Some(&selected_colors),
                &replay,
                1,
                1,
                POINT_COUNT,
                0,
                &mut target_components,
            )
        });
    result.unwrap();
    assert_eq!(
        allocation_count, 0,
        "warmed persistent materialized-helicity route allocated"
    );
    assert_eq!(
        allocated_bytes, 0,
        "warmed persistent materialized-helicity route allocated bytes"
    );
}

fn repeated_test_group(id: i64, output_index: usize, sector_id: i64) -> RawSumGroup {
    RawSumGroup {
        id,
        indices: vec![output_index],
        weight: 1.0,
        all_sector_weight: 1.0,
        sector_ids: vec![sector_id],
    }
}

fn legacy_color_contraction_totals(
    amplitudes: &[Complex<f64>],
    output_length: usize,
    groups: &[RawSumGroup],
    entries: &[ColorContractionEntry],
) -> Vec<f64> {
    amplitudes
        .chunks_exact(output_length)
        .map(|row| {
            let group_values = groups
                .iter()
                .map(|group| {
                    group
                        .indices
                        .iter()
                        .fold(Complex::new(0.0, 0.0), |total, index| total + row[*index])
                })
                .collect::<Vec<_>>();
            entries.iter().fold(0.0, |total, entry| {
                let product = group_values[entry.left_group_index]
                    * group_values[entry.right_group_index].conj();
                total
                    + entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im)
            })
        })
        .collect()
}

fn reduction_test_amplitude(
    output_length: usize,
    outputs: Vec<Complex<f64>>,
    groups: Vec<RawSumGroup>,
    entries: Vec<ColorContractionEntry>,
) -> AmplitudeRuntime {
    let contraction = ColorContractionRuntime::new(&groups, entries);
    AmplitudeRuntime {
        output_length,
        raw_sum_weights: vec![1.0; output_length],
        raw_sum_all_sector_weights: vec![1.0; output_length],
        raw_sum_color_sector_ids: vec![None; output_length],
        raw_sum_groups: groups,
        has_coherent_groups: true,
        color_contraction: Some(contraction),
        color_topology_replay: None,
        input_components: None,
        input_spans: Vec::new(),
        parameter_scratch_f64: Vec::new(),
        evaluator_output_scratch_f64: Vec::new(),
        output_scratch_f64: outputs,
        resolved_source_row_scratch_f64: Vec::new(),
        resolved_target_row_scratch_f64: Vec::new(),
        routed_reduction_scratch: RoutedReductionScratch::default(),
        materialized_helicity_direct_total_plans: Vec::new(),
        materialized_helicity_direct_total_plan_capacity: 0,
        materialized_helicity_direct_total_next_replacement: 0,
        materialized_helicity_direct_default_plan_by_selector_domain: Vec::new(),
        evaluator_output_order: None,
        evaluator: Some(empty_evaluator_group()),
    }
}

#[test]
fn compiled_symmetric_group_reducer_binds_group_ids_and_multi_root_odd_tail() {
    const POINT_COUNT: usize = 127;
    const PREPARED_CAPACITY: usize = 136;
    const GROUP_COUNT: usize = 13;
    const OUTPUT_COUNT: usize = 15;
    let destination_ids = vec![7, 0, 12, 3, 9, 1, 11, 4, 10, 2, 8, 5, 6];
    let vector_order = [12, 4, 1, 9, 0, 7, 3, 11, 2, 10, 5, 8, 6];
    let mut sector_by_id = [0usize; GROUP_COUNT];
    for (sector, destination) in destination_ids.iter().copied().enumerate() {
        sector_by_id[destination as usize] = sector;
    }
    let mut next_output = 0usize;
    let groups = vector_order
        .into_iter()
        .map(|id| {
            let root_count = if id == 7 { 3 } else { 1 };
            let indices = (next_output..next_output + root_count).collect::<Vec<_>>();
            next_output += root_count;
            RawSumGroup {
                id,
                indices,
                weight: 1.0,
                all_sector_weight: 1.0,
                sector_ids: vec![sector_by_id[id as usize] as i64],
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(next_output, OUTPUT_COUNT);
    let outputs = (0..POINT_COUNT * OUTPUT_COUNT)
        .map(|index| {
            c64(
                (index * 29 % 211) as f64 * 0.03125 - 2.5,
                (index * 43 % 197) as f64 * -0.015625 + 1.25,
            )
        })
        .collect::<Vec<_>>();
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: GROUP_COUNT,
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: None,
    };

    let duplicated_destination =
        crate::recurrence::RecurrenceColorContraction::symmetric_group_s3_for_runtime_test(
            vec![0; GROUP_COUNT],
            GROUP_COUNT as u32,
        );
    let duplicate_error = match build_compiled_symmetric_group_color_contraction_runtime(
        Some(&manifest),
        &groups,
        duplicated_destination,
    ) {
        Ok(_) => panic!("duplicate coherent destination unexpectedly bound"),
        Err(error) => error,
    };
    assert_eq!(duplicate_error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(duplicate_error.message().contains("symmetric-group FFT"));

    let sparse_groups = groups
        .iter()
        .map(|group| RawSumGroup {
            id: group.id,
            indices: group.indices.clone(),
            weight: group.weight,
            all_sector_weight: group.all_sector_weight,
            sector_ids: group
                .sector_ids
                .iter()
                .map(|sector| 2 * sector + 1)
                .collect(),
        })
        .collect::<Vec<_>>();
    let sparse_plan =
        crate::recurrence::RecurrenceColorContraction::symmetric_group_s3_for_runtime_test(
            destination_ids.clone(),
            GROUP_COUNT as u32,
        )
        .with_sparse_sector_domain_for_runtime_test();
    let sparse_contraction = build_compiled_symmetric_group_color_contraction_runtime(
        Some(&manifest),
        &sparse_groups,
        sparse_plan,
    )
    .expect("inactive physical sectors must not block a compiled FFT reducer");
    assert_eq!(
        sparse_contraction
            .symmetric_group
            .as_ref()
            .unwrap()
            .plan
            .active_sector_count(),
        GROUP_COUNT,
    );

    let plan = crate::recurrence::RecurrenceColorContraction::symmetric_group_s3_for_runtime_test(
        destination_ids.clone(),
        GROUP_COUNT as u32,
    );
    let reducer = match plan.runtime_reducer().unwrap() {
        crate::recurrence::RuntimeColorContractionReducer::SymmetricGroupFourier(reducer) => {
            reducer
        }
        _ => panic!("expected symmetric-group reducer"),
    };
    let mut expected_workspace = reducer.workspace(PREPARED_CAPACITY).unwrap();
    reducer
        .reduce_lanes(
            &mut expected_workspace,
            POINT_COUNT,
            |local_group, point| {
                let group_id = i64::from(destination_ids[local_group]);
                let group = groups.iter().find(|group| group.id == group_id).unwrap();
                let value = group.indices.iter().fold(c64(0.0, 0.0), |total, index| {
                    total + outputs[point * OUTPUT_COUNT + *index]
                });
                Ok((value.re, value.im))
            },
        )
        .unwrap();
    let expected = expected_workspace.reduced(POINT_COUNT).unwrap().to_vec();

    let contraction =
        build_compiled_symmetric_group_color_contraction_runtime(Some(&manifest), &groups, plan)
            .unwrap();
    assert_eq!(
        contraction
            .symmetric_group
            .as_ref()
            .unwrap()
            .ordered_group_indices,
        destination_ids
            .iter()
            .map(|destination| {
                groups
                    .iter()
                    .position(|group| group.id == i64::from(*destination))
                    .unwrap()
            })
            .collect::<Vec<_>>(),
    );
    let mut amplitude = reduction_test_amplitude(OUTPUT_COUNT, outputs.clone(), groups, Vec::new());
    amplitude.color_contraction = Some(contraction);
    amplitude
        .prepare_compiled_symmetric_group_workspace(PREPARED_CAPACITY)
        .unwrap();
    assert_eq!(
        amplitude
            .color_contraction
            .as_ref()
            .and_then(|contraction| contraction.symmetric_group.as_ref())
            .and_then(|symmetric_group| symmetric_group.workspace.as_ref())
            .unwrap()
            .lane_capacity(),
        PREPARED_CAPACITY,
    );

    let actual = plane_native_totals(&mut amplitude, &outputs, POINT_COUNT, None);
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() <= 1.0e-11 * expected.abs().max(1.0),
            "compiled symmetric-group reduction {actual} differs from reference {expected}",
        );
    }

    #[cfg(not(feature = "f64-symjit"))]
    {
        let mut workspace = crate::direct_arena::DirectArenaWorkspace::new(
            0,
            OUTPUT_COUNT as u32,
            POINT_COUNT as u32,
        )
        .unwrap();
        workspace.begin_tile(POINT_COUNT as u32).unwrap();
        let stride = workspace.point_stride() as usize;
        {
            let (_, _, values_re, values_im) = workspace.split_slices_mut();
            for point in 0..POINT_COUNT {
                for component in 0..OUTPUT_COUNT {
                    let value = outputs[point * OUTPUT_COUNT + component];
                    values_re[component * stride + point] = value.re;
                    values_im[component * stride + point] = value.im;
                }
            }
        }
        let (values_re, values_im) = workspace.amplitude_slices();
        let planes = crate::direct_arena::DirectAmplitudePlanes::new(
            values_re,
            values_im,
            stride as u32,
            POINT_COUNT as u32,
        )
        .unwrap();
        let mut totals = vec![0.0; POINT_COUNT];
        amplitude
            .reduce_planes_f64_into_selected_slice(planes, &mut totals, None)
            .unwrap();
        let (result, allocation_count, allocated_bytes) =
            super::evaluator::native_direct::tests::count_allocations(|| {
                amplitude.reduce_planes_f64_into_selected_slice(planes, &mut totals, None)
            });
        result.unwrap();
        assert_eq!(
            allocation_count, 0,
            "warmed symmetric-group reduction allocated"
        );
        assert_eq!(
            allocated_bytes, 0,
            "warmed symmetric-group reduction allocated bytes",
        );
    }
}

#[test]
fn repeated_real_color_blocks_match_legacy_reduction_for_permuted_outputs() {
    let groups = vec![
        repeated_test_group(10, 2, 100),
        repeated_test_group(11, 5, 100),
        repeated_test_group(12, 0, 200),
        repeated_test_group(13, 3, 200),
        repeated_test_group(14, 1, 300),
        repeated_test_group(15, 4, 300),
    ];
    // Deliberately interleave the two disconnected components and do not
    // present their left indices monotonically. Runtime canonicalization may
    // change floating-point association, but it must preserve the contraction.
    let entries = vec![
        ColorContractionEntry {
            left_group_index: 3,
            right_group_index: 5,
            weight_re: 0.5,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 0,
            weight_re: 1.25,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 3,
            weight_re: -0.75,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 4,
            right_group_index: 4,
            weight_re: 2.0,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
        ColorContractionEntry {
            left_group_index: 2,
            right_group_index: 4,
            weight_re: 0.5,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 1,
            weight_re: 1.25,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 2,
            weight_re: -0.75,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 5,
            right_group_index: 5,
            weight_re: 2.0,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
    ];
    let outputs = vec![
        c64(0.5, -1.0),
        c64(1.5, 0.25),
        c64(-0.75, 2.0),
        c64(0.125, -0.5),
        c64(2.25, 1.0),
        c64(-1.25, -0.75),
        c64(1.0, 0.5),
        c64(-0.25, 1.25),
        c64(0.75, -1.5),
        c64(2.0, 0.125),
        c64(-1.0, 0.75),
        c64(0.25, -2.0),
    ];
    let expected = legacy_color_contraction_totals(&outputs, 6, &groups, &entries);
    let mut amplitude = reduction_test_amplitude(6, outputs.clone(), groups, entries);
    let repeated = amplitude
        .color_contraction
        .as_ref()
        .and_then(|contraction| contraction.repeated_block.as_ref())
        .expect("two identical color components should be canonicalized");
    assert_eq!(repeated.component_count, 2);
    assert_eq!(repeated.entries.len(), 4);
    assert_eq!(
        repeated.singleton_output_indices.as_deref(),
        Some([2, 5, 0, 3, 1, 4].as_slice())
    );
    assert!(repeated.all_weights_real);

    let mut actual = vec![0.0; 2];
    amplitude
        .reduce_scratch_f64_into_selected_slice(2, &mut actual, None)
        .unwrap();
    let plane_native = plane_native_totals(&mut amplitude, &outputs, 2, None);
    assert_eq!(plane_native, actual);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "repeated-block reduction {actual} differs from legacy {expected}"
        );
    }
}

#[test]
fn repeated_color_block_requires_identical_component_coefficients() {
    let groups = vec![
        repeated_test_group(10, 0, 100),
        repeated_test_group(11, 1, 100),
        repeated_test_group(12, 2, 200),
        repeated_test_group(13, 3, 200),
    ];
    let entries = vec![
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 2,
            weight_re: 1.0,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 3,
            weight_re: 1.0 + f64::EPSILON,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
    ];
    let contraction = ColorContractionRuntime::new(&groups, entries);
    assert!(contraction.repeated_block.is_none());
}

#[test]
fn plane_native_expanded_color_contraction_matches_row_major_odd_tail() {
    const POINT_COUNT: usize = 127;
    let groups = vec![
        RawSumGroup {
            id: 10,
            indices: vec![0, 3],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![100],
        },
        repeated_test_group(11, 1, 200),
        repeated_test_group(12, 2, 300),
    ];
    let entries = vec![
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 0,
            weight_re: 1.25,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 1,
            weight_re: -0.75,
            weight_im: 0.125,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 2,
            weight_re: 0.5,
            weight_im: -0.25,
            symmetry_factor: 2.0,
        },
    ];
    let outputs = (0..POINT_COUNT * 4)
        .map(|index| {
            c64(
                (index * 13 % 97) as f64 * 0.0625 - 2.0,
                (index * 17 % 89) as f64 * -0.03125 + 1.5,
            )
        })
        .collect::<Vec<_>>();
    let mut amplitude = reduction_test_amplitude(4, outputs.clone(), groups, entries);
    assert!(
        amplitude
            .color_contraction
            .as_ref()
            .unwrap()
            .repeated_block
            .is_none()
    );
    let mut row_major = vec![0.0; POINT_COUNT];
    amplitude
        .reduce_scratch_f64_into_selected_slice(POINT_COUNT, &mut row_major, None)
        .unwrap();
    let plane_native = plane_native_totals(&mut amplitude, &outputs, POINT_COUNT, None);
    assert_eq!(plane_native, row_major);
}

#[test]
fn compact_repeated_color_manifest_builds_without_expanded_entries() {
    let groups = vec![
        repeated_test_group(10, 2, 100),
        repeated_test_group(11, 3, 100),
        repeated_test_group(12, 0, 200),
        repeated_test_group(13, 1, 200),
    ];
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: groups.len(),
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
            component_count: 2,
            component_group_ids: vec![10, 11, 12, 13],
            entries: vec![
                GenericRepeatedColorContractionEntryManifest {
                    _exact_weight: None,
                    left_group_index: 0,
                    right_group_index: 0,
                    weight: vec![1.25, 0.0],
                    symmetry_factor: 1.0,
                },
                GenericRepeatedColorContractionEntryManifest {
                    _exact_weight: None,
                    left_group_index: 0,
                    right_group_index: 1,
                    weight: vec![-0.75, 0.0],
                    symmetry_factor: 2.0,
                },
                GenericRepeatedColorContractionEntryManifest {
                    _exact_weight: None,
                    left_group_index: 1,
                    right_group_index: 1,
                    weight: vec![2.0, 0.0],
                    symmetry_factor: 1.0,
                },
            ],
            factorized_block: None,
        }),
    };
    let contraction = build_color_contraction_runtime(Some(&manifest), &groups)
        .unwrap()
        .expect("compact repeated contraction");
    assert!(contraction.entries.is_empty());
    assert_eq!(contraction.logical_entry_count().unwrap(), 6);
    let logical_entries = contraction.logical_entries().collect::<Vec<_>>();
    assert_eq!(
        logical_entries
            .iter()
            .map(|entry| (entry.left_group_index, entry.right_group_index))
            .collect::<Vec<_>>(),
        vec![(0, 0), (0, 2), (2, 2), (1, 1), (1, 3), (3, 3)]
    );

    let outputs = vec![
        c64(0.5, -1.0),
        c64(1.5, 0.25),
        c64(-0.75, 2.0),
        c64(0.125, -0.5),
    ];
    let expected = legacy_color_contraction_totals(&outputs, 4, &groups, &logical_entries)[0];
    let mut amplitude = test_amplitude_runtime(outputs.clone(), Some(contraction));
    amplitude.raw_sum_groups = groups;
    let mut actual = vec![0.0];
    amplitude
        .reduce_scratch_f64_into_selected_slice(1, &mut actual, None)
        .unwrap();
    assert!((actual[0] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0));
}

#[test]
fn compact_walsh_color_manifest_matches_expanded_repeated_reduction() {
    let component_count = 3;
    let output_indices = [7, 0, 10, 3, 5, 11, 1, 8, 4, 9, 2, 6];
    let groups = output_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(group_index, output_index)| {
            repeated_test_group(
                10 + group_index as i64,
                output_index,
                100 + (group_index / component_count) as i64,
            )
        })
        .collect::<Vec<_>>();
    let kernel = [4.0, 1.0, 2.0, 0.5];
    let entries = (0..4)
        .flat_map(|left_group_index| {
            (left_group_index..4).map(move |right_group_index| {
                GenericRepeatedColorContractionEntryManifest {
                    _exact_weight: None,
                    left_group_index,
                    right_group_index,
                    weight: vec![kernel[left_group_index ^ right_group_index], 0.0],
                    symmetry_factor: if left_group_index == right_group_index {
                        1.0
                    } else {
                        2.0
                    },
                }
            })
        })
        .collect::<Vec<_>>();
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: groups.len(),
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
            component_count,
            component_group_ids: (10..10 + groups.len() as i64).collect(),
            entries,
            factorized_block: Some(
                GenericFactorizedColorContractionBlockManifest::KleinFourWalsh {
                    cosets: vec![[0, 1, 2, 3]],
                },
            ),
        }),
    };
    let contraction = build_color_contraction_runtime(Some(&manifest), &groups)
        .unwrap()
        .expect("compact Walsh contraction");
    let repeated = contraction
        .repeated_block
        .as_ref()
        .expect("repeated contraction");
    let walsh = repeated.walsh_block.as_ref().expect("Walsh block");
    assert_eq!(walsh.cosets, vec![[0, 1, 2, 3]]);
    assert_eq!(walsh.entries.len(), 4);
    let logical_entries = contraction.logical_entries().collect::<Vec<_>>();
    let outputs = (0..12)
        .map(|index| {
            let value = index as f64 + 1.0;
            c64(0.25 * value, 0.125 * (5.0 - value))
        })
        .collect::<Vec<_>>();
    let expected = legacy_color_contraction_totals(&outputs, 12, &groups, &logical_entries)[0];
    let mut amplitude = test_amplitude_runtime(outputs.clone(), Some(contraction));
    amplitude.raw_sum_groups = groups;
    let mut actual = vec![0.0];

    amplitude
        .reduce_scratch_f64_into_selected_slice(1, &mut actual, None)
        .unwrap();
    let plane_native = plane_native_totals(&mut amplitude, &outputs, 1, None);

    assert!(
        (actual[0] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
        "Walsh reduction {} differs from expanded repeated reduction {expected}",
        actual[0]
    );
    assert_eq!(plane_native, actual);
}

#[test]
fn compact_c2k_walsh_h8_manifest_matches_expanded_repeated_reduction() {
    let component_count = 3;
    let local_group_count = 8;
    let output_count = component_count * local_group_count;
    let groups = (0..output_count)
        .map(|group_index| {
            repeated_test_group(
                10 + group_index as i64,
                (group_index * 7) % output_count,
                100 + (group_index / component_count) as i64,
            )
        })
        .collect::<Vec<_>>();
    let kernel = [8.0, 1.0, 2.0, 0.5, 3.0, 0.25, 0.75, 0.125];
    let entries = (0..local_group_count)
        .flat_map(|left_group_index| {
            (left_group_index..local_group_count).map(move |right_group_index| {
                GenericRepeatedColorContractionEntryManifest {
                    _exact_weight: None,
                    left_group_index,
                    right_group_index,
                    weight: vec![kernel[left_group_index ^ right_group_index], 0.0],
                    symmetry_factor: if left_group_index == right_group_index {
                        1.0
                    } else {
                        2.0
                    },
                }
            })
        })
        .collect::<Vec<_>>();
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: groups.len(),
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
            component_count,
            component_group_ids: (10..10 + groups.len() as i64).collect(),
            entries,
            factorized_block: Some(
                GenericFactorizedColorContractionBlockManifest::ElementaryAbelianWalsh {
                    rank: 3,
                    cosets: vec![(0..local_group_count).collect()],
                },
            ),
        }),
    };
    let contraction = build_color_contraction_runtime(Some(&manifest), &groups)
        .unwrap()
        .expect("compact C2^3 Walsh contraction");
    let repeated = contraction
        .repeated_block
        .as_ref()
        .expect("repeated contraction");
    let walsh = repeated.c2k_walsh_block.as_ref().expect("C2^3 Walsh block");
    assert_eq!(walsh.subgroup_order, 8);
    assert_eq!(walsh.cosets, vec![(0..8).collect::<Vec<_>>()]);
    assert_eq!(walsh.entries.len(), 8);
    let logical_entries = contraction.logical_entries().collect::<Vec<_>>();
    let outputs = (0..output_count)
        .map(|index| {
            let value = index as f64 + 1.0;
            c64(0.125 * value, 0.0625 * (7.0 - value))
        })
        .collect::<Vec<_>>();
    let expected =
        legacy_color_contraction_totals(&outputs, output_count, &groups, &logical_entries)[0];
    let mut amplitude = test_amplitude_runtime(outputs.clone(), Some(contraction));
    amplitude.raw_sum_groups = groups;
    let mut actual = vec![0.0];

    amplitude
        .reduce_scratch_f64_into_selected_slice(1, &mut actual, None)
        .unwrap();
    let plane_native = plane_native_totals(&mut amplitude, &outputs, 1, None);

    assert!(
        (actual[0] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
        "C2^3 Walsh reduction {} differs from expanded repeated reduction {expected}",
        actual[0]
    );
    assert_eq!(plane_native, actual);
}

#[test]
fn compact_c2k_walsh_h8_multiple_cosets_matches_expanded_reference() {
    const COMPONENT_COUNT: usize = 4;
    const COSET_COUNT: usize = 3;
    const SUBGROUP_ORDER: usize = 8;
    const LOCAL_GROUP_COUNT: usize = COSET_COUNT * SUBGROUP_ORDER;
    const OUTPUT_COUNT: usize = COMPONENT_COUNT * LOCAL_GROUP_COUNT;

    // Each row is ordered by the C2^3 generator bitmask, but the local group
    // indices deliberately form a shuffled partition.
    let cosets = vec![
        vec![17, 2, 21, 6, 13, 10, 1, 22],
        vec![5, 18, 8, 15, 3, 20, 12, 7],
        vec![23, 0, 14, 9, 19, 4, 16, 11],
    ];
    let mut local_coordinates = vec![(usize::MAX, usize::MAX); LOCAL_GROUP_COUNT];
    for (coset_index, coset) in cosets.iter().enumerate() {
        for (subgroup_index, local_group_index) in coset.iter().copied().enumerate() {
            local_coordinates[local_group_index] = (coset_index, subgroup_index);
        }
    }
    assert!(
        local_coordinates
            .iter()
            .all(|coordinates| coordinates.0 != usize::MAX)
    );

    // The six independent coset-pair kernels include negative, small, and
    // off-diagonal weights. XOR indexing makes every block exactly C2^3
    // circulant while retaining nontrivial couplings between all cosets.
    let kernels = [
        [3.25, -0.5, 0.125, 1.75, -2.0, 0.0625, 0.875, -0.25],
        [-1.5, 0.375, 2.25, -0.125, 0.75, -3.0, 0.03125, 1.0],
        [0.625, -2.5, 0.1875, 1.125, -0.75, 0.5, 2.0, -0.0625],
        [4.0, 0.25, -1.25, 0.5, 0.015625, -0.875, 1.5, -2.25],
        [-0.4375, 1.875, -0.03125, 2.5, 0.75, -1.0, 0.3125, 3.0],
        [2.75, -0.1875, 0.5625, -1.75, 1.25, 0.046875, -2.0, 0.375],
    ];
    let kernel_index = |left_coset: usize, right_coset: usize| match (
        left_coset.min(right_coset),
        left_coset.max(right_coset),
    ) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (1, 1) => 3,
        (1, 2) => 4,
        (2, 2) => 5,
        pair => panic!("unexpected coset pair {pair:?}"),
    };
    let mut entries = Vec::new();
    for left_group_index in 0..LOCAL_GROUP_COUNT {
        let (left_coset, left_subgroup) = local_coordinates[left_group_index];
        for (right_group_index, &(right_coset, right_subgroup)) in
            local_coordinates.iter().enumerate().skip(left_group_index)
        {
            entries.push(GenericRepeatedColorContractionEntryManifest {
                _exact_weight: None,
                left_group_index,
                right_group_index,
                weight: vec![
                    kernels[kernel_index(left_coset, right_coset)][left_subgroup ^ right_subgroup],
                    0.0,
                ],
                symmetry_factor: if left_group_index == right_group_index {
                    1.0
                } else {
                    2.0
                },
            });
        }
    }

    let group_id = |mapping_slot: usize| 10_000 + 13 * mapping_slot as i64;
    let component_group_ids = (0..OUTPUT_COUNT).map(group_id).collect::<Vec<_>>();
    let mut group_index_by_mapping_slot = vec![usize::MAX; OUTPUT_COUNT];
    let groups = (0..OUTPUT_COUNT)
        .map(|group_index| {
            let mapping_slot = (group_index * 37 + 11) % OUTPUT_COUNT;
            group_index_by_mapping_slot[mapping_slot] = group_index;
            repeated_test_group(
                group_id(mapping_slot),
                (mapping_slot * 29 + 7) % OUTPUT_COUNT,
                500 + (mapping_slot / COMPONENT_COUNT) as i64,
            )
        })
        .collect::<Vec<_>>();
    assert!(
        group_index_by_mapping_slot
            .iter()
            .all(|group_index| *group_index != usize::MAX)
    );

    // Build the independent expanded reference before moving the compact
    // entries into the manifest. This reproduces the logical matrix for every
    // repeated component without consulting the optimized runtime plan.
    let group_index_by_mapping_slot = &group_index_by_mapping_slot;
    let explicit_entries = (0..COMPONENT_COUNT)
        .flat_map(|component_index| {
            entries.iter().map(move |entry| ColorContractionEntry {
                left_group_index: group_index_by_mapping_slot
                    [entry.left_group_index * COMPONENT_COUNT + component_index],
                right_group_index: group_index_by_mapping_slot
                    [entry.right_group_index * COMPONENT_COUNT + component_index],
                weight_re: entry.weight[0],
                weight_im: entry.weight[1],
                symmetry_factor: entry.symmetry_factor,
            })
        })
        .collect::<Vec<_>>();
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: groups.len(),
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
            component_count: COMPONENT_COUNT,
            component_group_ids,
            entries,
            factorized_block: Some(
                GenericFactorizedColorContractionBlockManifest::ElementaryAbelianWalsh {
                    rank: 3,
                    cosets,
                },
            ),
        }),
    };
    let contraction = build_color_contraction_runtime(Some(&manifest), &groups)
        .unwrap()
        .expect("multi-coset C2^3 Walsh contraction");
    let repeated = contraction
        .repeated_block
        .as_ref()
        .expect("repeated contraction");
    let walsh = repeated.c2k_walsh_block.as_ref().expect("C2^3 Walsh block");
    assert_eq!(walsh.subgroup_order, SUBGROUP_ORDER);
    assert_eq!(walsh.cosets.len(), COSET_COUNT);
    assert!(
        walsh
            .entries
            .iter()
            .any(|entry| entry.left_group_index != entry.right_group_index),
        "transformed plan must retain off-diagonal coset couplings"
    );

    let point_count = 3;
    let mut outputs = Vec::with_capacity(point_count * OUTPUT_COUNT);
    for point_index in 0..point_count {
        let point_scale = [1.0, 0.03125, 5.5][point_index];
        for output_index in 0..OUTPUT_COUNT {
            let centered = output_index as f64 - (OUTPUT_COUNT as f64 - 1.0) * 0.5;
            let modulation = ((output_index * 17 + point_index * 5) % 19) as f64 - 9.0;
            outputs.push(c64(
                point_scale * (0.0234375 * centered + 0.0078125 * modulation),
                point_scale
                    * (-0.015625 * centered + 0.00390625 * (modulation * modulation - 13.0)),
            ));
        }
    }
    let expected =
        legacy_color_contraction_totals(&outputs, OUTPUT_COUNT, &groups, &explicit_entries);
    let mut amplitude = test_amplitude_runtime(outputs, Some(contraction));
    amplitude.output_length = OUTPUT_COUNT;
    amplitude.raw_sum_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_all_sector_weights = vec![1.0; OUTPUT_COUNT];
    amplitude.raw_sum_color_sector_ids = vec![None; OUTPUT_COUNT];
    amplitude.raw_sum_groups = groups;
    let mut actual = vec![0.0; point_count];

    amplitude
        .reduce_scratch_f64_into_selected_slice(point_count, &mut actual, None)
        .unwrap();

    for (point_index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let absolute_error = (actual - expected).abs();
        let tolerance = 1.0e-15 + 1.0e-12 * expected.abs();
        assert!(
            absolute_error <= tolerance,
            "multi-coset C2^3 Walsh point {point_index} differs from expanded reference: \
             actual={actual}, expected={expected}, absolute_error={absolute_error}, \
             tolerance={tolerance}"
        );
    }
}

#[test]
fn compact_c2k_walsh_generic_rank_four_matches_expanded_reduction() {
    let component_count = 2;
    let local_group_count = 16;
    let output_count = component_count * local_group_count;
    let groups = (0..output_count)
        .map(|group_index| {
            repeated_test_group(
                20 + group_index as i64,
                group_index,
                200 + (group_index / component_count) as i64,
            )
        })
        .collect::<Vec<_>>();
    let kernel = (0..local_group_count)
        .map(|index| 1.0 / (index + 1) as f64)
        .collect::<Vec<_>>();
    let entries = (0..local_group_count)
        .flat_map(|left_group_index| {
            let kernel = &kernel;
            (left_group_index..local_group_count).map(move |right_group_index| {
                GenericRepeatedColorContractionEntryManifest {
                    _exact_weight: None,
                    left_group_index,
                    right_group_index,
                    weight: vec![kernel[left_group_index ^ right_group_index], 0.0],
                    symmetry_factor: if left_group_index == right_group_index {
                        1.0
                    } else {
                        2.0
                    },
                }
            })
        })
        .collect::<Vec<_>>();
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: groups.len(),
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
            component_count,
            component_group_ids: (20..20 + groups.len() as i64).collect(),
            entries,
            factorized_block: Some(
                GenericFactorizedColorContractionBlockManifest::ElementaryAbelianWalsh {
                    rank: 4,
                    cosets: vec![(0..local_group_count).collect()],
                },
            ),
        }),
    };
    let contraction = build_color_contraction_runtime(Some(&manifest), &groups)
        .unwrap()
        .expect("compact C2^4 Walsh contraction");
    let logical_entries = contraction.logical_entries().collect::<Vec<_>>();
    let outputs = (0..output_count)
        .map(|index| c64(index as f64 * 0.03125, 1.0 - index as f64 * 0.015625))
        .collect::<Vec<_>>();
    let expected =
        legacy_color_contraction_totals(&outputs, output_count, &groups, &logical_entries)[0];
    let mut amplitude = test_amplitude_runtime(outputs, Some(contraction));
    amplitude.raw_sum_groups = groups;
    let mut actual = vec![0.0];

    amplitude
        .reduce_scratch_f64_into_selected_slice(1, &mut actual, None)
        .unwrap();

    assert!(
        (actual[0] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
        "generic C2^4 Walsh reduction {} differs from expanded repeated reduction {expected}",
        actual[0]
    );
}

#[test]
fn compact_walsh_color_manifest_rejects_malformed_or_noninvariant_plans() {
    let groups = (0..8)
        .map(|group_index| {
            repeated_test_group(
                10 + group_index as i64,
                group_index,
                100 + (group_index / 2) as i64,
            )
        })
        .collect::<Vec<_>>();
    let manifest = |cosets: Vec<[usize; 4]>,
                    entries: Vec<GenericRepeatedColorContractionEntryManifest>| {
        GenericColorContractionManifest {
            supported: true,
            reason: None,
            group_count: groups.len(),
            includes_color_factor: true,
            entries: Vec::new(),
            repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
                component_count: 2,
                component_group_ids: (10..18).collect(),
                entries,
                factorized_block: Some(
                    GenericFactorizedColorContractionBlockManifest::KleinFourWalsh { cosets },
                ),
            }),
        }
    };
    let cases = [
        ("duplicate index", manifest(vec![[0, 1, 2, 2]], Vec::new())),
        (
            "not invariant",
            manifest(
                vec![[0, 1, 2, 3]],
                vec![GenericRepeatedColorContractionEntryManifest {
                    _exact_weight: None,
                    left_group_index: 0,
                    right_group_index: 0,
                    weight: vec![1.0, 0.0],
                    symmetry_factor: 1.0,
                }],
            ),
        ),
    ];
    for (expected, manifest) in cases {
        let error = match build_color_contraction_runtime(Some(&manifest), &groups) {
            Ok(_) => panic!("malformed Walsh color contraction must fail"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(expected),
            "unexpected error for {expected}: {error}"
        );
    }

    let unknown = serde_json::from_value::<GenericFactorizedColorContractionBlockManifest>(
        json!({"kind": "future-transform", "cosets": [[0, 1, 2, 3]]}),
    );
    assert!(
        unknown.is_err(),
        "unknown factorization kind must fail closed"
    );
}

#[test]
fn compact_c2k_walsh_manifest_fails_closed_on_invalid_metadata() {
    let groups = (0..16)
        .map(|group_index| {
            repeated_test_group(
                10 + group_index as i64,
                group_index,
                100 + (group_index / 2) as i64,
            )
        })
        .collect::<Vec<_>>();
    let entry = |left_group_index: usize,
                 right_group_index: usize,
                 weight_re: f64|
     -> GenericRepeatedColorContractionEntryManifest {
        GenericRepeatedColorContractionEntryManifest {
            _exact_weight: None,
            left_group_index,
            right_group_index,
            weight: vec![weight_re, 0.0],
            symmetry_factor: if left_group_index == right_group_index {
                1.0
            } else {
                2.0
            },
        }
    };
    let manifest = |rank: usize,
                    cosets: Vec<Vec<usize>>,
                    entries: Vec<GenericRepeatedColorContractionEntryManifest>| {
        GenericColorContractionManifest {
            supported: true,
            reason: None,
            group_count: groups.len(),
            includes_color_factor: true,
            entries: Vec::new(),
            repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
                component_count: 2,
                component_group_ids: (10..26).collect(),
                entries,
                factorized_block: Some(
                    GenericFactorizedColorContractionBlockManifest::ElementaryAbelianWalsh {
                        rank,
                        cosets,
                    },
                ),
            }),
        }
    };
    let canonical_coset = vec![(0..8).collect::<Vec<_>>()];
    let cases = [
        (
            "rank must be at least three",
            manifest(2, vec![(0..4).collect()], Vec::new()),
        ),
        (
            "do not match rank or local groups",
            manifest(3, vec![(0..7).collect()], Vec::new()),
        ),
        (
            "duplicate index",
            manifest(3, vec![vec![0, 1, 2, 3, 4, 5, 6, 6]], Vec::new()),
        ),
        (
            "weights must be finite",
            manifest(3, canonical_coset.clone(), vec![entry(0, 0, f64::NAN)]),
        ),
        (
            "duplicate matrix entry",
            manifest(
                3,
                canonical_coset.clone(),
                vec![entry(0, 1, 1.0), entry(1, 0, 1.0)],
            ),
        ),
        (
            "not invariant",
            manifest(3, canonical_coset.clone(), vec![entry(0, 0, 1.0)]),
        ),
        (
            "subgroup order overflows",
            manifest(usize::BITS as usize, canonical_coset, Vec::new()),
        ),
    ];
    for (expected, manifest) in cases {
        let error = match build_color_contraction_runtime(Some(&manifest), &groups) {
            Ok(_) => panic!("malformed C2^k Walsh color contraction must fail"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(expected),
            "unexpected error for {expected}: {error}"
        );
    }

    for malformed in [
        json!({
            "kind": "elementary-abelian-walsh",
            "cosets": [[0, 1, 2, 3, 4, 5, 6, 7]],
        }),
        json!({
            "kind": "elementary-abelian-walsh",
            "rank": 3,
            "cosets": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "unexpected": true,
        }),
    ] {
        assert!(
            serde_json::from_value::<GenericFactorizedColorContractionBlockManifest>(malformed)
                .is_err(),
            "C2^k Walsh wire metadata must be exact"
        );
    }
}

#[test]
fn compact_repeated_color_manifest_rejects_duplicate_group_mapping() {
    let groups = vec![
        repeated_test_group(10, 0, 100),
        repeated_test_group(11, 1, 100),
    ];
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: groups.len(),
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
            component_count: 2,
            component_group_ids: vec![10, 10],
            entries: Vec::new(),
            factorized_block: None,
        }),
    };
    let error = match build_color_contraction_runtime(Some(&manifest), &groups) {
        Ok(_) => panic!("duplicate repeated color group mapping must fail"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("maps a coherent group more than once")
    );
}

#[test]
fn compact_repeated_color_manifest_rejects_malformed_storage() {
    let missing_entries = serde_json::from_value::<GenericColorContractionManifest>(json!({
        "supported": true,
        "group_count": 2,
        "includes_color_factor": true,
    }));
    assert!(missing_entries.is_err());

    let groups = vec![
        repeated_test_group(10, 0, 100),
        repeated_test_group(11, 1, 100),
    ];
    let repeated = |weight: Vec<f64>, left_group_index: usize| {
        Some(GenericRepeatedColorContractionBlockManifest {
            component_count: 2,
            component_group_ids: vec![10, 11],
            entries: vec![GenericRepeatedColorContractionEntryManifest {
                _exact_weight: None,
                left_group_index,
                right_group_index: 0,
                weight,
                symmetry_factor: 1.0,
            }],
            factorized_block: None,
        })
    };
    let malformed = [
        (
            "two components",
            GenericColorContractionManifest {
                supported: true,
                reason: None,
                group_count: 2,
                includes_color_factor: true,
                entries: Vec::new(),
                repeated_block: repeated(vec![1.0], 0),
            },
        ),
        (
            "out of bounds",
            GenericColorContractionManifest {
                supported: true,
                reason: None,
                group_count: 2,
                includes_color_factor: true,
                entries: Vec::new(),
                repeated_block: repeated(vec![1.0, 0.0], 1),
            },
        ),
        (
            "cannot mix",
            GenericColorContractionManifest {
                supported: true,
                reason: None,
                group_count: 2,
                includes_color_factor: true,
                entries: vec![GenericColorContractionEntryManifest {
                    _exact_weight: None,
                    left_group_id: 10,
                    right_group_id: 10,
                    weight: vec![1.0, 0.0],
                    symmetry_factor: 1.0,
                }],
                repeated_block: repeated(vec![1.0, 0.0], 0),
            },
        ),
    ];
    for (expected, manifest) in malformed {
        let error = match build_color_contraction_runtime(Some(&manifest), &groups) {
            Ok(_) => panic!("malformed repeated color storage must fail"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(expected),
            "unexpected error for {expected}: {error}"
        );
    }
}

#[test]
fn repeated_complex_color_blocks_match_legacy_reduction() {
    let groups = vec![
        repeated_test_group(10, 0, 100),
        repeated_test_group(11, 1, 100),
        repeated_test_group(12, 2, 200),
        repeated_test_group(13, 3, 200),
    ];
    let entries = vec![
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 2,
            weight_re: 0.75,
            weight_im: -0.25,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 3,
            weight_re: 0.75,
            weight_im: -0.25,
            symmetry_factor: 2.0,
        },
    ];
    let outputs = vec![
        c64(1.0, 2.0),
        c64(-0.5, 0.25),
        c64(0.75, -1.0),
        c64(2.0, 0.5),
    ];
    let expected = legacy_color_contraction_totals(&outputs, 4, &groups, &entries);
    let mut amplitude = reduction_test_amplitude(4, outputs.clone(), groups, entries);
    assert!(
        !amplitude
            .color_contraction
            .as_ref()
            .unwrap()
            .repeated_block
            .as_ref()
            .unwrap()
            .all_weights_real
    );
    let mut actual = vec![0.0];
    amplitude
        .reduce_scratch_f64_into_selected_slice(1, &mut actual, None)
        .unwrap();
    let plane_native = plane_native_totals(&mut amplitude, &outputs, 1, None);
    assert_eq!(plane_native, actual);
    assert!((actual[0] - expected[0]).abs() <= 1.0e-12 * expected[0].abs().max(1.0));
}

#[test]
fn contracted_color_coverage_does_not_warn_as_incomplete() {
    for color_accuracy in ["nlc", "full"] {
        let physics = test_physics_runtime(color_accuracy);
        let physics_v1 = physics.manifest.clone();
        let mut execution = empty_generic_runtime();
        execution.color_accuracy = color_accuracy.to_string();
        execution.physics = Some(Arc::new(physics));
        let mut runtime = NativeRuntime {
            root: PathBuf::new(),
            artifact_id: "0".repeat(64),
            runtime: execution,
            execution_lane: NativeExecutionLane::Compiled,
            process: "x x > y".to_string(),
            process_key: "x_x_to_y".to_string(),
            representative_process_id: "x_x_to_y".to_string(),
            external_permutation: vec![0, 1, 2],
            input_crossing_map: None,
            permutation_alias_of: None,
            final_state_permutation_alias_of: None,
            physics_v1: native_runtime::LazyProcessPhysicsV1::loaded(physics_v1),
            warnings_muted: false,
            warned_kinds: BTreeSet::new(),
            pending_warnings: Vec::new(),
            point_selector_scratch: PointSelectorExecutionScratch::default(),
            selector_simd_lane_width: 1,
            correlated: None,
            process_group: None,
        };

        runtime.record_resolved_warnings(None, None).unwrap();

        assert!(runtime.pending_warnings.is_empty());
    }
}

#[test]
fn inconsistent_helicity_weights_are_rejected() {
    let mut physics = test_physics_runtime("nlc").manifest;
    physics.helicities[1].coefficient = 2.0;

    let error = PhysicsRuntime::new(physics).err().unwrap();

    assert!(error.to_string().contains("inconsistent helicity weights"));
}

#[test]
fn numeric_reduction_binding_preserves_manifest_order_and_weights() {
    for color_accuracy in ["lc", "nlc", "full"] {
        let physics = test_physics_runtime(color_accuracy);
        let reduction = physics.reduction_by_group_id.get(&7).unwrap();
        let numeric = physics.numeric_reduction_by_group_id.get(&7).unwrap();

        let expected_helicity_indices = reduction
            .physical_helicity_ids
            .iter()
            .map(|id| physics.helicity_index_by_id[id])
            .collect::<Vec<_>>();
        assert_eq!(numeric.physical_helicity_indices, expected_helicity_indices);
        for (index, helicity) in physics.manifest.helicities.iter().enumerate() {
            assert_eq!(
                numeric.contains_helicity(index),
                reduction.physical_helicity_ids.contains(&helicity.id)
            );
        }
        assert_eq!(
            numeric.normalized_helicity_weights,
            physics.normalized_helicity_weights(reduction).unwrap()
        );
        assert_eq!(
            numeric.normalized_member_weights,
            physics.normalized_member_weights(reduction).unwrap()
        );

        let mut expected_color_weights = reduction
            .physical_color_ids
            .iter()
            .map(|id| {
                let index = physics.color_index_by_id[id];
                (
                    index,
                    physics.manifest.color_components[index].coefficient(),
                )
            })
            .collect::<Vec<_>>();
        let total_color_weight = expected_color_weights
            .iter()
            .map(|(_, weight)| *weight)
            .sum::<f64>();
        for (_, weight) in &mut expected_color_weights {
            *weight /= total_color_weight;
        }
        assert_eq!(numeric.normalized_color_weights, expected_color_weights);
    }
}

pub(super) fn empty_generic_runtime() -> ExecutionRuntime {
    ExecutionRuntime {
        process: "a b > c".to_string(),
        key: "p0".to_string(),
        color_accuracy: "lc".to_string(),
        external_pdg_order: Vec::new(),
        external_count: 0,
        parameter_count: 1,
        value_parameter_count: 0,
        momentum_parameter_count: 0,
        current_count: 0,
        source_count: 0,
        interaction_count: 0,
        stage_count: 0,
        amplitude_output_count: 0,
        lc_topology_replay_enabled: false,
        lc_topology_replay_mappings: Arc::new(Vec::new()),
        lc_topology_replay_public_mappings: Vec::new(),
        lc_topology_replay_routes: Vec::new(),
        lc_topology_replay_materialized_sector_ids: BTreeSet::new(),
        lc_resolved_replay_plan: None,
        lc_resolved_replay_selection_cache: None,
        lc_replay_flat_momenta_scratch: Vec::new(),
        lc_replay_target_components_scratch: Vec::new(),
        color_topology_replay_enabled: false,
        color_topology_replay_mappings: Arc::new(Vec::new()),
        color_replay_flat_momenta_scratch: Vec::new(),
        helicity_recurrence: None,
        compiled_helicity_execution_plan: None,
        compiled_color_execution_plan: None,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        compiled_direct_runtime: None,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        compiled_direct_color_schedules: BTreeMap::new(),
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        compiled_direct_helicity_schedules: BTreeMap::new(),
        helicity_sum_runtime: None,
        helicity_selector_runtimes: Vec::new(),
        helicity_selector_runtime_schedule_modes: Vec::new(),
        helicity_selector_lane_by_domain: BTreeMap::new(),
        color_selector_runtimes: BTreeMap::new(),
        runtime_unavailable_message: None,
        sources: Vec::new(),
        momentum_slots: Vec::new(),
        external_is_initial: Vec::new(),
        particle_masses: BTreeMap::new(),
        particle_mass_parameter_names: BTreeMap::new(),
        normalization_factor: 1.0,
        normalization_color_factor: 1.0,
        normalization_average_factor: 1.0,
        normalization_identical_factor: 1.0,
        normalization_qcd_coupling_power: 0,
        normalization_electroweak_coupling_power: 0,
        model_parameters: Vec::new(),
        model_parameter_name_to_index: BTreeMap::new(),
        model_parameter_runtime_slots: BTreeMap::new(),
        model_parameter_values_f64: vec![0.118],
        model_parameter_evaluator: None,
        physics_reduction_override: None,
        physics: None,
        stages: None,
        amplitude_stage: None,
        state_scratch_f64: Vec::new(),
        state_scratch_f64_requires_clear: false,
        values_scratch_f64: Vec::new(),
    }
}

fn lc_direct_total_test_group(id: i64, helicity_id: &str, color_id: &str) -> crate::ReductionGroup {
    crate::ReductionGroup {
        id: format!("group:{id}"),
        representative_helicity_id: helicity_id.to_string(),
        physical_helicity_ids: vec![helicity_id.to_string()],
        representative_color_id: color_id.to_string(),
        physical_color_ids: vec![color_id.to_string()],
    }
}

fn strict_lc_direct_total_test_groups() -> Vec<crate::ReductionGroup> {
    [
        ("hel:+-", "flow:0"),
        ("hel:+-", "flow:1"),
        ("hel:-+", "flow:0"),
        ("hel:-+", "flow:1"),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (helicity_id, color_id))| {
        lc_direct_total_test_group(7 + index as i64, helicity_id, color_id)
    })
    .collect()
}

fn lc_direct_total_test_runtime(groups: Vec<crate::ReductionGroup>) -> ExecutionRuntime {
    let mut manifest = test_physics_runtime("lc").manifest;
    manifest.helicities[1].computed = true;
    manifest.helicities[1].representative_id = manifest.helicities[1].id.clone();
    let crate::ColorComponent::LcFlow(second_flow) = &mut manifest.color_components[1] else {
        panic!("LC test physics must contain LC flows");
    };
    second_flow.computed = true;
    second_flow.representative_id = second_flow.id.clone();
    manifest.reduction.groups = groups.clone();
    let physics = PhysicsRuntime::new(manifest).expect("valid LC direct-total test physics");

    let mut amplitude = test_amplitude_runtime(vec![c64(1.0, 0.0); groups.len()], None);
    amplitude.raw_sum_groups = groups
        .iter()
        .enumerate()
        .map(|(index, group)| RawSumGroup {
            id: parse_reduction_group_id(&group.id).expect("numeric test reduction group id"),
            indices: vec![index],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![index as i64],
        })
        .collect();

    let mut runtime = empty_generic_runtime();
    runtime.amplitude_output_count = groups.len();
    runtime.physics = Some(Arc::new(physics));
    runtime.amplitude_stage = Some(amplitude);
    runtime
}

fn lc_direct_total_test_selectors() -> (BTreeSet<String>, BTreeSet<String>) {
    (
        BTreeSet::from(["hel:+-".to_string(), "hel:-+".to_string()]),
        BTreeSet::from(["flow:0".to_string(), "flow:1".to_string()]),
    )
}

#[test]
fn lc_direct_total_source_certification_accepts_exact_cartesian_partitions() {
    let runtime = lc_direct_total_test_runtime(strict_lc_direct_total_test_groups());
    let (helicity_ids, color_ids) = lc_direct_total_test_selectors();

    assert!(certifies_lc_direct_total_source(
        &runtime,
        &helicity_ids,
        &color_ids
    ));

    let grouped_runtime = lc_direct_total_test_runtime(vec![
        crate::ReductionGroup {
            id: "group:7".to_string(),
            representative_helicity_id: "hel:+-".to_string(),
            physical_helicity_ids: vec!["hel:+-".to_string(), "hel:-+".to_string()],
            representative_color_id: "flow:0".to_string(),
            physical_color_ids: vec!["flow:0".to_string()],
        },
        crate::ReductionGroup {
            id: "group:8".to_string(),
            representative_helicity_id: "hel:+-".to_string(),
            physical_helicity_ids: vec!["hel:+-".to_string(), "hel:-+".to_string()],
            representative_color_id: "flow:1".to_string(),
            physical_color_ids: vec!["flow:1".to_string()],
        },
    ]);
    assert!(certifies_lc_direct_total_source(
        &grouped_runtime,
        &helicity_ids,
        &color_ids
    ));
}

#[test]
fn lc_direct_total_source_certification_rejects_nonexact_auxiliary_reductions() {
    let (helicity_ids, color_ids) = lc_direct_total_test_selectors();

    let mut overbroad_groups = strict_lc_direct_total_test_groups();
    overbroad_groups[0].physical_helicity_ids = vec!["hel:+-".to_string(), "hel:-+".to_string()];
    let overbroad = lc_direct_total_test_runtime(overbroad_groups);
    assert!(
        !certifies_lc_direct_total_source(&overbroad, &helicity_ids, &color_ids),
        "an auxiliary reduction group may not cover extra source helicities"
    );

    let mut partial_groups = strict_lc_direct_total_test_groups();
    partial_groups.pop();
    let partial = lc_direct_total_test_runtime(partial_groups);
    assert!(
        !certifies_lc_direct_total_source(&partial, &helicity_ids, &color_ids),
        "an auxiliary reduction must cover every requested source component"
    );

    let mut duplicate_groups = strict_lc_direct_total_test_groups();
    duplicate_groups[3].representative_helicity_id = "hel:+-".to_string();
    duplicate_groups[3].physical_helicity_ids = vec!["hel:+-".to_string()];
    duplicate_groups[3].representative_color_id = "flow:0".to_string();
    duplicate_groups[3].physical_color_ids = vec!["flow:0".to_string()];
    let duplicate = lc_direct_total_test_runtime(duplicate_groups);
    assert!(
        !certifies_lc_direct_total_source(&duplicate, &helicity_ids, &color_ids),
        "duplicate source members may not stand in for a missing Cartesian member"
    );

    let mut mismatched_ids = lc_direct_total_test_runtime(strict_lc_direct_total_test_groups());
    mismatched_ids
        .amplitude_stage
        .as_mut()
        .expect("test amplitude")
        .raw_sum_groups[0]
        .id = 999;
    assert!(
        !certifies_lc_direct_total_source(&mismatched_ids, &helicity_ids, &color_ids),
        "the auxiliary evaluator groups must match the certified physics reduction"
    );
}

fn zero_native_runtime() -> NativeRuntime {
    let physics = test_physics_runtime("lc");
    let physics_v1 = physics.manifest.clone();
    let mut execution = empty_generic_runtime();
    execution.external_count = 3;
    execution.external_pdg_order = vec![1, -1, 23];
    execution.external_is_initial = vec![true, true, false];
    execution.physics = Some(Arc::new(physics));
    execution.stages = Some(Vec::new());
    let mut amplitude = test_amplitude_runtime(Vec::new(), None);
    amplitude.input_components = Some(Vec::new());
    execution.amplitude_stage = Some(amplitude);
    NativeRuntime {
        root: PathBuf::new(),
        artifact_id: "0".repeat(64),
        runtime: execution,
        execution_lane: NativeExecutionLane::Compiled,
        process: "x x > y".to_string(),
        process_key: "x_x_to_y".to_string(),
        representative_process_id: "x_x_to_y".to_string(),
        external_permutation: vec![0, 1, 2],
        input_crossing_map: None,
        permutation_alias_of: None,
        final_state_permutation_alias_of: None,
        physics_v1: native_runtime::LazyProcessPhysicsV1::loaded(physics_v1),
        warnings_muted: false,
        warned_kinds: BTreeSet::new(),
        pending_warnings: Vec::new(),
        point_selector_scratch: PointSelectorExecutionScratch::default(),
        selector_simd_lane_width: 1,
        correlated: None,
        process_group: None,
    }
}

#[test]
fn native_runtime_clear_is_a_noop_outside_the_on_the_fly_lane() {
    let mut runtime = zero_native_runtime();
    let artifact_id = runtime.artifact_id().to_string();
    let parameters = runtime.runtime.model_parameter_values_f64.clone();
    let metadata = runtime.metadata_json().unwrap();

    runtime.clear().unwrap();

    assert_eq!(runtime.artifact_id(), artifact_id);
    assert_eq!(runtime.runtime.model_parameter_values_f64, parameters);
    assert_eq!(runtime.metadata_json().unwrap(), metadata);
    assert!(
        runtime
            .on_the_fly_runtime_state_census_json()
            .unwrap()
            .is_none()
    );
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn scalar_on_the_fly_native_runtime() -> NativeRuntime {
    scalar_on_the_fly_native_runtime_with_color_contraction(None)
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn scalar_on_the_fly_native_runtime_with_color_contraction(
    color_contraction: Option<super::on_the_fly_load::LoadedOnTheFlyColorContractionV1>,
) -> NativeRuntime {
    use super::on_the_fly_lane::OnTheFlyNativeRuntime;
    use super::on_the_fly_selectors::{
        OnTheFlyCompactSelectorAdapterV1, OnTheFlyLcColorCoverageV1, OnTheFlyLcSelectorPolicyV1,
    };
    use super::recurrence_backend::on_the_fly_adapter_tests::{
        complete_scalar_direct_catalog, complete_scalar_prepared_pool, digest, source_domains,
    };
    use crate::recurrence::CheckedTableRange;
    use crate::recurrence::on_the_fly::scalar_adapter_test_seed;
    use crate::recurrence::template::{CouplingOrderTermRow, IndexedRangeRow};
    use crate::recurrence::validated_template_fixture;

    let mut template_input = validated_template_fixture().into_input();
    let spin_start = u64::try_from(template_input.i32_sequence_values.len()).unwrap();
    let spin_sequence_id = u32::try_from(template_input.i32_sequence_ranges.len()).unwrap();
    template_input.i32_sequence_ranges.push(IndexedRangeRow {
        id: spin_sequence_id,
        range: CheckedTableRange::new(spin_start, 2),
    });
    template_input.i32_sequence_values.extend([50_000, 50_000]);
    template_input.quantum_flows[0].input_spin_sequence_id = spin_sequence_id;
    template_input.coupling_order_ranges.push(IndexedRangeRow {
        id: 1,
        range: CheckedTableRange::new(0, 1),
    });
    template_input
        .coupling_order_terms
        .push(CouplingOrderTermRow {
            set_id: 1,
            name_string_id: 0,
            power: 1,
        });
    let templates = template_input.validate().unwrap();
    let summary = templates.summary();
    let direct_digest = digest(40);
    let direct = complete_scalar_direct_catalog(direct_digest);
    let seed = scalar_adapter_test_seed(
        summary.compiled_model_digest,
        summary.catalog_digest,
        summary.prepared_kernel_pack_digest,
        direct_digest,
    )
    .and_then(|seed| seed.with_selector_local_zero())
    .unwrap();
    let pool = complete_scalar_prepared_pool(&templates, direct_digest);
    let sources = pool.bind_source_domains(source_domains()).unwrap();
    let resolver = pool.into_on_the_fly_resolver(sources);
    let defaults = vec![
        crate::EagerComplex64::new(0.0, 0.0);
        usize::try_from(summary.parameter_count).unwrap()
    ];
    let selectors = OnTheFlyCompactSelectorAdapterV1::from_seed(
        &seed,
        OnTheFlyLcSelectorPolicyV1 {
            color_coverage: OnTheFlyLcColorCoverageV1::Complete,
            color_basis: Default::default(),
            reference_color_word: None,
            trace_reflections_folded: false,
        },
    )
    .unwrap();
    let lane = OnTheFlyNativeRuntime::new(
        templates,
        direct,
        seed,
        resolver,
        1,
        1,
        defaults,
        Vec::new(),
        &[],
    )
    .unwrap();
    let mut execution = empty_generic_runtime();
    execution.external_count = 2;
    execution.external_pdg_order = vec![900_000, 900_000];
    execution.external_is_initial = vec![true, false];
    execution.physics = None;
    execution.normalization_factor = 1.0;
    NativeRuntime {
        root: PathBuf::new(),
        artifact_id: "0".repeat(64),
        runtime: execution,
        execution_lane: NativeExecutionLane::OnTheFly(Box::new(
            native_runtime::OnTheFlyExecutionRuntime::new(lane, selectors, color_contraction),
        )),
        process: "s > s".to_string(),
        process_key: "s_to_s".to_string(),
        representative_process_id: "s_to_s".to_string(),
        external_permutation: vec![0, 1],
        input_crossing_map: None,
        permutation_alias_of: None,
        final_state_permutation_alias_of: None,
        physics_v1: native_runtime::LazyProcessPhysicsV1::unavailable(),
        warnings_muted: false,
        warned_kinds: BTreeSet::new(),
        pending_warnings: Vec::new(),
        point_selector_scratch: PointSelectorExecutionScratch::default(),
        selector_simd_lane_width: 1,
        correlated: None,
        process_group: None,
    }
}

#[cfg(all(
    any(feature = "f64-compiled", feature = "f64-symjit"),
    any(target_os = "linux", target_os = "macos")
))]
#[test]
fn on_the_fly_warmed_fixed_selection_is_allocation_free_and_last_family_only() {
    let mut runtime = scalar_on_the_fly_native_runtime();
    let point_count = 2;
    let momenta = vec![0.0; point_count * 2 * 4];
    let color = ["flow:singlet".to_string()];
    let selection_a = ["h:+0,+0".to_string()];
    let selection_b = ["h:+1,+1".to_string()];
    let mut output = vec![0.0; point_count];

    runtime
        .evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            Some(&selection_a),
            Some(&color),
            None,
            None,
            &mut output,
        )
        .unwrap();
    assert!(output.iter().all(|value| value.abs() > f64::EPSILON));
    let expected_a = output.clone();

    let (warm_a, allocations, bytes) = super::count_allocations(|| {
        runtime.evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            Some(&selection_a),
            Some(&color),
            None,
            None,
            &mut output,
        )
    });
    warm_a.unwrap();
    assert_eq!(output, expected_a);
    assert_eq!(allocations, 0, "warmed fixed OTF selection allocated");
    assert_eq!(bytes, 0, "warmed fixed OTF selection allocated bytes");

    let (cold_b, allocations, _) = super::count_allocations(|| {
        runtime.evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            Some(&selection_b),
            Some(&color),
            None,
            None,
            &mut output,
        )
    });
    cold_b.unwrap();
    assert!(allocations > 0, "replacement family was not prepared cold");
    assert_eq!(output, [0.0; 2]);

    let (warm_b, allocations, bytes) = super::count_allocations(|| {
        runtime.evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            Some(&selection_b),
            Some(&color),
            None,
            None,
            &mut output,
        )
    });
    warm_b.unwrap();
    assert_eq!(allocations, 0, "warmed replacement OTF selection allocated");
    assert_eq!(bytes, 0, "warmed replacement OTF selection allocated bytes");

    // Only the current family is retained. Returning to A must rebuild it,
    // after which the same A selection is allocation-free again.
    let (revisited_a, allocations, _) = super::count_allocations(|| {
        runtime.evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            Some(&selection_a),
            Some(&color),
            None,
            None,
            &mut output,
        )
    });
    revisited_a.unwrap();
    assert!(
        allocations > 0,
        "evicted fixed OTF selection was not rebuilt"
    );
    assert_eq!(output, expected_a);

    let (warm_revisited_a, allocations, bytes) = super::count_allocations(|| {
        runtime.evaluate_f64_into_with_selectors(
            &momenta,
            point_count,
            Some(&selection_a),
            Some(&color),
            None,
            None,
            &mut output,
        )
    });
    warm_revisited_a.unwrap();
    assert_eq!(allocations, 0, "rewarmed fixed OTF selection allocated");
    assert_eq!(bytes, 0, "rewarmed fixed OTF selection allocated bytes");
    let NativeExecutionLane::OnTheFly(execution) = &runtime.execution_lane else {
        panic!("test runtime changed execution lane");
    };
    assert_eq!(execution.retained_family_count(), 1);
    assert_eq!(execution.retained_selection_count(), 1);
}

#[cfg(all(
    any(feature = "f64-compiled", feature = "f64-symjit"),
    any(target_os = "linux", target_os = "macos")
))]
#[test]
fn on_the_fly_unprofiled_lc_and_contracted_match_profiles_and_allocate_nothing() {
    for contracted in [false, true] {
        let contraction = contracted.then(|| {
            super::on_the_fly_load::LoadedOnTheFlyColorContractionV1 {
                plan: crate::recurrence::RecurrenceColorContraction::expanded_identity_for_runtime_test(),
                destination_by_owner_ordinal: vec![0].into_boxed_slice(),
                point_tile_size: 1,
            }
        });
        let mut runtime = scalar_on_the_fly_native_runtime_with_color_contraction(contraction);
        if contracted {
            runtime.runtime.color_accuracy = "full".to_string();
        }
        let label = if contracted { "contracted" } else { "LC" };
        let point_count = 2;
        let momenta = vec![0.0; point_count * 2 * 4];
        let helicities = ["h:+0,+0".to_string()];
        let color = ["flow:singlet".to_string()];
        let colors = (!contracted).then_some(color.as_slice());
        let mut output = vec![f64::NAN; point_count];

        runtime
            .evaluate_f64_into_with_selectors(
                &momenta,
                point_count,
                Some(&helicities),
                colors,
                None,
                None,
                &mut output,
            )
            .unwrap();
        let expected = output.clone();
        let profiled = runtime
            .evaluate_f64_profile(&momenta, point_count, Some(&helicities), colors)
            .unwrap();
        assert!(
            profiled
                .values
                .iter()
                .zip(&expected)
                .all(|(candidate, expected)| candidate.to_bits() == expected.to_bits()),
            "unprofiled {label} values differ bitwise from the diagnostic profile"
        );
        assert!(
            profiled.profile.total_s.is_finite() && profiled.profile.total_s >= 0.0,
            "{label} profile produced invalid timing"
        );
        assert!(
            profiled.profile.recurrence_schedule_execution_count > 0,
            "{label} profile lost schedule counts"
        );
        assert!(
            profiled.profile.recurrence_source_call_count > 0,
            "{label} profile lost source-call counts"
        );
        assert!(
            profiled.profile.recurrence_closure_call_count > 0,
            "{label} profile lost closure-call counts"
        );
        let NativeExecutionLane::OnTheFly(execution) = &runtime.execution_lane else {
            panic!("test runtime changed execution lane");
        };
        let selected_census = execution.active_family_prepared_census().unwrap();
        assert_eq!(
            selected_census.query_count, 1,
            "{label} evaluated unselected helicities"
        );
        assert_eq!(selected_census.union_amplitude_destination_count, 1);

        runtime
            .evaluate_f64_into_with_selectors(
                &momenta,
                point_count,
                Some(&helicities),
                colors,
                None,
                None,
                &mut output,
            )
            .unwrap();
        let (result, allocations, bytes) = super::count_allocations(|| {
            runtime.evaluate_f64_into_with_selectors(
                &momenta,
                point_count,
                Some(&helicities),
                colors,
                None,
                None,
                &mut output,
            )
        });
        result.unwrap();
        assert_eq!(output, expected, "warmed {label} values changed");
        assert_eq!(allocations, 0, "warmed {label} OTF evaluation allocated");
        assert_eq!(bytes, 0, "warmed {label} OTF evaluation allocated bytes");
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_runtime_state_census_is_observational() {
    let retained_state = |runtime: &NativeRuntime| {
        let NativeExecutionLane::OnTheFly(lane) = &runtime.execution_lane else {
            panic!("test runtime changed execution lane");
        };
        (
            lane.retained_family_count(),
            lane.retained_selection_count(),
            lane.semantic_executor_binding_count().unwrap(),
        )
    };
    let census = |runtime: &NativeRuntime| {
        runtime
            .on_the_fly_runtime_state_census_json()
            .unwrap()
            .expect("on-the-fly runtime census is absent")
    };

    let mut runtime = scalar_on_the_fly_native_runtime();
    let metadata = runtime.metadata();
    assert_eq!(
        metadata.on_the_fly_requested_query_construction_threads,
        Some(1)
    );
    assert_eq!(
        metadata.on_the_fly_effective_query_construction_threads,
        Some(1)
    );
    let cold_state = retained_state(&runtime);
    let cold_census = census(&runtime);
    assert_eq!(census(&runtime), cold_census);
    assert_eq!(retained_state(&runtime), cold_state);

    let point_count = 2;
    let momenta = vec![0.0; point_count * 2 * 4];
    let first = runtime
        .evaluate_f64_with_selectors(&momenta, point_count, None, None, None, None)
        .unwrap();
    let warm_state = retained_state(&runtime);
    let warm_census = census(&runtime);
    assert_eq!(census(&runtime), warm_census);
    assert_eq!(retained_state(&runtime), warm_state);

    let repeated = runtime
        .evaluate_f64_with_selectors(&momenta, point_count, None, None, None, None)
        .unwrap();
    assert_eq!(repeated, first);
    assert_eq!(census(&runtime), warm_census);
    assert_eq!(retained_state(&runtime), warm_state);
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_contracted_color_selectors_fail_at_the_common_public_boundary() {
    let mut runtime = scalar_on_the_fly_native_runtime();
    // Selector capability validation is intentionally lane-neutral and keyed
    // by authenticated public accuracy.  No contracted request may reach the
    // compact structural planner.
    runtime.runtime.color_accuracy = "full".to_string();
    let momenta = vec![0.0; 2 * 4];
    let color_by_point = [0_u32];
    let error = runtime
        .evaluate_f64_with_selectors(&momenta, 1, None, None, None, Some(&color_by_point))
        .unwrap_err();
    assert!(error.to_string().contains("only for LC"));

    let color_ids = ["color:contracted".to_string()];
    let error = runtime.resolved_shape(None, Some(&color_ids)).unwrap_err();
    assert!(error.to_string().contains("only for LC"));
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_contracted_later_tile_failure_clears_both_selection_caches() {
    let contraction = super::on_the_fly_load::LoadedOnTheFlyColorContractionV1 {
        plan: crate::recurrence::RecurrenceColorContraction::expanded_identity_for_runtime_test(),
        destination_by_owner_ordinal: vec![0].into_boxed_slice(),
        point_tile_size: 1,
    };
    let mut runtime = scalar_on_the_fly_native_runtime_with_color_contraction(Some(contraction));
    runtime.runtime.color_accuracy = "full".to_string();
    let point_count = 2;
    let momenta = vec![0.0; point_count * 2 * 4];
    let selection_a = ["h:+0,+0".to_string()];
    let selection_b = ["h:+1,+1".to_string()];

    let a_before = runtime
        .evaluate_f64_with_selectors(&momenta, point_count, Some(&selection_a), None, None, None)
        .unwrap();
    let NativeExecutionLane::OnTheFly(execution) = &mut runtime.execution_lane else {
        panic!("test runtime changed execution lane");
    };
    assert_eq!(execution.retained_family_count(), 1);
    assert_eq!(execution.retained_selection_count(), 1);
    // Selection A is retained on both sides. For replacement B, attempt zero
    // commits B's cold family on tile one; attempt one fails before tile two,
    // while B's public wrapper selection is still pending.
    execution.fail_contracted_execution_at_for_test(Some(1));

    let error = runtime
        .evaluate_f64_with_selectors(&momenta, point_count, Some(&selection_b), None, None, None)
        .unwrap_err();
    assert!(error.to_string().contains("injected"));

    let NativeExecutionLane::OnTheFly(execution) = &mut runtime.execution_lane else {
        panic!("test runtime changed execution lane");
    };
    assert_eq!(execution.retained_family_count(), 0);
    assert_eq!(execution.retained_selection_count(), 0);
    execution.fail_contracted_execution_at_for_test(None);

    let a_after = runtime
        .evaluate_f64_with_selectors(&momenta, point_count, Some(&selection_a), None, None, None)
        .unwrap();
    assert_eq!(a_after, a_before);
    let NativeExecutionLane::OnTheFly(execution) = &runtime.execution_lane else {
        panic!("test runtime changed execution lane");
    };
    assert_eq!(execution.retained_family_count(), 1);
    assert_eq!(execution.retained_selection_count(), 1);
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_contracted_streaming_bounds_live_outcomes_and_retains_one_family() {
    let contraction = super::on_the_fly_load::LoadedOnTheFlyColorContractionV1 {
        plan: crate::recurrence::RecurrenceColorContraction::expanded_identity_for_runtime_test(),
        destination_by_owner_ordinal: vec![0].into_boxed_slice(),
        point_tile_size: 1,
    };
    let mut runtime = scalar_on_the_fly_native_runtime_with_color_contraction(Some(contraction));
    runtime.runtime.color_accuracy = "full".to_string();
    let momenta = vec![0.0; 2 * 4];

    let values = runtime
        .evaluate_f64_with_selectors(&momenta, 1, None, None, None, None)
        .unwrap();
    assert_eq!(values.len(), 1);
    let NativeExecutionLane::OnTheFly(execution) = &runtime.execution_lane else {
        panic!("test runtime changed execution lane");
    };
    // The complete scalar selector domain contains four helicities.  With a
    // one-thread construction pool, only one query outcome may be alive at a
    // time even though the complete H x S projection remains reusable.
    assert_eq!(execution.contracted_max_live_query_outcomes_for_test(), 1);
    assert_eq!(execution.retained_family_count(), 1);
    assert_eq!(execution.retained_selection_count(), 1);
    assert!(execution.active_family_prepared_census().is_some());
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_warm_up_reports_ordered_phases_and_reuses_the_committed_family() {
    let contraction = super::on_the_fly_load::LoadedOnTheFlyColorContractionV1 {
        plan: crate::recurrence::RecurrenceColorContraction::expanded_identity_for_runtime_test(),
        destination_by_owner_ordinal: vec![0].into_boxed_slice(),
        point_tile_size: 1,
    };
    let mut runtime = scalar_on_the_fly_native_runtime_with_color_contraction(Some(contraction));
    runtime.runtime.color_accuracy = "full".to_string();
    let momenta = vec![0.0; 2 * 4];
    let mut events = Vec::new();
    let mut observer = |event: &NativeOnTheFlyWarmUpEvent| {
        assert_eq!(event.schema_version, 1);
        assert!(event.completed <= event.total);
        assert!(event.elapsed_seconds.is_finite() && event.elapsed_seconds >= 0.0);
        events.push((event.kind, event.stage, event.completed, event.total));
        Ok(true)
    };
    let observer: &mut NativeOnTheFlyWarmUpObserver<'_> = &mut observer;
    let first = runtime
        .warm_up_on_the_fly_f64_with_selectors(&momenta, None, None, None, Some(observer))
        .unwrap();
    assert!(!first.already_warm);
    assert!(first.first_evaluation_completed);
    assert_eq!(first.query_count, 4);
    assert_eq!(first.warmed_query_count, 4);
    assert_eq!(
        events
            .iter()
            .map(|(_, stage, _, _)| *stage)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            NativeOnTheFlyWarmUpStage::ProcessPreparation,
            NativeOnTheFlyWarmUpStage::QueryFamily,
            NativeOnTheFlyWarmUpStage::FamilyFinalization,
            NativeOnTheFlyWarmUpStage::FirstEvaluation,
        ])
    );
    let stage_ordinals = events
        .iter()
        .map(|(_, stage, _, _)| match stage {
            NativeOnTheFlyWarmUpStage::ProcessPreparation => 0,
            NativeOnTheFlyWarmUpStage::QueryFamily => 1,
            NativeOnTheFlyWarmUpStage::FamilyFinalization => 2,
            NativeOnTheFlyWarmUpStage::FirstEvaluation => 3,
        })
        .collect::<Vec<_>>();
    assert!(stage_ordinals.windows(2).all(|pair| pair[0] <= pair[1]));

    let repeated = runtime
        .warm_up_on_the_fly_f64_with_selectors(&momenta, None, None, None, None)
        .unwrap();
    assert!(repeated.already_warm);
    assert_eq!(repeated.query_count, first.query_count);
    assert_eq!(repeated.warmed_query_count, 0);
    let values = runtime
        .evaluate_f64_with_selectors(&momenta, 1, None, None, None, None)
        .unwrap();
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_lc_warm_up_streams_intermediate_progress_and_matches_ordinary_evaluation() {
    let momenta = vec![0.0; 2 * 4];
    let mut warmed = scalar_on_the_fly_native_runtime();
    let mut events = Vec::new();
    let mut observer = |event: &NativeOnTheFlyWarmUpEvent| {
        events.push(event.clone());
        Ok(true)
    };
    let observer: &mut NativeOnTheFlyWarmUpObserver<'_> = &mut observer;
    let result = warmed
        .warm_up_on_the_fly_f64_with_selectors(&momenta, None, None, None, Some(observer))
        .unwrap();
    assert!(result.query_count > 1);
    assert!(events.iter().any(|event| {
        event.kind == NativeOnTheFlyWarmUpEventKind::Update
            && event.stage == NativeOnTheFlyWarmUpStage::QueryFamily
            && event.completed > 0
            && event.completed < event.total
    }));

    let mut per_stage = BTreeMap::<NativeOnTheFlyWarmUpStage, (u64, u64)>::new();
    for event in &events {
        assert!(event.completed <= event.total);
        let (total, last_completed) = per_stage
            .entry(event.stage)
            .or_insert((event.total, event.completed));
        assert_eq!(*total, event.total, "warm-up stage total changed");
        assert!(
            event.completed >= *last_completed,
            "warm-up progress regressed"
        );
        *last_completed = event.completed;
    }

    let warmed_value = warmed
        .evaluate_f64_with_selectors(&momenta, 1, None, None, None, None)
        .unwrap();
    let mut ordinary = scalar_on_the_fly_native_runtime();
    let ordinary_value = ordinary
        .evaluate_f64_with_selectors(&momenta, 1, None, None, None, None)
        .unwrap();
    assert_eq!(warmed_value.len(), ordinary_value.len());
    assert!(
        warmed_value
            .iter()
            .zip(&ordinary_value)
            .all(|(warm, ordinary)| warm.to_bits() == ordinary.to_bits())
    );
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_warm_up_core_override_preserves_nonzero_results_and_cache() {
    for color_accuracy in ["lc", "nlc", "full"] {
        let make_runtime = || {
            let contraction = (color_accuracy != "lc").then(|| {
                super::on_the_fly_load::LoadedOnTheFlyColorContractionV1 {
                    plan: crate::recurrence::RecurrenceColorContraction::expanded_identity_for_runtime_test(),
                    destination_by_owner_ordinal: vec![0].into_boxed_slice(),
                    point_tile_size: 1,
                }
            });
            let mut runtime = scalar_on_the_fly_native_runtime_with_color_contraction(contraction);
            runtime.runtime.color_accuracy = color_accuracy.to_string();
            runtime
        };
        let momenta = [0.0; 2 * 4];
        let mut serial = make_runtime();
        serial
            .warm_up_f64(&momenta, None, None, Some(1), None)
            .unwrap();
        let expected = serial
            .evaluate_f64_with_selectors(&momenta, 1, None, None, None, None)
            .unwrap();
        assert!(
            expected
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );

        let mut parallel = make_runtime();
        let effective = super::on_the_fly_load::effective_query_construction_threads(4);
        let caller = std::thread::current().id();
        let mut events = Vec::new();
        let mut observer = |event: &NativeOnTheFlyWarmUpEvent| {
            assert_eq!(std::thread::current().id(), caller);
            assert_eq!(event.workers, effective as u64);
            events.push(event.clone());
            Ok(true)
        };
        let first = parallel
            .warm_up_f64(&momenta, None, None, Some(4), Some(&mut observer))
            .unwrap();
        assert!(!first.already_warm);
        assert_eq!(first.query_count, 4);
        assert!(!events.is_empty());
        let actual = parallel
            .evaluate_f64_with_selectors(&momenta, 1, None, None, None, None)
            .unwrap();
        assert_eq!(actual, expected);
        let NativeExecutionLane::OnTheFly(serial_lane) = &serial.execution_lane else {
            panic!("test runtime changed execution lane");
        };
        let NativeExecutionLane::OnTheFly(parallel_lane) = &parallel.execution_lane else {
            panic!("test runtime changed execution lane");
        };
        assert_eq!(
            parallel_lane.active_family_prepared_census(),
            serial_lane.active_family_prepared_census(),
        );
        if color_accuracy != "lc" {
            assert_eq!(
                parallel_lane.contracted_max_live_query_outcomes_for_test(),
                effective.min(4),
            );
        }
        let metadata = parallel.metadata();
        assert_eq!(
            metadata.on_the_fly_requested_query_construction_threads,
            Some(1)
        );
        assert_eq!(
            metadata.on_the_fly_effective_query_construction_threads,
            Some(1)
        );

        // A different override and then the process default must both reuse
        // the exact committed family, rather than reconstructing it.
        for n_cores in [Some(1), None] {
            let mut observer = |event: &NativeOnTheFlyWarmUpEvent| {
                assert_eq!(event.workers, 1);
                Ok(true)
            };
            let repeated = parallel
                .warm_up_f64(&momenta, None, None, n_cores, Some(&mut observer))
                .unwrap();
            assert!(repeated.already_warm);
            assert_eq!(repeated.warmed_query_count, 0);
        }
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_warm_up_rejects_zero_cores_without_changing_cache() {
    let mut runtime = scalar_on_the_fly_native_runtime();
    let momenta = [0.0; 2 * 4];
    let cold = runtime.on_the_fly_runtime_state_census_json().unwrap();
    let error = runtime
        .warm_up_f64(&momenta, None, None, Some(0), None)
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
    assert!(error.to_string().contains("n_cores must be positive"));
    assert_eq!(
        runtime.on_the_fly_runtime_state_census_json().unwrap(),
        cold
    );

    runtime
        .warm_up_f64(&momenta, None, None, Some(2), None)
        .unwrap();
    let warm = runtime.on_the_fly_runtime_state_census_json().unwrap();
    let error = runtime
        .warm_up_f64(&momenta, None, None, Some(0), None)
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
    assert_eq!(
        runtime.on_the_fly_runtime_state_census_json().unwrap(),
        warm
    );
    let repeated = runtime
        .warm_up_f64(&momenta, None, None, None, None)
        .unwrap();
    assert!(repeated.already_warm);
    assert_eq!(repeated.warmed_query_count, 0);
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_contracted_midstream_warm_up_cancellation_rolls_back_to_retained_selection() {
    let contraction = super::on_the_fly_load::LoadedOnTheFlyColorContractionV1 {
        plan: crate::recurrence::RecurrenceColorContraction::expanded_identity_for_runtime_test(),
        destination_by_owner_ordinal: vec![0].into_boxed_slice(),
        point_tile_size: 1,
    };
    let mut runtime = scalar_on_the_fly_native_runtime_with_color_contraction(Some(contraction));
    runtime.runtime.color_accuracy = "full".to_string();
    let momenta = vec![0.0; 2 * 4];
    let retained_helicity = ["h:+0,+0".to_string()];
    runtime
        .warm_up_on_the_fly_f64_with_selectors(&momenta, Some(&retained_helicity), None, None, None)
        .unwrap();
    let expected = runtime
        .evaluate_f64_with_selectors(&momenta, 1, Some(&retained_helicity), None, None, None)
        .unwrap();

    let saw_midstream = std::cell::Cell::new(false);
    let mut observer = |event: &NativeOnTheFlyWarmUpEvent| {
        if event.kind == NativeOnTheFlyWarmUpEventKind::Update
            && event.stage == NativeOnTheFlyWarmUpStage::QueryFamily
            && event.completed < event.total
        {
            saw_midstream.set(true);
            return Ok(false);
        }
        Ok(true)
    };
    let observer: &mut NativeOnTheFlyWarmUpObserver<'_> = &mut observer;
    let error = runtime
        .warm_up_on_the_fly_f64_with_selectors(&momenta, None, None, Some(2), Some(observer))
        .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
    assert!(saw_midstream.get());
    let retained = runtime
        .evaluate_f64_with_selectors(&momenta, 1, Some(&retained_helicity), None, None, None)
        .unwrap();
    assert_eq!(retained, expected);
    let NativeExecutionLane::OnTheFly(execution) = &runtime.execution_lane else {
        panic!("test runtime changed execution lane");
    };
    assert_eq!(execution.retained_family_count(), 1);
    assert_eq!(execution.retained_selection_count(), 1);
    assert_eq!(execution.pending_family_count(), 0);
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_warm_up_terminal_notification_is_non_cancellable_after_commit() {
    let momenta = vec![0.0; 2 * 4];
    let mut runtime = scalar_on_the_fly_native_runtime();
    let terminal_delivery_count = std::cell::Cell::new(0);
    let mut failing_terminal = |event: &NativeOnTheFlyWarmUpEvent| {
        if event.kind == NativeOnTheFlyWarmUpEventKind::End
            && event.stage == NativeOnTheFlyWarmUpStage::FirstEvaluation
        {
            terminal_delivery_count.set(terminal_delivery_count.get() + 1);
            return Err(RusticolError::evaluation(
                "injected terminal notification failure",
            ));
        }
        Ok(true)
    };
    let observer: &mut NativeOnTheFlyWarmUpObserver<'_> = &mut failing_terminal;
    let first = runtime
        .warm_up_on_the_fly_f64_with_selectors(&momenta, None, None, None, Some(observer))
        .unwrap();
    assert!(first.first_evaluation_completed);
    assert_eq!(terminal_delivery_count.get(), 1);

    let mut rejecting_terminal = |event: &NativeOnTheFlyWarmUpEvent| {
        Ok(!(event.kind == NativeOnTheFlyWarmUpEventKind::End
            && event.stage == NativeOnTheFlyWarmUpStage::FirstEvaluation))
    };
    let observer: &mut NativeOnTheFlyWarmUpObserver<'_> = &mut rejecting_terminal;
    let repeated = runtime
        .warm_up_on_the_fly_f64_with_selectors(&momenta, None, None, None, Some(observer))
        .unwrap();
    assert!(repeated.already_warm);
    assert_eq!(repeated.warmed_query_count, 0);
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_warm_up_native_boundary_rejects_invalid_lane_and_selectors() {
    let mut compiled = zero_native_runtime();
    let compiled_momenta = vec![0.0; compiled.external_count() * 4];
    let error = compiled
        .warm_up_on_the_fly_f64_with_selectors(&compiled_momenta, None, None, None, None)
        .unwrap_err();
    assert_eq!(
        error.kind(),
        crate::RusticolErrorKind::UnsupportedRuntimeCapability
    );

    let contraction = super::on_the_fly_load::LoadedOnTheFlyColorContractionV1 {
        plan: crate::recurrence::RecurrenceColorContraction::expanded_identity_for_runtime_test(),
        destination_by_owner_ordinal: vec![0].into_boxed_slice(),
        point_tile_size: 1,
    };
    let mut contracted = scalar_on_the_fly_native_runtime_with_color_contraction(Some(contraction));
    contracted.runtime.color_accuracy = "full".to_string();
    let momenta = vec![0.0; contracted.external_count() * 4];
    let color = ["color:contracted".to_string()];
    let error = contracted
        .warm_up_on_the_fly_f64_with_selectors(&momenta, None, Some(&color), None, None)
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Selector);

    let mut lc = scalar_on_the_fly_native_runtime();
    let unknown_helicity = ["h:not-a-helicity".to_string()];
    let error = lc
        .warm_up_on_the_fly_f64_with_selectors(&momenta, Some(&unknown_helicity), None, None, None)
        .unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Selector);
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_warm_up_cancellation_preserves_the_last_committed_selection() {
    let mut runtime = scalar_on_the_fly_native_runtime();
    let momenta = vec![0.0; 2 * 4];
    let color = ["flow:singlet".to_string()];
    let selection_a = ["h:+0,+0".to_string()];
    let selection_b = ["h:+1,+1".to_string()];
    runtime
        .warm_up_on_the_fly_f64_with_selectors(
            &momenta,
            Some(&selection_a),
            Some(&color),
            None,
            None,
        )
        .unwrap();
    let expected = runtime
        .evaluate_f64_with_selectors(&momenta, 1, Some(&selection_a), Some(&color), None, None)
        .unwrap();

    let mut observer = |event: &NativeOnTheFlyWarmUpEvent| {
        Ok(!(event.kind == NativeOnTheFlyWarmUpEventKind::Start
            && event.stage == NativeOnTheFlyWarmUpStage::QueryFamily))
    };
    let observer: &mut NativeOnTheFlyWarmUpObserver<'_> = &mut observer;
    let error = runtime
        .warm_up_on_the_fly_f64_with_selectors(
            &momenta,
            Some(&selection_b),
            Some(&color),
            None,
            Some(observer),
        )
        .unwrap_err();
    assert!(error.to_string().contains("cancelled"));
    let retained = runtime
        .evaluate_f64_with_selectors(&momenta, 1, Some(&selection_a), Some(&color), None, None)
        .unwrap();
    assert_eq!(retained, expected);
    let NativeExecutionLane::OnTheFly(execution) = &runtime.execution_lane else {
        panic!("test runtime changed execution lane");
    };
    assert_eq!(execution.retained_family_count(), 1);
    assert_eq!(execution.retained_selection_count(), 1);
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_warm_up_rejects_more_than_one_point() {
    let mut runtime = scalar_on_the_fly_native_runtime();
    let error = runtime
        .warm_up_on_the_fly_f64_with_selectors(&[0.0; 2 * 2 * 4], None, None, None, None)
        .unwrap_err();
    assert!(error.to_string().contains("exactly one point"));
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[test]
fn on_the_fly_public_paths_vectorize_with_a_last_family_only_cache() {
    let retained = |runtime: &NativeRuntime| {
        let NativeExecutionLane::OnTheFly(lane) = &runtime.execution_lane else {
            panic!("test runtime changed execution lane");
        };
        (
            lane.retained_family_count(),
            lane.retained_selection_count(),
            lane.semantic_executor_binding_count().unwrap(),
        )
    };
    let census = |runtime: &NativeRuntime| -> Value {
        serde_json::from_str(
            runtime
                .on_the_fly_runtime_state_census_json()
                .unwrap()
                .as_deref()
                .expect("on-the-fly runtime census is absent"),
        )
        .unwrap()
    };
    let prepared_census = |runtime: &NativeRuntime| {
        let NativeExecutionLane::OnTheFly(lane) = &runtime.execution_lane else {
            panic!("test runtime changed execution lane");
        };
        lane.active_family_prepared_census()
            .expect("active on-the-fly family has no prepared census")
    };

    let mut runtime = scalar_on_the_fly_native_runtime();
    let point_count = 6;
    let momenta = vec![0.0; point_count * 2 * 4];
    let cold_census = census(&runtime);
    assert_eq!(
        cold_census["kind"],
        "rusticol-on-the-fly-runtime-state-census-v1"
    );
    assert_eq!(cold_census["process_id"], "s_to_s");
    assert_eq!(cold_census["family_cache_policy"], "last-family-only");
    assert_eq!(cold_census["family_cache_limit"], 1);
    assert_eq!(cold_census["process_preparation_count"], 0);
    assert_eq!(cold_census["retained_family_count"], 0);
    assert_eq!(cold_census["pending_family_count"], 0);
    assert_eq!(cold_census["retained_selection_count"], 0);
    assert_eq!(cold_census["semantic_executor_binding_count"], 0);
    assert!(cold_census["active_family_union_census"].is_null());

    // A: the complete compact selector domain. Only its first helicity is
    // executable; the other three are authenticated selector-local zeros.
    let global = runtime
        .evaluate_f64_with_selectors(&momenta, point_count, None, None, None, None)
        .unwrap();
    assert_eq!(global.len(), point_count);
    assert!(
        global
            .iter()
            .all(|value| value.is_finite() && value.abs() > f64::EPSILON)
    );
    let (family_count, selection_count, binding_count) = retained(&runtime);
    assert_eq!((family_count, selection_count), (1, 1));
    assert!(binding_count > 0);
    let global_census = census(&runtime);
    assert_eq!(global_census["process_preparation_count"], 1);
    assert_eq!(global_census["retained_family_count"], 1);
    assert_eq!(global_census["pending_family_count"], 0);
    assert_eq!(global_census["retained_selection_count"], 1);
    assert_eq!(global_census["retained_request_count"], 4);
    assert_eq!(global_census["retained_amplitude_destination_count"], 1);
    assert_eq!(
        global_census["retained_request_count"].as_u64().unwrap()
            - global_census["retained_amplitude_destination_count"]
                .as_u64()
                .unwrap(),
        3
    );
    let active = &global_census["active_family_union_census"];
    assert_eq!(active["basis"], "shared-query-family-union-v1");
    assert_eq!(active["scope"], "active-family-union");
    let prepared = prepared_census(&runtime);
    macro_rules! assert_mapped {
        ($($field:ident),+ $(,)?) => {
            $(assert_eq!(
                active[stringify!($field)].as_u64(),
                Some(u64::from(prepared.$field)),
                stringify!($field),
            );)+
        };
    }
    assert_mapped!(
        query_count,
        union_unique_current_count,
        union_unique_current_component_count,
        union_source_rows,
        union_contribution_rows,
        union_finalization_rows,
        union_closure_rows,
        union_amplitude_destination_count,
        union_source_executor_call_groups,
        union_contribution_executor_call_groups,
        union_finalization_executor_call_groups,
        union_closure_executor_call_groups,
    );
    assert!(prepared.union_unique_current_count <= prepared.union_unique_current_component_count);
    for role in ["source", "contribution", "finalization", "closure"] {
        let groups = active[format!("union_{role}_executor_call_groups")]
            .as_u64()
            .unwrap();
        let rows = active[format!("union_{role}_rows")].as_u64().unwrap();
        assert!(groups <= rows, "{role}");
    }

    // B: one explicit nonzero selector. Switching A -> B -> A retains only
    // the current public mapping and lower-lane family; evicted A is cold.
    let helicity = vec!["h:+0,+0".to_string()];
    let color = vec!["flow:singlet".to_string()];
    let selected = runtime
        .evaluate_f64_with_selectors(
            &momenta,
            point_count,
            Some(&helicity),
            Some(&color),
            None,
            None,
        )
        .unwrap();
    assert_eq!(selected, global);
    assert_eq!(retained(&runtime).0, 1);
    assert_eq!(retained(&runtime).1, 1);
    // The output retains all four physical helicities, but this selected
    // call prepares only one query, rather than evaluating all and filtering.
    let selected_census = census(&runtime);
    assert_eq!(selected_census["retained_request_count"], 1);
    assert_eq!(selected_census["retained_amplitude_destination_count"], 1);
    assert_eq!(
        selected_census["active_family_union_census"]["query_count"],
        1
    );

    let global_again = runtime
        .evaluate_f64_with_selectors(&momenta, point_count, None, None, None, None)
        .unwrap();
    assert_eq!(global_again, global);
    assert_eq!(retained(&runtime).0, 1);
    assert_eq!(retained(&runtime).1, 1);
    let revisited_census = census(&runtime);
    assert_eq!(revisited_census["process_preparation_count"], 1);
    assert_eq!(revisited_census["retained_family_count"], 1);
    assert_eq!(revisited_census["pending_family_count"], 0);
    assert_eq!(revisited_census["retained_selection_count"], 1);
    assert_eq!(
        revisited_census["active_family_union_census"],
        global_census["active_family_union_census"]
    );

    let resolved = runtime
        .evaluate_resolved_f64(&momenta, point_count, None, None)
        .unwrap();
    assert_eq!(resolved.shape(), (point_count, 4, 1));
    assert_eq!(resolved.totals(), global);
    assert_eq!(
        resolved.helicity_ids,
        ["h:+0,+0", "h:+0,+1", "h:+1,+0", "h:+1,+1"]
    );
    assert_eq!(resolved.color_ids, color);
    assert_eq!(retained(&runtime).0, 1);
    assert_eq!(retained(&runtime).1, 1);

    // Alternating selectors form two stable three-point partitions. The
    // nonzero partition reaches each prepared row group once, while the
    // selector-local-zero partition remains exactly zero.
    let helicity_by_point = [0, 3, 0, 3, 0, 3];
    let color_by_point = [0; 6];
    let expected_partitioned = global
        .iter()
        .enumerate()
        .map(|(index, value)| if index % 2 == 0 { *value } else { 0.0 })
        .collect::<Vec<_>>();
    let per_point = runtime
        .evaluate_f64_with_selectors(
            &momenta,
            point_count,
            None,
            None,
            Some(&helicity_by_point),
            Some(&color_by_point),
        )
        .unwrap();
    assert_eq!(per_point, expected_partitioned);
    assert_eq!(retained(&runtime).0, 1);
    assert_eq!(retained(&runtime).1, 1);
    let partitioned_census = census(&runtime);

    let per_point_again = runtime
        .evaluate_f64_with_selectors(
            &momenta,
            point_count,
            None,
            None,
            Some(&helicity_by_point),
            Some(&color_by_point),
        )
        .unwrap();
    assert_eq!(per_point_again, expected_partitioned);
    assert_eq!(retained(&runtime).0, 1);
    assert_eq!(retained(&runtime).1, 1);
    assert_eq!(census(&runtime), partitioned_census);

    let profiled = runtime
        .evaluate_f64_profile_with_selectors(
            &momenta,
            point_count,
            None,
            None,
            Some(&helicity_by_point),
            Some(&color_by_point),
        )
        .unwrap();
    assert_eq!(profiled.values, expected_partitioned);
    assert_eq!(profiled.profile.selector_plan_kind, "stable-grouped");
    assert_eq!(profiled.profile.selector_group_sizes, [3, 3]);
    assert_eq!(profiled.profile.recurrence_schedule_execution_count, 2);
    assert_eq!(
        profiled.profile.recurrence_momentum_scalar_value_count,
        u64::try_from(point_count * 2 * 4).unwrap()
    );
    assert!(profiled.profile.recurrence_source_call_count > 0);
    assert!(profiled.profile.recurrence_closure_call_count > 0);
    assert!(profiled.profile.recurrence_source_call_count < point_count as u64);
    assert!(profiled.profile.recurrence_closure_call_count < point_count as u64);
    assert!(
        runtime
            .benchmark_f64_wall_time_with_selectors(
                &momenta,
                point_count,
                2,
                None,
                None,
                Some(&helicity_by_point),
                Some(&color_by_point),
            )
            .unwrap()
            >= 0.0
    );
    assert_eq!(retained(&runtime).0, 1);
    assert_eq!(retained(&runtime).1, 1);
    assert_eq!(census(&runtime), partitioned_census);

    assert!(runtime.physics_v1.get().is_err());

    runtime.clear().unwrap();
    assert_eq!(retained(&runtime), (0, 0, 0));
    assert_eq!(census(&runtime), cold_census);
    let NativeExecutionLane::OnTheFly(lane) = &runtime.execution_lane else {
        panic!("test runtime changed execution lane");
    };
    assert_eq!(lane.point_major_scratch_state(), (0, 0));
    assert!(runtime.physics_v1.get().is_err());
    let rebuilt = runtime
        .evaluate_f64_with_selectors(&momenta, point_count, None, None, None, None)
        .unwrap();
    assert_eq!(rebuilt, global);
    let (family_count, selection_count, binding_count) = retained(&runtime);
    assert_eq!((family_count, selection_count), (1, 1));
    assert!(binding_count > 0);
    assert_eq!(census(&runtime), global_census);
}

#[test]
fn recurrence_selector_plan_is_bound_to_the_public_external_ordering() {
    let runtime = zero_native_runtime();
    let plan = NativeRecurrenceSelectorPlan {
        artifact_root: runtime.root.clone(),
        process_key: runtime.process_key.clone(),
        external_permutation: runtime.external_permutation.clone(),
        selected_helicities: None,
        selected_colors: None,
    };
    plan.ensure_matches(&runtime).unwrap();

    let mut reordered_runtime = zero_native_runtime();
    reordered_runtime.external_permutation = vec![1, 0, 2];
    let error = plan.ensure_matches(&reordered_runtime).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Selector);
    assert!(error.to_string().contains("external ordering"));
}

#[test]
fn native_kinematics_json_accepts_one_numeric_or_decimal_string_point() {
    let direct = serde_json::json!([
        ["10.0000000000000001", 0, 0.0, "10"],
        [10, 0, 0, -10],
        [20, "0", 0, 0]
    ]);
    let flat = native_runtime::parse_public_kinematics_point(&direct, 3).unwrap();
    assert_eq!(flat.len(), 12);
    assert_eq!(flat[0], 10.0);
    assert_eq!(flat[3], 10.0);
    assert_eq!(flat[7], -10.0);
    assert_eq!(flat[8], 20.0);

    let singleton = serde_json::json!([direct]);
    assert_eq!(
        native_runtime::parse_public_kinematics_point(&singleton, 3).unwrap(),
        flat
    );
}

#[test]
fn native_kinematics_json_rejects_multiple_points_and_invalid_components() {
    let point = serde_json::json!([[1, 0, 0, 1], [1, 0, 0, -1], [2, 0, 0, 0]]);
    for invalid in [
        serde_json::json!([point.clone(), point]),
        serde_json::json!([[1, 0, 0, 1], [1, 0, 0, -1]]),
        serde_json::json!([[1, 0, 0, true], [1, 0, 0, -1], [2, 0, 0, 0]]),
        serde_json::json!([[1, 0, 0, "1e400"], [1, 0, 0, -1], [2, 0, 0, 0]]),
        serde_json::json!([[1, 0, 1], [1, 0, 0, -1], [2, 0, 0, 0]]),
    ] {
        let error = native_runtime::parse_public_kinematics_point(&invalid, 3).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
    }
}

#[test]
fn native_f64_into_matches_allocating_wrapper_and_validates_output() {
    let point = [
        10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, -10.0, 20.0, 0.0, 0.0, 0.0,
    ];
    let momenta = point.repeat(4);
    let mut runtime = zero_native_runtime();
    let allocated = runtime.evaluate_f64(&momenta, 4).unwrap();
    let mut output = vec![f64::NAN; 4];
    runtime.evaluate_f64_into(&momenta, 4, &mut output).unwrap();
    assert_eq!(output, allocated);

    let helicity_by_point = [0_u32, 1, 2, 0];
    runtime
        .evaluate_f64_into_with_selectors(
            &momenta,
            4,
            None,
            None,
            Some(&helicity_by_point),
            None,
            &mut output,
        )
        .unwrap();
    assert_eq!(output, vec![0.0; 4]);

    let error = runtime
        .evaluate_f64_into(&momenta, 4, &mut [0.0; 3])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("output has length 3, expected 4")
    );
    let error = runtime
        .evaluate_f64_into(&momenta, 4, &mut [0.0; 5])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("output has length 5, expected 4")
    );
}

#[test]
fn native_f64_totals_bypass_point_selector_scratch() {
    let point = [
        10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, -10.0, 20.0, 0.0, 0.0, 0.0,
    ];
    let momenta = point.repeat(4);
    let mut runtime = zero_native_runtime();
    runtime
        .point_selector_scratch
        .planner
        .build(4, Some(&[0, 1, 0, 1]), None, 3, 1)
        .expect("seed point-selector planner");
    let retained_partitions = runtime.point_selector_scratch.planner.partitions().to_vec();
    assert!(!retained_partitions.is_empty());

    let allocated = runtime
        .evaluate_f64(&momenta, 4)
        .expect("evaluate owned totals without selectors");
    let mut output = [f64::NAN; 4];
    runtime
        .evaluate_f64_into(&momenta, 4, &mut output)
        .expect("evaluate borrowed totals without selectors");

    assert_eq!(output.as_slice(), allocated);
    assert_eq!(
        runtime.point_selector_scratch.planner.partitions(),
        retained_partitions,
        "no-selector totals must not clear or rebuild selector planner state"
    );
}

#[test]
fn current_parameter_values_include_complex_and_derived_slots() {
    let mut runtime = empty_generic_runtime();
    runtime.model_parameters = vec![
        GenericRuntimeModelParameterManifest {
            name: "aS".into(),
            kind: "external_parameter".into(),
            parameter_index: 0,
            default: 0.118,
            pdg: None,
            runtime_name: None,
            complex_component: None,
        },
        GenericRuntimeModelParameterManifest {
            name: "derived.real".into(),
            kind: "derived_parameter_component".into(),
            parameter_index: 1,
            default: 1.0,
            pdg: None,
            runtime_name: Some("derived".into()),
            complex_component: Some("real".into()),
        },
        GenericRuntimeModelParameterManifest {
            name: "derived.imag".into(),
            kind: "derived_parameter_component".into(),
            parameter_index: 2,
            default: 0.0,
            pdg: None,
            runtime_name: Some("derived".into()),
            complex_component: Some("imag".into()),
        },
    ];
    runtime.model_parameter_values_f64 = vec![0.101, 2.5, -0.75];
    assert_eq!(
        runtime.current_model_parameter_value("aS").unwrap(),
        (0.101, 0.0)
    );
    assert_eq!(
        runtime.current_model_parameter_value("derived").unwrap(),
        (2.5, -0.75)
    );
    assert!(runtime.current_model_parameter_value("unknown").is_err());
    runtime.model_parameter_values_f64[1] = 7.0;
    assert_eq!(
        runtime.current_model_parameter_value("derived").unwrap(),
        (7.0, -0.75)
    );
    runtime.model_parameter_values_f64.truncate(1);
    assert!(runtime.current_model_parameter_value("derived").is_err());
}

#[test]
fn model_parameter_override_batch_is_atomic() {
    let mut runtime = empty_generic_runtime();
    runtime.model_parameter_runtime_slots.insert(
        "alpha_s".to_string(),
        RuntimeParameterSlots {
            real: 0,
            imaginary: None,
        },
    );
    let invalid_batch = BTreeMap::from([
        ("alpha_s".to_string(), (0.101, 0.0)),
        ("unknown.parameter".to_string(), (1.0, 0.0)),
    ]);

    let error = runtime
        .apply_model_parameter_overrides(&invalid_batch)
        .unwrap_err();

    assert!(error.to_string().contains("unknown.parameter"));
    assert_eq!(runtime.model_parameter_values_f64, vec![0.118]);
    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.101, 0.0))]))
        .unwrap();
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
}

#[test]
fn model_parameter_overrides_are_atomic_across_helicity_lanes() {
    let mut runtime = empty_generic_runtime();
    runtime.model_parameter_runtime_slots.insert(
        "alpha_s".to_string(),
        RuntimeParameterSlots {
            real: 0,
            imaginary: None,
        },
    );
    let mut sum_runtime = empty_generic_runtime();
    sum_runtime.model_parameter_runtime_slots.insert(
        "alpha_s".to_string(),
        RuntimeParameterSlots {
            real: 0,
            imaginary: None,
        },
    );
    runtime.helicity_sum_runtime = Some(Box::new(sum_runtime));

    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.101, 0.0))]))
        .unwrap();

    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert_eq!(
        runtime
            .helicity_sum_runtime
            .as_ref()
            .unwrap()
            .model_parameter_values_f64,
        vec![0.101]
    );

    runtime
        .helicity_sum_runtime
        .as_mut()
        .unwrap()
        .model_parameter_runtime_slots
        .clear();
    let error = runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.089, 0.0))]))
        .unwrap_err();

    assert!(error.to_string().contains("alpha_s"));
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert_eq!(
        runtime
            .helicity_sum_runtime
            .as_ref()
            .unwrap()
            .model_parameter_values_f64,
        vec![0.101]
    );
}

#[test]
fn model_parameter_overrides_are_atomic_across_color_selector_lanes() {
    fn runtime_with_alpha_s() -> ExecutionRuntime {
        let mut runtime = empty_generic_runtime();
        runtime.model_parameter_runtime_slots.insert(
            "alpha_s".to_string(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        );
        runtime
    }

    let mut runtime = runtime_with_alpha_s();
    runtime
        .color_selector_runtimes
        .insert(0, Box::new(runtime_with_alpha_s()));
    runtime
        .color_selector_runtimes
        .insert(1, Box::new(runtime_with_alpha_s()));

    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.101, 0.0))]))
        .unwrap();
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert!(
        runtime
            .color_selector_runtimes
            .values()
            .all(|lane| lane.model_parameter_values_f64 == vec![0.101])
    );

    runtime
        .color_selector_runtimes
        .get_mut(&1)
        .unwrap()
        .model_parameter_runtime_slots
        .clear();
    let error = runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.089, 0.0))]))
        .unwrap_err();

    assert!(error.to_string().contains("alpha_s"));
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert!(
        runtime
            .color_selector_runtimes
            .values()
            .all(|lane| lane.model_parameter_values_f64 == vec![0.101])
    );
}

#[test]
fn model_parameter_overrides_are_atomic_across_shared_helicity_selector_lanes() {
    fn runtime_with_alpha_s() -> ExecutionRuntime {
        let mut runtime = empty_generic_runtime();
        runtime.model_parameter_runtime_slots.insert(
            "alpha_s".to_string(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        );
        runtime
    }

    let mut runtime = runtime_with_alpha_s();
    runtime
        .helicity_selector_runtimes
        .push(Box::new(runtime_with_alpha_s()));
    runtime
        .helicity_selector_runtime_schedule_modes
        .push(HelicitySelectorScheduleMode::ParentClosure);
    runtime
        .helicity_selector_runtimes
        .push(Box::new(runtime_with_alpha_s()));
    runtime
        .helicity_selector_runtime_schedule_modes
        .push(HelicitySelectorScheduleMode::ParentClosure);
    runtime.helicity_selector_lane_by_domain = BTreeMap::from([(0, 0), (1, 0), (2, 1)]);

    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.101, 0.0))]))
        .unwrap();
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert_eq!(runtime.helicity_selector_runtimes.len(), 2);
    assert_eq!(runtime.helicity_selector_lane_by_domain.len(), 3);
    assert!(
        runtime
            .helicity_selector_runtimes
            .iter()
            .all(|lane| lane.model_parameter_values_f64 == vec![0.101])
    );

    runtime.helicity_selector_runtimes[1]
        .model_parameter_runtime_slots
        .clear();
    let error = runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.089, 0.0))]))
        .unwrap_err();

    assert!(error.to_string().contains("alpha_s"));
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert!(
        runtime
            .helicity_selector_runtimes
            .iter()
            .all(|lane| lane.model_parameter_values_f64 == vec![0.101])
    );
}

#[test]
fn alias_external_order_initialization_reaches_shared_helicity_selector_lanes() {
    let mut runtime = empty_generic_runtime();
    runtime
        .helicity_selector_runtimes
        .push(Box::new(empty_generic_runtime()));
    runtime
        .helicity_selector_runtime_schedule_modes
        .push(HelicitySelectorScheduleMode::ParentClosure);
    runtime.helicity_selector_lane_by_domain = BTreeMap::from([(0, 0), (1, 0)]);

    runtime.set_external_pdg_order_recursive(&[1, -1, 23]);

    assert_eq!(runtime.external_pdg_order, vec![1, -1, 23]);
    assert_eq!(runtime.helicity_selector_runtimes.len(), 1);
    assert_eq!(
        runtime.helicity_selector_runtimes[0].external_pdg_order,
        vec![1, -1, 23]
    );
}

#[test]
fn model_parameter_override_rejects_mass_class_changes_atomically() {
    let mut runtime = empty_generic_runtime();
    runtime.model_parameter_values_f64 = vec![91.188];
    runtime.model_parameter_runtime_slots.insert(
        "MZ".to_string(),
        RuntimeParameterSlots {
            real: 0,
            imaginary: None,
        },
    );
    runtime
        .particle_mass_parameter_names
        .insert(23, "MZ".to_string());
    runtime.particle_masses.insert(23, 91.188);

    let error = runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("MZ".to_string(), (0.0, 0.0))]))
        .unwrap_err();

    assert!(error.to_string().contains("mass class"));
    assert!(error.to_string().contains("regenerate"));
    assert_eq!(runtime.model_parameter_values_f64, vec![91.188]);
    assert_eq!(runtime.particle_masses.get(&23), Some(&91.188));

    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("MZ".to_string(), (100.0, 0.0))]))
        .expect("massive-to-massive update remains valid");
    assert_eq!(runtime.model_parameter_values_f64, vec![100.0]);
    assert_eq!(runtime.particle_masses.get(&23), Some(&100.0));
}

#[test]
fn compiled_color_topology_lane_requires_physics_reduction() {
    let mut manifest: ExecutionManifest = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../src/pyamplicol/assets/selftest/portable-64le/artifact/processes/",
        "d_dbar_to_z/execution.json"
    )))
    .expect("parse packaged execution fixture");
    let mut lane = manifest.clone();
    lane.physics_reduction = None;
    lane.helicity_sum_execution = None;
    lane.helicity_selector_executions.clear();
    lane.color_selector_executions.clear();
    manifest
        .color_selector_executions
        .push(ColorSelectorExecutionManifest {
            materialized_sector_id: 0,
            execution: Box::new(lane),
        });

    let error = ExecutionRuntime::from_manifest(manifest)
        .err()
        .expect("color topology lane without reduction must fail");

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(
        error.to_string().contains("has no reduction override"),
        "unexpected error: {error}"
    );
}

#[test]
fn eager_native_profile_accepts_non_overlapping_top_level_phases() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 10.0e-3;
    profile.source_fill_s = 1.0e-3;
    profile.momentum_input_setup_s = 0.5e-3;
    profile.momentum_setup_s = 0.5e-3;
    profile.stage_evaluator_call_s = 7.0e-3;
    profile.eager_initialize_s = 0.5e-3;
    profile.eager_kernel_call_s = 6.0e-3;
    profile.eager_copy_out_s = 0.5e-3;

    profile.validate_eager_top_level_accounting().unwrap();
}

#[test]
fn native_profile_preserves_legacy_momentum_setup_aggregate() {
    let profile: NativeRuntimeProfile = RuntimeProfile {
        momentum_input_setup_s: 0.25,
        momentum_setup_s: 0.75,
        model_parameter_setup_s: 0.5,
        ..RuntimeProfile::default()
    }
    .into();

    assert_eq!(profile.momentum_input_setup_s, 0.25);
    assert_eq!(profile.momentum_setup_s, 0.75);
    assert_eq!(profile.model_parameter_setup_s, 0.5);
}

#[test]
fn eager_native_profile_rejects_top_level_overlap() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 10.0e-3;
    profile.source_fill_s = 4.0e-3;
    profile.stage_evaluator_call_s = 7.0e-3;

    let error = profile.validate_eager_top_level_accounting().unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
    assert!(error.to_string().contains("exclusive top-level phases"));
    assert!(error.to_string().contains("exceeding wall time"));
}

#[test]
fn recurrence_native_profile_accepts_nested_schedule_attribution() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 12.0e-3;
    profile.source_fill_s = 0.5e-3;
    profile.momentum_input_setup_s = 0.5e-3;
    profile.recurrence_momentum_fill_s = 1.0e-3;
    profile.recurrence_union_source_fill_s = 0.5e-3;
    profile.recurrence_schedule_s = 8.0e-3;
    profile.recurrence_source_kernel_s = 0.5e-3;
    profile.recurrence_contribution_kernel_s = 5.0e-3;
    profile.recurrence_finalization_s = 1.0e-3;
    profile.recurrence_closure_s = 0.5e-3;
    profile.recurrence_replay_output_mapping_s = 0.5e-3;
    profile.reduction_s = 0.5e-3;

    profile.validate_recurrence_top_level_accounting().unwrap();
}

#[test]
fn recurrence_native_profile_keeps_contracted_replay_schedule_out_of_reduction() {
    // A contracted replay owns three distinct top-level intervals: momentum
    // fill, schedule execution, and the post-schedule copy/contraction.  The
    // latter must not start before the scheduler call or it counts the
    // schedule twice and can exceed the enclosing wall clock.
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 26.959e-6;
    profile.recurrence_momentum_fill_s = 2.0e-6;
    profile.recurrence_schedule_s = 20.459e-6;
    profile.recurrence_replay_output_mapping_s = 0.5e-6;
    profile.reduction_s = 2.0e-6;

    profile.validate_recurrence_top_level_accounting().unwrap();

    let mut double_counted = profile;
    double_counted.reduction_s += double_counted.recurrence_schedule_s;
    let error = double_counted
        .validate_recurrence_top_level_accounting()
        .unwrap_err();
    assert!(error.to_string().contains("exclusive top-level phases"));
}

#[test]
fn recurrence_native_profile_rejects_top_level_overlap() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 10.0e-3;
    profile.recurrence_momentum_fill_s = 4.0e-3;
    profile.recurrence_schedule_s = 7.0e-3;

    let error = profile
        .validate_recurrence_top_level_accounting()
        .unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
    assert!(error.to_string().contains("exclusive top-level phases"));
    assert!(error.to_string().contains("exceeding wall time"));
}

#[test]
fn recurrence_native_profile_rejects_attribution_larger_than_schedule() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 10.0e-3;
    profile.recurrence_schedule_s = 7.0e-3;
    profile.recurrence_source_kernel_s = 2.0e-3;
    profile.recurrence_contribution_kernel_s = 6.0e-3;

    let error = profile
        .validate_recurrence_top_level_accounting()
        .unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
    assert!(error.to_string().contains("schedule sub-attribution"));
    assert!(
        error
            .to_string()
            .contains("exceeding inclusive recurrence schedule")
    );
}

#[test]
fn compiled_native_profile_rejects_top_level_overlap() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 10.0e-3;
    profile.stage_input_pack_s = 4.0e-3;
    profile.stage_evaluator_call_s = 7.0e-3;

    let error = profile
        .validate_compiled_top_level_accounting()
        .unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
    assert!(error.to_string().contains("exclusive top-level phases"));
    assert!(error.to_string().contains("exceeding wall time"));
}

#[test]
fn native_profile_accumulates_compiled_accounting() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile {
        orchestration_s: 1.0,
        stage_leaf_input_pack_s: 2.0,
        recurrence_schedule_s: 6.0,
        recurrence_contribution_kernel_s: 4.0,
        recurrence_schedule_execution_count: 2,
        recurrence_contribution_row_count: 7,
        stage_leaf_input_pack_by_stage_s: vec![2.0],
        stage_leaf_input_copy_component_count: 3,
        evaluator_backend_call_count: 4,
        scratch_reallocation_count: 5,
        ..RuntimeProfile::default()
    }
    .into();
    let repeated: NativeRuntimeProfile = RuntimeProfile {
        orchestration_s: 10.0,
        stage_leaf_input_pack_s: 20.0,
        recurrence_schedule_s: 60.0,
        recurrence_contribution_kernel_s: 40.0,
        recurrence_schedule_execution_count: 20,
        recurrence_contribution_row_count: 70,
        stage_leaf_input_pack_by_stage_s: vec![20.0, 30.0],
        stage_leaf_input_copy_component_count: 30,
        evaluator_backend_call_count: 40,
        scratch_reallocation_count: 50,
        ..RuntimeProfile::default()
    }
    .into();

    profile.accumulate(&repeated);

    assert_eq!(profile.orchestration_s, 11.0);
    assert_eq!(profile.stage_leaf_input_pack_s, 22.0);
    assert_eq!(profile.recurrence_schedule_s, 66.0);
    assert_eq!(profile.recurrence_contribution_kernel_s, 44.0);
    assert_eq!(profile.recurrence_schedule_execution_count, 22);
    assert_eq!(profile.recurrence_contribution_row_count, 77);
    assert_eq!(profile.stage_leaf_input_pack_by_stage_s, [22.0, 30.0]);
    assert_eq!(profile.stage_leaf_input_copy_component_count, 33);
    assert_eq!(profile.evaluator_backend_call_count, 44);
    assert_eq!(profile.observed_scratch_reallocation_count, 55);
}
