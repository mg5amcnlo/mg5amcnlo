// SPDX-License-Identifier: 0BSD

use super::*;

fn load_compiled_symmetric_group_color_contraction(
    payload: Option<&CompiledColorContractionPayloadManifest>,
    payloads: &EvaluatorPayloadStore,
    physics: &PhysicsRuntime,
) -> RusticolResult<Option<crate::recurrence::RecurrenceColorContraction>> {
    let Some(payload) = payload else {
        return Ok(None);
    };
    if !physics.has_contracted_color_axis() || physics.manifest.color_components.len() != 1 {
        return Err(RusticolError::compatibility(
            "compiled symmetric-group FFT diagnostic requires one contracted FullColour axis",
        ));
    }
    if physics.manifest.coverage.helicities != "selected" {
        return Err(RusticolError::compatibility(
            "compiled symmetric-group FFT diagnostic requires generation-time selected helicity coverage",
        ));
    }
    let live_helicity_count = physics
        .manifest
        .helicities
        .iter()
        .filter(|helicity| {
            helicity.computed && !helicity.structural_zero && helicity.coefficient != 0.0
        })
        .count();
    if live_helicity_count != 1 {
        return Err(RusticolError::compatibility(
            "compiled symmetric-group FFT diagnostic requires exactly one generation-time selected helicity",
        ));
    }
    let bytes = payloads.packed_member_bytes(
        &payload.path,
        crate::pacbin::PacbinMemberKind::ColorContraction,
    )?;
    crate::recurrence::decode_recurrence_color_contraction_v3(bytes).map(Some)
}

#[cfg(test)]
mod compiled_symmetric_group_load_tests {
    use super::*;

    #[test]
    fn complete_helicity_coverage_is_rejected_even_with_one_live_helicity() {
        let physics = crate::engine::tests::test_physics_runtime("full");
        assert_eq!(physics.manifest.coverage.helicities, "complete");
        assert_eq!(
            physics
                .manifest
                .helicities
                .iter()
                .filter(|helicity| {
                    helicity.computed && !helicity.structural_zero && helicity.coefficient != 0.0
                })
                .count(),
            1,
        );
        let payload = CompiledColorContractionPayloadManifest {
            path: "not-read.pacrclr3".to_string(),
        };
        let payloads = EvaluatorPayloadStore::directory(std::path::Path::new("."));

        let error =
            load_compiled_symmetric_group_color_contraction(Some(&payload), &payloads, &physics)
                .expect_err("complete helicity coverage must not enter the selected diagnostic");

        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.message().contains("selected helicity coverage"));
    }
}

fn validate_compiled_plane_arena_contract(manifest: &ExecutionManifest) -> RusticolResult<bool> {
    let declared_execution = manifest
        .required_runtime_capabilities
        .iter()
        .any(|value| value == COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY);
    let Some(stages) = manifest.compiled.stage_evaluators.as_ref() else {
        if declared_execution {
            return Err(RusticolError::integrity(
                "compiled-plane-arena-v1 is declared without fused stage evaluators",
            ));
        }
        return Ok(false);
    };
    let declared_stage = stages
        .required_runtime_capabilities
        .iter()
        .any(|value| value == COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY);
    let all_stages = stages
        .stages
        .iter()
        .chain(std::iter::once(&stages.amplitude_stage))
        .collect::<Vec<_>>();
    let all_direct_capable = all_stages.iter().all(|stage| {
        stage.evaluator.leaf_layout().is_ok_and(|leaves| {
            !leaves.is_empty()
                && leaves.iter().all(|leaf| {
                    matches!(
                        leaf.evaluator,
                        EvaluatorManifest::SymjitApplication { .. }
                            | EvaluatorManifest::CompiledComplex {
                                native_direct_application: Some(_),
                                ..
                            }
                    )
                })
        })
    });
    let direct_count = all_stages
        .iter()
        .filter(|stage| stage.compiled_plane_arena.is_some())
        .count();

    if !declared_execution && !declared_stage && direct_count == 0 {
        return Err(RusticolError::compatibility(
            "this compiled f64 artifact predates compiled-plane-arena-v1; regenerate it with \
             `pyamplicol generate` using the current pyAmpliCol build",
        ));
    }
    if !declared_execution
        || !declared_stage
        || direct_count != all_stages.len()
        || !all_direct_capable
    {
        return Err(RusticolError::integrity(
            "compiled-plane-arena-v1 capability, fused direct-capable leaves, and stage bindings \
             do not form one complete execution contract",
        ));
    }
    let momentum_slot_ids = manifest.compiled.momentum_slot_ids.as_deref().ok_or_else(|| {
        RusticolError::compatibility(
            "compiled execution omits generation-time momentum-slot selection; regenerate the artifact",
        )
    })?;
    validate_sorted_unique_ids(
        momentum_slot_ids,
        manifest.runtime_schema.momentum_slots.len(),
        "compiled momentum-slot ids",
    )?;
    for (position, stage) in all_stages.into_iter().enumerate() {
        validate_compiled_plane_arena_stage(
            stage,
            position == stages.stages.len(),
            manifest.runtime_schema.value_storage.component_count,
            manifest.runtime_schema.amplitude_stage.output_count,
            manifest
                .runtime_schema
                .parameter_layout
                .momentum_parameter_count,
            momentum_slot_ids,
        )?;
    }
    Ok(true)
}

fn validate_compiled_plane_arena_stage(
    stage: &GenericSerializedStageEvaluatorManifest,
    is_amplitude: bool,
    value_component_count: usize,
    amplitude_component_count: usize,
    momentum_component_count: usize,
    momentum_slot_ids: &[usize],
) -> RusticolResult<()> {
    let direct = stage.compiled_plane_arena.as_ref().ok_or_else(|| {
        RusticolError::integrity("compiled plane-arena fused stage has no direct bindings")
    })?;
    let symjit_contract = direct.application_abi == COMPILED_PLANE_DIRECT_APPLICATION_ABI
        && direct.source_application_abi == COMPILED_PLANE_SOURCE_APPLICATION_ABI;
    #[cfg(feature = "f64-compiled")]
    let native_contract = direct.application_abi
        == super::evaluator::native_compiled_direct::NATIVE_COMPILED_DIRECT_APPLICATION_ABI
        && direct.source_application_abi
            == super::evaluator::native_compiled_direct::NATIVE_COMPILED_DIRECT_APPLICATION_ABI;
    #[cfg(not(feature = "f64-compiled"))]
    let native_contract = false;
    if direct.schema_version != 1
        || direct.kind != "compiled-plane-arena-stage"
        || !(symjit_contract || native_contract)
        || direct.element_layout != "split-complex-component-major"
        || direct.output_operation != "overwrite"
        || direct.output_factor != "identity"
        || direct.input_output_aliasing != "forbidden"
        || direct.output_output_aliasing != "forbidden"
    {
        return Err(RusticolError::compatibility(format!(
            "compiled plane-arena stage {:?} has an incompatible direct ABI; regenerate the artifact",
            stage.evaluator_label
        )));
    }
    if stage.parameter_layout != "stage-local-value-momentum"
        || direct.input_bindings.len() != stage.parameter_count
        || direct.output_bindings.len() != stage.output_length
        || direct.leaves.is_empty()
    {
        return Err(RusticolError::integrity(format!(
            "compiled plane-arena stage {:?} has incomplete direct bindings",
            stage.evaluator_label
        )));
    }
    u32::try_from(value_component_count)
        .and_then(|_| u32::try_from(amplitude_component_count))
        .and_then(|_| u32::try_from(stage.parameter_count))
        .and_then(|_| u32::try_from(stage.output_length))
        .map_err(|_| {
            RusticolError::integrity(
                "compiled plane-arena plane or scalar address space exceeds u32",
            )
        })?;

    let mut canonical_inputs = vec![None; stage.parameter_count];
    for component in &stage.input_components {
        let slot = canonical_inputs
            .get_mut(component.parameter_index)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "compiled plane-arena input binding index is out of bounds",
                )
            })?;
        if slot
            .replace(CompiledPlaneInputBindingManifest {
                parameter_index: component.parameter_index,
                kind: component.kind.clone(),
                source_id: component.source_id,
                component: component.component,
                global_component: component.global_component,
                real_valued: component.real_valued,
            })
            .is_some()
        {
            return Err(RusticolError::integrity(
                "compiled plane-arena input bindings contain duplicates",
            ));
        }
    }
    if canonical_inputs.iter().any(Option::is_none)
        || canonical_inputs
            .into_iter()
            .map(Option::unwrap)
            .ne(direct.input_bindings.iter().cloned())
    {
        return Err(RusticolError::integrity(format!(
            "compiled plane-arena stage {:?} input bindings disagree with the DAG",
            stage.evaluator_label
        )));
    }

    let expected_arena = if is_amplitude { "amplitude" } else { "current" };
    let output_limit = if is_amplitude {
        amplitude_component_count
    } else {
        value_component_count
    };
    let mut canonical_outputs = vec![None; stage.output_length];
    let mut output_components = BTreeSet::new();
    for slot in &stage.output_slots {
        let output_len = slot
            .output_stop
            .checked_sub(slot.output_start)
            .filter(|length| slot.component_stop.checked_sub(slot.component_start) == Some(*length))
            .ok_or_else(|| {
                RusticolError::integrity("compiled plane-arena output binding range is invalid")
            })?;
        if slot.output_stop > stage.output_length || slot.component_stop > output_limit {
            return Err(RusticolError::integrity(
                "compiled plane-arena output binding is out of bounds",
            ));
        }
        for offset in 0..output_len {
            let output_index = slot.output_start + offset;
            let component = slot.component_start + offset;
            if !output_components.insert(component)
                || canonical_outputs[output_index]
                    .replace(CompiledPlaneOutputBindingManifest {
                        output_index,
                        arena: expected_arena.to_string(),
                        component,
                    })
                    .is_some()
            {
                return Err(RusticolError::integrity(
                    "compiled plane-arena output bindings alias",
                ));
            }
        }
    }
    if canonical_outputs.iter().any(Option::is_none)
        || canonical_outputs
            .into_iter()
            .map(Option::unwrap)
            .ne(direct.output_bindings.iter().cloned())
    {
        return Err(RusticolError::integrity(format!(
            "compiled plane-arena stage {:?} output bindings disagree with the DAG",
            stage.evaluator_label
        )));
    }
    if !is_amplitude {
        let input_currents = direct
            .input_bindings
            .iter()
            .filter(|binding| binding.kind == "value")
            .map(|binding| binding.global_component)
            .collect::<BTreeSet<_>>();
        if output_components
            .iter()
            .any(|component| input_currents.contains(component))
        {
            return Err(RusticolError::integrity(
                "compiled plane-arena fused stage aliases an input and output plane",
            ));
        }
    }

    let canonical_leaves = stage.evaluator.leaf_layout()?;
    if canonical_leaves.len() != direct.leaves.len() {
        return Err(RusticolError::integrity(
            "compiled plane-arena leaf bindings do not cover the evaluator",
        ));
    }
    for (canonical, binding) in canonical_leaves.iter().zip(&direct.leaves) {
        let (
            application_path,
            application_abi,
            input_len,
            output_len,
            optimization_level,
            supported_optimization_level,
        ) = match canonical.evaluator {
            EvaluatorManifest::SymjitApplication {
                input_len,
                output_len,
                optimization_level,
                plane_application,
                ..
            } => {
                let plane_application = plane_application.as_ref().ok_or_else(|| {
                    RusticolError::compatibility(
                        "compiled SymJIT leaf predates the standard plane application ABI; \
                         regenerate the artifact",
                    )
                })?;
                plane_application.validate(*input_len, *output_len, *optimization_level)?;
                (
                    plane_application.application_path.as_str(),
                    plane_application.application_abi.as_str(),
                    *input_len,
                    *output_len,
                    *optimization_level,
                    *optimization_level <= 3,
                )
            }
            EvaluatorManifest::CompiledComplex {
                function_name,
                input_len,
                output_len,
                native_direct_application: Some(application),
                ..
            } => {
                application.validate(function_name, *input_len, *output_len)?;
                (
                    application.library_path.as_str(),
                    application.application_abi.as_str(),
                    *input_len,
                    *output_len,
                    3,
                    true,
                )
            }
            _ => {
                return Err(RusticolError::integrity(
                    "compiled plane-arena stage contains a leaf without a direct application",
                ));
            }
        };
        if !supported_optimization_level {
            return Err(RusticolError::compatibility(
                "compiled-plane-arena-v1 supports compiled JIT optimization levels 0 through 3",
            ));
        }
        if binding.optimization_level != optimization_level {
            return Err(RusticolError::integrity(
                "compiled plane-arena leaf optimization level disagrees with its source payload",
            ));
        }
        if binding.direct_codegen_optimization_level != optimization_level {
            return Err(RusticolError::compatibility(
                "compiled plane-arena leaf direct code generation must match its standard \
                 P-kernel optimization level",
            ));
        }
        if binding.application_path != application_path
            || binding.source_application_abi != application_abi
            || binding.source_application_abi != direct.source_application_abi
            || binding.input_len != input_len
            || binding.output_len != output_len
            || binding.input_indices != canonical.input_indices
            || binding.output_start != canonical.output_range.start
            || binding.output_stop != canonical.output_range.end
        {
            return Err(RusticolError::integrity(format!(
                "compiled plane-arena stage {:?} leaf bindings disagree with its source payload",
                stage.evaluator_label
            )));
        }
        for &input_index in &binding.input_indices {
            let input = direct.input_bindings.get(input_index).ok_or_else(|| {
                RusticolError::integrity("compiled leaf input index is outside its bindings")
            })?;
            if input.kind == "momentum" {
                let component = input
                    .global_component
                    .checked_sub(value_component_count)
                    .filter(|component| *component < momentum_component_count)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "compiled leaf momentum input is outside momentum planes",
                        )
                    })?;
                if momentum_slot_ids.binary_search(&(component / 4)).is_err() {
                    return Err(RusticolError::integrity(
                        "compiled momentum-slot selection omits a leaf input",
                    ));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn build_lc_topology_replay_mappings(
    replay: Option<&LcTopologyReplayManifest>,
) -> RusticolResult<(LcTopologyReplayMappings, Vec<f64>)> {
    let data = build_lc_topology_replay_data(replay)?;
    let mapping_count = data.mappings.len();
    Ok((data.mappings, vec![1.0; mapping_count]))
}

pub(super) fn build_lc_topology_replay_data(
    replay: Option<&LcTopologyReplayManifest>,
) -> RusticolResult<LcTopologyReplayData> {
    let Some(replay) = replay else {
        return Ok(LcTopologyReplayData::default());
    };
    let contract_version = replay.contract_version.unwrap_or(1);
    if !matches!(contract_version, 1 | 2) {
        return Err(RusticolError::compatibility(format!(
            "unsupported LC topology replay contract version {contract_version}"
        )));
    }
    if !replay.enabled {
        if contract_version == 2
            && (!replay.groups.is_empty()
                || replay.replayed_sector_count != 0
                || !replay.materialized_sector_ids.is_empty()
                || !replay.residual_sector_ids.is_empty())
        {
            return Err(RusticolError::invalid_argument(
                "disabled LC topology replay contract v2 contains active coverage",
            ));
        }
        return Ok(LcTopologyReplayData::default());
    }
    if replay.mode != "external-label-permutation" {
        return Err(RusticolError::invalid_argument(format!(
            "unsupported LC topology replay mode {:?}",
            replay.mode
        )));
    }
    let mut routes_by_mapping =
        BTreeMap::<Vec<(usize, usize)>, Vec<LcTopologyReplaySectorRoute>>::new();
    let mut replayed_sector_ids = BTreeSet::new();
    let mut group_materialized_sector_ids = BTreeSet::new();
    for group in &replay.groups {
        if group.materialized_sector_id != group.representative_sector_id {
            return Err(RusticolError::invalid_argument(
                "LC topology replay currently requires the materialized sector to be the representative sector",
            ));
        }
        if group.sector_permutations.is_empty() {
            return Err(RusticolError::invalid_argument(
                "enabled LC topology replay group contains no sector permutations",
            ));
        }
        if contract_version == 2 {
            validate_lc_topology_replay_proof(group.proof.as_ref())?;
        }
        let active_sector_ids = if group.active_sector_ids.is_empty() && contract_version == 1 {
            group
                .sector_permutations
                .iter()
                .map(|permutation| permutation.sector_id)
                .collect::<Vec<_>>()
        } else {
            group.active_sector_ids.clone()
        };
        let active_sector_ids =
            unique_lc_sector_ids(&active_sector_ids, "LC topology replay active sector ids")?;
        if !active_sector_ids.contains(&group.representative_sector_id) {
            return Err(RusticolError::invalid_argument(
                "LC topology replay active sector ids do not include the representative sector",
            ));
        }
        if !group_materialized_sector_ids.insert(group.materialized_sector_id) {
            return Err(RusticolError::invalid_argument(
                "LC topology replay groups reuse a materialized representative sector",
            ));
        }
        for sector_id in &active_sector_ids {
            if !replayed_sector_ids.insert(*sector_id) {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay groups overlap in physical sector coverage",
                ));
            }
        }
        let mut seen_permutations = BTreeSet::new();
        for permutation in &group.sector_permutations {
            if !active_sector_ids.contains(&permutation.sector_id) {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay sector permutation is not listed in active sector ids",
                ));
            }
            if !seen_permutations.insert(permutation.sector_id) {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay sector permutations contain duplicate sector ids",
                ));
            }
            if !permutation.weight.is_finite() || permutation.weight <= 0.0 {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay sector permutation weights must be positive finite numbers",
                ));
            }
            if !matches!(permutation.sign, -1 | 1) {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay permutation signs must be -1 or 1",
                ));
            }
            let expected_factor = permutation.weight * f64::from(permutation.sign);
            let amplitude_factor = match permutation.factor.as_deref() {
                Some([real, imaginary]) => {
                    if !real.is_finite()
                        || !imaginary.is_finite()
                        || *imaginary != 0.0
                        || real.to_bits() != expected_factor.to_bits()
                    {
                        return Err(RusticolError::invalid_argument(
                            "LC topology replay signed factor does not match its weight and sign",
                        ));
                    }
                    [*real, *imaginary]
                }
                Some(_) => {
                    return Err(RusticolError::invalid_argument(
                        "LC topology replay signed factor must contain real and imaginary components",
                    ));
                }
                None if contract_version == 1 => [expected_factor, 0.0],
                None => {
                    return Err(RusticolError::invalid_argument(
                        "LC topology replay contract v2 permutation is missing its signed factor",
                    ));
                }
            };
            let mapping = normalized_lc_topology_label_permutation(&permutation.label_permutation)?;
            let route = LcTopologyReplaySectorRoute {
                physical_sector_id: permutation.sector_id,
                materialized_sector_id: group.materialized_sector_id,
                weight: permutation.weight,
                sign: permutation.sign,
                amplitude_factor,
                residual: false,
            };
            if route.squared_reduction_weight().to_bits() != route.weight.to_bits() {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay signed factor has inconsistent squared reduction semantics",
                ));
            }
            routes_by_mapping.entry(mapping).or_default().push(route);
        }
        if seen_permutations != active_sector_ids {
            return Err(RusticolError::invalid_argument(
                "LC topology replay sector permutations do not exactly cover the active sectors",
            ));
        }
    }
    if replayed_sector_ids.len() != replay.replayed_sector_count {
        return Err(RusticolError::invalid_argument(format!(
            "LC topology replay declares {} sectors but contains {} permutations",
            replay.replayed_sector_count,
            replayed_sector_ids.len()
        )));
    }
    let residual_sector_ids = unique_lc_sector_ids(
        &replay.residual_sector_ids,
        "LC topology replay residual sector ids",
    )?;
    if replayed_sector_ids
        .intersection(&residual_sector_ids)
        .next()
        .is_some()
    {
        return Err(RusticolError::invalid_argument(
            "LC topology replay residual sectors overlap replayed sectors",
        ));
    }
    for sector_id in &residual_sector_ids {
        routes_by_mapping
            .entry(Vec::new())
            .or_default()
            .push(LcTopologyReplaySectorRoute {
                physical_sector_id: *sector_id,
                materialized_sector_id: *sector_id,
                weight: 1.0,
                sign: 1,
                amplitude_factor: [1.0, 0.0],
                residual: true,
            });
    }

    let mut materialized_sector_ids = group_materialized_sector_ids;
    materialized_sector_ids.extend(residual_sector_ids.iter().copied());
    if contract_version == 2 {
        let declared_materialized = unique_lc_sector_ids(
            &replay.materialized_sector_ids,
            "LC topology replay materialized sector ids",
        )?;
        if declared_materialized != materialized_sector_ids {
            return Err(RusticolError::invalid_argument(
                "LC topology replay materialized coverage does not match groups and residuals",
            ));
        }
        let physical_sector_count = replay.physical_sector_count.ok_or_else(|| {
            RusticolError::invalid_argument(
                "LC topology replay contract v2 is missing physical_sector_count",
            )
        })?;
        let mut physical_sector_ids = replayed_sector_ids.clone();
        physical_sector_ids.extend(residual_sector_ids.iter().copied());
        let expected = (0..physical_sector_count)
            .map(|sector_id| sector_id as i64)
            .collect::<BTreeSet<_>>();
        if physical_sector_ids != expected {
            return Err(RusticolError::invalid_argument(
                "LC topology replay contract v2 does not cover every physical sector exactly once",
            ));
        }
    } else if !replay.materialized_sector_ids.is_empty()
        || !replay.residual_sector_ids.is_empty()
        || replay.physical_sector_count.is_some()
    {
        return Err(RusticolError::invalid_argument(
            "legacy LC topology replay contract contains contract-v2 coverage fields",
        ));
    }

    let mut mappings = Vec::with_capacity(routes_by_mapping.len());
    let mut routes = Vec::with_capacity(routes_by_mapping.len());
    for (mapping, mut mapping_routes) in routes_by_mapping {
        mapping_routes.sort_by_key(|route| route.physical_sector_id);
        mappings.push(mapping);
        routes.push(mapping_routes);
    }
    Ok(LcTopologyReplayData {
        mappings,
        routes,
        materialized_sector_ids,
    })
}

pub(super) fn build_color_topology_replay_data(
    replay: Option<&ColorTopologyReplayManifest>,
    color_accuracy: &str,
) -> RusticolResult<LcTopologyReplayData> {
    let Some(replay) = replay else {
        return Ok(LcTopologyReplayData::default());
    };
    if !replay.enabled {
        if !replay.groups.is_empty()
            || replay.replayed_sector_count != 0
            || !replay.materialized_sector_ids.is_empty()
            || !replay.residual_sector_ids.is_empty()
        {
            return Err(RusticolError::artifact(
                "disabled color topology replay contains active coverage",
            ));
        }
        return Ok(LcTopologyReplayData::default());
    }
    if !matches!(color_accuracy, "nlc" | "full")
        || replay.color_accuracy != color_accuracy
        || replay.mode != "external-label-permutation"
        || replay.contract_version != Some(3)
    {
        return Err(RusticolError::artifact(
            "color topology replay is not bound to the NLC/full artifact accuracy",
        ));
    }
    let compatibility = LcTopologyReplayManifest {
        enabled: replay.enabled,
        mode: replay.mode.clone(),
        contract_version: Some(2),
        physical_sector_count: replay.physical_sector_count,
        replayed_sector_count: replay.replayed_sector_count,
        materialized_sector_ids: replay.materialized_sector_ids.clone(),
        residual_sector_ids: replay.residual_sector_ids.clone(),
        groups: replay.groups.clone(),
    };
    build_lc_topology_replay_data(Some(&compatibility))
}

fn validate_lc_topology_replay_proof(
    proof: Option<&LcTopologyReplayProofManifest>,
) -> RusticolResult<()> {
    let proof = proof.ok_or_else(|| {
        RusticolError::invalid_argument(
            "LC topology replay contract v2 group is missing proof metadata",
        )
    })?;
    let algorithm = proof.algorithm.as_deref().unwrap_or_default();
    let digest = proof.digest.as_deref().unwrap_or_default();
    if proof.status != "proven"
        || algorithm.is_empty()
        || digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RusticolError::invalid_argument(
            "LC topology replay contract v2 requires a proven algorithm and lowercase SHA-256 digest",
        ));
    }
    Ok(())
}

fn unique_lc_sector_ids(values: &[i64], description: &str) -> RusticolResult<BTreeSet<i64>> {
    let result = values.iter().copied().collect::<BTreeSet<_>>();
    if result.len() != values.len() || result.iter().any(|sector_id| *sector_id < 0) {
        return Err(RusticolError::invalid_argument(format!(
            "{description} contain duplicates or negative ids"
        )));
    }
    Ok(result)
}

fn normalized_lc_topology_label_permutation(
    values: &[LcTopologyReplayLabelPermutationManifest],
) -> RusticolResult<Vec<(usize, usize)>> {
    let mut representatives = BTreeSet::new();
    let mut sectors = BTreeSet::new();
    let mut mapping = Vec::new();
    for item in values {
        if item.representative_label == 0 || item.sector_label == 0 {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutations must use one-based labels",
            ));
        }
        let representative = item.representative_label - 1;
        let sector = item.sector_label - 1;
        if !representatives.insert(representative) || !sectors.insert(sector) {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation is not one-to-one",
            ));
        }
        if representative != sector {
            mapping.push((representative, sector));
        }
    }
    if representatives != sectors {
        return Err(RusticolError::invalid_argument(
            "LC topology replay label permutation is not a permutation of its support",
        ));
    }
    mapping.sort_unstable();
    Ok(mapping)
}

#[derive(Clone, Debug)]
struct HelicityRecurrenceSchemaCurrent {
    is_source: bool,
    external_labels: Vec<usize>,
}

#[derive(Clone, Debug)]
struct HelicityRecurrenceSchemaSource {
    external_label: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_states: Vec<GenericSourceStateIrManifest>,
}

#[derive(Clone, Copy, Debug)]
struct HelicityRecurrenceSchemaRoot {
    output_index: usize,
    left_current_id: usize,
    right_current_id: usize,
    helicity_weight: f64,
}

#[derive(Clone, Debug)]
struct HelicityCompiledStageChunkCoverage {
    current_ids_by_chunk: Vec<BTreeSet<usize>>,
    input_current_ids_by_chunk: Vec<BTreeSet<usize>>,
}

#[derive(Clone, Debug)]
struct HelicityCompiledChunkCoverage {
    stages: Vec<HelicityCompiledStageChunkCoverage>,
    amplitude_root_ids_by_chunk: Vec<BTreeSet<usize>>,
    amplitude_input_current_ids_by_chunk: Vec<BTreeSet<usize>>,
}

#[derive(Clone, Debug)]
struct HelicityRecurrenceSchemaView {
    external_count: usize,
    currents: Vec<HelicityRecurrenceSchemaCurrent>,
    sources: BTreeMap<usize, HelicityRecurrenceSchemaSource>,
    roots: Vec<HelicityRecurrenceSchemaRoot>,
    compiled_chunks: Option<HelicityCompiledChunkCoverage>,
}

impl HelicityRecurrenceSchemaView {
    fn from_execution_manifest(manifest: &ExecutionManifest) -> RusticolResult<Self> {
        let currents = manifest
            .runtime_schema
            .current_storage
            .current_slots
            .iter()
            .map(|slot| HelicityRecurrenceSchemaCurrent {
                is_source: slot.is_source,
                external_labels: slot.external_labels.clone(),
            })
            .collect::<Vec<_>>();
        let mut sources = BTreeMap::new();
        for source in &manifest.runtime_schema.source_fill.sources {
            let declared_states = source
                .source_ir
                .states
                .iter()
                .map(|state| {
                    state.transformed(&source.applied_crossing).map_err(|message| {
                        RusticolError::artifact(format!(
                            "helicity recurrence source current {} has an invalid declared state: {message}",
                            source.current_id
                        ))
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let spin_state = serde_json::from_value(source.spin_state.clone()).map_err(|_| {
                RusticolError::artifact(format!(
                    "helicity recurrence source current {} has an invalid spin state",
                    source.current_id
                ))
            })?;
            let record = HelicityRecurrenceSchemaSource {
                external_label: source.leg_label,
                helicity: source.source_helicity,
                chirality: source.chirality,
                spin_state,
                declared_states,
            };
            if sources.insert(source.current_id, record).is_some() {
                return Err(RusticolError::artifact(format!(
                    "helicity recurrence runtime schema repeats source current {}",
                    source.current_id
                )));
            }
        }
        let roots = manifest
            .runtime_schema
            .amplitude_stage
            .roots
            .iter()
            .map(|root| HelicityRecurrenceSchemaRoot {
                output_index: root.output_index,
                left_current_id: root.left_current_id,
                right_current_id: root.right_current_id,
                helicity_weight: root.helicity_weight,
            })
            .collect::<Vec<_>>();
        let compiled_chunks = if manifest
            .runtime_schema
            .helicity_recurrence
            .as_ref()
            .is_some_and(|recurrence| recurrence.materialization.is_some())
        {
            manifest
                .compiled
                .stage_evaluators
                .as_ref()
                .map(|evaluators| {
                    build_helicity_compiled_chunk_coverage(evaluators, &manifest.runtime_schema)
                })
                .transpose()?
        } else {
            None
        };
        Ok(Self {
            external_count: manifest.runtime_schema.external_particles.len(),
            currents,
            sources,
            roots,
            compiled_chunks,
        })
    }
}

#[derive(Clone, Debug)]
struct CompiledValueInputLayout {
    current_ids_by_slot: BTreeMap<usize, usize>,
    current_ids_by_global_component: Vec<Option<usize>>,
}

fn compiled_value_input_layout(schema: &ExecutionPlan) -> RusticolResult<CompiledValueInputLayout> {
    let current_count = schema.current_storage.current_slots.len();
    let mut current_ids_by_slot = BTreeMap::new();
    let mut current_ids_by_global_component = vec![None; schema.value_storage.component_count];
    for slot in &schema.value_storage.value_slots {
        if slot.current_id >= current_count
            || slot.component_start >= slot.component_stop
            || slot.component_stop > current_ids_by_global_component.len()
        {
            return Err(RusticolError::artifact(format!(
                "value slot {} has stale current {} or component coverage",
                slot.value_slot_id, slot.current_id
            )));
        }
        if current_ids_by_slot
            .insert(slot.value_slot_id, slot.current_id)
            .is_some()
        {
            return Err(RusticolError::artifact(format!(
                "runtime schema repeats value slot {}",
                slot.value_slot_id
            )));
        }
        for owner in &mut current_ids_by_global_component[slot.component_start..slot.component_stop]
        {
            if owner.replace(slot.current_id).is_some() {
                return Err(RusticolError::artifact(format!(
                    "value slot {} overlaps another global value component range",
                    slot.value_slot_id
                )));
            }
        }
    }
    Ok(CompiledValueInputLayout {
        current_ids_by_slot,
        current_ids_by_global_component,
    })
}

fn compiled_input_current_ids_by_chunk(
    stage: &GenericSerializedStageEvaluatorManifest,
    value_inputs: &CompiledValueInputLayout,
    current_count: usize,
) -> RusticolResult<Vec<BTreeSet<usize>>> {
    if stage.parameter_layout != "stage-local-value-momentum" {
        return Err(RusticolError::artifact(format!(
            "compiled selector stage {:?} requires stage-local input metadata",
            stage.evaluator_label
        )));
    }
    let evaluator_input_len = stage.evaluator.io_len()?.0;
    if evaluator_input_len != stage.parameter_count {
        return Err(RusticolError::artifact(format!(
            "compiled selector stage {:?} evaluator input count does not match its parameter layout",
            stage.evaluator_label
        )));
    }
    let mut components_by_parameter = vec![None; stage.parameter_count];
    for component in &stage.input_components {
        if component.parameter_index >= stage.parameter_count
            || components_by_parameter[component.parameter_index]
                .replace(component)
                .is_some()
        {
            return Err(RusticolError::artifact(format!(
                "compiled selector stage {:?} has duplicate or stale input-component metadata",
                stage.evaluator_label
            )));
        }
    }
    if components_by_parameter.iter().any(Option::is_none) {
        return Err(RusticolError::artifact(format!(
            "compiled selector stage {:?} has incomplete input-component metadata",
            stage.evaluator_label
        )));
    }

    stage
        .evaluator
        .leaf_layout()?
        .into_iter()
        .map(|leaf| {
            let mut current_ids = BTreeSet::new();
            for parameter_index in leaf.input_indices {
                let component = components_by_parameter
                    .get(parameter_index)
                    .and_then(|component| *component)
                    .ok_or_else(|| {
                        RusticolError::artifact(format!(
                            "compiled selector stage {:?} chunk references an absent input component",
                            stage.evaluator_label
                        ))
                    })?;
                match component.kind.as_str() {
                    "value" => {
                        let declared_current_id = value_inputs
                            .current_ids_by_slot
                            .get(&component.source_id)
                            .copied()
                            .ok_or_else(|| {
                                RusticolError::artifact(format!(
                                    "compiled selector stage {:?} references absent value slot {}",
                                    stage.evaluator_label, component.source_id
                                ))
                            })?;
                        let current_id = value_inputs
                            .current_ids_by_global_component
                            .get(component.global_component)
                            .copied()
                            .flatten()
                            .ok_or_else(|| {
                                RusticolError::artifact(format!(
                                    "compiled selector stage {:?} references unowned global value component {}",
                                    stage.evaluator_label, component.global_component
                                ))
                            })?;
                        if current_id != declared_current_id {
                            return Err(RusticolError::artifact(format!(
                                "compiled selector stage {:?} value-slot and global-component dependencies disagree",
                                stage.evaluator_label
                            )));
                        }
                        if current_id >= current_count {
                            return Err(RusticolError::artifact(format!(
                                "compiled selector stage {:?} references stale current {current_id}",
                                stage.evaluator_label
                            )));
                        }
                        current_ids.insert(current_id);
                    }
                    "momentum" | "model_parameter" => {}
                    kind => {
                        return Err(RusticolError::artifact(format!(
                            "compiled selector stage {:?} has unsupported input kind {kind:?}",
                            stage.evaluator_label
                        )));
                    }
                }
            }
            Ok(current_ids)
        })
        .collect()
}

fn build_helicity_compiled_chunk_coverage(
    evaluators: &GenericStageEvaluatorArtifactsManifest,
    schema: &ExecutionPlan,
) -> RusticolResult<HelicityCompiledChunkCoverage> {
    let current_count = schema.current_storage.current_slots.len();
    let roots = schema
        .amplitude_stage
        .roots
        .iter()
        .map(|root| HelicityRecurrenceSchemaRoot {
            output_index: root.output_index,
            left_current_id: root.left_current_id,
            right_current_id: root.right_current_id,
            helicity_weight: root.helicity_weight,
        })
        .collect::<Vec<_>>();
    let value_inputs = compiled_value_input_layout(schema)?;
    build_helicity_compiled_chunk_coverage_from_parts(
        evaluators,
        current_count,
        &roots,
        &value_inputs,
    )
}

fn build_helicity_compiled_chunk_coverage_from_parts(
    evaluators: &GenericStageEvaluatorArtifactsManifest,
    current_count: usize,
    roots: &[HelicityRecurrenceSchemaRoot],
    value_inputs: &CompiledValueInputLayout,
) -> RusticolResult<HelicityCompiledChunkCoverage> {
    let mut stages = Vec::with_capacity(evaluators.stages.len());
    for stage in &evaluators.stages {
        let ranges = evaluator_output_chunk_ranges(&stage.evaluator)?;
        validate_evaluator_output_ranges(&ranges, stage.output_length, &stage.evaluator_label)?;
        let input_current_ids_by_chunk =
            compiled_input_current_ids_by_chunk(stage, value_inputs, current_count)?;
        if input_current_ids_by_chunk.len() != ranges.len() {
            return Err(RusticolError::artifact(format!(
                "compiled selector stage {:?} input and output chunk counts disagree",
                stage.evaluator_label
            )));
        }
        let mut current_ids_by_chunk = vec![BTreeSet::new(); ranges.len()];
        for slot in &stage.output_slots {
            let current_id = usize::try_from(slot.current_id).map_err(|_| {
                RusticolError::artifact(format!(
                    "helicity recurrence stage {} has a non-current output slot",
                    stage.stage_index
                ))
            })?;
            if current_id >= current_count
                || slot.output_start >= slot.output_stop
                || slot.output_stop > stage.output_length
            {
                return Err(RusticolError::artifact(format!(
                    "helicity recurrence stage {} has a stale current or output range",
                    stage.stage_index
                )));
            }
            let mut covered = false;
            for (chunk_index, (start, stop)) in ranges.iter().copied().enumerate() {
                if slot.output_start < stop && start < slot.output_stop {
                    current_ids_by_chunk[chunk_index].insert(current_id);
                    covered = true;
                }
            }
            if !covered {
                return Err(RusticolError::artifact(format!(
                    "helicity recurrence stage {} output for current {current_id} is not covered by an evaluator chunk",
                    stage.stage_index
                )));
            }
        }
        stages.push(HelicityCompiledStageChunkCoverage {
            current_ids_by_chunk,
            input_current_ids_by_chunk,
        });
    }

    let amplitude = &evaluators.amplitude_stage;
    let ranges = evaluator_output_chunk_ranges(&amplitude.evaluator)?;
    validate_evaluator_output_ranges(&ranges, amplitude.output_length, &amplitude.evaluator_label)?;
    let amplitude_input_current_ids_by_chunk =
        compiled_input_current_ids_by_chunk(amplitude, value_inputs, current_count)?;
    if amplitude_input_current_ids_by_chunk.len() != ranges.len() {
        return Err(RusticolError::artifact(
            "compiled selector amplitude input and output chunk counts disagree",
        ));
    }
    let mut amplitude_root_ids_by_chunk = vec![BTreeSet::new(); ranges.len()];
    let mut output_owner = vec![None; amplitude.output_length];
    let mut evaluator_column_by_root_output = vec![None; amplitude.output_length];
    for slot in &amplitude.output_slots {
        if slot.current_id != -1
            || slot.output_start >= slot.output_stop
            || slot.output_stop > amplitude.output_length
        {
            return Err(RusticolError::artifact(
                "helicity recurrence amplitude evaluator has an invalid output slot",
            ));
        }
        let output_len = slot.output_stop - slot.output_start;
        if slot.component_stop.checked_sub(slot.component_start) != Some(output_len) {
            return Err(RusticolError::artifact(
                "helicity recurrence amplitude evaluator has inconsistent output mapping",
            ));
        }
        for offset in 0..output_len {
            let evaluator_output = slot.output_start + offset;
            let root_output = slot.component_start + offset;
            if root_output >= amplitude.output_length
                || output_owner[evaluator_output].replace(()).is_some()
                || evaluator_column_by_root_output[root_output]
                    .replace(evaluator_output)
                    .is_some()
            {
                return Err(RusticolError::artifact(
                    "helicity recurrence amplitude evaluator has overlapping output slots",
                ));
            }
        }
    }
    for (root_id, root) in roots.iter().enumerate() {
        let evaluator_output = evaluator_column_by_root_output
            .get(root.output_index)
            .copied()
            .flatten()
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "helicity recurrence amplitude root {root_id} is not covered by an evaluator output slot"
                ))
            })?;
        if output_owner[evaluator_output].is_none() {
            return Err(RusticolError::artifact(format!(
                "helicity recurrence amplitude root {root_id} is not covered by an evaluator output slot"
            )));
        }
        let chunk_index = ranges
            .iter()
            .position(|(start, stop)| *start <= evaluator_output && evaluator_output < *stop)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "helicity recurrence amplitude root {root_id} is not covered by an evaluator chunk"
                ))
            })?;
        amplitude_root_ids_by_chunk[chunk_index].insert(root_id);
    }
    Ok(HelicityCompiledChunkCoverage {
        stages,
        amplitude_root_ids_by_chunk,
        amplitude_input_current_ids_by_chunk,
    })
}

fn validate_compiled_selector_schedule_closure(
    context: &str,
    current_is_source: &[bool],
    chunks: &HelicityCompiledChunkCoverage,
    active_stage_chunk_indices: &[Vec<usize>],
    active_amplitude_chunk_indices: &[usize],
    allowed_current_ids: Option<&BTreeSet<usize>>,
    allowed_root_ids: Option<&BTreeSet<usize>>,
) -> RusticolResult<()> {
    if active_stage_chunk_indices.len() != chunks.stages.len() {
        return Err(RusticolError::artifact(format!(
            "{context} has {} stage schedules, expected {}",
            active_stage_chunk_indices.len(),
            chunks.stages.len()
        )));
    }
    let mut available_current_ids = current_is_source
        .iter()
        .enumerate()
        .filter_map(|(current_id, is_source)| {
            (*is_source && allowed_current_ids.is_none_or(|allowed| allowed.contains(&current_id)))
                .then_some(current_id)
        })
        .collect::<BTreeSet<_>>();

    for (stage_index, (stage, active_chunks)) in chunks
        .stages
        .iter()
        .zip(active_stage_chunk_indices)
        .enumerate()
    {
        validate_sorted_unique_ids(
            active_chunks,
            stage.current_ids_by_chunk.len(),
            &format!("{context} stage {stage_index} active chunks"),
        )?;
        let mut stage_outputs = BTreeSet::new();
        for chunk_index in active_chunks {
            let output_current_ids = &stage.current_ids_by_chunk[*chunk_index];
            if let Some(allowed) = allowed_current_ids
                && !output_current_ids.is_subset(allowed)
            {
                let foreign = output_current_ids
                    .difference(allowed)
                    .copied()
                    .collect::<Vec<_>>();
                return Err(RusticolError::artifact(format!(
                    "{context} stage {stage_index} evaluator chunk {chunk_index} crosses selector domains through currents {foreign:?}",
                )));
            }
            let missing = stage.input_current_ids_by_chunk[*chunk_index]
                .difference(&available_current_ids)
                .copied()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(RusticolError::artifact(format!(
                    "{context} stage {stage_index} evaluator chunk {chunk_index} is missing compiled ancestors {missing:?}",
                )));
            }
            stage_outputs.extend(output_current_ids.iter().copied());
        }
        available_current_ids.extend(stage_outputs);
    }

    validate_sorted_unique_ids(
        active_amplitude_chunk_indices,
        chunks.amplitude_root_ids_by_chunk.len(),
        &format!("{context} active amplitude chunks"),
    )?;
    for chunk_index in active_amplitude_chunk_indices {
        let root_ids = &chunks.amplitude_root_ids_by_chunk[*chunk_index];
        if let Some(allowed) = allowed_root_ids
            && !root_ids.is_subset(allowed)
        {
            let foreign = root_ids.difference(allowed).copied().collect::<Vec<_>>();
            return Err(RusticolError::artifact(format!(
                "{context} amplitude evaluator chunk {chunk_index} crosses selector domains through roots {foreign:?}",
            )));
        }
        let missing = chunks.amplitude_input_current_ids_by_chunk[*chunk_index]
            .difference(&available_current_ids)
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(RusticolError::artifact(format!(
                "{context} amplitude evaluator chunk {chunk_index} is missing compiled ancestors {missing:?}",
            )));
        }
    }
    Ok(())
}

pub(super) fn build_compiled_color_execution_plan(
    evaluators: &GenericStageEvaluatorArtifactsManifest,
    schema: &ExecutionPlan,
    replay_materialized_sector_ids: &BTreeSet<i64>,
) -> RusticolResult<Option<CompiledColorExecutionPlan>> {
    let declared_materialized_sector_ids = evaluators
        .stages
        .iter()
        .chain(std::iter::once(&evaluators.amplitude_stage))
        .flat_map(|stage| &stage.output_slots)
        .flat_map(|slot| slot.color_selector_domain_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    if declared_materialized_sector_ids.is_empty() {
        return Ok(None);
    }
    if !replay_materialized_sector_ids.is_empty()
        && &declared_materialized_sector_ids != replay_materialized_sector_ids
    {
        return Err(RusticolError::artifact(
            "compiled color-selector metadata does not match LC topology-replay sectors",
        ));
    }
    let materialized_sector_ids = &declared_materialized_sector_ids;
    let chunks = build_helicity_compiled_chunk_coverage(evaluators, schema)?;
    let current_is_source = schema
        .current_storage
        .current_slots
        .iter()
        .map(|current| current.is_source)
        .collect::<Vec<_>>();

    let mut active_stages = materialized_sector_ids
        .iter()
        .map(|sector_id| (*sector_id, Vec::with_capacity(evaluators.stages.len())))
        .collect::<BTreeMap<_, _>>();
    for stage in &evaluators.stages {
        let domains_by_chunk = compiled_color_domains_by_chunk(stage, materialized_sector_ids)?;
        for sector_id in materialized_sector_ids {
            active_stages
                .get_mut(sector_id)
                .expect("materialized color schedule initialized")
                .push(
                    domains_by_chunk
                        .iter()
                        .enumerate()
                        .filter_map(|(chunk_index, domains)| {
                            domains.contains(sector_id).then_some(chunk_index)
                        })
                        .collect(),
                );
        }
    }

    let amplitude_domains =
        compiled_color_domains_by_chunk(&evaluators.amplitude_stage, materialized_sector_ids)?;
    let mut schedules_by_materialized_sector = BTreeMap::new();
    for sector_id in materialized_sector_ids {
        let active_amplitude_chunk_indices = amplitude_domains
            .iter()
            .enumerate()
            .filter_map(|(chunk_index, domains)| domains.contains(sector_id).then_some(chunk_index))
            .collect::<Vec<_>>();
        if active_amplitude_chunk_indices.is_empty() {
            return Err(RusticolError::artifact(format!(
                "compiled color selector sector {sector_id} has no amplitude chunk coverage"
            )));
        }
        let active_stage_chunk_indices = active_stages
            .remove(sector_id)
            .expect("materialized color schedule initialized");
        validate_compiled_selector_schedule_closure(
            &format!("compiled color selector sector {sector_id}"),
            &current_is_source,
            &chunks,
            &active_stage_chunk_indices,
            &active_amplitude_chunk_indices,
            None,
            None,
        )?;
        schedules_by_materialized_sector.insert(
            *sector_id,
            Arc::new(CompiledColorSelectorSchedule {
                active_stage_chunk_indices,
                active_amplitude_chunk_indices,
            }),
        );
    }
    Ok(Some(CompiledColorExecutionPlan {
        schedules_by_materialized_sector,
    }))
}

fn compiled_color_domains_by_chunk(
    stage: &GenericSerializedStageEvaluatorManifest,
    materialized_sector_ids: &BTreeSet<i64>,
) -> RusticolResult<Vec<BTreeSet<i64>>> {
    let ranges = evaluator_output_chunk_ranges(&stage.evaluator)?;
    validate_evaluator_output_ranges(&ranges, stage.output_length, &stage.evaluator_label)?;
    let mut domains_by_chunk = vec![None; ranges.len()];
    for slot in &stage.output_slots {
        if slot.output_start >= slot.output_stop || slot.output_stop > stage.output_length {
            return Err(RusticolError::artifact(format!(
                "compiled color selector stage {:?} has an invalid output slot",
                stage.evaluator_label
            )));
        }
        let domains = slot
            .color_selector_domain_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if domains.len() != slot.color_selector_domain_ids.len()
            || !domains.is_subset(materialized_sector_ids)
        {
            return Err(RusticolError::artifact(format!(
                "compiled color selector stage {:?} has invalid materialized-sector coverage",
                stage.evaluator_label
            )));
        }
        for (chunk_index, (start, stop)) in ranges.iter().copied().enumerate() {
            if slot.output_start < stop && start < slot.output_stop {
                if let Some(previous) = &domains_by_chunk[chunk_index] {
                    if previous != &domains {
                        return Err(RusticolError::artifact(format!(
                            "compiled color selector stage {:?} evaluator chunk {chunk_index} crosses selector domains",
                            stage.evaluator_label
                        )));
                    }
                } else {
                    domains_by_chunk[chunk_index] = Some(domains.clone());
                }
            }
        }
    }
    domains_by_chunk
        .into_iter()
        .enumerate()
        .map(|(chunk_index, domains)| {
            domains.ok_or_else(|| {
                RusticolError::artifact(format!(
                    "compiled color selector stage {:?} evaluator chunk {chunk_index} has no output ownership",
                    stage.evaluator_label
                ))
            })
        })
        .collect()
}

fn evaluator_output_chunk_ranges(
    evaluator: &EvaluatorManifest,
) -> RusticolResult<Vec<(usize, usize)>> {
    evaluator.leaf_layout().map(|layout| {
        layout
            .into_iter()
            .map(|leaf| (leaf.output_range.start, leaf.output_range.end))
            .collect()
    })
}

fn validate_evaluator_output_ranges(
    ranges: &[(usize, usize)],
    expected_output_length: usize,
    label: &str,
) -> RusticolResult<()> {
    if ranges.is_empty()
        || ranges.first().map(|range| range.0) != Some(0)
        || ranges.last().map(|range| range.1) != Some(expected_output_length)
        || ranges
            .windows(2)
            .any(|pair| pair[0].1 != pair[1].0 || pair[0].0 >= pair[0].1)
    {
        return Err(RusticolError::artifact(format!(
            "helicity recurrence evaluator {label:?} has inconsistent chunk output ranges"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct HelicityCurrentOwner {
    class_index: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

fn build_helicity_recurrence_runtime(
    manifest: &ExecutionManifest,
) -> RusticolResult<Option<HelicityRecurrenceRuntime>> {
    let recurrence = manifest.runtime_schema.helicity_recurrence.as_ref();
    let view = HelicityRecurrenceSchemaView::from_execution_manifest(manifest)?;
    build_helicity_recurrence_runtime_from_view(recurrence, &view)
}

fn detach_compiled_helicity_load_inputs(
    manifest: &mut ExecutionManifest,
) -> RusticolResult<(
    Option<GenericStageEvaluatorArtifactsManifest>,
    Option<HelicityRecurrenceRuntime>,
)> {
    let helicity_recurrence = build_helicity_recurrence_runtime(manifest)?;
    let stage_evaluators = manifest.compiled.stage_evaluators.take();
    // The authenticated runtime owns every derived selector schedule.  Hide the
    // source contract from the metadata-only constructor after its serialized
    // evaluators have moved into the payload loader so it cannot rederive the
    // schedules against an artificially empty evaluator set.
    manifest.runtime_schema.helicity_recurrence = None;
    Ok((stage_evaluators, helicity_recurrence))
}

fn build_helicity_recurrence_runtime_from_view(
    recurrence: Option<&HelicityRecurrenceManifest>,
    schema: &HelicityRecurrenceSchemaView,
) -> RusticolResult<Option<HelicityRecurrenceRuntime>> {
    let Some(recurrence) = recurrence else {
        return Ok(None);
    };
    if recurrence.kind != HELICITY_RECURRENCE_KIND {
        return Err(RusticolError::artifact(format!(
            "unsupported helicity recurrence kind {:?}",
            recurrence.kind
        )));
    }
    if recurrence.contract_version != HELICITY_RECURRENCE_CONTRACT_VERSION {
        return Err(RusticolError::compatibility(format!(
            "unsupported helicity recurrence contract version {}; regenerate the artifact",
            recurrence.contract_version
        )));
    }
    if recurrence.proof_algorithm != HELICITY_RECURRENCE_PROOF_ALGORITHM {
        return Err(RusticolError::artifact(format!(
            "unsupported helicity recurrence proof algorithm {:?}",
            recurrence.proof_algorithm
        )));
    }
    if recurrence.materialization.is_none()
        && (recurrence.current_count != schema.currents.len()
            || recurrence.amplitude_root_count != schema.roots.len())
    {
        return Err(RusticolError::artifact(
            "helicity recurrence counts do not match the final runtime schema",
        ));
    }

    let mut domain_keys = BTreeSet::new();
    let mut selector_domains = Vec::with_capacity(recurrence.selector_domains.len());
    for (expected_id, domain) in recurrence.selector_domains.iter().enumerate() {
        if domain.id != expected_id {
            return Err(RusticolError::artifact(format!(
                "helicity selector domain ids must be contiguous; expected {expected_id}, got {}",
                domain.id
            )));
        }
        let labels = validate_helicity_external_labels(
            domain
                .source_states
                .iter()
                .map(|state| state.external_label),
            schema.external_count,
            &format!("helicity selector domain {expected_id}"),
        )?;
        let is_complete = labels.len() == schema.external_count
            && labels
                .iter()
                .enumerate()
                .all(|(index, label)| *label == index + 1);
        if domain.complete != is_complete {
            return Err(RusticolError::artifact(format!(
                "helicity selector domain {expected_id} has inconsistent complete coverage"
            )));
        }
        let key = domain
            .source_states
            .iter()
            .map(|state| (state.external_label, state.helicity))
            .collect::<Vec<_>>();
        if !domain_keys.insert(key.clone()) {
            return Err(RusticolError::artifact(
                "helicity selector domains repeat a source-state assignment",
            ));
        }
        selector_domains.push(HelicitySelectorDomainRuntime {
            complete: domain.complete,
            source_states: key
                .into_iter()
                .map(|(label, helicity)| (label - 1, helicity))
                .collect(),
        });
    }

    let mut class_ids = BTreeMap::new();
    let mut current_owners = vec![None; recurrence.current_count];
    let mut recurrence_classes = Vec::with_capacity(recurrence.recurrence_classes.len());
    for (class_index, class) in recurrence.recurrence_classes.iter().enumerate() {
        validate_helicity_recurrence_proof(
            &class.class_id,
            "helicity-current-sha256:",
            &class.proof,
        )?;
        if class_ids
            .insert(class.class_id.as_str(), class_index)
            .is_some()
        {
            return Err(RusticolError::artifact(
                "helicity recurrence class ids must be unique",
            ));
        }
        if class.members.is_empty() {
            return Err(RusticolError::artifact(format!(
                "helicity recurrence class {:?} has no members",
                class.class_id
            )));
        }
        let external_labels = validate_helicity_external_labels(
            class.external_labels.iter().copied(),
            schema.external_count,
            &format!("helicity recurrence class {:?}", class.class_id),
        )?;
        if class.source_class && external_labels.len() != 1 {
            return Err(RusticolError::artifact(format!(
                "helicity source recurrence class {:?} must cover exactly one external label",
                class.class_id
            )));
        }
        let mut representative_seen = false;
        let mut members = Vec::with_capacity(class.members.len());
        for member in &class.members {
            validate_helicity_factor(
                member.factor,
                &format!("helicity recurrence current {}", member.current_id),
            )?;
            if member.current_id >= recurrence.current_count {
                return Err(RusticolError::artifact(format!(
                    "helicity recurrence class {:?} references stale current id {}",
                    class.class_id, member.current_id
                )));
            }
            let domain = recurrence
                .selector_domains
                .get(member.selector_domain_id)
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "helicity recurrence current {} references unknown selector domain {}",
                        member.current_id, member.selector_domain_id
                    ))
                })?;
            let domain_labels = domain
                .source_states
                .iter()
                .map(|state| state.external_label)
                .collect::<Vec<_>>();
            if domain_labels != external_labels {
                return Err(RusticolError::artifact(format!(
                    "helicity recurrence current {} does not match class {:?}",
                    member.current_id, class.class_id
                )));
            }
            if recurrence.materialization.is_none() {
                let current = &schema.currents[member.current_id];
                if current.is_source != class.source_class
                    || (!current.external_labels.is_empty()
                        && current.external_labels != external_labels)
                {
                    return Err(RusticolError::artifact(format!(
                        "helicity recurrence current {} does not match class {:?}",
                        member.current_id, class.class_id
                    )));
                }
            }
            if current_owners[member.current_id]
                .replace(HelicityCurrentOwner {
                    class_index,
                    selector_domain_id: member.selector_domain_id,
                    factor: member.factor,
                })
                .is_some()
            {
                return Err(RusticolError::artifact(format!(
                    "helicity recurrence current {} belongs to multiple classes",
                    member.current_id
                )));
            }
            if member.current_id == class.representative_current_id {
                representative_seen = true;
                if !helicity_factors_equal(member.factor, [1.0, 0.0]) {
                    return Err(RusticolError::artifact(format!(
                        "helicity recurrence representative current {} must carry the identity factor",
                        member.current_id
                    )));
                }
            }
            members.push(HelicityCurrentReplayMemberRuntime {
                current_id: member.current_id,
                selector_domain_id: member.selector_domain_id,
                factor: member.factor,
            });
        }
        if !representative_seen {
            return Err(RusticolError::artifact(format!(
                "helicity recurrence representative current {} is not a member of class {:?}",
                class.representative_current_id, class.class_id
            )));
        }
        recurrence_classes.push(HelicityRecurrenceClassRuntime {
            representative_current_id: class.representative_current_id,
            external_labels: external_labels.into_iter().map(|label| label - 1).collect(),
            source_class: class.source_class,
            members,
        });
    }

    validate_sorted_unique_ids(
        &recurrence.residual_current_ids,
        recurrence.current_count,
        "helicity recurrence residual current ids",
    )?;
    for current_id in &recurrence.residual_current_ids {
        if current_owners[*current_id].is_some() {
            return Err(RusticolError::artifact(format!(
                "helicity residual current {current_id} overlaps a proven recurrence class"
            )));
        }
    }
    if current_owners
        .iter()
        .filter(|owner| owner.is_some())
        .count()
        + recurrence.residual_current_ids.len()
        != recurrence.current_count
    {
        return Err(RusticolError::artifact(
            "helicity recurrence classes and residual currents do not cover the proof DAG",
        ));
    }
    if recurrence.materialization.is_none()
        && recurrence
            .residual_current_ids
            .iter()
            .any(|current_id| schema.currents[*current_id].is_source)
    {
        return Err(RusticolError::artifact(
            "helicity recurrence leaves a final runtime source current residual",
        ));
    }

    let source_class_current_ids = recurrence
        .recurrence_classes
        .iter()
        .filter(|class| class.source_class)
        .flat_map(|class| class.members.iter().map(|member| member.current_id))
        .collect::<BTreeSet<_>>();
    let mut mapped_source_ids = BTreeSet::new();
    let mut source_state_mappings = Vec::with_capacity(recurrence.source_state_mappings.len());
    for mapping in &recurrence.source_state_mappings {
        validate_helicity_factor(
            mapping.factor,
            &format!("helicity source mapping for current {}", mapping.current_id),
        )?;
        validate_sha256_digest(
            &mapping.source_contract_digest,
            &format!("helicity source mapping for current {}", mapping.current_id),
        )?;
        if !mapped_source_ids.insert(mapping.current_id) {
            return Err(RusticolError::artifact(format!(
                "helicity source current {} has multiple state mappings",
                mapping.current_id
            )));
        }
        let owner = current_owners
            .get(mapping.current_id)
            .and_then(|owner| *owner)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "helicity source mapping references residual or stale current {}",
                    mapping.current_id
                ))
            })?;
        let class = &recurrence.recurrence_classes[owner.class_index];
        let domain = recurrence
            .selector_domains
            .get(mapping.selector_domain_id)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "helicity source mapping current {} references unknown selector domain {}",
                    mapping.current_id, mapping.selector_domain_id
                ))
            })?;
        if !class.source_class
            || mapping.recurrence_class_id != class.class_id
            || mapping.representative_current_id != class.representative_current_id
            || mapping.selector_domain_id != owner.selector_domain_id
            || !helicity_factors_equal(mapping.factor, owner.factor)
            || !domain.source_states.iter().any(|state| {
                state.external_label == mapping.external_label && state.helicity == mapping.helicity
            })
        {
            return Err(RusticolError::artifact(format!(
                "helicity source mapping current {} is inconsistent with its recurrence class, selector domain, or SourceIR",
                mapping.current_id
            )));
        }
        if recurrence.materialization.is_none() {
            validate_helicity_source_against_runtime(
                mapping.current_id,
                mapping.external_label,
                mapping.helicity,
                mapping.chirality,
                &mapping.spin_state,
                mapping.declared_state_index,
                schema,
            )?;
        }
        source_state_mappings.push(HelicitySourceStateMappingRuntime {
            current_id: mapping.current_id,
            external_index: mapping.external_label - 1,
            helicity: mapping.helicity,
            chirality: mapping.chirality,
            spin_state: mapping.spin_state.clone(),
            declared_state_index: mapping.declared_state_index,
            selector_domain_id: mapping.selector_domain_id,
            recurrence_class_index: owner.class_index,
            representative_current_id: mapping.representative_current_id,
            factor: mapping.factor,
        });
    }
    if mapped_source_ids != source_class_current_ids {
        return Err(RusticolError::artifact(
            "helicity source-state mappings do not exactly cover proven source-class currents",
        ));
    }
    let residual_current_ids = recurrence
        .residual_current_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut root_owners = vec![false; recurrence.amplitude_root_count];
    let mut amplitude_class_ids = BTreeSet::new();
    let mut represented_complete_domains = BTreeSet::new();
    let mut amplitude_classes = Vec::with_capacity(recurrence.amplitude_classes.len());
    for class in &recurrence.amplitude_classes {
        validate_helicity_recurrence_proof(
            &class.class_id,
            "helicity-amplitude-sha256:",
            &class.proof,
        )?;
        if !amplitude_class_ids.insert(class.class_id.as_str()) {
            return Err(RusticolError::artifact(
                "helicity amplitude recurrence class ids must be unique",
            ));
        }
        if class.members.is_empty() {
            return Err(RusticolError::artifact(format!(
                "helicity amplitude class {:?} has no members",
                class.class_id
            )));
        }
        let mut representative_seen = false;
        let mut members = Vec::with_capacity(class.members.len());
        for member in &class.members {
            validate_helicity_factor(
                member.factor,
                &format!("helicity amplitude root {}", member.root_id),
            )?;
            if member.root_id >= recurrence.amplitude_root_count {
                return Err(RusticolError::artifact(format!(
                    "helicity amplitude class {:?} references stale root id {}",
                    class.class_id, member.root_id
                )));
            }
            if root_owners[member.root_id] {
                return Err(RusticolError::artifact(format!(
                    "helicity amplitude root {} belongs to multiple classes",
                    member.root_id
                )));
            }
            root_owners[member.root_id] = true;
            validate_complete_selector_ids(
                &member.selector_domain_ids,
                recurrence,
                &format!("helicity amplitude root {}", member.root_id),
            )?;
            represented_complete_domains.extend(member.selector_domain_ids.iter().copied());
            if recurrence.materialization.is_none() {
                let root = &schema.roots[member.root_id];
                if !root.helicity_weight.is_finite()
                    || (root.helicity_weight - member.selector_domain_ids.len() as f64).abs()
                        > 1.0e-12
                {
                    return Err(RusticolError::artifact(format!(
                        "helicity amplitude root {} selector coverage disagrees with its final runtime helicity weight",
                        member.root_id
                    )));
                }
                if residual_current_ids.contains(&root.left_current_id)
                    || residual_current_ids.contains(&root.right_current_id)
                {
                    return Err(RusticolError::artifact(format!(
                        "proven helicity amplitude root {} depends on a residual current",
                        member.root_id
                    )));
                }
            }
            representative_seen |= member.root_id == class.representative_root_id;
            members.push(HelicityAmplitudeReplayMemberRuntime {
                root_id: member.root_id,
                selector_domain_ids: member.selector_domain_ids.clone(),
                factor: member.factor,
            });
        }
        if !representative_seen {
            return Err(RusticolError::artifact(format!(
                "helicity amplitude representative root {} is not a member of class {:?}",
                class.representative_root_id, class.class_id
            )));
        }
        amplitude_classes.push(HelicityAmplitudeReplayClassRuntime {
            representative_root_id: class.representative_root_id,
            members,
        });
    }

    validate_sorted_unique_ids(
        &recurrence.residual_root_ids,
        recurrence.amplitude_root_count,
        "helicity recurrence residual root ids",
    )?;
    for root_id in &recurrence.residual_root_ids {
        if root_owners[*root_id] {
            return Err(RusticolError::artifact(format!(
                "helicity residual root {root_id} overlaps a proven amplitude class"
            )));
        }
    }
    if root_owners.iter().filter(|owned| **owned).count() + recurrence.residual_root_ids.len()
        != recurrence.amplitude_root_count
    {
        return Err(RusticolError::artifact(
            "helicity amplitude classes and residual roots do not cover the proof DAG",
        ));
    }

    validate_sorted_unique_ids(
        &recurrence.structural_zero_selector_domain_ids,
        recurrence.selector_domains.len(),
        "structural-zero helicity selector domain ids",
    )?;
    let structural_zero_domains = recurrence
        .structural_zero_selector_domain_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if structural_zero_domains
        .iter()
        .any(|domain_id| !recurrence.selector_domains[*domain_id].complete)
        || !structural_zero_domains.is_disjoint(&represented_complete_domains)
    {
        return Err(RusticolError::artifact(
            "structural-zero helicity domains must be complete and disjoint from amplitude coverage",
        ));
    }
    if recurrence.materialization.is_none() {
        let available_source_states = recurrence
            .source_state_mappings
            .iter()
            .map(|mapping| (mapping.external_label, mapping.helicity))
            .collect::<BTreeSet<_>>();
        let directly_executable_domains = recurrence
            .selector_domains
            .iter()
            .filter(|domain| {
                domain.source_states.iter().all(|state| {
                    available_source_states.contains(&(state.external_label, state.helicity))
                })
            })
            .map(|domain| domain.id)
            .collect::<BTreeSet<_>>();
        let mut replay_anchored_domains = BTreeSet::new();
        for class in &recurrence.amplitude_classes {
            for member in &class.members {
                if !member
                    .selector_domain_ids
                    .iter()
                    .any(|domain_id| directly_executable_domains.contains(domain_id))
                {
                    let domain_id =
                        member.selector_domain_ids.first().copied().ok_or_else(|| {
                            RusticolError::artifact(
                                "helicity amplitude replay member has no selector domains",
                            )
                        })?;
                    return Err(RusticolError::artifact(format!(
                        "helicity selector domain {domain_id} has no directly executable source-state anchor in its certified amplitude replay group"
                    )));
                }
                replay_anchored_domains.extend(member.selector_domain_ids.iter().copied());
            }
        }
        for domain in &recurrence.selector_domains {
            if structural_zero_domains.contains(&domain.id)
                || directly_executable_domains.contains(&domain.id)
                || replay_anchored_domains.contains(&domain.id)
            {
                continue;
            }
            if domain.complete {
                return Err(RusticolError::artifact(format!(
                    "helicity selector domain {} is neither directly executable nor covered by a certified amplitude replay group",
                    domain.id
                )));
            }
        }
    }
    let complete_domains = recurrence
        .selector_domains
        .iter()
        .filter(|domain| domain.complete)
        .map(|domain| domain.id)
        .collect::<BTreeSet<_>>();
    if complete_domains
        != represented_complete_domains
            .union(&structural_zero_domains)
            .copied()
            .collect()
    {
        return Err(RusticolError::artifact(
            "helicity amplitude and structural-zero domains do not cover every complete selector domain",
        ));
    }

    let actual_counts = HelicityRecurrenceProofCountsManifest {
        recurrence_class_count: recurrence.recurrence_classes.len(),
        optimized_recurrence_class_count: recurrence
            .recurrence_classes
            .iter()
            .filter(|class| class.members.len() > 1)
            .count(),
        optimized_current_count: recurrence
            .recurrence_classes
            .iter()
            .filter(|class| class.members.len() > 1)
            .map(|class| class.members.len())
            .sum(),
        residual_current_count: recurrence.residual_current_ids.len(),
        amplitude_class_count: recurrence.amplitude_classes.len(),
        optimized_amplitude_class_count: recurrence
            .amplitude_classes
            .iter()
            .filter(|class| class.members.len() > 1)
            .count(),
        residual_amplitude_count: recurrence.residual_root_ids.len(),
        source_state_mapping_count: recurrence.source_state_mappings.len(),
        physical_helicity_count: complete_domains.len(),
        structural_zero_helicity_count: structural_zero_domains.len(),
    };
    if recurrence.proof_counts != actual_counts {
        return Err(RusticolError::artifact(
            "helicity recurrence proof counts do not match validated manifest contents",
        ));
    }

    let materialization = recurrence
        .materialization
        .as_ref()
        .map(|materialization| {
            build_helicity_materialization_runtime(
                recurrence,
                materialization,
                schema,
                &current_owners,
                &complete_domains,
                &structural_zero_domains,
            )
        })
        .transpose()?;

    Ok(Some(HelicityRecurrenceRuntime {
        selector_domains,
        source_state_mappings,
        recurrence_classes,
        amplitude_classes,
        residual_current_ids: recurrence.residual_current_ids.clone(),
        residual_root_ids: recurrence.residual_root_ids.clone(),
        structural_zero_selector_domain_ids: recurrence.structural_zero_selector_domain_ids.clone(),
        materialization,
    }))
}

fn validate_helicity_source_against_runtime(
    current_id: usize,
    external_label: usize,
    helicity: i32,
    chirality: i32,
    spin_state: &GenericSourceSpinStateManifest,
    declared_state_index: usize,
    schema: &HelicityRecurrenceSchemaView,
) -> RusticolResult<()> {
    let source = schema.sources.get(&current_id).ok_or_else(|| {
        RusticolError::artifact(format!(
            "helicity source mapping current {current_id} is not a runtime source"
        ))
    })?;
    let declared_state = source
        .declared_states
        .get(declared_state_index)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "helicity source mapping current {current_id} has stale declared state index {declared_state_index}"
            ))
        })?;
    if external_label != source.external_label
        || helicity != source.helicity
        || chirality != source.chirality
        || spin_state != &source.spin_state
        || declared_state.helicity != helicity
        || declared_state.chirality != chirality
        || &declared_state.spin_state != spin_state
    {
        return Err(RusticolError::artifact(format!(
            "helicity source mapping current {current_id} is inconsistent with its SourceIR"
        )));
    }
    Ok(())
}

fn build_helicity_materialization_runtime(
    recurrence: &HelicityRecurrenceManifest,
    materialization: &HelicityRecurrenceMaterializationManifest,
    schema: &HelicityRecurrenceSchemaView,
    current_owners: &[Option<HelicityCurrentOwner>],
    complete_domains: &BTreeSet<usize>,
    structural_zero_domains: &BTreeSet<usize>,
) -> RusticolResult<HelicityRecurrenceMaterializationRuntime> {
    if materialization.kind != HELICITY_MATERIALIZATION_KIND {
        return Err(RusticolError::artifact(format!(
            "unsupported helicity recurrence materialization kind {:?}",
            materialization.kind
        )));
    }
    if materialization.contract_version != HELICITY_MATERIALIZATION_CONTRACT_VERSION {
        return Err(RusticolError::compatibility(format!(
            "unsupported helicity recurrence materialization contract version {}; regenerate the artifact",
            materialization.contract_version
        )));
    }
    if materialization.proof_current_count != recurrence.current_count
        || materialization.proof_root_count != recurrence.amplitude_root_count
    {
        return Err(RusticolError::artifact(
            "helicity materialization proof counts do not match the recurrence proof DAG",
        ));
    }
    if materialization.materialized_current_count != schema.currents.len()
        || materialization.materialized_root_count != schema.roots.len()
    {
        return Err(RusticolError::artifact(
            "helicity materialization counts do not match the final runtime schema",
        ));
    }
    if materialization.proof_to_materialized_current.len() != recurrence.current_count
        || materialization
            .proof_to_materialized_current
            .iter()
            .any(|current_id| *current_id >= materialization.materialized_current_count)
    {
        return Err(RusticolError::artifact(
            "helicity proof-to-materialized current map has a stale id or incorrect length",
        ));
    }

    let residual_targets = match materialization.strategy {
        HelicityMaterializationStrategy::Quotient => {
            validate_quotient_helicity_current_map(recurrence, materialization, schema)?
        }
        HelicityMaterializationStrategy::RetainedProofGraph => {
            validate_retained_helicity_current_map(recurrence, materialization, schema)?
        }
    };

    let source_routes = validate_helicity_materialized_source_routes(
        recurrence,
        materialization,
        schema,
        current_owners,
    )?;
    let amplitude_routes = validate_helicity_materialized_amplitude_routes(
        recurrence,
        materialization,
        schema,
        complete_domains,
        structural_zero_domains,
        &residual_targets,
    )?;
    let selector_schedules = validate_helicity_materialized_selector_schedules(
        recurrence,
        materialization,
        schema,
        complete_domains,
        structural_zero_domains,
    )?;

    Ok(HelicityRecurrenceMaterializationRuntime {
        strategy: materialization.strategy,
        proof_to_materialized_current: materialization.proof_to_materialized_current.clone(),
        source_routes,
        amplitude_routes,
        selector_schedules,
    })
}

fn validate_quotient_helicity_current_map(
    recurrence: &HelicityRecurrenceManifest,
    materialization: &HelicityRecurrenceMaterializationManifest,
    schema: &HelicityRecurrenceSchemaView,
) -> RusticolResult<BTreeSet<usize>> {
    let mut target_seen = vec![false; materialization.materialized_current_count];
    for class in &recurrence.recurrence_classes {
        let target = materialization.proof_to_materialized_current[class.representative_current_id];
        if class.members.iter().any(|member| {
            materialization.proof_to_materialized_current[member.current_id] != target
        }) {
            return Err(RusticolError::artifact(format!(
                "helicity recurrence class {:?} is split across materialized currents",
                class.class_id
            )));
        }
        if target_seen[target] {
            return Err(RusticolError::artifact(format!(
                "helicity materialized current {target} is shared by distinct proof classes"
            )));
        }
        target_seen[target] = true;
        let current = &schema.currents[target];
        if current.is_source != class.source_class
            || (!current.external_labels.is_empty()
                && current.external_labels != class.external_labels)
        {
            return Err(RusticolError::artifact(format!(
                "helicity materialized current {target} does not match proof class {:?}",
                class.class_id
            )));
        }
    }
    let mut residual_targets = BTreeSet::new();
    for proof_current_id in &recurrence.residual_current_ids {
        let target = materialization.proof_to_materialized_current[*proof_current_id];
        if target_seen[target] || !residual_targets.insert(target) {
            return Err(RusticolError::artifact(format!(
                "helicity residual proof current {proof_current_id} overlaps another materialized current"
            )));
        }
        if schema.currents[target].is_source {
            return Err(RusticolError::artifact(format!(
                "helicity residual proof current {proof_current_id} materializes as an unrouted source"
            )));
        }
        target_seen[target] = true;
    }
    if target_seen.iter().any(|seen| !seen) {
        return Err(RusticolError::artifact(
            "helicity proof-to-materialized current map does not cover every final current",
        ));
    }
    Ok(residual_targets)
}

fn validate_retained_helicity_current_map(
    recurrence: &HelicityRecurrenceManifest,
    materialization: &HelicityRecurrenceMaterializationManifest,
    schema: &HelicityRecurrenceSchemaView,
) -> RusticolResult<BTreeSet<usize>> {
    if materialization.materialized_current_count != recurrence.current_count
        || materialization.materialized_root_count != recurrence.amplitude_root_count
    {
        return Err(RusticolError::artifact(
            "retained helicity materialization must preserve every proof current and root",
        ));
    }
    if materialization
        .proof_to_materialized_current
        .iter()
        .enumerate()
        .any(|(proof_id, materialized_id)| proof_id != *materialized_id)
    {
        return Err(RusticolError::artifact(
            "retained helicity materialization requires an identity current map",
        ));
    }

    let mut current_seen = vec![false; recurrence.current_count];
    for class in &recurrence.recurrence_classes {
        for member in &class.members {
            let current = &schema.currents[member.current_id];
            if current_seen[member.current_id]
                || current.is_source != class.source_class
                || (!current.external_labels.is_empty()
                    && current.external_labels != class.external_labels)
            {
                return Err(RusticolError::artifact(format!(
                    "retained helicity current {} does not match proof class {:?}",
                    member.current_id, class.class_id
                )));
            }
            current_seen[member.current_id] = true;
        }
    }
    let mut residual_targets = BTreeSet::new();
    for proof_current_id in &recurrence.residual_current_ids {
        if current_seen[*proof_current_id] || !residual_targets.insert(*proof_current_id) {
            return Err(RusticolError::artifact(format!(
                "retained helicity residual current {proof_current_id} overlaps another proof current"
            )));
        }
        if schema.currents[*proof_current_id].is_source {
            return Err(RusticolError::artifact(format!(
                "retained helicity residual current {proof_current_id} is an unrouted source"
            )));
        }
        current_seen[*proof_current_id] = true;
    }
    if current_seen.iter().any(|seen| !seen) {
        return Err(RusticolError::artifact(
            "retained helicity materialization does not cover every proof current",
        ));
    }
    Ok(residual_targets)
}

fn validate_helicity_materialized_source_routes(
    recurrence: &HelicityRecurrenceManifest,
    materialization: &HelicityRecurrenceMaterializationManifest,
    schema: &HelicityRecurrenceSchemaView,
    current_owners: &[Option<HelicityCurrentOwner>],
) -> RusticolResult<Vec<HelicityMaterializedSourceRouteRuntime>> {
    if materialization.source_routes.len() != recurrence.source_state_mappings.len() {
        return Err(RusticolError::artifact(
            "helicity materialized source routes do not cover every proof source mapping",
        ));
    }
    let mut route_keys = BTreeSet::new();
    let mut routed_source_ids = BTreeSet::new();
    let mut runtime_routes = Vec::with_capacity(materialization.source_routes.len());
    for (mapping, route) in recurrence
        .source_state_mappings
        .iter()
        .zip(&materialization.source_routes)
    {
        validate_helicity_factor(
            route.factor,
            &format!(
                "helicity materialized source route for proof current {}",
                mapping.current_id
            ),
        )?;
        let owner = current_owners[mapping.current_id].ok_or_else(|| {
            RusticolError::artifact(format!(
                "helicity materialized source route references unproven current {}",
                mapping.current_id
            ))
        })?;
        let class = &recurrence.recurrence_classes[owner.class_index];
        let representative_factor = class
            .members
            .iter()
            .find(|member| member.current_id == class.representative_current_id)
            .map(|member| member.factor)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "helicity source class {:?} has no representative factor",
                    class.class_id
                ))
            })?;
        let expected_current_id = materialization.proof_to_materialized_current[mapping.current_id];
        let factor_matches = match materialization.strategy {
            HelicityMaterializationStrategy::Quotient => {
                helicity_factor_ratio_matches(route.factor, mapping.factor, representative_factor)
            }
            HelicityMaterializationStrategy::RetainedProofGraph => {
                helicity_factors_equal(route.factor, [1.0, 0.0])
            }
        };
        if route.materialized_current_id != expected_current_id
            || route.external_label != mapping.external_label
            || route.helicity != mapping.helicity
            || route.chirality != mapping.chirality
            || route.spin_state != mapping.spin_state
            || route.declared_state_index != mapping.declared_state_index
            || route.selector_domain_id != mapping.selector_domain_id
            || !factor_matches
        {
            return Err(RusticolError::artifact(format!(
                "helicity materialized source route for proof current {} is inconsistent",
                mapping.current_id
            )));
        }
        let source = schema
            .sources
            .get(&route.materialized_current_id)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "helicity materialized source route targets non-source current {}",
                    route.materialized_current_id
                ))
            })?;
        let declared_state = source
            .declared_states
            .get(route.declared_state_index)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "helicity materialized source route for current {} has stale declared state index {}",
                    route.materialized_current_id, route.declared_state_index
                ))
            })?;
        if source.external_label != route.external_label
            || declared_state.helicity != route.helicity
            || declared_state.chirality != route.chirality
            || declared_state.spin_state != route.spin_state
        {
            return Err(RusticolError::artifact(format!(
                "helicity materialized source route for current {} disagrees with SourceIR",
                route.materialized_current_id
            )));
        }
        if !route_keys.insert((
            route.materialized_current_id,
            route.selector_domain_id,
            route.external_label,
            route.helicity,
            route.chirality,
            route.declared_state_index,
        )) {
            return Err(RusticolError::artifact(
                "helicity materialized source routes contain a duplicate physical state",
            ));
        }
        routed_source_ids.insert(route.materialized_current_id);
        runtime_routes.push(HelicityMaterializedSourceRouteRuntime {
            materialized_current_id: route.materialized_current_id,
            external_index: route.external_label - 1,
            helicity: route.helicity,
            chirality: route.chirality,
            spin_state: route.spin_state.clone(),
            declared_state_index: route.declared_state_index,
            selector_domain_id: route.selector_domain_id,
            factor: route.factor,
        });
    }
    if routed_source_ids != schema.sources.keys().copied().collect() {
        return Err(RusticolError::artifact(
            "helicity materialized source routes do not cover every final runtime source",
        ));
    }
    Ok(runtime_routes)
}

fn validate_helicity_materialized_amplitude_routes(
    recurrence: &HelicityRecurrenceManifest,
    materialization: &HelicityRecurrenceMaterializationManifest,
    schema: &HelicityRecurrenceSchemaView,
    complete_domains: &BTreeSet<usize>,
    structural_zero_domains: &BTreeSet<usize>,
    residual_current_targets: &BTreeSet<usize>,
) -> RusticolResult<Vec<HelicityMaterializedAmplitudeRouteRuntime>> {
    match materialization.strategy {
        HelicityMaterializationStrategy::Quotient => validate_quotient_helicity_amplitude_routes(
            recurrence,
            materialization,
            schema,
            complete_domains,
            structural_zero_domains,
            residual_current_targets,
        ),
        HelicityMaterializationStrategy::RetainedProofGraph => {
            validate_retained_helicity_amplitude_routes(
                recurrence,
                materialization,
                schema,
                complete_domains,
                structural_zero_domains,
                residual_current_targets,
            )
        }
    }
}

fn validate_quotient_helicity_amplitude_routes(
    recurrence: &HelicityRecurrenceManifest,
    materialization: &HelicityRecurrenceMaterializationManifest,
    schema: &HelicityRecurrenceSchemaView,
    complete_domains: &BTreeSet<usize>,
    structural_zero_domains: &BTreeSet<usize>,
    residual_current_targets: &BTreeSet<usize>,
) -> RusticolResult<Vec<HelicityMaterializedAmplitudeRouteRuntime>> {
    let expected_root_count = recurrence
        .amplitude_classes
        .len()
        .checked_add(recurrence.residual_root_ids.len())
        .ok_or_else(|| {
            RusticolError::artifact("helicity materialized root count overflows usize")
        })?;
    let expected_route_count = recurrence
        .amplitude_classes
        .iter()
        .map(|class| class.members.len())
        .sum::<usize>()
        .checked_add(recurrence.residual_root_ids.len())
        .ok_or_else(|| {
            RusticolError::artifact("helicity materialized route count overflows usize")
        })?;
    if materialization.materialized_root_count != expected_root_count
        || materialization.amplitude_routes.len() != expected_route_count
    {
        return Err(RusticolError::artifact(
            "helicity materialized amplitude routes do not match proof class and residual coverage",
        ));
    }
    let mut output_indices = BTreeSet::new();
    for (root_id, root) in schema.roots.iter().enumerate() {
        if root.left_current_id >= schema.currents.len()
            || root.right_current_id >= schema.currents.len()
            || root.output_index >= schema.roots.len()
            || !output_indices.insert(root.output_index)
            || !root.helicity_weight.is_finite()
            || (root.helicity_weight - 1.0).abs() > 1.0e-12
        {
            return Err(RusticolError::artifact(format!(
                "helicity materialized amplitude root {root_id} is inconsistent with the final runtime schema"
            )));
        }
    }

    let mut runtime_routes = Vec::with_capacity(expected_route_count);
    let mut route_index = 0;
    let mut represented_domains = BTreeSet::new();
    for (materialized_root_id, class) in recurrence.amplitude_classes.iter().enumerate() {
        let representative_factor = class
            .members
            .iter()
            .find(|member| member.root_id == class.representative_root_id)
            .map(|member| member.factor)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "helicity amplitude class {:?} has no representative factor",
                    class.class_id
                ))
            })?;
        let root = &schema.roots[materialized_root_id];
        if residual_current_targets.contains(&root.left_current_id)
            || residual_current_targets.contains(&root.right_current_id)
        {
            return Err(RusticolError::artifact(format!(
                "proven helicity materialized amplitude root {materialized_root_id} depends on a residual current"
            )));
        }
        for member in &class.members {
            let route = &materialization.amplitude_routes[route_index];
            validate_helicity_factor(
                route.factor,
                &format!("helicity materialized amplitude route {route_index}"),
            )?;
            validate_complete_selector_ids(
                &route.selector_domain_ids,
                recurrence,
                &format!("helicity materialized amplitude route {route_index}"),
            )?;
            if route.materialized_root_id != materialized_root_id
                || route.selector_domain_ids != member.selector_domain_ids
                || route.residual
                || !helicity_factor_ratio_matches(
                    route.factor,
                    member.factor,
                    representative_factor,
                )
            {
                return Err(RusticolError::artifact(format!(
                    "helicity materialized amplitude route {route_index} disagrees with its proof member"
                )));
            }
            represented_domains.extend(route.selector_domain_ids.iter().copied());
            runtime_routes.push(HelicityMaterializedAmplitudeRouteRuntime {
                materialized_root_id: route.materialized_root_id,
                selector_domain_ids: route.selector_domain_ids.clone(),
                factor: route.factor,
                residual: false,
            });
            route_index += 1;
        }
    }
    for residual_index in 0..recurrence.residual_root_ids.len() {
        let route = &materialization.amplitude_routes[route_index];
        let materialized_root_id = recurrence.amplitude_classes.len() + residual_index;
        validate_complete_selector_ids(
            &route.selector_domain_ids,
            recurrence,
            &format!("helicity residual materialized amplitude route {route_index}"),
        )?;
        if route.materialized_root_id != materialized_root_id
            || route.selector_domain_ids.len() != 1
            || !route.residual
            || !helicity_factors_equal(route.factor, [1.0, 0.0])
        {
            return Err(RusticolError::artifact(format!(
                "helicity residual materialized amplitude route {route_index} is inconsistent"
            )));
        }
        represented_domains.extend(route.selector_domain_ids.iter().copied());
        runtime_routes.push(HelicityMaterializedAmplitudeRouteRuntime {
            materialized_root_id: route.materialized_root_id,
            selector_domain_ids: route.selector_domain_ids.clone(),
            factor: route.factor,
            residual: true,
        });
        route_index += 1;
    }
    let expected_nonzero_domains = complete_domains
        .difference(structural_zero_domains)
        .copied()
        .collect::<BTreeSet<_>>();
    if represented_domains != expected_nonzero_domains {
        return Err(RusticolError::artifact(
            "helicity materialized amplitude routes do not cover exactly the nonzero complete selector domains",
        ));
    }
    Ok(runtime_routes)
}

fn validate_retained_helicity_amplitude_routes(
    recurrence: &HelicityRecurrenceManifest,
    materialization: &HelicityRecurrenceMaterializationManifest,
    schema: &HelicityRecurrenceSchemaView,
    complete_domains: &BTreeSet<usize>,
    structural_zero_domains: &BTreeSet<usize>,
    residual_current_targets: &BTreeSet<usize>,
) -> RusticolResult<Vec<HelicityMaterializedAmplitudeRouteRuntime>> {
    if materialization.materialized_root_count != recurrence.amplitude_root_count
        || materialization.amplitude_routes.len() != recurrence.amplitude_root_count
    {
        return Err(RusticolError::artifact(
            "retained helicity amplitude routes must cover every proof root",
        ));
    }

    let mut expected_routes = vec![None; recurrence.amplitude_root_count];
    for class in &recurrence.amplitude_classes {
        for member in &class.members {
            if member.root_id >= expected_routes.len()
                || expected_routes[member.root_id]
                    .replace((member.selector_domain_ids.clone(), false))
                    .is_some()
            {
                return Err(RusticolError::artifact(format!(
                    "retained helicity amplitude root {} is covered more than once",
                    member.root_id
                )));
            }
        }
    }
    for root_id in &recurrence.residual_root_ids {
        if *root_id >= expected_routes.len()
            || expected_routes[*root_id]
                .replace((Vec::new(), true))
                .is_some()
        {
            return Err(RusticolError::artifact(format!(
                "retained helicity residual root {root_id} overlaps another proof root"
            )));
        }
    }
    if expected_routes.iter().any(Option::is_none) {
        return Err(RusticolError::artifact(
            "retained helicity amplitude routes do not cover the proof root space",
        ));
    }

    let mut output_indices = BTreeSet::new();
    let mut represented_domains = BTreeSet::new();
    let mut runtime_routes = Vec::with_capacity(materialization.amplitude_routes.len());
    for (root_id, (root, route)) in schema
        .roots
        .iter()
        .zip(&materialization.amplitude_routes)
        .enumerate()
    {
        if root.left_current_id >= schema.currents.len()
            || root.right_current_id >= schema.currents.len()
            || root.output_index >= schema.roots.len()
            || !output_indices.insert(root.output_index)
            || !root.helicity_weight.is_finite()
            || root.helicity_weight <= 0.0
        {
            return Err(RusticolError::artifact(format!(
                "retained helicity amplitude root {root_id} is inconsistent with the final runtime schema"
            )));
        }
        let (expected_domain_ids, residual) = expected_routes[root_id]
            .as_ref()
            .expect("retained root coverage checked above");
        validate_complete_selector_ids(
            &route.selector_domain_ids,
            recurrence,
            &format!("retained helicity amplitude route {root_id}"),
        )?;
        if route.materialized_root_id != root_id
            || route.residual != *residual
            || (!*residual
                && route.selector_domain_ids.as_slice() != expected_domain_ids.as_slice())
            || (*residual && route.selector_domain_ids.is_empty())
            || !helicity_factors_equal(route.factor, [root.helicity_weight.sqrt().recip(), 0.0])
        {
            return Err(RusticolError::artifact(format!(
                "retained helicity amplitude route {root_id} disagrees with its proof root"
            )));
        }
        if !*residual
            && (residual_current_targets.contains(&root.left_current_id)
                || residual_current_targets.contains(&root.right_current_id))
        {
            return Err(RusticolError::artifact(format!(
                "proven retained helicity amplitude root {root_id} depends on a residual current"
            )));
        }
        represented_domains.extend(route.selector_domain_ids.iter().copied());
        runtime_routes.push(HelicityMaterializedAmplitudeRouteRuntime {
            materialized_root_id: root_id,
            selector_domain_ids: route.selector_domain_ids.clone(),
            factor: route.factor,
            residual: *residual,
        });
    }
    let expected_nonzero_domains = complete_domains
        .difference(structural_zero_domains)
        .copied()
        .collect::<BTreeSet<_>>();
    if represented_domains != expected_nonzero_domains {
        return Err(RusticolError::artifact(
            "retained helicity amplitude routes do not cover exactly the nonzero complete selector domains",
        ));
    }
    Ok(runtime_routes)
}

fn validate_helicity_materialized_selector_schedules(
    recurrence: &HelicityRecurrenceManifest,
    materialization: &HelicityRecurrenceMaterializationManifest,
    schema: &HelicityRecurrenceSchemaView,
    complete_domains: &BTreeSet<usize>,
    structural_zero_domains: &BTreeSet<usize>,
) -> RusticolResult<Vec<HelicityMaterializedSelectorScheduleRuntime>> {
    let expected_domain_ids = complete_domains.iter().copied().collect::<Vec<_>>();
    if materialization.selector_schedules.len() != expected_domain_ids.len()
        || materialization
            .selector_schedules
            .iter()
            .map(|schedule| schedule.selector_domain_id)
            .ne(expected_domain_ids.iter().copied())
    {
        return Err(RusticolError::artifact(
            "helicity materialized selector schedules must cover complete domains in order",
        ));
    }
    let chunks = schema.compiled_chunks.as_ref().ok_or_else(|| {
        RusticolError::artifact(
            "helicity materialized selector schedules require serialized stage evaluators",
        )
    })?;
    let mut runtime_schedules = Vec::with_capacity(materialization.selector_schedules.len());
    for schedule in &materialization.selector_schedules {
        validate_sorted_unique_ids(
            &schedule.active_current_ids,
            schema.currents.len(),
            &format!(
                "helicity selector schedule {} active current ids",
                schedule.selector_domain_id
            ),
        )?;
        validate_sorted_unique_ids(
            &schedule.active_root_ids,
            schema.roots.len(),
            &format!(
                "helicity selector schedule {} active root ids",
                schedule.selector_domain_id
            ),
        )?;
        let structural_zero = structural_zero_domains.contains(&schedule.selector_domain_id);
        if schedule.structural_zero != structural_zero
            || (structural_zero
                && (!schedule.active_current_ids.is_empty()
                    || !schedule.active_root_ids.is_empty()))
        {
            return Err(RusticolError::artifact(format!(
                "helicity selector schedule {} disagrees with structural-zero proof coverage",
                schedule.selector_domain_id
            )));
        }
        let expected_root_ids = materialization
            .amplitude_routes
            .iter()
            .filter(|route| {
                route
                    .selector_domain_ids
                    .contains(&schedule.selector_domain_id)
            })
            .map(|route| route.materialized_root_id)
            .collect::<BTreeSet<_>>();
        if expected_root_ids
            != schedule
                .active_root_ids
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
        {
            return Err(RusticolError::artifact(format!(
                "helicity selector schedule {} active roots disagree with amplitude routes",
                schedule.selector_domain_id
            )));
        }
        if !structural_zero && schedule.active_root_ids.is_empty() {
            return Err(RusticolError::artifact(format!(
                "helicity selector schedule {} has no active amplitude roots",
                schedule.selector_domain_id
            )));
        }

        let active_current_ids = schedule
            .active_current_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        for root_id in &schedule.active_root_ids {
            let root = &schema.roots[*root_id];
            if !active_current_ids.contains(&root.left_current_id)
                || !active_current_ids.contains(&root.right_current_id)
            {
                return Err(RusticolError::artifact(format!(
                    "helicity selector schedule {} omits a parent of active root {root_id}",
                    schedule.selector_domain_id
                )));
            }
        }
        if !structural_zero {
            let domain = &recurrence.selector_domains[schedule.selector_domain_id];
            let active_source_ids = active_current_ids
                .iter()
                .filter(|current_id| schema.currents[**current_id].is_source)
                .copied()
                .collect::<BTreeSet<_>>();
            let active_source_labels = active_source_ids
                .iter()
                .map(|current_id| {
                    schema
                        .sources
                        .get(current_id)
                        .map(|source| source.external_label)
                        .ok_or_else(|| {
                            RusticolError::artifact(format!(
                                "helicity selector schedule {} references source current {} without source metadata",
                                schedule.selector_domain_id, current_id
                            ))
                        })
                })
                .collect::<RusticolResult<BTreeSet<_>>>()?;
            let expected_source_labels = domain
                .source_states
                .iter()
                .map(|state| state.external_label)
                .collect::<BTreeSet<_>>();
            if active_source_labels != expected_source_labels {
                return Err(RusticolError::artifact(format!(
                    "helicity selector schedule {} source-label coverage disagrees with its complete domain",
                    schedule.selector_domain_id
                )));
            }
        }

        let mut covered_non_source_ids = BTreeSet::new();
        let mut active_stage_chunk_indices = Vec::with_capacity(chunks.stages.len());
        for stage in &chunks.stages {
            let indices = stage
                .current_ids_by_chunk
                .iter()
                .enumerate()
                .filter_map(|(chunk_index, current_ids)| {
                    current_ids
                        .iter()
                        .any(|current_id| active_current_ids.contains(current_id))
                        .then_some(chunk_index)
                })
                .collect::<Vec<_>>();
            for chunk_index in &indices {
                covered_non_source_ids.extend(
                    stage.current_ids_by_chunk[*chunk_index]
                        .intersection(&active_current_ids)
                        .copied()
                        .filter(|current_id| !schema.currents[*current_id].is_source),
                );
            }
            active_stage_chunk_indices.push(indices);
        }
        let required_non_source_ids = active_current_ids
            .iter()
            .copied()
            .filter(|current_id| !schema.currents[*current_id].is_source)
            .collect::<BTreeSet<_>>();
        if covered_non_source_ids != required_non_source_ids {
            let missing = required_non_source_ids
                .difference(&covered_non_source_ids)
                .copied()
                .collect::<Vec<_>>();
            return Err(RusticolError::artifact(format!(
                "helicity selector schedule {} has active non-source currents without compiled stage output coverage: {missing:?}",
                schedule.selector_domain_id
            )));
        }

        let active_root_ids = schedule
            .active_root_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let active_amplitude_chunk_indices = chunks
            .amplitude_root_ids_by_chunk
            .iter()
            .enumerate()
            .filter_map(|(chunk_index, root_ids)| {
                root_ids
                    .iter()
                    .any(|root_id| active_root_ids.contains(root_id))
                    .then_some(chunk_index)
            })
            .collect::<Vec<_>>();
        let covered_root_ids = active_amplitude_chunk_indices
            .iter()
            .flat_map(|chunk_index| {
                chunks.amplitude_root_ids_by_chunk[*chunk_index]
                    .intersection(&active_root_ids)
                    .copied()
            })
            .collect::<BTreeSet<_>>();
        if covered_root_ids != active_root_ids {
            return Err(RusticolError::artifact(format!(
                "helicity selector schedule {} has active roots without compiled amplitude output coverage",
                schedule.selector_domain_id
            )));
        }
        let current_is_source = schema
            .currents
            .iter()
            .map(|current| current.is_source)
            .collect::<Vec<_>>();
        validate_compiled_selector_schedule_closure(
            &format!("helicity selector schedule {}", schedule.selector_domain_id),
            &current_is_source,
            chunks,
            &active_stage_chunk_indices,
            &active_amplitude_chunk_indices,
            Some(&active_current_ids),
            Some(&active_root_ids),
        )?;
        runtime_schedules.push(HelicityMaterializedSelectorScheduleRuntime {
            selector_domain_id: schedule.selector_domain_id,
            active_current_ids: schedule.active_current_ids.clone(),
            active_root_ids: schedule.active_root_ids.clone(),
            active_stage_chunk_indices,
            active_amplitude_chunk_indices,
            structural_zero,
        });
    }
    Ok(runtime_schedules)
}

fn validate_complete_selector_ids(
    selector_domain_ids: &[usize],
    recurrence: &HelicityRecurrenceManifest,
    context: &str,
) -> RusticolResult<()> {
    if selector_domain_ids.is_empty()
        || selector_domain_ids.iter().any(|domain_id| {
            recurrence
                .selector_domains
                .get(*domain_id)
                .is_none_or(|domain| !domain.complete)
        })
        || selector_domain_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(RusticolError::artifact(format!(
            "{context} requires distinct ordered complete selector domains"
        )));
    }
    Ok(())
}

fn helicity_factor_ratio_matches(
    actual: [f64; 2],
    numerator: [f64; 2],
    denominator: [f64; 2],
) -> bool {
    let product = [
        actual[0] * denominator[0] - actual[1] * denominator[1],
        actual[0] * denominator[1] + actual[1] * denominator[0],
    ];
    product
        .into_iter()
        .zip(numerator)
        .all(|(actual, expected)| {
            let tolerance = 64.0 * f64::EPSILON * expected.abs().max(1.0);
            (actual - expected).abs() <= tolerance
        })
}

fn validate_helicity_recurrence_proof(
    class_id: &str,
    class_prefix: &str,
    proof: &HelicityRecurrenceProofManifest,
) -> RusticolResult<()> {
    validate_sha256_digest(&proof.digest, class_id)?;
    if proof.status != "proven"
        || proof.algorithm != HELICITY_RECURRENCE_PROOF_ALGORITHM
        || class_id != format!("{class_prefix}{}", proof.digest)
        || proof
            .transition_contract_ids
            .iter()
            .any(|contract| contract.is_empty())
        || proof
            .transition_contract_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(RusticolError::artifact(format!(
            "helicity recurrence class {class_id:?} has invalid proof metadata"
        )));
    }
    Ok(())
}

fn validate_sha256_digest(digest: &str, context: &str) -> RusticolResult<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RusticolError::artifact(format!(
            "{context} requires a lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

fn validate_helicity_external_labels(
    labels: impl IntoIterator<Item = usize>,
    external_count: usize,
    context: &str,
) -> RusticolResult<Vec<usize>> {
    let labels = labels.into_iter().collect::<Vec<_>>();
    if labels.is_empty()
        || labels
            .iter()
            .any(|label| *label == 0 || *label > external_count)
        || labels.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(RusticolError::artifact(format!(
            "{context} contains invalid, duplicate, or unordered external labels"
        )));
    }
    Ok(labels)
}

fn validate_sorted_unique_ids(
    ids: &[usize],
    upper_bound: usize,
    context: &str,
) -> RusticolResult<()> {
    if ids.iter().any(|id| *id >= upper_bound) || ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(RusticolError::artifact(format!(
            "{context} contain stale, duplicate, or unordered ids"
        )));
    }
    Ok(())
}

fn validate_helicity_factor(factor: [f64; 2], context: &str) -> RusticolResult<()> {
    if !factor.iter().all(|component| component.is_finite()) || factor == [0.0, 0.0] {
        return Err(RusticolError::artifact(format!(
            "{context} has an invalid recurrence factor"
        )));
    }
    Ok(())
}

fn helicity_factors_equal(left: [f64; 2], right: [f64; 2]) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn build_runtime_parameter_slots(
    parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<BTreeMap<String, RuntimeParameterSlots>> {
    let mut direct = BTreeMap::new();
    let mut complex_components: BTreeMap<String, (Option<usize>, Option<usize>)> = BTreeMap::new();
    for parameter in parameters {
        if parameter.kind == "derived_parameter_component" {
            continue;
        }
        if let Some(runtime_name) = &parameter.runtime_name {
            let slots = complex_components
                .entry(runtime_name.clone())
                .or_insert((None, None));
            match parameter.complex_component.as_deref() {
                Some("real") if slots.0.replace(parameter.parameter_index).is_none() => {}
                Some("imag") if slots.1.replace(parameter.parameter_index).is_none() => {}
                Some(component) => {
                    return Err(RusticolError::invalid_argument(format!(
                        "runtime model parameter {runtime_name:?} has duplicate or invalid component {component:?}"
                    )));
                }
                None => {
                    return Err(RusticolError::invalid_argument(format!(
                        "runtime model parameter {runtime_name:?} is missing component metadata"
                    )));
                }
            }
        } else if direct
            .insert(
                parameter.name.clone(),
                RuntimeParameterSlots {
                    real: parameter.parameter_index,
                    imaginary: None,
                },
            )
            .is_some()
        {
            return Err(RusticolError::invalid_argument(format!(
                "duplicate runtime model parameter name {:?}",
                parameter.name
            )));
        }
    }
    for (name, (real, imaginary)) in complex_components {
        let (Some(real), Some(imaginary)) = (real, imaginary) else {
            return Err(RusticolError::invalid_argument(format!(
                "runtime model parameter {name:?} requires real and imaginary slots"
            )));
        };
        if direct
            .insert(
                name.clone(),
                RuntimeParameterSlots {
                    real,
                    imaginary: Some(imaginary),
                },
            )
            .is_some()
        {
            return Err(RusticolError::invalid_argument(format!(
                "duplicate runtime model parameter name {name:?}"
            )));
        }
    }
    Ok(direct)
}

impl ExecutionRuntime {
    pub(super) fn from_manifest(mut manifest: ExecutionManifest) -> RusticolResult<Self> {
        validate_helicity_sum_execution(&manifest)?;
        validate_helicity_selector_executions(&manifest)?;
        validate_color_selector_executions(&manifest)?;
        let helicity_sum_runtime = manifest
            .helicity_sum_execution
            .take()
            .map(|sum_manifest| Self::from_manifest(*sum_manifest).map(Box::new))
            .transpose()?;
        let helicity_selector_manifests =
            std::mem::take(&mut manifest.helicity_selector_executions);
        let mut helicity_selector_runtimes = Vec::with_capacity(helicity_selector_manifests.len());
        let mut helicity_selector_runtime_schedule_modes =
            Vec::with_capacity(helicity_selector_manifests.len());
        let mut helicity_selector_lane_by_domain = BTreeMap::new();
        for record in helicity_selector_manifests {
            let lane_index = helicity_selector_runtimes.len();
            for selector_domain_id in record.selector_domain_ids {
                helicity_selector_lane_by_domain.insert(selector_domain_id, lane_index);
            }
            helicity_selector_runtime_schedule_modes.push(record.schedule_mode);
            helicity_selector_runtimes.push(Box::new(Self::from_manifest(*record.execution)?));
        }
        let color_selector_runtimes = manifest
            .color_selector_executions
            .drain(..)
            .map(|record| {
                Ok((
                    record.materialized_sector_id,
                    Box::new(Self::from_manifest(*record.execution)?),
                ))
            })
            .collect::<RusticolResult<BTreeMap<_, _>>>()?;
        let topology_replay = manifest.compiled.lc_topology_replay.as_ref();
        let topology_replay_data = build_lc_topology_replay_data(topology_replay)?;
        let color_topology_replay = manifest.compiled.color_topology_replay.as_ref();
        let color_topology_replay_data =
            build_color_topology_replay_data(color_topology_replay, &manifest.color_accuracy)?;
        if !topology_replay_data.mappings.is_empty()
            && !color_topology_replay_data.mappings.is_empty()
        {
            return Err(RusticolError::artifact(
                "an artifact cannot enable LC and contracted-color topology replay together",
            ));
        }
        let compiled_color_execution_plan = manifest
            .compiled
            .stage_evaluators
            .as_ref()
            .map(|evaluators| {
                build_compiled_color_execution_plan(
                    evaluators,
                    &manifest.runtime_schema,
                    &topology_replay_data.materialized_sector_ids,
                )
            })
            .transpose()?
            .flatten();
        let topology_replay_mappings = topology_replay_data.mappings;
        let topology_replay_public_mappings = topology_replay_mappings.clone();
        let helicity_recurrence = build_helicity_recurrence_runtime(&manifest)?;
        let external_is_initial = manifest
            .runtime_schema
            .external_particles
            .iter()
            .map(|particle| particle.role == "initial")
            .collect::<Vec<_>>();
        let particle_masses = manifest
            .runtime_schema
            .model
            .as_ref()
            .map(|model| {
                model
                    .particles
                    .iter()
                    .map(|particle| (particle.pdg, particle.mass))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let particle_mass_parameter_names = manifest
            .runtime_schema
            .model
            .as_ref()
            .map(|model| {
                model
                    .particles
                    .iter()
                    .filter_map(|particle| {
                        particle
                            .mass_parameter
                            .as_ref()
                            .map(|name| (particle.pdg, name.clone()))
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let mut model_parameters = manifest.runtime_schema.model_parameters.clone();
        model_parameters.sort_by_key(|parameter| parameter.parameter_index);
        let model_parameter_values_f64 = model_parameters
            .iter()
            .map(|parameter| parameter.default)
            .collect::<Vec<_>>();
        let model_parameter_name_to_index = model_parameters
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
            .collect::<BTreeMap<_, _>>();
        let model_parameter_runtime_slots = build_runtime_parameter_slots(&model_parameters)?;
        let color_factor_in_contraction = manifest
            .runtime_schema
            .amplitude_stage
            .color_contraction
            .as_ref()
            .map(|contraction| contraction.supported && contraction.includes_color_factor)
            .unwrap_or(false);
        let (
            normalization_factor,
            normalization_color_factor,
            normalization_average_factor,
            normalization_identical_factor,
            normalization_qcd_coupling_power,
            normalization_electroweak_coupling_power,
        ) = manifest
            .runtime_schema
            .normalization
            .as_ref()
            .map(|normalization| {
                let color_factor = if color_factor_in_contraction {
                    1.0
                } else {
                    normalization.color_factor
                };
                (
                    color_factor * normalization.global_coupling_factor
                        / (normalization.average_factor * normalization.identical_factor),
                    color_factor,
                    normalization.average_factor,
                    normalization.identical_factor,
                    normalization.qcd_coupling_power,
                    normalization.electroweak_coupling_power,
                )
            })
            .unwrap_or((1.0, 1.0, 1.0, 1.0, 0, 0));
        Ok(Self {
            process: manifest.process,
            key: manifest.key,
            color_accuracy: manifest.color_accuracy,
            external_count: manifest.external_pdg_order.len(),
            external_pdg_order: manifest.external_pdg_order,
            parameter_count: manifest
                .runtime_schema
                .parameter_layout
                .value_component_count
                + manifest
                    .runtime_schema
                    .parameter_layout
                    .momentum_parameter_count
                + manifest
                    .runtime_schema
                    .parameter_layout
                    .model_parameter_count,
            value_parameter_count: manifest
                .runtime_schema
                .parameter_layout
                .value_component_count,
            momentum_parameter_count: manifest
                .runtime_schema
                .parameter_layout
                .momentum_parameter_count,
            current_count: manifest.dag_summary.current_count,
            source_count: manifest.dag_summary.source_count,
            interaction_count: manifest.dag_summary.interaction_count,
            stage_count: manifest.runtime_schema.stages.len(),
            amplitude_output_count: manifest.runtime_schema.amplitude_stage.output_count,
            lc_topology_replay_enabled: !topology_replay_mappings.is_empty(),
            lc_topology_replay_mappings: Arc::new(topology_replay_mappings),
            lc_topology_replay_public_mappings: topology_replay_public_mappings,
            lc_topology_replay_routes: topology_replay_data.routes,
            lc_topology_replay_materialized_sector_ids: topology_replay_data
                .materialized_sector_ids,
            lc_resolved_replay_plan: None,
            lc_resolved_replay_selection_cache: None,
            lc_replay_flat_momenta_scratch: Vec::new(),
            lc_replay_target_components_scratch: Vec::new(),
            color_topology_replay_enabled: !color_topology_replay_data.mappings.is_empty(),
            color_topology_replay_mappings: Arc::new(color_topology_replay_data.mappings),
            color_replay_flat_momenta_scratch: Vec::new(),
            helicity_recurrence,
            compiled_helicity_execution_plan: None,
            compiled_color_execution_plan,
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            compiled_direct_runtime: None,
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            compiled_direct_color_schedules: BTreeMap::new(),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            compiled_direct_helicity_schedules: BTreeMap::new(),
            helicity_sum_runtime,
            helicity_selector_runtimes,
            helicity_selector_runtime_schedule_modes,
            helicity_selector_lane_by_domain,
            color_selector_runtimes,
            runtime_unavailable_message: manifest.compiled.runtime_unavailable_message,
            sources: manifest.runtime_schema.source_fill.sources,
            momentum_slots: manifest.runtime_schema.momentum_slots,
            external_is_initial,
            particle_masses,
            particle_mass_parameter_names,
            normalization_factor,
            normalization_color_factor,
            normalization_average_factor,
            normalization_identical_factor,
            normalization_qcd_coupling_power,
            normalization_electroweak_coupling_power,
            model_parameters,
            model_parameter_name_to_index,
            model_parameter_runtime_slots,
            model_parameter_values_f64,
            model_parameter_evaluator: None,
            physics_reduction_override: manifest.physics_reduction,
            physics: None,
            stages: None,
            amplitude_stage: None,
            state_scratch_f64: Vec::new(),
            state_scratch_f64_requires_clear: false,
            values_scratch_f64: Vec::new(),
        })
    }

    fn compiled_direct_routed_reduction_footprint(
        &self,
        amplitude: &AmplitudeRuntime,
        physics: &PhysicsRuntime,
    ) -> RusticolResult<CompiledDirectRoutedReductionFootprint> {
        if !self.lc_topology_replay_enabled {
            return Ok(CompiledDirectRoutedReductionFootprint::default());
        }
        let materialized_sector_by_id =
            self.lc_materialized_sectors_by_id_from_groups(physics, &amplitude.raw_sum_groups)?;
        let replay_plan = physics.lc_resolved_replay_plan(
            &self.lc_topology_replay_public_mappings,
            &self.lc_topology_replay_routes,
            &materialized_sector_by_id,
        )?;
        let selection_key = physics.lc_resolved_replay_selection_key(None, None)?;
        let selection =
            physics.select_lc_resolved_replay_plan_for_key(&replay_plan, &selection_key)?;
        let maximum_source_component_count = selection
            .source_groups
            .iter()
            .map(|group| group.source_component_count)
            .max()
            .ok_or_else(|| {
                RusticolError::artifact(
                    "enabled LC topology replay has no authenticated source group",
                )
            })?;
        let maximum_target_component_count = selection
            .helicity_indices
            .len()
            .checked_mul(selection.color_indices.len())
            .filter(|count| *count != 0)
            .ok_or_else(|| {
                RusticolError::artifact(
                    "enabled LC topology replay has no authenticated target components",
                )
            })?;
        Ok(CompiledDirectRoutedReductionFootprint {
            maximum_source_component_count,
            maximum_target_component_count,
        })
    }

    pub(super) fn load_from_manifest_with_store(
        manifest: ExecutionManifest,
        payloads: &EvaluatorPayloadStore,
        sizing_physics: &PhysicsRuntime,
    ) -> RusticolResult<Self> {
        Self::load_from_manifest_with_store_and_replay_footprint(
            manifest,
            payloads,
            Some(CompiledDirectRoutedReductionFootprint::default()),
            sizing_physics,
        )
    }

    fn load_from_manifest_with_store_and_replay_footprint(
        mut manifest: ExecutionManifest,
        payloads: &EvaluatorPayloadStore,
        inherited_routed_footprint: Option<CompiledDirectRoutedReductionFootprint>,
        inherited_sizing_physics: &PhysicsRuntime,
    ) -> RusticolResult<Self> {
        validate_helicity_sum_execution(&manifest)?;
        validate_helicity_selector_executions(&manifest)?;
        validate_color_selector_executions(&manifest)?;
        let compiled_plane_arena = validate_compiled_plane_arena_contract(&manifest)?;
        let compiled_momentum_slot_ids = manifest.compiled.momentum_slot_ids.take();
        ensure_execution_capabilities_supported(&manifest)?;
        let helicity_sum_manifest = manifest.helicity_sum_execution.take();
        let helicity_selector_manifests =
            std::mem::take(&mut manifest.helicity_selector_executions);
        let color_selector_manifests = std::mem::take(&mut manifest.color_selector_executions);
        let compiled_color_payload = manifest.color_contraction_payload.take();
        let (stage_evaluators, helicity_recurrence) =
            detach_compiled_helicity_load_inputs(&mut manifest)?;
        let model_parameter_evaluator = manifest.compiled.model_parameter_evaluator.take();
        let replay_materialized_sector_ids =
            build_lc_topology_replay_data(manifest.compiled.lc_topology_replay.as_ref())?
                .materialized_sector_ids;
        let compiled_color_execution_plan = stage_evaluators
            .as_ref()
            .map(|evaluators| {
                build_compiled_color_execution_plan(
                    evaluators,
                    &manifest.runtime_schema,
                    &replay_materialized_sector_ids,
                )
            })
            .transpose()?
            .flatten();
        drop(replay_materialized_sector_ids);
        let amplitude_stage_manifest = manifest.runtime_schema.amplitude_stage.clone();
        let mut runtime = Self::from_manifest(manifest)?;
        if compiled_plane_arena {
            // These ids were selected at generation and checked alongside
            // the final leaf bindings above. Loading only binds that list;
            // it does not rediscover which momentum sums are live.
            runtime.momentum_slots = compiled_momentum_slot_ids
                .expect("validated compiled momentum-slot selection")
                .into_iter()
                .map(|id| runtime.momentum_slots[id].clone())
                .collect();
        }
        runtime.helicity_recurrence = helicity_recurrence;
        runtime.compiled_color_execution_plan = compiled_color_execution_plan;
        let sizing_physics = if let Some(reduction) = runtime.physics_reduction_override.as_ref() {
            let mut manifest = inherited_sizing_physics.manifest.clone();
            manifest.reduction = reduction.clone();
            PhysicsRuntime::new(manifest)?
        } else {
            inherited_sizing_physics.clone()
        };
        let mut compiled_color_contraction = load_compiled_symmetric_group_color_contraction(
            compiled_color_payload.as_ref(),
            payloads,
            &sizing_physics,
        )?;
        let routed_footprint = if runtime.lc_topology_replay_enabled {
            if let Some(stage_evaluators) = stage_evaluators.as_ref() {
                let sizing_amplitude = AmplitudeRuntime::load_reducer_only(
                    &amplitude_stage_manifest,
                    &stage_evaluators.amplitude_stage,
                )?;
                let own = runtime.compiled_direct_routed_reduction_footprint(
                    &sizing_amplitude,
                    &sizing_physics,
                )?;
                inherited_routed_footprint.map(|inherited| inherited.merge(own))
            } else {
                // The replay may route into a nested Direct engine, but this
                // legacy parent exposes no amplitude metadata from which to
                // authenticate its source/target shape.
                None
            }
        } else {
            inherited_routed_footprint
        };
        runtime.helicity_sum_runtime = helicity_sum_manifest
            .map(|sum_manifest| {
                Self::load_from_manifest_with_store_and_replay_footprint(
                    *sum_manifest,
                    payloads,
                    routed_footprint,
                    &sizing_physics,
                )
                .map(Box::new)
            })
            .transpose()?;
        runtime.helicity_selector_runtimes = Vec::with_capacity(helicity_selector_manifests.len());
        runtime.helicity_selector_runtime_schedule_modes =
            Vec::with_capacity(helicity_selector_manifests.len());
        runtime.helicity_selector_lane_by_domain.clear();
        for record in helicity_selector_manifests {
            let lane_index = runtime.helicity_selector_runtimes.len();
            for selector_domain_id in record.selector_domain_ids {
                runtime
                    .helicity_selector_lane_by_domain
                    .insert(selector_domain_id, lane_index);
            }
            runtime
                .helicity_selector_runtime_schedule_modes
                .push(record.schedule_mode);
            runtime.helicity_selector_runtimes.push(Box::new(
                Self::load_from_manifest_with_store_and_replay_footprint(
                    *record.execution,
                    payloads,
                    routed_footprint,
                    &sizing_physics,
                )?,
            ));
        }
        runtime.color_selector_runtimes = color_selector_manifests
            .into_iter()
            .map(|record| {
                Ok((
                    record.materialized_sector_id,
                    Box::new(Self::load_from_manifest_with_store_and_replay_footprint(
                        *record.execution,
                        payloads,
                        routed_footprint,
                        &sizing_physics,
                    )?),
                ))
            })
            .collect::<RusticolResult<BTreeMap<_, _>>>()?;
        if let Some(manifest) = model_parameter_evaluator {
            if manifest.kind != "generic-model-parameter-evaluator" {
                return Err(RusticolError::invalid_argument(format!(
                    "unsupported model-parameter evaluator kind {:?}",
                    manifest.kind
                )));
            }
            runtime.model_parameter_evaluator = Some(ModelParameterEvaluatorRuntime {
                input_parameter_indices: manifest.input_parameter_indices,
                outputs: manifest.outputs,
                evaluator: EvaluatorGroup::load_from_store(&manifest.evaluator, payloads)?,
            });
            runtime.refresh_derived_model_parameters()?;
        }
        if let Some(stage_evaluators) = stage_evaluators {
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            if compiled_plane_arena {
                #[cfg(feature = "symbolica-runtime")]
                let (stages, mut amplitude) = match compiled_color_contraction.take() {
                    Some(color) => (
                        Vec::new(),
                        AmplitudeRuntime::load_reducer_only_with_compiled_color(
                            &amplitude_stage_manifest,
                            &stage_evaluators.amplitude_stage,
                            Some(color),
                        )?,
                    ),
                    None => {
                        // Preserve the public exact-precision API for ordinary
                        // compiled artifacts without instantiating a second f64
                        // evaluator representation. The selected-helicity SG
                        // diagnostic rejects exact execution and therefore uses
                        // only the reducer construction above.
                        (
                            stage_evaluators
                                .stages
                                .iter()
                                .map(|stage| StageRuntime::load_exact_from_plane(stage, payloads))
                                .collect::<RusticolResult<Vec<_>>>()?,
                            AmplitudeRuntime::load_exact_from_plane(
                                &amplitude_stage_manifest,
                                &stage_evaluators.amplitude_stage,
                                payloads,
                            )?,
                        )
                    }
                };
                #[cfg(not(feature = "symbolica-runtime"))]
                let (stages, mut amplitude) = (
                    Vec::new(),
                    match compiled_color_contraction.take() {
                        Some(color) => AmplitudeRuntime::load_reducer_only_with_compiled_color(
                            &amplitude_stage_manifest,
                            &stage_evaluators.amplitude_stage,
                            Some(color),
                        )?,
                        None => AmplitudeRuntime::load_reducer_only(
                            &amplitude_stage_manifest,
                            &stage_evaluators.amplitude_stage,
                        )?,
                    },
                );
                let maximum_resolved_component_count = sizing_physics
                    .manifest
                    .helicities
                    .len()
                    .checked_mul(sizing_physics.manifest.color_components.len())
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "compiled Direct-Arena resolved footprint overflows",
                        )
                    })?;
                let reduction_footprint = if let Some(routed_footprint) = routed_footprint {
                    amplitude.compiled_direct_reduction_footprint(
                        maximum_resolved_component_count,
                        routed_footprint,
                    )?
                } else {
                    CompiledDirectReductionFootprint::Unauthenticated
                };
                let direct =
                    compiled_direct_prototype::CompiledDirectEnginePrototype::load_production(
                        &stage_evaluators.stages,
                        &stage_evaluators.amplitude_stage,
                        payloads,
                        &runtime.sources,
                        runtime.value_parameter_count,
                        runtime.momentum_parameter_count,
                        runtime.model_parameter_values_f64.len(),
                        stage_evaluators.amplitude_stage.output_length,
                        reduction_footprint,
                    )?;
                amplitude
                    .prepare_compiled_symmetric_group_workspace(direct.reduction_tile_capacity())?;
                runtime.stages = Some(stages);
                runtime.amplitude_stage = Some(amplitude);
                let mut color_schedules = BTreeMap::new();
                if let Some(color_plan) = runtime.compiled_color_execution_plan.as_ref() {
                    for (sector_id, schedule) in &color_plan.schedules_by_materialized_sector {
                        color_schedules
                            .insert(*sector_id, direct.bind_color_schedule(schedule.as_ref())?);
                    }
                }
                runtime.compiled_direct_color_schedules = color_schedules;
                runtime.compiled_direct_runtime = Some(direct);
            } else {
                if compiled_color_contraction.is_some() {
                    return Err(RusticolError::compatibility(
                        "compiled symmetric-group FFT diagnostic requires compiled-plane-arena-v1",
                    ));
                }
                let stages = stage_evaluators
                    .stages
                    .iter()
                    .map(|stage| StageRuntime::load(stage, payloads))
                    .collect::<RusticolResult<Vec<_>>>()?;
                runtime.stages = Some(stages);
                runtime.amplitude_stage = Some(AmplitudeRuntime::load(
                    &amplitude_stage_manifest,
                    &stage_evaluators.amplitude_stage,
                    payloads,
                )?);
            }
            #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
            {
                if compiled_plane_arena {
                    return Err(RusticolError::unsupported_runtime_capability(
                        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
                        "this Rusticol build has no compiled DirectApplication backend",
                    ));
                }
                let stages = stage_evaluators
                    .stages
                    .iter()
                    .map(|stage| StageRuntime::load(stage, payloads))
                    .collect::<RusticolResult<Vec<_>>>()?;
                runtime.stages = Some(stages);
                runtime.amplitude_stage = Some(AmplitudeRuntime::load(
                    &amplitude_stage_manifest,
                    &stage_evaluators.amplitude_stage,
                    payloads,
                )?);
            }
        }
        runtime.validate_color_topology_replay_amplitude()?;
        Ok(runtime)
    }

    fn validate_color_topology_replay_amplitude(&self) -> RusticolResult<()> {
        if !self.color_topology_replay_enabled {
            if self
                .amplitude_stage
                .as_ref()
                .and_then(|amplitude| amplitude.color_topology_replay.as_ref())
                .is_some()
            {
                return Err(RusticolError::artifact(
                    "amplitude color replay is present without its proof certificate",
                ));
            }
            return Ok(());
        }
        let amplitude = self.amplitude_stage.as_ref().ok_or_else(|| {
            RusticolError::artifact("color topology replay proof has no executable amplitude stage")
        })?;
        let amplitude_mappings = amplitude.color_topology_replay_mappings().ok_or_else(|| {
            RusticolError::artifact("color topology replay proof has no amplitude-level gather")
        })?;
        if amplitude_mappings.as_slice() != self.color_topology_replay_mappings.as_slice() {
            return Err(RusticolError::artifact(
                "color topology replay proof and amplitude gather mappings disagree",
            ));
        }
        Ok(())
    }

    pub(super) fn execution_unavailable_error(&self) -> RusticolError {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return RusticolError::invalid_argument(format!(
                "generic DAG schema-v3 artifact for {} loaded successfully, but it has no \
                 serialized generic stage evaluators",
                self.process
            ));
        }
        let detail = self.runtime_unavailable_message.as_deref().unwrap_or(
            "generic source filling and staged evaluator execution from schema-v3 metadata remain pending",
        );
        RusticolError::invalid_argument(format!(
            "generic DAG schema-v3 artifact for {} loaded successfully, but generic Rusticol \
             execution is not wired yet; {detail}",
            self.process,
        ))
    }
}

fn validate_helicity_sum_execution(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let Some(sum_manifest) = manifest.helicity_sum_execution.as_deref() else {
        return Ok(());
    };
    if sum_manifest.helicity_sum_execution.is_some()
        || !sum_manifest.helicity_selector_executions.is_empty()
    {
        return Err(RusticolError::integrity(
            "compiled helicity-sum execution cannot contain another auxiliary execution",
        ));
    }
    if sum_manifest.physics_reduction.is_none() {
        return Err(RusticolError::integrity(
            "compiled helicity-sum execution has no reduction override",
        ));
    }
    if sum_manifest
        .runtime_schema
        .helicity_recurrence
        .as_ref()
        .is_some_and(|recurrence| recurrence.materialization.is_some())
    {
        return Err(RusticolError::integrity(
            "compiled helicity-sum execution cannot contain helicity materialization",
        ));
    }
    if sum_manifest.process != manifest.process
        || sum_manifest.key != manifest.key
        || sum_manifest.color_accuracy != manifest.color_accuracy
        || sum_manifest.external_pdg_order != manifest.external_pdg_order
    {
        return Err(RusticolError::integrity(
            "compiled helicity-sum execution does not match its primary process contract",
        ));
    }
    Ok(())
}

fn validate_helicity_selector_executions(manifest: &ExecutionManifest) -> RusticolResult<()> {
    if manifest.helicity_selector_executions.is_empty() {
        return Ok(());
    }
    let recurrence = manifest
        .runtime_schema
        .helicity_recurrence
        .as_ref()
        .filter(|recurrence| recurrence.materialization.is_some())
        .ok_or_else(|| {
            RusticolError::integrity(
                "compiled helicity-selector executions require primary helicity materialization",
            )
        })?;
    let mut assigned_domains = BTreeSet::new();
    for (lane_index, record) in manifest.helicity_selector_executions.iter().enumerate() {
        if record.selector_domain_ids.is_empty() {
            return Err(RusticolError::integrity(format!(
                "compiled helicity-selector execution {lane_index} has no selector domains"
            )));
        }
        for selector_domain_id in &record.selector_domain_ids {
            let domain = recurrence
                .selector_domains
                .get(*selector_domain_id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "compiled helicity-selector execution {lane_index} references unknown selector domain {selector_domain_id}"
                    ))
                })?;
            if domain.id != *selector_domain_id || !domain.complete {
                return Err(RusticolError::integrity(format!(
                    "compiled helicity-selector execution {lane_index} references incomplete selector domain {selector_domain_id}"
                )));
            }
            if !assigned_domains.insert(*selector_domain_id) {
                return Err(RusticolError::integrity(format!(
                    "compiled helicity-selector executions contain duplicate selector domain {selector_domain_id}"
                )));
            }
        }

        let selector_manifest = record.execution.as_ref();
        if selector_manifest.helicity_sum_execution.is_some()
            || !selector_manifest.color_selector_executions.is_empty()
        {
            return Err(RusticolError::integrity(
                "compiled helicity-selector execution cannot contain a helicity-sum or color-selector auxiliary execution",
            ));
        }
        if !selector_manifest.helicity_selector_executions.is_empty()
            && record.schedule_mode != HelicitySelectorScheduleMode::NestedRuntime
        {
            return Err(RusticolError::integrity(
                "only a nested-runtime helicity-selector execution may contain closure lanes",
            ));
        }
        if selector_manifest.physics_reduction.is_none() {
            return Err(RusticolError::integrity(
                "compiled helicity-selector execution has no reduction override",
            ));
        }
        if selector_manifest.process != manifest.process
            || selector_manifest.key != manifest.key
            || selector_manifest.color_accuracy != manifest.color_accuracy
            || selector_manifest.external_pdg_order != manifest.external_pdg_order
        {
            return Err(RusticolError::integrity(
                "compiled helicity-selector execution does not match its primary process contract",
            ));
        }
        match record.schedule_mode {
            HelicitySelectorScheduleMode::ParentClosure => {
                if selector_manifest.compiled.lc_topology_replay.is_some()
                    || selector_manifest.compiled.color_topology_replay.is_some()
                    || selector_manifest
                        .runtime_schema
                        .amplitude_stage
                        .color_topology_replay
                        .is_some()
                {
                    return Err(RusticolError::integrity(
                        "parent-closure helicity-selector execution cannot own colour-topology replay",
                    ));
                }
                if selector_manifest
                    .runtime_schema
                    .helicity_recurrence
                    .as_ref()
                    .is_some_and(|lane_recurrence| lane_recurrence.materialization.is_some())
                {
                    return Err(RusticolError::integrity(
                        "parent-closure helicity-selector execution cannot contain helicity materialization",
                    ));
                }
                let primary_layout = &manifest.runtime_schema.parameter_layout;
                let selector_layout = &selector_manifest.runtime_schema.parameter_layout;
                if primary_layout.source_component_parameter_count
                    != selector_layout.source_component_parameter_count
                    || primary_layout.value_component_count != selector_layout.value_component_count
                    || primary_layout.momentum_parameter_count
                        != selector_layout.momentum_parameter_count
                    || primary_layout.model_parameter_count != selector_layout.model_parameter_count
                    || primary_layout.parameter_count_if_flattened
                        != selector_layout.parameter_count_if_flattened
                    || primary_layout.source_components_complex
                        != selector_layout.source_components_complex
                    || primary_layout.momentum_components_real
                        != selector_layout.momentum_components_real
                    || primary_layout.real_valued_inputs != selector_layout.real_valued_inputs
                    || manifest.runtime_schema.current_storage.component_count
                        != selector_manifest
                            .runtime_schema
                            .current_storage
                            .component_count
                    || manifest.runtime_schema.value_storage.component_count
                        != selector_manifest
                            .runtime_schema
                            .value_storage
                            .component_count
                    || manifest.runtime_schema.stages.len()
                        != selector_manifest.runtime_schema.stages.len()
                    || manifest.runtime_schema.amplitude_stage.output_count
                        != selector_manifest
                            .runtime_schema
                            .amplitude_stage
                            .output_count
                {
                    return Err(RusticolError::integrity(
                        "parent-closure helicity-selector execution layout does not match its primary execution",
                    ));
                }
            }
            HelicitySelectorScheduleMode::NestedRuntime => {
                let selector_recurrence = selector_manifest
                    .runtime_schema
                    .helicity_recurrence
                    .as_ref()
                    .filter(|lane_recurrence| lane_recurrence.materialization.is_some())
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "nested-runtime helicity-selector execution requires helicity materialization",
                        )
                    })?;
                let selector_materialization = selector_recurrence
                    .materialization
                    .as_ref()
                    .expect("nested-runtime materialization checked");
                let selector_schedule_domains = selector_materialization
                    .selector_schedules
                    .iter()
                    .map(|schedule| schedule.selector_domain_id)
                    .collect::<BTreeSet<_>>();
                let primary_complete_domains = recurrence
                    .selector_domains
                    .iter()
                    .filter(|domain| domain.complete)
                    .map(|domain| domain.source_states.clone())
                    .collect::<BTreeSet<_>>();
                let nested_complete_domains = selector_recurrence
                    .selector_domains
                    .iter()
                    .filter(|domain| domain.complete)
                    .map(|domain| domain.source_states.clone())
                    .collect::<BTreeSet<_>>();
                if nested_complete_domains != primary_complete_domains {
                    return Err(RusticolError::integrity(format!(
                        "nested-runtime helicity-selector execution {lane_index} disagrees with the primary complete-helicity domain contract"
                    )));
                }
                let primary_structural_zeros = recurrence
                    .structural_zero_selector_domain_ids
                    .iter()
                    .map(|domain_id| {
                        recurrence.selector_domains[*domain_id]
                            .source_states
                            .clone()
                    })
                    .collect::<BTreeSet<_>>();
                let nested_structural_zeros = selector_recurrence
                    .structural_zero_selector_domain_ids
                    .iter()
                    .map(|domain_id| {
                        selector_recurrence.selector_domains[*domain_id]
                            .source_states
                            .clone()
                    })
                    .collect::<BTreeSet<_>>();
                if nested_structural_zeros != primary_structural_zeros {
                    return Err(RusticolError::integrity(format!(
                        "nested-runtime helicity-selector execution {lane_index} disagrees with the primary structural-zero domain contract"
                    )));
                }
                for selector_domain_id in &record.selector_domain_ids {
                    let primary_domain = &recurrence.selector_domains[*selector_domain_id];
                    let nested_domain = selector_recurrence
                        .selector_domains
                        .iter()
                        .find(|domain| {
                            domain.complete && domain.source_states == primary_domain.source_states
                        })
                        .ok_or_else(|| {
                            RusticolError::integrity(format!(
                                "nested-runtime helicity-selector execution {lane_index} omits selector domain {selector_domain_id}"
                            ))
                        })?;
                    if !selector_schedule_domains.contains(&nested_domain.id) {
                        return Err(RusticolError::integrity(format!(
                            "nested-runtime helicity-selector execution {lane_index} has no materialized schedule for selector domain {selector_domain_id}"
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_color_selector_executions(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let mut sector_ids = BTreeSet::new();
    for record in &manifest.color_selector_executions {
        if !sector_ids.insert(record.materialized_sector_id) {
            return Err(RusticolError::integrity(format!(
                "compiled color-selector executions contain duplicate materialized sector {}",
                record.materialized_sector_id
            )));
        }
        let lane = record.execution.as_ref();
        if lane.helicity_sum_execution.is_some()
            || !lane.helicity_selector_executions.is_empty()
            || !lane.color_selector_executions.is_empty()
        {
            return Err(RusticolError::integrity(
                "compiled color-selector execution cannot contain auxiliary executions",
            ));
        }
        if lane.physics_reduction.is_none() {
            return Err(RusticolError::integrity(
                "compiled color-selector execution has no reduction override",
            ));
        }
        if lane.process != manifest.process
            || lane.key != manifest.key
            || lane.color_accuracy != manifest.color_accuracy
            || lane.external_pdg_order != manifest.external_pdg_order
        {
            return Err(RusticolError::integrity(
                "compiled color-selector execution does not match its primary process contract",
            ));
        }
    }
    Ok(())
}

fn ensure_execution_capabilities_supported(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let has_helicity_sum_execution = manifest.helicity_sum_execution.is_some();
    let has_helicity_selector_executions = !manifest.helicity_selector_executions.is_empty();
    let has_color_selector_executions = !manifest.color_selector_executions.is_empty();
    if let Some(sum_manifest) = manifest.helicity_sum_execution.as_deref() {
        ensure_execution_capabilities_supported(sum_manifest)?;
    }
    for record in &manifest.helicity_selector_executions {
        ensure_execution_capabilities_supported(record.execution.as_ref())?;
    }
    for record in &manifest.color_selector_executions {
        ensure_execution_capabilities_supported(record.execution.as_ref())?;
    }
    let mut execution_capabilities = BTreeSet::new();
    if let Some(parameters) = &manifest.compiled.model_parameter_evaluator {
        ensure_evaluator_capabilities_supported(&parameters.evaluator)?;
        let actual = evaluator_runtime_capabilities(&parameters.evaluator)?;
        validate_declared_capabilities(
            &parameters.required_runtime_capabilities,
            &actual,
            "model-parameter evaluator",
        )?;
        execution_capabilities.extend(actual);
    }
    if let Some(stages) = &manifest.compiled.stage_evaluators {
        let mut stage_capabilities = BTreeSet::new();
        for stage in &stages.stages {
            ensure_evaluator_capabilities_supported(&stage.evaluator)?;
            stage_capabilities.extend(evaluator_runtime_capabilities(&stage.evaluator)?);
        }
        ensure_evaluator_capabilities_supported(&stages.amplitude_stage.evaluator)?;
        stage_capabilities.extend(evaluator_runtime_capabilities(
            &stages.amplitude_stage.evaluator,
        )?);
        let has_compiled_selector_metadata = manifest
            .runtime_schema
            .helicity_recurrence
            .as_ref()
            .is_some_and(|recurrence| recurrence.materialization.is_some())
            || stages
                .stages
                .iter()
                .chain(std::iter::once(&stages.amplitude_stage))
                .flat_map(|stage| &stage.output_slots)
                .any(|slot| !slot.color_selector_domain_ids.is_empty());
        if stages
            .required_runtime_capabilities
            .iter()
            .any(|capability| capability == COMPILED_RUNTIME_SELECTORS_CAPABILITY)
        {
            if !has_compiled_selector_metadata {
                return Err(RusticolError::integrity(
                    "compiled runtime-selector capability has no selector metadata",
                ));
            }
            stage_capabilities.insert(COMPILED_RUNTIME_SELECTORS_CAPABILITY.to_string());
        }
        if stages
            .required_runtime_capabilities
            .iter()
            .any(|capability| capability == COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY)
        {
            let all_direct = stages
                .stages
                .iter()
                .chain(std::iter::once(&stages.amplitude_stage))
                .all(|stage| stage.compiled_plane_arena.is_some());
            if !all_direct {
                return Err(RusticolError::integrity(
                    "compiled-plane-arena-v1 capability has incomplete stage metadata",
                ));
            }
            stage_capabilities.insert(COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY.to_string());
        }
        validate_declared_capabilities(
            &stages.required_runtime_capabilities,
            &stage_capabilities,
            "stage evaluator set",
        )?;
        execution_capabilities.extend(stage_capabilities);
    }
    if has_helicity_sum_execution {
        execution_capabilities.extend(
            manifest
                .helicity_sum_execution
                .as_deref()
                .into_iter()
                .flat_map(|sum_manifest| {
                    sum_manifest.required_runtime_capabilities.iter().cloned()
                }),
        );
        execution_capabilities.insert(COMPILED_HELICITY_DUAL_LANE_CAPABILITY.to_string());
    }
    if !has_helicity_sum_execution
        && manifest
            .runtime_schema
            .helicity_recurrence
            .as_ref()
            .is_some_and(|recurrence| recurrence.materialization.is_some())
    {
        execution_capabilities.insert(COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY.to_string());
    }
    if has_helicity_selector_executions {
        execution_capabilities.extend(manifest.helicity_selector_executions.iter().flat_map(
            |record| {
                record
                    .execution
                    .required_runtime_capabilities
                    .iter()
                    .cloned()
            },
        ));
        execution_capabilities.insert(COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY.to_string());
        execution_capabilities.insert(COMPILED_RUNTIME_SELECTORS_CAPABILITY.to_string());
    }
    if has_color_selector_executions {
        execution_capabilities.extend(manifest.color_selector_executions.iter().flat_map(
            |record| {
                record
                    .execution
                    .required_runtime_capabilities
                    .iter()
                    .cloned()
            },
        ));
        execution_capabilities.insert(COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY.to_string());
        execution_capabilities.insert(COMPILED_RUNTIME_SELECTORS_CAPABILITY.to_string());
    }
    if let Some(factorized_block) = manifest
        .runtime_schema
        .amplitude_stage
        .color_contraction
        .as_ref()
        .and_then(|contraction| contraction.repeated_block.as_ref())
        .and_then(|repeated| repeated.factorized_block.as_ref())
    {
        execution_capabilities.insert(
            match factorized_block {
                GenericFactorizedColorContractionBlockManifest::KleinFourWalsh { .. } => {
                    COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY
                }
                GenericFactorizedColorContractionBlockManifest::ElementaryAbelianWalsh {
                    ..
                } => COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY,
            }
            .to_string(),
        );
    }
    if manifest.color_contraction_payload.is_some() {
        execution_capabilities.insert(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY.to_string());
    }
    validate_declared_execution_capabilities(
        &manifest.required_runtime_capabilities,
        &execution_capabilities,
    )?;
    Ok(())
}

fn validate_declared_execution_capabilities(
    declared: &[String],
    actual: &BTreeSet<String>,
) -> RusticolResult<()> {
    let declared_set = declared.iter().cloned().collect::<BTreeSet<_>>();
    if declared_set.len() == declared.len() && &declared_set == actual {
        return Ok(());
    }

    // Artifacts written before this capability was made explicit already carry
    // the complete primary-recurrence materialization. Accept only that exact
    // legacy omission; every other declaration mismatch remains an integrity
    // error.
    let mut legacy_actual = actual.clone();
    if declared_set.len() == declared.len()
        && legacy_actual.remove(COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY)
        && declared_set == legacy_actual
    {
        return Ok(());
    }

    Err(RusticolError::integrity(format!(
        "execution manifest capabilities {declared:?} do not match evaluator capabilities {actual:?}"
    )))
}

#[cfg(all(test, feature = "f64-symjit"))]
mod compiled_plane_arena_contract_tests {
    use super::*;
    use serde_json::{Value, json};

    fn source_binding(parameter_index: usize) -> Value {
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
    }

    fn arena_manifest_at(optimization_level: u8) -> ExecutionManifest {
        let mut value = crate::artifact::tests::minimal_execution_manifest(
            "p0",
            "a b > c",
            SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            crate::artifact::tests::direct_evaluator_manifest("evaluators/direct.symjit"),
        );
        value["compiled"]["momentum_slot_ids"] = json!([0, 1, 2]);
        let stage = &mut value["compiled"]["stage_evaluators"]["amplitude_stage"];
        stage["parameter_layout"] = json!("stage-local-value-momentum");
        stage["input_components"] = Value::Array((0..14).map(source_binding).collect());
        stage["evaluator"]["optimization_level"] = json!(optimization_level);
        stage["evaluator"]["plane_application"] = json!({
            "application_path": "evaluators/direct.plane.symjit",
            "application_abi": SYMJIT_PLANE_APPLICATION_ABI,
            "storage_abi": SYMJIT_APPLICATION_STORAGE_ABI,
            "element_layout": "split-complex-plane-major",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "input_complex_count": 14,
            "output_complex_count": 1,
            "input_plane_count": 28,
            "output_plane_count": 2,
            "compiler_type": "native",
            "translation_mode": "symbolica-structured-instructions",
            "optimization_level": optimization_level,
            "simd": true,
            "complex": true,
            "fast_math": true,
            "fast_complex": false,
            "compression": true,
            "threading": false,
            "direct_arena": true,
            "source_digest": "00".repeat(32),
            "target": {
                "word_bits": 64,
                "endianness": "little",
                "triple": "test-native",
                "cpu_features": [],
            },
        });
        stage["compiled_plane_arena"] = json!({
            "schema_version": 1,
            "kind": "compiled-plane-arena-stage",
            "application_abi": COMPILED_PLANE_DIRECT_APPLICATION_ABI,
            "source_application_abi": COMPILED_PLANE_SOURCE_APPLICATION_ABI,
            "element_layout": "split-complex-component-major",
            "output_operation": "overwrite",
            "output_factor": "identity",
            "input_output_aliasing": "forbidden",
            "output_output_aliasing": "forbidden",
            "input_bindings": (0..14).map(source_binding).collect::<Vec<_>>(),
            "output_bindings": [{
                "output_index": 0,
                "arena": "amplitude",
                "component": 0,
            }],
            "leaves": [{
                "application_path": "evaluators/direct.plane.symjit",
                "source_application_abi": COMPILED_PLANE_SOURCE_APPLICATION_ABI,
                "optimization_level": optimization_level,
                "direct_codegen_optimization_level": optimization_level,
                "input_len": 14,
                "output_len": 1,
                "input_indices": (0..14).collect::<Vec<_>>(),
                "output_start": 0,
                "output_stop": 1,
            }],
        });
        value["compiled"]["stage_evaluators"]["parameter_layout"] =
            json!("stage-local-value-momentum");
        value["compiled"]["stage_evaluators"]["required_runtime_capabilities"] = json!([
            COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
        ]);
        value["required_runtime_capabilities"] = json!([
            COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
        ]);
        serde_json::from_value(value).expect("deserialize compiled plane-arena fixture")
    }

    fn arena_manifest() -> ExecutionManifest {
        arena_manifest_at(3)
    }

    #[test]
    fn compiled_momentum_slot_selection_is_required_and_covers_final_inputs() {
        let mut manifest = arena_manifest();
        assert!(validate_compiled_plane_arena_contract(&manifest).unwrap());
        manifest.compiled.momentum_slot_ids = None;
        assert!(
            validate_compiled_plane_arena_contract(&manifest)
                .unwrap_err()
                .message()
                .contains("regenerate")
        );
        for ids in [
            vec![0, 1],
            vec![0, 0, 1, 2],
            vec![2, 1, 0],
            vec![0, 1, 2, 3],
        ] {
            manifest.compiled.momentum_slot_ids = Some(ids);
            assert!(validate_compiled_plane_arena_contract(&manifest).is_err());
        }
    }

    #[test]
    fn pre_arena_compiled_f64_requires_actionable_regeneration() {
        let value = crate::artifact::tests::minimal_execution_manifest(
            "p0",
            "a b > c",
            SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            crate::artifact::tests::direct_evaluator_manifest("evaluators/direct.symjit"),
        );
        let manifest =
            serde_json::from_value(value).expect("deserialize pre-arena compiled fixture");

        let error = validate_compiled_plane_arena_contract(&manifest)
            .expect_err("pre-arena compiled f64 must fail closed");

        let message = error.to_string();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(message.contains("compiled-plane-arena-v1"));
        assert!(message.contains("`pyamplicol generate`"));
    }

    #[test]
    fn pre_arena_non_o3_compiled_f64_fails_closed() {
        let mut value = crate::artifact::tests::minimal_execution_manifest(
            "p0",
            "a b > c",
            SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            crate::artifact::tests::direct_evaluator_manifest("evaluators/direct.symjit"),
        );
        value["compiled"]["stage_evaluators"]["amplitude_stage"]["evaluator"]["optimization_level"] =
            json!(1);
        let manifest =
            serde_json::from_value(value).expect("deserialize row-major compiled fixture");

        let error = validate_compiled_plane_arena_contract(&manifest)
            .expect_err("pre-arena non-O3 compiled f64 must fail closed");
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.to_string().contains("compiled-plane-arena-v1"));
    }

    #[test]
    fn compiled_plane_contract_rejects_predecessor_binding_abi() {
        let mut manifest = arena_manifest();
        manifest
            .compiled
            .stage_evaluators
            .as_mut()
            .unwrap()
            .amplitude_stage
            .compiled_plane_arena
            .as_mut()
            .unwrap()
            .application_abi = concat!("symjit-direct-", "application-storage-v1").to_string();

        let error = validate_compiled_plane_arena_contract(&manifest)
            .expect_err("the predecessor compiled-plane ABI must fail closed");

        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.to_string().contains("incompatible direct ABI"));
        assert!(error.to_string().contains("regenerate"));
    }

    #[test]
    fn complete_plane_contract_supports_jit_o0_through_o3() {
        for optimization_level in 0..=3 {
            let manifest = arena_manifest_at(optimization_level);

            assert!(validate_compiled_plane_arena_contract(&manifest).unwrap());
            ensure_execution_capabilities_supported(&manifest).unwrap();
        }
        assert!(
            supported_runtime_capabilities().contains(&COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY)
        );
    }

    #[test]
    fn plane_contract_rejects_binding_and_capability_drift() {
        let mut manifest = arena_manifest();
        manifest
            .compiled
            .stage_evaluators
            .as_mut()
            .unwrap()
            .amplitude_stage
            .compiled_plane_arena
            .as_mut()
            .unwrap()
            .input_bindings[2]
            .global_component = 99;
        let error = validate_compiled_plane_arena_contract(&manifest).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
        assert!(error.to_string().contains("input bindings disagree"));

        let mut manifest = arena_manifest();
        manifest
            .compiled
            .stage_evaluators
            .as_mut()
            .unwrap()
            .required_runtime_capabilities
            .retain(|value| value != COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY);
        let error = validate_compiled_plane_arena_contract(&manifest).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    }

    #[test]
    fn plane_contract_authenticates_leaf_optimization_level() {
        let mut manifest = arena_manifest_at(1);
        let stage = &mut manifest
            .compiled
            .stage_evaluators
            .as_mut()
            .unwrap()
            .amplitude_stage;
        stage.compiled_plane_arena.as_mut().unwrap().leaves[0].optimization_level = 2;

        let error = validate_compiled_plane_arena_contract(&manifest).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
        assert!(error.to_string().contains("optimization level disagrees"));
    }

    #[test]
    fn plane_contract_rejects_unknown_jit_optimization_level() {
        let mut manifest = arena_manifest();
        let stage = &mut manifest
            .compiled
            .stage_evaluators
            .as_mut()
            .unwrap()
            .amplitude_stage;
        let EvaluatorManifest::SymjitApplication {
            optimization_level,
            plane_application,
            ..
        } = &mut stage.evaluator
        else {
            panic!("compiled plane fixture must use one SymJIT evaluator");
        };
        *optimization_level = 4;
        plane_application.as_mut().unwrap().optimization_level = 4;
        let leaf = &mut stage.compiled_plane_arena.as_mut().unwrap().leaves[0];
        leaf.optimization_level = 4;
        leaf.direct_codegen_optimization_level = 4;

        let error = validate_compiled_plane_arena_contract(&manifest).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.to_string().contains("levels 0 through 3"));
    }
}

#[cfg(test)]
mod legacy_primary_recurrence_capability_tests {
    use super::*;

    fn capabilities(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn accepts_only_the_legacy_primary_recurrence_omission() {
        let direct = "symjit.application.complex-f64.v1";
        let declared = vec![direct.to_string()];
        let actual = capabilities(&[COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY, direct]);

        validate_declared_execution_capabilities(&declared, &actual)
            .expect("the legacy omission remains compatible");
    }

    #[test]
    fn rejects_every_other_capability_mismatch() {
        let direct = "symjit.application.complex-f64.v1";
        let declared = vec![direct.to_string()];
        let actual = capabilities(&[COMPILED_RUNTIME_SELECTORS_CAPABILITY, direct]);

        let error = validate_declared_execution_capabilities(&declared, &actual)
            .expect_err("unrelated missing capabilities remain invalid");
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    }
}

#[cfg(all(test, feature = "f64-symjit"))]
mod walsh_color_contraction_capability_tests {
    use super::*;
    use serde_json::json;

    const DIRECT: &str = "symjit.application.complex-f64.v1";

    fn manifest(has_factorized_block: bool, declares_capability: bool) -> ExecutionManifest {
        let mut value = crate::artifact::tests::minimal_execution_manifest(
            "p0",
            "a b > c",
            DIRECT,
            crate::artifact::tests::direct_evaluator_manifest("evaluators/direct.symjit"),
        );
        if has_factorized_block {
            value["runtime_schema"]["amplitude_stage"]["color_contraction"] = json!({
                "supported": true,
                "reason": null,
                "group_count": 8,
                "includes_color_factor": true,
                "entries": [],
                "repeated_block": {
                    "component_count": 2,
                    "component_group_ids": [0, 1, 2, 3, 4, 5, 6, 7],
                    "entries": [],
                    "factorized_block": {
                        "kind": "klein-four-walsh",
                        "cosets": [[0, 1, 2, 3]],
                    },
                },
            });
        }
        if declares_capability {
            value["required_runtime_capabilities"] =
                json!([COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY, DIRECT]);
        }
        serde_json::from_value(value).expect("deserialize Walsh capability fixture")
    }

    fn c2k_manifest(has_factorized_block: bool, declares_capability: bool) -> ExecutionManifest {
        let mut value = crate::artifact::tests::minimal_execution_manifest(
            "p0",
            "a b > c",
            DIRECT,
            crate::artifact::tests::direct_evaluator_manifest("evaluators/direct.symjit"),
        );
        if has_factorized_block {
            value["runtime_schema"]["amplitude_stage"]["color_contraction"] = json!({
                "supported": true,
                "reason": null,
                "group_count": 16,
                "includes_color_factor": true,
                "entries": [],
                "repeated_block": {
                    "component_count": 2,
                    "component_group_ids": [
                        0, 1, 2, 3, 4, 5, 6, 7,
                        8, 9, 10, 11, 12, 13, 14, 15,
                    ],
                    "entries": [],
                    "factorized_block": {
                        "kind": "elementary-abelian-walsh",
                        "rank": 3,
                        "cosets": [[0, 1, 2, 3, 4, 5, 6, 7]],
                    },
                },
            });
        }
        if declares_capability {
            value["required_runtime_capabilities"] =
                json!([COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY, DIRECT]);
        }
        serde_json::from_value(value).expect("deserialize C2^k Walsh capability fixture")
    }

    #[test]
    fn supported_capabilities_include_walsh_color_contraction() {
        assert!(
            supported_runtime_capabilities().contains(&COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY)
        );
        assert!(
            supported_runtime_capabilities()
                .contains(&COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY)
        );
    }

    #[test]
    fn walsh_color_contraction_requires_an_exact_capability_declaration() {
        ensure_execution_capabilities_supported(&manifest(false, false))
            .expect("legacy execution without Walsh metadata");
        ensure_execution_capabilities_supported(&manifest(true, true))
            .expect("Walsh execution with matching capability");

        for manifest in [manifest(true, false), manifest(false, true)] {
            let error = ensure_execution_capabilities_supported(&manifest)
                .expect_err("Walsh capability drift must fail closed");
            assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
            assert!(
                error
                    .to_string()
                    .contains(COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY),
                "{error}"
            );
        }
    }

    #[test]
    fn c2k_walsh_color_contraction_requires_its_distinct_exact_capability() {
        ensure_execution_capabilities_supported(&c2k_manifest(true, true))
            .expect("C2^k Walsh execution with matching capability");

        for manifest in [c2k_manifest(true, false), c2k_manifest(false, true), {
            let mut manifest = c2k_manifest(true, true);
            manifest.required_runtime_capabilities[0] =
                COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY.to_string();
            manifest
        }] {
            let error = ensure_execution_capabilities_supported(&manifest)
                .expect_err("C2^k Walsh capability drift must fail closed");
            assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
        }
    }
}

fn validate_declared_capabilities(
    declared: &[String],
    actual: &BTreeSet<String>,
    context: &str,
) -> RusticolResult<()> {
    let declared_set = declared.iter().cloned().collect::<BTreeSet<_>>();
    if declared_set.len() != declared.len() || &declared_set != actual {
        return Err(RusticolError::integrity(format!(
            "{context} capabilities {declared:?} do not match evaluator capabilities {actual:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod lc_topology_replay_contract_tests {
    use super::*;

    fn proof() -> LcTopologyReplayProofManifest {
        LcTopologyReplayProofManifest {
            status: "proven".to_string(),
            algorithm: Some("canonical-contract-test-v1".to_string()),
            digest: Some("a".repeat(64)),
        }
    }

    fn labels(values: &[(usize, usize)]) -> Vec<LcTopologyReplayLabelPermutationManifest> {
        values
            .iter()
            .map(
                |(representative_label, sector_label)| LcTopologyReplayLabelPermutationManifest {
                    representative_label: *representative_label,
                    sector_label: *sector_label,
                },
            )
            .collect()
    }

    fn permutation(
        sector_id: i64,
        weight: f64,
        sign: i8,
        mapping: &[(usize, usize)],
    ) -> LcTopologyReplaySectorPermutationManifest {
        LcTopologyReplaySectorPermutationManifest {
            sector_id,
            weight,
            sign,
            factor: Some(vec![weight * f64::from(sign), 0.0]),
            label_permutation: labels(mapping),
        }
    }

    fn group(
        representative_sector_id: i64,
        active_sector_ids: &[i64],
        target_sector_id: i64,
        sign: i8,
    ) -> LcTopologyReplayGroupManifest {
        LcTopologyReplayGroupManifest {
            representative_sector_id,
            materialized_sector_id: representative_sector_id,
            active_sector_ids: active_sector_ids.to_vec(),
            proof: Some(proof()),
            sector_permutations: vec![
                permutation(representative_sector_id, 1.0, 1, &[(1, 1), (2, 2)]),
                permutation(target_sector_id, 1.0, sign, &[(1, 2), (2, 1)]),
            ],
        }
    }

    fn v2_replay() -> LcTopologyReplayManifest {
        LcTopologyReplayManifest {
            enabled: true,
            mode: "external-label-permutation".to_string(),
            contract_version: Some(2),
            physical_sector_count: Some(5),
            replayed_sector_count: 4,
            materialized_sector_ids: vec![0, 2, 4],
            residual_sector_ids: vec![4],
            groups: vec![group(0, &[0, 1], 1, -1), group(2, &[2, 3], 3, 1)],
        }
    }

    #[test]
    fn v2_deduplicates_mappings_and_keeps_groups_and_residuals_additive() {
        let data = build_lc_topology_replay_data(Some(&v2_replay())).unwrap();

        assert_eq!(data.mappings, vec![vec![], vec![(0, 1), (1, 0)]]);
        assert_eq!(data.materialized_sector_ids, BTreeSet::from([0, 2, 4]));
        assert_eq!(
            data.routes[0]
                .iter()
                .map(|route| (route.physical_sector_id, route.materialized_sector_id))
                .collect::<Vec<_>>(),
            vec![(0, 0), (2, 2), (4, 4)]
        );
        assert_eq!(
            data.routes[1]
                .iter()
                .map(|route| (route.physical_sector_id, route.materialized_sector_id))
                .collect::<Vec<_>>(),
            vec![(1, 0), (3, 2)]
        );
        assert_eq!(data.routes[1][0].sign, -1);
        assert_eq!(data.routes[1][0].amplitude_factor, [-1.0, 0.0]);
        assert_eq!(data.routes[1][0].squared_reduction_weight(), 1.0);
    }

    #[test]
    fn v2_rejects_malformed_physical_and_materialized_coverage() {
        let mut replay = v2_replay();
        replay.materialized_sector_ids = vec![0, 2];
        let error = build_lc_topology_replay_data(Some(&replay)).unwrap_err();
        assert!(error.to_string().contains("materialized coverage"));

        let mut replay = v2_replay();
        replay.physical_sector_count = Some(6);
        let error = build_lc_topology_replay_data(Some(&replay)).unwrap_err();
        assert!(error.to_string().contains("every physical sector"));
    }

    #[test]
    fn v2_rejects_unproven_groups_and_inconsistent_signed_factors() {
        let mut replay = v2_replay();
        replay.groups[0].proof.as_mut().unwrap().status = "assumed".to_string();
        let error = build_lc_topology_replay_data(Some(&replay)).unwrap_err();
        assert!(error.to_string().contains("requires a proven"));

        let mut replay = v2_replay();
        replay.groups[0].sector_permutations[1].factor = Some(vec![1.0, 0.0]);
        let error = build_lc_topology_replay_data(Some(&replay)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match its weight and sign")
        );
    }

    #[test]
    fn legacy_contract_remains_loadable() {
        let mut replay = v2_replay();
        replay.contract_version = None;
        replay.physical_sector_count = None;
        replay.materialized_sector_ids.clear();
        replay.residual_sector_ids.clear();
        replay.replayed_sector_count = 4;
        for group in &mut replay.groups {
            group.proof = None;
            for permutation in &mut group.sector_permutations {
                permutation.factor = None;
            }
        }

        let data = build_lc_topology_replay_data(Some(&replay)).unwrap();
        assert_eq!(data.materialized_sector_ids, BTreeSet::from([0, 2]));
        assert_eq!(data.routes.iter().map(Vec::len).sum::<usize>(), 4);
    }
}

#[cfg(test)]
mod helicity_recurrence_contract_tests {
    use super::*;
    use serde_json::json;

    #[cfg(feature = "f64-symjit")]
    #[test]
    fn compiled_parent_detach_preserves_selected_materialized_stage_schedule() {
        const DIRECT: &str = "symjit.application.complex-f64.v1";
        let mut value = crate::artifact::tests::minimal_execution_manifest(
            "p0",
            "a > a",
            DIRECT,
            crate::artifact::tests::direct_evaluator_manifest("evaluators/direct.symjit"),
        );
        value["external_pdg_order"] = json!([1]);
        value["runtime_schema"]["external_particles"]
            .as_array_mut()
            .expect("external-particle fixture")
            .truncate(1);
        value["runtime_schema"]["current_storage"]["current_slots"][1]["is_source"] = json!(false);
        value["runtime_schema"]["source_fill"]["source_count"] = json!(1);
        value["runtime_schema"]["source_fill"]["sources"]
            .as_array_mut()
            .expect("source fixture")
            .truncate(1);
        let source = &mut value["runtime_schema"]["source_fill"]["sources"][0];
        source["source_helicity"] = json!(-1);
        source["chirality"] = json!(-1);
        source["spin_state"] = json!(-1);
        source["source_ir"]["states"] = json!([
            {"helicity": -1, "chirality": -1, "spin_state": -1},
            {"helicity": 1, "chirality": 1, "spin_state": 1},
        ]);
        let root = &mut value["runtime_schema"]["amplitude_stage"]["roots"][0];
        root["left_current_id"] = json!(1);
        root["right_current_id"] = json!(0);

        let evaluators = &mut value["compiled"]["stage_evaluators"];
        evaluators["parameter_layout"] = json!("stage-local-value-momentum");
        evaluators["parameter_count"] = json!(0);
        evaluators["value_parameter_count"] = json!(0);
        evaluators["momentum_parameter_count"] = json!(0);
        evaluators["real_valued_inputs"] = json!([]);
        evaluators["stage_count"] = json!(2);
        let mut current_stage = evaluators["amplitude_stage"].clone();
        current_stage["stage_index"] = json!(1);
        current_stage["stage_kind"] = json!("current-combine");
        current_stage["subset_size"] = json!(1);
        current_stage["evaluator_label"] = json!("materialized_current");
        current_stage["parameter_layout"] = json!("stage-local-value-momentum");
        current_stage["output_slots"] = json!([{
            "value_slot_id": 1,
            "current_id": 1,
            "variant": "unpropagated",
            "component_start": 0,
            "component_stop": 1,
            "output_start": 0,
            "output_stop": 1,
        }]);
        current_stage["input_value_slot_ids"] = json!([0]);
        current_stage["output_value_slot_ids"] = json!([1]);
        current_stage["input_components"] = json!([{
            "kind": "value",
            "source_id": 0,
            "component": 0,
            "global_component": 0,
            "parameter_index": 0,
            "real_valued": false,
        }]);
        current_stage["parameter_count"] = json!(1);
        current_stage["value_parameter_count"] = json!(1);
        current_stage["momentum_parameter_count"] = json!(0);
        current_stage["real_valued_inputs"] = json!([]);
        current_stage["evaluator"]["input_len"] = json!(1);
        evaluators["stages"] = json!([current_stage]);

        let amplitude = &mut evaluators["amplitude_stage"];
        amplitude["parameter_layout"] = json!("stage-local-value-momentum");
        amplitude["input_components"] = json!([
            {
                "kind": "value", "source_id": 0, "component": 0,
                "global_component": 0, "parameter_index": 0,
                "real_valued": false,
            },
            {
                "kind": "value", "source_id": 1, "component": 0,
                "global_component": 1, "parameter_index": 1,
                "real_valued": false,
            },
        ]);
        amplitude["parameter_count"] = json!(2);
        amplitude["value_parameter_count"] = json!(2);
        amplitude["momentum_parameter_count"] = json!(0);
        amplitude["real_valued_inputs"] = json!([]);
        amplitude["evaluator"]["input_len"] = json!(2);

        let mut manifest = serde_json::from_value::<ExecutionManifest>(value)
            .expect("deserialize compiled-parent fixture");
        let (recurrence, _) = valid_materialized_contract();
        manifest.runtime_schema.helicity_recurrence = Some(recurrence);

        let (stage_evaluators, helicity_recurrence) =
            detach_compiled_helicity_load_inputs(&mut manifest).unwrap();

        assert!(stage_evaluators.is_some());
        assert!(manifest.compiled.stage_evaluators.is_none());
        assert!(manifest.runtime_schema.helicity_recurrence.is_none());
        let materialization = helicity_recurrence
            .as_ref()
            .and_then(|recurrence| recurrence.materialization.as_ref())
            .expect("compiled parent retained its selected-helicity schedule");
        assert_eq!(
            materialization.selector_schedules[0].active_stage_chunk_indices,
            vec![vec![0]]
        );
        assert_eq!(
            materialization.selector_schedules[0].active_amplitude_chunk_indices,
            vec![0]
        );

        let mut runtime = ExecutionRuntime::from_manifest(manifest).unwrap();
        runtime.helicity_recurrence = helicity_recurrence;
        assert!(
            runtime
                .helicity_recurrence
                .as_ref()
                .is_some_and(|recurrence| recurrence.materialization.is_some())
        );
    }

    #[cfg(feature = "f64-symjit")]
    #[test]
    fn helicity_selector_lane_rejects_parent_layout_mismatch() {
        let value = crate::artifact::tests::minimal_helicity_selector_lane_execution();
        let mut manifest = serde_json::from_value::<ExecutionManifest>(value)
            .expect("deserialize selector-lane fixture");
        manifest.helicity_selector_executions[0]
            .execution
            .runtime_schema
            .parameter_layout
            .value_component_count += 1;

        let error = validate_helicity_selector_executions(&manifest).unwrap_err();

        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
        assert!(
            error
                .to_string()
                .contains("layout does not match its primary execution"),
            "{error}"
        );
    }

    #[cfg(feature = "f64-symjit")]
    #[test]
    fn parent_closure_helicity_selector_lane_rejects_topology_replay() {
        let mut value = crate::artifact::tests::minimal_helicity_selector_lane_execution();
        value["helicity_selector_executions"][0]["execution"]["compiled"]["color_topology_replay"] =
            json!({"enabled": true});
        let manifest = serde_json::from_value::<ExecutionManifest>(value)
            .expect("deserialize replay-owning selector-lane fixture");

        let error = validate_helicity_selector_executions(&manifest).unwrap_err();

        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
        assert!(
            error
                .to_string()
                .contains("cannot own colour-topology replay"),
            "{error}"
        );
    }

    #[cfg(feature = "f64-symjit")]
    #[test]
    fn helicity_selector_lane_capabilities_are_aggregated_from_every_record() {
        let value = crate::artifact::tests::minimal_helicity_selector_lane_execution();
        let manifest = serde_json::from_value::<ExecutionManifest>(value)
            .expect("deserialize selector-lane fixture");

        ensure_execution_capabilities_supported(&manifest).unwrap();
    }

    #[cfg(feature = "f64-symjit")]
    #[test]
    fn primary_recurrence_legacy_omission_does_not_mask_other_capability_drift() {
        let mut value = crate::artifact::tests::minimal_helicity_selector_lane_execution();
        value["required_runtime_capabilities"]
            .as_array_mut()
            .expect("capability array")
            .retain(|capability| {
                !matches!(
                    capability.as_str(),
                    Some(COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY)
                        | Some(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
                )
            });
        let manifest = serde_json::from_value::<ExecutionManifest>(value)
            .expect("deserialize selector-lane fixture");

        let error = ensure_execution_capabilities_supported(&manifest).unwrap_err();

        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity, "{error}");
        assert!(
            error
                .to_string()
                .contains(COMPILED_RUNTIME_SELECTORS_CAPABILITY),
            "{error}"
        );
    }

    #[cfg(feature = "f64-symjit")]
    #[test]
    fn nested_runtime_helicity_selector_lane_accepts_its_own_domain_schedule() {
        let mut value = crate::artifact::tests::minimal_helicity_selector_lane_execution();
        value["helicity_selector_executions"][0]["schedule_mode"] = json!("nested-runtime");
        let child = value["helicity_selector_executions"][0]["execution"].clone();
        let mut nested_recurrence = crate::artifact::tests::mock_helicity_materialization();
        nested_recurrence["materialization"]["selector_schedules"] = json!([
            {
                "selector_domain_id": 0,
                "active_current_ids": [0, 1],
                "active_root_ids": [0],
                "structural_zero": false,
            },
            {
                "selector_domain_id": 1,
                "active_current_ids": [0, 1],
                "active_root_ids": [0],
                "structural_zero": false,
            },
        ]);
        value["helicity_selector_executions"][0]["execution"]["runtime_schema"]["helicity_recurrence"] =
            nested_recurrence;
        value["helicity_selector_executions"][0]["execution"]["helicity_selector_executions"] = json!([
            {
                "selector_domain_ids": [0],
                "schedule_mode": "parent-closure",
                "execution": child.clone(),
            },
            {
                "selector_domain_ids": [1],
                "schedule_mode": "parent-closure",
                "execution": child,
            },
        ]);
        let manifest = serde_json::from_value::<ExecutionManifest>(value)
            .expect("deserialize nested-runtime selector-lane fixture");

        validate_helicity_selector_executions(&manifest).unwrap();
        validate_helicity_selector_executions(
            manifest.helicity_selector_executions[0].execution.as_ref(),
        )
        .unwrap();
    }

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn proof(digest: &str) -> HelicityRecurrenceProofManifest {
        HelicityRecurrenceProofManifest {
            status: "proven".to_string(),
            algorithm: HELICITY_RECURRENCE_PROOF_ALGORITHM.to_string(),
            digest: digest.to_string(),
            transition_contract_ids: Vec::new(),
        }
    }

    fn domain(id: usize, helicity: i32) -> HelicitySelectorDomainManifest {
        HelicitySelectorDomainManifest {
            id,
            complete: true,
            source_states: vec![HelicitySelectorSourceStateManifest {
                external_label: 1,
                helicity,
            }],
        }
    }

    fn source_mapping(
        current_id: usize,
        helicity: i32,
        chirality: i32,
        selector_domain_id: usize,
        source_digest: char,
    ) -> HelicitySourceStateMappingManifest {
        HelicitySourceStateMappingManifest {
            current_id,
            external_label: 1,
            helicity,
            chirality,
            spin_state: GenericSourceSpinStateManifest::Scalar(helicity),
            declared_state_index: 0,
            selector_domain_id,
            recurrence_class_id: format!("helicity-current-sha256:{}", digest('a')),
            representative_current_id: 0,
            source_contract_digest: digest(source_digest),
            factor: [1.0, 0.0],
        }
    }

    fn valid_recurrence() -> HelicityRecurrenceManifest {
        let current_digest = digest('a');
        let amplitude_digest = digest('b');
        HelicityRecurrenceManifest {
            kind: HELICITY_RECURRENCE_KIND.to_string(),
            contract_version: HELICITY_RECURRENCE_CONTRACT_VERSION,
            proof_algorithm: HELICITY_RECURRENCE_PROOF_ALGORITHM.to_string(),
            current_count: 2,
            amplitude_root_count: 1,
            proof_counts: HelicityRecurrenceProofCountsManifest {
                recurrence_class_count: 1,
                optimized_recurrence_class_count: 1,
                optimized_current_count: 2,
                residual_current_count: 0,
                amplitude_class_count: 1,
                optimized_amplitude_class_count: 0,
                residual_amplitude_count: 0,
                source_state_mapping_count: 2,
                physical_helicity_count: 2,
                structural_zero_helicity_count: 0,
            },
            selector_domains: vec![domain(0, -1), domain(1, 1)],
            source_state_mappings: vec![
                source_mapping(0, -1, -1, 0, 'c'),
                source_mapping(1, 1, 1, 1, 'd'),
            ],
            recurrence_classes: vec![HelicityRecurrenceClassManifest {
                class_id: format!("helicity-current-sha256:{current_digest}"),
                representative_current_id: 0,
                external_labels: vec![1],
                source_class: true,
                members: vec![
                    HelicityCurrentReplayMemberManifest {
                        current_id: 0,
                        selector_domain_id: 0,
                        factor: [1.0, 0.0],
                    },
                    HelicityCurrentReplayMemberManifest {
                        current_id: 1,
                        selector_domain_id: 1,
                        factor: [1.0, 0.0],
                    },
                ],
                proof: proof(&current_digest),
            }],
            amplitude_classes: vec![HelicityAmplitudeReplayClassManifest {
                class_id: format!("helicity-amplitude-sha256:{amplitude_digest}"),
                representative_root_id: 0,
                members: vec![HelicityAmplitudeReplayMemberManifest {
                    root_id: 0,
                    selector_domain_ids: vec![0, 1],
                    factor: [1.0, 0.0],
                }],
                proof: proof(&amplitude_digest),
            }],
            residual_current_ids: Vec::new(),
            residual_root_ids: Vec::new(),
            structural_zero_selector_domain_ids: Vec::new(),
            diagnostics: Vec::new(),
            materialization: None,
        }
    }

    fn source_schema(helicity: i32, chirality: i32) -> HelicityRecurrenceSchemaSource {
        HelicityRecurrenceSchemaSource {
            external_label: 1,
            helicity,
            chirality,
            spin_state: GenericSourceSpinStateManifest::Scalar(helicity),
            declared_states: vec![GenericSourceStateIrManifest {
                helicity,
                chirality,
                spin_state: GenericSourceSpinStateManifest::Scalar(helicity),
            }],
        }
    }

    fn valid_schema() -> HelicityRecurrenceSchemaView {
        HelicityRecurrenceSchemaView {
            external_count: 1,
            currents: vec![
                HelicityRecurrenceSchemaCurrent {
                    is_source: true,
                    external_labels: vec![1],
                },
                HelicityRecurrenceSchemaCurrent {
                    is_source: true,
                    external_labels: vec![1],
                },
            ],
            sources: BTreeMap::from([(0, source_schema(-1, -1)), (1, source_schema(1, 1))]),
            roots: vec![HelicityRecurrenceSchemaRoot {
                output_index: 0,
                left_current_id: 0,
                right_current_id: 1,
                helicity_weight: 2.0,
            }],
            compiled_chunks: None,
        }
    }

    fn valid_materialized_contract() -> (HelicityRecurrenceManifest, HelicityRecurrenceSchemaView) {
        let mut recurrence = valid_recurrence();
        recurrence.current_count = 4;
        recurrence.proof_counts.recurrence_class_count = 2;
        recurrence.proof_counts.optimized_recurrence_class_count = 2;
        recurrence.proof_counts.optimized_current_count = 4;
        recurrence.source_state_mappings[1].declared_state_index = 1;
        let generated_digest = digest('e');
        recurrence
            .recurrence_classes
            .push(HelicityRecurrenceClassManifest {
                class_id: format!("helicity-current-sha256:{generated_digest}"),
                representative_current_id: 2,
                external_labels: vec![1],
                source_class: false,
                members: vec![
                    HelicityCurrentReplayMemberManifest {
                        current_id: 2,
                        selector_domain_id: 0,
                        factor: [1.0, 0.0],
                    },
                    HelicityCurrentReplayMemberManifest {
                        current_id: 3,
                        selector_domain_id: 1,
                        factor: [1.0, 0.0],
                    },
                ],
                proof: proof(&generated_digest),
            });
        recurrence.materialization = Some(HelicityRecurrenceMaterializationManifest {
            kind: HELICITY_MATERIALIZATION_KIND.to_string(),
            contract_version: HELICITY_MATERIALIZATION_CONTRACT_VERSION,
            strategy: HelicityMaterializationStrategy::Quotient,
            proof_current_count: 4,
            proof_root_count: 1,
            materialized_current_count: 2,
            materialized_root_count: 1,
            proof_to_materialized_current: vec![0, 0, 1, 1],
            source_routes: vec![
                HelicityMaterializedSourceRouteManifest {
                    materialized_current_id: 0,
                    external_label: 1,
                    helicity: -1,
                    chirality: -1,
                    spin_state: GenericSourceSpinStateManifest::Scalar(-1),
                    declared_state_index: 0,
                    selector_domain_id: 0,
                    factor: [1.0, 0.0],
                },
                HelicityMaterializedSourceRouteManifest {
                    materialized_current_id: 0,
                    external_label: 1,
                    helicity: 1,
                    chirality: 1,
                    spin_state: GenericSourceSpinStateManifest::Scalar(1),
                    declared_state_index: 1,
                    selector_domain_id: 1,
                    factor: [1.0, 0.0],
                },
            ],
            amplitude_routes: vec![HelicityMaterializedAmplitudeRouteManifest {
                materialized_root_id: 0,
                selector_domain_ids: vec![0, 1],
                factor: [1.0, 0.0],
                residual: false,
            }],
            selector_schedules: vec![
                HelicityMaterializedSelectorScheduleManifest {
                    selector_domain_id: 0,
                    active_current_ids: vec![0, 1],
                    active_root_ids: vec![0],
                    structural_zero: false,
                },
                HelicityMaterializedSelectorScheduleManifest {
                    selector_domain_id: 1,
                    active_current_ids: vec![0, 1],
                    active_root_ids: vec![0],
                    structural_zero: false,
                },
            ],
        });
        let source = HelicityRecurrenceSchemaSource {
            external_label: 1,
            helicity: -1,
            chirality: -1,
            spin_state: GenericSourceSpinStateManifest::Scalar(-1),
            declared_states: vec![
                GenericSourceStateIrManifest {
                    helicity: -1,
                    chirality: -1,
                    spin_state: GenericSourceSpinStateManifest::Scalar(-1),
                },
                GenericSourceStateIrManifest {
                    helicity: 1,
                    chirality: 1,
                    spin_state: GenericSourceSpinStateManifest::Scalar(1),
                },
            ],
        };
        let schema = HelicityRecurrenceSchemaView {
            external_count: 1,
            currents: vec![
                HelicityRecurrenceSchemaCurrent {
                    is_source: true,
                    external_labels: vec![1],
                },
                HelicityRecurrenceSchemaCurrent {
                    is_source: false,
                    external_labels: vec![1],
                },
            ],
            sources: BTreeMap::from([(0, source)]),
            roots: vec![HelicityRecurrenceSchemaRoot {
                output_index: 0,
                left_current_id: 1,
                right_current_id: 0,
                helicity_weight: 1.0,
            }],
            compiled_chunks: Some(HelicityCompiledChunkCoverage {
                stages: vec![HelicityCompiledStageChunkCoverage {
                    current_ids_by_chunk: vec![BTreeSet::from([1])],
                    input_current_ids_by_chunk: vec![BTreeSet::from([0])],
                }],
                amplitude_root_ids_by_chunk: vec![BTreeSet::from([0])],
                amplitude_input_current_ids_by_chunk: vec![BTreeSet::from([0, 1])],
            }),
        };
        (recurrence, schema)
    }

    fn valid_retained_materialized_contract()
    -> (HelicityRecurrenceManifest, HelicityRecurrenceSchemaView) {
        let (mut recurrence, compact_schema) = valid_materialized_contract();
        let materialization = recurrence
            .materialization
            .as_mut()
            .expect("materialization fixture");
        materialization.strategy = HelicityMaterializationStrategy::RetainedProofGraph;
        materialization.materialized_current_count = 4;
        materialization.materialized_root_count = 1;
        materialization.proof_to_materialized_current = vec![0, 1, 2, 3];
        materialization.source_routes[1].materialized_current_id = 1;
        materialization.amplitude_routes[0].factor = [2.0_f64.sqrt().recip(), 0.0];
        for schedule in &mut materialization.selector_schedules {
            schedule.active_current_ids = vec![0, 2];
        }

        let source = compact_schema.sources[&0].clone();
        let mut flipped_source = source.clone();
        flipped_source.helicity = 1;
        flipped_source.chirality = 1;
        flipped_source.spin_state = GenericSourceSpinStateManifest::Scalar(1);
        let schema = HelicityRecurrenceSchemaView {
            external_count: 1,
            currents: vec![
                HelicityRecurrenceSchemaCurrent {
                    is_source: true,
                    external_labels: vec![1],
                },
                HelicityRecurrenceSchemaCurrent {
                    is_source: true,
                    external_labels: vec![1],
                },
                HelicityRecurrenceSchemaCurrent {
                    is_source: false,
                    external_labels: vec![1],
                },
                HelicityRecurrenceSchemaCurrent {
                    is_source: false,
                    external_labels: vec![1],
                },
            ],
            sources: BTreeMap::from([(0, source), (1, flipped_source)]),
            roots: vec![HelicityRecurrenceSchemaRoot {
                output_index: 0,
                left_current_id: 2,
                right_current_id: 0,
                helicity_weight: 2.0,
            }],
            compiled_chunks: Some(HelicityCompiledChunkCoverage {
                stages: vec![HelicityCompiledStageChunkCoverage {
                    current_ids_by_chunk: vec![BTreeSet::from([2]), BTreeSet::from([3])],
                    input_current_ids_by_chunk: vec![BTreeSet::from([0]), BTreeSet::from([1])],
                }],
                amplitude_root_ids_by_chunk: vec![BTreeSet::from([0])],
                amplitude_input_current_ids_by_chunk: vec![BTreeSet::from([0, 2])],
            }),
        };
        (recurrence, schema)
    }

    fn materialized_load_error(
        recurrence: &HelicityRecurrenceManifest,
        schema: &HelicityRecurrenceSchemaView,
    ) -> String {
        build_helicity_recurrence_runtime_from_view(Some(recurrence), schema)
            .unwrap_err()
            .to_string()
    }

    fn load_error(recurrence: &HelicityRecurrenceManifest) -> String {
        build_helicity_recurrence_runtime_from_view(Some(recurrence), &valid_schema())
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn valid_contract_builds_compact_runtime_and_absence_remains_compatible() {
        let schema = valid_schema();
        assert!(
            build_helicity_recurrence_runtime_from_view(None, &schema)
                .unwrap()
                .is_none()
        );

        let runtime =
            build_helicity_recurrence_runtime_from_view(Some(&valid_recurrence()), &schema)
                .unwrap()
                .unwrap();

        assert_eq!(runtime.selector_domains.len(), 2);
        assert_eq!(runtime.recurrence_classes.len(), 1);
        assert_eq!(runtime.recurrence_classes[0].members.len(), 2);
        assert_eq!(runtime.source_state_mappings.len(), 2);
        assert_eq!(runtime.source_state_mappings[1].recurrence_class_index, 0);
        assert_eq!(runtime.amplitude_classes.len(), 1);
        assert!(runtime.residual_current_ids.is_empty());
        assert!(runtime.residual_root_ids.is_empty());
    }

    #[test]
    fn strict_wire_types_reject_unknown_fields_and_bad_factor_shapes() {
        let error = serde_json::from_value::<HelicityRecurrenceProofManifest>(json!({
            "status": "proven",
            "algorithm": HELICITY_RECURRENCE_PROOF_ALGORITHM,
            "digest": digest('a'),
            "transition_contract_ids": [],
            "unexpected": true,
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));

        let error = serde_json::from_value::<HelicityCurrentReplayMemberManifest>(json!({
            "current_id": 0,
            "selector_domain_id": 0,
            "factor": [1.0],
        }))
        .unwrap_err();
        assert!(error.to_string().contains("length 2"));

        let error = serde_json::from_value::<HelicityMaterializedSelectorScheduleManifest>(json!({
            "selector_domain_id": 0,
            "active_current_ids": [],
            "active_root_ids": [],
            "structural_zero": true,
            "unexpected": true,
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn contract_and_proof_drift_are_rejected() {
        let mut recurrence = valid_recurrence();
        recurrence.contract_version += 1;
        assert!(load_error(&recurrence).contains("contract version"));

        let mut recurrence = valid_recurrence();
        recurrence.recurrence_classes[0].proof.status = "assumed".to_string();
        assert!(load_error(&recurrence).contains("invalid proof metadata"));

        let mut recurrence = valid_recurrence();
        recurrence.amplitude_classes[0].proof.digest = "A".repeat(64);
        assert!(load_error(&recurrence).contains("lowercase SHA-256"));
    }

    #[test]
    fn selector_domains_must_be_contiguous_unique_and_complete() {
        let mut recurrence = valid_recurrence();
        recurrence.selector_domains[1].id = 4;
        assert!(load_error(&recurrence).contains("must be contiguous"));

        let mut recurrence = valid_recurrence();
        recurrence.selector_domains[1].source_states[0].helicity = -1;
        assert!(load_error(&recurrence).contains("repeat a source-state assignment"));

        let mut recurrence = valid_recurrence();
        recurrence.selector_domains[0].complete = false;
        assert!(load_error(&recurrence).contains("inconsistent complete coverage"));
    }

    #[test]
    fn stale_current_ids_and_residual_overlap_are_rejected() {
        let mut recurrence = valid_recurrence();
        recurrence.recurrence_classes[0].members[1].current_id = 2;
        assert!(load_error(&recurrence).contains("stale current id"));

        let mut recurrence = valid_recurrence();
        recurrence.residual_current_ids = vec![0];
        assert!(load_error(&recurrence).contains("overlaps a proven recurrence class"));

        let mut recurrence = valid_recurrence();
        recurrence.recurrence_classes[0].members.pop();
        assert!(load_error(&recurrence).contains("do not cover the proof DAG"));
    }

    #[test]
    fn source_state_mappings_are_checked_against_domains_classes_and_source_ir() {
        let mut recurrence = valid_recurrence();
        recurrence.source_state_mappings[0].declared_state_index = 1;
        assert!(load_error(&recurrence).contains("stale declared state index"));

        let mut recurrence = valid_recurrence();
        recurrence.source_state_mappings[0].recurrence_class_id = "wrong".to_string();
        assert!(load_error(&recurrence).contains("inconsistent with its recurrence class"));

        let mut recurrence = valid_recurrence();
        recurrence.source_state_mappings.pop();
        assert!(load_error(&recurrence).contains("do not exactly cover"));
    }

    #[test]
    fn factors_root_overlap_and_structural_zero_overlap_are_rejected() {
        let mut recurrence = valid_recurrence();
        recurrence.recurrence_classes[0].members[0].factor = [0.0, 0.0];
        assert!(load_error(&recurrence).contains("invalid recurrence factor"));

        let mut recurrence = valid_recurrence();
        recurrence.recurrence_classes[0].members[0].factor = [-1.0, 0.0];
        assert!(load_error(&recurrence).contains("identity factor"));

        let mut recurrence = valid_recurrence();
        recurrence.residual_root_ids = vec![0];
        assert!(load_error(&recurrence).contains("overlaps a proven amplitude class"));

        let mut recurrence = valid_recurrence();
        recurrence.structural_zero_selector_domain_ids = vec![0];
        assert!(load_error(&recurrence).contains("disjoint from amplitude coverage"));
    }

    #[test]
    fn structural_zero_domains_may_reference_unmaterialized_source_states() {
        let mut recurrence = valid_recurrence();
        recurrence.selector_domains.push(domain(2, 0));
        recurrence.structural_zero_selector_domain_ids.push(2);
        recurrence.proof_counts.physical_helicity_count = 3;
        recurrence.proof_counts.structural_zero_helicity_count = 1;

        build_helicity_recurrence_runtime_from_view(Some(&recurrence), &valid_schema())
            .expect("structural-zero domains do not require executable source states");
    }

    #[test]
    fn nonzero_selector_domains_may_replay_from_an_executable_anchor() {
        let mut recurrence = valid_recurrence();
        recurrence.selector_domains.push(domain(2, 0));
        recurrence.amplitude_classes[0].members[0]
            .selector_domain_ids
            .push(2);
        recurrence.proof_counts.physical_helicity_count = 3;
        let mut schema = valid_schema();
        schema.roots[0].helicity_weight = 3.0;

        build_helicity_recurrence_runtime_from_view(Some(&recurrence), &schema)
            .expect("certified replay group contains directly executable domains");
    }

    #[test]
    fn nonzero_replay_groups_require_a_directly_executable_anchor() {
        let mut recurrence = valid_recurrence();
        recurrence.selector_domains.push(domain(2, 0));
        recurrence.amplitude_classes[0].members[0].selector_domain_ids = vec![2];
        recurrence.proof_counts.physical_helicity_count = 3;
        let mut schema = valid_schema();
        schema.roots[0].helicity_weight = 1.0;

        let error = materialized_load_error(&recurrence, &schema);
        assert!(
            error.contains("no directly executable source-state anchor"),
            "{error}"
        );
    }

    #[test]
    fn proof_counts_and_final_runtime_counts_must_match() {
        let mut recurrence = valid_recurrence();
        recurrence.proof_counts.optimized_current_count = 99;
        assert!(load_error(&recurrence).contains("proof counts"));

        let mut recurrence = valid_recurrence();
        recurrence.current_count = 3;
        assert!(load_error(&recurrence).contains("final runtime schema"));
    }

    #[test]
    fn materialized_contract_accepts_larger_proof_space_and_derives_active_chunks() {
        let (recurrence, schema) = valid_materialized_contract();
        let runtime = build_helicity_recurrence_runtime_from_view(Some(&recurrence), &schema)
            .unwrap()
            .unwrap();
        let materialization = runtime.materialization.unwrap();

        assert_eq!(
            materialization.proof_to_materialized_current,
            vec![0, 0, 1, 1]
        );
        assert_eq!(materialization.source_routes.len(), 2);
        assert_eq!(materialization.amplitude_routes.len(), 1);
        assert_eq!(materialization.selector_schedules.len(), 2);
        assert_eq!(
            materialization.selector_schedules[0].active_stage_chunk_indices,
            vec![vec![0]]
        );
        assert_eq!(
            materialization.selector_schedules[0].active_amplitude_chunk_indices,
            vec![0]
        );
    }

    #[test]
    fn retained_materialization_accepts_identity_proof_graph() {
        let (recurrence, schema) = valid_retained_materialized_contract();
        let runtime = build_helicity_recurrence_runtime_from_view(Some(&recurrence), &schema)
            .unwrap()
            .unwrap();
        let materialization = runtime.materialization.unwrap();

        assert_eq!(
            materialization.strategy,
            HelicityMaterializationStrategy::RetainedProofGraph
        );
        assert_eq!(
            materialization.proof_to_materialized_current,
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            materialization.selector_schedules[0].active_stage_chunk_indices,
            vec![vec![0]]
        );
    }

    #[test]
    fn retained_materialization_rejects_nonidentity_current_map() {
        let (mut recurrence, schema) = valid_retained_materialized_contract();
        recurrence
            .materialization
            .as_mut()
            .unwrap()
            .proof_to_materialized_current
            .swap(0, 1);

        assert!(materialized_load_error(&recurrence, &schema).contains("identity current map"));
    }

    #[test]
    fn materialized_contract_rejects_stale_maps_and_overlapping_targets() {
        let (mut recurrence, schema) = valid_materialized_contract();
        recurrence
            .materialization
            .as_mut()
            .unwrap()
            .proof_to_materialized_current
            .pop();
        assert!(materialized_load_error(&recurrence, &schema).contains("incorrect length"));

        let (mut recurrence, schema) = valid_materialized_contract();
        recurrence
            .materialization
            .as_mut()
            .unwrap()
            .proof_to_materialized_current = vec![0, 0, 0, 0];
        assert!(materialized_load_error(&recurrence, &schema).contains("distinct proof classes"));
    }

    #[test]
    fn materialized_routes_and_selector_domains_are_strict() {
        let (mut recurrence, schema) = valid_materialized_contract();
        recurrence.materialization.as_mut().unwrap().source_routes[1].declared_state_index = 0;
        assert!(materialized_load_error(&recurrence, &schema).contains("inconsistent"));

        let (mut recurrence, schema) = valid_materialized_contract();
        recurrence
            .materialization
            .as_mut()
            .unwrap()
            .amplitude_routes[0]
            .selector_domain_ids = vec![1, 0];
        assert!(
            materialized_load_error(&recurrence, &schema)
                .contains("ordered complete selector domains")
        );
    }

    #[test]
    fn materialized_schedules_require_compiled_coverage_for_active_currents() {
        let (recurrence, mut schema) = valid_materialized_contract();
        schema.compiled_chunks.as_mut().unwrap().stages[0].current_ids_by_chunk[0].clear();
        assert!(
            materialized_load_error(&recurrence, &schema)
                .contains("without compiled stage output coverage")
        );

        let (mut recurrence, schema) = valid_materialized_contract();
        recurrence
            .materialization
            .as_mut()
            .unwrap()
            .selector_schedules[0]
            .active_root_ids
            .clear();
        assert!(
            materialized_load_error(&recurrence, &schema)
                .contains("active roots disagree with amplitude routes")
        );
    }

    #[test]
    fn materialized_schedules_reject_cross_domain_chunks_and_missing_ancestors() {
        let (recurrence, mut schema) = valid_retained_materialized_contract();
        schema.compiled_chunks.as_mut().unwrap().stages[0].current_ids_by_chunk[0].insert(3);
        assert!(
            materialized_load_error(&recurrence, &schema)
                .contains("crosses selector domains through currents [3]")
        );

        let (recurrence, mut schema) = valid_materialized_contract();
        schema.compiled_chunks.as_mut().unwrap().stages[0].input_current_ids_by_chunk[0] =
            BTreeSet::from([1]);
        assert!(
            materialized_load_error(&recurrence, &schema)
                .contains("stage 0 evaluator chunk 0 is missing compiled ancestors [1]")
        );

        let (recurrence, mut schema) = valid_retained_materialized_contract();
        schema
            .compiled_chunks
            .as_mut()
            .unwrap()
            .amplitude_input_current_ids_by_chunk[0] = BTreeSet::from([3]);
        assert!(
            materialized_load_error(&recurrence, &schema)
                .contains("amplitude evaluator chunk 0 is missing compiled ancestors [3]")
        );
    }

    #[test]
    fn compiled_color_schedule_rejects_a_missing_transitive_ancestor() {
        let chunks = HelicityCompiledChunkCoverage {
            stages: vec![
                HelicityCompiledStageChunkCoverage {
                    current_ids_by_chunk: vec![BTreeSet::from([1])],
                    input_current_ids_by_chunk: vec![BTreeSet::from([0])],
                },
                HelicityCompiledStageChunkCoverage {
                    current_ids_by_chunk: vec![BTreeSet::from([2])],
                    input_current_ids_by_chunk: vec![BTreeSet::from([1])],
                },
            ],
            amplitude_root_ids_by_chunk: vec![BTreeSet::from([0])],
            amplitude_input_current_ids_by_chunk: vec![BTreeSet::from([2])],
        };

        validate_compiled_selector_schedule_closure(
            "compiled color selector sector 0",
            &[true, false, false],
            &chunks,
            &[vec![0], vec![0]],
            &[0],
            None,
            None,
        )
        .unwrap();

        let error = validate_compiled_selector_schedule_closure(
            "compiled color selector sector 0",
            &[true, false, false],
            &chunks,
            &[Vec::new(), vec![0]],
            &[0],
            None,
            None,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("stage 1 evaluator chunk 0 is missing compiled ancestors [1]")
        );
    }

    #[test]
    fn serialized_chunk_input_maps_resolve_actual_current_dependencies() {
        let leaf = |name: &str| EvaluatorManifest::CompiledComplex {
            runtime_capability: "test.runtime".to_string(),
            function_name: name.to_string(),
            input_len: 1,
            output_len: 1,
            library_path: "test.so".to_string(),
            evaluator_state_path: None,
            number_type: "complex-f64".to_string(),
            native_direct_application: None,
        };
        let stage = GenericSerializedStageEvaluatorManifest {
            stage_index: 1,
            stage_kind: "current-combine".to_string(),
            subset_size: Some(2),
            evaluator_label: "mapped-input-stage".to_string(),
            parameter_layout: "stage-local-value-momentum".to_string(),
            output_length: 2,
            output_slots: Vec::new(),
            input_value_slot_ids: vec![10, 11],
            output_value_slot_ids: Vec::new(),
            interaction_ids: Vec::new(),
            input_components: vec![
                GenericStageInputComponentManifest {
                    kind: "value".to_string(),
                    source_id: 10,
                    component: 0,
                    global_component: 0,
                    parameter_index: 0,
                    real_valued: false,
                },
                GenericStageInputComponentManifest {
                    kind: "value".to_string(),
                    source_id: 11,
                    component: 0,
                    global_component: 1,
                    parameter_index: 1,
                    real_valued: false,
                },
            ],
            parameter_count: 2,
            value_parameter_count: 2,
            momentum_parameter_count: 0,
            model_parameter_count: 0,
            real_valued_inputs: Vec::new(),
            expression_ready: true,
            blockers: Vec::new(),
            evaluator: EvaluatorManifest::Chunked {
                required_runtime_capabilities: Vec::new(),
                input_len: Some(2),
                chunk_input_indices: Some(vec![vec![1], vec![0]]),
                chunks: vec![leaf("right"), leaf("left")],
            },
            compiled_plane_arena: None,
        };

        let dependencies = compiled_input_current_ids_by_chunk(
            &stage,
            &CompiledValueInputLayout {
                current_ids_by_slot: BTreeMap::from([(10, 0), (11, 1)]),
                current_ids_by_global_component: vec![Some(0), Some(1)],
            },
            2,
        )
        .unwrap();

        assert_eq!(dependencies, vec![BTreeSet::from([1]), BTreeSet::from([0])]);

        let mut mismatched = stage;
        mismatched.input_components[0].global_component = 1;
        let error = compiled_input_current_ids_by_chunk(
            &mismatched,
            &CompiledValueInputLayout {
                current_ids_by_slot: BTreeMap::from([(10, 0), (11, 1)]),
                current_ids_by_global_component: vec![Some(0), Some(1)],
            },
            2,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("value-slot and global-component dependencies disagree")
        );
    }

    #[test]
    fn serialized_output_ranges_derive_stage_and_amplitude_chunk_coverage() {
        fn leaf(output_len: usize) -> EvaluatorManifest {
            EvaluatorManifest::CompiledComplex {
                runtime_capability: "test.runtime".to_string(),
                function_name: "test".to_string(),
                input_len: 0,
                output_len,
                library_path: "test.so".to_string(),
                evaluator_state_path: None,
                number_type: "complex-f64".to_string(),
                native_direct_application: None,
            }
        }
        fn chunked(output_lengths: &[usize]) -> EvaluatorManifest {
            EvaluatorManifest::Chunked {
                required_runtime_capabilities: Vec::new(),
                input_len: Some(0),
                chunk_input_indices: Some(vec![Vec::new(); output_lengths.len()]),
                chunks: output_lengths.iter().copied().map(leaf).collect(),
            }
        }
        fn stage(
            stage_index: usize,
            label: &str,
            output_slots: Vec<GenericStageOutputSlotManifest>,
            output_lengths: &[usize],
        ) -> GenericSerializedStageEvaluatorManifest {
            GenericSerializedStageEvaluatorManifest {
                stage_index,
                stage_kind: label.to_string(),
                subset_size: Some(stage_index + 1),
                evaluator_label: label.to_string(),
                parameter_layout: "stage-local-value-momentum".to_string(),
                output_length: output_lengths.iter().sum(),
                output_slots,
                input_value_slot_ids: Vec::new(),
                output_value_slot_ids: Vec::new(),
                interaction_ids: Vec::new(),
                input_components: Vec::new(),
                parameter_count: 0,
                value_parameter_count: 0,
                momentum_parameter_count: 0,
                model_parameter_count: 0,
                real_valued_inputs: Vec::new(),
                expression_ready: true,
                blockers: Vec::new(),
                evaluator: chunked(output_lengths),
                compiled_plane_arena: None,
            }
        }
        fn slot(
            current_id: isize,
            output_start: usize,
            output_stop: usize,
        ) -> GenericStageOutputSlotManifest {
            GenericStageOutputSlotManifest {
                value_slot_id: current_id,
                current_id,
                variant: if current_id < 0 {
                    "amplitude-root".to_string()
                } else {
                    "unpropagated".to_string()
                },
                component_start: output_start,
                component_stop: output_stop,
                output_start,
                output_stop,
                color_selector_domain_ids: Vec::new(),
            }
        }

        let stages = GenericStageEvaluatorArtifactsManifest {
            kind: "generic-dag-stage-evaluator-artifacts".to_string(),
            required_runtime_capabilities: Vec::new(),
            runtime_available: true,
            runtime_unavailable_message: None,
            parameter_count: 0,
            value_parameter_count: 0,
            momentum_parameter_count: 0,
            model_parameter_count: 0,
            real_valued_inputs: Vec::new(),
            parameter_layout: "stage-local-value-momentum".to_string(),
            stage_count: 1,
            stages: vec![stage(
                1,
                "current-combine",
                vec![slot(0, 0, 2), slot(1, 2, 4)],
                &[2, 2],
            )],
            amplitude_stage: stage(
                0,
                "amplitude-roots",
                vec![slot(-1, 0, 1), slot(-1, 1, 2)],
                &[1, 1],
            ),
        };
        let roots = vec![
            HelicityRecurrenceSchemaRoot {
                output_index: 0,
                left_current_id: 0,
                right_current_id: 0,
                helicity_weight: 1.0,
            },
            HelicityRecurrenceSchemaRoot {
                output_index: 1,
                left_current_id: 1,
                right_current_id: 1,
                helicity_weight: 1.0,
            },
        ];
        let coverage = build_helicity_compiled_chunk_coverage_from_parts(
            &stages,
            2,
            &roots,
            &CompiledValueInputLayout {
                current_ids_by_slot: BTreeMap::new(),
                current_ids_by_global_component: Vec::new(),
            },
        )
        .unwrap();

        assert_eq!(
            coverage.stages[0].current_ids_by_chunk,
            vec![BTreeSet::from([0]), BTreeSet::from([1])]
        );
        assert_eq!(
            coverage.amplitude_root_ids_by_chunk,
            vec![BTreeSet::from([0]), BTreeSet::from([1])]
        );

        let mut inactive_chunk = stage(
            1,
            "inactive-chunk",
            vec![slot(0, 0, 2), slot(1, 2, 4)],
            &[2, 2],
        );
        inactive_chunk.output_slots[0].color_selector_domain_ids = vec![0];
        assert_eq!(
            compiled_color_domains_by_chunk(&inactive_chunk, &BTreeSet::from([0_i64])).unwrap(),
            vec![BTreeSet::from([0]), BTreeSet::new()]
        );

        let mut cross_domain = stage(1, "cross-domain", vec![slot(0, 0, 2), slot(1, 2, 4)], &[4]);
        cross_domain.output_slots[0].color_selector_domain_ids = vec![0];
        cross_domain.output_slots[1].color_selector_domain_ids = vec![1];
        let error = compiled_color_domains_by_chunk(&cross_domain, &BTreeSet::from([0_i64, 1_i64]))
            .unwrap_err();
        assert!(error.to_string().contains("crosses selector domains"));
    }
}
