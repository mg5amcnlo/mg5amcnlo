// SPDX-License-Identifier: 0BSD

use super::*;

pub(super) fn load_execution_manifest_with_store(
    manifest: ExecutionManifest,
    payloads: &EvaluatorPayloadStore,
    sizing_physics: &PhysicsRuntime,
) -> RusticolResult<ExecutionRuntime> {
    validate_execution_manifest(&manifest)?;
    ExecutionRuntime::load_from_manifest_with_store(manifest, payloads, sizing_physics)
}

fn validate_execution_manifest(manifest: &ExecutionManifest) -> RusticolResult<()> {
    if manifest.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
        || manifest.kind != "pyamplicol-runtime-execution"
    {
        return Err(RusticolError::compatibility(format!(
            "unsupported execution manifest kind {} schema {}; regenerate the artifact with pyAmpliCol 0.1.0 or newer",
            manifest.kind, manifest.schema_version
        )));
    }
    if manifest.runtime_schema.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
        || manifest.runtime_schema.kind != "pyamplicol-runtime-execution-plan"
    {
        return Err(RusticolError::compatibility(format!(
            "unsupported execution-plan kind {} schema {}; regenerate the artifact with pyAmpliCol 0.1.0 or newer",
            manifest.runtime_schema.kind, manifest.runtime_schema.schema_version
        )));
    }
    if manifest.runtime_schema.process != manifest.process
        || manifest.runtime_schema.process_key != manifest.key
    {
        return Err(RusticolError::artifact(
            "generic runtime schema process identity does not match manifest",
        ));
    }
    if manifest.compiled.kind != "generic-dag-stage-blueprint" {
        return Err(RusticolError::artifact(
            "schema-v3 execution manifests require a generic-dag-stage-blueprint",
        ));
    }
    if !manifest.compiled.runtime_available || manifest.compiled.stage_evaluators.is_none() {
        return Err(RusticolError::artifact(
            "schema-v3 execution manifest has no serialized stage evaluators; regenerate the artifact",
        ));
    }
    if manifest.dag_summary.truncated {
        return Err(RusticolError::artifact(
            "generic DAG schema-v3 artifact was truncated during current construction",
        ));
    }
    if manifest.dag_summary.interaction_evaluation_count > manifest.dag_summary.interaction_count {
        return Err(RusticolError::integrity(
            "generic DAG interaction-evaluation count exceeds its interaction count",
        ));
    }
    let expected_materialization = BTreeMap::from([
        (
            "amplitude_root_count".to_string(),
            manifest.dag_summary.amplitude_root_count,
        ),
        (
            "current_count".to_string(),
            manifest.dag_summary.current_count,
        ),
        (
            "interaction_count".to_string(),
            manifest.dag_summary.interaction_count,
        ),
        (
            "interaction_evaluation_count".to_string(),
            manifest.dag_summary.interaction_evaluation_count,
        ),
        (
            "source_count".to_string(),
            manifest.dag_summary.source_count,
        ),
    ]);
    if manifest.materialization_census.abi != "pyamplicol-fully-resident-materialization-census-v1"
        || manifest.materialization_census.basis != "immutable-fully-resident-compiled-dag"
    {
        return Err(RusticolError::compatibility(
            "compiled materialization census ABI or basis is unsupported",
        ));
    }
    if !manifest.materialization_census.final_equals_peak
        || manifest.materialization_census.r#final != expected_materialization
        || manifest.materialization_census.peak != expected_materialization
    {
        return Err(RusticolError::integrity(
            "compiled materialization census disagrees with the persisted DAG",
        ));
    }
    if manifest.external_pdg_order.len() != manifest.runtime_schema.external_particles.len() {
        return Err(RusticolError::artifact(
            "generic runtime schema external particle count does not match external_pdg_order",
        ));
    }
    for (index, (pdg, particle)) in manifest
        .external_pdg_order
        .iter()
        .zip(&manifest.runtime_schema.external_particles)
        .enumerate()
    {
        if particle.index != index || particle.momentum_slot != index || particle.pdg != *pdg {
            return Err(RusticolError::artifact(format!(
                "generic external particle metadata mismatch at index {index}"
            )));
        }
        if particle.label != index + 1 {
            return Err(RusticolError::artifact(format!(
                "generic external particle labels must be contiguous; got {} at index {index}",
                particle.label
            )));
        }
        if particle.outgoing_pdg == 0 || (particle.role != "initial" && particle.role != "final") {
            return Err(RusticolError::artifact(format!(
                "generic external particle {index} has invalid role or outgoing PDG"
            )));
        }
    }
    validate_generic_parameter_layout(&manifest.runtime_schema)?;
    validate_generic_current_storage(manifest)?;
    validate_generic_value_storage(manifest)?;
    validate_generic_sources(manifest)?;
    validate_generic_momentum_slots(manifest)?;
    validate_generic_stages(manifest)?;
    validate_generic_amplitudes(manifest)?;
    validate_compiled_symmetric_group_diagnostic(manifest)?;
    validate_generic_stage_evaluators(manifest)?;
    validate_lc_topology_replay(manifest)?;
    if let Some(sum_manifest) = manifest.helicity_sum_execution.as_deref() {
        if sum_manifest.helicity_sum_execution.is_some()
            || !sum_manifest.helicity_selector_executions.is_empty()
        {
            return Err(RusticolError::integrity(
                "compiled helicity-sum execution contains a recursively nested auxiliary execution",
            ));
        }
        validate_execution_manifest(sum_manifest)?;
    }
    let mut helicity_selector_domain_ids = BTreeSet::new();
    for record in &manifest.helicity_selector_executions {
        for selector_domain_id in &record.selector_domain_ids {
            if !helicity_selector_domain_ids.insert(*selector_domain_id) {
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
                "compiled helicity-selector execution contains an unsupported auxiliary execution",
            ));
        }
        if !selector_manifest.helicity_selector_executions.is_empty()
            && record.schedule_mode != HelicitySelectorScheduleMode::NestedRuntime
        {
            return Err(RusticolError::integrity(
                "only a nested-runtime helicity-selector execution may contain closure lanes",
            ));
        }
        let has_materialization = selector_manifest
            .runtime_schema
            .helicity_recurrence
            .as_ref()
            .is_some_and(|recurrence| recurrence.materialization.is_some());
        match record.schedule_mode {
            HelicitySelectorScheduleMode::ParentClosure if has_materialization => {
                return Err(RusticolError::integrity(
                    "parent-closure helicity-selector execution cannot contain helicity materialization",
                ));
            }
            HelicitySelectorScheduleMode::NestedRuntime if !has_materialization => {
                return Err(RusticolError::integrity(
                    "nested-runtime helicity-selector execution requires helicity materialization",
                ));
            }
            _ => {}
        }
        validate_execution_manifest(selector_manifest)?;
    }
    let mut color_sector_ids = BTreeSet::new();
    for lane in &manifest.color_selector_executions {
        if !color_sector_ids.insert(lane.materialized_sector_id) {
            return Err(RusticolError::integrity(format!(
                "compiled color-selector executions contain duplicate materialized sector {}",
                lane.materialized_sector_id
            )));
        }
        if lane.execution.helicity_sum_execution.is_some()
            || !lane.execution.helicity_selector_executions.is_empty()
            || !lane.execution.color_selector_executions.is_empty()
        {
            return Err(RusticolError::integrity(
                "compiled color-selector execution cannot contain auxiliary executions",
            ));
        }
        validate_execution_manifest(lane.execution.as_ref())?;
    }
    Ok(())
}

fn validate_compiled_symmetric_group_diagnostic(
    manifest: &ExecutionManifest,
) -> RusticolResult<()> {
    let Some(payload) = manifest.color_contraction_payload.as_ref() else {
        return Ok(());
    };
    if payload.path.is_empty() {
        return Err(RusticolError::compatibility(
            "compiled symmetric-group FFT payload has an empty path",
        ));
    }
    if manifest.color_accuracy != "full"
        || manifest.helicity_sum_execution.is_some()
        || !manifest.helicity_selector_executions.is_empty()
        || !manifest.color_selector_executions.is_empty()
        || manifest.runtime_schema.helicity_recurrence.is_some()
        || manifest.compiled.lc_topology_replay.is_some()
        || manifest.compiled.color_topology_replay.is_some()
        || manifest
            .runtime_schema
            .amplitude_stage
            .color_topology_replay
            .is_some()
    {
        return Err(RusticolError::compatibility(
            "compiled symmetric-group FFT diagnostic supports only selected-helicity FullColour total execution without selector or topology-replay lanes",
        ));
    }
    let contraction = manifest
        .runtime_schema
        .amplitude_stage
        .color_contraction
        .as_ref()
        .ok_or_else(|| {
            RusticolError::integrity(
                "compiled symmetric-group FFT payload has no amplitude contraction metadata",
            )
        })?;
    if !contraction.supported
        || contraction.group_count == 0
        || !contraction.entries.is_empty()
        || contraction.repeated_block.is_some()
    {
        return Err(RusticolError::integrity(
            "compiled symmetric-group FFT diagnostic requires a compact payload-only amplitude contraction",
        ));
    }
    Ok(())
}

#[cfg(all(test, feature = "f64-symjit"))]
mod execution_summary_tests {
    use super::*;

    fn manifest() -> ExecutionManifest {
        serde_json::from_value(crate::artifact::tests::minimal_execution_manifest(
            "p0",
            "a b > c",
            SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            crate::artifact::tests::direct_evaluator_manifest("evaluators/direct.symjit"),
        ))
        .expect("deserialize compiled execution summary fixture")
    }

    #[test]
    fn compiled_summary_and_materialization_census_round_trip() {
        validate_execution_manifest(&manifest()).expect("validate exact compiled census");

        let mut excessive = manifest();
        excessive.dag_summary.interaction_evaluation_count = 1;
        let error = validate_execution_manifest(&excessive).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
        assert!(error.to_string().contains("exceeds"));

        let mut inconsistent = manifest();
        inconsistent
            .materialization_census
            .r#final
            .insert("interaction_evaluation_count".to_string(), 1);
        let error = validate_execution_manifest(&inconsistent).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
        assert!(error.to_string().contains("disagrees"));
    }
}

fn validate_lc_topology_replay(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let Some(replay) = manifest.compiled.lc_topology_replay.as_ref() else {
        return Ok(());
    };
    if !replay.enabled {
        return Ok(());
    }
    let (mappings, _weights) = build_lc_topology_replay_mappings(Some(replay))?;
    if mappings.is_empty() {
        return Err(RusticolError::artifact(
            "enabled LC topology replay contains no sector mappings",
        ));
    }
    let expected_legs = manifest.external_pdg_order.len();
    for mapping in &mappings {
        let mut seen = vec![false; expected_legs];
        for (representative_index, sector_index) in mapping {
            if *representative_index >= expected_legs || *sector_index >= expected_legs {
                return Err(RusticolError::artifact(
                    "LC topology replay mapping references an out-of-range external leg",
                ));
            }
            if seen[*representative_index] {
                return Err(RusticolError::artifact(
                    "LC topology replay mapping contains a duplicate representative label",
                ));
            }
            seen[*representative_index] = true;
        }
    }
    Ok(())
}

fn validate_generic_parameter_layout(schema: &ExecutionPlan) -> RusticolResult<()> {
    let layout = &schema.parameter_layout;
    if !layout.source_components_complex || !layout.momentum_components_real {
        return Err(RusticolError::artifact(
            "generic runtime schema requires complex source components and real momenta",
        ));
    }
    if layout.parameter_count_if_flattened
        != layout.source_component_parameter_count
            + layout.momentum_parameter_count
            + layout.model_parameter_count
    {
        return Err(RusticolError::artifact(
            "generic runtime schema flattened parameter count is inconsistent",
        ));
    }
    if layout.momentum_parameter_count != 4 * schema.momentum_slots.len() {
        return Err(RusticolError::artifact(
            "generic runtime schema momentum parameter count does not match momentum slots",
        ));
    }
    let expected_real_inputs = (layout.source_component_parameter_count
        ..layout.parameter_count_if_flattened)
        .collect::<Vec<_>>();
    if layout.real_valued_inputs != expected_real_inputs {
        return Err(RusticolError::artifact(
            "generic runtime schema real-valued input indices are inconsistent",
        ));
    }
    if schema.model_parameters.len() != layout.model_parameter_count {
        return Err(RusticolError::artifact(
            "generic runtime schema model-parameter count is inconsistent",
        ));
    }
    let mut seen_model_parameters = BTreeSet::new();
    let mut seen_model_parameter_names = BTreeSet::new();
    for parameter in &schema.model_parameters {
        if parameter.name.is_empty()
            || parameter.parameter_index >= layout.model_parameter_count
            || !parameter.default.is_finite()
            || !seen_model_parameters.insert(parameter.parameter_index)
            || !seen_model_parameter_names.insert(parameter.name.clone())
            || parameter
                .runtime_name
                .as_ref()
                .is_some_and(|name| name.is_empty())
            || !matches!(
                (
                    &parameter.runtime_name,
                    parameter.complex_component.as_deref(),
                ),
                (Some(_), Some("real" | "imag")) | (None, None)
            )
        {
            return Err(RusticolError::artifact(
                "generic runtime schema contains invalid model-parameter metadata",
            ));
        }
    }
    Ok(())
}

fn positive_json_integer(value: &Value) -> bool {
    if let Some(unsigned) = value.as_u64() {
        return unsigned > 0;
    }
    if let Some(signed) = value.as_i64() {
        return signed > 0;
    }
    if let Some(float_value) = value.as_f64() {
        return float_value.is_finite() && float_value > 0.0;
    }
    if let Some(text) = value.as_str() {
        if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
            return !hex.is_empty()
                && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
                && hex.bytes().any(|byte| byte != b'0');
        }
        return !text.is_empty()
            && text.bytes().all(|byte| byte.is_ascii_digit())
            && text.bytes().any(|byte| byte != b'0');
    }
    false
}

fn validate_generic_current_storage(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let storage = &manifest.runtime_schema.current_storage;
    if storage.number_type != "complex" {
        return Err(RusticolError::artifact(
            "generic current storage must use complex number type",
        ));
    }
    if storage.current_slots.len() != manifest.dag_summary.current_count {
        return Err(RusticolError::artifact(
            "generic current slot count does not match DAG summary",
        ));
    }
    let mut offset = 0usize;
    for (index, slot) in storage.current_slots.iter().enumerate() {
        if slot.current_id != index {
            return Err(RusticolError::artifact(format!(
                "generic current slot id mismatch at index {index}"
            )));
        }
        if slot.component_start != offset
            || slot.component_stop != slot.component_start + slot.dimension
            || slot.dimension == 0
        {
            return Err(RusticolError::artifact(format!(
                "generic current slot {index} has inconsistent component range"
            )));
        }
        if slot.particle_id == 0 || slot.external_mask == 0 || slot.momentum_mask == 0 {
            return Err(RusticolError::artifact(format!(
                "generic current slot {index} has invalid physics identity"
            )));
        }
        if !valid_current_chirality(slot.chirality) {
            return Err(RusticolError::artifact(format!(
                "generic current slot {index} has invalid quantum-flow metadata"
            )));
        }
        if !storage.metadata_compacted {
            if slot.external_labels.is_empty()
                || !positive_json_integer(&slot.helicity_ancestry)
                || slot.color_state.is_null()
            {
                return Err(RusticolError::artifact(format!(
                    "generic current slot {index} is missing current-index metadata"
                )));
            }
            if slot.flavour_flow.is_empty() {
                return Err(RusticolError::artifact(format!(
                    "generic current slot {index} has invalid quantum-flow metadata"
                )));
            }
            if slot.spin_state.is_null()
                || slot
                    .auxiliary_kind
                    .as_ref()
                    .is_some_and(|kind| kind.is_empty())
            {
                return Err(RusticolError::artifact(format!(
                    "generic current slot {index} has invalid spin/auxiliary metadata"
                )));
            }
        }
        offset = slot.component_stop;
    }
    if storage.component_count != offset {
        return Err(RusticolError::artifact(
            "generic current storage component_count is inconsistent",
        ));
    }
    Ok(())
}

fn validate_generic_value_storage(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let schema = &manifest.runtime_schema;
    let storage = &schema.value_storage;
    if storage.number_type != "complex" {
        return Err(RusticolError::artifact(
            "generic value storage must use complex number type",
        ));
    }
    if schema.parameter_layout.value_component_count != storage.component_count {
        return Err(RusticolError::artifact(
            "generic value component count does not match parameter layout",
        ));
    }
    if storage.value_slots.is_empty() && manifest.dag_summary.current_count != 0 {
        return Err(RusticolError::artifact(
            "generic value storage has no value slots",
        ));
    }
    let current_slots = &schema.current_storage.current_slots;
    let mut offset = 0usize;
    for (index, slot) in storage.value_slots.iter().enumerate() {
        if slot.value_slot_id != index
            || slot.component_start != offset
            || slot.component_stop != slot.component_start + slot.dimension
            || slot.dimension == 0
        {
            return Err(RusticolError::artifact(format!(
                "generic value slot {index} has inconsistent component range"
            )));
        }
        let current = current_slots.get(slot.current_id).ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic value slot {index} references missing current {}",
                slot.current_id
            ))
        })?;
        if current.dimension != slot.dimension || current.is_source != slot.is_source {
            return Err(RusticolError::artifact(format!(
                "generic value slot {index} does not match its current slot"
            )));
        }
        if slot.current_component_start != current.component_start
            || slot.current_component_stop != current.component_stop
            || slot.particle_id != current.particle_id
            || slot.external_mask != current.external_mask
            || (!storage.metadata_compacted && slot.external_labels != current.external_labels)
            || slot.momentum_mask != current.momentum_mask
            || slot.chirality != current.chirality
        {
            return Err(RusticolError::artifact(format!(
                "generic value slot {index} does not preserve its current identity"
            )));
        }
        validate_generic_propagator(index, slot, current)?;
        match slot.variant.as_str() {
            "source" => {
                if !slot.is_source || slot.applies_propagator {
                    return Err(RusticolError::artifact(format!(
                        "generic source value slot {index} is inconsistent"
                    )));
                }
            }
            "propagated" => {
                if slot.is_source || !slot.applies_propagator {
                    return Err(RusticolError::artifact(format!(
                        "generic propagated value slot {index} is inconsistent"
                    )));
                }
            }
            "unpropagated" => {
                if slot.is_source || slot.applies_propagator {
                    return Err(RusticolError::artifact(format!(
                        "generic unpropagated value slot {index} is inconsistent"
                    )));
                }
            }
            other => {
                return Err(RusticolError::artifact(format!(
                    "generic value slot {index} has unsupported variant {other:?}"
                )));
            }
        }
        offset = slot.component_stop;
    }
    if storage.component_count != offset {
        return Err(RusticolError::artifact(
            "generic value storage component_count is inconsistent",
        ));
    }
    Ok(())
}

fn validate_generic_propagator(
    index: usize,
    slot: &GenericValueSlotManifest,
    current: &GenericCurrentSlotManifest,
) -> RusticolResult<()> {
    let propagator = &slot.propagator;
    let identity = &propagator.identity;
    if propagator.particle_id != slot.particle_id
        || propagator.chirality != slot.chirality
        || identity.pdg_label != slot.particle_id
        || identity.canonical_id.is_empty()
        || identity.species_id.is_empty()
        || identity.anti_canonical_id.is_empty()
        || identity.display_name.is_empty()
        || identity.anti_display_name.is_empty()
        || identity.self_conjugate != (identity.pdg_label == identity.anti_pdg_label)
        || identity.self_conjugate != (identity.canonical_id == identity.anti_canonical_id)
        || propagator.backend.is_empty()
        || propagator.basis.is_empty()
        || propagator.kernel.is_empty()
        || propagator.description.is_empty()
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot {index} has inconsistent propagator identity"
        )));
    }
    if identity.self_conjugate
        != matches!(
            identity.orientation,
            GenericSourceOrientationManifest::SelfConjugate
        )
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot {index} has inconsistent propagator orientation"
        )));
    }
    for value in [
        propagator.numerator.as_ref(),
        propagator.denominator.as_ref(),
        propagator.mass_parameter.as_ref(),
        propagator.width_parameter.as_ref(),
        propagator.custom_source.as_ref(),
        propagator.auxiliary_policy.as_ref(),
    ] {
        if value.is_some_and(String::is_empty) {
            return Err(RusticolError::artifact(format!(
                "generic value slot {index} has empty propagator metadata"
            )));
        }
    }
    let expected_dimension = match propagator.kind {
        GenericPropagatorKindManifest::Scalar => Some(1),
        GenericPropagatorKindManifest::WeylFermion => Some(2),
        GenericPropagatorKindManifest::DiracFermion | GenericPropagatorKindManifest::Vector => {
            Some(4)
        }
        GenericPropagatorKindManifest::Spin2 => Some(16),
        GenericPropagatorKindManifest::Identity
        | GenericPropagatorKindManifest::Custom
        | GenericPropagatorKindManifest::Unsupported => None,
    };
    if expected_dimension.is_some_and(|dimension| dimension != current.dimension) {
        return Err(RusticolError::artifact(format!(
            "generic value slot {index} propagator dimension does not match its current"
        )));
    }
    if propagator.kind != GenericPropagatorKindManifest::Unsupported
        && !propagator.full_tensor_network_ready
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot {index} has an incomplete propagator contract"
        )));
    }
    match propagator.kind {
        GenericPropagatorKindManifest::Identity => {
            if propagator.applies_propagator
                || propagator.mass_class != GenericPropagatorMassClassManifest::NotApplicable
                || propagator.gauge.is_some()
                || propagator.auxiliary_policy.is_none()
            {
                return Err(RusticolError::artifact(format!(
                    "generic value slot {index} has an invalid identity propagator"
                )));
            }
        }
        GenericPropagatorKindManifest::Vector => {
            if !propagator.applies_propagator
                || !matches!(
                    propagator.gauge,
                    Some(
                        GenericPropagatorGaugeManifest::Feynman
                            | GenericPropagatorGaugeManifest::Unitary
                    )
                )
            {
                return Err(RusticolError::artifact(format!(
                    "generic value slot {index} has an invalid vector propagator"
                )));
            }
        }
        GenericPropagatorKindManifest::Spin2 => {
            if !propagator.applies_propagator
                || !matches!(
                    propagator.gauge,
                    Some(
                        GenericPropagatorGaugeManifest::DeDonder
                            | GenericPropagatorGaugeManifest::FierzPauli
                    )
                )
            {
                return Err(RusticolError::artifact(format!(
                    "generic value slot {index} has an invalid spin-2 propagator"
                )));
            }
        }
        GenericPropagatorKindManifest::Custom => {
            if !propagator.applies_propagator
                || propagator.gauge != Some(GenericPropagatorGaugeManifest::ModelSupplied)
                || propagator.custom_source.is_none()
            {
                return Err(RusticolError::artifact(format!(
                    "generic value slot {index} has an invalid custom propagator"
                )));
            }
        }
        GenericPropagatorKindManifest::Scalar
        | GenericPropagatorKindManifest::WeylFermion
        | GenericPropagatorKindManifest::DiracFermion
        | GenericPropagatorKindManifest::Unsupported => {
            if !propagator.applies_propagator || propagator.gauge.is_some() {
                return Err(RusticolError::artifact(format!(
                    "generic value slot {index} has an invalid standard propagator"
                )));
            }
        }
    }
    if propagator.kind != GenericPropagatorKindManifest::Identity
        && propagator.mass_class == GenericPropagatorMassClassManifest::NotApplicable
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot {index} has an invalid propagator mass class"
        )));
    }
    if propagator.goldstone_policy == GenericGoldstonePolicyManifest::Absorbed
        && !(propagator.kind == GenericPropagatorKindManifest::Vector
            && propagator.gauge == Some(GenericPropagatorGaugeManifest::Unitary))
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot {index} has an invalid Goldstone policy"
        )));
    }
    if propagator.goldstone_policy == GenericGoldstonePolicyManifest::ModelSupplied
        && propagator.kind != GenericPropagatorKindManifest::Custom
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot {index} has an invalid model-supplied Goldstone policy"
        )));
    }
    Ok(())
}

fn validate_generic_sources(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let schema = &manifest.runtime_schema;
    if schema.source_fill.source_count != manifest.dag_summary.source_count
        || schema.source_fill.sources.len() != manifest.dag_summary.source_count
    {
        return Err(RusticolError::artifact(
            "generic source count does not match DAG summary",
        ));
    }
    let current_slots = &schema.current_storage.current_slots;
    let mut source_offset = 0usize;
    let mut source_ir_by_momentum_slot = BTreeMap::new();
    let mut source_ir_by_canonical_id = BTreeMap::new();
    for (index, source) in schema.source_fill.sources.iter().enumerate() {
        if source.source_id != index {
            return Err(RusticolError::artifact(format!(
                "generic source id mismatch at index {index}"
            )));
        }
        let slot = current_slots.get(source.current_id).ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic source {index} references missing current {}",
                source.current_id
            ))
        })?;
        if !slot.is_source
            || slot.dimension != source.dimension
            || slot.particle_id != source.particle_id
        {
            return Err(RusticolError::artifact(format!(
                "generic source {index} does not match its current slot"
            )));
        }
        if source.current_component_start != slot.component_start
            || source.current_component_stop != slot.component_stop
            || source.source_parameter_start != source_offset
            || source.source_parameter_stop != source_offset + source.dimension
        {
            return Err(RusticolError::artifact(format!(
                "generic source {index} has inconsistent component offsets"
            )));
        }
        validate_value_slot_ref(
            &source.value_slot,
            source.current_id,
            Some("source"),
            &schema.value_storage,
        )?;
        if source.source_kind != "external-wavefunction" {
            return Err(RusticolError::artifact(format!(
                "unsupported generic source kind {:?}",
                source.source_kind
            )));
        }
        if source.side != "initial" && source.side != "final" {
            return Err(RusticolError::artifact(format!(
                "generic source {index} has invalid side {:?}",
                source.side
            )));
        }
        validate_source_wavefunction_metadata(index, source)?;
        validate_consistent_source_ir(
            &mut source_ir_by_momentum_slot,
            &mut source_ir_by_canonical_id,
            index,
            source,
        )?;
        let max_helicity =
            if source.source_ir.wavefunction_family == GenericWavefunctionFamilyManifest::Spin2 {
                2
            } else {
                1
            };
        if source.physical_pdg == 0
            || source.outgoing_pdg != source.particle_id
            || source.chirality.unsigned_abs() > 1
            || source.source_helicity.unsigned_abs() > max_helicity
            || source.spin_state.is_null()
            || !positive_json_integer(&source.helicity_ancestry)
            || source.color_state.is_null()
        {
            return Err(RusticolError::artifact(format!(
                "generic source {index} has invalid physics metadata"
            )));
        }
        let particle = schema
            .external_particles
            .get(source.input_momentum_slot)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "generic source {index} references missing momentum slot {}",
                    source.input_momentum_slot
                ))
            })?;
        if particle.label != source.leg_label
            || particle.role != source.side
            || particle.pdg != source.physical_pdg
            || particle.outgoing_pdg != source.outgoing_pdg
        {
            return Err(RusticolError::artifact(format!(
                "generic source {index} does not match its external particle"
            )));
        }
        source_offset = source.source_parameter_stop;
    }
    if source_offset != schema.parameter_layout.source_component_parameter_count {
        return Err(RusticolError::artifact(
            "generic source parameter count does not match source records",
        ));
    }
    Ok(())
}

pub(super) fn validate_consistent_source_ir(
    source_ir_by_momentum_slot: &mut BTreeMap<usize, GenericSourceIrManifest>,
    source_ir_by_canonical_id: &mut BTreeMap<String, GenericSourceIrManifest>,
    index: usize,
    source: &GenericSourceRecordManifest,
) -> RusticolResult<()> {
    if let Some(canonical) = source_ir_by_momentum_slot.get(&source.input_momentum_slot) {
        if canonical != &source.source_ir {
            return Err(RusticolError::artifact(format!(
                "generic source {index} disagrees with canonical SourceIR metadata for momentum slot {}",
                source.input_momentum_slot
            )));
        }
    } else {
        source_ir_by_momentum_slot.insert(source.input_momentum_slot, source.source_ir.clone());
    }

    let identity = &source.source_ir.identity;
    if let Some(canonical) = source_ir_by_canonical_id.get(&identity.canonical_id)
        && canonical != &source.source_ir
    {
        return Err(RusticolError::artifact(format!(
            "generic source {index} disagrees with canonical SourceIR metadata for oriented particle {:?}",
            identity.canonical_id
        )));
    }
    if let Some(antiparticle) = source_ir_by_canonical_id.get(&identity.anti_canonical_id) {
        validate_antiparticle_identity_pair(index, identity, &antiparticle.identity)?;
    }
    source_ir_by_canonical_id
        .entry(identity.canonical_id.clone())
        .or_insert_with(|| source.source_ir.clone());
    Ok(())
}

fn validate_antiparticle_identity_pair(
    index: usize,
    identity: &GenericParticleIdentityIrManifest,
    antiparticle: &GenericParticleIdentityIrManifest,
) -> RusticolResult<()> {
    let orientations_match = matches!(
        (identity.orientation, antiparticle.orientation),
        (
            GenericSourceOrientationManifest::Particle,
            GenericSourceOrientationManifest::Antiparticle,
        ) | (
            GenericSourceOrientationManifest::Antiparticle,
            GenericSourceOrientationManifest::Particle,
        ) | (
            GenericSourceOrientationManifest::SelfConjugate,
            GenericSourceOrientationManifest::SelfConjugate,
        )
    );
    if identity.anti_canonical_id != antiparticle.canonical_id
        || antiparticle.anti_canonical_id != identity.canonical_id
        || identity.species_id != antiparticle.species_id
        || identity.anti_pdg_label != antiparticle.pdg_label
        || antiparticle.anti_pdg_label != identity.pdg_label
        || identity.anti_display_name != antiparticle.display_name
        || antiparticle.anti_display_name != identity.display_name
        || identity.self_conjugate != antiparticle.self_conjugate
        || !orientations_match
    {
        return Err(RusticolError::artifact(format!(
            "generic source {index} has a non-involutive particle/antiparticle identity relation for {:?}",
            identity.canonical_id
        )));
    }
    Ok(())
}

pub(super) fn validate_source_wavefunction_metadata(
    index: usize,
    source: &GenericSourceRecordManifest,
) -> RusticolResult<()> {
    let source_ir = &source.source_ir;
    let identity = &source_ir.identity;
    if identity.canonical_id.is_empty()
        || identity.species_id.is_empty()
        || identity.anti_canonical_id.is_empty()
        || identity.display_name.is_empty()
        || identity.anti_display_name.is_empty()
    {
        return Err(RusticolError::artifact(format!(
            "generic source {index} has incomplete typed particle identity"
        )));
    }
    if identity.anti_pdg_label == 0 {
        return Err(RusticolError::artifact(format!(
            "generic source {index} has an invalid antiparticle relation"
        )));
    }
    let canonical_self_conjugate = identity.canonical_id == identity.anti_canonical_id;
    let pdg_self_conjugate = identity.pdg_label == identity.anti_pdg_label;
    if identity.self_conjugate != canonical_self_conjugate
        || identity.self_conjugate != pdg_self_conjugate
        || (identity.orientation == GenericSourceOrientationManifest::SelfConjugate)
            != identity.self_conjugate
    {
        return Err(RusticolError::artifact(format!(
            "generic source {index} orientation is inconsistent with its antiparticle relation"
        )));
    }

    if source_ir.component_dimension == 0
        || source_ir.states.is_empty()
        || source_ir.basis.is_empty()
        || source_ir
            .mass_parameter
            .as_ref()
            .is_some_and(String::is_empty)
        || source_ir
            .width_parameter
            .as_ref()
            .is_some_and(String::is_empty)
    {
        return Err(RusticolError::artifact(format!(
            "generic source {index} has invalid typed SourceIR metadata"
        )));
    }
    validate_crossing_ir(index, "declared", &source_ir.crossing)?;
    validate_crossing_ir(index, "applied", &source.applied_crossing)?;

    let expected_statistics = match source_ir.wavefunction_family {
        GenericWavefunctionFamilyManifest::Fermion => GenericParticleStatisticsManifest::Fermion,
        GenericWavefunctionFamilyManifest::Ghost => GenericParticleStatisticsManifest::Ghost,
        GenericWavefunctionFamilyManifest::Auxiliary => {
            GenericParticleStatisticsManifest::Auxiliary
        }
        GenericWavefunctionFamilyManifest::Scalar
        | GenericWavefunctionFamilyManifest::Vector
        | GenericWavefunctionFamilyManifest::Spin2 => GenericParticleStatisticsManifest::Boson,
    };
    if source_ir.statistics != expected_statistics {
        return Err(RusticolError::artifact(format!(
            "generic source {index} statistics disagree with its wavefunction family"
        )));
    }

    match (source_ir.wavefunction_family, source_ir.component_dimension) {
        (GenericWavefunctionFamilyManifest::Fermion, 2 | 4) => {
            if identity.self_conjugate {
                return Err(RusticolError::artifact(format!(
                    "generic source {index} is an unsupported self-conjugate fermion source"
                )));
            }
        }
        (GenericWavefunctionFamilyManifest::Scalar, 1)
        | (GenericWavefunctionFamilyManifest::Vector, 4)
        | (GenericWavefunctionFamilyManifest::Spin2, 16) => {}
        (kind, dimension) => {
            return Err(RusticolError::artifact(format!(
                "generic source {index} has unsupported wavefunction kind {:?} with dimension {dimension}",
                kind.as_str()
            )));
        }
    }

    for (field, matches) in [
        ("particle_id", source.particle_id == identity.pdg_label),
        (
            "anti_particle_id",
            source.anti_particle_id == identity.anti_pdg_label,
        ),
        (
            "source_orientation",
            source.source_orientation == identity.orientation,
        ),
        (
            "wavefunction_kind",
            source.wavefunction_kind == source_ir.wavefunction_family.as_str(),
        ),
        (
            "dimension",
            source.dimension == source_ir.component_dimension,
        ),
        ("source_basis", source.source_basis == source_ir.basis),
        (
            "crossing",
            source.crossing
                == source
                    .applied_crossing
                    .momentum_transform
                    .legacy_projection(),
        ),
    ] {
        if !matches {
            return Err(RusticolError::artifact(format!(
                "generic source {index} flattened field {field:?} disagrees with typed metadata"
            )));
        }
    }

    match source.side.as_str() {
        "initial" if source.applied_crossing != source_ir.crossing => {
            return Err(RusticolError::artifact(format!(
                "generic source {index} applied crossing does not match its declared SourceIR crossing"
            )));
        }
        "final" if !source.applied_crossing.is_identity() => {
            return Err(RusticolError::artifact(format!(
                "generic source {index} final-state applied crossing is not identity"
            )));
        }
        "initial" | "final" => {}
        _ => {
            return Err(RusticolError::artifact(format!(
                "generic source {index} has invalid side {:?}",
                source.side
            )));
        }
    }
    let current_spin_state: GenericSourceSpinStateManifest =
        serde_json::from_value(source.spin_state.clone()).map_err(|_| {
            RusticolError::artifact(format!(
                "generic source {index} has an invalid flattened spin state"
            ))
        })?;
    let current_state = GenericSourceStateIrManifest {
        helicity: source.source_helicity,
        chirality: source.chirality,
        spin_state: current_spin_state,
    };
    let max_helicity = if source_ir.wavefunction_family == GenericWavefunctionFamilyManifest::Spin2
    {
        2
    } else {
        1
    };
    let mut state_is_declared = false;
    for (state_index, declared_state) in source_ir.states.iter().enumerate() {
        if declared_state.chirality.unsigned_abs() > 1
            || declared_state.helicity.unsigned_abs() > max_helicity
        {
            return Err(RusticolError::artifact(format!(
                "generic source {index} SourceIR state {state_index} is outside the supported helicity/chirality range"
            )));
        }
        let transformed = declared_state
            .transformed(&source.applied_crossing)
            .map_err(|message| {
                RusticolError::artifact(format!(
                    "generic source {index} SourceIR state {state_index} is invalid: {message}"
                ))
            })?;
        state_is_declared |= transformed == current_state;
    }
    if !state_is_declared {
        return Err(RusticolError::artifact(format!(
            "generic source {index} current state is not declared by its typed SourceIR"
        )));
    }
    Ok(())
}

fn validate_crossing_ir(
    index: usize,
    label: &str,
    crossing: &GenericCrossingIrManifest,
) -> RusticolResult<()> {
    if ![-1, 1].contains(&crossing.helicity_factor)
        || ![-1, 1].contains(&crossing.chirality_factor)
        || ![-1, 1].contains(&crossing.spin_state_factor)
        || !crossing.phase.iter().all(|component| component.is_finite())
        || crossing.phase == [0.0, 0.0]
    {
        return Err(RusticolError::artifact(format!(
            "generic source {index} has invalid {label} CrossingIR metadata"
        )));
    }
    Ok(())
}

fn validate_generic_momentum_slots(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let schema = &manifest.runtime_schema;
    for (index, slot) in schema.momentum_slots.iter().enumerate() {
        if slot.momentum_slot_id != index
            || slot.component_start != 4 * index
            || slot.component_stop != slot.component_start + 4
            || !slot.real_valued
            || slot.momentum_mask == 0
            || slot.external_labels.is_empty()
        {
            return Err(RusticolError::artifact(format!(
                "generic momentum slot {index} is inconsistent"
            )));
        }
        for label in &slot.external_labels {
            if *label == 0 || *label > schema.external_particles.len() {
                return Err(RusticolError::artifact(format!(
                    "generic momentum slot {index} has invalid external label {label}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_generic_stages(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let schema = &manifest.runtime_schema;
    let current_count = schema.current_storage.current_slots.len();
    let value_count = schema.value_storage.value_slots.len();
    let momentum_count = schema.momentum_slots.len();
    let input_value_variants =
        build_input_value_variants(&schema.current_storage, &schema.value_storage)?;
    let mut seen_interactions = 0usize;
    let mut seen_interaction_ids = BTreeSet::new();
    for (stage_offset, stage) in schema.stages.iter().enumerate() {
        if stage.stage_index != stage_offset + 1
            || stage.stage_kind != "current-combine"
            || stage.subset_size < 2
        {
            return Err(RusticolError::artifact(format!(
                "generic stage {} is inconsistent",
                stage.stage_index
            )));
        }
        if stage.interactions_compacted {
            if !stage.interactions.is_empty()
                || stage.interaction_ids.len() != stage.interaction_count
            {
                return Err(RusticolError::artifact(format!(
                    "generic compact stage {} has inconsistent interaction metadata",
                    stage.stage_index
                )));
            }
            for interaction_id in &stage.interaction_ids {
                if !seen_interaction_ids.insert(*interaction_id) {
                    return Err(RusticolError::artifact(format!(
                        "generic compact stage {} repeats interaction {interaction_id}",
                        stage.stage_index
                    )));
                }
            }
        } else if stage.interaction_count != stage.interactions.len()
            || !stage.interaction_ids.is_empty()
        {
            return Err(RusticolError::artifact(format!(
                "generic stage {} is inconsistent",
                stage.stage_index
            )));
        }
        let mut input_current_membership = vec![false; current_count];
        let mut output_current_membership = vec![false; current_count];
        for (ids, membership) in [
            (&stage.input_current_ids, &mut input_current_membership),
            (&stage.output_current_ids, &mut output_current_membership),
        ] {
            for id in ids {
                if *id >= current_count {
                    return Err(RusticolError::artifact(format!(
                        "generic stage {} references invalid current {id}",
                        stage.stage_index
                    )));
                }
                membership[*id] = true;
            }
        }
        let mut input_value_membership = vec![false; value_count];
        let mut output_value_membership = vec![false; value_count];
        for (ids, membership) in [
            (&stage.input_value_slot_ids, &mut input_value_membership),
            (&stage.output_value_slot_ids, &mut output_value_membership),
        ] {
            for value_id in ids {
                if *value_id >= value_count {
                    return Err(RusticolError::artifact(format!(
                        "generic stage {} references invalid value slot {value_id}",
                        stage.stage_index
                    )));
                }
                membership[*value_id] = true;
            }
        }
        for interaction in &stage.interactions {
            if !seen_interaction_ids.insert(interaction.interaction_id) {
                return Err(RusticolError::artifact(format!(
                    "generic stage {} repeats interaction {}",
                    stage.stage_index, interaction.interaction_id
                )));
            }
            validate_generic_interaction(
                interaction,
                &schema.current_storage,
                &schema.value_storage,
                &input_value_variants,
                momentum_count,
            )?;
            if !input_current_membership[interaction.left_current_id]
                || !input_current_membership[interaction.right_current_id]
                || !output_current_membership[interaction.result_current_id]
            {
                return Err(RusticolError::artifact(format!(
                    "generic interaction {} is not listed in its stage inputs/outputs",
                    interaction.interaction_id
                )));
            }
            if !input_value_membership[interaction.left_value_slot.value_slot_id]
                || !input_value_membership[interaction.right_value_slot.value_slot_id]
                || interaction
                    .result_value_slots
                    .iter()
                    .any(|slot| !output_value_membership[slot.value_slot_id])
            {
                return Err(RusticolError::artifact(format!(
                    "generic interaction {} value slots are not listed in its stage inputs/outputs",
                    interaction.interaction_id
                )));
            }
        }
        seen_interactions += stage.interaction_count;
    }
    if seen_interactions != manifest.dag_summary.interaction_count {
        return Err(RusticolError::artifact(
            "generic stage interaction count does not match DAG summary",
        ));
    }
    Ok(())
}

fn validate_generic_interaction(
    interaction: &GenericInteractionManifest,
    current_storage: &GenericCurrentStorageManifest,
    value_storage: &GenericValueStorageManifest,
    input_value_variants: &[Option<GenericInputValueVariant>],
    momentum_count: usize,
) -> RusticolResult<()> {
    let current_count = current_storage.current_slots.len();
    for current_id in [
        interaction.left_current_id,
        interaction.right_current_id,
        interaction.result_current_id,
    ] {
        if current_id >= current_count {
            return Err(RusticolError::artifact(format!(
                "generic interaction {} references invalid current {current_id}",
                interaction.interaction_id
            )));
        }
    }
    for slot_id in [
        interaction.momentum_slots.left,
        interaction.momentum_slots.right,
        interaction.momentum_slots.result,
    ] {
        if slot_id >= momentum_count {
            return Err(RusticolError::artifact(format!(
                "generic interaction {} references invalid momentum slot {slot_id}",
                interaction.interaction_id
            )));
        }
    }
    validate_slot_ref(
        &interaction.left_slot,
        interaction.left_current_id,
        current_storage,
    )?;
    validate_slot_ref(
        &interaction.right_slot,
        interaction.right_current_id,
        current_storage,
    )?;
    validate_slot_ref(
        &interaction.result_slot,
        interaction.result_current_id,
        current_storage,
    )?;
    validate_value_slot_ref(
        &interaction.left_value_slot,
        interaction.left_current_id,
        Some(input_value_variant(
            input_value_variants,
            interaction.left_current_id,
        )?),
        value_storage,
    )?;
    validate_value_slot_ref(
        &interaction.right_value_slot,
        interaction.right_current_id,
        Some(input_value_variant(
            input_value_variants,
            interaction.right_current_id,
        )?),
        value_storage,
    )?;
    if interaction.result_value_slots.is_empty() {
        return Err(RusticolError::artifact(format!(
            "generic interaction {} has no result value slots",
            interaction.interaction_id
        )));
    }
    for result_slot in &interaction.result_value_slots {
        validate_value_slot_ref(
            result_slot,
            interaction.result_current_id,
            None,
            value_storage,
        )?;
        if result_slot.variant != "propagated" && result_slot.variant != "unpropagated" {
            return Err(RusticolError::artifact(format!(
                "generic interaction {} has invalid result value variant {:?}",
                interaction.interaction_id, result_slot.variant
            )));
        }
    }
    if interaction.vertex_kind < 0 {
        return Err(RusticolError::artifact(format!(
            "generic interaction {} has invalid vertex kind {}",
            interaction.interaction_id, interaction.vertex_kind
        )));
    }
    if interaction.vertex_particles.len() != 3
        || interaction.coupling.len() != 2
        || interaction.color_weight.len() != 2
        || interaction.accumulation != "sum-into-result-current"
        || interaction.lowering.kind != interaction.vertex_kind
        || interaction.lowering.full_tensor_network_ready != interaction.full_tensor_network_ready
        || interaction.lowering.backend.is_empty()
        || interaction.lowering.expression_head.is_empty()
        || interaction.lowering.kernel.is_empty()
        || interaction.lowering.input_roles.len() != 2
        || interaction.lowering.output_role.is_empty()
        || interaction.lowering.coupling_mode.is_empty()
        || interaction
            .lowering
            .tensor_names
            .iter()
            .any(|name| name.is_empty())
        || interaction.lowering.description.is_empty()
    {
        return Err(RusticolError::artifact(format!(
            "generic interaction {} has inconsistent model/lowering metadata",
            interaction.interaction_id
        )));
    }
    if interaction.result_requires_propagated_value
        && !interaction
            .result_value_slots
            .iter()
            .any(|slot| slot.variant == "propagated")
    {
        return Err(RusticolError::artifact(format!(
            "generic interaction {} declares a propagated result without a propagated slot",
            interaction.interaction_id
        )));
    }
    if interaction.result_requires_unpropagated_value
        && !interaction
            .result_value_slots
            .iter()
            .any(|slot| slot.variant == "unpropagated")
    {
        return Err(RusticolError::artifact(format!(
            "generic interaction {} declares an unpropagated result without an unpropagated slot",
            interaction.interaction_id
        )));
    }
    if !interaction.full_tensor_network_ready {
        return Err(RusticolError::artifact(format!(
            "generic interaction {} is not ready for tensor-network lowering",
            interaction.interaction_id
        )));
    }
    Ok(())
}

fn validate_generic_amplitudes(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let schema = &manifest.runtime_schema;
    let stage = &schema.amplitude_stage;
    if stage.stage_kind != "amplitude-roots"
        || stage.output_count != stage.roots.len()
        || stage.output_count != manifest.dag_summary.amplitude_root_count
    {
        return Err(RusticolError::artifact(
            "generic amplitude stage output count is inconsistent",
        ));
    }
    let current_count = schema.current_storage.current_slots.len();
    for (index, root) in stage.roots.iter().enumerate() {
        if root.output_index != index || root.root_id != index {
            return Err(RusticolError::artifact(format!(
                "generic amplitude root index mismatch at output {index}"
            )));
        }
        if root.left_current_id >= current_count || root.right_current_id >= current_count {
            return Err(RusticolError::artifact(format!(
                "generic amplitude root {index} references invalid current"
            )));
        }
        if root.kind != "direct-contraction" && root.kind != "vertex-closure" {
            return Err(RusticolError::artifact(format!(
                "generic amplitude root {index} has unsupported kind {:?}",
                root.kind
            )));
        }
        if root.coupling.len() != 2
            || root.color_weight.len() != 2
            || root.contraction.is_empty()
            || !root.helicity_weight.is_finite()
            || root.coherent_group_id.as_ref().is_some_and(Value::is_null)
        {
            return Err(RusticolError::artifact(format!(
                "generic amplitude root {index} has inconsistent physics metadata"
            )));
        }
        validate_amplitude_root_kind(index, root, &schema.current_storage)?;
        validate_slot_ref(
            &root.left_slot,
            root.left_current_id,
            &schema.current_storage,
        )?;
        validate_slot_ref(
            &root.right_slot,
            root.right_current_id,
            &schema.current_storage,
        )?;
        validate_value_slot_ref(
            &root.left_value_slot,
            root.left_current_id,
            Some(amplitude_value_variant(
                &schema.current_storage,
                root.left_current_id,
            )?),
            &schema.value_storage,
        )?;
        validate_value_slot_ref(
            &root.right_value_slot,
            root.right_current_id,
            Some(amplitude_value_variant(
                &schema.current_storage,
                root.right_current_id,
            )?),
            &schema.value_storage,
        )?;
        validate_amplitude_contraction(
            index,
            root,
            &schema.current_storage,
            &schema.value_storage,
        )?;
    }
    Ok(())
}

pub(super) fn validate_amplitude_contraction(
    index: usize,
    root: &GenericAmplitudeRootManifest,
    current_storage: &GenericCurrentStorageManifest,
    value_storage: &GenericValueStorageManifest,
) -> RusticolResult<()> {
    let contraction = &root.contraction_ir;
    if contraction.name.is_empty()
        || contraction.left_basis.is_empty()
        || contraction.right_basis.is_empty()
        || contraction.coefficients.is_empty()
        || contraction
            .metric_signature
            .as_ref()
            .is_some_and(String::is_empty)
        || contraction
            .coefficients
            .iter()
            .flatten()
            .any(|component| !component.is_finite())
        || !contraction
            .coefficients
            .iter()
            .any(|coefficient| *coefficient != [0.0, 0.0])
    {
        return Err(RusticolError::artifact(format!(
            "generic amplitude root {index} has invalid typed contraction metadata"
        )));
    }
    if root.contraction != contraction.name {
        return Err(RusticolError::artifact(format!(
            "generic amplitude root {index} contraction projection does not match ContractionIR"
        )));
    }

    if root.kind == "vertex-closure" {
        if contraction.coefficients.len() != 1
            || contraction.left_basis != "scalar"
            || contraction.right_basis != "scalar"
            || contraction.chirality_relation != GenericContractionChiralityRelationManifest::Any
        {
            return Err(RusticolError::artifact(format!(
                "generic vertex-closure root {index} does not declare a scalar projection"
            )));
        }
        return Ok(());
    }

    let left_current = current_storage
        .current_slots
        .get(root.left_current_id)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic amplitude root {index} references a missing left current"
            ))
        })?;
    let right_current = current_storage
        .current_slots
        .get(root.right_current_id)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic amplitude root {index} references a missing right current"
            ))
        })?;
    let left_value = value_storage
        .value_slots
        .get(root.left_value_slot.value_slot_id)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic amplitude root {index} references a missing left value slot"
            ))
        })?;
    let right_value = value_storage
        .value_slots
        .get(root.right_value_slot.value_slot_id)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic amplitude root {index} references a missing right value slot"
            ))
        })?;
    if contraction.coefficients.len() != root.left_value_slot.dimension
        || contraction.coefficients.len() != root.right_value_slot.dimension
        || contraction.left_basis != left_value.propagator.basis
        || contraction.right_basis != right_value.propagator.basis
    {
        return Err(RusticolError::artifact(format!(
            "generic amplitude root {index} ContractionIR does not match its value slots"
        )));
    }
    let chirality_matches = contraction_chiralities_match(
        contraction.chirality_relation,
        left_current.chirality,
        right_current.chirality,
    );
    if !chirality_matches {
        return Err(RusticolError::artifact(format!(
            "generic amplitude root {index} violates its ContractionIR chirality relation"
        )));
    }
    Ok(())
}

pub(super) fn validate_amplitude_root_kind(
    index: usize,
    root: &GenericAmplitudeRootManifest,
    current_storage: &GenericCurrentStorageManifest,
) -> RusticolResult<()> {
    match root.kind.as_str() {
        "direct-contraction" => {
            if root.vertex_kind.is_some() || root.vertex_particles.is_some() {
                return Err(RusticolError::artifact(format!(
                    "generic direct-contraction root {index} carries vertex metadata"
                )));
            }
        }
        "vertex-closure" => {
            let vertex_kind = root.vertex_kind.filter(|kind| *kind > 0).ok_or_else(|| {
                RusticolError::artifact(format!(
                    "generic vertex-closure root {index} has an invalid vertex kind"
                ))
            })?;
            let particles = root
                .vertex_particles
                .as_ref()
                .filter(|values| values.len() == 3)
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "generic vertex-closure root {index} is missing vertex particles"
                    ))
                })?;
            let left = current_storage
                .current_slots
                .get(root.left_current_id)
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "generic vertex-closure root {index} references a missing left current"
                    ))
                })?;
            let right = current_storage
                .current_slots
                .get(root.right_current_id)
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "generic vertex-closure root {index} references a missing right current"
                    ))
                })?;
            if particles[0] != left.particle_id || particles[1] != right.particle_id {
                return Err(RusticolError::artifact(format!(
                    "generic vertex-closure root {index} vertex {vertex_kind} does not match its input currents"
                )));
            }
        }
        _ => {
            return Err(RusticolError::artifact(format!(
                "generic amplitude root {index} has unsupported kind {:?}",
                root.kind
            )));
        }
    }
    Ok(())
}

pub(super) fn valid_current_chirality(chirality: i32) -> bool {
    (-1..=1).contains(&chirality)
}

pub(super) fn contraction_chiralities_match(
    relation: GenericContractionChiralityRelationManifest,
    left: i32,
    right: i32,
) -> bool {
    match relation {
        GenericContractionChiralityRelationManifest::Any => true,
        GenericContractionChiralityRelationManifest::Equal => left == right,
        GenericContractionChiralityRelationManifest::Opposite => {
            right.checked_neg().is_some_and(|opposite| left == opposite)
        }
    }
}

fn validate_generic_stage_evaluators(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let Some(stage_evaluators) = &manifest.compiled.stage_evaluators else {
        return Ok(());
    };
    let schema = &manifest.runtime_schema;
    let expected_parameter_count = schema.parameter_layout.value_component_count
        + schema.parameter_layout.momentum_parameter_count
        + schema.parameter_layout.model_parameter_count;
    let expected_real_inputs = (schema.parameter_layout.value_component_count
        ..expected_parameter_count)
        .collect::<Vec<_>>();
    let header_is_global = stage_evaluators.parameter_layout == "global-value-momentum"
        && stage_evaluators.parameter_count == expected_parameter_count
        && stage_evaluators.value_parameter_count == schema.parameter_layout.value_component_count
        && stage_evaluators.momentum_parameter_count
            == schema.parameter_layout.momentum_parameter_count
        && stage_evaluators.model_parameter_count == schema.parameter_layout.model_parameter_count
        && stage_evaluators.real_valued_inputs == expected_real_inputs;
    let header_is_stage_local = stage_evaluators.parameter_layout == "stage-local-value-momentum"
        && stage_evaluators.parameter_count == 0
        && stage_evaluators.value_parameter_count == 0
        && stage_evaluators.momentum_parameter_count == 0
        && stage_evaluators.model_parameter_count == 0
        && stage_evaluators.real_valued_inputs.is_empty();
    if stage_evaluators.kind != "generic-dag-stage-evaluator-artifacts"
        || (!header_is_global && !header_is_stage_local)
        || stage_evaluators.stage_count != schema.stages.len() + 1
        || stage_evaluators.stages.len() != schema.stages.len()
        || !stage_evaluators.runtime_available
        || stage_evaluators.runtime_unavailable_message.is_some()
    {
        return Err(RusticolError::artifact(
            "generic stage evaluator artifact header is inconsistent with runtime schema",
        ));
    }
    for (stage, runtime_stage) in stage_evaluators.stages.iter().zip(&schema.stages) {
        validate_generic_serialized_stage_evaluator(
            stage,
            runtime_stage.stage_index,
            "current-combine",
            Some(runtime_stage.subset_size),
            expected_parameter_count,
            Some(runtime_stage),
            None,
        )?;
    }
    validate_generic_serialized_stage_evaluator(
        &stage_evaluators.amplitude_stage,
        0,
        "amplitude-roots",
        None,
        expected_parameter_count,
        None,
        Some(&schema.amplitude_stage),
    )?;
    Ok(())
}

fn validate_generic_serialized_stage_evaluator(
    stage: &GenericSerializedStageEvaluatorManifest,
    expected_stage_index: usize,
    expected_stage_kind: &str,
    expected_subset_size: Option<usize>,
    expected_parameter_count: usize,
    runtime_stage: Option<&GenericStageManifest>,
    amplitude_stage: Option<&GenericAmplitudeStageManifest>,
) -> RusticolResult<()> {
    let stage_is_global = stage.parameter_layout == "global-value-momentum";
    let stage_is_local = stage.parameter_layout == "stage-local-value-momentum";
    if stage.stage_index != expected_stage_index
        || stage.stage_kind != expected_stage_kind
        || stage.subset_size != expected_subset_size
        || (!stage_is_global && !stage_is_local)
        || !stage.expression_ready
        || !stage.blockers.is_empty()
        || stage.evaluator_label.is_empty()
        || stage.output_length == 0
    {
        return Err(RusticolError::artifact(format!(
            "generic serialized stage evaluator {} is inconsistent",
            stage.evaluator_label
        )));
    }
    validate_generic_stage_output_slots(stage)?;
    let (input_len, output_len) = stage.evaluator.io_len()?;
    if stage_is_global && input_len != expected_parameter_count {
        return Err(RusticolError::artifact(format!(
            "generic serialized stage evaluator {} has inconsistent global evaluator input length",
            stage.evaluator_label
        )));
    }
    if stage_is_local {
        validate_generic_stage_input_components(stage, expected_parameter_count)?;
        if input_len != stage.parameter_count
            || stage.parameter_count != stage.input_components.len()
            || stage.value_parameter_count
                + stage.momentum_parameter_count
                + stage.model_parameter_count
                != stage.parameter_count
            || stage
                .real_valued_inputs
                .iter()
                .any(|index| *index >= stage.parameter_count)
        {
            return Err(RusticolError::artifact(format!(
                "generic serialized stage evaluator {} has inconsistent local evaluator input metadata",
                stage.evaluator_label
            )));
        }
    }
    if output_len != stage.output_length {
        return Err(RusticolError::artifact(format!(
            "generic serialized stage evaluator {} has inconsistent evaluator IO",
            stage.evaluator_label
        )));
    }
    if let Some(runtime_stage) = runtime_stage {
        let expected_interactions = if runtime_stage.interactions_compacted {
            runtime_stage.interaction_ids.clone()
        } else {
            runtime_stage
                .interactions
                .iter()
                .map(|interaction| interaction.interaction_id)
                .collect::<Vec<_>>()
        };
        if stage.input_value_slot_ids != runtime_stage.input_value_slot_ids
            || stage.output_value_slot_ids != runtime_stage.output_value_slot_ids
            || stage.interaction_ids != expected_interactions
        {
            return Err(RusticolError::artifact(format!(
                "generic serialized stage evaluator {} does not match runtime stage slots",
                stage.evaluator_label
            )));
        }
    }
    if let Some(amplitude_stage) = amplitude_stage
        && (!stage.output_value_slot_ids.is_empty()
            || !stage.interaction_ids.is_empty()
            || stage.output_length != amplitude_stage.output_count)
    {
        return Err(RusticolError::artifact(
            "generic serialized amplitude evaluator does not match amplitude stage",
        ));
    }
    Ok(())
}

fn validate_generic_stage_output_slots(
    stage: &GenericSerializedStageEvaluatorManifest,
) -> RusticolResult<()> {
    if stage.output_slots.is_empty() {
        return Err(RusticolError::artifact(format!(
            "generic serialized stage evaluator {} has no output slots",
            stage.evaluator_label
        )));
    }
    let mut max_output_stop = 0usize;
    for slot in &stage.output_slots {
        if slot.variant.is_empty()
            || slot.component_stop < slot.component_start
            || slot.output_stop <= slot.output_start
            || slot.output_stop > stage.output_length
        {
            return Err(RusticolError::artifact(format!(
                "generic serialized stage evaluator {} has invalid output slot",
                stage.evaluator_label
            )));
        }
        if stage.stage_kind == "amplitude-roots" {
            if slot.value_slot_id != -1 || slot.current_id != -1 || slot.variant != "amplitude-root"
            {
                return Err(RusticolError::artifact(
                    "generic serialized amplitude evaluator has invalid output slot metadata",
                ));
            }
        } else if slot.value_slot_id < 0 || slot.current_id < 0 || slot.variant == "amplitude-root"
        {
            return Err(RusticolError::artifact(format!(
                "generic serialized stage evaluator {} has invalid current output slot metadata",
                stage.evaluator_label
            )));
        }
        max_output_stop = max_output_stop.max(slot.output_stop);
    }
    if max_output_stop != stage.output_length {
        return Err(RusticolError::artifact(format!(
            "generic serialized stage evaluator {} output slots do not cover evaluator outputs",
            stage.evaluator_label
        )));
    }
    Ok(())
}

fn validate_generic_stage_input_components(
    stage: &GenericSerializedStageEvaluatorManifest,
    global_parameter_count: usize,
) -> RusticolResult<()> {
    if stage.input_components.is_empty() {
        return Err(RusticolError::artifact(format!(
            "generic serialized stage evaluator {} has no local input components",
            stage.evaluator_label
        )));
    }
    let mut seen_parameters = BTreeSet::new();
    for component in &stage.input_components {
        if component.kind != "value"
            && component.kind != "momentum"
            && component.kind != "model_parameter"
        {
            return Err(RusticolError::artifact(format!(
                "generic serialized stage evaluator {} has invalid local input kind {:?}",
                stage.evaluator_label, component.kind
            )));
        }
        if component.global_component >= global_parameter_count
            || component.parameter_index >= stage.parameter_count
            || component.source_id >= global_parameter_count
            || component.component >= global_parameter_count
            || !seen_parameters.insert(component.parameter_index)
        {
            return Err(RusticolError::artifact(format!(
                "generic serialized stage evaluator {} has invalid local input component map",
                stage.evaluator_label
            )));
        }
        if component.real_valued
            && component.kind != "momentum"
            && component.kind != "model_parameter"
        {
            return Err(RusticolError::artifact(format!(
                "generic serialized stage evaluator {} marks a complex local input as real",
                stage.evaluator_label
            )));
        }
    }
    if seen_parameters.len() != stage.parameter_count {
        return Err(RusticolError::artifact(format!(
            "generic serialized stage evaluator {} local input map is not dense",
            stage.evaluator_label
        )));
    }
    let real_inputs = stage
        .input_components
        .iter()
        .filter_map(|component| {
            if component.real_valued {
                Some(component.parameter_index)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if real_inputs != stage.real_valued_inputs {
        return Err(RusticolError::artifact(format!(
            "generic serialized stage evaluator {} local real-input metadata is inconsistent",
            stage.evaluator_label
        )));
    }
    Ok(())
}

pub(super) fn validate_slot_ref(
    slot: &GenericSlotRefManifest,
    current_id: usize,
    current_storage: &GenericCurrentStorageManifest,
) -> RusticolResult<()> {
    if slot.current_id != current_id
        || slot.component_stop != slot.component_start + slot.dimension
        || slot.dimension == 0
    {
        return Err(RusticolError::artifact(format!(
            "generic slot reference for current {current_id} is inconsistent"
        )));
    }
    let storage_slot = current_storage
        .current_slots
        .get(current_id)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic slot reference for current {current_id} points outside current storage"
            ))
        })?;
    if storage_slot.current_id != slot.current_id
        || storage_slot.component_start != slot.component_start
        || storage_slot.component_stop != slot.component_stop
        || storage_slot.dimension != slot.dimension
    {
        return Err(RusticolError::artifact(format!(
            "generic slot reference for current {current_id} does not match current storage"
        )));
    }
    Ok(())
}

fn validate_value_slot_ref(
    slot: &GenericValueSlotRefManifest,
    current_id: usize,
    expected_variant: Option<&str>,
    value_storage: &GenericValueStorageManifest,
) -> RusticolResult<()> {
    if slot.current_id != current_id
        || slot.component_stop != slot.component_start + slot.dimension
        || slot.dimension == 0
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot reference for current {current_id} is inconsistent"
        )));
    }
    if let Some(expected) = expected_variant
        && slot.variant != expected
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot reference for current {current_id} uses variant {:?}, expected {expected:?}",
            slot.variant
        )));
    }
    let storage_slot = value_storage
        .value_slots
        .get(slot.value_slot_id)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic value slot reference for current {current_id} points outside value storage"
            ))
        })?;
    if storage_slot.current_id != slot.current_id
        || storage_slot.variant != slot.variant
        || storage_slot.component_start != slot.component_start
        || storage_slot.component_stop != slot.component_stop
        || storage_slot.dimension != slot.dimension
    {
        return Err(RusticolError::artifact(format!(
            "generic value slot reference for current {current_id} does not match value storage"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenericInputValueVariant {
    Source,
    Propagated,
    Unpropagated,
}

impl GenericInputValueVariant {
    fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Propagated => "propagated",
            Self::Unpropagated => "unpropagated",
        }
    }
}

fn build_input_value_variants(
    current_storage: &GenericCurrentStorageManifest,
    value_storage: &GenericValueStorageManifest,
) -> RusticolResult<Vec<Option<GenericInputValueVariant>>> {
    let mut variants = current_storage
        .current_slots
        .iter()
        .map(|current| {
            current
                .is_source
                .then_some(GenericInputValueVariant::Source)
        })
        .collect::<Vec<_>>();
    for slot in &value_storage.value_slots {
        let current = current_storage
            .current_slots
            .get(slot.current_id)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "generic value slot references missing current {}",
                    slot.current_id
                ))
            })?;
        if current.is_source {
            continue;
        }
        match slot.variant.as_str() {
            "propagated" => {
                variants[slot.current_id] = Some(GenericInputValueVariant::Propagated);
            }
            "unpropagated"
                if variants[slot.current_id] != Some(GenericInputValueVariant::Propagated) =>
            {
                variants[slot.current_id] = Some(GenericInputValueVariant::Unpropagated);
            }
            _ => {}
        }
    }
    Ok(variants)
}

fn input_value_variant(
    input_value_variants: &[Option<GenericInputValueVariant>],
    current_id: usize,
) -> RusticolResult<&'static str> {
    let variant = input_value_variants.get(current_id).ok_or_else(|| {
        RusticolError::artifact(format!(
            "generic input value references missing current {current_id}"
        ))
    })?;
    variant
        .map(GenericInputValueVariant::as_str)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic input value references current {current_id} without an input value slot"
            ))
        })
}

fn amplitude_value_variant(
    current_storage: &GenericCurrentStorageManifest,
    current_id: usize,
) -> RusticolResult<&'static str> {
    let current = current_storage
        .current_slots
        .get(current_id)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "generic amplitude value references missing current {current_id}"
            ))
        })?;
    Ok(if current.is_source {
        "source"
    } else {
        "unpropagated"
    })
}
