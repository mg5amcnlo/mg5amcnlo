// SPDX-License-Identifier: 0BSD

use super::*;
use crate::EagerCouplingRow;

#[derive(Clone, Copy, Debug)]
struct RuntimeLogicalParameterSlots {
    real: usize,
    imaginary: Option<usize>,
}

pub(super) fn prepare_eager_parameter_state(
    pack: &PreparedKernelPackManifest,
    active_kernel_ids: &BTreeSet<u32>,
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
    coupling_bytes: &[u8],
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<(
    EagerParameterProjection,
    Vec<u8>,
    Option<ModelParameterEvaluatorRuntime>,
)> {
    let logical_slots = eager_runtime_logical_parameter_slots(runtime_parameters)?;
    let has_derived_parameters = runtime_parameters
        .iter()
        .any(|parameter| parameter.kind == "derived_parameter_component");
    let canonical_indices =
        prepared_parameter_indices(pack, active_kernel_ids, has_derived_parameters)?;
    let mut entries = Vec::with_capacity(canonical_indices.len());
    let mut base_parameter_count = 0usize;
    for (name, prepared_index) in &canonical_indices {
        let slots = logical_slots.get(name).ok_or_else(|| {
            RusticolError::integrity(format!(
                "prepared parameter {name:?} is absent from the process runtime schema"
            ))
        })?;
        entries.push(EagerParameterProjectionEntry {
            prepared_index: *prepared_index,
            runtime_real_index: slots.real,
            runtime_imaginary_index: slots.imaginary,
        });
        base_parameter_count = base_parameter_count.max(prepared_index + 1);
    }

    let mut couplings = EagerCouplingRow::decode_table(coupling_bytes)?;
    let mut synthetic_by_runtime_index = BTreeMap::new();
    for row in &mut couplings {
        for parameter_id in [&mut row.real_parameter_id, &mut row.imag_parameter_id] {
            if *parameter_id == crate::MISSING_U32 {
                continue;
            }
            let runtime_index = usize::try_from(*parameter_id).map_err(|_| {
                RusticolError::artifact("eager coupling parameter index does not fit usize")
            })?;
            if runtime_index >= runtime_parameters.len() {
                return Err(RusticolError::integrity(format!(
                    "eager coupling references runtime parameter {runtime_index}, but only {} exist",
                    runtime_parameters.len()
                )));
            }
            let prepared_index = if let Some(index) = synthetic_by_runtime_index.get(&runtime_index)
            {
                *index
            } else {
                let index = base_parameter_count
                    .checked_add(synthetic_by_runtime_index.len())
                    .ok_or_else(|| {
                        RusticolError::artifact("eager synthetic coupling parameters overflow")
                    })?;
                synthetic_by_runtime_index.insert(runtime_index, index);
                entries.push(EagerParameterProjectionEntry {
                    prepared_index: index,
                    runtime_real_index: runtime_index,
                    runtime_imaginary_index: None,
                });
                index
            };
            *parameter_id = u32::try_from(prepared_index).map_err(|_| {
                RusticolError::artifact("eager synthetic coupling parameter exceeds u32")
            })?;
        }
    }
    let parameter_count = base_parameter_count
        .checked_add(synthetic_by_runtime_index.len())
        .ok_or_else(|| RusticolError::artifact("eager prepared parameter count overflows"))?;
    entries.sort_unstable_by_key(|entry| entry.prepared_index);
    let model_parameter_evaluator = load_prepared_model_parameter_evaluator(
        pack,
        runtime_parameters,
        &logical_slots,
        payloads,
    )?;
    Ok((
        EagerParameterProjection {
            parameter_count,
            entries,
        },
        EagerCouplingRow::encode_table(&couplings)?,
        model_parameter_evaluator,
    ))
}

fn eager_runtime_logical_parameter_slots(
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<BTreeMap<String, RuntimeLogicalParameterSlots>> {
    let mut direct = BTreeMap::new();
    let mut complex = BTreeMap::<String, (Option<usize>, Option<usize>)>::new();
    for parameter in runtime_parameters {
        if let Some(runtime_name) = &parameter.runtime_name {
            let slots = complex.entry(runtime_name.clone()).or_default();
            let target = match parameter.complex_component.as_deref() {
                Some("real") => &mut slots.0,
                Some("imag") => &mut slots.1,
                other => {
                    return Err(RusticolError::integrity(format!(
                        "eager runtime parameter {runtime_name:?} has invalid component {other:?}"
                    )));
                }
            };
            if target.replace(parameter.parameter_index).is_some() {
                return Err(RusticolError::integrity(format!(
                    "eager runtime parameter {runtime_name:?} repeats a complex component"
                )));
            }
        } else if direct
            .insert(
                parameter.name.clone(),
                RuntimeLogicalParameterSlots {
                    real: parameter.parameter_index,
                    imaginary: None,
                },
            )
            .is_some()
        {
            return Err(RusticolError::integrity(format!(
                "eager runtime parameter {:?} is duplicated",
                parameter.name
            )));
        }
    }
    for (name, (real, imaginary)) in complex {
        let real = real.ok_or_else(|| {
            RusticolError::integrity(format!(
                "eager complex runtime parameter {name:?} lacks a real component"
            ))
        })?;
        if direct
            .insert(
                name.clone(),
                RuntimeLogicalParameterSlots { real, imaginary },
            )
            .is_some()
        {
            return Err(RusticolError::integrity(format!(
                "eager runtime parameter {name:?} has conflicting scalar and complex records"
            )));
        }
    }
    Ok(direct)
}

pub(super) fn prepared_parameter_indices(
    pack: &PreparedKernelPackManifest,
    active_kernel_ids: &BTreeSet<u32>,
    include_derivation_inputs: bool,
) -> RusticolResult<BTreeMap<String, usize>> {
    // Validate the stable parameter namespace across the complete authenticated
    // pack before projecting it to this process.  An inactive kernel must not
    // weaken pack integrity, but neither may it impose runtime dependencies on
    // a process that never invokes it.
    let all_indices = all_prepared_parameter_indices(pack)?;
    let mut selected_names = BTreeSet::new();
    let mut unresolved_active_ids = active_kernel_ids.clone();
    for kernel in &pack.kernels {
        let is_active = active_kernel_ids.contains(&kernel.kernel_id);
        if is_active {
            unresolved_active_ids.remove(&kernel.kernel_id);
        }
        if !is_active && (!include_derivation_inputs || kernel.contract_kind != "model-parameter") {
            continue;
        }
        for input in &kernel.input_contracts {
            if input.role != "model-parameter" {
                continue;
            }
            let name = input.model_parameter_name.as_ref().ok_or_else(|| {
                RusticolError::integrity("prepared model-parameter input lacks its name")
            })?;
            selected_names.insert(name.clone());
        }
    }
    if let Some(kernel_id) = unresolved_active_ids.first() {
        return Err(RusticolError::integrity(format!(
            "active prepared kernel {kernel_id} is absent from the authenticated kernel pack"
        )));
    }
    Ok(all_indices
        .into_iter()
        .filter(|(name, _)| selected_names.contains(name))
        .collect())
}

fn all_prepared_parameter_indices(
    pack: &PreparedKernelPackManifest,
) -> RusticolResult<BTreeMap<String, usize>> {
    let mut by_name = BTreeMap::new();
    let mut by_index = BTreeMap::new();
    for kernel in &pack.kernels {
        for input in &kernel.input_contracts {
            if input.role != "model-parameter" {
                continue;
            }
            let name = input.model_parameter_name.as_ref().ok_or_else(|| {
                RusticolError::integrity("prepared model-parameter input lacks its name")
            })?;
            let index = usize::try_from(input.model_parameter_index.ok_or_else(|| {
                RusticolError::integrity("prepared model-parameter input lacks its stable index")
            })?)
            .map_err(|_| RusticolError::artifact("prepared parameter index does not fit usize"))?;
            if by_name
                .insert(name.clone(), index)
                .is_some_and(|old| old != index)
            {
                return Err(RusticolError::integrity(format!(
                    "prepared parameter {name:?} has conflicting stable indices"
                )));
            }
            if by_index
                .insert(index, name.clone())
                .is_some_and(|old| old != *name)
            {
                return Err(RusticolError::integrity(format!(
                    "prepared parameter index {index} names multiple parameters"
                )));
            }
        }
    }
    Ok(by_name)
}

fn load_prepared_model_parameter_evaluator(
    pack: &PreparedKernelPackManifest,
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
    logical_slots: &BTreeMap<String, RuntimeLogicalParameterSlots>,
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<Option<ModelParameterEvaluatorRuntime>> {
    let mut kernels = pack
        .kernels
        .iter()
        .filter(|kernel| kernel.contract_kind == "model-parameter");
    let kernel = kernels.next();
    if kernels.next().is_some() {
        return Err(RusticolError::integrity(
            "prepared kernel pack contains multiple model-parameter kernels",
        ));
    }
    let derived_names = runtime_parameters
        .iter()
        .filter(|parameter| parameter.kind == "derived_parameter_component")
        .map(|parameter| {
            parameter
                .runtime_name
                .clone()
                .unwrap_or_else(|| parameter.name.clone())
        })
        .collect::<BTreeSet<_>>();
    if derived_names.is_empty() {
        return Ok(None);
    }
    let Some(kernel) = kernel else {
        return Err(RusticolError::integrity(
            "eager runtime schema contains derived parameters but its prepared pack has no derivation kernel",
        ));
    };

    let mut input_parameter_indices = Vec::with_capacity(kernel.input_contracts.len());
    for input in &kernel.input_contracts {
        let name = input.model_parameter_name.as_ref().ok_or_else(|| {
            RusticolError::integrity("prepared derivation input lacks its parameter name")
        })?;
        let slots = logical_slots.get(name).ok_or_else(|| {
            RusticolError::integrity(format!(
                "prepared derivation input {name:?} is absent from the runtime schema"
            ))
        })?;
        if slots.imaginary.is_some() {
            return Err(RusticolError::compatibility(format!(
                "prepared derivation input {name:?} is complex, but runtime schema v3 derivation inputs are scalar"
            )));
        }
        input_parameter_indices.push(slots.real);
    }

    let output_index_by_name = kernel
        .output_layout
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let name = value.strip_prefix("model-parameter:").ok_or_else(|| {
                RusticolError::integrity(format!(
                    "prepared derivation output layout {value:?} is invalid"
                ))
            })?;
            Ok((name.to_string(), index))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    let mut outputs = Vec::with_capacity(derived_names.len());
    for name in derived_names {
        let slots = logical_slots.get(&name).ok_or_else(|| {
            RusticolError::integrity(format!(
                "derived runtime parameter {name:?} has no logical slots"
            ))
        })?;
        let output_index = *output_index_by_name.get(&name).ok_or_else(|| {
            RusticolError::integrity(format!(
                "prepared derivation kernel does not output runtime parameter {name:?}"
            ))
        })?;
        outputs.push(GenericDerivedParameterOutputManifest {
            runtime_name: name,
            output_index,
            real_parameter_index: slots.real,
            imag_parameter_index: slots.imaginary,
        });
    }
    let evaluator_manifest = kernel.runtime_evaluator_manifest()?;
    let evaluator = EvaluatorGroup::load_from_store(&evaluator_manifest, payloads)?;
    Ok(Some(ModelParameterEvaluatorRuntime {
        input_parameter_indices,
        outputs,
        evaluator,
    }))
}

pub(super) fn load_prepared_model_parameter_evaluator_for_runtime(
    pack: &PreparedKernelPackManifest,
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<Option<ModelParameterEvaluatorRuntime>> {
    let logical_slots = eager_runtime_logical_parameter_slots(runtime_parameters)?;
    load_prepared_model_parameter_evaluator(pack, runtime_parameters, &logical_slots, payloads)
}

pub(super) fn validate_prepared_kernel_references(
    artifact: &VerifiedArtifact,
    payload_root: &Path,
    pack: &PreparedKernelPackManifest,
) -> RusticolResult<()> {
    let relative_root = payload_root.strip_prefix(artifact.root()).map_err(|_| {
        RusticolError::security("prepared kernel payload root escapes the verified artifact")
    })?;
    for kernel in &pack.kernels {
        validate_evaluator_state_path(artifact, relative_root, &kernel.exact_evaluator_state_path)?;
        let evaluator = kernel.runtime_evaluator_manifest()?;
        validate_evaluator_reference(artifact, relative_root, &evaluator)?;
        for path in kernel.extra_evaluator_payload_paths()? {
            validate_evaluator_state_path(artifact, relative_root, path)?;
        }
    }
    for variant in &pack.kernel_variants {
        let evaluator = variant.runtime_evaluator_manifest()?;
        validate_evaluator_reference(artifact, relative_root, &evaluator)?;
        for path in variant.extra_evaluator_payload_paths()? {
            validate_evaluator_state_path(artifact, relative_root, path)?;
        }
    }
    Ok(())
}
