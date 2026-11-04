// SPDX-License-Identifier: 0BSD

//! Checked construction of the common f64 state used by eager plan-v3.
//!
//! This adapter deliberately reads the compact typed plan, retained columns,
//! and canonical IR directly. It does not recreate the expanded
//! `runtime_schema` representation.

use super::eager_v3_decode::{
    DecodedEagerPrimitiveColumn, DecodedEagerRetainedTable, DecodedEagerRuntimeV3,
};
use super::eager_v3_manifest::EagerV3ExecutionManifest;
use super::*;
use serde::Deserialize;
use serde_json::{Value, json};

const MISSING_I32: i32 = i32::MIN;

/// Validated fields that differ between eager artifacts while the compiled
/// lane state remains empty.
pub(super) struct EagerV3CommonParts {
    process: String,
    key: String,
    color_accuracy: String,
    external_pdg_order: Vec<i32>,
    external_count: usize,
    parameter_count: usize,
    value_parameter_count: usize,
    momentum_parameter_count: usize,
    current_count: usize,
    source_count: usize,
    interaction_count: usize,
    stage_count: usize,
    amplitude_output_count: usize,
    sources: Vec<GenericSourceRecordManifest>,
    momentum_slots: Vec<GenericMomentumSlotManifest>,
    external_is_initial: Vec<bool>,
    particle_masses: BTreeMap<i32, f64>,
    particle_mass_parameter_names: BTreeMap<i32, String>,
    normalization_factor: f64,
    normalization_color_factor: f64,
    normalization_average_factor: f64,
    normalization_identical_factor: f64,
    normalization_qcd_coupling_power: usize,
    normalization_electroweak_coupling_power: usize,
    lc_topology_replay: Option<LcTopologyReplayManifest>,
    lc_topology_replay_data: LcTopologyReplayData,
    color_topology_replay: Option<ColorTopologyReplayManifest>,
    color_topology_replay_data: LcTopologyReplayData,
    model_parameters: Vec<GenericRuntimeModelParameterManifest>,
    model_parameter_name_to_index: BTreeMap<String, usize>,
    model_parameter_runtime_slots: BTreeMap<String, RuntimeParameterSlots>,
    model_parameter_values_f64: Vec<f64>,
    physics: Arc<PhysicsRuntime>,
}

/// One-line integration entry point for the plan-v3 loader.
pub(super) fn build_eager_v3_common_runtime(
    decoded: &DecodedEagerRuntimeV3,
    manifest: &EagerV3ExecutionManifest,
    physics: ProcessPhysicsV1,
) -> RusticolResult<ExecutionRuntime> {
    EagerV3CommonParts::checked(decoded, manifest, physics).map(EagerV3CommonParts::into_runtime)
}

/// Build the bounded schema fragment required by the lazy Python exact lane.
///
/// This deliberately omits compiled-stage metadata and all high-cardinality
/// proof tables. Those remain authenticated in PACBIN and are exposed through
/// the dedicated exact row bridge only when non-f64 execution is requested.
pub(super) fn build_eager_v3_exact_schema(
    decoded: &DecodedEagerRuntimeV3,
    manifest: &EagerV3ExecutionManifest,
    physics: ProcessPhysicsV1,
    amplitude_stage: Value,
) -> RusticolResult<Value> {
    EagerV3CommonParts::checked(decoded, manifest, physics)?
        .into_exact_schema(decoded, amplitude_stage)
}

impl EagerV3CommonParts {
    fn checked(
        decoded: &DecodedEagerRuntimeV3,
        manifest: &EagerV3ExecutionManifest,
        physics: ProcessPhysicsV1,
    ) -> RusticolResult<Self> {
        validate_identity(decoded, manifest, &physics)?;
        let retained = RetainedTables::new(&decoded.retained_tables);
        let metadata = retained.table("metadata")?;
        let normalization_ir_id = retained.scalar_u32(metadata, "normalization_ir_id")?;
        let normalization: CanonicalNormalization =
            canonical_ir(decoded, normalization_ir_id, "runtime normalization")?;
        normalization.validate()?;
        let lc_topology_replay = build_lc_topology_replay_manifest(decoded, &retained)?;
        let lc_topology_replay_data =
            super::runtime_load::build_lc_topology_replay_data(lc_topology_replay.as_ref())?;
        let color_topology_replay =
            build_color_topology_replay_manifest(decoded, &retained, &manifest.color_accuracy)?;
        let color_topology_replay_data = super::runtime_load::build_color_topology_replay_data(
            color_topology_replay.as_ref(),
            &manifest.color_accuracy,
        )?;

        let model_parameters = build_model_parameters(decoded, &retained)?;
        validate_physics_parameters(&model_parameters, &physics)?;
        let model_parameter_values_f64 = model_parameters
            .iter()
            .map(|parameter| parameter.default)
            .collect::<Vec<_>>();
        let model_parameter_name_to_index = model_parameters
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
            .collect::<BTreeMap<_, _>>();
        if model_parameter_name_to_index.len() != model_parameters.len() {
            return Err(integrity("eager model-parameter names are not unique"));
        }
        let model_parameter_runtime_slots = runtime_parameter_slots(&model_parameters)?;
        let (mut particle_masses, mut particle_mass_parameter_names) =
            direct_particle_masses(&model_parameters)?;

        let momentum_slots = build_momentum_slots(decoded, physics.external_particles.len())?;
        let sources = build_sources(
            decoded,
            &retained,
            &physics,
            &model_parameter_runtime_slots,
            &model_parameter_values_f64,
            &mut particle_masses,
            &mut particle_mass_parameter_names,
        )?;
        let value_parameter_count = component_count(&decoded.values, "value")?;
        let momentum_parameter_count = component_count(&decoded.momenta, "momentum")?;
        let parameter_count = value_parameter_count
            .checked_add(momentum_parameter_count)
            .and_then(|count| count.checked_add(model_parameters.len()))
            .ok_or_else(|| RusticolError::artifact("eager common parameter count overflows"))?;

        // Contracted-color reductions already carry the color contraction.
        let color_factor = if physics.color_accuracy.as_str() == "lc" {
            normalization.color_factor
        } else {
            1.0
        };
        let normalization_factor = color_factor * normalization.global_coupling_factor
            / (normalization.average_factor * normalization.identical_factor);
        if !normalization_factor.is_finite() {
            return Err(integrity("eager runtime normalization is not finite"));
        }
        let physics = Arc::new(PhysicsRuntime::new(physics)?);

        Ok(Self {
            process: manifest.process.clone(),
            key: manifest.key.clone(),
            color_accuracy: manifest.color_accuracy.clone(),
            external_pdg_order: manifest.external_pdg_order.clone(),
            external_count: manifest.external_pdg_order.len(),
            parameter_count,
            value_parameter_count,
            momentum_parameter_count,
            current_count: decoded.currents.len(),
            source_count: decoded.sources.len(),
            interaction_count: usize_count(
                manifest.dag_summary.interaction_count,
                "interaction count",
            )?,
            stage_count: decoded.stages.len(),
            amplitude_output_count: usize::try_from(decoded.dimensions.amplitude_count)
                .map_err(|_| RusticolError::artifact("eager amplitude count exceeds usize"))?,
            sources,
            momentum_slots,
            external_is_initial: physics
                .manifest
                .external_particles
                .iter()
                .map(|particle| particle.role == crate::ParticleRole::Initial)
                .collect(),
            particle_masses,
            particle_mass_parameter_names,
            normalization_factor,
            normalization_color_factor: color_factor,
            normalization_average_factor: normalization.average_factor,
            normalization_identical_factor: normalization.identical_factor,
            normalization_qcd_coupling_power: normalization.qcd_coupling_power,
            normalization_electroweak_coupling_power: normalization.electroweak_coupling_power,
            lc_topology_replay,
            lc_topology_replay_data,
            color_topology_replay,
            color_topology_replay_data,
            model_parameters,
            model_parameter_name_to_index,
            model_parameter_runtime_slots,
            model_parameter_values_f64,
            physics,
        })
    }

    fn into_runtime(self) -> ExecutionRuntime {
        let topology_replay_mappings = self.lc_topology_replay_data.mappings;
        let topology_replay_public_mappings = topology_replay_mappings.clone();
        let color_topology_replay_mappings = self.color_topology_replay_data.mappings;
        ExecutionRuntime {
            process: self.process,
            key: self.key,
            color_accuracy: self.color_accuracy,
            external_pdg_order: self.external_pdg_order,
            external_count: self.external_count,
            parameter_count: self.parameter_count,
            value_parameter_count: self.value_parameter_count,
            momentum_parameter_count: self.momentum_parameter_count,
            current_count: self.current_count,
            source_count: self.source_count,
            interaction_count: self.interaction_count,
            stage_count: self.stage_count,
            amplitude_output_count: self.amplitude_output_count,
            lc_topology_replay_enabled: !topology_replay_mappings.is_empty(),
            lc_topology_replay_mappings: Arc::new(topology_replay_mappings),
            lc_topology_replay_public_mappings: topology_replay_public_mappings,
            lc_topology_replay_routes: self.lc_topology_replay_data.routes,
            lc_topology_replay_materialized_sector_ids: self
                .lc_topology_replay_data
                .materialized_sector_ids,
            lc_resolved_replay_plan: None,
            lc_resolved_replay_selection_cache: None,
            lc_replay_flat_momenta_scratch: Vec::new(),
            lc_replay_target_components_scratch: Vec::new(),
            color_topology_replay_enabled: !color_topology_replay_mappings.is_empty(),
            color_topology_replay_mappings: Arc::new(color_topology_replay_mappings),
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
            sources: self.sources,
            momentum_slots: self.momentum_slots,
            external_is_initial: self.external_is_initial,
            particle_masses: self.particle_masses,
            particle_mass_parameter_names: self.particle_mass_parameter_names,
            normalization_factor: self.normalization_factor,
            normalization_color_factor: self.normalization_color_factor,
            normalization_average_factor: self.normalization_average_factor,
            normalization_identical_factor: self.normalization_identical_factor,
            normalization_qcd_coupling_power: self.normalization_qcd_coupling_power,
            normalization_electroweak_coupling_power: self.normalization_electroweak_coupling_power,
            model_parameters: self.model_parameters,
            model_parameter_name_to_index: self.model_parameter_name_to_index,
            model_parameter_runtime_slots: self.model_parameter_runtime_slots,
            model_parameter_values_f64: self.model_parameter_values_f64,
            model_parameter_evaluator: None,
            physics_reduction_override: None,
            physics: Some(self.physics),
            stages: None,
            amplitude_stage: None,
            state_scratch_f64: Vec::new(),
            state_scratch_f64_requires_clear: false,
            values_scratch_f64: Vec::new(),
        }
    }

    fn into_exact_schema(
        self,
        decoded: &DecodedEagerRuntimeV3,
        amplitude_stage: Value,
    ) -> RusticolResult<Value> {
        let source_current_ids = self
            .sources
            .iter()
            .map(|source| source.current_id)
            .collect::<BTreeSet<_>>();
        let current_component_count = decoded
            .dimensions
            .current_component_counts
            .iter()
            .try_fold(0_usize, |total, count| {
                total
                    .checked_add(*count as usize)
                    .ok_or_else(|| RusticolError::artifact("eager exact current layout overflows"))
            })?;
        let current_slots = decoded
            .currents
            .iter()
            .map(|row| {
                let start = usize_count(row.component_start, "exact current component start")?;
                let dimension = row.component_count as usize;
                Ok(json!({
                    "current_id": row.current_id,
                    "component_start": start,
                    "component_stop": start + dimension,
                    "dimension": dimension,
                    "is_source": source_current_ids.contains(&(row.current_id as usize)),
                }))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let value_slots = decoded
            .values
            .iter()
            .map(|row| {
                let start = usize_count(row.component_start, "exact value component start")?;
                let dimension = row.component_count as usize;
                let variant = match row.kind {
                    crate::eager_lowering_v3::EagerValueSlotKind::Source => "source",
                    crate::eager_lowering_v3::EagerValueSlotKind::Unpropagated => "unpropagated",
                    crate::eager_lowering_v3::EagerValueSlotKind::Propagated => "propagated",
                };
                Ok(json!({
                    "value_slot_id": row.value_slot_id,
                    "current_id": row.current_id,
                    "variant": variant,
                    "component_start": start,
                    "component_stop": start + dimension,
                    "dimension": dimension,
                }))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let external_particles = self
            .physics
            .manifest
            .external_particles
            .iter()
            .map(|particle| {
                json!({
                    "label": particle.label,
                    "index": particle.index,
                    "pdg": particle.pdg,
                    "role": match particle.role {
                        crate::ParticleRole::Initial => "initial",
                        crate::ParticleRole::Final => "final",
                    },
                    "momentum_slot": particle.momentum_slot,
                })
            })
            .collect::<Vec<_>>();
        let model_particles = self
            .particle_masses
            .iter()
            .map(|(pdg, mass)| {
                json!({
                    "pdg": pdg,
                    "mass": mass,
                    "mass_parameter": self.particle_mass_parameter_names.get(pdg),
                })
            })
            .collect::<Vec<_>>();
        let flattened_parameter_count = self
            .value_parameter_count
            .checked_add(self.momentum_parameter_count)
            .and_then(|count| count.checked_add(self.model_parameters.len()))
            .ok_or_else(|| RusticolError::artifact("eager exact parameter layout overflows"))?;

        Ok(json!({
            "kind": "pyamplicol-eager-exact-schema-v1",
            "parameter_layout": {
                "value_component_count": self.value_parameter_count,
                "momentum_parameter_count": self.momentum_parameter_count,
                "model_parameter_count": self.model_parameters.len(),
                "parameter_count_if_flattened": flattened_parameter_count,
            },
            "current_storage": {
                "component_count": current_component_count,
                "current_slots": current_slots,
            },
            "value_storage": {
                "component_count": self.value_parameter_count,
                "value_slots": value_slots,
            },
            "momentum_slots": self.momentum_slots,
            "external_particles": external_particles,
            "model": {"particles": model_particles},
            "model_parameters": self.model_parameters,
            "source_fill": {
                "source_count": self.sources.len(),
                "sources": self.sources,
            },
            "lc_topology_replay": self.lc_topology_replay,
            "color_topology_replay": self.color_topology_replay,
            "amplitude_stage": amplitude_stage,
        }))
    }
}

fn build_lc_topology_replay_manifest(
    decoded: &DecodedEagerRuntimeV3,
    retained: &RetainedTables<'_>,
) -> RusticolResult<Option<LcTopologyReplayManifest>> {
    build_topology_replay_manifest(decoded, retained, 1)
}

fn build_color_topology_replay_manifest(
    decoded: &DecodedEagerRuntimeV3,
    retained: &RetainedTables<'_>,
    color_accuracy: &str,
) -> RusticolResult<Option<ColorTopologyReplayManifest>> {
    let Some(compatibility) = build_topology_replay_manifest(decoded, retained, 2)? else {
        return Ok(None);
    };
    Ok(Some(ColorTopologyReplayManifest {
        enabled: compatibility.enabled,
        mode: compatibility.mode,
        contract_version: Some(3),
        color_accuracy: color_accuracy.to_string(),
        physical_sector_count: compatibility.physical_sector_count,
        replayed_sector_count: compatibility.replayed_sector_count,
        materialized_sector_ids: compatibility.materialized_sector_ids,
        residual_sector_ids: compatibility.residual_sector_ids,
        groups: compatibility.groups,
    }))
}

pub(super) fn load_eager_color_topology_replay_manifests(
    decoded: &DecodedEagerRuntimeV3,
    include_exact_weights: bool,
) -> RusticolResult<
    Option<(
        GenericColorTopologyReplayAmplitudeManifest,
        GenericColorContractionManifest,
    )>,
> {
    let retained = RetainedTables::new(&decoded.retained_tables);
    let metadata = retained.table("color_replay_metadata")?;
    let present = retained.scalar_u8(metadata, "present")?;
    let contract_version = retained.scalar_u32(metadata, "contract_version")?;
    let physical_group_count = usize_count(
        retained.scalar_u64(metadata, "physical_group_count")?,
        "physical group count",
    )?;
    let mapping_count = usize_count(
        retained.scalar_u64(metadata, "mapping_count")?,
        "mapping count",
    )?;
    let groups = retained.table("color_replay_physical_groups")?;
    let mappings = retained.table("color_replay_mappings")?;
    let permutations = retained.table("color_replay_permutations")?;
    let routes = retained.table("color_replay_routes")?;
    let contraction_metadata = retained.table("color_replay_contraction_metadata")?;
    let contraction_component_groups =
        retained.table("color_replay_contraction_component_groups")?;
    let contraction_entries = retained.table("color_replay_contraction_entries")?;
    if present == 0 {
        if contract_version != 0
            || physical_group_count != 0
            || mapping_count != 0
            || groups.row_count != 0
            || mappings.row_count != 0
            || permutations.row_count != 0
            || routes.row_count != 0
            || retained.scalar_u8(contraction_metadata, "present")? != 0
            || retained.scalar_u64(contraction_metadata, "group_count")? != 0
            || retained.scalar_u8(contraction_metadata, "repeated")? != 0
            || retained.scalar_u64(contraction_metadata, "component_count")? != 0
            || contraction_component_groups.row_count != 0
            || contraction_entries.row_count != 0
        {
            return Err(integrity(
                "absent eager color replay contains retained payload rows",
            ));
        }
        return Ok(None);
    }
    let contraction_repeated = retained.scalar_u8(contraction_metadata, "repeated")?;
    let contraction_component_count = usize_count(
        retained.scalar_u64(contraction_metadata, "component_count")?,
        "color replay contraction component count",
    )?;
    if present != 1
        || contract_version != 1
        || physical_group_count == 0
        || groups.row_count as usize != physical_group_count
        || mappings.row_count as usize != mapping_count
        || mapping_count == 0
        || retained.scalar_u8(contraction_metadata, "present")? != 1
        || retained.scalar_u64(contraction_metadata, "group_count")? != physical_group_count as u64
        || retained.scalar_u8(contraction_metadata, "includes_color_factor")? != 1
        || contraction_repeated > 1
        || (contraction_repeated == 0
            && (contraction_component_count != 0 || contraction_component_groups.row_count != 0))
        || (contraction_repeated == 1
            && (contraction_component_count < 2
                || contraction_component_groups.row_count as usize != physical_group_count))
    {
        return Err(integrity(
            "eager color replay amplitude metadata is inconsistent",
        ));
    }

    let group_ids = retained.u32_column(groups, "group_id", 1)?;
    let sector_ids = retained.u32_column(groups, "color_sector_id", 1)?;
    let word_ids = retained.u32_column(groups, "color_word_sequence_id", 1)?;
    let helicity_ids = retained.u32_column(groups, "helicity_values_sequence_id", 1)?;
    let helicity_weight_ids = retained.u32_column(groups, "helicity_weight_factor_id", 1)?;
    let physical_groups = (0..physical_group_count)
        .map(|index| {
            let helicity_weight =
                eager_exact_real_factor(decoded, helicity_weight_ids[index], "helicity weight")?;
            Ok(GenericColorTopologyReplayPhysicalGroupManifest {
                group_id: i64::from(group_ids[index]),
                helicities: i32_sequence(decoded, helicity_ids[index])?,
                color_sector_id: i64::from(sector_ids[index]),
                color_word: u32_sequence(decoded, word_ids[index], "color replay word")?
                    .into_iter()
                    .map(|value| value as usize)
                    .collect(),
                helicity_weight,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let permutation_starts = retained.u64_column(mappings, "permutation_start", 1)?;
    let permutation_counts = retained.u64_column(mappings, "permutation_count", 1)?;
    let route_starts = retained.u64_column(mappings, "route_start", 1)?;
    let route_counts = retained.u64_column(mappings, "route_count", 1)?;
    let representative_labels = retained.u32_column(permutations, "representative_label", 1)?;
    let sector_labels = retained.u32_column(permutations, "sector_label", 1)?;
    let source_group_ids = retained.u32_column(routes, "source_group_id", 1)?;
    let target_group_ids = retained.u32_column(routes, "target_group_id", 1)?;
    let route_factor_ids = retained.u32_column(routes, "factor_id", 1)?;
    let mut permutation_cursor = 0usize;
    let mut route_cursor = 0usize;
    let mut replay_mappings = Vec::with_capacity(mapping_count);
    for index in 0..mapping_count {
        let permutation_start =
            usize_count(permutation_starts[index], "color replay permutation start")?;
        let permutation_count =
            usize_count(permutation_counts[index], "color replay permutation count")?;
        let route_start = usize_count(route_starts[index], "color replay route start")?;
        let route_count = usize_count(route_counts[index], "color replay route count")?;
        if permutation_start != permutation_cursor || route_start != route_cursor {
            return Err(integrity(
                "eager color replay mappings are not canonical and contiguous",
            ));
        }
        let permutation_stop = permutation_start
            .checked_add(permutation_count)
            .ok_or_else(|| RusticolError::artifact("color replay permutation range overflows"))?;
        let route_stop = route_start
            .checked_add(route_count)
            .ok_or_else(|| RusticolError::artifact("color replay route range overflows"))?;
        if permutation_stop > permutations.row_count as usize
            || route_stop > routes.row_count as usize
        {
            return Err(integrity(
                "eager color replay mapping range is out of bounds",
            ));
        }
        replay_mappings.push(GenericColorTopologyReplayMappingManifest {
            label_permutation: (permutation_start..permutation_stop)
                .map(|row| LcTopologyReplayLabelPermutationManifest {
                    representative_label: representative_labels[row] as usize,
                    sector_label: sector_labels[row] as usize,
                })
                .collect(),
            group_routes: (route_start..route_stop)
                .map(|row| {
                    Ok(GenericColorTopologyReplayGroupRouteManifest {
                        source_group_id: i64::from(source_group_ids[row]),
                        target_group_id: i64::from(target_group_ids[row]),
                        factor: eager_exact_factor(decoded, route_factor_ids[row], "route factor")?,
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?,
        });
        permutation_cursor = permutation_stop;
        route_cursor = route_stop;
    }
    if permutation_cursor != permutations.row_count as usize
        || route_cursor != routes.row_count as usize
    {
        return Err(integrity(
            "eager color replay retained tables contain unreachable rows",
        ));
    }

    let left_ids = retained.u32_column(contraction_entries, "left_group_id", 1)?;
    let right_ids = retained.u32_column(contraction_entries, "right_group_id", 1)?;
    let left_indices = retained.u32_column(contraction_entries, "left_group_index", 1)?;
    let right_indices = retained.u32_column(contraction_entries, "right_group_index", 1)?;
    let weight_ids = retained.u32_column(contraction_entries, "weight_factor_id", 1)?;
    let symmetry_ids = retained.u32_column(contraction_entries, "symmetry_factor_id", 1)?;
    if contraction_entries.row_count == 0 {
        return Err(integrity(
            "eager color replay physical contraction is empty",
        ));
    }
    let (entries, repeated_block) = if contraction_repeated == 0 {
        if left_indices
            .iter()
            .chain(right_indices.iter())
            .any(|value| *value != u32::MAX)
        {
            return Err(integrity(
                "expanded eager color replay contraction contains local indices",
            ));
        }
        (
            (0..contraction_entries.row_count as usize)
                .map(|index| {
                    Ok(GenericColorContractionEntryManifest {
                        _exact_weight: if include_exact_weights {
                            eager_exact_color_weight(decoded, weight_ids[index])?
                        } else {
                            None
                        },
                        left_group_id: i64::from(left_ids[index]),
                        right_group_id: i64::from(right_ids[index]),
                        weight: eager_exact_factor(
                            decoded,
                            weight_ids[index],
                            "contraction weight",
                        )?,
                        symmetry_factor: eager_exact_real_factor(
                            decoded,
                            symmetry_ids[index],
                            "contraction symmetry",
                        )?,
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?,
            None,
        )
    } else {
        if left_ids
            .iter()
            .chain(right_ids.iter())
            .any(|value| *value != u32::MAX)
        {
            return Err(integrity(
                "repeated eager color replay contraction contains physical group IDs",
            ));
        }
        let component_group_ids = retained
            .u32_column(contraction_component_groups, "group_id", 1)?
            .iter()
            .map(|value| i64::from(*value))
            .collect();
        let repeated_entries = (0..contraction_entries.row_count as usize)
            .map(|index| {
                Ok(GenericRepeatedColorContractionEntryManifest {
                    _exact_weight: if include_exact_weights {
                        eager_exact_color_weight(decoded, weight_ids[index])?
                    } else {
                        None
                    },
                    left_group_index: left_indices[index] as usize,
                    right_group_index: right_indices[index] as usize,
                    weight: eager_exact_factor(decoded, weight_ids[index], "contraction weight")?,
                    symmetry_factor: eager_exact_real_factor(
                        decoded,
                        symmetry_ids[index],
                        "contraction symmetry",
                    )?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        (
            Vec::new(),
            Some(GenericRepeatedColorContractionBlockManifest {
                component_count: contraction_component_count,
                component_group_ids,
                entries: repeated_entries,
                factorized_block: None,
            }),
        )
    };
    let amplitude_manifest = GenericColorTopologyReplayAmplitudeManifest {
        contract_version,
        physical_group_count,
        physical_groups,
        mappings: replay_mappings,
    };
    let contraction_manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: physical_group_count,
        includes_color_factor: true,
        entries,
        repeated_block,
    };
    Ok(Some((amplitude_manifest, contraction_manifest)))
}

pub(super) fn build_eager_color_topology_replay_amplitude_runtime(
    decoded: &DecodedEagerRuntimeV3,
    materialized_groups: &[RawSumGroup],
    output_count: usize,
) -> RusticolResult<Option<AmplitudeRuntime>> {
    let Some((amplitude_manifest, contraction_manifest)) =
        load_eager_color_topology_replay_manifests(decoded, false)?
    else {
        return Ok(None);
    };
    AmplitudeRuntime::color_topology_replay_reducer(
        output_count,
        materialized_groups,
        &amplitude_manifest,
        &contraction_manifest,
    )
    .map(Some)
}

pub(super) fn eager_exact_color_weight(
    decoded: &DecodedEagerRuntimeV3,
    factor_id: u32,
) -> RusticolResult<Option<[String; 4]>> {
    let factor = decoded
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| integrity("eager exact colour weight is absent"))?;
    if factor.exact_source != 1 {
        return Ok(None);
    }
    let raw = decoded
        .exact_ir
        .get(factor.exact_ir_id as usize)
        .ok_or_else(|| integrity("eager exact colour weight source is absent"))?;
    let payload: serde_json::Value = serde_json::from_str(raw)
        .map_err(|error| integrity(format!("invalid exact colour weight source: {error}")))?;
    let source = &payload["source"];
    if source["kind"] != "rational-complex" {
        return Ok(None);
    }
    serde_json::from_value(source["value"].clone())
        .map(Some)
        .map_err(|error| integrity(format!("invalid rational colour weight: {error}")))
}

fn eager_exact_factor(
    decoded: &DecodedEagerRuntimeV3,
    factor_id: u32,
    context: &str,
) -> RusticolResult<Vec<f64>> {
    let factor = decoded
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| integrity(format!("eager color replay {context} is absent")))?;
    let real = f64::from_bits(factor.real_bits);
    let imaginary = f64::from_bits(factor.imaginary_bits);
    if !real.is_finite() || !imaginary.is_finite() {
        return Err(integrity(format!(
            "eager color replay {context} is not finite"
        )));
    }
    Ok(vec![real, imaginary])
}

fn eager_exact_real_factor(
    decoded: &DecodedEagerRuntimeV3,
    factor_id: u32,
    context: &str,
) -> RusticolResult<f64> {
    let factor = eager_exact_factor(decoded, factor_id, context)?;
    if factor[1] != 0.0 {
        return Err(integrity(format!(
            "eager color replay {context} is not real"
        )));
    }
    Ok(factor[0])
}

fn build_topology_replay_manifest(
    decoded: &DecodedEagerRuntimeV3,
    retained: &RetainedTables<'_>,
    expected_kind: u8,
) -> RusticolResult<Option<LcTopologyReplayManifest>> {
    let metadata = retained.table("lc_replay_metadata")?;
    let present = retained.scalar_u8(metadata, "present")?;
    let replay_kind = retained.scalar_u8(metadata, "replay_kind")?;
    let physical_sequence_id = retained.scalar_u32(metadata, "physical_sector_ids_sequence_id")?;
    let materialized_sequence_id =
        retained.scalar_u32(metadata, "materialized_sector_ids_sequence_id")?;
    let partitions = retained.table("lc_replay_partitions")?;
    let members = retained.table("lc_replay_members")?;
    let permutations = retained.table("lc_replay_permutations")?;
    let residuals = retained.table("lc_replay_residual_sectors")?;
    if present == 0 {
        if replay_kind != 0 {
            return Err(integrity(
                "absent eager topology replay has a nonzero replay kind",
            ));
        }
        if physical_sequence_id != crate::MISSING_U32
            || materialized_sequence_id != crate::MISSING_U32
            || partitions.row_count != 0
            || members.row_count != 0
            || permutations.row_count != 0
            || residuals.row_count != 0
        {
            return Err(integrity(
                "absent eager LC topology replay has retained payload rows",
            ));
        }
        return Ok(None);
    }
    if replay_kind != expected_kind {
        if replay_kind == 1 || replay_kind == 2 {
            return Ok(None);
        }
        return Err(integrity(
            "eager topology replay has an unknown replay kind",
        ));
    }
    if present != 1
        || physical_sequence_id == crate::MISSING_U32
        || materialized_sequence_id == crate::MISSING_U32
    {
        return Err(integrity(
            "eager LC topology replay metadata has an invalid presence contract",
        ));
    }

    let physical_sector_ids = u32_sequence(decoded, physical_sequence_id, "LC physical sectors")?;
    let materialized_sector_ids =
        u32_sequence(decoded, materialized_sequence_id, "LC materialized sectors")?;
    let representative_sector_ids =
        retained.u32_column(partitions, "representative_sector_id", 1)?;
    let partition_materialized_sector_ids =
        retained.u32_column(partitions, "materialized_sector_id", 1)?;
    let member_starts = retained.u64_column(partitions, "member_start", 1)?;
    let member_counts = retained.u64_column(partitions, "member_count", 1)?;
    let proof_algorithm_ids = retained.u32_column(partitions, "proof_algorithm_string_id", 1)?;
    let proof_digest_ids = retained.u32_column(partitions, "proof_digest_string_id", 1)?;

    let member_sector_ids = retained.u32_column(members, "sector_id", 1)?;
    let member_factor_ids = retained.u32_column(members, "factor_id", 1)?;
    let member_signs = retained.i32_column(members, "sign", 1)?;
    let permutation_starts = retained.u64_column(members, "permutation_start", 1)?;
    let permutation_counts = retained.u64_column(members, "permutation_count", 1)?;
    let representative_labels = retained.u32_column(permutations, "representative_label", 1)?;
    let sector_labels = retained.u32_column(permutations, "sector_label", 1)?;
    let residual_sector_ids = retained.u32_column(residuals, "sector_id", 1)?;

    let mut groups = Vec::with_capacity(partitions.row_count as usize);
    let mut member_cursor = 0_usize;
    let mut permutation_cursor = 0_usize;
    for partition_index in 0..partitions.row_count as usize {
        let member_start = usize_count(member_starts[partition_index], "LC replay member start")?;
        let member_count = usize_count(member_counts[partition_index], "LC replay member count")?;
        if member_start != member_cursor {
            return Err(integrity(
                "eager LC replay partition members are not canonical and contiguous",
            ));
        }
        let member_stop = member_start
            .checked_add(member_count)
            .ok_or_else(|| RusticolError::artifact("LC replay member range overflows"))?;
        if member_stop > members.row_count as usize {
            return Err(integrity("eager LC replay member range is out of bounds"));
        }
        let mut active_sector_ids = Vec::with_capacity(member_count);
        let mut sector_permutations = Vec::with_capacity(member_count);
        for member_index in member_start..member_stop {
            let permutation_start = usize_count(
                permutation_starts[member_index],
                "LC replay permutation start",
            )?;
            let permutation_count = usize_count(
                permutation_counts[member_index],
                "LC replay permutation count",
            )?;
            if permutation_start != permutation_cursor {
                return Err(integrity(
                    "eager LC replay permutations are not canonical and contiguous",
                ));
            }
            let permutation_stop = permutation_start
                .checked_add(permutation_count)
                .ok_or_else(|| RusticolError::artifact("LC replay permutation range overflows"))?;
            if permutation_stop > permutations.row_count as usize {
                return Err(integrity(
                    "eager LC replay permutation range is out of bounds",
                ));
            }
            let sign = i8::try_from(member_signs[member_index])
                .map_err(|_| integrity("eager LC replay sign does not fit i8"))?;
            let factor = decoded
                .exact_factors
                .get(member_factor_ids[member_index] as usize)
                .ok_or_else(|| integrity("eager LC replay factor is absent"))?;
            let amplitude_factor = [
                f64::from_bits(factor.real_bits),
                f64::from_bits(factor.imaginary_bits),
            ];
            let weight = amplitude_factor[0] * f64::from(sign);
            let label_permutation = (permutation_start..permutation_stop)
                .map(|index| LcTopologyReplayLabelPermutationManifest {
                    representative_label: representative_labels[index] as usize,
                    sector_label: sector_labels[index] as usize,
                })
                .collect();
            let sector_id = i64::from(member_sector_ids[member_index]);
            active_sector_ids.push(sector_id);
            sector_permutations.push(LcTopologyReplaySectorPermutationManifest {
                sector_id,
                weight,
                sign,
                factor: Some(amplitude_factor.to_vec()),
                label_permutation,
            });
            permutation_cursor = permutation_stop;
        }
        groups.push(LcTopologyReplayGroupManifest {
            representative_sector_id: i64::from(representative_sector_ids[partition_index]),
            materialized_sector_id: i64::from(partition_materialized_sector_ids[partition_index]),
            active_sector_ids,
            proof: Some(LcTopologyReplayProofManifest {
                status: "proven".to_string(),
                algorithm: Some(
                    string(
                        decoded,
                        proof_algorithm_ids[partition_index],
                        "LC replay proof algorithm",
                    )?
                    .to_string(),
                ),
                digest: Some(
                    string(
                        decoded,
                        proof_digest_ids[partition_index],
                        "LC replay proof digest",
                    )?
                    .to_string(),
                ),
            }),
            sector_permutations,
        });
        member_cursor = member_stop;
    }
    if member_cursor != members.row_count as usize
        || permutation_cursor != permutations.row_count as usize
    {
        return Err(integrity(
            "eager LC replay retained tables contain unreachable rows",
        ));
    }

    Ok(Some(LcTopologyReplayManifest {
        enabled: true,
        mode: "external-label-permutation".to_string(),
        contract_version: Some(2),
        physical_sector_count: Some(physical_sector_ids.len()),
        replayed_sector_count: members.row_count as usize,
        materialized_sector_ids: materialized_sector_ids.into_iter().map(i64::from).collect(),
        residual_sector_ids: residual_sector_ids.iter().copied().map(i64::from).collect(),
        groups,
    }))
}

fn validate_identity(
    decoded: &DecodedEagerRuntimeV3,
    manifest: &EagerV3ExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<()> {
    physics.validate()?;
    if decoded.process_key.as_ref() != manifest.key
        || physics.process_id != manifest.key
        || physics.process != manifest.process
        || physics.color_accuracy.as_str() != manifest.color_accuracy
        || physics.external_particles.len() != manifest.external_pdg_order.len()
        || physics
            .external_particles
            .iter()
            .map(|particle| particle.pdg)
            .ne(manifest.external_pdg_order.iter().copied())
    {
        return Err(integrity(
            "eager plan-v3 identity does not match its manifest and resolved physics",
        ));
    }
    if decoded.currents.len() != usize_count(manifest.dag_summary.current_count, "current count")?
        || decoded.sources.len() != usize_count(manifest.dag_summary.source_count, "source count")?
        || decoded.closures.is_empty()
    {
        return Err(integrity(
            "eager plan-v3 decoded counts do not match its DAG summary",
        ));
    }
    Ok(())
}

fn build_model_parameters(
    decoded: &DecodedEagerRuntimeV3,
    retained: &RetainedTables<'_>,
) -> RusticolResult<Vec<GenericRuntimeModelParameterManifest>> {
    let table = retained.table("model_parameters")?;
    let pdgs = retained.i32_column(table, "pdg", 1)?;
    if pdgs.len() != decoded.parameters.len() {
        return Err(integrity(
            "retained model-parameter PDGs have the wrong row count",
        ));
    }
    decoded
        .parameters
        .iter()
        .zip(pdgs)
        .map(|(row, pdg)| {
            let index = usize::try_from(row.parameter_id)
                .map_err(|_| RusticolError::artifact("parameter ID exceeds usize"))?;
            let factor = decoded
                .exact_factors
                .get(row.default_factor_id as usize)
                .ok_or_else(|| integrity("parameter default factor is absent"))?;
            let default = f64::from_bits(factor.real_bits);
            let imaginary = f64::from_bits(factor.imaginary_bits);
            if !default.is_finite() || imaginary != 0.0 {
                return Err(integrity(
                    "eager model-parameter default is not finite and real",
                ));
            }
            Ok(GenericRuntimeModelParameterManifest {
                name: string(decoded, row.name_string_id, "parameter name")?.to_owned(),
                kind: string(decoded, row.kind_string_id, "parameter kind")?.to_owned(),
                parameter_index: index,
                default,
                pdg: (*pdg != MISSING_I32).then_some(*pdg),
                runtime_name: optional_string(decoded, row.runtime_name_string_id)?
                    .map(str::to_owned),
                complex_component: match row.complex_component {
                    -1 => None,
                    0 => Some("real".to_owned()),
                    1 => Some("imag".to_owned()),
                    _ => return Err(integrity("invalid eager parameter complex component")),
                },
            })
        })
        .collect()
}

fn runtime_parameter_slots(
    parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<BTreeMap<String, RuntimeParameterSlots>> {
    let mut result = BTreeMap::new();
    let mut complex = BTreeMap::<String, (Option<usize>, Option<usize>)>::new();
    for parameter in parameters {
        if let Some(name) = &parameter.runtime_name {
            let slots = complex.entry(name.clone()).or_default();
            let slot = match parameter.complex_component.as_deref() {
                Some("real") => &mut slots.0,
                Some("imag") => &mut slots.1,
                _ => {
                    return Err(integrity(
                        "complex eager parameter lacks component metadata",
                    ));
                }
            };
            if slot.replace(parameter.parameter_index).is_some() {
                return Err(integrity("duplicate eager complex parameter component"));
            }
        } else if result
            .insert(
                parameter.name.clone(),
                RuntimeParameterSlots {
                    real: parameter.parameter_index,
                    imaginary: None,
                },
            )
            .is_some()
        {
            return Err(integrity("duplicate eager scalar parameter"));
        }
    }
    for (name, (real, imaginary)) in complex {
        let real = real.ok_or_else(|| integrity("complex eager parameter lacks a real slot"))?;
        if result
            .insert(name, RuntimeParameterSlots { real, imaginary })
            .is_some()
        {
            return Err(integrity("conflicting eager logical parameter"));
        }
    }
    Ok(result)
}

fn direct_particle_masses(
    parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<(BTreeMap<i32, f64>, BTreeMap<i32, String>)> {
    let mut masses = BTreeMap::new();
    let mut names = BTreeMap::new();
    for parameter in parameters {
        if parameter.kind != "particle_mass" {
            continue;
        }
        let pdg = parameter
            .pdg
            .ok_or_else(|| integrity("particle-mass parameter lacks a retained PDG"))?;
        if parameter.default < 0.0
            || masses.insert(pdg, parameter.default).is_some()
            || names.insert(pdg, parameter.name.clone()).is_some()
        {
            return Err(integrity("invalid or duplicate particle-mass parameter"));
        }
    }
    Ok((masses, names))
}

#[allow(clippy::too_many_arguments)]
fn build_sources(
    decoded: &DecodedEagerRuntimeV3,
    retained: &RetainedTables<'_>,
    physics: &ProcessPhysicsV1,
    runtime_slots: &BTreeMap<String, RuntimeParameterSlots>,
    parameter_values: &[f64],
    particle_masses: &mut BTreeMap<i32, f64>,
    mass_names: &mut BTreeMap<i32, String>,
) -> RusticolResult<Vec<GenericSourceRecordManifest>> {
    let currents = retained.table("currents")?;
    let particle_ids = retained.i32_column(currents, "particle_id", 1)?;
    let source_labels = retained.u32_column(currents, "source_leg_label", 1)?;
    let source_helicities = retained.i32_column(currents, "source_helicity", 1)?;
    let chiralities = retained.i32_column(currents, "chirality", 1)?;
    let spin_state_ids = retained.u32_column(currents, "spin_state_sequence_id", 1)?;
    let ancestry_ids = retained.u32_column(currents, "helicity_ancestry_bitset_id", 1)?;
    for column in [
        particle_ids.len(),
        source_labels.len(),
        source_helicities.len(),
        chiralities.len(),
        spin_state_ids.len(),
        ancestry_ids.len(),
    ] {
        if column != decoded.currents.len() {
            return Err(integrity(
                "retained current metadata has the wrong row count",
            ));
        }
    }

    let mut source_parameter_start = 0usize;
    let mut source_ir_by_momentum_slot = BTreeMap::new();
    let mut source_ir_by_canonical_id = BTreeMap::new();
    decoded
        .sources
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let current_index = row.current_id as usize;
            let current = decoded
                .currents
                .get(current_index)
                .ok_or_else(|| integrity("eager source current is absent"))?;
            let value = decoded
                .values
                .get(row.value_slot_id as usize)
                .ok_or_else(|| integrity("eager source value slot is absent"))?;
            let particle = physics
                .external_particles
                .get(row.input_momentum_slot as usize)
                .ok_or_else(|| integrity("eager source external particle is absent"))?;
            let source_ir: GenericSourceIrManifest =
                canonical_ir(decoded, row.source_ir_id, "source IR")?;
            let applied_crossing: GenericCrossingIrManifest =
                canonical_ir(decoded, row.crossing_ir_id, "source crossing IR")?;
            let declared = source_ir
                .states
                .get(row.declared_state_index as usize)
                .ok_or_else(|| integrity("eager source declared state is absent"))?;
            let state = declared
                .transformed(&applied_crossing)
                .map_err(|message| integrity(format!("invalid eager source state: {message}")))?;
            let crossing_factor = decoded
                .exact_factors
                .get(row.crossing_factor_id as usize)
                .ok_or_else(|| integrity("eager source crossing factor is absent"))?;
            if crossing_factor.real_bits != applied_crossing.phase[0].to_bits()
                || crossing_factor.imaginary_bits != applied_crossing.phase[1].to_bits()
            {
                return Err(integrity(
                    "eager source crossing factor disagrees with its IR",
                ));
            }
            let expected_outgoing = match particle.role {
                crate::ParticleRole::Initial => source_ir.identity.anti_pdg_label,
                crate::ParticleRole::Final => source_ir.identity.pdg_label,
            };
            if particle.label != row.external_label as usize
                || particle.momentum_slot != row.input_momentum_slot as usize
                || particle.pdg != expected_outgoing
                || particle_ids[current_index] != source_ir.identity.pdg_label
                || source_labels[current_index] != row.external_label
                || source_helicities[current_index] != state.helicity
                || chiralities[current_index] != state.chirality
                || current.component_count as usize != source_ir.component_dimension
                || value.component_count != current.component_count
                || value.current_id != row.current_id
            {
                return Err(integrity(
                    "eager source metadata is internally inconsistent",
                ));
            }
            let retained_spin = i32_sequence(decoded, spin_state_ids[current_index])?;
            let expected_spin = match &state.spin_state {
                GenericSourceSpinStateManifest::Scalar(value) => vec![*value],
                GenericSourceSpinStateManifest::Components(values) => values.clone(),
            };
            if retained_spin != expected_spin {
                return Err(integrity(
                    "eager source spin state disagrees with retained metadata",
                ));
            }
            bind_source_mass(
                &source_ir,
                runtime_slots,
                parameter_values,
                particle_masses,
                mass_names,
            )?;
            let dimension = source_ir.component_dimension;
            let current_start = usize_count(current.component_start, "source current start")?;
            let value_start = usize_count(value.component_start, "source value start")?;
            let source_parameter_stop = source_parameter_start
                .checked_add(dimension)
                .ok_or_else(|| RusticolError::artifact("source parameter layout overflows"))?;
            let side = match particle.role {
                crate::ParticleRole::Initial => "initial",
                crate::ParticleRole::Final => "final",
            };
            let source = GenericSourceRecordManifest {
                source_id: index,
                current_id: current_index,
                current_component_start: current_start,
                current_component_stop: current_start + dimension,
                value_slot: GenericValueSlotRefManifest {
                    value_slot_id: row.value_slot_id as usize,
                    current_id: current_index,
                    variant: "source".to_owned(),
                    component_start: value_start,
                    component_stop: value_start + dimension,
                    dimension,
                },
                source_parameter_start,
                source_parameter_stop,
                leg_label: row.external_label as usize,
                input_momentum_slot: row.input_momentum_slot as usize,
                side: side.to_owned(),
                crossing: applied_crossing
                    .momentum_transform
                    .legacy_projection()
                    .to_owned(),
                physical_pdg: particle.pdg,
                outgoing_pdg: source_ir.identity.pdg_label,
                particle_id: source_ir.identity.pdg_label,
                anti_particle_id: source_ir.identity.anti_pdg_label,
                source_kind: "external-wavefunction".to_owned(),
                wavefunction_kind: source_ir.wavefunction_family.as_str().to_owned(),
                source_orientation: source_ir.identity.orientation,
                source_basis: source_ir.basis.clone(),
                source_ir,
                applied_crossing,
                source_helicity: state.helicity,
                chirality: state.chirality,
                spin_state: spin_state_value(&state.spin_state),
                dimension,
                helicity_ancestry: Value::String(bitset_decimal(
                    decoded,
                    ancestry_ids[current_index],
                )?),
                color_state: json!({"accuracy": manifest_color_accuracy(physics)}),
            };
            super::validation::validate_source_wavefunction_metadata(index, &source)?;
            super::validation::validate_consistent_source_ir(
                &mut source_ir_by_momentum_slot,
                &mut source_ir_by_canonical_id,
                index,
                &source,
            )?;
            source_parameter_start = source_parameter_stop;
            Ok(source)
        })
        .collect()
}

fn bind_source_mass(
    source_ir: &GenericSourceIrManifest,
    runtime_slots: &BTreeMap<String, RuntimeParameterSlots>,
    parameter_values: &[f64],
    particle_masses: &mut BTreeMap<i32, f64>,
    mass_names: &mut BTreeMap<i32, String>,
) -> RusticolResult<()> {
    let Some(name) = source_ir.mass_parameter.as_ref() else {
        return Ok(());
    };
    let slots = runtime_slots.get(name).ok_or_else(|| {
        integrity(format!(
            "massive eager source references unbound mass parameter {name:?}"
        ))
    })?;
    if slots
        .imaginary
        .and_then(|index| parameter_values.get(index))
        .is_some_and(|value| *value != 0.0)
    {
        return Err(integrity(
            "eager source mass parameter has a nonzero imaginary component",
        ));
    }
    let mass = *parameter_values
        .get(slots.real)
        .ok_or_else(|| integrity("eager source mass parameter slot is absent"))?;
    if !mass.is_finite() || mass < 0.0 {
        return Err(integrity("eager source mass parameter is invalid"));
    }
    for pdg in [
        source_ir.identity.pdg_label,
        source_ir.identity.anti_pdg_label,
    ] {
        if particle_masses
            .insert(pdg, mass)
            .is_some_and(|previous| previous.to_bits() != mass.to_bits())
            || mass_names
                .insert(pdg, name.clone())
                .is_some_and(|previous| previous != *name)
        {
            return Err(integrity("eager source has conflicting mass bindings"));
        }
    }
    Ok(())
}

fn build_momentum_slots(
    decoded: &DecodedEagerRuntimeV3,
    external_count: usize,
) -> RusticolResult<Vec<GenericMomentumSlotManifest>> {
    decoded
        .momenta
        .iter()
        .map(|row| {
            if row.component_count != 4 {
                return Err(integrity(
                    "eager momentum slot does not contain four components",
                ));
            }
            let external_labels = bitset_labels(decoded, row.bitset_id, external_count)?;
            if external_labels.is_empty() {
                return Err(integrity("eager momentum slot has an empty external mask"));
            }
            let start = usize_count(row.component_start, "momentum component start")?;
            Ok(GenericMomentumSlotManifest {
                momentum_slot_id: row.momentum_slot_id as usize,
                momentum_mask: bitset_u64(decoded, row.bitset_id)?,
                external_labels,
                component_start: start,
                component_stop: start + 4,
                real_valued: true,
            })
        })
        .collect()
}

fn validate_physics_parameters(
    runtime: &[GenericRuntimeModelParameterManifest],
    physics: &ProcessPhysicsV1,
) -> RusticolResult<()> {
    let mut projected = BTreeMap::<String, (crate::ParameterKind, f64, f64)>::new();
    for parameter in runtime {
        let name = parameter
            .runtime_name
            .as_ref()
            .unwrap_or(&parameter.name)
            .clone();
        let kind = public_parameter_kind(&parameter.kind)?;
        let entry = projected.entry(name).or_insert((kind, 0.0, 0.0));
        if entry.0 != kind {
            return Err(integrity("eager logical parameter has inconsistent kinds"));
        }
        match parameter.complex_component.as_deref() {
            Some("imag") => entry.2 = parameter.default,
            _ => entry.1 = parameter.default,
        }
    }
    if projected.len() != physics.model_parameters.len() {
        return Err(integrity(
            "eager runtime and resolved physics parameter counts disagree",
        ));
    }
    for parameter in &physics.model_parameters {
        let Some((kind, real, imaginary)) = projected.get(&parameter.name) else {
            return Err(integrity(
                "resolved physics parameter is absent from eager runtime",
            ));
        };
        if *kind != parameter.kind
            || real.to_bits() != parameter.default_real.to_bits()
            || imaginary.to_bits() != parameter.default_imaginary.to_bits()
        {
            return Err(integrity(format!(
                "eager runtime parameter {:?} disagrees with resolved physics: \
                 runtime kind={kind:?} defaults=({:#018x},{:#018x}), \
                 physics kind={:?} defaults=({:#018x},{:#018x})",
                parameter.name,
                real.to_bits(),
                imaginary.to_bits(),
                parameter.kind,
                parameter.default_real.to_bits(),
                parameter.default_imaginary.to_bits(),
            )));
        }
    }
    Ok(())
}

fn public_parameter_kind(kind: &str) -> RusticolResult<crate::ParameterKind> {
    match kind {
        "normalization" => Ok(crate::ParameterKind::Normalization),
        "particle_mass" => Ok(crate::ParameterKind::Mass),
        "particle_width" => Ok(crate::ParameterKind::Width),
        "coupling_component" => Ok(crate::ParameterKind::Coupling),
        "external_parameter" | "external_parameter_component" => Ok(crate::ParameterKind::External),
        "derived_parameter_component" => Ok(crate::ParameterKind::Derived),
        _ => Err(RusticolError::compatibility(format!(
            "unsupported eager model-parameter kind {kind:?}"
        ))),
    }
}

#[derive(Deserialize)]
struct CanonicalNormalization {
    #[serde(default = "one")]
    color_factor: f64,
    #[serde(default = "one")]
    global_coupling_factor: f64,
    #[serde(default = "one")]
    average_factor: f64,
    #[serde(default = "one")]
    identical_factor: f64,
    #[serde(default)]
    qcd_coupling_power: usize,
    #[serde(default)]
    electroweak_coupling_power: usize,
}

impl CanonicalNormalization {
    fn validate(&self) -> RusticolResult<()> {
        if !self.color_factor.is_finite()
            || !self.global_coupling_factor.is_finite()
            || !self.average_factor.is_finite()
            || !self.identical_factor.is_finite()
            || self.average_factor <= 0.0
            || self.identical_factor <= 0.0
        {
            return Err(integrity("invalid eager runtime normalization IR"));
        }
        Ok(())
    }
}

fn one() -> f64 {
    1.0
}

struct RetainedTables<'a> {
    tables: &'a [DecodedEagerRetainedTable],
}

impl<'a> RetainedTables<'a> {
    fn new(tables: &'a [DecodedEagerRetainedTable]) -> Self {
        Self { tables }
    }

    fn table(&self, name: &str) -> RusticolResult<&'a DecodedEagerRetainedTable> {
        let mut matches = self
            .tables
            .iter()
            .filter(|table| table.name.as_ref() == name);
        let table = matches
            .next()
            .ok_or_else(|| integrity(format!("retained eager table {name:?} is absent")))?;
        if matches.next().is_some() {
            return Err(integrity(format!(
                "retained eager table {name:?} is duplicated"
            )));
        }
        Ok(table)
    }

    fn scalar_u32(&self, table: &DecodedEagerRetainedTable, name: &str) -> RusticolResult<u32> {
        let values = self.u32_column(table, name, 1)?;
        if table.row_count != 1 || values.len() != 1 {
            return Err(integrity(format!(
                "retained eager scalar {}.{name} has the wrong shape",
                table.name
            )));
        }
        Ok(values[0])
    }

    fn scalar_u8(&self, table: &DecodedEagerRetainedTable, name: &str) -> RusticolResult<u8> {
        let values = self.u8_column(table, name, 1)?;
        if table.row_count != 1 || values.len() != 1 {
            return Err(integrity(format!(
                "retained eager scalar {}.{name} has the wrong shape",
                table.name
            )));
        }
        Ok(values[0])
    }

    fn scalar_u64(&self, table: &DecodedEagerRetainedTable, name: &str) -> RusticolResult<u64> {
        let values = self.u64_column(table, name, 1)?;
        if table.row_count != 1 || values.len() != 1 {
            return Err(integrity(format!(
                "retained eager scalar {}.{name} has the wrong shape",
                table.name
            )));
        }
        Ok(values[0])
    }

    fn u8_column(
        &self,
        table: &'a DecodedEagerRetainedTable,
        name: &str,
        elements_per_row: u32,
    ) -> RusticolResult<&'a [u8]> {
        let column = retained_column(table, name, elements_per_row)?;
        match &column.values {
            DecodedEagerPrimitiveColumn::U8(values) => Ok(values),
            _ => Err(integrity(format!(
                "retained eager column {}.{name} is not u8",
                table.name
            ))),
        }
    }

    fn u32_column(
        &self,
        table: &'a DecodedEagerRetainedTable,
        name: &str,
        elements_per_row: u32,
    ) -> RusticolResult<&'a [u32]> {
        let column = retained_column(table, name, elements_per_row)?;
        match &column.values {
            DecodedEagerPrimitiveColumn::U32(values) => Ok(values),
            _ => Err(integrity(format!(
                "retained eager column {}.{name} is not u32",
                table.name
            ))),
        }
    }

    fn i32_column(
        &self,
        table: &'a DecodedEagerRetainedTable,
        name: &str,
        elements_per_row: u32,
    ) -> RusticolResult<&'a [i32]> {
        let column = retained_column(table, name, elements_per_row)?;
        match &column.values {
            DecodedEagerPrimitiveColumn::I32(values) => Ok(values),
            _ => Err(integrity(format!(
                "retained eager column {}.{name} is not i32",
                table.name
            ))),
        }
    }

    fn u64_column(
        &self,
        table: &'a DecodedEagerRetainedTable,
        name: &str,
        elements_per_row: u32,
    ) -> RusticolResult<&'a [u64]> {
        let column = retained_column(table, name, elements_per_row)?;
        match &column.values {
            DecodedEagerPrimitiveColumn::U64(values) => Ok(values),
            _ => Err(integrity(format!(
                "retained eager column {}.{name} is not u64",
                table.name
            ))),
        }
    }
}

fn retained_column<'a>(
    table: &'a DecodedEagerRetainedTable,
    name: &str,
    elements_per_row: u32,
) -> RusticolResult<&'a super::eager_v3_decode::DecodedEagerRetainedColumn> {
    let mut matches = table
        .columns
        .iter()
        .filter(|column| column.name.as_ref() == name);
    let column = matches.next().ok_or_else(|| {
        integrity(format!(
            "retained eager column {}.{name} is absent",
            table.name
        ))
    })?;
    if matches.next().is_some() || column.elements_per_row != elements_per_row {
        return Err(integrity(format!(
            "retained eager column {}.{name} has a non-canonical shape",
            table.name
        )));
    }
    Ok(column)
}

fn canonical_ir<T: for<'de> Deserialize<'de>>(
    decoded: &DecodedEagerRuntimeV3,
    id: u32,
    context: &str,
) -> RusticolResult<T> {
    let payload = decoded
        .exact_ir
        .get(id as usize)
        .ok_or_else(|| integrity(format!("{context} references absent canonical IR {id}")))?;
    serde_json::from_str(payload).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse eager {context} canonical IR {id}: {error}"
        ))
    })
}

fn string<'a>(
    decoded: &'a DecodedEagerRuntimeV3,
    id: u32,
    context: &str,
) -> RusticolResult<&'a str> {
    decoded
        .strings
        .get(id as usize)
        .map(AsRef::as_ref)
        .ok_or_else(|| integrity(format!("{context} string {id} is absent")))
}

fn optional_string(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<Option<&str>> {
    if id == crate::MISSING_U32 {
        Ok(None)
    } else {
        string(decoded, id, "optional parameter").map(Some)
    }
}

fn i32_sequence(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<Vec<i32>> {
    let range = decoded
        .i32_sequence_ranges
        .get(id as usize)
        .ok_or_else(|| integrity("retained current references absent i32 sequence"))?;
    checked_catalog_range(
        &decoded.i32_sequence_values,
        range.start,
        range.count,
        "i32 sequence",
    )
    .map(<[i32]>::to_vec)
}

fn u32_sequence(
    decoded: &DecodedEagerRuntimeV3,
    id: u32,
    context: &str,
) -> RusticolResult<Vec<u32>> {
    let range = decoded
        .u32_sequence_ranges
        .get(id as usize)
        .ok_or_else(|| integrity(format!("eager metadata references absent {context}")))?;
    checked_catalog_range(
        &decoded.u32_sequence_values,
        range.start,
        range.count,
        context,
    )
    .map(<[u32]>::to_vec)
}

fn bitset_words(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<&[u64]> {
    let range = decoded
        .bitset_ranges
        .get(id as usize)
        .ok_or_else(|| integrity("eager metadata references absent bitset"))?;
    checked_catalog_range(&decoded.bitset_words, range.start, range.count, "bitset")
}

fn bitset_u64(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<u64> {
    let words = bitset_words(decoded, id)?;
    if words.iter().skip(1).any(|word| *word != 0) {
        return Err(RusticolError::compatibility(
            "eager momentum metadata cannot project a bitset wider than 64 bits",
        ));
    }
    Ok(words.first().copied().unwrap_or(0))
}

fn bitset_decimal(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<String> {
    Ok(bitset_words_decimal(bitset_words(decoded, id)?))
}

fn bitset_words_decimal(words: &[u64]) -> String {
    const BASE: u128 = 1_000_000_000;
    const WORD_RADIX: u128 = 1_u128 << 64;

    let mut decimal_limbs = vec![0_u32];
    for word in words.iter().rev().copied() {
        let mut carry = u128::from(word);
        for limb in &mut decimal_limbs {
            let value = u128::from(*limb) * WORD_RADIX + carry;
            *limb = (value % BASE) as u32;
            carry = value / BASE;
        }
        while carry != 0 {
            decimal_limbs.push((carry % BASE) as u32);
            carry /= BASE;
        }
    }

    let mut limbs = decimal_limbs.iter().rev();
    let mut result = limbs.next().copied().unwrap_or(0).to_string();
    for limb in limbs {
        use std::fmt::Write as _;
        write!(&mut result, "{limb:09}").expect("writing to a string cannot fail");
    }
    result
}

fn bitset_labels(
    decoded: &DecodedEagerRuntimeV3,
    id: u32,
    external_count: usize,
) -> RusticolResult<Vec<usize>> {
    let mut labels = Vec::new();
    for (word_index, word) in bitset_words(decoded, id)?.iter().copied().enumerate() {
        for bit in 0..64 {
            if word & (1_u64 << bit) == 0 {
                continue;
            }
            let index = word_index
                .checked_mul(64)
                .and_then(|start| start.checked_add(bit))
                .ok_or_else(|| RusticolError::artifact("eager momentum label overflows"))?;
            if index >= external_count {
                return Err(integrity(
                    "eager momentum mask exceeds the external-particle count",
                ));
            }
            labels.push(index + 1);
        }
    }
    Ok(labels)
}

fn checked_catalog_range<'a, T>(
    values: &'a [T],
    start: u64,
    count: u64,
    context: &str,
) -> RusticolResult<&'a [T]> {
    let start = usize_count(start, context)?;
    let count = usize_count(count, context)?;
    let stop = start
        .checked_add(count)
        .ok_or_else(|| RusticolError::artifact(format!("eager {context} range overflows")))?;
    values
        .get(start..stop)
        .ok_or_else(|| integrity(format!("eager {context} range is out of bounds")))
}

fn component_count<T>(rows: &[T], context: &str) -> RusticolResult<usize>
where
    T: ComponentRow,
{
    rows.iter().try_fold(0usize, |total, row| {
        total
            .checked_add(row.component_count() as usize)
            .ok_or_else(|| RusticolError::artifact(format!("eager {context} layout overflows")))
    })
}

trait ComponentRow {
    fn component_count(&self) -> u32;
}

impl ComponentRow for crate::eager_lowering_v3::EagerPlanValueRow {
    fn component_count(&self) -> u32 {
        self.component_count
    }
}

impl ComponentRow for crate::eager_lowering_v3::EagerPlanMomentumRow {
    fn component_count(&self) -> u32 {
        self.component_count
    }
}

fn spin_state_value(state: &GenericSourceSpinStateManifest) -> Value {
    match state {
        GenericSourceSpinStateManifest::Scalar(value) => Value::from(*value),
        GenericSourceSpinStateManifest::Components(values) => json!(values),
    }
}

fn manifest_color_accuracy(physics: &ProcessPhysicsV1) -> &'static str {
    match physics.color_accuracy {
        crate::ColorAccuracy::Lc => "lc",
        crate::ColorAccuracy::Nlc => "nlc",
        crate::ColorAccuracy::Full => "full",
    }
}

fn usize_count(value: u64, context: &str) -> RusticolResult<usize> {
    usize::try_from(value)
        .map_err(|_| RusticolError::artifact(format!("eager {context} exceeds usize")))
}

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(message.into())
}

#[cfg(test)]
mod tests {
    use super::bitset_words_decimal;

    #[test]
    fn arbitrary_width_bitsets_project_to_decimal_strings() {
        assert_eq!(bitset_words_decimal(&[]), "0");
        assert_eq!(bitset_words_decimal(&[1]), "1");
        assert_eq!(bitset_words_decimal(&[u64::MAX]), "18446744073709551615");
        assert_eq!(bitset_words_decimal(&[0, 1]), "18446744073709551616");
        assert_eq!(bitset_words_decimal(&[1, 2]), "36893488147419103233");
    }
}
