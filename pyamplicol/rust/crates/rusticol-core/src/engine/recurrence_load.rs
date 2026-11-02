// SPDX-License-Identifier: 0BSD

//! Native loading of compact recurrence artifacts.

use super::eager_manifest::PreparedKernelPackManifest;
use super::evaluator::recurrence_source_direct::{
    DirectSourceDispatchDomainSpec, DirectSourceDispatchKey, DirectSourceDispatchVariantSpec,
    DirectSourceOrientation, DirectSourceTemplateSpec, DirectSourceWavefunctionFamily,
};
use super::recurrence_backend::NativeRecurrenceDirectExecutorBackend;
#[cfg(feature = "on-the-fly-test-support")]
use super::recurrence_backend::NativeRecurrencePreparedExecutorPool;
#[cfg(feature = "python-generation-bridge")]
use super::recurrence_bootstrap::{
    RecurrenceReadyCompanionV1, RecurrenceReadyRuntimeV1, RecurrenceReadySourceTemplateV1,
    RecurrenceReadySourceVariantV1,
};
use super::recurrence_bootstrap::{
    RecurrenceReadyExecutionV1, RecurrenceReadyPlanV1, RecurrenceReadySourceDomainV1,
    RecurrenceReadySourceFamilyV1, RecurrenceReadySourceKeyV1, RecurrenceReadySourceOrientationV1,
};
use super::recurrence_manifest::*;
use super::recurrence_process_pack::{
    ProcessDirectExecutorBinding, ProcessDirectExecutorPack, ProcessExecutorBackend,
    ProcessExecutorIdentities, ProcessExecutorTarget, RECURRENCE_PROCESS_BINDING_HEADER_SIZE_V4,
    RECURRENCE_PROCESS_BINDING_MAGIC_V4, RECURRENCE_PROCESS_BINDING_VERSION_V4,
    decode_process_executor_descriptors, read_string, semantic_digest_from_bytes,
};
use super::*;
use crate::pacbin::{PacbinMemberKind, PacbinReader};
#[cfg(feature = "on-the-fly-test-support")]
use crate::recurrence::direct_backend::{
    DirectCurrentObservation, begin_direct_current_observation, take_direct_current_observation,
};
#[cfg(feature = "on-the-fly-test-support")]
use crate::recurrence::direct_runtime::{
    DirectRecurrenceExecutionRuntime, DirectReplaySelectorPlan,
};
#[cfg(test)]
use crate::recurrence::on_the_fly::OnTheFlySourceExecutionSpecV1;
#[cfg(feature = "on-the-fly-test-support")]
use crate::recurrence::on_the_fly::hash_current_key;
#[cfg(feature = "on-the-fly-test-support")]
use crate::recurrence::on_the_fly::{
    ON_THE_FLY_WORK_CENSUS_BASIS_V1, OnTheFlyForbiddenWorkGuardV1, OnTheFlyQueryFamilyExecutorV1,
    OnTheFlySelectedQueryTraceV1, OnTheFlyStructuralInterpreter, OnTheFlyWorkspaceV1,
    QueryFamilyTraceInput, build_on_the_fly_selected_trace_v1,
};
use crate::recurrence::on_the_fly::{
    OnTheFlyProcessSeedV1, OnTheFlySourceOrientationV1, OnTheFlySourceWavefunctionFamilyV1,
};
#[cfg(feature = "on-the-fly-test-support")]
use crate::recurrence::{AuthenticatedRecurrenceBuilderInput, PreparedDirectExecutorCatalog};
use crate::recurrence::{
    DirectRecurrencePlan, FactorizedColorContractionKind,
    RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER, RECURRENCE_DIRECT_SCHEDULE_MEMBER,
    RecurrenceColorContraction, SemanticDigest, decode_recurrence_color_contraction_v3,
    decode_recurrence_direct_plan_v2, recurrence_color_contraction_digest,
    validate_recurrence_color_projection_certificate,
};

#[cfg(feature = "on-the-fly-test-support")]
const ON_THE_FLY_FAMILY_WORK_CENSUS_BASIS_V1: &str = "shared-query-family-union-v1";

pub(super) struct LoadedRecurrenceRuntime {
    pub(super) common: ExecutionRuntime,
    pub(super) lane: RecurrenceNativeRuntime,
}

fn companion_helicity_coefficient_is_canonical(structural_zero: bool, coefficient: f64) -> bool {
    coefficient == if structural_zero { 0.0 } else { 1.0 }
}

pub(super) struct LoadedRecurrencePlan {
    pub(super) plan: DirectRecurrencePlan,
    pub(super) process_executors: ProcessDirectExecutorPack,
}

struct LoadedRecurrenceProcessBinding {
    remap: RecurrenceProcessRemap,
    process_executors: ProcessDirectExecutorPack,
}

pub(super) fn load_recurrence_native_runtime(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
    physics: &ProcessPhysicsV1,
    _selection: &crate::ArtifactSelection,
) -> RusticolResult<LoadedRecurrenceRuntime> {
    let loaded_plan = load_plan(artifact, evaluator_root, manifest)?;
    let plan = loaded_plan.plan;
    let (mut common, parameter_defaults, parameter_projection, source_domains) =
        build_common_runtime(&plan, manifest, physics)?;
    let payload_root = recurrence_payload_root(artifact, manifest)?;
    let kernel_payloads = artifact.evaluator_payload_store(&payload_root)?;
    if manifest.helicity_selector_companion.is_some() {
        validate_recurrence_companion_physics(physics)?;
    }
    common.model_parameter_evaluator = if common
        .model_parameters
        .iter()
        .any(|parameter| parameter.kind == "derived_parameter_component")
    {
        let (pack_bytes, mut pack, fallback_payload_root) = load_prepared_pack(artifact, manifest)?;
        drop(pack_bytes);
        drop(pack.recurrence_template.take());
        drop(pack.recurrence_direct_template.take());
        if fallback_payload_root != payload_root {
            return Err(RusticolError::integrity(
                "recurrence parameter fallback resolved a different prepared payload root",
            ));
        }
        super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
            &pack,
            &common.model_parameters,
            &kernel_payloads,
        )?
    } else {
        None
    };
    common.refresh_derived_model_parameters()?;
    let helicity_selector_companion = if manifest.helicity_selector_companion.is_some() {
        Some(
            super::on_the_fly_load::load_recurrence_helicity_selector_companion(
                artifact,
                evaluator_root,
                manifest,
                physics,
                &kernel_payloads,
                &common,
            )?
            .runtime,
        )
    } else {
        None
    };
    let loaded_backend = NativeRecurrenceDirectExecutorBackend::load_from_process_pack(
        &loaded_plan.process_executors,
        &kernel_payloads,
        &plan,
        source_domains,
    )?;
    let (executors, backend_owners) = loaded_backend.into_parts();
    let public_flow_ids = public_flow_ids(&plan, &manifest.runtime_metadata, physics)?;
    let direct_helicity_to_physics = direct_helicity_to_physics(&plan, physics)?;
    let color_contraction = load_color_contraction(artifact, evaluator_root, manifest)?;
    let lane = RecurrenceNativeRuntime::new(
        plan,
        executors,
        backend_owners,
        parameter_defaults,
        parameter_projection,
        public_flow_ids,
        direct_helicity_to_physics,
        color_contraction,
        helicity_selector_companion,
    )?;
    Ok(LoadedRecurrenceRuntime { common, lane })
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
pub(super) fn load_recurrence_ready_runtime(
    artifact: &mut VerifiedArtifact,
    evaluator_root: &Path,
    ready: &RecurrenceReadyExecutionV1,
    physics: &ProcessPhysicsV1,
    evaluator_payload_container: &crate::artifact::EvaluatorPayloadContainerExtension,
) -> RusticolResult<LoadedRecurrenceRuntime> {
    validate_ready_runtime_capabilities(ready)?;
    if ready.runtime.external_is_initial.len() != physics.external_particles.len()
        || ready
            .runtime
            .external_is_initial
            .iter()
            .zip(&physics.external_particles)
            .any(|(is_initial, particle)| {
                *is_initial != (particle.role == crate::ParticleRole::Initial)
            })
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence external roles disagree with authenticated physics",
        ));
    }
    let loaded_primary = load_ready_plan(
        artifact,
        evaluator_root,
        ready,
        &ready.primary_plan,
        RecurrenceReadyPlanRoleV1::Primary,
    )?;
    let mut loaded_companion = ready
        .companion
        .as_ref()
        .map(|companion| {
            validate_recurrence_companion_physics(physics)?;
            if companion.process_digest != ready.primary_plan.process_binding.process_digest
                || companion.process_digest != companion.plan.process_binding.process_digest
            {
                return Err(RusticolError::integrity(
                    "process-ready recurrence companion belongs to a different process",
                ));
            }
            load_ready_plan(
                artifact,
                evaluator_root,
                ready,
                &companion.plan,
                RecurrenceReadyPlanRoleV1::HelicitySelector,
            )
        })
        .transpose()?;
    let has_derived_parameters = ready
        .runtime
        .runtime_parameters
        .iter()
        .any(|parameter| parameter.kind == "derived_parameter_component");
    let needs_evaluator_payloads = recurrence_ready_needs_evaluator_payloads(
        process_pack_requires_evaluator_payloads(&loaded_primary.process_executors),
        has_derived_parameters,
        ready.companion.is_some(),
    );
    if needs_evaluator_payloads {
        artifact.install_evaluator_payload_container(evaluator_payload_container)?;
    }

    let plan = loaded_primary.plan;
    let (mut common, parameter_defaults, parameter_projection, source_domains) =
        build_ready_common_runtime(&plan, ready)?;
    let payload_root = ready_recurrence_payload_root(artifact, ready)?;
    let kernel_payloads = artifact.evaluator_payload_store(&payload_root)?;
    common.model_parameter_evaluator = if has_derived_parameters {
        let (pack_bytes, mut pack, fallback_payload_root) =
            load_ready_prepared_pack(artifact, ready)?;
        drop(pack_bytes);
        drop(pack.recurrence_template.take());
        let direct = pack.recurrence_direct_template_catalog(
            &ready.prepared_kernel_pack_digest,
            &ready.direct_template_catalog_digest,
        )?;
        if direct.compiled_model_digest != ready.compiled_model_digest
            || direct.recurrence_template_catalog_digest != ready.recurrence_template_catalog_digest
        {
            return Err(RusticolError::integrity(
                "prepared parameter fallback belongs to a different recurrence model",
            ));
        }
        drop(direct);
        drop(pack.recurrence_direct_template.take());
        if fallback_payload_root != payload_root {
            return Err(RusticolError::integrity(
                "process-ready recurrence parameter fallback resolved a different payload root",
            ));
        }
        super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
            &pack,
            &common.model_parameters,
            &kernel_payloads,
        )?
    } else {
        None
    };
    common.refresh_derived_model_parameters()?;

    let helicity_selector_companion = match (ready.companion.as_ref(), loaded_companion.take()) {
        (Some(companion), Some(loaded)) => Some(
            super::on_the_fly_load::load_recurrence_helicity_selector_companion_from_ready_parts(
                artifact,
                loaded,
                &companion.plan,
                companion,
                ready,
                physics,
                &kernel_payloads,
            )?
            .runtime,
        ),
        (None, None) => None,
        _ => {
            return Err(RusticolError::internal(
                "process-ready recurrence companion load state drifted",
            ));
        }
    };
    let loaded_backend = NativeRecurrenceDirectExecutorBackend::load_from_process_pack(
        &loaded_primary.process_executors,
        &kernel_payloads,
        &plan,
        source_domains,
    )?;
    let (executors, backend_owners) = loaded_backend.into_parts();
    let public_flow_ids = validate_ready_public_flow_ids(
        &plan,
        &ready.runtime.public_flow_ids,
        &ready.runtime.public_flow_public_ids,
        physics,
    )?;
    let direct_helicity_to_physics = validate_ready_direct_helicity_to_physics(
        &plan,
        &ready.runtime.direct_helicity_to_physics,
        physics,
    )?;
    let color_contraction = ready
        .runtime
        .color_contraction
        .as_ref()
        .map(|reference| {
            load_color_contraction_reference(artifact, evaluator_root, &ready.key, reference)
        })
        .transpose()?;
    let lane = RecurrenceNativeRuntime::new(
        plan,
        executors,
        backend_owners,
        parameter_defaults,
        parameter_projection,
        public_flow_ids,
        direct_helicity_to_physics,
        color_contraction,
        helicity_selector_companion,
    )?;
    Ok(LoadedRecurrenceRuntime { common, lane })
}

fn process_pack_requires_evaluator_payloads(pack: &ProcessDirectExecutorPack) -> bool {
    pack.descriptors
        .iter()
        .any(|descriptor| process_binding_requires_evaluator_payloads(&descriptor.binding))
}

fn process_binding_requires_evaluator_payloads(binding: &ProcessDirectExecutorBinding) -> bool {
    match binding {
        ProcessDirectExecutorBinding::Source | ProcessDirectExecutorBinding::Intrinsic(_) => false,
        ProcessDirectExecutorBinding::Jit(_) | ProcessDirectExecutorBinding::Native(_) => true,
    }
}

const fn recurrence_ready_needs_evaluator_payloads(
    process_pack_requires_payloads: bool,
    has_derived_parameters: bool,
    has_companion: bool,
) -> bool {
    process_pack_requires_payloads
        || has_derived_parameters
        // Companion executors are intentionally conservative here. They may
        // warm a different sparse descriptor family than the primary plan.
        || has_companion
}

fn validate_ready_runtime_capabilities(ready: &RecurrenceReadyExecutionV1) -> RusticolResult<()> {
    let color_capability = match ready.color_accuracy.as_str() {
        "lc" => crate::recurrence::RECURRENCE_LC_COLOR_CAPABILITY,
        "nlc" | "full" => crate::recurrence::RECURRENCE_CONTRACTED_COLOR_CAPABILITY,
        other => {
            return Err(RusticolError::compatibility(format!(
                "process-ready recurrence has unsupported color accuracy {other:?}"
            )));
        }
    };
    let mut expected = vec![
        color_capability,
        crate::recurrence::RECURRENCE_RUNTIME_CAPABILITY,
    ];
    if ready
        .runtime
        .color_contraction
        .as_ref()
        .and_then(|reference| reference.factorization.as_ref())
        .is_some_and(|factorization| factorization.kind == "symmetric-group-fourier")
    {
        expected.push(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY);
    }
    if ready.companion.is_some() {
        expected.push(RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY);
    }
    expected.sort_unstable();
    if ready
        .required_runtime_capabilities
        .iter()
        .map(String::as_str)
        .ne(expected.iter().copied())
    {
        return Err(RusticolError::compatibility(format!(
            "process-ready recurrence must require exactly {expected:?}"
        )));
    }
    Ok(())
}

fn load_color_contraction(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
) -> RusticolResult<Option<RecurrenceColorContraction>> {
    let Some(reference) = manifest.runtime_metadata.color_contraction.as_ref() else {
        return Ok(None);
    };
    load_color_contraction_reference(artifact, evaluator_root, &manifest.key, reference).map(Some)
}

pub(super) fn load_color_contraction_reference(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    process_key: &str,
    reference: &RecurrenceColorContractionReference,
) -> RusticolResult<RecurrenceColorContraction> {
    let relative_root = evaluator_root
        .strip_prefix(artifact.root())
        .map_err(|_| RusticolError::security("recurrence process root escapes the artifact"))?;
    let logical = relative_root.join(&reference.path);
    let logical = logical
        .to_str()
        .ok_or_else(|| RusticolError::security("recurrence color payload path is not UTF-8"))?;
    let record = artifact.payload(logical)?;
    if record.role != PayloadRole::EvaluatorState
        || record.media_type != "application/octet-stream"
        || record.process_id.as_deref() != Some(process_key)
        || record.size_bytes != reference.size_bytes
        || record.sha256 != reference.sha256
    {
        return Err(RusticolError::integrity(
            "recurrence color-contraction payload disagrees with execution.json",
        ));
    }
    let bytes = artifact.read_payload(logical)?;
    let semantic_digest = SemanticDigest::new(recurrence_color_contraction_digest(&bytes))
        .map_err(|_| {
            RusticolError::integrity(
                "recurrence color-contraction payload has an invalid semantic digest",
            )
        })?;
    if semantic_digest.to_string() != reference.semantic_digest {
        return Err(RusticolError::integrity(
            "recurrence color-contraction payload digest is inconsistent",
        ));
    }
    let contraction = decode_recurrence_color_contraction_v3(&bytes)?;
    let accuracy = match contraction.accuracy() {
        crate::recurrence::RecurrenceColorAccuracy::Nlc => "nlc",
        crate::recurrence::RecurrenceColorAccuracy::Full => "full",
    };
    let storage = match contraction.storage() {
        crate::recurrence::RecurrenceColorStorage::Expanded => "expanded",
        crate::recurrence::RecurrenceColorStorage::Repeated => "repeated",
        crate::recurrence::RecurrenceColorStorage::ConvolutionKernels => "convolution-kernels",
    };
    if accuracy != reference.color_accuracy
        || storage != reference.storage
        || contraction.active_sector_count() as u64 != reference.active_sector_count
        || contraction.group_count() as u64 != reference.group_count
        || contraction.sector_count() as u64 != reference.sector_count
        || contraction.component_count() as u64 != reference.component_count
        || contraction.destination_count() as u64 != reference.destination_count
        || contraction.stored_entry_count() as u64 != reference.entry_count
        || contraction.logical_entry_count() as u64 != reference.logical_entry_count
        || contraction.includes_color_factor() != reference.includes_color_factor
    {
        return Err(RusticolError::integrity(
            "recurrence color-contraction payload disagrees with its bounded summary",
        ));
    }
    let factorization_matches = match (
        reference.factorization.as_ref(),
        contraction.factorization(),
    ) {
        (None, None) => true,
        (Some(reference), Some(factorization)) => {
            let kind = match factorization.kind() {
                FactorizedColorContractionKind::KleinFourWalsh => "klein-four-walsh",
                FactorizedColorContractionKind::ElementaryAbelianWalsh => {
                    "elementary-abelian-walsh"
                }
                FactorizedColorContractionKind::SymmetricGroupFourier => "symmetric-group-fourier",
            };
            reference.kind == kind
                && reference.rank == factorization.rank()
                && reference.coset_count == factorization.coset_count() as u64
        }
        _ => false,
    };
    if !factorization_matches {
        return Err(RusticolError::integrity(
            "recurrence color-contraction factorization disagrees with its bounded summary",
        ));
    }
    Ok(contraction)
}

pub(super) fn load_recurrence_exact_sections(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<NativeRecurrenceExactSections> {
    let (pack_bytes, pack, _payload_root) = load_prepared_pack(artifact, manifest)?;
    drop(pack_bytes);
    let plan = load_plan(artifact, evaluator_root, manifest)?.plan;
    let direct = pack.recurrence_direct_template_catalog(
        &manifest.prepared_kernel_pack_digest,
        &manifest.direct_template_catalog_digest,
    )?;
    if direct.catalog_digest != plan.direct_template_catalog_digest().to_string() {
        return Err(RusticolError::integrity(
            "prepared Direct-Arena catalog digest does not match the recurrence plan",
        ));
    }
    let public_flow_ids = public_flow_ids(&plan, &manifest.runtime_metadata, physics)?;
    let exact_factors = plan
        .exact_factors()
        .iter()
        .map(|factor| NativeRecurrenceExactFactor {
            real_numerator: factor.real().numerator().to_string(),
            real_denominator: factor.real().denominator().to_string(),
            imaginary_numerator: factor.imag().numerator().to_string(),
            imaginary_denominator: factor.imag().denominator().to_string(),
        })
        .collect();
    let executors = direct
        .templates
        .into_iter()
        .map(|template| NativeRecurrenceExactExecutor {
            direct_executor_id: template.direct_executor_id,
            role: template.role,
            destination_operation: template.destination_operation,
            parent_component_counts: template.parent_component_counts,
            destination_component_count: template.destination_component_count,
            momentum_operand_count: template.momentum_operand_count,
            prepared_kernel_id: template.payload_binding.prepared_kernel_id,
            runtime_template: template.payload_binding.runtime_template,
        })
        .collect();
    Ok(NativeRecurrenceExactSections {
        process_id: manifest.key.clone(),
        strategy: plan.strategy().as_str().to_string(),
        semantic_digest: manifest
            .plan
            .process_binding
            .process_semantic_digest
            .clone(),
        runtime_layout_digest: plan.runtime_layout_digest().to_string(),
        current_arena_components: plan.current_arena_components(),
        amplitude_destination_count: plan.amplitude_destination_count(),
        parameter_value_count: plan.parameter_value_count(),
        external_source_count: plan.external_source_count(),
        currents: plan.currents().to_vec(),
        sources: plan.sources().to_vec(),
        contributions: plan.contributions().to_vec(),
        finalizations: plan.finalizations().to_vec(),
        closures: plan.closures().to_vec(),
        row_groups: plan.row_groups().to_vec(),
        momentum_forms: plan.momentum_forms().to_vec(),
        momentum_terms: plan.momentum_terms().to_vec(),
        replay_targets: plan.replay_targets().to_vec(),
        source_permutations: plan.source_permutations().to_vec(),
        replay_momentum_signs: plan.replay_momentum_signs().to_vec(),
        replay_helicity_map: plan.replay_helicity_map().to_vec(),
        amplitude_destinations: plan.amplitude_destinations().to_vec(),
        resolved_helicities: plan.resolved_helicities().to_vec(),
        public_helicities: plan.public_helicities().to_vec(),
        source_state_assignments: plan.source_state_assignments().to_vec(),
        source_dispatch_variants: plan.source_dispatch_variants().to_vec(),
        source_embeddings: plan.source_embeddings().to_vec(),
        source_projections: plan.source_projections().to_vec(),
        resolved_source_selections: plan.resolved_source_selections().to_vec(),
        exact_factors,
        public_flow_ids,
        executors,
    })
}

fn public_flow_ids(
    plan: &DirectRecurrencePlan,
    metadata: &RecurrenceRuntimeMetadata,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<Vec<u32>> {
    let available = match plan.strategy() {
        crate::recurrence::RecurrenceStrategy::TopologyReplay => {
            let available = plan
                .replay_targets()
                .iter()
                .map(|target| target.public_flow_id)
                .collect::<BTreeSet<_>>();
            if available.len() != plan.replay_targets().len() {
                return Err(RusticolError::integrity(
                    "recurrence direct plan repeats a public replay target",
                ));
            }
            available
        }
        crate::recurrence::RecurrenceStrategy::AllFlowUnion => {
            let available = plan
                .amplitude_destinations()
                .iter()
                .map(|destination| destination.target_sector_id)
                .collect::<BTreeSet<_>>();
            if available.len() != plan.amplitude_destinations().len() {
                return Err(RusticolError::integrity(
                    "all-flow-union direct plan repeats a physical-flow destination",
                ));
            }
            available
        }
        crate::recurrence::RecurrenceStrategy::ContractedColorUnion => {
            if !metadata.public_color_flows.is_empty()
                || physics.color_components.len() != 1
                || !matches!(
                    physics.color_components.first(),
                    Some(PhysicsColorComponentV1::ContractedColor(_))
                )
            {
                return Err(RusticolError::integrity(
                    "contracted recurrence must expose exactly one contracted color component and no public flow bindings",
                ));
            }
            return Ok(Vec::new());
        }
    };
    if metadata.public_color_flows.len() != physics.color_components.len() {
        return Err(RusticolError::integrity(
            "recurrence public color-flow bindings do not cover the physics axis",
        ));
    }
    let mut seen = BTreeSet::new();
    let strategy = plan.strategy();
    let result = metadata
        .public_color_flows
        .iter()
        .zip(&physics.color_components)
        .map(|(binding, component)| {
            if binding.public_id != component.id() {
                return Err(RusticolError::integrity(
                    "recurrence public color-flow binding order disagrees with physics.json",
                ));
            }
            if !matches!(component, PhysicsColorComponentV1::LcFlow(_)) {
                return Err(RusticolError::integrity(
                    "recurrence public color-flow binding references a non-LC component",
                ));
            }
            let plan_sector_id = match strategy {
                crate::recurrence::RecurrenceStrategy::TopologyReplay => binding.target_sector_id,
                crate::recurrence::RecurrenceStrategy::AllFlowUnion => {
                    binding.construction_sector_id
                }
                crate::recurrence::RecurrenceStrategy::ContractedColorUnion => {
                    unreachable!("contracted recurrence has no public color-flow bindings")
                }
            };
            if plan_sector_id >= plan.physical_sector_count()
                || !available.contains(&plan_sector_id)
                || (strategy == crate::recurrence::RecurrenceStrategy::TopologyReplay
                    && !seen.insert(plan_sector_id))
            {
                return Err(RusticolError::integrity(
                    "recurrence public color-flow target is absent or repeated in the direct plan",
                ));
            }
            seen.insert(plan_sector_id);
            Ok(plan_sector_id)
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if seen != available {
        return Err(RusticolError::integrity(
            "recurrence direct-plan flow destinations do not match the public color-flow axis",
        ));
    }
    Ok(result)
}

fn validate_ready_public_flow_ids(
    plan: &DirectRecurrencePlan,
    stored: &[u32],
    stored_public_ids: &[String],
    physics: &ProcessPhysicsV1,
) -> RusticolResult<Vec<u32>> {
    let available = match plan.strategy() {
        crate::recurrence::RecurrenceStrategy::TopologyReplay => plan
            .replay_targets()
            .iter()
            .map(|target| target.public_flow_id)
            .collect::<BTreeSet<_>>(),
        crate::recurrence::RecurrenceStrategy::AllFlowUnion => plan
            .amplitude_destinations()
            .iter()
            .map(|destination| destination.target_sector_id)
            .collect::<BTreeSet<_>>(),
        crate::recurrence::RecurrenceStrategy::ContractedColorUnion => {
            if !stored.is_empty()
                || !stored_public_ids.is_empty()
                || physics.color_components.len() != 1
                || !matches!(
                    physics.color_components.first(),
                    Some(PhysicsColorComponentV1::ContractedColor(_))
                )
            {
                return Err(RusticolError::integrity(
                    "process-ready contracted recurrence has an invalid public color axis",
                ));
            }
            return Ok(Vec::new());
        }
    };
    if stored.len() != physics.color_components.len()
        || stored_public_ids.len() != physics.color_components.len()
        || physics
            .color_components
            .iter()
            .any(|component| !matches!(component, PhysicsColorComponentV1::LcFlow(_)))
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence flow mapping does not cover the LC physics axis",
        ));
    }
    if stored_public_ids
        .iter()
        .zip(&physics.color_components)
        .any(|(stored_id, component)| stored_id != component.id())
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence flow IDs disagree with the physics axis",
        ));
    }
    let mut seen = BTreeSet::new();
    for sector in stored {
        if *sector >= plan.physical_sector_count()
            || !available.contains(sector)
            || (plan.strategy() == crate::recurrence::RecurrenceStrategy::TopologyReplay
                && !seen.insert(*sector))
        {
            return Err(RusticolError::integrity(
                "process-ready recurrence flow mapping references an absent plan sector",
            ));
        }
        seen.insert(*sector);
    }
    if seen != available {
        return Err(RusticolError::integrity(
            "process-ready recurrence flow mapping does not cover the plan destinations",
        ));
    }
    Ok(stored.to_vec())
}

pub(super) fn direct_helicity_to_physics(
    plan: &DirectRecurrencePlan,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<Vec<usize>> {
    let mut result = Vec::with_capacity(plan.resolved_helicities().len());
    let mut seen_physics = BTreeSet::new();
    for (expected_id, descriptor) in plan.resolved_helicities().iter().enumerate() {
        if descriptor.id as usize != expected_id {
            return Err(RusticolError::integrity(
                "recurrence resolved-helicity IDs are not dense and ordered",
            ));
        }
        let start = usize::try_from(descriptor.public_helicity_start).map_err(|_| {
            RusticolError::artifact("recurrence public-helicity offset exceeds usize")
        })?;
        let count = usize::try_from(descriptor.public_helicity_count).map_err(|_| {
            RusticolError::artifact("recurrence public-helicity count exceeds usize")
        })?;
        let stop = start
            .checked_add(count)
            .ok_or_else(|| RusticolError::artifact("recurrence public-helicity range overflows"))?;
        let values = plan.public_helicities().get(start..stop).ok_or_else(|| {
            RusticolError::integrity(
                "recurrence resolved-helicity public vector is outside the direct plan",
            )
        })?;
        if values.len() != physics.external_particles.len() {
            return Err(RusticolError::integrity(
                "recurrence resolved-helicity width disagrees with the physics axis",
            ));
        }
        let matches = physics
            .helicities
            .iter()
            .enumerate()
            .filter_map(|(index, helicity)| (helicity.values == values).then_some(index))
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(RusticolError::integrity(format!(
                "recurrence resolved helicity {expected_id} maps to {} physics helicities",
                matches.len()
            )));
        }
        let physics_index = matches[0];
        if !seen_physics.insert(physics_index) {
            return Err(RusticolError::integrity(
                "recurrence resolved helicities repeat a physics helicity",
            ));
        }
        result.push(physics_index);
    }
    Ok(result)
}

pub(super) fn validate_ready_direct_helicity_to_physics(
    plan: &DirectRecurrencePlan,
    stored: &[u64],
    physics: &ProcessPhysicsV1,
) -> RusticolResult<Vec<usize>> {
    if stored.len() != plan.resolved_helicities().len() {
        return Err(RusticolError::integrity(
            "process-ready helicity mapping has the wrong size",
        ));
    }
    let mut result = Vec::with_capacity(stored.len());
    let mut seen = BTreeSet::new();
    for (expected_id, (descriptor, physics_index)) in
        plan.resolved_helicities().iter().zip(stored).enumerate()
    {
        if descriptor.id as usize != expected_id {
            return Err(RusticolError::integrity(
                "recurrence resolved-helicity IDs are not dense and ordered",
            ));
        }
        let physics_index = usize::try_from(*physics_index)
            .map_err(|_| RusticolError::artifact("physics helicity index exceeds usize"))?;
        if !seen.insert(physics_index) {
            return Err(RusticolError::integrity(
                "process-ready recurrence helicity mapping repeats a physics row",
            ));
        }
        let physics_helicity = physics.helicities.get(physics_index).ok_or_else(|| {
            RusticolError::integrity(
                "process-ready recurrence helicity mapping is outside the physics axis",
            )
        })?;
        let start = usize::try_from(descriptor.public_helicity_start)
            .map_err(|_| RusticolError::artifact("public-helicity offset exceeds usize"))?;
        let count = usize::try_from(descriptor.public_helicity_count)
            .map_err(|_| RusticolError::artifact("public-helicity count exceeds usize"))?;
        let stop = start
            .checked_add(count)
            .ok_or_else(|| RusticolError::artifact("public-helicity range overflows"))?;
        if plan.public_helicities().get(start..stop) != Some(physics_helicity.values.as_slice()) {
            return Err(RusticolError::integrity(
                "process-ready recurrence helicity mapping disagrees with the direct plan",
            ));
        }
        result.push(physics_index);
    }
    Ok(result)
}

fn load_prepared_pack(
    artifact: &VerifiedArtifact,
    manifest: &RecurrenceExecutionManifest,
) -> RusticolResult<(Vec<u8>, PreparedKernelPackManifest, PathBuf)> {
    load_prepared_pack_reference(
        artifact,
        &manifest.kernel_pack.manifest_path,
        &manifest.kernel_pack.payload_root,
    )
}

fn load_ready_prepared_pack(
    artifact: &VerifiedArtifact,
    ready: &RecurrenceReadyExecutionV1,
) -> RusticolResult<(Vec<u8>, PreparedKernelPackManifest, PathBuf)> {
    load_prepared_pack_reference(
        artifact,
        &ready.kernel_pack.manifest_path,
        &ready.kernel_pack.payload_root,
    )
}

fn load_prepared_pack_reference(
    artifact: &VerifiedArtifact,
    manifest_path: &str,
    payload_root: &str,
) -> RusticolResult<(Vec<u8>, PreparedKernelPackManifest, PathBuf)> {
    let manifest_path = confined_internal_path(
        manifest_path,
        "recurrence prepared kernel-pack manifest path",
    )?;
    let manifest_path = manifest_path.to_str().ok_or_else(|| {
        RusticolError::security("recurrence kernel-pack manifest path is not valid UTF-8")
    })?;
    if artifact.payload(manifest_path)?.role != PayloadRole::EvaluatorManifest {
        return Err(RusticolError::security(
            "recurrence kernel-pack manifest is not an evaluator-manifest payload",
        ));
    }
    let bytes = artifact.read_payload(manifest_path)?;
    let pack: PreparedKernelPackManifest = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse recurrence prepared kernel pack: {error}"
        ))
    })?;
    pack.validate()?;
    validate_recurrence_prepared_pack_outer_target(&artifact.manifest().producer.target, &pack)?;
    let payload_root = artifact.root().join(confined_internal_path(
        payload_root,
        "recurrence prepared kernel payload root",
    )?);
    Ok((bytes, pack, payload_root))
}

fn recurrence_payload_root(
    artifact: &VerifiedArtifact,
    manifest: &RecurrenceExecutionManifest,
) -> RusticolResult<PathBuf> {
    Ok(artifact.root().join(confined_internal_path(
        &manifest.kernel_pack.payload_root,
        "recurrence prepared kernel payload root",
    )?))
}

fn ready_recurrence_payload_root(
    artifact: &VerifiedArtifact,
    ready: &RecurrenceReadyExecutionV1,
) -> RusticolResult<PathBuf> {
    Ok(artifact.root().join(confined_internal_path(
        &ready.kernel_pack.payload_root,
        "process-ready recurrence kernel payload root",
    )?))
}

pub(super) fn validate_recurrence_prepared_pack_outer_target(
    outer_target: &crate::Target,
    pack: &PreparedKernelPackManifest,
) -> RusticolResult<()> {
    pack.validate_portable_process_artifact_target(outer_target, "recurrence")
}

fn load_plan(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
) -> RusticolResult<LoadedRecurrencePlan> {
    load_plan_summary(artifact, evaluator_root, manifest, &manifest.plan)
}

pub(super) fn load_plan_summary(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
    summary: &RecurrencePlanSummary,
) -> RusticolResult<LoadedRecurrencePlan> {
    load_plan_reference(
        artifact,
        evaluator_root,
        RecurrencePlanLoadSpec {
            process_key: &manifest.key,
            compiled_model_digest: &manifest.compiled_model_digest,
            recurrence_template_catalog_digest: &manifest.recurrence_template_catalog_digest,
            prepared_kernel_pack_digest: &manifest.prepared_kernel_pack_digest,
            direct_template_catalog_digest: &manifest.direct_template_catalog_digest,
            plan: RecurrenceReadyPlanRef {
                runtime_schedule: &summary.runtime_schedule,
                process_binding: &summary.process_binding,
                runtime_container_member: &summary.inspection_summary.runtime_container_member,
                color_projection_certificate: summary
                    .inspection_summary
                    .color_projection_certificate
                    .as_ref(),
                helicity_dispatch: summary.helicity_dispatch.as_ref(),
            },
            role: if summary.helicity_dispatch.is_some() {
                RecurrenceReadyPlanRoleV1::HelicitySelector
            } else {
                RecurrenceReadyPlanRoleV1::Primary
            },
        },
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RecurrenceReadyPlanRoleV1 {
    Primary,
    HelicitySelector,
}

struct RecurrenceReadyPlanRef<'a> {
    runtime_schedule: &'a RecurrenceRuntimeContainer,
    process_binding: &'a RecurrenceProcessBinding,
    runtime_container_member: &'a RecurrenceRuntimeContainerMember,
    color_projection_certificate: Option<&'a RecurrenceColorProjectionCertificate>,
    helicity_dispatch: Option<&'a RecurrenceHelicityDispatchReference>,
}

struct RecurrencePlanLoadSpec<'a> {
    process_key: &'a str,
    compiled_model_digest: &'a str,
    recurrence_template_catalog_digest: &'a str,
    prepared_kernel_pack_digest: &'a str,
    direct_template_catalog_digest: &'a str,
    plan: RecurrenceReadyPlanRef<'a>,
    role: RecurrenceReadyPlanRoleV1,
}

fn validate_ready_plan_reference(spec: &RecurrencePlanLoadSpec<'_>) -> RusticolResult<()> {
    let container = spec.plan.runtime_schedule;
    let binding = spec.plan.process_binding;
    if container.kind != RECURRENCE_RUNTIME_CONTAINER_KIND
        || container.schema_version != RECURRENCE_RUNTIME_CONTAINER_SCHEMA
        || container.storage_abi != RECURRENCE_RUNTIME_STORAGE_ABI
        || container.plan_member_path != RECURRENCE_DIRECT_SCHEDULE_MEMBER
    {
        return Err(RusticolError::compatibility(
            "unsupported process-ready recurrence schedule contract",
        ));
    }
    let expected_schedule_path = format!(
        "recurrence/schedules/{}/recurrence-runtime.pacbin",
        binding.schedule_digest
    );
    if container.path != expected_schedule_path
        || !(1..=2).contains(&container.member_count)
        || container.size_bytes == 0
        || container.unpacked_size_bytes == 0
        || container.unpacked_size_bytes > container.size_bytes
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence schedule metadata is inconsistent",
        ));
    }
    for digest in [
        &container.sha256,
        &container.index_sha256,
        &binding.schedule_digest,
        &binding.native_schedule_semantic_digest,
        &binding.process_digest,
        &binding.process_semantic_digest,
        &binding.sha256,
    ] {
        decode_sha256(digest)?;
    }
    let expected_binding_path = match spec.role {
        RecurrenceReadyPlanRoleV1::Primary => RECURRENCE_PROCESS_BINDING_PATH,
        RecurrenceReadyPlanRoleV1::HelicitySelector => RECURRENCE_HELICITY_SELECTOR_BINDING_PATH,
    };
    if binding.abi != RECURRENCE_PROCESS_BINDING_ABI
        || binding.path != expected_binding_path
        || binding.process_id != spec.process_key
        || binding.size_bytes == 0
        || binding.process_support_words.is_empty()
        || binding.process_support_words.last() == Some(&0)
        || binding
            .process_support_words
            .iter()
            .map(|word| word.count_ones())
            .sum::<u32>()
            != 1
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence binding metadata is inconsistent",
        ));
    }
    validate_ready_process_remap(&binding.remap)?;

    let member = spec.plan.runtime_container_member;
    if member.path != RECURRENCE_DIRECT_SCHEDULE_MEMBER
        || member.size_bytes == 0
        || member.container_size_bytes != container.size_bytes
        || member.size_bytes > member.container_size_bytes
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence plan-member metadata is inconsistent",
        ));
    }
    decode_sha256(&member.sha256)?;
    let certificate_size = if let Some(certificate) = spec.plan.color_projection_certificate {
        if certificate.path != RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER
            || certificate.schema_version != 1
            || certificate.proof_kind != "exact-rectangular-sum-projection"
            || !certificate.publishable
            || certificate.size_bytes == 0
        {
            return Err(RusticolError::compatibility(
                "unsupported process-ready recurrence projection certificate",
            ));
        }
        decode_sha256(&certificate.sha256)?;
        certificate.size_bytes
    } else {
        0
    };
    let expected_member_count = 1 + u64::from(certificate_size != 0);
    let expected_unpacked = member
        .size_bytes
        .checked_add(certificate_size)
        .ok_or_else(|| RusticolError::artifact("recurrence member sizes overflow u64"))?;
    if container.member_count != expected_member_count
        || container.unpacked_size_bytes != expected_unpacked
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence PACBIN member inventory is inconsistent",
        ));
    }
    match (spec.role, spec.plan.helicity_dispatch) {
        (RecurrenceReadyPlanRoleV1::Primary, None) => {}
        (RecurrenceReadyPlanRoleV1::HelicitySelector, Some(dispatch)) => {
            let expected_path = format!(
                "recurrence/schedules/{}/{}",
                binding.schedule_digest, RECURRENCE_HELICITY_DISPATCH_PATH
            );
            if dispatch.abi != crate::recurrence::RECURRENCE_HELICITY_DISPATCH_ABI
                || dispatch.path != expected_path
                || dispatch.size_bytes == 0
                || dispatch.resolved_helicity_count > u64::from(u32::MAX)
            {
                return Err(RusticolError::compatibility(
                    "unsupported process-ready recurrence helicity dispatch",
                ));
            }
            decode_sha256(&dispatch.sha256)?;
            decode_sha256(&dispatch.base_runtime_layout_digest)?;
        }
        _ => {
            return Err(RusticolError::integrity(
                "process-ready recurrence plan has the wrong helicity-dispatch role",
            ));
        }
    }
    Ok(())
}

fn validate_ready_process_remap(remap: &RecurrenceProcessRemap) -> RusticolResult<()> {
    decode_sha256(&remap.bijection_digest)?;
    validate_ready_permutation(&remap.source_slots, "source slots")?;
    validate_ready_permutation(&remap.public_flow_ids, "public flows")?;
    validate_ready_permutation(&remap.physical_sector_ids, "physical sectors")?;
    if remap.source_momentum_signs.len() != remap.source_slots.len()
        || remap.source_helicity_signs.len() != remap.source_slots.len()
        || remap
            .source_momentum_signs
            .iter()
            .chain(&remap.source_helicity_signs)
            .any(|sign| !matches!(sign, -1 | 1))
        || remap.source_state_offsets.len() != remap.source_slots.len() + 1
        || remap.source_state_offsets.first() != Some(&0)
        || remap.source_state_offsets.last().copied().map(u64::from)
            != Some(remap.source_state_indices.len() as u64)
        || remap
            .source_state_offsets
            .windows(2)
            .any(|window| window[0] > window[1])
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence source-state remap is malformed",
        ));
    }
    for window in remap.source_state_offsets.windows(2) {
        let start = usize::try_from(window[0])
            .map_err(|_| RusticolError::artifact("source-state offset exceeds usize"))?;
        let stop = usize::try_from(window[1])
            .map_err(|_| RusticolError::artifact("source-state offset exceeds usize"))?;
        validate_ready_permutation(
            remap.source_state_indices.get(start..stop).ok_or_else(|| {
                RusticolError::integrity("source-state remap range is out of bounds")
            })?,
            "source states",
        )?;
    }
    for mapping in [
        &remap.state_templates,
        &remap.source_templates,
        &remap.direct_executors,
        &remap.parameter_slots,
    ] {
        let mut previous = None;
        let mut sources = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for [source, target] in &mapping.changes {
            if *source >= mapping.count
                || *target >= mapping.count
                || source == target
                || previous.is_some_and(|value| value >= *source)
                || !targets.insert(*target)
            {
                return Err(RusticolError::integrity(
                    "process-ready sparse recurrence bijection is malformed",
                ));
            }
            previous = Some(*source);
            sources.insert(*source);
        }
        if sources != targets {
            return Err(RusticolError::integrity(
                "process-ready sparse recurrence changes do not form a bijection",
            ));
        }
    }
    Ok(())
}

fn validate_ready_permutation(values: &[u32], context: &str) -> RusticolResult<()> {
    let mut seen = vec![false; values.len()];
    for value in values {
        let index = usize::try_from(*value)
            .map_err(|_| RusticolError::artifact(format!("{context} index exceeds usize")))?;
        let slot = seen.get_mut(index).ok_or_else(|| {
            RusticolError::integrity(format!(
                "process-ready recurrence {context} is not a permutation"
            ))
        })?;
        if std::mem::replace(slot, true) {
            return Err(RusticolError::integrity(format!(
                "process-ready recurrence {context} repeats an index"
            )));
        }
    }
    Ok(())
}

pub(super) fn load_ready_plan(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    ready: &RecurrenceReadyExecutionV1,
    plan: &RecurrenceReadyPlanV1,
    role: RecurrenceReadyPlanRoleV1,
) -> RusticolResult<LoadedRecurrencePlan> {
    load_plan_reference(
        artifact,
        evaluator_root,
        RecurrencePlanLoadSpec {
            process_key: &ready.key,
            compiled_model_digest: &ready.compiled_model_digest,
            recurrence_template_catalog_digest: &ready.recurrence_template_catalog_digest,
            prepared_kernel_pack_digest: &ready.prepared_kernel_pack_digest,
            direct_template_catalog_digest: &ready.direct_template_catalog_digest,
            plan: RecurrenceReadyPlanRef {
                runtime_schedule: &plan.runtime_schedule,
                process_binding: &plan.process_binding,
                runtime_container_member: &plan.runtime_container_member,
                color_projection_certificate: plan.color_projection_certificate.as_ref(),
                helicity_dispatch: plan.helicity_dispatch.as_ref(),
            },
            role,
        },
    )
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn load_plan_reference(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    spec: RecurrencePlanLoadSpec<'_>,
) -> RusticolResult<LoadedRecurrencePlan> {
    #[cfg(feature = "on-the-fly-test-support")]
    crate::recurrence::on_the_fly::reject_forbidden_work_if_probed(
        crate::recurrence::on_the_fly::OnTheFlyForbiddenWorkV1::DirectPlanLoad,
    )?;
    validate_ready_plan_reference(&spec)?;
    let container = spec.plan.runtime_schedule;
    let path = artifact.root().join(&container.path);
    let payload = artifact.payload(&container.path)?;
    if payload.role != PayloadRole::EvaluatorState
        || payload.media_type != "application/octet-stream"
        || payload.process_id.is_some()
        || payload.executable
        || payload.size_bytes != container.size_bytes
        || payload.sha256 != container.sha256
    {
        return Err(RusticolError::integrity(
            "recurrence root schedule disagrees with its authenticated payload",
        ));
    }

    let process_binding = validate_process_binding(artifact, evaluator_root, &spec)?;

    let expected_file_sha = decode_sha256(&container.sha256)?;
    let container_file = artifact.open_payload_file(&container.path)?;
    let reader = PacbinReader::open_file_with_sha256(container_file, &path, &expected_file_sha)?;
    let index = reader.index();
    if index.file_size() != container.size_bytes
        || reader.container_size() as u64 != container.size_bytes
        || index.index_sha256().as_slice() != decode_sha256(&container.index_sha256)?.as_slice()
        || index.members().len() as u64 != container.member_count
        || index
            .members()
            .iter()
            .map(|member| member.length())
            .sum::<u64>()
            != container.unpacked_size_bytes
    {
        return Err(RusticolError::integrity(
            "recurrence runtime PACBIN metadata disagrees with execution.json",
        ));
    }
    let member = reader.member(RECURRENCE_DIRECT_SCHEDULE_MEMBER)?;
    let plan_metadata = spec.plan.runtime_container_member;
    if member.kind() != PacbinMemberKind::RecurrenceDirectPlan {
        return Err(RusticolError::compatibility(
            "recurrence runtime PACBIN contains an incompatible plan member",
        ));
    }
    if member.length() != plan_metadata.size_bytes
        || member.sha256().as_slice() != decode_sha256(&plan_metadata.sha256)?.as_slice()
    {
        return Err(RusticolError::integrity(
            "recurrence DirectPlan member disagrees with execution.json",
        ));
    }
    match (
        spec.plan.color_projection_certificate,
        index.members().len(),
    ) {
        (None, 1) => {}
        (Some(metadata), 2) => {
            let certificate = reader.member(RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER)?;
            if certificate.kind() != PacbinMemberKind::RecurrenceColorProjectionCertificate {
                return Err(RusticolError::compatibility(
                    "recurrence runtime PACBIN contains an incompatible projection certificate",
                ));
            }
            if certificate.length() != metadata.size_bytes
                || certificate.sha256().as_slice() != decode_sha256(&metadata.sha256)?.as_slice()
            {
                return Err(RusticolError::integrity(
                    "recurrence color-projection certificate disagrees with execution.json",
                ));
            }
            validate_recurrence_color_projection_certificate(
                reader.member_bytes(RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER)?,
            )?;
        }
        (None, 2) | (Some(_), 1) => {
            return Err(RusticolError::integrity(
                "recurrence runtime PACBIN projection certificate disagrees with execution.json",
            ));
        }
        (_, count) => {
            return Err(RusticolError::compatibility(format!(
                "recurrence runtime PACBIN contains {count} members; expected one plan and at most one projection certificate"
            )));
        }
    }
    let bytes = reader.member_bytes(RECURRENCE_DIRECT_SCHEDULE_MEMBER)?;
    let plan = decode_recurrence_direct_plan_v2(bytes)?;
    if plan.semantic_digest().to_string()
        != spec.plan.process_binding.native_schedule_semantic_digest()
    {
        return Err(RusticolError::integrity(
            "recurrence native schedule semantic digest disagrees with its process binding",
        ));
    }
    if plan.prepared_pack_digest().to_string() != spec.prepared_kernel_pack_digest
        || plan.direct_template_catalog_digest().to_string() != spec.direct_template_catalog_digest
    {
        return Err(RusticolError::integrity(
            "direct recurrence plan authentication digests disagree with execution.json",
        ));
    }
    let root_runtime_layout_digest = plan.runtime_layout_digest();
    let plan = apply_process_remap(plan, &process_binding.remap)?.into_runtime_compacted();
    let expected_prepared_pack_digest = semantic_digest_from_bytes(
        decode_sha256(spec.prepared_kernel_pack_digest)?,
        "prepared-kernel pack",
    )?;
    let expected_catalog_digest = semantic_digest_from_bytes(
        decode_sha256(spec.direct_template_catalog_digest)?,
        "direct-template catalog",
    )?;
    process_binding.process_executors.validate_for_plan(
        &plan,
        expected_prepared_pack_digest,
        expected_catalog_digest,
        root_runtime_layout_digest,
    )?;
    Ok(LoadedRecurrencePlan {
        plan,
        process_executors: process_binding.process_executors,
    })
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
#[cfg_attr(target_vendor = "apple", cold)]
fn validate_process_binding(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    spec: &RecurrencePlanLoadSpec<'_>,
) -> RusticolResult<LoadedRecurrenceProcessBinding> {
    let binding = spec.plan.process_binding;
    let relative_root = evaluator_root
        .strip_prefix(artifact.root())
        .map_err(|_| RusticolError::security("recurrence process root escapes the artifact"))?;
    let logical = relative_root.join(&binding.path);
    let logical = logical
        .to_str()
        .ok_or_else(|| RusticolError::security("recurrence process-binding path is not UTF-8"))?;
    let record = artifact.payload(logical)?;
    if record.role != PayloadRole::EvaluatorState
        || record.process_id.as_deref() != Some(spec.process_key)
        || record.size_bytes != binding.size_bytes
        || record.sha256 != binding.sha256
    {
        return Err(RusticolError::integrity(
            "recurrence process-binding payload disagrees with execution.json",
        ));
    }
    let bytes = artifact.read_payload(logical)?;
    if bytes.len() < RECURRENCE_PROCESS_BINDING_HEADER_SIZE_V4
        || bytes.get(..8) != Some(RECURRENCE_PROCESS_BINDING_MAGIC_V4.as_slice())
    {
        return Err(RusticolError::compatibility(
            "unsupported recurrence process-binding payload",
        ));
    }
    let u32_at = |offset: usize| {
        bytes
            .get(offset..offset + 4)
            .and_then(|raw| raw.try_into().ok())
            .map(u32::from_le_bytes)
            .ok_or_else(|| RusticolError::artifact("truncated recurrence process binding"))
    };
    if u32_at(8)? != RECURRENCE_PROCESS_BINDING_VERSION_V4
        || usize::try_from(u32_at(12)?).ok() != Some(RECURRENCE_PROCESS_BINDING_HEADER_SIZE_V4)
    {
        return Err(RusticolError::compatibility(
            "unsupported recurrence process-binding version",
        ));
    }
    let process_len = usize::try_from(u32_at(16)?)
        .map_err(|_| RusticolError::artifact("recurrence process ID is too large"))?;
    let word_count = usize::try_from(u32_at(20)?)
        .map_err(|_| RusticolError::artifact("recurrence support mask is too large"))?;
    let target_len = usize::try_from(u32_at(24)?)
        .map_err(|_| RusticolError::artifact("recurrence executor target is too large"))?;
    let cpu_feature_count = u32_at(28)?;
    let descriptor_count = u32_at(32)?;
    let catalog_executor_count = u32_at(36)?;
    let backend = ProcessExecutorBackend::decode(
        *bytes
            .get(40)
            .ok_or_else(|| RusticolError::artifact("truncated executor backend"))?,
    )?;
    let portable = match bytes.get(41).copied() {
        Some(0) => false,
        Some(1) => true,
        _ => {
            return Err(RusticolError::artifact(
                "recurrence executor portability is not a canonical boolean",
            ));
        }
    };
    if bytes.get(42) != Some(&64) || bytes.get(43) != Some(&0) {
        return Err(RusticolError::compatibility(
            "recurrence process executors require a 64-bit little-endian target",
        ));
    }
    let counts = (0..11)
        .map(|index| u32_at(44 + index * 4))
        .collect::<RusticolResult<Vec<_>>>()?;
    if counts[5] != catalog_executor_count {
        return Err(RusticolError::integrity(
            "recurrence executor catalog domain disagrees with its process remap",
        ));
    }
    let schedule_digest = read_header_digest(&bytes, 88, "root schedule")?;
    let process_semantic_digest = read_header_digest(&bytes, 120, "process semantic")?;
    let compiled_model_digest = semantic_digest_from_bytes(
        read_header_digest(&bytes, 152, "compiled model")?,
        "compiled model",
    )?;
    let recurrence_template_catalog_digest = semantic_digest_from_bytes(
        read_header_digest(&bytes, 184, "recurrence-template catalog")?,
        "recurrence-template catalog",
    )?;
    let prepared_kernel_pack_digest = semantic_digest_from_bytes(
        read_header_digest(&bytes, 216, "prepared-kernel pack")?,
        "prepared-kernel pack",
    )?;
    let direct_template_catalog_digest = semantic_digest_from_bytes(
        read_header_digest(&bytes, 248, "direct-template catalog")?,
        "direct-template catalog",
    )?;
    let runtime_layout_digest = semantic_digest_from_bytes(
        read_header_digest(&bytes, 280, "runtime layout")?,
        "runtime layout",
    )?;
    let process_digest = read_header_digest(&bytes, 312, "process")?;
    if schedule_digest != decode_sha256(&binding.schedule_digest)?
        || process_digest != decode_sha256(&binding.process_digest)?
        || process_semantic_digest != decode_sha256(&binding.process_semantic_digest)?
        || compiled_model_digest.to_string() != spec.compiled_model_digest
        || recurrence_template_catalog_digest.to_string() != spec.recurrence_template_catalog_digest
        || prepared_kernel_pack_digest.to_string() != spec.prepared_kernel_pack_digest
        || direct_template_catalog_digest.to_string() != spec.direct_template_catalog_digest
    {
        return Err(RusticolError::integrity(
            "recurrence process-executor identities disagree with execution.json",
        ));
    }
    let process_start = RECURRENCE_PROCESS_BINDING_HEADER_SIZE_V4;
    let process_end = process_start
        .checked_add(process_len)
        .ok_or_else(|| RusticolError::artifact("recurrence process binding overflows"))?;
    let target_end = process_end
        .checked_add(target_len)
        .ok_or_else(|| RusticolError::artifact("recurrence executor target range overflows"))?;
    if bytes.get(process_start..process_end) != Some(spec.process_key.as_bytes()) || word_count == 0
    {
        return Err(RusticolError::integrity(
            "recurrence process-binding payload is inconsistent",
        ));
    }
    let target_triple =
        std::str::from_utf8(bytes.get(process_end..target_end).ok_or_else(|| {
            RusticolError::artifact("truncated recurrence process-executor target")
        })?)
        .map_err(|_| RusticolError::artifact("recurrence executor target is not UTF-8"))?
        .to_owned();
    if target_triple.is_empty() {
        return Err(RusticolError::artifact(
            "recurrence process-executor target is empty",
        ));
    }
    let mut cursor = target_end;
    let cpu_features = (0..cpu_feature_count)
        .map(|_| read_string(&bytes, &mut cursor, "executor CPU feature"))
        .collect::<RusticolResult<Vec<_>>>()?;
    let target = ProcessExecutorTarget {
        backend,
        target_triple,
        portable,
        cpu_features,
    };
    target.validate(&artifact.manifest().producer.target)?;
    let words = read_u64_values(
        &bytes,
        &mut cursor,
        u32::try_from(word_count)
            .map_err(|_| RusticolError::artifact("recurrence support mask is too large"))?,
        "recurrence support mask",
    )?;
    if words != binding.process_support_words {
        return Err(RusticolError::integrity(
            "recurrence process support mask is inconsistent",
        ));
    }
    let source_slots = read_u32_rows(&bytes, &mut cursor, counts[0], 1, "source-slot remap")?
        .into_iter()
        .map(|row| row[0])
        .collect();
    let source_momentum_signs =
        read_i32_values(&bytes, &mut cursor, counts[0], "source momentum signs")?;
    let source_helicity_signs =
        read_i32_values(&bytes, &mut cursor, counts[0], "source helicity signs")?;
    let source_state_offsets = read_u32_rows(
        &bytes,
        &mut cursor,
        counts[0].checked_add(1).ok_or_else(|| {
            RusticolError::artifact("recurrence source-state offset count overflows")
        })?,
        1,
        "source-state remap offsets",
    )?
    .into_iter()
    .map(|row| row[0])
    .collect::<Vec<_>>();
    let source_state_count = source_state_offsets
        .last()
        .copied()
        .ok_or_else(|| RusticolError::artifact("recurrence source-state remap has no offsets"))?;
    let source_state_indices = read_u32_rows(
        &bytes,
        &mut cursor,
        source_state_count,
        1,
        "source-state remap indices",
    )?
    .into_iter()
    .map(|row| row[0])
    .collect();
    let public_flow_ids = read_u32_rows(&bytes, &mut cursor, counts[1], 1, "public-flow remap")?
        .into_iter()
        .map(|row| row[0])
        .collect();
    let physical_sector_ids =
        read_u32_rows(&bytes, &mut cursor, counts[2], 1, "physical-sector remap")?
            .into_iter()
            .map(|row| row[0])
            .collect();
    let state_templates = RecurrenceSparseBijection {
        count: counts[3],
        changes: read_u32_pairs(&bytes, &mut cursor, counts[7], "state-template remap")?,
    };
    let source_templates = RecurrenceSparseBijection {
        count: counts[4],
        changes: read_u32_pairs(&bytes, &mut cursor, counts[8], "source-template remap")?,
    };
    let direct_executors = RecurrenceSparseBijection {
        count: counts[5],
        changes: read_u32_pairs(&bytes, &mut cursor, counts[9], "direct-executor remap")?,
    };
    let parameter_slots = RecurrenceSparseBijection {
        count: counts[6],
        changes: read_u32_pairs(&bytes, &mut cursor, counts[10], "parameter-slot remap")?,
    };
    let decoded = RecurrenceProcessRemap {
        bijection_digest: binding.remap.bijection_digest.clone(),
        source_slots,
        source_momentum_signs,
        source_helicity_signs,
        source_state_offsets,
        source_state_indices,
        public_flow_ids,
        physical_sector_ids,
        state_templates,
        source_templates,
        direct_executors,
        parameter_slots,
    };
    if decoded != binding.remap {
        return Err(RusticolError::integrity(
            "recurrence process-binding binary remap disagrees with execution.json",
        ));
    }
    let process_executors = decode_process_executor_descriptors(
        &bytes,
        &mut cursor,
        descriptor_count,
        catalog_executor_count,
        target,
        ProcessExecutorIdentities {
            compiled_model_digest,
            recurrence_template_catalog_digest,
            prepared_kernel_pack_digest,
            direct_template_catalog_digest,
            runtime_layout_digest,
        },
    )?;
    if cursor != bytes.len() {
        return Err(RusticolError::integrity(
            "recurrence process-binding payload has trailing bytes",
        ));
    }
    Ok(LoadedRecurrenceProcessBinding {
        remap: decoded,
        process_executors,
    })
}

fn read_header_digest(bytes: &[u8], offset: usize, context: &str) -> RusticolResult<[u8; 32]> {
    bytes
        .get(offset..offset + 32)
        .ok_or_else(|| RusticolError::artifact(format!("truncated {context} digest")))?
        .try_into()
        .map_err(|_| RusticolError::internal("process-binding digest width drifted"))
}

fn read_u32_pairs(
    bytes: &[u8],
    cursor: &mut usize,
    count: u32,
    context: &str,
) -> RusticolResult<Vec<[u32; 2]>> {
    read_u32_rows(bytes, cursor, count, 2, context)?
        .into_iter()
        .map(|row| {
            row.try_into()
                .map_err(|_| RusticolError::internal("u32 pair width drifted"))
        })
        .collect()
}

fn read_u32_rows(
    bytes: &[u8],
    cursor: &mut usize,
    row_count: u32,
    row_width: usize,
    context: &str,
) -> RusticolResult<Vec<Vec<u32>>> {
    let value_count = usize::try_from(row_count)
        .ok()
        .and_then(|count| count.checked_mul(row_width))
        .ok_or_else(|| RusticolError::artifact(format!("{context} count overflows")))?;
    let byte_count = value_count
        .checked_mul(4)
        .ok_or_else(|| RusticolError::artifact(format!("{context} byte size overflows")))?;
    let end = cursor
        .checked_add(byte_count)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range overflows")))?;
    let payload = bytes
        .get(*cursor..end)
        .ok_or_else(|| RusticolError::artifact(format!("truncated {context}")))?;
    *cursor = end;
    payload
        .chunks_exact(row_width * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|raw| {
                    raw.try_into()
                        .map(u32::from_le_bytes)
                        .map_err(|_| RusticolError::artifact(format!("truncated {context}")))
                })
                .collect()
        })
        .collect()
}

fn read_i32_values(
    bytes: &[u8],
    cursor: &mut usize,
    count: u32,
    context: &str,
) -> RusticolResult<Vec<i32>> {
    let count = usize::try_from(count)
        .map_err(|_| RusticolError::artifact(format!("{context} count exceeds usize")))?;
    let byte_count = count
        .checked_mul(4)
        .ok_or_else(|| RusticolError::artifact(format!("{context} byte size overflows")))?;
    let end = cursor
        .checked_add(byte_count)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range overflows")))?;
    let payload = bytes
        .get(*cursor..end)
        .ok_or_else(|| RusticolError::artifact(format!("truncated {context}")))?;
    *cursor = end;
    payload
        .chunks_exact(4)
        .map(|raw| {
            raw.try_into()
                .map(i32::from_le_bytes)
                .map_err(|_| RusticolError::artifact(format!("truncated {context}")))
        })
        .collect()
}

fn read_u64_values(
    bytes: &[u8],
    cursor: &mut usize,
    count: u32,
    context: &str,
) -> RusticolResult<Vec<u64>> {
    let count = usize::try_from(count)
        .map_err(|_| RusticolError::artifact(format!("{context} count exceeds usize")))?;
    let byte_count = count
        .checked_mul(8)
        .ok_or_else(|| RusticolError::artifact(format!("{context} byte size overflows")))?;
    let end = cursor
        .checked_add(byte_count)
        .ok_or_else(|| RusticolError::artifact(format!("{context} range overflows")))?;
    let payload = bytes
        .get(*cursor..end)
        .ok_or_else(|| RusticolError::artifact(format!("truncated {context}")))?;
    *cursor = end;
    payload
        .chunks_exact(8)
        .map(|raw| {
            raw.try_into()
                .map(u64::from_le_bytes)
                .map_err(|_| RusticolError::artifact(format!("truncated {context}")))
        })
        .collect()
}

fn apply_process_remap(
    plan: DirectRecurrencePlan,
    remap: &RecurrenceProcessRemap,
) -> RusticolResult<DirectRecurrencePlan> {
    let source_count = usize::try_from(plan.external_source_count())
        .map_err(|_| RusticolError::artifact("recurrence source count exceeds usize"))?;
    let topology_flow_count = match plan.strategy() {
        crate::recurrence::RecurrenceStrategy::TopologyReplay => Some(plan.replay_targets().len()),
        crate::recurrence::RecurrenceStrategy::AllFlowUnion
        | crate::recurrence::RecurrenceStrategy::ContractedColorUnion => None,
    };
    if remap.source_slots.len() != source_count
        || remap.source_momentum_signs.len() != source_count
        || remap.source_helicity_signs.len() != source_count
        || topology_flow_count.is_some_and(|flow_count| remap.public_flow_ids.len() != flow_count)
        || remap.physical_sector_ids.len() != plan.physical_sector_count() as usize
        || remap.state_templates.count != plan.state_template_count()
        || remap.source_templates.count != plan.source_template_count()
        || remap.direct_executors.count != plan.direct_executor_count()
        || remap.parameter_slots.count != plan.parameter_value_count()
    {
        return Err(RusticolError::integrity(
            "recurrence process remap domain counts disagree with the root schedule",
        ));
    }
    let identity = remap
        .source_slots
        .iter()
        .enumerate()
        .all(|(index, value)| *value as usize == index)
        && remap.source_momentum_signs.iter().all(|value| *value == 1)
        && remap.source_helicity_signs.iter().all(|value| *value == 1)
        && remap
            .source_state_offsets
            .windows(2)
            .enumerate()
            .all(|(source_slot, window)| {
                let start = window[0] as usize;
                let stop = window[1] as usize;
                remap.source_state_indices[start..stop]
                    .iter()
                    .enumerate()
                    .all(|(state_index, value)| *value as usize == state_index)
                    && source_slot < remap.source_slots.len()
            })
        && remap
            .public_flow_ids
            .iter()
            .enumerate()
            .all(|(index, value)| *value as usize == index)
        && remap
            .physical_sector_ids
            .iter()
            .enumerate()
            .all(|(index, value)| *value as usize == index)
        && remap.state_templates.changes.is_empty()
        && remap.source_templates.changes.is_empty()
        && remap.direct_executors.changes.is_empty()
        && remap.parameter_slots.changes.is_empty();
    if identity {
        return Ok(plan);
    }
    if plan.strategy() != crate::recurrence::RecurrenceStrategy::TopologyReplay {
        return Err(RusticolError::compatibility(
            "cross-process recurrence sharing currently supports topology-replay schedules only",
        ));
    }
    if !remap.parameter_slots.changes.is_empty() {
        return Err(RusticolError::compatibility(
            "cross-process recurrence sharing does not support prepared-parameter reordering",
        ));
    }

    let mut parts = plan.into_parts();
    for current in &mut parts.currents {
        current.state_template_id =
            sparse_bijection_value(&remap.state_templates, current.state_template_id)?;
    }
    for source in &mut parts.sources {
        source.source_slot = *remap
            .source_slots
            .get(source.source_slot as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence source row is outside the process source remap",
                )
            })?;
        source.source_template_or_dispatch_domain = sparse_bijection_value(
            &remap.source_templates,
            source.source_template_or_dispatch_domain,
        )?;
    }
    for group in &mut parts.row_groups {
        group.direct_executor_id =
            sparse_bijection_value(&remap.direct_executors, group.direct_executor_id)?;
    }
    for target in &mut parts.replay_targets {
        target.public_flow_id = *remap
            .public_flow_ids
            .get(target.public_flow_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence replay target is outside the public-flow remap",
                )
            })?;
        let start = usize::try_from(target.source_permutation_start).map_err(|_| {
            RusticolError::artifact("recurrence replay permutation offset exceeds usize")
        })?;
        let stop = start
            .checked_add(target.source_permutation_count as usize)
            .ok_or_else(|| {
                RusticolError::artifact("recurrence replay permutation range overflows")
            })?;
        let permutations = parts
            .source_permutations
            .get_mut(start..stop)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence replay permutation is outside the root schedule",
                )
            })?;
        let momentum_signs = parts
            .replay_momentum_signs
            .get_mut(start..stop)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence replay momentum signs are outside the root schedule",
                )
            })?;
        for (source_slot, momentum_sign) in permutations.iter_mut().zip(momentum_signs.iter_mut()) {
            let root_slot = *source_slot as usize;
            *source_slot = *remap.source_slots.get(root_slot).ok_or_else(|| {
                RusticolError::integrity("recurrence replay source is outside the process remap")
            })?;
            *momentum_sign = momentum_sign
                .checked_mul(*remap.source_momentum_signs.get(root_slot).ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence replay source has no process momentum sign",
                    )
                })?)
                .ok_or_else(|| {
                    RusticolError::integrity("recurrence replay momentum sign overflows")
                })?;
        }
    }
    for descriptor in &parts.resolved_helicities {
        let start = usize::try_from(descriptor.public_helicity_start).map_err(|_| {
            RusticolError::artifact("recurrence public-helicity offset exceeds usize")
        })?;
        let stop = start
            .checked_add(descriptor.public_helicity_count as usize)
            .ok_or_else(|| RusticolError::artifact("recurrence public-helicity range overflows"))?;
        let source = parts
            .public_helicities
            .get(start..stop)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence public-helicity vector is outside the root schedule",
                )
            })?
            .to_vec();
        if source.len() != source_count {
            return Err(RusticolError::integrity(
                "recurrence public-helicity width disagrees with the process remap",
            ));
        }
        let target = parts
            .public_helicities
            .get_mut(start..stop)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence public-helicity vector is outside the root schedule",
                )
            })?;
        for (root_slot, value) in source.into_iter().enumerate() {
            let target_slot = remap.source_slots[root_slot] as usize;
            target[target_slot] = value
                .checked_mul(remap.source_helicity_signs[root_slot])
                .ok_or_else(|| RusticolError::integrity("recurrence public helicity overflows"))?;
        }
    }
    for descriptor in &parts.resolved_helicities {
        let start = usize::try_from(descriptor.source_state_start).map_err(|_| {
            RusticolError::artifact("recurrence source-state assignment offset exceeds usize")
        })?;
        let stop = start
            .checked_add(descriptor.source_state_count as usize)
            .ok_or_else(|| {
                RusticolError::artifact("recurrence source-state assignment range overflows")
            })?;
        let source = parts
            .source_state_assignments
            .get(start..stop)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence source-state assignment range is outside the root schedule",
                )
            })?
            .to_vec();
        if source.len() != source_count {
            return Err(RusticolError::integrity(
                "recurrence source-state assignment width disagrees with the process remap",
            ));
        }
        let target = parts
            .source_state_assignments
            .get_mut(start..stop)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence source-state assignment range is outside the root schedule",
                )
            })?;
        for assignment in source {
            let target_slot = *remap
                .source_slots
                .get(assignment.source_slot as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence source-state assignment is outside the process remap",
                    )
                })?;
            let state_start = *remap
                .source_state_offsets
                .get(assignment.source_slot as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence source-state assignment has no remap offset",
                    )
                })? as usize;
            let state_stop = *remap
                .source_state_offsets
                .get(assignment.source_slot as usize + 1)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence source-state assignment has no remap limit",
                    )
                })? as usize;
            let target_state_index = *remap
                .source_state_indices
                .get(state_start..state_stop)
                .and_then(|values| values.get(assignment.state_index as usize))
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence source-state assignment is outside its state remap",
                    )
                })?;
            target[target_slot as usize] = crate::recurrence::DirectSourceStateAssignment {
                source_slot: target_slot,
                state_index: target_state_index,
            };
        }
    }
    DirectRecurrencePlan::new(parts)
}

fn sparse_bijection_value(mapping: &RecurrenceSparseBijection, value: u32) -> RusticolResult<u32> {
    if value >= mapping.count {
        return Err(RusticolError::integrity(
            "recurrence process remap input is outside its domain",
        ));
    }
    Ok(mapping
        .changes
        .binary_search_by_key(&value, |change| change[0])
        .map(|index| mapping.changes[index][1])
        .unwrap_or(value))
}

type RecurrenceCommonRuntimeParts = (
    ExecutionRuntime,
    Vec<crate::EagerComplex64>,
    Vec<PreparedParameterProjectionEntry>,
    Vec<DirectSourceDispatchDomainSpec>,
);

fn recurrence_runtime_parameters(
    metadata: &RecurrenceRuntimeMetadata,
) -> Vec<GenericRuntimeModelParameterManifest> {
    recurrence_runtime_parameters_from(&metadata.runtime_parameters)
}

fn recurrence_runtime_parameters_from(
    parameters: &[RecurrenceRuntimeParameter],
) -> Vec<GenericRuntimeModelParameterManifest> {
    parameters
        .iter()
        .map(|parameter| GenericRuntimeModelParameterManifest {
            name: parameter.name.clone(),
            kind: parameter.kind.clone(),
            parameter_index: parameter.parameter_index as usize,
            default: parameter.default,
            pdg: None,
            runtime_name: parameter.runtime_name.clone(),
            complex_component: parameter.complex_component.clone(),
        })
        .collect()
}

#[derive(Clone, Copy)]
struct RecurrenceNormalizationValues {
    factor: f64,
    color_factor: f64,
}

fn recurrence_normalization_values(
    metadata: &RecurrenceRuntimeMetadata,
) -> RusticolResult<RecurrenceNormalizationValues> {
    recurrence_normalization_values_from(
        &metadata.normalization,
        metadata.color_contraction.as_ref(),
    )
}

fn recurrence_normalization_values_from(
    normalization: &RecurrenceNormalization,
    color_contraction: Option<&RecurrenceColorContractionReference>,
) -> RusticolResult<RecurrenceNormalizationValues> {
    if !normalization.couplings_in_stage_evaluators {
        return Err(RusticolError::compatibility(
            "recurrence execution requires local vertex couplings in prepared kernel calls",
        ));
    }
    let color_factor =
        if color_contraction.is_some_and(|contraction| contraction.includes_color_factor) {
            1.0
        } else {
            normalization.color_factor
        };
    let factor = color_factor * normalization.global_coupling_factor
        / (normalization.average_factor * normalization.identical_factor);
    if !factor.is_finite() {
        return Err(RusticolError::integrity(
            "recurrence runtime normalization is not finite",
        ));
    }
    Ok(RecurrenceNormalizationValues {
        factor,
        color_factor,
    })
}

#[cfg(feature = "on-the-fly-test-support")]
fn on_the_fly_prepared_parameters(
    metadata: &RecurrenceRuntimeMetadata,
    runtime_parameters: &[GenericRuntimeModelParameterManifest],
    mut model_parameter_evaluator: Option<ModelParameterEvaluatorRuntime>,
    overrides: &BTreeMap<String, [f64; 2]>,
) -> RusticolResult<Vec<[f64; 2]>> {
    let runtime_slots = runtime_parameter_slots(runtime_parameters)?;
    let mut runtime_values = runtime_parameters
        .iter()
        .map(|parameter| parameter.default)
        .collect::<Vec<_>>();
    for value in &runtime_values {
        if !value.is_finite() {
            return Err(RusticolError::integrity(
                "recurrence runtime parameter default is not finite",
            ));
        }
    }
    for value in &metadata.prepared_parameter_defaults {
        if !value[0].is_finite() || !value[1].is_finite() {
            return Err(RusticolError::integrity(
                "recurrence prepared parameter default is not finite",
            ));
        }
    }
    for (name, override_value) in overrides {
        if !override_value[0].is_finite() || !override_value[1].is_finite() {
            return Err(RusticolError::invalid_argument(format!(
                "on-the-fly parameter override {name:?} is not finite"
            )));
        }
        if runtime_parameters.iter().any(|parameter| {
            parameter.kind == "derived_parameter_component"
                && parameter.runtime_name.as_deref().unwrap_or(&parameter.name) == name
        }) {
            return Err(RusticolError::invalid_argument(format!(
                "derived on-the-fly parameter {name:?} is immutable"
            )));
        }
        let Some(slots) = runtime_slots.get(name).copied() else {
            return Err(RusticolError::invalid_argument(format!(
                "on-the-fly parameter override {name:?} is not used by the process"
            )));
        };
        let real = runtime_values.get_mut(slots.real).ok_or_else(|| {
            RusticolError::integrity("on-the-fly runtime parameter real slot is absent")
        })?;
        *real = override_value[0];
        if let Some(imaginary_slot) = slots.imaginary {
            let imaginary = runtime_values.get_mut(imaginary_slot).ok_or_else(|| {
                RusticolError::integrity("on-the-fly runtime parameter imaginary slot is absent")
            })?;
            *imaginary = override_value[1];
        } else if override_value[1] != 0.0 {
            return Err(RusticolError::invalid_argument(format!(
                "real on-the-fly parameter {name:?} cannot receive an imaginary value"
            )));
        }
    }
    super::model_parameters::refresh_derived_model_parameter_values(
        model_parameter_evaluator.as_mut(),
        &mut runtime_values,
    )?;

    let defaults = metadata
        .prepared_parameter_defaults
        .iter()
        .map(|[real, imaginary]| crate::EagerComplex64::new(*real, *imaginary))
        .collect::<Vec<_>>();
    let projection = metadata
        .parameter_projection
        .iter()
        .filter_map(|row| {
            row.prepared_parameter_id
                .map(|prepared_slot| (row, prepared_slot))
        })
        .map(|(row, prepared_slot)| {
            Ok(PreparedParameterProjectionEntry {
                runtime_slot: usize::try_from(row.runtime_slot).map_err(|_| {
                    RusticolError::integrity("on-the-fly runtime parameter slot exceeds usize")
                })?,
                prepared_slot: usize::try_from(prepared_slot).map_err(|_| {
                    RusticolError::integrity("on-the-fly prepared parameter slot exceeds usize")
                })?,
                component: u8::try_from(row.component).map_err(|_| {
                    RusticolError::integrity("on-the-fly parameter component exceeds u8")
                })?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    projected_prepared_parameter_values(&defaults, &projection, &runtime_values).map(|values| {
        values
            .into_iter()
            .map(|(real, imaginary)| [real, imaginary])
            .collect()
    })
}

#[cfg(feature = "on-the-fly-test-support")]
pub(super) fn on_the_fly_source_major_momenta(
    seed: &OnTheFlyProcessSeedV1,
    point_major: &[f64],
    point_count: u32,
    lorentz_component_count: u16,
) -> RusticolResult<Vec<f64>> {
    let mut source_major = Vec::new();
    let output_len = on_the_fly_source_major_momenta_into(
        seed,
        point_major,
        point_count,
        lorentz_component_count,
        &mut source_major,
    )?;
    source_major.truncate(output_len);
    Ok(source_major)
}

/// Transpose public point-major momenta into the source-major plane layout
/// without reallocating once `source_major` has reached the warmed capacity.
/// The returned length identifies the initialized prefix; callers may retain
/// a larger backing allocation for later batches.
pub(super) fn on_the_fly_source_major_momenta_into(
    seed: &OnTheFlyProcessSeedV1,
    point_major: &[f64],
    point_count: u32,
    lorentz_component_count: u16,
    source_major: &mut Vec<f64>,
) -> RusticolResult<usize> {
    let permutation = seed.external_permutation();
    let point_count_usize = point_count as usize;
    let lorentz_count = usize::from(lorentz_component_count);
    let point_width = permutation
        .len()
        .checked_mul(lorentz_count)
        .ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly momentum shape exceeds usize")
        })?;
    let expected = point_count_usize.checked_mul(point_width).ok_or_else(|| {
        RusticolError::invalid_argument("on-the-fly momentum shape exceeds usize")
    })?;
    if point_count == 0 || point_major.len() != expected {
        return Err(RusticolError::invalid_argument(format!(
            "on-the-fly point-major momenta contain {} scalars, expected {expected}",
            point_major.len()
        )));
    }
    if source_major.len() < expected {
        source_major
            .try_reserve_exact(expected - source_major.len())
            .map_err(|error| {
                RusticolError::invalid_argument(format!(
                    "on-the-fly momentum workspace allocation failed: {error}"
                ))
            })?;
        source_major.resize(expected, 0.0);
    }
    for (source_slot, public_slot) in permutation.iter().copied().enumerate() {
        let public_slot = usize::try_from(public_slot).map_err(|_| {
            RusticolError::integrity("on-the-fly public momentum slot exceeds usize")
        })?;
        if public_slot >= permutation.len() {
            return Err(RusticolError::integrity(
                "on-the-fly public momentum slot is out of bounds",
            ));
        }
        for lorentz in 0..lorentz_count {
            // The checked total shape bounds every plane offset. Walk the
            // strided input and contiguous destination directly instead of
            // repeating checked index arithmetic for every scalar.
            let input_offset = public_slot * lorentz_count + lorentz;
            let output_offset = (source_slot * lorentz_count + lorentz) * point_count_usize;
            let output = &mut source_major[output_offset..output_offset + point_count_usize];
            for (value, source) in output
                .iter_mut()
                .zip(point_major[input_offset..].iter().step_by(point_width))
            {
                *value = *source;
            }
        }
    }
    Ok(expected)
}

#[cfg(feature = "on-the-fly-test-support")]
struct OnTheFlyQueryTraceCacheV1<T> {
    entries: BTreeMap<SemanticDigest, T>,
    trace_build_count: u32,
    trace_cache_hit_count: std::cell::Cell<u32>,
}

#[cfg(feature = "on-the-fly-test-support")]
impl<T> Default for OnTheFlyQueryTraceCacheV1<T> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            trace_build_count: 0,
            trace_cache_hit_count: std::cell::Cell::new(0),
        }
    }
}

#[cfg(feature = "on-the-fly-test-support")]
impl<T> OnTheFlyQueryTraceCacheV1<T> {
    fn insert_built(&mut self, query_digest: SemanticDigest, trace: T) -> RusticolResult<()> {
        match self.entries.entry(query_digest) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(trace);
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(RusticolError::integrity(
                    "on-the-fly trace cache repeated a fresh query digest",
                ));
            }
        }
        self.trace_build_count = self
            .trace_build_count
            .checked_add(1)
            .ok_or_else(|| RusticolError::integrity("on-the-fly trace-build counter overflowed"))?;
        Ok(())
    }

    fn prepared(&self, query_digest: SemanticDigest) -> RusticolResult<&T> {
        self.entries.get(&query_digest).ok_or_else(|| {
            RusticolError::integrity("fresh on-the-fly trace disappeared from its cache")
        })
    }

    fn prepared_mut(&mut self, query_digest: SemanticDigest) -> RusticolResult<&mut T> {
        self.entries.get_mut(&query_digest).ok_or_else(|| {
            RusticolError::integrity("fresh on-the-fly trace disappeared from its cache")
        })
    }

    fn lookup(&self, query_digest: SemanticDigest) -> RusticolResult<&T> {
        let trace = self.entries.get(&query_digest).ok_or_else(|| {
            RusticolError::integrity("on-the-fly benchmark query disappeared from its trace cache")
        })?;
        self.trace_cache_hit_count.set(
            self.trace_cache_hit_count
                .get()
                .checked_add(1)
                .ok_or_else(|| {
                    RusticolError::integrity("on-the-fly trace-cache-hit counter overflowed")
                })?,
        );
        Ok(trace)
    }

    fn counts(&self) -> (u32, u32) {
        (self.trace_build_count, self.trace_cache_hit_count.get())
    }
}

#[cfg(feature = "on-the-fly-test-support")]
fn validate_on_the_fly_probe_outputs(
    raw_amplitudes: &[[f64; 2]],
    normalized_values: &[f64],
) -> RusticolResult<()> {
    if raw_amplitudes.len() != normalized_values.len() {
        return Err(RusticolError::integrity(
            "on-the-fly probe output lengths disagree",
        ));
    }
    for (point, ([real, imaginary], normalized)) in
        raw_amplitudes.iter().zip(normalized_values).enumerate()
    {
        if !real.is_finite() || !imaginary.is_finite() {
            return Err(RusticolError::evaluation(format!(
                "on-the-fly probe produced a non-finite raw amplitude at point {point}"
            )));
        }
        if !normalized.is_finite() {
            return Err(RusticolError::evaluation(format!(
                "on-the-fly probe produced a non-finite normalized value at point {point}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "on-the-fly-test-support")]
fn validate_on_the_fly_family_execution(
    census: crate::recurrence::OnTheFlyQueryFamilyCensusV1,
    report: crate::recurrence::on_the_fly::OnTheFlyQueryFamilyExecutionReportV1,
    require_cache_hit: bool,
) -> RusticolResult<()> {
    if report.cache_hit != require_cache_hit
        || report.source_calls != census.union_source_executor_call_groups
        || report.source_rows != census.union_source_rows
        || report.contribution_calls != census.union_contribution_executor_call_groups
        || report.contribution_rows != census.union_contribution_rows
        || report.finalization_calls != census.union_finalization_executor_call_groups
        || report.finalization_rows != census.union_finalization_rows
        || report.closure_calls != census.union_closure_executor_call_groups
        || report.closure_rows != census.union_closure_rows
    {
        return Err(RusticolError::integrity(
            "on-the-fly family execution differs from its prepared census/cache state",
        ));
    }
    Ok(())
}

#[cfg(feature = "on-the-fly-test-support")]
#[allow(clippy::too_many_arguments)]
fn execute_on_the_fly_query_family_v1(
    artifact: &VerifiedArtifact,
    manifest: &RecurrenceExecutionManifest,
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    direct: &PreparedDirectExecutorCatalog,
    selected_public_flow_id: u32,
    public_helicities: &[i32],
    query_family: &[(u32, Vec<i32>)],
    point_major_external_momenta: &[f64],
    point_count: u32,
    parameter_overrides: &BTreeMap<String, [f64; 2]>,
    benchmark: bool,
    benchmark_warmup_repetitions: u32,
    benchmark_repetitions: u32,
    enable_color_projection: bool,
) -> RusticolResult<NativeOnTheFlyArtifactProbeV1> {
    if query_family.is_empty() {
        return Err(RusticolError::invalid_argument(
            "on-the-fly query family must not be empty",
        ));
    }
    if query_family[0].0 != selected_public_flow_id
        || query_family[0].1.as_slice() != public_helicities
    {
        return Err(RusticolError::invalid_argument(
            "on-the-fly query family must begin with the required single selector",
        ));
    }
    if benchmark && benchmark_warmup_repetitions < 2 {
        return Err(RusticolError::invalid_argument(
            "on-the-fly family benchmark requires at least two warmup repetitions",
        ));
    }

    let selected_with_seed = query_family
        .iter()
        .map(|(flow, helicities)| {
            build_on_the_fly_selected_trace_v1(
                authenticated,
                direct,
                *flow,
                helicities,
                enable_color_projection,
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let seed = selected_with_seed
        .first()
        .ok_or_else(|| RusticolError::integrity("on-the-fly family compact seed is absent"))?
        .seed
        .clone();
    let seed_digest = seed.semantic_digest();
    let mut selected = selected_with_seed
        .into_iter()
        .map(|selected| {
            if selected.seed.semantic_digest() != seed_digest
                || selected.query.process_seed_digest() != seed_digest
                || selected.trace.seed_digest() != seed_digest
            {
                return Err(RusticolError::integrity(
                    "on-the-fly query family does not share one compact process identity",
                ));
            }
            Ok(OnTheFlySelectedQueryTraceV1 {
                query: selected.query,
                trace: selected.trace,
                projection: selected.projection,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let (pack_json, pack, payload_root) = load_prepared_pack(artifact, manifest)?;
    let payloads = artifact.evaluator_payload_store(&payload_root)?;
    let runtime_parameters = recurrence_runtime_parameters(&manifest.runtime_metadata);
    let model_parameter_evaluator =
        super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
            &pack,
            &runtime_parameters,
            &payloads,
        )?;
    let pool = NativeRecurrencePreparedExecutorPool::load_from_store(
        &pack_json,
        &payloads,
        &manifest.prepared_kernel_pack_digest,
        &manifest.direct_template_catalog_digest,
    )?;
    let source_domains = build_on_the_fly_source_domains(&seed, authenticated.template())?;
    let sources = pool.bind_source_domains(source_domains)?;
    let mut resolver = pool.into_on_the_fly_resolver(sources);
    resolver.bind_on_the_fly_family(authenticated.template(), direct, &seed, &mut selected)?;
    let semantic_executor_binding_count = resolver.semantic_executor_binding_count()?;
    let distinct_prepared_executor_count = resolver.distinct_prepared_executor_count()?;
    let trace_inputs = selected
        .iter()
        .map(|selected| QueryFamilyTraceInput {
            trace: &selected.trace,
            projection: selected.projection,
        })
        .collect::<Vec<_>>();
    let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
    let prepare_started = Instant::now();
    if executor.prepare(direct, &trace_inputs, point_count)? {
        return Err(RusticolError::integrity(
            "fresh on-the-fly family executor reported a cold cache hit",
        ));
    }
    let cold_prepare_seconds = profile_duration_seconds(prepare_started.elapsed());
    let census = executor
        .prepared_census()
        .ok_or_else(|| RusticolError::integrity("prepared on-the-fly family census is absent"))?;
    let prepared_parameters = on_the_fly_prepared_parameters(
        &manifest.runtime_metadata,
        &runtime_parameters,
        model_parameter_evaluator,
        parameter_overrides,
    )?
    .into_iter()
    .map(|value| (value[0], value[1]))
    .collect::<Vec<_>>();
    executor.set_parameters(&prepared_parameters)?;
    let source_major = on_the_fly_source_major_momenta(
        &seed,
        point_major_external_momenta,
        point_count,
        selected[0].trace.layout().lorentz_component_count(),
    )?;
    let output_len = query_family
        .len()
        .checked_mul(point_count as usize)
        .ok_or_else(|| RusticolError::invalid_argument("on-the-fly family output exceeds usize"))?;
    let mut outputs = vec![(0.0, 0.0); output_len];
    let mut execution_count = 0_u32;
    let mut last_report = None;
    let benchmark_elapsed_seconds = if benchmark {
        for warmup in 0..benchmark_warmup_repetitions {
            let report = executor.execute_into(&source_major, point_count, &mut outputs)?;
            validate_on_the_fly_family_execution(census, report, warmup != 0)?;
            execution_count = execution_count.checked_add(1).ok_or_else(|| {
                RusticolError::integrity("on-the-fly family execution count exceeds u32")
            })?;
            last_report = Some(report);
            std::hint::black_box(&outputs);
        }
        let started = Instant::now();
        for _ in 0..benchmark_repetitions {
            let report = executor.execute_into(&source_major, point_count, &mut outputs)?;
            validate_on_the_fly_family_execution(census, report, true)?;
            execution_count = execution_count.checked_add(1).ok_or_else(|| {
                RusticolError::integrity("on-the-fly family execution count exceeds u32")
            })?;
            last_report = Some(report);
            std::hint::black_box(&outputs);
        }
        Some(profile_duration_seconds(started.elapsed()))
    } else {
        let report = executor.execute_into(&source_major, point_count, &mut outputs)?;
        validate_on_the_fly_family_execution(census, report, false)?;
        execution_count = 1;
        last_report = Some(report);
        None
    };
    let report = last_report.ok_or_else(|| {
        RusticolError::integrity("on-the-fly family produced no execution report")
    })?;
    let normalization_factor = recurrence_normalization_values(&manifest.runtime_metadata)?.factor;
    let mut query_reports = Vec::new();
    for ((flow, helicities), (selected_query, amplitudes)) in query_family.iter().zip(
        selected
            .iter()
            .zip(outputs.chunks_exact(point_count as usize)),
    ) {
        let raw_amplitudes = amplitudes
            .iter()
            .map(|&(real, imaginary)| [real, imaginary])
            .collect::<Vec<_>>();
        let normalized_values = raw_amplitudes
            .iter()
            .map(|[real, imaginary]| {
                real.mul_add(*real, imaginary * imaginary) * normalization_factor
            })
            .collect::<Vec<_>>();
        validate_on_the_fly_probe_outputs(&raw_amplitudes, &normalized_values)?;
        query_reports.push(NativeOnTheFlyFamilyQueryProbeV1 {
            selected_public_flow_id: *flow,
            public_helicities: helicities.clone(),
            query_digest: selected_query.query.semantic_digest().to_string(),
            raw_amplitudes,
            normalized_values,
        });
    }
    let first = query_reports
        .first()
        .ok_or_else(|| RusticolError::integrity("on-the-fly family lost its first query output"))?;
    let first_raw_amplitudes = first.raw_amplitudes.clone();
    let first_normalized_values = first.normalized_values.clone();
    let benchmark_seconds_per_point = benchmark_elapsed_seconds
        .map(|elapsed| elapsed / (f64::from(benchmark_repetitions) * f64::from(point_count)));
    let first_selected = &selected[0];
    Ok(NativeOnTheFlyArtifactProbeV1 {
        artifact_id: artifact.manifest().artifact_id.clone(),
        process_id: manifest.key.clone(),
        seed_digest: seed.semantic_digest().to_string(),
        query_digest: first_selected.query.semantic_digest().to_string(),
        trace_digest: first_selected.trace.semantic_digest().to_string(),
        point_count,
        raw_amplitudes: first_raw_amplitudes,
        normalized_values: first_normalized_values,
        normalization_factor,
        query_family: Some(NativeOnTheFlyFamilyProbeV1 {
            queries: query_reports,
            census,
            execution_cache_hit: report.cache_hit,
            execution_source_calls: report.source_calls,
            execution_source_rows: report.source_rows,
            execution_contribution_calls: report.contribution_calls,
            execution_contribution_rows: report.contribution_rows,
            execution_finalization_calls: report.finalization_calls,
            execution_finalization_rows: report.finalization_rows,
            execution_closure_calls: report.closure_calls,
            execution_closure_rows: report.closure_rows,
            cold_prepare_seconds,
            benchmark_warmup_repetitions,
            benchmark_repetitions,
            benchmark_elapsed_seconds,
            benchmark_seconds_per_point,
        }),
        currents: Vec::new(),
        projection_enabled: first_selected.projection.enabled,
        projection_applied: first_selected.projection.applied,
        projection_counts: [
            first_selected.projection.pre,
            first_selected.projection.post,
        ],
        work_census_basis: ON_THE_FLY_FAMILY_WORK_CENSUS_BASIS_V1.to_owned(),
        logical_current_count: census.union_unique_current_count,
        resident_current_count: census.union_unique_current_count,
        resident_current_component_count: census.union_unique_current_component_count,
        source_operation_count: census.union_source_rows,
        contribution_operation_count: census.union_contribution_rows,
        finalization_operation_count: census.union_finalization_rows,
        closure_operation_count: census.union_closure_rows,
        total_kernel_application_count: census.union_kernel_application_count()?,
        semantic_executor_binding_count,
        distinct_prepared_executor_count,
        trace_build_count: census.query_count,
        // The prepared family cache is reported by `query_family`; this path
        // performs no query-trace cache lookup after construction.
        trace_cache_hit_count: 0,
        momentum_fill_count: execution_count,
        benchmark_warmup_repetitions,
        benchmark_repetitions,
        benchmark_elapsed_seconds,
        benchmark_seconds_per_point,
        direct_plan_load_attempts: 0,
        direct_plan_decode_attempts: 0,
        direct_plan_materialization_attempts: 0,
        established_builder_attempts: 0,
    })
}

#[cfg(feature = "on-the-fly-test-support")]
impl NativeRuntime {
    /// Execute one authenticated selector-local recurrence trace directly from
    /// a genuine artifact without loading or decoding its DirectPlan.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn on_the_fly_artifact_probe_v1(
        artifact_path: impl AsRef<Path>,
        process_id: &str,
        authenticated: &AuthenticatedRecurrenceBuilderInput,
        direct: &PreparedDirectExecutorCatalog,
        selected_public_flow_id: u32,
        public_helicities: &[i32],
        point_major_external_momenta: &[f64],
        point_count: u32,
        parameter_overrides: &BTreeMap<String, [f64; 2]>,
        tamper_executor_key: bool,
        benchmark: bool,
        benchmark_warmup_repetitions: u32,
        benchmark_repetitions: u32,
        collect_current_diagnostics: bool,
        enable_color_projection: bool,
        query_family: Option<&[(u32, Vec<i32>)]>,
    ) -> RusticolResult<NativeOnTheFlyArtifactProbeV1> {
        if benchmark && benchmark_repetitions == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly benchmark requires at least one timed repetition",
            ));
        }
        if !benchmark && (benchmark_warmup_repetitions != 0 || benchmark_repetitions != 0) {
            return Err(RusticolError::invalid_argument(
                "on-the-fly benchmark repetitions require benchmark=true",
            ));
        }
        if query_family.is_some() && collect_current_diagnostics {
            return Err(RusticolError::invalid_argument(
                "on-the-fly family probing does not collect per-current diagnostics",
            ));
        }
        if query_family.is_some() && tamper_executor_key {
            return Err(RusticolError::invalid_argument(
                "on-the-fly family probing does not support executor-key tampering",
            ));
        }
        let guard = OnTheFlyForbiddenWorkGuardV1::begin()?;
        let result = (|| {
            let artifact = VerifiedArtifact::open(artifact_path)?;
            let selection = artifact.select_process(Some(process_id))?;
            if selection.remaps_public_process() {
                return Err(RusticolError::invalid_argument(
                    "on-the-fly artifact probing requires a representative process ID",
                ));
            }
            let (loaded, _evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
            let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
            let physics =
                ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
            if physics.process_id != selection.process.id
                || physics.process != selection.process.expression
                || physics.color_accuracy.as_str() != selection.process.color_accuracy
                || physics
                    .external_particles
                    .iter()
                    .map(|particle| particle.pdg)
                    .ne(selection.process.external_pdgs.iter().copied())
            {
                return Err(RusticolError::integrity(format!(
                    "runtime physics payload {:?} does not match process {:?}",
                    selection.process.physics_path, selection.process.id
                )));
            }
            let manifest = match loaded {
                LoadedExecutionManifest::Recurrence(manifest) => manifest,
                LoadedExecutionManifest::Compiled(_)
                | LoadedExecutionManifest::EagerV3(_)
                | LoadedExecutionManifest::OnTheFly(_) => {
                    return Err(RusticolError::compatibility(
                        "on-the-fly artifact probing requires a recurrence artifact",
                    ));
                }
            };
            if authenticated
                .process()
                .semantic_identity()
                .input_digest()
                .to_string()
                != manifest.plan.builder_input_sha256
                || direct.direct_template_catalog_digest().to_string()
                    != manifest.direct_template_catalog_digest
                || authenticated
                    .template()
                    .summary()
                    .prepared_kernel_pack_digest
                    .to_string()
                    != manifest.prepared_kernel_pack_digest
            {
                return Err(RusticolError::integrity(
                    "retained on-the-fly canonical inputs do not belong to the selected artifact",
                ));
            }

            if let Some(query_family) = query_family {
                return execute_on_the_fly_query_family_v1(
                    &artifact,
                    &manifest,
                    authenticated,
                    direct,
                    selected_public_flow_id,
                    public_helicities,
                    query_family,
                    point_major_external_momenta,
                    point_count,
                    parameter_overrides,
                    benchmark,
                    benchmark_warmup_repetitions,
                    benchmark_repetitions,
                    enable_color_projection,
                );
            }

            let mut trace_cache = OnTheFlyQueryTraceCacheV1::default();
            let selected = build_on_the_fly_selected_trace_v1(
                authenticated,
                direct,
                selected_public_flow_id,
                public_helicities,
                enable_color_projection,
            )?;
            let query_digest = selected.query.semantic_digest();
            trace_cache.insert_built(query_digest, selected)?;
            if tamper_executor_key {
                trace_cache
                    .prepared_mut(query_digest)?
                    .trace
                    .tamper_first_prepared_executor_key_for_probe()?;
            }
            let (pack_json, pack, payload_root) = load_prepared_pack(&artifact, &manifest)?;
            let payloads = artifact.evaluator_payload_store(&payload_root)?;
            let runtime_parameters = recurrence_runtime_parameters(&manifest.runtime_metadata);
            let model_parameter_evaluator =
                super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
                    &pack,
                    &runtime_parameters,
                    &payloads,
                )?;
            let pool = NativeRecurrencePreparedExecutorPool::load_from_store(
                &pack_json,
                &payloads,
                &manifest.prepared_kernel_pack_digest,
                &manifest.direct_template_catalog_digest,
            )?;
            let selected = trace_cache.prepared(query_digest)?;
            let source_domains =
                build_on_the_fly_source_domains(&selected.seed, authenticated.template())?;
            let sources = pool.bind_source_domains(source_domains)?;
            let mut resolver = pool.into_on_the_fly_resolver(sources);
            {
                let selected = trace_cache.prepared_mut(query_digest)?;
                resolver.bind_on_the_fly_trace(
                    authenticated.template(),
                    direct,
                    &selected.seed,
                    &mut selected.trace,
                )?;
            }
            let selected = trace_cache.prepared(query_digest)?;
            let work_census = selected.trace.execution_work_census()?;
            let semantic_executor_binding_count = resolver.semantic_executor_binding_count()?;
            let distinct_prepared_executor_count = resolver.distinct_prepared_executor_count()?;
            let mut workspace = OnTheFlyWorkspaceV1::new(&selected.trace, point_count)?;
            for (parameter_id, value) in on_the_fly_prepared_parameters(
                &manifest.runtime_metadata,
                &runtime_parameters,
                model_parameter_evaluator,
                parameter_overrides,
            )?
            .into_iter()
            .enumerate()
            {
                workspace.set_parameter(
                    u32::try_from(parameter_id).map_err(|_| {
                        RusticolError::artifact("on-the-fly prepared parameter count exceeds u32")
                    })?,
                    value[0],
                    value[1],
                )?;
            }
            let source_major = on_the_fly_source_major_momenta(
                &selected.seed,
                point_major_external_momenta,
                point_count,
                selected.trace.layout().lorentz_component_count(),
            )?;
            let mut momentum_fill_count = 0_u32;
            let benchmark_elapsed_seconds = if !benchmark {
                workspace.fill_momenta_from_external(
                    &selected.trace,
                    &source_major,
                    point_count,
                )?;
                momentum_fill_count = 1;
                OnTheFlyStructuralInterpreter::execute(
                    &selected.trace,
                    &resolver,
                    &mut workspace,
                    point_count,
                )?;
                None
            } else {
                for _ in 0..benchmark_warmup_repetitions {
                    let cached = trace_cache.lookup(std::hint::black_box(query_digest))?;
                    workspace.fill_momenta_from_external(
                        &cached.trace,
                        &source_major,
                        point_count,
                    )?;
                    momentum_fill_count = momentum_fill_count.checked_add(1).ok_or_else(|| {
                        RusticolError::integrity("on-the-fly momentum-fill counter overflowed")
                    })?;
                    OnTheFlyStructuralInterpreter::execute(
                        &cached.trace,
                        &resolver,
                        &mut workspace,
                        point_count,
                    )?;
                    std::hint::black_box(workspace.amplitude(0)?);
                }
                let started = Instant::now();
                for _ in 0..benchmark_repetitions {
                    let cached = trace_cache.lookup(std::hint::black_box(query_digest))?;
                    workspace.fill_momenta_from_external(
                        &cached.trace,
                        &source_major,
                        point_count,
                    )?;
                    momentum_fill_count = momentum_fill_count.checked_add(1).ok_or_else(|| {
                        RusticolError::integrity("on-the-fly momentum-fill counter overflowed")
                    })?;
                    OnTheFlyStructuralInterpreter::execute(
                        &cached.trace,
                        &resolver,
                        &mut workspace,
                        point_count,
                    )?;
                    std::hint::black_box(workspace.amplitude(0)?);
                }
                Some(profile_duration_seconds(started.elapsed()))
            };
            let benchmark_seconds_per_point = benchmark_elapsed_seconds.map(|elapsed| {
                elapsed / (f64::from(benchmark_repetitions) * f64::from(point_count))
            });
            let (trace_build_count, trace_cache_hit_count) = trace_cache.counts();

            let raw_amplitudes = (0..point_count)
                .map(|point| workspace.amplitude(point).map(|(real, imag)| [real, imag]))
                .collect::<RusticolResult<Vec<_>>>()?;
            let normalization_factor =
                recurrence_normalization_values(&manifest.runtime_metadata)?.factor;
            let normalized_values: Vec<f64> = raw_amplitudes
                .iter()
                .map(|[real, imaginary]| {
                    (real.mul_add(*real, imaginary * imaginary)) * normalization_factor
                })
                .collect();
            validate_on_the_fly_probe_outputs(&raw_amplitudes, &normalized_values)?;
            let current_count = selected.trace.proof().current_count();
            let currents = if collect_current_diagnostics {
                (0..current_count)
                    .map(|current_id| {
                        let [_, component_count] =
                            selected.trace.current_component_range(current_id)?;
                        let mut values =
                            Vec::with_capacity(point_count as usize * component_count as usize);
                        for point in 0..point_count {
                            values.extend(
                                workspace
                                    .observed_current_components(
                                        &selected.trace,
                                        current_id,
                                        point,
                                    )?
                                    .into_iter()
                                    .map(|(real, imaginary)| [real, imaginary]),
                            );
                        }
                        Ok(NativeOnTheFlyCurrentProbeV1 {
                            semantic_digest: selected
                                .trace
                                .current_semantic_digest(current_id)?
                                .to_string(),
                            component_count,
                            values,
                        })
                    })
                    .collect::<RusticolResult<Vec<_>>>()?
            } else {
                Vec::new()
            };

            Ok(NativeOnTheFlyArtifactProbeV1 {
                artifact_id: artifact.manifest().artifact_id.clone(),
                process_id: manifest.key.clone(),
                seed_digest: selected.seed.semantic_digest().to_string(),
                query_digest: selected.query.semantic_digest().to_string(),
                trace_digest: selected.trace.semantic_digest().to_string(),
                point_count,
                raw_amplitudes,
                normalized_values,
                query_family: None,
                normalization_factor,
                currents,
                projection_enabled: selected.projection.enabled,
                projection_applied: selected.projection.applied,
                projection_counts: [selected.projection.pre, selected.projection.post],
                work_census_basis: ON_THE_FLY_WORK_CENSUS_BASIS_V1.to_owned(),
                logical_current_count: work_census.logical_current_count,
                resident_current_count: work_census.resident_current_count,
                resident_current_component_count: work_census.resident_current_component_count,
                source_operation_count: work_census.source_operation_count,
                contribution_operation_count: work_census.contribution_operation_count,
                finalization_operation_count: work_census.finalization_operation_count,
                closure_operation_count: work_census.closure_operation_count,
                total_kernel_application_count: work_census.total_kernel_application_count,
                semantic_executor_binding_count,
                distinct_prepared_executor_count,
                trace_build_count,
                trace_cache_hit_count,
                momentum_fill_count,
                benchmark_warmup_repetitions,
                benchmark_repetitions,
                benchmark_elapsed_seconds,
                benchmark_seconds_per_point,
                direct_plan_load_attempts: 0,
                direct_plan_decode_attempts: 0,
                direct_plan_materialization_attempts: 0,
                established_builder_attempts: 0,
            })
        })();
        let counts = guard.finish()?;
        let mut report = result?;
        report.direct_plan_load_attempts = counts.direct_plan_load_attempts;
        report.direct_plan_decode_attempts = counts.direct_plan_decode_attempts;
        report.direct_plan_materialization_attempts = counts.direct_plan_materialization_attempts;
        report.established_builder_attempts = counts.established_builder_attempts;
        if report.direct_plan_load_attempts != 0
            || report.direct_plan_decode_attempts != 0
            || report.direct_plan_materialization_attempts != 0
            || report.established_builder_attempts != 0
        {
            return Err(RusticolError::integrity(
                "on-the-fly artifact probe observed forbidden global work",
            ));
        }
        Ok(report)
    }
}

#[cfg(feature = "on-the-fly-test-support")]
fn direct_public_helicity_id(
    plan: &DirectRecurrencePlan,
    public_helicities: &[i32],
) -> RusticolResult<u32> {
    let mut matched = None;
    for descriptor in plan.resolved_helicities() {
        let start = usize::try_from(descriptor.public_helicity_start)
            .map_err(|_| RusticolError::integrity("direct public-helicity start exceeds usize"))?;
        let end = start
            .checked_add(descriptor.public_helicity_count as usize)
            .ok_or_else(|| RusticolError::integrity("direct public-helicity range overflows"))?;
        if plan.public_helicities().get(start..end) == Some(public_helicities) {
            if matched.replace(descriptor.id).is_some() {
                return Err(RusticolError::integrity(
                    "direct plan repeats the requested public helicity",
                ));
            }
        }
    }
    matched.ok_or_else(|| {
        RusticolError::invalid_argument(
            "requested public helicities are absent from the recurrence comparator",
        )
    })
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
fn invert_replay_helicity_map(
    public_flow_id: u32,
    helicity_map: &[u32],
    public_direct_helicity_id: u32,
) -> RusticolResult<u32> {
    let matches = helicity_map
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, mapped)| *mapped == public_direct_helicity_id)
        .map(|(representative, _)| representative)
        .collect::<Vec<_>>();
    let [representative] = matches.as_slice() else {
        return Err(RusticolError::integrity(format!(
            "replay flow {} maps {} representative helicities to public helicity {}",
            public_flow_id,
            matches.len(),
            public_direct_helicity_id,
        )));
    };
    u32::try_from(*representative)
        .map_err(|_| RusticolError::integrity("representative helicity ID exceeds u32"))
}

#[cfg(feature = "on-the-fly-test-support")]
fn representative_direct_helicity_id(
    selector: &DirectReplaySelectorPlan,
    public_direct_helicity_id: u32,
) -> RusticolResult<u32> {
    invert_replay_helicity_map(
        selector.public_flow_id(),
        selector.helicity_map(),
        public_direct_helicity_id,
    )
}

#[cfg(feature = "on-the-fly-test-support")]
fn direct_representative_source_states(
    plan: &DirectRecurrencePlan,
    representative_helicity_id: u32,
) -> RusticolResult<BTreeMap<u32, u32>> {
    let descriptor = plan
        .resolved_helicities()
        .get(representative_helicity_id as usize)
        .filter(|descriptor| descriptor.id == representative_helicity_id)
        .ok_or_else(|| RusticolError::integrity("representative helicity is absent"))?;
    let start = usize::try_from(descriptor.source_state_start)
        .map_err(|_| RusticolError::integrity("source-state start exceeds usize"))?;
    let end = start
        .checked_add(descriptor.source_state_count as usize)
        .ok_or_else(|| RusticolError::integrity("source-state range overflows usize"))?;
    plan.source_state_assignments()
        .get(start..end)
        .ok_or_else(|| RusticolError::integrity("source-state range is absent"))?
        .iter()
        .map(|assignment| Ok((assignment.source_slot, assignment.state_index)))
        .collect()
}

#[cfg(feature = "on-the-fly-test-support")]
fn current_belongs_to_representative_helicity(
    current: &crate::recurrence::RecurrenceCurrent,
    source_states: &BTreeMap<u32, u32>,
) -> bool {
    current
        .key()
        .helicity_identity()
        .local_source_states()
        .iter()
        .all(|assignment| {
            source_states.get(&assignment.source_slot()) == Some(&assignment.state_index())
        })
}

#[cfg(feature = "on-the-fly-test-support")]
fn diagnostic_complex_scale(value: (f64, f64), scale: (f64, f64)) -> (f64, f64) {
    (
        value.0 * scale.0 - value.1 * scale.1,
        value.0 * scale.1 + value.1 * scale.0,
    )
}

#[cfg(feature = "on-the-fly-test-support")]
impl NativeRuntime {
    /// Compare point zero from a genuine compact artifact with a separately
    /// authenticated recurrence artifact. Neither artifact is treated as a
    /// carrier for the other's execution payload.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn on_the_fly_execution_diagnostic_v1(
        on_the_fly_artifact_path: impl AsRef<Path>,
        on_the_fly_process_id: &str,
        recurrence_artifact_path: impl AsRef<Path>,
        recurrence_process_id: &str,
        authenticated: &AuthenticatedRecurrenceBuilderInput,
        retained_direct: &PreparedDirectExecutorCatalog,
        selected_public_flow_id: u32,
        public_helicities: &[i32],
        point_major_external_momenta: &[f64],
        parameter_overrides: &BTreeMap<String, [f64; 2]>,
    ) -> RusticolResult<NativeOnTheFlyExecutionDiagnosticV1> {
        let on_the_fly_artifact = VerifiedArtifact::open(on_the_fly_artifact_path)?;
        let on_the_fly_selection =
            on_the_fly_artifact.select_process(Some(on_the_fly_process_id))?;
        if on_the_fly_selection.remaps_public_process() {
            return Err(RusticolError::invalid_argument(
                "execution diagnostics require a representative on-the-fly process ID",
            ));
        }
        let (loaded_on_the_fly, on_the_fly_evaluator_root) =
            load_verified_evaluator(&on_the_fly_artifact, &on_the_fly_selection)?;
        let on_the_fly_manifest = match loaded_on_the_fly {
            LoadedExecutionManifest::OnTheFly(manifest) => manifest,
            _ => {
                return Err(RusticolError::compatibility(
                    "execution diagnostic artifact A must be a genuine on-the-fly artifact",
                ));
            }
        };
        let loaded_on_the_fly = super::on_the_fly_load::load_on_the_fly_native_runtime(
            &on_the_fly_artifact,
            &on_the_fly_evaluator_root,
            &on_the_fly_manifest,
            &on_the_fly_selection,
        )?;
        let super::on_the_fly_load::LoadedOnTheFlyRuntime {
            common: mut on_the_fly_common,
            lane: mut on_the_fly_lane,
            selectors: _,
            metadata_selectors: _,
            public_metadata: _,
            color_contraction: _,
        } = loaded_on_the_fly;

        let seed = on_the_fly_lane.seed();
        let retained_summary = authenticated.template().summary();
        if seed.process_digest() != authenticated.process().semantic_identity().process_digest()
            || seed.model_digest() != retained_summary.compiled_model_digest
            || seed.template_catalog_digest() != retained_summary.catalog_digest
            || seed.prepared_pack_digest() != retained_summary.prepared_kernel_pack_digest
            || seed.direct_catalog_digest() != retained_direct.direct_template_catalog_digest()
        {
            return Err(RusticolError::integrity(
                "retained diagnostic inputs do not belong to on-the-fly artifact A",
            ));
        }
        let on_the_fly_seed_model_digest = seed.model_digest();
        let on_the_fly_seed_template_digest = seed.template_catalog_digest();
        let on_the_fly_seed_pack_digest = seed.prepared_pack_digest();
        let on_the_fly_seed_direct_digest = seed.direct_catalog_digest();

        let overrides = parameter_overrides
            .iter()
            .map(|(name, value)| (name.clone(), (value[0], value[1])))
            .collect::<BTreeMap<_, _>>();
        on_the_fly_common.apply_model_parameter_overrides(&overrides)?;
        let on_the_fly_snapshot = on_the_fly_lane.execution_diagnostic_v1(
            authenticated,
            selected_public_flow_id,
            public_helicities,
            point_major_external_momenta,
            &on_the_fly_common.model_parameter_values_f64,
        )?;

        // The compact probe guard is deliberately not active here: artifact B
        // is an independent recurrence comparator whose DirectPlan must load.
        let recurrence_artifact = VerifiedArtifact::open(recurrence_artifact_path)?;
        let recurrence_selection =
            recurrence_artifact.select_process(Some(recurrence_process_id))?;
        if recurrence_selection.remaps_public_process() {
            return Err(RusticolError::invalid_argument(
                "execution diagnostics require a representative recurrence process ID",
            ));
        }
        let (loaded_recurrence, recurrence_evaluator_root) =
            load_verified_evaluator(&recurrence_artifact, &recurrence_selection)?;
        let recurrence_manifest = match loaded_recurrence {
            LoadedExecutionManifest::Recurrence(manifest) => manifest,
            _ => {
                return Err(RusticolError::compatibility(
                    "execution diagnostic artifact B must be a recurrence artifact",
                ));
            }
        };
        if on_the_fly_selection.process.expression != recurrence_selection.process.expression
            || on_the_fly_selection.process.external_pdgs
                != recurrence_selection.process.external_pdgs
            || on_the_fly_selection.process.color_accuracy
                != recurrence_selection.process.color_accuracy
            || recurrence_manifest.plan.builder_input_sha256
                != authenticated
                    .process()
                    .semantic_identity()
                    .input_digest()
                    .to_string()
            || recurrence_manifest.prepared_kernel_pack_digest
                != on_the_fly_seed_pack_digest.to_string()
            || recurrence_manifest.direct_template_catalog_digest
                != on_the_fly_seed_direct_digest.to_string()
        {
            return Err(RusticolError::integrity(
                "on-the-fly artifact A, recurrence artifact B, and retained inputs have different identities",
            ));
        }

        let plan = load_plan(
            &recurrence_artifact,
            &recurrence_evaluator_root,
            &recurrence_manifest,
        )?
        .plan;
        let (recurrence_pack_bytes, mut recurrence_pack, recurrence_payload_root) =
            load_prepared_pack(&recurrence_artifact, &recurrence_manifest)?;
        drop(recurrence_pack_bytes);
        let recurrence_direct_manifest = recurrence_pack.recurrence_direct_template_catalog(
            &recurrence_manifest.prepared_kernel_pack_digest,
            &recurrence_manifest.direct_template_catalog_digest,
        )?;
        if recurrence_direct_manifest.compiled_model_digest
            != on_the_fly_seed_model_digest.to_string()
            || recurrence_direct_manifest.recurrence_template_catalog_digest
                != on_the_fly_seed_template_digest.to_string()
        {
            return Err(RusticolError::integrity(
                "artifact A and comparator B use different model/template catalogs",
            ));
        }
        let recurrence_payloads =
            recurrence_artifact.evaluator_payload_store(&recurrence_payload_root)?;
        let recurrence_runtime_parameters =
            recurrence_runtime_parameters(&recurrence_manifest.runtime_metadata);
        let recurrence_parameter_evaluator =
            super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
                &recurrence_pack,
                &recurrence_runtime_parameters,
                &recurrence_payloads,
            )?;
        let recurrence_prepared_parameters = on_the_fly_prepared_parameters(
            &recurrence_manifest.runtime_metadata,
            &recurrence_runtime_parameters,
            recurrence_parameter_evaluator,
            parameter_overrides,
        )?;
        if recurrence_prepared_parameters.len() != on_the_fly_snapshot.prepared_parameters.len()
            || recurrence_prepared_parameters
                .iter()
                .zip(&on_the_fly_snapshot.prepared_parameters)
                .any(|(recurrence, on_the_fly)| {
                    recurrence[0].to_bits() != on_the_fly.0.to_bits()
                        || recurrence[1].to_bits() != on_the_fly.1.to_bits()
                })
        {
            return Err(RusticolError::integrity(
                "artifacts A and B prepared different parameter vectors",
            ));
        }

        let recurrence_runtime_slots = runtime_parameter_slots(&recurrence_runtime_parameters)?;
        let source_domains = build_direct_source_domains(
            &plan,
            &recurrence_manifest.runtime_metadata,
            &recurrence_runtime_slots,
        )?;
        let backend = NativeRecurrenceDirectExecutorBackend::load_from_verified_artifact(
            &mut recurrence_pack,
            &recurrence_artifact,
            &recurrence_payload_root,
            &plan,
            &recurrence_manifest.prepared_kernel_pack_digest,
            &recurrence_manifest.direct_template_catalog_digest,
            source_domains,
        )?;
        let (executors, _backend_owners) = backend.into_parts();
        let mut direct_runtime = DirectRecurrenceExecutionRuntime::new(plan.clone(), executors, 4)?;
        let parameter_re = on_the_fly_snapshot
            .prepared_parameters
            .iter()
            .map(|value| value.0)
            .collect::<Vec<_>>();
        let parameter_im = on_the_fly_snapshot
            .prepared_parameters
            .iter()
            .map(|value| value.1)
            .collect::<Vec<_>>();
        direct_runtime.set_parameters(&parameter_re, &parameter_im)?;
        let selector = direct_runtime.prepare_replay_selector(selected_public_flow_id)?;
        let public_direct_helicity_id = direct_public_helicity_id(&plan, public_helicities)?;
        let representative_direct_helicity_id =
            representative_direct_helicity_id(&selector, public_direct_helicity_id)?;
        let representative_source_states =
            direct_representative_source_states(&plan, representative_direct_helicity_id)?;
        let destination = plan
            .amplitude_destinations()
            .iter()
            .filter(|destination| {
                destination.target_sector_id == selector.representative_flow_id()
                    && destination.target_helicity_id_or_sentinel
                        == representative_direct_helicity_id
            })
            .copied()
            .collect::<Vec<_>>();
        let [destination] = destination.as_slice() else {
            return Err(RusticolError::integrity(format!(
                "selected recurrence lane has {} amplitude destinations, expected one",
                destination.len(),
            )));
        };

        begin_direct_current_observation(
            direct_runtime.plan(),
            Some(selector.representative_flow_id()),
            1,
        )?;
        let execution = direct_runtime
            .execute_replay_tile_from_external_unprofiled(
                &selector,
                1,
                point_major_external_momenta,
            )
            .and_then(|output| {
                Ok((
                    *output
                        .destination_re(destination.id)
                        .and_then(|values| values.first())
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "selected recurrence public amplitude real value is absent",
                            )
                        })?,
                    *output
                        .destination_im(destination.id)
                        .and_then(|values| values.first())
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "selected recurrence public amplitude imaginary value is absent",
                            )
                        })?,
                ))
            });
        let observation = take_direct_current_observation();
        let recurrence_public_amplitude = execution?;
        let observation: DirectCurrentObservation = observation?;
        let recurrence_representative_amplitude = observation
            .amplitudes_before_replay
            .get(&destination.id)
            .and_then(|values| values.first())
            .copied()
            .ok_or_else(|| {
                RusticolError::integrity(
                    "selected recurrence representative amplitude was not observed",
                )
            })?;
        let phase = selector.phase();
        let scale = (
            phase.0 * f64::from(selector.multiplicity()),
            phase.1 * f64::from(selector.multiplicity()),
        );
        let expected_public = diagnostic_complex_scale(recurrence_representative_amplitude, scale);
        if expected_public.0.to_bits() != recurrence_public_amplitude.0.to_bits()
            || expected_public.1.to_bits() != recurrence_public_amplitude.1.to_bits()
        {
            return Err(RusticolError::integrity(
                "recurrence replay scale was not applied exactly once",
            ));
        }

        let established = authenticated.build()?;
        if established.currents().len() != plan.currents().len() {
            return Err(RusticolError::integrity(
                "retained established current catalog differs from comparator B",
            ));
        }
        let mut on_the_fly_currents = BTreeMap::new();
        for current in &on_the_fly_snapshot.currents {
            if on_the_fly_currents
                .insert(current.semantic_digest, current)
                .is_some()
            {
                return Err(RusticolError::integrity(
                    "on-the-fly selected trace repeats a semantic current",
                ));
            }
        }
        let mut direct_currents = BTreeMap::new();
        let mut excluded_direct_current_count = 0_u32;
        for (semantic_current_id, observed) in &observation.currents {
            let current = established
                .currents()
                .get(*semantic_current_id as usize)
                .filter(|current| current.id() == *semantic_current_id)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "observed Direct current is absent from retained construction",
                    )
                })?;
            if !current_belongs_to_representative_helicity(current, &representative_source_states) {
                excluded_direct_current_count = excluded_direct_current_count.saturating_add(1);
                continue;
            }
            let color = established
                .dynamic_color_states()
                .get(current.key().dynamic_lc_color_state_id().get() as usize)
                .ok_or_else(|| {
                    RusticolError::integrity("observed Direct current color state is absent")
                })?;
            let digest = hash_current_key(current.key(), color)?;
            if !on_the_fly_currents.contains_key(&digest) {
                excluded_direct_current_count = excluded_direct_current_count.saturating_add(1);
                continue;
            }
            if direct_currents.insert(digest, observed).is_some() {
                return Err(RusticolError::integrity(
                    "selected Direct live slice repeats a semantic current",
                ));
            }
        }
        if direct_currents.len() != on_the_fly_currents.len() {
            return Err(RusticolError::integrity(format!(
                "selected current bijection has {} on-the-fly and {} Direct currents",
                on_the_fly_currents.len(),
                direct_currents.len(),
            )));
        }

        let mut ordered = on_the_fly_currents
            .iter()
            .map(|(digest, on_the_fly)| {
                let direct = direct_currents.get(digest).copied().ok_or_else(|| {
                    RusticolError::integrity(
                        "selected Direct live slice omits an on-the-fly current",
                    )
                })?;
                if on_the_fly.stage != u32::from(direct.descriptor.stage)
                    || on_the_fly.values.len() != usize::from(direct.descriptor.component_count)
                    || direct.values.len() != on_the_fly.values.len()
                {
                    return Err(RusticolError::integrity(
                        "matched semantic current has a different depth or component layout",
                    ));
                }
                Ok((on_the_fly.stage, *digest, *on_the_fly, direct))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        ordered.sort_by_key(|(stage, digest, _, _)| (*stage, digest.to_string()));
        let mut current_components = Vec::new();
        for (stage, digest, on_the_fly, direct) in ordered {
            for (component, (&on_the_fly_value, &recurrence_value)) in
                on_the_fly.values.iter().zip(&direct.values).enumerate()
            {
                for value in [on_the_fly_value, recurrence_value] {
                    if !value.0.is_finite() || !value.1.is_finite() {
                        return Err(RusticolError::evaluation(
                            "execution diagnostic observed a non-finite current value",
                        ));
                    }
                }
                current_components.push(NativeOnTheFlyExecutionComponentV1 {
                    dependency_depth: stage,
                    semantic_digest: digest.to_string(),
                    component: u32::try_from(component).map_err(|_| {
                        RusticolError::integrity("current component index exceeds u32")
                    })?,
                    on_the_fly: [on_the_fly_value.0, on_the_fly_value.1],
                    recurrence: [recurrence_value.0, recurrence_value.1],
                    on_the_fly_bits: [on_the_fly_value.0.to_bits(), on_the_fly_value.1.to_bits()],
                    recurrence_bits: [recurrence_value.0.to_bits(), recurrence_value.1.to_bits()],
                    absolute_delta: [
                        (on_the_fly_value.0 - recurrence_value.0).abs(),
                        (on_the_fly_value.1 - recurrence_value.1).abs(),
                    ],
                });
            }
        }
        let first_raw_bit_difference = current_components
            .iter()
            .position(|component| component.on_the_fly_bits != component.recurrence_bits);
        let raw_bit_difference_count = u32::try_from(
            current_components
                .iter()
                .filter(|component| component.on_the_fly_bits != component.recurrence_bits)
                .count(),
        )
        .map_err(|_| RusticolError::integrity("raw difference count exceeds u32"))?;
        let on_the_fly_public_amplitude = on_the_fly_snapshot.raw_amplitude;
        for value in [
            on_the_fly_public_amplitude,
            recurrence_representative_amplitude,
            recurrence_public_amplitude,
        ] {
            if !value.0.is_finite() || !value.1.is_finite() {
                return Err(RusticolError::evaluation(
                    "execution diagnostic observed a non-finite amplitude",
                ));
            }
        }

        Ok(NativeOnTheFlyExecutionDiagnosticV1 {
            on_the_fly_artifact_id: on_the_fly_artifact.manifest().artifact_id.clone(),
            on_the_fly_process_id: on_the_fly_manifest.key.clone(),
            recurrence_artifact_id: recurrence_artifact.manifest().artifact_id.clone(),
            recurrence_process_id: recurrence_manifest.key.clone(),
            seed_digest: on_the_fly_snapshot.seed_digest.to_string(),
            query_digest: on_the_fly_snapshot.query_digest.to_string(),
            trace_digest: on_the_fly_snapshot.trace_digest.to_string(),
            direct_plan_semantic_digest: plan.semantic_digest().to_string(),
            direct_runtime_layout_digest: plan.runtime_layout_digest().to_string(),
            public_flow_id: selector.public_flow_id(),
            representative_flow_id: selector.representative_flow_id(),
            public_helicities: public_helicities.to_vec(),
            public_direct_helicity_id,
            representative_direct_helicity_id,
            replay_phase: [phase.0, phase.1],
            replay_multiplicity: selector.multiplicity(),
            replay_scale: [scale.0, scale.1],
            on_the_fly_public_amplitude: [
                on_the_fly_public_amplitude.0,
                on_the_fly_public_amplitude.1,
            ],
            recurrence_representative_amplitude: [
                recurrence_representative_amplitude.0,
                recurrence_representative_amplitude.1,
            ],
            recurrence_public_amplitude: [
                recurrence_public_amplitude.0,
                recurrence_public_amplitude.1,
            ],
            amplitude_absolute_delta: [
                (on_the_fly_public_amplitude.0 - recurrence_public_amplitude.0).abs(),
                (on_the_fly_public_amplitude.1 - recurrence_public_amplitude.1).abs(),
            ],
            current_components,
            first_raw_bit_difference,
            raw_bit_difference_count,
            compared_current_count: u32::try_from(on_the_fly_currents.len())
                .map_err(|_| RusticolError::integrity("compared current count exceeds u32"))?,
            excluded_direct_current_count,
        })
    }
}

#[cfg(feature = "python-generation-bridge")]
pub(super) fn build_recurrence_ready_execution_v1(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<RecurrenceReadyExecutionV1> {
    let loaded_primary = load_plan(artifact, evaluator_root, manifest)?;
    let primary_plan = &loaded_primary.plan;
    let (common, _defaults, _projection, source_domains) =
        build_common_runtime(primary_plan, manifest, physics)?;
    let public_flow_ids = public_flow_ids(primary_plan, &manifest.runtime_metadata, physics)?;
    let primary_direct_helicity_to_physics = direct_helicity_to_physics(primary_plan, physics)?
        .into_iter()
        .map(|value| {
            u64::try_from(value)
                .map_err(|_| RusticolError::artifact("physics helicity index exceeds u64"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let runtime = RecurrenceReadyRuntimeV1 {
        runtime_parameters: manifest.runtime_metadata.runtime_parameters.clone(),
        prepared_parameter_defaults: manifest
            .runtime_metadata
            .prepared_parameter_defaults
            .clone(),
        parameter_projection: manifest.runtime_metadata.parameter_projection.clone(),
        source_domains: encode_ready_source_domains(&source_domains),
        external_is_initial: common.external_is_initial.clone(),
        particle_masses: common
            .particle_masses
            .iter()
            .map(|(pdg, mass)| (*pdg, *mass))
            .collect(),
        particle_mass_parameter_names: common
            .particle_mass_parameter_names
            .iter()
            .map(|(pdg, name)| (*pdg, name.clone()))
            .collect(),
        normalization: manifest.runtime_metadata.normalization.clone(),
        public_flow_ids,
        public_flow_public_ids: manifest
            .runtime_metadata
            .public_color_flows
            .iter()
            .map(|binding| binding.public_id.clone())
            .collect(),
        direct_helicity_to_physics: primary_direct_helicity_to_physics,
        color_contraction: manifest.runtime_metadata.color_contraction.clone(),
    };
    let companion = manifest
        .helicity_selector_companion
        .as_ref()
        .map(|companion| {
            validate_recurrence_companion_physics(physics)?;
            let loaded = load_plan_summary(artifact, evaluator_root, manifest, &companion.plan)?;
            let source_domains = build_direct_source_domains(
                &loaded.plan,
                &manifest.runtime_metadata,
                &common.model_parameter_runtime_slots,
            )?;
            let direct_helicity_to_physics = direct_helicity_to_physics(&loaded.plan, physics)?
                .into_iter()
                .map(|value| {
                    u64::try_from(value)
                        .map_err(|_| RusticolError::artifact("companion physics index exceeds u64"))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(RecurrenceReadyCompanionV1 {
                process_digest: companion.process_digest.clone(),
                plan: RecurrenceReadyPlanV1::from_summary(&companion.plan),
                source_domains: encode_ready_source_domains(&source_domains),
                direct_helicity_to_physics,
            })
        })
        .transpose()?;
    Ok(RecurrenceReadyExecutionV1 {
        schema_version: 1,
        required_runtime_capabilities: manifest.required_runtime_capabilities.clone(),
        process: manifest.process.clone(),
        key: manifest.key.clone(),
        color_accuracy: manifest.color_accuracy.clone(),
        external_pdg_order: manifest.external_pdg_order.clone(),
        compiled_model_digest: manifest.compiled_model_digest.clone(),
        recurrence_template_catalog_digest: manifest.recurrence_template_catalog_digest.clone(),
        prepared_kernel_pack_digest: manifest.prepared_kernel_pack_digest.clone(),
        direct_template_catalog_digest: manifest.direct_template_catalog_digest.clone(),
        kernel_pack: manifest.kernel_pack.clone(),
        primary_plan: RecurrenceReadyPlanV1::from_summary(&manifest.plan),
        runtime,
        companion,
    })
}

#[cfg(feature = "python-generation-bridge")]
fn encode_ready_source_domains(
    domains: &[DirectSourceDispatchDomainSpec],
) -> Vec<RecurrenceReadySourceDomainV1> {
    domains
        .iter()
        .map(|domain| RecurrenceReadySourceDomainV1 {
            variants: domain
                .variants
                .iter()
                .map(|variant| RecurrenceReadySourceVariantV1 {
                    key: match variant.key {
                        DirectSourceDispatchKey::SpinStateClass(value) => {
                            RecurrenceReadySourceKeyV1::SpinStateClass(value)
                        }
                        DirectSourceDispatchKey::RuntimeVariant {
                            source_row_id,
                            runtime_variant_id,
                        } => RecurrenceReadySourceKeyV1::RuntimeVariant {
                            source_row_id,
                            runtime_variant_id,
                        },
                    },
                    template: RecurrenceReadySourceTemplateV1 {
                        spin_state_class: variant.template.spin_state_class,
                        family: match variant.template.family {
                            DirectSourceWavefunctionFamily::Scalar => {
                                RecurrenceReadySourceFamilyV1::Scalar
                            }
                            DirectSourceWavefunctionFamily::WeylFermion => {
                                RecurrenceReadySourceFamilyV1::WeylFermion
                            }
                            DirectSourceWavefunctionFamily::DiracFermion => {
                                RecurrenceReadySourceFamilyV1::DiracFermion
                            }
                            DirectSourceWavefunctionFamily::Vector => {
                                RecurrenceReadySourceFamilyV1::Vector
                            }
                            DirectSourceWavefunctionFamily::Spin2 => {
                                RecurrenceReadySourceFamilyV1::Spin2
                            }
                        },
                        orientation: match variant.template.orientation {
                            DirectSourceOrientation::Particle => {
                                RecurrenceReadySourceOrientationV1::Particle
                            }
                            DirectSourceOrientation::Antiparticle => {
                                RecurrenceReadySourceOrientationV1::Antiparticle
                            }
                            DirectSourceOrientation::SelfConjugate => {
                                RecurrenceReadySourceOrientationV1::SelfConjugate
                            }
                        },
                        helicity: variant.template.helicity,
                        chirality: variant.template.chirality,
                        mass_parameter_index: variant.template.mass_parameter_index,
                    },
                })
                .collect(),
        })
        .collect()
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
pub(super) fn decode_ready_source_domains(
    domains: &[RecurrenceReadySourceDomainV1],
    plan: &DirectRecurrencePlan,
) -> RusticolResult<Vec<DirectSourceDispatchDomainSpec>> {
    if domains.len()
        != usize::try_from(plan.source_template_or_dispatch_count())
            .map_err(|_| RusticolError::artifact("source-domain count exceeds usize"))?
    {
        return Err(RusticolError::integrity(
            "process-ready source-domain count disagrees with the recurrence plan",
        ));
    }
    let domains = domains
        .iter()
        .map(|domain| DirectSourceDispatchDomainSpec {
            variants: domain
                .variants
                .iter()
                .map(|variant| DirectSourceDispatchVariantSpec {
                    key: match variant.key {
                        RecurrenceReadySourceKeyV1::SpinStateClass(value) => {
                            DirectSourceDispatchKey::SpinStateClass(value)
                        }
                        RecurrenceReadySourceKeyV1::RuntimeVariant {
                            source_row_id,
                            runtime_variant_id,
                        } => DirectSourceDispatchKey::RuntimeVariant {
                            source_row_id,
                            runtime_variant_id,
                        },
                    },
                    template: DirectSourceTemplateSpec {
                        spin_state_class: variant.template.spin_state_class,
                        family: match variant.template.family {
                            RecurrenceReadySourceFamilyV1::Scalar => {
                                DirectSourceWavefunctionFamily::Scalar
                            }
                            RecurrenceReadySourceFamilyV1::WeylFermion => {
                                DirectSourceWavefunctionFamily::WeylFermion
                            }
                            RecurrenceReadySourceFamilyV1::DiracFermion => {
                                DirectSourceWavefunctionFamily::DiracFermion
                            }
                            RecurrenceReadySourceFamilyV1::Vector => {
                                DirectSourceWavefunctionFamily::Vector
                            }
                            RecurrenceReadySourceFamilyV1::Spin2 => {
                                DirectSourceWavefunctionFamily::Spin2
                            }
                        },
                        orientation: match variant.template.orientation {
                            RecurrenceReadySourceOrientationV1::Particle => {
                                DirectSourceOrientation::Particle
                            }
                            RecurrenceReadySourceOrientationV1::Antiparticle => {
                                DirectSourceOrientation::Antiparticle
                            }
                            RecurrenceReadySourceOrientationV1::SelfConjugate => {
                                DirectSourceOrientation::SelfConjugate
                            }
                        },
                        helicity: variant.template.helicity,
                        chirality: variant.template.chirality,
                        mass_parameter_index: variant.template.mass_parameter_index,
                    },
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    validate_ready_source_domains_against_plan(&domains, plan)?;
    Ok(domains)
}

fn validate_ready_source_domains_against_plan(
    domains: &[DirectSourceDispatchDomainSpec],
    plan: &DirectRecurrencePlan,
) -> RusticolResult<()> {
    let mut expected = vec![BTreeMap::new(); domains.len()];
    match plan.strategy() {
        crate::recurrence::RecurrenceStrategy::TopologyReplay
        | crate::recurrence::RecurrenceStrategy::ContractedColorUnion => {
            for row in plan.sources() {
                let domain = expected
                    .get_mut(row.source_template_or_dispatch_domain as usize)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "process-ready fixed source references an absent domain",
                        )
                    })?;
                let key = DirectSourceDispatchKey::SpinStateClass(row.spin_state_class);
                if domain
                    .insert(key, row.spin_state_class)
                    .is_some_and(|previous| previous != row.spin_state_class)
                {
                    return Err(RusticolError::integrity(
                        "process-ready fixed source key has inconsistent spin semantics",
                    ));
                }
            }
        }
        crate::recurrence::RecurrenceStrategy::AllFlowUnion => {
            for variant in plan.source_dispatch_variants() {
                let domain = expected
                    .get_mut(variant.dispatch_domain_id as usize)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "process-ready union source references an absent domain",
                        )
                    })?;
                let key = DirectSourceDispatchKey::RuntimeVariant {
                    source_row_id: variant.source_row_id,
                    runtime_variant_id: variant.runtime_variant_id,
                };
                if domain
                    .insert(key, variant.crossed_spin_state_class)
                    .is_some_and(|previous| previous != variant.crossed_spin_state_class)
                {
                    return Err(RusticolError::integrity(
                        "process-ready union source key has inconsistent spin semantics",
                    ));
                }
            }
        }
    }
    validate_ready_source_domain_shape(domains, &expected, plan.parameter_value_count())
}

fn validate_ready_source_domain_shape(
    domains: &[DirectSourceDispatchDomainSpec],
    expected: &[BTreeMap<DirectSourceDispatchKey, i32>],
    parameter_count: u32,
) -> RusticolResult<()> {
    if domains.len() != expected.len() {
        return Err(RusticolError::integrity(
            "process-ready source-domain expectations have the wrong width",
        ));
    }
    let globally_referenced = expected
        .iter()
        .flat_map(BTreeMap::iter)
        .map(|(key, spin)| (*key, *spin))
        .collect::<BTreeMap<_, _>>();
    for (domain_id, (domain, expected)) in domains.iter().zip(expected).enumerate() {
        if domain.variants.is_empty()
            || domain
                .variants
                .windows(2)
                .any(|pair| pair[0].key >= pair[1].key)
            || domain.variants.iter().any(|variant| {
                variant
                    .template
                    .mass_parameter_index
                    .is_some_and(|slot| slot >= parameter_count)
            })
        {
            return Err(RusticolError::integrity(format!(
                "process-ready source domain {domain_id} is non-canonical or out of bounds"
            )));
        }

        if expected.is_empty() {
            // Model-global source-template IDs can be sparse for one process.
            // The established builder fills each unused slot with exactly one
            // inert copy of a referenced source so the row-addressable ABI
            // remains dense; it can never be selected by this plan.
            let [inert] = domain.variants.as_slice() else {
                return Err(RusticolError::integrity(format!(
                    "process-ready unused source domain {domain_id} is not a singleton"
                )));
            };
            if globally_referenced.get(&inert.key) != Some(&inert.template.spin_state_class) {
                return Err(RusticolError::integrity(format!(
                    "process-ready unused source domain {domain_id} is not an inert referenced source"
                )));
            }
            continue;
        }

        if domain.variants.len() != expected.len()
            || domain.variants.iter().any(|variant| {
                expected.get(&variant.key) != Some(&variant.template.spin_state_class)
                    || matches!(
                        variant.key,
                        DirectSourceDispatchKey::SpinStateClass(spin)
                            if spin != variant.template.spin_state_class
                    )
            })
        {
            return Err(RusticolError::integrity(format!(
                "process-ready source domain {domain_id} does not exactly cover its plan keys"
            )));
        }
    }
    Ok(())
}

fn validate_recurrence_companion_physics(physics: &ProcessPhysicsV1) -> RusticolResult<()> {
    if physics.coverage.helicities != "complete" || physics.coverage.color != "contracted" {
        return Err(RusticolError::integrity(
            "recurrence helicity-selector companion requires complete helicity and contracted-color coverage",
        ));
    }
    if physics.helicities.iter().any(|helicity| {
        !companion_helicity_coefficient_is_canonical(helicity.structural_zero, helicity.coefficient)
    }) {
        return Err(RusticolError::integrity(
            "recurrence helicity-selector companion requires unit physical-helicity coefficients and zero structural-zero coefficients",
        ));
    }
    Ok(())
}

fn recurrence_parameter_parts(
    plan: &DirectRecurrencePlan,
    defaults: &[[f64; 2]],
    projection: &[RecurrenceParameterProjection],
) -> RusticolResult<(
    Vec<crate::EagerComplex64>,
    Vec<PreparedParameterProjectionEntry>,
)> {
    if defaults
        .iter()
        .flatten()
        .any(|component| !component.is_finite())
    {
        return Err(RusticolError::integrity(
            "recurrence prepared parameter defaults are not finite",
        ));
    }
    let defaults = defaults
        .iter()
        .map(|[real, imaginary]| crate::EagerComplex64::new(*real, *imaginary))
        .collect::<Vec<_>>();
    if defaults.len()
        != usize::try_from(plan.parameter_value_count())
            .map_err(|_| RusticolError::artifact("recurrence parameter count exceeds usize"))?
    {
        return Err(RusticolError::integrity(
            "recurrence prepared defaults do not match the direct plan",
        ));
    }
    let projection = projection
        .iter()
        .filter_map(|row| {
            row.prepared_parameter_id
                .map(|prepared_slot| (row, prepared_slot))
        })
        .map(|(row, prepared_slot)| {
            Ok(PreparedParameterProjectionEntry {
                runtime_slot: usize::try_from(row.runtime_slot).map_err(|_| {
                    RusticolError::artifact("recurrence runtime parameter slot exceeds usize")
                })?,
                prepared_slot: usize::try_from(prepared_slot).map_err(|_| {
                    RusticolError::artifact("recurrence prepared parameter slot exceeds usize")
                })?,
                component: u8::try_from(row.component).map_err(|_| {
                    RusticolError::artifact("recurrence parameter component exceeds u8")
                })?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    Ok((defaults, projection))
}

fn build_common_runtime(
    plan: &DirectRecurrencePlan,
    manifest: &RecurrenceExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<RecurrenceCommonRuntimeParts> {
    let metadata = &manifest.runtime_metadata;
    let runtime_parameters = recurrence_runtime_parameters(metadata);
    let model_parameter_runtime_slots = runtime_parameter_slots(&runtime_parameters)?;
    let model_parameter_values_f64 = runtime_parameters
        .iter()
        .map(|parameter| parameter.default)
        .collect::<Vec<_>>();
    let (parameter_defaults, parameter_projection) = recurrence_parameter_parts(
        plan,
        &metadata.prepared_parameter_defaults,
        &metadata.parameter_projection,
    )?;

    let mut particle_masses = metadata
        .particle_masses
        .iter()
        .map(|row| (row.outgoing_pdg, row.mass))
        .collect::<BTreeMap<_, _>>();
    let mut particle_mass_parameter_names = BTreeMap::new();
    for source in &metadata.source_templates {
        let Some(name) = source.source_ir.mass_parameter.as_ref() else {
            continue;
        };
        for pdg in [
            source.source_ir.identity.pdg_label,
            source.source_ir.identity.anti_pdg_label,
        ] {
            particle_mass_parameter_names.insert(pdg, name.clone());
            if let Some(slots) = model_parameter_runtime_slots.get(name)
                && let Some(mass) = model_parameter_values_f64.get(slots.real)
            {
                particle_masses.insert(pdg, *mass);
            }
        }
    }

    let source_domains =
        build_direct_source_domains(plan, metadata, &model_parameter_runtime_slots)?;
    let normalization = &metadata.normalization;
    let normalization_values = recurrence_normalization_values(metadata)?;
    let color_factor = normalization_values.color_factor;
    let normalization_factor = normalization_values.factor;
    let external_count = manifest.external_pdg_order.len();
    if plan.external_source_count() as usize != external_count
        || metadata.external_legs.len() != external_count
    {
        return Err(RusticolError::integrity(
            "recurrence direct plan external-source count disagrees with process metadata",
        ));
    }
    let external_is_initial = physics
        .external_particles
        .iter()
        .map(|particle| particle.role == crate::ParticleRole::Initial)
        .collect();
    let common = assemble_common_runtime(
        plan,
        RecurrenceCommonRuntimeDefinition {
            process: manifest.process.clone(),
            key: manifest.key.clone(),
            color_accuracy: manifest.color_accuracy.clone(),
            external_pdg_order: manifest.external_pdg_order.clone(),
            runtime_parameters,
            model_parameter_runtime_slots,
            model_parameter_values_f64,
            external_is_initial,
            particle_masses,
            particle_mass_parameter_names,
            normalization_factor,
            normalization_color_factor: color_factor,
            normalization_average_factor: normalization.average_factor,
            normalization_identical_factor: normalization.identical_factor,
            normalization_qcd_coupling_power: normalization.qcd_coupling_power.unwrap_or(0),
            normalization_electroweak_coupling_power: normalization
                .electroweak_coupling_power
                .unwrap_or(0),
        },
    )?;
    Ok((
        common,
        parameter_defaults,
        parameter_projection,
        source_domains,
    ))
}

fn build_ready_common_runtime(
    plan: &DirectRecurrencePlan,
    ready: &RecurrenceReadyExecutionV1,
) -> RusticolResult<RecurrenceCommonRuntimeParts> {
    let runtime = &ready.runtime;
    validate_recurrence_parameter_runtime_parts(
        &runtime.runtime_parameters,
        &runtime.prepared_parameter_defaults,
        &runtime.parameter_projection,
        u64::from(plan.parameter_value_count()),
    )?;
    let runtime_parameters = recurrence_runtime_parameters_from(&runtime.runtime_parameters);
    let model_parameter_runtime_slots = runtime_parameter_slots(&runtime_parameters)?;
    let model_parameter_values_f64 = runtime_parameters
        .iter()
        .map(|parameter| parameter.default)
        .collect::<Vec<_>>();
    let (parameter_defaults, parameter_projection) = recurrence_parameter_parts(
        plan,
        &runtime.prepared_parameter_defaults,
        &runtime.parameter_projection,
    )?;
    let particle_masses = ready_pairs_to_map(&runtime.particle_masses, "particle mass", |mass| {
        mass.is_finite() && mass >= 0.0
    })?;
    let particle_mass_parameter_names =
        ready_string_pairs_to_map(&runtime.particle_mass_parameter_names, "particle mass name")?;
    if particle_mass_parameter_names
        .values()
        .any(|name| !model_parameter_runtime_slots.contains_key(name))
    {
        return Err(RusticolError::integrity(
            "process-ready particle mass references an absent runtime parameter",
        ));
    }
    if runtime.normalization.color_accuracy != ready.color_accuracy
        || !runtime.normalization.average_factor.is_finite()
        || runtime.normalization.average_factor <= 0.0
        || !runtime.normalization.identical_factor.is_finite()
        || runtime.normalization.identical_factor <= 0.0
    {
        return Err(RusticolError::integrity(
            "process-ready recurrence normalization is inconsistent",
        ));
    }
    let normalization = recurrence_normalization_values_from(
        &runtime.normalization,
        runtime.color_contraction.as_ref(),
    )?;
    let source_domains = decode_ready_source_domains(&runtime.source_domains, plan)?;
    let common = assemble_common_runtime(
        plan,
        RecurrenceCommonRuntimeDefinition {
            process: ready.process.clone(),
            key: ready.key.clone(),
            color_accuracy: ready.color_accuracy.clone(),
            external_pdg_order: ready.external_pdg_order.clone(),
            runtime_parameters,
            model_parameter_runtime_slots,
            model_parameter_values_f64,
            external_is_initial: runtime.external_is_initial.clone(),
            particle_masses,
            particle_mass_parameter_names,
            normalization_factor: normalization.factor,
            normalization_color_factor: normalization.color_factor,
            normalization_average_factor: runtime.normalization.average_factor,
            normalization_identical_factor: runtime.normalization.identical_factor,
            normalization_qcd_coupling_power: runtime.normalization.qcd_coupling_power.unwrap_or(0),
            normalization_electroweak_coupling_power: runtime
                .normalization
                .electroweak_coupling_power
                .unwrap_or(0),
        },
    )?;
    Ok((
        common,
        parameter_defaults,
        parameter_projection,
        source_domains,
    ))
}

fn ready_pairs_to_map(
    values: &[(i32, f64)],
    context: &str,
    valid: impl Fn(f64) -> bool,
) -> RusticolResult<BTreeMap<i32, f64>> {
    let mut result = BTreeMap::new();
    let mut previous = None;
    for (key, value) in values {
        if previous.is_some_and(|previous| previous >= *key)
            || result.insert(*key, *value).is_some()
            || !valid(*value)
        {
            return Err(RusticolError::integrity(format!(
                "process-ready recurrence {context} entries are not canonical"
            )));
        }
        previous = Some(*key);
    }
    Ok(result)
}

fn ready_string_pairs_to_map(
    values: &[(i32, String)],
    context: &str,
) -> RusticolResult<BTreeMap<i32, String>> {
    let mut result = BTreeMap::new();
    let mut previous = None;
    for (key, value) in values {
        if previous.is_some_and(|previous| previous >= *key)
            || value.is_empty()
            || result.insert(*key, value.clone()).is_some()
        {
            return Err(RusticolError::integrity(format!(
                "process-ready recurrence {context} entries are not canonical"
            )));
        }
        previous = Some(*key);
    }
    Ok(result)
}

struct RecurrenceCommonRuntimeDefinition {
    process: String,
    key: String,
    color_accuracy: String,
    external_pdg_order: Vec<i32>,
    runtime_parameters: Vec<GenericRuntimeModelParameterManifest>,
    model_parameter_runtime_slots: BTreeMap<String, RuntimeParameterSlots>,
    model_parameter_values_f64: Vec<f64>,
    external_is_initial: Vec<bool>,
    particle_masses: BTreeMap<i32, f64>,
    particle_mass_parameter_names: BTreeMap<i32, String>,
    normalization_factor: f64,
    normalization_color_factor: f64,
    normalization_average_factor: f64,
    normalization_identical_factor: f64,
    normalization_qcd_coupling_power: u64,
    normalization_electroweak_coupling_power: u64,
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn assemble_common_runtime(
    plan: &DirectRecurrencePlan,
    definition: RecurrenceCommonRuntimeDefinition,
) -> RusticolResult<ExecutionRuntime> {
    let RecurrenceCommonRuntimeDefinition {
        process,
        key,
        color_accuracy,
        external_pdg_order,
        runtime_parameters,
        model_parameter_runtime_slots,
        model_parameter_values_f64,
        external_is_initial,
        particle_masses,
        particle_mass_parameter_names,
        normalization_factor,
        normalization_color_factor,
        normalization_average_factor,
        normalization_identical_factor,
        normalization_qcd_coupling_power,
        normalization_electroweak_coupling_power,
    } = definition;
    let external_count = external_pdg_order.len();
    if plan.external_source_count() as usize != external_count
        || external_is_initial.len() != external_count
    {
        return Err(RusticolError::integrity(
            "recurrence common runtime external domain disagrees with its plan",
        ));
    }
    let model_parameter_name_to_index = runtime_parameters
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
        .collect::<BTreeMap<_, _>>();
    let source_count = plan.sources().len();
    let current_count = plan.currents().len();
    let interaction_count = plan.contributions().len();
    let stage_count = plan
        .row_groups()
        .iter()
        .map(|group| usize::from(group.stage))
        .max()
        .and_then(|stage| stage.checked_add(1))
        .ok_or_else(|| RusticolError::integrity("recurrence direct plan has no row groups"))?;
    let amplitude_output_count = plan.amplitude_destinations().len();
    Ok(ExecutionRuntime {
        process,
        key,
        color_accuracy,
        external_pdg_order,
        external_count,
        parameter_count: runtime_parameters.len(),
        value_parameter_count: 0,
        momentum_parameter_count: 0,
        current_count,
        source_count,
        interaction_count,
        stage_count,
        amplitude_output_count,
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
        // Direct-Arena source execution owns typed SourceIR dispatch domains.
        // The legacy source records are intentionally not reconstructed.
        sources: Vec::new(),
        momentum_slots: Vec::new(),
        external_is_initial,
        particle_masses,
        particle_mass_parameter_names,
        normalization_factor,
        normalization_color_factor,
        normalization_average_factor,
        normalization_identical_factor,
        normalization_qcd_coupling_power: usize::try_from(normalization_qcd_coupling_power)
            .map_err(|_| RusticolError::artifact("QCD coupling power exceeds usize"))?,
        normalization_electroweak_coupling_power: usize::try_from(
            normalization_electroweak_coupling_power,
        )
        .map_err(|_| RusticolError::artifact("electroweak coupling power exceeds usize"))?,
        model_parameters: runtime_parameters,
        model_parameter_name_to_index,
        model_parameter_runtime_slots,
        model_parameter_values_f64,
        model_parameter_evaluator: None,
        physics_reduction_override: None,
        physics: None,
        stages: None,
        amplitude_stage: None,
        state_scratch_f64: Vec::new(),
        state_scratch_f64_requires_clear: false,
        values_scratch_f64: Vec::new(),
    })
}

pub(super) fn build_direct_source_domains(
    plan: &DirectRecurrencePlan,
    metadata: &RecurrenceRuntimeMetadata,
    runtime_parameter_slots: &BTreeMap<String, RuntimeParameterSlots>,
) -> RusticolResult<Vec<DirectSourceDispatchDomainSpec>> {
    let templates = metadata
        .source_templates
        .iter()
        .map(|source| (source.source_template_id, source))
        .collect::<BTreeMap<_, _>>();
    let legs = metadata
        .external_legs
        .iter()
        .map(|leg| (leg.source_slot, leg))
        .collect::<BTreeMap<_, _>>();
    let domain_count = usize::try_from(plan.source_template_or_dispatch_count())
        .map_err(|_| RusticolError::artifact("recurrence source-domain count exceeds usize"))?;
    let mut variants =
        vec![BTreeMap::<DirectSourceDispatchKey, DirectSourceTemplateSpec>::new(); domain_count];

    match plan.strategy() {
        crate::recurrence::RecurrenceStrategy::TopologyReplay
        | crate::recurrence::RecurrenceStrategy::ContractedColorUnion => {
            for row in plan.sources() {
                let source = templates
                    .get(&row.source_template_or_dispatch_domain)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence direct plan references an absent SourceIR template",
                        )
                    })?;
                let leg = legs.get(&row.source_slot).ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence direct source references an absent external leg",
                    )
                })?;
                let spec = direct_source_template_spec(
                    source,
                    leg,
                    metadata,
                    runtime_parameter_slots,
                    plan.parameter_value_count(),
                )?;
                if row.spin_state_class != spec.spin_state_class {
                    return Err(RusticolError::integrity(
                        "recurrence direct source spin class disagrees with crossed SourceIR metadata",
                    ));
                }
                insert_source_domain_variant(
                    &mut variants,
                    row.source_template_or_dispatch_domain,
                    DirectSourceDispatchKey::SpinStateClass(row.spin_state_class),
                    spec,
                )?;
            }
        }
        crate::recurrence::RecurrenceStrategy::AllFlowUnion => {
            for variant in plan.source_dispatch_variants() {
                let row = plan
                    .sources()
                    .get(variant.source_row_id as usize)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence source variant references an absent source row",
                        )
                    })?;
                let source = templates.get(&variant.source_template_id).ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence source variant references an absent SourceIR template",
                    )
                })?;
                let leg = legs.get(&row.source_slot).ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence source variant references an absent external leg",
                    )
                })?;
                let spec = direct_source_template_spec(
                    source,
                    leg,
                    metadata,
                    runtime_parameter_slots,
                    plan.parameter_value_count(),
                )?;
                if row.source_template_or_dispatch_domain != variant.dispatch_domain_id
                    || spec.spin_state_class != variant.crossed_spin_state_class
                {
                    return Err(RusticolError::integrity(
                        "recurrence union source variant disagrees with its dispatch domain or crossed SourceIR state",
                    ));
                }
                insert_source_domain_variant(
                    &mut variants,
                    variant.dispatch_domain_id,
                    DirectSourceDispatchKey::RuntimeVariant {
                        source_row_id: variant.source_row_id,
                        runtime_variant_id: variant.runtime_variant_id,
                    },
                    spec,
                )?;
            }
        }
    }

    finish_direct_source_domains(variants)
}

fn finish_direct_source_domains(
    variants: Vec<BTreeMap<DirectSourceDispatchKey, DirectSourceTemplateSpec>>,
) -> RusticolResult<Vec<DirectSourceDispatchDomainSpec>> {
    let inert = variants
        .iter()
        .flat_map(BTreeMap::iter)
        .next()
        .map(|(key, spec)| (*key, *spec))
        .ok_or_else(|| RusticolError::integrity("recurrence direct plan has no source rows"))?;
    Ok(variants
        .into_iter()
        .map(|domain| DirectSourceDispatchDomainSpec {
            // Prepared source-template IDs are model-global and may be sparse
            // for one process. Unreferenced slots remain inert but preserve the
            // stable IDs addressed by DirectSourceRow.
            variants: if domain.is_empty() {
                vec![DirectSourceDispatchVariantSpec {
                    key: inert.0,
                    template: inert.1,
                }]
            } else {
                domain
                    .into_iter()
                    .map(|(key, template)| DirectSourceDispatchVariantSpec { key, template })
                    .collect()
            },
        })
        .collect())
}

/// Bind the compact process seed to the same crossed SourceIR semantics used
/// by the established direct-plan loader.  This boundary is deliberately
/// plan-independent: source-template IDs remain the authenticated model-level
/// IDs, including sparse slots that this process does not address.
pub(super) fn build_on_the_fly_source_domains(
    seed: &OnTheFlyProcessSeedV1,
    templates: &crate::recurrence::template::ValidatedRecurrenceTemplateInput,
) -> RusticolResult<Vec<DirectSourceDispatchDomainSpec>> {
    let summary = templates.summary();
    if seed.template_catalog_digest() != summary.catalog_digest
        || seed.model_digest() != summary.compiled_model_digest
        || seed.prepared_pack_digest() != summary.prepared_kernel_pack_digest
    {
        return Err(RusticolError::integrity(
            "on-the-fly source seed does not belong to the authenticated recurrence templates",
        ));
    }
    let domain_count = usize::try_from(summary.source_count)
        .map_err(|_| RusticolError::artifact("recurrence source-domain count exceeds usize"))?;
    let mut variants =
        vec![BTreeMap::<DirectSourceDispatchKey, DirectSourceTemplateSpec>::new(); domain_count];
    for compact in seed.source_execution_specs() {
        if compact.source_template_id as usize >= domain_count
            || compact
                .prepared_mass_parameter_slot
                .is_some_and(|slot| slot >= summary.parameter_count)
        {
            return Err(RusticolError::integrity(
                "on-the-fly source or mass-parameter slot is outside its authenticated template domain",
            ));
        }
        let source_template_id = compact.source_template_id;
        let spec = DirectSourceTemplateSpec {
            spin_state_class: compact.spin_state_class,
            family: direct_on_the_fly_source_family(compact.family),
            orientation: direct_on_the_fly_source_orientation(compact.orientation),
            helicity: compact.helicity,
            chirality: compact.chirality,
            mass_parameter_index: compact.prepared_mass_parameter_slot,
        };
        insert_source_domain_variant(
            &mut variants,
            source_template_id,
            DirectSourceDispatchKey::SpinStateClass(spec.spin_state_class),
            spec,
        )?;
    }
    finish_direct_source_domains(variants)
}

#[cfg(test)]
fn build_on_the_fly_source_domains_from_specs(
    specs: impl IntoIterator<Item = OnTheFlySourceExecutionSpecV1>,
    domain_count: usize,
    parameter_count: u32,
    metadata: &RecurrenceRuntimeMetadata,
    runtime_parameter_slots: &BTreeMap<String, RuntimeParameterSlots>,
) -> RusticolResult<Vec<DirectSourceDispatchDomainSpec>> {
    let templates = metadata
        .source_templates
        .iter()
        .map(|source| (source.source_template_id, source))
        .collect::<BTreeMap<_, _>>();
    let legs = metadata
        .external_legs
        .iter()
        .map(|leg| (leg.source_slot, leg))
        .collect::<BTreeMap<_, _>>();
    let mut variants =
        vec![BTreeMap::<DirectSourceDispatchKey, DirectSourceTemplateSpec>::new(); domain_count];
    for compact in specs {
        let source = templates.get(&compact.source_template_id).ok_or_else(|| {
            RusticolError::integrity("on-the-fly source references an absent SourceIR template")
        })?;
        let leg = legs.get(&compact.source_slot).ok_or_else(|| {
            RusticolError::integrity("on-the-fly source references an absent external leg")
        })?;
        let authoritative = direct_source_template_spec(
            source,
            leg,
            metadata,
            runtime_parameter_slots,
            parameter_count,
        )?;
        let compact = DirectSourceTemplateSpec {
            spin_state_class: compact.spin_state_class,
            family: direct_on_the_fly_source_family(compact.family),
            orientation: direct_on_the_fly_source_orientation(compact.orientation),
            helicity: compact.helicity,
            chirality: compact.chirality,
            mass_parameter_index: compact.prepared_mass_parameter_slot,
        };
        let authoritative = bind_on_the_fly_source_spec(compact, authoritative)?;
        insert_source_domain_variant(
            &mut variants,
            source.source_template_id,
            DirectSourceDispatchKey::SpinStateClass(authoritative.spin_state_class),
            authoritative,
        )?;
    }
    finish_direct_source_domains(variants)
}

#[cfg(test)]
fn bind_on_the_fly_source_spec(
    compact: DirectSourceTemplateSpec,
    authoritative: DirectSourceTemplateSpec,
) -> RusticolResult<DirectSourceTemplateSpec> {
    if compact.spin_state_class != authoritative.spin_state_class {
        return Err(RusticolError::integrity(
            "on-the-fly source spin class disagrees with crossed SourceIR metadata",
        ));
    }
    if compact.family != authoritative.family {
        return Err(RusticolError::integrity(
            "on-the-fly source family disagrees with crossed SourceIR metadata",
        ));
    }
    if compact.orientation != authoritative.orientation {
        return Err(RusticolError::integrity(
            "on-the-fly source orientation disagrees with crossed SourceIR metadata",
        ));
    }
    if compact.helicity != authoritative.helicity {
        return Err(RusticolError::integrity(
            "on-the-fly source helicity disagrees with crossed SourceIR metadata",
        ));
    }
    if compact.chirality != authoritative.chirality {
        return Err(RusticolError::integrity(
            "on-the-fly source chirality disagrees with crossed SourceIR metadata",
        ));
    }
    if compact.mass_parameter_index.is_some()
        && compact.mass_parameter_index != authoritative.mass_parameter_index
    {
        return Err(RusticolError::integrity(
            "on-the-fly source prepared mass parameter slot disagrees with SourceIR metadata",
        ));
    }
    Ok(authoritative)
}

const fn direct_on_the_fly_source_family(
    value: OnTheFlySourceWavefunctionFamilyV1,
) -> DirectSourceWavefunctionFamily {
    match value {
        OnTheFlySourceWavefunctionFamilyV1::Scalar => DirectSourceWavefunctionFamily::Scalar,
        OnTheFlySourceWavefunctionFamilyV1::WeylFermion => {
            DirectSourceWavefunctionFamily::WeylFermion
        }
        OnTheFlySourceWavefunctionFamilyV1::DiracFermion => {
            DirectSourceWavefunctionFamily::DiracFermion
        }
        OnTheFlySourceWavefunctionFamilyV1::Vector => DirectSourceWavefunctionFamily::Vector,
        OnTheFlySourceWavefunctionFamilyV1::Spin2 => DirectSourceWavefunctionFamily::Spin2,
    }
}

const fn direct_on_the_fly_source_orientation(
    value: OnTheFlySourceOrientationV1,
) -> DirectSourceOrientation {
    match value {
        OnTheFlySourceOrientationV1::Particle => DirectSourceOrientation::Particle,
        OnTheFlySourceOrientationV1::Antiparticle => DirectSourceOrientation::Antiparticle,
        OnTheFlySourceOrientationV1::SelfConjugate => DirectSourceOrientation::SelfConjugate,
    }
}

fn insert_source_domain_variant(
    domains: &mut [BTreeMap<DirectSourceDispatchKey, DirectSourceTemplateSpec>],
    domain_id: u32,
    key: DirectSourceDispatchKey,
    spec: DirectSourceTemplateSpec,
) -> RusticolResult<()> {
    let domain = domains.get_mut(domain_id as usize).ok_or_else(|| {
        RusticolError::integrity("recurrence direct source-domain ID is out of bounds")
    })?;
    if let Some(previous) = domain.insert(key, spec)
        && previous != spec
    {
        return Err(RusticolError::integrity(
            "recurrence source domain maps one dispatch key to different SourceIR semantics",
        ));
    }
    Ok(())
}

fn direct_source_template_spec(
    source: &RecurrenceSourceTemplate,
    leg: &RecurrenceExternalLeg,
    metadata: &RecurrenceRuntimeMetadata,
    runtime_parameter_slots: &BTreeMap<String, RuntimeParameterSlots>,
    parameter_count: u32,
) -> RusticolResult<DirectSourceTemplateSpec> {
    if source.source_ir.identity.pdg_label != leg.outgoing_pdg {
        return Err(RusticolError::integrity(
            "recurrence SourceIR particle disagrees with its external leg",
        ));
    }
    let crossing = if leg.is_initial {
        &source.crossing
    } else {
        &IDENTITY_RECURRENCE_CROSSING
    };
    let helicity = source
        .helicity
        .checked_mul(crossing.helicity_factor)
        .ok_or_else(|| RusticolError::integrity("recurrence source helicity overflows"))?;
    let chirality = source
        .chirality
        .checked_mul(crossing.chirality_factor)
        .ok_or_else(|| RusticolError::integrity("recurrence source chirality overflows"))?;
    let spin_state_class = source
        .spin_state
        .checked_mul(crossing.spin_state_factor)
        .ok_or_else(|| RusticolError::integrity("recurrence source spin state overflows"))?;
    let family = direct_source_family(&source.source_ir)?;
    let orientation = match source.source_ir.identity.orientation {
        RecurrenceSourceOrientation::Particle => DirectSourceOrientation::Particle,
        RecurrenceSourceOrientation::Antiparticle => DirectSourceOrientation::Antiparticle,
        RecurrenceSourceOrientation::SelfConjugate => DirectSourceOrientation::SelfConjugate,
    };
    let mass_parameter_index = source
        .source_ir
        .mass_parameter
        .as_deref()
        .map(|name| {
            prepared_parameter_index(
                name,
                runtime_parameter_slots,
                &metadata.parameter_projection,
                parameter_count,
            )
        })
        .transpose()?;
    if mass_parameter_index.is_none()
        && metadata
            .particle_masses
            .iter()
            .find(|mass| mass.outgoing_pdg == leg.outgoing_pdg)
            .is_some_and(|mass| mass.mass != 0.0)
    {
        return Err(RusticolError::compatibility(
            "massive recurrence SourceIR has no prepared mass-parameter slot",
        ));
    }
    Ok(DirectSourceTemplateSpec {
        spin_state_class,
        family,
        orientation,
        helicity,
        chirality,
        mass_parameter_index,
    })
}

const IDENTITY_RECURRENCE_CROSSING: RecurrenceGenericCrossingIr = RecurrenceGenericCrossingIr {
    momentum_transform: RecurrenceMomentumTransform::Identity,
    helicity_factor: 1,
    chirality_factor: 1,
    spin_state_factor: 1,
    phase: [1.0, 0.0],
};

fn direct_source_family(
    source: &RecurrenceGenericSourceIr,
) -> RusticolResult<DirectSourceWavefunctionFamily> {
    match (
        source.statistics,
        source.wavefunction_family,
        source.component_dimension,
    ) {
        (RecurrenceParticleStatistics::Boson, RecurrenceWavefunctionFamily::Scalar, 1) => {
            Ok(DirectSourceWavefunctionFamily::Scalar)
        }
        (RecurrenceParticleStatistics::Fermion, RecurrenceWavefunctionFamily::Fermion, 2) => {
            Ok(DirectSourceWavefunctionFamily::WeylFermion)
        }
        (RecurrenceParticleStatistics::Fermion, RecurrenceWavefunctionFamily::Fermion, 4) => {
            Ok(DirectSourceWavefunctionFamily::DiracFermion)
        }
        (RecurrenceParticleStatistics::Boson, RecurrenceWavefunctionFamily::Vector, 4) => {
            Ok(DirectSourceWavefunctionFamily::Vector)
        }
        (RecurrenceParticleStatistics::Boson, RecurrenceWavefunctionFamily::Spin2, 16) => {
            Ok(DirectSourceWavefunctionFamily::Spin2)
        }
        (_, RecurrenceWavefunctionFamily::Ghost, _)
        | (_, RecurrenceWavefunctionFamily::Auxiliary, _) => Err(RusticolError::compatibility(
            "Direct-Arena recurrence does not yet support ghost or auxiliary external sources",
        )),
        _ => Err(RusticolError::integrity(
            "recurrence SourceIR statistics, family, and component dimension are incompatible",
        )),
    }
}

fn prepared_parameter_index(
    name: &str,
    runtime_parameter_slots: &BTreeMap<String, RuntimeParameterSlots>,
    projection: &[RecurrenceParameterProjection],
    parameter_count: u32,
) -> RusticolResult<u32> {
    let runtime_slot = runtime_parameter_slots
        .get(name)
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "recurrence source mass parameter {name:?} has no runtime projection"
            ))
        })?
        .real;
    let row = projection
        .iter()
        .find(|row| row.runtime_slot as usize == runtime_slot && row.component == 0)
        .ok_or_else(|| {
            RusticolError::integrity(
                "recurrence source mass parameter has no real prepared projection",
            )
        })?;
    if row.runtime_name != name {
        return Err(RusticolError::integrity(
            "recurrence source mass projection is not its real component",
        ));
    }
    let prepared = row.prepared_parameter_id.ok_or_else(|| {
        RusticolError::compatibility(format!(
            "recurrence source mass parameter {name:?} has no prepared parameter slot"
        ))
    })?;
    if prepared >= parameter_count {
        return Err(RusticolError::integrity(
            "recurrence source mass prepared slot exceeds the direct plan",
        ));
    }
    Ok(prepared)
}

pub(super) fn runtime_parameter_slots(
    parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<BTreeMap<String, RuntimeParameterSlots>> {
    let mut result = BTreeMap::new();
    let mut complex = BTreeMap::<String, (Option<usize>, Option<usize>)>::new();
    for parameter in parameters {
        if let Some(name) = &parameter.runtime_name {
            let slots = complex.entry(name.clone()).or_default();
            let target = match parameter.complex_component.as_deref() {
                Some("real") => &mut slots.0,
                Some("imag") => &mut slots.1,
                _ => {
                    return Err(RusticolError::integrity(
                        "recurrence complex parameter lacks component metadata",
                    ));
                }
            };
            if target.replace(parameter.parameter_index).is_some() {
                return Err(RusticolError::integrity(
                    "recurrence complex parameter repeats a component",
                ));
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
            return Err(RusticolError::integrity(
                "recurrence runtime parameter names are not unique",
            ));
        }
    }
    for (name, (real, imaginary)) in complex {
        let real = real.ok_or_else(|| {
            RusticolError::integrity("recurrence complex parameter lacks a real component")
        })?;
        if result
            .insert(name, RuntimeParameterSlots { real, imaginary })
            .is_some()
        {
            return Err(RusticolError::integrity(
                "recurrence runtime parameter names conflict",
            ));
        }
    }
    Ok(result)
}

pub(super) fn decode_sha256(value: &str) -> RusticolResult<[u8; 32]> {
    if value.len() != 64 {
        return Err(RusticolError::integrity(
            "recurrence SHA-256 has an invalid encoded length",
        ));
    }
    let mut output = [0u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|_| {
            RusticolError::integrity("recurrence SHA-256 is not lowercase hexadecimal")
        })?;
    }
    Ok(output)
}

#[cfg(test)]
mod binding_decode_tests {
    use super::super::recurrence_process_pack::{
        ProcessDirectExecutorBinding, ProcessIntrinsicExecutor, ProcessJitExecutor,
        ProcessNativeExecutor,
    };
    use super::{
        DirectSourceDispatchDomainSpec, DirectSourceDispatchKey, DirectSourceDispatchVariantSpec,
        DirectSourceOrientation, DirectSourceTemplateSpec, DirectSourceWavefunctionFamily,
        OnTheFlySourceExecutionSpecV1, OnTheFlySourceOrientationV1,
        OnTheFlySourceWavefunctionFamilyV1, RecurrenceExternalLeg, RecurrenceGenericCrossingIr,
        RecurrenceGenericParticleIdentityIr, RecurrenceGenericSourceIr,
        RecurrenceMomentumTransform, RecurrenceNormalization, RecurrenceParameterProjection,
        RecurrenceParticleMass, RecurrenceParticleStatistics, RecurrenceRuntimeMetadata,
        RecurrenceRuntimeParameter, RecurrenceSourceOrientation, RecurrenceSourceTemplate,
        RecurrenceWavefunctionFamily, RuntimeParameterSlots,
        build_on_the_fly_source_domains_from_specs, companion_helicity_coefficient_is_canonical,
        invert_replay_helicity_map, process_binding_requires_evaluator_payloads, read_u64_values,
        recurrence_ready_needs_evaluator_payloads, validate_ready_source_domain_shape,
        validate_recurrence_parameter_runtime_parts,
    };
    #[cfg(feature = "on-the-fly-test-support")]
    use super::{OnTheFlyQueryTraceCacheV1, validate_on_the_fly_probe_outputs};
    use std::collections::BTreeMap;

    #[test]
    fn execution_diagnostic_inverts_nonidentity_replay_helicity_map() {
        // The replay map is representative ID -> public ID. Public helicity
        // zero is therefore represented by ID one, not by the same index.
        assert_eq!(invert_replay_helicity_map(17, &[2, 0, 1], 0).unwrap(), 1);
    }

    #[test]
    fn process_ready_evaluator_payload_predicate_is_exhaustive() {
        assert!(!process_binding_requires_evaluator_payloads(
            &ProcessDirectExecutorBinding::Source
        ));
        assert!(!process_binding_requires_evaluator_payloads(
            &ProcessDirectExecutorBinding::Intrinsic(ProcessIntrinsicExecutor {
                runtime_template: "identity-finalizer-v1".into(),
                scale: None,
            })
        ));
        assert!(process_binding_requires_evaluator_payloads(
            &ProcessDirectExecutorBinding::Jit(ProcessJitExecutor {
                optimization_level: 2,
                plane_compression: true,
                source_application_sha256: [0; 32],
                source_application_path: "application.bin".into(),
                source_application_abi: "test".into(),
                parameter_bindings: Vec::new(),
                input_plane_projections: Vec::new(),
                scalar_projections: Vec::new(),
                output_alias_inputs: Vec::new(),
            })
        ));
        assert!(process_binding_requires_evaluator_payloads(
            &ProcessDirectExecutorBinding::Native(ProcessNativeExecutor {
                prepared_kernel_id: 0,
                library_path: "kernel.so".into(),
                native_entry_point: "kernel".into(),
                coupling: None,
            })
        ));
    }

    #[test]
    fn process_ready_container_need_skips_plain_intrinsic_runtime() {
        assert!(!recurrence_ready_needs_evaluator_payloads(
            false, false, false
        ));
        for flags in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            assert!(recurrence_ready_needs_evaluator_payloads(
                flags.0, flags.1, flags.2
            ));
        }
    }

    #[test]
    fn process_ready_parameter_schema_rejects_manifest_parity_drift() {
        let defaults = vec![[1.0, 2.0]];
        let projection = vec![
            RecurrenceParameterProjection {
                runtime_slot: 0,
                runtime_name: "Z".into(),
                parameter_template_id: 0,
                prepared_parameter_id: Some(0),
                component: 0,
            },
            RecurrenceParameterProjection {
                runtime_slot: 1,
                runtime_name: "Z".into(),
                parameter_template_id: 0,
                prepared_parameter_id: Some(0),
                component: 1,
            },
        ];
        let parameters = vec![
            RecurrenceRuntimeParameter {
                name: "Z.real".into(),
                kind: "external_parameter_component".into(),
                parameter_index: 0,
                default: 1.0,
                runtime_name: Some("Z".into()),
                complex_component: Some("real".into()),
            },
            RecurrenceRuntimeParameter {
                name: "Z.imag".into(),
                kind: "external_parameter_component".into(),
                parameter_index: 1,
                default: 2.0,
                runtime_name: Some("Z".into()),
                complex_component: Some("imag".into()),
            },
        ];
        validate_recurrence_parameter_runtime_parts(&parameters, &defaults, &projection, 1)
            .unwrap();

        let mut malformed = parameters.clone();
        malformed[1].default = 3.0;
        let error =
            validate_recurrence_parameter_runtime_parts(&malformed, &defaults, &projection, 1)
                .expect_err("process-ready defaults must remain bound to prepared defaults");
        assert!(error.to_string().contains("prepared default"));

        let mut malformed = projection.clone();
        malformed[1].runtime_slot = 7;
        let error =
            validate_recurrence_parameter_runtime_parts(&parameters, &defaults, &malformed, 1)
                .expect_err("process-ready projection slots must remain dense");
        assert!(error.to_string().contains("dense"));
    }

    #[test]
    fn process_ready_sources_reject_key_or_mass_drift_from_plan() {
        let template = DirectSourceTemplateSpec {
            spin_state_class: 7,
            family: DirectSourceWavefunctionFamily::WeylFermion,
            orientation: DirectSourceOrientation::Particle,
            helicity: 1,
            chirality: -1,
            mass_parameter_index: Some(0),
        };
        let expected = vec![BTreeMap::from([(
            DirectSourceDispatchKey::SpinStateClass(7),
            7,
        )])];
        let valid = vec![DirectSourceDispatchDomainSpec {
            variants: vec![DirectSourceDispatchVariantSpec {
                key: DirectSourceDispatchKey::SpinStateClass(7),
                template,
            }],
        }];
        validate_ready_source_domain_shape(&valid, &expected, 1).unwrap();

        let mut malformed_key = valid.clone();
        malformed_key[0].variants[0].key = DirectSourceDispatchKey::SpinStateClass(8);
        let error = validate_ready_source_domain_shape(&malformed_key, &expected, 1)
            .expect_err("a process-ready source key must be plan-referenced");
        assert!(error.to_string().contains("plan keys"));

        let mut malformed_mass = valid;
        malformed_mass[0].variants[0].template.mass_parameter_index = Some(1);
        let error = validate_ready_source_domain_shape(&malformed_mass, &expected, 1)
            .expect_err("a process-ready source mass slot must be plan-bounded");
        assert!(error.to_string().contains("out of bounds"));
    }

    #[test]
    fn companion_helicity_coefficients_fail_closed_outside_physical_unit_weights() {
        assert!(companion_helicity_coefficient_is_canonical(false, 1.0));
        assert!(companion_helicity_coefficient_is_canonical(true, 0.0));
        for (structural_zero, coefficient) in
            [(false, 0.0), (false, 2.0), (true, 1.0), (true, f64::NAN)]
        {
            assert!(!companion_helicity_coefficient_is_canonical(
                structural_zero,
                coefficient,
            ));
        }
    }

    #[cfg(feature = "on-the-fly-test-support")]
    #[test]
    fn on_the_fly_query_cache_counts_builds_and_real_digest_lookups() {
        let digest = SemanticDigest::new([7; 32]).unwrap();
        let mut cache = OnTheFlyQueryTraceCacheV1::default();

        cache.insert_built(digest, 41_u32).unwrap();
        assert_eq!(cache.counts(), (1, 0));
        assert_eq!(*cache.prepared(digest).unwrap(), 41);
        assert_eq!(cache.counts(), (1, 0));
        assert_eq!(*cache.lookup(digest).unwrap(), 41);
        assert_eq!(*cache.lookup(digest).unwrap(), 41);
        assert_eq!(cache.counts(), (1, 2));

        let error = cache
            .insert_built(digest, 99)
            .expect_err("one query digest must not be rebuilt");
        assert!(error.to_string().contains("repeated a fresh query digest"));
        assert_eq!(*cache.prepared(digest).unwrap(), 41);
        assert_eq!(cache.counts(), (1, 2));
    }

    #[cfg(feature = "on-the-fly-test-support")]
    #[test]
    fn on_the_fly_probe_rejects_non_finite_outputs() {
        validate_on_the_fly_probe_outputs(&[[0.0, 1.0]], &[1.0]).unwrap();

        let raw_error = validate_on_the_fly_probe_outputs(&[[f64::NAN, 1.0]], &[1.0])
            .expect_err("non-finite raw amplitude must fail closed");
        assert!(raw_error.to_string().contains("non-finite raw amplitude"));

        let normalized_error = validate_on_the_fly_probe_outputs(&[[0.0, 1.0]], &[f64::INFINITY])
            .expect_err("non-finite normalized value must fail closed");
        assert!(
            normalized_error
                .to_string()
                .contains("non-finite normalized value")
        );
    }

    fn source_ir(
        name: &str,
        pdg: i32,
        family: RecurrenceWavefunctionFamily,
        statistics: RecurrenceParticleStatistics,
        dimension: u64,
        mass_parameter: Option<&str>,
    ) -> RecurrenceGenericSourceIr {
        RecurrenceGenericSourceIr {
            identity: RecurrenceGenericParticleIdentityIr {
                canonical_id: name.into(),
                species_id: name.into(),
                anti_canonical_id: format!("anti-{name}"),
                display_name: name.into(),
                anti_display_name: format!("anti-{name}"),
                pdg_label: pdg,
                anti_pdg_label: -pdg,
                orientation: if statistics == RecurrenceParticleStatistics::Fermion {
                    RecurrenceSourceOrientation::Particle
                } else {
                    RecurrenceSourceOrientation::SelfConjugate
                },
                self_conjugate: statistics != RecurrenceParticleStatistics::Fermion,
            },
            statistics,
            wavefunction_family: family,
            component_dimension: dimension,
            states: Vec::new(),
            crossing: RecurrenceGenericCrossingIr {
                momentum_transform: RecurrenceMomentumTransform::Identity,
                helicity_factor: 1,
                chirality_factor: 1,
                spin_state_factor: 1,
                phase: [1.0, 0.0],
            },
            basis: "test".into(),
            mass_parameter: mass_parameter.map(str::to_owned),
            width_parameter: None,
        }
    }

    fn source_domain_metadata() -> RecurrenceRuntimeMetadata {
        RecurrenceRuntimeMetadata {
            public_color_flows: Vec::new(),
            runtime_parameters: Vec::new(),
            prepared_parameter_defaults: vec![[0.0, 0.0]; 4],
            parameter_projection: vec![RecurrenceParameterProjection {
                runtime_slot: 0,
                runtime_name: "MF".into(),
                parameter_template_id: 0,
                prepared_parameter_id: Some(3),
                component: 0,
            }],
            source_templates: vec![
                RecurrenceSourceTemplate {
                    source_template_id: 0,
                    _current_state_template_id: 0,
                    dimension: 2,
                    helicity: -1,
                    chirality: 1,
                    spin_state: 7,
                    source_ir: source_ir(
                        "f",
                        1,
                        RecurrenceWavefunctionFamily::Fermion,
                        RecurrenceParticleStatistics::Fermion,
                        2,
                        Some("MF"),
                    ),
                    crossing: RecurrenceGenericCrossingIr {
                        momentum_transform: RecurrenceMomentumTransform::NegateFourMomentum,
                        helicity_factor: -1,
                        chirality_factor: -1,
                        spin_state_factor: 1,
                        phase: [1.0, 0.0],
                    },
                },
                RecurrenceSourceTemplate {
                    source_template_id: 2,
                    _current_state_template_id: 2,
                    dimension: 1,
                    helicity: 0,
                    chirality: 0,
                    spin_state: 0,
                    source_ir: source_ir(
                        "s",
                        25,
                        RecurrenceWavefunctionFamily::Scalar,
                        RecurrenceParticleStatistics::Boson,
                        1,
                        None,
                    ),
                    crossing: RecurrenceGenericCrossingIr {
                        momentum_transform: RecurrenceMomentumTransform::Identity,
                        helicity_factor: 1,
                        chirality_factor: 1,
                        spin_state_factor: 1,
                        phase: [1.0, 0.0],
                    },
                },
            ],
            external_legs: vec![
                RecurrenceExternalLeg {
                    source_slot: 0,
                    public_label: 0,
                    physical_pdg: 1,
                    outgoing_pdg: 1,
                    is_initial: true,
                },
                RecurrenceExternalLeg {
                    source_slot: 1,
                    public_label: 1,
                    physical_pdg: 25,
                    outgoing_pdg: 25,
                    is_initial: false,
                },
                RecurrenceExternalLeg {
                    source_slot: 2,
                    public_label: 2,
                    physical_pdg: 1,
                    outgoing_pdg: 1,
                    is_initial: false,
                },
            ],
            particle_masses: vec![RecurrenceParticleMass {
                outgoing_pdg: 1,
                mass: 2.0,
            }],
            normalization: RecurrenceNormalization {
                color_accuracy: "lc".into(),
                color_factor: 1.0,
                average_factor: 1.0,
                identical_factor: 1.0,
                global_coupling_factor: 1.0,
                qcd_coupling_power: None,
                electroweak_coupling_power: None,
                couplings_in_stage_evaluators: true,
                coupling_policy: "test".into(),
            },
            color_contraction: None,
        }
    }

    fn crossed_fermion(
        source_slot: u32,
        helicity: i32,
        chirality: i32,
    ) -> OnTheFlySourceExecutionSpecV1 {
        OnTheFlySourceExecutionSpecV1 {
            source_slot,
            source_template_id: 0,
            spin_state_class: 7,
            family: OnTheFlySourceWavefunctionFamilyV1::WeylFermion,
            orientation: OnTheFlySourceOrientationV1::Particle,
            helicity,
            chirality,
            prepared_mass_parameter_slot: Some(3),
        }
    }

    fn scalar_source() -> OnTheFlySourceExecutionSpecV1 {
        OnTheFlySourceExecutionSpecV1 {
            source_slot: 1,
            source_template_id: 2,
            spin_state_class: 0,
            family: OnTheFlySourceWavefunctionFamilyV1::Scalar,
            orientation: OnTheFlySourceOrientationV1::SelfConjugate,
            helicity: 0,
            chirality: 0,
            prepared_mass_parameter_slot: None,
        }
    }

    #[test]
    fn truncated_support_words_return_an_artifact_error() {
        let mut cursor = 0;
        let error = read_u64_values(&[0; 7], &mut cursor, 1, "recurrence support mask")
            .expect_err("truncated support words must be rejected");
        assert!(
            error
                .to_string()
                .contains("truncated recurrence support mask")
        );
        assert_eq!(cursor, 0);
    }

    #[test]
    fn compact_sources_match_crossing_mass_and_sparse_inert_domains() {
        let metadata = source_domain_metadata();
        let slots = BTreeMap::from([(
            "MF".into(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        )]);
        let fermion = crossed_fermion(0, 1, -1);
        let domains = build_on_the_fly_source_domains_from_specs(
            [fermion, fermion, scalar_source()],
            4,
            4,
            &metadata,
            &slots,
        )
        .unwrap();
        assert_eq!(domains.len(), 4);
        assert_eq!(domains[0].variants.len(), 1);
        assert_eq!(
            domains[0].variants[0].key,
            DirectSourceDispatchKey::SpinStateClass(7)
        );
        let template = domains[0].variants[0].template;
        assert_eq!(template.family, DirectSourceWavefunctionFamily::WeylFermion);
        assert_eq!(template.orientation, DirectSourceOrientation::Particle);
        assert_eq!((template.helicity, template.chirality), (1, -1));
        assert_eq!(template.mass_parameter_index, Some(3));
        assert_eq!(domains[1], domains[0]);
        assert_eq!(domains[3], domains[0]);
        assert_eq!(domains[2].variants.len(), 1);
        assert_eq!(
            domains[2].variants[0].template.family,
            DirectSourceWavefunctionFamily::Scalar
        );
    }

    #[test]
    fn compact_source_without_prebound_mass_uses_authoritative_source_ir_slot() {
        let metadata = source_domain_metadata();
        let slots = BTreeMap::from([(
            "MF".into(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        )]);
        let mut fermion = crossed_fermion(0, 1, -1);
        fermion.prepared_mass_parameter_slot = None;

        let domains =
            build_on_the_fly_source_domains_from_specs([fermion], 4, 4, &metadata, &slots).unwrap();

        assert_eq!(
            domains[0].variants[0].template.mass_parameter_index,
            Some(3)
        );
    }

    #[test]
    fn compact_source_rejects_conflicting_prebound_mass_slot() {
        let metadata = source_domain_metadata();
        let slots = BTreeMap::from([(
            "MF".into(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        )]);
        let mut fermion = crossed_fermion(0, 1, -1);
        fermion.prepared_mass_parameter_slot = Some(2);

        let error = build_on_the_fly_source_domains_from_specs([fermion], 4, 4, &metadata, &slots)
            .expect_err("an explicit compact mass slot must match SourceIR");

        assert!(error.to_string().contains("prepared mass parameter slot"));
    }

    #[test]
    fn compact_source_still_rejects_non_mass_semantic_mismatch() {
        let metadata = source_domain_metadata();
        let slots = BTreeMap::from([(
            "MF".into(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        )]);
        let mut fermion = crossed_fermion(0, 1, -1);
        fermion.helicity = -1;

        let error = build_on_the_fly_source_domains_from_specs([fermion], 4, 4, &metadata, &slots)
            .expect_err("compact helicity must remain exact");

        assert!(error.to_string().contains("source helicity"));
    }

    #[test]
    fn compact_sources_reject_one_key_with_conflicting_crossed_semantics() {
        let metadata = source_domain_metadata();
        let slots = BTreeMap::from([(
            "MF".into(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        )]);
        let error = build_on_the_fly_source_domains_from_specs(
            [crossed_fermion(0, 1, -1), crossed_fermion(2, -1, 1)],
            4,
            4,
            &metadata,
            &slots,
        )
        .expect_err("one dispatch key cannot have two crossed SourceIR meanings");
        assert!(error.to_string().contains("maps one dispatch key"));
    }
}
