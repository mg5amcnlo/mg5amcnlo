// SPDX-License-Identifier: 0BSD

//! Authenticated construction boundary for the private on-the-fly lane.
//!
//! Loading stops at the compact process seed and model-wide prepared catalogs.
//! It never opens or constructs a process-wide [`crate::recurrence::DirectRecurrencePlan`]
//! and it never materializes dense helicity/color axes. NLC/full authenticate
//! one recurrence-v3 color payload against the compact structural owner basis.
//! The existing runtime adapter owns public selector decoding after this boundary.

use super::eager_manifest::PreparedKernelPackManifest;
use super::on_the_fly_lane::OnTheFlyNativeRuntime;
use super::on_the_fly_manifest::{
    ON_THE_FLY_PROCESS_SEED_MEMBER, OnTheFlyColorCoverage, OnTheFlyExecutionManifest,
    OnTheFlyRuntimeContainer,
};
use super::on_the_fly_public_metadata::{
    OnTheFlyPublicMetadataV1, parse_on_the_fly_public_metadata,
};
use super::on_the_fly_selectors::{
    OnTheFlyCompactSelectorAdapterV1, OnTheFlyLcColorCoverageV1, OnTheFlyLcSelectorPolicyV1,
};
use super::recurrence_backend::{
    NativeRecurrencePreparedExecutorPool, OnTheFlySourceDomainBinding,
};
use super::recurrence_bootstrap::{
    RecurrenceReadyCompanionV1, RecurrenceReadyExecutionV1, RecurrenceReadyPlanV1,
};
use super::recurrence_lane::{
    PreparedParameterProjectionEntry, RecurrencePersistedHelicitySelectorRuntime,
};
use super::recurrence_load::{
    LoadedRecurrencePlan, build_direct_source_domains, build_on_the_fly_source_domains,
    decode_ready_source_domains, decode_sha256, direct_helicity_to_physics,
    load_color_contraction_reference, load_plan_summary, runtime_parameter_slots,
    validate_ready_direct_helicity_to_physics, validate_recurrence_prepared_pack_outer_target,
};
use super::*;
use crate::pacbin::{PacbinMemberKind, PacbinReader};
use crate::recurrence::on_the_fly::{
    OnTheFlyExternalColorRoleV1, OnTheFlyProcessSeedV1, decode_on_the_fly_process_seed_v1,
    validate_on_the_fly_source_mass_bindings_v1,
};
use crate::recurrence::template_json::project_owned_recurrence_template_catalog_json_v1;
use crate::recurrence::{
    FactorizedColorContractionKind, RecurrenceColorAccuracy, RecurrenceColorContraction,
    RecurrenceColorStorage, decode_recurrence_helicity_dispatch_v1,
};
use std::rc::Rc;

pub(super) struct LoadedOnTheFlyColorContractionV1 {
    pub(super) plan: RecurrenceColorContraction,
    pub(super) destination_by_owner_ordinal: Box<[u32]>,
    pub(super) point_tile_size: usize,
}

pub(super) struct LoadedOnTheFlyRuntime {
    pub(super) common: ExecutionRuntime,
    pub(super) lane: OnTheFlyNativeRuntime,
    pub(super) selectors: OnTheFlyCompactSelectorAdapterV1,
    pub(super) metadata_selectors: OnTheFlyCompactSelectorAdapterV1,
    pub(super) public_metadata: OnTheFlyPublicMetadataV1,
    pub(super) color_contraction: Option<LoadedOnTheFlyColorContractionV1>,
}

pub(super) struct LoadedRecurrenceHelicitySelectorCompanion {
    pub(super) runtime: RecurrencePersistedHelicitySelectorRuntime,
}

fn clamp_query_construction_threads(requested: usize, available: usize) -> usize {
    requested.min(available.max(1))
}

pub(super) fn effective_query_construction_threads(requested: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    clamp_query_construction_threads(requested, available)
}

fn validate_recurrence_companion_process_digest(
    primary_process_digest: &str,
    companion_process_digest: &str,
    decoded_seed_process_digest: &str,
) -> RusticolResult<()> {
    if primary_process_digest != companion_process_digest
        || primary_process_digest != decoded_seed_process_digest
    {
        return Err(RusticolError::integrity(
            "recurrence helicity-selector companion is bound to a different primary process",
        ));
    }
    Ok(())
}

pub(super) fn load_on_the_fly_native_runtime(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &OnTheFlyExecutionManifest,
    selection: &crate::ArtifactSelection,
) -> RusticolResult<LoadedOnTheFlyRuntime> {
    let public_metadata = load_public_metadata(artifact, manifest, &selection.process)?;
    let seed = load_process_seed(artifact, evaluator_root, manifest)?;
    if seed.identity() != manifest.runtime_metadata.process_seed_identity {
        return Err(RusticolError::integrity(
            "on-the-fly execution metadata does not identify its decoded process seed",
        ));
    }
    let (mut pack, payload_root) = load_prepared_pack(artifact, manifest)?;
    let raw_templates = pack.recurrence_template.take().ok_or_else(|| {
        RusticolError::compatibility(
            "on-the-fly execution requires the prepared recurrence template catalog",
        )
    })?;
    let raw_template_json = serde_json::from_str(raw_templates.get()).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse prepared recurrence template catalog: {error}"
        ))
    })?;
    drop(raw_templates);
    let templates =
        project_owned_recurrence_template_catalog_json_v1(raw_template_json)?.validate()?;
    let summary = templates.summary();
    if seed.template_catalog_digest() != summary.catalog_digest
        || seed.model_digest() != summary.compiled_model_digest
        || seed.prepared_pack_digest() != summary.prepared_kernel_pack_digest
    {
        return Err(RusticolError::integrity(
            "on-the-fly process seed disagrees with the authenticated recurrence template catalog",
        ));
    }
    let parameter_projection = manifest
        .runtime_metadata
        .parameter_projection
        .iter()
        .map(|row| {
            (
                row.parameter_template_id,
                row.prepared_parameter_id,
                row.component,
            )
        })
        .collect::<Vec<_>>();
    validate_on_the_fly_source_mass_bindings_v1(&seed, &templates, &parameter_projection)?;

    let payloads = artifact.evaluator_payload_store(&payload_root)?;
    let pool = NativeRecurrencePreparedExecutorPool::load_without_plan_from_validated_pack(
        &mut pack,
        &payloads,
        &seed.prepared_pack_digest().to_string(),
        &seed.direct_catalog_digest().to_string(),
    )?;
    let direct_catalog = pool.prepared_direct_catalog()?;
    if direct_catalog.direct_template_catalog_digest() != seed.direct_catalog_digest() {
        return Err(RusticolError::integrity(
            "on-the-fly prepared Direct-Arena catalog disagrees with the compact process seed",
        ));
    }
    let source_domains = build_on_the_fly_source_domains(&seed, &templates)?;
    let sources = pool.bind_source_domains(source_domains)?;
    let resolver = pool.into_on_the_fly_resolver(sources);

    let (mut common, defaults, projection) = build_common_runtime(manifest)?;
    common.model_parameter_evaluator =
        super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
            &pack,
            &common.model_parameters,
            &payloads,
        )?;
    common.refresh_derived_model_parameters()?;

    let metadata_selectors = OnTheFlyCompactSelectorAdapterV1::from_seed(
        &seed,
        selector_policy(&manifest.selector_policy),
    )?;
    let public_color_count = if manifest.uses_contracted_color() {
        1
    } else {
        metadata_selectors.color_count()
    };
    manifest
        .selector_policy
        .selector_census
        .validate_against(metadata_selectors.helicity_count(), public_color_count)?;
    let color_contraction = load_on_the_fly_color_contraction(
        artifact,
        evaluator_root,
        OnTheFlyColorContractionLoadV1 {
            process_key: &manifest.key,
            color_accuracy: &manifest.color_accuracy,
            requested_point_tile_size: manifest.runtime_options.point_tile_size,
            reference: manifest.runtime_metadata.color_contraction.as_ref(),
            seed: &seed,
            selectors: &metadata_selectors,
        },
    )?;
    let selectors = metadata_selectors
        .clone()
        .with_public_permutation(&selection.external_permutation)?;
    let requested_query_construction_threads =
        usize::try_from(manifest.runtime_options.query_construction_threads).map_err(|_| {
            RusticolError::artifact("on-the-fly query construction thread count exceeds usize")
        })?;
    let effective_query_construction_threads =
        effective_query_construction_threads(requested_query_construction_threads);
    let mut lane = OnTheFlyNativeRuntime::new(
        templates,
        direct_catalog,
        seed,
        resolver,
        requested_query_construction_threads,
        effective_query_construction_threads,
        defaults,
        projection,
        &common.model_parameter_values_f64,
    )?;
    if let Some(workspace) = symmetric_group_workspace_for_loaded_color(color_contraction.as_ref())?
    {
        lane.install_symmetric_group_color_workspace(workspace)?;
    }
    Ok(LoadedOnTheFlyRuntime {
        common,
        lane,
        selectors,
        metadata_selectors,
        public_metadata,
        color_contraction,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_recurrence_helicity_selector_companion(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &super::recurrence_manifest::RecurrenceExecutionManifest,
    physics: &ProcessPhysicsV1,
    payloads: &EvaluatorPayloadStore,
    common: &ExecutionRuntime,
) -> RusticolResult<LoadedRecurrenceHelicitySelectorCompanion> {
    let companion = manifest
        .helicity_selector_companion
        .as_ref()
        .ok_or_else(|| {
            RusticolError::integrity(
                "recurrence companion loader was called without a companion descriptor",
            )
        })?;
    let LoadedRecurrencePlan {
        plan,
        process_executors,
    } = load_plan_summary(artifact, evaluator_root, manifest, &companion.plan)?;
    validate_recurrence_companion_process_digest(
        &manifest.plan.process_binding.process_digest,
        &companion.process_digest,
        &companion.plan.process_binding.process_digest,
    )?;
    let dispatch = load_recurrence_companion_dispatch(
        artifact,
        &plan,
        companion.plan.helicity_dispatch.as_ref(),
    )?;
    let source_domains = build_direct_source_domains(
        &plan,
        &manifest.runtime_metadata,
        &common.model_parameter_runtime_slots,
    )?;
    let direct_helicity_to_physics = direct_helicity_to_physics(&plan, physics)?;
    finish_recurrence_helicity_selector_companion(
        process_executors,
        plan,
        dispatch,
        payloads,
        RecurrenceCompanionModelIdentities {
            compiled_model_digest: &manifest.compiled_model_digest,
            recurrence_template_catalog_digest: &manifest.recurrence_template_catalog_digest,
            prepared_kernel_pack_digest: &manifest.prepared_kernel_pack_digest,
            direct_template_catalog_digest: &manifest.direct_template_catalog_digest,
        },
        source_domains,
        direct_helicity_to_physics,
        physics.helicities.len(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_recurrence_helicity_selector_companion_from_ready_parts(
    artifact: &VerifiedArtifact,
    loaded_plan: LoadedRecurrencePlan,
    plan_reference: &RecurrenceReadyPlanV1,
    companion: &RecurrenceReadyCompanionV1,
    ready: &RecurrenceReadyExecutionV1,
    physics: &ProcessPhysicsV1,
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<LoadedRecurrenceHelicitySelectorCompanion> {
    validate_recurrence_companion_process_digest(
        &ready.primary_plan.process_binding.process_digest,
        &companion.process_digest,
        &plan_reference.process_binding.process_digest,
    )?;
    let LoadedRecurrencePlan {
        plan,
        process_executors,
    } = loaded_plan;
    let dispatch = load_recurrence_companion_dispatch(
        artifact,
        &plan,
        plan_reference.helicity_dispatch.as_ref(),
    )?;
    let source_domains = decode_ready_source_domains(&companion.source_domains, &plan)?;
    let direct_helicity_to_physics = validate_ready_direct_helicity_to_physics(
        &plan,
        &companion.direct_helicity_to_physics,
        physics,
    )?;
    finish_recurrence_helicity_selector_companion(
        process_executors,
        plan,
        dispatch,
        payloads,
        RecurrenceCompanionModelIdentities {
            compiled_model_digest: &ready.compiled_model_digest,
            recurrence_template_catalog_digest: &ready.recurrence_template_catalog_digest,
            prepared_kernel_pack_digest: &ready.prepared_kernel_pack_digest,
            direct_template_catalog_digest: &ready.direct_template_catalog_digest,
        },
        source_domains,
        direct_helicity_to_physics,
        physics.helicities.len(),
    )
}

fn load_recurrence_companion_dispatch(
    artifact: &VerifiedArtifact,
    plan: &crate::recurrence::DirectRecurrencePlan,
    reference: Option<&super::recurrence_manifest::RecurrenceHelicityDispatchReference>,
) -> RusticolResult<crate::recurrence::DirectHelicityDispatch> {
    let reference = reference.ok_or_else(|| {
        RusticolError::integrity("recurrence helicity-selector plan has no exact helicity dispatch")
    })?;
    let record = artifact.payload(&reference.path)?;
    if record.role != PayloadRole::EvaluatorState
        || record.media_type != "application/octet-stream"
        || record.process_id.is_some()
        || record.executable
        || record.size_bytes != reference.size_bytes
        || record.sha256 != reference.sha256
    {
        return Err(RusticolError::integrity(
            "recurrence helicity dispatch disagrees with its authenticated payload",
        ));
    }
    let bytes = artifact.read_payload(&reference.path)?;
    let dispatch = decode_recurrence_helicity_dispatch_v1(&bytes)?;
    dispatch.validate_for_plan(plan)?;
    if dispatch.runtime_layout_digest().to_string() != reference.base_runtime_layout_digest
        || dispatch.resolved_helicity_count() as u64 != reference.resolved_helicity_count
    {
        return Err(RusticolError::integrity(
            "decoded recurrence helicity dispatch disagrees with its bounded summary",
        ));
    }
    Ok(dispatch)
}

struct RecurrenceCompanionModelIdentities<'a> {
    compiled_model_digest: &'a str,
    recurrence_template_catalog_digest: &'a str,
    prepared_kernel_pack_digest: &'a str,
    direct_template_catalog_digest: &'a str,
}

#[allow(clippy::too_many_arguments)]
fn finish_recurrence_helicity_selector_companion(
    process_executors: super::recurrence_process_pack::ProcessDirectExecutorPack,
    plan: crate::recurrence::DirectRecurrencePlan,
    dispatch: crate::recurrence::DirectHelicityDispatch,
    payloads: &EvaluatorPayloadStore,
    identities: RecurrenceCompanionModelIdentities<'_>,
    source_domains: Vec<super::evaluator::recurrence_source_direct::DirectSourceDispatchDomainSpec>,
    direct_helicity_to_physics: Vec<usize>,
    physics_helicity_count: usize,
) -> RusticolResult<LoadedRecurrenceHelicitySelectorCompanion> {
    let pool = Rc::new(
        NativeRecurrencePreparedExecutorPool::load_for_direct_plan_from_process_pack(
            &process_executors,
            payloads,
            &plan,
        )?,
    );
    pool.validate_model_identities(
        identities.compiled_model_digest,
        identities.recurrence_template_catalog_digest,
        identities.prepared_kernel_pack_digest,
        identities.direct_template_catalog_digest,
    )?;
    let sources = OnTheFlySourceDomainBinding::load_for_process_plan(&plan, source_domains)?;
    let resolver = Rc::clone(&pool).into_shared_on_the_fly_resolver(sources);
    let runtime = RecurrencePersistedHelicitySelectorRuntime::new(
        plan,
        dispatch,
        resolver,
        direct_helicity_to_physics,
        physics_helicity_count,
    )?;
    Ok(LoadedRecurrenceHelicitySelectorCompanion { runtime })
}

fn symmetric_group_workspace_for_loaded_color(
    loaded: Option<&LoadedOnTheFlyColorContractionV1>,
) -> RusticolResult<Option<crate::recurrence::RuntimeSymmetricGroupColorWorkspace>> {
    let Some((reducer, point_tile_size)) = loaded.and_then(|loaded| {
        loaded
            .plan
            .runtime_reducer()
            .and_then(|reducer| match reducer {
                crate::recurrence::RuntimeColorContractionReducer::SymmetricGroupFourier(value) => {
                    Some((value, loaded.point_tile_size))
                }
                _ => None,
            })
    }) else {
        return Ok(None);
    };
    // `point_tile_size` was already clamped against the shared 512 MiB
    // workspace budget while authenticating the color payload.  Allocate the
    // final capacity here so the first multi-point evaluation is warmed too.
    Ok(Some(reducer.workspace(point_tile_size)?))
}

fn load_public_metadata(
    artifact: &VerifiedArtifact,
    manifest: &OnTheFlyExecutionManifest,
    outer: &crate::ArtifactProcess,
) -> RusticolResult<OnTheFlyPublicMetadataV1> {
    let record = artifact.payload(&outer.physics_path)?;
    if record.role != PayloadRole::RuntimePhysics
        || record.media_type != "application/json"
        || record.executable
        || record.process_id.as_deref() != Some(manifest.key.as_str())
    {
        return Err(RusticolError::integrity(
            "on-the-fly compact public metadata has the wrong authenticated payload role",
        ));
    }
    let bytes = artifact.read_payload(&outer.physics_path)?;
    parse_on_the_fly_public_metadata(&bytes, &outer.physics_path, outer, manifest)
}

fn selector_policy(
    policy: &super::on_the_fly_manifest::OnTheFlySelectorPolicy,
) -> OnTheFlyLcSelectorPolicyV1 {
    let color_coverage = match policy.color_coverage {
        OnTheFlyColorCoverage::Complete | OnTheFlyColorCoverage::Contracted => {
            // NLC/full still use the compact complete LC structural basis
            // internally; only the public axis is contracted.
            OnTheFlyLcColorCoverageV1::Complete
        }
    };
    OnTheFlyLcSelectorPolicyV1 {
        color_coverage,
        color_basis: policy.color_basis,
        reference_color_word: policy
            .reference_color_word
            .clone()
            .map(Vec::into_boxed_slice),
        trace_reflections_folded: policy.trace_reflections_folded,
    }
}

struct OnTheFlyColorContractionLoadV1<'a> {
    process_key: &'a str,
    color_accuracy: &'a str,
    requested_point_tile_size: u32,
    reference: Option<&'a super::recurrence_manifest::RecurrenceColorContractionReference>,
    seed: &'a OnTheFlyProcessSeedV1,
    selectors: &'a OnTheFlyCompactSelectorAdapterV1,
}

fn load_on_the_fly_color_contraction(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    request: OnTheFlyColorContractionLoadV1<'_>,
) -> RusticolResult<Option<LoadedOnTheFlyColorContractionV1>> {
    let OnTheFlyColorContractionLoadV1 {
        process_key,
        color_accuracy,
        requested_point_tile_size,
        reference,
        seed,
        selectors,
    } = request;
    let Some(reference) = reference else {
        return Ok(None);
    };
    let plan = load_color_contraction_reference(artifact, evaluator_root, process_key, reference)?;
    let expected_accuracy = match color_accuracy {
        "nlc" => RecurrenceColorAccuracy::Nlc,
        "full" => RecurrenceColorAccuracy::Full,
        _ => {
            return Err(RusticolError::integrity(
                "LC on-the-fly execution unexpectedly loaded contracted color",
            ));
        }
    };
    let owner_count = selectors.color_count();
    let owner_count_u32 = u32::try_from(owner_count).map_err(|_| {
        RusticolError::artifact("on-the-fly contracted structural owner count exceeds u32")
    })?;
    let fundamental_count = seed
        .source_anchors()
        .iter()
        .filter(|anchor| anchor.color_role() == OnTheFlyExternalColorRoleV1::Fundamental)
        .count();
    let alias_multiplicity = (1..=fundamental_count).try_fold(1usize, |value, factor| {
        value.checked_mul(factor).ok_or_else(|| {
            RusticolError::artifact(
                "on-the-fly contracted open-line alias multiplicity exceeds usize",
            )
        })
    })?;
    let expected_sector_count = owner_count.checked_mul(alias_multiplicity).ok_or_else(|| {
        RusticolError::artifact("on-the-fly contracted structural sector count exceeds usize")
    })?;
    let plan_sector_count = usize::try_from(plan.sector_count()).map_err(|_| {
        RusticolError::artifact("on-the-fly contracted payload sector count exceeds usize")
    })?;
    let direct_storage =
        plan.storage() == RecurrenceColorStorage::Expanded && plan.factorization().is_none();
    let symmetric_group_storage = plan.storage() == RecurrenceColorStorage::ConvolutionKernels
        && plan.factorization().is_some_and(|factorization| {
            factorization.kind() == FactorizedColorContractionKind::SymmetricGroupFourier
        });
    if plan.accuracy() != expected_accuracy
        || !(direct_storage || symmetric_group_storage)
        || !plan.includes_color_factor()
        || plan.component_count() != 1
        || plan.active_sector_count() != owner_count
        || plan.group_count() != owner_count_u32
        || plan.destination_count() != owner_count_u32
        || plan_sector_count != expected_sector_count
        || plan.owner_by_sector().contains(&u32::MAX)
    {
        return Err(RusticolError::integrity(
            "on-the-fly contracted color payload disagrees with the compact structural owner basis",
        ));
    }

    let fixed_owner_sectors = plan
        .owner_by_sector()
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(sector, owner)| (owner as usize == sector).then_some(owner))
        .collect::<Vec<_>>();
    if fixed_owner_sectors.len() != owner_count {
        return Err(RusticolError::integrity(
            "on-the-fly contracted color owner fixed points are incomplete",
        ));
    }
    let owner_class_sizes = plan.owner_by_sector().iter().copied().fold(
        BTreeMap::<u32, usize>::new(),
        |mut counts, owner| {
            *counts.entry(owner).or_default() += 1;
            counts
        },
    );
    if owner_class_sizes.len() != owner_count
        || owner_class_sizes
            .values()
            .any(|count| *count != alias_multiplicity)
    {
        return Err(RusticolError::integrity(
            "on-the-fly contracted owner classes disagree with whole-open-line block aliases",
        ));
    }
    let group_by_owner_sector = plan
        .sector_by_group()
        .iter()
        .copied()
        .zip(plan.component_by_group().iter().copied())
        .enumerate()
        .map(|(group_id, (sector, component))| {
            if component != 0 {
                return Err(RusticolError::integrity(
                    "on-the-fly contracted group has a nonzero component",
                ));
            }
            let group_id = u32::try_from(group_id).map_err(|_| {
                RusticolError::artifact("on-the-fly contracted group ID exceeds u32")
            })?;
            Ok((sector, group_id))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    if group_by_owner_sector.len() != owner_count {
        return Err(RusticolError::integrity(
            "on-the-fly contracted groups do not cover every structural owner once",
        ));
    }
    let destination_by_owner_ordinal = if direct_storage {
        let mut destinations = Vec::new();
        destinations
            .try_reserve_exact(owner_count)
            .map_err(|error| {
                RusticolError::artifact(format!(
                    "could not reserve on-the-fly contracted destination map: {error}"
                ))
            })?;
        for (owner_ordinal, owner_sector) in fixed_owner_sectors.iter().copied().enumerate() {
            let group_id = group_by_owner_sector
                .get(&owner_sector)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "on-the-fly contracted owner has no component-zero group",
                    )
                })?;
            let destination = plan.destination_by_group()[group_id as usize];
            let expected_destination = u32::try_from(owner_ordinal).map_err(|_| {
                RusticolError::artifact("on-the-fly contracted owner ordinal exceeds u32")
            })?;
            if destination != expected_destination
                || plan.ordered_group_ids().get(owner_ordinal).copied() != Some(group_id)
            {
                return Err(RusticolError::integrity(
                    "on-the-fly contracted destination order disagrees with the compact structural owner basis",
                ));
            }
            destinations.push(destination);
        }
        destinations
    } else {
        // Symmetric-group groups are channel/permutation ordered.  Their
        // authenticated destination projection maps that order back to the
        // compact selector-owner ordinals, so query construction consumes the
        // inverse map (owner ordinal -> Fourier group ID).
        let mut destinations = vec![u32::MAX; owner_count];
        for (group_id, owner_ordinal) in plan.destination_by_group().iter().copied().enumerate() {
            let owner_ordinal = usize::try_from(owner_ordinal).map_err(|_| {
                RusticolError::artifact("on-the-fly contracted owner ordinal exceeds usize")
            })?;
            let slot = destinations.get_mut(owner_ordinal).ok_or_else(|| {
                RusticolError::integrity(
                    "on-the-fly symmetric-group destination is outside the owner domain",
                )
            })?;
            if *slot != u32::MAX {
                return Err(RusticolError::integrity(
                    "on-the-fly symmetric-group destination projection is not one-to-one",
                ));
            }
            *slot = u32::try_from(group_id).map_err(|_| {
                RusticolError::artifact("on-the-fly symmetric-group group ID exceeds u32")
            })?;
        }
        if destinations.contains(&u32::MAX) {
            return Err(RusticolError::integrity(
                "on-the-fly symmetric-group destination projection is incomplete",
            ));
        }
        for (owner_ordinal, owner_sector) in fixed_owner_sectors.iter().copied().enumerate() {
            let expected_group = group_by_owner_sector
                .get(&owner_sector)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "on-the-fly contracted owner has no component-zero group",
                    )
                })?;
            if destinations[owner_ordinal] != expected_group {
                return Err(RusticolError::integrity(
                    "on-the-fly symmetric-group projection disagrees with the authenticated owner sectors",
                ));
            }
        }
        destinations
    };
    let requested_point_tile_size = usize::try_from(requested_point_tile_size).map_err(|_| {
        RusticolError::artifact("on-the-fly contracted point tile size exceeds usize")
    })?;
    let point_tile_size = match plan.runtime_reducer() {
        Some(crate::recurrence::RuntimeColorContractionReducer::SymmetricGroupFourier(reducer)) => {
            reducer.bounded_lane_capacity(requested_point_tile_size)?
        }
        _ => requested_point_tile_size,
    };
    Ok(Some(LoadedOnTheFlyColorContractionV1 {
        plan,
        destination_by_owner_ordinal: destination_by_owner_ordinal.into_boxed_slice(),
        point_tile_size,
    }))
}

fn load_process_seed(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &OnTheFlyExecutionManifest,
) -> RusticolResult<crate::recurrence::on_the_fly::OnTheFlyProcessSeedV1> {
    load_process_seed_from_container(
        artifact,
        evaluator_root,
        &manifest.key,
        &manifest.runtime_container,
    )
}

fn load_process_seed_from_container(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    process_key: &str,
    container: &OnTheFlyRuntimeContainer,
) -> RusticolResult<crate::recurrence::on_the_fly::OnTheFlyProcessSeedV1> {
    let payloads = artifact.evaluator_payload_store(evaluator_root)?;
    let logical_path = payloads.logical_path(&container.path)?;
    let record = artifact.payload(&logical_path)?;
    if record.role != PayloadRole::EvaluatorState
        || record.media_type != "application/octet-stream"
        || record.executable
        || record.process_id.as_deref() != Some(process_key)
    {
        return Err(RusticolError::integrity(
            "on-the-fly runtime container has the wrong authenticated payload role",
        ));
    }
    let path = artifact.root().join(&logical_path);
    let file = artifact.open_payload_file(&logical_path)?;
    let expected_sha = decode_sha256(&record.sha256)?;
    let reader = PacbinReader::open_file_with_sha256(file, &path, &expected_sha)?;
    if reader.members().len() != 1 {
        return Err(RusticolError::integrity(
            "on-the-fly runtime container must contain exactly its compact process seed",
        ));
    }
    let member = reader.member(ON_THE_FLY_PROCESS_SEED_MEMBER)?;
    if member.logical_path() != container.seed_member_path
        || member.kind() != PacbinMemberKind::OnTheFlyProcessSeed
    {
        return Err(RusticolError::integrity(
            "on-the-fly runtime container does not contain the canonical process-seed member",
        ));
    }
    decode_on_the_fly_process_seed_v1(reader.member_bytes(ON_THE_FLY_PROCESS_SEED_MEMBER)?)
}

fn load_prepared_pack(
    artifact: &VerifiedArtifact,
    manifest: &OnTheFlyExecutionManifest,
) -> RusticolResult<(PreparedKernelPackManifest, PathBuf)> {
    let path = confined_internal_path(
        &manifest.kernel_pack.manifest_path,
        "on-the-fly prepared kernel-pack manifest path",
    )?;
    let logical = path.to_str().ok_or_else(|| {
        RusticolError::security("on-the-fly prepared kernel-pack path is not UTF-8")
    })?;
    if artifact.payload(logical)?.role != PayloadRole::EvaluatorManifest {
        return Err(RusticolError::security(
            "on-the-fly prepared kernel-pack manifest has the wrong payload role",
        ));
    }
    let bytes = artifact.read_payload(logical)?;
    let pack: PreparedKernelPackManifest = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse on-the-fly prepared kernel pack: {error}"
        ))
    })?;
    pack.validate()?;
    validate_recurrence_prepared_pack_outer_target(&artifact.manifest().producer.target, &pack)?;
    let payload_root = artifact.root().join(confined_internal_path(
        &manifest.kernel_pack.payload_root,
        "on-the-fly prepared kernel payload root",
    )?);
    Ok((pack, payload_root))
}

fn build_common_runtime(
    manifest: &OnTheFlyExecutionManifest,
) -> RusticolResult<(
    ExecutionRuntime,
    Vec<crate::EagerComplex64>,
    Vec<PreparedParameterProjectionEntry>,
)> {
    let metadata = &manifest.runtime_metadata;
    let model_parameters = metadata
        .runtime_parameters
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
        .collect::<Vec<_>>();
    let model_parameter_runtime_slots = runtime_parameter_slots(&model_parameters)?;
    let model_parameter_values_f64 = model_parameters
        .iter()
        .map(|parameter| parameter.default)
        .collect::<Vec<_>>();
    let model_parameter_name_to_index = model_parameters
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
        .collect::<BTreeMap<_, _>>();
    let defaults = metadata
        .prepared_parameter_defaults
        .iter()
        .map(|[real, imaginary]| crate::EagerComplex64::new(*real, *imaginary))
        .collect::<Vec<_>>();
    let projection = metadata
        .parameter_projection
        .iter()
        .filter_map(|row| row.prepared_parameter_id.map(|prepared| (row, prepared)))
        .map(|(row, prepared)| {
            Ok(PreparedParameterProjectionEntry {
                runtime_slot: usize::try_from(row.runtime_slot).map_err(|_| {
                    RusticolError::artifact("on-the-fly runtime parameter slot exceeds usize")
                })?,
                prepared_slot: usize::try_from(prepared).map_err(|_| {
                    RusticolError::artifact("on-the-fly prepared parameter slot exceeds usize")
                })?,
                component: u8::try_from(row.component).map_err(|_| {
                    RusticolError::artifact("on-the-fly parameter component exceeds u8")
                })?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let normalization = &metadata.normalization;
    if !normalization.couplings_in_stage_evaluators {
        return Err(RusticolError::compatibility(
            "on-the-fly execution requires local vertex couplings in prepared kernel calls",
        ));
    }
    let color_factor = if metadata
        .color_contraction
        .as_ref()
        .is_some_and(|contraction| contraction.includes_color_factor)
    {
        1.0
    } else {
        normalization.color_factor
    };
    let normalization_factor = color_factor * normalization.global_coupling_factor
        / (normalization.average_factor * normalization.identical_factor);
    if !normalization_factor.is_finite() {
        return Err(RusticolError::integrity(
            "on-the-fly runtime normalization is not finite",
        ));
    }
    let particle_masses = metadata
        .particle_masses
        .iter()
        .map(|row| (row.outgoing_pdg, row.mass))
        .collect::<BTreeMap<_, _>>();
    let external_count = metadata.external_legs.len();
    let common = ExecutionRuntime {
        process: manifest.process.clone(),
        key: manifest.key.clone(),
        color_accuracy: manifest.color_accuracy.clone(),
        external_pdg_order: manifest.external_pdg_order.clone(),
        external_count,
        parameter_count: model_parameters.len(),
        value_parameter_count: 0,
        momentum_parameter_count: 0,
        current_count: 0,
        source_count: external_count,
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
        compiled_direct_runtime: None,
        compiled_direct_color_schedules: BTreeMap::new(),
        compiled_direct_helicity_schedules: BTreeMap::new(),
        helicity_sum_runtime: None,
        helicity_selector_runtimes: Vec::new(),
        helicity_selector_runtime_schedule_modes: Vec::new(),
        helicity_selector_lane_by_domain: BTreeMap::new(),
        color_selector_runtimes: BTreeMap::new(),
        runtime_unavailable_message: None,
        sources: Vec::new(),
        momentum_slots: Vec::new(),
        external_is_initial: metadata
            .external_legs
            .iter()
            .map(|leg| leg.is_initial)
            .collect(),
        particle_masses,
        particle_mass_parameter_names: BTreeMap::new(),
        normalization_factor,
        normalization_color_factor: color_factor,
        normalization_average_factor: normalization.average_factor,
        normalization_identical_factor: normalization.identical_factor,
        normalization_qcd_coupling_power: normalization.qcd_coupling_power.unwrap_or(0) as usize,
        normalization_electroweak_coupling_power: normalization
            .electroweak_coupling_power
            .unwrap_or(0) as usize,
        model_parameters,
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
    };
    Ok((common, defaults, projection))
}

#[cfg(test)]
mod tests {
    use super::{
        LoadedOnTheFlyColorContractionV1, clamp_query_construction_threads,
        symmetric_group_workspace_for_loaded_color, validate_recurrence_companion_process_digest,
    };
    use crate::recurrence::RecurrenceColorContraction;

    #[test]
    fn query_construction_threads_are_capped_by_positive_host_availability() {
        assert_eq!(clamp_query_construction_threads(8, 3), 3);
        assert_eq!(clamp_query_construction_threads(2, 8), 2);
        assert_eq!(clamp_query_construction_threads(4, 0), 1);
    }

    #[test]
    fn cross_process_companion_seed_swap_is_rejected_by_primary_binding_digest() {
        let primary = "10".repeat(32);
        let swapped = "20".repeat(32);
        validate_recurrence_companion_process_digest(&primary, &primary, &primary).unwrap();

        for (companion, decoded) in [
            (&swapped, &swapped),
            (&primary, &swapped),
            (&swapped, &primary),
        ] {
            let error = validate_recurrence_companion_process_digest(&primary, companion, decoded)
                .unwrap_err();
            assert!(error.to_string().contains("different primary process"));
        }
    }

    #[test]
    fn symmetric_group_workspace_is_installed_at_the_authenticated_tile_capacity() {
        let loaded = LoadedOnTheFlyColorContractionV1 {
            plan: RecurrenceColorContraction::symmetric_group_s3_for_runtime_test(
                (0..13).collect(),
                13,
            ),
            destination_by_owner_ordinal: (0..13).collect::<Vec<_>>().into_boxed_slice(),
            point_tile_size: 5,
        };
        let workspace = symmetric_group_workspace_for_loaded_color(Some(&loaded))
            .unwrap()
            .expect("symmetric-group workspace");
        assert_eq!(workspace.lane_capacity(), 5);
    }
}
