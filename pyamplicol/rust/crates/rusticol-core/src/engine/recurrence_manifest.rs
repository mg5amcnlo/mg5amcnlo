// SPDX-License-Identifier: 0BSD

//! Bounded parsing and semantic validation for recurrence execution manifests.

use crate::engine::{
    RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
    SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
};
use crate::recurrence::direct_backend::RECURRENCE_DIRECT_BACKEND_ABI;
use crate::recurrence::template::RECURRENCE_TEMPLATE_INPUT_ABI;
use crate::recurrence::{
    CLOSURE_TARGET_DOMAIN_PROOF_KIND, RECURRENCE_BUILDER_INPUT_ABI,
    RECURRENCE_COLOR_CONTRACTION_CODEC_ABI, RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER,
    RECURRENCE_CONTRACTED_COLOR_CAPABILITY, RECURRENCE_DIRECT_SCHEDULE_MEMBER,
    RECURRENCE_DIRECT_TEMPLATE_ABI, RECURRENCE_HELICITY_DISPATCH_ABI,
    RECURRENCE_LC_COLOR_CAPABILITY, RECURRENCE_PLAN_ABI, RECURRENCE_RUNTIME_CAPABILITY,
    RECURRENCE_RUNTIME_KIND, RECURRENCE_RUNTIME_LAYOUT_ABI, relation_certificate_algorithm,
};
use crate::{ArtifactProcess, PROCESS_ARTIFACT_SCHEMA_VERSION, RusticolError, RusticolResult};
use bincode::Decode;
#[cfg(any(feature = "python-generation-bridge", test))]
use bincode::Encode;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub(super) const RECURRENCE_RUNTIME_STORAGE_ABI: &str = "pacbin-v1";
pub(super) const RECURRENCE_RUNTIME_CONTAINER_KIND: &str =
    "pyamplicol-recurrence-runtime-container";
pub(super) const RECURRENCE_RUNTIME_CONTAINER_SCHEMA: u16 = 1;
pub(super) const RECURRENCE_PROCESS_BINDING_ABI: &str = "pyamplicol-recurrence-process-binding-v4";
pub(super) const RECURRENCE_PROCESS_BINDING_PATH: &str = "recurrence-binding.bin";
pub(super) const RECURRENCE_HELICITY_SELECTOR_BINDING_PATH: &str = "helicity-selector-binding.bin";
pub(super) const RECURRENCE_HELICITY_DISPATCH_PATH: &str = "recurrence-helicity-dispatch-v1.bin";
pub(super) const RECURRENCE_HELICITY_SELECTOR_COMPANION_KIND: &str =
    "pyamplicol-recurrence-helicity-selector-companion-v2";
pub(super) const RECURRENCE_PRIMARY_LOCAL_COLOR_VIEW: &str = "primary-local-color-view-v1";
pub(super) const RECURRENCE_COLOR_CONTRACTION_PATH: &str = "recurrence-color.bin";
pub(super) const RECURRENCE_KERNEL_PACK_MANIFEST_PATH: &str = "model/eager-kernel-pack.json";
pub(super) const RECURRENCE_KERNEL_PAYLOAD_ROOT: &str = "model/eager-kernels";

const MAX_RELATION_DISCOVERY_SAMPLES: usize = 16;

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(crate) struct RecurrenceExecutionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) process_digest: Option<String>,
    pub(super) builder_input_abi: String,
    pub(super) recurrence_plan_abi: String,
    pub(super) runtime_layout_abi: String,
    pub(super) direct_template_abi: String,
    pub(super) direct_backend_abi: String,
    pub(super) compiled_model_digest: String,
    pub(super) recurrence_template_catalog_digest: String,
    pub(super) prepared_kernel_pack_digest: String,
    pub(super) direct_template_catalog_digest: String,
    pub(super) kernel_pack: RecurrenceKernelPackReference,
    pub(super) runtime_options: RecurrenceRuntimeOptions,
    pub(super) runtime_metadata: RecurrenceRuntimeMetadata,
    pub(super) helicity_selector_companion: Option<RecurrenceHelicitySelectorCompanionManifest>,
    pub(super) plan: RecurrencePlanSummary,
    pub(super) recurrence_summary: RecurrenceSummary,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceKernelPackReference {
    pub(super) manifest_path: String,
    pub(super) payload_root: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeOptions {
    pub(super) point_tile_size: u32,
    pub(super) workspace_mib: u32,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrencePlanSummary {
    pub(super) kind: String,
    pub(super) builder_input_abi: String,
    pub(super) recurrence_plan_abi: String,
    pub(super) runtime_layout_abi: String,
    pub(super) direct_template_abi: String,
    pub(super) direct_backend_abi: String,
    pub(super) builder_input_sha256: String,
    pub(super) prepared_kernel_pack_digest: String,
    pub(super) direct_template_catalog_digest: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) runtime_schedule: RecurrenceRuntimeContainer,
    #[serde(default)]
    pub(super) helicity_dispatch: Option<RecurrenceHelicityDispatchReference>,
    pub(super) process_binding: RecurrenceProcessBinding,
    pub(super) inspection_summary: RecurrenceInspectionSummary,
}

/// One pre-lowered physical-colour schedule plus an exact runtime-helicity
/// dispatch.  Physics and color storage remain owned by the enclosing
/// recurrence process.
#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceHelicitySelectorCompanionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    pub(super) process_digest: String,
    pub(super) plan: RecurrencePlanSummary,
    pub(super) color_contraction: RecurrenceHelicitySelectorColorView,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceHelicityDispatchReference {
    pub(super) abi: String,
    pub(super) path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
    pub(super) base_runtime_layout_digest: String,
    pub(super) resolved_helicity_count: u64,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceHelicitySelectorColorView {
    pub(super) source: String,
    pub(super) view: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceProcessBinding {
    pub(super) abi: String,
    pub(super) process_id: String,
    pub(super) schedule_digest: String,
    pub(super) native_schedule_semantic_digest: String,
    pub(super) process_digest: String,
    pub(super) process_semantic_digest: String,
    pub(super) process_support_words: Vec<u64>,
    pub(super) remap: RecurrenceProcessRemap,
    pub(super) path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
}

#[derive(Clone, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceProcessRemap {
    pub(super) bijection_digest: String,
    pub(super) source_slots: Vec<u32>,
    pub(super) source_momentum_signs: Vec<i32>,
    pub(super) source_helicity_signs: Vec<i32>,
    pub(super) source_state_offsets: Vec<u32>,
    pub(super) source_state_indices: Vec<u32>,
    pub(super) public_flow_ids: Vec<u32>,
    pub(super) physical_sector_ids: Vec<u32>,
    pub(super) state_templates: RecurrenceSparseBijection,
    pub(super) source_templates: RecurrenceSparseBijection,
    pub(super) direct_executors: RecurrenceSparseBijection,
    pub(super) parameter_slots: RecurrenceSparseBijection,
}

#[derive(Clone, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceSparseBijection {
    pub(super) count: u32,
    pub(super) changes: Vec<[u32; 2]>,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeContainer {
    pub(super) kind: String,
    pub(super) schema_version: u16,
    pub(super) storage_abi: String,
    pub(super) path: String,
    pub(super) plan_member_path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
    pub(super) member_count: u64,
    pub(super) unpacked_size_bytes: u64,
    pub(super) index_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrenceLcFlowLayout {
    TopologyReplay,
    AllFlowUnion,
    ContractedColorUnion,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrenceCacheFootprintPolicy {
    SelectorActiveMaxV1,
    PersistedArenaV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrencePeakContributionCountSemantics {
    ResidentPendingContributionsV1,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceInspectionSummary {
    pub(super) execution_mode: String,
    pub(super) recurrence_plan_abi: String,
    pub(super) runtime_layout_abi: String,
    pub(super) direct_template_abi: String,
    pub(super) process_id: String,
    pub(super) lc_flow_layout: RecurrenceLcFlowLayout,
    pub(super) parameter_count: u64,
    pub(super) sector_count: u64,
    pub(super) prepared_kernel_count: u64,
    pub(super) direct_executor_count: u64,
    pub(super) semantic_digest: String,
    pub(super) schedule_digest: String,
    pub(super) runtime_layout_digest: String,
    pub(super) schedule: RecurrenceScheduleSummary,
    #[serde(default)]
    pub(super) selector_work_certificate: Option<RecurrenceSelectorWorkCertificate>,
    pub(super) construction: RecurrenceConstructionSummary,
    pub(super) direct_arena: RecurrenceDirectArenaSummary,
    pub(super) runtime_container_member: RecurrenceRuntimeContainerMember,
    #[serde(default)]
    pub(super) color_projection_certificate: Option<RecurrenceColorProjectionCertificate>,
    #[serde(default)]
    pub(super) relation_discovery: Option<RecurrenceRelationDiscoverySummary>,
    #[serde(default)]
    pub(super) structural_zero_physical_sector_certificate:
        Option<RecurrenceStructuralZeroPhysicalSectorCertificate>,
    pub(super) generation_timings_seconds: RecurrenceGenerationTimings,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceStructuralZeroPhysicalSectorCertificate {
    pub(super) proof_kind: String,
    pub(super) semantic_digest: String,
    pub(super) sector_count: u64,
    pub(super) resolved_helicity_count: u64,
    pub(super) certified_structural_zero_sector_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceScheduleSummary {
    pub(super) current_count: u64,
    pub(super) source_row_count: u64,
    pub(super) contribution_count: u64,
    pub(super) finalization_count: u64,
    pub(super) amplitude_destination_count: u64,
    pub(super) closure_term_count: u64,
    pub(super) replay_target_count: u64,
    pub(super) retained_helicity_count: u64,
    pub(super) resolved_helicity_count: u64,
    pub(super) exact_factor_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceSelectorPersistedUnion {
    pub(super) current_count: u64,
    pub(super) semantic_component_count: u64,
    pub(super) source_row_count: u64,
    pub(super) contribution_count: u64,
    pub(super) finalization_count: u64,
    pub(super) closure_count: u64,
    pub(super) row_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceSelectorRepresentative {
    pub(super) representative_sector_id: u64,
    pub(super) public_flow_count: u64,
    pub(super) current_count: u64,
    pub(super) semantic_component_count: u64,
    pub(super) source_row_count: u64,
    pub(super) contribution_count: u64,
    pub(super) finalization_count: u64,
    pub(super) closure_count: u64,
    pub(super) amplitude_destination_count: u64,
    pub(super) row_count: u64,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceSelectorWorkCertificate {
    pub(super) schema_version: u16,
    pub(super) binding: String,
    pub(super) persisted_union: RecurrenceSelectorPersistedUnion,
    pub(super) representatives: Vec<RecurrenceSelectorRepresentative>,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceConstructionSummary {
    pub(super) peak_current_count: u64,
    pub(super) peak_contribution_count: u64,
    // Typed deserialization authenticates the marker; publication audits
    // decide whether an absent legacy marker is admissible evidence.
    #[allow(dead_code)]
    #[serde(default)]
    pub(super) peak_contribution_count_semantics: Option<RecurrencePeakContributionCountSemantics>,
    pub(super) peak_dynamic_color_state_count: u64,
    pub(super) color_target_prune_count: u64,
    pub(super) candidate_parent_pair_count: u64,
    pub(super) peak_to_final_current_ratio: f64,
    pub(super) peak_to_final_contribution_ratio: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceDirectArenaSummary {
    pub(super) semantic_component_count: u64,
    pub(super) current_arena_components: u64,
    pub(super) arena_component_reuse_count: u64,
    pub(super) momentum_form_count: u64,
    pub(super) selector_domain_count: u64,
    #[serde(default)]
    pub(super) cache_footprint_policy: Option<RecurrenceCacheFootprintPolicy>,
    pub(super) row_group_count: u64,
    pub(super) packed_input_bytes: u64,
    pub(super) packed_output_bytes: u64,
    pub(super) scatter_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeContainerMember {
    pub(super) path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
    pub(super) container_size_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceColorProjectionCertificate {
    pub(super) path: String,
    pub(super) schema_version: u16,
    pub(super) proof_kind: String,
    pub(super) publishable: bool,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrenceRelationDiscoveryMode {
    Diagnostic,
    CertifiedReuse,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRelationDiscoverySummary {
    pub(super) schema_version: u16,
    pub(super) requested_mode: RecurrenceRelationDiscoveryMode,
    pub(super) state: String,
    pub(super) scope: RecurrenceRelationDiscoveryScope,
    pub(super) probe: RecurrenceRelationDiscoveryProbe,
    pub(super) certificate_replay: RecurrenceRelationCertificateReplay,
    pub(super) numerical_candidate_count: u64,
    pub(super) uncertified_candidate_count: u64,
    pub(super) rejected_hypothesis_count: u64,
    pub(super) exact_certified_relation_count: u64,
    pub(super) applied_relation_count: u64,
    pub(super) interaction_evaluation_count_before: u64,
    pub(super) interaction_evaluation_count_after: u64,
    pub(super) interaction_evaluation_savings: u64,
    pub(super) current_count_before: u64,
    pub(super) current_count_after: u64,
    pub(super) contribution_count_before: u64,
    pub(super) contribution_count_after: u64,
    pub(super) scale_copy_row_count: u64,
    pub(super) certificates: Vec<RecurrenceRelationCertificateSample>,
    pub(super) certificate_count: u64,
    pub(super) certificates_truncated: bool,
    pub(super) rejected_candidates: Vec<RecurrenceRelationRejectedCandidate>,
    #[serde(default)]
    pub(super) rejected_candidate_diagnostics: Option<RecurrenceRelationRejectedDiagnostics>,
    pub(super) follow_up_boundary: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRelationRejectedDiagnostics {
    pub(super) total_rejected_hypothesis_count: u64,
    pub(super) retained_count: u64,
    pub(super) truncated: bool,
    pub(super) truncation_policy: String,
    pub(super) retained_sha256: String,
    pub(super) full_census_sha256: String,
    pub(super) full_rejection_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRelationDiscoveryScope {
    pub(super) execution_mode: String,
    pub(super) color_accuracy: String,
    pub(super) representation: String,
    pub(super) lc_flow_layout: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRelationDiscoveryProbe {
    pub(super) status: String,
    pub(super) precision_digits: u32,
    pub(super) probe_count: u32,
    pub(super) verification_probe_count: u32,
    pub(super) effective_projection_count: u32,
    pub(super) seed: u64,
    pub(super) deterministic: bool,
    pub(super) candidate_only: bool,
    pub(super) tested_hypothesis_count: u64,
    pub(super) verification_rejected_count: u64,
    pub(super) runtime_parameter_schema_sha256: Option<String>,
    pub(super) candidate_observation_batch_sha256: Option<String>,
    pub(super) verification_observation_batch_sha256: Option<String>,
    pub(super) decision_sha256: Option<String>,
    pub(super) rejection_decision_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRelationCertificateReplay {
    pub(super) algorithm: String,
    pub(super) status: String,
    pub(super) certificate_set_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRelationCertificateSample {
    pub(super) algorithm: String,
    pub(super) current_id: u64,
    pub(super) representative_id: u64,
    pub(super) factor_exact_rational: RecurrenceRelationExactFactor,
    pub(super) current_term_vector_sha256: String,
    pub(super) representative_term_vector_sha256: String,
    pub(super) proof_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRelationExactFactor {
    pub(super) real_numerator: String,
    pub(super) real_denominator: String,
    pub(super) imag_numerator: String,
    pub(super) imag_denominator: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRelationRejectedCandidate {
    pub(super) reason: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenerationTimings {
    pub(super) python_extraction: f64,
    pub(super) catalog_authentication: f64,
    pub(super) semantic_construction: f64,
    pub(super) direct_lowering: f64,
    pub(super) serialization: f64,
    pub(super) native_total: f64,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceSummary {
    pub(super) lc_flow_layout: RecurrenceLcFlowLayout,
    pub(super) builder_input_abi: String,
    pub(super) template_input_abi: String,
    pub(super) current_count: u64,
    pub(super) contribution_count: u64,
    pub(super) closure_term_count: u64,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeMetadata {
    pub(super) public_color_flows: Vec<RecurrencePublicColorFlow>,
    pub(super) runtime_parameters: Vec<RecurrenceRuntimeParameter>,
    pub(super) prepared_parameter_defaults: Vec<[f64; 2]>,
    pub(super) parameter_projection: Vec<RecurrenceParameterProjection>,
    pub(super) source_templates: Vec<RecurrenceSourceTemplate>,
    pub(super) external_legs: Vec<RecurrenceExternalLeg>,
    pub(super) particle_masses: Vec<RecurrenceParticleMass>,
    pub(super) normalization: RecurrenceNormalization,
    pub(super) color_contraction: Option<RecurrenceColorContractionReference>,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceColorContractionReference {
    pub(super) abi: String,
    pub(super) path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
    pub(super) color_accuracy: String,
    pub(super) storage: String,
    pub(super) includes_color_factor: bool,
    pub(super) group_count: u64,
    pub(super) sector_count: u64,
    pub(super) active_sector_count: u64,
    pub(super) component_count: u64,
    pub(super) destination_count: u64,
    #[serde(default, deserialize_with = "deserialize_optional_u64_if_present")]
    pub(super) materialized_destination_count: Option<u64>,
    pub(super) entry_count: u64,
    pub(super) logical_entry_count: u64,
    pub(super) semantic_digest: String,
    #[serde(default)]
    pub(super) factorization: Option<RecurrenceColorFactorizationReference>,
    #[serde(default)]
    pub(super) fft_provenance: Option<RecurrenceColorFftProvenance>,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceColorFactorizationReference {
    pub(super) kind: String,
    pub(super) rank: u32,
    pub(super) coset_count: u64,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceColorFftProvenance {
    pub(super) method: String,
    pub(super) degree: u32,
    pub(super) channel_count: u64,
    pub(super) covered_local_group_count: u64,
    pub(super) residual_group_count: u64,
    pub(super) residual_entry_count: u64,
    pub(super) raw_kernel_bytes: u64,
    pub(super) transformed_kernel_bytes: u64,
    pub(super) capability: String,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrencePublicColorFlow {
    pub(super) public_id: String,
    pub(super) construction_sector_id: u32,
    pub(super) target_sector_id: u32,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceRuntimeParameter {
    pub(super) name: String,
    pub(super) kind: String,
    pub(super) parameter_index: u32,
    pub(super) default: f64,
    #[serde(default)]
    pub(super) runtime_name: Option<String>,
    #[serde(default)]
    pub(super) complex_component: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceParameterProjection {
    pub(super) runtime_slot: u32,
    pub(super) runtime_name: String,
    pub(super) parameter_template_id: u32,
    #[serde(deserialize_with = "deserialize_required_nullable_u32")]
    pub(super) prepared_parameter_id: Option<u32>,
    pub(super) component: u32,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceSourceTemplate {
    pub(super) source_template_id: u32,
    #[serde(rename = "current_state_template_id")]
    pub(super) _current_state_template_id: u32,
    pub(super) dimension: u64,
    pub(super) helicity: i32,
    pub(super) chirality: i32,
    pub(super) spin_state: i32,
    pub(super) source_ir: RecurrenceGenericSourceIr,
    pub(super) crossing: RecurrenceGenericCrossingIr,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceExternalLeg {
    pub(super) source_slot: u32,
    pub(super) public_label: u32,
    pub(super) physical_pdg: i32,
    pub(super) outgoing_pdg: i32,
    pub(super) is_initial: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceParticleMass {
    pub(super) outgoing_pdg: i32,
    pub(super) mass: f64,
}

#[derive(Clone, Debug, Deserialize, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceNormalization {
    pub(super) color_accuracy: String,
    pub(super) color_factor: f64,
    pub(super) average_factor: f64,
    pub(super) identical_factor: f64,
    pub(super) global_coupling_factor: f64,
    #[serde(default, deserialize_with = "deserialize_optional_u64_if_present")]
    pub(super) qcd_coupling_power: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_optional_u64_if_present")]
    pub(super) electroweak_coupling_power: Option<u64>,
    pub(super) couplings_in_stage_evaluators: bool,
    pub(super) coupling_policy: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrenceSourceOrientation {
    Particle,
    Antiparticle,
    SelfConjugate,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "lowercase")]
pub(super) enum RecurrenceParticleStatistics {
    Boson,
    Fermion,
    Ghost,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "lowercase")]
pub(super) enum RecurrenceWavefunctionFamily {
    Scalar,
    Fermion,
    Vector,
    Spin2,
    Ghost,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(rename_all = "kebab-case")]
pub(super) enum RecurrenceMomentumTransform {
    Identity,
    NegateFourMomentum,
}

#[derive(Clone, Debug, Deserialize, Decode, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(untagged)]
pub(super) enum RecurrenceSourceSpinState {
    Scalar(i32),
    Components(Vec<i32>),
}

#[derive(Clone, Debug, Deserialize, Decode, Eq, Ord, PartialEq, PartialOrd)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenericSourceStateIr {
    pub(super) helicity: i32,
    pub(super) chirality: i32,
    pub(super) spin_state: RecurrenceSourceSpinState,
}

#[derive(Clone, Debug, Deserialize, Decode, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenericCrossingIr {
    pub(super) momentum_transform: RecurrenceMomentumTransform,
    pub(super) helicity_factor: i32,
    pub(super) chirality_factor: i32,
    pub(super) spin_state_factor: i32,
    #[serde(deserialize_with = "deserialize_recurrence_phase")]
    pub(super) phase: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, Decode, Eq, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenericParticleIdentityIr {
    pub(super) canonical_id: String,
    pub(super) species_id: String,
    pub(super) anti_canonical_id: String,
    pub(super) display_name: String,
    pub(super) anti_display_name: String,
    pub(super) pdg_label: i32,
    pub(super) anti_pdg_label: i32,
    pub(super) orientation: RecurrenceSourceOrientation,
    pub(super) self_conjugate: bool,
}

#[derive(Clone, Debug, Deserialize, Decode, PartialEq)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[serde(deny_unknown_fields)]
pub(super) struct RecurrenceGenericSourceIr {
    pub(super) identity: RecurrenceGenericParticleIdentityIr,
    pub(super) statistics: RecurrenceParticleStatistics,
    pub(super) wavefunction_family: RecurrenceWavefunctionFamily,
    pub(super) component_dimension: u64,
    pub(super) states: Vec<RecurrenceGenericSourceStateIr>,
    pub(super) crossing: RecurrenceGenericCrossingIr,
    pub(super) basis: String,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    pub(super) mass_parameter: Option<String>,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    pub(super) width_parameter: Option<String>,
}

impl RecurrenceExecutionManifest {
    fn validate(&self, outer: &ArtifactProcess) -> RusticolResult<()> {
        if self.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
            || self.kind != RECURRENCE_RUNTIME_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence execution kind {:?} schema {}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.process != outer.expression
            || self.key != outer.id
            || self.color_accuracy != outer.color_accuracy
            || self.external_pdg_order != outer.external_pdgs
        {
            return Err(RusticolError::integrity(format!(
                "recurrence execution manifest does not match outer process {:?}",
                outer.id
            )));
        }
        let uses_symmetric_group_fft = self
            .runtime_metadata
            .color_contraction
            .as_ref()
            .and_then(|reference| reference.factorization.as_ref())
            .is_some_and(|factorization| factorization.kind == "symmetric-group-fourier");
        let has_helicity_selector_companion = self.helicity_selector_companion.is_some();
        validate_capabilities(
            &self.required_runtime_capabilities,
            &self.color_accuracy,
            uses_symmetric_group_fft,
            has_helicity_selector_companion,
            "recurrence execution manifest",
        )?;
        validate_capabilities(
            &outer.required_runtime_capabilities,
            &self.color_accuracy,
            uses_symmetric_group_fft,
            has_helicity_selector_companion,
            "outer recurrence process",
        )?;
        validate_direct_contract(
            &self.builder_input_abi,
            &self.recurrence_plan_abi,
            &self.runtime_layout_abi,
            &self.direct_template_abi,
            &self.direct_backend_abi,
            "recurrence execution manifest",
        )?;
        parse_sha256(&self.compiled_model_digest, "recurrence compiled model")?;
        parse_sha256(
            &self.recurrence_template_catalog_digest,
            "recurrence template catalog",
        )?;
        parse_sha256(
            &self.prepared_kernel_pack_digest,
            "recurrence prepared-kernel pack",
        )?;
        parse_sha256(
            &self.direct_template_catalog_digest,
            "recurrence direct-template catalog",
        )?;
        if self.kernel_pack.manifest_path != RECURRENCE_KERNEL_PACK_MANIFEST_PATH
            || self.kernel_pack.payload_root != RECURRENCE_KERNEL_PAYLOAD_ROOT
        {
            return Err(RusticolError::security(
                "recurrence kernel-pack paths are not canonical",
            ));
        }
        bounded_len("external PDG order", self.external_pdg_order.len())?;
        if self.external_pdg_order.is_empty() {
            return Err(RusticolError::artifact(
                "recurrence execution manifest has no external particles",
            ));
        }
        self.runtime_options.validate()?;
        self.plan.validate(RecurrencePlanValidation {
            process_key: &self.key,
            prepared_kernel_pack_digest: &self.prepared_kernel_pack_digest,
            direct_template_catalog_digest: &self.direct_template_catalog_digest,
            color_accuracy: &self.color_accuracy,
            uses_symmetric_group_fft,
            has_helicity_selector_companion,
            role: RecurrencePlanRole::Primary,
        })?;
        self.recurrence_summary.validate()?;
        self.runtime_metadata.validate(
            &self.external_pdg_order,
            &self.color_accuracy,
            &self.plan.inspection_summary,
        )?;
        if has_helicity_selector_companion != self.process_digest.is_some() {
            return Err(RusticolError::integrity(
                "recurrence process digest must appear exactly with its helicity-selector companion",
            ));
        }
        if let Some(process_digest) = &self.process_digest {
            parse_sha256(process_digest, "recurrence process")?;
            if process_digest != &self.plan.process_binding.process_digest {
                return Err(RusticolError::integrity(
                    "recurrence companion process digest disagrees with the primary process binding",
                ));
            }
        }
        if let Some(companion) = &self.helicity_selector_companion {
            companion.validate(self, uses_symmetric_group_fft)?;
        }

        let inspection = &self.plan.inspection_summary;
        let remap = &self.plan.process_binding.remap;
        if (inspection.lc_flow_layout != RecurrenceLcFlowLayout::ContractedColorUnion
            && remap.public_flow_ids.len() != self.runtime_metadata.public_color_flows.len())
            || remap.physical_sector_ids.len() as u64 != inspection.sector_count
        {
            return Err(RusticolError::integrity(
                "recurrence process binding domains disagree with runtime metadata",
            ));
        }
        let schedule = inspection.schedule;
        if inspection.lc_flow_layout != self.recurrence_summary.lc_flow_layout
            || schedule.current_count != self.recurrence_summary.current_count
            || schedule.contribution_count != self.recurrence_summary.contribution_count
            || schedule.closure_term_count != self.recurrence_summary.closure_term_count
        {
            return Err(RusticolError::integrity(
                "recurrence inspection counts disagree with the recurrence summary",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecurrencePlanRole {
    Primary,
    HelicitySelector,
}

struct RecurrencePlanValidation<'a> {
    process_key: &'a str,
    prepared_kernel_pack_digest: &'a str,
    direct_template_catalog_digest: &'a str,
    color_accuracy: &'a str,
    uses_symmetric_group_fft: bool,
    has_helicity_selector_companion: bool,
    role: RecurrencePlanRole,
}

impl RecurrenceRuntimeOptions {
    fn validate(self) -> RusticolResult<()> {
        if self.point_tile_size == 0 {
            return Err(RusticolError::artifact(
                "recurrence point_tile_size must be positive",
            ));
        }
        if self.workspace_mib == 0 {
            return Err(RusticolError::artifact(
                "recurrence workspace_mib must be positive",
            ));
        }
        Ok(())
    }
}

impl RecurrencePlanSummary {
    fn validate(&self, validation: RecurrencePlanValidation<'_>) -> RusticolResult<()> {
        let RecurrencePlanValidation {
            process_key,
            prepared_kernel_pack_digest,
            direct_template_catalog_digest,
            color_accuracy,
            uses_symmetric_group_fft,
            has_helicity_selector_companion,
            role,
        } = validation;
        if self.kind != RECURRENCE_RUNTIME_KIND {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence plan kind {:?}",
                self.kind
            )));
        }
        validate_direct_contract(
            &self.builder_input_abi,
            &self.recurrence_plan_abi,
            &self.runtime_layout_abi,
            &self.direct_template_abi,
            &self.direct_backend_abi,
            "nested recurrence plan",
        )?;
        parse_sha256(&self.builder_input_sha256, "recurrence builder input")?;
        parse_sha256(
            &self.prepared_kernel_pack_digest,
            "nested recurrence prepared-kernel pack",
        )?;
        parse_sha256(
            &self.direct_template_catalog_digest,
            "nested recurrence direct-template catalog",
        )?;
        if self.prepared_kernel_pack_digest != prepared_kernel_pack_digest {
            return Err(RusticolError::integrity(
                "recurrence prepared-kernel pack digest differs between execution and plan metadata",
            ));
        }
        if self.direct_template_catalog_digest != direct_template_catalog_digest {
            return Err(RusticolError::integrity(
                "recurrence direct-template catalog digest differs between execution and plan metadata",
            ));
        }
        validate_capabilities(
            &self.required_runtime_capabilities,
            color_accuracy,
            uses_symmetric_group_fft,
            has_helicity_selector_companion,
            "nested recurrence plan",
        )?;
        if self.helicity_dispatch.is_some() != (role == RecurrencePlanRole::HelicitySelector) {
            return Err(RusticolError::integrity(match role {
                RecurrencePlanRole::Primary => {
                    "primary recurrence plan must not carry a helicity dispatch"
                }
                RecurrencePlanRole::HelicitySelector => {
                    "recurrence helicity-selector plan requires a helicity dispatch"
                }
            }));
        }
        self.runtime_schedule.validate()?;
        let binding_path = match role {
            RecurrencePlanRole::Primary => RECURRENCE_PROCESS_BINDING_PATH,
            RecurrencePlanRole::HelicitySelector => RECURRENCE_HELICITY_SELECTOR_BINDING_PATH,
        };
        self.process_binding.validate(process_key, binding_path)?;
        self.inspection_summary
            .validate(process_key, color_accuracy)?;
        let schedule_identity_matches = match role {
            RecurrencePlanRole::Primary => {
                self.process_binding.schedule_digest == self.inspection_summary.schedule_digest
            }
            RecurrencePlanRole::HelicitySelector => {
                self.process_binding.native_schedule_semantic_digest
                    == self.inspection_summary.schedule_digest
            }
        };
        if !schedule_identity_matches
            || self.process_binding.process_semantic_digest
                != self.inspection_summary.semantic_digest
            || self.process_binding.process_semantic_digest != self.builder_input_sha256
        {
            return Err(RusticolError::integrity(
                "recurrence schedule/process binding disagrees with inspection metadata",
            ));
        }
        let expected_schedule_path = format!(
            "recurrence/schedules/{}/recurrence-runtime.pacbin",
            self.process_binding.schedule_digest
        );
        if self.runtime_schedule.path != expected_schedule_path {
            return Err(RusticolError::integrity(
                "recurrence root schedule path disagrees with its binding digest",
            ));
        }
        if let Some(dispatch) = self.helicity_dispatch.as_ref() {
            dispatch.validate(&self.process_binding, &self.inspection_summary)?;
        }
        let member = &self.inspection_summary.runtime_container_member;
        let certificate = self
            .inspection_summary
            .color_projection_certificate
            .as_ref();
        let expected_member_count = if certificate.is_some() { 2 } else { 1 };
        let expected_unpacked_size_bytes =
            certificate.map_or(Ok(member.size_bytes), |certificate| {
                member
                    .size_bytes
                    .checked_add(certificate.size_bytes)
                    .ok_or_else(|| {
                        RusticolError::artifact("recurrence runtime member sizes overflow u64")
                    })
            })?;
        if member.path != self.runtime_schedule.plan_member_path
            || member.container_size_bytes != self.runtime_schedule.size_bytes
            || self.runtime_schedule.member_count != expected_member_count
            || self.runtime_schedule.unpacked_size_bytes != expected_unpacked_size_bytes
        {
            return Err(RusticolError::integrity(
                "Direct-Arena v2 plan member metadata disagrees with its recurrence runtime container",
            ));
        }
        Ok(())
    }
}

impl RecurrenceHelicitySelectorCompanionManifest {
    fn validate(
        &self,
        primary: &RecurrenceExecutionManifest,
        uses_symmetric_group_fft: bool,
    ) -> RusticolResult<()> {
        if self.schema_version != 2 || self.kind != RECURRENCE_HELICITY_SELECTOR_COMPANION_KIND {
            return Err(RusticolError::compatibility(
                "unsupported persisted recurrence helicity-selector companion; regenerate the artifact",
            ));
        }
        if !matches!(primary.color_accuracy.as_str(), "nlc" | "full") {
            return Err(RusticolError::integrity(
                "only contracted recurrence may carry a persisted helicity-selector companion",
            ));
        }
        parse_sha256(&self.process_digest, "recurrence helicity-selector process")?;
        if primary.process_digest.as_deref() != Some(self.process_digest.as_str())
            || primary.plan.process_binding.process_digest != self.process_digest
            || self.plan.process_binding.process_digest != self.process_digest
        {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector companion disagrees with the primary process digest",
            ));
        }

        // The v4 binding payload authenticates the compiled-model and
        // recurrence-template identities.  At the bounded manifest boundary,
        // the shared process digest plus the two explicitly repeated pack
        // identities prevent a selector plan from crossing model owners.
        parse_sha256(
            &primary.compiled_model_digest,
            "primary recurrence compiled model",
        )?;
        parse_sha256(
            &primary.recurrence_template_catalog_digest,
            "primary recurrence template catalog",
        )?;
        self.plan.validate(RecurrencePlanValidation {
            process_key: &primary.key,
            prepared_kernel_pack_digest: &primary.prepared_kernel_pack_digest,
            direct_template_catalog_digest: &primary.direct_template_catalog_digest,
            color_accuracy: &primary.color_accuracy,
            uses_symmetric_group_fft,
            has_helicity_selector_companion: true,
            role: RecurrencePlanRole::HelicitySelector,
        })?;

        let primary_inspection = &primary.plan.inspection_summary;
        let inspection = &self.plan.inspection_summary;
        let schedule = inspection.schedule;
        if inspection.lc_flow_layout != RecurrenceLcFlowLayout::AllFlowUnion
            || schedule.replay_target_count != 0
            || schedule.amplitude_destination_count == 0
            || schedule.amplitude_destination_count != inspection.sector_count
            || self.plan.process_binding.remap.physical_sector_ids.len() as u64
                != inspection.sector_count
        {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector plan is not a dense physical all-flow union",
            ));
        }
        if schedule.resolved_helicity_count == 0
            || schedule.resolved_helicity_count != schedule.retained_helicity_count
            || schedule.resolved_helicity_count
                != primary_inspection.schedule.retained_helicity_count
        {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector plan does not retain its full resolved-helicity domain",
            ));
        }
        if inspection.prepared_kernel_count != primary_inspection.prepared_kernel_count
            || inspection.direct_executor_count != primary_inspection.direct_executor_count
            || self.plan.process_binding.process_support_words
                != primary.plan.process_binding.process_support_words
        {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector model identities disagree with the primary recurrence plan",
            ));
        }

        let primary_color = primary
            .runtime_metadata
            .color_contraction
            .as_ref()
            .ok_or_else(|| {
                RusticolError::integrity(
                    "persisted recurrence helicity selection requires a primary color reducer",
                )
            })?;
        self.color_contraction
            .validate(primary_color, &primary.color_accuracy, inspection)?;
        Ok(())
    }
}

impl RecurrenceHelicityDispatchReference {
    fn validate(
        &self,
        binding: &RecurrenceProcessBinding,
        inspection: &RecurrenceInspectionSummary,
    ) -> RusticolResult<()> {
        if self.abi != RECURRENCE_HELICITY_DISPATCH_ABI {
            return Err(unsupported_abi("recurrence helicity-dispatch", &self.abi));
        }
        parse_sha256(&self.sha256, "recurrence helicity-dispatch payload")?;
        parse_sha256(
            &self.base_runtime_layout_digest,
            "recurrence helicity-dispatch base runtime layout",
        )?;
        if self.size_bytes == 0 || u32::try_from(self.resolved_helicity_count).is_err() {
            return Err(RusticolError::artifact(
                "recurrence helicity-dispatch size or resolved-helicity count is outside its wire domain",
            ));
        }
        if self.base_runtime_layout_digest != inspection.runtime_layout_digest
            || self.resolved_helicity_count != inspection.schedule.resolved_helicity_count
            || self.resolved_helicity_count != inspection.schedule.retained_helicity_count
        {
            return Err(RusticolError::integrity(
                "recurrence helicity dispatch disagrees with its base runtime layout or helicity domain",
            ));
        }
        let expected_schedule_digest = recurrence_helicity_selector_schedule_digest(
            &inspection.schedule_digest,
            &self.sha256,
        )?;
        if binding.schedule_digest != expected_schedule_digest {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector root digest does not bind its Direct plan and dispatch",
            ));
        }
        let expected_path = format!(
            "recurrence/schedules/{}/{}",
            binding.schedule_digest, RECURRENCE_HELICITY_DISPATCH_PATH
        );
        if self.path != expected_path {
            return Err(RusticolError::security(
                "recurrence helicity-dispatch path is not canonical",
            ));
        }
        Ok(())
    }
}

impl RecurrenceHelicitySelectorColorView {
    fn validate(
        &self,
        primary: &RecurrenceColorContractionReference,
        color_accuracy: &str,
        selector_inspection: &RecurrenceInspectionSummary,
    ) -> RusticolResult<()> {
        if self.source != "primary" || self.view != RECURRENCE_PRIMARY_LOCAL_COLOR_VIEW {
            return Err(RusticolError::compatibility(
                "unsupported recurrence helicity-selector color view",
            ));
        }
        if primary.color_accuracy != color_accuracy
            || !matches!(color_accuracy, "nlc" | "full")
            || primary.abi != RECURRENCE_COLOR_CONTRACTION_CODEC_ABI
        {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector primary color view has the wrong owner",
            ));
        }
        let factorization_kind = primary
            .factorization
            .as_ref()
            .map(|factorization| factorization.kind.as_str());
        let supported_reducer = matches!(
            (primary.storage.as_str(), factorization_kind),
            ("expanded", None)
                | (
                    "repeated",
                    None | Some("klein-four-walsh") | Some("elementary-abelian-walsh")
                )
                | ("convolution-kernels", Some("symmetric-group-fourier"))
        );
        if !supported_reducer
            || (primary.storage == "expanded" && primary.component_count != 1)
            || primary.component_count == 0
            || !primary.group_count.is_multiple_of(primary.component_count)
            || primary.group_count / primary.component_count != selector_inspection.sector_count
        {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector primary local-color group shape or reducer mode is incompatible",
            ));
        }
        Ok(())
    }
}

fn recurrence_helicity_selector_schedule_digest(
    base_schedule_digest: &str,
    dispatch_sha256: &str,
) -> RusticolResult<String> {
    parse_sha256(
        base_schedule_digest,
        "recurrence helicity-selector base schedule",
    )?;
    parse_sha256(dispatch_sha256, "recurrence helicity-selector dispatch")?;
    let payload = json!({
        "base_schedule_digest": base_schedule_digest,
        "contract": "pyamplicol-recurrence-helicity-selector-schedule-v1",
        "helicity_dispatch": {
            "abi": RECURRENCE_HELICITY_DISPATCH_ABI,
            "sha256": dispatch_sha256,
        },
    });
    let bytes = serde_json::to_vec(&payload).map_err(|error| {
        RusticolError::serialization(format!(
            "could not encode recurrence helicity-selector schedule identity: {error}"
        ))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

impl RecurrenceRuntimeContainer {
    fn validate(&self) -> RusticolResult<()> {
        if self.kind != RECURRENCE_RUNTIME_CONTAINER_KIND {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence runtime container kind {:?}",
                self.kind
            )));
        }
        if self.schema_version != RECURRENCE_RUNTIME_CONTAINER_SCHEMA {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence runtime container schema {}",
                self.schema_version
            )));
        }
        if self.storage_abi != RECURRENCE_RUNTIME_STORAGE_ABI {
            return Err(unsupported_abi(
                "recurrence runtime storage",
                &self.storage_abi,
            ));
        }
        let expected_prefix = "recurrence/schedules/";
        let expected_suffix = "/recurrence-runtime.pacbin";
        if !self.path.starts_with(expected_prefix)
            || !self.path.ends_with(expected_suffix)
            || self.path[expected_prefix.len()..self.path.len() - expected_suffix.len()].len() != 64
            || self.path.contains("..")
        {
            return Err(RusticolError::security(
                "recurrence root schedule path is not canonical",
            ));
        }
        if self.plan_member_path != RECURRENCE_DIRECT_SCHEDULE_MEMBER {
            return Err(RusticolError::security(format!(
                "Direct-Arena v2 schedule member path must be exactly {RECURRENCE_DIRECT_SCHEDULE_MEMBER:?}"
            )));
        }
        if !(1..=2).contains(&self.member_count) {
            return Err(RusticolError::integrity(
                "Direct-Arena v2 recurrence runtime container must declare one or two members",
            ));
        }
        if self.size_bytes == 0 {
            return Err(RusticolError::artifact(
                "recurrence runtime container size must be positive",
            ));
        }
        if self.unpacked_size_bytes == 0 || self.unpacked_size_bytes > self.size_bytes {
            return Err(RusticolError::artifact(
                "recurrence runtime unpacked size is outside canonical bounds",
            ));
        }
        parse_sha256(&self.sha256, "recurrence runtime container")?;
        parse_sha256(&self.index_sha256, "recurrence runtime index")?;
        Ok(())
    }
}

impl RecurrenceProcessBinding {
    pub(super) fn native_schedule_semantic_digest(&self) -> &str {
        &self.native_schedule_semantic_digest
    }

    fn validate(&self, process_key: &str, expected_path: &str) -> RusticolResult<()> {
        if self.abi != RECURRENCE_PROCESS_BINDING_ABI || self.path != expected_path {
            return Err(RusticolError::compatibility(
                "unsupported recurrence process-binding contract",
            ));
        }
        if self.process_id != process_key {
            return Err(RusticolError::integrity(
                "recurrence process binding has the wrong process ID",
            ));
        }
        parse_sha256(&self.schedule_digest, "recurrence root schedule")?;
        parse_sha256(
            &self.native_schedule_semantic_digest,
            "recurrence native schedule semantic digest",
        )?;
        parse_sha256(&self.process_digest, "recurrence process")?;
        parse_sha256(
            &self.process_semantic_digest,
            "recurrence process semantic binding",
        )?;
        parse_sha256(&self.sha256, "recurrence process-binding payload")?;
        self.remap.validate()?;
        if self.process_support_words.is_empty()
            || self.process_support_words.last() == Some(&0)
            || self
                .process_support_words
                .iter()
                .map(|word| word.count_ones())
                .sum::<u32>()
                != 1
            || self.size_bytes == 0
        {
            return Err(RusticolError::integrity(
                "recurrence process binding has invalid support words or size metadata",
            ));
        }
        Ok(())
    }
}

impl RecurrenceProcessRemap {
    fn validate(&self) -> RusticolResult<()> {
        parse_sha256(&self.bijection_digest, "recurrence process bijection")?;
        validate_permutation(&self.source_slots, "recurrence source-slot remap")?;
        if self.source_momentum_signs.len() != self.source_slots.len()
            || self.source_helicity_signs.len() != self.source_slots.len()
            || self
                .source_momentum_signs
                .iter()
                .chain(&self.source_helicity_signs)
                .any(|sign| !matches!(sign, -1 | 1))
        {
            return Err(RusticolError::integrity(
                "recurrence process binding has invalid source signs",
            ));
        }
        validate_ragged_permutations(
            &self.source_state_offsets,
            &self.source_state_indices,
            self.source_slots.len(),
            "recurrence source-state remap",
        )?;
        validate_permutation(&self.public_flow_ids, "recurrence public-flow remap")?;
        validate_permutation(
            &self.physical_sector_ids,
            "recurrence physical-sector remap",
        )?;
        self.state_templates
            .validate("recurrence state-template remap")?;
        self.source_templates
            .validate("recurrence source-template remap")?;
        self.direct_executors
            .validate("recurrence direct-executor remap")?;
        self.parameter_slots
            .validate("recurrence parameter-slot remap")?;
        Ok(())
    }
}

impl RecurrenceSparseBijection {
    fn validate(&self, context: &str) -> RusticolResult<()> {
        let count = usize::try_from(self.count)
            .map_err(|_| RusticolError::artifact(format!("{context} count exceeds usize")))?;
        bounded_len(context, count)?;
        let mut mapping = (0..self.count).collect::<Vec<_>>();
        let mut previous = None;
        for [source, target] in &self.changes {
            if *source >= self.count
                || *target >= self.count
                || source == target
                || previous.is_some_and(|value| value >= *source)
            {
                return Err(RusticolError::integrity(format!(
                    "{context} contains a noncanonical change"
                )));
            }
            previous = Some(*source);
            mapping[*source as usize] = *target;
        }
        validate_permutation(&mapping, context)
    }
}

fn validate_permutation(values: &[u32], context: &str) -> RusticolResult<()> {
    bounded_len(context, values.len())?;
    if values
        .iter()
        .copied()
        .any(|value| value as usize >= values.len())
        || values.iter().copied().collect::<BTreeSet<_>>().len() != values.len()
    {
        return Err(RusticolError::integrity(format!(
            "{context} is not a dense zero-based permutation"
        )));
    }
    Ok(())
}

fn validate_ragged_permutations(
    offsets: &[u32],
    values: &[u32],
    row_count: usize,
    context: &str,
) -> RusticolResult<()> {
    if offsets.len() != row_count.saturating_add(1)
        || offsets.first().copied() != Some(0)
        || offsets.last().copied().map(|value| value as usize) != Some(values.len())
        || offsets.windows(2).any(|window| window[0] > window[1])
    {
        return Err(RusticolError::integrity(format!(
            "{context} offsets are invalid"
        )));
    }
    for window in offsets.windows(2) {
        let start = window[0] as usize;
        let stop = window[1] as usize;
        validate_permutation(&values[start..stop], context)?;
    }
    Ok(())
}

impl RecurrenceInspectionSummary {
    fn validate(&self, process_key: &str, color_accuracy: &str) -> RusticolResult<()> {
        if self.execution_mode != "recurrence"
            || self.recurrence_plan_abi != RECURRENCE_PLAN_ABI
            || self.runtime_layout_abi != RECURRENCE_RUNTIME_LAYOUT_ABI
            || self.direct_template_abi != RECURRENCE_DIRECT_TEMPLATE_ABI
        {
            return Err(RusticolError::compatibility(
                "recurrence inspection summary does not match Direct-Arena v2",
            ));
        }
        if self.process_id != process_key {
            return Err(RusticolError::integrity(
                "recurrence inspection process id does not match its execution manifest",
            ));
        }
        validate_counts(
            "recurrence inspection summary",
            &[
                self.parameter_count,
                self.sector_count,
                self.prepared_kernel_count,
                self.direct_executor_count,
            ],
        )?;
        parse_sha256(&self.semantic_digest, "recurrence semantic plan")?;
        parse_sha256(&self.schedule_digest, "recurrence root schedule")?;
        parse_sha256(
            &self.runtime_layout_digest,
            "recurrence Direct-Arena runtime layout",
        )?;
        self.schedule.validate()?;
        self.construction.validate(self.schedule)?;
        self.direct_arena.validate(self.lc_flow_layout)?;
        if self.selector_work_certificate.is_some()
            != self.direct_arena.cache_footprint_policy.is_some()
        {
            return Err(RusticolError::integrity(
                "recurrence selector-work certificate and cache-footprint policy must be declared together",
            ));
        }
        if let Some(certificate) = self.selector_work_certificate.as_ref() {
            certificate.validate(
                self.lc_flow_layout,
                self.schedule,
                self.direct_arena,
                self.sector_count,
            )?;
        }
        self.runtime_container_member.validate()?;
        if let Some(certificate) = self.color_projection_certificate.as_ref() {
            certificate.validate(self.lc_flow_layout)?;
        }
        if let Some(discovery) = self.relation_discovery.as_ref() {
            discovery.validate(color_accuracy, self.lc_flow_layout, self.schedule)?;
        }
        match (
            self.structural_zero_physical_sector_certificate.as_ref(),
            self.lc_flow_layout,
        ) {
            (Some(certificate), RecurrenceLcFlowLayout::ContractedColorUnion) => {
                certificate.validate(self.sector_count, self.schedule.resolved_helicity_count)?;
            }
            (None, RecurrenceLcFlowLayout::ContractedColorUnion) => {
                return Err(RusticolError::integrity(
                    "contracted recurrence inspection omitted its structural-zero physical-sector certificate",
                ));
            }
            (Some(_), _) => {
                return Err(RusticolError::integrity(
                    "structural-zero physical-sector certificate is restricted to contracted recurrence",
                ));
            }
            (None, _) => {}
        }
        self.generation_timings_seconds.validate()?;
        if self.schedule.source_row_count > self.schedule.current_count
            || self.schedule.finalization_count > self.schedule.current_count
        {
            return Err(RusticolError::integrity(
                "recurrence inspection schedule disagrees with its top-level counts",
            ));
        }
        Ok(())
    }
}

impl RecurrenceStructuralZeroPhysicalSectorCertificate {
    fn validate(
        &self,
        physical_sector_count: u64,
        resolved_helicity_count: u64,
    ) -> RusticolResult<()> {
        if self.proof_kind != CLOSURE_TARGET_DOMAIN_PROOF_KIND {
            return Err(RusticolError::compatibility(
                "unsupported recurrence structural-zero physical-sector proof",
            ));
        }
        parse_sha256(
            &self.semantic_digest,
            "recurrence structural-zero physical-sector proof",
        )?;
        if self.sector_count != physical_sector_count
            || self.resolved_helicity_count != resolved_helicity_count
            || self.certified_structural_zero_sector_count > self.sector_count
        {
            return Err(RusticolError::integrity(
                "recurrence structural-zero physical-sector proof disagrees with its inspection domain",
            ));
        }
        Ok(())
    }
}

fn selector_row_count(
    source_rows: u64,
    contributions: u64,
    finalizations: u64,
    closures: u64,
) -> RusticolResult<u64> {
    source_rows
        .checked_add(contributions)
        .and_then(|total| total.checked_add(finalizations))
        .and_then(|total| total.checked_add(closures))
        .ok_or_else(|| RusticolError::artifact("recurrence selector row count overflows u64"))
}

impl RecurrenceSelectorWorkCertificate {
    fn validate(
        &self,
        layout: RecurrenceLcFlowLayout,
        schedule: RecurrenceScheduleSummary,
        direct_arena: RecurrenceDirectArenaSummary,
        sector_count: u64,
    ) -> RusticolResult<()> {
        if self.schema_version != 1 || self.binding != "runtime_layout_digest" {
            return Err(RusticolError::compatibility(
                "recurrence selector-work certificate identity is unsupported",
            ));
        }
        self.persisted_union.validate(schedule, direct_arena)?;
        bounded_len(
            "recurrence selector-work representatives",
            self.representatives.len(),
        )?;
        let has_representatives = !self.representatives.is_empty();
        let has_replay_targets = schedule.replay_target_count != 0;
        match layout {
            RecurrenceLcFlowLayout::TopologyReplay if !has_representatives => {
                return Err(RusticolError::integrity(
                    "recurrence selector-work representatives disagree with the flow layout",
                ));
            }
            RecurrenceLcFlowLayout::AllFlowUnion if has_representatives => {
                return Err(RusticolError::integrity(
                    "recurrence selector-work representatives disagree with the flow layout",
                ));
            }
            // Schema-v1 contracted artifacts predating representative-level
            // accounting legitimately carry replay targets with an empty
            // representative list.  New writers emit representatives, but
            // their absence must remain load-compatible until the certificate
            // schema itself is versioned.
            RecurrenceLcFlowLayout::ContractedColorUnion
                if has_representatives && !has_replay_targets =>
            {
                return Err(RusticolError::integrity(
                    "recurrence selector-work representatives disagree with the flow layout",
                ));
            }
            RecurrenceLcFlowLayout::TopologyReplay
            | RecurrenceLcFlowLayout::AllFlowUnion
            | RecurrenceLcFlowLayout::ContractedColorUnion => {}
        }
        let mut sector_ids = BTreeSet::new();
        let mut public_flow_count = 0_u64;
        for representative in &self.representatives {
            representative.validate(
                self.persisted_union,
                schedule.amplitude_destination_count,
                sector_count,
            )?;
            if !sector_ids.insert(representative.representative_sector_id) {
                return Err(RusticolError::integrity(
                    "recurrence selector-work representative sector is duplicated",
                ));
            }
            public_flow_count = public_flow_count
                .checked_add(representative.public_flow_count)
                .ok_or_else(|| {
                    RusticolError::artifact(
                        "recurrence selector-work public-flow count overflows u64",
                    )
                })?;
        }
        if has_representatives && public_flow_count != schedule.replay_target_count {
            return Err(RusticolError::integrity(
                "recurrence selector-work public-flow count disagrees with replay targets",
            ));
        }
        Ok(())
    }
}

impl RecurrenceSelectorPersistedUnion {
    fn validate(
        self,
        schedule: RecurrenceScheduleSummary,
        direct_arena: RecurrenceDirectArenaSummary,
    ) -> RusticolResult<()> {
        validate_counts(
            "recurrence persisted selector-work summary",
            &[
                self.current_count,
                self.semantic_component_count,
                self.source_row_count,
                self.contribution_count,
                self.finalization_count,
                self.closure_count,
                self.row_count,
            ],
        )?;
        if self.current_count != schedule.current_count
            || self.semantic_component_count != direct_arena.semantic_component_count
            || self.source_row_count != schedule.source_row_count
            || self.contribution_count != schedule.contribution_count
            || self.finalization_count != schedule.finalization_count
            || self.closure_count != schedule.closure_term_count
            || self.row_count
                != selector_row_count(
                    self.source_row_count,
                    self.contribution_count,
                    self.finalization_count,
                    self.closure_count,
                )?
        {
            return Err(RusticolError::integrity(
                "recurrence persisted selector-work summary disagrees with the inspection schedule",
            ));
        }
        Ok(())
    }
}

impl RecurrenceSelectorRepresentative {
    fn validate(
        self,
        persisted: RecurrenceSelectorPersistedUnion,
        amplitude_destination_count: u64,
        sector_count: u64,
    ) -> RusticolResult<()> {
        validate_counts(
            "recurrence selector-work representative",
            &[
                self.representative_sector_id,
                self.public_flow_count,
                self.current_count,
                self.semantic_component_count,
                self.source_row_count,
                self.contribution_count,
                self.finalization_count,
                self.closure_count,
                self.amplitude_destination_count,
                self.row_count,
            ],
        )?;
        if self.representative_sector_id >= sector_count
            || self.public_flow_count == 0
            || self.current_count > persisted.current_count
            || self.semantic_component_count > persisted.semantic_component_count
            || self.source_row_count > persisted.source_row_count
            || self.contribution_count > persisted.contribution_count
            || self.finalization_count > persisted.finalization_count
            || self.closure_count > persisted.closure_count
            || self.amplitude_destination_count > amplitude_destination_count
            || self.row_count
                != selector_row_count(
                    self.source_row_count,
                    self.contribution_count,
                    self.finalization_count,
                    self.closure_count,
                )?
        {
            return Err(RusticolError::integrity(
                "recurrence selector-work representative is inconsistent",
            ));
        }
        Ok(())
    }
}

impl RecurrenceConstructionSummary {
    fn validate(self, schedule: RecurrenceScheduleSummary) -> RusticolResult<()> {
        validate_counts(
            "recurrence construction summary",
            &[
                self.peak_current_count,
                self.peak_contribution_count,
                self.peak_dynamic_color_state_count,
                self.color_target_prune_count,
                self.candidate_parent_pair_count,
            ],
        )?;
        if self.peak_current_count < schedule.current_count
            || self.peak_contribution_count < schedule.contribution_count
            || !self.peak_to_final_current_ratio.is_finite()
            || !self.peak_to_final_contribution_ratio.is_finite()
            || self.peak_to_final_current_ratio < 0.0
            || self.peak_to_final_contribution_ratio < 0.0
            || (schedule.current_count != 0 && self.peak_to_final_current_ratio < 1.0)
            || (schedule.contribution_count != 0 && self.peak_to_final_contribution_ratio < 1.0)
        {
            return Err(RusticolError::integrity(
                "recurrence construction summary is inconsistent with the final schedule",
            ));
        }
        Ok(())
    }
}

impl RecurrenceScheduleSummary {
    fn validate(self) -> RusticolResult<()> {
        validate_counts(
            "recurrence schedule summary",
            &[
                self.current_count,
                self.source_row_count,
                self.contribution_count,
                self.finalization_count,
                self.amplitude_destination_count,
                self.closure_term_count,
                self.replay_target_count,
                self.retained_helicity_count,
                self.resolved_helicity_count,
                self.exact_factor_count,
            ],
        )?;
        if self.resolved_helicity_count > self.retained_helicity_count {
            return Err(RusticolError::integrity(
                "recurrence resolved helicities exceed retained helicities",
            ));
        }
        Ok(())
    }
}

impl RecurrenceDirectArenaSummary {
    fn validate(self, layout: RecurrenceLcFlowLayout) -> RusticolResult<()> {
        validate_counts(
            "recurrence Direct-Arena summary",
            &[
                self.semantic_component_count,
                self.current_arena_components,
                self.arena_component_reuse_count,
                self.momentum_form_count,
                self.selector_domain_count,
                self.row_group_count,
                self.packed_input_bytes,
                self.packed_output_bytes,
                self.scatter_bytes,
            ],
        )?;
        let expected_semantic = self
            .current_arena_components
            .checked_add(self.arena_component_reuse_count)
            .ok_or_else(|| {
                RusticolError::artifact("recurrence Direct-Arena component count overflows u64")
            })?;
        if self.semantic_component_count != expected_semantic {
            return Err(RusticolError::integrity(
                "recurrence Direct-Arena component reuse count is inconsistent",
            ));
        }
        if self.packed_input_bytes != 0 || self.packed_output_bytes != 0 || self.scatter_bytes != 0
        {
            return Err(RusticolError::compatibility(
                "Direct-Arena v2 must not declare packed input, packed output, or scatter bytes",
            ));
        }
        let expected_policy = match layout {
            RecurrenceLcFlowLayout::TopologyReplay => {
                RecurrenceCacheFootprintPolicy::SelectorActiveMaxV1
            }
            RecurrenceLcFlowLayout::AllFlowUnion | RecurrenceLcFlowLayout::ContractedColorUnion => {
                RecurrenceCacheFootprintPolicy::PersistedArenaV1
            }
        };
        if self
            .cache_footprint_policy
            .is_some_and(|policy| policy != expected_policy)
        {
            return Err(RusticolError::integrity(
                "recurrence Direct-Arena cache-footprint policy disagrees with its flow layout",
            ));
        }
        Ok(())
    }
}

impl RecurrenceRuntimeContainerMember {
    fn validate(&self) -> RusticolResult<()> {
        if self.path != RECURRENCE_DIRECT_SCHEDULE_MEMBER {
            return Err(RusticolError::security(format!(
                "Direct-Arena v2 inspection member path must be exactly {RECURRENCE_DIRECT_SCHEDULE_MEMBER:?}"
            )));
        }
        if self.size_bytes == 0 {
            return Err(RusticolError::artifact(
                "Direct-Arena v2 plan member size is outside canonical bounds",
            ));
        }
        if self.container_size_bytes == 0 || self.size_bytes > self.container_size_bytes {
            return Err(RusticolError::artifact(
                "Direct-Arena v2 plan member/container sizes are inconsistent",
            ));
        }
        parse_sha256(&self.sha256, "recurrence Direct-Arena plan member")?;
        Ok(())
    }
}

impl RecurrenceColorProjectionCertificate {
    fn validate(&self, layout: RecurrenceLcFlowLayout) -> RusticolResult<()> {
        if layout != RecurrenceLcFlowLayout::TopologyReplay
            || self.path != RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER
            || self.schema_version != 1
            || self.proof_kind != "exact-rectangular-sum-projection"
        {
            return Err(RusticolError::compatibility(
                "unsupported recurrence color-projection certificate contract",
            ));
        }
        if !self.publishable || self.size_bytes == 0 {
            return Err(RusticolError::integrity(
                "recurrence color-projection certificate metadata is not publishable",
            ));
        }
        parse_sha256(&self.sha256, "recurrence color-projection certificate")?;
        Ok(())
    }
}

impl RecurrenceRelationDiscoverySummary {
    fn validate(
        &self,
        color_accuracy: &str,
        layout: RecurrenceLcFlowLayout,
        schedule: RecurrenceScheduleSummary,
    ) -> RusticolResult<()> {
        if self.schema_version != 1 {
            return Err(RusticolError::compatibility(
                "unsupported recurrence relation-discovery summary schema",
            ));
        }
        let expected_layout = match layout {
            RecurrenceLcFlowLayout::TopologyReplay => "topology-replay",
            RecurrenceLcFlowLayout::AllFlowUnion => "all-flow-union",
            RecurrenceLcFlowLayout::ContractedColorUnion => "contracted-color-union",
        };
        if self.scope.execution_mode != "recurrence"
            || self.scope.representation != "recurrence-direct-plan-v2"
            || self.scope.lc_flow_layout != expected_layout
            || self.scope.color_accuracy != color_accuracy
        {
            return Err(RusticolError::compatibility(
                "unsupported recurrence relation-discovery scope",
            ));
        }
        for (value, context) in [
            (&self.state, "recurrence relation-discovery state"),
            (
                &self.scope.color_accuracy,
                "recurrence relation-discovery color accuracy",
            ),
            (
                &self.follow_up_boundary,
                "recurrence relation-discovery follow-up boundary",
            ),
        ] {
            validate_text(value, context)?;
        }
        validate_counts(
            "recurrence relation-discovery summary",
            &[
                self.numerical_candidate_count,
                self.uncertified_candidate_count,
                self.rejected_hypothesis_count,
                self.exact_certified_relation_count,
                self.applied_relation_count,
                self.interaction_evaluation_count_before,
                self.interaction_evaluation_count_after,
                self.interaction_evaluation_savings,
                self.current_count_before,
                self.current_count_after,
                self.contribution_count_before,
                self.contribution_count_after,
                self.scale_copy_row_count,
                self.certificate_count,
            ],
        )?;
        // Every u64 seed is valid. It remains part of the strict manifest
        // identity even though no runtime arithmetic depends on it.
        let _probe_seed = self.probe.seed;
        if self.probe.status != "completed"
            || self.probe.precision_digits < 80
            || self.probe.probe_count < 2
            || self.probe.verification_probe_count < 2
            || self.probe.effective_projection_count < self.probe.probe_count
            || !self.probe.deterministic
        {
            return Err(RusticolError::integrity(
                "recurrence relation-discovery probe contract is invalid",
            ));
        }
        parse_sha256(
            &self.probe.rejection_decision_sha256,
            "recurrence rejected-hypothesis decisions",
        )?;
        for (digest, context) in [
            (
                self.probe.runtime_parameter_schema_sha256.as_ref(),
                "recurrence runtime-parameter schema",
            ),
            (
                self.probe.candidate_observation_batch_sha256.as_ref(),
                "recurrence candidate observation batch",
            ),
            (
                self.probe.verification_observation_batch_sha256.as_ref(),
                "recurrence verification observation batch",
            ),
            (
                self.probe.decision_sha256.as_ref(),
                "recurrence full hypothesis decisions",
            ),
        ] {
            if let Some(digest) = digest {
                parse_sha256(digest, context)?;
            }
        }
        if !self.probe.candidate_only {
            let authenticated_raw_commitments = [
                self.probe.runtime_parameter_schema_sha256.as_ref(),
                self.probe.candidate_observation_batch_sha256.as_ref(),
                self.probe.verification_observation_batch_sha256.as_ref(),
                self.probe.decision_sha256.as_ref(),
            ];
            let certified = self.exact_certified_relation_count;
            if authenticated_raw_commitments
                .iter()
                .any(|digest| digest.is_none())
                || self.numerical_candidate_count
                    != certified
                        .checked_add(self.probe.verification_rejected_count)
                        .ok_or_else(|| {
                            RusticolError::integrity("recurrence raw candidate count overflows")
                        })?
                || self.uncertified_candidate_count != self.probe.verification_rejected_count
                || self.uncertified_candidate_count
                    != self
                        .numerical_candidate_count
                        .checked_sub(certified)
                        .ok_or_else(|| {
                            RusticolError::integrity("recurrence raw uncertified count underflows")
                        })?
                || self.rejected_hypothesis_count
                    != self
                        .probe
                        .tested_hypothesis_count
                        .checked_sub(certified)
                        .ok_or_else(|| {
                            RusticolError::integrity("recurrence raw rejection count underflows")
                        })?
            {
                return Err(RusticolError::integrity(
                    "recurrence authenticated raw rejection census is inconsistent",
                ));
            }
        }
        let certified = self.exact_certified_relation_count;
        let applied = self.applied_relation_count;
        let expected_state = match self.requested_mode {
            RecurrenceRelationDiscoveryMode::Diagnostic => {
                if applied != 0 || self.scale_copy_row_count != 0 {
                    return Err(RusticolError::integrity(
                        "diagnostic recurrence relation discovery altered the schedule",
                    ));
                }
                "diagnostic-only"
            }
            RecurrenceRelationDiscoveryMode::CertifiedReuse => {
                if applied != certified || self.scale_copy_row_count != applied {
                    return Err(RusticolError::integrity(
                        "certified recurrence relation-discovery counts disagree",
                    ));
                }
                if applied == 0 {
                    "diagnostic-only"
                } else {
                    "exact-certified-applied"
                }
            }
        };
        if self.state != expected_state
            || self.current_count_before != schedule.current_count
            || self.current_count_after != schedule.current_count
            || self.interaction_evaluation_count_before != self.contribution_count_before
            || self.interaction_evaluation_count_after > self.interaction_evaluation_count_before
            || self.interaction_evaluation_savings
                != self
                    .interaction_evaluation_count_before
                    .checked_sub(self.interaction_evaluation_count_after)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence relation-discovery interaction counts underflow",
                        )
                    })?
            || self.contribution_count_after
                != self
                    .interaction_evaluation_count_after
                    .checked_add(certified)
                    .ok_or_else(|| {
                        RusticolError::artifact(
                            "recurrence relation-discovery contribution count overflows",
                        )
                    })?
            || match self.requested_mode {
                RecurrenceRelationDiscoveryMode::Diagnostic => {
                    self.contribution_count_before != schedule.contribution_count
                }
                RecurrenceRelationDiscoveryMode::CertifiedReuse => {
                    self.contribution_count_after != schedule.contribution_count
                }
            }
        {
            return Err(RusticolError::integrity(
                "recurrence relation-discovery summary disagrees with the schedule",
            ));
        }
        if self.certificate_count != certified
            || self.certificates.len()
                != usize::try_from(certified.min(MAX_RELATION_DISCOVERY_SAMPLES as u64)).map_err(
                    |_| {
                        RusticolError::artifact(
                            "recurrence relation-discovery sample count exceeds usize",
                        )
                    },
                )?
            || self.certificates_truncated != (certified > MAX_RELATION_DISCOVERY_SAMPLES as u64)
            || self.certificates.len() > MAX_RELATION_DISCOVERY_SAMPLES
            || self.rejected_candidates.len() > MAX_RELATION_DISCOVERY_SAMPLES
        {
            return Err(RusticolError::integrity(
                "recurrence relation-discovery bounded samples are inconsistent",
            ));
        }
        let replay = &self.certificate_replay;
        if replay.algorithm != relation_certificate_algorithm()
            || replay.status
                != if certified == 0 {
                    "no-certified-relations"
                } else {
                    "verified"
                }
        {
            return Err(RusticolError::compatibility(
                "unsupported recurrence relation-certificate replay contract",
            ));
        }
        parse_sha256(
            &replay.certificate_set_sha256,
            "recurrence relation certificate set",
        )?;
        for certificate in &self.certificates {
            certificate.validate(self.current_count_before)?;
        }
        for rejected in &self.rejected_candidates {
            validate_text(
                &rejected.reason,
                "recurrence relation-discovery rejection reason",
            )?;
        }
        if !self.probe.candidate_only && self.rejected_candidate_diagnostics.is_none() {
            return Err(RusticolError::integrity(
                "full recurrence numerical probes require rejected-candidate diagnostics",
            ));
        }
        if let Some(diagnostics) = self.rejected_candidate_diagnostics.as_ref() {
            let retained = u64::try_from(self.rejected_candidates.len()).map_err(|_| {
                RusticolError::artifact(
                    "recurrence relation-discovery rejection sample count exceeds u64",
                )
            })?;
            let retention_is_consistent = match diagnostics.truncation_policy.as_str() {
                "first-16-in-canonical-discovery-order-v1" => {
                    retained
                        == diagnostics
                            .total_rejected_hypothesis_count
                            .min(MAX_RELATION_DISCOVERY_SAMPLES as u64)
                }
                "none-authenticated-full-rejection-digest-v1" => retained == 0,
                _ => false,
            };
            if diagnostics.total_rejected_hypothesis_count != self.rejected_hypothesis_count
                || diagnostics.retained_count != retained
                || diagnostics.truncated != (diagnostics.total_rejected_hypothesis_count > retained)
                || !retention_is_consistent
            {
                return Err(RusticolError::integrity(
                    "recurrence relation-discovery rejection diagnostics are inconsistent",
                ));
            }
            let retained_rows = self
                .rejected_candidates
                .iter()
                .map(|rejected| json!({"reason": &rejected.reason}))
                .collect::<Vec<_>>();
            let retained_payload = json!({
                "abi": "pyamplicol-rejected-relation-diagnostics-v1",
                "rows": retained_rows,
            });
            let retained_bytes = serde_json::to_vec(&retained_payload).map_err(|error| {
                RusticolError::artifact(format!(
                    "could not canonicalize recurrence retained rejection diagnostics: {error}"
                ))
            })?;
            let expected_retained_sha256 = format!("{:x}", Sha256::digest(retained_bytes));
            if diagnostics.retained_sha256 != expected_retained_sha256 {
                return Err(RusticolError::integrity(
                    "recurrence retained rejection diagnostics commitment is inconsistent",
                ));
            }
            let full_census_payload = json!({
                "abi": "pyamplicol-relation-discovery-full-census-v1",
                "tested_hypothesis_count":
                    self.probe.tested_hypothesis_count,
                "numerical_candidate_count": self.numerical_candidate_count,
                "verification_rejected_count":
                    self.probe.verification_rejected_count,
                "uncertified_candidate_count":
                    self.uncertified_candidate_count,
                "certified_relation_count":
                    self.exact_certified_relation_count,
                "rejected_hypothesis_count":
                    self.rejected_hypothesis_count,
                "applied_relation_count": self.applied_relation_count,
                "runtime_parameter_schema_sha256":
                    self.probe.runtime_parameter_schema_sha256,
                "candidate_observation_batch_sha256":
                    self.probe.candidate_observation_batch_sha256,
                "verification_observation_batch_sha256":
                    self.probe.verification_observation_batch_sha256,
                "decision_sha256": self.probe.decision_sha256,
                "rejection_decision_sha256":
                    self.probe.rejection_decision_sha256,
            });
            let full_census_bytes = serde_json::to_vec(&full_census_payload).map_err(|error| {
                RusticolError::artifact(format!(
                    "could not canonicalize recurrence rejection census: {error}"
                ))
            })?;
            let expected_full_census_sha256 = format!("{:x}", Sha256::digest(full_census_bytes));
            if diagnostics.full_census_sha256 != expected_full_census_sha256
                || diagnostics.full_rejection_sha256 != self.probe.rejection_decision_sha256
            {
                return Err(RusticolError::integrity(
                    "recurrence full rejection census or decision commitment is inconsistent",
                ));
            }
            for (digest, context) in [
                (
                    &diagnostics.retained_sha256,
                    "recurrence retained rejection diagnostics",
                ),
                (
                    &diagnostics.full_census_sha256,
                    "recurrence rejection full census",
                ),
                (
                    &diagnostics.full_rejection_sha256,
                    "recurrence full rejection decisions",
                ),
            ] {
                parse_sha256(digest, context)?;
            }
        }
        Ok(())
    }
}

impl RecurrenceRelationCertificateSample {
    fn validate(&self, current_count: u64) -> RusticolResult<()> {
        if self.algorithm != relation_certificate_algorithm()
            || self.current_id >= current_count
            || self.representative_id >= current_count
            || self.current_id == self.representative_id
        {
            return Err(RusticolError::integrity(
                "recurrence relation-certificate sample is inconsistent",
            ));
        }
        self.factor_exact_rational.validate()?;
        for (digest, context) in [
            (
                &self.current_term_vector_sha256,
                "recurrence current term vector",
            ),
            (
                &self.representative_term_vector_sha256,
                "recurrence representative term vector",
            ),
            (&self.proof_sha256, "recurrence relation proof"),
        ] {
            parse_sha256(digest, context)?;
        }
        Ok(())
    }
}

impl RecurrenceRelationExactFactor {
    fn validate(&self) -> RusticolResult<()> {
        for (value, context) in [
            (&self.real_numerator, "recurrence relation real numerator"),
            (
                &self.real_denominator,
                "recurrence relation real denominator",
            ),
            (
                &self.imag_numerator,
                "recurrence relation imaginary numerator",
            ),
            (
                &self.imag_denominator,
                "recurrence relation imaginary denominator",
            ),
        ] {
            validate_text(value, context)?;
            value.parse::<i128>().map_err(|_| {
                RusticolError::artifact(format!("{context} is not a bounded integer"))
            })?;
        }
        if self.real_denominator == "0" || self.imag_denominator == "0" {
            return Err(RusticolError::artifact(
                "recurrence relation factor denominator is zero",
            ));
        }
        Ok(())
    }
}

impl RecurrenceGenerationTimings {
    fn validate(self) -> RusticolResult<()> {
        let values = [
            self.python_extraction,
            self.catalog_authentication,
            self.semantic_construction,
            self.direct_lowering,
            self.serialization,
            self.native_total,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(RusticolError::artifact(
                "recurrence Direct-Arena generation timings must be finite and nonnegative",
            ));
        }
        Ok(())
    }
}

impl RecurrenceSummary {
    fn validate(&self) -> RusticolResult<()> {
        if self.builder_input_abi != RECURRENCE_BUILDER_INPUT_ABI {
            return Err(unsupported_abi(
                "recurrence builder input",
                &self.builder_input_abi,
            ));
        }
        if self.template_input_abi != RECURRENCE_TEMPLATE_INPUT_ABI {
            return Err(unsupported_abi(
                "recurrence template input",
                &self.template_input_abi,
            ));
        }
        validate_counts(
            "recurrence summary",
            &[
                self.current_count,
                self.contribution_count,
                self.closure_term_count,
            ],
        )
    }
}

impl RecurrenceRuntimeMetadata {
    fn validate(
        &self,
        external_pdgs: &[i32],
        color_accuracy: &str,
        inspection: &RecurrenceInspectionSummary,
    ) -> RusticolResult<()> {
        for (context, len) in [
            ("public color flows", self.public_color_flows.len()),
            ("runtime parameters", self.runtime_parameters.len()),
            (
                "prepared parameter defaults",
                self.prepared_parameter_defaults.len(),
            ),
            ("parameter projection", self.parameter_projection.len()),
            ("source templates", self.source_templates.len()),
            ("external legs", self.external_legs.len()),
            ("particle masses", self.particle_masses.len()),
        ] {
            bounded_len(context, len)?;
        }
        let expected_public_flow_count = match inspection.lc_flow_layout {
            RecurrenceLcFlowLayout::TopologyReplay => Some(inspection.schedule.replay_target_count),
            // Folded trace reflections remain distinct public flows while
            // sharing one authenticated construction sector.
            RecurrenceLcFlowLayout::AllFlowUnion => None,
            RecurrenceLcFlowLayout::ContractedColorUnion => Some(0),
        };
        if expected_public_flow_count
            .is_some_and(|count| self.public_color_flows.len() as u64 != count)
        {
            return Err(RusticolError::integrity(
                "recurrence public color-flow count does not match the Direct-Arena replay count",
            ));
        }
        let mut public_ids = BTreeSet::new();
        let mut construction_sector_ids = BTreeSet::new();
        for (target_sector_id, flow) in self.public_color_flows.iter().enumerate() {
            validate_text(&flow.public_id, "recurrence public color-flow ID")?;
            if !public_ids.insert(flow.public_id.as_str()) {
                return Err(RusticolError::artifact(
                    "recurrence public color-flow IDs are not unique",
                ));
            }
            if u64::from(flow.construction_sector_id) >= inspection.sector_count {
                return Err(RusticolError::integrity(
                    "recurrence public color flow references an absent construction sector",
                ));
            }
            construction_sector_ids.insert(flow.construction_sector_id);
            if flow.target_sector_id as usize != target_sector_id
                || flow.target_sector_id as usize >= self.public_color_flows.len()
            {
                return Err(RusticolError::integrity(
                    "recurrence public color-flow target sectors are not dense or in range",
                ));
            }
        }
        if inspection.lc_flow_layout == RecurrenceLcFlowLayout::AllFlowUnion
            && (construction_sector_ids.len() as u64 != inspection.sector_count
                || construction_sector_ids.iter().copied().ne(0..u32::try_from(
                    inspection.sector_count,
                )
                .map_err(|_| {
                    RusticolError::artifact("recurrence Direct-Arena sector count exceeds u32")
                })?))
        {
            return Err(RusticolError::integrity(
                "recurrence public color flows do not cover every Direct-Arena construction sector",
            ));
        }
        match inspection.lc_flow_layout {
            RecurrenceLcFlowLayout::TopologyReplay | RecurrenceLcFlowLayout::AllFlowUnion => {
                if color_accuracy != "lc" || self.public_color_flows.is_empty() {
                    return Err(RusticolError::artifact(
                        "LC recurrence runtime metadata requires public physical color flows",
                    ));
                }
                if self.color_contraction.is_some() {
                    return Err(RusticolError::artifact(
                        "LC recurrence must not carry a contracted-color payload",
                    ));
                }
            }
            RecurrenceLcFlowLayout::ContractedColorUnion => {
                if !matches!(color_accuracy, "nlc" | "full") || !self.public_color_flows.is_empty()
                {
                    return Err(RusticolError::artifact(
                        "contracted recurrence metadata requires NLC/full color and no public color-flow axis",
                    ));
                }
                let color_contraction = self.color_contraction.as_ref().ok_or_else(|| {
                    RusticolError::artifact(
                        "NLC/full recurrence has no contracted-color payload reference",
                    )
                })?;
                color_contraction.validate(color_accuracy, inspection)?;
            }
        }
        validate_recurrence_parameter_runtime_parts(
            &self.runtime_parameters,
            &self.prepared_parameter_defaults,
            &self.parameter_projection,
            inspection.parameter_count,
        )?;

        if self.source_templates.is_empty() {
            return Err(RusticolError::artifact(
                "recurrence runtime metadata has no source templates",
            ));
        }
        let mut previous_source_id = None;
        let mut source_pdgs = BTreeSet::new();
        for (index, source) in self.source_templates.iter().enumerate() {
            if previous_source_id.is_some_and(|previous| previous >= source.source_template_id) {
                return Err(RusticolError::artifact(
                    "recurrence source templates are not in strict source-template ID order",
                ));
            }
            previous_source_id = Some(source.source_template_id);
            source.validate(index)?;
            source_pdgs.insert(source.source_ir.identity.pdg_label);
        }

        if self.external_legs.len() != external_pdgs.len() {
            return Err(RusticolError::integrity(
                "recurrence runtime external legs do not match the execution PDG order",
            ));
        }
        let mut public_labels = BTreeSet::new();
        let mut saw_final = false;
        for (source_slot, (leg, expected_pdg)) in
            self.external_legs.iter().zip(external_pdgs).enumerate()
        {
            if leg.source_slot as usize != source_slot
                || leg.public_label == 0
                || !public_labels.insert(leg.public_label)
            {
                return Err(RusticolError::artifact(
                    "recurrence external source slots or public labels are not canonical",
                ));
            }
            if leg.physical_pdg != *expected_pdg {
                return Err(RusticolError::integrity(
                    "recurrence external physical PDGs disagree with the execution manifest",
                ));
            }
            if saw_final && leg.is_initial {
                return Err(RusticolError::artifact(
                    "recurrence initial-state legs must precede final-state legs",
                ));
            }
            saw_final |= !leg.is_initial;
            if !source_pdgs.contains(&leg.outgoing_pdg) {
                return Err(RusticolError::integrity(format!(
                    "recurrence external outgoing PDG {} has no source template",
                    leg.outgoing_pdg
                )));
            }
        }

        let mut mass_pdgs = BTreeSet::new();
        let mut previous_mass_pdg = None;
        for row in &self.particle_masses {
            if previous_mass_pdg.is_some_and(|previous| previous >= row.outgoing_pdg)
                || !mass_pdgs.insert(row.outgoing_pdg)
            {
                return Err(RusticolError::artifact(
                    "recurrence particle masses are not in strict outgoing-PDG order",
                ));
            }
            previous_mass_pdg = Some(row.outgoing_pdg);
            if !row.mass.is_finite() || row.mass < 0.0 {
                return Err(RusticolError::artifact(
                    "recurrence particle masses must be finite and nonnegative",
                ));
            }
        }
        if mass_pdgs != source_pdgs {
            return Err(RusticolError::integrity(
                "recurrence particle masses do not cover exactly the source-template particles",
            ));
        }
        self.normalization.validate(color_accuracy)
    }
}

/// Validate the operational parameter schema shared by the broad manifest
/// and the compact process-ready recipe.  Keeping this boundary shared avoids
/// accepting a recipe that would bind runtime slots differently from the
/// generation manifest it was derived from.
pub(super) fn validate_recurrence_parameter_runtime_parts(
    parameters: &[RecurrenceRuntimeParameter],
    prepared_parameter_defaults: &[[f64; 2]],
    parameter_projection: &[RecurrenceParameterProjection],
    expected_parameter_count: u64,
) -> RusticolResult<()> {
    if prepared_parameter_defaults.len() as u64 != expected_parameter_count {
        return Err(RusticolError::integrity(
            "recurrence prepared-parameter defaults do not match the plan parameter count",
        ));
    }
    if prepared_parameter_defaults
        .iter()
        .flatten()
        .any(|component| !component.is_finite())
    {
        return Err(RusticolError::artifact(
            "recurrence prepared-parameter defaults must be finite complex-f64 values",
        ));
    }
    validate_parameter_projection(prepared_parameter_defaults.len(), parameter_projection)?;
    validate_runtime_parameters(
        parameters,
        prepared_parameter_defaults,
        parameter_projection,
    )
}

fn validate_runtime_parameters(
    parameters: &[RecurrenceRuntimeParameter],
    prepared_parameter_defaults: &[[f64; 2]],
    parameter_projection: &[RecurrenceParameterProjection],
) -> RusticolResult<()> {
    if parameters.len() != parameter_projection.len() {
        return Err(RusticolError::integrity(
            "recurrence runtime parameters do not cover the parameter projection",
        ));
    }
    let mut previous_runtime_name: Option<&str> = None;
    let mut previous_kind: Option<&str> = None;
    for (parameter_index, (parameter, projection)) in
        parameters.iter().zip(parameter_projection).enumerate()
    {
        if parameter.parameter_index as usize != parameter_index
            || parameter.parameter_index != projection.runtime_slot
        {
            return Err(RusticolError::artifact(
                "recurrence runtime parameter indices must be dense and match projection slots",
            ));
        }
        validate_text(&parameter.name, "recurrence runtime parameter name")?;
        validate_text(&parameter.kind, "recurrence runtime parameter kind")?;
        if !parameter.default.is_finite() {
            return Err(RusticolError::artifact(
                "recurrence runtime parameter defaults must be finite",
            ));
        }

        match (&parameter.runtime_name, &parameter.complex_component) {
            (Some(runtime_name), Some(component)) => {
                validate_text(runtime_name, "recurrence runtime parameter public name")?;
                let expected_component = match projection.component {
                    0 => "real",
                    1 => "imag",
                    _ => {
                        return Err(RusticolError::integrity(
                            "recurrence runtime parameter has an invalid projected component",
                        ));
                    }
                };
                if component != expected_component
                    || projection.runtime_name != *runtime_name
                    || parameter.name != format!("{runtime_name}.{component}")
                {
                    return Err(RusticolError::integrity(
                        "recurrence complex runtime parameter disagrees with its projection",
                    ));
                }
                if previous_runtime_name == Some(runtime_name.as_str())
                    && previous_kind != Some(parameter.kind.as_str())
                {
                    return Err(RusticolError::integrity(
                        "recurrence complex runtime parameter components have different kinds",
                    ));
                }
                previous_runtime_name = Some(runtime_name);
                previous_kind = Some(&parameter.kind);
            }
            (None, None) => {
                if projection.component != 0 || parameter.name != projection.runtime_name {
                    return Err(RusticolError::integrity(
                        "recurrence real runtime parameter disagrees with its projection",
                    ));
                }
                previous_runtime_name = None;
                previous_kind = None;
            }
            _ => {
                return Err(RusticolError::artifact(
                    "recurrence runtime_name and complex_component must be both present or both absent",
                ));
            }
        }

        if let Some(prepared_id) = projection.prepared_parameter_id {
            let expected_default =
                prepared_parameter_defaults[prepared_id as usize][projection.component as usize];
            if parameter.default != expected_default {
                return Err(RusticolError::integrity(
                    "recurrence runtime parameter default disagrees with its prepared default",
                ));
            }
        }
    }
    Ok(())
}

fn validate_parameter_projection(
    parameter_count: usize,
    parameter_projection: &[RecurrenceParameterProjection],
) -> RusticolResult<()> {
    let mut previous_key: Option<(&str, u32)> = None;
    let mut current_name: Option<&str> = None;
    let mut current_template = None;
    let mut current_prepared = None;
    for (runtime_slot, row) in parameter_projection.iter().enumerate() {
        if row.runtime_slot as usize != runtime_slot {
            return Err(RusticolError::artifact(
                "recurrence runtime parameter slots must be dense and ordered from zero",
            ));
        }
        validate_text(&row.runtime_name, "recurrence runtime parameter name")?;
        if row.component > 1 {
            return Err(RusticolError::artifact(
                "recurrence runtime parameter component must be zero or one",
            ));
        }
        if row.parameter_template_id as usize >= parameter_count
            || row
                .prepared_parameter_id
                .is_some_and(|id| id as usize >= parameter_count)
        {
            return Err(RusticolError::integrity(
                "recurrence parameter projection references an absent parameter",
            ));
        }
        let key = (row.runtime_name.as_str(), row.component);
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(RusticolError::artifact(
                "recurrence parameter projection is not in strict name/component order",
            ));
        }
        previous_key = Some(key);
        if current_name == Some(row.runtime_name.as_str()) {
            if row.component != 1
                || current_template != Some(row.parameter_template_id)
                || current_prepared != Some(row.prepared_parameter_id)
            {
                return Err(RusticolError::integrity(
                    "recurrence complex parameter projection rows are inconsistent",
                ));
            }
        } else {
            if row.component != 0 {
                return Err(RusticolError::artifact(
                    "recurrence parameter projection must begin each parameter at component zero",
                ));
            }
            current_name = Some(row.runtime_name.as_str());
            current_template = Some(row.parameter_template_id);
            current_prepared = Some(row.prepared_parameter_id);
        }
    }
    Ok(())
}

impl RecurrenceColorContractionReference {
    fn validate(
        &self,
        color_accuracy: &str,
        inspection: &RecurrenceInspectionSummary,
    ) -> RusticolResult<()> {
        if self.abi != RECURRENCE_COLOR_CONTRACTION_CODEC_ABI
            || self.path != RECURRENCE_COLOR_CONTRACTION_PATH
        {
            return Err(RusticolError::compatibility(
                "unsupported recurrence color-contraction payload contract",
            ));
        }
        if self.color_accuracy != color_accuracy
            || !matches!(self.color_accuracy.as_str(), "nlc" | "full")
            || !matches!(
                self.storage.as_str(),
                "expanded" | "repeated" | "convolution-kernels"
            )
        {
            return Err(RusticolError::integrity(
                "recurrence color-contraction summary has incompatible accuracy or storage",
            ));
        }
        if !self.includes_color_factor {
            return Err(RusticolError::integrity(
                "NLC/full recurrence color contraction must include the color factor",
            ));
        }
        if let Some(factorization) = &self.factorization {
            let valid_kind_and_rank = match factorization.kind.as_str() {
                "klein-four-walsh" => factorization.rank == 2,
                "elementary-abelian-walsh" => (3..u32::BITS).contains(&factorization.rank),
                "symmetric-group-fourier" => (2..=10).contains(&factorization.rank),
                _ => false,
            };
            let valid_storage = match factorization.kind.as_str() {
                "symmetric-group-fourier" => self.storage == "convolution-kernels",
                _ => self.storage == "repeated",
            };
            if !valid_storage || !valid_kind_and_rank || factorization.coset_count == 0 {
                return Err(RusticolError::integrity(
                    "recurrence color factorization summary is inconsistent",
                ));
            }
        }
        if (self.storage == "convolution-kernels")
            != self
                .factorization
                .as_ref()
                .is_some_and(|value| value.kind == "symmetric-group-fourier")
        {
            return Err(RusticolError::integrity(
                "recurrence convolution-kernel storage requires symmetric-group Fourier factorization",
            ));
        }
        match (&self.factorization, &self.fft_provenance) {
            (Some(factorization), Some(provenance))
                if factorization.kind == "symmetric-group-fourier" =>
            {
                self.validate_fft_provenance(factorization, provenance)?;
            }
            (Some(factorization), None) if factorization.kind == "symmetric-group-fourier" => {
                return Err(RusticolError::integrity(
                    "recurrence symmetric-group Fourier summary is missing FFT provenance",
                ));
            }
            (_, Some(_)) => {
                return Err(RusticolError::integrity(
                    "recurrence non-FFT color summary carries FFT provenance",
                ));
            }
            _ => {}
        }
        parse_sha256(&self.sha256, "recurrence color-contraction payload")?;
        parse_sha256(
            &self.semantic_digest,
            "recurrence color-contraction semantic payload",
        )?;
        let expected_destination_count = if inspection.schedule.replay_target_count != 0 {
            inspection
                .sector_count
                .checked_mul(inspection.schedule.resolved_helicity_count)
                .ok_or_else(|| {
                    RusticolError::artifact(
                        "recurrence replay color-contraction destination count overflows",
                    )
                })?
        } else {
            inspection.schedule.amplitude_destination_count
        };
        if self.sha256 != self.semantic_digest
            || self.size_bytes == 0
            || self.sector_count != inspection.sector_count
            || self.component_count != inspection.schedule.resolved_helicity_count
            || self.destination_count != expected_destination_count
            || self.materialized_destination_count.is_some_and(|count| {
                count != inspection.schedule.amplitude_destination_count
                    || count > self.destination_count
            })
            || self.active_sector_count == 0
            || self.active_sector_count > self.sector_count
            || self.group_count < self.active_sector_count
            || self.group_count
                > self
                    .active_sector_count
                    .checked_mul(self.component_count)
                    .ok_or_else(|| {
                        RusticolError::artifact(
                            "recurrence color-contraction active group bound overflows",
                        )
                    })?
            || self.entry_count == 0
            || self.logical_entry_count < self.entry_count
        {
            return Err(RusticolError::integrity(
                "recurrence color-contraction summary is inconsistent with the Direct-Arena schedule",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_fft_provenance(
        &self,
        factorization: &RecurrenceColorFactorizationReference,
        provenance: &RecurrenceColorFftProvenance,
    ) -> RusticolResult<()> {
        let group_order = (2..=factorization.rank)
            .try_fold(1_u64, |value, factor| value.checked_mul(u64::from(factor)));
        let Some(group_order) = group_order else {
            return Err(RusticolError::artifact(
                "recurrence symmetric-group order overflows u64",
            ));
        };
        let Some(covered_local_group_count) = factorization.coset_count.checked_mul(group_order)
        else {
            return Err(RusticolError::artifact(
                "recurrence FFT-covered local group count overflows u64",
            ));
        };
        if self.component_count == 0 || !self.group_count.is_multiple_of(self.component_count) {
            return Err(RusticolError::integrity(
                "recurrence symmetric-group component map is not rectangular",
            ));
        }
        let local_group_count = self.group_count / self.component_count;
        let Some(residual_group_count) = local_group_count.checked_sub(covered_local_group_count)
        else {
            return Err(RusticolError::integrity(
                "recurrence FFT-covered groups exceed the local color domain",
            ));
        };
        let upper_triangle_count = |count: u64| {
            count
                .checked_add(1)
                .and_then(|successor| count.checked_mul(successor))
                .and_then(|value| value.checked_div(2))
        };
        let Some(total_pair_count) = upper_triangle_count(local_group_count) else {
            return Err(RusticolError::artifact(
                "recurrence local color pair count overflows u64",
            ));
        };
        let Some(covered_pair_count) = upper_triangle_count(covered_local_group_count) else {
            return Err(RusticolError::artifact(
                "recurrence FFT-covered color pair count overflows u64",
            ));
        };
        let residual_entry_count = total_pair_count - covered_pair_count;
        let Some(channel_pair_count) = upper_triangle_count(factorization.coset_count) else {
            return Err(RusticolError::artifact(
                "recurrence FFT channel-pair count overflows u64",
            ));
        };
        let Some(kernel_entry_count) = channel_pair_count.checked_mul(group_order) else {
            return Err(RusticolError::artifact(
                "recurrence FFT kernel entry count overflows u64",
            ));
        };
        let Some(raw_kernel_bytes) = kernel_entry_count.checked_mul(16) else {
            return Err(RusticolError::artifact(
                "recurrence FFT kernel storage exceeds u64",
            ));
        };
        let Some(transformed_kernel_bytes) = kernel_entry_count.checked_mul(8) else {
            return Err(RusticolError::artifact(
                "recurrence transformed FFT kernel storage exceeds u64",
            ));
        };
        let expected_entry_count = kernel_entry_count
            .checked_add(residual_entry_count)
            .ok_or_else(|| RusticolError::artifact("recurrence FFT entry count overflows u64"))?;
        let expected_logical_entry_count = expected_entry_count
            .checked_mul(self.component_count)
            .ok_or_else(|| {
            RusticolError::artifact("recurrence FFT logical entry count overflows u64")
        })?;
        if provenance.method != "symmetric-group-fourier"
            || provenance.degree != factorization.rank
            || provenance.channel_count != factorization.coset_count
            || provenance.covered_local_group_count != covered_local_group_count
            || provenance.residual_group_count != residual_group_count
            || provenance.residual_entry_count != residual_entry_count
            || provenance.raw_kernel_bytes != raw_kernel_bytes
            || provenance.transformed_kernel_bytes != transformed_kernel_bytes
            || provenance.capability != SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY
            || self.entry_count != expected_entry_count
            || self.logical_entry_count != expected_logical_entry_count
        {
            return Err(RusticolError::integrity(
                "recurrence symmetric-group FFT provenance is inconsistent",
            ));
        }
        Ok(())
    }
}

impl RecurrenceSourceTemplate {
    fn validate(&self, index: usize) -> RusticolResult<()> {
        if self.dimension == 0 || u32::try_from(self.dimension).is_err() {
            return Err(RusticolError::artifact(format!(
                "recurrence source template {index} has an unsupported dimension"
            )));
        }
        self.source_ir.validate(index)?;
        self.crossing.validate(index, "source-template")?;
        if self.dimension != self.source_ir.component_dimension
            || self.crossing != self.source_ir.crossing
        {
            return Err(RusticolError::integrity(format!(
                "recurrence source template {index} disagrees with its typed SourceIR"
            )));
        }
        let declared = self.source_ir.states.iter().any(|state| {
            state.helicity == self.helicity
                && state.chirality == self.chirality
                && state.spin_state == RecurrenceSourceSpinState::Scalar(self.spin_state)
        });
        if !declared {
            return Err(RusticolError::artifact(format!(
                "recurrence source template {index} state is not declared by its typed SourceIR"
            )));
        }
        Ok(())
    }
}

impl RecurrenceGenericSourceIr {
    fn validate(&self, index: usize) -> RusticolResult<()> {
        let identity = &self.identity;
        for (label, value) in [
            ("canonical id", identity.canonical_id.as_str()),
            ("species id", identity.species_id.as_str()),
            ("anti-canonical id", identity.anti_canonical_id.as_str()),
            ("display name", identity.display_name.as_str()),
            ("anti-display name", identity.anti_display_name.as_str()),
        ] {
            validate_text(value, &format!("recurrence source {index} {label}"))?;
        }
        if identity.anti_pdg_label == 0 {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} has an invalid antiparticle relation"
            )));
        }
        let canonical_self_conjugate = identity.canonical_id == identity.anti_canonical_id;
        let pdg_self_conjugate = identity.pdg_label == identity.anti_pdg_label;
        if identity.self_conjugate != canonical_self_conjugate
            || identity.self_conjugate != pdg_self_conjugate
            || (identity.orientation == RecurrenceSourceOrientation::SelfConjugate)
                != identity.self_conjugate
        {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} orientation is inconsistent with its antiparticle relation"
            )));
        }
        validate_text(&self.basis, &format!("recurrence source {index} basis"))?;
        for (label, value) in [
            ("mass parameter", self.mass_parameter.as_deref()),
            ("width parameter", self.width_parameter.as_deref()),
        ] {
            if let Some(value) = value {
                validate_text(value, &format!("recurrence source {index} {label}"))?;
            }
        }
        if self.component_dimension == 0 || u32::try_from(self.component_dimension).is_err() {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} has an unsupported component dimension"
            )));
        }
        bounded_len("recurrence SourceIR states", self.states.len())?;
        if self.states.is_empty() {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} has no declared SourceIR states"
            )));
        }
        self.crossing.validate(index, "declared")?;

        let expected_statistics = match self.wavefunction_family {
            RecurrenceWavefunctionFamily::Fermion => RecurrenceParticleStatistics::Fermion,
            RecurrenceWavefunctionFamily::Ghost => RecurrenceParticleStatistics::Ghost,
            RecurrenceWavefunctionFamily::Auxiliary => RecurrenceParticleStatistics::Auxiliary,
            RecurrenceWavefunctionFamily::Scalar
            | RecurrenceWavefunctionFamily::Vector
            | RecurrenceWavefunctionFamily::Spin2 => RecurrenceParticleStatistics::Boson,
        };
        if self.statistics != expected_statistics {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} statistics disagree with its wavefunction family"
            )));
        }
        match (self.wavefunction_family, self.component_dimension) {
            (RecurrenceWavefunctionFamily::Fermion, 2 | 4) => {
                if identity.self_conjugate {
                    return Err(RusticolError::artifact(format!(
                        "recurrence source {index} is an unsupported self-conjugate fermion source"
                    )));
                }
            }
            (RecurrenceWavefunctionFamily::Scalar, 1)
            | (RecurrenceWavefunctionFamily::Vector, 4)
            | (RecurrenceWavefunctionFamily::Spin2, 16) => {}
            _ => {
                return Err(RusticolError::artifact(format!(
                    "recurrence source {index} has an unsupported wavefunction family/dimension pair"
                )));
            }
        }

        let max_helicity = if self.wavefunction_family == RecurrenceWavefunctionFamily::Spin2 {
            2
        } else {
            1
        };
        let mut states = BTreeSet::new();
        for (state_index, state) in self.states.iter().enumerate() {
            if state.chirality.unsigned_abs() > 1 || state.helicity.unsigned_abs() > max_helicity {
                return Err(RusticolError::artifact(format!(
                    "recurrence source {index} SourceIR state {state_index} is outside the supported helicity/chirality range"
                )));
            }
            if let RecurrenceSourceSpinState::Components(components) = &state.spin_state {
                bounded_len("structured source spin state", components.len())?;
                if components.is_empty() || components.len() as u64 > self.component_dimension {
                    return Err(RusticolError::artifact(format!(
                        "recurrence source {index} SourceIR state {state_index} has invalid structured spin metadata"
                    )));
                }
            }
            if !states.insert(state) {
                return Err(RusticolError::artifact(format!(
                    "recurrence source {index} contains duplicate SourceIR states"
                )));
            }
        }
        Ok(())
    }
}

impl RecurrenceGenericCrossingIr {
    fn validate(&self, index: usize, label: &str) -> RusticolResult<()> {
        if ![-1, 1].contains(&self.helicity_factor)
            || ![-1, 1].contains(&self.chirality_factor)
            || ![-1, 1].contains(&self.spin_state_factor)
            || !self.phase.iter().all(|component| component.is_finite())
            || self.phase == [0.0, 0.0]
        {
            return Err(RusticolError::artifact(format!(
                "recurrence source {index} has invalid {label} CrossingIR metadata"
            )));
        }
        Ok(())
    }
}

impl RecurrenceNormalization {
    fn validate(&self, color_accuracy: &str) -> RusticolResult<()> {
        if self.color_accuracy != color_accuracy {
            return Err(RusticolError::integrity(
                "recurrence normalization color accuracy disagrees with the process",
            ));
        }
        if [
            self.color_factor,
            self.average_factor,
            self.identical_factor,
            self.global_coupling_factor,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(RusticolError::artifact(
                "recurrence normalization factors must be finite",
            ));
        }
        if self.average_factor <= 0.0 || self.identical_factor <= 0.0 {
            return Err(RusticolError::artifact(
                "recurrence averaging and identical-particle factors must be positive",
            ));
        }
        if self.qcd_coupling_power.is_some() != self.electroweak_coupling_power.is_some() {
            return Err(RusticolError::artifact(
                "recurrence normalization coupling powers must be both present or both absent",
            ));
        }
        validate_counts(
            "recurrence normalization",
            &[
                self.qcd_coupling_power.unwrap_or(0),
                self.electroweak_coupling_power.unwrap_or(0),
            ],
        )?;
        validate_text(
            &self.coupling_policy,
            "recurrence normalization coupling policy",
        )?;
        Ok(())
    }
}

/// Parse and validate one bounded recurrence `execution.json`.
pub(super) fn parse_recurrence_execution_manifest(
    bytes: &[u8],
    path: &str,
    outer: &ArtifactProcess,
) -> RusticolResult<RecurrenceExecutionManifest> {
    let manifest: RecurrenceExecutionManifest = serde_json::from_slice(bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse bounded recurrence execution manifest {path:?}: {error}"
        ))
    })?;
    manifest.validate(outer)?;
    Ok(manifest)
}

fn validate_capabilities(
    capabilities: &[String],
    color_accuracy: &str,
    uses_symmetric_group_fft: bool,
    has_helicity_selector_companion: bool,
    context: &str,
) -> RusticolResult<()> {
    let color_capability = match color_accuracy {
        "lc" => RECURRENCE_LC_COLOR_CAPABILITY,
        "nlc" | "full" => RECURRENCE_CONTRACTED_COLOR_CAPABILITY,
        other => {
            return Err(RusticolError::compatibility(format!(
                "{context} has unsupported recurrence color accuracy {other:?}"
            )));
        }
    };
    let mut expected = Vec::from([color_capability, RECURRENCE_RUNTIME_CAPABILITY]);
    if uses_symmetric_group_fft {
        expected.push(SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY);
    }
    if has_helicity_selector_companion {
        expected.push(RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY);
    }
    expected.sort_unstable();
    if capabilities.iter().map(String::as_str).collect::<Vec<_>>() != expected {
        return Err(RusticolError::compatibility(format!(
            "{context} must require exactly {expected:?}"
        )));
    }
    Ok(())
}

fn validate_direct_contract(
    builder_input_abi: &str,
    recurrence_plan_abi: &str,
    runtime_layout_abi: &str,
    direct_template_abi: &str,
    direct_backend_abi: &str,
    context: &str,
) -> RusticolResult<()> {
    for (label, actual, expected) in [
        (
            "builder input",
            builder_input_abi,
            RECURRENCE_BUILDER_INPUT_ABI,
        ),
        ("plan", recurrence_plan_abi, RECURRENCE_PLAN_ABI),
        (
            "runtime layout",
            runtime_layout_abi,
            RECURRENCE_RUNTIME_LAYOUT_ABI,
        ),
        (
            "direct template",
            direct_template_abi,
            RECURRENCE_DIRECT_TEMPLATE_ABI,
        ),
        (
            "direct backend",
            direct_backend_abi,
            RECURRENCE_DIRECT_BACKEND_ABI,
        ),
    ] {
        if actual != expected {
            return Err(RusticolError::compatibility(format!(
                "{context} has unsupported Direct-Arena v2 {label} ABI {actual:?}; regenerate the recurrence artifact"
            )));
        }
    }
    Ok(())
}

fn bounded_len(context: &str, len: usize) -> RusticolResult<()> {
    if u32::try_from(len).is_err() {
        return Err(RusticolError::artifact(format!(
            "{context} count exceeds the u32 identifier domain"
        )));
    }
    Ok(())
}

fn validate_counts(context: &str, counts: &[u64]) -> RusticolResult<()> {
    if counts.iter().any(|count| usize::try_from(*count).is_err()) {
        return Err(RusticolError::artifact(format!(
            "{context} count exceeds the platform index domain"
        )));
    }
    Ok(())
}

fn validate_text(value: &str, context: &str) -> RusticolResult<()> {
    if value.is_empty() {
        return Err(RusticolError::artifact(format!("{context} is empty")));
    }
    Ok(())
}

fn unsupported_abi(context: &str, value: &str) -> RusticolError {
    RusticolError::compatibility(format!("unsupported {context} ABI {value:?}"))
}

fn parse_sha256(value: &str, context: &str) -> RusticolResult<[u8; 32]> {
    if value.len() != 64
        || value
            .as_bytes()
            .iter()
            .any(|byte| !matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(RusticolError::artifact(format!(
            "{context} SHA-256 must be 64 lowercase hexadecimal characters"
        )));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        digest[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(digest)
}

fn hex_nibble(value: u8) -> u8 {
    match value {
        b'0'..=b'9' => value - b'0',
        b'a'..=b'f' => value - b'a' + 10,
        _ => unreachable!("digest validation precedes hexadecimal decoding"),
    }
}

fn deserialize_required_nullable_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Value::deserialize(deserializer)? {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value)),
        _ => Err(<D::Error as serde::de::Error>::custom(
            "expected a string or null",
        )),
    }
}

fn deserialize_required_nullable_u32<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<u32>::deserialize(deserializer)
}

fn deserialize_recurrence_phase<'de, D>(deserializer: D) -> Result<[f64; 2], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    if let Ok(pair) = serde_json::from_value::<[f64; 2]>(value.clone()) {
        return Ok(pair);
    }
    let object = value.as_object().ok_or_else(|| {
        serde::de::Error::custom("recurrence crossing phase must be a complex-f64 pair")
    })?;
    let component = |prefix: &str| -> Result<f64, D::Error> {
        let numerator = object
            .get(&format!("{prefix}_numerator"))
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("invalid exact recurrence phase"))?
            .parse::<f64>()
            .map_err(|_| serde::de::Error::custom("invalid exact recurrence phase numerator"))?;
        let denominator = object
            .get(&format!("{prefix}_denominator"))
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("invalid exact recurrence phase"))?
            .parse::<f64>()
            .map_err(|_| serde::de::Error::custom("invalid exact recurrence phase denominator"))?;
        let value = numerator / denominator;
        if denominator == 0.0 || !value.is_finite() {
            return Err(serde::de::Error::custom(
                "exact recurrence phase is not finite",
            ));
        }
        Ok(value)
    };
    Ok([component("real")?, component("imag")?])
}

fn deserialize_optional_u64_if_present<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    u64::deserialize(deserializer).map(Some)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn capabilities() -> Value {
        json!([
            RECURRENCE_LC_COLOR_CAPABILITY,
            RECURRENCE_RUNTIME_CAPABILITY
        ])
    }

    fn crossing() -> Value {
        json!({
            "momentum_transform": "identity",
            "helicity_factor": 1,
            "chirality_factor": 1,
            "spin_state_factor": 1,
            "phase": [1.0, 0.0]
        })
    }

    fn source_ir() -> Value {
        json!({
            "identity": {
                "canonical_id": "model:test:state:x",
                "species_id": "model:test:species:x",
                "anti_canonical_id": "model:test:state:x",
                "display_name": "x",
                "anti_display_name": "x",
                "pdg_label": 25,
                "anti_pdg_label": 25,
                "orientation": "self-conjugate",
                "self_conjugate": true
            },
            "statistics": "boson",
            "wavefunction_family": "scalar",
            "component_dimension": 1,
            "states": [{"helicity": 0, "chirality": 0, "spin_state": 0}],
            "crossing": crossing(),
            "basis": "scalar",
            "mass_parameter": null,
            "width_parameter": null
        })
    }

    fn runtime_metadata() -> Value {
        json!({
            "public_color_flows": [{
                "public_id": "flow:1",
                "construction_sector_id": 0,
                "target_sector_id": 0
            }],
            "runtime_parameters": [{
                "name": "mass.x",
                "kind": "external_parameter",
                "parameter_index": 0,
                "default": 1.0,
                "runtime_name": null,
                "complex_component": null
            }],
            "prepared_parameter_defaults": [[1.0, 0.0]],
            "parameter_projection": [{
                "runtime_slot": 0,
                "runtime_name": "mass.x",
                "parameter_template_id": 0,
                "prepared_parameter_id": 0,
                "component": 0
            }],
            "source_templates": [{
                "source_template_id": 0,
                "current_state_template_id": 0,
                "dimension": 1,
                "helicity": 0,
                "chirality": 0,
                "spin_state": 0,
                "source_ir": source_ir(),
                "crossing": crossing()
            }],
            "external_legs": [{
                "source_slot": 0,
                "public_label": 1,
                "physical_pdg": 25,
                "outgoing_pdg": 25,
                "is_initial": false
            }],
            "particle_masses": [{"outgoing_pdg": 25, "mass": 0.0}],
            "color_contraction": null,
            "normalization": {
                "color_accuracy": "lc",
                "color_factor": 1.0,
                "average_factor": 1.0,
                "identical_factor": 1.0,
                "global_coupling_factor": 1.0,
                "qcd_coupling_power": 0,
                "electroweak_coupling_power": 0,
                "couplings_in_stage_evaluators": true,
                "coupling_policy": "test"
            }
        })
    }

    fn inspection_summary() -> Value {
        json!({
            "execution_mode": "recurrence",
            "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
            "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
            "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "process_id": "x_to_x",
            "lc_flow_layout": "topology-replay",
            "parameter_count": 1,
            "sector_count": 1,
            "prepared_kernel_count": 1,
            "direct_executor_count": 4,
            "semantic_digest": "11".repeat(32),
            "schedule_digest": "aa".repeat(32),
            "runtime_layout_digest": "77".repeat(32),
            "schedule": {
                "current_count": 1,
                "source_row_count": 1,
                "contribution_count": 0,
                "finalization_count": 1,
                "amplitude_destination_count": 1,
                "closure_term_count": 0,
                "replay_target_count": 1,
                "retained_helicity_count": 1,
                "resolved_helicity_count": 1,
                "exact_factor_count": 1
            },
            "construction": {
                "peak_current_count": 1,
                "peak_contribution_count": 0,
                "peak_dynamic_color_state_count": 1,
                "color_target_prune_count": 0,
                "candidate_parent_pair_count": 0,
                "peak_to_final_current_ratio": 1.0,
                "peak_to_final_contribution_ratio": 0.0
            },
            "direct_arena": {
                "semantic_component_count": 1,
                "current_arena_components": 1,
                "arena_component_reuse_count": 0,
                "momentum_form_count": 1,
                "selector_domain_count": 1,
                "row_group_count": 4,
                "packed_input_bytes": 0,
                "packed_output_bytes": 0,
                "scatter_bytes": 0
            },
            "runtime_container_member": {
                "path": RECURRENCE_DIRECT_SCHEDULE_MEMBER,
                "size_bytes": 512,
                "sha256": "88".repeat(32),
                "container_size_bytes": 4096
            },
            "generation_timings_seconds": {
                "python_extraction": 0.1,
                "catalog_authentication": 0.1,
                "semantic_construction": 0.1,
                "direct_lowering": 0.1,
                "serialization": 0.1,
                "native_total": 0.5
            }
        })
    }

    pub(crate) fn manifest() -> Value {
        let plan = json!({
            "kind": RECURRENCE_RUNTIME_KIND,
            "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
            "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
            "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
            "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "direct_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
            "builder_input_sha256": "11".repeat(32),
            "prepared_kernel_pack_digest": "44".repeat(32),
            "direct_template_catalog_digest": "55".repeat(32),
            "required_runtime_capabilities": capabilities(),
            "runtime_schedule": {
                "kind": RECURRENCE_RUNTIME_CONTAINER_KIND,
                "schema_version": RECURRENCE_RUNTIME_CONTAINER_SCHEMA,
                "storage_abi": RECURRENCE_RUNTIME_STORAGE_ABI,
                "path": format!(
                    "recurrence/schedules/{}/recurrence-runtime.pacbin",
                    "aa".repeat(32)
                ),
                "plan_member_path": RECURRENCE_DIRECT_SCHEDULE_MEMBER,
                "size_bytes": 4096,
                "sha256": "22".repeat(32),
                "member_count": 1,
                "unpacked_size_bytes": 512,
                "index_sha256": "33".repeat(32)
            },
            "process_binding": {
                "abi": RECURRENCE_PROCESS_BINDING_ABI,
                "process_id": "x_to_x",
                "schedule_digest": "aa".repeat(32),
                "native_schedule_semantic_digest": "ab".repeat(32),
                "process_digest": "12".repeat(32),
                "process_semantic_digest": "11".repeat(32),
                "process_support_words": [1],
                "remap": {
                    "bijection_digest": "66".repeat(32),
                    "source_slots": [0],
                    "source_momentum_signs": [1],
                    "source_helicity_signs": [1],
                    "source_state_offsets": [0, 1],
                    "source_state_indices": [0],
                    "public_flow_ids": [0],
                    "physical_sector_ids": [0],
                    "state_templates": {"count": 1, "changes": []},
                    "source_templates": {"count": 1, "changes": []},
                    "direct_executors": {"count": 4, "changes": []},
                    "parameter_slots": {"count": 1, "changes": []}
                },
                "path": RECURRENCE_PROCESS_BINDING_PATH,
                "size_bytes": 256,
                "sha256": "99".repeat(32)
            },
            "inspection_summary": inspection_summary()
        });
        json!({
            "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
            "kind": RECURRENCE_RUNTIME_KIND,
            "required_runtime_capabilities": capabilities(),
            "process": "x > x",
            "key": "x_to_x",
            "color_accuracy": "lc",
            "external_pdg_order": [25],
            "process_digest": null,
            "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
            "recurrence_plan_abi": RECURRENCE_PLAN_ABI,
            "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
            "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
            "direct_backend_abi": RECURRENCE_DIRECT_BACKEND_ABI,
            "compiled_model_digest": "bb".repeat(32),
            "recurrence_template_catalog_digest": "cc".repeat(32),
            "prepared_kernel_pack_digest": "44".repeat(32),
            "direct_template_catalog_digest": "55".repeat(32),
            "kernel_pack": {
                "manifest_path": RECURRENCE_KERNEL_PACK_MANIFEST_PATH,
                "payload_root": RECURRENCE_KERNEL_PAYLOAD_ROOT
            },
            "runtime_options": {"point_tile_size": 8, "workspace_mib": 64},
            "runtime_metadata": runtime_metadata(),
            "helicity_selector_companion": null,
            "plan": plan,
            "recurrence_summary": {
                "lc_flow_layout": "topology-replay",
                "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
                "template_input_abi": RECURRENCE_TEMPLATE_INPUT_ABI,
                "current_count": 1,
                "contribution_count": 0,
                "closure_term_count": 0
            }
        })
    }

    pub(crate) fn outer() -> ArtifactProcess {
        ArtifactProcess {
            id: "x_to_x".to_owned(),
            expression: "x > x".to_owned(),
            color_accuracy: "lc".to_owned(),
            external_pdgs: vec![25],
            physics_path: "processes/x_to_x/physics.json".to_owned(),
            required_runtime_capabilities: vec![
                RECURRENCE_LC_COLOR_CAPABILITY.to_owned(),
                RECURRENCE_RUNTIME_CAPABILITY.to_owned(),
            ],
            aliases: Vec::new(),
        }
    }

    fn parse(value: &Value) -> RusticolResult<RecurrenceExecutionManifest> {
        parse_recurrence_execution_manifest(
            &serde_json::to_vec(value).unwrap(),
            "processes/x_to_x/execution.json",
            &outer(),
        )
    }

    fn add_python_selector_work_certificate(value: &mut Value) {
        value["plan"]["inspection_summary"]["selector_work_certificate"] = json!({
            "schema_version": 1,
            "binding": "runtime_layout_digest",
            "persisted_union": {
                "current_count": 1,
                "semantic_component_count": 1,
                "source_row_count": 1,
                "contribution_count": 0,
                "finalization_count": 1,
                "closure_count": 0,
                "row_count": 2
            },
            "representatives": [{
                "representative_sector_id": 0,
                "public_flow_count": 1,
                "current_count": 1,
                "semantic_component_count": 1,
                "source_row_count": 1,
                "contribution_count": 0,
                "finalization_count": 1,
                "closure_count": 0,
                "amplitude_destination_count": 1,
                "row_count": 2
            }]
        });
        value["plan"]["inspection_summary"]["direct_arena"]["cache_footprint_policy"] =
            json!("selector-active-max-v1");
    }

    fn add_python_color_projection_certificate(value: &mut Value) {
        value["plan"]["inspection_summary"]["color_projection_certificate"] = json!({
            "path": RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER,
            "schema_version": 1,
            "proof_kind": "exact-rectangular-sum-projection",
            "publishable": true,
            "size_bytes": 128,
            "sha256": "cc".repeat(32)
        });
        value["plan"]["runtime_schedule"]["member_count"] = json!(2);
        value["plan"]["runtime_schedule"]["unpacked_size_bytes"] = json!(640);
    }

    fn add_python_relation_discovery(value: &mut Value, requested_mode: &str) {
        let retained_payload = json!({
            "abi": "pyamplicol-rejected-relation-diagnostics-v1",
            "rows": [],
        });
        let retained_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&retained_payload).unwrap())
        );
        let rejection_decision_sha256 = "aa".repeat(32);
        let full_census = json!({
            "abi": "pyamplicol-relation-discovery-full-census-v1",
            "tested_hypothesis_count": 0,
            "numerical_candidate_count": 0,
            "verification_rejected_count": 0,
            "uncertified_candidate_count": 0,
            "certified_relation_count": 0,
            "rejected_hypothesis_count": 0,
            "applied_relation_count": 0,
            "runtime_parameter_schema_sha256": null,
            "candidate_observation_batch_sha256": null,
            "verification_observation_batch_sha256": null,
            "decision_sha256": null,
            "rejection_decision_sha256": rejection_decision_sha256,
        });
        let full_census_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&full_census).unwrap())
        );
        value["plan"]["inspection_summary"]["relation_discovery"] = json!({
            "schema_version": 1,
            "requested_mode": requested_mode,
            "state": "diagnostic-only",
            "scope": {
                "execution_mode": "recurrence",
                "color_accuracy": "lc",
                "representation": "recurrence-direct-plan-v2",
                "lc_flow_layout": "topology-replay"
            },
            "probe": {
                "status": "completed",
                "precision_digits": 96,
                "probe_count": 4,
                "verification_probe_count": 4,
                "effective_projection_count": 7,
                "seed": 17,
                "deterministic": true,
                "candidate_only": true,
                "tested_hypothesis_count": 0,
                "verification_rejected_count": 0,
                "runtime_parameter_schema_sha256": null,
                "candidate_observation_batch_sha256": null,
                "verification_observation_batch_sha256": null,
                "decision_sha256": null,
                "rejection_decision_sha256": rejection_decision_sha256
            },
            "certificate_replay": {
                "algorithm": relation_certificate_algorithm(),
                "status": "no-certified-relations",
                "certificate_set_sha256": "dd".repeat(32)
            },
            "numerical_candidate_count": 0,
            "uncertified_candidate_count": 0,
            "rejected_hypothesis_count": 0,
            "exact_certified_relation_count": 0,
            "applied_relation_count": 0,
            "interaction_evaluation_count_before": 0,
            "interaction_evaluation_count_after": 0,
            "interaction_evaluation_savings": 0,
            "current_count_before": 1,
            "current_count_after": 1,
            "contribution_count_before": 0,
            "contribution_count_after": 0,
            "scale_copy_row_count": 0,
            "certificates": [],
            "certificate_count": 0,
            "certificates_truncated": false,
            "rejected_candidates": [],
            "rejected_candidate_diagnostics": {
                "total_rejected_hypothesis_count": 0,
                "retained_count": 0,
                "truncated": false,
                "truncation_policy": "first-16-in-canonical-discovery-order-v1",
                "retained_sha256": retained_sha256,
                "full_census_sha256": full_census_sha256,
                "full_rejection_sha256": "aa".repeat(32)
            },
            "follow_up_boundary": "exact relation discovery is opt-in"
        });
    }

    fn hzz_outer(color_accuracy: &str) -> ArtifactProcess {
        ArtifactProcess {
            id: "d_dbar_to_z_z_z".to_owned(),
            expression: "d d~ > z z z".to_owned(),
            color_accuracy: color_accuracy.to_owned(),
            external_pdgs: vec![1, -1, 23, 23, 23],
            physics_path: "processes/d_dbar_to_z_z_z/physics.json".to_owned(),
            required_runtime_capabilities: vec![
                if color_accuracy == "lc" {
                    RECURRENCE_LC_COLOR_CAPABILITY
                } else {
                    RECURRENCE_CONTRACTED_COLOR_CAPABILITY
                }
                .to_owned(),
                RECURRENCE_RUNTIME_CAPABILITY.to_owned(),
            ],
            aliases: Vec::new(),
        }
    }

    fn hzz_companion_outer(color_accuracy: &str) -> ArtifactProcess {
        let mut outer = hzz_outer(color_accuracy);
        outer
            .required_runtime_capabilities
            .push(RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY.to_owned());
        outer.required_runtime_capabilities.sort_unstable();
        outer
    }

    fn parse_hzz_fixture(
        bytes: &[u8],
        color_accuracy: &str,
    ) -> RusticolResult<RecurrenceExecutionManifest> {
        parse_recurrence_execution_manifest(
            bytes,
            "processes/d_dbar_to_z_z_z/execution.json",
            &hzz_outer(color_accuracy),
        )
    }

    fn parse_hzz_companion_fixture(
        bytes: &[u8],
        color_accuracy: &str,
    ) -> RusticolResult<RecurrenceExecutionManifest> {
        parse_recurrence_execution_manifest(
            bytes,
            "processes/d_dbar_to_z_z_z/execution.json",
            &hzz_companion_outer(color_accuracy),
        )
    }

    fn legacy_full_hzz_fixture_value() -> Value {
        let mut value: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/recurrence_execution_hzz_full_v2.json"
        ))
        .unwrap();
        upgrade_hzz_fixture_manifest(&mut value);
        value
    }

    fn full_hzz_fixture_value() -> Value {
        let mut value = legacy_full_hzz_fixture_value();
        add_hzz_contracted_selector_representative(&mut value);
        value
    }

    fn add_hzz_contracted_selector_representative(value: &mut Value) {
        value["plan"]["inspection_summary"]["selector_work_certificate"]["representatives"] = json!([{
            "representative_sector_id": 0,
            "public_flow_count": 1,
            "current_count": 193,
            "semantic_component_count": 431,
            "source_row_count": 13,
            "contribution_count": 450,
            "finalization_count": 126,
            "closure_count": 54,
            "amplitude_destination_count": 54,
            "row_count": 643
        }]);
    }

    fn add_hzz_helicity_selector_companion(value: &mut Value) {
        let process_digest = "10".repeat(32);
        let builder_input_digest = "20".repeat(32);
        let base_schedule_digest = "21".repeat(32);
        let runtime_layout_digest = "22".repeat(32);
        let dispatch_sha256 = "23".repeat(32);
        let composite_schedule_digest =
            recurrence_helicity_selector_schedule_digest(&base_schedule_digest, &dispatch_sha256)
                .unwrap();
        let resolved_helicity_count = 108_u64;

        for pointer in [
            "/required_runtime_capabilities",
            "/plan/required_runtime_capabilities",
        ] {
            let capabilities = value.pointer_mut(pointer).unwrap();
            capabilities.as_array_mut().unwrap().push(json!(
                RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY
            ));
            capabilities
                .as_array_mut()
                .unwrap()
                .sort_by(|left, right| left.as_str().unwrap().cmp(right.as_str().unwrap()));
        }

        value["process_digest"] = json!(process_digest.clone());
        value["plan"]["process_binding"]["process_digest"] = json!(process_digest.clone());
        let mut plan = value["plan"].clone();
        plan["builder_input_sha256"] = json!(builder_input_digest.clone());
        plan["process_binding"]["schedule_digest"] = json!(composite_schedule_digest.clone());
        plan["process_binding"]["native_schedule_semantic_digest"] =
            json!(base_schedule_digest.clone());
        plan["process_binding"]["process_digest"] = json!(process_digest.clone());
        plan["process_binding"]["process_semantic_digest"] = json!(builder_input_digest.clone());
        plan["process_binding"]["path"] = json!(RECURRENCE_HELICITY_SELECTOR_BINDING_PATH);
        plan["process_binding"]["sha256"] = json!("24".repeat(32));
        plan["runtime_schedule"]["path"] = json!(format!(
            "recurrence/schedules/{composite_schedule_digest}/recurrence-runtime.pacbin"
        ));
        plan["runtime_schedule"]["sha256"] = json!("25".repeat(32));
        plan["inspection_summary"]["lc_flow_layout"] = json!("all-flow-union");
        plan["inspection_summary"]["semantic_digest"] = json!(builder_input_digest);
        plan["inspection_summary"]["schedule_digest"] = json!(base_schedule_digest);
        plan["inspection_summary"]["runtime_layout_digest"] = json!(runtime_layout_digest.clone());
        plan["inspection_summary"]["schedule"]["amplitude_destination_count"] = json!(1);
        plan["inspection_summary"]["schedule"]["replay_target_count"] = json!(0);
        plan["inspection_summary"]["schedule"]["retained_helicity_count"] =
            json!(resolved_helicity_count);
        plan["inspection_summary"]["schedule"]["resolved_helicity_count"] =
            json!(resolved_helicity_count);
        plan["inspection_summary"]
            .as_object_mut()
            .unwrap()
            .remove("structural_zero_physical_sector_certificate");
        plan["inspection_summary"]["selector_work_certificate"]["representatives"] = json!([]);
        plan["helicity_dispatch"] = json!({
            "abi": RECURRENCE_HELICITY_DISPATCH_ABI,
            "path": format!(
                "recurrence/schedules/{composite_schedule_digest}/{RECURRENCE_HELICITY_DISPATCH_PATH}"
            ),
            "size_bytes": 96,
            "sha256": dispatch_sha256,
            "base_runtime_layout_digest": runtime_layout_digest,
            "resolved_helicity_count": resolved_helicity_count,
        });
        value["helicity_selector_companion"] = json!({
            "schema_version": 2,
            "kind": RECURRENCE_HELICITY_SELECTOR_COMPANION_KIND,
            "process_digest": process_digest,
            "plan": plan,
            "color_contraction": {
                "source": "primary",
                "view": RECURRENCE_PRIMARY_LOCAL_COLOR_VIEW,
            },
        });
    }

    fn legacy_nlc_hzz_fixture_value() -> Value {
        let mut value: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/recurrence_execution_hzz_nlc.json"
        ))
        .unwrap();
        upgrade_hzz_fixture_manifest(&mut value);
        value
    }

    fn nlc_hzz_fixture_value() -> Value {
        let mut value = legacy_nlc_hzz_fixture_value();
        add_hzz_contracted_selector_representative(&mut value);
        value
    }

    fn lc_hzz_fixture_value() -> Value {
        let mut value: Value = serde_json::from_slice(include_bytes!(
            "../../tests/fixtures/recurrence_execution_hzz_lc.json"
        ))
        .unwrap();
        upgrade_hzz_fixture_manifest(&mut value);
        value
    }

    fn upgrade_hzz_fixture_manifest(value: &mut Value) {
        value["compiled_model_digest"] = json!("bb".repeat(32));
        value["recurrence_template_catalog_digest"] = json!("cc".repeat(32));
        value["plan"]["process_binding"]["abi"] = json!(RECURRENCE_PROCESS_BINDING_ABI);
        value["plan"]["process_binding"]["native_schedule_semantic_digest"] =
            value["plan"]["process_binding"]["schedule_digest"].clone();
        value["plan"]["process_binding"]["process_digest"] = json!("12".repeat(32));
        if value["plan"]["inspection_summary"]["lc_flow_layout"] == json!("contracted-color-union")
        {
            let sector_count = value["plan"]["inspection_summary"]["sector_count"].clone();
            let resolved_helicity_count =
                value["plan"]["inspection_summary"]["schedule"]["resolved_helicity_count"].clone();
            value["plan"]["inspection_summary"]["structural_zero_physical_sector_certificate"] = json!({
                "proof_kind": CLOSURE_TARGET_DOMAIN_PROOF_KIND,
                "semantic_digest": "de".repeat(32),
                "sector_count": sector_count,
                "resolved_helicity_count": resolved_helicity_count,
                "certified_structural_zero_sector_count": 0
            });
        }
    }

    fn manifest_bytes_with_size(size: usize) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(&manifest()).unwrap();
        assert!(bytes.len() <= size);
        bytes.resize(size, b' ');
        bytes
    }

    #[test]
    fn accepts_strict_typed_recurrence_manifest() {
        let parsed = parse(&manifest()).unwrap();
        assert_eq!(parsed.runtime_metadata.public_color_flows.len(), 1);
        assert_eq!(parsed.runtime_metadata.source_templates.len(), 1);
        assert_eq!(parsed.plan.runtime_schedule.member_count, 1);
        assert_eq!(
            parsed.plan.inspection_summary.direct_arena.row_group_count,
            4
        );
    }

    #[test]
    fn accepts_and_validates_generated_structural_zero_sector_provenance() {
        let value = full_hzz_fixture_value();
        let parsed = parse_hzz_fixture(&serde_json::to_vec(&value).unwrap(), "full").unwrap();
        let certificate = parsed
            .plan
            .inspection_summary
            .structural_zero_physical_sector_certificate
            .unwrap();
        assert_eq!(certificate.proof_kind, CLOSURE_TARGET_DOMAIN_PROOF_KIND);
        assert_eq!(certificate.sector_count, 1);
        assert_eq!(certificate.resolved_helicity_count, 54);

        let mut mismatched = value;
        mismatched["plan"]["inspection_summary"]["structural_zero_physical_sector_certificate"]["sector_count"] =
            json!(2);
        let error =
            parse_hzz_fixture(&serde_json::to_vec(&mismatched).unwrap(), "full").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("disagrees with its inspection domain")
        );
    }

    #[test]
    fn accepts_resident_peak_contribution_semantics() {
        let mut value = manifest();
        value["plan"]["inspection_summary"]["construction"]["peak_contribution_count_semantics"] =
            json!("resident-pending-contributions-v1");

        let parsed = parse(&value).unwrap();

        assert_eq!(
            parsed
                .plan
                .inspection_summary
                .construction
                .peak_contribution_count_semantics,
            Some(RecurrencePeakContributionCountSemantics::ResidentPendingContributionsV1)
        );
    }

    #[test]
    fn rejects_unknown_peak_contribution_semantics() {
        let mut value = manifest();
        value["plan"]["inspection_summary"]["construction"]["peak_contribution_count_semantics"] =
            json!("unknown");

        assert!(parse(&value).is_err());
    }

    #[test]
    fn accepts_python_emitted_selector_work_schema() {
        let mut value = manifest();
        add_python_selector_work_certificate(&mut value);

        let parsed = parse(&value).unwrap();

        assert_eq!(
            parsed
                .plan
                .inspection_summary
                .direct_arena
                .cache_footprint_policy,
            Some(RecurrenceCacheFootprintPolicy::SelectorActiveMaxV1)
        );
    }

    #[test]
    fn accepts_contracted_selector_work_representatives() {
        let value = full_hzz_fixture_value();

        let parsed = parse_hzz_fixture(&serde_json::to_vec(&value).unwrap(), "full").unwrap();

        assert_eq!(
            parsed
                .plan
                .inspection_summary
                .selector_work_certificate
                .as_ref()
                .unwrap()
                .representatives
                .len(),
            1
        );
    }

    #[test]
    fn accepts_persisted_helicity_selector_companion_v2() {
        let mut value = full_hzz_fixture_value();
        add_hzz_helicity_selector_companion(&mut value);

        let parsed =
            parse_hzz_companion_fixture(&serde_json::to_vec(&value).unwrap(), "full").unwrap();

        assert_eq!(parsed.process_digest, Some("10".repeat(32)));
        let companion = parsed.helicity_selector_companion.unwrap();
        assert_eq!(companion.schema_version, 2);
        assert_eq!(companion.kind, RECURRENCE_HELICITY_SELECTOR_COMPANION_KIND);
        assert_eq!(
            companion.plan.inspection_summary.lc_flow_layout,
            RecurrenceLcFlowLayout::AllFlowUnion
        );
        assert_eq!(
            companion
                .plan
                .helicity_dispatch
                .as_ref()
                .unwrap()
                .resolved_helicity_count,
            108
        );
        assert_eq!(companion.color_contraction.source, "primary");
        assert_eq!(
            companion.color_contraction.view,
            RECURRENCE_PRIMARY_LOCAL_COLOR_VIEW
        );
    }

    #[test]
    fn helicity_selector_composite_digest_matches_the_python_writer() {
        assert_eq!(
            recurrence_helicity_selector_schedule_digest(&"21".repeat(32), &"23".repeat(32),)
                .unwrap(),
            "5913a2ef2448054b070d57414e621e2a9892676d7ecac59ec161fbf82b3580e7"
        );
    }

    #[test]
    fn rejects_incomplete_or_tampered_helicity_selector_companion_contract() {
        let mut baseline = full_hzz_fixture_value();
        add_hzz_helicity_selector_companion(&mut baseline);

        let mut missing_process_digest = baseline.clone();
        missing_process_digest["process_digest"] = Value::Null;
        assert!(
            parse_hzz_companion_fixture(
                &serde_json::to_vec(&missing_process_digest).unwrap(),
                "full",
            )
            .is_err()
        );

        let mut swapped_primary_process = baseline.clone();
        swapped_primary_process["plan"]["process_binding"]["process_digest"] =
            json!("30".repeat(32));
        assert!(
            parse_hzz_companion_fixture(
                &serde_json::to_vec(&swapped_primary_process).unwrap(),
                "full",
            )
            .is_err()
        );

        let mut missing_capability = baseline.clone();
        missing_capability["required_runtime_capabilities"]
            .as_array_mut()
            .unwrap()
            .retain(|value| {
                value.as_str() != Some(RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY)
            });
        assert!(
            parse_hzz_companion_fixture(&serde_json::to_vec(&missing_capability).unwrap(), "full",)
                .is_err()
        );

        for (pointer, replacement) in [
            ("/helicity_selector_companion/schema_version", json!(1)),
            (
                "/helicity_selector_companion/kind",
                json!("pyamplicol-recurrence-helicity-selector-companion-v1"),
            ),
            (
                "/helicity_selector_companion/process_digest",
                json!("40".repeat(32)),
            ),
            (
                "/helicity_selector_companion/plan/prepared_kernel_pack_digest",
                json!("41".repeat(32)),
            ),
            (
                "/helicity_selector_companion/plan/direct_template_catalog_digest",
                json!("42".repeat(32)),
            ),
            (
                "/helicity_selector_companion/plan/inspection_summary/lc_flow_layout",
                json!("contracted-color-union"),
            ),
            (
                "/helicity_selector_companion/plan/inspection_summary/schedule/amplitude_destination_count",
                json!(2),
            ),
            (
                "/helicity_selector_companion/plan/inspection_summary/schedule/retained_helicity_count",
                json!(107),
            ),
            (
                "/plan/inspection_summary/schedule/retained_helicity_count",
                json!(107),
            ),
            (
                "/helicity_selector_companion/plan/process_binding/native_schedule_semantic_digest",
                json!("43".repeat(32)),
            ),
            (
                "/helicity_selector_companion/plan/process_binding/schedule_digest",
                json!("44".repeat(32)),
            ),
            (
                "/helicity_selector_companion/plan/process_binding/path",
                json!(RECURRENCE_PROCESS_BINDING_PATH),
            ),
            (
                "/helicity_selector_companion/plan/runtime_schedule/path",
                json!("recurrence-runtime.pacbin"),
            ),
            (
                "/helicity_selector_companion/plan/helicity_dispatch/abi",
                json!("pyamplicol-recurrence-helicity-dispatch-v0"),
            ),
            (
                "/helicity_selector_companion/plan/helicity_dispatch/path",
                json!(RECURRENCE_HELICITY_DISPATCH_PATH),
            ),
            (
                "/helicity_selector_companion/plan/helicity_dispatch/size_bytes",
                json!(0),
            ),
            (
                "/helicity_selector_companion/plan/helicity_dispatch/sha256",
                json!("not-a-sha"),
            ),
            (
                "/helicity_selector_companion/plan/helicity_dispatch/base_runtime_layout_digest",
                json!("45".repeat(32)),
            ),
            (
                "/helicity_selector_companion/plan/helicity_dispatch/resolved_helicity_count",
                json!(107),
            ),
            (
                "/helicity_selector_companion/color_contraction/source",
                json!("companion"),
            ),
            (
                "/helicity_selector_companion/color_contraction/view",
                json!("primary-local-color-view-v2"),
            ),
            (
                "/runtime_metadata/color_contraction/storage",
                json!("expanded"),
            ),
        ] {
            let mut invalid = baseline.clone();
            *invalid.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                parse_hzz_companion_fixture(&serde_json::to_vec(&invalid).unwrap(), "full")
                    .is_err(),
                "accepted invalid companion field at {pointer}",
            );
        }
    }

    #[test]
    fn accepts_legacy_contracted_replay_without_selector_work_representatives() {
        for (value, accuracy) in [
            (legacy_full_hzz_fixture_value(), "full"),
            (legacy_nlc_hzz_fixture_value(), "nlc"),
        ] {
            let parsed = parse_hzz_fixture(&serde_json::to_vec(&value).unwrap(), accuracy).unwrap();
            assert!(
                parsed
                    .plan
                    .inspection_summary
                    .selector_work_certificate
                    .as_ref()
                    .unwrap()
                    .representatives
                    .is_empty()
            );
        }
    }

    #[test]
    fn rejects_contracted_selector_work_representatives_without_replay_targets() {
        let mut value = full_hzz_fixture_value();
        value["plan"]["inspection_summary"]["schedule"]["replay_target_count"] = json!(0);

        assert!(
            parse_hzz_fixture(&serde_json::to_vec(&value).unwrap(), "full")
                .unwrap_err()
                .to_string()
                .contains("representatives disagree with the flow layout")
        );
    }

    #[test]
    fn accepts_python_emitted_color_projection_certificate() {
        let mut value = manifest();
        add_python_selector_work_certificate(&mut value);
        add_python_color_projection_certificate(&mut value);

        let parsed = parse(&value).unwrap();

        assert_eq!(parsed.plan.runtime_schedule.member_count, 2);
        assert_eq!(parsed.plan.runtime_schedule.unpacked_size_bytes, 640);
        assert_eq!(
            parsed
                .plan
                .inspection_summary
                .color_projection_certificate
                .as_ref()
                .unwrap()
                .size_bytes,
            128
        );
    }

    #[test]
    fn accepts_python_emitted_relation_discovery_summaries() {
        for mode in ["diagnostic", "certified-reuse"] {
            let mut value = manifest();
            add_python_relation_discovery(&mut value, mode);

            let parsed = parse(&value).unwrap();

            assert_eq!(
                parsed
                    .plan
                    .inspection_summary
                    .relation_discovery
                    .as_ref()
                    .unwrap()
                    .certificate_count,
                0
            );
        }
    }

    #[test]
    fn rejects_relation_discovery_that_mutates_diagnostic_mode() {
        let mut value = manifest();
        add_python_relation_discovery(&mut value, "diagnostic");
        value["plan"]["inspection_summary"]["relation_discovery"]["applied_relation_count"] =
            json!(1);

        assert!(
            parse(&value)
                .unwrap_err()
                .to_string()
                .contains("diagnostic recurrence relation discovery altered")
        );
    }

    #[test]
    fn rejects_tampered_rejected_diagnostics_commitment() {
        let mut value = manifest();
        add_python_relation_discovery(&mut value, "diagnostic");
        value["plan"]["inspection_summary"]["relation_discovery"]["rejected_candidate_diagnostics"]
            ["retained_sha256"] = json!("00".repeat(32));

        assert!(
            parse(&value)
                .unwrap_err()
                .to_string()
                .contains("retained rejection diagnostics commitment")
        );
    }

    #[test]
    fn rejects_tampered_full_rejection_census_and_decision_commitments() {
        for field in ["full_census_sha256", "full_rejection_sha256"] {
            let mut value = manifest();
            add_python_relation_discovery(&mut value, "diagnostic");
            value["plan"]["inspection_summary"]["relation_discovery"]["rejected_candidate_diagnostics"]
                [field] = json!("00".repeat(32));

            assert!(
                parse(&value)
                    .unwrap_err()
                    .to_string()
                    .contains("full rejection census or decision commitment"),
                "tampered {field} was not rejected"
            );
        }
    }

    fn mark_relation_discovery_as_full_probe(value: &mut Value) {
        let discovery = &mut value["plan"]["inspection_summary"]["relation_discovery"];
        discovery["probe"]["candidate_only"] = json!(false);
        for field in [
            "runtime_parameter_schema_sha256",
            "candidate_observation_batch_sha256",
            "verification_observation_batch_sha256",
            "decision_sha256",
        ] {
            discovery["probe"][field] = json!("bb".repeat(32));
        }
    }

    #[test]
    fn rejects_full_probe_with_removed_rejected_diagnostics() {
        let mut value = manifest();
        add_python_relation_discovery(&mut value, "diagnostic");
        mark_relation_discovery_as_full_probe(&mut value);
        value["plan"]["inspection_summary"]["relation_discovery"]
            .as_object_mut()
            .unwrap()
            .remove("rejected_candidate_diagnostics");

        assert!(
            parse(&value)
                .unwrap_err()
                .to_string()
                .contains("require rejected-candidate diagnostics")
        );
    }

    #[test]
    fn rejects_full_probe_with_null_rejected_diagnostics() {
        let mut value = manifest();
        add_python_relation_discovery(&mut value, "diagnostic");
        mark_relation_discovery_as_full_probe(&mut value);
        value["plan"]["inspection_summary"]["relation_discovery"]["rejected_candidate_diagnostics"] =
            Value::Null;

        assert!(
            parse(&value)
                .unwrap_err()
                .to_string()
                .contains("require rejected-candidate diagnostics")
        );
    }

    #[test]
    fn rejects_raw_uncertified_count_that_disagrees_with_verification_failures() {
        let mut value = manifest();
        add_python_relation_discovery(&mut value, "diagnostic");
        mark_relation_discovery_as_full_probe(&mut value);
        let discovery = &mut value["plan"]["inspection_summary"]["relation_discovery"];
        discovery["uncertified_candidate_count"] = json!(1);

        assert!(
            parse(&value)
                .unwrap_err()
                .to_string()
                .contains("authenticated raw rejection census is inconsistent")
        );
    }

    #[test]
    fn accepts_actual_python_emitted_full_lc_and_nlc_manifests() {
        let fixtures = [
            (full_hzz_fixture_value(), "full", Some(54)),
            (lc_hzz_fixture_value(), "lc", None),
            (nlc_hzz_fixture_value(), "nlc", Some(54)),
        ];

        for (value, color_accuracy, expected_materialized_destinations) in fixtures {
            let parsed =
                parse_hzz_fixture(&serde_json::to_vec(&value).unwrap(), color_accuracy).unwrap();
            assert_eq!(
                parsed
                    .runtime_metadata
                    .color_contraction
                    .as_ref()
                    .and_then(|metadata| metadata.materialized_destination_count),
                expected_materialized_destinations
            );
        }
    }

    #[test]
    fn materialized_destination_count_is_backward_compatible_and_validated() {
        let mut value = full_hzz_fixture_value();
        value["runtime_metadata"]["color_contraction"]
            .as_object_mut()
            .unwrap()
            .remove("materialized_destination_count");
        parse_hzz_fixture(&serde_json::to_vec(&value).unwrap(), "full").unwrap();

        value["runtime_metadata"]["color_contraction"]["materialized_destination_count"] =
            json!(53);
        assert!(
            parse_hzz_fixture(&serde_json::to_vec(&value).unwrap(), "full")
                .unwrap_err()
                .to_string()
                .contains("color-contraction summary is inconsistent")
        );
    }

    #[test]
    fn rejects_inconsistent_color_projection_certificate_metadata() {
        let mut bad_members = manifest();
        add_python_selector_work_certificate(&mut bad_members);
        add_python_color_projection_certificate(&mut bad_members);
        bad_members["plan"]["runtime_schedule"]["member_count"] = json!(1);
        assert!(
            parse(&bad_members)
                .unwrap_err()
                .to_string()
                .contains("plan member metadata disagrees")
        );

        let mut bad_unpacked_size = manifest();
        add_python_selector_work_certificate(&mut bad_unpacked_size);
        add_python_color_projection_certificate(&mut bad_unpacked_size);
        bad_unpacked_size["plan"]["runtime_schedule"]["unpacked_size_bytes"] = json!(639);
        assert!(
            parse(&bad_unpacked_size)
                .unwrap_err()
                .to_string()
                .contains("plan member metadata disagrees")
        );

        let mut not_publishable = manifest();
        add_python_selector_work_certificate(&mut not_publishable);
        add_python_color_projection_certificate(&mut not_publishable);
        not_publishable["plan"]["inspection_summary"]["color_projection_certificate"]["publishable"] =
            json!(false);
        assert!(
            parse(&not_publishable)
                .unwrap_err()
                .to_string()
                .contains("metadata is not publishable")
        );

        let mut wrong_layout = manifest();
        add_python_color_projection_certificate(&mut wrong_layout);
        wrong_layout["plan"]["inspection_summary"]["lc_flow_layout"] = json!("all-flow-union");
        wrong_layout["recurrence_summary"]["lc_flow_layout"] = json!("all-flow-union");
        assert!(
            parse(&wrong_layout)
                .unwrap_err()
                .to_string()
                .contains("unsupported recurrence color-projection certificate contract")
        );
    }

    #[test]
    fn accepts_persisted_selector_work_schema() {
        for layout in ["all-flow-union", "contracted-color-union"] {
            let mut value = inspection_summary();
            value["lc_flow_layout"] = json!(layout);
            value["schedule"]["replay_target_count"] = json!(0);
            if layout == "contracted-color-union" {
                value["structural_zero_physical_sector_certificate"] = json!({
                    "proof_kind": CLOSURE_TARGET_DOMAIN_PROOF_KIND,
                    "semantic_digest": "de".repeat(32),
                    "sector_count": 1,
                    "resolved_helicity_count": 1,
                    "certified_structural_zero_sector_count": 0
                });
            }
            value["selector_work_certificate"] = json!({
                "schema_version": 1,
                "binding": "runtime_layout_digest",
                "persisted_union": {
                    "current_count": 1,
                    "semantic_component_count": 1,
                    "source_row_count": 1,
                    "contribution_count": 0,
                    "finalization_count": 1,
                    "closure_count": 0,
                    "row_count": 2
                },
                "representatives": []
            });
            value["direct_arena"]["cache_footprint_policy"] = json!("persisted-arena-v1");
            let parsed = serde_json::from_value::<RecurrenceInspectionSummary>(value).unwrap();

            parsed.validate("x_to_x", "lc").unwrap();
        }
    }

    #[test]
    fn rejects_cache_footprint_policy_for_another_flow_layout() {
        let mut value = manifest();
        add_python_selector_work_certificate(&mut value);
        value["plan"]["inspection_summary"]["direct_arena"]["cache_footprint_policy"] =
            json!("persisted-arena-v1");

        let error = parse(&value).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("cache-footprint policy disagrees with its flow layout")
        );
    }

    #[test]
    fn rejects_standalone_selector_work_schema_member() {
        let mut policy_only = manifest();
        policy_only["plan"]["inspection_summary"]["direct_arena"]["cache_footprint_policy"] =
            json!("selector-active-max-v1");
        assert!(
            parse(&policy_only)
                .unwrap_err()
                .to_string()
                .contains("must be declared together")
        );

        let mut certificate_only = manifest();
        add_python_selector_work_certificate(&mut certificate_only);
        certificate_only["plan"]["inspection_summary"]["direct_arena"]
            .as_object_mut()
            .unwrap()
            .remove("cache_footprint_policy");
        assert!(
            parse(&certificate_only)
                .unwrap_err()
                .to_string()
                .contains("must be declared together")
        );
    }

    #[test]
    fn rejects_inconsistent_selector_work_certificates() {
        let mut wrong_schema = manifest();
        add_python_selector_work_certificate(&mut wrong_schema);
        wrong_schema["plan"]["inspection_summary"]["selector_work_certificate"]["schema_version"] =
            json!(2);
        assert!(
            parse(&wrong_schema)
                .unwrap_err()
                .to_string()
                .contains("certificate identity is unsupported")
        );

        let mut wrong_binding = manifest();
        add_python_selector_work_certificate(&mut wrong_binding);
        wrong_binding["plan"]["inspection_summary"]["selector_work_certificate"]["binding"] =
            json!("semantic_digest");
        assert!(
            parse(&wrong_binding)
                .unwrap_err()
                .to_string()
                .contains("certificate identity is unsupported")
        );

        let mut wrong_union = manifest();
        add_python_selector_work_certificate(&mut wrong_union);
        wrong_union["plan"]["inspection_summary"]["selector_work_certificate"]["persisted_union"]
            ["row_count"] = json!(3);
        assert!(
            parse(&wrong_union)
                .unwrap_err()
                .to_string()
                .contains("persisted selector-work summary disagrees")
        );

        let mut duplicate = manifest();
        add_python_selector_work_certificate(&mut duplicate);
        let representative =
            duplicate["plan"]["inspection_summary"]["selector_work_certificate"]["representatives"]
                [0]
            .clone();
        duplicate["plan"]["inspection_summary"]["selector_work_certificate"]["representatives"]
            .as_array_mut()
            .unwrap()
            .push(representative);
        duplicate["plan"]["inspection_summary"]["schedule"]["replay_target_count"] = json!(2);
        assert!(
            parse(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("representative sector is duplicated")
        );

        let mut out_of_range = manifest();
        add_python_selector_work_certificate(&mut out_of_range);
        out_of_range["plan"]["inspection_summary"]["selector_work_certificate"]["representatives"]
            [0]["representative_sector_id"] = json!(1);
        assert!(
            parse(&out_of_range)
                .unwrap_err()
                .to_string()
                .contains("representative is inconsistent")
        );

        let mut wrong_public_flow_count = manifest();
        add_python_selector_work_certificate(&mut wrong_public_flow_count);
        wrong_public_flow_count["plan"]["inspection_summary"]["selector_work_certificate"]["representatives"]
            [0]["public_flow_count"] = json!(2);
        assert!(
            parse(&wrong_public_flow_count)
                .unwrap_err()
                .to_string()
                .contains("public-flow count disagrees with replay targets")
        );
    }

    #[test]
    fn accepts_observed_large_recurrence_manifest() {
        let bytes = manifest_bytes_with_size(40_826_959);
        let parsed = parse_recurrence_execution_manifest(
            &bytes,
            "processes/x_to_x/execution.json",
            &outer(),
        )
        .unwrap();

        assert_eq!(parsed.key, "x_to_x");
    }

    #[test]
    fn accepts_runtime_options_above_old_reader_ceilings() {
        let mut value = manifest();
        value["runtime_options"]["point_tile_size"] = json!(1_048_577);
        value["runtime_options"]["workspace_mib"] = json!(4097);

        parse(&value).unwrap();

        value["runtime_options"]["workspace_mib"] = json!(0);
        assert!(parse(&value).is_err());

        value["runtime_options"]["workspace_mib"] = json!(u64::from(u32::MAX) + 1);
        assert!(parse(&value).is_err());

        value["runtime_options"]["workspace_mib"] = json!(4097);
        value["runtime_options"]["point_tile_size"] = json!(u64::from(u32::MAX) + 1);
        assert!(parse(&value).is_err());
    }

    #[test]
    fn accepts_large_authenticated_recurrence_payload_metadata() {
        let mut value = manifest();
        add_python_selector_work_certificate(&mut value);
        add_python_color_projection_certificate(&mut value);

        let member_size = 1_u64 << 40;
        let certificate_size = member_size + 1;
        let unpacked_size = member_size + certificate_size;
        let container_size = unpacked_size + 4096;
        value["plan"]["runtime_schedule"]["size_bytes"] = json!(container_size);
        value["plan"]["runtime_schedule"]["unpacked_size_bytes"] = json!(unpacked_size);
        value["plan"]["process_binding"]["size_bytes"] = json!(1_u64 << 30);
        value["plan"]["inspection_summary"]["runtime_container_member"]["size_bytes"] =
            json!(member_size);
        value["plan"]["inspection_summary"]["runtime_container_member"]["container_size_bytes"] =
            json!(container_size);
        value["plan"]["inspection_summary"]["color_projection_certificate"]["size_bytes"] =
            json!(certificate_size);

        let parsed = parse(&value).unwrap();

        assert_eq!(parsed.plan.runtime_schedule.size_bytes, container_size);
        assert_eq!(
            parsed.plan.runtime_schedule.unpacked_size_bytes,
            unpacked_size
        );
        assert_eq!(parsed.plan.process_binding.size_bytes, 1_u64 << 30);
        assert_eq!(
            parsed
                .plan
                .inspection_summary
                .runtime_container_member
                .size_bytes,
            member_size
        );
        assert_eq!(
            parsed
                .plan
                .inspection_summary
                .color_projection_certificate
                .unwrap()
                .size_bytes,
            certificate_size
        );
    }

    #[test]
    fn rejects_zero_recurrence_payload_metadata() {
        let mut baseline = manifest();
        add_python_selector_work_certificate(&mut baseline);
        add_python_color_projection_certificate(&mut baseline);

        for pointer in [
            "/plan/runtime_schedule/size_bytes",
            "/plan/runtime_schedule/unpacked_size_bytes",
            "/plan/process_binding/size_bytes",
            "/plan/inspection_summary/runtime_container_member/size_bytes",
            "/plan/inspection_summary/runtime_container_member/container_size_bytes",
            "/plan/inspection_summary/color_projection_certificate/size_bytes",
        ] {
            let mut value = baseline.clone();
            *value.pointer_mut(pointer).unwrap() = json!(0);
            assert!(
                parse(&value).is_err(),
                "accepted zero metadata at {pointer}"
            );
        }

        let mut color = full_hzz_fixture_value();
        color["runtime_metadata"]["color_contraction"]["size_bytes"] = json!(0);
        assert!(
            parse_hzz_fixture(&serde_json::to_vec(&color).unwrap(), "full").is_err(),
            "accepted zero recurrence color payload size"
        );
    }

    #[test]
    fn accepts_large_recurrence_color_payload_metadata() {
        let mut value = full_hzz_fixture_value();
        value["runtime_metadata"]["color_contraction"]["size_bytes"] = json!(1_u64 << 40);

        let parsed = parse_hzz_fixture(&serde_json::to_vec(&value).unwrap(), "full").unwrap();

        assert_eq!(
            parsed
                .runtime_metadata
                .color_contraction
                .unwrap()
                .size_bytes,
            1_u64 << 40
        );
    }

    #[test]
    fn accepts_large_generated_text_but_still_rejects_empty_text() {
        let mut value = manifest();
        let coupling_policy = "x".repeat(4097);
        value["runtime_metadata"]["normalization"]["coupling_policy"] = json!(coupling_policy);

        let parsed = parse(&value).unwrap();
        assert_eq!(
            parsed.runtime_metadata.normalization.coupling_policy.len(),
            4097
        );

        value["runtime_metadata"]["normalization"]["coupling_policy"] = json!("");
        assert!(
            parse(&value)
                .unwrap_err()
                .to_string()
                .contains("recurrence normalization coupling policy is empty")
        );
    }

    #[test]
    fn rejects_unknown_runtime_metadata_fields() {
        let mut value = manifest();
        value["runtime_metadata"]["opaque"] = json!({});
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("unknown field `opaque`"));
    }

    #[test]
    fn rejects_public_color_flow_outside_construction_sectors() {
        let mut value = manifest();
        value["runtime_metadata"]["public_color_flows"][0]["construction_sector_id"] = json!(1);
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("absent construction sector"));
    }

    #[test]
    fn rejects_noncanonical_public_color_flow_target_sector() {
        let mut value = manifest();
        value["runtime_metadata"]["public_color_flows"][0]["target_sector_id"] = json!(1);
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("not dense or in range"));
    }

    #[test]
    fn rejects_source_ir_and_crossing_disagreement() {
        let mut value = manifest();
        value["runtime_metadata"]["source_templates"][0]["crossing"]["phase"] = json!([0.0, 1.0]);
        let error = parse(&value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("disagrees with its typed SourceIR")
        );
    }

    #[test]
    fn rejects_noncanonical_container_and_capability_metadata() {
        let mut bad_path = manifest();
        bad_path["plan"]["runtime_schedule"]["path"] = json!("../recurrence-runtime.pacbin");
        assert!(
            parse(&bad_path)
                .unwrap_err()
                .to_string()
                .contains("canonical")
        );

        let mut bad_capabilities = manifest();
        bad_capabilities["required_runtime_capabilities"] = json!([
            RECURRENCE_RUNTIME_CAPABILITY,
            RECURRENCE_LC_COLOR_CAPABILITY
        ]);
        assert!(
            parse(&bad_capabilities)
                .unwrap_err()
                .to_string()
                .contains("must require exactly")
        );

        let mut bad_member_path = manifest();
        bad_member_path["plan"]["runtime_schedule"]["plan_member_path"] =
            json!("plan/recurrence-plan-v1.bin");
        assert!(
            parse(&bad_member_path)
                .unwrap_err()
                .to_string()
                .contains("Direct-Arena v2 schedule member path")
        );
    }

    #[test]
    fn rejects_pre_direct_arena_abis() {
        let mut value = manifest();
        value["recurrence_plan_abi"] = json!("pyamplicol-recurrence-plan-v1");
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("Direct-Arena v2 plan ABI"));
        assert!(error.to_string().contains("regenerate"));
    }

    #[test]
    fn rejects_cross_record_direct_digest_mismatches() {
        let mut bad_pack = manifest();
        bad_pack["plan"]["prepared_kernel_pack_digest"] = json!("99".repeat(32));
        assert!(
            parse(&bad_pack)
                .unwrap_err()
                .to_string()
                .contains("prepared-kernel pack digest differs")
        );

        let mut bad_catalog = manifest();
        bad_catalog["plan"]["direct_template_catalog_digest"] = json!("aa".repeat(32));
        assert!(
            parse(&bad_catalog)
                .unwrap_err()
                .to_string()
                .contains("direct-template catalog digest differs")
        );
    }

    #[test]
    fn rejects_malformed_direct_authentication_digests() {
        let mut value = manifest();
        value["direct_template_catalog_digest"] = json!("A5".repeat(32));
        let error = parse(&value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("64 lowercase hexadecimal characters")
        );
    }

    #[test]
    fn rejects_malformed_native_schedule_semantic_digest() {
        let mut value = manifest();
        value["plan"]["process_binding"]["native_schedule_semantic_digest"] =
            json!("AB".repeat(32));
        let error = parse(&value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("64 lowercase hexadecimal characters")
        );
    }

    #[test]
    fn rejects_missing_native_schedule_semantic_digest() {
        let mut value = manifest();
        value["plan"]["process_binding"]
            .as_object_mut()
            .unwrap()
            .remove("native_schedule_semantic_digest");
        let error = parse(&value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("native_schedule_semantic_digest"),
            "{error}"
        );
    }

    #[test]
    fn rejects_inconsistent_direct_container_member_metadata() {
        let mut value = manifest();
        value["plan"]["inspection_summary"]["runtime_container_member"]["container_size_bytes"] =
            json!(4095);
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("plan member metadata disagrees"));
    }

    #[test]
    fn rejects_runtime_metadata_cross_record_mismatches() {
        let mut bad_parameter_count = manifest();
        bad_parameter_count["plan"]["inspection_summary"]["parameter_count"] = json!(2);
        assert!(
            parse(&bad_parameter_count)
                .unwrap_err()
                .to_string()
                .contains("plan parameter count")
        );

        let mut bad_external = manifest();
        bad_external["runtime_metadata"]["external_legs"][0]["physical_pdg"] = json!(24);
        assert!(
            parse(&bad_external)
                .unwrap_err()
                .to_string()
                .contains("physical PDGs")
        );
    }

    #[test]
    fn rejects_nonbijective_source_state_remaps() {
        let mut duplicate = manifest();
        duplicate["plan"]["process_binding"]["remap"]["source_state_offsets"] = json!([0, 2]);
        duplicate["plan"]["process_binding"]["remap"]["source_state_indices"] = json!([0, 0]);
        assert!(
            parse(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("dense zero-based permutation")
        );

        let mut invalid_offsets = manifest();
        invalid_offsets["plan"]["process_binding"]["remap"]["source_state_offsets"] = json!([1, 2]);
        invalid_offsets["plan"]["process_binding"]["remap"]["source_state_indices"] = json!([0, 1]);
        assert!(
            parse(&invalid_offsets)
                .unwrap_err()
                .to_string()
                .contains("offsets are invalid")
        );
    }
}
