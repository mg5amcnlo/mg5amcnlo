// SPDX-License-Identifier: 0BSD

//! Bounded execution-manifest and container preflight for eager plan-v3.
//!
//! This module deliberately stops before decoding an `EagerExecutionPlan`.
//! It authenticates the small JSON summary and the complete PACBIN container,
//! then returns the mapped reader to the compact runtime loader.

use crate::eager_layout::{
    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY, EAGER_LOWERING_INPUT_ABI, EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY, EAGER_RUNTIME_CONTAINER_KIND, EAGER_RUNTIME_CONTAINER_SCHEMA,
    EAGER_RUNTIME_LAYOUT_ABI, EagerSectionHeader, EagerSectionKind,
};
use crate::pacbin::{PacbinMemberKind, PacbinReader};
use crate::{ArtifactProcess, PROCESS_ARTIFACT_SCHEMA_VERSION, RusticolError, RusticolResult};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;

pub(super) const EAGER_EXECUTION_KIND: &str = "pyamplicol-runtime-eager-execution";
pub(super) const EAGER_RUNTIME_STORAGE_ABI: &str = "pacbin-v1";
pub(super) const EAGER_RUNTIME_CONTAINER_PATH: &str = "eager-runtime.pacbin";
pub(super) const EAGER_KERNEL_PACK_MANIFEST_PATH: &str = "model/eager-kernel-pack.json";
pub(super) const EAGER_KERNEL_PAYLOAD_ROOT: &str = "model/eager-kernels";
const LEGACY_EAGER_PLAN_ABI: &str = "pyamplicol-eager-plan-v2";
const LEGACY_EAGER_RUNTIME_CAPABILITY: &str = "rusticol.eager-dag.complex-f64.v1";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3ExecutionManifest {
    pub(super) schema_version: u32,
    pub(super) kind: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) eager_plan_abi: String,
    pub(super) kernel_pack: EagerV3KernelPackReference,
    pub(super) runtime_options: EagerV3RuntimeOptions,
    pub(super) plan: EagerV3PlanSummary,
    pub(super) dag_summary: EagerV3DagSummary,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3KernelPackReference {
    pub(super) manifest_path: String,
    pub(super) payload_root: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3RuntimeOptions {
    pub(super) point_tile_size: u64,
    pub(super) workspace_mib: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3PlanSummary {
    pub(super) kind: String,
    pub(super) eager_plan_abi: String,
    pub(super) lowering_input_abi: String,
    pub(super) lowering_input_sha256: String,
    pub(super) runtime_layout_abi: String,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) runtime_container: EagerV3RuntimeContainer,
    pub(super) inspection_summary: EagerV3InspectionSummary,
    pub(super) materialization_census: EagerV3MaterializationCensus,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3RuntimeContainer {
    pub(super) kind: String,
    pub(super) schema_version: u16,
    pub(super) storage_abi: String,
    pub(super) path: String,
    pub(super) size_bytes: u64,
    pub(super) sha256: String,
    pub(super) member_count: u64,
    pub(super) unpacked_size_bytes: u64,
    pub(super) index_sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3DagSummary {
    pub(super) current_count: u64,
    pub(super) source_count: u64,
    pub(super) interaction_count: u64,
    pub(super) interaction_evaluation_count: u64,
    pub(super) amplitude_root_count: u64,
    pub(super) truncated: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3MaterializationCensus {
    pub(super) abi: String,
    pub(super) basis: String,
    pub(super) r#final: BTreeMap<String, u64>,
    pub(super) peak: BTreeMap<String, u64>,
    pub(super) final_equals_peak: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3InspectionSummary {
    pub(super) execution_mode: String,
    pub(super) eager_plan_abi: String,
    pub(super) runtime_layout_abi: String,
    pub(super) process_id: String,
    pub(super) model_name: String,
    pub(super) current_count: u64,
    pub(super) value_count: u64,
    pub(super) momentum_count: u64,
    pub(super) source_count: u64,
    pub(super) parameter_count: u64,
    pub(super) stage_count: u64,
    pub(super) coupling_count: u64,
    pub(super) invocation_count: u64,
    pub(super) attachment_count: u64,
    pub(super) finalization_count: u64,
    pub(super) closure_count: u64,
    pub(super) selector_domain_count: u64,
    pub(super) reduction_group_count: u64,
    pub(super) reduction_entry_count: u64,
    pub(super) retained_table_count: u64,
    pub(super) current_component_count: u64,
    pub(super) value_component_count: u64,
    pub(super) momentum_component_count: u64,
    #[serde(default)]
    pub(super) selector_work: Option<EagerV3SelectorWork>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EagerV3SelectorWork {
    pub(super) abi: String,
    pub(super) selected_flow_selector_count: u64,
    pub(super) selected_flow_current_count: u64,
    pub(super) selected_flow_evaluation_count: u64,
    pub(super) selected_flow_attachment_count: u64,
    pub(super) all_flow_selector_count: u64,
    pub(super) all_flow_current_count: u64,
    pub(super) all_flow_evaluation_count: u64,
    pub(super) all_flow_attachment_count: u64,
    pub(super) contracted_current_count: u64,
    pub(super) contracted_evaluation_count: u64,
    pub(super) contracted_attachment_count: u64,
}

impl EagerV3ExecutionManifest {
    fn validate(&self, outer: &ArtifactProcess) -> RusticolResult<()> {
        if self.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
            || self.kind != EAGER_EXECUTION_KIND
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager execution kind {:?} schema {}; regenerate the artifact",
                self.kind, self.schema_version
            )));
        }
        if self.process != outer.expression
            || self.key != outer.id
            || self.color_accuracy != outer.color_accuracy
            || self.external_pdg_order != outer.external_pdgs
        {
            return Err(RusticolError::integrity(format!(
                "eager plan-v3 execution manifest does not match outer process {:?}",
                outer.id
            )));
        }
        validate_direct_capabilities(
            &self.required_runtime_capabilities,
            "eager execution manifest",
        )?;
        validate_direct_capabilities(&outer.required_runtime_capabilities, "outer eager process")?;
        if self.eager_plan_abi != EAGER_PLAN_ABI {
            return Err(unsupported_abi("eager plan", &self.eager_plan_abi));
        }
        if self.kernel_pack.manifest_path != EAGER_KERNEL_PACK_MANIFEST_PATH
            || self.kernel_pack.payload_root != EAGER_KERNEL_PAYLOAD_ROOT
        {
            return Err(RusticolError::security(
                "eager plan-v3 kernel-pack paths are not canonical",
            ));
        }
        self.runtime_options.validate()?;
        self.plan.validate(&self.key)?;
        self.dag_summary.validate()?;
        self.plan
            .materialization_census
            .validate(&self.plan.inspection_summary, &self.dag_summary)?;
        if self.plan.inspection_summary.current_count != self.dag_summary.current_count
            || self.plan.inspection_summary.source_count != self.dag_summary.source_count
            || self.plan.inspection_summary.invocation_count
                != self.dag_summary.interaction_evaluation_count
        {
            return Err(RusticolError::integrity(
                "eager plan-v3 inspection counts disagree with the DAG summary",
            ));
        }
        Ok(())
    }
}

impl EagerV3RuntimeOptions {
    fn validate(self) -> RusticolResult<()> {
        if self.point_tile_size == 0 {
            return Err(RusticolError::artifact(
                "eager point_tile_size must be positive",
            ));
        }
        if self.workspace_mib == 0 {
            return Err(RusticolError::artifact(
                "eager workspace_mib must be positive",
            ));
        }
        Ok(())
    }
}

impl EagerV3PlanSummary {
    fn validate(&self, process_key: &str) -> RusticolResult<()> {
        if self.kind != EAGER_EXECUTION_KIND {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager plan kind {:?}",
                self.kind
            )));
        }
        if self.eager_plan_abi != EAGER_PLAN_ABI {
            return Err(unsupported_abi("nested eager plan", &self.eager_plan_abi));
        }
        if self.lowering_input_abi != EAGER_LOWERING_INPUT_ABI {
            return Err(unsupported_abi(
                "eager lowering input",
                &self.lowering_input_abi,
            ));
        }
        if self.runtime_layout_abi != EAGER_RUNTIME_LAYOUT_ABI {
            return Err(unsupported_abi(
                "eager runtime layout",
                &self.runtime_layout_abi,
            ));
        }
        parse_sha256(&self.lowering_input_sha256, "eager lowering input")?;
        validate_direct_capabilities(&self.required_runtime_capabilities, "nested eager plan")?;
        self.runtime_container.validate()?;
        self.inspection_summary.validate(process_key)?;
        Ok(())
    }
}

impl EagerV3RuntimeContainer {
    fn validate(&self) -> RusticolResult<()> {
        if self.kind != EAGER_RUNTIME_CONTAINER_KIND {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager runtime container kind {:?}",
                self.kind
            )));
        }
        if self.schema_version != EAGER_RUNTIME_CONTAINER_SCHEMA {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager runtime container schema {}",
                self.schema_version
            )));
        }
        if self.storage_abi != EAGER_RUNTIME_STORAGE_ABI {
            return Err(unsupported_abi("eager runtime storage", &self.storage_abi));
        }
        if self.path != EAGER_RUNTIME_CONTAINER_PATH {
            return Err(RusticolError::security(
                "eager runtime container path must be exactly eager-runtime.pacbin",
            ));
        }
        if self.member_count != EXPECTED_EAGER_MEMBERS.len() as u64 {
            return Err(RusticolError::integrity(format!(
                "eager runtime container must declare exactly {} members",
                EXPECTED_EAGER_MEMBERS.len()
            )));
        }
        if self.size_bytes == 0 {
            return Err(RusticolError::artifact(
                "eager runtime container must not be empty",
            ));
        }
        if self.unpacked_size_bytes == 0 || self.unpacked_size_bytes > self.size_bytes {
            return Err(RusticolError::artifact(
                "eager runtime unpacked size must be nonzero and not exceed the container size",
            ));
        }
        parse_sha256(&self.sha256, "eager runtime container")?;
        parse_sha256(&self.index_sha256, "eager runtime index")?;
        Ok(())
    }
}

impl EagerV3DagSummary {
    fn validate(self) -> RusticolResult<()> {
        validate_counts(
            "eager DAG summary",
            &[
                self.current_count,
                self.source_count,
                self.interaction_count,
                self.interaction_evaluation_count,
                self.amplitude_root_count,
            ],
        )?;
        if self.interaction_evaluation_count > self.interaction_count {
            return Err(RusticolError::integrity(
                "eager DAG interaction-evaluation count exceeds its interaction count",
            ));
        }
        if self.truncated {
            return Err(RusticolError::compatibility(
                "truncated DAGs cannot be loaded as eager plan-v3 artifacts",
            ));
        }
        Ok(())
    }
}

impl EagerV3MaterializationCensus {
    fn validate(
        &self,
        inspection: &EagerV3InspectionSummary,
        dag: &EagerV3DagSummary,
    ) -> RusticolResult<()> {
        if self.abi != "pyamplicol-fully-resident-materialization-census-v1"
            || self.basis != "immutable-fully-resident-eager-plan"
        {
            return Err(RusticolError::compatibility(
                "eager materialization census ABI or basis is unsupported",
            ));
        }
        let expected = BTreeMap::from([
            ("amplitude_root_count".to_string(), dag.amplitude_root_count),
            ("attachment_count".to_string(), inspection.attachment_count),
            ("closure_count".to_string(), inspection.closure_count),
            ("current_count".to_string(), dag.current_count),
            (
                "finalization_count".to_string(),
                inspection.finalization_count,
            ),
            ("interaction_count".to_string(), dag.interaction_count),
            ("invocation_count".to_string(), inspection.invocation_count),
            ("source_count".to_string(), dag.source_count),
        ]);
        validate_counts(
            "eager materialization census",
            &self.r#final.values().copied().collect::<Vec<_>>(),
        )?;
        validate_counts(
            "eager materialization census peak",
            &self.peak.values().copied().collect::<Vec<_>>(),
        )?;
        if !self.final_equals_peak || self.r#final != expected || self.peak != expected {
            return Err(RusticolError::integrity(
                "eager materialization census disagrees with the persisted plan",
            ));
        }
        Ok(())
    }
}

impl EagerV3InspectionSummary {
    fn validate(&self, process_key: &str) -> RusticolResult<()> {
        if self.execution_mode != "eager"
            || self.eager_plan_abi != EAGER_PLAN_ABI
            || self.runtime_layout_abi != EAGER_RUNTIME_LAYOUT_ABI
        {
            return Err(RusticolError::compatibility(
                "eager inspection summary ABI does not match plan-v3",
            ));
        }
        if self.process_id != process_key {
            return Err(RusticolError::integrity(
                "eager inspection process id does not match its execution manifest",
            ));
        }
        if self.model_name.is_empty() {
            return Err(RusticolError::artifact(
                "eager inspection model name must not be empty",
            ));
        }
        validate_counts(
            "eager inspection summary",
            &[
                self.current_count,
                self.value_count,
                self.momentum_count,
                self.source_count,
                self.parameter_count,
                self.stage_count,
                self.coupling_count,
                self.invocation_count,
                self.attachment_count,
                self.finalization_count,
                self.closure_count,
                self.selector_domain_count,
                self.reduction_group_count,
                self.reduction_entry_count,
                self.retained_table_count,
                self.current_component_count,
                self.value_component_count,
                self.momentum_component_count,
            ],
        )?;
        if let Some(selector_work) = &self.selector_work {
            selector_work.validate(self)?;
        }
        Ok(())
    }
}

impl EagerV3SelectorWork {
    fn validate(&self, summary: &EagerV3InspectionSummary) -> RusticolResult<()> {
        if self.abi != "pyamplicol-eager-selector-work-v1" {
            return Err(RusticolError::compatibility(
                "eager selector-work summary ABI is unsupported",
            ));
        }
        validate_counts(
            "eager selector-work summary",
            &[
                self.selected_flow_selector_count,
                self.selected_flow_current_count,
                self.selected_flow_evaluation_count,
                self.selected_flow_attachment_count,
                self.all_flow_selector_count,
                self.all_flow_current_count,
                self.all_flow_evaluation_count,
                self.all_flow_attachment_count,
                self.contracted_current_count,
                self.contracted_evaluation_count,
                self.contracted_attachment_count,
            ],
        )?;
        if self.selected_flow_current_count > summary.current_count
            || self.all_flow_current_count > summary.current_count
            || self.contracted_current_count != summary.current_count
            || self.selected_flow_evaluation_count > summary.invocation_count
            || self.all_flow_evaluation_count > summary.invocation_count
            || self.contracted_evaluation_count != summary.invocation_count
            || self.selected_flow_attachment_count > summary.attachment_count
            || self.all_flow_attachment_count > summary.attachment_count
            || self.contracted_attachment_count != summary.attachment_count
        {
            return Err(RusticolError::integrity(
                "eager selector-work counts exceed or disagree with the persisted plan",
            ));
        }
        Ok(())
    }
}

/// Parse and validate one eager plan-v3 `execution.json`.
///
/// Legacy plan-v2 markers are detected before full
/// JSON deserialization, avoiding traversal of an expanded `runtime_schema`.
pub(super) fn parse_eager_v3_execution_manifest(
    bytes: &[u8],
    outer: &ArtifactProcess,
) -> RusticolResult<EagerV3ExecutionManifest> {
    reject_legacy_eager_manifest(bytes)?;
    let manifest: EagerV3ExecutionManifest = serde_json::from_slice(bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse eager plan-v3 execution manifest: {error}"
        ))
    })?;
    manifest.validate(outer)?;
    Ok(manifest)
}

/// Open, authenticate, and structurally preflight `eager-runtime.pacbin`.
pub(super) fn open_eager_v3_runtime_container(
    container_file: File,
    display_path: &Path,
    manifest: &EagerV3ExecutionManifest,
) -> RusticolResult<PacbinReader> {
    manifest.plan.runtime_container.validate()?;
    let container = &manifest.plan.runtime_container;
    // A complete container digest authenticates every indexed member. Hashing
    // and parsing use the exact checked file description supplied by the
    // verified artifact, avoiding a path-replacement race.
    // PACBIN still validates its index, canonical layout, and boundaries.
    let expected_file_sha = parse_sha256(&container.sha256, "eager runtime container")?;
    let reader =
        PacbinReader::open_file_with_sha256(container_file, display_path, &expected_file_sha)?;
    let index = reader.index();
    let mapped_size = u64::try_from(reader.container_size())
        .map_err(|_| RusticolError::integrity("eager runtime PACBIN size exceeds u64"))?;
    if index.file_size() != container.size_bytes || mapped_size != container.size_bytes {
        return Err(RusticolError::integrity(
            "eager runtime PACBIN size does not match execution manifest",
        ));
    }
    let expected_index_sha = parse_sha256(&container.index_sha256, "eager runtime index")?;
    if index.index_sha256() != &expected_index_sha {
        return Err(RusticolError::integrity(
            "eager runtime PACBIN index digest mismatch",
        ));
    }
    if index.members().len() as u64 != container.member_count {
        return Err(RusticolError::integrity(
            "eager runtime PACBIN member count does not match execution manifest",
        ));
    }

    let mut seen = BTreeSet::new();
    let mut unpacked_size = 0_u64;
    for member in reader.members() {
        let expected = expected_member(member.logical_path()).ok_or_else(|| {
            RusticolError::security(format!(
                "unexpected eager runtime PACBIN member path {:?}",
                member.logical_path()
            ))
        })?;
        if member.kind() != expected.member_kind {
            return Err(RusticolError::integrity(format!(
                "eager runtime PACBIN member {:?} has kind {:?}, expected {:?}",
                member.logical_path(),
                member.kind(),
                expected.member_kind
            )));
        }
        let (header, _) = EagerSectionHeader::decode(reader.member_bytes(member.logical_path())?)?;
        if header.kind() != expected.section_kind {
            return Err(RusticolError::integrity(format!(
                "eager runtime PACBIN member {:?} has section kind {:?}, expected {:?}",
                member.logical_path(),
                header.kind(),
                expected.section_kind
            )));
        }
        unpacked_size = unpacked_size.checked_add(member.length()).ok_or_else(|| {
            RusticolError::integrity("eager runtime PACBIN unpacked size exceeds u64")
        })?;
        seen.insert(member.logical_path());
    }
    if seen.len() != EXPECTED_EAGER_MEMBERS.len()
        || EXPECTED_EAGER_MEMBERS
            .iter()
            .any(|expected| !seen.contains(expected.path))
    {
        return Err(RusticolError::integrity(
            "eager runtime PACBIN is missing a required member",
        ));
    }
    if unpacked_size != container.unpacked_size_bytes {
        return Err(RusticolError::integrity(
            "eager runtime PACBIN unpacked size does not match execution manifest",
        ));
    }
    Ok(reader)
}

fn reject_legacy_eager_manifest(bytes: &[u8]) -> RusticolResult<()> {
    if contains_bytes(bytes, LEGACY_EAGER_PLAN_ABI.as_bytes())
        || contains_bytes(bytes, LEGACY_EAGER_RUNTIME_CAPABILITY.as_bytes())
    {
        return Err(RusticolError::compatibility(
            "legacy eager plan-v2 artifacts are unsupported by the compact eager runtime; regenerate the artifact with `pyamplicol generate`",
        ));
    }
    Ok(())
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn validate_direct_capabilities(capabilities: &[String], context: &str) -> RusticolResult<()> {
    if capabilities == [EAGER_RUNTIME_CAPABILITY] {
        return Err(RusticolError::compatibility(format!(
            "{context} predates {EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY:?}; regenerate the eager \
             artifact with the current `pyamplicol generate` before loading it"
        )));
    }
    if capabilities
        != [
            EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
            EAGER_RUNTIME_CAPABILITY,
        ]
    {
        return Err(RusticolError::compatibility(format!(
            "{context} must require exactly {:?}",
            [
                EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
                EAGER_RUNTIME_CAPABILITY,
            ]
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

#[derive(Clone, Copy)]
pub(super) struct ExpectedEagerMember {
    pub(super) path: &'static str,
    pub(super) member_kind: PacbinMemberKind,
    pub(super) section_kind: EagerSectionKind,
}

const fn metadata(path: &'static str, section_kind: EagerSectionKind) -> ExpectedEagerMember {
    ExpectedEagerMember {
        path,
        member_kind: PacbinMemberKind::EagerRuntimeMetadata,
        section_kind,
    }
}

const fn table(path: &'static str, section_kind: EagerSectionKind) -> ExpectedEagerMember {
    ExpectedEagerMember {
        path,
        member_kind: PacbinMemberKind::EagerRuntimeTable,
        section_kind,
    }
}

pub(super) const EXPECTED_EAGER_MEMBERS: &[ExpectedEagerMember] = &[
    table(
        "catalogs/bitsets/populations.bin",
        EagerSectionKind::Metadata,
    ),
    table("catalogs/bitsets/ranges.bin", EagerSectionKind::Metadata),
    table("catalogs/bitsets/words.bin", EagerSectionKind::Metadata),
    table("catalogs/exact-factors.bin", EagerSectionKind::ExactFactors),
    metadata("catalogs/exact-ir/bytes.bin", EagerSectionKind::Metadata),
    metadata("catalogs/exact-ir/ranges.bin", EagerSectionKind::Metadata),
    table(
        "catalogs/i32-sequences/ranges.bin",
        EagerSectionKind::Metadata,
    ),
    table(
        "catalogs/i32-sequences/values.bin",
        EagerSectionKind::Metadata,
    ),
    metadata(
        "catalogs/semantic-limitations/bytes.bin",
        EagerSectionKind::Metadata,
    ),
    metadata(
        "catalogs/semantic-limitations/ranges.bin",
        EagerSectionKind::Metadata,
    ),
    metadata("catalogs/strings/bytes.bin", EagerSectionKind::Metadata),
    metadata("catalogs/strings/ranges.bin", EagerSectionKind::Metadata),
    table(
        "catalogs/u32-sequences/ranges.bin",
        EagerSectionKind::Metadata,
    ),
    table(
        "catalogs/u32-sequences/values.bin",
        EagerSectionKind::Metadata,
    ),
    metadata(
        "inspection/summary.bin",
        EagerSectionKind::InspectionSummary,
    ),
    metadata("metadata/core.bin", EagerSectionKind::Metadata),
    metadata("metadata/identity-bytes.bin", EagerSectionKind::Metadata),
    metadata("metadata/identity-ranges.bin", EagerSectionKind::Metadata),
    table("reductions/entries.bin", EagerSectionKind::ReductionEntries),
    table("reductions/groups.bin", EagerSectionKind::ReductionGroups),
    table("retained/columns.bin", EagerSectionKind::Metadata),
    metadata("retained/name-bytes.bin", EagerSectionKind::Metadata),
    metadata("retained/name-ranges.bin", EagerSectionKind::Metadata),
    table("retained/tables.bin", EagerSectionKind::Metadata),
    table("retained/values-f64-bits.bin", EagerSectionKind::Metadata),
    table("retained/values-i32.bin", EagerSectionKind::Metadata),
    table("retained/values-u32.bin", EagerSectionKind::Metadata),
    table("retained/values-u64.bin", EagerSectionKind::Metadata),
    table("retained/values-u8.bin", EagerSectionKind::Metadata),
    table("selectors/colors.bin", EagerSectionKind::SelectorDomains),
    table("selectors/domains.bin", EagerSectionKind::SelectorDomains),
    table(
        "selectors/helicities.bin",
        EagerSectionKind::SelectorDomains,
    ),
    table(
        "selectors/memberships.bin",
        EagerSectionKind::SelectorMemberships,
    ),
    table("tables/attachments.bin", EagerSectionKind::Attachments),
    table("tables/closures.bin", EagerSectionKind::Closures),
    table("tables/couplings.bin", EagerSectionKind::Couplings),
    table("tables/currents.bin", EagerSectionKind::CurrentLayout),
    table("tables/direct-coefficients.bin", EagerSectionKind::Closures),
    table("tables/finalizations.bin", EagerSectionKind::Finalizations),
    table("tables/invocations.bin", EagerSectionKind::Invocations),
    table("tables/momenta.bin", EagerSectionKind::MomentumLayout),
    table("tables/parameters.bin", EagerSectionKind::ParameterLayout),
    table("tables/sources.bin", EagerSectionKind::SourceFill),
    table("tables/stages.bin", EagerSectionKind::Stages),
    table("tables/values.bin", EagerSectionKind::ValueLayout),
];

fn expected_member(path: &str) -> Option<&'static ExpectedEagerMember> {
    EXPECTED_EAGER_MEMBERS
        .binary_search_by_key(&path, |member| member.path)
        .ok()
        .map(|index| &EXPECTED_EAGER_MEMBERS[index])
}
