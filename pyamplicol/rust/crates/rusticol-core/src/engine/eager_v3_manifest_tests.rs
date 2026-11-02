// SPDX-License-Identifier: 0BSD

use super::eager_v3_manifest::*;
use crate::eager_layout::{
    EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY, EAGER_LOWERING_INPUT_ABI, EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY, EAGER_RUNTIME_CONTAINER_KIND, EAGER_RUNTIME_CONTAINER_SCHEMA,
    EAGER_RUNTIME_LAYOUT_ABI, EAGER_SECTION_HEADER_SIZE, EagerSectionHeader, EagerSectionKind,
};
use crate::pacbin::{PacbinReader, PacbinWriteMember, PacbinWriteOptions, write_pacbin_atomic};
use crate::{ArtifactProcess, PROCESS_ARTIFACT_SCHEMA_VERSION, RusticolErrorKind};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    outer: ArtifactProcess,
    manifest: Value,
}

impl Fixture {
    fn new() -> Self {
        let root = temporary_directory();
        fs::create_dir_all(&root).unwrap();
        let metadata = write_container(&root, None, None);
        let outer = outer_process();
        let manifest = manifest_value(&outer, &metadata);
        Self {
            root,
            outer,
            manifest,
        }
    }

    fn parse(&self) -> crate::RusticolResult<EagerV3ExecutionManifest> {
        parse_eager_v3_execution_manifest(&serde_json::to_vec(&self.manifest).unwrap(), &self.outer)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn open_fixture_runtime_container(
    root: &Path,
    manifest: &EagerV3ExecutionManifest,
) -> crate::RusticolResult<PacbinReader> {
    let path = root.join(EAGER_RUNTIME_CONTAINER_PATH);
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt as _;
        fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&path)
    };
    #[cfg(not(unix))]
    let file = fs::File::open(&path);
    let file = file.map_err(|error| {
        crate::RusticolError::security(format!(
            "could not open eager runtime container {}: {error}",
            path.display()
        ))
    })?;
    open_eager_v3_runtime_container(file, &path, manifest)
}

#[derive(Clone)]
struct ContainerMetadata {
    size_bytes: u64,
    sha256: String,
    member_count: u64,
    unpacked_size_bytes: u64,
    index_sha256: String,
}

#[test]
fn valid_manifest_and_container_preflight() {
    let fixture = Fixture::new();
    let manifest = fixture.parse().unwrap();
    let reader = open_fixture_runtime_container(&fixture.root, &manifest).unwrap();
    assert_eq!(reader.members().len(), EXPECTED_EAGER_MEMBERS.len());
}

#[cfg(unix)]
#[test]
fn opened_runtime_container_does_not_reopen_a_replaced_path() {
    let fixture = Fixture::new();
    let manifest = fixture.parse().unwrap();
    let path = fixture.root.join(EAGER_RUNTIME_CONTAINER_PATH);
    let displaced = fixture.root.join("authenticated-runtime.pacbin");
    let file = fs::File::open(&path).unwrap();
    fs::rename(&path, &displaced).unwrap();
    fs::write(&path, b"replacement bytes").unwrap();

    let reader = open_eager_v3_runtime_container(file, &path, &manifest).unwrap();
    assert_eq!(reader.members().len(), EXPECTED_EAGER_MEMBERS.len());
}

#[test]
fn outer_process_identity_must_match() {
    for (field, value) in [
        ("schema_version", json!(4)),
        ("kind", json!("wrong-kind")),
        ("process", json!("g g > g g")),
        ("key", json!("wrong-key")),
        ("color_accuracy", json!("lc")),
        ("external_pdg_order", json!([21, 21, 21, 21])),
    ] {
        let mut fixture = Fixture::new();
        fixture.manifest[field] = value;
        let error = fixture.parse().unwrap_err();
        assert!(
            matches!(
                error.kind(),
                RusticolErrorKind::Compatibility | RusticolErrorKind::Integrity
            ),
            "field {field}: {error}"
        );
    }
}

#[test]
fn every_plan_abi_and_capability_mismatch_fails_closed() {
    let mutations: &[(&[&str], Value)] = &[
        (&["eager_plan_abi"], json!("pyamplicol-eager-plan-v999")),
        (
            &["plan", "eager_plan_abi"],
            json!("pyamplicol-eager-plan-v999"),
        ),
        (
            &["plan", "lowering_input_abi"],
            json!("pyamplicol-eager-lowering-input-v999"),
        ),
        (
            &["plan", "runtime_layout_abi"],
            json!("pyamplicol-eager-runtime-layout-v999"),
        ),
        (
            &["required_runtime_capabilities"],
            json!(["rusticol.eager-dag.complex-f64.v1"]),
        ),
        (
            &["plan", "required_runtime_capabilities"],
            json!(["rusticol.eager-runtime-layout.complex-f64.v2"]),
        ),
    ];
    for (path, replacement) in mutations {
        let mut fixture = Fixture::new();
        set_value(&mut fixture.manifest, path, replacement.clone());
        let error = fixture.parse().unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility, "{path:?}");
    }

    let mut fixture = Fixture::new();
    fixture.outer.required_runtime_capabilities = vec!["wrong-capability".to_string()];
    assert_eq!(
        fixture.parse().unwrap_err().kind(),
        RusticolErrorKind::Compatibility
    );
}

#[test]
fn pre_arena_plan_v3_has_actionable_regeneration_error() {
    let mut fixture = Fixture::new();
    let legacy = json!([EAGER_RUNTIME_CAPABILITY]);
    fixture.manifest["required_runtime_capabilities"] = legacy.clone();
    fixture.manifest["plan"]["required_runtime_capabilities"] = legacy;
    fixture.outer.required_runtime_capabilities = vec![EAGER_RUNTIME_CAPABILITY.to_string()];
    let error = fixture.parse().unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("predates"));
    assert!(error.to_string().contains("pyamplicol generate"));
}

#[test]
fn container_contract_mismatches_fail_closed() {
    let mutations: &[(&[&str], Value)] = &[
        (
            &["plan", "runtime_container", "kind"],
            json!("wrong-container"),
        ),
        (&["plan", "runtime_container", "schema_version"], json!(2)),
        (
            &["plan", "runtime_container", "storage_abi"],
            json!("pacbin-v2"),
        ),
        (
            &["plan", "runtime_container", "path"],
            json!("other.pacbin"),
        ),
        (&["plan", "runtime_container", "size_bytes"], json!(0)),
        (
            &["plan", "runtime_container", "sha256"],
            json!("A".repeat(64)),
        ),
        (&["plan", "runtime_container", "member_count"], json!(44)),
        (
            &["plan", "runtime_container", "unpacked_size_bytes"],
            json!(0),
        ),
        (
            &["plan", "runtime_container", "unpacked_size_bytes"],
            json!(u64::MAX),
        ),
        (
            &["plan", "runtime_container", "index_sha256"],
            json!("A".repeat(64)),
        ),
    ];
    for (path, replacement) in mutations {
        let mut fixture = Fixture::new();
        set_value(&mut fixture.manifest, path, replacement.clone());
        assert!(
            fixture.parse().is_err(),
            "mutation unexpectedly passed: {path:?}"
        );
    }
}

#[test]
fn large_writer_representable_container_metadata_is_accepted() {
    let mut fixture = Fixture::new();
    let size_bytes = 1_u64 << 40;
    fixture.manifest["plan"]["runtime_container"]["size_bytes"] = json!(size_bytes);
    fixture.manifest["plan"]["runtime_container"]["unpacked_size_bytes"] = json!(size_bytes - 4096);

    fixture.parse().unwrap();
}

#[test]
fn long_writer_valid_model_name_is_accepted() {
    let mut fixture = Fixture::new();
    fixture.manifest["plan"]["inspection_summary"]["model_name"] =
        json!("m".repeat(1024 * 1024 + 1));

    fixture.parse().unwrap();

    fixture.manifest["plan"]["inspection_summary"]["model_name"] = json!("");
    assert!(fixture.parse().is_err());
}

#[test]
fn runtime_options_require_positive_values_and_summary_relationships_hold() {
    let mutations: &[(&[&str], Value)] = &[
        (&["runtime_options", "point_tile_size"], json!(0)),
        (&["runtime_options", "workspace_mib"], json!(0)),
        (&["dag_summary", "interaction_count"], json!(1_u64 << 49)),
        (
            &["dag_summary", "interaction_evaluation_count"],
            json!(1_u64 << 49),
        ),
        (
            &["plan", "inspection_summary", "invocation_count"],
            json!(1_u64 << 49),
        ),
        (
            &[
                "plan",
                "inspection_summary",
                "selector_work",
                "selected_flow_current_count",
            ],
            json!(12),
        ),
        (
            &[
                "plan",
                "inspection_summary",
                "selector_work",
                "contracted_attachment_count",
            ],
            json!(18),
        ),
    ];
    for (path, replacement) in mutations {
        let mut fixture = Fixture::new();
        set_value(&mut fixture.manifest, path, replacement.clone());
        assert!(
            fixture.parse().is_err(),
            "mutation unexpectedly passed: {path:?}"
        );
    }
}

#[test]
fn runtime_options_accept_writer_valid_values_above_old_reader_ceilings() {
    let mut fixture = Fixture::new();
    fixture.manifest["runtime_options"]["point_tile_size"] = json!(1_048_577);
    fixture.manifest["runtime_options"]["workspace_mib"] = json!(4097);

    fixture.parse().unwrap();
}

#[test]
fn interaction_evaluation_count_matches_the_persisted_plan() {
    let mut fixture = Fixture::new();
    fixture.manifest["dag_summary"]["interaction_evaluation_count"] = json!(24);
    let error = fixture.parse().unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("exceeds"));

    let mut fixture = Fixture::new();
    fixture.manifest["dag_summary"]["interaction_evaluation_count"] = json!(16);
    let error = fixture.parse().unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("disagree"));
}

#[test]
fn materialization_census_matches_the_persisted_plan() {
    for (path, replacement) in [
        (
            &["plan", "materialization_census", "basis"][..],
            json!("dynamic"),
        ),
        (
            &[
                "plan",
                "materialization_census",
                "final",
                "invocation_count",
            ][..],
            json!(16),
        ),
        (
            &["plan", "materialization_census", "final_equals_peak"][..],
            json!(false),
        ),
    ] {
        let mut fixture = Fixture::new();
        set_value(&mut fixture.manifest, path, replacement);
        assert!(fixture.parse().is_err(), "{path:?}");
    }
}

#[test]
fn traversal_and_unknown_fields_are_rejected_during_manifest_parse() {
    let mut fixture = Fixture::new();
    fixture.manifest["plan"]["runtime_container"]["path"] = json!("../eager-runtime.pacbin");
    assert_eq!(
        fixture.parse().unwrap_err().kind(),
        RusticolErrorKind::Security
    );

    let mut fixture = Fixture::new();
    fixture.manifest["plan"]["runtime_container"]["extra"] = json!(1);
    assert_eq!(
        fixture.parse().unwrap_err().kind(),
        RusticolErrorKind::Serialization
    );
}

#[cfg(unix)]
#[test]
fn symlinked_runtime_container_is_rejected() {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new();
    let manifest = fixture.parse().unwrap();
    let path = fixture.root.join(EAGER_RUNTIME_CONTAINER_PATH);
    let target = fixture.root.join("target.pacbin");
    fs::rename(&path, &target).unwrap();
    symlink(&target, &path).unwrap();
    let error = open_fixture_runtime_container(&fixture.root, &manifest).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Security);
}

#[test]
fn payload_digest_mismatch_is_rejected_before_container_use() {
    let mut fixture = Fixture::new();
    fixture.manifest["plan"]["runtime_container"]["sha256"] = json!("0".repeat(64));
    let manifest = fixture.parse().unwrap();
    let error = open_fixture_runtime_container(&fixture.root, &manifest).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("payload digest mismatch"));
}

#[test]
fn index_and_unpacked_metadata_must_match_container() {
    for (field, value) in [
        ("index_sha256", json!("0".repeat(64))),
        ("unpacked_size_bytes", json!(1)),
    ] {
        let mut fixture = Fixture::new();
        fixture.manifest["plan"]["runtime_container"][field] = value;
        let manifest = fixture.parse().unwrap();
        let error = open_fixture_runtime_container(&fixture.root, &manifest).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Integrity, "{field}");
    }

    let mut fixture = Fixture::new();
    let declared = fixture.manifest["plan"]["runtime_container"]["size_bytes"]
        .as_u64()
        .unwrap();
    fixture.manifest["plan"]["runtime_container"]["size_bytes"] = json!(declared + 1);
    let manifest = fixture.parse().unwrap();
    assert_eq!(
        open_fixture_runtime_container(&fixture.root, &manifest)
            .unwrap_err()
            .kind(),
        RusticolErrorKind::Integrity
    );
}

#[test]
fn incorrect_section_kind_is_rejected() {
    let mut fixture = Fixture::new();
    let metadata = write_container_with_section_override(
        &fixture.root,
        "tables/values.bin",
        EagerSectionKind::CurrentLayout,
    );
    fixture.manifest = manifest_value(&fixture.outer, &metadata);
    let manifest = fixture.parse().unwrap();
    let error = open_fixture_runtime_container(&fixture.root, &manifest).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("section kind"));
}

#[test]
fn unexpected_member_path_is_rejected() {
    let mut fixture = Fixture::new();
    let metadata = write_container(
        &fixture.root,
        Some(("tables/values.bin", "tables/unexpected.bin")),
        None,
    );
    fixture.manifest = manifest_value(&fixture.outer, &metadata);
    let manifest = fixture.parse().unwrap();
    let error = open_fixture_runtime_container(&fixture.root, &manifest).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Security);
    assert!(error.to_string().contains("unexpected"));
}

#[test]
fn incorrect_member_kind_is_rejected() {
    let mut fixture = Fixture::new();
    let metadata = write_container(
        &fixture.root,
        None,
        Some((
            "tables/values.bin",
            crate::pacbin::PacbinMemberKind::EagerRuntimeMetadata,
        )),
    );
    fixture.manifest = manifest_value(&fixture.outer, &metadata);
    let manifest = fixture.parse().unwrap();
    let error = open_fixture_runtime_container(&fixture.root, &manifest).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("has kind"));
}

#[test]
fn legacy_plan_v2_reports_regeneration_before_large_manifest_parse() {
    let outer = outer_process();
    let mut bytes = br#"{"eager_plan_abi":"pyamplicol-eager-plan-v2","kind":"pyamplicol-runtime-eager-execution","runtime_schema":"#
        .to_vec();
    bytes.extend(std::iter::repeat_n(b'x', 2 * 1024 * 1024));
    bytes.extend_from_slice(br#""}"#);
    let error = parse_eager_v3_execution_manifest(&bytes, &outer).unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("regenerate"));
    assert!(error.to_string().contains("plan-v2"));
}

fn outer_process() -> ArtifactProcess {
    ArtifactProcess {
        id: "d_dbar_to_z_g".to_string(),
        expression: "d d~ > z g".to_string(),
        color_accuracy: "full".to_string(),
        external_pdgs: vec![1, -1, 23, 21],
        physics_path: "physics.json".to_string(),
        required_runtime_capabilities: direct_capabilities(),
        aliases: Vec::new(),
    }
}

fn manifest_value(outer: &ArtifactProcess, metadata: &ContainerMetadata) -> Value {
    let capabilities = json!(direct_capabilities());
    json!({
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": EAGER_EXECUTION_KIND,
        "required_runtime_capabilities": capabilities,
        "process": outer.expression,
        "key": outer.id,
        "color_accuracy": outer.color_accuracy,
        "external_pdg_order": outer.external_pdgs,
        "eager_plan_abi": EAGER_PLAN_ABI,
        "kernel_pack": {
            "manifest_path": EAGER_KERNEL_PACK_MANIFEST_PATH,
            "payload_root": EAGER_KERNEL_PAYLOAD_ROOT,
        },
        "runtime_options": {"point_tile_size": 1024, "workspace_mib": 256},
        "plan": {
            "kind": EAGER_EXECUTION_KIND,
            "eager_plan_abi": EAGER_PLAN_ABI,
            "lowering_input_abi": EAGER_LOWERING_INPUT_ABI,
            "lowering_input_sha256": "1".repeat(64),
            "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
            "required_runtime_capabilities": capabilities,
            "runtime_container": {
                "kind": EAGER_RUNTIME_CONTAINER_KIND,
                "schema_version": EAGER_RUNTIME_CONTAINER_SCHEMA,
                "storage_abi": EAGER_RUNTIME_STORAGE_ABI,
                "path": EAGER_RUNTIME_CONTAINER_PATH,
                "size_bytes": metadata.size_bytes,
                "sha256": metadata.sha256,
                "member_count": metadata.member_count,
                "unpacked_size_bytes": metadata.unpacked_size_bytes,
                "index_sha256": metadata.index_sha256,
            },
            "inspection_summary": inspection_summary(outer),
            "materialization_census": materialization_census(),
        },
        "dag_summary": {
            "current_count": 11,
            "source_count": 4,
            "interaction_count": 23,
            "interaction_evaluation_count": 17,
            "amplitude_root_count": 5,
            "truncated": false,
        },
    })
}

fn materialization_census() -> Value {
    let counts = json!({
        "source_count": 4,
        "current_count": 11,
        "interaction_count": 23,
        "amplitude_root_count": 5,
        "invocation_count": 17,
        "attachment_count": 19,
        "closure_count": 5,
        "finalization_count": 11,
    });
    json!({
        "abi": "pyamplicol-fully-resident-materialization-census-v1",
        "basis": "immutable-fully-resident-eager-plan",
        "final": counts.clone(),
        "peak": counts,
        "final_equals_peak": true,
    })
}

fn direct_capabilities() -> Vec<String> {
    vec![
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY.to_string(),
        EAGER_RUNTIME_CAPABILITY.to_string(),
    ]
}

fn inspection_summary(outer: &ArtifactProcess) -> Value {
    json!({
        "execution_mode": "eager",
        "eager_plan_abi": EAGER_PLAN_ABI,
        "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
        "process_id": outer.id,
        "model_name": "built-in-sm",
        "current_count": 11,
        "value_count": 12,
        "momentum_count": 13,
        "source_count": 4,
        "parameter_count": 5,
        "stage_count": 3,
        "coupling_count": 7,
        "invocation_count": 17,
        "attachment_count": 19,
        "finalization_count": 11,
        "closure_count": 5,
        "selector_domain_count": 2,
        "reduction_group_count": 3,
        "reduction_entry_count": 9,
        "retained_table_count": 21,
        "current_component_count": 31,
        "value_component_count": 37,
        "momentum_component_count": 41,
        "selector_work": {
            "abi": "pyamplicol-eager-selector-work-v1",
            "selected_flow_selector_count": 3,
            "selected_flow_current_count": 7,
            "selected_flow_evaluation_count": 9,
            "selected_flow_attachment_count": 9,
            "all_flow_selector_count": 5,
            "all_flow_current_count": 8,
            "all_flow_evaluation_count": 10,
            "all_flow_attachment_count": 10,
            "contracted_current_count": 11,
            "contracted_evaluation_count": 17,
            "contracted_attachment_count": 19,
        },
    })
}

fn write_container(
    root: &Path,
    path_override: Option<(&str, &str)>,
    kind_override: Option<(&str, crate::pacbin::PacbinMemberKind)>,
) -> ContainerMetadata {
    let mut payloads = Vec::with_capacity(EXPECTED_EAGER_MEMBERS.len());
    for expected in EXPECTED_EAGER_MEMBERS {
        payloads.push(
            EagerSectionHeader::new(expected.section_kind, 1, 0)
                .unwrap()
                .encode()
                .to_vec(),
        );
    }
    assert!(
        payloads
            .iter()
            .all(|payload| payload.len() == EAGER_SECTION_HEADER_SIZE)
    );
    let mut members = Vec::with_capacity(EXPECTED_EAGER_MEMBERS.len());
    for (expected, payload) in EXPECTED_EAGER_MEMBERS.iter().zip(&payloads) {
        let path = path_override
            .filter(|(original, _)| *original == expected.path)
            .map_or(expected.path, |(_, replacement)| replacement);
        let kind = kind_override
            .filter(|(original, _)| *original == expected.path)
            .map_or(expected.member_kind, |(_, replacement)| replacement);
        members.push(PacbinWriteMember::from_bytes(path, kind, payload).unwrap());
    }
    let path = root.join(EAGER_RUNTIME_CONTAINER_PATH);
    let index = write_pacbin_atomic(&path, members, PacbinWriteOptions::default()).unwrap();
    let bytes = fs::read(&path).unwrap();
    ContainerMetadata {
        size_bytes: bytes.len() as u64,
        sha256: hex_sha256(&bytes),
        member_count: index.members().len() as u64,
        unpacked_size_bytes: index.members().iter().map(|member| member.length()).sum(),
        index_sha256: hex_bytes(index.index_sha256()),
    }
}

fn write_container_with_section_override(
    root: &Path,
    override_path: &str,
    override_kind: EagerSectionKind,
) -> ContainerMetadata {
    let mut payloads = Vec::with_capacity(EXPECTED_EAGER_MEMBERS.len());
    for expected in EXPECTED_EAGER_MEMBERS {
        let section_kind = if expected.path == override_path {
            override_kind
        } else {
            expected.section_kind
        };
        payloads.push(
            EagerSectionHeader::new(section_kind, 1, 0)
                .unwrap()
                .encode()
                .to_vec(),
        );
    }
    let members: Vec<_> = EXPECTED_EAGER_MEMBERS
        .iter()
        .zip(&payloads)
        .map(|(expected, payload)| {
            PacbinWriteMember::from_bytes(expected.path, expected.member_kind, payload).unwrap()
        })
        .collect();
    let path = root.join(EAGER_RUNTIME_CONTAINER_PATH);
    let index = write_pacbin_atomic(&path, members, PacbinWriteOptions::default()).unwrap();
    let bytes = fs::read(&path).unwrap();
    ContainerMetadata {
        size_bytes: bytes.len() as u64,
        sha256: hex_sha256(&bytes),
        member_count: index.members().len() as u64,
        unpacked_size_bytes: index.members().iter().map(|member| member.length()).sum(),
        index_sha256: hex_bytes(index.index_sha256()),
    }
}

fn set_value(root: &mut Value, path: &[&str], replacement: Value) {
    let (last, parents) = path.split_last().unwrap();
    let mut current = root;
    for name in parents {
        current = &mut current[*name];
    }
    current[*last] = replacement;
}

fn hex_sha256(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn temporary_directory() -> PathBuf {
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rusticol-eager-v3-manifest-{}-{sequence}",
        std::process::id()
    ))
}
