// SPDX-License-Identifier: 0BSD

use crate::pacbin::{PacbinFileReader, PacbinMemberKind, PacbinReader};
use crate::{
    ARTIFACT_MANIFEST_FILE, C_ABI_VERSION, COMPILED_MODEL_SCHEMA_VERSION,
    PROCESS_ARTIFACT_SCHEMA_VERSION, PYTHON_API_VERSION, RUNTIME_PHYSICS_SCHEMA_VERSION,
    RuntimeCapability, RusticolError, RusticolResult, TOML_SCHEMA_VERSION,
};
use bincode::Decode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "f64-compiled")]
use std::sync::Mutex;
#[cfg(feature = "f64-compiled")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "f64-compiled")]
use std::time::{SystemTime, UNIX_EPOCH};

const ARTIFACT_IDENTITY_EXTENSION: &str = "artifact_identity";
const ARTIFACT_IDENTITY_KIND: &str = "pyamplicol-runtime-payload-identity";
const ARTIFACT_IDENTITY_SCHEMA_VERSION: u64 = 1;
#[cfg(test)]
const RUNTIME_IDENTITY_PAYLOAD_ROLES: [&str; 5] = [
    "compiled-model",
    "evaluator-manifest",
    "evaluator-state",
    "model-parameters",
    "runtime-physics",
];
const EVALUATOR_PAYLOAD_CONTAINER_EXTENSION: &str = "evaluator_payload_container";
const EVALUATOR_PAYLOAD_CONTAINER_KIND: &str = "pyamplicol-evaluator-payload-container";
const EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI: &str = "pacbin-v1";
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
const RECURRENCE_BOOTSTRAP_FILENAME: &str = "recurrence-bootstrap.bin";
/// Holds declared payloads that are not stored as loose files.
pub const PAYLOAD_CONTAINER_PATH: &str = "payloads.pacbin";
const SUPPORTED_ARTIFACT_TARGETS: [&str; 3] = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
];

pub(crate) const PORTABLE_64LE_ARTIFACT_TARGET: &str = "portable-64le";
#[cfg(feature = "f64-compiled")]
const NATIVE_LIBRARY_SNAPSHOT_ATTEMPTS: u64 = 128;
#[cfg(feature = "f64-compiled")]
static NEXT_NATIVE_LIBRARY_SNAPSHOT: AtomicU64 = AtomicU64::new(0);

fn rusticol_package_version() -> &'static str {
    option_env!("PYAMPLICOL_PACKAGE_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactKind {
    PyamplicolProcess,
    PyamplicolProcessSet,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(deny_unknown_fields)]
pub struct VersionSet {
    pub python_api: u32,
    pub toml: u32,
    pub compiled_model: u32,
    pub process_artifact: u32,
    pub runtime_physics: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbolica_serialization: Option<String>,
    pub c_abi: u32,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub triple: String,
    pub cpu_features: Vec<String>,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    pub distribution: String,
    pub version: String,
    pub versions: VersionSet,
    pub target: Target,
    #[serde(default)]
    pub git_revision: Option<String>,
    #[serde(default)]
    pub native_build_inputs_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelSourceKind {
    BuiltInSm,
    BuiltInSmHeft,
    Ufo,
    UfoJson,
    CompiledModel,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactModel {
    pub name: String,
    pub source_kind: ModelSourceKind,
    pub content_sha256: String,
    pub compiled_schema_version: u32,
    #[serde(default)]
    pub restriction: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigurationAdjustment {
    pub path: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactConfiguration {
    pub toml_schema_version: u32,
    pub requested_path: String,
    pub effective_path: String,
    pub adjustments: Vec<ConfigurationAdjustment>,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(deny_unknown_fields)]
pub struct ProcessAlias {
    pub id: String,
    pub expression: String,
    pub external_pdgs: Vec<i32>,
    pub external_permutation: Vec<usize>,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(deny_unknown_fields)]
pub struct ArtifactProcess {
    pub id: String,
    pub expression: String,
    pub color_accuracy: String,
    pub external_pdgs: Vec<i32>,
    pub physics_path: String,
    pub required_runtime_capabilities: Vec<String>,
    pub aliases: Vec<ProcessAlias>,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRuntime {
    pub engine: String,
    pub engine_version: String,
    pub evaluator_manifest_path: String,
    pub required_runtime_capabilities: Vec<String>,
    #[serde(default)]
    pub api_bundle_path: Option<String>,
}

#[derive(Clone, Copy, Debug, Decode, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(rename_all = "kebab-case")]
pub enum PayloadRole {
    ConfigurationRequested,
    ConfigurationEffective,
    CompiledModel,
    RuntimePhysics,
    EvaluatorManifest,
    EvaluatorState,
    ModelParameters,
    ValidationMomenta,
    StructuralSourceProof,
    ApiSource,
    ApiBuildFile,
    SdkMetadata,
    ProcessGroups,
    PayloadContainer,
}

#[derive(Clone, Debug, Decode, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub path: String,
    pub role: PayloadRole,
    pub media_type: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub executable: bool,
    #[serde(default)]
    pub target: Option<Target>,
    #[serde(default)]
    pub process_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub source: String,
    pub license: String,
    #[serde(default)]
    pub content_sha256: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub patch_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub kind: ArtifactKind,
    pub artifact_id: String,
    pub created_utc: String,
    pub producer: Producer,
    pub model: ArtifactModel,
    pub configuration: ArtifactConfiguration,
    pub processes: Vec<ArtifactProcess>,
    #[serde(default)]
    pub default_process_id: Option<String>,
    pub runtime: ArtifactRuntime,
    pub payloads: Vec<Payload>,
    pub dependencies: Vec<Dependency>,
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactSelection {
    pub process: ArtifactProcess,
    pub requested_id: String,
    pub alias: Option<ProcessAlias>,
    /// Canonical public process expression selected by the caller.
    pub public_expression: String,
    /// Public external-particle order selected by the caller.
    pub external_pdgs: Vec<i32>,
    /// Representative external index to public external index.
    pub external_permutation: Vec<usize>,
    /// Whether the permutation was inferred from a concrete expression rather
    /// than persisted as an explicit artifact alias.
    pub inferred_permutation: bool,
    /// Present when the caller selected a process that was not generated and
    /// is routed through `process-groups.json` to `process`.
    pub process_group: Option<ProcessGroupRelation>,
}

impl ArtifactSelection {
    /// Whether public metadata differs from the stored representative.
    pub fn remaps_public_process(&self) -> bool {
        self.alias.is_some() || self.inferred_permutation || self.process_group.is_some()
    }
}

/// Relation of a selected process-group member to its generated representative:
/// `ME_member(p) = numerator / denominator * ME_representative(q)` with
/// `q[r] = p[external_permutation[r]]`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessGroupRelation {
    pub factor_numerator: u64,
    pub factor_denominator: u64,
    /// The relation includes global charge conjugation: public LC color-flow
    /// words are the reversed words of the permuted representative.
    pub conjugated: bool,
    /// Member particle names in the public external order.
    pub external_particles: Vec<String>,
}

impl ProcessGroupRelation {
    pub fn has_unit_factor(&self) -> bool {
        self.factor_numerator == self.factor_denominator
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedArtifact {
    root: PathBuf,
    manifest_path: PathBuf,
    manifest: ArtifactManifest,
    payloads: Arc<BTreeMap<String, Payload>>,
    evaluator_payload_container: Option<Arc<PacbinReader>>,
    payload_container: Option<Arc<PacbinFileReader>>,
    #[cfg(feature = "f64-compiled")]
    native_library_cache: Arc<Mutex<BTreeMap<String, Arc<PinnedNativeLibrary>>>>,
}

/// Authenticated inputs for the narrow recurrence-only loading path.
///
/// The returned `VerifiedArtifact` owns the same checked payload inventory and
/// evaluator container used by the generic loader.  Only the broad evaluator
/// set and generation-oriented extension trees are skipped.
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub(crate) struct RecurrenceBootstrapArtifact {
    pub(crate) artifact: VerifiedArtifact,
    pub(crate) selection: ArtifactSelection,
    pub(crate) ready_execution: crate::engine::RecurrenceReadyExecutionV1,
    pub(crate) physics: crate::ProcessPhysics,
    pub(crate) evaluator_payload_container: EvaluatorPayloadContainerExtension,
}

#[derive(Clone, Debug, Decode, Deserialize)]
#[cfg_attr(
    any(feature = "python-generation-bridge", test),
    derive(bincode::Encode)
)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluatorPayloadContainerExtension {
    pub(crate) kind: String,
    pub(crate) schema_version: u32,
    pub(crate) storage_abi: String,
    pub(crate) path: String,
    pub(crate) member_count: u64,
    pub(crate) unpacked_size_bytes: u64,
    pub(crate) index_sha256: String,
}

/// One evaluator payload resolved from a legacy loose file or a packed member.
#[derive(Clone, Debug)]
pub(crate) enum EvaluatorPayloadSource {
    File(PathBuf),
    AuthenticatedFile {
        root: PathBuf,
        payload: Payload,
    },
    Packed {
        container: Arc<PacbinReader>,
        logical_path: String,
    },
    /// A declared payload stored in the artifact's payload container.
    PackedDeclared {
        container: Arc<PacbinFileReader>,
        payload: Payload,
    },
}

/// A dynamic library loaded only from process-private snapshot bytes.
///
/// The original artifact path is retained solely for diagnostics. Symbol
/// lookup always addresses `library`, which was opened from an authenticated
/// private materialization that is unlinked immediately on Unix.
pub(crate) struct PinnedNativeLibrary {
    #[cfg(feature = "f64-compiled")]
    library: Option<libloading::Library>,
    display_path: PathBuf,
    #[cfg(feature = "f64-compiled")]
    temporary: Option<TemporaryNativeLibrary>,
}

impl std::fmt::Debug for PinnedNativeLibrary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnedNativeLibrary")
            .field("display_path", &self.display_path)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "f64-compiled")]
impl std::ops::Deref for PinnedNativeLibrary {
    type Target = libloading::Library;

    fn deref(&self) -> &Self::Target {
        self.library
            .as_ref()
            .expect("pinned native library is live until Drop")
    }
}

impl PinnedNativeLibrary {
    #[cfg(feature = "f64-compiled")]
    fn from_bytes(bytes: &[u8], display_path: PathBuf) -> RusticolResult<Arc<Self>> {
        let mut failures = Vec::new();
        for root in native_library_snapshot_roots() {
            let temporary = match TemporaryNativeLibrary::create(bytes, &display_path, &root) {
                Ok(temporary) => temporary,
                Err(error) => {
                    failures.push(format!("{}: {error}", root.display()));
                    continue;
                }
            };
            let library = match unsafe { libloading::Library::new(&temporary.path) } {
                Ok(library) => library,
                Err(error) => {
                    failures.push(format!("{}: {error}", root.display()));
                    continue;
                }
            };
            #[cfg(unix)]
            let temporary = {
                temporary.remove_now()?;
                None
            };
            #[cfg(not(unix))]
            let temporary = Some(temporary);
            return Ok(Arc::new(Self {
                library: Some(library),
                display_path,
                temporary,
            }));
        }
        Err(RusticolError::evaluation(format!(
            "could not load authenticated native evaluator library {}; snapshot roots failed: {}. \
             The snapshot filesystem must be writable and executable (not mounted noexec); set \
             PYAMPLICOL_NATIVE_SNAPSHOT_ROOT to a private executable filesystem",
            display_path.display(),
            failures.join("; ")
        )))
    }

    pub(crate) fn display_path(&self) -> &Path {
        &self.display_path
    }

    #[cfg(all(test, feature = "f64-compiled"))]
    pub(crate) fn from_test_path(path: &Path) -> RusticolResult<Arc<Self>> {
        let bytes = fs::read(path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not read native-library test fixture {}: {error}",
                path.display()
            ))
        })?;
        Self::from_bytes(&bytes, path.to_path_buf())
    }
}

#[cfg(feature = "f64-compiled")]
impl Drop for PinnedNativeLibrary {
    fn drop(&mut self) {
        // Windows does not permit removal while the library is loaded. Drop
        // the handle first, then let the retained materialization guard clean
        // up. Unix snapshots are already unlinked after successful dlopen.
        drop(self.library.take());
        drop(self.temporary.take());
    }
}

#[cfg(feature = "f64-compiled")]
struct TemporaryNativeLibrary {
    directory: PathBuf,
    path: PathBuf,
    active: bool,
}

#[cfg(feature = "f64-compiled")]
impl TemporaryNativeLibrary {
    fn create(bytes: &[u8], display_path: &Path, root: &Path) -> RusticolResult<Self> {
        if !root.is_dir() {
            return Err(RusticolError::artifact(format!(
                "native-library snapshot root {} is not a directory",
                root.display()
            )));
        }
        let suffix = display_path
            .extension()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("bin");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        for _ in 0..NATIVE_LIBRARY_SNAPSHOT_ATTEMPTS {
            let sequence = NEXT_NATIVE_LIBRARY_SNAPSHOT.fetch_add(1, Ordering::Relaxed);
            let directory = root.join(format!(
                ".pyamplicol-native-{}-{now}-{sequence}",
                std::process::id()
            ));
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt as _;
                builder.mode(0o700);
            }
            match builder.create(&directory) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(RusticolError::artifact(format!(
                        "could not create private native-library snapshot directory {}: {error}",
                        directory.display()
                    )));
                }
            }
            let path = directory.join(format!("payload.{suffix}"));
            let result = (|| {
                let mut options = OpenOptions::new();
                options.create_new(true).write(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt as _;
                    options.mode(0o600);
                }
                let mut file = options.open(&path).map_err(|error| {
                    RusticolError::artifact(format!(
                        "could not create native-library snapshot {}: {error}",
                        path.display()
                    ))
                })?;
                file.write_all(bytes).map_err(|error| {
                    RusticolError::artifact(format!(
                        "could not write native-library snapshot {}: {error}",
                        path.display()
                    ))
                })?;
                file.flush().map_err(|error| {
                    RusticolError::artifact(format!(
                        "could not flush native-library snapshot {}: {error}",
                        path.display()
                    ))
                })?;
                drop(file);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt as _;
                    fs::set_permissions(&path, fs::Permissions::from_mode(0o500)).map_err(
                        |error| {
                            RusticolError::artifact(format!(
                                "could not protect native-library snapshot {}: {error}",
                                path.display()
                            ))
                        },
                    )?;
                }
                Ok(Self {
                    directory: directory.clone(),
                    path: path.clone(),
                    active: true,
                })
            })();
            if result.is_err() {
                let _ = fs::remove_file(&path);
                let _ = fs::remove_dir(&directory);
            }
            return result;
        }
        Err(RusticolError::artifact(
            "could not allocate a unique private native-library snapshot",
        ))
    }

    #[cfg(unix)]
    fn remove_now(mut self) -> RusticolResult<()> {
        fs::remove_file(&self.path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not unlink loaded native-library snapshot {}: {error}",
                self.path.display()
            ))
        })?;
        fs::remove_dir(&self.directory).map_err(|error| {
            RusticolError::artifact(format!(
                "could not remove native-library snapshot directory {}: {error}",
                self.directory.display()
            ))
        })?;
        self.active = false;
        Ok(())
    }
}

#[cfg(feature = "f64-compiled")]
fn native_library_snapshot_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(configured) =
        std::env::var_os("PYAMPLICOL_NATIVE_SNAPSHOT_ROOT").filter(|value| !value.is_empty())
    {
        let configured = PathBuf::from(configured);
        if !roots.contains(&configured) {
            roots.push(configured);
        }
    }
    let system = std::env::temp_dir();
    if !roots.contains(&system) {
        roots.push(system);
    }
    roots
}

#[cfg(feature = "f64-compiled")]
impl Drop for TemporaryNativeLibrary {
    fn drop(&mut self) {
        if self.active {
            let _ = fs::remove_file(&self.path);
            let _ = fs::remove_dir(&self.directory);
        }
    }
}

impl EvaluatorPayloadSource {
    pub(crate) fn read(&self) -> RusticolResult<Cow<'_, [u8]>> {
        match self {
            Self::File(path) => fs::read(path).map(Cow::Owned).map_err(|error| {
                RusticolError::artifact(format!(
                    "could not read evaluator payload {}: {error}",
                    path.display()
                ))
            }),
            Self::AuthenticatedFile { root, payload } => {
                read_declared_payload(root, payload).map(Cow::Owned)
            }
            Self::Packed {
                container,
                logical_path,
            } => container.member_bytes(logical_path).map(Cow::Borrowed),
            Self::PackedDeclared { container, payload } => {
                read_packed_payload(container, payload).map(Cow::Owned)
            }
        }
    }

    pub(crate) fn display_name(&self) -> String {
        match self {
            Self::File(path) => path.display().to_string(),
            Self::AuthenticatedFile { root, payload } => {
                root.join(&payload.path).display().to_string()
            }
            Self::Packed { logical_path, .. } => format!("evaluators.pacbin:{logical_path}"),
            Self::PackedDeclared { payload, .. } => {
                format!("{PAYLOAD_CONTAINER_PATH}:{}", payload.path)
            }
        }
    }
}

/// A path-scoped resolver shared by compiled and eager evaluator loaders.
#[derive(Clone, Debug)]
pub(crate) struct EvaluatorPayloadStore {
    artifact_root: PathBuf,
    relative_root: PathBuf,
    container: Option<Arc<PacbinReader>>,
    payloads: Option<Arc<BTreeMap<String, Payload>>>,
    payload_container: Option<Arc<PacbinFileReader>>,
    #[cfg(feature = "f64-compiled")]
    native_library_cache: Arc<Mutex<BTreeMap<String, Arc<PinnedNativeLibrary>>>>,
}

impl EvaluatorPayloadStore {
    pub(crate) fn directory(root: &Path) -> Self {
        Self {
            artifact_root: root.to_path_buf(),
            relative_root: root.to_path_buf(),
            container: None,
            payloads: None,
            payload_container: None,
            #[cfg(feature = "f64-compiled")]
            native_library_cache: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub(crate) fn logical_path(&self, value: &str) -> RusticolResult<String> {
        let relative = confined_evaluator_path(value)?;
        artifact_logical_path(&self.artifact_root, &self.relative_root.join(relative))
    }

    pub(crate) fn source(&self, value: &str) -> RusticolResult<EvaluatorPayloadSource> {
        let logical_path = self.logical_path(value)?;
        let path = self.artifact_root.join(&logical_path);
        if let Some(container) = &self.container
            && container.member(&logical_path).is_ok()
        {
            return Ok(EvaluatorPayloadSource::Packed {
                container: container.clone(),
                logical_path,
            });
        }
        if let Some(payloads) = &self.payloads {
            let payload = payloads.get(&logical_path).ok_or_else(|| {
                RusticolError::security(format!(
                    "evaluator payload {logical_path:?} is not declared by the verified artifact"
                ))
            })?;
            if payload.role != PayloadRole::EvaluatorState {
                return Err(RusticolError::security(format!(
                    "evaluator payload {logical_path:?} has role {:?}, expected evaluator-state",
                    payload.role
                )));
            }
            if let Some(container) = &self.payload_container
                && container.contains(&logical_path)
            {
                return Ok(EvaluatorPayloadSource::PackedDeclared {
                    container: container.clone(),
                    payload: payload.clone(),
                });
            }
            return Ok(EvaluatorPayloadSource::AuthenticatedFile {
                root: self.artifact_root.clone(),
                payload: payload.clone(),
            });
        }
        Ok(EvaluatorPayloadSource::File(path))
    }

    pub(crate) fn packed_member_bytes(
        &self,
        value: &str,
        expected_kind: PacbinMemberKind,
    ) -> RusticolResult<&[u8]> {
        let logical_path = self.logical_path(value)?;
        let container = self.container.as_ref().ok_or_else(|| {
            RusticolError::compatibility(format!(
                "typed evaluator payload {logical_path:?} requires evaluators.pacbin storage"
            ))
        })?;
        let member = container.member(&logical_path)?;
        if member.kind() != expected_kind {
            return Err(RusticolError::integrity(format!(
                "evaluator payload {logical_path:?} has pacbin member kind {:?}, expected {:?}",
                member.kind(),
                expected_kind,
            )));
        }
        container.member_bytes(&logical_path)
    }

    pub(crate) fn load_native_library(
        &self,
        value: &str,
    ) -> RusticolResult<Arc<PinnedNativeLibrary>> {
        #[cfg(not(feature = "f64-compiled"))]
        {
            let _ = value;
            return Err(RusticolError::compatibility(
                "native evaluator libraries require the f64-compiled feature",
            ));
        }
        #[cfg(feature = "f64-compiled")]
        {
            let relative = confined_evaluator_path(value)?;
            let path = self.relative_root.join(relative);
            let logical_path = artifact_logical_path(&self.artifact_root, &path)?;
            if let Some(library) = self
                .native_library_cache
                .lock()
                .map_err(|_| {
                    RusticolError::evaluation("native evaluator library cache is poisoned")
                })?
                .get(&logical_path)
                .cloned()
            {
                return Ok(library);
            }
            let source = self.source(value)?;
            if let EvaluatorPayloadSource::Packed { logical_path, .. } = &source {
                return Err(RusticolError::compatibility(format!(
                    "native evaluator library {logical_path:?} cannot be loaded from pacbin storage"
                )));
            }
            let display_path = PathBuf::from(source.display_name());
            let bytes = source.read()?;
            let loaded = PinnedNativeLibrary::from_bytes(bytes.as_ref(), display_path)?;
            let mut cache = self.native_library_cache.lock().map_err(|_| {
                RusticolError::evaluation("native evaluator library cache is poisoned")
            })?;
            Ok(cache
                .entry(logical_path)
                .or_insert_with(|| Arc::clone(&loaded))
                .clone())
        }
    }
}

impl VerifiedArtifact {
    /// Try the authenticated recurrence-only bootstrap without materializing
    /// broad generation/reporting extension trees.
    ///
    /// An absent conventional sidecar is not an error: callers may fall back
    /// to the generic loader for artifacts produced before this optimization.
    /// File presence opts a stable representative process into this compact
    /// path; every subsequent failure is reported rather than falling back to
    /// less constrained inputs.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub(crate) fn try_open_recurrence_bootstrap(
        path: impl AsRef<Path>,
        requested_process: Option<&str>,
    ) -> RusticolResult<Option<RecurrenceBootstrapArtifact>> {
        // Explicit aliases and expressions retain the generic selector path.
        // An omitted selector can use the image only when the conventional
        // single-process tree contains exactly one stable-ID sidecar.
        if requested_process.is_some_and(|process_id| {
            validate_public_id(process_id, "requested process id").is_err()
        }) {
            return Ok(None);
        }
        let requested = path.as_ref();
        reject_symlink_chain(requested)?;
        let (root, manifest_path) = locate_manifest(requested)?;
        reject_symlink_chain(&manifest_path)?;
        let discovered_process_id = if requested_process.is_none() {
            discover_single_recurrence_bootstrap_process(&root)?
        } else {
            None
        };
        let discovered_default = discovered_process_id.is_some();
        let Some(process_id) = requested_process.or(discovered_process_id.as_deref()) else {
            return Ok(None);
        };
        let bootstrap_path = format!("processes/{}/{}", process_id, RECURRENCE_BOOTSTRAP_FILENAME);
        let bootstrap_file = root.join(&bootstrap_path);
        let bootstrap_bytes = match read_bounded_recurrence_bootstrap(&root, &bootstrap_file)? {
            None if discovered_default => {
                return Err(RusticolError::integrity(format!(
                    "discovered default recurrence bootstrap {bootstrap_path:?} disappeared before opening"
                )));
            }
            None => return Ok(None),
            Some(bytes) => bytes,
        };
        let decoded = crate::engine::decode_recurrence_bootstrap_image_v1(&bootstrap_bytes)?;
        if decoded.process.id != process_id {
            return Err(RusticolError::integrity(format!(
                "recurrence bootstrap at {bootstrap_path:?} belongs to process {:?}",
                decoded.process.id
            )));
        }
        validate_recurrence_bootstrap_metadata(&decoded.producer, &decoded.runtime)?;

        let bootstrap_size = u64::try_from(bootstrap_bytes.len()).map_err(|_| {
            RusticolError::artifact("recurrence bootstrap size exceeds the u64 domain")
        })?;
        let bootstrap_payload = Payload {
            path: bootstrap_path.clone(),
            role: PayloadRole::EvaluatorState,
            media_type: "application/octet-stream".to_string(),
            size_bytes: bootstrap_size,
            sha256: format!("{:x}", Sha256::digest(&bootstrap_bytes)),
            executable: false,
            target: Some(decoded.producer.target.clone()),
            process_id: Some(process_id.to_string()),
        };
        let mut declared = decoded.payloads;
        let mut insertion_index = declared.len();
        let mut previous_path: Option<&str> = None;
        for (index, payload) in declared.iter().enumerate() {
            if previous_path.is_some_and(|previous| previous >= payload.path.as_str()) {
                return Err(RusticolError::integrity(
                    "recurrence bootstrap payload inventory must be strictly path-sorted",
                ));
            }
            if payload.path == bootstrap_path {
                return Err(RusticolError::integrity(format!(
                    "recurrence bootstrap inventory illegally contains its own record {bootstrap_path:?}"
                )));
            }
            if insertion_index == declared.len() && payload.path.as_str() > bootstrap_path.as_str()
            {
                insertion_index = index;
            }
            previous_path = Some(&payload.path);
        }
        declared.insert(insertion_index, bootstrap_payload);
        let manifest_artifact_id = read_canonical_manifest_artifact_id(&manifest_path)?;
        let computed_artifact_id = compute_payload_artifact_id(&declared)?;
        if computed_artifact_id != manifest_artifact_id {
            return Err(RusticolError::integrity(format!(
                "recurrence bootstrap payload identity {computed_artifact_id} disagrees with artifact manifest identity {manifest_artifact_id}"
            )));
        }

        let process = decoded.process;
        validate_recurrence_bootstrap_process(&process)?;
        let selection = representative_selection(&process);
        let manifest = recurrence_bootstrap_manifest(
            manifest_artifact_id,
            decoded.producer,
            decoded.runtime,
            process,
            declared,
        );
        let payloads = validated_payload_declarations(&manifest)?;
        for payload in payloads.values() {
            if let Some(target) = &payload.target {
                validate_payload_target(&manifest.producer.target, target, &payload.path)?;
                validate_target(target, &format!("payload {}", payload.path))?;
            }
        }
        validate_recurrence_bootstrap_source_payload(
            &payloads,
            &decoded.execution_path,
            &decoded.execution_sha256,
            PayloadRole::EvaluatorManifest,
            process_id,
        )?;
        validate_recurrence_bootstrap_source_payload(
            &payloads,
            &decoded.physics_path,
            &decoded.physics_sha256,
            PayloadRole::RuntimePhysics,
            process_id,
        )?;
        if decoded.physics_path != selection.process.physics_path {
            return Err(RusticolError::integrity(
                "recurrence bootstrap physics path disagrees with the selected process",
            ));
        }
        validate_evaluator_payload_container_declaration(
            &decoded.evaluator_payload_container,
            &payloads,
        )?;
        let payload_container = load_bootstrap_payload_container(&root, &payloads)?;
        let artifact = Self {
            root,
            manifest_path,
            manifest,
            payloads: Arc::new(payloads),
            // The process binding says whether this process has any JIT/native
            // leaves.  Avoid opening and retaining the model-wide evaluator
            // PACBIN for source/intrinsic-only recurrence processes.
            evaluator_payload_container: None,
            payload_container,
            #[cfg(feature = "f64-compiled")]
            native_library_cache: Arc::new(Mutex::new(BTreeMap::new())),
        };
        Ok(Some(RecurrenceBootstrapArtifact {
            artifact,
            selection,
            ready_execution: decoded.ready_execution,
            physics: decoded.physics,
            evaluator_payload_container: decoded.evaluator_payload_container,
        }))
    }

    /// Open a trusted artifact directory or a direct v3 manifest path.
    ///
    /// Schema, references, and confined paths are checked. Payload digests are
    /// deliberately not recomputed on the normal runtime path.
    pub fn open(path: impl AsRef<Path>) -> RusticolResult<Self> {
        Self::open_with_manifest_preflight(path, |_| Ok(()))
    }

    pub(crate) fn open_with_manifest_preflight(
        path: impl AsRef<Path>,
        preflight: impl FnOnce(&ArtifactManifest) -> RusticolResult<()>,
    ) -> RusticolResult<Self> {
        let requested = path.as_ref();
        reject_symlink_chain(requested)?;
        let (root, manifest_path) = locate_manifest(requested)?;
        reject_symlink_chain(&manifest_path)?;
        let metadata = fs::metadata(&manifest_path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not inspect artifact manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(RusticolError::security(format!(
                "artifact manifest {} is not a regular file",
                manifest_path.display()
            )));
        }
        let bytes = fs::read(&manifest_path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not read artifact manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        let header: Value = serde_json::from_slice(&bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse artifact manifest {} as JSON: {error}",
                manifest_path.display()
            ))
        })?;
        let schema_version = header
            .get("schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if matches!(schema_version, 1 | 2) {
            return Err(RusticolError::compatibility(format!(
                "process artifact schema v{schema_version} is unsupported and unsafe to migrate; regenerate it with `pyamplicol generate` to produce schema v3"
            )));
        }
        if schema_version != u64::from(PROCESS_ARTIFACT_SCHEMA_VERSION) {
            return Err(RusticolError::compatibility(format!(
                "unsupported process artifact schema {schema_version}; this runtime requires schema v{PROCESS_ARTIFACT_SCHEMA_VERSION}"
            )));
        }
        validate_artifact_identity_header(&header)?;
        reject_forbidden_nulls(&header)?;
        let manifest: ArtifactManifest = serde_json::from_slice(&bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "artifact manifest {} does not conform to schema v3: {error}",
                manifest_path.display()
            ))
        })?;
        validate_manifest(&manifest)?;
        preflight(&manifest)?;

        let payloads = validated_payload_declarations(&manifest)?;
        validate_references(&manifest, &payloads)?;
        let evaluator_payload_container =
            load_evaluator_payload_container(&root, &manifest, &payloads)?;
        let payload_container = load_payload_container(&root, &payloads)?;
        let payloads = Arc::new(payloads);
        Ok(Self {
            root,
            manifest_path,
            manifest,
            payloads,
            evaluator_payload_container,
            payload_container,
            #[cfg(feature = "f64-compiled")]
            native_library_cache: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Resolve a selector against the manifest, then against non-generated
    /// members of the declared process-group map. The map is read only when
    /// no stored process ID, alias, or stored expression (exact or
    /// side-permuted) matches.
    pub fn select_process(&self, requested: Option<&str>) -> RusticolResult<ArtifactSelection> {
        self.manifest
            .select_process_with_groups(requested, || self.process_groups())
    }

    fn process_groups(&self) -> RusticolResult<Option<ProcessGroups>> {
        let mut declared = self
            .payloads
            .values()
            .filter(|payload| payload.role == PayloadRole::ProcessGroups);
        let Some(payload) = declared.next() else {
            return Ok(None);
        };
        if declared.next().is_some() {
            return Err(RusticolError::artifact(
                "artifact declares more than one process-groups payload",
            ));
        }
        let bytes = self.read_declared(payload)?;
        if format!("{:x}", Sha256::digest(&bytes)) != payload.sha256 {
            return Err(RusticolError::integrity(format!(
                "process-groups payload {:?} does not match its declared sha256",
                payload.path
            )));
        }
        ProcessGroups::from_json(&bytes, &self.manifest).map(Some)
    }

    pub fn payload(&self, path: &str) -> RusticolResult<&Payload> {
        self.payloads.get(path).ok_or_else(|| {
            RusticolError::security(format!("artifact path {path:?} is not a declared payload"))
        })
    }

    pub fn read_payload(&self, path: &str) -> RusticolResult<Vec<u8>> {
        let payload = self.payload(path)?;
        self.read_declared(payload)
    }

    fn read_declared(&self, payload: &Payload) -> RusticolResult<Vec<u8>> {
        match &self.payload_container {
            Some(container) if container.contains(&payload.path) => {
                read_packed_payload(container, payload)
            }
            _ => read_declared_payload(&self.root, payload),
        }
    }

    fn is_packed(&self, path: &str) -> bool {
        self.payload_container
            .as_ref()
            .is_some_and(|container| container.contains(path))
    }

    /// Open a declared payload that is itself a pacbin container, loose or packed,
    /// and authenticate it against `expected_sha256`.
    pub(crate) fn open_payload_pacbin(
        &self,
        path: &str,
        expected_sha256: &[u8; 32],
    ) -> RusticolResult<PacbinReader> {
        if self.is_packed(path) {
            let bytes = self.read_payload(path)?;
            if <[u8; 32]>::from(Sha256::digest(&bytes)) != *expected_sha256 {
                return Err(RusticolError::integrity(format!(
                    "pacbin container {path:?} payload digest mismatch"
                )));
            }
            return PacbinReader::from_bytes(bytes);
        }
        let file = self.open_payload_file(path)?;
        PacbinReader::open_file_with_sha256(file, &self.root.join(path), expected_sha256)
    }

    /// Open one declared loose payload and pin the checked file description.
    ///
    /// Callers that stream or map a payload must authenticate this returned
    /// file, rather than reopening `path`, so path replacement cannot change
    /// the bytes between declaration checks and use.
    pub(crate) fn open_payload_file(&self, path: &str) -> RusticolResult<File> {
        let payload = self.payload(path)?;
        if self.is_packed(path) {
            return Err(RusticolError::compatibility(format!(
                "payload {path:?} is stored in {PAYLOAD_CONTAINER_PATH} and cannot be opened as a file"
            )));
        }
        open_checked_payload(&self.root, payload)
    }

    /// The location of a declared payload; packed payloads have no file there.
    pub(crate) fn payload_path(&self, path: &str) -> RusticolResult<PathBuf> {
        let payload = self.payload(path)?;
        if !self.is_packed(path) {
            drop(open_checked_payload(&self.root, payload)?);
        }
        Ok(self.root.join(path))
    }

    pub(crate) fn evaluator_payload_store(
        &self,
        relative_root: &Path,
    ) -> RusticolResult<EvaluatorPayloadStore> {
        if !relative_root.starts_with(&self.root) {
            return Err(RusticolError::security(
                "evaluator payload root escapes the artifact root",
            ));
        }
        Ok(EvaluatorPayloadStore {
            artifact_root: self.root.clone(),
            relative_root: relative_root.to_path_buf(),
            container: self.evaluator_payload_container.clone(),
            payloads: Some(self.payloads.clone()),
            payload_container: self.payload_container.clone(),
            #[cfg(feature = "f64-compiled")]
            native_library_cache: self.native_library_cache.clone(),
        })
    }

    pub(crate) fn install_evaluator_payload_container(
        &mut self,
        extension: &EvaluatorPayloadContainerExtension,
    ) -> RusticolResult<()> {
        if self.evaluator_payload_container.is_none() {
            self.evaluator_payload_container = Some(
                load_evaluator_payload_container_extension(&self.root, extension, &self.payloads)?
                    .ok_or_else(|| {
                        RusticolError::internal(
                            "declared evaluator payload container resolved to no reader",
                        )
                    })?,
            );
        }
        Ok(())
    }

    pub(crate) fn has_evaluator_payload(&self, path: &str) -> RusticolResult<bool> {
        if let Some(payload) = self.payloads.get(path) {
            return Ok(payload.role == PayloadRole::EvaluatorState);
        }
        Ok(self
            .evaluator_payload_container
            .as_ref()
            .is_some_and(|container| container.member(path).is_ok()))
    }
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn validated_payload_declarations(
    manifest: &ArtifactManifest,
) -> RusticolResult<BTreeMap<String, Payload>> {
    let mut payloads = BTreeMap::new();
    let mut portable_paths = BTreeSet::new();
    for payload in &manifest.payloads {
        validate_payload_declaration(payload)?;
        validate_relative_path(&payload.path, "payload path")?;
        if payload.path == ARTIFACT_MANIFEST_FILE {
            return Err(RusticolError::security(format!(
                "{ARTIFACT_MANIFEST_FILE} is reserved for the artifact manifest"
            )));
        }
        let portable = payload.path.to_ascii_lowercase();
        if !portable_paths.insert(portable) {
            return Err(RusticolError::security(format!(
                "duplicate or case-colliding payload path {:?}",
                payload.path
            )));
        }
        if payloads
            .insert(payload.path.clone(), payload.clone())
            .is_some()
        {
            return Err(RusticolError::security(format!(
                "duplicate payload path {:?}",
                payload.path
            )));
        }
    }
    Ok(payloads)
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn read_bounded_recurrence_bootstrap(root: &Path, path: &Path) -> RusticolResult<Option<Vec<u8>>> {
    reject_symlink_chain(path)?;
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(RusticolError::artifact(format!(
                "could not inspect recurrence bootstrap {}: {error}",
                path.display()
            )));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(RusticolError::security(format!(
            "recurrence bootstrap {} must be a regular non-symlink file",
            path.display()
        )));
    }
    let size = usize::try_from(metadata.len()).map_err(|_| {
        RusticolError::artifact("recurrence bootstrap size exceeds this platform's usize domain")
    })?;
    if size == 0 {
        return Err(RusticolError::artifact(format!(
            "recurrence bootstrap {} is empty",
            path.display()
        )));
    }
    #[cfg(unix)]
    if metadata_is_executable(&metadata) {
        return Err(RusticolError::security(format!(
            "recurrence bootstrap {} must not be executable",
            path.display()
        )));
    }
    let canonical = path.canonicalize().map_err(|error| {
        RusticolError::security(format!(
            "could not resolve recurrence bootstrap {}: {error}",
            path.display()
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(RusticolError::security(format!(
            "recurrence bootstrap {} escapes the artifact root",
            path.display()
        )));
    }
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt as _;
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(path)
    };
    #[cfg(not(unix))]
    let file = OpenOptions::new().read(true).open(path);
    let file = file.map_err(|error| {
        RusticolError::artifact(format!(
            "could not open recurrence bootstrap {}: {error}",
            path.display()
        ))
    })?;
    let pinned = file.metadata().map_err(|error| {
        RusticolError::security(format!(
            "could not inspect opened recurrence bootstrap {}: {error}",
            path.display()
        ))
    })?;
    if !pinned.is_file() || pinned.len() != metadata.len() {
        return Err(RusticolError::integrity(format!(
            "recurrence bootstrap {} changed while being opened",
            path.display()
        )));
    }
    reject_symlink_chain(path)?;
    let path_metadata = fs::symlink_metadata(path).map_err(|error| {
        RusticolError::security(format!(
            "could not re-inspect recurrence bootstrap {}: {error}",
            path.display()
        ))
    })?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(RusticolError::security(format!(
            "recurrence bootstrap {} is not a regular non-symlink file",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if pinned.dev() != path_metadata.dev() || pinned.ino() != path_metadata.ino() {
            return Err(RusticolError::security(format!(
                "recurrence bootstrap {} was replaced while being opened",
                path.display()
            )));
        }
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|error| {
        RusticolError::artifact(format!(
            "could not allocate {size} recurrence bootstrap bytes: {error}"
        ))
    })?;
    // Read at most one byte beyond the pinned size. This detects a concurrent
    // append without allowing a growing file to bypass the allocation bound.
    let mut bounded = file.take(metadata.len().saturating_add(1));
    bounded.read_to_end(&mut bytes).map_err(|error| {
        RusticolError::artifact(format!(
            "could not read recurrence bootstrap {}: {error}",
            path.display()
        ))
    })?;
    if bytes.len() != size {
        return Err(RusticolError::integrity(format!(
            "recurrence bootstrap {} changed while being read",
            path.display()
        )));
    }
    Ok(Some(bytes))
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn discover_single_recurrence_bootstrap_process(root: &Path) -> RusticolResult<Option<String>> {
    let processes = root.join("processes");
    let metadata = match fs::symlink_metadata(&processes) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(RusticolError::artifact(format!(
                "could not inspect recurrence process root {}: {error}",
                processes.display()
            )));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(RusticolError::security(format!(
            "recurrence process root {} must be a regular non-symlink directory",
            processes.display()
        )));
    }

    let entries = fs::read_dir(&processes).map_err(|error| {
        RusticolError::artifact(format!(
            "could not enumerate recurrence process root {}: {error}",
            processes.display()
        ))
    })?;
    let mut selected: Option<String> = None;
    for entry in entries {
        let entry = entry.map_err(|error| {
            RusticolError::artifact(format!(
                "could not enumerate recurrence process root {}: {error}",
                processes.display()
            ))
        })?;
        let Ok(process_id) = entry.file_name().into_string() else {
            continue;
        };
        if validate_public_id(&process_id, "recurrence process directory").is_err() {
            continue;
        }
        let file_type = entry.file_type().map_err(|error| {
            RusticolError::artifact(format!(
                "could not inspect recurrence process directory {}: {error}",
                entry.path().display()
            ))
        })?;
        if file_type.is_symlink() {
            return Err(RusticolError::security(format!(
                "recurrence process directory {} must not be a symlink",
                entry.path().display()
            )));
        }
        if !file_type.is_dir() {
            continue;
        }
        let sidecar = entry.path().join(RECURRENCE_BOOTSTRAP_FILENAME);
        let sidecar_metadata = match fs::symlink_metadata(&sidecar) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(RusticolError::artifact(format!(
                    "could not inspect recurrence bootstrap {}: {error}",
                    sidecar.display()
                )));
            }
        };
        if sidecar_metadata.file_type().is_symlink() || !sidecar_metadata.is_file() {
            return Err(RusticolError::security(format!(
                "recurrence bootstrap {} must be a regular non-symlink file",
                sidecar.display()
            )));
        }
        if selected.replace(process_id).is_some() {
            return Err(RusticolError::integrity(
                "default recurrence bootstrap selection is ambiguous",
            ));
        }
    }
    Ok(selected)
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn read_canonical_manifest_artifact_id(path: &Path) -> RusticolResult<String> {
    const PREFIX: &[u8] = b"{\"artifact_id\":\"";
    const SUFFIX: &[u8] = b"\",";
    let read_len = PREFIX.len() + 64 + SUFFIX.len();
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt as _;
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(path)
    };
    #[cfg(not(unix))]
    let mut file = OpenOptions::new().read(true).open(path);
    let mut file = file.map_err(|error| {
        RusticolError::artifact(format!(
            "could not open artifact manifest {}: {error}",
            path.display()
        ))
    })?;
    if !file
        .metadata()
        .map_err(|error| {
            RusticolError::security(format!(
                "could not inspect artifact manifest {}: {error}",
                path.display()
            ))
        })?
        .is_file()
    {
        return Err(RusticolError::security(format!(
            "artifact manifest {} is not a regular file",
            path.display()
        )));
    }
    let mut prefix = vec![0_u8; read_len];
    file.read_exact(&mut prefix).map_err(|error| {
        RusticolError::serialization(format!(
            "artifact manifest {} has no canonical artifact-ID prefix: {error}",
            path.display()
        ))
    })?;
    let digest = &prefix[PREFIX.len()..PREFIX.len() + 64];
    if !prefix.starts_with(PREFIX)
        || &prefix[PREFIX.len() + 64..] != SUFFIX
        || !digest
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(RusticolError::integrity(format!(
            "artifact manifest {} is not canonical at its artifact-ID prefix",
            path.display()
        )));
    }
    String::from_utf8(digest.to_vec()).map_err(|error| {
        RusticolError::internal(format!(
            "validated ASCII artifact ID was not UTF-8: {error}"
        ))
    })
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn compute_payload_artifact_id(payloads: &[Payload]) -> RusticolResult<String> {
    let mut canonical = String::new();
    canonical
        .try_reserve(payloads.len().checked_mul(256).unwrap_or_default())
        .map_err(|error| {
            RusticolError::artifact(format!(
                "could not allocate recurrence artifact identity buffer: {error}"
            ))
        })?;
    canonical.push_str("{\"kind\":\"pyamplicol-runtime-payload-identity\",\"payloads\":[");
    let mut previous_path: Option<&str> = None;
    let mut first = true;
    for payload in payloads.iter().filter(|payload| {
        matches!(
            payload.role,
            PayloadRole::CompiledModel
                | PayloadRole::EvaluatorManifest
                | PayloadRole::EvaluatorState
                | PayloadRole::ModelParameters
                | PayloadRole::RuntimePhysics
        )
    }) {
        if previous_path.is_some_and(|previous| previous >= payload.path.as_str()) {
            return Err(RusticolError::integrity(
                "recurrence bootstrap payload inventory must be strictly path-sorted",
            ));
        }
        previous_path = Some(&payload.path);
        if !first {
            canonical.push(',');
        }
        first = false;
        canonical.push_str("{\"executable\":");
        canonical.push_str(if payload.executable { "true" } else { "false" });
        canonical.push_str(",\"media_type\":");
        write_python_json_string(&payload.media_type, &mut canonical);
        canonical.push_str(",\"path\":");
        write_python_json_string(&payload.path, &mut canonical);
        if let Some(process_id) = &payload.process_id {
            canonical.push_str(",\"process_id\":");
            write_python_json_string(process_id, &mut canonical);
        }
        canonical.push_str(",\"role\":");
        write_python_json_string(payload_role_name(payload.role), &mut canonical);
        canonical.push_str(",\"sha256\":");
        write_python_json_string(&payload.sha256, &mut canonical);
        canonical.push_str(",\"size_bytes\":");
        let _ = write!(canonical, "{}", payload.size_bytes);
        if let Some(target) = &payload.target {
            canonical.push_str(",\"target\":{\"cpu_features\":[");
            for (feature_index, feature) in target.cpu_features.iter().enumerate() {
                if feature_index != 0 {
                    canonical.push(',');
                }
                write_python_json_string(feature, &mut canonical);
            }
            canonical.push_str("],\"triple\":");
            write_python_json_string(&target.triple, &mut canonical);
            canonical.push('}');
        }
        canonical.push('}');
    }
    canonical.push_str("],\"schema_version\":1}\n");
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn payload_role_name(role: PayloadRole) -> &'static str {
    match role {
        PayloadRole::ConfigurationRequested => "configuration-requested",
        PayloadRole::ConfigurationEffective => "configuration-effective",
        PayloadRole::CompiledModel => "compiled-model",
        PayloadRole::RuntimePhysics => "runtime-physics",
        PayloadRole::EvaluatorManifest => "evaluator-manifest",
        PayloadRole::EvaluatorState => "evaluator-state",
        PayloadRole::ModelParameters => "model-parameters",
        PayloadRole::ValidationMomenta => "validation-momenta",
        PayloadRole::StructuralSourceProof => "structural-source-proof",
        PayloadRole::ApiSource => "api-source",
        PayloadRole::ApiBuildFile => "api-build-file",
        PayloadRole::SdkMetadata => "sdk-metadata",
        PayloadRole::ProcessGroups => "process-groups",
        PayloadRole::PayloadContainer => "payload-container",
    }
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn validate_recurrence_bootstrap_metadata(
    producer: &Producer,
    runtime: &ArtifactRuntime,
) -> RusticolResult<()> {
    if producer.distribution != "pyamplicol"
        || runtime.engine != "rusticol"
        || runtime.engine_version != producer.version
        || !compatible_distribution_version(&producer.version)
    {
        return Err(RusticolError::compatibility(format!(
            "recurrence bootstrap producer/runtime {}/{}/{} is incompatible with Rusticol {}",
            producer.distribution,
            producer.version,
            runtime.engine_version,
            rusticol_package_version()
        )));
    }
    let versions = &producer.versions;
    for (name, found, expected) in [
        ("python API", versions.python_api, PYTHON_API_VERSION),
        ("TOML", versions.toml, TOML_SCHEMA_VERSION),
        (
            "compiled model",
            versions.compiled_model,
            COMPILED_MODEL_SCHEMA_VERSION,
        ),
        (
            "process artifact",
            versions.process_artifact,
            PROCESS_ARTIFACT_SCHEMA_VERSION,
        ),
        (
            "runtime physics",
            versions.runtime_physics,
            RUNTIME_PHYSICS_SCHEMA_VERSION,
        ),
        ("C ABI", versions.c_abi, C_ABI_VERSION),
    ] {
        if found != expected {
            return Err(RusticolError::compatibility(format!(
                "recurrence bootstrap {name} version {found} is incompatible with {expected}"
            )));
        }
    }
    if let Some(revision) = &producer.git_revision
        && (revision.len() != 40
            || !revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()))
    {
        return Err(RusticolError::artifact(
            "recurrence bootstrap producer git revision is not canonical",
        ));
    }
    if let Some(identity) = &producer.native_build_inputs_sha256 {
        validate_sha256(identity, "recurrence bootstrap native build identity")?;
    }
    validate_relative_path(
        &runtime.evaluator_manifest_path,
        "recurrence bootstrap evaluator manifest path",
    )?;
    if let Some(path) = &runtime.api_bundle_path {
        validate_relative_path(path, "recurrence bootstrap API bundle path")?;
    }
    let capabilities = validate_runtime_capabilities(
        &runtime.required_runtime_capabilities,
        "recurrence bootstrap runtime capabilities",
    )?;
    validate_target(&producer.target, "recurrence bootstrap producer")?;
    validate_portable_runtime_capabilities(&producer.target, &capabilities)
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn validate_recurrence_bootstrap_process(process: &ArtifactProcess) -> RusticolResult<()> {
    validate_public_id(&process.id, "recurrence bootstrap process id")?;
    if process.expression.is_empty()
        || process.external_pdgs.len() < 3
        || !matches!(process.color_accuracy.as_str(), "lc" | "nlc" | "full")
    {
        return Err(RusticolError::artifact(format!(
            "recurrence bootstrap process {:?} has invalid public metadata",
            process.id
        )));
    }
    let expected_physics = format!("processes/{}/physics.json", process.id);
    if process.physics_path != expected_physics {
        return Err(RusticolError::security(format!(
            "recurrence bootstrap process {:?} physics path must be {expected_physics:?}",
            process.id
        )));
    }
    validate_runtime_capabilities(
        &process.required_runtime_capabilities,
        "recurrence bootstrap process capabilities",
    )?;
    if process.aliases.is_empty() {
        return Ok(());
    }
    let initial_count = parse_process_expression(&process.expression)
        .map(|tokens| tokens.initial.len())
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "recurrence bootstrap process expression {:?} is invalid",
                process.expression
            ))
        })?;
    let mut aliases = BTreeSet::new();
    for alias in &process.aliases {
        validate_public_id(&alias.id, "recurrence bootstrap alias id")?;
        if !aliases.insert(alias.id.as_str())
            || alias.expression.is_empty()
            || alias.external_pdgs.len() != process.external_pdgs.len()
        {
            return Err(RusticolError::artifact(format!(
                "recurrence bootstrap alias {:?} is invalid",
                alias.id
            )));
        }
        validate_permutation(
            &alias.external_permutation,
            process.external_pdgs.len(),
            &format!("alias {:?}", alias.id),
        )?;
        if !permutation_preserves_process_sides(&alias.external_permutation, initial_count) {
            return Err(RusticolError::artifact(format!(
                "recurrence bootstrap alias {:?} crosses process sides",
                alias.id
            )));
        }
    }
    Ok(())
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn validate_recurrence_bootstrap_source_payload(
    payloads: &BTreeMap<String, Payload>,
    path: &str,
    sha256: &str,
    role: PayloadRole,
    process_id: &str,
) -> RusticolResult<()> {
    let payload = payloads.get(path).ok_or_else(|| {
        RusticolError::security(format!(
            "recurrence bootstrap source {path:?} is not a declared payload"
        ))
    })?;
    if payload.role != role
        || payload.media_type != "application/json"
        || payload.process_id.as_deref() != Some(process_id)
        || payload.executable
        || payload.sha256 != sha256
    {
        return Err(RusticolError::integrity(format!(
            "recurrence bootstrap source {path:?} disagrees with its payload declaration"
        )));
    }
    Ok(())
}

fn recurrence_bootstrap_manifest(
    artifact_id: String,
    producer: Producer,
    runtime: ArtifactRuntime,
    process: ArtifactProcess,
    payloads: Vec<Payload>,
) -> ArtifactManifest {
    let process_id = process.id.clone();
    ArtifactManifest {
        schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
        kind: ArtifactKind::PyamplicolProcess,
        artifact_id,
        created_utc: "1970-01-01T00:00:00Z".to_string(),
        producer,
        model: ArtifactModel {
            name: "recurrence-bootstrap".to_string(),
            source_kind: ModelSourceKind::CompiledModel,
            content_sha256: "0".repeat(64),
            compiled_schema_version: COMPILED_MODEL_SCHEMA_VERSION,
            restriction: None,
        },
        configuration: ArtifactConfiguration {
            toml_schema_version: TOML_SCHEMA_VERSION,
            requested_path: "config/requested.toml".to_string(),
            effective_path: "config/effective.toml".to_string(),
            adjustments: Vec::new(),
        },
        processes: vec![process],
        default_process_id: Some(process_id),
        runtime,
        payloads,
        dependencies: Vec::new(),
        extensions: BTreeMap::new(),
    }
}

/// Construct the same narrowly-scoped artifact view while generation still
/// owns the private staging directory.  This is used only to lower the ready
/// recipe through the authoritative PACBIN/binding loaders; the staging path
/// itself is never serialized.
#[cfg(feature = "python-generation-bridge")]
pub(crate) fn open_staged_recurrence_bootstrap_artifact(
    root: PathBuf,
    producer: Producer,
    runtime: ArtifactRuntime,
    process: ArtifactProcess,
    payloads: Vec<Payload>,
    evaluator_payload_container: EvaluatorPayloadContainerExtension,
) -> RusticolResult<VerifiedArtifact> {
    let root = fs::canonicalize(&root).map_err(|error| {
        RusticolError::artifact(format!(
            "could not resolve recurrence bootstrap staging root {}: {error}",
            root.display()
        ))
    })?;
    if !root.is_dir() {
        return Err(RusticolError::artifact(format!(
            "recurrence bootstrap staging root {} is not a directory",
            root.display()
        )));
    }
    validate_recurrence_bootstrap_metadata(&producer, &runtime)?;
    validate_recurrence_bootstrap_process(&process)?;
    let manifest =
        recurrence_bootstrap_manifest("0".repeat(64), producer, runtime, process, payloads);
    let payloads = validated_payload_declarations(&manifest)?;
    for payload in payloads.values() {
        if let Some(target) = &payload.target {
            validate_payload_target(&manifest.producer.target, target, &payload.path)?;
            validate_target(target, &format!("payload {}", payload.path))?;
        }
    }
    validate_evaluator_payload_container_declaration(&evaluator_payload_container, &payloads)?;
    let payload_container = load_payload_container(&root, &payloads)?;
    Ok(VerifiedArtifact {
        manifest_path: root.join(ARTIFACT_MANIFEST_FILE),
        root,
        manifest,
        payloads: Arc::new(payloads),
        // Process-ready lowering reads only plan/binding/color payloads.
        evaluator_payload_container: None,
        payload_container,
        #[cfg(feature = "f64-compiled")]
        native_library_cache: Arc::new(Mutex::new(BTreeMap::new())),
    })
}

/// Open the payload container, checking that every member is a declared payload
/// with the declared size and digest.
fn load_payload_container(
    root: &Path,
    payloads: &BTreeMap<String, Payload>,
) -> RusticolResult<Option<Arc<PacbinFileReader>>> {
    let mut declared = payloads
        .values()
        .filter(|payload| payload.role == PayloadRole::PayloadContainer);
    let Some(payload) = declared.next() else {
        return Ok(None);
    };
    if declared.next().is_some() || payload.path != PAYLOAD_CONTAINER_PATH {
        return Err(RusticolError::artifact(format!(
            "an artifact declares at most one payload container, at {PAYLOAD_CONTAINER_PATH:?}"
        )));
    }
    open_payload_container(root, payload, payloads, true).map(Some)
}

/// The payload container next to a recurrence bootstrap. Its inventory predates
/// the container, so only the members it declares are checked and used.
fn load_bootstrap_payload_container(
    root: &Path,
    payloads: &BTreeMap<String, Payload>,
) -> RusticolResult<Option<Arc<PacbinFileReader>>> {
    let metadata = match fs::symlink_metadata(root.join(PAYLOAD_CONTAINER_PATH)) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(RusticolError::security(format!(
                "could not inspect {PAYLOAD_CONTAINER_PATH:?}: {error}"
            )));
        }
    };
    let container = Payload {
        path: PAYLOAD_CONTAINER_PATH.to_string(),
        role: PayloadRole::PayloadContainer,
        media_type: "application/octet-stream".to_string(),
        size_bytes: metadata.len(),
        sha256: String::new(),
        executable: false,
        target: None,
        process_id: None,
    };
    open_payload_container(root, &container, payloads, false).map(Some)
}

fn open_payload_container(
    root: &Path,
    payload: &Payload,
    payloads: &BTreeMap<String, Payload>,
    require_declared: bool,
) -> RusticolResult<Arc<PacbinFileReader>> {
    let file = open_checked_payload(root, payload)?;
    let reader = PacbinFileReader::open_file(file, &root.join(&payload.path))?;
    if reader.index().file_size() != payload.size_bytes {
        return Err(RusticolError::integrity(
            "payload container size disagrees with its declaration",
        ));
    }
    for member in reader.index().members() {
        let Some(packed) = payloads.get(member.logical_path()) else {
            if require_declared {
                return Err(RusticolError::integrity(format!(
                    "payload container member {:?} is not a declared payload",
                    member.logical_path()
                )));
            }
            continue;
        };
        if member.kind() != PacbinMemberKind::DeclaredPayload
            || packed.role == PayloadRole::PayloadContainer
            || packed.executable
            || member.length() != packed.size_bytes
            || hex_digest(member.sha256()) != packed.sha256
        {
            return Err(RusticolError::integrity(format!(
                "invalid payload container member {:?}",
                member.logical_path()
            )));
        }
    }
    Ok(Arc::new(reader))
}

fn read_packed_payload(container: &PacbinFileReader, payload: &Payload) -> RusticolResult<Vec<u8>> {
    validate_payload_declaration(payload)?;
    let bytes = container.read_member(&payload.path)?;
    if u64::try_from(bytes.len()).ok() != Some(payload.size_bytes) {
        return Err(RusticolError::integrity(format!(
            "packed payload {:?} disagrees with its declared size",
            payload.path
        )));
    }
    Ok(bytes)
}

fn load_evaluator_payload_container(
    root: &Path,
    manifest: &ArtifactManifest,
    payloads: &BTreeMap<String, Payload>,
) -> RusticolResult<Option<Arc<PacbinReader>>> {
    let Some(raw) = manifest
        .extensions
        .get(EVALUATOR_PAYLOAD_CONTAINER_EXTENSION)
    else {
        return Ok(None);
    };
    let extension: EvaluatorPayloadContainerExtension = serde_json::from_value(raw.clone())
        .map_err(|error| {
            RusticolError::artifact(format!(
                "artifact extension {EVALUATOR_PAYLOAD_CONTAINER_EXTENSION:?} is invalid: {error}"
            ))
        })?;
    load_evaluator_payload_container_extension(root, &extension, payloads)
}

fn load_evaluator_payload_container_extension(
    root: &Path,
    extension: &EvaluatorPayloadContainerExtension,
    payloads: &BTreeMap<String, Payload>,
) -> RusticolResult<Option<Arc<PacbinReader>>> {
    let payload = validate_evaluator_payload_container_declaration(extension, payloads)?;
    let expected_payload_sha = parse_sha256(
        &payload.sha256,
        "evaluator payload container payload SHA-256",
    )?;
    let container_path = root.join(&extension.path);
    let container_file = open_checked_payload(root, payload)?;
    let reader = PacbinReader::open_file_with_sha256(
        container_file,
        &container_path,
        &expected_payload_sha,
    )?;
    let index = reader.index();
    if index.version() != 1
        || u64::try_from(index.members().len()).unwrap_or(u64::MAX) != extension.member_count
        || hex_digest(index.index_sha256()) != extension.index_sha256
    {
        return Err(RusticolError::integrity(
            "evaluator payload container metadata disagrees with its authenticated index",
        ));
    }
    let unpacked_size = index.members().iter().try_fold(0_u64, |total, member| {
        total
            .checked_add(member.length())
            .ok_or_else(|| RusticolError::integrity("packed evaluator size exceeds u64"))
    })?;
    if unpacked_size != extension.unpacked_size_bytes {
        return Err(RusticolError::integrity(
            "evaluator payload container unpacked size disagrees with its index",
        ));
    }
    for member in index.members() {
        validate_relative_path(member.logical_path(), "packed evaluator logical path")?;
        if payloads.contains_key(member.logical_path())
            || member.logical_path() == extension.path
            || !matches!(
                member.kind(),
                PacbinMemberKind::SymjitApplication
                    | PacbinMemberKind::SymbolicaExactState
                    | PacbinMemberKind::ColorContraction
            )
        {
            return Err(RusticolError::integrity(format!(
                "invalid packed evaluator member {:?}",
                member.logical_path()
            )));
        }
    }
    Ok(Some(Arc::new(reader)))
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn validate_evaluator_payload_container_declaration<'a>(
    extension: &EvaluatorPayloadContainerExtension,
    payloads: &'a BTreeMap<String, Payload>,
) -> RusticolResult<&'a Payload> {
    if extension.kind != EVALUATOR_PAYLOAD_CONTAINER_KIND
        || extension.schema_version != 1
        || extension.storage_abi != EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI
    {
        return Err(RusticolError::compatibility(format!(
            "unsupported evaluator payload container kind/version/ABI: {:?}/{}/{}",
            extension.kind, extension.schema_version, extension.storage_abi
        )));
    }
    validate_relative_path(&extension.path, "evaluator payload container path")?;
    validate_sha256(
        &extension.index_sha256,
        "evaluator payload container index_sha256",
    )?;
    let payload = payloads.get(&extension.path).ok_or_else(|| {
        RusticolError::security(format!(
            "evaluator payload container {:?} is not a declared payload",
            extension.path
        ))
    })?;
    if payload.role != PayloadRole::EvaluatorState
        || payload.media_type != "application/octet-stream"
        || payload.process_id.is_some()
    {
        return Err(RusticolError::artifact(
            "evaluator payload container must be a root evaluator-state octet-stream payload",
        ));
    }
    parse_sha256(
        &payload.sha256,
        "evaluator payload container payload SHA-256",
    )?;
    Ok(payload)
}

fn confined_evaluator_path(value: &str) -> RusticolResult<&Path> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(RusticolError::security(format!(
            "evaluator payload path {value:?} is not a confined relative path"
        )));
    }
    Ok(path)
}

fn artifact_logical_path(root: &Path, path: &Path) -> RusticolResult<String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| RusticolError::security("evaluator payload path escapes the artifact root"))?;
    let logical = relative
        .to_str()
        .ok_or_else(|| RusticolError::security("evaluator payload path is not valid UTF-8"))?;
    Ok(logical.replace(std::path::MAIN_SEPARATOR, "/"))
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(result, "{byte:02x}");
    }
    result
}

impl ArtifactManifest {
    pub(crate) fn select_process(
        &self,
        requested: Option<&str>,
    ) -> RusticolResult<ArtifactSelection> {
        self.select_process_with_groups(requested, || Ok(None))
    }

    /// Resolve `requested` in precedence order. `groups` is called at most
    /// once, and only after every stored process, alias, and side permutation
    /// of their expressions failed to match.
    pub(crate) fn select_process_with_groups(
        &self,
        requested: Option<&str>,
        groups: impl FnOnce() -> RusticolResult<Option<ProcessGroups>>,
    ) -> RusticolResult<ArtifactSelection> {
        let selected_id = if let Some(requested) = requested {
            requested
        } else if let Some(default) = self.default_process_id.as_deref() {
            default
        } else if self.processes.len() == 1 {
            self.processes[0].id.as_str()
        } else {
            return Err(RusticolError::selector(
                "this process-set artifact has no default process; select a process or alias id",
            ));
        };
        for process in &self.processes {
            if process.id == selected_id {
                return Ok(representative_selection(process));
            }
            if let Some(alias) = process.aliases.iter().find(|alias| alias.id == selected_id) {
                return Ok(alias_selection(process, alias));
            }
        }

        let requested_expression = normalize_process_expression(selected_id);
        let mut process_expression_matches = self
            .processes
            .iter()
            .filter(|process| {
                normalize_process_expression(&process.expression) == requested_expression
            })
            .map(representative_selection)
            .collect::<Vec<_>>();
        match process_expression_matches.len() {
            1 => {
                return Ok(process_expression_matches
                    .pop()
                    .expect("one process expression match"));
            }
            count if count > 1 => {
                let mut matching_ids = process_expression_matches
                    .iter()
                    .map(|selection| selection.requested_id.as_str())
                    .collect::<Vec<_>>();
                matching_ids.sort_unstable();
                return Err(RusticolError::selector(format!(
                    "process expression {selected_id:?} is ambiguous; select one of these stable ids: {}",
                    matching_ids.join(", ")
                )));
            }
            _ => {}
        }
        let mut alias_expression_matches = self
            .processes
            .iter()
            .flat_map(|process| {
                process
                    .aliases
                    .iter()
                    .filter(|alias| {
                        normalize_process_expression(&alias.expression) == requested_expression
                    })
                    .map(|alias| alias_selection(process, alias))
            })
            .collect::<Vec<_>>();
        match alias_expression_matches.len() {
            1 => {
                return Ok(alias_expression_matches
                    .pop()
                    .expect("one alias expression match"));
            }
            count if count > 1 => {
                let mut matching_ids = alias_expression_matches
                    .iter()
                    .map(|selection| selection.requested_id.as_str())
                    .collect::<Vec<_>>();
                matching_ids.sort_unstable();
                return Err(RusticolError::selector(format!(
                    "process expression {selected_id:?} is ambiguous; select one of these stable ids: {}",
                    matching_ids.join(", ")
                )));
            }
            _ => {}
        }

        let Some(requested_tokens) = parse_process_expression(selected_id) else {
            if let Some(groups) = groups()?
                && let Some(selection) = groups.select(self, selected_id, None)?
            {
                return Ok(selection);
            }
            if selected_id.contains('>') {
                return Err(RusticolError::selector(format!(
                    "concrete process expression {selected_id:?} must contain one '>' with at least one particle on each side"
                )));
            }
            return Err(RusticolError::selector(format!(
                "unknown process id or alias id {selected_id:?}"
            )));
        };
        let mut permutation_matches = Vec::new();
        let mut crossing_only_matches = Vec::new();
        for process in &self.processes {
            let Some(representative_tokens) = parse_process_expression(&process.expression) else {
                continue;
            };
            if let Some(permutation) =
                side_preserving_process_permutation(&representative_tokens, &requested_tokens)
            {
                let mut external_pdgs = vec![0; process.external_pdgs.len()];
                for (representative_index, public_index) in permutation.iter().copied().enumerate()
                {
                    external_pdgs[public_index] = process.external_pdgs[representative_index];
                }
                permutation_matches.push(ArtifactSelection {
                    process: process.clone(),
                    // An inferred expression has no persisted alias identity;
                    // retain the representative stable process id.
                    requested_id: process.id.clone(),
                    alias: None,
                    public_expression: requested_tokens.canonical.clone(),
                    external_pdgs,
                    external_permutation: permutation,
                    inferred_permutation: true,
                    process_group: None,
                });
            } else if process_tokens_are_permutation_equivalent_ignoring_sides(
                &representative_tokens,
                &requested_tokens,
            ) {
                crossing_only_matches.push(process.id.as_str());
            }
        }
        if permutation_matches.len() > 1 {
            // Multiprocess artifacts commonly retain both incoming orderings
            // of the same physical channel. Prefer the representative that
            // already agrees with the requested ordering in the most slots;
            // only genuinely tied representatives remain ambiguous.
            let minimum_displacement = permutation_matches
                .iter()
                .map(|selection| permutation_displacement(&selection.external_permutation))
                .min()
                .expect("more than one permutation match");
            permutation_matches.retain(|selection| {
                permutation_displacement(&selection.external_permutation) == minimum_displacement
            });
        }
        match permutation_matches.len() {
            1 => return Ok(permutation_matches.pop().expect("one permutation match")),
            count if count > 1 => {
                let mut matching_ids = permutation_matches
                    .iter()
                    .map(|selection| selection.process.id.as_str())
                    .collect::<Vec<_>>();
                matching_ids.sort_unstable();
                matching_ids.dedup();
                return Err(RusticolError::selector(format!(
                    "process expression {selected_id:?} is permutation-ambiguous; select one of these stable ids: {}",
                    matching_ids.join(", ")
                )));
            }
            _ => {}
        }
        if let Some(groups) = groups()?
            && let Some(selection) = groups.select(self, selected_id, Some(&requested_tokens))?
        {
            return Ok(selection);
        }
        if !crossing_only_matches.is_empty() {
            crossing_only_matches.sort_unstable();
            crossing_only_matches.dedup();
            return Err(RusticolError::selector(format!(
                "process expression {selected_id:?} would move a particle across the '>' boundary; only permutations within the incoming and outgoing sides are supported (matching stable ids: {})",
                crossing_only_matches.join(", ")
            )));
        }
        Err(RusticolError::selector(format!(
            "unknown process id, alias id, or concrete process expression {selected_id:?}"
        )))
    }
}

fn representative_selection(process: &ArtifactProcess) -> ArtifactSelection {
    ArtifactSelection {
        process: process.clone(),
        requested_id: process.id.clone(),
        alias: None,
        public_expression: process.expression.clone(),
        external_pdgs: process.external_pdgs.clone(),
        external_permutation: (0..process.external_pdgs.len()).collect(),
        inferred_permutation: false,
        process_group: None,
    }
}

fn alias_selection(process: &ArtifactProcess, alias: &ProcessAlias) -> ArtifactSelection {
    ArtifactSelection {
        process: process.clone(),
        requested_id: alias.id.clone(),
        alias: Some(alias.clone()),
        public_expression: alias.expression.clone(),
        external_pdgs: alias.external_pdgs.clone(),
        external_permutation: alias.external_permutation.clone(),
        inferred_permutation: false,
        process_group: None,
    }
}

#[derive(Clone, Debug)]
struct ProcessExpressionTokens {
    initial: Vec<String>,
    final_state: Vec<String>,
    canonical: String,
}

fn parse_process_expression(expression: &str) -> Option<ProcessExpressionTokens> {
    let tokens = expression
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>();
    let separators = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (token == ">").then_some(index))
        .collect::<Vec<_>>();
    let [separator] = separators.as_slice() else {
        return None;
    };
    if *separator == 0 || *separator + 1 == tokens.len() {
        return None;
    }
    let initial = tokens[..*separator].to_vec();
    let final_state = tokens[*separator + 1..].to_vec();
    Some(ProcessExpressionTokens {
        canonical: format!("{} > {}", initial.join(" "), final_state.join(" ")),
        initial,
        final_state,
    })
}

fn side_preserving_process_permutation(
    representative: &ProcessExpressionTokens,
    public: &ProcessExpressionTokens,
) -> Option<Vec<usize>> {
    if representative.initial.len() != public.initial.len()
        || representative.final_state.len() != public.final_state.len()
    {
        return None;
    }
    let initial = deterministic_side_permutation(&representative.initial, &public.initial, 0)?;
    let final_offset = representative.initial.len();
    let final_state = deterministic_side_permutation(
        &representative.final_state,
        &public.final_state,
        final_offset,
    )?;
    Some(initial.into_iter().chain(final_state).collect())
}

/// Match repeated particles deterministically by assigning each representative
/// occurrence to the first still-unused equal public occurrence.
fn deterministic_side_permutation(
    representative: &[String],
    public: &[String],
    offset: usize,
) -> Option<Vec<usize>> {
    let mut used = vec![false; public.len()];
    representative
        .iter()
        .map(|particle| {
            let public_index = public.iter().enumerate().find_map(|(index, candidate)| {
                (!used[index] && candidate == particle).then_some(index)
            })?;
            used[public_index] = true;
            Some(offset + public_index)
        })
        .collect()
}

fn permutation_displacement(permutation: &[usize]) -> usize {
    permutation
        .iter()
        .copied()
        .enumerate()
        .filter(|(representative_index, public_index)| representative_index != public_index)
        .count()
}

fn process_tokens_are_permutation_equivalent_ignoring_sides(
    representative: &ProcessExpressionTokens,
    public: &ProcessExpressionTokens,
) -> bool {
    let mut representative = representative
        .initial
        .iter()
        .chain(&representative.final_state)
        .cloned()
        .collect::<Vec<_>>();
    let mut public = public
        .initial
        .iter()
        .chain(&public.final_state)
        .cloned()
        .collect::<Vec<_>>();
    representative.sort_unstable();
    public.sort_unstable();
    representative == public
}

fn normalize_process_expression(expression: &str) -> String {
    expression
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

const PROCESS_GROUPS_KIND: &str = "pyamplicol-process-groups";
const PROCESS_GROUPS_SCHEMA_VERSION: u64 = 1;

// Unknown keys (`analyses`, `basis`, `warning`, ...) are ignored on purpose.
#[derive(Deserialize)]
struct ProcessGroupsDocument {
    kind: String,
    schema_version: u64,
    groups: Vec<ProcessGroupRecord>,
}

#[derive(Deserialize)]
struct ProcessGroupRecord {
    representative: ProcessGroupRepresentativeRecord,
    members: Vec<ProcessGroupMemberRecord>,
}

#[derive(Deserialize)]
struct ProcessGroupRepresentativeRecord {
    process_id: String,
}

#[derive(Deserialize)]
struct ProcessGroupMemberRecord {
    process_id: String,
    process: String,
    pdgs: Vec<i32>,
    initial_count: usize,
    factor: ProcessGroupFactorRecord,
    permutation: Vec<usize>,
    conjugated: bool,
    generated: bool,
}

#[derive(Deserialize)]
struct ProcessGroupFactorRecord {
    numerator: i64,
    denominator: i64,
}

/// The non-generated members of a validated `process-groups.json`.
#[derive(Clone, Debug)]
pub(crate) struct ProcessGroups {
    members: Vec<ProcessGroupMember>,
}

#[derive(Clone, Debug)]
struct ProcessGroupMember {
    id: String,
    expression: String,
    tokens: ProcessExpressionTokens,
    /// Particle names and PDGs in the member's own order.
    particles: Vec<String>,
    pdgs: Vec<i32>,
    representative_id: String,
    /// Representative external index to member external index.
    permutation: Vec<usize>,
    factor_numerator: u64,
    factor_denominator: u64,
    conjugated: bool,
}

impl ProcessGroups {
    pub(crate) fn from_json(bytes: &[u8], manifest: &ArtifactManifest) -> RusticolResult<Self> {
        let document: ProcessGroupsDocument = serde_json::from_slice(bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse the process-groups payload: {error}"
            ))
        })?;
        if document.kind != PROCESS_GROUPS_KIND
            || document.schema_version != PROCESS_GROUPS_SCHEMA_VERSION
        {
            return Err(RusticolError::compatibility(format!(
                "unsupported process-groups payload kind {:?} schema {}",
                document.kind, document.schema_version
            )));
        }
        let stored_ids = manifest
            .processes
            .iter()
            .flat_map(|process| {
                std::iter::once(process.id.as_str())
                    .chain(process.aliases.iter().map(|alias| alias.id.as_str()))
            })
            .collect::<BTreeSet<_>>();
        let mut member_ids = BTreeSet::new();
        let mut members = Vec::new();
        for group in document.groups {
            let representative_id = group.representative.process_id;
            let process = manifest
                .processes
                .iter()
                .find(|process| process.id == representative_id)
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "process-group representative {representative_id:?} is not a stored process"
                    ))
                })?;
            let count = process.external_pdgs.len();
            let initial_count = parse_process_expression(&process.expression)
                .map(|tokens| tokens.initial.len())
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "process {representative_id:?} has no incoming and outgoing particles"
                    ))
                })?;
            let identity = group.members.first().is_some_and(|member| {
                member.process_id == representative_id
                    && member.generated
                    && !member.conjugated
                    && member.factor.numerator == member.factor.denominator
                    && member.permutation.iter().copied().eq(0..count)
            });
            if !identity {
                return Err(RusticolError::artifact(format!(
                    "process group {representative_id:?} must list its representative first with the identity relation"
                )));
            }
            for member in group.members {
                let id = member.process_id;
                validate_public_id(&id, "process-group member id")?;
                if !member_ids.insert(id.clone()) {
                    return Err(RusticolError::artifact(format!(
                        "process-group member {id:?} is listed more than once"
                    )));
                }
                if member.generated {
                    if !manifest.processes.iter().any(|process| process.id == id) {
                        return Err(RusticolError::artifact(format!(
                            "generated process-group member {id:?} is not a stored process"
                        )));
                    }
                    continue;
                }
                if stored_ids.contains(id.as_str()) {
                    return Err(RusticolError::artifact(format!(
                        "non-generated process-group member {id:?} reuses a stored process or alias id"
                    )));
                }
                let tokens = parse_process_expression(&member.process)
                    .filter(|tokens| {
                        tokens.initial.len() == initial_count
                            && tokens.initial.len() + tokens.final_state.len() == count
                    })
                    .ok_or_else(|| {
                        RusticolError::artifact(format!(
                            "process-group member {id:?} expression {:?} does not have the incoming and outgoing particle counts of {representative_id:?}",
                            member.process
                        ))
                    })?;
                if member.pdgs.len() != count || member.initial_count != initial_count {
                    return Err(RusticolError::artifact(format!(
                        "process-group member {id:?} PDGs or initial count disagree with {representative_id:?}"
                    )));
                }
                validate_permutation(
                    &member.permutation,
                    count,
                    &format!("process-group member {id:?}"),
                )?;
                if !permutation_preserves_process_sides(&member.permutation, initial_count) {
                    return Err(RusticolError::artifact(format!(
                        "process-group member {id:?} permutation moves a particle across the '>' boundary"
                    )));
                }
                let (factor_numerator, factor_denominator) = positive_rational(
                    member.factor.numerator,
                    member.factor.denominator,
                )
                .ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "process-group member {id:?} factor {}/{} is not a positive rational",
                        member.factor.numerator, member.factor.denominator
                    ))
                })?;
                members.push(ProcessGroupMember {
                    particles: member
                        .process
                        .split_whitespace()
                        .filter(|token| *token != ">")
                        .map(str::to_string)
                        .collect(),
                    id,
                    expression: member.process,
                    tokens,
                    pdgs: member.pdgs,
                    representative_id: representative_id.clone(),
                    permutation: member.permutation,
                    factor_numerator,
                    factor_denominator,
                    conjugated: member.conjugated,
                });
            }
        }
        Ok(Self { members })
    }

    /// Match a member by stable ID, exact expression, or side permutation of
    /// its expression, mirroring the precedence used for stored processes.
    fn select(
        &self,
        manifest: &ArtifactManifest,
        selected_id: &str,
        requested: Option<&ProcessExpressionTokens>,
    ) -> RusticolResult<Option<ArtifactSelection>> {
        if let Some(member) = self.members.iter().find(|member| member.id == selected_id) {
            return member.selection(manifest, None).map(Some);
        }
        let Some(requested) = requested else {
            return Ok(None);
        };
        let exact = self
            .members
            .iter()
            .filter(|member| member.tokens.canonical == requested.canonical)
            .collect::<Vec<_>>();
        match exact.as_slice() {
            [] => {}
            [member] => return member.selection(manifest, None).map(Some),
            _ => {
                return Err(ambiguous_member_selection(
                    selected_id,
                    "is ambiguous",
                    exact.iter().map(|member| member.id.as_str()),
                ));
            }
        }
        let mut matches = self
            .members
            .iter()
            .filter_map(|member| {
                side_preserving_process_permutation(&member.tokens, requested)
                    .map(|permutation| (member, permutation))
            })
            .collect::<Vec<_>>();
        if let Some(minimum) = matches
            .iter()
            .map(|(_, permutation)| permutation_displacement(permutation))
            .min()
        {
            matches.retain(|(_, permutation)| permutation_displacement(permutation) == minimum);
        }
        match matches.as_slice() {
            [] => Ok(None),
            [(member, permutation)] => member
                .selection(manifest, Some((permutation, &requested.canonical)))
                .map(Some),
            _ => Err(ambiguous_member_selection(
                selected_id,
                "is permutation-ambiguous",
                matches.iter().map(|(member, _)| member.id.as_str()),
            )),
        }
    }
}

impl ProcessGroupMember {
    /// `public` is the member-index to public-index permutation and the
    /// public expression of an inferred ordering; `None` keeps the member's.
    fn selection(
        &self,
        manifest: &ArtifactManifest,
        public: Option<(&[usize], &str)>,
    ) -> RusticolResult<ArtifactSelection> {
        let process = manifest
            .processes
            .iter()
            .find(|process| process.id == self.representative_id)
            .ok_or_else(|| {
                RusticolError::internal("process-group representative left the manifest")
            })?;
        let count = self.pdgs.len();
        let identity = (0..count).collect::<Vec<_>>();
        let (member_to_public, public_expression) =
            public.unwrap_or((&identity, self.expression.as_str()));
        let mut external_pdgs = vec![0; count];
        let mut particles = vec![String::new(); count];
        for (member_index, public_index) in member_to_public.iter().copied().enumerate() {
            external_pdgs[public_index] = self.pdgs[member_index];
            particles[public_index].clone_from(&self.particles[member_index]);
        }
        Ok(ArtifactSelection {
            process: process.clone(),
            requested_id: self.id.clone(),
            alias: None,
            public_expression: public_expression.to_string(),
            external_pdgs,
            // Representative -> member -> public.
            external_permutation: self
                .permutation
                .iter()
                .map(|member_index| member_to_public[*member_index])
                .collect(),
            inferred_permutation: public.is_some(),
            process_group: Some(ProcessGroupRelation {
                factor_numerator: self.factor_numerator,
                factor_denominator: self.factor_denominator,
                conjugated: self.conjugated,
                external_particles: particles,
            }),
        })
    }
}

fn ambiguous_member_selection<'a>(
    selected_id: &str,
    problem: &str,
    ids: impl Iterator<Item = &'a str>,
) -> RusticolError {
    let mut ids = ids.collect::<Vec<_>>();
    ids.sort_unstable();
    RusticolError::selector(format!(
        "process expression {selected_id:?} {problem}; select one of these process-group member ids: {}",
        ids.join(", ")
    ))
}

/// Reduce `numerator / denominator` and return it when it is positive.
fn positive_rational(numerator: i64, denominator: i64) -> Option<(u64, u64)> {
    if numerator == 0 || denominator == 0 || (numerator < 0) != (denominator < 0) {
        return None;
    }
    let (mut numerator, mut denominator) = (numerator.unsigned_abs(), denominator.unsigned_abs());
    let (mut left, mut right) = (numerator, denominator);
    while right != 0 {
        (left, right) = (right, left % right);
    }
    numerator /= left;
    denominator /= left;
    Some((numerator, denominator))
}

#[cfg(test)]
fn compute_artifact_id(manifest: &Value) -> RusticolResult<String> {
    let payloads = manifest
        .get("payloads")
        .and_then(Value::as_array)
        .ok_or_else(|| RusticolError::artifact("payloads must be an array"))?;
    let mut runtime_records = Vec::new();
    for (index, payload) in payloads.iter().enumerate() {
        let role = payload.get("role").and_then(Value::as_str).ok_or_else(|| {
            RusticolError::artifact(format!("payloads[{index}].role must be a non-empty string"))
        })?;
        if RUNTIME_IDENTITY_PAYLOAD_ROLES.contains(&role) {
            runtime_records.push(payload.clone());
        }
    }
    runtime_records.sort_by(|left, right| {
        left.get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .cmp(
                right
                    .get("path")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
    });
    let content = serde_json::json!({
        "kind": "pyamplicol-runtime-payload-identity",
        "schema_version": 1,
        "payloads": runtime_records,
    });
    let mut canonical = String::new();
    write_python_canonical_json(&content, &mut canonical)?;
    canonical.push('\n');
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

#[cfg(test)]
fn write_python_canonical_json(value: &Value, output: &mut String) -> RusticolResult<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(true) => output.push_str("true"),
        Value::Bool(false) => output.push_str("false"),
        Value::Number(number) => output.push_str(&python_number(number)),
        Value::String(value) => write_python_json_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_python_canonical_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_python_json_string(key, output);
                output.push(':');
                write_python_canonical_json(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

#[cfg(test)]
fn python_number(number: &serde_json::Number) -> String {
    let rendered = number.to_string();
    let Some((mantissa, exponent)) = rendered.split_once('e') else {
        return rendered;
    };
    let (sign, digits) = if let Some(digits) = exponent.strip_prefix('-') {
        ('-', digits)
    } else if let Some(digits) = exponent.strip_prefix('+') {
        ('+', digits)
    } else {
        ('+', exponent)
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

fn write_python_json_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0020}'..='\u{007e}' => output.push(character),
            character if u32::from(character) <= 0xffff => {
                let _ = write!(output, "\\u{:04x}", u32::from(character));
            }
            character => {
                let scalar = u32::from(character) - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                let _ = write!(output, "\\u{high:04x}\\u{low:04x}");
            }
        }
    }
    output.push('"');
}

fn reject_forbidden_nulls(manifest: &Value) -> RusticolResult<()> {
    if manifest
        .get("runtime")
        .and_then(Value::as_object)
        .is_some_and(|runtime| !runtime.contains_key("api_bundle_path"))
    {
        return Err(RusticolError::artifact(
            "runtime.api_bundle_path is required and must be a relative path or null",
        ));
    }
    for (collection, keys) in [
        ("payloads", &["target", "process_id"] as &[&str]),
        (
            "dependencies",
            &["content_sha256", "revision", "patch_sha256"] as &[&str],
        ),
    ] {
        let Some(items) = manifest.get(collection).and_then(Value::as_array) else {
            continue;
        };
        for (index, item) in items.iter().enumerate() {
            for key in keys {
                if item.get(*key).is_some_and(Value::is_null) {
                    return Err(RusticolError::artifact(format!(
                        "{collection}[{index}].{key} may be omitted but may not be null"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn locate_manifest(requested: &Path) -> RusticolResult<(PathBuf, PathBuf)> {
    let metadata = fs::metadata(requested).map_err(|error| {
        RusticolError::artifact(format!(
            "could not inspect artifact path {}: {error}",
            requested.display()
        ))
    })?;
    if metadata.is_file() {
        if requested.file_name().and_then(|name| name.to_str()) != Some(ARTIFACT_MANIFEST_FILE) {
            return Err(RusticolError::artifact(format!(
                "artifact manifest must be named {ARTIFACT_MANIFEST_FILE}"
            )));
        }
        let manifest = requested.canonicalize().map_err(|error| {
            RusticolError::artifact(format!(
                "could not resolve artifact manifest {}: {error}",
                requested.display()
            ))
        })?;
        let root = manifest.parent().ok_or_else(|| {
            RusticolError::artifact("artifact manifest has no containing directory")
        })?;
        return Ok((root.to_path_buf(), manifest));
    }
    if !metadata.is_dir() {
        return Err(RusticolError::security(format!(
            "artifact path {} is neither a regular file nor a directory",
            requested.display()
        )));
    }
    let root = requested.canonicalize().map_err(|error| {
        RusticolError::artifact(format!(
            "could not resolve artifact directory {}: {error}",
            requested.display()
        ))
    })?;
    let manifest = root.join(ARTIFACT_MANIFEST_FILE);
    if manifest.exists() {
        Ok((root, manifest))
    } else {
        Err(RusticolError::artifact(format!(
            "artifact directory does not contain {ARTIFACT_MANIFEST_FILE}"
        )))
    }
}

fn validate_manifest(manifest: &ArtifactManifest) -> RusticolResult<()> {
    validate_sha256(&manifest.artifact_id, "artifact_id")?;
    validate_datetime(&manifest.created_utc)?;
    validate_artifact_identity_extension(&manifest.extensions)?;
    if manifest.producer.distribution != "pyamplicol" {
        return Err(RusticolError::compatibility(format!(
            "unsupported artifact producer {:?}; expected pyamplicol",
            manifest.producer.distribution
        )));
    }
    if !compatible_distribution_version(&manifest.producer.version)
        || !compatible_distribution_version(&manifest.runtime.engine_version)
    {
        return Err(RusticolError::compatibility(format!(
            "artifact producer/runtime version {}/{} is incompatible with Rusticol {}",
            manifest.producer.version,
            manifest.runtime.engine_version,
            rusticol_package_version()
        )));
    }
    let versions = &manifest.producer.versions;
    let expected = [
        ("python API", versions.python_api, PYTHON_API_VERSION),
        ("TOML", versions.toml, TOML_SCHEMA_VERSION),
        (
            "compiled model",
            versions.compiled_model,
            COMPILED_MODEL_SCHEMA_VERSION,
        ),
        (
            "process artifact",
            versions.process_artifact,
            PROCESS_ARTIFACT_SCHEMA_VERSION,
        ),
        (
            "runtime physics",
            versions.runtime_physics,
            RUNTIME_PHYSICS_SCHEMA_VERSION,
        ),
        ("C ABI", versions.c_abi, C_ABI_VERSION),
    ];
    for (name, found, required) in expected {
        if found != required {
            return Err(RusticolError::compatibility(format!(
                "artifact {name} version {found} is incompatible with required version {required}; regenerate the artifact"
            )));
        }
    }
    if manifest.runtime.engine != "rusticol" {
        return Err(RusticolError::compatibility(format!(
            "unsupported runtime engine {:?}",
            manifest.runtime.engine
        )));
    }
    validate_target(&manifest.producer.target, "producer")?;
    for payload in &manifest.payloads {
        if let Some(target) = &payload.target {
            validate_payload_target(&manifest.producer.target, target, &payload.path)?;
            validate_target(target, &format!("payload {}", payload.path))?;
        }
    }
    if let Some(revision) = &manifest.producer.git_revision
        && (revision.len() != 40
            || !revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()))
    {
        return Err(RusticolError::artifact(
            "producer.git_revision must be 40 lowercase hexadecimal characters",
        ));
    }
    if let Some(native_inputs) = &manifest.producer.native_build_inputs_sha256 {
        validate_sha256(native_inputs, "producer.native_build_inputs_sha256")?;
    }
    validate_sha256(&manifest.model.content_sha256, "model content_sha256")?;
    if manifest.model.name.is_empty()
        || manifest
            .model
            .restriction
            .as_ref()
            .is_some_and(String::is_empty)
    {
        return Err(RusticolError::artifact(
            "model name and any model restriction must be non-empty",
        ));
    }
    if manifest.model.compiled_schema_version != COMPILED_MODEL_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "compiled model schema {} is incompatible with required schema {}",
            manifest.model.compiled_schema_version, COMPILED_MODEL_SCHEMA_VERSION
        )));
    }
    if manifest.configuration.toml_schema_version != TOML_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "configuration TOML schema {} is incompatible with required schema {}",
            manifest.configuration.toml_schema_version, TOML_SCHEMA_VERSION
        )));
    }
    validate_relative_path(
        &manifest.configuration.requested_path,
        "requested configuration path",
    )?;
    validate_relative_path(
        &manifest.configuration.effective_path,
        "effective configuration path",
    )?;
    for (index, adjustment) in manifest.configuration.adjustments.iter().enumerate() {
        if adjustment.path.is_empty() || adjustment.reason.is_empty() {
            return Err(RusticolError::artifact(format!(
                "configuration adjustment {index} requires non-empty path and reason"
            )));
        }
    }
    validate_relative_path(
        &manifest.runtime.evaluator_manifest_path,
        "evaluator manifest path",
    )?;
    let runtime_capabilities = validate_runtime_capabilities(
        &manifest.runtime.required_runtime_capabilities,
        "runtime.required_runtime_capabilities",
    )?;
    if let Some(path) = &manifest.runtime.api_bundle_path {
        validate_relative_path(path, "API bundle path")?;
    }
    if manifest.processes.is_empty() {
        return Err(RusticolError::artifact(
            "artifact must contain at least one process",
        ));
    }
    match manifest.kind {
        ArtifactKind::PyamplicolProcess if manifest.processes.len() != 1 => {
            return Err(RusticolError::artifact(
                "pyamplicol-process artifacts must contain exactly one process",
            ));
        }
        _ => {}
    }
    let mut public_ids = BTreeSet::new();
    let mut process_capabilities = BTreeSet::new();
    for process in &manifest.processes {
        validate_public_id(&process.id, "process id")?;
        validate_relative_path(&process.physics_path, "runtime physics path")?;
        let process_expression = parse_process_expression(&process.expression);
        if process.expression.is_empty() || process.external_pdgs.len() < 3 {
            return Err(RusticolError::artifact(format!(
                "process {} has an empty expression or fewer than three external particles",
                process.id
            )));
        }
        let process_expression = process_expression.ok_or_else(|| {
            RusticolError::artifact(format!(
                "process {:?} expression must contain one '>' with particles on both sides",
                process.id
            ))
        })?;
        if process_expression.initial.len() + process_expression.final_state.len()
            != process.external_pdgs.len()
        {
            return Err(RusticolError::artifact(format!(
                "process {:?} expression particle count does not match external_pdgs",
                process.id
            )));
        }
        if !matches!(process.color_accuracy.as_str(), "lc" | "nlc" | "full") {
            return Err(RusticolError::artifact(format!(
                "process {} has unsupported color accuracy {:?}",
                process.id, process.color_accuracy
            )));
        }
        process_capabilities.extend(validate_runtime_capabilities(
            &process.required_runtime_capabilities,
            &format!("process {:?}.required_runtime_capabilities", process.id),
        )?);
        if !public_ids.insert(&process.id) {
            return Err(RusticolError::artifact(format!(
                "duplicate public process id {:?}",
                process.id
            )));
        }
        for alias in &process.aliases {
            validate_public_id(&alias.id, "process alias id")?;
            if alias.expression.is_empty() || !public_ids.insert(&alias.id) {
                return Err(RusticolError::artifact(format!(
                    "duplicate or invalid process alias id {:?}",
                    alias.id
                )));
            }
            validate_permutation(
                &alias.external_permutation,
                process.external_pdgs.len(),
                &format!("alias {:?}", alias.id),
            )?;
            let alias_expression =
                parse_process_expression(&alias.expression).ok_or_else(|| {
                    RusticolError::artifact(format!(
                        "alias {:?} expression must contain one '>' with particles on both sides",
                        alias.id
                    ))
                })?;
            if alias_expression.initial.len() != process_expression.initial.len()
                || alias_expression.final_state.len() != process_expression.final_state.len()
                || !permutation_preserves_process_sides(
                    &alias.external_permutation,
                    process_expression.initial.len(),
                )
            {
                return Err(RusticolError::artifact(format!(
                    "alias {:?} may only permute particles within the incoming and outgoing sides",
                    alias.id
                )));
            }
            let mut expected_external_pdgs = vec![0; process.external_pdgs.len()];
            for (representative_index, alias_index) in
                alias.external_permutation.iter().copied().enumerate()
            {
                expected_external_pdgs[alias_index] = process.external_pdgs[representative_index];
            }
            if alias.external_pdgs != expected_external_pdgs {
                return Err(RusticolError::artifact(format!(
                    "alias {:?} external_pdgs {:?} does not match external_permutation {:?}; expected {:?}",
                    alias.id,
                    alias.external_pdgs,
                    alias.external_permutation,
                    expected_external_pdgs,
                )));
            }
        }
    }
    if runtime_capabilities != process_capabilities {
        return Err(RusticolError::artifact(
            "runtime.required_runtime_capabilities must equal the union of process capability declarations",
        ));
    }
    validate_portable_runtime_capabilities(&manifest.producer.target, &runtime_capabilities)?;
    if let Some(default) = &manifest.default_process_id
        && !public_ids.contains(default)
    {
        return Err(RusticolError::artifact(format!(
            "default process id {default:?} does not identify a process or alias"
        )));
    }
    if manifest.payloads.is_empty() {
        return Err(RusticolError::artifact(
            "artifact must declare at least one payload",
        ));
    }
    let mut dependencies = BTreeSet::new();
    for dependency in &manifest.dependencies {
        if dependency.name.is_empty()
            || dependency.version.is_empty()
            || dependency.source.is_empty()
            || dependency.license.is_empty()
            || !dependencies.insert(dependency.name.to_ascii_lowercase())
        {
            return Err(RusticolError::artifact(format!(
                "invalid or duplicate dependency {:?}",
                dependency.name
            )));
        }
        for (name, value) in [
            ("content_sha256", dependency.content_sha256.as_deref()),
            ("patch_sha256", dependency.patch_sha256.as_deref()),
        ] {
            if let Some(value) = value {
                validate_sha256(value, &format!("dependency {} {name}", dependency.name))?;
            }
        }
        if dependency.revision.as_ref().is_some_and(String::is_empty) {
            return Err(RusticolError::artifact(format!(
                "dependency {} revision must be non-empty when present",
                dependency.name
            )));
        }
    }
    Ok(())
}

fn validate_artifact_identity_extension(
    extensions: &BTreeMap<String, Value>,
) -> RusticolResult<()> {
    let Some(policy) = extensions
        .get(ARTIFACT_IDENTITY_EXTENSION)
        .and_then(Value::as_object)
    else {
        return Err(RusticolError::compatibility(
            "artifact predates the required runtime-payload identity contract; \
             regenerate it with the current pyAmpliCol",
        ));
    };
    let valid = policy.len() == 2
        && policy.get("kind").and_then(Value::as_str) == Some(ARTIFACT_IDENTITY_KIND)
        && policy.get("schema_version").and_then(Value::as_u64)
            == Some(ARTIFACT_IDENTITY_SCHEMA_VERSION);
    if !valid {
        return Err(RusticolError::compatibility(
            "artifact uses an unsupported artifact identity contract; \
             regenerate it with the current pyAmpliCol",
        ));
    }
    Ok(())
}

fn validate_artifact_identity_header(manifest: &Value) -> RusticolResult<()> {
    let Some(extensions) = manifest.get("extensions").and_then(Value::as_object) else {
        return Err(RusticolError::compatibility(
            "artifact predates the required runtime-payload identity contract; \
             regenerate it with the current pyAmpliCol",
        ));
    };
    let ordered = extensions
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    validate_artifact_identity_extension(&ordered)
}

fn compatible_distribution_version(version: &str) -> bool {
    #[cfg(test)]
    if std::env::var_os("RUSTICOL_TEST_ALLOW_VERSION_MISMATCH").is_some() {
        return true;
    }
    canonical_distribution_version(version)
        == canonical_distribution_version(rusticol_package_version())
}

fn canonical_distribution_version(version: &str) -> String {
    let canonical = version.replace("-dev.", ".dev");
    // Candidate build hashes identify native builds, not process formats.
    // Explicit schema, ABI, and evaluator-storage versions are checked when
    // loading the output; a runtime fix must not require process regeneration.
    if let Some((base, fingerprint)) = canonical.split_once("+candidate.")
        && base.contains(".dev")
        && fingerprint.len() == 12
        && fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return base.to_owned();
    }
    canonical
}

fn validate_references(
    manifest: &ArtifactManifest,
    payloads: &BTreeMap<String, Payload>,
) -> RusticolResult<()> {
    let process_ids = manifest
        .processes
        .iter()
        .map(|process| process.id.as_str())
        .collect::<BTreeSet<_>>();
    require_payload_role(
        payloads,
        &manifest.configuration.requested_path,
        PayloadRole::ConfigurationRequested,
        None,
    )?;
    require_payload_role(
        payloads,
        &manifest.configuration.effective_path,
        PayloadRole::ConfigurationEffective,
        None,
    )?;
    require_payload_role(
        payloads,
        &manifest.runtime.evaluator_manifest_path,
        PayloadRole::EvaluatorManifest,
        None,
    )?;
    for process in &manifest.processes {
        require_payload_role(
            payloads,
            &process.physics_path,
            PayloadRole::RuntimePhysics,
            Some(&process.id),
        )?;
    }
    for payload in payloads
        .values()
        .filter(|payload| payload.role == PayloadRole::StructuralSourceProof)
    {
        if let Some(process_id) = payload.process_id.as_deref()
            && !process_ids.contains(process_id)
        {
            return Err(RusticolError::artifact(format!(
                "structural-source-proof payload {:?} belongs to unknown process {process_id:?}",
                payload.path
            )));
        }
    }
    if let Some(api_path) = &manifest.runtime.api_bundle_path {
        let prefix = format!("{}/", api_path.trim_end_matches('/'));
        if !payloads.keys().any(|path| path.starts_with(&prefix)) {
            return Err(RusticolError::artifact(format!(
                "API bundle path {api_path:?} contains no declared payload"
            )));
        }
    }
    Ok(())
}

fn require_payload_role(
    payloads: &BTreeMap<String, Payload>,
    path: &str,
    role: PayloadRole,
    process_id: Option<&str>,
) -> RusticolResult<()> {
    let payload = payloads.get(path).ok_or_else(|| {
        RusticolError::security(format!(
            "referenced artifact path {path:?} is not a declared payload"
        ))
    })?;
    if payload.role != role {
        return Err(RusticolError::security(format!(
            "referenced payload {path:?} has role {:?}, expected {:?}",
            payload.role, role
        )));
    }
    if let (Some(expected), Some(found)) = (process_id, payload.process_id.as_deref())
        && expected != found
    {
        return Err(RusticolError::artifact(format!(
            "payload {path:?} belongs to process {found:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn metadata_is_executable(metadata: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        false
    }
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn read_declared_payload(root: &Path, payload: &Payload) -> RusticolResult<Vec<u8>> {
    validate_payload_declaration(payload)?;
    let mut file = open_checked_payload(root, payload)?;
    let expected_size = usize::try_from(payload.size_bytes).map_err(|_| {
        RusticolError::artifact(format!(
            "payload {:?} is too large for this platform",
            payload.path
        ))
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(expected_size).map_err(|error| {
        RusticolError::artifact(format!(
            "could not allocate {} bytes for payload {:?}: {error}",
            payload.size_bytes, payload.path
        ))
    })?;
    file.read_to_end(&mut bytes).map_err(|error| {
        RusticolError::artifact(format!(
            "could not read payload {:?}: {error}",
            payload.path
        ))
    })?;
    if bytes.len() != expected_size {
        return Err(RusticolError::integrity(format!(
            "payload {:?} changed while being read: expected {} bytes, read {}",
            payload.path,
            payload.size_bytes,
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn validate_payload_declaration(payload: &Payload) -> RusticolResult<()> {
    validate_sha256(&payload.sha256, &format!("payload {} sha256", payload.path))?;
    if payload.media_type.is_empty() {
        return Err(RusticolError::artifact(format!(
            "payload {:?} has an empty media type",
            payload.path
        )));
    }
    if let Some(process_id) = &payload.process_id {
        validate_public_id(process_id, "payload process id")?;
    }
    if payload.role == PayloadRole::EvaluatorState && payload.target.is_none() {
        return Err(RusticolError::artifact(format!(
            "evaluator-state payload {:?} is missing required target metadata",
            payload.path
        )));
    }
    if payload.role == PayloadRole::StructuralSourceProof {
        let process_id = payload.process_id.as_deref().ok_or_else(|| {
            RusticolError::artifact(format!(
                "structural-source-proof payload {:?} is missing required process id",
                payload.path
            ))
        })?;
        let expected_path = format!("processes/{process_id}/structural-source-proof.json");
        if payload.path != expected_path {
            return Err(RusticolError::artifact(format!(
                "structural-source-proof payload {:?} must use exact path {expected_path:?}",
                payload.path
            )));
        }
        if payload.media_type != "application/json" {
            return Err(RusticolError::artifact(format!(
                "structural-source-proof payload {:?} must use media type application/json",
                payload.path
            )));
        }
        if payload.target.is_some() {
            return Err(RusticolError::artifact(format!(
                "structural-source-proof payload {:?} may not declare target metadata",
                payload.path
            )));
        }
    }
    if payload.executable
        && !matches!(
            payload.role,
            PayloadRole::EvaluatorState | PayloadRole::ApiSource | PayloadRole::ApiBuildFile
        )
    {
        return Err(RusticolError::security(format!(
            "payload {:?} has role {:?}, which may not be executable",
            payload.path, payload.role
        )));
    }
    Ok(())
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn open_checked_payload(root: &Path, payload: &Payload) -> RusticolResult<File> {
    let path = root.join(&payload.path);
    reject_symlink_chain(&path)?;
    let canonical = path.canonicalize().map_err(|error| {
        RusticolError::security(format!(
            "could not resolve payload {:?}: {error}",
            payload.path
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(RusticolError::security(format!(
            "payload {:?} escapes the artifact root",
            payload.path
        )));
    }
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt as _;
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&path)
    };
    #[cfg(not(unix))]
    let file = OpenOptions::new().read(true).open(&path);
    let file = file.map_err(|error| {
        RusticolError::artifact(format!(
            "could not open payload {:?}: {error}",
            payload.path
        ))
    })?;
    let metadata = file.metadata().map_err(|error| {
        RusticolError::security(format!(
            "could not inspect payload {:?}: {error}",
            payload.path
        ))
    })?;
    if !metadata.is_file() {
        return Err(RusticolError::security(format!(
            "payload {:?} is not a regular file",
            payload.path
        )));
    }
    if metadata.len() != payload.size_bytes {
        return Err(RusticolError::integrity(format!(
            "payload {:?} has size {}, expected {}",
            payload.path,
            metadata.len(),
            payload.size_bytes
        )));
    }
    #[cfg(unix)]
    {
        let executable = metadata_is_executable(&metadata);
        if executable != payload.executable {
            return Err(RusticolError::security(format!(
                "payload {:?} executable mode is {}, but the manifest declares {}",
                payload.path, executable, payload.executable
            )));
        }
    }
    reject_symlink_chain(&path)?;
    let path_metadata = fs::symlink_metadata(&path).map_err(|error| {
        RusticolError::security(format!(
            "could not re-inspect payload {:?}: {error}",
            payload.path
        ))
    })?;
    if path_metadata.file_type().is_symlink() || !path_metadata.file_type().is_file() {
        return Err(RusticolError::security(format!(
            "payload {:?} is not a regular non-symlink file",
            payload.path
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if metadata.dev() != path_metadata.dev() || metadata.ino() != path_metadata.ino() {
            return Err(RusticolError::security(format!(
                "payload {:?} was replaced while being opened",
                payload.path
            )));
        }
    }
    Ok(file)
}

fn validate_relative_path(value: &str, description: &str) -> RusticolResult<()> {
    if value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.contains('\0')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || Path::new(value).is_absolute()
    {
        return Err(RusticolError::security(format!(
            "{description} {value:?} is not a normalized confined relative path"
        )));
    }
    Ok(())
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn reject_symlink_chain(path: &Path) -> RusticolResult<()> {
    let mut ancestors = path.ancestors().collect::<Vec<_>>();
    ancestors.reverse();
    for ancestor in ancestors {
        let Ok(metadata) = fs::symlink_metadata(ancestor) else {
            continue;
        };
        if metadata.file_type().is_symlink() {
            return Err(RusticolError::security(format!(
                "artifact path {} contains a symlink at {}",
                path.display(),
                ancestor.display()
            )));
        }
    }
    Ok(())
}

fn validate_sha256(value: &str, description: &str) -> RusticolResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(RusticolError::artifact(format!(
            "{description} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn parse_sha256(value: &str, description: &str) -> RusticolResult<[u8; 32]> {
    validate_sha256(value, description)?;
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
        _ => unreachable!("SHA-256 validation precedes hexadecimal decoding"),
    }
}

fn validate_public_id(value: &str, description: &str) -> RusticolResult<()> {
    let valid = !value.is_empty()
        && value.len() <= 255
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric()
                || (index > 0 && matches!(byte, b'.' | b'_' | b':' | b'+' | b',' | b'~' | b'-'))
        });
    if !valid {
        return Err(RusticolError::artifact(format!(
            "invalid {description} {value:?}"
        )));
    }
    Ok(())
}

fn validate_datetime(value: &str) -> RusticolResult<()> {
    let bytes = value.as_bytes();
    let digits = |start: usize, length: usize| -> Option<u32> {
        let slice = bytes.get(start..start + length)?;
        if !slice.iter().all(u8::is_ascii_digit) {
            return None;
        }
        slice.iter().try_fold(0_u32, |value, digit| {
            value.checked_mul(10)?.checked_add(u32::from(*digit - b'0'))
        })
    };
    let year = digits(0, 4);
    let month = digits(5, 2);
    let day = digits(8, 2);
    let hour = digits(11, 2);
    let minute = digits(14, 2);
    let second = digits(17, 2);
    let date_valid = match (year, month, day) {
        (Some(year), Some(month @ 1..=12), Some(day)) => {
            let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
            let maximum = match month {
                2 if leap => 29,
                2 => 28,
                4 | 6 | 9 | 11 => 30,
                _ => 31,
            };
            (1..=maximum).contains(&day)
        }
        _ => false,
    };
    let time_valid = matches!(hour, Some(0..=23))
        && matches!(minute, Some(0..=59))
        && matches!(second, Some(0..=60));
    let separators_valid = bytes.get(4) == Some(&b'-')
        && bytes.get(7) == Some(&b'-')
        && matches!(bytes.get(10), Some(b'T' | b't'))
        && bytes.get(13) == Some(&b':')
        && bytes.get(16) == Some(&b':');
    let mut offset = 19;
    if bytes.get(offset) == Some(&b'.') {
        offset += 1;
        let fractional_start = offset;
        while bytes.get(offset).is_some_and(u8::is_ascii_digit) {
            offset += 1;
        }
        if offset == fractional_start {
            offset = bytes.len() + 1;
        }
    }
    let zone_valid = match bytes.get(offset) {
        Some(b'Z' | b'z') => offset + 1 == bytes.len(),
        Some(b'+' | b'-') => {
            offset + 6 == bytes.len()
                && bytes.get(offset + 3) == Some(&b':')
                && matches!(digits(offset + 1, 2), Some(0..=23))
                && matches!(digits(offset + 4, 2), Some(0..=59))
        }
        _ => false,
    };
    let valid = bytes.len() >= 20 && date_valid && time_valid && separators_valid && zone_valid;
    if !valid {
        return Err(RusticolError::artifact(format!(
            "created_utc {value:?} is not an RFC 3339 date-time"
        )));
    }
    Ok(())
}

fn validate_permutation(values: &[usize], size: usize, owner: &str) -> RusticolResult<()> {
    if values.len() != size {
        return Err(RusticolError::artifact(format!(
            "{owner} permutation has length {}, expected {size}",
            values.len()
        )));
    }
    let found = values.iter().copied().collect::<BTreeSet<_>>();
    let expected = (0..size).collect::<BTreeSet<_>>();
    if found != expected {
        return Err(RusticolError::artifact(format!(
            "{owner} permutation is not a complete zero-based permutation"
        )));
    }
    Ok(())
}

fn permutation_preserves_process_sides(values: &[usize], initial_count: usize) -> bool {
    values
        .iter()
        .copied()
        .enumerate()
        .all(|(representative_index, public_index)| {
            (representative_index < initial_count) == (public_index < initial_count)
        })
}

fn validate_runtime_capabilities(
    values: &[String],
    description: &str,
) -> RusticolResult<BTreeSet<String>> {
    if values.is_empty() {
        return Err(RusticolError::artifact(format!(
            "{description} must contain at least one capability"
        )));
    }
    let capabilities = values.iter().cloned().collect::<BTreeSet<_>>();
    if capabilities.len() != values.len() {
        return Err(RusticolError::artifact(format!(
            "{description} must not contain duplicates"
        )));
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(RusticolError::artifact(format!(
            "{description} must be sorted"
        )));
    }
    let known = [
        RuntimeCapability::CompiledColorContractionWalshC2kV1,
        RuntimeCapability::CompiledColorContractionWalshV1,
        RuntimeCapability::CompiledColorTopologyLanesV1,
        RuntimeCapability::CompiledHelicityDualLaneV1,
        RuntimeCapability::CompiledHelicityPrimaryRecurrenceV1,
        RuntimeCapability::CompiledHelicitySelectorUnionV1,
        RuntimeCapability::CompiledPlaneArenaV1,
        RuntimeCapability::CompiledRuntimeSelectorsV1,
        RuntimeCapability::EagerDagComplexF64V1,
        RuntimeCapability::EagerDirectArenaV1,
        RuntimeCapability::EagerRuntimeLayoutComplexF64V1,
        RuntimeCapability::EagerLcTopologyReplayComplexF64V1,
        RuntimeCapability::RecurrenceRuntimeComplexF64V1,
        RuntimeCapability::RecurrenceLcColorV1,
        RuntimeCapability::RecurrenceContractedColorV1,
        RuntimeCapability::RecurrenceHelicitySelectorCompanionV2,
        RuntimeCapability::SymmetricGroupFftColorContractionV1,
        RuntimeCapability::OnTheFlyRuntimeComplexF64V1,
        RuntimeCapability::OnTheFlyContractedColorV1,
        RuntimeCapability::OnTheFlyLcColorV1,
        RuntimeCapability::SymjitApplicationComplexF64V1,
        RuntimeCapability::SymbolicaLegacyJitContainerComplexF64V1,
        RuntimeCapability::SymbolicaCompiledCppComplexF64V1,
        RuntimeCapability::SymbolicaCompiledAsmComplexF64V1,
    ]
    .map(RuntimeCapability::as_str)
    .into_iter()
    .collect::<BTreeSet<_>>();
    let unknown = capabilities
        .iter()
        .filter(|capability| !known.contains(capability.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(RusticolError::artifact(format!(
            "{description} contains unsupported capabilities: {}",
            unknown.join(", ")
        )));
    }
    Ok(capabilities)
}

fn validate_target(target: &Target, description: &str) -> RusticolResult<()> {
    let current = current_target_triple();
    if !SUPPORTED_ARTIFACT_TARGETS.contains(&current) {
        return Err(RusticolError::compatibility(format!(
            "Rusticol process artifacts are not supported on runtime target {current:?}"
        )));
    }
    let required = normalized_cpu_features(target, description)?;
    if target.triple == PORTABLE_64LE_ARTIFACT_TARGET {
        if usize::BITS != 64 || !cfg!(target_endian = "little") {
            return Err(RusticolError::compatibility(format!(
                "{description} target {PORTABLE_64LE_ARTIFACT_TARGET:?} requires a 64-bit little-endian runtime"
            )));
        }
        if !required.is_empty() {
            return Err(RusticolError::compatibility(format!(
                "{description} target {PORTABLE_64LE_ARTIFACT_TARGET:?} must not require CPU features"
            )));
        }
        return Ok(());
    }
    if target.triple != current {
        return Err(RusticolError::compatibility(format!(
            "{description} target {:?} is incompatible with runtime target {current:?}",
            target.triple
        )));
    }
    let available = detected_cpu_features().into_iter().collect::<BTreeSet<_>>();
    let unavailable = required.difference(&available).cloned().collect::<Vec<_>>();
    if !unavailable.is_empty() {
        return Err(RusticolError::compatibility(format!(
            "{description} requires unavailable CPU features {unavailable:?}"
        )));
    }
    Ok(())
}

fn validate_portable_runtime_capabilities(
    target: &Target,
    capabilities: &BTreeSet<String>,
) -> RusticolResult<()> {
    if target.triple != PORTABLE_64LE_ARTIFACT_TARGET {
        return Ok(());
    }
    let forbidden = [
        RuntimeCapability::SymbolicaLegacyJitContainerComplexF64V1.as_str(),
        RuntimeCapability::SymbolicaCompiledCppComplexF64V1.as_str(),
        RuntimeCapability::SymbolicaCompiledAsmComplexF64V1.as_str(),
    ];
    if forbidden
        .iter()
        .any(|capability| capabilities.contains(*capability))
    {
        return Err(RusticolError::compatibility(
            "portable-64le process artifacts cannot contain C++, ASM, or legacy JIT evaluators; those evaluator families remain target-specific",
        ));
    }
    Ok(())
}

fn validate_payload_target(
    producer: &Target,
    payload: &Target,
    payload_path: &str,
) -> RusticolResult<()> {
    if payload.triple != producer.triple {
        return Err(RusticolError::compatibility(format!(
            "payload {payload_path:?} target {:?} does not match producer target {:?}",
            payload.triple, producer.triple
        )));
    }
    let producer_features = normalized_cpu_features(producer, "producer")?;
    let payload_features = normalized_cpu_features(payload, &format!("payload {payload_path}"))?;
    if payload_features != producer_features {
        return Err(RusticolError::compatibility(format!(
            "payload {payload_path:?} CPU features {payload_features:?} do not match producer CPU features {producer_features:?}"
        )));
    }
    Ok(())
}

fn normalized_cpu_features(target: &Target, description: &str) -> RusticolResult<BTreeSet<String>> {
    let mut features = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for feature in &target.cpu_features {
        let canonical = !feature.is_empty()
            && feature.bytes().enumerate().all(|(index, byte)| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || (index > 0 && matches!(byte, b'.' | b'-'))
            });
        if !canonical {
            return Err(RusticolError::artifact(format!(
                "{description} target CPU feature {feature:?} is not a canonical feature ID"
            )));
        }
        if previous.is_some_and(|value| value >= feature.as_str()) {
            return Err(RusticolError::artifact(format!(
                "{description} target CPU features must be sorted and unique"
            )));
        }
        previous = Some(feature);
        features.insert(feature.clone());
    }
    Ok(features)
}

/// Return the current Rusticol target and every CPU feature it can verify at runtime.
///
/// An empty feature list on an artifact means the architecture's baseline ISA. Native
/// evaluator producers use this detected, canonical list as a conservative requirement.
pub fn runtime_target_info() -> Target {
    Target {
        triple: current_target_triple().to_string(),
        cpu_features: detected_cpu_features(),
    }
}

fn current_target_triple() -> &'static str {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "aarch64-apple-darwin"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        "x86_64-apple-darwin"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    {
        "x86_64-unknown-linux-gnu"
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
    )))]
    {
        "unsupported-target"
    }
}

fn detected_cpu_features() -> Vec<String> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        let detected = [
            ("adx", std::is_x86_feature_detected!("adx")),
            ("aes", std::is_x86_feature_detected!("aes")),
            ("avx", std::is_x86_feature_detected!("avx")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("avx512bf16", std::is_x86_feature_detected!("avx512bf16")),
            (
                "avx512bitalg",
                std::is_x86_feature_detected!("avx512bitalg"),
            ),
            ("avx512bw", std::is_x86_feature_detected!("avx512bw")),
            ("avx512cd", std::is_x86_feature_detected!("avx512cd")),
            ("avx512dq", std::is_x86_feature_detected!("avx512dq")),
            ("avx512f", std::is_x86_feature_detected!("avx512f")),
            ("avx512ifma", std::is_x86_feature_detected!("avx512ifma")),
            ("avx512vbmi", std::is_x86_feature_detected!("avx512vbmi")),
            ("avx512vbmi2", std::is_x86_feature_detected!("avx512vbmi2")),
            ("avx512vl", std::is_x86_feature_detected!("avx512vl")),
            ("avx512vnni", std::is_x86_feature_detected!("avx512vnni")),
            (
                "avx512vpopcntdq",
                std::is_x86_feature_detected!("avx512vpopcntdq"),
            ),
            ("bmi1", std::is_x86_feature_detected!("bmi1")),
            ("bmi2", std::is_x86_feature_detected!("bmi2")),
            ("cmpxchg16b", std::is_x86_feature_detected!("cmpxchg16b")),
            ("f16c", std::is_x86_feature_detected!("f16c")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("fxsr", std::is_x86_feature_detected!("fxsr")),
            ("gfni", std::is_x86_feature_detected!("gfni")),
            ("lzcnt", std::is_x86_feature_detected!("lzcnt")),
            ("movbe", std::is_x86_feature_detected!("movbe")),
            ("pclmulqdq", std::is_x86_feature_detected!("pclmulqdq")),
            ("popcnt", std::is_x86_feature_detected!("popcnt")),
            ("rdrand", std::is_x86_feature_detected!("rdrand")),
            ("rdseed", std::is_x86_feature_detected!("rdseed")),
            ("rtm", std::is_x86_feature_detected!("rtm")),
            ("sha", std::is_x86_feature_detected!("sha")),
            ("sse", std::is_x86_feature_detected!("sse")),
            ("sse2", std::is_x86_feature_detected!("sse2")),
            ("sse3", std::is_x86_feature_detected!("sse3")),
            ("sse4.1", std::is_x86_feature_detected!("sse4.1")),
            ("sse4.2", std::is_x86_feature_detected!("sse4.2")),
            ("ssse3", std::is_x86_feature_detected!("ssse3")),
            ("vaes", std::is_x86_feature_detected!("vaes")),
            ("vpclmulqdq", std::is_x86_feature_detected!("vpclmulqdq")),
            ("xsave", std::is_x86_feature_detected!("xsave")),
            ("xsavec", std::is_x86_feature_detected!("xsavec")),
            ("xsaveopt", std::is_x86_feature_detected!("xsaveopt")),
            ("xsaves", std::is_x86_feature_detected!("xsaves")),
        ];
        features.extend(
            detected
                .into_iter()
                .filter(|(_, available)| *available)
                .map(|(name, _)| name.to_string()),
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        let detected = [
            ("aes", std::arch::is_aarch64_feature_detected!("aes")),
            ("bf16", std::arch::is_aarch64_feature_detected!("bf16")),
            ("bti", std::arch::is_aarch64_feature_detected!("bti")),
            ("crc", std::arch::is_aarch64_feature_detected!("crc")),
            ("dit", std::arch::is_aarch64_feature_detected!("dit")),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("dpb", std::arch::is_aarch64_feature_detected!("dpb")),
            ("dpb2", std::arch::is_aarch64_feature_detected!("dpb2")),
            ("f32mm", std::arch::is_aarch64_feature_detected!("f32mm")),
            ("f64mm", std::arch::is_aarch64_feature_detected!("f64mm")),
            ("fcma", std::arch::is_aarch64_feature_detected!("fcma")),
            ("fhm", std::arch::is_aarch64_feature_detected!("fhm")),
            ("flagm", std::arch::is_aarch64_feature_detected!("flagm")),
            ("fp", std::arch::is_aarch64_feature_detected!("fp")),
            ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
            (
                "frintts",
                std::arch::is_aarch64_feature_detected!("frintts"),
            ),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
            ("jsconv", std::arch::is_aarch64_feature_detected!("jsconv")),
            ("lse", std::arch::is_aarch64_feature_detected!("lse")),
            ("lse2", std::arch::is_aarch64_feature_detected!("lse2")),
            ("mte", std::arch::is_aarch64_feature_detected!("mte")),
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            ("paca", std::arch::is_aarch64_feature_detected!("paca")),
            ("pacg", std::arch::is_aarch64_feature_detected!("pacg")),
            ("rand", std::arch::is_aarch64_feature_detected!("rand")),
            ("rcpc", std::arch::is_aarch64_feature_detected!("rcpc")),
            ("rcpc2", std::arch::is_aarch64_feature_detected!("rcpc2")),
            ("rdm", std::arch::is_aarch64_feature_detected!("rdm")),
            ("sb", std::arch::is_aarch64_feature_detected!("sb")),
            ("sha2", std::arch::is_aarch64_feature_detected!("sha2")),
            ("sha3", std::arch::is_aarch64_feature_detected!("sha3")),
            ("sm4", std::arch::is_aarch64_feature_detected!("sm4")),
            ("ssbs", std::arch::is_aarch64_feature_detected!("ssbs")),
            ("sve", std::arch::is_aarch64_feature_detected!("sve")),
            ("sve2", std::arch::is_aarch64_feature_detected!("sve2")),
            (
                "sve2-aes",
                std::arch::is_aarch64_feature_detected!("sve2-aes"),
            ),
            (
                "sve2-bitperm",
                std::arch::is_aarch64_feature_detected!("sve2-bitperm"),
            ),
            (
                "sve2-sha3",
                std::arch::is_aarch64_feature_detected!("sve2-sha3"),
            ),
            (
                "sve2-sm4",
                std::arch::is_aarch64_feature_detected!("sve2-sm4"),
            ),
            ("tme", std::arch::is_aarch64_feature_detected!("tme")),
        ];
        features.extend(
            detected
                .into_iter()
                .filter(|(_, available)| *available)
                .map(|(name, _)| name.to_string()),
        );
    }
    features.sort();
    features
}

#[cfg(test)]
#[path = "artifact_tests.rs"]
pub(crate) mod tests;
