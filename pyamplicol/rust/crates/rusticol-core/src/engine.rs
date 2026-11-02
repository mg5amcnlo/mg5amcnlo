// SPDX-License-Identifier: 0BSD

#[cfg(not(feature = "symbolica-runtime"))]
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
#[cfg(feature = "symbolica-runtime")]
use symbolica::evaluate::JITCompiledEvaluator;
#[cfg(feature = "symbolica-runtime")]
use symbolica::prelude::{
    BatchEvaluator, Complex, DoubleFloat, EvaluationDomain, ExpressionEvaluator, Float,
    JITCompilationSettings, Rational, Real, RealLike,
};

#[cfg(feature = "symbolica-runtime")]
use crate::artifact::EvaluatorPayloadSource;
use crate::artifact::EvaluatorPayloadStore;
use crate::{
    ColorComponent as PhysicsColorComponentV1, PROCESS_ARTIFACT_SCHEMA_VERSION, PayloadRole,
    ProcessPhysics as ProcessPhysicsV1, RusticolError, RusticolResult, VerifiedArtifact,
};

// Keep replay state within the useful cache working set. Larger batches make
// stage-local gather/scatter substantially slower for high-flow LC workloads.
const MAX_LC_TOPOLOGY_REPLAY_EXPANDED_POINTS: usize = 2048;
const LC_SECTOR_SELECTOR_PARAMETER: &str = "runtime.lc_sector_id";
const HELICITY_RECURRENCE_CONTRACT_VERSION: u32 = 1;
const HELICITY_RECURRENCE_KIND: &str = "pyamplicol-helicity-recurrence";
const HELICITY_RECURRENCE_PROOF_ALGORITHM: &str = "canonical-source-transition-dependency-shape-v1";
const HELICITY_MATERIALIZATION_CONTRACT_VERSION: u32 = 1;
const HELICITY_MATERIALIZATION_KIND: &str = "pyamplicol-helicity-recurrence-materialization";

type LcTopologyReplayMappings = Vec<Vec<(usize, usize)>>;

#[derive(Clone, Debug, Default)]
struct LcTopologyReplayData {
    mappings: LcTopologyReplayMappings,
    routes: Vec<Vec<LcTopologyReplaySectorRoute>>,
    materialized_sector_ids: BTreeSet<i64>,
}

#[derive(Clone, Debug)]
struct LcTopologyReplaySectorRoute {
    physical_sector_id: i64,
    materialized_sector_id: i64,
    weight: f64,
    sign: i8,
    amplitude_factor: [f64; 2],
    residual: bool,
}

impl LcTopologyReplaySectorRoute {
    fn squared_reduction_weight(&self) -> f64 {
        // The replay sign is an amplitude-level relation.  LC resolved output
        // is diagonal in the physical flow, so sign^2 = 1.  Multiplying the
        // signed factor by the sign retains that convention explicitly.
        self.amplitude_factor[0] * f64::from(self.sign)
    }
}

#[derive(Clone, Copy, Debug)]
struct LcMaterializedSector {
    color_index: usize,
    reduction_weight: f64,
}

#[derive(Clone, Debug)]
struct LcResolvedReplayRoute {
    source_index: usize,
    target_index: usize,
    weight: f64,
}

#[derive(Clone, Debug)]
struct LcResolvedReplayEntry {
    routes: Vec<LcResolvedReplayRoute>,
}

#[derive(Clone, Debug)]
struct LcResolvedReplayPlan {
    #[cfg(test)]
    entries: Vec<LcResolvedReplayEntry>,
    routes_by_target: Vec<Vec<LcResolvedReplayTargetRoute>>,
    color_count: usize,
}

#[derive(Clone, Debug)]
struct LcResolvedReplayTargetRoute {
    mapping_index: usize,
    source_index: usize,
    weight: f64,
}

#[derive(Clone, Debug)]
struct LcResolvedReplaySelection {
    #[cfg(test)]
    mapping_indices: Vec<usize>,
    #[cfg(test)]
    entries: Vec<LcResolvedReplayEntry>,
    #[cfg(test)]
    source_helicity_indices: Vec<Vec<usize>>,
    #[cfg(test)]
    source_color_indices: Vec<Vec<usize>>,
    source_groups: Vec<LcResolvedReplaySourceGroup>,
    helicity_indices: Vec<usize>,
    color_indices: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LcResolvedReplaySelectionKey {
    helicity_indices: Option<Vec<usize>>,
    color_indices: Option<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct LcResolvedReplaySourceGroup {
    mapping_indices: Vec<usize>,
    entries: Vec<LcResolvedReplayEntry>,
    helicity_ids: BTreeSet<String>,
    color_ids: BTreeSet<String>,
    materialized_sector_ids: Option<BTreeSet<i64>>,
    direct_total_plan: Option<LcDirectTotalPlan>,
    source_component_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct LcDirectTotalPlan {
    mapping_index: usize,
    materialized_sector_id: i64,
    scale: f64,
}

fn uniform_lc_replay_total_scale(
    replay_entry: &LcResolvedReplayEntry,
    source_component_count: usize,
    target_component_count: usize,
) -> Option<f64> {
    if source_component_count == 0 {
        return None;
    }
    let mut source_scales = vec![0.0; source_component_count];
    for route in &replay_entry.routes {
        if route.source_index >= source_component_count
            || route.target_index >= target_component_count
            || !route.weight.is_finite()
        {
            return None;
        }
        source_scales[route.source_index] += route.weight;
    }
    let scale = source_scales[0];
    if !scale.is_finite() || source_scales[1..].iter().any(|value| *value != scale) {
        return None;
    }
    Some(scale)
}

pub const SYMJIT_APPLICATION_RUNTIME_CAPABILITY: &str = "symjit.application.complex-f64.v1";
pub const SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY: &str =
    "symbolica.legacy-jit-container.complex-f64.v1";
pub const SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY: &str = "symbolica.compiled-cpp.complex-f64.v1";
pub const SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY: &str = "symbolica.compiled-asm.complex-f64.v1";
pub const EAGER_DAG_RUNTIME_CAPABILITY: &str = crate::EAGER_RUNTIME_CAPABILITY;
pub const EAGER_RUNTIME_LAYOUT_CAPABILITY: &str = crate::eager_layout::EAGER_RUNTIME_CAPABILITY;
pub const EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY: &str =
    crate::eager_layout::EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY;
pub const EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY: &str =
    crate::EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY;
pub const RECURRENCE_RUNTIME_CAPABILITY: &str = crate::recurrence::RECURRENCE_RUNTIME_CAPABILITY;
pub const RECURRENCE_LC_COLOR_RUNTIME_CAPABILITY: &str =
    crate::recurrence::RECURRENCE_LC_COLOR_CAPABILITY;
pub const RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY: &str =
    crate::recurrence::RECURRENCE_CONTRACTED_COLOR_CAPABILITY;
pub const RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY: &str =
    "rusticol.recurrence-helicity-selector-companion.v2";
pub const ON_THE_FLY_RUNTIME_CAPABILITY: &str = "rusticol.on-the-fly.complex-f64.v1";
pub const ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY: &str =
    "rusticol.on-the-fly.contracted-color.v1";
pub const ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY: &str = "rusticol.on-the-fly.lc-color.v1";
pub const SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY: &str =
    "rusticol.color-contraction.symmetric-group-fft.v1";
pub const COMPILED_RUNTIME_SELECTORS_CAPABILITY: &str = "rusticol.compiled.runtime-selectors.v1";
pub const COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY: &str = "compiled-plane-arena-v1";
pub const COMPILED_PLANE_DIRECT_APPLICATION_ABI: &str = "pyamplicol-compiled-plane-kernel-v2";
pub const COMPILED_PLANE_SOURCE_APPLICATION_ABI: &str = "pyamplicol-symjit-plane-application-v2";
pub const COMPILED_HELICITY_DUAL_LANE_CAPABILITY: &str = "rusticol.compiled.helicity-dual-lane.v1";
pub const COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY: &str =
    "rusticol.compiled.helicity-selector-union.v1";
pub const COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY: &str =
    "rusticol.compiled.helicity-primary-recurrence.v1";
pub const COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY: &str =
    "rusticol.compiled.color-contraction-walsh.v1";
pub const COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY: &str =
    "rusticol.compiled.color-contraction-walsh-c2k.v1";
pub const COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY: &str =
    "rusticol.compiled.color-topology-lanes.v1";
pub const SYMJIT_APPLICATION_STORAGE_ABI: &str = "symjit-application-storage-v3";
pub const SYMJIT_PLANE_APPLICATION_ABI: &str = "pyamplicol-symjit-plane-application-v2";

#[doc(hidden)]
pub fn preflight_prepared_kernel_pack(
    manifest_path: &Path,
    payload_root: &Path,
) -> RusticolResult<usize> {
    let bytes = fs::read(manifest_path).map_err(|error| {
        RusticolError::artifact(format!(
            "could not read prepared kernel pack {}: {error}",
            manifest_path.display()
        ))
    })?;
    let pack: PreparedKernelPackManifest = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse prepared kernel pack {}: {error}",
            manifest_path.display()
        ))
    })?;
    pack.validate()?;
    PreparedEvaluatorBackend::preflight_all(&pack, payload_root)
}

#[cfg(feature = "f64-symjit")]
pub fn eager_direct_descriptor_for_source_application_bytes(
    source_bytes: &[u8],
    input_complex_count: u32,
    output_complex_count: u32,
    display_path: &Path,
) -> RusticolResult<Vec<u8>> {
    evaluator::symjit_eager_direct::eager_direct_descriptor_for_source_application_bytes(
        source_bytes,
        input_complex_count,
        output_complex_count,
        display_path,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeCapability {
    CompiledColorContractionWalshC2kV1,
    CompiledColorContractionWalshV1,
    CompiledColorTopologyLanesV1,
    CompiledHelicityDualLaneV1,
    CompiledHelicityPrimaryRecurrenceV1,
    CompiledHelicitySelectorUnionV1,
    CompiledPlaneArenaV1,
    CompiledRuntimeSelectorsV1,
    EagerDagComplexF64V1,
    EagerDirectArenaV1,
    EagerRuntimeLayoutComplexF64V1,
    EagerLcTopologyReplayComplexF64V1,
    RecurrenceRuntimeComplexF64V1,
    RecurrenceLcColorV1,
    RecurrenceContractedColorV1,
    RecurrenceHelicitySelectorCompanionV2,
    SymmetricGroupFftColorContractionV1,
    OnTheFlyRuntimeComplexF64V1,
    OnTheFlyContractedColorV1,
    OnTheFlyLcColorV1,
    SymjitApplicationComplexF64V1,
    SymbolicaLegacyJitContainerComplexF64V1,
    SymbolicaCompiledCppComplexF64V1,
    SymbolicaCompiledAsmComplexF64V1,
}

impl RuntimeCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompiledColorContractionWalshC2kV1 => {
                COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY
            }
            Self::CompiledColorContractionWalshV1 => COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY,
            Self::CompiledColorTopologyLanesV1 => COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
            Self::CompiledHelicityDualLaneV1 => COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
            Self::CompiledHelicityPrimaryRecurrenceV1 => {
                COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
            }
            Self::CompiledHelicitySelectorUnionV1 => COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
            Self::CompiledPlaneArenaV1 => COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            Self::CompiledRuntimeSelectorsV1 => COMPILED_RUNTIME_SELECTORS_CAPABILITY,
            Self::EagerDagComplexF64V1 => EAGER_DAG_RUNTIME_CAPABILITY,
            Self::EagerDirectArenaV1 => EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
            Self::EagerRuntimeLayoutComplexF64V1 => EAGER_RUNTIME_LAYOUT_CAPABILITY,
            Self::EagerLcTopologyReplayComplexF64V1 => EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
            Self::RecurrenceRuntimeComplexF64V1 => RECURRENCE_RUNTIME_CAPABILITY,
            Self::RecurrenceLcColorV1 => RECURRENCE_LC_COLOR_RUNTIME_CAPABILITY,
            Self::RecurrenceContractedColorV1 => RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            Self::RecurrenceHelicitySelectorCompanionV2 => {
                RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY
            }
            Self::SymmetricGroupFftColorContractionV1 => {
                SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY
            }
            Self::OnTheFlyRuntimeComplexF64V1 => ON_THE_FLY_RUNTIME_CAPABILITY,
            Self::OnTheFlyContractedColorV1 => ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
            Self::OnTheFlyLcColorV1 => ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
            Self::SymjitApplicationComplexF64V1 => SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            Self::SymbolicaLegacyJitContainerComplexF64V1 => {
                SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
            }
            Self::SymbolicaCompiledCppComplexF64V1 => SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
            Self::SymbolicaCompiledAsmComplexF64V1 => SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
        }
    }
}

pub fn supported_runtime_capabilities() -> Vec<&'static str> {
    let mut capabilities = vec![
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        EAGER_RUNTIME_LAYOUT_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        RECURRENCE_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        RECURRENCE_LC_COLOR_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        ON_THE_FLY_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
        #[cfg(feature = "f64-symjit")]
        SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
        #[cfg(feature = "symbolica-runtime")]
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        #[cfg(feature = "f64-compiled")]
        SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
        #[cfg(feature = "f64-compiled")]
        SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
    ];
    capabilities.sort_unstable();
    capabilities
}

pub(crate) fn ensure_runtime_capabilities_supported<'a>(
    capabilities: impl IntoIterator<Item = &'a str>,
) -> RusticolResult<()> {
    for capability in capabilities {
        if !runtime_capability_is_supported(capability) {
            let supported = supported_runtime_capabilities();
            return Err(RusticolError::unsupported_runtime_capability(
                capability,
                format!("this Rusticol build supports {supported:?}"),
            ));
        }
    }
    Ok(())
}

/// Test one capability on the successful runtime-load path without allocating
/// and sorting a complete capability set.  The public reporting helper above
/// still returns the complete sorted list when diagnostics need it.
fn runtime_capability_is_supported(capability: &str) -> bool {
    match capability {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY
        | COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY
        | COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY
        | COMPILED_HELICITY_DUAL_LANE_CAPABILITY
        | COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY
        | COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY
        | COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY
        | COMPILED_RUNTIME_SELECTORS_CAPABILITY
        | EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY
        | EAGER_RUNTIME_LAYOUT_CAPABILITY
        | RECURRENCE_RUNTIME_CAPABILITY
        | RECURRENCE_LC_COLOR_RUNTIME_CAPABILITY
        | RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY
        | RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY
        | SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY
        | ON_THE_FLY_RUNTIME_CAPABILITY
        | ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY
        | ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY => true,
        #[cfg(feature = "f64-symjit")]
        SYMJIT_APPLICATION_RUNTIME_CAPABILITY => true,
        #[cfg(feature = "symbolica-runtime")]
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY => true,
        #[cfg(feature = "f64-compiled")]
        SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY | SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY => {
            true
        }
        _ => false,
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ExecutionManifestHeader {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    kind: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionSetManifest {
    schema_version: u32,
    kind: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
    processes: Vec<ExecutionSetEntry>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionSetEntry {
    process_id: String,
    manifest_path: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputCrossingMapEntry {
    target_index: usize,
    source_index: usize,
    sign: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionManifest {
    schema_version: u32,
    kind: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
    process: String,
    key: String,
    color_accuracy: String,
    external_pdg_order: Vec<i32>,
    compiled: EvaluatorSetManifest,
    dag_summary: ExecutionSummary,
    materialization_census: ExecutionMaterializationCensus,
    runtime_schema: ExecutionPlan,
    #[serde(default)]
    color_contraction_payload: Option<CompiledColorContractionPayloadManifest>,
    #[serde(default)]
    physics_reduction: Option<crate::Reduction>,
    #[serde(default)]
    helicity_sum_execution: Option<Box<ExecutionManifest>>,
    #[serde(default)]
    helicity_selector_executions: Vec<HelicitySelectorExecutionManifest>,
    #[serde(default)]
    color_selector_executions: Vec<ColorSelectorExecutionManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledColorContractionPayloadManifest {
    path: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HelicitySelectorExecutionManifest {
    selector_domain_ids: Vec<usize>,
    #[serde(default)]
    schedule_mode: HelicitySelectorScheduleMode,
    execution: Box<ExecutionManifest>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum HelicitySelectorScheduleMode {
    #[default]
    ParentClosure,
    NestedRuntime,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColorSelectorExecutionManifest {
    materialized_sector_id: i64,
    execution: Box<ExecutionManifest>,
}

#[derive(Clone, Debug)]
struct EvaluatorSetManifest {
    kind: String,
    runtime_available: bool,
    runtime_unavailable_message: Option<String>,
    lc_topology_replay: Option<LcTopologyReplayManifest>,
    color_topology_replay: Option<ColorTopologyReplayManifest>,
    model_parameter_evaluator: Option<GenericModelParameterEvaluatorManifest>,
    stage_evaluators: Option<GenericStageEvaluatorArtifactsManifest>,
    momentum_slot_ids: Option<Vec<usize>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluatorSetManifestWire {
    kind: String,
    runtime_available: bool,
    runtime_unavailable_message: Option<String>,
    #[serde(default)]
    lc_topology_replay: Option<LcTopologyReplayManifest>,
    #[serde(default)]
    color_topology_replay: Option<ColorTopologyReplayManifest>,
    // Current Python artifacts mirror this additive contract under `compiled`.
    // Runtime loading uses the authoritative runtime_schema copy below.
    #[serde(default)]
    helicity_recurrence: Option<HelicityRecurrenceManifest>,
    #[serde(default)]
    model_parameter_evaluator: Option<GenericModelParameterEvaluatorManifest>,
    stage_evaluators: Option<GenericStageEvaluatorArtifactsManifest>,
    #[serde(default)]
    momentum_slot_ids: Option<Vec<usize>>,
}

impl<'de> Deserialize<'de> for EvaluatorSetManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = EvaluatorSetManifestWire::deserialize(deserializer)?;
        let _validated_additive_mirror = wire.helicity_recurrence;
        Ok(Self {
            kind: wire.kind,
            runtime_available: wire.runtime_available,
            runtime_unavailable_message: wire.runtime_unavailable_message,
            lc_topology_replay: wire.lc_topology_replay,
            color_topology_replay: wire.color_topology_replay,
            model_parameter_evaluator: wire.model_parameter_evaluator,
            stage_evaluators: wire.stage_evaluators,
            momentum_slot_ids: wire.momentum_slot_ids,
        })
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ColorTopologyReplayManifest {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    contract_version: Option<u32>,
    #[serde(default)]
    color_accuracy: String,
    #[serde(default)]
    physical_sector_count: Option<usize>,
    #[serde(default)]
    replayed_sector_count: usize,
    #[serde(default)]
    materialized_sector_ids: Vec<i64>,
    #[serde(default)]
    residual_sector_ids: Vec<i64>,
    #[serde(default)]
    groups: Vec<LcTopologyReplayGroupManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericModelParameterEvaluatorManifest {
    kind: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
    input_parameter_indices: Vec<usize>,
    outputs: Vec<GenericDerivedParameterOutputManifest>,
    evaluator: EvaluatorManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericDerivedParameterOutputManifest {
    runtime_name: String,
    output_index: usize,
    real_parameter_index: usize,
    imag_parameter_index: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayManifest {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    contract_version: Option<u32>,
    #[serde(default)]
    physical_sector_count: Option<usize>,
    #[serde(default)]
    replayed_sector_count: usize,
    #[serde(default)]
    materialized_sector_ids: Vec<i64>,
    #[serde(default)]
    residual_sector_ids: Vec<i64>,
    #[serde(default)]
    groups: Vec<LcTopologyReplayGroupManifest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayGroupManifest {
    representative_sector_id: i64,
    materialized_sector_id: i64,
    #[serde(default)]
    active_sector_ids: Vec<i64>,
    #[serde(default)]
    proof: Option<LcTopologyReplayProofManifest>,
    #[serde(default)]
    sector_permutations: Vec<LcTopologyReplaySectorPermutationManifest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayProofManifest {
    #[serde(default)]
    status: String,
    #[serde(default)]
    algorithm: Option<String>,
    #[serde(default)]
    digest: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplaySectorPermutationManifest {
    sector_id: i64,
    #[serde(default = "default_lc_topology_replay_weight")]
    weight: f64,
    #[serde(default = "default_lc_topology_replay_sign")]
    sign: i8,
    #[serde(default)]
    factor: Option<Vec<f64>>,
    #[serde(default)]
    label_permutation: Vec<LcTopologyReplayLabelPermutationManifest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LcTopologyReplayLabelPermutationManifest {
    representative_label: usize,
    sector_label: usize,
}

fn default_lc_topology_replay_weight() -> f64 {
    1.0
}

fn default_lc_topology_replay_sign() -> i8 {
    1
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceManifest {
    kind: String,
    contract_version: u32,
    proof_algorithm: String,
    current_count: usize,
    amplitude_root_count: usize,
    proof_counts: HelicityRecurrenceProofCountsManifest,
    selector_domains: Vec<HelicitySelectorDomainManifest>,
    source_state_mappings: Vec<HelicitySourceStateMappingManifest>,
    recurrence_classes: Vec<HelicityRecurrenceClassManifest>,
    amplitude_classes: Vec<HelicityAmplitudeReplayClassManifest>,
    residual_current_ids: Vec<usize>,
    residual_root_ids: Vec<usize>,
    structural_zero_selector_domain_ids: Vec<usize>,
    diagnostics: Vec<String>,
    #[serde(default)]
    materialization: Option<HelicityRecurrenceMaterializationManifest>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceProofCountsManifest {
    recurrence_class_count: usize,
    optimized_recurrence_class_count: usize,
    optimized_current_count: usize,
    residual_current_count: usize,
    amplitude_class_count: usize,
    optimized_amplitude_class_count: usize,
    residual_amplitude_count: usize,
    source_state_mapping_count: usize,
    physical_helicity_count: usize,
    structural_zero_helicity_count: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicitySelectorDomainManifest {
    id: usize,
    complete: bool,
    source_states: Vec<HelicitySelectorSourceStateManifest>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(deny_unknown_fields)]
struct HelicitySelectorSourceStateManifest {
    external_label: usize,
    helicity: i32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityCurrentReplayMemberManifest {
    current_id: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceClassManifest {
    class_id: String,
    representative_current_id: usize,
    external_labels: Vec<usize>,
    source_class: bool,
    members: Vec<HelicityCurrentReplayMemberManifest>,
    proof: HelicityRecurrenceProofManifest,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceProofManifest {
    status: String,
    algorithm: String,
    digest: String,
    transition_contract_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicitySourceStateMappingManifest {
    current_id: usize,
    external_label: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_state_index: usize,
    selector_domain_id: usize,
    recurrence_class_id: String,
    representative_current_id: usize,
    source_contract_digest: String,
    factor: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityAmplitudeReplayMemberManifest {
    root_id: usize,
    selector_domain_ids: Vec<usize>,
    factor: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityAmplitudeReplayClassManifest {
    class_id: String,
    representative_root_id: usize,
    members: Vec<HelicityAmplitudeReplayMemberManifest>,
    proof: HelicityRecurrenceProofManifest,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityRecurrenceMaterializationManifest {
    kind: String,
    contract_version: u32,
    #[serde(default)]
    strategy: HelicityMaterializationStrategy,
    proof_current_count: usize,
    proof_root_count: usize,
    materialized_current_count: usize,
    materialized_root_count: usize,
    proof_to_materialized_current: Vec<usize>,
    source_routes: Vec<HelicityMaterializedSourceRouteManifest>,
    amplitude_routes: Vec<HelicityMaterializedAmplitudeRouteManifest>,
    selector_schedules: Vec<HelicityMaterializedSelectorScheduleManifest>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum HelicityMaterializationStrategy {
    #[default]
    Quotient,
    RetainedProofGraph,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityMaterializedSourceRouteManifest {
    materialized_current_id: usize,
    external_label: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_state_index: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityMaterializedAmplitudeRouteManifest {
    materialized_root_id: usize,
    selector_domain_ids: Vec<usize>,
    factor: [f64; 2],
    residual: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct HelicityMaterializedSelectorScheduleManifest {
    selector_domain_id: usize,
    active_current_ids: Vec<usize>,
    active_root_ids: Vec<usize>,
    structural_zero: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityRecurrenceRuntime {
    selector_domains: Vec<HelicitySelectorDomainRuntime>,
    source_state_mappings: Vec<HelicitySourceStateMappingRuntime>,
    recurrence_classes: Vec<HelicityRecurrenceClassRuntime>,
    amplitude_classes: Vec<HelicityAmplitudeReplayClassRuntime>,
    residual_current_ids: Vec<usize>,
    residual_root_ids: Vec<usize>,
    structural_zero_selector_domain_ids: Vec<usize>,
    materialization: Option<HelicityRecurrenceMaterializationRuntime>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicitySelectorDomainRuntime {
    complete: bool,
    source_states: Vec<(usize, i32)>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityCurrentReplayMemberRuntime {
    current_id: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityRecurrenceClassRuntime {
    representative_current_id: usize,
    external_labels: Vec<usize>,
    source_class: bool,
    members: Vec<HelicityCurrentReplayMemberRuntime>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicitySourceStateMappingRuntime {
    current_id: usize,
    external_index: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_state_index: usize,
    selector_domain_id: usize,
    recurrence_class_index: usize,
    representative_current_id: usize,
    factor: [f64; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityAmplitudeReplayMemberRuntime {
    root_id: usize,
    selector_domain_ids: Vec<usize>,
    factor: [f64; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityAmplitudeReplayClassRuntime {
    representative_root_id: usize,
    members: Vec<HelicityAmplitudeReplayMemberRuntime>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityRecurrenceMaterializationRuntime {
    strategy: HelicityMaterializationStrategy,
    proof_to_materialized_current: Vec<usize>,
    source_routes: Vec<HelicityMaterializedSourceRouteRuntime>,
    amplitude_routes: Vec<HelicityMaterializedAmplitudeRouteRuntime>,
    selector_schedules: Vec<HelicityMaterializedSelectorScheduleRuntime>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityMaterializedSourceRouteRuntime {
    materialized_current_id: usize,
    external_index: usize,
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
    declared_state_index: usize,
    selector_domain_id: usize,
    factor: [f64; 2],
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityMaterializedAmplitudeRouteRuntime {
    materialized_root_id: usize,
    selector_domain_ids: Vec<usize>,
    factor: [f64; 2],
    residual: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct HelicityMaterializedSelectorScheduleRuntime {
    selector_domain_id: usize,
    active_current_ids: Vec<usize>,
    active_root_ids: Vec<usize>,
    active_stage_chunk_indices: Vec<Vec<usize>>,
    active_amplitude_chunk_indices: Vec<usize>,
    structural_zero: bool,
}

#[derive(Clone, Debug)]
struct CompiledColorSelectorSchedule {
    active_stage_chunk_indices: Vec<Vec<usize>>,
    active_amplitude_chunk_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct CompiledColorExecutionPlan {
    schedules_by_materialized_sector: BTreeMap<i64, Arc<CompiledColorSelectorSchedule>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericStageEvaluatorArtifactsManifest {
    kind: String,
    #[serde(default)]
    required_runtime_capabilities: Vec<String>,
    runtime_available: bool,
    runtime_unavailable_message: Option<String>,
    parameter_count: usize,
    value_parameter_count: usize,
    momentum_parameter_count: usize,
    #[serde(default)]
    model_parameter_count: usize,
    real_valued_inputs: Vec<usize>,
    parameter_layout: String,
    stage_count: usize,
    stages: Vec<GenericSerializedStageEvaluatorManifest>,
    amplitude_stage: GenericSerializedStageEvaluatorManifest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericSerializedStageEvaluatorManifest {
    stage_index: usize,
    stage_kind: String,
    subset_size: Option<usize>,
    evaluator_label: String,
    parameter_layout: String,
    output_length: usize,
    output_slots: Vec<GenericStageOutputSlotManifest>,
    input_value_slot_ids: Vec<usize>,
    output_value_slot_ids: Vec<usize>,
    interaction_ids: Vec<usize>,
    #[serde(default)]
    input_components: Vec<GenericStageInputComponentManifest>,
    #[serde(default)]
    parameter_count: usize,
    #[serde(default)]
    value_parameter_count: usize,
    #[serde(default)]
    momentum_parameter_count: usize,
    #[serde(default)]
    model_parameter_count: usize,
    #[serde(default)]
    real_valued_inputs: Vec<usize>,
    expression_ready: bool,
    blockers: Vec<String>,
    evaluator: EvaluatorManifest,
    #[serde(default)]
    compiled_plane_arena: Option<CompiledPlaneArenaStageManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledPlaneArenaStageManifest {
    schema_version: u32,
    kind: String,
    application_abi: String,
    source_application_abi: String,
    element_layout: String,
    output_operation: String,
    output_factor: String,
    input_output_aliasing: String,
    output_output_aliasing: String,
    input_bindings: Vec<CompiledPlaneInputBindingManifest>,
    output_bindings: Vec<CompiledPlaneOutputBindingManifest>,
    leaves: Vec<CompiledPlaneLeafManifest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CompiledPlaneInputBindingManifest {
    parameter_index: usize,
    kind: String,
    source_id: usize,
    component: usize,
    global_component: usize,
    #[serde(default)]
    real_valued: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CompiledPlaneOutputBindingManifest {
    output_index: usize,
    arena: String,
    component: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CompiledPlaneLeafManifest {
    application_path: String,
    source_application_abi: String,
    optimization_level: u8,
    direct_codegen_optimization_level: u8,
    input_len: usize,
    output_len: usize,
    input_indices: Vec<usize>,
    output_start: usize,
    output_stop: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericStageInputComponentManifest {
    kind: String,
    source_id: usize,
    component: usize,
    global_component: usize,
    parameter_index: usize,
    #[serde(default)]
    real_valued: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericStageOutputSlotManifest {
    value_slot_id: isize,
    current_id: isize,
    variant: String,
    component_start: usize,
    component_stop: usize,
    output_start: usize,
    output_stop: usize,
    #[serde(default)]
    color_selector_domain_ids: Vec<i64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionSummary {
    current_count: usize,
    source_count: usize,
    interaction_count: usize,
    interaction_evaluation_count: usize,
    amplitude_root_count: usize,
    truncated: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionMaterializationCensus {
    abi: String,
    basis: String,
    r#final: BTreeMap<String, usize>,
    peak: BTreeMap<String, usize>,
    final_equals_peak: bool,
}

impl ExecutionMaterializationCensus {
    fn from_summary(summary: &ExecutionSummary) -> Self {
        let counts = BTreeMap::from([
            (
                "amplitude_root_count".to_string(),
                summary.amplitude_root_count,
            ),
            ("current_count".to_string(), summary.current_count),
            ("interaction_count".to_string(), summary.interaction_count),
            (
                "interaction_evaluation_count".to_string(),
                summary.interaction_evaluation_count,
            ),
            ("source_count".to_string(), summary.source_count),
        ]);
        Self {
            abi: "pyamplicol-fully-resident-materialization-census-v1".to_string(),
            basis: "immutable-fully-resident-compiled-dag".to_string(),
            r#final: counts.clone(),
            peak: counts,
            final_equals_peak: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionPlan {
    schema_version: u32,
    kind: String,
    process_key: String,
    process: String,
    external_particles: Vec<GenericExternalParticleManifest>,
    #[serde(default)]
    model: Option<GenericRuntimeModelManifest>,
    #[serde(default)]
    model_parameters: Vec<GenericRuntimeModelParameterManifest>,
    #[serde(default)]
    normalization: Option<GenericRuntimeNormalizationManifest>,
    parameter_layout: GenericParameterLayoutManifest,
    current_storage: GenericCurrentStorageManifest,
    value_storage: GenericValueStorageManifest,
    source_fill: GenericSourceFillManifest,
    momentum_slots: Vec<GenericMomentumSlotManifest>,
    stages: Vec<GenericStageManifest>,
    amplitude_stage: GenericAmplitudeStageManifest,
    #[serde(default)]
    helicity_recurrence: Option<HelicityRecurrenceManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericExternalParticleManifest {
    label: usize,
    index: usize,
    pdg: i32,
    outgoing_pdg: i32,
    role: String,
    momentum_slot: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRuntimeModelManifest {
    #[serde(default)]
    particles: Vec<GenericRuntimeParticleManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericRuntimeModelParameterManifest {
    name: String,
    kind: String,
    parameter_index: usize,
    #[serde(default)]
    default: f64,
    #[serde(default)]
    pdg: Option<i32>,
    #[serde(default)]
    runtime_name: Option<String>,
    #[serde(default)]
    complex_component: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRuntimeParticleManifest {
    pdg: i32,
    #[serde(default)]
    mass: f64,
    #[serde(default)]
    mass_parameter: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRuntimeNormalizationManifest {
    #[serde(default = "default_one_f64")]
    color_factor: f64,
    #[serde(default = "default_one_f64")]
    global_coupling_factor: f64,
    #[serde(default = "default_one_f64")]
    average_factor: f64,
    #[serde(default = "default_one_f64")]
    identical_factor: f64,
    #[serde(default)]
    qcd_coupling_power: usize,
    #[serde(default)]
    electroweak_coupling_power: usize,
}

fn default_one_f64() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericParameterLayoutManifest {
    source_component_parameter_count: usize,
    momentum_parameter_count: usize,
    #[serde(default)]
    model_parameter_count: usize,
    parameter_count_if_flattened: usize,
    value_component_count: usize,
    source_components_complex: bool,
    momentum_components_real: bool,
    real_valued_inputs: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericCurrentStorageManifest {
    component_count: usize,
    number_type: String,
    #[serde(default)]
    metadata_compacted: bool,
    current_slots: Vec<GenericCurrentSlotManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericCurrentSlotManifest {
    current_id: usize,
    component_start: usize,
    component_stop: usize,
    dimension: usize,
    is_source: bool,
    particle_id: i32,
    external_mask: u64,
    #[serde(default)]
    external_labels: Vec<usize>,
    #[serde(default)]
    helicity_ancestry: Value,
    chirality: i32,
    #[serde(default)]
    spin_state: Value,
    #[serde(default)]
    flavour_flow: Vec<i32>,
    #[serde(default)]
    color_state: Value,
    momentum_mask: u64,
    auxiliary_kind: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericValueStorageManifest {
    component_count: usize,
    number_type: String,
    #[serde(default)]
    metadata_compacted: bool,
    value_slots: Vec<GenericValueSlotManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericValueSlotManifest {
    value_slot_id: usize,
    current_id: usize,
    variant: String,
    component_start: usize,
    component_stop: usize,
    dimension: usize,
    current_component_start: usize,
    current_component_stop: usize,
    is_source: bool,
    applies_propagator: bool,
    particle_id: i32,
    external_mask: u64,
    #[serde(default)]
    external_labels: Vec<usize>,
    momentum_mask: u64,
    chirality: i32,
    propagator: GenericPropagatorIrManifest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericPropagatorKindManifest {
    Identity,
    Scalar,
    WeylFermion,
    DiracFermion,
    Vector,
    Spin2,
    Custom,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericPropagatorMassClassManifest {
    Massless,
    Massive,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericPropagatorGaugeManifest {
    Feynman,
    Unitary,
    DeDonder,
    FierzPauli,
    ModelSupplied,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum GenericGoldstonePolicyManifest {
    NotApplicable,
    Absorbed,
    Explicit,
    ModelSupplied,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericPropagatorIrManifest {
    identity: GenericParticleIdentityIrManifest,
    particle_id: i32,
    chirality: i32,
    kind: GenericPropagatorKindManifest,
    backend: String,
    basis: String,
    applies_propagator: bool,
    kernel: String,
    full_tensor_network_ready: bool,
    mass_class: GenericPropagatorMassClassManifest,
    gauge: Option<GenericPropagatorGaugeManifest>,
    numerator: Option<String>,
    denominator: Option<String>,
    mass_parameter: Option<String>,
    width_parameter: Option<String>,
    custom_source: Option<String>,
    auxiliary_policy: Option<String>,
    goldstone_policy: GenericGoldstonePolicyManifest,
    description: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceFillManifest {
    source_count: usize,
    sources: Vec<GenericSourceRecordManifest>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum GenericSourceOrientationManifest {
    Particle,
    Antiparticle,
    SelfConjugate,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GenericParticleStatisticsManifest {
    Boson,
    Fermion,
    Ghost,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GenericWavefunctionFamilyManifest {
    Scalar,
    Fermion,
    Vector,
    Spin2,
    Ghost,
    Auxiliary,
}

impl GenericWavefunctionFamilyManifest {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Fermion => "fermion",
            Self::Vector => "vector",
            Self::Spin2 => "spin2",
            Self::Ghost => "ghost",
            Self::Auxiliary => "auxiliary",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum GenericMomentumTransformManifest {
    Identity,
    NegateFourMomentum,
}

impl GenericMomentumTransformManifest {
    const fn legacy_projection(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::NegateFourMomentum => "negate-incoming-momentum",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
enum GenericSourceSpinStateManifest {
    Scalar(i32),
    Components(Vec<i32>),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceStateIrManifest {
    helicity: i32,
    chirality: i32,
    spin_state: GenericSourceSpinStateManifest,
}

impl GenericSourceStateIrManifest {
    fn transformed(&self, crossing: &GenericCrossingIrManifest) -> Result<Self, &'static str> {
        let helicity = self
            .helicity
            .checked_mul(crossing.helicity_factor)
            .ok_or("source crossing overflows the helicity state")?;
        let chirality = self
            .chirality
            .checked_mul(crossing.chirality_factor)
            .ok_or("source crossing overflows the chirality state")?;
        let spin_state = match (&self.spin_state, crossing.spin_state_factor) {
            (state, 1) => state.clone(),
            (GenericSourceSpinStateManifest::Scalar(state), factor) => {
                GenericSourceSpinStateManifest::Scalar(
                    state
                        .checked_mul(factor)
                        .ok_or("source crossing overflows the spin state")?,
                )
            }
            (GenericSourceSpinStateManifest::Components(_), _) => {
                return Err("crossing cannot multiply a structured source spin state");
            }
        };
        Ok(Self {
            helicity,
            chirality,
            spin_state,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericCrossingIrManifest {
    momentum_transform: GenericMomentumTransformManifest,
    helicity_factor: i32,
    chirality_factor: i32,
    spin_state_factor: i32,
    phase: [f64; 2],
}

impl GenericCrossingIrManifest {
    fn is_identity(&self) -> bool {
        self.momentum_transform == GenericMomentumTransformManifest::Identity
            && self.helicity_factor == 1
            && self.chirality_factor == 1
            && self.spin_state_factor == 1
            && self.phase == [1.0, 0.0]
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericParticleIdentityIrManifest {
    canonical_id: String,
    species_id: String,
    anti_canonical_id: String,
    display_name: String,
    anti_display_name: String,
    pdg_label: i32,
    anti_pdg_label: i32,
    orientation: GenericSourceOrientationManifest,
    self_conjugate: bool,
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceIrManifest {
    identity: GenericParticleIdentityIrManifest,
    statistics: GenericParticleStatisticsManifest,
    wavefunction_family: GenericWavefunctionFamilyManifest,
    component_dimension: usize,
    states: Vec<GenericSourceStateIrManifest>,
    crossing: GenericCrossingIrManifest,
    basis: String,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    mass_parameter: Option<String>,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    width_parameter: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericSourceRecordManifest {
    source_id: usize,
    current_id: usize,
    current_component_start: usize,
    current_component_stop: usize,
    value_slot: GenericValueSlotRefManifest,
    source_parameter_start: usize,
    source_parameter_stop: usize,
    leg_label: usize,
    input_momentum_slot: usize,
    side: String,
    crossing: String,
    physical_pdg: i32,
    outgoing_pdg: i32,
    particle_id: i32,
    anti_particle_id: i32,
    source_kind: String,
    wavefunction_kind: String,
    source_orientation: GenericSourceOrientationManifest,
    source_basis: String,
    source_ir: GenericSourceIrManifest,
    applied_crossing: GenericCrossingIrManifest,
    source_helicity: i32,
    chirality: i32,
    spin_state: Value,
    dimension: usize,
    helicity_ancestry: Value,
    color_state: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericMomentumSlotManifest {
    momentum_slot_id: usize,
    momentum_mask: u64,
    external_labels: Vec<usize>,
    component_start: usize,
    component_stop: usize,
    real_valued: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericStageManifest {
    stage_index: usize,
    stage_kind: String,
    subset_size: usize,
    input_current_ids: Vec<usize>,
    output_current_ids: Vec<usize>,
    input_value_slot_ids: Vec<usize>,
    output_value_slot_ids: Vec<usize>,
    interaction_count: usize,
    #[serde(default)]
    interactions_compacted: bool,
    #[serde(default)]
    interaction_ids: Vec<usize>,
    #[serde(default)]
    interactions: Vec<GenericInteractionManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericInteractionManifest {
    interaction_id: usize,
    vertex_kind: i32,
    vertex_particles: Vec<i32>,
    left_current_id: usize,
    right_current_id: usize,
    result_current_id: usize,
    left_slot: GenericSlotRefManifest,
    right_slot: GenericSlotRefManifest,
    result_slot: GenericSlotRefManifest,
    left_value_slot: GenericValueSlotRefManifest,
    right_value_slot: GenericValueSlotRefManifest,
    result_value_slots: Vec<GenericValueSlotRefManifest>,
    result_requires_propagated_value: bool,
    result_requires_unpropagated_value: bool,
    momentum_slots: GenericInteractionMomentumSlotsManifest,
    coupling: Vec<f64>,
    color_weight: Vec<f64>,
    accumulation: String,
    lowering: GenericLoweringManifest,
    full_tensor_network_ready: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericLoweringManifest {
    kind: i32,
    backend: String,
    tensor_names: Vec<String>,
    expression_head: String,
    full_tensor_network_ready: bool,
    description: String,
    kernel: String,
    input_roles: Vec<String>,
    output_role: String,
    coupling_mode: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericInteractionMomentumSlotsManifest {
    left: usize,
    right: usize,
    result: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericSlotRefManifest {
    current_id: usize,
    component_start: usize,
    component_stop: usize,
    dimension: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericAmplitudeStageManifest {
    stage_kind: String,
    output_count: usize,
    #[serde(default)]
    color_contraction: Option<GenericColorContractionManifest>,
    #[serde(default)]
    color_topology_replay: Option<GenericColorTopologyReplayAmplitudeManifest>,
    roots: Vec<GenericAmplitudeRootManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericColorTopologyReplayAmplitudeManifest {
    contract_version: u32,
    physical_group_count: usize,
    physical_groups: Vec<GenericColorTopologyReplayPhysicalGroupManifest>,
    mappings: Vec<GenericColorTopologyReplayMappingManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericColorTopologyReplayPhysicalGroupManifest {
    group_id: i64,
    helicities: Vec<i32>,
    color_sector_id: i64,
    color_word: Vec<usize>,
    helicity_weight: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericColorTopologyReplayMappingManifest {
    label_permutation: Vec<LcTopologyReplayLabelPermutationManifest>,
    group_routes: Vec<GenericColorTopologyReplayGroupRouteManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericColorTopologyReplayGroupRouteManifest {
    source_group_id: i64,
    target_group_id: i64,
    factor: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericColorContractionManifest {
    supported: bool,
    #[serde(default)]
    reason: Option<String>,
    group_count: usize,
    #[serde(default)]
    includes_color_factor: bool,
    entries: Vec<GenericColorContractionEntryManifest>,
    #[serde(default)]
    repeated_block: Option<GenericRepeatedColorContractionBlockManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericColorContractionEntryManifest {
    left_group_id: i64,
    right_group_id: i64,
    weight: Vec<f64>,
    // Retained for Python arbitrary precision; native contraction keeps its f64 weight.
    #[serde(default, rename = "exact_weight")]
    _exact_weight: Option<[String; 4]>,
    #[serde(default = "default_symmetry_factor")]
    symmetry_factor: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRepeatedColorContractionBlockManifest {
    component_count: usize,
    component_group_ids: Vec<i64>,
    entries: Vec<GenericRepeatedColorContractionEntryManifest>,
    #[serde(default)]
    factorized_block: Option<GenericFactorizedColorContractionBlockManifest>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum GenericFactorizedColorContractionBlockManifest {
    #[serde(rename = "klein-four-walsh")]
    KleinFourWalsh { cosets: Vec<[usize; 4]> },
    #[serde(rename = "elementary-abelian-walsh")]
    ElementaryAbelianWalsh {
        rank: usize,
        cosets: Vec<Vec<usize>>,
    },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericRepeatedColorContractionEntryManifest {
    left_group_index: usize,
    right_group_index: usize,
    weight: Vec<f64>,
    #[serde(default, rename = "exact_weight")]
    _exact_weight: Option<[String; 4]>,
    #[serde(default = "default_symmetry_factor")]
    symmetry_factor: f64,
}

fn default_symmetry_factor() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericAmplitudeRootManifest {
    output_index: usize,
    root_id: usize,
    kind: String,
    left_current_id: usize,
    right_current_id: usize,
    left_slot: GenericSlotRefManifest,
    right_slot: GenericSlotRefManifest,
    left_value_slot: GenericValueSlotRefManifest,
    right_value_slot: GenericValueSlotRefManifest,
    vertex_kind: Option<i32>,
    vertex_particles: Option<Vec<i32>>,
    coupling: Vec<f64>,
    color_weight: Vec<f64>,
    #[serde(default)]
    color_sector_id: Option<i64>,
    contraction: String,
    contraction_ir: GenericContractionIrManifest,
    coherent_group_id: Option<Value>,
    helicity_weight: f64,
    #[serde(default)]
    all_sector_weight: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
enum GenericContractionChiralityRelationManifest {
    Any,
    Equal,
    Opposite,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenericContractionIrManifest {
    name: String,
    left_basis: String,
    right_basis: String,
    coefficients: Vec<[f64; 2]>,
    chirality_relation: GenericContractionChiralityRelationManifest,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    metric_signature: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenericValueSlotRefManifest {
    value_slot_id: usize,
    current_id: usize,
    variant: String,
    component_start: usize,
    component_stop: usize,
    dimension: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeCompiledDirectTargetManifest {
    triple: String,
    cpu_features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeCompiledDirectApplicationManifest {
    application_abi: String,
    function_name: String,
    source_path: String,
    library_path: String,
    target: NativeCompiledDirectTargetManifest,
    evaluator_state_sha256: String,
    instruction_count: u32,
    temporary_count: u32,
    input_plane_count: u32,
    scalar_input_count: u32,
    output_plane_count: u32,
    simd_lane_width: u32,
    logical_stack_bytes: u32,
    output_semantics: String,
}

impl NativeCompiledDirectApplicationManifest {
    fn validate(
        &self,
        expected_function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> RusticolResult<()> {
        const APPLICATION_ABI: &str = "pyamplicol-native-compiled-direct-application-v1";
        const OUTPUT_SEMANTICS: &str = "factor-free-overwrite";
        const MAXIMUM_LOGICAL_STACK_BYTES: u32 = 64 * 1024;

        if self.application_abi != APPLICATION_ABI {
            return Err(RusticolError::compatibility(format!(
                "native compiled DirectApplication declares ABI {:?}, expected {APPLICATION_ABI:?}",
                self.application_abi
            )));
        }
        if self.function_name != expected_function_name {
            return Err(RusticolError::integrity(
                "native compiled DirectApplication function identity does not match its evaluator",
            ));
        }
        if self.source_path.is_empty()
            || self.library_path.is_empty()
            || self.source_path == self.library_path
            || self.target.triple.is_empty()
            || self.target.triple.contains('\0')
        {
            return Err(RusticolError::integrity(
                "native compiled DirectApplication paths or target are invalid",
            ));
        }
        if self
            .target
            .cpu_features
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || self
                .target
                .cpu_features
                .iter()
                .any(|feature| feature.is_empty() || feature.contains('\0'))
        {
            return Err(RusticolError::integrity(
                "native compiled DirectApplication CPU features are not sorted and unique",
            ));
        }
        if self.evaluator_state_sha256.len() != 64
            || !self
                .evaluator_state_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(RusticolError::integrity(
                "native compiled DirectApplication evaluator-state digest is invalid",
            ));
        }
        if self.instruction_count == 0
            || self.input_plane_count == 0
            || !matches!(self.simd_lane_width, 2 | 4)
            || self.logical_stack_bytes == 0
            || self.logical_stack_bytes > MAXIMUM_LOGICAL_STACK_BYTES
            || self.output_semantics != OUTPUT_SEMANTICS
        {
            return Err(RusticolError::integrity(
                "native compiled DirectApplication execution shape is invalid",
            ));
        }

        let expected_output_planes = u32::try_from(output_len.checked_mul(2).ok_or_else(|| {
            RusticolError::integrity("native compiled DirectApplication output shape overflows")
        })?)
        .map_err(|_| {
            RusticolError::integrity("native compiled DirectApplication output shape exceeds u32")
        })?;
        if self.output_plane_count != expected_output_planes {
            return Err(RusticolError::integrity(
                "native compiled DirectApplication output planes do not match its evaluator",
            ));
        }

        let descriptor_count = self
            .input_plane_count
            .checked_add(self.scalar_input_count)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "native compiled DirectApplication input descriptor count overflows",
                )
            })?;
        let minimum_descriptors = u32::try_from(input_len).map_err(|_| {
            RusticolError::integrity(
                "native compiled DirectApplication logical input count exceeds u32",
            )
        })?;
        let maximum_descriptors = minimum_descriptors.checked_mul(2).ok_or_else(|| {
            RusticolError::integrity(
                "native compiled DirectApplication maximum descriptor count overflows",
            )
        })?;
        if descriptor_count < minimum_descriptors || descriptor_count > maximum_descriptors {
            return Err(RusticolError::integrity(
                "native compiled DirectApplication descriptors do not cover evaluator inputs",
            ));
        }

        let expected_stack_bytes = self
            .temporary_count
            .checked_add(u32::try_from(output_len).map_err(|_| {
                RusticolError::integrity(
                    "native compiled DirectApplication output count exceeds u32",
                )
            })?)
            .and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_mul(std::mem::size_of::<f64>() as u32))
            .and_then(|bytes| bytes.checked_mul(self.simd_lane_width))
            .ok_or_else(|| {
                RusticolError::integrity(
                    "native compiled DirectApplication logical stack shape overflows",
                )
            })?;
        if self.logical_stack_bytes != expected_stack_bytes {
            return Err(RusticolError::integrity(
                "native compiled DirectApplication logical stack metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind")]
enum EvaluatorManifest {
    #[serde(rename = "symjit-application-evaluator")]
    SymjitApplication {
        runtime_capability: String,
        application_path: String,
        application_abi: String,
        input_len: usize,
        output_len: usize,
        element_layout: String,
        batch_layout: String,
        compiler_type: String,
        translation_mode: String,
        optimization_level: u8,
        word_bits: u8,
        endianness: String,
        required_defuns: Vec<String>,
        #[serde(default)]
        plane_application: Option<SymjitPlaneApplicationManifest>,
        evaluator_state_path: Option<String>,
        evaluator_state_runtime_capability: Option<String>,
    },
    #[serde(rename = "jit-symbolica-evaluator")]
    Jit {
        runtime_capability: String,
        input_len: usize,
        output_len: usize,
        evaluator_state_path: String,
    },
    #[serde(rename = "compiled-complex-evaluator")]
    CompiledComplex {
        runtime_capability: String,
        function_name: String,
        input_len: usize,
        output_len: usize,
        library_path: String,
        evaluator_state_path: Option<String>,
        number_type: String,
        #[serde(default)]
        native_direct_application: Option<NativeCompiledDirectApplicationManifest>,
    },
    #[serde(rename = "chunked-symbolica-evaluator")]
    Chunked {
        required_runtime_capabilities: Vec<String>,
        #[serde(default)]
        input_len: Option<usize>,
        #[serde(default)]
        chunk_input_indices: Option<Vec<Vec<usize>>>,
        chunks: Vec<EvaluatorManifest>,
    },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SymjitPlaneApplicationManifest {
    application_path: String,
    application_abi: String,
    storage_abi: String,
    element_layout: String,
    descriptor_order: String,
    input_complex_count: usize,
    output_complex_count: usize,
    input_plane_count: usize,
    output_plane_count: usize,
    compiler_type: String,
    translation_mode: String,
    optimization_level: u8,
    simd: bool,
    complex: bool,
    fast_math: bool,
    fast_complex: bool,
    compression: bool,
    threading: bool,
    direct_arena: bool,
    source_digest: String,
    target: Value,
}

impl SymjitPlaneApplicationManifest {
    fn validate(
        &self,
        input_complex_count: usize,
        output_complex_count: usize,
        optimization_level: u8,
    ) -> RusticolResult<()> {
        let expected_inputs = input_complex_count
            .checked_mul(2)
            .ok_or_else(|| RusticolError::integrity("SymJIT plane input count overflows usize"))?;
        let expected_outputs = output_complex_count
            .checked_mul(2)
            .ok_or_else(|| RusticolError::integrity("SymJIT plane output count overflows usize"))?;
        let digest_valid = self.source_digest.len() == 64
            && self
                .source_digest
                .bytes()
                .all(|value| value.is_ascii_hexdigit() && !value.is_ascii_uppercase());
        if self.application_path.is_empty()
            || self.application_abi != SYMJIT_PLANE_APPLICATION_ABI
            || self.storage_abi != SYMJIT_APPLICATION_STORAGE_ABI
            || self.element_layout != "split-complex-plane-major"
            || self.descriptor_order != "inputs-re-im-then-outputs-re-im"
            || self.input_complex_count != input_complex_count
            || self.output_complex_count != output_complex_count
            || self.input_plane_count != expected_inputs
            || self.output_plane_count != expected_outputs
            || self.compiler_type != "native"
            || self.translation_mode != "symbolica-structured-instructions"
            || self.optimization_level != optimization_level
            || !self.simd
            || !self.complex
            || !self.fast_math
            || self.fast_complex
            || self.threading
            || !self.direct_arena
            || !digest_valid
            || !valid_symjit_plane_target(&self.target)
        {
            return Err(RusticolError::compatibility(
                "SymJIT plane application metadata is incompatible; regenerate the artifact",
            ));
        }
        Ok(())
    }
}

fn valid_symjit_plane_target(target: &Value) -> bool {
    let Some(target) = target.as_object() else {
        return false;
    };
    if target.keys().any(|key| {
        !matches!(
            key.as_str(),
            "word_bits" | "endianness" | "triple" | "cpu_features"
        )
    }) || target.get("word_bits").and_then(Value::as_u64) != Some(64)
        || target.get("endianness").and_then(Value::as_str) != Some("little")
    {
        return false;
    }
    if let Some(triple) = target.get("triple") {
        let Some(triple) = triple.as_str() else {
            return false;
        };
        if triple.is_empty() || triple.contains('\0') {
            return false;
        }
    }
    if let Some(features) = target.get("cpu_features") {
        let Some(features) = features.as_array() else {
            return false;
        };
        let mut previous: Option<&str> = None;
        for feature in features {
            let Some(feature) = feature.as_str() else {
                return false;
            };
            if feature.is_empty()
                || feature.contains('\0')
                || previous.is_some_and(|previous| previous >= feature)
            {
                return false;
            }
            previous = Some(feature);
        }
    }
    true
}

/// One leaf in the canonical evaluator preorder used by loading, selector
/// coverage, and Direct-Arena lowering.
///
/// Keeping the composed root-input map and output range together prevents
/// independently implemented flattening walks from silently assigning
/// selector chunk indices to different leaves.
struct EvaluatorLeafLayout<'a> {
    evaluator: &'a EvaluatorManifest,
    input_indices: Vec<usize>,
    output_range: std::ops::Range<usize>,
}

impl EvaluatorManifest {
    fn io_len(&self) -> RusticolResult<(usize, usize)> {
        match self {
            Self::SymjitApplication {
                input_len,
                output_len,
                ..
            }
            | Self::Jit {
                input_len,
                output_len,
                ..
            } => Ok((*input_len, *output_len)),
            Self::CompiledComplex {
                function_name,
                input_len,
                output_len,
                ..
            } => {
                if let Self::CompiledComplex {
                    native_direct_application: Some(application),
                    ..
                } = self
                {
                    application.validate(function_name, *input_len, *output_len)?;
                }
                Ok((*input_len, *output_len))
            }
            Self::Chunked {
                input_len,
                chunk_input_indices,
                chunks,
                ..
            } => {
                if chunks.is_empty() {
                    return Err(RusticolError::artifact(
                        "generic serialized evaluator chunk list is empty",
                    ));
                }
                let mut child_layouts = Vec::with_capacity(chunks.len());
                let mut output_len = 0usize;
                for chunk in chunks {
                    let layout = chunk.io_len()?;
                    output_len = output_len.checked_add(layout.1).ok_or_else(|| {
                        RusticolError::artifact(
                            "generic serialized evaluator output length overflows usize",
                        )
                    })?;
                    child_layouts.push(layout);
                }
                match (input_len, chunk_input_indices) {
                    (None, None) => {
                        let parent_input_len = child_layouts[0].0;
                        if child_layouts
                            .iter()
                            .any(|(child_input_len, _)| *child_input_len != parent_input_len)
                        {
                            return Err(RusticolError::artifact(
                                "legacy chunked evaluator children have inconsistent input lengths",
                            ));
                        }
                        Ok((parent_input_len, output_len))
                    }
                    (Some(parent_input_len), Some(input_indices)) => {
                        if input_indices.len() != chunks.len() {
                            return Err(RusticolError::artifact(
                                "chunked evaluator input maps do not match evaluator chunks",
                            ));
                        }
                        for (indices, (child_input_len, _)) in
                            input_indices.iter().zip(&child_layouts)
                        {
                            if indices.len() != *child_input_len
                                || indices.iter().any(|index| *index >= *parent_input_len)
                                || indices.windows(2).any(|pair| pair[0] >= pair[1])
                            {
                                return Err(RusticolError::artifact(
                                    "chunked evaluator input map is inconsistent with child inputs",
                                ));
                            }
                        }
                        Ok((*parent_input_len, output_len))
                    }
                    _ => Err(RusticolError::artifact(
                        "chunked evaluator input metadata is incomplete",
                    )),
                }
            }
        }
    }

    fn leaf_layout(&self) -> RusticolResult<Vec<EvaluatorLeafLayout<'_>>> {
        fn append_leaf_layouts<'a>(
            evaluator: &'a EvaluatorManifest,
            parent_inputs: &[usize],
            output_cursor: &mut usize,
            leaves: &mut Vec<EvaluatorLeafLayout<'a>>,
        ) -> RusticolResult<()> {
            match evaluator {
                EvaluatorManifest::Chunked {
                    input_len,
                    chunk_input_indices,
                    chunks,
                    ..
                } => {
                    evaluator.io_len()?;
                    if let Some(input_len) = input_len
                        && *input_len != parent_inputs.len()
                    {
                        return Err(RusticolError::artifact(
                            "chunked evaluator parent input mapping has an inconsistent length",
                        ));
                    }
                    match chunk_input_indices {
                        Some(chunk_inputs) => {
                            for (chunk, indices) in chunks.iter().zip(chunk_inputs) {
                                let mapped = indices
                                    .iter()
                                    .map(|index| {
                                        parent_inputs.get(*index).copied().ok_or_else(|| {
                                            RusticolError::artifact(
                                                "chunked evaluator input map references an absent parent input",
                                            )
                                        })
                                    })
                                    .collect::<RusticolResult<Vec<_>>>()?;
                                append_leaf_layouts(chunk, &mapped, output_cursor, leaves)?;
                            }
                        }
                        None => {
                            for chunk in chunks {
                                append_leaf_layouts(chunk, parent_inputs, output_cursor, leaves)?;
                            }
                        }
                    }
                }
                _ => {
                    let (input_len, output_len) = evaluator.io_len()?;
                    if input_len != parent_inputs.len() {
                        return Err(RusticolError::artifact(
                            "evaluator leaf input mapping has an inconsistent length",
                        ));
                    }
                    if output_len == 0 {
                        return Err(RusticolError::artifact(
                            "evaluator leaf has an empty output range",
                        ));
                    }
                    let output_stop = output_cursor.checked_add(output_len).ok_or_else(|| {
                        RusticolError::artifact("evaluator leaf output range overflows usize")
                    })?;
                    leaves.push(EvaluatorLeafLayout {
                        evaluator,
                        input_indices: parent_inputs.to_vec(),
                        output_range: *output_cursor..output_stop,
                    });
                    *output_cursor = output_stop;
                }
            }
            Ok(())
        }

        let (root_input_len, root_output_len) = self.io_len()?;
        let root_inputs = (0..root_input_len).collect::<Vec<_>>();
        let mut output_cursor = 0;
        let mut leaves = Vec::new();
        append_leaf_layouts(self, &root_inputs, &mut output_cursor, &mut leaves)?;
        if output_cursor != root_output_len {
            return Err(RusticolError::artifact(
                "evaluator leaf output ranges do not cover the root output",
            ));
        }
        Ok(leaves)
    }
}

struct EvaluatorGroup {
    evaluators: Vec<LoadedEvaluator>,
    input_len: usize,
    input_mappings: Vec<Option<Vec<usize>>>,
    input_mapping_spans: Vec<Vec<(usize, usize, usize)>>,
    output_len: usize,
    chunk_parameter_scratch_f64: Vec<Complex<f64>>,
    chunk_scratch_f64: Vec<Complex<f64>>,
    chunk_parameter_scratch_aosoa_f64: Vec<f64>,
    chunk_scratch_aosoa_f64: Vec<f64>,
    chunk_input_mapping_scratch: Vec<usize>,
}

// Evaluator groups already own these values behind a Vec allocation. Boxing
// only the SymJIT variant would add an indirection to every hot kernel call.
#[allow(clippy::large_enum_variant)]
enum F64Evaluator {
    #[cfg(feature = "symbolica-runtime")]
    ExactOnly,
    #[cfg(feature = "f64-symjit")]
    SymjitApplication(SymjitApplicationEvaluator),
    #[cfg(feature = "f64-compiled")]
    Compiled(CompiledComplexF64Evaluator),
    #[cfg(feature = "symbolica-runtime")]
    Jit(JITCompiledEvaluator<Complex<f64>>),
}

struct LoadedEvaluator {
    eval: F64Evaluator,
    #[cfg(feature = "symbolica-runtime")]
    exact_eval: Option<ExpressionEvaluator<Complex<Rational>>>,
    #[cfg(feature = "symbolica-runtime")]
    exact_eval_source: Option<EvaluatorPayloadSource>,
    #[cfg(feature = "symbolica-runtime")]
    double_eval: Option<ExpressionEvaluator<Complex<DoubleFloat>>>,
    #[cfg(feature = "symbolica-runtime")]
    arb_eval: Option<(u32, ExpressionEvaluator<Complex<Float>>)>,
    input_len: usize,
    output_len: usize,
}

#[cfg(feature = "symbolica-runtime")]
trait RusticolHighPrecisionNumber:
    Real + RealLike + From<f64> + PartialOrd + Clone + EvaluationDomain
where
    Complex<Self>: Real + EvaluationDomain,
{
    fn evaluate_loaded(
        evaluator: &mut LoadedEvaluator,
        params: &[Complex<Self>],
        out: &mut [Complex<Self>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<()>;
}

#[cfg(feature = "symbolica-runtime")]
impl RusticolHighPrecisionNumber for DoubleFloat {
    fn evaluate_loaded(
        evaluator: &mut LoadedEvaluator,
        params: &[Complex<Self>],
        out: &mut [Complex<Self>],
        _binary_precision: Option<u32>,
    ) -> RusticolResult<()> {
        if evaluator.double_eval.is_none() {
            let exact = evaluator.exact_evaluator()?.clone();
            evaluator.double_eval =
                Some(exact.map_coeff(&|c| {
                    Complex::new(DoubleFloat::from(&c.re), DoubleFloat::from(&c.im))
                }));
        }
        evaluator
            .double_eval
            .as_mut()
            .expect("double evaluator initialized")
            .evaluate(params, out);
        Ok(())
    }
}

#[cfg(feature = "symbolica-runtime")]
impl RusticolHighPrecisionNumber for Float {
    fn evaluate_loaded(
        evaluator: &mut LoadedEvaluator,
        params: &[Complex<Self>],
        out: &mut [Complex<Self>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<()> {
        let binary_precision = binary_precision.ok_or_else(|| {
            RusticolError::invalid_argument(
                "arbitrary-precision evaluation needs a binary precision",
            )
        })?;
        let rebuild = evaluator
            .arb_eval
            .as_ref()
            .map(|(precision, _)| *precision != binary_precision)
            .unwrap_or(true);
        if rebuild {
            let exact = evaluator.exact_evaluator()?.clone();
            evaluator.arb_eval = Some((
                binary_precision,
                exact.map_coeff_with_prec(
                    &|c| {
                        Complex::new(
                            c.re.to_multi_prec_float(binary_precision),
                            c.im.to_multi_prec_float(binary_precision),
                        )
                    },
                    binary_precision,
                ),
            ));
        }
        evaluator
            .arb_eval
            .as_mut()
            .expect("arbitrary-precision evaluator initialized")
            .1
            .evaluate(params, out);
        Ok(())
    }
}

struct RawSumGroup {
    id: i64,
    indices: Vec<usize>,
    weight: f64,
    all_sector_weight: f64,
    sector_ids: Vec<i64>,
}

struct ColorContractionRuntime {
    group_count: usize,
    entries: Vec<ColorContractionEntry>,
    repeated_block: Option<RepeatedColorContractionBlock>,
    symmetric_group: Option<CompiledSymmetricGroupColorContraction>,
    group_scratch_f64: Vec<Complex<f64>>,
}

struct CompiledSymmetricGroupColorContraction {
    plan: crate::recurrence::RecurrenceColorContraction,
    ordered_group_indices: Vec<usize>,
    workspace: Option<crate::recurrence::RuntimeSymmetricGroupColorWorkspace>,
}

#[derive(Clone, Copy)]
struct ColorContractionEntry {
    left_group_index: usize,
    right_group_index: usize,
    weight_re: f64,
    weight_im: f64,
    symmetry_factor: f64,
}

/// One color matrix shared by several disconnected contraction components.
///
/// Helicity-summed NLC/full-color plans contain one isomorphic color block per
/// physical helicity. Keeping only one canonical block avoids streaming the
/// same sparse matrix metadata once per helicity and lays out group values so
/// that the repeated component dimension is contiguous.
struct RepeatedColorContractionBlock {
    component_count: usize,
    component_group_indices: Vec<usize>,
    singleton_output_indices: Option<Vec<usize>>,
    entries: Vec<ColorContractionEntry>,
    all_weights_real: bool,
    walsh_block: Option<WalshColorContractionBlock>,
    c2k_walsh_block: Option<C2kWalshColorContractionBlock>,
}

/// Four real symmetric blocks obtained from a normalized C2 x C2 Walsh basis.
///
/// The cosets map canonical local color groups to each four-point input
/// transform. Entries address the transformed character/coset axis and retain
/// a contiguous repeated-component dimension.
struct WalshColorContractionBlock {
    cosets: Vec<[usize; 4]>,
    entries: Vec<ColorContractionEntry>,
}

/// Real symmetric blocks diagonalized in an unnormalized C2^k Walsh basis.
///
/// Each coset is ordered by the generator bitmask, so the group product is
/// integer XOR. The transformed entries carry the single inverse-subgroup-order
/// factor. Rank three uses a dedicated eight-point butterfly; higher ranks use
/// the same in-place radix-two transform over reusable scratch.
struct C2kWalshColorContractionBlock {
    subgroup_order: usize,
    cosets: Vec<Vec<usize>>,
    entries: Vec<ColorContractionEntry>,
}

impl ColorContractionRuntime {
    fn new(groups: &[RawSumGroup], entries: Vec<ColorContractionEntry>) -> Self {
        let repeated_block = repeated_color_contraction_block(groups, &entries);
        Self {
            group_count: groups.len(),
            entries,
            repeated_block,
            symmetric_group: None,
            group_scratch_f64: Vec::new(),
        }
    }

    fn from_repeated_block(
        groups: &[RawSumGroup],
        component_count: usize,
        component_group_indices: Vec<usize>,
        entries: Vec<ColorContractionEntry>,
        walsh_block: Option<WalshColorContractionBlock>,
        c2k_walsh_block: Option<C2kWalshColorContractionBlock>,
    ) -> Self {
        let singleton_output_indices = component_group_indices
            .iter()
            .map(
                |group_index| match groups[*group_index].indices.as_slice() {
                    [output_index] => Some(*output_index),
                    _ => None,
                },
            )
            .collect::<Option<Vec<_>>>();
        let repeated_block = RepeatedColorContractionBlock {
            component_count,
            component_group_indices,
            singleton_output_indices,
            all_weights_real: entries.iter().all(|entry| entry.weight_im == 0.0),
            entries,
            walsh_block,
            c2k_walsh_block,
        };
        Self {
            group_count: groups.len(),
            entries: Vec::new(),
            repeated_block: Some(repeated_block),
            symmetric_group: None,
            group_scratch_f64: Vec::new(),
        }
    }

    fn from_symmetric_group(
        groups: &[RawSumGroup],
        plan: crate::recurrence::RecurrenceColorContraction,
        ordered_group_indices: Vec<usize>,
    ) -> Self {
        Self {
            group_count: groups.len(),
            entries: Vec::new(),
            repeated_block: None,
            symmetric_group: Some(CompiledSymmetricGroupColorContraction {
                plan,
                ordered_group_indices,
                workspace: None,
            }),
            group_scratch_f64: Vec::new(),
        }
    }

    const fn is_symmetric_group(&self) -> bool {
        self.symmetric_group.is_some()
    }

    fn logical_entry_count(&self) -> RusticolResult<usize> {
        if self.is_symmetric_group() {
            return Err(RusticolError::compatibility(
                "compiled symmetric-group FFT diagnostic contraction has no dense logical-entry view",
            ));
        }
        if self.entries.is_empty() {
            let Some(block) = self.repeated_block.as_ref() else {
                return Ok(0);
            };
            block
                .component_count
                .checked_mul(block.entries.len())
                .ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "repeated colour contraction logical entry count overflows",
                    )
                })
        } else {
            Ok(self.entries.len())
        }
    }

    fn logical_entries(&self) -> ColorContractionEntries<'_> {
        if self.entries.is_empty()
            && let Some(block) = self.repeated_block.as_ref()
        {
            return ColorContractionEntries::Repeated {
                block,
                component_index: 0,
                entry_index: 0,
            };
        }
        ColorContractionEntries::Expanded(self.entries.iter().copied())
    }
}

enum ColorContractionEntries<'a> {
    Expanded(std::iter::Copied<std::slice::Iter<'a, ColorContractionEntry>>),
    Repeated {
        block: &'a RepeatedColorContractionBlock,
        component_index: usize,
        entry_index: usize,
    },
}

impl Iterator for ColorContractionEntries<'_> {
    type Item = ColorContractionEntry;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Expanded(entries) => entries.next(),
            Self::Repeated {
                block,
                component_index,
                entry_index,
            } => {
                if *component_index >= block.component_count || block.entries.is_empty() {
                    return None;
                }
                let entry = block.entries[*entry_index];
                let component = *component_index;
                let left_group_index = block.component_group_indices
                    [entry.left_group_index * block.component_count + component];
                let right_group_index = block.component_group_indices
                    [entry.right_group_index * block.component_count + component];
                *entry_index += 1;
                if *entry_index == block.entries.len() {
                    *entry_index = 0;
                    *component_index += 1;
                }
                Some(ColorContractionEntry {
                    left_group_index,
                    right_group_index,
                    ..entry
                })
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalColorContractionEntry {
    left_group_index: usize,
    right_group_index: usize,
    weight_re_bits: u64,
    weight_im_bits: u64,
    symmetry_factor_bits: u64,
}

fn color_component_root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn repeated_color_contraction_block(
    groups: &[RawSumGroup],
    entries: &[ColorContractionEntry],
) -> Option<RepeatedColorContractionBlock> {
    if groups.len() < 2 || entries.is_empty() {
        return None;
    }

    let mut parent = (0..groups.len()).collect::<Vec<_>>();
    let mut component_size = vec![1usize; groups.len()];
    for entry in entries {
        let mut left = color_component_root(&mut parent, entry.left_group_index);
        let mut right = color_component_root(&mut parent, entry.right_group_index);
        if left == right {
            continue;
        }
        if component_size[left] < component_size[right] {
            std::mem::swap(&mut left, &mut right);
        }
        parent[right] = left;
        component_size[left] += component_size[right];
    }

    let mut components_by_root = BTreeMap::<usize, Vec<usize>>::new();
    for group_index in 0..groups.len() {
        let root = color_component_root(&mut parent, group_index);
        components_by_root
            .entry(root)
            .or_default()
            .push(group_index);
    }
    let mut components = components_by_root.into_values().collect::<Vec<_>>();
    components.sort_by_key(|component| component.iter().copied().min().unwrap_or(usize::MAX));
    if components.len() < 2 {
        return None;
    }
    let groups_per_component = components[0].len();
    if groups_per_component == 0
        || components
            .iter()
            .any(|component| component.len() != groups_per_component)
    {
        return None;
    }

    let component_maps = components
        .iter()
        .map(|component| {
            let mut by_sector_signature = BTreeMap::<Vec<i64>, usize>::new();
            for group_index in component {
                let mut signature = groups[*group_index].sector_ids.clone();
                signature.sort_unstable();
                if by_sector_signature
                    .insert(signature, *group_index)
                    .is_some()
                {
                    return None;
                }
            }
            Some(by_sector_signature)
        })
        .collect::<Option<Vec<_>>>()?;
    let canonical_signatures = component_maps[0].keys().cloned().collect::<Vec<_>>();
    if component_maps
        .iter()
        .skip(1)
        .any(|component| component.keys().ne(canonical_signatures.iter()))
    {
        return None;
    }

    let mut component_index_by_group = vec![usize::MAX; groups.len()];
    let mut local_index_by_group = vec![usize::MAX; groups.len()];
    let mut component_group_indices = Vec::with_capacity(groups.len());
    for (local_index, signature) in canonical_signatures.iter().enumerate() {
        for (component_index, component) in component_maps.iter().enumerate() {
            let group_index = component[signature];
            component_index_by_group[group_index] = component_index;
            local_index_by_group[group_index] = local_index;
            component_group_indices.push(group_index);
        }
    }

    let mut entries_by_component =
        vec![Vec::<CanonicalColorContractionEntry>::new(); components.len()];
    for entry in entries {
        let component_index = component_index_by_group[entry.left_group_index];
        if component_index == usize::MAX
            || component_index != component_index_by_group[entry.right_group_index]
        {
            return None;
        }
        entries_by_component[component_index].push(CanonicalColorContractionEntry {
            left_group_index: local_index_by_group[entry.left_group_index],
            right_group_index: local_index_by_group[entry.right_group_index],
            weight_re_bits: entry.weight_re.to_bits(),
            weight_im_bits: entry.weight_im.to_bits(),
            symmetry_factor_bits: entry.symmetry_factor.to_bits(),
        });
    }
    for component_entries in &mut entries_by_component {
        component_entries.sort_unstable();
    }
    if entries_by_component
        .iter()
        .skip(1)
        .any(|component_entries| component_entries != &entries_by_component[0])
    {
        return None;
    }

    let entries = entries_by_component[0]
        .iter()
        .map(|entry| ColorContractionEntry {
            left_group_index: entry.left_group_index,
            right_group_index: entry.right_group_index,
            weight_re: f64::from_bits(entry.weight_re_bits),
            weight_im: f64::from_bits(entry.weight_im_bits),
            symmetry_factor: f64::from_bits(entry.symmetry_factor_bits),
        })
        .collect::<Vec<_>>();
    let singleton_output_indices = component_group_indices
        .iter()
        .map(
            |group_index| match groups[*group_index].indices.as_slice() {
                [output_index] => Some(*output_index),
                _ => None,
            },
        )
        .collect::<Option<Vec<_>>>();
    Some(RepeatedColorContractionBlock {
        component_count: components.len(),
        component_group_indices,
        singleton_output_indices,
        all_weights_real: entries.iter().all(|entry| entry.weight_im == 0.0),
        entries,
        walsh_block: None,
        c2k_walsh_block: None,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct EvaluatorBatchProfile {
    leaf_input_pack_s: f64,
    legacy_evaluator_call_s: f64,
    evaluator_call_s: f64,
    output_gather_s: f64,
    leaf_input_copy_component_count: u64,
    output_gather_component_count: u64,
    backend_call_count: u64,
    scratch_reallocation_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct StageEvaluationProfile {
    input_pack_s: f64,
    evaluator: EvaluatorBatchProfile,
    output_assign_s: f64,
    input_copy_component_count: u64,
    output_assign_component_count: u64,
    scratch_reallocation_count: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct AmplitudeEvaluationProfile {
    input_pack_s: f64,
    evaluator: EvaluatorBatchProfile,
    output_remap_s: f64,
    input_copy_component_count: u64,
    output_remap_component_count: u64,
    scratch_reallocation_count: u64,
}

#[derive(Clone, Debug, Default)]
struct RuntimeProfile {
    orchestration_s: f64,
    state_prepare_s: f64,
    state_clear_s: f64,
    source_fill_s: f64,
    momentum_input_setup_s: f64,
    momentum_setup_s: f64,
    model_parameter_setup_s: f64,
    stage_input_pack_s: f64,
    stage_leaf_input_pack_s: f64,
    stage_evaluator_call_s: f64,
    stage_backend_call_s: f64,
    stage_evaluator_output_gather_s: f64,
    stage_evaluator_s: f64,
    output_assign_s: f64,
    amplitude_input_pack_s: f64,
    amplitude_leaf_input_pack_s: f64,
    amplitude_evaluator_call_s: f64,
    amplitude_backend_call_s: f64,
    amplitude_evaluator_output_gather_s: f64,
    amplitude_output_remap_s: f64,
    amplitude_evaluator_s: f64,
    reduction_s: f64,
    resolved_reduction_materialization_s: f64,
    total_materialization_s: f64,
    final_output_copy_s: f64,
    total_s: f64,
    stage_input_pack_by_stage_s: Vec<f64>,
    stage_leaf_input_pack_by_stage_s: Vec<f64>,
    stage_evaluator_call_by_stage_s: Vec<f64>,
    stage_backend_call_by_stage_s: Vec<f64>,
    stage_evaluator_output_gather_by_stage_s: Vec<f64>,
    stage_output_assign_by_stage_s: Vec<f64>,
    state_component_count: u64,
    state_clear_component_count: u64,
    source_component_count: u64,
    momentum_component_count: u64,
    model_parameter_component_count: u64,
    stage_input_copy_component_count: u64,
    stage_leaf_input_copy_component_count: u64,
    stage_evaluator_output_gather_component_count: u64,
    stage_output_assign_component_count: u64,
    amplitude_input_copy_component_count: u64,
    amplitude_leaf_input_copy_component_count: u64,
    amplitude_evaluator_output_gather_component_count: u64,
    amplitude_output_remap_component_count: u64,
    evaluator_backend_call_count: u64,
    compiled_direct_arena_engine_count: u64,
    compiled_direct_arena_call_count: u64,
    compiled_direct_arena_minimum_effective_tile_capacity: u64,
    compiled_direct_arena_maximum_physical_scalar_values_per_point: u64,
    compiled_direct_arena_maximum_hot_scalar_values_per_point: u64,
    compiled_direct_arena_maximum_source_scalar_values_per_point: u64,
    compiled_direct_arena_maximum_reduction_scalar_values_per_point: u64,
    compiled_direct_arena_boundary_input_bytes: u64,
    compiled_direct_arena_boundary_current_output_bytes: u64,
    compiled_direct_arena_boundary_amplitude_output_bytes: u64,
    /// Broadcast-plane refresh traffic internal to compiled P-kernels.
    compiled_direct_arena_internal_broadcast_bytes: u64,
    reduction_input_component_count: u64,
    resolved_materialized_component_count: u64,
    total_materialized_value_count: u64,
    final_output_copy_value_count: u64,
    scratch_reallocation_count: u64,
    eager_initialize_s: f64,
    eager_gather_s: f64,
    eager_kernel_call_s: f64,
    eager_invocation_scatter_s: f64,
    eager_finalization_s: f64,
    eager_scatter_finalization_s: f64,
    eager_closure_s: f64,
    eager_reduction_s: f64,
    eager_copy_out_s: f64,
    /// P-kernel scratch reads and writes wholly inside eager Direct-Arena execution.
    eager_internal_scratch_bytes: u64,
    /// Point-independent coupling and parameter refreshes wholly inside eager
    /// Direct-Arena execution.
    eager_internal_broadcast_bytes: u64,
    recurrence_momentum_fill_s: f64,
    recurrence_union_source_fill_s: f64,
    /// Inclusive top-level recurrence schedule envelope. The role-specific
    /// recurrence fields below are internal attribution and must not be added
    /// to top-level accounting.
    recurrence_schedule_s: f64,
    recurrence_source_kernel_s: f64,
    recurrence_contribution_kernel_s: f64,
    recurrence_finalization_s: f64,
    recurrence_closure_s: f64,
    recurrence_replay_output_mapping_s: f64,
    recurrence_momentum_scalar_value_count: u64,
    recurrence_schedule_execution_count: u64,
    recurrence_replay_schedule_execution_count: u64,
    recurrence_union_schedule_execution_count: u64,
    recurrence_union_source_row_count: u64,
    recurrence_replay_output_value_count: u64,
    recurrence_source_call_count: u64,
    recurrence_source_row_count: u64,
    recurrence_contribution_call_count: u64,
    recurrence_contribution_row_count: u64,
    recurrence_finalization_call_count: u64,
    recurrence_finalization_row_count: u64,
    recurrence_closure_call_count: u64,
    recurrence_closure_row_count: u64,
    /// Legacy recurrence Direct-Arena packet counters. These remain private
    /// because they authenticate the generated-kernel boundary rather than
    /// the public diagnostic profiler's input/output materialization.
    recurrence_direct_packed_input_bytes: u64,
    recurrence_direct_packed_output_bytes: u64,
    recurrence_direct_scatter_bytes: u64,
    /// Lane-neutral Direct-Arena traffic counters observed by recurrence.
    recurrence_direct_packet_input_bytes: u64,
    recurrence_direct_packet_output_bytes: u64,
    recurrence_direct_gather_bytes: u64,
    recurrence_direct_traffic_scatter_bytes: u64,
    recurrence_direct_remap_bytes: u64,
    /// P-kernel output scratch traffic entirely inside recurrence execution.
    recurrence_internal_scratch_bytes: u64,
    /// P-kernel broadcast-plane traffic entirely inside recurrence execution.
    recurrence_internal_broadcast_bytes: u64,
}

// Saved SymJIT applications are native payloads and do not currently guarantee
// Rust's AArch64 callee-saved FP registers. Keep elapsed values integer-backed
// until the payload has returned, then use an opaque conversion boundary so
// LLVM cannot retain an f64 across that call.
#[inline(never)]
fn profile_duration_seconds(duration: Duration) -> f64 {
    std::hint::black_box(duration).as_secs_f64()
}

impl RuntimeProfile {
    fn add_sector(&mut self, sector: &RuntimeProfile) {
        self.orchestration_s += sector.orchestration_s;
        self.state_prepare_s += sector.state_prepare_s;
        self.state_clear_s += sector.state_clear_s;
        self.source_fill_s += sector.source_fill_s;
        self.momentum_input_setup_s += sector.momentum_input_setup_s;
        self.momentum_setup_s += sector.momentum_setup_s;
        self.model_parameter_setup_s += sector.model_parameter_setup_s;
        self.stage_input_pack_s += sector.stage_input_pack_s;
        self.stage_leaf_input_pack_s += sector.stage_leaf_input_pack_s;
        self.stage_evaluator_call_s += sector.stage_evaluator_call_s;
        self.stage_backend_call_s += sector.stage_backend_call_s;
        self.stage_evaluator_output_gather_s += sector.stage_evaluator_output_gather_s;
        self.stage_evaluator_s += sector.stage_evaluator_s;
        self.output_assign_s += sector.output_assign_s;
        self.amplitude_input_pack_s += sector.amplitude_input_pack_s;
        self.amplitude_leaf_input_pack_s += sector.amplitude_leaf_input_pack_s;
        self.amplitude_evaluator_call_s += sector.amplitude_evaluator_call_s;
        self.amplitude_backend_call_s += sector.amplitude_backend_call_s;
        self.amplitude_evaluator_output_gather_s += sector.amplitude_evaluator_output_gather_s;
        self.amplitude_output_remap_s += sector.amplitude_output_remap_s;
        self.amplitude_evaluator_s += sector.amplitude_evaluator_s;
        self.reduction_s += sector.reduction_s;
        self.resolved_reduction_materialization_s += sector.resolved_reduction_materialization_s;
        self.total_materialization_s += sector.total_materialization_s;
        self.final_output_copy_s += sector.final_output_copy_s;
        self.state_component_count += sector.state_component_count;
        self.state_clear_component_count += sector.state_clear_component_count;
        self.source_component_count += sector.source_component_count;
        self.momentum_component_count += sector.momentum_component_count;
        self.model_parameter_component_count += sector.model_parameter_component_count;
        self.stage_input_copy_component_count += sector.stage_input_copy_component_count;
        self.stage_leaf_input_copy_component_count += sector.stage_leaf_input_copy_component_count;
        self.stage_evaluator_output_gather_component_count +=
            sector.stage_evaluator_output_gather_component_count;
        self.stage_output_assign_component_count += sector.stage_output_assign_component_count;
        self.amplitude_input_copy_component_count += sector.amplitude_input_copy_component_count;
        self.amplitude_leaf_input_copy_component_count +=
            sector.amplitude_leaf_input_copy_component_count;
        self.amplitude_evaluator_output_gather_component_count +=
            sector.amplitude_evaluator_output_gather_component_count;
        self.amplitude_output_remap_component_count +=
            sector.amplitude_output_remap_component_count;
        self.evaluator_backend_call_count += sector.evaluator_backend_call_count;
        self.compiled_direct_arena_engine_count += sector.compiled_direct_arena_engine_count;
        self.compiled_direct_arena_call_count += sector.compiled_direct_arena_call_count;
        self.compiled_direct_arena_minimum_effective_tile_capacity = match (
            self.compiled_direct_arena_minimum_effective_tile_capacity,
            sector.compiled_direct_arena_minimum_effective_tile_capacity,
        ) {
            (0, value) | (value, 0) => value,
            (left, right) => left.min(right),
        };
        self.compiled_direct_arena_maximum_physical_scalar_values_per_point = self
            .compiled_direct_arena_maximum_physical_scalar_values_per_point
            .max(sector.compiled_direct_arena_maximum_physical_scalar_values_per_point);
        self.compiled_direct_arena_maximum_hot_scalar_values_per_point = self
            .compiled_direct_arena_maximum_hot_scalar_values_per_point
            .max(sector.compiled_direct_arena_maximum_hot_scalar_values_per_point);
        self.compiled_direct_arena_maximum_source_scalar_values_per_point = self
            .compiled_direct_arena_maximum_source_scalar_values_per_point
            .max(sector.compiled_direct_arena_maximum_source_scalar_values_per_point);
        self.compiled_direct_arena_maximum_reduction_scalar_values_per_point = self
            .compiled_direct_arena_maximum_reduction_scalar_values_per_point
            .max(sector.compiled_direct_arena_maximum_reduction_scalar_values_per_point);
        self.compiled_direct_arena_boundary_input_bytes +=
            sector.compiled_direct_arena_boundary_input_bytes;
        self.compiled_direct_arena_boundary_current_output_bytes +=
            sector.compiled_direct_arena_boundary_current_output_bytes;
        self.compiled_direct_arena_boundary_amplitude_output_bytes +=
            sector.compiled_direct_arena_boundary_amplitude_output_bytes;
        self.compiled_direct_arena_internal_broadcast_bytes +=
            sector.compiled_direct_arena_internal_broadcast_bytes;
        self.reduction_input_component_count += sector.reduction_input_component_count;
        self.resolved_materialized_component_count += sector.resolved_materialized_component_count;
        self.total_materialized_value_count += sector.total_materialized_value_count;
        self.final_output_copy_value_count += sector.final_output_copy_value_count;
        self.scratch_reallocation_count += sector.scratch_reallocation_count;
        self.eager_initialize_s += sector.eager_initialize_s;
        self.eager_gather_s += sector.eager_gather_s;
        self.eager_kernel_call_s += sector.eager_kernel_call_s;
        self.eager_invocation_scatter_s += sector.eager_invocation_scatter_s;
        self.eager_finalization_s += sector.eager_finalization_s;
        self.eager_scatter_finalization_s += sector.eager_scatter_finalization_s;
        self.eager_closure_s += sector.eager_closure_s;
        self.eager_reduction_s += sector.eager_reduction_s;
        self.eager_copy_out_s += sector.eager_copy_out_s;
        self.eager_internal_scratch_bytes += sector.eager_internal_scratch_bytes;
        self.eager_internal_broadcast_bytes += sector.eager_internal_broadcast_bytes;
        self.recurrence_momentum_fill_s += sector.recurrence_momentum_fill_s;
        self.recurrence_union_source_fill_s += sector.recurrence_union_source_fill_s;
        self.recurrence_schedule_s += sector.recurrence_schedule_s;
        self.recurrence_source_kernel_s += sector.recurrence_source_kernel_s;
        self.recurrence_contribution_kernel_s += sector.recurrence_contribution_kernel_s;
        self.recurrence_finalization_s += sector.recurrence_finalization_s;
        self.recurrence_closure_s += sector.recurrence_closure_s;
        self.recurrence_replay_output_mapping_s += sector.recurrence_replay_output_mapping_s;
        self.recurrence_momentum_scalar_value_count +=
            sector.recurrence_momentum_scalar_value_count;
        self.recurrence_schedule_execution_count += sector.recurrence_schedule_execution_count;
        self.recurrence_replay_schedule_execution_count +=
            sector.recurrence_replay_schedule_execution_count;
        self.recurrence_union_schedule_execution_count +=
            sector.recurrence_union_schedule_execution_count;
        self.recurrence_union_source_row_count += sector.recurrence_union_source_row_count;
        self.recurrence_replay_output_value_count += sector.recurrence_replay_output_value_count;
        self.recurrence_source_call_count += sector.recurrence_source_call_count;
        self.recurrence_source_row_count += sector.recurrence_source_row_count;
        self.recurrence_contribution_call_count += sector.recurrence_contribution_call_count;
        self.recurrence_contribution_row_count += sector.recurrence_contribution_row_count;
        self.recurrence_finalization_call_count += sector.recurrence_finalization_call_count;
        self.recurrence_finalization_row_count += sector.recurrence_finalization_row_count;
        self.recurrence_closure_call_count += sector.recurrence_closure_call_count;
        self.recurrence_closure_row_count += sector.recurrence_closure_row_count;
        self.recurrence_direct_packed_input_bytes += sector.recurrence_direct_packed_input_bytes;
        self.recurrence_direct_packed_output_bytes += sector.recurrence_direct_packed_output_bytes;
        self.recurrence_direct_scatter_bytes += sector.recurrence_direct_scatter_bytes;
        self.recurrence_direct_packet_input_bytes += sector.recurrence_direct_packet_input_bytes;
        self.recurrence_direct_packet_output_bytes += sector.recurrence_direct_packet_output_bytes;
        self.recurrence_direct_gather_bytes += sector.recurrence_direct_gather_bytes;
        self.recurrence_direct_traffic_scatter_bytes +=
            sector.recurrence_direct_traffic_scatter_bytes;
        self.recurrence_direct_remap_bytes += sector.recurrence_direct_remap_bytes;
        self.recurrence_internal_scratch_bytes += sector.recurrence_internal_scratch_bytes;
        self.recurrence_internal_broadcast_bytes += sector.recurrence_internal_broadcast_bytes;
        add_profile_vector(
            &mut self.stage_input_pack_by_stage_s,
            &sector.stage_input_pack_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_leaf_input_pack_by_stage_s,
            &sector.stage_leaf_input_pack_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_evaluator_call_by_stage_s,
            &sector.stage_evaluator_call_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_backend_call_by_stage_s,
            &sector.stage_backend_call_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_evaluator_output_gather_by_stage_s,
            &sector.stage_evaluator_output_gather_by_stage_s,
        );
        add_profile_vector(
            &mut self.stage_output_assign_by_stage_s,
            &sector.stage_output_assign_by_stage_s,
        );
    }

    fn validate_recurrence_direct_boundary_traffic(&self) -> RusticolResult<()> {
        if self.recurrence_direct_packed_input_bytes
            | self.recurrence_direct_packed_output_bytes
            | self.recurrence_direct_scatter_bytes
            != 0
        {
            return Err(RusticolError::integrity(
                "recurrence Direct-Arena execution recorded legacy pack/scatter boundary traffic",
            ));
        }
        if self.recurrence_direct_packet_input_bytes
            | self.recurrence_direct_packet_output_bytes
            | self.recurrence_direct_gather_bytes
            | self.recurrence_direct_traffic_scatter_bytes
            | self.recurrence_direct_remap_bytes
            != 0
        {
            return Err(RusticolError::integrity(
                "recurrence Direct-Arena execution recorded packet/gather/scatter/remap traffic",
            ));
        }
        Ok(())
    }
}

fn add_profile_vector(target: &mut Vec<f64>, source: &[f64]) {
    if target.len() < source.len() {
        target.resize(source.len(), 0.0);
    }
    for (index, value) in source.iter().enumerate() {
        target[index] += value;
    }
}

struct ExecutionRuntime {
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
    lc_topology_replay_enabled: bool,
    lc_topology_replay_mappings: Arc<LcTopologyReplayMappings>,
    lc_topology_replay_public_mappings: LcTopologyReplayMappings,
    lc_topology_replay_routes: Vec<Vec<LcTopologyReplaySectorRoute>>,
    lc_topology_replay_materialized_sector_ids: BTreeSet<i64>,
    lc_resolved_replay_plan: Option<Arc<LcResolvedReplayPlan>>,
    lc_resolved_replay_selection_cache:
        Option<(LcResolvedReplaySelectionKey, Arc<LcResolvedReplaySelection>)>,
    lc_replay_flat_momenta_scratch: Vec<f64>,
    lc_replay_target_components_scratch: Vec<f64>,
    color_topology_replay_enabled: bool,
    color_topology_replay_mappings: Arc<LcTopologyReplayMappings>,
    color_replay_flat_momenta_scratch: Vec<f64>,
    #[allow(dead_code)] // Loaded now and consumed by the subsequent selector-execution milestone.
    helicity_recurrence: Option<HelicityRecurrenceRuntime>,
    compiled_helicity_execution_plan: Option<CompiledHelicityExecutionPlan>,
    compiled_color_execution_plan: Option<CompiledColorExecutionPlan>,
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    compiled_direct_runtime: Option<compiled_direct_prototype::CompiledDirectEnginePrototype>,
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    compiled_direct_color_schedules:
        BTreeMap<i64, compiled_direct_prototype::CompiledDirectValidatedSchedule>,
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    compiled_direct_helicity_schedules:
        BTreeMap<usize, compiled_direct_prototype::CompiledDirectValidatedSchedule>,
    helicity_sum_runtime: Option<Box<ExecutionRuntime>>,
    // Lane runtimes are large recursive owners; boxing keeps their addresses stable and avoids
    // moving them when this selector index grows.
    #[allow(clippy::vec_box)]
    helicity_selector_runtimes: Vec<Box<ExecutionRuntime>>,
    helicity_selector_runtime_schedule_modes: Vec<HelicitySelectorScheduleMode>,
    helicity_selector_lane_by_domain: BTreeMap<usize, usize>,
    color_selector_runtimes: BTreeMap<i64, Box<ExecutionRuntime>>,
    runtime_unavailable_message: Option<String>,
    sources: Vec<GenericSourceRecordManifest>,
    // Iterable fill list, not indexed by canonical momentum-slot ID. Compiled
    // lanes retain only generated live slots, with their global offsets intact.
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
    model_parameters: Vec<GenericRuntimeModelParameterManifest>,
    model_parameter_name_to_index: BTreeMap<String, usize>,
    model_parameter_runtime_slots: BTreeMap<String, RuntimeParameterSlots>,
    model_parameter_values_f64: Vec<f64>,
    model_parameter_evaluator: Option<ModelParameterEvaluatorRuntime>,
    physics_reduction_override: Option<crate::Reduction>,
    physics: Option<Arc<PhysicsRuntime>>,
    stages: Option<Vec<StageRuntime>>,
    amplitude_stage: Option<AmplitudeRuntime>,
    state_scratch_f64: Vec<Complex<f64>>,
    state_scratch_f64_requires_clear: bool,
    values_scratch_f64: Vec<f64>,
}

#[derive(Clone)]
struct PhysicsRuntime {
    binding_id: u64,
    manifest: ProcessPhysicsV1,
    helicity_index_by_id: BTreeMap<String, usize>,
    helicity_members_by_representative: Vec<Vec<usize>>,
    color_index_by_id: BTreeMap<String, usize>,
    reduction_by_group_id: BTreeMap<i64, crate::ReductionGroup>,
    numeric_reduction_by_group_id: BTreeMap<i64, NumericReductionGroup>,
}

#[derive(Clone)]
struct NumericReductionGroup {
    physical_helicity_indices: Vec<usize>,
    helicity_membership: Vec<bool>,
    normalized_helicity_weights: Vec<(usize, f64)>,
    normalized_color_weights: Vec<(usize, f64)>,
    normalized_member_weights: Vec<(usize, usize, f64)>,
}

impl NumericReductionGroup {
    #[inline(always)]
    fn contains_helicity(&self, helicity_index: usize) -> bool {
        self.helicity_membership
            .get(helicity_index)
            .copied()
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug)]
struct ResolvedValues<T> {
    values: Vec<T>,
    point_count: usize,
    helicity_indices: Vec<usize>,
    color_indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeRuntimeMetadata {
    pub abi_version: u32,
    pub schema_version: u32,
    pub execution_mode: String,
    pub prepared_backend: Option<String>,
    pub eager_effective_point_tile_size: Option<usize>,
    pub eager_workspace_bytes: Option<usize>,
    pub on_the_fly_requested_query_construction_threads: Option<usize>,
    pub on_the_fly_effective_query_construction_threads: Option<usize>,
    pub compiled_direct_minimum_effective_tile_capacity: Option<usize>,
    pub compiled_direct_maximum_physical_scalar_values_per_point: Option<usize>,
    pub compiled_direct_maximum_hot_scalar_values_per_point: Option<usize>,
    pub compiled_direct_maximum_source_scalar_values_per_point: Option<usize>,
    pub compiled_direct_maximum_reduction_scalar_values_per_point: Option<usize>,
    pub process: String,
    pub process_key: String,
    pub representative_process: String,
    pub representative_process_key: String,
    /// Full representative-index to public-index external permutation.
    pub external_permutation: Vec<usize>,
    /// Representative process key when a stored or inferred permutation is active.
    pub permutation_alias_of: Option<String>,
    pub final_state_permutation_alias_of: Option<String>,
    /// Representative process key when the selected process was not generated
    /// and is routed through the artifact's process-group map.
    pub process_group_member_of: Option<String>,
    /// Exact factor applied to every representative matrix element.
    pub process_group_factor: NativeRationalFactor,
    /// Whether the selected process is the charge conjugate of its
    /// representative; public LC color-flow words are then reversed.
    pub charge_conjugated: bool,
    pub color_accuracy: String,
    pub external_pdg_order: Vec<i32>,
    pub external_count: usize,
    pub current_count: usize,
    pub source_count: usize,
    pub interaction_count: usize,
    pub stage_count: usize,
    pub amplitude_output_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct NativeRationalFactor {
    pub numerator: u64,
    pub denominator: u64,
}

/// Public relabelling of a process-group member evaluated by its representative.
#[derive(Clone, Debug)]
struct ProcessGroupRouting {
    representative_process_key: String,
    factor: NativeRationalFactor,
    conjugated: bool,
}

/// Prepared batch-global selectors for allocation-free recurrence evaluation.
///
/// Selector resolution and validation may allocate while this handle is
/// created. Reusing it with
/// [`NativeRuntime::evaluate_f64_into_with_recurrence_selector_plan`] borrows
/// the retained selector sets and caller-owned input/output storage.
#[derive(Clone, Debug)]
pub struct NativeRecurrenceSelectorPlan {
    artifact_root: PathBuf,
    process_key: String,
    external_permutation: Vec<usize>,
    selected_helicities: Option<BTreeSet<String>>,
    selected_colors: Option<BTreeSet<String>>,
}

impl NativeRecurrenceSelectorPlan {
    fn ensure_matches(&self, runtime: &NativeRuntime) -> RusticolResult<()> {
        if self.artifact_root != runtime.root
            || self.process_key != runtime.process_key
            || self.external_permutation != runtime.external_permutation
        {
            return Err(RusticolError::selector(
                "recurrence selector plan belongs to a different artifact, process, or external ordering",
            ));
        }
        Ok(())
    }
}

/// Exact-required sections decoded lazily from one authenticated eager plan-v3
/// artifact. This bridge is intentionally private to pyAmpliCol's Python exact
/// executor; f64 execution never constructs it.
#[derive(Debug)]
pub struct NativeEagerExactSections {
    pub process_id: String,
    pub exact_schema: Value,
    pub reduction_groups: Value,
    pub selector_group_ids: Vec<u32>,
    pub selector_domains: Vec<Vec<u32>>,
    pub couplings: Vec<NativeEagerExactCoupling>,
    pub stages: Vec<NativeEagerExactStage>,
    pub invocations: Vec<NativeEagerExactInvocation>,
    pub attachments: Vec<NativeEagerExactAttachment>,
    pub finalizations: Vec<NativeEagerExactFinalization>,
    pub closures: Vec<NativeEagerExactClosure>,
}

#[derive(Debug)]
pub struct NativeEagerExactCoupling {
    pub real_parameter_id: u32,
    pub imaginary_parameter_id: u32,
    pub constant_real: String,
    pub constant_imaginary: String,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeEagerExactStage {
    pub stage_index: u32,
    pub invocation_start: u64,
    pub invocation_count: u64,
    pub attachment_start: u64,
    pub attachment_count: u64,
    pub finalization_start: u64,
    pub finalization_count: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeEagerExactInvocation {
    pub kernel_id: u32,
    pub left_value_slot_id: u32,
    pub right_value_slot_id: u32,
    pub left_momentum_slot_id: u32,
    pub right_momentum_slot_id: u32,
    pub coupling_slot_id: u32,
    pub output_factor_source: u8,
    pub attachment_start: u64,
    pub attachment_count: u64,
    pub selector_domain_id: u32,
}

#[derive(Debug)]
pub struct NativeEagerExactAttachment {
    pub result_current_id: u32,
    pub factor_numerators: Vec<[String; 2]>,
    pub factor_denominator: Option<[String; 2]>,
    pub selector_domain_id: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeEagerExactFinalization {
    pub kernel_id: u32,
    pub current_id: u32,
    pub unpropagated_value_slot_id: u32,
    pub propagated_value_slot_id: u32,
    pub momentum_slot_id: u32,
    pub unpropagated_selector_domain_id: u32,
    pub propagated_selector_domain_id: u32,
}

#[derive(Debug)]
pub struct NativeEagerExactClosure {
    pub kernel_id: u32,
    pub left_value_slot_id: u32,
    pub right_value_slot_id: u32,
    pub amplitude_index: u32,
    pub coupling_slot_id: u32,
    pub output_factor_source: u8,
    pub factor_numerators: Vec<[String; 2]>,
    pub factor_denominator: Option<[String; 2]>,
    pub direct_coefficients: Option<Vec<[String; 2]>>,
    pub coherent_group_id: u32,
    pub selector_domain_id: u32,
}

/// Exact-required sections decoded from one authenticated recurrence plan-v2
/// artifact. Python consumes these immutable rows with the retained exact
/// evaluator payloads; the Direct-Arena f64 runtime never constructs them.
#[derive(Debug)]
pub struct NativeRecurrenceExactSections {
    pub process_id: String,
    pub strategy: String,
    pub semantic_digest: String,
    pub runtime_layout_digest: String,
    pub current_arena_components: u32,
    pub amplitude_destination_count: u32,
    pub parameter_value_count: u32,
    pub external_source_count: u32,
    pub currents: Vec<crate::recurrence::DirectCurrentDescriptor>,
    pub sources: Vec<crate::recurrence::DirectSourceRow>,
    pub contributions: Vec<crate::recurrence::DirectContributionRow>,
    pub finalizations: Vec<crate::recurrence::DirectFinalizationRow>,
    pub closures: Vec<crate::recurrence::DirectClosureRow>,
    pub row_groups: Vec<crate::recurrence::DirectRowGroupDescriptor>,
    pub momentum_forms: Vec<crate::recurrence::DirectMomentumFormDescriptor>,
    pub momentum_terms: Vec<crate::recurrence::DirectMomentumTerm>,
    pub replay_targets: Vec<crate::recurrence::DirectReplayTargetDescriptor>,
    pub source_permutations: Vec<u32>,
    pub replay_momentum_signs: Vec<i32>,
    pub replay_helicity_map: Vec<u32>,
    pub amplitude_destinations: Vec<crate::recurrence::DirectAmplitudeDestinationDescriptor>,
    pub resolved_helicities: Vec<crate::recurrence::DirectResolvedHelicityDescriptor>,
    pub public_helicities: Vec<i32>,
    pub source_state_assignments: Vec<crate::recurrence::DirectSourceStateAssignment>,
    pub source_dispatch_variants: Vec<crate::recurrence::DirectSourceDispatchVariantDescriptor>,
    pub source_embeddings: Vec<crate::recurrence::DirectSourceEmbeddingRow>,
    pub source_projections: Vec<crate::recurrence::DirectSourceProjectionRow>,
    pub resolved_source_selections: Vec<crate::recurrence::DirectResolvedSourceSelection>,
    pub exact_factors: Vec<NativeRecurrenceExactFactor>,
    pub public_flow_ids: Vec<u32>,
    pub executors: Vec<NativeRecurrenceExactExecutor>,
}

#[derive(Debug)]
pub struct NativeRecurrenceExactFactor {
    pub real_numerator: String,
    pub real_denominator: String,
    pub imaginary_numerator: String,
    pub imaginary_denominator: String,
}

#[derive(Clone, Debug)]
pub struct NativeRecurrenceExactExecutor {
    pub direct_executor_id: u32,
    pub role: String,
    pub destination_operation: String,
    pub parent_component_counts: Vec<u32>,
    pub destination_component_count: u32,
    pub momentum_operand_count: u32,
    pub prepared_kernel_id: Option<u32>,
    pub runtime_template: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeRuntimeProfile {
    pub native_input_pack_s: f64,
    pub native_input_crossing_s: f64,
    pub orchestration_s: f64,
    pub state_prepare_s: f64,
    pub state_clear_s: f64,
    pub source_fill_s: f64,
    /// Exclusive momentum-input setup. Unlike `momentum_setup_s`, this does
    /// not include model-parameter setup.
    pub momentum_input_setup_s: f64,
    /// Backward-compatible aggregate of momentum-input and model-parameter
    /// setup.
    pub momentum_setup_s: f64,
    pub model_parameter_setup_s: f64,
    /// Top-level stage input envelope. In composed selected-chunk paths this
    /// owns the leaf gather; in full-stage paths it owns only the parent
    /// stage-input gather.
    pub stage_input_pack_s: f64,
    /// Internal attribution. This is owned by `stage_input_pack_s` for
    /// composed selected-chunk paths and by `stage_evaluator_call_s` for
    /// full-stage paths, so it must not be added to the top-level sum.
    pub stage_leaf_input_pack_s: f64,
    /// Top-level evaluator envelope. This includes leaf gathering for
    /// full-stage paths and excludes it for composed selected-chunk paths.
    pub stage_evaluator_call_s: f64,
    pub stage_backend_call_s: f64,
    pub stage_evaluator_output_gather_s: f64,
    pub stage_evaluator_s: f64,
    pub output_assign_s: f64,
    /// Top-level amplitude input envelope. In composed selected-chunk paths
    /// this owns the amplitude leaf gather.
    pub amplitude_input_pack_s: f64,
    /// Internal attribution owned either by `amplitude_input_pack_s` or
    /// `amplitude_evaluator_call_s`; never an additional top-level phase.
    pub amplitude_leaf_input_pack_s: f64,
    /// Top-level amplitude evaluator envelope. Full-stage paths include leaf
    /// gathering; composed selected-chunk paths exclude it.
    pub amplitude_evaluator_call_s: f64,
    pub amplitude_backend_call_s: f64,
    pub amplitude_evaluator_output_gather_s: f64,
    pub amplitude_output_remap_s: f64,
    pub amplitude_evaluator_s: f64,
    pub reduction_s: f64,
    /// Inclusive attribution: the resolved-result construction occurs inside
    /// `reduction_s` and must not be added to exclusive top-level phases.
    pub resolved_reduction_materialization_s: f64,
    pub total_materialization_s: f64,
    pub final_output_copy_s: f64,
    pub total_s: f64,
    pub stage_input_pack_by_stage_s: Vec<f64>,
    pub stage_leaf_input_pack_by_stage_s: Vec<f64>,
    pub stage_evaluator_call_by_stage_s: Vec<f64>,
    pub stage_backend_call_by_stage_s: Vec<f64>,
    pub stage_evaluator_output_gather_by_stage_s: Vec<f64>,
    pub stage_output_assign_by_stage_s: Vec<f64>,
    pub eager_initialize_s: f64,
    pub eager_gather_s: f64,
    pub eager_kernel_call_s: f64,
    pub eager_invocation_scatter_s: f64,
    pub eager_finalization_s: f64,
    pub eager_scatter_finalization_s: f64,
    pub eager_closure_s: f64,
    pub eager_reduction_s: f64,
    pub eager_copy_out_s: f64,
    pub recurrence_momentum_fill_s: f64,
    pub recurrence_union_source_fill_s: f64,
    /// Inclusive top-level recurrence schedule envelope.
    pub recurrence_schedule_s: f64,
    /// Internal attribution owned by `recurrence_schedule_s`.
    pub recurrence_source_kernel_s: f64,
    /// Internal attribution owned by `recurrence_schedule_s`.
    pub recurrence_contribution_kernel_s: f64,
    /// Internal attribution owned by `recurrence_schedule_s`.
    pub recurrence_finalization_s: f64,
    /// Internal attribution owned by `recurrence_schedule_s`.
    pub recurrence_closure_s: f64,
    pub recurrence_replay_output_mapping_s: f64,
    pub recurrence_momentum_scalar_value_count: u64,
    pub recurrence_schedule_execution_count: u64,
    pub recurrence_replay_schedule_execution_count: u64,
    pub recurrence_union_schedule_execution_count: u64,
    pub recurrence_union_source_row_count: u64,
    pub recurrence_replay_output_value_count: u64,
    pub recurrence_source_call_count: u64,
    pub recurrence_source_row_count: u64,
    pub recurrence_contribution_call_count: u64,
    pub recurrence_contribution_row_count: u64,
    pub recurrence_finalization_call_count: u64,
    pub recurrence_finalization_row_count: u64,
    pub recurrence_closure_call_count: u64,
    pub recurrence_closure_row_count: u64,
    pub selector_planner_s: f64,
    pub selector_gather_s: f64,
    pub selector_scatter_s: f64,
    pub selector_plan_kind: String,
    pub selector_group_sizes: Vec<usize>,
    pub selector_reordered_point_count: usize,
    pub selector_simd_lane_width: usize,
    pub selector_simd_occupancy: f64,
    pub native_input_component_count: u64,
    pub native_input_pack_bytes: u64,
    pub native_input_crossing_bytes: u64,
    /// Explicit nested native-input containers allocated for this call.
    pub native_input_container_allocation_count: u64,
    pub state_component_count: u64,
    pub state_clear_component_count: u64,
    pub source_component_count: u64,
    pub momentum_component_count: u64,
    pub model_parameter_component_count: u64,
    pub stage_input_copy_component_count: u64,
    pub stage_leaf_input_copy_component_count: u64,
    pub stage_evaluator_output_gather_component_count: u64,
    pub stage_output_assign_component_count: u64,
    pub amplitude_input_copy_component_count: u64,
    pub amplitude_leaf_input_copy_component_count: u64,
    pub amplitude_evaluator_output_gather_component_count: u64,
    pub amplitude_output_remap_component_count: u64,
    pub evaluator_backend_call_count: u64,
    /// Number of compiled Direct-Arena engines authenticated for this profile.
    pub compiled_direct_arena_engine_count: u64,
    /// Calls observed directly on compiled Direct-Arena evaluator leaves.
    pub compiled_direct_arena_call_count: u64,
    /// Minimum effective tile across the compiled Direct engine fleet.
    pub compiled_direct_arena_minimum_effective_tile_capacity: u64,
    /// Maximum authenticated physical scalar-plane footprint per point.
    pub compiled_direct_arena_maximum_physical_scalar_values_per_point: u64,
    /// Maximum phase-local cache footprint per point.
    pub compiled_direct_arena_maximum_hot_scalar_values_per_point: u64,
    pub compiled_direct_arena_maximum_source_scalar_values_per_point: u64,
    pub compiled_direct_arena_maximum_reduction_scalar_values_per_point: u64,
    /// Developer-adapter traffic is forbidden in the production Arena lane.
    pub compiled_direct_arena_boundary_input_bytes: u64,
    pub compiled_direct_arena_boundary_current_output_bytes: u64,
    pub compiled_direct_arena_boundary_amplitude_output_bytes: u64,
    pub reduction_input_component_count: u64,
    pub selector_gather_point_count: u64,
    pub selector_gather_bytes: u64,
    pub selector_scatter_value_count: u64,
    pub resolved_materialized_component_count: u64,
    pub total_materialized_value_count: u64,
    pub final_output_copy_value_count: u64,
    /// Capacity-changing reallocations observed in instrumented reusable hot
    /// buffers. This is intentionally not a process-wide allocation count.
    pub observed_scratch_reallocation_count: u64,
    /// Explicit final native output vector allocated for this call.
    pub native_output_allocation_count: u64,
}

impl From<RuntimeProfile> for NativeRuntimeProfile {
    fn from(profile: RuntimeProfile) -> Self {
        Self {
            native_input_pack_s: 0.0,
            native_input_crossing_s: 0.0,
            orchestration_s: profile.orchestration_s,
            state_prepare_s: profile.state_prepare_s,
            state_clear_s: profile.state_clear_s,
            source_fill_s: profile.source_fill_s,
            momentum_input_setup_s: profile.momentum_input_setup_s,
            momentum_setup_s: profile.momentum_setup_s,
            model_parameter_setup_s: profile.model_parameter_setup_s,
            stage_input_pack_s: profile.stage_input_pack_s,
            stage_leaf_input_pack_s: profile.stage_leaf_input_pack_s,
            stage_evaluator_call_s: profile.stage_evaluator_call_s,
            stage_backend_call_s: profile.stage_backend_call_s,
            stage_evaluator_output_gather_s: profile.stage_evaluator_output_gather_s,
            stage_evaluator_s: profile.stage_evaluator_s,
            output_assign_s: profile.output_assign_s,
            amplitude_input_pack_s: profile.amplitude_input_pack_s,
            amplitude_leaf_input_pack_s: profile.amplitude_leaf_input_pack_s,
            amplitude_evaluator_call_s: profile.amplitude_evaluator_call_s,
            amplitude_backend_call_s: profile.amplitude_backend_call_s,
            amplitude_evaluator_output_gather_s: profile.amplitude_evaluator_output_gather_s,
            amplitude_output_remap_s: profile.amplitude_output_remap_s,
            amplitude_evaluator_s: profile.amplitude_evaluator_s,
            reduction_s: profile.reduction_s,
            resolved_reduction_materialization_s: profile.resolved_reduction_materialization_s,
            total_materialization_s: profile.total_materialization_s,
            final_output_copy_s: profile.final_output_copy_s,
            total_s: profile.total_s,
            stage_input_pack_by_stage_s: profile.stage_input_pack_by_stage_s,
            stage_leaf_input_pack_by_stage_s: profile.stage_leaf_input_pack_by_stage_s,
            stage_evaluator_call_by_stage_s: profile.stage_evaluator_call_by_stage_s,
            stage_backend_call_by_stage_s: profile.stage_backend_call_by_stage_s,
            stage_evaluator_output_gather_by_stage_s: profile
                .stage_evaluator_output_gather_by_stage_s,
            stage_output_assign_by_stage_s: profile.stage_output_assign_by_stage_s,
            eager_initialize_s: profile.eager_initialize_s,
            eager_gather_s: profile.eager_gather_s,
            eager_kernel_call_s: profile.eager_kernel_call_s,
            eager_invocation_scatter_s: profile.eager_invocation_scatter_s,
            eager_finalization_s: profile.eager_finalization_s,
            eager_scatter_finalization_s: profile.eager_scatter_finalization_s,
            eager_closure_s: profile.eager_closure_s,
            eager_reduction_s: profile.eager_reduction_s,
            eager_copy_out_s: profile.eager_copy_out_s,
            recurrence_momentum_fill_s: profile.recurrence_momentum_fill_s,
            recurrence_union_source_fill_s: profile.recurrence_union_source_fill_s,
            recurrence_schedule_s: profile.recurrence_schedule_s,
            recurrence_source_kernel_s: profile.recurrence_source_kernel_s,
            recurrence_contribution_kernel_s: profile.recurrence_contribution_kernel_s,
            recurrence_finalization_s: profile.recurrence_finalization_s,
            recurrence_closure_s: profile.recurrence_closure_s,
            recurrence_replay_output_mapping_s: profile.recurrence_replay_output_mapping_s,
            recurrence_momentum_scalar_value_count: profile.recurrence_momentum_scalar_value_count,
            recurrence_schedule_execution_count: profile.recurrence_schedule_execution_count,
            recurrence_replay_schedule_execution_count: profile
                .recurrence_replay_schedule_execution_count,
            recurrence_union_schedule_execution_count: profile
                .recurrence_union_schedule_execution_count,
            recurrence_union_source_row_count: profile.recurrence_union_source_row_count,
            recurrence_replay_output_value_count: profile.recurrence_replay_output_value_count,
            recurrence_source_call_count: profile.recurrence_source_call_count,
            recurrence_source_row_count: profile.recurrence_source_row_count,
            recurrence_contribution_call_count: profile.recurrence_contribution_call_count,
            recurrence_contribution_row_count: profile.recurrence_contribution_row_count,
            recurrence_finalization_call_count: profile.recurrence_finalization_call_count,
            recurrence_finalization_row_count: profile.recurrence_finalization_row_count,
            recurrence_closure_call_count: profile.recurrence_closure_call_count,
            recurrence_closure_row_count: profile.recurrence_closure_row_count,
            selector_planner_s: 0.0,
            selector_gather_s: 0.0,
            selector_scatter_s: 0.0,
            selector_plan_kind: "none".to_string(),
            selector_group_sizes: Vec::new(),
            selector_reordered_point_count: 0,
            selector_simd_lane_width: 1,
            selector_simd_occupancy: 1.0,
            native_input_component_count: 0,
            native_input_pack_bytes: 0,
            native_input_crossing_bytes: 0,
            native_input_container_allocation_count: 0,
            state_component_count: profile.state_component_count,
            state_clear_component_count: profile.state_clear_component_count,
            source_component_count: profile.source_component_count,
            momentum_component_count: profile.momentum_component_count,
            model_parameter_component_count: profile.model_parameter_component_count,
            stage_input_copy_component_count: profile.stage_input_copy_component_count,
            stage_leaf_input_copy_component_count: profile.stage_leaf_input_copy_component_count,
            stage_evaluator_output_gather_component_count: profile
                .stage_evaluator_output_gather_component_count,
            stage_output_assign_component_count: profile.stage_output_assign_component_count,
            amplitude_input_copy_component_count: profile.amplitude_input_copy_component_count,
            amplitude_leaf_input_copy_component_count: profile
                .amplitude_leaf_input_copy_component_count,
            amplitude_evaluator_output_gather_component_count: profile
                .amplitude_evaluator_output_gather_component_count,
            amplitude_output_remap_component_count: profile.amplitude_output_remap_component_count,
            evaluator_backend_call_count: profile.evaluator_backend_call_count,
            compiled_direct_arena_engine_count: profile.compiled_direct_arena_engine_count,
            compiled_direct_arena_call_count: profile.compiled_direct_arena_call_count,
            compiled_direct_arena_minimum_effective_tile_capacity: profile
                .compiled_direct_arena_minimum_effective_tile_capacity,
            compiled_direct_arena_maximum_physical_scalar_values_per_point: profile
                .compiled_direct_arena_maximum_physical_scalar_values_per_point,
            compiled_direct_arena_maximum_hot_scalar_values_per_point: profile
                .compiled_direct_arena_maximum_hot_scalar_values_per_point,
            compiled_direct_arena_maximum_source_scalar_values_per_point: profile
                .compiled_direct_arena_maximum_source_scalar_values_per_point,
            compiled_direct_arena_maximum_reduction_scalar_values_per_point: profile
                .compiled_direct_arena_maximum_reduction_scalar_values_per_point,
            compiled_direct_arena_boundary_input_bytes: profile
                .compiled_direct_arena_boundary_input_bytes,
            compiled_direct_arena_boundary_current_output_bytes: profile
                .compiled_direct_arena_boundary_current_output_bytes,
            compiled_direct_arena_boundary_amplitude_output_bytes: profile
                .compiled_direct_arena_boundary_amplitude_output_bytes,
            reduction_input_component_count: profile.reduction_input_component_count,
            selector_gather_point_count: 0,
            selector_gather_bytes: 0,
            selector_scatter_value_count: 0,
            resolved_materialized_component_count: profile.resolved_materialized_component_count,
            total_materialized_value_count: profile.total_materialized_value_count,
            final_output_copy_value_count: profile.final_output_copy_value_count,
            observed_scratch_reallocation_count: profile.scratch_reallocation_count,
            native_output_allocation_count: 0,
        }
    }
}

impl NativeRuntimeProfile {
    fn validate_eager_top_level_accounting(&self) -> RusticolResult<()> {
        self.validate_top_level_accounting(true)
    }

    fn validate_compiled_top_level_accounting(&self) -> RusticolResult<()> {
        self.validate_top_level_accounting(false)
    }

    fn validate_recurrence_top_level_accounting(&self) -> RusticolResult<()> {
        let phases = [
            ("native input pack", self.native_input_pack_s),
            ("native input crossing", self.native_input_crossing_s),
            ("runtime orchestration", self.orchestration_s),
            ("state preparation", self.state_prepare_s),
            ("state clearing", self.state_clear_s),
            ("source preparation", self.source_fill_s),
            ("external momentum flatten", self.momentum_input_setup_s),
            ("model parameter setup", self.model_parameter_setup_s),
            (
                "recurrence momentum-form fill",
                self.recurrence_momentum_fill_s,
            ),
            (
                "recurrence union source fill",
                self.recurrence_union_source_fill_s,
            ),
            ("inclusive recurrence schedule", self.recurrence_schedule_s),
            (
                "recurrence replay output mapping",
                self.recurrence_replay_output_mapping_s,
            ),
            ("reduction", self.reduction_s),
            ("total materialization", self.total_materialization_s),
            ("final output copy", self.final_output_copy_s),
            ("selector planning", self.selector_planner_s),
            ("selector gather", self.selector_gather_s),
            ("selector scatter", self.selector_scatter_s),
        ];
        self.validate_exclusive_phases(&phases)?;

        let schedule_attribution = [
            ("source kernels", self.recurrence_source_kernel_s),
            (
                "contribution kernels",
                self.recurrence_contribution_kernel_s,
            ),
            ("current finalization", self.recurrence_finalization_s),
            ("amplitude closure", self.recurrence_closure_s),
        ];
        for (label, value) in &schedule_attribution {
            if !value.is_finite() || *value < 0.0 {
                return Err(RusticolError::internal(format!(
                    "native recurrence profile has invalid {label} time {value:.9e}s"
                )));
            }
        }
        let attributed = schedule_attribution
            .iter()
            .map(|(_, value)| value)
            .sum::<f64>();
        let tolerance = 1.0e-9_f64.max(self.recurrence_schedule_s * 1.0e-12);
        if attributed > self.recurrence_schedule_s + tolerance {
            return Err(RusticolError::internal(format!(
                "native recurrence schedule sub-attribution accounts for {attributed:.9e}s, exceeding inclusive recurrence schedule time {schedule:.9e}s",
                schedule = self.recurrence_schedule_s,
            )));
        }
        Ok(())
    }

    fn validate_exclusive_phases(&self, phases: &[(&str, f64)]) -> RusticolResult<()> {
        if !self.total_s.is_finite() || self.total_s < 0.0 {
            return Err(RusticolError::internal(format!(
                "native profile has invalid wall time {:.9e}s",
                self.total_s,
            )));
        }
        for (label, value) in phases {
            if !value.is_finite() || *value < 0.0 {
                return Err(RusticolError::internal(format!(
                    "native profile has invalid {label} time {value:.9e}s"
                )));
            }
        }
        let accounted = phases.iter().map(|(_, value)| value).sum::<f64>();
        let tolerance = 1.0e-9_f64.max(self.total_s * 1.0e-12);
        if accounted > self.total_s + tolerance {
            return Err(RusticolError::internal(format!(
                "native profile exclusive top-level phases account for {accounted:.9e}s, exceeding wall time {wall:.9e}s",
                wall = self.total_s,
            )));
        }
        Ok(())
    }

    fn validate_top_level_accounting(&self, eager: bool) -> RusticolResult<()> {
        let mut phases = vec![
            ("native input pack", self.native_input_pack_s),
            ("native input crossing", self.native_input_crossing_s),
            ("runtime orchestration", self.orchestration_s),
            ("state preparation", self.state_prepare_s),
            ("state clearing", self.state_clear_s),
            ("source fill", self.source_fill_s),
            ("momentum input setup", self.momentum_input_setup_s),
            ("model parameter setup", self.model_parameter_setup_s),
        ];
        if eager {
            phases.push(("inclusive eager execution", self.stage_evaluator_call_s));
        } else {
            phases.extend([
                ("stage input pack", self.stage_input_pack_s),
                ("stage evaluator calls", self.stage_evaluator_call_s),
                ("stage output assignment", self.output_assign_s),
                ("amplitude input pack", self.amplitude_input_pack_s),
                ("amplitude evaluator calls", self.amplitude_evaluator_call_s),
                ("reduction", self.reduction_s),
            ]);
        }
        phases.extend([
            ("total materialization", self.total_materialization_s),
            ("final output copy", self.final_output_copy_s),
            ("selector planning", self.selector_planner_s),
            ("selector gather", self.selector_gather_s),
            ("selector scatter", self.selector_scatter_s),
        ]);
        self.validate_exclusive_phases(&phases)
    }

    #[inline(never)]
    fn accumulate(&mut self, other: &Self) {
        self.native_input_pack_s += other.native_input_pack_s;
        self.native_input_crossing_s += other.native_input_crossing_s;
        self.orchestration_s += other.orchestration_s;
        self.state_prepare_s += other.state_prepare_s;
        self.state_clear_s += other.state_clear_s;
        self.source_fill_s += other.source_fill_s;
        self.momentum_input_setup_s += other.momentum_input_setup_s;
        self.momentum_setup_s += other.momentum_setup_s;
        self.model_parameter_setup_s += other.model_parameter_setup_s;
        self.stage_input_pack_s += other.stage_input_pack_s;
        self.stage_leaf_input_pack_s += other.stage_leaf_input_pack_s;
        self.stage_evaluator_call_s += other.stage_evaluator_call_s;
        self.stage_backend_call_s += other.stage_backend_call_s;
        self.stage_evaluator_output_gather_s += other.stage_evaluator_output_gather_s;
        self.stage_evaluator_s += other.stage_evaluator_s;
        self.output_assign_s += other.output_assign_s;
        self.amplitude_input_pack_s += other.amplitude_input_pack_s;
        self.amplitude_leaf_input_pack_s += other.amplitude_leaf_input_pack_s;
        self.amplitude_evaluator_call_s += other.amplitude_evaluator_call_s;
        self.amplitude_backend_call_s += other.amplitude_backend_call_s;
        self.amplitude_evaluator_output_gather_s += other.amplitude_evaluator_output_gather_s;
        self.amplitude_output_remap_s += other.amplitude_output_remap_s;
        self.amplitude_evaluator_s += other.amplitude_evaluator_s;
        self.reduction_s += other.reduction_s;
        self.resolved_reduction_materialization_s += other.resolved_reduction_materialization_s;
        self.total_materialization_s += other.total_materialization_s;
        self.final_output_copy_s += other.final_output_copy_s;
        self.total_s += other.total_s;
        accumulate_profile_stages(
            &mut self.stage_input_pack_by_stage_s,
            &other.stage_input_pack_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_leaf_input_pack_by_stage_s,
            &other.stage_leaf_input_pack_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_evaluator_call_by_stage_s,
            &other.stage_evaluator_call_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_backend_call_by_stage_s,
            &other.stage_backend_call_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_evaluator_output_gather_by_stage_s,
            &other.stage_evaluator_output_gather_by_stage_s,
        );
        accumulate_profile_stages(
            &mut self.stage_output_assign_by_stage_s,
            &other.stage_output_assign_by_stage_s,
        );
        self.eager_initialize_s += other.eager_initialize_s;
        self.eager_gather_s += other.eager_gather_s;
        self.eager_kernel_call_s += other.eager_kernel_call_s;
        self.eager_invocation_scatter_s += other.eager_invocation_scatter_s;
        self.eager_finalization_s += other.eager_finalization_s;
        self.eager_scatter_finalization_s += other.eager_scatter_finalization_s;
        self.eager_closure_s += other.eager_closure_s;
        self.eager_reduction_s += other.eager_reduction_s;
        self.eager_copy_out_s += other.eager_copy_out_s;
        self.recurrence_momentum_fill_s += other.recurrence_momentum_fill_s;
        self.recurrence_union_source_fill_s += other.recurrence_union_source_fill_s;
        self.recurrence_schedule_s += other.recurrence_schedule_s;
        self.recurrence_source_kernel_s += other.recurrence_source_kernel_s;
        self.recurrence_contribution_kernel_s += other.recurrence_contribution_kernel_s;
        self.recurrence_finalization_s += other.recurrence_finalization_s;
        self.recurrence_closure_s += other.recurrence_closure_s;
        self.recurrence_replay_output_mapping_s += other.recurrence_replay_output_mapping_s;
        self.recurrence_momentum_scalar_value_count += other.recurrence_momentum_scalar_value_count;
        self.recurrence_schedule_execution_count += other.recurrence_schedule_execution_count;
        self.recurrence_replay_schedule_execution_count +=
            other.recurrence_replay_schedule_execution_count;
        self.recurrence_union_schedule_execution_count +=
            other.recurrence_union_schedule_execution_count;
        self.recurrence_union_source_row_count += other.recurrence_union_source_row_count;
        self.recurrence_replay_output_value_count += other.recurrence_replay_output_value_count;
        self.recurrence_source_call_count += other.recurrence_source_call_count;
        self.recurrence_source_row_count += other.recurrence_source_row_count;
        self.recurrence_contribution_call_count += other.recurrence_contribution_call_count;
        self.recurrence_contribution_row_count += other.recurrence_contribution_row_count;
        self.recurrence_finalization_call_count += other.recurrence_finalization_call_count;
        self.recurrence_finalization_row_count += other.recurrence_finalization_row_count;
        self.recurrence_closure_call_count += other.recurrence_closure_call_count;
        self.recurrence_closure_row_count += other.recurrence_closure_row_count;
        self.selector_planner_s += other.selector_planner_s;
        self.selector_gather_s += other.selector_gather_s;
        self.selector_scatter_s += other.selector_scatter_s;
        self.native_input_component_count += other.native_input_component_count;
        self.native_input_pack_bytes += other.native_input_pack_bytes;
        self.native_input_crossing_bytes += other.native_input_crossing_bytes;
        self.native_input_container_allocation_count +=
            other.native_input_container_allocation_count;
        self.state_component_count += other.state_component_count;
        self.state_clear_component_count += other.state_clear_component_count;
        self.source_component_count += other.source_component_count;
        self.momentum_component_count += other.momentum_component_count;
        self.model_parameter_component_count += other.model_parameter_component_count;
        self.stage_input_copy_component_count += other.stage_input_copy_component_count;
        self.stage_leaf_input_copy_component_count += other.stage_leaf_input_copy_component_count;
        self.stage_evaluator_output_gather_component_count +=
            other.stage_evaluator_output_gather_component_count;
        self.stage_output_assign_component_count += other.stage_output_assign_component_count;
        self.amplitude_input_copy_component_count += other.amplitude_input_copy_component_count;
        self.amplitude_leaf_input_copy_component_count +=
            other.amplitude_leaf_input_copy_component_count;
        self.amplitude_evaluator_output_gather_component_count +=
            other.amplitude_evaluator_output_gather_component_count;
        self.amplitude_output_remap_component_count += other.amplitude_output_remap_component_count;
        self.evaluator_backend_call_count += other.evaluator_backend_call_count;
        self.compiled_direct_arena_engine_count += other.compiled_direct_arena_engine_count;
        self.compiled_direct_arena_call_count += other.compiled_direct_arena_call_count;
        self.compiled_direct_arena_minimum_effective_tile_capacity = match (
            self.compiled_direct_arena_minimum_effective_tile_capacity,
            other.compiled_direct_arena_minimum_effective_tile_capacity,
        ) {
            (0, value) | (value, 0) => value,
            (left, right) => left.min(right),
        };
        self.compiled_direct_arena_maximum_physical_scalar_values_per_point = self
            .compiled_direct_arena_maximum_physical_scalar_values_per_point
            .max(other.compiled_direct_arena_maximum_physical_scalar_values_per_point);
        self.compiled_direct_arena_maximum_hot_scalar_values_per_point = self
            .compiled_direct_arena_maximum_hot_scalar_values_per_point
            .max(other.compiled_direct_arena_maximum_hot_scalar_values_per_point);
        self.compiled_direct_arena_maximum_source_scalar_values_per_point = self
            .compiled_direct_arena_maximum_source_scalar_values_per_point
            .max(other.compiled_direct_arena_maximum_source_scalar_values_per_point);
        self.compiled_direct_arena_maximum_reduction_scalar_values_per_point = self
            .compiled_direct_arena_maximum_reduction_scalar_values_per_point
            .max(other.compiled_direct_arena_maximum_reduction_scalar_values_per_point);
        self.compiled_direct_arena_boundary_input_bytes +=
            other.compiled_direct_arena_boundary_input_bytes;
        self.compiled_direct_arena_boundary_current_output_bytes +=
            other.compiled_direct_arena_boundary_current_output_bytes;
        self.compiled_direct_arena_boundary_amplitude_output_bytes +=
            other.compiled_direct_arena_boundary_amplitude_output_bytes;
        self.reduction_input_component_count += other.reduction_input_component_count;
        self.selector_gather_point_count += other.selector_gather_point_count;
        self.selector_gather_bytes += other.selector_gather_bytes;
        self.selector_scatter_value_count += other.selector_scatter_value_count;
        self.resolved_materialized_component_count += other.resolved_materialized_component_count;
        self.total_materialized_value_count += other.total_materialized_value_count;
        self.final_output_copy_value_count += other.final_output_copy_value_count;
        self.observed_scratch_reallocation_count += other.observed_scratch_reallocation_count;
        self.native_output_allocation_count += other.native_output_allocation_count;
        if self.selector_plan_kind == "none" && other.selector_plan_kind != "none" {
            self.selector_plan_kind
                .clone_from(&other.selector_plan_kind);
            self.selector_group_sizes
                .clone_from(&other.selector_group_sizes);
            self.selector_reordered_point_count = other.selector_reordered_point_count;
            self.selector_simd_lane_width = other.selector_simd_lane_width;
            self.selector_simd_occupancy = other.selector_simd_occupancy;
        }
    }
}

fn accumulate_profile_stages(target: &mut Vec<f64>, source: &[f64]) {
    if target.len() < source.len() {
        target.resize(source.len(), 0.0);
    }
    for (target, source) in target.iter_mut().zip(source) {
        *target += source;
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeProfiledEvaluation {
    pub values: Vec<f64>,
    pub profile: NativeRuntimeProfile,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeResolvedEvaluation {
    /// Row-major storage with layout `[point][helicity][color]`.
    pub values: Vec<f64>,
    pub point_count: usize,
    pub helicity_ids: Vec<String>,
    pub color_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeDecimalEvaluation {
    pub values: Vec<String>,
    pub decimal_digits: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeDecimalResolvedEvaluation {
    /// Row-major storage with layout `[point][helicity][color]`.
    pub values: Vec<String>,
    pub totals: Vec<String>,
    pub point_count: usize,
    pub helicity_ids: Vec<String>,
    pub color_ids: Vec<String>,
    pub decimal_digits: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeExternalParticle {
    pub label: usize,
    pub index: usize,
    pub side: String,
    pub role: String,
    pub particle: String,
    pub outgoing_particle: String,
    pub pdg: i32,
    pub outgoing_pdg: i32,
    pub particle_class: String,
    pub momentum_slot: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeHelicityConfiguration {
    pub id: String,
    pub index: usize,
    pub helicities: Vec<i32>,
    pub representative_id: String,
    pub computed: bool,
    pub structural_zero: bool,
    pub coefficient: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeColorComponent {
    pub id: String,
    pub index: usize,
    pub kind: String,
    pub word: Vec<usize>,
    pub representative_id: String,
    pub computed: bool,
    pub coefficient: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeModelParameter {
    pub name: String,
    pub kind: String,
    pub parameter_index: usize,
    pub default: f64,
    pub default_imaginary: f64,
    pub mutable: bool,
}

impl NativeResolvedEvaluation {
    pub fn shape(&self) -> (usize, usize, usize) {
        (
            self.point_count,
            self.helicity_ids.len(),
            self.color_ids.len(),
        )
    }

    pub fn totals(&self) -> Vec<f64> {
        let component_count = self.helicity_ids.len() * self.color_ids.len();
        self.values
            .chunks(component_count)
            .map(|point| point.iter().sum())
            .collect()
    }
}

impl NativeDecimalResolvedEvaluation {
    pub fn shape(&self) -> (usize, usize, usize) {
        (
            self.point_count,
            self.helicity_ids.len(),
            self.color_ids.len(),
        )
    }
}

#[cfg(feature = "on-the-fly-test-support")]
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct NativeOnTheFlyCurrentProbeV1 {
    pub semantic_digest: String,
    pub component_count: u32,
    /// Point-major, then component-major split-complex values.
    pub values: Vec<[f64; 2]>,
}

#[cfg(feature = "on-the-fly-test-support")]
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct NativeOnTheFlyFamilyQueryProbeV1 {
    pub selected_public_flow_id: u32,
    pub public_helicities: Vec<i32>,
    pub query_digest: String,
    pub raw_amplitudes: Vec<[f64; 2]>,
    pub normalized_values: Vec<f64>,
}

#[cfg(feature = "on-the-fly-test-support")]
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct NativeOnTheFlyFamilyProbeV1 {
    pub queries: Vec<NativeOnTheFlyFamilyQueryProbeV1>,
    pub census: crate::recurrence::OnTheFlyQueryFamilyCensusV1,
    pub execution_cache_hit: bool,
    pub execution_source_calls: u32,
    pub execution_source_rows: u32,
    pub execution_contribution_calls: u32,
    pub execution_contribution_rows: u32,
    pub execution_finalization_calls: u32,
    pub execution_finalization_rows: u32,
    pub execution_closure_calls: u32,
    pub execution_closure_rows: u32,
    pub cold_prepare_seconds: f64,
    pub benchmark_warmup_repetitions: u32,
    pub benchmark_repetitions: u32,
    pub benchmark_elapsed_seconds: Option<f64>,
    pub benchmark_seconds_per_point: Option<f64>,
}

#[cfg(feature = "on-the-fly-test-support")]
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct NativeOnTheFlyArtifactProbeV1 {
    pub artifact_id: String,
    pub process_id: String,
    pub seed_digest: String,
    pub query_digest: String,
    pub trace_digest: String,
    pub point_count: u32,
    pub raw_amplitudes: Vec<[f64; 2]>,
    pub normalized_values: Vec<f64>,
    pub query_family: Option<NativeOnTheFlyFamilyProbeV1>,
    pub normalization_factor: f64,
    pub currents: Vec<NativeOnTheFlyCurrentProbeV1>,
    pub projection_enabled: bool,
    pub projection_applied: bool,
    pub projection_counts: [[u32; 3]; 2],
    pub work_census_basis: String,
    pub logical_current_count: u32,
    pub resident_current_count: u32,
    pub resident_current_component_count: u32,
    pub source_operation_count: u32,
    pub contribution_operation_count: u32,
    pub finalization_operation_count: u32,
    pub closure_operation_count: u32,
    pub total_kernel_application_count: u32,
    pub semantic_executor_binding_count: u32,
    pub distinct_prepared_executor_count: u32,
    pub trace_build_count: u32,
    pub trace_cache_hit_count: u32,
    pub momentum_fill_count: u32,
    pub benchmark_warmup_repetitions: u32,
    pub benchmark_repetitions: u32,
    pub benchmark_elapsed_seconds: Option<f64>,
    pub benchmark_seconds_per_point: Option<f64>,
    pub direct_plan_load_attempts: u32,
    pub direct_plan_decode_attempts: u32,
    pub direct_plan_materialization_attempts: u32,
    pub established_builder_attempts: u32,
}

#[cfg(feature = "on-the-fly-test-support")]
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct NativeOnTheFlyExecutionComponentV1 {
    pub dependency_depth: u32,
    pub semantic_digest: String,
    pub component: u32,
    pub on_the_fly: [f64; 2],
    pub recurrence: [f64; 2],
    pub on_the_fly_bits: [u64; 2],
    pub recurrence_bits: [u64; 2],
    pub absolute_delta: [f64; 2],
}

#[cfg(feature = "on-the-fly-test-support")]
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct NativeOnTheFlyExecutionDiagnosticV1 {
    pub on_the_fly_artifact_id: String,
    pub on_the_fly_process_id: String,
    pub recurrence_artifact_id: String,
    pub recurrence_process_id: String,
    pub seed_digest: String,
    pub query_digest: String,
    pub trace_digest: String,
    pub direct_plan_semantic_digest: String,
    pub direct_runtime_layout_digest: String,
    pub public_flow_id: u32,
    pub representative_flow_id: u32,
    pub public_helicities: Vec<i32>,
    pub public_direct_helicity_id: u32,
    pub representative_direct_helicity_id: u32,
    pub replay_phase: [f64; 2],
    pub replay_multiplicity: u32,
    pub replay_scale: [f64; 2],
    pub on_the_fly_public_amplitude: [f64; 2],
    pub recurrence_representative_amplitude: [f64; 2],
    pub recurrence_public_amplitude: [f64; 2],
    pub amplitude_absolute_delta: [f64; 2],
    pub current_components: Vec<NativeOnTheFlyExecutionComponentV1>,
    pub first_raw_bit_difference: Option<usize>,
    pub raw_bit_difference_count: u32,
    pub compared_current_count: u32,
    pub excluded_direct_current_count: u32,
}

/// Python-independent schema-v3 process runtime.
///
/// The input momentum layout is `[point][external particle][E, px, py, pz]`.
/// Instances are mutable and must not be called concurrently; independent
/// instances can be used from separate threads.
pub struct NativeRuntime {
    root: PathBuf,
    artifact_id: String,
    runtime: ExecutionRuntime,
    execution_lane: NativeExecutionLane,
    process: String,
    process_key: String,
    representative_process_id: String,
    external_permutation: Vec<usize>,
    input_crossing_map: Option<Vec<InputCrossingMapEntry>>,
    permutation_alias_of: Option<String>,
    final_state_permutation_alias_of: Option<String>,
    // Absent for stored processes, aliases, and permutations of either.
    process_group: Option<Box<ProcessGroupRouting>>,
    physics_v1: native_runtime::LazyProcessPhysicsV1,
    warnings_muted: bool,
    warned_kinds: BTreeSet<String>,
    pending_warnings: Vec<String>,
    point_selector_scratch: PointSelectorExecutionScratch,
    selector_simd_lane_width: usize,
    // Cold, opt-in metadata and vector defaults; ordinary evaluation never reads it.
    correlated: Option<correlated::CorrelatedRuntime>,
}

enum NativeExecutionLane {
    Compiled,
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    Eager(Box<EagerNativeRuntime>),
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    Recurrence(Box<RecurrenceNativeRuntime>),
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    OnTheFly(Box<native_runtime::OnTheFlyExecutionRuntime>),
}

impl NativeExecutionLane {
    #[cfg(feature = "symbolica-runtime")]
    const fn is_eager(&self) -> bool {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            matches!(self, Self::Eager(_))
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            false
        }
    }

    #[cfg(feature = "symbolica-runtime")]
    const fn is_recurrence(&self) -> bool {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            matches!(self, Self::Recurrence(_))
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            false
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RuntimeParameterSlots {
    real: usize,
    imaginary: Option<usize>,
}

struct ModelParameterEvaluatorRuntime {
    input_parameter_indices: Vec<usize>,
    outputs: Vec<GenericDerivedParameterOutputManifest>,
    evaluator: EvaluatorGroup,
}

struct StageRuntime {
    outputs: Vec<(usize, usize)>,
    output_spans: Vec<(usize, usize, usize)>,
    chunk_outputs: Vec<Vec<(usize, usize)>>,
    chunk_output_spans: Vec<Vec<(usize, usize, usize)>>,
    input_components: Option<Vec<usize>>,
    input_spans: Vec<(usize, usize, usize)>,
    parameter_scratch_f64: Vec<Complex<f64>>,
    output_scratch_f64: Vec<Complex<f64>>,
    evaluator: EvaluatorGroup,
}

#[derive(Default)]
struct RoutedReductionScratch {
    helicity_indices: Vec<usize>,
    color_indices: Vec<usize>,
    helicity_positions: Vec<Option<usize>>,
    color_positions: Vec<Option<usize>>,
    selected_member_weights: Vec<(usize, usize, f64)>,
    selected_member_weight_ranges: Vec<std::ops::Range<usize>>,
    direct_group_re: Vec<f64>,
    direct_group_im: Vec<f64>,
    direct_source_components: Vec<f64>,
    direct_target_components: Vec<f64>,
    direct_totals: Vec<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MaterializedHelicityDirectTotalPlanKey {
    physics_binding_id: u64,
    helicity_index: usize,
    root_factor_bits: Vec<(usize, [u64; 2])>,
    color_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct MaterializedHelicityDirectTotalRoot {
    output_index: usize,
    factor: Complex<f64>,
}

#[derive(Clone, Debug)]
struct MaterializedHelicityDirectTotalGroup {
    root_range: std::ops::Range<usize>,
    all_sector_weight: f64,
    identity_output_index: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct MaterializedHelicityDirectTotalColorGroup {
    group_index: usize,
    weight: f64,
}

#[derive(Clone, Copy, Debug)]
struct MaterializedHelicityDirectTotalContractionEntry {
    left_group_index: usize,
    right_group_index: usize,
    weight_re: f64,
    weight_im: f64,
    symmetry_factor: f64,
}

#[derive(Clone, Copy, Debug)]
struct MaterializedHelicityDirectDefaultPlanCacheEntry {
    physics_binding_id: u64,
    helicity_index: usize,
    plan_index: usize,
}

#[derive(Clone, Debug)]
enum MaterializedHelicityDirectTotalReduction {
    Lc {
        color_group_ranges: Vec<std::ops::Range<usize>>,
        color_groups: Vec<MaterializedHelicityDirectTotalColorGroup>,
    },
    Contracted {
        entries: Vec<MaterializedHelicityDirectTotalContractionEntry>,
    },
}

#[derive(Clone, Debug)]
struct MaterializedHelicityDirectTotalPlan {
    key: MaterializedHelicityDirectTotalPlanKey,
    roots: Vec<MaterializedHelicityDirectTotalRoot>,
    groups: Vec<MaterializedHelicityDirectTotalGroup>,
    reduction: MaterializedHelicityDirectTotalReduction,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CompiledDirectRoutedReductionFootprint {
    maximum_source_component_count: usize,
    maximum_target_component_count: usize,
}

impl CompiledDirectRoutedReductionFootprint {
    fn merge(self, other: Self) -> Self {
        Self {
            maximum_source_component_count: self
                .maximum_source_component_count
                .max(other.maximum_source_component_count),
            maximum_target_component_count: self
                .maximum_target_component_count
                .max(other.maximum_target_component_count),
        }
    }

    const fn is_empty(self) -> bool {
        self.maximum_source_component_count == 0 && self.maximum_target_component_count == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompiledDirectReducerKind {
    Plain,
    Coherent,
    Contracted { group_count: usize },
    SymmetricGroup { workspace_scalars_per_lane: usize },
    ColorTopologyReplay { physical_group_count: usize },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum CompiledDirectReductionFootprint {
    /// The maximum reducer working set was derived entirely from authenticated
    /// amplitude, physics, and replay metadata.
    Authenticated { hot_scalar_values_per_point: usize },
    /// A future/legacy execution shape did not expose enough static metadata.
    /// Production Direct tiling must fail closed to one point.
    #[default]
    Unauthenticated,
}

impl CompiledDirectReductionFootprint {
    fn authenticated(
        kind: CompiledDirectReducerKind,
        maximum_group_component_count: usize,
        maximum_resolved_component_count: usize,
        routed: CompiledDirectRoutedReductionFootprint,
    ) -> RusticolResult<Self> {
        let maximum_group_plane_scalars = maximum_group_component_count
            .max(1)
            .checked_mul(2)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "compiled Direct-Arena reduction group footprint overflows",
                )
            })?;
        let (amplitude_input_scalars, reducer_workspace_scalars, supports_resolved) = match kind {
            CompiledDirectReducerKind::Plain => (2, 0, false),
            CompiledDirectReducerKind::Coherent => (maximum_group_plane_scalars, 2, true),
            CompiledDirectReducerKind::Contracted { group_count } => (
                maximum_group_plane_scalars,
                group_count.checked_mul(2).ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena contracted reduction footprint overflows",
                    )
                })?,
                true,
            ),
            CompiledDirectReducerKind::SymmetricGroup {
                workspace_scalars_per_lane,
            } => (
                maximum_group_plane_scalars,
                workspace_scalars_per_lane,
                false,
            ),
            CompiledDirectReducerKind::ColorTopologyReplay {
                physical_group_count,
            } => (
                maximum_group_plane_scalars,
                physical_group_count.checked_mul(2).ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena topology-replay footprint overflows",
                    )
                })?,
                true,
            ),
        };
        // One real total plane is live in every reducer. Reducer storage is a
        // cache-local working-set estimate only; none of it belongs to the
        // Direct workspace batching target.
        let total_phase = amplitude_input_scalars
            .checked_add(reducer_workspace_scalars)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                RusticolError::integrity(
                    "compiled Direct-Arena total-reduction footprint overflows",
                )
            })?;
        let resolved_phase = if supports_resolved {
            total_phase
                .checked_add(maximum_resolved_component_count)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena resolved-reduction footprint overflows",
                    )
                })?
        } else {
            total_phase
        };
        let routed_phase = if routed.is_empty() {
            0
        } else {
            // Routed reducers materialize a split-complex group, one real
            // totals plane, and authenticated source/target component planes.
            // The target may be owned by an outer public-batch replay, but its
            // tile slice is still part of the phase-local cache working set.
            amplitude_input_scalars
                .checked_add(2)
                .and_then(|value| value.checked_add(1))
                .and_then(|value| value.checked_add(routed.maximum_source_component_count))
                .and_then(|value| value.checked_add(routed.maximum_target_component_count))
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena routed-reduction footprint overflows",
                    )
                })?
        };
        Ok(Self::Authenticated {
            hot_scalar_values_per_point: total_phase.max(resolved_phase).max(routed_phase),
        })
    }

    const fn hot_scalar_values_per_point(self) -> Option<usize> {
        match self {
            Self::Authenticated {
                hot_scalar_values_per_point,
            } => Some(hot_scalar_values_per_point),
            Self::Unauthenticated => None,
        }
    }
}

struct AmplitudeRuntime {
    output_length: usize,
    raw_sum_weights: Vec<f64>,
    raw_sum_all_sector_weights: Vec<f64>,
    raw_sum_color_sector_ids: Vec<Option<i64>>,
    raw_sum_groups: Vec<RawSumGroup>,
    has_coherent_groups: bool,
    color_contraction: Option<ColorContractionRuntime>,
    color_topology_replay: Option<ColorTopologyReplayAmplitudeRuntime>,
    input_components: Option<Vec<usize>>,
    input_spans: Vec<(usize, usize, usize)>,
    parameter_scratch_f64: Vec<Complex<f64>>,
    evaluator_output_scratch_f64: Vec<Complex<f64>>,
    output_scratch_f64: Vec<Complex<f64>>,
    resolved_source_row_scratch_f64: Vec<f64>,
    resolved_target_row_scratch_f64: Vec<f64>,
    routed_reduction_scratch: RoutedReductionScratch,
    materialized_helicity_direct_total_plans: Vec<MaterializedHelicityDirectTotalPlan>,
    materialized_helicity_direct_total_plan_capacity: usize,
    materialized_helicity_direct_total_next_replacement: usize,
    materialized_helicity_direct_default_plan_by_selector_domain:
        Vec<Option<MaterializedHelicityDirectDefaultPlanCacheEntry>>,
    evaluator_output_order: Option<Vec<usize>>,
    evaluator: Option<EvaluatorGroup>,
}

struct ColorTopologyReplayAmplitudeRuntime {
    mappings: Vec<ColorTopologyReplayAmplitudeMapping>,
    physical_groups: Vec<RawSumGroup>,
    physical_group_helicities: Vec<Vec<i32>>,
    color_contraction: Option<ColorContractionRuntime>,
    unit_weights: Vec<f64>,
    no_sector_ids: Vec<Option<i64>>,
    physical_group_scratch_f64: Vec<Complex<f64>>,
    covered_groups: Vec<bool>,
}

struct ColorTopologyReplayAmplitudeMapping {
    label_permutation: Vec<(usize, usize)>,
    group_routes: Vec<ColorTopologyReplayAmplitudeGroupRoute>,
}

struct ColorTopologyReplayAmplitudeGroupRoute {
    source_group_index: usize,
    target_group_index: usize,
    factor: Complex<f64>,
}

mod runtime_load;
use runtime_load::*;

mod model_parameters;
use model_parameters::*;

mod evaluation;
use evaluation::resolved_f64_totals;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod compiled_direct_prototype;
mod helicity_lane;
use helicity_lane::*;
mod momentum;
use momentum::*;
mod sources;

mod validation;
use validation::*;

mod correlated;
mod native_runtime;
pub use correlated::{
    NativeCorrelatedEvaluation, NativeCorrelatedRequest, NativeSpinCorrelationVectors,
};

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod on_the_fly_warm_up;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub use on_the_fly_warm_up::{
    NativeOnTheFlyWarmUpEvent, NativeOnTheFlyWarmUpEventKind, NativeOnTheFlyWarmUpObserver,
    NativeOnTheFlyWarmUpResult, NativeOnTheFlyWarmUpStage,
};

mod artifact_load;
use artifact_load::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_backend;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use eager_backend::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod recurrence_backend;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod recurrence_lane;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use recurrence_lane::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod recurrence_load;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use recurrence_load::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub(crate) mod recurrence_manifest;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use recurrence_manifest::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod recurrence_bootstrap;
#[cfg(feature = "python-generation-bridge")]
pub(crate) use recurrence_bootstrap::build_recurrence_bootstrap_image_v1;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub(crate) use recurrence_bootstrap::{
    RecurrenceReadyExecutionV1, decode_recurrence_bootstrap_image_v1,
};

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod recurrence_bootstrap_physics;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use recurrence_bootstrap_physics::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod recurrence_process_pack;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod on_the_fly_lane;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod on_the_fly_load;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod on_the_fly_manifest;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod on_the_fly_public_metadata;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod on_the_fly_selectors;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[allow(dead_code)]
mod eager_manifest;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use eager_manifest::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_v3_common;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_v3_decode;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_v3_load;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_v3_manifest;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[allow(dead_code)]
mod eager_load;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use eager_load::*;

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_lane;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
use eager_lane::*;

mod physics;

mod point_selectors;
use point_selectors::*;

#[path = "evaluator.rs"]
mod evaluator;
#[cfg(all(
    test,
    feature = "f64-compiled",
    not(feature = "f64-symjit"),
    any(target_os = "linux", target_os = "macos")
))]
pub(crate) use evaluator::native_direct::tests::count_allocations;
pub(crate) use evaluator::recurrence_closure_direct::execute_closure_reduce_rows;
pub(crate) use evaluator::recurrence_intrinsic_direct::execute_interaction_terminal_closures;
#[cfg(all(test, feature = "f64-symjit"))]
pub(crate) use evaluator::symjit_direct::tests::count_allocations;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub(crate) use evaluator::symjit_eager_direct;
#[cfg(any(test, feature = "python-generation-bridge"))]
pub(crate) use evaluator::symjit_plane::compile_symbolica_program_to_plane_application_bytes;
use evaluator::*;

#[path = "wavefunctions.rs"]
mod wavefunctions;
use wavefunctions::*;

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "engine/source_metadata_tests.rs"]
mod source_metadata_tests;

#[cfg(test)]
#[path = "engine/quantum_number_flow_tests.rs"]
mod quantum_number_flow_tests;

#[cfg(test)]
#[path = "engine/contraction_metadata_tests.rs"]
mod contraction_metadata_tests;

#[cfg(all(test, any(feature = "f64-compiled", feature = "f64-symjit")))]
#[path = "engine/eager_integration_tests.rs"]
mod eager_integration_tests;

#[cfg(all(test, any(feature = "f64-compiled", feature = "f64-symjit")))]
#[path = "engine/recurrence_integration_tests.rs"]
mod recurrence_integration_tests;

#[cfg(all(test, any(feature = "f64-compiled", feature = "f64-symjit")))]
#[path = "engine/eager_v3_manifest_tests.rs"]
mod eager_v3_manifest_tests;
