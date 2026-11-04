// SPDX-License-Identifier: 0BSD

//! Python-independent Rusticol runtime core.

#[cfg(not(any(
    feature = "f64-compiled",
    feature = "f64-symjit",
    feature = "symbolica-runtime"
)))]
compile_error!("rusticol-core requires at least one evaluator runtime feature");

mod artifact;
pub mod direct_arena;
pub mod eager_layout;
mod eager_lowering_v3;
mod eager_plan_v3_pacbin;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod eager_runtime;
mod eager_tables;
mod engine;
mod error;
mod metadata;
pub mod pacbin;
pub mod recurrence;

pub use artifact::{
    ArtifactKind, ArtifactManifest, ArtifactProcess, ArtifactSelection, Payload, PayloadRole,
    ProcessAlias, ProcessGroupRelation, Target, VerifiedArtifact, runtime_target_info,
};
pub use eager_lowering_v3::*;
pub use eager_plan_v3_pacbin::*;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub use eager_runtime::{
    DEFAULT_EAGER_POINT_TILE_SIZE, DEFAULT_EAGER_WORKSPACE_MIB,
    EAGER_HOMOGENEOUS_LINEAR_CURRENT_PROOF, EAGER_INDEPENDENT_BLOCK_SIZE, EagerComplex64,
    EagerDirectClosureSpec, EagerExecutionPlan, EagerExecutionRuntime, EagerKernelBackend,
    EagerKernelCall, EagerKernelInput, EagerKernelRole, EagerKernelSpec, EagerPlanDefinition,
    EagerPlanDimensions, EagerPlanPayloads, EagerReductionEntry, EagerReductionGroup,
    EagerRuntimeOptions, EagerSelectorPayloads, EagerSelectorStagePayload, EagerStagePayload,
};
pub use eager_tables::{
    EAGER_KERNEL_ABI, EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
    EAGER_OUTPUT_FACTOR_COUPLING_IMAG, EAGER_OUTPUT_FACTOR_COUPLING_REAL, EAGER_OUTPUT_FACTOR_NONE,
    EAGER_PLAN_ABI, EAGER_RUNTIME_CAPABILITY, EAGER_SELECTOR_DOMAINS_ABI, EagerAttachmentRow,
    EagerClosureRow, EagerCouplingRow, EagerFinalizationRow, EagerInvocationRow,
    EagerSelectorDomainIdRow, EagerSelectorDomainRow, EagerSelectorGroupRow, MISSING_U32,
};
#[cfg(feature = "f64-symjit")]
pub use engine::eager_direct_descriptor_for_source_application_bytes;
pub use engine::{
    NativeColorComponent, NativeCorrelatedEvaluation, NativeCorrelatedRequest,
    NativeDecimalEvaluation, NativeDecimalResolvedEvaluation, NativeEagerExactAttachment,
    NativeEagerExactClosure, NativeEagerExactCoupling, NativeEagerExactFinalization,
    NativeEagerExactInvocation, NativeEagerExactSections, NativeEagerExactStage,
    NativeExternalParticle, NativeHelicityConfiguration, NativeModelParameter,
    NativeProfiledEvaluation, NativeRationalFactor, NativeRecurrenceExactExecutor,
    NativeRecurrenceExactFactor, NativeRecurrenceExactSections, NativeRecurrenceSelectorPlan,
    NativeResolvedEvaluation, NativeRuntime, NativeRuntimeMetadata, NativeRuntimeProfile,
    NativeSpinCorrelationVectors, RuntimeCapability, preflight_prepared_kernel_pack,
    supported_runtime_capabilities,
};
#[cfg(feature = "on-the-fly-test-support")]
pub use engine::{
    NativeOnTheFlyArtifactProbeV1, NativeOnTheFlyCurrentProbeV1,
    NativeOnTheFlyExecutionComponentV1, NativeOnTheFlyExecutionDiagnosticV1,
    NativeOnTheFlyFamilyProbeV1, NativeOnTheFlyFamilyQueryProbeV1,
};
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub use engine::{
    NativeOnTheFlyWarmUpEvent, NativeOnTheFlyWarmUpEventKind, NativeOnTheFlyWarmUpObserver,
    NativeOnTheFlyWarmUpResult, NativeOnTheFlyWarmUpStage,
};
pub use error::{RusticolError, RusticolErrorKind, RusticolResult};
pub use metadata::{
    ColorAccuracy, ColorComponent, ContractedColor, Coverage, ExternalParticle, Helicity,
    LcColorFlow, ModelParameter, ParameterKind, ParticleRole, ProcessPhysics, Reduction,
    ReductionGroup, ReductionKind, SelectorCapabilities,
};

pub const PYTHON_API_VERSION: u32 = 1;
pub const TOML_SCHEMA_VERSION: u32 = 1;
pub const COMPILED_MODEL_SCHEMA_VERSION: u32 = 9;
pub const PROCESS_ARTIFACT_SCHEMA_VERSION: u32 = 3;
pub const RUNTIME_PHYSICS_SCHEMA_VERSION: u32 = 1;
pub const C_ABI_VERSION: u32 = 1;
pub const SYMBOLICA_SERIALIZATION_ABI: &str = "symbolica-bincode2-v1";

/// Unstable implementation details shared with pyAmpliCol's private PyO3
/// extension. This module is absent from ordinary/default rusticol-core builds.
#[cfg(feature = "python-generation-bridge")]
#[doc(hidden)]
pub mod __private {
    use crate::RusticolResult;
    use crate::recurrence::template::{
        OwnedRecurrenceTemplateInput, ValidatedRecurrenceTemplateInput,
    };
    use crate::recurrence::{PreparedDirectExecutorCatalog, SemanticDigest};

    /// Unstable cold-path bridge for the private PyO3 artifact generator.
    #[doc(hidden)]
    pub fn compile_symbolica_program_to_plane_application_bytes(
        program_repr: &str,
        input_complex_count: usize,
        output_complex_count: usize,
        optimization_level: u8,
        compress: bool,
    ) -> RusticolResult<Vec<u8>> {
        crate::engine::compile_symbolica_program_to_plane_application_bytes(
            program_repr,
            input_complex_count,
            output_complex_count,
            optimization_level,
            compress,
        )
    }

    /// Build one authenticated compact recurrence cold-load image from the
    /// already staged, ordinary JSON payloads and typed payload inventory.
    #[doc(hidden)]
    pub fn build_recurrence_bootstrap_image_v1(
        context_json: &[u8],
        execution_json: &[u8],
        physics_json: &[u8],
    ) -> RusticolResult<Vec<u8>> {
        crate::engine::build_recurrence_bootstrap_image_v1(
            context_json,
            execution_json,
            physics_json,
        )
    }

    /// Unstable cold-path bridge for constructing the compact private
    /// on-the-fly process seed from source-only projection JSON.
    #[doc(hidden)]
    pub fn project_recurrence_template_catalog_json_v1(
        value: &serde_json::Value,
    ) -> RusticolResult<OwnedRecurrenceTemplateInput> {
        crate::recurrence::template_json::project_recurrence_template_catalog_json_v1(value)
    }

    /// Unstable cold-path bridge for encoding ordered validated compact seeds.
    #[doc(hidden)]
    pub fn build_on_the_fly_process_seed_bytes_batch_v1(
        ordered_source_projection_jsons: &[Vec<u8>],
        templates: &ValidatedRecurrenceTemplateInput,
        direct_catalog: &PreparedDirectExecutorCatalog,
        prepared_pack_digest: SemanticDigest,
    ) -> RusticolResult<Vec<Vec<u8>>> {
        let mut projections = Vec::new();
        projections
            .try_reserve_exact(ordered_source_projection_jsons.len())
            .map_err(|error| {
                crate::RusticolError::invalid_argument(format!(
                    "process-projection allocation failed: {error}"
                ))
            })?;
        for (index, source_projection_json) in ordered_source_projection_jsons.iter().enumerate() {
            let projection =
                crate::recurrence::on_the_fly::parse_on_the_fly_process_seed_projection_v1(
                    source_projection_json,
                )
                .map_err(|error| {
                    crate::RusticolError::with_kind(
                        error.kind(),
                        format!(
                            "on-the-fly process seed projection at index {index} failed: {error}"
                        ),
                    )
                })?;
            projections.push(projection);
        }
        let seeds = crate::recurrence::on_the_fly::build_on_the_fly_process_seeds_v1(
            projections,
            templates,
            direct_catalog,
            prepared_pack_digest,
        )?;
        let mut encoded = Vec::new();
        encoded.try_reserve_exact(seeds.len()).map_err(|error| {
            crate::RusticolError::invalid_argument(format!(
                "encoded process-seed allocation failed: {error}"
            ))
        })?;
        for (index, seed) in seeds.iter().enumerate() {
            encoded.push(
                crate::recurrence::on_the_fly::seed_codec::encode_on_the_fly_process_seed_v1(seed)
                    .map_err(|error| {
                        crate::RusticolError::with_kind(
                            error.kind(),
                            format!(
                                "on-the-fly process seed encoding at index {index} failed: {error}"
                            ),
                        )
                    })?,
            );
        }
        Ok(encoded)
    }

    /// Unstable singleton wrapper retained for the private PyO3 generator.
    #[doc(hidden)]
    pub fn build_on_the_fly_process_seed_bytes_v1(
        source_projection_json: &[u8],
        templates: &ValidatedRecurrenceTemplateInput,
        direct_catalog: &PreparedDirectExecutorCatalog,
        prepared_pack_digest: SemanticDigest,
    ) -> RusticolResult<Vec<u8>> {
        let payloads = build_on_the_fly_process_seed_bytes_batch_v1(
            &[source_projection_json.to_vec()],
            templates,
            direct_catalog,
            prepared_pack_digest,
        )?;
        let [payload] = <[Vec<u8>; 1]>::try_from(payloads).map_err(|payloads| {
            crate::RusticolError::internal(format!(
                "singleton process-seed byte batch returned {} payloads",
                payloads.len()
            ))
        })?;
        Ok(payload)
    }

    /// Unstable cold-path bridge for inspecting one authoritative native
    /// process seed without exposing the internal seed representation.
    #[doc(hidden)]
    pub fn inspect_on_the_fly_process_seed_identity_json_v1(
        payload: &[u8],
    ) -> RusticolResult<String> {
        let seed = crate::recurrence::on_the_fly::decode_on_the_fly_process_seed_v1(payload)?;
        serde_json::to_string(&seed.identity()).map_err(|error| {
            crate::RusticolError::serialization(format!(
                "could not serialize on-the-fly process-seed identity: {error}"
            ))
        })
    }
}
pub const ARTIFACT_MANIFEST_FILE: &str = "artifact.json";
