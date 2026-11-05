// SPDX-License-Identifier: 0BSD

//! Model-generic compact recurrence construction and execution.

mod arena;
mod builder;
mod color;
mod color_contraction;
mod construct;
mod contact_orbit_owner;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
mod diagnostic;
pub mod direct_backend;
mod direct_codec;
mod direct_helicity_dispatch;
mod direct_lowering;
mod direct_pacbin;
mod direct_plan;
pub mod direct_runtime;
mod exact;
mod fermion_ordering;
mod helicity_support;
mod input;
mod layout;
pub(crate) mod on_the_fly;
pub mod process;
mod program;
mod relation;
mod symmetric_group_fft;
pub mod template;
pub(crate) mod template_json;

pub use arena::{
    DirectArenaAssignment, DirectArenaInterval, DirectArenaLayout, assign_direct_arena,
    recurrence_direct_arena_layout,
};
pub use builder::AuthenticatedRecurrenceBuilderInput;
pub use color::{
    DynamicLCColorState, DynamicLCColorStateInterner, LCColorComponent, LCColorComponentKind,
    LCColorComponentOperation, LCColorComponentRole, LCColorEndpoint, LCColorParentPort,
    LCColorPortBinding, LCColorPortWiring, LCColorSourceSeed, LCColorSourceSeedOperation,
    LCColorTransitionWitness,
};
pub(crate) use color_contraction::RuntimeSymmetricGroupColorWorkspace;
pub use color_contraction::{
    CanonicalColorContractionEntries, CanonicalColorContractionEntry, FactorizedColorContraction,
    FactorizedColorContractionKind, RECURRENCE_COLOR_CONTRACTION_CODEC_ABI,
    RawColorContractionEntry, RecurrenceColorAccuracy, RecurrenceColorContraction,
    RecurrenceColorStorage, RuntimeColorContractionEntries, RuntimeColorContractionEntry,
    RuntimeColorContractionReducer, RuntimeFactorizedColorContraction,
    RuntimeFactorizedColorContractionEntry, RuntimeSymmetricGroupColorContraction,
    RuntimeSymmetricGroupKernel, decode_recurrence_color_contraction_v3,
    recurrence_color_contraction_digest,
};
pub use construct::RecurrenceBuildProgress;
#[doc(hidden)]
pub use construct::RecurrenceGenerationTelemetry;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[doc(hidden)]
pub use diagnostic::ConstructionTransitionDiagnosticRowV1;
pub use direct_codec::{decode_recurrence_direct_plan_v2, encode_recurrence_direct_plan_v2};
pub use direct_helicity_dispatch::{
    DirectHelicityDispatch, DirectHelicityDispatchDescriptor, DirectHelicityDispatchParts,
    DirectHelicityRowGroupDescriptor, DirectHelicitySupportDomainDescriptor,
    RECURRENCE_HELICITY_DISPATCH_ABI, RecurrenceHelicityDispatchMetadata,
    decode_recurrence_helicity_dispatch_v1, encode_recurrence_helicity_dispatch_v1,
    write_recurrence_helicity_dispatch_v1_atomic,
};
#[cfg(test)]
pub(crate) use direct_lowering::validated_template_fixture;
pub use direct_lowering::{
    DirectRecurrenceRuntimeOptions, PreparedDirectExecutorBinding, PreparedDirectExecutorCatalog,
    PreparedDirectExecutorKey, lower_recurrence_direct_plan_v2,
    lower_recurrence_direct_plan_v2_with_helicity_dispatch,
    lower_recurrence_direct_plan_v2_with_relation_discovery, lower_recurrence_direct_v2,
};
pub(crate) use direct_pacbin::validate_recurrence_color_projection_certificate;
pub use direct_pacbin::{
    RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER, RECURRENCE_DIRECT_SCHEDULE_MEMBER,
    RecurrenceDirectPacbinMetadata, load_recurrence_direct_plan_pacbin,
    write_recurrence_direct_plan_pacbin,
    write_recurrence_direct_plan_pacbin_with_projection_certificate,
};
pub(crate) use direct_plan::DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
#[cfg(test)]
pub(crate) use direct_plan::tests::{
    valid_parts as valid_direct_plan_parts_fixture, valid_plan as valid_direct_plan_fixture,
};
pub use direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_NONE_U32,
    DirectAmplitudeDestinationDescriptor, DirectClosureRow, DirectContributionRow,
    DirectCurrentDescriptor, DirectDestinationOperation, DirectExecutorRole, DirectFinalizationRow,
    DirectMomentumFormDescriptor, DirectMomentumTerm, DirectNodeKind, DirectRecurrencePlan,
    DirectRecurrencePlanParts, DirectReplayTargetDescriptor, DirectResolvedHelicityDescriptor,
    DirectResolvedSourceSelection, DirectRowGroupDescriptor, DirectSelectorDomainDescriptor,
    DirectSelectorWorkSummary, DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow,
    DirectSourceProjectionRow, DirectSourceRow, DirectSourceStateAssignment,
    RECURRENCE_DIRECT_PLAN_ABI, RECURRENCE_DIRECT_RUNTIME_CAPABILITY,
    RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI, RECURRENCE_DIRECT_TEMPLATE_ABI,
};
pub use exact::{ExactComplexRational, ExactRational};
pub use helicity_support::{
    AllFlowHelicityMask, AllFlowHelicitySupportProjection, project_all_flow_helicity_support,
};
pub use input::{
    CanonicalInputSection, CheckedTableRange, MultiwordMaskCatalogView,
    RecurrenceBuilderInputHeader, canonical_input_digest, checked_u32_len, checked_u64_len,
    checked_usize, validate_equal_column_lengths, validate_header_and_sections,
    validate_packed_ranges, validate_ranges_within, validate_u32_references,
};
pub use layout::{
    CanonicalMomentumLinearForm, ContributionKey, CurrentCoreKey, CurrentHelicityIdentity,
    CurrentSourceBinding, DynamicLCColorStateId, LCColorWitnessTermId, MomentumTerm,
    RecurrenceNodeKind, RecurrenceStrategy, SemanticDigest, SourceStateAssignment,
};
#[cfg(feature = "on-the-fly-test-support")]
#[doc(hidden)]
pub use on_the_fly::{
    OnTheFlyQueryFamilyCensusV1, OnTheFlyTestSupportReportV1, on_the_fly_query_family_census_v1,
    on_the_fly_test_support_probe_v1,
};
pub use program::{
    CLOSURE_TARGET_DOMAIN_PROOF_KIND, ClosureCandidateDomainCertificateV1,
    ClosureExecutionProofGroupV2, ClosureProofContributionV2, ClosureProofMetadataV2,
    ClosureTargetDomainCertificateV1, RecurrenceAmplitudeDestination, RecurrenceClosureTerm,
    RecurrenceContribution, RecurrenceCurrent, RecurrenceFinalization, RecurrenceProgram,
    RecurrenceReplayTarget, RecurrenceResolvedHelicity, ReflectionCertificateV1,
    ThreeLineTraversalCertificateV1, ThreeLineTraversalKindV1, closure_component_factor_digest_v2,
    closure_proof_semantic_completeness_digest_v2,
    closure_proof_semantic_completeness_digest_with_three_line_v2,
    closure_selector_domain_digest_v2, three_line_traversal_proof_digest_v1,
};
pub use relation::{
    NUMERICAL_RELATION_CERTIFICATE_ALGORITHM, RecurrenceCurrentRelationCertificate,
    RecurrenceNumericalCurrentMapping, RecurrenceNumericalRelationEvidence,
    RecurrenceRelationDiscoveryMode, RecurrenceRelationDiscoveryOptions,
    RecurrenceRelationDiscoveryReport, authenticate_recurrence_numerical_relation_provenance,
    recurrence_numerical_source_semantics_sha256, relation_certificate_algorithm,
};
pub use symmetric_group_fft::{
    MAX_SYMMETRIC_GROUP_FFT_DEGREE, SymmetricGroupComplex64, SymmetricGroupFftBlock,
    SymmetricGroupFftPlan, SymmetricGroupFftWorkspace,
};
/// Semantic prepared-model companion ABI.
pub const RECURRENCE_TEMPLATE_ABI: &str = "pyamplicol-recurrence-template-v1";
/// Python-to-Rust recurrence builder input ABI.
pub const RECURRENCE_BUILDER_INPUT_ABI: &str = "pyamplicol-recurrence-builder-input-v2";
/// Bounded Rust-to-Python recurrence builder result ABI.
pub const RECURRENCE_BUILDER_RESULT_ABI: &str = "pyamplicol-recurrence-builder-result-v2";
/// Direct-arena recurrence plan ABI.
pub const RECURRENCE_PLAN_ABI: &str = RECURRENCE_DIRECT_PLAN_ABI;
/// Direct-arena native recurrence layout ABI.
pub const RECURRENCE_RUNTIME_LAYOUT_ABI: &str = RECURRENCE_DIRECT_RUNTIME_LAYOUT_ABI;
/// Process runtime kind stored in execution metadata.
pub const RECURRENCE_RUNTIME_KIND: &str = "pyamplicol-runtime-recurrence-execution";
/// Native complex-f64 recurrence capability.
pub const RECURRENCE_RUNTIME_CAPABILITY: &str = RECURRENCE_DIRECT_RUNTIME_CAPABILITY;
/// LC color capability required by recurrence.
pub const RECURRENCE_LC_COLOR_CAPABILITY: &str = "rusticol.recurrence-color.lc.v1";
/// Contracted NLC/full-color capability required by recurrence.
pub const RECURRENCE_CONTRACTED_COLOR_CAPABILITY: &str = "rusticol.recurrence-color.contracted.v1";
/// The builder input always consists of explicitly little-endian columns.
pub const RECURRENCE_INPUT_ENDIANNESS: &str = "little";

/// Internal all-flow source marker; authenticated model/process inputs must
/// never carry this value as an ordinary concrete spin state.
pub(super) const DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS: i32 = i32::MIN;

fn validate_concrete_spin_state(value: i32, context: &str) -> crate::RusticolResult<()> {
    if value == DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS {
        return Err(crate::RusticolError::invalid_argument(format!(
            "{context} uses the reserved dynamic-union source-spin sentinel"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod direct_backend_tests;
#[cfg(test)]
mod tests;
