// SPDX-License-Identifier: 0BSD

//! Checked model-wide input for the compact recurrence builder.
//!
//! The Python producer projects a semantic recurrence-template catalog into
//! fixed-width primitive columns.  A Python decoder can zip those columns
//! directly into the row types below, construct [`OwnedRecurrenceTemplateInput`],
//! and call [`OwnedRecurrenceTemplateInput::validate`].  Validation resolves no
//! model-specific assumptions: it checks canonical catalogs, exact factors,
//! references, evaluator contracts, and propagator coverage before the state
//! builder is allowed to consume the input.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    CheckedTableRange, ExactComplexRational, LCColorComponentKind, LCColorComponentRole,
    LCColorSourceSeed, LCColorSourceSeedOperation, RECURRENCE_TEMPLATE_ABI, SemanticDigest,
    validate_concrete_spin_state, validate_packed_ranges, validate_u32_references,
};
use crate::{RusticolError, RusticolResult};

pub const RECURRENCE_TEMPLATE_INPUT_ABI: &str = "pyamplicol-recurrence-template-input-v1";
pub const RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION: u32 = 2;
pub const RECURRENCE_TEMPLATE_CANONICALIZATION_ABI: &str = "pyamplicol-canonical-json-v1";
pub const RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI: &str = "pyamplicol-exact-complex-rational-v1";
pub const MISSING_U32: u32 = u32::MAX;
const CONTACT_ORBIT_ALGORITHM: &str = "compiler-certified-contact-orbit";
const CONTACT_ORBIT_ALGORITHM_VERSION: u32 = 2;
const CONTACT_ORBIT_EVALUATOR_CLASS: &str = "constant-scalar-literal-singlet-self-conjugate-boson";
const HEFT_CONTACT_ORBIT_EVALUATOR_CLASS: &str = "scalar-heft-colored-contact-tree";

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ParameterKind {
    External = 0,
    Derived = 1,
    Constant = 2,
}

impl TryFrom<u8> for ParameterKind {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::External),
            1 => Ok(Self::Derived),
            2 => Ok(Self::Constant),
            _ => Err(invalid(format!("unsupported parameter kind {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ParameterValueType {
    Real = 0,
    Complex = 1,
}

impl TryFrom<u8> for ParameterValueType {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Real),
            1 => Ok(Self::Complex),
            _ => Err(invalid(format!("unsupported parameter value type {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CurrentOrientation {
    Particle = 0,
    Antiparticle = 1,
    SelfConjugate = 2,
}

impl TryFrom<u8> for CurrentOrientation {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Particle),
            1 => Ok(Self::Antiparticle),
            2 => Ok(Self::SelfConjugate),
            _ => Err(invalid(format!("unsupported current orientation {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ParticleStatistics {
    Boson = 0,
    Fermion = 1,
}

impl TryFrom<u8> for ParticleStatistics {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Boson),
            1 => Ok(Self::Fermion),
            _ => Err(invalid(format!("unsupported particle statistics {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum EvaluatorContractKind {
    Source = 0,
    Vertex = 1,
    Propagator = 2,
    Closure = 3,
    ModelParameter = 4,
}

impl TryFrom<u8> for EvaluatorContractKind {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Source),
            1 => Ok(Self::Vertex),
            2 => Ok(Self::Propagator),
            3 => Ok(Self::Closure),
            4 => Ok(Self::ModelParameter),
            _ => Err(invalid(format!(
                "unsupported evaluator contract kind {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EvaluatorCallableKind {
    PreparedKernel = 0,
    RusticolTemplate = 1,
}

impl TryFrom<u8> for EvaluatorCallableKind {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::PreparedKernel),
            1 => Ok(Self::RusticolTemplate),
            _ => Err(invalid(format!(
                "unsupported evaluator callable kind {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum OutputFactorSource {
    None = 0,
    CouplingReal = 1,
    CouplingImag = 2,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum ContactOrbitStage {
    Partial = 0,
    Final = 1,
}

impl TryFrom<u8> for ContactOrbitStage {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Partial),
            1 => Ok(Self::Final),
            _ => Err(invalid(format!("unsupported contact-orbit stage {value}"))),
        }
    }
}

impl TryFrom<u8> for OutputFactorSource {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::CouplingReal),
            2 => Ok(Self::CouplingImag),
            _ => Err(invalid(format!("unsupported output-factor source {value}"))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedRangeRow {
    pub id: u32,
    pub range: CheckedTableRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogHeaderRow {
    pub schema_version: u32,
    pub abi_string_id: u32,
    pub canonicalization_abi_string_id: u32,
    pub exact_scalar_abi_string_id: u32,
    pub compiled_model_digest_id: u32,
    pub prepared_kernel_pack_digest_id: u32,
    pub catalog_digest_id: u32,
    pub parameter_count: u32,
    pub current_state_count: u32,
    pub source_count: u32,
    pub quantum_flow_count: u32,
    pub contact_orbit_certificate_count: u32,
    pub contact_orbit_step_count: u32,
    pub transition_count: u32,
    pub propagator_count: u32,
    pub closure_count: u32,
    pub color_contraction_count: u32,
    pub symmetry_proof_count: u32,
    pub runtime_helicity_contract_count: u32,
    pub evaluator_binding_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CouplingOrderTermRow {
    pub set_id: u32,
    pub name_string_id: u32,
    pub power: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DigestCatalogRow {
    pub id: u32,
    pub value: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactFactorRow {
    pub id: u32,
    pub real_numerator_string_id: u32,
    pub real_denominator_string_id: u32,
    pub imag_numerator_string_id: u32,
    pub imag_denominator_string_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantumNumberFlowTermRow {
    pub flow_id: u32,
    pub name_string_id: u32,
    pub expression_string_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParameterRow {
    pub id: u32,
    pub template_string_id: u32,
    pub name_string_id: u32,
    pub kind: u8,
    pub value_type: u8,
    pub mutable: u8,
    pub default_factor_id: u32,
    pub exact_expression_digest_id: u32,
    pub dependency_sequence_id: u32,
    pub prepared_parameter_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CurrentStateRow {
    pub id: u32,
    pub template_string_id: u32,
    pub particle_id: i32,
    pub anti_particle_id: i32,
    pub species_string_id: u32,
    pub orientation: u8,
    pub statistics: u8,
    pub color_representation: i32,
    pub basis_string_id: u32,
    pub tensor_ordering_sequence_id: u32,
    pub dimension: u32,
    pub chirality: i32,
    pub lc_color_shape_string_id: u32,
    pub auxiliary_kind_string_id: u32,
    pub mass_parameter_id: u32,
    pub width_parameter_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceRow {
    pub id: u32,
    pub template_string_id: u32,
    pub state_template_id: u32,
    pub crossing_string_id: u32,
    pub wavefunction_family_string_id: u32,
    pub helicity: i32,
    pub spin_state: i32,
    pub flavour_flow_id: u32,
    pub quantum_number_flow_id: u32,
    pub lc_color_seed_operation: u8,
    pub lc_color_seed_shape_string_id: u32,
    pub lc_color_seed_component_kind: u8,
    pub lc_color_seed_component_role: u8,
    pub lc_color_seed_proof_digest_id: u32,
    pub lc_color_seed_provenance_sequence_id: u32,
    pub wavefunction_expression_digest_id: u32,
    pub evaluator_binding_id: u32,
    pub mass_parameter_id: u32,
    pub width_parameter_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeHelicityContractRow {
    pub id: u32,
    pub template_string_id: u32,
    pub full_state_template_id: u32,
    pub variant_range: CheckedTableRange,
    pub proof_algorithm_string_id: u32,
    pub proof_digest_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeHelicityVariantRow {
    pub id: u32,
    pub contract_id: u32,
    pub source_template_id: u32,
    pub source_state_template_id: u32,
    pub embedding_range: CheckedTableRange,
    pub projection_range: CheckedTableRange,
    pub proof_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeHelicityEmbeddingRow {
    pub variant_id: u32,
    pub full_component: u32,
    pub source_component: u32,
    pub factor_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeHelicityProjectionRow {
    pub variant_id: u32,
    pub source_component: u32,
    pub full_component: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantumFlowRow {
    pub id: u32,
    pub template_string_id: u32,
    pub input_state_sequence_id: u32,
    pub input_spin_sequence_id: u32,
    pub input_flavour_sequence_id: u32,
    pub input_quantum_sequence_id: u32,
    pub flavour_flow_operation_string_id: u32,
    pub quantum_number_flow_operation_string_id: u32,
    pub coupling_order_set_id: u32,
    pub result_state_template_id: u32,
    pub result_spin_state: i32,
    pub result_flavour_flow_id: u32,
    pub result_quantum_number_flow_id: u32,
    pub exact_coupling_factor_id: u32,
    pub predicate_digest_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContactOrbitCertificateRow {
    pub id: u32,
    pub template_string_id: u32,
    pub algorithm_string_id: u32,
    pub algorithm_version: u32,
    pub term_id: u32,
    pub vertex_string_id: u32,
    pub particle_string_sequence_id: u32,
    pub evaluator_class_string_id: u32,
    pub physical_leg_equivalence_sequence_id: u32,
    pub reconstruction_factor_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContactOrbitStepRow {
    pub id: u32,
    pub template_string_id: u32,
    pub certificate_id: u32,
    pub stage: u8,
    pub result_leg: u8,
    pub left_covered_leg_sequence_id: u32,
    pub right_covered_leg_sequence_id: u32,
    pub source_particle_leg_sequence_id: u32,
    pub reconstruction_factor_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransitionRow {
    pub id: u32,
    pub template_string_id: u32,
    pub input_state_sequence_id: u32,
    pub result_state_template_id: u32,
    pub quantum_flow_template_id: u32,
    pub evaluator_binding_id: u32,
    pub canonical_input_order_sequence_id: u32,
    pub momentum_convention_sequence_id: u32,
    pub coupling_parameter_sequence_id: u32,
    pub coupling_order_set_id: u32,
    pub color_contraction_template_id: u32,
    pub binding_coupling_factor_id: u32,
    pub exact_factor_id: u32,
    pub output_factor_source: u8,
    pub equivalence_class_string_id: u32,
    pub input_exchange_factor_id: u32,
    pub output_projection_string_id: u32,
    pub contact_orbit_step_sequence_id: u32,
    pub contact_orbit_step_semantic_digest_sequence_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PropagatorRow {
    pub id: u32,
    pub template_string_id: u32,
    pub state_template_id: u32,
    pub applies_propagator: u8,
    pub evaluator_binding_id: u32,
    pub numerator_expression_digest_id: u32,
    pub denominator_expression_digest_id: u32,
    pub mass_parameter_id: u32,
    pub width_parameter_id: u32,
    pub gauge_string_id: u32,
    pub linearity_proof_template_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClosureRow {
    pub id: u32,
    pub template_string_id: u32,
    pub input_state_sequence_id: u32,
    pub result_state_template_id: u32,
    pub evaluator_binding_id: u32,
    pub canonical_input_order_sequence_id: u32,
    pub coupling_parameter_sequence_id: u32,
    pub coupling_order_set_id: u32,
    pub eligible_quantum_flow_sequence_id: u32,
    pub color_contraction_template_id: u32,
    pub binding_coupling_factor_id: u32,
    pub exact_factor_id: u32,
    pub output_factor_source: u8,
    pub equivalence_class_string_id: u32,
    pub input_exchange_factor_id: u32,
    pub projection_string_id: u32,
    pub component_coefficient_sequence_id: u32,
    pub chirality_relation_string_id: u32,
    pub metric_signature_string_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ColorContractionRow {
    pub id: u32,
    pub template_string_id: u32,
    pub rule_kind_string_id: u32,
    pub input_representation_sequence_id: u32,
    pub has_output_representation: u8,
    pub output_representation: i32,
    pub ordered_open_string_arity: u32,
    pub exact_coefficient_factor_id: u32,
    pub witness_start: u64,
    pub witness_count: u64,
    pub nc_term_start: u64,
    pub nc_term_count: u64,
    pub expression_digest_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LCColorTransitionWitnessRow {
    pub color_contraction_id: u32,
    pub ordinal: u32,
    pub left_shape_string_id: u32,
    pub right_shape_string_id: u32,
    pub input_permutation: u8,
    pub reverse_parent_mask: u8,
    pub component_operation: u8,
    pub result_component_kind: u8,
    pub result_component_role: u8,
    pub result_shape_string_id: u32,
    pub exact_factor_id: u32,
    pub proof_digest_id: u32,
    pub input_port_pairing_sequence_id: u32,
    pub result_port_binding_sequence_id: u32,
    pub provenance_sequence_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ColorNcTermRow {
    pub color_contraction_id: u32,
    pub exponent: i32,
    pub factor_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymmetryProofRow {
    pub id: u32,
    pub template_string_id: u32,
    pub proof_algorithm_string_id: u32,
    pub subject_template_sequence_id: u32,
    pub input_permutation_sequence_id: u32,
    pub exact_phase_factor_id: u32,
    pub expression_digest_sequence_id: u32,
    pub witness_digest_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluatorBindingRow {
    pub id: u32,
    pub resolver_key_string_id: u32,
    pub prepared_kernel_id: u32,
    pub contract_kind: u8,
    pub callable_signature_digest_id: u32,
    pub input_state_sequence_id: u32,
    pub output_state_template_id: u32,
    pub input_layout_sequence_id: u32,
    pub output_layout_sequence_id: u32,
    pub exact_expression_digest_sequence_id: u32,
    pub semantic_template_sequence_id: u32,
    pub callable_kind: u8,
    pub runtime_template_string_id: u32,
    pub semantic_digest_id: u32,
}

macro_rules! define_template_inputs {
    ($( $field:ident : $item:ty ),+ $(,)?) => {
        #[derive(Clone, Copy, Debug)]
        pub struct RecurrenceTemplateInputView<'a> {
            pub input_abi: &'a str,
            pub catalog_digest: SemanticDigest,
            pub compiled_model_digest: SemanticDigest,
            pub prepared_kernel_pack_digest: SemanticDigest,
            $(pub $field: &'a [$item],)+
        }

        #[derive(Clone, Debug)]
        pub struct OwnedRecurrenceTemplateInput {
            pub input_abi: String,
            pub catalog_digest: SemanticDigest,
            pub compiled_model_digest: SemanticDigest,
            pub prepared_kernel_pack_digest: SemanticDigest,
            $(pub $field: Vec<$item>,)+
        }

        impl OwnedRecurrenceTemplateInput {
            pub fn as_view(&self) -> RecurrenceTemplateInputView<'_> {
                RecurrenceTemplateInputView {
                    input_abi: &self.input_abi,
                    catalog_digest: self.catalog_digest,
                    compiled_model_digest: self.compiled_model_digest,
                    prepared_kernel_pack_digest: self.prepared_kernel_pack_digest,
                    $($field: &self.$field,)+
                }
            }

            pub fn validate(self) -> RusticolResult<ValidatedRecurrenceTemplateInput> {
                let summary = self.as_view().validate()?;
                Ok(ValidatedRecurrenceTemplateInput { input: self, summary })
            }
        }
    };
}

define_template_inputs! {
    catalog_header: CatalogHeaderRow,
    coupling_order_ranges: IndexedRangeRow,
    coupling_order_terms: CouplingOrderTermRow,
    contact_orbit_certificates: ContactOrbitCertificateRow,
    contact_orbit_steps: ContactOrbitStepRow,
    current_states: CurrentStateRow,
    digest_catalog: DigestCatalogRow,
    evaluator_bindings: EvaluatorBindingRow,
    exact_factors: ExactFactorRow,
    flavour_flow_ranges: IndexedRangeRow,
    flavour_flow_values: i32,
    i32_sequence_ranges: IndexedRangeRow,
    i32_sequence_values: i32,
    parameters: ParameterRow,
    propagators: PropagatorRow,
    quantum_flows: QuantumFlowRow,
    quantum_number_flow_ranges: IndexedRangeRow,
    quantum_number_flow_terms: QuantumNumberFlowTermRow,
    runtime_helicity_contracts: RuntimeHelicityContractRow,
    runtime_helicity_variants: RuntimeHelicityVariantRow,
    runtime_helicity_embeddings: RuntimeHelicityEmbeddingRow,
    runtime_helicity_projections: RuntimeHelicityProjectionRow,
    sources: SourceRow,
    string_ranges: CheckedTableRange,
    string_bytes: u8,
    symmetry_proofs: SymmetryProofRow,
    transitions: TransitionRow,
    closures: ClosureRow,
    color_contractions: ColorContractionRow,
    lc_color_transition_witnesses: LCColorTransitionWitnessRow,
    color_nc_terms: ColorNcTermRow,
    u32_sequence_ranges: IndexedRangeRow,
    u32_sequence_values: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecurrenceTemplateValidationSummary {
    pub catalog_digest: SemanticDigest,
    pub compiled_model_digest: SemanticDigest,
    pub prepared_kernel_pack_digest: SemanticDigest,
    pub parameter_count: u32,
    pub current_state_count: u32,
    pub source_count: u32,
    pub quantum_flow_count: u32,
    pub contact_orbit_certificate_count: u32,
    pub contact_orbit_step_count: u32,
    pub transition_count: u32,
    pub propagator_count: u32,
    pub closure_count: u32,
    pub color_contraction_count: u32,
    pub lc_color_transition_witness_count: u32,
    pub symmetry_proof_count: u32,
    pub runtime_helicity_contract_count: u32,
    pub evaluator_binding_count: u32,
    pub prepared_kernel_count: u32,
}

#[derive(Clone, Debug)]
pub struct ValidatedRecurrenceTemplateInput {
    input: OwnedRecurrenceTemplateInput,
    summary: RecurrenceTemplateValidationSummary,
}

/// Stable semantic section names used by process/template cross-authentication.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RecurrenceSemanticTemplateKind {
    Parameter,
    CurrentState,
    Source,
    QuantumFlow,
    ContactOrbitCertificate,
    ContactOrbitStep,
    Transition,
    Propagator,
    Closure,
    ColorContraction,
    SymmetryProof,
}

impl RecurrenceSemanticTemplateKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parameter => "parameter",
            Self::CurrentState => "current-state",
            Self::Source => "source",
            Self::QuantumFlow => "quantum-flow",
            Self::ContactOrbitCertificate => "contact-orbit-certificate",
            Self::ContactOrbitStep => "contact-orbit-step",
            Self::Transition => "transition",
            Self::Propagator => "propagator",
            Self::Closure => "closure",
            Self::ColorContraction => "color-contraction",
            Self::SymmetryProof => "symmetry-proof",
        }
    }

    pub fn parse(value: &str) -> RusticolResult<Self> {
        match value {
            "parameter" => Ok(Self::Parameter),
            "current-state" => Ok(Self::CurrentState),
            "source" => Ok(Self::Source),
            "quantum-flow" => Ok(Self::QuantumFlow),
            "contact-orbit-certificate" => Ok(Self::ContactOrbitCertificate),
            "contact-orbit-step" => Ok(Self::ContactOrbitStep),
            "transition" => Ok(Self::Transition),
            "propagator" => Ok(Self::Propagator),
            "closure" => Ok(Self::Closure),
            "color-contraction" => Ok(Self::ColorContraction),
            "symmetry-proof" => Ok(Self::SymmetryProof),
            _ => Err(invalid(format!(
                "unsupported recurrence semantic-template kind {value:?}"
            ))),
        }
    }
}

/// One authenticated model-wide semantic record addressable by process input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecurrenceSemanticTemplateIdentity {
    pub kind: RecurrenceSemanticTemplateKind,
    pub template_id: u32,
    pub semantic_digest: SemanticDigest,
    pub prepared_kernel_id: Option<u32>,
}

/// Bounded index used to bind one process input to its prepared model catalog.
#[derive(Clone, Debug)]
pub struct RecurrenceTemplateSemanticIndex {
    pub compiled_model_digest: SemanticDigest,
    pub catalog_digest: SemanticDigest,
    records: BTreeMap<(RecurrenceSemanticTemplateKind, u32), RecurrenceSemanticTemplateIdentity>,
}

impl RecurrenceTemplateSemanticIndex {
    pub fn record(
        &self,
        kind: RecurrenceSemanticTemplateKind,
        template_id: u32,
    ) -> Option<RecurrenceSemanticTemplateIdentity> {
        self.records.get(&(kind, template_id)).copied()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl ValidatedRecurrenceTemplateInput {
    pub const fn summary(&self) -> RecurrenceTemplateValidationSummary {
        self.summary
    }

    pub const fn input(&self) -> &OwnedRecurrenceTemplateInput {
        &self.input
    }

    pub fn into_input(self) -> OwnedRecurrenceTemplateInput {
        self.input
    }

    /// Return the authenticated kind/default contract for every parameter.
    ///
    /// Numerical recurrence probes use this model-owned catalog to anchor
    /// mutable runtime defaults. Derived parameters deliberately retain no
    /// semantic probe default because their prepared slots are overwritten by
    /// the authenticated parameter-derivation evaluator.
    pub fn parameter_kind_and_default_factors(
        &self,
    ) -> RusticolResult<Vec<(ParameterKind, Option<ExactComplexRational>)>> {
        let input = self.input.as_view();
        let catalogs = input.validate_catalogs()?;
        input
            .parameters
            .iter()
            .map(|row| {
                Ok((
                    ParameterKind::try_from(row.kind)?,
                    optional_factor(
                        &catalogs.factors,
                        row.default_factor_id,
                        "parameter default",
                    )?
                    .copied(),
                ))
            })
            .collect()
    }

    pub(crate) fn closure_component_coefficient_catalog(
        &self,
    ) -> RusticolResult<Vec<Box<[ExactComplexRational]>>> {
        let input = self.input.as_view();
        let catalogs = input.validate_catalogs()?;
        input
            .closures
            .iter()
            .enumerate()
            .map(|(expected_id, row)| {
                let expected_id = u32::try_from(expected_id)
                    .map_err(|_| invalid("closure template count exceeds u32"))?;
                if row.id != expected_id {
                    return Err(invalid(format!(
                        "closure template {expected_id} has noncanonical ID {}",
                        row.id
                    )));
                }
                u32_sequence(
                    input,
                    row.component_coefficient_sequence_id,
                    "closure component coefficients",
                )?
                .iter()
                .map(|factor_id| {
                    catalogs
                        .factors
                        .get(*factor_id as usize)
                        .copied()
                        .ok_or_else(|| invalid("closure component factor ID is out of range"))
                })
                .collect::<RusticolResult<Vec<_>>>()
                .map(Vec::into_boxed_slice)
            })
            .collect()
    }

    pub fn semantic_index(&self) -> RusticolResult<RecurrenceTemplateSemanticIndex> {
        let input = self.input.as_view();
        let catalogs = input.validate_catalogs()?;

        let mut prepared_kernel_by_template_string_id = BTreeMap::new();
        for binding in input.evaluator_bindings {
            let prepared_kernel_id =
                (binding.prepared_kernel_id != MISSING_U32).then_some(binding.prepared_kernel_id);
            for template_string_id in u32_sequence(
                input,
                binding.semantic_template_sequence_id,
                "evaluator semantic templates",
            )? {
                if let Some(previous) = prepared_kernel_by_template_string_id
                    .insert(*template_string_id, prepared_kernel_id)
                    && previous != prepared_kernel_id
                {
                    return Err(invalid(
                        "one semantic template has inconsistent prepared-kernel ownership",
                    ));
                }
            }
        }

        let mut records = BTreeMap::new();
        macro_rules! register {
            ($kind:expr, $rows:expr) => {
                for row in $rows {
                    let semantic_digest = *catalogs
                        .digests
                        .get(row.semantic_digest_id as usize)
                        .ok_or_else(|| invalid("semantic-template digest ID is out of range"))?;
                    let identity = RecurrenceSemanticTemplateIdentity {
                        kind: $kind,
                        template_id: row.id,
                        semantic_digest,
                        prepared_kernel_id: prepared_kernel_by_template_string_id
                            .get(&row.template_string_id)
                            .copied()
                            .flatten(),
                    };
                    if records.insert(($kind, row.id), identity).is_some() {
                        return Err(invalid(format!(
                            "duplicate {} recurrence template ID {}",
                            $kind.as_str(),
                            row.id
                        )));
                    }
                }
            };
        }
        register!(RecurrenceSemanticTemplateKind::Parameter, input.parameters);
        register!(
            RecurrenceSemanticTemplateKind::CurrentState,
            input.current_states
        );
        register!(RecurrenceSemanticTemplateKind::Source, input.sources);
        register!(
            RecurrenceSemanticTemplateKind::QuantumFlow,
            input.quantum_flows
        );
        register!(
            RecurrenceSemanticTemplateKind::ContactOrbitCertificate,
            input.contact_orbit_certificates
        );
        register!(
            RecurrenceSemanticTemplateKind::ContactOrbitStep,
            input.contact_orbit_steps
        );
        register!(
            RecurrenceSemanticTemplateKind::Transition,
            input.transitions
        );
        register!(
            RecurrenceSemanticTemplateKind::Propagator,
            input.propagators
        );
        register!(RecurrenceSemanticTemplateKind::Closure, input.closures);
        register!(
            RecurrenceSemanticTemplateKind::ColorContraction,
            input.color_contractions
        );
        register!(
            RecurrenceSemanticTemplateKind::SymmetryProof,
            input.symmetry_proofs
        );

        Ok(RecurrenceTemplateSemanticIndex {
            compiled_model_digest: self.summary.compiled_model_digest,
            catalog_digest: self.summary.catalog_digest,
            records,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TemplateKind {
    Parameter,
    CurrentState,
    Source,
    RuntimeHelicity,
    QuantumFlow,
    ContactOrbitCertificate,
    ContactOrbitStep,
    Transition,
    Propagator,
    Closure,
    ColorContraction,
    SymmetryProof,
}

impl TemplateKind {
    const fn evaluator_contract(self) -> Option<EvaluatorContractKind> {
        match self {
            Self::Parameter => Some(EvaluatorContractKind::ModelParameter),
            Self::Source => Some(EvaluatorContractKind::Source),
            Self::Transition => Some(EvaluatorContractKind::Vertex),
            Self::Propagator => Some(EvaluatorContractKind::Propagator),
            Self::Closure => Some(EvaluatorContractKind::Closure),
            Self::CurrentState
            | Self::RuntimeHelicity
            | Self::QuantumFlow
            | Self::ContactOrbitCertificate
            | Self::ContactOrbitStep
            | Self::ColorContraction
            | Self::SymmetryProof => None,
        }
    }
}

struct ValidatedCatalogs<'a> {
    strings: Vec<&'a str>,
    digests: Vec<SemanticDigest>,
    factors: Vec<ExactComplexRational>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CallableContractKey {
    contract_kind: EvaluatorContractKind,
    callable_signature_digest_id: u32,
    input_layout: Vec<u32>,
    output_layout: Vec<u32>,
    exact_expression_digests: Vec<u32>,
}

impl<'a> RecurrenceTemplateInputView<'a> {
    pub fn validate(self) -> RusticolResult<RecurrenceTemplateValidationSummary> {
        if self.input_abi != RECURRENCE_TEMPLATE_INPUT_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence template input ABI {:?}; expected {:?}",
                self.input_abi, RECURRENCE_TEMPLATE_INPUT_ABI
            )));
        }
        let catalogs = self.validate_catalogs()?;
        self.validate_header(&catalogs)?;

        let mut template_kinds = BTreeMap::new();
        let mut semantic_digests = BTreeSet::new();
        self.validate_parameters(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_current_states(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_sources_basic(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_runtime_helicity_contracts(
            &catalogs,
            &mut template_kinds,
            &mut semantic_digests,
        )?;
        self.validate_quantum_flows(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_contact_orbit_records(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_color_contractions(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_symmetry_proofs_basic(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_transitions_basic(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_propagators_basic(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        self.validate_closures_basic(&catalogs, &mut template_kinds, &mut semantic_digests)?;
        let prepared_kernel_count =
            self.validate_evaluators(&catalogs, &template_kinds, &mut semantic_digests)?;
        self.validate_proof_subjects(&template_kinds)?;
        self.validate_evaluator_state_contracts(&template_kinds)?;

        Ok(RecurrenceTemplateValidationSummary {
            catalog_digest: self.catalog_digest,
            compiled_model_digest: self.compiled_model_digest,
            prepared_kernel_pack_digest: self.prepared_kernel_pack_digest,
            parameter_count: checked_len(self.parameters.len(), "parameters")?,
            current_state_count: checked_len(self.current_states.len(), "current states")?,
            source_count: checked_len(self.sources.len(), "sources")?,
            quantum_flow_count: checked_len(self.quantum_flows.len(), "quantum flows")?,
            contact_orbit_certificate_count: checked_len(
                self.contact_orbit_certificates.len(),
                "contact-orbit certificates",
            )?,
            contact_orbit_step_count: checked_len(
                self.contact_orbit_steps.len(),
                "contact-orbit steps",
            )?,
            transition_count: checked_len(self.transitions.len(), "transitions")?,
            propagator_count: checked_len(self.propagators.len(), "propagators")?,
            closure_count: checked_len(self.closures.len(), "closures")?,
            color_contraction_count: checked_len(
                self.color_contractions.len(),
                "color contractions",
            )?,
            lc_color_transition_witness_count: checked_len(
                self.lc_color_transition_witnesses.len(),
                "LC color transition witnesses",
            )?,
            symmetry_proof_count: checked_len(self.symmetry_proofs.len(), "symmetry proofs")?,
            runtime_helicity_contract_count: checked_len(
                self.runtime_helicity_contracts.len(),
                "runtime-helicity contracts",
            )?,
            evaluator_binding_count: checked_len(
                self.evaluator_bindings.len(),
                "evaluator bindings",
            )?,
            prepared_kernel_count,
        })
    }

    fn validate_catalogs(self) -> RusticolResult<ValidatedCatalogs<'a>> {
        validate_packed_ranges(
            "string catalog",
            self.string_ranges,
            self.string_bytes.len(),
        )?;
        let mut strings = Vec::with_capacity(self.string_ranges.len());
        let mut previous = None;
        for (index, range) in self.string_ranges.iter().copied().enumerate() {
            let bytes = &self.string_bytes
                [range.as_usize_range(self.string_bytes.len(), &format!("string {index}"))?];
            let value = std::str::from_utf8(bytes)
                .map_err(|error| invalid(format!("string {index} is not UTF-8: {error}")))?;
            if value.is_empty() {
                return Err(invalid(format!("string catalog row {index} is empty")));
            }
            if let Some(previous) = previous
                && previous >= value
            {
                return Err(invalid(format!(
                    "string catalog is not in strict canonical order at row {index}"
                )));
            }
            previous = Some(value);
            strings.push(value);
        }

        validate_canonical_ids(
            "digest catalog",
            self.digest_catalog.iter().map(|row| row.id),
        )?;
        let mut digests = Vec::with_capacity(self.digest_catalog.len());
        let mut previous = None;
        for (index, row) in self.digest_catalog.iter().enumerate() {
            let value = SemanticDigest::new(row.value).map_err(|error| {
                invalid(format!("digest catalog row {index} is invalid: {error}"))
            })?;
            if let Some(previous) = previous
                && previous >= value
            {
                return Err(invalid(format!(
                    "digest catalog is not in strict canonical order at row {index}"
                )));
            }
            previous = Some(value);
            digests.push(value);
        }

        self.validate_sequence_catalogs(&strings)?;
        let factors = self.validate_exact_factors(&strings)?;
        Ok(ValidatedCatalogs {
            strings,
            digests,
            factors,
        })
    }

    fn validate_sequence_catalogs(self, strings: &[&str]) -> RusticolResult<()> {
        validate_sequence_catalog(
            "u32 sequence catalog",
            self.u32_sequence_ranges,
            self.u32_sequence_values,
            false,
        )?;
        validate_sequence_catalog(
            "i32 sequence catalog",
            self.i32_sequence_ranges,
            self.i32_sequence_values,
            false,
        )?;
        validate_sequence_catalog(
            "flavour-flow catalog",
            self.flavour_flow_ranges,
            self.flavour_flow_values,
            true,
        )?;
        self.validate_coupling_order_catalog(strings)?;
        self.validate_quantum_number_flow_catalog(strings)?;
        Ok(())
    }

    fn validate_coupling_order_catalog(self, strings: &[&str]) -> RusticolResult<()> {
        validate_indexed_ranges(
            "coupling-order catalog",
            self.coupling_order_ranges,
            self.coupling_order_terms.len(),
        )?;
        let mut previous: Option<Vec<(u32, u32)>> = None;
        for (set_index, range) in self.coupling_order_ranges.iter().enumerate() {
            let terms = &self.coupling_order_terms[range
                .range
                .as_usize_range(self.coupling_order_terms.len(), "coupling-order set")?];
            let mut current = Vec::with_capacity(terms.len());
            let mut previous_name = None;
            for term in terms {
                if term.set_id as usize != set_index {
                    return Err(invalid(format!(
                        "coupling-order term names set {}, expected {set_index}",
                        term.set_id
                    )));
                }
                required_string(strings, term.name_string_id, "coupling-order name")?;
                if let Some(previous_name) = previous_name
                    && previous_name >= term.name_string_id
                {
                    return Err(invalid(format!(
                        "coupling-order set {set_index} is not strictly name ordered"
                    )));
                }
                previous_name = Some(term.name_string_id);
                current.push((term.name_string_id, term.power));
            }
            if let Some(previous) = previous.as_ref()
                && previous.as_slice() >= current.as_slice()
            {
                return Err(invalid(format!(
                    "coupling-order catalog is not in strict canonical order at set {set_index}"
                )));
            }
            previous = Some(current);
        }
        Ok(())
    }

    fn validate_quantum_number_flow_catalog(self, strings: &[&str]) -> RusticolResult<()> {
        validate_indexed_ranges(
            "quantum-number-flow catalog",
            self.quantum_number_flow_ranges,
            self.quantum_number_flow_terms.len(),
        )?;
        let mut previous: Option<Vec<(u32, u32)>> = None;
        for (flow_index, range) in self.quantum_number_flow_ranges.iter().enumerate() {
            let terms = &self.quantum_number_flow_terms[range
                .range
                .as_usize_range(self.quantum_number_flow_terms.len(), "quantum-number flow")?];
            let mut current = Vec::with_capacity(terms.len());
            let mut previous_name = None;
            for term in terms {
                if term.flow_id as usize != flow_index {
                    return Err(invalid(format!(
                        "quantum-number-flow term names flow {}, expected {flow_index}",
                        term.flow_id
                    )));
                }
                required_string(strings, term.name_string_id, "quantum-number name")?;
                required_string(
                    strings,
                    term.expression_string_id,
                    "quantum-number expression",
                )?;
                if let Some(previous_name) = previous_name
                    && previous_name >= term.name_string_id
                {
                    return Err(invalid(format!(
                        "quantum-number flow {flow_index} is not strictly name ordered"
                    )));
                }
                previous_name = Some(term.name_string_id);
                current.push((term.name_string_id, term.expression_string_id));
            }
            if let Some(previous) = previous.as_ref()
                && previous.as_slice() >= current.as_slice()
            {
                return Err(invalid(format!(
                    "quantum-number-flow catalog is not in strict canonical order at flow {flow_index}"
                )));
            }
            previous = Some(current);
        }
        Ok(())
    }

    fn validate_exact_factors(self, strings: &[&str]) -> RusticolResult<Vec<ExactComplexRational>> {
        validate_canonical_ids(
            "exact-factor catalog",
            self.exact_factors.iter().map(|row| row.id),
        )?;
        let mut factors = Vec::with_capacity(self.exact_factors.len());
        let mut previous = None;
        for (index, row) in self.exact_factors.iter().enumerate() {
            let real_numerator = required_string(
                strings,
                row.real_numerator_string_id,
                "exact real numerator",
            )?;
            let real_denominator = required_string(
                strings,
                row.real_denominator_string_id,
                "exact real denominator",
            )?;
            let imag_numerator = required_string(
                strings,
                row.imag_numerator_string_id,
                "exact imaginary numerator",
            )?;
            let imag_denominator = required_string(
                strings,
                row.imag_denominator_string_id,
                "exact imaginary denominator",
            )?;
            let raw_key = (
                parse_canonical_i128(real_numerator, false, "exact real numerator")?,
                parse_canonical_i128(real_denominator, true, "exact real denominator")?,
                parse_canonical_i128(imag_numerator, false, "exact imaginary numerator")?,
                parse_canonical_i128(imag_denominator, true, "exact imaginary denominator")?,
            );
            let factor = ExactComplexRational::parse_parts(
                real_numerator,
                real_denominator,
                imag_numerator,
                imag_denominator,
            )
            .map_err(|error| invalid(format!("exact factor {index} is invalid: {error}")))?;
            if factor.real().numerator() != raw_key.0
                || factor.real().denominator() != raw_key.1
                || factor.imag().numerator() != raw_key.2
                || factor.imag().denominator() != raw_key.3
            {
                return Err(invalid(format!(
                    "exact factor {index} is not reduced canonically"
                )));
            }
            if let Some(previous) = previous
                && previous >= raw_key
            {
                return Err(invalid(format!(
                    "exact-factor catalog is not in strict canonical order at row {index}"
                )));
            }
            previous = Some(raw_key);
            factors.push(factor);
        }
        Ok(factors)
    }

    fn validate_header(self, catalogs: &ValidatedCatalogs<'_>) -> RusticolResult<()> {
        if self.catalog_header.len() != 1 {
            return Err(invalid(format!(
                "catalog_header must contain one row, found {}",
                self.catalog_header.len()
            )));
        }
        let header = self.catalog_header[0];
        if header.schema_version != RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence template input schema {}; expected {}",
                header.schema_version, RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION
            )));
        }
        require_string_value(
            &catalogs.strings,
            header.abi_string_id,
            RECURRENCE_TEMPLATE_ABI,
            "recurrence template ABI",
        )?;
        require_string_value(
            &catalogs.strings,
            header.canonicalization_abi_string_id,
            RECURRENCE_TEMPLATE_CANONICALIZATION_ABI,
            "recurrence canonicalization ABI",
        )?;
        require_string_value(
            &catalogs.strings,
            header.exact_scalar_abi_string_id,
            RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI,
            "recurrence exact-scalar ABI",
        )?;
        if required_digest(
            &catalogs.digests,
            header.compiled_model_digest_id,
            "compiled-model digest",
        )? != self.compiled_model_digest
        {
            return Err(invalid(
                "compiled-model digest does not match the input envelope",
            ));
        }
        if required_digest(
            &catalogs.digests,
            header.prepared_kernel_pack_digest_id,
            "prepared-kernel-pack digest",
        )? != self.prepared_kernel_pack_digest
        {
            return Err(invalid(
                "prepared-kernel-pack digest does not match the input envelope",
            ));
        }
        if required_digest(
            &catalogs.digests,
            header.catalog_digest_id,
            "catalog digest",
        )? != self.catalog_digest
        {
            return Err(invalid("catalog digest does not match the input envelope"));
        }

        let expected = [
            ("parameters", header.parameter_count, self.parameters.len()),
            (
                "current states",
                header.current_state_count,
                self.current_states.len(),
            ),
            ("sources", header.source_count, self.sources.len()),
            (
                "quantum flows",
                header.quantum_flow_count,
                self.quantum_flows.len(),
            ),
            (
                "contact-orbit certificates",
                header.contact_orbit_certificate_count,
                self.contact_orbit_certificates.len(),
            ),
            (
                "contact-orbit steps",
                header.contact_orbit_step_count,
                self.contact_orbit_steps.len(),
            ),
            (
                "transitions",
                header.transition_count,
                self.transitions.len(),
            ),
            (
                "propagators",
                header.propagator_count,
                self.propagators.len(),
            ),
            ("closures", header.closure_count, self.closures.len()),
            (
                "color contractions",
                header.color_contraction_count,
                self.color_contractions.len(),
            ),
            (
                "symmetry proofs",
                header.symmetry_proof_count,
                self.symmetry_proofs.len(),
            ),
            (
                "runtime-helicity contracts",
                header.runtime_helicity_contract_count,
                self.runtime_helicity_contracts.len(),
            ),
            (
                "evaluator bindings",
                header.evaluator_binding_count,
                self.evaluator_bindings.len(),
            ),
        ];
        for (label, declared, actual) in expected {
            if usize::try_from(declared).ok() != Some(actual) {
                return Err(invalid(format!(
                    "catalog header declares {declared} {label}, found {actual}"
                )));
            }
        }
        Ok(())
    }

    fn validate_parameters(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "parameter",
            self.parameters.iter().map(|row| row.id),
            self.parameters.iter().map(|row| row.template_string_id),
        )?;
        for (index, row) in self.parameters.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::Parameter,
                "parameter",
            )?;
            register_semantic_digest(semantic_digests, row.semantic_digest_id, "parameter")?;
            required_string(
                &catalogs.strings,
                row.template_string_id,
                "parameter template",
            )?;
            required_string(&catalogs.strings, row.name_string_id, "parameter name")?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "parameter semantic digest",
            )?;
            let kind = ParameterKind::try_from(row.kind)?;
            let value_type = ParameterValueType::try_from(row.value_type)?;
            validate_bool(row.mutable, "parameter mutable")?;
            let default = optional_factor(
                &catalogs.factors,
                row.default_factor_id,
                "parameter default",
            )?;
            optional_digest(
                &catalogs.digests,
                row.exact_expression_digest_id,
                "parameter exact expression",
            )?;
            let dependencies =
                u32_sequence(self, row.dependency_sequence_id, "parameter dependencies")?;
            validate_strict_u32(dependencies, "parameter dependencies")?;
            validate_u32_references(
                dependencies,
                self.parameters.len(),
                "parameter dependencies",
            )?;
            if dependencies.contains(&(index as u32)) {
                return Err(invalid(format!(
                    "parameter {index} depends directly on itself"
                )));
            }
            match kind {
                ParameterKind::Derived => {
                    if row.exact_expression_digest_id == MISSING_U32 {
                        return Err(invalid(format!(
                            "derived parameter {index} has no exact expression digest"
                        )));
                    }
                }
                ParameterKind::External | ParameterKind::Constant => {
                    if default.is_none() {
                        return Err(invalid(format!(
                            "external/constant parameter {index} has no exact default"
                        )));
                    }
                }
            }
            if value_type == ParameterValueType::Real
                && default.is_some_and(|value| !value.imag().is_zero())
            {
                return Err(invalid(format!(
                    "real parameter {index} has a complex default"
                )));
            }
        }
        self.validate_parameter_dependency_graph()?;
        Ok(())
    }

    fn validate_parameter_dependency_graph(self) -> RusticolResult<()> {
        let mut indegree = vec![0usize; self.parameters.len()];
        let mut dependents = vec![Vec::<usize>::new(); self.parameters.len()];
        for (index, row) in self.parameters.iter().enumerate() {
            let dependencies =
                u32_sequence(self, row.dependency_sequence_id, "parameter dependencies")?;
            indegree[index] = dependencies.len();
            for dependency in dependencies {
                dependents[*dependency as usize].push(index);
            }
        }
        let mut ready = indegree
            .iter()
            .enumerate()
            .filter_map(|(index, count)| (*count == 0).then_some(index))
            .collect::<Vec<_>>();
        let mut visited = 0usize;
        while let Some(index) = ready.pop() {
            visited += 1;
            for dependent in &dependents[index] {
                indegree[*dependent] -= 1;
                if indegree[*dependent] == 0 {
                    ready.push(*dependent);
                }
            }
        }
        if visited != self.parameters.len() {
            let cyclic = indegree
                .iter()
                .enumerate()
                .filter_map(|(index, count)| (*count > 0).then_some(index.to_string()))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(invalid(format!(
                "parameter dependency graph contains a cycle involving: {cyclic}"
            )));
        }
        Ok(())
    }

    fn validate_current_states(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "current state",
            self.current_states.iter().map(|row| row.id),
            self.current_states.iter().map(|row| row.template_string_id),
        )?;
        for (index, row) in self.current_states.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::CurrentState,
                "current state",
            )?;
            register_semantic_digest(semantic_digests, row.semantic_digest_id, "current state")?;
            for (id, label) in [
                (row.template_string_id, "current-state template"),
                (row.species_string_id, "current-state species"),
                (row.basis_string_id, "current-state basis"),
            ] {
                required_string(&catalogs.strings, id, label)?;
            }
            let color_shape = required_string(
                &catalogs.strings,
                row.lc_color_shape_string_id,
                "current-state LC color shape",
            )?;
            validate_lc_color_shape(
                color_shape,
                row.color_representation,
                &format!("current state {index}"),
            )?;
            optional_string(
                &catalogs.strings,
                row.auxiliary_kind_string_id,
                "current-state auxiliary kind",
            )?;
            CurrentOrientation::try_from(row.orientation)?;
            ParticleStatistics::try_from(row.statistics)?;
            if row.dimension == 0 {
                return Err(invalid(format!("current state {index} has zero dimension")));
            }
            let tensor_ordering = u32_sequence(
                self,
                row.tensor_ordering_sequence_id,
                "current tensor ordering",
            )?;
            if tensor_ordering.len() != row.dimension as usize {
                return Err(invalid(format!(
                    "current state {index} dimension {} does not match tensor ordering length {}",
                    row.dimension,
                    tensor_ordering.len()
                )));
            }
            for string_id in tensor_ordering {
                required_string(&catalogs.strings, *string_id, "current tensor label")?;
            }
            optional_reference(
                row.mass_parameter_id,
                self.parameters.len(),
                "current mass parameter",
            )?;
            optional_reference(
                row.width_parameter_id,
                self.parameters.len(),
                "current width parameter",
            )?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "current-state semantic digest",
            )?;
        }
        Ok(())
    }

    fn validate_sources_basic(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "source",
            self.sources.iter().map(|row| row.id),
            self.sources.iter().map(|row| row.template_string_id),
        )?;
        for row in self.sources {
            validate_concrete_spin_state(row.spin_state, "source template spin state")?;
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::Source,
                "source",
            )?;
            register_semantic_digest(semantic_digests, row.semantic_digest_id, "source")?;
            required_reference(
                row.state_template_id,
                self.current_states.len(),
                "source state",
            )?;
            required_reference(
                row.evaluator_binding_id,
                self.evaluator_bindings.len(),
                "source evaluator",
            )?;
            required_reference(
                row.flavour_flow_id,
                self.flavour_flow_ranges.len(),
                "source flavour flow",
            )?;
            required_reference(
                row.quantum_number_flow_id,
                self.quantum_number_flow_ranges.len(),
                "source quantum-number flow",
            )?;
            let state = self.current_states[row.state_template_id as usize];
            if row.lc_color_seed_shape_string_id != state.lc_color_shape_string_id {
                return Err(invalid(
                    "source LC color seed shape does not match its current state",
                ));
            }
            let seed_shape = required_string(
                &catalogs.strings,
                row.lc_color_seed_shape_string_id,
                "source LC color seed shape",
            )?;
            validate_lc_color_shape(
                seed_shape,
                state.color_representation,
                "source LC color seed",
            )?;
            let seed_operation = LCColorSourceSeedOperation::try_from(row.lc_color_seed_operation)?;
            let seed_component_kind = if row.lc_color_seed_component_kind == u8::MAX {
                None
            } else {
                Some(LCColorComponentKind::try_from(
                    row.lc_color_seed_component_kind,
                )?)
            };
            let seed_component_role =
                LCColorComponentRole::try_from(row.lc_color_seed_component_role)?;
            let expected_seed = match seed_shape {
                "singlet-forest" => (
                    LCColorSourceSeedOperation::Empty,
                    None,
                    LCColorComponentRole::None,
                ),
                "fundamental-open-string" | "antifundamental-open-string" => (
                    LCColorSourceSeedOperation::Singleton,
                    Some(LCColorComponentKind::OpenString),
                    LCColorComponentRole::Active,
                ),
                "adjoint-segment" => (
                    LCColorSourceSeedOperation::Singleton,
                    Some(LCColorComponentKind::AdjointSegment),
                    LCColorComponentRole::Active,
                ),
                _ => unreachable!("validated LC color shape"),
            };
            if (seed_operation, seed_component_kind, seed_component_role) != expected_seed {
                return Err(invalid(
                    "source LC color seed does not match its certified output shape",
                ));
            }
            let seed_proof = required_digest(
                &catalogs.digests,
                row.lc_color_seed_proof_digest_id,
                "source LC color seed proof",
            )?;
            LCColorSourceSeed::new(
                seed_operation,
                row.lc_color_seed_shape_string_id,
                seed_component_kind,
                seed_component_role,
                seed_proof,
            )?;
            let seed_provenance = u32_sequence(
                self,
                row.lc_color_seed_provenance_sequence_id,
                "source LC color seed provenance",
            )?;
            validate_string_pair_sequence(
                &catalogs.strings,
                seed_provenance,
                "source LC color seed provenance",
            )?;
            optional_reference(
                row.mass_parameter_id,
                self.parameters.len(),
                "source mass parameter",
            )?;
            optional_reference(
                row.width_parameter_id,
                self.parameters.len(),
                "source width parameter",
            )?;
            required_string(&catalogs.strings, row.template_string_id, "source template")?;
            required_string(&catalogs.strings, row.crossing_string_id, "source crossing")?;
            let family = required_string(
                &catalogs.strings,
                row.wavefunction_family_string_id,
                "source wavefunction family",
            )?;
            if !matches!(
                family,
                "scalar" | "fermion" | "vector" | "spin2" | "ghost" | "auxiliary"
            ) {
                return Err(invalid(format!(
                    "unsupported source wavefunction family {family:?}"
                )));
            }
            required_digest(
                &catalogs.digests,
                row.wavefunction_expression_digest_id,
                "source wavefunction expression",
            )?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "source semantic digest",
            )?;
        }
        Ok(())
    }

    fn validate_runtime_helicity_contracts(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "runtime-helicity contract",
            self.runtime_helicity_contracts.iter().map(|row| row.id),
            self.runtime_helicity_contracts
                .iter()
                .map(|row| row.template_string_id),
        )?;
        validate_canonical_ids(
            "runtime-helicity variant",
            self.runtime_helicity_variants.iter().map(|row| row.id),
        )?;
        let contract_ranges = self
            .runtime_helicity_contracts
            .iter()
            .map(|row| row.variant_range)
            .collect::<Vec<_>>();
        validate_packed_ranges(
            "runtime-helicity variants",
            &contract_ranges,
            self.runtime_helicity_variants.len(),
        )?;
        let embedding_ranges = self
            .runtime_helicity_variants
            .iter()
            .map(|row| row.embedding_range)
            .collect::<Vec<_>>();
        validate_packed_ranges(
            "runtime-helicity embeddings",
            &embedding_ranges,
            self.runtime_helicity_embeddings.len(),
        )?;
        let projection_ranges = self
            .runtime_helicity_variants
            .iter()
            .map(|row| row.projection_range)
            .collect::<Vec<_>>();
        validate_packed_ranges(
            "runtime-helicity projections",
            &projection_ranges,
            self.runtime_helicity_projections.len(),
        )?;

        let mut owned_sources = BTreeSet::new();
        for contract in self.runtime_helicity_contracts {
            register_template(
                template_kinds,
                contract.template_string_id,
                TemplateKind::RuntimeHelicity,
                "runtime-helicity contract",
            )?;
            register_semantic_digest(
                semantic_digests,
                contract.semantic_digest_id,
                "runtime-helicity contract",
            )?;
            required_string(
                &catalogs.strings,
                contract.template_string_id,
                "runtime-helicity template",
            )?;
            required_string(
                &catalogs.strings,
                contract.proof_algorithm_string_id,
                "runtime-helicity proof algorithm",
            )?;
            required_digest(
                &catalogs.digests,
                contract.proof_digest_id,
                "runtime-helicity proof",
            )?;
            required_digest(
                &catalogs.digests,
                contract.semantic_digest_id,
                "runtime-helicity semantic digest",
            )?;
            let full_state_index = required_reference(
                contract.full_state_template_id,
                self.current_states.len(),
                "runtime-helicity full state",
            )?;
            let full_dimension = self.current_states[full_state_index].dimension as usize;
            let variants = &self.runtime_helicity_variants[contract.variant_range.as_usize_range(
                self.runtime_helicity_variants.len(),
                "runtime-helicity variants",
            )?];
            if variants.is_empty() {
                return Err(invalid(format!(
                    "runtime-helicity contract {} has no variants",
                    contract.id
                )));
            }
            for variant in variants {
                if variant.contract_id != contract.id {
                    return Err(invalid(format!(
                        "runtime-helicity variant {} names contract {}, expected {}",
                        variant.id, variant.contract_id, contract.id
                    )));
                }
                let source_index = required_reference(
                    variant.source_template_id,
                    self.sources.len(),
                    "runtime-helicity source",
                )?;
                if !owned_sources.insert(variant.source_template_id) {
                    return Err(invalid(format!(
                        "source template {} belongs to multiple runtime-helicity variants",
                        variant.source_template_id
                    )));
                }
                let source_state_index = required_reference(
                    variant.source_state_template_id,
                    self.current_states.len(),
                    "runtime-helicity source state",
                )?;
                if self.sources[source_index].state_template_id != variant.source_state_template_id
                {
                    return Err(invalid(format!(
                        "runtime-helicity variant {} source-state contract is inconsistent",
                        variant.id
                    )));
                }
                required_digest(
                    &catalogs.digests,
                    variant.proof_digest_id,
                    "runtime-helicity variant proof",
                )?;
                let source_dimension = self.current_states[source_state_index].dimension as usize;
                let embeddings =
                    &self.runtime_helicity_embeddings[variant.embedding_range.as_usize_range(
                        self.runtime_helicity_embeddings.len(),
                        "runtime-helicity embedding",
                    )?];
                let projections =
                    &self.runtime_helicity_projections[variant.projection_range.as_usize_range(
                        self.runtime_helicity_projections.len(),
                        "runtime-helicity projection",
                    )?];
                if embeddings.len() != full_dimension || projections.len() != source_dimension {
                    return Err(invalid(format!(
                        "runtime-helicity variant {} dimensions do not match its current states",
                        variant.id
                    )));
                }
                for (component, embedding) in embeddings.iter().enumerate() {
                    if embedding.variant_id != variant.id
                        || embedding.full_component as usize != component
                    {
                        return Err(invalid(format!(
                            "runtime-helicity variant {} has noncanonical embedding rows",
                            variant.id
                        )));
                    }
                    let factor = required_factor(
                        &catalogs.factors,
                        embedding.factor_id,
                        "runtime-helicity embedding factor",
                    )?;
                    if embedding.source_component == MISSING_U32 {
                        if !factor.is_zero() {
                            return Err(invalid(
                                "an absent runtime-helicity source component has nonzero factor",
                            ));
                        }
                    } else if embedding.source_component as usize >= source_dimension
                        || factor.is_zero()
                    {
                        return Err(invalid(
                            "a runtime-helicity source embedding is out of range or zero",
                        ));
                    }
                }
                for (component, projection) in projections.iter().enumerate() {
                    if projection.variant_id != variant.id
                        || projection.source_component as usize != component
                        || projection.full_component as usize >= full_dimension
                    {
                        return Err(invalid(format!(
                            "runtime-helicity variant {} has invalid projection rows",
                            variant.id
                        )));
                    }
                    if embeddings[projection.full_component as usize].source_component
                        != projection.source_component
                    {
                        return Err(invalid(format!(
                            "runtime-helicity variant {} embedding and projection disagree",
                            variant.id
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_quantum_flows(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "quantum flow",
            self.quantum_flows.iter().map(|row| row.id),
            self.quantum_flows.iter().map(|row| row.template_string_id),
        )?;
        let mut static_quantum_flows = BTreeMap::<u32, u32>::new();
        for (index, row) in self.quantum_flows.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::QuantumFlow,
                "quantum flow",
            )?;
            register_semantic_digest(semantic_digests, row.semantic_digest_id, "quantum flow")?;
            required_string(
                &catalogs.strings,
                row.template_string_id,
                "quantum-flow template",
            )?;
            let states = u32_sequence(
                self,
                row.input_state_sequence_id,
                "quantum-flow input states",
            )?;
            if states.len() != 2 {
                return Err(invalid(format!(
                    "quantum flow {index} must have binary input arity"
                )));
            }
            validate_u32_references(
                states,
                self.current_states.len(),
                "quantum-flow input states",
            )?;
            let spins = i32_sequence(self, row.input_spin_sequence_id, "quantum-flow spins")?;
            for spin in spins {
                validate_concrete_spin_state(*spin, "quantum-flow input spin state")?;
            }
            validate_concrete_spin_state(row.result_spin_state, "quantum-flow result spin state")?;
            let flavour_ids = u32_sequence(
                self,
                row.input_flavour_sequence_id,
                "quantum-flow flavour flows",
            )?;
            let quantum_ids = u32_sequence(
                self,
                row.input_quantum_sequence_id,
                "quantum-flow quantum-number flows",
            )?;
            if states.len() != spins.len()
                || states.len() != flavour_ids.len()
                || states.len() != quantum_ids.len()
            {
                return Err(invalid(format!(
                    "quantum flow {index} input contract columns have different arity"
                )));
            }
            validate_u32_references(
                flavour_ids,
                self.flavour_flow_ranges.len(),
                "quantum-flow flavour flows",
            )?;
            validate_u32_references(
                quantum_ids,
                self.quantum_number_flow_ranges.len(),
                "quantum-flow quantum-number flows",
            )?;
            let operation = required_string(
                &catalogs.strings,
                row.flavour_flow_operation_string_id,
                "quantum-flow flavour operation",
            )?;
            if !matches!(
                operation,
                "constant-result"
                    | "append-left-result"
                    | "append-right-result"
                    | "concat-left-right-result"
            ) {
                return Err(invalid(format!(
                    "quantum flow {index} has unsupported flavour operation {operation:?}"
                )));
            }
            let quantum_operation = required_string(
                &catalogs.strings,
                row.quantum_number_flow_operation_string_id,
                "quantum-flow quantum-number operation",
            )?;
            if quantum_operation != "particle-static-result" {
                return Err(invalid(format!(
                    "quantum flow {index} has unsupported quantum-number operation {quantum_operation:?}"
                )));
            }
            required_reference(
                row.result_state_template_id,
                self.current_states.len(),
                "quantum-flow result state",
            )?;
            required_reference(
                row.result_flavour_flow_id,
                self.flavour_flow_ranges.len(),
                "quantum-flow result flavour flow",
            )?;
            required_reference(
                row.result_quantum_number_flow_id,
                self.quantum_number_flow_ranges.len(),
                "quantum-flow result quantum-number flow",
            )?;
            let previous_quantum_flow = static_quantum_flows
                .entry(row.result_state_template_id)
                .or_insert(row.result_quantum_number_flow_id);
            if *previous_quantum_flow != row.result_quantum_number_flow_id {
                return Err(invalid(format!(
                    "quantum flow {index} violates particle-static quantum-number flow"
                )));
            }
            validate_flavour_flow_witness(
                self,
                index,
                operation,
                flavour_ids,
                row.result_state_template_id,
                row.result_flavour_flow_id,
            )?;
            required_reference(
                row.coupling_order_set_id,
                self.coupling_order_ranges.len(),
                "quantum-flow coupling-order set",
            )?;
            required_factor(
                &catalogs.factors,
                row.exact_coupling_factor_id,
                "quantum-flow exact coupling",
            )?;
            required_digest(
                &catalogs.digests,
                row.predicate_digest_id,
                "quantum-flow predicate",
            )?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "quantum-flow semantic digest",
            )?;
        }
        Ok(())
    }

    fn validate_contact_orbit_records(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "contact-orbit certificate",
            self.contact_orbit_certificates.iter().map(|row| row.id),
            self.contact_orbit_certificates
                .iter()
                .map(|row| row.template_string_id),
        )?;
        for (index, row) in self.contact_orbit_certificates.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::ContactOrbitCertificate,
                "contact-orbit certificate",
            )?;
            register_semantic_digest(
                semantic_digests,
                row.semantic_digest_id,
                "contact-orbit certificate",
            )?;
            required_string(
                &catalogs.strings,
                row.template_string_id,
                "contact-orbit certificate template",
            )?;
            require_string_value(
                &catalogs.strings,
                row.algorithm_string_id,
                CONTACT_ORBIT_ALGORITHM,
                "contact-orbit certificate algorithm",
            )?;
            if row.algorithm_version != CONTACT_ORBIT_ALGORITHM_VERSION {
                return Err(invalid(format!(
                    "contact-orbit certificate {index} has unsupported algorithm version {}",
                    row.algorithm_version
                )));
            }
            required_string(
                &catalogs.strings,
                row.vertex_string_id,
                "contact-orbit certificate vertex",
            )?;
            let evaluator_class = required_string(
                &catalogs.strings,
                row.evaluator_class_string_id,
                "contact-orbit evaluator class",
            )?;
            if !matches!(
                evaluator_class,
                CONTACT_ORBIT_EVALUATOR_CLASS | HEFT_CONTACT_ORBIT_EVALUATOR_CLASS
            ) {
                return Err(RusticolError::compatibility(format!(
                    "unsupported contact-orbit evaluator class {evaluator_class:?}"
                )));
            }
            let particles = u32_sequence(
                self,
                row.particle_string_sequence_id,
                "contact-orbit certificate particles",
            )?;
            let equivalence_classes = u32_sequence(
                self,
                row.physical_leg_equivalence_sequence_id,
                "contact-orbit certificate physical-leg equivalence classes",
            )?;
            if !matches!(particles.len(), 4 | 5) || equivalence_classes.len() != particles.len() {
                return Err(invalid(format!(
                    "contact-orbit certificate {index} must describe four or five physical legs"
                )));
            }
            for particle in particles {
                required_string(
                    &catalogs.strings,
                    *particle,
                    "contact-orbit certificate particle",
                )?;
            }
            let mut class_by_particle = BTreeMap::<u32, u32>::new();
            let mut next_class = 0u32;
            let expected_classes = particles
                .iter()
                .map(|particle| {
                    *class_by_particle.entry(*particle).or_insert_with(|| {
                        let class = next_class;
                        next_class += 1;
                        class
                    })
                })
                .collect::<Vec<_>>();
            if equivalence_classes != expected_classes.as_slice() {
                return Err(invalid(format!(
                    "contact-orbit certificate {index} has noncanonical physical-leg equivalence classes"
                )));
            }
            let reconstruction = required_factor(
                &catalogs.factors,
                row.reconstruction_factor_id,
                "contact-orbit certificate reconstruction factor",
            )?;
            if *reconstruction != ExactComplexRational::ONE {
                return Err(invalid(format!(
                    "contact-orbit certificate {index} reconstruction factor is not one"
                )));
            }
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "contact-orbit certificate semantic digest",
            )?;
        }

        validate_record_ids(
            "contact-orbit step",
            self.contact_orbit_steps.iter().map(|row| row.id),
            self.contact_orbit_steps
                .iter()
                .map(|row| row.template_string_id),
        )?;
        for (index, row) in self.contact_orbit_steps.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::ContactOrbitStep,
                "contact-orbit step",
            )?;
            register_semantic_digest(
                semantic_digests,
                row.semantic_digest_id,
                "contact-orbit step",
            )?;
            required_string(
                &catalogs.strings,
                row.template_string_id,
                "contact-orbit step template",
            )?;
            required_reference(
                row.certificate_id,
                self.contact_orbit_certificates.len(),
                "contact-orbit step certificate",
            )?;
            let certificate = self
                .contact_orbit_certificates
                .get(row.certificate_id as usize)
                .ok_or_else(|| invalid("contact-orbit step certificate is absent"))?;
            let arity = u32_sequence(
                self,
                certificate.physical_leg_equivalence_sequence_id,
                "contact-orbit step certificate arity",
            )?
            .len();
            let stage = ContactOrbitStage::try_from(row.stage)?;
            if usize::from(row.result_leg) >= arity {
                return Err(invalid(format!(
                    "contact-orbit step {index} result leg is outside arity"
                )));
            }
            let left = u32_sequence(
                self,
                row.left_covered_leg_sequence_id,
                "contact-orbit left covered legs",
            )?;
            let right = u32_sequence(
                self,
                row.right_covered_leg_sequence_id,
                "contact-orbit right covered legs",
            )?;
            if left.is_empty()
                || right.is_empty()
                || left.iter().chain(right).any(|leg| *leg as usize >= arity)
            {
                return Err(invalid(format!(
                    "contact-orbit step {index} has invalid covered legs"
                )));
            }
            let covered = left.iter().chain(right).copied().collect::<BTreeSet<_>>();
            if covered.len() != left.len() + right.len() {
                return Err(invalid(format!(
                    "contact-orbit step {index} repeats a covered leg"
                )));
            }
            let sources = i32_sequence(
                self,
                row.source_particle_leg_sequence_id,
                "contact-orbit source-particle legs",
            )?;
            if sources.len() != 3 || sources.iter().any(|leg| *leg < -1 || *leg >= arity as i32) {
                return Err(invalid(format!(
                    "contact-orbit step {index} has invalid source-particle legs"
                )));
            }
            let reconstruction = required_factor(
                &catalogs.factors,
                row.reconstruction_factor_id,
                "contact-orbit step reconstruction factor",
            )?;
            if *reconstruction != ExactComplexRational::ONE {
                return Err(invalid(format!(
                    "contact-orbit step {index} reconstruction factor is not one"
                )));
            }
            match stage {
                ContactOrbitStage::Partial => {
                    if left.len() != 1
                        || right.len() != 1
                        || sources != [left[0] as i32, right[0] as i32, -1]
                        || covered.contains(&(row.result_leg as u32))
                    {
                        return Err(invalid(format!(
                            "contact-orbit partial step {index} has inconsistent lineage"
                        )));
                    }
                }
                ContactOrbitStage::Final => {
                    let expected_covered = (0u32..arity as u32)
                        .filter(|leg| *leg != row.result_leg as u32)
                        .collect::<BTreeSet<_>>();
                    let expected_sources = [
                        if left.len() > 1 { -1 } else { left[0] as i32 },
                        if right.len() > 1 { -1 } else { right[0] as i32 },
                        row.result_leg as i32,
                    ];
                    if covered != expected_covered
                        || left.len() > 2
                        || right.len() > 2
                        || sources != expected_sources
                    {
                        return Err(invalid(format!(
                            "contact-orbit final step {index} has inconsistent lineage"
                        )));
                    }
                }
            }
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "contact-orbit step semantic digest",
            )?;
        }
        Ok(())
    }

    fn validate_color_contractions(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "color contraction",
            self.color_contractions.iter().map(|row| row.id),
            self.color_contractions
                .iter()
                .map(|row| row.template_string_id),
        )?;
        let ranges: Vec<_> = self
            .color_contractions
            .iter()
            .map(|row| CheckedTableRange::new(row.nc_term_start, row.nc_term_count))
            .collect();
        validate_packed_ranges("color Nc polynomial", &ranges, self.color_nc_terms.len())?;
        let witness_ranges: Vec<_> = self
            .color_contractions
            .iter()
            .map(|row| CheckedTableRange::new(row.witness_start, row.witness_count))
            .collect();
        validate_packed_ranges(
            "LC color transition witnesses",
            &witness_ranges,
            self.lc_color_transition_witnesses.len(),
        )?;
        for (index, row) in self.color_contractions.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::ColorContraction,
                "color contraction",
            )?;
            register_semantic_digest(
                semantic_digests,
                row.semantic_digest_id,
                "color contraction",
            )?;
            required_string(&catalogs.strings, row.template_string_id, "color template")?;
            required_string(
                &catalogs.strings,
                row.rule_kind_string_id,
                "color rule kind",
            )?;
            let input_representations = i32_sequence(
                self,
                row.input_representation_sequence_id,
                "color input representations",
            )?;
            validate_color_contract_shape(
                index,
                input_representations,
                row.has_output_representation,
                row.output_representation,
                row.ordered_open_string_arity,
            )?;
            let witnesses = &self.lc_color_transition_witnesses[witness_ranges[index]
                .as_usize_range(
                    self.lc_color_transition_witnesses.len(),
                    "LC color transition witnesses",
                )?];
            if witnesses.is_empty() {
                return Err(invalid(format!(
                    "color contraction {index} has no executable LC transition witness"
                )));
            }
            let has_output = row.has_output_representation == 1;
            for (ordinal, witness) in witnesses.iter().enumerate() {
                if witness.color_contraction_id as usize != index {
                    return Err(invalid(format!(
                        "LC color witness names contraction {}, expected {index}",
                        witness.color_contraction_id
                    )));
                }
                if witness.ordinal as usize != ordinal {
                    return Err(invalid(format!(
                        "LC color witness for contraction {index} has ordinal {}, expected {ordinal}",
                        witness.ordinal
                    )));
                }
                let left_shape = required_string(
                    &catalogs.strings,
                    witness.left_shape_string_id,
                    "LC color witness left shape",
                )?;
                let right_shape = required_string(
                    &catalogs.strings,
                    witness.right_shape_string_id,
                    "LC color witness right shape",
                )?;
                validate_lc_color_shape_name(left_shape, "LC color witness left shape")?;
                validate_lc_color_shape_name(right_shape, "LC color witness right shape")?;
                if input_representations.len() != 2 {
                    return Err(invalid(format!(
                        "LC color witness for contraction {index} requires exactly two input representations"
                    )));
                }
                match witness.input_permutation {
                    0 | 1 => {}
                    value => {
                        return Err(invalid(format!(
                            "LC color witness input permutation must be 0 or 1, found {value}"
                        )));
                    }
                }
                for (shape, input) in [left_shape, right_shape].into_iter().zip([0usize, 1usize]) {
                    validate_lc_color_shape(
                        shape,
                        input_representations[input],
                        "LC color witness input",
                    )?;
                }
                if witness.reverse_parent_mask > 3 {
                    return Err(invalid(format!(
                        "LC color witness reverse-parent mask {} exceeds two inputs",
                        witness.reverse_parent_mask
                    )));
                }
                let is_join = witness.component_operation == 0;
                let is_close = witness.component_operation == 5;
                if witness.component_operation > 5 {
                    return Err(invalid(format!(
                        "LC color witness operation {} is not supported",
                        witness.component_operation
                    )));
                }
                let component_kind = if witness.result_component_kind == u8::MAX {
                    None
                } else {
                    Some(LCColorComponentKind::try_from(
                        witness.result_component_kind,
                    )?)
                };
                let component_role = LCColorComponentRole::try_from(witness.result_component_role)?;
                if is_join {
                    if component_kind.is_none() || component_role == LCColorComponentRole::None {
                        return Err(invalid(
                            "LC color join witness requires component kind and role",
                        ));
                    }
                    let expected_role = if row.output_representation.abs() == 1 {
                        LCColorComponentRole::Passive
                    } else {
                        LCColorComponentRole::Active
                    };
                    if component_role != expected_role {
                        return Err(invalid(
                            "LC color join role does not match its output representation",
                        ));
                    }
                } else if is_close {
                    if component_role != LCColorComponentRole::None {
                        return Err(invalid("LC color closure result role must be none"));
                    }
                    let expected_kind = match input_representations {
                        [left, right] if left.abs() == 1 && right.abs() == 1 => None,
                        [left, right]
                            if (*left == 3 && *right == -3) || (*left == -3 && *right == 3) =>
                        {
                            Some(LCColorComponentKind::OpenString)
                        }
                        [8, 8] => Some(LCColorComponentKind::Trace),
                        _ => {
                            return Err(invalid(
                                "LC color closure has unsupported input representations",
                            ));
                        }
                    };
                    if component_kind != expected_kind {
                        return Err(invalid(
                            "LC color closure component kind does not match its inputs",
                        ));
                    }
                } else {
                    if component_kind.is_some() {
                        return Err(invalid(
                            "only LC color join/closure witnesses may declare a component kind",
                        ));
                    }
                    let expected_role = if matches!(witness.component_operation, 2 | 3) {
                        LCColorComponentRole::Active
                    } else {
                        LCColorComponentRole::None
                    };
                    if component_role != expected_role {
                        return Err(invalid(
                            "LC color witness result role does not match its operation",
                        ));
                    }
                }
                if is_close {
                    if has_output || witness.result_shape_string_id != MISSING_U32 {
                        return Err(invalid(
                            "LC color closure witness must have neither an output representation nor a result shape",
                        ));
                    }
                } else {
                    if !has_output {
                        return Err(invalid(
                            "non-closure LC color witness requires an output representation",
                        ));
                    }
                    let result_shape = required_string(
                        &catalogs.strings,
                        witness.result_shape_string_id,
                        "LC color witness result shape",
                    )?;
                    validate_lc_color_shape(
                        result_shape,
                        row.output_representation,
                        "LC color witness result",
                    )?;
                }
                let factor = required_factor(
                    &catalogs.factors,
                    witness.exact_factor_id,
                    "LC color witness factor",
                )?;
                if *factor == ExactComplexRational::ZERO {
                    return Err(invalid("LC color witness factor must be nonzero"));
                }
                required_digest(
                    &catalogs.digests,
                    witness.proof_digest_id,
                    "LC color witness proof",
                )?;
                let input_port_pairings = u32_sequence(
                    self,
                    witness.input_port_pairing_sequence_id,
                    "LC color witness input-port pairings",
                )?;
                if input_port_pairings.len() % 4 != 0 {
                    return Err(invalid(
                        "LC color witness input-port pairings are not divisible by four",
                    ));
                }
                let result_port_bindings = u32_sequence(
                    self,
                    witness.result_port_binding_sequence_id,
                    "LC color witness result-port bindings",
                )?;
                if result_port_bindings.len() % 2 != 0 {
                    return Err(invalid(
                        "LC color witness result-port bindings are not divisible by two",
                    ));
                }
                let mut consumed_ports = BTreeSet::new();
                for port in input_port_pairings
                    .chunks_exact(2)
                    .chain(result_port_bindings.chunks_exact(2))
                {
                    let parent = port[0] as usize;
                    let local = port[1] as usize;
                    let representation = *input_representations.get(parent).ok_or_else(|| {
                        invalid("LC color witness port references an absent parent")
                    })?;
                    let port_count = match representation {
                        1 => 0,
                        3 | -3 => 1,
                        8 => 2,
                        _ => {
                            return Err(invalid(
                                "LC color witness port uses an unsupported representation",
                            ));
                        }
                    };
                    if local >= port_count {
                        return Err(invalid(
                            "LC color witness port references an absent local port",
                        ));
                    }
                    if !consumed_ports.insert((parent, local)) {
                        return Err(invalid(
                            "LC color witness consumes one parent port more than once",
                        ));
                    }
                }
                let expected_port_count = input_representations
                    .iter()
                    .map(|representation| match representation {
                        1 => Ok(0_usize),
                        3 | -3 => Ok(1),
                        8 => Ok(2),
                        _ => Err(invalid(
                            "LC color witness uses an unsupported representation",
                        )),
                    })
                    .sum::<RusticolResult<usize>>()?;
                if consumed_ports.len() != expected_port_count {
                    return Err(invalid(
                        "LC color witness does not consume every parent port exactly once",
                    ));
                }
                let provenance = u32_sequence(
                    self,
                    witness.provenance_sequence_id,
                    "LC color witness provenance",
                )?;
                validate_string_pair_sequence(
                    &catalogs.strings,
                    provenance,
                    "LC color witness provenance",
                )?;
            }
            required_factor(
                &catalogs.factors,
                row.exact_coefficient_factor_id,
                "color exact coefficient",
            )?;
            required_digest(
                &catalogs.digests,
                row.expression_digest_id,
                "color expression",
            )?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "color semantic digest",
            )?;
            let terms = &self.color_nc_terms
                [ranges[index].as_usize_range(self.color_nc_terms.len(), "color Nc terms")?];
            let mut previous_exponent = None;
            for term in terms {
                if term.color_contraction_id as usize != index {
                    return Err(invalid(format!(
                        "color Nc term names contraction {}, expected {index}",
                        term.color_contraction_id
                    )));
                }
                if term.exponent < 0 {
                    return Err(invalid(format!(
                        "color contraction {index} has a negative Nc exponent"
                    )));
                }
                if let Some(previous_exponent) = previous_exponent
                    && previous_exponent >= term.exponent
                {
                    return Err(invalid(format!(
                        "color contraction {index} Nc powers are not strictly ordered"
                    )));
                }
                let coefficient =
                    required_factor(&catalogs.factors, term.factor_id, "color Nc coefficient")?;
                if *coefficient == ExactComplexRational::ZERO {
                    return Err(invalid(format!(
                        "color contraction {index} retains an exact-zero Nc term"
                    )));
                }
                previous_exponent = Some(term.exponent);
            }
        }
        Ok(())
    }

    fn validate_symmetry_proofs_basic(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "symmetry proof",
            self.symmetry_proofs.iter().map(|row| row.id),
            self.symmetry_proofs
                .iter()
                .map(|row| row.template_string_id),
        )?;
        for (index, row) in self.symmetry_proofs.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::SymmetryProof,
                "symmetry proof",
            )?;
            register_semantic_digest(semantic_digests, row.semantic_digest_id, "symmetry proof")?;
            required_string(&catalogs.strings, row.template_string_id, "proof template")?;
            let algorithm = required_string(
                &catalogs.strings,
                row.proof_algorithm_string_id,
                "proof algorithm",
            )?;
            if !is_supported_proof_algorithm(algorithm) {
                return Err(invalid(format!(
                    "symmetry proof {index} uses unsupported algorithm {algorithm:?}"
                )));
            }
            let subjects = u32_sequence(self, row.subject_template_sequence_id, "proof subjects")?;
            if subjects.is_empty() {
                return Err(invalid(format!("symmetry proof {index} has no subjects")));
            }
            for subject in subjects {
                required_string(&catalogs.strings, *subject, "proof subject template")?;
            }
            validate_permutation(
                u32_sequence(
                    self,
                    row.input_permutation_sequence_id,
                    "proof input permutation",
                )?,
                "proof input permutation",
            )?;
            let phase = required_factor(
                &catalogs.factors,
                row.exact_phase_factor_id,
                "proof exact phase",
            )?;
            if *phase == ExactComplexRational::ZERO {
                return Err(invalid(format!("symmetry proof {index} has zero phase")));
            }
            let expressions = u32_sequence(
                self,
                row.expression_digest_sequence_id,
                "proof expression digests",
            )?;
            if expressions.is_empty() {
                return Err(invalid(format!(
                    "symmetry proof {index} has no expression digests"
                )));
            }
            validate_u32_references(
                expressions,
                catalogs.digests.len(),
                "proof expression digests",
            )?;
            required_digest(&catalogs.digests, row.witness_digest_id, "proof witness")?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "proof semantic digest",
            )?;
        }
        Ok(())
    }

    fn validate_transitions_basic(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "transition",
            self.transitions.iter().map(|row| row.id),
            self.transitions.iter().map(|row| row.template_string_id),
        )?;
        for (index, row) in self.transitions.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::Transition,
                "transition",
            )?;
            register_semantic_digest(semantic_digests, row.semantic_digest_id, "transition")?;
            required_string(
                &catalogs.strings,
                row.template_string_id,
                "transition template",
            )?;
            let inputs =
                u32_sequence(self, row.input_state_sequence_id, "transition input states")?;
            if inputs.is_empty() {
                return Err(invalid(format!("transition {index} has no inputs")));
            }
            validate_u32_references(inputs, self.current_states.len(), "transition input states")?;
            required_reference(
                row.result_state_template_id,
                self.current_states.len(),
                "transition result state",
            )?;
            required_reference(
                row.quantum_flow_template_id,
                self.quantum_flows.len(),
                "transition quantum flow",
            )?;
            required_reference(
                row.evaluator_binding_id,
                self.evaluator_bindings.len(),
                "transition evaluator",
            )?;
            let permutation = u32_sequence(
                self,
                row.canonical_input_order_sequence_id,
                "transition canonical input order",
            )?;
            validate_permutation_of(
                permutation,
                inputs.len(),
                "transition canonical input order",
            )?;
            let momentum = u32_sequence(
                self,
                row.momentum_convention_sequence_id,
                "transition momentum convention",
            )?;
            if momentum.len() != inputs.len() {
                return Err(invalid(format!(
                    "transition {index} momentum convention does not cover every input"
                )));
            }
            for string_id in momentum {
                required_string(
                    &catalogs.strings,
                    *string_id,
                    "transition momentum convention",
                )?;
            }
            let parameters = u32_sequence(
                self,
                row.coupling_parameter_sequence_id,
                "transition coupling parameters",
            )?;
            validate_strict_u32(parameters, "transition coupling parameters")?;
            validate_u32_references(
                parameters,
                self.parameters.len(),
                "transition coupling parameters",
            )?;
            required_reference(
                row.coupling_order_set_id,
                self.coupling_order_ranges.len(),
                "transition coupling-order set",
            )?;
            required_reference(
                row.color_contraction_template_id,
                self.color_contractions.len(),
                "transition color contraction",
            )?;
            required_factor(
                &catalogs.factors,
                row.binding_coupling_factor_id,
                "transition binding coupling",
            )?;
            required_factor(
                &catalogs.factors,
                row.exact_factor_id,
                "transition exact factor",
            )?;
            OutputFactorSource::try_from(row.output_factor_source)?;
            required_string(
                &catalogs.strings,
                row.equivalence_class_string_id,
                "transition equivalence class",
            )?;
            if let Some(factor) = optional_factor(
                &catalogs.factors,
                row.input_exchange_factor_id,
                "transition input-exchange factor",
            )? && *factor == ExactComplexRational::ZERO
            {
                return Err(invalid(format!(
                    "transition {index} has a zero input-exchange factor"
                )));
            }
            required_string(
                &catalogs.strings,
                row.output_projection_string_id,
                "transition output projection",
            )?;
            let contact_orbit_steps = u32_sequence(
                self,
                row.contact_orbit_step_sequence_id,
                "transition contact-orbit steps",
            )?;
            let contact_orbit_step_digests = u32_sequence(
                self,
                row.contact_orbit_step_semantic_digest_sequence_id,
                "transition contact-orbit step semantic digests",
            )?;
            if contact_orbit_steps.len() != contact_orbit_step_digests.len() {
                return Err(invalid(format!(
                    "transition {index} has incomplete contact-orbit step bindings"
                )));
            }
            if contact_orbit_steps
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(invalid(format!(
                    "transition {index} contact-orbit steps are not canonical"
                )));
            }
            for (&step_id, &step_digest_id) in
                contact_orbit_steps.iter().zip(contact_orbit_step_digests)
            {
                let step_index = required_reference(
                    step_id,
                    self.contact_orbit_steps.len(),
                    "transition contact-orbit step",
                )?;
                let expected_digest_id = self.contact_orbit_steps[step_index].semantic_digest_id;
                if step_digest_id != expected_digest_id {
                    return Err(invalid(format!(
                        "transition {index} contact-orbit step digest does not match its record"
                    )));
                }
                required_digest(
                    &catalogs.digests,
                    step_digest_id,
                    "transition contact-orbit step semantic digest",
                )?;
            }
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "transition semantic digest",
            )?;

            let flow = self.quantum_flows[row.quantum_flow_template_id as usize];
            let flow_inputs = u32_sequence(
                self,
                flow.input_state_sequence_id,
                "quantum-flow input states",
            )?;
            if inputs != flow_inputs
                || row.result_state_template_id != flow.result_state_template_id
                || row.coupling_order_set_id != flow.coupling_order_set_id
            {
                return Err(invalid(format!(
                    "transition {index} and quantum-flow state/coupling contracts do not match"
                )));
            }
        }
        Ok(())
    }

    fn validate_propagators_basic(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "propagator",
            self.propagators.iter().map(|row| row.id),
            self.propagators.iter().map(|row| row.template_string_id),
        )?;
        let mut state_owners = vec![None; self.current_states.len()];
        for (index, row) in self.propagators.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::Propagator,
                "propagator",
            )?;
            register_semantic_digest(semantic_digests, row.semantic_digest_id, "propagator")?;
            required_string(
                &catalogs.strings,
                row.template_string_id,
                "propagator template",
            )?;
            let state = required_reference(
                row.state_template_id,
                self.current_states.len(),
                "propagator state",
            )?;
            if let Some(previous) = state_owners[state].replace(index) {
                return Err(invalid(format!(
                    "current state {state} has propagator rows {previous} and {index}"
                )));
            }
            let active = validate_bool(row.applies_propagator, "applies_propagator")?;
            let evaluator = optional_reference(
                row.evaluator_binding_id,
                self.evaluator_bindings.len(),
                "propagator evaluator",
            )?;
            let numerator = optional_digest(
                &catalogs.digests,
                row.numerator_expression_digest_id,
                "propagator numerator",
            )?;
            let denominator = optional_digest(
                &catalogs.digests,
                row.denominator_expression_digest_id,
                "propagator denominator",
            )?;
            if active {
                if evaluator.is_none() || numerator.is_none() || denominator.is_none() {
                    return Err(invalid(format!(
                        "active propagator {index} requires evaluator, numerator, and denominator"
                    )));
                }
            } else if evaluator.is_some() || numerator.is_some() || denominator.is_some() {
                return Err(invalid(format!(
                    "identity propagator {index} carries evaluator or expression metadata"
                )));
            }
            optional_reference(
                row.mass_parameter_id,
                self.parameters.len(),
                "propagator mass parameter",
            )?;
            optional_reference(
                row.width_parameter_id,
                self.parameters.len(),
                "propagator width parameter",
            )?;
            optional_string(&catalogs.strings, row.gauge_string_id, "propagator gauge")?;
            optional_reference(
                row.linearity_proof_template_id,
                self.symmetry_proofs.len(),
                "propagator linearity proof",
            )?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "propagator semantic digest",
            )?;
        }
        if let Some((state, _)) = state_owners
            .iter()
            .enumerate()
            .find(|(_, owner)| owner.is_none())
        {
            return Err(invalid(format!(
                "current state {state} has no active or identity propagator"
            )));
        }
        Ok(())
    }

    fn validate_closures_basic(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &mut BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<()> {
        validate_record_ids(
            "closure",
            self.closures.iter().map(|row| row.id),
            self.closures.iter().map(|row| row.template_string_id),
        )?;
        for (index, row) in self.closures.iter().enumerate() {
            register_template(
                template_kinds,
                row.template_string_id,
                TemplateKind::Closure,
                "closure",
            )?;
            register_semantic_digest(semantic_digests, row.semantic_digest_id, "closure")?;
            required_string(
                &catalogs.strings,
                row.template_string_id,
                "closure template",
            )?;
            let inputs = u32_sequence(self, row.input_state_sequence_id, "closure input states")?;
            if inputs.is_empty() {
                return Err(invalid(format!("closure {index} has no inputs")));
            }
            validate_u32_references(inputs, self.current_states.len(), "closure input states")?;
            let result_state = optional_reference(
                row.result_state_template_id,
                self.current_states.len(),
                "closure result state",
            )?;
            required_reference(
                row.evaluator_binding_id,
                self.evaluator_bindings.len(),
                "closure evaluator",
            )?;
            validate_permutation_of(
                u32_sequence(
                    self,
                    row.canonical_input_order_sequence_id,
                    "closure canonical input order",
                )?,
                inputs.len(),
                "closure canonical input order",
            )?;
            let parameters = u32_sequence(
                self,
                row.coupling_parameter_sequence_id,
                "closure coupling parameters",
            )?;
            validate_strict_u32(parameters, "closure coupling parameters")?;
            validate_u32_references(
                parameters,
                self.parameters.len(),
                "closure coupling parameters",
            )?;
            required_reference(
                row.coupling_order_set_id,
                self.coupling_order_ranges.len(),
                "closure coupling-order set",
            )?;
            let eligible_flows = u32_sequence(
                self,
                row.eligible_quantum_flow_sequence_id,
                "closure eligible quantum flows",
            )?;
            validate_strict_u32(eligible_flows, "closure eligible quantum flows")?;
            validate_u32_references(
                eligible_flows,
                self.quantum_flows.len(),
                "closure eligible quantum flows",
            )?;
            let evaluator = &self.evaluator_bindings[row.evaluator_binding_id as usize];
            match EvaluatorCallableKind::try_from(evaluator.callable_kind)? {
                EvaluatorCallableKind::PreparedKernel if eligible_flows.is_empty() => {
                    return Err(invalid(format!(
                        "prepared closure {index} has no eligible quantum-flow witness"
                    )));
                }
                EvaluatorCallableKind::RusticolTemplate if !eligible_flows.is_empty() => {
                    return Err(invalid(format!(
                        "direct Rusticol closure {index} carries prepared quantum-flow witnesses"
                    )));
                }
                EvaluatorCallableKind::PreparedKernel if result_state.is_none() => {
                    return Err(invalid(format!(
                        "prepared closure {index} has no result-state contract"
                    )));
                }
                EvaluatorCallableKind::RusticolTemplate if result_state.is_some() => {
                    return Err(invalid(format!(
                        "direct Rusticol closure {index} carries a result-state contract"
                    )));
                }
                _ => {}
            }
            for flow_id in eligible_flows {
                let flow = &self.quantum_flows[*flow_id as usize];
                let flow_inputs = u32_sequence(
                    self,
                    flow.input_state_sequence_id,
                    "closure quantum-flow input states",
                )?;
                if flow_inputs != inputs {
                    return Err(invalid(format!(
                        "closure {index} and eligible quantum flow {flow_id} have different input-state contracts"
                    )));
                }
                if Some(flow.result_state_template_id as usize) != result_state {
                    return Err(invalid(format!(
                        "closure {index} and eligible quantum flow {flow_id} have different result-state contracts"
                    )));
                }
                if flow.coupling_order_set_id != row.coupling_order_set_id {
                    return Err(invalid(format!(
                        "closure {index} and eligible quantum flow {flow_id} have different coupling-order contracts"
                    )));
                }
            }
            required_reference(
                row.color_contraction_template_id,
                self.color_contractions.len(),
                "closure color contraction",
            )?;
            required_factor(
                &catalogs.factors,
                row.binding_coupling_factor_id,
                "closure binding coupling",
            )?;
            required_factor(
                &catalogs.factors,
                row.exact_factor_id,
                "closure exact factor",
            )?;
            OutputFactorSource::try_from(row.output_factor_source)?;
            required_string(
                &catalogs.strings,
                row.equivalence_class_string_id,
                "closure equivalence class",
            )?;
            if let Some(factor) = optional_factor(
                &catalogs.factors,
                row.input_exchange_factor_id,
                "closure input-exchange factor",
            )? && *factor == ExactComplexRational::ZERO
            {
                return Err(invalid(format!(
                    "closure {index} has a zero input-exchange factor"
                )));
            }
            required_string(
                &catalogs.strings,
                row.projection_string_id,
                "closure projection",
            )?;
            let coefficients = u32_sequence(
                self,
                row.component_coefficient_sequence_id,
                "closure component coefficients",
            )?;
            validate_u32_references(
                coefficients,
                catalogs.factors.len(),
                "closure component coefficients",
            )?;
            let chirality = required_string(
                &catalogs.strings,
                row.chirality_relation_string_id,
                "closure chirality relation",
            )?;
            if !matches!(chirality, "any" | "equal" | "opposite") {
                return Err(invalid(format!(
                    "closure {index} has unsupported chirality relation {chirality:?}"
                )));
            }
            optional_string(
                &catalogs.strings,
                row.metric_signature_string_id,
                "closure metric signature",
            )?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "closure semantic digest",
            )?;
        }
        Ok(())
    }

    fn validate_evaluators(
        self,
        catalogs: &ValidatedCatalogs<'_>,
        template_kinds: &BTreeMap<u32, TemplateKind>,
        semantic_digests: &mut BTreeSet<u32>,
    ) -> RusticolResult<u32> {
        validate_record_ids(
            "evaluator binding",
            self.evaluator_bindings.iter().map(|row| row.id),
            self.evaluator_bindings
                .iter()
                .map(|row| row.resolver_key_string_id),
        )?;
        let mut prepared_contracts = BTreeMap::new();
        let mut runtime_contracts = BTreeMap::new();
        let mut semantic_owners = BTreeMap::new();
        let mut prepared_ids = BTreeSet::new();
        for (index, row) in self.evaluator_bindings.iter().enumerate() {
            register_semantic_digest(
                semantic_digests,
                row.semantic_digest_id,
                "evaluator binding",
            )?;
            required_string(
                &catalogs.strings,
                row.resolver_key_string_id,
                "evaluator resolver key",
            )?;
            let contract_kind = EvaluatorContractKind::try_from(row.contract_kind)?;
            let callable_kind = EvaluatorCallableKind::try_from(row.callable_kind)?;
            required_digest(
                &catalogs.digests,
                row.callable_signature_digest_id,
                "evaluator callable signature",
            )?;
            required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "evaluator semantic digest",
            )?;
            let inputs = u32_sequence(self, row.input_state_sequence_id, "evaluator input states")?;
            validate_u32_references(inputs, self.current_states.len(), "evaluator input states")?;
            let output = optional_reference(
                row.output_state_template_id,
                self.current_states.len(),
                "evaluator output state",
            )?;
            let input_layout =
                u32_sequence(self, row.input_layout_sequence_id, "evaluator input layout")?;
            let output_layout = u32_sequence(
                self,
                row.output_layout_sequence_id,
                "evaluator output layout",
            )?;
            let exact_expressions = u32_sequence(
                self,
                row.exact_expression_digest_sequence_id,
                "evaluator exact expressions",
            )?;
            if input_layout.is_empty() || output_layout.is_empty() || exact_expressions.is_empty() {
                return Err(invalid(format!(
                    "evaluator binding {index} has an empty layout or expression contract"
                )));
            }
            validate_u32_references(
                input_layout,
                catalogs.strings.len(),
                "evaluator input layout",
            )?;
            validate_u32_references(
                output_layout,
                catalogs.strings.len(),
                "evaluator output layout",
            )?;
            validate_u32_references(
                exact_expressions,
                catalogs.digests.len(),
                "evaluator exact expressions",
            )?;
            if output_layout.len() != exact_expressions.len() {
                return Err(invalid(format!(
                    "evaluator binding {index} output layout and exact expressions do not align"
                )));
            }
            if let Some(output) = output
                && output_layout.len() != self.current_states[output].dimension as usize
            {
                return Err(invalid(format!(
                    "evaluator binding {index} output layout does not match state dimension"
                )));
            }
            match contract_kind {
                EvaluatorContractKind::Source
                | EvaluatorContractKind::Vertex
                | EvaluatorContractKind::Propagator => {
                    if output.is_none() {
                        return Err(invalid(format!(
                            "evaluator binding {index} requires an output state"
                        )));
                    }
                }
                EvaluatorContractKind::Closure | EvaluatorContractKind::ModelParameter => {
                    if output.is_some() {
                        return Err(invalid(format!(
                            "evaluator binding {index} cannot produce a current state"
                        )));
                    }
                }
            }
            match callable_kind {
                EvaluatorCallableKind::PreparedKernel => {
                    if row.prepared_kernel_id == MISSING_U32
                        || row.runtime_template_string_id != MISSING_U32
                    {
                        return Err(invalid(format!(
                            "prepared-kernel evaluator binding {index} has invalid callable identity"
                        )));
                    }
                    prepared_ids.insert(row.prepared_kernel_id);
                }
                EvaluatorCallableKind::RusticolTemplate => {
                    if row.prepared_kernel_id != MISSING_U32
                        || row.runtime_template_string_id == MISSING_U32
                    {
                        return Err(invalid(format!(
                            "Rusticol-template evaluator binding {index} has invalid callable identity"
                        )));
                    }
                    required_string(
                        &catalogs.strings,
                        row.runtime_template_string_id,
                        "evaluator runtime template",
                    )?;
                    validate_rusticol_runtime_template(catalogs, row, contract_kind, index)?;
                }
            }

            let semantic_templates = u32_sequence(
                self,
                row.semantic_template_sequence_id,
                "evaluator semantic templates",
            )?;
            if semantic_templates.is_empty() {
                return Err(invalid(format!(
                    "evaluator binding {index} has no semantic templates"
                )));
            }
            validate_strict_u32(semantic_templates, "evaluator semantic templates")?;
            for template in semantic_templates {
                let template_kind = template_kinds.get(template).ok_or_else(|| {
                    invalid(format!(
                        "evaluator binding {index} references unknown semantic template string id {template}"
                    ))
                })?;
                if template_kind.evaluator_contract() != Some(contract_kind) {
                    return Err(invalid(format!(
                        "evaluator binding {index} contract kind does not match semantic template {template}"
                    )));
                }
                if let Some(previous) = semantic_owners.insert(*template, index)
                    && previous != index
                {
                    return Err(invalid(format!(
                        "semantic template {template} belongs to evaluator bindings {previous} and {index}"
                    )));
                }
            }

            let callable_contract = CallableContractKey {
                contract_kind,
                callable_signature_digest_id: row.callable_signature_digest_id,
                input_layout: input_layout.to_vec(),
                output_layout: output_layout.to_vec(),
                exact_expression_digests: exact_expressions.to_vec(),
            };
            let (owner, previous) = match callable_kind {
                EvaluatorCallableKind::PreparedKernel => (
                    "prepared kernel ID",
                    prepared_contracts.insert(row.prepared_kernel_id, callable_contract.clone()),
                ),
                EvaluatorCallableKind::RusticolTemplate => (
                    "Rusticol runtime template",
                    runtime_contracts
                        .insert(row.runtime_template_string_id, callable_contract.clone()),
                ),
            };
            if previous.is_some_and(|previous| previous != callable_contract) {
                return Err(invalid(format!(
                    "{owner} has inconsistent evaluator callable contracts"
                )));
            }
        }
        checked_len(prepared_ids.len(), "prepared kernels")
    }

    fn validate_proof_subjects(
        self,
        template_kinds: &BTreeMap<u32, TemplateKind>,
    ) -> RusticolResult<()> {
        for (index, proof) in self.symmetry_proofs.iter().enumerate() {
            let subjects =
                u32_sequence(self, proof.subject_template_sequence_id, "proof subjects")?;
            for subject in subjects {
                if !template_kinds.contains_key(subject) {
                    return Err(invalid(format!(
                        "symmetry proof {index} references unknown template string id {subject}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_evaluator_state_contracts(
        self,
        _template_kinds: &BTreeMap<u32, TemplateKind>,
    ) -> RusticolResult<()> {
        for source in self.sources {
            self.validate_evaluator_contract(
                source.evaluator_binding_id,
                EvaluatorContractKind::Source,
                &[],
                Some(source.state_template_id),
                source.template_string_id,
            )?;
        }
        for transition in self.transitions {
            self.validate_evaluator_contract(
                transition.evaluator_binding_id,
                EvaluatorContractKind::Vertex,
                u32_sequence(
                    self,
                    transition.input_state_sequence_id,
                    "transition states",
                )?,
                Some(transition.result_state_template_id),
                transition.template_string_id,
            )?;
        }
        for propagator in self.propagators {
            if propagator.evaluator_binding_id != MISSING_U32 {
                self.validate_evaluator_contract(
                    propagator.evaluator_binding_id,
                    EvaluatorContractKind::Propagator,
                    &[propagator.state_template_id],
                    Some(propagator.state_template_id),
                    propagator.template_string_id,
                )?;
            }
        }
        for closure in self.closures {
            self.validate_evaluator_contract(
                closure.evaluator_binding_id,
                EvaluatorContractKind::Closure,
                u32_sequence(self, closure.input_state_sequence_id, "closure states")?,
                None,
                closure.template_string_id,
            )?;
        }
        Ok(())
    }

    fn validate_evaluator_contract(
        self,
        evaluator_id: u32,
        expected_kind: EvaluatorContractKind,
        expected_inputs: &[u32],
        expected_output: Option<u32>,
        semantic_template_string_id: u32,
    ) -> RusticolResult<()> {
        let evaluator = self
            .evaluator_bindings
            .get(evaluator_id as usize)
            .ok_or_else(|| invalid(format!("unknown evaluator binding {evaluator_id}")))?;
        if EvaluatorContractKind::try_from(evaluator.contract_kind)? != expected_kind {
            return Err(invalid(format!(
                "semantic template string id {semantic_template_string_id} uses the wrong evaluator contract kind"
            )));
        }
        let inputs = u32_sequence(self, evaluator.input_state_sequence_id, "evaluator states")?;
        if inputs != expected_inputs {
            return Err(invalid(format!(
                "semantic template string id {semantic_template_string_id} evaluator input states do not match"
            )));
        }
        if optional_id(evaluator.output_state_template_id) != expected_output {
            return Err(invalid(format!(
                "semantic template string id {semantic_template_string_id} evaluator output state does not match"
            )));
        }
        let templates = u32_sequence(
            self,
            evaluator.semantic_template_sequence_id,
            "evaluator semantic templates",
        )?;
        if templates
            .binary_search(&semantic_template_string_id)
            .is_err()
        {
            return Err(invalid(format!(
                "semantic template string id {semantic_template_string_id} is absent from its evaluator binding"
            )));
        }
        Ok(())
    }
}

fn validate_rusticol_runtime_template(
    catalogs: &ValidatedCatalogs<'_>,
    row: &EvaluatorBindingRow,
    contract_kind: EvaluatorContractKind,
    index: usize,
) -> RusticolResult<()> {
    let runtime_template = required_string(
        &catalogs.strings,
        row.runtime_template_string_id,
        "evaluator runtime template",
    )?;
    let signature = required_digest(
        &catalogs.digests,
        row.callable_signature_digest_id,
        "evaluator callable signature",
    )?
    .to_string();
    let signature_suffix = &signature[..24];
    match contract_kind {
        EvaluatorContractKind::Source => {
            const PREFIX: &str = "rusticol.source-fill.";
            let suffix = format!(".v1:{signature_suffix}");
            let family = runtime_template
                .strip_prefix(PREFIX)
                .and_then(|value| value.strip_suffix(&suffix))
                .ok_or_else(|| {
                    invalid(format!(
                        "Rusticol source evaluator binding {index} has an unauthenticated runtime template"
                    ))
                })?;
            if !matches!(
                family,
                "scalar" | "fermion" | "vector" | "spin2" | "ghost" | "auxiliary"
            ) {
                return Err(invalid(format!(
                    "Rusticol source evaluator binding {index} has unsupported family {family:?}"
                )));
            }
        }
        EvaluatorContractKind::Closure => {
            let expected = format!("rusticol.closure-reduce.v1:{signature_suffix}");
            if runtime_template != expected {
                return Err(invalid(format!(
                    "Rusticol closure evaluator binding {index} has an unauthenticated runtime template"
                )));
            }
        }
        _ => {
            return Err(invalid(format!(
                "Rusticol runtime templates are not registered for evaluator binding {index} contract {contract_kind:?}"
            )));
        }
    }
    Ok(())
}

fn validate_indexed_ranges(
    label: &str,
    ranges: &[IndexedRangeRow],
    value_len: usize,
) -> RusticolResult<()> {
    validate_canonical_ids(label, ranges.iter().map(|row| row.id))?;
    let checked: Vec<_> = ranges.iter().map(|row| row.range).collect();
    validate_packed_ranges(label, &checked, value_len)
}

fn validate_sequence_catalog<T: Ord>(
    label: &str,
    ranges: &[IndexedRangeRow],
    values: &[T],
    require_nonempty: bool,
) -> RusticolResult<()> {
    validate_indexed_ranges(label, ranges, values.len())?;
    let mut previous: Option<&[T]> = None;
    for (index, row) in ranges.iter().enumerate() {
        let sequence = &values[row.range.as_usize_range(values.len(), label)?];
        if require_nonempty && sequence.is_empty() {
            return Err(invalid(format!("{label} row {index} is empty")));
        }
        if let Some(previous) = previous
            && previous >= sequence
        {
            return Err(invalid(format!(
                "{label} is not in strict canonical order at row {index}"
            )));
        }
        previous = Some(sequence);
    }
    Ok(())
}

fn validate_canonical_ids(label: &str, ids: impl IntoIterator<Item = u32>) -> RusticolResult<()> {
    for (index, id) in ids.into_iter().enumerate() {
        if usize::try_from(id).ok() != Some(index) {
            return Err(invalid(format!(
                "{label} row {index} has noncanonical id {id}"
            )));
        }
    }
    Ok(())
}

fn validate_record_ids(
    label: &str,
    ids: impl IntoIterator<Item = u32>,
    identity_string_ids: impl IntoIterator<Item = u32>,
) -> RusticolResult<()> {
    validate_canonical_ids(label, ids)?;
    let mut previous = None;
    for (index, id) in identity_string_ids.into_iter().enumerate() {
        if let Some(previous) = previous
            && previous >= id
        {
            return Err(invalid(format!(
                "{label} records are not in strict semantic-identity order at row {index}"
            )));
        }
        previous = Some(id);
    }
    Ok(())
}

fn validate_bool(value: u8, label: &str) -> RusticolResult<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid(format!(
            "{label} must be encoded as 0 or 1, found {value}"
        ))),
    }
}

fn validate_color_contract_shape(
    row: usize,
    input_representations: &[i32],
    has_output_representation: u8,
    output_representation: i32,
    ordered_open_string_arity: u32,
) -> RusticolResult<()> {
    let has_output = validate_bool(
        has_output_representation,
        "color output-representation presence",
    )?;
    if !has_output && output_representation != 0 {
        return Err(invalid(format!(
            "color contraction {row} has no output representation but carries nonzero payload {output_representation}"
        )));
    }

    // Every ordered open string has two endpoints. This slot-count bound is
    // representation independent and therefore also applies to external
    // models using an otherwise unknown representation encoding.
    let endpoint_slots = input_representations
        .len()
        .checked_add(usize::from(has_output))
        .ok_or_else(|| invalid("color endpoint-slot count exceeds usize"))?;
    let arity = usize::try_from(ordered_open_string_arity)
        .map_err(|_| invalid("color open-string arity exceeds usize"))?;
    if arity > endpoint_slots / 2 {
        return Err(invalid(format!(
            "color contraction {row} declares {arity} ordered open strings but has only {endpoint_slots} endpoint slots"
        )));
    }

    let representations_are_standard = input_representations
        .iter()
        .copied()
        .chain(has_output.then_some(output_representation))
        .all(|value| matches!(value, -3 | 1 | 3 | 8));
    if representations_are_standard {
        let fundamental_endpoints = input_representations
            .iter()
            .copied()
            .chain(has_output.then_some(output_representation))
            .filter(|value| matches!(value, -3 | 3))
            .count();
        if arity > fundamental_endpoints / 2 {
            return Err(invalid(format!(
                "color contraction {row} declares {arity} ordered open strings but its standard representations provide only {fundamental_endpoints} fundamental endpoints"
            )));
        }
    }
    Ok(())
}

fn validate_lc_color_shape_name(value: &str, label: &str) -> RusticolResult<()> {
    if matches!(
        value,
        "singlet-forest"
            | "fundamental-open-string"
            | "antifundamental-open-string"
            | "adjoint-segment"
    ) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{label} has unsupported LC color shape {value:?}"
        )))
    }
}

fn validate_lc_color_shape(value: &str, representation: i32, label: &str) -> RusticolResult<()> {
    validate_lc_color_shape_name(value, label)?;
    let expected = match representation {
        1 => "singlet-forest",
        3 => "fundamental-open-string",
        -3 => "antifundamental-open-string",
        8 => "adjoint-segment",
        other => {
            return Err(invalid(format!(
                "{label} uses unsupported LC representation {other}"
            )));
        }
    };
    if value != expected {
        return Err(invalid(format!(
            "{label} maps LC representation {representation} to {value:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn checked_len(length: usize, label: &str) -> RusticolResult<u32> {
    u32::try_from(length).map_err(|_| invalid(format!("{label} count {length} exceeds u32")))
}

fn required_reference(value: u32, target_len: usize, label: &str) -> RusticolResult<usize> {
    let value =
        usize::try_from(value).map_err(|_| invalid(format!("{label} id {value} exceeds usize")))?;
    if value >= target_len {
        return Err(invalid(format!(
            "{label} id {value} exceeds target length {target_len}"
        )));
    }
    Ok(value)
}

fn optional_reference(value: u32, target_len: usize, label: &str) -> RusticolResult<Option<usize>> {
    if value == MISSING_U32 {
        Ok(None)
    } else {
        required_reference(value, target_len, label).map(Some)
    }
}

const fn optional_id(value: u32) -> Option<u32> {
    if value == MISSING_U32 {
        None
    } else {
        Some(value)
    }
}

fn required_string<'a>(strings: &[&'a str], id: u32, label: &str) -> RusticolResult<&'a str> {
    strings
        .get(required_reference(id, strings.len(), label)?)
        .copied()
        .ok_or_else(|| invalid(format!("{label} is absent")))
}

fn optional_string<'a>(
    strings: &[&'a str],
    id: u32,
    label: &str,
) -> RusticolResult<Option<&'a str>> {
    optional_reference(id, strings.len(), label)?
        .map(|index| {
            strings
                .get(index)
                .copied()
                .ok_or_else(|| invalid(format!("{label} is absent")))
        })
        .transpose()
}

fn require_string_value(
    strings: &[&str],
    id: u32,
    expected: &str,
    label: &str,
) -> RusticolResult<()> {
    let actual = required_string(strings, id, label)?;
    if actual != expected {
        return Err(RusticolError::compatibility(format!(
            "unsupported {label} {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_digest(
    digests: &[SemanticDigest],
    id: u32,
    label: &str,
) -> RusticolResult<SemanticDigest> {
    digests
        .get(required_reference(id, digests.len(), label)?)
        .copied()
        .ok_or_else(|| invalid(format!("{label} is absent")))
}

fn validate_string_pair_sequence(
    strings: &[&str],
    values: &[u32],
    label: &str,
) -> RusticolResult<()> {
    if !values.len().is_multiple_of(2) {
        return Err(invalid(format!("{label} must contain key/value pairs")));
    }
    let mut previous_key = None;
    for pair in values.chunks_exact(2) {
        let key = required_string(strings, pair[0], &format!("{label} key"))?;
        required_string(strings, pair[1], &format!("{label} value"))?;
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(invalid(format!("{label} keys must be sorted and unique")));
        }
        previous_key = Some(key);
    }
    Ok(())
}

fn optional_digest(
    digests: &[SemanticDigest],
    id: u32,
    label: &str,
) -> RusticolResult<Option<SemanticDigest>> {
    optional_reference(id, digests.len(), label)?
        .map(|index| {
            digests
                .get(index)
                .copied()
                .ok_or_else(|| invalid(format!("{label} is absent")))
        })
        .transpose()
}

fn required_factor<'a>(
    factors: &'a [ExactComplexRational],
    id: u32,
    label: &str,
) -> RusticolResult<&'a ExactComplexRational> {
    factors
        .get(required_reference(id, factors.len(), label)?)
        .ok_or_else(|| invalid(format!("{label} is absent")))
}

fn optional_factor<'a>(
    factors: &'a [ExactComplexRational],
    id: u32,
    label: &str,
) -> RusticolResult<Option<&'a ExactComplexRational>> {
    optional_reference(id, factors.len(), label)?
        .map(|index| {
            factors
                .get(index)
                .ok_or_else(|| invalid(format!("{label} is absent")))
        })
        .transpose()
}

fn sequence<'a, T>(
    ranges: &[IndexedRangeRow],
    values: &'a [T],
    id: u32,
    label: &str,
) -> RusticolResult<&'a [T]> {
    let row = ranges
        .get(required_reference(id, ranges.len(), label)?)
        .ok_or_else(|| invalid(format!("{label} range is absent")))?;
    let range = row.range.as_usize_range(values.len(), label)?;
    Ok(&values[range])
}

fn u32_sequence<'a>(
    input: RecurrenceTemplateInputView<'a>,
    id: u32,
    label: &str,
) -> RusticolResult<&'a [u32]> {
    sequence(
        input.u32_sequence_ranges,
        input.u32_sequence_values,
        id,
        label,
    )
}

fn i32_sequence<'a>(
    input: RecurrenceTemplateInputView<'a>,
    id: u32,
    label: &str,
) -> RusticolResult<&'a [i32]> {
    sequence(
        input.i32_sequence_ranges,
        input.i32_sequence_values,
        id,
        label,
    )
}

fn validate_flavour_flow_witness(
    input: RecurrenceTemplateInputView<'_>,
    row: usize,
    operation: &str,
    input_flavour_ids: &[u32],
    result_state_id: u32,
    result_flavour_id: u32,
) -> RusticolResult<()> {
    if input_flavour_ids.len() != 2 {
        return Err(invalid(format!(
            "quantum flow {row} must have two flavour-flow inputs"
        )));
    }
    let left = sequence(
        input.flavour_flow_ranges,
        input.flavour_flow_values,
        input_flavour_ids[0],
        "quantum-flow left flavour flow",
    )?;
    let right = sequence(
        input.flavour_flow_ranges,
        input.flavour_flow_values,
        input_flavour_ids[1],
        "quantum-flow right flavour flow",
    )?;
    let result = sequence(
        input.flavour_flow_ranges,
        input.flavour_flow_values,
        result_flavour_id,
        "quantum-flow result flavour flow",
    )?;
    let result_particle = input.current_states[result_state_id as usize].particle_id;
    let mut expected = Vec::with_capacity(left.len() + right.len() + 1);
    match operation {
        "constant-result" => expected.push(result_particle),
        "append-left-result" => {
            expected.extend_from_slice(left);
            if left.last().copied() != Some(result_particle) {
                expected.push(result_particle);
            }
        }
        "append-right-result" => {
            expected.extend_from_slice(right);
            if right.last().copied() != Some(result_particle) {
                expected.push(result_particle);
            }
        }
        "concat-left-right-result" => {
            expected.extend_from_slice(left);
            expected.extend_from_slice(right);
            expected.push(result_particle);
        }
        _ => return Err(invalid("unsupported flavour-flow operation")),
    }
    if result != expected {
        return Err(invalid(format!(
            "quantum flow {row} flavour operation does not reproduce its stored result witness"
        )));
    }
    Ok(())
}

fn validate_strict_u32(values: &[u32], label: &str) -> RusticolResult<()> {
    for (index, pair) in values.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(invalid(format!(
                "{label} is not strictly ordered at item {}",
                index + 1
            )));
        }
    }
    Ok(())
}

fn validate_permutation(values: &[u32], label: &str) -> RusticolResult<()> {
    validate_permutation_of(values, values.len(), label)
}

fn validate_permutation_of(values: &[u32], arity: usize, label: &str) -> RusticolResult<()> {
    if values.len() != arity {
        return Err(invalid(format!(
            "{label} length {} does not match arity {arity}",
            values.len()
        )));
    }
    let mut seen = vec![false; arity];
    for value in values {
        let index = usize::try_from(*value)
            .map_err(|_| invalid(format!("{label} item {value} exceeds usize")))?;
        if index >= arity || seen[index] {
            return Err(invalid(format!(
                "{label} is not a permutation of 0..{arity}"
            )));
        }
        seen[index] = true;
    }
    Ok(())
}

fn register_template(
    templates: &mut BTreeMap<u32, TemplateKind>,
    string_id: u32,
    kind: TemplateKind,
    label: &str,
) -> RusticolResult<()> {
    if let Some(previous) = templates.insert(string_id, kind) {
        return Err(invalid(format!(
            "{label} reuses global template string id {string_id} already owned by {previous:?}"
        )));
    }
    Ok(())
}

fn register_semantic_digest(
    digests: &mut BTreeSet<u32>,
    digest_id: u32,
    label: &str,
) -> RusticolResult<()> {
    if !digests.insert(digest_id) {
        return Err(invalid(format!(
            "{label} reuses semantic digest id {digest_id}"
        )));
    }
    Ok(())
}

fn parse_canonical_i128(value: &str, positive: bool, label: &str) -> RusticolResult<i128> {
    let parsed = value
        .parse::<i128>()
        .map_err(|_| invalid(format!("{label} {value:?} is outside the i128 domain")))?;
    if parsed.to_string() != value {
        return Err(invalid(format!(
            "{label} {value:?} is not a canonical decimal integer"
        )));
    }
    if parsed == i128::MIN {
        return Err(invalid(format!(
            "{label} cannot use i128::MIN in the symmetric exact-rational domain"
        )));
    }
    if positive && parsed <= 0 {
        return Err(invalid(format!("{label} must be positive")));
    }
    Ok(parsed)
}

fn is_supported_proof_algorithm(value: &str) -> bool {
    matches!(
        value,
        "canonical-crossing-bijection-v1"
            | "canonical-current-word-reversal-v1"
            | "canonical-kernel-input-exchange-v1"
            | "canonical-model-contract-label-equivariance-v1"
            | "canonical-recurrence-replay-witness-v1"
            | "canonical-recurrence-union-witness-v1"
            | "canonical-source-transition-dependency-shape-v1"
            | "canonical-trace-amplitude-reversal-v1"
            | "exact-expression-identity-v1"
            | "prepared-kernel-homogeneous-complex-linear-current-v1"
            | "prepared-kernel-independent-current-block-v1"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_canonical_input() -> OwnedRecurrenceTemplateInput {
        let mut strings = vec![
            RECURRENCE_TEMPLATE_INPUT_ABI,
            RECURRENCE_TEMPLATE_ABI,
            RECURRENCE_TEMPLATE_CANONICALIZATION_ABI,
            RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI,
        ];
        strings.sort_unstable();
        strings.dedup();
        let string_id =
            |value: &str| u32::try_from(strings.binary_search(&value).unwrap()).unwrap();
        let mut string_ranges = Vec::new();
        let mut string_bytes = Vec::new();
        for value in &strings {
            string_ranges.push(CheckedTableRange::new(
                string_bytes.len() as u64,
                value.len() as u64,
            ));
            string_bytes.extend_from_slice(value.as_bytes());
        }

        let compiled_model_digest = SemanticDigest::new([1; 32]).unwrap();
        let prepared_kernel_pack_digest = SemanticDigest::new([2; 32]).unwrap();
        let catalog_digest = SemanticDigest::new([3; 32]).unwrap();
        OwnedRecurrenceTemplateInput {
            input_abi: RECURRENCE_TEMPLATE_INPUT_ABI.to_owned(),
            catalog_digest,
            compiled_model_digest,
            prepared_kernel_pack_digest,
            catalog_header: vec![CatalogHeaderRow {
                schema_version: RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
                abi_string_id: string_id(RECURRENCE_TEMPLATE_ABI),
                canonicalization_abi_string_id: string_id(RECURRENCE_TEMPLATE_CANONICALIZATION_ABI),
                exact_scalar_abi_string_id: string_id(RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI),
                compiled_model_digest_id: 0,
                prepared_kernel_pack_digest_id: 1,
                catalog_digest_id: 2,
                parameter_count: 0,
                current_state_count: 0,
                source_count: 0,
                quantum_flow_count: 0,
                contact_orbit_certificate_count: 0,
                contact_orbit_step_count: 0,
                transition_count: 0,
                propagator_count: 0,
                closure_count: 0,
                color_contraction_count: 0,
                symmetry_proof_count: 0,
                runtime_helicity_contract_count: 0,
                evaluator_binding_count: 0,
            }],
            coupling_order_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 0),
            }],
            coupling_order_terms: vec![],
            contact_orbit_certificates: vec![],
            contact_orbit_steps: vec![],
            current_states: vec![],
            digest_catalog: vec![
                DigestCatalogRow {
                    id: 0,
                    value: [1; 32],
                },
                DigestCatalogRow {
                    id: 1,
                    value: [2; 32],
                },
                DigestCatalogRow {
                    id: 2,
                    value: [3; 32],
                },
            ],
            evaluator_bindings: vec![],
            exact_factors: vec![],
            flavour_flow_ranges: vec![],
            flavour_flow_values: vec![],
            i32_sequence_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 0),
            }],
            i32_sequence_values: vec![],
            parameters: vec![],
            propagators: vec![],
            quantum_flows: vec![],
            quantum_number_flow_ranges: vec![],
            quantum_number_flow_terms: vec![],
            runtime_helicity_contracts: vec![],
            runtime_helicity_variants: vec![],
            runtime_helicity_embeddings: vec![],
            runtime_helicity_projections: vec![],
            sources: vec![],
            string_ranges,
            string_bytes,
            symmetry_proofs: vec![],
            transitions: vec![],
            closures: vec![],
            color_contractions: vec![],
            lc_color_transition_witnesses: vec![],
            color_nc_terms: vec![],
            u32_sequence_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 0),
            }],
            u32_sequence_values: vec![],
        }
    }

    #[test]
    fn output_factor_source_is_closed() {
        assert_eq!(
            OutputFactorSource::try_from(0).unwrap(),
            OutputFactorSource::None
        );
        assert!(OutputFactorSource::try_from(3).is_err());
    }

    #[test]
    fn canonical_i128_rejects_aliases_and_asymmetric_minimum() {
        assert_eq!(parse_canonical_i128("-17", false, "value").unwrap(), -17);
        assert!(parse_canonical_i128("+17", false, "value").is_err());
        assert!(parse_canonical_i128("017", false, "value").is_err());
        assert!(parse_canonical_i128(&i128::MIN.to_string(), false, "value").is_err());
    }

    #[test]
    fn permutations_are_checked_exactly() {
        validate_permutation_of(&[2, 0, 1], 3, "test permutation").unwrap();
        assert!(validate_permutation_of(&[0, 0, 1], 3, "test permutation").is_err());
        assert!(validate_permutation_of(&[0, 1], 3, "test permutation").is_err());
    }

    #[test]
    fn canonical_sequence_order_rejects_duplicates() {
        let ranges = [
            IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 1),
            },
            IndexedRangeRow {
                id: 1,
                range: CheckedTableRange::new(1, 1),
            },
        ];
        assert!(validate_sequence_catalog("test", &ranges, &[4_u32, 4], false).is_err());
    }

    #[test]
    fn absent_color_output_requires_zero_sentinel() {
        validate_color_contract_shape(0, &[3, -3], 0, 0, 1).unwrap();
        assert!(validate_color_contract_shape(0, &[3, -3], 0, 8, 1).is_err());
        assert!(validate_color_contract_shape(0, &[3, -3], 2, 0, 1).is_err());
    }

    #[test]
    fn color_open_string_arity_respects_available_endpoints() {
        validate_color_contract_shape(0, &[3, 8], 1, 3, 1).unwrap();
        assert!(validate_color_contract_shape(0, &[3, 8], 1, 3, 2).is_err());
        assert!(validate_color_contract_shape(0, &[8, 8], 1, 8, 1).is_err());

        // Unknown representation encodings retain the generic endpoint-slot
        // bound without acquiring a model-specific fundamental-role rule.
        validate_color_contract_shape(0, &[41, -41], 0, 0, 1).unwrap();
    }

    #[test]
    fn empty_model_catalog_validates_end_to_end() {
        let validated = empty_canonical_input().validate().unwrap();
        assert_eq!(validated.summary().current_state_count, 0);
        assert_eq!(validated.summary().prepared_kernel_count, 0);
        let semantic_index = validated.semantic_index().unwrap();
        assert!(semantic_index.is_empty());
        assert_eq!(
            semantic_index.compiled_model_digest,
            validated.summary().compiled_model_digest
        );
        assert_eq!(
            semantic_index.catalog_digest,
            validated.summary().catalog_digest
        );
    }

    #[test]
    fn closure_component_coefficient_catalog_decodes_every_template() {
        let mut input = crate::recurrence::validated_template_fixture().into_input();
        let unit = input.exact_factors[0];
        input.exact_factors.push(ExactFactorRow {
            id: 1,
            real_numerator_string_id: unit.real_numerator_string_id,
            real_denominator_string_id: unit.real_denominator_string_id,
            imag_numerator_string_id: unit.real_numerator_string_id,
            imag_denominator_string_id: unit.imag_denominator_string_id,
        });
        input.closures[1].component_coefficient_sequence_id =
            input.closures[1].canonical_input_order_sequence_id;

        let catalog = input
            .validate()
            .unwrap()
            .closure_component_coefficient_catalog()
            .unwrap();
        let unit_plus_i = ExactComplexRational::parse_parts("1", "1", "1", "1").unwrap();
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog[0].as_ref(), &[ExactComplexRational::ONE]);
        assert_eq!(
            catalog[1].as_ref(),
            &[ExactComplexRational::ONE, unit_plus_i]
        );
    }

    #[test]
    fn closure_component_coefficient_catalog_rejects_corrupt_validated_state() {
        let mut templates = crate::recurrence::validated_template_fixture();
        templates.input.closures[0].component_coefficient_sequence_id =
            templates.input.closures[0].canonical_input_order_sequence_id;
        let error = templates
            .closure_component_coefficient_catalog()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("closure component factor ID is out of range"),
            "unexpected validation error: {error}"
        );

        let mut templates = crate::recurrence::validated_template_fixture();
        templates.input.closures[0].component_coefficient_sequence_id = MISSING_U32;
        let error = templates
            .closure_component_coefficient_catalog()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("closure component coefficients id")
                && error.contains("exceeds target length"),
            "unexpected validation error: {error}"
        );

        let mut templates = crate::recurrence::validated_template_fixture();
        templates.input.closures[1].id = 7;
        let error = templates
            .closure_component_coefficient_catalog()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("closure template 1 has noncanonical ID 7"),
            "unexpected validation error: {error}"
        );
    }

    #[test]
    fn indirect_parameter_dependency_cycles_are_rejected() {
        let mut input = empty_canonical_input();
        input.catalog_header[0].parameter_count = 2;
        input.u32_sequence_ranges.extend([
            IndexedRangeRow {
                id: 1,
                range: CheckedTableRange::new(0, 1),
            },
            IndexedRangeRow {
                id: 2,
                range: CheckedTableRange::new(1, 1),
            },
        ]);
        input.u32_sequence_values.extend([0, 1]);
        let mut parameter_template_ids = [
            input.catalog_header[0].abi_string_id,
            input.catalog_header[0].canonicalization_abi_string_id,
        ];
        parameter_template_ids.sort_unstable();
        input.parameters = vec![
            ParameterRow {
                id: 0,
                template_string_id: parameter_template_ids[0],
                name_string_id: input.catalog_header[0].canonicalization_abi_string_id,
                kind: ParameterKind::Derived as u8,
                value_type: ParameterValueType::Complex as u8,
                mutable: 0,
                default_factor_id: MISSING_U32,
                exact_expression_digest_id: 0,
                dependency_sequence_id: 2,
                prepared_parameter_id: MISSING_U32,
                semantic_digest_id: 0,
            },
            ParameterRow {
                id: 1,
                template_string_id: parameter_template_ids[1],
                name_string_id: input.catalog_header[0].exact_scalar_abi_string_id,
                kind: ParameterKind::Derived as u8,
                value_type: ParameterValueType::Complex as u8,
                mutable: 0,
                default_factor_id: MISSING_U32,
                exact_expression_digest_id: 1,
                dependency_sequence_id: 1,
                prepared_parameter_id: MISSING_U32,
                semantic_digest_id: 1,
            },
        ];

        let error = input.validate().unwrap_err().to_string();
        assert!(
            error.contains("parameter dependency graph contains a cycle"),
            "unexpected validation error: {error}"
        );
    }
}
