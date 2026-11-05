// SPDX-License-Identifier: 0BSD

//! Strict projection of the retained semantic recurrence-template JSON.
//!
//! Prepared kernel packs retain the model-wide recurrence template as the
//! canonical `RecurrenceTemplateCatalog.to_dict()` object.  This module turns
//! that authenticated semantic object directly into the primitive owned core
//! columns.  It deliberately has no Python, NumPy, process-builder, or direct
//! plan dependency.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Deserializer};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::template::{
    CatalogHeaderRow, ClosureRow, ColorContractionRow, ColorNcTermRow, ContactOrbitCertificateRow,
    ContactOrbitStepRow, CouplingOrderTermRow, CurrentStateRow, DigestCatalogRow,
    EvaluatorBindingRow, ExactFactorRow, IndexedRangeRow, LCColorTransitionWitnessRow, MISSING_U32,
    OwnedRecurrenceTemplateInput, ParameterRow, PropagatorRow, QuantumFlowRow,
    QuantumNumberFlowTermRow, RECURRENCE_TEMPLATE_CANONICALIZATION_ABI,
    RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI, RECURRENCE_TEMPLATE_INPUT_ABI,
    RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION, RuntimeHelicityContractRow,
    RuntimeHelicityEmbeddingRow, RuntimeHelicityProjectionRow, RuntimeHelicityVariantRow,
    SourceRow, SymmetryProofRow, TransitionRow,
};
use super::{CheckedTableRange, ExactComplexRational, RECURRENCE_TEMPLATE_ABI, SemanticDigest};
use crate::{RusticolError, RusticolResult};

const ROOT_SECTIONS: [(&str, &str, &str); 13] = [
    ("parameters", "parameter", "template_id"),
    ("current_states", "current-state", "template_id"),
    ("sources", "source", "template_id"),
    ("quantum_flows", "quantum-flow", "template_id"),
    (
        "contact_orbit_certificates",
        "contact-orbit-certificate",
        "template_id",
    ),
    ("contact_orbit_steps", "contact-orbit-step", "template_id"),
    ("transitions", "transition", "template_id"),
    ("propagators", "propagator", "template_id"),
    ("closures", "closure", "template_id"),
    ("color_contractions", "color-contraction", "template_id"),
    ("symmetry_proofs", "symmetry-proof", "template_id"),
    (
        "runtime_helicity_contracts",
        "runtime-helicity-contract",
        "template_id",
    ),
    ("evaluator_bindings", "evaluator-binding", "resolver_key"),
];

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

#[derive(Clone, Debug)]
struct RequiredOption<T>(Option<T>);

impl<'de, T> Deserialize<'de> for RequiredOption<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(Self)
    }
}

#[derive(Clone, Debug, Default)]
enum OptionalField<T> {
    #[default]
    Absent,
    Present(T),
}

impl<'de, T> Deserialize<'de> for OptionalField<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Self::Present)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticCatalogJson {
    header: CatalogHeaderJson,
    parameters: Vec<ParameterJson>,
    current_states: Vec<CurrentStateJson>,
    sources: Vec<SourceJson>,
    quantum_flows: Vec<QuantumFlowJson>,
    contact_orbit_certificates: Vec<ContactOrbitCertificateJson>,
    contact_orbit_steps: Vec<ContactOrbitStepJson>,
    transitions: Vec<TransitionJson>,
    propagators: Vec<PropagatorJson>,
    closures: Vec<ClosureJson>,
    color_contractions: Vec<ColorContractionJson>,
    symmetry_proofs: Vec<SymmetryProofJson>,
    runtime_helicity_contracts: Vec<RuntimeHelicityContractJson>,
    evaluator_bindings: Vec<EvaluatorBindingJson>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogHeaderJson {
    abi: String,
    canonicalization_abi: String,
    compiled_model_digest: String,
    exact_scalar_abi: String,
    prepared_kernel_pack_digest: String,
    catalog_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactFactorJson {
    real_numerator: String,
    real_denominator: String,
    imag_numerator: String,
    imag_denominator: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FactorKey {
    parts: [i128; 4],
    strings: [String; 4],
}

impl ExactFactorJson {
    fn key(&self) -> RusticolResult<FactorKey> {
        let value = ExactComplexRational::parse_parts(
            &self.real_numerator,
            &self.real_denominator,
            &self.imag_numerator,
            &self.imag_denominator,
        )?;
        let real = value.real().to_string();
        let imag = value.imag().to_string();
        let expected_real = format!("{}/{}", self.real_numerator, self.real_denominator);
        let expected_imag = format!("{}/{}", self.imag_numerator, self.imag_denominator);
        if real != expected_real || imag != expected_imag {
            return Err(invalid(
                "exact complex rational components are not canonical reduced decimals",
            ));
        }
        Ok(FactorKey {
            parts: [
                parse_i128(&self.real_numerator, "real numerator")?,
                parse_i128(&self.real_denominator, "real denominator")?,
                parse_i128(&self.imag_numerator, "imaginary numerator")?,
                parse_i128(&self.imag_denominator, "imaginary denominator")?,
            ],
            strings: [
                self.real_numerator.clone(),
                self.real_denominator.clone(),
                self.imag_numerator.clone(),
                self.imag_denominator.clone(),
            ],
        })
    }
}

fn parse_i128(value: &str, context: &str) -> RusticolResult<i128> {
    value
        .parse::<i128>()
        .map_err(|_| invalid(format!("{context} is outside the exact i128 domain")))
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    name: String,
    parameter_kind: String,
    value_type: String,
    mutable: bool,
    default_value: RequiredOption<ExactFactorJson>,
    exact_expression_digest: RequiredOption<String>,
    dependency_parameter_ids: Vec<String>,
    prepared_parameter_id: RequiredOption<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CurrentStateJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    particle_id: i32,
    anti_particle_id: i32,
    species_id: String,
    orientation: String,
    statistics: String,
    color_representation: i32,
    basis: String,
    tensor_ordering: Vec<String>,
    dimension: u32,
    chirality: i32,
    lc_color_shape_kind: String,
    auxiliary_kind: RequiredOption<String>,
    mass_parameter_id: RequiredOption<String>,
    width_parameter_id: RequiredOption<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LCColorSourceSeedJson {
    operation: String,
    output_shape_kind: String,
    component_kind: RequiredOption<String>,
    component_role: String,
    proof_digest: String,
    provenance: Vec<[String; 2]>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    state_template_id: String,
    crossing: String,
    wavefunction_family: String,
    helicity: i32,
    spin_state: i32,
    flavour_flow: Vec<i32>,
    quantum_number_flow: Vec<[String; 2]>,
    lc_color_seed: LCColorSourceSeedJson,
    wavefunction_expression_digest: String,
    evaluator_resolver_key: String,
    mass_parameter_id: RequiredOption<String>,
    width_parameter_id: RequiredOption<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantumFlowJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    input_state_template_ids: Vec<String>,
    input_spin_states: Vec<i32>,
    input_flavour_flows: Vec<Vec<i32>>,
    input_quantum_number_flows: Vec<Vec<[String; 2]>>,
    flavour_flow_operation: String,
    quantum_number_flow_operation: String,
    coupling_orders: Vec<(String, u32)>,
    result_state_template_id: String,
    result_spin_state: i32,
    result_flavour_flow: Vec<i32>,
    result_quantum_number_flow: Vec<[String; 2]>,
    exact_coupling: ExactFactorJson,
    predicate_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContactOrbitCertificateJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    algorithm: String,
    algorithm_version: u32,
    term_id: u32,
    vertex: String,
    particles: Vec<String>,
    evaluator_class: String,
    physical_leg_equivalence_classes: Vec<u32>,
    reconstruction_factor: ExactFactorJson,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContactOrbitStepJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    certificate_template_id: String,
    stage: String,
    result_leg: u8,
    left_covered_legs: Vec<u32>,
    right_covered_legs: Vec<u32>,
    source_particle_legs: [i32; 3],
    reconstruction_factor: ExactFactorJson,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TransitionJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    input_state_template_ids: Vec<String>,
    result_state_template_id: String,
    quantum_flow_template_id: String,
    evaluator_resolver_key: String,
    canonical_input_order: Vec<u32>,
    momentum_convention: Vec<String>,
    coupling_parameter_ids: Vec<String>,
    coupling_orders: Vec<(String, u32)>,
    color_contraction_template_id: String,
    binding_coupling: ExactFactorJson,
    exact_factor: ExactFactorJson,
    output_factor_source: String,
    equivalence_class: String,
    input_exchange_factor: RequiredOption<ExactFactorJson>,
    output_projection: String,
    contact_orbit_step_template_ids: Vec<String>,
    contact_orbit_step_semantic_digests: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PropagatorJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    state_template_id: String,
    applies_propagator: bool,
    evaluator_resolver_key: RequiredOption<String>,
    numerator_expression_digest: RequiredOption<String>,
    denominator_expression_digest: RequiredOption<String>,
    mass_parameter_id: RequiredOption<String>,
    width_parameter_id: RequiredOption<String>,
    gauge: RequiredOption<String>,
    linearity_proof_template_id: RequiredOption<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClosureJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    input_state_template_ids: Vec<String>,
    result_state_template_id: RequiredOption<String>,
    evaluator_resolver_key: String,
    canonical_input_order: Vec<u32>,
    coupling_parameter_ids: Vec<String>,
    coupling_orders: Vec<(String, u32)>,
    eligible_quantum_flow_template_ids: Vec<String>,
    color_contraction_template_id: String,
    binding_coupling: ExactFactorJson,
    exact_factor: ExactFactorJson,
    output_factor_source: String,
    equivalence_class: String,
    input_exchange_factor: RequiredOption<ExactFactorJson>,
    projection: String,
    component_coefficients: Vec<ExactFactorJson>,
    chirality_relation: String,
    metric_signature: RequiredOption<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LCColorTransitionWitnessJson {
    component_operation: String,
    exact_factor: ExactFactorJson,
    input_permutation: [u8; 2],
    input_shape_kinds: [String; 2],
    proof_digest: String,
    provenance: Vec<[String; 2]>,
    result_component_kind: RequiredOption<String>,
    result_component_role: String,
    result_shape_kind: RequiredOption<String>,
    reverse_parent_mask: u8,
    #[serde(default)]
    input_port_pairings: OptionalField<Vec<[[u32; 2]; 2]>>,
    #[serde(default)]
    result_port_bindings: OptionalField<Vec<[u32; 2]>>,
}

type CanonicalPortSlices<'a> = (&'a [[[u32; 2]; 2]], &'a [[u32; 2]]);

impl LCColorTransitionWitnessJson {
    fn canonical_ports(&self) -> RusticolResult<CanonicalPortSlices<'_>> {
        match (&self.input_port_pairings, &self.result_port_bindings) {
            (OptionalField::Absent, OptionalField::Absent) => Ok((&[], &[])),
            (OptionalField::Present(pairings), OptionalField::Present(bindings))
                if !pairings.is_empty() || !bindings.is_empty() =>
            {
                Ok((pairings, bindings))
            }
            (OptionalField::Present(_), OptionalField::Present(_)) => Err(invalid(
                "LC color witness must omit empty canonical port fields",
            )),
            _ => Err(invalid(
                "LC color witness port-pairing and result-binding fields must be paired",
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ColorContractionJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    rule_kind: String,
    input_representations: Vec<i32>,
    output_representation: RequiredOption<i32>,
    ordered_open_string_arity: u32,
    exact_coefficient: ExactFactorJson,
    nc_polynomial: Vec<(i32, ExactFactorJson)>,
    expression_digest: String,
    transition_witnesses: Vec<LCColorTransitionWitnessJson>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SymmetryProofJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    proof_algorithm: String,
    subject_template_ids: Vec<String>,
    input_permutation: Vec<u32>,
    exact_phase: ExactFactorJson,
    expression_digests: Vec<String>,
    witness_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeHelicityVariantJson {
    source_template_id: String,
    source_state_template_id: String,
    embedding_source_components: Vec<Option<u32>>,
    embedding_factors: Vec<ExactFactorJson>,
    projection_full_components: Vec<u32>,
    proof_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeHelicityContractJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    template_id: String,
    full_state_template_id: String,
    variants: Vec<RuntimeHelicityVariantJson>,
    proof_algorithm: String,
    proof_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluatorBindingJson {
    #[serde(rename = "record_kind")]
    _record_kind: String,
    semantic_digest: String,
    resolver_key: String,
    prepared_kernel_id: RequiredOption<u32>,
    contract_kind: String,
    callable_signature: String,
    input_state_template_ids: Vec<String>,
    output_state_template_id: RequiredOption<String>,
    input_layout: Vec<String>,
    output_layout: Vec<String>,
    exact_expression_digests: Vec<String>,
    semantic_template_ids: Vec<String>,
    callable_kind: String,
    runtime_template: RequiredOption<String>,
}

#[derive(Clone, Debug)]
struct Catalog<T: Ord> {
    values: Vec<T>,
    ids: BTreeMap<T, u32>,
}

impl<T: Clone + Ord> Catalog<T> {
    fn from_set(values: BTreeSet<T>, context: &str) -> RusticolResult<Self> {
        let values = values.into_iter().collect::<Vec<_>>();
        let mut ids = BTreeMap::new();
        for (index, value) in values.iter().cloned().enumerate() {
            ids.insert(value, checked_u32(index, context)?);
        }
        Ok(Self { values, ids })
    }

    fn id(&self, value: &T, context: &str) -> RusticolResult<u32> {
        self.ids
            .get(value)
            .copied()
            .ok_or_else(|| invalid(format!("{context} is absent from its canonical catalog")))
    }
}

fn checked_u32(value: usize, context: &str) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{context} exceeds u32")))
}

fn checked_u64(value: usize, context: &str) -> RusticolResult<u64> {
    u64::try_from(value).map_err(|_| invalid(format!("{context} exceeds u64")))
}

fn optional_id<T: Ord>(
    value: Option<&T>,
    ids: &BTreeMap<T, u32>,
    context: &str,
) -> RusticolResult<u32> {
    match value {
        None => Ok(MISSING_U32),
        Some(value) => ids
            .get(value)
            .copied()
            .ok_or_else(|| invalid(format!("unknown {context}"))),
    }
}

fn mapped_ids(
    values: &[String],
    ids: &BTreeMap<String, u32>,
    context: &str,
) -> RusticolResult<Vec<u32>> {
    values
        .iter()
        .map(|value| {
            ids.get(value)
                .copied()
                .ok_or_else(|| invalid(format!("unknown {context} {value:?}")))
        })
        .collect()
}

fn enum_id(value: &str, choices: &[(&str, u8)], context: &str) -> RusticolResult<u8> {
    choices
        .iter()
        .find_map(|(name, id)| (*name == value).then_some(*id))
        .ok_or_else(|| invalid(format!("unsupported {context} {value:?}")))
}

/// Project one retained `RecurrenceTemplateCatalog.to_dict()` value into the
/// validated, owned core recurrence-template input.
///
/// The caller must pass the semantic catalog object itself, not a surrounding
/// prepared-pack object.  The function authenticates every record digest and
/// the catalog digest before deriving any primitive IDs.
#[cfg(any(feature = "python-generation-bridge", test))]
pub(crate) fn project_recurrence_template_catalog_json_v1(
    value: &Value,
) -> RusticolResult<OwnedRecurrenceTemplateInput> {
    authenticate_semantic_json(value)?;
    project_authenticated_recurrence_template_catalog_json_v1(value.clone())
}

/// Project an owned semantic catalog without retaining its JSON representation.
pub(crate) fn project_owned_recurrence_template_catalog_json_v1(
    value: Value,
) -> RusticolResult<OwnedRecurrenceTemplateInput> {
    authenticate_semantic_json(&value)?;
    project_authenticated_recurrence_template_catalog_json_v1(value)
}

fn project_authenticated_recurrence_template_catalog_json_v1(
    value: Value,
) -> RusticolResult<OwnedRecurrenceTemplateInput> {
    let catalog: SemanticCatalogJson = serde_json::from_value(value).map_err(|error| {
        invalid(format!(
            "invalid recurrence-template semantic JSON shape: {error}"
        ))
    })?;
    validate_header(&catalog.header)?;
    validate_witness_port_shapes(&catalog)?;
    project_catalog(&catalog)
}

fn validate_header(header: &CatalogHeaderJson) -> RusticolResult<()> {
    if header.abi != RECURRENCE_TEMPLATE_ABI {
        return Err(invalid(format!(
            "unsupported recurrence-template ABI {:?}",
            header.abi
        )));
    }
    if header.canonicalization_abi != RECURRENCE_TEMPLATE_CANONICALIZATION_ABI {
        return Err(invalid(format!(
            "unsupported recurrence-template canonicalization ABI {:?}",
            header.canonicalization_abi
        )));
    }
    if header.exact_scalar_abi != RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI {
        return Err(invalid(format!(
            "unsupported recurrence-template exact-scalar ABI {:?}",
            header.exact_scalar_abi
        )));
    }
    parse_digest(&header.compiled_model_digest, "compiled-model digest")?;
    parse_digest(
        &header.prepared_kernel_pack_digest,
        "prepared-kernel-pack digest",
    )?;
    parse_digest(&header.catalog_digest, "catalog digest")?;
    Ok(())
}

fn validate_witness_port_shapes(catalog: &SemanticCatalogJson) -> RusticolResult<()> {
    for color in &catalog.color_contractions {
        for witness in &color.transition_witnesses {
            witness.canonical_ports()?;
        }
    }
    for contract in &catalog.runtime_helicity_contracts {
        let mut previous_source: Option<&str> = None;
        for variant in &contract.variants {
            if previous_source
                .is_some_and(|previous| previous >= variant.source_template_id.as_str())
            {
                return Err(invalid(
                    "runtime-helicity variants must be in strict source-template order",
                ));
            }
            previous_source = Some(&variant.source_template_id);
            if variant.embedding_source_components.len() != variant.embedding_factors.len() {
                return Err(invalid(
                    "runtime-helicity embedding components and factors must have equal length",
                ));
            }
        }
    }
    Ok(())
}

fn authenticate_semantic_json(value: &Value) -> RusticolResult<()> {
    let root = value
        .as_object()
        .ok_or_else(|| invalid("recurrence-template catalog must be a JSON object"))?;
    let header = root
        .get("header")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("recurrence-template catalog header must be an object"))?;
    let expected_catalog_digest = header
        .get("catalog_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("recurrence-template catalog digest must be a string"))?;
    parse_digest(expected_catalog_digest, "catalog digest")?;

    for (section, expected_kind, identity_field) in ROOT_SECTIONS {
        let records = root
            .get(section)
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("recurrence-template {section} must be an array")))?;
        let mut previous_identity: Option<&str> = None;
        for (index, record) in records.iter().enumerate() {
            let object = record.as_object().ok_or_else(|| {
                invalid(format!(
                    "recurrence-template {section} row {index} must be an object"
                ))
            })?;
            let kind = object
                .get("record_kind")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid(format!(
                        "recurrence-template {section} row {index} lacks record_kind"
                    ))
                })?;
            if kind != expected_kind {
                return Err(invalid(format!(
                    "recurrence-template {section} row {index} has record kind {kind:?}, expected {expected_kind:?}"
                )));
            }
            let identity = object
                .get(identity_field)
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid(format!(
                        "recurrence-template {section} row {index} lacks string {identity_field}"
                    ))
                })?;
            if identity.is_empty() || previous_identity.is_some_and(|previous| previous >= identity)
            {
                return Err(invalid(format!(
                    "recurrence-template {section} identities are not in strict canonical order at row {index}"
                )));
            }
            previous_identity = Some(identity);

            let expected = object
                .get("semantic_digest")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid(format!(
                        "recurrence-template {section} row {index} lacks semantic_digest"
                    ))
                })?;
            parse_digest(expected, "record semantic digest")?;
            let mut payload = record.clone();
            payload
                .as_object_mut()
                .expect("record object was checked")
                .remove("semantic_digest");
            let actual = canonical_json_digest(&payload)?;
            if expected != actual {
                return Err(invalid(format!(
                    "stale recurrence-template semantic digest in {section} row {index}"
                )));
            }
        }
    }

    let mut catalog_payload = value.clone();
    catalog_payload
        .as_object_mut()
        .and_then(|root| root.get_mut("header"))
        .and_then(Value::as_object_mut)
        .expect("catalog/header objects were checked")
        .remove("catalog_digest");
    let actual_catalog_digest = canonical_json_digest(&catalog_payload)?;
    if expected_catalog_digest != actual_catalog_digest {
        return Err(invalid("stale recurrence-template catalog digest"));
    }
    Ok(())
}

fn parse_digest(value: &str, context: &str) -> RusticolResult<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!(
            "{context} must be a lowercase SHA-256 digest"
        )));
    }
    let mut bytes = [0_u8; 32];
    for (index, output) in bytes.iter_mut().enumerate() {
        let offset = index * 2;
        *output = u8::from_str_radix(&value[offset..offset + 2], 16)
            .map_err(|_| invalid(format!("{context} contains invalid hexadecimal")))?;
    }
    if bytes == [0; 32] {
        return Err(invalid(format!("{context} must not be all zero")));
    }
    Ok(bytes)
}

fn semantic_digest(value: &str, context: &str) -> RusticolResult<SemanticDigest> {
    SemanticDigest::new(parse_digest(value, context)?)
}

fn canonical_json_digest(value: &Value) -> RusticolResult<String> {
    let mut canonical = String::new();
    write_canonical_json(value, &mut canonical)?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn write_canonical_json(value: &Value, output: &mut String) -> RusticolResult<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&python_json_number(value)),
        Value::String(value) => write_ascii_json_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_canonical_json(value, output)?;
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
                write_ascii_json_string(key, output);
                output.push(':');
                write_canonical_json(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn python_json_number(number: &serde_json::Number) -> String {
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

fn write_ascii_json_string(value: &str, output: &mut String) {
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

#[derive(Debug)]
struct SectionIds {
    parameters: BTreeMap<String, u32>,
    current_states: BTreeMap<String, u32>,
    sources: BTreeMap<String, u32>,
    quantum_flows: BTreeMap<String, u32>,
    contact_orbit_certificates: BTreeMap<String, u32>,
    contact_orbit_steps: BTreeMap<String, u32>,
    color_contractions: BTreeMap<String, u32>,
    symmetry_proofs: BTreeMap<String, u32>,
    evaluator_bindings: BTreeMap<String, u32>,
}

fn identity_ids<'a>(
    values: impl IntoIterator<Item = &'a String>,
    context: &str,
) -> RusticolResult<BTreeMap<String, u32>> {
    let mut result = BTreeMap::new();
    for (index, value) in values.into_iter().enumerate() {
        if result
            .insert(value.clone(), checked_u32(index, context)?)
            .is_some()
        {
            return Err(invalid(format!("duplicate {context} identity {value:?}")));
        }
    }
    Ok(result)
}

impl SectionIds {
    fn build(catalog: &SemanticCatalogJson) -> RusticolResult<Self> {
        Ok(Self {
            parameters: identity_ids(
                catalog.parameters.iter().map(|record| &record.template_id),
                "parameter",
            )?,
            current_states: identity_ids(
                catalog
                    .current_states
                    .iter()
                    .map(|record| &record.template_id),
                "current-state",
            )?,
            sources: identity_ids(
                catalog.sources.iter().map(|record| &record.template_id),
                "source",
            )?,
            quantum_flows: identity_ids(
                catalog
                    .quantum_flows
                    .iter()
                    .map(|record| &record.template_id),
                "quantum-flow",
            )?,
            contact_orbit_certificates: identity_ids(
                catalog
                    .contact_orbit_certificates
                    .iter()
                    .map(|record| &record.template_id),
                "contact-orbit certificate",
            )?,
            contact_orbit_steps: identity_ids(
                catalog
                    .contact_orbit_steps
                    .iter()
                    .map(|record| &record.template_id),
                "contact-orbit step",
            )?,
            color_contractions: identity_ids(
                catalog
                    .color_contractions
                    .iter()
                    .map(|record| &record.template_id),
                "color contraction",
            )?,
            symmetry_proofs: identity_ids(
                catalog
                    .symmetry_proofs
                    .iter()
                    .map(|record| &record.template_id),
                "symmetry proof",
            )?,
            evaluator_bindings: identity_ids(
                catalog
                    .evaluator_bindings
                    .iter()
                    .map(|record| &record.resolver_key),
                "evaluator binding",
            )?,
        })
    }
}

#[derive(Debug)]
struct PrimitiveCatalogs {
    strings: Catalog<String>,
    digests: Catalog<[u8; 32]>,
    factors: Catalog<FactorKey>,
    flavour_flows: Catalog<Vec<i32>>,
    quantum_number_flows: Catalog<Vec<(String, String)>>,
    coupling_orders: Catalog<Vec<(String, u32)>>,
    u32_sequences: Catalog<Vec<u32>>,
    i32_sequences: Catalog<Vec<i32>>,
}

fn qflow(values: &[[String; 2]]) -> Vec<(String, String)> {
    values
        .iter()
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect()
}

fn add_optional_string(values: &mut BTreeSet<String>, value: &RequiredOption<String>) {
    if let Some(value) = &value.0 {
        values.insert(value.clone());
    }
}

fn add_digest(values: &mut BTreeSet<[u8; 32]>, value: &str, context: &str) -> RusticolResult<()> {
    values.insert(parse_digest(value, context)?);
    Ok(())
}

fn add_optional_digest(
    values: &mut BTreeSet<[u8; 32]>,
    value: &RequiredOption<String>,
    context: &str,
) -> RusticolResult<()> {
    if let Some(value) = &value.0 {
        add_digest(values, value, context)?;
    }
    Ok(())
}

fn add_factor(values: &mut BTreeSet<FactorKey>, value: &ExactFactorJson) -> RusticolResult<()> {
    values.insert(value.key()?);
    Ok(())
}

fn add_optional_factor(
    values: &mut BTreeSet<FactorKey>,
    value: &RequiredOption<ExactFactorJson>,
) -> RusticolResult<()> {
    if let Some(value) = &value.0 {
        add_factor(values, value)?;
    }
    Ok(())
}

fn collect_strings(catalog: &SemanticCatalogJson) -> RusticolResult<BTreeSet<String>> {
    let mut values = BTreeSet::from([
        RECURRENCE_TEMPLATE_INPUT_ABI.to_owned(),
        catalog.header.abi.clone(),
        catalog.header.canonicalization_abi.clone(),
        catalog.header.exact_scalar_abi.clone(),
    ]);
    for record in &catalog.parameters {
        values.extend([record.template_id.clone(), record.name.clone()]);
        values.extend(record.dependency_parameter_ids.iter().cloned());
    }
    for record in &catalog.current_states {
        values.extend([
            record.template_id.clone(),
            record.species_id.clone(),
            record.basis.clone(),
            record.lc_color_shape_kind.clone(),
        ]);
        values.extend(record.tensor_ordering.iter().cloned());
        add_optional_string(&mut values, &record.auxiliary_kind);
        add_optional_string(&mut values, &record.mass_parameter_id);
        add_optional_string(&mut values, &record.width_parameter_id);
    }
    for record in &catalog.sources {
        values.extend([
            record.template_id.clone(),
            record.crossing.clone(),
            record.wavefunction_family.clone(),
            record.evaluator_resolver_key.clone(),
            record.lc_color_seed.output_shape_kind.clone(),
        ]);
        for pair in &record.lc_color_seed.provenance {
            values.extend(pair.iter().cloned());
        }
        for pair in &record.quantum_number_flow {
            values.extend(pair.iter().cloned());
        }
    }
    for record in &catalog.quantum_flows {
        values.extend([
            record.template_id.clone(),
            record.flavour_flow_operation.clone(),
            record.quantum_number_flow_operation.clone(),
        ]);
        for flow in record
            .input_quantum_number_flows
            .iter()
            .chain(std::iter::once(&record.result_quantum_number_flow))
        {
            for pair in flow {
                values.extend(pair.iter().cloned());
            }
        }
        values.extend(record.coupling_orders.iter().map(|(name, _)| name.clone()));
    }
    for record in &catalog.contact_orbit_certificates {
        values.extend([
            record.template_id.clone(),
            record.algorithm.clone(),
            record.vertex.clone(),
            record.evaluator_class.clone(),
        ]);
        values.extend(record.particles.iter().cloned());
    }
    for record in &catalog.contact_orbit_steps {
        values.extend([
            record.template_id.clone(),
            record.certificate_template_id.clone(),
            record.stage.clone(),
        ]);
    }
    for record in &catalog.transitions {
        values.extend([
            record.template_id.clone(),
            record.evaluator_resolver_key.clone(),
            record.output_factor_source.clone(),
            record.equivalence_class.clone(),
            record.output_projection.clone(),
        ]);
        values.extend(record.momentum_convention.iter().cloned());
        values.extend(record.coupling_orders.iter().map(|(name, _)| name.clone()));
    }
    for record in &catalog.propagators {
        values.insert(record.template_id.clone());
        add_optional_string(&mut values, &record.evaluator_resolver_key);
        add_optional_string(&mut values, &record.gauge);
    }
    for record in &catalog.closures {
        values.extend([
            record.template_id.clone(),
            record.evaluator_resolver_key.clone(),
            record.output_factor_source.clone(),
            record.equivalence_class.clone(),
            record.projection.clone(),
            record.chirality_relation.clone(),
        ]);
        add_optional_string(&mut values, &record.metric_signature);
        values.extend(record.coupling_orders.iter().map(|(name, _)| name.clone()));
    }
    for record in &catalog.color_contractions {
        values.extend([record.template_id.clone(), record.rule_kind.clone()]);
        for witness in &record.transition_witnesses {
            values.extend(witness.input_shape_kinds.iter().cloned());
            add_optional_string(&mut values, &witness.result_shape_kind);
            for pair in &witness.provenance {
                values.extend(pair.iter().cloned());
            }
        }
    }
    for record in &catalog.symmetry_proofs {
        values.extend([record.template_id.clone(), record.proof_algorithm.clone()]);
        values.extend(record.subject_template_ids.iter().cloned());
    }
    for record in &catalog.runtime_helicity_contracts {
        values.extend([
            record.template_id.clone(),
            record.full_state_template_id.clone(),
            record.proof_algorithm.clone(),
        ]);
        for variant in &record.variants {
            values.extend([
                variant.source_template_id.clone(),
                variant.source_state_template_id.clone(),
            ]);
        }
    }
    for record in &catalog.evaluator_bindings {
        values.insert(record.resolver_key.clone());
        values.extend(record.input_layout.iter().cloned());
        values.extend(record.output_layout.iter().cloned());
        values.extend(record.semantic_template_ids.iter().cloned());
        add_optional_string(&mut values, &record.runtime_template);
    }

    for factor in collect_factors(catalog)? {
        values.extend(factor.strings);
    }
    Ok(values)
}

fn collect_digests(catalog: &SemanticCatalogJson) -> RusticolResult<BTreeSet<[u8; 32]>> {
    let mut values = BTreeSet::new();
    add_digest(
        &mut values,
        &catalog.header.catalog_digest,
        "catalog digest",
    )?;
    add_digest(
        &mut values,
        &catalog.header.compiled_model_digest,
        "compiled-model digest",
    )?;
    add_digest(
        &mut values,
        &catalog.header.prepared_kernel_pack_digest,
        "prepared-kernel-pack digest",
    )?;
    macro_rules! record_digests {
        ($records:expr) => {
            for record in $records {
                add_digest(
                    &mut values,
                    &record.semantic_digest,
                    "record semantic digest",
                )?;
            }
        };
    }
    record_digests!(&catalog.parameters);
    record_digests!(&catalog.current_states);
    record_digests!(&catalog.sources);
    record_digests!(&catalog.quantum_flows);
    record_digests!(&catalog.contact_orbit_certificates);
    record_digests!(&catalog.contact_orbit_steps);
    record_digests!(&catalog.transitions);
    record_digests!(&catalog.propagators);
    record_digests!(&catalog.closures);
    record_digests!(&catalog.color_contractions);
    record_digests!(&catalog.symmetry_proofs);
    record_digests!(&catalog.runtime_helicity_contracts);
    record_digests!(&catalog.evaluator_bindings);

    for record in &catalog.parameters {
        add_optional_digest(
            &mut values,
            &record.exact_expression_digest,
            "parameter expression digest",
        )?;
    }
    for record in &catalog.sources {
        add_digest(
            &mut values,
            &record.wavefunction_expression_digest,
            "source expression digest",
        )?;
        add_digest(
            &mut values,
            &record.lc_color_seed.proof_digest,
            "source color proof digest",
        )?;
    }
    for record in &catalog.quantum_flows {
        add_digest(
            &mut values,
            &record.predicate_digest,
            "quantum predicate digest",
        )?;
    }
    for record in &catalog.transitions {
        for digest in &record.contact_orbit_step_semantic_digests {
            add_digest(&mut values, digest, "contact-orbit step digest")?;
        }
    }
    for record in &catalog.propagators {
        add_optional_digest(
            &mut values,
            &record.numerator_expression_digest,
            "propagator numerator digest",
        )?;
        add_optional_digest(
            &mut values,
            &record.denominator_expression_digest,
            "propagator denominator digest",
        )?;
    }
    for record in &catalog.color_contractions {
        add_digest(
            &mut values,
            &record.expression_digest,
            "color expression digest",
        )?;
        for witness in &record.transition_witnesses {
            add_digest(
                &mut values,
                &witness.proof_digest,
                "color witness proof digest",
            )?;
        }
    }
    for record in &catalog.symmetry_proofs {
        add_digest(
            &mut values,
            &record.witness_digest,
            "symmetry witness digest",
        )?;
        for digest in &record.expression_digests {
            add_digest(&mut values, digest, "symmetry expression digest")?;
        }
    }
    for record in &catalog.runtime_helicity_contracts {
        add_digest(&mut values, &record.proof_digest, "helicity proof digest")?;
        for variant in &record.variants {
            add_digest(
                &mut values,
                &variant.proof_digest,
                "helicity variant proof digest",
            )?;
        }
    }
    for record in &catalog.evaluator_bindings {
        add_digest(
            &mut values,
            &record.callable_signature,
            "callable signature digest",
        )?;
        for digest in &record.exact_expression_digests {
            add_digest(&mut values, digest, "evaluator expression digest")?;
        }
    }
    Ok(values)
}

fn collect_factors(catalog: &SemanticCatalogJson) -> RusticolResult<BTreeSet<FactorKey>> {
    let mut values = BTreeSet::new();
    for record in &catalog.parameters {
        add_optional_factor(&mut values, &record.default_value)?;
    }
    for record in &catalog.transitions {
        add_factor(&mut values, &record.binding_coupling)?;
        add_factor(&mut values, &record.exact_factor)?;
        add_optional_factor(&mut values, &record.input_exchange_factor)?;
    }
    for record in &catalog.contact_orbit_certificates {
        add_factor(&mut values, &record.reconstruction_factor)?;
    }
    for record in &catalog.contact_orbit_steps {
        add_factor(&mut values, &record.reconstruction_factor)?;
    }
    for record in &catalog.quantum_flows {
        add_factor(&mut values, &record.exact_coupling)?;
    }
    for record in &catalog.closures {
        add_factor(&mut values, &record.binding_coupling)?;
        add_factor(&mut values, &record.exact_factor)?;
        add_optional_factor(&mut values, &record.input_exchange_factor)?;
        for factor in &record.component_coefficients {
            add_factor(&mut values, factor)?;
        }
    }
    for record in &catalog.color_contractions {
        add_factor(&mut values, &record.exact_coefficient)?;
        for (_, factor) in &record.nc_polynomial {
            add_factor(&mut values, factor)?;
        }
        for witness in &record.transition_witnesses {
            add_factor(&mut values, &witness.exact_factor)?;
        }
    }
    for record in &catalog.symmetry_proofs {
        add_factor(&mut values, &record.exact_phase)?;
    }
    for record in &catalog.runtime_helicity_contracts {
        for variant in &record.variants {
            for factor in &variant.embedding_factors {
                add_factor(&mut values, factor)?;
            }
        }
    }
    Ok(values)
}

impl PrimitiveCatalogs {
    fn build(catalog: &SemanticCatalogJson, ids: &SectionIds) -> RusticolResult<Self> {
        let strings = Catalog::from_set(collect_strings(catalog)?, "string catalog")?;
        let digests = Catalog::from_set(collect_digests(catalog)?, "digest catalog")?;
        let factors = Catalog::from_set(collect_factors(catalog)?, "factor catalog")?;

        let mut flavour_values = BTreeSet::new();
        let mut quantum_values = BTreeSet::new();
        for record in &catalog.sources {
            flavour_values.insert(record.flavour_flow.clone());
            quantum_values.insert(qflow(&record.quantum_number_flow));
        }
        for record in &catalog.quantum_flows {
            flavour_values.extend(record.input_flavour_flows.iter().cloned());
            flavour_values.insert(record.result_flavour_flow.clone());
            quantum_values.extend(
                record
                    .input_quantum_number_flows
                    .iter()
                    .map(|flow| qflow(flow)),
            );
            quantum_values.insert(qflow(&record.result_quantum_number_flow));
        }
        let flavour_flows = Catalog::from_set(flavour_values, "flavour-flow catalog")?;
        let quantum_number_flows =
            Catalog::from_set(quantum_values, "quantum-number-flow catalog")?;

        let mut coupling_values = BTreeSet::from([Vec::new()]);
        coupling_values.extend(
            catalog
                .quantum_flows
                .iter()
                .map(|record| record.coupling_orders.clone()),
        );
        coupling_values.extend(
            catalog
                .transitions
                .iter()
                .map(|record| record.coupling_orders.clone()),
        );
        coupling_values.extend(
            catalog
                .closures
                .iter()
                .map(|record| record.coupling_orders.clone()),
        );
        let coupling_orders = Catalog::from_set(coupling_values, "coupling-order catalog")?;

        let mut u32_values = BTreeSet::from([Vec::new()]);
        let mut i32_values = BTreeSet::from([Vec::new()]);
        for record in &catalog.parameters {
            u32_values.insert(mapped_ids(
                &record.dependency_parameter_ids,
                &ids.parameters,
                "parameter dependency",
            )?);
        }
        for record in &catalog.current_states {
            u32_values.insert(
                record
                    .tensor_ordering
                    .iter()
                    .map(|value| string_id(&strings, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
        }
        for record in &catalog.sources {
            u32_values.insert(
                record
                    .lc_color_seed
                    .provenance
                    .iter()
                    .flat_map(|pair| pair.iter())
                    .map(|value| string_id(&strings, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
        }
        for record in &catalog.quantum_flows {
            u32_values.insert(mapped_ids(
                &record.input_state_template_ids,
                &ids.current_states,
                "quantum-flow input state",
            )?);
            u32_values.insert(
                record
                    .input_flavour_flows
                    .iter()
                    .map(|value| flavour_flows.id(value, "input flavour flow"))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
            u32_values.insert(
                record
                    .input_quantum_number_flows
                    .iter()
                    .map(|value| {
                        quantum_number_flows.id(&qflow(value), "input quantum-number flow")
                    })
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
            i32_values.insert(record.input_spin_states.clone());
        }
        for record in &catalog.transitions {
            u32_values.insert(mapped_ids(
                &record.input_state_template_ids,
                &ids.current_states,
                "transition input state",
            )?);
            u32_values.insert(record.canonical_input_order.clone());
            u32_values.insert(
                record
                    .momentum_convention
                    .iter()
                    .map(|value| string_id(&strings, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
            u32_values.insert(mapped_ids(
                &record.coupling_parameter_ids,
                &ids.parameters,
                "transition coupling parameter",
            )?);
            u32_values.insert(mapped_ids(
                &record.contact_orbit_step_template_ids,
                &ids.contact_orbit_steps,
                "transition contact-orbit step",
            )?);
            u32_values.insert(
                record
                    .contact_orbit_step_semantic_digests
                    .iter()
                    .map(|value| digest_id(&digests, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
        }
        for record in &catalog.contact_orbit_certificates {
            u32_values.insert(
                record
                    .particles
                    .iter()
                    .map(|value| string_id(&strings, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
            u32_values.insert(record.physical_leg_equivalence_classes.to_vec());
        }
        for record in &catalog.contact_orbit_steps {
            u32_values.insert(record.left_covered_legs.clone());
            u32_values.insert(record.right_covered_legs.clone());
            i32_values.insert(record.source_particle_legs.to_vec());
        }
        for record in &catalog.closures {
            u32_values.insert(mapped_ids(
                &record.input_state_template_ids,
                &ids.current_states,
                "closure input state",
            )?);
            u32_values.insert(record.canonical_input_order.clone());
            u32_values.insert(mapped_ids(
                &record.coupling_parameter_ids,
                &ids.parameters,
                "closure coupling parameter",
            )?);
            u32_values.insert(mapped_ids(
                &record.eligible_quantum_flow_template_ids,
                &ids.quantum_flows,
                "closure quantum flow",
            )?);
            u32_values.insert(
                record
                    .component_coefficients
                    .iter()
                    .map(|value| factor_id(&factors, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
        }
        for record in &catalog.color_contractions {
            i32_values.insert(record.input_representations.clone());
            for witness in &record.transition_witnesses {
                let (pairings, bindings) = witness.canonical_ports()?;
                u32_values.insert(
                    pairings
                        .iter()
                        .flat_map(|pairing| pairing.iter())
                        .flat_map(|port| port.iter().copied())
                        .collect(),
                );
                u32_values.insert(
                    bindings
                        .iter()
                        .flat_map(|port| port.iter().copied())
                        .collect(),
                );
                u32_values.insert(
                    witness
                        .provenance
                        .iter()
                        .flat_map(|pair| pair.iter())
                        .map(|value| string_id(&strings, value))
                        .collect::<RusticolResult<Vec<_>>>()?,
                );
            }
        }
        for record in &catalog.symmetry_proofs {
            u32_values.insert(
                record
                    .subject_template_ids
                    .iter()
                    .map(|value| string_id(&strings, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
            u32_values.insert(record.input_permutation.clone());
            u32_values.insert(
                record
                    .expression_digests
                    .iter()
                    .map(|value| digest_id(&digests, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
        }
        for record in &catalog.evaluator_bindings {
            u32_values.insert(mapped_ids(
                &record.input_state_template_ids,
                &ids.current_states,
                "evaluator input state",
            )?);
            u32_values.insert(
                record
                    .input_layout
                    .iter()
                    .map(|value| string_id(&strings, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
            u32_values.insert(
                record
                    .output_layout
                    .iter()
                    .map(|value| string_id(&strings, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
            u32_values.insert(
                record
                    .exact_expression_digests
                    .iter()
                    .map(|value| digest_id(&digests, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
            u32_values.insert(
                record
                    .semantic_template_ids
                    .iter()
                    .map(|value| string_id(&strings, value))
                    .collect::<RusticolResult<Vec<_>>>()?,
            );
        }

        Ok(Self {
            strings,
            digests,
            factors,
            flavour_flows,
            quantum_number_flows,
            coupling_orders,
            u32_sequences: Catalog::from_set(u32_values, "u32 sequence catalog")?,
            i32_sequences: Catalog::from_set(i32_values, "i32 sequence catalog")?,
        })
    }
}

fn string_id(catalog: &Catalog<String>, value: &str) -> RusticolResult<u32> {
    catalog.ids.get(value).copied().ok_or_else(|| {
        invalid(format!(
            "string {value:?} is absent from the canonical catalog"
        ))
    })
}

fn digest_id(catalog: &Catalog<[u8; 32]>, value: &str) -> RusticolResult<u32> {
    catalog.id(&parse_digest(value, "digest catalog lookup")?, "digest")
}

fn factor_id(catalog: &Catalog<FactorKey>, value: &ExactFactorJson) -> RusticolResult<u32> {
    catalog.id(&value.key()?, "exact factor")
}

fn optional_factor_id(
    catalog: &Catalog<FactorKey>,
    value: &RequiredOption<ExactFactorJson>,
) -> RusticolResult<u32> {
    match &value.0 {
        Some(value) => factor_id(catalog, value),
        None => Ok(MISSING_U32),
    }
}

fn optional_digest_id(
    catalog: &Catalog<[u8; 32]>,
    value: &RequiredOption<String>,
) -> RusticolResult<u32> {
    match &value.0 {
        Some(value) => digest_id(catalog, value),
        None => Ok(MISSING_U32),
    }
}

fn pack_sequences<T: Clone + Ord>(
    catalog: &Catalog<Vec<T>>,
    context: &str,
) -> RusticolResult<(Vec<IndexedRangeRow>, Vec<T>)> {
    let mut ranges = Vec::with_capacity(catalog.values.len());
    let mut flat = Vec::new();
    for (index, values) in catalog.values.iter().enumerate() {
        ranges.push(IndexedRangeRow {
            id: checked_u32(index, context)?,
            range: CheckedTableRange::new(
                checked_u64(flat.len(), context)?,
                checked_u64(values.len(), context)?,
            ),
        });
        flat.extend(values.iter().cloned());
    }
    Ok((ranges, flat))
}

fn project_catalog(catalog: &SemanticCatalogJson) -> RusticolResult<OwnedRecurrenceTemplateInput> {
    let ids = SectionIds::build(catalog)?;
    let primitive = PrimitiveCatalogs::build(catalog, &ids)?;

    let mut string_ranges = Vec::with_capacity(primitive.strings.values.len());
    let mut string_bytes = Vec::new();
    for value in &primitive.strings.values {
        let bytes = value.as_bytes();
        string_ranges.push(CheckedTableRange::new(
            checked_u64(string_bytes.len(), "string byte offset")?,
            checked_u64(bytes.len(), "string byte length")?,
        ));
        string_bytes.extend_from_slice(bytes);
    }

    let digest_catalog = primitive
        .digests
        .values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            Ok(DigestCatalogRow {
                id: checked_u32(index, "digest catalog row")?,
                value,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let exact_factors = primitive
        .factors
        .values
        .iter()
        .enumerate()
        .map(|(index, factor)| {
            Ok(ExactFactorRow {
                id: checked_u32(index, "exact-factor row")?,
                real_numerator_string_id: string_id(&primitive.strings, &factor.strings[0])?,
                real_denominator_string_id: string_id(&primitive.strings, &factor.strings[1])?,
                imag_numerator_string_id: string_id(&primitive.strings, &factor.strings[2])?,
                imag_denominator_string_id: string_id(&primitive.strings, &factor.strings[3])?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let (flavour_flow_ranges, flavour_flow_values) =
        pack_sequences(&primitive.flavour_flows, "flavour-flow")?;
    let (i32_sequence_ranges, i32_sequence_values) =
        pack_sequences(&primitive.i32_sequences, "i32 sequence")?;
    let (u32_sequence_ranges, u32_sequence_values) =
        pack_sequences(&primitive.u32_sequences, "u32 sequence")?;
    let (coupling_order_ranges, _) = pack_sequences(&primitive.coupling_orders, "coupling-order")?;
    let coupling_order_terms = primitive
        .coupling_orders
        .values
        .iter()
        .enumerate()
        .flat_map(|(set_index, values)| {
            values
                .iter()
                .map(move |(name, power)| (set_index, name, power))
        })
        .map(|(set_index, name, power)| {
            Ok(CouplingOrderTermRow {
                set_id: checked_u32(set_index, "coupling-order set")?,
                name_string_id: string_id(&primitive.strings, name)?,
                power: *power,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let (quantum_number_flow_ranges, _) =
        pack_sequences(&primitive.quantum_number_flows, "quantum-number flow")?;
    let quantum_number_flow_terms = primitive
        .quantum_number_flows
        .values
        .iter()
        .enumerate()
        .flat_map(|(flow_index, values)| {
            values
                .iter()
                .map(move |(name, expression)| (flow_index, name, expression))
        })
        .map(|(flow_index, name, expression)| {
            Ok(QuantumNumberFlowTermRow {
                flow_id: checked_u32(flow_index, "quantum-number flow")?,
                name_string_id: string_id(&primitive.strings, name)?,
                expression_string_id: string_id(&primitive.strings, expression)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let catalog_header = vec![CatalogHeaderRow {
        schema_version: RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
        abi_string_id: string_id(&primitive.strings, &catalog.header.abi)?,
        canonicalization_abi_string_id: string_id(
            &primitive.strings,
            &catalog.header.canonicalization_abi,
        )?,
        exact_scalar_abi_string_id: string_id(
            &primitive.strings,
            &catalog.header.exact_scalar_abi,
        )?,
        compiled_model_digest_id: digest_id(
            &primitive.digests,
            &catalog.header.compiled_model_digest,
        )?,
        prepared_kernel_pack_digest_id: digest_id(
            &primitive.digests,
            &catalog.header.prepared_kernel_pack_digest,
        )?,
        catalog_digest_id: digest_id(&primitive.digests, &catalog.header.catalog_digest)?,
        parameter_count: checked_u32(catalog.parameters.len(), "parameter count")?,
        current_state_count: checked_u32(catalog.current_states.len(), "current-state count")?,
        source_count: checked_u32(catalog.sources.len(), "source count")?,
        quantum_flow_count: checked_u32(catalog.quantum_flows.len(), "quantum-flow count")?,
        contact_orbit_certificate_count: checked_u32(
            catalog.contact_orbit_certificates.len(),
            "contact-orbit certificate count",
        )?,
        contact_orbit_step_count: checked_u32(
            catalog.contact_orbit_steps.len(),
            "contact-orbit step count",
        )?,
        transition_count: checked_u32(catalog.transitions.len(), "transition count")?,
        propagator_count: checked_u32(catalog.propagators.len(), "propagator count")?,
        closure_count: checked_u32(catalog.closures.len(), "closure count")?,
        color_contraction_count: checked_u32(
            catalog.color_contractions.len(),
            "color-contraction count",
        )?,
        symmetry_proof_count: checked_u32(catalog.symmetry_proofs.len(), "symmetry-proof count")?,
        runtime_helicity_contract_count: checked_u32(
            catalog.runtime_helicity_contracts.len(),
            "runtime-helicity contract count",
        )?,
        evaluator_binding_count: checked_u32(
            catalog.evaluator_bindings.len(),
            "evaluator-binding count",
        )?,
    }];

    let parameters = catalog
        .parameters
        .iter()
        .enumerate()
        .map(|(index, record)| {
            Ok(ParameterRow {
                id: checked_u32(index, "parameter row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                name_string_id: string_id(&primitive.strings, &record.name)?,
                kind: enum_id(
                    &record.parameter_kind,
                    &[("external", 0), ("derived", 1), ("constant", 2)],
                    "parameter kind",
                )?,
                value_type: enum_id(
                    &record.value_type,
                    &[("real", 0), ("complex", 1)],
                    "parameter value type",
                )?,
                mutable: u8::from(record.mutable),
                default_factor_id: optional_factor_id(&primitive.factors, &record.default_value)?,
                exact_expression_digest_id: optional_digest_id(
                    &primitive.digests,
                    &record.exact_expression_digest,
                )?,
                dependency_sequence_id: primitive.u32_sequences.id(
                    &mapped_ids(
                        &record.dependency_parameter_ids,
                        &ids.parameters,
                        "parameter dependency",
                    )?,
                    "parameter dependency sequence",
                )?,
                prepared_parameter_id: record.prepared_parameter_id.0.unwrap_or(MISSING_U32),
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let current_states = catalog
        .current_states
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let tensor_ordering = record
                .tensor_ordering
                .iter()
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(CurrentStateRow {
                id: checked_u32(index, "current-state row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                particle_id: record.particle_id,
                anti_particle_id: record.anti_particle_id,
                species_string_id: string_id(&primitive.strings, &record.species_id)?,
                orientation: enum_id(
                    &record.orientation,
                    &[("particle", 0), ("antiparticle", 1), ("self-conjugate", 2)],
                    "current orientation",
                )?,
                statistics: enum_id(
                    &record.statistics,
                    &[("boson", 0), ("fermion", 1)],
                    "particle statistics",
                )?,
                color_representation: record.color_representation,
                basis_string_id: string_id(&primitive.strings, &record.basis)?,
                tensor_ordering_sequence_id: primitive
                    .u32_sequences
                    .id(&tensor_ordering, "tensor-ordering sequence")?,
                dimension: record.dimension,
                chirality: record.chirality,
                lc_color_shape_string_id: string_id(
                    &primitive.strings,
                    &record.lc_color_shape_kind,
                )?,
                auxiliary_kind_string_id: optional_string_id(
                    &primitive.strings,
                    &record.auxiliary_kind,
                )?,
                mass_parameter_id: optional_id(
                    record.mass_parameter_id.0.as_ref(),
                    &ids.parameters,
                    "current mass parameter",
                )?,
                width_parameter_id: optional_id(
                    record.width_parameter_id.0.as_ref(),
                    &ids.parameters,
                    "current width parameter",
                )?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let sources = catalog
        .sources
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let provenance = record
                .lc_color_seed
                .provenance
                .iter()
                .flat_map(|pair| pair.iter())
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(SourceRow {
                id: checked_u32(index, "source row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                state_template_id: required_id(
                    &ids.current_states,
                    &record.state_template_id,
                    "source state",
                )?,
                crossing_string_id: string_id(&primitive.strings, &record.crossing)?,
                wavefunction_family_string_id: string_id(
                    &primitive.strings,
                    &record.wavefunction_family,
                )?,
                helicity: record.helicity,
                spin_state: record.spin_state,
                flavour_flow_id: primitive
                    .flavour_flows
                    .id(&record.flavour_flow, "source flavour flow")?,
                quantum_number_flow_id: primitive.quantum_number_flows.id(
                    &qflow(&record.quantum_number_flow),
                    "source quantum-number flow",
                )?,
                lc_color_seed_operation: enum_id(
                    &record.lc_color_seed.operation,
                    &[("empty", 0), ("singleton", 1)],
                    "LC source-seed operation",
                )?,
                lc_color_seed_shape_string_id: string_id(
                    &primitive.strings,
                    &record.lc_color_seed.output_shape_kind,
                )?,
                lc_color_seed_component_kind: optional_enum_id(
                    &record.lc_color_seed.component_kind,
                    &[("open-string", 0), ("adjoint-segment", 1), ("trace", 2)],
                    "LC source-seed component kind",
                    255,
                )?,
                lc_color_seed_component_role: enum_id(
                    &record.lc_color_seed.component_role,
                    &[("active", 0), ("passive", 1), ("none", 2)],
                    "LC source-seed component role",
                )?,
                lc_color_seed_proof_digest_id: digest_id(
                    &primitive.digests,
                    &record.lc_color_seed.proof_digest,
                )?,
                lc_color_seed_provenance_sequence_id: primitive
                    .u32_sequences
                    .id(&provenance, "LC source provenance sequence")?,
                wavefunction_expression_digest_id: digest_id(
                    &primitive.digests,
                    &record.wavefunction_expression_digest,
                )?,
                evaluator_binding_id: required_id(
                    &ids.evaluator_bindings,
                    &record.evaluator_resolver_key,
                    "source evaluator",
                )?,
                mass_parameter_id: optional_id(
                    record.mass_parameter_id.0.as_ref(),
                    &ids.parameters,
                    "source mass parameter",
                )?,
                width_parameter_id: optional_id(
                    record.width_parameter_id.0.as_ref(),
                    &ids.parameters,
                    "source width parameter",
                )?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let quantum_flows = catalog
        .quantum_flows
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let input_flavours = record
                .input_flavour_flows
                .iter()
                .map(|value| primitive.flavour_flows.id(value, "input flavour flow"))
                .collect::<RusticolResult<Vec<_>>>()?;
            let input_quantum = record
                .input_quantum_number_flows
                .iter()
                .map(|value| {
                    primitive
                        .quantum_number_flows
                        .id(&qflow(value), "input quantum-number flow")
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(QuantumFlowRow {
                id: checked_u32(index, "quantum-flow row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                input_state_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    mapped_ids(
                        &record.input_state_template_ids,
                        &ids.current_states,
                        "quantum-flow input state",
                    )?,
                    "quantum-flow input-state sequence",
                )?,
                input_spin_sequence_id: sequence_id(
                    &primitive.i32_sequences,
                    record.input_spin_states.clone(),
                    "quantum-flow spin sequence",
                )?,
                input_flavour_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    input_flavours,
                    "quantum-flow flavour sequence",
                )?,
                input_quantum_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    input_quantum,
                    "quantum-flow quantum-number sequence",
                )?,
                flavour_flow_operation_string_id: string_id(
                    &primitive.strings,
                    &record.flavour_flow_operation,
                )?,
                quantum_number_flow_operation_string_id: string_id(
                    &primitive.strings,
                    &record.quantum_number_flow_operation,
                )?,
                coupling_order_set_id: primitive
                    .coupling_orders
                    .id(&record.coupling_orders, "quantum-flow coupling orders")?,
                result_state_template_id: required_id(
                    &ids.current_states,
                    &record.result_state_template_id,
                    "quantum-flow result state",
                )?,
                result_spin_state: record.result_spin_state,
                result_flavour_flow_id: primitive
                    .flavour_flows
                    .id(&record.result_flavour_flow, "result flavour flow")?,
                result_quantum_number_flow_id: primitive.quantum_number_flows.id(
                    &qflow(&record.result_quantum_number_flow),
                    "result quantum-number flow",
                )?,
                exact_coupling_factor_id: factor_id(&primitive.factors, &record.exact_coupling)?,
                predicate_digest_id: digest_id(&primitive.digests, &record.predicate_digest)?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let contact_orbit_certificates = catalog
        .contact_orbit_certificates
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let particles = record
                .particles
                .iter()
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(ContactOrbitCertificateRow {
                id: checked_u32(index, "contact-orbit certificate row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                algorithm_string_id: string_id(&primitive.strings, &record.algorithm)?,
                algorithm_version: record.algorithm_version,
                term_id: record.term_id,
                vertex_string_id: string_id(&primitive.strings, &record.vertex)?,
                particle_string_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    particles,
                    "contact-orbit particle sequence",
                )?,
                evaluator_class_string_id: string_id(&primitive.strings, &record.evaluator_class)?,
                physical_leg_equivalence_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    record.physical_leg_equivalence_classes.to_vec(),
                    "contact-orbit equivalence sequence",
                )?,
                reconstruction_factor_id: factor_id(
                    &primitive.factors,
                    &record.reconstruction_factor,
                )?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let contact_orbit_steps = catalog
        .contact_orbit_steps
        .iter()
        .enumerate()
        .map(|(index, record)| {
            Ok(ContactOrbitStepRow {
                id: checked_u32(index, "contact-orbit step row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                certificate_id: required_id(
                    &ids.contact_orbit_certificates,
                    &record.certificate_template_id,
                    "contact-orbit certificate",
                )?,
                stage: enum_id(
                    &record.stage,
                    &[("partial", 0), ("final", 1)],
                    "contact-orbit stage",
                )?,
                result_leg: record.result_leg,
                left_covered_leg_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    record.left_covered_legs.clone(),
                    "contact-orbit left legs",
                )?,
                right_covered_leg_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    record.right_covered_legs.clone(),
                    "contact-orbit right legs",
                )?,
                source_particle_leg_sequence_id: sequence_id(
                    &primitive.i32_sequences,
                    record.source_particle_legs.to_vec(),
                    "contact-orbit source legs",
                )?,
                reconstruction_factor_id: factor_id(
                    &primitive.factors,
                    &record.reconstruction_factor,
                )?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let transitions = catalog
        .transitions
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let momentum = record
                .momentum_convention
                .iter()
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            let step_digests = record
                .contact_orbit_step_semantic_digests
                .iter()
                .map(|value| digest_id(&primitive.digests, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(TransitionRow {
                id: checked_u32(index, "transition row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                input_state_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    mapped_ids(
                        &record.input_state_template_ids,
                        &ids.current_states,
                        "transition input state",
                    )?,
                    "transition input-state sequence",
                )?,
                result_state_template_id: required_id(
                    &ids.current_states,
                    &record.result_state_template_id,
                    "transition result state",
                )?,
                quantum_flow_template_id: required_id(
                    &ids.quantum_flows,
                    &record.quantum_flow_template_id,
                    "transition quantum flow",
                )?,
                evaluator_binding_id: required_id(
                    &ids.evaluator_bindings,
                    &record.evaluator_resolver_key,
                    "transition evaluator",
                )?,
                canonical_input_order_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    record.canonical_input_order.clone(),
                    "transition canonical input order",
                )?,
                momentum_convention_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    momentum,
                    "transition momentum convention",
                )?,
                coupling_parameter_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    mapped_ids(
                        &record.coupling_parameter_ids,
                        &ids.parameters,
                        "transition coupling parameter",
                    )?,
                    "transition coupling parameters",
                )?,
                coupling_order_set_id: primitive
                    .coupling_orders
                    .id(&record.coupling_orders, "transition coupling orders")?,
                color_contraction_template_id: required_id(
                    &ids.color_contractions,
                    &record.color_contraction_template_id,
                    "transition color contraction",
                )?,
                binding_coupling_factor_id: factor_id(
                    &primitive.factors,
                    &record.binding_coupling,
                )?,
                exact_factor_id: factor_id(&primitive.factors, &record.exact_factor)?,
                output_factor_source: enum_id(
                    &record.output_factor_source,
                    &[("none", 0), ("coupling-real", 1), ("coupling-imag", 2)],
                    "transition output-factor source",
                )?,
                equivalence_class_string_id: string_id(
                    &primitive.strings,
                    &record.equivalence_class,
                )?,
                input_exchange_factor_id: optional_factor_id(
                    &primitive.factors,
                    &record.input_exchange_factor,
                )?,
                output_projection_string_id: string_id(
                    &primitive.strings,
                    &record.output_projection,
                )?,
                contact_orbit_step_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    mapped_ids(
                        &record.contact_orbit_step_template_ids,
                        &ids.contact_orbit_steps,
                        "transition contact-orbit step",
                    )?,
                    "transition contact-orbit steps",
                )?,
                contact_orbit_step_semantic_digest_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    step_digests,
                    "transition contact-orbit digests",
                )?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let propagators = catalog
        .propagators
        .iter()
        .enumerate()
        .map(|(index, record)| {
            Ok(PropagatorRow {
                id: checked_u32(index, "propagator row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                state_template_id: required_id(
                    &ids.current_states,
                    &record.state_template_id,
                    "propagator state",
                )?,
                applies_propagator: u8::from(record.applies_propagator),
                evaluator_binding_id: optional_id(
                    record.evaluator_resolver_key.0.as_ref(),
                    &ids.evaluator_bindings,
                    "propagator evaluator",
                )?,
                numerator_expression_digest_id: optional_digest_id(
                    &primitive.digests,
                    &record.numerator_expression_digest,
                )?,
                denominator_expression_digest_id: optional_digest_id(
                    &primitive.digests,
                    &record.denominator_expression_digest,
                )?,
                mass_parameter_id: optional_id(
                    record.mass_parameter_id.0.as_ref(),
                    &ids.parameters,
                    "propagator mass parameter",
                )?,
                width_parameter_id: optional_id(
                    record.width_parameter_id.0.as_ref(),
                    &ids.parameters,
                    "propagator width parameter",
                )?,
                gauge_string_id: optional_string_id(&primitive.strings, &record.gauge)?,
                linearity_proof_template_id: optional_id(
                    record.linearity_proof_template_id.0.as_ref(),
                    &ids.symmetry_proofs,
                    "propagator linearity proof",
                )?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let closures = catalog
        .closures
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let coefficients = record
                .component_coefficients
                .iter()
                .map(|value| factor_id(&primitive.factors, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(ClosureRow {
                id: checked_u32(index, "closure row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                input_state_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    mapped_ids(
                        &record.input_state_template_ids,
                        &ids.current_states,
                        "closure input state",
                    )?,
                    "closure input states",
                )?,
                result_state_template_id: optional_id(
                    record.result_state_template_id.0.as_ref(),
                    &ids.current_states,
                    "closure result state",
                )?,
                evaluator_binding_id: required_id(
                    &ids.evaluator_bindings,
                    &record.evaluator_resolver_key,
                    "closure evaluator",
                )?,
                canonical_input_order_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    record.canonical_input_order.clone(),
                    "closure canonical input order",
                )?,
                coupling_parameter_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    mapped_ids(
                        &record.coupling_parameter_ids,
                        &ids.parameters,
                        "closure coupling parameter",
                    )?,
                    "closure coupling parameters",
                )?,
                coupling_order_set_id: primitive
                    .coupling_orders
                    .id(&record.coupling_orders, "closure coupling orders")?,
                eligible_quantum_flow_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    mapped_ids(
                        &record.eligible_quantum_flow_template_ids,
                        &ids.quantum_flows,
                        "closure quantum flow",
                    )?,
                    "closure eligible quantum flows",
                )?,
                color_contraction_template_id: required_id(
                    &ids.color_contractions,
                    &record.color_contraction_template_id,
                    "closure color contraction",
                )?,
                binding_coupling_factor_id: factor_id(
                    &primitive.factors,
                    &record.binding_coupling,
                )?,
                exact_factor_id: factor_id(&primitive.factors, &record.exact_factor)?,
                output_factor_source: enum_id(
                    &record.output_factor_source,
                    &[("none", 0), ("coupling-real", 1), ("coupling-imag", 2)],
                    "closure output-factor source",
                )?,
                equivalence_class_string_id: string_id(
                    &primitive.strings,
                    &record.equivalence_class,
                )?,
                input_exchange_factor_id: optional_factor_id(
                    &primitive.factors,
                    &record.input_exchange_factor,
                )?,
                projection_string_id: string_id(&primitive.strings, &record.projection)?,
                component_coefficient_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    coefficients,
                    "closure component coefficients",
                )?,
                chirality_relation_string_id: string_id(
                    &primitive.strings,
                    &record.chirality_relation,
                )?,
                metric_signature_string_id: optional_string_id(
                    &primitive.strings,
                    &record.metric_signature,
                )?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let mut color_contractions = Vec::with_capacity(catalog.color_contractions.len());
    let mut lc_color_transition_witnesses = Vec::new();
    let mut color_nc_terms = Vec::new();
    for (color_index, record) in catalog.color_contractions.iter().enumerate() {
        let color_id = checked_u32(color_index, "color-contraction row")?;
        let witness_start =
            checked_u64(lc_color_transition_witnesses.len(), "color witness offset")?;
        let nc_term_start = checked_u64(color_nc_terms.len(), "color Nc offset")?;
        for (ordinal, witness) in record.transition_witnesses.iter().enumerate() {
            let (pairings, bindings) = witness.canonical_ports()?;
            let flattened_pairings = pairings
                .iter()
                .flat_map(|pairing| pairing.iter())
                .flat_map(|port| port.iter().copied())
                .collect::<Vec<_>>();
            let flattened_bindings = bindings
                .iter()
                .flat_map(|port| port.iter().copied())
                .collect::<Vec<_>>();
            let provenance = witness
                .provenance
                .iter()
                .flat_map(|pair| pair.iter())
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            let input_permutation = match witness.input_permutation {
                [0, 1] => 0,
                [1, 0] => 1,
                value => {
                    return Err(invalid(format!(
                        "unsupported LC witness input permutation {value:?}"
                    )));
                }
            };
            lc_color_transition_witnesses.push(LCColorTransitionWitnessRow {
                color_contraction_id: color_id,
                ordinal: checked_u32(ordinal, "color witness ordinal")?,
                left_shape_string_id: string_id(&primitive.strings, &witness.input_shape_kinds[0])?,
                right_shape_string_id: string_id(
                    &primitive.strings,
                    &witness.input_shape_kinds[1],
                )?,
                input_permutation,
                reverse_parent_mask: witness.reverse_parent_mask,
                component_operation: enum_id(
                    &witness.component_operation,
                    &[
                        ("concatenate-join", 0),
                        ("concatenate-keep", 1),
                        ("inherit-left", 2),
                        ("inherit-right", 3),
                        ("empty", 4),
                        ("close", 5),
                    ],
                    "LC witness component operation",
                )?,
                result_component_kind: optional_enum_id(
                    &witness.result_component_kind,
                    &[("open-string", 0), ("adjoint-segment", 1), ("trace", 2)],
                    "LC witness component kind",
                    255,
                )?,
                result_component_role: enum_id(
                    &witness.result_component_role,
                    &[("active", 0), ("passive", 1), ("none", 2)],
                    "LC witness component role",
                )?,
                result_shape_string_id: optional_string_id(
                    &primitive.strings,
                    &witness.result_shape_kind,
                )?,
                exact_factor_id: factor_id(&primitive.factors, &witness.exact_factor)?,
                proof_digest_id: digest_id(&primitive.digests, &witness.proof_digest)?,
                input_port_pairing_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    flattened_pairings,
                    "LC witness port pairings",
                )?,
                result_port_binding_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    flattened_bindings,
                    "LC witness result bindings",
                )?,
                provenance_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    provenance,
                    "LC witness provenance",
                )?,
            });
        }
        for (exponent, factor) in &record.nc_polynomial {
            color_nc_terms.push(ColorNcTermRow {
                color_contraction_id: color_id,
                exponent: *exponent,
                factor_id: factor_id(&primitive.factors, factor)?,
            });
        }
        color_contractions.push(ColorContractionRow {
            id: color_id,
            template_string_id: string_id(&primitive.strings, &record.template_id)?,
            rule_kind_string_id: string_id(&primitive.strings, &record.rule_kind)?,
            input_representation_sequence_id: sequence_id(
                &primitive.i32_sequences,
                record.input_representations.clone(),
                "color input representations",
            )?,
            has_output_representation: u8::from(record.output_representation.0.is_some()),
            output_representation: record.output_representation.0.unwrap_or(0),
            ordered_open_string_arity: record.ordered_open_string_arity,
            exact_coefficient_factor_id: factor_id(&primitive.factors, &record.exact_coefficient)?,
            witness_start,
            witness_count: checked_u64(record.transition_witnesses.len(), "color witness count")?,
            nc_term_start,
            nc_term_count: checked_u64(record.nc_polynomial.len(), "color Nc count")?,
            expression_digest_id: digest_id(&primitive.digests, &record.expression_digest)?,
            semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
        });
    }

    let symmetry_proofs = catalog
        .symmetry_proofs
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let subjects = record
                .subject_template_ids
                .iter()
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            let expressions = record
                .expression_digests
                .iter()
                .map(|value| digest_id(&primitive.digests, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(SymmetryProofRow {
                id: checked_u32(index, "symmetry-proof row")?,
                template_string_id: string_id(&primitive.strings, &record.template_id)?,
                proof_algorithm_string_id: string_id(&primitive.strings, &record.proof_algorithm)?,
                subject_template_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    subjects,
                    "symmetry-proof subjects",
                )?,
                input_permutation_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    record.input_permutation.clone(),
                    "symmetry-proof input permutation",
                )?,
                exact_phase_factor_id: factor_id(&primitive.factors, &record.exact_phase)?,
                expression_digest_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    expressions,
                    "symmetry-proof expressions",
                )?,
                witness_digest_id: digest_id(&primitive.digests, &record.witness_digest)?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let mut runtime_helicity_contracts = Vec::new();
    let mut runtime_helicity_variants = Vec::new();
    let mut runtime_helicity_embeddings = Vec::new();
    let mut runtime_helicity_projections = Vec::new();
    for (contract_index, record) in catalog.runtime_helicity_contracts.iter().enumerate() {
        let contract_id = checked_u32(contract_index, "runtime-helicity contract")?;
        let variant_start = checked_u64(
            runtime_helicity_variants.len(),
            "runtime-helicity variant offset",
        )?;
        for variant in &record.variants {
            let variant_id =
                checked_u32(runtime_helicity_variants.len(), "runtime-helicity variant")?;
            let embedding_start = checked_u64(
                runtime_helicity_embeddings.len(),
                "runtime-helicity embedding offset",
            )?;
            for (full_component, (source_component, factor)) in variant
                .embedding_source_components
                .iter()
                .zip(&variant.embedding_factors)
                .enumerate()
            {
                runtime_helicity_embeddings.push(RuntimeHelicityEmbeddingRow {
                    variant_id,
                    full_component: checked_u32(full_component, "full helicity component")?,
                    source_component: source_component.unwrap_or(MISSING_U32),
                    factor_id: factor_id(&primitive.factors, factor)?,
                });
            }
            let projection_start = checked_u64(
                runtime_helicity_projections.len(),
                "runtime-helicity projection offset",
            )?;
            for (source_component, full_component) in variant
                .projection_full_components
                .iter()
                .copied()
                .enumerate()
            {
                runtime_helicity_projections.push(RuntimeHelicityProjectionRow {
                    variant_id,
                    source_component: checked_u32(source_component, "source helicity component")?,
                    full_component,
                });
            }
            runtime_helicity_variants.push(RuntimeHelicityVariantRow {
                id: variant_id,
                contract_id,
                source_template_id: required_id(
                    &ids.sources,
                    &variant.source_template_id,
                    "runtime-helicity source",
                )?,
                source_state_template_id: required_id(
                    &ids.current_states,
                    &variant.source_state_template_id,
                    "runtime-helicity source state",
                )?,
                embedding_range: CheckedTableRange::new(
                    embedding_start,
                    checked_u64(
                        variant.embedding_source_components.len(),
                        "runtime-helicity embedding count",
                    )?,
                ),
                projection_range: CheckedTableRange::new(
                    projection_start,
                    checked_u64(
                        variant.projection_full_components.len(),
                        "runtime-helicity projection count",
                    )?,
                ),
                proof_digest_id: digest_id(&primitive.digests, &variant.proof_digest)?,
            });
        }
        runtime_helicity_contracts.push(RuntimeHelicityContractRow {
            id: contract_id,
            template_string_id: string_id(&primitive.strings, &record.template_id)?,
            full_state_template_id: required_id(
                &ids.current_states,
                &record.full_state_template_id,
                "runtime-helicity full state",
            )?,
            variant_range: CheckedTableRange::new(
                variant_start,
                checked_u64(record.variants.len(), "runtime-helicity variant count")?,
            ),
            proof_algorithm_string_id: string_id(&primitive.strings, &record.proof_algorithm)?,
            proof_digest_id: digest_id(&primitive.digests, &record.proof_digest)?,
            semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
        });
    }

    let evaluator_bindings = catalog
        .evaluator_bindings
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let input_layout = record
                .input_layout
                .iter()
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            let output_layout = record
                .output_layout
                .iter()
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            let expressions = record
                .exact_expression_digests
                .iter()
                .map(|value| digest_id(&primitive.digests, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            let semantic_templates = record
                .semantic_template_ids
                .iter()
                .map(|value| string_id(&primitive.strings, value))
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(EvaluatorBindingRow {
                id: checked_u32(index, "evaluator-binding row")?,
                resolver_key_string_id: string_id(&primitive.strings, &record.resolver_key)?,
                prepared_kernel_id: record.prepared_kernel_id.0.unwrap_or(MISSING_U32),
                contract_kind: enum_id(
                    &record.contract_kind,
                    &[
                        ("source", 0),
                        ("vertex", 1),
                        ("propagator", 2),
                        ("closure", 3),
                        ("model-parameter", 4),
                    ],
                    "evaluator contract kind",
                )?,
                callable_signature_digest_id: digest_id(
                    &primitive.digests,
                    &record.callable_signature,
                )?,
                input_state_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    mapped_ids(
                        &record.input_state_template_ids,
                        &ids.current_states,
                        "evaluator input state",
                    )?,
                    "evaluator input states",
                )?,
                output_state_template_id: optional_id(
                    record.output_state_template_id.0.as_ref(),
                    &ids.current_states,
                    "evaluator output state",
                )?,
                input_layout_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    input_layout,
                    "evaluator input layout",
                )?,
                output_layout_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    output_layout,
                    "evaluator output layout",
                )?,
                exact_expression_digest_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    expressions,
                    "evaluator expressions",
                )?,
                semantic_template_sequence_id: sequence_id(
                    &primitive.u32_sequences,
                    semantic_templates,
                    "evaluator semantic templates",
                )?,
                callable_kind: enum_id(
                    &record.callable_kind,
                    &[("prepared-kernel", 0), ("rusticol-template", 1)],
                    "evaluator callable kind",
                )?,
                runtime_template_string_id: optional_string_id(
                    &primitive.strings,
                    &record.runtime_template,
                )?,
                semantic_digest_id: digest_id(&primitive.digests, &record.semantic_digest)?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let input = OwnedRecurrenceTemplateInput {
        input_abi: RECURRENCE_TEMPLATE_INPUT_ABI.to_owned(),
        catalog_digest: semantic_digest(&catalog.header.catalog_digest, "catalog digest")?,
        compiled_model_digest: semantic_digest(
            &catalog.header.compiled_model_digest,
            "compiled-model digest",
        )?,
        prepared_kernel_pack_digest: semantic_digest(
            &catalog.header.prepared_kernel_pack_digest,
            "prepared-kernel-pack digest",
        )?,
        catalog_header,
        coupling_order_ranges,
        coupling_order_terms,
        contact_orbit_certificates,
        contact_orbit_steps,
        current_states,
        digest_catalog,
        evaluator_bindings,
        exact_factors,
        flavour_flow_ranges,
        flavour_flow_values,
        i32_sequence_ranges,
        i32_sequence_values,
        parameters,
        propagators,
        quantum_flows,
        quantum_number_flow_ranges,
        quantum_number_flow_terms,
        runtime_helicity_contracts,
        runtime_helicity_variants,
        runtime_helicity_embeddings,
        runtime_helicity_projections,
        sources,
        string_ranges,
        string_bytes,
        symmetry_proofs,
        transitions,
        closures,
        color_contractions,
        lc_color_transition_witnesses,
        color_nc_terms,
        u32_sequence_ranges,
        u32_sequence_values,
    };
    input.as_view().validate()?;
    Ok(input)
}

fn required_id(ids: &BTreeMap<String, u32>, value: &str, context: &str) -> RusticolResult<u32> {
    ids.get(value)
        .copied()
        .ok_or_else(|| invalid(format!("unknown {context} {value:?}")))
}

fn optional_string_id(
    catalog: &Catalog<String>,
    value: &RequiredOption<String>,
) -> RusticolResult<u32> {
    match &value.0 {
        Some(value) => string_id(catalog, value),
        None => Ok(MISSING_U32),
    }
}

fn optional_enum_id(
    value: &RequiredOption<String>,
    choices: &[(&str, u8)],
    context: &str,
    missing: u8,
) -> RusticolResult<u8> {
    match &value.0 {
        Some(value) => enum_id(value, choices, context),
        None => Ok(missing),
    }
}

fn sequence_id<T: Clone + Ord>(
    catalog: &Catalog<Vec<T>>,
    values: Vec<T>,
    context: &str,
) -> RusticolResult<u32> {
    catalog.id(&values, context)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{
        ROOT_SECTIONS, canonical_json_digest, project_owned_recurrence_template_catalog_json_v1,
        project_recurrence_template_catalog_json_v1, write_canonical_json,
    };

    const COMPILED_DIGEST: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    const PACK_DIGEST: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const EMPTY_CATALOG_DIGEST: &str =
        "93c542973b80e35e8300b4c25182cb0900040f9893dc6671b4af628062b9fbec";

    fn empty_catalog() -> Value {
        json!({
            "header": {
                "abi": "pyamplicol-recurrence-template-v1",
                "canonicalization_abi": "pyamplicol-canonical-json-v1",
                "compiled_model_digest": COMPILED_DIGEST,
                "exact_scalar_abi": "pyamplicol-exact-complex-rational-v1",
                "prepared_kernel_pack_digest": PACK_DIGEST,
                "catalog_digest": EMPTY_CATALOG_DIGEST,
            },
            "parameters": [],
            "current_states": [],
            "sources": [],
            "quantum_flows": [],
            "contact_orbit_certificates": [],
            "contact_orbit_steps": [],
            "transitions": [],
            "propagators": [],
            "closures": [],
            "color_contractions": [],
            "symmetry_proofs": [],
            "runtime_helicity_contracts": [],
            "evaluator_bindings": [],
        })
    }

    fn refresh_catalog_digest(value: &mut Value) {
        let mut payload = value.clone();
        payload["header"]
            .as_object_mut()
            .expect("test header")
            .remove("catalog_digest");
        let digest = canonical_json_digest(&payload).expect("canonical test digest");
        value["header"]["catalog_digest"] = Value::String(digest);
    }

    #[test]
    fn project_empty_catalog_matches_known_python_contract() {
        let input = project_recurrence_template_catalog_json_v1(&empty_catalog())
            .expect("known Python empty catalog must project");
        assert_eq!(input.input_abi, "pyamplicol-recurrence-template-input-v1");
        assert_eq!(input.catalog_header.len(), 1);
        assert_eq!(input.catalog_header[0].parameter_count, 0);
        assert_eq!(input.catalog_header[0].evaluator_binding_count, 0);
        assert_eq!(input.string_ranges.len(), 4);
        assert_eq!(input.digest_catalog.len(), 3);
        assert_eq!(input.coupling_order_ranges.len(), 1);
        assert_eq!(input.u32_sequence_ranges.len(), 1);
        assert_eq!(input.i32_sequence_ranges.len(), 1);
        assert!(input.exact_factors.is_empty());
        input
            .clone()
            .validate()
            .expect("projected empty catalog must satisfy core validation");
    }

    #[test]
    fn owned_projection_preserves_the_authenticated_contract() {
        let catalog = empty_catalog();
        let retained = project_recurrence_template_catalog_json_v1(&catalog)
            .expect("retained catalog must project")
            .validate()
            .expect("retained projection must validate");
        let owned = project_owned_recurrence_template_catalog_json_v1(catalog)
            .expect("owned catalog must project")
            .validate()
            .expect("owned projection must validate");
        assert_eq!(owned.summary(), retained.summary());

        let mut stale = empty_catalog();
        stale["header"]["catalog_digest"] = Value::String(COMPILED_DIGEST.to_owned());
        let error = project_owned_recurrence_template_catalog_json_v1(stale)
            .expect_err("owned stale catalog must fail closed");
        assert!(
            error
                .to_string()
                .contains("stale recurrence-template catalog digest"),
            "unexpected validation error: {error}"
        );
    }

    #[test]
    fn root_inventory_is_exact_and_all_thirteen_sections_are_required() {
        assert_eq!(ROOT_SECTIONS.len(), 13);

        let mut unknown = empty_catalog();
        unknown["opaque"] = json!([]);
        refresh_catalog_digest(&mut unknown);
        let error = project_recurrence_template_catalog_json_v1(&unknown)
            .expect_err("unknown root field must fail closed");
        assert!(
            error.to_string().contains("unknown field `opaque`"),
            "unexpected validation error: {error}"
        );

        for (section, _, _) in ROOT_SECTIONS {
            let mut missing = empty_catalog();
            missing
                .as_object_mut()
                .expect("test catalog")
                .remove(section);
            let error = project_recurrence_template_catalog_json_v1(&missing)
                .expect_err("missing semantic section must fail closed");
            assert!(
                error.to_string().contains(section),
                "missing {section} produced unexpected validation error: {error}"
            );
        }
    }

    #[test]
    fn stale_catalog_digest_fails_before_projection() {
        let mut catalog = empty_catalog();
        catalog["header"]["catalog_digest"] = Value::String(COMPILED_DIGEST.to_owned());
        let error = project_recurrence_template_catalog_json_v1(&catalog)
            .expect_err("stale catalog digest must fail closed");
        assert!(
            error
                .to_string()
                .contains("stale recurrence-template catalog digest"),
            "unexpected validation error: {error}"
        );
    }

    #[test]
    fn canonical_json_matches_python_ascii_contract() {
        let value = json!({"z": "é\n", "a": [true, null, "λ", -2]});
        let mut output = String::new();
        write_canonical_json(&value, &mut output).expect("canonical writer");
        assert_eq!(
            output,
            "{\"a\":[true,null,\"\\u03bb\",-2],\"z\":\"\\u00e9\\n\"}"
        );
    }
}
