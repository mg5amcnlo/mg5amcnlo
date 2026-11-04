// SPDX-License-Identifier: 0BSD

//! Checked process-wide input for the compact recurrence builder.
//!
//! The Python producer encodes `pyamplicol-recurrence-builder-input-v2` as
//! little-endian structure-of-arrays tables.  A boundary decoder can zip those
//! columns into the fixed-width row types below and then construct
//! [`OwnedRecurrenceProcessInput`].  This module validates process semantics
//! and exposes typed template references and semantic identities for a later
//! composite process-plus-template authentication step.  It deliberately does
//! not construct recurrence states or an execution schedule.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;
use sha2::{Digest, Sha256};

use super::{
    CheckedTableRange, ExactComplexRational, MultiwordMaskCatalogView,
    RECURRENCE_BUILDER_INPUT_ABI, RecurrenceStrategy, SemanticDigest, checked_u32_len,
    validate_concrete_spin_state, validate_packed_ranges, validate_u32_references,
};
use crate::{RusticolError, RusticolResult};

pub const RECURRENCE_PROCESS_INPUT_SCHEMA_VERSION: u32 = 1;
pub const RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI: &str =
    "pyamplicol-recurrence-fermion-pairing-columnar-v1";
pub const RECURRENCE_FERMION_PAIRING_SCHEMA_VERSION: u32 = 1;
pub const RECURRENCE_NO_FERMION_LINE: u32 = u32::MAX;

const RECURRENCE_FERMION_PAIRING_PROOF_ALGORITHM: &str = "canonical-external-fermion-pairing-v1";

const MISSING_ID: u32 = u32::MAX;
const REQUIRED_DIGEST_ROLE_COUNT: usize = 4;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

struct FermionPairingCatalogs<'a> {
    strings: Vec<&'a str>,
}

impl<'a> FermionPairingInputView<'a> {
    fn validate_internal(
        self,
        expected_process_id: &str,
    ) -> RusticolResult<FermionPairingValidationSummary> {
        if self.input_abi != RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported fermion-pairing columnar ABI {:?}; expected {:?}",
                self.input_abi, RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI
            )));
        }
        if self.header.len() != 1 {
            return Err(invalid(format!(
                "fermion-pairing header must contain one row, found {}",
                self.header.len()
            )));
        }
        if self.rules.is_empty() {
            return Err(invalid("fermion-pairing catalog has no rules"));
        }

        let catalogs = self.validate_catalogs()?;
        let header = self.header[0];
        if header.schema_version != RECURRENCE_FERMION_PAIRING_SCHEMA_VERSION {
            return Err(RusticolError::compatibility(format!(
                "unsupported fermion-pairing schema {}; expected {}",
                header.schema_version, RECURRENCE_FERMION_PAIRING_SCHEMA_VERSION
            )));
        }
        require_string_value(
            &catalogs.strings,
            header.abi_string_id,
            RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI,
            "fermion-pairing ABI",
        )?;
        require_string_value(
            &catalogs.strings,
            header.proof_algorithm_string_id,
            RECURRENCE_FERMION_PAIRING_PROOF_ALGORITHM,
            "fermion-pairing proof algorithm",
        )?;
        let process_key = required_string(
            &catalogs.strings,
            header.process_key_string_id,
            "fermion-pairing process key",
        )?;
        if process_key != expected_process_id {
            return Err(invalid(format!(
                "fermion-pairing process key {process_key:?} does not match {expected_process_id:?}"
            )));
        }
        self.validate_counts_and_ranges(header)?;
        self.validate_endpoints(&catalogs, header)?;
        self.validate_classes(&catalogs, header)?;
        self.validate_exact_integers()?;
        self.validate_rules(&catalogs, header)?;

        let topology_digest = SemanticDigest::new(header.topology_digest).map_err(|error| {
            invalid(format!(
                "fermion-pairing topology digest is invalid: {error}"
            ))
        })?;
        let semantic_digest = SemanticDigest::new(header.semantic_digest).map_err(|error| {
            invalid(format!(
                "fermion-pairing semantic digest is invalid: {error}"
            ))
        })?;
        Ok(FermionPairingValidationSummary {
            columnar_digest: self.declared_columnar_digest,
            topology_digest,
            semantic_digest,
            source_count: header.source_count,
            endpoint_count: header.endpoint_count,
            pairing_class_count: header.pairing_class_count,
            rule_count: header.rule_count,
        })
    }

    fn validate_catalogs(self) -> RusticolResult<FermionPairingCatalogs<'a>> {
        validate_packed_ranges(
            "fermion-pairing strings",
            self.string_ranges,
            self.string_bytes.len(),
        )?;
        let mut strings = Vec::with_capacity(self.string_ranges.len());
        let mut previous: Option<&[u8]> = None;
        for (index, range) in self.string_ranges.iter().copied().enumerate() {
            let bytes = &self.string_bytes[range.as_usize_range(
                self.string_bytes.len(),
                &format!("fermion-pairing string {index}"),
            )?];
            if bytes.is_empty() || previous.is_some_and(|value| value >= bytes) {
                return Err(invalid(format!(
                    "fermion-pairing string row {index} is empty or not canonical"
                )));
            }
            let value = std::str::from_utf8(bytes).map_err(|error| {
                invalid(format!(
                    "fermion-pairing string row {index} is not UTF-8: {error}"
                ))
            })?;
            previous = Some(bytes);
            strings.push(value);
        }
        Ok(FermionPairingCatalogs { strings })
    }

    fn validate_counts_and_ranges(self, header: FermionPairingHeaderRow) -> RusticolResult<()> {
        for (label, declared, actual) in [
            ("endpoints", header.endpoint_count, self.endpoints.len()),
            (
                "pairing classes",
                header.pairing_class_count,
                self.pairing_classes.len(),
            ),
            ("rules", header.rule_count, self.rules.len()),
            (
                "exact integers",
                header.exact_integer_count,
                self.exact_integers.len(),
            ),
            ("strings", header.string_count, self.string_ranges.len()),
        ] {
            if declared != checked_u32_len(actual, label)? {
                return Err(invalid(format!(
                    "fermion-pairing header {label} count does not match its table"
                )));
            }
        }
        for (label, declared, actual) in [
            (
                "endpoint state IDs",
                header.endpoint_state_template_count,
                self.endpoint_state_template_ids.len(),
            ),
            (
                "endpoint anti-state IDs",
                header.endpoint_anti_state_template_count,
                self.endpoint_anti_state_template_ids.len(),
            ),
            (
                "endpoint basis IDs",
                header.endpoint_basis_count,
                self.endpoint_basis_ids.len(),
            ),
            (
                "endpoint color representations",
                header.endpoint_color_representation_count,
                self.endpoint_color_representations.len(),
            ),
            (
                "class fundamental slots",
                header.class_fundamental_slot_count,
                self.class_fundamental_slots.len(),
            ),
            (
                "class antifundamental slots",
                header.class_antifundamental_slot_count,
                self.class_antifundamental_slots.len(),
            ),
            (
                "class reference pairings",
                header.class_reference_pairing_count,
                self.class_reference_pairings.len(),
            ),
            (
                "rule class indices",
                header.rule_class_pairing_index_count,
                self.rule_class_pairing_indices.len(),
            ),
            (
                "rule endpoint pairings",
                header.rule_endpoint_pairing_count,
                self.rule_endpoint_pairings.len(),
            ),
            (
                "rule source permutations",
                header.rule_source_permutation_count,
                self.rule_source_slot_permutations.len(),
            ),
            (
                "rule lineages",
                header.rule_lineage_count,
                self.rule_lineages.len(),
            ),
            (
                "exact-integer limbs",
                header.exact_integer_limb_count,
                self.exact_integer_limbs.len(),
            ),
            (
                "string bytes",
                header.string_byte_count,
                self.string_bytes.len(),
            ),
        ] {
            let actual = u64::try_from(actual)
                .map_err(|_| invalid(format!("fermion-pairing {label} count exceeds u64")))?;
            if declared != actual {
                return Err(invalid(format!(
                    "fermion-pairing header {label} count does not match its table"
                )));
            }
        }
        if header.no_fermion_line != RECURRENCE_NO_FERMION_LINE {
            return Err(invalid("fermion-pairing lineage sentinel is stale"));
        }
        validate_pairing_ranges(
            "endpoint state IDs",
            self.endpoints.iter().map(|row| row.state_template_range),
            self.endpoint_state_template_ids.len(),
        )?;
        validate_pairing_ranges(
            "endpoint anti-state IDs",
            self.endpoints
                .iter()
                .map(|row| row.anti_state_template_range),
            self.endpoint_anti_state_template_ids.len(),
        )?;
        validate_pairing_ranges(
            "endpoint basis IDs",
            self.endpoints.iter().map(|row| row.basis_range),
            self.endpoint_basis_ids.len(),
        )?;
        validate_pairing_ranges(
            "endpoint color representations",
            self.endpoints
                .iter()
                .map(|row| row.color_representation_range),
            self.endpoint_color_representations.len(),
        )?;
        validate_pairing_ranges(
            "class fundamental slots",
            self.pairing_classes
                .iter()
                .map(|row| row.fundamental_slot_range),
            self.class_fundamental_slots.len(),
        )?;
        validate_pairing_ranges(
            "class antifundamental slots",
            self.pairing_classes
                .iter()
                .map(|row| row.antifundamental_slot_range),
            self.class_antifundamental_slots.len(),
        )?;
        validate_pairing_ranges(
            "class reference pairings",
            self.pairing_classes
                .iter()
                .map(|row| row.reference_pairing_range),
            self.class_reference_pairings.len(),
        )?;
        validate_pairing_ranges(
            "rule class indices",
            self.rules.iter().map(|row| row.class_pairing_index_range),
            self.rule_class_pairing_indices.len(),
        )?;
        validate_pairing_ranges(
            "rule endpoint pairings",
            self.rules.iter().map(|row| row.endpoint_pairing_range),
            self.rule_endpoint_pairings.len(),
        )?;
        validate_pairing_ranges(
            "rule source permutations",
            self.rules.iter().map(|row| row.source_permutation_range),
            self.rule_source_slot_permutations.len(),
        )?;
        validate_pairing_ranges(
            "rule lineages",
            self.rules.iter().map(|row| row.lineage_range),
            self.rule_lineages.len(),
        )?;
        validate_pairing_ranges(
            "exact-integer limbs",
            self.exact_integers.iter().map(|row| row.limb_range),
            self.exact_integer_limbs.len(),
        )
    }

    fn validate_endpoints(
        self,
        catalogs: &FermionPairingCatalogs<'_>,
        header: FermionPairingHeaderRow,
    ) -> RusticolResult<()> {
        validate_canonical_ids(
            "fermion-pairing endpoint IDs",
            self.endpoints.iter().map(|row| row.endpoint_id),
        )?;
        let mut previous_slot = None;
        for (index, endpoint) in self.endpoints.iter().copied().enumerate() {
            required_reference(
                endpoint.source_slot,
                header.source_count as usize,
                "fermion endpoint source",
            )?;
            if previous_slot.is_some_and(|value| value >= endpoint.source_slot) {
                return Err(invalid(format!(
                    "fermion endpoints are not in source-slot order at row {index}"
                )));
            }
            previous_slot = Some(endpoint.source_slot);
            required_reference(
                endpoint.species_class_id,
                self.pairing_classes.len(),
                "fermion endpoint species class",
            )?;
            required_string(
                &catalogs.strings,
                endpoint.species_string_id,
                "fermion endpoint species",
            )?;
            if !matches!(endpoint.particle_orientation, 0 | 1)
                || !matches!(endpoint.color_orientation, 0 | 1)
            {
                return Err(invalid(format!(
                    "fermion endpoint {index} has an invalid orientation"
                )));
            }
            SemanticDigest::new(endpoint.contract_digest).map_err(|error| {
                invalid(format!(
                    "fermion endpoint {index} digest is invalid: {error}"
                ))
            })?;
            for (ids, range, label) in [
                (
                    self.endpoint_state_template_ids,
                    endpoint.state_template_range,
                    "endpoint state IDs",
                ),
                (
                    self.endpoint_anti_state_template_ids,
                    endpoint.anti_state_template_range,
                    "endpoint anti-state IDs",
                ),
                (
                    self.endpoint_basis_ids,
                    endpoint.basis_range,
                    "endpoint basis IDs",
                ),
            ] {
                let ids = pairing_range_slice(ids, range, label)?;
                if ids.is_empty() {
                    return Err(invalid(format!(
                        "fermion endpoint {index} has empty {label}"
                    )));
                }
                validate_strict_u32_values(ids, catalogs.strings.len(), label)?;
            }
            let representations = pairing_range_slice(
                self.endpoint_color_representations,
                endpoint.color_representation_range,
                "endpoint color representations",
            )?;
            if representations.is_empty() {
                return Err(invalid(format!(
                    "fermion endpoint {index} has no color representation"
                )));
            }
            validate_strict_i32_values(representations, "endpoint color representations")?;
        }
        Ok(())
    }

    fn validate_classes(
        self,
        catalogs: &FermionPairingCatalogs<'_>,
        header: FermionPairingHeaderRow,
    ) -> RusticolResult<()> {
        validate_canonical_ids(
            "fermion-pairing class IDs",
            self.pairing_classes.iter().map(|row| row.class_id),
        )?;
        let mut species = BTreeSet::new();
        for (index, class) in self.pairing_classes.iter().copied().enumerate() {
            if class.species_class_id != class.class_id {
                return Err(invalid(format!(
                    "fermion class {index} species-class ID is not aligned"
                )));
            }
            let species_id = required_string(
                &catalogs.strings,
                class.species_string_id,
                "fermion class species",
            )?;
            if !species.insert(species_id) {
                return Err(invalid(format!(
                    "fermion class {index} repeats species {species_id:?}"
                )));
            }
            let fundamental = pairing_range_slice(
                self.class_fundamental_slots,
                class.fundamental_slot_range,
                "class fundamental slots",
            )?;
            let antifundamental = pairing_range_slice(
                self.class_antifundamental_slots,
                class.antifundamental_slot_range,
                "class antifundamental slots",
            )?;
            let references = pairing_range_slice(
                self.class_reference_pairings,
                class.reference_pairing_range,
                "class reference pairings",
            )?;
            validate_strict_u32_values(
                fundamental,
                header.source_count as usize,
                "class fundamental slots",
            )?;
            validate_strict_u32_values(
                antifundamental,
                header.source_count as usize,
                "class antifundamental slots",
            )?;
            if fundamental.is_empty()
                || fundamental.len() != antifundamental.len()
                || references.len() != fundamental.len()
            {
                return Err(invalid(format!(
                    "fermion class {index} has unbalanced endpoint ranges"
                )));
            }
            for ((fundamental, antifundamental), reference) in fundamental
                .iter()
                .copied()
                .zip(antifundamental.iter().copied())
                .zip(references.iter().copied())
            {
                if reference.fundamental_source_slot != fundamental
                    || reference.antifundamental_source_slot != antifundamental
                {
                    return Err(invalid(format!(
                        "fermion class {index} reference pairings are stale"
                    )));
                }
            }
            let expected_pairing_count = checked_factorial_u64(
                fundamental.len(),
                &format!("fermion class {index} pairing count"),
            )?;
            if class.pairing_count != expected_pairing_count {
                return Err(invalid(format!(
                    "fermion class {index} declares {} pairings, expected {expected_pairing_count}",
                    class.pairing_count
                )));
            }
            SemanticDigest::new(class.proof_digest).map_err(|error| {
                invalid(format!(
                    "fermion class {index} proof digest is invalid: {error}"
                ))
            })?;
        }
        for endpoint in self.endpoints {
            let class = self.pairing_classes[endpoint.species_class_id as usize];
            if endpoint.species_string_id != class.species_string_id {
                return Err(invalid(
                    "fermion endpoint species disagrees with its pairing class",
                ));
            }
            let slots = if endpoint.color_orientation == 0 {
                pairing_range_slice(
                    self.class_fundamental_slots,
                    class.fundamental_slot_range,
                    "class fundamental slots",
                )?
            } else {
                pairing_range_slice(
                    self.class_antifundamental_slots,
                    class.antifundamental_slot_range,
                    "class antifundamental slots",
                )?
            };
            if slots.binary_search(&endpoint.source_slot).is_err() {
                return Err(invalid(
                    "fermion endpoint is absent from its pairing class orientation",
                ));
            }
        }
        Ok(())
    }

    fn validate_exact_integers(self) -> RusticolResult<()> {
        validate_canonical_ids(
            "fermion exact-integer IDs",
            self.exact_integers.iter().map(|row| row.integer_id),
        )?;
        for (index, row) in self.exact_integers.iter().copied().enumerate() {
            if !matches!(row.sign, -1..=1) {
                return Err(invalid(format!(
                    "exact integer {index} has an invalid sign"
                )));
            }
            let limbs = pairing_range_slice(
                self.exact_integer_limbs,
                row.limb_range,
                "exact-integer limbs",
            )?;
            if (row.sign == 0 && !limbs.is_empty())
                || (row.sign != 0 && (limbs.is_empty() || limbs.last() == Some(&0)))
            {
                return Err(invalid(format!(
                    "exact integer {index} has a noncanonical magnitude"
                )));
            }
        }
        Ok(())
    }

    fn validate_rules(
        self,
        catalogs: &FermionPairingCatalogs<'_>,
        header: FermionPairingHeaderRow,
    ) -> RusticolResult<()> {
        let expected_rule_count =
            self.pairing_classes
                .iter()
                .enumerate()
                .try_fold(1_u64, |count, (index, class)| {
                    count.checked_mul(class.pairing_count).ok_or_else(|| {
                        invalid(format!(
                            "combined fermion-pairing rule count overflows u64 at class {index}"
                        ))
                    })
                })?;
        let actual_rule_count = u64::try_from(self.rules.len())
            .map_err(|_| invalid("fermion-pairing rule count exceeds u64"))?;
        if actual_rule_count != expected_rule_count {
            return Err(invalid(format!(
                "fermion-pairing catalog contains {actual_rule_count} rules, expected {expected_rule_count}"
            )));
        }
        validate_canonical_ids(
            "fermion-pairing rule IDs",
            self.rules.iter().map(|row| row.rule_id),
        )?;
        for (index, rule) in self.rules.iter().copied().enumerate() {
            let class_indices = pairing_range_slice(
                self.rule_class_pairing_indices,
                rule.class_pairing_index_range,
                "rule class indices",
            )?;
            if class_indices.len() != self.pairing_classes.len() {
                return Err(invalid(format!(
                    "fermion rule {index} does not reference every class"
                )));
            }
            for (expected_class, encoded) in class_indices.iter().copied().enumerate() {
                let expected_class = u32::try_from(expected_class)
                    .map_err(|_| invalid("fermion pairing class index exceeds u32"))?;
                if encoded.class_id != expected_class
                    || encoded.pairing_index
                        >= self.pairing_classes[expected_class as usize].pairing_count
                {
                    return Err(invalid(format!(
                        "fermion rule {index} has an invalid class-local pairing index"
                    )));
                }
            }
            let pairs = pairing_range_slice(
                self.rule_endpoint_pairings,
                rule.endpoint_pairing_range,
                "rule endpoint pairings",
            )?;
            if pairs.windows(2).any(|pair| {
                (
                    pair[0].fundamental_source_slot,
                    pair[0].antifundamental_source_slot,
                ) >= (
                    pair[1].fundamental_source_slot,
                    pair[1].antifundamental_source_slot,
                )
            }) {
                return Err(invalid(format!(
                    "fermion rule {index} endpoint pairs are not canonical"
                )));
            }
            for pair in pairs {
                for source_slot in [
                    pair.fundamental_source_slot,
                    pair.antifundamental_source_slot,
                ] {
                    required_reference(
                        source_slot,
                        header.source_count as usize,
                        "rule endpoint source",
                    )?;
                }
            }
            let permutation = pairing_range_slice(
                self.rule_source_slot_permutations,
                rule.source_permutation_range,
                "rule source permutation",
            )?;
            validate_permutation(
                permutation,
                header.source_count as usize,
                &format!("fermion rule {index} source permutation"),
            )?;
            let lineage =
                pairing_range_slice(self.rule_lineages, rule.lineage_range, "rule lineage")?;
            if lineage.len() != header.source_count as usize
                || lineage.iter().any(|line| {
                    *line != RECURRENCE_NO_FERMION_LINE && (*line as usize) >= pairs.len()
                })
            {
                return Err(invalid(format!("fermion rule {index} lineage is invalid")));
            }
            if !matches!(rule.fermion_parity, -1 | 1) || rule.multiplicity != 1 {
                return Err(invalid(format!(
                    "fermion rule {index} parity or multiplicity is invalid"
                )));
            }
            require_string_value(
                &catalogs.strings,
                rule.proof_algorithm_string_id,
                RECURRENCE_FERMION_PAIRING_PROOF_ALGORITHM,
                "fermion rule proof algorithm",
            )?;
            SemanticDigest::new(rule.proof_digest).map_err(|error| {
                invalid(format!(
                    "fermion rule {index} proof digest is invalid: {error}"
                ))
            })?;
            self.validate_rule_factor(rule, index)?;
        }
        Ok(())
    }

    fn validate_rule_factor(
        self,
        rule: FermionPairingRuleRow,
        rule_index: usize,
    ) -> RusticolResult<()> {
        for (id, expected, label) in [
            (
                rule.real_numerator_integer_id,
                rule.fermion_parity,
                "real numerator",
            ),
            (rule.real_denominator_integer_id, 1, "real denominator"),
            (rule.imag_numerator_integer_id, 0, "imaginary numerator"),
            (rule.imag_denominator_integer_id, 1, "imaginary denominator"),
        ] {
            if !exact_integer_equals(self, id, expected)? {
                return Err(invalid(format!(
                    "fermion rule {rule_index} {label} does not match its exact parity factor"
                )));
            }
        }
        Ok(())
    }
}

/// Physical leading-colour sector representation used by process input v1.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum ProcessLCSectorKind {
    Singlet = 0,
    OpenLines = 1,
    SingleTrace = 2,
}

impl TryFrom<u8> for ProcessLCSectorKind {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Singlet),
            1 => Ok(Self::OpenLines),
            2 => Ok(Self::SingleTrace),
            _ => Err(invalid(format!(
                "unsupported recurrence LC sector kind {value}"
            ))),
        }
    }
}

/// Closed semantic-template kinds referenced by process input v1.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProcessSemanticTemplateKind {
    Parameter,
    CurrentState,
    Source,
    QuantumFlow,
    Transition,
    Propagator,
    Closure,
    ColorContraction,
    SymmetryProof,
}

impl ProcessSemanticTemplateKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parameter => "parameter",
            Self::CurrentState => "current-state",
            Self::Source => "source",
            Self::QuantumFlow => "quantum-flow",
            Self::Transition => "transition",
            Self::Propagator => "propagator",
            Self::Closure => "closure",
            Self::ColorContraction => "color-contraction",
            Self::SymmetryProof => "symmetry-proof",
        }
    }
}

impl TryFrom<&str> for ProcessSemanticTemplateKind {
    type Error = RusticolError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "parameter" => Ok(Self::Parameter),
            "current-state" => Ok(Self::CurrentState),
            "source" => Ok(Self::Source),
            "quantum-flow" => Ok(Self::QuantumFlow),
            "transition" => Ok(Self::Transition),
            "propagator" => Ok(Self::Propagator),
            "closure" => Ok(Self::Closure),
            "color-contraction" => Ok(Self::ColorContraction),
            "symmetry-proof" => Ok(Self::SymmetryProof),
            _ => Err(invalid(format!(
                "unsupported process semantic-template kind {value:?}"
            ))),
        }
    }
}

/// Typed identity of one model-wide semantic template.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcessSemanticTemplateId {
    pub kind: ProcessSemanticTemplateKind,
    pub template_id: u32,
}

/// Decoded process-side claim about one model-wide semantic template.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcessSemanticTemplateReference {
    pub typed_id: ProcessSemanticTemplateId,
    pub semantic_digest: SemanticDigest,
    pub prepared_kernel_id: Option<u32>,
}

/// Semantic digest role carried by the process header.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ProcessDigestRole {
    Process,
    ModelCatalog,
    PreparedCatalog,
    ColorPlan,
    Extension(String),
}

impl ProcessDigestRole {
    pub fn decode(value: &str) -> RusticolResult<Self> {
        if value.is_empty() {
            return Err(invalid("process semantic digest role must not be empty"));
        }
        Ok(match value {
            "process" => Self::Process,
            "model-catalog" => Self::ModelCatalog,
            "prepared-catalog" => Self::PreparedCatalog,
            "color-plan" => Self::ColorPlan,
            extension => Self::Extension(extension.to_owned()),
        })
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Process => "process",
            Self::ModelCatalog => "model-catalog",
            Self::PreparedCatalog => "prepared-catalog",
            Self::ColorPlan => "color-plan",
            Self::Extension(value) => value,
        }
    }
}

/// Decoded semantic identity needed by composite catalog authentication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceProcessSemanticIdentity {
    input_digest: SemanticDigest,
    process_digest: SemanticDigest,
    model_catalog_digest: SemanticDigest,
    prepared_catalog_digest: SemanticDigest,
    color_plan_digest: SemanticDigest,
    extension_digests: BTreeMap<String, SemanticDigest>,
}

impl RecurrenceProcessSemanticIdentity {
    pub const fn input_digest(&self) -> SemanticDigest {
        self.input_digest
    }

    pub const fn process_digest(&self) -> SemanticDigest {
        self.process_digest
    }

    pub const fn model_catalog_digest(&self) -> SemanticDigest {
        self.model_catalog_digest
    }

    pub const fn prepared_catalog_digest(&self) -> SemanticDigest {
        self.prepared_catalog_digest
    }

    pub const fn color_plan_digest(&self) -> SemanticDigest {
        self.color_plan_digest
    }

    pub const fn extension_digests(&self) -> &BTreeMap<String, SemanticDigest> {
        &self.extension_digests
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessBitsetRangeRow {
    pub id: u32,
    pub range: CheckedTableRange,
    pub bit_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessCouplingLimitRow {
    pub name_string_id: u32,
    pub minimum: u32,
    pub maximum: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessDigestCatalogRow {
    pub id: u32,
    pub value: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessExactFactorRow {
    pub id: u32,
    pub real_numerator_string_id: u32,
    pub real_denominator_string_id: u32,
    pub imag_numerator_string_id: u32,
    pub imag_denominator_string_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessExternalLegRow {
    pub source_slot: u32,
    pub public_label: u32,
    pub physical_pdg: i32,
    pub outgoing_pdg: i32,
    pub is_initial: u8,
    pub is_fermionic: u8,
    pub source_state_range: CheckedTableRange,
    pub momentum_mask_id: u32,
    pub support_mask_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessHeaderRow {
    pub schema_version: u32,
    pub abi_string_id: u32,
    pub process_id_string_id: u32,
    pub layout: u8,
    pub selected_flow_mode: u8,
    pub selected_source_mode: u8,
    pub external_leg_count: u32,
    pub physical_sector_count: u32,
    pub public_flow_count: u32,
    pub replay_partition_count: u32,
    pub coupling_limit_count: u32,
    pub parameter_projection_count: u32,
    pub process_support_mask_id: u32,
}

impl ProcessHeaderRow {
    pub fn strategy(self) -> RusticolResult<RecurrenceStrategy> {
        RecurrenceStrategy::try_from(u32::from(self.layout))
    }

    pub fn selected_flow_mode(self) -> RusticolResult<bool> {
        decode_bool(self.selected_flow_mode, "header.selected_flow_mode")
    }

    pub fn selected_source_mode(self) -> RusticolResult<bool> {
        decode_bool(self.selected_source_mode, "header.selected_source_mode")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessHeaderDigestRow {
    pub role_string_id: u32,
    pub digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessLCOpenStringRow {
    pub sector_id: u32,
    pub ordinal: u32,
    pub fundamental_source_slot: u32,
    pub antifundamental_source_slot: u32,
    pub adjoint_sequence_id: u32,
    pub singlet_sequence_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessNormalizationRow {
    pub factor_id: u32,
    pub convention_string_id: u32,
    pub semantic_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessParameterProjectionRow {
    pub runtime_slot: u32,
    pub runtime_name_string_id: u32,
    pub parameter_template_id: u32,
    pub prepared_parameter_id: u32,
    pub component: u32,
}

impl ProcessParameterProjectionRow {
    pub const fn prepared_parameter_id(self) -> Option<u32> {
        if self.prepared_parameter_id == MISSING_ID {
            None
        } else {
            Some(self.prepared_parameter_id)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessPhysicalLCSectorRow {
    pub sector_id: u32,
    pub public_id_string_id: u32,
    pub kind: u8,
    pub closure_source_slot: u32,
    pub closure_proof_algorithm_string_id: u32,
    pub closure_proof_digest_id: u32,
    pub open_string_range: CheckedTableRange,
    pub trace_sequence_id: u32,
    pub singlet_sequence_id: u32,
    pub word_sequence_id: u32,
    pub support_mask_id: u32,
}

impl ProcessPhysicalLCSectorRow {
    pub fn kind(self) -> RusticolResult<ProcessLCSectorKind> {
        ProcessLCSectorKind::try_from(self.kind)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessPublicLCFlowRow {
    pub flow_id: u32,
    pub public_id_string_id: u32,
    pub construction_sector_id: u32,
    pub word_sequence_id: u32,
    pub source_slot_permutation_sequence_id: u32,
    pub reduction_weight_factor_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessReplayPartitionRow {
    pub partition_id: u32,
    pub representative_sector_id: u32,
    pub materialized_sector_id: u32,
    pub target_range: CheckedTableRange,
    pub proof_algorithm_string_id: u32,
    pub proof_digest_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessReplayTargetRow {
    pub partition_id: u32,
    pub sector_id: u32,
    pub external_permutation_sequence_id: u32,
    pub source_slot_permutation_sequence_id: u32,
    pub amplitude_phase_factor_id: u32,
    pub fermion_sign: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessSelectedPublicFlowRow {
    pub flow_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessSelectedSourceStateRow {
    pub source_slot: u32,
    pub source_state_index: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessSemanticTemplateReferenceRow {
    pub kind_string_id: u32,
    pub template_id: u32,
    pub semantic_digest_id: u32,
    pub prepared_kernel_id: u32,
}

impl ProcessSemanticTemplateReferenceRow {
    pub const fn prepared_kernel_id(self) -> Option<u32> {
        if self.prepared_kernel_id == MISSING_ID {
            None
        } else {
            Some(self.prepared_kernel_id)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessSourceStateRow {
    pub source_slot: u32,
    pub state_index: u32,
    pub public_helicity: i32,
    pub chirality: i32,
    pub spin_state: i32,
    pub current_state_template_id: u32,
    pub source_template_id: u32,
    pub momentum_sign: i32,
    pub crossing_phase_factor_id: u32,
}

/// Fixed-width header for one process-owned external-fermion pairing catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FermionPairingHeaderRow {
    pub schema_version: u32,
    pub abi_string_id: u32,
    pub process_key_string_id: u32,
    pub proof_algorithm_string_id: u32,
    pub source_count: u32,
    pub endpoint_count: u32,
    pub pairing_class_count: u32,
    pub rule_count: u32,
    pub endpoint_state_template_count: u64,
    pub endpoint_anti_state_template_count: u64,
    pub endpoint_basis_count: u64,
    pub endpoint_color_representation_count: u64,
    pub class_fundamental_slot_count: u64,
    pub class_antifundamental_slot_count: u64,
    pub class_reference_pairing_count: u64,
    pub rule_class_pairing_index_count: u64,
    pub rule_endpoint_pairing_count: u64,
    pub rule_source_permutation_count: u64,
    pub rule_lineage_count: u64,
    pub exact_integer_count: u32,
    pub exact_integer_limb_count: u64,
    pub string_count: u32,
    pub string_byte_count: u64,
    pub no_fermion_line: u32,
    pub topology_digest: [u8; 32],
    pub semantic_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FermionPairingEndpointRow {
    pub endpoint_id: u32,
    pub source_slot: u32,
    pub public_label: u32,
    pub species_class_id: u32,
    pub species_string_id: u32,
    pub particle_orientation: u8,
    pub color_orientation: u8,
    pub state_template_range: CheckedTableRange,
    pub anti_state_template_range: CheckedTableRange,
    pub basis_range: CheckedTableRange,
    pub color_representation_range: CheckedTableRange,
    pub contract_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FermionPairingClassRow {
    pub class_id: u32,
    pub species_class_id: u32,
    pub species_string_id: u32,
    pub fundamental_slot_range: CheckedTableRange,
    pub antifundamental_slot_range: CheckedTableRange,
    pub reference_pairing_range: CheckedTableRange,
    pub pairing_count: u64,
    pub proof_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FermionPairingRuleRow {
    pub rule_id: u32,
    pub class_pairing_index_range: CheckedTableRange,
    pub endpoint_pairing_range: CheckedTableRange,
    pub source_permutation_range: CheckedTableRange,
    pub lineage_range: CheckedTableRange,
    pub fermion_parity: i32,
    pub real_numerator_integer_id: u32,
    pub real_denominator_integer_id: u32,
    pub imag_numerator_integer_id: u32,
    pub imag_denominator_integer_id: u32,
    pub multiplicity: u64,
    pub proof_algorithm_string_id: u32,
    pub proof_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FermionPairingClassPairingIndexRow {
    pub class_id: u32,
    pub pairing_index: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FermionPairingEndpointPairRow {
    pub fundamental_source_slot: u32,
    pub antifundamental_source_slot: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FermionPairingExactIntegerRow {
    pub integer_id: u32,
    pub sign: i32,
    pub limb_range: CheckedTableRange,
}

#[derive(Clone, Copy, Debug)]
pub struct FermionPairingInputView<'a> {
    pub input_abi: &'a str,
    pub declared_columnar_digest: SemanticDigest,
    pub header: &'a [FermionPairingHeaderRow],
    pub endpoints: &'a [FermionPairingEndpointRow],
    pub endpoint_state_template_ids: &'a [u32],
    pub endpoint_anti_state_template_ids: &'a [u32],
    pub endpoint_basis_ids: &'a [u32],
    pub endpoint_color_representations: &'a [i32],
    pub pairing_classes: &'a [FermionPairingClassRow],
    pub class_fundamental_slots: &'a [u32],
    pub class_antifundamental_slots: &'a [u32],
    pub class_reference_pairings: &'a [FermionPairingEndpointPairRow],
    pub rules: &'a [FermionPairingRuleRow],
    pub rule_class_pairing_indices: &'a [FermionPairingClassPairingIndexRow],
    pub rule_endpoint_pairings: &'a [FermionPairingEndpointPairRow],
    pub rule_source_slot_permutations: &'a [u32],
    pub rule_lineages: &'a [u32],
    pub exact_integers: &'a [FermionPairingExactIntegerRow],
    pub exact_integer_limbs: &'a [u64],
    pub string_ranges: &'a [CheckedTableRange],
    pub string_bytes: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct OwnedFermionPairingInput {
    pub input_abi: String,
    pub declared_columnar_digest: SemanticDigest,
    pub header: Vec<FermionPairingHeaderRow>,
    pub endpoints: Vec<FermionPairingEndpointRow>,
    pub endpoint_state_template_ids: Vec<u32>,
    pub endpoint_anti_state_template_ids: Vec<u32>,
    pub endpoint_basis_ids: Vec<u32>,
    pub endpoint_color_representations: Vec<i32>,
    pub pairing_classes: Vec<FermionPairingClassRow>,
    pub class_fundamental_slots: Vec<u32>,
    pub class_antifundamental_slots: Vec<u32>,
    pub class_reference_pairings: Vec<FermionPairingEndpointPairRow>,
    pub rules: Vec<FermionPairingRuleRow>,
    pub rule_class_pairing_indices: Vec<FermionPairingClassPairingIndexRow>,
    pub rule_endpoint_pairings: Vec<FermionPairingEndpointPairRow>,
    pub rule_source_slot_permutations: Vec<u32>,
    pub rule_lineages: Vec<u32>,
    pub exact_integers: Vec<FermionPairingExactIntegerRow>,
    pub exact_integer_limbs: Vec<u64>,
    pub string_ranges: Vec<CheckedTableRange>,
    pub string_bytes: Vec<u8>,
}

impl OwnedFermionPairingInput {
    pub fn as_view(&self) -> FermionPairingInputView<'_> {
        FermionPairingInputView {
            input_abi: &self.input_abi,
            declared_columnar_digest: self.declared_columnar_digest,
            header: &self.header,
            endpoints: &self.endpoints,
            endpoint_state_template_ids: &self.endpoint_state_template_ids,
            endpoint_anti_state_template_ids: &self.endpoint_anti_state_template_ids,
            endpoint_basis_ids: &self.endpoint_basis_ids,
            endpoint_color_representations: &self.endpoint_color_representations,
            pairing_classes: &self.pairing_classes,
            class_fundamental_slots: &self.class_fundamental_slots,
            class_antifundamental_slots: &self.class_antifundamental_slots,
            class_reference_pairings: &self.class_reference_pairings,
            rules: &self.rules,
            rule_class_pairing_indices: &self.rule_class_pairing_indices,
            rule_endpoint_pairings: &self.rule_endpoint_pairings,
            rule_source_slot_permutations: &self.rule_source_slot_permutations,
            rule_lineages: &self.rule_lineages,
            exact_integers: &self.exact_integers,
            exact_integer_limbs: &self.exact_integer_limbs,
            string_ranges: &self.string_ranges,
            string_bytes: &self.string_bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FermionPairingValidationSummary {
    columnar_digest: SemanticDigest,
    topology_digest: SemanticDigest,
    semantic_digest: SemanticDigest,
    source_count: u32,
    endpoint_count: u32,
    pairing_class_count: u32,
    rule_count: u32,
}

impl FermionPairingValidationSummary {
    pub const fn columnar_digest(&self) -> SemanticDigest {
        self.columnar_digest
    }

    pub const fn topology_digest(&self) -> SemanticDigest {
        self.topology_digest
    }

    pub const fn semantic_digest(&self) -> SemanticDigest {
        self.semantic_digest
    }

    pub const fn source_count(&self) -> u32 {
        self.source_count
    }

    pub const fn endpoint_count(&self) -> u32 {
        self.endpoint_count
    }

    pub const fn pairing_class_count(&self) -> u32 {
        self.pairing_class_count
    }

    pub const fn rule_count(&self) -> u32 {
        self.rule_count
    }
}

macro_rules! define_process_inputs {
    ($( $field:ident : $item:ty ),+ $(,)?) => {
        #[derive(Clone, Copy, Debug)]
        pub struct RecurrenceProcessInputView<'a> {
            pub input_abi: &'a str,
            pub declared_input_digest: SemanticDigest,
            pub fermion_pairing: Option<FermionPairingInputView<'a>>,
            $(pub $field: &'a [$item],)+
        }

        #[derive(Clone, Debug)]
        pub struct OwnedRecurrenceProcessInput {
            pub input_abi: String,
            pub declared_input_digest: SemanticDigest,
            pub fermion_pairing: Option<OwnedFermionPairingInput>,
            $(pub $field: Vec<$item>,)+
        }

        impl OwnedRecurrenceProcessInput {
            pub fn as_view(&self) -> RecurrenceProcessInputView<'_> {
                RecurrenceProcessInputView {
                    input_abi: &self.input_abi,
                    declared_input_digest: self.declared_input_digest,
                    fermion_pairing: self.fermion_pairing.as_ref().map(OwnedFermionPairingInput::as_view),
                    $($field: &self.$field,)+
                }
            }

            pub fn validate(self) -> RusticolResult<ValidatedRecurrenceProcessInput> {
                let validated = self.as_view().validate()?;
                Ok(ValidatedRecurrenceProcessInput {
                    input: self,
                    summary: validated.summary,
                    semantic_identity: validated.semantic_identity,
                    template_references: validated.template_references,
                    fermion_pairing_summary: validated.fermion_pairing_summary,
                })
            }
        }
    };
}

define_process_inputs! {
    bitset_ranges: ProcessBitsetRangeRow,
    bitset_words: u64,
    coupling_limits: ProcessCouplingLimitRow,
    digest_catalog: ProcessDigestCatalogRow,
    exact_factors: ProcessExactFactorRow,
    external_legs: ProcessExternalLegRow,
    header: ProcessHeaderRow,
    header_digests: ProcessHeaderDigestRow,
    lc_open_strings: ProcessLCOpenStringRow,
    normalization: ProcessNormalizationRow,
    parameter_projection: ProcessParameterProjectionRow,
    physical_lc_sectors: ProcessPhysicalLCSectorRow,
    public_lc_flows: ProcessPublicLCFlowRow,
    replay_partitions: ProcessReplayPartitionRow,
    replay_targets: ProcessReplayTargetRow,
    selected_public_flow_coverage: ProcessSelectedPublicFlowRow,
    selected_source_coverage: ProcessSelectedSourceStateRow,
    semantic_template_references: ProcessSemanticTemplateReferenceRow,
    source_states: ProcessSourceStateRow,
    string_ranges: CheckedTableRange,
    string_bytes: u8,
    u32_sequence_ranges: CheckedTableRange,
    u32_sequence_values: u32,
}

/// Counts and process identity produced by process-input validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceProcessValidationSummary {
    process_id: String,
    strategy: RecurrenceStrategy,
    selected_flow_mode: bool,
    selected_source_mode: bool,
    external_leg_count: u32,
    source_state_count: u32,
    physical_sector_count: u32,
    public_flow_count: u32,
    replay_partition_count: u32,
    replay_target_count: u32,
    template_reference_count: u32,
}

impl RecurrenceProcessValidationSummary {
    pub fn process_id(&self) -> &str {
        &self.process_id
    }

    pub const fn strategy(&self) -> RecurrenceStrategy {
        self.strategy
    }

    pub const fn selected_flow_mode(&self) -> bool {
        self.selected_flow_mode
    }

    pub const fn selected_source_mode(&self) -> bool {
        self.selected_source_mode
    }

    pub const fn external_leg_count(&self) -> u32 {
        self.external_leg_count
    }

    pub const fn source_state_count(&self) -> u32 {
        self.source_state_count
    }

    pub const fn physical_sector_count(&self) -> u32 {
        self.physical_sector_count
    }

    pub const fn public_flow_count(&self) -> u32 {
        self.public_flow_count
    }

    pub const fn replay_partition_count(&self) -> u32 {
        self.replay_partition_count
    }

    pub const fn replay_target_count(&self) -> u32 {
        self.replay_target_count
    }

    pub const fn template_reference_count(&self) -> u32 {
        self.template_reference_count
    }
}

/// Validated process payload retained for later builder and catalog checks.
#[derive(Clone, Debug)]
pub struct ValidatedRecurrenceProcessInput {
    input: OwnedRecurrenceProcessInput,
    summary: RecurrenceProcessValidationSummary,
    semantic_identity: RecurrenceProcessSemanticIdentity,
    template_references: Vec<ProcessSemanticTemplateReference>,
    fermion_pairing_summary: Option<FermionPairingValidationSummary>,
}

/// Zero-copy access to a process-owned pairing catalog after full validation.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedFermionPairingCatalog<'a> {
    input: FermionPairingInputView<'a>,
    summary: &'a FermionPairingValidationSummary,
}

impl<'a> ValidatedFermionPairingCatalog<'a> {
    pub const fn summary(self) -> &'a FermionPairingValidationSummary {
        self.summary
    }

    pub const fn endpoints(self) -> &'a [FermionPairingEndpointRow] {
        self.input.endpoints
    }

    pub const fn pairing_classes(self) -> &'a [FermionPairingClassRow] {
        self.input.pairing_classes
    }

    pub const fn rules(self) -> &'a [FermionPairingRuleRow] {
        self.input.rules
    }

    pub fn class_pairing_indices(
        self,
        rule: FermionPairingRuleRow,
    ) -> RusticolResult<&'a [FermionPairingClassPairingIndexRow]> {
        pairing_range_slice(
            self.input.rule_class_pairing_indices,
            rule.class_pairing_index_range,
            "fermion pairing rule class indices",
        )
    }

    pub fn endpoint_pairings(
        self,
        rule: FermionPairingRuleRow,
    ) -> RusticolResult<&'a [FermionPairingEndpointPairRow]> {
        pairing_range_slice(
            self.input.rule_endpoint_pairings,
            rule.endpoint_pairing_range,
            "fermion pairing rule endpoint pairs",
        )
    }

    pub fn source_slot_permutation(self, rule: FermionPairingRuleRow) -> RusticolResult<&'a [u32]> {
        pairing_range_slice(
            self.input.rule_source_slot_permutations,
            rule.source_permutation_range,
            "fermion pairing rule source permutation",
        )
    }

    pub fn lineage(self, rule: FermionPairingRuleRow) -> RusticolResult<&'a [u32]> {
        pairing_range_slice(
            self.input.rule_lineages,
            rule.lineage_range,
            "fermion pairing rule lineage",
        )
    }
}

impl ValidatedRecurrenceProcessInput {
    pub const fn input(&self) -> &OwnedRecurrenceProcessInput {
        &self.input
    }

    pub const fn summary(&self) -> &RecurrenceProcessValidationSummary {
        &self.summary
    }

    pub const fn semantic_identity(&self) -> &RecurrenceProcessSemanticIdentity {
        &self.semantic_identity
    }

    pub fn template_references(&self) -> &[ProcessSemanticTemplateReference] {
        &self.template_references
    }

    pub fn template_reference(
        &self,
        typed_id: ProcessSemanticTemplateId,
    ) -> Option<&ProcessSemanticTemplateReference> {
        self.template_references
            .binary_search_by_key(&typed_id, |reference| reference.typed_id)
            .ok()
            .map(|index| &self.template_references[index])
    }

    pub fn fermion_pairing_catalog(&self) -> Option<ValidatedFermionPairingCatalog<'_>> {
        Some(ValidatedFermionPairingCatalog {
            input: self.input.fermion_pairing.as_ref()?.as_view(),
            summary: self.fermion_pairing_summary.as_ref()?,
        })
    }

    pub const fn fermion_pairing_summary(&self) -> Option<&FermionPairingValidationSummary> {
        self.fermion_pairing_summary.as_ref()
    }

    pub fn into_input(self) -> OwnedRecurrenceProcessInput {
        self.input
    }
}

struct ValidatedProcessMetadata {
    summary: RecurrenceProcessValidationSummary,
    semantic_identity: RecurrenceProcessSemanticIdentity,
    template_references: Vec<ProcessSemanticTemplateReference>,
    fermion_pairing_summary: Option<FermionPairingValidationSummary>,
}

struct ProcessCatalogs<'a> {
    strings: Vec<&'a str>,
    digests: Vec<SemanticDigest>,
    factors: Vec<ExactComplexRational>,
}

impl<'a> RecurrenceProcessInputView<'a> {
    fn validate(self) -> RusticolResult<ValidatedProcessMetadata> {
        if self.input_abi != RECURRENCE_BUILDER_INPUT_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence process input ABI {:?}; expected {:?}",
                self.input_abi, RECURRENCE_BUILDER_INPUT_ABI
            )));
        }

        let catalogs = self.validate_catalogs()?;
        let (strategy, selected_flow_mode, selected_source_mode, process_id) =
            self.validate_header(&catalogs)?;
        let semantic_identity = self.validate_semantic_identity(&catalogs)?;
        let template_references = self.validate_template_references(&catalogs)?;

        self.validate_dense_ids()?;
        self.validate_common_references(&catalogs)?;
        self.validate_external_legs(&catalogs, &template_references)?;
        self.validate_color_sectors(&catalogs)?;
        self.validate_public_flows(&catalogs)?;
        self.validate_replay(&catalogs, strategy)?;
        self.validate_generation_coverage(selected_flow_mode, selected_source_mode)?;
        self.validate_couplings_and_parameters(&catalogs, &template_references)?;
        self.validate_normalization(&catalogs)?;
        let fermion_pairing_summary =
            self.validate_fermion_pairing(process_id, &semantic_identity)?;

        Ok(ValidatedProcessMetadata {
            summary: RecurrenceProcessValidationSummary {
                process_id: process_id.to_owned(),
                strategy,
                selected_flow_mode,
                selected_source_mode,
                external_leg_count: checked_u32_len(
                    self.external_legs.len(),
                    "process external legs",
                )?,
                source_state_count: checked_u32_len(
                    self.source_states.len(),
                    "process source states",
                )?,
                physical_sector_count: checked_u32_len(
                    self.physical_lc_sectors.len(),
                    "physical LC sectors",
                )?,
                public_flow_count: checked_u32_len(self.public_lc_flows.len(), "public LC flows")?,
                replay_partition_count: checked_u32_len(
                    self.replay_partitions.len(),
                    "replay partitions",
                )?,
                replay_target_count: checked_u32_len(self.replay_targets.len(), "replay targets")?,
                template_reference_count: checked_u32_len(
                    template_references.len(),
                    "semantic-template references",
                )?,
            },
            semantic_identity,
            template_references,
            fermion_pairing_summary,
        })
    }

    fn validate_catalogs(self) -> RusticolResult<ProcessCatalogs<'a>> {
        validate_packed_ranges(
            "recurrence process strings",
            self.string_ranges,
            self.string_bytes.len(),
        )?;
        let mut strings = Vec::with_capacity(self.string_ranges.len());
        let mut previous: Option<&[u8]> = None;
        for (index, range) in self.string_ranges.iter().copied().enumerate() {
            let bytes = &self.string_bytes[range
                .as_usize_range(self.string_bytes.len(), &format!("process string {index}"))?];
            if bytes.is_empty() {
                return Err(invalid(format!(
                    "recurrence process string catalog row {index} is empty"
                )));
            }
            if let Some(previous) = previous
                && previous >= bytes
            {
                return Err(invalid(format!(
                    "recurrence process string catalog is not in strict canonical byte order at row {index}"
                )));
            }
            let value = std::str::from_utf8(bytes).map_err(|error| {
                invalid(format!(
                    "recurrence process string catalog row {index} is not UTF-8: {error}"
                ))
            })?;
            previous = Some(bytes);
            strings.push(value);
        }

        validate_packed_ranges(
            "recurrence process u32 sequences",
            self.u32_sequence_ranges,
            self.u32_sequence_values.len(),
        )?;
        let mut previous_sequence: Option<&[u32]> = None;
        for (index, range) in self.u32_sequence_ranges.iter().copied().enumerate() {
            let values = &self.u32_sequence_values[range.as_usize_range(
                self.u32_sequence_values.len(),
                &format!("process u32 sequence {index}"),
            )?];
            if let Some(previous) = previous_sequence
                && previous >= values
            {
                return Err(invalid(format!(
                    "recurrence process u32-sequence catalog is not in strict canonical order at row {index}"
                )));
            }
            previous_sequence = Some(values);
        }

        validate_canonical_ids(
            "process bitset catalog",
            self.bitset_ranges.iter().map(|row| row.id),
        )?;
        let mask_ranges = self
            .bitset_ranges
            .iter()
            .map(|row| row.range)
            .collect::<Vec<_>>();
        let populations = self
            .bitset_ranges
            .iter()
            .map(|row| row.bit_count)
            .collect::<Vec<_>>();
        MultiwordMaskCatalogView {
            ranges: &mask_ranges,
            populations: &populations,
            words: self.bitset_words,
        }
        .validate(true)?;

        validate_canonical_ids(
            "process digest catalog",
            self.digest_catalog.iter().map(|row| row.id),
        )?;
        let mut digests = Vec::with_capacity(self.digest_catalog.len());
        let mut previous_digest = None;
        for (index, row) in self.digest_catalog.iter().enumerate() {
            let value = SemanticDigest::new(row.value).map_err(|error| {
                invalid(format!(
                    "recurrence process digest catalog row {index} is invalid: {error}"
                ))
            })?;
            if let Some(previous) = previous_digest
                && previous >= value
            {
                return Err(invalid(format!(
                    "recurrence process digest catalog is not in strict canonical order at row {index}"
                )));
            }
            previous_digest = Some(value);
            digests.push(value);
        }

        validate_canonical_ids(
            "process exact-factor catalog",
            self.exact_factors.iter().map(|row| row.id),
        )?;
        let mut factors = Vec::with_capacity(self.exact_factors.len());
        let mut previous_factor = None;
        for (index, row) in self.exact_factors.iter().enumerate() {
            let real_numerator = required_string(
                &strings,
                row.real_numerator_string_id,
                "process exact real numerator",
            )?;
            let real_denominator = required_string(
                &strings,
                row.real_denominator_string_id,
                "process exact real denominator",
            )?;
            let imag_numerator = required_string(
                &strings,
                row.imag_numerator_string_id,
                "process exact imaginary numerator",
            )?;
            let imag_denominator = required_string(
                &strings,
                row.imag_denominator_string_id,
                "process exact imaginary denominator",
            )?;
            let factor = ExactComplexRational::parse_parts(
                real_numerator,
                real_denominator,
                imag_numerator,
                imag_denominator,
            )
            .map_err(|error| {
                invalid(format!(
                    "recurrence process exact factor {index} is invalid: {error}"
                ))
            })?;
            let canonical = (
                factor.real().numerator(),
                factor.real().denominator(),
                factor.imag().numerator(),
                factor.imag().denominator(),
            );
            if canonical.0.to_string() != real_numerator
                || canonical.1.to_string() != real_denominator
                || canonical.2.to_string() != imag_numerator
                || canonical.3.to_string() != imag_denominator
            {
                return Err(invalid(format!(
                    "recurrence process exact factor {index} is not in canonical reduced decimal form"
                )));
            }
            if let Some(previous) = previous_factor
                && previous >= canonical
            {
                return Err(invalid(format!(
                    "recurrence process exact-factor catalog is not in strict canonical order at row {index}"
                )));
            }
            previous_factor = Some(canonical);
            factors.push(factor);
        }

        Ok(ProcessCatalogs {
            strings,
            digests,
            factors,
        })
    }

    fn validate_header<'b>(
        self,
        catalogs: &'b ProcessCatalogs<'_>,
    ) -> RusticolResult<(RecurrenceStrategy, bool, bool, &'b str)> {
        if self.header.len() != 1 {
            return Err(invalid(format!(
                "recurrence process header must contain one row, found {}",
                self.header.len()
            )));
        }
        let header = self.header[0];
        if header.schema_version != RECURRENCE_PROCESS_INPUT_SCHEMA_VERSION {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence process input schema {}; expected {}",
                header.schema_version, RECURRENCE_PROCESS_INPUT_SCHEMA_VERSION
            )));
        }
        require_string_value(
            &catalogs.strings,
            header.abi_string_id,
            RECURRENCE_BUILDER_INPUT_ABI,
            "recurrence process ABI",
        )?;
        let process_id = required_string(
            &catalogs.strings,
            header.process_id_string_id,
            "recurrence process ID",
        )?;
        let strategy = header.strategy()?;
        let selected_flow_mode = header.selected_flow_mode()?;
        let selected_source_mode = header.selected_source_mode()?;

        let expected = [
            (
                "external legs",
                header.external_leg_count,
                self.external_legs.len(),
            ),
            (
                "physical LC sectors",
                header.physical_sector_count,
                self.physical_lc_sectors.len(),
            ),
            (
                "public LC flows",
                header.public_flow_count,
                self.public_lc_flows.len(),
            ),
            (
                "replay partitions",
                header.replay_partition_count,
                self.replay_partitions.len(),
            ),
            (
                "coupling limits",
                header.coupling_limit_count,
                self.coupling_limits.len(),
            ),
            (
                "parameter projections",
                header.parameter_projection_count,
                self.parameter_projection.len(),
            ),
        ];
        for (label, declared, actual) in expected {
            let actual = checked_u32_len(actual, label)?;
            if declared != actual {
                return Err(invalid(format!(
                    "recurrence process header declares {declared} {label}, found {actual}"
                )));
            }
        }
        required_reference(
            header.process_support_mask_id,
            self.bitset_ranges.len(),
            "process support mask",
        )?;
        if selected_flow_mode == self.selected_public_flow_coverage.is_empty() {
            return Err(invalid(
                "selected-flow header mode disagrees with process coverage rows",
            ));
        }
        if selected_source_mode == self.selected_source_coverage.is_empty() {
            return Err(invalid(
                "selected-source header mode disagrees with process coverage rows",
            ));
        }
        Ok((
            strategy,
            selected_flow_mode,
            selected_source_mode,
            process_id,
        ))
    }

    fn validate_semantic_identity(
        self,
        catalogs: &ProcessCatalogs<'_>,
    ) -> RusticolResult<RecurrenceProcessSemanticIdentity> {
        if self.header_digests.len() < REQUIRED_DIGEST_ROLE_COUNT {
            return Err(invalid(
                "recurrence process header has too few semantic digest roles",
            ));
        }
        let mut values = BTreeMap::new();
        let mut previous_role = None;
        for (index, row) in self.header_digests.iter().enumerate() {
            let role_text = required_string(
                &catalogs.strings,
                row.role_string_id,
                "process semantic digest role",
            )?;
            if let Some(previous) = previous_role
                && previous >= role_text
            {
                return Err(invalid(format!(
                    "process semantic digest roles are not in strict canonical order at row {index}"
                )));
            }
            previous_role = Some(role_text);
            let role = ProcessDigestRole::decode(role_text)?;
            let digest =
                required_digest(&catalogs.digests, row.digest_id, "process semantic digest")?;
            if values.insert(role, digest).is_some() {
                return Err(invalid(format!(
                    "process semantic digest role {role_text:?} is repeated"
                )));
            }
        }

        let process_digest = take_required_role(&mut values, ProcessDigestRole::Process)?;
        let model_catalog_digest =
            take_required_role(&mut values, ProcessDigestRole::ModelCatalog)?;
        let prepared_catalog_digest =
            take_required_role(&mut values, ProcessDigestRole::PreparedCatalog)?;
        let color_plan_digest = take_required_role(&mut values, ProcessDigestRole::ColorPlan)?;
        let extension_digests = values
            .into_iter()
            .map(|(role, digest)| (role.as_str().to_owned(), digest))
            .collect();
        Ok(RecurrenceProcessSemanticIdentity {
            input_digest: self.declared_input_digest,
            process_digest,
            model_catalog_digest,
            prepared_catalog_digest,
            color_plan_digest,
            extension_digests,
        })
    }

    fn validate_template_references(
        self,
        catalogs: &ProcessCatalogs<'_>,
    ) -> RusticolResult<Vec<ProcessSemanticTemplateReference>> {
        let mut references = Vec::with_capacity(self.semantic_template_references.len());
        let mut seen_ids = BTreeSet::new();
        let mut seen_digests = BTreeSet::new();
        let mut previous_raw = None;
        for (index, row) in self
            .semantic_template_references
            .iter()
            .copied()
            .enumerate()
        {
            let kind_text = required_string(
                &catalogs.strings,
                row.kind_string_id,
                "process semantic-template kind",
            )?;
            let raw_key = (kind_text, row.template_id);
            if let Some(previous) = previous_raw
                && previous >= raw_key
            {
                return Err(invalid(format!(
                    "process semantic-template references are not in strict canonical order at row {index}"
                )));
            }
            previous_raw = Some(raw_key);
            let typed_id = ProcessSemanticTemplateId {
                kind: ProcessSemanticTemplateKind::try_from(kind_text)?,
                template_id: row.template_id,
            };
            if !seen_ids.insert(typed_id) {
                return Err(invalid(format!(
                    "process repeats semantic-template reference ({kind_text:?}, {})",
                    row.template_id
                )));
            }
            let semantic_digest = required_digest(
                &catalogs.digests,
                row.semantic_digest_id,
                "semantic-template digest",
            )?;
            if !seen_digests.insert(semantic_digest) {
                return Err(invalid(format!(
                    "process semantic-template reference row {index} reuses semantic digest {semantic_digest}"
                )));
            }
            references.push(ProcessSemanticTemplateReference {
                typed_id,
                semantic_digest,
                prepared_kernel_id: row.prepared_kernel_id(),
            });
        }
        references.sort_unstable_by_key(|reference| reference.typed_id);
        Ok(references)
    }

    fn validate_dense_ids(self) -> RusticolResult<()> {
        validate_canonical_ids(
            "external source slots",
            self.external_legs.iter().map(|row| row.source_slot),
        )?;
        validate_canonical_ids(
            "physical LC sector IDs",
            self.physical_lc_sectors.iter().map(|row| row.sector_id),
        )?;
        validate_canonical_ids(
            "public LC flow IDs",
            self.public_lc_flows.iter().map(|row| row.flow_id),
        )?;
        validate_canonical_ids(
            "replay partition IDs",
            self.replay_partitions.iter().map(|row| row.partition_id),
        )?;
        validate_canonical_ids(
            "runtime parameter slots",
            self.parameter_projection.iter().map(|row| row.runtime_slot),
        )?;
        Ok(())
    }

    fn validate_common_references(self, catalogs: &ProcessCatalogs<'_>) -> RusticolResult<()> {
        let string_count = catalogs.strings.len();
        validate_u32_references(
            &self
                .coupling_limits
                .iter()
                .map(|row| row.name_string_id)
                .collect::<Vec<_>>(),
            string_count,
            "coupling-limit names",
        )?;
        validate_u32_references(
            &self
                .physical_lc_sectors
                .iter()
                .flat_map(|row| {
                    [
                        row.public_id_string_id,
                        row.closure_proof_algorithm_string_id,
                    ]
                })
                .chain(
                    self.public_lc_flows
                        .iter()
                        .map(|row| row.public_id_string_id),
                )
                .chain(
                    self.replay_partitions
                        .iter()
                        .map(|row| row.proof_algorithm_string_id),
                )
                .chain(
                    self.normalization
                        .iter()
                        .map(|row| row.convention_string_id),
                )
                .chain(
                    self.parameter_projection
                        .iter()
                        .map(|row| row.runtime_name_string_id),
                )
                .collect::<Vec<_>>(),
            string_count,
            "process string references",
        )?;

        validate_u32_references(
            &self
                .physical_lc_sectors
                .iter()
                .map(|row| row.closure_proof_digest_id)
                .collect::<Vec<_>>(),
            catalogs.digests.len(),
            "physical LC closure proof digests",
        )?;

        let sequence_count = self.u32_sequence_ranges.len();
        validate_u32_references(
            &self
                .lc_open_strings
                .iter()
                .flat_map(|row| [row.adjoint_sequence_id, row.singlet_sequence_id])
                .chain(self.physical_lc_sectors.iter().flat_map(|row| {
                    [
                        row.trace_sequence_id,
                        row.singlet_sequence_id,
                        row.word_sequence_id,
                    ]
                }))
                .chain(self.public_lc_flows.iter().flat_map(|row| {
                    [
                        row.word_sequence_id,
                        row.source_slot_permutation_sequence_id,
                    ]
                }))
                .chain(self.replay_targets.iter().flat_map(|row| {
                    [
                        row.external_permutation_sequence_id,
                        row.source_slot_permutation_sequence_id,
                    ]
                }))
                .collect::<Vec<_>>(),
            sequence_count,
            "process u32-sequence references",
        )?;

        let bitset_count = self.bitset_ranges.len();
        validate_u32_references(
            &self
                .external_legs
                .iter()
                .flat_map(|row| [row.momentum_mask_id, row.support_mask_id])
                .chain(
                    self.physical_lc_sectors
                        .iter()
                        .map(|row| row.support_mask_id),
                )
                .collect::<Vec<_>>(),
            bitset_count,
            "process bitset references",
        )?;

        let factor_count = catalogs.factors.len();
        validate_u32_references(
            &self
                .source_states
                .iter()
                .map(|row| row.crossing_phase_factor_id)
                .chain(
                    self.public_lc_flows
                        .iter()
                        .map(|row| row.reduction_weight_factor_id),
                )
                .chain(
                    self.replay_targets
                        .iter()
                        .map(|row| row.amplitude_phase_factor_id),
                )
                .chain(self.normalization.iter().map(|row| row.factor_id))
                .collect::<Vec<_>>(),
            factor_count,
            "process exact-factor references",
        )?;

        validate_u32_references(
            &self
                .normalization
                .iter()
                .map(|row| row.semantic_digest_id)
                .chain(self.replay_partitions.iter().map(|row| row.proof_digest_id))
                .collect::<Vec<_>>(),
            catalogs.digests.len(),
            "process semantic-digest references",
        )?;
        Ok(())
    }

    fn validate_external_legs(
        self,
        catalogs: &ProcessCatalogs<'_>,
        references: &[ProcessSemanticTemplateReference],
    ) -> RusticolResult<()> {
        if self.external_legs.is_empty() {
            return Err(invalid(
                "recurrence process input requires at least one external leg",
            ));
        }
        let ranges = self
            .external_legs
            .iter()
            .map(|row| row.source_state_range)
            .collect::<Vec<_>>();
        validate_packed_ranges("external source states", &ranges, self.source_states.len())?;

        let mask_ranges = self
            .bitset_ranges
            .iter()
            .map(|row| row.range)
            .collect::<Vec<_>>();
        let populations = self
            .bitset_ranges
            .iter()
            .map(|row| row.bit_count)
            .collect::<Vec<_>>();
        let masks = MultiwordMaskCatalogView {
            ranges: &mask_ranges,
            populations: &populations,
            words: self.bitset_words,
        };

        let mut public_labels = BTreeSet::new();
        for (source_slot, leg) in self.external_legs.iter().copied().enumerate() {
            decode_bool(leg.is_initial, "external_legs.is_initial")?;
            decode_bool(leg.is_fermionic, "external_legs.is_fermionic")?;
            if !public_labels.insert(leg.public_label) {
                return Err(invalid(format!(
                    "external leg {source_slot} repeats public label {}",
                    leg.public_label
                )));
            }
            if leg.source_state_range.count == 0 {
                return Err(invalid(format!(
                    "external leg {source_slot} retains no source states"
                )));
            }
            let source_slot_u32 = u32::try_from(source_slot)
                .map_err(|_| invalid("external source slot exceeds u32"))?;
            let momentum_population = self.bitset_ranges[leg.momentum_mask_id as usize].bit_count;
            if momentum_population != 1
                || !masks.contains(leg.momentum_mask_id, source_slot as u64)?
            {
                return Err(invalid(format!(
                    "external leg {source_slot} momentum mask must contain exactly its source slot"
                )));
            }

            let rows = leg.source_state_range.as_usize_range(
                self.source_states.len(),
                &format!("external leg {source_slot} source states"),
            )?;
            let mut helicities = BTreeSet::new();
            let mut source_template_ids = BTreeSet::new();
            for (state_index, row_index) in rows.enumerate() {
                let state = self.source_states[row_index];
                if state.source_slot != source_slot_u32 {
                    return Err(invalid(format!(
                        "source-state row {row_index} belongs to source slot {}, expected {source_slot_u32}",
                        state.source_slot
                    )));
                }
                let expected_state_index = u32::try_from(state_index)
                    .map_err(|_| invalid("source-state index exceeds u32"))?;
                if state.state_index != expected_state_index {
                    return Err(invalid(format!(
                        "source-state row {row_index} has local index {}, expected {expected_state_index}",
                        state.state_index
                    )));
                }
                if !helicities.insert(state.public_helicity) {
                    return Err(invalid(format!(
                        "external leg {source_slot} repeats public helicity {}",
                        state.public_helicity
                    )));
                }
                if !source_template_ids.insert(state.source_template_id) {
                    return Err(invalid(format!(
                        "external leg {source_slot} repeats source template {}",
                        state.source_template_id
                    )));
                }
                validate_concrete_spin_state(
                    state.spin_state,
                    &format!("source-state row {row_index} spin state"),
                )?;
                if !matches!(state.momentum_sign, -1 | 1) {
                    return Err(invalid(format!(
                        "source-state row {row_index} momentum sign must be -1 or 1"
                    )));
                }
                require_process_template(
                    references,
                    ProcessSemanticTemplateId {
                        kind: ProcessSemanticTemplateKind::CurrentState,
                        template_id: state.current_state_template_id,
                    },
                    &format!("source-state row {row_index} current state"),
                )?;
                require_process_template(
                    references,
                    ProcessSemanticTemplateId {
                        kind: ProcessSemanticTemplateKind::Source,
                        template_id: state.source_template_id,
                    },
                    &format!("source-state row {row_index} source template"),
                )?;
                required_reference(
                    state.crossing_phase_factor_id,
                    catalogs.factors.len(),
                    "source-state crossing phase",
                )?;
            }
        }
        Ok(())
    }

    fn validate_color_sectors(self, catalogs: &ProcessCatalogs<'_>) -> RusticolResult<()> {
        if self.physical_lc_sectors.is_empty() {
            return Err(invalid(
                "recurrence process input requires at least one physical LC sector",
            ));
        }
        let ranges = self
            .physical_lc_sectors
            .iter()
            .map(|row| row.open_string_range)
            .collect::<Vec<_>>();
        validate_packed_ranges(
            "physical LC open strings",
            &ranges,
            self.lc_open_strings.len(),
        )?;

        let mut public_ids = BTreeSet::new();
        for (sector_index, sector) in self.physical_lc_sectors.iter().copied().enumerate() {
            let public_id = required_string(
                &catalogs.strings,
                sector.public_id_string_id,
                "physical LC sector public ID",
            )?;
            if !public_ids.insert(public_id) {
                return Err(invalid(format!(
                    "physical LC sector {sector_index} repeats public identifier {public_id:?}"
                )));
            }
            let kind = sector.kind()?;
            required_reference(
                sector.closure_source_slot,
                self.external_legs.len(),
                "physical LC closure source slot",
            )?;
            let closure_algorithm = required_string(
                &catalogs.strings,
                sector.closure_proof_algorithm_string_id,
                "physical LC closure-anchor proof algorithm",
            )?;
            if !matches!(
                closure_algorithm,
                "canonical-lc-closure-anchor-v2"
                    | "canonical-lc-closure-anchor-v3"
                    | "canonical-lc-closure-anchor-v4"
            ) {
                return Err(invalid(format!(
                    "physical LC sector {sector_index} uses unsupported closure-anchor proof algorithm {closure_algorithm:?}"
                )));
            }
            let closure_proof_digest = required_digest(
                &catalogs.digests,
                sector.closure_proof_digest_id,
                "physical LC closure-anchor proof digest",
            )?;
            let trace = self.sequence(
                sector.trace_sequence_id,
                &format!("physical LC sector {sector_index} trace"),
            )?;
            let singlets = self.sequence(
                sector.singlet_sequence_id,
                &format!("physical LC sector {sector_index} singlets"),
            )?;
            let word = self.sequence(
                sector.word_sequence_id,
                &format!("physical LC sector {sector_index} word"),
            )?;
            validate_source_slot_sequence(
                trace,
                self.external_legs.len(),
                &format!("physical LC sector {sector_index} trace"),
            )?;
            validate_source_slot_sequence(
                singlets,
                self.external_legs.len(),
                &format!("physical LC sector {sector_index} singlets"),
            )?;
            validate_source_slot_sequence(
                word,
                self.external_legs.len(),
                &format!("physical LC sector {sector_index} word"),
            )?;
            if let Some(expected_anchor) = word.last().copied() {
                if closure_algorithm == "canonical-lc-closure-anchor-v2"
                    && sector.closure_source_slot != expected_anchor
                {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v2 closure source slot {} does not match terminal color-word endpoint {expected_anchor}",
                        sector.closure_source_slot
                    )));
                }
                if closure_algorithm == "canonical-lc-closure-anchor-v3"
                    && kind != ProcessLCSectorKind::OpenLines
                    && sector.closure_source_slot != expected_anchor
                {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v3 non-open-line closure source slot {} does not match terminal color-word endpoint {expected_anchor}",
                        sector.closure_source_slot
                    )));
                }
            } else if kind != ProcessLCSectorKind::Singlet {
                return Err(invalid(format!(
                    "non-singlet LC sector {sector_index} has no color word for its closure anchor"
                )));
            }

            match kind {
                ProcessLCSectorKind::OpenLines if sector.open_string_range.count == 0 => {
                    return Err(invalid(format!(
                        "open-lines LC sector {sector_index} has no open strings"
                    )));
                }
                ProcessLCSectorKind::OpenLines if !trace.is_empty() => {
                    return Err(invalid(format!(
                        "open-lines LC sector {sector_index} unexpectedly carries a trace"
                    )));
                }
                ProcessLCSectorKind::SingleTrace if trace.is_empty() => {
                    return Err(invalid(format!(
                        "single-trace LC sector {sector_index} has an empty trace"
                    )));
                }
                ProcessLCSectorKind::SingleTrace if sector.open_string_range.count != 0 => {
                    return Err(invalid(format!(
                        "single-trace LC sector {sector_index} carries open strings"
                    )));
                }
                ProcessLCSectorKind::Singlet
                    if sector.open_string_range.count != 0 || !trace.is_empty() =>
                {
                    return Err(invalid(format!(
                        "singlet LC sector {sector_index} carries colored strings"
                    )));
                }
                _ => {}
            }

            let rows = sector.open_string_range.as_usize_range(
                self.lc_open_strings.len(),
                &format!("physical LC sector {sector_index} open strings"),
            )?;
            let mut open_blocks = Vec::new();
            for (ordinal, row_index) in rows.enumerate() {
                let line = self.lc_open_strings[row_index];
                let expected_sector = u32::try_from(sector_index)
                    .map_err(|_| invalid("physical LC sector index exceeds u32"))?;
                let expected_ordinal = u32::try_from(ordinal)
                    .map_err(|_| invalid("LC open-string ordinal exceeds u32"))?;
                if line.sector_id != expected_sector || line.ordinal != expected_ordinal {
                    return Err(invalid(format!(
                        "LC open-string row {row_index} has parent/ordinal ({}, {}), expected ({expected_sector}, {expected_ordinal})",
                        line.sector_id, line.ordinal
                    )));
                }
                required_reference(
                    line.fundamental_source_slot,
                    self.external_legs.len(),
                    "LC fundamental source slot",
                )?;
                required_reference(
                    line.antifundamental_source_slot,
                    self.external_legs.len(),
                    "LC antifundamental source slot",
                )?;
                let adjoints = self.sequence(
                    line.adjoint_sequence_id,
                    &format!("LC open-string row {row_index} adjoints"),
                )?;
                let line_singlets = self.sequence(
                    line.singlet_sequence_id,
                    &format!("LC open-string row {row_index} singlets"),
                )?;
                validate_source_slot_sequence(
                    adjoints,
                    self.external_legs.len(),
                    &format!("LC open-string row {row_index} adjoints"),
                )?;
                validate_source_slot_sequence(
                    line_singlets,
                    self.external_legs.len(),
                    &format!("LC open-string row {row_index} singlets"),
                )?;
                let mut block = Vec::with_capacity(adjoints.len() + 2);
                block.push(line.fundamental_source_slot);
                block.extend_from_slice(adjoints);
                block.push(line.antifundamental_source_slot);
                open_blocks.push(block);
            }
            if closure_algorithm == "canonical-lc-closure-anchor-v4" {
                if kind != ProcessLCSectorKind::SingleTrace
                    || sector.open_string_range.count != 0
                    || !singlets.is_empty()
                    || trace != word
                {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v4 closure anchoring requires one complete cyclic single-trace source domain"
                    )));
                }
                let mut sorted_word = word.to_vec();
                sorted_word.sort_unstable();
                let expected_domain = (0..self.external_legs.len())
                    .map(|slot| {
                        u32::try_from(slot).map_err(|_| invalid("external source slot exceeds u32"))
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                if sorted_word != expected_domain {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v4 trace does not cover every external source exactly once"
                    )));
                }
                let expected_anchor = word.iter().copied().min().ok_or_else(|| {
                    invalid(format!(
                        "physical LC sector {sector_index} v4 colour word is empty"
                    ))
                })?;
                if sector.closure_source_slot != expected_anchor {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v4 closure source slot {} is not the minimum cyclic trace source {expected_anchor}",
                        sector.closure_source_slot
                    )));
                }
                let fermionic_source_slots = self
                    .external_legs
                    .iter()
                    .enumerate()
                    .filter(|(_, leg)| leg.is_fermionic == 1)
                    .map(|(slot, _)| {
                        u32::try_from(slot).map_err(|_| invalid("external source slot exceeds u32"))
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                let mut state_template_ids = BTreeSet::new();
                for (source_slot, leg) in self.external_legs.iter().copied().enumerate() {
                    let rows = leg.source_state_range.as_usize_range(
                        self.source_states.len(),
                        &format!(
                            "physical LC sector {sector_index} v4 source {source_slot} states"
                        ),
                    )?;
                    if rows.is_empty() {
                        return Err(invalid(format!(
                            "physical LC sector {sector_index} v4 source {source_slot} has no current-state contract"
                        )));
                    }
                    state_template_ids.extend(
                        self.source_states[rows]
                            .iter()
                            .map(|state| state.current_state_template_id),
                    );
                }
                if !fermionic_source_slots.is_empty() || state_template_ids.len() != 1 {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v4 closure anchoring requires one nonfermionic current-state template across every source"
                    )));
                }
            }
            if closure_algorithm == "canonical-lc-closure-anchor-v3" {
                if kind != ProcessLCSectorKind::OpenLines {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} uses v3 closure anchoring outside an open-line sector"
                    )));
                }
                let mut physical_blocks = Vec::with_capacity(open_blocks.len());
                let mut used = vec![false; open_blocks.len()];
                let mut cursor = 0usize;
                while cursor < word.len() {
                    let matches = open_blocks
                        .iter()
                        .enumerate()
                        .filter_map(|(index, block)| {
                            (!used[index]
                                && word.get(cursor..cursor.saturating_add(block.len()))
                                    == Some(block.as_slice()))
                            .then_some(index)
                        })
                        .collect::<Vec<_>>();
                    if matches.len() != 1 {
                        return Err(invalid(format!(
                            "physical LC sector {sector_index} v3 public word has {} open-line block matches at source position {cursor}",
                            matches.len()
                        )));
                    }
                    let block_index = matches[0];
                    used[block_index] = true;
                    cursor = cursor.saturating_add(open_blocks[block_index].len());
                    physical_blocks.push(open_blocks[block_index].clone());
                }
                if cursor != word.len()
                    || physical_blocks.len() != open_blocks.len()
                    || used.iter().any(|value| !value)
                {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v3 public word is not an exact complete-open-line block traversal"
                    )));
                }
                let distinguished_source_slot = word.iter().copied().min().ok_or_else(|| {
                    invalid(format!(
                        "physical LC sector {sector_index} v3 colour word is empty"
                    ))
                })?;
                let distinguished_positions = physical_blocks
                    .iter()
                    .enumerate()
                    .filter_map(|(index, block)| {
                        block.contains(&distinguished_source_slot).then_some(index)
                    })
                    .collect::<Vec<_>>();
                if distinguished_positions.len() != 1 {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v3 minimum coloured source slot belongs to {} open-line blocks",
                        distinguished_positions.len()
                    )));
                }
                let first = distinguished_positions[0];
                let closure_blocks = physical_blocks[first..]
                    .iter()
                    .chain(physical_blocks[..first].iter())
                    .cloned()
                    .collect::<Vec<_>>();
                let closure_traversal = closure_blocks
                    .iter()
                    .flat_map(|block| block.iter().copied())
                    .collect::<Vec<_>>();
                let expected_anchor = closure_traversal.last().copied().ok_or_else(|| {
                    invalid(format!(
                        "physical LC sector {sector_index} v3 closure traversal is empty"
                    ))
                })?;
                if sector.closure_source_slot != expected_anchor {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v3 closure source slot {} does not match independently reconstructed minimum-coloured-source block rotation endpoint {expected_anchor}",
                        sector.closure_source_slot
                    )));
                }
                let process_source_label_order = self
                    .external_legs
                    .iter()
                    .map(|leg| leg.public_label)
                    .collect::<Vec<_>>();
                let fermionic_source_slots = self
                    .external_legs
                    .iter()
                    .enumerate()
                    .filter(|(_, leg)| leg.is_fermionic == 1)
                    .map(|(slot, _)| {
                        u32::try_from(slot).map_err(|_| invalid("external source slot exceeds u32"))
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                let sector_kind = match kind {
                    ProcessLCSectorKind::OpenLines => "open-lines",
                    ProcessLCSectorKind::SingleTrace => "single-trace",
                    ProcessLCSectorKind::Singlet => "singlet",
                };
                let payload = json!({
                    "algorithm": "canonical-lc-closure-anchor-v3",
                    "closure_block_source_slots": closure_blocks,
                    "closure_source_slot": sector.closure_source_slot,
                    "closure_traversal_source_slots": closure_traversal,
                    "component_order_policy": if open_blocks.len() > 1 {
                        "independent-open-string-block-permutation"
                    } else {
                        "exact-ordered-components"
                    },
                    "distinguished_source_block": closure_blocks[0],
                    "distinguished_source_slot": distinguished_source_slot,
                    "external_source_count": self.external_legs.len(),
                    "fermionic_source_slots": fermionic_source_slots,
                    "open_string_count": open_blocks.len(),
                    "physical_block_source_slots": physical_blocks,
                    "policy": "minimum-coloured-source-slot-open-line-block-rotation",
                    "process_source_label_order": process_source_label_order,
                    "sector_id": sector.sector_id,
                    "sector_kind": sector_kind,
                    "selected_closure_block": closure_blocks.last().ok_or_else(|| {
                        invalid(format!(
                            "physical LC sector {sector_index} v3 has no selected closure block"
                        ))
                    })?,
                    "singlet_source_slots": singlets,
                    "word_source_slots": word,
                });
                let canonical = serde_json::to_vec(&payload).map_err(|error| {
                    invalid(format!(
                        "physical LC sector {sector_index} v3 closure proof cannot be canonicalized: {error}"
                    ))
                })?;
                let expected_digest: [u8; 32] = Sha256::digest(canonical).into();
                if closure_proof_digest.as_bytes() != &expected_digest {
                    return Err(invalid(format!(
                        "physical LC sector {sector_index} v3 closure proof digest does not match the independently reconstructed minimum-coloured-source block rotation"
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_public_flows(self, catalogs: &ProcessCatalogs<'_>) -> RusticolResult<()> {
        if self.public_lc_flows.is_empty() {
            return Err(invalid(
                "recurrence process input requires at least one public LC flow",
            ));
        }
        let mut public_ids = BTreeSet::new();
        for (flow_index, flow) in self.public_lc_flows.iter().copied().enumerate() {
            let public_id = required_string(
                &catalogs.strings,
                flow.public_id_string_id,
                "public LC flow identifier",
            )?;
            if !public_ids.insert(public_id) {
                return Err(invalid(format!(
                    "public LC flow {flow_index} repeats identifier {public_id:?}"
                )));
            }
            let construction_sector = self
                .physical_lc_sectors
                .get(required_reference(
                    flow.construction_sector_id,
                    self.physical_lc_sectors.len(),
                    "public-flow construction sector",
                )?)
                .copied()
                .ok_or_else(|| invalid("validated construction sector disappeared"))?;
            let permutation = self.sequence(
                flow.source_slot_permutation_sequence_id,
                &format!("public LC flow {flow_index} source permutation"),
            )?;
            validate_permutation(
                permutation,
                self.external_legs.len(),
                &format!("public LC flow {flow_index} source permutation"),
            )?;
            self.validate_source_contract_permutation(
                permutation,
                &format!("public LC flow {flow_index}"),
            )?;
            let construction_word = self.sequence(
                construction_sector.word_sequence_id,
                &format!("public LC flow {flow_index} construction word"),
            )?;
            let public_word = self.sequence(
                flow.word_sequence_id,
                &format!("public LC flow {flow_index} word"),
            )?;
            let reflected_trace = construction_sector.kind()? == ProcessLCSectorKind::SingleTrace
                && construction_word.len() > 2
                && permutation
                    .iter()
                    .copied()
                    .eq(0..u32::try_from(permutation.len())
                        .map_err(|_| invalid("public LC flow source count exceeds u32"))?)
                && public_word.first() == construction_word.first()
                && public_word[1..]
                    .iter()
                    .copied()
                    .eq(construction_word[1..].iter().rev().copied());
            if !reflected_trace {
                validate_mapped_word(
                    construction_word,
                    permutation,
                    public_word,
                    &format!("public LC flow {flow_index}"),
                )?;
            }
        }
        Ok(())
    }

    fn validate_replay(
        self,
        _catalogs: &ProcessCatalogs<'_>,
        strategy: RecurrenceStrategy,
    ) -> RusticolResult<()> {
        if !strategy.uses_topology_replay_targets()
            && (!self.replay_partitions.is_empty() || !self.replay_targets.is_empty())
        {
            return Err(invalid(
                "non-replay recurrence input must not carry topology-replay rows",
            ));
        }
        let ranges = self
            .replay_partitions
            .iter()
            .map(|row| row.target_range)
            .collect::<Vec<_>>();
        validate_packed_ranges(
            "replay partition targets",
            &ranges,
            self.replay_targets.len(),
        )?;

        let mut covered_sectors = BTreeSet::new();
        for (partition_index, partition) in self.replay_partitions.iter().copied().enumerate() {
            required_reference(
                partition.representative_sector_id,
                self.physical_lc_sectors.len(),
                "replay representative sector",
            )?;
            required_reference(
                partition.materialized_sector_id,
                self.physical_lc_sectors.len(),
                "replay materialized sector",
            )?;
            if partition.target_range.count == 0 {
                return Err(invalid(format!(
                    "replay partition {partition_index} has no targets"
                )));
            }
            let representative_word_id = self.physical_lc_sectors
                [partition.representative_sector_id as usize]
                .word_sequence_id;
            let representative_word = self.sequence(
                representative_word_id,
                &format!("replay partition {partition_index} representative word"),
            )?;
            let representative_anchor = self.physical_lc_sectors
                [partition.representative_sector_id as usize]
                .closure_source_slot;
            let rows = partition.target_range.as_usize_range(
                self.replay_targets.len(),
                &format!("replay partition {partition_index} targets"),
            )?;
            let mut contains_representative = false;
            for row_index in rows {
                let target = self.replay_targets[row_index];
                let expected_partition = u32::try_from(partition_index)
                    .map_err(|_| invalid("replay partition index exceeds u32"))?;
                if target.partition_id != expected_partition {
                    return Err(invalid(format!(
                        "replay target row {row_index} belongs to partition {}, expected {expected_partition}",
                        target.partition_id
                    )));
                }
                required_reference(
                    target.sector_id,
                    self.physical_lc_sectors.len(),
                    "replay target sector",
                )?;
                if !covered_sectors.insert(target.sector_id) {
                    return Err(invalid(format!(
                        "physical LC sector {} belongs to multiple replay partitions",
                        target.sector_id
                    )));
                }
                contains_representative |= target.sector_id == partition.representative_sector_id;
                if !matches!(target.fermion_sign, -1 | 1) {
                    return Err(invalid(format!(
                        "replay target row {row_index} fermion sign must be -1 or 1"
                    )));
                }
                let external_permutation = self.sequence(
                    target.external_permutation_sequence_id,
                    &format!("replay target row {row_index} external permutation"),
                )?;
                let source_permutation = self.sequence(
                    target.source_slot_permutation_sequence_id,
                    &format!("replay target row {row_index} source permutation"),
                )?;
                validate_permutation(
                    external_permutation,
                    self.external_legs.len(),
                    &format!("replay target row {row_index} external permutation"),
                )?;
                validate_permutation(
                    source_permutation,
                    self.external_legs.len(),
                    &format!("replay target row {row_index} source permutation"),
                )?;
                if external_permutation != source_permutation {
                    return Err(invalid(format!(
                        "replay target row {row_index} external and source permutations differ"
                    )));
                }
                self.validate_source_contract_permutation(
                    source_permutation,
                    &format!("replay target row {row_index}"),
                )?;
                let target_word_id =
                    self.physical_lc_sectors[target.sector_id as usize].word_sequence_id;
                let target_word = self.sequence(
                    target_word_id,
                    &format!("replay target row {row_index} target word"),
                )?;
                validate_mapped_word(
                    representative_word,
                    source_permutation,
                    target_word,
                    &format!("replay target row {row_index}"),
                )?;
                let target_anchor =
                    self.physical_lc_sectors[target.sector_id as usize].closure_source_slot;
                let mapped_anchor = *source_permutation
                    .get(representative_anchor as usize)
                    .ok_or_else(|| {
                        invalid("replay representative closure anchor is out of range")
                    })?;
                if mapped_anchor != target_anchor {
                    return Err(invalid(format!(
                        "replay target row {row_index} maps representative closure anchor {representative_anchor} to {mapped_anchor}, expected {target_anchor}"
                    )));
                }
            }
            if !contains_representative {
                return Err(invalid(format!(
                    "replay partition {partition_index} does not contain its representative sector"
                )));
            }
        }
        Ok(())
    }

    fn validate_source_contract_permutation(
        self,
        permutation: &[u32],
        context: &str,
    ) -> RusticolResult<()> {
        for (representative_slot, target_slot) in permutation.iter().copied().enumerate() {
            let target_slot = required_reference(
                target_slot,
                self.external_legs.len(),
                "replay target source slot",
            )?;
            let representative = self.external_legs[representative_slot];
            let target = self.external_legs[target_slot];
            if representative.is_initial != target.is_initial {
                return Err(invalid(format!(
                    "{context} maps source slot {representative_slot} across the initial/final crossing boundary"
                )));
            }
            let contracts = |leg: ProcessExternalLegRow,
                             label: &str|
             -> RusticolResult<BTreeMap<u32, (i32, u32)>> {
                let rows = leg.source_state_range.as_usize_range(
                    self.source_states.len(),
                    &format!("{context} {label} source states"),
                )?;
                let mut result = BTreeMap::new();
                for row_index in rows {
                    let state = self.source_states[row_index];
                    if result
                        .insert(
                            state.source_template_id,
                            (state.momentum_sign, state.crossing_phase_factor_id),
                        )
                        .is_some()
                    {
                        return Err(invalid(format!(
                            "{context} {label} repeats source template {}",
                            state.source_template_id
                        )));
                    }
                }
                Ok(result)
            };
            let representative_contracts = contracts(representative, "representative")?;
            let target_contracts = contracts(target, "target")?;
            if representative_contracts != target_contracts {
                return Err(invalid(format!(
                    "{context} maps source slot {representative_slot} onto slot {target_slot} with a different source crossing contract"
                )));
            }
        }
        Ok(())
    }

    fn validate_generation_coverage(
        self,
        selected_flow_mode: bool,
        selected_source_mode: bool,
    ) -> RusticolResult<()> {
        if selected_flow_mode == self.selected_public_flow_coverage.is_empty() {
            return Err(invalid(
                "selected-flow mode disagrees with generation coverage rows",
            ));
        }
        if selected_source_mode == self.selected_source_coverage.is_empty() {
            return Err(invalid(
                "selected-source mode disagrees with generation coverage rows",
            ));
        }
        let selected_flows = self
            .selected_public_flow_coverage
            .iter()
            .map(|row| row.flow_id)
            .collect::<Vec<_>>();
        validate_strict_ids("selected public-flow coverage", &selected_flows)?;
        validate_u32_references(
            &selected_flows,
            self.public_lc_flows.len(),
            "selected public-flow coverage",
        )?;

        let mut previous = None;
        let mut covered_source_slots = BTreeSet::new();
        for (row_index, row) in self.selected_source_coverage.iter().copied().enumerate() {
            let key = (row.source_slot, row.source_state_index);
            if let Some(previous) = previous
                && previous >= key
            {
                return Err(invalid(format!(
                    "selected source coverage is not in strict canonical order at row {row_index}"
                )));
            }
            previous = Some(key);
            let leg = self
                .external_legs
                .get(required_reference(
                    row.source_slot,
                    self.external_legs.len(),
                    "selected source slot",
                )?)
                .ok_or_else(|| invalid("validated selected source slot disappeared"))?;
            if u64::from(row.source_state_index) >= leg.source_state_range.count {
                return Err(invalid(format!(
                    "selected source coverage row {row_index} references absent state {} for source slot {}",
                    row.source_state_index, row.source_slot
                )));
            }
            covered_source_slots.insert(row.source_slot);
        }
        if selected_source_mode && covered_source_slots.len() != self.external_legs.len() {
            return Err(invalid(
                "selected source coverage must retain at least one state for every external source slot",
            ));
        }
        Ok(())
    }

    fn validate_couplings_and_parameters(
        self,
        catalogs: &ProcessCatalogs<'_>,
        references: &[ProcessSemanticTemplateReference],
    ) -> RusticolResult<()> {
        let mut previous_name = None;
        for (row_index, row) in self.coupling_limits.iter().copied().enumerate() {
            let name =
                required_string(&catalogs.strings, row.name_string_id, "coupling-limit name")?;
            if let Some(previous) = previous_name
                && previous >= name
            {
                return Err(invalid(format!(
                    "coupling limits are not in strict canonical name order at row {row_index}"
                )));
            }
            previous_name = Some(name);
            if row.minimum > row.maximum {
                return Err(invalid(format!(
                    "coupling limit {name:?} minimum {} exceeds maximum {}",
                    row.minimum, row.maximum
                )));
            }
        }

        let mut runtime_components = BTreeSet::new();
        for (row_index, row) in self.parameter_projection.iter().copied().enumerate() {
            let runtime_name = required_string(
                &catalogs.strings,
                row.runtime_name_string_id,
                "runtime parameter name",
            )?;
            if !runtime_components.insert((runtime_name, row.component)) {
                return Err(invalid(format!(
                    "parameter projection row {row_index} repeats runtime parameter component ({runtime_name:?}, {})",
                    row.component
                )));
            }
            require_process_template(
                references,
                ProcessSemanticTemplateId {
                    kind: ProcessSemanticTemplateKind::Parameter,
                    template_id: row.parameter_template_id,
                },
                &format!("parameter projection row {row_index}"),
            )?;
        }
        Ok(())
    }

    fn validate_normalization(self, catalogs: &ProcessCatalogs<'_>) -> RusticolResult<()> {
        if self.normalization.len() != 1 {
            return Err(invalid(format!(
                "recurrence process normalization must contain one row, found {}",
                self.normalization.len()
            )));
        }
        let row = self.normalization[0];
        required_reference(
            row.factor_id,
            catalogs.factors.len(),
            "process normalization factor",
        )?;
        required_string(
            &catalogs.strings,
            row.convention_string_id,
            "process normalization convention",
        )?;
        required_digest(
            &catalogs.digests,
            row.semantic_digest_id,
            "process normalization semantic digest",
        )?;
        Ok(())
    }

    fn validate_fermion_pairing(
        self,
        process_id: &str,
        semantic_identity: &RecurrenceProcessSemanticIdentity,
    ) -> RusticolResult<Option<FermionPairingValidationSummary>> {
        let topology_role = semantic_identity
            .extension_digests()
            .get("fermion-pairing-topology")
            .copied();
        let semantic_role = semantic_identity
            .extension_digests()
            .get("fermion-pairing-semantic")
            .copied();
        let Some(pairing) = self.fermion_pairing else {
            if topology_role.is_some() || semantic_role.is_some() {
                return Err(invalid(
                    "recurrence process claims fermion-pairing digests without pairing tables",
                ));
            }
            return Ok(None);
        };
        let topology_role = topology_role.ok_or_else(|| {
            invalid("recurrence fermion-pairing tables lack their topology digest role")
        })?;
        let semantic_role = semantic_role.ok_or_else(|| {
            invalid("recurrence fermion-pairing tables lack their semantic digest role")
        })?;
        let summary = pairing.validate_internal(process_id)?;
        if summary.topology_digest != topology_role {
            return Err(invalid(
                "fermion-pairing topology digest disagrees with the process semantic identity",
            ));
        }
        if summary.semantic_digest != semantic_role {
            return Err(invalid(
                "fermion-pairing semantic digest disagrees with the process semantic identity",
            ));
        }
        if summary.source_count as usize != self.external_legs.len() {
            return Err(invalid(
                "fermion-pairing source count does not match recurrence external legs",
            ));
        }

        let mut expected_orientations = BTreeMap::<u32, u8>::new();
        for line in self.lc_open_strings {
            for (source_slot, orientation) in [
                (line.fundamental_source_slot, 0_u8),
                (line.antifundamental_source_slot, 1_u8),
            ] {
                if let Some(previous) = expected_orientations.insert(source_slot, orientation)
                    && previous != orientation
                {
                    return Err(invalid(
                        "LC sectors assign inconsistent orientations to one fermion endpoint",
                    ));
                }
            }
        }
        let endpoint_slots = pairing
            .endpoints
            .iter()
            .map(|row| row.source_slot)
            .collect::<BTreeSet<_>>();
        if endpoint_slots != expected_orientations.keys().copied().collect() {
            return Err(invalid(
                "fermion-pairing endpoints do not cover the LC open-line endpoints",
            ));
        }
        for endpoint in pairing.endpoints {
            let source = self
                .external_legs
                .get(required_reference(
                    endpoint.source_slot,
                    self.external_legs.len(),
                    "fermion-pairing endpoint source",
                )?)
                .ok_or_else(|| invalid("validated pairing endpoint source disappeared"))?;
            if !decode_bool(source.is_fermionic, "external_legs.is_fermionic")? {
                return Err(invalid(
                    "fermion-pairing endpoint references a non-fermionic source",
                ));
            }
            if endpoint.public_label != source.public_label {
                return Err(invalid(
                    "fermion-pairing endpoint public label disagrees with its source",
                ));
            }
            if Some(&endpoint.color_orientation) != expected_orientations.get(&endpoint.source_slot)
            {
                return Err(invalid(
                    "fermion-pairing endpoint orientation disagrees with LC sectors",
                ));
            }
        }
        Ok(Some(summary))
    }

    fn sequence(self, sequence_id: u32, label: &str) -> RusticolResult<&'a [u32]> {
        let range = self
            .u32_sequence_ranges
            .get(required_reference(
                sequence_id,
                self.u32_sequence_ranges.len(),
                label,
            )?)
            .copied()
            .ok_or_else(|| invalid(format!("validated {label} disappeared")))?;
        let values = &self.u32_sequence_values
            [range.as_usize_range(self.u32_sequence_values.len(), label)?];
        Ok(values)
    }
}

fn pairing_range_slice<'a, T>(
    values: &'a [T],
    range: CheckedTableRange,
    label: &str,
) -> RusticolResult<&'a [T]> {
    Ok(&values[range.as_usize_range(values.len(), label)?])
}

fn checked_factorial_u64(value: usize, label: &str) -> RusticolResult<u64> {
    (2..=value).try_fold(1_u64, |product, factor| {
        let factor =
            u64::try_from(factor).map_err(|_| invalid(format!("{label} factor exceeds u64")))?;
        product
            .checked_mul(factor)
            .ok_or_else(|| invalid(format!("{label} exceeds u64")))
    })
}

fn validate_pairing_ranges(
    label: &str,
    ranges: impl Iterator<Item = CheckedTableRange>,
    value_len: usize,
) -> RusticolResult<()> {
    let ranges = ranges.collect::<Vec<_>>();
    validate_packed_ranges(label, &ranges, value_len)
}

fn validate_strict_u32_values(values: &[u32], bound: usize, label: &str) -> RusticolResult<()> {
    let mut previous = None;
    for (index, value) in values.iter().copied().enumerate() {
        required_reference(value, bound, label)?;
        if let Some(previous) = previous
            && previous >= value
        {
            return Err(invalid(format!(
                "{label} are not in strict canonical order at row {index}"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_strict_i32_values(values: &[i32], label: &str) -> RusticolResult<()> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid(format!(
            "{label} are not in strict canonical order"
        )));
    }
    Ok(())
}

fn exact_integer_equals(
    input: FermionPairingInputView<'_>,
    integer_id: u32,
    expected: i32,
) -> RusticolResult<bool> {
    let row = input
        .exact_integers
        .get(required_reference(
            integer_id,
            input.exact_integers.len(),
            "fermion-pairing exact integer",
        )?)
        .copied()
        .ok_or_else(|| invalid("validated fermion-pairing exact integer disappeared"))?;
    let limbs = pairing_range_slice(
        input.exact_integer_limbs,
        row.limb_range,
        "fermion-pairing exact-integer magnitude",
    )?;
    Ok(match expected {
        -1 => row.sign == -1 && limbs == [1],
        0 => row.sign == 0 && limbs.is_empty(),
        1 => row.sign == 1 && limbs == [1],
        _ => false,
    })
}

fn decode_bool(value: u8, label: &str) -> RusticolResult<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid(format!(
            "{label} must be zero or one, found {value}"
        ))),
    }
}

fn required_reference(id: u32, target_len: usize, label: &str) -> RusticolResult<usize> {
    let index =
        usize::try_from(id).map_err(|_| invalid(format!("{label} id {id} exceeds usize")))?;
    if index >= target_len {
        return Err(invalid(format!(
            "{label} references id {id}, target length is {target_len}"
        )));
    }
    Ok(index)
}

fn required_string<'a>(strings: &'a [&str], id: u32, label: &str) -> RusticolResult<&'a str> {
    strings
        .get(required_reference(id, strings.len(), label)?)
        .copied()
        .ok_or_else(|| invalid(format!("validated {label} disappeared")))
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
        .ok_or_else(|| invalid(format!("validated {label} disappeared")))
}

fn take_required_role(
    values: &mut BTreeMap<ProcessDigestRole, SemanticDigest>,
    role: ProcessDigestRole,
) -> RusticolResult<SemanticDigest> {
    values.remove(&role).ok_or_else(|| {
        invalid(format!(
            "recurrence process header is missing semantic digest role {:?}",
            role.as_str()
        ))
    })
}

fn validate_canonical_ids(
    label: &str,
    values: impl IntoIterator<Item = u32>,
) -> RusticolResult<()> {
    for (index, value) in values.into_iter().enumerate() {
        let expected =
            u32::try_from(index).map_err(|_| invalid(format!("{label} row index exceeds u32")))?;
        if value != expected {
            return Err(invalid(format!(
                "{label} row {index} contains id {value}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn validate_strict_ids(label: &str, values: &[u32]) -> RusticolResult<()> {
    let mut previous = None;
    for (index, value) in values.iter().copied().enumerate() {
        if let Some(previous) = previous
            && previous >= value
        {
            return Err(invalid(format!(
                "{label} is not in strict ascending order at row {index}"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_source_slot_sequence(
    values: &[u32],
    external_leg_count: usize,
    label: &str,
) -> RusticolResult<()> {
    let mut seen = BTreeSet::new();
    for value in values.iter().copied() {
        required_reference(value, external_leg_count, label)?;
        if !seen.insert(value) {
            return Err(invalid(format!(
                "{label} repeats external source slot {value}"
            )));
        }
    }
    Ok(())
}

fn validate_permutation(values: &[u32], expected_len: usize, label: &str) -> RusticolResult<()> {
    if values.len() != expected_len {
        return Err(invalid(format!(
            "{label} has {} entries, expected {expected_len}",
            values.len()
        )));
    }
    let mut seen = vec![false; expected_len];
    for value in values.iter().copied() {
        let index = required_reference(value, expected_len, label)?;
        if seen[index] {
            return Err(invalid(format!("{label} repeats source slot {value}")));
        }
        seen[index] = true;
    }
    Ok(())
}

fn validate_mapped_word(
    source_word: &[u32],
    permutation: &[u32],
    target_word: &[u32],
    label: &str,
) -> RusticolResult<()> {
    if source_word.len() != target_word.len() {
        return Err(invalid(format!(
            "{label} maps a word of length {} onto length {}",
            source_word.len(),
            target_word.len()
        )));
    }
    for (position, (source_slot, expected_target)) in source_word
        .iter()
        .copied()
        .zip(target_word.iter().copied())
        .enumerate()
    {
        let actual_target = permutation
            .get(required_reference(
                source_slot,
                permutation.len(),
                &format!("{label} source word"),
            )?)
            .copied()
            .ok_or_else(|| invalid(format!("validated {label} permutation entry disappeared")))?;
        if actual_target != expected_target {
            return Err(invalid(format!(
                "{label} maps word position {position} to {actual_target}, expected {expected_target}"
            )));
        }
    }
    Ok(())
}

fn require_process_template(
    references: &[ProcessSemanticTemplateReference],
    typed_id: ProcessSemanticTemplateId,
    label: &str,
) -> RusticolResult<()> {
    if references
        .binary_search_by_key(&typed_id, |reference| reference.typed_id)
        .is_err()
    {
        return Err(invalid(format!(
            "{label} references absent semantic template ({:?}, {})",
            typed_id.kind.as_str(),
            typed_id.template_id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    const TEST_PROCESS_ID: &str = "max-quark-pairs-scaling";

    fn test_digest(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn test_range(start: usize, count: usize) -> CheckedTableRange {
        CheckedTableRange::new(
            u64::try_from(start).expect("test table start fits u64"),
            u64::try_from(count).expect("test table count fits u64"),
        )
    }

    fn v3_no_rotation_digest() -> [u8; 32] {
        let payload = json!({
            "algorithm": "canonical-lc-closure-anchor-v3",
            "closure_block_source_slots": [[1, 0], [2, 3]],
            "closure_source_slot": 3,
            "closure_traversal_source_slots": [1, 0, 2, 3],
            "component_order_policy": "independent-open-string-block-permutation",
            "distinguished_source_block": [1, 0],
            "distinguished_source_slot": 0,
            "external_source_count": 4,
            "fermionic_source_slots": [0, 1, 2, 3],
            "open_string_count": 2,
            "physical_block_source_slots": [[1, 0], [2, 3]],
            "policy": "minimum-coloured-source-slot-open-line-block-rotation",
            "process_source_label_order": [1, 2, 3, 4],
            "sector_id": 0,
            "sector_kind": "open-lines",
            "selected_closure_block": [2, 3],
            "singlet_source_slots": [],
            "word_source_slots": [1, 0, 2, 3],
        });
        Sha256::digest(serde_json::to_vec(&payload).expect("test v3 payload must canonicalize"))
            .into()
    }

    fn validate_v3_color_fixture(
        word: &[u32],
        public_labels: [u32; 4],
        open_strings: &[ProcessLCOpenStringRow],
        closure_source_slot: u32,
        closure_digest: [u8; 32],
    ) -> RusticolResult<()> {
        let external_legs = public_labels
            .into_iter()
            .enumerate()
            .map(|(source_slot, public_label)| ProcessExternalLegRow {
                source_slot: u32::try_from(source_slot).expect("test source slot fits u32"),
                public_label,
                physical_pdg: 1,
                outgoing_pdg: 1,
                is_initial: u8::from(source_slot < 2),
                is_fermionic: 1,
                source_state_range: test_range(0, 0),
                momentum_mask_id: 0,
                support_mask_id: 0,
            })
            .collect::<Vec<_>>();
        let physical_sectors = [ProcessPhysicalLCSectorRow {
            sector_id: 0,
            public_id_string_id: 0,
            kind: ProcessLCSectorKind::OpenLines as u8,
            closure_source_slot,
            closure_proof_algorithm_string_id: 1,
            closure_proof_digest_id: 0,
            open_string_range: test_range(0, open_strings.len()),
            trace_sequence_id: 0,
            singlet_sequence_id: 0,
            word_sequence_id: 1,
            support_mask_id: 0,
        }];
        let mut sequence_values = word.to_vec();
        sequence_values.push(2);
        let sequence_ranges = [
            test_range(0, 0),
            test_range(0, word.len()),
            test_range(word.len(), 1),
        ];
        let strings = vec!["flow:2,1,3,4", "canonical-lc-closure-anchor-v3"];
        let catalogs = ProcessCatalogs {
            strings,
            digests: vec![
                SemanticDigest::new(closure_digest).expect("test closure digest must be nonzero"),
            ],
            factors: Vec::new(),
        };
        let input = RecurrenceProcessInputView {
            input_abi: RECURRENCE_BUILDER_INPUT_ABI,
            declared_input_digest: SemanticDigest::new(test_digest(31))
                .expect("test process digest is nonzero"),
            fermion_pairing: None,
            bitset_ranges: &[],
            bitset_words: &[],
            coupling_limits: &[],
            digest_catalog: &[],
            exact_factors: &[],
            external_legs: &external_legs,
            header: &[],
            header_digests: &[],
            lc_open_strings: open_strings,
            normalization: &[],
            parameter_projection: &[],
            physical_lc_sectors: &physical_sectors,
            public_lc_flows: &[],
            replay_partitions: &[],
            replay_targets: &[],
            selected_public_flow_coverage: &[],
            selected_source_coverage: &[],
            semantic_template_references: &[],
            source_states: &[],
            string_ranges: &[],
            string_bytes: &[],
            u32_sequence_ranges: &sequence_ranges,
            u32_sequence_values: &sequence_values,
        };
        input.validate_color_sectors(&catalogs)
    }

    fn validate_v4_color_fixture(
        word: &[u32],
        closure_source_slot: u32,
        state_template_ids: [u32; 4],
        fermionic_source_slot: Option<usize>,
    ) -> RusticolResult<()> {
        let external_legs = (0..4)
            .map(|source_slot| ProcessExternalLegRow {
                source_slot,
                public_label: source_slot + 1,
                physical_pdg: 21,
                outgoing_pdg: 21,
                is_initial: u8::from(source_slot < 2),
                is_fermionic: u8::from(fermionic_source_slot == Some(source_slot as usize)),
                source_state_range: test_range(source_slot as usize, 1),
                momentum_mask_id: 0,
                support_mask_id: 0,
            })
            .collect::<Vec<_>>();
        let source_states = state_template_ids
            .into_iter()
            .enumerate()
            .map(
                |(source_slot, current_state_template_id)| ProcessSourceStateRow {
                    source_slot: source_slot as u32,
                    state_index: 0,
                    public_helicity: 1,
                    chirality: 0,
                    spin_state: 1,
                    current_state_template_id,
                    source_template_id: source_slot as u32,
                    momentum_sign: 1,
                    crossing_phase_factor_id: 0,
                },
            )
            .collect::<Vec<_>>();
        let physical_sectors = [ProcessPhysicalLCSectorRow {
            sector_id: 0,
            public_id_string_id: 0,
            kind: ProcessLCSectorKind::SingleTrace as u8,
            closure_source_slot,
            closure_proof_algorithm_string_id: 1,
            closure_proof_digest_id: 0,
            open_string_range: test_range(0, 0),
            trace_sequence_id: 1,
            singlet_sequence_id: 0,
            word_sequence_id: 1,
            support_mask_id: 0,
        }];
        let sequence_ranges = [test_range(0, 0), test_range(0, word.len())];
        let catalogs = ProcessCatalogs {
            strings: vec!["flow:1,2,3,4", "canonical-lc-closure-anchor-v4"],
            digests: vec![
                SemanticDigest::new(test_digest(33)).expect("test closure digest must be nonzero"),
            ],
            factors: Vec::new(),
        };
        let input = RecurrenceProcessInputView {
            input_abi: RECURRENCE_BUILDER_INPUT_ABI,
            declared_input_digest: SemanticDigest::new(test_digest(32))
                .expect("test process digest is nonzero"),
            fermion_pairing: None,
            bitset_ranges: &[],
            bitset_words: &[],
            coupling_limits: &[],
            digest_catalog: &[],
            exact_factors: &[],
            external_legs: &external_legs,
            header: &[],
            header_digests: &[],
            lc_open_strings: &[],
            normalization: &[],
            parameter_projection: &[],
            physical_lc_sectors: &physical_sectors,
            public_lc_flows: &[],
            replay_partitions: &[],
            replay_targets: &[],
            selected_public_flow_coverage: &[],
            selected_source_coverage: &[],
            semantic_template_references: &[],
            source_states: &source_states,
            string_ranges: &[],
            string_bytes: &[],
            u32_sequence_ranges: &sequence_ranges,
            u32_sequence_values: word,
        };
        input.validate_color_sectors(&catalogs)
    }

    #[test]
    fn v3_closure_recomputes_no_rotation_mapping_and_rejects_stale_inputs() {
        let open_strings = [
            ProcessLCOpenStringRow {
                sector_id: 0,
                ordinal: 0,
                fundamental_source_slot: 1,
                antifundamental_source_slot: 0,
                adjoint_sequence_id: 0,
                singlet_sequence_id: 0,
            },
            ProcessLCOpenStringRow {
                sector_id: 0,
                ordinal: 1,
                fundamental_source_slot: 2,
                antifundamental_source_slot: 3,
                adjoint_sequence_id: 0,
                singlet_sequence_id: 0,
            },
        ];
        let digest = v3_no_rotation_digest();
        validate_v3_color_fixture(&[1, 0, 2, 3], [1, 2, 3, 4], &open_strings, 3, digest)
            .expect("valid no-rotation v3 fixture must pass");

        for error in [
            validate_v3_color_fixture(&[2, 3, 1, 0], [1, 2, 3, 4], &open_strings, 3, digest)
                .expect_err("stale public block order must fail"),
            validate_v3_color_fixture(&[1, 0, 2, 3], [99, 2, 3, 4], &open_strings, 3, digest)
                .expect_err("stale public-label/source map must fail"),
            validate_v3_color_fixture(&[1, 0, 2, 3], [1, 2, 3, 4], &open_strings, 0, digest)
                .expect_err("stale closure source slot must fail"),
        ] {
            assert!(
                error.to_string().contains("v3 closure"),
                "unexpected v3 tamper error: {error}"
            );
        }

        let mut endpoint_tamper = open_strings;
        endpoint_tamper[0].fundamental_source_slot = 2;
        assert!(
            validate_v3_color_fixture(&[1, 0, 2, 3], [1, 2, 3, 4], &endpoint_tamper, 3, digest,)
                .expect_err("stale endpoint map must fail")
                .to_string()
                .contains("v3 public word")
        );

        let mut adjoint_tamper = open_strings;
        adjoint_tamper[0].adjoint_sequence_id = 2;
        assert!(
            validate_v3_color_fixture(&[1, 0, 2, 3], [1, 2, 3, 4], &adjoint_tamper, 3, digest,)
                .expect_err("stale adjoint map must fail")
                .to_string()
                .contains("v3 public word")
        );
    }

    fn encode_test_strings(
        values: impl IntoIterator<Item = String>,
    ) -> (Vec<u8>, Vec<CheckedTableRange>, BTreeMap<String, u32>) {
        let values = values.into_iter().collect::<BTreeSet<_>>();
        let mut bytes = Vec::new();
        let mut ranges = Vec::with_capacity(values.len());
        let mut ids = BTreeMap::new();
        for (index, value) in values.into_iter().enumerate() {
            let start = bytes.len();
            bytes.extend_from_slice(value.as_bytes());
            ranges.push(test_range(start, value.len()));
            ids.insert(
                value,
                u32::try_from(index).expect("test string count fits u32"),
            );
        }
        (bytes, ranges, ids)
    }

    fn unrank_test_permutation(values: &[u32], mut rank: u64) -> Vec<u32> {
        let mut available = values.to_vec();
        let mut permutation = Vec::with_capacity(values.len());
        for remaining in (1..=values.len()).rev() {
            let block = checked_factorial_u64(remaining - 1, "test permutation block")
                .expect("test permutation factorial fits u64");
            let index = usize::try_from(rank / block).expect("test permutation index fits usize");
            rank %= block;
            permutation.push(available.remove(index));
        }
        assert_eq!(rank, 0);
        permutation
    }

    fn test_permutation_parity(values: &[u32]) -> i32 {
        let inversions = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                values[index + 1..]
                    .iter()
                    .filter(|other| value > *other)
                    .count()
            })
            .sum::<usize>();
        if inversions % 2 == 0 { 1 } else { -1 }
    }

    fn pairing_fixture(pair_counts_by_species: &[usize]) -> OwnedFermionPairingInput {
        assert!(pair_counts_by_species.iter().all(|count| *count > 0));
        let species = (0..pair_counts_by_species.len())
            .map(|index| format!("species-{index:03}"))
            .collect::<Vec<_>>();
        let strings = [
            RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI.to_owned(),
            RECURRENCE_FERMION_PAIRING_PROOF_ALGORITHM.to_owned(),
            TEST_PROCESS_ID.to_owned(),
            "anti-state".to_owned(),
            "basis".to_owned(),
            "state".to_owned(),
        ]
        .into_iter()
        .chain(species.iter().cloned());
        let (string_bytes, string_ranges, string_ids) = encode_test_strings(strings);
        let string_id = |value: &str| {
            *string_ids
                .get(value)
                .unwrap_or_else(|| panic!("missing test string {value:?}"))
        };

        let mut endpoints = Vec::new();
        let mut endpoint_state_template_ids = Vec::new();
        let mut endpoint_anti_state_template_ids = Vec::new();
        let mut endpoint_basis_ids = Vec::new();
        let mut endpoint_color_representations = Vec::new();
        let mut pairing_classes = Vec::new();
        let mut class_fundamental_slots = Vec::new();
        let mut class_antifundamental_slots = Vec::new();
        let mut class_reference_pairings = Vec::new();
        let mut class_fundamentals = Vec::new();
        let mut class_antifundamentals = Vec::new();
        let mut next_source_slot = 0_u32;

        for (class_id, (&pair_count, species)) in pair_counts_by_species
            .iter()
            .zip(species.iter())
            .enumerate()
        {
            let mut fundamental = Vec::with_capacity(pair_count);
            let mut antifundamental = Vec::with_capacity(pair_count);
            for _ in 0..pair_count {
                fundamental.push(next_source_slot);
                antifundamental.push(next_source_slot + 1);
                next_source_slot += 2;
            }

            for (fundamental_slot, antifundamental_slot) in fundamental
                .iter()
                .copied()
                .zip(antifundamental.iter().copied())
            {
                for (source_slot, particle_orientation, color_orientation, representation) in [
                    (fundamental_slot, 0, 0, 3),
                    (antifundamental_slot, 1, 1, -3),
                ] {
                    let state_start = endpoint_state_template_ids.len();
                    endpoint_state_template_ids.push(string_id("state"));
                    let anti_state_start = endpoint_anti_state_template_ids.len();
                    endpoint_anti_state_template_ids.push(string_id("anti-state"));
                    let basis_start = endpoint_basis_ids.len();
                    endpoint_basis_ids.push(string_id("basis"));
                    let representation_start = endpoint_color_representations.len();
                    endpoint_color_representations.push(representation);
                    endpoints.push(FermionPairingEndpointRow {
                        endpoint_id: u32::try_from(endpoints.len())
                            .expect("test endpoint count fits u32"),
                        source_slot,
                        public_label: source_slot + 1,
                        species_class_id: u32::try_from(class_id)
                            .expect("test class count fits u32"),
                        species_string_id: string_id(species),
                        particle_orientation,
                        color_orientation,
                        state_template_range: test_range(state_start, 1),
                        anti_state_template_range: test_range(anti_state_start, 1),
                        basis_range: test_range(basis_start, 1),
                        color_representation_range: test_range(representation_start, 1),
                        contract_digest: test_digest(4),
                    });
                }
            }

            let fundamental_start = class_fundamental_slots.len();
            class_fundamental_slots.extend_from_slice(&fundamental);
            let antifundamental_start = class_antifundamental_slots.len();
            class_antifundamental_slots.extend_from_slice(&antifundamental);
            let reference_start = class_reference_pairings.len();
            class_reference_pairings.extend(
                fundamental
                    .iter()
                    .copied()
                    .zip(antifundamental.iter().copied())
                    .map(|(fundamental_source_slot, antifundamental_source_slot)| {
                        FermionPairingEndpointPairRow {
                            fundamental_source_slot,
                            antifundamental_source_slot,
                        }
                    }),
            );
            pairing_classes.push(FermionPairingClassRow {
                class_id: u32::try_from(class_id).expect("test class count fits u32"),
                species_class_id: u32::try_from(class_id).expect("test class count fits u32"),
                species_string_id: string_id(species),
                fundamental_slot_range: test_range(fundamental_start, pair_count),
                antifundamental_slot_range: test_range(antifundamental_start, pair_count),
                reference_pairing_range: test_range(reference_start, pair_count),
                pairing_count: checked_factorial_u64(pair_count, "test pairing count")
                    .expect("test pairing count fits u64"),
                proof_digest: test_digest(5),
            });
            class_fundamentals.push(fundamental);
            class_antifundamentals.push(antifundamental);
        }

        let source_count = usize::try_from(next_source_slot).expect("test source count fits usize");
        let total_rule_count = pairing_classes
            .iter()
            .try_fold(1_u64, |count, class| count.checked_mul(class.pairing_count))
            .expect("test rule count fits u64");
        let total_rule_count =
            usize::try_from(total_rule_count).expect("test rule count fits usize");
        let mut rules = Vec::with_capacity(total_rule_count);
        let mut rule_class_pairing_indices = Vec::new();
        let mut rule_endpoint_pairings = Vec::new();
        let mut rule_source_slot_permutations = Vec::new();
        let mut rule_lineages = Vec::new();

        for rule_id in 0..total_rule_count {
            let class_index_start = rule_class_pairing_indices.len();
            let endpoint_pairing_start = rule_endpoint_pairings.len();
            let mut mixed_index = u64::try_from(rule_id).expect("test rule ID fits u64");
            let mut parity = 1;
            for (class_id, class) in pairing_classes.iter().enumerate() {
                let local_index = mixed_index % class.pairing_count;
                mixed_index /= class.pairing_count;
                rule_class_pairing_indices.push(FermionPairingClassPairingIndexRow {
                    class_id: u32::try_from(class_id).expect("test class count fits u32"),
                    pairing_index: local_index,
                });
                let permutation =
                    unrank_test_permutation(&class_antifundamentals[class_id], local_index);
                parity *= test_permutation_parity(&permutation);
                rule_endpoint_pairings.extend(
                    class_fundamentals[class_id]
                        .iter()
                        .copied()
                        .zip(permutation)
                        .map(|(fundamental_source_slot, antifundamental_source_slot)| {
                            FermionPairingEndpointPairRow {
                                fundamental_source_slot,
                                antifundamental_source_slot,
                            }
                        }),
                );
            }
            assert_eq!(mixed_index, 0);
            let source_permutation_start = rule_source_slot_permutations.len();
            rule_source_slot_permutations.extend(0..next_source_slot);
            let lineage_start = rule_lineages.len();
            rule_lineages.extend(std::iter::repeat_n(
                RECURRENCE_NO_FERMION_LINE,
                source_count,
            ));
            rules.push(FermionPairingRuleRow {
                rule_id: u32::try_from(rule_id).expect("test rule count fits u32"),
                class_pairing_index_range: test_range(class_index_start, pairing_classes.len()),
                endpoint_pairing_range: test_range(
                    endpoint_pairing_start,
                    pair_counts_by_species.iter().sum(),
                ),
                source_permutation_range: test_range(source_permutation_start, source_count),
                lineage_range: test_range(lineage_start, source_count),
                fermion_parity: parity,
                real_numerator_integer_id: if parity < 0 { 0 } else { 2 },
                real_denominator_integer_id: 2,
                imag_numerator_integer_id: 1,
                imag_denominator_integer_id: 2,
                multiplicity: 1,
                proof_algorithm_string_id: string_id(RECURRENCE_FERMION_PAIRING_PROOF_ALGORITHM),
                proof_digest: test_digest(6),
            });
        }

        let exact_integers = vec![
            FermionPairingExactIntegerRow {
                integer_id: 0,
                sign: -1,
                limb_range: test_range(0, 1),
            },
            FermionPairingExactIntegerRow {
                integer_id: 1,
                sign: 0,
                limb_range: test_range(1, 0),
            },
            FermionPairingExactIntegerRow {
                integer_id: 2,
                sign: 1,
                limb_range: test_range(1, 1),
            },
        ];
        let exact_integer_limbs = vec![1, 1];
        let u32_len = |value: usize| u32::try_from(value).expect("test count fits u32");
        let u64_len = |value: usize| u64::try_from(value).expect("test count fits u64");
        let header = FermionPairingHeaderRow {
            schema_version: RECURRENCE_FERMION_PAIRING_SCHEMA_VERSION,
            abi_string_id: string_id(RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI),
            process_key_string_id: string_id(TEST_PROCESS_ID),
            proof_algorithm_string_id: string_id(RECURRENCE_FERMION_PAIRING_PROOF_ALGORITHM),
            source_count: u32_len(source_count),
            endpoint_count: u32_len(endpoints.len()),
            pairing_class_count: u32_len(pairing_classes.len()),
            rule_count: u32_len(rules.len()),
            endpoint_state_template_count: u64_len(endpoint_state_template_ids.len()),
            endpoint_anti_state_template_count: u64_len(endpoint_anti_state_template_ids.len()),
            endpoint_basis_count: u64_len(endpoint_basis_ids.len()),
            endpoint_color_representation_count: u64_len(endpoint_color_representations.len()),
            class_fundamental_slot_count: u64_len(class_fundamental_slots.len()),
            class_antifundamental_slot_count: u64_len(class_antifundamental_slots.len()),
            class_reference_pairing_count: u64_len(class_reference_pairings.len()),
            rule_class_pairing_index_count: u64_len(rule_class_pairing_indices.len()),
            rule_endpoint_pairing_count: u64_len(rule_endpoint_pairings.len()),
            rule_source_permutation_count: u64_len(rule_source_slot_permutations.len()),
            rule_lineage_count: u64_len(rule_lineages.len()),
            exact_integer_count: u32_len(exact_integers.len()),
            exact_integer_limb_count: u64_len(exact_integer_limbs.len()),
            string_count: u32_len(string_ranges.len()),
            string_byte_count: u64_len(string_bytes.len()),
            no_fermion_line: RECURRENCE_NO_FERMION_LINE,
            topology_digest: test_digest(7),
            semantic_digest: test_digest(8),
        };
        OwnedFermionPairingInput {
            input_abi: RECURRENCE_FERMION_PAIRING_COLUMNAR_ABI.to_owned(),
            declared_columnar_digest: SemanticDigest::new(test_digest(9))
                .expect("test digest is nonzero"),
            header: vec![header],
            endpoints,
            endpoint_state_template_ids,
            endpoint_anti_state_template_ids,
            endpoint_basis_ids,
            endpoint_color_representations,
            pairing_classes,
            class_fundamental_slots,
            class_antifundamental_slots,
            class_reference_pairings,
            rules,
            rule_class_pairing_indices,
            rule_endpoint_pairings,
            rule_source_slot_permutations,
            rule_lineages,
            exact_integers,
            exact_integer_limbs,
            string_ranges,
            string_bytes,
        }
    }

    #[test]
    fn v4_closure_validates_cyclic_anchor_and_homogeneous_source_contract() {
        let word = [0, 2, 1, 3];
        validate_v4_color_fixture(&word, 0, [7; 4], None)
            .expect("valid cyclic trace anchor must pass");

        for error in [
            validate_v4_color_fixture(&word, 1, [7; 4], None)
                .expect_err("noncanonical anchor must fail"),
            validate_v4_color_fixture(&word, 0, [7, 7, 8, 7], None)
                .expect_err("heterogeneous current states must fail"),
            validate_v4_color_fixture(&word, 0, [7; 4], Some(2))
                .expect_err("fermionic trace source must fail"),
        ] {
            assert!(
                error.to_string().contains("v4"),
                "unexpected validation error: {error}"
            );
        }
    }

    fn generation_coverage_view<'a>(
        header: &'a [ProcessHeaderRow],
        external_legs: &'a [ProcessExternalLegRow],
        public_lc_flows: &'a [ProcessPublicLCFlowRow],
        selected_public_flow_coverage: &'a [ProcessSelectedPublicFlowRow],
        selected_source_coverage: &'a [ProcessSelectedSourceStateRow],
    ) -> RecurrenceProcessInputView<'a> {
        RecurrenceProcessInputView {
            input_abi: RECURRENCE_BUILDER_INPUT_ABI,
            declared_input_digest: SemanticDigest::new(test_digest(31))
                .expect("test process digest is nonzero"),
            fermion_pairing: None,
            bitset_ranges: &[],
            bitset_words: &[],
            coupling_limits: &[],
            digest_catalog: &[],
            exact_factors: &[],
            external_legs,
            header,
            header_digests: &[],
            lc_open_strings: &[],
            normalization: &[],
            parameter_projection: &[],
            physical_lc_sectors: &[],
            public_lc_flows,
            replay_partitions: &[],
            replay_targets: &[],
            selected_public_flow_coverage,
            selected_source_coverage,
            semantic_template_references: &[],
            source_states: &[],
            string_ranges: &[],
            string_bytes: &[],
            u32_sequence_ranges: &[],
            u32_sequence_values: &[],
        }
    }

    fn generation_coverage_header(
        strategy: RecurrenceStrategy,
        selected_flow_mode: bool,
        selected_source_mode: bool,
    ) -> ProcessHeaderRow {
        ProcessHeaderRow {
            schema_version: RECURRENCE_PROCESS_INPUT_SCHEMA_VERSION,
            abi_string_id: 0,
            process_id_string_id: 0,
            layout: u8::try_from(strategy.as_u32()).expect("strategy fits u8"),
            selected_flow_mode: u8::from(selected_flow_mode),
            selected_source_mode: u8::from(selected_source_mode),
            external_leg_count: 2,
            physical_sector_count: 0,
            public_flow_count: 2,
            replay_partition_count: 0,
            coupling_limit_count: 0,
            parameter_projection_count: 0,
            process_support_mask_id: 0,
        }
    }

    fn generation_coverage_external_legs() -> [ProcessExternalLegRow; 2] {
        [
            ProcessExternalLegRow {
                source_slot: 0,
                public_label: 1,
                physical_pdg: 1,
                outgoing_pdg: 1,
                is_initial: 1,
                is_fermionic: 1,
                source_state_range: test_range(0, 2),
                momentum_mask_id: 0,
                support_mask_id: 0,
            },
            ProcessExternalLegRow {
                source_slot: 1,
                public_label: 2,
                physical_pdg: -1,
                outgoing_pdg: -1,
                is_initial: 1,
                is_fermionic: 1,
                source_state_range: test_range(2, 2),
                momentum_mask_id: 0,
                support_mask_id: 0,
            },
        ]
    }

    fn generation_coverage_public_flows() -> [ProcessPublicLCFlowRow; 2] {
        [
            ProcessPublicLCFlowRow {
                flow_id: 0,
                public_id_string_id: 0,
                construction_sector_id: 0,
                word_sequence_id: 0,
                source_slot_permutation_sequence_id: 0,
                reduction_weight_factor_id: 0,
            },
            ProcessPublicLCFlowRow {
                flow_id: 1,
                public_id_string_id: 0,
                construction_sector_id: 0,
                word_sequence_id: 0,
                source_slot_permutation_sequence_id: 0,
                reduction_weight_factor_id: 0,
            },
        ]
    }

    #[test]
    fn all_flow_union_accepts_generation_specialized_coverage() {
        let header = [generation_coverage_header(
            RecurrenceStrategy::AllFlowUnion,
            true,
            true,
        )];
        let external_legs = generation_coverage_external_legs();
        let public_flows = generation_coverage_public_flows();
        let selected_flows = [ProcessSelectedPublicFlowRow { flow_id: 1 }];
        let selected_sources = [
            ProcessSelectedSourceStateRow {
                source_slot: 0,
                source_state_index: 1,
            },
            ProcessSelectedSourceStateRow {
                source_slot: 1,
                source_state_index: 0,
            },
        ];
        let input = generation_coverage_view(
            &header,
            &external_legs,
            &public_flows,
            &selected_flows,
            &selected_sources,
        );

        input
            .validate_generation_coverage(true, true)
            .expect("all-flow-union accepts valid generation specialization");
    }

    #[test]
    fn all_flow_union_accepts_omitted_generation_axes() {
        let header = [generation_coverage_header(
            RecurrenceStrategy::AllFlowUnion,
            false,
            false,
        )];
        let external_legs = generation_coverage_external_legs();
        let public_flows = generation_coverage_public_flows();
        let input = generation_coverage_view(&header, &external_legs, &public_flows, &[], &[]);

        input
            .validate_generation_coverage(false, false)
            .expect("omitted union axes retain complete coverage");
    }

    #[test]
    fn selected_source_coverage_rejects_missing_external_slot() {
        let header = [generation_coverage_header(
            RecurrenceStrategy::AllFlowUnion,
            false,
            true,
        )];
        let external_legs = generation_coverage_external_legs();
        let public_flows = generation_coverage_public_flows();
        let selected_sources = [ProcessSelectedSourceStateRow {
            source_slot: 0,
            source_state_index: 1,
        }];
        let input = generation_coverage_view(
            &header,
            &external_legs,
            &public_flows,
            &[],
            &selected_sources,
        );

        let error = input
            .validate_generation_coverage(false, true)
            .expect_err("partial source coverage must fail closed");
        assert!(error.to_string().contains("every external source slot"));
    }

    #[test]
    fn selected_union_coverage_rejects_out_of_range_rows() {
        let header = [generation_coverage_header(
            RecurrenceStrategy::AllFlowUnion,
            true,
            true,
        )];
        let external_legs = generation_coverage_external_legs();
        let public_flows = generation_coverage_public_flows();
        let selected_flows = [ProcessSelectedPublicFlowRow { flow_id: 2 }];
        let selected_sources = [
            ProcessSelectedSourceStateRow {
                source_slot: 0,
                source_state_index: 0,
            },
            ProcessSelectedSourceStateRow {
                source_slot: 1,
                source_state_index: 2,
            },
        ];
        let input = generation_coverage_view(
            &header,
            &external_legs,
            &public_flows,
            &selected_flows,
            &selected_sources,
        );

        assert!(input.validate_generation_coverage(true, true).is_err());
    }

    #[test]
    fn semantic_template_kinds_are_closed_and_round_trip() {
        for kind in [
            ProcessSemanticTemplateKind::Parameter,
            ProcessSemanticTemplateKind::CurrentState,
            ProcessSemanticTemplateKind::Source,
            ProcessSemanticTemplateKind::QuantumFlow,
            ProcessSemanticTemplateKind::Transition,
            ProcessSemanticTemplateKind::Propagator,
            ProcessSemanticTemplateKind::Closure,
            ProcessSemanticTemplateKind::ColorContraction,
            ProcessSemanticTemplateKind::SymmetryProof,
        ] {
            assert_eq!(
                ProcessSemanticTemplateKind::try_from(kind.as_str()),
                Ok(kind)
            );
        }
        assert!(ProcessSemanticTemplateKind::try_from("model-specific").is_err());
    }

    #[test]
    fn process_digest_roles_decode_required_and_extension_roles() {
        assert_eq!(
            ProcessDigestRole::decode("process").unwrap(),
            ProcessDigestRole::Process
        );
        assert_eq!(
            ProcessDigestRole::decode("future-proof").unwrap(),
            ProcessDigestRole::Extension("future-proof".to_owned())
        );
        assert!(ProcessDigestRole::decode("").is_err());
    }

    #[test]
    fn permutation_validation_rejects_duplicates_and_missing_entries() {
        assert!(validate_permutation(&[2, 0, 1], 3, "test permutation").is_ok());
        assert!(validate_permutation(&[0, 0, 2], 3, "test permutation").is_err());
        assert!(validate_permutation(&[0, 1], 3, "test permutation").is_err());
    }

    #[test]
    fn user_max_quark_pairs_above_eight_has_no_hidden_endpoint_cap() {
        let input = pairing_fixture(&[1; 9]);
        let summary = input
            .as_view()
            .validate_internal(TEST_PROCESS_ID)
            .expect("nine independently flavored quark pairs must validate");
        assert_eq!(summary.endpoint_count, 18);
        assert_eq!(summary.pairing_class_count, 9);
        assert_eq!(summary.rule_count, 1);
    }

    #[test]
    fn user_max_quark_pairs_allows_more_than_legacy_pairing_and_rule_caps() {
        let input = pairing_fixture(&[7]);
        assert_eq!(input.pairing_classes[0].pairing_count, 5_040);
        let summary = input
            .as_view()
            .validate_internal(TEST_PROCESS_ID)
            .expect("seven same-species quark pairs and all 5,040 rules must validate");
        assert_eq!(summary.endpoint_count, 14);
        assert_eq!(summary.pairing_class_count, 1);
        assert_eq!(summary.rule_count, 5_040);
    }

    #[test]
    fn pairing_growth_stops_only_at_checked_integer_representation() {
        assert_eq!(
            checked_factorial_u64(20, "test pairing count").unwrap(),
            2_432_902_008_176_640_000
        );
        let error = checked_factorial_u64(21, "test pairing count").unwrap_err();
        assert!(error.to_string().contains("exceeds u64"));
    }
}
