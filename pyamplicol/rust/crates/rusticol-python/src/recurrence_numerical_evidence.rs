// SPDX-License-Identifier: 0BSD

//! Fail-closed authentication of recurrence numerical-current evidence.
//!
//! Python supplies complete raw candidate and independent-verification captures.
//! This module independently reconstructs their schema, probe contexts, exact
//! rational residuals, decision census, certificates, and current mappings
//! before the native Direct-Arena lowering is allowed to apply any reuse.

use flate2::read::ZlibDecoder;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rusticol_core::recurrence::process::ValidatedRecurrenceProcessInput;
use rusticol_core::recurrence::template::{ParameterKind, ValidatedRecurrenceTemplateInput};
use rusticol_core::recurrence::{
    RecurrenceNumericalCurrentMapping, RecurrenceNumericalRelationEvidence,
    RecurrenceRelationDiscoveryMode,
};
use rusticol_core::{RusticolError, RusticolResult};
use serde_json::{Map as JsonMap, Value as JsonValue, value::RawValue as JsonRawValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read, Write};
use std::ops::Range;
use std::str::FromStr;
use std::time::Instant;

use crate::recurrence::{
    canonical_json_bytes, canonical_json_sha256, hex_digest, invalid, json_array, json_bool,
    json_field, json_nonempty_string, json_object, json_string, json_u32, require_json_fields,
    require_json_string_value, semantic_digest_from_hex,
};

type ExactProbeComplex = (BigRational, BigRational);

const COMPRESSED_EVIDENCE_MAGIC: &[u8; 8] = b"PACNCEZ1";
const COMPRESSED_EVIDENCE_HEADER_BYTES: usize = 8 + 8 + 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RawByteRange {
    start: usize,
    end: usize,
}

impl RawByteRange {
    fn from_usize(start: usize, end: usize, limit: usize, context: &str) -> RusticolResult<Self> {
        if start > end || end > limit {
            return Err(invalid(format!("{context} is outside the raw evidence")));
        }
        Ok(Self { start, end })
    }

    fn as_usize(self, limit: usize, context: &str) -> RusticolResult<Range<usize>> {
        let range = self.start..self.end;
        if range.start > range.end || range.end > limit {
            return Err(invalid(format!("{context} is outside the raw evidence")));
        }
        Ok(range)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RawObservationRow {
    row: RawByteRange,
    values: RawByteRange,
}

struct LocatedObservationArrays {
    metadata_bytes: Vec<u8>,
    candidate: RawByteRange,
    verification: RawByteRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LocatedObservationRanges {
    candidate: RawByteRange,
    verification: RawByteRange,
}

fn borrowed_raw_value_range(
    bytes: &[u8],
    value: &JsonRawValue,
    context: &str,
) -> RusticolResult<RawByteRange> {
    let raw = value.get().as_bytes();
    let bytes_start = bytes.as_ptr() as usize;
    let raw_start = raw.as_ptr() as usize;
    let start = raw_start
        .checked_sub(bytes_start)
        .ok_or_else(|| invalid(format!("{context} does not borrow the raw evidence")))?;
    let end = start
        .checked_add(raw.len())
        .ok_or_else(|| invalid(format!("{context} byte range overflows usize")))?;
    RawByteRange::from_usize(start, end, bytes.len(), context)
}

fn locate_raw_observation_ranges(bytes: &[u8]) -> RusticolResult<LocatedObservationRanges> {
    let root = serde_json::from_slice::<BTreeMap<String, &JsonRawValue>>(bytes)
        .map_err(|error| invalid(format!("numerical relation evidence is not JSON: {error}")))?;
    let locate = |field: &str| -> RusticolResult<RawByteRange> {
        let capture = root
            .get(field)
            .ok_or_else(|| invalid(format!("numerical relation evidence has no {field}")))?;
        let capture_fields = serde_json::from_str::<BTreeMap<String, &JsonRawValue>>(capture.get())
            .map_err(|error| {
                invalid(format!(
                    "{field} raw numerical capture is not JSON: {error}"
                ))
            })?;
        let observations = capture_fields
            .get("observations")
            .ok_or_else(|| invalid(format!("{field} raw numerical capture has no observations")))?;
        borrowed_raw_value_range(bytes, observations, &format!("{field} observations"))
    };
    let candidate = locate("candidate_capture")?;
    let verification = locate("verification_capture")?;
    let candidate_range = candidate.as_usize(bytes.len(), "candidate observation-array range")?;
    let verification_range =
        verification.as_usize(bytes.len(), "verification observation-array range")?;
    if candidate_range.end > verification_range.start {
        return Err(invalid(
            "raw numerical observation arrays overlap or are not canonical",
        ));
    }
    Ok(LocatedObservationRanges {
        candidate,
        verification,
    })
}

fn raw_observation_metadata_byte_count(
    bytes: &[u8],
    ranges: LocatedObservationRanges,
) -> RusticolResult<usize> {
    let candidate_range = ranges
        .candidate
        .as_usize(bytes.len(), "candidate observation-array range")?;
    let verification_range = ranges
        .verification
        .as_usize(bytes.len(), "verification observation-array range")?;
    let removed = candidate_range
        .len()
        .checked_add(verification_range.len())
        .ok_or_else(|| invalid("raw observation-array byte count overflows usize"))?;
    bytes
        .len()
        .checked_sub(removed)
        .and_then(|size| size.checked_add(4))
        .ok_or_else(|| invalid("raw numerical metadata size overflows usize"))
}

fn materialize_raw_observation_metadata(
    bytes: &[u8],
    ranges: LocatedObservationRanges,
) -> RusticolResult<Vec<u8>> {
    let candidate_range = ranges
        .candidate
        .as_usize(bytes.len(), "candidate observation-array range")?;
    let verification_range = ranges
        .verification
        .as_usize(bytes.len(), "verification observation-array range")?;
    let capacity = raw_observation_metadata_byte_count(bytes, ranges)?;
    let mut metadata_bytes = Vec::with_capacity(capacity);
    metadata_bytes.extend_from_slice(&bytes[..candidate_range.start]);
    metadata_bytes.extend_from_slice(b"[]");
    metadata_bytes.extend_from_slice(&bytes[candidate_range.end..verification_range.start]);
    metadata_bytes.extend_from_slice(b"[]");
    metadata_bytes.extend_from_slice(&bytes[verification_range.end..]);
    if metadata_bytes.len() != capacity {
        return Err(invalid(
            "raw numerical metadata materialization length drifted",
        ));
    }
    Ok(metadata_bytes)
}

#[cfg(test)]
fn locate_raw_observation_arrays(bytes: &[u8]) -> RusticolResult<LocatedObservationArrays> {
    let ranges = locate_raw_observation_ranges(bytes)?;
    let metadata_bytes = materialize_raw_observation_metadata(bytes, ranges)?;
    Ok(LocatedObservationArrays {
        metadata_bytes,
        candidate: ranges.candidate,
        verification: ranges.verification,
    })
}

struct CanonicalObservationCursor<'a> {
    bytes: &'a [u8],
    position: usize,
    absolute_start: usize,
}

impl<'a> CanonicalObservationCursor<'a> {
    fn expect(&mut self, expected: &[u8], context: &str) -> RusticolResult<()> {
        let end = self
            .position
            .checked_add(expected.len())
            .ok_or_else(|| invalid(format!("{context} offset overflows usize")))?;
        if self.bytes.get(self.position..end) != Some(expected) {
            return Err(invalid(format!("{context} is malformed or not canonical")));
        }
        self.position = end;
        Ok(())
    }

    fn absolute_position(&self, context: &str) -> RusticolResult<usize> {
        self.absolute_start
            .checked_add(self.position)
            .ok_or_else(|| invalid(format!("{context} offset overflows usize")))
    }

    fn decimal_string(&mut self, context: &str) -> RusticolResult<&'a str> {
        self.expect(b"\"", context)?;
        let remaining = self
            .bytes
            .get(self.position..)
            .ok_or_else(|| invalid(format!("{context} is truncated")))?;
        let width = remaining
            .iter()
            .position(|byte| *byte == b'"')
            .ok_or_else(|| invalid(format!("{context} is unterminated")))?;
        let end = self
            .position
            .checked_add(width)
            .ok_or_else(|| invalid(format!("{context} width overflows usize")))?;
        let text = std::str::from_utf8(
            self.bytes
                .get(self.position..end)
                .ok_or_else(|| invalid(format!("{context} is truncated")))?,
        )
        .map_err(|_| invalid(format!("{context} is not ASCII")))?;
        validate_canonical_decimal_text(text, context)?;
        self.position = end;
        self.expect(b"\"", context)?;
        Ok(text)
    }
}

fn scan_raw_observation_array(
    bytes: &[u8],
    array: RawByteRange,
    source: &RawSourceSemantics,
    point_count: usize,
    label: &str,
) -> RusticolResult<Vec<RawObservationRow>> {
    let array_range = array.as_usize(bytes.len(), &format!("{label} observation array"))?;
    let raw_array = bytes
        .get(array_range.clone())
        .ok_or_else(|| invalid(format!("{label} observation array is absent")))?;
    let mut cursor = CanonicalObservationCursor {
        bytes: raw_array,
        position: 0,
        absolute_start: array_range.start,
    };
    cursor.expect(b"[", &format!("{label} observation array"))?;
    let mut rows = Vec::with_capacity(source.currents.len());
    for (index, current) in source.currents.iter().enumerate() {
        if index != 0 {
            cursor.expect(b",", &format!("{label} observation row {index}"))?;
        }
        let row_start = cursor.absolute_position(&format!("{label} observation row {index}"))?;
        cursor.expect(
            b"{\"current_id\":",
            &format!("{label} observation row {index}"),
        )?;
        cursor.expect(
            current.current_id.to_string().as_bytes(),
            &format!("{label} observation row {index} current ID"),
        )?;
        cursor.expect(
            b",\"dimension\":",
            &format!("{label} observation row {index}"),
        )?;
        cursor.expect(
            current.dimension.to_string().as_bytes(),
            &format!("{label} observation row {index} dimension"),
        )?;
        cursor.expect(b",\"values\":", &format!("{label} observation row {index}"))?;
        let values_start =
            cursor.absolute_position(&format!("{label} observation row {index} values"))?;
        cursor.expect(b"[", &format!("{label} observation row {index} values"))?;
        let expected_width = point_count
            .checked_mul(usize::from(current.dimension))
            .ok_or_else(|| {
                invalid(format!(
                    "{label} observation row {index} width overflows usize"
                ))
            })?;
        for component in 0..expected_width {
            let context = format!("{label} observation row {index} component {component}");
            if component != 0 {
                cursor.expect(b",", &context)?;
            }
            cursor.expect(b"[", &context)?;
            cursor.decimal_string(&format!("{context} real part"))?;
            cursor.expect(b",", &context)?;
            cursor.decimal_string(&format!("{context} imaginary part"))?;
            cursor.expect(b"]", &context)?;
        }
        cursor.expect(b"]", &format!("{label} observation row {index} values"))?;
        let values_end =
            cursor.absolute_position(&format!("{label} observation row {index} values"))?;
        cursor.expect(b"}", &format!("{label} observation row {index}"))?;
        let row_end = cursor.absolute_position(&format!("{label} observation row {index}"))?;
        rows.push(RawObservationRow {
            row: RawByteRange::from_usize(
                row_start,
                row_end,
                bytes.len(),
                &format!("{label} observation row {index}"),
            )?,
            values: RawByteRange::from_usize(
                values_start,
                values_end,
                bytes.len(),
                &format!("{label} observation row {index} values"),
            )?,
        });
    }
    cursor.expect(b"]", &format!("{label} observation array"))?;
    if cursor.position != cursor.bytes.len() {
        return Err(invalid(format!(
            "{label} observation array has trailing or extra rows"
        )));
    }
    Ok(rows)
}

struct CanonicalJsonMatcher<'a> {
    expected: &'a [u8],
    offset: usize,
    matches: bool,
}

impl Write for CanonicalJsonMatcher<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let end = self.offset.saturating_add(buffer.len());
        if end > self.expected.len() || self.expected[self.offset..end] != *buffer {
            self.matches = false;
        }
        self.offset = end;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn validate_canonical_json_bytes(
    value: &JsonValue,
    expected: &[u8],
    context: &str,
) -> RusticolResult<()> {
    let mut matcher = CanonicalJsonMatcher {
        expected,
        offset: 0,
        matches: true,
    };
    serde_json::to_writer(&mut matcher, value)
        .map_err(|error| invalid(format!("could not canonicalize {context}: {error}")))?;
    if !matcher.matches || matcher.offset != expected.len() {
        return Err(invalid(format!("{context} is not canonical ASCII JSON")));
    }
    Ok(())
}

fn validate_raw_json_lexical_structure(bytes: &[u8]) -> RusticolResult<()> {
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0_usize;
    for &byte in bytes {
        if !byte.is_ascii() {
            return Err(invalid(
                "numerical relation evidence must be canonical ASCII JSON",
            ));
        }
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => {
                in_string = true;
            }
            b'{' | b'[' => {
                depth += 1;
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
    }
    if in_string || depth != 0 {
        return Err(invalid(
            "numerical relation evidence JSON lexical structure is incomplete",
        ));
    }
    Ok(())
}

fn decode_compressed_evidence(bytes: &[u8]) -> RusticolResult<Vec<u8>> {
    if bytes.len() < COMPRESSED_EVIDENCE_HEADER_BYTES {
        return Err(invalid(
            "compressed numerical relation evidence header is truncated",
        ));
    }
    if bytes.get(..COMPRESSED_EVIDENCE_MAGIC.len()) != Some(COMPRESSED_EVIDENCE_MAGIC) {
        return Err(invalid(
            "compressed numerical relation evidence has invalid magic",
        ));
    }
    let declared_bytes = u64::from_be_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| invalid("compressed numerical evidence length header is malformed"))?,
    );
    let declared_bytes = usize::try_from(declared_bytes)
        .map_err(|_| invalid("compressed numerical evidence length exceeds usize"))?;
    if declared_bytes == 0 {
        return Err(invalid(
            "compressed numerical relation evidence declares an empty payload",
        ));
    }

    let expected_digest = bytes
        .get(16..COMPRESSED_EVIDENCE_HEADER_BYTES)
        .ok_or_else(|| invalid("compressed numerical evidence digest header is malformed"))?;
    let compressed = bytes
        .get(COMPRESSED_EVIDENCE_HEADER_BYTES..)
        .ok_or_else(|| invalid("compressed numerical evidence payload is truncated"))?;
    if compressed.is_empty() {
        return Err(invalid(
            "compressed numerical relation evidence payload is empty",
        ));
    }

    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(declared_bytes)
        .map_err(|_| invalid("could not reserve decompressed numerical evidence"))?;
    let mut decoder = ZlibDecoder::new(compressed);
    {
        let maximum_read = u64::try_from(declared_bytes)
            .ok()
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| invalid("compressed numerical decompression limit overflows u64"))?;
        let mut limited = decoder.by_ref().take(maximum_read);
        limited.read_to_end(&mut decoded).map_err(|error| {
            invalid(format!(
                "compressed numerical relation evidence could not be decompressed: {error}"
            ))
        })?;
    }
    if decoded.len() != declared_bytes {
        return Err(invalid(format!(
            "compressed numerical relation evidence decompressed length disagrees with its declaration (declared={declared_bytes}, actual={})",
            decoded.len()
        )));
    }
    if usize::try_from(decoder.total_in()).ok() != Some(compressed.len()) {
        return Err(invalid(
            "compressed numerical relation evidence has trailing or unconsumed transport bytes",
        ));
    }
    if Sha256::digest(&decoded).as_slice() != expected_digest {
        return Err(RusticolError::integrity(
            "compressed numerical relation evidence digest mismatch",
        ));
    }
    Ok(decoded)
}

#[derive(Clone, Debug)]
struct RawNumericalCapture<'a> {
    raw_bytes: &'a [u8],
    observation_rows: Vec<RawObservationRow>,
    point_count: usize,
    point_sha256s: Vec<String>,
    kinematic_sha256s: Vec<String>,
    parameter_context_sha256s: Vec<String>,
    parameter_contexts: Vec<Vec<BigRational>>,
    dimensions: Vec<u16>,
    observation_batch_sha256: String,
    capture_contract_sha256: String,
}

#[derive(Clone, Debug)]
struct RawSourceCurrent {
    current_id: u32,
    is_source: bool,
    contract_key: Vec<u8>,
    dimension: u16,
    selector_domain_id: u32,
}

#[derive(Clone, Debug)]
struct RawSourceSemantics {
    value: JsonValue,
    process_id: String,
    physical_pdgs: Vec<i32>,
    strategy: String,
    selector_schedule: JsonValue,
    currents: Vec<RawSourceCurrent>,
}

#[derive(Clone, Debug)]
struct DerivedNumericalRelation {
    mapping: RecurrenceNumericalCurrentMapping,
    certificate: JsonValue,
    mapping_json: JsonValue,
}

#[derive(Clone, Debug)]
struct RawNumericalCandidateIndex {
    #[cfg(test)]
    observation_index: usize,
    scalar_component: usize,
    entries: Vec<(BigRational, u32)>,
    selected_values: Vec<(u32, ExactProbeComplex)>,
}

impl RawNumericalCandidateIndex {
    fn selected_value(&self, current_id: u32, context: &str) -> RusticolResult<&ExactProbeComplex> {
        let index = self
            .selected_values
            .binary_search_by_key(&current_id, |(candidate_id, _value)| *candidate_id)
            .map_err(|_| invalid(format!("{context} selected observation is absent")))?;
        Ok(&self.selected_values[index].1)
    }
}

#[derive(Clone, Debug)]
struct RawNumericalDerivation {
    relations: Vec<DerivedNumericalRelation>,
    numerical_candidate_count: usize,
    verification_rejected_count: usize,
    rejected_hypothesis_count: usize,
    tested_hypothesis_count: usize,
    theoretical_pair_hypothesis_count: usize,
    screened_pair_hypothesis_count: usize,
    zero_hypothesis_count: usize,
    candidate_index_contract_count: usize,
    exhaustive_fallback_contract_count: usize,
    decision_sha256: String,
    rejection_decision_sha256: String,
}

/// Internal, generation-only measurements for numerical-evidence replay.
///
/// This record never enters a Python result, recurrence artifact, runtime
/// payload, semantic digest, or public ABI.
#[doc(hidden)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct RecurrenceEvidenceAuthenticationTelemetry {
    pub transport_decode_nanoseconds: u64,
    pub capture_authentication_nanoseconds: u64,
    pub candidate_index_nanoseconds: u64,
    pub hypothesis_replay_nanoseconds: u64,
    pub derivation_finalization_nanoseconds: u64,
    pub evidence_finalization_nanoseconds: u64,
    pub total_nanoseconds: u64,
    pub candidate_index_observation_row_scan_count: usize,
    pub retained_selected_observation_count: usize,
    pub selected_observation_reuse_count: usize,
    pub zero_residual_streamed_value_count: usize,
    pub materialized_observation_vector_count: usize,
    pub rational_string_materialization_count: usize,
    pub rational_string_use_count: usize,
    pub rational_string_reuse_count: usize,
}

impl RecurrenceEvidenceAuthenticationTelemetry {
    fn checked_add(target: &mut usize, value: usize, description: &str) -> RusticolResult<()> {
        *target = target
            .checked_add(value)
            .ok_or_else(|| invalid(format!("{description} exceeds usize")))?;
        Ok(())
    }

    fn record_rational_string_uses(&mut self, count: usize) -> RusticolResult<()> {
        Self::checked_add(
            &mut self.rational_string_use_count,
            count,
            "evidence-authentication rational-string use count",
        )
    }

    fn finish_rational_string_census(&mut self) -> RusticolResult<()> {
        self.rational_string_reuse_count = self
            .rational_string_use_count
            .checked_sub(self.rational_string_materialization_count)
            .ok_or_else(|| {
                invalid(
                    "evidence-authentication rational-string use count is below materialization count",
                )
            })?;
        Ok(())
    }
}

fn elapsed_nanoseconds(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Clone, Debug)]
struct AuthenticatedRuntimeParameterSlot {
    default: BigRational,
    probe_policy: &'static str,
}

#[derive(Clone, Debug)]
pub(super) struct AuthenticatedRuntimeParameterContract {
    schema: JsonValue,
    schema_sha256: String,
    slots: Vec<AuthenticatedRuntimeParameterSlot>,
    process_id: String,
    physical_pdgs: Vec<i32>,
}

#[derive(Clone, Debug)]
struct RelationResiduals {
    maximum_ratio: BigRational,
    strings: RelationResidualStrings,
}

#[derive(Clone, Debug)]
struct RelationResidualStrings {
    maximum_absolute: String,
    maximum_relative: String,
    maximum_ratio: String,
}

pub(super) fn authenticated_runtime_parameter_contract(
    process: &ValidatedRecurrenceProcessInput,
    template: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<AuthenticatedRuntimeParameterContract> {
    const ABI: &str = "pyamplicol-recurrence-runtime-parameter-schema-v2";
    const MATERIAL_POLICY: &str = "native-template-default-perturbed-v1";
    const DERIVED_POLICY: &str = "derived-overwritten-fixed-zero-v1";
    let input = process.input();
    let parameter_contracts = template.parameter_kind_and_default_factors()?;
    let mut parameters = Vec::with_capacity(input.parameter_projection.len());
    let mut slots = Vec::with_capacity(input.parameter_projection.len());
    for (index, row) in input.parameter_projection.iter().enumerate() {
        if row.runtime_slot as usize != index {
            return Err(invalid(
                "authenticated runtime parameter projection is not dense",
            ));
        }
        let range = input
            .string_ranges
            .get(row.runtime_name_string_id as usize)
            .copied()
            .ok_or_else(|| invalid("runtime-parameter name string range is absent"))?
            .as_usize_range(
                input.string_bytes.len(),
                "runtime-parameter name string range",
            )?;
        let runtime_name = std::str::from_utf8(&input.string_bytes[range])
            .map_err(|_| invalid("runtime-parameter name is not UTF-8"))?;
        let (kind, default_factor) = parameter_contracts
            .get(row.parameter_template_id as usize)
            .copied()
            .ok_or_else(|| invalid("runtime-parameter projection references an absent template"))?;
        let (default, default_binary64, probe_policy) = if kind == ParameterKind::Derived {
            (BigRational::zero(), 0.0_f64, DERIVED_POLICY)
        } else {
            let factor = default_factor.ok_or_else(|| {
                invalid("material runtime parameter has no authenticated template default")
            })?;
            let exact = match row.component {
                0 => factor.real(),
                1 => factor.imag(),
                _ => {
                    return Err(invalid(
                        "runtime-parameter component is not real or imaginary",
                    ));
                }
            };
            let rational = BigRational::new(
                BigInt::from(exact.numerator()),
                BigInt::from(exact.denominator()),
            );
            let binary64 = rational.to_f64().ok_or_else(|| {
                invalid("authenticated runtime-parameter default is outside binary64")
            })?;
            if !binary64.is_finite() {
                return Err(invalid(
                    "authenticated runtime-parameter default is not finite binary64",
                ));
            }
            (
                exact_binary64_rational(binary64)?,
                binary64,
                MATERIAL_POLICY,
            )
        };
        parameters.push(serde_json::json!({
            "runtime_slot": row.runtime_slot,
            "runtime_name": runtime_name,
            "parameter_template_id": row.parameter_template_id,
            "prepared_parameter_id": row.prepared_parameter_id(),
            "component": row.component,
            "default_binary64": python_f64_hex_signed(default_binary64),
            "probe_policy": probe_policy,
        }));
        slots.push(AuthenticatedRuntimeParameterSlot {
            default,
            probe_policy,
        });
    }
    let schema = serde_json::json!({
        "abi": ABI,
        "parameters": parameters,
    });
    let schema_sha256 =
        canonical_json_sha256(&schema, "native recurrence runtime-parameter schema")?;
    Ok(AuthenticatedRuntimeParameterContract {
        schema,
        schema_sha256,
        slots,
        process_id: process.summary().process_id().to_owned(),
        physical_pdgs: input
            .external_legs
            .iter()
            .map(|leg| leg.physical_pdg)
            .collect(),
    })
}

pub(super) fn parse_numerical_relation_evidence(
    bytes: &[u8],
    runtime_parameters: &AuthenticatedRuntimeParameterContract,
) -> RusticolResult<RecurrenceNumericalRelationEvidence> {
    parse_numerical_relation_evidence_with_telemetry(bytes, runtime_parameters)
        .map(|(evidence, _telemetry)| evidence)
}

#[doc(hidden)]
pub(super) fn parse_numerical_relation_evidence_with_telemetry(
    bytes: &[u8],
    runtime_parameters: &AuthenticatedRuntimeParameterContract,
) -> RusticolResult<(
    RecurrenceNumericalRelationEvidence,
    RecurrenceEvidenceAuthenticationTelemetry,
)> {
    let total_started = Instant::now();
    let mut telemetry = RecurrenceEvidenceAuthenticationTelemetry::default();
    if bytes.is_empty() {
        return Err(invalid("numerical relation evidence must not be empty"));
    }
    let evidence = if bytes.starts_with(COMPRESSED_EVIDENCE_MAGIC) {
        let decode_started = Instant::now();
        let decoded = decode_compressed_evidence(bytes)?;
        telemetry.transport_decode_nanoseconds = elapsed_nanoseconds(decode_started);
        parse_numerical_relation_evidence_v3(&decoded, runtime_parameters, &mut telemetry)?
    } else {
        parse_numerical_relation_evidence_v3(bytes, runtime_parameters, &mut telemetry)?
    };
    telemetry.total_nanoseconds = elapsed_nanoseconds(total_started);
    Ok((evidence, telemetry))
}

fn parse_numerical_relation_evidence_v3(
    bytes: &[u8],
    runtime_parameters: &AuthenticatedRuntimeParameterContract,
    telemetry: &mut RecurrenceEvidenceAuthenticationTelemetry,
) -> RusticolResult<RecurrenceNumericalRelationEvidence> {
    const ABI: &str = "pyamplicol-recurrence-numerical-current-evidence-v3";
    const RELATION_SET_ABI: &str = "pyamplicol-authenticated-numerical-current-relation-set-v2";
    validate_raw_json_lexical_structure(bytes)?;
    let observation_ranges = locate_raw_observation_ranges(bytes)?;
    let metadata_bytes = materialize_raw_observation_metadata(bytes, observation_ranges)?;
    let located_observations = LocatedObservationArrays {
        metadata_bytes,
        candidate: observation_ranges.candidate,
        verification: observation_ranges.verification,
    };
    let value: JsonValue = serde_json::from_slice(&located_observations.metadata_bytes)
        .map_err(|error| invalid(format!("numerical relation evidence is not JSON: {error}")))?;
    validate_canonical_json_bytes(
        &value,
        &located_observations.metadata_bytes,
        "numerical relation evidence metadata",
    )?;
    let object = json_object(&value, "numerical relation evidence")?;
    require_json_fields(
        object,
        &[
            "abi",
            "requested_mode",
            "schedule_semantic_digest",
            "baseline_runtime_layout_digest",
            "source_semantics",
            "source_semantics_sha256",
            "runtime_parameter_schema",
            "runtime_parameter_schema_sha256",
            "candidate_capture",
            "verification_capture",
            "certificate_algorithm",
            "certificate_set_sha256",
            "precision_digits",
            "probe_count",
            "verification_probe_count",
            "relative_tolerance_binary64",
            "absolute_tolerance_binary64",
            "seed",
            "candidate_index",
            "numerical_candidate_count",
            "verification_rejected_count",
            "rejected_hypothesis_count",
            "tested_hypothesis_count",
            "decision_sha256",
            "rejection_decision_sha256",
            "certificates",
            "mappings",
        ],
        "numerical relation evidence",
    )?;
    require_json_string_value(object, "abi", ABI, "numerical relation evidence")?;
    let requested_mode = RecurrenceRelationDiscoveryMode::parse(json_string(
        object,
        "requested_mode",
        "numerical relation evidence mode",
    )?)?;
    if requested_mode == RecurrenceRelationDiscoveryMode::Off {
        return Err(invalid("off mode cannot carry numerical relation evidence"));
    }
    let schedule_semantic_digest = evidence_sha256(
        object,
        "schedule_semantic_digest",
        "numerical relation schedule digest",
    )?;
    let baseline_runtime_layout_digest = evidence_sha256(
        object,
        "baseline_runtime_layout_digest",
        "numerical relation baseline layout digest",
    )?;
    let source_semantics_sha256 = evidence_sha256(
        object,
        "source_semantics_sha256",
        "numerical relation source semantics",
    )?;
    let source = parse_raw_source_semantics(
        json_field(object, "source_semantics", "numerical relation evidence")?,
        &source_semantics_sha256,
        &schedule_semantic_digest,
        &baseline_runtime_layout_digest,
        runtime_parameters,
    )?;
    let runtime_parameter_schema_sha256 = evidence_sha256(
        object,
        "runtime_parameter_schema_sha256",
        "recurrence runtime parameter schema",
    )?;
    let runtime_parameter_count = validate_raw_runtime_parameter_schema(
        json_field(
            object,
            "runtime_parameter_schema",
            "numerical relation evidence",
        )?,
        &runtime_parameter_schema_sha256,
        runtime_parameters,
    )?;
    let certificate_algorithm = json_nonempty_string(
        object,
        "certificate_algorithm",
        "numerical relation certificate algorithm",
    )?
    .to_owned();
    if certificate_algorithm != rusticol_core::recurrence::NUMERICAL_RELATION_CERTIFICATE_ALGORITHM
    {
        return Err(invalid(
            "numerical relation evidence uses an unsupported certificate algorithm",
        ));
    }
    let precision_digits = json_u32(object, "precision_digits", "numerical relation precision")?;
    let probe_count = json_u32(
        object,
        "probe_count",
        "numerical relation candidate probe count",
    )?;
    let verification_probe_count = json_u32(
        object,
        "verification_probe_count",
        "numerical relation verification probe count",
    )?;
    raw_evidence_geometry_counts(
        &source,
        probe_count,
        verification_probe_count,
        runtime_parameter_count,
    )?;
    let relative_tolerance_hex = json_string(
        object,
        "relative_tolerance_binary64",
        "numerical relation relative tolerance encoding",
    )?;
    let absolute_tolerance_hex = json_string(
        object,
        "absolute_tolerance_binary64",
        "numerical relation absolute tolerance encoding",
    )?;
    let (relative_tolerance, relative_exact) = parse_nonnegative_python_f64_hex(
        relative_tolerance_hex,
        "numerical relation relative tolerance encoding",
    )?;
    let (absolute_tolerance, absolute_exact) = parse_nonnegative_python_f64_hex(
        absolute_tolerance_hex,
        "numerical relation absolute tolerance encoding",
    )?;
    if relative_tolerance == 0.0 && absolute_tolerance == 0.0 {
        return Err(RusticolError::integrity(
            "numerical relation tolerances cannot both be zero",
        ));
    }
    if json_string(
        object,
        "relative_tolerance_binary64",
        "numerical relation relative tolerance encoding",
    )? != python_f64_hex(relative_tolerance)
        || json_string(
            object,
            "absolute_tolerance_binary64",
            "numerical relation absolute tolerance encoding",
        )? != python_f64_hex(absolute_tolerance)
    {
        return Err(RusticolError::integrity(
            "numerical relation tolerance encoding disagrees with its binary64 value",
        ));
    }
    let seed = json_u64(object, "seed", "numerical relation seed")?;
    let capture_authentication_started = Instant::now();
    let candidate = parse_raw_numerical_capture(
        bytes,
        located_observations.candidate,
        json_field(object, "candidate_capture", "numerical relation evidence")?,
        "candidate",
        "candidate-current-probes-v1",
        precision_digits,
        probe_count,
        &source,
        &source_semantics_sha256,
        &runtime_parameter_schema_sha256,
        runtime_parameter_count,
        seed,
    )?;
    let verification = parse_raw_numerical_capture(
        bytes,
        located_observations.verification,
        json_field(
            object,
            "verification_capture",
            "numerical relation evidence",
        )?,
        "verification",
        "independent-verification-current-probes-v1",
        precision_digits,
        verification_probe_count,
        &source,
        &source_semantics_sha256,
        &runtime_parameter_schema_sha256,
        runtime_parameter_count,
        seed,
    )?;
    validate_independent_raw_captures(&candidate, &verification, runtime_parameters, seed)?;
    telemetry.capture_authentication_nanoseconds =
        elapsed_nanoseconds(capture_authentication_started);

    let derivation = derive_raw_numerical_relations(
        &source,
        &candidate,
        &verification,
        precision_digits,
        seed,
        &relative_exact,
        &absolute_exact,
        json_string(
            object,
            "relative_tolerance_binary64",
            "numerical relation relative tolerance encoding",
        )?,
        json_string(
            object,
            "absolute_tolerance_binary64",
            "numerical relation absolute tolerance encoding",
        )?,
        &runtime_parameter_schema_sha256,
        &certificate_algorithm,
        telemetry,
    )?;
    let evidence_finalization_started = Instant::now();
    let supplied_decision_sha256 = evidence_sha256(
        object,
        "decision_sha256",
        "numerical relation decision chain",
    )?;
    if supplied_decision_sha256 != derivation.decision_sha256 {
        return Err(RusticolError::integrity(format!(
            "numerical relation decision digest was not derived from every raw hypothesis: \
             supplied {supplied_decision_sha256}, derived {}",
            derivation.decision_sha256
        )));
    }
    let supplied_rejection_sha256 = evidence_sha256(
        object,
        "rejection_decision_sha256",
        "numerical relation rejection decision chain",
    )?;
    if supplied_rejection_sha256 != derivation.rejection_decision_sha256 {
        return Err(RusticolError::integrity(
            "numerical relation rejection digest was not derived from every rejected raw hypothesis",
        ));
    }
    for (field, expected) in [
        (
            "numerical_candidate_count",
            derivation.numerical_candidate_count,
        ),
        (
            "verification_rejected_count",
            derivation.verification_rejected_count,
        ),
        (
            "tested_hypothesis_count",
            derivation.tested_hypothesis_count,
        ),
        (
            "rejected_hypothesis_count",
            derivation.rejected_hypothesis_count,
        ),
    ] {
        if evidence_usize(object, field, "numerical relation evidence counter")? != expected {
            return Err(RusticolError::integrity(format!(
                "numerical relation evidence {field} was not derived from raw observations"
            )));
        }
    }
    validate_candidate_index_claim(
        json_field(object, "candidate_index", "numerical relation evidence")?,
        &derivation,
    )?;
    let supplied_certificates =
        json_array(object, "certificates", "numerical relation certificates")?;
    let supplied_mappings = json_array(object, "mappings", "numerical relation mappings")?;
    let derived_certificates = derivation
        .relations
        .iter()
        .map(|relation| relation.certificate.clone())
        .collect::<Vec<_>>();
    let derived_mappings = derivation
        .relations
        .iter()
        .map(|relation| relation.mapping_json.clone())
        .collect::<Vec<_>>();
    if supplied_certificates != derived_certificates || supplied_mappings != derived_mappings {
        return Err(RusticolError::integrity(
            "numerical relation certificates or mappings were not derived from raw observations",
        ));
    }
    let relation_set = serde_json::json!({
        "abi": RELATION_SET_ABI,
        "certificates": derived_certificates,
        "mappings": derived_mappings,
    });
    let actual_certificate_set_sha256 =
        canonical_json_sha256(&relation_set, "numerical relation certificate set")?;
    let certificate_set_sha256 = evidence_sha256(
        object,
        "certificate_set_sha256",
        "numerical relation certificate set",
    )?;
    if certificate_set_sha256 != actual_certificate_set_sha256 {
        return Err(RusticolError::integrity(
            "numerical relation certificate-set digest does not match native raw-evidence replay",
        ));
    }
    let evidence = RecurrenceNumericalRelationEvidence {
        requested_mode,
        schedule_semantic_digest,
        baseline_runtime_layout_digest,
        source_semantics_sha256,
        runtime_parameter_schema_sha256,
        candidate_observation_batch_sha256: candidate.observation_batch_sha256,
        verification_observation_batch_sha256: verification.observation_batch_sha256,
        decision_sha256: derivation.decision_sha256,
        rejection_decision_sha256: derivation.rejection_decision_sha256,
        certificate_algorithm,
        certificate_set_sha256,
        precision_digits,
        probe_count,
        verification_probe_count,
        relative_tolerance,
        absolute_tolerance,
        seed,
        numerical_candidate_count: derivation.numerical_candidate_count,
        verification_rejected_count: derivation.verification_rejected_count,
        rejected_hypothesis_count: derivation.rejected_hypothesis_count,
        tested_hypothesis_count: derivation.tested_hypothesis_count,
        mappings: derivation
            .relations
            .into_iter()
            .map(|relation| relation.mapping)
            .collect(),
    };
    telemetry.evidence_finalization_nanoseconds =
        elapsed_nanoseconds(evidence_finalization_started);
    Ok(evidence)
}

fn parse_raw_source_semantics(
    value: &JsonValue,
    claimed_sha256: &str,
    schedule_semantic_digest: &str,
    baseline_runtime_layout_digest: &str,
    authenticated_process: &AuthenticatedRuntimeParameterContract,
) -> RusticolResult<RawSourceSemantics> {
    const ABI: &str = "pyamplicol-recurrence-numerical-current-source-v3";
    let object = json_object(value, "recurrence numerical source semantics")?;
    require_json_fields(
        object,
        &[
            "abi",
            "process_id",
            "strategy",
            "schedule_semantic_digest",
            "baseline_runtime_layout_digest",
            "selector_schedule",
            "currents",
        ],
        "recurrence numerical source semantics",
    )?;
    require_json_string_value(object, "abi", ABI, "recurrence numerical source semantics")?;
    if canonical_json_sha256(value, "recurrence numerical source semantics")? != claimed_sha256 {
        return Err(RusticolError::integrity(
            "recurrence numerical source-semantics digest does not match its raw payload",
        ));
    }
    require_json_string_value(
        object,
        "schedule_semantic_digest",
        schedule_semantic_digest,
        "recurrence numerical source semantics",
    )?;
    require_json_string_value(
        object,
        "baseline_runtime_layout_digest",
        baseline_runtime_layout_digest,
        "recurrence numerical source semantics",
    )?;
    require_json_string_value(
        object,
        "process_id",
        &authenticated_process.process_id,
        "recurrence numerical source semantics",
    )?;
    if authenticated_process.physical_pdgs.is_empty() {
        return Err(invalid(
            "authenticated recurrence process has no physical external particles",
        ));
    }
    let strategy =
        json_nonempty_string(object, "strategy", "recurrence numerical source semantics")?
            .to_owned();
    if !matches!(
        strategy.as_str(),
        "topology-replay" | "all-flow-union" | "contracted-color-union"
    ) {
        return Err(invalid("raw recurrence source has an unsupported strategy"));
    }
    let current_values = json_array(object, "currents", "recurrence numerical source currents")?;
    let mut currents = Vec::with_capacity(current_values.len());
    for (index, current_value) in current_values.iter().enumerate() {
        let context = format!("recurrence numerical source current {index}");
        let current = json_object(current_value, &context)?;
        require_json_fields(current, &["current_id", "is_source", "contract"], &context)?;
        let current_id = json_u32(current, "current_id", &context)?;
        if current_id as usize != index {
            return Err(invalid(format!(
                "{context} has non-canonical current ID {current_id}"
            )));
        }
        let is_source = json_field(current, "is_source", &context)?
            .as_bool()
            .ok_or_else(|| invalid(format!("{context} source flag is not boolean")))?;
        let contract = json_field(current, "contract", &context)?.clone();
        let contract_values = contract
            .as_array()
            .ok_or_else(|| invalid(format!("{context} contract is not an array")))?;
        if contract_values.len() != 9 {
            return Err(invalid(format!("{context} contract width is not nine")));
        }
        let dimension_u64 = contract_values[3]
            .as_u64()
            .ok_or_else(|| invalid(format!("{context} dimension is not unsigned")))?;
        let dimension = u16::try_from(dimension_u64)
            .map_err(|_| invalid(format!("{context} dimension exceeds u16")))?;
        if dimension == 0 {
            return Err(invalid(format!("{context} dimension is zero")));
        }
        let selector_domain_id = contract_values[5]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| invalid(format!("{context} selector domain is not u32")))?;
        currents.push(RawSourceCurrent {
            current_id,
            is_source,
            contract_key: canonical_json_bytes(&contract, &context)?,
            dimension,
            selector_domain_id,
        });
    }
    Ok(RawSourceSemantics {
        value: value.clone(),
        process_id: authenticated_process.process_id.clone(),
        physical_pdgs: authenticated_process.physical_pdgs.clone(),
        strategy,
        selector_schedule: json_field(
            object,
            "selector_schedule",
            "recurrence numerical source semantics",
        )?
        .clone(),
        currents,
    })
}

fn validate_candidate_index_claim(
    value: &JsonValue,
    derivation: &RawNumericalDerivation,
) -> RusticolResult<()> {
    const ALGORITHM: &str = "complete-contract-anchor-tolerance-window-v1";
    let object = json_object(value, "raw numerical candidate index")?;
    require_json_fields(
        object,
        &[
            "algorithm",
            "completeness",
            "contract_count",
            "exhaustive_fallback_contract_count",
            "theoretical_pair_hypothesis_count",
            "screened_pair_hypothesis_count",
            "zero_hypothesis_count",
            "nearest_rejected_scope",
        ],
        "raw numerical candidate index",
    )?;
    require_json_string_value(
        object,
        "algorithm",
        ALGORITHM,
        "raw numerical candidate index",
    )?;
    require_json_string_value(
        object,
        "completeness",
        "complete-within-configured-tolerance",
        "raw numerical candidate index",
    )?;
    require_json_string_value(
        object,
        "nearest_rejected_scope",
        "zero-and-tolerance-window-screened-hypotheses",
        "raw numerical candidate index",
    )?;
    for (field, expected) in [
        ("contract_count", derivation.candidate_index_contract_count),
        (
            "exhaustive_fallback_contract_count",
            derivation.exhaustive_fallback_contract_count,
        ),
        (
            "theoretical_pair_hypothesis_count",
            derivation.theoretical_pair_hypothesis_count,
        ),
        (
            "screened_pair_hypothesis_count",
            derivation.screened_pair_hypothesis_count,
        ),
        ("zero_hypothesis_count", derivation.zero_hypothesis_count),
    ] {
        if evidence_usize(object, field, "raw numerical candidate index")? != expected {
            return Err(RusticolError::integrity(format!(
                "raw numerical candidate index {field} was not independently derived"
            )));
        }
    }
    Ok(())
}

fn raw_evidence_geometry_counts(
    source: &RawSourceSemantics,
    probe_count: u32,
    verification_probe_count: u32,
    runtime_parameter_count: usize,
) -> RusticolResult<(usize, usize, usize, usize)> {
    let current_count = source.currents.len();
    let row_count = source
        .currents
        .len()
        .checked_mul(2)
        .and_then(|count| count.checked_add(runtime_parameter_count))
        .ok_or_else(|| invalid("raw numerical row count overflows usize"))?;
    let component_count = source.currents.iter().try_fold(0_usize, |total, current| {
        total
            .checked_add(usize::from(current.dimension))
            .ok_or_else(|| invalid("raw numerical component count overflows usize"))
    })?;
    let point_count = usize::try_from(probe_count)
        .ok()
        .and_then(|candidate| {
            usize::try_from(verification_probe_count)
                .ok()
                .and_then(|verification| candidate.checked_add(verification))
        })
        .ok_or_else(|| invalid("raw numerical probe count overflows usize"))?;
    let observation_scalar_count = component_count
        .checked_mul(point_count)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| invalid("raw numerical scalar count overflows usize"))?;
    let parameter_scalar_count = runtime_parameter_count
        .checked_mul(point_count)
        .ok_or_else(|| invalid("raw parameter scalar count overflows usize"))?;
    let scalar_count = observation_scalar_count
        .checked_add(parameter_scalar_count)
        .ok_or_else(|| invalid("raw total scalar count overflows usize"))?;
    Ok((current_count, component_count, scalar_count, row_count))
}

fn validate_raw_runtime_parameter_schema(
    value: &JsonValue,
    claimed_sha256: &str,
    expected: &AuthenticatedRuntimeParameterContract,
) -> RusticolResult<usize> {
    if value != &expected.schema || claimed_sha256 != expected.schema_sha256 {
        return Err(RusticolError::integrity(
            "recurrence runtime parameter schema/defaults do not match the authenticated native template input",
        ));
    }
    Ok(expected.slots.len())
}

fn canonical_json_sha256_with_raw_arrays(
    value: &JsonValue,
    raw_arrays: &[(&str, &[u8])],
    context: &str,
) -> RusticolResult<String> {
    let canonical = canonical_json_bytes(value, context)?;
    let mut replacements = Vec::with_capacity(raw_arrays.len());
    let mut names = BTreeSet::new();
    for (placeholder, raw_array) in raw_arrays {
        if !names.insert(*placeholder) {
            return Err(invalid(format!(
                "{context} raw-array placeholder is duplicated"
            )));
        }
        let encoded_placeholder = canonical_json_bytes(
            &serde_json::json!([placeholder]),
            &format!("{context} raw-array placeholder"),
        )?;
        let mut matches = canonical
            .windows(encoded_placeholder.len())
            .enumerate()
            .filter_map(|(index, window)| (window == encoded_placeholder).then_some(index));
        let start = matches
            .next()
            .ok_or_else(|| invalid(format!("{context} has no raw-array placeholder")))?;
        if matches.next().is_some() {
            return Err(invalid(format!(
                "{context} raw-array placeholder is ambiguous"
            )));
        }
        let end = start
            .checked_add(encoded_placeholder.len())
            .ok_or_else(|| invalid(format!("{context} placeholder range overflows usize")))?;
        replacements.push((start, end, *raw_array));
    }
    replacements.sort_by_key(|replacement| replacement.0);
    let mut digest = Sha256::new();
    let mut cursor = 0;
    for (start, end, raw_array) in replacements {
        if start < cursor {
            return Err(invalid(format!("{context} raw-array placeholders overlap")));
        }
        digest.update(&canonical[cursor..start]);
        digest.update(raw_array);
        cursor = end;
    }
    digest.update(&canonical[cursor..]);
    Ok(hex_digest(digest.finalize()))
}

#[allow(clippy::too_many_arguments)]
fn observation_batch_sha256_from_raw(
    raw_bytes: &[u8],
    observation_array: RawByteRange,
    source_semantics_sha256: &str,
    runtime_parameter_schema_sha256: &str,
    point_sha256s: &[String],
    selector_context_sha256s: &[String],
    parameter_context_sha256s: &[String],
    batch_abi: &str,
    context: &str,
) -> RusticolResult<String> {
    const PLACEHOLDER: &str = "__pyamplicol_authenticated_raw_array_placeholder__";
    let range = observation_array.as_usize(raw_bytes.len(), context)?;
    let raw_array = raw_bytes
        .get(range)
        .ok_or_else(|| invalid(format!("{context} raw observation array is absent")))?;
    canonical_json_sha256_with_raw_arrays(
        &serde_json::json!({
            "abi": batch_abi,
            "source_semantics_sha256": source_semantics_sha256,
            "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
            "point_sha256s": point_sha256s,
            "selector_context_sha256s": selector_context_sha256s,
            "parameter_context_sha256s": parameter_context_sha256s,
            "currents": [PLACEHOLDER],
        }),
        &[(PLACEHOLDER, raw_array)],
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn parse_raw_numerical_capture<'a>(
    raw_bytes: &'a [u8],
    observation_array: RawByteRange,
    value: &JsonValue,
    label: &str,
    domain: &str,
    precision_digits: u32,
    expected_point_count: u32,
    source: &RawSourceSemantics,
    source_semantics_sha256: &str,
    runtime_parameter_schema_sha256: &str,
    runtime_parameter_count: usize,
    seed: u64,
) -> RusticolResult<RawNumericalCapture<'a>> {
    const ABI: &str = "pyamplicol-recurrence-current-observation-capture-v2";
    const BATCH_ABI: &str = "pyamplicol-recurrence-current-observation-batch-v2";
    const PARAMETER_CONTEXT_ABI: &str = "pyamplicol-recurrence-parameter-context-v1";
    let context = format!("{label} raw numerical capture");
    let object = json_object(value, &context)?;
    require_json_fields(
        object,
        &[
            "abi",
            "precision_digits",
            "point_count",
            "point_sha256s",
            "kinematic_sha256s",
            "parameter_contexts",
            "parameter_context_sha256s",
            "context_sha256s",
            "points",
            "current_count",
            "runtime_parameter_schema_sha256",
            "source_semantics_sha256",
            "observation_batch_sha256",
            "capture_contract_sha256",
            "context_policy",
            "complete_current_components",
            "point_major",
            "current_dimensions_sha256",
            "evaluator",
            "kinematic_binary64",
            "selector_contexts",
            "current_dimensions",
            "observations",
        ],
        &context,
    )?;
    require_json_string_value(object, "abi", ABI, &context)?;
    if json_u32(object, "precision_digits", &context)? != precision_digits
        || json_u32(object, "point_count", &context)? != expected_point_count
        || expected_point_count < 2
    {
        return Err(invalid(format!("{context} probe geometry is inconsistent")));
    }
    require_json_string_value(
        object,
        "runtime_parameter_schema_sha256",
        runtime_parameter_schema_sha256,
        &context,
    )?;
    require_json_string_value(
        object,
        "source_semantics_sha256",
        source_semantics_sha256,
        &context,
    )?;
    require_json_string_value(
        object,
        "evaluator",
        "recurrence-direct-plan-decimal-symbolica-exact",
        &context,
    )?;
    if !json_bool(object, "complete_current_components", &context)?
        || !json_bool(object, "point_major", &context)?
    {
        return Err(invalid(format!(
            "{context} does not contain complete point-major current components"
        )));
    }
    let expected_policy = match source.strategy.as_str() {
        "topology-replay" => "seeded-replay-target-per-physical-point-v1",
        "all-flow-union" => "seeded-helicity-per-selector-domain-and-physical-point-v1",
        "contracted-color-union" => "fixed-contracted-source-schedule-v1",
        _ => return Err(invalid("raw recurrence source strategy is unsupported")),
    };
    require_json_string_value(object, "context_policy", expected_policy, &context)?;

    let point_values = json_array(object, "points", &context)?;
    let point_sha256s = parse_sha256_array(object, "point_sha256s", &context)?;
    let kinematic_values = json_array(object, "kinematic_binary64", &context)?;
    let kinematic_sha256s = parse_sha256_array(object, "kinematic_sha256s", &context)?;
    let selector_contexts = json_array(object, "selector_contexts", &context)?;
    let selector_context_sha256s = parse_sha256_array(object, "context_sha256s", &context)?;
    let parameter_values = json_array(object, "parameter_contexts", &context)?;
    let parameter_context_sha256s =
        parse_sha256_array(object, "parameter_context_sha256s", &context)?;
    let point_count = expected_point_count as usize;
    if [
        point_values.len(),
        point_sha256s.len(),
        kinematic_values.len(),
        kinematic_sha256s.len(),
        selector_contexts.len(),
        selector_context_sha256s.len(),
        parameter_values.len(),
        parameter_context_sha256s.len(),
    ]
    .into_iter()
    .any(|count| count != point_count)
    {
        return Err(invalid(format!(
            "{context} does not cover every ordered probe slot"
        )));
    }
    if point_sha256s.iter().collect::<BTreeSet<_>>().len() != point_count
        || kinematic_sha256s.iter().collect::<BTreeSet<_>>().len() != point_count
    {
        return Err(invalid(format!(
            "{context} physical probes are not distinct"
        )));
    }

    let mut parameter_contexts = Vec::with_capacity(point_count);
    for point_index in 0..point_count {
        let point_context = format!("{context} point {point_index}");
        let actual_point_sha256 = canonical_json_sha256(
            &point_values[point_index],
            &format!("{point_context} record"),
        )?;
        if point_sha256s[point_index] != actual_point_sha256 {
            return Err(RusticolError::integrity(format!(
                "{point_context} digest does not match its raw point"
            )));
        }
        validate_raw_kinematic(
            &point_values[point_index],
            &kinematic_values[point_index],
            source,
            domain_seed(seed, domain, point_index),
            &point_context,
        )?;
        let actual_kinematic_sha256 = canonical_json_sha256(
            &kinematic_values[point_index],
            &format!("{point_context} binary64 kinematics"),
        )?;
        if kinematic_sha256s[point_index] != actual_kinematic_sha256 {
            return Err(RusticolError::integrity(format!(
                "{point_context} kinematic digest does not match its raw binary64 values"
            )));
        }
        let expected_selector = expected_selector_context(source, seed, domain, point_index)?;
        if selector_contexts[point_index] != expected_selector
            || selector_context_sha256s[point_index]
                != canonical_json_sha256(&expected_selector, &format!("{point_context} selector"))?
        {
            return Err(RusticolError::integrity(format!(
                "{point_context} selector context is stale or non-deterministic"
            )));
        }
        let raw_parameters = parameter_values[point_index]
            .as_array()
            .ok_or_else(|| invalid(format!("{point_context} parameters are not an array")))?;
        if raw_parameters.len() != runtime_parameter_count {
            return Err(invalid(format!(
                "{point_context} parameters do not cover every runtime slot"
            )));
        }
        let parsed_parameters = raw_parameters
            .iter()
            .enumerate()
            .map(|(slot, value)| {
                let text = value.as_str().ok_or_else(|| {
                    invalid(format!("{point_context} parameter {slot} is not a string"))
                })?;
                parse_canonical_decimal(text, &format!("{point_context} parameter {slot}"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let parameter_payload = serde_json::json!({
            "abi": PARAMETER_CONTEXT_ABI,
            "domain": domain,
            "point_index": point_index,
            "point_sha256": point_sha256s[point_index],
            "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
            "values": raw_parameters,
        });
        if parameter_context_sha256s[point_index]
            != canonical_json_sha256(
                &parameter_payload,
                &format!("{point_context} parameter context"),
            )?
        {
            return Err(RusticolError::integrity(format!(
                "{point_context} parameter-context digest does not match its raw values"
            )));
        }
        parameter_contexts.push(parsed_parameters);
    }

    let dimension_values = json_array(object, "current_dimensions", &context)?;
    if dimension_values.len() != source.currents.len()
        || json_u32(object, "current_count", &context)? as usize != source.currents.len()
    {
        return Err(invalid(format!(
            "{context} current census does not match source semantics"
        )));
    }
    let observation_rows =
        scan_raw_observation_array(raw_bytes, observation_array, source, point_count, label)?;
    let mut dimensions = Vec::with_capacity(source.currents.len());
    for (index, source_current) in source.currents.iter().enumerate() {
        let row_context = format!("{context} current {index}");
        let dimension_row = json_object(&dimension_values[index], &row_context)?;
        require_json_fields(dimension_row, &["current_id", "dimension"], &row_context)?;
        let current_id = json_u32(dimension_row, "current_id", &row_context)?;
        let dimension = json_u32(dimension_row, "dimension", &row_context)?;
        if current_id != source_current.current_id
            || dimension != u32::from(source_current.dimension)
        {
            return Err(invalid(format!(
                "{row_context} dimension or ID disagrees with source semantics"
            )));
        }
        dimensions.push(source_current.dimension);
    }
    let dimensions_object = dimensions
        .iter()
        .enumerate()
        .map(|(current_id, dimension)| {
            (
                current_id.to_string(),
                JsonValue::from(u64::from(*dimension)),
            )
        })
        .collect::<JsonMap<_, _>>();
    if evidence_sha256(object, "current_dimensions_sha256", &context)?
        != canonical_json_sha256(
            &JsonValue::Object(dimensions_object),
            &format!("{context} dimensions"),
        )?
    {
        return Err(RusticolError::integrity(format!(
            "{context} dimension digest does not match the raw census"
        )));
    }
    let observation_batch_sha256 = evidence_sha256(object, "observation_batch_sha256", &context)?;
    if observation_batch_sha256
        != observation_batch_sha256_from_raw(
            raw_bytes,
            observation_array,
            source_semantics_sha256,
            runtime_parameter_schema_sha256,
            &point_sha256s,
            &selector_context_sha256s,
            &parameter_context_sha256s,
            BATCH_ABI,
            &format!("{context} observation batch"),
        )?
    {
        return Err(RusticolError::integrity(format!(
            "{context} observation-batch digest does not match raw observations"
        )));
    }
    let capture_contract_sha256 = evidence_sha256(object, "capture_contract_sha256", &context)?;
    let capture_payload = serde_json::json!({
        "abi": ABI,
        "precision_digits": precision_digits,
        "source_semantics_sha256": source_semantics_sha256,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "point_sha256s": point_sha256s,
        "kinematic_sha256s": kinematic_sha256s,
        "selector_context_sha256s": selector_context_sha256s,
        "parameter_context_sha256s": parameter_context_sha256s,
        "current_dimensions": dimension_values,
        "observation_batch_sha256": observation_batch_sha256,
        "context_policy": expected_policy,
    });
    if capture_contract_sha256
        != canonical_json_sha256(&capture_payload, &format!("{context} contract"))?
    {
        return Err(RusticolError::integrity(format!(
            "{context} contract digest does not match raw capture"
        )));
    }
    Ok(RawNumericalCapture {
        raw_bytes,
        observation_rows,
        point_count,
        point_sha256s,
        kinematic_sha256s,
        parameter_context_sha256s,
        parameter_contexts,
        dimensions,
        observation_batch_sha256,
        capture_contract_sha256,
    })
}

fn parse_sha256_array(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<Vec<String>> {
    json_array(object, field, context)?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let text = value
                .as_str()
                .ok_or_else(|| invalid(format!("{context} {field}[{index}] is not a string")))?;
            semantic_digest_from_hex(text, &format!("{context} {field}[{index}]"))?;
            Ok(text.to_owned())
        })
        .collect()
}

fn validate_raw_kinematic(
    point: &JsonValue,
    kinematic: &JsonValue,
    source: &RawSourceSemantics,
    expected_seed: u64,
    context: &str,
) -> RusticolResult<()> {
    let point_object = json_object(point, &format!("{context} point record"))?;
    require_json_fields(
        point_object,
        &[
            "schema_version",
            "kind",
            "process_id",
            "process",
            "seed",
            "available",
            "error",
            "points",
        ],
        &format!("{context} point record"),
    )?;
    if json_u32(point_object, "schema_version", context)? != 1
        || json_string(point_object, "kind", context)? != "pyamplicol-rusticol-validation-momenta"
        || !json_bool(point_object, "available", context)?
        || !json_field(point_object, "error", context)?.is_null()
    {
        return Err(invalid(format!(
            "{context} validation point is unavailable"
        )));
    }
    require_json_string_value(point_object, "process_id", &source.process_id, context)?;
    json_nonempty_string(point_object, "process", context)?;
    if json_u64(point_object, "seed", context)? != expected_seed {
        return Err(RusticolError::integrity(format!(
            "{context} validation-point seed is stale"
        )));
    }
    let point_sets = json_array(point_object, "points", context)?;
    if point_sets.len() != 1 {
        return Err(invalid(format!(
            "{context} validation point must contain one momentum set"
        )));
    }
    let particles = point_sets[0]
        .as_array()
        .ok_or_else(|| invalid(format!("{context} particle set is not an array")))?;
    let kinematic_rows = kinematic
        .as_array()
        .ok_or_else(|| invalid(format!("{context} kinematics are not an array")))?;
    if particles.len() != kinematic_rows.len()
        || particles.len() != source.physical_pdgs.len()
        || particles.is_empty()
    {
        return Err(invalid(format!(
            "{context} point, process, and binary64 kinematic widths disagree"
        )));
    }
    for (particle_index, ((particle, raw_row), expected_pdg)) in particles
        .iter()
        .zip(kinematic_rows)
        .zip(&source.physical_pdgs)
        .enumerate()
    {
        let particle_context = format!("{context} particle {particle_index}");
        let particle = json_object(particle, &particle_context)?;
        require_json_fields(particle, &["pdg", "momentum"], &particle_context)?;
        let actual_pdg = particle
            .get("pdg")
            .and_then(JsonValue::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| invalid(format!("{particle_context} PDG is not an i32")))?;
        if actual_pdg != *expected_pdg {
            return Err(RusticolError::integrity(format!(
                "{particle_context} PDG does not match the authenticated recurrence process"
            )));
        }
        let momentum = json_array(particle, "momentum", &particle_context)?;
        let binary = raw_row
            .as_array()
            .filter(|row| row.len() == 4)
            .ok_or_else(|| {
                invalid(format!(
                    "{particle_context} binary64 momentum is not four-wide"
                ))
            })?;
        if momentum.len() != 4 {
            return Err(invalid(format!(
                "{particle_context} decimal momentum is not four-wide"
            )));
        }
        for component in 0..4 {
            let decimal = momentum[component].as_str().ok_or_else(|| {
                invalid(format!(
                    "{particle_context} momentum component {component} is not a string"
                ))
            })?;
            parse_canonical_decimal(
                decimal,
                &format!("{particle_context} momentum component {component}"),
            )?;
            let binary_value = decimal.parse::<f64>().map_err(|_| {
                invalid(format!(
                    "{particle_context} momentum component {component} is not binary64"
                ))
            })?;
            if !binary_value.is_finite()
                || binary[component].as_str() != Some(python_f64_hex_signed(binary_value).as_str())
            {
                return Err(RusticolError::integrity(format!(
                    "{particle_context} decimal and binary64 momentum disagree at component {component}"
                )));
            }
        }
    }
    Ok(())
}

fn all_flow_selector_domain_helicities(
    source: &RawSourceSemantics,
) -> RusticolResult<BTreeMap<u32, Vec<(usize, u32)>>> {
    let schedule = json_object(
        &source.selector_schedule,
        "recurrence numerical selector schedule",
    )?;
    require_json_fields(
        schedule,
        &["policy", "resolved_helicities"],
        "recurrence union selector schedule",
    )?;
    require_json_string_value(
        schedule,
        "policy",
        "seeded-helicity-per-selector-domain-and-physical-point-v1",
        "recurrence union selector schedule",
    )?;
    let helicities = json_array(
        schedule,
        "resolved_helicities",
        "recurrence union selector schedule",
    )?;
    let mut by_domain = BTreeMap::<u32, Vec<(usize, u32)>>::new();
    for (index, value) in helicities.iter().enumerate() {
        let helicity = json_object(value, "recurrence resolved helicity")?;
        require_json_fields(
            helicity,
            &["helicity_index", "helicity_id", "selector_domain_id"],
            "recurrence resolved helicity",
        )?;
        if json_u32(helicity, "helicity_index", "recurrence resolved helicity")? as usize != index {
            return Err(invalid(
                "recurrence resolved-helicity index is not canonical",
            ));
        }
        by_domain
            .entry(json_u32(
                helicity,
                "selector_domain_id",
                "recurrence resolved helicity",
            )?)
            .or_default()
            .push((
                index,
                json_u32(helicity, "helicity_id", "recurrence resolved helicity")?,
            ));
    }
    if by_domain.is_empty() {
        return Err(invalid("recurrence union selector schedule is empty"));
    }
    Ok(by_domain)
}

fn expected_selector_context(
    source: &RawSourceSemantics,
    seed: u64,
    domain: &str,
    point_index: usize,
) -> RusticolResult<JsonValue> {
    let selector_seed = domain_seed(seed, domain, point_index);
    match source.strategy.as_str() {
        "topology-replay" => {
            let schedule = json_object(
                &source.selector_schedule,
                "recurrence numerical selector schedule",
            )?;
            require_json_fields(
                schedule,
                &["policy", "replay_targets"],
                "recurrence topology selector schedule",
            )?;
            require_json_string_value(
                schedule,
                "policy",
                "seeded-replay-target-per-physical-point-v1",
                "recurrence topology selector schedule",
            )?;
            let targets = json_array(
                schedule,
                "replay_targets",
                "recurrence topology selector schedule",
            )?;
            if targets.is_empty() {
                return Err(invalid("recurrence topology selector schedule is empty"));
            }
            let target_index = selector_seed as usize % targets.len();
            let target = json_object(&targets[target_index], "recurrence replay target")?;
            require_json_fields(
                target,
                &[
                    "target_index",
                    "public_flow_id",
                    "representative_id",
                    "selector_domain_id",
                ],
                "recurrence replay target",
            )?;
            if json_u32(target, "target_index", "recurrence replay target")? as usize
                != target_index
            {
                return Err(invalid("recurrence replay target index is not canonical"));
            }
            Ok(serde_json::json!({
                "strategy": source.strategy,
                "target_index": target_index,
                "public_flow_id": json_u32(target, "public_flow_id", "recurrence replay target")?,
                "representative_id": json_u32(target, "representative_id", "recurrence replay target")?,
                "selector_domain_id": json_u32(target, "selector_domain_id", "recurrence replay target")?,
            }))
        }
        "all-flow-union" => {
            let selected = all_flow_selector_domain_helicities(source)?
                .into_iter()
                .map(|(selector_domain_id, rows)| {
                    let (helicity_index, helicity_id) = rows[selector_seed as usize % rows.len()];
                    serde_json::json!({
                        "selector_domain_id": selector_domain_id,
                        "helicity_index": helicity_index,
                        "helicity_id": helicity_id,
                    })
                })
                .collect::<Vec<_>>();
            Ok(serde_json::json!({
                "strategy": source.strategy,
                "selector_domain_helicities": selected,
            }))
        }
        "contracted-color-union" => {
            let schedule = json_object(
                &source.selector_schedule,
                "recurrence numerical selector schedule",
            )?;
            require_json_fields(
                schedule,
                &["policy", "fixed_source_schedule"],
                "recurrence contracted selector schedule",
            )?;
            require_json_string_value(
                schedule,
                "policy",
                "fixed-contracted-source-schedule-v1",
                "recurrence contracted selector schedule",
            )?;
            if !json_bool(
                schedule,
                "fixed_source_schedule",
                "recurrence contracted selector schedule",
            )? {
                return Err(invalid(
                    "recurrence contracted source schedule is not fixed",
                ));
            }
            Ok(serde_json::json!({
                "strategy": source.strategy,
                "fixed_source_schedule": true,
            }))
        }
        _ => Err(invalid("recurrence selector strategy is unsupported")),
    }
}

fn validate_independent_raw_captures(
    candidate: &RawNumericalCapture<'_>,
    verification: &RawNumericalCapture<'_>,
    runtime_parameters: &AuthenticatedRuntimeParameterContract,
    seed: u64,
) -> RusticolResult<()> {
    if candidate.dimensions != verification.dimensions
        || candidate.observation_rows.len() != verification.observation_rows.len()
        || !sets_are_disjoint(&candidate.point_sha256s, &verification.point_sha256s)
        || !sets_are_disjoint(
            &candidate.kinematic_sha256s,
            &verification.kinematic_sha256s,
        )
        || !sets_are_disjoint(
            &candidate.parameter_context_sha256s,
            &verification.parameter_context_sha256s,
        )
    {
        return Err(RusticolError::integrity(
            "candidate and verification raw captures are not independent",
        ));
    }
    let expected_candidate = deterministic_parameter_contexts(
        &runtime_parameters.slots,
        candidate.parameter_contexts.len(),
        seed,
        "candidate-current-parameter-probes-v1",
        true,
    )?;
    let expected_verification = deterministic_parameter_contexts(
        &runtime_parameters.slots,
        verification.parameter_contexts.len(),
        seed,
        "independent-verification-current-parameter-probes-v1",
        false,
    )?;
    if candidate.parameter_contexts != expected_candidate
        || verification.parameter_contexts != expected_verification
    {
        return Err(RusticolError::integrity(
            "raw recurrence parameter contexts are stale or non-deterministic",
        ));
    }
    Ok(())
}

fn sets_are_disjoint(left: &[String], right: &[String]) -> bool {
    let left = left.iter().collect::<BTreeSet<_>>();
    right.iter().all(|value| !left.contains(value))
}

fn deterministic_parameter_contexts(
    slots: &[AuthenticatedRuntimeParameterSlot],
    count: usize,
    seed: u64,
    domain: &str,
    include_defaults: bool,
) -> RusticolResult<Vec<Vec<BigRational>>> {
    (0..count)
        .map(|probe_index| {
            if include_defaults && probe_index == 0 {
                return Ok(slots.iter().map(|slot| slot.default.clone()).collect());
            }
            slots
                .iter()
                .enumerate()
                .map(|(parameter_index, slot)| {
                    if slot.probe_policy == "derived-overwritten-fixed-zero-v1" {
                        return Ok(slot.default.clone());
                    }
                    let digest = Sha256::digest(
                        format!("{seed}:{domain}:{probe_index}:{parameter_index}").as_bytes(),
                    );
                    let unsigned = u64::from_be_bytes(
                        digest[..8]
                            .try_into()
                            .map_err(|_| invalid("parameter digest width is not eight"))?,
                    );
                    let signed = i128::from(unsigned) - (1_i128 << 63);
                    let signed = if signed == 0 { 1 } else { signed };
                    let scale = slot.default.abs().max(BigRational::one());
                    Ok(&slot.default
                        + scale * BigRational::new(BigInt::from(signed), BigInt::one() << 67_usize))
                })
                .collect()
        })
        .collect()
}

fn domain_seed(seed: u64, domain: &str, index: usize) -> u64 {
    let digest = Sha256::digest(format!("{seed}:{domain}:{index}").as_bytes());
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

fn validate_canonical_decimal_text<'a>(
    value: &'a str,
    context: &str,
) -> RusticolResult<(&'a str, &'a str, bool)> {
    if value.is_empty() || value.starts_with('+') || value == "-0" || value.contains(['e', 'E']) {
        return Err(invalid(format!(
            "{context} is not a canonical finite decimal"
        )));
    }
    let (negative, unsigned) = value
        .strip_prefix('-')
        .map_or((false, value), |rest| (true, rest));
    let (integer, fraction) = unsigned
        .split_once('.')
        .map_or((unsigned, None), |(integer, fraction)| {
            (integer, Some(fraction))
        });
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || integer.len() > 1 && integer.starts_with('0')
        || fraction.is_some_and(|fraction| {
            fraction.is_empty()
                || !fraction.bytes().all(|byte| byte.is_ascii_digit())
                || fraction.ends_with('0')
        })
    {
        return Err(invalid(format!(
            "{context} is not a canonical finite decimal"
        )));
    }
    let fraction = fraction.unwrap_or("");
    if negative && integer == "0" && fraction.bytes().all(|byte| byte == b'0') {
        return Err(invalid(format!("{context} is negative zero")));
    }
    Ok((integer, fraction, negative))
}

fn parse_canonical_decimal(value: &str, context: &str) -> RusticolResult<BigRational> {
    let (integer, fraction, negative) = validate_canonical_decimal_text(value, context)?;
    let digits = format!("{integer}{fraction}");
    let mut numerator = BigInt::from_str(&digits)
        .map_err(|_| invalid(format!("{context} decimal numerator is invalid")))?;
    if negative {
        numerator = -numerator;
    }
    let denominator = BigInt::from(10_u8).pow(
        u32::try_from(fraction.len())
            .map_err(|_| invalid(format!("{context} decimal scale exceeds u32")))?,
    );
    Ok(BigRational::new(numerator, denominator))
}

fn exact_binary64_rational(value: f64) -> RusticolResult<BigRational> {
    if !value.is_finite() {
        return Err(invalid("binary64 value is not finite"));
    }
    if value == 0.0 {
        return Ok(BigRational::zero());
    }
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
    };
    let mut numerator = BigInt::from(significand);
    let mut denominator = BigInt::one();
    if exponent >= 0 {
        numerator <<= exponent as usize;
    } else {
        denominator <<= (-exponent) as usize;
    }
    if negative {
        numerator = -numerator;
    }
    Ok(BigRational::new(numerator, denominator))
}

fn parse_nonnegative_python_f64_hex(
    text: &str,
    context: &str,
) -> RusticolResult<(f64, BigRational)> {
    if text.starts_with('-') || text.len() > 24 {
        return Err(invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        )));
    }
    let (mantissa, exponent_text) = text.split_once('p').ok_or_else(|| {
        invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        ))
    })?;
    let (leading, fraction_text) = mantissa
        .strip_prefix("0x")
        .and_then(|value| value.split_once('.'))
        .ok_or_else(|| {
            invalid(format!(
                "{context} is not a canonical nonnegative finite binary64 encoding"
            ))
        })?;
    if fraction_text.is_empty()
        || fraction_text.len() > 13
        || !fraction_text.bytes().all(|byte| byte.is_ascii_hexdigit())
        || !matches!(leading, "0" | "1")
    {
        return Err(invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        )));
    }
    let exponent = i32::from_str(exponent_text).map_err(|_| {
        invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        ))
    })?;
    let parsed_fraction = u64::from_str_radix(fraction_text, 16).map_err(|_| {
        invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        ))
    })?;
    let padding = 4 * (13 - fraction_text.len());
    let fraction = parsed_fraction
        .checked_shl(
            u32::try_from(padding)
                .map_err(|_| invalid(format!("{context} fraction padding is invalid")))?,
        )
        .ok_or_else(|| invalid(format!("{context} fraction is too wide")))?;
    let bits = match leading {
        "1" if (-1022..=1023).contains(&exponent) => {
            let exponent_bits = u64::try_from(exponent + 1023)
                .map_err(|_| invalid(format!("{context} exponent is invalid")))?;
            (exponent_bits << 52) | fraction
        }
        "0" if exponent == -1022 && fraction != 0 => fraction,
        "0" if exponent == 0 && fraction == 0 => 0,
        _ => {
            return Err(invalid(format!(
                "{context} is not a canonical nonnegative finite binary64 encoding"
            )));
        }
    };
    let value = f64::from_bits(bits);
    if !value.is_finite() || python_f64_hex(value) != text {
        return Err(invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        )));
    }
    let exact = exact_binary64_rational(value)?;
    Ok((value, exact))
}

fn python_f64_hex_signed(value: f64) -> String {
    let sign = if value.is_sign_negative() { "-" } else { "" };
    format!("{sign}{}", python_f64_hex(value.abs()))
}

fn raw_observation_row<'a>(
    capture: &'a RawNumericalCapture<'_>,
    current_id: u32,
    context: &str,
) -> RusticolResult<&'a RawObservationRow> {
    capture
        .observation_rows
        .get(current_id as usize)
        .ok_or_else(|| invalid(format!("{context} row is absent")))
}

fn visit_raw_observation_text<'a, F>(
    capture: &RawNumericalCapture<'a>,
    current_id: u32,
    context: &str,
    mut visit: F,
) -> RusticolResult<()>
where
    F: FnMut(usize, &'a str, &'a str) -> RusticolResult<()>,
{
    let row = raw_observation_row(capture, current_id, context)?;
    let values_range = row.values.as_usize(capture.raw_bytes.len(), context)?;
    let values = capture
        .raw_bytes
        .get(values_range.clone())
        .ok_or_else(|| invalid(format!("{context} values are absent")))?;
    let dimension = capture
        .dimensions
        .get(current_id as usize)
        .copied()
        .ok_or_else(|| invalid(format!("{context} dimension is absent")))?;
    let expected_width = capture
        .point_count
        .checked_mul(usize::from(dimension))
        .ok_or_else(|| invalid(format!("{context} width overflows usize")))?;
    let mut cursor = CanonicalObservationCursor {
        bytes: values,
        position: 0,
        absolute_start: values_range.start,
    };
    cursor.expect(b"[", context)?;
    for index in 0..expected_width {
        if index != 0 {
            cursor.expect(b",", context)?;
        }
        cursor.expect(b"[", context)?;
        let real = cursor.decimal_string(&format!("{context} component {index} real part"))?;
        cursor.expect(b",", context)?;
        let imaginary =
            cursor.decimal_string(&format!("{context} component {index} imaginary part"))?;
        cursor.expect(b"]", context)?;
        visit(index, real, imaginary)?;
    }
    cursor.expect(b"]", context)?;
    if cursor.position != cursor.bytes.len() {
        return Err(invalid(format!(
            "{context} has trailing or extra observation values"
        )));
    }
    Ok(())
}

fn parse_raw_observation_values(
    capture: &RawNumericalCapture<'_>,
    current_id: u32,
    context: &str,
) -> RusticolResult<Vec<ExactProbeComplex>> {
    let dimension = capture
        .dimensions
        .get(current_id as usize)
        .copied()
        .ok_or_else(|| invalid(format!("{context} dimension is absent")))?;
    let width = capture
        .point_count
        .checked_mul(usize::from(dimension))
        .ok_or_else(|| invalid(format!("{context} width overflows usize")))?;
    let mut values = Vec::with_capacity(width);
    visit_raw_observation_text(capture, current_id, context, |index, real, imaginary| {
        values.push((
            parse_canonical_decimal(real, &format!("{context} component {index} real part"))?,
            parse_canonical_decimal(
                imaginary,
                &format!("{context} component {index} imaginary part"),
            )?,
        ));
        Ok(())
    })?;
    Ok(values)
}

#[cfg(test)]
fn raw_observation_value(
    capture: &RawNumericalCapture<'_>,
    current_id: u32,
    observation_index: usize,
    context: &str,
) -> RusticolResult<ExactProbeComplex> {
    let mut selected = None;
    visit_raw_observation_text(capture, current_id, context, |index, real, imaginary| {
        if index == observation_index {
            selected = Some((
                parse_canonical_decimal(real, &format!("{context} selected real part"))?,
                parse_canonical_decimal(imaginary, &format!("{context} selected imaginary part"))?,
            ));
        }
        Ok(())
    })?;
    selected.ok_or_else(|| invalid(format!("{context} selected observation is absent")))
}

fn raw_observation_values_bytes<'a>(
    capture: &RawNumericalCapture<'a>,
    current_id: u32,
    context: &str,
) -> RusticolResult<&'a [u8]> {
    let row = raw_observation_row(capture, current_id, context)?;
    let range = row.values.as_usize(capture.raw_bytes.len(), context)?;
    capture
        .raw_bytes
        .get(range)
        .ok_or_else(|| invalid(format!("{context} values are absent")))
}

fn exact_probe_scalar(value: &ExactProbeComplex, scalar_component: usize) -> &BigRational {
    if scalar_component == 0 {
        &value.0
    } else {
        &value.1
    }
}

fn build_raw_numerical_candidate_indexes(
    source: &RawSourceSemantics,
    candidate: &RawNumericalCapture<'_>,
    telemetry: &mut RecurrenceEvidenceAuthenticationTelemetry,
) -> RusticolResult<BTreeMap<Vec<u8>, RawNumericalCandidateIndex>> {
    let mut members_by_contract = BTreeMap::<Vec<u8>, Vec<u32>>::new();
    for current in &source.currents {
        members_by_contract
            .entry(current.contract_key.clone())
            .or_default()
            .push(current.current_id);
    }
    members_by_contract
        .into_iter()
        .map(|(contract, members)| {
            let first_id = members
                .first()
                .copied()
                .ok_or_else(|| invalid("raw numerical candidate index has no members"))?;
            let dimension = candidate
                .dimensions
                .get(first_id as usize)
                .copied()
                .ok_or_else(|| invalid("raw numerical candidate index dimension is absent"))?;
            let width = candidate
                .point_count
                .checked_mul(usize::from(dimension))
                .ok_or_else(|| invalid("raw numerical candidate index width overflows usize"))?;
            let scalar_width = width
                .checked_mul(2)
                .ok_or_else(|| invalid("raw numerical candidate scalar width overflows usize"))?;
            let mut scalar_columns = (0..scalar_width)
                .map(|_| Vec::<&str>::with_capacity(members.len()))
                .collect::<Vec<_>>();
            for current_id in &members {
                if candidate.dimensions.get(*current_id as usize) != Some(&dimension) {
                    return Err(invalid(
                        "raw numerical candidate index has inconsistent widths",
                    ));
                }
                visit_raw_observation_text(
                    candidate,
                    *current_id,
                    "raw numerical candidate index",
                    |index, real, imaginary| {
                        scalar_columns[index * 2].push(real);
                        scalar_columns[index * 2 + 1].push(imaginary);
                        Ok(())
                    },
                )?;
                RecurrenceEvidenceAuthenticationTelemetry::checked_add(
                    &mut telemetry.candidate_index_observation_row_scan_count,
                    1,
                    "evidence-authentication candidate-index row-scan count",
                )?;
            }
            let mut best_choice = (0_usize, 0_usize);
            let mut best_score = (0_usize, 0_usize);
            let mut sorted_column = Vec::<&str>::with_capacity(members.len());
            for observation_index in 0..width {
                for scalar_component in 0..2 {
                    let column = &scalar_columns[observation_index * 2 + scalar_component];
                    sorted_column.clear();
                    sorted_column.extend_from_slice(column);
                    sorted_column.sort_unstable();
                    let unique = usize::from(!sorted_column.is_empty())
                        + sorted_column
                            .windows(2)
                            .filter(|pair| pair[0] != pair[1])
                            .count();
                    let nonzero = sorted_column.iter().filter(|value| **value != "0").count();
                    let score = (unique, nonzero);
                    if score > best_score
                        || score == best_score
                            && (observation_index, scalar_component) < best_choice
                    {
                        best_score = score;
                        best_choice = (observation_index, scalar_component);
                    }
                }
            }
            let selected_column_start = best_choice
                .0
                .checked_mul(2)
                .ok_or_else(|| invalid("raw numerical selected column overflows usize"))?;
            let real_column = &scalar_columns[selected_column_start];
            let imaginary_column = &scalar_columns[selected_column_start + 1];
            if real_column.len() != members.len() || imaginary_column.len() != members.len() {
                return Err(invalid(
                    "raw numerical selected observation column is incomplete",
                ));
            }
            let selected_values = members
                .iter()
                .zip(real_column.iter().zip(imaginary_column.iter()))
                .map(|(current_id, (real, imaginary))| {
                    Ok((
                        *current_id,
                        (
                            parse_canonical_decimal(
                                real,
                                "raw numerical candidate index selected real part",
                            )?,
                            parse_canonical_decimal(
                                imaginary,
                                "raw numerical candidate index selected imaginary part",
                            )?,
                        ),
                    ))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            RecurrenceEvidenceAuthenticationTelemetry::checked_add(
                &mut telemetry.retained_selected_observation_count,
                selected_values.len(),
                "evidence-authentication retained selected-observation count",
            )?;
            let mut entries = selected_values
                .iter()
                .map(|(current_id, value)| {
                    (
                        exact_probe_scalar(value, best_choice.1).clone(),
                        *current_id,
                    )
                })
                .collect::<Vec<_>>();
            RecurrenceEvidenceAuthenticationTelemetry::checked_add(
                &mut telemetry.selected_observation_reuse_count,
                selected_values.len(),
                "evidence-authentication selected-observation reuse count",
            )?;
            entries.sort();
            Ok((
                contract,
                RawNumericalCandidateIndex {
                    #[cfg(test)]
                    observation_index: best_choice.0,
                    scalar_component: best_choice.1,
                    entries,
                    selected_values,
                },
            ))
        })
        .collect()
}

fn raw_numerical_tolerance_window_ids(
    index: &RawNumericalCandidateIndex,
    selected: &ExactProbeComplex,
    current_id: u32,
    relation_kind: &str,
    relative_tolerance: &BigRational,
    absolute_tolerance: &BigRational,
) -> RusticolResult<BTreeSet<u32>> {
    if !matches!(relation_kind, "equal" | "opposite") {
        return Err(invalid(
            "raw numerical candidate index relation is unsupported",
        ));
    }
    if relative_tolerance >= &BigRational::one() {
        return Ok(index
            .entries
            .iter()
            .filter_map(|(_value, representative_id)| {
                (*representative_id < current_id).then_some(*representative_id)
            })
            .collect());
    }
    let selected_scalar = exact_probe_scalar(selected, index.scalar_component);
    let target = if relation_kind == "equal" {
        selected_scalar.clone()
    } else {
        -selected_scalar
    };
    // For complex-pair infinity norms X (current), Y (representative), and D
    // (residual), passing gives D <= a+r*max(X,Y) and reverse triangle gives
    // Y <= X+D.  Hence r<1 implies D <= (a+rX)/(1-r); the indexed scalar
    // residual is <= D, so this window cannot discard a passing relation.
    let current_scale = selected.0.abs().max(selected.1.abs());
    let radius = (absolute_tolerance + relative_tolerance * current_scale)
        / (BigRational::one() - relative_tolerance);
    let lower_value = &target - &radius;
    let upper_value = target + radius;
    let lower = index
        .entries
        .partition_point(|(value, _current_id)| value < &lower_value);
    let upper = index
        .entries
        .partition_point(|(value, _current_id)| value <= &upper_value);
    Ok(index.entries[lower..upper]
        .iter()
        .filter_map(|(_value, representative_id)| {
            (*representative_id < current_id).then_some(*representative_id)
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn derive_raw_numerical_relations(
    source: &RawSourceSemantics,
    candidate: &RawNumericalCapture<'_>,
    verification: &RawNumericalCapture<'_>,
    precision_digits: u32,
    seed: u64,
    relative_tolerance: &BigRational,
    absolute_tolerance: &BigRational,
    relative_tolerance_hex: &str,
    absolute_tolerance_hex: &str,
    runtime_parameter_schema_sha256: &str,
    certificate_algorithm: &str,
    telemetry: &mut RecurrenceEvidenceAuthenticationTelemetry,
) -> RusticolResult<RawNumericalDerivation> {
    const DECISION_CHAIN_ABI: &str = "pyamplicol-recurrence-numerical-decision-chain-v1";
    const REJECTION_CHAIN_ABI: &str = "pyamplicol-recurrence-rejected-numerical-decision-chain-v1";
    let candidate_index_started = Instant::now();
    let candidate_indexes = build_raw_numerical_candidate_indexes(source, candidate, telemetry)?;
    telemetry.candidate_index_nanoseconds = elapsed_nanoseconds(candidate_index_started);
    let mut prior_by_contract = BTreeMap::<Vec<u8>, Vec<u32>>::new();
    let mut derived = Vec::new();
    let mut numerical_candidate_count = 0_usize;
    let mut verification_rejected_count = 0_usize;
    let mut tested_hypothesis_count = 0_usize;
    let mut theoretical_pair_hypothesis_count = 0_usize;
    let mut screened_pair_hypothesis_count = 0_usize;
    let mut zero_hypothesis_count = 0_usize;
    let suppressed_selector_domains = if source.strategy == "all-flow-union" {
        all_flow_selector_domain_helicities(source)?
            .into_iter()
            .filter_map(|(domain_id, helicities)| (helicities.len() > 1).then_some(domain_id))
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    let source_semantics_sha256 =
        canonical_json_sha256(&source.value, "raw numerical source semantics")?;
    let mut decision_chain = canonical_json_sha256(
        &serde_json::json!({
            "abi": DECISION_CHAIN_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": verification.capture_contract_sha256,
        }),
        "raw numerical decision-chain root",
    )?;
    let mut rejection_chain = canonical_json_sha256(
        &serde_json::json!({
            "abi": REJECTION_CHAIN_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": verification.capture_contract_sha256,
        }),
        "raw numerical rejection-chain root",
    )?;
    let hypothesis_replay_started = Instant::now();
    for current in &source.currents {
        let prior = prior_by_contract
            .entry(current.contract_key.clone())
            .or_default();
        if current.is_source {
            prior.push(current.current_id);
            continue;
        }
        if suppressed_selector_domains.contains(&current.selector_domain_id) {
            continue;
        }
        let candidate_index = candidate_indexes
            .get(&current.contract_key)
            .ok_or_else(|| invalid("raw numerical candidate index contract is absent"))?;
        let selected = candidate_index
            .selected_value(current.current_id, "raw numerical candidate index current")?;
        RecurrenceEvidenceAuthenticationTelemetry::checked_add(
            &mut telemetry.selected_observation_reuse_count,
            1,
            "evidence-authentication selected-observation reuse count",
        )?;
        let equal_representatives = raw_numerical_tolerance_window_ids(
            candidate_index,
            selected,
            current.current_id,
            "equal",
            relative_tolerance,
            absolute_tolerance,
        )?;
        let opposite_representatives = raw_numerical_tolerance_window_ids(
            candidate_index,
            selected,
            current.current_id,
            "opposite",
            relative_tolerance,
            absolute_tolerance,
        )?;
        theoretical_pair_hypothesis_count = theoretical_pair_hypothesis_count
            .checked_add(
                prior
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| invalid("theoretical pair count overflows usize"))?,
            )
            .ok_or_else(|| invalid("theoretical pair count overflows usize"))?;
        screened_pair_hypothesis_count = screened_pair_hypothesis_count
            .checked_add(equal_representatives.len())
            .and_then(|count| count.checked_add(opposite_representatives.len()))
            .ok_or_else(|| invalid("screened pair count overflows usize"))?;
        zero_hypothesis_count = zero_hypothesis_count
            .checked_add(1)
            .ok_or_else(|| invalid("zero hypothesis count overflows usize"))?;
        screened_pair_hypothesis_count
            .checked_add(zero_hypothesis_count)
            .ok_or_else(|| invalid("total screened hypothesis count overflows usize"))?;
        let mut hypotheses = vec![("zero", None)];
        for representative_id in equal_representatives
            .union(&opposite_representatives)
            .copied()
        {
            if equal_representatives.contains(&representative_id) {
                hypotheses.push(("equal", Some(representative_id)));
            }
            if opposite_representatives.contains(&representative_id) {
                hypotheses.push(("opposite", Some(representative_id)));
            }
        }
        for (relation_kind, representative_id) in hypotheses {
            tested_hypothesis_count = tested_hypothesis_count
                .checked_add(1)
                .ok_or_else(|| invalid("numerical hypothesis count overflows usize"))?;
            let candidate_residuals = relation_residuals(
                relation_kind,
                current.current_id,
                representative_id,
                candidate,
                relative_tolerance,
                absolute_tolerance,
                telemetry,
            )?;
            if candidate_residuals.maximum_ratio > BigRational::one() {
                decision_chain = advance_decision_chain(
                    &decision_chain,
                    current.current_id,
                    representative_id,
                    relation_kind,
                    &candidate_residuals,
                    None,
                    false,
                    telemetry,
                )?;
                rejection_chain = advance_rejection_chain(
                    &rejection_chain,
                    current.current_id,
                    representative_id,
                    relation_kind,
                    &candidate_residuals,
                    "candidate-observations-not-equal",
                    telemetry,
                )?;
                continue;
            }
            numerical_candidate_count = numerical_candidate_count
                .checked_add(1)
                .ok_or_else(|| invalid("numerical candidate count overflows usize"))?;
            let verification_residuals = relation_residuals(
                relation_kind,
                current.current_id,
                representative_id,
                verification,
                relative_tolerance,
                absolute_tolerance,
                telemetry,
            )?;
            if verification_residuals.maximum_ratio > BigRational::one() {
                verification_rejected_count = verification_rejected_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("verification rejection count overflows usize"))?;
                decision_chain = advance_decision_chain(
                    &decision_chain,
                    current.current_id,
                    representative_id,
                    relation_kind,
                    &candidate_residuals,
                    Some(&verification_residuals),
                    false,
                    telemetry,
                )?;
                rejection_chain = advance_rejection_chain(
                    &rejection_chain,
                    current.current_id,
                    representative_id,
                    relation_kind,
                    &verification_residuals,
                    "independent-verification-rejected-candidate",
                    telemetry,
                )?;
                continue;
            }
            let execution_representative_id = if relation_kind == "zero" {
                prior.first().copied().unwrap_or(current.current_id)
            } else {
                representative_id
                    .ok_or_else(|| invalid("non-zero numerical relation has no representative"))?
            };
            derived.push(build_derived_numerical_relation(
                source,
                current,
                representative_id,
                execution_representative_id,
                relation_kind,
                candidate,
                verification,
                &candidate_residuals,
                &verification_residuals,
                precision_digits,
                seed,
                relative_tolerance_hex,
                absolute_tolerance_hex,
                runtime_parameter_schema_sha256,
                certificate_algorithm,
                telemetry,
            )?);
            decision_chain = advance_decision_chain(
                &decision_chain,
                current.current_id,
                representative_id,
                relation_kind,
                &candidate_residuals,
                Some(&verification_residuals),
                true,
                telemetry,
            )?;
            break;
        }
        prior.push(current.current_id);
    }
    telemetry.hypothesis_replay_nanoseconds = elapsed_nanoseconds(hypothesis_replay_started);
    let derivation_finalization_started = Instant::now();
    let rejected_hypothesis_count = tested_hypothesis_count
        .checked_sub(derived.len())
        .ok_or_else(|| invalid("raw numerical rejected-hypothesis count underflows"))?;
    let decision_sha256 = canonical_json_sha256(
        &serde_json::json!({
            "abi": DECISION_CHAIN_ABI,
            "chain_tail_sha256": decision_chain,
            "tested_hypothesis_count": tested_hypothesis_count,
            "theoretical_pair_hypothesis_count": theoretical_pair_hypothesis_count,
            "screened_pair_hypothesis_count": screened_pair_hypothesis_count,
            "zero_hypothesis_count": zero_hypothesis_count,
            "numerical_candidate_count": numerical_candidate_count,
            "verification_rejected_count": verification_rejected_count,
            "rejected_hypothesis_count": rejected_hypothesis_count,
            "certified_relation_count": derived.len(),
        }),
        "raw numerical decision-chain census",
    )?;
    let rejection_decision_sha256 = rejection_chain;
    telemetry.finish_rational_string_census()?;
    telemetry.derivation_finalization_nanoseconds =
        elapsed_nanoseconds(derivation_finalization_started);
    Ok(RawNumericalDerivation {
        relations: derived,
        numerical_candidate_count,
        verification_rejected_count,
        rejected_hypothesis_count,
        tested_hypothesis_count,
        theoretical_pair_hypothesis_count,
        screened_pair_hypothesis_count,
        zero_hypothesis_count,
        candidate_index_contract_count: candidate_indexes.len(),
        exhaustive_fallback_contract_count: if relative_tolerance >= &BigRational::one() {
            candidate_indexes.len()
        } else {
            0
        },
        decision_sha256,
        rejection_decision_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
fn advance_decision_chain(
    previous: &str,
    current_id: u32,
    representative_id: Option<u32>,
    relation_kind: &str,
    candidate: &RelationResiduals,
    verification: Option<&RelationResiduals>,
    selected: bool,
    telemetry: &mut RecurrenceEvidenceAuthenticationTelemetry,
) -> RusticolResult<String> {
    telemetry.record_rational_string_uses(if verification.is_some() { 6 } else { 3 })?;
    let row = serde_json::json!({
        "current_id": current_id,
        "representative_id": representative_id,
        "relation_kind": relation_kind,
        "candidate_maximum_absolute_residual": candidate.strings.maximum_absolute.as_str(),
        "candidate_maximum_relative_residual": candidate.strings.maximum_relative.as_str(),
        "candidate_maximum_tolerance_ratio": candidate.strings.maximum_ratio.as_str(),
        "candidate_accepted": candidate.maximum_ratio <= BigRational::one(),
        "verification_maximum_absolute_residual": verification.map(|value| value.strings.maximum_absolute.as_str()),
        "verification_maximum_relative_residual": verification.map(|value| value.strings.maximum_relative.as_str()),
        "verification_maximum_tolerance_ratio": verification.map(|value| value.strings.maximum_ratio.as_str()),
        "verification_accepted": verification.map(|value| value.maximum_ratio <= BigRational::one()),
        "selected": selected,
    });
    let previous = semantic_digest_from_hex(previous, "raw numerical decision-chain tail")?;
    let mut digest = Sha256::new();
    digest.update(previous.as_bytes());
    digest.update(canonical_json_bytes(&row, "raw numerical decision row")?);
    Ok(hex_digest(digest.finalize()))
}

fn advance_rejection_chain(
    previous: &str,
    current_id: u32,
    representative_id: Option<u32>,
    relation_kind: &str,
    residuals: &RelationResiduals,
    reason: &str,
    telemetry: &mut RecurrenceEvidenceAuthenticationTelemetry,
) -> RusticolResult<String> {
    telemetry.record_rational_string_uses(3)?;
    let row = serde_json::json!({
        "current_id": current_id,
        "representative_id": representative_id,
        "relation_kind": relation_kind,
        "reason": reason,
        "maximum_absolute_residual": residuals.strings.maximum_absolute.as_str(),
        "maximum_relative_residual": residuals.strings.maximum_relative.as_str(),
        "maximum_tolerance_ratio": residuals.strings.maximum_ratio.as_str(),
    });
    let previous = semantic_digest_from_hex(previous, "raw numerical rejection-chain predecessor")?;
    let mut digest = Sha256::new();
    digest.update(previous.as_bytes());
    digest.update(canonical_json_bytes(
        &row,
        "raw numerical rejected hypothesis",
    )?);
    Ok(hex_digest(digest.finalize()))
}

fn relation_residuals(
    relation_kind: &str,
    current_id: u32,
    representative_id: Option<u32>,
    capture: &RawNumericalCapture<'_>,
    relative_tolerance: &BigRational,
    absolute_tolerance: &BigRational,
    telemetry: &mut RecurrenceEvidenceAuthenticationTelemetry,
) -> RusticolResult<RelationResiduals> {
    let mut maximum_absolute = BigRational::zero();
    let mut maximum_relative = BigRational::zero();
    let mut maximum_ratio = BigRational::zero();
    if relation_kind == "zero" {
        if representative_id.is_some() {
            return Err(invalid("raw numerical zero relation has a representative"));
        }
        visit_raw_observation_text(
            capture,
            current_id,
            "raw numerical current observations",
            |index, real, imaginary| {
                let current_value = (
                    parse_canonical_decimal(
                        real,
                        &format!("raw numerical current observations component {index} real part"),
                    )?,
                    parse_canonical_decimal(
                        imaginary,
                        &format!(
                            "raw numerical current observations component {index} imaginary part"
                        ),
                    )?,
                );
                accumulate_relation_residual(
                    relation_kind,
                    &current_value,
                    None,
                    relative_tolerance,
                    absolute_tolerance,
                    &mut maximum_absolute,
                    &mut maximum_relative,
                    &mut maximum_ratio,
                )
            },
        )?;
        let streamed_value_count = capture
            .dimensions
            .get(current_id as usize)
            .copied()
            .and_then(|dimension| capture.point_count.checked_mul(usize::from(dimension)))
            .ok_or_else(|| invalid("raw numerical zero-residual width overflows usize"))?;
        RecurrenceEvidenceAuthenticationTelemetry::checked_add(
            &mut telemetry.zero_residual_streamed_value_count,
            streamed_value_count,
            "evidence-authentication zero-residual streamed-value count",
        )?;
    } else {
        let current = parse_raw_observation_values(
            capture,
            current_id,
            "raw numerical current observations",
        )?;
        let representative_id = representative_id
            .ok_or_else(|| invalid("raw numerical non-zero relation has no representative"))?;
        let representative = parse_raw_observation_values(
            capture,
            representative_id,
            "raw numerical representative observations",
        )?;
        RecurrenceEvidenceAuthenticationTelemetry::checked_add(
            &mut telemetry.materialized_observation_vector_count,
            2,
            "evidence-authentication materialized observation-vector count",
        )?;
        if representative.len() != current.len() {
            return Err(invalid("raw numerical relation width is invalid"));
        }
        for (current_value, representative_value) in current.iter().zip(&representative) {
            accumulate_relation_residual(
                relation_kind,
                current_value,
                Some(representative_value),
                relative_tolerance,
                absolute_tolerance,
                &mut maximum_absolute,
                &mut maximum_relative,
                &mut maximum_ratio,
            )?;
        }
    }
    let strings = RelationResidualStrings {
        maximum_absolute: rational_string(&maximum_absolute),
        maximum_relative: rational_string(&maximum_relative),
        maximum_ratio: rational_string(&maximum_ratio),
    };
    RecurrenceEvidenceAuthenticationTelemetry::checked_add(
        &mut telemetry.rational_string_materialization_count,
        3,
        "evidence-authentication rational-string materialization count",
    )?;
    Ok(RelationResiduals {
        maximum_ratio,
        strings,
    })
}

#[allow(clippy::too_many_arguments)]
fn accumulate_relation_residual(
    relation_kind: &str,
    current_value: &ExactProbeComplex,
    representative_value: Option<&ExactProbeComplex>,
    relative_tolerance: &BigRational,
    absolute_tolerance: &BigRational,
    maximum_absolute: &mut BigRational,
    maximum_relative: &mut BigRational,
    maximum_ratio: &mut BigRational,
) -> RusticolResult<()> {
    let current_scale = current_value.0.abs().max(current_value.1.abs());
    let (difference, scale) = match relation_kind {
        "zero" => {
            if representative_value.is_some() {
                return Err(invalid("raw numerical zero relation has a representative"));
            }
            (current_scale.clone(), current_scale)
        }
        "equal" | "opposite" => {
            let representative_value = representative_value
                .ok_or_else(|| invalid("raw numerical non-zero relation has no representative"))?;
            let (right_real, right_imaginary) = if relation_kind == "equal" {
                (
                    representative_value.0.clone(),
                    representative_value.1.clone(),
                )
            } else {
                (
                    -representative_value.0.clone(),
                    -representative_value.1.clone(),
                )
            };
            (
                (&current_value.0 - right_real)
                    .abs()
                    .max((&current_value.1 - right_imaginary).abs()),
                current_scale
                    .max(representative_value.0.abs())
                    .max(representative_value.1.abs()),
            )
        }
        _ => return Err(invalid("raw numerical relation kind is unsupported")),
    };
    let allowed = absolute_tolerance + relative_tolerance * &scale;
    let (relative, ratio) = if difference.is_zero() {
        (BigRational::zero(), BigRational::zero())
    } else {
        if allowed <= BigRational::zero() {
            return Err(invalid("raw numerical tolerance is not positive"));
        }
        (
            &difference / scale.max(absolute_tolerance.clone()),
            &difference / allowed,
        )
    };
    *maximum_absolute = std::mem::take(maximum_absolute).max(difference);
    *maximum_relative = std::mem::take(maximum_relative).max(relative);
    *maximum_ratio = std::mem::take(maximum_ratio).max(ratio);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_derived_numerical_relation(
    source: &RawSourceSemantics,
    current: &RawSourceCurrent,
    representative_id: Option<u32>,
    execution_representative_id: u32,
    relation_kind: &str,
    candidate: &RawNumericalCapture<'_>,
    verification: &RawNumericalCapture<'_>,
    candidate_residuals: &RelationResiduals,
    verification_residuals: &RelationResiduals,
    precision_digits: u32,
    seed: u64,
    relative_tolerance_hex: &str,
    absolute_tolerance_hex: &str,
    runtime_parameter_schema_sha256: &str,
    certificate_algorithm: &str,
    telemetry: &mut RecurrenceEvidenceAuthenticationTelemetry,
) -> RusticolResult<DerivedNumericalRelation> {
    let factor = match relation_kind {
        "equal" => rusticol_core::recurrence::ExactComplexRational::ONE,
        "opposite" => rusticol_core::recurrence::ExactComplexRational::ONE.checked_neg()?,
        "zero" => rusticol_core::recurrence::ExactComplexRational::ZERO,
        _ => return Err(invalid("raw numerical relation kind is unsupported")),
    };
    let factor_integer = match relation_kind {
        "equal" => serde_json::json!([1, 0]),
        "opposite" => serde_json::json!([-1, 0]),
        "zero" => serde_json::json!([0, 0]),
        _ => unreachable!(),
    };
    let candidate_observations_sha256 = relation_observation_sha256(
        current.current_id,
        representative_id,
        relation_kind,
        candidate,
    )?;
    let verification_observations_sha256 = relation_observation_sha256(
        current.current_id,
        representative_id,
        relation_kind,
        verification,
    )?;
    let probe_contract = serde_json::json!({
        "algorithm": certificate_algorithm,
        "source_semantics_sha256": canonical_json_sha256(
            &source.value,
            "raw numerical source semantics",
        )?,
        "current_id": current.current_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "relation_kind": relation_kind,
        "precision_digits": precision_digits,
        "seed": seed,
        "candidate_domain": "candidate-current-probes-v1",
        "verification_domain": "independent-verification-current-probes-v1",
        "relative_tolerance_binary64": relative_tolerance_hex,
        "absolute_tolerance_binary64": absolute_tolerance_hex,
        "candidate_probe_count": candidate.point_sha256s.len(),
        "verification_probe_count": verification.point_sha256s.len(),
        "current_dimension": current.dimension,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "candidate_capture_sha256": candidate.capture_contract_sha256,
        "verification_capture_sha256": verification.capture_contract_sha256,
        "candidate_observations_sha256": candidate_observations_sha256,
        "verification_observations_sha256": verification_observations_sha256,
    });
    let probe_contract_sha256 =
        canonical_json_sha256(&probe_contract, "raw numerical probe contract")?;
    telemetry.record_rational_string_uses(12)?;
    let proof = serde_json::json!({
        "algorithm": certificate_algorithm,
        "source_semantics_sha256": probe_contract["source_semantics_sha256"],
        "current_id": current.current_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "relation_kind": relation_kind,
        "precision_digits": precision_digits,
        "seed": seed,
        "candidate_domain": "candidate-current-probes-v1",
        "verification_domain": "independent-verification-current-probes-v1",
        "relative_tolerance_binary64": relative_tolerance_hex,
        "absolute_tolerance_binary64": absolute_tolerance_hex,
        "candidate_probe_count": candidate.point_sha256s.len(),
        "verification_probe_count": verification.point_sha256s.len(),
        "current_dimension": current.dimension,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "candidate_capture_sha256": candidate.capture_contract_sha256,
        "verification_capture_sha256": verification.capture_contract_sha256,
        "candidate_observations_sha256": candidate_observations_sha256,
        "verification_observations_sha256": verification_observations_sha256,
        "proof_kind": "authenticated-numerical",
        "factor_integer": factor_integer,
        "candidate_maximum_absolute_residual": candidate_residuals.strings.maximum_absolute.as_str(),
        "candidate_maximum_relative_residual": candidate_residuals.strings.maximum_relative.as_str(),
        "candidate_maximum_tolerance_ratio": candidate_residuals.strings.maximum_ratio.as_str(),
        "verification_maximum_absolute_residual": verification_residuals.strings.maximum_absolute.as_str(),
        "verification_maximum_relative_residual": verification_residuals.strings.maximum_relative.as_str(),
        "verification_maximum_tolerance_ratio": verification_residuals.strings.maximum_ratio.as_str(),
        "probe_contract_sha256": probe_contract_sha256,
    });
    let proof_sha256 = canonical_json_sha256(&proof, "raw numerical relation proof")?;
    let certificate = serde_json::json!({
        "algorithm": certificate_algorithm,
        "proof_kind": "authenticated-numerical",
        "relation_kind": relation_kind,
        "current_id": current.current_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "factor_integer": factor_integer,
        "source_semantics_sha256": probe_contract["source_semantics_sha256"],
        "precision_digits": precision_digits,
        "seed": seed,
        "relative_tolerance_binary64": relative_tolerance_hex,
        "absolute_tolerance_binary64": absolute_tolerance_hex,
        "candidate_probe_count": candidate.point_sha256s.len(),
        "verification_probe_count": verification.point_sha256s.len(),
        "current_dimension": current.dimension,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "candidate_capture_sha256": candidate.capture_contract_sha256,
        "verification_capture_sha256": verification.capture_contract_sha256,
        "candidate_maximum_absolute_residual": candidate_residuals.strings.maximum_absolute.as_str(),
        "candidate_maximum_relative_residual": candidate_residuals.strings.maximum_relative.as_str(),
        "candidate_maximum_tolerance_ratio": candidate_residuals.strings.maximum_ratio.as_str(),
        "verification_maximum_absolute_residual": verification_residuals.strings.maximum_absolute.as_str(),
        "verification_maximum_relative_residual": verification_residuals.strings.maximum_relative.as_str(),
        "verification_maximum_tolerance_ratio": verification_residuals.strings.maximum_ratio.as_str(),
        "candidate_observations_sha256": candidate_observations_sha256,
        "verification_observations_sha256": verification_observations_sha256,
        "probe_contract_sha256": probe_contract_sha256,
        "proof_sha256": proof_sha256,
    });
    let mapping_json = serde_json::json!({
        "current_id": current.current_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "relation_kind": relation_kind,
        "factor_integer": factor_integer,
        "current_dimension": current.dimension,
        "certificate_proof_sha256": proof_sha256,
        "candidate_observations_sha256": candidate_observations_sha256,
        "verification_observations_sha256": verification_observations_sha256,
    });
    Ok(DerivedNumericalRelation {
        mapping: RecurrenceNumericalCurrentMapping {
            current_id: current.current_id,
            representative_id,
            execution_representative_id,
            relation_kind: relation_kind.to_owned(),
            factor,
            current_dimension: current.dimension,
            certificate_proof_sha256: proof_sha256,
            candidate_observations_sha256,
            verification_observations_sha256,
        },
        certificate,
        mapping_json,
    })
}

fn relation_observation_sha256(
    current_id: u32,
    representative_id: Option<u32>,
    relation_kind: &str,
    capture: &RawNumericalCapture<'_>,
) -> RusticolResult<String> {
    const CURRENT_PLACEHOLDER: &str = "__pyamplicol_authenticated_current_values_placeholder__";
    const REPRESENTATIVE_PLACEHOLDER: &str =
        "__pyamplicol_authenticated_representative_values_placeholder__";
    let current_values =
        raw_observation_values_bytes(capture, current_id, "raw relation current observations")?;
    let current_dimension = capture
        .dimensions
        .get(current_id as usize)
        .copied()
        .ok_or_else(|| invalid("raw relation current dimension is absent"))?;
    let (representative_values, representative_raw) =
        if let Some(representative_id) = representative_id {
            (
                serde_json::json!([REPRESENTATIVE_PLACEHOLDER]),
                Some(raw_observation_values_bytes(
                    capture,
                    representative_id,
                    "raw relation representative observations",
                )?),
            )
        } else {
            (JsonValue::Null, None)
        };
    let payload = serde_json::json!({
        "abi": "pyamplicol-recurrence-relation-observation-v2",
        "capture_contract_sha256": capture.capture_contract_sha256,
        "current_id": current_id,
        "representative_id": representative_id,
        "relation_kind": relation_kind,
        "current_dimension": current_dimension,
        "current_values": [CURRENT_PLACEHOLDER],
        "representative_values": representative_values,
    });
    let mut replacements = vec![(CURRENT_PLACEHOLDER, current_values)];
    if let Some(representative_raw) = representative_raw {
        replacements.push((REPRESENTATIVE_PLACEHOLDER, representative_raw));
    }
    canonical_json_sha256_with_raw_arrays(
        &payload,
        &replacements,
        "raw numerical relation observations",
    )
}

fn rational_string(value: &BigRational) -> String {
    if value.denom().is_one() {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

fn evidence_sha256(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<String> {
    let value = json_string(object, field, context)?;
    semantic_digest_from_hex(value, context)?;
    Ok(value.to_owned())
}

fn evidence_usize(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<usize> {
    json_field(object, field, context)?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid(format!("{context} must be a nonnegative usize")))
}

fn json_u64(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<u64> {
    json_field(object, field, context)?
        .as_u64()
        .ok_or_else(|| invalid(format!("{context} must be a nonnegative u64")))
}

fn python_f64_hex(value: f64) -> String {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff_ffff_ffff == 0 {
        return "0x0.0p+0".to_owned();
    }
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (leading, exponent) = if exponent_bits == 0 {
        (0, -1022)
    } else {
        (1, exponent_bits - 1023)
    };
    format!("0x{leading}.{fraction:013x}p{exponent:+}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::ZlibEncoder};
    use rusticol_core::recurrence::RecurrenceRelationDiscoveryOptions;
    use serde_json::json;

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn empty_runtime_parameter_contract() -> AuthenticatedRuntimeParameterContract {
        let schema = json!({
            "abi": "pyamplicol-recurrence-runtime-parameter-schema-v2",
            "parameters": [],
        });
        AuthenticatedRuntimeParameterContract {
            schema_sha256: canonical_json_sha256(&schema, "empty runtime parameter schema")
                .unwrap(),
            schema,
            slots: Vec::new(),
            process_id: "raw-v3-test".to_owned(),
            physical_pdgs: vec![1],
        }
    }

    fn parse_fixture(bytes: &[u8]) -> RusticolResult<RecurrenceNumericalRelationEvidence> {
        parse_numerical_relation_evidence(bytes, &empty_runtime_parameter_contract())
    }

    fn scanner_source(metadata_bytes: &[u8]) -> (JsonValue, RawSourceSemantics) {
        let metadata: JsonValue = serde_json::from_slice(metadata_bytes).unwrap();
        let object = metadata.as_object().unwrap();
        let source_semantics_sha256 = object["source_semantics_sha256"].as_str().unwrap();
        let source = parse_raw_source_semantics(
            &object["source_semantics"],
            source_semantics_sha256,
            object["schedule_semantic_digest"].as_str().unwrap(),
            object["baseline_runtime_layout_digest"].as_str().unwrap(),
            &empty_runtime_parameter_contract(),
        )
        .unwrap();
        (metadata, source)
    }

    fn capture_from_observation_array<'a>(
        encoded: &'a [u8],
        source: &RawSourceSemantics,
        point_count: usize,
    ) -> RawNumericalCapture<'a> {
        let observation_array =
            RawByteRange::from_usize(0, encoded.len(), encoded.len(), "test observation array")
                .unwrap();
        let observation_rows =
            scan_raw_observation_array(encoded, observation_array, source, point_count, "test")
                .unwrap();
        RawNumericalCapture {
            raw_bytes: encoded,
            observation_rows,
            point_count,
            point_sha256s: Vec::new(),
            kinematic_sha256s: Vec::new(),
            parameter_context_sha256s: Vec::new(),
            parameter_contexts: Vec::new(),
            dimensions: source
                .currents
                .iter()
                .map(|current| current.dimension)
                .collect(),
            observation_batch_sha256: digest(80),
            capture_contract_sha256: digest(81),
        }
    }

    fn refresh_numerical_certificate_set_digest(value: &mut JsonValue) {
        let object = value.as_object().unwrap();
        let relation_set = json!({
            "abi": "pyamplicol-authenticated-numerical-current-relation-set-v2",
            "certificates": object["certificates"].clone(),
            "mappings": object["mappings"].clone(),
        });
        let digest =
            canonical_json_sha256(&relation_set, "test numerical relation certificate set")
                .unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("certificate_set_sha256".to_owned(), json!(digest));
    }

    fn raw_capture_fixture(
        domain: &str,
        point_base: u32,
        source_semantics_sha256: &str,
        runtime_parameter_schema_sha256: &str,
        seed: u64,
    ) -> JsonValue {
        let points = (0..4_u32)
            .map(|index| {
                let value = point_base + index;
                json!({
                    "schema_version": 1,
                    "kind": "pyamplicol-rusticol-validation-momenta",
                    "process_id": "raw-v3-test",
                    "process": "1 -> 1",
                    "seed": domain_seed(seed, domain, index as usize),
                    "available": true,
                    "error": null,
                    "points": [[{
                        "pdg": 1,
                        "momentum": [value.to_string(), "0", "0", "0"],
                    }]],
                })
            })
            .collect::<Vec<_>>();
        let point_sha256s = points
            .iter()
            .map(|point| canonical_json_sha256(point, "fixture point").unwrap())
            .collect::<Vec<_>>();
        let kinematic_binary64 = (0..4_u32)
            .map(|index| {
                json!([[
                    python_f64_hex(f64::from(point_base + index)),
                    "0x0.0p+0",
                    "0x0.0p+0",
                    "0x0.0p+0",
                ]])
            })
            .collect::<Vec<_>>();
        let kinematic_sha256s = kinematic_binary64
            .iter()
            .map(|row| canonical_json_sha256(row, "fixture kinematics").unwrap())
            .collect::<Vec<_>>();
        let selector_contexts = (0..4)
            .map(|_| {
                json!({
                    "strategy": "contracted-color-union",
                    "fixed_source_schedule": true,
                })
            })
            .collect::<Vec<_>>();
        let selector_context_sha256s = selector_contexts
            .iter()
            .map(|row| canonical_json_sha256(row, "fixture selector").unwrap())
            .collect::<Vec<_>>();
        let parameter_contexts = vec![json!([]); 4];
        let parameter_context_sha256s = (0..4)
            .map(|point_index| {
                canonical_json_sha256(
                    &json!({
                        "abi": "pyamplicol-recurrence-parameter-context-v1",
                        "domain": domain,
                        "point_index": point_index,
                        "point_sha256": point_sha256s[point_index],
                        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
                        "values": [],
                    }),
                    "fixture parameter context",
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let observation_scalars = [
            [1_i64, 1, 1, 1],
            [2, 3, 4, 5],
            [2, 3, 4, 5],
            [-2, -3, -4, -5],
            [0, 0, 0, 0],
            [7, 8, 9, 11],
        ];
        let observations = observation_scalars
            .iter()
            .enumerate()
            .map(|(current_id, values)| {
                json!({
                    "current_id": current_id,
                    "dimension": 1,
                    "values": values
                        .iter()
                        .map(|value| json!([value.to_string(), "0"]))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        let dimensions = (0..observation_scalars.len())
            .map(|current_id| json!({"current_id": current_id, "dimension": 1}))
            .collect::<Vec<_>>();
        let observation_batch_sha256 = canonical_json_sha256(
            &json!({
                "abi": "pyamplicol-recurrence-current-observation-batch-v2",
                "source_semantics_sha256": source_semantics_sha256,
                "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
                "point_sha256s": point_sha256s,
                "selector_context_sha256s": selector_context_sha256s,
                "parameter_context_sha256s": parameter_context_sha256s,
                "currents": observations,
            }),
            "fixture observation batch",
        )
        .unwrap();
        let capture_contract_sha256 = canonical_json_sha256(
            &json!({
                "abi": "pyamplicol-recurrence-current-observation-capture-v2",
                "precision_digits": 96,
                "source_semantics_sha256": source_semantics_sha256,
                "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
                "point_sha256s": point_sha256s,
                "kinematic_sha256s": kinematic_sha256s,
                "selector_context_sha256s": selector_context_sha256s,
                "parameter_context_sha256s": parameter_context_sha256s,
                "current_dimensions": dimensions,
                "observation_batch_sha256": observation_batch_sha256,
                "context_policy": "fixed-contracted-source-schedule-v1",
            }),
            "fixture capture contract",
        )
        .unwrap();
        let dimensions_object = (0..observation_scalars.len())
            .map(|current_id| (current_id.to_string(), json!(1)))
            .collect::<JsonMap<_, _>>();
        let value = json!({
            "abi": "pyamplicol-recurrence-current-observation-capture-v2",
            "precision_digits": 96,
            "point_count": 4,
            "point_sha256s": point_sha256s,
            "kinematic_sha256s": kinematic_sha256s,
            "parameter_contexts": parameter_contexts,
            "parameter_context_sha256s": parameter_context_sha256s,
            "context_sha256s": selector_context_sha256s,
            "points": points,
            "current_count": observation_scalars.len(),
            "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
            "source_semantics_sha256": source_semantics_sha256,
            "observation_batch_sha256": observation_batch_sha256,
            "capture_contract_sha256": capture_contract_sha256,
            "context_policy": "fixed-contracted-source-schedule-v1",
            "complete_current_components": true,
            "point_major": true,
            "current_dimensions_sha256": canonical_json_sha256(
                &JsonValue::Object(dimensions_object),
                "fixture dimensions",
            ).unwrap(),
            "evaluator": "recurrence-direct-plan-decimal-symbolica-exact",
            "kinematic_binary64": kinematic_binary64,
            "selector_contexts": selector_contexts,
            "current_dimensions": dimensions,
            "observations": observations,
        });
        value
    }

    fn canonical_numerical_relation_evidence() -> JsonValue {
        canonical_numerical_relation_evidence_with_tolerances(1.0e-70, 1.0e-80)
    }

    fn canonical_numerical_relation_evidence_with_tolerances(
        relative_tolerance: f64,
        absolute_tolerance: f64,
    ) -> JsonValue {
        let seed = 0x5059_414d_u64;
        let source_semantics = json!({
            "abi": "pyamplicol-recurrence-numerical-current-source-v3",
            "process_id": "raw-v3-test",
            "strategy": "contracted-color-union",
            "schedule_semantic_digest": digest(40),
            "baseline_runtime_layout_digest": digest(41),
            "selector_schedule": {
                "policy": "fixed-contracted-source-schedule-v1",
                "fixed_source_schedule": true,
            },
            "currents": (0..6_u32).map(|current_id| json!({
                "current_id": current_id,
                "is_source": current_id == 0,
                "contract": [
                    if current_id == 0 { 0 } else { 1 },
                    1, 7, 1, 0, 0, u32::MAX, u32::MAX, ["1", "1", "0", "1"],
                ],
            })).collect::<Vec<_>>(),
        });
        let source_semantics_sha256 =
            canonical_json_sha256(&source_semantics, "fixture source semantics").unwrap();
        let source = parse_raw_source_semantics(
            &source_semantics,
            &source_semantics_sha256,
            &digest(40),
            &digest(41),
            &empty_runtime_parameter_contract(),
        )
        .unwrap();
        let runtime_parameter_schema = json!({
            "abi": "pyamplicol-recurrence-runtime-parameter-schema-v2",
            "parameters": [],
        });
        let runtime_parameter_schema_sha256 = canonical_json_sha256(
            &runtime_parameter_schema,
            "fixture runtime parameter schema",
        )
        .unwrap();
        let candidate_value = raw_capture_fixture(
            "candidate-current-probes-v1",
            1,
            &source_semantics_sha256,
            &runtime_parameter_schema_sha256,
            seed,
        );
        let verification_value = raw_capture_fixture(
            "independent-verification-current-probes-v1",
            101,
            &source_semantics_sha256,
            &runtime_parameter_schema_sha256,
            seed,
        );
        let fixture_wire = canonical_json_bytes(
            &json!({
                "candidate_capture": candidate_value.clone(),
                "verification_capture": verification_value.clone(),
            }),
            "raw capture fixture wire",
        )
        .unwrap();
        let located = locate_raw_observation_arrays(&fixture_wire).unwrap();
        let metadata: JsonValue = serde_json::from_slice(&located.metadata_bytes).unwrap();
        let candidate = parse_raw_numerical_capture(
            &fixture_wire,
            located.candidate,
            &metadata["candidate_capture"],
            "candidate",
            "candidate-current-probes-v1",
            96,
            4,
            &source,
            &source_semantics_sha256,
            &runtime_parameter_schema_sha256,
            0,
            seed,
        )
        .unwrap();
        let verification = parse_raw_numerical_capture(
            &fixture_wire,
            located.verification,
            &metadata["verification_capture"],
            "verification",
            "independent-verification-current-probes-v1",
            96,
            4,
            &source,
            &source_semantics_sha256,
            &runtime_parameter_schema_sha256,
            0,
            seed,
        )
        .unwrap();
        validate_independent_raw_captures(
            &candidate,
            &verification,
            &empty_runtime_parameter_contract(),
            seed,
        )
        .unwrap();
        let mut derivation_telemetry = RecurrenceEvidenceAuthenticationTelemetry::default();
        let derivation = derive_raw_numerical_relations(
            &source,
            &candidate,
            &verification,
            96,
            seed,
            &exact_binary64_rational(relative_tolerance).unwrap(),
            &exact_binary64_rational(absolute_tolerance).unwrap(),
            &python_f64_hex(relative_tolerance),
            &python_f64_hex(absolute_tolerance),
            &runtime_parameter_schema_sha256,
            rusticol_core::recurrence::NUMERICAL_RELATION_CERTIFICATE_ALGORITHM,
            &mut derivation_telemetry,
        )
        .unwrap();
        assert_eq!(
            derivation
                .relations
                .iter()
                .map(|relation| relation.mapping.relation_kind.as_str())
                .collect::<Vec<_>>(),
            ["equal", "opposite", "zero"],
        );
        let certificates = derivation
            .relations
            .iter()
            .map(|relation| relation.certificate.clone())
            .collect::<Vec<_>>();
        let mappings = derivation
            .relations
            .iter()
            .map(|relation| relation.mapping_json.clone())
            .collect::<Vec<_>>();
        let mut evidence = json!({
            "abi": "pyamplicol-recurrence-numerical-current-evidence-v3",
            "requested_mode": "certified-reuse",
            "schedule_semantic_digest": digest(40),
            "baseline_runtime_layout_digest": digest(41),
            "source_semantics": source_semantics,
            "source_semantics_sha256": source_semantics_sha256,
            "runtime_parameter_schema": runtime_parameter_schema,
            "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
            "candidate_capture": candidate_value,
            "verification_capture": verification_value,
            "certificate_algorithm":
                rusticol_core::recurrence::NUMERICAL_RELATION_CERTIFICATE_ALGORITHM,
            "certificate_set_sha256": digest(0),
            "precision_digits": 96,
            "probe_count": 4,
            "verification_probe_count": 4,
            "relative_tolerance_binary64": python_f64_hex(relative_tolerance),
            "absolute_tolerance_binary64": python_f64_hex(absolute_tolerance),
            "seed": seed,
            "candidate_index": {
                "algorithm": "complete-contract-anchor-tolerance-window-v1",
                "completeness": "complete-within-configured-tolerance",
                "contract_count": derivation.candidate_index_contract_count,
                "exhaustive_fallback_contract_count":
                    derivation.exhaustive_fallback_contract_count,
                "theoretical_pair_hypothesis_count":
                    derivation.theoretical_pair_hypothesis_count,
                "screened_pair_hypothesis_count":
                    derivation.screened_pair_hypothesis_count,
                "zero_hypothesis_count": derivation.zero_hypothesis_count,
                "nearest_rejected_scope":
                    "zero-and-tolerance-window-screened-hypotheses",
            },
            "numerical_candidate_count": derivation.numerical_candidate_count,
            "verification_rejected_count": derivation.verification_rejected_count,
            "rejected_hypothesis_count": derivation.rejected_hypothesis_count,
            "tested_hypothesis_count": derivation.tested_hypothesis_count,
            "decision_sha256": derivation.decision_sha256,
            "rejection_decision_sha256":
                derivation.rejection_decision_sha256,
            "certificates": certificates,
            "mappings": mappings,
        });
        refresh_numerical_certificate_set_digest(&mut evidence);
        evidence
    }

    #[test]
    fn numerical_relation_evidence_recomputes_every_digest_layer() {
        let evidence = canonical_numerical_relation_evidence();
        let encoded = canonical_json_bytes(&evidence, "test numerical evidence").unwrap();
        assert_eq!(encoded.len(), 16_778);
        assert_eq!(
            hex_digest(Sha256::digest(&encoded)),
            "aceda1cf45eddf400159f8ce7f60c3bc0de399273fd7a035b5343544d7b19c96",
        );
        assert_eq!(
            evidence["decision_sha256"],
            "e1899c8791f374e406550435e7ff3e3c57044be2a9161b3f39a023df283cc403",
        );
        assert_eq!(
            evidence["rejection_decision_sha256"],
            "ff294e07e4c84a5c6645c71ad1b9c55aff8dc5f7bfa302b3f20c367ffcfa01be",
        );
        assert_eq!(
            evidence["certificate_set_sha256"],
            "5c500444d772f455df8650acaa1d38ca75638452b84c8e9bc0a5037756a04c97",
        );
        let parsed = parse_fixture(&encoded).unwrap();
        assert_eq!(parsed.mappings.len(), 3);
        assert_eq!(
            parsed.rejected_hypothesis_count,
            parsed.tested_hypothesis_count - parsed.mappings.len(),
        );
        assert!(parsed.rejected_hypothesis_count > parsed.verification_rejected_count);

        let mut stale_set = evidence.clone();
        stale_set["certificates"][0]["current_dimension"] = json!(3);
        let stale_bytes =
            canonical_json_bytes(&stale_set, "stale numerical certificate set").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("not derived from raw observations")
        );

        let mut stale_proof = evidence.clone();
        stale_proof["certificates"][0]["candidate_maximum_absolute_residual"] = json!("1e-90");
        refresh_numerical_certificate_set_digest(&mut stale_proof);
        let stale_bytes = canonical_json_bytes(&stale_proof, "stale numerical proof").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("not derived from raw observations")
        );

        let mut stale_mapping = evidence.clone();
        stale_mapping["mappings"][0]["current_dimension"] = json!(3);
        refresh_numerical_certificate_set_digest(&mut stale_mapping);
        let stale_bytes = canonical_json_bytes(&stale_mapping, "stale numerical mapping").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("not derived from raw observations")
        );

        let mut stale_tolerance = evidence;
        stale_tolerance["relative_tolerance_binary64"] = json!("0x1.0p+0");
        let stale_bytes =
            canonical_json_bytes(&stale_tolerance, "stale numerical tolerance").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("canonical nonnegative finite binary64")
        );

        let mut stale_index = canonical_numerical_relation_evidence();
        stale_index["candidate_index"]["screened_pair_hypothesis_count"] = json!(99);
        let stale_bytes =
            canonical_json_bytes(&stale_index, "stale numerical candidate index").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("was not independently derived")
        );
    }

    #[test]
    fn profiled_evidence_authentication_is_semantically_identical() {
        let evidence = canonical_numerical_relation_evidence();
        let encoded = canonical_json_bytes(&evidence, "profiled numerical evidence").unwrap();
        let expected = parse_fixture(&encoded).unwrap();
        let (profiled, telemetry) = parse_numerical_relation_evidence_with_telemetry(
            &encoded,
            &empty_runtime_parameter_contract(),
        )
        .unwrap();

        assert_eq!(profiled, expected);
        assert_eq!(telemetry.transport_decode_nanoseconds, 0);
        assert!(telemetry.total_nanoseconds >= telemetry.candidate_index_nanoseconds);
        assert!(telemetry.total_nanoseconds >= telemetry.hypothesis_replay_nanoseconds);
        assert!(telemetry.total_nanoseconds >= telemetry.evidence_finalization_nanoseconds);
        assert_eq!(
            (
                telemetry.candidate_index_observation_row_scan_count,
                telemetry.retained_selected_observation_count,
                telemetry.selected_observation_reuse_count,
                telemetry.zero_residual_streamed_value_count,
                telemetry.materialized_observation_vector_count,
                telemetry.rational_string_materialization_count,
                telemetry.rational_string_use_count,
                telemetry.rational_string_reuse_count,
            ),
            (6, 6, 11, 24, 8, 30, 78, 48),
        );

        let compressed = compressed_envelope(&encoded);
        let (compressed_profiled, compressed_telemetry) =
            parse_numerical_relation_evidence_with_telemetry(
                &compressed,
                &empty_runtime_parameter_contract(),
            )
            .unwrap();
        assert_eq!(compressed_profiled, expected);
        assert!(
            compressed_telemetry.total_nanoseconds
                >= compressed_telemetry.transport_decode_nanoseconds
        );
        assert_eq!(
            compressed_telemetry.rational_string_reuse_count,
            telemetry.rational_string_reuse_count,
        );
    }

    #[test]
    fn candidate_index_retains_the_exact_selected_observations() {
        let source = RawSourceSemantics {
            value: json!({}),
            process_id: "selected-observation-test".to_owned(),
            physical_pdgs: vec![1],
            strategy: "contracted-color-union".to_owned(),
            selector_schedule: json!({}),
            currents: (0..3)
                .map(|current_id| RawSourceCurrent {
                    current_id,
                    is_source: false,
                    contract_key: vec![7],
                    dimension: 1,
                    selector_domain_id: 0,
                })
                .collect(),
        };
        let bytes = canonical_json_bytes(
            &json!([
                {"current_id": 0, "dimension": 1, "values": [["1", "10"], ["4", "0"]]},
                {"current_id": 1, "dimension": 1, "values": [["2", "10"], ["4", "0"]]},
                {"current_id": 2, "dimension": 1, "values": [["3", "10"], ["4", "0"]]},
            ]),
            "selected observation fixture",
        )
        .unwrap();
        let capture = capture_from_observation_array(&bytes, &source, 2);
        let mut telemetry = RecurrenceEvidenceAuthenticationTelemetry::default();
        let indexes =
            build_raw_numerical_candidate_indexes(&source, &capture, &mut telemetry).unwrap();
        let index = indexes.get(&vec![7]).unwrap();

        assert_eq!((index.observation_index, index.scalar_component), (0, 0));
        for current_id in 0..3 {
            assert_eq!(
                index
                    .selected_value(current_id, "retained selected observation")
                    .unwrap(),
                &raw_observation_value(
                    &capture,
                    current_id,
                    index.observation_index,
                    "retained selected observation",
                )
                .unwrap(),
            );
        }
        assert_eq!(telemetry.candidate_index_observation_row_scan_count, 3);
        assert_eq!(telemetry.retained_selected_observation_count, 3);
        assert_eq!(telemetry.selected_observation_reuse_count, 3);
        assert!(
            index
                .selected_value(u32::MAX, "retained selected observation")
                .unwrap_err()
                .to_string()
                .contains("selected observation is absent")
        );
    }

    #[test]
    fn multi_helicity_all_flow_domains_contribute_no_raw_hypotheses() {
        let source_value = json!({"synthetic": "multi-helicity-all-flow"});
        let source = RawSourceSemantics {
            value: source_value.clone(),
            process_id: "unsafe-selector-domain-test".to_owned(),
            physical_pdgs: vec![1],
            strategy: "all-flow-union".to_owned(),
            selector_schedule: json!({
                "policy": "seeded-helicity-per-selector-domain-and-physical-point-v1",
                "resolved_helicities": [
                    {"helicity_index": 0, "helicity_id": 0, "selector_domain_id": 0},
                    {"helicity_index": 1, "helicity_id": 2, "selector_domain_id": 0},
                ],
            }),
            currents: (0..3)
                .map(|current_id| RawSourceCurrent {
                    current_id,
                    is_source: current_id == 0,
                    contract_key: vec![7],
                    dimension: 1,
                    selector_domain_id: 0,
                })
                .collect(),
        };
        let bytes = canonical_json_bytes(
            &json!([
                {"current_id": 0, "dimension": 1, "values": [["1", "0"], ["1", "0"]]},
                {"current_id": 1, "dimension": 1, "values": [["1", "0"], ["1", "0"]]},
                {"current_id": 2, "dimension": 1, "values": [["1", "0"], ["1", "0"]]},
            ]),
            "unsafe selector-domain observations",
        )
        .unwrap();
        let candidate = capture_from_observation_array(&bytes, &source, 2);
        let verification = capture_from_observation_array(&bytes, &source, 2);
        let runtime_parameter_schema_sha256 = digest(82);
        let mut telemetry = RecurrenceEvidenceAuthenticationTelemetry::default();
        let derivation = derive_raw_numerical_relations(
            &source,
            &candidate,
            &verification,
            96,
            67,
            &exact_binary64_rational(1.0e-70).unwrap(),
            &exact_binary64_rational(1.0e-80).unwrap(),
            &python_f64_hex(1.0e-70),
            &python_f64_hex(1.0e-80),
            &runtime_parameter_schema_sha256,
            rusticol_core::recurrence::NUMERICAL_RELATION_CERTIFICATE_ALGORITHM,
            &mut telemetry,
        )
        .unwrap();

        assert!(derivation.relations.is_empty());
        assert_eq!(derivation.tested_hypothesis_count, 0);
        assert_eq!(derivation.theoretical_pair_hypothesis_count, 0);
        assert_eq!(derivation.screened_pair_hypothesis_count, 0);
        assert_eq!(derivation.zero_hypothesis_count, 0);
        let chain_root = canonical_json_sha256(
            &json!({
                "abi": "pyamplicol-recurrence-numerical-decision-chain-v1",
                "source_semantics_sha256": canonical_json_sha256(
                    &source_value,
                    "unsafe selector-domain source",
                ).unwrap(),
                "candidate_capture_sha256": candidate.capture_contract_sha256,
                "verification_capture_sha256": verification.capture_contract_sha256,
            }),
            "unsafe selector-domain decision root",
        )
        .unwrap();
        let expected = canonical_json_sha256(
            &json!({
                "abi": "pyamplicol-recurrence-numerical-decision-chain-v1",
                "chain_tail_sha256": chain_root,
                "tested_hypothesis_count": 0,
                "theoretical_pair_hypothesis_count": 0,
                "screened_pair_hypothesis_count": 0,
                "zero_hypothesis_count": 0,
                "numerical_candidate_count": 0,
                "verification_rejected_count": 0,
                "rejected_hypothesis_count": 0,
                "certified_relation_count": 0,
            }),
            "unsafe selector-domain decision census",
        )
        .unwrap();
        assert_eq!(derivation.decision_sha256, expected);
    }

    #[test]
    fn zero_residuals_stream_without_materialized_comparison_vectors() {
        let source = RawSourceSemantics {
            value: json!({}),
            process_id: "zero-residual-test".to_owned(),
            physical_pdgs: vec![1],
            strategy: "contracted-color-union".to_owned(),
            selector_schedule: json!({}),
            currents: vec![RawSourceCurrent {
                current_id: 0,
                is_source: false,
                contract_key: vec![1],
                dimension: 1,
                selector_domain_id: 0,
            }],
        };
        let bytes = canonical_json_bytes(
            &json!([
                {"current_id": 0, "dimension": 1, "values": [["3", "4"], ["0", "0"]]},
            ]),
            "zero residual fixture",
        )
        .unwrap();
        let capture = capture_from_observation_array(&bytes, &source, 2);
        let mut telemetry = RecurrenceEvidenceAuthenticationTelemetry::default();
        let residuals = relation_residuals(
            "zero",
            0,
            None,
            &capture,
            &BigRational::zero(),
            &BigRational::one(),
            &mut telemetry,
        )
        .unwrap();

        assert_eq!(residuals.maximum_ratio, BigRational::from_integer(4.into()));
        assert_eq!(residuals.strings.maximum_absolute, "4");
        assert_eq!(residuals.strings.maximum_relative, "1");
        assert_eq!(residuals.strings.maximum_ratio, "4");
        assert_eq!(telemetry.zero_residual_streamed_value_count, 2);
        assert_eq!(telemetry.materialized_observation_vector_count, 0);
        assert_eq!(telemetry.rational_string_materialization_count, 3);
    }

    #[test]
    #[ignore = "manual recurrence evidence-authentication microbenchmark"]
    fn benchmark_canonical_evidence_authentication() {
        let evidence = canonical_numerical_relation_evidence();
        let encoded = canonical_json_bytes(&evidence, "benchmark numerical evidence").unwrap();
        let started = std::time::Instant::now();
        for _ in 0..50 {
            let parsed = parse_fixture(&encoded).unwrap();
            assert_eq!(parsed.mappings.len(), 3);
        }
        eprintln!(
            "evidence-authentication-benchmark iterations=50 elapsed_ns={}",
            started.elapsed().as_nanos()
        );
    }

    #[test]
    fn raw_probe_identity_is_bound_to_the_authenticated_process_and_seed_domain() {
        fn mutate_first_candidate_point(
            mut evidence: JsonValue,
            field: &str,
            value: JsonValue,
        ) -> Vec<u8> {
            evidence["candidate_capture"]["points"][0][field] = value;
            let digest = canonical_json_sha256(
                &evidence["candidate_capture"]["points"][0],
                "tampered candidate point",
            )
            .unwrap();
            evidence["candidate_capture"]["point_sha256s"][0] = json!(digest);
            canonical_json_bytes(&evidence, "tampered probe identity").unwrap()
        }

        let wrong_id = mutate_first_candidate_point(
            canonical_numerical_relation_evidence(),
            "process_id",
            json!("unrelated-process"),
        );
        assert!(
            parse_fixture(&wrong_id)
                .unwrap_err()
                .to_string()
                .contains("process_id")
        );

        let wrong_pdg = mutate_first_candidate_point(
            canonical_numerical_relation_evidence(),
            "points",
            json!([[{
                "pdg": -1,
                "momentum": ["1", "0", "0", "0"],
            }]]),
        );
        assert!(
            parse_fixture(&wrong_pdg)
                .unwrap_err()
                .to_string()
                .contains("authenticated recurrence process")
        );

        let wrong_seed =
            mutate_first_candidate_point(canonical_numerical_relation_evidence(), "seed", json!(0));
        assert!(
            parse_fixture(&wrong_seed)
                .unwrap_err()
                .to_string()
                .contains("validation-point seed is stale")
        );
    }

    #[test]
    fn numerical_relation_evidence_accepts_canonical_binary64_tolerance_boundaries() {
        for (relative, absolute) in [
            (1.0e-8, 1.0e-9),
            (f64::from_bits(1), 0.0),
            (f64::MIN_POSITIVE, f64::from_bits(1)),
            (0.25, 0.0),
        ] {
            let evidence =
                canonical_numerical_relation_evidence_with_tolerances(relative, absolute);
            let encoded = canonical_json_bytes(&evidence, "boundary tolerance evidence").unwrap();
            let parsed = parse_fixture(&encoded).unwrap();
            assert_eq!(parsed.relative_tolerance.to_bits(), relative.to_bits());
            assert_eq!(parsed.absolute_tolerance.to_bits(), absolute.to_bits());
        }
        let (maximum, exact) =
            parse_nonnegative_python_f64_hex(&python_f64_hex(f64::MAX), "maximum").unwrap();
        assert_eq!(maximum, f64::MAX);
        assert_eq!(exact, exact_binary64_rational(f64::MAX).unwrap());
    }

    #[test]
    fn numerical_relation_evidence_rejects_nonfinite_negative_zero_and_noncanonical_tolerances() {
        for invalid_tolerance in [
            "nan",
            "inf",
            "-0x0.0p+0",
            "0x1.0p-3",
            "0X1.0000000000000P-3",
        ] {
            let mut evidence = canonical_numerical_relation_evidence();
            evidence["relative_tolerance_binary64"] = json!(invalid_tolerance);
            let encoded = canonical_json_bytes(&evidence, "invalid tolerance evidence").unwrap();
            assert!(
                parse_fixture(&encoded)
                    .unwrap_err()
                    .to_string()
                    .contains("canonical nonnegative finite binary64")
            );
        }
        let mut zero = canonical_numerical_relation_evidence();
        zero["relative_tolerance_binary64"] = json!("0x0.0p+0");
        zero["absolute_tolerance_binary64"] = json!("0x0.0p+0");
        let encoded = canonical_json_bytes(&zero, "zero tolerance evidence").unwrap();
        assert!(
            parse_fixture(&encoded)
                .unwrap_err()
                .to_string()
                .contains("cannot both be zero")
        );
    }

    #[test]
    fn observation_locator_builds_canonical_metadata_and_authenticated_offsets() {
        let evidence = canonical_numerical_relation_evidence();
        let encoded = canonical_json_bytes(&evidence, "observation locator fixture").unwrap();
        let located = locate_raw_observation_arrays(&encoded).unwrap();
        let (metadata, source) = scanner_source(&located.metadata_bytes);
        validate_canonical_json_bytes(
            &metadata,
            &located.metadata_bytes,
            "observation locator metadata",
        )
        .unwrap();
        assert_eq!(metadata["candidate_capture"]["observations"], json!([]));
        assert_eq!(metadata["verification_capture"]["observations"], json!([]));
        let candidate =
            scan_raw_observation_array(&encoded, located.candidate, &source, 4, "candidate")
                .unwrap();
        let verification =
            scan_raw_observation_array(&encoded, located.verification, &source, 4, "verification")
                .unwrap();
        assert_eq!(candidate.len(), source.currents.len());
        assert_eq!(verification.len(), source.currents.len());
        assert!(candidate.iter().all(|row| row.row.start < row.row.end));
        assert!(
            candidate
                .iter()
                .all(|row| row.row.start <= row.values.start && row.values.end <= row.row.end)
        );
    }

    #[test]
    fn observation_scanner_rejects_truncated_reordered_extra_and_wrong_width_rows() {
        let evidence = canonical_numerical_relation_evidence();
        let encoded = canonical_json_bytes(&evidence, "observation scanner fixture").unwrap();
        let located = locate_raw_observation_arrays(&encoded).unwrap();
        let (_metadata, source) = scanner_source(&located.metadata_bytes);
        let candidate_range = located
            .candidate
            .as_usize(encoded.len(), "candidate fixture")
            .unwrap();
        let candidate = &encoded[candidate_range];

        let truncated = &candidate[..candidate.len() - 1];
        let truncated_range =
            RawByteRange::from_usize(0, truncated.len(), truncated.len(), "truncated").unwrap();
        assert!(
            scan_raw_observation_array(truncated, truncated_range, &source, 4, "truncated",)
                .unwrap_err()
                .to_string()
                .contains("malformed or not canonical")
        );

        let text = std::str::from_utf8(candidate).unwrap();
        let reordered = text.replacen(
            "{\"current_id\":0,\"dimension\":1,",
            "{\"dimension\":1,\"current_id\":0,",
            1,
        );
        assert_ne!(reordered, text);
        let reordered_range =
            RawByteRange::from_usize(0, reordered.len(), reordered.len(), "reordered").unwrap();
        assert!(
            scan_raw_observation_array(
                reordered.as_bytes(),
                reordered_range,
                &source,
                4,
                "reordered",
            )
            .unwrap_err()
            .to_string()
            .contains("malformed or not canonical")
        );

        let mut extra = candidate[..candidate.len() - 1].to_vec();
        extra.extend_from_slice(b",{}]");
        let extra_range = RawByteRange::from_usize(0, extra.len(), extra.len(), "extra").unwrap();
        assert!(
            scan_raw_observation_array(&extra, extra_range, &source, 4, "extra")
                .unwrap_err()
                .to_string()
                .contains("malformed or not canonical")
        );

        let mut wrong_width = evidence["candidate_capture"]["observations"].clone();
        wrong_width[0]["values"].as_array_mut().unwrap().pop();
        let wrong_width =
            canonical_json_bytes(&wrong_width, "wrong-width observation fixture").unwrap();
        let wrong_width_range =
            RawByteRange::from_usize(0, wrong_width.len(), wrong_width.len(), "wrong width")
                .unwrap();
        assert!(
            scan_raw_observation_array(&wrong_width, wrong_width_range, &source, 4, "wrong width",)
                .unwrap_err()
                .to_string()
                .contains("malformed or not canonical")
        );

        let mut wide_scalar = evidence["candidate_capture"]["observations"].clone();
        wide_scalar[0]["values"][0][0] = json!("1".repeat(16_385));
        let wide_scalar =
            canonical_json_bytes(&wide_scalar, "wide-scalar observation fixture").unwrap();
        let wide_scalar_range =
            RawByteRange::from_usize(0, wide_scalar.len(), wide_scalar.len(), "wide scalar")
                .unwrap();
        scan_raw_observation_array(&wide_scalar, wide_scalar_range, &source, 4, "wide scalar")
            .unwrap();

        if usize::BITS > 32 {
            let offset = u32::MAX as usize + 1;
            assert_eq!(
                RawByteRange::from_usize(offset, offset, offset, "large offset")
                    .unwrap()
                    .as_usize(offset, "large offset")
                    .unwrap(),
                offset..offset,
            );
        }
    }

    #[test]
    fn raw_candidate_index_is_complete_at_equal_and_opposite_boundaries() {
        fn rational(numerator: i128, denominator: i128) -> BigRational {
            BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
        }
        let mut entries = vec![
            (rational(100, 9), 0),
            (rational(-100, 9), 1),
            (BigRational::from_integer(BigInt::from(1_000)), 2),
            (BigRational::from_integer(BigInt::from(10)), 4),
        ];
        entries.sort();
        let index = RawNumericalCandidateIndex {
            observation_index: 0,
            scalar_component: 0,
            entries,
            selected_values: Vec::new(),
        };
        let current = (
            BigRational::from_integer(BigInt::from(10)),
            BigRational::zero(),
        );
        let relative = rational(1, 10);
        let absolute = BigRational::zero();
        let boundary_residual = rational(10, 9);
        assert_eq!((&current.0 - rational(100, 9)).abs(), boundary_residual,);
        assert_eq!(&relative * rational(100, 9).abs(), boundary_residual,);
        assert_eq!(
            raw_numerical_tolerance_window_ids(&index, &current, 4, "equal", &relative, &absolute,)
                .unwrap(),
            BTreeSet::from([0]),
        );
        assert_eq!(
            raw_numerical_tolerance_window_ids(
                &index, &current, 4, "opposite", &relative, &absolute,
            )
            .unwrap(),
            BTreeSet::from([1]),
        );
    }

    #[test]
    fn numerical_relation_options_reject_stale_probe_and_counter_contracts() {
        let evidence = canonical_numerical_relation_evidence();
        let encoded = canonical_json_bytes(&evidence, "test numerical evidence").unwrap();
        let parsed = parse_fixture(&encoded).unwrap();
        let options = RecurrenceRelationDiscoveryOptions::new(
            RecurrenceRelationDiscoveryMode::CertifiedReuse,
            96,
            4,
            4,
            1.0e-70,
            1.0e-80,
            0x5059_414d,
            "lc",
        )
        .unwrap();
        options
            .clone()
            .with_numerical_evidence(parsed.clone())
            .unwrap();

        let mut stale_probe = parsed.clone();
        stale_probe.probe_count = 5;
        assert!(
            options
                .clone()
                .with_numerical_evidence(stale_probe)
                .unwrap_err()
                .to_string()
                .contains("probe contract")
        );

        let mut stale_counters = parsed;
        stale_counters.verification_rejected_count = 1;
        assert!(
            options
                .with_numerical_evidence(stale_counters)
                .unwrap_err()
                .to_string()
                .contains("candidate counters")
        );
    }

    #[test]
    fn raw_numerical_replay_rejects_context_nonfinite_and_decision_tampering() {
        let evidence = canonical_numerical_relation_evidence();

        let mut selector_tamper = evidence.clone();
        selector_tamper["candidate_capture"]["selector_contexts"][0]["fixed_source_schedule"] =
            json!(false);
        let error =
            parse_fixture(&canonical_json_bytes(&selector_tamper, "selector tamper").unwrap())
                .unwrap_err()
                .to_string();
        assert!(error.contains("selector context"));

        let mut nonfinite = evidence.clone();
        nonfinite["candidate_capture"]["observations"][1]["values"][0][0] = json!("NaN");
        let error =
            parse_fixture(&canonical_json_bytes(&nonfinite, "nonfinite raw observation").unwrap())
                .unwrap_err()
                .to_string();
        assert!(error.contains("canonical finite decimal"));

        let mut stale_decision = evidence;
        stale_decision["decision_sha256"] = json!(digest(99));
        let error =
            parse_fixture(&canonical_json_bytes(&stale_decision, "stale decision digest").unwrap())
                .unwrap_err()
                .to_string();
        assert!(error.contains("decision digest"));

        let mut stale_rejection = canonical_numerical_relation_evidence();
        stale_rejection["rejection_decision_sha256"] = json!(digest(98));
        let error = parse_fixture(
            &canonical_json_bytes(&stale_rejection, "stale rejection digest").unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("rejection digest"));
    }

    #[test]
    fn raw_numerical_tolerance_boundary_and_parameter_domains_are_exact() {
        let defaults = vec![AuthenticatedRuntimeParameterSlot {
            default: BigRational::new(BigInt::from(3), BigInt::from(2)),
            probe_policy: "native-template-default-perturbed-v1",
        }];
        let candidate = deterministic_parameter_contexts(
            &defaults,
            4,
            17,
            "candidate-current-parameter-probes-v1",
            true,
        )
        .unwrap();
        let verification = deterministic_parameter_contexts(
            &defaults,
            4,
            17,
            "independent-verification-current-parameter-probes-v1",
            false,
        )
        .unwrap();
        assert_eq!(candidate[0], vec![defaults[0].default.clone()]);
        assert!(
            candidate
                .iter()
                .all(|row| !verification.iter().any(|other| row == other))
        );

        let source = RawSourceSemantics {
            value: json!({}),
            process_id: "residual-test".to_owned(),
            physical_pdgs: vec![1],
            strategy: "contracted-color-union".to_owned(),
            selector_schedule: json!({}),
            currents: (0..2)
                .map(|current_id| RawSourceCurrent {
                    current_id,
                    is_source: false,
                    contract_key: vec![1],
                    dimension: 1,
                    selector_domain_id: 0,
                })
                .collect(),
        };
        let boundary_bytes = canonical_json_bytes(
            &json!([
                {"current_id": 0, "dimension": 1, "values": [["1", "0"]]},
                {"current_id": 1, "dimension": 1, "values": [["2", "0"]]},
            ]),
            "boundary observation capture",
        )
        .unwrap();
        let boundary_capture = capture_from_observation_array(&boundary_bytes, &source, 1);
        let mut telemetry = RecurrenceEvidenceAuthenticationTelemetry::default();
        let boundary = relation_residuals(
            "equal",
            1,
            Some(0),
            &boundary_capture,
            &BigRational::zero(),
            &BigRational::one(),
            &mut telemetry,
        )
        .unwrap();
        assert_eq!(boundary.maximum_ratio, BigRational::one());
        let outside_bytes = canonical_json_bytes(
            &json!([
                {"current_id": 0, "dimension": 1, "values": [["1", "0"]]},
                {"current_id": 1, "dimension": 1, "values": [["2.000001", "0"]]},
            ]),
            "outside observation capture",
        )
        .unwrap();
        let outside_capture = capture_from_observation_array(&outside_bytes, &source, 1);
        let outside = relation_residuals(
            "equal",
            1,
            Some(0),
            &outside_capture,
            &BigRational::zero(),
            &BigRational::one(),
            &mut telemetry,
        )
        .unwrap();
        assert!(outside.maximum_ratio > BigRational::one());
    }

    #[test]
    fn authenticated_parameter_defaults_reject_shifted_neighborhoods() {
        fn contract(default: i64) -> AuthenticatedRuntimeParameterContract {
            let default = BigRational::from_integer(BigInt::from(default));
            let schema = json!({
                "abi": "pyamplicol-recurrence-runtime-parameter-schema-v2",
                "parameters": [{
                    "runtime_slot": 0,
                    "runtime_name": "mass",
                    "parameter_template_id": 0,
                    "prepared_parameter_id": 0,
                    "component": 0,
                    "default_binary64": python_f64_hex_signed(default.to_f64().unwrap()),
                    "probe_policy": "native-template-default-perturbed-v1",
                }],
            });
            AuthenticatedRuntimeParameterContract {
                schema_sha256: canonical_json_sha256(&schema, "test runtime parameter schema")
                    .unwrap(),
                schema,
                slots: vec![AuthenticatedRuntimeParameterSlot {
                    default,
                    probe_policy: "native-template-default-perturbed-v1",
                }],
                process_id: "parameter-test".to_owned(),
                physical_pdgs: vec![1],
            }
        }

        let authenticated = contract(3);
        let shifted = contract(5);
        assert!(
            validate_raw_runtime_parameter_schema(
                &shifted.schema,
                &shifted.schema_sha256,
                &authenticated,
            )
            .unwrap_err()
            .to_string()
            .contains("authenticated native template input")
        );
    }

    #[test]
    fn native_parameter_default_hex_preserves_negative_binary64_signs() {
        assert_eq!(python_f64_hex_signed(-1.5), "-0x1.8000000000000p+0");
        assert_eq!(
            python_f64_hex_signed(f64::from_bits(0x8000_0000_0000_0001)),
            "-0x0.0000000000001p-1022"
        );
        assert_eq!(python_f64_hex_signed(0.0), "0x0.0p+0");
    }

    #[test]
    fn derived_parameter_probe_slots_are_fixed_and_tampering_fails() {
        let contract = AuthenticatedRuntimeParameterContract {
            schema: json!({}),
            schema_sha256: digest(90),
            slots: vec![AuthenticatedRuntimeParameterSlot {
                default: BigRational::zero(),
                probe_policy: "derived-overwritten-fixed-zero-v1",
            }],
            process_id: "parameter-test".to_owned(),
            physical_pdgs: vec![1],
        };
        let candidate_contexts = deterministic_parameter_contexts(
            &contract.slots,
            4,
            17,
            "candidate-current-parameter-probes-v1",
            true,
        )
        .unwrap();
        let verification_contexts = deterministic_parameter_contexts(
            &contract.slots,
            4,
            17,
            "independent-verification-current-parameter-probes-v1",
            false,
        )
        .unwrap();
        assert!(
            candidate_contexts
                .iter()
                .chain(&verification_contexts)
                .all(|row| row == &[BigRational::zero()])
        );
        let capture = |parameter_contexts, base: u8| RawNumericalCapture {
            raw_bytes: b"[]",
            observation_rows: Vec::new(),
            point_count: 4,
            point_sha256s: (0..4).map(|index| digest(base + index)).collect(),
            kinematic_sha256s: (0..4).map(|index| digest(base + 10 + index)).collect(),
            parameter_context_sha256s: (0..4).map(|index| digest(base + 20 + index)).collect(),
            parameter_contexts,
            dimensions: Vec::new(),
            observation_batch_sha256: digest(base + 30),
            capture_contract_sha256: digest(base + 31),
        };
        let mut candidate = capture(candidate_contexts, 1);
        let verification = capture(verification_contexts, 101);
        validate_independent_raw_captures(&candidate, &verification, &contract, 17).unwrap();
        candidate.parameter_contexts[1][0] = BigRational::one();
        assert!(
            validate_independent_raw_captures(&candidate, &verification, &contract, 17)
                .unwrap_err()
                .to_string()
                .contains("stale or non-deterministic")
        );
    }

    #[test]
    fn parameter_rows_are_included_in_raw_geometry() {
        let source = RawSourceSemantics {
            value: json!({}),
            process_id: "memory-preflight-test".to_owned(),
            physical_pdgs: vec![1],
            strategy: "contracted-color-union".to_owned(),
            selector_schedule: json!({}),
            currents: Vec::new(),
        };
        assert_eq!(
            raw_evidence_geometry_counts(&source, 2, 2, 400_000).unwrap(),
            (0, 0, 1_600_000, 400_000)
        );
    }

    fn compressed_envelope(raw: &[u8]) -> Vec<u8> {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(raw).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut envelope = Vec::with_capacity(COMPRESSED_EVIDENCE_HEADER_BYTES + compressed.len());
        envelope.extend_from_slice(COMPRESSED_EVIDENCE_MAGIC);
        envelope.extend_from_slice(&(raw.len() as u64).to_be_bytes());
        envelope.extend_from_slice(&Sha256::digest(raw));
        envelope.extend_from_slice(&compressed);
        envelope
    }

    #[test]
    fn compressed_transport_checks_declared_length_and_digest_before_json_parsing() {
        let raw = br#"{"candidate_capture":{"observations":[]},"verification_capture":{"observations":[]}}"#;
        let encoded = compressed_envelope(raw);
        assert_eq!(decode_compressed_evidence(&encoded).unwrap(), raw);

        let mut digest_tamper = encoded.clone();
        digest_tamper[16] ^= 1;
        assert!(
            decode_compressed_evidence(&digest_tamper)
                .unwrap_err()
                .to_string()
                .contains("digest mismatch")
        );

        let mut length_tamper = encoded.clone();
        length_tamper[8..16].copy_from_slice(&((raw.len() - 1) as u64).to_be_bytes());
        assert!(
            decode_compressed_evidence(&length_tamper)
                .unwrap_err()
                .to_string()
                .contains("decompressed length")
        );

        let mut trailing = encoded;
        trailing.push(0);
        assert!(
            decode_compressed_evidence(&trailing)
                .unwrap_err()
                .to_string()
                .contains("trailing or unconsumed")
        );
    }
}
