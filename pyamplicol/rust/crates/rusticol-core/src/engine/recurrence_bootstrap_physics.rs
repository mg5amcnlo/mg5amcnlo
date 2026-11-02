// SPDX-License-Identifier: 0BSD

//! Compact native image of recurrence process physics.
//!
//! This is an internal cold-load cache, not a stable interchange format.  The
//! enclosing PACBIN member authenticates its bytes; this codec supplies an
//! independently recognizable ABI, file-length checks, exact consumption, and
//! the same semantic validation as the canonical JSON path.

use crate::{
    ColorAccuracy, ColorComponent, Coverage, ExternalParticle, Helicity, ModelParameter,
    ProcessPhysics as ProcessPhysicsV1, Reduction, RusticolError, RusticolResult,
    SelectorCapabilities,
};
use bincode::Decode;
#[cfg(any(feature = "python-generation-bridge", test))]
use bincode::Encode;
use bincode::de::{Decoder, read::Reader};
use bincode::error::{AllowedEnumVariants, DecodeError};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};
use std::collections::BTreeMap;

/// Native, internal serialization ABI for one already-resolved physics image.
/// Artifacts carrying this image must be regenerated when this wire domain
/// changes.
pub(super) const RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI: &str =
    "pyamplicol-recurrence-bootstrap-physics-v1";

const RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_MAGIC: &[u8; 8] = b"PACRPHM1";
const RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_VERSION: u16 = 1;
const RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_HEADER_BYTES: usize = 16;

// serde_json's ordinary parser rejects nesting beyond 128 levels.  Apply the
// same bound to the native representation so a recursive Decode cannot be used
// to exhaust the stack.
const COMPACT_JSON_MAX_DEPTH: usize = 128;

#[derive(Clone, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
#[bincode(decode_context = "CompactJsonDecodeContext")]
struct RecurrenceBootstrapPhysicsV1 {
    schema_version: u32,
    kind: String,
    process_id: String,
    process: String,
    color_accuracy: ColorAccuracy,
    coverage: Coverage,
    external_particles: Vec<ExternalParticle>,
    helicities: Vec<Helicity>,
    color_components: Vec<ColorComponent>,
    reduction: Reduction,
    model_parameters: Vec<ModelParameter>,
    selectors: SelectorCapabilities,
    extensions: CompactJsonObject,
}

impl RecurrenceBootstrapPhysicsV1 {
    #[cfg(any(feature = "python-generation-bridge", test))]
    fn from_physics(physics: &ProcessPhysicsV1) -> RusticolResult<Self> {
        Ok(Self {
            schema_version: physics.schema_version,
            kind: physics.kind.clone(),
            process_id: physics.process_id.clone(),
            process: physics.process.clone(),
            color_accuracy: physics.color_accuracy,
            coverage: physics.coverage.clone(),
            external_particles: physics.external_particles.clone(),
            helicities: physics.helicities.clone(),
            color_components: physics.color_components.clone(),
            reduction: physics.reduction.clone(),
            model_parameters: physics.model_parameters.clone(),
            selectors: physics.selectors.clone(),
            extensions: CompactJsonObject::from_extensions(&physics.extensions)?,
        })
    }

    fn into_physics(self) -> RusticolResult<ProcessPhysicsV1> {
        Ok(ProcessPhysicsV1 {
            schema_version: self.schema_version,
            kind: self.kind,
            process_id: self.process_id,
            process: self.process,
            color_accuracy: self.color_accuracy,
            coverage: self.coverage,
            external_particles: self.external_particles,
            helicities: self.helicities,
            color_components: self.color_components,
            reduction: self.reduction,
            model_parameters: self.model_parameters,
            selectors: self.selectors,
            extensions: self.extensions.into_extensions()?,
        })
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
struct CompactJsonObject(Vec<(String, CompactJsonValue)>);

impl CompactJsonObject {
    #[cfg(any(feature = "python-generation-bridge", test))]
    fn from_extensions(extensions: &BTreeMap<String, JsonValue>) -> RusticolResult<Self> {
        Self::from_entries(extensions.iter(), 0)
    }

    #[cfg(any(feature = "python-generation-bridge", test))]
    fn from_json_map(values: &JsonMap<String, JsonValue>, depth: usize) -> RusticolResult<Self> {
        Self::from_entries(values.iter(), depth)
    }

    #[cfg(any(feature = "python-generation-bridge", test))]
    fn from_entries<'a>(
        entries: impl ExactSizeIterator<Item = (&'a String, &'a JsonValue)>,
        depth: usize,
    ) -> RusticolResult<Self> {
        let mut compact = Vec::new();
        compact.try_reserve_exact(entries.len()).map_err(|error| {
            compact_json_bound_error(format!("could not allocate compact JSON object: {error}"))
        })?;
        for (key, value) in entries {
            compact.push((key.clone(), CompactJsonValue::from_json(value, depth)?));
        }
        compact.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        Ok(Self(compact))
    }

    fn into_extensions(self) -> RusticolResult<BTreeMap<String, JsonValue>> {
        let mut extensions = BTreeMap::new();
        for (key, value) in self.0 {
            if extensions.insert(key, value.into_json()?).is_some() {
                return Err(RusticolError::integrity(
                    "compact recurrence physics extensions contain a duplicate key",
                ));
            }
        }
        Ok(extensions)
    }

    fn into_json_map(self) -> RusticolResult<JsonMap<String, JsonValue>> {
        let mut values = JsonMap::new();
        for (key, value) in self.0 {
            if values.insert(key, value.into_json()?).is_some() {
                return Err(RusticolError::integrity(
                    "compact recurrence physics JSON object contains a duplicate key",
                ));
            }
        }
        Ok(values)
    }
}

impl Decode<CompactJsonDecodeContext> for CompactJsonObject {
    fn decode<D: Decoder<Context = CompactJsonDecodeContext>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        let len = decode_compact_json_length(decoder, "object")?;
        decoder.claim_container_read::<(String, CompactJsonValue)>(len)?;
        let mut entries = Vec::new();
        entries.try_reserve_exact(len).map_err(|error| {
            DecodeError::OtherString(format!("could not allocate compact JSON object: {error}"))
        })?;
        for _ in 0..len {
            decoder.unclaim_bytes_read(std::mem::size_of::<(String, CompactJsonValue)>());
            let key = decode_compact_json_string(decoder, "object key")?;
            if entries
                .last()
                .is_some_and(|(previous, _): &(String, CompactJsonValue)| previous >= &key)
            {
                return Err(DecodeError::Other(
                    "compact JSON object keys must be unique and strictly ordered",
                ));
            }
            let value = CompactJsonValue::decode(decoder)?;
            entries.push((key, value));
        }
        Ok(Self(entries))
    }
}

bincode::impl_borrow_decode_with_context!(CompactJsonObject, CompactJsonDecodeContext);

#[derive(Clone, Debug)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
enum CompactJsonValue {
    Null,
    Bool(bool),
    U64(u64),
    I64(i64),
    F64Bits(u64),
    String(String),
    Array(Vec<Self>),
    Object(CompactJsonObject),
}

impl CompactJsonValue {
    #[cfg(any(feature = "python-generation-bridge", test))]
    fn from_json(value: &JsonValue, depth: usize) -> RusticolResult<Self> {
        if depth > COMPACT_JSON_MAX_DEPTH {
            return Err(compact_json_bound_error(format!(
                "nesting exceeds the {COMPACT_JSON_MAX_DEPTH}-level limit"
            )));
        }
        match value {
            JsonValue::Null => Ok(Self::Null),
            JsonValue::Bool(value) => Ok(Self::Bool(*value)),
            JsonValue::Number(value) if value.is_u64() => Ok(Self::U64(
                value.as_u64().expect("u64 classification checked"),
            )),
            JsonValue::Number(value) if value.is_i64() => Ok(Self::I64(
                value.as_i64().expect("i64 classification checked"),
            )),
            JsonValue::Number(value) => {
                let value = value.as_f64().ok_or_else(|| {
                    RusticolError::artifact(
                        "compact recurrence physics extension has an unsupported JSON number",
                    )
                })?;
                if !value.is_finite() {
                    return Err(RusticolError::artifact(
                        "compact recurrence physics extension has a non-finite JSON number",
                    ));
                }
                Ok(Self::F64Bits(value.to_bits()))
            }
            JsonValue::String(value) => Ok(Self::String(value.clone())),
            JsonValue::Array(values) => {
                let mut compact = Vec::new();
                compact.try_reserve_exact(values.len()).map_err(|error| {
                    compact_json_bound_error(format!(
                        "could not allocate compact JSON array: {error}"
                    ))
                })?;
                for value in values {
                    compact.push(Self::from_json(value, depth + 1)?);
                }
                Ok(Self::Array(compact))
            }
            JsonValue::Object(values) => Ok(Self::Object(CompactJsonObject::from_json_map(
                values,
                depth + 1,
            )?)),
        }
    }

    fn into_json(self) -> RusticolResult<JsonValue> {
        match self {
            Self::Null => Ok(JsonValue::Null),
            Self::Bool(value) => Ok(JsonValue::Bool(value)),
            Self::U64(value) => Ok(JsonValue::Number(JsonNumber::from(value))),
            Self::I64(value) => Ok(JsonValue::Number(JsonNumber::from(value))),
            Self::F64Bits(bits) => {
                let value = f64::from_bits(bits);
                let number = JsonNumber::from_f64(value).ok_or_else(|| {
                    RusticolError::integrity(
                        "compact recurrence physics extension contains a non-finite JSON number",
                    )
                })?;
                Ok(JsonValue::Number(number))
            }
            Self::String(value) => Ok(JsonValue::String(value)),
            Self::Array(values) => values
                .into_iter()
                .map(Self::into_json)
                .collect::<RusticolResult<Vec<_>>>()
                .map(JsonValue::Array),
            Self::Object(values) => values.into_json_map().map(JsonValue::Object),
        }
    }
}

impl Decode<CompactJsonDecodeContext> for CompactJsonValue {
    fn decode<D: Decoder<Context = CompactJsonDecodeContext>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        {
            let context = decoder.context();
            if context.depth > COMPACT_JSON_MAX_DEPTH {
                return Err(DecodeError::Other(
                    "compact JSON nesting exceeds its depth limit",
                ));
            }
            if context.nodes_remaining == 0 {
                return Err(DecodeError::Other(
                    "compact JSON node count cannot fit in the encoded payload",
                ));
            }
            context.nodes_remaining -= 1;
        }

        let variant = <u32 as Decode<CompactJsonDecodeContext>>::decode(decoder)?;
        match variant {
            0 => Ok(Self::Null),
            1 => bool::decode(decoder).map(Self::Bool),
            2 => u64::decode(decoder).map(Self::U64),
            3 => i64::decode(decoder).map(Self::I64),
            4 => {
                let bits = u64::decode(decoder)?;
                if !f64::from_bits(bits).is_finite() {
                    return Err(DecodeError::Other("compact JSON number must be finite"));
                }
                Ok(Self::F64Bits(bits))
            }
            5 => decode_compact_json_string(decoder, "string").map(Self::String),
            6 => decode_compact_json_children(decoder).map(Self::Array),
            7 => decode_compact_json_object(decoder).map(Self::Object),
            found => Err(DecodeError::UnexpectedVariant {
                type_name: "CompactJsonValue",
                allowed: &COMPACT_JSON_ALLOWED_VARIANTS,
                found,
            }),
        }
    }
}

bincode::impl_borrow_decode_with_context!(CompactJsonValue, CompactJsonDecodeContext);

static COMPACT_JSON_ALLOWED_VARIANTS: AllowedEnumVariants =
    AllowedEnumVariants::Range { min: 0, max: 7 };

#[derive(Clone, Copy, Debug)]
struct CompactJsonDecodeContext {
    depth: usize,
    payload_bytes: usize,
    nodes_remaining: usize,
}

impl CompactJsonDecodeContext {
    fn new(payload_bytes: usize) -> Self {
        Self {
            depth: 0,
            payload_bytes,
            // Each node needs at least one encoded variant byte.
            nodes_remaining: payload_bytes,
        }
    }
}

fn decode_compact_json_children<D: Decoder<Context = CompactJsonDecodeContext>>(
    decoder: &mut D,
) -> Result<Vec<CompactJsonValue>, DecodeError> {
    let len = decode_compact_json_length(decoder, "array")?;
    decoder.claim_container_read::<CompactJsonValue>(len)?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        DecodeError::OtherString(format!("could not allocate compact JSON array: {error}"))
    })?;
    let previous_depth = decoder.context().depth;
    decoder.context().depth = previous_depth + 1;
    let result = (|| {
        for _ in 0..len {
            decoder.unclaim_bytes_read(std::mem::size_of::<CompactJsonValue>());
            values.push(CompactJsonValue::decode(decoder)?);
        }
        Ok(values)
    })();
    decoder.context().depth = previous_depth;
    result
}

fn decode_compact_json_object<D: Decoder<Context = CompactJsonDecodeContext>>(
    decoder: &mut D,
) -> Result<CompactJsonObject, DecodeError> {
    let previous_depth = decoder.context().depth;
    decoder.context().depth = previous_depth + 1;
    let result = CompactJsonObject::decode(decoder);
    decoder.context().depth = previous_depth;
    result
}

fn decode_compact_json_length<D: Decoder<Context = CompactJsonDecodeContext>>(
    decoder: &mut D,
    description: &str,
) -> Result<usize, DecodeError> {
    let wire_len = u64::decode(decoder)?;
    let len = usize::try_from(wire_len).map_err(|_| DecodeError::OutsideUsizeRange(wire_len))?;
    if len > decoder.context().payload_bytes {
        return Err(DecodeError::OtherString(format!(
            "compact JSON {description} count {len} cannot fit in the encoded payload"
        )));
    }
    Ok(len)
}

fn decode_compact_json_string<D: Decoder<Context = CompactJsonDecodeContext>>(
    decoder: &mut D,
    description: &str,
) -> Result<String, DecodeError> {
    let wire_len = u64::decode(decoder)?;
    let len = usize::try_from(wire_len).map_err(|_| DecodeError::OutsideUsizeRange(wire_len))?;
    if len > decoder.context().payload_bytes {
        return Err(DecodeError::OtherString(format!(
            "compact JSON {description} length {len} cannot fit in the encoded payload"
        )));
    }
    decoder.claim_bytes_read(len)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|error| {
        DecodeError::OtherString(format!(
            "could not allocate compact JSON {description}: {error}"
        ))
    })?;
    bytes.resize(len, 0);
    decoder.reader().read(&mut bytes)?;
    String::from_utf8(bytes).map_err(|error| DecodeError::Utf8 {
        inner: error.utf8_error(),
    })
}

#[cfg(any(feature = "python-generation-bridge", test))]
fn compact_json_bound_error(detail: impl std::fmt::Display) -> RusticolError {
    RusticolError::artifact(format!(
        "compact recurrence physics extension exceeds its bounds: {detail}"
    ))
}

/// Encode one typed physics payload for the native recurrence bootstrap.
#[cfg(any(feature = "python-generation-bridge", test))]
pub(super) fn encode_recurrence_bootstrap_physics_v1(
    physics: &ProcessPhysicsV1,
) -> RusticolResult<Vec<u8>> {
    let wire = RecurrenceBootstrapPhysicsV1::from_physics(physics)?;
    encode_recurrence_bootstrap_physics_wire_v1(&wire)
}

#[cfg(any(feature = "python-generation-bridge", test))]
fn encode_recurrence_bootstrap_physics_wire_v1(
    physics: &RecurrenceBootstrapPhysicsV1,
) -> RusticolResult<Vec<u8>> {
    let body = bincode::encode_to_vec(physics, bincode::config::standard()).map_err(|error| {
        RusticolError::serialization(format!(
            "could not encode {RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI}: {error}"
        ))
    })?;
    let body_len = u32::try_from(body.len()).map_err(|_| {
        RusticolError::artifact(format!(
            "{RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI} body length exceeds its u32 wire domain"
        ))
    })?;
    let encoded_len = RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_HEADER_BYTES
        .checked_add(body.len())
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "{RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI} encoded length overflows usize"
            ))
        })?;
    let mut encoded = Vec::new();
    encoded.try_reserve_exact(encoded_len).map_err(|error| {
        RusticolError::artifact(format!(
            "could not allocate {encoded_len} bytes for {RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI}: {error}"
        ))
    })?;
    encoded.extend_from_slice(RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_MAGIC);
    encoded.extend_from_slice(&RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_VERSION.to_le_bytes());
    encoded.extend_from_slice(&0_u16.to_le_bytes());
    encoded.extend_from_slice(&body_len.to_le_bytes());
    encoded.extend_from_slice(&body);
    Ok(encoded)
}

/// Decode and semantically validate one native recurrence physics
/// payload. Exact consumption is part of the ABI.
#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
pub(super) fn decode_recurrence_bootstrap_physics_v1(
    bytes: &[u8],
) -> RusticolResult<ProcessPhysicsV1> {
    let header = bytes
        .get(..RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_HEADER_BYTES)
        .ok_or_else(|| {
            RusticolError::serialization(format!(
                "truncated {RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI} header"
            ))
        })?;
    if header.get(..8) != Some(RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_MAGIC.as_slice()) {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence physics binary magic; expected {RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI}"
        )));
    }
    let version = u16::from_le_bytes(
        header[8..10]
            .try_into()
            .expect("fixed recurrence physics version field"),
    );
    if version != RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_VERSION {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence physics binary version {version}; expected {RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_VERSION}"
        )));
    }
    let reserved = u16::from_le_bytes(
        header[10..12]
            .try_into()
            .expect("fixed recurrence physics reserved field"),
    );
    if reserved != 0 {
        return Err(RusticolError::compatibility(
            "recurrence physics binary reserved flags must be zero",
        ));
    }
    let declared_body_len = usize::try_from(u32::from_le_bytes(
        header[12..16]
            .try_into()
            .expect("fixed recurrence physics length field"),
    ))
    .map_err(|_| {
        RusticolError::artifact(format!(
            "{RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI} body length exceeds usize"
        ))
    })?;
    let expected_len = RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_HEADER_BYTES
        .checked_add(declared_body_len)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "{RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI} declared length overflows usize"
            ))
        })?;
    if bytes.len() != expected_len {
        return Err(RusticolError::serialization(format!(
            "{RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI} payload length is {}, expected {expected_len}",
            bytes.len(),
        )));
    }

    let body = &bytes[RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_HEADER_BYTES..];
    let (wire, consumed): (RecurrenceBootstrapPhysicsV1, usize) =
        bincode::decode_from_slice_with_context(
            body,
            bincode::config::standard(),
            CompactJsonDecodeContext::new(body.len()),
        )
        .map_err(|error| {
            RusticolError::serialization(format!(
                "could not decode {RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI}: {error}"
            ))
        })?;
    if consumed != body.len() {
        return Err(RusticolError::serialization(format!(
            "{RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_ABI} contains {} trailing bytes",
            body.len() - consumed,
        )));
    }
    let physics = wire.into_physics()?;
    physics.validate()?;
    Ok(physics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ColorAccuracy, ColorComponent, ContractedColor, Coverage, ExternalParticle, Helicity,
        ModelParameter, ParameterKind, ParticleRole, RUNTIME_PHYSICS_SCHEMA_VERSION, Reduction,
        ReductionGroup, ReductionKind, SelectorCapabilities,
    };
    use serde_json::json;

    fn physics_fixture() -> ProcessPhysicsV1 {
        let mut extensions = BTreeMap::new();
        extensions.insert("array".to_owned(), json!([null, true, {"nested": [1, 2]}]));
        extensions.insert("float".to_owned(), json!(0.125));
        extensions.insert("i64".to_owned(), json!(i64::MIN));
        extensions.insert("negative_zero".to_owned(), json!(-0.0));
        extensions.insert("string".to_owned(), json!("extension text"));
        extensions.insert("u64".to_owned(), json!(u64::MAX));

        ProcessPhysicsV1 {
            schema_version: RUNTIME_PHYSICS_SCHEMA_VERSION,
            kind: "pyamplicol-resolved-physics".to_owned(),
            process_id: "p0".to_owned(),
            process: "a b > c".to_owned(),
            color_accuracy: ColorAccuracy::Full,
            coverage: Coverage {
                helicities: "complete".to_owned(),
                color: "contracted".to_owned(),
                color_kind: "contracted-color".to_owned(),
                structural_zero_helicity_count: 0,
            },
            external_particles: (0..3)
                .map(|index| ExternalParticle {
                    index,
                    label: index + 1,
                    particle: format!("particle-{index}"),
                    pdg: index as i32 + 1,
                    role: if index < 2 {
                        ParticleRole::Initial
                    } else {
                        ParticleRole::Final
                    },
                    momentum_slot: index,
                    momentum_components: [
                        "E".to_owned(),
                        "px".to_owned(),
                        "py".to_owned(),
                        "pz".to_owned(),
                    ],
                })
                .collect(),
            helicities: vec![Helicity {
                id: "helicity:0".to_owned(),
                index: 0,
                values: vec![1, -1, 1],
                computed: true,
                structural_zero: false,
                representative_id: "helicity:0".to_owned(),
                coefficient: 1.0,
            }],
            color_components: vec![ColorComponent::ContractedColor(ContractedColor {
                id: "contracted".to_owned(),
                index: 0,
                description: "coherent contracted color".to_owned(),
            })],
            reduction: Reduction {
                kind: ReductionKind::ContractedColor,
                groups: vec![ReductionGroup {
                    id: "group:0".to_owned(),
                    representative_helicity_id: "helicity:0".to_owned(),
                    representative_color_id: "contracted".to_owned(),
                    physical_helicity_ids: vec!["helicity:0".to_owned()],
                    physical_color_ids: vec!["contracted".to_owned()],
                }],
            },
            model_parameters: vec![ModelParameter {
                name: "coupling".to_owned(),
                kind: ParameterKind::Coupling,
                default_real: f64::from_bits(0x3fdd_8fdb_d004_403d),
                default_imaginary: -0.0,
                mutable: true,
            }],
            selectors: SelectorCapabilities {
                helicity: true,
                color_flow: false,
                contracted_color: false,
            },
            extensions,
        }
    }

    #[test]
    fn compact_recurrence_physics_round_trips_every_json_value_exactly() {
        let physics = physics_fixture();
        physics.validate().expect("valid physics fixture");
        let encoded = encode_recurrence_bootstrap_physics_v1(&physics)
            .expect("encode compact recurrence physics");
        assert_eq!(
            encoded.get(..8),
            Some(RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_MAGIC.as_slice())
        );

        let decoded = decode_recurrence_bootstrap_physics_v1(&encoded)
            .expect("decode compact recurrence physics");
        assert_eq!(decoded, physics);
        assert_eq!(
            decoded.extensions["negative_zero"]
                .as_f64()
                .expect("negative zero number")
                .to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            encode_recurrence_bootstrap_physics_v1(&decoded)
                .expect("deterministic compact recurrence physics re-encode"),
            encoded
        );
    }

    #[test]
    fn compact_recurrence_physics_rejects_format_extensions_and_oversize() {
        let encoded = encode_recurrence_bootstrap_physics_v1(&physics_fixture())
            .expect("encode compact recurrence physics");

        let mut future_version = encoded.clone();
        future_version[8..10]
            .copy_from_slice(&(RECURRENCE_BOOTSTRAP_PHYSICS_BINARY_VERSION + 1).to_le_bytes());
        assert!(
            decode_recurrence_bootstrap_physics_v1(&future_version)
                .unwrap_err()
                .message()
                .contains("version")
        );

        let mut future_flags = encoded.clone();
        future_flags[10..12].copy_from_slice(&1_u16.to_le_bytes());
        assert!(
            decode_recurrence_bootstrap_physics_v1(&future_flags)
                .unwrap_err()
                .message()
                .contains("reserved")
        );

        let mut trailing_body = encoded.clone();
        let body_len = u32::from_le_bytes(trailing_body[12..16].try_into().unwrap());
        trailing_body[12..16].copy_from_slice(&(body_len + 1).to_le_bytes());
        trailing_body.push(0);
        assert!(
            decode_recurrence_bootstrap_physics_v1(&trailing_body)
                .unwrap_err()
                .message()
                .contains("trailing")
        );

        let mut oversized = encoded;
        oversized[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            decode_recurrence_bootstrap_physics_v1(&oversized)
                .unwrap_err()
                .message()
                .contains("payload length")
        );
    }

    #[test]
    fn compact_recurrence_physics_rejects_noncanonical_and_deep_extensions() {
        let mut wire = RecurrenceBootstrapPhysicsV1::from_physics(&physics_fixture())
            .expect("construct compact recurrence physics");
        wire.extensions = CompactJsonObject(vec![
            ("z".to_owned(), CompactJsonValue::Null),
            ("a".to_owned(), CompactJsonValue::Null),
        ]);
        let encoded = encode_recurrence_bootstrap_physics_wire_v1(&wire)
            .expect("encode noncanonical compact object");
        assert!(
            decode_recurrence_bootstrap_physics_v1(&encoded)
                .unwrap_err()
                .message()
                .contains("strictly ordered")
        );

        let mut value = CompactJsonValue::Null;
        for _ in 0..=COMPACT_JSON_MAX_DEPTH {
            value = CompactJsonValue::Array(vec![value]);
        }
        wire.extensions = CompactJsonObject(vec![("deep".to_owned(), value)]);
        let encoded = encode_recurrence_bootstrap_physics_wire_v1(&wire)
            .expect("encode over-deep compact extension");
        assert!(
            decode_recurrence_bootstrap_physics_v1(&encoded)
                .unwrap_err()
                .message()
                .contains("depth")
        );
    }

    #[test]
    fn compact_recurrence_physics_revalidates_typed_semantics_and_numbers() {
        let mut wire = RecurrenceBootstrapPhysicsV1::from_physics(&physics_fixture())
            .expect("construct compact recurrence physics");
        wire.process.clear();
        let encoded = encode_recurrence_bootstrap_physics_wire_v1(&wire)
            .expect("encode invalid compact recurrence physics");
        assert!(
            decode_recurrence_bootstrap_physics_v1(&encoded)
                .unwrap_err()
                .message()
                .contains("invalid process")
        );

        wire.process = "a b > c".to_owned();
        wire.extensions = CompactJsonObject(vec![(
            "nan".to_owned(),
            CompactJsonValue::F64Bits(f64::NAN.to_bits()),
        )]);
        let encoded = encode_recurrence_bootstrap_physics_wire_v1(&wire)
            .expect("encode non-finite compact JSON number");
        assert!(
            decode_recurrence_bootstrap_physics_v1(&encoded)
                .unwrap_err()
                .message()
                .contains("finite")
        );
    }
}
