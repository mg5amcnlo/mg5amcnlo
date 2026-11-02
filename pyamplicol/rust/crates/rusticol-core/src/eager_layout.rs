// SPDX-License-Identifier: 0BSD

//! Fixed-width wire contracts for compact eager runtime artifacts.
//!
//! This module defines only the versioned storage foundation. Eager lowering
//! and runtime loading are implemented separately and consume these validated
//! little-endian sections.

use crate::{RusticolError, RusticolResult};

pub const EAGER_LOWERING_INPUT_ABI: &str = "pyamplicol-eager-lowering-input-v1";
pub const EAGER_PLAN_ABI: &str = "pyamplicol-eager-plan-v3";
pub const EAGER_RUNTIME_LAYOUT_ABI: &str = "pyamplicol-eager-runtime-layout-v1";
pub const EAGER_RUNTIME_CAPABILITY: &str = "rusticol.eager-runtime-layout.complex-f64.v1";
pub const EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY: &str = "eager-direct-arena-v1";
pub const EAGER_DIRECT_SOURCE_APPLICATION_ABI: &str = "pyamplicol-symjit-plane-application-v2";
pub const EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI: &str =
    "pyamplicol-eager-native-direct-table-v1";
pub const EAGER_DIRECT_TABLE_DESCRIPTOR_ABI: &str = "pyamplicol-eager-plane-table-descriptor-v1";
pub const EAGER_DIRECT_TABLE_BINDING_ABI: &str = "pyamplicol-eager-plane-table-binding-v2";
pub const EAGER_RUNTIME_CONTAINER_KIND: &str = "pyamplicol-eager-runtime-container";
pub const EAGER_RUNTIME_CONTAINER_SCHEMA: u16 = 1;
pub const EAGER_SECTION_SCHEMA: u16 = 1;
pub const EAGER_SECTION_HEADER_SIZE: usize = 64;
pub const EAGER_SECTION_REFERENCE_SIZE: usize = 24;

const SECTION_MAGIC: &[u8; 8] = b"PACERT\0\0";
const SUPPORTED_FLAGS: u16 = 0;

/// Stable section identifiers used by `eager-runtime.pacbin` members.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u16)]
pub enum EagerSectionKind {
    Metadata = 1,
    CurrentLayout = 2,
    ValueLayout = 3,
    MomentumLayout = 4,
    SourceFill = 5,
    ParameterLayout = 6,
    Stages = 7,
    Couplings = 8,
    Invocations = 9,
    Attachments = 10,
    Finalizations = 11,
    Closures = 12,
    SelectorDomains = 13,
    SelectorMemberships = 14,
    ReductionGroups = 15,
    ReductionEntries = 16,
    ExactFactors = 17,
    InspectionSummary = 18,
}

impl EagerSectionKind {
    fn parse(value: u16) -> RusticolResult<Self> {
        match value {
            1 => Ok(Self::Metadata),
            2 => Ok(Self::CurrentLayout),
            3 => Ok(Self::ValueLayout),
            4 => Ok(Self::MomentumLayout),
            5 => Ok(Self::SourceFill),
            6 => Ok(Self::ParameterLayout),
            7 => Ok(Self::Stages),
            8 => Ok(Self::Couplings),
            9 => Ok(Self::Invocations),
            10 => Ok(Self::Attachments),
            11 => Ok(Self::Finalizations),
            12 => Ok(Self::Closures),
            13 => Ok(Self::SelectorDomains),
            14 => Ok(Self::SelectorMemberships),
            15 => Ok(Self::ReductionGroups),
            16 => Ok(Self::ReductionEntries),
            17 => Ok(Self::ExactFactors),
            18 => Ok(Self::InspectionSummary),
            _ => Err(RusticolError::compatibility(format!(
                "unknown eager runtime section kind: {value}"
            ))),
        }
    }
}

/// Header preceding every fixed-width eager runtime section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerSectionHeader {
    kind: EagerSectionKind,
    record_size: u32,
    record_count: u64,
    payload_length: u64,
}

impl EagerSectionHeader {
    pub fn new(
        kind: EagerSectionKind,
        record_size: u32,
        record_count: u64,
    ) -> RusticolResult<Self> {
        if record_size == 0 {
            return Err(RusticolError::invalid_argument(
                "eager section record size must be positive",
            ));
        }
        let payload_length = u64::from(record_size)
            .checked_mul(record_count)
            .ok_or_else(|| {
                RusticolError::invalid_argument("eager section payload length exceeds u64")
            })?;
        Ok(Self {
            kind,
            record_size,
            record_count,
            payload_length,
        })
    }

    pub fn kind(&self) -> EagerSectionKind {
        self.kind
    }

    pub fn record_size(&self) -> u32 {
        self.record_size
    }

    pub fn record_count(&self) -> u64 {
        self.record_count
    }

    pub fn payload_length(&self) -> u64 {
        self.payload_length
    }

    pub fn encode(&self) -> [u8; EAGER_SECTION_HEADER_SIZE] {
        let mut bytes = [0_u8; EAGER_SECTION_HEADER_SIZE];
        bytes[0..8].copy_from_slice(SECTION_MAGIC);
        put_u16(&mut bytes, 8, EAGER_SECTION_SCHEMA);
        put_u16(&mut bytes, 10, EAGER_SECTION_HEADER_SIZE as u16);
        put_u16(&mut bytes, 12, self.kind as u16);
        put_u16(&mut bytes, 14, SUPPORTED_FLAGS);
        put_u32(&mut bytes, 16, self.record_size);
        put_u64(&mut bytes, 24, self.record_count);
        put_u64(&mut bytes, 32, EAGER_SECTION_HEADER_SIZE as u64);
        put_u64(&mut bytes, 40, self.payload_length);
        bytes
    }

    /// Decode and validate one complete section, returning its record payload.
    pub fn decode(section: &[u8]) -> RusticolResult<(Self, &[u8])> {
        if section.len() < EAGER_SECTION_HEADER_SIZE {
            return Err(RusticolError::integrity(
                "truncated eager runtime section header",
            ));
        }
        let header = &section[..EAGER_SECTION_HEADER_SIZE];
        if &header[0..8] != SECTION_MAGIC {
            return Err(RusticolError::integrity(
                "invalid eager runtime section magic",
            ));
        }
        let schema = read_u16(header, 8, "eager section schema")?;
        if schema != EAGER_SECTION_SCHEMA {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager runtime section schema: {schema}"
            )));
        }
        let header_size = read_u16(header, 10, "eager section header size")?;
        if usize::from(header_size) != EAGER_SECTION_HEADER_SIZE {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager runtime section header size: {header_size}"
            )));
        }
        let kind = EagerSectionKind::parse(read_u16(header, 12, "eager section kind")?)?;
        let flags = read_u16(header, 14, "eager section flags")?;
        if flags != SUPPORTED_FLAGS {
            return Err(RusticolError::compatibility(format!(
                "unknown eager runtime section flags: {flags}"
            )));
        }
        let record_size = read_u32(header, 16, "eager section record size")?;
        if record_size == 0 {
            return Err(RusticolError::integrity(
                "eager section record size must be positive",
            ));
        }
        if read_u32(header, 20, "eager section reserved field")? != 0
            || header[48..64].iter().any(|value| *value != 0)
        {
            return Err(RusticolError::integrity(
                "eager section reserved fields must be zero",
            ));
        }
        let record_count = read_u64(header, 24, "eager section record count")?;
        let payload_offset = read_u64(header, 32, "eager section payload offset")?;
        if payload_offset != EAGER_SECTION_HEADER_SIZE as u64 {
            return Err(RusticolError::integrity(
                "eager section payload offset is not canonical",
            ));
        }
        let payload_length = read_u64(header, 40, "eager section payload length")?;
        let expected_payload_length = u64::from(record_size)
            .checked_mul(record_count)
            .ok_or_else(|| RusticolError::integrity("eager section payload length exceeds u64"))?;
        if payload_length != expected_payload_length {
            return Err(RusticolError::integrity(
                "eager section payload length disagrees with record shape",
            ));
        }
        let expected_total = payload_offset
            .checked_add(payload_length)
            .ok_or_else(|| RusticolError::integrity("eager section size exceeds u64"))?;
        let actual_total = u64::try_from(section.len())
            .map_err(|_| RusticolError::integrity("eager section size exceeds u64"))?;
        if actual_total < expected_total {
            return Err(RusticolError::integrity(
                "truncated eager runtime section payload",
            ));
        }
        if actual_total > expected_total {
            return Err(RusticolError::integrity(
                "eager runtime section has trailing bytes",
            ));
        }
        let payload_start = usize::try_from(payload_offset).map_err(|_| {
            RusticolError::integrity("eager section offset exceeds platform bounds")
        })?;
        Ok((
            Self {
                kind,
                record_size,
                record_count,
                payload_length,
            },
            &section[payload_start..],
        ))
    }
}

/// Fixed-width reference to a contiguous record range in another section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerSectionReference {
    kind: EagerSectionKind,
    start: u64,
    count: u64,
}

impl EagerSectionReference {
    pub fn new(kind: EagerSectionKind, start: u64, count: u64) -> RusticolResult<Self> {
        start.checked_add(count).ok_or_else(|| {
            RusticolError::invalid_argument("eager section reference exceeds u64")
        })?;
        Ok(Self { kind, start, count })
    }

    pub fn kind(&self) -> EagerSectionKind {
        self.kind
    }

    pub fn start(&self) -> u64 {
        self.start
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn encode(&self) -> [u8; EAGER_SECTION_REFERENCE_SIZE] {
        let mut bytes = [0_u8; EAGER_SECTION_REFERENCE_SIZE];
        put_u16(&mut bytes, 0, self.kind as u16);
        put_u16(&mut bytes, 2, SUPPORTED_FLAGS);
        put_u64(&mut bytes, 8, self.start);
        put_u64(&mut bytes, 16, self.count);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> RusticolResult<Self> {
        if bytes.len() < EAGER_SECTION_REFERENCE_SIZE {
            return Err(RusticolError::integrity(
                "truncated eager section reference",
            ));
        }
        if bytes.len() > EAGER_SECTION_REFERENCE_SIZE {
            return Err(RusticolError::integrity(
                "eager section reference has trailing bytes",
            ));
        }
        let kind = EagerSectionKind::parse(read_u16(bytes, 0, "eager reference kind")?)?;
        let flags = read_u16(bytes, 2, "eager reference flags")?;
        if flags != SUPPORTED_FLAGS {
            return Err(RusticolError::compatibility(format!(
                "unknown eager section reference flags: {flags}"
            )));
        }
        if read_u32(bytes, 4, "eager reference reserved field")? != 0 {
            return Err(RusticolError::integrity(
                "eager section reference reserved field must be zero",
            ));
        }
        let start = read_u64(bytes, 8, "eager reference start")?;
        let count = read_u64(bytes, 16, "eager reference count")?;
        Self::new(kind, start, count).map_err(|error| RusticolError::integrity(error.to_string()))
    }

    pub fn validate_against(&self, target: &EagerSectionHeader) -> RusticolResult<()> {
        if self.kind != target.kind {
            return Err(RusticolError::integrity(format!(
                "eager section reference kind {:?} does not match target {:?}",
                self.kind, target.kind
            )));
        }
        let end = self
            .start
            .checked_add(self.count)
            .ok_or_else(|| RusticolError::integrity("eager section reference exceeds u64"))?;
        if self.start > target.record_count || end > target.record_count {
            return Err(RusticolError::integrity(
                "eager section reference exceeds target record bounds",
            ));
        }
        Ok(())
    }
}

fn read_u16(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u16> {
    let value: [u8; 2] = checked_slice(bytes, offset, 2, label)?
        .try_into()
        .map_err(|_| RusticolError::integrity(format!("truncated {label}")))?;
    Ok(u16::from_le_bytes(value))
}

fn read_u32(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u32> {
    let value: [u8; 4] = checked_slice(bytes, offset, 4, label)?
        .try_into()
        .map_err(|_| RusticolError::integrity(format!("truncated {label}")))?;
    Ok(u32::from_le_bytes(value))
}

fn read_u64(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u64> {
    let value: [u8; 8] = checked_slice(bytes, offset, 8, label)?
        .try_into()
        .map_err(|_| RusticolError::integrity(format!("truncated {label}")))?;
    Ok(u64::from_le_bytes(value))
}

fn checked_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    length: usize,
    label: &str,
) -> RusticolResult<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| RusticolError::integrity(format!("{label} range overflow")))?;
    bytes
        .get(offset..end)
        .ok_or_else(|| RusticolError::integrity(format!("truncated {label}")))
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
#[path = "eager_layout_tests.rs"]
mod tests;
