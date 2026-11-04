// SPDX-License-Identifier: 0BSD

use std::ops::Range;

use sha2::{Digest, Sha256};

use super::{
    RECURRENCE_BUILDER_INPUT_ABI, RECURRENCE_BUILDER_RESULT_ABI,
    RECURRENCE_CONTRACTED_COLOR_CAPABILITY, RECURRENCE_INPUT_ENDIANNESS,
    RECURRENCE_LC_COLOR_CAPABILITY, RECURRENCE_PLAN_ABI, RECURRENCE_RUNTIME_CAPABILITY,
    RECURRENCE_RUNTIME_KIND, RECURRENCE_RUNTIME_LAYOUT_ABI, RECURRENCE_TEMPLATE_ABI,
    RecurrenceStrategy, SemanticDigest,
};
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Checked offset/count pair used by all flat recurrence catalogs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CheckedTableRange {
    pub start: u64,
    pub count: u64,
}

impl CheckedTableRange {
    pub const fn new(start: u64, count: u64) -> Self {
        Self { start, count }
    }

    pub fn end(self, label: &str) -> RusticolResult<u64> {
        self.start.checked_add(self.count).ok_or_else(|| {
            invalid(format!(
                "{label} range start {} plus count {} exceeds u64",
                self.start, self.count
            ))
        })
    }

    pub fn as_usize_range(self, value_len: usize, label: &str) -> RusticolResult<Range<usize>> {
        let end = self.end(label)?;
        let start = checked_usize(self.start, label)?;
        let end = checked_usize(end, label)?;
        if end > value_len {
            return Err(invalid(format!(
                "{label} range {}..{} exceeds table length {value_len}",
                self.start,
                self.start + self.count
            )));
        }
        Ok(start..end)
    }
}

/// Borrowed canonical multiword-mask catalog.
///
/// Words are little-endian by significance.  Zero is encoded as an empty
/// range and nonzero masks must not contain a trailing zero word.
#[derive(Clone, Copy, Debug)]
pub struct MultiwordMaskCatalogView<'a> {
    pub ranges: &'a [CheckedTableRange],
    pub populations: &'a [u64],
    pub words: &'a [u64],
}

impl<'a> MultiwordMaskCatalogView<'a> {
    pub fn validate(self, require_strict_order: bool) -> RusticolResult<()> {
        validate_equal_column_lengths(
            "multiword mask catalog",
            &[
                ("ranges", self.ranges.len()),
                ("populations", self.populations.len()),
            ],
        )?;
        validate_packed_ranges("multiword mask", self.ranges, self.words.len())?;

        let mut previous: Option<&[u64]> = None;
        for (index, (range, expected_population)) in self
            .ranges
            .iter()
            .copied()
            .zip(self.populations.iter().copied())
            .enumerate()
        {
            let words = &self.words
                [range.as_usize_range(self.words.len(), &format!("multiword mask {index}"))?];
            if words.last() == Some(&0) {
                return Err(invalid(format!(
                    "multiword mask {index} has a noncanonical trailing zero word"
                )));
            }
            let population = words.iter().try_fold(0_u64, |total, word| {
                total
                    .checked_add(u64::from(word.count_ones()))
                    .ok_or_else(|| invalid("multiword mask population exceeds u64"))
            })?;
            if population != expected_population {
                return Err(invalid(format!(
                    "multiword mask {index} declares population {expected_population}, found {population}"
                )));
            }
            if require_strict_order
                && let Some(previous) = previous
                && compare_masks(previous, words) != std::cmp::Ordering::Less
            {
                return Err(invalid(format!(
                    "multiword mask catalog is not in strict canonical order at row {index}"
                )));
            }
            previous = Some(words);
        }
        Ok(())
    }

    pub fn words(self, mask_id: u32) -> RusticolResult<&'a [u64]> {
        let row = self.ranges.get(mask_id as usize).ok_or_else(|| {
            invalid(format!(
                "multiword mask id {mask_id} exceeds catalog length {}",
                self.ranges.len()
            ))
        })?;
        let range = row.as_usize_range(self.words.len(), "multiword mask")?;
        Ok(&self.words[range])
    }

    pub fn contains(self, mask_id: u32, bit: u64) -> RusticolResult<bool> {
        let words = self.words(mask_id)?;
        let word_index = checked_usize(bit / 64, "multiword mask bit index")?;
        Ok(words
            .get(word_index)
            .is_some_and(|word| word & (1_u64 << (bit % 64)) != 0))
    }
}

/// A named, fixed-row-width little-endian builder-input section.
#[derive(Clone, Copy, Debug)]
pub struct CanonicalInputSection<'a> {
    pub name: &'a str,
    pub row_width: u32,
    pub row_count: u64,
    pub bytes: &'a [u8],
}

impl CanonicalInputSection<'_> {
    pub fn validate(self) -> RusticolResult<()> {
        validate_section_name(self.name)?;
        if self.row_width == 0 {
            return Err(invalid(format!(
                "recurrence input section {:?} has zero row width",
                self.name
            )));
        }
        let expected = u64::from(self.row_width)
            .checked_mul(self.row_count)
            .ok_or_else(|| {
                invalid(format!(
                    "recurrence input section {:?} byte count exceeds u64",
                    self.name
                ))
            })?;
        let actual = checked_u64_len(self.bytes.len(), self.name)?;
        if actual != expected {
            return Err(invalid(format!(
                "recurrence input section {:?} expects {expected} bytes, found {actual}",
                self.name
            )));
        }
        Ok(())
    }
}

/// Deterministic header for `pyamplicol-recurrence-builder-input-v2`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceBuilderInputHeader {
    pub template_abi: String,
    pub builder_input_abi: String,
    pub builder_result_abi: String,
    pub plan_abi: String,
    pub runtime_layout_abi: String,
    pub runtime_kind: String,
    pub runtime_capability: String,
    pub color_capability: String,
    pub endianness: String,
    pub strategy: RecurrenceStrategy,
    pub section_count: u32,
    pub template_catalog_digest: SemanticDigest,
    pub process_digest: SemanticDigest,
    pub semantic_digest: SemanticDigest,
    pub input_digest: SemanticDigest,
}

impl RecurrenceBuilderInputHeader {
    pub fn canonical(
        strategy: RecurrenceStrategy,
        section_count: u32,
        template_catalog_digest: SemanticDigest,
        process_digest: SemanticDigest,
        semantic_digest: SemanticDigest,
        input_digest: SemanticDigest,
    ) -> Self {
        Self {
            template_abi: RECURRENCE_TEMPLATE_ABI.to_owned(),
            builder_input_abi: RECURRENCE_BUILDER_INPUT_ABI.to_owned(),
            builder_result_abi: RECURRENCE_BUILDER_RESULT_ABI.to_owned(),
            plan_abi: RECURRENCE_PLAN_ABI.to_owned(),
            runtime_layout_abi: RECURRENCE_RUNTIME_LAYOUT_ABI.to_owned(),
            runtime_kind: RECURRENCE_RUNTIME_KIND.to_owned(),
            runtime_capability: RECURRENCE_RUNTIME_CAPABILITY.to_owned(),
            color_capability: match strategy {
                RecurrenceStrategy::ContractedColorUnion => RECURRENCE_CONTRACTED_COLOR_CAPABILITY,
                RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::AllFlowUnion => {
                    RECURRENCE_LC_COLOR_CAPABILITY
                }
            }
            .to_owned(),
            endianness: RECURRENCE_INPUT_ENDIANNESS.to_owned(),
            strategy,
            section_count,
            template_catalog_digest,
            process_digest,
            semantic_digest,
            input_digest,
        }
    }

    pub fn validate_identity(&self) -> RusticolResult<()> {
        validate_constant("template ABI", &self.template_abi, RECURRENCE_TEMPLATE_ABI)?;
        validate_constant(
            "builder input ABI",
            &self.builder_input_abi,
            RECURRENCE_BUILDER_INPUT_ABI,
        )?;
        validate_constant(
            "builder result ABI",
            &self.builder_result_abi,
            RECURRENCE_BUILDER_RESULT_ABI,
        )?;
        validate_constant("plan ABI", &self.plan_abi, RECURRENCE_PLAN_ABI)?;
        validate_constant(
            "runtime layout ABI",
            &self.runtime_layout_abi,
            RECURRENCE_RUNTIME_LAYOUT_ABI,
        )?;
        validate_constant("runtime kind", &self.runtime_kind, RECURRENCE_RUNTIME_KIND)?;
        validate_constant(
            "runtime capability",
            &self.runtime_capability,
            RECURRENCE_RUNTIME_CAPABILITY,
        )?;
        let expected_color_capability = match self.strategy {
            RecurrenceStrategy::ContractedColorUnion => RECURRENCE_CONTRACTED_COLOR_CAPABILITY,
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::AllFlowUnion => {
                RECURRENCE_LC_COLOR_CAPABILITY
            }
        };
        validate_constant(
            "color capability",
            &self.color_capability,
            expected_color_capability,
        )?;
        validate_constant(
            "input endianness",
            &self.endianness,
            RECURRENCE_INPUT_ENDIANNESS,
        )?;
        Ok(())
    }
}

pub fn checked_u32_len(length: usize, label: &str) -> RusticolResult<u32> {
    u32::try_from(length).map_err(|_| invalid(format!("{label} length {length} exceeds u32")))
}

pub fn checked_u64_len(length: usize, label: &str) -> RusticolResult<u64> {
    u64::try_from(length).map_err(|_| invalid(format!("{label} length {length} exceeds u64")))
}

pub fn checked_usize(value: u64, label: &str) -> RusticolResult<usize> {
    usize::try_from(value).map_err(|_| invalid(format!("{label} value {value} exceeds usize")))
}

pub fn validate_equal_column_lengths(
    table: &str,
    columns: &[(&str, usize)],
) -> RusticolResult<usize> {
    let Some((first_name, expected)) = columns.first().copied() else {
        return Err(invalid(format!("{table} has no columns")));
    };
    checked_u32_len(expected, &format!("{table}.{first_name}"))?;
    for (name, length) in columns.iter().copied().skip(1) {
        checked_u32_len(length, &format!("{table}.{name}"))?;
        if length != expected {
            return Err(invalid(format!(
                "{table} column {name:?} has length {length}, expected {expected} from {first_name:?}"
            )));
        }
    }
    Ok(expected)
}

pub fn validate_u32_references(
    values: &[u32],
    target_len: usize,
    label: &str,
) -> RusticolResult<()> {
    checked_u32_len(target_len, &format!("{label} target"))?;
    for (row, value) in values.iter().copied().enumerate() {
        let value_index = usize::try_from(value).map_err(|_| {
            invalid(format!(
                "{label} row {row} id {value} exceeds usize on this platform"
            ))
        })?;
        if value_index >= target_len {
            return Err(invalid(format!(
                "{label} row {row} references id {value}, target length is {target_len}"
            )));
        }
    }
    Ok(())
}

pub fn validate_ranges_within(
    label: &str,
    ranges: &[CheckedTableRange],
    value_len: usize,
) -> RusticolResult<()> {
    checked_u32_len(ranges.len(), &format!("{label} ranges"))?;
    for (index, range) in ranges.iter().copied().enumerate() {
        range.as_usize_range(value_len, &format!("{label} {index}"))?;
    }
    Ok(())
}

pub fn validate_packed_ranges(
    label: &str,
    ranges: &[CheckedTableRange],
    value_len: usize,
) -> RusticolResult<()> {
    validate_ranges_within(label, ranges, value_len)?;
    let mut expected_start = 0_u64;
    for (index, range) in ranges.iter().copied().enumerate() {
        if range.start != expected_start {
            return Err(invalid(format!(
                "{label} range {index} starts at {}, expected packed offset {expected_start}",
                range.start
            )));
        }
        expected_start = range.end(&format!("{label} {index}"))?;
    }
    if checked_usize(expected_start, label)? != value_len {
        return Err(invalid(format!(
            "{label} packed ranges cover {expected_start} values, table contains {value_len}"
        )));
    }
    Ok(())
}

pub fn canonical_input_digest(
    header: &RecurrenceBuilderInputHeader,
    sections: &[CanonicalInputSection<'_>],
) -> RusticolResult<SemanticDigest> {
    header.validate_identity()?;
    validate_sections(sections)?;
    let actual_section_count = checked_u32_len(sections.len(), "recurrence input sections")?;
    if header.section_count != actual_section_count {
        return Err(invalid(format!(
            "recurrence input header declares {} sections, found {actual_section_count}",
            header.section_count
        )));
    }
    let mut digest = Sha256::new();
    digest_field(&mut digest, RECURRENCE_BUILDER_INPUT_ABI.as_bytes())?;
    digest_field(&mut digest, RECURRENCE_TEMPLATE_ABI.as_bytes())?;
    digest_field(&mut digest, RECURRENCE_PLAN_ABI.as_bytes())?;
    digest_field(&mut digest, RECURRENCE_RUNTIME_LAYOUT_ABI.as_bytes())?;
    digest.update(header.strategy.as_u32().to_le_bytes());
    digest.update(header.section_count.to_le_bytes());
    digest.update(header.template_catalog_digest.as_bytes());
    digest.update(header.process_digest.as_bytes());
    digest.update(header.semantic_digest.as_bytes());
    for section in sections {
        digest_field(&mut digest, section.name.as_bytes())?;
        digest.update(section.row_width.to_le_bytes());
        digest.update(section.row_count.to_le_bytes());
        digest_field(&mut digest, section.bytes)?;
    }
    SemanticDigest::new(digest.finalize().into())
}

pub fn validate_header_and_sections(
    header: &RecurrenceBuilderInputHeader,
    sections: &[CanonicalInputSection<'_>],
) -> RusticolResult<()> {
    header.validate_identity()?;
    let actual_count = checked_u32_len(sections.len(), "recurrence input sections")?;
    if header.section_count != actual_count {
        return Err(invalid(format!(
            "recurrence input header declares {} sections, found {actual_count}",
            header.section_count
        )));
    }
    let actual = canonical_input_digest(header, sections)?;
    if actual != header.input_digest {
        return Err(invalid(format!(
            "recurrence input digest mismatch: expected {}, found {}",
            header.input_digest, actual
        )));
    }
    Ok(())
}

fn validate_sections(sections: &[CanonicalInputSection<'_>]) -> RusticolResult<()> {
    checked_u32_len(sections.len(), "recurrence input sections")?;
    let mut previous = None;
    for section in sections {
        section.validate()?;
        if let Some(previous) = previous
            && previous >= section.name
        {
            return Err(invalid(format!(
                "recurrence input sections are not in strict canonical name order at {:?}",
                section.name
            )));
        }
        previous = Some(section.name);
    }
    Ok(())
}

fn validate_section_name(name: &str) -> RusticolResult<()> {
    if name.is_empty()
        || name.len() > 255
        || !name.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'_' | b'-' | b'.' | b'/')
        })
        || name.starts_with('/')
        || name.ends_with('/')
        || name.contains("//")
        || name
            .split('/')
            .any(|component| component == "." || component == "..")
    {
        return Err(invalid(format!(
            "invalid recurrence input section name {name:?}"
        )));
    }
    Ok(())
}

fn validate_constant(label: &str, value: &str, expected: &str) -> RusticolResult<()> {
    if value != expected {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence {label} {value:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn digest_field(digest: &mut Sha256, bytes: &[u8]) -> RusticolResult<()> {
    let length = checked_u64_len(bytes.len(), "recurrence digest field")?;
    digest.update(length.to_le_bytes());
    digest.update(bytes);
    Ok(())
}

fn compare_masks(left: &[u64], right: &[u64]) -> std::cmp::Ordering {
    left.len()
        .cmp(&right.len())
        .then_with(|| left.iter().rev().cmp(right.iter().rev()))
}
