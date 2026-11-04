// SPDX-License-Identifier: 0BSD

use crate::{RusticolError, RusticolResult};
use std::mem::size_of;

pub const EAGER_PLAN_ABI: &str = "pyamplicol-eager-plan-v2";
pub const EAGER_KERNEL_ABI: &str = "pyamplicol-eager-kernel-v1";
pub const EAGER_SELECTOR_DOMAINS_ABI: &str = "pyamplicol-eager-selector-domains-v1";
pub const EAGER_RUNTIME_CAPABILITY: &str = "rusticol.eager-dag.complex-f64.v1";
pub const EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY: &str =
    "rusticol.eager-dag.lc-topology-replay.v1";
pub const MISSING_U32: u32 = u32::MAX;
pub const EAGER_OUTPUT_FACTOR_NONE: u32 = 0;
pub const EAGER_OUTPUT_FACTOR_COUPLING_REAL: u32 = 1;
pub const EAGER_OUTPUT_FACTOR_COUPLING_IMAG: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EagerInvocationRow {
    pub kernel_id: u32,
    pub left_value_slot_id: u32,
    pub right_value_slot_id: u32,
    pub left_momentum_slot_id: u32,
    pub right_momentum_slot_id: u32,
    pub coupling_slot_id: u32,
    pub output_factor_source: u32,
    pub attachment_start: u64,
    pub attachment_count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EagerAttachmentRow {
    pub result_current_id: u32,
    pub factor_real: f64,
    pub factor_imag: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EagerCouplingRow {
    pub real_parameter_id: u32,
    pub imag_parameter_id: u32,
    pub constant_real: f64,
    pub constant_imag: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EagerFinalizationRow {
    pub kernel_id: u32,
    pub current_id: u32,
    pub unpropagated_value_slot_id: u32,
    pub propagated_value_slot_id: u32,
    pub momentum_slot_id: u32,
}

impl EagerFinalizationRow {
    pub fn applies_kernel(self) -> bool {
        self.kernel_id != MISSING_U32
    }

    pub fn stores_unpropagated(self) -> bool {
        self.unpropagated_value_slot_id != MISSING_U32
    }

    pub fn stores_propagated(self) -> bool {
        self.propagated_value_slot_id != MISSING_U32
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EagerClosureRow {
    pub kernel_id: u32,
    pub left_value_slot_id: u32,
    pub right_value_slot_id: u32,
    pub amplitude_index: u32,
    pub coupling_slot_id: u32,
    pub output_factor_source: u32,
    pub factor_real: f64,
    pub factor_imag: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerSelectorDomainRow {
    pub member_start: u64,
    pub member_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerSelectorGroupRow {
    pub coherent_group_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerSelectorDomainIdRow {
    pub domain_id: u32,
}

trait FixedWidthRow: Sized {
    const NAME: &'static str;
    const WIDTH: usize;

    fn validate_for_encoding(&self, row_index: usize) -> RusticolResult<()>;
    fn encode_into(&self, output: &mut Vec<u8>);
    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self>;
}

macro_rules! impl_table_api {
    ($row:ty) => {
        impl $row {
            pub const ENCODED_LEN: usize = <Self as FixedWidthRow>::WIDTH;

            pub fn encode_table(rows: &[Self]) -> RusticolResult<Vec<u8>> {
                encode_rows(rows)
            }

            pub fn decode_table(payload: &[u8]) -> RusticolResult<Vec<Self>> {
                decode_rows(payload)
            }
        }
    };
}

impl FixedWidthRow for EagerInvocationRow {
    const NAME: &'static str = "invocation";
    const WIDTH: usize = 7 * size_of::<u32>() + 2 * size_of::<u64>();

    fn validate_for_encoding(&self, row_index: usize) -> RusticolResult<()> {
        validate_output_factor_source_for_encoding(Self::NAME, row_index, self.output_factor_source)
    }

    fn encode_into(&self, output: &mut Vec<u8>) {
        push_u32(output, self.kernel_id);
        push_u32(output, self.left_value_slot_id);
        push_u32(output, self.right_value_slot_id);
        push_u32(output, self.left_momentum_slot_id);
        push_u32(output, self.right_momentum_slot_id);
        push_u32(output, self.coupling_slot_id);
        push_u32(output, self.output_factor_source);
        push_u64(output, self.attachment_start);
        push_u64(output, self.attachment_count);
    }

    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self> {
        let mut reader = RowReader::new(Self::NAME, row_index, bytes);
        let row = Self {
            kernel_id: reader.read_u32()?,
            left_value_slot_id: reader.read_u32()?,
            right_value_slot_id: reader.read_u32()?,
            left_momentum_slot_id: reader.read_u32()?,
            right_momentum_slot_id: reader.read_u32()?,
            coupling_slot_id: reader.read_u32()?,
            output_factor_source: reader.read_u32()?,
            attachment_start: reader.read_u64()?,
            attachment_count: reader.read_u64()?,
        };
        reader.finish()?;
        validate_output_factor_source_from_payload(
            Self::NAME,
            row_index,
            row.output_factor_source,
        )?;
        Ok(row)
    }
}

impl FixedWidthRow for EagerAttachmentRow {
    const NAME: &'static str = "attachment";
    const WIDTH: usize = size_of::<u32>() + 2 * size_of::<f64>();

    fn validate_for_encoding(&self, row_index: usize) -> RusticolResult<()> {
        validate_finite_for_encoding(Self::NAME, row_index, "factor_real", self.factor_real)?;
        validate_finite_for_encoding(Self::NAME, row_index, "factor_imag", self.factor_imag)
    }

    fn encode_into(&self, output: &mut Vec<u8>) {
        push_u32(output, self.result_current_id);
        push_f64(output, self.factor_real);
        push_f64(output, self.factor_imag);
    }

    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self> {
        let mut reader = RowReader::new(Self::NAME, row_index, bytes);
        let row = Self {
            result_current_id: reader.read_u32()?,
            factor_real: reader.read_f64()?,
            factor_imag: reader.read_f64()?,
        };
        reader.finish()?;
        validate_finite_from_payload(Self::NAME, row_index, "factor_real", row.factor_real)?;
        validate_finite_from_payload(Self::NAME, row_index, "factor_imag", row.factor_imag)?;
        Ok(row)
    }
}

impl FixedWidthRow for EagerCouplingRow {
    const NAME: &'static str = "coupling";
    const WIDTH: usize = 2 * size_of::<u32>() + 2 * size_of::<f64>();

    fn validate_for_encoding(&self, row_index: usize) -> RusticolResult<()> {
        validate_finite_for_encoding(Self::NAME, row_index, "constant_real", self.constant_real)?;
        validate_finite_for_encoding(Self::NAME, row_index, "constant_imag", self.constant_imag)
    }

    fn encode_into(&self, output: &mut Vec<u8>) {
        push_u32(output, self.real_parameter_id);
        push_u32(output, self.imag_parameter_id);
        push_f64(output, self.constant_real);
        push_f64(output, self.constant_imag);
    }

    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self> {
        let mut reader = RowReader::new(Self::NAME, row_index, bytes);
        let row = Self {
            real_parameter_id: reader.read_u32()?,
            imag_parameter_id: reader.read_u32()?,
            constant_real: reader.read_f64()?,
            constant_imag: reader.read_f64()?,
        };
        reader.finish()?;
        validate_finite_from_payload(Self::NAME, row_index, "constant_real", row.constant_real)?;
        validate_finite_from_payload(Self::NAME, row_index, "constant_imag", row.constant_imag)?;
        Ok(row)
    }
}

impl FixedWidthRow for EagerFinalizationRow {
    const NAME: &'static str = "finalization";
    const WIDTH: usize = 5 * size_of::<u32>();

    fn validate_for_encoding(&self, _row_index: usize) -> RusticolResult<()> {
        Ok(())
    }

    fn encode_into(&self, output: &mut Vec<u8>) {
        push_u32(output, self.kernel_id);
        push_u32(output, self.current_id);
        push_u32(output, self.unpropagated_value_slot_id);
        push_u32(output, self.propagated_value_slot_id);
        push_u32(output, self.momentum_slot_id);
    }

    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self> {
        let mut reader = RowReader::new(Self::NAME, row_index, bytes);
        let row = Self {
            kernel_id: reader.read_u32()?,
            current_id: reader.read_u32()?,
            unpropagated_value_slot_id: reader.read_u32()?,
            propagated_value_slot_id: reader.read_u32()?,
            momentum_slot_id: reader.read_u32()?,
        };
        reader.finish()?;
        Ok(row)
    }
}

impl FixedWidthRow for EagerClosureRow {
    const NAME: &'static str = "closure";
    const WIDTH: usize = 6 * size_of::<u32>() + 2 * size_of::<f64>();

    fn validate_for_encoding(&self, row_index: usize) -> RusticolResult<()> {
        validate_output_factor_source_for_encoding(
            Self::NAME,
            row_index,
            self.output_factor_source,
        )?;
        validate_finite_for_encoding(Self::NAME, row_index, "factor_real", self.factor_real)?;
        validate_finite_for_encoding(Self::NAME, row_index, "factor_imag", self.factor_imag)
    }

    fn encode_into(&self, output: &mut Vec<u8>) {
        push_u32(output, self.kernel_id);
        push_u32(output, self.left_value_slot_id);
        push_u32(output, self.right_value_slot_id);
        push_u32(output, self.amplitude_index);
        push_u32(output, self.coupling_slot_id);
        push_u32(output, self.output_factor_source);
        push_f64(output, self.factor_real);
        push_f64(output, self.factor_imag);
    }

    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self> {
        let mut reader = RowReader::new(Self::NAME, row_index, bytes);
        let row = Self {
            kernel_id: reader.read_u32()?,
            left_value_slot_id: reader.read_u32()?,
            right_value_slot_id: reader.read_u32()?,
            amplitude_index: reader.read_u32()?,
            coupling_slot_id: reader.read_u32()?,
            output_factor_source: reader.read_u32()?,
            factor_real: reader.read_f64()?,
            factor_imag: reader.read_f64()?,
        };
        reader.finish()?;
        validate_output_factor_source_from_payload(
            Self::NAME,
            row_index,
            row.output_factor_source,
        )?;
        validate_finite_from_payload(Self::NAME, row_index, "factor_real", row.factor_real)?;
        validate_finite_from_payload(Self::NAME, row_index, "factor_imag", row.factor_imag)?;
        Ok(row)
    }
}

impl FixedWidthRow for EagerSelectorDomainRow {
    const NAME: &'static str = "selector-domain";
    const WIDTH: usize = 2 * size_of::<u64>();

    fn validate_for_encoding(&self, _row_index: usize) -> RusticolResult<()> {
        Ok(())
    }

    fn encode_into(&self, output: &mut Vec<u8>) {
        push_u64(output, self.member_start);
        push_u64(output, self.member_count);
    }

    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self> {
        let mut reader = RowReader::new(Self::NAME, row_index, bytes);
        let row = Self {
            member_start: reader.read_u64()?,
            member_count: reader.read_u64()?,
        };
        reader.finish()?;
        Ok(row)
    }
}

impl FixedWidthRow for EagerSelectorGroupRow {
    const NAME: &'static str = "selector-domain-group";
    const WIDTH: usize = size_of::<u32>();

    fn validate_for_encoding(&self, _row_index: usize) -> RusticolResult<()> {
        Ok(())
    }

    fn encode_into(&self, output: &mut Vec<u8>) {
        push_u32(output, self.coherent_group_id);
    }

    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self> {
        let mut reader = RowReader::new(Self::NAME, row_index, bytes);
        let row = Self {
            coherent_group_id: reader.read_u32()?,
        };
        reader.finish()?;
        Ok(row)
    }
}

impl FixedWidthRow for EagerSelectorDomainIdRow {
    const NAME: &'static str = "selector-domain-id";
    const WIDTH: usize = size_of::<u32>();

    fn validate_for_encoding(&self, _row_index: usize) -> RusticolResult<()> {
        Ok(())
    }

    fn encode_into(&self, output: &mut Vec<u8>) {
        push_u32(output, self.domain_id);
    }

    fn decode_from(bytes: &[u8], row_index: usize) -> RusticolResult<Self> {
        let mut reader = RowReader::new(Self::NAME, row_index, bytes);
        let row = Self {
            domain_id: reader.read_u32()?,
        };
        reader.finish()?;
        Ok(row)
    }
}

impl_table_api!(EagerInvocationRow);
impl_table_api!(EagerAttachmentRow);
impl_table_api!(EagerCouplingRow);
impl_table_api!(EagerFinalizationRow);
impl_table_api!(EagerClosureRow);
impl_table_api!(EagerSelectorDomainRow);
impl_table_api!(EagerSelectorGroupRow);
impl_table_api!(EagerSelectorDomainIdRow);

fn encode_rows<Row: FixedWidthRow>(rows: &[Row]) -> RusticolResult<Vec<u8>> {
    let byte_count = rows.len().checked_mul(Row::WIDTH).ok_or_else(|| {
        RusticolError::invalid_argument(format!(
            "eager {} table row count overflows its encoded length",
            Row::NAME
        ))
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(byte_count).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "could not reserve {byte_count} bytes for eager {} table: {error}",
            Row::NAME
        ))
    })?;
    for (row_index, row) in rows.iter().enumerate() {
        row.validate_for_encoding(row_index)?;
        row.encode_into(&mut output);
    }
    if output.len() != byte_count {
        return Err(RusticolError::internal(format!(
            "eager {} encoder produced {} bytes instead of {byte_count}",
            Row::NAME,
            output.len()
        )));
    }
    Ok(output)
}

fn decode_rows<Row: FixedWidthRow>(payload: &[u8]) -> RusticolResult<Vec<Row>> {
    if !payload.len().is_multiple_of(Row::WIDTH) {
        return Err(RusticolError::artifact(format!(
            "eager {} table has {} bytes, not a multiple of {}",
            Row::NAME,
            payload.len(),
            Row::WIDTH
        )));
    }
    let row_count = payload.len() / Row::WIDTH;
    let expected_len = row_count.checked_mul(Row::WIDTH).ok_or_else(|| {
        RusticolError::artifact(format!(
            "eager {} table row count overflows its encoded length",
            Row::NAME
        ))
    })?;
    if expected_len != payload.len() {
        return Err(RusticolError::artifact(format!(
            "eager {} table length is inconsistent",
            Row::NAME
        )));
    }

    let mut rows = Vec::new();
    rows.try_reserve_exact(row_count).map_err(|error| {
        RusticolError::artifact(format!(
            "could not reserve {row_count} eager {} rows: {error}",
            Row::NAME
        ))
    })?;
    for (row_index, bytes) in payload.chunks_exact(Row::WIDTH).enumerate() {
        rows.push(Row::decode_from(bytes, row_index)?);
    }
    Ok(rows)
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_f64(output: &mut Vec<u8>, value: f64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn output_factor_source_is_valid(source: u32) -> bool {
    matches!(
        source,
        EAGER_OUTPUT_FACTOR_NONE
            | EAGER_OUTPUT_FACTOR_COUPLING_REAL
            | EAGER_OUTPUT_FACTOR_COUPLING_IMAG
    )
}

fn validate_output_factor_source_for_encoding(
    table: &str,
    row_index: usize,
    source: u32,
) -> RusticolResult<()> {
    if !output_factor_source_is_valid(source) {
        return Err(RusticolError::invalid_argument(format!(
            "eager {table} row {row_index} has unsupported output factor source {source}"
        )));
    }
    Ok(())
}

fn validate_output_factor_source_from_payload(
    table: &str,
    row_index: usize,
    source: u32,
) -> RusticolResult<()> {
    if !output_factor_source_is_valid(source) {
        return Err(RusticolError::artifact(format!(
            "eager {table} row {row_index} has unsupported output factor source {source}"
        )));
    }
    Ok(())
}

fn validate_finite_for_encoding(
    table: &str,
    row_index: usize,
    field: &str,
    value: f64,
) -> RusticolResult<()> {
    if !value.is_finite() {
        return Err(RusticolError::invalid_argument(format!(
            "eager {table} row {row_index} has non-finite {field}"
        )));
    }
    Ok(())
}

fn validate_finite_from_payload(
    table: &str,
    row_index: usize,
    field: &str,
    value: f64,
) -> RusticolResult<()> {
    if !value.is_finite() {
        return Err(RusticolError::artifact(format!(
            "eager {table} row {row_index} has non-finite {field}"
        )));
    }
    Ok(())
}

struct RowReader<'a> {
    table: &'static str,
    row_index: usize,
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> RowReader<'a> {
    fn new(table: &'static str, row_index: usize, bytes: &'a [u8]) -> Self {
        Self {
            table,
            row_index,
            bytes,
            offset: 0,
        }
    }

    fn read_u32(&mut self) -> RusticolResult<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> RusticolResult<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_f64(&mut self) -> RusticolResult<f64> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }

    fn read_array<const N: usize>(&mut self) -> RusticolResult<[u8; N]> {
        let end = self.offset.checked_add(N).ok_or_else(|| {
            RusticolError::artifact(format!(
                "eager {} row {} byte offset overflow",
                self.table, self.row_index
            ))
        })?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(|| {
            RusticolError::artifact(format!(
                "eager {} row {} is truncated at byte {}",
                self.table, self.row_index, self.offset
            ))
        })?;
        let value = bytes.try_into().map_err(|_| {
            RusticolError::internal(format!(
                "could not decode eager {} row {}",
                self.table, self.row_index
            ))
        })?;
        self.offset = end;
        Ok(value)
    }

    fn finish(&self) -> RusticolResult<()> {
        if self.offset != self.bytes.len() {
            return Err(RusticolError::artifact(format!(
                "eager {} row {} has {} trailing bytes",
                self.table,
                self.row_index,
                self.bytes.len() - self.offset
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RusticolErrorKind;

    #[test]
    fn constants_match_python_contract() {
        assert_eq!(EAGER_PLAN_ABI, "pyamplicol-eager-plan-v2");
        assert_eq!(EAGER_KERNEL_ABI, "pyamplicol-eager-kernel-v1");
        assert_eq!(
            EAGER_RUNTIME_CAPABILITY,
            "rusticol.eager-dag.complex-f64.v1"
        );
        assert_eq!(MISSING_U32, u32::MAX);
    }

    #[test]
    fn invocation_matches_python_little_endian_fixture() {
        let row = EagerInvocationRow {
            kernel_id: 0x0102_0304,
            left_value_slot_id: 0x1112_1314,
            right_value_slot_id: 0x2122_2324,
            left_momentum_slot_id: 0x3132_3334,
            right_momentum_slot_id: 0x4142_4344,
            coupling_slot_id: 0x5152_5354,
            output_factor_source: EAGER_OUTPUT_FACTOR_COUPLING_IMAG,
            attachment_start: 0x0102_0304_0506_0708,
            attachment_count: 0x1112_1314_1516_1718,
        };
        let expected = [
            0x04, 0x03, 0x02, 0x01, 0x14, 0x13, 0x12, 0x11, 0x24, 0x23, 0x22, 0x21, 0x34, 0x33,
            0x32, 0x31, 0x44, 0x43, 0x42, 0x41, 0x54, 0x53, 0x52, 0x51, 0x02, 0x00, 0x00, 0x00,
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13,
            0x12, 0x11,
        ];

        assert_eq!(EagerInvocationRow::ENCODED_LEN, 44);
        assert_eq!(EagerInvocationRow::encode_table(&[row]).unwrap(), expected);
        assert_eq!(EagerInvocationRow::decode_table(&expected).unwrap(), [row]);
    }

    #[test]
    fn attachment_matches_python_little_endian_fixture() {
        let row = EagerAttachmentRow {
            result_current_id: 0x0102_0304,
            factor_real: 1.5,
            factor_imag: -2.25,
        };
        let expected = [
            0x04, 0x03, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x3f, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x02, 0xc0,
        ];

        assert_eq!(EagerAttachmentRow::ENCODED_LEN, 20);
        assert_eq!(EagerAttachmentRow::encode_table(&[row]).unwrap(), expected);
        assert_eq!(EagerAttachmentRow::decode_table(&expected).unwrap(), [row]);
    }

    #[test]
    fn coupling_matches_python_little_endian_fixture() {
        let row = EagerCouplingRow {
            real_parameter_id: MISSING_U32,
            imag_parameter_id: 0x0102_0304,
            constant_real: 3.5,
            constant_imag: -0.5,
        };
        let expected = [
            0xff, 0xff, 0xff, 0xff, 0x04, 0x03, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x0c, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0xbf,
        ];

        assert_eq!(EagerCouplingRow::ENCODED_LEN, 24);
        assert_eq!(EagerCouplingRow::encode_table(&[row]).unwrap(), expected);
        assert_eq!(EagerCouplingRow::decode_table(&expected).unwrap(), [row]);
    }

    #[test]
    fn finalization_matches_python_little_endian_fixture() {
        let row = EagerFinalizationRow {
            kernel_id: MISSING_U32,
            current_id: 0x0102_0304,
            unpropagated_value_slot_id: 0x1112_1314,
            propagated_value_slot_id: MISSING_U32,
            momentum_slot_id: 0x2122_2324,
        };
        let expected = [
            0xff, 0xff, 0xff, 0xff, 0x04, 0x03, 0x02, 0x01, 0x14, 0x13, 0x12, 0x11, 0xff, 0xff,
            0xff, 0xff, 0x24, 0x23, 0x22, 0x21,
        ];

        assert_eq!(EagerFinalizationRow::ENCODED_LEN, 20);
        assert_eq!(
            EagerFinalizationRow::encode_table(&[row]).unwrap(),
            expected
        );
        assert_eq!(
            EagerFinalizationRow::decode_table(&expected).unwrap(),
            [row]
        );
        assert!(!row.applies_kernel());
        assert!(row.stores_unpropagated());
        assert!(!row.stores_propagated());
    }

    #[test]
    fn closure_matches_python_little_endian_fixture() {
        let row = EagerClosureRow {
            kernel_id: 0x0102_0304,
            left_value_slot_id: 0x1112_1314,
            right_value_slot_id: 0x2122_2324,
            amplitude_index: 0x3132_3334,
            coupling_slot_id: 0x4142_4344,
            output_factor_source: EAGER_OUTPUT_FACTOR_COUPLING_REAL,
            factor_real: 1.5,
            factor_imag: -2.25,
        };
        let expected = [
            0x04, 0x03, 0x02, 0x01, 0x14, 0x13, 0x12, 0x11, 0x24, 0x23, 0x22, 0x21, 0x34, 0x33,
            0x32, 0x31, 0x44, 0x43, 0x42, 0x41, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0xf8, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xc0,
        ];

        assert_eq!(EagerClosureRow::ENCODED_LEN, 40);
        assert_eq!(EagerClosureRow::encode_table(&[row]).unwrap(), expected);
        assert_eq!(EagerClosureRow::decode_table(&expected).unwrap(), [row]);
    }

    #[test]
    fn round_trips_multiple_rows_and_boundary_values() {
        let invocations = [
            EagerInvocationRow {
                kernel_id: 0,
                left_value_slot_id: MISSING_U32,
                right_value_slot_id: 0,
                left_momentum_slot_id: MISSING_U32,
                right_momentum_slot_id: 0,
                coupling_slot_id: MISSING_U32,
                output_factor_source: EAGER_OUTPUT_FACTOR_NONE,
                attachment_start: 0,
                attachment_count: u64::MAX,
            },
            EagerInvocationRow {
                kernel_id: MISSING_U32,
                left_value_slot_id: 0,
                right_value_slot_id: MISSING_U32,
                left_momentum_slot_id: 0,
                right_momentum_slot_id: MISSING_U32,
                coupling_slot_id: 0,
                output_factor_source: EAGER_OUTPUT_FACTOR_COUPLING_REAL,
                attachment_start: u64::MAX,
                attachment_count: 0,
            },
        ];
        let attachments = [EagerAttachmentRow {
            result_current_id: MISSING_U32,
            factor_real: f64::MAX,
            factor_imag: f64::MIN,
        }];
        let couplings = [EagerCouplingRow {
            real_parameter_id: MISSING_U32,
            imag_parameter_id: MISSING_U32,
            constant_real: -0.0,
            constant_imag: f64::MIN_POSITIVE,
        }];
        let finalizations = [EagerFinalizationRow {
            kernel_id: MISSING_U32,
            current_id: MISSING_U32,
            unpropagated_value_slot_id: MISSING_U32,
            propagated_value_slot_id: MISSING_U32,
            momentum_slot_id: MISSING_U32,
        }];
        let closures = [EagerClosureRow {
            kernel_id: MISSING_U32,
            left_value_slot_id: MISSING_U32,
            right_value_slot_id: MISSING_U32,
            amplitude_index: MISSING_U32,
            coupling_slot_id: MISSING_U32,
            output_factor_source: EAGER_OUTPUT_FACTOR_NONE,
            factor_real: f64::MIN_POSITIVE,
            factor_imag: -0.0,
        }];

        assert_eq!(
            EagerInvocationRow::decode_table(
                &EagerInvocationRow::encode_table(&invocations).unwrap()
            )
            .unwrap(),
            invocations
        );
        assert_eq!(
            EagerAttachmentRow::decode_table(
                &EagerAttachmentRow::encode_table(&attachments).unwrap()
            )
            .unwrap(),
            attachments
        );
        assert_eq!(
            EagerCouplingRow::decode_table(&EagerCouplingRow::encode_table(&couplings).unwrap())
                .unwrap(),
            couplings
        );
        assert_eq!(
            EagerFinalizationRow::decode_table(
                &EagerFinalizationRow::encode_table(&finalizations).unwrap()
            )
            .unwrap(),
            finalizations
        );
        assert_eq!(
            EagerClosureRow::decode_table(&EagerClosureRow::encode_table(&closures).unwrap())
                .unwrap(),
            closures
        );
    }

    #[test]
    fn empty_tables_round_trip() {
        assert!(EagerInvocationRow::encode_table(&[]).unwrap().is_empty());
        assert!(EagerInvocationRow::decode_table(&[]).unwrap().is_empty());
        assert!(EagerCouplingRow::decode_table(&[]).unwrap().is_empty());
    }

    #[test]
    fn truncated_payloads_are_artifact_errors() {
        let cases = [
            EagerInvocationRow::decode_table(&[0; EagerInvocationRow::ENCODED_LEN - 1])
                .unwrap_err(),
            EagerAttachmentRow::decode_table(&[0; EagerAttachmentRow::ENCODED_LEN - 1])
                .unwrap_err(),
            EagerCouplingRow::decode_table(&[0; EagerCouplingRow::ENCODED_LEN - 1]).unwrap_err(),
            EagerFinalizationRow::decode_table(&[0; EagerFinalizationRow::ENCODED_LEN - 1])
                .unwrap_err(),
            EagerClosureRow::decode_table(&[0; EagerClosureRow::ENCODED_LEN - 1]).unwrap_err(),
        ];

        for error in cases {
            assert_eq!(error.kind(), RusticolErrorKind::Artifact);
            assert!(error.to_string().contains("not a multiple"));
        }
    }

    #[test]
    fn non_finite_outgoing_factors_are_invalid_arguments() {
        let attachment = EagerAttachmentRow {
            result_current_id: 0,
            factor_real: f64::NAN,
            factor_imag: 0.0,
        };
        let coupling = EagerCouplingRow {
            real_parameter_id: MISSING_U32,
            imag_parameter_id: MISSING_U32,
            constant_real: 0.0,
            constant_imag: f64::INFINITY,
        };
        let closure = EagerClosureRow {
            kernel_id: 0,
            left_value_slot_id: 0,
            right_value_slot_id: 0,
            amplitude_index: 0,
            coupling_slot_id: 0,
            output_factor_source: EAGER_OUTPUT_FACTOR_NONE,
            factor_real: f64::NEG_INFINITY,
            factor_imag: 0.0,
        };

        for error in [
            EagerAttachmentRow::encode_table(&[attachment]).unwrap_err(),
            EagerCouplingRow::encode_table(&[coupling]).unwrap_err(),
            EagerClosureRow::encode_table(&[closure]).unwrap_err(),
        ] {
            assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
            assert!(error.to_string().contains("non-finite"));
        }
    }

    #[test]
    fn non_finite_payload_factors_are_artifact_errors() {
        let mut attachment = vec![0; EagerAttachmentRow::ENCODED_LEN];
        attachment[4..12].copy_from_slice(&f64::INFINITY.to_le_bytes());
        let mut coupling = vec![0; EagerCouplingRow::ENCODED_LEN];
        coupling[8..16].copy_from_slice(&f64::NAN.to_le_bytes());
        let mut closure = vec![0; EagerClosureRow::ENCODED_LEN];
        closure[24..32].copy_from_slice(&f64::NEG_INFINITY.to_le_bytes());

        for error in [
            EagerAttachmentRow::decode_table(&attachment).unwrap_err(),
            EagerCouplingRow::decode_table(&coupling).unwrap_err(),
            EagerClosureRow::decode_table(&closure).unwrap_err(),
        ] {
            assert_eq!(error.kind(), RusticolErrorKind::Artifact);
            assert!(error.to_string().contains("non-finite"));
        }
    }
}
