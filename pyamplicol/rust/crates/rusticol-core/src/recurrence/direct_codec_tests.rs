// SPDX-License-Identifier: 0BSD

use super::*;
use crate::recurrence::direct_plan::tests::valid_plan;
use std::io::{self, Write};

#[derive(Default)]
struct CountingWriter {
    written: u64,
}

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.written += bytes.len() as u64;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn direct_codec_round_trips_deterministically() {
    let plan = valid_plan();
    let first = encode_recurrence_direct_plan_v2(&plan).unwrap();
    let second = encode_recurrence_direct_plan_v2(&plan).unwrap();
    assert_eq!(first, second);
    let decoded = decode_recurrence_direct_plan_v2(&first).unwrap();
    assert_eq!(
        decoded
            .closure_proofs()
            .three_line_traversal_certificates()
            .len(),
        1
    );
    assert_eq!(decoded, plan);
}

#[test]
fn direct_codec_stream_matches_the_vec_adapter() {
    let plan = valid_plan();
    let expected = encode_recurrence_direct_plan_v2(&plan).unwrap();
    let mut streamed = Vec::new();
    let byte_count = encode_recurrence_direct_plan_v2_to_writer(&plan, &mut streamed).unwrap();
    assert_eq!(streamed, expected);
    assert_eq!(byte_count, expected.len() as u64);
}

#[test]
fn direct_codec_rejects_runtime_compacted_plans() {
    let complete = valid_plan();
    let runtime_layout_digest = complete.runtime_layout_digest();
    let closure_proof_digest = complete
        .closure_proofs()
        .expected_semantic_completeness_digest();
    let plan = complete.into_runtime_compacted();
    assert_eq!(plan.runtime_layout_digest(), runtime_layout_digest);
    assert_eq!(
        plan.closure_proofs()
            .expected_semantic_completeness_digest(),
        closure_proof_digest
    );
    assert!(plan.closure_proofs().contributions().is_empty());
    assert!(plan.closure_proofs().groups().is_empty());
    assert!(
        encode_recurrence_direct_plan_v2(&plan)
            .unwrap_err()
            .to_string()
            .contains("runtime-compacted plan cannot be serialized")
    );
}

#[test]
fn direct_codec_stream_crosses_the_former_eight_gib_boundary_without_allocating() {
    const EIGHT_GIB: u64 = 8 * 1024 * 1024 * 1024;
    let mut writer = Writer::with_bytes_written(CountingWriter::default(), EIGHT_GIB - 1);
    writer.raw(&[1, 2]).unwrap();
    assert_eq!(writer.bytes_written, EIGHT_GIB + 1);
    assert_eq!(writer.destination.written, 2);
}

#[test]
fn direct_codec_stream_rejects_only_true_u64_length_overflow() {
    let mut writer = Writer::with_bytes_written(CountingWriter::default(), u64::MAX);
    let error = writer.raw(&[1]).unwrap_err();
    assert!(error.to_string().contains("payload length overflows u64"));
    assert_eq!(writer.destination.written, 0);
}

#[test]
fn direct_codec_rejects_every_tested_truncation_boundary() {
    let bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    for length in [0, 1, 7, 8, 15, 64, bytes.len() / 2, bytes.len() - 1] {
        let error = decode_recurrence_direct_plan_v2(&bytes[..length]).unwrap_err();
        assert!(
            error.to_string().contains("truncated")
                || error.to_string().contains("unsupported recurrence")
                || error.to_string().contains("cannot fit"),
            "unexpected error at truncation {length}: {error}"
        );
    }
}

#[test]
fn direct_codec_rejects_old_v1_magic_explicitly() {
    let mut bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    bytes[..8].copy_from_slice(b"PACRPLAN");
    let error = decode_recurrence_direct_plan_v2(&bytes).unwrap_err();
    assert!(error.to_string().contains("regenerate with direct-plan v2"));
}

#[test]
fn direct_codec_detects_a_valid_field_mutation_through_layout_digest() {
    let mut bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    // Header offset 20 is point_tile_size. Keep it nonzero and structurally
    // valid so the authenticated layout digest is the rejecting invariant.
    bytes[20..24].copy_from_slice(&128_u32.to_le_bytes());
    let error = decode_recurrence_direct_plan_v2(&bytes).unwrap_err();
    assert!(error.to_string().contains("runtime-layout digest mismatch"));
}

#[test]
fn direct_codec_rejects_nonzero_reserved_fields() {
    let mut bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    bytes[12..16].copy_from_slice(&1_u32.to_le_bytes());
    let error = decode_recurrence_direct_plan_v2(&bytes).unwrap_err();
    assert!(error.to_string().contains("header flags"));
}

#[test]
fn direct_codec_authenticates_three_line_traversal_payloads() {
    let mut bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    let last = bytes
        .last_mut()
        .expect("encoded three-line proof payload is nonempty");
    *last ^= 1;
    let error = decode_recurrence_direct_plan_v2(&bytes).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("three-line traversal proof digest mismatch")
    );
}
