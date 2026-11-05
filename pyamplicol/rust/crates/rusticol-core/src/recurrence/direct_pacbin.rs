// SPDX-License-Identifier: 0BSD

//! PACBIN publication and authenticated loading for direct-plan v2.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[cfg(test)]
use super::direct_codec::encode_recurrence_direct_plan_v2;
use super::direct_codec::{
    decode_recurrence_direct_plan_v2, encode_recurrence_direct_plan_v2_to_writer,
};
use super::direct_plan::DirectRecurrencePlan;
use crate::pacbin::{
    PACBIN_DEFAULT_CHUNK_SIZE, PacbinMemberKind, PacbinReader, PacbinWriteMember,
    PacbinWriteOptions, create_temporary_file, write_pacbin_atomic,
};
use crate::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};

pub const RECURRENCE_DIRECT_SCHEDULE_MEMBER: &str = "schedule/recurrence-direct-schedule-v2.bin";
pub const RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER: &str =
    "proof/recurrence-color-projection-v1.bin";
const COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC: &[u8] = b"PYAMP-COLOR-PROJECTION-BODY-V1\0\0";

struct TemporaryPlanPayload(PathBuf);

impl Drop for TemporaryPlanPayload {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceDirectPacbinMetadata {
    pub container_size: u64,
    pub member_count: u64,
    pub unpacked_size_bytes: u64,
    pub index_sha256: [u8; 32],
    pub plan_payload_size: u64,
    pub plan_sha256: [u8; 32],
    pub projection_certificate_payload_size: Option<u64>,
    pub projection_certificate_sha256: Option<[u8; 32]>,
}

pub fn write_recurrence_direct_plan_pacbin(
    destination: impl AsRef<Path>,
    plan: &DirectRecurrencePlan,
) -> RusticolResult<RecurrenceDirectPacbinMetadata> {
    write_recurrence_direct_plan_pacbin_with_projection_certificate(destination, plan, None)
}

pub fn write_recurrence_direct_plan_pacbin_with_projection_certificate(
    destination: impl AsRef<Path>,
    plan: &DirectRecurrencePlan,
    projection_certificate: Option<&[u8]>,
) -> RusticolResult<RecurrenceDirectPacbinMetadata> {
    let destination = destination.as_ref();
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (payload_path, payload_file) = create_temporary_file(destination, parent)?;
    let payload_guard = TemporaryPlanPayload(payload_path);
    let mut payload_writer = BufWriter::with_capacity(PACBIN_DEFAULT_CHUNK_SIZE, payload_file);
    let encoded_payload_size =
        encode_recurrence_direct_plan_v2_to_writer(plan, &mut payload_writer)?;
    payload_writer.flush().map_err(|error| {
        RusticolError::artifact(format!(
            "could not flush recurrence direct-plan temporary payload {}: {error}",
            payload_guard.0.display()
        ))
    })?;
    drop(payload_writer);
    let plan_member = PacbinWriteMember::from_path(
        RECURRENCE_DIRECT_SCHEDULE_MEMBER,
        PacbinMemberKind::RecurrenceDirectPlan,
        &payload_guard.0,
    )?;
    let mut members = vec![plan_member];
    if let Some(certificate) = projection_certificate {
        validate_recurrence_color_projection_certificate(certificate)?;
        members.push(PacbinWriteMember::from_bytes(
            RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER,
            PacbinMemberKind::RecurrenceColorProjectionCertificate,
            certificate,
        )?);
    }
    let index = write_pacbin_atomic(destination, members, PacbinWriteOptions::default())?;
    let indexed = index
        .members()
        .iter()
        .find(|member| member.logical_path() == RECURRENCE_DIRECT_SCHEDULE_MEMBER)
        .ok_or_else(|| RusticolError::artifact("direct recurrence PACBIN has no plan member"))?;
    let projection_indexed = index
        .members()
        .iter()
        .find(|member| member.logical_path() == RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER);
    if indexed.length() != encoded_payload_size {
        return Err(RusticolError::artifact(format!(
            "direct recurrence PACBIN plan length differs from the streamed payload: encoded={encoded_payload_size}, indexed={}",
            indexed.length()
        )));
    }
    Ok(RecurrenceDirectPacbinMetadata {
        container_size: index.file_size(),
        member_count: index.members().len() as u64,
        unpacked_size_bytes: index.members().iter().map(|member| member.length()).sum(),
        index_sha256: *index.index_sha256(),
        plan_payload_size: indexed.length(),
        plan_sha256: *indexed.sha256(),
        projection_certificate_payload_size: projection_indexed.map(|member| member.length()),
        projection_certificate_sha256: projection_indexed.map(|member| *member.sha256()),
    })
}

fn validate_projection_certificate_body(body: &[u8]) -> RusticolResult<()> {
    let minimum = COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC.len() + 4 + 32;
    if body.len() < minimum
        || !body.starts_with(COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC)
        || body[COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC.len()
            ..COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC.len() + 4]
            != 1_u32.to_le_bytes()
    {
        return Err(RusticolError::integrity(
            "recurrence color-projection certificate body has invalid framing",
        ));
    }
    let digest_start = body.len() - 32;
    let actual: [u8; 32] = Sha256::digest(&body[..digest_start]).into();
    if body[digest_start..] != actual {
        return Err(RusticolError::integrity(
            "recurrence color-projection certificate body digest mismatch",
        ));
    }
    Ok(())
}

pub(crate) fn validate_recurrence_color_projection_certificate(
    payload: &[u8],
) -> RusticolResult<()> {
    validate_projection_certificate_body(payload)
}

pub fn load_recurrence_direct_plan_pacbin(
    source: impl AsRef<Path>,
) -> RusticolResult<DirectRecurrencePlan> {
    let reader = PacbinReader::open(source)?;
    if !matches!(reader.members().len(), 1 | 2) {
        return Err(RusticolError::compatibility(format!(
            "direct recurrence PACBIN must contain one plan and at most one projection certificate, found {} members",
            reader.members().len()
        )));
    }
    let member = reader
        .member(RECURRENCE_DIRECT_SCHEDULE_MEMBER)
        .map_err(|_| {
            RusticolError::compatibility(
                "unsupported recurrence payload; regenerate with direct-plan v2",
            )
        })?;
    if member.kind() != PacbinMemberKind::RecurrenceDirectPlan {
        return Err(RusticolError::compatibility(
            "direct recurrence plan has the wrong PACBIN member kind",
        ));
    }
    if reader.members().len() == 2 {
        let certificate = reader
            .member(RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER)
            .map_err(|_| {
                RusticolError::compatibility(
                    "direct recurrence PACBIN has an unsupported second member",
                )
            })?;
        if certificate.kind() != PacbinMemberKind::RecurrenceColorProjectionCertificate {
            return Err(RusticolError::compatibility(
                "recurrence color-projection certificate has the wrong PACBIN member kind",
            ));
        }
        validate_recurrence_color_projection_certificate(
            reader.member_bytes(RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER)?,
        )?;
    }
    decode_recurrence_direct_plan_v2(reader.member_bytes(RECURRENCE_DIRECT_SCHEDULE_MEMBER)?)
}

#[cfg(test)]
#[path = "direct_pacbin_tests.rs"]
mod tests;
