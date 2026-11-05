// SPDX-License-Identifier: 0BSD

use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};

use super::*;
use crate::pacbin::{
    PacbinMemberKind, PacbinReader, PacbinWriteMember, PacbinWriteOptions, write_pacbin_atomic,
};
use crate::recurrence::direct_plan::tests::valid_plan;

fn temporary_directory(label: &str) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let module = module_path!().replace("::", "-");
    let path = std::env::temp_dir().join(format!(
        "rusticol-recurrence-direct-{module}-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    path
}

fn projection_certificate() -> Vec<u8> {
    let mut body = COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC.to_vec();
    body.extend_from_slice(&1_u32.to_le_bytes());
    body.extend_from_slice(b"deterministic-structural-proof");
    let digest: [u8; 32] = Sha256::digest(&body).into();
    body.extend_from_slice(&digest);
    body
}

#[test]
fn direct_pacbin_round_trips_with_one_aligned_indexed_member() {
    let directory = temporary_directory("roundtrip");
    let path = directory.join("recurrence-runtime.pacbin");
    let plan = valid_plan();
    let metadata = write_recurrence_direct_plan_pacbin(&path, &plan).unwrap();
    assert_eq!(load_recurrence_direct_plan_pacbin(&path).unwrap(), plan);

    let reader = PacbinReader::open(&path).unwrap();
    let member = reader.member(RECURRENCE_DIRECT_SCHEDULE_MEMBER).unwrap();
    assert_eq!(member.kind(), PacbinMemberKind::RecurrenceDirectPlan);
    assert_eq!(member.offset() % 64, 0);
    assert_eq!(member.length(), metadata.plan_payload_size);
    assert_eq!(member.sha256(), &metadata.plan_sha256);
    assert_eq!(
        reader
            .member_range(RECURRENCE_DIRECT_SCHEDULE_MEMBER, 0, 8)
            .unwrap(),
        b"PACRDAP2"
    );
    let retained: Vec<_> = fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(
        retained,
        [std::ffi::OsString::from("recurrence-runtime.pacbin")]
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn direct_pacbin_is_deterministic() {
    let directory = temporary_directory("deterministic");
    let first = directory.join("first.pacbin");
    let second = directory.join("second.pacbin");
    write_recurrence_direct_plan_pacbin(&first, &valid_plan()).unwrap();
    write_recurrence_direct_plan_pacbin(&second, &valid_plan()).unwrap();
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn direct_pacbin_removes_the_streaming_temporary_payload_on_error() {
    let directory = temporary_directory("stream-cleanup");
    let path = directory.join("recurrence-runtime.pacbin");
    let error = write_recurrence_direct_plan_pacbin_with_projection_certificate(
        &path,
        &valid_plan(),
        Some(b"invalid certificate"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("certificate"));
    assert_eq!(fs::read_dir(&directory).unwrap().count(), 0);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn direct_pacbin_round_trips_with_a_structural_projection_certificate() {
    let directory = temporary_directory("projection-certificate");
    let path = directory.join("recurrence-runtime.pacbin");
    let plan = valid_plan();
    let certificate = projection_certificate();
    let metadata = write_recurrence_direct_plan_pacbin_with_projection_certificate(
        &path,
        &plan,
        Some(&certificate),
    )
    .unwrap();
    assert_eq!(metadata.member_count, 2);
    assert_eq!(
        metadata.projection_certificate_payload_size,
        Some(certificate.len() as u64)
    );
    assert_eq!(load_recurrence_direct_plan_pacbin(&path).unwrap(), plan);
    let reader = PacbinReader::open(&path).unwrap();
    let member = reader
        .member(RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER)
        .unwrap();
    assert_eq!(
        member.kind(),
        PacbinMemberKind::RecurrenceColorProjectionCertificate
    );
    assert_eq!(
        reader
            .member_bytes(RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER)
            .unwrap(),
        certificate
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn direct_pacbin_rejects_a_corrupt_projection_certificate() {
    let directory = temporary_directory("projection-certificate-corrupt");
    let path = directory.join("recurrence-runtime.pacbin");
    let payload = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    let mut certificate = projection_certificate();
    certificate[COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC.len() + 12] ^= 1;
    write_pacbin_atomic(
        &path,
        [
            PacbinWriteMember::from_bytes(
                RECURRENCE_DIRECT_SCHEDULE_MEMBER,
                PacbinMemberKind::RecurrenceDirectPlan,
                &payload,
            )
            .unwrap(),
            PacbinWriteMember::from_bytes(
                RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER,
                PacbinMemberKind::RecurrenceColorProjectionCertificate,
                &certificate,
            )
            .unwrap(),
        ],
        PacbinWriteOptions::default(),
    )
    .unwrap();
    assert!(load_recurrence_direct_plan_pacbin(&path).is_err());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn direct_pacbin_rejects_truncation_and_payload_corruption() {
    let directory = temporary_directory("corruption");
    let path = directory.join("recurrence-runtime.pacbin");
    write_recurrence_direct_plan_pacbin(&path, &valid_plan()).unwrap();
    let original = fs::read(&path).unwrap();

    fs::write(&path, &original[..original.len() - 1]).unwrap();
    assert!(load_recurrence_direct_plan_pacbin(&path).is_err());

    fs::write(&path, &original).unwrap();
    let reader = PacbinReader::open(&path).unwrap();
    let payload_offset = usize::try_from(
        reader
            .member(RECURRENCE_DIRECT_SCHEDULE_MEMBER)
            .unwrap()
            .offset(),
    )
    .unwrap();
    drop(reader);
    let mut corrupted = original;
    corrupted[payload_offset + 20] ^= 1;
    fs::write(&path, corrupted).unwrap();
    assert!(load_recurrence_direct_plan_pacbin(&path).is_err());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn direct_pacbin_rejects_old_v1_path_without_aliasing() {
    let directory = temporary_directory("old-path");
    let path = directory.join("recurrence-runtime.pacbin");
    let payload = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    write_pacbin_atomic(
        &path,
        [PacbinWriteMember::from_bytes(
            "plan/recurrence-plan-v1.bin",
            PacbinMemberKind::RecurrenceDirectPlan,
            &payload,
        )
        .unwrap()],
        PacbinWriteOptions::default(),
    )
    .unwrap();
    let error = load_recurrence_direct_plan_pacbin(&path).unwrap_err();
    assert!(error.to_string().contains("regenerate with direct-plan v2"));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn direct_pacbin_rejects_wrong_member_kind_and_extra_members() {
    let directory = temporary_directory("allowlist");
    let path = directory.join("recurrence-runtime.pacbin");
    let payload = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    write_pacbin_atomic(
        &path,
        [PacbinWriteMember::from_bytes(
            RECURRENCE_DIRECT_SCHEDULE_MEMBER,
            PacbinMemberKind::EagerRuntimeTable,
            &payload,
        )
        .unwrap()],
        PacbinWriteOptions::default(),
    )
    .unwrap();
    assert!(
        load_recurrence_direct_plan_pacbin(&path)
            .unwrap_err()
            .to_string()
            .contains("wrong PACBIN member kind")
    );

    write_pacbin_atomic(
        &path,
        [
            PacbinWriteMember::from_bytes(
                RECURRENCE_DIRECT_SCHEDULE_MEMBER,
                PacbinMemberKind::RecurrenceDirectPlan,
                &payload,
            )
            .unwrap(),
            PacbinWriteMember::from_bytes(
                "metadata/extra.bin",
                PacbinMemberKind::EagerRuntimeMetadata,
                b"unexpected",
            )
            .unwrap(),
        ],
        PacbinWriteOptions::default(),
    )
    .unwrap();
    assert!(
        load_recurrence_direct_plan_pacbin(&path)
            .unwrap_err()
            .to_string()
            .contains("unsupported second member")
    );
    fs::remove_dir_all(directory).unwrap();
}
