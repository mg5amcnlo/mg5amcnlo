// SPDX-License-Identifier: 0BSD

//! Authenticated compact cold-load image for recurrence artifacts.
//!
//! The image deliberately contains no copy of its own digest or the final
//! artifact ID.  Generation encodes the staged payload inventory, publishes
//! this image as one more ordinary evaluator-state payload, and lets normal
//! artifact finalization bind the image digest into the artifact ID.  At load
//! time the image record can therefore be reconstructed from the checked file
//! itself without a self-hash cycle or a broad JSON-manifest parse. The
//! authenticated image is the runtime authority for producer, process, and
//! execution metadata; the outer manifest contributes only its canonical
//! payload artifact ID on this deliberately narrow path.

use super::*;
use crate::artifact::{ArtifactRuntime, EvaluatorPayloadContainerExtension, Producer};
use crate::{ArtifactProcess, Payload};
use bincode::Decode;
#[cfg(any(feature = "python-generation-bridge", test))]
use bincode::Encode;
#[cfg(feature = "python-generation-bridge")]
use serde::Deserialize;
use sha2::{Digest, Sha256};
#[cfg(feature = "python-generation-bridge")]
use std::path::PathBuf;

pub(crate) const RECURRENCE_BOOTSTRAP_IMAGE_ABI: &str = "pyamplicol-recurrence-bootstrap-image-v1";
const RECURRENCE_BOOTSTRAP_IMAGE_MAGIC: &[u8; 8] = b"PACRBIN1";
const RECURRENCE_BOOTSTRAP_IMAGE_VERSION: u16 = 1;
const RECURRENCE_BOOTSTRAP_IMAGE_SCHEMA_VERSION: u16 = 1;
const RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES: usize = 64;

#[cfg(feature = "python-generation-bridge")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecurrenceBootstrapBuildContextV1 {
    schema_version: u16,
    /// Private staging root used only while lowering the process-ready image.
    /// It is deliberately not serialized into the artifact.
    staged_root: PathBuf,
    producer: Producer,
    runtime: ArtifactRuntime,
    process: ArtifactProcess,
    payloads: Vec<Payload>,
    evaluator_payload_container: EvaluatorPayloadContainerExtension,
}

/// Only the authenticated, operational pieces needed to reopen one direct
/// recurrence plan.  The schedule PACBIN and process binding remain the sole
/// authorities; this value is a source-digest-bound load recipe, not a second
/// copy of the plan.
#[derive(Clone, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) struct RecurrenceReadyPlanV1 {
    pub(super) runtime_schedule: RecurrenceRuntimeContainer,
    pub(super) process_binding: RecurrenceProcessBinding,
    pub(super) runtime_container_member: RecurrenceRuntimeContainerMember,
    pub(super) color_projection_certificate: Option<RecurrenceColorProjectionCertificate>,
    pub(super) helicity_dispatch: Option<RecurrenceHelicityDispatchReference>,
}

impl RecurrenceReadyPlanV1 {
    #[cfg(feature = "python-generation-bridge")]
    pub(super) fn from_summary(summary: &RecurrencePlanSummary) -> Self {
        Self {
            runtime_schedule: summary.runtime_schedule.clone(),
            process_binding: summary.process_binding.clone(),
            runtime_container_member: summary.inspection_summary.runtime_container_member.clone(),
            color_projection_certificate: summary
                .inspection_summary
                .color_projection_certificate
                .clone(),
            helicity_dispatch: summary.helicity_dispatch.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) enum RecurrenceReadySourceFamilyV1 {
    Scalar,
    WeylFermion,
    DiracFermion,
    Vector,
    Spin2,
}

#[derive(Clone, Copy, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) enum RecurrenceReadySourceOrientationV1 {
    Particle,
    Antiparticle,
    SelfConjugate,
}

#[derive(Clone, Copy, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) enum RecurrenceReadySourceKeyV1 {
    SpinStateClass(i32),
    RuntimeVariant {
        source_row_id: u32,
        runtime_variant_id: u32,
    },
}

#[derive(Clone, Copy, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) struct RecurrenceReadySourceTemplateV1 {
    pub(super) spin_state_class: i32,
    pub(super) family: RecurrenceReadySourceFamilyV1,
    pub(super) orientation: RecurrenceReadySourceOrientationV1,
    pub(super) helicity: i32,
    pub(super) chirality: i32,
    pub(super) mass_parameter_index: Option<u32>,
}

#[derive(Clone, Copy, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) struct RecurrenceReadySourceVariantV1 {
    pub(super) key: RecurrenceReadySourceKeyV1,
    pub(super) template: RecurrenceReadySourceTemplateV1,
}

#[derive(Clone, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) struct RecurrenceReadySourceDomainV1 {
    pub(super) variants: Vec<RecurrenceReadySourceVariantV1>,
}

#[derive(Clone, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) struct RecurrenceReadyRuntimeV1 {
    pub(super) runtime_parameters: Vec<RecurrenceRuntimeParameter>,
    pub(super) prepared_parameter_defaults: Vec<[f64; 2]>,
    pub(super) parameter_projection: Vec<RecurrenceParameterProjection>,
    pub(super) source_domains: Vec<RecurrenceReadySourceDomainV1>,
    pub(super) external_is_initial: Vec<bool>,
    pub(super) particle_masses: Vec<(i32, f64)>,
    pub(super) particle_mass_parameter_names: Vec<(i32, String)>,
    pub(super) normalization: RecurrenceNormalization,
    /// Public physics-flow ordinal to the plan's construction sector.  This
    /// cannot be reconstructed for folded all-flow schedules.
    pub(super) public_flow_ids: Vec<u32>,
    /// Public IDs in the same order, retained so the compact decoder checks
    /// the sector mapping against the authenticated physics axis.
    pub(super) public_flow_public_ids: Vec<String>,
    pub(super) direct_helicity_to_physics: Vec<u64>,
    pub(super) color_contraction: Option<RecurrenceColorContractionReference>,
}

#[derive(Clone, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(super) struct RecurrenceReadyCompanionV1 {
    pub(super) process_digest: String,
    pub(super) plan: RecurrenceReadyPlanV1,
    pub(super) source_domains: Vec<RecurrenceReadySourceDomainV1>,
    pub(super) direct_helicity_to_physics: Vec<u64>,
}

#[derive(Clone, Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
pub(crate) struct RecurrenceReadyExecutionV1 {
    pub(super) schema_version: u16,
    pub(super) required_runtime_capabilities: Vec<String>,
    pub(super) process: String,
    pub(super) key: String,
    pub(super) color_accuracy: String,
    pub(super) external_pdg_order: Vec<i32>,
    pub(super) compiled_model_digest: String,
    pub(super) recurrence_template_catalog_digest: String,
    pub(super) prepared_kernel_pack_digest: String,
    pub(super) direct_template_catalog_digest: String,
    pub(super) kernel_pack: RecurrenceKernelPackReference,
    pub(super) primary_plan: RecurrenceReadyPlanV1,
    pub(super) runtime: RecurrenceReadyRuntimeV1,
    pub(super) companion: Option<RecurrenceReadyCompanionV1>,
}

fn validate_ready_execution_outer(
    ready: &RecurrenceReadyExecutionV1,
    outer: &ArtifactProcess,
) -> RusticolResult<()> {
    if ready.schema_version != RECURRENCE_BOOTSTRAP_IMAGE_SCHEMA_VERSION
        || ready.process != outer.expression
        || ready.key != outer.id
        || ready.color_accuracy != outer.color_accuracy
        || ready.external_pdg_order != outer.external_pdgs
    {
        return Err(RusticolError::integrity(format!(
            "recurrence process-ready recipe does not match outer process {:?}",
            outer.id
        )));
    }
    if ready.kernel_pack.manifest_path != RECURRENCE_KERNEL_PACK_MANIFEST_PATH
        || ready.kernel_pack.payload_root != RECURRENCE_KERNEL_PAYLOAD_ROOT
    {
        return Err(RusticolError::security(
            "recurrence process-ready recipe has non-canonical prepared-pack paths",
        ));
    }
    for (label, digest) in [
        ("compiled model", &ready.compiled_model_digest),
        (
            "recurrence-template catalog",
            &ready.recurrence_template_catalog_digest,
        ),
        ("prepared-kernel pack", &ready.prepared_kernel_pack_digest),
        (
            "direct-template catalog",
            &ready.direct_template_catalog_digest,
        ),
    ] {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(RusticolError::integrity(format!(
                "recurrence process-ready {label} digest is not canonical SHA-256"
            )));
        }
    }
    validate_capability_list_match(
        &outer.required_runtime_capabilities,
        &ready.required_runtime_capabilities,
        "outer process and recurrence process-ready recipe",
    )
}

#[derive(Debug, Decode)]
#[cfg_attr(any(feature = "python-generation-bridge", test), derive(Encode))]
struct RecurrenceBootstrapImageBodyV1 {
    schema_version: u16,
    producer: Producer,
    runtime: ArtifactRuntime,
    process: ArtifactProcess,
    /// Ordinary payload declarations before this image is registered.
    payloads: Vec<Payload>,
    evaluator_payload_container: EvaluatorPayloadContainerExtension,
    execution_path: String,
    execution_sha256: String,
    physics_path: String,
    physics_sha256: String,
    ready_execution: RecurrenceReadyExecutionV1,
    physics_binary: Vec<u8>,
}

pub(crate) struct DecodedRecurrenceBootstrapV1 {
    pub(crate) producer: Producer,
    pub(crate) runtime: ArtifactRuntime,
    pub(crate) process: ArtifactProcess,
    pub(crate) payloads: Vec<Payload>,
    pub(crate) evaluator_payload_container: EvaluatorPayloadContainerExtension,
    pub(crate) execution_path: String,
    pub(crate) execution_sha256: String,
    pub(crate) physics_path: String,
    pub(crate) physics_sha256: String,
    pub(crate) ready_execution: RecurrenceReadyExecutionV1,
    pub(crate) physics: ProcessPhysicsV1,
}

#[cfg(feature = "python-generation-bridge")]
pub(crate) fn build_recurrence_bootstrap_image_v1(
    context_json: &[u8],
    execution_json: &[u8],
    physics_json: &[u8],
) -> RusticolResult<Vec<u8>> {
    let context: RecurrenceBootstrapBuildContextV1 =
        serde_json::from_slice(context_json).map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse recurrence bootstrap build context: {error}"
            ))
        })?;
    if context.schema_version != RECURRENCE_BOOTSTRAP_IMAGE_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence bootstrap build-context schema {}; expected {}",
            context.schema_version, RECURRENCE_BOOTSTRAP_IMAGE_SCHEMA_VERSION
        )));
    }
    let execution_path = format!("processes/{}/execution.json", context.process.id);
    let execution =
        parse_recurrence_execution_manifest(execution_json, &execution_path, &context.process)?;
    let physics_path = context.process.physics_path.clone();
    let physics = ProcessPhysicsV1::from_json(physics_json, &physics_path)?;
    validate_bootstrap_representative_physics(&physics, &context.process)?;
    let execution_sha256 = require_build_source_payload(
        &context.payloads,
        &execution_path,
        PayloadRole::EvaluatorManifest,
        &context.process.id,
        execution_json,
    )?
    .sha256
    .clone();
    let physics_sha256 = require_build_source_payload(
        &context.payloads,
        &physics_path,
        PayloadRole::RuntimePhysics,
        &context.process.id,
        physics_json,
    )?
    .sha256
    .clone();
    let staged_artifact = crate::artifact::open_staged_recurrence_bootstrap_artifact(
        context.staged_root,
        context.producer.clone(),
        context.runtime.clone(),
        context.process.clone(),
        context.payloads.clone(),
        context.evaluator_payload_container.clone(),
    )?;
    let evaluator_root = staged_artifact
        .root()
        .join("processes")
        .join(&context.process.id);
    let ready_execution = build_recurrence_ready_execution_v1(
        &staged_artifact,
        &evaluator_root,
        &execution,
        &physics,
    )?;
    let body = RecurrenceBootstrapImageBodyV1 {
        schema_version: RECURRENCE_BOOTSTRAP_IMAGE_SCHEMA_VERSION,
        producer: context.producer,
        runtime: context.runtime,
        process: context.process,
        payloads: context.payloads,
        evaluator_payload_container: context.evaluator_payload_container,
        execution_path,
        execution_sha256,
        physics_path,
        physics_sha256,
        ready_execution,
        physics_binary: encode_recurrence_bootstrap_physics_v1(&physics)?,
    };
    encode_image_body(&body)
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
pub(crate) fn decode_recurrence_bootstrap_image_v1(
    bytes: &[u8],
) -> RusticolResult<DecodedRecurrenceBootstrapV1> {
    let body_bytes = checked_image_body(bytes)?;
    let (body, consumed): (RecurrenceBootstrapImageBodyV1, usize) =
        bincode::decode_from_slice(body_bytes, bincode::config::standard()).map_err(|error| {
            RusticolError::serialization(format!(
                "could not decode {RECURRENCE_BOOTSTRAP_IMAGE_ABI}: {error}"
            ))
        })?;
    if consumed != body_bytes.len() {
        return Err(RusticolError::serialization(format!(
            "{RECURRENCE_BOOTSTRAP_IMAGE_ABI} contains {} trailing body bytes",
            body_bytes.len() - consumed
        )));
    }
    if body.schema_version != RECURRENCE_BOOTSTRAP_IMAGE_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence bootstrap body schema {}; expected {}",
            body.schema_version, RECURRENCE_BOOTSTRAP_IMAGE_SCHEMA_VERSION
        )));
    }
    let physics = decode_recurrence_bootstrap_physics_v1(&body.physics_binary)?;
    validate_bootstrap_representative_physics(&physics, &body.process)?;
    validate_ready_execution_outer(&body.ready_execution, &body.process)?;
    Ok(DecodedRecurrenceBootstrapV1 {
        producer: body.producer,
        runtime: body.runtime,
        process: body.process,
        payloads: body.payloads,
        evaluator_payload_container: body.evaluator_payload_container,
        execution_path: body.execution_path,
        execution_sha256: body.execution_sha256,
        physics_path: body.physics_path,
        physics_sha256: body.physics_sha256,
        ready_execution: body.ready_execution,
        physics,
    })
}

#[cfg(feature = "python-generation-bridge")]
fn require_build_source_payload<'a>(
    payloads: &'a [Payload],
    path: &str,
    role: PayloadRole,
    process_id: &str,
    bytes: &[u8],
) -> RusticolResult<&'a Payload> {
    let payload = payloads
        .iter()
        .find(|payload| payload.path == path)
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "recurrence bootstrap source {path:?} is absent from the staged payload inventory"
            ))
        })?;
    let size = u64::try_from(bytes.len()).map_err(|_| {
        RusticolError::artifact(format!(
            "recurrence bootstrap source {path:?} exceeds the u64 size domain"
        ))
    })?;
    let digest = format!("{:x}", Sha256::digest(bytes));
    if payload.role != role
        || payload.media_type != "application/json"
        || payload.process_id.as_deref() != Some(process_id)
        || payload.executable
        || payload.size_bytes != size
        || payload.sha256 != digest
    {
        return Err(RusticolError::integrity(format!(
            "recurrence bootstrap source {path:?} disagrees with its staged payload declaration"
        )));
    }
    Ok(payload)
}

fn validate_bootstrap_representative_physics(
    physics: &ProcessPhysicsV1,
    process: &ArtifactProcess,
) -> RusticolResult<()> {
    if physics.process_id != process.id
        || physics.process != process.expression
        || physics.color_accuracy.as_str() != process.color_accuracy
        || physics
            .external_particles
            .iter()
            .map(|particle| particle.pdg)
            .ne(process.external_pdgs.iter().copied())
    {
        return Err(RusticolError::integrity(format!(
            "recurrence bootstrap physics does not match process {:?}",
            process.id
        )));
    }
    Ok(())
}

#[cfg(feature = "python-generation-bridge")]
fn encode_image_body(body: &RecurrenceBootstrapImageBodyV1) -> RusticolResult<Vec<u8>> {
    let encoded = bincode::encode_to_vec(body, bincode::config::standard()).map_err(|error| {
        RusticolError::serialization(format!(
            "could not encode {RECURRENCE_BOOTSTRAP_IMAGE_ABI}: {error}"
        ))
    })?;
    let body_len = u64::try_from(encoded.len()).map_err(|_| {
        RusticolError::artifact(format!(
            "{RECURRENCE_BOOTSTRAP_IMAGE_ABI} body length exceeds u64"
        ))
    })?;
    let total_len = RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES
        .checked_add(encoded.len())
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "{RECURRENCE_BOOTSTRAP_IMAGE_ABI} total length overflows usize"
            ))
        })?;
    let mut image = Vec::new();
    image.try_reserve_exact(total_len).map_err(|error| {
        RusticolError::artifact(format!(
            "could not allocate {total_len} bytes for {RECURRENCE_BOOTSTRAP_IMAGE_ABI}: {error}"
        ))
    })?;
    image.extend_from_slice(RECURRENCE_BOOTSTRAP_IMAGE_MAGIC);
    image.extend_from_slice(&RECURRENCE_BOOTSTRAP_IMAGE_VERSION.to_le_bytes());
    image.extend_from_slice(&0_u16.to_le_bytes());
    image.extend_from_slice(&(RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES as u32).to_le_bytes());
    image.extend_from_slice(&body_len.to_le_bytes());
    image.extend_from_slice(&Sha256::digest(&encoded));
    image.extend_from_slice(&[0_u8; 8]);
    debug_assert_eq!(image.len(), RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES);
    image.extend_from_slice(&encoded);
    Ok(image)
}

fn checked_image_body(bytes: &[u8]) -> RusticolResult<&[u8]> {
    let header = bytes
        .get(..RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES)
        .ok_or_else(|| {
            RusticolError::serialization(format!(
                "truncated {RECURRENCE_BOOTSTRAP_IMAGE_ABI} header"
            ))
        })?;
    if header.get(..8) != Some(RECURRENCE_BOOTSTRAP_IMAGE_MAGIC.as_slice()) {
        return Err(RusticolError::compatibility(format!(
            "unsupported recurrence bootstrap magic; expected {RECURRENCE_BOOTSTRAP_IMAGE_ABI}"
        )));
    }
    let version = u16::from_le_bytes(header[8..10].try_into().expect("fixed version field"));
    let flags = u16::from_le_bytes(header[10..12].try_into().expect("fixed flags field"));
    let header_bytes = u32::from_le_bytes(
        header[12..16]
            .try_into()
            .expect("fixed header-length field"),
    );
    if version != RECURRENCE_BOOTSTRAP_IMAGE_VERSION
        || flags != 0
        || header_bytes != RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES as u32
        || header[56..64].iter().any(|byte| *byte != 0)
    {
        return Err(RusticolError::compatibility(format!(
            "unsupported {RECURRENCE_BOOTSTRAP_IMAGE_ABI} version/flags/header"
        )));
    }
    let body_len = usize::try_from(u64::from_le_bytes(
        header[16..24].try_into().expect("fixed body-length field"),
    ))
    .map_err(|_| {
        RusticolError::artifact(format!(
            "{RECURRENCE_BOOTSTRAP_IMAGE_ABI} body length exceeds usize"
        ))
    })?;
    let expected_len = RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES
        .checked_add(body_len)
        .ok_or_else(|| {
            RusticolError::artifact(format!(
                "{RECURRENCE_BOOTSTRAP_IMAGE_ABI} length overflows usize"
            ))
        })?;
    if bytes.len() != expected_len {
        return Err(RusticolError::serialization(format!(
            "{RECURRENCE_BOOTSTRAP_IMAGE_ABI} length is {}, expected {expected_len}",
            bytes.len()
        )));
    }
    let body = &bytes[RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES..];
    if Sha256::digest(body).as_slice() != &header[24..56] {
        return Err(RusticolError::integrity(format!(
            "{RECURRENCE_BOOTSTRAP_IMAGE_ABI} body SHA-256 mismatch"
        )));
    }
    Ok(body)
}

#[cfg(test)]
mod image_tests {
    use super::*;

    fn framed(body: &[u8]) -> Vec<u8> {
        let mut image = Vec::with_capacity(RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES + body.len());
        image.extend_from_slice(RECURRENCE_BOOTSTRAP_IMAGE_MAGIC);
        image.extend_from_slice(&RECURRENCE_BOOTSTRAP_IMAGE_VERSION.to_le_bytes());
        image.extend_from_slice(&0_u16.to_le_bytes());
        image.extend_from_slice(&(RECURRENCE_BOOTSTRAP_IMAGE_HEADER_BYTES as u32).to_le_bytes());
        image.extend_from_slice(&(body.len() as u64).to_le_bytes());
        image.extend_from_slice(&Sha256::digest(body));
        image.extend_from_slice(&[0_u8; 8]);
        image.extend_from_slice(body);
        image
    }

    #[test]
    fn outer_image_header_authenticates_body_and_exact_length() {
        let image = framed(b"bincode-body");
        assert_eq!(checked_image_body(&image).unwrap(), b"bincode-body");

        let mut corrupted = image.clone();
        *corrupted.last_mut().unwrap() ^= 1;
        assert_eq!(
            checked_image_body(&corrupted).unwrap_err().kind(),
            crate::RusticolErrorKind::Integrity
        );

        let mut trailing = image;
        trailing.push(0);
        assert_eq!(
            checked_image_body(&trailing).unwrap_err().kind(),
            crate::RusticolErrorKind::Serialization
        );
    }
}
