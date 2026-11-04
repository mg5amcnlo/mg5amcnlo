// SPDX-License-Identifier: 0BSD

//! Canonical private payload codec for the compact on-the-fly process seed.
//!
//! The eventual PACBIN member owns payload hashing and authentication. This
//! codec therefore carries only the seed's semantic inputs; derived semantic
//! digests are recomputed by the existing constructors during decoding. The
//! raw-amplitude normalization factor is exact one by seed construction, so
//! schema v1 implies it instead of storing a redundant field.

use std::io::{self, Write};
use std::str;

use super::source_seed::{
    OnTheFlyCouplingOrderPolicyV1, OnTheFlyExternalColorRoleV1, OnTheFlyPairingClassV1,
    OnTheFlyPairingEndpointV1, OnTheFlyProcessSeedV1, OnTheFlySourceAnchorV1,
    OnTheFlySourceOrientationV1, OnTheFlySourceStateV1, OnTheFlySourceWavefunctionFamilyV1,
};
use crate::recurrence::{ExactComplexRational, ExactRational, SemanticDigest};
use crate::{RusticolError, RusticolResult};

const MAGIC: &[u8; 8] = b"PACOTFSD";
const SCHEMA_VERSION: u32 = 1;
const SEMANTIC_HASH_REVISION: u32 = 2;
#[cfg(test)]
const RETAINED_DIGEST_COUNT: u64 = 6;
const DIGEST_BYTES: u64 = 32;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("on-the-fly process-seed codec: {}", message.into()))
}

struct Writer<W> {
    destination: W,
    bytes_written: u64,
}

impl<W: Write> Writer<W> {
    fn new(destination: W) -> Self {
        Self {
            destination,
            bytes_written: 0,
        }
    }

    #[cfg(test)]
    fn with_bytes_written(destination: W, bytes_written: u64) -> Self {
        Self {
            destination,
            bytes_written,
        }
    }

    fn raw(&mut self, bytes: &[u8]) -> RusticolResult<()> {
        let byte_count =
            u64::try_from(bytes.len()).map_err(|_| invalid("payload write length exceeds u64"))?;
        let end = self
            .bytes_written
            .checked_add(byte_count)
            .ok_or_else(|| invalid("payload byte accounting overflows u64"))?;
        self.destination.write_all(bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "could not stream on-the-fly process-seed payload: {error}"
            ))
        })?;
        self.bytes_written = end;
        Ok(())
    }

    fn u8(&mut self, value: u8) -> RusticolResult<()> {
        self.raw(&[value])
    }

    fn u32(&mut self, value: u32) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn i32(&mut self, value: i32) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn i128(&mut self, value: i128) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn count(&mut self, value: usize, label: &str) -> RusticolResult<()> {
        self.u64(u64::try_from(value).map_err(|_| invalid(format!("{label} count exceeds u64")))?)
    }

    fn boolean(&mut self, value: bool) -> RusticolResult<()> {
        self.u8(u8::from(value))
    }

    fn digest(&mut self, value: SemanticDigest) -> RusticolResult<()> {
        self.raw(value.as_bytes())
    }

    fn optional_digest(&mut self, value: Option<SemanticDigest>) -> RusticolResult<()> {
        match value {
            None => self.u8(0),
            Some(value) => {
                self.u8(1)?;
                self.digest(value)
            }
        }
    }

    fn optional_u32(&mut self, value: Option<u32>) -> RusticolResult<()> {
        match value {
            None => self.u8(0),
            Some(value) => {
                self.u8(1)?;
                self.u32(value)
            }
        }
    }

    fn exact(&mut self, value: ExactComplexRational) -> RusticolResult<()> {
        for part in [value.real(), value.imag()] {
            self.i128(part.numerator())?;
            self.i128(part.denominator())?;
        }
        Ok(())
    }

    fn string(&mut self, value: &str, label: &str) -> RusticolResult<()> {
        self.count(value.len(), label)?;
        self.raw(value.as_bytes())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: u64,
    length: u64,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> RusticolResult<Self> {
        Ok(Self {
            bytes,
            offset: 0,
            length: u64::try_from(bytes.len())
                .map_err(|_| invalid("payload length exceeds u64"))?,
        })
    }

    fn take(&mut self, count: u64, label: &str) -> RusticolResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| invalid(format!("{label} byte range overflows u64")))?;
        if end > self.length {
            return Err(invalid(format!(
                "truncated {label} at byte {}: need {count}, have {}",
                self.offset,
                self.length.saturating_sub(self.offset)
            )));
        }
        let start = usize::try_from(self.offset)
            .map_err(|_| invalid(format!("{label} offset exceeds usize")))?;
        let end = usize::try_from(end)
            .map_err(|_| invalid(format!("{label} end offset exceeds usize")))?;
        let result = self
            .bytes
            .get(start..end)
            .ok_or_else(|| invalid(format!("{label} byte range is outside the payload")))?;
        self.offset =
            u64::try_from(end).map_err(|_| invalid(format!("{label} end offset exceeds u64")))?;
        Ok(result)
    }

    fn u8(&mut self, label: &str) -> RusticolResult<u8> {
        Ok(self.take(1, label)?[0])
    }

    fn u32(&mut self, label: &str) -> RusticolResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4, label)?.try_into().expect("checked read"),
        ))
    }

    fn i32(&mut self, label: &str) -> RusticolResult<i32> {
        Ok(i32::from_le_bytes(
            self.take(4, label)?.try_into().expect("checked read"),
        ))
    }

    fn u64(&mut self, label: &str) -> RusticolResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8, label)?.try_into().expect("checked read"),
        ))
    }

    fn i128(&mut self, label: &str) -> RusticolResult<i128> {
        Ok(i128::from_le_bytes(
            self.take(16, label)?.try_into().expect("checked read"),
        ))
    }

    fn count(&mut self, minimum_item_bytes: u64, label: &str) -> RusticolResult<usize> {
        let count = self.u64(&format!("{label} count"))?;
        let minimum_bytes = count
            .checked_mul(minimum_item_bytes)
            .ok_or_else(|| invalid(format!("{label} minimum byte length overflows u64")))?;
        if minimum_bytes > self.length.saturating_sub(self.offset) {
            return Err(invalid(format!(
                "{label} count cannot fit in the remaining payload"
            )));
        }
        usize::try_from(count).map_err(|_| invalid(format!("{label} count exceeds usize")))
    }

    fn boolean(&mut self, label: &str) -> RusticolResult<bool> {
        match self.u8(label)? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(invalid(format!(
                "{label} has malformed boolean discriminant {value}"
            ))),
        }
    }

    fn digest(&mut self, label: &str) -> RusticolResult<SemanticDigest> {
        let value: [u8; 32] = self
            .take(DIGEST_BYTES, label)?
            .try_into()
            .expect("checked read");
        SemanticDigest::new(value).map_err(|error| invalid(error.message()))
    }

    fn optional_digest(&mut self, label: &str) -> RusticolResult<Option<SemanticDigest>> {
        match self.u8(&format!("{label} presence"))? {
            0 => Ok(None),
            1 => Ok(Some(self.digest(label)?)),
            value => Err(invalid(format!(
                "{label} has malformed optional discriminant {value}"
            ))),
        }
    }

    fn optional_u32(&mut self, label: &str) -> RusticolResult<Option<u32>> {
        match self.u8(&format!("{label} presence"))? {
            0 => Ok(None),
            1 => Ok(Some(self.u32(label)?)),
            value => Err(invalid(format!(
                "{label} has malformed optional discriminant {value}"
            ))),
        }
    }

    fn exact(&mut self, label: &str) -> RusticolResult<ExactComplexRational> {
        Ok(ExactComplexRational::new(
            self.rational(&format!("{label} real"))?,
            self.rational(&format!("{label} imaginary"))?,
        ))
    }

    fn rational(&mut self, label: &str) -> RusticolResult<ExactRational> {
        let numerator = self.i128(&format!("{label} numerator"))?;
        let denominator = self.i128(&format!("{label} denominator"))?;
        let value =
            ExactRational::new(numerator, denominator).map_err(|error| invalid(error.message()))?;
        if value.numerator() != numerator || value.denominator() != denominator {
            return Err(invalid(format!("{label} is not canonically reduced")));
        }
        Ok(value)
    }

    fn string(&mut self, label: &str) -> RusticolResult<String> {
        let byte_count = self.u64(&format!("{label} byte count"))?;
        let bytes = self.take(byte_count, label)?;
        let value = str::from_utf8(bytes)
            .map_err(|error| invalid(format!("{label} is not UTF-8: {error}")))?;
        let byte_count = usize::try_from(byte_count)
            .map_err(|_| invalid(format!("{label} byte count exceeds usize")))?;
        let mut owned = String::new();
        owned.try_reserve_exact(byte_count).map_err(|error| {
            invalid(format!(
                "could not reserve {byte_count} bytes for {label}: {error}"
            ))
        })?;
        owned.push_str(value);
        Ok(owned)
    }

    fn finish(self) -> RusticolResult<()> {
        if self.offset != self.length {
            return Err(invalid(format!(
                "payload contains {} trailing bytes",
                self.length - self.offset
            )));
        }
        Ok(())
    }
}

fn reserved_vec<T>(count: usize, label: &str) -> RusticolResult<Vec<T>> {
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(|error| {
        invalid(format!(
            "could not reserve {count} rows for {label}: {error}"
        ))
    })?;
    Ok(result)
}

fn encode_to_writer<W: Write>(seed: &OnTheFlyProcessSeedV1, destination: W) -> RusticolResult<u64> {
    let mut writer = Writer::new(destination);
    writer.raw(MAGIC)?;
    writer.u32(SCHEMA_VERSION)?;
    writer.u32(SEMANTIC_HASH_REVISION)?;
    for digest in [
        seed.process_digest,
        seed.model_digest,
        seed.template_catalog_digest,
        seed.prepared_pack_digest,
        seed.direct_catalog_digest,
        seed.normalization_semantic_digest,
    ] {
        writer.digest(digest)?;
    }
    writer.string(&seed.normalization_convention, "normalization convention")?;
    writer.count(seed.source_anchors.len(), "source anchors")?;
    for anchor in &seed.source_anchors {
        writer.u32(anchor.source_slot)?;
        writer.u32(anchor.external_label)?;
        writer.boolean(anchor.is_initial)?;
        writer.u8(anchor.color_role as u8)?;
        writer.boolean(anchor.is_fermionic)?;
        writer.optional_digest(anchor.pairing_source_contract_digest)?;
        writer.count(anchor.states.len(), "source states")?;
        for state in &anchor.states {
            writer.u32(state.state_index)?;
            writer.i32(state.public_helicity)?;
            writer.i32(state.source_helicity)?;
            writer.u32(state.source_template_id)?;
            writer.u32(state.current_state_template_id)?;
            writer.digest(state.source_semantic_digest)?;
            writer.digest(state.current_state_semantic_digest)?;
            writer.i32(state.momentum_sign)?;
            writer.exact(state.crossing_phase)?;
            writer.i32(state.spin_state)?;
            writer.i32(state.chirality)?;
            writer.count(state.flavour_flow.len(), "source flavour flow")?;
            for flavour in &state.flavour_flow {
                writer.i32(*flavour)?;
            }
            writer.u32(state.quantum_number_flow_id)?;
            writer.digest(state.color_seed_proof_digest)?;
            writer.u8(state.source_family as u8)?;
            writer.u8(state.source_orientation as u8)?;
            writer.optional_u32(state.prepared_mass_parameter_slot)?;
        }
    }
    writer.count(
        seed.external_permutation.len(),
        "external gather permutation",
    )?;
    for slot in &seed.external_permutation {
        writer.u32(*slot)?;
    }
    writer.u8(seed.coupling_order_policy as u8)?;
    writer.count(seed.coupling_limits.len(), "coupling policy rows")?;
    for (hierarchy, limit) in seed
        .coupling_hierarchies
        .iter()
        .zip(seed.coupling_limits.iter())
    {
        writer.u32(*hierarchy)?;
        writer.optional_u32(*limit)?;
    }
    writer.count(seed.pairing_classes.len(), "pairing classes")?;
    for pairing_class in &seed.pairing_classes {
        writer.string(&pairing_class.species, "pairing species")?;
        writer.digest(pairing_class.species_semantic_digest)?;
        for (label, endpoints) in [
            (
                "fundamental pairing endpoints",
                pairing_class.fundamental_endpoints.as_ref(),
            ),
            (
                "antifundamental pairing endpoints",
                pairing_class.antifundamental_endpoints.as_ref(),
            ),
        ] {
            writer.count(endpoints.len(), label)?;
            for endpoint in endpoints {
                writer.u32(endpoint.source_slot)?;
                writer.digest(endpoint.source_contract_digest)?;
            }
        }
    }
    Ok(writer.bytes_written)
}

/// Encode the canonical schema-v1 payload for a compact semantic seed.
///
/// This remains crate-private until the PACBIN member integration owns it.
#[allow(dead_code)]
pub(crate) fn encode_on_the_fly_process_seed_v1(
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<Vec<u8>> {
    let byte_count = encode_to_writer(seed, io::sink())?;
    let capacity =
        usize::try_from(byte_count).map_err(|_| invalid("encoded payload length exceeds usize"))?;
    let mut payload = Vec::new();
    payload.try_reserve_exact(capacity).map_err(|error| {
        invalid(format!(
            "could not reserve {capacity} bytes for encoded payload: {error}"
        ))
    })?;
    let actual = encode_to_writer(seed, &mut payload)?;
    if actual != byte_count || payload.len() != capacity {
        return Err(RusticolError::internal(
            "on-the-fly process-seed encoder produced an inconsistent byte count",
        ));
    }
    Ok(payload)
}

fn read_color_role(reader: &mut Reader<'_>) -> RusticolResult<OnTheFlyExternalColorRoleV1> {
    match reader.u8("source anchor color role")? {
        0 => Ok(OnTheFlyExternalColorRoleV1::Singlet),
        1 => Ok(OnTheFlyExternalColorRoleV1::Fundamental),
        2 => Ok(OnTheFlyExternalColorRoleV1::Antifundamental),
        3 => Ok(OnTheFlyExternalColorRoleV1::Adjoint),
        value => Err(invalid(format!(
            "malformed source anchor color-role discriminant {value}"
        ))),
    }
}

fn read_source_family(
    reader: &mut Reader<'_>,
) -> RusticolResult<OnTheFlySourceWavefunctionFamilyV1> {
    match reader.u8("source wavefunction family")? {
        0 => Ok(OnTheFlySourceWavefunctionFamilyV1::Scalar),
        1 => Ok(OnTheFlySourceWavefunctionFamilyV1::WeylFermion),
        2 => Ok(OnTheFlySourceWavefunctionFamilyV1::DiracFermion),
        3 => Ok(OnTheFlySourceWavefunctionFamilyV1::Vector),
        4 => Ok(OnTheFlySourceWavefunctionFamilyV1::Spin2),
        value => Err(invalid(format!(
            "malformed source wavefunction-family discriminant {value}"
        ))),
    }
}

fn read_source_orientation(reader: &mut Reader<'_>) -> RusticolResult<OnTheFlySourceOrientationV1> {
    match reader.u8("source orientation")? {
        0 => Ok(OnTheFlySourceOrientationV1::Particle),
        1 => Ok(OnTheFlySourceOrientationV1::Antiparticle),
        2 => Ok(OnTheFlySourceOrientationV1::SelfConjugate),
        value => Err(invalid(format!(
            "malformed source-orientation discriminant {value}"
        ))),
    }
}

fn read_coupling_order_policy(
    reader: &mut Reader<'_>,
) -> RusticolResult<OnTheFlyCouplingOrderPolicyV1> {
    match reader.u8("coupling-order policy")? {
        0 => Ok(OnTheFlyCouplingOrderPolicyV1::Minimal),
        1 => Ok(OnTheFlyCouplingOrderPolicyV1::Explicit),
        value => Err(invalid(format!(
            "malformed coupling-order policy discriminant {value}"
        ))),
    }
}

fn read_source_state(reader: &mut Reader<'_>) -> RusticolResult<OnTheFlySourceStateV1> {
    let state_index = reader.u32("source state index")?;
    let public_helicity = reader.i32("source state public helicity")?;
    let source_helicity = reader.i32("source state source helicity")?;
    let source_template_id = reader.u32("source state source-template ID")?;
    let current_state_template_id = reader.u32("source state current-template ID")?;
    let source_semantic_digest = reader.digest("source state source semantic digest")?;
    let current_state_semantic_digest = reader.digest("source state current semantic digest")?;
    let momentum_sign = reader.i32("source state momentum sign")?;
    let crossing_phase = reader.exact("source state crossing phase")?;
    let spin_state = reader.i32("source state spin state")?;
    let chirality = reader.i32("source state chirality")?;
    let flavour_count = reader.count(4, "source flavour flow")?;
    let mut flavour_flow = reserved_vec(flavour_count, "source flavour flow")?;
    for _ in 0..flavour_count {
        flavour_flow.push(reader.i32("source flavour-flow entry")?);
    }
    let quantum_number_flow_id = reader.u32("source state quantum-number flow ID")?;
    let color_seed_proof_digest = reader.digest("source state color-seed proof digest")?;
    let source_family = read_source_family(reader)?;
    let source_orientation = read_source_orientation(reader)?;
    let prepared_mass_parameter_slot =
        reader.optional_u32("source state prepared-mass parameter slot")?;
    OnTheFlySourceStateV1::new(
        state_index,
        public_helicity,
        source_helicity,
        source_template_id,
        current_state_template_id,
        source_semantic_digest,
        current_state_semantic_digest,
        momentum_sign,
        crossing_phase,
        spin_state,
        chirality,
        flavour_flow,
        quantum_number_flow_id,
        color_seed_proof_digest,
        source_family,
        source_orientation,
        prepared_mass_parameter_slot,
    )
}

fn read_pairing_endpoints(
    reader: &mut Reader<'_>,
    label: &str,
) -> RusticolResult<Vec<OnTheFlyPairingEndpointV1>> {
    const MINIMUM_ENDPOINT_BYTES: u64 = 4 + DIGEST_BYTES;
    let count = reader.count(MINIMUM_ENDPOINT_BYTES, label)?;
    let mut endpoints = reserved_vec(count, label)?;
    let mut previous_slot = None;
    for _ in 0..count {
        let source_slot = reader.u32(&format!("{label} source slot"))?;
        if previous_slot.is_some_and(|previous| previous >= source_slot) {
            return Err(invalid(format!(
                "{label} keys are not in canonical strict order"
            )));
        }
        previous_slot = Some(source_slot);
        endpoints.push(OnTheFlyPairingEndpointV1 {
            source_slot,
            source_contract_digest: reader.digest(&format!("{label} source-contract digest"))?,
        });
    }
    Ok(endpoints)
}

/// Decode exactly one canonical schema-v1 compact semantic seed payload.
///
/// Version mismatches are rejected rather than guessed, leaving a clean seam
/// for a future schema that carries NLC/full-color seed fields.
#[allow(dead_code)]
pub(crate) fn decode_on_the_fly_process_seed_v1(
    bytes: &[u8],
) -> RusticolResult<OnTheFlyProcessSeedV1> {
    const MINIMUM_STATE_BYTES: u64 =
        5 * 4 + 2 * DIGEST_BYTES + 4 + 4 * 16 + 2 * 4 + 8 + 4 + DIGEST_BYTES + 1 + 1 + 1 + 4;
    const MINIMUM_ANCHOR_BYTES: u64 = 4 + 4 + 1 + 1 + 1 + 1 + 8 + MINIMUM_STATE_BYTES;
    const MINIMUM_ENDPOINT_BYTES: u64 = 4 + DIGEST_BYTES;
    const MINIMUM_PAIRING_CLASS_BYTES: u64 =
        8 + 1 + DIGEST_BYTES + 8 + MINIMUM_ENDPOINT_BYTES + 8 + MINIMUM_ENDPOINT_BYTES;

    let mut reader = Reader::new(bytes)?;
    if reader.take(8, "magic")? != MAGIC {
        return Err(invalid(
            "unsupported payload magic; expected on-the-fly process-seed schema v1",
        ));
    }
    let schema_version = reader.u32("schema version")?;
    if schema_version != SCHEMA_VERSION {
        return Err(invalid(format!(
            "unsupported on-the-fly process-seed schema version {schema_version}; expected {SCHEMA_VERSION}"
        )));
    }
    let semantic_hash_revision = reader.u32("semantic hash revision")?;
    if semantic_hash_revision != SEMANTIC_HASH_REVISION {
        return Err(invalid(format!(
            "unsupported on-the-fly process-seed semantic hash revision {semantic_hash_revision}; expected {SEMANTIC_HASH_REVISION}"
        )));
    }

    let process_digest = reader.digest("process digest")?;
    let model_digest = reader.digest("model digest")?;
    let template_catalog_digest = reader.digest("template-catalog digest")?;
    let prepared_pack_digest = reader.digest("prepared-pack digest")?;
    let direct_catalog_digest = reader.digest("direct-catalog digest")?;
    let normalization_semantic_digest = reader.digest("normalization semantic digest")?;
    let normalization_convention = reader.string("normalization convention")?;

    let anchor_count = reader.count(MINIMUM_ANCHOR_BYTES, "source anchors")?;
    let mut source_anchors = reserved_vec(anchor_count, "source anchors")?;
    for anchor_index in 0..anchor_count {
        let source_slot = reader.u32("source anchor slot")?;
        let expected_slot = u32::try_from(anchor_index)
            .map_err(|_| invalid("source anchor index exceeds its u32 semantic domain"))?;
        if source_slot != expected_slot {
            return Err(invalid(
                "source anchor keys are not in canonical dense order",
            ));
        }
        let external_label = reader.u32("source anchor external label")?;
        let is_initial = reader.boolean("source anchor initial-state flag")?;
        let color_role = read_color_role(&mut reader)?;
        let is_fermionic = reader.boolean("source anchor fermion flag")?;
        let pairing_source_contract_digest =
            reader.optional_digest("source anchor pairing-source contract digest")?;
        let state_count = reader.count(MINIMUM_STATE_BYTES, "source states")?;
        let mut states = reserved_vec(state_count, "source states")?;
        let mut previous_state_index = None;
        for _ in 0..state_count {
            let state = read_source_state(&mut reader)?;
            if previous_state_index.is_some_and(|previous| previous >= state.state_index) {
                return Err(invalid(
                    "source state keys are not in canonical strict order",
                ));
            }
            previous_state_index = Some(state.state_index);
            states.push(state);
        }
        source_anchors.push(OnTheFlySourceAnchorV1::new(
            source_slot,
            external_label,
            is_initial,
            color_role,
            is_fermionic,
            pairing_source_contract_digest,
            states,
        )?);
    }

    let permutation_count = reader.count(4, "external gather permutation")?;
    let mut external_permutation = reserved_vec(permutation_count, "external gather permutation")?;
    for _ in 0..permutation_count {
        external_permutation.push(reader.u32("external gather-permutation slot")?);
    }

    let coupling_order_policy = read_coupling_order_policy(&mut reader)?;
    let coupling_row_count = reader.count(1, "coupling policy rows")?;
    let mut coupling_hierarchies = reserved_vec(coupling_row_count, "coupling hierarchies")?;
    let mut coupling_limits = reserved_vec(coupling_row_count, "coupling limits")?;
    for _ in 0..coupling_row_count {
        coupling_hierarchies.push(reader.u32("coupling hierarchy")?);
        coupling_limits.push(reader.optional_u32("explicit coupling limit")?);
    }

    let pairing_class_count = reader.count(MINIMUM_PAIRING_CLASS_BYTES, "pairing classes")?;
    let mut pairing_classes = reserved_vec(pairing_class_count, "pairing classes")?;
    for _ in 0..pairing_class_count {
        let species = reader.string("pairing species")?;
        if pairing_classes
            .last()
            .is_some_and(|previous: &OnTheFlyPairingClassV1| {
                previous.species.as_ref() >= species.as_str()
            })
        {
            return Err(invalid(
                "pairing class species keys are not in canonical strict order",
            ));
        }
        let species_semantic_digest = reader.digest("pairing species semantic digest")?;
        let fundamental_endpoints =
            read_pairing_endpoints(&mut reader, "fundamental pairing endpoints")?;
        let antifundamental_endpoints =
            read_pairing_endpoints(&mut reader, "antifundamental pairing endpoints")?;
        pairing_classes.push(OnTheFlyPairingClassV1::new(
            species,
            species_semantic_digest,
            fundamental_endpoints,
            antifundamental_endpoints,
        )?);
    }
    reader.finish()?;

    OnTheFlyProcessSeedV1::new(
        process_digest,
        model_digest,
        template_catalog_digest,
        prepared_pack_digest,
        direct_catalog_digest,
        normalization_semantic_digest,
        normalization_convention,
        ExactComplexRational::ONE,
        source_anchors,
        external_permutation,
        coupling_order_policy,
        coupling_hierarchies,
        coupling_limits,
        pairing_classes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::on_the_fly::{
        ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI, OnTheFlyProcessSeedIdentityV1,
        scalar_adapter_test_seed,
    };

    #[derive(Default)]
    struct CountingWriter {
        bytes_written: u64,
    }

    impl Write for CountingWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes_written += bytes.len() as u64;
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn state(source_slot: u32, state_index: u32) -> OnTheFlySourceStateV1 {
        let crossing_phase = if source_slot == 0 && state_index == 0 {
            ExactComplexRational::new(
                ExactRational::new(-1, 2).unwrap(),
                ExactRational::new(1, 3).unwrap(),
            )
        } else {
            ExactComplexRational::ONE
        };
        OnTheFlySourceStateV1::new(
            state_index,
            if state_index == 0 { -1 } else { 1 },
            if source_slot.is_multiple_of(2) { -1 } else { 1 },
            100 + source_slot * 2 + state_index,
            200 + source_slot * 2 + state_index,
            digest(20 + (source_slot * 2 + state_index) as u8),
            digest(30 + (source_slot * 2 + state_index) as u8),
            if source_slot.is_multiple_of(2) { -1 } else { 1 },
            crossing_phase,
            50_000 + source_slot as i32,
            if source_slot.is_multiple_of(2) { -1 } else { 1 },
            if state_index == 0 {
                vec![11 + source_slot as i32, 21 + source_slot as i32]
            } else {
                vec![31 + source_slot as i32]
            },
            300 + source_slot,
            digest(40 + source_slot as u8),
            OnTheFlySourceWavefunctionFamilyV1::DiracFermion,
            if source_slot.is_multiple_of(2) {
                OnTheFlySourceOrientationV1::Particle
            } else {
                OnTheFlySourceOrientationV1::Antiparticle
            },
            (state_index == 0).then_some(400 + source_slot),
        )
        .unwrap()
    }

    fn ordinary_seed(reverse_inputs: bool) -> OnTheFlyProcessSeedV1 {
        ordinary_seed_with_limits(reverse_inputs, vec![Some(0), Some(3)])
    }

    fn ordinary_seed_with_limits(
        reverse_inputs: bool,
        coupling_limits: Vec<Option<u32>>,
    ) -> OnTheFlyProcessSeedV1 {
        let contracts = [digest(60), digest(61), digest(62), digest(63)];
        let roles = [
            OnTheFlyExternalColorRoleV1::Fundamental,
            OnTheFlyExternalColorRoleV1::Antifundamental,
            OnTheFlyExternalColorRoleV1::Fundamental,
            OnTheFlyExternalColorRoleV1::Antifundamental,
        ];
        let mut anchors = (0..4)
            .map(|source_slot| {
                let mut states = vec![state(source_slot, 0)];
                if source_slot == 0 {
                    states.push(state(source_slot, 1));
                }
                if reverse_inputs {
                    states.reverse();
                }
                OnTheFlySourceAnchorV1::new(
                    source_slot,
                    10 + source_slot,
                    source_slot < 2,
                    roles[source_slot as usize],
                    true,
                    Some(contracts[source_slot as usize]),
                    states,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        if reverse_inputs {
            anchors.reverse();
        }
        let alpha = OnTheFlyPairingClassV1::new(
            "alpha",
            digest(70),
            vec![OnTheFlyPairingEndpointV1 {
                source_slot: 0,
                source_contract_digest: contracts[0],
            }],
            vec![OnTheFlyPairingEndpointV1 {
                source_slot: 1,
                source_contract_digest: contracts[1],
            }],
        )
        .unwrap();
        let bravo = OnTheFlyPairingClassV1::new(
            "bravo",
            digest(71),
            vec![OnTheFlyPairingEndpointV1 {
                source_slot: 2,
                source_contract_digest: contracts[2],
            }],
            vec![OnTheFlyPairingEndpointV1 {
                source_slot: 3,
                source_contract_digest: contracts[3],
            }],
        )
        .unwrap();
        let pairing_classes = if reverse_inputs {
            vec![bravo, alpha]
        } else {
            vec![alpha, bravo]
        };
        OnTheFlyProcessSeedV1::new(
            digest(1),
            digest(2),
            digest(3),
            digest(4),
            digest(5),
            digest(6),
            "raw-amplitude-v1",
            ExactComplexRational::ONE,
            anchors,
            vec![2, 0, 3, 1],
            OnTheFlyCouplingOrderPolicyV1::Minimal,
            vec![1, 2],
            coupling_limits,
            pairing_classes,
        )
        .unwrap()
    }

    fn little_u64(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    fn normalization_length_offset() -> usize {
        MAGIC.len() + 4 + 4 + (RETAINED_DIGEST_COUNT * DIGEST_BYTES) as usize
    }

    fn source_anchor_count_offset(bytes: &[u8]) -> usize {
        let normalization_length_offset = normalization_length_offset();
        normalization_length_offset + 8 + little_u64(bytes, normalization_length_offset) as usize
    }

    fn first_source_anchor_offset(bytes: &[u8]) -> usize {
        source_anchor_count_offset(bytes) + 8
    }

    fn find_unique_bytes(bytes: &[u8], needle: &[u8]) -> usize {
        let mut matches = bytes
            .windows(needle.len())
            .enumerate()
            .filter_map(|(offset, window)| (window == needle).then_some(offset));
        let offset = matches.next().unwrap();
        assert!(matches.next().is_none(), "test needle is not unique");
        offset
    }

    #[test]
    fn on_the_fly_seed_codec_round_trips_deterministically() {
        let canonical = ordinary_seed(false);
        let reordered_inputs = ordinary_seed(true);
        assert_eq!(canonical, reordered_inputs);
        let first = encode_on_the_fly_process_seed_v1(&canonical).unwrap();
        let second = encode_on_the_fly_process_seed_v1(&reordered_inputs).unwrap();
        assert_eq!(first, second);
        let decoded = decode_on_the_fly_process_seed_v1(&first).unwrap();
        assert_eq!(decoded, canonical);
        let identity = decoded.identity();
        assert_eq!(identity.abi, ON_THE_FLY_PROCESS_SEED_IDENTITY_ABI);
        assert_eq!(identity.process_digest, digest(1).to_string());
        assert_eq!(identity.external_permutation, [2, 0, 3, 1]);
        assert_eq!(identity.external_sources.len(), 4);
        assert_eq!(identity.external_sources[0].states.len(), 2);
        assert_eq!(
            identity.external_sources[0].states[0].prepared_mass_parameter_slot,
            Some(400)
        );
        let encoded_identity = serde_json::to_string(&identity).unwrap();
        let decoded_identity: OnTheFlyProcessSeedIdentityV1 =
            serde_json::from_str(&encoded_identity).unwrap();
        assert_eq!(decoded_identity, identity);
    }

    #[test]
    fn on_the_fly_seed_codec_round_trips_multiple_unbounded_orders() {
        let seed = ordinary_seed_with_limits(false, vec![None, None]);
        let bytes = encode_on_the_fly_process_seed_v1(&seed).unwrap();
        let decoded = decode_on_the_fly_process_seed_v1(&bytes).unwrap();

        assert_eq!(decoded.explicit_coupling_limits(), [None, None]);
        assert_eq!(decoded, seed);
    }

    #[test]
    fn on_the_fly_seed_codec_preserves_live_scalar_seed_identity() {
        let original =
            scalar_adapter_test_seed(digest(81), digest(82), digest(83), digest(84)).unwrap();
        let payload = encode_on_the_fly_process_seed_v1(&original).unwrap();
        let decoded = decode_on_the_fly_process_seed_v1(&payload).unwrap();

        assert_eq!(decoded.semantic_digest(), original.semantic_digest());
        assert_eq!(decoded, original);
    }

    #[test]
    fn on_the_fly_seed_codec_decode_encode_is_byte_identical() {
        let bytes = encode_on_the_fly_process_seed_v1(&ordinary_seed(false)).unwrap();
        let decoded = decode_on_the_fly_process_seed_v1(&bytes).unwrap();
        assert_eq!(encode_on_the_fly_process_seed_v1(&decoded).unwrap(), bytes);
    }

    #[test]
    fn on_the_fly_seed_codec_rejects_representative_truncations() {
        let bytes = encode_on_the_fly_process_seed_v1(&ordinary_seed(false)).unwrap();
        for length in [
            0,
            1,
            MAGIC.len() - 1,
            MAGIC.len(),
            MAGIC.len() + 4,
            MAGIC.len() + 8,
            normalization_length_offset(),
            bytes.len() / 2,
            bytes.len() - 1,
        ] {
            assert!(
                decode_on_the_fly_process_seed_v1(&bytes[..length]).is_err(),
                "truncation at byte {length} was accepted"
            );
        }
    }

    #[test]
    fn on_the_fly_seed_codec_rejects_trailing_bytes() {
        let mut bytes = encode_on_the_fly_process_seed_v1(&ordinary_seed(false)).unwrap();
        bytes.push(0xa5);
        let error = decode_on_the_fly_process_seed_v1(&bytes).unwrap_err();
        assert!(error.to_string().contains("trailing bytes"));
    }

    #[test]
    fn on_the_fly_seed_codec_rejects_future_schema_and_hash_revisions() {
        let bytes = encode_on_the_fly_process_seed_v1(&ordinary_seed(false)).unwrap();
        let mut future_schema = bytes.clone();
        future_schema[MAGIC.len()..MAGIC.len() + 4]
            .copy_from_slice(&(SCHEMA_VERSION + 1).to_le_bytes());
        assert!(
            decode_on_the_fly_process_seed_v1(&future_schema)
                .unwrap_err()
                .to_string()
                .contains("schema version")
        );
        let mut future_hash = bytes;
        future_hash[MAGIC.len() + 4..MAGIC.len() + 8]
            .copy_from_slice(&(SEMANTIC_HASH_REVISION + 1).to_le_bytes());
        assert!(
            decode_on_the_fly_process_seed_v1(&future_hash)
                .unwrap_err()
                .to_string()
                .contains("semantic hash revision")
        );
    }

    #[test]
    fn on_the_fly_seed_codec_rejects_malformed_discriminants() {
        let bytes = encode_on_the_fly_process_seed_v1(&ordinary_seed(false)).unwrap();
        let anchor = first_source_anchor_offset(&bytes);
        for (offset, expected) in [
            (anchor + 8, "boolean discriminant"),
            (anchor + 9, "color-role discriminant"),
            (anchor + 11, "optional discriminant"),
        ] {
            let mut malformed = bytes.clone();
            malformed[offset] = 0xff;
            let error = decode_on_the_fly_process_seed_v1(&malformed).unwrap_err();
            assert!(
                error.to_string().contains(expected),
                "unexpected malformed-discriminant error: {error}"
            );
        }
    }

    #[test]
    fn on_the_fly_seed_codec_rejects_duplicate_and_noncanonical_pairing_keys() {
        let bytes = encode_on_the_fly_process_seed_v1(&ordinary_seed(false)).unwrap();
        let alpha = find_unique_bytes(&bytes, b"alpha");
        let bravo = find_unique_bytes(&bytes, b"bravo");
        for (offset, replacement) in [(bravo, b"alpha"), (alpha, b"zebra")] {
            let mut malformed = bytes.clone();
            malformed[offset..offset + replacement.len()].copy_from_slice(replacement);
            let error = decode_on_the_fly_process_seed_v1(&malformed).unwrap_err();
            assert!(
                error.to_string().contains("canonical strict order"),
                "unexpected pairing-key error: {error}"
            );
        }
    }

    #[test]
    fn on_the_fly_seed_codec_rejects_u64_count_overflow_before_allocation() {
        let mut bytes = encode_on_the_fly_process_seed_v1(&ordinary_seed(false)).unwrap();
        let offset = source_anchor_count_offset(&bytes);
        bytes[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let error = decode_on_the_fly_process_seed_v1(&bytes).unwrap_err();
        assert!(error.to_string().contains("overflows u64"));
    }

    #[test]
    fn on_the_fly_seed_codec_rejects_count_that_cannot_fit_before_allocation() {
        let mut bytes = encode_on_the_fly_process_seed_v1(&ordinary_seed(false)).unwrap();
        let offset = source_anchor_count_offset(&bytes);
        let hostile_count = u64::try_from(bytes.len()).unwrap() + 1;
        bytes[offset..offset + 8].copy_from_slice(&hostile_count.to_le_bytes());
        let error = decode_on_the_fly_process_seed_v1(&bytes).unwrap_err();
        assert!(error.to_string().contains("cannot fit"));
    }

    #[test]
    fn on_the_fly_seed_codec_checked_writer_rejects_injected_u64_overflow() {
        let mut writer = Writer::with_bytes_written(CountingWriter::default(), u64::MAX - 1);
        let error = writer.raw(&[1, 2]).unwrap_err();
        assert!(error.to_string().contains("accounting overflows u64"));
        assert_eq!(writer.destination.bytes_written, 0);
    }
}
