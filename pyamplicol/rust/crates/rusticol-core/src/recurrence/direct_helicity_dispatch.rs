// SPDX-License-Identifier: 0BSD

//! Authenticated helicity dispatch for one persisted all-flow direct plan.
//!
//! The direct-plan-v2 tables remain the canonical physical-color schedule.
//! This sidecar partitions their non-source rows into maximal
//! uniform-helicity runs and stores one canonical run-ID list per resolved
//! helicity.  A cold selector bind can therefore copy only its active rows,
//! rederive destination initialization, and build schedule-local fanout and
//! interaction programs.  Warm execution never consults this metadata.

use std::collections::BTreeSet;
use std::io::{BufWriter, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use super::direct_plan::{DIRECT_NONE_U32, DirectExecutorRole, DirectRecurrencePlan};
use super::{RecurrenceStrategy, SemanticDigest};
use crate::pacbin::{PACBIN_DEFAULT_CHUNK_SIZE, create_temporary_file};
use crate::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};

pub const RECURRENCE_HELICITY_DISPATCH_ABI: &str = "pyamplicol-recurrence-helicity-dispatch-v1";

const MAGIC: &[u8; 8] = b"PACRHDS1";
const VERSION: u32 = 1;
const MAX_TABLE_ROWS: u64 = u32::MAX as u64;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("recurrence helicity dispatch: {}", message.into()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectHelicitySupportDomainDescriptor {
    pub word_start: u64,
    pub word_count: u32,
}

/// One contiguous base-plan row run with a uniform helicity support mask.
///
/// Source rows are deliberately absent.  The all-flow source dispatcher
/// already consumes the exact per-helicity source-selection table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectHelicityRowGroupDescriptor {
    pub stage: u16,
    pub role: DirectExecutorRole,
    pub direct_executor_id: u32,
    pub row_start: u64,
    pub row_count: u32,
    pub support_domain_id: u32,
}

/// Canonical range into [`DirectHelicityDispatch::dispatch_group_ids`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct DirectHelicityDispatchDescriptor {
    pub resolved_helicity_id: u32,
    pub group_id_start: u64,
    pub group_id_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectHelicityDispatchParts {
    pub runtime_layout_digest: SemanticDigest,
    pub source_row_count: u64,
    pub contribution_row_count: u64,
    pub finalization_row_count: u64,
    pub closure_row_count: u64,
    pub amplitude_destination_count: u32,
    pub direct_executor_count: u32,
    pub resolved_helicity_count: u32,
    pub support_domains: Vec<DirectHelicitySupportDomainDescriptor>,
    pub support_words: Vec<u64>,
    pub row_groups: Vec<DirectHelicityRowGroupDescriptor>,
    pub dispatches: Vec<DirectHelicityDispatchDescriptor>,
    pub dispatch_group_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectHelicityDispatch {
    parts: DirectHelicityDispatchParts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecurrenceHelicityDispatchMetadata {
    pub size_bytes: u64,
    pub sha256: [u8; 32],
}

impl DirectHelicityDispatch {
    pub fn new(parts: DirectHelicityDispatchParts) -> RusticolResult<Self> {
        validate_parts(&parts)?;
        Ok(Self { parts })
    }

    pub const fn runtime_layout_digest(&self) -> SemanticDigest {
        self.parts.runtime_layout_digest
    }

    pub const fn source_row_count(&self) -> u64 {
        self.parts.source_row_count
    }

    pub const fn contribution_row_count(&self) -> u64 {
        self.parts.contribution_row_count
    }

    pub const fn finalization_row_count(&self) -> u64 {
        self.parts.finalization_row_count
    }

    pub const fn closure_row_count(&self) -> u64 {
        self.parts.closure_row_count
    }

    pub const fn amplitude_destination_count(&self) -> u32 {
        self.parts.amplitude_destination_count
    }

    pub const fn direct_executor_count(&self) -> u32 {
        self.parts.direct_executor_count
    }

    pub const fn resolved_helicity_count(&self) -> u32 {
        self.parts.resolved_helicity_count
    }

    pub fn support_domains(&self) -> &[DirectHelicitySupportDomainDescriptor] {
        &self.parts.support_domains
    }

    pub fn support_words(&self) -> &[u64] {
        &self.parts.support_words
    }

    pub fn row_groups(&self) -> &[DirectHelicityRowGroupDescriptor] {
        &self.parts.row_groups
    }

    pub fn dispatches(&self) -> &[DirectHelicityDispatchDescriptor] {
        &self.parts.dispatches
    }

    pub fn dispatch_group_ids(&self) -> &[u32] {
        &self.parts.dispatch_group_ids
    }

    pub fn group_ids_for_helicity(&self, resolved_helicity_id: u32) -> RusticolResult<&[u32]> {
        let descriptor = self
            .parts
            .dispatches
            .get(resolved_helicity_id as usize)
            .filter(|descriptor| descriptor.resolved_helicity_id == resolved_helicity_id)
            .ok_or_else(|| {
                invalid(format!(
                    "resolved helicity {resolved_helicity_id} is absent"
                ))
            })?;
        let start = usize::try_from(descriptor.group_id_start)
            .map_err(|_| invalid("dispatch group-ID start exceeds usize"))?;
        let end = start
            .checked_add(descriptor.group_id_count as usize)
            .ok_or_else(|| invalid("dispatch group-ID range overflows usize"))?;
        self.parts
            .dispatch_group_ids
            .get(start..end)
            .ok_or_else(|| invalid("dispatch group-ID range is out of bounds"))
    }

    pub fn support_domain_contains(
        &self,
        support_domain_id: u32,
        resolved_helicity_id: u32,
    ) -> RusticolResult<bool> {
        support_domain_contains_parts(&self.parts, support_domain_id, resolved_helicity_id)
    }

    /// Pair the sidecar with its exact all-flow direct plan.
    ///
    /// Container/member SHA-256 authenticates the bytes. This check establishes
    /// the semantic pairing once, at the authoritative load boundary.
    pub fn validate_for_plan(&self, plan: &DirectRecurrencePlan) -> RusticolResult<()> {
        if plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "helicity dispatch requires an all-flow-union direct plan",
            ));
        }
        if self.runtime_layout_digest() != plan.runtime_layout_digest() {
            return Err(RusticolError::integrity(
                "recurrence helicity dispatch runtime-layout digest does not match its plan",
            ));
        }
        let expected_counts = [
            u64::try_from(plan.sources().len())
                .map_err(|_| invalid("plan source-row count exceeds u64"))?,
            u64::try_from(plan.contributions().len())
                .map_err(|_| invalid("plan contribution-row count exceeds u64"))?,
            u64::try_from(plan.finalizations().len())
                .map_err(|_| invalid("plan finalization-row count exceeds u64"))?,
            u64::try_from(plan.closures().len())
                .map_err(|_| invalid("plan closure-row count exceeds u64"))?,
        ];
        let actual_counts = [
            self.source_row_count(),
            self.contribution_row_count(),
            self.finalization_row_count(),
            self.closure_row_count(),
        ];
        if actual_counts != expected_counts
            || self.amplitude_destination_count() != plan.amplitude_destination_count()
            || self.direct_executor_count() != plan.direct_executor_count()
            || self.resolved_helicity_count() as usize != plan.resolved_helicities().len()
            || u64::from(self.resolved_helicity_count()) != plan.retained_helicity_count()
        {
            return Err(RusticolError::integrity(
                "recurrence helicity dispatch dimensions do not match its plan",
            ));
        }
        validate_base_row_group_partition(plan, &self.parts.row_groups)
    }

    pub fn into_parts(self) -> DirectHelicityDispatchParts {
        self.parts
    }
}

fn checked_u32_table_count(label: &str, value: u64) -> RusticolResult<()> {
    if value > MAX_TABLE_ROWS {
        return Err(invalid(format!("{label} count exceeds the u32 ID domain")));
    }
    Ok(())
}

fn support_domain_words(
    parts: &DirectHelicityDispatchParts,
    support_domain_id: u32,
) -> RusticolResult<&[u64]> {
    let descriptor = parts
        .support_domains
        .get(support_domain_id as usize)
        .ok_or_else(|| invalid("support-domain ID is out of bounds"))?;
    let start = usize::try_from(descriptor.word_start)
        .map_err(|_| invalid("support-domain word start exceeds usize"))?;
    let end = start
        .checked_add(descriptor.word_count as usize)
        .ok_or_else(|| invalid("support-domain word range overflows usize"))?;
    parts
        .support_words
        .get(start..end)
        .ok_or_else(|| invalid("support-domain word range is out of bounds"))
}

fn support_domain_contains_parts(
    parts: &DirectHelicityDispatchParts,
    support_domain_id: u32,
    resolved_helicity_id: u32,
) -> RusticolResult<bool> {
    if resolved_helicity_id >= parts.resolved_helicity_count {
        return Err(invalid("resolved helicity is out of bounds"));
    }
    let words = support_domain_words(parts, support_domain_id)?;
    let word = resolved_helicity_id as usize / 64;
    let bit = resolved_helicity_id % 64;
    Ok(words
        .get(word)
        .is_some_and(|value| value & (1_u64 << bit) != 0))
}

fn validate_parts(parts: &DirectHelicityDispatchParts) -> RusticolResult<()> {
    for (label, count) in [
        ("source row", parts.source_row_count),
        ("contribution row", parts.contribution_row_count),
        ("finalization row", parts.finalization_row_count),
        ("closure row", parts.closure_row_count),
        ("support domain", parts.support_domains.len() as u64),
        ("support word", parts.support_words.len() as u64),
        ("split row group", parts.row_groups.len() as u64),
        ("dispatch group ID", parts.dispatch_group_ids.len() as u64),
    ] {
        checked_u32_table_count(label, count)?;
    }
    if parts.source_row_count == 0
        || parts.contribution_row_count == 0
        || parts.closure_row_count == 0
        || parts.amplitude_destination_count == 0
        || parts.direct_executor_count == 0
        || parts.resolved_helicity_count == 0
        || parts.support_domains.is_empty()
        || parts.row_groups.is_empty()
    {
        return Err(invalid(
            "dispatch requires nonempty source, contribution, closure, destination, executor, helicity, support-domain, and row-group tables",
        ));
    }
    if parts.dispatches.len() != parts.resolved_helicity_count as usize {
        return Err(invalid(
            "dispatch descriptor table does not cover every resolved helicity",
        ));
    }

    validate_support_domains(parts)?;
    validate_split_row_groups(parts)?;
    validate_dispatch_csr(parts)
}

fn validate_support_domains(parts: &DirectHelicityDispatchParts) -> RusticolResult<()> {
    let maximum_words = parts.resolved_helicity_count.div_ceil(64) as usize;
    let mut next = 0_u64;
    let mut unique = BTreeSet::<Vec<u64>>::new();
    for (domain_id, descriptor) in parts.support_domains.iter().enumerate() {
        if descriptor.word_start != next {
            return Err(invalid(format!(
                "support domain {domain_id} does not continue the packed word table"
            )));
        }
        let words = support_domain_words(parts, domain_id as u32)?;
        if words.len() > maximum_words || words.last() == Some(&0) {
            return Err(invalid(format!(
                "support domain {domain_id} is not a canonical compact helicity mask"
            )));
        }
        if words.len() == maximum_words && !parts.resolved_helicity_count.is_multiple_of(64) {
            let live_bits = parts.resolved_helicity_count % 64;
            let high_mask = !((1_u64 << live_bits) - 1);
            if words.last().is_some_and(|word| word & high_mask != 0) {
                return Err(invalid(format!(
                    "support domain {domain_id} sets bits outside the resolved-helicity axis"
                )));
            }
        }
        if !unique.insert(words.to_vec()) {
            return Err(invalid(format!(
                "support domain {domain_id} duplicates an earlier interned mask"
            )));
        }
        next = next
            .checked_add(u64::from(descriptor.word_count))
            .ok_or_else(|| invalid("support-domain word partition overflows u64"))?;
    }
    if next != parts.support_words.len() as u64 {
        return Err(invalid(
            "support domains do not partition the support-word table",
        ));
    }
    Ok(())
}

fn role_partition_index(role: DirectExecutorRole) -> RusticolResult<usize> {
    match role {
        DirectExecutorRole::Contribution => Ok(0),
        DirectExecutorRole::Finalization => Ok(1),
        DirectExecutorRole::Closure => Ok(2),
        DirectExecutorRole::Source => Err(invalid(
            "static source groups must not appear in helicity dispatch",
        )),
    }
}

fn validate_split_row_groups(parts: &DirectHelicityDispatchParts) -> RusticolResult<()> {
    let mut next = [0_u64; 3];
    let mut previous_order = None;
    for (group_id, group) in parts.row_groups.iter().enumerate() {
        if group.row_count == 0 {
            return Err(invalid(format!("split row group {group_id} is empty")));
        }
        let role_index = role_partition_index(group.role)?;
        if group.direct_executor_id != DIRECT_NONE_U32
            && group.direct_executor_id >= parts.direct_executor_count
        {
            return Err(invalid(format!(
                "split row group {group_id} executor is out of bounds"
            )));
        }
        if group.direct_executor_id == DIRECT_NONE_U32
            && group.role != DirectExecutorRole::Contribution
        {
            return Err(invalid(format!(
                "split row group {group_id} has a missing non-contribution executor"
            )));
        }
        if group.support_domain_id as usize >= parts.support_domains.len() {
            return Err(invalid(format!(
                "split row group {group_id} support domain is out of bounds"
            )));
        }
        let order = (group.stage, group.role);
        if previous_order.is_some_and(|previous| order < previous) {
            return Err(invalid(
                "split row groups are not in canonical schedule order",
            ));
        }
        previous_order = Some(order);
        if group.row_start != next[role_index] {
            return Err(invalid(format!(
                "split row group {group_id} does not continue its role's row table"
            )));
        }
        next[role_index] = group
            .row_start
            .checked_add(u64::from(group.row_count))
            .ok_or_else(|| invalid("split row-group range overflows u64"))?;
    }
    let expected = [
        parts.contribution_row_count,
        parts.finalization_row_count,
        parts.closure_row_count,
    ];
    if next != expected {
        return Err(invalid(
            "split row groups do not partition every non-source base-plan row",
        ));
    }
    Ok(())
}

fn validate_dispatch_csr(parts: &DirectHelicityDispatchParts) -> RusticolResult<()> {
    let mut next = 0_u64;
    let mut starts = Vec::with_capacity(parts.dispatches.len());
    let mut ends = Vec::with_capacity(parts.dispatches.len());
    for (helicity_id, descriptor) in parts.dispatches.iter().enumerate() {
        if descriptor.resolved_helicity_id != helicity_id as u32
            || descriptor.group_id_start != next
        {
            return Err(invalid(format!(
                "resolved-helicity dispatch {helicity_id} is not canonical"
            )));
        }
        let start = usize::try_from(descriptor.group_id_start)
            .map_err(|_| invalid("dispatch group-ID start exceeds usize"))?;
        let end = start
            .checked_add(descriptor.group_id_count as usize)
            .ok_or_else(|| invalid("dispatch group-ID range overflows usize"))?;
        let group_ids = parts
            .dispatch_group_ids
            .get(start..end)
            .ok_or_else(|| invalid("dispatch group-ID range is out of bounds"))?;
        if group_ids
            .iter()
            .any(|group_id| *group_id as usize >= parts.row_groups.len())
            || !group_ids.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(invalid(format!(
                "resolved-helicity dispatch {helicity_id} has invalid or unordered group IDs"
            )));
        }
        starts.push(start);
        ends.push(end);
        next = next
            .checked_add(u64::from(descriptor.group_id_count))
            .ok_or_else(|| invalid("dispatch group-ID partition overflows u64"))?;
    }
    if next != parts.dispatch_group_ids.len() as u64 {
        return Err(invalid(
            "resolved-helicity dispatches do not partition the group-ID table",
        ));
    }

    // The masks are the audit representation and the CSR is the cold-bind
    // accelerator. Establish their exact equivalence once without allocating
    // an H-by-group matrix.
    let mut cursors = starts;
    for (group_id, group) in parts.row_groups.iter().enumerate() {
        for (word_index, mut word) in support_domain_words(parts, group.support_domain_id)?
            .iter()
            .copied()
            .enumerate()
        {
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let helicity_id = word_index * 64 + bit;
                let cursor = cursors
                    .get_mut(helicity_id)
                    .ok_or_else(|| invalid("support mask exceeds the helicity axis"))?;
                let actual = parts.dispatch_group_ids.get(*cursor).copied();
                if actual != Some(group_id as u32) {
                    return Err(invalid(format!(
                        "resolved-helicity dispatch {helicity_id} disagrees with support masks at group {group_id}"
                    )));
                }
                *cursor += 1;
                word &= word - 1;
            }
        }
    }
    if cursors != ends {
        return Err(invalid(
            "resolved-helicity dispatch contains groups absent from its support masks",
        ));
    }
    Ok(())
}

fn validate_base_row_group_partition(
    plan: &DirectRecurrencePlan,
    groups: &[DirectHelicityRowGroupDescriptor],
) -> RusticolResult<()> {
    let mut group_index = 0usize;
    for base in plan
        .row_groups()
        .iter()
        .filter(|descriptor| descriptor.role != DirectExecutorRole::Source)
    {
        let base_end = base
            .row_start
            .checked_add(u64::from(base.row_count))
            .ok_or_else(|| invalid("base row-group range overflows u64"))?;
        let mut next = base.row_start;
        while next < base_end {
            let group = groups.get(group_index).ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence helicity dispatch omits a base-plan row-group suffix",
                )
            })?;
            if group.stage != base.stage
                || group.role != base.role
                || group.direct_executor_id != base.direct_executor_id
                || group.row_start != next
            {
                return Err(RusticolError::integrity(
                    "recurrence helicity dispatch split group does not match its base row group",
                ));
            }
            next = next
                .checked_add(u64::from(group.row_count))
                .ok_or_else(|| invalid("split row-group range overflows u64"))?;
            if next > base_end {
                return Err(RusticolError::integrity(
                    "recurrence helicity dispatch split group crosses a base row-group boundary",
                ));
            }
            group_index += 1;
        }
    }
    if group_index != groups.len() {
        return Err(RusticolError::integrity(
            "recurrence helicity dispatch has groups beyond the base-plan schedule",
        ));
    }
    Ok(())
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

    fn raw(&mut self, bytes: &[u8]) -> RusticolResult<()> {
        let count =
            u64::try_from(bytes.len()).map_err(|_| invalid("payload write length exceeds u64"))?;
        self.bytes_written = self
            .bytes_written
            .checked_add(count)
            .ok_or_else(|| invalid("payload length overflows u64"))?;
        self.destination.write_all(bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "could not stream recurrence helicity dispatch: {error}"
            ))
        })
    }

    fn u16(&mut self, value: u16) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> RusticolResult<()> {
        self.raw(&value.to_le_bytes())
    }

    fn digest(&mut self, value: SemanticDigest) -> RusticolResult<()> {
        self.raw(value.as_bytes())
    }
}

pub fn encode_recurrence_helicity_dispatch_v1(
    dispatch: &DirectHelicityDispatch,
) -> RusticolResult<Vec<u8>> {
    let mut payload = Vec::new();
    encode_recurrence_helicity_dispatch_v1_to_writer(dispatch, &mut payload)?;
    Ok(payload)
}

pub(crate) fn encode_recurrence_helicity_dispatch_v1_to_writer<W: Write>(
    dispatch: &DirectHelicityDispatch,
    destination: W,
) -> RusticolResult<u64> {
    let parts = &dispatch.parts;
    let mut writer = Writer::new(destination);
    writer.raw(MAGIC)?;
    writer.u32(VERSION)?;
    writer.u32(0)?;
    writer.digest(parts.runtime_layout_digest)?;
    writer.u64(parts.source_row_count)?;
    writer.u64(parts.contribution_row_count)?;
    writer.u64(parts.finalization_row_count)?;
    writer.u64(parts.closure_row_count)?;
    writer.u32(parts.amplitude_destination_count)?;
    writer.u32(parts.direct_executor_count)?;
    writer.u32(parts.resolved_helicity_count)?;
    writer.u32(parts.support_domains.len() as u32)?;
    writer.u32(parts.row_groups.len() as u32)?;
    writer.u32(parts.dispatches.len() as u32)?;
    writer.u64(parts.support_words.len() as u64)?;
    writer.u64(parts.dispatch_group_ids.len() as u64)?;

    for descriptor in &parts.support_domains {
        writer.u64(descriptor.word_start)?;
        writer.u32(descriptor.word_count)?;
        writer.u32(0)?;
    }
    for word in &parts.support_words {
        writer.u64(*word)?;
    }
    for group in &parts.row_groups {
        writer.u16(group.stage)?;
        writer.u16(group.role as u16)?;
        writer.u32(group.direct_executor_id)?;
        writer.u64(group.row_start)?;
        writer.u32(group.row_count)?;
        writer.u32(group.support_domain_id)?;
    }
    for descriptor in &parts.dispatches {
        writer.u32(descriptor.resolved_helicity_id)?;
        writer.u32(0)?;
        writer.u64(descriptor.group_id_start)?;
        writer.u32(descriptor.group_id_count)?;
        writer.u32(0)?;
    }
    for group_id in &parts.dispatch_group_ids {
        writer.u32(*group_id)?;
    }
    Ok(writer.bytes_written)
}

struct HashingWriter<W> {
    inner: W,
    digest: Sha256,
}

impl<W> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            digest: Sha256::new(),
        }
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.digest.update(&bytes[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Stream one dispatch sidecar to a same-directory temporary file and publish
/// it atomically. The returned digest is computed over the bytes being written,
/// so callers need no second read or hashing pass.
pub fn write_recurrence_helicity_dispatch_v1_atomic(
    destination: impl AsRef<Path>,
    dispatch: &DirectHelicityDispatch,
) -> RusticolResult<RecurrenceHelicityDispatchMetadata> {
    let destination = destination.as_ref();
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "destination directory does not exist: {}",
            parent.display()
        )));
    }
    let (temporary_path, temporary_file) = create_temporary_file(destination, parent)?;
    let result = (|| {
        let buffered = BufWriter::with_capacity(PACBIN_DEFAULT_CHUNK_SIZE, temporary_file);
        let mut writer = HashingWriter::new(buffered);
        let size_bytes = encode_recurrence_helicity_dispatch_v1_to_writer(dispatch, &mut writer)?;
        writer.flush().map_err(|error| {
            RusticolError::artifact(format!(
                "could not flush recurrence helicity dispatch {}: {error}",
                temporary_path.display()
            ))
        })?;
        #[cfg(unix)]
        writer
            .inner
            .get_ref()
            .set_permissions(std::fs::Permissions::from_mode(0o644))
            .map_err(|error| {
                RusticolError::artifact(format!(
                    "could not set recurrence helicity dispatch permissions {}: {error}",
                    temporary_path.display()
                ))
            })?;
        writer.inner.get_ref().sync_all().map_err(|error| {
            RusticolError::artifact(format!(
                "could not sync recurrence helicity dispatch {}: {error}",
                temporary_path.display()
            ))
        })?;
        let HashingWriter { inner, digest } = writer;
        let sha256: [u8; 32] = digest.finalize().into();
        drop(inner);
        std::fs::rename(&temporary_path, destination).map_err(|error| {
            RusticolError::artifact(format!(
                "could not atomically publish recurrence helicity dispatch {}: {error}",
                destination.display()
            ))
        })?;
        // The containing artifact root is content-addressed. Best-effort
        // directory sync mirrors PACBIN publication without adding another
        // platform-specific durability contract.
        if let Ok(directory) = std::fs::File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(RecurrenceHelicityDispatchMetadata { size_bytes, sha256 })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize, label: &str) -> RusticolResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| invalid(format!("{label} offset overflows usize")))?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(|| {
            invalid(format!(
                "truncated {label} at byte {}: need {count}, have {}",
                self.offset,
                self.bytes.len().saturating_sub(self.offset)
            ))
        })?;
        self.offset = end;
        Ok(bytes)
    }

    fn u16(&mut self, label: &str) -> RusticolResult<u16> {
        Ok(u16::from_le_bytes(
            self.take(2, label)?.try_into().expect("checked read"),
        ))
    }

    fn u32(&mut self, label: &str) -> RusticolResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4, label)?.try_into().expect("checked read"),
        ))
    }

    fn u64(&mut self, label: &str) -> RusticolResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8, label)?.try_into().expect("checked read"),
        ))
    }

    fn digest(&mut self, label: &str) -> RusticolResult<SemanticDigest> {
        let bytes = self.take(32, label)?.try_into().expect("checked read");
        SemanticDigest::new(bytes).map_err(|error| invalid(error.message()))
    }

    fn reserved_u32(&mut self, label: &str) -> RusticolResult<()> {
        let value = self.u32(label)?;
        if value != 0 {
            return Err(invalid(format!("{label} must be zero, found {value}")));
        }
        Ok(())
    }

    fn checked_count(&self, value: u64, row_bytes: usize, label: &str) -> RusticolResult<usize> {
        checked_u32_table_count(label, value)?;
        let count =
            usize::try_from(value).map_err(|_| invalid(format!("{label} count exceeds usize")))?;
        if count > self.bytes.len().saturating_sub(self.offset) / row_bytes {
            return Err(invalid(format!(
                "{label} count cannot fit in the remaining payload"
            )));
        }
        Ok(count)
    }

    fn finish(self) -> RusticolResult<()> {
        if self.offset != self.bytes.len() {
            return Err(invalid(format!(
                "payload contains {} trailing bytes",
                self.bytes.len() - self.offset
            )));
        }
        Ok(())
    }
}

pub fn decode_recurrence_helicity_dispatch_v1(
    bytes: &[u8],
) -> RusticolResult<DirectHelicityDispatch> {
    let mut reader = Reader::new(bytes);
    if reader.take(MAGIC.len(), "magic")? != MAGIC {
        return Err(invalid("payload has the wrong magic"));
    }
    if reader.u32("version")? != VERSION {
        return Err(invalid("payload has an unsupported version"));
    }
    reader.reserved_u32("header reserved")?;
    let runtime_layout_digest = reader.digest("runtime-layout digest")?;
    let source_row_count = reader.u64("source-row count")?;
    let contribution_row_count = reader.u64("contribution-row count")?;
    let finalization_row_count = reader.u64("finalization-row count")?;
    let closure_row_count = reader.u64("closure-row count")?;
    let amplitude_destination_count = reader.u32("amplitude-destination count")?;
    let direct_executor_count = reader.u32("direct-executor count")?;
    let resolved_helicity_count = reader.u32("resolved-helicity count")?;
    let support_domain_count = reader.u32("support-domain count")? as usize;
    let row_group_count = reader.u32("split row-group count")? as usize;
    let dispatch_count = reader.u32("dispatch count")? as usize;
    let support_word_count = reader.u64("support-word count")?;
    let dispatch_group_id_count = reader.u64("dispatch group-ID count")?;

    reader.checked_count(support_domain_count as u64, 16, "support domain")?;
    let mut support_domains = Vec::with_capacity(support_domain_count);
    for _ in 0..support_domain_count {
        let word_start = reader.u64("support-domain word start")?;
        let word_count = reader.u32("support-domain word count")?;
        reader.reserved_u32("support-domain reserved")?;
        support_domains.push(DirectHelicitySupportDomainDescriptor {
            word_start,
            word_count,
        });
    }
    let support_word_count = reader.checked_count(support_word_count, 8, "support word")?;
    let mut support_words = Vec::with_capacity(support_word_count);
    for _ in 0..support_word_count {
        support_words.push(reader.u64("support word")?);
    }
    reader.checked_count(row_group_count as u64, 24, "split row group")?;
    let mut row_groups = Vec::with_capacity(row_group_count);
    for _ in 0..row_group_count {
        row_groups.push(DirectHelicityRowGroupDescriptor {
            stage: reader.u16("split row-group stage")?,
            role: DirectExecutorRole::try_from(reader.u16("split row-group role")?)?,
            direct_executor_id: reader.u32("split row-group executor")?,
            row_start: reader.u64("split row-group row start")?,
            row_count: reader.u32("split row-group row count")?,
            support_domain_id: reader.u32("split row-group support domain")?,
        });
    }
    reader.checked_count(dispatch_count as u64, 24, "dispatch")?;
    let mut dispatches = Vec::with_capacity(dispatch_count);
    for _ in 0..dispatch_count {
        let resolved_helicity_id = reader.u32("dispatch helicity ID")?;
        reader.reserved_u32("dispatch leading reserved")?;
        let group_id_start = reader.u64("dispatch group-ID start")?;
        let group_id_count = reader.u32("dispatch group-ID count")?;
        reader.reserved_u32("dispatch trailing reserved")?;
        dispatches.push(DirectHelicityDispatchDescriptor {
            resolved_helicity_id,
            group_id_start,
            group_id_count,
        });
    }
    let dispatch_group_id_count =
        reader.checked_count(dispatch_group_id_count, 4, "dispatch group ID")?;
    let mut dispatch_group_ids = Vec::with_capacity(dispatch_group_id_count);
    for _ in 0..dispatch_group_id_count {
        dispatch_group_ids.push(reader.u32("dispatch group ID")?);
    }
    reader.finish()?;

    DirectHelicityDispatch::new(DirectHelicityDispatchParts {
        runtime_layout_digest,
        source_row_count,
        contribution_row_count,
        finalization_row_count,
        closure_row_count,
        amplitude_destination_count,
        direct_executor_count,
        resolved_helicity_count,
        support_domains,
        support_words,
        row_groups,
        dispatches,
        dispatch_group_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn valid_parts() -> DirectHelicityDispatchParts {
        DirectHelicityDispatchParts {
            runtime_layout_digest: digest(0x22),
            source_row_count: 2,
            contribution_row_count: 4,
            finalization_row_count: 2,
            closure_row_count: 2,
            amplitude_destination_count: 2,
            direct_executor_count: 6,
            resolved_helicity_count: 3,
            // Domain 0 is exact zero, 1 is H={0,2}, 2 is H={1}, and 3 is universal.
            support_domains: vec![
                DirectHelicitySupportDomainDescriptor {
                    word_start: 0,
                    word_count: 0,
                },
                DirectHelicitySupportDomainDescriptor {
                    word_start: 0,
                    word_count: 1,
                },
                DirectHelicitySupportDomainDescriptor {
                    word_start: 1,
                    word_count: 1,
                },
                DirectHelicitySupportDomainDescriptor {
                    word_start: 2,
                    word_count: 1,
                },
            ],
            support_words: vec![0b101, 0b010, 0b111],
            row_groups: vec![
                DirectHelicityRowGroupDescriptor {
                    stage: 1,
                    role: DirectExecutorRole::Contribution,
                    direct_executor_id: 1,
                    row_start: 0,
                    row_count: 2,
                    support_domain_id: 1,
                },
                DirectHelicityRowGroupDescriptor {
                    stage: 1,
                    role: DirectExecutorRole::Contribution,
                    direct_executor_id: 1,
                    row_start: 2,
                    row_count: 1,
                    support_domain_id: 2,
                },
                DirectHelicityRowGroupDescriptor {
                    stage: 1,
                    role: DirectExecutorRole::Contribution,
                    direct_executor_id: 2,
                    row_start: 3,
                    row_count: 1,
                    support_domain_id: 3,
                },
                DirectHelicityRowGroupDescriptor {
                    stage: 1,
                    role: DirectExecutorRole::Finalization,
                    direct_executor_id: 3,
                    row_start: 0,
                    row_count: 2,
                    support_domain_id: 3,
                },
                DirectHelicityRowGroupDescriptor {
                    stage: 1,
                    role: DirectExecutorRole::Closure,
                    direct_executor_id: 4,
                    row_start: 0,
                    row_count: 1,
                    support_domain_id: 1,
                },
                DirectHelicityRowGroupDescriptor {
                    stage: 1,
                    role: DirectExecutorRole::Closure,
                    direct_executor_id: 4,
                    row_start: 1,
                    row_count: 1,
                    support_domain_id: 2,
                },
            ],
            dispatches: vec![
                DirectHelicityDispatchDescriptor {
                    resolved_helicity_id: 0,
                    group_id_start: 0,
                    group_id_count: 4,
                },
                DirectHelicityDispatchDescriptor {
                    resolved_helicity_id: 1,
                    group_id_start: 4,
                    group_id_count: 4,
                },
                DirectHelicityDispatchDescriptor {
                    resolved_helicity_id: 2,
                    group_id_start: 8,
                    group_id_count: 4,
                },
            ],
            dispatch_group_ids: vec![0, 2, 3, 4, 1, 2, 3, 5, 0, 2, 3, 4],
        }
    }

    #[test]
    fn dispatch_codec_round_trips_canonical_support_and_csr() {
        let dispatch = DirectHelicityDispatch::new(valid_parts()).unwrap();
        let bytes = encode_recurrence_helicity_dispatch_v1(&dispatch).unwrap();
        let decoded = decode_recurrence_helicity_dispatch_v1(&bytes).unwrap();
        assert_eq!(decoded, dispatch);
        assert_eq!(decoded.group_ids_for_helicity(1).unwrap(), &[1, 2, 3, 5]);
        assert!(decoded.support_domain_contains(1, 2).unwrap());
        assert!(!decoded.support_domain_contains(1, 1).unwrap());
        assert_eq!(
            encode_recurrence_helicity_dispatch_v1(&decoded).unwrap(),
            bytes
        );
    }

    #[test]
    fn atomic_writer_reports_the_published_payload_identity() {
        let dispatch = DirectHelicityDispatch::new(valid_parts()).unwrap();
        let path = std::env::temp_dir().join(format!(
            "rusticol-helicity-dispatch-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let metadata = write_recurrence_helicity_dispatch_v1_atomic(&path, &dispatch).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let expected_sha256: [u8; 32] = Sha256::digest(&bytes).into();
        assert_eq!(metadata.size_bytes, bytes.len() as u64);
        assert_eq!(metadata.sha256, expected_sha256);
        assert_eq!(
            decode_recurrence_helicity_dispatch_v1(&bytes).unwrap(),
            dispatch
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dispatch_rejects_csr_that_disagrees_with_support_masks() {
        let mut parts = valid_parts();
        parts.dispatch_group_ids[4] = 0;
        let error = DirectHelicityDispatch::new(parts).unwrap_err();
        assert!(error.message().contains("disagrees with support masks"));
    }

    #[test]
    fn dispatch_rejects_static_source_groups() {
        let mut parts = valid_parts();
        parts.row_groups[0].role = DirectExecutorRole::Source;
        let error = DirectHelicityDispatch::new(parts).unwrap_err();
        assert!(error.message().contains("static source groups"));
    }

    #[test]
    fn dispatch_rejects_noncanonical_support_masks_and_trailing_bytes() {
        let mut parts = valid_parts();
        parts.support_words[2] |= 1 << 63;
        let error = DirectHelicityDispatch::new(parts).unwrap_err();
        assert!(
            error
                .message()
                .contains("outside the resolved-helicity axis")
        );

        let dispatch = DirectHelicityDispatch::new(valid_parts()).unwrap();
        let mut bytes = encode_recurrence_helicity_dispatch_v1(&dispatch).unwrap();
        bytes.push(0);
        let error = decode_recurrence_helicity_dispatch_v1(&bytes).unwrap_err();
        assert!(error.message().contains("trailing bytes"));
    }

    #[test]
    fn plan_pairing_fails_closed_before_accepting_a_non_union_plan() {
        let dispatch = DirectHelicityDispatch::new(valid_parts()).unwrap();
        let plan = crate::recurrence::valid_direct_plan_fixture();
        let error = dispatch.validate_for_plan(&plan).unwrap_err();
        assert!(error.message().contains("all-flow-union"));
    }
}
