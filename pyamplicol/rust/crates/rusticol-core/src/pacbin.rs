// SPDX-License-Identifier: 0BSD

//! Strict reader for portable pyAmpliCol evaluator containers.
//!
//! `pacbin-v1` stores fixed-size metadata, aligned uncompressed member
//! payloads, a sorted variable-width index, and a footer. The index and every
//! payload are authenticated with SHA-256. This module is intentionally
//! independent of artifact loading: it owns one container as immutable bytes
//! and provides indexed borrowed slices without extracting files.

use crate::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Deref;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

pub const PACBIN_VERSION: u16 = 1;
pub const PACBIN_ALIGNMENT: u32 = 64;
pub const PACBIN_DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;
pub const PACBIN_DEFAULT_MODE: u32 = 0o644;

const HEADER_MAGIC: &[u8; 8] = b"PACBIN\0\0";
const INDEX_MAGIC: &[u8; 8] = b"PACIDX\0\0";
const FOOTER_MAGIC: &[u8; 8] = b"PACEND\0\0";
const SUPPORTED_FLAGS: u32 = 0;
const HEADER_SIZE: usize = 64;
const INDEX_HEADER_SIZE: usize = 32;
const INDEX_ENTRY_SIZE: usize = 56;
const FOOTER_SIZE: usize = 64;
const INDEX_ALIGNMENT: u64 = 8;
const TEMPORARY_FILE_ATTEMPTS: u64 = 128;

static NEXT_TEMPORARY_FILE: AtomicU64 = AtomicU64::new(0);

/// Portable payload kinds defined by `pacbin-v1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum PacbinMemberKind {
    SymjitApplication = 1,
    SymbolicaExactState = 2,
    NativeLibrary = 3,
    EagerRuntimeMetadata = 4,
    EagerRuntimeTable = 5,
    RecurrenceDirectPlan = 7,
    RecurrenceColorProjectionCertificate = 8,
    OnTheFlyProcessSeed = 9,
    ColorContraction = 10,
}

impl PacbinMemberKind {
    fn parse(value: u16) -> RusticolResult<Self> {
        match value {
            1 => Ok(Self::SymjitApplication),
            2 => Ok(Self::SymbolicaExactState),
            3 => Ok(Self::NativeLibrary),
            4 => Ok(Self::EagerRuntimeMetadata),
            5 => Ok(Self::EagerRuntimeTable),
            7 => Ok(Self::RecurrenceDirectPlan),
            8 => Ok(Self::RecurrenceColorProjectionCertificate),
            9 => Ok(Self::OnTheFlyProcessSeed),
            10 => Ok(Self::ColorContraction),
            _ => Err(compatibility(format!(
                "unknown pacbin member kind: {value}"
            ))),
        }
    }
}

/// One bounded source streamed into a `pacbin-v1` member.
pub enum PacbinWriteSource<'a> {
    Bytes(&'a [u8]),
    Path(PathBuf),
    Reader(&'a mut dyn Read),
}

/// One normalized member supplied to the deterministic Rust writer.
pub struct PacbinWriteMember<'a> {
    logical_path: String,
    kind: PacbinMemberKind,
    source: PacbinWriteSource<'a>,
}

impl<'a> PacbinWriteMember<'a> {
    pub fn from_bytes(
        logical_path: &str,
        kind: PacbinMemberKind,
        payload: &'a [u8],
    ) -> RusticolResult<Self> {
        Self::new(logical_path, kind, PacbinWriteSource::Bytes(payload))
    }

    pub fn from_path(
        logical_path: &str,
        kind: PacbinMemberKind,
        source: impl Into<PathBuf>,
    ) -> RusticolResult<Self> {
        Self::new(logical_path, kind, PacbinWriteSource::Path(source.into()))
    }

    pub fn from_reader(
        logical_path: &str,
        kind: PacbinMemberKind,
        source: &'a mut dyn Read,
    ) -> RusticolResult<Self> {
        Self::new(logical_path, kind, PacbinWriteSource::Reader(source))
    }

    fn new(
        logical_path: &str,
        kind: PacbinMemberKind,
        source: PacbinWriteSource<'a>,
    ) -> RusticolResult<Self> {
        Ok(Self {
            logical_path: normalize_logical_path(logical_path)?,
            kind,
            source,
        })
    }

    pub fn logical_path(&self) -> &str {
        &self.logical_path
    }

    pub fn kind(&self) -> PacbinMemberKind {
        self.kind
    }
}

/// Validated options for one atomic `pacbin-v1` publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PacbinWriteOptions {
    chunk_size: usize,
    mode: u32,
}

impl PacbinWriteOptions {
    pub fn new(chunk_size: usize, mode: u32) -> RusticolResult<Self> {
        if chunk_size == 0 {
            return Err(RusticolError::invalid_argument(
                "pacbin chunk size must be a positive integer",
            ));
        }
        if mode > 0o777 {
            return Err(RusticolError::invalid_argument(
                "pacbin publication mode must be between 0o000 and 0o777",
            ));
        }
        Ok(Self { chunk_size, mode })
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn mode(&self) -> u32 {
        self.mode
    }
}

impl Default for PacbinWriteOptions {
    fn default() -> Self {
        Self {
            chunk_size: PACBIN_DEFAULT_CHUNK_SIZE,
            mode: PACBIN_DEFAULT_MODE,
        }
    }
}

/// Validated metadata for one member.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacbinMember {
    logical_path: String,
    kind: PacbinMemberKind,
    offset: u64,
    length: u64,
    sha256: [u8; 32],
}

impl PacbinMember {
    pub fn logical_path(&self) -> &str {
        &self.logical_path
    }

    pub fn kind(&self) -> PacbinMemberKind {
        self.kind
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn length(&self) -> u64 {
        self.length
    }

    pub fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }
}

/// Validated top-level container metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacbinIndex {
    version: u16,
    index_offset: u64,
    index_sha256: [u8; 32],
    file_size: u64,
    members: Vec<PacbinMember>,
}

impl PacbinIndex {
    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn index_offset(&self) -> u64 {
        self.index_offset
    }

    pub fn index_sha256(&self) -> &[u8; 32] {
        &self.index_sha256
    }

    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    pub fn members(&self) -> &[PacbinMember] {
        &self.members
    }
}

/// An authenticated, indexed `pacbin-v1` container.
///
/// Files are copied into an anonymous read-only mapping on Unix so large
/// evaluator packs remain outside the Rust heap without retaining a mutable
/// relationship to the source file. Containers constructed from bytes retain
/// owned storage.
#[derive(Debug)]
pub struct PacbinReader {
    bytes: PacbinStorage,
    index: PacbinIndex,
}

#[derive(Debug)]
enum PacbinStorage {
    Owned(Box<[u8]>),
    #[cfg(unix)]
    Mapped(ReadOnlyMmap),
}

impl AsRef<[u8]> for PacbinStorage {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            #[cfg(unix)]
            Self::Mapped(bytes) => bytes,
        }
    }
}

#[cfg(unix)]
struct ReadOnlyMmap {
    pointer: NonNull<u8>,
    length: usize,
}

#[cfg(unix)]
impl std::fmt::Debug for ReadOnlyMmap {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReadOnlyMmap")
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

#[cfg(unix)]
impl ReadOnlyMmap {
    fn snapshot(source: &mut File, path: &Path, length: usize) -> RusticolResult<(Self, [u8; 32])> {
        if length == 0 {
            return Err(RusticolError::artifact(format!(
                "cannot map empty pacbin container {}",
                path.display()
            )));
        }
        // Anonymous memory is the only portable Unix backing store that never
        // exposes a writable pathname or file descriptor. A temporary file,
        // even after unlink, can still be mutated through a descriptor opened
        // during its named lifetime. Copying directly into anonymous pages
        // also ensures that the digest and parser consume one retained image.
        let raw_pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if raw_pointer == libc::MAP_FAILED {
            return Err(RusticolError::artifact(format!(
                "could not allocate the {length}-byte anonymous immutable snapshot required for \
                 pacbin container {}: {}. Ensure the process has at least {length} bytes of \
                 available virtual memory and RAM or swap",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        let Some(pointer) = NonNull::new(raw_pointer.cast::<u8>()) else {
            // SAFETY: this pointer/length pair came from mmap above and has
            // not otherwise been retained.
            unsafe {
                libc::munmap(raw_pointer, length);
            }
            return Err(RusticolError::artifact(format!(
                "anonymous memory-mapping returned a null pointer for pacbin container {}",
                path.display()
            )));
        };
        let snapshot = Self { pointer, length };

        source.seek(SeekFrom::Start(0)).map_err(|error| {
            RusticolError::artifact(format!(
                "could not rewind pacbin container {}: {error}",
                path.display()
            ))
        })?;
        // SAFETY: `snapshot` exclusively owns a writable anonymous mapping of
        // exactly `length` bytes until `mprotect` below.
        let destination =
            unsafe { std::slice::from_raw_parts_mut(snapshot.pointer.as_ptr(), snapshot.length) };
        for chunk in destination.chunks_mut(PACBIN_DEFAULT_CHUNK_SIZE) {
            source.read_exact(chunk).map_err(|error| {
                RusticolError::artifact(format!(
                    "could not snapshot pacbin container {} into anonymous memory: {error}",
                    path.display()
                ))
            })?;
        }
        let mut trailing = [0_u8; 1];
        let trailing_count = source.read(&mut trailing).map_err(|error| {
            RusticolError::artifact(format!(
                "could not finish snapshotting pacbin container {}: {error}",
                path.display()
            ))
        })?;
        if trailing_count != 0 {
            return Err(RusticolError::artifact(format!(
                "pacbin container {} grew while being snapshotted",
                path.display()
            )));
        }

        // Seal before authentication. No retained mutable reference exists,
        // and mprotect removes write permission from the mapping itself. The
        // SHA-256 digest and all parsing therefore address the same immutable
        // bytes.
        let protection_result =
            unsafe { libc::mprotect(snapshot.pointer.as_ptr().cast(), length, libc::PROT_READ) };
        if protection_result != 0 {
            return Err(RusticolError::artifact(format!(
                "could not make the anonymous pacbin snapshot for {} read-only: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        let digest = Sha256::digest(&*snapshot).into();
        Ok((snapshot, digest))
    }
}

#[cfg(unix)]
impl Deref for ReadOnlyMmap {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: mmap established a read-only mapping of exactly `length`
        // bytes and this object keeps it alive until Drop.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
}

#[cfg(unix)]
unsafe impl Send for ReadOnlyMmap {}
#[cfg(unix)]
unsafe impl Sync for ReadOnlyMmap {}

#[cfg(unix)]
impl Drop for ReadOnlyMmap {
    fn drop(&mut self) {
        // SAFETY: this pointer/length pair came from the successful mmap in
        // `map` and is unmapped exactly once here.
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length);
        }
    }
}

impl PacbinReader {
    /// Open and fully authenticate one container file.
    pub fn open(path: impl AsRef<Path>) -> RusticolResult<Self> {
        Self::open_with_payload_verification(path, true)
    }

    /// Open and authenticate the exact mapped bytes against an enclosing
    /// manifest digest. Hashing and parsing share one storage object, so a
    /// path replacement cannot substitute a different container between them.
    #[cfg(test)]
    pub(crate) fn open_with_sha256(
        path: impl AsRef<Path>,
        expected_sha256: &[u8; 32],
    ) -> RusticolResult<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not open pacbin container {}: {error}",
                path.display()
            ))
        })?;
        Self::open_file_with_sha256(file, path, expected_sha256)
    }

    /// Authenticate and snapshot an already-open, metadata-checked container.
    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub(crate) fn open_file_with_sha256(
        file: File,
        display_path: &Path,
        expected_sha256: &[u8; 32],
    ) -> RusticolResult<Self> {
        let (storage, actual) = Self::open_file_storage(file, display_path)?;
        if &actual != expected_sha256 {
            return Err(RusticolError::integrity(format!(
                "pacbin container {} payload digest mismatch",
                display_path.display()
            )));
        }
        Self::from_storage(storage, false)
    }

    fn open_with_payload_verification(
        path: impl AsRef<Path>,
        verify_payloads: bool,
    ) -> RusticolResult<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not open pacbin container {}: {error}",
                path.display()
            ))
        })?;
        let (storage, _) = Self::open_file_storage(file, path)?;
        Self::from_storage(storage, verify_payloads)
    }

    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    fn open_file_storage(mut file: File, path: &Path) -> RusticolResult<(PacbinStorage, [u8; 32])> {
        let expected_length = file
            .metadata()
            .map_err(|error| {
                RusticolError::artifact(format!(
                    "could not inspect pacbin container {}: {error}",
                    path.display()
                ))
            })?
            .len();
        let capacity = usize::try_from(expected_length).map_err(|_| {
            RusticolError::artifact(format!(
                "pacbin container {} is too large for this platform",
                path.display()
            ))
        })?;
        #[cfg(unix)]
        let (storage, digest) = ReadOnlyMmap::snapshot(&mut file, path, capacity)?;
        #[cfg(unix)]
        let storage = PacbinStorage::Mapped(storage);
        #[cfg(not(unix))]
        let (storage, digest) = {
            let read_limit = expected_length.checked_add(1).ok_or_else(|| {
                RusticolError::artifact(format!(
                    "pacbin container {} exceeds the supported file size",
                    path.display()
                ))
            })?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(capacity).map_err(|error| {
                RusticolError::artifact(format!(
                    "could not allocate {capacity} bytes for pacbin container {}: {error}",
                    path.display()
                ))
            })?;
            file.take(read_limit)
                .read_to_end(&mut bytes)
                .map_err(|error| {
                    RusticolError::artifact(format!(
                        "could not read pacbin container {}: {error}",
                        path.display()
                    ))
                })?;
            if bytes.len() != capacity {
                return Err(RusticolError::artifact(format!(
                    "pacbin container {} changed while being read: expected {capacity} bytes, read {}",
                    path.display(),
                    bytes.len()
                )));
            }
            let digest = Sha256::digest(&bytes).into();
            (PacbinStorage::Owned(bytes.into_boxed_slice()), digest)
        };
        Ok((storage, digest))
    }

    /// Own and fully authenticate already-read container bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> RusticolResult<Self> {
        Self::from_storage(PacbinStorage::Owned(bytes.into_boxed_slice()), true)
    }

    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    fn from_storage(bytes: PacbinStorage, verify_payloads: bool) -> RusticolResult<Self> {
        let index = parse_and_validate(bytes.as_ref(), verify_payloads)?;
        Ok(Self { bytes, index })
    }

    pub fn index(&self) -> &PacbinIndex {
        &self.index
    }

    pub fn members(&self) -> &[PacbinMember] {
        self.index.members()
    }

    /// Resolve a member by normalized logical path in logarithmic time.
    pub fn member(&self, logical_path: &str) -> RusticolResult<&PacbinMember> {
        let normalized = normalize_logical_path(logical_path)?;
        self.index
            .members
            .binary_search_by(|member| member.logical_path.as_bytes().cmp(normalized.as_bytes()))
            .map(|index| &self.index.members[index])
            .map_err(|_| {
                RusticolError::invalid_argument(format!("unknown pacbin member: {normalized}"))
            })
    }

    /// Borrow a complete authenticated member without unpacking it.
    pub fn member_bytes(&self, logical_path: &str) -> RusticolResult<&[u8]> {
        let member = self.member(logical_path)?;
        member_slice(self.bytes.as_ref(), member.offset, member.length)
    }

    /// Borrow a bounded range relative to one authenticated member.
    pub fn member_range(
        &self,
        logical_path: &str,
        offset: u64,
        length: u64,
    ) -> RusticolResult<&[u8]> {
        let member = self.member(logical_path)?;
        if offset > member.length || length > member.length - offset {
            return Err(RusticolError::invalid_argument(
                "member read exceeds indexed payload bounds",
            ));
        }
        let absolute_offset = member
            .offset
            .checked_add(offset)
            .ok_or_else(|| RusticolError::invalid_argument("member read offset exceeds u64"))?;
        member_slice(self.bytes.as_ref(), absolute_offset, length)
    }

    /// Re-authenticate all member payloads held by this reader.
    pub fn verify_payloads(&self) -> RusticolResult<()> {
        verify_payload_digests(self.bytes.as_ref(), &self.index.members)
    }

    /// Return the total owned container size.
    pub fn container_size(&self) -> usize {
        self.bytes.as_ref().len()
    }
}

/// Atomically stream and publish one deterministic `pacbin-v1` container.
///
/// Member metadata is retained in memory, while payload bytes are copied and
/// hashed through one bounded buffer. The destination remains untouched until
/// a complete temporary file has been flushed and synced.
pub fn write_pacbin_atomic<'a>(
    destination: impl AsRef<Path>,
    members: impl IntoIterator<Item = PacbinWriteMember<'a>>,
    options: PacbinWriteOptions,
) -> RusticolResult<PacbinIndex> {
    let destination = destination.as_ref();
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(RusticolError::invalid_argument(format!(
            "pacbin destination directory does not exist: {}",
            parent.display()
        )));
    }
    let mut members = prepare_write_members(members)?;
    let expected_index_size = validate_write_index_bounds(&members)?;
    let (temporary_path, mut temporary_file) = create_temporary_file(destination, parent)?;

    let result = (|| {
        let index = write_pacbin_file(
            &mut temporary_file,
            &mut members,
            options.chunk_size,
            expected_index_size,
        )?;
        #[cfg(unix)]
        temporary_file
            .set_permissions(std::fs::Permissions::from_mode(options.mode))
            .map_err(|error| artifact_io_error("set permissions on", &temporary_path, error))?;
        temporary_file
            .flush()
            .map_err(|error| artifact_io_error("flush", &temporary_path, error))?;
        temporary_file
            .sync_all()
            .map_err(|error| artifact_io_error("sync", &temporary_path, error))?;
        drop(temporary_file);
        std::fs::rename(&temporary_path, destination).map_err(|error| {
            RusticolError::artifact(format!(
                "could not atomically publish pacbin container {}: {error}",
                destination.display()
            ))
        })?;
        sync_directory_best_effort(parent);
        Ok(index)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result
}

fn prepare_write_members<'a>(
    members: impl IntoIterator<Item = PacbinWriteMember<'a>>,
) -> RusticolResult<Vec<PacbinWriteMember<'a>>> {
    let mut members: Vec<_> = members.into_iter().collect();
    members.sort_by(|left, right| {
        left.logical_path
            .as_bytes()
            .cmp(right.logical_path.as_bytes())
    });
    let mut seen_paths = BTreeSet::new();
    let mut seen_folded = BTreeSet::new();
    for member in &members {
        if !seen_paths.insert(member.logical_path.clone()) {
            return Err(RusticolError::invalid_argument(format!(
                "duplicate pacbin member path: {}",
                member.logical_path
            )));
        }
        let folded = member.logical_path.to_ascii_lowercase();
        if !seen_folded.insert(folded) {
            return Err(RusticolError::invalid_argument(format!(
                "case-colliding pacbin member path: {}",
                member.logical_path
            )));
        }
    }
    Ok(members)
}

fn validate_write_index_bounds(members: &[PacbinWriteMember<'_>]) -> RusticolResult<u64> {
    u64::try_from(members.len())
        .map_err(|_| RusticolError::invalid_argument("pacbin member count exceeds u64"))?;
    let mut index_size = INDEX_HEADER_SIZE as u64;
    for member in members {
        let path_length = u64::try_from(member.logical_path.len()).map_err(|_| {
            RusticolError::invalid_argument("pacbin logical path exceeds u64 byte length")
        })?;
        if path_length > u64::from(u32::MAX) {
            return Err(RusticolError::invalid_argument(format!(
                "pacbin member path exceeds u32 byte length: {path_length} bytes"
            )));
        }
        let record_size = (INDEX_ENTRY_SIZE as u64)
            .checked_add(path_length)
            .and_then(|value| value.checked_add(padding_length(value, INDEX_ALIGNMENT)))
            .ok_or_else(|| RusticolError::invalid_argument("pacbin index size exceeds u64"))?;
        index_size = index_size
            .checked_add(record_size)
            .ok_or_else(|| RusticolError::invalid_argument("pacbin index size exceeds u64"))?;
    }
    Ok(index_size)
}

pub(crate) fn create_temporary_file(
    destination: &Path,
    parent: &Path,
) -> RusticolResult<(PathBuf, File)> {
    let file_name = destination
        .file_name()
        .ok_or_else(|| RusticolError::invalid_argument("pacbin destination must name a file"))?;
    for _ in 0..TEMPORARY_FILE_ATTEMPTS {
        let sequence = NEXT_TEMPORARY_FILE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = OsString::from(".");
        temporary_name.push(file_name);
        temporary_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
        let temporary_path = parent.join(temporary_name);
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&temporary_path) {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(artifact_io_error(
                    "create temporary pacbin container",
                    &temporary_path,
                    error,
                ));
            }
        }
    }
    Err(RusticolError::artifact(format!(
        "could not allocate a unique temporary pacbin path for {}",
        destination.display()
    )))
}

fn write_pacbin_file(
    destination: &mut File,
    members: &mut [PacbinWriteMember<'_>],
    chunk_size: usize,
    expected_index_size: u64,
) -> RusticolResult<PacbinIndex> {
    destination.seek(SeekFrom::Start(0)).map_err(|error| {
        RusticolError::artifact(format!("could not seek pacbin output: {error}"))
    })?;
    destination.set_len(0).map_err(|error| {
        RusticolError::artifact(format!("could not truncate pacbin output: {error}"))
    })?;
    write_all(destination, &[0; HEADER_SIZE], "pacbin header")?;

    let mut buffer = Vec::new();
    buffer.try_reserve_exact(chunk_size).map_err(|error| {
        RusticolError::artifact(format!(
            "could not allocate {chunk_size} byte pacbin streaming buffer: {error}"
        ))
    })?;
    buffer.resize(chunk_size, 0);

    let mut indexed_members = Vec::new();
    indexed_members
        .try_reserve_exact(members.len())
        .map_err(|error| {
            RusticolError::artifact(format!(
                "could not allocate pacbin index for {} members: {error}",
                members.len()
            ))
        })?;
    for member in members {
        write_alignment_padding(destination, u64::from(PACBIN_ALIGNMENT))?;
        let offset = stream_position(destination, "pacbin member offset")?;
        let (length, sha256) = match &mut member.source {
            PacbinWriteSource::Bytes(payload) => {
                let mut source = std::io::Cursor::new(*payload);
                stream_member(&mut source, destination, &mut buffer, &member.logical_path)?
            }
            PacbinWriteSource::Path(path) => {
                if !path.is_file() {
                    return Err(RusticolError::invalid_argument(format!(
                        "pacbin member source is not a file: {}",
                        path.display()
                    )));
                }
                let mut source = File::open(&*path)
                    .map_err(|error| artifact_io_error("open pacbin member source", path, error))?;
                stream_member(&mut source, destination, &mut buffer, &member.logical_path)?
            }
            PacbinWriteSource::Reader(source) => stream_member(
                &mut **source,
                destination,
                &mut buffer,
                &member.logical_path,
            )?,
        };
        indexed_members.push(PacbinMember {
            logical_path: member.logical_path.clone(),
            kind: member.kind,
            offset,
            length,
            sha256,
        });
    }

    write_alignment_padding(destination, u64::from(PACBIN_ALIGNMENT))?;
    let index_offset = stream_position(destination, "pacbin index offset")?;
    let member_count = u64::try_from(indexed_members.len())
        .map_err(|_| RusticolError::artifact("pacbin member count exceeds u64"))?;
    let mut index_digest = Sha256::new();
    let mut index_header = [0_u8; INDEX_HEADER_SIZE];
    index_header[0..8].copy_from_slice(INDEX_MAGIC);
    encode_u16(&mut index_header, 8, PACBIN_VERSION);
    encode_u16(&mut index_header, 10, INDEX_HEADER_SIZE as u16);
    encode_u32(&mut index_header, 12, SUPPORTED_FLAGS);
    encode_u64(&mut index_header, 16, member_count);
    write_hashed(
        destination,
        &mut index_digest,
        &index_header,
        "pacbin index",
    )?;

    for member in &indexed_members {
        let path_bytes = member.logical_path.as_bytes();
        let path_length = u32::try_from(path_bytes.len())
            .map_err(|_| RusticolError::artifact("pacbin logical path exceeds u32 byte length"))?;
        let mut entry = [0_u8; INDEX_ENTRY_SIZE];
        encode_u32(&mut entry, 0, path_length);
        encode_u16(&mut entry, 4, member.kind as u16);
        encode_u16(&mut entry, 6, SUPPORTED_FLAGS as u16);
        encode_u64(&mut entry, 8, member.offset);
        encode_u64(&mut entry, 16, member.length);
        entry[24..56].copy_from_slice(&member.sha256);
        write_hashed(destination, &mut index_digest, &entry, "pacbin index")?;
        write_hashed(destination, &mut index_digest, path_bytes, "pacbin index")?;
        let record_length = (INDEX_ENTRY_SIZE as u64) + u64::from(path_length);
        let padding = padding_length(record_length, INDEX_ALIGNMENT);
        if padding != 0 {
            let zeros = [0_u8; INDEX_ALIGNMENT as usize];
            let padding = usize::try_from(padding)
                .expect("pacbin index padding is smaller than INDEX_ALIGNMENT");
            write_hashed(
                destination,
                &mut index_digest,
                &zeros[..padding],
                "pacbin index",
            )?;
        }
    }
    let actual_index_size = stream_position(destination, "pacbin index end")?
        .checked_sub(index_offset)
        .ok_or_else(|| RusticolError::artifact("pacbin index position underflow"))?;
    if actual_index_size != expected_index_size {
        return Err(RusticolError::integrity(
            "pacbin index size disagrees with preflight",
        ));
    }
    let index_sha256: [u8; 32] = index_digest.finalize().into();

    let mut footer = [0_u8; FOOTER_SIZE];
    footer[0..8].copy_from_slice(FOOTER_MAGIC);
    encode_u16(&mut footer, 8, PACBIN_VERSION);
    encode_u16(&mut footer, 10, FOOTER_SIZE as u16);
    encode_u32(&mut footer, 12, SUPPORTED_FLAGS);
    encode_u64(&mut footer, 16, index_offset);
    encode_u64(&mut footer, 24, member_count);
    footer[32..64].copy_from_slice(&index_sha256);
    write_all(destination, &footer, "pacbin footer")?;
    let file_size = stream_position(destination, "pacbin file size")?;

    let mut header = [0_u8; HEADER_SIZE];
    header[0..8].copy_from_slice(HEADER_MAGIC);
    encode_u16(&mut header, 8, PACBIN_VERSION);
    encode_u16(&mut header, 10, HEADER_SIZE as u16);
    encode_u32(&mut header, 12, SUPPORTED_FLAGS);
    encode_u32(&mut header, 16, PACBIN_ALIGNMENT);
    encode_u64(&mut header, 24, index_offset);
    encode_u64(&mut header, 32, member_count);
    destination.seek(SeekFrom::Start(0)).map_err(|error| {
        RusticolError::artifact(format!("could not seek pacbin output: {error}"))
    })?;
    write_all(destination, &header, "pacbin header")?;
    destination
        .seek(SeekFrom::Start(file_size))
        .map_err(|error| {
            RusticolError::artifact(format!("could not seek pacbin output: {error}"))
        })?;
    destination.set_len(file_size).map_err(|error| {
        RusticolError::artifact(format!("could not truncate pacbin output: {error}"))
    })?;

    Ok(PacbinIndex {
        version: PACBIN_VERSION,
        index_offset,
        index_sha256,
        file_size,
        members: indexed_members,
    })
}

fn stream_member<R: Read + ?Sized>(
    source: &mut R,
    destination: &mut File,
    buffer: &mut [u8],
    logical_path: &str,
) -> RusticolResult<(u64, [u8; 32])> {
    let mut digest = Sha256::new();
    let mut length = 0_u64;
    loop {
        let read = source.read(buffer).map_err(|error| {
            RusticolError::artifact(format!(
                "could not read pacbin member {logical_path}: {error}"
            ))
        })?;
        if read == 0 {
            break;
        }
        length = length
            .checked_add(u64::try_from(read).expect("read byte count fits u64"))
            .ok_or_else(|| RusticolError::artifact("pacbin member length exceeds u64"))?;
        digest.update(&buffer[..read]);
        write_all(destination, &buffer[..read], "pacbin member payload")?;
    }
    Ok((length, digest.finalize().into()))
}

fn write_alignment_padding(destination: &mut File, alignment: u64) -> RusticolResult<()> {
    let position = stream_position(destination, "pacbin output position")?;
    let padding = padding_length(position, alignment);
    if padding != 0 {
        let zeros = [0_u8; PACBIN_ALIGNMENT as usize];
        let padding = usize::try_from(padding)
            .expect("pacbin payload padding is smaller than PACBIN_ALIGNMENT");
        write_all(destination, &zeros[..padding], "pacbin payload padding")?;
    }
    Ok(())
}

fn write_hashed(
    destination: &mut File,
    digest: &mut Sha256,
    payload: &[u8],
    label: &str,
) -> RusticolResult<()> {
    write_all(destination, payload, label)?;
    digest.update(payload);
    Ok(())
}

fn write_all(destination: &mut File, payload: &[u8], label: &str) -> RusticolResult<()> {
    destination.write_all(payload).map_err(|error| {
        RusticolError::artifact(format!("short write while writing {label}: {error}"))
    })
}

fn stream_position(destination: &mut File, label: &str) -> RusticolResult<u64> {
    destination
        .stream_position()
        .map_err(|error| RusticolError::artifact(format!("could not read {label}: {error}")))
}

fn encode_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn encode_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn encode_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn artifact_io_error(action: &str, path: &Path, error: std::io::Error) -> RusticolError {
    RusticolError::artifact(format!("could not {action} {}: {error}", path.display()))
}

#[cfg(unix)]
fn sync_directory_best_effort(path: &Path) {
    if let Ok(directory) = File::open(path) {
        let _ = directory.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_directory_best_effort(_path: &Path) {}

/// Normalize a portable ASCII member path using the Python codec's POSIX rules.
///
/// Generated evaluator paths are ASCII. Rejecting non-ASCII input is
/// intentional: without weakening Python's NFC and full Unicode case-folding
/// collision guarantees, the runtime accepts the complete generated-path
/// domain and fails closed for paths outside it.
pub fn normalize_logical_path(value: &str) -> RusticolResult<String> {
    normalize_logical_path_impl(value).map_err(RusticolError::invalid_argument)
}

fn normalize_logical_path_impl(value: &str) -> Result<String, String> {
    if value.is_empty() || value.contains('\0') {
        return Err("pacbin logical path must be non-empty and contain no NUL".to_string());
    }
    if !value.is_ascii() {
        return Err(
            "pacbin logical path must use the portable ASCII evaluator-path subset".to_string(),
        );
    }
    if value.contains('\\') {
        return Err("pacbin logical path must use POSIX '/' separators".to_string());
    }
    if value.starts_with('/') {
        return Err("pacbin logical path must be relative".to_string());
    }

    let mut parts = Vec::new();
    for part in value.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err("pacbin logical path must not contain '..'".to_string());
        }
        parts.push(part);
    }
    if parts.is_empty() {
        return Err("pacbin logical path must name a member".to_string());
    }
    Ok(parts.join("/"))
}

#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
#[cfg_attr(target_vendor = "apple", inline(never))]
fn parse_and_validate(bytes: &[u8], verify_payloads: bool) -> RusticolResult<PacbinIndex> {
    let minimum_size = HEADER_SIZE
        .checked_add(INDEX_HEADER_SIZE)
        .and_then(|value| value.checked_add(FOOTER_SIZE))
        .expect("pacbin fixed structure sizes fit usize");
    if bytes.len() < minimum_size {
        return Err(integrity("truncated pacbin container"));
    }
    let file_size =
        u64::try_from(bytes.len()).map_err(|_| integrity("pacbin file size exceeds u64"))?;

    let header = checked_slice(bytes, 0, HEADER_SIZE, "pacbin header")?;
    if &header[0..8] != HEADER_MAGIC {
        return Err(integrity("invalid pacbin header magic"));
    }
    let version = u16_at(header, 8, "pacbin header version")?;
    let header_size = u16_at(header, 10, "pacbin header size")?;
    let header_flags = u32_at(header, 12, "pacbin header flags")?;
    validate_contract("header", version, header_size, HEADER_SIZE, header_flags)?;
    let alignment = u32_at(header, 16, "pacbin payload alignment")?;
    if alignment != PACBIN_ALIGNMENT {
        return Err(compatibility(format!(
            "unsupported pacbin payload alignment: {alignment}"
        )));
    }
    if u32_at(header, 20, "pacbin header reserved field")? != 0
        || header[40..64].iter().any(|value| *value != 0)
    {
        return Err(integrity("pacbin header reserved fields must be zero"));
    }
    let index_offset = u64_at(header, 24, "pacbin index offset")?;
    let member_count = u64_at(header, 32, "pacbin member count")?;
    if index_offset % u64::from(PACBIN_ALIGNMENT) != 0 {
        return Err(integrity("pacbin index offset is not payload-aligned"));
    }

    let footer_offset = bytes
        .len()
        .checked_sub(FOOTER_SIZE)
        .ok_or_else(|| integrity("truncated pacbin footer"))?;
    let footer_offset_u64 =
        u64::try_from(footer_offset).map_err(|_| integrity("pacbin footer offset exceeds u64"))?;
    if index_offset < HEADER_SIZE as u64 || index_offset >= footer_offset_u64 {
        return Err(integrity("pacbin index offset is out of bounds"));
    }

    let footer = checked_slice(bytes, footer_offset, FOOTER_SIZE, "pacbin footer")?;
    if &footer[0..8] != FOOTER_MAGIC {
        return Err(integrity("invalid pacbin footer magic"));
    }
    let footer_version = u16_at(footer, 8, "pacbin footer version")?;
    let footer_size = u16_at(footer, 10, "pacbin footer size")?;
    let footer_flags = u32_at(footer, 12, "pacbin footer flags")?;
    validate_contract(
        "footer",
        footer_version,
        footer_size,
        FOOTER_SIZE,
        footer_flags,
    )?;
    if u64_at(footer, 16, "pacbin footer index offset")? != index_offset {
        return Err(integrity(
            "pacbin footer index offset disagrees with header",
        ));
    }
    if u64_at(footer, 24, "pacbin footer member count")? != member_count {
        return Err(integrity(
            "pacbin footer member count disagrees with header",
        ));
    }
    let expected_index_digest: [u8; 32] = footer[32..64]
        .try_into()
        .map_err(|_| integrity("truncated pacbin footer digest"))?;

    let index_offset_usize = usize::try_from(index_offset)
        .map_err(|_| integrity("pacbin index offset exceeds platform bounds"))?;
    let available_index_bytes = footer_offset_u64 - index_offset;
    validate_index_bounds(member_count, available_index_bytes)?;
    let index_bytes = checked_slice(
        bytes,
        index_offset_usize,
        usize::try_from(available_index_bytes)
            .map_err(|_| integrity("pacbin index size exceeds platform bounds"))?,
        "pacbin index",
    )?;
    let actual_index_digest: [u8; 32] = Sha256::digest(index_bytes).into();

    let index_header = checked_slice(index_bytes, 0, INDEX_HEADER_SIZE, "pacbin index header")?;
    if &index_header[0..8] != INDEX_MAGIC {
        return Err(integrity("invalid pacbin index magic"));
    }
    let index_version = u16_at(index_header, 8, "pacbin index version")?;
    let index_header_size = u16_at(index_header, 10, "pacbin index size")?;
    let index_flags = u32_at(index_header, 12, "pacbin index flags")?;
    validate_contract(
        "index",
        index_version,
        index_header_size,
        INDEX_HEADER_SIZE,
        index_flags,
    )?;
    if u64_at(index_header, 24, "pacbin index reserved field")? != 0 {
        return Err(integrity("pacbin index reserved field must be zero"));
    }
    if u64_at(index_header, 16, "pacbin index member count")? != member_count {
        return Err(integrity("pacbin index member count disagrees with header"));
    }

    let member_capacity = usize::try_from(member_count)
        .map_err(|_| integrity("pacbin member count exceeds platform bounds"))?;
    let mut members = Vec::new();
    members
        .try_reserve_exact(member_capacity)
        .map_err(|error| {
            integrity(format!(
                "could not allocate pacbin index for {member_count} members: {error}"
            ))
        })?;
    let mut seen_paths = BTreeSet::new();
    let mut seen_folded_paths = BTreeSet::new();
    let mut previous_path_bytes: Option<Vec<u8>> = None;
    let mut cursor = INDEX_HEADER_SIZE;
    for _ in 0..member_count {
        let entry = checked_slice(index_bytes, cursor, INDEX_ENTRY_SIZE, "pacbin index entry")?;
        let path_length = u32_at(entry, 0, "pacbin member path length")?;
        let kind = PacbinMemberKind::parse(u16_at(entry, 4, "pacbin member kind")?)?;
        let entry_flags = u16_at(entry, 6, "pacbin member flags")?;
        if entry_flags != SUPPORTED_FLAGS as u16 {
            return Err(compatibility(format!(
                "unknown pacbin member flags: {entry_flags}"
            )));
        }
        let offset = u64_at(entry, 8, "pacbin member offset")?;
        let length = u64_at(entry, 16, "pacbin member length")?;
        let sha256: [u8; 32] = entry[24..56]
            .try_into()
            .map_err(|_| integrity("truncated pacbin member digest"))?;
        cursor = cursor
            .checked_add(INDEX_ENTRY_SIZE)
            .ok_or_else(|| integrity("pacbin index cursor overflow"))?;
        let path_length_usize = usize::try_from(path_length)
            .map_err(|_| integrity("pacbin member path length exceeds platform bounds"))?;
        let path_bytes =
            checked_slice(index_bytes, cursor, path_length_usize, "pacbin member path")?;
        let logical_path = std::str::from_utf8(path_bytes)
            .map_err(|_| integrity("pacbin member path is not valid UTF-8"))?;
        let normalized = normalize_logical_path_impl(logical_path).map_err(integrity)?;
        if normalized != logical_path {
            return Err(integrity(format!(
                "pacbin member path is not canonical: {logical_path:?}"
            )));
        }
        let folded = logical_path.to_ascii_lowercase();
        if !seen_paths.insert(logical_path.to_string()) {
            return Err(integrity(format!(
                "duplicate pacbin member path: {logical_path}"
            )));
        }
        if !seen_folded_paths.insert(folded) {
            return Err(integrity(format!(
                "case-colliding pacbin member path: {logical_path}"
            )));
        }
        if previous_path_bytes
            .as_deref()
            .is_some_and(|previous| path_bytes <= previous)
        {
            return Err(integrity("pacbin index paths are not strictly sorted"));
        }
        previous_path_bytes = Some(path_bytes.to_vec());
        cursor = cursor
            .checked_add(path_length_usize)
            .ok_or_else(|| integrity("pacbin index cursor overflow"))?;
        let record_length = (INDEX_ENTRY_SIZE as u64)
            .checked_add(u64::from(path_length))
            .ok_or_else(|| integrity("pacbin index record length exceeds u64"))?;
        let padding = padding_length(record_length, INDEX_ALIGNMENT);
        let padding_usize = usize::try_from(padding)
            .map_err(|_| integrity("pacbin index padding exceeds platform bounds"))?;
        let padding_bytes =
            checked_slice(index_bytes, cursor, padding_usize, "pacbin index padding")?;
        if padding_bytes.iter().any(|value| *value != 0) {
            return Err(integrity("pacbin index padding must be zero"));
        }
        cursor = cursor
            .checked_add(padding_usize)
            .ok_or_else(|| integrity("pacbin index cursor overflow"))?;
        members.push(PacbinMember {
            logical_path: logical_path.to_string(),
            kind,
            offset,
            length,
            sha256,
        });
    }
    if cursor != index_bytes.len() {
        return Err(integrity("pacbin index has trailing or missing bytes"));
    }
    if actual_index_digest != expected_index_digest {
        return Err(integrity("pacbin index digest mismatch"));
    }

    validate_canonical_payload_layout(bytes, &members, index_offset)?;
    if verify_payloads {
        verify_payload_digests(bytes, &members)?;
    }
    Ok(PacbinIndex {
        version,
        index_offset,
        index_sha256: actual_index_digest,
        file_size,
        members,
    })
}

fn validate_contract(
    label: &str,
    version: u16,
    encoded_size: u16,
    expected_size: usize,
    flags: u32,
) -> RusticolResult<()> {
    if version != PACBIN_VERSION {
        return Err(compatibility(format!(
            "unsupported pacbin {label} version: {version}"
        )));
    }
    if usize::from(encoded_size) != expected_size {
        return Err(compatibility(format!(
            "unsupported pacbin {label} size: {encoded_size}"
        )));
    }
    if flags != SUPPORTED_FLAGS {
        return Err(compatibility(format!(
            "unknown pacbin {label} flags: {flags}"
        )));
    }
    Ok(())
}

fn validate_index_bounds(member_count: u64, available_index_bytes: u64) -> RusticolResult<()> {
    let minimum_record_size = (INDEX_ENTRY_SIZE as u64)
        .checked_add(INDEX_ALIGNMENT)
        .expect("pacbin minimum record size fits u64");
    if member_count > available_index_bytes / minimum_record_size {
        return Err(integrity("pacbin member count cannot fit in index"));
    }
    Ok(())
}

fn validate_canonical_payload_layout(
    bytes: &[u8],
    members: &[PacbinMember],
    index_offset: u64,
) -> RusticolResult<()> {
    let mut expected_offset = HEADER_SIZE as u64;
    for member in members {
        member.offset.checked_add(member.length).ok_or_else(|| {
            integrity(format!(
                "pacbin member range overflows u64: {}",
                member.logical_path
            ))
        })?;
        let unaligned_offset = expected_offset;
        let padding = padding_length(unaligned_offset, u64::from(PACBIN_ALIGNMENT));
        expected_offset = expected_offset
            .checked_add(padding)
            .ok_or_else(|| integrity("pacbin payload alignment exceeds u64"))?;
        if member.offset < expected_offset {
            return Err(integrity(format!(
                "overlapping pacbin member payload: {}",
                member.logical_path
            )));
        }
        if member.offset > expected_offset {
            return Err(integrity(format!(
                "non-canonical pacbin member gap: {}",
                member.logical_path
            )));
        }
        if member.offset % u64::from(PACBIN_ALIGNMENT) != 0 {
            return Err(integrity(format!(
                "unaligned pacbin member payload: {}",
                member.logical_path
            )));
        }
        let end = member.offset + member.length;
        if end > index_offset {
            return Err(integrity(format!(
                "pacbin member payload is out of bounds: {}",
                member.logical_path
            )));
        }
        validate_zero_region(bytes, unaligned_offset, padding, "pacbin payload padding")?;
        expected_offset = end;
    }

    let trailing_padding = padding_length(expected_offset, u64::from(PACBIN_ALIGNMENT));
    let canonical_index_offset = expected_offset
        .checked_add(trailing_padding)
        .ok_or_else(|| integrity("pacbin payload region exceeds u64"))?;
    if canonical_index_offset != index_offset {
        if canonical_index_offset > index_offset {
            return Err(integrity("pacbin payloads overlap the index"));
        }
        return Err(integrity(
            "pacbin payload region has a non-canonical trailing gap",
        ));
    }
    validate_zero_region(
        bytes,
        expected_offset,
        trailing_padding,
        "pacbin payload padding",
    )?;
    Ok(())
}

fn verify_payload_digests(bytes: &[u8], members: &[PacbinMember]) -> RusticolResult<()> {
    for member in members {
        let payload = member_slice(bytes, member.offset, member.length)?;
        let digest: [u8; 32] = Sha256::digest(payload).into();
        if digest != member.sha256 {
            return Err(integrity(format!(
                "pacbin member digest mismatch: {}",
                member.logical_path
            )));
        }
    }
    Ok(())
}

fn member_slice(bytes: &[u8], offset: u64, length: u64) -> RusticolResult<&[u8]> {
    let start = usize::try_from(offset)
        .map_err(|_| integrity("pacbin member offset exceeds platform bounds"))?;
    let length = usize::try_from(length)
        .map_err(|_| integrity("pacbin member length exceeds platform bounds"))?;
    checked_slice(bytes, start, length, "pacbin member payload")
}

fn validate_zero_region(bytes: &[u8], offset: u64, length: u64, label: &str) -> RusticolResult<()> {
    let start = usize::try_from(offset)
        .map_err(|_| integrity(format!("{label} offset exceeds platform bounds")))?;
    let length = usize::try_from(length)
        .map_err(|_| integrity(format!("{label} length exceeds platform bounds")))?;
    let region = checked_slice(bytes, start, length, label)?;
    if region.iter().any(|value| *value != 0) {
        return Err(integrity(format!("{label} must be zero")));
    }
    Ok(())
}

fn checked_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    length: usize,
    label: &str,
) -> RusticolResult<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| integrity(format!("{label} range overflow")))?;
    bytes
        .get(offset..end)
        .ok_or_else(|| integrity(format!("truncated {label}")))
}

fn u16_at(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u16> {
    let value: [u8; 2] = checked_slice(bytes, offset, 2, label)?
        .try_into()
        .map_err(|_| integrity(format!("truncated {label}")))?;
    Ok(u16::from_le_bytes(value))
}

fn u32_at(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u32> {
    let value: [u8; 4] = checked_slice(bytes, offset, 4, label)?
        .try_into()
        .map_err(|_| integrity(format!("truncated {label}")))?;
    Ok(u32::from_le_bytes(value))
}

fn u64_at(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u64> {
    let value: [u8; 8] = checked_slice(bytes, offset, 8, label)?
        .try_into()
        .map_err(|_| integrity(format!("truncated {label}")))?;
    Ok(u64::from_le_bytes(value))
}

fn padding_length(position: u64, alignment: u64) -> u64 {
    (alignment - position % alignment) % alignment
}

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(message)
}

fn compatibility(message: impl Into<String>) -> RusticolError {
    RusticolError::compatibility(message)
}

#[cfg(test)]
#[path = "pacbin_tests.rs"]
mod tests;
