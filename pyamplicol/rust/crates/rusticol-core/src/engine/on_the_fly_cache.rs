// SPDX-License-Identifier: 0BSD

//! One streamed snapshot of retained OTF structure, independent of numeric inputs.

use super::super::on_the_fly_lane::OnTheFlyLaneSnapshotV1;
use super::*;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};

const MAGIC: [u8; 8] = *b"PACOTFC1";
const VERSION: u32 = 1;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(bincode::Encode, bincode::Decode, Debug, PartialEq)]
struct Header {
    magic: [u8; 8],
    version: u32,
    artifact_id: String,
    process_key: String,
    external_permutation: Vec<usize>,
}

impl Header {
    fn for_runtime(runtime: &NativeRuntime) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            artifact_id: runtime.artifact_id.clone(),
            process_key: runtime.process_key.clone(),
            external_permutation: runtime.external_permutation.clone(),
        }
    }
}

fn io_error(context: &str, error: impl std::fmt::Display) -> RusticolError {
    RusticolError::serialization(format!("OTF cache {context}: {error}"))
}

struct HashWriter<W> {
    inner: W,
    hash: Sha256,
}

impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.hash.update(&bytes[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn encode<T: bincode::Encode>(value: T, writer: &mut impl Write) -> RusticolResult<()> {
    bincode::encode_into_std_write(value, writer, bincode::config::standard())
        .map(|_| ())
        .map_err(|error| io_error("encoding failed", error))
}

fn decode<T: bincode::Decode<()>>(reader: &mut impl Read) -> RusticolResult<T> {
    bincode::decode_from_std_read(reader, bincode::config::standard())
        .map_err(|error| io_error("decoding failed", error))
}

/// Write into an owned sibling temporary file, then atomically replace the
/// destination. Failed saves do not truncate a previously usable snapshot.
fn write_snapshot(
    path: &Path,
    write_body: impl FnOnce(&mut HashWriter<BufWriter<File>>) -> RusticolResult<()>,
) -> RusticolResult<()> {
    let file_name = path
        .file_name()
        .ok_or_else(|| RusticolError::invalid_argument("OTF cache path must name a file"))?;
    let mut temporary_name = file_name.to_os_string();
    temporary_name.push(format!(
        ".otf-tmp-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary = path.with_file_name(temporary_name);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| io_error("file creation failed", error))?;
    let result = (|| {
        let mut writer = HashWriter {
            inner: BufWriter::new(file),
            hash: Sha256::new(),
        };
        write_body(&mut writer)?;
        let digest = writer.hash.finalize();
        writer
            .inner
            .write_all(&digest)
            .map_err(|error| io_error("checksum write failed", error))?;
        writer
            .inner
            .flush()
            .map_err(|error| io_error("flush failed", error))?;
        drop(writer.inner);
        fs::rename(&temporary, path).map_err(|error| io_error("file replacement failed", error))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Verify the single body checksum before decoding native row indices. This
/// reads sequentially and never buffers a second copy of a large family.
fn open_snapshot(path: &Path) -> RusticolResult<std::io::Take<BufReader<File>>> {
    let file = File::open(path).map_err(|error| io_error("open failed", error))?;
    let length = file
        .metadata()
        .map_err(|error| io_error("metadata failed", error))?
        .len();
    let body_length = length
        .checked_sub(32)
        .ok_or_else(|| RusticolError::serialization("OTF cache is truncated"))?;
    let mut reader = BufReader::new(file);
    let mut hash = Sha256::new();
    let mut remaining = body_length;
    let mut buffer = [0_u8; 64 * 1024];
    while remaining != 0 {
        let size = usize::try_from(remaining.min(buffer.len() as u64)).expect("bounded read size");
        reader
            .read_exact(&mut buffer[..size])
            .map_err(|error| io_error("body read failed", error))?;
        hash.update(&buffer[..size]);
        remaining -= size as u64;
    }
    let mut expected = [0_u8; 32];
    reader
        .read_exact(&mut expected)
        .map_err(|error| io_error("checksum read failed", error))?;
    let actual: [u8; 32] = hash.finalize().into();
    if actual != expected {
        return Err(RusticolError::integrity("OTF cache checksum mismatch"));
    }
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| io_error("rewind failed", error))?;
    Ok(reader.take(body_length))
}

pub(super) fn save(
    owner: &NativeRuntime,
    runtime: &OnTheFlyExecutionRuntime,
    path: &Path,
) -> RusticolResult<()> {
    if runtime.prepared_selections.len() > 1
        || runtime.last_prepared_selection.is_some() && runtime.last_prepared_selection != Some(0)
    {
        return Err(RusticolError::internal(
            "OTF selection retention policy changed; update the cache format",
        ));
    }
    let selection = runtime
        .prepared_selections
        .first()
        .map(|selected| &selected.identity);
    let lane = runtime.lane.snapshot()?;
    write_snapshot(path, |writer| {
        encode(Header::for_runtime(owner), writer)?;
        encode(selection, writer)?;
        encode(lane, writer)
    })
}

pub(super) fn load(owner: &mut NativeRuntime, path: &Path) -> RusticolResult<()> {
    let mut reader = open_snapshot(path)?;
    let header: Header = decode(&mut reader)?;
    if header.magic != MAGIC || header.version != VERSION {
        return Err(RusticolError::compatibility(
            "unsupported OTF cache format; save it with this version",
        ));
    }
    if header != Header::for_runtime(owner) {
        return Err(RusticolError::compatibility(
            "OTF cache belongs to a different process output or external ordering",
        ));
    }
    let identity: Option<OnTheFlySelectionIdentityV1> = decode(&mut reader)?;
    let lane: OnTheFlyLaneSnapshotV1 = decode(&mut reader)?;
    if reader.limit() != 0 {
        return Err(RusticolError::serialization("OTF cache has trailing data"));
    }
    if identity.is_some() != lane.has_family() {
        return Err(RusticolError::integrity(
            "OTF cache selection and family disagree",
        ));
    }
    let NativeExecutionLane::OnTheFly(runtime) = &mut owner.execution_lane else {
        return Err(RusticolError::compatibility(
            "load_cache() requires an on-the-fly runtime",
        ));
    };
    runtime.restore_saved_selection(identity, lane)
}

impl OnTheFlyExecutionRuntime {
    fn restore_saved_selection(
        &mut self,
        identity: Option<OnTheFlySelectionIdentityV1>,
        lane: OnTheFlyLaneSnapshotV1,
    ) -> RusticolResult<()> {
        let mut restored = Vec::new();
        let mut requests = None;
        let mut contracted_count = None;
        if let Some(identity) = identity {
            let selected = self.selectors.selection_from_ordinals(
                self.lane.seed(),
                identity.helicity_ordinals.as_deref(),
                identity.color_ordinals.as_deref(),
            )?;
            let helicity_indices = (0..selected.helicity_count())
                .map(|i| selected.helicity_ordinal_at(i))
                .collect::<RusticolResult<Vec<_>>>()?;
            let helicity_ids = (0..selected.helicity_count())
                .map(|i| selected.helicity_id_at(i))
                .collect::<RusticolResult<Vec<_>>>()?;
            let (color_indices, color_ids, query_count) = match &self.reduction {
                OnTheFlyReductionV1::Lc => {
                    let indices = (0..selected.color_count())
                        .map(|i| selected.color_ordinal_at(i))
                        .collect::<RusticolResult<Vec<_>>>()?;
                    let ids = (0..selected.color_count())
                        .map(|i| selected.color_id_at(i))
                        .collect::<RusticolResult<Vec<_>>>()?;
                    let decoded = selected.iter().collect::<RusticolResult<Vec<_>>>()?;
                    let count = decoded.len();
                    requests = Some(decoded);
                    (indices, ids, count)
                }
                OnTheFlyReductionV1::Contracted(plan) => {
                    if identity.color_ordinals.is_some() {
                        return Err(RusticolError::integrity(
                            "saved contracted OTF selection contains colour selectors",
                        ));
                    }
                    let count = plan.destination_by_owner_ordinal.len();
                    contracted_count = Some(count);
                    let query_count =
                        helicity_indices.len().checked_mul(count).ok_or_else(|| {
                            RusticolError::integrity("saved OTF query count overflows")
                        })?;
                    (vec![0], vec!["color:contracted".to_string()], query_count)
                }
            };
            restored.push(OnTheFlyPreparedSelectionV1 {
                identity,
                helicity_indices: helicity_indices.into_boxed_slice(),
                color_indices: color_indices.into_boxed_slice(),
                helicity_ids: helicity_ids.into_boxed_slice(),
                color_ids: color_ids.into_boxed_slice(),
                query_count,
            });
        }
        let contracted =
            contracted_count.map(|count| (restored[0].helicity_indices.as_ref(), count));
        self.lane.restore_snapshot(lane, requests, contracted)?;
        self.last_prepared_selection = (!restored.is_empty()).then_some(0);
        self.prepared_selections = restored;
        self.pending_prepared_selection = None;
        self.point_major_scratch = Vec::new();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_snapshot_checksum_rejects_damage_and_failed_save_keeps_previous_file() {
        let path = std::env::temp_dir().join(format!(
            "rusticol-otf-cache-{}-{}.bin",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        write_snapshot(&path, |writer| encode((42_u32, vec![1_u32, 2, 3]), writer)).unwrap();
        let mut reader = open_snapshot(&path).unwrap();
        let decoded: (u32, Vec<u32>) = decode(&mut reader).unwrap();
        assert_eq!(decoded, (42, vec![1, 2, 3]));
        assert_eq!(reader.limit(), 0);
        assert!(
            write_snapshot(&path, |_| Err(RusticolError::serialization("test failure"))).is_err()
        );
        assert!(open_snapshot(&path).is_ok());
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert!(open_snapshot(&path).is_err());
        fs::write(&path, &bytes[..8]).unwrap();
        assert!(open_snapshot(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
