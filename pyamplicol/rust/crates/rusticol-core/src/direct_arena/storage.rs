// SPDX-License-Identifier: 0BSD

//! Aligned storage and deterministic, shape-derived point tiling.

use std::mem::size_of;

use crate::{RusticolError, RusticolResult};

use super::DirectArenaView;

pub const DIRECT_ARENA_ALIGNMENT: usize = 64;
const DIRECT_ARENA_ALIGNMENT_SCALARS: u32 = (DIRECT_ARENA_ALIGNMENT / size_of::<f64>()) as u32;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Binary64 storage exposing an exactly-sized 64-byte-aligned logical range.
///
/// The small alignment reserve belongs to the allocator and is intentionally
/// outside any caller-authenticated logical workspace budget.
pub struct AlignedF64Buffer {
    storage: Vec<f64>,
    start: usize,
    len: usize,
}

impl AlignedF64Buffer {
    pub fn zeroed(len: usize, label: &str) -> RusticolResult<Self> {
        if len == 0 {
            return Ok(Self {
                storage: Vec::new(),
                start: 0,
                len: 0,
            });
        }
        let alignment_values = DIRECT_ARENA_ALIGNMENT
            .checked_div(size_of::<f64>())
            .filter(|value| *value != 0)
            .ok_or_else(|| invalid("arena alignment is invalid for binary64 storage"))?;
        let storage_len = len
            .checked_add(alignment_values - 1)
            .ok_or_else(|| invalid(format!("{label} arena length overflows usize")))?;
        let mut storage = Vec::new();
        storage.try_reserve_exact(storage_len).map_err(|error| {
            RusticolError::internal(format!(
                "could not allocate Direct-Arena {label} storage: {error}"
            ))
        })?;
        storage.resize(storage_len, 0.0);
        let address = storage.as_ptr() as usize;
        let byte_offset =
            (DIRECT_ARENA_ALIGNMENT - address % DIRECT_ARENA_ALIGNMENT) % DIRECT_ARENA_ALIGNMENT;
        if !byte_offset.is_multiple_of(size_of::<f64>()) {
            return Err(RusticolError::internal(
                "Direct-Arena allocation cannot provide binary64 alignment",
            ));
        }
        let start = byte_offset / size_of::<f64>();
        if start.checked_add(len).is_none_or(|end| end > storage.len()) {
            return Err(RusticolError::internal(
                "aligned Direct-Arena range exceeds its allocation",
            ));
        }
        Ok(Self {
            storage,
            start,
            len,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const f64 {
        self.as_slice().as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.as_mut_slice().as_mut_ptr()
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.storage[self.start..self.start + self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.storage[self.start..self.start + self.len]
    }

    pub fn allocation_counters(&self) -> DirectArenaAllocationCounters {
        DirectArenaAllocationCounters {
            allocation_requests: u64::from(!self.is_empty()),
            requested_bytes: u64::try_from(self.storage.len().saturating_mul(size_of::<f64>()))
                .unwrap_or(u64::MAX),
        }
    }
}

/// Logical allocation requests made for persistent Direct-Arena buffers.
///
/// `requested_bytes` describes the buffer lengths requested from `Vec`; an
/// allocator may reserve more physical capacity. Use a counting allocator
/// when actual allocator calls or resident bytes are required.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectArenaAllocationCounters {
    pub allocation_requests: u64,
    pub requested_bytes: u64,
}

impl DirectArenaAllocationCounters {
    pub fn checked_add(self, other: Self) -> RusticolResult<Self> {
        Ok(Self {
            allocation_requests: self
                .allocation_requests
                .checked_add(other.allocation_requests)
                .ok_or_else(|| invalid("Direct-Arena allocation request count overflows u64"))?,
            requested_bytes: self
                .requested_bytes
                .checked_add(other.requested_bytes)
                .ok_or_else(|| invalid("Direct-Arena requested byte count overflows u64"))?,
        })
    }
}

/// Persistent split-complex current and amplitude storage for one lane.
pub struct DirectArenaWorkspace {
    current_re: AlignedF64Buffer,
    current_im: AlignedF64Buffer,
    amplitude_re: AlignedF64Buffer,
    amplitude_im: AlignedF64Buffer,
    current_plane_count: u32,
    amplitude_plane_count: u32,
    tile_capacity: u32,
    point_stride: u32,
    active_point_count: u32,
    allocation_counters: DirectArenaAllocationCounters,
}

impl DirectArenaWorkspace {
    pub fn new(
        current_plane_count: u32,
        amplitude_plane_count: u32,
        tile_capacity: u32,
    ) -> RusticolResult<Self> {
        if tile_capacity == 0 {
            return Err(invalid(
                "Direct-Arena workspace tile capacity must be positive",
            ));
        }
        let point_stride = checked_aligned_point_stride(tile_capacity)?;
        Self::new_with_layout(
            current_plane_count,
            amplitude_plane_count,
            tile_capacity,
            point_stride,
        )
    }

    /// Allocate the explicit packed layout used only for a proven-generic
    /// singleton direct execution. Ordinary capacity-one workspaces remain
    /// aligned through [`Self::new`].
    pub(crate) fn new_packed_singleton(
        current_plane_count: u32,
        amplitude_plane_count: u32,
    ) -> RusticolResult<Self> {
        Self::new_with_layout(current_plane_count, amplitude_plane_count, 1, 1)
    }

    fn new_with_layout(
        current_plane_count: u32,
        amplitude_plane_count: u32,
        tile_capacity: u32,
        point_stride: u32,
    ) -> RusticolResult<Self> {
        let current_len =
            checked_plane_scalar_len(current_plane_count, point_stride, "current arena")?;
        let amplitude_len =
            checked_plane_scalar_len(amplitude_plane_count, point_stride, "amplitude arena")?;
        let current_re = AlignedF64Buffer::zeroed(current_len, "current real")?;
        let current_im = AlignedF64Buffer::zeroed(current_len, "current imaginary")?;
        let amplitude_re = AlignedF64Buffer::zeroed(amplitude_len, "amplitude real")?;
        let amplitude_im = AlignedF64Buffer::zeroed(amplitude_len, "amplitude imaginary")?;
        let allocation_counters = [&current_re, &current_im, &amplitude_re, &amplitude_im]
            .into_iter()
            .try_fold(DirectArenaAllocationCounters::default(), |total, buffer| {
                total.checked_add(buffer.allocation_counters())
            })?;
        Ok(Self {
            current_re,
            current_im,
            amplitude_re,
            amplitude_im,
            current_plane_count,
            amplitude_plane_count,
            tile_capacity,
            point_stride,
            active_point_count: 0,
            allocation_counters,
        })
    }

    pub const fn tile_capacity(&self) -> u32 {
        self.tile_capacity
    }

    pub const fn point_stride(&self) -> u32 {
        self.point_stride
    }

    pub const fn active_point_count(&self) -> u32 {
        self.active_point_count
    }

    pub const fn allocation_counters(&self) -> DirectArenaAllocationCounters {
        self.allocation_counters
    }

    pub fn current_slices(&self) -> (&[f64], &[f64]) {
        (self.current_re.as_slice(), self.current_im.as_slice())
    }

    pub fn amplitude_slices(&self) -> (&[f64], &[f64]) {
        (self.amplitude_re.as_slice(), self.amplitude_im.as_slice())
    }

    pub fn split_slices_mut(&mut self) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        (
            self.current_re.as_mut_slice(),
            self.current_im.as_mut_slice(),
            self.amplitude_re.as_mut_slice(),
            self.amplitude_im.as_mut_slice(),
        )
    }

    /// Bind the persistent storage to one active tile without modifying it.
    ///
    /// The physical pitch and pointer identity remain stable across full and
    /// tail tiles. Only `0..point_count` in each plane is semantically active.
    pub fn begin_tile(&mut self, point_count: u32) -> RusticolResult<()> {
        if point_count == 0 || point_count > self.tile_capacity {
            return Err(invalid(format!(
                "Direct-Arena workspace point count {point_count} is outside 1..={}",
                self.tile_capacity
            )));
        }
        self.active_point_count = point_count;
        Ok(())
    }

    pub fn clear_current_active(
        &mut self,
        component_base: u32,
        component_count: u32,
    ) -> RusticolResult<()> {
        clear_split_active_range(
            self.current_re.as_mut_slice(),
            self.current_im.as_mut_slice(),
            self.current_plane_count,
            self.point_stride,
            self.active_point_count,
            component_base,
            component_count,
            "current arena",
        )
    }

    pub fn clear_amplitude_active(
        &mut self,
        component_base: u32,
        component_count: u32,
    ) -> RusticolResult<()> {
        clear_split_active_range(
            self.amplitude_re.as_mut_slice(),
            self.amplitude_im.as_mut_slice(),
            self.amplitude_plane_count,
            self.point_stride,
            self.active_point_count,
            component_base,
            component_count,
            "amplitude arena",
        )
    }

    pub fn view(&mut self) -> RusticolResult<DirectArenaView> {
        if self.active_point_count == 0 {
            return Err(invalid(
                "Direct-Arena workspace view requires an active tile",
            ));
        }
        let view = DirectArenaView {
            current_re: self.current_re.as_mut_ptr(),
            current_im: self.current_im.as_mut_ptr(),
            current_scalar_len: u64::try_from(self.current_re.len())
                .map_err(|_| invalid("current arena view length exceeds u64"))?,
            amplitude_re: self.amplitude_re.as_mut_ptr(),
            amplitude_im: self.amplitude_im.as_mut_ptr(),
            amplitude_scalar_len: u64::try_from(self.amplitude_re.len())
                .map_err(|_| invalid("amplitude arena view length exceeds u64"))?,
            point_stride: self.point_stride,
        };
        view.validate()?;
        Ok(view)
    }

    pub fn point_tiles(
        &self,
        requested_points: u32,
        tile_points: u32,
    ) -> RusticolResult<DirectPointTiles> {
        if requested_points == 0 || tile_points == 0 || tile_points > self.tile_capacity {
            return Err(invalid(format!(
                "Direct-Arena point tiling requires positive counts and tile size at most {}",
                self.tile_capacity
            )));
        }
        Ok(DirectPointTiles {
            next_point: 0,
            requested_points,
            tile_points,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectPointTile {
    pub point_start: u32,
    pub point_count: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DirectPointTiles {
    next_point: u32,
    requested_points: u32,
    tile_points: u32,
}

impl Iterator for DirectPointTiles {
    type Item = DirectPointTile;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_point == self.requested_points {
            return None;
        }
        let point_count = (self.requested_points - self.next_point).min(self.tile_points);
        let tile = DirectPointTile {
            point_start: self.next_point,
            point_count,
        };
        self.next_point += point_count;
        Some(tile)
    }
}

pub fn checked_aligned_point_stride(tile_capacity: u32) -> RusticolResult<u32> {
    if DIRECT_ARENA_ALIGNMENT_SCALARS == 0 {
        return Err(RusticolError::internal(
            "Direct-Arena alignment is smaller than binary64 storage",
        ));
    }
    let remainder = tile_capacity % DIRECT_ARENA_ALIGNMENT_SCALARS;
    if remainder == 0 {
        return Ok(tile_capacity);
    }
    tile_capacity
        .checked_add(DIRECT_ARENA_ALIGNMENT_SCALARS - remainder)
        .ok_or_else(|| invalid("aligned Direct-Arena point stride exceeds u32"))
}

/// Clear only active points in a validated contiguous split-complex plane range.
#[allow(clippy::too_many_arguments)]
pub fn clear_split_active_range(
    values_re: &mut [f64],
    values_im: &mut [f64],
    plane_count: u32,
    point_stride: u32,
    active_point_count: u32,
    component_base: u32,
    component_count: u32,
    label: &str,
) -> RusticolResult<()> {
    if active_point_count == 0 || active_point_count > point_stride {
        return Err(invalid(format!(
            "direct {label} clear requires an active count within its physical pitch"
        )));
    }
    if component_count == 0 {
        return Err(invalid(format!(
            "direct {label} clear range must contain a component"
        )));
    }
    let expected_len = checked_plane_scalar_len(plane_count, point_stride, label)?;
    if values_re.len() != expected_len || values_im.len() != expected_len {
        return Err(invalid(format!(
            "direct {label} split storage has an inconsistent length"
        )));
    }
    let component_stop = component_base
        .checked_add(component_count)
        .filter(|stop| *stop <= plane_count)
        .ok_or_else(|| invalid(format!("direct {label} clear range is out of bounds")))?;
    let point_stride = point_stride as usize;
    let active_point_count = active_point_count as usize;
    for component in component_base..component_stop {
        let start = (component as usize)
            .checked_mul(point_stride)
            .ok_or_else(|| invalid(format!("direct {label} clear range overflows usize")))?;
        let stop = start
            .checked_add(active_point_count)
            .ok_or_else(|| invalid(format!("direct {label} clear range overflows usize")))?;
        values_re[start..stop].fill(0.0);
        values_im[start..stop].fill(0.0);
    }
    Ok(())
}

pub fn checked_plane_scalar_len(
    plane_count: u32,
    point_stride: u32,
    label: &str,
) -> RusticolResult<usize> {
    usize::try_from(plane_count)
        .ok()
        .and_then(|planes| planes.checked_mul(point_stride as usize))
        .ok_or_else(|| invalid(format!("{label} length overflows usize")))
}

/// Portable upper bound for the physical point pitch of row-outer Direct-Arena lanes.
///
/// Table callables revisit their parent component planes for every row. A
/// larger pitch spreads those planes farther apart and expands the live parent
/// footprint even when the active point count is smaller than the allocated
/// capacity. Keeping the pitch bounded preserves locality without consulting
/// the host architecture or runtime timings. Lanes remain free to request a
/// smaller tile or apply a smaller workspace target.
pub const DIRECT_ARENA_LOCALITY_POINT_CAP: u32 = 64;

/// Select a semantic tile capacity using its padded footprint.
///
/// `scalar_values_per_point` includes every physically pitched plane and both
/// halves of split-complex planes. The workspace target excludes the
/// allocator's small base-alignment reserve and grows to fit one aligned tile.
/// It controls batching, not which processes can run. `cache_target_bytes`
/// is a separate locality target; lanes may pass `usize::MAX` and apply their
/// own cache heuristic.
pub fn deterministic_point_tile_size(
    requested_points: u32,
    workspace_bytes: usize,
    cache_target_bytes: usize,
    scalar_values_per_point: usize,
) -> RusticolResult<u32> {
    deterministic_point_tile_size_with_cache_footprint(
        requested_points,
        workspace_bytes,
        cache_target_bytes,
        scalar_values_per_point,
        scalar_values_per_point,
    )
}

/// Select a semantic tile using distinct physical and cache-local footprints.
///
/// The workspace target uses `physical_scalar_values_per_point`, including
/// the final aligned pitch. The cache target instead uses the maximum
/// phase-local working set. Keeping
/// the two inputs separate prevents cold persistent planes from shrinking a
/// tile whose hot kernels touch only a bounded subset.
pub fn deterministic_point_tile_size_with_cache_footprint(
    requested_points: u32,
    workspace_bytes: usize,
    cache_target_bytes: usize,
    physical_scalar_values_per_point: usize,
    cache_scalar_values_per_point: usize,
) -> RusticolResult<u32> {
    if requested_points == 0
        || physical_scalar_values_per_point == 0
        || cache_scalar_values_per_point == 0
    {
        return Err(invalid(
            "Direct-Arena tiling requires positive points and physical/cache per-point shapes",
        ));
    }
    let physical_bytes_per_point = physical_scalar_values_per_point
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| invalid("Direct-Arena per-point byte count overflows usize"))?;
    let cache_bytes_per_point = cache_scalar_values_per_point
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| invalid("Direct-Arena cache footprint overflows usize"))?;
    let minimum_physical_bytes = physical_bytes_per_point
        .checked_mul(DIRECT_ARENA_ALIGNMENT_SCALARS as usize)
        .ok_or_else(|| invalid("minimum aligned Direct-Arena tile byte count overflows usize"))?;
    let workspace_bytes = workspace_bytes.max(minimum_physical_bytes);
    let largest_aligned_u32 = u32::MAX - u32::MAX % DIRECT_ARENA_ALIGNMENT_SCALARS;
    let workspace_stride = (workspace_bytes / physical_bytes_per_point)
        .min(largest_aligned_u32 as usize)
        / DIRECT_ARENA_ALIGNMENT_SCALARS as usize
        * DIRECT_ARENA_ALIGNMENT_SCALARS as usize;
    let workspace_capacity = u32::try_from(workspace_stride)
        .map_err(|_| invalid("aligned Direct-Arena workspace capacity exceeds u32"))?;
    let cache_points =
        greatest_power_of_two_not_exceeding((cache_target_bytes / cache_bytes_per_point).max(1));
    let tile_capacity = requested_points
        .min(workspace_capacity)
        .min(u32::try_from(cache_points).unwrap_or(u32::MAX).max(1));
    let point_stride = checked_aligned_point_stride(tile_capacity)?;
    let physical_bytes = (point_stride as usize)
        .checked_mul(physical_bytes_per_point)
        .ok_or_else(|| invalid("padded Direct-Arena tile byte count overflows usize"))?;
    if physical_bytes > workspace_bytes {
        return Err(invalid(format!(
            "padded Direct-Arena tile accounting mismatch: {physical_bytes} bytes \
             exceeds derived workspace size {workspace_bytes}"
        )));
    }
    Ok(tile_capacity)
}

fn greatest_power_of_two_not_exceeding(value: usize) -> usize {
    if value == 0 {
        return 1;
    }
    1_usize << (usize::BITS - 1 - value.leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_and_each_physical_plane_are_aligned() {
        let values = AlignedF64Buffer::zeroed(129, "test").unwrap();
        assert_eq!(values.len(), 129);
        assert_eq!((values.as_ptr() as usize) % DIRECT_ARENA_ALIGNMENT, 0);
        for (capacity, stride) in [(1, 8), (7, 8), (127, 128), (129, 136)] {
            let mut workspace = DirectArenaWorkspace::new(3, 2, capacity).unwrap();
            assert_eq!(workspace.point_stride(), stride);
            workspace.begin_tile(capacity).unwrap();
            let view = workspace.view().unwrap();
            for component in 0..3_usize {
                assert_eq!(
                    unsafe { view.current_re.add(component * stride as usize) } as usize
                        % DIRECT_ARENA_ALIGNMENT,
                    0
                );
            }
        }
    }

    #[test]
    fn tiling_uses_padded_physical_workspace_target() {
        assert_eq!(
            deterministic_point_tile_size(129, 2064, usize::MAX, 2).unwrap(),
            128
        );
        assert_eq!(
            deterministic_point_tile_size(129, 2176, usize::MAX, 2).unwrap(),
            129
        );
        assert_eq!(
            deterministic_point_tile_size(129, 2175, usize::MAX, 2).unwrap(),
            128
        );
        assert_eq!(
            deterministic_point_tile_size(1, 127, usize::MAX, 2).unwrap(),
            1
        );
        assert_eq!(
            deterministic_point_tile_size(1, 128, usize::MAX, 2).unwrap(),
            1
        );
    }

    #[test]
    fn tiling_separates_physical_and_cache_footprints() {
        assert_eq!(
            deterministic_point_tile_size_with_cache_footprint(32, 32 * 1024, 4 * 1024, 128, 8,)
                .unwrap(),
            32,
            "cold physical planes must not consume the phase-local cache target"
        );
        assert_eq!(
            deterministic_point_tile_size_with_cache_footprint(64, 16 * 1024, usize::MAX, 128, 1,)
                .unwrap(),
            16,
            "the padded physical allocation controls the tile size"
        );
        assert_eq!(
            deterministic_point_tile_size_with_cache_footprint(64, usize::MAX, 4 * 1024, 1, 128,)
                .unwrap(),
            4,
            "the cache bound follows the phase-local footprint"
        );
        assert!(
            deterministic_point_tile_size_with_cache_footprint(1, usize::MAX, usize::MAX, 1, 0,)
                .is_err()
        );
    }

    #[test]
    fn observed_large_minimum_pitch_can_exceed_the_workspace_target() {
        const PHYSICAL_SCALARS_PER_POINT: usize = 4_249_960;
        const REQUIRED_BYTES: usize = 271_997_440;

        assert_eq!(
            deterministic_point_tile_size_with_cache_footprint(
                1024,
                REQUIRED_BYTES,
                4 * 1024 * 1024,
                PHYSICAL_SCALARS_PER_POINT,
                1,
            )
            .unwrap(),
            8
        );
        assert_eq!(
            deterministic_point_tile_size_with_cache_footprint(
                1024,
                REQUIRED_BYTES - 1,
                4 * 1024 * 1024,
                PHYSICAL_SCALARS_PER_POINT,
                1,
            )
            .unwrap(),
            8
        );
    }

    #[test]
    fn allocator_alignment_reserve_is_outside_the_logical_budget() {
        assert_eq!(
            deterministic_point_tile_size(1, 128, usize::MAX, 2).unwrap(),
            1
        );
        let workspace = DirectArenaWorkspace::new(1, 0, 1).unwrap();
        assert_eq!(workspace.point_stride(), 8);
        assert_eq!(workspace.allocation_counters().allocation_requests, 2);
        assert!(
            workspace.allocation_counters().requested_bytes > 128,
            "allocator-owned base-alignment reserve must not reduce logical capacity"
        );
    }

    #[test]
    fn packed_singleton_is_explicit_and_does_not_change_aligned_capacity_one() {
        assert_eq!(
            DirectArenaWorkspace::new(2, 1, 1).unwrap().point_stride(),
            8
        );
        let packed = DirectArenaWorkspace::new_packed_singleton(2, 1).unwrap();
        assert_eq!(packed.tile_capacity(), 1);
        assert_eq!(packed.point_stride(), 1);
        assert_eq!(packed.current_slices().0.len(), 2);
        assert_eq!(packed.amplitude_slices().0.len(), 1);
    }

    #[test]
    fn selective_clear_preserves_padding_and_other_planes() {
        let mut workspace = DirectArenaWorkspace::new(2, 1, 9).unwrap();
        workspace.begin_tile(9).unwrap();
        {
            let (current_re, current_im, amplitude_re, amplitude_im) = workspace.split_slices_mut();
            current_re.fill(7.0);
            current_im.fill(-7.0);
            amplitude_re.fill(3.0);
            amplitude_im.fill(-3.0);
        }
        workspace.begin_tile(7).unwrap();
        workspace.clear_current_active(1, 1).unwrap();
        workspace.clear_amplitude_active(0, 1).unwrap();
        let stride = workspace.point_stride() as usize;
        let (current_re, current_im) = workspace.current_slices();
        assert!(current_re[..stride].iter().all(|value| *value == 7.0));
        assert!(
            current_re[stride..stride + 7]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert_eq!(current_re[stride + 7], 7.0);
        assert_eq!(current_im[stride + 7], -7.0);
        let (amplitude_re, amplitude_im) = workspace.amplitude_slices();
        assert!(amplitude_re[..7].iter().all(|value| *value == 0.0));
        assert_eq!(amplitude_re[7], 3.0);
        assert_eq!(amplitude_im[7], -3.0);
    }
}
