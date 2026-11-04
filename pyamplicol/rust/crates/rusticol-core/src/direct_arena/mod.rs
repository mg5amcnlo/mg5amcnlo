// SPDX-License-Identifier: 0BSD

//! Lane-neutral Direct-Arena building blocks.
//!
//! This module contains no recurrence-current, eager-row, or compiled-stage
//! semantics. Each execution lane owns its authenticated plan, event schedule,
//! selectors, reductions, and hot loop.

mod abi;
mod layout;
mod profile;
mod reduction;
mod storage;

pub use abi::{
    DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView, DirectPlaneShape,
    validate_direct_views,
};
pub use layout::{
    DirectArenaAssignment, DirectArenaInterval, DirectArenaLayout, assign_direct_arena,
};
pub use profile::{DirectArenaTrafficCounters, DirectArenaTrafficKind};
pub use reduction::DirectAmplitudePlanes;
pub use storage::{
    AlignedF64Buffer, DIRECT_ARENA_ALIGNMENT, DIRECT_ARENA_LOCALITY_POINT_CAP,
    DirectArenaAllocationCounters, DirectArenaWorkspace, DirectPointTile, DirectPointTiles,
    checked_aligned_point_stride, checked_plane_scalar_len, clear_split_active_range,
    deterministic_point_tile_size, deterministic_point_tile_size_with_cache_footprint,
};
