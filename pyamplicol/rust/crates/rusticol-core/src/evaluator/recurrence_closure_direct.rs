// SPDX-License-Identifier: 0BSD

//! Allocation-free component-metric closures for Direct-Arena recurrence.

use crate::recurrence::DirectClosureRow;
use crate::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
};
use std::ffi::{c_int, c_void};

const STATUS_INVALID_ARGUMENT: c_int = 2;
const STATUS_BOUNDS: c_int = 4;

/// Contract two split-complex currents directly into amplitude planes.
///
/// Coefficient planes are stored in the authenticated plan factor catalog.
/// The point loop is innermost so each current and destination access remains
/// contiguous and auto-vectorizable.
pub(crate) unsafe extern "C" fn execute_closure_reduce_rows(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectClosureRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    if point_count == 0
        || point_count > arena.point_stride
        || (row_count != 0 && rows.is_null())
        || arena.current_re.is_null()
        || arena.current_im.is_null()
        || arena.amplitude_re.is_null()
        || arena.amplitude_im.is_null()
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    let stride = arena.point_stride as usize;
    let points = point_count as usize;

    for row in rows {
        if row.parent1_component_base_or_sentinel == u32::MAX
            || row.component_count == 0
            || row.exact_factor_id >= factors.value_count
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let coefficient_end = match row
            .component_factor_start
            .checked_add(u32::from(row.component_count))
        {
            Some(end) if end <= factors.value_count => end,
            _ => return STATUS_BOUNDS,
        };
        debug_assert!(coefficient_end <= factors.value_count);

        let row_factor_re = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        let row_factor_im = unsafe { *factors.values_im.add(row.exact_factor_id as usize) };
        let destination_base = match (row.amplitude_destination_id as usize).checked_mul(stride) {
            Some(value) => value,
            None => return STATUS_BOUNDS,
        };
        if destination_base
            .checked_add(points)
            .is_none_or(|end| end > arena.amplitude_scalar_len as usize)
        {
            return STATUS_BOUNDS;
        }

        for component in 0..u32::from(row.component_count) {
            let coefficient_id = row.component_factor_start + component;
            let coefficient_re = unsafe { *factors.values_re.add(coefficient_id as usize) };
            let coefficient_im = unsafe { *factors.values_im.add(coefficient_id as usize) };
            let scale_re = coefficient_re * row_factor_re - coefficient_im * row_factor_im;
            let scale_im = coefficient_re * row_factor_im + coefficient_im * row_factor_re;

            let left_plane = match (row.parent0_component_base as usize)
                .checked_add(component as usize)
                .and_then(|plane| plane.checked_mul(stride))
            {
                Some(value) => value,
                None => return STATUS_BOUNDS,
            };
            let right_plane = match (row.parent1_component_base_or_sentinel as usize)
                .checked_add(component as usize)
                .and_then(|plane| plane.checked_mul(stride))
            {
                Some(value) => value,
                None => return STATUS_BOUNDS,
            };
            if left_plane
                .checked_add(points)
                .is_none_or(|end| end > arena.current_scalar_len as usize)
                || right_plane
                    .checked_add(points)
                    .is_none_or(|end| end > arena.current_scalar_len as usize)
            {
                return STATUS_BOUNDS;
            }

            for point in 0..points {
                let left = left_plane + point;
                let right = right_plane + point;
                let destination = destination_base + point;
                let left_re = unsafe { *arena.current_re.add(left) };
                let left_im = unsafe { *arena.current_im.add(left) };
                let right_re = unsafe { *arena.current_re.add(right) };
                let right_im = unsafe { *arena.current_im.add(right) };
                let product_re = left_re * right_re - left_im * right_im;
                let product_im = left_re * right_im + left_im * right_re;
                unsafe {
                    *arena.amplitude_re.add(destination) +=
                        product_re * scale_re - product_im * scale_im;
                    *arena.amplitude_im.add(destination) +=
                        product_re * scale_im + product_im * scale_re;
                }
            }
        }
    }
    DIRECT_STATUS_OK
}
