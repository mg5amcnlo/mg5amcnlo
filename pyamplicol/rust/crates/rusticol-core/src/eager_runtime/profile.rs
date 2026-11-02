// SPDX-License-Identifier: 0BSD

//! Instrumented eager execution used only by the explicit native profile path.
//!
//! The ordinary scheduler deliberately stays in `execute.rs` without timer
//! branches or `Instant` calls. These routines mirror its row-major packet
//! contract while placing timers exactly at the phase boundaries exposed to
//! users.

use super::plan::{ClosureExecutionRows, ComponentRange, EagerStagePlan};
use super::runtime::{
    EagerWorkspace, KernelPacket, LinearFinalizationPacket, PacketRole, StageSchedule,
};
use super::{
    EagerComplex64, EagerExecutionProfile, EagerKernelBackend, EagerKernelCall, EagerKernelSpec,
};
use crate::{RusticolError, RusticolResult};
use std::collections::BTreeMap;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // Retained by the explicit packet-scheduler profiling oracle.
pub(super) fn execute_stage_profiled<B: EagerKernelBackend>(
    stage: &EagerStagePlan,
    schedule: &StageSchedule,
    kernels: &BTreeMap<u32, EagerKernelSpec>,
    workspace: &mut EagerWorkspace,
    backend: &mut B,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
    profile: &mut EagerExecutionProfile,
) -> RusticolResult<()> {
    let clear_started = Instant::now();
    for range in &stage.zero_current_ranges {
        for component in range.start..range.start + range.len {
            let target = component * workspace.tile_capacity;
            workspace.currents[target..target + tile_points].fill(EagerComplex64::new(0.0, 0.0));
        }
    }
    profile.invocation_scatter += clear_started.elapsed();
    for packet in &schedule.invocation_packets {
        debug_assert_eq!(packet.role, PacketRole::Invocation);
        let items = &stage.invocations[packet.item_range.clone()];
        let (lane_count, evaluator_input_components, evaluator_output_components) =
            super::execute::invocation_packet_shape(packet, items.len(), tile_points)?;
        let input_len = evaluator_input_components * lane_count;
        let output_len = evaluator_output_components * lane_count;
        let (inputs, outputs) = packet_slices(&mut workspace.packet, input_len, output_len)?;
        let kernel = kernels.get(&packet.kernel_id).ok_or_else(|| {
            RusticolError::internal(format!(
                "eager schedule lost invocation kernel {}",
                packet.kernel_id
            ))
        })?;

        let gather_started = Instant::now();
        if packet.independent_block_size == 1 {
            super::execute::gather_invocations(
                items,
                &kernel.inputs,
                inputs,
                packet.input_components,
                point_count,
                tile_start,
                tile_points,
                workspace.tile_capacity,
                &workspace.values,
                momenta,
                &workspace.couplings,
                model_parameters,
            );
        } else {
            super::execute::gather_blocked_invocations(
                items,
                &kernel.inputs,
                inputs,
                packet.input_components,
                packet.independent_block_size,
                tile_points,
                workspace.tile_capacity,
                &workspace.values,
            );
        }
        profile.gather += gather_started.elapsed();

        let kernel_started = Instant::now();
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            independent_block_size: packet.independent_block_size as u32,
            lane_count,
            input_component_count: evaluator_input_components,
            output_component_count: evaluator_output_components,
            inputs,
            outputs,
        })?;
        profile.kernel_call += kernel_started.elapsed();
        profile.backend_call_count = profile.backend_call_count.saturating_add(1);

        let scatter_started = Instant::now();
        if packet.independent_block_size == 1 {
            super::execute::scatter_invocations(
                items,
                &stage.attachments,
                outputs,
                lane_count,
                tile_points,
                workspace.tile_capacity,
                &workspace.couplings,
                &mut workspace.currents,
            );
        } else {
            super::execute::scatter_blocked_invocations(
                items,
                &stage.attachments,
                outputs,
                packet.output_components,
                packet.independent_block_size,
                tile_points,
                workspace.tile_capacity,
                &workspace.couplings,
                &mut workspace.currents,
            );
        }
        profile.invocation_scatter += scatter_started.elapsed();
    }

    let copies_started = Instant::now();
    for item in &stage.finalization_copies {
        copy_component_range(
            &workspace.currents,
            item.current,
            &mut workspace.values,
            item.unpropagated,
            workspace.tile_capacity,
            tile_points,
        );
    }
    profile.finalization += copies_started.elapsed();

    for packet in &schedule.finalization_packets {
        debug_assert_eq!(packet.role, PacketRole::Finalization);
        let items = &stage.finalizations[packet.item_range.clone()];
        let lane_count = packet
            .linear_finalization
            .as_ref()
            .map_or(items.len(), LinearFinalizationPacket::lane_multiplier)
            * tile_points;
        let input_len = packet.input_components * lane_count;
        let output_len = packet.output_components * lane_count;
        let (inputs, outputs) = packet_slices(&mut workspace.packet, input_len, output_len)?;
        let kernel = kernels.get(&packet.kernel_id).ok_or_else(|| {
            RusticolError::internal(format!(
                "eager schedule lost finalization kernel {}",
                packet.kernel_id
            ))
        })?;

        let gather_started = Instant::now();
        if let Some(linear) = &packet.linear_finalization {
            super::execute::gather_linear_finalizations(
                linear,
                &kernel.inputs,
                inputs,
                packet.input_components,
                point_count,
                tile_start,
                tile_points,
                momenta,
                model_parameters,
            );
        } else {
            super::execute::gather_finalizations(
                items,
                &kernel.inputs,
                inputs,
                packet.input_components,
                point_count,
                tile_start,
                tile_points,
                workspace.tile_capacity,
                &workspace.currents,
                momenta,
                model_parameters,
            );
        }
        profile.finalization += gather_started.elapsed();

        let kernel_started = Instant::now();
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            independent_block_size: 1,
            lane_count,
            input_component_count: packet.input_components,
            output_component_count: packet.output_components,
            inputs,
            outputs,
        })?;
        profile.kernel_call += kernel_started.elapsed();
        profile.backend_call_count = profile.backend_call_count.saturating_add(1);

        let scatter_started = Instant::now();
        if let Some(linear) = &packet.linear_finalization {
            super::execute::scatter_linear_finalizations(
                items,
                linear,
                outputs,
                lane_count,
                tile_points,
                workspace.tile_capacity,
                &workspace.currents,
                &mut workspace.values,
            )?;
        } else {
            super::execute::scatter_finalizations(
                items,
                outputs,
                lane_count,
                tile_points,
                workspace.tile_capacity,
                &mut workspace.values,
            )?;
        }
        profile.finalization += scatter_started.elapsed();
    }
    Ok(())
}

#[allow(dead_code)] // Retained by the explicit packet-scheduler profiling oracle.
pub(super) fn execute_closures_profiled<B: EagerKernelBackend>(
    rows: ClosureExecutionRows<'_>,
    packets: &[KernelPacket],
    workspace: &mut EagerWorkspace,
    backend: &mut B,
    tile_points: usize,
    model_parameters: &[EagerComplex64],
    profile: &mut EagerExecutionProfile,
) -> RusticolResult<()> {
    for packet in packets {
        debug_assert_eq!(packet.role, PacketRole::Closure);
        let items = &rows.closures[packet.item_range.clone()];
        let lane_count = items.len() * tile_points;
        let input_len = packet.input_components * lane_count;
        let output_len = packet.output_components * lane_count;
        let (inputs, outputs) = packet_slices(&mut workspace.packet, input_len, output_len)?;
        let kernel = rows.kernels.get(&packet.kernel_id).ok_or_else(|| {
            RusticolError::internal(format!(
                "eager schedule lost closure kernel {}",
                packet.kernel_id
            ))
        })?;

        let gather_started = Instant::now();
        super::execute::gather_closures(
            items,
            &kernel.inputs,
            inputs,
            packet.input_components,
            tile_points,
            workspace.tile_capacity,
            &workspace.values,
            &workspace.couplings,
            model_parameters,
        );
        profile.closure += gather_started.elapsed();

        let kernel_started = Instant::now();
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            independent_block_size: 1,
            lane_count,
            input_component_count: packet.input_components,
            output_component_count: packet.output_components,
            inputs,
            outputs,
        })?;
        profile.kernel_call += kernel_started.elapsed();
        profile.backend_call_count = profile.backend_call_count.saturating_add(1);

        let scatter_started = Instant::now();
        super::execute::scatter_closures(
            items,
            outputs,
            lane_count,
            tile_points,
            workspace.tile_capacity,
            &workspace.couplings,
            &mut workspace.amplitudes,
        );
        profile.closure += scatter_started.elapsed();
    }

    let direct_started = Instant::now();
    super::execute::execute_direct_closures(
        rows.direct_closures,
        &workspace.values,
        workspace.tile_capacity,
        tile_points,
        &mut workspace.amplitudes,
    );
    profile.closure += direct_started.elapsed();
    Ok(())
}

#[allow(dead_code)] // Shared only by the retained profiled packet-scheduler path.
fn packet_slices(
    packet: &mut [EagerComplex64],
    input_len: usize,
    output_len: usize,
) -> RusticolResult<(&mut [EagerComplex64], &mut [EagerComplex64])> {
    let total = input_len
        .checked_add(output_len)
        .ok_or_else(|| RusticolError::internal("eager active packet size overflows"))?;
    if total > packet.len() {
        return Err(RusticolError::internal(format!(
            "eager active packet needs {total} elements, workspace has {}",
            packet.len()
        )));
    }
    let active = &mut packet[..total];
    Ok(active.split_at_mut(input_len))
}

#[allow(dead_code)] // Shared only by the retained profiled packet-scheduler path.
fn copy_component_range(
    source: &[EagerComplex64],
    source_range: ComponentRange,
    target: &mut [EagerComplex64],
    target_range: ComponentRange,
    stride: usize,
    point_count: usize,
) {
    debug_assert_eq!(source_range.len, target_range.len);
    for component in 0..source_range.len {
        let source_start = (source_range.start + component) * stride;
        let target_start = (target_range.start + component) * stride;
        target[target_start..target_start + point_count]
            .copy_from_slice(&source[source_start..source_start + point_count]);
    }
}
