// SPDX-License-Identifier: 0BSD

use super::plan::{
    ClosureExecutionRows, ComponentRange, EagerExecutionPlan, EagerStagePlan, ScheduledAttachment,
    ScheduledClosure, ScheduledDirectClosure, ScheduledFinalization, ScheduledInvocation,
};
use super::runtime::{
    EagerWorkspace, KernelPacket, LinearFinalizationPacket, PacketRole, StageSchedule,
};
use super::{
    EagerComplex64, EagerKernelBackend, EagerKernelCall, EagerKernelInput, EagerKernelSpec,
};
use crate::{
    EAGER_OUTPUT_FACTOR_COUPLING_IMAG, EAGER_OUTPUT_FACTOR_COUPLING_REAL, EAGER_OUTPUT_FACTOR_NONE,
    RusticolError, RusticolResult,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum AccumulationFactor {
    One,
    NegativeOne,
    ImaginaryUnit,
    NegativeImaginaryUnit,
    Real(f64),
    Imaginary(f64),
    Generic(EagerComplex64),
}

impl AccumulationFactor {
    #[inline]
    pub(super) fn from_parts(real: f64, imag: f64) -> Self {
        match (real, imag) {
            (1.0, 0.0) => Self::One,
            (-1.0, 0.0) => Self::NegativeOne,
            (0.0, 1.0) => Self::ImaginaryUnit,
            (0.0, -1.0) => Self::NegativeImaginaryUnit,
            (_, 0.0) => Self::Real(real),
            (0.0, _) => Self::Imaginary(imag),
            _ => Self::Generic(EagerComplex64::new(real, imag)),
        }
    }

    #[inline(always)]
    pub(super) fn accumulate(self, target: &mut EagerComplex64, value: EagerComplex64) {
        match self {
            Self::One => *target += value,
            Self::NegativeOne => *target -= value,
            Self::ImaginaryUnit => {
                target.re -= value.im;
                target.im += value.re;
            }
            Self::NegativeImaginaryUnit => {
                target.re += value.im;
                target.im -= value.re;
            }
            Self::Real(factor) => {
                target.re += factor * value.re;
                target.im += factor * value.im;
            }
            Self::Imaginary(factor) => {
                target.re -= factor * value.im;
                target.im += factor * value.re;
            }
            Self::Generic(factor) => *target += factor * value,
        }
    }

    #[inline(always)]
    pub(super) fn assign(self, target: &mut EagerComplex64, value: EagerComplex64) {
        *target = match self {
            Self::One => value,
            Self::NegativeOne => -value,
            Self::ImaginaryUnit => EagerComplex64::new(-value.im, value.re),
            Self::NegativeImaginaryUnit => EagerComplex64::new(value.im, -value.re),
            Self::Real(factor) => EagerComplex64::new(factor * value.re, factor * value.im),
            Self::Imaginary(factor) => EagerComplex64::new(-factor * value.im, factor * value.re),
            Self::Generic(factor) => factor * value,
        };
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn scatter_strided(
        self,
        initializes: bool,
        source: &[EagerComplex64],
        source_start: usize,
        source_stride: usize,
        target: &mut [EagerComplex64],
        target_start: usize,
        count: usize,
    ) {
        debug_assert!(count == 0 || source_start + (count - 1) * source_stride < source.len());
        debug_assert!(target_start + count <= target.len());
        // Bounds are established above from validated packet/current ranges.
        // Specializing the factor outside the point loop avoids repeated enum
        // dispatch in the eager scatter hot path.
        unsafe {
            let mut source_ptr = source.as_ptr().add(source_start);
            let mut target_ptr = target.as_mut_ptr().add(target_start);
            macro_rules! scatter {
                ($operation:expr) => {
                    for _ in 0..count {
                        let value = *source_ptr;
                        $operation(&mut *target_ptr, value);
                        source_ptr = source_ptr.add(source_stride);
                        target_ptr = target_ptr.add(1);
                    }
                };
            }
            match (initializes, self) {
                (true, Self::One) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = value
                    })
                }
                (true, Self::NegativeOne) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = -value
                    })
                }
                (true, Self::ImaginaryUnit) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = EagerComplex64::new(-value.im, value.re)
                    })
                }
                (true, Self::NegativeImaginaryUnit) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = EagerComplex64::new(value.im, -value.re)
                    })
                }
                (true, Self::Real(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = EagerComplex64::new(factor * value.re, factor * value.im)
                    })
                }
                (true, Self::Imaginary(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = EagerComplex64::new(-factor * value.im, factor * value.re)
                    })
                }
                (true, Self::Generic(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = factor * value
                    })
                }
                (false, Self::One) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target += value
                    })
                }
                (false, Self::NegativeOne) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target -= value
                    })
                }
                (false, Self::ImaginaryUnit) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        target.re -= value.im;
                        target.im += value.re;
                    })
                }
                (false, Self::NegativeImaginaryUnit) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        target.re += value.im;
                        target.im -= value.re;
                    })
                }
                (false, Self::Real(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        target.re += factor * value.re;
                        target.im += factor * value.im;
                    })
                }
                (false, Self::Imaginary(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        target.re -= factor * value.im;
                        target.im += factor * value.re;
                    })
                }
                (false, Self::Generic(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target += factor * value
                    })
                }
            }
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn scatter_interleaved<const COMPONENTS: usize>(
        self,
        initializes: bool,
        source: &[EagerComplex64],
        source_start: usize,
        target: &mut [EagerComplex64],
        target_component_start: usize,
        target_component_stride: usize,
        count: usize,
    ) {
        self.scatter_interleaved_strided::<COMPONENTS>(
            initializes,
            source,
            source_start,
            COMPONENTS,
            target,
            target_component_start,
            target_component_stride,
            count,
        );
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn scatter_interleaved_strided<const COMPONENTS: usize>(
        self,
        initializes: bool,
        source: &[EagerComplex64],
        source_start: usize,
        source_row_stride: usize,
        target: &mut [EagerComplex64],
        target_component_start: usize,
        target_component_stride: usize,
        count: usize,
    ) {
        debug_assert!(COMPONENTS > 0);
        debug_assert!(source_row_stride >= COMPONENTS);
        debug_assert!(
            count == 0
                || source_start + (count - 1) * source_row_stride + COMPONENTS <= source.len()
        );
        debug_assert!(
            count == 0
                || (target_component_start + COMPONENTS - 1) * target_component_stride + count
                    <= target.len()
        );
        // SymJIT returns lane-major rows while eager current storage is
        // component-major. Handle the common small output widths in one pass
        // so source reads stay contiguous and each target component advances
        // linearly.
        unsafe {
            let mut source_ptr = source.as_ptr().add(source_start);
            let target_base = target.as_mut_ptr();
            let mut target_ptrs = [target_base; COMPONENTS];
            for (component, target_ptr) in target_ptrs.iter_mut().enumerate() {
                *target_ptr =
                    target_base.add((target_component_start + component) * target_component_stride);
            }
            macro_rules! scatter {
                ($operation:expr) => {
                    for _ in 0..count {
                        for component in 0..COMPONENTS {
                            let value = *source_ptr.add(component);
                            $operation(&mut *target_ptrs[component], value);
                            target_ptrs[component] = target_ptrs[component].add(1);
                        }
                        source_ptr = source_ptr.add(source_row_stride);
                    }
                };
            }
            match (initializes, self) {
                (true, Self::One) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| *target = value)
                }
                (true, Self::NegativeOne) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| *target = -value)
                }
                (true, Self::ImaginaryUnit) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = EagerComplex64::new(-value.im, value.re)
                    })
                }
                (true, Self::NegativeImaginaryUnit) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = EagerComplex64::new(value.im, -value.re)
                    })
                }
                (true, Self::Real(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = EagerComplex64::new(factor * value.re, factor * value.im)
                    })
                }
                (true, Self::Imaginary(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target = EagerComplex64::new(-factor * value.im, factor * value.re)
                    })
                }
                (true, Self::Generic(factor)) => scatter!(
                    |target: &mut EagerComplex64, value: EagerComplex64| *target = factor * value
                ),
                (false, Self::One) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| *target += value)
                }
                (false, Self::NegativeOne) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| *target -= value)
                }
                (false, Self::ImaginaryUnit) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        target.re -= value.im;
                        target.im += value.re;
                    })
                }
                (false, Self::NegativeImaginaryUnit) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        target.re += value.im;
                        target.im -= value.re;
                    })
                }
                (false, Self::Real(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        target.re += factor * value.re;
                        target.im += factor * value.im;
                    })
                }
                (false, Self::Imaginary(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        target.re -= factor * value.im;
                        target.im += factor * value.re;
                    })
                }
                (false, Self::Generic(factor)) => {
                    scatter!(|target: &mut EagerComplex64, value: EagerComplex64| {
                        *target += factor * value
                    })
                }
            }
        }
    }
}

#[inline(always)]
fn output_factor_scale(source: u32, coupling: EagerComplex64) -> f64 {
    match source {
        EAGER_OUTPUT_FACTOR_NONE => 1.0,
        EAGER_OUTPUT_FACTOR_COUPLING_REAL => coupling.re,
        EAGER_OUTPUT_FACTOR_COUPLING_IMAG => coupling.im,
        _ => unreachable!("validated eager output factor source"),
    }
}

pub(super) fn initialize_tile(
    plan: &EagerExecutionPlan,
    workspace: &mut EagerWorkspace,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    initial_values: &[EagerComplex64],
    zero_amplitude_indices: &[usize],
) {
    for range in &plan.initial_value_ranges {
        for component in range.start..range.start + range.len {
            let source = component * point_count + tile_start;
            let target = component * workspace.tile_capacity;
            workspace.values[target..target + tile_points]
                .copy_from_slice(&initial_values[source..source + tile_points]);
        }
    }
    for amplitude in zero_amplitude_indices {
        let target = amplitude * workspace.tile_capacity;
        workspace.amplitudes[target..target + tile_points].fill(EagerComplex64::new(0.0, 0.0));
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_stage<B: EagerKernelBackend>(
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
) -> RusticolResult<()> {
    for range in &stage.zero_current_ranges {
        for component in range.start..range.start + range.len {
            let target = component * workspace.tile_capacity;
            workspace.currents[target..target + tile_points].fill(EagerComplex64::new(0.0, 0.0));
        }
    }
    for packet in &schedule.invocation_packets {
        debug_assert_eq!(packet.role, PacketRole::Invocation);
        let items = &stage.invocations[packet.item_range.clone()];
        let (lane_count, evaluator_input_components, evaluator_output_components) =
            invocation_packet_shape(packet, items.len(), tile_points)?;
        let input_len = evaluator_input_components * lane_count;
        let output_len = evaluator_output_components * lane_count;
        let (inputs, outputs) = packet_slices(&mut workspace.packet, input_len, output_len)?;
        let kernel = kernels.get(&packet.kernel_id).ok_or_else(|| {
            RusticolError::internal(format!(
                "eager schedule lost invocation kernel {}",
                packet.kernel_id
            ))
        })?;
        if packet.independent_block_size == 1 {
            gather_invocations(
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
            gather_blocked_invocations(
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
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            independent_block_size: packet.independent_block_size as u32,
            lane_count,
            input_component_count: evaluator_input_components,
            output_component_count: evaluator_output_components,
            inputs,
            outputs,
        })?;
        if packet.independent_block_size == 1 {
            scatter_invocations(
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
            scatter_blocked_invocations(
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
    }

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
        if let Some(linear) = &packet.linear_finalization {
            gather_linear_finalizations(
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
            gather_finalizations(
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
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            independent_block_size: 1,
            lane_count,
            input_component_count: packet.input_components,
            output_component_count: packet.output_components,
            inputs,
            outputs,
        })?;
        if let Some(linear) = &packet.linear_finalization {
            scatter_linear_finalizations(
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
            scatter_finalizations(
                items,
                outputs,
                lane_count,
                tile_points,
                workspace.tile_capacity,
                &mut workspace.values,
            )?;
        }
    }
    Ok(())
}

pub(super) fn invocation_packet_shape(
    packet: &KernelPacket,
    item_count: usize,
    tile_points: usize,
) -> RusticolResult<(usize, usize, usize)> {
    let block_size = packet.independent_block_size;
    if block_size == 0 || !item_count.is_multiple_of(block_size) {
        return Err(RusticolError::internal(
            "eager invocation packet has an invalid independent block shape",
        ));
    }
    let lane_count = item_count
        .checked_div(block_size)
        .and_then(|value| value.checked_mul(tile_points))
        .ok_or_else(|| RusticolError::invalid_argument("eager invocation lanes overflow"))?;
    let input_components = packet
        .input_components
        .checked_mul(block_size)
        .ok_or_else(|| RusticolError::invalid_argument("eager blocked input width overflows"))?;
    let output_components = packet
        .output_components
        .checked_mul(block_size)
        .ok_or_else(|| RusticolError::invalid_argument("eager blocked output width overflows"))?;
    Ok((lane_count, input_components, output_components))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gather_blocked_invocations(
    items: &[ScheduledInvocation],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    input_component_count: usize,
    block_size: usize,
    tile_points: usize,
    tile_capacity: usize,
    values: &[EagerComplex64],
) {
    debug_assert_eq!(block_size, super::EAGER_INDEPENDENT_BLOCK_SIZE as usize);
    debug_assert_eq!(items.len() % block_size, 0);
    debug_assert!(descriptors.iter().all(|descriptor| matches!(
        descriptor,
        EagerKernelInput::FirstCurrentComponent(_) | EagerKernelInput::SecondCurrentComponent(_)
    )));
    match input_component_count {
        6 => gather_current_only_blocked::<6, 4>(
            items,
            descriptors,
            inputs,
            tile_points,
            tile_capacity,
            values,
        ),
        8 => gather_current_only_blocked::<8, 4>(
            items,
            descriptors,
            inputs,
            tile_points,
            tile_capacity,
            values,
        ),
        10 => gather_current_only_blocked::<10, 4>(
            items,
            descriptors,
            inputs,
            tile_points,
            tile_capacity,
            values,
        ),
        _ => gather_current_only_blocked_generic(
            items,
            descriptors,
            inputs,
            block_size,
            tile_points,
            tile_capacity,
            values,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gather_invocations(
    items: &[ScheduledInvocation],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    input_component_count: usize,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    tile_capacity: usize,
    values: &[EagerComplex64],
    momenta: &[f64],
    couplings: &[EagerComplex64],
    model_parameters: &[EagerComplex64],
) {
    if descriptors.iter().all(|descriptor| {
        matches!(
            descriptor,
            EagerKernelInput::FirstCurrentComponent(_)
                | EagerKernelInput::SecondCurrentComponent(_)
        )
    }) {
        match input_component_count {
            6 => {
                gather_current_only_invocations::<6>(
                    items,
                    descriptors,
                    inputs,
                    tile_points,
                    tile_capacity,
                    values,
                );
                return;
            }
            8 => {
                gather_current_only_invocations::<8>(
                    items,
                    descriptors,
                    inputs,
                    tile_points,
                    tile_capacity,
                    values,
                );
                return;
            }
            10 => {
                gather_current_only_invocations::<10>(
                    items,
                    descriptors,
                    inputs,
                    tile_points,
                    tile_capacity,
                    values,
                );
                return;
            }
            _ => {}
        }
    }
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let coupling = couplings[item.row.coupling_slot_id as usize];
        for (target_component, descriptor) in descriptors.iter().copied().enumerate() {
            match descriptor {
                EagerKernelInput::FirstCurrentComponent(component) => gather_complex_component(
                    values,
                    item.left_values,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::SecondCurrentComponent(component) => gather_complex_component(
                    values,
                    item.right_values,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::FirstMomentumComponent(component) => gather_real_component(
                    momenta,
                    item.left_momenta,
                    component,
                    point_count,
                    tile_start,
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::SecondMomentumComponent(component) => gather_real_component(
                    momenta,
                    item.right_momenta,
                    component,
                    point_count,
                    tile_start,
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::CouplingReal => fill_packet_component(
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                    EagerComplex64::new(coupling.re, 0.0),
                ),
                EagerKernelInput::CouplingImag => fill_packet_component(
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                    EagerComplex64::new(coupling.im, 0.0),
                ),
                EagerKernelInput::ModelParameter(parameter) => fill_packet_component(
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                    model_parameters[parameter as usize],
                ),
            }
        }
    }
}

fn gather_current_only_invocations<const COMPONENTS: usize>(
    items: &[ScheduledInvocation],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    tile_points: usize,
    tile_capacity: usize,
    values: &[EagerComplex64],
) {
    debug_assert_eq!(descriptors.len(), COMPONENTS);
    debug_assert_eq!(inputs.len(), items.len() * tile_points * COMPONENTS);
    for (item_index, item) in items.iter().enumerate() {
        let mut source_starts = [0usize; COMPONENTS];
        for (component, descriptor) in descriptors.iter().copied().enumerate() {
            source_starts[component] = match descriptor {
                EagerKernelInput::FirstCurrentComponent(source_component) => {
                    debug_assert!((source_component as usize) < item.left_values.len);
                    (item.left_values.start + source_component as usize) * tile_capacity
                }
                EagerKernelInput::SecondCurrentComponent(source_component) => {
                    debug_assert!((source_component as usize) < item.right_values.len);
                    (item.right_values.start + source_component as usize) * tile_capacity
                }
                _ => unreachable!("current-only eager gather descriptor"),
            };
            debug_assert!(source_starts[component] + tile_points <= values.len());
        }
        let target_start = item_index * tile_points * COMPONENTS;
        debug_assert!(target_start + tile_points * COMPONENTS <= inputs.len());
        // Read each component stream linearly while writing complete SymJIT
        // input rows contiguously. Const arities let LLVM unroll the small
        // inner transpose used by the dominant prepared vertex kernels.
        unsafe {
            let source_base = values.as_ptr();
            let target_base = inputs.as_mut_ptr().add(target_start);
            for point in 0..tile_points {
                let target = target_base.add(point * COMPONENTS);
                for (component, source_start) in source_starts.iter().copied().enumerate() {
                    *target.add(component) = *source_base.add(source_start + point);
                }
            }
        }
    }
}

fn gather_current_only_blocked<const COMPONENTS: usize, const BLOCK_SIZE: usize>(
    items: &[ScheduledInvocation],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    tile_points: usize,
    tile_capacity: usize,
    values: &[EagerComplex64],
) {
    debug_assert_eq!(descriptors.len(), COMPONENTS);
    debug_assert_eq!(items.len() % BLOCK_SIZE, 0);
    debug_assert_eq!(inputs.len(), items.len() * tile_points * COMPONENTS);
    for (block_index, block) in items.chunks_exact(BLOCK_SIZE).enumerate() {
        let mut source_starts = [[0usize; COMPONENTS]; BLOCK_SIZE];
        for (lane, item) in block.iter().enumerate() {
            for (component, descriptor) in descriptors.iter().copied().enumerate() {
                source_starts[lane][component] = match descriptor {
                    EagerKernelInput::FirstCurrentComponent(source_component) => {
                        debug_assert!((source_component as usize) < item.left_values.len);
                        (item.left_values.start + source_component as usize) * tile_capacity
                    }
                    EagerKernelInput::SecondCurrentComponent(source_component) => {
                        debug_assert!((source_component as usize) < item.right_values.len);
                        (item.right_values.start + source_component as usize) * tile_capacity
                    }
                    _ => unreachable!("validated blocked current-only descriptor"),
                };
            }
        }
        let target_start = block_index * tile_points * BLOCK_SIZE * COMPONENTS;
        unsafe {
            let source_base = values.as_ptr();
            let target_base = inputs.as_mut_ptr().add(target_start);
            for point in 0..tile_points {
                let target = target_base.add(point * BLOCK_SIZE * COMPONENTS);
                for (lane, lane_starts) in source_starts.iter().enumerate() {
                    let lane_target = target.add(lane * COMPONENTS);
                    for (component, source_start) in lane_starts.iter().copied().enumerate() {
                        *lane_target.add(component) = *source_base.add(source_start + point);
                    }
                }
            }
        }
    }
}

fn gather_current_only_blocked_generic(
    items: &[ScheduledInvocation],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    block_size: usize,
    tile_points: usize,
    tile_capacity: usize,
    values: &[EagerComplex64],
) {
    let component_count = descriptors.len();
    debug_assert_eq!(items.len() % block_size, 0);
    debug_assert_eq!(inputs.len(), items.len() * tile_points * component_count);
    for (block_index, block) in items.chunks_exact(block_size).enumerate() {
        for point in 0..tile_points {
            let target_row = (block_index * tile_points + point) * block_size * component_count;
            for (lane, item) in block.iter().enumerate() {
                let lane_target = target_row + lane * component_count;
                for (component, descriptor) in descriptors.iter().copied().enumerate() {
                    let (range, source_component) = match descriptor {
                        EagerKernelInput::FirstCurrentComponent(source_component) => {
                            (item.left_values, source_component)
                        }
                        EagerKernelInput::SecondCurrentComponent(source_component) => {
                            (item.right_values, source_component)
                        }
                        _ => unreachable!("validated blocked current-only descriptor"),
                    };
                    inputs[lane_target + component] =
                        values[(range.start + source_component as usize) * tile_capacity + point];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_invocations(
    items: &[ScheduledInvocation],
    attachments: &[ScheduledAttachment],
    outputs: &[EagerComplex64],
    lane_count: usize,
    tile_points: usize,
    tile_capacity: usize,
    couplings: &[EagerComplex64],
    currents: &mut [EagerComplex64],
) {
    let output_components = outputs.len() / lane_count;
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let output_scale = output_factor_scale(
            item.row.output_factor_source,
            couplings[item.row.coupling_slot_id as usize],
        );
        let item_attachments = &attachments[item.attachment_range.clone()];
        let source_start = lane_start * output_components;
        let fused = match (item_attachments.len(), output_components) {
            (2, 2) => scatter_signed_fanout::<2, 2>(
                item_attachments,
                outputs,
                source_start,
                output_scale,
                currents,
                tile_capacity,
                tile_points,
            ),
            (2, 4) => scatter_signed_fanout::<4, 2>(
                item_attachments,
                outputs,
                source_start,
                output_scale,
                currents,
                tile_capacity,
                tile_points,
            ),
            (2, 6) => scatter_signed_fanout::<6, 2>(
                item_attachments,
                outputs,
                source_start,
                output_scale,
                currents,
                tile_capacity,
                tile_points,
            ),
            (4, 2) => scatter_signed_fanout::<2, 4>(
                item_attachments,
                outputs,
                source_start,
                output_scale,
                currents,
                tile_capacity,
                tile_points,
            ),
            (4, 4) => scatter_signed_fanout::<4, 4>(
                item_attachments,
                outputs,
                source_start,
                output_scale,
                currents,
                tile_capacity,
                tile_points,
            ),
            (4, 6) => scatter_signed_fanout::<6, 4>(
                item_attachments,
                outputs,
                source_start,
                output_scale,
                currents,
                tile_capacity,
                tile_points,
            ),
            _ => false,
        };
        if fused {
            continue;
        }
        for attachment in item_attachments {
            let factor = AccumulationFactor::from_parts(
                attachment.row.factor_real * output_scale,
                attachment.row.factor_imag * output_scale,
            );
            debug_assert_eq!(attachment.current.len, output_components);
            match output_components {
                2 => {
                    factor.scatter_interleaved::<2>(
                        attachment.initializes_current,
                        outputs,
                        source_start,
                        currents,
                        attachment.current.start,
                        tile_capacity,
                        tile_points,
                    );
                    continue;
                }
                4 => {
                    factor.scatter_interleaved::<4>(
                        attachment.initializes_current,
                        outputs,
                        source_start,
                        currents,
                        attachment.current.start,
                        tile_capacity,
                        tile_points,
                    );
                    continue;
                }
                6 => {
                    factor.scatter_interleaved::<6>(
                        attachment.initializes_current,
                        outputs,
                        source_start,
                        currents,
                        attachment.current.start,
                        tile_capacity,
                        tile_points,
                    );
                    continue;
                }
                8 => {
                    factor.scatter_interleaved::<8>(
                        attachment.initializes_current,
                        outputs,
                        source_start,
                        currents,
                        attachment.current.start,
                        tile_capacity,
                        tile_points,
                    );
                    continue;
                }
                _ => {}
            }
            for component in 0..output_components {
                let target = (attachment.current.start + component) * tile_capacity;
                factor.scatter_strided(
                    attachment.initializes_current,
                    outputs,
                    source_start + component,
                    output_components,
                    currents,
                    target,
                    tile_points,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_blocked_invocations(
    items: &[ScheduledInvocation],
    attachments: &[ScheduledAttachment],
    outputs: &[EagerComplex64],
    output_components: usize,
    block_size: usize,
    tile_points: usize,
    tile_capacity: usize,
    couplings: &[EagerComplex64],
    currents: &mut [EagerComplex64],
) {
    debug_assert_eq!(block_size, super::EAGER_INDEPENDENT_BLOCK_SIZE as usize);
    debug_assert_eq!(items.len() % block_size, 0);
    let source_row_stride = block_size * output_components;
    debug_assert_eq!(
        outputs.len(),
        (items.len() / block_size) * tile_points * source_row_stride
    );
    for (item_index, item) in items.iter().enumerate() {
        let block_index = item_index / block_size;
        let lane_index = item_index % block_size;
        let source_start =
            block_index * tile_points * source_row_stride + lane_index * output_components;
        let output_scale = output_factor_scale(
            item.row.output_factor_source,
            couplings[item.row.coupling_slot_id as usize],
        );
        for attachment in &attachments[item.attachment_range.clone()] {
            let factor = AccumulationFactor::from_parts(
                attachment.row.factor_real * output_scale,
                attachment.row.factor_imag * output_scale,
            );
            debug_assert_eq!(attachment.current.len, output_components);
            match output_components {
                2 => factor.scatter_interleaved_strided::<2>(
                    attachment.initializes_current,
                    outputs,
                    source_start,
                    source_row_stride,
                    currents,
                    attachment.current.start,
                    tile_capacity,
                    tile_points,
                ),
                4 => factor.scatter_interleaved_strided::<4>(
                    attachment.initializes_current,
                    outputs,
                    source_start,
                    source_row_stride,
                    currents,
                    attachment.current.start,
                    tile_capacity,
                    tile_points,
                ),
                6 => factor.scatter_interleaved_strided::<6>(
                    attachment.initializes_current,
                    outputs,
                    source_start,
                    source_row_stride,
                    currents,
                    attachment.current.start,
                    tile_capacity,
                    tile_points,
                ),
                8 => factor.scatter_interleaved_strided::<8>(
                    attachment.initializes_current,
                    outputs,
                    source_start,
                    source_row_stride,
                    currents,
                    attachment.current.start,
                    tile_capacity,
                    tile_points,
                ),
                _ => {
                    for component in 0..output_components {
                        factor.scatter_strided(
                            attachment.initializes_current,
                            outputs,
                            source_start + component,
                            source_row_stride,
                            currents,
                            (attachment.current.start + component) * tile_capacity,
                            tile_points,
                        );
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_signed_fanout<const COMPONENTS: usize, const FANOUT: usize>(
    attachments: &[ScheduledAttachment],
    outputs: &[EagerComplex64],
    source_start: usize,
    output_scale: f64,
    currents: &mut [EagerComplex64],
    tile_capacity: usize,
    tile_points: usize,
) -> bool {
    let Ok(attachments) = <&[ScheduledAttachment; FANOUT]>::try_from(attachments) else {
        return false;
    };
    let mut negative = [false; FANOUT];
    for (index, attachment) in attachments.iter().enumerate() {
        if attachment.row.factor_imag != 0.0
            || (attachment.row.factor_real != 1.0 && attachment.row.factor_real != -1.0)
        {
            return false;
        }
        debug_assert_eq!(attachment.current.len, COMPONENTS);
        negative[index] = attachment.row.factor_real == -1.0;
    }
    debug_assert!(source_start + tile_points * COMPONENTS <= outputs.len());
    unsafe {
        let target_base = currents.as_mut_ptr();
        let mut target_ptrs = [[target_base; FANOUT]; COMPONENTS];
        for (attachment_index, attachment) in attachments.iter().enumerate() {
            for (component, component_targets) in target_ptrs.iter_mut().enumerate() {
                component_targets[attachment_index] =
                    target_base.add((attachment.current.start + component) * tile_capacity);
            }
        }
        let mut source_ptr = outputs.as_ptr().add(source_start);
        for _ in 0..tile_points {
            for (component, component_targets) in target_ptrs.iter_mut().enumerate() {
                let value = *source_ptr.add(component);
                let scaled = match output_scale {
                    1.0 => value,
                    -1.0 => -value,
                    scale => EagerComplex64::new(scale * value.re, scale * value.im),
                };
                for (attachment_index, target_ptr) in component_targets.iter_mut().enumerate() {
                    let target = &mut **target_ptr;
                    let contribution = if negative[attachment_index] {
                        -scaled
                    } else {
                        scaled
                    };
                    if attachments[attachment_index].initializes_current {
                        *target = contribution;
                    } else {
                        *target += contribution;
                    }
                    *target_ptr = target_ptr.add(1);
                }
            }
            source_ptr = source_ptr.add(COMPONENTS);
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gather_finalizations(
    items: &[ScheduledFinalization],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    input_component_count: usize,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    tile_capacity: usize,
    currents: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
) {
    if descriptors.len() == input_component_count {
        match input_component_count {
            6 if finalization_inputs_are_current_momentum::<2>(descriptors) => {
                gather_current_momentum_finalizations::<2, 6>(
                    items,
                    descriptors,
                    inputs,
                    point_count,
                    tile_start,
                    tile_points,
                    tile_capacity,
                    currents,
                    momenta,
                );
                return;
            }
            8 if finalization_inputs_are_current_momentum::<4>(descriptors) => {
                gather_current_momentum_finalizations::<4, 8>(
                    items,
                    descriptors,
                    inputs,
                    point_count,
                    tile_start,
                    tile_points,
                    tile_capacity,
                    currents,
                    momenta,
                );
                return;
            }
            _ => {}
        }
    }
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        for (target_component, descriptor) in descriptors.iter().copied().enumerate() {
            match descriptor {
                EagerKernelInput::FirstCurrentComponent(component) => gather_complex_component(
                    currents,
                    item.current,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::FirstMomentumComponent(component) => gather_real_component(
                    momenta,
                    item.momentum,
                    component,
                    point_count,
                    tile_start,
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::ModelParameter(parameter) => fill_packet_component(
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                    model_parameters[parameter as usize],
                ),
                _ => unreachable!("validated eager finalization descriptor"),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gather_linear_finalizations(
    linear: &LinearFinalizationPacket,
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    input_component_count: usize,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
) {
    debug_assert_eq!(descriptors.len(), input_component_count);
    debug_assert_eq!(
        inputs.len(),
        linear.lane_multiplier() * tile_points * input_component_count
    );
    let basis_count = linear.current_components.len();
    for (group_index, momentum) in linear.momentum_groups.iter().copied().enumerate() {
        for basis_index in 0..basis_count {
            let lane_start = (group_index * basis_count + basis_index) * tile_points;
            for (target_component, descriptor) in descriptors.iter().copied().enumerate() {
                match descriptor {
                    EagerKernelInput::FirstCurrentComponent(component) => {
                        let value = EagerComplex64::new(
                            if linear.current_components[basis_index] == component {
                                1.0
                            } else {
                                0.0
                            },
                            0.0,
                        );
                        fill_packet_component(
                            inputs,
                            target_component,
                            input_component_count,
                            lane_start,
                            tile_points,
                            value,
                        );
                    }
                    EagerKernelInput::FirstMomentumComponent(component) => gather_real_component(
                        momenta,
                        momentum,
                        component,
                        point_count,
                        tile_start,
                        inputs,
                        target_component,
                        input_component_count,
                        lane_start,
                        tile_points,
                    ),
                    EagerKernelInput::ModelParameter(parameter) => fill_packet_component(
                        inputs,
                        target_component,
                        input_component_count,
                        lane_start,
                        tile_points,
                        model_parameters[parameter as usize],
                    ),
                    _ => unreachable!("validated eager finalization descriptor"),
                }
            }
        }
    }
}

fn finalization_inputs_are_current_momentum<const CURRENT_COMPONENTS: usize>(
    descriptors: &[EagerKernelInput],
) -> bool {
    descriptors[..CURRENT_COMPONENTS]
        .iter()
        .all(|descriptor| matches!(descriptor, EagerKernelInput::FirstCurrentComponent(_)))
        && descriptors[CURRENT_COMPONENTS..]
            .iter()
            .all(|descriptor| matches!(descriptor, EagerKernelInput::FirstMomentumComponent(_)))
}

#[allow(clippy::too_many_arguments)]
fn gather_current_momentum_finalizations<
    const CURRENT_COMPONENTS: usize,
    const COMPONENTS: usize,
>(
    items: &[ScheduledFinalization],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    tile_capacity: usize,
    currents: &[EagerComplex64],
    momenta: &[f64],
) {
    debug_assert_eq!(COMPONENTS, CURRENT_COMPONENTS + 4);
    debug_assert_eq!(descriptors.len(), COMPONENTS);
    debug_assert_eq!(inputs.len(), items.len() * tile_points * COMPONENTS);
    for (item_index, item) in items.iter().enumerate() {
        let mut current_starts = [0usize; CURRENT_COMPONENTS];
        for (component, descriptor) in descriptors[..CURRENT_COMPONENTS]
            .iter()
            .copied()
            .enumerate()
        {
            let EagerKernelInput::FirstCurrentComponent(source_component) = descriptor else {
                unreachable!("validated eager current/momentum finalization descriptor");
            };
            debug_assert!((source_component as usize) < item.current.len);
            current_starts[component] =
                (item.current.start + source_component as usize) * tile_capacity;
        }
        let mut momentum_starts = [0usize; 4];
        for (component, descriptor) in descriptors[CURRENT_COMPONENTS..]
            .iter()
            .copied()
            .enumerate()
        {
            let EagerKernelInput::FirstMomentumComponent(source_component) = descriptor else {
                unreachable!("validated eager current/momentum finalization descriptor");
            };
            debug_assert!((source_component as usize) < item.momentum.len);
            momentum_starts[component] =
                (item.momentum.start + source_component as usize) * point_count + tile_start;
        }
        let target_start = item_index * tile_points * COMPONENTS;
        unsafe {
            let current_base = currents.as_ptr();
            let momentum_base = momenta.as_ptr();
            let target_base = inputs.as_mut_ptr().add(target_start);
            for point in 0..tile_points {
                let target = target_base.add(point * COMPONENTS);
                for (component, source_start) in current_starts.iter().copied().enumerate() {
                    *target.add(component) = *current_base.add(source_start + point);
                }
                for (component, source_start) in momentum_starts.iter().copied().enumerate() {
                    *target.add(CURRENT_COMPONENTS + component) =
                        EagerComplex64::new(*momentum_base.add(source_start + point), 0.0);
                }
            }
        }
    }
}

pub(super) fn scatter_finalizations(
    items: &[ScheduledFinalization],
    outputs: &[EagerComplex64],
    lane_count: usize,
    tile_points: usize,
    tile_capacity: usize,
    values: &mut [EagerComplex64],
) -> RusticolResult<()> {
    let output_components = outputs.len() / lane_count;
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let Some(target) = item.propagated else {
            return Err(RusticolError::internal(
                "eager finalization schedule lost its propagated output",
            ));
        };
        debug_assert_eq!(target.len, output_components);
        let source_start = lane_start * output_components;
        match output_components {
            2 => {
                AccumulationFactor::One.scatter_interleaved::<2>(
                    true,
                    outputs,
                    source_start,
                    values,
                    target.start,
                    tile_capacity,
                    tile_points,
                );
                continue;
            }
            4 => {
                AccumulationFactor::One.scatter_interleaved::<4>(
                    true,
                    outputs,
                    source_start,
                    values,
                    target.start,
                    tile_capacity,
                    tile_points,
                );
                continue;
            }
            6 => {
                AccumulationFactor::One.scatter_interleaved::<6>(
                    true,
                    outputs,
                    source_start,
                    values,
                    target.start,
                    tile_capacity,
                    tile_points,
                );
                continue;
            }
            8 => {
                AccumulationFactor::One.scatter_interleaved::<8>(
                    true,
                    outputs,
                    source_start,
                    values,
                    target.start,
                    tile_capacity,
                    tile_points,
                );
                continue;
            }
            _ => {}
        }
        for component in 0..output_components {
            let target_start = (target.start + component) * tile_capacity;
            AccumulationFactor::One.scatter_strided(
                true,
                outputs,
                source_start + component,
                output_components,
                values,
                target_start,
                tile_points,
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_linear_finalizations(
    items: &[ScheduledFinalization],
    linear: &LinearFinalizationPacket,
    outputs: &[EagerComplex64],
    lane_count: usize,
    tile_points: usize,
    tile_capacity: usize,
    currents: &[EagerComplex64],
    values: &mut [EagerComplex64],
) -> RusticolResult<()> {
    debug_assert_eq!(linear.item_group_indices.len(), items.len());
    let output_components = outputs.len() / lane_count;
    let basis_stride = linear.current_components.len();
    for (item_index, item) in items.iter().enumerate() {
        let Some(target) = item.propagated else {
            return Err(RusticolError::internal(
                "eager linear finalization lost its propagated output",
            ));
        };
        debug_assert_eq!(target.len, output_components);
        let group_start = linear.item_group_indices[item_index] * basis_stride * tile_points;
        for output_component in 0..output_components {
            let target_start = (target.start + output_component) * tile_capacity;
            for (basis_index, source_component) in
                linear.current_components.iter().copied().enumerate()
            {
                let basis_start = (group_start + basis_index * tile_points) * output_components
                    + output_component;
                let source_start = (item.current.start + source_component as usize) * tile_capacity;
                if basis_index == 0 {
                    for point in 0..tile_points {
                        let coefficient = outputs[basis_start + point * output_components];
                        values[target_start + point] = coefficient * currents[source_start + point];
                    }
                } else {
                    for point in 0..tile_points {
                        let coefficient = outputs[basis_start + point * output_components];
                        values[target_start + point] +=
                            coefficient * currents[source_start + point];
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn execute_closures<B: EagerKernelBackend>(
    rows: ClosureExecutionRows<'_>,
    packets: &[KernelPacket],
    workspace: &mut EagerWorkspace,
    backend: &mut B,
    tile_points: usize,
    model_parameters: &[EagerComplex64],
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
        gather_closures(
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
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            independent_block_size: 1,
            lane_count,
            input_component_count: packet.input_components,
            output_component_count: packet.output_components,
            inputs,
            outputs,
        })?;
        scatter_closures(
            items,
            outputs,
            lane_count,
            tile_points,
            workspace.tile_capacity,
            &workspace.couplings,
            &mut workspace.amplitudes,
        );
    }
    execute_direct_closures(
        rows.direct_closures,
        &workspace.values,
        workspace.tile_capacity,
        tile_points,
        &mut workspace.amplitudes,
    );
    Ok(())
}

pub(super) fn copy_tile_amplitudes(
    plan: &EagerExecutionPlan,
    workspace: &EagerWorkspace,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    amplitudes: &mut [EagerComplex64],
) {
    for amplitude in 0..plan.amplitude_count {
        let source = amplitude * workspace.tile_capacity;
        let target = amplitude * point_count + tile_start;
        amplitudes[target..target + tile_points]
            .copy_from_slice(&workspace.amplitudes[source..source + tile_points]);
    }
}

pub(super) fn copy_selected_tile_amplitudes(
    amplitude_indices: &[usize],
    workspace: &EagerWorkspace,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    amplitudes: &mut [EagerComplex64],
) {
    for amplitude in amplitude_indices {
        let source = amplitude * workspace.tile_capacity;
        let target = amplitude * point_count + tile_start;
        amplitudes[target..target + tile_points]
            .copy_from_slice(&workspace.amplitudes[source..source + tile_points]);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gather_closures(
    items: &[ScheduledClosure],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    input_component_count: usize,
    tile_points: usize,
    tile_capacity: usize,
    values: &[EagerComplex64],
    couplings: &[EagerComplex64],
    model_parameters: &[EagerComplex64],
) {
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let coupling = couplings[item.row.coupling_slot_id as usize];
        for (target_component, descriptor) in descriptors.iter().copied().enumerate() {
            match descriptor {
                EagerKernelInput::FirstCurrentComponent(component) => gather_complex_component(
                    values,
                    item.left_values,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::SecondCurrentComponent(component) => gather_complex_component(
                    values,
                    item.right_values,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::CouplingReal => fill_packet_component(
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                    EagerComplex64::new(coupling.re, 0.0),
                ),
                EagerKernelInput::CouplingImag => fill_packet_component(
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                    EagerComplex64::new(coupling.im, 0.0),
                ),
                EagerKernelInput::ModelParameter(parameter) => fill_packet_component(
                    inputs,
                    target_component,
                    input_component_count,
                    lane_start,
                    tile_points,
                    model_parameters[parameter as usize],
                ),
                _ => unreachable!("validated eager closure descriptor"),
            }
        }
    }
}

pub(super) fn scatter_closures(
    items: &[ScheduledClosure],
    outputs: &[EagerComplex64],
    lane_count: usize,
    tile_points: usize,
    tile_capacity: usize,
    couplings: &[EagerComplex64],
    amplitudes: &mut [EagerComplex64],
) {
    let output_components = outputs.len() / lane_count;
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let output_scale = output_factor_scale(
            item.row.output_factor_source,
            couplings[item.row.coupling_slot_id as usize],
        );
        let factor = AccumulationFactor::from_parts(
            item.row.factor_real * output_scale,
            item.row.factor_imag * output_scale,
        );
        let target = item.row.amplitude_index as usize * tile_capacity;
        factor.scatter_strided(
            item.initializes_amplitude,
            outputs,
            lane_start * output_components,
            output_components,
            amplitudes,
            target,
            tile_points,
        );
    }
}

pub(super) fn execute_direct_closures(
    closures: &[ScheduledDirectClosure],
    values: &[EagerComplex64],
    tile_capacity: usize,
    tile_points: usize,
    amplitudes: &mut [EagerComplex64],
) {
    for closure in closures {
        let factor =
            AccumulationFactor::from_parts(closure.row.factor_real, closure.row.factor_imag);
        let target = closure.row.amplitude_index as usize * tile_capacity;
        debug_assert_eq!(closure.left_values.len, closure.coefficients.len());
        debug_assert_eq!(closure.right_values.len, closure.coefficients.len());
        debug_assert!(target + tile_points <= amplitudes.len());
        // Preserve the exact per-point contraction order while replacing two
        // component-index multiplications per term with linear pointer walks.
        // Direct closures are numerous for small complete-color workloads.
        unsafe {
            let left_base = values
                .as_ptr()
                .add(closure.left_values.start * tile_capacity);
            let right_base = values
                .as_ptr()
                .add(closure.right_values.start * tile_capacity);
            let mut target_ptr = amplitudes.as_mut_ptr().add(target);
            for point in 0..tile_points {
                let mut left_ptr = left_base.add(point);
                let mut right_ptr = right_base.add(point);
                let mut contraction = EagerComplex64::new(0.0, 0.0);
                for coefficient in &closure.coefficients {
                    contraction += *coefficient * *left_ptr * *right_ptr;
                    left_ptr = left_ptr.add(tile_capacity);
                    right_ptr = right_ptr.add(tile_capacity);
                }
                if closure.initializes_amplitude {
                    factor.assign(&mut *target_ptr, contraction);
                } else {
                    factor.accumulate(&mut *target_ptr, contraction);
                }
                target_ptr = target_ptr.add(1);
            }
        }
    }
}

pub(super) fn reduce_tile(
    plan: &EagerExecutionPlan,
    workspace: &mut EagerWorkspace,
    tile_points: usize,
) {
    for (group_index, group) in plan.reduction_groups.iter().enumerate() {
        let target = group_index * workspace.tile_capacity;
        let (first_amplitude, remaining_amplitudes) = group
            .amplitude_indices
            .split_first()
            .expect("validated eager reduction group is nonempty");
        let first_source = *first_amplitude as usize * workspace.tile_capacity;
        workspace.reduction_groups[target..target + tile_points]
            .copy_from_slice(&workspace.amplitudes[first_source..first_source + tile_points]);
        for amplitude_index in remaining_amplitudes {
            let source = *amplitude_index as usize * workspace.tile_capacity;
            for point in 0..tile_points {
                workspace.reduction_groups[target + point] += workspace.amplitudes[source + point];
            }
        }
    }
    let (first_entry, remaining_entries) = plan
        .reduction_entries
        .split_first()
        .expect("validated eager reduction table is nonempty");
    let left = first_entry.left_group_index as usize * workspace.tile_capacity;
    let right = first_entry.right_group_index as usize * workspace.tile_capacity;
    for point in 0..tile_points {
        let product = workspace.reduction_groups[left + point]
            * workspace.reduction_groups[right + point].conj();
        workspace.reduced[point] = (first_entry.coefficient * product).re;
    }
    for entry in remaining_entries {
        let left = entry.left_group_index as usize * workspace.tile_capacity;
        let right = entry.right_group_index as usize * workspace.tile_capacity;
        for point in 0..tile_points {
            let product = workspace.reduction_groups[left + point]
                * workspace.reduction_groups[right + point].conj();
            workspace.reduced[point] += (entry.coefficient * product).re;
        }
    }
}

pub(super) fn reduce_selected_tile(
    plan: &EagerExecutionPlan,
    active_group_indices: &[usize],
    active_entry_indices: &[usize],
    selected_group_weights: &[f64],
    workspace: &mut EagerWorkspace,
    tile_points: usize,
) -> RusticolResult<()> {
    if selected_group_weights.len() != active_group_indices.len() * tile_points {
        return Err(RusticolError::invalid_argument(
            "eager selected reduction group weights have an inconsistent shape",
        ));
    }
    workspace.reduced[..tile_points].fill(0.0);
    for group_index in active_group_indices {
        let group = &plan.reduction_groups[*group_index];
        let target = *group_index * workspace.tile_capacity;
        let (first_amplitude, remaining_amplitudes) = group
            .amplitude_indices
            .split_first()
            .expect("validated eager reduction group is nonempty");
        let first_source = *first_amplitude as usize * workspace.tile_capacity;
        workspace.reduction_groups[target..target + tile_points]
            .copy_from_slice(&workspace.amplitudes[first_source..first_source + tile_points]);
        for amplitude_index in remaining_amplitudes {
            let source = *amplitude_index as usize * workspace.tile_capacity;
            for point in 0..tile_points {
                workspace.reduction_groups[target + point] += workspace.amplitudes[source + point];
            }
        }
    }
    for entry_index in active_entry_indices {
        let entry = &plan.reduction_entries[*entry_index];
        let left_group_index = entry.left_group_index as usize;
        let right_group_index = entry.right_group_index as usize;
        let left = left_group_index * workspace.tile_capacity;
        let right = entry.right_group_index as usize * workspace.tile_capacity;
        let left_position = active_group_indices
            .binary_search(&left_group_index)
            .map_err(|_| {
                RusticolError::integrity(
                    "eager selected reduction entry references an inactive left group",
                )
            })?;
        let right_position = active_group_indices
            .binary_search(&right_group_index)
            .map_err(|_| {
                RusticolError::integrity(
                    "eager selected reduction entry references an inactive right group",
                )
            })?;
        for point in 0..tile_points {
            let weight = selected_group_weights[left_position * tile_points + point];
            let right_weight = selected_group_weights[right_position * tile_points + point];
            if weight.to_bits() != right_weight.to_bits() {
                return Err(RusticolError::integrity(
                    "eager selected contraction groups have inconsistent physical weights",
                ));
            }
            let product = workspace.reduction_groups[left + point]
                * workspace.reduction_groups[right + point].conj();
            workspace.reduced[point] += weight * (entry.coefficient * product).re;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn copy_tile_results(
    plan: &EagerExecutionPlan,
    workspace: &EagerWorkspace,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    amplitudes: &mut [EagerComplex64],
    reduced: &mut [f64],
) {
    copy_tile_amplitudes(
        plan,
        workspace,
        point_count,
        tile_start,
        tile_points,
        amplitudes,
    );
    reduced[tile_start..tile_start + tile_points]
        .copy_from_slice(&workspace.reduced[..tile_points]);
}

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

#[allow(clippy::too_many_arguments)]
fn gather_complex_component(
    source: &[EagerComplex64],
    range: ComponentRange,
    component: u32,
    source_stride: usize,
    source_point_start: usize,
    target: &mut [EagerComplex64],
    target_component: usize,
    target_component_count: usize,
    target_lane_start: usize,
    point_count: usize,
) {
    let component = component as usize;
    debug_assert!(component < range.len);
    let source_start = (range.start + component) * source_stride + source_point_start;
    let target_start = target_lane_start * target_component_count + target_component;
    debug_assert!(source_start + point_count <= source.len());
    debug_assert!(
        point_count == 0
            || target_start + (point_count - 1) * target_component_count < target.len()
    );
    // The plan validator and packet sizing above establish both bounds. Raw
    // pointers avoid repeated checks and index multiplication in this hot
    // component-major to row-major transpose.
    unsafe {
        let mut source_ptr = source.as_ptr().add(source_start);
        let mut target_ptr = target.as_mut_ptr().add(target_start);
        for _ in 0..point_count {
            *target_ptr = *source_ptr;
            source_ptr = source_ptr.add(1);
            target_ptr = target_ptr.add(target_component_count);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_real_component(
    source: &[f64],
    range: ComponentRange,
    component: u32,
    source_stride: usize,
    source_point_start: usize,
    target: &mut [EagerComplex64],
    target_component: usize,
    target_component_count: usize,
    target_lane_start: usize,
    point_count: usize,
) {
    let component = component as usize;
    debug_assert!(component < range.len);
    let source_start = (range.start + component) * source_stride + source_point_start;
    let target_start = target_lane_start * target_component_count + target_component;
    debug_assert!(source_start + point_count <= source.len());
    debug_assert!(
        point_count == 0
            || target_start + (point_count - 1) * target_component_count < target.len()
    );
    // Safety follows from the validated source range and active packet size.
    unsafe {
        let mut source_ptr = source.as_ptr().add(source_start);
        let mut target_ptr = target.as_mut_ptr().add(target_start);
        for _ in 0..point_count {
            *target_ptr = EagerComplex64::new(*source_ptr, 0.0);
            source_ptr = source_ptr.add(1);
            target_ptr = target_ptr.add(target_component_count);
        }
    }
}

fn fill_packet_component(
    target: &mut [EagerComplex64],
    target_component: usize,
    target_component_count: usize,
    target_lane_start: usize,
    point_count: usize,
    value: EagerComplex64,
) {
    let target_start = target_lane_start * target_component_count + target_component;
    debug_assert!(
        point_count == 0
            || target_start + (point_count - 1) * target_component_count < target.len()
    );
    // Safety follows from the active packet size established by packet_slices.
    unsafe {
        let mut target_ptr = target.as_mut_ptr().add(target_start);
        for _ in 0..point_count {
            *target_ptr = value;
            target_ptr = target_ptr.add(target_component_count);
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EagerAttachmentRow, EagerFinalizationRow, EagerInvocationRow, MISSING_U32};

    fn c64(real: f64, imag: f64) -> EagerComplex64 {
        EagerComplex64::new(real, imag)
    }

    #[test]
    fn packet_gather_uses_lane_major_component_rows() {
        let complex_source = [
            c64(1.0, 10.0),
            c64(2.0, 20.0),
            c64(3.0, 30.0),
            c64(4.0, 40.0),
            c64(5.0, 50.0),
            c64(6.0, 60.0),
        ];
        let real_source = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let mut packet = vec![c64(0.0, 0.0); 4 * 3];

        gather_complex_component(
            &complex_source,
            ComponentRange { start: 1, len: 1 },
            0,
            3,
            0,
            &mut packet,
            1,
            3,
            1,
            2,
        );
        gather_real_component(
            &real_source,
            ComponentRange { start: 0, len: 2 },
            1,
            3,
            0,
            &mut packet,
            0,
            3,
            1,
            2,
        );
        fill_packet_component(&mut packet, 2, 3, 1, 2, c64(-1.0, 2.0));

        assert_eq!(
            packet,
            vec![
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(14.0, 0.0),
                c64(4.0, 40.0),
                c64(-1.0, 2.0),
                c64(15.0, 0.0),
                c64(5.0, 50.0),
                c64(-1.0, 2.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
            ]
        );
    }

    #[test]
    fn selected_amplitude_copy_leaves_inactive_slots_untouched() {
        let workspace = EagerWorkspace {
            tile_capacity: 3,
            values: Vec::new(),
            currents: Vec::new(),
            amplitudes: vec![
                c64(1.0, 0.0),
                c64(2.0, 0.0),
                c64(3.0, 0.0),
                c64(10.0, 0.0),
                c64(20.0, 0.0),
                c64(30.0, 0.0),
                c64(100.0, 0.0),
                c64(200.0, 0.0),
                c64(300.0, 0.0),
            ],
            reduction_groups: Vec::new(),
            couplings: Vec::new(),
            reduced: Vec::new(),
            packet: Vec::new(),
        };
        let sentinel = c64(-99.0, 1.0);
        let mut output = vec![sentinel; 3 * 5];

        copy_selected_tile_amplitudes(&[1], &workspace, 5, 2, 2, &mut output);

        assert!(output[..5].iter().all(|value| *value == sentinel));
        assert_eq!(
            output[5..10],
            [sentinel, sentinel, c64(10.0, 0.0), c64(20.0, 0.0), sentinel]
        );
        assert!(output[10..].iter().all(|value| *value == sentinel));
    }

    #[test]
    fn proven_linear_finalization_reuses_shared_momentum_dependence() {
        let linear = LinearFinalizationPacket {
            momentum_groups: vec![ComponentRange { start: 0, len: 1 }],
            item_group_indices: vec![0, 0],
            current_components: vec![0],
        };
        let descriptors = [
            EagerKernelInput::FirstCurrentComponent(0),
            EagerKernelInput::FirstMomentumComponent(0),
            EagerKernelInput::ModelParameter(0),
        ];
        let mut inputs = vec![c64(0.0, 0.0); linear.lane_multiplier() * 2 * 3];
        gather_linear_finalizations(
            &linear,
            &descriptors,
            &mut inputs,
            3,
            2,
            0,
            2,
            &[2.0, 3.0],
            &[c64(5.0, 0.0)],
        );
        assert_eq!(
            inputs,
            [
                c64(1.0, 0.0),
                c64(2.0, 0.0),
                c64(5.0, 0.0),
                c64(1.0, 0.0),
                c64(3.0, 0.0),
                c64(5.0, 0.0),
            ]
        );

        let rows = [
            ScheduledFinalization {
                row: EagerFinalizationRow {
                    kernel_id: 1,
                    current_id: 0,
                    unpropagated_value_slot_id: MISSING_U32,
                    propagated_value_slot_id: 0,
                    momentum_slot_id: 0,
                },
                current: ComponentRange { start: 0, len: 1 },
                propagated: Some(ComponentRange { start: 0, len: 1 }),
                momentum: ComponentRange { start: 0, len: 1 },
                selector_domain_id: None,
            },
            ScheduledFinalization {
                row: EagerFinalizationRow {
                    kernel_id: 1,
                    current_id: 1,
                    unpropagated_value_slot_id: MISSING_U32,
                    propagated_value_slot_id: 1,
                    momentum_slot_id: 0,
                },
                current: ComponentRange { start: 1, len: 1 },
                propagated: Some(ComponentRange { start: 1, len: 1 }),
                momentum: ComponentRange { start: 0, len: 1 },
                selector_domain_id: None,
            },
        ];
        // f(current, momentum, parameter) = current * (momentum + parameter).
        let outputs = [c64(7.0, 0.0), c64(8.0, 0.0)];
        let currents = [
            c64(10.0, 0.0),
            c64(20.0, 0.0),
            c64(30.0, 0.0),
            c64(40.0, 0.0),
        ];
        let mut values = vec![c64(0.0, 0.0); 4];
        scatter_linear_finalizations(&rows, &linear, &outputs, 2, 2, 2, &currents, &mut values)
            .unwrap();
        assert_eq!(
            values,
            [
                c64(70.0, 0.0),
                c64(160.0, 0.0),
                c64(210.0, 0.0),
                c64(320.0, 0.0),
            ]
        );
    }

    #[test]
    fn row_major_scatter_specializes_exact_unit_factors() {
        let invocation = ScheduledInvocation {
            row: EagerInvocationRow {
                kernel_id: 0,
                left_value_slot_id: 0,
                right_value_slot_id: 0,
                left_momentum_slot_id: 0,
                right_momentum_slot_id: 0,
                coupling_slot_id: 0,
                output_factor_source: EAGER_OUTPUT_FACTOR_NONE,
                attachment_start: 0,
                attachment_count: 4,
            },
            left_values: ComponentRange { start: 0, len: 1 },
            right_values: ComponentRange { start: 0, len: 1 },
            left_momenta: ComponentRange { start: 0, len: 1 },
            right_momenta: ComponentRange { start: 0, len: 1 },
            attachment_range: 0..4,
            selector_domain_id: None,
        };
        let factors = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
        let attachments = factors
            .into_iter()
            .enumerate()
            .map(|(index, (factor_real, factor_imag))| ScheduledAttachment {
                row: EagerAttachmentRow {
                    result_current_id: index as u32,
                    factor_real,
                    factor_imag,
                },
                current: ComponentRange {
                    start: index * 2,
                    len: 2,
                },
                selector_domain_id: None,
                initializes_current: true,
            })
            .collect::<Vec<_>>();
        let outputs = [c64(1.0, 2.0), c64(3.0, 4.0), c64(5.0, 6.0), c64(7.0, 8.0)];
        let mut currents = vec![c64(0.0, 0.0); 8 * 2];

        scatter_invocations(
            &[invocation],
            &attachments,
            &outputs,
            2,
            2,
            2,
            &[c64(1.0, 0.0)],
            &mut currents,
        );

        for (factor_index, (real, imag)) in factors.into_iter().enumerate() {
            let factor = c64(real, imag);
            for component in 0..2 {
                for point in 0..2 {
                    let expected = factor * outputs[point * 2 + component];
                    let actual = currents[(factor_index * 2 + component) * 2 + point];
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn factor_specialization_keeps_generic_complex_fallback() {
        let initial = c64(2.0, -3.0);
        let value = c64(5.0, 7.0);
        for (real, imag) in [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (2.5, -0.75),
        ] {
            let mut actual = initial;
            AccumulationFactor::from_parts(real, imag).accumulate(&mut actual, value);
            assert_eq!(actual, initial + c64(real, imag) * value);
        }
    }

    #[test]
    fn signed_fanout_scales_once_and_preserves_accumulation_mode() {
        let attachments = [
            ScheduledAttachment {
                row: EagerAttachmentRow {
                    result_current_id: 0,
                    factor_real: 1.0,
                    factor_imag: 0.0,
                },
                current: ComponentRange { start: 0, len: 2 },
                selector_domain_id: None,
                initializes_current: true,
            },
            ScheduledAttachment {
                row: EagerAttachmentRow {
                    result_current_id: 1,
                    factor_real: -1.0,
                    factor_imag: 0.0,
                },
                current: ComponentRange { start: 2, len: 2 },
                selector_domain_id: None,
                initializes_current: false,
            },
        ];
        let outputs = [c64(1.0, 2.0), c64(3.0, 4.0), c64(5.0, 6.0), c64(7.0, 8.0)];
        let mut currents = vec![c64(0.0, 0.0); 8];
        currents[4..].fill(c64(10.0, -10.0));

        assert!(scatter_signed_fanout::<2, 2>(
            &attachments,
            &outputs,
            0,
            2.5,
            &mut currents,
            2,
            2,
        ));

        assert_eq!(
            currents,
            [
                c64(2.5, 5.0),
                c64(12.5, 15.0),
                c64(7.5, 10.0),
                c64(17.5, 20.0),
                c64(7.5, -15.0),
                c64(-2.5, -25.0),
                c64(2.5, -20.0),
                c64(-7.5, -30.0),
            ]
        );
    }

    #[test]
    fn invocation_output_factor_uses_live_coupling_component() {
        let invocation = ScheduledInvocation {
            row: EagerInvocationRow {
                kernel_id: 0,
                left_value_slot_id: 0,
                right_value_slot_id: 0,
                left_momentum_slot_id: 0,
                right_momentum_slot_id: 0,
                coupling_slot_id: 0,
                output_factor_source: EAGER_OUTPUT_FACTOR_COUPLING_REAL,
                attachment_start: 0,
                attachment_count: 1,
            },
            left_values: ComponentRange { start: 0, len: 1 },
            right_values: ComponentRange { start: 0, len: 1 },
            left_momenta: ComponentRange { start: 0, len: 1 },
            right_momenta: ComponentRange { start: 0, len: 1 },
            attachment_range: 0..1,
            selector_domain_id: None,
        };
        let attachments = [ScheduledAttachment {
            row: EagerAttachmentRow {
                result_current_id: 0,
                factor_real: 0.0,
                factor_imag: 1.0,
            },
            current: ComponentRange { start: 0, len: 2 },
            selector_domain_id: None,
            initializes_current: true,
        }];
        let outputs = [c64(1.0, 2.0), c64(3.0, 4.0)];
        let mut currents = vec![c64(0.0, 0.0); 2];

        scatter_invocations(
            &[invocation],
            &attachments,
            &outputs,
            1,
            1,
            1,
            &[c64(2.5, 7.0)],
            &mut currents,
        );

        assert_eq!(currents, [c64(-5.0, 2.5), c64(-10.0, 7.5)]);
    }
}
