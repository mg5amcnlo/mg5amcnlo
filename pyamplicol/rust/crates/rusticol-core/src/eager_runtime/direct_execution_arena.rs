// SPDX-License-Identifier: 0BSD

//! Developer-gated whole-plan eager Direct-Arena execution.
//!
//! The packet scheduler remains an explicit validation oracle while this lane
//! is gated. Production-selected direct execution never falls back to it:
//! authenticated table callables consume persistent planes for every vertex,
//! propagator, and kernel closure, while copies, direct contractions, and
//! reductions operate on the same canonical storage.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use crate::direct_arena::{DIRECT_ARENA_LOCALITY_POINT_CAP, DirectArenaLayout};
use crate::engine::symjit_eager_direct::{
    EAGER_DIRECT_SOURCE_APPLICATION_ABI, EAGER_DIRECT_TABLE_BINDING_ABI,
    EAGER_DIRECT_TABLE_DESCRIPTOR_ABI, EagerDirectArenaPlaneBinding, EagerDirectTableRows,
    EagerDirectTableWorkspace, LoadedSymjitEagerDirectTable,
};
use crate::{
    EAGER_OUTPUT_FACTOR_COUPLING_IMAG, EAGER_OUTPUT_FACTOR_COUPLING_REAL, EAGER_OUTPUT_FACTOR_NONE,
    MISSING_U32, RusticolError, RusticolResult,
};

use super::direct_invocation_arena::{
    EagerDirectPreparedApplication, EagerDirectPreparedKernel, SemanticCatalog, assigned_component,
    count_u32, derive_event_layout, invalid, push_u32, resolve_coupling, row_active,
    validate_active_groups, validate_callable_identity, validate_component_buffer,
};
use super::execute::AccumulationFactor;
use super::plan::{
    ComponentRange, EagerExecutionPlan, EagerStagePlan, ScheduledClosure, ScheduledFinalization,
    ScheduledInvocation,
};
use super::{
    EagerComplex64, EagerKernelInput, EagerKernelRole, EagerPlanV3Sections, EagerRuntimeOptions,
    EagerScheduleAuditRow,
};

struct DirectCall {
    kernel_id: u32,
    application: Arc<LoadedSymjitEagerDirectTable>,
    rows: EagerDirectTableRows,
}

#[derive(Clone, Copy)]
struct DirectFactorSpec {
    factor: EagerComplex64,
    coupling_slot_id: Option<u32>,
    output_factor_source: u32,
}

#[derive(Clone, Copy)]
struct DirectCopy {
    source_component_base: u32,
    destination_component_base: u32,
    component_count: u32,
}

struct DirectStage {
    stage_index: u32,
    invocation_calls: Box<[DirectCall]>,
    unpropagated_copies: Box<[DirectCopy]>,
    finalization_calls: Box<[DirectCall]>,
}

struct DirectCoefficientClosure {
    left_component_base: u32,
    right_component_base: u32,
    component_count: u32,
    amplitude_index: u32,
    factor: AccumulationFactor,
    initializes_amplitude: bool,
    coefficients: Box<[EagerComplex64]>,
}

struct DirectSchedule {
    stages: Box<[DirectStage]>,
    closure_calls: Box<[DirectCall]>,
    direct_closures: Box<[DirectCoefficientClosure]>,
    factor_specs: Box<[DirectFactorSpec]>,
    active_amplitude_indices: Box<[usize]>,
    active_reduction_group_indices: Box<[usize]>,
    active_reduction_entry_indices: Box<[usize]>,
    active_reduction_group_ids: Box<[u32]>,
}

struct SelectedSchedule {
    active_groups: Box<[u32]>,
    schedule: DirectSchedule,
}

#[derive(Clone, Copy)]
struct PointWorkGroup {
    schedule_index: Option<usize>,
    point_start: usize,
    point_count: usize,
}

#[derive(Default)]
struct PointSelectedSchedules {
    signature_offsets: Vec<usize>,
    signature_groups: Vec<u32>,
    point_work_group_ids: Vec<usize>,
    grouped_point_indices: Vec<usize>,
    cursors: Vec<usize>,
    work_groups: Vec<PointWorkGroup>,
    schedules: Vec<SelectedSchedule>,
}

/// Whole-plan eager Direct-Arena runtime.
///
/// Construction owns all lowered callables and immutable row tables. The
/// warmed unprofiled methods mutate only persistent planes, factors, reduction
/// scratch, and caller-provided outputs.
pub(crate) struct EagerDirectExecutionRuntime {
    plan: EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: DirectArenaLayout,
    workspace: EagerDirectTableWorkspace,
    applications: BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>,
    full_schedule: DirectSchedule,
    selected_schedule: Option<SelectedSchedule>,
    point_selected: PointSelectedSchedules,
    initial_value_slots: Box<[u32]>,
    value_ranges: Box<[ComponentRange]>,
    momentum_ranges: Box<[ComponentRange]>,
    reduction_groups: Vec<EagerComplex64>,
    reduced_tile: Vec<f64>,
    broadcast_parameter_bits: Box<[[u64; 2]]>,
    broadcasts_initialized: bool,
    shared_zero_plane_index: Option<u32>,
    elide_structural_zero_fills: bool,
    internal_broadcast_bytes: u64,
    workspace_bytes: usize,
}

impl EagerDirectExecutionRuntime {
    pub(crate) fn from_plan_v3_sections(
        sections: EagerPlanV3Sections<'_>,
        prepared: &[EagerDirectPreparedKernel<'_>],
        options: EagerRuntimeOptions,
    ) -> RusticolResult<Self> {
        if options.point_tile_size == 0 {
            return Err(invalid(
                "eager direct whole-plan point tile size must be positive",
            ));
        }
        let plan = EagerExecutionPlan::from_plan_v3_sections(sections)?;
        let catalog = SemanticCatalog::new(sections)?;
        let layout = derive_event_layout(sections, catalog, &plan)?;
        let requested_tile = u32::try_from(options.point_tile_size)
            .map_err(|_| invalid("eager direct point tile size exceeds u32"))?
            .min(DIRECT_ARENA_LOCALITY_POINT_CAP);

        let applications = load_applications(&plan, prepared)?;
        let shared_zero_plane_index =
            shared_zero_plane_index(sections, catalog, &layout, &applications)?;
        let elide_structural_zero_fills = shared_zero_plane_index.is_some()
            && applications
                .values()
                .all(|application| application.is_symjit());
        let full_schedule = build_direct_schedule(
            &plan,
            catalog,
            &layout,
            &applications,
            shared_zero_plane_index,
            None,
        )?;
        let factor_capacity = full_schedule.factor_specs.len().max(1);
        let (symjit_descriptor_capacity, symjit_scratch_plane_count) = applications
            .values()
            .filter_map(|application| application.symjit_workspace_shape())
            .fold((0usize, 0usize), |maximum, shape| {
                (maximum.0.max(shape.0), maximum.1.max(shape.1))
            });
        let broadcast_cache_bytes = plan
            .parameter_count
            .checked_mul(std::mem::size_of::<[u64; 2]>())
            .ok_or_else(|| invalid("eager direct broadcast cache size overflows"))?;
        let descriptor_bytes =
            EagerDirectTableWorkspace::symjit_descriptor_storage_bytes(symjit_descriptor_capacity)?;
        let factor_bytes = factor_capacity
            .checked_mul(2 * size_of::<f64>())
            .ok_or_else(|| invalid("eager direct factor workspace size overflows"))?;
        let fixed_workspace_bytes = broadcast_cache_bytes
            .checked_add(descriptor_bytes)
            .and_then(|bytes| bytes.checked_add(factor_bytes))
            .ok_or_else(|| invalid("eager direct fixed workspace size overflows"))?;
        let arena_workspace_budget = options
            .workspace_bytes
            .saturating_sub(fixed_workspace_bytes);
        let tile_capacity = direct_tile_capacity(
            requested_tile,
            arena_workspace_budget,
            layout.component_count(),
            count_u32(plan.amplitude_count, "amplitude planes")?,
            count_u32(plan.reduction_groups.len(), "reduction groups")?,
            symjit_scratch_plane_count,
        )?;

        let value_ranges = (0..sections.values.len())
            .map(|id| plan.values.get(id as u32, "eager direct value"))
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        let momentum_ranges = (0..sections.momenta.len())
            .map(|id| plan.momenta.get(id as u32, "eager direct momentum"))
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        let mut produced_values = vec![false; sections.values.len()];
        for row in sections.finalizations {
            for value in [row.unpropagated_value_slot_id, row.propagated_value_slot_id] {
                if value != MISSING_U32 {
                    produced_values[value as usize] = true;
                }
            }
        }
        let initial_value_slots = produced_values
            .iter()
            .enumerate()
            .filter_map(|(id, produced)| (!produced).then_some(id as u32))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let plane_bindings = plane_bindings(&layout, plan.amplitude_count)?;
        let mut workspace = EagerDirectTableWorkspace::new(
            layout.component_count(),
            count_u32(plan.amplitude_count, "amplitude planes")?,
            tile_capacity,
            &plane_bindings,
            shared_zero_plane_index,
            Vec::new(),
            vec![0.0; factor_capacity],
            vec![0.0; factor_capacity],
        )?;
        workspace.begin_tile(tile_capacity)?;
        validate_schedule_calls(
            &full_schedule,
            &mut workspace,
            shared_zero_plane_index,
            tile_capacity,
        )?;

        let stride = workspace.arena().point_stride() as usize;
        let reduction_groups =
            vec![EagerComplex64::new(0.0, 0.0); plan.reduction_groups.len() * stride];
        let reduced_tile = vec![0.0; stride];
        let broadcast_parameter_bits = vec![[0, 0]; plan.parameter_count].into_boxed_slice();
        let workspace_bytes = direct_workspace_bytes(
            &workspace,
            reduction_groups.len(),
            reduced_tile.len(),
            factor_capacity,
        )?
        .checked_add(broadcast_cache_bytes)
        .ok_or_else(|| invalid("eager direct workspace byte accounting overflows"))?;

        Ok(Self {
            plan,
            catalog,
            layout,
            workspace,
            applications,
            full_schedule,
            selected_schedule: None,
            point_selected: PointSelectedSchedules::default(),
            initial_value_slots,
            value_ranges,
            momentum_ranges,
            reduction_groups,
            reduced_tile,
            broadcast_parameter_bits,
            broadcasts_initialized: false,
            shared_zero_plane_index,
            elide_structural_zero_fills,
            internal_broadcast_bytes: 0,
            workspace_bytes,
        })
    }

    pub(crate) fn plan(&self) -> &EagerExecutionPlan {
        &self.plan
    }

    pub(crate) fn effective_point_tile_size(&self) -> usize {
        self.workspace.arena().tile_capacity() as usize
    }

    pub(crate) const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub(crate) const fn internal_scratch_bytes(&self) -> u64 {
        self.workspace.internal_scratch_bytes()
    }

    pub(crate) const fn internal_broadcast_bytes(&self) -> u64 {
        self.internal_broadcast_bytes
    }

    fn refresh_broadcast_planes(
        &mut self,
        model_parameters: &[EagerComplex64],
    ) -> RusticolResult<()> {
        fn parameter_changed(
            cached: &[[u64; 2]],
            parameters: &[EagerComplex64],
            parameter: usize,
        ) -> bool {
            let value = parameters[parameter];
            cached[parameter] != [value.re.to_bits(), value.im.to_bits()]
        }

        let initializing = !self.broadcasts_initialized;
        if !initializing
            && !(0..model_parameters.len()).any(|parameter| {
                parameter_changed(&self.broadcast_parameter_bits, model_parameters, parameter)
            })
        {
            return Ok(());
        }

        let stride = self.workspace.arena().point_stride() as usize;
        let (real, imaginary, _, _) = self.workspace.split_arena_slices_mut();
        let mut written_planes = 0_u64;
        for (coupling_id, row) in self.plan.couplings.iter().copied().enumerate() {
            let coupling_changed = initializing
                || (row.real_parameter_id != MISSING_U32
                    && parameter_changed(
                        &self.broadcast_parameter_bits,
                        model_parameters,
                        row.real_parameter_id as usize,
                    ))
                || (row.imag_parameter_id != MISSING_U32
                    && parameter_changed(
                        &self.broadcast_parameter_bits,
                        model_parameters,
                        row.imag_parameter_id as usize,
                    ));
            if !coupling_changed {
                continue;
            }
            let coupling = resolve_coupling(row, model_parameters);
            let real_component = assigned_component(
                &self.layout,
                self.catalog
                    .coupling_real(count_u32(coupling_id, "coupling")?)?,
                0,
            )?;
            let imaginary_component = assigned_component(
                &self.layout,
                self.catalog
                    .coupling_imag(count_u32(coupling_id, "coupling")?)?,
                0,
            )?;
            if self.elide_structural_zero_fills {
                fill_real_input_plane(real, stride, real_component, stride, coupling.re);
                fill_real_input_plane(real, stride, imaginary_component, stride, coupling.im);
                written_planes = written_planes.saturating_add(2);
            } else {
                fill_input_plane(
                    real,
                    imaginary,
                    stride,
                    real_component,
                    stride,
                    coupling.re,
                    0.0,
                );
                fill_input_plane(
                    real,
                    imaginary,
                    stride,
                    imaginary_component,
                    stride,
                    coupling.im,
                    0.0,
                );
                written_planes = written_planes.saturating_add(4);
            }
        }
        for (parameter, value) in model_parameters.iter().copied().enumerate() {
            if !initializing
                && !parameter_changed(&self.broadcast_parameter_bits, model_parameters, parameter)
            {
                continue;
            }
            fill_input_plane(
                real,
                imaginary,
                stride,
                assigned_component(
                    &self.layout,
                    self.catalog
                        .parameter(count_u32(parameter, "model parameter")?)?,
                    0,
                )?,
                stride,
                value.re,
                value.im,
            );
            written_planes = written_planes.saturating_add(2);
            self.broadcast_parameter_bits[parameter] = [value.re.to_bits(), value.im.to_bits()];
        }
        self.broadcasts_initialized = true;
        self.internal_broadcast_bytes = self.internal_broadcast_bytes.saturating_add(
            written_planes
                .saturating_mul(stride as u64)
                .saturating_mul(std::mem::size_of::<f64>() as u64),
        );
        Ok(())
    }

    /// Execute and reduce directly from arena amplitude planes without
    /// materializing the amplitude matrix in caller memory.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_totals_into(
        &mut self,
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        reduced: &mut [f64],
    ) -> RusticolResult<()> {
        validate_io(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            None,
            Some(reduced),
        )?;
        self.refresh_broadcast_planes(model_parameters)?;
        let tile_capacity = self.effective_point_tile_size();
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = tile_capacity.min(point_count - tile_start);
            execute_contiguous_tile(
                &self.plan,
                self.catalog,
                &self.layout,
                &mut self.workspace,
                &self.initial_value_slots,
                &self.value_ranges,
                &self.momentum_ranges,
                self.elide_structural_zero_fills,
                &self.full_schedule,
                &mut self.reduction_groups,
                point_count,
                tile_start,
                tile_points,
                initial_values,
                momenta,
                model_parameters,
            )?;
            reduce_full_tile(
                &self.plan,
                &self.workspace,
                &mut self.reduction_groups,
                &mut self.reduced_tile,
                tile_points,
            );
            reduced[tile_start..tile_start + tile_points]
                .copy_from_slice(&self.reduced_tile[..tile_points]);
            tile_start += tile_points;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_into(
        &mut self,
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
        reduced: &mut [f64],
    ) -> RusticolResult<()> {
        validate_io(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            Some(amplitudes),
            Some(reduced),
        )?;
        self.refresh_broadcast_planes(model_parameters)?;
        let tile_capacity = self.effective_point_tile_size();
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = tile_capacity.min(point_count - tile_start);
            execute_contiguous_tile(
                &self.plan,
                self.catalog,
                &self.layout,
                &mut self.workspace,
                &self.initial_value_slots,
                &self.value_ranges,
                &self.momentum_ranges,
                self.elide_structural_zero_fills,
                &self.full_schedule,
                &mut self.reduction_groups,
                point_count,
                tile_start,
                tile_points,
                initial_values,
                momenta,
                model_parameters,
            )?;
            reduce_full_tile(
                &self.plan,
                &self.workspace,
                &mut self.reduction_groups,
                &mut self.reduced_tile,
                tile_points,
            );
            copy_amplitude_tile(
                &self.workspace,
                0..self.plan.amplitude_count,
                point_count,
                tile_start,
                tile_points,
                amplitudes,
            );
            reduced[tile_start..tile_start + tile_points]
                .copy_from_slice(&self.reduced_tile[..tile_points]);
            tile_start += tile_points;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_profile_into(
        &mut self,
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
        reduced: &mut [f64],
    ) -> RusticolResult<super::EagerExecutionProfile> {
        validate_io(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            Some(amplitudes),
            Some(reduced),
        )?;
        let total_start = Instant::now();
        let internal_scratch_before = self.workspace.internal_scratch_bytes();
        let internal_broadcast_before = self.internal_broadcast_bytes;
        let mut profile = super::EagerExecutionProfile::default();
        let phase = Instant::now();
        self.refresh_broadcast_planes(model_parameters)?;
        profile.gather += phase.elapsed();
        let tile_capacity = self.effective_point_tile_size();
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = tile_capacity.min(point_count - tile_start);

            let phase = Instant::now();
            self.workspace
                .begin_tile(count_u32(tile_points, "active point tile")?)?;
            self.workspace
                .clear_amplitude_active(0, count_u32(self.plan.amplitude_count, "amplitudes")?)?;
            profile.initialize += phase.elapsed();

            let phase = Instant::now();
            fill_contiguous_tile_inputs(
                &self.plan,
                self.catalog,
                &self.layout,
                &mut self.workspace,
                &self.initial_value_slots,
                &self.value_ranges,
                &self.momentum_ranges,
                self.elide_structural_zero_fills,
                point_count,
                tile_start,
                tile_points,
                initial_values,
                momenta,
                model_parameters,
            )?;
            fill_schedule_factors(
                &self.plan,
                &self.full_schedule,
                &mut self.workspace,
                model_parameters,
            )?;
            profile.gather += phase.elapsed();

            execute_schedule_profiled(
                &self.full_schedule,
                &mut self.workspace,
                &mut self.reduction_groups,
                count_u32(tile_points, "active point tile")?,
                &mut profile,
            )?;

            let phase = Instant::now();
            reduce_full_tile(
                &self.plan,
                &self.workspace,
                &mut self.reduction_groups,
                &mut self.reduced_tile,
                tile_points,
            );
            profile.reduction += phase.elapsed();

            let phase = Instant::now();
            copy_amplitude_tile(
                &self.workspace,
                0..self.plan.amplitude_count,
                point_count,
                tile_start,
                tile_points,
                amplitudes,
            );
            reduced[tile_start..tile_start + tile_points]
                .copy_from_slice(&self.reduced_tile[..tile_points]);
            profile.copy_out += phase.elapsed();
            tile_start += tile_points;
        }
        profile.total = total_start.elapsed();
        profile.internal_scratch_bytes = self
            .workspace
            .internal_scratch_bytes()
            .saturating_sub(internal_scratch_before);
        profile.internal_broadcast_bytes = self
            .internal_broadcast_bytes
            .saturating_sub(internal_broadcast_before);
        Ok(profile)
    }

    pub(crate) fn full_schedule_call_counts(&self) -> (u64, u64, u64) {
        let mut calls = 0_u64;
        let mut invocations = 0_u64;
        let mut destinations = 0_u64;
        for stage in &self.full_schedule.stages {
            for call in stage
                .invocation_calls
                .iter()
                .chain(stage.finalization_calls.iter())
            {
                calls += 1;
                invocations += u64::from(call.rows.invocation_count());
                destinations += u64::from(call.rows.attachment_count());
            }
        }
        for call in &self.full_schedule.closure_calls {
            calls += 1;
            invocations += u64::from(call.rows.invocation_count());
            destinations += u64::from(call.rows.attachment_count());
        }
        (calls, invocations, destinations)
    }

    pub(crate) fn schedule_audit(
        &mut self,
        active_groups: Option<&[u32]>,
    ) -> RusticolResult<Vec<EagerScheduleAuditRow>> {
        if let Some(groups) = active_groups {
            self.prepare_selected_schedule(groups)?;
        }
        let schedule = if active_groups.is_some() {
            &self
                .selected_schedule
                .as_ref()
                .ok_or_else(|| RusticolError::internal("eager direct selected schedule is absent"))?
                .schedule
        } else {
            &self.full_schedule
        };
        let mut rows = Vec::new();
        for stage in &schedule.stages {
            for call in &stage.invocation_calls {
                rows.push(EagerScheduleAuditRow {
                    stage_index: Some(stage.stage_index),
                    role: "invocation",
                    kernel_id: Some(call.kernel_id),
                    call_count: 1,
                    row_count: call.rows.invocation_count() as usize,
                    destination_count: call.rows.attachment_count() as usize,
                });
            }
            if !stage.unpropagated_copies.is_empty() {
                rows.push(EagerScheduleAuditRow {
                    stage_index: Some(stage.stage_index),
                    role: "copy",
                    kernel_id: None,
                    call_count: 0,
                    row_count: stage.unpropagated_copies.len(),
                    destination_count: stage
                        .unpropagated_copies
                        .iter()
                        .map(|copy| copy.component_count as usize)
                        .sum(),
                });
            }
            for call in &stage.finalization_calls {
                rows.push(EagerScheduleAuditRow {
                    stage_index: Some(stage.stage_index),
                    role: "finalization",
                    kernel_id: Some(call.kernel_id),
                    call_count: 1,
                    row_count: call.rows.invocation_count() as usize,
                    destination_count: call.rows.attachment_count() as usize,
                });
            }
        }
        for call in &schedule.closure_calls {
            rows.push(EagerScheduleAuditRow {
                stage_index: None,
                role: "closure",
                kernel_id: Some(call.kernel_id),
                call_count: 1,
                row_count: call.rows.invocation_count() as usize,
                destination_count: call.rows.attachment_count() as usize,
            });
        }
        if !schedule.direct_closures.is_empty() {
            rows.push(EagerScheduleAuditRow {
                stage_index: None,
                role: "direct-closure",
                kernel_id: None,
                call_count: 0,
                row_count: schedule.direct_closures.len(),
                destination_count: schedule.direct_closures.len(),
            });
        }
        Ok(rows)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_selected_profile_into(
        &mut self,
        active_groups: &[u32],
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
    ) -> RusticolResult<super::EagerExecutionProfile> {
        validate_io(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            Some(amplitudes),
            None,
        )?;
        let total_start = Instant::now();
        let internal_scratch_before = self.workspace.internal_scratch_bytes();
        let internal_broadcast_before = self.internal_broadcast_bytes;
        let phase = Instant::now();
        self.prepare_selected_schedule(active_groups)?;
        self.refresh_broadcast_planes(model_parameters)?;
        let mut profile = super::EagerExecutionProfile {
            initialize: phase.elapsed(),
            ..super::EagerExecutionProfile::default()
        };
        let selected = self
            .selected_schedule
            .as_ref()
            .ok_or_else(|| RusticolError::internal("eager direct selected schedule is absent"))?;
        let tile_capacity = self.effective_point_tile_size();
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = tile_capacity.min(point_count - tile_start);

            let phase = Instant::now();
            self.workspace
                .begin_tile(count_u32(tile_points, "active point tile")?)?;
            self.workspace
                .clear_amplitude_active(0, count_u32(self.plan.amplitude_count, "amplitudes")?)?;
            profile.initialize += phase.elapsed();

            let phase = Instant::now();
            fill_contiguous_tile_inputs(
                &self.plan,
                self.catalog,
                &self.layout,
                &mut self.workspace,
                &self.initial_value_slots,
                &self.value_ranges,
                &self.momentum_ranges,
                self.elide_structural_zero_fills,
                point_count,
                tile_start,
                tile_points,
                initial_values,
                momenta,
                model_parameters,
            )?;
            fill_schedule_factors(
                &self.plan,
                &selected.schedule,
                &mut self.workspace,
                model_parameters,
            )?;
            profile.gather += phase.elapsed();

            execute_schedule_profiled(
                &selected.schedule,
                &mut self.workspace,
                &mut self.reduction_groups,
                count_u32(tile_points, "active point tile")?,
                &mut profile,
            )?;

            let phase = Instant::now();
            copy_selected_amplitude_tile(
                &self.workspace,
                &selected.schedule.active_amplitude_indices,
                point_count,
                tile_start,
                tile_points,
                amplitudes,
            );
            profile.copy_out += phase.elapsed();
            tile_start += tile_points;
        }
        profile.total = total_start.elapsed();
        profile.internal_scratch_bytes = self
            .workspace
            .internal_scratch_bytes()
            .saturating_sub(internal_scratch_before);
        profile.internal_broadcast_bytes = self
            .internal_broadcast_bytes
            .saturating_sub(internal_broadcast_before);
        Ok(profile)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_selected_active_amplitudes_into(
        &mut self,
        active_groups: &[u32],
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
    ) -> RusticolResult<()> {
        validate_io(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            Some(amplitudes),
            None,
        )?;
        self.prepare_selected_schedule(active_groups)?;
        self.refresh_broadcast_planes(model_parameters)?;
        let selected = self
            .selected_schedule
            .as_ref()
            .ok_or_else(|| RusticolError::internal("eager direct selected schedule is absent"))?;
        let tile_capacity = self.workspace.arena().tile_capacity() as usize;
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = tile_capacity.min(point_count - tile_start);
            execute_contiguous_tile(
                &self.plan,
                self.catalog,
                &self.layout,
                &mut self.workspace,
                &self.initial_value_slots,
                &self.value_ranges,
                &self.momentum_ranges,
                self.elide_structural_zero_fills,
                &selected.schedule,
                &mut self.reduction_groups,
                point_count,
                tile_start,
                tile_points,
                initial_values,
                momenta,
                model_parameters,
            )?;
            copy_selected_amplitude_tile(
                &self.workspace,
                &selected.schedule.active_amplitude_indices,
                point_count,
                tile_start,
                tile_points,
                amplitudes,
            );
            tile_start += tile_points;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_point_selected_group_sets_into(
        &mut self,
        group_offsets: &[usize],
        active_groups: &[u32],
        active_group_weights: &[f64],
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        reduced: &mut [f64],
    ) -> RusticolResult<()> {
        validate_io(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            None,
            Some(reduced),
        )?;
        validate_point_selectors(
            &self.plan,
            group_offsets,
            active_groups,
            active_group_weights,
            point_count,
        )?;
        self.prepare_point_selected_schedules(group_offsets, active_groups)?;
        self.refresh_broadcast_planes(model_parameters)?;
        reduced.fill(0.0);

        let tile_capacity = self.workspace.arena().tile_capacity() as usize;
        for work in self.point_selected.work_groups.iter().copied() {
            let Some(schedule_index) = work.schedule_index else {
                continue;
            };
            let schedule = &self.point_selected.schedules[schedule_index].schedule;
            let stop = work
                .point_start
                .checked_add(work.point_count)
                .ok_or_else(|| invalid("eager direct point work range overflows"))?;
            let points = self
                .point_selected
                .grouped_point_indices
                .get(work.point_start..stop)
                .ok_or_else(|| invalid("eager direct point work range is invalid"))?;
            for point_tile in points.chunks(tile_capacity) {
                execute_indexed_tile(
                    &self.plan,
                    self.catalog,
                    &self.layout,
                    &mut self.workspace,
                    &self.initial_value_slots,
                    &self.value_ranges,
                    &self.momentum_ranges,
                    self.elide_structural_zero_fills,
                    schedule,
                    &mut self.reduction_groups,
                    point_count,
                    point_tile,
                    initial_values,
                    momenta,
                    model_parameters,
                )?;
                reduce_selected_indexed_tile(
                    &self.plan,
                    schedule,
                    &self.workspace,
                    &mut self.reduction_groups,
                    &mut self.reduced_tile,
                    group_offsets,
                    active_groups,
                    active_group_weights,
                    point_tile,
                )?;
                for (tile_point, original_point) in point_tile.iter().copied().enumerate() {
                    reduced[original_point] = self.reduced_tile[tile_point];
                }
            }
        }
        Ok(())
    }

    fn prepare_selected_schedule(&mut self, active_groups: &[u32]) -> RusticolResult<()> {
        validate_active_groups(&self.plan, Some(active_groups))?;
        if self
            .selected_schedule
            .as_ref()
            .is_some_and(|selected| selected.active_groups.as_ref() == active_groups)
        {
            return Ok(());
        }
        let schedule = build_direct_schedule(
            &self.plan,
            self.catalog,
            &self.layout,
            &self.applications,
            self.shared_zero_plane_index,
            Some(active_groups),
        )?;
        let tile_capacity = self.workspace.arena().tile_capacity();
        self.workspace.begin_tile(tile_capacity)?;
        validate_schedule_calls(
            &schedule,
            &mut self.workspace,
            self.shared_zero_plane_index,
            tile_capacity,
        )?;
        self.selected_schedule = Some(SelectedSchedule {
            active_groups: active_groups.into(),
            schedule,
        });
        Ok(())
    }

    fn prepare_point_selected_schedules(
        &mut self,
        offsets: &[usize],
        groups: &[u32],
    ) -> RusticolResult<()> {
        if self.point_selected.signature_offsets == offsets
            && self.point_selected.signature_groups == groups
        {
            return Ok(());
        }
        self.point_selected.point_work_group_ids.clear();
        self.point_selected.work_groups.clear();
        self.point_selected
            .point_work_group_ids
            .try_reserve(offsets.len().saturating_sub(1))
            .map_err(|error| {
                invalid(format!(
                    "could not reserve eager direct point selector work: {error}"
                ))
            })?;
        for point in 0..offsets.len() - 1 {
            let selected = &groups[offsets[point]..offsets[point + 1]];
            let schedule_index = if selected.is_empty() {
                None
            } else if let Some(index) = self
                .point_selected
                .schedules
                .iter()
                .position(|cached| cached.active_groups.as_ref() == selected)
            {
                Some(index)
            } else {
                let schedule = build_direct_schedule(
                    &self.plan,
                    self.catalog,
                    &self.layout,
                    &self.applications,
                    self.shared_zero_plane_index,
                    Some(selected),
                )?;
                let tile_capacity = self.workspace.arena().tile_capacity();
                self.workspace.begin_tile(tile_capacity)?;
                validate_schedule_calls(
                    &schedule,
                    &mut self.workspace,
                    self.shared_zero_plane_index,
                    tile_capacity,
                )?;
                self.point_selected.schedules.push(SelectedSchedule {
                    active_groups: selected.into(),
                    schedule,
                });
                Some(self.point_selected.schedules.len() - 1)
            };
            let work_id = self
                .point_selected
                .work_groups
                .iter()
                .position(|work| work.schedule_index == schedule_index)
                .unwrap_or_else(|| {
                    self.point_selected.work_groups.push(PointWorkGroup {
                        schedule_index,
                        point_start: 0,
                        point_count: 0,
                    });
                    self.point_selected.work_groups.len() - 1
                });
            self.point_selected.work_groups[work_id].point_count += 1;
            self.point_selected.point_work_group_ids.push(work_id);
        }
        self.point_selected.cursors.clear();
        let mut cursor = 0usize;
        for work in &mut self.point_selected.work_groups {
            work.point_start = cursor;
            cursor = cursor
                .checked_add(work.point_count)
                .ok_or_else(|| invalid("eager direct grouped point count overflows"))?;
            self.point_selected.cursors.push(work.point_start);
        }
        self.point_selected.grouped_point_indices.clear();
        self.point_selected.grouped_point_indices.resize(cursor, 0);
        for (point, work_id) in self
            .point_selected
            .point_work_group_ids
            .iter()
            .copied()
            .enumerate()
        {
            let target = self.point_selected.cursors[work_id];
            self.point_selected.grouped_point_indices[target] = point;
            self.point_selected.cursors[work_id] += 1;
        }
        self.point_selected.signature_offsets.clear();
        self.point_selected
            .signature_offsets
            .extend_from_slice(offsets);
        self.point_selected.signature_groups.clear();
        self.point_selected
            .signature_groups
            .extend_from_slice(groups);
        Ok(())
    }
}

fn load_applications(
    plan: &EagerExecutionPlan,
    prepared: &[EagerDirectPreparedKernel<'_>],
) -> RusticolResult<BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>> {
    let mut applications = BTreeMap::new();
    for artifact in prepared {
        if applications.contains_key(&artifact.kernel_id) {
            return Err(RusticolError::integrity(format!(
                "eager direct prepared catalog repeats kernel {}",
                artifact.kernel_id
            )));
        }
        let kernel = plan.kernels.get(&artifact.kernel_id).ok_or_else(|| {
            RusticolError::integrity(format!(
                "eager direct prepared catalog references unknown kernel {}",
                artifact.kernel_id
            ))
        })?;
        validate_callable_identity(kernel, artifact)?;
        let application = match &artifact.application {
            EagerDirectPreparedApplication::Symjit {
                source_application,
                descriptor,
                expected_optimization_level,
                expected_compression,
            } => LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
                source_application,
                descriptor,
                artifact.display_path.clone(),
                EAGER_DIRECT_SOURCE_APPLICATION_ABI,
                EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
                EAGER_DIRECT_TABLE_BINDING_ABI,
                *expected_optimization_level,
                *expected_compression,
            )?,
            EagerDirectPreparedApplication::Native {
                library,
                function_name,
                source_application_abi,
                invocation_stride,
                attachment_stride,
                target_triple,
                evaluator_state_sha256,
                simd_lane_width,
            } => LoadedSymjitEagerDirectTable::load_native_application(
                Arc::clone(library),
                function_name,
                artifact.display_path.clone(),
                source_application_abi,
                *invocation_stride,
                *attachment_stride,
                count_u32(artifact.inputs.len(), "native eager input width")?,
                artifact.output_component_count,
                target_triple,
                evaluator_state_sha256,
                *simd_lane_width,
            )?,
        };
        applications.insert(artifact.kernel_id, Arc::new(application));
    }
    Ok(applications)
}

fn plane_bindings(
    layout: &DirectArenaLayout,
    amplitude_count: usize,
) -> RusticolResult<Vec<EagerDirectArenaPlaneBinding>> {
    let capacity = layout
        .component_count()
        .checked_add(count_u32(amplitude_count, "amplitude planes")?)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| invalid("eager direct plane catalog overflows"))?;
    let mut bindings = Vec::new();
    bindings
        .try_reserve_exact(capacity as usize)
        .map_err(|error| invalid(format!("could not reserve eager direct planes: {error}")))?;
    for component in 0..layout.component_count() {
        bindings.push(EagerDirectArenaPlaneBinding::CurrentReal(component));
        bindings.push(EagerDirectArenaPlaneBinding::CurrentImag(component));
    }
    for amplitude in 0..count_u32(amplitude_count, "amplitude planes")? {
        bindings.push(EagerDirectArenaPlaneBinding::AmplitudeReal(amplitude));
        bindings.push(EagerDirectArenaPlaneBinding::AmplitudeImag(amplitude));
    }
    Ok(bindings)
}

/// Reuse one already-allocated, event-pinned imaginary plane as the structural
/// zero required by real-only complex P-kernel inputs.
///
/// `derive_event_layout` pins momenta, couplings, and parameters through the
/// final event. Momentum and split coupling semantics are real-only, so their
/// imaginary halves are initialized to zero and are never legal destinations.
/// No extra plane or artifact-ABI entry is required.
fn shared_zero_plane_index(
    sections: EagerPlanV3Sections<'_>,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    applications: &BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>,
) -> RusticolResult<Option<u32>> {
    if !applications
        .values()
        .any(|application| application.is_symjit())
    {
        return Ok(None);
    }
    let physical_component = if let Some(momentum) = sections
        .momenta
        .iter()
        .find(|momentum| momentum.component_count != 0)
    {
        Some(assigned_component(
            layout,
            catalog.momentum(momentum.momentum_slot_id)?,
            0,
        )?)
    } else if let Some(coupling) = sections.couplings.first() {
        Some(assigned_component(
            layout,
            catalog.coupling_real(coupling.coupling_id)?,
            0,
        )?)
    } else {
        None
    };
    physical_component
        .map(|component| {
            component
                .checked_mul(2)
                .and_then(|plane| plane.checked_add(1))
                .ok_or_else(|| invalid("eager direct shared-zero plane id overflows"))
        })
        .transpose()
}

fn build_direct_schedule(
    plan: &EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    applications: &BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>,
    shared_zero_plane_index: Option<u32>,
    active_groups: Option<&[u32]>,
) -> RusticolResult<DirectSchedule> {
    validate_active_groups(plan, active_groups)?;
    let mut factor_specs = Vec::new();
    let mut stages = Vec::with_capacity(plan.stages.len());
    for stage in &plan.stages {
        stages.push(build_direct_stage(
            plan,
            stage,
            catalog,
            layout,
            applications,
            shared_zero_plane_index,
            active_groups,
            &mut factor_specs,
        )?);
    }
    let (closure_calls, direct_closures) = build_direct_closures(
        plan,
        catalog,
        layout,
        applications,
        shared_zero_plane_index,
        active_groups,
        &mut factor_specs,
    )?;
    let (
        active_amplitude_indices,
        active_reduction_group_indices,
        active_reduction_entry_indices,
        active_reduction_group_ids,
    ) = selected_reduction_metadata(plan, active_groups);
    Ok(DirectSchedule {
        stages: stages.into_boxed_slice(),
        closure_calls: closure_calls.into_boxed_slice(),
        direct_closures: direct_closures.into_boxed_slice(),
        factor_specs: factor_specs.into_boxed_slice(),
        active_amplitude_indices: active_amplitude_indices.into_boxed_slice(),
        active_reduction_group_indices: active_reduction_group_indices.into_boxed_slice(),
        active_reduction_entry_indices: active_reduction_entry_indices.into_boxed_slice(),
        active_reduction_group_ids: active_reduction_group_ids.into_boxed_slice(),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_direct_stage(
    plan: &EagerExecutionPlan,
    stage: &EagerStagePlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    applications: &BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>,
    shared_zero_plane_index: Option<u32>,
    active_groups: Option<&[u32]>,
    factor_specs: &mut Vec<DirectFactorSpec>,
) -> RusticolResult<DirectStage> {
    let invocation_calls = build_invocation_calls(
        plan,
        stage,
        catalog,
        layout,
        applications,
        shared_zero_plane_index,
        active_groups,
        factor_specs,
    )?;
    let mut copies = Vec::new();
    for copy in &stage.finalization_copies {
        if !row_active(plan, copy.selector_domain_id, active_groups)? {
            continue;
        }
        let (current_id, _) = stage_current_id(stage, copy.current)?;
        let source = assigned_component(layout, catalog.current(current_id)?, 0)?;
        let value_id = value_id_for_range(plan, copy.unpropagated)?;
        let destination = assigned_component(layout, catalog.value(value_id)?, 0)?;
        copies.push(DirectCopy {
            source_component_base: source,
            destination_component_base: destination,
            component_count: count_u32(copy.current.len, "finalization copy width")?,
        });
    }
    let finalization_calls = build_finalization_calls(
        plan,
        stage,
        catalog,
        layout,
        applications,
        shared_zero_plane_index,
        active_groups,
        factor_specs,
    )?;
    Ok(DirectStage {
        stage_index: stage.stage_index,
        invocation_calls: invocation_calls.into_boxed_slice(),
        unpropagated_copies: copies.into_boxed_slice(),
        finalization_calls: finalization_calls.into_boxed_slice(),
    })
}

fn stage_current_id(
    stage: &EagerStagePlan,
    range: ComponentRange,
) -> RusticolResult<(u32, ComponentRange)> {
    stage
        .finalizations
        .iter()
        .find(|item| item.current == range)
        .map(|item| (item.row.current_id, item.current))
        .or_else(|| {
            stage.attachments.iter().find_map(|item| {
                (item.current == range).then_some((item.row.result_current_id, item.current))
            })
        })
        .ok_or_else(|| RusticolError::internal("eager direct stage lost a compact current range"))
}

fn value_id_for_range(plan: &EagerExecutionPlan, range: ComponentRange) -> RusticolResult<u32> {
    plan.values
        .id_for_range(range, "eager direct finalization value")
}

#[allow(clippy::too_many_arguments)]
fn build_invocation_calls(
    plan: &EagerExecutionPlan,
    stage: &EagerStagePlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    applications: &BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>,
    shared_zero_plane_index: Option<u32>,
    active_groups: Option<&[u32]>,
    factor_specs: &mut Vec<DirectFactorSpec>,
) -> RusticolResult<Vec<DirectCall>> {
    let mut calls = Vec::new();
    let mut written_currents = BTreeSet::new();
    let mut cursor = 0usize;
    while cursor < stage.invocations.len() {
        let kernel_id = stage.invocations[cursor].row.kernel_id;
        let stop = stage.invocations[cursor..]
            .iter()
            .position(|row| row.row.kernel_id != kernel_id)
            .map_or(stage.invocations.len(), |offset| cursor + offset);
        let kernel = plan.kernels.get(&kernel_id).ok_or_else(|| {
            RusticolError::internal(format!("eager direct schedule lost kernel {kernel_id}"))
        })?;
        let kernel_shared_zero_plane_index = shared_zero_plane_index.filter(|_| {
            applications
                .get(&kernel_id)
                .is_some_and(|application| application.is_symjit())
        });
        let mut invocation_bytes = Vec::new();
        let mut attachment_bytes = Vec::new();
        let mut local_attachment_count = 0_u32;
        for invocation in &stage.invocations[cursor..stop] {
            if !row_active(plan, invocation.selector_domain_id, active_groups)? {
                continue;
            }
            let retained = stage.attachments[invocation.attachment_range.clone()]
                .iter()
                .map(|attachment| row_active(plan, attachment.selector_domain_id, active_groups))
                .collect::<RusticolResult<Vec<_>>>()?;
            let active_attachment_count = retained.iter().filter(|active| **active).count();
            if active_attachment_count == 0 {
                continue;
            }
            encode_invocation_inputs(
                &mut invocation_bytes,
                invocation,
                &kernel.inputs,
                catalog,
                layout,
                kernel_shared_zero_plane_index,
            )?;
            push_u32(&mut invocation_bytes, local_attachment_count);
            push_u32(
                &mut invocation_bytes,
                count_u32(active_attachment_count, "active invocation attachments")?,
            );
            for (attachment, active) in stage.attachments[invocation.attachment_range.clone()]
                .iter()
                .zip(retained)
            {
                if !active {
                    continue;
                }
                encode_current_destination(
                    &mut attachment_bytes,
                    attachment.row.result_current_id,
                    kernel.output_component_count,
                    catalog,
                    layout,
                )?;
                let factor_index = count_u32(factor_specs.len(), "eager direct factors")?;
                factor_specs.push(DirectFactorSpec {
                    factor: EagerComplex64::new(
                        attachment.row.factor_real,
                        attachment.row.factor_imag,
                    ),
                    coupling_slot_id: Some(invocation.row.coupling_slot_id),
                    output_factor_source: invocation.row.output_factor_source,
                });
                push_u32(&mut attachment_bytes, factor_index);
                push_u32(
                    &mut attachment_bytes,
                    u32::from(!written_currents.insert(attachment.row.result_current_id)),
                );
                local_attachment_count = local_attachment_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("eager direct attachment count overflows"))?;
            }
        }
        if !invocation_bytes.is_empty() {
            calls.push(load_call(
                applications,
                kernel_id,
                invocation_bytes,
                attachment_bytes,
            )?);
        }
        cursor = stop;
    }
    Ok(calls)
}

#[allow(clippy::too_many_arguments)]
fn build_finalization_calls(
    plan: &EagerExecutionPlan,
    stage: &EagerStagePlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    applications: &BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>,
    shared_zero_plane_index: Option<u32>,
    active_groups: Option<&[u32]>,
    factor_specs: &mut Vec<DirectFactorSpec>,
) -> RusticolResult<Vec<DirectCall>> {
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    while cursor < stage.finalizations.len() {
        let kernel_id = stage.finalizations[cursor].row.kernel_id;
        let stop = stage.finalizations[cursor..]
            .iter()
            .position(|row| row.row.kernel_id != kernel_id)
            .map_or(stage.finalizations.len(), |offset| cursor + offset);
        let kernel = plan.kernels.get(&kernel_id).ok_or_else(|| {
            RusticolError::internal(format!("eager direct finalization lost kernel {kernel_id}"))
        })?;
        if kernel.role != EagerKernelRole::Finalization {
            return Err(RusticolError::integrity(format!(
                "eager direct finalization kernel {kernel_id} has role {:?}",
                kernel.role
            )));
        }
        let kernel_shared_zero_plane_index = shared_zero_plane_index.filter(|_| {
            applications
                .get(&kernel_id)
                .is_some_and(|application| application.is_symjit())
        });
        let mut invocation_bytes = Vec::new();
        let mut attachment_bytes = Vec::new();
        let mut attachment_count = 0_u32;
        for item in &stage.finalizations[cursor..stop] {
            if !row_active(plan, item.selector_domain_id, active_groups)? {
                continue;
            }
            encode_finalization_inputs(
                &mut invocation_bytes,
                item,
                &kernel.inputs,
                catalog,
                layout,
                kernel_shared_zero_plane_index,
            )?;
            push_u32(&mut invocation_bytes, attachment_count);
            push_u32(&mut invocation_bytes, 1);
            let propagated = item.propagated.ok_or_else(|| {
                RusticolError::internal("eager direct finalization lost propagated output")
            })?;
            let value_id = value_id_for_range(plan, propagated)?;
            encode_value_destination(
                &mut attachment_bytes,
                value_id,
                kernel.output_component_count,
                catalog,
                layout,
            )?;
            let factor_index = count_u32(factor_specs.len(), "eager direct factors")?;
            factor_specs.push(DirectFactorSpec {
                factor: EagerComplex64::new(1.0, 0.0),
                coupling_slot_id: None,
                output_factor_source: EAGER_OUTPUT_FACTOR_NONE,
            });
            push_u32(&mut attachment_bytes, factor_index);
            push_u32(&mut attachment_bytes, 0);
            attachment_count = attachment_count
                .checked_add(1)
                .ok_or_else(|| invalid("eager direct finalization count overflows"))?;
        }
        if !invocation_bytes.is_empty() {
            calls.push(load_call(
                applications,
                kernel_id,
                invocation_bytes,
                attachment_bytes,
            )?);
        }
        cursor = stop;
    }
    Ok(calls)
}

#[allow(clippy::too_many_arguments)]
fn build_direct_closures(
    plan: &EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    applications: &BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>,
    shared_zero_plane_index: Option<u32>,
    active_groups: Option<&[u32]>,
    factor_specs: &mut Vec<DirectFactorSpec>,
) -> RusticolResult<(Vec<DirectCall>, Vec<DirectCoefficientClosure>)> {
    let mut calls = Vec::new();
    let mut written_amplitudes = BTreeSet::new();
    let mut cursor = 0usize;
    while cursor < plan.closures.len() {
        let kernel_id = plan.closures[cursor].row.kernel_id;
        let stop = plan.closures[cursor..]
            .iter()
            .position(|row| row.row.kernel_id != kernel_id)
            .map_or(plan.closures.len(), |offset| cursor + offset);
        let kernel = plan.kernels.get(&kernel_id).ok_or_else(|| {
            RusticolError::internal(format!("eager direct closure lost kernel {kernel_id}"))
        })?;
        if kernel.role != EagerKernelRole::Closure {
            return Err(RusticolError::integrity(format!(
                "eager direct closure kernel {kernel_id} has role {:?}",
                kernel.role
            )));
        }
        let kernel_shared_zero_plane_index = shared_zero_plane_index.filter(|_| {
            applications
                .get(&kernel_id)
                .is_some_and(|application| application.is_symjit())
        });
        let mut invocation_bytes = Vec::new();
        let mut attachment_bytes = Vec::new();
        let mut attachment_count = 0_u32;
        for item in &plan.closures[cursor..stop] {
            if !row_active(plan, item.selector_domain_id, active_groups)? {
                continue;
            }
            encode_closure_inputs(
                &mut invocation_bytes,
                item,
                &kernel.inputs,
                catalog,
                layout,
                kernel_shared_zero_plane_index,
            )?;
            push_u32(&mut invocation_bytes, attachment_count);
            push_u32(&mut invocation_bytes, 1);
            encode_amplitude_destination(
                &mut attachment_bytes,
                item.row.amplitude_index,
                kernel.output_component_count,
                layout,
            )?;
            let factor_index = count_u32(factor_specs.len(), "eager direct factors")?;
            factor_specs.push(DirectFactorSpec {
                factor: EagerComplex64::new(item.row.factor_real, item.row.factor_imag),
                coupling_slot_id: Some(item.row.coupling_slot_id),
                output_factor_source: item.row.output_factor_source,
            });
            push_u32(&mut attachment_bytes, factor_index);
            push_u32(
                &mut attachment_bytes,
                u32::from(!written_amplitudes.insert(item.row.amplitude_index)),
            );
            attachment_count = attachment_count
                .checked_add(1)
                .ok_or_else(|| invalid("eager direct closure count overflows"))?;
        }
        if !invocation_bytes.is_empty() {
            calls.push(load_call(
                applications,
                kernel_id,
                invocation_bytes,
                attachment_bytes,
            )?);
        }
        cursor = stop;
    }

    let mut direct = Vec::new();
    for item in &plan.direct_closures {
        if !row_active(plan, item.selector_domain_id, active_groups)? {
            continue;
        }
        let left_id = value_id_for_range(plan, item.left_values)?;
        let right_id = value_id_for_range(plan, item.right_values)?;
        direct.push(DirectCoefficientClosure {
            left_component_base: assigned_component(layout, catalog.value(left_id)?, 0)?,
            right_component_base: assigned_component(layout, catalog.value(right_id)?, 0)?,
            component_count: count_u32(item.coefficients.len(), "direct closure width")?,
            amplitude_index: item.row.amplitude_index,
            factor: AccumulationFactor::from_parts(item.row.factor_real, item.row.factor_imag),
            initializes_amplitude: written_amplitudes.insert(item.row.amplitude_index),
            coefficients: item.coefficients.clone().into_boxed_slice(),
        });
    }
    Ok((calls, direct))
}

fn load_call(
    applications: &BTreeMap<u32, Arc<LoadedSymjitEagerDirectTable>>,
    kernel_id: u32,
    invocation_bytes: Vec<u8>,
    attachment_bytes: Vec<u8>,
) -> RusticolResult<DirectCall> {
    let application = applications.get(&kernel_id).cloned().ok_or_else(|| {
        RusticolError::compatibility(format!(
            "eager direct execution has no prepared table callable for kernel {kernel_id}"
        ))
    })?;
    let rows = application.load_rows(invocation_bytes, attachment_bytes)?;
    Ok(DirectCall {
        kernel_id,
        application,
        rows,
    })
}

fn encode_invocation_inputs(
    output: &mut Vec<u8>,
    invocation: &ScheduledInvocation,
    descriptors: &[EagerKernelInput],
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    shared_zero_plane_index: Option<u32>,
) -> RusticolResult<()> {
    encode_inputs(
        output,
        descriptors,
        |descriptor| match descriptor {
            EagerKernelInput::FirstCurrentComponent(component) => Ok((
                catalog.value(invocation.row.left_value_slot_id)?,
                *component,
                false,
            )),
            EagerKernelInput::SecondCurrentComponent(component) => Ok((
                catalog.value(invocation.row.right_value_slot_id)?,
                *component,
                false,
            )),
            EagerKernelInput::FirstMomentumComponent(component) => Ok((
                catalog.momentum(invocation.row.left_momentum_slot_id)?,
                *component,
                true,
            )),
            EagerKernelInput::SecondMomentumComponent(component) => Ok((
                catalog.momentum(invocation.row.right_momentum_slot_id)?,
                *component,
                true,
            )),
            EagerKernelInput::CouplingReal => Ok((
                catalog.coupling_real(invocation.row.coupling_slot_id)?,
                0,
                true,
            )),
            EagerKernelInput::CouplingImag => Ok((
                catalog.coupling_imag(invocation.row.coupling_slot_id)?,
                0,
                true,
            )),
            EagerKernelInput::ModelParameter(parameter) => {
                Ok((catalog.parameter(*parameter)?, 0, false))
            }
        },
        layout,
        shared_zero_plane_index,
    )
}

fn encode_finalization_inputs(
    output: &mut Vec<u8>,
    item: &ScheduledFinalization,
    descriptors: &[EagerKernelInput],
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    shared_zero_plane_index: Option<u32>,
) -> RusticolResult<()> {
    encode_inputs(
        output,
        descriptors,
        |descriptor| match descriptor {
            EagerKernelInput::FirstCurrentComponent(component) => {
                Ok((catalog.current(item.row.current_id)?, *component, false))
            }
            EagerKernelInput::FirstMomentumComponent(component) => Ok((
                catalog.momentum(item.row.momentum_slot_id)?,
                *component,
                true,
            )),
            EagerKernelInput::ModelParameter(parameter) => {
                Ok((catalog.parameter(*parameter)?, 0, false))
            }
            _ => Err(RusticolError::integrity(
                "eager direct finalization has an invalid semantic input",
            )),
        },
        layout,
        shared_zero_plane_index,
    )
}

fn encode_closure_inputs(
    output: &mut Vec<u8>,
    item: &ScheduledClosure,
    descriptors: &[EagerKernelInput],
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    shared_zero_plane_index: Option<u32>,
) -> RusticolResult<()> {
    encode_inputs(
        output,
        descriptors,
        |descriptor| match descriptor {
            EagerKernelInput::FirstCurrentComponent(component) => Ok((
                catalog.value(item.row.left_value_slot_id)?,
                *component,
                false,
            )),
            EagerKernelInput::SecondCurrentComponent(component) => Ok((
                catalog.value(item.row.right_value_slot_id)?,
                *component,
                false,
            )),
            EagerKernelInput::CouplingReal => {
                Ok((catalog.coupling_real(item.row.coupling_slot_id)?, 0, true))
            }
            EagerKernelInput::CouplingImag => {
                Ok((catalog.coupling_imag(item.row.coupling_slot_id)?, 0, true))
            }
            EagerKernelInput::ModelParameter(parameter) => {
                Ok((catalog.parameter(*parameter)?, 0, false))
            }
            _ => Err(RusticolError::integrity(
                "eager direct closure has an invalid semantic input",
            )),
        },
        layout,
        shared_zero_plane_index,
    )
}

fn encode_inputs(
    output: &mut Vec<u8>,
    descriptors: &[EagerKernelInput],
    mut resolve: impl FnMut(&EagerKernelInput) -> RusticolResult<(u32, u32, bool)>,
    layout: &DirectArenaLayout,
    shared_zero_plane_index: Option<u32>,
) -> RusticolResult<()> {
    for descriptor in descriptors {
        let (semantic, component, real_only) = resolve(descriptor)?;
        let physical = assigned_component(layout, semantic, component)?;
        if let Some(zero_plane) = shared_zero_plane_index.filter(|_| real_only) {
            push_current_real_with_zero(output, physical, zero_plane)?;
        } else {
            push_current_plane_pair(output, physical)?;
        }
    }
    Ok(())
}

fn encode_current_destination(
    output: &mut Vec<u8>,
    current_id: u32,
    width: u32,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
) -> RusticolResult<()> {
    let semantic = catalog.current(current_id)?;
    for component in 0..width {
        push_current_plane_pair(output, assigned_component(layout, semantic, component)?)?;
    }
    Ok(())
}

fn encode_value_destination(
    output: &mut Vec<u8>,
    value_id: u32,
    width: u32,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
) -> RusticolResult<()> {
    let semantic = catalog.value(value_id)?;
    for component in 0..width {
        push_current_plane_pair(output, assigned_component(layout, semantic, component)?)?;
    }
    Ok(())
}

fn encode_amplitude_destination(
    output: &mut Vec<u8>,
    amplitude_index: u32,
    width: u32,
    layout: &DirectArenaLayout,
) -> RusticolResult<()> {
    if width != 1 {
        return Err(RusticolError::integrity(
            "eager direct closure output width must be one",
        ));
    }
    let base = layout
        .component_count()
        .checked_mul(2)
        .and_then(|value| value.checked_add(amplitude_index.checked_mul(2)?))
        .ok_or_else(|| invalid("eager direct amplitude plane id overflows"))?;
    push_u32(output, base);
    push_u32(
        output,
        base.checked_add(1)
            .ok_or_else(|| invalid("eager direct amplitude plane id overflows"))?,
    );
    Ok(())
}

fn push_current_plane_pair(output: &mut Vec<u8>, component: u32) -> RusticolResult<()> {
    let real = component
        .checked_mul(2)
        .ok_or_else(|| invalid("eager direct current plane id overflows"))?;
    push_u32(output, real);
    push_u32(
        output,
        real.checked_add(1)
            .ok_or_else(|| invalid("eager direct current plane id overflows"))?,
    );
    Ok(())
}

fn push_current_real_with_zero(
    output: &mut Vec<u8>,
    component: u32,
    zero_plane: u32,
) -> RusticolResult<()> {
    let real = component
        .checked_mul(2)
        .ok_or_else(|| invalid("eager direct current plane id overflows"))?;
    push_u32(output, real);
    push_u32(output, zero_plane);
    Ok(())
}

#[allow(clippy::type_complexity)]
fn selected_reduction_metadata(
    plan: &EagerExecutionPlan,
    active_groups: Option<&[u32]>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<u32>) {
    let group_active =
        |group_id: u32| active_groups.is_none_or(|groups| groups.binary_search(&group_id).is_ok());
    let active_reduction_group_indices = plan
        .reduction_groups
        .iter()
        .enumerate()
        .filter_map(|(index, group)| group_active(group.coherent_group_id).then_some(index))
        .collect::<Vec<_>>();
    let mut active_amplitude_indices = active_reduction_group_indices
        .iter()
        .flat_map(|index| {
            plan.reduction_groups[*index]
                .amplitude_indices
                .iter()
                .copied()
        })
        .map(|index| index as usize)
        .collect::<Vec<_>>();
    active_amplitude_indices.sort_unstable();
    active_amplitude_indices.dedup();
    let active_reduction_entry_indices = plan
        .reduction_entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let left = entry.left_group_index as usize;
            let right = entry.right_group_index as usize;
            (active_reduction_group_indices.binary_search(&left).is_ok()
                && active_reduction_group_indices.binary_search(&right).is_ok())
            .then_some(index)
        })
        .collect::<Vec<_>>();
    let active_reduction_group_ids = active_reduction_group_indices
        .iter()
        .map(|index| plan.reduction_groups[*index].coherent_group_id)
        .collect();
    (
        active_amplitude_indices,
        active_reduction_group_indices,
        active_reduction_entry_indices,
        active_reduction_group_ids,
    )
}

fn validate_schedule_calls(
    schedule: &DirectSchedule,
    workspace: &mut EagerDirectTableWorkspace,
    shared_zero_plane_index: Option<u32>,
    point_count: u32,
) -> RusticolResult<()> {
    if let Some(zero_component) = shared_zero_plane_index.map(|plane| plane / 2) {
        for stage in &schedule.stages {
            for copy in &stage.unpropagated_copies {
                let destination_stop = copy
                    .destination_component_base
                    .checked_add(copy.component_count)
                    .ok_or_else(|| invalid("eager direct copy destination range overflows"))?;
                if (copy.destination_component_base..destination_stop).contains(&zero_component) {
                    return Err(RusticolError::integrity(format!(
                        "eager direct copy destination range {}..{destination_stop} overlaps \
                         read-only shared-zero component {zero_component}",
                        copy.destination_component_base
                    )));
                }
            }
        }
    }
    for stage in &schedule.stages {
        for call in stage
            .invocation_calls
            .iter()
            .chain(stage.finalization_calls.iter())
        {
            call.application
                .validate_call(&call.rows, workspace, 0, point_count)?;
        }
    }
    for call in &schedule.closure_calls {
        call.application
            .validate_call(&call.rows, workspace, 0, point_count)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_contiguous_tile(
    plan: &EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    workspace: &mut EagerDirectTableWorkspace,
    initial_value_slots: &[u32],
    value_ranges: &[ComponentRange],
    momentum_ranges: &[ComponentRange],
    elide_structural_zero_fills: bool,
    schedule: &DirectSchedule,
    closure_scratch: &mut [EagerComplex64],
    source_point_count: usize,
    tile_start: usize,
    tile_points: usize,
    initial_values: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    execute_tile(
        plan,
        catalog,
        layout,
        workspace,
        initial_value_slots,
        value_ranges,
        momentum_ranges,
        elide_structural_zero_fills,
        schedule,
        closure_scratch,
        source_point_count,
        tile_points,
        initial_values,
        momenta,
        model_parameters,
        |point| tile_start + point,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_indexed_tile(
    plan: &EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    workspace: &mut EagerDirectTableWorkspace,
    initial_value_slots: &[u32],
    value_ranges: &[ComponentRange],
    momentum_ranges: &[ComponentRange],
    elide_structural_zero_fills: bool,
    schedule: &DirectSchedule,
    closure_scratch: &mut [EagerComplex64],
    source_point_count: usize,
    point_indices: &[usize],
    initial_values: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    execute_tile(
        plan,
        catalog,
        layout,
        workspace,
        initial_value_slots,
        value_ranges,
        momentum_ranges,
        elide_structural_zero_fills,
        schedule,
        closure_scratch,
        source_point_count,
        point_indices.len(),
        initial_values,
        momenta,
        model_parameters,
        |point| point_indices[point],
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_tile(
    plan: &EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    workspace: &mut EagerDirectTableWorkspace,
    initial_value_slots: &[u32],
    value_ranges: &[ComponentRange],
    momentum_ranges: &[ComponentRange],
    elide_structural_zero_fills: bool,
    schedule: &DirectSchedule,
    closure_scratch: &mut [EagerComplex64],
    source_point_count: usize,
    tile_points: usize,
    initial_values: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
    source_point: impl Fn(usize) -> usize,
) -> RusticolResult<()> {
    let point_count = count_u32(tile_points, "active point tile")?;
    workspace.begin_tile(point_count)?;
    workspace.clear_amplitude_active(0, count_u32(plan.amplitude_count, "amplitudes")?)?;
    fill_tile_inputs(
        plan,
        catalog,
        layout,
        workspace,
        initial_value_slots,
        value_ranges,
        momentum_ranges,
        elide_structural_zero_fills,
        source_point_count,
        tile_points,
        initial_values,
        momenta,
        model_parameters,
        source_point,
    )?;
    fill_schedule_factors(plan, schedule, workspace, model_parameters)?;
    execute_schedule(schedule, workspace, closure_scratch, point_count)
}

#[allow(clippy::too_many_arguments)]
fn fill_contiguous_tile_inputs(
    plan: &EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    workspace: &mut EagerDirectTableWorkspace,
    initial_value_slots: &[u32],
    value_ranges: &[ComponentRange],
    momentum_ranges: &[ComponentRange],
    elide_structural_zero_fills: bool,
    source_point_count: usize,
    tile_start: usize,
    tile_points: usize,
    initial_values: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    fill_tile_inputs(
        plan,
        catalog,
        layout,
        workspace,
        initial_value_slots,
        value_ranges,
        momentum_ranges,
        elide_structural_zero_fills,
        source_point_count,
        tile_points,
        initial_values,
        momenta,
        model_parameters,
        |point| tile_start + point,
    )
}

#[allow(clippy::too_many_arguments)]
fn fill_tile_inputs(
    _plan: &EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
    workspace: &mut EagerDirectTableWorkspace,
    initial_value_slots: &[u32],
    value_ranges: &[ComponentRange],
    momentum_ranges: &[ComponentRange],
    elide_structural_zero_fills: bool,
    source_point_count: usize,
    tile_points: usize,
    initial_values: &[EagerComplex64],
    momenta: &[f64],
    _model_parameters: &[EagerComplex64],
    source_point: impl Fn(usize) -> usize,
) -> RusticolResult<()> {
    let stride = workspace.arena().point_stride() as usize;
    {
        let (real, imaginary, _, _) = workspace.split_arena_slices_mut();
        for &slot in initial_value_slots {
            let range = value_ranges[slot as usize];
            let semantic = catalog.value(slot)?;
            for component in 0..range.len {
                let target = assigned_component(
                    layout,
                    semantic,
                    count_u32(component, "initial value component")?,
                )? as usize
                    * stride;
                let source = (range.start + component) * source_point_count;
                for point in 0..tile_points {
                    let value = initial_values[source + source_point(point)];
                    real[target + point] = value.re;
                    imaginary[target + point] = value.im;
                }
            }
        }
        for (slot, range) in momentum_ranges.iter().copied().enumerate() {
            let semantic = catalog.momentum(count_u32(slot, "momentum slot")?)?;
            for component in 0..range.len {
                let target = assigned_component(
                    layout,
                    semantic,
                    count_u32(component, "momentum component")?,
                )? as usize
                    * stride;
                let source = (range.start + component) * source_point_count;
                for point in 0..tile_points {
                    real[target + point] = momenta[source + source_point(point)];
                }
                if !elide_structural_zero_fills {
                    imaginary[target..target + tile_points].fill(0.0);
                }
            }
        }
    }
    Ok(())
}

fn fill_input_plane(
    real: &mut [f64],
    imaginary: &mut [f64],
    stride: usize,
    component: u32,
    points: usize,
    value_re: f64,
    value_im: f64,
) {
    let start = component as usize * stride;
    real[start..start + points].fill(value_re);
    imaginary[start..start + points].fill(value_im);
}

fn fill_real_input_plane(
    real: &mut [f64],
    stride: usize,
    component: u32,
    points: usize,
    value: f64,
) {
    let start = component as usize * stride;
    real[start..start + points].fill(value);
}

fn fill_schedule_factors(
    plan: &EagerExecutionPlan,
    schedule: &DirectSchedule,
    workspace: &mut EagerDirectTableWorkspace,
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    let (factor_re, factor_im) = workspace.factors_mut();
    if schedule.factor_specs.len() > factor_re.len() {
        return Err(RusticolError::internal(
            "eager direct selected factors exceed the persistent catalog",
        ));
    }
    for (index, spec) in schedule.factor_specs.iter().copied().enumerate() {
        let scale = if let Some(coupling_slot_id) = spec.coupling_slot_id {
            let coupling = resolve_coupling(
                *plan
                    .couplings
                    .get(coupling_slot_id as usize)
                    .ok_or_else(|| {
                        RusticolError::internal("eager direct factor lost its coupling")
                    })?,
                model_parameters,
            );
            match spec.output_factor_source {
                EAGER_OUTPUT_FACTOR_NONE => 1.0,
                EAGER_OUTPUT_FACTOR_COUPLING_REAL => coupling.re,
                EAGER_OUTPUT_FACTOR_COUPLING_IMAG => coupling.im,
                _ => {
                    return Err(RusticolError::integrity(
                        "eager direct factor has an invalid output-factor source",
                    ));
                }
            }
        } else {
            if spec.output_factor_source != EAGER_OUTPUT_FACTOR_NONE {
                return Err(RusticolError::integrity(
                    "constant eager direct factor has a dynamic source",
                ));
            }
            1.0
        };
        factor_re[index] = spec.factor.re * scale;
        factor_im[index] = spec.factor.im * scale;
    }
    Ok(())
}

fn execute_schedule(
    schedule: &DirectSchedule,
    workspace: &mut EagerDirectTableWorkspace,
    closure_scratch: &mut [EagerComplex64],
    point_count: u32,
) -> RusticolResult<()> {
    for stage in &schedule.stages {
        execute_calls(&stage.invocation_calls, workspace, point_count)?;
        execute_copies(&stage.unpropagated_copies, workspace, point_count);
        execute_calls(&stage.finalization_calls, workspace, point_count)?;
    }
    execute_calls(&schedule.closure_calls, workspace, point_count)?;
    execute_direct_coefficient_closures(
        &schedule.direct_closures,
        workspace,
        closure_scratch,
        point_count as usize,
    );
    Ok(())
}

fn execute_schedule_profiled(
    schedule: &DirectSchedule,
    workspace: &mut EagerDirectTableWorkspace,
    closure_scratch: &mut [EagerComplex64],
    point_count: u32,
    profile: &mut super::EagerExecutionProfile,
) -> RusticolResult<()> {
    for stage in &schedule.stages {
        let phase = Instant::now();
        execute_calls(&stage.invocation_calls, workspace, point_count)?;
        profile.kernel_call += phase.elapsed();
        profile.backend_call_count = profile
            .backend_call_count
            .saturating_add(stage.invocation_calls.len() as u64);

        let phase = Instant::now();
        execute_copies(&stage.unpropagated_copies, workspace, point_count);
        profile.invocation_scatter += phase.elapsed();

        let phase = Instant::now();
        execute_calls(&stage.finalization_calls, workspace, point_count)?;
        profile.finalization += phase.elapsed();
        profile.backend_call_count = profile
            .backend_call_count
            .saturating_add(stage.finalization_calls.len() as u64);
    }
    let phase = Instant::now();
    execute_calls(&schedule.closure_calls, workspace, point_count)?;
    execute_direct_coefficient_closures(
        &schedule.direct_closures,
        workspace,
        closure_scratch,
        point_count as usize,
    );
    profile.closure += phase.elapsed();
    profile.backend_call_count = profile
        .backend_call_count
        .saturating_add(schedule.closure_calls.len() as u64);
    Ok(())
}

fn execute_calls(
    calls: &[DirectCall],
    workspace: &mut EagerDirectTableWorkspace,
    point_count: u32,
) -> RusticolResult<()> {
    for call in calls {
        // SAFETY: construction authenticated immutable rows, callable
        // descriptors, plane catalogs, aliases, and the maximum point range.
        unsafe {
            call.application
                .evaluate_validated_unchecked(&call.rows, workspace, 0, point_count)?;
        }
    }
    Ok(())
}

fn execute_copies(
    copies: &[DirectCopy],
    workspace: &mut EagerDirectTableWorkspace,
    point_count: u32,
) {
    let stride = workspace.arena().point_stride() as usize;
    let points = point_count as usize;
    let (real, imaginary, _, _) = workspace.split_arena_slices_mut();
    for copy in copies {
        for component in 0..copy.component_count as usize {
            let source = (copy.source_component_base as usize + component) * stride;
            let target = (copy.destination_component_base as usize + component) * stride;
            for point in 0..points {
                real[target + point] = real[source + point];
                imaginary[target + point] = imaginary[source + point];
            }
        }
    }
}

fn execute_direct_coefficient_closures(
    closures: &[DirectCoefficientClosure],
    workspace: &mut EagerDirectTableWorkspace,
    contraction_scratch: &mut [EagerComplex64],
    point_count: usize,
) {
    let stride = workspace.arena().point_stride() as usize;
    debug_assert!(contraction_scratch.len() >= stride);
    let (current_re, current_im, amplitude_re, amplitude_im) = workspace.split_arena_slices_mut();
    for closure in closures {
        let target = closure.amplitude_index as usize * stride;
        contraction_scratch[..point_count].fill(EagerComplex64::new(0.0, 0.0));
        // Component-outer, point-inner traversal keeps both source planes
        // sequential. Each point still observes components in the original
        // order, preserving its contraction summation semantics exactly.
        for component in 0..closure.component_count as usize {
            for (point, contraction) in contraction_scratch[..point_count].iter_mut().enumerate() {
                let left = (closure.left_component_base as usize + component) * stride + point;
                let right = (closure.right_component_base as usize + component) * stride + point;
                *contraction += closure.coefficients[component]
                    * EagerComplex64::new(current_re[left], current_im[left])
                    * EagerComplex64::new(current_re[right], current_im[right]);
            }
        }
        for (point, contraction) in contraction_scratch[..point_count]
            .iter()
            .copied()
            .enumerate()
        {
            let mut value =
                EagerComplex64::new(amplitude_re[target + point], amplitude_im[target + point]);
            if closure.initializes_amplitude {
                closure.factor.assign(&mut value, contraction);
            } else {
                closure.factor.accumulate(&mut value, contraction);
            }
            amplitude_re[target + point] = value.re;
            amplitude_im[target + point] = value.im;
        }
    }
}

fn reduce_full_tile(
    plan: &EagerExecutionPlan,
    workspace: &EagerDirectTableWorkspace,
    groups: &mut [EagerComplex64],
    reduced: &mut [f64],
    point_count: usize,
) {
    let stride = workspace.arena().point_stride() as usize;
    let (amplitude_re, amplitude_im) = workspace.arena().amplitude_slices();
    for (group_index, group) in plan.reduction_groups.iter().enumerate() {
        let target = group_index * stride;
        let (first, remaining) = group
            .amplitude_indices
            .split_first()
            .expect("validated eager reduction group");
        let first_source = *first as usize * stride;
        for point in 0..point_count {
            groups[target + point] = EagerComplex64::new(
                amplitude_re[first_source + point],
                amplitude_im[first_source + point],
            );
        }
        for amplitude in remaining {
            let source = *amplitude as usize * stride;
            for point in 0..point_count {
                groups[target + point] +=
                    EagerComplex64::new(amplitude_re[source + point], amplitude_im[source + point]);
            }
        }
    }
    let (first, remaining) = plan
        .reduction_entries
        .split_first()
        .expect("validated eager reduction entries");
    let left = first.left_group_index as usize * stride;
    let right = first.right_group_index as usize * stride;
    for point in 0..point_count {
        reduced[point] =
            (first.coefficient * groups[left + point] * groups[right + point].conj()).re;
    }
    for entry in remaining {
        let left = entry.left_group_index as usize * stride;
        let right = entry.right_group_index as usize * stride;
        for point in 0..point_count {
            reduced[point] +=
                (entry.coefficient * groups[left + point] * groups[right + point].conj()).re;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reduce_selected_indexed_tile(
    plan: &EagerExecutionPlan,
    schedule: &DirectSchedule,
    workspace: &EagerDirectTableWorkspace,
    groups: &mut [EagerComplex64],
    reduced: &mut [f64],
    group_offsets: &[usize],
    selected_groups: &[u32],
    selected_group_weights: &[f64],
    point_indices: &[usize],
) -> RusticolResult<()> {
    let stride = workspace.arena().point_stride() as usize;
    let (amplitude_re, amplitude_im) = workspace.arena().amplitude_slices();
    reduced[..point_indices.len()].fill(0.0);
    for group_index in schedule.active_reduction_group_indices.iter().copied() {
        let group = &plan.reduction_groups[group_index];
        let target = group_index * stride;
        let (first, remaining) = group
            .amplitude_indices
            .split_first()
            .expect("validated eager reduction group");
        let source = *first as usize * stride;
        for point in 0..point_indices.len() {
            groups[target + point] =
                EagerComplex64::new(amplitude_re[source + point], amplitude_im[source + point]);
        }
        for amplitude in remaining {
            let source = *amplitude as usize * stride;
            for point in 0..point_indices.len() {
                groups[target + point] +=
                    EagerComplex64::new(amplitude_re[source + point], amplitude_im[source + point]);
            }
        }
    }
    for entry_index in schedule.active_reduction_entry_indices.iter().copied() {
        let entry = &plan.reduction_entries[entry_index];
        let left_group = entry.left_group_index as usize;
        let right_group = entry.right_group_index as usize;
        let left = left_group * stride;
        let right = right_group * stride;
        let left_position = schedule
            .active_reduction_group_indices
            .binary_search(&left_group)
            .map_err(|_| RusticolError::internal("eager direct reduction lost left group"))?;
        let right_position = schedule
            .active_reduction_group_indices
            .binary_search(&right_group)
            .map_err(|_| RusticolError::internal("eager direct reduction lost right group"))?;
        let left_id = schedule.active_reduction_group_ids[left_position];
        let right_id = schedule.active_reduction_group_ids[right_position];
        for (tile_point, original_point) in point_indices.iter().copied().enumerate() {
            let selected =
                &selected_groups[group_offsets[original_point]..group_offsets[original_point + 1]];
            let left_weight = selected
                .binary_search(&left_id)
                .ok()
                .map(|position| selected_group_weights[group_offsets[original_point] + position])
                .ok_or_else(|| RusticolError::integrity("eager direct left weight is absent"))?;
            let right_weight = selected
                .binary_search(&right_id)
                .ok()
                .map(|position| selected_group_weights[group_offsets[original_point] + position])
                .ok_or_else(|| RusticolError::integrity("eager direct right weight is absent"))?;
            if left_weight.to_bits() != right_weight.to_bits() {
                return Err(RusticolError::integrity(
                    "eager direct selected contraction weights disagree",
                ));
            }
            reduced[tile_point] += left_weight
                * (entry.coefficient
                    * groups[left + tile_point]
                    * groups[right + tile_point].conj())
                .re;
        }
    }
    Ok(())
}

fn copy_amplitude_tile(
    workspace: &EagerDirectTableWorkspace,
    amplitudes: std::ops::Range<usize>,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    output: &mut [EagerComplex64],
) {
    let stride = workspace.arena().point_stride() as usize;
    let (real, imaginary) = workspace.arena().amplitude_slices();
    for amplitude in amplitudes {
        let source = amplitude * stride;
        let target = amplitude * point_count + tile_start;
        for point in 0..tile_points {
            output[target + point] =
                EagerComplex64::new(real[source + point], imaginary[source + point]);
        }
    }
}

fn copy_selected_amplitude_tile(
    workspace: &EagerDirectTableWorkspace,
    amplitude_indices: &[usize],
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    output: &mut [EagerComplex64],
) {
    let stride = workspace.arena().point_stride() as usize;
    let (real, imaginary) = workspace.arena().amplitude_slices();
    for amplitude in amplitude_indices {
        let source = *amplitude * stride;
        let target = *amplitude * point_count + tile_start;
        for point in 0..tile_points {
            output[target + point] =
                EagerComplex64::new(real[source + point], imaginary[source + point]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_io(
    plan: &EagerExecutionPlan,
    point_count: usize,
    initial_values: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
    amplitudes: Option<&[EagerComplex64]>,
    reduced: Option<&[f64]>,
) -> RusticolResult<()> {
    if point_count == 0 {
        return Err(invalid(
            "eager direct evaluation requires at least one point",
        ));
    }
    let points = count_u32(point_count, "point")?;
    validate_component_buffer(
        "initial values",
        initial_values.len(),
        plan.values.component_count,
        points,
    )?;
    validate_component_buffer(
        "momenta",
        momenta.len(),
        plan.momenta.component_count,
        points,
    )?;
    if model_parameters.len() != plan.parameter_count
        || model_parameters
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(invalid(
            "eager direct model parameters have an invalid shape or value",
        ));
    }
    if let Some(amplitudes) = amplitudes {
        let expected = plan
            .amplitude_count
            .checked_mul(point_count)
            .ok_or_else(|| invalid("eager direct amplitude length overflows"))?;
        if amplitudes.len() != expected {
            return Err(invalid(format!(
                "eager direct amplitudes have length {}, expected {expected}",
                amplitudes.len()
            )));
        }
    }
    if reduced.is_some_and(|values| values.len() != point_count) {
        return Err(invalid("eager direct reduced output has the wrong length"));
    }
    Ok(())
}

fn validate_point_selectors(
    plan: &EagerExecutionPlan,
    offsets: &[usize],
    groups: &[u32],
    weights: &[f64],
    point_count: usize,
) -> RusticolResult<()> {
    if offsets.len() != point_count + 1
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&groups.len())
        || weights.len() != groups.len()
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(invalid("eager direct per-point selector shape is invalid"));
    }
    let known = plan.selector_domains.as_ref().ok_or_else(|| {
        RusticolError::compatibility(
            "eager direct per-point selectors require selector-domain metadata",
        )
    })?;
    for point in 0..point_count {
        let selected = &groups[offsets[point]..offsets[point + 1]];
        if selected.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(format!(
                "eager direct selector groups for point {point} are not sorted and unique"
            )));
        }
        if let Some(unknown) = selected
            .iter()
            .find(|group| known.group_ids.binary_search(group).is_err())
        {
            return Err(invalid(format!(
                "eager direct point {point} references unknown group {unknown}"
            )));
        }
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(invalid("eager direct selector weights must be finite"));
    }
    Ok(())
}

fn direct_tile_capacity(
    requested: u32,
    workspace_bytes: usize,
    current_planes: u32,
    amplitude_planes: u32,
    reduction_groups: u32,
    symjit_scratch_planes: usize,
) -> RusticolResult<u32> {
    for candidate in (1..=requested).rev() {
        let stride = crate::direct_arena::checked_aligned_point_stride(candidate)? as usize;
        let scalar_planes = usize::try_from(
            current_planes
                .checked_add(amplitude_planes)
                .and_then(|count| count.checked_mul(2))
                .ok_or_else(|| invalid("eager direct workspace planes overflow"))?,
        )
        .map_err(|_| invalid("eager direct workspace plane count exceeds usize"))?
        .checked_add(symjit_scratch_planes)
        .ok_or_else(|| invalid("eager direct scratch plane count overflows"))?;
        let bytes = scalar_planes
            .checked_mul(stride)
            .and_then(|count| count.checked_mul(size_of::<f64>()))
            .and_then(|bytes| {
                (reduction_groups as usize)
                    .checked_mul(stride)
                    .and_then(|count| count.checked_mul(size_of::<EagerComplex64>()))
                    .and_then(|groups| bytes.checked_add(groups))
            })
            .and_then(|bytes| {
                stride
                    .checked_mul(size_of::<f64>())
                    .and_then(|reduced| bytes.checked_add(reduced))
            })
            .ok_or_else(|| invalid("eager direct workspace size overflows"))?;
        // The target may reduce batching, but must not reject a process whose
        // minimum aligned tile is larger. Fill the already allocated pitch.
        if bytes <= workspace_bytes
            || candidate <= (crate::direct_arena::DIRECT_ARENA_ALIGNMENT / size_of::<f64>()) as u32
        {
            return Ok(candidate);
        }
    }
    Err(invalid(
        "eager Direct-Arena workspace cannot hold one active point",
    ))
}

fn direct_workspace_bytes(
    workspace: &EagerDirectTableWorkspace,
    reduction_complex_count: usize,
    reduced_count: usize,
    factor_count: usize,
) -> RusticolResult<usize> {
    let symjit_workspace_bytes = workspace.symjit_workspace_bytes()?;
    usize::try_from(workspace.arena().allocation_counters().requested_bytes)
        .ok()
        .and_then(|arena| {
            reduction_complex_count
                .checked_mul(size_of::<EagerComplex64>())
                .and_then(|reduction| arena.checked_add(reduction))
        })
        .and_then(|bytes| {
            reduced_count
                .checked_mul(size_of::<f64>())
                .and_then(|reduced| bytes.checked_add(reduced))
        })
        .and_then(|bytes| {
            factor_count
                .checked_mul(2 * size_of::<f64>())
                .and_then(|factors| bytes.checked_add(factors))
        })
        .and_then(|bytes| bytes.checked_add(symjit_workspace_bytes))
        .ok_or_else(|| invalid("eager direct workspace byte accounting overflows"))
}

#[cfg(all(test, feature = "f64-symjit", target_arch = "aarch64"))]
mod tests {
    use std::path::PathBuf;

    use super::super::plan_v3_tests::Fixture;
    use super::*;
    use crate::engine::compile_symbolica_program_to_plane_application_bytes;
    use crate::engine::count_allocations;
    use crate::engine::symjit_eager_direct::eager_direct_descriptor_for_source_application_bytes;

    fn source_and_descriptor() -> (Vec<u8>, Vec<u8>) {
        let bytes = compile_symbolica_program_to_plane_application_bytes(
            "([('add', ('temp', 0), [('param', 0), ('param', 1)], 0), \
              ('assign', ('out', 0), ('temp', 0))], 3, [])",
            3,
            1,
            2,
            true,
        )
        .expect("compile eager whole-plan P-kernel");
        let descriptor = eager_direct_descriptor_for_source_application_bytes(
            &bytes,
            3,
            1,
            std::path::Path::new("eager-whole-plan-test.symjit"),
        )
        .expect("encode table descriptor");
        (bytes, descriptor)
    }

    fn runtime(fixture: &Fixture, source: &[u8], descriptor: &[u8]) -> EagerDirectExecutionRuntime {
        runtime_with_point_tile(fixture, source, descriptor, 128)
    }

    fn runtime_with_point_tile(
        fixture: &Fixture,
        source: &[u8],
        descriptor: &[u8],
        point_tile_size: usize,
    ) -> EagerDirectExecutionRuntime {
        let inputs = [
            EagerKernelInput::FirstCurrentComponent(0),
            EagerKernelInput::SecondCurrentComponent(0),
            EagerKernelInput::CouplingReal,
        ];
        let prepared = [
            EagerDirectPreparedKernel {
                kernel_id: 10,
                role: EagerKernelRole::Vertex,
                inputs: &inputs,
                output_component_count: 1,
                application: EagerDirectPreparedApplication::Symjit {
                    source_application: source,
                    descriptor,
                    expected_optimization_level: 2,
                    expected_compression: true,
                },
                display_path: PathBuf::from("whole-plan-vertex.symjit"),
            },
            EagerDirectPreparedKernel {
                kernel_id: 11,
                role: EagerKernelRole::Closure,
                inputs: &inputs,
                output_component_count: 1,
                application: EagerDirectPreparedApplication::Symjit {
                    source_application: source,
                    descriptor,
                    expected_optimization_level: 2,
                    expected_compression: true,
                },
                display_path: PathBuf::from("whole-plan-closure.symjit"),
            },
        ];
        EagerDirectExecutionRuntime::from_plan_v3_sections(
            fixture.sections(),
            &prepared,
            EagerRuntimeOptions {
                point_tile_size,
                workspace_bytes: 16 * 1024 * 1024,
            },
        )
        .expect("construct eager whole-plan Direct-Arena runtime")
    }

    fn inputs(points: usize) -> (Vec<EagerComplex64>, Vec<f64>) {
        let mut values = vec![EagerComplex64::new(-91.0, 37.0); 3 * points];
        for point in 0..points {
            values[point] = EagerComplex64::new(1.0 + point as f64 / 8.0, -0.25 * point as f64);
            values[points + point] =
                EagerComplex64::new(3.0 + point as f64 / 4.0, 0.5 * point as f64);
        }
        let momenta = (0..12 * points).map(|index| index as f64 / 16.0).collect();
        (values, momenta)
    }

    fn expected(values: &[EagerComplex64], points: usize, point: usize) -> (EagerComplex64, f64) {
        let current = (values[point] + values[points + point]) * 15.0;
        let amplitude = (current + values[point]) * 77.0;
        let reduced = (EagerComplex64::new(221.0, 0.0) * amplitude * amplitude.conj()).re;
        (amplitude, reduced)
    }

    fn assert_close(actual: f64, expected: f64, label: &str) {
        let tolerance = 1.0e-15 + 1.0e-12 * actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: {actual:.17e} != {expected:.17e} (tolerance {tolerance:.3e})"
        );
    }

    #[test]
    fn whole_plan_preserves_all_required_tile_tails_and_empty_point_selectors() {
        let fixture = Fixture::new();
        let (source, descriptor) = source_and_descriptor();
        let mut direct = runtime(&fixture, &source, &descriptor);
        assert_eq!(direct.effective_point_tile_size(), 64);

        for points in [1_usize, 7, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025] {
            let (values, momenta) = inputs(points);
            let mut amplitudes = vec![EagerComplex64::new(-13.0, 29.0); points];
            let mut reduced = vec![-1.0; points];
            direct
                .evaluate_into(
                    points,
                    &values,
                    &momenta,
                    &[],
                    &mut amplitudes,
                    &mut reduced,
                )
                .expect("evaluate eager whole-plan tail");
            let mut totals_only = vec![-2.0; points];
            direct
                .evaluate_totals_into(points, &values, &momenta, &[], &mut totals_only)
                .expect("evaluate eager totals-only tail");
            for point in 0..points {
                let (expected_amplitude, expected_reduced) = expected(&values, points, point);
                assert_close(
                    amplitudes[point].re,
                    expected_amplitude.re,
                    "amplitude real",
                );
                assert_close(
                    amplitudes[point].im,
                    expected_amplitude.im,
                    "amplitude imaginary",
                );
                assert_close(reduced[point], expected_reduced, "reduced total");
                assert_eq!(
                    totals_only[point].to_bits(),
                    reduced[point].to_bits(),
                    "totals-only reduction order drifted at point {point}"
                );
            }
        }

        let points = 129;
        let (values, momenta) = inputs(points);
        let mut full_amplitudes = vec![EagerComplex64::new(0.0, 0.0); points];
        let mut full_reduced = vec![0.0; points];
        direct
            .evaluate_into(
                points,
                &values,
                &momenta,
                &[],
                &mut full_amplitudes,
                &mut full_reduced,
            )
            .expect("evaluate full eager selector reference");
        let profiled = direct
            .evaluate_profile_into(
                points,
                &values,
                &momenta,
                &[],
                &mut full_amplitudes,
                &mut full_reduced,
            )
            .expect("profile full eager selector reference");
        assert!(
            profiled.internal_scratch_bytes > 0,
            "profile must expose eager P-kernel scratch traffic separately"
        );

        let mut selected_amplitudes = vec![EagerComplex64::new(-13.0, 29.0); points];
        direct
            .evaluate_selected_active_amplitudes_into(
                &[7],
                points,
                &values,
                &momenta,
                &[],
                &mut selected_amplitudes,
            )
            .expect("evaluate selected eager amplitudes");
        for (selected, full) in selected_amplitudes.iter().zip(&full_amplitudes) {
            assert_close(selected.re, full.re, "selected amplitude real");
            assert_close(selected.im, full.im, "selected amplitude imaginary");
        }

        let mut offsets = Vec::with_capacity(points + 1);
        let mut groups = Vec::with_capacity(points / 2);
        let mut weights = Vec::with_capacity(points / 2);
        offsets.push(0);
        for point in 0..points {
            if point.is_multiple_of(2) {
                groups.push(7);
                weights.push(1.0);
            }
            offsets.push(groups.len());
        }
        let mut point_selected = vec![-1.0; points];
        direct
            .evaluate_point_selected_group_sets_into(
                &offsets,
                &groups,
                &weights,
                points,
                &values,
                &momenta,
                &[],
                &mut point_selected,
            )
            .expect("evaluate mixed empty/non-empty eager point selectors");
        for point in 0..points {
            let expected = if point.is_multiple_of(2) {
                full_reduced[point]
            } else {
                0.0
            };
            assert_close(point_selected[point], expected, "point-selected total");
        }
    }

    #[test]
    fn whole_plan_preserves_requested_tiles_below_the_locality_cap() {
        let fixture = Fixture::new();
        let (source, descriptor) = source_and_descriptor();
        let direct = runtime_with_point_tile(&fixture, &source, &descriptor, 32);
        assert_eq!(direct.effective_point_tile_size(), 32);
    }

    #[test]
    fn broadcast_planes_refresh_once_and_only_for_changed_parameters() {
        let mut fixture = Fixture::new();
        fixture.prepared_parameter_count = 1;
        let (source, descriptor) = source_and_descriptor();
        let mut direct = runtime(&fixture, &source, &descriptor);
        let points = 7;
        let (values, momenta) = inputs(points);
        let mut amplitudes = vec![EagerComplex64::new(0.0, 0.0); points];
        let mut reduced = vec![0.0; points];
        let parameter = [EagerComplex64::new(2.0, -3.0)];

        let first = direct
            .evaluate_profile_into(
                points,
                &values,
                &momenta,
                &parameter,
                &mut amplitudes,
                &mut reduced,
            )
            .expect("initialize eager broadcast planes");
        assert!(direct.elide_structural_zero_fills);
        let zero_plane = direct
            .shared_zero_plane_index
            .expect("SymJIT eager execution must reserve a shared-zero plane");
        assert_eq!(zero_plane % 2, 1);
        let zero_component = (zero_plane / 2) as usize;
        let point_stride = direct.workspace.arena().point_stride() as usize;
        let (_, current_imaginary) = direct.workspace.arena().current_slices();
        assert!(
            current_imaginary[zero_component * point_stride..(zero_component + 1) * point_stride]
                .iter()
                .all(|value| *value == 0.0),
            "shared-zero plane changed during eager execution"
        );
        let stride = direct.effective_point_tile_size() as u64;
        assert_eq!(
            first.internal_broadcast_bytes,
            4 * stride * std::mem::size_of::<f64>() as u64,
            "initial refresh must fill the two real coupling planes and the split-complex parameter"
        );
        for point in 0..points {
            let (expected_amplitude, expected_reduced) = expected(&values, points, point);
            assert_close(
                amplitudes[point].re,
                expected_amplitude.re,
                "parameter-cache amplitude real",
            );
            assert_close(
                amplitudes[point].im,
                expected_amplitude.im,
                "parameter-cache amplitude imaginary",
            );
            assert_close(
                reduced[point],
                expected_reduced,
                "parameter-cache reduced total",
            );
        }

        let unchanged = direct
            .evaluate_profile_into(
                points,
                &values,
                &momenta,
                &parameter,
                &mut amplitudes,
                &mut reduced,
            )
            .expect("reuse eager broadcast planes");
        assert_eq!(unchanged.internal_broadcast_bytes, 0);

        let changed_parameter = [EagerComplex64::new(5.0, -3.0)];
        let changed = direct
            .evaluate_profile_into(
                points,
                &values,
                &momenta,
                &changed_parameter,
                &mut amplitudes,
                &mut reduced,
            )
            .expect("refresh changed eager parameter plane");
        assert_eq!(
            changed.internal_broadcast_bytes,
            2 * stride * std::mem::size_of::<f64>() as u64,
            "a changed unused parameter must not rewrite stationary coupling planes"
        );

        let (result, allocations, bytes) = count_allocations(|| {
            direct.evaluate_into(
                points,
                &values,
                &momenta,
                &changed_parameter,
                &mut amplitudes,
                &mut reduced,
            )
        });
        result.expect("repeat eager evaluation with cached parameter");
        assert_eq!(
            (allocations, bytes),
            (0, 0),
            "cached eager broadcast execution allocated"
        );
    }

    #[test]
    fn warmed_whole_plan_and_selector_caches_allocate_zero() {
        let fixture = Fixture::new();
        let (source, descriptor) = source_and_descriptor();
        let mut direct = runtime(&fixture, &source, &descriptor);
        let points = 129;
        let (values, momenta) = inputs(points);
        let mut amplitudes = vec![EagerComplex64::new(0.0, 0.0); points];
        let mut reduced = vec![0.0; points];
        direct
            .evaluate_into(
                points,
                &values,
                &momenta,
                &[],
                &mut amplitudes,
                &mut reduced,
            )
            .expect("warm whole-plan execution");
        let (result, allocations, bytes) = count_allocations(|| {
            direct.evaluate_into(
                points,
                &values,
                &momenta,
                &[],
                &mut amplitudes,
                &mut reduced,
            )
        });
        result.expect("repeat whole-plan execution");
        assert_eq!(
            (allocations, bytes),
            (0, 0),
            "warmed whole-plan execution allocated"
        );

        direct
            .evaluate_totals_into(points, &values, &momenta, &[], &mut reduced)
            .expect("warm totals-only execution");
        let (result, allocations, bytes) = count_allocations(|| {
            direct.evaluate_totals_into(points, &values, &momenta, &[], &mut reduced)
        });
        result.expect("repeat totals-only execution");
        assert_eq!(
            (allocations, bytes),
            (0, 0),
            "warmed totals-only execution allocated"
        );

        direct
            .evaluate_selected_active_amplitudes_into(
                &[7],
                points,
                &values,
                &momenta,
                &[],
                &mut amplitudes,
            )
            .expect("warm global selector schedule");
        let (result, allocations, bytes) = count_allocations(|| {
            direct.evaluate_selected_active_amplitudes_into(
                &[7],
                points,
                &values,
                &momenta,
                &[],
                &mut amplitudes,
            )
        });
        result.expect("repeat global selector schedule");
        assert_eq!(
            (allocations, bytes),
            (0, 0),
            "warmed global selector execution allocated"
        );

        let mut offsets = Vec::with_capacity(points + 1);
        let mut groups = Vec::with_capacity(points);
        let mut weights = Vec::with_capacity(points);
        offsets.push(0);
        for point in 0..points {
            if !point.is_multiple_of(3) {
                groups.push(7);
                weights.push(1.0);
            }
            offsets.push(groups.len());
        }
        direct
            .evaluate_point_selected_group_sets_into(
                &offsets,
                &groups,
                &weights,
                points,
                &values,
                &momenta,
                &[],
                &mut reduced,
            )
            .expect("warm per-point selector schedules");
        let (result, allocations, bytes) = count_allocations(|| {
            direct.evaluate_point_selected_group_sets_into(
                &offsets,
                &groups,
                &weights,
                points,
                &values,
                &momenta,
                &[],
                &mut reduced,
            )
        });
        result.expect("repeat per-point selector schedules");
        assert_eq!(
            (allocations, bytes),
            (0, 0),
            "warmed per-point selector execution allocated"
        );
    }
}
