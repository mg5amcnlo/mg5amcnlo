// SPDX-License-Identifier: 0BSD

//! Developer-gated eager invocation/fanout Direct-Arena scheduler.
//!
//! This first scheduler slice deliberately stops at the stage boundary:
//! finalization and closure remain independent eager events. Unlike the
//! adapter-only ABI tests, this module consumes authenticated plan-v3 rows,
//! derives their event lifetimes, allocates physical planes with the shared
//! allocator, preserves selector pruning and invocation/fanout order, and
//! executes real prepared SymJIT applications through immutable binding-v2
//! tables.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use crate::artifact::PinnedNativeLibrary;
use crate::direct_arena::{
    DirectArenaInterval, DirectArenaLayout, DirectArenaTrafficCounters, assign_direct_arena,
};
use crate::engine::symjit_eager_direct::{
    EAGER_DIRECT_SOURCE_APPLICATION_ABI, EAGER_DIRECT_TABLE_BINDING_ABI,
    EAGER_DIRECT_TABLE_DESCRIPTOR_ABI, EagerDirectArenaPlaneBinding, EagerDirectTableRows,
    EagerDirectTableWorkspace, LoadedSymjitEagerDirectTable,
};
use crate::{
    EAGER_OUTPUT_FACTOR_COUPLING_IMAG, EAGER_OUTPUT_FACTOR_COUPLING_REAL, EAGER_OUTPUT_FACTOR_NONE,
    MISSING_U32, RusticolError, RusticolResult,
};

use super::plan::{ComponentRange, EagerExecutionPlan, ScheduledAttachment, ScheduledInvocation};
use super::{
    EagerComplex64, EagerKernelInput, EagerKernelRole, EagerKernelSpec, EagerPlanV3Sections,
};

/// One prepared source application and its portable eager table descriptor.
pub(crate) enum EagerDirectPreparedApplication<'a> {
    Symjit {
        source_application: &'a [u8],
        descriptor: &'a [u8],
        expected_optimization_level: u32,
        expected_compression: bool,
    },
    Native {
        library: &'a Arc<PinnedNativeLibrary>,
        function_name: &'a str,
        source_application_abi: &'a str,
        invocation_stride: u32,
        attachment_stride: u32,
        target_triple: &'a str,
        evaluator_state_sha256: &'a str,
        simd_lane_width: u32,
    },
}

pub(crate) struct EagerDirectPreparedKernel<'a> {
    pub kernel_id: u32,
    pub role: EagerKernelRole,
    pub inputs: &'a [EagerKernelInput],
    pub output_component_count: u32,
    pub application: EagerDirectPreparedApplication<'a>,
    pub display_path: PathBuf,
}

struct EagerDirectStageCall {
    application: LoadedSymjitEagerDirectTable,
    rows: EagerDirectTableRows,
}

#[derive(Clone, Copy)]
struct FactorSpec {
    factor: EagerComplex64,
    coupling_slot_id: u32,
    output_factor_source: u32,
}

#[derive(Clone, Copy)]
pub(super) struct SemanticCatalog {
    value_start: u32,
    current_start: u32,
    momentum_start: u32,
    coupling_real_start: u32,
    coupling_imag_start: u32,
    parameter_start: u32,
    semantic_count: u32,
}

impl SemanticCatalog {
    pub(super) fn new(sections: EagerPlanV3Sections<'_>) -> RusticolResult<Self> {
        let value_count = count_u32(sections.values.len(), "value slots")?;
        let current_count = count_u32(sections.currents.len(), "currents")?;
        let momentum_count = count_u32(sections.momenta.len(), "momentum slots")?;
        let coupling_count = count_u32(sections.couplings.len(), "couplings")?;
        let parameter_count = sections.prepared_parameter_count;
        let value_start = 0;
        let current_start = value_count;
        let momentum_start = current_start
            .checked_add(current_count)
            .ok_or_else(|| invalid("eager direct semantic current range overflows"))?;
        let coupling_real_start = momentum_start
            .checked_add(momentum_count)
            .ok_or_else(|| invalid("eager direct semantic momentum range overflows"))?;
        let coupling_imag_start = coupling_real_start
            .checked_add(coupling_count)
            .ok_or_else(|| invalid("eager direct semantic coupling range overflows"))?;
        let parameter_start = coupling_imag_start
            .checked_add(coupling_count)
            .ok_or_else(|| invalid("eager direct semantic coupling range overflows"))?;
        let semantic_count = parameter_start
            .checked_add(parameter_count)
            .ok_or_else(|| invalid("eager direct semantic parameter range overflows"))?;
        Ok(Self {
            value_start,
            current_start,
            momentum_start,
            coupling_real_start,
            coupling_imag_start,
            parameter_start,
            semantic_count,
        })
    }

    pub(super) fn value(self, id: u32) -> RusticolResult<u32> {
        checked_semantic(self.value_start, id, self.current_start, "value")
    }

    pub(super) fn current(self, id: u32) -> RusticolResult<u32> {
        checked_semantic(self.current_start, id, self.momentum_start, "current")
    }

    pub(super) fn momentum(self, id: u32) -> RusticolResult<u32> {
        checked_semantic(
            self.momentum_start,
            id,
            self.coupling_real_start,
            "momentum",
        )
    }

    pub(super) fn coupling_real(self, id: u32) -> RusticolResult<u32> {
        checked_semantic(
            self.coupling_real_start,
            id,
            self.coupling_imag_start,
            "coupling-real",
        )
    }

    pub(super) fn coupling_imag(self, id: u32) -> RusticolResult<u32> {
        checked_semantic(
            self.coupling_imag_start,
            id,
            self.parameter_start,
            "coupling-imag",
        )
    }

    pub(super) fn parameter(self, id: u32) -> RusticolResult<u32> {
        checked_semantic(self.parameter_start, id, self.semantic_count, "parameter")
    }
}

/// Real plan-v3 eager invocation/fanout prototype.
///
/// The production cutover will extend this envelope with finalization,
/// closure, and reduction event executors. Keeping this type separate from
/// [`super::EagerExecutionRuntime`] makes the temporary developer dual-run
/// explicit and prevents a packet fallback from being mistaken for a direct
/// lane.
pub(crate) struct EagerDirectInvocationPrototype {
    plan: EagerExecutionPlan,
    catalog: SemanticCatalog,
    layout: DirectArenaLayout,
    workspace: EagerDirectTableWorkspace,
    calls: Box<[EagerDirectStageCall]>,
    factor_specs: Box<[FactorSpec]>,
    initial_value_slots: Box<[u32]>,
    value_ranges: Box<[ComponentRange]>,
    momentum_ranges: Box<[ComponentRange]>,
    traffic: DirectArenaTrafficCounters,
}

impl EagerDirectInvocationPrototype {
    /// Construct one selector-specialized direct invocation schedule.
    ///
    /// `active_groups=None` retains every physical group. A sorted, unique
    /// slice (including the empty slice) produces the same dependency-pruned
    /// invocation and attachment order as the existing eager selector lane.
    pub(crate) fn from_plan_v3_sections(
        sections: EagerPlanV3Sections<'_>,
        prepared: &[EagerDirectPreparedKernel<'_>],
        active_groups: Option<&[u32]>,
        tile_capacity: u32,
    ) -> RusticolResult<Self> {
        if tile_capacity == 0 {
            return Err(invalid(
                "eager direct invocation tile capacity must be positive",
            ));
        }
        let plan = EagerExecutionPlan::from_plan_v3_sections(sections)?;
        if plan.stages.len() != 1 {
            return Err(RusticolError::compatibility(format!(
                "the eager direct invocation prototype currently accepts one stage, found {}; \
                 finalization and closure remain separate developer-oracle events",
                plan.stages.len()
            )));
        }
        Self::from_validated_plan_stage(
            sections,
            plan,
            prepared,
            active_groups,
            tile_capacity,
            0,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_validated_plan_stage(
        sections: EagerPlanV3Sections<'_>,
        plan: EagerExecutionPlan,
        prepared: &[EagerDirectPreparedKernel<'_>],
        active_groups: Option<&[u32]>,
        tile_capacity: u32,
        stage_position: usize,
        kernel_filter: Option<u32>,
    ) -> RusticolResult<Self> {
        validate_active_groups(&plan, active_groups)?;
        let catalog = SemanticCatalog::new(sections)?;
        let layout = derive_event_layout(sections, catalog, &plan)?;
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

        let mut plane_bindings = Vec::new();
        plane_bindings
            .try_reserve_exact(layout.component_count() as usize * 2)
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not reserve eager direct plane bindings: {error}"
                ))
            })?;
        for component in 0..layout.component_count() {
            plane_bindings.push(EagerDirectArenaPlaneBinding::CurrentReal(component));
            plane_bindings.push(EagerDirectArenaPlaneBinding::CurrentImag(component));
        }

        let stage = plan.stages.get(stage_position).ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "eager direct stage position {stage_position} is out of bounds"
            ))
        })?;
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
            applications.insert(artifact.kernel_id, application);
        }

        let mut calls = Vec::new();
        let mut factor_specs = Vec::new();
        // Selector specialization changes which contribution is the first
        // retained write to a current. Recompute overwrite/add semantics in
        // exactly the retained, stable execution order rather than inheriting
        // flags from the full schedule.
        let mut selected_written_currents = BTreeSet::new();
        let mut cursor = 0usize;
        while cursor < stage.invocations.len() {
            let kernel_id = stage.invocations[cursor].row.kernel_id;
            let run_stop = stage.invocations[cursor..]
                .iter()
                .position(|row| row.row.kernel_id != kernel_id)
                .map_or(stage.invocations.len(), |offset| cursor + offset);
            if kernel_filter.is_some_and(|selected| selected != kernel_id) {
                cursor = run_stop;
                continue;
            }
            let kernel = plan.kernels.get(&kernel_id).ok_or_else(|| {
                RusticolError::internal(format!("eager direct schedule lost kernel {kernel_id}"))
            })?;
            let mut invocation_bytes = Vec::new();
            let mut attachment_bytes = Vec::new();
            let mut local_attachment_count = 0_u32;
            for invocation in &stage.invocations[cursor..run_stop] {
                if !row_active(&plan, invocation.selector_domain_id, active_groups)? {
                    continue;
                }
                let active_attachment_count = stage.attachments
                    [invocation.attachment_range.clone()]
                .iter()
                .map(|attachment| row_active(&plan, attachment.selector_domain_id, active_groups))
                .collect::<RusticolResult<Vec<_>>>()?
                .into_iter()
                .filter(|active| *active)
                .count();
                if active_attachment_count == 0 {
                    continue;
                }
                encode_invocation_inputs(
                    &mut invocation_bytes,
                    invocation,
                    &kernel.inputs,
                    catalog,
                    &layout,
                )?;
                push_u32(&mut invocation_bytes, local_attachment_count);
                push_u32(
                    &mut invocation_bytes,
                    count_u32(active_attachment_count, "active attachments")?,
                );
                for attachment in &stage.attachments[invocation.attachment_range.clone()] {
                    if !row_active(&plan, attachment.selector_domain_id, active_groups)? {
                        continue;
                    }
                    encode_attachment_destinations(
                        &mut attachment_bytes,
                        attachment,
                        kernel.output_component_count,
                        catalog,
                        &layout,
                    )?;
                    let factor_index = count_u32(factor_specs.len(), "direct factors")?;
                    push_u32(&mut attachment_bytes, factor_index);
                    push_u32(
                        &mut attachment_bytes,
                        retained_attachment_operation(
                            &mut selected_written_currents,
                            attachment.row.result_current_id,
                        ),
                    );
                    factor_specs.push(FactorSpec {
                        factor: EagerComplex64::new(
                            attachment.row.factor_real,
                            attachment.row.factor_imag,
                        ),
                        coupling_slot_id: invocation.row.coupling_slot_id,
                        output_factor_source: invocation.row.output_factor_source,
                    });
                    local_attachment_count = local_attachment_count
                        .checked_add(1)
                        .ok_or_else(|| invalid("eager direct attachment count overflows"))?;
                }
            }
            if !invocation_bytes.is_empty() {
                let application = applications.remove(&kernel_id).ok_or_else(|| {
                    RusticolError::compatibility(format!(
                        "eager direct invocation schedule has no prepared table for kernel {kernel_id}"
                    ))
                })?;
                let rows = application.load_rows(invocation_bytes, attachment_bytes)?;
                calls.push(EagerDirectStageCall { application, rows });
            }
            cursor = run_stop;
        }

        let factor_len = factor_specs.len().max(1);
        let mut workspace = EagerDirectTableWorkspace::new(
            layout.component_count(),
            count_u32(plan.amplitude_count, "amplitude planes")?,
            tile_capacity,
            &plane_bindings,
            None,
            Vec::new(),
            vec![0.0; factor_len],
            vec![0.0; factor_len],
        )?;
        workspace.begin_tile(tile_capacity)?;
        for call in &calls {
            call.application
                .validate_call(&call.rows, &mut workspace, 0, tile_capacity)?;
        }

        Ok(Self {
            plan,
            catalog,
            layout,
            workspace,
            calls: calls.into_boxed_slice(),
            factor_specs: factor_specs.into_boxed_slice(),
            initial_value_slots,
            value_ranges,
            momentum_ranges,
            traffic: DirectArenaTrafficCounters::default(),
        })
    }

    /// Extract one invocation-only stage slice from a fully validated
    /// multistage plan for the retained-artifact oracle. This test-only entry
    /// point does not relax the production constructor: finalization, closure,
    /// and whole-plan multistage execution remain fail-closed.
    #[cfg(test)]
    pub(crate) fn from_multistage_stage_for_test(
        sections: EagerPlanV3Sections<'_>,
        prepared: &[EagerDirectPreparedKernel<'_>],
        active_groups: Option<&[u32]>,
        tile_capacity: u32,
        stage_position: usize,
        kernel_id: u32,
    ) -> RusticolResult<Self> {
        if tile_capacity == 0 {
            return Err(invalid(
                "eager direct invocation tile capacity must be positive",
            ));
        }
        let plan = EagerExecutionPlan::from_plan_v3_sections(sections)?;
        if plan.stages.len() < 2 {
            return Err(RusticolError::compatibility(
                "the retained-artifact stage oracle requires a genuine multistage eager plan",
            ));
        }
        Self::from_validated_plan_stage(
            sections,
            plan,
            prepared,
            active_groups,
            tile_capacity,
            stage_position,
            Some(kernel_id),
        )
    }

    pub(crate) fn layout(&self) -> &DirectArenaLayout {
        &self.layout
    }

    pub(crate) const fn traffic(&self) -> DirectArenaTrafficCounters {
        self.traffic
    }

    pub(crate) fn current_value(
        &self,
        current_id: u32,
        component: u32,
        point: u32,
    ) -> RusticolResult<EagerComplex64> {
        if point >= self.workspace.arena().active_point_count() {
            return Err(invalid(
                "eager direct current point is outside the active tile",
            ));
        }
        let semantic = self.catalog.current(current_id)?;
        let physical = assigned_component(&self.layout, semantic, component)?;
        let stride = self.workspace.arena().point_stride() as usize;
        let index = physical as usize * stride + point as usize;
        let (real, imaginary) = self.workspace.arena().current_slices();
        Ok(EagerComplex64::new(real[index], imaginary[index]))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_invocations(
        &mut self,
        point_count: u32,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
    ) -> RusticolResult<()> {
        if point_count == 0 || point_count > self.workspace.arena().tile_capacity() {
            return Err(invalid(format!(
                "eager direct invocation point count {point_count} is outside 1..={}",
                self.workspace.arena().tile_capacity()
            )));
        }
        validate_component_buffer(
            "initial values",
            initial_values.len(),
            self.plan.values.component_count,
            point_count,
        )?;
        validate_component_buffer(
            "momenta",
            momenta.len(),
            self.plan.momenta.component_count,
            point_count,
        )?;
        if model_parameters.len() != self.plan.parameter_count {
            return Err(invalid(format!(
                "eager direct model parameter count is {}, expected {}",
                model_parameters.len(),
                self.plan.parameter_count
            )));
        }
        if model_parameters
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(invalid("eager direct model parameters must be finite"));
        }

        self.workspace.begin_tile(point_count)?;
        self.workspace
            .clear_current_active(0, self.layout.component_count())?;
        let stride = self.workspace.arena().point_stride() as usize;
        let points = point_count as usize;
        {
            let (real, imaginary, _, _) = self.workspace.split_arena_slices_mut();
            for &slot in &self.initial_value_slots {
                let range = self.value_ranges[slot as usize];
                let semantic = self.catalog.value(slot)?;
                for component in 0..range.len {
                    let physical = assigned_component(
                        &self.layout,
                        semantic,
                        count_u32(component, "value component")?,
                    )? as usize;
                    let target = physical * stride;
                    let source = (range.start + component) * points;
                    for point in 0..points {
                        let value = initial_values[source + point];
                        real[target + point] = value.re;
                        imaginary[target + point] = value.im;
                    }
                }
            }
            for (slot, range) in self.momentum_ranges.iter().copied().enumerate() {
                let semantic = self.catalog.momentum(slot as u32)?;
                for component in 0..range.len {
                    let physical = assigned_component(
                        &self.layout,
                        semantic,
                        count_u32(component, "momentum component")?,
                    )? as usize;
                    let target = physical * stride;
                    let source = (range.start + component) * points;
                    for point in 0..points {
                        real[target + point] = momenta[source + point];
                        imaginary[target + point] = 0.0;
                    }
                }
            }
            for (coupling_id, row) in self.plan.couplings.iter().copied().enumerate() {
                let coupling = resolve_coupling(row, model_parameters);
                fill_real_input_plane(
                    real,
                    imaginary,
                    stride,
                    assigned_component(
                        &self.layout,
                        self.catalog.coupling_real(coupling_id as u32)?,
                        0,
                    )?,
                    points,
                    coupling.re,
                );
                fill_real_input_plane(
                    real,
                    imaginary,
                    stride,
                    assigned_component(
                        &self.layout,
                        self.catalog.coupling_imag(coupling_id as u32)?,
                        0,
                    )?,
                    points,
                    coupling.im,
                );
            }
            for (parameter_id, value) in model_parameters.iter().copied().enumerate() {
                let physical = assigned_component(
                    &self.layout,
                    self.catalog.parameter(parameter_id as u32)?,
                    0,
                )? as usize;
                let target = physical * stride;
                real[target..target + points].fill(value.re);
                imaginary[target..target + points].fill(value.im);
            }
        }
        {
            let (factor_re, factor_im) = self.workspace.factors_mut();
            for (index, spec) in self.factor_specs.iter().copied().enumerate() {
                let coupling = resolve_coupling(
                    self.plan.couplings[spec.coupling_slot_id as usize],
                    model_parameters,
                );
                let scale = match spec.output_factor_source {
                    EAGER_OUTPUT_FACTOR_NONE => 1.0,
                    EAGER_OUTPUT_FACTOR_COUPLING_REAL => coupling.re,
                    EAGER_OUTPUT_FACTOR_COUPLING_IMAG => coupling.im,
                    _ => {
                        return Err(RusticolError::integrity(
                            "eager direct factor references an invalid output-factor source",
                        ));
                    }
                };
                factor_re[index] = spec.factor.re * scale;
                factor_im[index] = spec.factor.im * scale;
            }
        }
        self.execute_preinitialized_invocations(point_count)
    }

    /// Execute immutable table rows after the caller has initialized the
    /// arena and factor catalog for the active tile.
    fn execute_preinitialized_invocations(&mut self, point_count: u32) -> RusticolResult<()> {
        for call in &self.calls {
            self.traffic
                .record_call(call.rows.invocation_count(), point_count);
            // SAFETY: constructor validation authenticates immutable rows,
            // descriptors, plane catalogs, and aliases. This method changes
            // only plane/factor contents and the checked active tail.
            unsafe {
                call.application.evaluate_validated_unchecked(
                    &call.rows,
                    &mut self.workspace,
                    0,
                    point_count,
                )?;
            }
        }
        self.traffic.validate_direct()?;
        Ok(())
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EagerDirectRetainedStageCandidate {
    pub(crate) stage_position: usize,
    pub(crate) stage_index: u32,
    pub(crate) kernel_id: u32,
    pub(crate) selector_group_id: u32,
    pub(crate) full_invocation_count: usize,
    pub(crate) full_attachment_count: usize,
    pub(crate) selected_invocation_count: usize,
    pub(crate) selected_attachment_count: usize,
    pub(crate) distinct_destination_count: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EagerDirectRetainedStageVariantEvidence {
    pub(crate) invocation_count: usize,
    pub(crate) attachment_count: usize,
    pub(crate) comparison_count: usize,
    pub(crate) bitwise_comparison_count: usize,
    pub(crate) maximum_absolute_error: f64,
    pub(crate) maximum_relative_error: f64,
    pub(crate) direct_median_ns: f64,
    pub(crate) direct_mad_ns: f64,
    pub(crate) packet_median_ns: f64,
    pub(crate) packet_mad_ns: f64,
    pub(crate) direct_over_packet: f64,
    pub(crate) direct_call_traffic: DirectArenaTrafficCounters,
    pub(crate) arena_initialization_write_bytes: u64,
    pub(crate) factor_fill_write_bytes: u64,
    pub(crate) packet_input_materialization_bytes_per_call: u64,
    pub(crate) packet_output_materialization_bytes_per_call: u64,
    pub(crate) packet_scatter_bytes_per_call: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EagerDirectRetainedStageEvidence {
    pub(crate) candidate: EagerDirectRetainedStageCandidate,
    pub(crate) full: EagerDirectRetainedStageVariantEvidence,
    pub(crate) selected: EagerDirectRetainedStageVariantEvidence,
}

/// Pick a retained-artifact kernel run that proves both a genuine multi-row,
/// multi-destination table and nontrivial runtime selector pruning.
#[cfg(test)]
pub(crate) fn select_retained_multistage_candidate(
    sections: EagerPlanV3Sections<'_>,
) -> RusticolResult<EagerDirectRetainedStageCandidate> {
    let plan = EagerExecutionPlan::from_plan_v3_sections(sections)?;
    if plan.stages.len() < 2 {
        return Err(RusticolError::compatibility(
            "the retained eager stage oracle requires a multistage plan",
        ));
    }
    let selector = plan.selector_domains.as_ref().ok_or_else(|| {
        RusticolError::compatibility(
            "the retained eager stage oracle requires runtime selector domains",
        )
    })?;
    let mut best = None;
    for (stage_position, stage) in plan.stages.iter().enumerate() {
        let mut cursor = 0;
        while cursor < stage.invocations.len() {
            let kernel_id = stage.invocations[cursor].row.kernel_id;
            let stop = stage.invocations[cursor..]
                .iter()
                .position(|row| row.row.kernel_id != kernel_id)
                .map_or(stage.invocations.len(), |offset| cursor + offset);
            let run = &stage.invocations[cursor..stop];
            let full_attachment_count = run
                .iter()
                .map(|invocation| invocation.attachment_range.len())
                .sum::<usize>();
            let distinct_destinations = run
                .iter()
                .flat_map(|invocation| {
                    stage.attachments[invocation.attachment_range.clone()]
                        .iter()
                        .map(|attachment| attachment.row.result_current_id)
                })
                .collect::<BTreeSet<_>>();
            if run.len() >= 2 && distinct_destinations.len() >= 2 {
                for &group_id in &selector.group_ids {
                    let groups = [group_id];
                    let (selected_invocations, selected_attachments) =
                        retained_run_counts(&plan, stage, run, Some(&groups))?;
                    if selected_invocations == 0 || selected_invocations >= run.len() {
                        continue;
                    }
                    let candidate = EagerDirectRetainedStageCandidate {
                        stage_position,
                        stage_index: stage.stage_index,
                        kernel_id,
                        selector_group_id: group_id,
                        full_invocation_count: run.len(),
                        full_attachment_count,
                        selected_invocation_count: selected_invocations,
                        selected_attachment_count: selected_attachments,
                        distinct_destination_count: distinct_destinations.len(),
                    };
                    if best
                        .as_ref()
                        .is_none_or(|previous: &EagerDirectRetainedStageCandidate| {
                            (
                                candidate.full_invocation_count,
                                candidate.distinct_destination_count,
                                candidate.selected_invocation_count,
                            ) > (
                                previous.full_invocation_count,
                                previous.distinct_destination_count,
                                previous.selected_invocation_count,
                            )
                        })
                    {
                        best = Some(candidate);
                    }
                }
            }
            cursor = stop;
        }
    }
    best.ok_or_else(|| {
        RusticolError::compatibility(
            "retained eager plan has no multi-row/multi-destination kernel run with nontrivial selector pruning",
        )
    })
}

#[cfg(test)]
fn retained_run_counts(
    plan: &EagerExecutionPlan,
    stage: &super::plan::EagerStagePlan,
    run: &[ScheduledInvocation],
    active_groups: Option<&[u32]>,
) -> RusticolResult<(usize, usize)> {
    let mut invocations = 0;
    let mut attachments = 0;
    for invocation in run {
        if !row_active(plan, invocation.selector_domain_id, active_groups)? {
            continue;
        }
        let retained = stage.attachments[invocation.attachment_range.clone()]
            .iter()
            .map(|attachment| row_active(plan, attachment.selector_domain_id, active_groups))
            .collect::<RusticolResult<Vec<_>>>()?
            .into_iter()
            .filter(|active| *active)
            .count();
        if retained != 0 {
            invocations += 1;
            attachments += retained;
        }
    }
    Ok((invocations, attachments))
}

/// Execute one real retained multistage artifact's chosen invocation run
/// through both the Direct-Arena table ABI and the established packet
/// gather/evaluate/scatter implementation.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_retained_multistage_oracle<B: super::EagerKernelBackend>(
    sections: EagerPlanV3Sections<'_>,
    prepared: &[EagerDirectPreparedKernel<'_>],
    candidate: EagerDirectRetainedStageCandidate,
    backend: &mut B,
    model_parameters: &[EagerComplex64],
    point_count: u32,
    sample_count: usize,
    repetitions: usize,
) -> RusticolResult<EagerDirectRetainedStageEvidence> {
    if point_count == 0 || sample_count < 3 || sample_count.is_multiple_of(2) || repetitions == 0 {
        return Err(invalid(
            "retained eager stage oracle needs points, odd sample count >= 3, and repetitions",
        ));
    }
    let selected_groups = [candidate.selector_group_id];
    let full = run_retained_stage_variant(
        sections,
        prepared,
        candidate,
        backend,
        model_parameters,
        point_count,
        sample_count,
        repetitions,
        None,
    )?;
    let selected = run_retained_stage_variant(
        sections,
        prepared,
        candidate,
        backend,
        model_parameters,
        point_count,
        sample_count,
        repetitions,
        Some(&selected_groups),
    )?;
    if full.invocation_count != candidate.full_invocation_count
        || full.attachment_count != candidate.full_attachment_count
        || selected.invocation_count != candidate.selected_invocation_count
        || selected.attachment_count != candidate.selected_attachment_count
    {
        return Err(RusticolError::internal(
            "retained eager stage oracle selection counts changed after candidate selection",
        ));
    }
    Ok(EagerDirectRetainedStageEvidence {
        candidate,
        full,
        selected,
    })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn run_retained_stage_variant<B: super::EagerKernelBackend>(
    sections: EagerPlanV3Sections<'_>,
    prepared: &[EagerDirectPreparedKernel<'_>],
    candidate: EagerDirectRetainedStageCandidate,
    backend: &mut B,
    model_parameters: &[EagerComplex64],
    point_count: u32,
    sample_count: usize,
    repetitions: usize,
    active_groups: Option<&[u32]>,
) -> RusticolResult<EagerDirectRetainedStageVariantEvidence> {
    use std::hint::black_box;

    use super::execute::execute_stage;
    use super::plan::{EagerStagePlan, mark_initial_current_writes};
    use super::runtime::{EagerWorkspace, KernelPacket, PacketRole, StageSchedule};

    let packet_plan = EagerExecutionPlan::from_plan_v3_sections(sections)?;
    let source_stage = packet_plan
        .stages
        .get(candidate.stage_position)
        .ok_or_else(|| invalid("retained eager candidate stage is out of bounds"))?;
    if source_stage.stage_index != candidate.stage_index {
        return Err(RusticolError::internal(
            "retained eager candidate stage identity changed",
        ));
    }
    let kernel = packet_plan
        .kernels
        .get(&candidate.kernel_id)
        .ok_or_else(|| invalid("retained eager candidate kernel is absent"))?;

    let mut invocations = Vec::new();
    let mut attachments = Vec::new();
    for invocation in &source_stage.invocations {
        if invocation.row.kernel_id != candidate.kernel_id
            || !row_active(&packet_plan, invocation.selector_domain_id, active_groups)?
        {
            continue;
        }
        let attachment_start = attachments.len();
        for attachment in &source_stage.attachments[invocation.attachment_range.clone()] {
            if row_active(&packet_plan, attachment.selector_domain_id, active_groups)? {
                attachments.push(*attachment);
            }
        }
        if attachments.len() == attachment_start {
            continue;
        }
        let mut retained = invocation.clone();
        retained.attachment_range = attachment_start..attachments.len();
        invocations.push(retained);
    }
    if invocations.is_empty() || attachments.is_empty() {
        return Err(RusticolError::internal(
            "retained eager stage variant unexpectedly has no work",
        ));
    }
    let zero_current_ranges = mark_initial_current_writes(&invocations, &mut attachments, &[]);
    let stage = EagerStagePlan {
        stage_index: source_stage.stage_index,
        current_component_count: source_stage.current_component_count,
        invocations,
        attachments,
        finalization_copies: Vec::new(),
        finalizations: Vec::new(),
        zero_current_ranges,
    };
    let packet = KernelPacket {
        role: PacketRole::Invocation,
        kernel_id: candidate.kernel_id,
        independent_block_size: 1,
        item_range: 0..stage.invocations.len(),
        input_components: kernel.inputs.len(),
        output_components: kernel.output_component_count as usize,
        linear_finalization: None,
    };
    let schedule = StageSchedule {
        invocation_packets: vec![packet],
        finalization_packets: Vec::new(),
    };

    let points = point_count as usize;
    let values = deterministic_complex_components(packet_plan.values.component_count, points);
    let momenta = deterministic_real_components(packet_plan.momenta.component_count, points);
    let couplings = packet_plan
        .couplings
        .iter()
        .copied()
        .map(|row| resolve_coupling(row, model_parameters))
        .collect::<Vec<_>>();
    let packet_complex_count = stage
        .invocations
        .len()
        .checked_mul(points)
        .and_then(|lanes| {
            kernel
                .inputs
                .len()
                .checked_add(kernel.output_component_count as usize)
                .and_then(|width| lanes.checked_mul(width))
        })
        .ok_or_else(|| invalid("retained eager packet workspace overflows"))?;
    let mut packet_workspace = EagerWorkspace {
        tile_capacity: points,
        values: values.clone(),
        currents: vec![
            EagerComplex64::new(0.0, 0.0);
            source_stage.current_component_count * points
        ],
        amplitudes: vec![EagerComplex64::new(0.0, 0.0); packet_plan.amplitude_count * points],
        reduction_groups: vec![
            EagerComplex64::new(0.0, 0.0);
            packet_plan.reduction_groups.len() * points
        ],
        couplings,
        reduced: vec![0.0; points],
        packet: vec![EagerComplex64::new(0.0, 0.0); packet_complex_count],
    };

    let mut direct = EagerDirectInvocationPrototype::from_multistage_stage_for_test(
        sections,
        prepared,
        active_groups,
        point_count,
        candidate.stage_position,
        candidate.kernel_id,
    )?;
    initialize_all_retained_stage_inputs(
        &mut direct,
        &stage,
        point_count,
        &values,
        &momenta,
        model_parameters,
    )?;
    direct.execute_preinitialized_invocations(point_count)?;
    execute_stage(
        &stage,
        &schedule,
        &packet_plan.kernels,
        &mut packet_workspace,
        backend,
        points,
        0,
        points,
        &momenta,
        model_parameters,
    )?;
    let mut comparison_count = 0;
    let mut bitwise_comparison_count = 0;
    let mut maximum_absolute_error = 0.0_f64;
    let mut maximum_relative_error = 0.0_f64;
    let mut compared = BTreeSet::new();
    for attachment in &stage.attachments {
        for component in 0..attachment.current.len {
            if !compared.insert((attachment.row.result_current_id, component)) {
                continue;
            }
            for point in 0..points {
                let actual = direct.current_value(
                    attachment.row.result_current_id,
                    component as u32,
                    point as u32,
                )?;
                let expected = packet_workspace.currents
                    [(attachment.current.start + component) * points + point];
                if (actual.re.to_bits(), actual.im.to_bits())
                    == (expected.re.to_bits(), expected.im.to_bits())
                {
                    bitwise_comparison_count += 1;
                } else {
                    for (actual_part, expected_part) in
                        [(actual.re, expected.re), (actual.im, expected.im)]
                    {
                        let absolute = (actual_part - expected_part).abs();
                        let relative = absolute / expected_part.abs().max(f64::MIN_POSITIVE);
                        maximum_absolute_error = maximum_absolute_error.max(absolute);
                        maximum_relative_error = maximum_relative_error.max(relative);
                        if absolute > 1.0e-15 + 1.0e-12 * expected_part.abs() {
                            return Err(RusticolError::integrity(format!(
                                "retained eager Direct-Arena stage differs from packet execution for \
                                 current {} component {component} point {point}: \
                                 {actual:?} != {expected:?}",
                                attachment.row.result_current_id
                            )));
                        }
                    }
                }
                comparison_count += 1;
            }
        }
    }

    let mut direct_samples = Vec::with_capacity(sample_count);
    let mut packet_samples = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        if sample.is_multiple_of(2) {
            direct_samples.push(measure_repeated(repetitions, || {
                direct.execute_preinitialized_invocations(point_count)
            })?);
            packet_samples.push(measure_repeated(repetitions, || {
                execute_stage(
                    &stage,
                    &schedule,
                    &packet_plan.kernels,
                    &mut packet_workspace,
                    backend,
                    points,
                    0,
                    points,
                    &momenta,
                    model_parameters,
                )
            })?);
        } else {
            packet_samples.push(measure_repeated(repetitions, || {
                execute_stage(
                    &stage,
                    &schedule,
                    &packet_plan.kernels,
                    &mut packet_workspace,
                    backend,
                    points,
                    0,
                    points,
                    &momenta,
                    model_parameters,
                )
            })?);
            direct_samples.push(measure_repeated(repetitions, || {
                direct.execute_preinitialized_invocations(point_count)
            })?);
        }
    }
    black_box(direct.current_value(stage.attachments[0].row.result_current_id, 0, 0)?);
    black_box(packet_workspace.currents[stage.attachments[0].current.start * points]);
    let (direct_median_ns, direct_mad_ns) = median_mad(&mut direct_samples);
    let (packet_median_ns, packet_mad_ns) = median_mad(&mut packet_samples);
    let direct_call_traffic = direct.traffic();
    direct_call_traffic.validate_direct()?;

    let used_value_components = stage
        .invocations
        .iter()
        .flat_map(|invocation| {
            [
                invocation.row.left_value_slot_id,
                invocation.row.right_value_slot_id,
            ]
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|slot| direct.value_ranges[slot as usize].len as u64)
        .sum::<u64>();
    let arena_initialization_write_bytes = [
        direct.layout.component_count() as u64,
        used_value_components,
        packet_plan.momenta.component_count as u64,
        (packet_plan.couplings.len() as u64).saturating_mul(2),
        packet_plan.parameter_count as u64,
    ]
    .into_iter()
    .sum::<u64>()
    .saturating_mul(u64::from(point_count))
    .saturating_mul(16);
    let factor_fill_write_bytes = (direct.factor_specs.len() as u64).saturating_mul(16);
    let lane_count = (stage.invocations.len() as u64).saturating_mul(u64::from(point_count));
    let packet_input_materialization_bytes_per_call = lane_count
        .saturating_mul(kernel.inputs.len() as u64)
        .saturating_mul(16);
    let packet_output_materialization_bytes_per_call = lane_count
        .saturating_mul(u64::from(kernel.output_component_count))
        .saturating_mul(16);
    let packet_scatter_bytes_per_call = (stage.attachments.len() as u64)
        .saturating_mul(u64::from(point_count))
        .saturating_mul(u64::from(kernel.output_component_count))
        .saturating_mul(16);

    Ok(EagerDirectRetainedStageVariantEvidence {
        invocation_count: stage.invocations.len(),
        attachment_count: stage.attachments.len(),
        comparison_count,
        bitwise_comparison_count,
        maximum_absolute_error,
        maximum_relative_error,
        direct_median_ns,
        direct_mad_ns,
        packet_median_ns,
        packet_mad_ns,
        direct_over_packet: direct_median_ns / packet_median_ns,
        direct_call_traffic,
        arena_initialization_write_bytes,
        factor_fill_write_bytes,
        packet_input_materialization_bytes_per_call,
        packet_output_materialization_bytes_per_call,
        packet_scatter_bytes_per_call,
    })
}

#[cfg(test)]
fn deterministic_complex_components(components: usize, points: usize) -> Vec<EagerComplex64> {
    (0..components)
        .flat_map(|component| {
            (0..points).map(move |point| {
                let seed = (component * points + point + 1) as f64;
                EagerComplex64::new(seed.mul_add(0.000_976_562_5, 0.25), -seed / 2048.0)
            })
        })
        .collect()
}

#[cfg(test)]
fn deterministic_real_components(components: usize, points: usize) -> Vec<f64> {
    (0..components)
        .flat_map(|component| {
            (0..points).map(move |point| {
                let seed = (component * points + point + 1) as f64;
                seed.mul_add(0.001_953_125, 0.5)
            })
        })
        .collect()
}

#[cfg(test)]
fn initialize_all_retained_stage_inputs(
    direct: &mut EagerDirectInvocationPrototype,
    stage: &super::plan::EagerStagePlan,
    point_count: u32,
    values: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    validate_component_buffer(
        "retained-stage values",
        values.len(),
        direct.plan.values.component_count,
        point_count,
    )?;
    validate_component_buffer(
        "retained-stage momenta",
        momenta.len(),
        direct.plan.momenta.component_count,
        point_count,
    )?;
    if model_parameters.len() != direct.plan.parameter_count {
        return Err(invalid(
            "retained-stage model parameter count does not match the plan",
        ));
    }
    direct.workspace.begin_tile(point_count)?;
    direct
        .workspace
        .clear_current_active(0, direct.layout.component_count())?;
    let stride = direct.workspace.arena().point_stride() as usize;
    let points = point_count as usize;
    {
        let (real, imaginary, _, _) = direct.workspace.split_arena_slices_mut();
        let used_value_slots = stage
            .invocations
            .iter()
            .flat_map(|invocation| {
                [
                    invocation.row.left_value_slot_id,
                    invocation.row.right_value_slot_id,
                ]
            })
            .collect::<BTreeSet<_>>();
        for slot in used_value_slots {
            let range = direct.value_ranges[slot as usize];
            let semantic = direct.catalog.value(slot)?;
            for component in 0..range.len {
                let physical = assigned_component(
                    &direct.layout,
                    semantic,
                    count_u32(component, "retained value component")?,
                )? as usize;
                let target = physical * stride;
                let source = (range.start + component) * points;
                for point in 0..points {
                    let value = values[source + point];
                    real[target + point] = value.re;
                    imaginary[target + point] = value.im;
                }
            }
        }
        for (slot, range) in direct.momentum_ranges.iter().copied().enumerate() {
            let semantic = direct.catalog.momentum(slot as u32)?;
            for component in 0..range.len {
                let physical = assigned_component(
                    &direct.layout,
                    semantic,
                    count_u32(component, "retained momentum component")?,
                )? as usize;
                let target = physical * stride;
                let source = (range.start + component) * points;
                for point in 0..points {
                    real[target + point] = momenta[source + point];
                    imaginary[target + point] = 0.0;
                }
            }
        }
        for (coupling_id, row) in direct.plan.couplings.iter().copied().enumerate() {
            let coupling = resolve_coupling(row, model_parameters);
            fill_real_input_plane(
                real,
                imaginary,
                stride,
                assigned_component(
                    &direct.layout,
                    direct.catalog.coupling_real(coupling_id as u32)?,
                    0,
                )?,
                points,
                coupling.re,
            );
            fill_real_input_plane(
                real,
                imaginary,
                stride,
                assigned_component(
                    &direct.layout,
                    direct.catalog.coupling_imag(coupling_id as u32)?,
                    0,
                )?,
                points,
                coupling.im,
            );
        }
        for (parameter_id, value) in model_parameters.iter().copied().enumerate() {
            let physical = assigned_component(
                &direct.layout,
                direct.catalog.parameter(parameter_id as u32)?,
                0,
            )? as usize;
            let target = physical * stride;
            real[target..target + points].fill(value.re);
            imaginary[target..target + points].fill(value.im);
        }
    }
    {
        let (factor_re, factor_im) = direct.workspace.factors_mut();
        for (index, spec) in direct.factor_specs.iter().copied().enumerate() {
            let coupling = resolve_coupling(
                direct.plan.couplings[spec.coupling_slot_id as usize],
                model_parameters,
            );
            let scale = match spec.output_factor_source {
                EAGER_OUTPUT_FACTOR_NONE => 1.0,
                EAGER_OUTPUT_FACTOR_COUPLING_REAL => coupling.re,
                EAGER_OUTPUT_FACTOR_COUPLING_IMAG => coupling.im,
                _ => {
                    return Err(RusticolError::integrity(
                        "retained eager factor has an invalid output-factor source",
                    ));
                }
            };
            factor_re[index] = spec.factor.re * scale;
            factor_im[index] = spec.factor.im * scale;
        }
    }
    Ok(())
}

#[cfg(test)]
fn measure_repeated(
    repetitions: usize,
    mut operation: impl FnMut() -> RusticolResult<()>,
) -> RusticolResult<f64> {
    use std::time::Instant;

    let started = Instant::now();
    for _ in 0..repetitions {
        operation()?;
    }
    Ok(started.elapsed().as_nanos() as f64 / repetitions as f64)
}

#[cfg(test)]
fn median_mad(values: &mut [f64]) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let median = values[values.len() / 2];
    let mut deviations = values
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    (median, deviations[deviations.len() / 2])
}

pub(super) fn derive_event_layout(
    sections: EagerPlanV3Sections<'_>,
    catalog: SemanticCatalog,
    plan: &EagerExecutionPlan,
) -> RusticolResult<DirectArenaLayout> {
    let count = catalog.semantic_count as usize;
    let mut widths = vec![1_u32; count];
    let mut first = vec![None::<u64>; count];
    let mut last = vec![None::<u64>; count];

    let mut produced_values = vec![false; sections.values.len()];
    for row in sections.finalizations {
        for value in [row.unpropagated_value_slot_id, row.propagated_value_slot_id] {
            if value != MISSING_U32 {
                produced_values[value as usize] = true;
            }
        }
    }
    for row in sections.values {
        widths[catalog.value(row.value_slot_id)? as usize] = row.component_count;
        if !produced_values[row.value_slot_id as usize] {
            define_semantic(&mut first, &mut last, catalog.value(row.value_slot_id)?, 0)?;
        }
    }
    for row in sections.currents {
        widths[catalog.current(row.current_id)? as usize] = row.component_count;
    }
    for row in sections.momenta {
        let id = catalog.momentum(row.momentum_slot_id)?;
        widths[id as usize] = row.component_count;
        define_semantic(&mut first, &mut last, id, 0)?;
    }
    for coupling in 0..sections.couplings.len() {
        define_semantic(
            &mut first,
            &mut last,
            catalog.coupling_real(coupling as u32)?,
            0,
        )?;
        define_semantic(
            &mut first,
            &mut last,
            catalog.coupling_imag(coupling as u32)?,
            0,
        )?;
    }
    for parameter in 0..sections.prepared_parameter_count {
        define_semantic(&mut first, &mut last, catalog.parameter(parameter)?, 0)?;
    }

    let mut event = 1_u64;
    for stage in &plan.stages {
        // The validated plan has already applied the stable equal-kernel sort
        // used by both packet and Direct-Arena execution. Liveness must follow
        // this execution order rather than the producer table order.
        for invocation in &stage.invocations {
            let row = invocation.row;
            use_semantic(
                &first,
                &mut last,
                catalog.value(row.left_value_slot_id)?,
                event,
            )?;
            use_semantic(
                &first,
                &mut last,
                catalog.value(row.right_value_slot_id)?,
                event,
            )?;
            use_semantic(
                &first,
                &mut last,
                catalog.momentum(row.left_momentum_slot_id)?,
                event,
            )?;
            use_semantic(
                &first,
                &mut last,
                catalog.momentum(row.right_momentum_slot_id)?,
                event,
            )?;
            use_semantic(
                &first,
                &mut last,
                catalog.coupling_real(row.coupling_slot_id)?,
                event,
            )?;
            use_semantic(
                &first,
                &mut last,
                catalog.coupling_imag(row.coupling_slot_id)?,
                event,
            )?;
            touch_kernel_parameters(
                plan.kernels.get(&row.kernel_id),
                catalog,
                &first,
                &mut last,
                event,
            )?;
            for attachment in &stage.attachments[invocation.attachment_range.clone()] {
                define_or_use_semantic(
                    &mut first,
                    &mut last,
                    catalog.current(attachment.row.result_current_id)?,
                    event,
                )?;
            }
            event = event
                .checked_add(1)
                .ok_or_else(|| invalid("eager direct event index overflows"))?;
        }

        // Packet execution performs every unpropagated copy before any
        // propagated finalization call. Sharing current/value planes therefore
        // requires distinct events in precisely that order.
        for copy in &stage.finalization_copies {
            let current_id = stage_current_id_for_layout(stage, copy.current)?;
            let value_id = plan
                .values
                .id_for_range(copy.unpropagated, "eager direct finalization copy")?;
            use_semantic(&first, &mut last, catalog.current(current_id)?, event)?;
            define_semantic(&mut first, &mut last, catalog.value(value_id)?, event)?;
            event = event
                .checked_add(1)
                .ok_or_else(|| invalid("eager direct event index overflows"))?;
        }
        for item in &stage.finalizations {
            let row = item.row;
            use_semantic(&first, &mut last, catalog.current(row.current_id)?, event)?;
            use_semantic(
                &first,
                &mut last,
                catalog.momentum(row.momentum_slot_id)?,
                event,
            )?;
            let propagated = item.propagated.ok_or_else(|| {
                RusticolError::internal("eager direct finalization lost propagated output")
            })?;
            let value_id = plan
                .values
                .id_for_range(propagated, "eager direct propagated finalization")?;
            define_semantic(&mut first, &mut last, catalog.value(value_id)?, event)?;
            touch_kernel_parameters(
                plan.kernels.get(&row.kernel_id),
                catalog,
                &first,
                &mut last,
                event,
            )?;
            event = event
                .checked_add(1)
                .ok_or_else(|| invalid("eager direct event index overflows"))?;
        }
    }
    for closure in &plan.closures {
        let row = closure.row;
        use_semantic(
            &first,
            &mut last,
            catalog.value(row.left_value_slot_id)?,
            event,
        )?;
        use_semantic(
            &first,
            &mut last,
            catalog.value(row.right_value_slot_id)?,
            event,
        )?;
        if row.coupling_slot_id != MISSING_U32 {
            use_semantic(
                &first,
                &mut last,
                catalog.coupling_real(row.coupling_slot_id)?,
                event,
            )?;
            use_semantic(
                &first,
                &mut last,
                catalog.coupling_imag(row.coupling_slot_id)?,
                event,
            )?;
        }
        touch_kernel_parameters(
            plan.kernels.get(&row.kernel_id),
            catalog,
            &first,
            &mut last,
            event,
        )?;
        event = event
            .checked_add(1)
            .ok_or_else(|| invalid("eager direct event index overflows"))?;
    }
    for closure in &plan.direct_closures {
        use_semantic(
            &first,
            &mut last,
            catalog.value(closure.row.left_value_slot_id)?,
            event,
        )?;
        use_semantic(
            &first,
            &mut last,
            catalog.value(closure.row.right_value_slot_id)?,
            event,
        )?;
        event = event
            .checked_add(1)
            .ok_or_else(|| invalid("eager direct event index overflows"))?;
    }
    let final_event = event.saturating_sub(1);
    for id in catalog.momentum_start..catalog.semantic_count {
        if first[id as usize].is_some() {
            last[id as usize] = Some(final_event);
        }
    }
    let intervals = widths
        .into_iter()
        .enumerate()
        .map(|(id, width)| {
            let first_use = first[id].unwrap_or(0);
            let last_use = last[id].unwrap_or(final_event).max(first_use);
            DirectArenaInterval::new(id as u32, first_use, last_use, width)
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    assign_direct_arena(&intervals)
}

fn stage_current_id_for_layout(
    stage: &super::plan::EagerStagePlan,
    range: ComponentRange,
) -> RusticolResult<u32> {
    stage
        .finalizations
        .iter()
        .find_map(|item| (item.current == range).then_some(item.row.current_id))
        .or_else(|| {
            stage
                .attachments
                .iter()
                .find_map(|item| (item.current == range).then_some(item.row.result_current_id))
        })
        .ok_or_else(|| RusticolError::internal("eager direct liveness lost a stage current"))
}

fn touch_kernel_parameters(
    kernel: Option<&super::EagerKernelSpec>,
    catalog: SemanticCatalog,
    first: &[Option<u64>],
    last: &mut [Option<u64>],
    event: u64,
) -> RusticolResult<()> {
    let Some(kernel) = kernel else {
        return Ok(());
    };
    for input in &kernel.inputs {
        if let EagerKernelInput::ModelParameter(parameter) = *input {
            use_semantic(first, last, catalog.parameter(parameter)?, event)?;
        }
    }
    Ok(())
}

fn encode_invocation_inputs(
    output: &mut Vec<u8>,
    invocation: &ScheduledInvocation,
    descriptors: &[EagerKernelInput],
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
) -> RusticolResult<()> {
    for descriptor in descriptors {
        let (semantic, component) = match *descriptor {
            EagerKernelInput::FirstCurrentComponent(component) => {
                (catalog.value(invocation.row.left_value_slot_id)?, component)
            }
            EagerKernelInput::SecondCurrentComponent(component) => (
                catalog.value(invocation.row.right_value_slot_id)?,
                component,
            ),
            EagerKernelInput::FirstMomentumComponent(component) => (
                catalog.momentum(invocation.row.left_momentum_slot_id)?,
                component,
            ),
            EagerKernelInput::SecondMomentumComponent(component) => (
                catalog.momentum(invocation.row.right_momentum_slot_id)?,
                component,
            ),
            EagerKernelInput::CouplingReal => {
                (catalog.coupling_real(invocation.row.coupling_slot_id)?, 0)
            }
            EagerKernelInput::CouplingImag => {
                (catalog.coupling_imag(invocation.row.coupling_slot_id)?, 0)
            }
            EagerKernelInput::ModelParameter(parameter) => (catalog.parameter(parameter)?, 0),
        };
        let physical = assigned_component(layout, semantic, component)?;
        push_u32(
            output,
            physical
                .checked_mul(2)
                .ok_or_else(|| invalid("eager direct plane catalog overflows"))?,
        );
        push_u32(
            output,
            physical
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| invalid("eager direct plane catalog overflows"))?,
        );
    }
    Ok(())
}

fn encode_attachment_destinations(
    output: &mut Vec<u8>,
    attachment: &ScheduledAttachment,
    output_component_count: u32,
    catalog: SemanticCatalog,
    layout: &DirectArenaLayout,
) -> RusticolResult<()> {
    if attachment.current.len != output_component_count as usize {
        return Err(RusticolError::integrity(
            "eager direct attachment width does not match its kernel output",
        ));
    }
    let semantic = catalog.current(attachment.row.result_current_id)?;
    for component in 0..output_component_count {
        let physical = assigned_component(layout, semantic, component)?;
        push_u32(
            output,
            physical
                .checked_mul(2)
                .ok_or_else(|| invalid("eager direct destination catalog overflows"))?,
        );
        push_u32(
            output,
            physical
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| invalid("eager direct destination catalog overflows"))?,
        );
    }
    Ok(())
}

pub(super) fn row_active(
    plan: &EagerExecutionPlan,
    domain: Option<u32>,
    active_groups: Option<&[u32]>,
) -> RusticolResult<bool> {
    let Some(active_groups) = active_groups else {
        return Ok(true);
    };
    // An explicitly empty runtime selector is a structural zero, including
    // for legacy/domain-less rows. It must never fall through to an
    // unconditional row.
    if active_groups.is_empty() {
        return Ok(false);
    }
    let Some(domain) = domain else {
        return Ok(true);
    };
    let selector = plan.selector_domains.as_ref().ok_or_else(|| {
        RusticolError::integrity("eager direct selector row has no selector-domain plan")
    })?;
    let members = selector.memberships.get(domain as usize).ok_or_else(|| {
        RusticolError::integrity("eager direct row references an unknown selector domain")
    })?;
    Ok(members
        .iter()
        .any(|member| active_groups.binary_search(member).is_ok()))
}

fn retained_attachment_operation(written_currents: &mut BTreeSet<u32>, current_id: u32) -> u32 {
    u32::from(!written_currents.insert(current_id))
}

pub(super) fn validate_active_groups(
    plan: &EagerExecutionPlan,
    active_groups: Option<&[u32]>,
) -> RusticolResult<()> {
    let Some(active_groups) = active_groups else {
        return Ok(());
    };
    if active_groups.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid(
            "eager direct active groups must be sorted and unique",
        ));
    }
    let selector = plan.selector_domains.as_ref().ok_or_else(|| {
        RusticolError::compatibility(
            "eager direct selector specialization requires selector domains",
        )
    })?;
    if let Some(unknown) = active_groups
        .iter()
        .find(|group| selector.group_ids.binary_search(group).is_err())
    {
        return Err(invalid(format!(
            "eager direct selector references unknown group {unknown}"
        )));
    }
    Ok(())
}

pub(super) fn validate_callable_identity(
    kernel: &EagerKernelSpec,
    prepared: &EagerDirectPreparedKernel<'_>,
) -> RusticolResult<()> {
    if prepared.role != kernel.role
        || prepared.inputs != kernel.inputs
        || prepared.output_component_count != kernel.output_component_count
    {
        return Err(RusticolError::integrity(format!(
            "eager direct callable identity for kernel {} does not match its plan spec: \
             role {:?}/{:?}, ordered inputs {:?}/{:?}, output width {}/{}",
            kernel.kernel_id,
            prepared.role,
            kernel.role,
            prepared.inputs,
            kernel.inputs,
            prepared.output_component_count,
            kernel.output_component_count,
        )));
    }
    Ok(())
}

pub(super) fn assigned_component(
    layout: &DirectArenaLayout,
    semantic: u32,
    component: u32,
) -> RusticolResult<u32> {
    let assignment = layout.assignment(semantic).ok_or_else(|| {
        RusticolError::integrity(format!(
            "eager direct layout lost semantic value {semantic}"
        ))
    })?;
    if component >= assignment.component_count {
        return Err(RusticolError::integrity(format!(
            "eager direct semantic value {semantic} component {component} is out of bounds"
        )));
    }
    assignment
        .component_base
        .checked_add(component)
        .ok_or_else(|| invalid("eager direct component assignment overflows"))
}

pub(super) fn resolve_coupling(
    row: crate::EagerCouplingRow,
    parameters: &[EagerComplex64],
) -> EagerComplex64 {
    let real = if row.real_parameter_id == MISSING_U32 {
        row.constant_real
    } else {
        parameters[row.real_parameter_id as usize].re
    };
    let imaginary = if row.imag_parameter_id == MISSING_U32 {
        row.constant_imag
    } else {
        parameters[row.imag_parameter_id as usize].re
    };
    EagerComplex64::new(real, imaginary)
}

fn fill_real_input_plane(
    real: &mut [f64],
    imaginary: &mut [f64],
    stride: usize,
    physical: u32,
    points: usize,
    value: f64,
) {
    let start = physical as usize * stride;
    real[start..start + points].fill(value);
    imaginary[start..start + points].fill(0.0);
}

fn define_semantic(
    first: &mut [Option<u64>],
    last: &mut [Option<u64>],
    id: u32,
    event: u64,
) -> RusticolResult<()> {
    let index = id as usize;
    if first[index].replace(event).is_some() {
        return Err(RusticolError::integrity(format!(
            "eager direct semantic value {id} is defined more than once"
        )));
    }
    last[index] = Some(event);
    Ok(())
}

fn define_or_use_semantic(
    first: &mut [Option<u64>],
    last: &mut [Option<u64>],
    id: u32,
    event: u64,
) -> RusticolResult<()> {
    let index = id as usize;
    if first[index].is_none() {
        first[index] = Some(event);
    }
    last[index] = Some(event);
    Ok(())
}

fn use_semantic(
    first: &[Option<u64>],
    last: &mut [Option<u64>],
    id: u32,
    event: u64,
) -> RusticolResult<()> {
    if first[id as usize].is_none() {
        return Err(RusticolError::integrity(format!(
            "eager direct semantic value {id} is used before definition"
        )));
    }
    last[id as usize] = Some(event);
    Ok(())
}

fn checked_semantic(start: u32, id: u32, stop: u32, label: &str) -> RusticolResult<u32> {
    start
        .checked_add(id)
        .filter(|value| *value < stop)
        .ok_or_else(|| invalid(format!("eager direct {label} id {id} is out of bounds")))
}

pub(super) fn validate_component_buffer(
    label: &str,
    actual: usize,
    components: usize,
    points: u32,
) -> RusticolResult<()> {
    let expected = components
        .checked_mul(points as usize)
        .ok_or_else(|| invalid(format!("eager direct {label} length overflows")))?;
    if actual != expected {
        return Err(invalid(format!(
            "eager direct {label} has length {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn row_range<'a, T>(rows: &'a [T], start: u64, count: u64, label: &str) -> RusticolResult<&'a [T]> {
    let start = usize::try_from(start)
        .map_err(|_| invalid(format!("eager direct {label} start exceeds usize")))?;
    let count = usize::try_from(count)
        .map_err(|_| invalid(format!("eager direct {label} count exceeds usize")))?;
    let stop = start
        .checked_add(count)
        .ok_or_else(|| invalid(format!("eager direct {label} range overflows")))?;
    rows.get(start..stop)
        .ok_or_else(|| invalid(format!("eager direct {label} range is out of bounds")))
}

pub(super) fn count_u32(value: usize, label: &str) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("eager direct {label} count exceeds u32")))
}

pub(super) fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

#[cfg(all(test, feature = "f64-symjit", target_arch = "aarch64"))]
mod tests {
    use std::hint::black_box;
    use std::time::Instant;

    use symjit::{Applet, Application, Compiler, Config, Expr};

    use super::super::plan_v3_tests::Fixture;
    use super::*;
    use crate::engine::compile_symbolica_program_to_plane_application_bytes;
    use crate::engine::count_allocations;
    use crate::engine::symjit_eager_direct::eager_direct_descriptor_for_source_application_bytes;

    fn source_application() -> Application {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let x = Expr::var("x");
        let y = Expr::var("y");
        let coupling = Expr::var("coupling");
        Compiler::with_config(config)
            .compile_params(&[], &[&x + &y], &[x, y, coupling])
            .unwrap()
    }

    fn ordinary_applet() -> Applet {
        let mut source = source_application();
        source.prepare_simd();
        source.seal().unwrap()
    }

    fn source_and_descriptor() -> (Vec<u8>, Vec<u8>) {
        let bytes = compile_symbolica_program_to_plane_application_bytes(
            "([('add', ('temp', 0), [('param', 0), ('param', 1)], 0), \
              ('assign', ('out', 0), ('temp', 0))], 3, [])",
            3,
            1,
            2,
            true,
        )
        .unwrap();
        let descriptor = eager_direct_descriptor_for_source_application_bytes(
            &bytes,
            3,
            1,
            std::path::Path::new("eager-invocation-test.symjit"),
        )
        .unwrap();
        (bytes, descriptor)
    }

    fn fanout_fixture() -> Fixture {
        let mut fixture = Fixture::new();
        fixture.currents.push(crate::EagerPlanCurrentRow {
            current_id: 3,
            component_start: 3,
            component_count: 1,
            momentum_slot_id: 3,
            flags: 0,
        });
        fixture.values.push(crate::EagerPlanValueRow {
            value_slot_id: 3,
            current_id: 3,
            component_start: 3,
            component_count: 1,
            kind: crate::EagerValueSlotKind::Unpropagated,
        });
        fixture.momenta.push(crate::EagerPlanMomentumRow {
            momentum_slot_id: 3,
            bitset_id: 3,
            component_start: 12,
            component_count: 4,
        });
        let mut second_attachment = fixture.attachments[0];
        second_attachment.result_current_id = 3;
        fixture.attachments.push(second_attachment);
        fixture.invocations[0].attachment_count = 2;
        fixture.stages[0].attachment_count = 2;
        fixture.stages[0].finalization_count = 2;
        let mut second_finalization = fixture.finalizations[0];
        second_finalization.current_id = 3;
        second_finalization.unpropagated_value_slot_id = 3;
        second_finalization.momentum_slot_id = 3;
        fixture.finalizations.push(second_finalization);
        let mut second_closure = fixture.closures[0];
        second_closure.root_id = 1;
        second_closure.left_value_slot_id = 3;
        fixture.closures.push(second_closure);
        fixture
    }

    fn inputs(points: usize) -> (Vec<EagerComplex64>, Vec<f64>) {
        let mut values = vec![EagerComplex64::new(-91.0, 37.0); 4 * points];
        for point in 0..points {
            values[point] = EagerComplex64::new(1.0 + point as f64 / 8.0, -0.25 * point as f64);
            values[points + point] =
                EagerComplex64::new(3.0 + point as f64 / 4.0, 0.5 * point as f64);
        }
        let momenta = (0..16 * points).map(|index| index as f64 / 16.0).collect();
        (values, momenta)
    }

    struct PacketOracle {
        applet: Applet,
        inputs: Vec<EagerComplex64>,
        outputs: Vec<EagerComplex64>,
        currents: Vec<EagerComplex64>,
    }

    impl PacketOracle {
        fn new(capacity: usize) -> Self {
            Self {
                applet: ordinary_applet(),
                inputs: vec![EagerComplex64::new(0.0, 0.0); 3 * capacity],
                outputs: vec![EagerComplex64::new(0.0, 0.0); capacity],
                currents: vec![EagerComplex64::new(0.0, 0.0); capacity],
            }
        }

        fn evaluate(&mut self, points: usize, values: &[EagerComplex64]) {
            for point in 0..points {
                self.inputs[3 * point] = values[point];
                self.inputs[3 * point + 1] = values[points + point];
                self.inputs[3 * point + 2] = EagerComplex64::new(2.0, 3.0);
            }
            self.applet.evaluate_matrix(
                &self.inputs[..3 * points],
                &mut self.outputs[..points],
                points,
            );
            for point in 0..points {
                let contribution = self.outputs[point] * EagerComplex64::new(15.0, 0.0);
                self.currents[point] = contribution;
            }
        }
    }

    fn prototype<'a>(
        fixture: &Fixture,
        source: &'a [u8],
        descriptor: &'a [u8],
        active_groups: Option<&[u32]>,
    ) -> EagerDirectInvocationPrototype {
        let inputs = [
            EagerKernelInput::FirstCurrentComponent(0),
            EagerKernelInput::SecondCurrentComponent(0),
            EagerKernelInput::CouplingReal,
        ];
        let artifact = EagerDirectPreparedKernel {
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
            display_path: PathBuf::from("real-plan-v3-eager-k10.symjit"),
        };
        EagerDirectInvocationPrototype::from_plan_v3_sections(
            fixture.sections(),
            &[artifact],
            active_groups,
            129,
        )
        .unwrap()
    }

    #[test]
    fn real_plan_v3_rows_match_packet_oracle_for_fanout_and_tails() {
        let fixture = fanout_fixture();
        let (source, descriptor) = source_and_descriptor();
        let mut direct = prototype(&fixture, &source, &descriptor, None);
        assert!(
            direct.layout().reused_semantic_components() > 0,
            "event-derived layout should reuse at least one dead semantic range"
        );
        let mut packet = PacketOracle::new(129);
        for points in [7_usize, 127, 128, 129] {
            let (values, momenta) = inputs(points);
            direct
                .evaluate_invocations(points as u32, &values, &momenta, &[])
                .unwrap();
            packet.evaluate(points, &values);
            for (point, expected) in packet.currents[..points].iter().copied().enumerate() {
                let actual = direct.current_value(2, 0, point as u32).unwrap();
                assert_eq!(
                    (actual.re.to_bits(), actual.im.to_bits()),
                    (expected.re.to_bits(), expected.im.to_bits()),
                    "point {point} at tail {points}"
                );
                let second = direct.current_value(3, 0, point as u32).unwrap();
                assert_eq!(
                    (second.re.to_bits(), second.im.to_bits()),
                    (expected.re.to_bits(), expected.im.to_bits()),
                    "second fanout destination at point {point} tail {points}"
                );
            }
        }
        let traffic = direct.traffic();
        assert!(traffic.calls >= 4);
        assert!(traffic.rows >= 4);
        assert_eq!(
            (
                traffic.packet_input_bytes,
                traffic.packet_output_bytes,
                traffic.gather_bytes,
                traffic.scatter_bytes,
                traffic.remap_bytes,
            ),
            (0, 0, 0, 0, 0)
        );
    }

    #[test]
    fn empty_runtime_selector_is_a_structural_zero_without_a_table_call() {
        let fixture = fanout_fixture();
        let mut direct = EagerDirectInvocationPrototype::from_plan_v3_sections(
            fixture.sections(),
            &[],
            Some(&[]),
            129,
        )
        .unwrap();
        let (values, momenta) = inputs(129);
        direct
            .evaluate_invocations(129, &values, &momenta, &[])
            .unwrap();
        for point in 0..129 {
            let actual = direct.current_value(2, 0, point).unwrap();
            assert_eq!((actual.re.to_bits(), actual.im.to_bits()), (0, 0));
            let second = direct.current_value(3, 0, point).unwrap();
            assert_eq!((second.re.to_bits(), second.im.to_bits()), (0, 0));
        }
        assert_eq!(direct.traffic().calls, 0);
        direct.traffic().validate_direct().unwrap();
    }

    #[test]
    fn empty_runtime_selector_prunes_domainless_rows_too() {
        let fixture = fanout_fixture();
        let plan = EagerExecutionPlan::from_plan_v3_sections(fixture.sections()).unwrap();
        assert!(!row_active(&plan, None, Some(&[])).unwrap());
        assert!(row_active(&plan, None, Some(&[7])).unwrap());
    }

    #[test]
    fn selector_pruning_recomputes_first_retained_current_write() {
        // The full schedule marked the first row as the initializer. Runtime
        // selection prunes it and retains the second row, whose stale
        // full-plan flag was "add".
        let original_initializes = [true, false];
        let retained = [false, true];
        let mut written = BTreeSet::new();
        let operations = original_initializes
            .into_iter()
            .zip(retained)
            .filter(|&(_original, keep)| keep)
            .map(|_| retained_attachment_operation(&mut written, 23))
            .collect::<Vec<_>>();
        assert_eq!(operations, [0], "first retained write must overwrite");

        // Lock the bitwise reason this cannot reuse the stale flag: assigning
        // -0 preserves its sign, while adding -0 to a cleared +0 loses it.
        let contribution = -0.0_f64;
        let mut recomputed = 0.0_f64;
        if operations[0] == 0 {
            recomputed = contribution;
        } else {
            recomputed += contribution;
        }
        let mut stale = 0.0_f64;
        stale += contribution;
        assert_eq!(recomputed.to_bits(), contribution.to_bits());
        assert_ne!(stale.to_bits(), contribution.to_bits());
    }

    #[test]
    fn matching_dimension_callable_with_wrong_semantic_identity_is_rejected() {
        let fixture = fanout_fixture();
        let (source, descriptor) = source_and_descriptor();
        let wrong_inputs = [
            EagerKernelInput::SecondCurrentComponent(0),
            EagerKernelInput::FirstCurrentComponent(0),
            EagerKernelInput::CouplingReal,
        ];
        let artifact = EagerDirectPreparedKernel {
            kernel_id: 10,
            role: EagerKernelRole::Vertex,
            inputs: &wrong_inputs,
            output_component_count: 1,
            application: EagerDirectPreparedApplication::Symjit {
                source_application: &source,
                descriptor: &descriptor,
                expected_optimization_level: 2,
                expected_compression: true,
            },
            display_path: PathBuf::from("wrong-semantic-identity.symjit"),
        };
        let result = EagerDirectInvocationPrototype::from_plan_v3_sections(
            fixture.sections(),
            &[artifact],
            None,
            129,
        );
        let Err(error) = result else {
            panic!("matching dimensions must not authenticate the wrong callable");
        };
        assert!(error.to_string().contains("callable identity"));
        assert!(error.to_string().contains("ordered inputs"));
    }

    #[test]
    fn warmed_real_plan_v3_invocation_slice_allocates_zero() {
        let fixture = fanout_fixture();
        let (source, descriptor) = source_and_descriptor();
        let mut direct = prototype(&fixture, &source, &descriptor, None);
        let (values, momenta) = inputs(129);
        direct
            .evaluate_invocations(129, &values, &momenta, &[])
            .unwrap();
        let (result, allocations, bytes) =
            count_allocations(|| direct.evaluate_invocations(129, &values, &momenta, &[]));
        result.unwrap();
        assert_eq!((allocations, bytes), (0, 0));
    }

    #[test]
    #[ignore = "local interleaved native timing evidence; no timing assertion"]
    fn benchmark_real_plan_v3_direct_slice_against_packet_execution() {
        const SAMPLES: usize = 9;
        const REPETITIONS: usize = 10_000;
        const POINTS: usize = 129;

        let fixture = fanout_fixture();
        let (source, descriptor) = source_and_descriptor();
        let mut direct = prototype(&fixture, &source, &descriptor, None);
        let (values, momenta) = inputs(POINTS);
        let mut packet = PacketOracle::new(POINTS);
        direct
            .evaluate_invocations(POINTS as u32, &values, &momenta, &[])
            .unwrap();
        packet.evaluate(POINTS, &values);

        let mut direct_ns = Vec::with_capacity(SAMPLES);
        let mut packet_ns = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            if sample.is_multiple_of(2) {
                direct_ns.push(measure_direct(&mut direct, REPETITIONS));
                packet_ns.push(measure_packet(&mut packet, &values, REPETITIONS));
            } else {
                packet_ns.push(measure_packet(&mut packet, &values, REPETITIONS));
                direct_ns.push(measure_direct(&mut direct, REPETITIONS));
            }
        }
        direct_ns.sort_by(f64::total_cmp);
        packet_ns.sort_by(f64::total_cmp);
        let direct_median = direct_ns[SAMPLES / 2];
        let packet_median = packet_ns[SAMPLES / 2];
        eprintln!(
            "eager real-plan-v3 invocation benchmark: samples={SAMPLES} \
             repetitions={REPETITIONS} points={POINTS} \
             direct_median_ns/call={direct_median:.3} \
             packet_median_ns/call={packet_median:.3} \
             direct_over_packet={:.6}",
            direct_median / packet_median
        );
    }

    fn measure_direct(direct: &mut EagerDirectInvocationPrototype, repetitions: usize) -> f64 {
        let started = Instant::now();
        for _ in 0..repetitions {
            direct.execute_preinitialized_invocations(129).unwrap();
        }
        black_box(direct.current_value(2, 0, 0).unwrap());
        started.elapsed().as_nanos() as f64 / repetitions as f64
    }

    fn measure_packet(
        packet: &mut PacketOracle,
        values: &[EagerComplex64],
        repetitions: usize,
    ) -> f64 {
        let started = Instant::now();
        for _ in 0..repetitions {
            packet.evaluate(129, values);
        }
        black_box(packet.currents[0]);
        started.elapsed().as_nanos() as f64 / repetitions as f64
    }
}
