// SPDX-License-Identifier: 0BSD

#![allow(dead_code)] // Query-local compatibility path; production uses persisted families.

use super::*;
use crate::recurrence::{
    DirectResolvedSourceSelection, DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow,
};

/// Borrowed authenticated tables used to cold-bind one selected-helicity
/// source program.
///
/// The direct plan remains the serialized authority.  A process-owned source
/// resolver consumes this view once when a persisted helicity family is
/// prepared and returns compact executable rows; no table from this view is
/// retained by the warm program.
#[derive(Clone, Copy)]
pub(crate) struct OnTheFlyUnionSourceBindingView<'a> {
    pub(crate) sources: &'a [DirectSourceRow],
    pub(crate) variants: &'a [DirectSourceDispatchVariantDescriptor],
    pub(crate) embeddings: &'a [DirectSourceEmbeddingRow],
    pub(crate) selections: &'a [DirectResolvedSourceSelection],
    pub(crate) exact_factors: &'a [ExactComplexRational],
    pub(crate) current_component_count: u32,
    pub(crate) momentum_form_count: u32,
    pub(crate) parameter_count: u32,
}

/// Opaque, family-local source program produced by one cold authenticated
/// binding.  Implementations own every template and embedding descriptor they
/// need, so execution neither borrows the direct plan nor searches a source
/// domain.
pub(crate) trait OnTheFlyBoundUnionSourceProgram {
    /// Execute against views already authenticated by `validate_direct_views`.
    ///
    /// # Safety
    ///
    /// The caller must provide live, nonaliasing Direct-Arena views whose
    /// declared shapes match the plan used for cold binding.
    unsafe fn execute(
        &self,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        point_count: u32,
    ) -> RusticolResult<()>;
}

/// Resolved prepared executor borrowed from a lane-owned pool/binding.
#[derive(Clone, Copy)]
pub(crate) struct ResolvedOnTheFlyExecutor {
    pub(crate) direct_executor_id: u32,
    pub(crate) handle: DirectExecutorHandle,
    pub(crate) parent_permutation: [u8; 2],
    /// Exact process-bound proof that this prepared callback accepts a
    /// singleton arena whose physical point stride is one. Query families
    /// use the packed layout only when every resolved row-group member
    /// carries this capability.
    pub(crate) packed_singleton_capable: bool,
    /// Authenticated intrinsic-only native contribution capability used by
    /// fanout and interaction rewrites. SymJIT, compiled-native, source,
    /// finalization, and closure bindings keep None.
    pub(crate) interaction_capability: Option<DirectInteractionExecutorCapability>,
}

/// Semantic resolver implemented by the lane-owned prepared executor pool.
pub(crate) trait OnTheFlyPreparedExecutorResolver {
    fn resolve(&self, key: OnTheFlyExecutorKeyV1) -> RusticolResult<ResolvedOnTheFlyExecutor>;

    /// Resolve an executor already authenticated by a persisted direct plan.
    /// Query-local traces use [`Self::resolve`]; persisted helicity families
    /// need no second semantic-key catalog because the plan layout owns the
    /// dense executor IDs and row parent order.
    fn resolve_direct_executor(
        &self,
        _direct_executor_id: u32,
        _role: DirectExecutorRole,
    ) -> RusticolResult<ResolvedOnTheFlyExecutor> {
        Err(integrity(
            "prepared executor resolver does not support persisted direct-plan IDs",
        ))
    }

    /// Runtime-helicity source dispatcher owned beside the prepared pool.
    fn union_source_dispatch(&self) -> RusticolResult<DirectUnionSourceDispatchHandle> {
        Err(integrity(
            "prepared executor resolver has no all-flow-union source dispatcher",
        ))
    }

    /// Cold-bind one selected all-flow source schedule.  The default keeps
    /// query-only resolvers deliberately incapable of executing a persisted
    /// selector family.
    fn bind_union_source_program(
        &self,
        _view: OnTheFlyUnionSourceBindingView<'_>,
    ) -> RusticolResult<Box<dyn OnTheFlyBoundUnionSourceProgram>> {
        Err(integrity(
            "prepared executor resolver cannot cold-bind all-flow-union sources",
        ))
    }

    fn union_source_packed_singleton_capable(&self) -> bool {
        false
    }

    /// Invalidate any prepared-context caches that borrow row or arena pointers.
    /// Native/Rust callbacks retain no such pointers and need no work here.
    fn invalidate_row_tables(&self) -> RusticolResult<()>;
}

fn checked_scalar_len(planes: u32, stride: u32, label: &str) -> RusticolResult<usize> {
    usize::try_from(planes)
        .ok()
        .and_then(|planes| planes.checked_mul(stride as usize))
        .ok_or_else(|| invalid(format!("{label} scalar length exceeds usize")))
}

fn exact_factor_parts(value: ExactComplexRational) -> RusticolResult<(f64, f64)> {
    let real = value.real().numerator() as f64 / value.real().denominator() as f64;
    let imag = value.imag().numerator() as f64 / value.imag().denominator() as f64;
    if !real.is_finite() || !imag.is_finite() {
        return Err(invalid(
            "exact factor cannot be represented as finite binary64",
        ));
    }
    Ok((real, imag))
}

/// Owned aligned numeric state for one on-the-fly lane.  Structural traces do
/// not own this storage and parameter mutation never rebuilds them.
pub(crate) struct OnTheFlyWorkspaceV1 {
    current_re: AlignedF64Buffer,
    current_im: AlignedF64Buffer,
    amplitude_re: AlignedF64Buffer,
    amplitude_im: AlignedF64Buffer,
    momenta: AlignedF64Buffer,
    parameters_re: AlignedF64Buffer,
    parameters_im: AlignedF64Buffer,
    factors_re: AlignedF64Buffer,
    factors_im: AlignedF64Buffer,
    trace_digest: SemanticDigest,
    source_count: u32,
    logical_point_capacity: u32,
    active_point_count: u32,
    momentum_form_count: u32,
    lorentz_component_count: u16,
    point_stride: u32,
}

impl OnTheFlyWorkspaceV1 {
    pub(crate) fn new(
        trace: &OnTheFlyStructuralTraceV1,
        logical_point_capacity: u32,
    ) -> RusticolResult<Self> {
        let layout = trace.layout;
        if layout.source_count == 0
            || logical_point_capacity == 0
            || layout.lorentz_component_count == 0
        {
            return Err(integrity(
                "trace has an empty authenticated workspace shape",
            ));
        }
        if trace.layout.momentum_form_count == 0
            || trace.layout.exact_factor_count as usize != trace.exact_factors.len()
        {
            return Err(integrity("trace workspace layout is inconsistent"));
        }
        let point_stride = checked_aligned_point_stride(logical_point_capacity)?;
        let current_len = checked_scalar_len(
            trace.layout.current_component_count,
            point_stride,
            "current arena",
        )?;
        let amplitude_len = checked_scalar_len(
            trace.layout.amplitude_component_count,
            point_stride,
            "amplitude arena",
        )?;
        let momentum_planes = trace
            .layout
            .momentum_form_count
            .checked_mul(u32::from(layout.lorentz_component_count))
            .ok_or_else(|| invalid("momentum plane count exceeds u32"))?;
        let momentum_len = checked_scalar_len(momentum_planes, point_stride, "momentum arena")?;
        let factor_count = trace.exact_factors.len();
        let mut factors_re = AlignedF64Buffer::zeroed(factor_count, "on-the-fly factor real")?;
        let mut factors_im = AlignedF64Buffer::zeroed(factor_count, "on-the-fly factor imaginary")?;
        for (index, factor) in trace.exact_factors.iter().copied().enumerate() {
            let (real, imag) = exact_factor_parts(factor)?;
            factors_re.as_mut_slice()[index] = real;
            factors_im.as_mut_slice()[index] = imag;
        }
        Ok(Self {
            current_re: AlignedF64Buffer::zeroed(current_len, "on-the-fly current real")?,
            current_im: AlignedF64Buffer::zeroed(current_len, "on-the-fly current imaginary")?,
            amplitude_re: AlignedF64Buffer::zeroed(amplitude_len, "on-the-fly amplitude real")?,
            amplitude_im: AlignedF64Buffer::zeroed(
                amplitude_len,
                "on-the-fly amplitude imaginary",
            )?,
            momenta: AlignedF64Buffer::zeroed(momentum_len, "on-the-fly momenta")?,
            parameters_re: AlignedF64Buffer::zeroed(
                usize::try_from(layout.parameter_count)
                    .map_err(|_| invalid("parameter count exceeds usize"))?,
                "on-the-fly parameter real",
            )?,
            parameters_im: AlignedF64Buffer::zeroed(
                usize::try_from(layout.parameter_count)
                    .map_err(|_| invalid("parameter count exceeds usize"))?,
                "on-the-fly parameter imaginary",
            )?,
            factors_re,
            factors_im,
            trace_digest: trace.semantic_digest(),
            source_count: layout.source_count,
            logical_point_capacity,
            active_point_count: 0,
            momentum_form_count: trace.layout.momentum_form_count,
            lorentz_component_count: layout.lorentz_component_count,
            point_stride,
        })
    }

    pub(crate) const fn point_stride(&self) -> u32 {
        self.point_stride
    }

    pub(crate) fn set_parameter(
        &mut self,
        parameter_id: u32,
        real: f64,
        imag: f64,
    ) -> RusticolResult<()> {
        self.active_point_count = 0;
        if !real.is_finite() || !imag.is_finite() {
            return Err(invalid("runtime parameter value must be finite"));
        }
        let index = parameter_id as usize;
        *self
            .parameters_re
            .as_mut_slice()
            .get_mut(index)
            .ok_or_else(|| invalid("runtime parameter ID is out of bounds"))? = real;
        *self
            .parameters_im
            .as_mut_slice()
            .get_mut(index)
            .ok_or_else(|| invalid("runtime parameter ID is out of bounds"))? = imag;
        Ok(())
    }

    pub(crate) fn set_momentum_value(
        &mut self,
        momentum_form_id: u32,
        lorentz_component: u16,
        point_index: u32,
        value: f64,
    ) -> RusticolResult<()> {
        self.active_point_count = 0;
        if !value.is_finite()
            || momentum_form_id >= self.momentum_form_count
            || lorentz_component >= self.lorentz_component_count
            || point_index >= self.logical_point_capacity
        {
            return Err(invalid("momentum workspace coordinate or value is invalid"));
        }
        let plane = usize::try_from(momentum_form_id)
            .ok()
            .and_then(|form| form.checked_mul(usize::from(self.lorentz_component_count)))
            .and_then(|base| base.checked_add(usize::from(lorentz_component)))
            .ok_or_else(|| invalid("momentum plane index exceeds usize"))?;
        let index = plane
            .checked_mul(self.point_stride as usize)
            .and_then(|base| base.checked_add(point_index as usize))
            .ok_or_else(|| invalid("momentum scalar index exceeds usize"))?;
        self.momenta.as_mut_slice()[index] = value;
        Ok(())
    }

    pub(crate) fn fill_momenta_from_external(
        &mut self,
        trace: &OnTheFlyStructuralTraceV1,
        external_momenta: &[f64],
        point_count: u32,
    ) -> RusticolResult<()> {
        self.active_point_count = 0;
        if self.trace_digest != trace.semantic_digest() {
            return Err(integrity(
                "workspace belongs to a different structural trace",
            ));
        }
        if point_count == 0 || point_count > self.logical_point_capacity {
            return Err(invalid("active point count is outside the workspace"));
        }
        let expected = usize::try_from(self.source_count)
            .ok()
            .and_then(|sources| sources.checked_mul(usize::from(self.lorentz_component_count)))
            .and_then(|planes| planes.checked_mul(point_count as usize))
            .ok_or_else(|| invalid("external momentum shape exceeds usize"))?;
        if external_momenta.len() != expected {
            return Err(invalid(format!(
                "external momentum input has {} scalars, expected {expected}",
                external_momenta.len()
            )));
        }
        for (form_id, form) in trace.momentum_forms.iter().enumerate() {
            for lorentz in 0..usize::from(self.lorentz_component_count) {
                for point in 0..point_count as usize {
                    let mut value = 0.0;
                    for term in form.terms() {
                        if term.source_slot >= self.source_count {
                            return Err(integrity("momentum form source slot is out of bounds"));
                        }
                        let input_index = (term.source_slot as usize
                            * usize::from(self.lorentz_component_count)
                            + lorentz)
                            * point_count as usize
                            + point;
                        value += f64::from(term.coefficient) * external_momenta[input_index];
                    }
                    self.set_momentum_value(form_id as u32, lorentz as u16, point as u32, value)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn amplitude(&self, point_index: u32) -> RusticolResult<(f64, f64)> {
        if point_index >= self.active_point_count {
            return Err(invalid(
                "amplitude point is outside the last successful execution",
            ));
        }
        let index = point_index as usize;
        Ok((
            self.amplitude_re.as_slice()[index],
            self.amplitude_im.as_slice()[index],
        ))
    }

    fn clear_active(&mut self, point_count: u32) -> RusticolResult<()> {
        if point_count == 0 || point_count > self.logical_point_capacity {
            return Err(invalid("active point count is outside the workspace"));
        }
        for (real, imag) in [
            (&mut self.current_re, &mut self.current_im),
            (&mut self.amplitude_re, &mut self.amplitude_im),
        ] {
            for plane in 0..real.len() / self.point_stride as usize {
                let start = plane * self.point_stride as usize;
                let end = start + point_count as usize;
                real.as_mut_slice()[start..end].fill(0.0);
                imag.as_mut_slice()[start..end].fill(0.0);
            }
        }
        Ok(())
    }

    fn raw_views(
        &mut self,
    ) -> RusticolResult<(
        DirectArenaView,
        DirectMomentumView,
        DirectParameterView,
        DirectFactorView,
    )> {
        let arena = DirectArenaView {
            current_re: self.current_re.as_mut_ptr(),
            current_im: self.current_im.as_mut_ptr(),
            current_scalar_len: self.current_re.len() as u64,
            amplitude_re: self.amplitude_re.as_mut_ptr(),
            amplitude_im: self.amplitude_im.as_mut_ptr(),
            amplitude_scalar_len: self.amplitude_re.len() as u64,
            point_stride: self.point_stride,
        };
        let momenta = DirectMomentumView {
            values: self.momenta.as_ptr(),
            scalar_len: self.momenta.len() as u64,
            form_count: self.momentum_form_count,
            lorentz_component_count: self.lorentz_component_count,
            point_stride: self.point_stride,
        };
        let parameters = DirectParameterView {
            values_re: self.parameters_re.as_ptr(),
            values_im: self.parameters_im.as_ptr(),
            value_count: self.parameters_re.len() as u32,
        };
        let factors = DirectFactorView {
            values_re: self.factors_re.as_ptr(),
            values_im: self.factors_im.as_ptr(),
            value_count: self.factors_re.len() as u32,
        };
        validate_direct_views(arena, momenta, parameters, factors)?;
        Ok((arena, momenta, parameters, factors))
    }
}

fn validate_resolved_executor(
    key: OnTheFlyExecutorKeyV1,
    resolved: ResolvedOnTheFlyExecutor,
) -> RusticolResult<()> {
    if resolved.handle.role() != key.role() {
        return Err(integrity(
            "resolved executor role differs from its semantic key",
        ));
    }
    match key.role() {
        DirectExecutorRole::Contribution => match resolved.parent_permutation {
            [0, 1] | [1, 0] => {}
            _ => {
                return Err(integrity(
                    "resolved binary executor has an invalid permutation",
                ));
            }
        },
        DirectExecutorRole::Source
        | DirectExecutorRole::Finalization
        | DirectExecutorRole::Closure => {
            if resolved.parent_permutation != [0, 1] {
                return Err(integrity(
                    "unary executor unexpectedly permutes parent rows",
                ));
            }
        }
    }
    Ok(())
}

/// Direct structural interpreter.  It resolves every operation by semantic
/// identity and invokes genuine prepared Direct-Arena kernels without a plan.
pub(crate) struct OnTheFlyStructuralInterpreter;

impl OnTheFlyStructuralInterpreter {
    pub(crate) fn execute(
        trace: &OnTheFlyStructuralTraceV1,
        resolver: &impl OnTheFlyPreparedExecutorResolver,
        workspace: &mut OnTheFlyWorkspaceV1,
        point_count: u32,
    ) -> RusticolResult<()> {
        if !trace.prepared_executor_rows_bound() {
            return Err(integrity(
                "prepared executor parent order is not bound into stable trace rows",
            ));
        }
        if workspace.trace_digest != trace.semantic_digest() {
            return Err(integrity(
                "workspace belongs to a different structural trace",
            ));
        }
        workspace.active_point_count = 0;
        workspace.clear_active(point_count)?;
        let (arena, momenta, parameters, factors) = workspace.raw_views()?;
        for operation in trace.operations.iter() {
            let key = operation.key();
            let resolved = resolver.resolve(key)?;
            validate_resolved_executor(key, resolved)?;
            clear_direct_executor_error_detail();
            let status: c_int = unsafe {
                match (operation, resolved.handle) {
                    (
                        OnTheFlyTraceOperationV1::Source { row, .. },
                        DirectExecutorHandle::Source { call, context },
                    ) => call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        row,
                        1,
                        point_count,
                    ),
                    (
                        OnTheFlyTraceOperationV1::Contribution { row, .. },
                        DirectExecutorHandle::Contribution { call, context },
                    ) => call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        row,
                        1,
                        point_count,
                    ),
                    (
                        OnTheFlyTraceOperationV1::Finalization { row, .. },
                        DirectExecutorHandle::Finalization { call, context },
                    ) => call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        row,
                        1,
                        point_count,
                    ),
                    (
                        OnTheFlyTraceOperationV1::Closure { row, .. },
                        DirectExecutorHandle::Closure { call, context },
                    ) => call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        row,
                        1,
                        point_count,
                    ),
                    _ => {
                        return Err(integrity(
                            "resolved executor handle variant differs from trace operation",
                        ));
                    }
                }
            };
            crate::recurrence::direct_backend::check_status(
                key.role(),
                resolved.direct_executor_id,
                status,
            )?;
        }
        workspace.active_point_count = point_count;
        Ok(())
    }
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
impl OnTheFlyWorkspaceV1 {
    pub(crate) fn observed_current_components(
        &self,
        trace: &OnTheFlyStructuralTraceV1,
        current_id: u32,
        point_index: u32,
    ) -> RusticolResult<Vec<(f64, f64)>> {
        if self.trace_digest != trace.semantic_digest() || point_index >= self.active_point_count {
            return Err(invalid(
                "current observation does not match the last successful trace execution",
            ));
        }
        let [component_base, component_count] = trace.current_component_range(current_id)?;
        (0..component_count)
            .map(|component_offset| {
                let plane = component_base
                    .checked_add(component_offset)
                    .ok_or_else(|| integrity("current observation plane overflows"))?;
                let index = usize::try_from(plane)
                    .ok()
                    .and_then(|plane| plane.checked_mul(self.point_stride as usize))
                    .and_then(|base| base.checked_add(point_index as usize))
                    .ok_or_else(|| integrity("current observation index exceeds usize"))?;
                Ok((
                    *self
                        .current_re
                        .as_slice()
                        .get(index)
                        .ok_or_else(|| integrity("current observation real value is absent"))?,
                    *self.current_im.as_slice().get(index).ok_or_else(|| {
                        integrity("current observation imaginary value is absent")
                    })?,
                ))
            })
            .collect()
    }
}
