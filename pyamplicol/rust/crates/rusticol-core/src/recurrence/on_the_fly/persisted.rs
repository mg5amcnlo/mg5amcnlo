// SPDX-License-Identifier: 0BSD

//! Cold-bound helicity families projected from one persisted all-flow plan.
//!
//! The sidecar's CSR is consulted only by [`PersistedHelicityFamilyExecutorV1::prepare`].
//! Each retained helicity owns compact boxed non-source rows with authoritative
//! initialization flags and the same schedule-local singleton fanout and
//! interaction programs used by query-built families. Warm execution invokes
//! one cold-bound exact union-source program followed by those rows; it
//! performs no support-mask scan, selector branch, or source-catalog lookup.

use super::*;
use crate::direct_arena::{
    AlignedF64Buffer, DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
    checked_aligned_point_stride, validate_direct_views,
};
#[cfg(test)]
use crate::recurrence::direct_backend::DIRECT_STATUS_OK;
use crate::recurrence::direct_backend::{
    DirectWorkspace, check_status, execute_certified_reuse_rows,
};
use crate::recurrence::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_NONE_U32, DirectHelicityDispatch,
    DirectRecurrencePlan, DirectResolvedHelicityDescriptor, DirectResolvedSourceSelection,
    DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow,
};

#[derive(Debug, Eq, PartialEq)]
enum PersistedFamilyRowsV1 {
    Contribution(Box<[DirectContributionRow]>),
    Finalization(Box<[DirectFinalizationRow]>),
    Closure(Box<[DirectClosureRow]>),
}

impl PersistedFamilyRowsV1 {
    fn len(&self) -> usize {
        match self {
            Self::Contribution(rows) => rows.len(),
            Self::Finalization(rows) => rows.len(),
            Self::Closure(rows) => rows.len(),
        }
    }
}

enum PersistedFamilyRowsDraftV1 {
    Contribution(Vec<DirectContributionRow>),
    Finalization(Vec<DirectFinalizationRow>),
    Closure(Vec<DirectClosureRow>),
}

impl PersistedFamilyRowsDraftV1 {
    const fn role(&self) -> DirectExecutorRole {
        match self {
            Self::Contribution(_) => DirectExecutorRole::Contribution,
            Self::Finalization(_) => DirectExecutorRole::Finalization,
            Self::Closure(_) => DirectExecutorRole::Closure,
        }
    }

    fn append_from_plan(
        &mut self,
        plan: &DirectRecurrencePlan,
        role: DirectExecutorRole,
        row_start: u64,
        row_count: u32,
    ) -> RusticolResult<()> {
        let start = usize::try_from(row_start)
            .map_err(|_| integrity("persisted family row start exceeds usize"))?;
        let end = start
            .checked_add(row_count as usize)
            .ok_or_else(|| integrity("persisted family row range overflows usize"))?;
        match (self, role) {
            (Self::Contribution(destination), DirectExecutorRole::Contribution) => destination
                .extend_from_slice(
                    plan.contributions().get(start..end).ok_or_else(|| {
                        integrity("persisted family contribution range is absent")
                    })?,
                ),
            (Self::Finalization(destination), DirectExecutorRole::Finalization) => destination
                .extend_from_slice(
                    plan.finalizations().get(start..end).ok_or_else(|| {
                        integrity("persisted family finalization range is absent")
                    })?,
                ),
            (Self::Closure(destination), DirectExecutorRole::Closure) => destination
                .extend_from_slice(
                    plan.closures()
                        .get(start..end)
                        .ok_or_else(|| integrity("persisted family closure range is absent"))?,
                ),
            (_, DirectExecutorRole::Source) => {
                return Err(integrity(
                    "persisted family must not contain static source rows",
                ));
            }
            _ => {
                return Err(integrity(
                    "persisted family row storage differs from its role",
                ));
            }
        }
        Ok(())
    }

    fn freeze(self) -> PersistedFamilyRowsV1 {
        match self {
            Self::Contribution(rows) => {
                PersistedFamilyRowsV1::Contribution(rows.into_boxed_slice())
            }
            Self::Finalization(rows) => {
                PersistedFamilyRowsV1::Finalization(rows.into_boxed_slice())
            }
            Self::Closure(rows) => PersistedFamilyRowsV1::Closure(rows.into_boxed_slice()),
        }
    }
}

struct PersistedFamilyGroupDraftV1 {
    stage: u32,
    direct_executor_id: u32,
    rows: PersistedFamilyRowsDraftV1,
}

impl PersistedFamilyGroupDraftV1 {
    fn accepts(&self, stage: u32, role: DirectExecutorRole, direct_executor_id: u32) -> bool {
        self.stage == stage
            && self.rows.role() == role
            && self.direct_executor_id == direct_executor_id
    }
}

struct PersistedFamilyGroupV1 {
    stage: u32,
    role: DirectExecutorRole,
    direct_executor_id: u32,
    rows: PersistedFamilyRowsV1,
}

#[allow(dead_code)] // Retains authenticated helicity descriptors for inspection/probe builds.
struct PersistedPlanLayoutV1 {
    runtime_layout_digest: SemanticDigest,
    dispatch_address: usize,
    source_count: u32,
    lorentz_component_count: u16,
    parameter_count: u32,
    current_component_count: u32,
    amplitude_destination_count: u32,
    resolved_helicity_count: u32,
    resolved_helicities: Box<[DirectResolvedHelicityDescriptor]>,
    public_helicities: Box<[i32]>,
    momentum_forms: Box<[CanonicalMomentumLinearForm]>,
    exact_factors: Box<[ExactComplexRational]>,
    sources: Box<[DirectSourceRow]>,
    source_dispatch_variants: Box<[DirectSourceDispatchVariantDescriptor]>,
    source_embeddings: Box<[DirectSourceEmbeddingRow]>,
}

impl PersistedPlanLayoutV1 {
    fn new(
        plan: &DirectRecurrencePlan,
        dispatch: &DirectHelicityDispatch,
        lorentz_component_count: u16,
    ) -> RusticolResult<Self> {
        if lorentz_component_count == 0 {
            return Err(invalid(
                "persisted helicity family Lorentz component count is zero",
            ));
        }
        let mut momentum_forms = Vec::with_capacity(plan.momentum_forms().len());
        for descriptor in plan.momentum_forms() {
            let start = usize::try_from(descriptor.term_start)
                .map_err(|_| integrity("persisted momentum-form start exceeds usize"))?;
            let end = start
                .checked_add(descriptor.term_count as usize)
                .ok_or_else(|| integrity("persisted momentum-form range overflows usize"))?;
            let terms = plan
                .momentum_terms()
                .get(start..end)
                .ok_or_else(|| integrity("persisted momentum-form term range is out of bounds"))?;
            momentum_forms.push(CanonicalMomentumLinearForm::new(
                terms
                    .iter()
                    .map(|term| MomentumTerm {
                        source_slot: term.source_slot,
                        coefficient: term.coefficient,
                    })
                    .collect(),
            )?);
        }
        Ok(Self {
            runtime_layout_digest: plan.runtime_layout_digest(),
            dispatch_address: std::ptr::from_ref(dispatch).addr(),
            source_count: plan.external_source_count(),
            lorentz_component_count,
            parameter_count: plan.parameter_value_count(),
            current_component_count: plan.current_arena_components(),
            amplitude_destination_count: plan.amplitude_destination_count(),
            resolved_helicity_count: u32::try_from(plan.resolved_helicities().len())
                .map_err(|_| integrity("persisted resolved-helicity count exceeds u32"))?,
            resolved_helicities: plan.resolved_helicities().to_vec().into_boxed_slice(),
            public_helicities: plan.public_helicities().to_vec().into_boxed_slice(),
            momentum_forms: momentum_forms.into_boxed_slice(),
            exact_factors: plan.exact_factors().to_vec().into_boxed_slice(),
            sources: plan.sources().to_vec().into_boxed_slice(),
            source_dispatch_variants: plan.source_dispatch_variants().to_vec().into_boxed_slice(),
            source_embeddings: plan.source_embeddings().to_vec().into_boxed_slice(),
        })
    }
}

struct PersistedHelicityFamilyV1 {
    resolved_helicity_id: u32,
    source_selections: Box<[DirectResolvedSourceSelection]>,
    groups: Box<[PersistedFamilyGroupV1]>,
    unique_current_count: u32,
    unique_current_component_count: u32,
}

struct BoundPersistedHelicityFamilyV1 {
    family: PersistedHelicityFamilyV1,
    workspace: PersistedHelicityWorkspaceV1,
    resolved_groups: Box<[Option<ResolvedOnTheFlyExecutor>]>,
    union_source_program: Box<dyn OnTheFlyBoundUnionSourceProgram>,
    packed_singleton_capable: bool,
    interaction_program: DirectInteractionProgram,
    singleton_fanout_program: DirectSingletonContributionFanoutProgram,
    applied_parameter_version: u64,
    descriptor_exposed: bool,
    last_execution_report: Option<PersistedHelicityFamilyExecutionReportV1>,
}

/// Borrowed dense physical-color amplitude tile for one selected helicity.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Accessors are consumed by runtime adapters outside default core builds.
pub(crate) struct PersistedHelicityAmplitudeTileV1<'a> {
    amplitude_re: &'a [f64],
    amplitude_im: &'a [f64],
    destination_count: u32,
    point_count: u32,
    point_stride: u32,
    resolved_helicity_id: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PersistedHelicityFamilyExecutionReportV1 {
    pub(crate) cache_hit: bool,
    pub(crate) source_calls: u32,
    pub(crate) source_rows: u32,
    pub(crate) contribution_calls: u32,
    pub(crate) contribution_rows: u32,
    pub(crate) finalization_calls: u32,
    pub(crate) finalization_rows: u32,
    pub(crate) closure_calls: u32,
    pub(crate) closure_rows: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PersistedHelicityFamilyCacheCensusV1 {
    pub(crate) resolved_helicity_count: u32,
    pub(crate) retained_family_count: u32,
    pub(crate) retained_row_count: u64,
    pub(crate) active_resolved_helicity_id: Option<u32>,
}

/// Legacy query-family-shaped census for the retained persisted family.
///
/// This is derived at cold bind plus the most recent execution. Inspecting it
/// never revisits the dispatch CSR or scans support masks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PersistedHelicityFamilyInspectionCensusV1 {
    pub(crate) query_count: u32,
    pub(crate) union_unique_current_count: u32,
    pub(crate) union_unique_current_component_count: u32,
    pub(crate) union_source_rows: u32,
    pub(crate) union_contribution_rows: u32,
    pub(crate) union_finalization_rows: u32,
    pub(crate) union_closure_rows: u32,
    pub(crate) union_amplitude_destination_count: u32,
    pub(crate) union_source_executor_call_groups: u32,
    pub(crate) union_contribution_executor_call_groups: u32,
    pub(crate) union_finalization_executor_call_groups: u32,
    pub(crate) union_closure_executor_call_groups: u32,
    pub(crate) semantic_executor_binding_count: u32,
}

#[allow(dead_code)] // Accessors are consumed by runtime adapters outside default core builds.
impl<'a> PersistedHelicityAmplitudeTileV1<'a> {
    pub(crate) const fn amplitude_re(self) -> &'a [f64] {
        self.amplitude_re
    }

    pub(crate) const fn amplitude_im(self) -> &'a [f64] {
        self.amplitude_im
    }

    pub(crate) const fn destination_count(self) -> u32 {
        self.destination_count
    }

    pub(crate) const fn point_count(self) -> u32 {
        self.point_count
    }

    pub(crate) const fn point_stride(self) -> u32 {
        self.point_stride
    }

    pub(crate) const fn resolved_helicity_id(self) -> u32 {
        self.resolved_helicity_id
    }

    pub(crate) fn destination_re(self, destination_id: u32) -> RusticolResult<&'a [f64]> {
        let start = destination_id as usize * self.point_stride as usize;
        let end = start
            .checked_add(self.point_count as usize)
            .ok_or_else(|| integrity("persisted amplitude destination range overflows usize"))?;
        self.amplitude_re
            .get(start..end)
            .ok_or_else(|| integrity("persisted amplitude real destination is out of bounds"))
    }

    pub(crate) fn destination_im(self, destination_id: u32) -> RusticolResult<&'a [f64]> {
        let start = destination_id as usize * self.point_stride as usize;
        let end = start
            .checked_add(self.point_count as usize)
            .ok_or_else(|| integrity("persisted amplitude destination range overflows usize"))?;
        self.amplitude_im
            .get(start..end)
            .ok_or_else(|| integrity("persisted amplitude imaginary destination is out of bounds"))
    }
}

fn scalar_len(planes: u32, stride: u32, label: &str) -> RusticolResult<usize> {
    usize::try_from(planes)
        .ok()
        .and_then(|planes| planes.checked_mul(stride as usize))
        .ok_or_else(|| invalid(format!("persisted helicity {label} size exceeds usize")))
}

fn factor_parts(value: ExactComplexRational) -> RusticolResult<(f64, f64)> {
    let real = value.real().numerator() as f64 / value.real().denominator() as f64;
    let imag = value.imag().numerator() as f64 / value.imag().denominator() as f64;
    if !real.is_finite() || !imag.is_finite() {
        return Err(invalid(
            "persisted helicity exact factor is not finite in binary64",
        ));
    }
    Ok((real, imag))
}

struct PersistedHelicityWorkspaceV1 {
    current_re: AlignedF64Buffer,
    current_im: AlignedF64Buffer,
    amplitude_re: AlignedF64Buffer,
    amplitude_im: AlignedF64Buffer,
    momenta: AlignedF64Buffer,
    parameters_re: AlignedF64Buffer,
    parameters_im: AlignedF64Buffer,
    factors_re: AlignedF64Buffer,
    factors_im: AlignedF64Buffer,
    logical_point_capacity: u32,
    active_point_count: u32,
    point_stride: u32,
}

impl PersistedHelicityWorkspaceV1 {
    fn new(
        layout: &PersistedPlanLayoutV1,
        logical_point_capacity: u32,
        packed_singleton_capable: bool,
    ) -> RusticolResult<Self> {
        if layout.source_count == 0
            || layout.lorentz_component_count == 0
            || layout.current_component_count == 0
            || layout.amplitude_destination_count == 0
            || layout.momentum_forms.is_empty()
            || layout.exact_factors.is_empty()
            || logical_point_capacity == 0
        {
            return Err(integrity(
                "persisted helicity family has an empty workspace shape",
            ));
        }
        let point_stride = if logical_point_capacity == 1 && packed_singleton_capable {
            1
        } else {
            checked_aligned_point_stride(logical_point_capacity)?
        };
        let current_len = scalar_len(
            layout.current_component_count,
            point_stride,
            "current arena",
        )?;
        let amplitude_len = scalar_len(
            layout.amplitude_destination_count,
            point_stride,
            "amplitude arena",
        )?;
        let momentum_form_count = u32::try_from(layout.momentum_forms.len())
            .map_err(|_| invalid("persisted helicity momentum-form count exceeds u32"))?;
        let momentum_planes = momentum_form_count
            .checked_mul(u32::from(layout.lorentz_component_count))
            .ok_or_else(|| invalid("persisted helicity momentum-plane count exceeds u32"))?;
        let momentum_len = scalar_len(momentum_planes, point_stride, "momentum arena")?;
        let mut factors_re =
            AlignedF64Buffer::zeroed(layout.exact_factors.len(), "persisted helicity factor real")?;
        let mut factors_im = AlignedF64Buffer::zeroed(
            layout.exact_factors.len(),
            "persisted helicity factor imaginary",
        )?;
        for (index, factor) in layout.exact_factors.iter().copied().enumerate() {
            let (real, imag) = factor_parts(factor)?;
            factors_re.as_mut_slice()[index] = real;
            factors_im.as_mut_slice()[index] = imag;
        }
        let parameter_count = usize::try_from(layout.parameter_count)
            .map_err(|_| invalid("persisted helicity parameter count exceeds usize"))?;
        Ok(Self {
            current_re: AlignedF64Buffer::zeroed(current_len, "persisted helicity current real")?,
            current_im: AlignedF64Buffer::zeroed(
                current_len,
                "persisted helicity current imaginary",
            )?,
            amplitude_re: AlignedF64Buffer::zeroed(
                amplitude_len,
                "persisted helicity amplitude real",
            )?,
            amplitude_im: AlignedF64Buffer::zeroed(
                amplitude_len,
                "persisted helicity amplitude imaginary",
            )?,
            momenta: AlignedF64Buffer::zeroed(momentum_len, "persisted helicity momenta")?,
            parameters_re: AlignedF64Buffer::zeroed(
                parameter_count,
                "persisted helicity parameter real",
            )?,
            parameters_im: AlignedF64Buffer::zeroed(
                parameter_count,
                "persisted helicity parameter imaginary",
            )?,
            factors_re,
            factors_im,
            logical_point_capacity,
            active_point_count: 0,
            point_stride,
        })
    }

    fn refresh_parameters(&mut self, parameters: &[(f64, f64)]) -> RusticolResult<()> {
        self.active_point_count = 0;
        if parameters.len() != self.parameters_re.len() {
            return Err(invalid(format!(
                "persisted helicity received {} parameters, expected {}",
                parameters.len(),
                self.parameters_re.len()
            )));
        }
        for (index, &(real, imag)) in parameters.iter().enumerate() {
            if !real.is_finite() || !imag.is_finite() {
                return Err(invalid("persisted helicity parameter value is not finite"));
            }
            self.parameters_re.as_mut_slice()[index] = real;
            self.parameters_im.as_mut_slice()[index] = imag;
        }
        Ok(())
    }

    fn refresh_inputs(
        &mut self,
        layout: &PersistedPlanLayoutV1,
        external_momenta: &[f64],
        point_count: u32,
    ) -> RusticolResult<()> {
        self.active_point_count = 0;
        if point_count == 0 || point_count > self.logical_point_capacity {
            return Err(invalid(
                "persisted helicity point count is outside its workspace",
            ));
        }
        let expected = usize::try_from(layout.source_count)
            .ok()
            .and_then(|sources| sources.checked_mul(usize::from(layout.lorentz_component_count)))
            .and_then(|planes| planes.checked_mul(point_count as usize))
            .ok_or_else(|| invalid("persisted helicity external momentum shape exceeds usize"))?;
        if external_momenta.len() != expected {
            return Err(invalid(format!(
                "persisted helicity received {} momentum scalars, expected {expected}",
                external_momenta.len()
            )));
        }
        for (form_id, form) in layout.momentum_forms.iter().enumerate() {
            for lorentz in 0..usize::from(layout.lorentz_component_count) {
                for point in 0..point_count as usize {
                    let mut value = 0.0;
                    for term in form.terms() {
                        if term.source_slot >= layout.source_count {
                            return Err(integrity(
                                "persisted momentum source slot is out of bounds",
                            ));
                        }
                        // Recurrence lanes expose point-major external input:
                        // [point][source][Lorentz]. Keep that contract here so
                        // the persisted companion needs no transpose scratch.
                        let input_index = (point * layout.source_count as usize
                            + term.source_slot as usize)
                            * usize::from(layout.lorentz_component_count)
                            + lorentz;
                        value += f64::from(term.coefficient) * external_momenta[input_index];
                    }
                    let plane = form_id
                        .checked_mul(usize::from(layout.lorentz_component_count))
                        .and_then(|base| base.checked_add(lorentz))
                        .ok_or_else(|| invalid("persisted momentum plane exceeds usize"))?;
                    let index = plane
                        .checked_mul(self.point_stride as usize)
                        .and_then(|base| base.checked_add(point))
                        .ok_or_else(|| invalid("persisted momentum index exceeds usize"))?;
                    self.momenta.as_mut_slice()[index] = value;
                }
            }
        }
        // A selected H may have no live closure for a physical destination.
        // Clear every physical-color plane, not merely the selected closure
        // list, so H=A -> H=B -> H=A cannot expose a stale amplitude.
        for destination in 0..layout.amplitude_destination_count as usize {
            let start = destination * self.point_stride as usize;
            let end = start + point_count as usize;
            self.amplitude_re.as_mut_slice()[start..end].fill(0.0);
            self.amplitude_im.as_mut_slice()[start..end].fill(0.0);
        }
        Ok(())
    }

    fn raw_views(
        &mut self,
        layout: &PersistedPlanLayoutV1,
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
            form_count: u32::try_from(layout.momentum_forms.len())
                .map_err(|_| integrity("persisted momentum-form count exceeds u32"))?,
            lorentz_component_count: layout.lorentz_component_count,
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

    fn direct_workspace(&mut self, layout: &PersistedPlanLayoutV1) -> DirectWorkspace<'_> {
        DirectWorkspace {
            current_re: self.current_re.as_mut_slice(),
            current_im: self.current_im.as_mut_slice(),
            amplitude_re: self.amplitude_re.as_mut_slice(),
            amplitude_im: self.amplitude_im.as_mut_slice(),
            momenta: self.momenta.as_slice(),
            momentum_form_count: layout.momentum_forms.len() as u32,
            lorentz_component_count: layout.lorentz_component_count,
            parameters_re: self.parameters_re.as_slice(),
            parameters_im: self.parameters_im.as_slice(),
            factors_re: self.factors_re.as_slice(),
            factors_im: self.factors_im.as_slice(),
            point_stride: self.point_stride,
        }
    }

    fn tile(
        &self,
        layout: &PersistedPlanLayoutV1,
        resolved_helicity_id: u32,
        point_count: u32,
    ) -> RusticolResult<PersistedHelicityAmplitudeTileV1<'_>> {
        if self.active_point_count != point_count {
            return Err(integrity(
                "persisted helicity tile is not the last successful execution",
            ));
        }
        Ok(PersistedHelicityAmplitudeTileV1 {
            amplitude_re: self.amplitude_re.as_slice(),
            amplitude_im: self.amplitude_im.as_slice(),
            destination_count: layout.amplitude_destination_count,
            point_count,
            point_stride: self.point_stride,
            resolved_helicity_id,
        })
    }
}

fn selected_source_rows(
    plan: &DirectRecurrencePlan,
    resolved_helicity_id: u32,
) -> RusticolResult<Box<[DirectResolvedSourceSelection]>> {
    let descriptor = plan
        .resolved_helicities()
        .get(resolved_helicity_id as usize)
        .filter(|descriptor| descriptor.id == resolved_helicity_id)
        .ok_or_else(|| integrity("persisted resolved helicity is absent from its plan"))?;
    let start = usize::try_from(descriptor.source_selection_start)
        .map_err(|_| integrity("persisted source-selection start exceeds usize"))?;
    let end = start
        .checked_add(descriptor.source_selection_count as usize)
        .ok_or_else(|| integrity("persisted source-selection range overflows usize"))?;
    let selections = plan
        .resolved_source_selections()
        .get(start..end)
        .ok_or_else(|| integrity("persisted source-selection range is out of bounds"))?;
    if selections.is_empty() {
        return Err(integrity(
            "persisted resolved helicity has no source selections",
        ));
    }
    Ok(selections.to_vec().into_boxed_slice())
}

fn new_rows_draft(role: DirectExecutorRole) -> RusticolResult<PersistedFamilyRowsDraftV1> {
    match role {
        DirectExecutorRole::Contribution => {
            Ok(PersistedFamilyRowsDraftV1::Contribution(Vec::new()))
        }
        DirectExecutorRole::Finalization => {
            Ok(PersistedFamilyRowsDraftV1::Finalization(Vec::new()))
        }
        DirectExecutorRole::Closure => Ok(PersistedFamilyRowsDraftV1::Closure(Vec::new())),
        DirectExecutorRole::Source => Err(integrity(
            "persisted helicity dispatch contains a static source group",
        )),
    }
}

fn rederive_contribution_initialization(groups: &mut [PersistedFamilyGroupDraftV1]) {
    let mut initialized = BTreeSet::<(u32, u32)>::new();
    for group in groups {
        let PersistedFamilyRowsDraftV1::Contribution(rows) = &mut group.rows else {
            continue;
        };
        for row in rows {
            row.flags &= !DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
            if initialized.insert((group.stage, row.destination_component_base)) {
                row.flags |= DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
            }
        }
    }
}

fn order_contributions_for_fanout(groups: &mut [PersistedFamilyGroupDraftV1]) {
    for group in groups {
        let PersistedFamilyRowsDraftV1::Contribution(rows) = &mut group.rows else {
            continue;
        };
        let stage = group.stage;
        let executor_id = group.direct_executor_id;
        crate::recurrence::direct_lowering::order_contributions_for_runtime_fanout_by(
            rows,
            |row| {
                (
                    stage,
                    executor_id,
                    *row,
                    (
                        row.destination_component_base,
                        row.exact_factor_id,
                        row.flags & !DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
                    ),
                )
            },
        );
    }
}

/// Cold semantic lookup over a physically reused Direct-Arena layout.
///
/// A component base is unique only among currents live in the same stage. The
/// persisted family census therefore must not treat it as a global current ID.
struct PersistedCurrentIndexV1 {
    current_by_source_row: BTreeMap<u32, u32>,
    current_by_stage_and_base: BTreeMap<(u32, u32), u32>,
    currents_by_base: BTreeMap<u32, Vec<u32>>,
}

impl PersistedCurrentIndexV1 {
    fn new(plan: &DirectRecurrencePlan) -> RusticolResult<Self> {
        let mut current_by_source_row = BTreeMap::new();
        let mut current_by_stage_and_base = BTreeMap::new();
        let mut currents_by_base = BTreeMap::<u32, Vec<u32>>::new();
        for current in plan.currents() {
            let current_id = current.semantic_current_id;
            if current_by_stage_and_base
                .insert(
                    (u32::from(current.stage), current.component_base),
                    current_id,
                )
                .is_some()
            {
                return Err(integrity(
                    "persisted plan repeats a current component base within one stage",
                ));
            }
            if current.source_row_or_sentinel != DIRECT_NONE_U32
                && current_by_source_row
                    .insert(current.source_row_or_sentinel, current_id)
                    .is_some()
            {
                return Err(integrity(
                    "persisted plan repeats a source-row current binding",
                ));
            }
            currents_by_base
                .entry(current.component_base)
                .or_default()
                .push(current_id);
        }
        for current_ids in currents_by_base.values_mut() {
            current_ids.sort_unstable_by_key(|current_id| {
                let current = &plan.currents()[*current_id as usize];
                (current.first_use, current.last_use, *current_id)
            });
        }
        Ok(Self {
            current_by_source_row,
            current_by_stage_and_base,
            currents_by_base,
        })
    }

    fn source_current(&self, source_row_id: u32, label: &str) -> RusticolResult<u32> {
        self.current_by_source_row
            .get(&source_row_id)
            .copied()
            .ok_or_else(|| integrity(format!("persisted {label} has no semantic source current")))
    }

    fn destination_current(
        &self,
        stage: u32,
        component_base: u32,
        label: &str,
    ) -> RusticolResult<u32> {
        self.current_by_stage_and_base
            .get(&(stage, component_base))
            .copied()
            .ok_or_else(|| {
                integrity(format!(
                    "persisted {label} references no current defined at its stage"
                ))
            })
    }

    fn live_current(
        &self,
        plan: &DirectRecurrencePlan,
        stage: u32,
        component_base: u32,
        label: &str,
    ) -> RusticolResult<u32> {
        let current_ids = self.currents_by_base.get(&component_base).ok_or_else(|| {
            integrity(format!(
                "persisted {label} references a missing current component base"
            ))
        })?;
        let end = current_ids
            .partition_point(|current_id| plan.currents()[*current_id as usize].first_use <= stage);
        let current_id = current_ids
            .get(end.saturating_sub(1))
            .copied()
            .filter(|_| end != 0)
            .ok_or_else(|| {
                integrity(format!(
                    "persisted {label} references no current live at its stage"
                ))
            })?;
        let current = plan
            .currents()
            .get(current_id as usize)
            .filter(|current| current.semantic_current_id == current_id)
            .ok_or_else(|| integrity("persisted current index has a noncanonical semantic ID"))?;
        if stage > current.last_use {
            return Err(integrity(format!(
                "persisted {label} references no current live at its stage"
            )));
        }
        Ok(current_id)
    }
}

fn family_current_census(
    plan: &DirectRecurrencePlan,
    source_selections: &[DirectResolvedSourceSelection],
    groups: &[PersistedFamilyGroupV1],
) -> RusticolResult<(u32, u32)> {
    let current_index = PersistedCurrentIndexV1::new(plan)?;
    let mut active_currents = BTreeSet::<u32>::new();

    for selection in source_selections {
        let variant = plan
            .source_dispatch_variants()
            .get(selection.dispatch_variant_id as usize)
            .ok_or_else(|| integrity("persisted source selection has no dispatch variant"))?;
        let source = plan
            .sources()
            .get(variant.source_row_id as usize)
            .ok_or_else(|| integrity("persisted source dispatch has no source row"))?;
        let current_id = current_index.source_current(variant.source_row_id, "source row")?;
        let current = &plan.currents()[current_id as usize];
        if current.component_base != source.destination_component_base {
            return Err(integrity(
                "persisted source row differs from its semantic current binding",
            ));
        }
        active_currents.insert(current_id);
    }
    for group in groups {
        match &group.rows {
            PersistedFamilyRowsV1::Contribution(rows) => {
                for row in rows {
                    active_currents.insert(current_index.live_current(
                        plan,
                        group.stage,
                        row.parent0_component_base,
                        "contribution parent",
                    )?);
                    if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 {
                        active_currents.insert(current_index.live_current(
                            plan,
                            group.stage,
                            row.parent1_component_base_or_sentinel,
                            "contribution parent",
                        )?);
                    }
                    active_currents.insert(current_index.destination_current(
                        group.stage,
                        row.destination_component_base,
                        "contribution destination",
                    )?);
                }
            }
            PersistedFamilyRowsV1::Finalization(rows) => {
                for row in rows {
                    active_currents.insert(current_index.destination_current(
                        group.stage,
                        row.component_base,
                        "finalization row",
                    )?);
                }
            }
            PersistedFamilyRowsV1::Closure(rows) => {
                for row in rows {
                    active_currents.insert(current_index.live_current(
                        plan,
                        group.stage,
                        row.parent0_component_base,
                        "closure parent",
                    )?);
                    if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 {
                        active_currents.insert(current_index.live_current(
                            plan,
                            group.stage,
                            row.parent1_component_base_or_sentinel,
                            "closure parent",
                        )?);
                    }
                }
            }
        }
    }

    let unique_current_count = u32::try_from(active_currents.len())
        .map_err(|_| integrity("persisted active current count exceeds u32"))?;
    let unique_current_component_count =
        active_currents
            .into_iter()
            .try_fold(0u32, |count, current_id| {
                count
                    .checked_add(u32::from(
                        plan.currents()[current_id as usize].component_count,
                    ))
                    .ok_or_else(|| {
                        integrity("persisted active current component count exceeds u32")
                    })
            })?;
    Ok((unique_current_count, unique_current_component_count))
}

fn build_family(
    plan: &DirectRecurrencePlan,
    dispatch: &DirectHelicityDispatch,
    resolved_helicity_id: u32,
) -> RusticolResult<PersistedHelicityFamilyV1> {
    let mut drafts = Vec::<PersistedFamilyGroupDraftV1>::new();
    for &group_id in dispatch.group_ids_for_helicity(resolved_helicity_id)? {
        let descriptor = dispatch
            .row_groups()
            .get(group_id as usize)
            .ok_or_else(|| integrity("persisted helicity group ID is out of bounds"))?;
        let stage = u32::from(descriptor.stage);
        if !drafts.last().is_some_and(|draft| {
            draft.accepts(stage, descriptor.role, descriptor.direct_executor_id)
        }) {
            drafts.push(PersistedFamilyGroupDraftV1 {
                stage,
                direct_executor_id: descriptor.direct_executor_id,
                rows: new_rows_draft(descriptor.role)?,
            });
        }
        drafts
            .last_mut()
            .expect("persisted group draft was just inserted")
            .rows
            .append_from_plan(
                plan,
                descriptor.role,
                descriptor.row_start,
                descriptor.row_count,
            )?;
    }
    // A resolved helicity can be a certified structural zero. Its empty CSR
    // is executable: source dispatch remains exact and the full amplitude
    // clear below produces a deterministic dense zero tile.
    order_contributions_for_fanout(&mut drafts);
    rederive_contribution_initialization(&mut drafts);
    let groups = drafts
        .into_iter()
        .map(|draft| PersistedFamilyGroupV1 {
            stage: draft.stage,
            role: draft.rows.role(),
            direct_executor_id: draft.direct_executor_id,
            rows: draft.rows.freeze(),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let source_selections = selected_source_rows(plan, resolved_helicity_id)?;
    let (unique_current_count, unique_current_component_count) =
        family_current_census(plan, &source_selections, &groups)?;
    Ok(PersistedHelicityFamilyV1 {
        resolved_helicity_id,
        source_selections,
        groups,
        unique_current_count,
        unique_current_component_count,
    })
}

fn bind_groups<R: OnTheFlyPreparedExecutorResolver>(
    resolver: &R,
    family: &PersistedHelicityFamilyV1,
) -> RusticolResult<Box<[Option<ResolvedOnTheFlyExecutor>]>> {
    family
        .groups
        .iter()
        .map(|group| {
            if group.role == DirectExecutorRole::Contribution
                && group.direct_executor_id == DIRECT_NONE_U32
            {
                let PersistedFamilyRowsV1::Contribution(rows) = &group.rows else {
                    return Err(integrity(
                        "certified-reuse persisted group has non-contribution storage",
                    ));
                };
                if rows
                    .iter()
                    .any(|row| row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0)
                {
                    return Err(integrity(
                        "missing-executor persisted group is not certified reuse",
                    ));
                }
                return Ok(None);
            }
            let resolved =
                resolver.resolve_direct_executor(group.direct_executor_id, group.role)?;
            if resolved.direct_executor_id != group.direct_executor_id
                || resolved.handle.role() != group.role
            {
                return Err(integrity(
                    "persisted family executor binding differs from its direct row group",
                ));
            }
            Ok(Some(resolved))
        })
        .collect::<RusticolResult<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

fn direct_group_views<'a>(
    family: &'a PersistedHelicityFamilyV1,
    resolved_groups: &[Option<ResolvedOnTheFlyExecutor>],
) -> RusticolResult<(Vec<DirectInteractionGroupView<'a>>, usize)> {
    if family.groups.len() != resolved_groups.len() {
        return Err(integrity(
            "persisted family resolved group count is inconsistent",
        ));
    }
    let mut views = Vec::with_capacity(family.groups.len());
    let mut contribution_count = 0usize;
    for (group, resolved) in family.groups.iter().zip(resolved_groups) {
        views.push(match &group.rows {
            PersistedFamilyRowsV1::Contribution(rows) => {
                contribution_count =
                    contribution_count.checked_add(rows.len()).ok_or_else(|| {
                        integrity("persisted family contribution count exceeds usize")
                    })?;
                DirectInteractionGroupView::contribution(
                    group.stage,
                    group.direct_executor_id,
                    resolved.map(|resolved| resolved.handle),
                    resolved.and_then(|resolved| resolved.interaction_capability),
                    rows,
                )
            }
            PersistedFamilyRowsV1::Finalization(_) => DirectInteractionGroupView::other(
                group.stage,
                group.role,
                group.direct_executor_id,
                resolved.map(|resolved| resolved.handle),
            ),
            PersistedFamilyRowsV1::Closure(rows) => {
                let resolved = resolved
                    .ok_or_else(|| integrity("persisted closure group has no resolved executor"))?;
                DirectInteractionGroupView::closure(
                    group.stage,
                    group.direct_executor_id,
                    resolved.handle,
                    rows,
                )
            }
        });
    }
    Ok((views, contribution_count))
}

fn bind_programs(
    plan: &DirectRecurrencePlan,
    layout: &PersistedPlanLayoutV1,
    family: &PersistedHelicityFamilyV1,
    resolved_groups: &[Option<ResolvedOnTheFlyExecutor>],
) -> RusticolResult<(
    DirectInteractionProgram,
    DirectSingletonContributionFanoutProgram,
)> {
    let (views, contribution_count) = direct_group_views(family, resolved_groups)?;
    let schedule = DirectInteractionScheduleView::new(
        RecurrenceStrategy::ContractedColorUnion,
        &views,
        plan.currents(),
        contribution_count,
    );
    let interaction = DirectInteractionProgram::build(schedule)?;
    let fanout = DirectSingletonContributionFanoutProgram::build(
        &views,
        layout.current_component_count,
        u32::try_from(layout.momentum_forms.len())
            .map_err(|_| integrity("persisted momentum-form count exceeds u32"))?,
        layout.parameter_count,
        u32::try_from(layout.exact_factors.len())
            .map_err(|_| integrity("persisted exact-factor count exceeds u32"))?,
    )?;
    Ok((interaction, fanout))
}

fn increment(value: &mut u32, by: u32, label: &str) -> RusticolResult<()> {
    *value = value
        .checked_add(by)
        .ok_or_else(|| integrity(format!("persisted helicity {label} exceeds u32")))?;
    Ok(())
}

fn record_profiled_group(
    report: &mut PersistedHelicityFamilyExecutionReportV1,
    role: DirectExecutorRole,
    row_count: u32,
) -> RusticolResult<()> {
    match role {
        DirectExecutorRole::Source => {
            increment(&mut report.source_calls, 1, "source calls")?;
            increment(&mut report.source_rows, row_count, "source rows")
        }
        DirectExecutorRole::Contribution => {
            increment(&mut report.contribution_calls, 1, "contribution calls")?;
            increment(
                &mut report.contribution_rows,
                row_count,
                "contribution rows",
            )
        }
        DirectExecutorRole::Finalization => {
            increment(&mut report.finalization_calls, 1, "finalization calls")?;
            increment(
                &mut report.finalization_rows,
                row_count,
                "finalization rows",
            )
        }
        DirectExecutorRole::Closure => {
            increment(&mut report.closure_calls, 1, "closure calls")?;
            increment(&mut report.closure_rows, row_count, "closure rows")
        }
    }
}

fn execute_union_source(
    program: &dyn OnTheFlyBoundUnionSourceProgram,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    point_count: u32,
) -> RusticolResult<()> {
    // `raw_views` authenticated the descriptor bundle immediately before this
    // call. The source program authenticated every plan-owned table and bound
    // all catalog lookups during `prepare`.
    unsafe { program.execute(arena, momenta, parameters, point_count) }
}

fn execute_bound_family<Report, RecordGroup>(
    layout: &PersistedPlanLayoutV1,
    cached: &mut BoundPersistedHelicityFamilyV1,
    point_count: u32,
    mut report: Report,
    mut record_group: RecordGroup,
) -> RusticolResult<Report>
where
    RecordGroup: FnMut(&mut Report, DirectExecutorRole, u32) -> RusticolResult<()>,
{
    clear_direct_executor_error_detail();
    // The workspace shape is immutable while a family is bound. Authenticate
    // its raw descriptors once per tile, then reuse the Copy views for the
    // source dispatcher and every row group, exactly as the OTF family path
    // does. Revalidating their pointer ranges inside each group is redundant.
    let (arena, momenta, parameters, factors) = cached.workspace.raw_views(layout)?;
    execute_union_source(
        cached.union_source_program.as_ref(),
        arena,
        momenta,
        parameters,
        point_count,
    )?;
    let source_row_count = u32::try_from(cached.family.source_selections.len())
        .map_err(|_| integrity("persisted source-selection count exceeds u32"))?;
    record_group(&mut report, DirectExecutorRole::Source, source_row_count)?;

    for (group_index, (group, resolved)) in cached
        .family
        .groups
        .iter()
        .zip(cached.resolved_groups.iter().copied())
        .enumerate()
    {
        let row_count = u32::try_from(group.rows.len())
            .map_err(|_| integrity("persisted row-group count exceeds u32"))?;
        if group.role == DirectExecutorRole::Source {
            return Err(integrity(
                "persisted non-source schedule contains a source group",
            ));
        }
        record_group(&mut report, group.role, row_count)?;

        if point_count == 1 {
            match cached.interaction_program.action(group_index) {
                DirectInteractionGroupAction::Normal => {
                    if let PersistedFamilyRowsV1::Contribution(rows) = &group.rows
                        && group.direct_executor_id != DIRECT_NONE_U32
                        && cached.singleton_fanout_program.execute_group(
                            group_index,
                            rows,
                            arena,
                            momenta,
                            parameters,
                            factors,
                            point_count,
                        )?
                    {
                        continue;
                    }
                }
                DirectInteractionGroupAction::Consumed => continue,
                DirectInteractionGroupAction::Execute(stage_index) => {
                    cached.interaction_program.execute::<false>(
                        stage_index,
                        arena,
                        momenta,
                        parameters,
                        factors,
                    )?;
                    continue;
                }
            }
        }

        if group.role == DirectExecutorRole::Contribution
            && group.direct_executor_id == DIRECT_NONE_U32
        {
            let PersistedFamilyRowsV1::Contribution(rows) = &group.rows else {
                return Err(integrity("certified-reuse group has non-contribution rows"));
            };
            let mut workspace = cached.workspace.direct_workspace(layout);
            execute_certified_reuse_rows(rows, &mut workspace, point_count)?;
            continue;
        }

        let resolved = resolved
            .ok_or_else(|| integrity("persisted callable row group has no resolved executor"))?;
        let status = unsafe {
            match (&group.rows, resolved.handle) {
                (
                    PersistedFamilyRowsV1::Contribution(rows),
                    DirectExecutorHandle::Contribution { call, context },
                ) => call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    row_count,
                    point_count,
                ),
                (
                    PersistedFamilyRowsV1::Finalization(rows),
                    DirectExecutorHandle::Finalization { call, context },
                ) => call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    row_count,
                    point_count,
                ),
                (
                    PersistedFamilyRowsV1::Closure(rows),
                    DirectExecutorHandle::Closure { call, context },
                ) => call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    row_count,
                    point_count,
                ),
                _ => {
                    return Err(integrity(
                        "persisted family handle differs from its row storage",
                    ));
                }
            }
        };
        check_status(group.role, group.direct_executor_id, status)?;
    }
    cached.workspace.active_point_count = point_count;
    Ok(report)
}

/// One retained cold binding for the active auxiliary-plan helicity.
/// Repeated calls for that H stay on the bound row/program path and never
/// revisit the dispatch CSR. Switching H replaces this workspace deliberately
/// instead of multiplying recurrence arenas by the full helicity count.
pub(crate) struct PersistedHelicityFamilyExecutorV1<R: OnTheFlyPreparedExecutorResolver> {
    family: Option<BoundPersistedHelicityFamilyV1>,
    layout: Option<PersistedPlanLayoutV1>,
    resolver: R,
    parameter_state: Vec<(f64, f64)>,
    parameter_version: u64,
    active_cache_hit: bool,
}

#[allow(dead_code)] // Inspection/control methods are used by feature-specific runtime adapters.
impl<R: OnTheFlyPreparedExecutorResolver> PersistedHelicityFamilyExecutorV1<R> {
    pub(crate) const fn new(resolver: R) -> Self {
        Self {
            family: None,
            layout: None,
            resolver,
            parameter_state: Vec::new(),
            parameter_version: 0,
            active_cache_hit: false,
        }
    }

    pub(crate) const fn resolver(&self) -> &R {
        &self.resolver
    }

    pub(crate) const fn resolver_mut(&mut self) -> &mut R {
        &mut self.resolver
    }

    fn invalidate_exposed_row_tables(&mut self) -> RusticolResult<()> {
        if !self
            .family
            .as_ref()
            .is_some_and(|family| family.descriptor_exposed)
        {
            return Ok(());
        }
        self.resolver.invalidate_row_tables()?;
        if let Some(family) = &mut self.family {
            family.descriptor_exposed = false;
        }
        Ok(())
    }

    pub(crate) fn clear_families(&mut self) -> RusticolResult<()> {
        self.invalidate_exposed_row_tables()?;
        self.family = None;
        self.layout = None;
        self.active_cache_hit = false;
        Ok(())
    }

    pub(crate) fn cache_census(&self) -> PersistedHelicityFamilyCacheCensusV1 {
        PersistedHelicityFamilyCacheCensusV1 {
            resolved_helicity_count: self
                .layout
                .as_ref()
                .map_or(0, |layout| layout.resolved_helicity_count),
            retained_family_count: u32::from(self.family.is_some()),
            retained_row_count: self
                .family
                .as_ref()
                .into_iter()
                .flat_map(|family| family.family.groups.iter())
                .map(|group| group.rows.len() as u64)
                .sum(),
            active_resolved_helicity_id: self
                .family
                .as_ref()
                .map(|family| family.family.resolved_helicity_id),
        }
    }

    pub(crate) fn active_family_inspection_census(
        &self,
    ) -> Option<PersistedHelicityFamilyInspectionCensusV1> {
        let layout = self.layout.as_ref()?;
        let cached = self.family.as_ref()?;
        // The option is the successful-execution marker. Derive diagnostic
        // counts only when inspection is requested so ordinary unprofiled
        // evaluation does no per-group report bookkeeping.
        cached.last_execution_report?;
        let mut report = PersistedHelicityFamilyExecutionReportV1 {
            source_calls: 1,
            source_rows: u32::try_from(cached.family.source_selections.len()).ok()?,
            ..PersistedHelicityFamilyExecutionReportV1::default()
        };
        let mut executor_ids = BTreeSet::<u32>::new();
        for selection in &cached.family.source_selections {
            let variant = layout
                .source_dispatch_variants
                .get(selection.dispatch_variant_id as usize)?;
            if variant.direct_executor_id != DIRECT_NONE_U32 {
                executor_ids.insert(variant.direct_executor_id);
            }
        }
        for group in &cached.family.groups {
            let row_count = u32::try_from(group.rows.len()).ok()?;
            match group.role {
                DirectExecutorRole::Source => return None,
                DirectExecutorRole::Contribution => {
                    increment(&mut report.contribution_calls, 1, "contribution calls").ok()?;
                    increment(
                        &mut report.contribution_rows,
                        row_count,
                        "contribution rows",
                    )
                    .ok()?;
                }
                DirectExecutorRole::Finalization => {
                    increment(&mut report.finalization_calls, 1, "finalization calls").ok()?;
                    increment(
                        &mut report.finalization_rows,
                        row_count,
                        "finalization rows",
                    )
                    .ok()?;
                }
                DirectExecutorRole::Closure => {
                    increment(&mut report.closure_calls, 1, "closure calls").ok()?;
                    increment(&mut report.closure_rows, row_count, "closure rows").ok()?;
                }
            }
            if group.direct_executor_id != DIRECT_NONE_U32 {
                executor_ids.insert(group.direct_executor_id);
            }
        }
        let semantic_executor_binding_count = u32::try_from(executor_ids.len()).ok()?;
        Some(PersistedHelicityFamilyInspectionCensusV1 {
            query_count: layout.amplitude_destination_count,
            union_unique_current_count: cached.family.unique_current_count,
            union_unique_current_component_count: cached.family.unique_current_component_count,
            union_source_rows: report.source_rows,
            union_contribution_rows: report.contribution_rows,
            union_finalization_rows: report.finalization_rows,
            union_closure_rows: report.closure_rows,
            union_amplitude_destination_count: layout.amplitude_destination_count,
            union_source_executor_call_groups: report.source_calls,
            union_contribution_executor_call_groups: report.contribution_calls,
            union_finalization_executor_call_groups: report.finalization_calls,
            union_closure_executor_call_groups: report.closure_calls,
            semantic_executor_binding_count,
        })
    }

    pub(crate) fn resolved_helicities(
        &self,
    ) -> RusticolResult<&[DirectResolvedHelicityDescriptor]> {
        self.layout
            .as_ref()
            .map(|layout| layout.resolved_helicities.as_ref())
            .ok_or_else(|| invalid("persisted helicity descriptors require a cold prepare"))
    }

    pub(crate) fn public_helicities(&self) -> RusticolResult<&[i32]> {
        self.layout
            .as_ref()
            .map(|layout| layout.public_helicities.as_ref())
            .ok_or_else(|| invalid("persisted public helicities require a cold prepare"))
    }

    pub(crate) fn set_parameters(&mut self, parameters: &[(f64, f64)]) -> RusticolResult<()> {
        if parameters
            .iter()
            .any(|(real, imag)| !real.is_finite() || !imag.is_finite())
        {
            return Err(invalid("persisted helicity parameter value is not finite"));
        }
        if self.parameter_state == parameters {
            return Ok(());
        }
        let mut replacement = Vec::new();
        replacement
            .try_reserve_exact(parameters.len())
            .map_err(|error| {
                invalid(format!(
                    "persisted helicity parameter allocation failed: {error}"
                ))
            })?;
        replacement.extend_from_slice(parameters);
        self.parameter_state = replacement;
        self.parameter_version = self
            .parameter_version
            .checked_add(1)
            .ok_or_else(|| invalid("persisted helicity parameter version exceeds u64"))?;
        Ok(())
    }

    /// Update from the recurrence lane's persistent split-complex parameter
    /// planes. Equality is checked in place; unchanged warm preparation does
    /// not allocate or rewrite any retained workspace.
    pub(crate) fn set_parameter_planes(
        &mut self,
        parameters_re: &[f64],
        parameters_im: &[f64],
    ) -> RusticolResult<()> {
        if parameters_re.len() != parameters_im.len() {
            return Err(invalid(
                "persisted helicity parameter planes have different lengths",
            ));
        }
        if parameters_re
            .iter()
            .zip(parameters_im)
            .any(|(real, imag)| !real.is_finite() || !imag.is_finite())
        {
            return Err(invalid("persisted helicity parameter value is not finite"));
        }
        if self.parameter_state.len() == parameters_re.len()
            && self
                .parameter_state
                .iter()
                .zip(parameters_re.iter().zip(parameters_im))
                .all(|(&(cached_re, cached_im), (&real, &imag))| {
                    cached_re == real && cached_im == imag
                })
        {
            return Ok(());
        }
        let mut replacement = Vec::new();
        replacement
            .try_reserve_exact(parameters_re.len())
            .map_err(|error| {
                invalid(format!(
                    "persisted helicity parameter allocation failed: {error}"
                ))
            })?;
        replacement.extend(
            parameters_re
                .iter()
                .copied()
                .zip(parameters_im.iter().copied()),
        );
        self.parameter_state = replacement;
        self.parameter_version = self
            .parameter_version
            .checked_add(1)
            .ok_or_else(|| invalid("persisted helicity parameter version exceeds u64"))?;
        Ok(())
    }

    fn ensure_layout(
        &mut self,
        plan: &DirectRecurrencePlan,
        dispatch: &DirectHelicityDispatch,
        lorentz_component_count: u16,
    ) -> RusticolResult<()> {
        let matches = self.layout.as_ref().is_some_and(|layout| {
            layout.runtime_layout_digest == plan.runtime_layout_digest()
                && layout.dispatch_address == std::ptr::from_ref(dispatch).addr()
                && layout.lorentz_component_count == lorentz_component_count
        });
        if matches {
            return Ok(());
        }
        dispatch.validate_for_plan(plan)?;
        self.clear_families()?;
        self.layout = Some(PersistedPlanLayoutV1::new(
            plan,
            dispatch,
            lorentz_component_count,
        )?);
        Ok(())
    }

    /// Cold-select one full auxiliary-plan resolved helicity.
    ///
    /// Returns true when the optimized family was already retained. All CSR
    /// traversal, row copying, INIT derivation, and program construction occur
    /// here and are absent from warmed execution.
    pub(crate) fn prepare(
        &mut self,
        plan: &DirectRecurrencePlan,
        dispatch: &DirectHelicityDispatch,
        resolved_helicity_id: u32,
        lorentz_component_count: u16,
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        self.ensure_layout(plan, dispatch, lorentz_component_count)?;
        let layout = self
            .layout
            .as_ref()
            .expect("persisted layout was just established");
        if resolved_helicity_id >= layout.resolved_helicity_count {
            return Err(invalid(
                "persisted resolved helicity is outside the auxiliary plan",
            ));
        }

        if self
            .family
            .as_ref()
            .is_some_and(|family| family.family.resolved_helicity_id == resolved_helicity_id)
        {
            let needs_resize = self.family.as_ref().is_some_and(|family| {
                logical_point_capacity > family.workspace.logical_point_capacity
            });
            if needs_resize {
                let family = self
                    .family
                    .as_ref()
                    .expect("retained persisted family disappeared");
                let replacement = PersistedHelicityWorkspaceV1::new(
                    layout,
                    logical_point_capacity,
                    family.packed_singleton_capable,
                )?;
                self.invalidate_exposed_row_tables()?;
                let family = self
                    .family
                    .as_mut()
                    .expect("retained persisted family disappeared after invalidation");
                family.workspace = replacement;
                family.applied_parameter_version = u64::MAX;
            }
            self.active_cache_hit = true;
            return Ok(true);
        }

        let family = build_family(plan, dispatch, resolved_helicity_id)?;
        let resolved_groups = bind_groups(&self.resolver, &family)?;
        let packed_singleton_capable = self.resolver.union_source_packed_singleton_capable()
            && resolved_groups
                .iter()
                .flatten()
                .all(|resolved| resolved.packed_singleton_capable);
        let workspace = PersistedHelicityWorkspaceV1::new(
            layout,
            logical_point_capacity,
            packed_singleton_capable,
        )?;
        let (interaction_program, singleton_fanout_program) =
            bind_programs(plan, layout, &family, &resolved_groups)?;
        let union_source_program =
            self.resolver
                .bind_union_source_program(OnTheFlyUnionSourceBindingView {
                    sources: &layout.sources,
                    variants: &layout.source_dispatch_variants,
                    embeddings: &layout.source_embeddings,
                    selections: &family.source_selections,
                    exact_factors: &layout.exact_factors,
                    current_component_count: layout.current_component_count,
                    momentum_form_count: u32::try_from(layout.momentum_forms.len())
                        .map_err(|_| integrity("persisted momentum-form count exceeds u32"))?,
                    parameter_count: layout.parameter_count,
                })?;
        let bound = BoundPersistedHelicityFamilyV1 {
            family,
            workspace,
            resolved_groups,
            union_source_program,
            packed_singleton_capable,
            interaction_program,
            singleton_fanout_program,
            applied_parameter_version: u64::MAX,
            descriptor_exposed: false,
            last_execution_report: None,
        };
        self.invalidate_exposed_row_tables()?;
        self.family = Some(bound);
        self.active_cache_hit = false;
        Ok(false)
    }

    fn execute_active<Report, RecordGroup>(
        &mut self,
        external_momenta: &[f64],
        point_count: u32,
        report: Report,
        record_group: RecordGroup,
    ) -> RusticolResult<Report>
    where
        RecordGroup: FnMut(&mut Report, DirectExecutorRole, u32) -> RusticolResult<()>,
    {
        let layout = self
            .layout
            .as_ref()
            .ok_or_else(|| invalid("persisted helicity execution requires a cold prepare"))?;
        let family = self
            .family
            .as_mut()
            .ok_or_else(|| integrity("active persisted helicity family is absent"))?;
        if family.applied_parameter_version != self.parameter_version {
            family.workspace.refresh_parameters(&self.parameter_state)?;
            family.applied_parameter_version = self.parameter_version;
        }
        family
            .workspace
            .refresh_inputs(layout, external_momenta, point_count)?;
        family.descriptor_exposed = true;
        let report = execute_bound_family(layout, family, point_count, report, record_group)?;
        self.active_cache_hit = true;
        Ok(report)
    }

    fn mark_execution_report(&mut self, report: PersistedHelicityFamilyExecutionReportV1) {
        self.family
            .as_mut()
            .expect("successful persisted execution lost its active family")
            .last_execution_report = Some(report);
    }

    pub(crate) fn execute_tile_unprofiled(
        &mut self,
        external_momenta: &[f64],
        point_count: u32,
    ) -> RusticolResult<PersistedHelicityAmplitudeTileV1<'_>> {
        self.execute_active(external_momenta, point_count, (), |_, _, _| Ok(()))?;
        self.mark_execution_report(PersistedHelicityFamilyExecutionReportV1::default());
        self.active_tile(point_count)
    }

    pub(crate) fn execute_tile_profiled(
        &mut self,
        external_momenta: &[f64],
        point_count: u32,
    ) -> RusticolResult<(
        PersistedHelicityAmplitudeTileV1<'_>,
        PersistedHelicityFamilyExecutionReportV1,
    )> {
        let report = self.execute_active(
            external_momenta,
            point_count,
            PersistedHelicityFamilyExecutionReportV1 {
                cache_hit: self.active_cache_hit,
                ..PersistedHelicityFamilyExecutionReportV1::default()
            },
            record_profiled_group,
        )?;
        self.mark_execution_report(report);
        let tile = self.active_tile(point_count)?;
        Ok((tile, report))
    }

    fn active_tile(
        &self,
        point_count: u32,
    ) -> RusticolResult<PersistedHelicityAmplitudeTileV1<'_>> {
        let layout = self
            .layout
            .as_ref()
            .ok_or_else(|| integrity("persisted helicity layout disappeared"))?;
        let family = self
            .family
            .as_ref()
            .ok_or_else(|| integrity("persisted helicity active family is absent"))?;
        family
            .workspace
            .tile(layout, family.family.resolved_helicity_id, point_count)
    }

    fn validate_output_shape(
        &self,
        point_count: u32,
        output_re: &[f64],
        output_im: &[f64],
    ) -> RusticolResult<()> {
        let destination_count = self
            .layout
            .as_ref()
            .ok_or_else(|| invalid("persisted helicity output requires a cold prepare"))?
            .amplitude_destination_count;
        let expected = destination_count as usize * point_count as usize;
        if output_re.len() != expected || output_im.len() != expected {
            return Err(invalid(format!(
                "persisted helicity outputs have shapes ({}, {}), expected ({expected}, {expected})",
                output_re.len(),
                output_im.len()
            )));
        }
        Ok(())
    }

    fn copy_active_outputs(
        &self,
        point_count: u32,
        output_re: &mut [f64],
        output_im: &mut [f64],
    ) -> RusticolResult<()> {
        self.validate_output_shape(point_count, output_re, output_im)?;
        let tile = self.active_tile(point_count)?;
        for destination in 0..tile.destination_count() as usize {
            let source_start = destination * tile.point_stride() as usize;
            let source_end = source_start + point_count as usize;
            let destination_start = destination * point_count as usize;
            let destination_end = destination_start + point_count as usize;
            output_re[destination_start..destination_end]
                .copy_from_slice(&tile.amplitude_re()[source_start..source_end]);
            output_im[destination_start..destination_end]
                .copy_from_slice(&tile.amplitude_im()[source_start..source_end]);
        }
        Ok(())
    }

    pub(crate) fn execute_into_unprofiled(
        &mut self,
        external_momenta: &[f64],
        point_count: u32,
        output_re: &mut [f64],
        output_im: &mut [f64],
    ) -> RusticolResult<()> {
        self.validate_output_shape(point_count, output_re, output_im)?;
        self.execute_active(external_momenta, point_count, (), |_, _, _| Ok(()))?;
        self.mark_execution_report(PersistedHelicityFamilyExecutionReportV1::default());
        self.copy_active_outputs(point_count, output_re, output_im)
    }

    pub(crate) fn execute_into_profiled(
        &mut self,
        external_momenta: &[f64],
        point_count: u32,
        output_re: &mut [f64],
        output_im: &mut [f64],
    ) -> RusticolResult<PersistedHelicityFamilyExecutionReportV1> {
        self.validate_output_shape(point_count, output_re, output_im)?;
        let mut report = PersistedHelicityFamilyExecutionReportV1 {
            cache_hit: self.active_cache_hit,
            ..PersistedHelicityFamilyExecutionReportV1::default()
        };
        report =
            self.execute_active(external_momenta, point_count, report, record_profiled_group)?;
        self.mark_execution_report(report);
        self.copy_active_outputs(point_count, output_re, output_im)?;
        Ok(report)
    }
}

impl<R: OnTheFlyPreparedExecutorResolver> Drop for PersistedHelicityFamilyExecutorV1<R> {
    fn drop(&mut self) {
        if self.invalidate_exposed_row_tables().is_ok() {
            return;
        }
        // A callback unexpectedly retaining row pointers is safer with leaked
        // private rows than with a dangling descriptor during teardown.
        if let Some(family) = self.family.take()
            && family.descriptor_exposed
        {
            std::mem::forget(family);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{
        DirectHelicityDispatchDescriptor, DirectHelicityDispatchParts,
        DirectHelicityRowGroupDescriptor, DirectHelicitySupportDomainDescriptor,
        DirectSourceProjectionRow,
    };
    use std::cell::Cell;
    use std::ffi::c_void;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn test_layout() -> PersistedPlanLayoutV1 {
        PersistedPlanLayoutV1 {
            runtime_layout_digest: digest(0x44),
            dispatch_address: 0,
            source_count: 2,
            lorentz_component_count: 2,
            parameter_count: 0,
            current_component_count: 4,
            amplitude_destination_count: 2,
            resolved_helicity_count: 2,
            resolved_helicities: Box::new([]),
            public_helicities: Box::new([]),
            momentum_forms: vec![
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot: 0,
                    coefficient: 1,
                }])
                .unwrap(),
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot: 1,
                    coefficient: -1,
                }])
                .unwrap(),
            ]
            .into_boxed_slice(),
            exact_factors: vec![ExactComplexRational::ONE].into_boxed_slice(),
            sources: Box::new([]),
            source_dispatch_variants: Box::new([]),
            source_embeddings: Box::new([]),
        }
    }

    fn contribution(destination: u32, flags: u32) -> DirectContributionRow {
        DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 1,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: destination,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags,
        }
    }

    struct ProbeBoundUnionSourceProgram {
        value: f64,
    }

    impl OnTheFlyBoundUnionSourceProgram for ProbeBoundUnionSourceProgram {
        unsafe fn execute(
            &self,
            arena: DirectArenaView,
            _momenta: DirectMomentumView,
            _parameters: DirectParameterView,
            point_count: u32,
        ) -> RusticolResult<()> {
            let current_re = unsafe {
                std::slice::from_raw_parts_mut(arena.current_re, arena.current_scalar_len as usize)
            };
            let current_im = unsafe {
                std::slice::from_raw_parts_mut(arena.current_im, arena.current_scalar_len as usize)
            };
            for component in 0..2usize {
                for point in 0..point_count as usize {
                    let index = component * arena.point_stride as usize + point;
                    current_re[index] = self.value;
                    current_im[index] = 0.0;
                }
            }
            Ok(())
        }
    }

    unsafe extern "C" fn contribution_copy(
        _context: *const c_void,
        arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        _factors: DirectFactorView,
        rows: *const DirectContributionRow,
        row_count: u32,
        point_count: u32,
    ) -> c_int {
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        let current_re = unsafe {
            std::slice::from_raw_parts_mut(arena.current_re, arena.current_scalar_len as usize)
        };
        let current_im = unsafe {
            std::slice::from_raw_parts_mut(arena.current_im, arena.current_scalar_len as usize)
        };
        for row in rows {
            for point in 0..point_count as usize {
                let source =
                    row.parent0_component_base as usize * arena.point_stride as usize + point;
                let destination =
                    row.destination_component_base as usize * arena.point_stride as usize + point;
                if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
                    current_re[destination] = current_re[source];
                    current_im[destination] = current_im[source];
                } else {
                    current_re[destination] += current_re[source];
                    current_im[destination] += current_im[source];
                }
            }
        }
        DIRECT_STATUS_OK
    }

    unsafe extern "C" fn finalize_noop(
        _context: *const c_void,
        _arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        _factors: DirectFactorView,
        _rows: *const DirectFinalizationRow,
        _row_count: u32,
        _point_count: u32,
    ) -> c_int {
        DIRECT_STATUS_OK
    }

    unsafe extern "C" fn close_copy(
        _context: *const c_void,
        arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        _factors: DirectFactorView,
        rows: *const DirectClosureRow,
        row_count: u32,
        point_count: u32,
    ) -> c_int {
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        let current_re = unsafe {
            std::slice::from_raw_parts(arena.current_re, arena.current_scalar_len as usize)
        };
        let current_im = unsafe {
            std::slice::from_raw_parts(arena.current_im, arena.current_scalar_len as usize)
        };
        let amplitude_re = unsafe {
            std::slice::from_raw_parts_mut(arena.amplitude_re, arena.amplitude_scalar_len as usize)
        };
        let amplitude_im = unsafe {
            std::slice::from_raw_parts_mut(arena.amplitude_im, arena.amplitude_scalar_len as usize)
        };
        for row in rows {
            for point in 0..point_count as usize {
                let source =
                    row.parent0_component_base as usize * arena.point_stride as usize + point;
                let destination =
                    row.amplitude_destination_id as usize * arena.point_stride as usize + point;
                amplitude_re[destination] += current_re[source];
                amplitude_im[destination] += current_im[source];
            }
        }
        DIRECT_STATUS_OK
    }

    struct ProbeResolver {
        invalidations: Cell<u32>,
        source_bindings: Cell<u32>,
    }

    impl OnTheFlyPreparedExecutorResolver for ProbeResolver {
        fn resolve(&self, _key: OnTheFlyExecutorKeyV1) -> RusticolResult<ResolvedOnTheFlyExecutor> {
            Err(integrity("semantic resolver is unused by persisted test"))
        }

        fn resolve_direct_executor(
            &self,
            direct_executor_id: u32,
            role: DirectExecutorRole,
        ) -> RusticolResult<ResolvedOnTheFlyExecutor> {
            let handle = match (direct_executor_id, role) {
                (1, DirectExecutorRole::Contribution) => DirectExecutorHandle::Contribution {
                    call: contribution_copy,
                    context: std::ptr::null(),
                },
                (2, DirectExecutorRole::Finalization) => DirectExecutorHandle::Finalization {
                    call: finalize_noop,
                    context: std::ptr::null(),
                },
                (3, DirectExecutorRole::Closure) => DirectExecutorHandle::Closure {
                    call: close_copy,
                    context: std::ptr::null(),
                },
                _ => return Err(integrity("unexpected persisted test executor")),
            };
            Ok(ResolvedOnTheFlyExecutor {
                direct_executor_id,
                handle,
                parent_permutation: [0, 1],
                packed_singleton_capable: true,
                interaction_capability: None,
            })
        }

        fn bind_union_source_program(
            &self,
            view: OnTheFlyUnionSourceBindingView<'_>,
        ) -> RusticolResult<Box<dyn OnTheFlyBoundUnionSourceProgram>> {
            let selection = view
                .selections
                .first()
                .ok_or_else(|| integrity("persisted test source selection is absent"))?;
            self.source_bindings.set(self.source_bindings.get() + 1);
            Ok(Box::new(ProbeBoundUnionSourceProgram {
                value: selection.dispatch_variant_id as f64 + 1.0,
            }))
        }

        fn union_source_packed_singleton_capable(&self) -> bool {
            true
        }

        fn invalidate_row_tables(&self) -> RusticolResult<()> {
            self.invalidations.set(self.invalidations.get() + 1);
            Ok(())
        }
    }

    fn all_flow_plan() -> DirectRecurrencePlan {
        let mut parts = crate::recurrence::valid_direct_plan_parts_fixture();
        parts.strategy = RecurrenceStrategy::AllFlowUnion;
        parts.retained_helicity_count = 2;
        parts.runtime_helicity_contract_count = 1;
        parts.runtime_helicity_variant_count = 2;
        parts.sources[0].source_template_or_dispatch_domain = 0;
        parts.row_groups[0].direct_executor_id = DIRECT_NONE_U32;
        parts.replay_targets[0].helicity_map_count = 0;
        parts.replay_helicity_map.clear();
        parts.amplitude_destinations[0].target_helicity_id_or_sentinel = DIRECT_NONE_U32;
        parts.resolved_helicities = vec![
            DirectResolvedHelicityDescriptor {
                source_state_start: 0,
                source_selection_start: 0,
                public_helicity_start: 0,
                id: 0,
                source_state_count: 1,
                source_selection_count: 1,
                public_helicity_count: 1,
                selector_domain_id: 0,
            },
            DirectResolvedHelicityDescriptor {
                source_state_start: 1,
                source_selection_start: 1,
                public_helicity_start: 1,
                id: 1,
                source_state_count: 1,
                source_selection_count: 1,
                public_helicity_count: 1,
                selector_domain_id: 0,
            },
        ];
        parts.source_state_assignments = vec![
            crate::recurrence::DirectSourceStateAssignment {
                source_slot: 0,
                state_index: 0,
            },
            crate::recurrence::DirectSourceStateAssignment {
                source_slot: 0,
                state_index: 1,
            },
        ];
        parts.public_helicities = vec![-1, 1];
        parts.source_dispatch_variants = (0..2)
            .map(|id| DirectSourceDispatchVariantDescriptor {
                embedding_start: id * 2,
                projection_start: id * 2,
                source_row_id: 0,
                dispatch_domain_id: 0,
                runtime_variant_id: id as u32,
                source_state_index: id as u32,
                source_template_id: 0,
                source_state_template_id: 0,
                crossed_state_template_id: 0,
                crossed_spin_state_class: id as i32,
                direct_executor_id: 0,
                crossing_exact_factor_id: 0,
                embedding_count: 2,
                projection_count: 2,
            })
            .collect();
        parts.source_embeddings = vec![
            DirectSourceEmbeddingRow {
                full_component: 0,
                source_component_or_sentinel: 0,
                exact_factor_id: 0,
            },
            DirectSourceEmbeddingRow {
                full_component: 1,
                source_component_or_sentinel: 1,
                exact_factor_id: 0,
            },
            DirectSourceEmbeddingRow {
                full_component: 0,
                source_component_or_sentinel: 0,
                exact_factor_id: 0,
            },
            DirectSourceEmbeddingRow {
                full_component: 1,
                source_component_or_sentinel: 1,
                exact_factor_id: 0,
            },
        ];
        parts.source_projections = vec![
            DirectSourceProjectionRow {
                source_component: 0,
                full_component: 0,
            },
            DirectSourceProjectionRow {
                source_component: 1,
                full_component: 1,
            },
            DirectSourceProjectionRow {
                source_component: 0,
                full_component: 0,
            },
            DirectSourceProjectionRow {
                source_component: 1,
                full_component: 1,
            },
        ];
        parts.resolved_source_selections = vec![
            DirectResolvedSourceSelection {
                source_slot: 0,
                dispatch_variant_id: 0,
            },
            DirectResolvedSourceSelection {
                source_slot: 0,
                dispatch_variant_id: 1,
            },
        ];
        DirectRecurrencePlan::new(parts).unwrap()
    }

    fn all_flow_dispatch(plan: &DirectRecurrencePlan) -> DirectHelicityDispatch {
        DirectHelicityDispatch::new(DirectHelicityDispatchParts {
            runtime_layout_digest: plan.runtime_layout_digest(),
            source_row_count: 1,
            contribution_row_count: 1,
            finalization_row_count: 1,
            closure_row_count: 1,
            amplitude_destination_count: 1,
            direct_executor_count: 4,
            resolved_helicity_count: 2,
            support_domains: vec![DirectHelicitySupportDomainDescriptor {
                word_start: 0,
                word_count: 1,
            }],
            support_words: vec![0b01],
            row_groups: vec![
                DirectHelicityRowGroupDescriptor {
                    stage: 1,
                    role: DirectExecutorRole::Contribution,
                    direct_executor_id: 1,
                    row_start: 0,
                    row_count: 1,
                    support_domain_id: 0,
                },
                DirectHelicityRowGroupDescriptor {
                    stage: 1,
                    role: DirectExecutorRole::Finalization,
                    direct_executor_id: 2,
                    row_start: 0,
                    row_count: 1,
                    support_domain_id: 0,
                },
                DirectHelicityRowGroupDescriptor {
                    stage: 2,
                    role: DirectExecutorRole::Closure,
                    direct_executor_id: 3,
                    row_start: 0,
                    row_count: 1,
                    support_domain_id: 0,
                },
            ],
            dispatches: vec![
                DirectHelicityDispatchDescriptor {
                    resolved_helicity_id: 0,
                    group_id_start: 0,
                    group_id_count: 3,
                },
                DirectHelicityDispatchDescriptor {
                    resolved_helicity_id: 1,
                    group_id_start: 3,
                    group_id_count: 0,
                },
            ],
            dispatch_group_ids: vec![0, 1, 2],
        })
        .unwrap()
    }

    fn all_flow_plan_with_reused_component_base() -> DirectRecurrencePlan {
        let mut parts = all_flow_plan().into_parts();
        parts.current_arena_components = 2;
        parts.currents[0].last_use = 0;
        parts.currents[1].component_base = 0;
        parts.contributions[0].destination_component_base = 0;
        parts.finalizations[0].component_base = 0;
        parts.closures[0].parent0_component_base = 0;
        DirectRecurrencePlan::new(parts).unwrap()
    }

    #[test]
    fn persisted_census_distinguishes_reused_component_bases_by_liveness() {
        let plan = all_flow_plan_with_reused_component_base();
        let index = PersistedCurrentIndexV1::new(&plan).unwrap();
        assert_eq!(index.source_current(0, "test source").unwrap(), 0);
        assert_eq!(
            index.live_current(&plan, 0, 0, "test stage zero").unwrap(),
            0
        );
        assert_eq!(
            index.live_current(&plan, 1, 0, "test stage one").unwrap(),
            1
        );

        let dispatch = all_flow_dispatch(&plan);
        let family = build_family(&plan, &dispatch, 0).unwrap();
        assert_eq!(family.unique_current_count, 2);
        assert_eq!(family.unique_current_component_count, 3);
    }

    #[test]
    fn selected_rows_rederive_init_without_dropping_certified_reuse() {
        let mut groups = vec![
            PersistedFamilyGroupDraftV1 {
                stage: 1,
                direct_executor_id: DIRECT_NONE_U32,
                rows: PersistedFamilyRowsDraftV1::Contribution(vec![contribution(
                    8,
                    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE,
                )]),
            },
            PersistedFamilyGroupDraftV1 {
                stage: 1,
                direct_executor_id: 2,
                rows: PersistedFamilyRowsDraftV1::Contribution(vec![
                    contribution(8, DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION),
                    contribution(12, 0),
                ]),
            },
        ];
        rederive_contribution_initialization(&mut groups);
        let PersistedFamilyRowsDraftV1::Contribution(first) = &groups[0].rows else {
            panic!("wrong row kind")
        };
        let PersistedFamilyRowsDraftV1::Contribution(second) = &groups[1].rows else {
            panic!("wrong row kind")
        };
        assert_ne!(first[0].flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, 0);
        assert_ne!(
            first[0].flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            0
        );
        assert_eq!(
            second[0].flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            0
        );
        assert_ne!(
            second[1].flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            0
        );
    }

    #[test]
    fn point_major_batch_input_builds_asymmetric_momentum_planes() {
        let layout = test_layout();
        let mut workspace = PersistedHelicityWorkspaceV1::new(&layout, 2, false).unwrap();
        // [p0: s0=(1,2), s1=(3,4)], [p1: s0=(10,20), s1=(30,40)]
        workspace
            .refresh_inputs(&layout, &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 2)
            .unwrap();
        let stride = workspace.point_stride as usize;
        let momenta = workspace.momenta.as_slice();
        assert_eq!(&momenta[0..2], &[1.0, 10.0]);
        assert_eq!(&momenta[stride..stride + 2], &[2.0, 20.0]);
        assert_eq!(&momenta[2 * stride..2 * stride + 2], &[-3.0, -30.0]);
        assert_eq!(&momenta[3 * stride..3 * stride + 2], &[-4.0, -40.0]);
    }

    #[test]
    fn full_destination_clear_prevents_a_b_a_stale_outputs() {
        let layout = test_layout();
        let mut workspace = PersistedHelicityWorkspaceV1::new(&layout, 1, true).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];

        // H=A writes both destinations.
        workspace.refresh_inputs(&layout, &input, 1).unwrap();
        workspace.amplitude_re.as_mut_slice()[0] = 7.0;
        workspace.amplitude_re.as_mut_slice()[1] = 11.0;

        // H=B has no closure for destination 1; it must observe zero there.
        workspace.refresh_inputs(&layout, &input, 1).unwrap();
        workspace.amplitude_re.as_mut_slice()[0] = 13.0;
        assert_eq!(workspace.amplitude_re.as_slice(), &[13.0, 0.0]);
        assert_eq!(workspace.amplitude_im.as_slice(), &[0.0, 0.0]);

        // Returning to A also starts from a completely clean dense C tile.
        workspace.refresh_inputs(&layout, &input, 1).unwrap();
        workspace.amplitude_re.as_mut_slice()[1] = 17.0;
        assert_eq!(workspace.amplitude_re.as_slice(), &[0.0, 17.0]);
    }

    #[test]
    fn unprofiled_execution_defers_diagnostic_counts_until_inspection() {
        let plan = all_flow_plan();
        let dispatch = all_flow_dispatch(&plan);
        let resolver = ProbeResolver {
            invalidations: Cell::new(0),
            source_bindings: Cell::new(0),
        };
        let mut executor = PersistedHelicityFamilyExecutorV1::new(resolver);
        executor.set_parameter_planes(&[4.0], &[-2.0]).unwrap();
        assert!(!executor.prepare(&plan, &dispatch, 0, 4, 1).unwrap());

        let input = [10.0, 20.0, 30.0, 40.0];
        let tile = executor.execute_tile_unprofiled(&input, 1).unwrap();
        assert_eq!(tile.destination_re(0).unwrap(), &[1.0]);
        assert_eq!(
            executor.family.as_ref().unwrap().last_execution_report,
            Some(PersistedHelicityFamilyExecutionReportV1::default())
        );

        // Inspection reconstructs the exact logical census outside the warm
        // execution path even though the unprofiled marker has no counters.
        let census = executor.active_family_inspection_census().unwrap();
        assert_eq!(census.union_source_executor_call_groups, 1);
        assert_eq!(census.union_contribution_executor_call_groups, 1);
        assert_eq!(census.union_finalization_executor_call_groups, 1);
        assert_eq!(census.union_closure_executor_call_groups, 1);
        assert_eq!(census.union_source_rows, 1);
        assert_eq!(census.union_contribution_rows, 1);
        assert_eq!(census.union_finalization_rows, 1);
        assert_eq!(census.union_closure_rows, 1);

        let (_, report) = executor.execute_tile_profiled(&input, 1).unwrap();
        assert!(report.cache_hit);
        assert_eq!(report.source_calls, 1);
        assert_eq!(report.contribution_calls, 1);
        assert_eq!(report.finalization_calls, 1);
        assert_eq!(report.closure_calls, 1);
        assert_eq!(
            executor.family.as_ref().unwrap().last_execution_report,
            Some(report)
        );
        assert_eq!(executor.resolver().source_bindings.get(), 1);
    }

    #[test]
    fn persisted_executor_retains_one_h_and_keeps_warm_execution_borrowed() {
        let plan = all_flow_plan();
        let dispatch = all_flow_dispatch(&plan);
        dispatch.validate_for_plan(&plan).unwrap();
        let resolver = ProbeResolver {
            invalidations: Cell::new(0),
            source_bindings: Cell::new(0),
        };
        let mut executor = PersistedHelicityFamilyExecutorV1::new(resolver);

        executor.set_parameter_planes(&[3.0], &[-2.0]).unwrap();
        let parameter_pointer = executor.parameter_state.as_ptr();
        let parameter_capacity = executor.parameter_state.capacity();
        let parameter_version = executor.parameter_version;
        executor.set_parameter_planes(&[3.0], &[-2.0]).unwrap();
        assert_eq!(executor.parameter_state.as_ptr(), parameter_pointer);
        assert_eq!(executor.parameter_state.capacity(), parameter_capacity);
        assert_eq!(executor.parameter_version, parameter_version);
        executor.set_parameter_planes(&[4.0], &[-2.0]).unwrap();
        assert_ne!(executor.parameter_state.as_ptr(), parameter_pointer);
        assert_eq!(executor.parameter_version, parameter_version + 1);

        let input = [10.0, 20.0, 30.0, 40.0];
        assert!(!executor.prepare(&plan, &dispatch, 0, 4, 1).unwrap());
        assert_eq!(executor.resolver().source_bindings.get(), 1);
        assert_eq!(
            executor.cache_census(),
            PersistedHelicityFamilyCacheCensusV1 {
                resolved_helicity_count: 2,
                retained_family_count: 1,
                retained_row_count: 3,
                active_resolved_helicity_id: Some(0),
            }
        );
        let (tile, report) = executor.execute_tile_profiled(&input, 1).unwrap();
        assert_eq!(tile.resolved_helicity_id(), 0);
        assert_eq!(tile.destination_count(), 1);
        assert_eq!(tile.point_count(), 1);
        assert_eq!(tile.point_stride(), 1);
        assert_eq!(tile.destination_re(0).unwrap(), &[1.0]);
        assert_eq!(tile.destination_im(0).unwrap(), &[0.0]);
        assert_eq!(
            report,
            PersistedHelicityFamilyExecutionReportV1 {
                cache_hit: false,
                source_calls: 1,
                source_rows: 1,
                contribution_calls: 1,
                contribution_rows: 1,
                finalization_calls: 1,
                finalization_rows: 1,
                closure_calls: 1,
                closure_rows: 1,
            }
        );
        let census = executor.active_family_inspection_census().unwrap();
        assert_eq!(census.query_count, 1);
        assert!(census.union_unique_current_count > 0);
        assert!(census.union_unique_current_count <= census.union_unique_current_component_count);
        assert_eq!(census.union_source_rows, 1);
        assert_eq!(census.union_contribution_rows, 1);
        assert_eq!(census.union_finalization_rows, 1);
        assert_eq!(census.union_closure_rows, 1);
        assert_eq!(census.union_amplitude_destination_count, 1);
        assert_eq!(census.union_source_executor_call_groups, 1);
        assert_eq!(census.union_contribution_executor_call_groups, 1);
        assert_eq!(census.union_finalization_executor_call_groups, 1);
        assert_eq!(census.union_closure_executor_call_groups, 1);
        assert_eq!(census.semantic_executor_binding_count, 4);

        assert!(executor.prepare(&plan, &dispatch, 0, 4, 1).unwrap());
        assert_eq!(executor.resolver().source_bindings.get(), 1);
        let (tile, report) = executor.execute_tile_profiled(&input, 1).unwrap();
        assert!(report.cache_hit);
        assert_eq!(tile.destination_re(0).unwrap(), &[1.0]);

        // H=1 has exact source dispatch but no live non-source rows. Replacing
        // H=0 still retains one family and leaves the dense destination zero.
        assert!(!executor.prepare(&plan, &dispatch, 1, 4, 1).unwrap());
        assert_eq!(executor.resolver().source_bindings.get(), 2);
        assert_eq!(
            executor.cache_census(),
            PersistedHelicityFamilyCacheCensusV1 {
                resolved_helicity_count: 2,
                retained_family_count: 1,
                retained_row_count: 0,
                active_resolved_helicity_id: Some(1),
            }
        );
        let tile = executor.execute_tile_unprofiled(&input, 1).unwrap();
        assert_eq!(tile.destination_re(0).unwrap(), &[0.0]);
        assert_eq!(tile.destination_im(0).unwrap(), &[0.0]);
        let census = executor.active_family_inspection_census().unwrap();
        assert_eq!(census.union_source_rows, 1);
        assert_eq!(census.union_contribution_rows, 0);
        assert_eq!(census.union_finalization_rows, 0);
        assert_eq!(census.union_closure_rows, 0);
        assert_eq!(census.semantic_executor_binding_count, 1);

        // A -> B -> A is a cold switch, but never multiplies retained arenas.
        assert!(!executor.prepare(&plan, &dispatch, 0, 4, 1).unwrap());
        assert_eq!(executor.resolver().source_bindings.get(), 3);
        assert_eq!(executor.cache_census().retained_family_count, 1);
        let tile = executor.execute_tile_unprofiled(&input, 1).unwrap();
        assert_eq!(tile.destination_re(0).unwrap(), &[1.0]);
        assert!(executor.resolver().invalidations.get() >= 2);

        executor.clear_families().unwrap();
        assert_eq!(
            executor.cache_census(),
            PersistedHelicityFamilyCacheCensusV1::default()
        );
        assert!(executor.active_family_inspection_census().is_none());
    }
}
