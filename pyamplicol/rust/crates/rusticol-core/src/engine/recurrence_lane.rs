// SPDX-License-Identifier: 0BSD

//! Native adapter from public runtime inputs to Direct-Arena recurrence.

use super::recurrence_backend::{
    NativeOnTheFlyPreparedExecutorResolver, NativeRecurrenceDirectExecutorOwners,
};
use super::*;
use crate::direct_arena::DirectArenaTrafficCounters;
use crate::recurrence::direct_backend::{
    DirectExecutionCounters, DirectExecutionRoleTimings, DirectExecutorCatalog,
};
use crate::recurrence::direct_runtime::{
    DirectRecurrenceExecutionRuntime, DirectRecurrenceTileOutput, DirectReplaySelectorPlan,
    DirectRuntimeActivityCounters, DirectRuntimePhaseTimings, DirectUnionHelicitySelectorPlan,
};
use crate::recurrence::on_the_fly::{
    PersistedHelicityAmplitudeTileV1, PersistedHelicityFamilyCacheCensusV1,
    PersistedHelicityFamilyExecutionReportV1, PersistedHelicityFamilyExecutorV1,
    PersistedHelicityFamilyInspectionCensusV1,
};
use crate::recurrence::{
    DIRECT_NONE_U32, DirectAmplitudeDestinationDescriptor, DirectHelicityDispatch,
    DirectRecurrencePlan, RecurrenceColorContraction, RecurrenceColorStorage, RecurrenceStrategy,
    RuntimeColorContractionEntry, RuntimeColorContractionReducer,
    RuntimeSymmetricGroupColorWorkspace,
};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(super) struct PreparedParameterProjectionEntry {
    pub(super) runtime_slot: usize,
    pub(super) prepared_slot: usize,
    pub(super) component: u8,
}

/// One process-local Direct-Arena runtime.
///
/// `_backend_owner` keeps every immutable source/SymJIT context addressed by
/// the scheduler's lightweight function handles alive. All point-dependent
/// storage belongs to `scheduler` or the fixed-size external-momentum scratch.
pub(super) struct RecurrenceNativeRuntime {
    scheduler: DirectRecurrenceExecutionRuntime,
    _backend_owner: NativeRecurrenceDirectExecutorOwners,
    backend_name: String,
    selectors: RecurrenceNativeSelectors,
    parameter_defaults: Vec<crate::EagerComplex64>,
    parameter_projection: Vec<PreparedParameterProjectionEntry>,
    external_source_count: usize,
    effective_point_tile_size: usize,
    external_momenta: Vec<f64>,
    contracted_replay_re: Vec<f64>,
    contracted_replay_im: Vec<f64>,
    color_transform_re: Vec<f64>,
    color_transform_im: Vec<f64>,
    symmetric_group_color_workspace: Option<RuntimeSymmetricGroupColorWorkspace>,
    replay_color_index_scratch: Vec<usize>,
    selected_replay_cache:
        BTreeMap<BTreeSet<String>, BTreeMap<usize, Arc<DirectReplaySelectorPlan>>>,
    selected_replay_programs: BTreeMap<(u32, Vec<u32>), Arc<DirectReplaySelectorPlan>>,
    helicity_selector_companion: Option<RecurrencePersistedHelicitySelectorRuntime>,
}

pub(super) struct RecurrencePersistedHelicitySelectorRuntime {
    plan: DirectRecurrencePlan,
    dispatch: DirectHelicityDispatch,
    executor: PersistedHelicityFamilyExecutorV1<NativeOnTheFlyPreparedExecutorResolver>,
    resolved_helicity_by_physics: Box<[u32]>,
    primary_color_view: Option<PrimaryLocalColorViewV1>,
}

struct PrimaryLocalColorViewV1 {
    auxiliary_destination_by_local_group: Box<[u32]>,
    local_group_by_primary_destination: Box<[u32]>,
}

impl RecurrencePersistedHelicitySelectorRuntime {
    pub(super) fn new(
        plan: DirectRecurrencePlan,
        dispatch: DirectHelicityDispatch,
        resolver: NativeOnTheFlyPreparedExecutorResolver,
        direct_helicity_to_physics: Vec<usize>,
        physics_helicity_count: usize,
    ) -> RusticolResult<Self> {
        if plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(RusticolError::integrity(
                "persisted recurrence helicity selector is not all-flow-union",
            ));
        }
        dispatch.validate_for_plan(&plan)?;
        if direct_helicity_to_physics.len() != plan.resolved_helicities().len() {
            return Err(RusticolError::integrity(
                "persisted selector helicity mapping does not cover its auxiliary plan",
            ));
        }
        let mut resolved_helicity_by_physics = vec![DIRECT_NONE_U32; physics_helicity_count];
        for (resolved_helicity_id, physics_helicity_id) in
            direct_helicity_to_physics.into_iter().enumerate()
        {
            let slot = resolved_helicity_by_physics
                .get_mut(physics_helicity_id)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "persisted selector helicity maps outside the public physics axis",
                    )
                })?;
            if *slot != DIRECT_NONE_U32 {
                return Err(RusticolError::integrity(
                    "persisted selector helicity mapping repeats a public helicity",
                ));
            }
            *slot = u32::try_from(resolved_helicity_id).map_err(|_| {
                RusticolError::artifact("persisted selector resolved helicity exceeds u32")
            })?;
        }
        if resolved_helicity_by_physics.contains(&DIRECT_NONE_U32) {
            return Err(RusticolError::integrity(
                "persisted selector does not cover the complete public helicity axis",
            ));
        }
        Ok(Self {
            plan,
            dispatch,
            executor: PersistedHelicityFamilyExecutorV1::new(resolver),
            resolved_helicity_by_physics: resolved_helicity_by_physics.into_boxed_slice(),
            primary_color_view: None,
        })
    }

    fn bind_primary_color_view(
        &mut self,
        contraction: &RecurrenceColorContraction,
    ) -> RusticolResult<()> {
        if self.primary_color_view.is_some() {
            return Err(RusticolError::internal(
                "persisted selector primary color view was bound twice",
            ));
        }
        self.primary_color_view = Some(primary_local_color_destination_map(
            contraction,
            self.plan.amplitude_destinations(),
        )?);
        Ok(())
    }

    #[inline(always)]
    fn resolved_helicity_id(&self, physics_helicity_id: usize) -> RusticolResult<u32> {
        self.resolved_helicity_by_physics
            .get(physics_helicity_id)
            .copied()
            .ok_or_else(|| {
                RusticolError::integrity(
                    "persisted selector public helicity is outside its resolved map",
                )
            })
    }

    fn clear(&mut self) -> RusticolResult<()> {
        self.executor.clear_families()
    }

    fn cache_census(&self) -> PersistedHelicityFamilyCacheCensusV1 {
        self.executor.cache_census()
    }

    fn active_family_inspection_census(&self) -> Option<PersistedHelicityFamilyInspectionCensusV1> {
        self.executor.active_family_inspection_census()
    }
}

fn primary_local_color_destination_map(
    contraction: &RecurrenceColorContraction,
    auxiliary_destinations: &[DirectAmplitudeDestinationDescriptor],
) -> RusticolResult<PrimaryLocalColorViewV1> {
    if contraction.storage() == RecurrenceColorStorage::Expanded
        && contraction.component_count() != 1
    {
        return Err(RusticolError::compatibility(
            "expanded persisted helicity color view must contain exactly one component",
        ));
    }
    let component_count = contraction.component_count() as usize;
    let local_group_count = match contraction.storage() {
        RecurrenceColorStorage::Expanded => contraction.group_count() as usize,
        RecurrenceColorStorage::Repeated | RecurrenceColorStorage::ConvolutionKernels => {
            contraction.local_group_count() as usize
        }
    };
    if local_group_count == 0
        || component_count == 0
        || auxiliary_destinations.len() != local_group_count
    {
        return Err(RusticolError::integrity(
            "persisted selector physical destinations disagree with the primary local color domain",
        ));
    }

    let active_owner_sectors = contraction
        .owner_by_sector()
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(sector, owner)| (owner as usize == sector).then_some(sector))
        .collect::<Vec<_>>();
    if active_owner_sectors.len() != local_group_count {
        return Err(RusticolError::integrity(
            "primary local color groups do not match its canonical active owners",
        ));
    }
    let mut dense_rank_by_sector = vec![DIRECT_NONE_U32; contraction.owner_by_sector().len()];
    for (rank, sector) in active_owner_sectors.into_iter().enumerate() {
        dense_rank_by_sector[sector] = u32::try_from(rank)
            .map_err(|_| RusticolError::artifact("primary color owner rank exceeds u32"))?;
    }

    let mut auxiliary_destination_by_rank = vec![DIRECT_NONE_U32; local_group_count];
    for destination in auxiliary_destinations {
        if destination.target_helicity_id_or_sentinel != DIRECT_NONE_U32 {
            return Err(RusticolError::integrity(
                "persisted selector auxiliary destination fixes a numerical helicity",
            ));
        }
        let slot = auxiliary_destination_by_rank
            .get_mut(destination.target_sector_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "persisted selector auxiliary destination has a non-dense physical sector",
                )
            })?;
        if *slot != DIRECT_NONE_U32 {
            return Err(RusticolError::integrity(
                "persisted selector repeats an auxiliary physical sector",
            ));
        }
        *slot = destination.id;
    }
    if auxiliary_destination_by_rank.contains(&DIRECT_NONE_U32) {
        return Err(RusticolError::integrity(
            "persisted selector does not cover every dense auxiliary physical sector",
        ));
    }

    let ordered_groups = contraction.ordered_group_ids();
    let sectors = contraction.sector_by_group();
    let components = contraction.component_by_group();
    let destinations = contraction.destination_by_group();
    let mut auxiliary_destination_by_local_group = Vec::with_capacity(local_group_count);
    let mut local_group_by_primary_destination =
        vec![DIRECT_NONE_U32; contraction.destination_count() as usize];
    for local_group in 0..local_group_count {
        let mut owner_sector = None;
        for component in 0..component_count {
            let ordered_index = local_group
                .checked_mul(component_count)
                .and_then(|base| base.checked_add(component))
                .ok_or_else(|| RusticolError::artifact("primary local color index overflows"))?;
            let group_id = *ordered_groups
                .get(ordered_index)
                .ok_or_else(|| RusticolError::integrity("primary ordered color group is absent"))?
                as usize;
            let sector = *sectors.get(group_id).ok_or_else(|| {
                RusticolError::integrity("primary color group has no physical sector")
            })? as usize;
            if components.get(group_id).copied() != Some(component as u32)
                || contraction.owner_by_sector().get(sector).copied() != Some(sector as u32)
                || owner_sector.is_some_and(|expected| expected != sector)
            {
                return Err(RusticolError::integrity(
                    "primary local color block mixes components or non-owner sectors",
                ));
            }
            owner_sector = Some(sector);
            let primary_destination = *destinations.get(group_id).ok_or_else(|| {
                RusticolError::integrity("primary color group has no amplitude destination")
            })? as usize;
            let slot = local_group_by_primary_destination
                .get_mut(primary_destination)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "primary color destination is outside its authenticated domain",
                    )
                })?;
            let local_group_u32 = u32::try_from(local_group)
                .map_err(|_| RusticolError::artifact("primary local color group exceeds u32"))?;
            if *slot != DIRECT_NONE_U32 && *slot != local_group_u32 {
                return Err(RusticolError::integrity(
                    "primary color destination maps to multiple local groups",
                ));
            }
            *slot = local_group_u32;
        }
        let owner_sector = owner_sector.expect("positive component count was validated");
        let dense_rank = *dense_rank_by_sector.get(owner_sector).ok_or_else(|| {
            RusticolError::integrity("primary color owner has no dense canonical rank")
        })?;
        if dense_rank == DIRECT_NONE_U32 {
            return Err(RusticolError::integrity(
                "primary color local group references an inactive owner",
            ));
        }
        auxiliary_destination_by_local_group
            .push(auxiliary_destination_by_rank[dense_rank as usize]);
    }
    Ok(PrimaryLocalColorViewV1 {
        auxiliary_destination_by_local_group: auxiliary_destination_by_local_group
            .into_boxed_slice(),
        local_group_by_primary_destination: local_group_by_primary_destination.into_boxed_slice(),
    })
}

struct ContractedReplayRoute {
    selector: DirectReplaySelectorPlan,
    destination_copies: Vec<(u32, u32)>,
}

// Boxing the contracted-color payload would add an indirection to every
// contracted recurrence tile; this enum is instantiated only once per runtime.
#[allow(clippy::large_enum_variant)]
enum RecurrenceNativeSelectors {
    TopologyReplay {
        replay_selectors: Vec<DirectReplaySelectorPlan>,
        direct_helicity_to_physics: Vec<usize>,
    },
    AllFlowUnion {
        helicity_selectors_by_physics: Vec<Option<DirectUnionHelicitySelectorPlan>>,
        destination_by_public_flow: Vec<u32>,
    },
    ContractedColorUnion {
        contraction: RecurrenceColorContraction,
        destination_physics_helicity: Vec<usize>,
        replay_routes: Vec<ContractedReplayRoute>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompanionSingletonSelection {
    StructuralZero,
    Runnable {
        physics_helicity_id: usize,
        resolved_helicity_id: u32,
    },
}

/// Borrowed LC selector and reduction state shared by topology replay and the
/// future compact on-the-fly lane.
///
/// Constructing this view is allocation-free. It deliberately exposes only
/// the public axes and orbit weights needed after an amplitude is computed;
/// an on-the-fly runtime can therefore provide the same backing slices/maps
/// without materializing a dense [`ProcessPhysicsV1`].
#[derive(Clone, Copy)]
pub(super) struct LcSelectorReductionView<'a> {
    helicities: &'a [crate::Helicity],
    color_components: &'a [crate::ColorComponent],
    helicity_index_by_id: &'a BTreeMap<String, usize>,
    color_index_by_id: &'a BTreeMap<String, usize>,
    helicity_members_by_representative: &'a [Vec<usize>],
}

impl<'a> LcSelectorReductionView<'a> {
    pub(super) const fn from_parts(
        helicities: &'a [crate::Helicity],
        color_components: &'a [crate::ColorComponent],
        helicity_index_by_id: &'a BTreeMap<String, usize>,
        color_index_by_id: &'a BTreeMap<String, usize>,
        helicity_members_by_representative: &'a [Vec<usize>],
    ) -> Self {
        Self {
            helicities,
            color_components,
            helicity_index_by_id,
            color_index_by_id,
            helicity_members_by_representative,
        }
    }

    fn validate_selector_ids(
        self,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<()> {
        if let Some(id) = selected_helicities.and_then(|ids| {
            ids.iter()
                .find(|id| !self.helicity_index_by_id.contains_key(*id))
        }) {
            return Err(RusticolError::selector(format!(
                "unknown resolved helicity id {id:?}"
            )));
        }
        if let Some(id) = selected_colors.and_then(|ids| {
            ids.iter()
                .find(|id| !self.color_index_by_id.contains_key(*id))
        }) {
            return Err(RusticolError::selector(format!(
                "unknown resolved color component id {id:?}"
            )));
        }
        Ok(())
    }

    fn selected_indices(
        available: &BTreeMap<String, usize>,
        ids: Option<&BTreeSet<String>>,
        kind: &str,
    ) -> RusticolResult<Vec<usize>> {
        let mut indices = if let Some(ids) = ids {
            ids.iter()
                .map(|id| {
                    available.get(id).copied().ok_or_else(|| {
                        RusticolError::selector(format!("unknown resolved {kind} id {id:?}"))
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?
        } else {
            (0..available.len()).collect()
        };
        indices.sort_unstable();
        Ok(indices)
    }

    fn selected_helicity_indices(
        self,
        ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Vec<usize>> {
        Self::selected_indices(self.helicity_index_by_id, ids, "helicity")
    }

    fn selected_color_indices(self, ids: Option<&BTreeSet<String>>) -> RusticolResult<Vec<usize>> {
        Self::selected_indices(self.color_index_by_id, ids, "color component")
    }

    fn retain_selected_color_indices(
        self,
        ids: &BTreeSet<String>,
        indices: &mut Vec<usize>,
    ) -> RusticolResult<()> {
        indices.clear();
        indices.reserve(ids.len());
        for id in ids {
            indices.push(*self.color_index_by_id.get(id).ok_or_else(|| {
                RusticolError::selector(format!("unknown resolved color component id {id:?}"))
            })?);
        }
        indices.sort_unstable();
        Ok(())
    }

    #[inline(always)]
    fn helicity(self, index: usize) -> &'a crate::Helicity {
        &self.helicities[index]
    }

    #[inline(always)]
    fn color(self, index: usize) -> &'a crate::ColorComponent {
        &self.color_components[index]
    }

    #[inline(always)]
    fn color_is_computed(self, index: usize) -> bool {
        match self.color(index) {
            crate::ColorComponent::LcFlow(flow) => flow.computed,
            crate::ColorComponent::ContractedColor(_) => true,
        }
    }

    #[inline(always)]
    fn helicity_orbit_members(self, representative: usize) -> &'a [usize] {
        self.helicity_members_by_representative
            .get(representative)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    #[inline(always)]
    fn helicity_is_selected(self, selected: Option<&BTreeSet<String>>, index: usize) -> bool {
        selected.is_none_or(|ids| ids.contains(&self.helicity(index).id))
    }

    #[inline(always)]
    fn helicity_orbit_weight(
        self,
        selected: Option<&BTreeSet<String>>,
        representative_index: usize,
    ) -> f64 {
        self.helicity_orbit_members(representative_index)
            .iter()
            .copied()
            .filter(|index| self.helicity_is_selected(selected, *index))
            .map(|index| self.helicity(index).coefficient)
            .sum()
    }

    #[cfg(test)]
    #[inline(always)]
    fn color_is_selected(self, selected: Option<&BTreeSet<String>>, index: usize) -> bool {
        selected.is_none_or(|ids| ids.contains(self.color(index).id()))
    }
}

#[inline(always)]
fn visit_replay_color_indices(
    color_count: usize,
    selected: Option<&[usize]>,
    mut visit: impl FnMut(usize) -> RusticolResult<()>,
) -> RusticolResult<()> {
    if let Some(selected) = selected {
        for &color_index in selected {
            visit(color_index)?;
        }
    } else {
        for color_index in 0..color_count {
            visit(color_index)?;
        }
    }
    Ok(())
}

impl PhysicsRuntime {
    pub(super) fn lc_selector_reduction_view(&self) -> LcSelectorReductionView<'_> {
        LcSelectorReductionView::from_parts(
            &self.manifest.helicities,
            &self.manifest.color_components,
            &self.helicity_index_by_id,
            &self.color_index_by_id,
            &self.helicity_members_by_representative,
        )
    }
}

/// Apply the established LC diagonal reduction to one complex amplitude
/// stream. Both topology replay's split real/imaginary planes and the future
/// on-the-fly query-major output can provide their native storage through the
/// inlined accessor without copying or changing the reduction formula.
#[inline(always)]
pub(super) fn accumulate_lc_diagonal_amplitude(
    point_count: usize,
    weight: f64,
    mut value: impl FnMut(usize) -> (f64, f64),
    mut accumulate: impl FnMut(usize, f64),
) {
    for point in 0..point_count {
        let (real, imaginary) = value(point);
        accumulate(point, weight * real.mul_add(real, imaginary * imaginary));
    }
}

/// Initialize prepared complex parameters from authenticated defaults and the
/// mutable runtime projection shared by recurrence and on-the-fly execution.
/// Every fallible shape/index check completes before either destination plane
/// is modified.
pub(super) fn initialize_prepared_parameter_planes(
    defaults: &[crate::EagerComplex64],
    projection: &[PreparedParameterProjectionEntry],
    runtime_values: &[f64],
    destination_re: &mut [f64],
    destination_im: &mut [f64],
) -> RusticolResult<()> {
    if destination_re.len() != defaults.len() || destination_im.len() != defaults.len() {
        return Err(RusticolError::integrity(
            "prepared parameter workspace has the wrong size",
        ));
    }
    for entry in projection {
        if entry.prepared_slot >= defaults.len()
            || entry.runtime_slot >= runtime_values.len()
            || entry.component > 1
        {
            return Err(RusticolError::integrity(
                "runtime parameter projection is outside its prepared layout",
            ));
        }
    }
    for ((real, imaginary), default) in destination_re
        .iter_mut()
        .zip(destination_im.iter_mut())
        .zip(defaults)
    {
        *real = default.re;
        *imaginary = default.im;
    }
    for entry in projection {
        let value = runtime_values[entry.runtime_slot];
        if entry.component == 0 {
            destination_re[entry.prepared_slot] = value;
        } else {
            destination_im[entry.prepared_slot] = value;
        }
    }
    Ok(())
}

pub(super) fn projected_prepared_parameter_values(
    defaults: &[crate::EagerComplex64],
    projection: &[PreparedParameterProjectionEntry],
    runtime_values: &[f64],
) -> RusticolResult<Vec<(f64, f64)>> {
    let mut real = vec![0.0; defaults.len()];
    let mut imaginary = vec![0.0; defaults.len()];
    initialize_prepared_parameter_planes(
        defaults,
        projection,
        runtime_values,
        &mut real,
        &mut imaginary,
    )?;
    Ok(real.into_iter().zip(imaginary).collect())
}

/// Existing point-major resolved-output placement shared by recurrence and
/// on-the-fly LC execution. The selected axis order remains
/// `[point][helicity][color]` without requiring a persisted dense flow table.
#[derive(Clone, Copy, Debug)]
pub(super) struct LcResolvedOutputLayout {
    point_count: usize,
    helicity_count: usize,
    color_count: usize,
}

impl LcResolvedOutputLayout {
    pub(super) fn new(
        point_count: usize,
        helicity_count: usize,
        color_count: usize,
    ) -> RusticolResult<Self> {
        if point_count == 0 || helicity_count == 0 || color_count == 0 {
            return Err(RusticolError::invalid_argument(
                "resolved LC output shape is empty",
            ));
        }
        let layout = Self {
            point_count,
            helicity_count,
            color_count,
        };
        layout.output_len()?;
        Ok(layout)
    }

    pub(super) fn output_len(self) -> RusticolResult<usize> {
        self.point_count
            .checked_mul(self.helicity_count)
            .and_then(|count| count.checked_mul(self.color_count))
            .ok_or_else(|| RusticolError::invalid_argument("resolved LC output shape overflows"))
    }

    #[inline(always)]
    pub(super) fn index(
        self,
        point: usize,
        helicity: usize,
        color: usize,
    ) -> RusticolResult<usize> {
        if point >= self.point_count || helicity >= self.helicity_count || color >= self.color_count
        {
            return Err(RusticolError::integrity(
                "resolved LC output coordinate is outside its selected axes",
            ));
        }
        point
            .checked_mul(self.helicity_count)
            .and_then(|base| base.checked_add(helicity))
            .and_then(|base| base.checked_mul(self.color_count))
            .and_then(|base| base.checked_add(color))
            .ok_or_else(|| RusticolError::invalid_argument("resolved LC output index overflows"))
    }
}

impl RecurrenceNativeRuntime {
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub(super) fn new(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        backend_owner: NativeRecurrenceDirectExecutorOwners,
        parameter_defaults: Vec<crate::EagerComplex64>,
        parameter_projection: Vec<PreparedParameterProjectionEntry>,
        public_flow_ids: Vec<u32>,
        direct_helicity_to_physics: Vec<usize>,
        color_contraction: Option<RecurrenceColorContraction>,
        mut helicity_selector_companion: Option<RecurrencePersistedHelicitySelectorRuntime>,
    ) -> RusticolResult<Self> {
        let parameter_count = usize::try_from(plan.parameter_value_count())
            .map_err(|_| RusticolError::artifact("recurrence parameter count exceeds usize"))?;
        if parameter_defaults.len() != parameter_count {
            return Err(RusticolError::integrity(format!(
                "recurrence prepared defaults have length {}, expected {parameter_count}",
                parameter_defaults.len()
            )));
        }
        for entry in &parameter_projection {
            if entry.prepared_slot >= parameter_count || entry.component > 1 {
                return Err(RusticolError::artifact(
                    "recurrence parameter projection is outside the prepared layout",
                ));
            }
        }
        if direct_helicity_to_physics.len() != plan.resolved_helicities().len() {
            return Err(RusticolError::integrity(
                "recurrence resolved-helicity mapping does not cover the direct plan",
            ));
        }
        let backend_name = backend_owner.summary().backend.clone();
        let strategy = plan.strategy();
        let has_color_contraction = color_contraction.is_some();
        if public_flow_ids.is_empty() && strategy != RecurrenceStrategy::ContractedColorUnion {
            return Err(RusticolError::integrity(
                "recurrence public color-flow mapping is empty",
            ));
        }
        let scheduler = match strategy {
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => {
                DirectRecurrenceExecutionRuntime::new(plan, executors, 4)?
            }
            RecurrenceStrategy::AllFlowUnion => {
                let dispatch = backend_owner.union_source_dispatch()?;
                DirectRecurrenceExecutionRuntime::new_with_union_source_dispatch(
                    plan, executors, 4, dispatch,
                )?
            }
        };
        let point_tile_capacity = scheduler.point_tile_size() as usize;
        let mut effective_point_tile_size = point_tile_capacity;
        let mut color_transform_scratch_len = 0usize;
        let mut symmetric_group_color_workspace = None;
        if let Some(contraction) = color_contraction.as_ref() {
            match contraction.runtime_reducer() {
                Some(RuntimeColorContractionReducer::Walsh(_)) => {
                    color_transform_scratch_len = usize::try_from(contraction.local_group_count())
                        .ok()
                        .and_then(|groups| groups.checked_mul(point_tile_capacity))
                        .ok_or_else(|| {
                            RusticolError::artifact(
                                "recurrence factorized color workspace overflows usize",
                            )
                        })?;
                }
                Some(RuntimeColorContractionReducer::SymmetricGroupFourier(reducer)) => {
                    (effective_point_tile_size, symmetric_group_color_workspace) =
                        lazy_symmetric_group_workspace_for_point_tile(
                            reducer,
                            point_tile_capacity,
                        )?;
                }
                None => {}
            }
        }
        let selectors = match strategy {
            RecurrenceStrategy::TopologyReplay => {
                let replay_selectors = public_flow_ids
                    .into_iter()
                    .map(|public_flow_id| scheduler.prepare_replay_selector(public_flow_id))
                    .collect::<RusticolResult<Vec<_>>>()?;
                validate_replay_destination_helicity_mappings(
                    scheduler.plan(),
                    &replay_selectors,
                    &direct_helicity_to_physics,
                )?;
                RecurrenceNativeSelectors::TopologyReplay {
                    replay_selectors,
                    direct_helicity_to_physics,
                }
            }
            RecurrenceStrategy::AllFlowUnion => {
                let physics_helicity_count = direct_helicity_to_physics
                    .iter()
                    .copied()
                    .max()
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "all-flow-union recurrence has no public helicities",
                        )
                    })?;
                let mut helicity_selectors_by_physics = vec![None; physics_helicity_count];
                for (direct_id, physics_id) in
                    direct_helicity_to_physics.iter().copied().enumerate()
                {
                    let selector = scheduler.prepare_union_helicity_selector(
                        u32::try_from(direct_id).map_err(|_| {
                            RusticolError::artifact("recurrence direct helicity ID exceeds u32")
                        })?,
                    )?;
                    if helicity_selectors_by_physics[physics_id]
                        .replace(selector)
                        .is_some()
                    {
                        return Err(RusticolError::integrity(
                            "all-flow-union recurrence repeats a public helicity",
                        ));
                    }
                }
                if helicity_selectors_by_physics.iter().any(Option::is_none) {
                    return Err(RusticolError::integrity(
                        "all-flow-union recurrence does not cover the public helicity axis",
                    ));
                }
                let destination_by_public_flow =
                    union_destination_ids(scheduler.plan(), &public_flow_ids)?;
                RecurrenceNativeSelectors::AllFlowUnion {
                    helicity_selectors_by_physics,
                    destination_by_public_flow,
                }
            }
            RecurrenceStrategy::ContractedColorUnion => {
                if !public_flow_ids.is_empty() {
                    return Err(RusticolError::integrity(
                        "contracted recurrence unexpectedly exposes public color flows",
                    ));
                }
                let contraction = color_contraction.ok_or_else(|| {
                    RusticolError::integrity(
                        "contracted recurrence has no color-contraction payload",
                    )
                })?;
                let destination_physics_helicity =
                    contracted_destination_helicity_map(&contraction, &direct_helicity_to_physics)?;
                let replay_routes = contracted_replay_routes(&scheduler, &contraction)?;
                RecurrenceNativeSelectors::ContractedColorUnion {
                    contraction,
                    destination_physics_helicity,
                    replay_routes,
                }
            }
        };
        if strategy != RecurrenceStrategy::ContractedColorUnion && has_color_contraction {
            return Err(RusticolError::integrity(
                "LC recurrence unexpectedly carries a color-contraction payload",
            ));
        }
        if helicity_selector_companion.is_some()
            && strategy != RecurrenceStrategy::ContractedColorUnion
        {
            return Err(RusticolError::integrity(
                "recurrence helicity-selector companion requires contracted-color-union",
            ));
        }
        if let Some(companion) = &mut helicity_selector_companion {
            if companion.plan.parameter_value_count() != scheduler.plan().parameter_value_count()
                || companion.plan.prepared_pack_digest() != scheduler.plan().prepared_pack_digest()
                || companion.plan.direct_template_catalog_digest()
                    != scheduler.plan().direct_template_catalog_digest()
            {
                return Err(RusticolError::integrity(
                    "persisted selector prepared-parameter layout disagrees with the primary recurrence plan",
                ));
            }
            let RecurrenceNativeSelectors::ContractedColorUnion { contraction, .. } = &selectors
            else {
                return Err(RusticolError::integrity(
                    "recurrence helicity-selector companion has no primary color contraction",
                ));
            };
            companion.bind_primary_color_view(contraction)?;
        }
        let external_source_count = usize::try_from(scheduler.plan().external_source_count())
            .map_err(|_| RusticolError::artifact("recurrence source count exceeds usize"))?;
        scheduler
            .point_tile_size()
            .try_into()
            .ok()
            .and_then(|points: usize| points.checked_mul(external_source_count))
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| {
                RusticolError::artifact("recurrence external-momentum workspace overflows usize")
            })?;
        let mut external_momenta = Vec::new();
        ensure_external_momentum_workspace_capacity(
            &mut external_momenta,
            1,
            external_source_count,
            effective_point_tile_size,
        )?;
        let contracted_replay_len = match &selectors {
            RecurrenceNativeSelectors::ContractedColorUnion {
                contraction,
                replay_routes,
                ..
            } if !replay_routes.is_empty() => usize::try_from(contraction.destination_count())
                .ok()
                .and_then(|destinations| {
                    usize::try_from(scheduler.point_tile_size())
                        .ok()
                        .and_then(|points| destinations.checked_mul(points))
                })
                .ok_or_else(|| {
                    RusticolError::artifact("contracted replay amplitude workspace overflows usize")
                })?,
            _ => 0,
        };

        Ok(Self {
            scheduler,
            _backend_owner: backend_owner,
            backend_name,
            selectors,
            parameter_defaults,
            parameter_projection,
            external_source_count,
            effective_point_tile_size,
            external_momenta,
            contracted_replay_re: vec![0.0; contracted_replay_len],
            contracted_replay_im: vec![0.0; contracted_replay_len],
            color_transform_re: vec![0.0; color_transform_scratch_len],
            color_transform_im: vec![0.0; color_transform_scratch_len],
            symmetric_group_color_workspace,
            replay_color_index_scratch: Vec::new(),
            selected_replay_cache: BTreeMap::new(),
            selected_replay_programs: BTreeMap::new(),
            helicity_selector_companion,
        })
    }

    pub(super) fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub(super) fn effective_point_tile_size(&self) -> usize {
        self.effective_point_tile_size
    }

    pub(super) fn clear_helicity_selector_companion(&mut self) -> RusticolResult<()> {
        if let Some(companion) = &mut self.helicity_selector_companion {
            companion.clear()?;
        }
        Ok(())
    }

    pub(super) fn helicity_selector_companion_state_census(
        &self,
        process_id: &str,
    ) -> RusticolResult<Option<serde_json::Value>> {
        let Some(companion) = &self.helicity_selector_companion else {
            return Ok(None);
        };
        Ok(Some(Self::persisted_helicity_selector_state_census(
            process_id,
            companion.cache_census(),
            companion.active_family_inspection_census(),
        )))
    }

    fn persisted_helicity_selector_state_census(
        process_id: &str,
        cache: PersistedHelicityFamilyCacheCensusV1,
        active: Option<PersistedHelicityFamilyInspectionCensusV1>,
    ) -> serde_json::Value {
        let active_family_union_census = active.map(|census| {
            serde_json::json!({
                "basis": "shared-query-family-union-v1",
                "scope": "active-family-union",
                "query_count": census.query_count,
                "union_unique_current_count": census.union_unique_current_count,
                "union_unique_current_component_count": (
                    census.union_unique_current_component_count
                ),
                "union_source_rows": census.union_source_rows,
                "union_contribution_rows": census.union_contribution_rows,
                "union_finalization_rows": census.union_finalization_rows,
                "union_closure_rows": census.union_closure_rows,
                "union_amplitude_destination_count": (
                    census.union_amplitude_destination_count
                ),
                "union_source_executor_call_groups": (
                    census.union_source_executor_call_groups
                ),
                "union_contribution_executor_call_groups": (
                    census.union_contribution_executor_call_groups
                ),
                "union_finalization_executor_call_groups": (
                    census.union_finalization_executor_call_groups
                ),
                "union_closure_executor_call_groups": (
                    census.union_closure_executor_call_groups
                ),
            })
        });
        let retained_request_count = active.map_or(0, |census| census.query_count);
        let retained_amplitude_destination_count =
            active.map_or(0, |census| census.union_amplitude_destination_count);
        let retained_executor_handle_count = cache.retained_family_count;
        let semantic_executor_binding_count =
            active.map_or(0, |census| census.semantic_executor_binding_count);
        serde_json::json!({
            "kind": "rusticol-on-the-fly-runtime-state-census-v1",
            "process_id": process_id,
            "family_cache_policy": "last-family-only",
            "family_cache_limit": 1,
            "process_preparation_count": cache.retained_family_count,
            "retained_family_count": cache.retained_family_count,
            "pending_family_count": 0,
            "retained_selection_count": cache.retained_family_count,
            "retained_request_count": retained_request_count,
            "retained_amplitude_destination_count": retained_amplitude_destination_count,
            "retained_executor_handle_count": retained_executor_handle_count,
            "retained_query_local_trace_count": 0,
            "retained_embedded_lookup_key_count": 0,
            "semantic_executor_binding_count": semantic_executor_binding_count,
            "active_family_union_census": active_family_union_census,
        })
    }

    pub(super) fn validate_global_selectors(
        &self,
        physics: &PhysicsRuntime,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<()> {
        self.validate_public_axis_lengths(physics)?;
        if matches!(
            &self.selectors,
            RecurrenceNativeSelectors::TopologyReplay { .. }
        ) {
            physics
                .lc_selector_reduction_view()
                .validate_selector_ids(selected_helicities, selected_colors)?;
        } else {
            validate_recurrence_selector_ids(physics, selected_helicities, selected_colors)?;
        }
        if matches!(
            &self.selectors,
            RecurrenceNativeSelectors::ContractedColorUnion { .. }
        ) {
            reject_contracted_color_selector(selected_colors)?;
        }
        Ok(())
    }

    /// Classify the exact singleton selector served by the complete compact
    /// companion. `Some(false)` is an authenticated structural zero and must
    /// return without preparing either execution lane.
    fn companion_singleton_selection(
        &self,
        physics: &PhysicsRuntime,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Option<CompanionSingletonSelection>> {
        if self.helicity_selector_companion.is_none()
            || !matches!(
                &self.selectors,
                RecurrenceNativeSelectors::ContractedColorUnion { .. }
            )
            || selected_colors.is_some()
        {
            return Ok(None);
        }
        let Some(selected_helicities) = selected_helicities else {
            return Ok(None);
        };
        if selected_helicities.len() != 1 {
            return Ok(None);
        }
        let selected_id = selected_helicities
            .first()
            .expect("singleton helicity selector disappeared");
        let helicity_index = physics
            .helicity_index_by_id
            .get(selected_id)
            .copied()
            .ok_or_else(|| {
                RusticolError::selector(format!("unknown resolved helicity id {selected_id:?}"))
            })?;
        let helicity = &physics.manifest.helicities[helicity_index];
        if helicity.structural_zero || helicity.coefficient == 0.0 {
            return Ok(Some(CompanionSingletonSelection::StructuralZero));
        }
        let resolved_helicity_id = self
            .helicity_selector_companion
            .as_ref()
            .expect("classified recurrence companion disappeared")
            .resolved_helicity_id(helicity_index)?;
        Ok(Some(CompanionSingletonSelection::Runnable {
            physics_helicity_id: helicity_index,
            resolved_helicity_id,
        }))
    }

    fn prepare_persisted_companion_parameters(
        &mut self,
        common: &ExecutionRuntime,
    ) -> RusticolResult<()> {
        self.prepare_parameters(common)?;
        let companion = self
            .helicity_selector_companion
            .as_mut()
            .ok_or_else(|| RusticolError::internal("persisted helicity companion disappeared"))?;
        let (parameters_re, parameters_im) = self.scheduler.parameters_mut();
        companion
            .executor
            .set_parameter_planes(parameters_re, parameters_im)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_persisted_companion_view_into_unprofiled(
        &mut self,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        physics_helicity_id: usize,
        resolved_helicity_id: u32,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let mut tile_start = 0usize;
        while tile_start < batch.point_count() {
            let tile_stop =
                (tile_start + self.effective_point_tile_size()).min(batch.point_count());
            let point_count =
                self.flatten_external_tile_view::<false>(batch.subview(tile_start, tile_stop)?)?;
            self.ensure_primary_symmetric_group_color_workspace_for_capacity(point_count)?;
            let input_len = self.external_tile_input_len(point_count)?;
            let point_count_u32 = direct_point_count(point_count)?;
            let Self {
                selectors: RecurrenceNativeSelectors::ContractedColorUnion { contraction, .. },
                external_momenta,
                color_transform_re,
                color_transform_im,
                symmetric_group_color_workspace,
                helicity_selector_companion: Some(companion),
                ..
            } = self
            else {
                return Err(RusticolError::internal(
                    "persisted helicity companion lost its contracted recurrence lane",
                ));
            };
            let RecurrencePersistedHelicitySelectorRuntime {
                plan,
                dispatch,
                executor,
                primary_color_view: Some(color_view),
                ..
            } = companion
            else {
                return Err(RusticolError::integrity(
                    "persisted helicity companion has no primary local-color view",
                ));
            };
            executor.prepare(plan, dispatch, resolved_helicity_id, 4, point_count_u32)?;
            let tile = executor
                .execute_tile_unprofiled(&external_momenta[..input_len], point_count_u32)?;
            contract_primary_local_color_tile(
                tile,
                contraction,
                color_view,
                physics,
                physics_helicity_id,
                normalization_factor,
                color_transform_re,
                color_transform_im,
                symmetric_group_color_workspace.as_mut(),
                |point, value| output[tile_start + point] += value,
            )?;
            tile_start = tile_stop;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_persisted_companion_profiled(
        &mut self,
        common: &ExecutionRuntime,
        physics: &PhysicsRuntime,
        batch: &[Vec<[f64; 4]>],
        physics_helicity_id: usize,
        resolved_helicity_id: u32,
        mut accumulate: impl FnMut(usize, f64),
    ) -> RusticolResult<RuntimeProfile> {
        let total_started = Instant::now();
        let parameter_started = Instant::now();
        self.prepare_persisted_companion_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut input_setup = Duration::ZERO;
        let mut execution = Duration::ZERO;
        let mut reduction = Duration::ZERO;
        let mut report = persisted_execution_report_aggregate();
        let mut momentum_scalar_values = 0_u64;
        let mut tile_start = 0usize;
        while tile_start < batch.len() {
            let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
            let input_started = Instant::now();
            let point_count = self.flatten_external_tile::<false>(&batch[tile_start..tile_stop])?;
            input_setup += input_started.elapsed();
            self.ensure_primary_symmetric_group_color_workspace_for_capacity(point_count)?;
            let input_len = self.external_tile_input_len(point_count)?;
            let point_count_u32 = direct_point_count(point_count)?;
            let Self {
                selectors: RecurrenceNativeSelectors::ContractedColorUnion { contraction, .. },
                external_momenta,
                color_transform_re,
                color_transform_im,
                symmetric_group_color_workspace,
                helicity_selector_companion: Some(companion),
                ..
            } = self
            else {
                return Err(RusticolError::internal(
                    "persisted helicity companion lost its contracted recurrence lane",
                ));
            };
            let RecurrencePersistedHelicitySelectorRuntime {
                plan,
                dispatch,
                executor,
                primary_color_view: Some(color_view),
                ..
            } = companion
            else {
                return Err(RusticolError::integrity(
                    "persisted helicity companion has no primary local-color view",
                ));
            };
            let execution_started = Instant::now();
            executor.prepare(plan, dispatch, resolved_helicity_id, 4, point_count_u32)?;
            let (tile, tile_report) =
                executor.execute_tile_profiled(&external_momenta[..input_len], point_count_u32)?;
            execution += execution_started.elapsed();
            add_persisted_execution_report(&mut report, tile_report);
            momentum_scalar_values = momentum_scalar_values.saturating_add(
                (plan.momentum_forms().len() as u64)
                    .saturating_mul(4)
                    .saturating_mul(point_count as u64),
            );
            let reduction_started = Instant::now();
            contract_primary_local_color_tile(
                tile,
                contraction,
                color_view,
                physics,
                physics_helicity_id,
                common.normalization_factor,
                color_transform_re,
                color_transform_im,
                symmetric_group_color_workspace.as_mut(),
                |point, value| accumulate(tile_start + point, value),
            )?;
            reduction += reduction_started.elapsed();
            tile_start = tile_stop;
        }
        Ok(persisted_companion_profile(
            total_started.elapsed(),
            parameter_setup,
            input_setup,
            execution,
            reduction,
            momentum_scalar_values,
            report,
        ))
    }

    pub(super) fn run_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        self.run_f64_with_global_selectors(common, batch, None, None)
    }

    /// Clock-free totals path used by the native f64 ABI.
    ///
    /// Momentum input and result storage remain borrowed throughout this
    /// call. Selector membership is tested against the authenticated public
    /// axes directly so a warmed evaluation does not allocate temporary
    /// index vectors.
    pub(super) fn run_f64_view_into_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        if output.len() != batch.point_count() {
            return Err(RusticolError::invalid_argument(format!(
                "recurrence output has length {}, expected {}",
                output.len(),
                batch.point_count()
            )));
        }
        if batch.external_count() != self.external_source_count {
            return Err(RusticolError::invalid_argument(format!(
                "recurrence input has {} external momenta, expected {}",
                batch.external_count(),
                self.external_source_count
            )));
        }
        let physics = common.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        self.validate_global_selectors(physics, selected_helicities, selected_colors)?;
        output.fill(0.0);
        if selected_helicities.is_some_and(BTreeSet::is_empty) {
            return Ok(());
        }
        if let Some(selection) =
            self.companion_singleton_selection(physics, selected_helicities, selected_colors)?
        {
            match selection {
                CompanionSingletonSelection::StructuralZero => return Ok(()),
                CompanionSingletonSelection::Runnable {
                    physics_helicity_id,
                    resolved_helicity_id,
                } => {
                    self.prepare_persisted_companion_parameters(common)?;
                    return self.run_persisted_companion_view_into_unprofiled(
                        physics,
                        common.normalization_factor,
                        batch,
                        physics_helicity_id,
                        resolved_helicity_id,
                        output,
                    );
                }
            }
        }
        self.prepare_parameters(common)?;

        match &self.selectors {
            RecurrenceNativeSelectors::TopologyReplay { .. } => self
                .run_replay_view_into_unprofiled(
                    physics.lc_selector_reduction_view(),
                    common.normalization_factor,
                    batch,
                    selected_helicities,
                    selected_colors,
                    output,
                ),
            RecurrenceNativeSelectors::AllFlowUnion { .. } => self.run_union_view_into_unprofiled(
                physics,
                common.normalization_factor,
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => self
                .run_contracted_view_into_unprofiled(
                    physics,
                    common.normalization_factor,
                    batch,
                    selected_helicities,
                    selected_colors,
                    output,
                ),
        }
    }

    fn run_replay_view_into_unprofiled(
        &mut self,
        reduction: LcSelectorReductionView<'_>,
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let Some(selected_colors) = selected_colors else {
            return visit_replay_color_indices(
                reduction.color_components.len(),
                None,
                |color_index| {
                    self.run_replay_color_view_into_unprofiled(
                        reduction,
                        normalization_factor,
                        batch,
                        selected_helicities,
                        output,
                        color_index,
                    )
                },
            );
        };

        let mut color_indices = std::mem::take(&mut self.replay_color_index_scratch);
        let result = (|| {
            reduction.retain_selected_color_indices(selected_colors, &mut color_indices)?;
            visit_replay_color_indices(
                reduction.color_components.len(),
                Some(&color_indices),
                |color_index| {
                    self.run_replay_color_view_into_unprofiled(
                        reduction,
                        normalization_factor,
                        batch,
                        selected_helicities,
                        output,
                        color_index,
                    )
                },
            )
        })();
        self.replay_color_index_scratch = color_indices;
        result
    }

    /// Restrict work only when the caller restricts the physical helicity
    /// axis. The original helicity-summed selectors remain untouched. Plans
    /// are prepared once per requested subset/flow, outside the point loop,
    /// and retained across selector changes and intervening summed calls.
    fn selected_replay_color_plan(
        &mut self,
        reduction: LcSelectorReductionView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        color_index: usize,
    ) -> RusticolResult<Option<Arc<DirectReplaySelectorPlan>>> {
        let Some(selected_helicities) = selected_helicities else {
            return Ok(None);
        };
        if selected_helicities.len() == reduction.helicity_index_by_id.len() {
            // Selector IDs were validated at the public boundary. Spelling
            // out the complete axis is the same work as omitting it.
            return Ok(None);
        }
        if let Some(plan) = self
            .selected_replay_cache
            .get(selected_helicities)
            .and_then(|plans| plans.get(&color_index))
        {
            return Ok(Some(Arc::clone(plan)));
        }
        let RecurrenceNativeSelectors::TopologyReplay {
            replay_selectors,
            direct_helicity_to_physics,
        } = &self.selectors
        else {
            return Err(RusticolError::internal(
                "selected replay plan requires topology replay",
            ));
        };
        let selector = replay_selectors.get(color_index).ok_or_else(|| {
            RusticolError::integrity("recurrence replay selector is outside the public color axis")
        })?;
        let destinations = selected_replay_destination_ids(
            self.scheduler.plan().amplitude_destinations(),
            selector.representative_flow_id(),
            selector.helicity_map(),
            direct_helicity_to_physics,
            reduction,
            selected_helicities,
        )?;
        let program_key = (selector.representative_flow_id(), destinations);
        // Different public flows and physical aliases can select the same
        // representative currents. Share their dependency program instead
        // of retaining another copy for every flow/permutation.
        let selected = if let Some(program) = self.selected_replay_programs.get(&program_key) {
            Arc::new(
                self.scheduler
                    .prepare_replay_selector_with_destination_program(selector, program)?,
            )
        } else {
            let selected = Arc::new(
                self.scheduler
                    .prepare_replay_selector_for_destinations(selector, &program_key.1)?,
            );
            self.selected_replay_programs
                .insert(program_key, Arc::clone(&selected));
            selected
        };
        self.selected_replay_cache
            .entry(selected_helicities.clone())
            .or_default()
            .insert(color_index, Arc::clone(&selected));
        Ok(Some(selected))
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn run_replay_color_view_into_unprofiled(
        &mut self,
        reduction: LcSelectorReductionView<'_>,
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        output: &mut [f64],
        color_index: usize,
    ) -> RusticolResult<()> {
        if !reduction.color_is_computed(color_index) {
            return Ok(());
        }
        let color_weight = reduction.color(color_index).coefficient();
        let selected_replay =
            self.selected_replay_color_plan(reduction, selected_helicities, color_index)?;

        let mut tile_start = 0usize;
        while tile_start < batch.point_count() {
            let tile_stop =
                (tile_start + self.effective_point_tile_size()).min(batch.point_count());
            let point_count =
                self.flatten_external_tile_view::<true>(batch.subview(tile_start, tile_stop)?)?;
            let input_len = self.external_tile_input_len(point_count)?;
            let (replay_selector, direct_helicity_to_physics) = match &self.selectors {
                RecurrenceNativeSelectors::TopologyReplay {
                    replay_selectors,
                    direct_helicity_to_physics,
                } => (
                    replay_selectors.get(color_index).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence replay selector is outside the public color axis",
                        )
                    })?,
                    direct_helicity_to_physics,
                ),
                RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                RecurrenceNativeSelectors::ContractedColorUnion { .. } => unreachable!(),
            };
            let replay_selector = selected_replay.as_deref().unwrap_or(replay_selector);
            let direct_output = self
                .scheduler
                .execute_replay_tile_from_external_planes::<false>(
                    replay_selector,
                    direct_point_count(point_count)?,
                    &self.external_momenta[..input_len],
                )?;

            for amplitude in direct_output.selected_amplitudes() {
                let amplitude = amplitude?;
                let helicity_index = replay_destination_physics_helicity(
                    replay_selector.helicity_map(),
                    direct_helicity_to_physics,
                    amplitude.destination.target_helicity_id_or_sentinel,
                )?;
                let helicity = reduction.helicity(helicity_index);
                if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                    continue;
                }
                let helicity_weight =
                    reduction.helicity_orbit_weight(selected_helicities, helicity_index);
                if helicity_weight == 0.0 {
                    continue;
                }
                let weight = helicity_weight * color_weight * normalization_factor;
                accumulate_lc_diagonal_amplitude(
                    point_count,
                    weight,
                    |point| (amplitude.real[point], amplitude.imaginary[point]),
                    |point, value| output[tile_start + point] += value,
                );
            }
            tile_start = tile_stop;
        }
        Ok(())
    }

    fn run_union_view_into_unprofiled(
        &mut self,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        for helicity_index in 0..physics.manifest.helicities.len() {
            if !recurrence_helicity_is_selected(physics, selected_helicities, helicity_index) {
                continue;
            }
            let helicity = &physics.manifest.helicities[helicity_index];
            if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                continue;
            }
            let selector = self.union_helicity_selector(helicity_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.point_count() {
                let tile_stop =
                    (tile_start + self.effective_point_tile_size()).min(batch.point_count());
                let point_count =
                    self.flatten_external_tile_view::<true>(batch.subview(tile_start, tile_stop)?)?;
                let input_len = self.external_tile_input_len(point_count)?;
                let destination_by_public_flow = match &self.selectors {
                    RecurrenceNativeSelectors::AllFlowUnion {
                        destination_by_public_flow,
                        ..
                    } => destination_by_public_flow,
                    RecurrenceNativeSelectors::TopologyReplay { .. } => unreachable!(),
                    RecurrenceNativeSelectors::ContractedColorUnion { .. } => unreachable!(),
                };
                let direct_output = self
                    .scheduler
                    .execute_union_tile_from_external_planes::<false>(
                        &selector,
                        direct_point_count(point_count)?,
                        &self.external_momenta[..input_len],
                    )?;

                for color_index in 0..physics.manifest.color_components.len() {
                    if !physics.color_is_computed(color_index)
                        || !recurrence_color_is_selected(physics, selected_colors, color_index)
                    {
                        continue;
                    }
                    let destination_id =
                        *destination_by_public_flow.get(color_index).ok_or_else(|| {
                            RusticolError::integrity(
                                "all-flow-union color is outside retained public coverage",
                            )
                        })?;
                    let values_re =
                        direct_output
                            .destination_re(destination_id)
                            .ok_or_else(|| {
                                RusticolError::integrity(
                                    "all-flow-union amplitude destination is absent",
                                )
                            })?;
                    let values_im =
                        direct_output
                            .destination_im(destination_id)
                            .ok_or_else(|| {
                                RusticolError::integrity(
                                    "all-flow-union amplitude destination is absent",
                                )
                            })?;
                    let color_weight = physics.manifest.color_components[color_index].coefficient();
                    let weight = helicity.coefficient * color_weight * normalization_factor;
                    for point in 0..point_count {
                        output[tile_start + point] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                tile_start = tile_stop;
            }
        }
        Ok(())
    }

    fn run_contracted_view_into_unprofiled(
        &mut self,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        reject_contracted_color_selector(selected_colors)?;
        let mut tile_start = 0usize;
        while tile_start < batch.point_count() {
            let tile_stop =
                (tile_start + self.effective_point_tile_size()).min(batch.point_count());
            let point_count =
                self.flatten_external_tile_view::<true>(batch.subview(tile_start, tile_stop)?)?;
            let input_len = self.external_tile_input_len(point_count)?;
            self.execute_and_contract_contracted_tile(
                point_count,
                input_len,
                physics,
                selected_helicities,
                normalization_factor,
                false,
                |point, _helicity, value| output[tile_start + point] += value,
            )?;
            tile_start = tile_stop;
        }
        Ok(())
    }

    pub(super) fn run_f64_with_global_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion { .. } => {
                return self.run_union_f64_with_global_selectors(
                    common,
                    batch,
                    selected_helicities,
                    selected_colors,
                );
            }
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                return self.run_contracted_f64_with_global_selectors(
                    common,
                    batch,
                    selected_helicities,
                    selected_colors,
                );
            }
            RecurrenceNativeSelectors::TopologyReplay { .. } => {}
        }
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let reduction_view = physics.lc_selector_reduction_view();
        let helicity_indices = reduction_view.selected_helicity_indices(selected_helicities)?;
        let color_indices = reduction_view.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;

        let mut values = vec![0.0; batch.len()];
        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for color_index in color_indices {
            if !reduction_view.color_is_computed(color_index) {
                continue;
            }
            let color_weight = reduction_view.color(color_index).coefficient();
            let selected_replay =
                self.selected_replay_color_plan(reduction_view, selected_helicities, color_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count =
                    self.flatten_external_tile::<true>(&batch[tile_start..tile_stop])?;
                external_momentum_flatten += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|values| values.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let (replay_selector, direct_helicity_to_physics) = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        replay_selectors,
                        direct_helicity_to_physics,
                    } => (
                        replay_selectors.get(color_index).ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence replay selector is outside the public color axis",
                            )
                        })?,
                        direct_helicity_to_physics,
                    ),
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                    RecurrenceNativeSelectors::ContractedColorUnion { .. } => unreachable!(),
                };
                let replay_selector = selected_replay.as_deref().unwrap_or(replay_selector);
                let output = self
                    .scheduler
                    .execute_replay_tile_from_external_planes::<true>(
                        replay_selector,
                        u32::try_from(point_count).map_err(|_| {
                            RusticolError::invalid_argument(
                                "recurrence point tile exceeds the native u32 ABI",
                            )
                        })?,
                        &self.external_momenta[..input_len],
                    )?;

                let reduction_started = Instant::now();
                for amplitude in output.selected_amplitudes() {
                    let amplitude = amplitude?;
                    let helicity_index = replay_destination_physics_helicity(
                        replay_selector.helicity_map(),
                        direct_helicity_to_physics,
                        amplitude.destination.target_helicity_id_or_sentinel,
                    )?;
                    let helicity = reduction_view.helicity(helicity_index);
                    if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0
                    {
                        continue;
                    }
                    let helicity_weight =
                        reduction_view.helicity_orbit_weight(selected_helicities, helicity_index);
                    if helicity_weight == 0.0 {
                        continue;
                    }
                    let weight = helicity_weight * color_weight * common.normalization_factor;
                    accumulate_lc_diagonal_amplitude(
                        point_count,
                        weight,
                        |point| (amplitude.real[point], amplitude.imaginary[point]),
                        |point, value| values[tile_start + point] += value,
                    );
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            values,
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    pub(super) fn run_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion { .. } => {
                return self.run_union_resolved_f64(
                    common,
                    batch,
                    selected_helicities,
                    selected_colors,
                );
            }
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                return self.run_contracted_resolved_f64(
                    common,
                    batch,
                    selected_helicities,
                    selected_colors,
                );
            }
            RecurrenceNativeSelectors::TopologyReplay { .. } => {}
        }
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "resolved recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved recurrence execution requires physics metadata")
        })?;
        let reduction_view = physics.lc_selector_reduction_view();
        let helicity_indices = reduction_view.selected_helicity_indices(selected_helicities)?;
        let color_indices = reduction_view.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let output_layout =
            LcResolvedOutputLayout::new(batch.len(), helicity_indices.len(), color_indices.len())?;
        let mut values = vec![0.0; output_layout.output_len()?];
        let mut helicity_position = vec![None; reduction_view.helicities.len()];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            helicity_position[index] = Some(position);
        }

        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for (color_position, color_index) in color_indices.iter().copied().enumerate() {
            if !reduction_view.color_is_computed(color_index) {
                continue;
            }
            let color_weight = reduction_view.color(color_index).coefficient();
            let selected_replay =
                self.selected_replay_color_plan(reduction_view, selected_helicities, color_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count =
                    self.flatten_external_tile::<true>(&batch[tile_start..tile_stop])?;
                external_momentum_flatten += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let (replay_selector, direct_helicity_to_physics) = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        replay_selectors,
                        direct_helicity_to_physics,
                    } => (
                        replay_selectors.get(color_index).ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence replay selector is outside the public color axis",
                            )
                        })?,
                        direct_helicity_to_physics,
                    ),
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                    RecurrenceNativeSelectors::ContractedColorUnion { .. } => unreachable!(),
                };
                let replay_selector = selected_replay.as_deref().unwrap_or(replay_selector);
                let output = self
                    .scheduler
                    .execute_replay_tile_from_external_planes::<true>(
                        replay_selector,
                        u32::try_from(point_count).map_err(|_| {
                            RusticolError::invalid_argument(
                                "recurrence point tile exceeds the native u32 ABI",
                            )
                        })?,
                        &self.external_momenta[..input_len],
                    )?;

                let reduction_started = Instant::now();
                for amplitude in output.selected_amplitudes() {
                    let amplitude = amplitude?;
                    let helicity_index = replay_destination_physics_helicity(
                        replay_selector.helicity_map(),
                        direct_helicity_to_physics,
                        amplitude.destination.target_helicity_id_or_sentinel,
                    )?;
                    let helicity = reduction_view.helicity(helicity_index);
                    if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0
                    {
                        continue;
                    }
                    for physical_helicity in reduction_view.helicity_orbit_members(helicity_index) {
                        let Some(helicity_position) = helicity_position[*physical_helicity] else {
                            continue;
                        };
                        let weight = reduction_view.helicity(*physical_helicity).coefficient
                            * color_weight
                            * common.normalization_factor;
                        accumulate_lc_diagonal_amplitude(
                            point_count,
                            weight,
                            |point| (amplitude.real[point], amplitude.imaginary[point]),
                            |point, value| {
                                let target = output_layout
                                    .index(tile_start + point, helicity_position, color_position)
                                    .expect("validated resolved LC coordinate");
                                values[target] += value;
                            },
                        );
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            ResolvedValues {
                values,
                point_count: batch.len(),
                helicity_indices,
                color_indices,
            },
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn run_contracted_f64_with_global_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "recurrence evaluation requires at least one point",
            ));
        }
        reject_contracted_color_selector(selected_colors)?;
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        self.validate_public_axes(&physics, &helicity_indices, &[0])?;
        let mut values = vec![0.0; batch.len()];
        if selected_helicities.is_some_and(BTreeSet::is_empty) {
            return Ok((values, RuntimeProfile::default()));
        }
        if let Some(selection) =
            self.companion_singleton_selection(&physics, selected_helicities, selected_colors)?
        {
            return match selection {
                CompanionSingletonSelection::StructuralZero => {
                    Ok((values, RuntimeProfile::default()))
                }
                CompanionSingletonSelection::Runnable {
                    physics_helicity_id,
                    resolved_helicity_id,
                } => {
                    let profile = self.run_persisted_companion_profiled(
                        common,
                        &physics,
                        batch,
                        physics_helicity_id,
                        resolved_helicity_id,
                        |point, value| values[point] += value,
                    )?;
                    Ok((values, profile))
                }
            };
        }
        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        let mut tile_start = 0usize;
        while tile_start < batch.len() {
            let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
            let flatten_started = Instant::now();
            let point_count = self.flatten_external_tile::<true>(&batch[tile_start..tile_stop])?;
            external_momentum_flatten += flatten_started.elapsed();
            let input_len = self.external_tile_input_len(point_count)?;
            reduction += self.execute_and_contract_contracted_tile(
                point_count,
                input_len,
                &physics,
                selected_helicities,
                common.normalization_factor,
                true,
                |point, _helicity, value| values[tile_start + point] += value,
            )?;
            tile_start = tile_stop;
        }

        Ok((
            values,
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn run_contracted_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "resolved recurrence evaluation requires at least one point",
            ));
        }
        reject_contracted_color_selector(selected_colors)?;
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        self.validate_public_axes(&physics, &helicity_indices, &[0])?;
        let mut helicity_position = vec![None; physics.manifest.helicities.len()];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            helicity_position[index] = Some(position);
        }
        let component_count = helicity_indices.len();
        let mut values = vec![
            0.0;
            batch.len().checked_mul(component_count).ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved output overflows")
            })?
        ];
        if selected_helicities.is_some_and(BTreeSet::is_empty) {
            return Ok((
                ResolvedValues {
                    values,
                    point_count: batch.len(),
                    helicity_indices,
                    color_indices: vec![0],
                },
                RuntimeProfile::default(),
            ));
        }
        if let Some(selection) =
            self.companion_singleton_selection(&physics, selected_helicities, selected_colors)?
        {
            let profile = match selection {
                CompanionSingletonSelection::StructuralZero => RuntimeProfile::default(),
                CompanionSingletonSelection::Runnable {
                    physics_helicity_id,
                    resolved_helicity_id,
                } => {
                    let position = helicity_position[physics_helicity_id]
                        .expect("singleton persisted helicity has no resolved output position");
                    self.run_persisted_companion_profiled(
                        common,
                        &physics,
                        batch,
                        physics_helicity_id,
                        resolved_helicity_id,
                        |point, value| {
                            values[point * component_count + position] += value;
                        },
                    )?
                }
            };
            return Ok((
                ResolvedValues {
                    values,
                    point_count: batch.len(),
                    helicity_indices,
                    color_indices: vec![0],
                },
                profile,
            ));
        }
        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        let mut tile_start = 0usize;
        while tile_start < batch.len() {
            let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
            let flatten_started = Instant::now();
            let point_count = self.flatten_external_tile::<true>(&batch[tile_start..tile_stop])?;
            external_momentum_flatten += flatten_started.elapsed();
            let input_len = self.external_tile_input_len(point_count)?;
            reduction += self.execute_and_contract_contracted_tile(
                point_count,
                input_len,
                &physics,
                selected_helicities,
                common.normalization_factor,
                true,
                |point, helicity, value| {
                    let position = helicity_position[helicity]
                        .expect("selected contracted helicity has a result position");
                    values[(tile_start + point) * component_count + position] += value;
                },
            )?;
            tile_start = tile_stop;
        }

        Ok((
            ResolvedValues {
                values,
                point_count: batch.len(),
                helicity_indices,
                color_indices: vec![0],
            },
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn run_union_f64_with_global_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        let color_indices = physics.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let color_destinations = color_indices
            .iter()
            .copied()
            .map(|color_index| Ok((color_index, self.union_destination_id(color_index)?)))
            .collect::<RusticolResult<Vec<_>>>()?;

        let mut values = vec![0.0; batch.len()];
        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for helicity_index in helicity_indices {
            let helicity = &physics.manifest.helicities[helicity_index];
            if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                continue;
            }
            let selector = self.union_helicity_selector(helicity_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count =
                    self.flatten_external_tile::<true>(&batch[tile_start..tile_stop])?;
                external_momentum_flatten += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let output = self
                    .scheduler
                    .execute_union_tile_from_external_planes::<true>(
                        &selector,
                        u32::try_from(point_count).map_err(|_| {
                            RusticolError::invalid_argument(
                                "recurrence point tile exceeds the native u32 ABI",
                            )
                        })?,
                        &self.external_momenta[..input_len],
                    )?;

                let reduction_started = Instant::now();
                for (color_index, destination_id) in color_destinations.iter().copied() {
                    if !physics.color_is_computed(color_index) {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let color_weight = physics.manifest.color_components[color_index].coefficient();
                    let weight = helicity.coefficient * color_weight * common.normalization_factor;
                    for point in 0..point_count {
                        values[tile_start + point] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            values,
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn run_union_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "resolved recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        let color_indices = physics.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let color_destinations = color_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(color_position, color_index)| {
                Ok((
                    color_position,
                    color_index,
                    self.union_destination_id(color_index)?,
                ))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved shape overflows")
            })?;
        let mut values = vec![
            0.0;
            batch.len().checked_mul(component_count).ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved output overflows")
            })?
        ];

        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for (helicity_position, helicity_index) in helicity_indices.iter().copied().enumerate() {
            let helicity = &physics.manifest.helicities[helicity_index];
            if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                continue;
            }
            let selector = self.union_helicity_selector(helicity_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count =
                    self.flatten_external_tile::<true>(&batch[tile_start..tile_stop])?;
                external_momentum_flatten += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let output = self
                    .scheduler
                    .execute_union_tile_from_external_planes::<true>(
                        &selector,
                        u32::try_from(point_count).map_err(|_| {
                            RusticolError::invalid_argument(
                                "recurrence point tile exceeds the native u32 ABI",
                            )
                        })?,
                        &self.external_momenta[..input_len],
                    )?;

                let reduction_started = Instant::now();
                for (color_position, color_index, destination_id) in
                    color_destinations.iter().copied()
                {
                    if !physics.color_is_computed(color_index) {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let color_weight = physics.manifest.color_components[color_index].coefficient();
                    let weight = helicity.coefficient * color_weight * common.normalization_factor;
                    for point in 0..point_count {
                        let target = (tile_start + point) * component_count
                            + helicity_position * color_indices.len()
                            + color_position;
                        values[target] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            ResolvedValues {
                values,
                point_count: batch.len(),
                helicity_indices,
                color_indices,
            },
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn union_helicity_selector(
        &self,
        physics_helicity_index: usize,
    ) -> RusticolResult<DirectUnionHelicitySelectorPlan> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion {
                helicity_selectors_by_physics,
                ..
            } => helicity_selectors_by_physics
                .get(physics_helicity_index)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union helicity is outside retained public coverage",
                    )
                }),
            RecurrenceNativeSelectors::TopologyReplay { .. } => Err(RusticolError::integrity(
                "union selector requested from topology replay",
            )),
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                Err(RusticolError::integrity(
                    "union selector requested from contracted-color recurrence",
                ))
            }
        }
    }

    fn union_destination_id(&self, color_index: usize) -> RusticolResult<u32> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion {
                destination_by_public_flow,
                ..
            } => destination_by_public_flow
                .get(color_index)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union color is outside retained public coverage",
                    )
                }),
            RecurrenceNativeSelectors::TopologyReplay { .. } => Err(RusticolError::integrity(
                "union destination requested from topology replay",
            )),
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                Err(RusticolError::integrity(
                    "union destination requested from contracted-color recurrence",
                ))
            }
        }
    }

    fn prepare_parameters(&mut self, common: &ExecutionRuntime) -> RusticolResult<()> {
        let (parameters_re, parameters_im) = self.scheduler.parameters_mut();
        initialize_prepared_parameter_planes(
            &self.parameter_defaults,
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            parameters_re,
            parameters_im,
        )
    }

    // The ordinary direct scheduler consumes contiguous source/component
    // planes. Persisted-helicity companion entrypoints retain their existing
    // point-major contract by requesting SOURCE_MAJOR=false. Both layouts use
    // the same preallocated workspace and the same single input-copy pass.
    fn flatten_external_tile<const SOURCE_MAJOR: bool>(
        &mut self,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<usize> {
        let point_count = batch.len();
        for point in batch {
            if point.len() != self.external_source_count {
                return Err(RusticolError::invalid_argument(format!(
                    "recurrence point has {} external momenta, expected {}",
                    point.len(),
                    self.external_source_count
                )));
            }
        }
        ensure_external_momentum_workspace_capacity(
            &mut self.external_momenta,
            point_count,
            self.external_source_count,
            self.effective_point_tile_size,
        )?;
        for (point_index, point) in batch.iter().enumerate() {
            for (source_slot, momentum) in point.iter().enumerate() {
                write_external_momentum::<SOURCE_MAJOR>(
                    &mut self.external_momenta,
                    point_count,
                    self.external_source_count,
                    point_index,
                    source_slot,
                    momentum,
                );
            }
        }
        Ok(point_count)
    }

    fn flatten_external_tile_view<const SOURCE_MAJOR: bool>(
        &mut self,
        batch: F64MomentumBatchView<'_>,
    ) -> RusticolResult<usize> {
        let point_count = batch.point_count();
        if batch.external_count() != self.external_source_count {
            return Err(RusticolError::invalid_argument(format!(
                "recurrence input has {} external momenta, expected {}",
                batch.external_count(),
                self.external_source_count
            )));
        }
        ensure_external_momentum_workspace_capacity(
            &mut self.external_momenta,
            point_count,
            self.external_source_count,
            self.effective_point_tile_size,
        )?;
        for point_index in 0..point_count {
            let point = batch.point(point_index);
            for source_slot in 0..self.external_source_count {
                let momentum = point.momentum(source_slot).ok_or_else(|| {
                    RusticolError::integrity(
                        "validated recurrence momentum view is missing an external leg",
                    )
                })?;
                write_external_momentum::<SOURCE_MAJOR>(
                    &mut self.external_momenta,
                    point_count,
                    self.external_source_count,
                    point_index,
                    source_slot,
                    &momentum,
                );
            }
        }
        Ok(point_count)
    }

    fn external_tile_input_len(&self, point_count: usize) -> RusticolResult<usize> {
        external_momentum_scalar_len(point_count, self.external_source_count)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_and_contract_contracted_tile(
        &mut self,
        point_count: usize,
        input_len: usize,
        physics: &PhysicsRuntime,
        selected_helicities: Option<&BTreeSet<String>>,
        normalization_factor: f64,
        profiled: bool,
        accumulate: impl FnMut(usize, usize, f64),
    ) -> RusticolResult<Duration> {
        {
            let RecurrenceNativeSelectors::ContractedColorUnion {
                destination_physics_helicity,
                ..
            } = &self.selectors
            else {
                return Err(RusticolError::integrity(
                    "contracted execution requested from a non-contracted recurrence",
                ));
            };
            if let Some(selected_helicities) = selected_helicities
                && !contracted_selection_has_computed_destination(
                    destination_physics_helicity,
                    physics,
                    selected_helicities,
                )?
            {
                // The selected public orbit is structural-zero or has no computed
                // representative. The caller zeroed its output before entering
                // this tile, so no recurrence, color work, or lazy FFT workspace
                // is required. An absent selector deliberately still means full
                // execution.
                return Ok(Duration::ZERO);
            }
        }
        self.ensure_primary_symmetric_group_color_workspace_for_capacity(point_count)?;
        let Self {
            scheduler,
            selectors:
                RecurrenceNativeSelectors::ContractedColorUnion {
                    contraction,
                    destination_physics_helicity,
                    replay_routes,
                },
            external_momenta,
            contracted_replay_re,
            contracted_replay_im,
            color_transform_re,
            color_transform_im,
            symmetric_group_color_workspace,
            ..
        } = self
        else {
            return Err(RusticolError::integrity(
                "contracted execution requested from a non-contracted recurrence",
            ));
        };
        let input = &external_momenta[..input_len];
        let point_count_u32 = direct_point_count(point_count)?;

        if replay_routes.is_empty() {
            let direct_output = if profiled {
                scheduler
                    .execute_contracted_tile_from_external_planes::<true>(point_count_u32, input)?
            } else {
                scheduler
                    .execute_contracted_tile_from_external_planes::<false>(point_count_u32, input)?
            };
            let reduction_started = profiled.then(Instant::now);
            contract_color_tile(
                &direct_output,
                contraction,
                destination_physics_helicity,
                physics,
                selected_helicities,
                normalization_factor,
                color_transform_re,
                color_transform_im,
                symmetric_group_color_workspace.as_mut(),
                accumulate,
            )?;
            return Ok(reduction_started.map_or(Duration::ZERO, |started| started.elapsed()));
        }

        let point_stride = scheduler.point_tile_size() as usize;
        let destination_count = contraction.destination_count() as usize;
        let required = destination_count.checked_mul(point_stride).ok_or_else(|| {
            RusticolError::integrity("contracted replay amplitude workspace overflows usize")
        })?;
        if contracted_replay_re.len() < required || contracted_replay_im.len() < required {
            return Err(RusticolError::integrity(
                "contracted replay amplitude workspace is too small",
            ));
        }
        let mut replay_output_copy = Duration::ZERO;
        for route in replay_routes {
            let direct_output = if profiled {
                scheduler.execute_replay_tile_from_external_planes::<true>(
                    &route.selector,
                    point_count_u32,
                    input,
                )?
            } else {
                scheduler.execute_replay_tile_from_external_planes::<false>(
                    &route.selector,
                    point_count_u32,
                    input,
                )?
            };
            // The scheduler profiles momentum fill, schedule execution, and
            // replay scaling independently.  Start this clock only after it
            // returns: including the scheduler call here would attribute the
            // complete recurrence execution a second time as reduction.
            let replay_output_copy_started = profiled.then(Instant::now);
            for &(source_destination, physical_destination) in &route.destination_copies {
                let source_re = direct_output
                    .destination_re(source_destination)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "contracted replay source amplitude destination is absent",
                        )
                    })?;
                let source_im = direct_output
                    .destination_im(source_destination)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "contracted replay source amplitude destination is absent",
                        )
                    })?;
                let start = physical_destination as usize * point_stride;
                contracted_replay_re[start..start + point_count].copy_from_slice(source_re);
                contracted_replay_im[start..start + point_count].copy_from_slice(source_im);
            }
            if let Some(started) = replay_output_copy_started {
                replay_output_copy += started.elapsed();
            }
        }
        let tile = ContractedReplayTile {
            values_re: contracted_replay_re,
            values_im: contracted_replay_im,
            point_count,
            point_stride,
            destination_count,
        };
        let reduction_started = profiled.then(Instant::now);
        contract_color_tile(
            &tile,
            contraction,
            destination_physics_helicity,
            physics,
            selected_helicities,
            normalization_factor,
            color_transform_re,
            color_transform_im,
            symmetric_group_color_workspace.as_mut(),
            accumulate,
        )?;
        Ok(replay_output_copy
            + reduction_started.map_or(Duration::ZERO, |started| started.elapsed()))
    }

    fn ensure_primary_symmetric_group_color_workspace_for_capacity(
        &mut self,
        point_capacity: usize,
    ) -> RusticolResult<()> {
        let Self {
            selectors,
            symmetric_group_color_workspace,
            ..
        } = self;
        let RecurrenceNativeSelectors::ContractedColorUnion { contraction, .. } = selectors else {
            return Ok(());
        };
        let Some(RuntimeColorContractionReducer::SymmetricGroupFourier(reducer)) =
            contraction.runtime_reducer()
        else {
            return Ok(());
        };
        ensure_symmetric_group_color_workspace_capacity(
            symmetric_group_color_workspace,
            reducer,
            point_capacity,
        )
    }

    fn validate_public_axis_lengths(&self, physics: &PhysicsRuntime) -> RusticolResult<()> {
        let (color_count, helicity_count) = match &self.selectors {
            RecurrenceNativeSelectors::TopologyReplay {
                replay_selectors, ..
            } => (replay_selectors.len(), physics.manifest.helicities.len()),
            RecurrenceNativeSelectors::AllFlowUnion {
                helicity_selectors_by_physics,
                destination_by_public_flow,
            } => (
                destination_by_public_flow.len(),
                helicity_selectors_by_physics.len(),
            ),
            RecurrenceNativeSelectors::ContractedColorUnion {
                destination_physics_helicity,
                ..
            } => {
                if destination_physics_helicity
                    .iter()
                    .any(|index| *index >= physics.manifest.helicities.len())
                {
                    return Err(RusticolError::integrity(
                        "contracted recurrence maps a destination outside the public helicity axis",
                    ));
                }
                (1, physics.manifest.helicities.len())
            }
        };
        if color_count != physics.manifest.color_components.len()
            || helicity_count != physics.manifest.helicities.len()
        {
            return Err(RusticolError::integrity(
                "recurrence selectors do not cover the public physics axes",
            ));
        }
        Ok(())
    }

    fn validate_public_axes(
        &self,
        physics: &PhysicsRuntime,
        helicity_indices: &[usize],
        color_indices: &[usize],
    ) -> RusticolResult<()> {
        self.validate_public_axis_lengths(physics)?;
        let color_count = physics.manifest.color_components.len();
        let helicity_count = physics.manifest.helicities.len();
        if color_indices.iter().any(|index| *index >= color_count)
            || helicity_indices
                .iter()
                .any(|index| *index >= helicity_count)
        {
            return Err(RusticolError::integrity(
                "recurrence selector mapping is outside the public physics axes",
            ));
        }
        Ok(())
    }
}

fn lazy_symmetric_group_workspace_for_point_tile(
    reducer: &crate::recurrence::RuntimeSymmetricGroupColorContraction,
    requested_point_tile_size: usize,
) -> RusticolResult<(usize, Option<RuntimeSymmetricGroupColorWorkspace>)> {
    Ok((
        reducer.bounded_lane_capacity(requested_point_tile_size)?,
        None,
    ))
}

fn ensure_symmetric_group_color_workspace_capacity(
    workspace: &mut Option<RuntimeSymmetricGroupColorWorkspace>,
    reducer: &crate::recurrence::RuntimeSymmetricGroupColorContraction,
    point_capacity: usize,
) -> RusticolResult<()> {
    if let Some(workspace) = workspace.as_mut() {
        return workspace.ensure_lane_capacity(reducer, point_capacity);
    }
    *workspace = Some(reducer.workspace(point_capacity)?);
    Ok(())
}

fn external_momentum_scalar_len(
    point_count: usize,
    external_source_count: usize,
) -> RusticolResult<usize> {
    point_count
        .checked_mul(external_source_count)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            RusticolError::invalid_argument("recurrence external-momentum tile length overflows")
        })
}

#[inline]
fn write_external_momentum<const SOURCE_MAJOR: bool>(
    output: &mut [f64],
    point_count: usize,
    source_count: usize,
    point: usize,
    source: usize,
    momentum: &[f64; 4],
) {
    if SOURCE_MAJOR {
        for (component, &value) in momentum.iter().enumerate() {
            output[(source * 4 + component) * point_count + point] = value;
        }
    } else {
        let start = (point * source_count + source) * 4;
        output[start..start + 4].copy_from_slice(momentum);
    }
}

fn ensure_external_momentum_workspace_capacity(
    workspace: &mut Vec<f64>,
    point_count: usize,
    external_source_count: usize,
    maximum_point_count: usize,
) -> RusticolResult<usize> {
    if point_count > maximum_point_count {
        return Err(RusticolError::invalid_argument(
            "recurrence point tile exceeds its persistent workspace",
        ));
    }
    let required = external_momentum_scalar_len(point_count, external_source_count)?;
    if workspace.len() < required {
        workspace
            .try_reserve_exact(required - workspace.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "recurrence external-momentum workspace allocation failed: {error}"
                ))
            })?;
        workspace.resize(required, 0.0);
    }
    Ok(required)
}

fn validate_recurrence_selector_ids(
    physics: &PhysicsRuntime,
    selected_helicities: Option<&BTreeSet<String>>,
    selected_colors: Option<&BTreeSet<String>>,
) -> RusticolResult<()> {
    if let Some(ids) = selected_helicities
        && let Some(id) = ids
            .iter()
            .find(|id| !physics.helicity_index_by_id.contains_key(*id))
    {
        return Err(RusticolError::selector(format!(
            "unknown resolved helicity id {id:?}"
        )));
    }
    if let Some(ids) = selected_colors
        && let Some(id) = ids
            .iter()
            .find(|id| !physics.color_index_by_id.contains_key(*id))
    {
        return Err(RusticolError::selector(format!(
            "unknown resolved color component id {id:?}"
        )));
    }
    Ok(())
}

#[inline(always)]
fn recurrence_helicity_is_selected(
    physics: &PhysicsRuntime,
    selected: Option<&BTreeSet<String>>,
    index: usize,
) -> bool {
    selected.is_none_or(|ids| ids.contains(&physics.manifest.helicities[index].id))
}

#[inline(always)]
fn recurrence_helicity_orbit_weight(
    physics: &PhysicsRuntime,
    selected: Option<&BTreeSet<String>>,
    representative_index: usize,
) -> f64 {
    physics
        .helicity_orbit_members(representative_index)
        .iter()
        .copied()
        .filter(|index| recurrence_helicity_is_selected(physics, selected, *index))
        .map(|index| physics.manifest.helicities[index].coefficient)
        .sum()
}

#[inline(always)]
fn recurrence_color_is_selected(
    physics: &PhysicsRuntime,
    selected: Option<&BTreeSet<String>>,
    index: usize,
) -> bool {
    selected.is_none_or(|ids| ids.contains(physics.manifest.color_components[index].id()))
}

fn direct_point_count(point_count: usize) -> RusticolResult<u32> {
    u32::try_from(point_count).map_err(|_| {
        RusticolError::invalid_argument("recurrence point tile exceeds the native u32 ABI")
    })
}

fn reject_contracted_color_selector(
    selected_colors: Option<&BTreeSet<String>>,
) -> RusticolResult<()> {
    if selected_colors.is_some() {
        return Err(RusticolError::selector(
            "contracted NLC/full recurrence does not expose a color-flow selector",
        ));
    }
    Ok(())
}

fn contracted_destination_helicity_map(
    contraction: &RecurrenceColorContraction,
    direct_helicity_to_physics: &[usize],
) -> RusticolResult<Vec<usize>> {
    let destination_count = contraction.destination_count() as usize;
    let group_count = contraction.group_count() as usize;
    let symmetric_group = matches!(
        contraction.runtime_reducer(),
        Some(RuntimeColorContractionReducer::SymmetricGroupFourier(_))
    );
    if contraction.component_count() as usize != direct_helicity_to_physics.len()
        || contraction.destination_by_group().len() != group_count
        || contraction.sector_by_group().len() != group_count
        || contraction.component_by_group().len() != group_count
        || (!symmetric_group && group_count != destination_count)
    {
        return Err(RusticolError::integrity(
            "contracted color dimensions disagree with the public helicity map",
        ));
    }

    let mut destination_physics_helicity = vec![usize::MAX; destination_count];
    for (group_id, destination_id) in contraction
        .destination_by_group()
        .iter()
        .copied()
        .enumerate()
    {
        let expected_direct_helicity = contraction.component_by_group()[group_id] as usize;
        let physics_helicity = *direct_helicity_to_physics
            .get(expected_direct_helicity)
            .ok_or_else(|| {
                RusticolError::integrity("contracted destination has no public helicity mapping")
            })?;
        let slot = destination_physics_helicity
            .get_mut(destination_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted destination mapping exceeds its authenticated range",
                )
            })?;
        if *slot != usize::MAX {
            return Err(RusticolError::integrity(
                "contracted color map repeats an amplitude destination",
            ));
        }
        *slot = physics_helicity;
    }
    if !symmetric_group && destination_physics_helicity.contains(&usize::MAX) {
        return Err(RusticolError::integrity(
            "contracted color map does not cover every amplitude destination",
        ));
    }
    if !symmetric_group {
        for entry in contraction.canonical_logical_entries() {
            if contraction.component_by_group()[entry.left_group_id as usize]
                != contraction.component_by_group()[entry.right_group_id as usize]
            {
                return Err(RusticolError::integrity(
                    "contracted color entry mixes different helicity components",
                ));
            }
        }
    }
    Ok(destination_physics_helicity)
}

fn contracted_selection_has_computed_destination(
    destination_physics_helicity: &[usize],
    physics: &PhysicsRuntime,
    selected_helicities: &BTreeSet<String>,
) -> RusticolResult<bool> {
    for &helicity_index in destination_physics_helicity {
        // Symmetric-group payloads may reserve authenticated amplitude slots
        // that are absent from the active local-group table.
        if helicity_index == usize::MAX {
            continue;
        }
        let helicity = physics
            .manifest
            .helicities
            .get(helicity_index)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted destination selector references an absent public helicity",
                )
            })?;
        if !helicity.computed
            || helicity.structural_zero
            || helicity.coefficient == 0.0
            || recurrence_helicity_orbit_weight(physics, Some(selected_helicities), helicity_index)
                == 0.0
        {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

fn contracted_replay_routes(
    scheduler: &DirectRecurrenceExecutionRuntime,
    contraction: &RecurrenceColorContraction,
) -> RusticolResult<Vec<ContractedReplayRoute>> {
    let plan = scheduler.plan();
    if plan.strategy() != RecurrenceStrategy::ContractedColorUnion {
        return Err(RusticolError::integrity(
            "contracted replay routes require a contracted-color plan",
        ));
    }
    if plan.replay_targets().is_empty() {
        if contraction.destination_count() as usize != plan.amplitude_destinations().len() {
            return Err(RusticolError::integrity(
                "non-replayed contracted color destinations disagree with the Direct plan",
            ));
        }
        return Ok(Vec::new());
    }
    if contraction.sector_count() != plan.physical_sector_count()
        || contraction.component_count() as usize != plan.resolved_helicities().len()
    {
        return Err(RusticolError::integrity(
            "contracted replay domain disagrees with the Direct plan",
        ));
    }

    let destination_count = contraction.destination_count() as usize;
    let component_count = contraction.component_count() as usize;
    let symmetric_group = matches!(
        contraction.runtime_reducer(),
        Some(RuntimeColorContractionReducer::SymmetricGroupFourier(_))
    );
    let mut covered = vec![false; destination_count];
    let mut routes = Vec::with_capacity(plan.replay_targets().len());
    for target in plan.replay_targets() {
        let selector = scheduler.prepare_replay_selector(target.public_flow_id)?;
        let target_sector = selector.public_flow_id() as usize;
        let mut destination_copies = Vec::with_capacity(component_count);
        for destination in plan
            .amplitude_destinations()
            .iter()
            .filter(|destination| destination.target_sector_id == selector.representative_flow_id())
        {
            let direct_helicity = destination.target_helicity_id_or_sentinel;
            if direct_helicity == DIRECT_NONE_U32 {
                return Err(RusticolError::integrity(
                    "contracted replay destination lacks a resolved helicity",
                ));
            }
            let mapped_helicity = selector
                .helicity_map()
                .get(direct_helicity as usize)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity("contracted replay destination helicity is not mapped")
                })? as usize;
            let physical_destination = target_sector
                .checked_mul(component_count)
                .and_then(|offset| offset.checked_add(mapped_helicity))
                .ok_or_else(|| {
                    RusticolError::artifact(
                        "contracted replay physical destination overflows usize",
                    )
                })?;
            if physical_destination >= destination_count || covered[physical_destination] {
                return Err(RusticolError::integrity(
                    "contracted replay route is not a dense physical color/helicity bijection",
                ));
            }
            if !symmetric_group
                && (contraction.sector_by_group()[physical_destination] as usize != target_sector
                    || contraction.component_by_group()[physical_destination] as usize
                        != mapped_helicity
                    || contraction.destination_by_group()[physical_destination] as usize
                        != physical_destination)
            {
                return Err(RusticolError::integrity(
                    "contracted replay route disagrees with the direct color projection",
                ));
            }
            covered[physical_destination] = true;
            destination_copies.push((
                destination.id,
                u32::try_from(physical_destination).map_err(|_| {
                    RusticolError::artifact("contracted replay physical destination exceeds u32")
                })?,
            ));
        }
        if destination_copies.len() != component_count {
            return Err(RusticolError::integrity(
                "contracted replay representative lacks complete helicity coverage",
            ));
        }
        routes.push(ContractedReplayRoute {
            selector,
            destination_copies,
        });
    }
    if covered.contains(&false) {
        return Err(RusticolError::integrity(
            "contracted replay routes do not cover every physical color/helicity destination",
        ));
    }
    Ok(routes)
}

#[allow(clippy::too_many_arguments)]
fn contract_primary_local_color_tile(
    output: PersistedHelicityAmplitudeTileV1<'_>,
    contraction: &RecurrenceColorContraction,
    view: &PrimaryLocalColorViewV1,
    physics: &PhysicsRuntime,
    physics_helicity_id: usize,
    normalization_factor: f64,
    color_transform_re: &mut [f64],
    color_transform_im: &mut [f64],
    symmetric_group_workspace: Option<&mut RuntimeSymmetricGroupColorWorkspace>,
    mut accumulate: impl FnMut(usize, f64),
) -> RusticolResult<()> {
    let helicity = physics
        .manifest
        .helicities
        .get(physics_helicity_id)
        .ok_or_else(|| {
            RusticolError::integrity("persisted selector references an absent public helicity")
        })?;
    if helicity.structural_zero || helicity.coefficient == 0.0 {
        return Err(RusticolError::internal(
            "structural-zero helicity reached persisted color contraction",
        ));
    }
    let scale = helicity.coefficient * normalization_factor;
    match contraction.runtime_reducer() {
        Some(RuntimeColorContractionReducer::Walsh(factorization)) => {
            contract_primary_local_walsh_tile(
                output,
                factorization,
                &view.auxiliary_destination_by_local_group,
                scale,
                color_transform_re,
                color_transform_im,
                &mut accumulate,
            )
        }
        Some(RuntimeColorContractionReducer::SymmetricGroupFourier(reducer)) => {
            let workspace = symmetric_group_workspace.ok_or_else(|| {
                RusticolError::integrity("symmetric-group recurrence color workspace is absent")
            })?;
            contract_primary_local_symmetric_group_tile(
                output,
                reducer,
                workspace,
                &view.auxiliary_destination_by_local_group,
                scale,
                &mut accumulate,
            )
        }
        None => {
            contract_primary_local_direct_tile(output, contraction, view, scale, &mut accumulate)
        }
    }
}

fn contract_primary_local_direct_tile(
    output: PersistedHelicityAmplitudeTileV1<'_>,
    contraction: &RecurrenceColorContraction,
    view: &PrimaryLocalColorViewV1,
    scale: f64,
    accumulate: &mut impl FnMut(usize, f64),
) -> RusticolResult<()> {
    for raw in contraction.entries() {
        let local_group = |stored_group_id: u32| -> RusticolResult<usize> {
            match contraction.storage() {
                RecurrenceColorStorage::Repeated => {
                    let local = stored_group_id as usize;
                    if local >= view.auxiliary_destination_by_local_group.len() {
                        return Err(RusticolError::integrity(
                            "repeated primary color entry is outside its local group domain",
                        ));
                    }
                    Ok(local)
                }
                RecurrenceColorStorage::Expanded => {
                    let primary_destination = contraction
                        .destination_by_group()
                        .get(stored_group_id as usize)
                        .copied()
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "expanded primary color entry references an absent group",
                            )
                        })?;
                    let local = view
                        .local_group_by_primary_destination
                        .get(primary_destination as usize)
                        .copied()
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "expanded primary color destination is out of bounds",
                            )
                        })?;
                    if local == DIRECT_NONE_U32 {
                        return Err(RusticolError::integrity(
                            "expanded primary color destination has no local owner",
                        ));
                    }
                    Ok(local as usize)
                }
                RecurrenceColorStorage::ConvolutionKernels => Err(RusticolError::internal(
                    "factorized primary color reached the direct reducer",
                )),
            }
        };
        let left_local = local_group(raw.left_group_id)?;
        let right_local = local_group(raw.right_group_id)?;
        let entry = RuntimeColorContractionEntry {
            left_destination_id: view.auxiliary_destination_by_local_group[left_local],
            right_destination_id: view.auxiliary_destination_by_local_group[right_local],
            coefficient_re: raw.weight_re * raw.symmetry_factor,
            coefficient_im: raw.weight_im * raw.symmetry_factor,
        };
        let left_re = output.destination_re(entry.left_destination_id)?;
        let left_im = output.destination_im(entry.left_destination_id)?;
        let right_re = output.destination_re(entry.right_destination_id)?;
        let right_im = output.destination_im(entry.right_destination_id)?;
        for point in 0..output.point_count() as usize {
            accumulate(
                point,
                scale
                    * entry.contract_real_bilinear(
                        left_re[point],
                        left_im[point],
                        right_re[point],
                        right_im[point],
                    ),
            );
        }
    }
    Ok(())
}

fn contract_primary_local_symmetric_group_tile(
    output: PersistedHelicityAmplitudeTileV1<'_>,
    reducer: &crate::recurrence::RuntimeSymmetricGroupColorContraction,
    workspace: &mut RuntimeSymmetricGroupColorWorkspace,
    auxiliary_destination_by_local_group: &[u32],
    scale: f64,
    accumulate: &mut impl FnMut(usize, f64),
) -> RusticolResult<()> {
    if reducer.local_group_count() != auxiliary_destination_by_local_group.len() {
        return Err(RusticolError::integrity(
            "symmetric-group primary local-color view has the wrong domain",
        ));
    }
    let point_count = output.point_count() as usize;
    reducer.reduce_lanes(workspace, point_count, |local_group, point| {
        let destination = auxiliary_destination_by_local_group[local_group];
        let real = output.destination_re(destination)?;
        let imaginary = output.destination_im(destination)?;
        Ok((real[point], imaginary[point]))
    })?;
    for (point, contracted) in workspace.reduced(point_count)?.iter().copied().enumerate() {
        accumulate(point, scale * contracted);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn contract_primary_local_walsh_tile(
    output: PersistedHelicityAmplitudeTileV1<'_>,
    factorization: &crate::recurrence::RuntimeFactorizedColorContraction,
    auxiliary_destination_by_local_group: &[u32],
    scale: f64,
    color_transform_re: &mut [f64],
    color_transform_im: &mut [f64],
    accumulate: &mut impl FnMut(usize, f64),
) -> RusticolResult<()> {
    let point_count = output.point_count() as usize;
    let local_group_count = auxiliary_destination_by_local_group.len();
    let scratch_len = local_group_count.checked_mul(point_count).ok_or_else(|| {
        RusticolError::integrity("factorized primary local-color scratch length overflows")
    })?;
    if color_transform_re.len() < scratch_len || color_transform_im.len() < scratch_len {
        return Err(RusticolError::integrity(
            "factorized primary local-color workspace has the wrong shape",
        ));
    }
    let transform_re = &mut color_transform_re[..scratch_len];
    let transform_im = &mut color_transform_im[..scratch_len];
    for (local_group, destination) in auxiliary_destination_by_local_group
        .iter()
        .copied()
        .enumerate()
    {
        let source_re = output.destination_re(destination)?;
        let source_im = output.destination_im(destination)?;
        let start = local_group * point_count;
        transform_re[start..start + point_count].copy_from_slice(source_re);
        transform_im[start..start + point_count].copy_from_slice(source_im);
    }

    if factorization.subgroup_order() == 4 {
        let amplitude_scale = factorization.amplitude_scale();
        for coset in factorization.cosets() {
            let [g0, g1, g2, g3] = <[u32; 4]>::try_from(coset.as_slice()).map_err(|_| {
                RusticolError::integrity("rank-two color factorization has a malformed coset")
            })?;
            let starts = [g0, g1, g2, g3].map(|group| group as usize * point_count);
            for point in 0..point_count {
                let re0 = transform_re[starts[0] + point];
                let re1 = transform_re[starts[1] + point];
                let re2 = transform_re[starts[2] + point];
                let re3 = transform_re[starts[3] + point];
                let im0 = transform_im[starts[0] + point];
                let im1 = transform_im[starts[1] + point];
                let im2 = transform_im[starts[2] + point];
                let im3 = transform_im[starts[3] + point];
                let re01 = re0 + re1;
                let re23 = re2 + re3;
                let re01_difference = re0 - re1;
                let re23_difference = re2 - re3;
                let im01 = im0 + im1;
                let im23 = im2 + im3;
                let im01_difference = im0 - im1;
                let im23_difference = im2 - im3;
                transform_re[starts[0] + point] = (re01 + re23) * amplitude_scale;
                transform_re[starts[1] + point] =
                    (re01_difference + re23_difference) * amplitude_scale;
                transform_re[starts[2] + point] = (re01 - re23) * amplitude_scale;
                transform_re[starts[3] + point] =
                    (re01_difference - re23_difference) * amplitude_scale;
                transform_im[starts[0] + point] = (im01 + im23) * amplitude_scale;
                transform_im[starts[1] + point] =
                    (im01_difference + im23_difference) * amplitude_scale;
                transform_im[starts[2] + point] = (im01 - im23) * amplitude_scale;
                transform_im[starts[3] + point] =
                    (im01_difference - im23_difference) * amplitude_scale;
            }
        }
    } else {
        for coset in factorization.cosets() {
            let mut stride = 1usize;
            while stride < factorization.subgroup_order() {
                for start in (0..factorization.subgroup_order()).step_by(stride * 2) {
                    for offset in 0..stride {
                        let left_start = coset[start + offset] as usize * point_count;
                        let right_start = coset[start + stride + offset] as usize * point_count;
                        for point in 0..point_count {
                            let left_re = transform_re[left_start + point];
                            let right_re = transform_re[right_start + point];
                            let left_im = transform_im[left_start + point];
                            let right_im = transform_im[right_start + point];
                            transform_re[left_start + point] = left_re + right_re;
                            transform_re[right_start + point] = left_re - right_re;
                            transform_im[left_start + point] = left_im + right_im;
                            transform_im[right_start + point] = left_im - right_im;
                        }
                    }
                }
                stride *= 2;
            }
        }
    }

    for entry in factorization.entries() {
        let left_start = entry.left_group_index as usize * point_count;
        let right_start = entry.right_group_index as usize * point_count;
        for point in 0..point_count {
            let product_re = transform_re[left_start + point].mul_add(
                transform_re[right_start + point],
                transform_im[left_start + point] * transform_im[right_start + point],
            );
            let product_im = transform_im[left_start + point].mul_add(
                transform_re[right_start + point],
                -transform_re[left_start + point] * transform_im[right_start + point],
            );
            let contracted = entry
                .coefficient_re
                .mul_add(product_re, -entry.coefficient_im * product_im);
            accumulate(point, scale * contracted);
        }
    }
    Ok(())
}

trait ContractedAmplitudeTile {
    fn point_count(&self) -> usize;
    fn destination_re(&self, destination_id: u32) -> Option<&[f64]>;
    fn destination_im(&self, destination_id: u32) -> Option<&[f64]>;
}

impl ContractedAmplitudeTile for DirectRecurrenceTileOutput<'_> {
    fn point_count(&self) -> usize {
        DirectRecurrenceTileOutput::point_count(self) as usize
    }

    fn destination_re(&self, destination_id: u32) -> Option<&[f64]> {
        DirectRecurrenceTileOutput::destination_re(self, destination_id)
    }

    fn destination_im(&self, destination_id: u32) -> Option<&[f64]> {
        DirectRecurrenceTileOutput::destination_im(self, destination_id)
    }
}

struct ContractedReplayTile<'a> {
    values_re: &'a [f64],
    values_im: &'a [f64],
    point_count: usize,
    point_stride: usize,
    destination_count: usize,
}

impl ContractedAmplitudeTile for ContractedReplayTile<'_> {
    fn point_count(&self) -> usize {
        self.point_count
    }

    fn destination_re(&self, destination_id: u32) -> Option<&[f64]> {
        self.destination(self.values_re, destination_id)
    }

    fn destination_im(&self, destination_id: u32) -> Option<&[f64]> {
        self.destination(self.values_im, destination_id)
    }
}

impl ContractedReplayTile<'_> {
    fn destination<'a>(&self, values: &'a [f64], destination_id: u32) -> Option<&'a [f64]> {
        let destination = destination_id as usize;
        if destination >= self.destination_count {
            return None;
        }
        let start = destination.checked_mul(self.point_stride)?;
        values.get(start..start + self.point_count)
    }
}

// These arguments are the authenticated tile views and selector/reduction
// contract; grouping them would obscure ownership without reducing call state.
#[allow(clippy::too_many_arguments)]
fn contract_color_tile<T: ContractedAmplitudeTile>(
    output: &T,
    contraction: &RecurrenceColorContraction,
    destination_physics_helicity: &[usize],
    physics: &PhysicsRuntime,
    selected_helicities: Option<&BTreeSet<String>>,
    normalization_factor: f64,
    color_transform_re: &mut [f64],
    color_transform_im: &mut [f64],
    symmetric_group_workspace: Option<&mut RuntimeSymmetricGroupColorWorkspace>,
    mut accumulate: impl FnMut(usize, usize, f64),
) -> RusticolResult<()> {
    let point_count = output.point_count();
    match contraction.runtime_reducer() {
        Some(RuntimeColorContractionReducer::Walsh(factorization)) => {
            return contract_factorized_color_tile(
                output,
                contraction,
                factorization,
                destination_physics_helicity,
                physics,
                selected_helicities,
                normalization_factor,
                color_transform_re,
                color_transform_im,
                &mut accumulate,
            );
        }
        Some(RuntimeColorContractionReducer::SymmetricGroupFourier(reducer)) => {
            let workspace = symmetric_group_workspace.ok_or_else(|| {
                RusticolError::integrity("symmetric-group recurrence color workspace is absent")
            })?;
            return contract_symmetric_group_color_tile(
                output,
                contraction,
                reducer,
                workspace,
                destination_physics_helicity,
                physics,
                selected_helicities,
                normalization_factor,
                &mut accumulate,
            );
        }
        None => {}
    }
    for entry in contraction.runtime_entries() {
        let left_helicity = *destination_physics_helicity
            .get(entry.left_destination_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted left destination has no public helicity mapping",
                )
            })?;
        let right_helicity = *destination_physics_helicity
            .get(entry.right_destination_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted right destination has no public helicity mapping",
                )
            })?;
        if left_helicity != right_helicity {
            return Err(RusticolError::integrity(
                "contracted color entry mixes public helicities",
            ));
        }
        let helicity = physics
            .manifest
            .helicities
            .get(left_helicity)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted color entry references an absent public helicity",
                )
            })?;
        if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
            continue;
        }
        if recurrence_helicity_orbit_weight(physics, selected_helicities, left_helicity) == 0.0 {
            continue;
        }
        let left_re = output
            .destination_re(entry.left_destination_id)
            .ok_or_else(|| {
                RusticolError::integrity("contracted left amplitude destination is absent")
            })?;
        let left_im = output
            .destination_im(entry.left_destination_id)
            .ok_or_else(|| {
                RusticolError::integrity("contracted left amplitude destination is absent")
            })?;
        let right_re = output
            .destination_re(entry.right_destination_id)
            .ok_or_else(|| {
                RusticolError::integrity("contracted right amplitude destination is absent")
            })?;
        let right_im = output
            .destination_im(entry.right_destination_id)
            .ok_or_else(|| {
                RusticolError::integrity("contracted right amplitude destination is absent")
            })?;
        for point in 0..point_count {
            let contracted = entry.contract_real_bilinear(
                left_re[point],
                left_im[point],
                right_re[point],
                right_im[point],
            );
            for physical_helicity in physics.helicity_orbit_members(left_helicity) {
                if !recurrence_helicity_is_selected(
                    physics,
                    selected_helicities,
                    *physical_helicity,
                ) {
                    continue;
                }
                let scale = physics.manifest.helicities[*physical_helicity].coefficient
                    * normalization_factor;
                accumulate(point, *physical_helicity, scale * contracted);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn contract_symmetric_group_color_tile<T: ContractedAmplitudeTile>(
    output: &T,
    contraction: &RecurrenceColorContraction,
    reducer: &crate::recurrence::RuntimeSymmetricGroupColorContraction,
    workspace: &mut RuntimeSymmetricGroupColorWorkspace,
    destination_physics_helicity: &[usize],
    physics: &PhysicsRuntime,
    selected_helicities: Option<&BTreeSet<String>>,
    normalization_factor: f64,
    accumulate: &mut impl FnMut(usize, usize, f64),
) -> RusticolResult<()> {
    let point_count = output.point_count();
    let component_count = contraction.component_count() as usize;
    if reducer.local_group_count() != contraction.local_group_count() as usize {
        return Err(RusticolError::integrity(
            "symmetric-group recurrence reducer has the wrong local-group domain",
        ));
    }

    for component_index in 0..component_count {
        let representative_destination = contraction
            .ordered_destination_id(0, component_index)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "symmetric-group recurrence component has no amplitude destination",
                )
            })?;
        let helicity_index = *destination_physics_helicity
            .get(representative_destination as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "symmetric-group recurrence destination has no public helicity mapping",
                )
            })?;
        let helicity = physics
            .manifest
            .helicities
            .get(helicity_index)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "symmetric-group recurrence component references an absent helicity",
                )
            })?;
        if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
            continue;
        }
        if recurrence_helicity_orbit_weight(physics, selected_helicities, helicity_index) == 0.0 {
            continue;
        }

        for local_group_index in 0..reducer.local_group_count() {
            let destination_id = contraction
                .ordered_destination_id(local_group_index, component_index)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "symmetric-group recurrence local group has no amplitude destination",
                    )
                })?;
            if destination_physics_helicity
                .get(destination_id as usize)
                .copied()
                != Some(helicity_index)
                || output.destination_re(destination_id).is_none()
                || output.destination_im(destination_id).is_none()
            {
                return Err(RusticolError::integrity(
                    "symmetric-group recurrence local groups mix helicities or lack amplitudes",
                ));
            }
        }

        reducer.reduce_lanes(workspace, point_count, |local_group_index, point| {
            let destination_id = contraction
                .ordered_destination_id(local_group_index, component_index)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "symmetric-group recurrence local group has no amplitude destination",
                    )
                })?;
            let real = output.destination_re(destination_id).ok_or_else(|| {
                RusticolError::integrity(
                    "symmetric-group recurrence real amplitude destination is absent",
                )
            })?;
            let imaginary = output.destination_im(destination_id).ok_or_else(|| {
                RusticolError::integrity(
                    "symmetric-group recurrence imaginary amplitude destination is absent",
                )
            })?;
            Ok((
                *real.get(point).ok_or_else(|| {
                    RusticolError::integrity(
                        "symmetric-group recurrence real amplitude tile is too short",
                    )
                })?,
                *imaginary.get(point).ok_or_else(|| {
                    RusticolError::integrity(
                        "symmetric-group recurrence imaginary amplitude tile is too short",
                    )
                })?,
            ))
        })?;

        for (point, contracted) in workspace.reduced(point_count)?.iter().copied().enumerate() {
            for physical_helicity in physics.helicity_orbit_members(helicity_index) {
                if !recurrence_helicity_is_selected(
                    physics,
                    selected_helicities,
                    *physical_helicity,
                ) {
                    continue;
                }
                let scale = physics.manifest.helicities[*physical_helicity].coefficient
                    * normalization_factor;
                accumulate(point, *physical_helicity, scale * contracted);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn contract_factorized_color_tile<T: ContractedAmplitudeTile>(
    output: &T,
    contraction: &RecurrenceColorContraction,
    factorization: &crate::recurrence::RuntimeFactorizedColorContraction,
    destination_physics_helicity: &[usize],
    physics: &PhysicsRuntime,
    selected_helicities: Option<&BTreeSet<String>>,
    normalization_factor: f64,
    color_transform_re: &mut [f64],
    color_transform_im: &mut [f64],
    accumulate: &mut impl FnMut(usize, usize, f64),
) -> RusticolResult<()> {
    let point_count = output.point_count();
    let local_group_count = contraction.local_group_count() as usize;
    let scratch_len = local_group_count.checked_mul(point_count).ok_or_else(|| {
        RusticolError::integrity("factorized recurrence color scratch length overflows")
    })?;
    if color_transform_re.len() < scratch_len || color_transform_im.len() < scratch_len {
        return Err(RusticolError::integrity(
            "factorized recurrence color workspace is smaller than the point tile",
        ));
    }
    let transform_re = &mut color_transform_re[..scratch_len];
    let transform_im = &mut color_transform_im[..scratch_len];
    let component_count = contraction.component_count() as usize;

    for component_index in 0..component_count {
        let representative_destination = contraction
            .ordered_destination_id(0, component_index)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence component has no amplitude destination",
                )
            })?;
        let helicity_index = *destination_physics_helicity
            .get(representative_destination as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence destination has no public helicity mapping",
                )
            })?;
        let helicity = physics
            .manifest
            .helicities
            .get(helicity_index)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence color component references an absent helicity",
                )
            })?;
        if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
            continue;
        }
        if recurrence_helicity_orbit_weight(physics, selected_helicities, helicity_index) == 0.0 {
            continue;
        }

        for local_group_index in 0..local_group_count {
            let destination_id = contraction
                .ordered_destination_id(local_group_index, component_index)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "factorized recurrence local group has no amplitude destination",
                    )
                })?;
            if destination_physics_helicity
                .get(destination_id as usize)
                .copied()
                != Some(helicity_index)
            {
                return Err(RusticolError::integrity(
                    "factorized recurrence local groups mix public helicities",
                ));
            }
            let source_re = output.destination_re(destination_id).ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence real amplitude destination is absent",
                )
            })?;
            let source_im = output.destination_im(destination_id).ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence imaginary amplitude destination is absent",
                )
            })?;
            let start = local_group_index * point_count;
            transform_re[start..start + point_count].copy_from_slice(source_re);
            transform_im[start..start + point_count].copy_from_slice(source_im);
        }

        if factorization.subgroup_order() == 4 {
            let amplitude_scale = factorization.amplitude_scale();
            for coset in factorization.cosets() {
                let [g0, g1, g2, g3] = <[u32; 4]>::try_from(coset.as_slice()).map_err(|_| {
                    RusticolError::integrity(
                        "rank-two recurrence color factorization has a malformed coset",
                    )
                })?;
                let starts = [
                    g0 as usize * point_count,
                    g1 as usize * point_count,
                    g2 as usize * point_count,
                    g3 as usize * point_count,
                ];
                for point in 0..point_count {
                    let re0 = transform_re[starts[0] + point];
                    let re1 = transform_re[starts[1] + point];
                    let re2 = transform_re[starts[2] + point];
                    let re3 = transform_re[starts[3] + point];
                    let im0 = transform_im[starts[0] + point];
                    let im1 = transform_im[starts[1] + point];
                    let im2 = transform_im[starts[2] + point];
                    let im3 = transform_im[starts[3] + point];
                    let re01 = re0 + re1;
                    let re23 = re2 + re3;
                    let re01_difference = re0 - re1;
                    let re23_difference = re2 - re3;
                    let im01 = im0 + im1;
                    let im23 = im2 + im3;
                    let im01_difference = im0 - im1;
                    let im23_difference = im2 - im3;
                    transform_re[starts[0] + point] = (re01 + re23) * amplitude_scale;
                    transform_re[starts[1] + point] =
                        (re01_difference + re23_difference) * amplitude_scale;
                    transform_re[starts[2] + point] = (re01 - re23) * amplitude_scale;
                    transform_re[starts[3] + point] =
                        (re01_difference - re23_difference) * amplitude_scale;
                    transform_im[starts[0] + point] = (im01 + im23) * amplitude_scale;
                    transform_im[starts[1] + point] =
                        (im01_difference + im23_difference) * amplitude_scale;
                    transform_im[starts[2] + point] = (im01 - im23) * amplitude_scale;
                    transform_im[starts[3] + point] =
                        (im01_difference - im23_difference) * amplitude_scale;
                }
            }
        } else {
            for coset in factorization.cosets() {
                let mut stride = 1usize;
                while stride < factorization.subgroup_order() {
                    for start in (0..factorization.subgroup_order()).step_by(stride * 2) {
                        for offset in 0..stride {
                            let left_group = coset[start + offset] as usize;
                            let right_group = coset[start + stride + offset] as usize;
                            let left_start = left_group * point_count;
                            let right_start = right_group * point_count;
                            for point in 0..point_count {
                                let left_re = transform_re[left_start + point];
                                let right_re = transform_re[right_start + point];
                                let left_im = transform_im[left_start + point];
                                let right_im = transform_im[right_start + point];
                                transform_re[left_start + point] = left_re + right_re;
                                transform_re[right_start + point] = left_re - right_re;
                                transform_im[left_start + point] = left_im + right_im;
                                transform_im[right_start + point] = left_im - right_im;
                            }
                        }
                    }
                    stride *= 2;
                }
            }
        }

        for entry in factorization.entries() {
            let left_start = entry.left_group_index as usize * point_count;
            let right_start = entry.right_group_index as usize * point_count;
            for point in 0..point_count {
                let product_re = transform_re[left_start + point].mul_add(
                    transform_re[right_start + point],
                    transform_im[left_start + point] * transform_im[right_start + point],
                );
                let product_im = transform_im[left_start + point].mul_add(
                    transform_re[right_start + point],
                    -transform_re[left_start + point] * transform_im[right_start + point],
                );
                let contracted = entry
                    .coefficient_re
                    .mul_add(product_re, -entry.coefficient_im * product_im);
                for physical_helicity in physics.helicity_orbit_members(helicity_index) {
                    if !recurrence_helicity_is_selected(
                        physics,
                        selected_helicities,
                        *physical_helicity,
                    ) {
                        continue;
                    }
                    let scale = physics.manifest.helicities[*physical_helicity].coefficient
                        * normalization_factor;
                    accumulate(point, *physical_helicity, scale * contracted);
                }
            }
        }
    }
    Ok(())
}

fn union_destination_ids(
    plan: &DirectRecurrencePlan,
    public_flow_ids: &[u32],
) -> RusticolResult<Vec<u32>> {
    if plan.strategy() != RecurrenceStrategy::AllFlowUnion {
        return Err(RusticolError::integrity(
            "union destination mapping requires an all-flow-union plan",
        ));
    }
    let mut destination_by_sector = BTreeMap::new();
    for destination in plan.amplitude_destinations() {
        if destination.target_helicity_id_or_sentinel != DIRECT_NONE_U32 {
            return Err(RusticolError::integrity(
                "all-flow-union amplitude destination fixes a numerical helicity",
            ));
        }
        if destination_by_sector
            .insert(destination.target_sector_id, destination.id)
            .is_some()
        {
            return Err(RusticolError::integrity(
                "all-flow-union repeats a physical-flow destination",
            ));
        }
    }
    let result = public_flow_ids
        .iter()
        .map(|sector_id| {
            destination_by_sector
                .get(sector_id)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union public flow has no amplitude destination",
                    )
                })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if result.iter().copied().collect::<BTreeSet<_>>()
        != destination_by_sector
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
    {
        return Err(RusticolError::integrity(
            "all-flow-union amplitude destinations do not match the public flow axis",
        ));
    }
    Ok(result)
}

fn validate_replay_destination_helicity_mappings(
    plan: &DirectRecurrencePlan,
    replay_selectors: &[DirectReplaySelectorPlan],
    direct_helicity_to_physics: &[usize],
) -> RusticolResult<()> {
    for destination in plan.amplitude_destinations() {
        replay_target_helicity_index(
            destination.target_helicity_id_or_sentinel,
            plan.resolved_helicities().len(),
        )?;
    }

    for selector in replay_selectors {
        if selector.helicity_map().len() != plan.resolved_helicities().len() {
            return Err(RusticolError::integrity(format!(
                "recurrence replay flow {} has incomplete helicity coverage",
                selector.public_flow_id()
            )));
        }
        for mapped_direct_id in selector.helicity_map() {
            replay_mapped_direct_physics_helicity(direct_helicity_to_physics, *mapped_direct_id)?;
        }
    }
    Ok(())
}

/// Resolve selection in the physical helicity domain, after the requested
/// flow's relabelling. A parity/numerical alias requests its computed
/// representative's destination; a structural zero requests no work.
fn selected_replay_destination_ids(
    destinations: &[DirectAmplitudeDestinationDescriptor],
    representative_flow_id: u32,
    replay_helicity_map: &[u32],
    direct_helicity_to_physics: &[usize],
    reduction: LcSelectorReductionView<'_>,
    selected_helicities: &BTreeSet<String>,
) -> RusticolResult<Vec<u32>> {
    let mut selected = Vec::new();
    for destination in destinations {
        if destination.target_sector_id != representative_flow_id {
            continue;
        }
        let physics_helicity = replay_destination_physics_helicity(
            replay_helicity_map,
            direct_helicity_to_physics,
            destination.target_helicity_id_or_sentinel,
        )?;
        let helicity = reduction.helicity(physics_helicity);
        if helicity.computed
            && !helicity.structural_zero
            && helicity.coefficient != 0.0
            && reduction.helicity_orbit_weight(Some(selected_helicities), physics_helicity) != 0.0
        {
            selected.push(destination.id);
        }
    }
    Ok(selected)
}

fn replay_destination_physics_helicity(
    replay_helicity_map: &[u32],
    direct_helicity_to_physics: &[usize],
    target_helicity_id_or_sentinel: u32,
) -> RusticolResult<usize> {
    let target_helicity_index =
        replay_target_helicity_index(target_helicity_id_or_sentinel, replay_helicity_map.len())?;
    let mapped_direct_id = replay_helicity_map[target_helicity_index];
    replay_mapped_direct_physics_helicity(direct_helicity_to_physics, mapped_direct_id)
}

fn replay_target_helicity_index(
    target_helicity_id_or_sentinel: u32,
    helicity_count: usize,
) -> RusticolResult<usize> {
    if target_helicity_id_or_sentinel == DIRECT_NONE_U32 {
        return Err(RusticolError::integrity(
            "topology-replay amplitude destination lacks a resolved helicity",
        ));
    }
    let target_helicity_index = target_helicity_id_or_sentinel as usize;
    if target_helicity_index >= helicity_count {
        return Err(RusticolError::integrity(
            "recurrence amplitude destination helicity is not replay-mapped",
        ));
    }
    Ok(target_helicity_index)
}

fn replay_mapped_direct_physics_helicity(
    direct_helicity_to_physics: &[usize],
    mapped_direct_id: u32,
) -> RusticolResult<usize> {
    direct_helicity_to_physics
        .get(mapped_direct_id as usize)
        .copied()
        .ok_or_else(|| {
            RusticolError::integrity("recurrence replay helicity has no public physics mapping")
        })
}

#[cfg(test)]
mod replay_destination_helicity_tests {
    use super::*;

    fn auxiliary_physical_destinations(count: u32) -> Vec<DirectAmplitudeDestinationDescriptor> {
        (0..count)
            .map(|id| DirectAmplitudeDestinationDescriptor {
                closure_row_start: 0,
                id,
                target_sector_id: id,
                target_helicity_id_or_sentinel: DIRECT_NONE_U32,
                closure_row_count: 0,
                selector_domain_id: 0,
            })
            .collect()
    }

    #[test]
    fn primary_local_color_view_uses_dense_owner_ranks_and_supports_expanded_one_component() {
        let contraction =
            RecurrenceColorContraction::symmetric_group_s3_for_runtime_test((0..13).collect(), 13)
                .with_sparse_sector_domain_for_runtime_test();
        let mut auxiliary = auxiliary_physical_destinations(13);
        for destination in &mut auxiliary {
            destination.id = 12 - destination.id;
        }
        auxiliary.reverse();
        let view = primary_local_color_destination_map(&contraction, &auxiliary).unwrap();
        assert_eq!(
            view.auxiliary_destination_by_local_group.as_ref(),
            &[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        );

        let expanded = RecurrenceColorContraction::expanded_identity_for_runtime_test();
        let expanded_view =
            primary_local_color_destination_map(&expanded, &auxiliary_physical_destinations(1))
                .unwrap();
        assert_eq!(
            expanded_view.auxiliary_destination_by_local_group.as_ref(),
            &[0]
        );
        assert_eq!(
            expanded_view.local_group_by_primary_destination.as_ref(),
            &[0]
        );
    }

    #[test]
    fn primary_local_color_view_rejects_reordered_auxiliary_sector_identity() {
        let contraction =
            RecurrenceColorContraction::symmetric_group_s3_for_runtime_test((0..13).collect(), 13)
                .with_sparse_sector_domain_for_runtime_test();
        let mut auxiliary = auxiliary_physical_destinations(13);
        auxiliary[0].target_sector_id = 1;
        let error = match primary_local_color_destination_map(&contraction, &auxiliary) {
            Ok(_) => panic!("duplicate/missing dense auxiliary owner must fail closed"),
            Err(error) => error,
        };
        assert!(
            error
                .message()
                .contains("repeats an auxiliary physical sector")
        );
    }

    #[test]
    fn persisted_companion_census_keeps_legacy_outer_contract() {
        let cold = RecurrenceNativeRuntime::persisted_helicity_selector_state_census(
            "gg_to_gg",
            PersistedHelicityFamilyCacheCensusV1::default(),
            None,
        );
        assert_eq!(cold["kind"], "rusticol-on-the-fly-runtime-state-census-v1");
        assert_eq!(cold["family_cache_policy"], "last-family-only");
        assert_eq!(cold["family_cache_limit"], 1);
        assert_eq!(cold["retained_family_count"], 0);
        assert_eq!(cold["retained_executor_handle_count"], 0);
        assert!(cold["active_family_union_census"].is_null());

        let warm = RecurrenceNativeRuntime::persisted_helicity_selector_state_census(
            "gg_to_gg",
            PersistedHelicityFamilyCacheCensusV1 {
                resolved_helicity_count: 4,
                retained_family_count: 1,
                retained_row_count: 17,
                active_resolved_helicity_id: Some(2),
            },
            Some(PersistedHelicityFamilyInspectionCensusV1 {
                query_count: 3,
                union_unique_current_count: 5,
                union_unique_current_component_count: 9,
                union_source_rows: 2,
                union_contribution_rows: 7,
                union_finalization_rows: 2,
                union_closure_rows: 3,
                union_amplitude_destination_count: 3,
                union_source_executor_call_groups: 1,
                union_contribution_executor_call_groups: 2,
                union_finalization_executor_call_groups: 1,
                union_closure_executor_call_groups: 1,
                semantic_executor_binding_count: 4,
            }),
        );
        assert_eq!(warm["process_preparation_count"], 1);
        assert_eq!(warm["retained_family_count"], 1);
        assert_eq!(warm["retained_selection_count"], 1);
        assert_eq!(warm["retained_request_count"], 3);
        assert_eq!(warm["retained_amplitude_destination_count"], 3);
        assert_eq!(warm["retained_executor_handle_count"], 1);
        assert_eq!(warm["semantic_executor_binding_count"], 4);
        assert_eq!(
            warm["active_family_union_census"]["union_unique_current_count"],
            5
        );
        assert_eq!(warm["active_family_union_census"]["union_closure_rows"], 3);
        let fields = warm
            .as_object()
            .expect("persisted census must be an object")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            fields,
            BTreeSet::from([
                "kind",
                "process_id",
                "family_cache_policy",
                "family_cache_limit",
                "process_preparation_count",
                "retained_family_count",
                "pending_family_count",
                "retained_selection_count",
                "retained_request_count",
                "retained_amplitude_destination_count",
                "retained_executor_handle_count",
                "retained_query_local_trace_count",
                "retained_embedded_lookup_key_count",
                "semantic_executor_binding_count",
                "active_family_union_census",
            ])
        );
    }

    #[test]
    fn prepared_parameter_projection_validates_before_modifying_planes() {
        let defaults = [
            crate::EagerComplex64::new(1.0, -1.0),
            crate::EagerComplex64::new(2.0, -2.0),
        ];
        let runtime_values = [11.0, 22.0];
        let mut real = [91.0, 92.0];
        let mut imaginary = [-91.0, -92.0];
        let invalid = [PreparedParameterProjectionEntry {
            runtime_slot: 2,
            prepared_slot: 0,
            component: 0,
        }];

        initialize_prepared_parameter_planes(
            &defaults,
            &invalid,
            &runtime_values,
            &mut real,
            &mut imaginary,
        )
        .expect_err("invalid projection must fail closed");
        assert_eq!(real, [91.0, 92.0]);
        assert_eq!(imaginary, [-91.0, -92.0]);

        let projection = [
            PreparedParameterProjectionEntry {
                runtime_slot: 1,
                prepared_slot: 0,
                component: 0,
            },
            PreparedParameterProjectionEntry {
                runtime_slot: 0,
                prepared_slot: 1,
                component: 1,
            },
        ];
        initialize_prepared_parameter_planes(
            &defaults,
            &projection,
            &runtime_values,
            &mut real,
            &mut imaginary,
        )
        .unwrap();
        assert_eq!(real, [22.0, 2.0]);
        assert_eq!(imaginary, [-1.0, 11.0]);
    }

    #[test]
    fn resolved_lc_output_layout_is_point_helicity_color_major() {
        let layout = LcResolvedOutputLayout::new(2, 3, 4).unwrap();
        assert_eq!(layout.output_len().unwrap(), 24);
        assert_eq!(layout.index(0, 0, 0).unwrap(), 0);
        assert_eq!(layout.index(0, 1, 0).unwrap(), 4);
        assert_eq!(layout.index(1, 0, 0).unwrap(), 12);
        assert_eq!(layout.index(1, 2, 3).unwrap(), 23);
        assert!(layout.index(2, 0, 0).is_err());
        assert!(LcResolvedOutputLayout::new(1, 0, 1).is_err());
    }

    #[test]
    fn compact_lc_selector_reduction_view_matches_topology_replay_semantics() {
        let helicities = vec![
            crate::Helicity {
                id: "h0".into(),
                index: 0,
                values: vec![1, -1],
                computed: true,
                structural_zero: false,
                representative_id: "h0".into(),
                coefficient: 0.25,
            },
            crate::Helicity {
                id: "h1".into(),
                index: 1,
                values: vec![-1, 1],
                computed: false,
                structural_zero: false,
                representative_id: "h0".into(),
                coefficient: 0.75,
            },
        ];
        let colors = vec![
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "c0".into(),
                index: 0,
                word: vec![0, 1],
                computed: true,
                representative_id: "c0".into(),
                coefficient: 2.0,
            }),
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "c1".into(),
                index: 1,
                word: vec![1, 0],
                computed: false,
                representative_id: "c0".into(),
                coefficient: 3.0,
            }),
        ];
        let helicity_ids = BTreeMap::from([("h0".into(), 0), ("h1".into(), 1)]);
        let color_ids = BTreeMap::from([("c0".into(), 0), ("c1".into(), 1)]);
        let orbits = vec![vec![0, 1], Vec::new()];
        let view = LcSelectorReductionView::from_parts(
            &helicities,
            &colors,
            &helicity_ids,
            &color_ids,
            &orbits,
        );
        let selected_helicities = BTreeSet::from(["h1".to_string()]);
        let selected_colors = BTreeSet::from(["c0".to_string()]);

        view.validate_selector_ids(Some(&selected_helicities), Some(&selected_colors))
            .unwrap();
        assert_eq!(
            view.selected_helicity_indices(Some(&selected_helicities))
                .unwrap(),
            [1]
        );
        assert_eq!(
            view.selected_color_indices(Some(&selected_colors)).unwrap(),
            [0]
        );
        assert_eq!(view.helicity_orbit_weight(None, 0), 1.0);
        assert_eq!(
            view.helicity_orbit_weight(Some(&selected_helicities), 0),
            0.75
        );
        assert!(view.color_is_computed(0));
        assert!(!view.color_is_computed(1));
        assert!(view.color_is_selected(Some(&selected_colors), 0));
        assert!(
            view.validate_selector_ids(Some(&BTreeSet::from(["missing".into()])), None)
                .is_err()
        );
    }

    #[test]
    fn selected_replay_destinations_follow_flow_permutation_and_physical_orbits() {
        let helicities = (0..5)
            .map(|index| crate::Helicity {
                id: format!("h{index}"),
                index,
                values: Vec::new(),
                computed: !matches!(index, 1 | 3),
                structural_zero: index == 3,
                representative_id: format!("h{}", if index == 1 { 0 } else { index }),
                coefficient: if index == 4 { 0.0 } else { 1.0 },
            })
            .collect::<Vec<_>>();
        let helicity_ids = helicities
            .iter()
            .enumerate()
            .map(|(index, helicity)| (helicity.id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let colors = BTreeMap::new();
        let orbits = vec![vec![0, 1], vec![], vec![2], vec![], vec![4]];
        let reduction =
            LcSelectorReductionView::from_parts(&helicities, &[], &helicity_ids, &colors, &orbits);
        let mut destinations = (0..4)
            .map(|id| DirectAmplitudeDestinationDescriptor {
                id,
                target_sector_id: 7,
                target_helicity_id_or_sentinel: id,
                closure_row_start: 0,
                closure_row_count: 0,
                selector_domain_id: 0,
            })
            .collect::<Vec<_>>();
        destinations.push(DirectAmplitudeDestinationDescriptor {
            id: 4,
            target_sector_id: 8,
            target_helicity_id_or_sentinel: 0,
            closure_row_start: 0,
            closure_row_count: 0,
            selector_domain_id: 0,
        });
        let selected = |ids: &[&str]| {
            selected_replay_destination_ids(
                &destinations,
                7,
                &[1, 0, 2, 3],
                &[0, 2, 3, 4],
                reduction,
                &ids.iter().map(|id| (*id).to_string()).collect(),
            )
            .unwrap()
        };

        // Selecting an uncomputed physical alias must retain the current of
        // its computed representative, after the non-identity flow map.
        assert_eq!(selected(&["h1"]), [1]);
        assert_eq!(selected(&["h0"]), [1]);
        assert_eq!(selected(&["h2"]), [0]);
        assert_eq!(selected(&["h1", "h2"]), [0, 1]);
        assert_eq!(selected(&["h0", "h1"]), [1]);
        assert!(selected(&["h3"]).is_empty());
        assert!(selected(&["h4"]).is_empty());
        assert!(selected(&[]).is_empty());
        let invalid = BTreeSet::from(["not-a-helicity".to_string()]);
        assert!(
            reduction
                .validate_selector_ids(Some(&invalid), None)
                .is_err()
        );
    }

    #[test]
    fn unprofiled_replay_selected_colors_are_sorted_by_public_index() {
        let helicity_ids = BTreeMap::new();
        let color_ids = BTreeMap::from([
            ("alpha".to_string(), 4),
            ("middle".to_string(), 0),
            ("zeta".to_string(), 2),
        ]);
        let view = LcSelectorReductionView::from_parts(&[], &[], &helicity_ids, &color_ids, &[]);
        let selected = BTreeSet::from([
            "alpha".to_string(),
            "middle".to_string(),
            "zeta".to_string(),
        ]);
        let mut scratch = vec![usize::MAX];

        view.retain_selected_color_indices(&selected, &mut scratch)
            .unwrap();

        assert_eq!(scratch, [0, 2, 4]);
        let mut visited = Vec::new();
        visit_replay_color_indices(5, Some(&scratch), |color_index| {
            visited.push(color_index);
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, scratch);
    }

    #[test]
    fn unprofiled_replay_all_colors_keep_public_axis_order() {
        let mut visited = Vec::new();
        visit_replay_color_indices(4, None, |color_index| {
            visited.push(color_index);
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, [0, 1, 2, 3]);
    }

    #[test]
    fn unprofiled_replay_color_scratch_retains_allocation() {
        let helicity_ids = BTreeMap::new();
        let color_ids = BTreeMap::from([
            ("c0".to_string(), 0),
            ("c1".to_string(), 1),
            ("c2".to_string(), 2),
            ("c3".to_string(), 3),
        ]);
        let view = LcSelectorReductionView::from_parts(&[], &[], &helicity_ids, &color_ids, &[]);
        let all = BTreeSet::from([
            "c0".to_string(),
            "c1".to_string(),
            "c2".to_string(),
            "c3".to_string(),
        ]);
        let subset = BTreeSet::from(["c1".to_string(), "c3".to_string()]);
        let mut scratch = Vec::new();

        view.retain_selected_color_indices(&all, &mut scratch)
            .unwrap();
        let capacity = scratch.capacity();
        let allocation = scratch.as_ptr();
        view.retain_selected_color_indices(&subset, &mut scratch)
            .unwrap();

        assert_eq!(scratch, [1, 3]);
        assert_eq!(scratch.capacity(), capacity);
        assert_eq!(scratch.as_ptr(), allocation);
    }

    #[test]
    fn composes_distinct_replay_flow_permutations_without_destination_tables() {
        let direct_helicity_to_physics = [2, 0, 1];
        let flow_zero = [0, 1, 2];
        let flow_one = [2, 0, 1];

        for (replay_helicity_map, expected) in
            [(&flow_zero[..], [2, 0, 1]), (&flow_one[..], [1, 2, 0])]
        {
            let actual = (0..3)
                .map(|target_helicity_id| {
                    replay_destination_physics_helicity(
                        replay_helicity_map,
                        &direct_helicity_to_physics,
                        target_helicity_id,
                    )
                    .expect("compose authenticated replay and physics mappings")
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn rejects_absent_or_out_of_range_replay_helicity_mappings() {
        let missing = replay_destination_physics_helicity(&[0], &[0], DIRECT_NONE_U32)
            .expect_err("sentinel destination helicity must fail closed");
        assert_eq!(
            missing.message(),
            "topology-replay amplitude destination lacks a resolved helicity"
        );

        let incomplete = replay_destination_physics_helicity(&[0], &[0], 1)
            .expect_err("incomplete replay permutation must fail closed");
        assert_eq!(
            incomplete.message(),
            "recurrence amplitude destination helicity is not replay-mapped"
        );

        let out_of_range = replay_destination_physics_helicity(&[1], &[0], 0)
            .expect_err("out-of-range direct helicity must fail closed");
        assert_eq!(
            out_of_range.message(),
            "recurrence replay helicity has no public physics mapping"
        );
    }

    #[test]
    fn symmetric_group_recurrence_workspace_is_lazy_and_first_allocation_is_bounded() {
        let contraction =
            RecurrenceColorContraction::symmetric_group_s3_for_runtime_test((0..13).collect(), 13);
        let RuntimeColorContractionReducer::SymmetricGroupFourier(reducer) =
            contraction.runtime_reducer().unwrap()
        else {
            panic!("expected symmetric-group runtime reducer")
        };
        let (effective_point_tile_size, workspace) =
            lazy_symmetric_group_workspace_for_point_tile(reducer, 5).unwrap();
        assert_eq!(effective_point_tile_size, 5);
        assert!(workspace.is_none());

        let mut workspace = None;
        ensure_symmetric_group_color_workspace_capacity(&mut workspace, reducer, 1).unwrap();
        assert_eq!(workspace.as_ref().unwrap().lane_capacity(), 1);

        ensure_symmetric_group_color_workspace_capacity(&mut workspace, reducer, 1).unwrap();
        assert_eq!(workspace.as_ref().unwrap().lane_capacity(), 1);

        ensure_symmetric_group_color_workspace_capacity(&mut workspace, reducer, 3).unwrap();
        assert_eq!(workspace.as_ref().unwrap().lane_capacity(), 3);

        ensure_symmetric_group_color_workspace_capacity(&mut workspace, reducer, 2).unwrap();
        assert_eq!(workspace.as_ref().unwrap().lane_capacity(), 3);
    }

    #[test]
    fn recurrence_external_planes_match_nested_contiguous_and_crossed_views() {
        let source_count = 3;
        let values = (0..7 * source_count * 4)
            .map(|index| {
                if index % 9 == 0 {
                    -0.0
                } else {
                    index as f64 + 0.5
                }
            })
            .collect::<Vec<_>>();
        let crossing = [
            InputCrossingMapEntry {
                target_index: 0,
                source_index: 2,
                sign: -1.0,
            },
            InputCrossingMapEntry {
                target_index: 1,
                source_index: 0,
                sign: 1.0,
            },
            InputCrossingMapEntry {
                target_index: 2,
                source_index: 1,
                sign: -1.0,
            },
        ];
        for lookup in [None, Some(crossing.as_slice())] {
            let contiguous = F64MomentumBatchView::from_contiguous_prevalidated(
                &values,
                7,
                source_count,
                lookup,
            )
            .unwrap();
            let nested = contiguous.materialize_nested();
            let nested_view = F64MomentumBatchView::from_nested(&nested, source_count).unwrap();
            for (start, stop) in [(0, 7), (2, 6), (6, 7)] {
                let count = stop - start;
                let mut expected = vec![0.0_f64; source_count * 4 * count];
                for source in 0..source_count {
                    for component in 0..4 {
                        for point in 0..count {
                            expected[(source * 4 + component) * count + point] =
                                nested[start + point][source][component];
                        }
                    }
                }
                for view in [contiguous, nested_view] {
                    let view = view.subview(start, stop).unwrap();
                    let mut source_major = vec![17.0; expected.len() + 8];
                    let mut point_major = vec![17.0; expected.len() + 8];
                    for point in 0..count {
                        for source in 0..source_count {
                            let momentum = view.point(point).momentum(source).unwrap();
                            write_external_momentum::<true>(
                                &mut source_major,
                                count,
                                source_count,
                                point,
                                source,
                                &momentum,
                            );
                            write_external_momentum::<false>(
                                &mut point_major,
                                count,
                                source_count,
                                point,
                                source,
                                &momentum,
                            );
                            assert_eq!(
                                point_major[(point * source_count + source) * 4
                                    ..(point * source_count + source + 1) * 4],
                                momentum
                            );
                        }
                    }
                    assert_eq!(
                        source_major[..expected.len()]
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
                    );
                    assert!(
                        source_major[expected.len()..]
                            .iter()
                            .all(|&value| value == 17.0)
                    );
                    assert!(
                        point_major[expected.len()..]
                            .iter()
                            .all(|&value| value == 17.0)
                    );
                }
            }
        }
    }

    #[test]
    fn recurrence_external_momentum_workspace_starts_singleton_and_grows_without_shrinking() {
        let external_source_count = 4;
        let maximum_point_count = 5;
        let mut workspace = Vec::new();

        let singleton_len = ensure_external_momentum_workspace_capacity(
            &mut workspace,
            1,
            external_source_count,
            maximum_point_count,
        )
        .unwrap();
        assert_eq!(singleton_len, external_source_count * 4);
        assert_eq!(workspace.len(), singleton_len);
        workspace[0] = 17.0;

        let singleton_ptr = workspace.as_ptr();
        let singleton_capacity = workspace.capacity();
        ensure_external_momentum_workspace_capacity(
            &mut workspace,
            1,
            external_source_count,
            maximum_point_count,
        )
        .unwrap();
        assert_eq!(workspace.as_ptr(), singleton_ptr);
        assert_eq!(workspace.capacity(), singleton_capacity);

        let grown_len = ensure_external_momentum_workspace_capacity(
            &mut workspace,
            3,
            external_source_count,
            maximum_point_count,
        )
        .unwrap();
        assert_eq!(grown_len, 3 * external_source_count * 4);
        assert_eq!(workspace.len(), grown_len);
        assert_eq!(workspace[0], 17.0);
        assert!(workspace[singleton_len..].iter().all(|value| *value == 0.0));

        let grown_ptr = workspace.as_ptr();
        let grown_capacity = workspace.capacity();
        ensure_external_momentum_workspace_capacity(
            &mut workspace,
            2,
            external_source_count,
            maximum_point_count,
        )
        .unwrap();
        assert_eq!(workspace.len(), grown_len);
        assert_eq!(workspace.as_ptr(), grown_ptr);
        assert_eq!(workspace.capacity(), grown_capacity);

        let error = ensure_external_momentum_workspace_capacity(
            &mut workspace,
            maximum_point_count + 1,
            external_source_count,
            maximum_point_count,
        )
        .expect_err("workspace growth past the persistent tile limit must fail");
        assert_eq!(
            error.message(),
            "recurrence point tile exceeds its persistent workspace"
        );
        assert_eq!(workspace.len(), grown_len);
    }

    #[test]
    fn symmetric_group_recurrence_dispatch_handles_sparse_owner_destinations_and_partial_tiles() {
        let contraction = RecurrenceColorContraction::symmetric_group_s3_for_runtime_test(
            (0..13).map(|group| group + 2).collect(),
            20,
        );
        let destination_physics_helicity =
            contracted_destination_helicity_map(&contraction, &[0]).unwrap();
        assert_eq!(destination_physics_helicity.len(), 20);
        assert_eq!(destination_physics_helicity[..2], [usize::MAX; 2]);
        assert!(
            destination_physics_helicity[2..15]
                .iter()
                .all(|helicity| *helicity == 0)
        );
        assert!(
            destination_physics_helicity[15..]
                .iter()
                .all(|helicity| *helicity == usize::MAX)
        );

        let mut physics = PhysicsRuntime {
            binding_id: 0,
            manifest: ProcessPhysicsV1 {
                schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
                kind: "pyamplicol-resolved-physics".to_string(),
                process_id: "fft-runtime-test".to_string(),
                process: "g g > g g".to_string(),
                color_accuracy: crate::ColorAccuracy::Full,
                coverage: crate::Coverage {
                    helicities: "complete".to_string(),
                    color: "contracted".to_string(),
                    color_kind: "contracted-color".to_string(),
                    structural_zero_helicity_count: 0,
                },
                external_particles: Vec::new(),
                helicities: vec![crate::Helicity {
                    id: "h0".to_string(),
                    index: 0,
                    values: Vec::new(),
                    representative_id: "h0".to_string(),
                    computed: true,
                    structural_zero: false,
                    coefficient: 1.0,
                }],
                color_components: vec![crate::ColorComponent::ContractedColor(
                    crate::ContractedColor {
                        id: "contracted".to_string(),
                        index: 0,
                        description: "contracted color".to_string(),
                    },
                )],
                reduction: crate::Reduction {
                    kind: crate::ReductionKind::ContractedColor,
                    groups: Vec::new(),
                },
                model_parameters: Vec::new(),
                selectors: crate::SelectorCapabilities {
                    helicity: true,
                    color_flow: false,
                    contracted_color: false,
                },
                extensions: BTreeMap::new(),
            },
            helicity_index_by_id: BTreeMap::from([("h0".to_string(), 0)]),
            helicity_members_by_representative: vec![vec![0]],
            color_index_by_id: BTreeMap::from([("contracted".to_string(), 0)]),
            reduction_by_group_id: BTreeMap::new(),
            numeric_reduction_by_group_id: BTreeMap::new(),
        };
        let point_count = 3;
        let point_stride = point_count;
        let mut values_re = vec![0.0; 20 * point_stride];
        let mut values_im = vec![0.0; 20 * point_stride];
        let mut local_values = vec![(0.0, 0.0); 13 * point_count];
        for group in 0..13 {
            for point in 0..point_count {
                let value = (
                    (group * 7 + point * 3) as f64 / 11.0 - 1.0,
                    (group * 5 + point) as f64 / 13.0 - 0.4,
                );
                local_values[group * point_count + point] = value;
                values_re[(group + 2) * point_stride + point] = value.0;
                values_im[(group + 2) * point_stride + point] = value.1;
            }
        }
        let tile = ContractedReplayTile {
            values_re: &values_re,
            values_im: &values_im,
            point_count,
            point_stride,
            destination_count: 20,
        };
        let RuntimeColorContractionReducer::SymmetricGroupFourier(reducer) =
            contraction.runtime_reducer().unwrap()
        else {
            panic!("expected symmetric-group runtime reducer")
        };
        let covered_group_count = reducer.channel_count() * reducer.group_order();
        assert!(reducer.channel_count() > 0);
        assert!(reducer.local_group_count() > covered_group_count);
        assert!(reducer.residual_entries().iter().any(|entry| {
            entry.left_group_index < covered_group_count as u32
                && entry.right_group_index >= covered_group_count as u32
        }));
        let expected = RecurrenceColorContraction::symmetric_group_s3_dense_for_runtime_test(
            &local_values,
            point_count,
        );

        let mut workspace = reducer.workspace(1).unwrap();
        let mut actual = vec![0.0; point_count];
        contract_color_tile(
            &tile,
            &contraction,
            &destination_physics_helicity,
            &physics,
            None,
            1.0,
            &mut [],
            &mut [],
            Some(&mut workspace),
            |point, helicity, value| {
                assert_eq!(helicity, 0);
                actual[point] += value;
            },
        )
        .unwrap();
        assert_eq!(workspace.lane_capacity(), point_count);
        for (actual, expected) in actual.into_iter().zip(expected) {
            let scale = expected.abs().max(1.0);
            assert!((actual - expected).abs() <= 2.0e-11 * scale);
        }

        physics.manifest.helicities[0].computed = false;
        physics.manifest.helicities[0].structural_zero = true;
        let selected = BTreeSet::from(["h0".to_string()]);
        assert!(
            !contracted_selection_has_computed_destination(
                &destination_physics_helicity,
                &physics,
                &selected,
            )
            .unwrap()
        );
    }
}

#[derive(Clone, Copy)]
struct DirectProfileSnapshot {
    phases: DirectRuntimePhaseTimings,
    roles: DirectExecutionRoleTimings,
    execution: DirectExecutionCounters,
    traffic: DirectArenaTrafficCounters,
    activity: DirectRuntimeActivityCounters,
    internal_scratch_bytes: u64,
    internal_broadcast_bytes: u64,
}

impl DirectProfileSnapshot {
    fn capture(
        runtime: &DirectRecurrenceExecutionRuntime,
        owners: &NativeRecurrenceDirectExecutorOwners,
    ) -> Self {
        let (internal_scratch_bytes, internal_broadcast_bytes) = owners.internal_traffic_bytes();
        Self {
            phases: runtime.phase_timings(),
            roles: runtime.role_timings(),
            execution: runtime.counters(),
            traffic: runtime.traffic_counters(),
            activity: runtime.activity_counters(),
            internal_scratch_bytes,
            internal_broadcast_bytes,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct DirectProfileDelta {
    pub(super) momentum_fill: Duration,
    pub(super) union_source_fill: Duration,
    pub(super) direct_execution: Duration,
    pub(super) replay_output_mapping: Duration,
    pub(super) source_kernel: Duration,
    pub(super) contribution_kernel: Duration,
    pub(super) finalization: Duration,
    pub(super) closure: Duration,
    pub(super) execution: DirectExecutionCounters,
    pub(super) traffic: DirectArenaTrafficCounters,
    pub(super) activity: DirectRuntimeActivityCounters,
    pub(super) internal_scratch_bytes: u64,
    pub(super) internal_broadcast_bytes: u64,
}

fn add_persisted_execution_report(
    target: &mut PersistedHelicityFamilyExecutionReportV1,
    source: PersistedHelicityFamilyExecutionReportV1,
) {
    target.cache_hit &= source.cache_hit;
    target.source_calls = target.source_calls.saturating_add(source.source_calls);
    target.source_rows = target.source_rows.saturating_add(source.source_rows);
    target.contribution_calls = target
        .contribution_calls
        .saturating_add(source.contribution_calls);
    target.contribution_rows = target
        .contribution_rows
        .saturating_add(source.contribution_rows);
    target.finalization_calls = target
        .finalization_calls
        .saturating_add(source.finalization_calls);
    target.finalization_rows = target
        .finalization_rows
        .saturating_add(source.finalization_rows);
    target.closure_calls = target.closure_calls.saturating_add(source.closure_calls);
    target.closure_rows = target.closure_rows.saturating_add(source.closure_rows);
}

fn persisted_execution_report_aggregate() -> PersistedHelicityFamilyExecutionReportV1 {
    PersistedHelicityFamilyExecutionReportV1 {
        // Conjunction uses true as its identity, so an all-hit tile sequence
        // remains a hit while any cold tile turns the aggregate into a miss.
        cache_hit: true,
        ..PersistedHelicityFamilyExecutionReportV1::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn persisted_companion_profile(
    total: Duration,
    parameter_setup: Duration,
    input_setup: Duration,
    execution: Duration,
    reduction: Duration,
    momentum_scalar_values: u64,
    report: PersistedHelicityFamilyExecutionReportV1,
) -> RuntimeProfile {
    let calls = u64::from(report.source_calls)
        + u64::from(report.contribution_calls)
        + u64::from(report.finalization_calls)
        + u64::from(report.closure_calls);
    let rows = u64::from(report.source_rows)
        + u64::from(report.contribution_rows)
        + u64::from(report.finalization_rows)
        + u64::from(report.closure_rows);
    let schedule_executions = u64::from(report.source_calls);
    direct_profile_from_delta(
        total,
        parameter_setup,
        input_setup,
        reduction,
        DirectProfileDelta {
            direct_execution: execution,
            execution: DirectExecutionCounters {
                source_calls: u64::from(report.source_calls),
                source_rows: u64::from(report.source_rows),
                contribution_calls: u64::from(report.contribution_calls),
                contribution_rows: u64::from(report.contribution_rows),
                finalization_calls: u64::from(report.finalization_calls),
                finalization_rows: u64::from(report.finalization_rows),
                closure_calls: u64::from(report.closure_calls),
                closure_rows: u64::from(report.closure_rows),
                ..DirectExecutionCounters::default()
            },
            traffic: DirectArenaTrafficCounters {
                calls,
                rows,
                ..DirectArenaTrafficCounters::default()
            },
            activity: DirectRuntimeActivityCounters {
                momentum_fill_calls: schedule_executions,
                momentum_scalar_values_filled: momentum_scalar_values,
                schedule_executions,
                union_source_dispatch_calls: u64::from(report.source_calls),
                union_source_rows: u64::from(report.source_rows),
                union_schedule_executions: schedule_executions,
                ..DirectRuntimeActivityCounters::default()
            },
            ..DirectProfileDelta::default()
        },
    )
}

fn direct_profile(
    total: Duration,
    parameter_setup: Duration,
    external_momentum_flatten: Duration,
    reduction: Duration,
    before: DirectProfileSnapshot,
    after: DirectProfileSnapshot,
) -> RuntimeProfile {
    let momentum_fill = after
        .phases
        .momentum_fill
        .saturating_sub(before.phases.momentum_fill);
    let union_source_fill = after
        .phases
        .union_source_fill
        .saturating_sub(before.phases.union_source_fill);
    let direct_execution = after
        .phases
        .direct_execution
        .saturating_sub(before.phases.direct_execution);
    let replay_output_mapping = after
        .phases
        .replay_output_mapping
        .saturating_sub(before.phases.replay_output_mapping);
    let source_kernel = after.roles.source.saturating_sub(before.roles.source);
    let contribution_kernel = after
        .roles
        .contribution
        .saturating_sub(before.roles.contribution);
    let finalization = after
        .roles
        .finalization
        .saturating_sub(before.roles.finalization);
    let closure = after.roles.closure.saturating_sub(before.roles.closure);
    let execution = DirectExecutionCounters {
        source_calls: after
            .execution
            .source_calls
            .saturating_sub(before.execution.source_calls),
        source_rows: after
            .execution
            .source_rows
            .saturating_sub(before.execution.source_rows),
        contribution_calls: after
            .execution
            .contribution_calls
            .saturating_sub(before.execution.contribution_calls),
        contribution_rows: after
            .execution
            .contribution_rows
            .saturating_sub(before.execution.contribution_rows),
        finalization_calls: after
            .execution
            .finalization_calls
            .saturating_sub(before.execution.finalization_calls),
        finalization_rows: after
            .execution
            .finalization_rows
            .saturating_sub(before.execution.finalization_rows),
        closure_calls: after
            .execution
            .closure_calls
            .saturating_sub(before.execution.closure_calls),
        closure_rows: after
            .execution
            .closure_rows
            .saturating_sub(before.execution.closure_rows),
        packed_input_bytes: after
            .execution
            .packed_input_bytes
            .saturating_sub(before.execution.packed_input_bytes),
        packed_output_bytes: after
            .execution
            .packed_output_bytes
            .saturating_sub(before.execution.packed_output_bytes),
        scatter_bytes: after
            .execution
            .scatter_bytes
            .saturating_sub(before.execution.scatter_bytes),
    };
    let traffic = DirectArenaTrafficCounters {
        calls: after.traffic.calls.saturating_sub(before.traffic.calls),
        rows: after.traffic.rows.saturating_sub(before.traffic.rows),
        points: after.traffic.points.saturating_sub(before.traffic.points),
        packet_input_bytes: after
            .traffic
            .packet_input_bytes
            .saturating_sub(before.traffic.packet_input_bytes),
        packet_output_bytes: after
            .traffic
            .packet_output_bytes
            .saturating_sub(before.traffic.packet_output_bytes),
        gather_bytes: after
            .traffic
            .gather_bytes
            .saturating_sub(before.traffic.gather_bytes),
        scatter_bytes: after
            .traffic
            .scatter_bytes
            .saturating_sub(before.traffic.scatter_bytes),
        remap_bytes: after
            .traffic
            .remap_bytes
            .saturating_sub(before.traffic.remap_bytes),
    };
    let internal_scratch_bytes = after
        .internal_scratch_bytes
        .saturating_sub(before.internal_scratch_bytes);
    let internal_broadcast_bytes = after
        .internal_broadcast_bytes
        .saturating_sub(before.internal_broadcast_bytes);
    let activity = DirectRuntimeActivityCounters {
        momentum_fill_calls: after
            .activity
            .momentum_fill_calls
            .saturating_sub(before.activity.momentum_fill_calls),
        momentum_forms_filled: after
            .activity
            .momentum_forms_filled
            .saturating_sub(before.activity.momentum_forms_filled),
        momentum_terms_filled: after
            .activity
            .momentum_terms_filled
            .saturating_sub(before.activity.momentum_terms_filled),
        momentum_scalar_values_filled: after
            .activity
            .momentum_scalar_values_filled
            .saturating_sub(before.activity.momentum_scalar_values_filled),
        schedule_executions: after
            .activity
            .schedule_executions
            .saturating_sub(before.activity.schedule_executions),
        replay_schedule_executions: after
            .activity
            .replay_schedule_executions
            .saturating_sub(before.activity.replay_schedule_executions),
        replay_output_values_scaled: after
            .activity
            .replay_output_values_scaled
            .saturating_sub(before.activity.replay_output_values_scaled),
        union_source_dispatch_calls: after
            .activity
            .union_source_dispatch_calls
            .saturating_sub(before.activity.union_source_dispatch_calls),
        union_source_rows: after
            .activity
            .union_source_rows
            .saturating_sub(before.activity.union_source_rows),
        union_schedule_executions: after
            .activity
            .union_schedule_executions
            .saturating_sub(before.activity.union_schedule_executions),
    };
    direct_profile_from_delta(
        total,
        parameter_setup,
        external_momentum_flatten,
        reduction,
        DirectProfileDelta {
            momentum_fill,
            union_source_fill,
            direct_execution,
            replay_output_mapping,
            source_kernel,
            contribution_kernel,
            finalization,
            closure,
            execution,
            traffic,
            activity,
            internal_scratch_bytes,
            internal_broadcast_bytes,
        },
    )
}

pub(super) fn direct_profile_from_delta(
    total: Duration,
    parameter_setup: Duration,
    external_momentum_flatten: Duration,
    reduction: Duration,
    delta: DirectProfileDelta,
) -> RuntimeProfile {
    RuntimeProfile {
        momentum_setup_s: profile_duration_seconds(external_momentum_flatten + delta.momentum_fill),
        momentum_input_setup_s: profile_duration_seconds(external_momentum_flatten),
        model_parameter_setup_s: profile_duration_seconds(parameter_setup),
        stage_evaluator_call_s: profile_duration_seconds(delta.direct_execution),
        stage_evaluator_s: profile_duration_seconds(delta.direct_execution),
        recurrence_momentum_fill_s: profile_duration_seconds(delta.momentum_fill),
        recurrence_union_source_fill_s: profile_duration_seconds(delta.union_source_fill),
        recurrence_schedule_s: profile_duration_seconds(delta.direct_execution),
        recurrence_source_kernel_s: profile_duration_seconds(delta.source_kernel),
        recurrence_contribution_kernel_s: profile_duration_seconds(delta.contribution_kernel),
        recurrence_finalization_s: profile_duration_seconds(delta.finalization),
        recurrence_closure_s: profile_duration_seconds(delta.closure),
        recurrence_replay_output_mapping_s: profile_duration_seconds(delta.replay_output_mapping),
        recurrence_momentum_scalar_value_count: delta.activity.momentum_scalar_values_filled,
        recurrence_schedule_execution_count: delta.activity.schedule_executions,
        recurrence_replay_schedule_execution_count: delta.activity.replay_schedule_executions,
        recurrence_union_schedule_execution_count: delta.activity.union_schedule_executions,
        recurrence_union_source_row_count: delta.activity.union_source_rows,
        recurrence_replay_output_value_count: delta.activity.replay_output_values_scaled,
        recurrence_source_call_count: delta.execution.source_calls,
        recurrence_source_row_count: delta.execution.source_rows,
        recurrence_contribution_call_count: delta.execution.contribution_calls,
        recurrence_contribution_row_count: delta.execution.contribution_rows,
        recurrence_finalization_call_count: delta.execution.finalization_calls,
        recurrence_finalization_row_count: delta.execution.finalization_rows,
        recurrence_closure_call_count: delta.execution.closure_calls,
        recurrence_closure_row_count: delta.execution.closure_rows,
        recurrence_direct_packed_input_bytes: delta.execution.packed_input_bytes,
        recurrence_direct_packed_output_bytes: delta.execution.packed_output_bytes,
        // Fanout broadcasts stay within the persistent Direct-Arena current
        // workspace. They are not the retired evaluator-boundary scatter.
        recurrence_direct_scatter_bytes: 0,
        recurrence_direct_packet_input_bytes: delta.traffic.packet_input_bytes,
        recurrence_direct_packet_output_bytes: delta.traffic.packet_output_bytes,
        recurrence_direct_gather_bytes: delta.traffic.gather_bytes,
        recurrence_direct_traffic_scatter_bytes: delta.traffic.scatter_bytes,
        recurrence_direct_remap_bytes: delta.traffic.remap_bytes,
        recurrence_internal_scratch_bytes: delta.internal_scratch_bytes,
        recurrence_internal_broadcast_bytes: delta
            .internal_broadcast_bytes
            .saturating_add(delta.execution.scatter_bytes),
        reduction_s: profile_duration_seconds(reduction),
        total_s: profile_duration_seconds(total),
        ..RuntimeProfile::default()
    }
}

#[cfg(test)]
mod direct_profile_tests {
    use super::*;

    #[test]
    fn persisted_cache_hit_aggregation_is_true_only_when_every_tile_hits() {
        let mut all_hits = persisted_execution_report_aggregate();
        add_persisted_execution_report(
            &mut all_hits,
            PersistedHelicityFamilyExecutionReportV1 {
                cache_hit: true,
                ..PersistedHelicityFamilyExecutionReportV1::default()
            },
        );
        add_persisted_execution_report(
            &mut all_hits,
            PersistedHelicityFamilyExecutionReportV1 {
                cache_hit: true,
                ..PersistedHelicityFamilyExecutionReportV1::default()
            },
        );
        assert!(all_hits.cache_hit);

        add_persisted_execution_report(
            &mut all_hits,
            PersistedHelicityFamilyExecutionReportV1::default(),
        );
        assert!(!all_hits.cache_hit);
    }

    #[test]
    fn fanout_scatter_is_reported_as_internal_broadcast_traffic() {
        let profile = direct_profile_from_delta(
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            Duration::ZERO,
            DirectProfileDelta {
                execution: DirectExecutionCounters {
                    scatter_bytes: 96,
                    ..DirectExecutionCounters::default()
                },
                internal_broadcast_bytes: 32,
                ..DirectProfileDelta::default()
            },
        );

        assert_eq!(profile.recurrence_direct_scatter_bytes, 0);
        assert_eq!(profile.recurrence_internal_broadcast_bytes, 128);
        profile
            .validate_recurrence_direct_boundary_traffic()
            .unwrap();
    }
}
