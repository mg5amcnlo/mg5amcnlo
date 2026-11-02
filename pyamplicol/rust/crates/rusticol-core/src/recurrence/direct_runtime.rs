// SPDX-License-Identifier: 0BSD

//! Owned workspace for direct-arena recurrence execution.
//!
//! Ordinary construction performs every allocation needed by a point tile.
//! Packed-singleton-capable runtimes allocate only their scalar point storage
//! initially and materialize the aligned tiled storage on first use. Callers
//! fill the persistent momentum and parameter arenas, then execute the
//! authenticated direct plan without resizing the active runtime storage.

use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(any(test, feature = "on-the-fly-test-support"))]
use super::direct_backend::observe_direct_amplitudes_before_replay;
use super::direct_backend::{
    DIRECT_STATUS_OK, DirectContributionFanoutProgram, DirectCurrentClearRange,
    DirectExecutionCounters, DirectExecutionRoleTimings, DirectExecutorCatalog, DirectFactorView,
    DirectMomentumView, DirectParameterView, DirectStageClearRanges,
    DirectUnionSourceDispatchHandle, DirectWorkspace, execute_direct_plan_destination_selected,
    execute_direct_plan_profiled_with_fanout_and_traffic,
    execute_direct_plan_selected_profiled_with_fanout_and_traffic,
    execute_direct_plan_selected_unprofiled_with_fanout,
    execute_direct_plan_unprofiled_with_fanout,
};
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
    DirectAmplitudeDestinationDescriptor, DirectExecutorRole, DirectRecurrencePlan,
    DirectResolvedHelicityDescriptor, DirectRowGroupDescriptor,
};
use super::{DIRECT_NONE_U32, DirectNodeKind, RecurrenceStrategy, SemanticDigest};
use crate::direct_arena::{
    AlignedF64Buffer, DIRECT_ARENA_ALIGNMENT, DirectArenaAllocationCounters,
    DirectArenaTrafficCounters, DirectArenaWorkspace, checked_plane_scalar_len,
    deterministic_point_tile_size, validate_direct_views,
};
use crate::{RusticolError, RusticolResult};

/// Alignment used for the SIMD-facing numeric arenas.
pub const DIRECT_RUNTIME_ARENA_ALIGNMENT: usize = DIRECT_ARENA_ALIGNMENT;
const DIRECT_RUNTIME_CACHE_TARGET_BYTES: usize = 4 * 1024 * 1024;

/// Borrowed component-major amplitude output for the most recent tile.
#[derive(Clone, Copy, Debug)]
pub struct DirectRecurrenceTileOutput<'a> {
    amplitude_re: &'a [f64],
    amplitude_im: &'a [f64],
    amplitude_destinations: &'a [DirectAmplitudeDestinationDescriptor],
    point_count: u32,
    point_stride: u32,
    destination_count: u32,
    public_flow_id: Option<u32>,
    representative_flow_id: Option<u32>,
    selected_destination_ids: Option<&'a [u32]>,
}

/// One already-selected amplitude and its active point planes. The public
/// random-access methods retain their independent selector checks; internal
/// reductions can consume this borrowed view without repeating those checks.
pub(crate) struct DirectRecurrenceAmplitudeView<'a> {
    pub(crate) destination: &'a DirectAmplitudeDestinationDescriptor,
    pub(crate) real: &'a [f64],
    pub(crate) imaginary: &'a [f64],
}

impl<'a> DirectRecurrenceTileOutput<'a> {
    pub const fn point_count(&self) -> u32 {
        self.point_count
    }

    pub const fn point_stride(&self) -> u32 {
        self.point_stride
    }

    pub const fn destination_count(&self) -> u32 {
        self.destination_count
    }

    pub const fn public_flow_id(&self) -> Option<u32> {
        self.public_flow_id
    }

    pub const fn representative_flow_id(&self) -> Option<u32> {
        self.representative_flow_id
    }

    /// Full persistent real arena, including inactive tile tails.
    pub const fn storage_re(&self) -> &'a [f64] {
        self.amplitude_re
    }

    /// Full persistent imaginary arena, including inactive tile tails.
    pub const fn storage_im(&self) -> &'a [f64] {
        self.amplitude_im
    }

    pub fn destination_re(&self, destination_id: u32) -> Option<&[f64]> {
        self.destination(self.amplitude_re, destination_id)
    }

    pub fn destination_im(&self, destination_id: u32) -> Option<&[f64]> {
        self.destination(self.amplitude_im, destination_id)
    }

    pub fn selected_destination_ids(&self) -> impl Iterator<Item = u32> + '_ {
        let unrestricted = if self.selected_destination_ids.is_some() {
            &[][..]
        } else {
            self.amplitude_destinations
        };
        self.selected_destination_ids
            .unwrap_or_default()
            .iter()
            .copied()
            .chain(
                unrestricted
                    .iter()
                    .filter(|destination| {
                        self.representative_flow_id.is_none_or(|representative| {
                            destination.target_sector_id == representative
                        })
                    })
                    .map(|destination| destination.id),
            )
    }

    pub(crate) fn selected_amplitudes(
        &self,
    ) -> impl Iterator<Item = RusticolResult<DirectRecurrenceAmplitudeView<'_>>> + '_ {
        self.selected_destination_ids().map(|destination_id| {
            // Explicit destination selections were checked against the
            // representative when their selector was prepared. Unrestricted
            // iteration above already filters by that representative.
            let destination = self
                .amplitude_destinations
                .get(destination_id as usize)
                .ok_or_else(|| invalid("selected amplitude descriptor is absent"))?;
            let start = (destination_id as usize)
                .checked_mul(self.point_stride as usize)
                .ok_or_else(|| invalid("selected amplitude offset exceeds usize"))?;
            let end = start
                .checked_add(self.point_count as usize)
                .ok_or_else(|| invalid("selected amplitude range exceeds usize"))?;
            let real = self
                .amplitude_re
                .get(start..end)
                .ok_or_else(|| invalid("selected real amplitude plane is absent"))?;
            let imaginary = self
                .amplitude_im
                .get(start..end)
                .ok_or_else(|| invalid("selected imaginary amplitude plane is absent"))?;
            Ok(DirectRecurrenceAmplitudeView {
                destination,
                real,
                imaginary,
            })
        })
    }

    fn destination<'b>(&self, values: &'b [f64], destination_id: u32) -> Option<&'b [f64]> {
        if destination_id >= self.destination_count {
            return None;
        }
        if self
            .selected_destination_ids
            .is_some_and(|ids| ids.binary_search(&destination_id).is_err())
        {
            return None;
        }
        let descriptor = self.amplitude_destinations.get(destination_id as usize)?;
        if self
            .representative_flow_id
            .is_some_and(|representative| descriptor.target_sector_id != representative)
        {
            return None;
        }
        let start = destination_id as usize * self.point_stride as usize;
        values.get(start..start + self.point_count as usize)
    }
}

/// Prepared, plan-authenticated topology-replay selection.
#[derive(Clone)]
pub struct DirectReplaySelectorPlan {
    runtime_layout_digest: SemanticDigest,
    public_flow_id: u32,
    representative_flow_id: u32,
    source_permutation: Arc<[u32]>,
    source_momentum_signs: Arc<[i32]>,
    helicity_map: Arc<[u32]>,
    phase_re: f64,
    phase_im: f64,
    multiplicity: u32,
    row_groups: Arc<[DirectRowGroupDescriptor]>,
    amplitude_clear_ranges: Arc<[Range<usize>]>,
    destination_program: Option<Arc<DirectReplayDestinationProgram>>,
}

/// A cold dependency closure, reusable across flows with the same representative.
struct DirectReplayDestinationProgram {
    destination_ids: Arc<[u32]>,
    row_groups: Box<[DirectRowGroupDescriptor]>,
    amplitude_clear_ranges: Box<[Range<usize>]>,
    current_clear_stages: Box<[DirectStageClearRanges]>,
}

/// Prepared, plan-authenticated all-flow-union helicity selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectUnionHelicitySelectorPlan {
    runtime_layout_digest: SemanticDigest,
    resolved_helicity_id: u32,
    source_selection_start: usize,
    source_selection_count: usize,
}

impl DirectUnionHelicitySelectorPlan {
    pub const fn resolved_helicity_id(&self) -> u32 {
        self.resolved_helicity_id
    }
}

impl DirectReplaySelectorPlan {
    pub const fn public_flow_id(&self) -> u32 {
        self.public_flow_id
    }

    pub const fn representative_flow_id(&self) -> u32 {
        self.representative_flow_id
    }

    pub const fn phase(&self) -> (f64, f64) {
        (self.phase_re, self.phase_im)
    }

    pub const fn multiplicity(&self) -> u32 {
        self.multiplicity
    }

    pub fn mapped_external_source_slot(&self, representative_source_slot: u32) -> Option<u32> {
        self.source_permutation
            .get(representative_source_slot as usize)
            .copied()
    }

    pub(crate) fn helicity_map(&self) -> &[u32] {
        &self.helicity_map
    }

    pub fn selected_row_group_count(&self) -> usize {
        self.selected_row_groups().len()
    }

    pub fn selected_row_count(&self) -> u64 {
        self.selected_row_groups()
            .iter()
            .map(|group| u64::from(group.row_count))
            .sum()
    }

    fn selected_row_groups(&self) -> &[DirectRowGroupDescriptor] {
        self.destination_program
            .as_ref()
            .map_or(self.row_groups.as_ref(), |program| {
                program.row_groups.as_ref()
            })
    }

    fn identity(&self) -> DirectReplaySelectorIdentity {
        DirectReplaySelectorIdentity {
            runtime_layout_digest: self.runtime_layout_digest,
            public_flow_id: self.public_flow_id,
            representative_flow_id: self.representative_flow_id,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectRuntimeActivityCounters {
    pub momentum_fill_calls: u64,
    pub momentum_forms_filled: u64,
    pub momentum_terms_filled: u64,
    pub momentum_scalar_values_filled: u64,
    pub schedule_executions: u64,
    pub replay_schedule_executions: u64,
    pub replay_output_values_scaled: u64,
    pub union_source_dispatch_calls: u64,
    pub union_source_rows: u64,
    pub union_schedule_executions: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectRuntimePhaseTimings {
    pub momentum_fill: Duration,
    pub union_source_fill: Duration,
    pub direct_execution: Duration,
    pub replay_output_mapping: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectReplaySelectorIdentity {
    runtime_layout_digest: SemanticDigest,
    public_flow_id: u32,
    representative_flow_id: u32,
}

struct DirectPointStorage {
    arena: DirectArenaWorkspace,
    momenta: AlignedF64Buffer,
}

#[allow(clippy::too_many_arguments)]
fn allocate_point_storage(
    current_component_count: u32,
    amplitude_destination_count: u32,
    momentum_plane_count: u32,
    momentum_form_count: u32,
    lorentz_component_count: u16,
    tile_capacity: u32,
    packed_singleton: bool,
    parameters: DirectParameterView,
    factors: DirectFactorView,
) -> RusticolResult<DirectPointStorage> {
    let mut arena = if packed_singleton {
        DirectArenaWorkspace::new_packed_singleton(
            current_component_count,
            amplitude_destination_count,
        )?
    } else {
        DirectArenaWorkspace::new(
            current_component_count,
            amplitude_destination_count,
            tile_capacity,
        )?
    };
    let point_stride = arena.point_stride();
    arena.begin_tile(if packed_singleton { 1 } else { tile_capacity })?;
    let momentum_label = if packed_singleton {
        "packed singleton momentum arena"
    } else {
        "momentum arena"
    };
    let momenta = AlignedF64Buffer::zeroed(
        checked_plane_scalar_len(momentum_plane_count, point_stride, momentum_label)?,
        if packed_singleton {
            "packed singleton momentum"
        } else {
            "momentum"
        },
    )?;
    validate_direct_views(
        arena.view()?,
        DirectMomentumView {
            values: momenta.as_ptr(),
            scalar_len: u64::try_from(momenta.len())
                .map_err(|_| invalid("momentum arena exceeds u64"))?,
            form_count: momentum_form_count,
            lorentz_component_count,
            point_stride,
        },
        parameters,
        factors,
    )?;
    Ok(DirectPointStorage { arena, momenta })
}

/// One owned, thread-local direct recurrence execution workspace.
pub struct DirectRecurrenceExecutionRuntime {
    plan: DirectRecurrencePlan,
    executors: DirectExecutorCatalog,
    arena: DirectArenaWorkspace,
    momenta: AlignedF64Buffer,
    parameters_re: AlignedF64Buffer,
    parameters_im: AlignedF64Buffer,
    factors_re: AlignedF64Buffer,
    factors_im: AlignedF64Buffer,
    contribution_fanout: DirectContributionFanoutProgram,
    union_source_dispatch: Option<DirectUnionSourceDispatchHandle>,
    additive_amplitude_ranges: Vec<Range<usize>>,
    momentum_form_count: u32,
    lorentz_component_count: u16,
    runtime_current_component_count: u32,
    momentum_plane_count: u32,
    tile_capacity: u32,
    point_stride: u32,
    alternate_point_storage: Option<DirectPointStorage>,
    packed_singleton_active: bool,
    momenta_are_compact: bool,
    momentum_filled_points: u32,
    momentum_replay_selector: Option<DirectReplaySelectorIdentity>,
    last_point_count: u32,
    last_public_flow_id: Option<u32>,
    last_representative_flow_id: Option<u32>,
    last_selected_destination_ids: Option<Arc<[u32]>>,
    counters: DirectExecutionCounters,
    allocation_counters: DirectArenaAllocationCounters,
    traffic_counters: DirectArenaTrafficCounters,
    role_timings: DirectExecutionRoleTimings,
    activity_counters: DirectRuntimeActivityCounters,
    timings: DirectRuntimePhaseTimings,
}

impl DirectRecurrenceExecutionRuntime {
    /// Allocate the initial workspace and convert the plan's exact factors to
    /// their persistent binary64 split-complex catalog. Packed-capable
    /// runtimes defer aligned tiled point storage until it is needed.
    pub fn new(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        lorentz_component_count: u16,
    ) -> RusticolResult<Self> {
        Self::new_inner(plan, executors, lorentz_component_count, None)
    }

    /// Construct an all-flow-union workspace with its authenticated SourceIR
    /// dispatcher. The dispatcher context is owned by the same loaded backend
    /// owner that backs `executors` and must outlive this runtime.
    pub fn new_with_union_source_dispatch(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        lorentz_component_count: u16,
        union_source_dispatch: DirectUnionSourceDispatchHandle,
    ) -> RusticolResult<Self> {
        if plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "union source dispatch requires an all-flow-union recurrence plan",
            ));
        }
        Self::new_inner(
            plan,
            executors,
            lorentz_component_count,
            Some(union_source_dispatch),
        )
    }

    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    fn new_inner(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        lorentz_component_count: u16,
        union_source_dispatch: Option<DirectUnionSourceDispatchHandle>,
    ) -> RusticolResult<Self> {
        if lorentz_component_count == 0 {
            return Err(invalid("Lorentz component count must be positive"));
        }

        let contribution_fanout = DirectContributionFanoutProgram::build(&plan, &executors)?;
        let runtime_current_component_count = plan
            .current_arena_components()
            .checked_add(contribution_fanout.scratch_component_count())
            .ok_or_else(|| invalid("direct fanout current arena exceeds u32"))?;

        let momentum_form_count = u32::try_from(plan.momentum_forms().len())
            .map_err(|_| invalid("momentum form count exceeds u32"))?;
        let split_complex_scalar_count = usize::try_from(runtime_current_component_count)
            .ok()
            .and_then(|current| current.checked_mul(2))
            .and_then(|current| {
                usize::try_from(plan.amplitude_destination_count())
                    .ok()
                    .and_then(|amplitude| amplitude.checked_mul(2))
                    .and_then(|amplitude| current.checked_add(amplitude))
            })
            .ok_or_else(|| invalid("split-complex per-point workspace size overflows usize"))?;
        let cache_split_complex_scalar_count = replay_cache_split_complex_scalar_count(
            &plan,
            split_complex_scalar_count,
            contribution_fanout.scratch_component_count(),
        )?;
        let momentum_scalar_count = usize::try_from(momentum_form_count)
            .ok()
            .and_then(|forms| forms.checked_mul(usize::from(lorentz_component_count)))
            .ok_or_else(|| invalid("per-point momentum workspace size overflows usize"))?;
        let per_point_scalar_count = split_complex_scalar_count
            .checked_add(momentum_scalar_count)
            .ok_or_else(|| invalid("per-point workspace size overflows usize"))?;
        let workspace_bytes = usize::try_from(plan.workspace_mib())
            .ok()
            .and_then(|mib| mib.checked_mul(1024 * 1024))
            .ok_or_else(|| invalid("workspace byte target overflows usize"))?;
        let workspace_tile = deterministic_point_tile_size(
            plan.point_tile_size(),
            workspace_bytes,
            usize::MAX,
            per_point_scalar_count,
        )?;
        let split_complex_per_point_bytes = cache_split_complex_scalar_count
            .checked_mul(size_of::<f64>())
            .filter(|bytes| *bytes != 0)
            .ok_or_else(|| invalid("split-complex per-point bytes overflow usize"))?;
        let cache_tile = greatest_power_of_two_not_exceeding(
            DIRECT_RUNTIME_CACHE_TARGET_BYTES / split_complex_per_point_bytes,
        );
        // The accepted recurrence cache policy is deliberately lane-local:
        // it counts current+amplitude split halves, while the workspace
        // target above counts those halves plus every momentum plane.
        // Fill at least one already allocated aligned pitch when the request
        // allows it, rather than repeatedly running a few lanes.
        let minimum_tiled_points = (DIRECT_RUNTIME_ARENA_ALIGNMENT / size_of::<f64>()) as u32;
        let packed_singleton_capable = executors.packed_singleton_capable();
        let tile_capacity = workspace_tile.min(
            u32::try_from(cache_tile)
                .unwrap_or(u32::MAX)
                .max(minimum_tiled_points),
        );
        let momentum_plane_count = usize::try_from(momentum_form_count)
            .ok()
            .and_then(|forms| forms.checked_mul(usize::from(lorentz_component_count)))
            .ok_or_else(|| invalid("momentum plane count overflows usize"))?;
        let momentum_plane_count_u32 = u32::try_from(momentum_plane_count)
            .map_err(|_| invalid("momentum plane count exceeds u32"))?;
        let parameter_len = usize::try_from(plan.parameter_value_count())
            .map_err(|_| invalid("parameter value count exceeds usize"))?;
        let factor_len = plan
            .exact_factors()
            .len()
            .checked_add(usize::from(contribution_fanout.needs_unit_factor()))
            .ok_or_else(|| invalid("direct fanout factor catalog exceeds usize"))?;

        let additive_amplitude_ranges = amplitude_clear_ranges(&plan)?;
        let parameters_re = AlignedF64Buffer::zeroed(parameter_len, "parameter real")?;
        let parameters_im = AlignedF64Buffer::zeroed(parameter_len, "parameter imaginary")?;
        let mut factors_re = AlignedF64Buffer::zeroed(factor_len, "factor real")?;
        let mut factors_im = AlignedF64Buffer::zeroed(factor_len, "factor imaginary")?;
        for ((factor_re, factor_im), factor) in factors_re
            .as_mut_slice()
            .iter_mut()
            .zip(factors_im.as_mut_slice())
            .zip(plan.exact_factors())
        {
            *factor_re = factor.real().numerator() as f64 / factor.real().denominator() as f64;
            *factor_im = factor.imag().numerator() as f64 / factor.imag().denominator() as f64;
        }
        if contribution_fanout.needs_unit_factor() {
            let unit = plan.exact_factors().len();
            factors_re.as_mut_slice()[unit] = 1.0;
            factors_im.as_mut_slice()[unit] = 0.0;
        }
        let parameter_view = DirectParameterView {
            values_re: parameters_re.as_ptr(),
            values_im: parameters_im.as_ptr(),
            value_count: u32::try_from(parameters_re.len())
                .map_err(|_| invalid("parameter catalog exceeds u32"))?,
        };
        let factor_view = DirectFactorView {
            values_re: factors_re.as_ptr(),
            values_im: factors_im.as_ptr(),
            value_count: u32::try_from(factors_re.len())
                .map_err(|_| invalid("factor catalog exceeds u32"))?,
        };
        let point_storage = allocate_point_storage(
            runtime_current_component_count,
            plan.amplitude_destination_count(),
            momentum_plane_count_u32,
            momentum_form_count,
            lorentz_component_count,
            tile_capacity,
            packed_singleton_capable,
            parameter_view,
            factor_view,
        )?;
        let DirectPointStorage { arena, momenta } = point_storage;
        let point_stride = arena.point_stride();
        let allocation_counters = [
            momenta.allocation_counters(),
            parameters_re.allocation_counters(),
            parameters_im.allocation_counters(),
            factors_re.allocation_counters(),
            factors_im.allocation_counters(),
        ]
        .into_iter()
        .try_fold(arena.allocation_counters(), |total, counters| {
            total.checked_add(counters)
        })?;

        Ok(Self {
            plan,
            executors,
            arena,
            momenta,
            parameters_re,
            parameters_im,
            factors_re,
            factors_im,
            contribution_fanout,
            union_source_dispatch,
            additive_amplitude_ranges,
            momentum_form_count,
            lorentz_component_count,
            runtime_current_component_count,
            momentum_plane_count: momentum_plane_count_u32,
            tile_capacity,
            point_stride,
            alternate_point_storage: None,
            packed_singleton_active: packed_singleton_capable,
            momenta_are_compact: false,
            momentum_filled_points: 0,
            momentum_replay_selector: None,
            last_point_count: 0,
            last_public_flow_id: None,
            last_representative_flow_id: None,
            last_selected_destination_ids: None,
            counters: DirectExecutionCounters::default(),
            allocation_counters,
            traffic_counters: DirectArenaTrafficCounters::default(),
            role_timings: DirectExecutionRoleTimings::default(),
            activity_counters: DirectRuntimeActivityCounters::default(),
            timings: DirectRuntimePhaseTimings::default(),
        })
    }

    pub const fn plan(&self) -> &DirectRecurrencePlan {
        &self.plan
    }

    pub const fn point_tile_size(&self) -> u32 {
        self.tile_capacity
    }

    /// Physical component-plane pitch passed through every direct ABI view.
    /// This is `1` while packed singleton storage is active and changes to the
    /// aligned tiled pitch after a successful layout transition.
    pub const fn point_stride(&self) -> u32 {
        self.point_stride
    }

    pub const fn momentum_form_count(&self) -> u32 {
        self.momentum_form_count
    }

    pub const fn lorentz_component_count(&self) -> u16 {
        self.lorentz_component_count
    }

    /// Resolve one physical flow once. The returned selector owns its fixed
    /// source permutation and can be reused for every tile without allocation.
    pub fn prepare_replay_selector(
        &self,
        public_flow_id: u32,
    ) -> RusticolResult<DirectReplaySelectorPlan> {
        if !self.plan.strategy().uses_topology_replay_targets() {
            return Err(invalid(
                "replay selectors require a replay-capable recurrence plan",
            ));
        }
        let mut matches = self
            .plan
            .replay_targets()
            .iter()
            .filter(|target| target.public_flow_id == public_flow_id);
        let target = matches
            .next()
            .ok_or_else(|| invalid(format!("public flow {public_flow_id} is not retained")))?;
        if matches.next().is_some() {
            return Err(RusticolError::integrity(format!(
                "direct recurrence public flow {public_flow_id} has multiple replay targets"
            )));
        }
        if !self
            .plan
            .amplitude_destinations()
            .iter()
            .any(|destination| destination.target_sector_id == target.representative_id)
        {
            return Err(RusticolError::integrity(format!(
                "direct recurrence replay representative {} has no amplitude destination",
                target.representative_id
            )));
        }

        let start = usize::try_from(target.source_permutation_start).map_err(|_| {
            RusticolError::integrity(
                "direct recurrence replay source permutation start exceeds usize",
            )
        })?;
        let count = usize::try_from(target.source_permutation_count).map_err(|_| {
            RusticolError::integrity(
                "direct recurrence replay source permutation count exceeds usize",
            )
        })?;
        let end = start.checked_add(count).ok_or_else(|| {
            RusticolError::integrity(
                "direct recurrence replay source permutation range overflows usize",
            )
        })?;
        let permutation = self
            .plan
            .source_permutations()
            .get(start..end)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence replay source permutation is out of bounds",
                )
            })?;
        let mut source_permutation = Vec::new();
        source_permutation
            .try_reserve_exact(permutation.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate direct recurrence replay selector: {error}"
                ))
            })?;
        source_permutation.extend_from_slice(permutation);
        let momentum_signs = self
            .plan
            .replay_momentum_signs()
            .get(start..end)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence replay momentum-sign map is out of bounds",
                )
            })?;
        let mut source_momentum_signs = Vec::new();
        source_momentum_signs
            .try_reserve_exact(momentum_signs.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate direct recurrence replay momentum signs: {error}"
                ))
            })?;
        source_momentum_signs.extend_from_slice(momentum_signs);
        let helicity_start = usize::try_from(target.helicity_map_start).map_err(|_| {
            RusticolError::integrity("direct recurrence replay helicity-map start exceeds usize")
        })?;
        let helicity_count = usize::try_from(target.helicity_map_count).map_err(|_| {
            RusticolError::integrity("direct recurrence replay helicity-map count exceeds usize")
        })?;
        let helicity_end = helicity_start.checked_add(helicity_count).ok_or_else(|| {
            RusticolError::integrity("direct recurrence replay helicity-map range overflows usize")
        })?;
        let mapped_helicities = self
            .plan
            .replay_helicity_map()
            .get(helicity_start..helicity_end)
            .ok_or_else(|| {
                RusticolError::integrity("direct recurrence replay helicity map is out of bounds")
            })?;
        let mut helicity_map = Vec::new();
        helicity_map
            .try_reserve_exact(mapped_helicities.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate direct recurrence replay helicity map: {error}"
                ))
            })?;
        helicity_map.extend_from_slice(mapped_helicities);

        let phase = *self
            .plan
            .exact_factors()
            .get(target.phase_exact_factor_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity("direct recurrence replay phase factor is out of bounds")
            })?;
        let row_groups = selected_replay_row_groups(&self.plan, target.representative_id)?;
        let amplitude_clear_ranges =
            selected_amplitude_clear_ranges(&self.plan, target.representative_id)?;
        Ok(DirectReplaySelectorPlan {
            runtime_layout_digest: self.plan.runtime_layout_digest(),
            public_flow_id,
            representative_flow_id: target.representative_id,
            source_permutation: source_permutation.into(),
            source_momentum_signs: source_momentum_signs.into(),
            helicity_map: helicity_map.into(),
            phase_re: phase.real().numerator() as f64 / phase.real().denominator() as f64,
            phase_im: phase.imag().numerator() as f64 / phase.imag().denominator() as f64,
            multiplicity: target.multiplicity,
            row_groups: row_groups.into(),
            amplitude_clear_ranges: amplitude_clear_ranges.into(),
            destination_program: None,
        })
    }

    /// Prepare just the dependency closure of the requested representative
    /// amplitude destinations. Sources remain helicity-generic in the plan;
    /// numerical current identities, including identities between helicities,
    /// are followed through their real parent dependencies here.
    pub fn prepare_replay_selector_for_destinations(
        &self,
        base: &DirectReplaySelectorPlan,
        needed_destinations: &[u32],
    ) -> RusticolResult<DirectReplaySelectorPlan> {
        self.validate_replay_selector(base)?;
        let mut selector = base.clone();
        selector.destination_program = Some(Arc::new(selected_replay_destination_program(
            &self.plan,
            selector.representative_flow_id,
            &selector.row_groups,
            needed_destinations,
        )?));
        Ok(selector)
    }

    /// Reuse an already prepared closure for another public flow represented
    /// by the same sector. Only the momentum/helicity replay mapping differs.
    pub fn prepare_replay_selector_with_destination_program(
        &self,
        base: &DirectReplaySelectorPlan,
        prepared: &DirectReplaySelectorPlan,
    ) -> RusticolResult<DirectReplaySelectorPlan> {
        self.validate_replay_selector(base)?;
        self.validate_replay_selector(prepared)?;
        let mut selector = base.clone();
        if selector.representative_flow_id != prepared.representative_flow_id {
            return Err(invalid(
                "replay destination program has a different representative",
            ));
        }
        selector.destination_program = Some(Arc::clone(
            prepared
                .destination_program
                .as_ref()
                .ok_or_else(|| invalid("replay selector has no restricted destination program"))?,
        ));
        Ok(selector)
    }

    /// Resolve one retained helicity once for all-flow-union execution.
    ///
    /// The returned selector is a pair of authenticated ranges into the
    /// immutable plan. Reusing it for later tiles performs no allocation.
    pub fn prepare_union_helicity_selector(
        &self,
        resolved_helicity_id: u32,
    ) -> RusticolResult<DirectUnionHelicitySelectorPlan> {
        if self.plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "union helicity selectors require an all-flow-union recurrence plan",
            ));
        }
        if self.union_source_dispatch.is_none() {
            return Err(invalid(
                "all-flow-union recurrence runtime has no source dispatcher",
            ));
        }
        let descriptor = self
            .plan
            .resolved_helicities()
            .get(resolved_helicity_id as usize)
            .filter(|descriptor| descriptor.id == resolved_helicity_id)
            .ok_or_else(|| {
                invalid(format!(
                    "resolved helicity {resolved_helicity_id} is not retained"
                ))
            })?;
        let (source_selection_start, source_selection_count) =
            self.validate_union_helicity_descriptor(descriptor)?;
        Ok(DirectUnionHelicitySelectorPlan {
            runtime_layout_digest: self.plan.runtime_layout_digest(),
            resolved_helicity_id,
            source_selection_start,
            source_selection_count,
        })
    }

    /// Mutable form-major, Lorentz-component-major momentum arena using the
    /// compatibility-facing semantic tile pitch returned by
    /// [`Self::point_tile_size`].
    ///
    /// The generic Direct-Arena ABI uses [`Self::point_stride`] internally.
    /// When alignment padding separates physical planes, this accessor
    /// compacts active plane storage in place and restores it before the next
    /// direct call. A packed runtime allocates its aligned tiled storage on
    /// first legacy access; no auxiliary packet buffer is involved.
    pub fn momenta_mut(&mut self) -> RusticolResult<&mut [f64]> {
        self.force_tiled_layout()?;
        self.compact_momenta_for_legacy_access();
        self.momentum_filled_points = self.tile_capacity;
        self.momentum_replay_selector = None;
        let compact_len = self.compact_momentum_len();
        Ok(&mut self.momenta.as_mut_slice()[..compact_len])
    }

    /// Mutable physical Direct-Arena momentum storage.
    ///
    /// Plane starts use [`Self::point_stride`], including alignment padding.
    /// A packed runtime allocates its aligned tiled storage on first access.
    pub fn physical_momenta_mut(&mut self) -> RusticolResult<&mut [f64]> {
        self.force_tiled_layout()?;
        self.restore_physical_momenta();
        self.momentum_filled_points = self.tile_capacity;
        self.momentum_replay_selector = None;
        Ok(self.momenta.as_mut_slice())
    }

    /// One persistent momentum plane using the compatibility-facing semantic
    /// tile pitch, including its inactive semantic tile tail. A packed runtime
    /// allocates its aligned tiled storage on first valid access.
    pub fn momentum_plane_mut(
        &mut self,
        momentum_form_id: u32,
        lorentz_component: u16,
    ) -> RusticolResult<Option<&mut [f64]>> {
        if momentum_form_id >= self.momentum_form_count
            || lorentz_component >= self.lorentz_component_count
        {
            return Ok(None);
        }
        self.force_tiled_layout()?;
        self.compact_momenta_for_legacy_access();
        self.momentum_filled_points = self.tile_capacity;
        self.momentum_replay_selector = None;
        let plane = momentum_form_id as usize * usize::from(self.lorentz_component_count)
            + usize::from(lorentz_component);
        let start = plane * self.tile_capacity as usize;
        Ok(self
            .momenta
            .as_mut_slice()
            .get_mut(start..start + self.tile_capacity as usize))
    }

    /// Fill every canonical momentum form from point-major external
    /// `[point][source_slot][lorentz_component]` four-momenta.
    pub fn fill_momenta_from_external(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<()> {
        self.fill_momenta_from_external_impl::<true, false>(
            selector,
            point_count,
            external_four_momenta,
        )
    }

    pub fn fill_momenta_from_external_unprofiled(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<()> {
        self.fill_momenta_from_external_impl::<false, false>(
            selector,
            point_count,
            external_four_momenta,
        )
    }

    fn fill_momenta_from_external_impl<const PROFILE: bool, const SOURCE_MAJOR: bool>(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<()> {
        self.validate_replay_selector(selector)?;
        self.validate_point_count(point_count)?;
        if self.lorentz_component_count != 4 {
            return Err(invalid(format!(
                "external four-momentum fill requires 4 Lorentz components, runtime has {}",
                self.lorentz_component_count
            )));
        }
        let source_count = usize::try_from(self.plan.external_source_count())
            .map_err(|_| invalid("external source count exceeds usize"))?;
        let expected_len = (point_count as usize)
            .checked_mul(source_count)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| invalid("point-major external momentum length overflows usize"))?;
        if external_four_momenta.len() != expected_len {
            return Err(invalid(format!(
                "point-major external momentum length is {}, expected {expected_len}",
                external_four_momenta.len()
            )));
        }

        self.activate_point_layout(point_count)?;
        self.restore_physical_momenta();
        let started = PROFILE.then(Instant::now);
        let point_stride = self.point_stride as usize;
        let active_points = point_count as usize;
        let previous_points = self.momentum_filled_points as usize;
        let form_count = self.plan.momentum_forms().len();
        let terms = self.plan.momentum_terms();
        let momenta = self.momenta.as_mut_slice();
        for (form_id, form) in self.plan.momentum_forms().iter().enumerate() {
            // The immutable plan has already validated term ranges, source
            // slots and replay maps. Consume those persisted instructions once
            // per term, not once per component and phase-space point.
            let term_start = form.term_start as usize;
            let form_terms = &terms[term_start..term_start + form.term_count as usize];
            let form_start = form_id * 4 * point_stride;
            fill_momentum_form_from_external::<SOURCE_MAJOR>(
                &mut momenta[form_start..form_start + 4 * point_stride],
                point_stride,
                active_points,
                external_four_momenta,
                source_count,
                form_terms.iter().map(|term| {
                    let slot = term.source_slot as usize;
                    (
                        selector.source_permutation[slot] as usize,
                        f64::from(term.coefficient)
                            * f64::from(selector.source_momentum_signs[slot]),
                    )
                }),
            );
        }
        if active_points < previous_points {
            for plane in 0..form_count * 4 {
                let start = plane * point_stride + active_points;
                let end = plane * point_stride + previous_points;
                momenta[start..end].fill(0.0);
            }
        }

        self.momentum_filled_points = point_count;
        self.momentum_replay_selector = Some(selector.identity());
        if PROFILE {
            self.activity_counters.momentum_fill_calls =
                self.activity_counters.momentum_fill_calls.saturating_add(1);
            self.activity_counters.momentum_forms_filled = self
                .activity_counters
                .momentum_forms_filled
                .saturating_add(form_count as u64);
            self.activity_counters.momentum_terms_filled = self
                .activity_counters
                .momentum_terms_filled
                .saturating_add(terms.len() as u64);
            let scalar_values = (form_count as u64)
                .saturating_mul(4)
                .saturating_mul(u64::from(point_count));
            self.activity_counters.momentum_scalar_values_filled = self
                .activity_counters
                .momentum_scalar_values_filled
                .saturating_add(scalar_values);
            self.timings.momentum_fill += started
                .expect("profiled momentum fill has a start time")
                .elapsed();
        }
        Ok(())
    }

    /// Fill canonical momentum forms without a replay permutation.
    fn fill_identity_momenta_from_external<const PROFILE: bool, const SOURCE_MAJOR: bool>(
        &mut self,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<()> {
        if !self.plan.strategy().materializes_all_color_sectors() {
            return Err(invalid(
                "identity momentum fill requires a complete-color recurrence plan",
            ));
        }
        self.validate_point_count(point_count)?;
        if self.lorentz_component_count != 4 {
            return Err(invalid(format!(
                "external four-momentum fill requires 4 Lorentz components, runtime has {}",
                self.lorentz_component_count
            )));
        }
        let source_count = usize::try_from(self.plan.external_source_count())
            .map_err(|_| invalid("external source count exceeds usize"))?;
        let expected_len = (point_count as usize)
            .checked_mul(source_count)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| invalid("point-major external momentum length overflows usize"))?;
        if external_four_momenta.len() != expected_len {
            return Err(invalid(format!(
                "point-major external momentum length is {}, expected {expected_len}",
                external_four_momenta.len()
            )));
        }

        self.activate_point_layout(point_count)?;
        self.restore_physical_momenta();
        let started = PROFILE.then(Instant::now);
        let point_stride = self.point_stride as usize;
        let active_points = point_count as usize;
        let previous_points = self.momentum_filled_points as usize;
        let form_count = self.plan.momentum_forms().len();
        let terms = self.plan.momentum_terms();
        let momenta = self.momenta.as_mut_slice();
        for (form_id, form) in self.plan.momentum_forms().iter().enumerate() {
            let term_start = form.term_start as usize;
            let form_terms = &terms[term_start..term_start + form.term_count as usize];
            let form_start = form_id * 4 * point_stride;
            fill_momentum_form_from_external::<SOURCE_MAJOR>(
                &mut momenta[form_start..form_start + 4 * point_stride],
                point_stride,
                active_points,
                external_four_momenta,
                source_count,
                form_terms
                    .iter()
                    .map(|term| (term.source_slot as usize, f64::from(term.coefficient))),
            );
        }
        if active_points < previous_points {
            for plane in 0..form_count * 4 {
                let start = plane * point_stride + active_points;
                let end = plane * point_stride + previous_points;
                momenta[start..end].fill(0.0);
            }
        }

        self.momentum_filled_points = point_count;
        self.momentum_replay_selector = None;
        if PROFILE {
            self.activity_counters.momentum_fill_calls =
                self.activity_counters.momentum_fill_calls.saturating_add(1);
            self.activity_counters.momentum_forms_filled = self
                .activity_counters
                .momentum_forms_filled
                .saturating_add(form_count as u64);
            self.activity_counters.momentum_terms_filled = self
                .activity_counters
                .momentum_terms_filled
                .saturating_add(terms.len() as u64);
            self.activity_counters.momentum_scalar_values_filled = self
                .activity_counters
                .momentum_scalar_values_filled
                .saturating_add(
                    (form_count as u64)
                        .saturating_mul(4)
                        .saturating_mul(u64::from(point_count)),
                );
            self.timings.momentum_fill += started
                .expect("profiled union momentum fill has a start time")
                .elapsed();
        }
        Ok(())
    }

    pub fn parameters_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        (
            self.parameters_re.as_mut_slice(),
            self.parameters_im.as_mut_slice(),
        )
    }

    pub fn factors_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        let public_count = self.plan.exact_factors().len();
        (
            &mut self.factors_re.as_mut_slice()[..public_count],
            &mut self.factors_im.as_mut_slice()[..public_count],
        )
    }

    pub fn set_parameters(&mut self, values_re: &[f64], values_im: &[f64]) -> RusticolResult<()> {
        validate_split_values("parameter", self.parameters_re.len(), values_re, values_im)?;
        self.parameters_re.as_mut_slice().copy_from_slice(values_re);
        self.parameters_im.as_mut_slice().copy_from_slice(values_im);
        Ok(())
    }

    pub fn set_factors(&mut self, values_re: &[f64], values_im: &[f64]) -> RusticolResult<()> {
        let public_count = self.plan.exact_factors().len();
        validate_split_values("factor", public_count, values_re, values_im)?;
        self.factors_re.as_mut_slice()[..public_count].copy_from_slice(values_re);
        self.factors_im.as_mut_slice()[..public_count].copy_from_slice(values_im);
        Ok(())
    }

    /// Current planes in the active packed or tiled physical layout.
    pub fn current_arenas(&self) -> (&[f64], &[f64]) {
        let (real, imaginary) = self.arena.current_slices();
        let public_len = self.plan.current_arena_components() as usize * self.point_stride as usize;
        (&real[..public_len], &imaginary[..public_len])
    }

    /// Amplitude planes in the active packed or tiled physical layout.
    pub fn amplitude_arenas(&self) -> (&[f64], &[f64]) {
        self.arena.amplitude_slices()
    }

    /// Cumulative direct calls and rows since construction or the last reset.
    pub const fn counters(&self) -> DirectExecutionCounters {
        self.counters
    }

    pub const fn activity_counters(&self) -> DirectRuntimeActivityCounters {
        self.activity_counters
    }

    pub const fn allocation_counters(&self) -> DirectArenaAllocationCounters {
        self.allocation_counters
    }

    pub const fn traffic_counters(&self) -> DirectArenaTrafficCounters {
        self.traffic_counters
    }

    pub const fn role_timings(&self) -> DirectExecutionRoleTimings {
        self.role_timings
    }

    pub const fn phase_timings(&self) -> DirectRuntimePhaseTimings {
        self.timings
    }

    pub fn reset_counters(&mut self) {
        self.counters = DirectExecutionCounters::default();
        self.traffic_counters = DirectArenaTrafficCounters::default();
        self.role_timings = DirectExecutionRoleTimings::default();
        self.activity_counters = DirectRuntimeActivityCounters::default();
        self.timings = DirectRuntimePhaseTimings::default();
    }

    pub fn outputs(&self) -> Option<DirectRecurrenceTileOutput<'_>> {
        (self.last_point_count != 0).then(|| {
            self.borrowed_output(
                self.last_point_count,
                self.last_public_flow_id,
                self.last_representative_flow_id,
            )
        })
    }

    /// Execute one tile using the persistent inputs already held by this
    /// workspace.
    pub fn execute_tile(
        &mut self,
        point_count: u32,
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        if self.plan.strategy() == RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "all-flow-union execution requires a prepared runtime helicity selector",
            ));
        }
        self.execute_direct_tile(point_count)?;
        self.last_point_count = point_count;
        self.last_public_flow_id = None;
        self.last_representative_flow_id = None;
        Ok(self.borrowed_output(point_count, None, None))
    }

    /// Execute a tile whose momentum forms were filled for `selector`, then
    /// map representative amplitudes in place to the public flow convention.
    pub fn execute_replay_tile(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.execute_replay_tile_impl::<true>(selector, point_count)
    }

    pub fn execute_replay_tile_unprofiled(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.execute_replay_tile_impl::<false>(selector, point_count)
    }

    fn execute_replay_tile_impl<const PROFILE: bool>(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.validate_replay_selector(selector)?;
        self.validate_point_count(point_count)?;
        self.activate_point_layout(point_count)?;
        if self.momentum_filled_points != point_count
            || self.momentum_replay_selector != Some(selector.identity())
        {
            return Err(invalid(
                "replay tile momentum forms were not filled for this selector and point count",
            ));
        }

        self.execute_selected_direct_tile_impl::<PROFILE>(selector, point_count)?;
        self.scale_replay_amplitudes::<PROFILE>(
            selector,
            selector
                .destination_program
                .as_ref()
                .map(|program| program.destination_ids.as_ref()),
            point_count,
        )?;
        self.last_point_count = point_count;
        self.last_public_flow_id = Some(selector.public_flow_id);
        self.last_representative_flow_id = Some(selector.representative_flow_id);
        self.last_selected_destination_ids = selector
            .destination_program
            .as_ref()
            .map(|program| Arc::clone(&program.destination_ids));
        Ok(self.borrowed_output(
            point_count,
            self.last_public_flow_id,
            self.last_representative_flow_id,
        ))
    }

    fn scale_replay_amplitudes<const PROFILE: bool>(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        selected_destination_ids: Option<&[u32]>,
        point_count: u32,
    ) -> RusticolResult<()> {
        let started = PROFILE.then(Instant::now);
        let scale_re = selector.phase_re * f64::from(selector.multiplicity);
        let scale_im = selector.phase_im * f64::from(selector.multiplicity);
        let point_stride = self.point_stride as usize;
        let active_points = point_count as usize;
        let mut scaled_values = 0_u64;
        let (_, _, amplitude_re, amplitude_im) = self.arena.split_slices_mut();
        let unrestricted = if selected_destination_ids.is_some() {
            0
        } else {
            self.plan.amplitude_destinations().len()
        };
        for destination_id in selected_destination_ids
            .unwrap_or_default()
            .iter()
            .map(|&id| id as usize)
            .chain(0..unrestricted)
        {
            let destination = self.plan.amplitude_destinations()[destination_id];
            if destination.target_sector_id != selector.representative_flow_id {
                continue;
            }
            let start = destination_id * point_stride;
            let end = start + active_points;
            let values_re = &mut amplitude_re[start..end];
            let values_im = &mut amplitude_im[start..end];
            for point in 0..active_points {
                let value_re = values_re[point];
                let value_im = values_im[point];
                values_re[point] = value_re * scale_re - value_im * scale_im;
                values_im[point] = value_re * scale_im + value_im * scale_re;
            }
            scaled_values = scaled_values.saturating_add(u64::from(point_count));
        }
        if PROFILE {
            self.activity_counters.replay_schedule_executions = self
                .activity_counters
                .replay_schedule_executions
                .saturating_add(1);
            self.activity_counters.replay_output_values_scaled = self
                .activity_counters
                .replay_output_values_scaled
                .saturating_add(scaled_values);
            self.timings.replay_output_mapping += started
                .expect("profiled replay mapping has a start time")
                .elapsed();
        }
        Ok(())
    }

    /// Fill and execute one selected replay tile without any intermediate
    /// momentum-form materialization by the caller.
    pub fn execute_replay_tile_from_external(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.fill_momenta_from_external(selector, point_count, external_four_momenta)?;
        self.execute_replay_tile(selector, point_count)
    }

    pub fn execute_replay_tile_from_external_unprofiled(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.fill_momenta_from_external_unprofiled(selector, point_count, external_four_momenta)?;
        self.execute_replay_tile_unprofiled(selector, point_count)
    }

    /// Internal lane input is `[source][component][point]`, written directly
    /// into its existing external-input workspace. The public point-major
    /// entrypoints remain unchanged; this adds no preparation or transpose.
    pub(crate) fn execute_replay_tile_from_external_planes<const PROFILE: bool>(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.fill_momenta_from_external_impl::<PROFILE, true>(
            selector,
            point_count,
            external_four_momenta,
        )?;
        self.execute_replay_tile_impl::<PROFILE>(selector, point_count)
    }

    /// Fill one runtime helicity's union sources and execute the compact
    /// all-flow schedule exactly once.
    pub fn execute_union_tile_from_external(
        &mut self,
        selector: &DirectUnionHelicitySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.execute_union_tile_from_external_impl::<true, false>(
            selector,
            point_count,
            external_four_momenta,
        )
    }

    pub fn execute_union_tile_from_external_unprofiled(
        &mut self,
        selector: &DirectUnionHelicitySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.execute_union_tile_from_external_impl::<false, false>(
            selector,
            point_count,
            external_four_momenta,
        )
    }

    pub(crate) fn execute_union_tile_from_external_planes<const PROFILE: bool>(
        &mut self,
        selector: &DirectUnionHelicitySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.execute_union_tile_from_external_impl::<PROFILE, true>(
            selector,
            point_count,
            external_four_momenta,
        )
    }

    fn execute_union_tile_from_external_impl<const PROFILE: bool, const SOURCE_MAJOR: bool>(
        &mut self,
        selector: &DirectUnionHelicitySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.validate_union_selector(selector)?;
        self.fill_identity_momenta_from_external::<PROFILE, SOURCE_MAJOR>(
            point_count,
            external_four_momenta,
        )?;
        self.execute_union_sources::<PROFILE>(selector, point_count)?;
        self.execute_direct_tile_impl::<PROFILE>(point_count)?;
        if PROFILE {
            self.activity_counters.union_schedule_executions = self
                .activity_counters
                .union_schedule_executions
                .saturating_add(1);
        }
        self.last_point_count = point_count;
        self.last_public_flow_id = None;
        self.last_representative_flow_id = None;
        Ok(self.borrowed_output(point_count, None, None))
    }

    /// Execute a complete contracted-color schedule with fixed source templates.
    pub fn execute_contracted_tile_from_external(
        &mut self,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.execute_contracted_tile_from_external_impl::<true, false>(
            point_count,
            external_four_momenta,
        )
    }

    pub fn execute_contracted_tile_from_external_unprofiled(
        &mut self,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.execute_contracted_tile_from_external_impl::<false, false>(
            point_count,
            external_four_momenta,
        )
    }

    pub(crate) fn execute_contracted_tile_from_external_planes<const PROFILE: bool>(
        &mut self,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.execute_contracted_tile_from_external_impl::<PROFILE, true>(
            point_count,
            external_four_momenta,
        )
    }

    fn execute_contracted_tile_from_external_impl<const PROFILE: bool, const SOURCE_MAJOR: bool>(
        &mut self,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        if self.plan.strategy() != RecurrenceStrategy::ContractedColorUnion {
            return Err(invalid(
                "contracted-color execution requires a contracted-color recurrence plan",
            ));
        }
        self.fill_identity_momenta_from_external::<PROFILE, SOURCE_MAJOR>(
            point_count,
            external_four_momenta,
        )?;
        self.execute_direct_tile_impl::<PROFILE>(point_count)?;
        self.last_point_count = point_count;
        self.last_public_flow_id = None;
        self.last_representative_flow_id = None;
        Ok(self.borrowed_output(point_count, None, None))
    }

    fn execute_union_sources<const PROFILE: bool>(
        &mut self,
        selector: &DirectUnionHelicitySelectorPlan,
        point_count: u32,
    ) -> RusticolResult<()> {
        self.restore_physical_momenta();
        let handle = self
            .union_source_dispatch
            .ok_or_else(|| invalid("all-flow-union recurrence runtime has no source dispatcher"))?;
        let selection_end = selector
            .source_selection_start
            .checked_add(selector.source_selection_count)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence union source-selection range overflows usize",
                )
            })?;
        let selections = self
            .plan
            .resolved_source_selections()
            .get(selector.source_selection_start..selection_end)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence union source-selection range is out of bounds",
                )
            })?;

        let momentum_scalar_len =
            u64::try_from(self.momenta.len()).map_err(|_| invalid("momentum arena exceeds u64"))?;
        let parameter_count = u32::try_from(self.parameters_re.len())
            .map_err(|_| invalid("parameter catalog exceeds u32"))?;
        let factor_count = u32::try_from(self.factors_re.len())
            .map_err(|_| invalid("factor catalog exceeds u32"))?;
        let source_count = u32::try_from(self.plan.sources().len())
            .map_err(|_| invalid("source row count exceeds u32"))?;
        let variant_count = u32::try_from(self.plan.source_dispatch_variants().len())
            .map_err(|_| invalid("source-dispatch variant count exceeds u32"))?;
        let embedding_count = u32::try_from(self.plan.source_embeddings().len())
            .map_err(|_| invalid("source embedding count exceeds u32"))?;
        let selection_count = u32::try_from(selections.len())
            .map_err(|_| invalid("source selection count exceeds u32"))?;

        self.arena.begin_tile(point_count)?;
        let arena = self.arena.view()?;
        let momenta = DirectMomentumView {
            values: self.momenta.as_ptr(),
            scalar_len: momentum_scalar_len,
            form_count: self.momentum_form_count,
            lorentz_component_count: self.lorentz_component_count,
            point_stride: self.point_stride,
        };
        let parameters = DirectParameterView {
            values_re: self.parameters_re.as_ptr(),
            values_im: self.parameters_im.as_ptr(),
            value_count: parameter_count,
        };
        let factors = DirectFactorView {
            values_re: self.factors_re.as_ptr(),
            values_im: self.factors_im.as_ptr(),
            value_count: factor_count,
        };
        validate_direct_views(arena, momenta, parameters, factors)?;

        let started = PROFILE.then(Instant::now);
        if PROFILE {
            self.traffic_counters
                .record_call(selection_count, point_count);
        }
        let status = unsafe {
            (handle.call)(
                handle.context,
                arena,
                momenta,
                parameters,
                factors,
                self.plan.sources().as_ptr(),
                source_count,
                self.plan.source_dispatch_variants().as_ptr(),
                variant_count,
                self.plan.source_embeddings().as_ptr(),
                embedding_count,
                selections.as_ptr(),
                selection_count,
                point_count,
            )
        };
        if PROFILE {
            self.timings.union_source_fill += started
                .expect("profiled union source fill has a start time")
                .elapsed();
        }
        if status != DIRECT_STATUS_OK {
            return Err(RusticolError::evaluation(format!(
                "direct recurrence union source dispatcher returned status {status}"
            )));
        }
        if PROFILE {
            self.counters.source_calls = self.counters.source_calls.saturating_add(1);
            self.counters.source_rows = self
                .counters
                .source_rows
                .saturating_add(u64::from(selection_count));
            self.activity_counters.union_source_dispatch_calls = self
                .activity_counters
                .union_source_dispatch_calls
                .saturating_add(1);
            self.activity_counters.union_source_rows = self
                .activity_counters
                .union_source_rows
                .saturating_add(u64::from(selection_count));
        }
        Ok(())
    }

    fn execute_direct_tile(&mut self, point_count: u32) -> RusticolResult<()> {
        self.execute_direct_tile_impl::<true>(point_count)
    }

    fn execute_direct_tile_impl<const PROFILE: bool>(
        &mut self,
        point_count: u32,
    ) -> RusticolResult<()> {
        self.execute_direct_tile_rows_impl::<PROFILE>(point_count, None)
    }

    fn execute_selected_direct_tile_impl<const PROFILE: bool>(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
    ) -> RusticolResult<()> {
        self.execute_direct_tile_rows_impl::<PROFILE>(point_count, Some(selector))
    }

    fn execute_direct_tile_rows_impl<const PROFILE: bool>(
        &mut self,
        point_count: u32,
        selector: Option<&DirectReplaySelectorPlan>,
    ) -> RusticolResult<()> {
        self.validate_point_count(point_count)?;
        self.activate_point_layout(point_count)?;
        self.restore_physical_momenta();
        self.arena.begin_tile(point_count)?;
        self.last_point_count = 0;
        self.last_public_flow_id = None;
        self.last_representative_flow_id = None;
        // `execute_direct_plan` initializes every current stage exactly once
        self.last_selected_destination_ids = None;
        // before its first contribution group. Re-clearing all current planes
        // here only duplicates that work.
        let amplitude_ranges =
            selector.map_or(self.additive_amplitude_ranges.as_slice(), |selector| {
                selector
                    .destination_program
                    .as_ref()
                    .map_or(selector.amplitude_clear_ranges.as_ref(), |program| {
                        program.amplitude_clear_ranges.as_ref()
                    })
            });
        for range in amplitude_ranges {
            let component_base = u32::try_from(range.start).map_err(|_| {
                RusticolError::integrity("direct recurrence amplitude clear base exceeds u32")
            })?;
            let component_count = u32::try_from(range.end - range.start).map_err(|_| {
                RusticolError::integrity("direct recurrence amplitude clear count exceeds u32")
            })?;
            self.arena
                .clear_amplitude_active(component_base, component_count)?;
        }

        {
            let (current_re, current_im, amplitude_re, amplitude_im) =
                self.arena.split_slices_mut();
            let mut workspace = DirectWorkspace {
                current_re,
                current_im,
                amplitude_re,
                amplitude_im,
                momenta: self.momenta.as_slice(),
                momentum_form_count: self.momentum_form_count,
                lorentz_component_count: self.lorentz_component_count,
                parameters_re: self.parameters_re.as_slice(),
                parameters_im: self.parameters_im.as_slice(),
                factors_re: self.factors_re.as_slice(),
                factors_im: self.factors_im.as_slice(),
                point_stride: self.point_stride,
            };
            let started = PROFILE.then(Instant::now);
            let result = if let Some((selector, program)) = selector.and_then(|selector| {
                selector
                    .destination_program
                    .as_ref()
                    .map(|program| (selector, program))
            }) {
                execute_direct_plan_destination_selected::<PROFILE>(
                    &self.plan,
                    &program.row_groups,
                    selector.representative_flow_id,
                    &program.current_clear_stages,
                    &self.executors,
                    &mut workspace,
                    point_count,
                    &mut self.counters,
                    &mut self.role_timings,
                    &mut self.traffic_counters,
                )
            } else {
                match (PROFILE, selector) {
                    (true, Some(selector)) => {
                        execute_direct_plan_selected_profiled_with_fanout_and_traffic(
                            &self.plan,
                            &selector.row_groups,
                            selector.representative_flow_id,
                            &self.executors,
                            &self.contribution_fanout,
                            &mut workspace,
                            point_count,
                            &mut self.counters,
                            &mut self.role_timings,
                            &mut self.traffic_counters,
                        )
                    }
                    (false, Some(selector)) => execute_direct_plan_selected_unprofiled_with_fanout(
                        &self.plan,
                        &selector.row_groups,
                        selector.representative_flow_id,
                        &self.executors,
                        &self.contribution_fanout,
                        &mut workspace,
                        point_count,
                    ),
                    (true, None) => execute_direct_plan_profiled_with_fanout_and_traffic(
                        &self.plan,
                        &self.executors,
                        &self.contribution_fanout,
                        &mut workspace,
                        point_count,
                        &mut self.counters,
                        &mut self.role_timings,
                        &mut self.traffic_counters,
                    ),
                    (false, None) => execute_direct_plan_unprofiled_with_fanout(
                        &self.plan,
                        &self.executors,
                        &self.contribution_fanout,
                        &mut workspace,
                        point_count,
                    ),
                }
            };
            if PROFILE {
                self.timings.direct_execution += started
                    .expect("profiled direct execution has a start time")
                    .elapsed();
            }
            result?;
            #[cfg(any(test, feature = "on-the-fly-test-support"))]
            observe_direct_amplitudes_before_replay(
                &self.plan,
                selector.map(DirectReplaySelectorPlan::representative_flow_id),
                &workspace,
                point_count,
            )?;
        }

        if PROFILE {
            self.activity_counters.schedule_executions =
                self.activity_counters.schedule_executions.saturating_add(1);
        }
        Ok(())
    }

    fn borrowed_output(
        &self,
        point_count: u32,
        public_flow_id: Option<u32>,
        representative_flow_id: Option<u32>,
    ) -> DirectRecurrenceTileOutput<'_> {
        let (amplitude_re, amplitude_im) = self.arena.amplitude_slices();
        DirectRecurrenceTileOutput {
            amplitude_re,
            amplitude_im,
            amplitude_destinations: self.plan.amplitude_destinations(),
            point_count,
            point_stride: self.point_stride,
            destination_count: self.plan.amplitude_destination_count(),
            public_flow_id,
            representative_flow_id,
            selected_destination_ids: self.last_selected_destination_ids.as_deref(),
        }
    }

    fn validate_point_count(&self, point_count: u32) -> RusticolResult<()> {
        if point_count == 0 {
            return Err(invalid("point count must be positive"));
        }
        if point_count > self.tile_capacity {
            return Err(invalid(format!(
                "point count {point_count} exceeds point tile size {}",
                self.tile_capacity
            )));
        }
        Ok(())
    }

    fn compact_momentum_len(&self) -> usize {
        self.momentum_form_count as usize
            * usize::from(self.lorentz_component_count)
            * self.tile_capacity as usize
    }

    fn activate_point_layout(&mut self, point_count: u32) -> RusticolResult<()> {
        let packed = point_count == 1;
        if packed == self.packed_singleton_active || !self.executors.packed_singleton_capable() {
            return Ok(());
        }
        if !packed {
            self.ensure_tiled_point_storage()?;
        }
        if self.alternate_point_storage.is_none() {
            return Ok(());
        }
        if packed {
            self.restore_physical_momenta();
            self.copy_momentum_point_zero_to_alternate();
            let exact_singleton = self.momentum_filled_points == 1;
            let legacy_full_tile = self.momentum_replay_selector.is_none()
                && self.momentum_filled_points == self.tile_capacity;
            let replay_selector = exact_singleton
                .then_some(self.momentum_replay_selector)
                .flatten();
            self.swap_point_storage();
            self.momentum_filled_points = u32::from(exact_singleton || legacy_full_tile);
            self.momentum_replay_selector = replay_selector;
            return Ok(());
        }
        self.copy_momentum_point_zero_to_alternate();
        self.swap_point_storage();
        Ok(())
    }

    fn force_tiled_layout(&mut self) -> RusticolResult<()> {
        if self.packed_singleton_active {
            self.ensure_tiled_point_storage()?;
            self.copy_momentum_point_zero_to_alternate();
            self.swap_point_storage();
        }
        Ok(())
    }

    fn ensure_tiled_point_storage(&mut self) -> RusticolResult<()> {
        if self.alternate_point_storage.is_some() {
            return Ok(());
        }
        let parameter_view = DirectParameterView {
            values_re: self.parameters_re.as_ptr(),
            values_im: self.parameters_im.as_ptr(),
            value_count: u32::try_from(self.parameters_re.len())
                .map_err(|_| invalid("parameter catalog exceeds u32"))?,
        };
        let factor_view = DirectFactorView {
            values_re: self.factors_re.as_ptr(),
            values_im: self.factors_im.as_ptr(),
            value_count: u32::try_from(self.factors_re.len())
                .map_err(|_| invalid("factor catalog exceeds u32"))?,
        };
        let storage = allocate_point_storage(
            self.runtime_current_component_count,
            self.plan.amplitude_destination_count(),
            self.momentum_plane_count,
            self.momentum_form_count,
            self.lorentz_component_count,
            self.tile_capacity,
            false,
            parameter_view,
            factor_view,
        )?;
        let allocation_counters = self
            .allocation_counters
            .checked_add(storage.arena.allocation_counters())?
            .checked_add(storage.momenta.allocation_counters())?;
        self.alternate_point_storage = Some(storage);
        self.allocation_counters = allocation_counters;
        Ok(())
    }

    fn copy_momentum_point_zero_to_alternate(&mut self) {
        let source_stride = self.point_stride as usize;
        let alternate = self
            .alternate_point_storage
            .as_mut()
            .expect("point-zero copy requires alternate point storage");
        let destination_stride = alternate.arena.point_stride() as usize;
        let plane_count = self.momentum_plane_count as usize;
        assert_eq!(
            self.momenta.len(),
            plane_count * source_stride,
            "active momentum storage must cover every physical plane"
        );
        assert_eq!(
            alternate.momenta.len(),
            plane_count * destination_stride,
            "alternate momentum storage must cover every physical plane"
        );
        let source = self.momenta.as_slice();
        let destination = alternate.momenta.as_mut_slice();
        for plane in 0..plane_count {
            destination[plane * destination_stride] = source[plane * source_stride];
        }
    }

    fn swap_point_storage(&mut self) {
        let alternate = self
            .alternate_point_storage
            .as_mut()
            .expect("layout switching requires alternate point storage");
        std::mem::swap(&mut self.arena, &mut alternate.arena);
        std::mem::swap(&mut self.momenta, &mut alternate.momenta);
        self.point_stride = self.arena.point_stride();
        self.momenta_are_compact = false;
        self.momentum_filled_points = 0;
        self.momentum_replay_selector = None;
        self.packed_singleton_active = !self.packed_singleton_active;
        self.last_point_count = 0;
        self.last_public_flow_id = None;
        self.last_representative_flow_id = None;
    }

    fn compact_momenta_for_legacy_access(&mut self) {
        if self.momenta_are_compact {
            return;
        }
        let plane_count =
            self.momentum_form_count as usize * usize::from(self.lorentz_component_count);
        let tile_capacity = self.tile_capacity as usize;
        let point_stride = self.point_stride as usize;
        if tile_capacity != point_stride {
            let momenta = self.momenta.as_mut_slice();
            for plane in 0..plane_count {
                let source_start = plane * point_stride;
                let destination_start = plane * tile_capacity;
                momenta.copy_within(
                    source_start..source_start + tile_capacity,
                    destination_start,
                );
            }
        }
        self.momenta_are_compact = true;
    }

    fn restore_physical_momenta(&mut self) {
        if !self.momenta_are_compact {
            return;
        }
        let plane_count =
            self.momentum_form_count as usize * usize::from(self.lorentz_component_count);
        let tile_capacity = self.tile_capacity as usize;
        let point_stride = self.point_stride as usize;
        if tile_capacity != point_stride {
            let momenta = self.momenta.as_mut_slice();
            for plane in (0..plane_count).rev() {
                let source_start = plane * tile_capacity;
                let destination_start = plane * point_stride;
                momenta.copy_within(
                    source_start..source_start + tile_capacity,
                    destination_start,
                );
            }
        }
        self.momenta_are_compact = false;
    }

    fn validate_replay_selector(&self, selector: &DirectReplaySelectorPlan) -> RusticolResult<()> {
        if selector.runtime_layout_digest != self.plan.runtime_layout_digest() {
            return Err(RusticolError::integrity(
                "direct recurrence replay selector belongs to a different plan",
            ));
        }
        if selector.source_permutation.len() != self.plan.external_source_count() as usize {
            return Err(RusticolError::integrity(
                "direct recurrence replay selector source permutation has an inconsistent length",
            ));
        }
        if selector.source_momentum_signs.len() != selector.source_permutation.len() {
            return Err(RusticolError::integrity(
                "direct recurrence replay selector has inconsistent momentum-sign coverage",
            ));
        }
        Ok(())
    }

    fn validate_union_helicity_descriptor(
        &self,
        descriptor: &DirectResolvedHelicityDescriptor,
    ) -> RusticolResult<(usize, usize)> {
        let start = usize::try_from(descriptor.source_selection_start).map_err(|_| {
            RusticolError::integrity("direct recurrence union source-selection start exceeds usize")
        })?;
        let count = usize::try_from(descriptor.source_selection_count).map_err(|_| {
            RusticolError::integrity("direct recurrence union source-selection count exceeds usize")
        })?;
        if count != self.plan.external_source_count() as usize {
            return Err(RusticolError::integrity(
                "direct recurrence union helicity does not select every external source",
            ));
        }
        let end = start.checked_add(count).ok_or_else(|| {
            RusticolError::integrity(
                "direct recurrence union source-selection range overflows usize",
            )
        })?;
        let selections = self
            .plan
            .resolved_source_selections()
            .get(start..end)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence union source-selection range is out of bounds",
                )
            })?;
        for (source_slot, selection) in selections.iter().enumerate() {
            if selection.source_slot != source_slot as u32 {
                return Err(RusticolError::integrity(
                    "direct recurrence union source selections are not in source-slot order",
                ));
            }
            let variant = self
                .plan
                .source_dispatch_variants()
                .get(selection.dispatch_variant_id as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence union source selection references an absent variant",
                    )
                })?;
            let source = self
                .plan
                .sources()
                .get(variant.source_row_id as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence union source variant references an absent row",
                    )
                })?;
            if source.source_slot != selection.source_slot
                || source.source_template_or_dispatch_domain != variant.dispatch_domain_id
            {
                return Err(RusticolError::integrity(
                    "direct recurrence union source selection disagrees with its variant",
                ));
            }
            let embedding_start = usize::try_from(variant.embedding_start).map_err(|_| {
                RusticolError::integrity(
                    "direct recurrence union source embedding start exceeds usize",
                )
            })?;
            let embedding_end = embedding_start
                .checked_add(variant.embedding_count as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence union source embedding range overflows usize",
                    )
                })?;
            if self
                .plan
                .source_embeddings()
                .get(embedding_start..embedding_end)
                .is_none()
            {
                return Err(RusticolError::integrity(
                    "direct recurrence union source embedding range is out of bounds",
                ));
            }
        }
        Ok((start, count))
    }

    fn validate_union_selector(
        &self,
        selector: &DirectUnionHelicitySelectorPlan,
    ) -> RusticolResult<()> {
        if self.plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "union helicity selector requires an all-flow-union recurrence plan",
            ));
        }
        if selector.runtime_layout_digest != self.plan.runtime_layout_digest() {
            return Err(RusticolError::integrity(
                "direct recurrence union helicity selector belongs to a different plan",
            ));
        }
        let descriptor = self
            .plan
            .resolved_helicities()
            .get(selector.resolved_helicity_id as usize)
            .filter(|descriptor| descriptor.id == selector.resolved_helicity_id)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence union helicity selector references an absent helicity",
                )
            })?;
        let (start, count) = self.validate_union_helicity_descriptor(descriptor)?;
        if start != selector.source_selection_start || count != selector.source_selection_count {
            return Err(RusticolError::integrity(
                "direct recurrence union helicity selector ranges do not match the plan",
            ));
        }
        Ok(())
    }
}

fn replay_cache_split_complex_scalar_count(
    plan: &DirectRecurrencePlan,
    persisted_split_complex_scalar_count: usize,
    fanout_scratch_component_count: u32,
) -> RusticolResult<usize> {
    if plan.strategy() != RecurrenceStrategy::TopologyReplay {
        return Ok(persisted_split_complex_scalar_count);
    }
    let representative_sectors = plan
        .replay_targets()
        .iter()
        .map(|target| target.representative_id)
        .collect::<BTreeSet<_>>();
    if representative_sectors.is_empty() {
        return Ok(persisted_split_complex_scalar_count);
    }
    let current_arena_components = u64::from(plan.current_arena_components());
    let mut maximum = 0_u64;
    for representative_sector_id in representative_sectors {
        let summary = plan.selector_work_summary(representative_sector_id)?;
        let active_current_components = summary
            .semantic_component_count
            .min(current_arena_components);
        let active_complex_planes = active_current_components
            .checked_add(summary.amplitude_destination_count)
            .and_then(|count| count.checked_add(u64::from(fanout_scratch_component_count)))
            .ok_or_else(|| invalid("selected replay cache plane count overflows u64"))?;
        let active_split_scalars = active_complex_planes
            .checked_mul(2)
            .ok_or_else(|| invalid("selected replay cache scalar count overflows u64"))?;
        maximum = maximum.max(active_split_scalars);
    }
    let maximum = usize::try_from(maximum)
        .map_err(|_| invalid("selected replay cache scalar count exceeds usize"))?;
    if maximum == 0 {
        return Err(invalid(
            "topology-replay selector domains expose no cache-resident work",
        ));
    }
    Ok(maximum.min(persisted_split_complex_scalar_count))
}

fn greatest_power_of_two_not_exceeding(value: usize) -> usize {
    if value == 0 {
        return 1;
    }
    1_usize << (usize::BITS - 1 - value.leading_zeros())
}

/// Component storage can be recycled after a current's last use. Resolve
/// parents by lifetime, not by their physical component address alone.
struct ReplayCurrentIndex {
    by_stage_and_base: BTreeMap<(u16, u32), usize>,
    by_base: BTreeMap<u32, Vec<usize>>,
}

impl ReplayCurrentIndex {
    fn new(plan: &DirectRecurrencePlan) -> RusticolResult<Self> {
        let mut by_stage_and_base = BTreeMap::new();
        let mut by_base = BTreeMap::<u32, Vec<usize>>::new();
        for (id, current) in plan.currents().iter().enumerate() {
            if by_stage_and_base
                .insert((current.stage, current.component_base), id)
                .is_some()
            {
                return Err(invalid("replay current address is repeated within a stage"));
            }
            by_base.entry(current.component_base).or_default().push(id);
        }
        for ids in by_base.values_mut() {
            ids.sort_unstable_by_key(|&id| {
                let current = &plan.currents()[id];
                (current.first_use, current.last_use, id)
            });
        }
        Ok(Self {
            by_stage_and_base,
            by_base,
        })
    }

    fn destination(&self, stage: u16, base: u32) -> RusticolResult<usize> {
        self.by_stage_and_base
            .get(&(stage, base))
            .copied()
            .ok_or_else(|| invalid("replay contribution has no current at its stage"))
    }

    fn parent(&self, plan: &DirectRecurrencePlan, stage: u16, base: u32) -> RusticolResult<usize> {
        let ids = self
            .by_base
            .get(&base)
            .ok_or_else(|| invalid("replay parent current address is absent"))?;
        let end = ids.partition_point(|&id| plan.currents()[id].first_use <= u32::from(stage));
        ids.get(end.wrapping_sub(1))
            .copied()
            .filter(|&id| plan.currents()[id].last_use >= u32::from(stage))
            .ok_or_else(|| invalid("replay parent current is not live at its use stage"))
    }
}

fn selected_replay_destination_program(
    plan: &DirectRecurrencePlan,
    representative: u32,
    flow_groups: &[DirectRowGroupDescriptor],
    destination_ids: &[u32],
) -> RusticolResult<DirectReplayDestinationProgram> {
    if destination_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid(
            "replay amplitude destination IDs must be sorted and unique",
        ));
    }
    let mut amplitude_marks =
        zeroed_marks(plan.amplitude_destinations().len(), "destination selection")?;
    for &id in destination_ids {
        let destination = plan
            .amplitude_destinations()
            .get(id as usize)
            .filter(|destination| destination.target_sector_id == representative)
            .ok_or_else(|| {
                invalid("replay amplitude destination is outside the selected representative")
            })?;
        amplitude_marks[destination.id as usize] = 1;
    }

    let index = ReplayCurrentIndex::new(plan)?;
    let mut source_marks = zeroed_marks(plan.sources().len(), "source selection")?;
    let mut contribution_marks =
        zeroed_marks(plan.contributions().len(), "contribution selection")?;
    let mut finalization_marks =
        zeroed_marks(plan.finalizations().len(), "finalization selection")?;
    let mut closure_marks = zeroed_marks(plan.closures().len(), "closure selection")?;
    let mut current_marks = zeroed_marks(plan.currents().len(), "current selection")?;
    let mut contributions_by_current = vec![Vec::<(usize, u16)>::new(); plan.currents().len()];
    let mut finalizations_by_current = vec![Vec::<usize>::new(); plan.currents().len()];
    let mut pending = Vec::new();
    for group in flow_groups {
        let start = group.row_start as usize;
        let end = start + group.row_count as usize;
        match group.role {
            DirectExecutorRole::Contribution => {
                for row_id in start..end {
                    let row = &plan.contributions()[row_id];
                    let current = index.destination(group.stage, row.destination_component_base)?;
                    contributions_by_current[current].push((row_id, group.stage));
                }
            }
            DirectExecutorRole::Finalization => {
                for row_id in start..end {
                    let row = &plan.finalizations()[row_id];
                    let current = index.destination(group.stage, row.component_base)?;
                    finalizations_by_current[current].push(row_id);
                }
            }
            DirectExecutorRole::Closure => {
                for (row, mark) in plan.closures()[start..end]
                    .iter()
                    .zip(&mut closure_marks[start..end])
                {
                    if amplitude_marks[row.amplitude_destination_id as usize] == 0 {
                        continue;
                    }
                    *mark = 1;
                    pending.push(index.parent(plan, group.stage, row.parent0_component_base)?);
                    if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 {
                        pending.push(index.parent(
                            plan,
                            group.stage,
                            row.parent1_component_base_or_sentinel,
                        )?);
                    }
                }
            }
            DirectExecutorRole::Source => {}
        }
    }
    while let Some(current_id) = pending.pop() {
        if current_marks[current_id] != 0 {
            continue;
        }
        current_marks[current_id] = 1;
        let current = &plan.currents()[current_id];
        if current.source_row_or_sentinel != DIRECT_NONE_U32 {
            source_marks[current.source_row_or_sentinel as usize] = 1;
        }
        for &row_id in &finalizations_by_current[current_id] {
            finalization_marks[row_id] = 1;
        }
        for &(row_id, stage) in &contributions_by_current[current_id] {
            let row = &plan.contributions()[row_id];
            contribution_marks[row_id] = 1;
            pending.push(index.parent(plan, stage, row.parent0_component_base)?);
            if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 {
                pending.push(index.parent(plan, stage, row.parent1_component_base_or_sentinel)?);
            }
        }
    }

    let mut row_groups = Vec::new();
    for group in flow_groups {
        let marks = match group.role {
            DirectExecutorRole::Source => &source_marks,
            DirectExecutorRole::Contribution => &contribution_marks,
            DirectExecutorRole::Finalization => &finalization_marks,
            DirectExecutorRole::Closure => &closure_marks,
        };
        let end = group.row_start as usize + group.row_count as usize;
        let mut row = group.row_start as usize;
        while row < end {
            if marks[row] == 0 {
                row += 1;
                continue;
            }
            let start = row;
            while row < end && marks[row] != 0 {
                row += 1;
            }
            row_groups.push(DirectRowGroupDescriptor {
                row_start: start as u64,
                row_count: (row - start) as u32,
                ..*group
            });
        }
    }
    let mut clear_by_stage = BTreeMap::<u16, Vec<Range<u32>>>::new();
    for (id, current) in plan.currents().iter().enumerate() {
        if current_marks[id] == 0 || current.node_kind != DirectNodeKind::Current {
            continue;
        }
        // Dependency closure retains every contribution to a live current in
        // this flow selection, in its persisted execution order. When that
        // first retained row overwrites the destination, clearing it would
        // duplicate the write. Check the already-built row list here rather
        // than adding a separate analysis or any work to tile execution.
        // A valid custom plan may select away the global initializer through
        // a narrower row selector domain; retain its ordinary clear below.
        if contributions_by_current[id]
            .first()
            .is_some_and(|&(row_id, _)| {
                let row = &plan.contributions()[row_id];
                if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION == 0 {
                    return false;
                }
                // Retain self-read clears even for a persisted zero factor:
                // the low-level runtime allows callers to replace factors.
                // Native output widths were checked during runtime loading;
                // copy widths live in the row and must cover the full target.
                row.parent0_component_base != row.destination_component_base
                    && row.parent1_component_base_or_sentinel != row.destination_component_base
                    && (row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0
                        || row.parent0_momentum_form_id == u32::from(current.component_count))
            })
        {
            continue;
        }
        clear_by_stage.entry(current.stage).or_default().push(
            current.component_base..current.component_base + u32::from(current.component_count),
        );
    }
    let current_clear_stages = clear_by_stage
        .into_iter()
        .map(|(stage, mut ranges)| {
            ranges.sort_unstable_by_key(|range| range.start);
            let mut merged = Vec::<DirectCurrentClearRange>::new();
            for range in ranges {
                if let Some(last) = merged.last_mut()
                    && range.start <= last.component_base + last.component_count
                {
                    last.component_count = (last.component_base + last.component_count)
                        .max(range.end)
                        - last.component_base;
                    continue;
                }
                merged.push(DirectCurrentClearRange {
                    component_base: range.start,
                    component_count: range.end - range.start,
                });
            }
            DirectStageClearRanges {
                stage,
                ranges: merged.into_boxed_slice(),
            }
        })
        .collect::<Vec<_>>();
    Ok(DirectReplayDestinationProgram {
        destination_ids: Arc::from(destination_ids),
        row_groups: row_groups.into_boxed_slice(),
        amplitude_clear_ranges: compact_ranges(&amplitude_marks, "destination selection")?
            .into_boxed_slice(),
        current_clear_stages: current_clear_stages.into_boxed_slice(),
    })
}

fn selected_replay_row_groups(
    plan: &DirectRecurrencePlan,
    representative_sector_id: u32,
) -> RusticolResult<Vec<DirectRowGroupDescriptor>> {
    let mut selected = Vec::new();
    for descriptor in plan.row_groups() {
        let start = usize::try_from(descriptor.row_start).map_err(|_| {
            RusticolError::integrity("direct recurrence row-group start exceeds usize")
        })?;
        let count = descriptor.row_count as usize;
        let mut offset = 0usize;
        while offset < count {
            let run_start = offset;
            let selector_domain_id =
                row_selector_domain_id(plan, descriptor.role, start + run_start)?;
            while offset < count
                && row_selector_domain_id(plan, descriptor.role, start + offset)?
                    == selector_domain_id
            {
                offset += 1;
            }
            // Preserve selector-domain boundaries even when adjacent domains
            // are both active, so full-flow fanout runs retain their original
            // authenticated partitions.
            if !plan.selector_domain_contains(selector_domain_id, representative_sector_id)? {
                continue;
            }
            let mut run = *descriptor;
            run.row_start = descriptor
                .row_start
                .checked_add(run_start as u64)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence selected row-group start overflows u64",
                    )
                })?;
            run.row_count = u32::try_from(offset - run_start).map_err(|_| {
                RusticolError::integrity("direct recurrence selected row-group count exceeds u32")
            })?;
            selected.push(run);
        }
    }
    if selected.is_empty() {
        return Err(RusticolError::integrity(format!(
            "direct recurrence representative sector {representative_sector_id} has no executable rows"
        )));
    }
    Ok(selected)
}

fn row_selector_domain_id(
    plan: &DirectRecurrencePlan,
    role: DirectExecutorRole,
    row_index: usize,
) -> RusticolResult<u32> {
    match role {
        DirectExecutorRole::Source => plan
            .sources()
            .get(row_index)
            .map(|row| row.selector_domain_id),
        DirectExecutorRole::Contribution => plan
            .contributions()
            .get(row_index)
            .map(|row| row.selector_domain_id),
        DirectExecutorRole::Finalization => plan
            .finalizations()
            .get(row_index)
            .map(|row| row.selector_domain_id),
        DirectExecutorRole::Closure => plan
            .closures()
            .get(row_index)
            .map(|row| row.selector_domain_id),
    }
    .ok_or_else(|| RusticolError::integrity("direct recurrence selected row is out of bounds"))
}

fn selected_amplitude_clear_ranges(
    plan: &DirectRecurrencePlan,
    representative_sector_id: u32,
) -> RusticolResult<Vec<Range<usize>>> {
    let destination_count = usize::try_from(plan.amplitude_destination_count())
        .map_err(|_| invalid("amplitude destination count exceeds usize"))?;
    let mut marked = zeroed_marks(destination_count, "selected amplitude clear map")?;
    for destination in plan.amplitude_destinations() {
        if plan
            .selector_domain_contains(destination.selector_domain_id, representative_sector_id)?
        {
            mark_range(
                &mut marked,
                destination.id as usize,
                1,
                "selected amplitude",
            )?;
        }
    }
    compact_ranges(&marked, "selected amplitude")
}

fn amplitude_clear_ranges(plan: &DirectRecurrencePlan) -> RusticolResult<Vec<Range<usize>>> {
    let destination_count = usize::try_from(plan.amplitude_destination_count())
        .map_err(|_| invalid("amplitude destination count exceeds usize"))?;
    let mut marked = zeroed_marks(destination_count, "amplitude clear map")?;
    for row in plan.closures() {
        mark_range(
            &mut marked,
            row.amplitude_destination_id as usize,
            1,
            "amplitude",
        )?;
    }
    compact_ranges(&marked, "amplitude")
}

fn zeroed_marks(len: usize, label: &str) -> RusticolResult<Vec<u8>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        RusticolError::internal(format!(
            "could not allocate direct recurrence {label}: {error}"
        ))
    })?;
    values.resize(len, 0);
    Ok(values)
}

fn mark_range(marked: &mut [u8], start: usize, count: usize, label: &str) -> RusticolResult<()> {
    let end = start
        .checked_add(count)
        .ok_or_else(|| RusticolError::integrity(format!("{label} clear range overflows usize")))?;
    let range = marked.get_mut(start..end).ok_or_else(|| {
        RusticolError::integrity(format!("{label} clear range exceeds its arena"))
    })?;
    range.fill(1);
    Ok(())
}

fn compact_ranges(marked: &[u8], label: &str) -> RusticolResult<Vec<Range<usize>>> {
    let mut ranges = Vec::new();
    ranges.try_reserve_exact(marked.len()).map_err(|error| {
        RusticolError::internal(format!(
            "could not allocate direct recurrence {label} clear ranges: {error}"
        ))
    })?;
    let mut index = 0;
    while index < marked.len() {
        if marked[index] == 0 {
            index += 1;
            continue;
        }
        let start = index;
        while index < marked.len() && marked[index] != 0 {
            index += 1;
        }
        ranges.push(start..index);
    }
    Ok(ranges)
}

/// Interpret an already validated, generation-time momentum form. Only
/// independent phase-space points are reordered: each point keeps its original
/// initial addition and term order, including cancellation and signed zero.
/// There is no auxiliary input buffer or selection-dependent preparation.
#[inline]
fn fill_momentum_form_from_external<const SOURCE_MAJOR: bool>(
    form_planes: &mut [f64],
    point_stride: usize,
    active_points: usize,
    external_four_momenta: &[f64],
    source_count: usize,
    mut terms: impl Iterator<Item = (usize, f64)>,
) {
    let Some(first) = terms.next() else {
        for plane in form_planes.chunks_exact_mut(point_stride) {
            plane[..active_points].fill(0.0);
        }
        return;
    };
    fill_external_momentum_term::<true, SOURCE_MAJOR>(
        form_planes,
        point_stride,
        active_points,
        external_four_momenta,
        source_count,
        first,
    );
    for term in terms {
        fill_external_momentum_term::<false, SOURCE_MAJOR>(
            form_planes,
            point_stride,
            active_points,
            external_four_momenta,
            source_count,
            term,
        );
    }
}

#[inline]
fn fill_external_momentum_term<const INITIALIZE: bool, const SOURCE_MAJOR: bool>(
    form_planes: &mut [f64],
    point_stride: usize,
    active_points: usize,
    external_four_momenta: &[f64],
    source_count: usize,
    (external_slot, coefficient): (usize, f64),
) {
    let external_point_stride = source_count * 4;
    for (component, plane) in form_planes.chunks_exact_mut(point_stride).enumerate() {
        if SOURCE_MAJOR {
            let input_start = (external_slot * 4 + component) * active_points;
            for (value, source) in plane[..active_points]
                .iter_mut()
                .zip(&external_four_momenta[input_start..input_start + active_points])
            {
                if INITIALIZE {
                    *value = 0.0 + coefficient * source;
                } else {
                    *value += coefficient * source;
                }
            }
        } else {
            let input_start = external_slot * 4 + component;
            for (value, source) in plane[..active_points].iter_mut().zip(
                external_four_momenta[input_start..]
                    .iter()
                    .step_by(external_point_stride),
            ) {
                if INITIALIZE {
                    *value = 0.0 + coefficient * source;
                } else {
                    *value += coefficient * source;
                }
            }
        }
    }
}

fn validate_split_values(
    label: &str,
    expected_len: usize,
    values_re: &[f64],
    values_im: &[f64],
) -> RusticolResult<()> {
    if values_re.len() != expected_len || values_im.len() != expected_len {
        return Err(invalid(format!(
            "{label} split-complex lengths are {}, {}, expected {expected_len}",
            values_re.len(),
            values_im.len()
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!(
        "direct recurrence execution runtime: {}",
        message.into()
    ))
}

#[cfg(test)]
#[path = "direct_runtime_tests.rs"]
mod tests;
