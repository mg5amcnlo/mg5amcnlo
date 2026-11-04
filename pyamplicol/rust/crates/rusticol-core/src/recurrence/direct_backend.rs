// SPDX-License-Identifier: 0BSD

//! Direct-arena prepared executor ABI for compact recurrence schedules.
//!
//! This module deliberately does not depend on the eager runtime. Direct
//! executors consume fixed-width recurrence rows and persistent split-complex
//! arenas. They never receive a packed evaluator input or output buffer.

use super::RecurrenceStrategy;
use super::SemanticDigest;
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
    DIRECT_NONE_U32, DirectClosureRow, DirectContributionRow, DirectCurrentDescriptor,
    DirectExecutorRole, DirectFinalizationRow, DirectNodeKind, DirectRecurrencePlan,
    DirectRowGroupDescriptor, DirectSourceRow,
};
pub use super::direct_plan::{
    DirectResolvedSourceSelection, DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow,
};
use crate::direct_arena::{DirectArenaTrafficCounters, clear_split_active_range};
pub use crate::direct_arena::{
    DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
};
use crate::{RusticolError, RusticolResult};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_int, c_void};
use std::mem::size_of;
use std::time::{Duration, Instant};

pub const RECURRENCE_DIRECT_BACKEND_ABI: &str = "rusticol.recurrence-direct-backend.v1";

pub const DIRECT_STATUS_OK: c_int = 0;

thread_local! {
    static DIRECT_EXECUTOR_ERROR_DETAIL: RefCell<Option<RusticolError>> = const { RefCell::new(None) };
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    static DIRECT_CURRENT_OBSERVATION: RefCell<Option<DirectCurrentObservation>> = const { RefCell::new(None) };
}

/// One test-only semantic-current observation from the authenticated direct
/// runtime.  This is deliberately absent from release builds and public ABIs.
#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DirectObservedCurrentValue {
    pub descriptor: super::direct_plan::DirectCurrentDescriptor,
    /// Split-complex values in component-major, then point-major order.
    pub values: Vec<(f64, f64)>,
}

/// Test-only capture of the current values produced for one exact direct-plan
/// execution.  The two plan digests and selected sector bind the values to the
/// authenticated schedule and query which produced them.
#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DirectCurrentObservation {
    pub plan_semantic_digest: SemanticDigest,
    pub runtime_layout_digest: SemanticDigest,
    pub selected_sector_id: Option<u32>,
    pub point_count: u32,
    pub currents: BTreeMap<u32, DirectObservedCurrentValue>,
    /// Representative amplitudes captured after closures and before replay
    /// phase/multiplicity scaling, keyed by DirectPlan destination ID.
    pub amplitudes_before_replay: BTreeMap<u32, Vec<(f64, f64)>>,
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) fn begin_direct_current_observation(
    plan: &DirectRecurrencePlan,
    selected_sector_id: Option<u32>,
    point_count: u32,
) -> RusticolResult<()> {
    if point_count == 0 {
        return Err(RusticolError::invalid_argument(
            "direct current observation requires at least one point",
        ));
    }
    DIRECT_CURRENT_OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(RusticolError::invalid_argument(
                "direct current observation is already active on this thread",
            ));
        }
        *slot = Some(DirectCurrentObservation {
            plan_semantic_digest: plan.semantic_digest(),
            runtime_layout_digest: plan.runtime_layout_digest(),
            selected_sector_id,
            point_count,
            currents: BTreeMap::new(),
            amplitudes_before_replay: BTreeMap::new(),
        });
        Ok(())
    })
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) fn take_direct_current_observation() -> RusticolResult<DirectCurrentObservation> {
    DIRECT_CURRENT_OBSERVATION.with(|slot| {
        slot.borrow_mut().take().ok_or_else(|| {
            RusticolError::invalid_argument(
                "direct current observation is not active on this thread",
            )
        })
    })
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
fn direct_current_observation_active() -> bool {
    DIRECT_CURRENT_OBSERVATION.with(|slot| slot.borrow().is_some())
}

#[cfg(not(any(test, feature = "on-the-fly-test-support")))]
const fn direct_current_observation_active() -> bool {
    false
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(super) fn observe_direct_amplitudes_before_replay(
    plan: &DirectRecurrencePlan,
    selected_sector_id: Option<u32>,
    workspace: &DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let active = DIRECT_CURRENT_OBSERVATION.with(|slot| slot.borrow().is_some());
    if !active {
        return Ok(());
    }
    DIRECT_CURRENT_OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let observation = slot.as_mut().ok_or_else(|| {
            RusticolError::integrity("direct current observation disappeared before amplitudes")
        })?;
        if observation.plan_semantic_digest != plan.semantic_digest()
            || observation.runtime_layout_digest != plan.runtime_layout_digest()
            || observation.selected_sector_id != selected_sector_id
            || observation.point_count != point_count
        {
            return Err(RusticolError::integrity(
                "direct amplitude observation does not match this plan execution",
            ));
        }
        let stride = workspace.point_stride as usize;
        let active_points = point_count as usize;
        for destination in plan.amplitude_destinations().iter().filter(|destination| {
            selected_sector_id.is_none_or(|sector| destination.target_sector_id == sector)
        }) {
            let start = destination.id as usize * stride;
            let end = start.checked_add(active_points).ok_or_else(|| {
                RusticolError::integrity("direct amplitude observation range overflows usize")
            })?;
            let real = workspace.amplitude_re.get(start..end).ok_or_else(|| {
                RusticolError::integrity("direct amplitude observation real range is absent")
            })?;
            let imag = workspace.amplitude_im.get(start..end).ok_or_else(|| {
                RusticolError::integrity("direct amplitude observation imaginary range is absent")
            })?;
            observation.amplitudes_before_replay.insert(
                destination.id,
                real.iter().copied().zip(imag.iter().copied()).collect(),
            );
        }
        Ok(())
    })
}

/// Clear the synchronous detail channel before executing one direct plan.
pub(crate) fn clear_direct_executor_error_detail() {
    DIRECT_EXECUTOR_ERROR_DETAIL.with(|detail| *detail.borrow_mut() = None);
}

/// Preserve the first concrete callback failure across the narrow
/// integer-status ABI.
pub(crate) fn record_direct_executor_error_detail(detail: RusticolError) {
    DIRECT_EXECUTOR_ERROR_DETAIL.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(detail);
        }
    });
}

fn take_direct_executor_error_detail() -> Option<RusticolError> {
    DIRECT_EXECUTOR_ERROR_DETAIL.with(|detail| detail.borrow_mut().take())
}

pub type DirectSourceExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectSourceRow,
    u32,
    u32,
) -> c_int;

/// Runtime-helicity source dispatcher used only by all-flow-union plans.
///
/// Unlike a static source executor, this callable receives one authenticated
/// resolved selection plus the process-bound variant and embedding catalogs.
/// It fills every full source current before the remaining direct schedule is
/// executed. The pointed-to tables and context must outlive the call.
pub type DirectUnionSourceDispatchExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectSourceRow,
    u32,
    *const DirectSourceDispatchVariantDescriptor,
    u32,
    *const DirectSourceEmbeddingRow,
    u32,
    *const DirectResolvedSourceSelection,
    u32,
    u32,
) -> c_int;

/// Typed handle for one immutable all-flow-union source dispatcher.
#[derive(Clone, Copy)]
pub struct DirectUnionSourceDispatchHandle {
    pub call: DirectUnionSourceDispatchExecutor,
    pub context: *const c_void,
}

pub type DirectContributionExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectContributionRow,
    u32,
    u32,
) -> c_int;

pub(crate) type DirectContributionFanoutExecutor = for<'a> unsafe fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    DirectContributionFanoutBatch<'a>,
) -> RusticolResult<()>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum DirectInteractionIntrinsicKind {
    ColorOrderedThreeVector,
    VectorWedgeVector,
    AntisymmetricTensorVector,
}

pub(crate) type DirectInteractionBundleExecutor = for<'a> unsafe fn(
    DirectInteractionExecutorContexts,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    DirectInteractionBundleBatch<'a>,
) -> RusticolResult<()>;

#[derive(Clone, Copy)]
pub(crate) struct DirectInteractionExecutorContexts {
    pub(crate) color: *const c_void,
    pub(crate) vector_wedge_vector: *const c_void,
    pub(crate) antisymmetric_tensor_vector: *const c_void,
}

#[derive(Clone, Copy)]
pub(crate) struct DirectContributionFanoutExecutorHandle {
    pub(crate) call: DirectContributionFanoutExecutor,
    pub(crate) context: *const c_void,
    pub(crate) destination_component_count: u32,
    pub(crate) parent_component_counts: [u32; 2],
    pub(crate) requires_two_momenta: bool,
    pub(crate) required_parameter_count: u32,
    pub(crate) interaction_kind: DirectInteractionIntrinsicKind,
    pub(crate) interaction_call: Option<DirectInteractionBundleExecutor>,
}

pub type DirectFinalizationExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectFinalizationRow,
    u32,
    u32,
) -> c_int;

pub type DirectClosureExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectClosureRow,
    u32,
    u32,
) -> c_int;

#[derive(Clone, Copy)]
pub enum DirectExecutorHandle {
    Source {
        call: DirectSourceExecutor,
        context: *const c_void,
    },
    Contribution {
        call: DirectContributionExecutor,
        context: *const c_void,
    },
    Finalization {
        call: DirectFinalizationExecutor,
        context: *const c_void,
    },
    Closure {
        call: DirectClosureExecutor,
        context: *const c_void,
    },
}

impl DirectExecutorHandle {
    pub(crate) const fn role(self) -> DirectExecutorRole {
        match self {
            Self::Source { .. } => DirectExecutorRole::Source,
            Self::Contribution { .. } => DirectExecutorRole::Contribution,
            Self::Finalization { .. } => DirectExecutorRole::Finalization,
            Self::Closure { .. } => DirectExecutorRole::Closure,
        }
    }
}

pub struct DirectExecutorCatalog {
    handles: Box<[Option<DirectExecutorHandle>]>,
    contribution_metadata: Box<[Option<DirectContributionExecutionMetadata>]>,
    contribution_fanout: Box<[Option<DirectContributionFanoutExecutorHandle>]>,
    packed_singleton_capable: bool,
    plan_layout_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
}

/// Authenticated contribution shape and row-local scalar dependency.
///
/// `exact_factor_is_kernel_input` is fail-closed. It is false only when the
/// prepared template proves that the exact factor is applied exclusively by
/// the destination adapter (currently Rusticol contribution intrinsics).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectContributionExecutionMetadata {
    pub(crate) destination_component_count: u32,
    pub(crate) exact_factor_is_kernel_input: bool,
}

impl DirectContributionExecutionMetadata {
    pub(crate) fn new(
        destination_component_count: u32,
        exact_factor_is_kernel_input: bool,
    ) -> RusticolResult<Self> {
        if destination_component_count == 0 {
            return Err(RusticolError::integrity(
                "direct contribution executor has zero destination components",
            ));
        }
        Ok(Self {
            destination_component_count,
            exact_factor_is_kernel_input,
        })
    }
}

impl DirectExecutorCatalog {
    pub fn new(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<DirectExecutorHandle>,
    ) -> RusticolResult<Self> {
        let sparse_handles = handles.into_iter().map(Some).collect::<Vec<_>>();
        let metadata = conservative_contribution_metadata(plan, sparse_handles.len())?;
        Self::new_sparse_with_metadata(
            plan,
            direct_template_catalog_digest,
            sparse_handles,
            metadata,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn new_sparse(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<Option<DirectExecutorHandle>>,
    ) -> RusticolResult<Self> {
        let metadata = conservative_contribution_metadata(plan, handles.len())?;
        Self::new_sparse_with_metadata(plan, direct_template_catalog_digest, handles, metadata)
    }

    pub(crate) fn new_sparse_with_metadata(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<Option<DirectExecutorHandle>>,
        contribution_metadata: Vec<Option<DirectContributionExecutionMetadata>>,
    ) -> RusticolResult<Self> {
        let contribution_fanout = vec![None; handles.len()];
        Self::new_sparse_with_metadata_and_fanout(
            plan,
            direct_template_catalog_digest,
            handles,
            contribution_metadata,
            contribution_fanout,
        )
    }

    pub(crate) fn new_sparse_with_metadata_and_fanout(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<Option<DirectExecutorHandle>>,
        contribution_metadata: Vec<Option<DirectContributionExecutionMetadata>>,
        contribution_fanout: Vec<Option<DirectContributionFanoutExecutorHandle>>,
    ) -> RusticolResult<Self> {
        if handles.is_empty() {
            return Err(RusticolError::invalid_argument(
                "direct recurrence executor catalog must not be empty",
            ));
        }
        if contribution_metadata.len() != handles.len() {
            return Err(RusticolError::integrity(
                "direct contribution metadata does not cover the executor catalog",
            ));
        }
        if contribution_fanout.len() != handles.len() {
            return Err(RusticolError::integrity(
                "direct contribution fanout capabilities do not cover the executor catalog",
            ));
        }
        for (executor_id, capability) in contribution_fanout.iter().copied().enumerate() {
            let Some(capability) = capability else {
                continue;
            };
            if !matches!(
                handles[executor_id],
                Some(DirectExecutorHandle::Contribution { .. })
            ) {
                return Err(RusticolError::integrity(format!(
                    "direct fused fanout capability {executor_id} is not a loaded contribution executor"
                )));
            }
            let metadata = contribution_metadata[executor_id].ok_or_else(|| {
                RusticolError::integrity(format!(
                    "direct fused fanout capability {executor_id} has no contribution metadata"
                ))
            })?;
            if metadata.exact_factor_is_kernel_input
                || metadata.destination_component_count != capability.destination_component_count
            {
                return Err(RusticolError::integrity(format!(
                    "direct fused fanout capability {executor_id} is incompatible with its contribution metadata"
                )));
            }
        }
        if direct_template_catalog_digest != plan.direct_template_catalog_digest() {
            return Err(RusticolError::integrity(format!(
                "loaded direct-template catalog digest {direct_template_catalog_digest} does not match plan {}",
                plan.direct_template_catalog_digest()
            )));
        }
        let catalog = Self {
            handles: handles.into_boxed_slice(),
            contribution_metadata: contribution_metadata.into_boxed_slice(),
            contribution_fanout: contribution_fanout.into_boxed_slice(),
            packed_singleton_capable: false,
            plan_layout_digest: plan.runtime_layout_digest(),
            direct_template_catalog_digest,
        };
        catalog.validate_for_plan(plan)?;
        Ok(catalog)
    }

    pub(crate) const fn packed_singleton_capable(&self) -> bool {
        self.packed_singleton_capable
    }

    pub(crate) fn mark_packed_singleton_capable(mut self) -> Self {
        self.packed_singleton_capable = true;
        self
    }

    fn contribution_metadata(
        &self,
        executor_id: u32,
    ) -> RusticolResult<DirectContributionExecutionMetadata> {
        self.contribution_metadata
            .get(executor_id as usize)
            .and_then(|metadata| *metadata)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "direct contribution executor {executor_id} has no authenticated execution metadata"
                ))
            })
    }

    fn contribution_fanout(
        &self,
        executor_id: u32,
    ) -> Option<DirectContributionFanoutExecutorHandle> {
        self.contribution_fanout
            .get(executor_id as usize)
            .and_then(|capability| *capability)
    }

    fn require(
        &self,
        executor_id: u32,
        role: DirectExecutorRole,
    ) -> RusticolResult<DirectExecutorHandle> {
        let handle = self
            .handles
            .get(executor_id as usize)
            .and_then(|handle| *handle)
            .ok_or_else(|| {
                RusticolError::evaluation(format!(
                    "direct recurrence executor {executor_id} is not loaded"
                ))
            })?;
        if handle.role() != role {
            return Err(RusticolError::integrity(format!(
                "direct recurrence executor {executor_id} has role {:?}, expected {role:?}",
                handle.role()
            )));
        }
        Ok(handle)
    }

    fn validate_for_plan(&self, plan: &DirectRecurrencePlan) -> RusticolResult<()> {
        if self.handles.len() != plan.direct_executor_count() as usize {
            return Err(RusticolError::integrity(format!(
                "direct recurrence executor catalog has {} entries, expected {}",
                self.handles.len(),
                plan.direct_executor_count()
            )));
        }
        if self.contribution_metadata.len() != self.handles.len() {
            return Err(RusticolError::integrity(
                "direct contribution metadata has the wrong executor domain",
            ));
        }
        if self.contribution_fanout.len() != self.handles.len() {
            return Err(RusticolError::integrity(
                "direct contribution fanout capabilities have the wrong executor domain",
            ));
        }
        if self.direct_template_catalog_digest != plan.direct_template_catalog_digest() {
            return Err(RusticolError::integrity(
                "direct recurrence executor catalog belongs to a different prepared template catalog",
            ));
        }
        for descriptor in plan.row_groups() {
            if descriptor.role == DirectExecutorRole::Source
                && descriptor.direct_executor_id == DIRECT_NONE_U32
                && plan.strategy() == RecurrenceStrategy::AllFlowUnion
            {
                continue;
            }
            if descriptor.role == DirectExecutorRole::Contribution
                && descriptor.direct_executor_id == DIRECT_NONE_U32
            {
                continue;
            }
            self.require(descriptor.direct_executor_id, descriptor.role)?;
            if descriptor.role == DirectExecutorRole::Contribution {
                self.contribution_metadata(descriptor.direct_executor_id)?;
            }
        }
        Ok(())
    }
}

fn conservative_contribution_metadata(
    plan: &DirectRecurrencePlan,
    executor_count: usize,
) -> RusticolResult<Vec<Option<DirectContributionExecutionMetadata>>> {
    let current_components = plan
        .currents()
        .iter()
        .filter(|current| current.node_kind == DirectNodeKind::Current)
        .map(|current| {
            (
                (current.stage, current.component_base),
                u32::from(current.component_count),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut metadata = vec![None; executor_count];
    for descriptor in plan.row_groups().iter().filter(|descriptor| {
        descriptor.role == DirectExecutorRole::Contribution
            && descriptor.direct_executor_id != DIRECT_NONE_U32
    }) {
        let start = usize::try_from(descriptor.row_start).map_err(|_| {
            RusticolError::integrity("direct contribution row-group start exceeds usize")
        })?;
        let end = start
            .checked_add(descriptor.row_count as usize)
            .ok_or_else(|| {
                RusticolError::integrity("direct contribution row-group range overflows usize")
            })?;
        let rows = plan.contributions().get(start..end).ok_or_else(|| {
            RusticolError::integrity("direct contribution row-group range is out of bounds")
        })?;
        for row in rows {
            let component_count = current_components
                .get(&(descriptor.stage, row.destination_component_base))
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity("direct contribution destination current is absent")
                })?;
            let candidate = DirectContributionExecutionMetadata::new(component_count, true)?;
            let slot = metadata
                .get_mut(descriptor.direct_executor_id as usize)
                .ok_or_else(|| {
                    RusticolError::integrity("direct contribution executor is outside its catalog")
                })?;
            if slot.is_some_and(|existing| existing != candidate) {
                return Err(RusticolError::integrity(format!(
                    "direct contribution executor {} has inconsistent destination shapes",
                    descriptor.direct_executor_id
                )));
            }
            *slot = Some(candidate);
        }
    }
    Ok(metadata)
}

const DIRECT_CONTRIBUTION_FANOUT_BATCH_CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectContributionFanoutRun {
    pub(crate) row_start: u64,
    pub(crate) row_count: u32,
    pub(crate) scratch_component_base: u32,
    pub(crate) bundle_start: u64,
    pub(crate) bundle_count: u32,
}

/// Maximal consecutive fanout outputs which share their effective scale.
///
/// `DIRECT_NONE_U32` denotes unity because factor-consuming executors already
/// applied the authenticated row factor while producing the representative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectContributionFanoutBundle {
    pub(crate) row_start: u64,
    pub(crate) row_count: u32,
    pub(crate) effective_factor_id_or_sentinel: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct DirectContributionFanoutBatch<'a> {
    pub(crate) rows: &'a [DirectContributionRow],
    pub(crate) runs: &'a [DirectContributionFanoutRun],
    pub(crate) bundles: &'a [DirectContributionFanoutBundle],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DirectInteractionVectorParent {
    Parent0,
    Parent1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectInteractionOperand {
    pub(crate) tensor_component_base: u32,
    pub(crate) vector_parent: DirectInteractionVectorParent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DirectInteractionOperands {
    None,
    One(DirectInteractionOperand),
    Two([DirectInteractionOperand; 2]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectInteractionAnchor {
    pub(crate) parent_component_bases: [u32; 2],
    pub(crate) parent_momentum_form_ids: [u32; 2],
    pub(crate) operands: DirectInteractionOperands,
    pub(crate) output_start: u32,
    pub(crate) output_end: u32,
    pub(crate) tensor_output_start: u32,
    pub(crate) tensor_output_end: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectInteractionOutput {
    pub(crate) destination_component_base: u32,
    /// Color, followed by the zero, one, or two tensor-vector factors in the
    /// cold-authenticated contribution order. Unused tail entries are ignored.
    pub(crate) exact_factor_ids: [u32; 3],
    pub(crate) initialize: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectInteractionTensorOutput {
    pub(crate) destination_component_base: u32,
    pub(crate) exact_factor_id: u32,
    pub(crate) initialize: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectInteractionTerminalClosure {
    pub(crate) root_component_base: u32,
    pub(crate) amplitude_destination_id: u32,
    pub(crate) exact_factor_id: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct DirectInteractionTerminalBatch<'a> {
    pub(crate) partner_component_base: u32,
    pub(crate) component_factor_start: u32,
    pub(crate) amplitude_destination_count: u32,
    pub(crate) closures: &'a [DirectInteractionTerminalClosure],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DirectInteractionRuntimeRequirements {
    pub(crate) current_component_count: u32,
    pub(crate) momentum_form_count: u32,
    pub(crate) parameter_count: u32,
    pub(crate) factor_count: u32,
}

impl DirectInteractionRuntimeRequirements {
    fn include_current_span(&mut self, base: u32, component_count: u32) -> Option<()> {
        self.current_component_count = self
            .current_component_count
            .max(base.checked_add(component_count)?);
        Some(())
    }

    fn include_momentum(&mut self, momentum_form_id: u32) -> Option<()> {
        self.momentum_form_count = self
            .momentum_form_count
            .max(momentum_form_id.checked_add(1)?);
        Some(())
    }

    fn include_factor(&mut self, exact_factor_id: u32) -> Option<()> {
        self.factor_count = self.factor_count.max(exact_factor_id.checked_add(1)?);
        Some(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DirectInteractionBundleBatch<'a> {
    pub(crate) anchors: &'a [DirectInteractionAnchor],
    pub(crate) outputs: &'a [DirectInteractionOutput],
    pub(crate) tensor_outputs: &'a [DirectInteractionTensorOutput],
    pub(crate) terminal: Option<DirectInteractionTerminalBatch<'a>>,
    pub(crate) requirements: DirectInteractionRuntimeRequirements,
    pub(crate) atv_before_color: bool,
}

/// Authenticated native fanout capability for one contribution executor.
///
/// Only intrinsic executors expose this metadata/handle pair. The generic
/// fanout schedule uses the fused handle directly; the interaction schedule
/// additionally requires its optional interaction call. Prepared SymJIT/native
/// calls and nonmatching model/process combinations retain the ordinary row
/// executor path.
#[derive(Clone, Copy)]
pub(crate) struct DirectInteractionExecutorCapability {
    metadata: DirectContributionExecutionMetadata,
    fanout: DirectContributionFanoutExecutorHandle,
}

impl DirectInteractionExecutorCapability {
    pub(crate) fn new(
        handle: DirectExecutorHandle,
        metadata: DirectContributionExecutionMetadata,
        fanout: DirectContributionFanoutExecutorHandle,
    ) -> RusticolResult<Self> {
        if handle.role() != DirectExecutorRole::Contribution {
            return Err(RusticolError::integrity(
                "direct interaction capability is attached to a non-contribution executor",
            ));
        }
        if metadata.exact_factor_is_kernel_input
            || metadata.destination_component_count != fanout.destination_component_count
        {
            return Err(RusticolError::integrity(
                "direct interaction capability conflicts with authenticated contribution metadata",
            ));
        }
        Ok(Self { metadata, fanout })
    }
}

#[cfg(test)]
mod interaction_capability_tests {
    use super::*;

    unsafe extern "C" fn contribution_stub(
        _context: *const c_void,
        _arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        _factors: DirectFactorView,
        _rows: *const DirectContributionRow,
        _row_count: u32,
        _point_count: u32,
    ) -> c_int {
        DIRECT_STATUS_OK
    }

    unsafe extern "C" fn source_stub(
        _context: *const c_void,
        _arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        _factors: DirectFactorView,
        _rows: *const DirectSourceRow,
        _row_count: u32,
        _point_count: u32,
    ) -> c_int {
        DIRECT_STATUS_OK
    }

    unsafe fn fanout_stub(
        _context: *const c_void,
        _arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        _factors: DirectFactorView,
        _batch: DirectContributionFanoutBatch<'_>,
    ) -> RusticolResult<()> {
        Ok(())
    }

    fn fanout(destination_component_count: u32) -> DirectContributionFanoutExecutorHandle {
        DirectContributionFanoutExecutorHandle {
            call: fanout_stub,
            context: std::ptr::null(),
            destination_component_count,
            parent_component_counts: [4, 4],
            requires_two_momenta: true,
            required_parameter_count: 0,
            interaction_kind: DirectInteractionIntrinsicKind::ColorOrderedThreeVector,
            interaction_call: None,
        }
    }

    #[test]
    fn interaction_capability_requires_authenticated_output_only_contribution_shape() {
        let contribution = DirectExecutorHandle::Contribution {
            call: contribution_stub,
            context: std::ptr::null(),
        };
        let output_only = DirectContributionExecutionMetadata::new(4, false).unwrap();
        assert!(
            DirectInteractionExecutorCapability::new(contribution, output_only, fanout(4)).is_ok()
        );
        assert!(
            DirectInteractionExecutorCapability::new(
                contribution,
                DirectContributionExecutionMetadata::new(4, true).unwrap(),
                fanout(4),
            )
            .is_err()
        );
        assert!(
            DirectInteractionExecutorCapability::new(contribution, output_only, fanout(6)).is_err()
        );
        assert!(
            DirectInteractionExecutorCapability::new(
                DirectExecutorHandle::Source {
                    call: source_stub,
                    context: std::ptr::null(),
                },
                output_only,
                fanout(4),
            )
            .is_err()
        );
    }
}

#[derive(Clone, Copy)]
enum DirectInteractionRowsView<'a> {
    Contribution(&'a [DirectContributionRow]),
    Closure(&'a [DirectClosureRow]),
    Other,
}

/// Zero-copy view of one authenticated schedule group.
#[derive(Clone, Copy)]
pub(crate) struct DirectInteractionGroupView<'a> {
    stage: u32,
    role: DirectExecutorRole,
    direct_executor_id: u32,
    handle: Option<DirectExecutorHandle>,
    capability: Option<DirectInteractionExecutorCapability>,
    rows: DirectInteractionRowsView<'a>,
}

impl<'a> DirectInteractionGroupView<'a> {
    pub(crate) const fn contribution(
        stage: u32,
        direct_executor_id: u32,
        handle: Option<DirectExecutorHandle>,
        capability: Option<DirectInteractionExecutorCapability>,
        rows: &'a [DirectContributionRow],
    ) -> Self {
        Self {
            stage,
            role: DirectExecutorRole::Contribution,
            direct_executor_id,
            handle,
            capability,
            rows: DirectInteractionRowsView::Contribution(rows),
        }
    }

    pub(crate) const fn closure(
        stage: u32,
        direct_executor_id: u32,
        handle: DirectExecutorHandle,
        rows: &'a [DirectClosureRow],
    ) -> Self {
        Self {
            stage,
            role: DirectExecutorRole::Closure,
            direct_executor_id,
            handle: Some(handle),
            capability: None,
            rows: DirectInteractionRowsView::Closure(rows),
        }
    }

    pub(crate) const fn other(
        stage: u32,
        role: DirectExecutorRole,
        direct_executor_id: u32,
        handle: Option<DirectExecutorHandle>,
    ) -> Self {
        Self {
            stage,
            role,
            direct_executor_id,
            handle,
            capability: None,
            rows: DirectInteractionRowsView::Other,
        }
    }

    fn contributions(self) -> Option<&'a [DirectContributionRow]> {
        match self.rows {
            DirectInteractionRowsView::Contribution(rows) => Some(rows),
            DirectInteractionRowsView::Closure(_) | DirectInteractionRowsView::Other => None,
        }
    }

    fn closures(self) -> Option<&'a [DirectClosureRow]> {
        match self.rows {
            DirectInteractionRowsView::Closure(rows) => Some(rows),
            DirectInteractionRowsView::Contribution(_) | DirectInteractionRowsView::Other => None,
        }
    }
}

/// Borrowed authenticated inputs needed by the native interaction compiler.
/// Row storage remains owned by the direct plan or prepared OTF family.
pub(crate) struct DirectInteractionScheduleView<'a> {
    strategy: RecurrenceStrategy,
    groups: &'a [DirectInteractionGroupView<'a>],
    currents: &'a [DirectCurrentDescriptor],
    contribution_row_count: usize,
}

impl<'a> DirectInteractionScheduleView<'a> {
    pub(crate) const fn new(
        strategy: RecurrenceStrategy,
        groups: &'a [DirectInteractionGroupView<'a>],
        currents: &'a [DirectCurrentDescriptor],
        contribution_row_count: usize,
    ) -> Self {
        Self {
            strategy,
            groups,
            currents,
            contribution_row_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectContributionFanoutStep {
    Direct {
        row_start: u64,
        row_count: u32,
    },
    Fanout {
        row_start: u64,
        row_count: u32,
        representative_start: u64,
        representative_count: u32,
        run_start: u64,
        run_count: u32,
        destination_component_count: u32,
    },
}

impl DirectContributionFanoutStep {
    const fn row_start(self) -> u64 {
        match self {
            Self::Direct { row_start, .. } | Self::Fanout { row_start, .. } => row_start,
        }
    }

    const fn row_count(self) -> u32 {
        match self {
            Self::Direct { row_count, .. } | Self::Fanout { row_count, .. } => row_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectContributionFanoutGroup {
    stage: u16,
    direct_executor_id: u32,
    row_start: u64,
    row_count: u32,
    step_start: u64,
    step_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectSingletonFanoutStep {
    Direct {
        row_start: u32,
        row_count: u32,
    },
    Fanout {
        row_start: u32,
        row_count: u32,
        run_start: u64,
        run_count: u32,
    },
}

impl DirectSingletonFanoutStep {
    const fn row_start(self) -> u32 {
        match self {
            Self::Direct { row_start, .. } | Self::Fanout { row_start, .. } => row_start,
        }
    }

    const fn row_count(self) -> u32 {
        match self {
            Self::Direct { row_count, .. } | Self::Fanout { row_count, .. } => row_count,
        }
    }
}

#[derive(Clone, Copy)]
struct DirectSingletonFanoutGroup {
    direct_executor_id: u32,
    row_count: u32,
    step_start: u64,
    step_count: u32,
    raw: DirectExecutorHandle,
    fused: DirectContributionFanoutExecutorHandle,
}

/// Cold family-local native fanout schedule for singleton-point execution.
///
/// Unlike [`DirectContributionFanoutProgram`], this needs no private arena
/// suffix: authenticated intrinsic fanout executors evaluate one input class
/// and write every destination directly.  Batched point execution remains on
/// the ordinary row executor path.
pub(crate) struct DirectSingletonContributionFanoutProgram {
    groups: Box<[Option<DirectSingletonFanoutGroup>]>,
    steps: Box<[DirectSingletonFanoutStep]>,
    runs: Box<[DirectContributionFanoutRun]>,
    bundles: Box<[DirectContributionFanoutBundle]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DirectInteractionGroupAction {
    Normal,
    Execute(u32),
    Consumed,
}

struct DirectInteractionStage {
    group_indices: Box<[u32]>,
    closure_group_index: Option<u32>,
    call: DirectInteractionBundleExecutor,
    contexts: DirectInteractionExecutorContexts,
    atv_before_color: bool,
    anchors: Box<[DirectInteractionAnchor]>,
    outputs: Box<[DirectInteractionOutput]>,
    tensor_outputs: Box<[DirectInteractionTensorOutput]>,
    terminal: Option<DirectInteractionTerminalProgram>,
    requirements: DirectInteractionRuntimeRequirements,
    logical_row_count: u64,
}

struct DirectInteractionTerminalProgram {
    partner_component_base: u32,
    component_factor_start: u32,
    amplitude_destination_count: u32,
    closures: Box<[DirectInteractionTerminalClosure]>,
}

/// Cold, schedule-local native interaction rewrite shared by persisted direct
/// plans and dynamically prepared on-the-fly families.
pub(crate) struct DirectInteractionProgram {
    actions: Box<[DirectInteractionGroupAction]>,
    stages: Box<[DirectInteractionStage]>,
}

/// Immutable, plan-local execution rewrite for contribution kinematics.
///
/// Rows remain the authoritative cold schedule. This program copies only the
/// compact input, destination, and live-factor metadata needed to evaluate
/// each authenticated input class once and replay its deterministic writes.
pub(crate) struct DirectContributionFanoutProgram {
    plan_layout_digest: SemanticDigest,
    groups: Box<[DirectContributionFanoutGroup]>,
    steps: Box<[DirectContributionFanoutStep]>,
    representatives: Box<[DirectContributionRow]>,
    runs: Box<[DirectContributionFanoutRun]>,
    bundles: Box<[DirectContributionFanoutBundle]>,
    scratch_component_base: u32,
    scratch_component_count: u32,
    unit_factor_id: u32,
    needs_unit_factor: bool,
    interactions: DirectInteractionProgram,
    full_singleton_steps: Box<[DirectFullSingletonStep]>,
}

/// Cold-authenticated dispatch for the common full, unselected singleton.
///
/// The persisted plan remains authoritative. These steps retain only handles
/// and already-validated row offsets, avoiding a warmed catalog lookup and
/// row-range reconstruction for every small group. Selected, observed, and
/// multi-point execution continue through the generic schedule.
#[derive(Clone, Copy)]
enum DirectFullSingletonStep {
    Interaction {
        stage_index: u32,
    },
    Call {
        direct_executor_id: u32,
        handle: DirectExecutorHandle,
        row_start: usize,
        row_count: u32,
    },
    CertifiedReuse {
        row_start: usize,
        row_count: u32,
    },
    Fanout {
        group_index: u32,
        call: DirectContributionExecutor,
        context: *const c_void,
        fused: Option<DirectContributionFanoutExecutorHandle>,
    },
}

impl DirectContributionFanoutProgram {
    pub(crate) fn build(
        plan: &DirectRecurrencePlan,
        executors: &DirectExecutorCatalog,
    ) -> RusticolResult<Self> {
        if executors.plan_layout_digest != plan.runtime_layout_digest() {
            return Err(RusticolError::integrity(
                "direct fanout executor catalog belongs to a different plan",
            ));
        }
        let scratch_component_base = plan.current_arena_components();
        let unit_factor_id = u32::try_from(plan.exact_factors().len())
            .map_err(|_| RusticolError::integrity("direct exact-factor catalog exceeds u32"))?;
        let mut groups = Vec::new();
        let mut steps = Vec::new();
        let mut representatives = Vec::new();
        let mut runs = Vec::new();
        let mut bundles = Vec::new();
        let mut scratch_component_count = 0_u32;
        let mut needs_unit_factor = false;
        let destination_component_counts = plan
            .currents()
            .iter()
            .filter(|current| current.node_kind == DirectNodeKind::Current)
            .map(|current| {
                (
                    (current.stage, current.component_base),
                    u32::from(current.component_count),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let arena_component_count = plan.current_arena_components();
        let momentum_form_count = u32::try_from(plan.momentum_forms().len())
            .map_err(|_| RusticolError::integrity("direct momentum-form count exceeds u32"))?;

        for descriptor in plan.row_groups().iter().filter(|descriptor| {
            descriptor.role == DirectExecutorRole::Contribution
                && descriptor.direct_executor_id != DIRECT_NONE_U32
        }) {
            let metadata = executors.contribution_metadata(descriptor.direct_executor_id)?;
            let fused = executors.contribution_fanout(descriptor.direct_executor_id);
            let group_step_start = u64::try_from(steps.len())
                .map_err(|_| RusticolError::integrity("direct fanout step count exceeds u64"))?;
            let start = usize::try_from(descriptor.row_start).map_err(|_| {
                RusticolError::integrity("direct fanout row-group start exceeds usize")
            })?;
            let end = start
                .checked_add(descriptor.row_count as usize)
                .ok_or_else(|| {
                    RusticolError::integrity("direct fanout row-group range overflows usize")
                })?;
            let descriptor_rows = plan.contributions().get(start..end).ok_or_else(|| {
                RusticolError::integrity("direct fanout row-group range is out of bounds")
            })?;
            for row in descriptor_rows {
                let authenticated_component_count = destination_component_counts
                    .get(&(descriptor.stage, row.destination_component_base))
                    .copied()
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct fanout contribution destination current is absent",
                        )
                    })?;
                if metadata.destination_component_count != authenticated_component_count {
                    return Err(RusticolError::integrity(format!(
                        "direct contribution executor {} declares {} destination components, expected {authenticated_component_count}",
                        descriptor.direct_executor_id, metadata.destination_component_count
                    )));
                }
                if let Some(fused) = fused {
                    validate_fused_fanout_row(
                        *row,
                        fused.parent_component_counts,
                        fused.requires_two_momenta,
                        arena_component_count,
                        momentum_form_count,
                    )?;
                }
            }
            let mut selector_offset = 0_usize;
            while selector_offset < descriptor_rows.len() {
                let selector_domain_id = descriptor_rows[selector_offset].selector_domain_id;
                let selector_start = selector_offset;
                while selector_offset < descriptor_rows.len()
                    && descriptor_rows[selector_offset].selector_domain_id == selector_domain_id
                {
                    selector_offset += 1;
                }
                let selector_end = selector_offset;
                let segment_step_start = steps.len();
                let mut pending_runs = Vec::<(usize, usize)>::new();
                let mut offset = selector_start;
                while offset < selector_end {
                    let run_start = offset;
                    offset += 1;
                    while offset < selector_end
                        && contribution_fanout_inputs_equal(
                            descriptor_rows[run_start],
                            descriptor_rows[offset],
                            metadata.exact_factor_is_kernel_input,
                        )
                    {
                        offset += 1;
                    }
                    let run_count = offset - run_start;
                    if run_count == 1 {
                        flush_pending_fanout_batch(
                            plan,
                            descriptor,
                            metadata,
                            scratch_component_base,
                            unit_factor_id,
                            &mut pending_runs,
                            &mut steps,
                            &mut representatives,
                            &mut runs,
                            &mut bundles,
                            &mut scratch_component_count,
                            &mut needs_unit_factor,
                        )?;
                        push_direct_fanout_step(
                            &mut steps,
                            segment_step_start,
                            descriptor.row_start + run_start as u64,
                            1,
                        )?;
                    } else {
                        pending_runs.push((run_start, run_count));
                        if pending_runs.len() == DIRECT_CONTRIBUTION_FANOUT_BATCH_CAPACITY {
                            flush_pending_fanout_batch(
                                plan,
                                descriptor,
                                metadata,
                                scratch_component_base,
                                unit_factor_id,
                                &mut pending_runs,
                                &mut steps,
                                &mut representatives,
                                &mut runs,
                                &mut bundles,
                                &mut scratch_component_count,
                                &mut needs_unit_factor,
                            )?;
                        }
                    }
                }
                flush_pending_fanout_batch(
                    plan,
                    descriptor,
                    metadata,
                    scratch_component_base,
                    unit_factor_id,
                    &mut pending_runs,
                    &mut steps,
                    &mut representatives,
                    &mut runs,
                    &mut bundles,
                    &mut scratch_component_count,
                    &mut needs_unit_factor,
                )?;
            }
            groups.push(DirectContributionFanoutGroup {
                stage: descriptor.stage,
                direct_executor_id: descriptor.direct_executor_id,
                row_start: descriptor.row_start,
                row_count: descriptor.row_count,
                step_start: group_step_start,
                step_count: u32::try_from(steps.len() as u64 - group_step_start).map_err(|_| {
                    RusticolError::integrity("direct fanout group step count exceeds u32")
                })?,
            });
        }
        let interactions = DirectInteractionProgram::build_for_direct_plan(plan, executors)?;
        let mut program = Self {
            plan_layout_digest: plan.runtime_layout_digest(),
            groups: groups.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            representatives: representatives.into_boxed_slice(),
            runs: runs.into_boxed_slice(),
            bundles: bundles.into_boxed_slice(),
            scratch_component_base,
            scratch_component_count,
            unit_factor_id,
            needs_unit_factor,
            interactions,
            full_singleton_steps: Box::new([]),
        };
        program.full_singleton_steps = program.compile_full_singleton_steps(plan, executors)?;
        Ok(program)
    }

    fn compile_full_singleton_steps(
        &self,
        plan: &DirectRecurrencePlan,
        executors: &DirectExecutorCatalog,
    ) -> RusticolResult<Box<[DirectFullSingletonStep]>> {
        let mut compiled = Vec::with_capacity(plan.row_groups().len());
        for (group_index, descriptor) in plan.row_groups().iter().enumerate() {
            match self.interactions.action(group_index) {
                DirectInteractionGroupAction::Consumed => continue,
                DirectInteractionGroupAction::Execute(stage_index) => {
                    compiled.push(DirectFullSingletonStep::Interaction { stage_index });
                    continue;
                }
                DirectInteractionGroupAction::Normal => {}
            }
            if descriptor.role == DirectExecutorRole::Source
                && descriptor.direct_executor_id == DIRECT_NONE_U32
                && plan.strategy() == RecurrenceStrategy::AllFlowUnion
            {
                continue;
            }
            let row_start = usize::try_from(descriptor.row_start).map_err(|_| {
                RusticolError::integrity("flattened direct row-group start exceeds usize")
            })?;
            let row_end = row_start
                .checked_add(descriptor.row_count as usize)
                .ok_or_else(|| {
                    RusticolError::integrity("flattened direct row-group range overflows usize")
                })?;
            let authoritative_len = match descriptor.role {
                DirectExecutorRole::Source => plan.sources().len(),
                DirectExecutorRole::Contribution => plan.contributions().len(),
                DirectExecutorRole::Finalization => plan.finalizations().len(),
                DirectExecutorRole::Closure => plan.closures().len(),
            };
            if row_end > authoritative_len {
                return Err(RusticolError::integrity(
                    "flattened direct row-group range is out of bounds",
                ));
            }
            if descriptor.role == DirectExecutorRole::Contribution
                && descriptor.direct_executor_id == DIRECT_NONE_U32
            {
                compiled.push(DirectFullSingletonStep::CertifiedReuse {
                    row_start,
                    row_count: descriptor.row_count,
                });
                continue;
            }
            let handle = executors.require(descriptor.direct_executor_id, descriptor.role)?;
            if descriptor.role == DirectExecutorRole::Contribution {
                let steps = self.group_steps(descriptor)?;
                if steps
                    .iter()
                    .all(|step| matches!(step, DirectContributionFanoutStep::Direct { .. }))
                {
                    compiled.push(DirectFullSingletonStep::Call {
                        direct_executor_id: descriptor.direct_executor_id,
                        handle,
                        row_start,
                        row_count: descriptor.row_count,
                    });
                    continue;
                }
                let DirectExecutorHandle::Contribution { call, context } = handle else {
                    return Err(RusticolError::integrity(
                        "flattened direct contribution handle changed role",
                    ));
                };
                compiled.push(DirectFullSingletonStep::Fanout {
                    group_index: u32::try_from(group_index).map_err(|_| {
                        RusticolError::integrity("flattened direct group index exceeds u32")
                    })?,
                    call,
                    context,
                    fused: executors.contribution_fanout(descriptor.direct_executor_id),
                });
                continue;
            }
            compiled.push(DirectFullSingletonStep::Call {
                direct_executor_id: descriptor.direct_executor_id,
                handle,
                row_start,
                row_count: descriptor.row_count,
            });
        }
        Ok(compiled.into_boxed_slice())
    }

    fn execute_full_singleton_unprofiled(
        &self,
        plan: &DirectRecurrencePlan,
        executors: &DirectExecutorCatalog,
        workspace: &mut DirectWorkspace<'_>,
    ) -> RusticolResult<()> {
        clear_direct_executor_error_detail();
        workspace.validate(1)?;
        if executors.plan_layout_digest != plan.runtime_layout_digest()
            || self.plan_layout_digest != plan.runtime_layout_digest()
        {
            return Err(RusticolError::integrity(
                "flattened direct execution belongs to a different plan",
            ));
        }
        let (arena, momenta, parameters, factors) = workspace.raw_views()?;
        for step in &self.full_singleton_steps {
            match *step {
                DirectFullSingletonStep::Interaction { stage_index } => {
                    self.interactions.execute::<false>(
                        stage_index,
                        arena,
                        momenta,
                        parameters,
                        factors,
                    )?;
                }
                DirectFullSingletonStep::Call {
                    direct_executor_id,
                    handle,
                    row_start,
                    row_count,
                } => {
                    let status = unsafe {
                        match handle {
                            DirectExecutorHandle::Source { call, context } => call(
                                context,
                                arena,
                                momenta,
                                parameters,
                                factors,
                                plan.sources().as_ptr().add(row_start),
                                row_count,
                                1,
                            ),
                            DirectExecutorHandle::Contribution { call, context } => call(
                                context,
                                arena,
                                momenta,
                                parameters,
                                factors,
                                plan.contributions().as_ptr().add(row_start),
                                row_count,
                                1,
                            ),
                            DirectExecutorHandle::Finalization { call, context } => call(
                                context,
                                arena,
                                momenta,
                                parameters,
                                factors,
                                plan.finalizations().as_ptr().add(row_start),
                                row_count,
                                1,
                            ),
                            DirectExecutorHandle::Closure { call, context } => call(
                                context,
                                arena,
                                momenta,
                                parameters,
                                factors,
                                plan.closures().as_ptr().add(row_start),
                                row_count,
                                1,
                            ),
                        }
                    };
                    check_status(handle.role(), direct_executor_id, status)?;
                }
                DirectFullSingletonStep::CertifiedReuse {
                    row_start,
                    row_count,
                } => {
                    let row_end = row_start + row_count as usize;
                    execute_certified_reuse_rows(
                        &plan.contributions()[row_start..row_end],
                        workspace,
                        1,
                    )?;
                }
                DirectFullSingletonStep::Fanout {
                    group_index,
                    call,
                    context,
                    fused,
                } => {
                    let descriptor = &plan.row_groups()[group_index as usize];
                    execute_contribution_group_with_fanout(
                        plan, descriptor, self, call, context, fused, workspace, 1,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(crate) const fn scratch_component_count(&self) -> u32 {
        self.scratch_component_count
    }

    pub(crate) const fn needs_unit_factor(&self) -> bool {
        self.needs_unit_factor
    }

    #[cfg(test)]
    pub(crate) fn row_counts(&self) -> (u64, u64) {
        let logical = self
            .groups
            .iter()
            .map(|group| u64::from(group.row_count))
            .sum();
        let evaluated = self
            .steps
            .iter()
            .map(|step| match step {
                DirectContributionFanoutStep::Direct { row_count, .. } => u64::from(*row_count),
                DirectContributionFanoutStep::Fanout {
                    representative_count,
                    ..
                } => u64::from(*representative_count),
            })
            .sum();
        (logical, evaluated)
    }

    #[cfg(test)]
    pub(crate) fn scatter_bundle_counts(&self) -> (u64, u64) {
        (
            self.runs.iter().map(|run| u64::from(run.row_count)).sum(),
            self.bundles.len() as u64,
        )
    }

    fn group_steps(
        &self,
        descriptor: &DirectRowGroupDescriptor,
    ) -> RusticolResult<&[DirectContributionFanoutStep]> {
        if descriptor.role != DirectExecutorRole::Contribution
            || descriptor.direct_executor_id == DIRECT_NONE_U32
        {
            return Err(RusticolError::integrity(
                "direct fanout requested for a non-callable contribution group",
            ));
        }
        let requested_end = descriptor
            .row_start
            .checked_add(u64::from(descriptor.row_count))
            .ok_or_else(|| RusticolError::integrity("direct fanout row range overflows u64"))?;
        // Groups retain the authenticated row-group order, so locate the one
        // containing a selected subrange without a warmed linear catalog scan.
        let group_index = self
            .groups
            .partition_point(|group| group.row_start <= descriptor.row_start)
            .checked_sub(1)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct contribution row group is absent from its fanout program",
                )
            })?;
        let group = self
            .groups
            .get(group_index)
            .filter(|group| {
                let group_end = group.row_start + u64::from(group.row_count);
                group.stage == descriptor.stage
                    && group.direct_executor_id == descriptor.direct_executor_id
                    && descriptor.row_start >= group.row_start
                    && requested_end <= group_end
            })
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct contribution row group is absent from its fanout program",
                )
            })?;
        let step_start = usize::try_from(group.step_start)
            .map_err(|_| RusticolError::integrity("direct fanout step start exceeds usize"))?;
        let step_end = step_start
            .checked_add(group.step_count as usize)
            .ok_or_else(|| RusticolError::integrity("direct fanout step range overflows usize"))?;
        let group_steps = self.steps.get(step_start..step_end).ok_or_else(|| {
            RusticolError::integrity("direct fanout group step range is out of bounds")
        })?;
        let first = group_steps
            .iter()
            .position(|step| step.row_start() == descriptor.row_start)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "selected direct fanout range does not begin on a step boundary",
                )
            })?;
        let mut stop = first;
        let mut covered = descriptor.row_start;
        while stop < group_steps.len() && covered < requested_end {
            let step = group_steps[stop];
            if step.row_start() != covered {
                return Err(RusticolError::integrity(
                    "direct fanout steps do not continuously cover their row group",
                ));
            }
            covered = covered
                .checked_add(u64::from(step.row_count()))
                .ok_or_else(|| {
                    RusticolError::integrity("direct fanout step coverage overflows u64")
                })?;
            stop += 1;
        }
        if covered != requested_end {
            return Err(RusticolError::integrity(
                "selected direct fanout range does not end on a step boundary",
            ));
        }
        Ok(&group_steps[first..stop])
    }
}

impl DirectSingletonContributionFanoutProgram {
    pub(crate) fn build(
        schedule_groups: &[DirectInteractionGroupView<'_>],
        arena_component_count: u32,
        momentum_form_count: u32,
        parameter_count: u32,
        factor_count: u32,
    ) -> RusticolResult<Self> {
        let mut groups = vec![None; schedule_groups.len()];
        let mut steps = Vec::new();
        let mut runs = Vec::new();
        let mut bundles = Vec::new();
        for (group_index, group) in schedule_groups.iter().copied().enumerate() {
            let Some(rows) = group.contributions() else {
                continue;
            };
            let Some(capability) = group.capability else {
                continue;
            };
            let fanout = capability.fanout;
            if rows.is_empty()
                || rows.len() > u32::MAX as usize
                || fanout.required_parameter_count > parameter_count
                || rows.iter().any(|row| {
                    row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0
                        || row.exact_factor_id >= factor_count
                        || row
                            .destination_component_base
                            .checked_add(fanout.destination_component_count)
                            .is_none_or(|end| end > arena_component_count)
                })
            {
                continue;
            }
            for row in rows {
                validate_fused_fanout_row(
                    *row,
                    fanout.parent_component_counts,
                    fanout.requires_two_momenta,
                    arena_component_count,
                    momentum_form_count,
                )?;
            }

            let step_start = u64::try_from(steps.len())
                .map_err(|_| RusticolError::integrity("singleton fanout step count exceeds u64"))?;
            let mut pending_runs = Vec::<(u32, u32)>::new();
            let mut offset = 0_usize;
            while offset < rows.len() {
                let run_start = offset;
                offset += 1;
                while offset < rows.len()
                    && contribution_fanout_inputs_equal(rows[run_start], rows[offset], false)
                {
                    offset += 1;
                }
                let run_count = offset - run_start;
                if run_count == 1 {
                    flush_singleton_fanout_batch(
                        rows,
                        &mut pending_runs,
                        &mut steps,
                        &mut runs,
                        &mut bundles,
                    )?;
                    push_singleton_direct_step(
                        &mut steps,
                        step_start as usize,
                        u32::try_from(run_start).map_err(|_| {
                            RusticolError::integrity(
                                "singleton fanout direct-row start exceeds u32",
                            )
                        })?,
                        1,
                    )?;
                } else {
                    pending_runs.push((
                        u32::try_from(run_start).map_err(|_| {
                            RusticolError::integrity("singleton fanout run start exceeds u32")
                        })?,
                        u32::try_from(run_count).map_err(|_| {
                            RusticolError::integrity("singleton fanout run count exceeds u32")
                        })?,
                    ));
                    if pending_runs.len() == DIRECT_CONTRIBUTION_FANOUT_BATCH_CAPACITY {
                        flush_singleton_fanout_batch(
                            rows,
                            &mut pending_runs,
                            &mut steps,
                            &mut runs,
                            &mut bundles,
                        )?;
                    }
                }
            }
            flush_singleton_fanout_batch(
                rows,
                &mut pending_runs,
                &mut steps,
                &mut runs,
                &mut bundles,
            )?;
            let step_count = u32::try_from(steps.len() as u64 - step_start).map_err(|_| {
                RusticolError::integrity("singleton fanout group step count exceeds u32")
            })?;
            let group_steps = &steps[step_start as usize..];
            if !group_steps
                .iter()
                .any(|step| matches!(step, DirectSingletonFanoutStep::Fanout { .. }))
            {
                steps.truncate(step_start as usize);
                continue;
            }
            let raw = group.handle.ok_or_else(|| {
                RusticolError::integrity("singleton fanout group has no raw contribution handle")
            })?;
            if raw.role() != DirectExecutorRole::Contribution {
                return Err(RusticolError::integrity(
                    "singleton fanout group has a non-contribution raw handle",
                ));
            }
            groups[group_index] = Some(DirectSingletonFanoutGroup {
                direct_executor_id: group.direct_executor_id,
                row_count: rows.len() as u32,
                step_start,
                step_count,
                raw,
                fused: fanout,
            });
        }
        Ok(Self {
            groups: groups.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            runs: runs.into_boxed_slice(),
            bundles: bundles.into_boxed_slice(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_group(
        &self,
        group_index: usize,
        rows: &[DirectContributionRow],
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        factors: DirectFactorView,
        point_count: u32,
    ) -> RusticolResult<bool> {
        if point_count != 1 {
            return Ok(false);
        }
        let Some(group) = self.groups.get(group_index).and_then(|group| *group) else {
            return Ok(false);
        };
        if rows.len() != group.row_count as usize {
            return Err(RusticolError::integrity(
                "singleton fanout rows changed after cold compilation",
            ));
        }
        let step_start = usize::try_from(group.step_start)
            .map_err(|_| RusticolError::integrity("singleton fanout step start exceeds usize"))?;
        let step_end = step_start
            .checked_add(group.step_count as usize)
            .ok_or_else(|| RusticolError::integrity("singleton fanout step range overflows"))?;
        let steps = self.steps.get(step_start..step_end).ok_or_else(|| {
            RusticolError::integrity("singleton fanout step range is out of bounds")
        })?;
        let DirectExecutorHandle::Contribution { call, context } = group.raw else {
            return Err(RusticolError::integrity(
                "singleton fanout raw handle changed role",
            ));
        };
        let mut covered = 0_u32;
        for step in steps {
            if step.row_start() != covered {
                return Err(RusticolError::integrity(
                    "singleton fanout steps do not continuously cover their row group",
                ));
            }
            match *step {
                DirectSingletonFanoutStep::Direct {
                    row_start,
                    row_count,
                } => {
                    let start = row_start as usize;
                    let end = start.checked_add(row_count as usize).ok_or_else(|| {
                        RusticolError::integrity("singleton fanout direct range overflows")
                    })?;
                    let direct_rows = rows.get(start..end).ok_or_else(|| {
                        RusticolError::integrity("singleton fanout direct range is out of bounds")
                    })?;
                    let status = unsafe {
                        call(
                            context,
                            arena,
                            momenta,
                            parameters,
                            factors,
                            direct_rows.as_ptr(),
                            row_count,
                            1,
                        )
                    };
                    check_status(
                        DirectExecutorRole::Contribution,
                        group.direct_executor_id,
                        status,
                    )?;
                }
                DirectSingletonFanoutStep::Fanout {
                    run_start,
                    run_count,
                    ..
                } => {
                    let start = usize::try_from(run_start).map_err(|_| {
                        RusticolError::integrity("singleton fanout run start exceeds usize")
                    })?;
                    let end = start.checked_add(run_count as usize).ok_or_else(|| {
                        RusticolError::integrity("singleton fanout run range overflows")
                    })?;
                    let runs = self.runs.get(start..end).ok_or_else(|| {
                        RusticolError::integrity("singleton fanout run range is out of bounds")
                    })?;
                    unsafe {
                        (group.fused.call)(
                            group.fused.context,
                            arena,
                            momenta,
                            parameters,
                            factors,
                            DirectContributionFanoutBatch {
                                rows,
                                runs,
                                bundles: &self.bundles,
                            },
                        )
                    }?;
                }
            }
            covered = covered.checked_add(step.row_count()).ok_or_else(|| {
                RusticolError::integrity("singleton fanout step coverage overflows u32")
            })?;
        }
        if covered != group.row_count {
            return Err(RusticolError::integrity(
                "singleton fanout steps do not cover their row group",
            ));
        }
        Ok(true)
    }

    #[cfg(test)]
    pub(crate) fn row_counts(&self) -> (u64, u64) {
        let logical = self
            .groups
            .iter()
            .flatten()
            .map(|group| u64::from(group.row_count))
            .sum();
        let evaluated = self
            .steps
            .iter()
            .map(|step| match step {
                DirectSingletonFanoutStep::Direct { row_count, .. } => u64::from(*row_count),
                DirectSingletonFanoutStep::Fanout { run_count, .. } => u64::from(*run_count),
            })
            .sum();
        (logical, evaluated)
    }
}

#[derive(Clone, Copy)]
struct DirectInteractionRawRun {
    row_start: u32,
    row_count: u32,
}

type DirectInteractionInputKey = (u32, u32, u32, u32, u32);

fn interaction_input_key(row: DirectContributionRow) -> DirectInteractionInputKey {
    (
        row.selector_domain_id,
        row.parent0_component_base,
        row.parent1_component_base_or_sentinel,
        row.parent0_momentum_form_id,
        row.parent1_momentum_form_id_or_sentinel,
    )
}

fn interaction_raw_runs(rows: &[DirectContributionRow]) -> Option<Vec<DirectInteractionRawRun>> {
    if rows
        .iter()
        .any(|row| row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0)
    {
        return None;
    }
    let mut runs = Vec::new();
    let mut offset = 0_usize;
    while offset < rows.len() {
        let run_start = offset;
        let key = interaction_input_key(rows[offset]);
        offset += 1;
        while offset < rows.len() && interaction_input_key(rows[offset]) == key {
            offset += 1;
        }
        runs.push(DirectInteractionRawRun {
            row_start: u32::try_from(run_start).ok()?,
            row_count: u32::try_from(offset - run_start).ok()?,
        });
    }
    Some(runs)
}

fn interaction_rows(
    rows: &[DirectContributionRow],
    run: DirectInteractionRawRun,
) -> Option<&[DirectContributionRow]> {
    let start = run.row_start as usize;
    rows.get(start..start.checked_add(run.row_count as usize)?)
}

fn interaction_destination_offsets(rows: &[DirectContributionRow]) -> Option<BTreeMap<u32, u32>> {
    let destinations = rows
        .iter()
        .enumerate()
        .map(|(offset, row)| Some((row.destination_component_base, u32::try_from(offset).ok()?)))
        .collect::<Option<BTreeMap<_, _>>>()?;
    (destinations.len() == rows.len()).then_some(destinations)
}

fn interaction_destination_bijection(
    color: &BTreeMap<u32, u32>,
    atv: &BTreeMap<u32, u32>,
) -> Option<Vec<u32>> {
    if color.len() != atv.len() || !color.keys().eq(atv.keys()) {
        return None;
    }
    let mut row_offsets = vec![DIRECT_NONE_U32; color.len()];
    for (&destination, &color_offset) in color {
        let atv_offset = *atv.get(&destination)?;
        *row_offsets.get_mut(color_offset as usize)? = atv_offset;
    }
    (!row_offsets.contains(&DIRECT_NONE_U32)).then_some(row_offsets)
}

fn interaction_vector_initialization_is_canonical(
    ordered_groups: &[&[DirectContributionRow]],
) -> bool {
    let mut seen = BTreeSet::new();
    ordered_groups
        .iter()
        .flat_map(|rows| rows.iter())
        .all(|row| {
            let first = seen.insert(row.destination_component_base);
            let initializes = row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0;
            initializes == first
        })
}

#[cfg(test)]
mod interaction_initialization_tests {
    use super::*;

    fn row(destination_component_base: u32, initializes: bool) -> DirectContributionRow {
        DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 4,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 1,
            destination_component_base,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: if initializes {
                DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
            } else {
                0
            },
        }
    }

    #[test]
    fn interaction_initialization_requires_the_first_authoritative_write() {
        let first_group = [row(20, true), row(24, true), row(20, false)];
        let second_group = [row(24, false)];
        assert!(interaction_vector_initialization_is_canonical(&[
            &first_group,
            &second_group,
        ]));

        let late_initializer = [row(20, false), row(20, true)];
        assert!(!interaction_vector_initialization_is_canonical(&[
            &late_initializer,
        ]));
        let duplicate_initializer = [row(20, true), row(20, true)];
        assert!(!interaction_vector_initialization_is_canonical(&[
            &duplicate_initializer,
        ]));
    }
}

fn interaction_kind(
    group: DirectInteractionGroupView<'_>,
) -> Option<DirectInteractionIntrinsicKind> {
    if group.role != DirectExecutorRole::Contribution
        || group.direct_executor_id == DIRECT_NONE_U32
        || group.capability?.metadata.exact_factor_is_kernel_input
    {
        return None;
    }
    Some(group.capability?.fanout.interaction_kind)
}

fn direct_interaction_group_views<'a>(
    plan: &'a DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
) -> RusticolResult<Vec<DirectInteractionGroupView<'a>>> {
    let mut groups = Vec::with_capacity(plan.row_groups().len());
    for descriptor in plan.row_groups() {
        let start = usize::try_from(descriptor.row_start)
            .map_err(|_| RusticolError::integrity("direct interaction row start exceeds usize"))?;
        let end = start
            .checked_add(descriptor.row_count as usize)
            .ok_or_else(|| {
                RusticolError::integrity("direct interaction row range overflows usize")
            })?;
        let stage = u32::from(descriptor.stage);
        groups.push(match descriptor.role {
            DirectExecutorRole::Contribution => {
                let rows = plan.contributions().get(start..end).ok_or_else(|| {
                    RusticolError::integrity("direct interaction contribution rows are absent")
                })?;
                let handle = (descriptor.direct_executor_id != DIRECT_NONE_U32)
                    .then(|| {
                        executors.require(
                            descriptor.direct_executor_id,
                            DirectExecutorRole::Contribution,
                        )
                    })
                    .transpose()?;
                let capability = executors
                    .contribution_fanout(descriptor.direct_executor_id)
                    .map(|fanout| {
                        let handle = handle.ok_or_else(|| {
                            RusticolError::integrity(
                                "direct interaction capability has no contribution handle",
                            )
                        })?;
                        DirectInteractionExecutorCapability::new(
                            handle,
                            executors.contribution_metadata(descriptor.direct_executor_id)?,
                            fanout,
                        )
                    })
                    .transpose()?;
                DirectInteractionGroupView::contribution(
                    stage,
                    descriptor.direct_executor_id,
                    handle,
                    capability,
                    rows,
                )
            }
            DirectExecutorRole::Closure => {
                let rows = plan.closures().get(start..end).ok_or_else(|| {
                    RusticolError::integrity("direct interaction closure rows are absent")
                })?;
                DirectInteractionGroupView::closure(
                    stage,
                    descriptor.direct_executor_id,
                    executors.require(descriptor.direct_executor_id, descriptor.role)?,
                    rows,
                )
            }
            role => DirectInteractionGroupView::other(
                stage,
                role,
                descriptor.direct_executor_id,
                (descriptor.direct_executor_id != DIRECT_NONE_U32)
                    .then(|| executors.require(descriptor.direct_executor_id, role))
                    .transpose()?,
            ),
        });
    }
    Ok(groups)
}

impl DirectInteractionProgram {
    fn build_for_direct_plan(
        plan: &DirectRecurrencePlan,
        executors: &DirectExecutorCatalog,
    ) -> RusticolResult<Self> {
        let groups = direct_interaction_group_views(plan, executors)?;
        let schedule = DirectInteractionScheduleView::new(
            plan.strategy(),
            &groups,
            plan.currents(),
            plan.contributions().len(),
        );
        Self::build(schedule)
    }

    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub(crate) fn build(schedule: DirectInteractionScheduleView<'_>) -> RusticolResult<Self> {
        let mut actions = vec![DirectInteractionGroupAction::Normal; schedule.groups.len()];
        let mut stages = Vec::new();
        let mut stage_groups = BTreeMap::<u32, Vec<usize>>::new();
        for (group_index, group) in schedule.groups.iter().enumerate() {
            if group.role == DirectExecutorRole::Contribution {
                stage_groups
                    .entry(group.stage)
                    .or_default()
                    .push(group_index);
            }
        }
        let raw_runs = schedule
            .groups
            .iter()
            .enumerate()
            .filter_map(|(group_index, &group)| {
                interaction_kind(group)?;
                Some((group_index, interaction_raw_runs(group.contributions()?)?))
            })
            .collect::<BTreeMap<_, _>>();

        for group_indices in stage_groups.values() {
            let Some(candidate) = match_interaction_stage(&schedule, group_indices, &raw_runs)
            else {
                continue;
            };
            let stage_index = u32::try_from(stages.len()).map_err(|_| {
                RusticolError::integrity("direct interaction stage count exceeds u32")
            })?;
            let first_group = *candidate
                .group_indices
                .iter()
                .min()
                .expect("interaction has contribution groups");
            for &group_index in &candidate.group_indices {
                actions[group_index] = if group_index == first_group {
                    DirectInteractionGroupAction::Execute(stage_index)
                } else {
                    DirectInteractionGroupAction::Consumed
                };
            }
            if let Some(group_index) = candidate.closure_group_index {
                actions[group_index] = DirectInteractionGroupAction::Consumed;
            }
            let stored_group_indices = candidate
                .group_indices
                .iter()
                .copied()
                .map(|group_index| {
                    u32::try_from(group_index).map_err(|_| {
                        RusticolError::integrity("direct interaction group index exceeds u32")
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?
                .into_boxed_slice();
            let closure_group_index = candidate
                .closure_group_index
                .map(|group_index| {
                    u32::try_from(group_index).map_err(|_| {
                        RusticolError::integrity("direct terminal closure group index exceeds u32")
                    })
                })
                .transpose()?;
            stages.push(DirectInteractionStage {
                group_indices: stored_group_indices,
                closure_group_index,
                call: candidate.call,
                contexts: candidate.contexts,
                atv_before_color: candidate.atv_before_color,
                anchors: candidate.anchors.into_boxed_slice(),
                outputs: candidate.outputs.into_boxed_slice(),
                tensor_outputs: candidate.tensor_outputs.into_boxed_slice(),
                terminal: candidate.terminal,
                requirements: candidate.requirements,
                logical_row_count: candidate.logical_row_count,
            });
        }

        Ok(Self {
            actions: actions.into_boxed_slice(),
            stages: stages.into_boxed_slice(),
        })
    }

    pub(crate) fn action(&self, group_index: usize) -> DirectInteractionGroupAction {
        self.actions
            .get(group_index)
            .copied()
            .unwrap_or(DirectInteractionGroupAction::Normal)
    }

    pub(crate) fn execute<const PROFILE: bool>(
        &self,
        stage_index: u32,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        factors: DirectFactorView,
    ) -> RusticolResult<Duration> {
        let stage = self.stages.get(stage_index as usize).ok_or_else(|| {
            RusticolError::integrity("direct interaction stage index is out of bounds")
        })?;
        execute_direct_interaction_stage_views::<PROFILE>(
            stage, arena, momenta, parameters, factors,
        )
    }

    pub(crate) fn logical_groups(
        &self,
        stage_index: u32,
    ) -> RusticolResult<(&[u32], Option<u32>, u64)> {
        let stage = self.stages.get(stage_index as usize).ok_or_else(|| {
            RusticolError::integrity("direct interaction stage index is out of bounds")
        })?;
        Ok((
            &stage.group_indices,
            stage.closure_group_index,
            stage.logical_row_count,
        ))
    }
}

struct DirectInteractionStageCandidate {
    group_indices: Vec<usize>,
    closure_group_index: Option<usize>,
    call: DirectInteractionBundleExecutor,
    contexts: DirectInteractionExecutorContexts,
    atv_before_color: bool,
    logical_row_count: u64,
    anchors: Vec<DirectInteractionAnchor>,
    outputs: Vec<DirectInteractionOutput>,
    tensor_outputs: Vec<DirectInteractionTensorOutput>,
    terminal: Option<DirectInteractionTerminalProgram>,
    requirements: DirectInteractionRuntimeRequirements,
}

struct DirectInteractionMatchedSupplement {
    row_start: u32,
    vector_parent_side: u8,
    row_offsets: Vec<u32>,
}

// The sibling-wedge rewrite adds enough hot code to pay for itself only on
// large schedules. Keep the narrower color/ATV interaction rewrite below this
// cold, process-independent workload boundary. Terminal scalarization is a
// separate win: once the cold liveness proof succeeds it removes an entire
// four-component root/closure round trip, including on small schedules.
const DIRECT_LARGE_SCHEDULE_INTERACTION_MIN_ROWS: usize = 1_000_000;

#[inline]
fn direct_large_schedule_interaction_rewrite_enabled(
    strategy: RecurrenceStrategy,
    contribution_row_count: usize,
) -> bool {
    strategy == RecurrenceStrategy::ContractedColorUnion
        && contribution_row_count >= DIRECT_LARGE_SCHEDULE_INTERACTION_MIN_ROWS
}

#[inline]
fn direct_terminal_interaction_rewrite_enabled(strategy: RecurrenceStrategy) -> bool {
    strategy == RecurrenceStrategy::ContractedColorUnion
}

fn match_interaction_stage(
    schedule: &DirectInteractionScheduleView<'_>,
    group_indices: &[usize],
    raw_runs: &BTreeMap<usize, Vec<DirectInteractionRawRun>>,
) -> Option<DirectInteractionStageCandidate> {
    if !(2..=3).contains(&group_indices.len())
        || !group_indices.windows(2).all(|pair| pair[1] == pair[0] + 1)
    {
        return None;
    }
    let mut color = None;
    let mut wedge = None;
    let mut atv = None;
    for &group_index in group_indices {
        let group = schedule.groups[group_index];
        let slot = match interaction_kind(group)? {
            DirectInteractionIntrinsicKind::ColorOrderedThreeVector => &mut color,
            DirectInteractionIntrinsicKind::VectorWedgeVector => &mut wedge,
            DirectInteractionIntrinsicKind::AntisymmetricTensorVector => &mut atv,
        };
        if slot.replace(group_index).is_some() {
            return None;
        }
    }
    let (color_index, atv_index) = (color?, atv?);
    if wedge.is_some() != (group_indices.len() == 3) {
        return None;
    }
    let color = schedule.groups[color_index];
    let atv = schedule.groups[atv_index];
    let color_group_rows = color.contributions()?;
    let atv_group_rows = atv.contributions()?;
    let ordered_vector_groups = if color_index < atv_index {
        [color_group_rows, atv_group_rows]
    } else {
        [atv_group_rows, color_group_rows]
    };
    if !interaction_vector_initialization_is_canonical(&ordered_vector_groups) {
        return None;
    }
    let color_capability = color.capability?.fanout;
    let atv_capability = atv.capability?.fanout;
    let call = color_capability.interaction_call?;
    let color_runs = raw_runs.get(&color_index)?.clone();
    let atv_runs = raw_runs.get(&atv_index)?;
    let fuse_wedge = direct_large_schedule_interaction_rewrite_enabled(
        schedule.strategy,
        schedule.contribution_row_count,
    );
    let wedge_program = if let Some(wedge_index) = wedge.filter(|_| fuse_wedge) {
        let group = schedule.groups[wedge_index];
        let capability = group.capability?.fanout;
        if capability.destination_component_count != 6
            || capability.parent_component_counts != [4, 4]
            || capability.requires_two_momenta
        {
            return None;
        }
        let wedge_runs = raw_runs.get(&wedge_index)?.clone();
        if wedge_runs.len() != color_runs.len() {
            return None;
        }
        for (&color_run, &wedge_run) in color_runs.iter().zip(&wedge_runs) {
            let color_rows = interaction_rows(color_group_rows, color_run)?;
            let wedge_rows = interaction_rows(group.contributions()?, wedge_run)?;
            if color_run.row_count != wedge_run.row_count
                || interaction_input_key(*color_rows.first()?)
                    != interaction_input_key(*wedge_rows.first()?)
                || !color_rows
                    .iter()
                    .zip(wedge_rows)
                    .all(|(color_row, wedge_row)| {
                        color_row.exact_factor_id == wedge_row.exact_factor_id
                    })
            {
                return None;
            }
        }
        Some((capability, wedge_runs))
    } else {
        None
    };

    type AssociationKey = (u32, u32, u32, u32, u32, u32);
    let mut anchors_by_vector = BTreeMap::<AssociationKey, Vec<(usize, u8)>>::new();
    let mut color_destination_offsets = Vec::with_capacity(color_runs.len());
    for (anchor_index, run) in color_runs.iter().copied().enumerate() {
        let rows = interaction_rows(color_group_rows, run)?;
        let first = *rows.first()?;
        let destinations = interaction_destination_offsets(rows)?;
        let minimum_destination = *destinations.keys().next()?;
        let maximum_destination = *destinations.keys().next_back()?;
        for (side, base, momentum) in [
            (
                0_u8,
                first.parent0_component_base,
                first.parent0_momentum_form_id,
            ),
            (
                1_u8,
                first.parent1_component_base_or_sentinel,
                first.parent1_momentum_form_id_or_sentinel,
            ),
        ] {
            anchors_by_vector
                .entry((
                    first.selector_domain_id,
                    base,
                    momentum,
                    run.row_count,
                    minimum_destination,
                    maximum_destination,
                ))
                .or_default()
                .push((anchor_index, side));
        }
        color_destination_offsets.push(destinations);
    }
    let mut matched_supplements = (0..color_runs.len())
        .map(|_| Vec::new())
        .collect::<Vec<Vec<DirectInteractionMatchedSupplement>>>();
    let mut used_sides = vec![[false; 2]; color_runs.len()];
    for &atv_run in atv_runs {
        let atv_rows = interaction_rows(atv_group_rows, atv_run)?;
        let representative = *atv_rows.first()?;
        let atv_destination_offsets = interaction_destination_offsets(atv_rows)?;
        let minimum_destination = *atv_destination_offsets.keys().next()?;
        let maximum_destination = *atv_destination_offsets.keys().next_back()?;
        let candidates = anchors_by_vector.get(&(
            representative.selector_domain_id,
            representative.parent1_component_base_or_sentinel,
            representative.parent1_momentum_form_id_or_sentinel,
            atv_run.row_count,
            minimum_destination,
            maximum_destination,
        ))?;
        let mut matches = candidates.iter().copied().filter(|(anchor_index, _)| {
            color_destination_offsets[*anchor_index]
                .keys()
                .eq(atv_destination_offsets.keys())
        });
        let (anchor_index, vector_parent_side) = matches.next()?;
        if matches.next().is_some() || used_sides[anchor_index][vector_parent_side as usize] {
            return None;
        }
        used_sides[anchor_index][vector_parent_side as usize] = true;
        let row_offsets = interaction_destination_bijection(
            &color_destination_offsets[anchor_index],
            &atv_destination_offsets,
        )?;
        matched_supplements[anchor_index].push(DirectInteractionMatchedSupplement {
            row_start: atv_run.row_start,
            vector_parent_side,
            row_offsets,
        });
    }
    if matched_supplements.iter().flatten().count() == 0 {
        return None;
    }

    // Identical destination ranges are expected across formula groups and
    // repeated interactions. Distinct partially-overlapping ranges would make
    // a single aggregated write ambiguous, so reject that stage.
    let mut destination_widths = BTreeMap::<u32, u32>::new();
    for &group_index in group_indices {
        let group = schedule.groups[group_index];
        let capability = group.capability?.fanout;
        let rows = group.contributions()?;
        for row in rows {
            if destination_widths
                .insert(
                    row.destination_component_base,
                    capability.destination_component_count,
                )
                .is_some_and(|width| width != capability.destination_component_count)
            {
                return None;
            }
        }
    }
    let mut previous_end = 0_u32;
    for (&base, &width) in &destination_widths {
        if base < previous_end {
            return None;
        }
        previous_end = base.checked_add(width)?;
    }

    let mut seen_vector_destinations = BTreeSet::new();
    let mut candidate_anchors = Vec::with_capacity(color_runs.len());
    let mut candidate_outputs = Vec::with_capacity(color_group_rows.len());
    let mut candidate_tensor_outputs = Vec::with_capacity(
        wedge
            .and_then(|group_index| schedule.groups[group_index].contributions())
            .map_or(0, <[DirectContributionRow]>::len),
    );
    let mut requirements = DirectInteractionRuntimeRequirements {
        parameter_count: color_capability
            .required_parameter_count
            .max(atv_capability.required_parameter_count),
        ..DirectInteractionRuntimeRequirements::default()
    };
    if let Some((capability, _)) = &wedge_program {
        requirements.parameter_count = requirements
            .parameter_count
            .max(capability.required_parameter_count);
    }
    for (anchor_index, color_run) in color_runs.into_iter().enumerate() {
        let color_rows = interaction_rows(color_group_rows, color_run)?;
        let representative = *color_rows.first()?;
        requirements.include_current_span(
            representative.parent0_component_base,
            color_capability.parent_component_counts[0],
        )?;
        requirements.include_current_span(
            representative.parent1_component_base_or_sentinel,
            color_capability.parent_component_counts[1],
        )?;
        requirements.include_momentum(representative.parent0_momentum_form_id)?;
        requirements.include_momentum(representative.parent1_momentum_form_id_or_sentinel)?;

        let supplements_for_anchor = &matched_supplements[anchor_index];
        if supplements_for_anchor.len() > 2 {
            return None;
        }
        let mut compiled_operands = [DirectInteractionOperand {
            tensor_component_base: 0,
            vector_parent: DirectInteractionVectorParent::Parent0,
        }; 2];
        let mut supplement_rows = [None; 2];
        for (slot, supplement) in supplements_for_anchor.iter().enumerate() {
            let rows = interaction_rows(
                atv_group_rows,
                DirectInteractionRawRun {
                    row_start: supplement.row_start,
                    row_count: color_run.row_count,
                },
            )?;
            let atv_representative = *rows.first()?;
            let vector_parent = match supplement.vector_parent_side {
                0 => DirectInteractionVectorParent::Parent0,
                1 => DirectInteractionVectorParent::Parent1,
                _ => return None,
            };
            requirements.include_current_span(
                atv_representative.parent0_component_base,
                atv_capability.parent_component_counts[0],
            )?;
            compiled_operands[slot] = DirectInteractionOperand {
                tensor_component_base: atv_representative.parent0_component_base,
                vector_parent,
            };
            supplement_rows[slot] = Some(rows);
        }
        let operands = match supplements_for_anchor.len() {
            0 => DirectInteractionOperands::None,
            1 => DirectInteractionOperands::One(compiled_operands[0]),
            2 => DirectInteractionOperands::Two(compiled_operands),
            _ => unreachable!("interaction supplement count was bounded above"),
        };

        let output_start = u32::try_from(candidate_outputs.len()).ok()?;
        for (offset, color_row) in color_rows.iter().enumerate() {
            requirements.include_current_span(
                color_row.destination_component_base,
                color_capability.destination_component_count,
            )?;
            requirements.include_factor(color_row.exact_factor_id)?;
            let mut exact_factor_ids = [color_row.exact_factor_id, 0, 0];
            for (slot, supplement) in supplements_for_anchor.iter().enumerate() {
                let atv_offset = *supplement.row_offsets.get(offset)?;
                let atv_row = *supplement_rows[slot]?.get(atv_offset as usize)?;
                if atv_row.destination_component_base != color_row.destination_component_base {
                    return None;
                }
                requirements.include_factor(atv_row.exact_factor_id)?;
                exact_factor_ids[slot + 1] = atv_row.exact_factor_id;
            }
            candidate_outputs.push(DirectInteractionOutput {
                destination_component_base: color_row.destination_component_base,
                exact_factor_ids,
                initialize: seen_vector_destinations.insert(color_row.destination_component_base),
            });
        }
        let output_end = u32::try_from(candidate_outputs.len()).ok()?;
        let tensor_output_start = u32::try_from(candidate_tensor_outputs.len()).ok()?;
        if let Some((wedge_capability, wedge_runs)) = &wedge_program {
            let wedge_rows = interaction_rows(
                schedule.groups[wedge?].contributions()?,
                *wedge_runs.get(anchor_index)?,
            )?;
            for wedge_row in wedge_rows {
                requirements.include_current_span(
                    wedge_row.destination_component_base,
                    wedge_capability.destination_component_count,
                )?;
                requirements.include_factor(wedge_row.exact_factor_id)?;
                candidate_tensor_outputs.push(DirectInteractionTensorOutput {
                    destination_component_base: wedge_row.destination_component_base,
                    exact_factor_id: wedge_row.exact_factor_id,
                    initialize: wedge_row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
                        != 0,
                });
            }
        }
        let tensor_output_end = u32::try_from(candidate_tensor_outputs.len()).ok()?;
        candidate_anchors.push(DirectInteractionAnchor {
            parent_component_bases: [
                representative.parent0_component_base,
                representative.parent1_component_base_or_sentinel,
            ],
            parent_momentum_form_ids: [
                representative.parent0_momentum_form_id,
                representative.parent1_momentum_form_id_or_sentinel,
            ],
            operands,
            output_start,
            output_end,
            tensor_output_start,
            tensor_output_end,
        });
    }

    let tensor_destinations = candidate_tensor_outputs
        .iter()
        .map(|output| output.destination_component_base)
        .collect::<BTreeSet<_>>();
    if candidate_anchors
        .iter()
        .any(|anchor| match anchor.operands {
            DirectInteractionOperands::None => false,
            DirectInteractionOperands::One(operand) => {
                tensor_destinations.contains(&operand.tensor_component_base)
            }
            DirectInteractionOperands::Two(operands) => operands
                .iter()
                .any(|operand| tensor_destinations.contains(&operand.tensor_component_base)),
        })
    {
        return None;
    }

    let mut native_group_indices = vec![color_index, atv_index];
    if let Some(wedge_index) = wedge.filter(|_| fuse_wedge) {
        native_group_indices.push(wedge_index);
    }
    native_group_indices.sort_unstable();
    let logical_row_count = native_group_indices
        .iter()
        .map(|&group_index| {
            schedule.groups[group_index]
                .contributions()
                .map(|rows| rows.len() as u64)
                .unwrap_or(0)
        })
        .sum();
    let mut candidate = DirectInteractionStageCandidate {
        group_indices: native_group_indices,
        closure_group_index: None,
        call,
        contexts: DirectInteractionExecutorContexts {
            color: color_capability.context,
            vector_wedge_vector: wedge_program
                .as_ref()
                .map_or(std::ptr::null(), |(capability, _)| capability.context),
            antisymmetric_tensor_vector: atv_capability.context,
        },
        atv_before_color: atv_index < color_index,
        logical_row_count,
        anchors: candidate_anchors,
        outputs: candidate_outputs,
        tensor_outputs: candidate_tensor_outputs,
        terminal: None,
        requirements,
    };
    attach_terminal_interaction_program(schedule, &mut candidate);
    Some(candidate)
}

fn attach_terminal_interaction_program(
    schedule: &DirectInteractionScheduleView<'_>,
    candidate: &mut DirectInteractionStageCandidate,
) {
    let Some((closure_group_index, terminal)) =
        compile_terminal_interaction_program(schedule, candidate)
    else {
        return;
    };
    let mut requirements = candidate.requirements;
    if requirements
        .include_current_span(terminal.partner_component_base, 4)
        .is_none()
    {
        return;
    }
    for component in 0..4 {
        let Some(factor_id) = terminal.component_factor_start.checked_add(component) else {
            return;
        };
        if requirements.include_factor(factor_id).is_none() {
            return;
        }
    }
    for closure in terminal.closures.iter() {
        if requirements
            .include_factor(closure.exact_factor_id)
            .is_none()
        {
            return;
        }
    }
    candidate.requirements = requirements;
    candidate.closure_group_index = Some(closure_group_index);
    candidate.terminal = Some(terminal);
}

fn compile_terminal_interaction_program(
    schedule: &DirectInteractionScheduleView<'_>,
    candidate: &DirectInteractionStageCandidate,
) -> Option<(usize, DirectInteractionTerminalProgram)> {
    if !direct_terminal_interaction_rewrite_enabled(schedule.strategy)
        || !candidate.tensor_outputs.is_empty()
        || candidate.group_indices.len() != 2
        || candidate.outputs.is_empty()
        || schedule.currents.is_empty()
    {
        return None;
    }
    let mut closure_groups = schedule
        .groups
        .iter()
        .enumerate()
        .filter(|(_, group)| group.role == DirectExecutorRole::Closure);
    let (closure_group_index, closure_group) = closure_groups.next()?;
    if closure_groups.next().is_some() {
        return None;
    }
    let DirectExecutorHandle::Closure { call, .. } = closure_group.handle? else {
        return None;
    };
    if !std::ptr::fn_addr_eq(
        call,
        crate::engine::execute_closure_reduce_rows as DirectClosureExecutor,
    ) {
        return None;
    }
    let interaction_stage = schedule.groups[*candidate.group_indices.first()?].stage;
    if candidate
        .group_indices
        .iter()
        .any(|&group_index| schedule.groups[group_index].stage != interaction_stage)
        || interaction_stage.checked_add(1)? != closure_group.stage
    {
        return None;
    }
    let closure_event = closure_group.stage;

    let current_by_stage_and_base = schedule
        .currents
        .iter()
        .filter(|current| current.node_kind == DirectNodeKind::Current)
        .map(|current| ((u32::from(current.stage), current.component_base), current))
        .collect::<BTreeMap<_, _>>();
    let root_bases = candidate
        .outputs
        .iter()
        .map(|output| output.destination_component_base)
        .collect::<BTreeSet<_>>();
    let mut root_semantic_ids = BTreeSet::new();
    for &base in &root_bases {
        let root = *current_by_stage_and_base.get(&(interaction_stage, base))?;
        if root.component_count != 4
            || root.finalization_row_or_sentinel != DIRECT_NONE_U32
            || root.last_use != closure_event
        {
            return None;
        }
        root_semantic_ids.insert(root.semantic_current_id);
    }

    // Every write to a scalarized root must belong to this native stage.
    // Otherwise the scalar scratch would omit a contribution or close before
    // the final writer.
    let consumed_groups = candidate
        .group_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    for (group_index, group) in schedule.groups.iter().enumerate().filter(|(_, group)| {
        group.role == DirectExecutorRole::Contribution && group.stage == interaction_stage
    }) {
        if group.contributions()?.iter().any(|row| {
            root_bases.contains(&row.destination_component_base)
                && !consumed_groups.contains(&group_index)
        }) {
            return None;
        }
    }
    // No scalarized root may feed another contribution at this or a later
    // stage; the closure group below must be its only remaining consumer.
    for group in schedule.groups.iter().filter(|group| {
        group.role == DirectExecutorRole::Contribution && group.stage >= interaction_stage
    }) {
        if group.contributions()?.iter().any(|row| {
            root_bases.contains(&row.parent0_component_base)
                || root_bases.contains(&row.parent1_component_base_or_sentinel)
        }) {
            return None;
        }
    }

    // At the closure event arena liveness makes these bases unambiguous even
    // though the arena is reused between earlier stages.
    let mut live_current_by_base = BTreeMap::new();
    for current in schedule
        .currents
        .iter()
        .filter(|current| current.last_use == closure_event)
    {
        if live_current_by_base
            .insert(current.component_base, current)
            .is_some()
        {
            return None;
        }
    }
    let mut partner = None;
    let mut component_factor_start = None;
    let mut attached_roots = BTreeSet::new();
    let mut amplitude_destinations = BTreeSet::new();
    let mut amplitude_destination_count = 0_u32;
    let closure_rows = closure_group.closures()?;
    if closure_rows.is_empty() {
        return None;
    }
    let mut closures = Vec::with_capacity(closure_rows.len());
    for row in closure_rows {
        if row.component_count != 4 || !amplitude_destinations.insert(row.amplitude_destination_id)
        {
            return None;
        }
        amplitude_destination_count =
            amplitude_destination_count.max(row.amplitude_destination_id.checked_add(1)?);
        let parent0 = *live_current_by_base.get(&row.parent0_component_base)?;
        let parent1 = *live_current_by_base.get(&row.parent1_component_base_or_sentinel)?;
        let parent0_is_root = root_semantic_ids.contains(&parent0.semantic_current_id);
        let parent1_is_root = root_semantic_ids.contains(&parent1.semantic_current_id);
        if parent0_is_root == parent1_is_root {
            return None;
        }
        let (root, terminal_partner) = if parent0_is_root {
            (parent0, parent1)
        } else {
            (parent1, parent0)
        };
        if terminal_partner.component_count != 4
            || u32::from(terminal_partner.stage) >= interaction_stage
            || terminal_partner.last_use != closure_event
            || terminal_partner.finalization_row_or_sentinel != DIRECT_NONE_U32
        {
            return None;
        }
        let partner_key = (
            terminal_partner.semantic_current_id,
            terminal_partner.component_base,
        );
        if partner
            .replace(partner_key)
            .is_some_and(|seen| seen != partner_key)
            || component_factor_start
                .replace(row.component_factor_start)
                .is_some_and(|seen| seen != row.component_factor_start)
        {
            return None;
        }
        attached_roots.insert(root.semantic_current_id);
        closures.push(DirectInteractionTerminalClosure {
            root_component_base: root.component_base,
            amplitude_destination_id: row.amplitude_destination_id,
            exact_factor_id: row.exact_factor_id,
        });
    }
    if attached_roots != root_semantic_ids {
        return None;
    }
    Some((
        closure_group_index,
        DirectInteractionTerminalProgram {
            partner_component_base: partner?.1,
            component_factor_start: component_factor_start?,
            amplitude_destination_count,
            closures: closures.into_boxed_slice(),
        },
    ))
}

#[cfg(test)]
mod terminal_current_descriptor_tests {
    use super::*;

    unsafe fn interaction_stub(
        _contexts: DirectInteractionExecutorContexts,
        _arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        _factors: DirectFactorView,
        _batch: DirectInteractionBundleBatch<'_>,
    ) -> RusticolResult<()> {
        Ok(())
    }

    fn candidate() -> DirectInteractionStageCandidate {
        DirectInteractionStageCandidate {
            group_indices: vec![0, 1],
            closure_group_index: None,
            call: interaction_stub,
            contexts: DirectInteractionExecutorContexts {
                color: std::ptr::null(),
                vector_wedge_vector: std::ptr::null(),
                antisymmetric_tensor_vector: std::ptr::null(),
            },
            atv_before_color: false,
            logical_row_count: 2,
            anchors: Vec::new(),
            outputs: vec![DirectInteractionOutput {
                destination_component_base: 20,
                exact_factor_ids: [0; 3],
                initialize: true,
            }],
            tensor_outputs: Vec::new(),
            terminal: None,
            requirements: DirectInteractionRuntimeRequirements::default(),
        }
    }

    #[test]
    fn terminal_compiler_requires_and_consumes_current_descriptors() {
        let contribution = DirectContributionRow {
            parent0_component_base: 4,
            parent1_component_base_or_sentinel: 8,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 1,
            destination_component_base: 20,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
        };
        let closure = DirectClosureRow {
            parent0_component_base: 20,
            parent1_component_base_or_sentinel: 0,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            amplitude_destination_id: 0,
            exact_factor_id: 0,
            component_factor_start: 0,
            component_count: 4,
            selector_domain_id: 0,
            flags: 0,
        };
        let contribution0 = [contribution];
        let contribution1 = [DirectContributionRow {
            flags: 0,
            ..contribution
        }];
        let closures = [closure];
        let groups = [
            DirectInteractionGroupView::contribution(1, 1, None, None, &contribution0),
            DirectInteractionGroupView::contribution(1, 2, None, None, &contribution1),
            DirectInteractionGroupView::closure(
                2,
                3,
                DirectExecutorHandle::Closure {
                    call: crate::engine::execute_closure_reduce_rows,
                    context: std::ptr::null(),
                },
                &closures,
            ),
        ];
        let currents = [
            DirectCurrentDescriptor {
                semantic_current_id: 0,
                node_kind: DirectNodeKind::Current,
                state_template_id: 0,
                component_base: 20,
                component_count: 4,
                momentum_form_id: 0,
                stage: 1,
                selector_domain_id: 0,
                first_use: 1,
                last_use: 2,
                source_row_or_sentinel: DIRECT_NONE_U32,
                finalization_row_or_sentinel: DIRECT_NONE_U32,
            },
            DirectCurrentDescriptor {
                semantic_current_id: 1,
                node_kind: DirectNodeKind::Source,
                state_template_id: 0,
                component_base: 0,
                component_count: 4,
                momentum_form_id: 0,
                stage: 0,
                selector_domain_id: 0,
                first_use: 0,
                last_use: 2,
                source_row_or_sentinel: 0,
                finalization_row_or_sentinel: DIRECT_NONE_U32,
            },
        ];
        let without_currents = DirectInteractionScheduleView::new(
            RecurrenceStrategy::ContractedColorUnion,
            &groups,
            &[],
            2,
        );
        assert!(compile_terminal_interaction_program(&without_currents, &candidate()).is_none());

        let with_currents = DirectInteractionScheduleView::new(
            RecurrenceStrategy::ContractedColorUnion,
            &groups,
            &currents,
            2,
        );
        let (closure_group, terminal) =
            compile_terminal_interaction_program(&with_currents, &candidate()).unwrap();
        assert_eq!(closure_group, 2);
        assert_eq!(terminal.partner_component_base, 0);
        assert_eq!(terminal.closures.len(), 1);
        assert_eq!(terminal.closures[0].root_component_base, 20);
    }
}

fn validate_fused_fanout_row(
    row: DirectContributionRow,
    parent_component_counts: [u32; 2],
    requires_two_momenta: bool,
    arena_component_count: u32,
    momentum_form_count: u32,
) -> RusticolResult<()> {
    for (base, required) in [
        (row.parent0_component_base, parent_component_counts[0]),
        (
            row.parent1_component_base_or_sentinel,
            parent_component_counts[1],
        ),
    ] {
        if base == DIRECT_NONE_U32
            || base
                .checked_add(required)
                .is_none_or(|end| end > arena_component_count)
        {
            return Err(RusticolError::integrity(
                "direct fused fanout parent component span is incompatible with its intrinsic",
            ));
        }
    }
    if requires_two_momenta
        && (row.parent0_momentum_form_id >= momentum_form_count
            || row.parent1_momentum_form_id_or_sentinel >= momentum_form_count)
    {
        return Err(RusticolError::integrity(
            "direct fused fanout momentum forms are incompatible with its intrinsic",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod fused_fanout_validation_tests {
    use super::*;

    fn interaction_row(destination_component_base: u32) -> DirectContributionRow {
        DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 4,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 1,
            destination_component_base,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        }
    }

    #[test]
    fn interaction_destination_bijection_accepts_nonidentity_and_rejects_nonsets() {
        let color = [interaction_row(8), interaction_row(16), interaction_row(24)];
        let permuted = [interaction_row(24), interaction_row(8), interaction_row(16)];
        let color = interaction_destination_offsets(&color).unwrap();
        let permuted = interaction_destination_offsets(&permuted).unwrap();
        assert_eq!(
            interaction_destination_bijection(&color, &permuted).unwrap(),
            [1, 2, 0]
        );

        assert!(
            interaction_destination_offsets(&[interaction_row(8), interaction_row(8),]).is_none()
        );
        let different = interaction_destination_offsets(&[
            interaction_row(8),
            interaction_row(16),
            interaction_row(28),
        ])
        .unwrap();
        assert!(interaction_destination_bijection(&color, &different).is_none());
    }

    #[test]
    fn rejects_short_parent_spans_and_missing_momentum_forms() {
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 4,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 1,
            destination_component_base: 8,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert!(validate_fused_fanout_row(row, [9, 4], false, 8, 2).is_err());
        assert!(validate_fused_fanout_row(row, [4, 4], true, 8, 1).is_err());
        validate_fused_fanout_row(row, [4, 4], true, 8, 2).unwrap();
    }

    #[test]
    fn interaction_execution_gate_is_full_unselected_single_point_only() {
        let plan = crate::recurrence::direct_plan::tests::valid_plan();
        let groups = plan.row_groups();
        assert!(direct_interaction_execution_enabled::<true>(
            RecurrenceStrategy::TopologyReplay,
            groups,
            groups,
            None,
            1,
        ));
        assert!(!direct_interaction_execution_enabled::<false>(
            RecurrenceStrategy::TopologyReplay,
            groups,
            groups,
            None,
            1,
        ));
        assert!(!direct_interaction_execution_enabled::<true>(
            RecurrenceStrategy::TopologyReplay,
            groups,
            groups,
            None,
            2,
        ));
        assert!(!direct_interaction_execution_enabled::<true>(
            RecurrenceStrategy::TopologyReplay,
            groups,
            groups,
            Some(0),
            1,
        ));
        assert!(!direct_interaction_execution_enabled::<true>(
            RecurrenceStrategy::AllFlowUnion,
            groups,
            groups,
            None,
            1,
        ));
        assert!(!direct_interaction_execution_enabled::<true>(
            RecurrenceStrategy::ContractedColorUnion,
            groups,
            &groups[1..],
            None,
            1,
        ));
    }

    #[test]
    fn large_schedule_interaction_rewrite_uses_only_strategy_and_row_count() {
        assert!(!direct_large_schedule_interaction_rewrite_enabled(
            RecurrenceStrategy::ContractedColorUnion,
            DIRECT_LARGE_SCHEDULE_INTERACTION_MIN_ROWS - 1,
        ));
        assert!(direct_large_schedule_interaction_rewrite_enabled(
            RecurrenceStrategy::ContractedColorUnion,
            DIRECT_LARGE_SCHEDULE_INTERACTION_MIN_ROWS,
        ));
        assert!(!direct_large_schedule_interaction_rewrite_enabled(
            RecurrenceStrategy::TopologyReplay,
            DIRECT_LARGE_SCHEDULE_INTERACTION_MIN_ROWS,
        ));
        assert!(!direct_large_schedule_interaction_rewrite_enabled(
            RecurrenceStrategy::AllFlowUnion,
            usize::MAX,
        ));
        assert!(direct_terminal_interaction_rewrite_enabled(
            RecurrenceStrategy::ContractedColorUnion,
        ));
        assert!(!direct_terminal_interaction_rewrite_enabled(
            RecurrenceStrategy::TopologyReplay,
        ));
        assert!(!direct_terminal_interaction_rewrite_enabled(
            RecurrenceStrategy::AllFlowUnion,
        ));
    }
}

fn push_singleton_direct_step(
    steps: &mut Vec<DirectSingletonFanoutStep>,
    group_step_start: usize,
    row_start: u32,
    row_count: u32,
) -> RusticolResult<()> {
    if steps.len() > group_step_start
        && let Some(DirectSingletonFanoutStep::Direct {
            row_start: previous_start,
            row_count: previous_count,
        }) = steps.last_mut()
        && previous_start
            .checked_add(*previous_count)
            .is_some_and(|end| end == row_start)
    {
        *previous_count = previous_count.checked_add(row_count).ok_or_else(|| {
            RusticolError::integrity("singleton fanout direct-step count exceeds u32")
        })?;
        return Ok(());
    }
    steps.push(DirectSingletonFanoutStep::Direct {
        row_start,
        row_count,
    });
    Ok(())
}

fn flush_singleton_fanout_batch(
    rows: &[DirectContributionRow],
    pending_runs: &mut Vec<(u32, u32)>,
    steps: &mut Vec<DirectSingletonFanoutStep>,
    runs: &mut Vec<DirectContributionFanoutRun>,
    bundles: &mut Vec<DirectContributionFanoutBundle>,
) -> RusticolResult<()> {
    let Some(&(first_row_start, _)) = pending_runs.first() else {
        return Ok(());
    };
    let run_start = u64::try_from(runs.len())
        .map_err(|_| RusticolError::integrity("singleton fanout run count exceeds u64"))?;
    let mut logical_row_count = 0_u32;
    let mut expected_start = first_row_start;
    for &(row_start, row_count) in pending_runs.iter() {
        if row_count < 2 || row_start != expected_start {
            return Err(RusticolError::integrity(
                "singleton fanout pending runs are not contiguous reusable classes",
            ));
        }
        push_output_only_fanout_run(rows, u64::from(row_start), row_count, 0, runs, bundles)?;
        logical_row_count = logical_row_count.checked_add(row_count).ok_or_else(|| {
            RusticolError::integrity("singleton fanout batch row count exceeds u32")
        })?;
        expected_start = expected_start.checked_add(row_count).ok_or_else(|| {
            RusticolError::integrity("singleton fanout pending-run coverage exceeds u32")
        })?;
    }
    steps.push(DirectSingletonFanoutStep::Fanout {
        row_start: first_row_start,
        row_count: logical_row_count,
        run_start,
        run_count: u32::try_from(pending_runs.len()).map_err(|_| {
            RusticolError::integrity("singleton fanout batch run count exceeds u32")
        })?,
    });
    pending_runs.clear();
    Ok(())
}

/// Append the shared run/bundle representation used by both persisted direct
/// plans and family-local singleton fanout. Output-only contribution kernels
/// leave the authenticated exact factor for this destination replay step.
fn push_output_only_fanout_run(
    rows: &[DirectContributionRow],
    row_start: u64,
    row_count: u32,
    scratch_component_base: u32,
    runs: &mut Vec<DirectContributionFanoutRun>,
    bundles: &mut Vec<DirectContributionFanoutBundle>,
) -> RusticolResult<()> {
    let start = usize::try_from(row_start)
        .map_err(|_| RusticolError::integrity("direct fanout row start exceeds usize"))?;
    let end = start
        .checked_add(row_count as usize)
        .ok_or_else(|| RusticolError::integrity("direct fanout row range overflows usize"))?;
    let run_rows = rows
        .get(start..end)
        .ok_or_else(|| RusticolError::integrity("direct fanout row range is out of bounds"))?;
    let bundle_start = u64::try_from(bundles.len())
        .map_err(|_| RusticolError::integrity("direct fanout bundle count exceeds u64"))?;
    let mut offset = 0_usize;
    while offset < run_rows.len() {
        let factor_id = run_rows[offset].exact_factor_id;
        let first = offset;
        offset += 1;
        while offset < run_rows.len() && run_rows[offset].exact_factor_id == factor_id {
            offset += 1;
        }
        bundles.push(DirectContributionFanoutBundle {
            row_start: row_start
                .checked_add(u64::try_from(first).map_err(|_| {
                    RusticolError::integrity("direct fanout bundle offset exceeds u64")
                })?)
                .ok_or_else(|| RusticolError::integrity("direct fanout bundle start overflows"))?,
            row_count: u32::try_from(offset - first)
                .map_err(|_| RusticolError::integrity("direct fanout factor bundle exceeds u32"))?,
            effective_factor_id_or_sentinel: factor_id,
        });
    }
    let bundle_count = u32::try_from(
        u64::try_from(bundles.len())
            .map_err(|_| RusticolError::integrity("direct fanout bundle count exceeds u64"))?
            .checked_sub(bundle_start)
            .ok_or_else(|| RusticolError::integrity("direct fanout bundle range underflows"))?,
    )
    .map_err(|_| RusticolError::integrity("direct fanout run bundle count exceeds u32"))?;
    runs.push(DirectContributionFanoutRun {
        row_start,
        row_count,
        scratch_component_base,
        bundle_start,
        bundle_count,
    });
    Ok(())
}

fn contribution_fanout_inputs_equal(
    left: DirectContributionRow,
    right: DirectContributionRow,
    exact_factor_is_kernel_input: bool,
) -> bool {
    left.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0
        && right.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0
        && left.selector_domain_id == right.selector_domain_id
        && left.parent0_component_base == right.parent0_component_base
        && left.parent1_component_base_or_sentinel == right.parent1_component_base_or_sentinel
        && left.parent0_momentum_form_id == right.parent0_momentum_form_id
        && left.parent1_momentum_form_id_or_sentinel == right.parent1_momentum_form_id_or_sentinel
        && (!exact_factor_is_kernel_input || left.exact_factor_id == right.exact_factor_id)
}

fn push_direct_fanout_step(
    steps: &mut Vec<DirectContributionFanoutStep>,
    segment_step_start: usize,
    row_start: u64,
    row_count: u32,
) -> RusticolResult<()> {
    if steps.len() > segment_step_start
        && let Some(DirectContributionFanoutStep::Direct {
            row_start: previous_start,
            row_count: previous_count,
        }) = steps.last_mut()
        && previous_start
            .checked_add(u64::from(*previous_count))
            .is_some_and(|end| end == row_start)
    {
        *previous_count = previous_count.checked_add(row_count).ok_or_else(|| {
            RusticolError::integrity("direct fanout direct-step row count exceeds u32")
        })?;
        return Ok(());
    }
    steps.push(DirectContributionFanoutStep::Direct {
        row_start,
        row_count,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn flush_pending_fanout_batch(
    plan: &DirectRecurrencePlan,
    descriptor: &DirectRowGroupDescriptor,
    metadata: DirectContributionExecutionMetadata,
    scratch_component_base: u32,
    unit_factor_id: u32,
    pending_runs: &mut Vec<(usize, usize)>,
    steps: &mut Vec<DirectContributionFanoutStep>,
    representatives: &mut Vec<DirectContributionRow>,
    runs: &mut Vec<DirectContributionFanoutRun>,
    bundles: &mut Vec<DirectContributionFanoutBundle>,
    scratch_component_count: &mut u32,
    needs_unit_factor: &mut bool,
) -> RusticolResult<()> {
    if pending_runs.is_empty() {
        return Ok(());
    }
    let representative_start = u64::try_from(representatives.len())
        .map_err(|_| RusticolError::integrity("direct fanout representative count exceeds u64"))?;
    let run_table_start = u64::try_from(runs.len())
        .map_err(|_| RusticolError::integrity("direct fanout run count exceeds u64"))?;
    let first_row_start = descriptor
        .row_start
        .checked_add(pending_runs[0].0 as u64)
        .ok_or_else(|| RusticolError::integrity("direct fanout row start overflows u64"))?;
    let mut logical_row_count = 0_u32;
    for (slot, &(relative_start, run_count)) in pending_runs.iter().enumerate() {
        let row_index = usize::try_from(descriptor.row_start)
            .ok()
            .and_then(|start| start.checked_add(relative_start))
            .ok_or_else(|| RusticolError::integrity("direct fanout row index overflows usize"))?;
        let mut representative = *plan.contributions().get(row_index).ok_or_else(|| {
            RusticolError::integrity("direct fanout representative row is out of bounds")
        })?;
        let slot_base = u32::try_from(slot)
            .ok()
            .and_then(|slot| slot.checked_mul(metadata.destination_component_count))
            .and_then(|offset| scratch_component_base.checked_add(offset))
            .ok_or_else(|| {
                RusticolError::integrity("direct fanout scratch component base exceeds u32")
            })?;
        representative.destination_component_base = slot_base;
        representative.flags = DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
        if !metadata.exact_factor_is_kernel_input {
            representative.exact_factor_id = unit_factor_id;
            *needs_unit_factor = true;
        }
        representatives.push(representative);
        let run_row_start = descriptor
            .row_start
            .checked_add(relative_start as u64)
            .ok_or_else(|| RusticolError::integrity("direct fanout run start overflows u64"))?;
        let run_row_count = u32::try_from(run_count)
            .map_err(|_| RusticolError::integrity("direct fanout run exceeds u32"))?;
        if metadata.exact_factor_is_kernel_input {
            let bundle_start = u64::try_from(bundles.len())
                .map_err(|_| RusticolError::integrity("direct fanout bundle count exceeds u64"))?;
            bundles.push(DirectContributionFanoutBundle {
                row_start: run_row_start,
                row_count: run_row_count,
                effective_factor_id_or_sentinel: DIRECT_NONE_U32,
            });
            runs.push(DirectContributionFanoutRun {
                row_start: run_row_start,
                row_count: run_row_count,
                scratch_component_base: slot_base,
                bundle_start,
                bundle_count: 1,
            });
        } else {
            push_output_only_fanout_run(
                plan.contributions(),
                run_row_start,
                run_row_count,
                slot_base,
                runs,
                bundles,
            )?;
        }
        logical_row_count = logical_row_count
            .checked_add(u32::try_from(run_count).map_err(|_| {
                RusticolError::integrity("direct fanout logical run count exceeds u32")
            })?)
            .ok_or_else(|| RusticolError::integrity("direct fanout batch row count exceeds u32"))?;
    }
    let representative_count = u32::try_from(pending_runs.len())
        .map_err(|_| RusticolError::integrity("direct fanout batch exceeds u32"))?;
    let required_scratch = representative_count
        .checked_mul(metadata.destination_component_count)
        .ok_or_else(|| RusticolError::integrity("direct fanout scratch size exceeds u32"))?;
    *scratch_component_count = (*scratch_component_count).max(required_scratch);
    steps.push(DirectContributionFanoutStep::Fanout {
        row_start: first_row_start,
        row_count: logical_row_count,
        representative_start,
        representative_count,
        run_start: run_table_start,
        run_count: representative_count,
        destination_component_count: metadata.destination_component_count,
    });
    pending_runs.clear();
    Ok(())
}

pub struct DirectWorkspace<'a> {
    pub current_re: &'a mut [f64],
    pub current_im: &'a mut [f64],
    pub amplitude_re: &'a mut [f64],
    pub amplitude_im: &'a mut [f64],
    pub momenta: &'a [f64],
    pub momentum_form_count: u32,
    pub lorentz_component_count: u16,
    pub parameters_re: &'a [f64],
    pub parameters_im: &'a [f64],
    pub factors_re: &'a [f64],
    pub factors_im: &'a [f64],
    pub point_stride: u32,
}

impl DirectWorkspace<'_> {
    fn validate(&self, point_count: u32) -> RusticolResult<()> {
        if point_count == 0 || self.point_stride == 0 || point_count > self.point_stride {
            return Err(RusticolError::invalid_argument(
                "direct recurrence point count must be within the workspace stride",
            ));
        }
        if self.current_re.len() != self.current_im.len() {
            return Err(RusticolError::invalid_argument(
                "direct recurrence current real and imaginary arenas differ in length",
            ));
        }
        if self.amplitude_re.len() != self.amplitude_im.len() {
            return Err(RusticolError::invalid_argument(
                "direct recurrence amplitude real and imaginary arenas differ in length",
            ));
        }
        let point_stride = self.point_stride as usize;
        if !self.current_re.len().is_multiple_of(point_stride)
            || !self.amplitude_re.len().is_multiple_of(point_stride)
        {
            return Err(RusticolError::invalid_argument(
                "direct recurrence arena does not contain whole point-stride planes",
            ));
        }
        if self.parameters_re.len() != self.parameters_im.len()
            || self.factors_re.len() != self.factors_im.len()
        {
            return Err(RusticolError::invalid_argument(
                "direct recurrence split-complex catalogs differ in length",
            ));
        }
        let expected_momenta = usize::try_from(self.momentum_form_count)
            .ok()
            .and_then(|forms| forms.checked_mul(usize::from(self.lorentz_component_count)))
            .and_then(|planes| planes.checked_mul(self.point_stride as usize))
            .ok_or_else(|| {
                RusticolError::invalid_argument("direct recurrence momentum arena overflows")
            })?;
        if self.momenta.len() != expected_momenta {
            return Err(RusticolError::invalid_argument(
                "direct recurrence momentum arena has an inconsistent length",
            ));
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
        let current_scalar_len = u64::try_from(self.current_re.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence current arena exceeds u64")
        })?;
        let amplitude_scalar_len = u64::try_from(self.amplitude_re.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence amplitude arena exceeds u64")
        })?;
        let momentum_scalar_len = u64::try_from(self.momenta.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence momentum arena exceeds u64")
        })?;
        let parameter_count = u32::try_from(self.parameters_re.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence parameter catalog exceeds u32")
        })?;
        let factor_count = u32::try_from(self.factors_re.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence factor catalog exceeds u32")
        })?;
        Ok((
            DirectArenaView {
                current_re: self.current_re.as_mut_ptr(),
                current_im: self.current_im.as_mut_ptr(),
                current_scalar_len,
                amplitude_re: self.amplitude_re.as_mut_ptr(),
                amplitude_im: self.amplitude_im.as_mut_ptr(),
                amplitude_scalar_len,
                point_stride: self.point_stride,
            },
            DirectMomentumView {
                values: self.momenta.as_ptr(),
                scalar_len: momentum_scalar_len,
                form_count: self.momentum_form_count,
                lorentz_component_count: self.lorentz_component_count,
                point_stride: self.point_stride,
            },
            DirectParameterView {
                values_re: self.parameters_re.as_ptr(),
                values_im: self.parameters_im.as_ptr(),
                value_count: parameter_count,
            },
            DirectFactorView {
                values_re: self.factors_re.as_ptr(),
                values_im: self.factors_im.as_ptr(),
                value_count: factor_count,
            },
        ))
    }

    fn clear_current_stage(
        &mut self,
        plan: &DirectRecurrencePlan,
        stage: u16,
        point_count: u32,
        selected_sector_id: Option<u32>,
    ) -> RusticolResult<()> {
        let current_plane_count = u32::try_from(self.current_re.len() / self.point_stride as usize)
            .map_err(|_| {
                RusticolError::integrity(
                    "direct recurrence current plane count exceeds u32 during stage clear",
                )
            })?;
        for current in plan.currents().iter().filter(|current| {
            current.node_kind == DirectNodeKind::Current && current.stage == stage
        }) {
            if let Some(sector_id) = selected_sector_id
                && !plan.selector_domain_contains(current.selector_domain_id, sector_id)?
            {
                continue;
            }
            clear_split_active_range(
                self.current_re,
                self.current_im,
                current_plane_count,
                self.point_stride,
                point_count,
                current.component_base,
                u32::from(current.component_count),
                "recurrence current arena",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectExecutionCounters {
    pub source_calls: u64,
    pub source_rows: u64,
    pub contribution_calls: u64,
    pub contribution_rows: u64,
    pub finalization_calls: u64,
    pub finalization_rows: u64,
    pub closure_calls: u64,
    pub closure_rows: u64,
    pub packed_input_bytes: u64,
    pub packed_output_bytes: u64,
    pub scatter_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectExecutionRoleTimings {
    pub source: Duration,
    pub contribution: Duration,
    pub finalization: Duration,
    pub closure: Duration,
}

pub fn execute_direct_plan(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
) -> RusticolResult<()> {
    let mut unused_timings = DirectExecutionRoleTimings::default();
    execute_direct_plan_impl::<true, true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        None,
        workspace,
        point_count,
        counters,
        &mut unused_timings,
        None,
    )
}

pub fn execute_direct_plan_profiled(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
) -> RusticolResult<()> {
    execute_direct_plan_impl::<true, true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        None,
        workspace,
        point_count,
        counters,
        timings,
        None,
    )
}

#[allow(dead_code)]
pub(crate) fn execute_direct_plan_profiled_with_traffic(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    traffic: &mut DirectArenaTrafficCounters,
) -> RusticolResult<()> {
    execute_direct_plan_impl::<true, true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        None,
        workspace,
        point_count,
        counters,
        timings,
        Some(traffic),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_direct_plan_profiled_with_fanout_and_traffic(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    fanout: &DirectContributionFanoutProgram,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    traffic: &mut DirectArenaTrafficCounters,
) -> RusticolResult<()> {
    execute_direct_plan_impl::<true, true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        Some(fanout),
        workspace,
        point_count,
        counters,
        timings,
        Some(traffic),
    )
}

/// Execute the authenticated schedule without touching profiling counters.
///
/// This is a distinct const-generic specialization so release builds remove
/// every role-counter branch from the ordinary native evaluation path.
pub fn execute_direct_plan_unprofiled(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let mut unused = DirectExecutionCounters::default();
    let mut unused_timings = DirectExecutionRoleTimings::default();
    execute_direct_plan_impl::<false, true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        None,
        workspace,
        point_count,
        &mut unused,
        &mut unused_timings,
        None,
    )
}

pub(crate) fn execute_direct_plan_unprofiled_with_fanout(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    fanout: &DirectContributionFanoutProgram,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    if point_count == 1
        && plan.strategy() != RecurrenceStrategy::AllFlowUnion
        && !direct_current_observation_active()
    {
        return fanout.execute_full_singleton_unprofiled(plan, executors, workspace);
    }
    let mut unused = DirectExecutionCounters::default();
    let mut unused_timings = DirectExecutionRoleTimings::default();
    execute_direct_plan_impl::<false, true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        Some(fanout),
        workspace,
        point_count,
        &mut unused,
        &mut unused_timings,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn execute_direct_plan_selected_profiled_with_traffic(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    traffic: &mut DirectArenaTrafficCounters,
) -> RusticolResult<()> {
    execute_direct_plan_impl::<true, false>(
        plan,
        row_groups,
        Some(selected_sector_id),
        executors,
        None,
        workspace,
        point_count,
        counters,
        timings,
        Some(traffic),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_direct_plan_selected_profiled_with_fanout_and_traffic(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    executors: &DirectExecutorCatalog,
    fanout: &DirectContributionFanoutProgram,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    traffic: &mut DirectArenaTrafficCounters,
) -> RusticolResult<()> {
    execute_direct_plan_impl::<true, false>(
        plan,
        row_groups,
        Some(selected_sector_id),
        executors,
        Some(fanout),
        workspace,
        point_count,
        counters,
        timings,
        Some(traffic),
    )
}

#[allow(dead_code)]
pub(crate) fn execute_direct_plan_selected_unprofiled(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let mut unused = DirectExecutionCounters::default();
    let mut unused_timings = DirectExecutionRoleTimings::default();
    execute_direct_plan_impl::<false, false>(
        plan,
        row_groups,
        Some(selected_sector_id),
        executors,
        None,
        workspace,
        point_count,
        &mut unused,
        &mut unused_timings,
        None,
    )
}

pub(crate) fn execute_direct_plan_selected_unprofiled_with_fanout(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    executors: &DirectExecutorCatalog,
    fanout: &DirectContributionFanoutProgram,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let mut unused = DirectExecutionCounters::default();
    let mut unused_timings = DirectExecutionRoleTimings::default();
    execute_direct_plan_impl::<false, false>(
        plan,
        row_groups,
        Some(selected_sector_id),
        executors,
        Some(fanout),
        workspace,
        point_count,
        &mut unused,
        &mut unused_timings,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_direct_plan_impl<const PROFILE: bool, const ALLOW_INTERACTION_REWRITE: bool>(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: Option<u32>,
    executors: &DirectExecutorCatalog,
    fanout: Option<&DirectContributionFanoutProgram>,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    traffic: Option<&mut DirectArenaTrafficCounters>,
) -> RusticolResult<()> {
    execute_direct_plan_with_clears_impl::<PROFILE, ALLOW_INTERACTION_REWRITE, false>(
        plan,
        row_groups,
        selected_sector_id,
        executors,
        fanout,
        workspace,
        point_count,
        counters,
        timings,
        traffic,
        None,
    )
}

#[derive(Clone, Copy)]
pub(crate) struct DirectCurrentClearRange {
    pub(crate) component_base: u32,
    pub(crate) component_count: u32,
}

pub(crate) struct DirectStageClearRanges {
    pub(crate) stage: u16,
    pub(crate) ranges: Box<[DirectCurrentClearRange]>,
}

/// Restricted destination closures are prepared once. Their row runs may cut
/// through a full-plan fanout step, so execute the authenticated raw rows and
/// clear only dependency-closed stage spans whose first retained contribution
/// does not initialize them, with no current-table scan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_direct_plan_destination_selected<const PROFILE: bool>(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    current_clear_stages: &[DirectStageClearRanges],
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    traffic: &mut DirectArenaTrafficCounters,
) -> RusticolResult<()> {
    execute_direct_plan_with_clears_impl::<PROFILE, false, true>(
        plan,
        row_groups,
        Some(selected_sector_id),
        executors,
        None,
        workspace,
        point_count,
        counters,
        timings,
        PROFILE.then_some(traffic),
        Some(current_clear_stages),
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_direct_plan_with_clears_impl<
    const PROFILE: bool,
    const ALLOW_INTERACTION_REWRITE: bool,
    const PREPARED_CLEARS: bool,
>(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: Option<u32>,
    executors: &DirectExecutorCatalog,
    fanout: Option<&DirectContributionFanoutProgram>,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    mut traffic: Option<&mut DirectArenaTrafficCounters>,
    current_clear_stages: Option<&[DirectStageClearRanges]>,
) -> RusticolResult<()> {
    clear_direct_executor_error_detail();
    workspace.validate(point_count)?;
    if executors.plan_layout_digest != plan.runtime_layout_digest() {
        return Err(RusticolError::integrity(
            "direct recurrence executor catalog belongs to a different plan",
        ));
    }
    if fanout.is_some_and(|fanout| fanout.plan_layout_digest != plan.runtime_layout_digest()) {
        return Err(RusticolError::integrity(
            "direct contribution fanout program belongs to a different plan",
        ));
    }
    let observation_active = direct_current_observation_active();
    let interaction_enabled = direct_interaction_execution_enabled::<ALLOW_INTERACTION_REWRITE>(
        plan.strategy(),
        plan.row_groups(),
        row_groups,
        selected_sector_id,
        point_count,
    ) && !observation_active;
    // Observations promise materialized semantic currents. Bypass both native
    // interaction rewrites and generic fanout scratch/replay while a capture
    // is active so every authenticated row writes its ordinary destination.
    let fanout = if observation_active { None } else { fanout };
    // The existing dependency-closure preparation omits currents whose first
    // retained contribution already overwrites their destination. Most fixed-
    // source selections therefore need no stage clear at all. Unrestricted
    // selection and all-flow-union retain their existing clearing semantics.
    let requires_contribution_stage_clear = (selected_sector_id.is_some()
        || plan.strategy() == RecurrenceStrategy::AllFlowUnion)
        && (!PREPARED_CLEARS || current_clear_stages.is_none_or(|stages| !stages.is_empty()));
    let mut initialized_contribution_stage = None;
    for (group_index, descriptor) in row_groups.iter().enumerate() {
        if descriptor.role == DirectExecutorRole::Source
            && descriptor.direct_executor_id == DIRECT_NONE_U32
            && plan.strategy() == RecurrenceStrategy::AllFlowUnion
        {
            // The selected union source variants were filled once by
            // `DirectRecurrenceExecutionRuntime` before entering this static
            // contribution/finalization/closure schedule.
            continue;
        }
        if interaction_enabled && let Some(fanout) = fanout {
            match fanout.interactions.action(group_index) {
                DirectInteractionGroupAction::Normal => {}
                DirectInteractionGroupAction::Consumed => continue,
                DirectInteractionGroupAction::Execute(stage_index) => {
                    let (logical_groups, closure_group_index, logical_row_count) =
                        fanout.interactions.logical_groups(stage_index)?;
                    let started = PROFILE.then(Instant::now);
                    let (arena, momenta, parameters, factors) = workspace.raw_views()?;
                    let closure_elapsed = fanout.interactions.execute::<PROFILE>(
                        stage_index,
                        arena,
                        momenta,
                        parameters,
                        factors,
                    )?;
                    if PROFILE {
                        counters.contribution_calls += logical_groups.len() as u64;
                        counters.contribution_rows += logical_row_count;
                        if let Some(closure_group_index) = closure_group_index {
                            let closure = &row_groups[closure_group_index as usize];
                            counters.closure_calls += 1;
                            counters.closure_rows += u64::from(closure.row_count);
                        }
                        let elapsed = started
                            .expect("profiled interaction stage has a start time")
                            .elapsed();
                        timings.contribution += elapsed.saturating_sub(closure_elapsed);
                        timings.closure += closure_elapsed;
                        if let Some(traffic) = traffic.as_deref_mut() {
                            for &logical_group in logical_groups {
                                let logical_group = logical_group as usize;
                                let logical = &row_groups[logical_group];
                                traffic.record_call(logical.row_count, point_count);
                            }
                            if let Some(closure_group_index) = closure_group_index {
                                let closure = &row_groups[closure_group_index as usize];
                                traffic.record_call(closure.row_count, point_count);
                            }
                        }
                    }
                    #[cfg(any(test, feature = "on-the-fly-test-support"))]
                    for &logical_group in logical_groups {
                        let logical_group = logical_group as usize;
                        let logical = &row_groups[logical_group];
                        let start = usize::try_from(logical.row_start).map_err(|_| {
                            RusticolError::integrity(
                                "direct interaction observation row start exceeds usize",
                            )
                        })?;
                        let end =
                            start
                                .checked_add(logical.row_count as usize)
                                .ok_or_else(|| {
                                    RusticolError::integrity(
                                        "direct interaction observation row range overflows usize",
                                    )
                                })?;
                        observe_direct_current_rows(
                            plan,
                            logical,
                            start,
                            end,
                            selected_sector_id,
                            workspace,
                            point_count,
                        )?;
                    }
                    continue;
                }
            }
        }
        // A full topology/contracted schedule executes every authenticated
        // initializer, so its writes replace the stage clear. All-flow-union
        // sources are prepared outside this static row schedule and may leave
        // additive destinations relying on the clear even without a selected
        // sector. Selected execution likewise visits only a logical subset.
        if requires_contribution_stage_clear
            && descriptor.role == DirectExecutorRole::Contribution
            && initialized_contribution_stage != Some(descriptor.stage)
        {
            if PREPARED_CLEARS {
                let stages = current_clear_stages.ok_or_else(|| {
                    RusticolError::integrity("selected recurrence has no prepared stage clears")
                })?;
                if let Ok(stage_index) =
                    stages.binary_search_by_key(&descriptor.stage, |entry| entry.stage)
                {
                    let current_plane_count =
                        u32::try_from(workspace.current_re.len() / workspace.point_stride as usize)
                            .map_err(|_| {
                                RusticolError::integrity(
                                    "selected recurrence current plane count exceeds u32",
                                )
                            })?;
                    for range in stages[stage_index].ranges.iter() {
                        clear_split_active_range(
                            workspace.current_re,
                            workspace.current_im,
                            current_plane_count,
                            workspace.point_stride,
                            point_count,
                            range.component_base,
                            range.component_count,
                            "selected recurrence current arena",
                        )?;
                    }
                }
            } else {
                workspace.clear_current_stage(
                    plan,
                    descriptor.stage,
                    point_count,
                    selected_sector_id,
                )?;
            }
            initialized_contribution_stage = Some(descriptor.stage);
        }
        let start = usize::try_from(descriptor.row_start).map_err(|_| {
            RusticolError::integrity("direct recurrence row-group start exceeds usize")
        })?;
        let count = usize::try_from(descriptor.row_count).map_err(|_| {
            RusticolError::integrity("direct recurrence row-group count exceeds usize")
        })?;
        let end = start.checked_add(count).ok_or_else(|| {
            RusticolError::integrity("direct recurrence row-group range overflows usize")
        })?;
        if descriptor.role == DirectExecutorRole::Contribution
            && descriptor.direct_executor_id == DIRECT_NONE_U32
        {
            let rows = plan.contributions().get(start..end).ok_or_else(|| {
                RusticolError::integrity("certified-reuse contribution row group is out of bounds")
            })?;
            let started = PROFILE.then(Instant::now);
            execute_certified_reuse_rows(rows, workspace, point_count)?;
            #[cfg(any(test, feature = "on-the-fly-test-support"))]
            observe_direct_current_rows(
                plan,
                descriptor,
                start,
                end,
                selected_sector_id,
                workspace,
                point_count,
            )?;
            if PROFILE {
                counters.contribution_calls += 1;
                counters.contribution_rows += u64::from(descriptor.row_count);
                timings.contribution += started
                    .expect("profiled certified-reuse group has a start time")
                    .elapsed();
                if let Some(traffic) = traffic.as_deref_mut() {
                    traffic.record_call(descriptor.row_count, point_count);
                }
            }
            continue;
        }
        if descriptor.role == DirectExecutorRole::Contribution
            && let Some(fanout) = fanout
        {
            let handle = executors.require(
                descriptor.direct_executor_id,
                DirectExecutorRole::Contribution,
            )?;
            let DirectExecutorHandle::Contribution { call, context } = handle else {
                return Err(RusticolError::integrity(
                    "direct fanout executor does not have the contribution role",
                ));
            };
            if PROFILE && let Some(traffic) = traffic.as_deref_mut() {
                traffic.record_call(descriptor.row_count, point_count);
            }
            let started = PROFILE.then(Instant::now);
            let scatter_bytes = execute_contribution_group_with_fanout(
                plan,
                descriptor,
                fanout,
                call,
                context,
                executors.contribution_fanout(descriptor.direct_executor_id),
                workspace,
                point_count,
            )?;
            if PROFILE {
                counters.contribution_calls += 1;
                counters.contribution_rows += u64::from(descriptor.row_count);
                counters.scatter_bytes = counters.scatter_bytes.saturating_add(scatter_bytes);
                timings.contribution += started
                    .expect("profiled direct fanout group has a start time")
                    .elapsed();
            }
            #[cfg(any(test, feature = "on-the-fly-test-support"))]
            observe_direct_current_rows(
                plan,
                descriptor,
                start,
                end,
                selected_sector_id,
                workspace,
                point_count,
            )?;
            continue;
        }
        let (arena, momenta, parameters, factors) = workspace.raw_views()?;
        let handle = executors.require(descriptor.direct_executor_id, descriptor.role)?;
        if PROFILE && let Some(traffic) = traffic.as_deref_mut() {
            traffic.record_call(descriptor.row_count, point_count);
        }
        let started = PROFILE.then(Instant::now);
        let status = unsafe {
            match handle {
                DirectExecutorHandle::Source { call, context } => {
                    let rows = plan.sources().get(start..end).ok_or_else(|| {
                        RusticolError::integrity(
                            "direct recurrence source row group is out of bounds",
                        )
                    })?;
                    if PROFILE {
                        counters.source_calls += 1;
                        counters.source_rows += u64::from(descriptor.row_count);
                    }
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        descriptor.row_count,
                        point_count,
                    )
                }
                DirectExecutorHandle::Contribution { call, context } => {
                    let rows = plan.contributions().get(start..end).ok_or_else(|| {
                        RusticolError::integrity(
                            "direct recurrence contribution row group is out of bounds",
                        )
                    })?;
                    if PROFILE {
                        counters.contribution_calls += 1;
                        counters.contribution_rows += u64::from(descriptor.row_count);
                    }
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        descriptor.row_count,
                        point_count,
                    )
                }
                DirectExecutorHandle::Finalization { call, context } => {
                    let rows = plan.finalizations().get(start..end).ok_or_else(|| {
                        RusticolError::integrity(
                            "direct recurrence finalization row group is out of bounds",
                        )
                    })?;
                    if PROFILE {
                        counters.finalization_calls += 1;
                        counters.finalization_rows += u64::from(descriptor.row_count);
                    }
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        descriptor.row_count,
                        point_count,
                    )
                }
                DirectExecutorHandle::Closure { call, context } => {
                    let rows = plan.closures().get(start..end).ok_or_else(|| {
                        RusticolError::integrity(
                            "direct recurrence closure row group is out of bounds",
                        )
                    })?;
                    if PROFILE {
                        counters.closure_calls += 1;
                        counters.closure_rows += u64::from(descriptor.row_count);
                    }
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        descriptor.row_count,
                        point_count,
                    )
                }
            }
        };
        if PROFILE {
            let elapsed = started
                .expect("profiled direct row group has a start time")
                .elapsed();
            match descriptor.role {
                DirectExecutorRole::Source => timings.source += elapsed,
                DirectExecutorRole::Contribution => timings.contribution += elapsed,
                DirectExecutorRole::Finalization => timings.finalization += elapsed,
                DirectExecutorRole::Closure => timings.closure += elapsed,
            }
        }
        check_status(descriptor.role, descriptor.direct_executor_id, status)?;
        #[cfg(any(test, feature = "on-the-fly-test-support"))]
        observe_direct_current_rows(
            plan,
            descriptor,
            start,
            end,
            selected_sector_id,
            workspace,
            point_count,
        )?;
    }
    Ok(())
}

#[inline(always)]
fn direct_interaction_execution_enabled<const ALLOW_INTERACTION_REWRITE: bool>(
    strategy: RecurrenceStrategy,
    authoritative_groups: &[DirectRowGroupDescriptor],
    requested_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: Option<u32>,
    point_count: u32,
) -> bool {
    ALLOW_INTERACTION_REWRITE
        && point_count == 1
        && selected_sector_id.is_none()
        && strategy != RecurrenceStrategy::AllFlowUnion
        && requested_groups.len() == authoritative_groups.len()
        && std::ptr::eq(requested_groups.as_ptr(), authoritative_groups.as_ptr())
}

fn execute_direct_interaction_stage_views<const PROFILE: bool>(
    stage: &DirectInteractionStage,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
) -> RusticolResult<Duration> {
    let terminal = stage
        .terminal
        .as_ref()
        .map(|terminal| DirectInteractionTerminalBatch {
            partner_component_base: terminal.partner_component_base,
            component_factor_start: terminal.component_factor_start,
            amplitude_destination_count: terminal.amplitude_destination_count,
            closures: &terminal.closures,
        });
    unsafe {
        (stage.call)(
            stage.contexts,
            arena,
            momenta,
            parameters,
            factors,
            DirectInteractionBundleBatch {
                anchors: &stage.anchors,
                outputs: &stage.outputs,
                tensor_outputs: &stage.tensor_outputs,
                terminal,
                requirements: stage.requirements,
                atv_before_color: stage.atv_before_color,
            },
        )?;
    }
    let closure_started = (PROFILE && terminal.is_some()).then(Instant::now);
    if let Some(terminal) = terminal {
        unsafe {
            crate::engine::execute_interaction_terminal_closures(arena, factors, terminal)?;
        }
    }
    Ok(closure_started.map_or(Duration::ZERO, |started| started.elapsed()))
}

#[allow(clippy::too_many_arguments)]
fn execute_contribution_group_with_fanout(
    plan: &DirectRecurrencePlan,
    descriptor: &DirectRowGroupDescriptor,
    fanout: &DirectContributionFanoutProgram,
    call: DirectContributionExecutor,
    context: *const c_void,
    fused: Option<DirectContributionFanoutExecutorHandle>,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<u64> {
    let current_plane_count = u32::try_from(
        workspace.current_re.len() / workspace.point_stride as usize,
    )
    .map_err(|_| RusticolError::integrity("direct fanout current plane count exceeds u32"))?;
    let scratch_end = fanout
        .scratch_component_base
        .checked_add(fanout.scratch_component_count)
        .ok_or_else(|| RusticolError::integrity("direct fanout scratch range overflows u32"))?;
    if fanout.scratch_component_base != plan.current_arena_components()
        || scratch_end > current_plane_count
    {
        return Err(RusticolError::integrity(
            "direct fanout scratch range is outside its private current arena",
        ));
    }
    if fanout.needs_unit_factor {
        let factor = fanout.unit_factor_id as usize;
        if workspace
            .factors_re
            .get(factor)
            .map(|value| value.to_bits())
            != Some(1.0_f64.to_bits())
            || workspace
                .factors_im
                .get(factor)
                .map(|value| value.to_bits())
                != Some(0.0_f64.to_bits())
        {
            return Err(RusticolError::integrity(
                "direct fanout private exact factor is not complex unity",
            ));
        }
    }
    let steps = fanout.group_steps(descriptor)?;
    let (arena, momenta, parameters, factors) = workspace.raw_views()?;
    let mut scatter_bytes = 0_u64;
    for step in steps {
        match *step {
            DirectContributionFanoutStep::Direct {
                row_start,
                row_count,
            } => {
                let start = usize::try_from(row_start).map_err(|_| {
                    RusticolError::integrity("direct fanout direct-row start exceeds usize")
                })?;
                let end = start.checked_add(row_count as usize).ok_or_else(|| {
                    RusticolError::integrity("direct fanout direct-row range overflows usize")
                })?;
                let rows = plan.contributions().get(start..end).ok_or_else(|| {
                    RusticolError::integrity("direct fanout direct-row range is out of bounds")
                })?;
                let status = unsafe {
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        row_count,
                        point_count,
                    )
                };
                check_status(
                    DirectExecutorRole::Contribution,
                    descriptor.direct_executor_id,
                    status,
                )?;
            }
            DirectContributionFanoutStep::Fanout {
                row_count,
                representative_start,
                representative_count,
                run_start,
                run_count,
                destination_component_count,
                ..
            } => {
                if point_count == 1
                    && let Some(fused) = fused
                {
                    let start = usize::try_from(run_start).map_err(|_| {
                        RusticolError::integrity("direct fanout run start exceeds usize")
                    })?;
                    let end = start.checked_add(run_count as usize).ok_or_else(|| {
                        RusticolError::integrity("direct fanout run range overflows usize")
                    })?;
                    let runs = fanout.runs.get(start..end).ok_or_else(|| {
                        RusticolError::integrity("direct fanout run range is out of bounds")
                    })?;
                    debug_assert_eq!(
                        fused.destination_component_count,
                        destination_component_count
                    );
                    unsafe {
                        (fused.call)(
                            fused.context,
                            arena,
                            momenta,
                            parameters,
                            factors,
                            DirectContributionFanoutBatch {
                                rows: plan.contributions(),
                                runs,
                                bundles: &fanout.bundles,
                            },
                        )
                    }?;
                    continue;
                }
                let representative_start = usize::try_from(representative_start).map_err(|_| {
                    RusticolError::integrity("direct fanout representative start exceeds usize")
                })?;
                let representative_end = representative_start
                    .checked_add(representative_count as usize)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct fanout representative range overflows usize",
                        )
                    })?;
                let representatives = fanout
                    .representatives
                    .get(representative_start..representative_end)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct fanout representative range is out of bounds",
                        )
                    })?;
                let status = unsafe {
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        representatives.as_ptr(),
                        representative_count,
                        point_count,
                    )
                };
                check_status(
                    DirectExecutorRole::Contribution,
                    descriptor.direct_executor_id,
                    status,
                )?;
                let run_start = usize::try_from(run_start).map_err(|_| {
                    RusticolError::integrity("direct fanout run start exceeds usize")
                })?;
                let run_end = run_start.checked_add(run_count as usize).ok_or_else(|| {
                    RusticolError::integrity("direct fanout run range overflows usize")
                })?;
                let batch_runs = fanout.runs.get(run_start..run_end).ok_or_else(|| {
                    RusticolError::integrity("direct fanout run range is out of bounds")
                })?;
                debug_assert_eq!(run_count, representative_count);
                debug_assert_eq!(
                    batch_runs
                        .iter()
                        .map(|run| u64::from(run.row_count))
                        .sum::<u64>(),
                    u64::from(row_count)
                );
                for run in batch_runs {
                    scatter_contribution_fanout_run(
                        plan,
                        fanout,
                        *run,
                        destination_component_count,
                        workspace,
                        point_count,
                    )?;
                }
                scatter_bytes = scatter_bytes.saturating_add(
                    u64::from(row_count)
                        .saturating_mul(u64::from(destination_component_count))
                        .saturating_mul(u64::from(point_count))
                        .saturating_mul(4 * size_of::<f64>() as u64),
                );
            }
        }
    }
    Ok(scatter_bytes)
}

#[derive(Clone, Copy)]
pub(crate) enum DirectContributionFanoutScale {
    Unity,
    NegativeUnity,
    Real(f64),
    Imaginary(f64),
    Complex(f64, f64),
}

impl DirectContributionFanoutScale {
    #[inline(always)]
    pub(crate) fn new(real: f64, imaginary: f64) -> Self {
        if imaginary == 0.0 {
            if real == 1.0 {
                Self::Unity
            } else if real == -1.0 {
                Self::NegativeUnity
            } else {
                Self::Real(real)
            }
        } else if real == 0.0 {
            Self::Imaginary(imaginary)
        } else {
            Self::Complex(real, imaginary)
        }
    }

    #[inline(always)]
    pub(crate) fn apply(self, real: f64, imaginary: f64) -> (f64, f64) {
        match self {
            Self::Unity => (real, imaginary),
            Self::NegativeUnity => (-real, -imaginary),
            Self::Real(scale) => (real * scale, imaginary * scale),
            Self::Imaginary(scale) => (-imaginary * scale, real * scale),
            Self::Complex(scale_re, scale_im) => (
                real * scale_re - imaginary * scale_im,
                real * scale_im + imaginary * scale_re,
            ),
        }
    }
}

fn scatter_contribution_fanout_run(
    plan: &DirectRecurrencePlan,
    fanout: &DirectContributionFanoutProgram,
    run: DirectContributionFanoutRun,
    destination_component_count: u32,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let stride = workspace.point_stride as usize;
    let active_points = point_count as usize;
    let source_end = run
        .scratch_component_base
        .checked_add(destination_component_count)
        .ok_or_else(|| RusticolError::integrity("direct fanout source range overflows u32"))?;
    let current_plane_count = u32::try_from(workspace.current_re.len() / stride)
        .map_err(|_| RusticolError::integrity("direct fanout current plane count exceeds u32"))?;
    if run.scratch_component_base < fanout.scratch_component_base
        || source_end > current_plane_count
    {
        return Err(RusticolError::integrity(
            "direct fanout source range is outside its private current arena",
        ));
    }
    let bundle_start = usize::try_from(run.bundle_start)
        .map_err(|_| RusticolError::integrity("direct fanout bundle start exceeds usize"))?;
    let bundle_end = bundle_start
        .checked_add(run.bundle_count as usize)
        .ok_or_else(|| RusticolError::integrity("direct fanout bundle range overflows usize"))?;
    let bundles = fanout
        .bundles
        .get(bundle_start..bundle_end)
        .ok_or_else(|| RusticolError::integrity("direct fanout bundle range is out of bounds"))?;
    debug_assert_eq!(
        bundles
            .iter()
            .map(|bundle| u64::from(bundle.row_count))
            .sum::<u64>(),
        u64::from(run.row_count)
    );

    let current_re = workspace.current_re.as_mut_ptr();
    let current_im = workspace.current_im.as_mut_ptr();
    for bundle in bundles {
        let start = usize::try_from(bundle.row_start).map_err(|_| {
            RusticolError::integrity("direct fanout bundle row start exceeds usize")
        })?;
        let end = start
            .checked_add(bundle.row_count as usize)
            .ok_or_else(|| {
                RusticolError::integrity("direct fanout bundle row range overflows usize")
            })?;
        let rows = plan.contributions().get(start..end).ok_or_else(|| {
            RusticolError::integrity("direct fanout bundle row range is out of bounds")
        })?;
        let scale = if bundle.effective_factor_id_or_sentinel == DIRECT_NONE_U32 {
            DirectContributionFanoutScale::Unity
        } else {
            let factor = bundle.effective_factor_id_or_sentinel as usize;
            DirectContributionFanoutScale::new(
                *workspace.factors_re.get(factor).ok_or_else(|| {
                    RusticolError::integrity("direct fanout scatter factor is out of bounds")
                })?,
                *workspace.factors_im.get(factor).ok_or_else(|| {
                    RusticolError::integrity("direct fanout scatter factor is out of bounds")
                })?,
            )
        };
        for component in 0..destination_component_count {
            let source_start = (run.scratch_component_base as usize + component as usize) * stride;
            for point in 0..active_points {
                let source = source_start + point;
                // SAFETY: the source run is checked against the runtime arena
                // above. The destination rows and their full component spans
                // were authenticated by `DirectRecurrencePlan` and matched to
                // this executor's destination shape when the fanout program
                // was built. `workspace.validate` established point < stride.
                let (scaled_re, scaled_im) =
                    unsafe { scale.apply(*current_re.add(source), *current_im.add(source)) };
                for row in rows {
                    let destination =
                        (row.destination_component_base as usize + component as usize) * stride
                            + point;
                    // SAFETY: see the source/destination proof immediately
                    // above. Scratch components form a private suffix, so none
                    // of these writes can alias the representative source.
                    unsafe {
                        if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
                            *current_re.add(destination) = scaled_re;
                            *current_im.add(destination) = scaled_im;
                        } else {
                            *current_re.add(destination) += scaled_re;
                            *current_im.add(destination) += scaled_im;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[allow(clippy::too_many_arguments)]
fn observe_direct_current_rows(
    plan: &DirectRecurrencePlan,
    descriptor: &DirectRowGroupDescriptor,
    start: usize,
    end: usize,
    selected_sector_id: Option<u32>,
    workspace: &DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let active = DIRECT_CURRENT_OBSERVATION.with(|slot| slot.borrow().is_some());
    if !active || descriptor.role == DirectExecutorRole::Closure {
        return Ok(());
    }

    let affected = match descriptor.role {
        DirectExecutorRole::Source => plan
            .currents()
            .iter()
            .filter(|current| {
                current.source_row_or_sentinel != DIRECT_NONE_U32
                    && (start..end).contains(&(current.source_row_or_sentinel as usize))
            })
            .map(|current| current.semantic_current_id)
            .collect::<BTreeSet<_>>(),
        DirectExecutorRole::Contribution => {
            let destinations = plan
                .contributions()
                .get(start..end)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation contribution range is out of bounds",
                    )
                })?
                .iter()
                .map(|row| row.destination_component_base)
                .collect::<BTreeSet<_>>();
            plan.currents()
                .iter()
                .filter(|current| {
                    current.node_kind == DirectNodeKind::Current
                        && current.stage == descriptor.stage
                        && destinations.iter().any(|destination| {
                            *destination >= current.component_base
                                && *destination
                                    < current.component_base + u32::from(current.component_count)
                        })
                })
                .map(|current| current.semantic_current_id)
                .collect::<BTreeSet<_>>()
        }
        DirectExecutorRole::Finalization => plan
            .currents()
            .iter()
            .filter(|current| {
                current.finalization_row_or_sentinel != DIRECT_NONE_U32
                    && (start..end).contains(&(current.finalization_row_or_sentinel as usize))
            })
            .map(|current| current.semantic_current_id)
            .collect::<BTreeSet<_>>(),
        DirectExecutorRole::Closure => BTreeSet::new(),
    };
    if affected.is_empty() {
        return Ok(());
    }

    DIRECT_CURRENT_OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let observation = slot.as_mut().ok_or_else(|| {
            RusticolError::integrity("direct current observation disappeared during execution")
        })?;
        if observation.plan_semantic_digest != plan.semantic_digest()
            || observation.runtime_layout_digest != plan.runtime_layout_digest()
            || observation.selected_sector_id != selected_sector_id
            || observation.point_count != point_count
        {
            return Err(RusticolError::integrity(
                "direct current observation does not match this plan execution",
            ));
        }
        let stride = workspace.point_stride as usize;
        let active_points = point_count as usize;
        for semantic_current_id in affected {
            let current = plan
                .currents()
                .get(semantic_current_id as usize)
                .filter(|current| current.semantic_current_id == semantic_current_id)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation semantic current is absent",
                    )
                })?;
            if let Some(sector_id) = selected_sector_id
                && !plan.selector_domain_contains(current.selector_domain_id, sector_id)?
            {
                continue;
            }
            let mut values = Vec::with_capacity(
                usize::from(current.component_count)
                    .checked_mul(active_points)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct current observation value count overflows usize",
                        )
                    })?,
            );
            for component in 0..usize::from(current.component_count) {
                let plane = (current.component_base as usize)
                    .checked_add(component)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct current observation component range overflows usize",
                        )
                    })?;
                let begin = plane.checked_mul(stride).ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation arena range overflows usize",
                    )
                })?;
                let finish = begin.checked_add(active_points).ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation arena range overflows usize",
                    )
                })?;
                let real = workspace.current_re.get(begin..finish).ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation real range is out of bounds",
                    )
                })?;
                let imaginary = workspace.current_im.get(begin..finish).ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation imaginary range is out of bounds",
                    )
                })?;
                values.extend(real.iter().copied().zip(imaginary.iter().copied()));
            }
            observation.currents.insert(
                semantic_current_id,
                DirectObservedCurrentValue {
                    descriptor: *current,
                    values,
                },
            );
        }
        Ok(())
    })
}

pub(crate) fn execute_certified_reuse_rows(
    rows: &[DirectContributionRow],
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let stride = workspace.point_stride as usize;
    let point_count = point_count as usize;
    for (index, row) in rows.iter().enumerate() {
        if row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0
            || row.parent1_component_base_or_sentinel != DIRECT_NONE_U32
            || row.parent1_momentum_form_id_or_sentinel != DIRECT_NONE_U32
        {
            return Err(RusticolError::integrity(format!(
                "certified-reuse row {index} has an invalid relation encoding"
            )));
        }
        let component_count = row.parent0_momentum_form_id as usize;
        if component_count == 0 {
            return Err(RusticolError::integrity(format!(
                "certified-reuse row {index} has zero components"
            )));
        }
        let factor_index = row.exact_factor_id as usize;
        let factor_re = *workspace
            .factors_re
            .get(factor_index)
            .ok_or_else(|| RusticolError::integrity("certified-reuse factor is out of bounds"))?;
        let factor_im = *workspace
            .factors_im
            .get(factor_index)
            .ok_or_else(|| RusticolError::integrity("certified-reuse factor is out of bounds"))?;
        for component in 0..component_count {
            let source_plane = row.parent0_component_base as usize + component;
            let destination_plane = row.destination_component_base as usize + component;
            let source_start = source_plane.checked_mul(stride).ok_or_else(|| {
                RusticolError::integrity("certified-reuse source range overflows usize")
            })?;
            let destination_start = destination_plane.checked_mul(stride).ok_or_else(|| {
                RusticolError::integrity("certified-reuse destination range overflows usize")
            })?;
            let source_end = source_start.checked_add(point_count).ok_or_else(|| {
                RusticolError::integrity("certified-reuse source range overflows usize")
            })?;
            let destination_end = destination_start.checked_add(point_count).ok_or_else(|| {
                RusticolError::integrity("certified-reuse destination range overflows usize")
            })?;
            if source_end > workspace.current_re.len()
                || source_end > workspace.current_im.len()
                || destination_end > workspace.current_re.len()
                || destination_end > workspace.current_im.len()
            {
                return Err(RusticolError::integrity(
                    "certified-reuse source or destination range is out of bounds",
                ));
            }
            // Certified reuse is an exact algebraic scale-copy. In
            // particular, an exact zero factor overwrites with zero even when
            // the representative aliases a freshly initialized destination.
            // Do not evaluate `NaN * 0.0`: without the former unconditional
            // stage clear that would preserve stale arena payloads instead of
            // materializing the certified zero current.
            if factor_re == 0.0 && factor_im == 0.0 {
                workspace.current_re[destination_start..destination_end].fill(0.0);
                workspace.current_im[destination_start..destination_end].fill(0.0);
                continue;
            }
            for point in 0..point_count {
                let source_index = source_start + point;
                let destination_index = destination_start + point;
                let source_re = *workspace.current_re.get(source_index).ok_or_else(|| {
                    RusticolError::integrity("certified-reuse source real range is out of bounds")
                })?;
                let source_im = *workspace.current_im.get(source_index).ok_or_else(|| {
                    RusticolError::integrity(
                        "certified-reuse source imaginary range is out of bounds",
                    )
                })?;
                *workspace
                    .current_re
                    .get_mut(destination_index)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "certified-reuse destination real range is out of bounds",
                        )
                    })? = source_re * factor_re - source_im * factor_im;
                *workspace
                    .current_im
                    .get_mut(destination_index)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "certified-reuse destination imaginary range is out of bounds",
                        )
                    })? = source_re * factor_im + source_im * factor_re;
            }
        }
    }
    Ok(())
}

pub(crate) fn check_status(
    role: DirectExecutorRole,
    executor_id: u32,
    status: c_int,
) -> RusticolResult<()> {
    if status == DIRECT_STATUS_OK {
        return Ok(());
    }
    if let Some(detail) = take_direct_executor_error_detail() {
        Err(RusticolError::with_kind(
            detail.kind(),
            format!(
                "direct recurrence {role:?} executor {executor_id} failed: {}",
                detail.message()
            ),
        ))
    } else {
        Err(RusticolError::evaluation(format!(
            "direct recurrence {role:?} executor {executor_id} returned status {status}"
        )))
    }
}
