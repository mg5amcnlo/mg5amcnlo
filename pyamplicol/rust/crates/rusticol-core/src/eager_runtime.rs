// SPDX-License-Identifier: 0BSD

use crate::{RusticolError, RusticolResult};
use num_complex::Complex;
use std::time::Duration;

pub type EagerComplex64 = Complex<f64>;

pub const DEFAULT_EAGER_POINT_TILE_SIZE: usize = 1024;
pub const DEFAULT_EAGER_WORKSPACE_MIB: usize = 256;
pub const EAGER_HOMOGENEOUS_LINEAR_CURRENT_PROOF: &str =
    "prepared-kernel-homogeneous-complex-linear-current-v1";
pub const EAGER_INDEPENDENT_BLOCK_SIZE: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EagerKernelRole {
    Vertex,
    Finalization,
    Closure,
}

/// One prepared evaluator input in its exact, deterministic parameter order.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EagerKernelInput {
    FirstCurrentComponent(u32),
    SecondCurrentComponent(u32),
    FirstMomentumComponent(u32),
    SecondMomentumComponent(u32),
    CouplingReal,
    CouplingImag,
    ModelParameter(u32),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerKernelSpec {
    pub kernel_id: u32,
    pub role: EagerKernelRole,
    pub inputs: Vec<EagerKernelInput>,
    pub output_component_count: u32,
    pub homogeneous_linear_first_current: bool,
    /// Prepared evaluator width for independent invocation blocks.
    ///
    /// A value greater than one is valid only for vertex kernels whose
    /// prepared pack contains the correspondingly widened evaluator. The
    /// scheduler retains the scalar evaluator for incomplete tails.
    pub independent_block_size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerPlanDimensions {
    pub value_slot_component_counts: Vec<u32>,
    pub momentum_slot_component_counts: Vec<u32>,
    pub current_component_counts: Vec<u32>,
    pub parameter_count: u32,
    pub amplitude_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EagerDirectClosureSpec {
    pub closure_index: u32,
    pub coefficients: Vec<EagerComplex64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerReductionGroup {
    pub coherent_group_id: u32,
    pub amplitude_indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EagerReductionEntry {
    pub left_group_index: u32,
    pub right_group_index: u32,
    pub coefficient: EagerComplex64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EagerPlanDefinition {
    pub dimensions: EagerPlanDimensions,
    pub kernels: Vec<EagerKernelSpec>,
    pub direct_closures: Vec<EagerDirectClosureSpec>,
    pub reduction_groups: Vec<EagerReductionGroup>,
    pub reduction_entries: Vec<EagerReductionEntry>,
}

#[derive(Clone, Copy, Debug)]
pub struct EagerStagePayload<'a> {
    pub stage_index: u32,
    pub invocations: &'a [u8],
    pub attachments: &'a [u8],
    pub finalizations: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct EagerSelectorStagePayload<'a> {
    pub stage_index: u32,
    pub invocation_domains: &'a [u8],
    pub attachment_domains: &'a [u8],
    pub unpropagated_finalization_domains: &'a [u8],
    pub propagated_finalization_domains: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct EagerSelectorPayloads<'a> {
    pub domains: &'a [u8],
    pub domain_group_ids: &'a [u8],
    pub stages: &'a [EagerSelectorStagePayload<'a>],
    pub closure_domains: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct EagerPlanPayloads<'a> {
    pub couplings: &'a [u8],
    pub stages: &'a [EagerStagePayload<'a>],
    pub closures: &'a [u8],
    pub selector_domains: Option<EagerSelectorPayloads<'a>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerRuntimeOptions {
    pub point_tile_size: usize,
    /// Batch workspace target; one executable point may require more memory.
    pub workspace_bytes: usize,
}

impl EagerRuntimeOptions {
    pub fn from_mib(point_tile_size: usize, workspace_mib: usize) -> RusticolResult<Self> {
        let workspace_bytes = workspace_mib.checked_mul(1024 * 1024).ok_or_else(|| {
            RusticolError::invalid_argument("eager workspace size overflows bytes")
        })?;
        Ok(Self {
            point_tile_size,
            workspace_bytes,
        })
    }
}

impl Default for EagerRuntimeOptions {
    fn default() -> Self {
        Self {
            point_tile_size: DEFAULT_EAGER_POINT_TILE_SIZE,
            workspace_bytes: DEFAULT_EAGER_WORKSPACE_MIB * 1024 * 1024,
        }
    }
}

pub struct EagerKernelCall<'a> {
    pub kernel_id: u32,
    pub independent_block_size: u32,
    pub lane_count: usize,
    pub input_component_count: usize,
    pub output_component_count: usize,
    pub inputs: &'a [EagerComplex64],
    pub outputs: &'a mut [EagerComplex64],
}

/// Evaluates row-major packets whose lanes are ordered by invocation, then point.
///
/// The scheduler sorts invocations stably by kernel id while loading a plan. Each packet
/// therefore contains one kernel only; within that packet, all points for the first
/// invocation precede all points for the next invocation. Input components follow the
/// exact order in [`EagerKernelSpec::inputs`], so component `c` of lane `l` is stored at
/// `l * input_component_count + c`. Outputs use the same lane-major row layout.
pub trait EagerKernelBackend {
    fn evaluate_batch(&mut self, call: EagerKernelCall<'_>) -> RusticolResult<()>;
}

/// Opt-in wall-clock accounting for one eager scheduler evaluation.
///
/// Every field is an aggregate over all point tiles. The phases are mutually
/// exclusive, so [`Self::accounted`] can be compared directly with
/// [`Self::total`]. Timer bookkeeping and validation account for the small
/// unclassified remainder.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EagerExecutionProfile {
    pub(crate) initialize: Duration,
    pub(crate) gather: Duration,
    pub(crate) kernel_call: Duration,
    pub(crate) invocation_scatter: Duration,
    pub(crate) finalization: Duration,
    pub(crate) closure: Duration,
    pub(crate) reduction: Duration,
    pub(crate) copy_out: Duration,
    pub(crate) total: Duration,
    pub(crate) backend_call_count: u64,
    /// P-kernel output scratch reads and writes wholly inside the eager arena.
    /// This is diagnostic traffic, never arena-boundary materialization.
    pub(crate) internal_scratch_bytes: u64,
    /// Point-independent coupling and parameter plane refreshes inside the
    /// persistent eager arena. This is not arena-boundary traffic.
    pub(crate) internal_broadcast_bytes: u64,
}

impl EagerExecutionProfile {
    pub(crate) fn accounted(self) -> Duration {
        self.initialize
            + self.gather
            + self.kernel_call
            + self.invocation_scatter
            + self.finalization
            + self.closure
            + self.reduction
            + self.copy_out
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EagerScheduleAuditRow {
    pub(crate) stage_index: Option<u32>,
    pub(crate) role: &'static str,
    pub(crate) kernel_id: Option<u32>,
    pub(crate) call_count: usize,
    pub(crate) row_count: usize,
    pub(crate) destination_count: usize,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod direct_execution_arena;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
mod direct_invocation_arena;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub(crate) use direct_execution_arena::EagerDirectExecutionRuntime;
#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
pub(crate) use direct_invocation_arena::{
    EagerDirectPreparedApplication, EagerDirectPreparedKernel,
};
#[cfg(all(test, any(feature = "f64-compiled", feature = "f64-symjit")))]
pub(crate) use direct_invocation_arena::{
    run_retained_multistage_oracle, select_retained_multistage_candidate,
};
mod execute;
mod plan;
mod plan_v3;
mod profile;
mod runtime;

#[cfg(test)]
mod plan_v3_tests;

pub use plan::EagerExecutionPlan;
pub(crate) use plan_v3::EagerPlanV3Sections;
pub use runtime::EagerExecutionRuntime;
