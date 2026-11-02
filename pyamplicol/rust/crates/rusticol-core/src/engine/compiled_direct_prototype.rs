// SPDX-License-Identifier: 0BSD

//! Production compiled Direct-Arena execution.
//!
//! This module consumes the existing stage/chunk manifests, lowers each
//! already-fused SymJIT O3 leaf to DirectApplication v3 at load time, and
//! executes the unchanged leaf schedule against persistent component-major
//! planes. Canonical manifest component IDs are cold-bound to a stage-granular
//! physical layout: values with disjoint whole-stage lifetimes may reuse a
//! plane while every Arena-v1 stage retains its no-alias contract.

#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[cfg(feature = "f64-symjit")]
use std::path::PathBuf;
use std::ptr;

#[cfg(feature = "f64-compiled")]
use super::evaluator::native_compiled_direct::{
    BoundNativeCompiledDirectStage, LoadedNativeCompiledDirectStage,
    NativeCompiledDirectArenaPlane, NativeCompiledDirectOutputBinding,
    NativeCompiledDirectPlaneBinding, NativeCompiledDirectScalarBinding,
};
#[cfg(feature = "f64-symjit")]
use super::evaluator::symjit_compiled_direct::{
    BoundSymjitCompiledDirectStage, CompiledDirectArenaPlane, CompiledDirectOutputBinding,
    CompiledDirectPlaneBinding, CompiledDirectScalarBinding, CompiledDirectSourceInputBinding,
    LoadedSymjitCompiledDirectStage,
};
#[cfg(feature = "f64-symjit")]
use super::evaluator::{SymjitApplicationMetadata, validate_manifest_metadata};
use super::sources::{ResolvedSourceWavefunction, RuntimeSourceState};
use super::*;
#[cfg(test)]
use crate::direct_arena::checked_aligned_point_stride;
use crate::direct_arena::{
    AlignedF64Buffer, DirectAmplitudePlanes, DirectArenaAllocationCounters,
    DirectArenaTrafficCounters, DirectArenaWorkspace, DirectMomentumView, DirectParameterView,
    deterministic_point_tile_size_with_cache_footprint,
};

// Fused applications repeatedly traverse many split-complex parent/output
// planes. Keeping a small, fixed power-of-two point tile preserves that
// cross-stage locality on both cache-rich and cache-poor CPUs without an
// online host-specific tuner.
const COMPILED_DIRECT_LOCALITY_POINT_CAP: u32 = 32;
const COMPILED_DIRECT_CACHE_TARGET_BYTES: usize = 4 * 1024 * 1024;

fn compiled_direct_cache_target_bytes() -> usize {
    #[cfg(test)]
    if let Some(value) = std::env::var_os("RUSTICOL_TEST_COMPILED_DIRECT_CACHE_TARGET_BYTES") {
        let value = value
            .to_str()
            .expect("compiled Direct test cache target must be UTF-8");
        if value == "hard-budget" {
            return usize::MAX;
        }
        return value
            .parse()
            .expect("compiled Direct test cache target must be bytes or hard-budget");
    }
    COMPILED_DIRECT_CACHE_TARGET_BYTES
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompiledDirectPrototypeTraffic {
    /// Direct source-wavefunction writes into canonical current planes.
    pub(crate) source_fill_bytes: u64,
    /// Direct crossed-momentum writes into canonical momentum planes.
    pub(crate) momentum_fill_bytes: u64,
    /// Point-independent model-scalar writes.
    pub(crate) parameter_fill_bytes: u64,
    /// Internal persistent P-kernel broadcast-plane refresh traffic.
    pub(crate) scalar_broadcast_fill_bytes: u64,
    /// Stale-safety clears of amplitude planes not overwritten by a schedule.
    pub(crate) amplitude_clear_bytes: u64,
    /// Developer-oracle row-major to arena boundary traffic.
    pub(crate) boundary_input_bytes: u64,
    /// Explicit developer-only current extraction traffic.
    pub(crate) boundary_current_output_bytes: u64,
    /// Explicit developer-only amplitude extraction traffic.
    pub(crate) boundary_amplitude_output_bytes: u64,
    /// Hot compiled-leaf calls. Its forbidden traffic byte fields must remain
    /// zero even though call/row/point counts advance structurally.
    pub(crate) leaf: DirectArenaTrafficCounters,
}

#[derive(Clone, Debug)]
struct DirectLeafPlan {
    input_current_components: Box<[usize]>,
    output_current_components: Box<[usize]>,
    output_amplitude_components: Box<[usize]>,
    scalar_values_per_point: usize,
    persistent_scalar_values_per_point: usize,
}

/// Call-local memo for correlated spin assignments at identical momenta and
/// model parameters. It is never read or allocated by ordinary evaluation.
#[derive(Default)]
pub(crate) struct CorrelatedStageMemo {
    leaves: Vec<Option<CorrelatedLeafMemo>>,
    #[cfg(test)]
    reused_leaves: usize,
}

struct CorrelatedLeafMemo {
    inputs: Vec<u64>,
    current_outputs: Vec<[f64; 2]>,
    amplitude_outputs: Vec<[f64; 2]>,
}

// Leaves are cold-built and kept inline to avoid a separate allocation and
// indirection for each warmed stage call.
#[allow(clippy::large_enum_variant)]
enum LoadedCompiledDirectLeaf {
    #[cfg(feature = "f64-symjit")]
    Symjit(LoadedSymjitCompiledDirectStage),
    #[cfg(feature = "f64-compiled")]
    Native(LoadedNativeCompiledDirectStage),
}

#[allow(clippy::large_enum_variant)]
enum BoundCompiledDirectLeaf {
    #[cfg(feature = "f64-symjit")]
    Symjit(BoundSymjitCompiledDirectStage),
    #[cfg(feature = "f64-compiled")]
    Native(BoundNativeCompiledDirectStage),
}

impl LoadedCompiledDirectLeaf {
    fn persistent_scalar_values_per_point(&self) -> usize {
        match self {
            #[cfg(feature = "f64-symjit")]
            Self::Symjit(leaf) => leaf.broadcast_plane_count(),
            #[cfg(feature = "f64-compiled")]
            Self::Native(_) => 0,
        }
    }

    unsafe fn bind(
        self,
        arena: crate::direct_arena::DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        _zero_plane: &AlignedF64Buffer,
        current_component_map: &[u32],
    ) -> RusticolResult<BoundCompiledDirectLeaf> {
        match self {
            #[cfg(feature = "f64-symjit")]
            Self::Symjit(leaf) => {
                // SAFETY: forwarded from the engine's fixed-allocation bind.
                unsafe {
                    leaf.bind_with_current_map(
                        arena,
                        momenta,
                        parameters,
                        _zero_plane,
                        current_component_map,
                    )
                }
                .map(BoundCompiledDirectLeaf::Symjit)
            }
            #[cfg(feature = "f64-compiled")]
            Self::Native(leaf) => {
                // SAFETY: forwarded from the engine's fixed-allocation bind.
                unsafe {
                    leaf.bind_with_current_map(arena, momenta, parameters, current_component_map)
                }
                .map(BoundCompiledDirectLeaf::Native)
            }
        }
    }
}

impl BoundCompiledDirectLeaf {
    fn allocation_counters(&self) -> DirectArenaAllocationCounters {
        match self {
            #[cfg(feature = "f64-symjit")]
            Self::Symjit(leaf) => leaf.allocation_counters(),
            #[cfg(feature = "f64-compiled")]
            Self::Native(_) => DirectArenaAllocationCounters::default(),
        }
    }

    fn evaluate(&mut self, point_start: u32, point_count: u32) -> RusticolResult<u64> {
        match self {
            #[cfg(feature = "f64-symjit")]
            Self::Symjit(leaf) => leaf.evaluate(point_start, point_count),
            #[cfg(feature = "f64-compiled")]
            Self::Native(leaf) => {
                leaf.evaluate(point_start, point_count)?;
                Ok(0)
            }
        }
    }
}

struct LoadedStage {
    leaves: Vec<LoadedCompiledDirectLeaf>,
    leaf_plans: Vec<DirectLeafPlan>,
}

struct BoundStage {
    leaves: Vec<BoundCompiledDirectLeaf>,
    leaf_plans: Vec<DirectLeafPlan>,
}

const UNMAPPED_CURRENT_COMPONENT: u32 = u32::MAX;

#[derive(Clone, Debug)]
struct CompiledDirectCurrentLayout {
    canonical_to_physical: Box<[u32]>,
    physical_component_count: u32,
    live_component_count: usize,
}

impl CompiledDirectCurrentLayout {
    fn identity(canonical_component_count: usize) -> RusticolResult<Self> {
        let physical_component_count = u32::try_from(canonical_component_count).map_err(|_| {
            RusticolError::integrity("compiled Direct-Arena current plane count exceeds u32")
        })?;
        Ok(Self {
            canonical_to_physical: (0..physical_component_count)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            physical_component_count,
            live_component_count: canonical_component_count,
        })
    }

    fn physical_component(&self, canonical_component: usize) -> RusticolResult<u32> {
        self.canonical_to_physical
            .get(canonical_component)
            .copied()
            .filter(|component| *component != UNMAPPED_CURRENT_COMPONENT)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "compiled Direct-Arena canonical current component {canonical_component} has \
                     no physical liveness assignment"
                ))
            })
    }

    fn physical_map(&self) -> &[u32] {
        &self.canonical_to_physical
    }

    fn is_identity(&self) -> bool {
        self.live_component_count == self.canonical_to_physical.len()
            && self.physical_component_count as usize == self.canonical_to_physical.len()
            && self
                .canonical_to_physical
                .iter()
                .copied()
                .enumerate()
                .all(|(canonical, physical)| physical as usize == canonical)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompiledDirectCurrentLayoutPolicy {
    Identity { tile_capacity: u32 },
    StageLivenessProduction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompiledDirectTileSizing {
    pub(crate) effective_tile_capacity: usize,
    pub(crate) reduction_tile_capacity: usize,
    pub(crate) physical_scalar_values_per_point: usize,
    pub(crate) persistent_scalar_values_per_point: usize,
    pub(crate) hot_scalar_values_per_point: usize,
    execution_scalar_values_per_point: usize,
    pub(crate) source_scalar_values_per_point: usize,
    pub(crate) reduction_scalar_values_per_point: usize,
}

impl CompiledDirectTileSizing {
    fn reduction_tile_capacity_for_footprint(
        self,
        reduction_footprint: CompiledDirectReductionFootprint,
    ) -> RusticolResult<usize> {
        let Some(reduction_scalars) = reduction_footprint.hot_scalar_values_per_point() else {
            return Ok(self.reduction_tile_capacity);
        };
        // A selected routed total only touches its selected source/target
        // components. The all-selector footprint remains the default for
        // other reducers. Never exceed the already allocated execution tile:
        // this changes cache locality, not the arena's allocation bound.
        let capacity = u32::try_from(self.effective_tile_capacity).map_err(|_| {
            RusticolError::integrity("compiled Direct-Arena tile capacity exceeds u32")
        })?;
        Ok(deterministic_point_tile_size_with_cache_footprint(
            capacity,
            usize::MAX,
            compiled_direct_cache_target_bytes(),
            self.physical_scalar_values_per_point,
            self.execution_scalar_values_per_point
                .max(reduction_scalars),
        )? as usize)
    }
}

trait DirectStagePlan {
    fn leaf_plans(&self) -> &[DirectLeafPlan];
}

impl DirectStagePlan for LoadedStage {
    fn leaf_plans(&self) -> &[DirectLeafPlan] {
        &self.leaf_plans
    }
}

impl DirectStagePlan for BoundStage {
    fn leaf_plans(&self) -> &[DirectLeafPlan] {
        &self.leaf_plans
    }
}

#[allow(clippy::too_many_arguments)]
fn compiled_direct_tile_sizing(
    stages: &[LoadedStage],
    amplitude: &LoadedStage,
    current_layout: &CompiledDirectCurrentLayout,
    source_components: &[usize],
    momentum_component_count: usize,
    amplitude_component_count: usize,
    reduction_footprint: CompiledDirectReductionFootprint,
    layout_policy: CompiledDirectCurrentLayoutPolicy,
) -> RusticolResult<CompiledDirectTileSizing> {
    let persistent_scalar_values_per_point = stages
        .iter()
        .flat_map(|stage| stage.leaf_plans.iter())
        .chain(amplitude.leaf_plans.iter())
        .try_fold(0usize, |total, leaf| {
            total
                .checked_add(leaf.persistent_scalar_values_per_point)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena persistent scalar footprint overflows",
                    )
                })
        })?;
    let physical_scalar_values_per_point = usize::try_from(current_layout.physical_component_count)
        .ok()
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(momentum_component_count))
        .and_then(|value| {
            amplitude_component_count
                .checked_mul(2)
                .and_then(|amplitudes| value.checked_add(amplitudes))
        })
        // The shared zero plane is point-pitched even though every leaf
        // aliases the same immutable storage.
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(persistent_scalar_values_per_point))
        .ok_or_else(|| {
            RusticolError::integrity("compiled Direct-Arena physical per-point shape overflows")
        })?;
    let source_scalar_values_per_point = source_components
        .len()
        .checked_mul(2)
        .and_then(|value| value.checked_add(momentum_component_count))
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            RusticolError::integrity("compiled Direct-Arena source working set overflows")
        })?;
    let maximum_leaf_scalar_values_per_point = stages
        .iter()
        .flat_map(|stage| stage.leaf_plans.iter())
        .chain(amplitude.leaf_plans.iter())
        .map(|leaf| leaf.scalar_values_per_point)
        .max()
        .ok_or_else(|| {
            RusticolError::integrity(
                "compiled Direct-Arena production schedule has no evaluator leaves",
            )
        })?;
    let authenticated_reduction_scalar_values_per_point =
        reduction_footprint.hot_scalar_values_per_point();
    let execution_scalar_values_per_point =
        maximum_leaf_scalar_values_per_point.max(source_scalar_values_per_point);
    let hot_scalar_values_per_point = execution_scalar_values_per_point
        .max(authenticated_reduction_scalar_values_per_point.unwrap_or(0));
    let (effective_tile_capacity, reduction_tile_capacity) = match layout_policy {
        // This developer/test contract remains exact: the adaptive locality
        // policy does not change it.
        CompiledDirectCurrentLayoutPolicy::Identity { tile_capacity } => {
            let capacity = usize::try_from(tile_capacity).map_err(|_| {
                RusticolError::integrity("compiled Direct-Arena identity tile exceeds usize")
            })?;
            (capacity, capacity)
        }
        CompiledDirectCurrentLayoutPolicy::StageLivenessProduction => {
            if authenticated_reduction_scalar_values_per_point.is_none() {
                // Adaptive production tiling is only valid when every
                // reducer reachable from this engine has a static footprint.
                // Still route the one-point fallback through the common
                // selector so its padded physical allocation remains checked.
                let capacity = deterministic_point_tile_size_with_cache_footprint(
                    1,
                    usize::MAX,
                    usize::MAX,
                    physical_scalar_values_per_point,
                    hot_scalar_values_per_point,
                )? as usize;
                (capacity, capacity)
            } else {
                let execution_capacity = deterministic_point_tile_size_with_cache_footprint(
                    COMPILED_DIRECT_LOCALITY_POINT_CAP,
                    usize::MAX,
                    compiled_direct_cache_target_bytes(),
                    physical_scalar_values_per_point,
                    execution_scalar_values_per_point,
                )? as usize;
                let reduction_capacity = deterministic_point_tile_size_with_cache_footprint(
                    COMPILED_DIRECT_LOCALITY_POINT_CAP,
                    usize::MAX,
                    compiled_direct_cache_target_bytes(),
                    physical_scalar_values_per_point,
                    hot_scalar_values_per_point,
                )? as usize;
                (execution_capacity, reduction_capacity)
            }
        }
    };
    Ok(CompiledDirectTileSizing {
        effective_tile_capacity,
        reduction_tile_capacity,
        physical_scalar_values_per_point,
        persistent_scalar_values_per_point,
        hot_scalar_values_per_point,
        execution_scalar_values_per_point,
        source_scalar_values_per_point,
        reduction_scalar_values_per_point: authenticated_reduction_scalar_values_per_point
            .unwrap_or(0),
    })
}

/// Cold-bound canonical preorder leaf schedule.
///
/// Fields remain private so arbitrary caller indices cannot bypass the
/// production runtime coverage validation that creates selector schedules.
pub(crate) struct CompiledDirectValidatedSchedule {
    active_stage_leaves: Box<[Box<[usize]>]>,
    active_amplitude_leaves: Box<[usize]>,
    inactive_amplitude_components: Box<[usize]>,
}

/// Persistent prototype executor for one compiled schedule.
///
/// The descriptor objects contain pointers into the boxed arena and aligned
/// side buffers. Those allocations never resize or move after construction.
pub(crate) struct CompiledDirectEnginePrototype {
    // Bound descriptors must drop before the allocations they address. Rust
    // drops fields in declaration order, so keep these owners first.
    stages: Vec<BoundStage>,
    amplitude: BoundStage,
    arena: Box<DirectArenaWorkspace>,
    momenta: Box<AlignedF64Buffer>,
    parameter_re: Box<AlignedF64Buffer>,
    parameter_im: Box<AlignedF64Buffer>,
    zero_plane: Box<AlignedF64Buffer>,
    full_schedule: CompiledDirectValidatedSchedule,
    helicity_color_schedules:
        BTreeMap<usize, BTreeMap<BTreeSet<i64>, Arc<CompiledDirectValidatedSchedule>>>,
    source_components: Box<[usize]>,
    default_source_states: Box<[GenericSourceStateIrManifest]>,
    source_wavefunction_scratch: Vec<Complex<f64>>,
    current_layout: CompiledDirectCurrentLayout,
    tile_sizing: CompiledDirectTileSizing,
    value_component_count: usize,
    momentum_component_count: usize,
    model_parameter_count: usize,
    amplitude_component_count: usize,
    traffic: CompiledDirectPrototypeTraffic,
    allocation_counters: DirectArenaAllocationCounters,
}

impl CompiledDirectEnginePrototype {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        stages: &[GenericSerializedStageEvaluatorManifest],
        amplitude: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
        source_components: &[usize],
        source_scratch_len: usize,
        value_component_count: usize,
        momentum_component_count: usize,
        model_parameter_count: usize,
        amplitude_component_count: usize,
        tile_capacity: u32,
    ) -> RusticolResult<Self> {
        Self::load_with_current_layout_policy(
            stages,
            amplitude,
            payloads,
            source_components,
            source_scratch_len,
            value_component_count,
            momentum_component_count,
            model_parameter_count,
            amplitude_component_count,
            CompiledDirectReductionFootprint::default(),
            CompiledDirectCurrentLayoutPolicy::Identity { tile_capacity },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_with_current_layout_policy(
        stages: &[GenericSerializedStageEvaluatorManifest],
        amplitude: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
        source_components: &[usize],
        source_scratch_len: usize,
        value_component_count: usize,
        momentum_component_count: usize,
        model_parameter_count: usize,
        amplitude_component_count: usize,
        reduction_footprint: CompiledDirectReductionFootprint,
        layout_policy: CompiledDirectCurrentLayoutPolicy,
    ) -> RusticolResult<Self> {
        if value_component_count == 0 || amplitude_component_count == 0 {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena prototype requires current and amplitude planes",
            ));
        }
        if !momentum_component_count.is_multiple_of(4) {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena momentum component count must be a multiple of four",
            ));
        }
        if source_components.is_empty() || source_scratch_len == 0 {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena requires canonical source-component ownership",
            ));
        }
        let mut previous_source = None;
        for &component in source_components {
            if component >= value_component_count
                || previous_source.is_some_and(|previous| previous >= component)
            {
                return Err(RusticolError::integrity(
                    "compiled Direct-Arena source components are not sorted, unique, and in bounds",
                ));
            }
            previous_source = Some(component);
        }
        let structural_zero_components =
            canonical_structural_zero_components(stages, source_components, value_component_count)?;
        let loaded_stages = stages
            .iter()
            .map(|stage| {
                load_stage(
                    stage,
                    false,
                    payloads,
                    value_component_count,
                    momentum_component_count,
                    amplitude_component_count,
                    &structural_zero_components,
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let loaded_amplitude = load_stage(
            amplitude,
            true,
            payloads,
            value_component_count,
            momentum_component_count,
            amplitude_component_count,
            &structural_zero_components,
        )?;

        let full_stage_leaves = loaded_stages
            .iter()
            .map(|stage| (0..stage.leaves.len()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let full_amplitude_leaves = (0..loaded_amplitude.leaves.len()).collect::<Vec<_>>();
        let full_schedule = validate_schedule(
            &loaded_stages,
            &loaded_amplitude,
            value_component_count,
            source_components,
            amplitude_component_count,
            &full_stage_leaves,
            &full_amplitude_leaves,
        )?;
        let current_layout = match layout_policy {
            CompiledDirectCurrentLayoutPolicy::Identity { .. } => {
                CompiledDirectCurrentLayout::identity(value_component_count)?
            }
            CompiledDirectCurrentLayoutPolicy::StageLivenessProduction => {
                derive_stage_current_layout(
                    &loaded_stages,
                    &loaded_amplitude,
                    source_components,
                    value_component_count,
                )?
            }
        };
        let tile_sizing = compiled_direct_tile_sizing(
            &loaded_stages,
            &loaded_amplitude,
            &current_layout,
            source_components,
            momentum_component_count,
            amplitude_component_count,
            reduction_footprint,
            layout_policy,
        )?;
        let tile_capacity = u32::try_from(tile_sizing.effective_tile_capacity).map_err(|_| {
            RusticolError::integrity("compiled Direct-Arena tile capacity exceeds u32")
        })?;
        let mut arena = Box::new(DirectArenaWorkspace::new(
            current_layout.physical_component_count,
            u32::try_from(amplitude_component_count).map_err(|_| {
                RusticolError::integrity("compiled Direct-Arena amplitude plane count exceeds u32")
            })?,
            tile_capacity,
        )?);
        arena.begin_tile(tile_capacity)?;
        let stride = arena.point_stride() as usize;
        let momenta_len = momentum_component_count
            .checked_mul(stride)
            .ok_or_else(|| RusticolError::integrity("compiled momentum arena length overflows"))?;
        let mut momenta = Box::new(AlignedF64Buffer::zeroed(
            momenta_len,
            "compiled prototype momenta",
        )?);
        let mut parameter_re = Box::new(AlignedF64Buffer::zeroed(
            model_parameter_count,
            "compiled prototype parameter real",
        )?);
        let mut parameter_im = Box::new(AlignedF64Buffer::zeroed(
            model_parameter_count,
            "compiled prototype parameter imaginary",
        )?);
        let zero_plane = Box::new(AlignedF64Buffer::zeroed(
            stride,
            "compiled prototype shared zero plane",
        )?);

        let arena_view = arena.view()?;
        let momentum_view = DirectMomentumView {
            values: momenta.as_mut_ptr(),
            scalar_len: momenta.len() as u64,
            form_count: u32::try_from(momentum_component_count / 4).map_err(|_| {
                RusticolError::integrity("compiled momentum form count exceeds u32")
            })?,
            lorentz_component_count: 4,
            point_stride: arena_view.point_stride,
        };
        let parameter_view = DirectParameterView {
            values_re: if parameter_re.is_empty() {
                ptr::null()
            } else {
                parameter_re.as_mut_ptr()
            },
            values_im: if parameter_im.is_empty() {
                ptr::null()
            } else {
                parameter_im.as_mut_ptr()
            },
            value_count: u32::try_from(model_parameter_count).map_err(|_| {
                RusticolError::integrity("compiled model parameter count exceeds u32")
            })?,
        };

        // SAFETY: arena and side buffers are fixed-size heap allocations owned
        // by the returned executor. No field resizes them, and evaluate methods
        // require exclusive access to the executor.
        let bind_stage = |loaded: LoadedStage| -> RusticolResult<BoundStage> {
            Ok(BoundStage {
                leaves: loaded
                    .leaves
                    .into_iter()
                    .map(|leaf| unsafe {
                        leaf.bind(
                            arena_view,
                            momentum_view,
                            parameter_view,
                            &zero_plane,
                            current_layout.physical_map(),
                        )
                    })
                    .collect::<RusticolResult<Vec<_>>>()?,
                leaf_plans: loaded.leaf_plans,
            })
        };
        let stages = loaded_stages
            .into_iter()
            .map(bind_stage)
            .collect::<RusticolResult<Vec<_>>>()?;
        let amplitude = bind_stage(loaded_amplitude)?;
        let allocation_counters = [
            arena.allocation_counters(),
            momenta.allocation_counters(),
            parameter_re.allocation_counters(),
            parameter_im.allocation_counters(),
            zero_plane.allocation_counters(),
            DirectArenaAllocationCounters {
                allocation_requests: 1,
                requested_bytes: u64::try_from(
                    source_scratch_len.saturating_mul(std::mem::size_of::<Complex<f64>>()),
                )
                .unwrap_or(u64::MAX),
            },
        ]
        .into_iter()
        .try_fold(
            DirectArenaAllocationCounters::default(),
            |total, counters| total.checked_add(counters),
        )?;
        let allocation_counters = stages
            .iter()
            .flat_map(|stage| stage.leaves.iter())
            .chain(amplitude.leaves.iter())
            .try_fold(allocation_counters, |total, leaf| {
                total.checked_add(leaf.allocation_counters())
            })?;

        Ok(Self {
            stages,
            amplitude,
            arena,
            momenta,
            parameter_re,
            parameter_im,
            zero_plane,
            full_schedule,
            helicity_color_schedules: BTreeMap::new(),
            source_components: source_components.to_vec().into_boxed_slice(),
            default_source_states: Box::new([]),
            source_wavefunction_scratch: vec![c64(0.0, 0.0); source_scratch_len],
            current_layout,
            tile_sizing,
            value_component_count,
            momentum_component_count,
            model_parameter_count,
            amplitude_component_count,
            traffic: CompiledDirectPrototypeTraffic::default(),
            allocation_counters,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_production(
        stages: &[GenericSerializedStageEvaluatorManifest],
        amplitude: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
        sources: &[GenericSourceRecordManifest],
        value_component_count: usize,
        momentum_component_count: usize,
        model_parameter_count: usize,
        amplitude_component_count: usize,
        reduction_footprint: CompiledDirectReductionFootprint,
    ) -> RusticolResult<Self> {
        let (source_components, source_scratch_len) =
            canonical_source_layout(sources, value_component_count)?;
        let mut direct = Self::load_with_current_layout_policy(
            stages,
            amplitude,
            payloads,
            &source_components,
            source_scratch_len,
            value_component_count,
            momentum_component_count,
            model_parameter_count,
            amplitude_component_count,
            reduction_footprint,
            CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
        )?;
        direct.default_source_states = sources
            .iter()
            .map(ExecutionRuntime::default_runtime_source_state)
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        Ok(direct)
    }

    pub(crate) const fn traffic(&self) -> CompiledDirectPrototypeTraffic {
        self.traffic
    }

    pub(crate) const fn allocation_counters(&self) -> DirectArenaAllocationCounters {
        self.allocation_counters
    }

    pub(crate) const fn tile_capacity(&self) -> usize {
        self.arena.tile_capacity() as usize
    }

    pub(crate) const fn reduction_tile_capacity(&self) -> usize {
        self.tile_sizing.reduction_tile_capacity
    }

    pub(crate) fn reduction_tile_capacity_for_footprint(
        &self,
        reduction_footprint: CompiledDirectReductionFootprint,
    ) -> RusticolResult<usize> {
        self.tile_sizing
            .reduction_tile_capacity_for_footprint(reduction_footprint)
    }

    pub(crate) const fn tile_sizing(&self) -> CompiledDirectTileSizing {
        self.tile_sizing
    }

    /// Fill one active tile directly from the public borrowed momentum view
    /// and immutable runtime metadata. No point-major global state exists.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_tile_from_inputs(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        sources: &[GenericSourceRecordManifest],
        source_states: Option<&[RuntimeSourceState]>,
        external_count: usize,
        particle_masses: &BTreeMap<i32, f64>,
        momentum_slots: &[GenericMomentumSlotManifest],
        external_is_initial: &[bool],
        model_parameter_values: &[f64],
    ) -> RusticolResult<()> {
        let point_count = batch.point_count();
        if point_count == 0 || point_count > self.tile_capacity() {
            return Err(RusticolError::invalid_argument(format!(
                "compiled Direct-Arena input tile has {point_count} points, expected 1..={}",
                self.tile_capacity()
            )));
        }
        if batch.external_count() != external_count {
            return Err(RusticolError::invalid_argument(format!(
                "compiled Direct-Arena input has {} external legs, expected {external_count}",
                batch.external_count()
            )));
        }
        if let Some(states) = source_states
            && states.len() != sources.len()
        {
            return Err(RusticolError::invalid_argument(format!(
                "compiled Direct-Arena source-state count {} does not match source count {}",
                states.len(),
                sources.len()
            )));
        }
        if source_states.is_none() && self.default_source_states.len() != sources.len() {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena default source-state count {} does not match source count {}",
                self.default_source_states.len(),
                sources.len()
            )));
        }
        if model_parameter_values.len() != self.model_parameter_count {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena model-scalar count {} does not match {}",
                model_parameter_values.len(),
                self.model_parameter_count
            )));
        }

        self.arena
            .begin_tile(u32::try_from(point_count).map_err(|_| {
                RusticolError::invalid_argument("compiled Direct-Arena point count exceeds u32")
            })?)?;
        let stride = self.arena.point_stride() as usize;
        {
            let (current_re, current_im, _, _) = self.arena.split_slices_mut();
            let scratch = &mut self.source_wavefunction_scratch;
            for (source_index, source) in sources.iter().enumerate() {
                let start = source.value_slot.component_start;
                let stop = source.value_slot.component_stop;
                let dimension = stop.checked_sub(start).ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena source component range underflows",
                    )
                })?;
                if dimension == 0 || stop > self.value_component_count || dimension > scratch.len()
                {
                    return Err(RusticolError::integrity(format!(
                        "compiled Direct-Arena source {} has an invalid component range",
                        source.source_id
                    )));
                }
                // Load-time current-layout validation covers every source;
                // the physical mapping is immutable across tiles.
                let physical_components = &self.current_layout.physical_map()[start..stop];
                let runtime_state = source_states.map(|states| &states[source_index]);
                let factor = runtime_state.map_or(c64(1.0, 0.0), |state| state.factor);
                if factor == c64(0.0, 0.0) {
                    for &component in physical_components {
                        let target = component as usize * stride;
                        current_re[target..target + point_count].fill(0.0);
                        current_im[target..target + point_count].fill(0.0);
                    }
                    continue;
                }
                let state = runtime_state.map_or_else(
                    || &self.default_source_states[source_index],
                    |state| &state.state,
                );
                let output = &mut scratch[..dimension];
                let wavefunction = ResolvedSourceWavefunction::new(
                    source,
                    state,
                    external_count,
                    particle_masses,
                    dimension,
                )?;
                for point_index in 0..point_count {
                    let point = batch.point(point_index);
                    wavefunction.write(&point, output)?;
                    if factor != c64(1.0, 0.0) {
                        for value in output.iter_mut() {
                            *value *= factor;
                        }
                    }
                    for (&component, value) in physical_components.iter().zip(output.iter()) {
                        let target = component as usize * stride + point_index;
                        current_re[target] = value.re;
                        current_im[target] = value.im;
                    }
                }
            }
        }

        fill_compiled_momentum_planes(
            batch,
            momentum_slots,
            external_is_initial,
            self.momentum_component_count,
            stride,
            self.momenta.as_mut_slice(),
        )?;
        for (index, value) in model_parameter_values.iter().copied().enumerate() {
            self.parameter_re.as_mut_slice()[index] = value;
            self.parameter_im.as_mut_slice()[index] = 0.0;
        }

        self.traffic.source_fill_bytes = self.traffic.source_fill_bytes.saturating_add(
            point_count
                .saturating_mul(self.source_components.len())
                .saturating_mul(2 * std::mem::size_of::<f64>()) as u64,
        );
        self.traffic.momentum_fill_bytes = self.traffic.momentum_fill_bytes.saturating_add(
            point_count
                .saturating_mul(momentum_slots.len().saturating_mul(4))
                .saturating_mul(std::mem::size_of::<f64>()) as u64,
        );
        self.traffic.parameter_fill_bytes = self.traffic.parameter_fill_bytes.saturating_add(
            self.model_parameter_count
                .saturating_mul(2 * std::mem::size_of::<f64>()) as u64,
        );
        Ok(())
    }

    /// Transpose the outer row-major state once at the engine boundary.
    ///
    /// This is not leaf gather traffic: every compiled leaf subsequently
    /// consumes and produces the authoritative arena planes directly.
    pub(crate) fn begin_tile_from_state(
        &mut self,
        point_count: usize,
        global_parameter_count: usize,
        state: &[Complex<f64>],
    ) -> RusticolResult<()> {
        if !self.current_layout.is_identity() {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena full-state developer input requires an identity current \
                 layout",
            ));
        }
        if point_count == 0
            || point_count > self.arena.tile_capacity() as usize
            || state.len() != point_count.saturating_mul(global_parameter_count)
        {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena prototype state shape is invalid",
            ));
        }
        let required_parameters = self
            .value_component_count
            .checked_add(self.momentum_component_count)
            .and_then(|value| value.checked_add(self.model_parameter_count))
            .ok_or_else(|| RusticolError::integrity("compiled parameter layout overflows"))?;
        if global_parameter_count < required_parameters {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena prototype state omits required parameters",
            ));
        }

        self.arena.begin_tile(point_count as u32)?;
        let stride = self.arena.point_stride() as usize;
        {
            let (current_re, current_im, amplitude_re, amplitude_im) =
                self.arena.split_slices_mut();
            for component in 0..self.value_component_count {
                let plane = component * stride;
                for point in 0..point_count {
                    let value = state[point * global_parameter_count + component];
                    current_re[plane + point] = value.re;
                    current_im[plane + point] = value.im;
                }
            }
            for component in 0..self.amplitude_component_count {
                let plane = component * stride;
                amplitude_re[plane..plane + point_count].fill(0.0);
                amplitude_im[plane..plane + point_count].fill(0.0);
            }
        }
        let momentum_start = self.value_component_count;
        for component in 0..self.momentum_component_count {
            let plane = component * stride;
            for point in 0..point_count {
                self.momenta.as_mut_slice()[plane + point] =
                    state[point * global_parameter_count + momentum_start + component].re;
            }
        }
        let parameter_start = momentum_start + self.momentum_component_count;
        if self.model_parameter_count != 0 {
            for index in 0..self.model_parameter_count {
                let value = state[parameter_start + index];
                self.parameter_re.as_mut_slice()[index] = value.re;
                self.parameter_im.as_mut_slice()[index] = value.im;
                for point in 1..point_count {
                    if state[point * global_parameter_count + parameter_start + index] != value {
                        return Err(RusticolError::invalid_argument(
                            "compiled Direct-Arena model parameters vary across a point tile",
                        ));
                    }
                }
            }
        }
        let input_scalars = point_count
            .checked_mul(
                self.value_component_count
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(self.momentum_component_count))
                    .ok_or_else(|| {
                        RusticolError::integrity("compiled boundary input traffic overflows")
                    })?,
            )
            .and_then(|value| value.checked_add(self.model_parameter_count.saturating_mul(2)))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| RusticolError::integrity("compiled boundary input bytes overflow"))?;
        self.traffic.boundary_input_bytes = self
            .traffic
            .boundary_input_bytes
            .saturating_add(input_scalars as u64);
        Ok(())
    }

    /// Test-only row-major source boundary for the production liveness layout.
    ///
    /// A liveness arena cannot represent arbitrary intermediate values at one
    /// instant. It can, however, be initialized exactly like production by
    /// clearing every physical plane and scattering only canonical sources.
    #[cfg(test)]
    fn begin_tile_from_state_sources_for_test(
        &mut self,
        point_count: usize,
        global_parameter_count: usize,
        state: &[Complex<f64>],
    ) -> RusticolResult<()> {
        if point_count == 0
            || point_count > self.arena.tile_capacity() as usize
            || state.len() != point_count.saturating_mul(global_parameter_count)
        {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena production test state shape is invalid",
            ));
        }
        let required_parameters = self
            .value_component_count
            .checked_add(self.momentum_component_count)
            .and_then(|value| value.checked_add(self.model_parameter_count))
            .ok_or_else(|| RusticolError::integrity("compiled parameter layout overflows"))?;
        if global_parameter_count < required_parameters {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena production test state omits required parameters",
            ));
        }
        self.arena.begin_tile(point_count as u32)?;
        self.arena
            .clear_current_active(0, self.current_layout.physical_component_count)?;
        self.arena
            .clear_amplitude_active(0, self.amplitude_component_count as u32)?;
        let stride = self.arena.point_stride() as usize;
        {
            let (current_re, current_im, _, _) = self.arena.split_slices_mut();
            for canonical_component in self.source_components.iter().copied() {
                let physical_component =
                    self.current_layout
                        .physical_component(canonical_component)? as usize;
                let plane = physical_component * stride;
                for point in 0..point_count {
                    let value = state[point * global_parameter_count + canonical_component];
                    current_re[plane + point] = value.re;
                    current_im[plane + point] = value.im;
                }
            }
        }
        let momentum_start = self.value_component_count;
        for component in 0..self.momentum_component_count {
            let plane = component * stride;
            for point in 0..point_count {
                self.momenta.as_mut_slice()[plane + point] =
                    state[point * global_parameter_count + momentum_start + component].re;
            }
        }
        let parameter_start = momentum_start + self.momentum_component_count;
        for index in 0..self.model_parameter_count {
            let value = state[parameter_start + index];
            self.parameter_re.as_mut_slice()[index] = value.re;
            self.parameter_im.as_mut_slice()[index] = value.im;
            for point in 1..point_count {
                if state[point * global_parameter_count + parameter_start + index] != value {
                    return Err(RusticolError::invalid_argument(
                        "compiled Direct-Arena model parameters vary across a production test tile",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Execute the existing full schedule. No input packet, gather, output
    /// scatter, remap, or allocation occurs in this method.
    pub(crate) fn evaluate_all(&mut self, point_count: usize) -> RusticolResult<()> {
        evaluate_validated_schedule(
            &mut self.stages,
            &mut self.amplitude,
            &self.full_schedule,
            point_count_u32(point_count, self.arena.active_point_count())?,
            &mut self.traffic,
        )
    }

    /// Replace every retained source slot of a declared leg by the literal
    /// public vector. Apply only the same SourceIR crossing phase as ordinary
    /// wavefunctions; no projection, normalization or conjugation is applied.
    pub(crate) fn replace_correlated_sources(
        &mut self,
        sources: &[GenericSourceRecordManifest],
        vectors: &correlated::NativeSpinCorrelationVectors,
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        point_count_u32(point_count, self.arena.active_point_count())?;
        let stride = self.arena.point_stride() as usize;
        let (re, im, _, _) = self.arena.split_slices_mut();
        for source in sources {
            let Some(batch) = vectors.get(&source.leg_label) else {
                continue;
            };
            let [pr, pi] = source.applied_crossing.phase;
            // Traverse Lorentz components in arena order, not the outer batch
            // axis: each point below selects its own vector (or broadcasts it).
            #[allow(clippy::needless_range_loop)]
            for component in 0..4 {
                let physical = self
                    .current_layout
                    .physical_component(source.value_slot.component_start + component)?
                    as usize;
                for point in 0..point_count {
                    let [vr, vi] = batch[if batch.len() == 1 {
                        0
                    } else {
                        point_start + point
                    }][component];
                    re[physical * stride + point] = vr * pr - vi * pi;
                    im[physical * stride + point] = vr * pi + vi * pr;
                }
            }
        }
        Ok(())
    }

    /// Reuse a leaf only when all of its current inputs are bit-identical.
    /// Momenta and model parameters must stay fixed for the memo's lifetime;
    /// the correlated caller creates a fresh memo for each point tile. Saved
    /// outputs are restored because the production arena reuses dead slots.
    pub(crate) fn evaluate_correlated_reusing(
        &mut self,
        point_count: usize,
        memo: &mut CorrelatedStageMemo,
    ) -> RusticolResult<()> {
        let count = point_count_u32(point_count, self.arena.active_point_count())?;
        let leaf_count = self
            .stages
            .iter()
            .map(|stage| stage.leaves.len())
            .sum::<usize>()
            + self.amplitude.leaves.len();
        if memo.leaves.is_empty() {
            memo.leaves.resize_with(leaf_count, || None);
        }
        if memo.leaves.len() != leaf_count {
            return Err(RusticolError::internal(
                "correlated stage memo belongs to a different compiled plan",
            ));
        }
        let stride = self.arena.point_stride() as usize;
        let mut index = 0;
        for stage in self
            .stages
            .iter_mut()
            .chain(std::iter::once(&mut self.amplitude))
        {
            for (leaf, plan) in stage.leaves.iter_mut().zip(&stage.leaf_plans) {
                let input_components = plan
                    .input_current_components
                    .iter()
                    .map(|&component| self.current_layout.physical_component(component))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let output_components = plan
                    .output_current_components
                    .iter()
                    .map(|&component| self.current_layout.physical_component(component))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let (re, im) = self.arena.current_slices();
                let inputs: Vec<_> = input_components
                    .iter()
                    .flat_map(|&component| {
                        (0..point_count).flat_map(move |point| {
                            [
                                re[component as usize * stride + point].to_bits(),
                                im[component as usize * stride + point].to_bits(),
                            ]
                        })
                    })
                    .collect();
                if let Some(saved) = &memo.leaves[index]
                    && saved.inputs == inputs
                {
                    #[cfg(test)]
                    {
                        memo.reused_leaves += 1;
                    }
                    let (re, im, ar, ai) = self.arena.split_slices_mut();
                    for (position, &component) in output_components.iter().enumerate() {
                        for point in 0..point_count {
                            let value = saved.current_outputs[position * point_count + point];
                            re[component as usize * stride + point] = value[0];
                            im[component as usize * stride + point] = value[1];
                        }
                    }
                    for (position, &component) in
                        plan.output_amplitude_components.iter().enumerate()
                    {
                        for point in 0..point_count {
                            let value = saved.amplitude_outputs[position * point_count + point];
                            ar[component * stride + point] = value[0];
                            ai[component * stride + point] = value[1];
                        }
                    }
                } else {
                    leaf.evaluate(0, count)?;
                    let (re, im) = self.arena.current_slices();
                    let current_outputs = output_components
                        .iter()
                        .flat_map(|&component| {
                            (0..point_count).map(move |point| {
                                [
                                    re[component as usize * stride + point],
                                    im[component as usize * stride + point],
                                ]
                            })
                        })
                        .collect();
                    let (re, im) = self.arena.amplitude_slices();
                    let amplitude_outputs = plan
                        .output_amplitude_components
                        .iter()
                        .flat_map(|&component| {
                            (0..point_count).map(move |point| {
                                [
                                    re[component * stride + point],
                                    im[component * stride + point],
                                ]
                            })
                        })
                        .collect();
                    memo.leaves[index] = Some(CorrelatedLeafMemo {
                        inputs,
                        current_outputs,
                        amplitude_outputs,
                    });
                }
                index += 1;
            }
        }
        Ok(())
    }

    /// Bind an already producer-validated compiled color schedule to this
    /// concrete direct plan. This remains cold-path validation: hot execution
    /// accepts only the opaque returned schedule.
    pub(crate) fn bind_color_schedule(
        &self,
        schedule: &CompiledColorSelectorSchedule,
    ) -> RusticolResult<CompiledDirectValidatedSchedule> {
        validate_schedule(
            &self.stages,
            &self.amplitude,
            self.value_component_count,
            &self.source_components,
            self.amplitude_component_count,
            &schedule.active_stage_chunk_indices,
            &schedule.active_amplitude_chunk_indices,
        )
    }

    /// Bind an already producer-validated compiled helicity schedule to this
    /// concrete direct plan.
    pub(crate) fn bind_helicity_schedule(
        &self,
        schedule: &CompiledHelicitySelectorSchedule,
    ) -> RusticolResult<CompiledDirectValidatedSchedule> {
        validate_schedule(
            &self.stages,
            &self.amplitude,
            self.value_component_count,
            &self.source_components,
            self.amplitude_component_count,
            &schedule.active_stage_chunk_indices,
            &schedule.active_amplitude_chunk_indices,
        )
    }

    /// Reuse the closure common to a helicity and a union of requested colours.
    /// Binding is done once per selection, never per tile or phase-space point.
    pub(crate) fn bind_helicity_color_schedule(
        &mut self,
        selector_domain_id: usize,
        helicity: &CompiledDirectValidatedSchedule,
        selected_sector_ids: &BTreeSet<i64>,
        color_schedules: &BTreeMap<i64, CompiledDirectValidatedSchedule>,
    ) -> RusticolResult<Arc<CompiledDirectValidatedSchedule>> {
        if let Some(schedule) = self
            .helicity_color_schedules
            .get(&selector_domain_id)
            .and_then(|colors| colors.get(selected_sector_ids))
        {
            return Ok(Arc::clone(schedule));
        }
        if selected_sector_ids.is_empty() {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena colour selection is empty",
            ));
        }
        let colors = selected_sector_ids
            .iter()
            .map(|id| {
                color_schedules.get(id).ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "compiled Direct-Arena colour schedule {id} is missing"
                    ))
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let (stages, amplitude) = intersect_helicity_color_schedules(helicity, &colors)?;
        let schedule = Arc::new(validate_schedule(
            &self.stages,
            &self.amplitude,
            self.value_component_count,
            &self.source_components,
            self.amplitude_component_count,
            &stages,
            &amplitude,
        )?);
        self.helicity_color_schedules
            .entry(selector_domain_id)
            .or_default()
            .insert(selected_sector_ids.clone(), Arc::clone(&schedule));
        Ok(schedule)
    }

    /// Execute a cold-bound canonical preorder leaf schedule.
    pub(crate) fn evaluate_validated(
        &mut self,
        point_count: usize,
        schedule: &CompiledDirectValidatedSchedule,
    ) -> RusticolResult<()> {
        self.clear_inactive_amplitude_components(schedule)?;
        evaluate_validated_schedule(
            &mut self.stages,
            &mut self.amplitude,
            schedule,
            point_count_u32(point_count, self.arena.active_point_count())?,
            &mut self.traffic,
        )
    }

    /// Execute an authenticated materialized-helicity schedule without
    /// clearing amplitude planes outside its active root closure.
    ///
    /// Materialized-helicity reducers consume only roots carried by the same
    /// authenticated selector schedule. Inactive planes may therefore retain
    /// values from an earlier tile and must not impose an all-helicity memory
    /// pass on a singleton-helicity execution.
    pub(crate) fn evaluate_validated_materialized_helicity(
        &mut self,
        point_count: usize,
        schedule: &CompiledDirectValidatedSchedule,
    ) -> RusticolResult<()> {
        evaluate_validated_schedule(
            &mut self.stages,
            &mut self.amplitude,
            schedule,
            point_count_u32(point_count, self.arena.active_point_count())?,
            &mut self.traffic,
        )
    }

    fn clear_inactive_amplitude_components(
        &mut self,
        schedule: &CompiledDirectValidatedSchedule,
    ) -> RusticolResult<()> {
        let mut cursor = 0;
        while cursor < schedule.inactive_amplitude_components.len() {
            let start = schedule.inactive_amplitude_components[cursor];
            let mut stop = start + 1;
            cursor += 1;
            while cursor < schedule.inactive_amplitude_components.len()
                && schedule.inactive_amplitude_components[cursor] == stop
            {
                stop += 1;
                cursor += 1;
            }
            self.arena.clear_amplitude_active(
                u32::try_from(start).map_err(|_| {
                    RusticolError::integrity("compiled amplitude clear start exceeds u32")
                })?,
                u32::try_from(stop - start).map_err(|_| {
                    RusticolError::integrity("compiled amplitude clear length exceeds u32")
                })?,
            )?;
            self.traffic.amplitude_clear_bytes = self.traffic.amplitude_clear_bytes.saturating_add(
                (stop - start)
                    .saturating_mul(self.arena.active_point_count() as usize)
                    .saturating_mul(2 * std::mem::size_of::<f64>()) as u64,
            );
        }
        Ok(())
    }

    pub(crate) fn copy_current_to_state(
        &mut self,
        point_count: usize,
        global_parameter_count: usize,
        state: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        if !self.current_layout.is_identity() {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena full-state developer output requires an identity current \
                 layout",
            ));
        }
        if state.len() != point_count.saturating_mul(global_parameter_count) {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena output state shape is invalid",
            ));
        }
        let stride = self.arena.point_stride() as usize;
        let (current_re, current_im) = self.arena.current_slices();
        for component in 0..self.value_component_count {
            let plane = component * stride;
            for point in 0..point_count {
                state[point * global_parameter_count + component] =
                    Complex::new(current_re[plane + point], current_im[plane + point]);
            }
        }
        self.traffic.boundary_current_output_bytes =
            self.traffic.boundary_current_output_bytes.saturating_add(
                point_count
                    .saturating_mul(self.value_component_count)
                    .saturating_mul(2 * std::mem::size_of::<f64>()) as u64,
            );
        Ok(())
    }

    /// Borrow the canonical split-complex planes directly for the plane-native
    /// reducer. No amplitude packing or traffic accounting is involved.
    pub(crate) fn amplitude_planes(&self) -> RusticolResult<DirectAmplitudePlanes<'_>> {
        let (amplitude_re, amplitude_im) = self.arena.amplitude_slices();
        DirectAmplitudePlanes::new(
            amplitude_re,
            amplitude_im,
            self.arena.point_stride(),
            self.arena.active_point_count(),
        )
    }

    /// Explicit developer-only AoS extraction, excluded from measured leaf and
    /// plane-native reduction paths.
    pub(crate) fn extract_amplitudes_row_major(
        &mut self,
        point_count: usize,
        output: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        if output.len() != point_count.saturating_mul(self.amplitude_component_count) {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena amplitude output shape is invalid",
            ));
        }
        let stride = self.arena.point_stride() as usize;
        let (amplitude_re, amplitude_im) = self.arena.amplitude_slices();
        for component in 0..self.amplitude_component_count {
            let plane = component * stride;
            for point in 0..point_count {
                output[point * self.amplitude_component_count + component] =
                    Complex::new(amplitude_re[plane + point], amplitude_im[plane + point]);
            }
        }
        self.traffic.boundary_amplitude_output_bytes =
            self.traffic.boundary_amplitude_output_bytes.saturating_add(
                point_count
                    .saturating_mul(self.amplitude_component_count)
                    .saturating_mul(2 * std::mem::size_of::<f64>()) as u64,
            );
        Ok(())
    }
}

#[cfg(feature = "f64-symjit")]
pub(crate) fn compiled_direct_symjit_supported(
    evaluators: &GenericStageEvaluatorArtifactsManifest,
) -> RusticolResult<bool> {
    for stage in evaluators
        .stages
        .iter()
        .chain(std::iter::once(&evaluators.amplitude_stage))
    {
        for leaf in stage.evaluator.leaf_layout()? {
            if !matches!(leaf.evaluator, EvaluatorManifest::SymjitApplication { .. }) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Execute the momentum sums already recorded in the process output. Only
/// the point loop is transposed: label order, crossings and the initial
/// positive-zero addition remain identical to the scalar implementation.
fn fill_compiled_momentum_planes(
    batch: F64MomentumBatchView<'_>,
    slots: &[GenericMomentumSlotManifest],
    external_is_initial: &[bool],
    component_count: usize,
    stride: usize,
    momenta: &mut [f64],
) -> RusticolResult<()> {
    let point_count = batch.point_count();
    if stride < point_count || component_count.checked_mul(stride) != Some(momenta.len()) {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena momentum plane shape is invalid",
        ));
    }
    for slot in slots {
        if slot.component_stop.checked_sub(slot.component_start) != Some(4)
            || slot.component_stop > component_count
        {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena momentum slot {} has an invalid component range",
                slot.momentum_slot_id
            )));
        }
        if slot.external_labels.is_empty() {
            for component in slot.component_start..slot.component_stop {
                let start = component * stride;
                momenta[start..start + point_count].fill(0.0);
            }
            continue;
        }
        if point_count == 1 {
            // A singleton has no point-plane loop to vectorize. Read all four
            // components and resolve its public crossing once per term, as in
            // the original scalar accumulator, after static slot pruning.
            let point = batch.point(0);
            let mut value = [0.0; 4];
            for &label in &slot.external_labels {
                let (external_index, sign) = compiled_momentum_term(
                    label,
                    slot.momentum_slot_id,
                    batch.external_count(),
                    external_is_initial,
                )?;
                let momentum = point
                    .momentum(external_index)
                    .ok_or_else(|| RusticolError::integrity("compiled momentum input is absent"))?;
                for (sum, component) in value.iter_mut().zip(momentum) {
                    *sum += sign * component;
                }
            }
            for (component, value) in value.into_iter().enumerate() {
                momenta[(slot.component_start + component) * stride] = value;
            }
            continue;
        }
        for (term, label) in slot.external_labels.iter().enumerate() {
            let (external_index, sign) = compiled_momentum_term(
                *label,
                slot.momentum_slot_id,
                batch.external_count(),
                external_is_initial,
            )?;
            for component in 0..4 {
                let start = (slot.component_start + component) * stride;
                let output = &mut momenta[start..start + point_count];
                if term == 0 {
                    // Avoid a separate zero-fill pass, especially for the
                    // common single-external-momentum slots.
                    batch.accumulate_component_into::<false>(
                        external_index,
                        component,
                        sign,
                        output,
                    )?;
                } else {
                    batch.accumulate_component_into::<true>(
                        external_index,
                        component,
                        sign,
                        output,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn compiled_momentum_term(
    label: usize,
    momentum_slot_id: usize,
    external_count: usize,
    external_is_initial: &[bool],
) -> RusticolResult<(usize, f64)> {
    let external_index = label.checked_sub(1).ok_or_else(|| {
        RusticolError::integrity("compiled Direct-Arena momentum labels are one-based")
    })?;
    if external_index >= external_count {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena momentum slot {momentum_slot_id} references absent external leg {label}"
        )));
    }
    let sign = if *external_is_initial.get(external_index).ok_or_else(|| {
        RusticolError::integrity("compiled Direct-Arena external-side metadata is incomplete")
    })? {
        -1.0
    } else {
        1.0
    };
    Ok((external_index, sign))
}

fn canonical_source_layout(
    sources: &[GenericSourceRecordManifest],
    value_component_count: usize,
) -> RusticolResult<(Vec<usize>, usize)> {
    if sources.is_empty() {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena runtime schema has no sources",
        ));
    }
    let mut source_components = Vec::new();
    let mut max_dimension = 0usize;
    for (source_index, source) in sources.iter().enumerate() {
        if source.source_id != source_index
            || source.value_slot.current_id != source.current_id
            || source.value_slot.component_start != source.current_component_start
            || source.value_slot.component_stop != source.current_component_stop
        {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena source {source_index} does not preserve canonical current ownership"
            )));
        }
        let start = source.value_slot.component_start;
        let stop = source.value_slot.component_stop;
        let dimension = stop
            .checked_sub(start)
            .filter(|value| *value != 0)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "compiled Direct-Arena source {source_index} has an empty or reversed range"
                ))
            })?;
        if stop > value_component_count || dimension != source.dimension {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena source {source_index} exceeds canonical current storage"
            )));
        }
        source_components.extend(start..stop);
        max_dimension = max_dimension.max(dimension);
    }
    if source_components
        .windows(2)
        .any(|window| window[0] >= window[1])
    {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena source component ownership overlaps or is unordered",
        ));
    }
    Ok((source_components, max_dimension))
}

fn canonical_structural_zero_components(
    stages: &[GenericSerializedStageEvaluatorManifest],
    source_components: &[usize],
    value_component_count: usize,
) -> RusticolResult<BTreeSet<usize>> {
    let sources = source_components.iter().copied().collect::<BTreeSet<_>>();
    let mut produced = BTreeSet::new();
    for stage in stages {
        for slot in &stage.output_slots {
            let output_len = slot
                .output_stop
                .checked_sub(slot.output_start)
                .ok_or_else(|| {
                    RusticolError::integrity("compiled Direct-Arena output slot range underflows")
                })?;
            if slot.component_stop.checked_sub(slot.component_start) != Some(output_len)
                || slot.component_stop > value_component_count
            {
                return Err(RusticolError::integrity(format!(
                    "compiled Direct-Arena stage {:?} has an invalid canonical output range",
                    stage.evaluator_label
                )));
            }
            for component in slot.component_start..slot.component_stop {
                if sources.contains(&component) {
                    return Err(RusticolError::integrity(format!(
                        "compiled Direct-Arena stage {:?} writes canonical source component \
                         {component}",
                        stage.evaluator_label
                    )));
                }
                produced.insert(component);
            }
        }
    }
    Ok((0..value_component_count)
        .filter(|component| !sources.contains(component) && !produced.contains(component))
        .collect())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompiledDirectCurrentLifetime {
    canonical_component: usize,
    first_event: u64,
    last_event: u64,
}

fn derive_stage_current_layout(
    stages: &[LoadedStage],
    amplitude: &LoadedStage,
    source_components: &[usize],
    canonical_component_count: usize,
) -> RusticolResult<CompiledDirectCurrentLayout> {
    let mut first_events = vec![None; canonical_component_count];
    let mut last_events = vec![None; canonical_component_count];
    for &component in source_components {
        define_current_component(
            &mut first_events,
            &mut last_events,
            component,
            0,
            "source fill",
        )?;
    }

    for (stage_index, stage) in stages.iter().enumerate() {
        let event = u64::try_from(stage_index)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                RusticolError::integrity("compiled Direct-Arena stage-liveness event exceeds u64")
            })?;
        for leaf in &stage.leaf_plans {
            for &component in &leaf.input_current_components {
                use_current_component(
                    &first_events,
                    &mut last_events,
                    component,
                    event,
                    "current stage input",
                )?;
            }
        }
        // Arena-v1 declares no input/output or output/output aliasing for the
        // complete fused stage, not merely for one leaf. Give every output the
        // same inclusive definition event so the allocator preserves that
        // whole-stage contract.
        for leaf in &stage.leaf_plans {
            for &component in &leaf.output_current_components {
                define_current_component(
                    &mut first_events,
                    &mut last_events,
                    component,
                    event,
                    "current stage output",
                )?;
            }
        }
    }

    let amplitude_event = u64::try_from(stages.len())
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            RusticolError::integrity("compiled Direct-Arena amplitude-liveness event exceeds u64")
        })?;
    for leaf in &amplitude.leaf_plans {
        for &component in &leaf.input_current_components {
            use_current_component(
                &first_events,
                &mut last_events,
                component,
                amplitude_event,
                "amplitude stage input",
            )?;
        }
    }

    let mut lifetimes = first_events
        .into_iter()
        .zip(last_events)
        .enumerate()
        .filter_map(|(canonical_component, (first_event, last_event))| {
            first_event.map(|first_event| {
                last_event
                    .map(|last_event| CompiledDirectCurrentLifetime {
                        canonical_component,
                        first_event,
                        last_event,
                    })
                    .ok_or_else(|| {
                        RusticolError::integrity(format!(
                            "compiled Direct-Arena current component {canonical_component} has a \
                             definition but no lifetime end"
                        ))
                    })
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if lifetimes.is_empty() {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena stage liveness has no live current components",
        ));
    }
    lifetimes.sort_unstable_by_key(|lifetime| (lifetime.first_event, lifetime.canonical_component));

    // Unit-width interval coloring in O(N log N). At every definition event,
    // release only ranges whose inclusive lifetime ended strictly earlier and
    // deterministically reuse the lowest free physical plane.
    let mut active = BinaryHeap::<Reverse<(u64, u32)>>::new();
    let mut free = BinaryHeap::<Reverse<u32>>::new();
    let mut next_physical = 0_u32;
    let mut canonical_to_physical = vec![UNMAPPED_CURRENT_COMPONENT; canonical_component_count];
    for lifetime in &lifetimes {
        while let Some(Reverse((last_event, physical_component))) = active.peek().copied() {
            if last_event >= lifetime.first_event {
                break;
            }
            active.pop();
            free.push(Reverse(physical_component));
        }
        let physical_component = if let Some(Reverse(component)) = free.pop() {
            component
        } else {
            let component = next_physical;
            next_physical = next_physical.checked_add(1).ok_or_else(|| {
                RusticolError::integrity(
                    "compiled Direct-Arena physical current plane count exceeds u32",
                )
            })?;
            component
        };
        canonical_to_physical[lifetime.canonical_component] = physical_component;
        active.push(Reverse((lifetime.last_event, physical_component)));
    }

    let layout = CompiledDirectCurrentLayout {
        canonical_to_physical: canonical_to_physical.into_boxed_slice(),
        physical_component_count: next_physical,
        live_component_count: lifetimes.len(),
    };
    validate_compiled_current_layout(&layout, &lifetimes, stages, amplitude)?;
    Ok(layout)
}

fn define_current_component(
    first_events: &mut [Option<u64>],
    last_events: &mut [Option<u64>],
    component: usize,
    event: u64,
    context: &str,
) -> RusticolResult<()> {
    let first = first_events.get_mut(component).ok_or_else(|| {
        RusticolError::integrity(format!(
            "compiled Direct-Arena {context} component {component} is out of bounds"
        ))
    })?;
    let last = last_events.get_mut(component).ok_or_else(|| {
        RusticolError::integrity(format!(
            "compiled Direct-Arena {context} component {component} is out of bounds"
        ))
    })?;
    if first.replace(event).is_some() || last.replace(event).is_some() {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena {context} component {component} has multiple producers"
        )));
    }
    Ok(())
}

fn use_current_component(
    first_events: &[Option<u64>],
    last_events: &mut [Option<u64>],
    component: usize,
    event: u64,
    context: &str,
) -> RusticolResult<()> {
    let first = first_events
        .get(component)
        .copied()
        .flatten()
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "compiled Direct-Arena {context} component {component} is absent or used before \
                 its producer"
            ))
        })?;
    if first > event {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena {context} component {component} precedes its producer"
        )));
    }
    let last = last_events.get_mut(component).ok_or_else(|| {
        RusticolError::integrity(format!(
            "compiled Direct-Arena {context} component {component} is out of bounds"
        ))
    })?;
    *last = Some(last.unwrap_or(first).max(event));
    Ok(())
}

fn validate_compiled_current_layout(
    layout: &CompiledDirectCurrentLayout,
    lifetimes: &[CompiledDirectCurrentLifetime],
    stages: &[LoadedStage],
    amplitude: &LoadedStage,
) -> RusticolResult<()> {
    if layout.physical_component_count == 0
        || layout.live_component_count != lifetimes.len()
        || layout.canonical_to_physical.is_empty()
    {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena current-liveness accounting is inconsistent",
        ));
    }
    let mut physical_chains =
        vec![Vec::<CompiledDirectCurrentLifetime>::new(); layout.physical_component_count as usize];
    for lifetime in lifetimes {
        if lifetime.first_event > lifetime.last_event {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena current component {} has a reversed lifetime",
                lifetime.canonical_component
            )));
        }
        let physical = layout.physical_component(lifetime.canonical_component)?;
        physical_chains
            .get_mut(physical as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "compiled Direct-Arena liveness assignment exceeds physical storage",
                )
            })?
            .push(*lifetime);
    }
    for chain in &mut physical_chains {
        chain.sort_unstable_by_key(|lifetime| (lifetime.first_event, lifetime.canonical_component));
        for pair in chain.windows(2) {
            if pair[0].last_event >= pair[1].first_event {
                return Err(RusticolError::integrity(format!(
                    "compiled Direct-Arena live current components {} and {} alias physical plane",
                    pair[0].canonical_component, pair[1].canonical_component
                )));
            }
        }
    }
    for (stage_index, stage) in stages.iter().enumerate() {
        validate_stage_physical_aliases(
            layout,
            stage,
            &format!("compiled Direct-Arena stage {stage_index}"),
        )?;
    }
    validate_stage_physical_aliases(layout, amplitude, "compiled Direct-Arena amplitude stage")?;
    Ok(())
}

fn validate_stage_physical_aliases(
    layout: &CompiledDirectCurrentLayout,
    stage: &LoadedStage,
    context: &str,
) -> RusticolResult<()> {
    let mut canonical_inputs = BTreeSet::new();
    let mut canonical_outputs = BTreeSet::new();
    for leaf in &stage.leaf_plans {
        for &component in &leaf.input_current_components {
            canonical_inputs.insert(component);
        }
        for &component in &leaf.output_current_components {
            if !canonical_outputs.insert(component) {
                return Err(RusticolError::integrity(format!(
                    "{context} repeats an output current component"
                )));
            }
        }
    }
    let mut inputs = BTreeSet::new();
    for component in canonical_inputs {
        if !inputs.insert(layout.physical_component(component)?) {
            return Err(RusticolError::integrity(format!(
                "{context} aliases distinct live input current components"
            )));
        }
    }
    let mut outputs = BTreeSet::new();
    for component in canonical_outputs {
        if !outputs.insert(layout.physical_component(component)?) {
            return Err(RusticolError::integrity(format!(
                "{context} aliases distinct output current components"
            )));
        }
    }
    if inputs.iter().any(|component| outputs.contains(component)) {
        return Err(RusticolError::integrity(format!(
            "{context} aliases an input and output current component"
        )));
    }
    Ok(())
}

fn point_count_u32(point_count: usize, active: u32) -> RusticolResult<u32> {
    let point_count = u32::try_from(point_count)
        .map_err(|_| RusticolError::invalid_argument("compiled point count exceeds u32"))?;
    if point_count == 0 || point_count != active {
        return Err(RusticolError::invalid_argument(
            "compiled Direct-Arena point count does not match the active tile",
        ));
    }
    Ok(point_count)
}

fn evaluate_bound_stage_leaves(
    stage: &mut BoundStage,
    point_count: u32,
    selected: &[usize],
    traffic: &mut CompiledDirectPrototypeTraffic,
) -> RusticolResult<()> {
    for &leaf_index in selected {
        let broadcast_bytes = stage.leaves[leaf_index].evaluate(0, point_count)?;
        traffic.scalar_broadcast_fill_bytes = traffic
            .scalar_broadcast_fill_bytes
            .checked_add(broadcast_bytes)
            .ok_or_else(|| {
                RusticolError::internal("compiled scalar broadcast traffic counter overflows u64")
            })?;
        traffic.leaf.record_call(1, point_count);
    }
    Ok(())
}

fn intersect_helicity_color_schedules(
    helicity: &CompiledDirectValidatedSchedule,
    colors: &[&CompiledDirectValidatedSchedule],
) -> RusticolResult<(Vec<Vec<usize>>, Vec<usize>)> {
    if colors
        .iter()
        .any(|color| color.active_stage_leaves.len() != helicity.active_stage_leaves.len())
    {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena selector schedules have different stage counts",
        ));
    }
    let intersect = |selected: &[usize], union: BTreeSet<usize>| {
        selected
            .iter()
            .copied()
            .filter(|id| union.contains(id))
            .collect()
    };
    let stages = helicity
        .active_stage_leaves
        .iter()
        .enumerate()
        .map(|(index, leaves)| {
            intersect(
                leaves,
                colors
                    .iter()
                    .flat_map(|color| color.active_stage_leaves[index].iter().copied())
                    .collect(),
            )
        })
        .collect();
    let amplitude = intersect(
        &helicity.active_amplitude_leaves,
        colors
            .iter()
            .flat_map(|color| color.active_amplitude_leaves.iter().copied())
            .collect(),
    );
    Ok((stages, amplitude))
}

fn validate_schedule<S: DirectStagePlan>(
    stages: &[S],
    amplitude: &S,
    value_component_count: usize,
    source_components: &[usize],
    amplitude_component_count: usize,
    active_stage_leaves: &[Vec<usize>],
    active_amplitude_leaves: &[usize],
) -> RusticolResult<CompiledDirectValidatedSchedule> {
    if active_stage_leaves.len() != stages.len() {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena schedule has the wrong stage count",
        ));
    }
    let mut produced = BTreeSet::new();
    for (stage_index, stage) in stages.iter().enumerate() {
        for (leaf_index, leaf) in stage.leaf_plans().iter().enumerate() {
            for &component in &leaf.output_current_components {
                if !produced.insert(component) {
                    return Err(RusticolError::integrity(format!(
                        "compiled Direct-Arena stage {stage_index} leaf {leaf_index} duplicates \
                         producer for current component {component}"
                    )));
                }
            }
        }
    }
    let declared_sources = source_components.iter().copied().collect::<BTreeSet<_>>();
    if declared_sources.len() != source_components.len()
        || declared_sources
            .iter()
            .any(|component| *component >= value_component_count || produced.contains(component))
    {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena source ownership aliases or exceeds produced current storage",
        ));
    }
    // Value storage is a stable global numbering space and can contain
    // components omitted by closure pruning. Such holes are neither sources
    // nor produced values. They are valid only while no selected leaf reads
    // them; the ordered prerequisite checks below enforce that invariant.
    let mut available = declared_sources;
    let mut validated_stages = Vec::with_capacity(stages.len());
    for (stage_index, (stage, selected)) in stages.iter().zip(active_stage_leaves).enumerate() {
        validate_sorted_leaf_indices(
            selected,
            stage.leaf_plans().len(),
            &format!("compiled Direct-Arena stage {stage_index} schedule"),
        )?;
        let mut stage_outputs = BTreeSet::new();
        for &leaf_index in selected {
            let leaf = &stage.leaf_plans()[leaf_index];
            let missing = leaf
                .input_current_components
                .iter()
                .copied()
                .filter(|component| !available.contains(component))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(RusticolError::integrity(format!(
                    "compiled Direct-Arena stage {stage_index} leaf {leaf_index} is missing \
                     prerequisite current components {missing:?}"
                )));
            }
            stage_outputs.extend(leaf.output_current_components.iter().copied());
        }
        available.extend(stage_outputs);
        validated_stages.push(selected.clone().into_boxed_slice());
    }
    validate_sorted_leaf_indices(
        active_amplitude_leaves,
        amplitude.leaf_plans().len(),
        "compiled Direct-Arena amplitude schedule",
    )?;
    let mut selected_amplitudes = BTreeSet::new();
    for &leaf_index in active_amplitude_leaves {
        let leaf = &amplitude.leaf_plans()[leaf_index];
        let missing = leaf
            .input_current_components
            .iter()
            .copied()
            .filter(|component| !available.contains(component))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena amplitude leaf {leaf_index} is missing prerequisite \
                 current components {missing:?}"
            )));
        }
        selected_amplitudes.extend(leaf.output_amplitude_components.iter().copied());
    }
    if selected_amplitudes
        .iter()
        .any(|component| *component >= amplitude_component_count)
    {
        return Err(RusticolError::integrity(
            "compiled Direct-Arena amplitude schedule exceeds canonical output planes",
        ));
    }
    let inactive_amplitude_components = (0..amplitude_component_count)
        .filter(|component| !selected_amplitudes.contains(component))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(CompiledDirectValidatedSchedule {
        active_stage_leaves: validated_stages.into_boxed_slice(),
        active_amplitude_leaves: active_amplitude_leaves.to_vec().into_boxed_slice(),
        inactive_amplitude_components,
    })
}

fn validate_sorted_leaf_indices(
    selected: &[usize],
    leaf_count: usize,
    label: &str,
) -> RusticolResult<()> {
    let mut previous = None;
    for &index in selected {
        if index >= leaf_count || previous.is_some_and(|value| value >= index) {
            return Err(RusticolError::integrity(format!(
                "{label} is not sorted, unique, and in bounds"
            )));
        }
        previous = Some(index);
    }
    Ok(())
}

fn evaluate_validated_schedule(
    stages: &mut [BoundStage],
    amplitude: &mut BoundStage,
    schedule: &CompiledDirectValidatedSchedule,
    point_count: u32,
    traffic: &mut CompiledDirectPrototypeTraffic,
) -> RusticolResult<()> {
    for (stage, selected) in stages.iter_mut().zip(&schedule.active_stage_leaves) {
        evaluate_bound_stage_leaves(stage, point_count, selected, traffic)?;
    }
    evaluate_bound_stage_leaves(
        amplitude,
        point_count,
        &schedule.active_amplitude_leaves,
        traffic,
    )?;
    traffic.leaf.validate_direct()
}

fn load_stage(
    stage: &GenericSerializedStageEvaluatorManifest,
    is_amplitude: bool,
    payloads: &EvaluatorPayloadStore,
    value_component_count: usize,
    momentum_component_count: usize,
    amplitude_component_count: usize,
    structural_zero_components: &BTreeSet<usize>,
) -> RusticolResult<LoadedStage> {
    let direct = stage.compiled_plane_arena.as_ref().ok_or_else(|| {
        RusticolError::integrity(format!(
            "compiled Direct-Arena stage {:?} has no serialized plane bindings",
            stage.evaluator_label
        ))
    })?;
    if stage.parameter_layout != "stage-local-value-momentum"
        || direct.input_bindings.len() != stage.parameter_count
    {
        return Err(RusticolError::compatibility(format!(
            "compiled Direct-Arena stage {:?} requires complete serialized input bindings",
            stage.evaluator_label
        )));
    }
    let mut components = vec![None; stage.parameter_count];
    for component in &direct.input_bindings {
        if component.parameter_index >= components.len()
            || components[component.parameter_index]
                .replace(component)
                .is_some()
        {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} has invalid serialized input bindings",
                stage.evaluator_label
            )));
        }
    }
    if components.iter().any(Option::is_none) {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena stage {:?} has incomplete serialized input bindings",
            stage.evaluator_label
        )));
    }

    let output_limit = if is_amplitude {
        amplitude_component_count
    } else {
        value_component_count
    };
    let mut output_components = vec![None; stage.output_length];
    let mut seen_output_components = BTreeSet::new();
    let expected_arena = if is_amplitude { "amplitude" } else { "current" };
    for binding in &direct.output_bindings {
        if binding.output_index >= output_components.len()
            || binding.component >= output_limit
            || binding.arena != expected_arena
        {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} has an invalid serialized output binding",
                stage.evaluator_label
            )));
        }
        if !seen_output_components.insert(binding.component) {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} aliases output component {}",
                stage.evaluator_label, binding.component
            )));
        }
        if output_components[binding.output_index]
            .replace(binding)
            .is_some()
        {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} has duplicate serialized output indices",
                stage.evaluator_label
            )));
        }
    }
    if output_components.iter().any(Option::is_none) {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena stage {:?} does not cover every output",
            stage.evaluator_label
        )));
    }
    if is_amplitude
        && (stage.output_length != amplitude_component_count
            || seen_output_components.len() != amplitude_component_count
            || seen_output_components
                .iter()
                .copied()
                .ne(0..amplitude_component_count))
    {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena amplitude stage {:?} is not a complete canonical output \
             permutation",
            stage.evaluator_label
        )));
    }

    let leaf_layout = stage.evaluator.leaf_layout()?;
    let mut loaded = Vec::with_capacity(leaf_layout.len());
    let mut leaf_plans = Vec::with_capacity(leaf_layout.len());
    let mut output_cursor = 0usize;
    for (leaf_index, leaf) in leaf_layout.into_iter().enumerate() {
        let (input_len, output_len) = leaf.evaluator.io_len()?;
        let direct_leaf = direct.leaves.get(leaf_index).ok_or_else(|| {
            RusticolError::integrity("compiled plane-arena leaf bindings are incomplete")
        })?;
        let direct_input_indices = direct_leaf.input_indices.as_slice();
        let direct_output_range = direct_leaf.output_start..direct_leaf.output_stop;
        let direct_input_len = direct_leaf.input_len;
        let direct_output_len = direct_leaf.output_len;
        if direct_input_indices.len() != direct_input_len {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} leaf input map has the wrong length",
                stage.evaluator_label
            )));
        }
        if direct_input_len != input_len
            || direct_output_len != output_len
            || direct_input_indices != leaf.input_indices
            || direct_output_range != leaf.output_range
        {
            return Err(RusticolError::integrity(format!(
                "compiled Direct-Arena stage {:?} leaf bindings disagree with canonical layout",
                stage.evaluator_label
            )));
        }
        if direct_output_range.start != output_cursor
            || direct_output_range.end
                != output_cursor
                    .checked_add(direct_output_len)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "compiled Direct-Arena leaf output range overflows",
                        )
                    })?
            || direct_output_range.end > output_components.len()
        {
            return Err(RusticolError::integrity(
                "compiled Direct-Arena canonical leaf output range disagrees with the stage",
            ));
        }
        let output_stop = direct_output_range.end;
        let scalar_values_per_point = direct_input_len
            .checked_add(direct_output_len)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                RusticolError::integrity("compiled Direct-Arena leaf-local footprint overflows")
            })?;

        let mut input_currents = BTreeSet::new();
        let mut input_components = Vec::with_capacity(direct_input_len);
        for &parameter_index in direct_input_indices {
            let component = components
                .get(parameter_index)
                .and_then(|value| *value)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena leaf references an absent stage input",
                    )
                })?;
            if component.kind == "value"
                && !structural_zero_components.contains(&component.global_component)
            {
                input_currents.insert(component.global_component);
            }
            input_components.push(component);
        }
        let canonical_outputs = output_components[output_cursor..output_stop]
            .iter()
            .map(|value| value.expect("output coverage validated"))
            .collect::<Vec<_>>();

        let loaded_leaf = match leaf.evaluator {
            #[cfg(feature = "f64-symjit")]
            EvaluatorManifest::SymjitApplication {
                runtime_capability,
                application_abi,
                element_layout,
                batch_layout,
                compiler_type,
                translation_mode,
                optimization_level,
                word_bits,
                endianness,
                required_defuns,
                plane_application,
                ..
            } => {
                if *optimization_level > 3 {
                    return Err(RusticolError::compatibility(
                        "compiled-plane-arena-v1 supports compiled JIT optimization levels 0 \
                         through 3",
                    ));
                }
                validate_manifest_metadata(&SymjitApplicationMetadata {
                    runtime_capability,
                    application_abi,
                    input_len,
                    output_len,
                    element_layout,
                    batch_layout,
                    compiler_type,
                    translation_mode,
                    optimization_level: *optimization_level,
                    word_bits: *word_bits,
                    endianness,
                    required_defuns,
                })?;
                let plane_application = plane_application.as_ref().ok_or_else(|| {
                    RusticolError::compatibility(
                        "compiled SymJIT leaf predates the standard plane \
                         application ABI; regenerate the artifact",
                    )
                })?;
                plane_application.validate(input_len, output_len, *optimization_level)?;
                if direct_leaf.application_path != plane_application.application_path
                    || direct_leaf.source_application_abi != plane_application.application_abi
                    || direct_leaf.source_application_abi != direct.source_application_abi
                    || direct.source_application_abi != SYMJIT_PLANE_APPLICATION_ABI
                    || direct.application_abi != COMPILED_PLANE_DIRECT_APPLICATION_ABI
                    || direct_leaf.optimization_level != *optimization_level
                    || direct_leaf.direct_codegen_optimization_level != *optimization_level
                {
                    return Err(RusticolError::integrity(
                        "compiled SymJIT Direct-Arena leaf identity is inconsistent",
                    ));
                }
                let mut source_inputs = Vec::with_capacity(direct_input_len * 2);
                let mut plane_bindings = Vec::new();
                let mut scalar_bindings = Vec::new();
                for component in &input_components {
                    append_component_bindings(
                        component,
                        value_component_count,
                        momentum_component_count,
                        structural_zero_components,
                        &mut source_inputs,
                        &mut plane_bindings,
                        &mut scalar_bindings,
                    )?;
                }
                let outputs = symjit_output_bindings(&canonical_outputs)?;
                let source = payloads.source(&plane_application.application_path)?;
                let bytes = source.read()?;
                LoadedCompiledDirectLeaf::Symjit(
                    LoadedSymjitCompiledDirectStage::load_source_bytes(
                        bytes.as_ref(),
                        PathBuf::from(source.display_name()),
                        &direct_leaf.source_application_abi,
                        direct_leaf.optimization_level,
                        plane_application.compression,
                        source_inputs,
                        plane_bindings,
                        scalar_bindings,
                        outputs,
                    )?,
                )
            }
            #[cfg(feature = "f64-compiled")]
            EvaluatorManifest::CompiledComplex {
                runtime_capability,
                function_name,
                number_type,
                native_direct_application: Some(application),
                ..
            } => {
                if !matches!(
                    runtime_capability.as_str(),
                    SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY
                        | SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY
                ) || number_type != "complex"
                {
                    return Err(RusticolError::compatibility(
                        "compiled native Direct-Arena leaf has an incompatible backend",
                    ));
                }
                application.validate(function_name, input_len, output_len)?;
                if direct_leaf.optimization_level != 3
                    || direct_leaf.direct_codegen_optimization_level != 3
                    || direct_leaf.application_path != application.library_path
                    || direct_leaf.source_application_abi != application.application_abi
                    || direct.source_application_abi != application.application_abi
                    || direct.application_abi != application.application_abi
                {
                    return Err(RusticolError::integrity(
                        "compiled native Direct-Arena leaf identity is inconsistent",
                    ));
                }
                let mut plane_bindings = Vec::new();
                let mut scalar_bindings = Vec::new();
                for component in &input_components {
                    append_native_component_bindings(
                        component,
                        value_component_count,
                        momentum_component_count,
                        structural_zero_components,
                        &mut plane_bindings,
                        &mut scalar_bindings,
                    )?;
                }
                let outputs = native_output_bindings(&canonical_outputs)?;
                let library = payloads.load_native_library(&application.library_path)?;
                LoadedCompiledDirectLeaf::Native(LoadedNativeCompiledDirectStage::load(
                    library,
                    function_name,
                    &application.application_abi,
                    plane_bindings,
                    scalar_bindings,
                    outputs,
                )?)
            }
            #[cfg(feature = "f64-compiled")]
            EvaluatorManifest::CompiledComplex {
                native_direct_application: None,
                ..
            } => {
                return Err(RusticolError::compatibility(
                    "compiled native Direct-Arena leaf has no plane-native DirectApplication; \
                     regenerate the artifact",
                ));
            }
            _ => {
                return Err(RusticolError::compatibility(format!(
                    "compiled Direct-Arena stage {:?} contains an unsupported leaf",
                    stage.evaluator_label
                )));
            }
        };
        let persistent_scalar_values_per_point = loaded_leaf.persistent_scalar_values_per_point();
        loaded.push(loaded_leaf);
        leaf_plans.push(DirectLeafPlan {
            input_current_components: input_currents
                .into_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            output_current_components: if is_amplitude {
                Box::new([])
            } else {
                output_components[output_cursor..output_stop]
                    .iter()
                    .map(|value| value.expect("output coverage validated").component)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            },
            output_amplitude_components: if is_amplitude {
                output_components[output_cursor..output_stop]
                    .iter()
                    .map(|value| value.expect("output coverage validated").component)
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            } else {
                Box::new([])
            },
            scalar_values_per_point,
            persistent_scalar_values_per_point,
        });
        output_cursor = output_stop;
    }
    if output_cursor != stage.output_length {
        return Err(RusticolError::integrity(format!(
            "compiled Direct-Arena stage {:?} leaf outputs do not cover the stage",
            stage.evaluator_label
        )));
    }
    Ok(LoadedStage {
        leaves: loaded,
        leaf_plans,
    })
}

#[cfg(feature = "f64-symjit")]
fn symjit_output_bindings(
    outputs: &[&CompiledPlaneOutputBindingManifest],
) -> RusticolResult<Vec<CompiledDirectOutputBinding>> {
    let mut bindings = Vec::with_capacity(outputs.len() * 2);
    for output in outputs {
        let component = u32::try_from(output.component)
            .map_err(|_| RusticolError::integrity("compiled output plane index exceeds u32"))?;
        let plane = |imaginary| match output.arena.as_str() {
            "current" => Ok(CompiledDirectArenaPlane::Current {
                component,
                imaginary,
            }),
            "amplitude" => Ok(CompiledDirectArenaPlane::Amplitude {
                component,
                imaginary,
            }),
            _ => Err(RusticolError::integrity(
                "compiled output binding names an unsupported arena",
            )),
        };
        bindings.push(CompiledDirectOutputBinding(plane(false)?));
        bindings.push(CompiledDirectOutputBinding(plane(true)?));
    }
    Ok(bindings)
}

#[cfg(feature = "f64-compiled")]
fn native_output_bindings(
    outputs: &[&CompiledPlaneOutputBindingManifest],
) -> RusticolResult<Vec<NativeCompiledDirectOutputBinding>> {
    let mut bindings = Vec::with_capacity(outputs.len() * 2);
    for output in outputs {
        let component = u32::try_from(output.component)
            .map_err(|_| RusticolError::integrity("compiled output plane index exceeds u32"))?;
        let plane = |imaginary| match output.arena.as_str() {
            "current" => Ok(NativeCompiledDirectArenaPlane::Current {
                component,
                imaginary,
            }),
            "amplitude" => Ok(NativeCompiledDirectArenaPlane::Amplitude {
                component,
                imaginary,
            }),
            _ => Err(RusticolError::integrity(
                "compiled output binding names an unsupported arena",
            )),
        };
        bindings.push(NativeCompiledDirectOutputBinding(plane(false)?));
        bindings.push(NativeCompiledDirectOutputBinding(plane(true)?));
    }
    Ok(bindings)
}

#[cfg(feature = "f64-compiled")]
fn append_native_component_bindings(
    component: &CompiledPlaneInputBindingManifest,
    value_component_count: usize,
    momentum_component_count: usize,
    structural_zero_components: &BTreeSet<usize>,
    planes: &mut Vec<NativeCompiledDirectPlaneBinding>,
    scalars: &mut Vec<NativeCompiledDirectScalarBinding>,
) -> RusticolResult<()> {
    match component.kind.as_str() {
        "value" => {
            if component.global_component >= value_component_count {
                return Err(RusticolError::integrity(
                    "compiled Direct-Arena value input is outside current planes",
                ));
            }
            if structural_zero_components.contains(&component.global_component) {
                planes.push(NativeCompiledDirectPlaneBinding::Zero);
                if !component.real_valued {
                    planes.push(NativeCompiledDirectPlaneBinding::Zero);
                }
                return Ok(());
            }
            let current = |imaginary| {
                NativeCompiledDirectPlaneBinding::Arena(NativeCompiledDirectArenaPlane::Current {
                    component: component.global_component as u32,
                    imaginary,
                })
            };
            planes.push(current(false));
            if !component.real_valued {
                planes.push(current(true));
            }
        }
        "momentum" => {
            let local = component
                .global_component
                .checked_sub(value_component_count)
                .filter(|index| *index < momentum_component_count)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena momentum input is outside momentum planes",
                    )
                })?;
            planes.push(NativeCompiledDirectPlaneBinding::Momentum {
                form: u32::try_from(local / 4)
                    .map_err(|_| RusticolError::integrity("compiled momentum form exceeds u32"))?,
                lorentz_component: (local % 4) as u16,
            });
            if !component.real_valued {
                planes.push(NativeCompiledDirectPlaneBinding::Zero);
            }
        }
        "model_parameter" => {
            let index = u32::try_from(component.source_id).map_err(|_| {
                RusticolError::integrity("compiled model parameter index exceeds u32")
            })?;
            scalars.push(NativeCompiledDirectScalarBinding::Parameter {
                index,
                imaginary: false,
            });
            if !component.real_valued {
                scalars.push(NativeCompiledDirectScalarBinding::Parameter {
                    index,
                    imaginary: true,
                });
            }
        }
        kind => {
            return Err(RusticolError::compatibility(format!(
                "compiled Direct-Arena does not support input kind {kind:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "f64-symjit")]
fn append_component_bindings(
    component: &CompiledPlaneInputBindingManifest,
    value_component_count: usize,
    momentum_component_count: usize,
    structural_zero_components: &BTreeSet<usize>,
    source_inputs: &mut Vec<CompiledDirectSourceInputBinding>,
    planes: &mut Vec<CompiledDirectPlaneBinding>,
    scalars: &mut Vec<CompiledDirectScalarBinding>,
) -> RusticolResult<()> {
    fn push_plane(
        source_inputs: &mut Vec<CompiledDirectSourceInputBinding>,
        planes: &mut Vec<CompiledDirectPlaneBinding>,
        binding: CompiledDirectPlaneBinding,
    ) -> RusticolResult<()> {
        let index = u32::try_from(planes.len()).map_err(|_| {
            RusticolError::integrity("compiled Direct-Arena plane binding count exceeds u32")
        })?;
        source_inputs.push(CompiledDirectSourceInputBinding::Plane(index));
        planes.push(binding);
        Ok(())
    }
    fn push_scalar(
        source_inputs: &mut Vec<CompiledDirectSourceInputBinding>,
        scalars: &mut Vec<CompiledDirectScalarBinding>,
        binding: CompiledDirectScalarBinding,
    ) -> RusticolResult<()> {
        let index = u32::try_from(scalars.len()).map_err(|_| {
            RusticolError::integrity("compiled Direct-Arena scalar binding count exceeds u32")
        })?;
        source_inputs.push(CompiledDirectSourceInputBinding::Scalar(index));
        scalars.push(binding);
        Ok(())
    }

    match component.kind.as_str() {
        "value" => {
            if component.global_component >= value_component_count {
                return Err(RusticolError::integrity(
                    "compiled Direct-Arena value input is outside current planes",
                ));
            }
            if structural_zero_components.contains(&component.global_component) {
                push_plane(source_inputs, planes, CompiledDirectPlaneBinding::Zero)?;
                push_plane(source_inputs, planes, CompiledDirectPlaneBinding::Zero)?;
                return Ok(());
            }
            let current = |imaginary| {
                CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                    component: component.global_component as u32,
                    imaginary,
                })
            };
            push_plane(source_inputs, planes, current(false))?;
            push_plane(source_inputs, planes, current(true))?;
        }
        "momentum" => {
            let local = component
                .global_component
                .checked_sub(value_component_count)
                .filter(|index| *index < momentum_component_count)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "compiled Direct-Arena momentum input is outside momentum planes",
                    )
                })?;
            push_plane(
                source_inputs,
                planes,
                CompiledDirectPlaneBinding::Momentum {
                    form: u32::try_from(local / 4).map_err(|_| {
                        RusticolError::integrity("compiled momentum form exceeds u32")
                    })?,
                    lorentz_component: (local % 4) as u16,
                },
            )?;
            push_plane(source_inputs, planes, CompiledDirectPlaneBinding::Zero)?;
        }
        "model_parameter" => {
            let index = u32::try_from(component.source_id).map_err(|_| {
                RusticolError::integrity("compiled model parameter index exceeds u32")
            })?;
            push_scalar(
                source_inputs,
                scalars,
                CompiledDirectScalarBinding::Parameter {
                    index,
                    imaginary: false,
                },
            )?;
            push_scalar(
                source_inputs,
                scalars,
                CompiledDirectScalarBinding::Parameter {
                    index,
                    imaginary: true,
                },
            )?;
        }
        kind => {
            return Err(RusticolError::compatibility(format!(
                "compiled Direct-Arena does not support input kind {kind:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "f64-symjit"))]
mod tests {
    fn momentum_slot(start: usize, labels: &[usize]) -> GenericMomentumSlotManifest {
        GenericMomentumSlotManifest {
            momentum_slot_id: start / 4,
            momentum_mask: 0,
            external_labels: labels.to_vec(),
            component_start: start,
            component_stop: start + 4,
            real_valued: true,
        }
    }

    #[test]
    fn compiled_momentum_planes_preserve_scalar_order_crossing_and_partial_tiles() {
        let nested = (0..19)
            .map(|point| {
                let offset = point as f64;
                vec![
                    [1e16, -0.0, 3.0 + offset, -1e16],
                    [-1e16, 0.0, 5.0 - offset, 1.0],
                    [1.0, -0.0, -7.0, 1e16],
                ]
            })
            .collect::<Vec<_>>();
        let flat = nested
            .iter()
            .flatten()
            .flatten()
            .copied()
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
        let views = [
            F64MomentumBatchView::from_nested(&nested, 3).unwrap(),
            F64MomentumBatchView::from_contiguous_prevalidated(&flat, 19, 3, None).unwrap(),
            F64MomentumBatchView::from_contiguous_prevalidated(&flat, 19, 3, Some(&crossing))
                .unwrap(),
        ];
        let slots = [
            momentum_slot(0, &[1]),
            momentum_slot(4, &[3, 1, 2]),
            momentum_slot(8, &[3, 2, 1, 3]),
            momentum_slot(16, &[]),
        ];
        let initial = [true, false, true];
        let stride = 24;
        let mut actual = vec![23.0_f64; 20 * stride];
        let pointer = actual.as_ptr();
        for view in views {
            for count in [1, 17, 3, 1] {
                let batch = view.subview(1, 1 + count).unwrap();
                let mut expected = actual.clone();
                for slot in &slots {
                    for point_index in 0..count {
                        let point = batch.point(point_index);
                        let mut momentum = [0.0; 4];
                        for &label in &slot.external_labels {
                            let external = label - 1;
                            let sign = if initial[external] { -1.0 } else { 1.0 };
                            for (target, value) in
                                momentum.iter_mut().zip(point.momentum(external).unwrap())
                            {
                                *target += sign * value;
                            }
                        }
                        for (component, value) in momentum.into_iter().enumerate() {
                            expected[(slot.component_start + component) * stride + point_index] =
                                value;
                        }
                    }
                }
                fill_compiled_momentum_planes(batch, &slots, &initial, 20, stride, &mut actual)
                    .unwrap();
                assert_eq!(actual.as_ptr(), pointer);
                for (&actual, expected) in actual.iter().zip(expected) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn compiled_momentum_planes_reject_invalid_generated_ranges_and_labels() {
        let values = [0.0; 24];
        let batch =
            F64MomentumBatchView::from_contiguous_prevalidated(&values, 2, 3, None).unwrap();
        let mut output = [0.0; 32];
        let valid = momentum_slot(0, &[1]);
        for slot in [
            momentum_slot(8, &[1]),
            momentum_slot(0, &[0]),
            momentum_slot(0, &[4]),
            GenericMomentumSlotManifest {
                component_stop: 3,
                ..valid.clone()
            },
            GenericMomentumSlotManifest {
                component_stop: 0,
                component_start: 4,
                ..valid.clone()
            },
        ] {
            for input in [batch, batch.subview(0, 1).unwrap()] {
                assert!(
                    fill_compiled_momentum_planes(
                        input,
                        std::slice::from_ref(&slot),
                        &[true; 3],
                        8,
                        4,
                        &mut output,
                    )
                    .is_err()
                );
            }
        }
        assert!(
            fill_compiled_momentum_planes(
                batch,
                std::slice::from_ref(&valid),
                &[],
                8,
                4,
                &mut output,
            )
            .is_err()
        );
        assert!(
            fill_compiled_momentum_planes(
                batch,
                std::slice::from_ref(&valid),
                &[true; 3],
                8,
                4,
                &mut output[..31],
            )
            .is_err()
        );
        assert!(
            fill_compiled_momentum_planes(batch, &[valid], &[true; 3], 8, 1, &mut output[..8],)
                .is_err()
        );
    }

    #[test]
    fn joint_selection_intersects_helicity_with_union_of_colors_in_leaf_order() {
        let schedule =
            |stages: Vec<Vec<usize>>, amplitudes: Vec<usize>| CompiledDirectValidatedSchedule {
                active_stage_leaves: stages.into_iter().map(Vec::into_boxed_slice).collect(),
                active_amplitude_leaves: amplitudes.into_boxed_slice(),
                inactive_amplitude_components: Box::new([]),
            };
        let helicity = schedule(vec![vec![0, 1, 3], vec![0, 2]], vec![1, 2, 4]);
        let first_color = schedule(vec![vec![0, 2], vec![0]], vec![1, 3]);
        let second_color = schedule(vec![vec![0, 3], vec![1, 2]], vec![2, 3]);
        let (stages, amplitudes) =
            intersect_helicity_color_schedules(&helicity, &[&first_color, &second_color]).unwrap();
        assert_eq!(stages, vec![vec![0, 3], vec![0, 2]]);
        assert_eq!(amplitudes, vec![1, 2]);
        let wrong_stages = schedule(vec![], vec![]);
        assert!(intersect_helicity_color_schedules(&helicity, &[&wrong_stages]).is_err());
    }
    use super::super::evaluator::count_test_allocations;
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;
    use symjit::{Compiler, Config, Expr, Storage};

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    fn liveness_leaf(inputs: &[usize], outputs: &[usize]) -> DirectLeafPlan {
        DirectLeafPlan {
            input_current_components: inputs.to_vec().into_boxed_slice(),
            output_current_components: outputs.to_vec().into_boxed_slice(),
            output_amplitude_components: Box::new([]),
            scalar_values_per_point: 1,
            persistent_scalar_values_per_point: 0,
        }
    }

    fn liveness_stage(leaves: Vec<DirectLeafPlan>) -> LoadedStage {
        LoadedStage {
            leaves: Vec::new(),
            leaf_plans: leaves,
        }
    }

    #[test]
    fn stage_liveness_reuses_only_strictly_dead_current_planes() {
        let stages = vec![
            liveness_stage(vec![liveness_leaf(&[0], &[1])]),
            liveness_stage(vec![liveness_leaf(&[1], &[2])]),
        ];
        let amplitude = liveness_stage(vec![liveness_leaf(&[2], &[])]);
        let expected = derive_stage_current_layout(&stages, &amplitude, &[0], 5).unwrap();

        assert_eq!(expected.physical_component_count, 2);
        assert_eq!(expected.live_component_count, 3);
        assert_eq!(expected.physical_component(0).unwrap(), 0);
        assert_eq!(expected.physical_component(1).unwrap(), 1);
        assert_eq!(expected.physical_component(2).unwrap(), 0);
        assert_eq!(
            expected.canonical_to_physical[3..],
            [UNMAPPED_CURRENT_COMPONENT, UNMAPPED_CURRENT_COMPONENT]
        );
        for _ in 0..32 {
            let actual = derive_stage_current_layout(&stages, &amplitude, &[0], 5).unwrap();
            assert_eq!(actual.canonical_to_physical, expected.canonical_to_physical);
            assert_eq!(
                actual.physical_component_count,
                expected.physical_component_count
            );
        }
    }

    #[test]
    fn stage_liveness_preserves_arena_v1_whole_stage_no_aliasing() {
        let stages = vec![
            liveness_stage(vec![liveness_leaf(&[0], &[1]), liveness_leaf(&[0], &[2])]),
            liveness_stage(vec![liveness_leaf(&[1, 2], &[3])]),
        ];
        let amplitude = liveness_stage(vec![liveness_leaf(&[3], &[])]);
        let layout = derive_stage_current_layout(&stages, &amplitude, &[0], 4).unwrap();

        let first_stage_planes = [0, 1, 2]
            .into_iter()
            .map(|component| layout.physical_component(component).unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            first_stage_planes.len(),
            3,
            "same-stage input and every output must remain disjoint"
        );
        assert_ne!(
            layout.physical_component(1).unwrap(),
            layout.physical_component(2).unwrap(),
            "outputs from separate leaves still share the fused-stage no-alias contract"
        );
    }

    #[test]
    fn stage_liveness_rejects_a_current_use_without_a_producer() {
        let stages = vec![liveness_stage(vec![liveness_leaf(&[1], &[2])])];
        let amplitude = liveness_stage(vec![liveness_leaf(&[2], &[])]);
        let error = derive_stage_current_layout(&stages, &amplitude, &[0], 3).unwrap_err();
        assert!(error.message().contains("used before its producer"));
    }

    fn test_payload_directory() -> PathBuf {
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target"));
        target.join(format!(
            "compiled-direct-engine-test-{}-{}",
            std::process::id(),
            TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn source_application(inputs: &[Expr], outputs: &[Expr]) -> Vec<u8> {
        let mut config = Config::default();
        config.set_opt_level(3);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        config.set_compress(true);
        let application = Compiler::with_config(config)
            .compile_params(&[], outputs, inputs)
            .unwrap();
        assert_eq!(application.count_states, 0);
        assert_eq!(application.count_params, inputs.len() * 2);
        assert_eq!(application.count_obs, outputs.len() * 2);
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn evaluator(
        path: &str,
        plane_path: &str,
        plane_program: &str,
        input_len: usize,
        output_len: usize,
    ) -> EvaluatorManifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability: SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string(),
            application_path: path.to_string(),
            application_abi: SYMJIT_APPLICATION_STORAGE_ABI.to_string(),
            input_len,
            output_len,
            element_layout: "complex-f64".to_string(),
            batch_layout: "row-major".to_string(),
            compiler_type: "native".to_string(),
            translation_mode: "indirect".to_string(),
            optimization_level: 3,
            word_bits: 64,
            endianness: "little".to_string(),
            required_defuns: Vec::new(),
            plane_application: Some(SymjitPlaneApplicationManifest {
                application_path: plane_path.to_string(),
                application_abi: SYMJIT_PLANE_APPLICATION_ABI.to_string(),
                storage_abi: SYMJIT_APPLICATION_STORAGE_ABI.to_string(),
                element_layout: "split-complex-plane-major".to_string(),
                descriptor_order: "inputs-re-im-then-outputs-re-im".to_string(),
                input_complex_count: input_len,
                output_complex_count: output_len,
                input_plane_count: input_len * 2,
                output_plane_count: output_len * 2,
                compiler_type: "native".to_string(),
                translation_mode: "symbolica-structured-instructions".to_string(),
                optimization_level: 3,
                simd: true,
                complex: true,
                fast_math: true,
                fast_complex: false,
                compression: true,
                threading: false,
                direct_arena: true,
                source_digest: format!("{:x}", Sha256::digest(plane_program.as_bytes())),
                target: serde_json::json!({
                    "word_bits": 64,
                    "endianness": "little",
                    "triple": "test-native",
                    "cpu_features": [],
                }),
            }),
            evaluator_state_path: None,
            evaluator_state_runtime_capability: None,
        }
    }

    fn native_direct_fixture() -> (PathBuf, String) {
        let directory = test_payload_directory();
        fs::create_dir_all(&directory).unwrap();
        let source = directory.join("native_direct_leaf.cpp");
        let library_name = if cfg!(target_os = "macos") {
            "libnative_direct_leaf.dylib"
        } else {
            "libnative_direct_leaf.so"
        };
        let library = directory.join(library_name);
        fs::write(
            &source,
            r#"#include <cstdint>
struct Metadata {
  std::uint32_t abi_version;
  std::uint32_t struct_size;
  std::uint32_t flags;
  std::uint32_t input_plane_count;
  std::uint32_t scalar_input_count;
  std::uint32_t output_plane_count;
  std::uint32_t simd_lane_width;
  std::uint32_t reserved;
};
struct InputPlane { const double* values; };
struct OutputPlane { double* values; };
struct Scalar { const double* value; };
extern "C" Metadata native_direct_leaf_direct_application_v1_metadata() noexcept {
  return Metadata{1u, sizeof(Metadata), 63u, 2u, 0u, 2u, 2u, 0u};
}
extern "C" int native_direct_leaf_direct_application_v1(
    const InputPlane* inputs, std::uint32_t input_count,
    const Scalar*, std::uint32_t scalar_count,
    const OutputPlane* outputs, std::uint32_t output_count,
    std::uint32_t point_start, std::uint32_t point_count) noexcept {
  if (inputs == nullptr || outputs == nullptr || input_count != 2u ||
      scalar_count != 0u || output_count != 2u || point_count == 0u) return 2;
  for (std::uint32_t offset = 0; offset < point_count; ++offset) {
    const std::uint32_t point = point_start + offset;
    outputs[0].values[point] = 2.0 * inputs[0].values[point];
    outputs[1].values[point] = 2.0 * inputs[1].values[point];
  }
  return 0;
}
"#,
        )
        .unwrap();
        let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
        let mut command = Command::new(compiler);
        command.args(["-std=c++17", "-O3"]);
        if cfg!(target_os = "macos") {
            command.arg("-dynamiclib");
        } else {
            command.args(["-shared", "-fPIC"]);
        }
        let output = command
            .arg(&source)
            .arg("-o")
            .arg(&library)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "could not compile native plane fixture: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        (directory, library_name.to_string())
    }

    fn native_evaluator(library_path: &str, runtime_capability: &str) -> EvaluatorManifest {
        EvaluatorManifest::CompiledComplex {
            runtime_capability: runtime_capability.to_string(),
            function_name: "native_direct_leaf".to_string(),
            input_len: 1,
            output_len: 1,
            library_path: "absent-dense-library".to_string(),
            evaluator_state_path: None,
            number_type: "complex".to_string(),
            native_direct_application: Some(NativeCompiledDirectApplicationManifest {
                application_abi:
                    super::super::evaluator::native_compiled_direct::
                        NATIVE_COMPILED_DIRECT_APPLICATION_ABI
                        .to_string(),
                function_name: "native_direct_leaf".to_string(),
                source_path: "native_direct_leaf.cpp".to_string(),
                library_path: library_path.to_string(),
                target: NativeCompiledDirectTargetManifest {
                    triple: crate::runtime_target_info().triple,
                    cpu_features: Vec::new(),
                },
                evaluator_state_sha256: "a".repeat(64),
                instruction_count: 1,
                temporary_count: 0,
                input_plane_count: 2,
                scalar_input_count: 0,
                output_plane_count: 2,
                simd_lane_width: 2,
                logical_stack_bytes: 32,
                output_semantics: "factor-free-overwrite".to_string(),
            }),
        }
    }

    fn input_component(
        kind: &str,
        source_id: usize,
        component: usize,
        global_component: usize,
        parameter_index: usize,
        real_valued: bool,
    ) -> GenericStageInputComponentManifest {
        GenericStageInputComponentManifest {
            kind: kind.to_string(),
            source_id,
            component,
            global_component,
            parameter_index,
            real_valued,
        }
    }

    fn output_slot(
        component: usize,
        current_id: isize,
        output_column: usize,
    ) -> GenericStageOutputSlotManifest {
        GenericStageOutputSlotManifest {
            value_slot_id: current_id,
            current_id,
            variant: "test".to_string(),
            component_start: component,
            component_stop: component + 1,
            output_start: output_column,
            output_stop: output_column + 1,
            color_selector_domain_ids: Vec::new(),
        }
    }

    fn stage_manifest(
        label: &str,
        inputs: Vec<GenericStageInputComponentManifest>,
        output_components: &[usize],
        evaluator: EvaluatorManifest,
        amplitude: bool,
    ) -> GenericSerializedStageEvaluatorManifest {
        let leaf_layout = evaluator.leaf_layout().unwrap();
        let mut direct_application_abi = None;
        let mut direct_source_abi = None;
        let direct_leaves = leaf_layout
            .iter()
            .map(|leaf| {
                let (application_path, application_abi, optimization_level, input_len, output_len) =
                    match leaf.evaluator {
                        EvaluatorManifest::SymjitApplication {
                            plane_application: Some(plane_application),
                            optimization_level,
                            input_len,
                            output_len,
                            ..
                        } => (
                            &plane_application.application_path,
                            &plane_application.application_abi,
                            *optimization_level,
                            *input_len,
                            *output_len,
                        ),
                        EvaluatorManifest::CompiledComplex {
                            input_len,
                            output_len,
                            native_direct_application: Some(application),
                            ..
                        } => (
                            &application.library_path,
                            &application.application_abi,
                            3,
                            *input_len,
                            *output_len,
                        ),
                        _ => panic!("compiled Direct-Arena test leaf must be direct-capable"),
                    };
                if let Some(expected) = direct_source_abi.as_ref() {
                    assert_eq!(expected, application_abi);
                } else {
                    direct_source_abi = Some(application_abi.clone());
                };
                if direct_application_abi.is_none() {
                    direct_application_abi =
                        Some(if application_abi == SYMJIT_PLANE_APPLICATION_ABI {
                            COMPILED_PLANE_DIRECT_APPLICATION_ABI.to_string()
                        } else {
                            application_abi.clone()
                        });
                }
                CompiledPlaneLeafManifest {
                    application_path: application_path.clone(),
                    source_application_abi: application_abi.clone(),
                    optimization_level,
                    direct_codegen_optimization_level: optimization_level,
                    input_len,
                    output_len,
                    input_indices: leaf.input_indices.clone(),
                    output_start: leaf.output_range.start,
                    output_stop: leaf.output_range.end,
                }
            })
            .collect();
        let direct_inputs = inputs
            .iter()
            .map(|input| CompiledPlaneInputBindingManifest {
                parameter_index: input.parameter_index,
                kind: input.kind.clone(),
                source_id: input.source_id,
                component: input.component,
                global_component: input.global_component,
                real_valued: input.real_valued,
            })
            .collect();
        let direct_outputs = output_components
            .iter()
            .copied()
            .enumerate()
            .map(
                |(output_index, component)| CompiledPlaneOutputBindingManifest {
                    output_index,
                    arena: if amplitude { "amplitude" } else { "current" }.to_string(),
                    component,
                },
            )
            .collect();
        let compiled_plane_arena = Some(CompiledPlaneArenaStageManifest {
            schema_version: 1,
            kind: "compiled-plane-arena-stage".to_string(),
            application_abi: direct_application_abi.unwrap(),
            source_application_abi: direct_source_abi.unwrap(),
            element_layout: "split-complex-component-major".to_string(),
            output_operation: "overwrite".to_string(),
            output_factor: "identity".to_string(),
            input_output_aliasing: "forbidden".to_string(),
            output_output_aliasing: "forbidden".to_string(),
            input_bindings: direct_inputs,
            output_bindings: direct_outputs,
            leaves: direct_leaves,
        });
        let value_parameter_count = inputs.iter().filter(|item| item.kind == "value").count();
        let momentum_parameter_count = inputs.iter().filter(|item| item.kind == "momentum").count();
        let model_parameter_count = inputs
            .iter()
            .filter(|item| item.kind == "model_parameter")
            .count();
        let real_valued_inputs = inputs
            .iter()
            .filter_map(|item| item.real_valued.then_some(item.parameter_index))
            .collect();
        GenericSerializedStageEvaluatorManifest {
            stage_index: usize::from(amplitude),
            stage_kind: if amplitude {
                "amplitude".to_string()
            } else {
                "interaction".to_string()
            },
            subset_size: None,
            evaluator_label: label.to_string(),
            parameter_layout: "stage-local-value-momentum".to_string(),
            output_length: output_components.len(),
            output_slots: output_components
                .iter()
                .copied()
                .enumerate()
                .map(|(column, component)| {
                    output_slot(
                        component,
                        if amplitude { -1 } else { component as isize },
                        column,
                    )
                })
                .collect(),
            input_value_slot_ids: Vec::new(),
            output_value_slot_ids: Vec::new(),
            interaction_ids: Vec::new(),
            parameter_count: inputs.len(),
            input_components: inputs,
            value_parameter_count,
            momentum_parameter_count,
            model_parameter_count,
            real_valued_inputs,
            expression_ready: true,
            blockers: Vec::new(),
            evaluator,
            compiled_plane_arena,
        }
    }

    fn assert_close(actual: Complex<f64>, expected: Complex<f64>) {
        let expected_norm = expected.re.hypot(expected.im);
        let delta = actual - expected;
        let delta_norm = delta.re.hypot(delta.im);
        let tolerance = 1.0e-15 + 1.0e-12 * expected_norm;
        assert!(
            delta_norm <= tolerance,
            "actual={actual:?} expected={expected:?} tolerance={tolerance}"
        );
    }

    fn assert_close_real(actual: f64, expected: f64, context: &str) {
        let tolerance = 1.0e-15 + 1.0e-12 * actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: actual={actual:.17e} expected={expected:.17e} tolerance={tolerance:.17e}"
        );
    }

    #[test]
    fn cpp_and_asm_native_leaves_execute_only_through_canonical_planes() {
        let (payload_root, library_name) = native_direct_fixture();
        let payloads = EvaluatorPayloadStore::directory(&payload_root);
        for runtime_capability in [
            SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
            SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
        ] {
            let amplitude = stage_manifest(
                runtime_capability,
                vec![input_component("value", 0, 0, 0, 0, false)],
                &[0],
                native_evaluator(&library_name, runtime_capability),
                true,
            );
            let mut direct = CompiledDirectEnginePrototype::load(
                &[],
                &amplitude,
                &payloads,
                &[0],
                1,
                1,
                4,
                0,
                1,
                129,
            )
            .unwrap();
            let input = (0..129)
                .map(|point| Complex::new(point as f64 + 0.25, 0.5 - point as f64))
                .collect::<Vec<_>>();
            let state = input
                .iter()
                .flat_map(|value| {
                    [
                        *value,
                        Complex::new(0.0, 0.0),
                        Complex::new(0.0, 0.0),
                        Complex::new(0.0, 0.0),
                        Complex::new(0.0, 0.0),
                    ]
                })
                .collect::<Vec<_>>();
            direct.begin_tile_from_state(129, 5, &state).unwrap();
            direct.evaluate_all(129).unwrap();
            let mut output = vec![Complex::new(0.0, 0.0); 129];
            direct
                .extract_amplitudes_row_major(129, &mut output)
                .unwrap();
            for (actual, input) in output.into_iter().zip(&input) {
                assert_eq!(actual, *input * 2.0);
            }
            assert_eq!(direct.traffic().leaf.calls, 1);
            direct.traffic().leaf.validate_direct().unwrap();
        }
        fs::remove_dir_all(payload_root).unwrap();
    }

    fn disable_compiled_direct_recursive(runtime: &mut ExecutionRuntime) {
        runtime.compiled_direct_runtime = None;
        runtime.compiled_direct_color_schedules.clear();
        runtime.compiled_direct_helicity_schedules.clear();
        if let Some(sum_runtime) = runtime.helicity_sum_runtime.as_deref_mut() {
            disable_compiled_direct_recursive(sum_runtime);
        }
        for selector_runtime in &mut runtime.helicity_selector_runtimes {
            disable_compiled_direct_recursive(selector_runtime);
        }
        for selector_runtime in runtime.color_selector_runtimes.values_mut() {
            disable_compiled_direct_recursive(selector_runtime);
        }
    }

    fn compiled_direct_runtime_summary(runtime: &ExecutionRuntime) -> (usize, u64, u64, usize) {
        let (mut engine_count, mut requested_bytes, mut hot_calls, mut minimum_tile_capacity) =
            (0usize, 0u64, 0u64, usize::MAX);
        if let Some(direct) = runtime.compiled_direct_runtime.as_ref() {
            engine_count += 1;
            requested_bytes =
                requested_bytes.saturating_add(direct.allocation_counters().requested_bytes);
            hot_calls = hot_calls.saturating_add(direct.traffic().leaf.calls);
            minimum_tile_capacity = minimum_tile_capacity.min(direct.tile_capacity());
            direct
                .traffic()
                .leaf
                .validate_direct()
                .expect("production compiled leaf traffic remains direct");
            assert_eq!(direct.traffic().boundary_input_bytes, 0);
            assert_eq!(direct.traffic().boundary_current_output_bytes, 0);
            assert_eq!(direct.traffic().boundary_amplitude_output_bytes, 0);
        }
        let children = runtime
            .helicity_sum_runtime
            .iter()
            .map(Box::as_ref)
            .chain(runtime.helicity_selector_runtimes.iter().map(Box::as_ref))
            .chain(runtime.color_selector_runtimes.values().map(Box::as_ref));
        for child in children {
            let (child_engines, child_bytes, child_calls, child_tile) =
                compiled_direct_runtime_summary(child);
            engine_count += child_engines;
            requested_bytes = requested_bytes.saturating_add(child_bytes);
            hot_calls = hot_calls.saturating_add(child_calls);
            minimum_tile_capacity = minimum_tile_capacity.min(child_tile);
        }
        (
            engine_count,
            requested_bytes,
            hot_calls,
            minimum_tile_capacity,
        )
    }

    fn assert_compiled_direct_tile_geometry(runtime: &ExecutionRuntime) {
        if let Some(direct) = runtime.compiled_direct_runtime.as_ref() {
            let sizing = direct.tile_sizing();
            let stride = direct.arena.point_stride() as usize;
            let logical_tile_bytes = stride
                .checked_mul(sizing.physical_scalar_values_per_point)
                .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
                .expect("retained Direct logical tile bytes fit usize");
            assert!(logical_tile_bytes > 0);
            if sizing.reduction_scalar_values_per_point == 0 {
                assert_eq!(
                    sizing.effective_tile_capacity, 1,
                    "unauthenticated retained reducer did not fail closed"
                );
            }
        }
        let children = runtime
            .helicity_sum_runtime
            .iter()
            .map(Box::as_ref)
            .chain(runtime.helicity_selector_runtimes.iter().map(Box::as_ref))
            .chain(runtime.color_selector_runtimes.values().map(Box::as_ref));
        for child in children {
            assert_compiled_direct_tile_geometry(child);
        }
    }

    fn retained_validation_point(runtime: &NativeRuntime) -> Vec<f64> {
        let path = runtime
            .root()
            .join("processes")
            .join(&runtime.metadata().representative_process_key)
            .join("validation-momenta.json");
        let validation: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("read retained validation momenta"))
                .expect("parse retained validation momenta");
        validation["points"][0]
            .as_array()
            .expect("one retained validation point")
            .iter()
            .flat_map(|leg| {
                leg["momentum"]
                    .as_array()
                    .expect("four retained momentum components")
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .expect("decimal retained momentum string")
                            .parse::<f64>()
                            .expect("retained f64 momentum")
                    })
            })
            .collect()
    }

    #[test]
    fn compiled_direct_reducer_shapes_are_statically_authenticated() {
        let routed = CompiledDirectRoutedReductionFootprint::default();
        let hot =
            |kind, maximum_group_component_count, maximum_resolved_component_count, routed| {
                CompiledDirectReductionFootprint::authenticated(
                    kind,
                    maximum_group_component_count,
                    maximum_resolved_component_count,
                    routed,
                )
                .unwrap()
                .hot_scalar_values_per_point()
                .unwrap()
            };
        assert_eq!(hot(CompiledDirectReducerKind::Plain, 1, 0, routed), 3);
        assert_eq!(hot(CompiledDirectReducerKind::Coherent, 5, 7, routed), 20);
        assert_eq!(
            hot(
                CompiledDirectReducerKind::Contracted { group_count: 11 },
                5,
                7,
                routed,
            ),
            40
        );
        assert_eq!(
            hot(
                CompiledDirectReducerKind::ColorTopologyReplay {
                    physical_group_count: 13,
                },
                5,
                7,
                routed,
            ),
            44
        );
        assert_eq!(
            hot(
                CompiledDirectReducerKind::Coherent,
                5,
                7,
                CompiledDirectRoutedReductionFootprint {
                    maximum_source_component_count: 512,
                    maximum_target_component_count: 1,
                },
            ),
            526,
            "authenticated replay source components can dominate the reducer"
        );
        assert_eq!(
            hot(
                CompiledDirectReducerKind::Coherent,
                5,
                7,
                CompiledDirectRoutedReductionFootprint {
                    maximum_source_component_count: 1,
                    maximum_target_component_count: 1_024,
                },
            ),
            1_038,
            "authenticated replay target components can dominate the reducer"
        );
    }

    #[test]
    fn selected_routed_reduction_tile_uses_actual_components_within_allocated_capacity() {
        let mut leaf = liveness_leaf(&[0], &[1]);
        leaf.scalar_values_per_point = 4_320;
        let stages = vec![liveness_stage(vec![leaf])];
        let amplitude = liveness_stage(vec![liveness_leaf(&[1], &[])]);
        let layout = CompiledDirectCurrentLayout::identity(100_000).unwrap();
        let sizing = compiled_direct_tile_sizing(
            &stages,
            &amplitude,
            &layout,
            &[0, 1, 2, 3],
            24,
            32,
            CompiledDirectReductionFootprint::Authenticated {
                hot_scalar_values_per_point: 188_167,
            },
            CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
        )
        .unwrap();
        assert_eq!(sizing.effective_tile_capacity, 32);
        assert_eq!(sizing.reduction_tile_capacity, 2);
        let selected_footprint = |source, target| {
            CompiledDirectReductionFootprint::authenticated(
                CompiledDirectReducerKind::Coherent,
                1,
                0,
                CompiledDirectRoutedReductionFootprint {
                    maximum_source_component_count: source,
                    maximum_target_component_count: target,
                },
            )
            .unwrap()
        };
        assert_eq!(
            sizing
                .reduction_tile_capacity_for_footprint(selected_footprint(64, 512))
                .unwrap(),
            32,
            "one selected flow must not inherit the full color-axis working set"
        );
        assert_eq!(
            sizing
                .reduction_tile_capacity_for_footprint(selected_footprint(64, 100_000))
                .unwrap(),
            4,
            "a genuinely large selected target still limits cache locality"
        );
        assert_eq!(
            sizing
                .reduction_tile_capacity_for_footprint(selected_footprint(100_000, 64))
                .unwrap(),
            4,
            "selected source scratch is included as well as target scratch"
        );
        let smaller_allocation = CompiledDirectTileSizing {
            effective_tile_capacity: 8,
            ..sizing
        };
        assert_eq!(
            smaller_allocation
                .reduction_tile_capacity_for_footprint(selected_footprint(1, 1))
                .unwrap(),
            8,
            "selected reduction must not enlarge the already allocated arena"
        );
        assert_eq!(
            sizing
                .reduction_tile_capacity_for_footprint(
                    CompiledDirectReductionFootprint::Unauthenticated,
                )
                .unwrap(),
            sizing.reduction_tile_capacity,
            "unknown reducer shapes retain the existing conservative capacity"
        );
    }

    #[test]
    fn compiled_direct_tile_policy_uses_authenticated_phase_local_footprints() {
        let mut leaf = liveness_leaf(&[0], &[1]);
        leaf.scalar_values_per_point = 4_320;
        leaf.persistent_scalar_values_per_point = 17;
        let stages = vec![liveness_stage(vec![leaf])];
        let amplitude = liveness_stage(vec![liveness_leaf(&[1], &[])]);
        let layout = CompiledDirectCurrentLayout::identity(100_000).unwrap();
        let reduction = CompiledDirectReductionFootprint::Authenticated {
            hot_scalar_values_per_point: 4_096,
        };
        let expected = compiled_direct_tile_sizing(
            &stages,
            &amplitude,
            &layout,
            &[0, 1, 2, 3],
            24,
            8_192,
            reduction,
            CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
        )
        .unwrap();
        assert_eq!(expected.effective_tile_capacity, 32);
        assert_eq!(expected.hot_scalar_values_per_point, 4_320);
        assert_eq!(expected.source_scalar_values_per_point, 33);
        assert_eq!(expected.reduction_scalar_values_per_point, 4_096);
        assert_eq!(expected.persistent_scalar_values_per_point, 17);
        assert_eq!(
            expected.physical_scalar_values_per_point,
            200_000 + 24 + 16_384 + 1 + 17,
            "the hard footprint includes every persistent point-pitched scalar plane"
        );
        let expected_stride =
            checked_aligned_point_stride(expected.effective_tile_capacity as u32).unwrap();
        let expected_direct_bytes = expected_stride as usize
            * expected.physical_scalar_values_per_point
            * std::mem::size_of::<f64>();
        assert!(
            expected_direct_bytes > COMPILED_DIRECT_CACHE_TARGET_BYTES,
            "the soft cache target must not reject a larger physical arena"
        );
        for _ in 0..32 {
            assert_eq!(
                compiled_direct_tile_sizing(
                    &stages,
                    &amplitude,
                    &layout,
                    &[0, 1, 2, 3],
                    24,
                    8_192,
                    reduction,
                    CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
                )
                .unwrap(),
                expected,
                "authenticated sizing must be deterministic"
            );
        }

        let many_sources = (0..20_000).collect::<Vec<_>>();
        let source_limited = compiled_direct_tile_sizing(
            &stages,
            &amplitude,
            &layout,
            &many_sources,
            24,
            1,
            CompiledDirectReductionFootprint::authenticated(
                CompiledDirectReducerKind::Plain,
                1,
                0,
                CompiledDirectRoutedReductionFootprint::default(),
            )
            .unwrap(),
            CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
        )
        .unwrap();
        assert_eq!(source_limited.source_scalar_values_per_point, 40_025);
        assert_eq!(source_limited.hot_scalar_values_per_point, 40_025);
        assert_eq!(
            source_limited.effective_tile_capacity, 8,
            "authenticated source preparation can be the cache-limiting phase"
        );

        let large_reduction = CompiledDirectReductionFootprint::Authenticated {
            hot_scalar_values_per_point: 32_768,
        };
        let reduction_limited = compiled_direct_tile_sizing(
            &stages,
            &amplitude,
            &layout,
            &[0],
            4,
            1,
            large_reduction,
            CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
        )
        .unwrap();
        assert_eq!(
            reduction_limited.effective_tile_capacity, 32,
            "a cold resolved-reduction shape must not fragment total evaluation"
        );
        assert_eq!(
            reduction_limited.reduction_tile_capacity, 16,
            "resolved and routed reductions retain their smaller cache-local tile"
        );

        let hard_limited_layout = CompiledDirectCurrentLayout::identity(1_000_000).unwrap();
        assert_eq!(
            compiled_direct_tile_sizing(
                &stages,
                &amplitude,
                &hard_limited_layout,
                &[0],
                4,
                1,
                CompiledDirectReductionFootprint::authenticated(
                    CompiledDirectReducerKind::Plain,
                    1,
                    0,
                    CompiledDirectRoutedReductionFootprint::default(),
                )
                .unwrap(),
                CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
            )
            .unwrap()
            .effective_tile_capacity,
            32,
            "large physical arenas retain the locality-limited tile"
        );
        let hard_limited = compiled_direct_tile_sizing(
            &stages,
            &amplitude,
            &hard_limited_layout,
            &[0],
            4,
            1,
            CompiledDirectReductionFootprint::authenticated(
                CompiledDirectReducerKind::Plain,
                1,
                0,
                CompiledDirectRoutedReductionFootprint::default(),
            )
            .unwrap(),
            CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
        )
        .unwrap();
        let hard_stride =
            checked_aligned_point_stride(hard_limited.effective_tile_capacity as u32).unwrap();
        let hard_direct_bytes = hard_stride as usize
            * hard_limited.physical_scalar_values_per_point
            * std::mem::size_of::<f64>();
        assert!(
            hard_direct_bytes > 256 * 1024 * 1024,
            "the regression geometry must exceed the removed fixed workspace bound"
        );
        assert_eq!(
            compiled_direct_tile_sizing(
                &stages,
                &amplitude,
                &layout,
                &[0],
                4,
                1,
                CompiledDirectReductionFootprint::Unauthenticated,
                CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
            )
            .unwrap()
            .effective_tile_capacity,
            1,
            "production adaptive tiling fails closed without a static reducer footprint"
        );
        let unauthenticated_over_budget_layout = CompiledDirectCurrentLayout {
            canonical_to_physical: vec![0].into_boxed_slice(),
            physical_component_count: 2_100_000,
            live_component_count: 1,
        };
        assert_eq!(
            compiled_direct_tile_sizing(
                &stages,
                &amplitude,
                &unauthenticated_over_budget_layout,
                &[0],
                4,
                1,
                CompiledDirectReductionFootprint::Unauthenticated,
                CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
            )
            .unwrap()
            .effective_tile_capacity,
            1,
            "the unauthenticated fallback remains one point without a fixed byte ceiling"
        );
        assert_eq!(
            compiled_direct_tile_sizing(
                &stages,
                &amplitude,
                &hard_limited_layout,
                &[0],
                4,
                1,
                CompiledDirectReductionFootprint::default(),
                CompiledDirectCurrentLayoutPolicy::Identity { tile_capacity: 129 },
            )
            .unwrap()
            .effective_tile_capacity,
            129,
            "explicit developer/test identity capacity remains exact"
        );
    }

    #[test]
    fn manifest_driven_engine_matches_row_major_compiled_stages_at_all_tail_boundaries() {
        const VALUE_COMPONENTS: usize = 4;
        const MOMENTUM_COMPONENTS: usize = 4;
        const MODEL_PARAMETERS: usize = 1;
        const GLOBAL_PARAMETERS: usize = VALUE_COMPONENTS + MOMENTUM_COMPONENTS + MODEL_PARAMETERS;
        const PRODUCT_PROGRAM: &str =
            "([('mul', ('out', 0), [('param', 0), ('param', 1)], 0)], 2, [])";
        const SHIFTED_PROGRAM: &str =
            "([('add', ('out', 0), [('param', 0), ('param', 1)], 0)], 2, [])";

        let payload_root = test_payload_directory();
        fs::create_dir_all(&payload_root).unwrap();
        let x = Expr::var("x");
        let energy = Expr::var("energy");
        let coupling = Expr::var("coupling");
        let product = &x * &energy;
        let shifted = &x + &coupling;
        let product_bytes =
            source_application(&[x.clone(), energy.clone()], std::slice::from_ref(&product));
        let shifted_bytes = source_application(
            &[x.clone(), coupling.clone()],
            std::slice::from_ref(&shifted),
        );
        fs::write(payload_root.join("product.symjit"), &product_bytes).unwrap();
        fs::write(payload_root.join("shifted.symjit"), &shifted_bytes).unwrap();
        let left = Expr::var("left");
        let right = Expr::var("right");
        let amplitude_value = &left * &right;
        let amplitude_bytes =
            source_application(&[left, right], std::slice::from_ref(&amplitude_value));
        fs::write(payload_root.join("amplitude.symjit"), &amplitude_bytes).unwrap();
        for (path, program) in [
            ("product.plane.symjit", PRODUCT_PROGRAM),
            ("shifted.plane.symjit", SHIFTED_PROGRAM),
            ("amplitude.plane.symjit", PRODUCT_PROGRAM),
        ] {
            fs::write(
                payload_root.join(path),
                compile_symbolica_program_to_plane_application_bytes(program, 2, 1, 3, true)
                    .unwrap(),
            )
            .unwrap();
        }

        let chunked = EvaluatorManifest::Chunked {
            required_runtime_capabilities: vec![SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string()],
            input_len: Some(3),
            chunk_input_indices: Some(vec![vec![0, 1], vec![0, 2]]),
            chunks: vec![
                evaluator(
                    "product.symjit",
                    "product.plane.symjit",
                    PRODUCT_PROGRAM,
                    2,
                    1,
                ),
                evaluator(
                    "shifted.symjit",
                    "shifted.plane.symjit",
                    SHIFTED_PROGRAM,
                    2,
                    1,
                ),
            ],
        };
        let stage = stage_manifest(
            "real-fused-stage",
            vec![
                input_component("value", 0, 0, 0, 0, false),
                input_component("momentum", 0, 0, VALUE_COMPONENTS, 1, true),
                input_component(
                    "model_parameter",
                    0,
                    0,
                    VALUE_COMPONENTS + MOMENTUM_COMPONENTS,
                    2,
                    true,
                ),
            ],
            &[1, 2],
            chunked,
            false,
        );
        let amplitude = stage_manifest(
            "real-amplitude-stage",
            vec![
                input_component("value", 1, 0, 1, 0, false),
                input_component("value", 2, 0, 2, 1, false),
            ],
            &[0],
            evaluator(
                "amplitude.symjit",
                "amplitude.plane.symjit",
                PRODUCT_PROGRAM,
                2,
                1,
            ),
            true,
        );
        let legacy_amplitude = stage_manifest(
            "legacy-amplitude-stage",
            vec![
                input_component("value", 1, 0, 1, 0, false),
                input_component("value", 2, 0, 2, 1, false),
            ],
            &[3],
            evaluator(
                "amplitude.symjit",
                "amplitude.plane.symjit",
                PRODUCT_PROGRAM,
                2,
                1,
            ),
            false,
        );
        let payloads = EvaluatorPayloadStore::directory(&payload_root);
        let mut legacy_stage = StageRuntime::load(&stage, &payloads).unwrap();
        let mut legacy_amplitude_stage = StageRuntime::load(&legacy_amplitude, &payloads).unwrap();
        let mut direct = CompiledDirectEnginePrototype::load(
            std::slice::from_ref(&stage),
            &amplitude,
            &payloads,
            &[0],
            1,
            VALUE_COMPONENTS,
            MOMENTUM_COMPONENTS,
            MODEL_PARAMETERS,
            1,
            129,
        )
        .unwrap();
        let existing_selector = CompiledColorSelectorSchedule {
            active_stage_chunk_indices: vec![vec![0, 1]],
            active_amplitude_chunk_indices: vec![0],
        };
        let direct_selector = direct.bind_color_schedule(&existing_selector).unwrap();
        let color_schedules =
            BTreeMap::from([(7, direct.bind_color_schedule(&existing_selector).unwrap())]);
        let selected_colors = BTreeSet::from([7]);
        let joint = direct
            .bind_helicity_color_schedule(11, &direct_selector, &selected_colors, &color_schedules)
            .unwrap();
        let cached = direct
            .bind_helicity_color_schedule(11, &direct_selector, &selected_colors, &color_schedules)
            .unwrap();
        assert!(Arc::ptr_eq(&joint, &cached));
        assert_eq!(
            joint.active_stage_leaves,
            direct_selector.active_stage_leaves
        );
        assert_eq!(
            joint.active_amplitude_leaves,
            direct_selector.active_amplitude_leaves
        );
        assert!(
            direct
                .bind_helicity_color_schedule(
                    11,
                    &direct_selector,
                    &BTreeSet::new(),
                    &color_schedules,
                )
                .is_err()
        );
        assert!(
            direct
                .bind_helicity_color_schedule(
                    11,
                    &direct_selector,
                    &BTreeSet::from([8]),
                    &color_schedules,
                )
                .is_err()
        );
        let initial_state = |point_count: usize| {
            let mut initial = vec![Complex::new(0.0, 0.0); point_count * GLOBAL_PARAMETERS];
            for point in 0..point_count {
                let row = point * GLOBAL_PARAMETERS;
                initial[row] =
                    Complex::new(0.25 + point as f64 / 128.0, -0.5 + point as f64 / 512.0);
                initial[row + VALUE_COMPONENTS] = Complex::new(20.0 + point as f64 / 64.0, 0.0);
                initial[row + VALUE_COMPONENTS + MOMENTUM_COMPONENTS] = Complex::new(1.25, 0.0);
            }
            initial
        };

        for point_count in [7_usize, 127, 128, 129] {
            let initial = initial_state(point_count);

            let mut legacy = initial.clone();
            legacy_stage
                .evaluate_f64_into_state(point_count, GLOBAL_PARAMETERS, &mut legacy)
                .unwrap();
            legacy_amplitude_stage
                .evaluate_f64_into_state(point_count, GLOBAL_PARAMETERS, &mut legacy)
                .unwrap();

            direct
                .begin_tile_from_state(point_count, GLOBAL_PARAMETERS, &initial)
                .unwrap();
            direct
                .evaluate_validated(point_count, &direct_selector)
                .unwrap();
            let (allocation_result, allocations, allocated_bytes) =
                count_test_allocations(|| direct.evaluate_validated(point_count, &direct_selector));
            allocation_result.unwrap();
            assert_eq!(allocations, 0, "warmed direct engine call allocated");
            assert_eq!(
                allocated_bytes, 0,
                "warmed direct engine call allocated bytes"
            );
            let traffic_before_extract = direct.traffic();
            assert!(traffic_before_extract.boundary_input_bytes > 0);
            assert!(traffic_before_extract.leaf.calls > 0);
            assert!(traffic_before_extract.leaf.points > 0);
            traffic_before_extract.leaf.validate_direct().unwrap();
            let planes = direct.amplitude_planes().unwrap();
            assert_eq!(planes.component_count().unwrap(), 1);
            assert_eq!(planes.point_count(), point_count as u32);

            let mut direct_state = initial.clone();
            direct
                .copy_current_to_state(point_count, GLOBAL_PARAMETERS, &mut direct_state)
                .unwrap();
            let mut direct_amplitude = vec![Complex::new(f64::NAN, f64::NAN); point_count];
            direct
                .extract_amplitudes_row_major(point_count, &mut direct_amplitude)
                .unwrap();
            let traffic_after_extract = direct.traffic();
            assert_eq!(
                traffic_after_extract.boundary_current_output_bytes
                    - traffic_before_extract.boundary_current_output_bytes,
                (point_count * VALUE_COMPONENTS * 2 * std::mem::size_of::<f64>()) as u64
            );
            assert_eq!(
                traffic_after_extract.boundary_amplitude_output_bytes
                    - traffic_before_extract.boundary_amplitude_output_bytes,
                (point_count * 2 * std::mem::size_of::<f64>()) as u64
            );
            traffic_after_extract.leaf.validate_direct().unwrap();
            for (point, direct_value) in direct_amplitude.iter().enumerate() {
                let row = point * GLOBAL_PARAMETERS;
                assert_close(direct_state[row + 1], legacy[row + 1]);
                assert_close(direct_state[row + 2], legacy[row + 2]);
                assert_close(*direct_value, legacy[row + 3]);
            }

            // Correlated memo reuse must restore overwritten planes, and a
            // genuinely changed source must invalidate the dependent leaves.
            let mut memo = CorrelatedStageMemo::default();
            for repeat in 0..3 {
                let mut correlated_state = initial.clone();
                if repeat == 2 {
                    correlated_state[0] += Complex::new(0.125, -0.25);
                }
                direct
                    .begin_tile_from_state(point_count, GLOBAL_PARAMETERS, &correlated_state)
                    .unwrap();
                direct
                    .evaluate_correlated_reusing(point_count, &mut memo)
                    .unwrap();
                let mut correlated_amplitude = vec![Complex::new(0.0, 0.0); point_count];
                direct
                    .extract_amplitudes_row_major(point_count, &mut correlated_amplitude)
                    .unwrap();
                legacy_stage
                    .evaluate_f64_into_state(point_count, GLOBAL_PARAMETERS, &mut correlated_state)
                    .unwrap();
                legacy_amplitude_stage
                    .evaluate_f64_into_state(point_count, GLOBAL_PARAMETERS, &mut correlated_state)
                    .unwrap();
                for (point, value) in correlated_amplitude.iter().enumerate() {
                    assert_close(*value, correlated_state[point * GLOBAL_PARAMETERS + 3]);
                }
                if repeat == 1 {
                    assert_eq!(memo.reused_leaves, 3);
                }
                if repeat == 2 {
                    assert_eq!(
                        memo.reused_leaves, 3,
                        "changed current inputs must invalidate the leaf"
                    );
                }
            }

            if point_count == 129 {
                const SAMPLES: usize = 7;
                const REPEATS: usize = 2_000;
                let mut direct_ns = [0_u128; SAMPLES];
                let mut legacy_ns = [0_u128; SAMPLES];
                for sample in 0..SAMPLES {
                    let mut time_direct = || {
                        let start = Instant::now();
                        for _ in 0..REPEATS {
                            direct
                                .evaluate_validated(point_count, &direct_selector)
                                .unwrap();
                        }
                        start.elapsed().as_nanos() / REPEATS as u128
                    };
                    let mut time_legacy = || {
                        let start = Instant::now();
                        for _ in 0..REPEATS {
                            legacy_stage
                                .evaluate_f64_into_state(
                                    point_count,
                                    GLOBAL_PARAMETERS,
                                    &mut legacy,
                                )
                                .unwrap();
                            legacy_amplitude_stage
                                .evaluate_f64_into_state(
                                    point_count,
                                    GLOBAL_PARAMETERS,
                                    &mut legacy,
                                )
                                .unwrap();
                        }
                        start.elapsed().as_nanos() / REPEATS as u128
                    };
                    if sample.is_multiple_of(2) {
                        direct_ns[sample] = time_direct();
                        legacy_ns[sample] = time_legacy();
                    } else {
                        legacy_ns[sample] = time_legacy();
                        direct_ns[sample] = time_direct();
                    }
                    let direct_planes = direct.amplitude_planes().unwrap();
                    std::hint::black_box((&legacy, direct_planes));
                }
                direct_ns.sort_unstable();
                legacy_ns.sort_unstable();
                let direct_median = direct_ns[SAMPLES / 2];
                let legacy_median = legacy_ns[SAMPLES / 2];
                eprintln!(
                    "compiled-direct-engine-prototype points={point_count} samples={SAMPLES} \
                     repeats={REPEATS} direct_ns={direct_ns:?} legacy_ns={legacy_ns:?} \
                     direct_median_ns={direct_median} legacy_median_ns={legacy_median} \
                     speedup={:.6}",
                    legacy_median as f64 / direct_median as f64
                );
            }
        }

        let mut tiled_direct = CompiledDirectEnginePrototype::load_with_current_layout_policy(
            std::slice::from_ref(&stage),
            &amplitude,
            &payloads,
            &[0],
            1,
            VALUE_COMPONENTS,
            MOMENTUM_COMPONENTS,
            MODEL_PARAMETERS,
            1,
            CompiledDirectReductionFootprint::authenticated(
                CompiledDirectReducerKind::Plain,
                1,
                0,
                CompiledDirectRoutedReductionFootprint::default(),
            )
            .unwrap(),
            CompiledDirectCurrentLayoutPolicy::StageLivenessProduction,
        )
        .unwrap();
        let production_tile = tiled_direct.tile_capacity();
        assert_eq!(production_tile, 32);
        let tiled_selector = tiled_direct
            .bind_color_schedule(&existing_selector)
            .unwrap();
        for point_count in [
            1_usize,
            production_tile - 1,
            production_tile,
            production_tile + 1,
            production_tile * 2 + 1,
            production_tile * 4 + 1,
        ] {
            let initial = initial_state(point_count);
            let mut legacy = initial.clone();
            legacy_stage
                .evaluate_f64_into_state(point_count, GLOBAL_PARAMETERS, &mut legacy)
                .unwrap();
            legacy_amplitude_stage
                .evaluate_f64_into_state(point_count, GLOBAL_PARAMETERS, &mut legacy)
                .unwrap();
            let mut tiled_amplitudes = vec![Complex::new(f64::NAN, f64::NAN); point_count];
            let tiles = tiled_direct
                .arena
                .point_tiles(point_count as u32, tiled_direct.tile_capacity() as u32)
                .unwrap();
            for tile in tiles {
                let point_start = tile.point_start as usize;
                let tile_points = tile.point_count as usize;
                let point_stop = point_start + tile_points;
                tiled_direct
                    .begin_tile_from_state_sources_for_test(
                        tile_points,
                        GLOBAL_PARAMETERS,
                        &initial[point_start * GLOBAL_PARAMETERS..point_stop * GLOBAL_PARAMETERS],
                    )
                    .unwrap();
                tiled_direct
                    .evaluate_validated(tile_points, &tiled_selector)
                    .unwrap();
                tiled_direct
                    .extract_amplitudes_row_major(
                        tile_points,
                        &mut tiled_amplitudes[point_start..point_stop],
                    )
                    .unwrap();
            }
            for (point, actual) in tiled_amplitudes.iter().copied().enumerate() {
                assert_close(actual, legacy[point * GLOBAL_PARAMETERS + 3]);
            }
        }

        direct
            .begin_tile_from_state(
                7,
                GLOBAL_PARAMETERS,
                &vec![Complex::new(0.0, 0.0); 7 * GLOBAL_PARAMETERS],
            )
            .unwrap();
        direct
            .evaluate_validated(7, &direct_selector)
            .expect("runtime-validated selector schedule must execute");
        let duplicate = CompiledColorSelectorSchedule {
            active_stage_chunk_indices: vec![vec![0, 0]],
            active_amplitude_chunk_indices: vec![0],
        };
        assert!(direct.bind_color_schedule(&duplicate).is_err());
        let missing_prerequisite = CompiledColorSelectorSchedule {
            active_stage_chunk_indices: vec![Vec::new()],
            active_amplitude_chunk_indices: vec![0],
        };
        assert!(direct.bind_color_schedule(&missing_prerequisite).is_err());
    }

    #[test]
    fn retained_compiled_o3_artifact_sizing_is_independently_deterministic() {
        let Some(root) = std::env::var_os("RUSTICOL_COMPILED_DIRECT_ARTIFACT") else {
            return;
        };
        let root = PathBuf::from(root);
        let first =
            NativeRuntime::load(&root, None, None).expect("load first retained Direct runtime");
        let second =
            NativeRuntime::load(&root, None, None).expect("load second retained Direct runtime");
        let first_summary = compiled_direct_runtime_summary(&first.runtime);
        let second_summary = compiled_direct_runtime_summary(&second.runtime);
        assert_eq!(
            first_summary, second_summary,
            "independent loads derived different nested Direct configurations"
        );
        assert!(
            first_summary.0 > 0,
            "retained sizing fixture has no Direct engine"
        );
        assert_eq!(
            first.metadata_json().unwrap(),
            second.metadata_json().unwrap(),
            "independent loads reported different nested Direct provenance"
        );
        assert_compiled_direct_tile_geometry(&first.runtime);
        assert_compiled_direct_tile_geometry(&second.runtime);
        eprintln!(
            "retained-compiled-direct-sizing engines={} requested_bytes={} \
             minimum_tile={} metadata={}",
            first_summary.0,
            first_summary.1,
            first_summary.3,
            first.metadata_json().unwrap()
        );
    }

    #[test]
    fn retained_compiled_o3_artifact_dual_runs_canonical_leaf_schedule() {
        let Some(root) = std::env::var_os("RUSTICOL_COMPILED_DIRECT_ARTIFACT") else {
            return;
        };
        let artifact = VerifiedArtifact::open(PathBuf::from(root))
            .expect("open retained compiled Direct-Arena artifact");
        let selection = artifact
            .select_process(None)
            .expect("select retained process");
        let (loaded, evaluator_root) =
            load_verified_evaluator(&artifact, &selection).expect("load retained manifest");
        let LoadedExecutionManifest::Compiled(execution) = loaded else {
            panic!("retained Direct-Arena fixture is not a compiled execution");
        };
        let stages = execution
            .compiled
            .stage_evaluators
            .as_ref()
            .expect("retained compiled execution has generic stages");
        let payloads = artifact
            .evaluator_payload_store(&evaluator_root)
            .expect("open retained evaluator payloads");
        let layout = &execution.runtime_schema.parameter_layout;
        let value_component_count = layout.value_component_count;
        let momentum_component_count = layout.momentum_parameter_count;
        let model_parameter_count = layout.model_parameter_count;
        let amplitude_component_count = stages.amplitude_stage.output_length;

        let mut legacy_stages = stages
            .stages
            .iter()
            .map(|stage| StageRuntime::load(stage, &payloads))
            .collect::<RusticolResult<Vec<_>>>()
            .expect("load retained row-major stages");
        let mut legacy_amplitude = AmplitudeRuntime::load(
            &execution.runtime_schema.amplitude_stage,
            &stages.amplitude_stage,
            &payloads,
        )
        .expect("load retained row-major amplitude stage");
        let (source_components, source_scratch_len) = canonical_source_layout(
            &execution.runtime_schema.source_fill.sources,
            value_component_count,
        )
        .expect("derive retained canonical sources");
        let mut direct = CompiledDirectEnginePrototype::load(
            &stages.stages,
            &stages.amplitude_stage,
            &payloads,
            &source_components,
            source_scratch_len,
            value_component_count,
            momentum_component_count,
            model_parameter_count,
            amplitude_component_count,
            129,
        )
        .expect("lower retained compiled O3 stages");

        let color_plan = build_compiled_color_execution_plan(
            stages,
            &execution.runtime_schema,
            &BTreeSet::new(),
        )
        .expect("validate retained selector coverage")
        .expect("retained LC artifact has selector coverage");
        let (_, retained_schedule) = color_plan
            .schedules_by_materialized_sector
            .iter()
            .next()
            .expect("retained LC artifact has a materialized sector");
        let direct_schedule = direct
            .bind_color_schedule(retained_schedule)
            .expect("bind production-validated retained selector schedule");

        let parameter_count =
            value_component_count + momentum_component_count + model_parameter_count;
        for point_count in [7_usize, 127, 128, 129] {
            let mut initial = vec![Complex::new(0.0, 0.0); point_count * parameter_count];
            for point in 0..point_count {
                let row = point * parameter_count;
                for component in 0..value_component_count {
                    initial[row + component] = Complex::new(
                        0.25 + component as f64 / 97.0 + point as f64 / 4096.0,
                        -0.125 + component as f64 / 211.0 - point as f64 / 8192.0,
                    );
                }
                for component in 0..momentum_component_count {
                    initial[row + value_component_count + component] =
                        Complex::new(2.0 + component as f64 / 17.0 + point as f64 / 1024.0, 0.0);
                }
                for component in 0..model_parameter_count {
                    initial[row + value_component_count + momentum_component_count + component] =
                        Complex::new(0.75 + component as f64 / 31.0, 0.0);
                }
            }

            let mut legacy = initial.clone();
            for (stage_index, stage) in legacy_stages.iter_mut().enumerate() {
                stage
                    .evaluate_active_chunks_f64_into_state(
                        point_count,
                        parameter_count,
                        &mut legacy,
                        &retained_schedule.active_stage_chunk_indices[stage_index],
                    )
                    .expect("evaluate retained selected row-major stage");
            }
            legacy_amplitude
                .evaluate_active_chunks_f64_into_scratch(
                    point_count,
                    &legacy,
                    &retained_schedule.active_amplitude_chunk_indices,
                )
                .expect("evaluate retained selected row-major amplitude stage");

            direct
                .begin_tile_from_state(point_count, parameter_count, &initial)
                .expect("transpose retained tile");
            direct
                .evaluate_validated(point_count, &direct_schedule)
                .expect("evaluate retained Direct-Arena selector schedule");
            let (result, allocations, allocated_bytes) =
                count_test_allocations(|| direct.evaluate_validated(point_count, &direct_schedule));
            result.expect("repeat retained Direct-Arena selector schedule");
            assert_eq!(allocations, 0, "retained warmed direct call allocated");
            assert_eq!(
                allocated_bytes, 0,
                "retained warmed direct call allocated bytes"
            );
            direct.traffic().leaf.validate_direct().unwrap();

            let mut direct_amplitudes =
                vec![Complex::new(f64::NAN, f64::NAN); point_count * amplitude_component_count];
            direct
                .extract_amplitudes_row_major(point_count, &mut direct_amplitudes)
                .expect("extract retained amplitudes for oracle comparison");
            assert_eq!(
                legacy_amplitude.output_scratch_f64.len(),
                direct_amplitudes.len()
            );
            for (direct_value, legacy_value) in direct_amplitudes
                .iter()
                .copied()
                .zip(legacy_amplitude.output_scratch_f64.iter().copied())
            {
                assert_close(direct_value, legacy_value);
            }

            if point_count == 129 {
                const SAMPLES: usize = 7;
                const REPEATS: usize = 200;
                let mut direct_ns = [0_u128; SAMPLES];
                let mut legacy_ns = [0_u128; SAMPLES];
                for sample in 0..SAMPLES {
                    let mut time_direct = || {
                        let start = Instant::now();
                        for _ in 0..REPEATS {
                            direct
                                .evaluate_validated(point_count, &direct_schedule)
                                .unwrap();
                        }
                        start.elapsed().as_nanos() / REPEATS as u128
                    };
                    let mut time_legacy = || {
                        let start = Instant::now();
                        for _ in 0..REPEATS {
                            for (stage_index, stage) in legacy_stages.iter_mut().enumerate() {
                                stage
                                    .evaluate_active_chunks_f64_into_state(
                                        point_count,
                                        parameter_count,
                                        &mut legacy,
                                        &retained_schedule.active_stage_chunk_indices[stage_index],
                                    )
                                    .unwrap();
                            }
                            legacy_amplitude
                                .evaluate_active_chunks_f64_into_scratch(
                                    point_count,
                                    &legacy,
                                    &retained_schedule.active_amplitude_chunk_indices,
                                )
                                .unwrap();
                        }
                        start.elapsed().as_nanos() / REPEATS as u128
                    };
                    if sample.is_multiple_of(2) {
                        direct_ns[sample] = time_direct();
                        legacy_ns[sample] = time_legacy();
                    } else {
                        legacy_ns[sample] = time_legacy();
                        direct_ns[sample] = time_direct();
                    }
                    std::hint::black_box((
                        direct.amplitude_planes().unwrap(),
                        &legacy_amplitude.output_scratch_f64,
                    ));
                }
                direct_ns.sort_unstable();
                legacy_ns.sort_unstable();
                let direct_median = direct_ns[SAMPLES / 2];
                let legacy_median = legacy_ns[SAMPLES / 2];
                eprintln!(
                    "retained-compiled-direct points={point_count} samples={SAMPLES} \
                     repeats={REPEATS} direct_ns={direct_ns:?} legacy_ns={legacy_ns:?} \
                     direct_median_ns={direct_median} legacy_median_ns={legacy_median} \
                     speedup={:.6}",
                    legacy_median as f64 / direct_median as f64
                );
            }
        }
    }

    #[test]
    fn retained_compiled_o3_artifact_native_runtime_direct_matches_legacy_contract() {
        let Some(root) = std::env::var_os("RUSTICOL_COMPILED_DIRECT_ARTIFACT") else {
            return;
        };
        let root = PathBuf::from(root);
        let mut direct = NativeRuntime::load(&root, None, None)
            .expect("load retained production Direct runtime");
        let mut legacy =
            NativeRuntime::load(&root, None, None).expect("load retained legacy oracle runtime");
        assert_eq!(
            compiled_direct_runtime_summary(&direct.runtime),
            compiled_direct_runtime_summary(&legacy.runtime),
            "independent loads must derive the same nested Direct configuration"
        );
        assert_eq!(
            direct.metadata_json().unwrap(),
            legacy.metadata_json().unwrap(),
            "independent loads must report deterministic nested Direct provenance"
        );
        disable_compiled_direct_recursive(&mut legacy.runtime);

        let (engine_count, requested_bytes, _, minimum_tile_capacity) =
            compiled_direct_runtime_summary(&direct.runtime);
        assert!(engine_count > 0, "retained runtime has one Direct engine");
        assert!(requested_bytes > 0, "retained Direct arenas own storage");
        assert!(
            minimum_tile_capacity > 0,
            "retained Direct engines have a nonzero tile"
        );
        eprintln!(
            "retained-compiled-direct-production engines={engine_count} \
             requested_bytes={requested_bytes} minimum_tile_capacity={minimum_tile_capacity}"
        );
        let metadata = direct.metadata();
        assert_eq!(
            metadata.compiled_direct_minimum_effective_tile_capacity,
            Some(minimum_tile_capacity)
        );
        assert!(
            metadata
                .compiled_direct_maximum_physical_scalar_values_per_point
                .is_some_and(|value| value > 0)
        );
        assert!(
            metadata
                .compiled_direct_maximum_hot_scalar_values_per_point
                .is_some_and(|value| value > 0)
        );
        assert!(
            metadata
                .compiled_direct_maximum_source_scalar_values_per_point
                .is_some_and(|value| value > 0)
        );
        assert!(
            metadata
                .compiled_direct_maximum_reduction_scalar_values_per_point
                .is_some_and(|value| value > 0)
        );
        assert_eq!(
            direct.metadata_json().unwrap(),
            direct.metadata_json().unwrap(),
            "compiled Direct effective configuration provenance must be deterministic"
        );

        let point = retained_validation_point(&direct);
        for point_count in [1usize, 7, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025] {
            let momenta = point.repeat(point_count);
            let mut direct_values = vec![f64::NAN; point_count];
            let mut legacy_values = vec![f64::NAN; point_count];
            direct
                .evaluate_f64_into(&momenta, point_count, &mut direct_values)
                .expect("evaluate retained production Direct totals");
            legacy
                .evaluate_f64_into(&momenta, point_count, &mut legacy_values)
                .expect("evaluate retained legacy totals");
            for (point_index, (direct_value, legacy_value)) in direct_values
                .iter()
                .copied()
                .zip(legacy_values.iter().copied())
                .enumerate()
            {
                assert_close_real(
                    direct_value,
                    legacy_value,
                    &format!("unselected total point={point_index} count={point_count}"),
                );
            }
        }

        let helicities = direct.helicities().expect("retained helicity metadata");
        let computed_helicities = helicities
            .iter()
            .filter(|helicity| helicity.computed && !helicity.structural_zero)
            .collect::<Vec<_>>();
        assert!(
            !computed_helicities.is_empty(),
            "retained artifact has one nonzero computed helicity"
        );
        let selected_helicity = computed_helicities[0].id.clone();
        let colors = direct.color_components().expect("retained color metadata");
        assert!(!colors.is_empty(), "retained artifact has one color");
        // Prefer a non-representative physical flow so the test exercises a
        // real topology-label permutation rather than only the identity map.
        let selected_color = colors
            .last()
            .expect("retained artifact has one color")
            .id
            .clone();

        let selector_points = 129usize;
        let selector_momenta = point.repeat(selector_points);
        let mut selector_cases = vec![(
            "all-flows-single-helicity",
            Some(std::slice::from_ref(&selected_helicity)),
            None,
        )];
        if direct.metadata().color_accuracy == "lc" {
            selector_cases.push((
                "single-flow-helicity-sum",
                None,
                Some(std::slice::from_ref(&selected_color)),
            ));
        }
        for (label, selected_helicities, selected_colors) in selector_cases {
            let direct_values = direct
                .evaluate_f64_with_selectors(
                    &selector_momenta,
                    selector_points,
                    selected_helicities,
                    selected_colors,
                    None,
                    None,
                )
                .expect("evaluate retained Direct global selector");
            let legacy_values = legacy
                .evaluate_f64_with_selectors(
                    &selector_momenta,
                    selector_points,
                    selected_helicities,
                    selected_colors,
                    None,
                    None,
                )
                .expect("evaluate retained legacy global selector");
            let resolved = direct
                .evaluate_resolved_f64(
                    &selector_momenta,
                    selector_points,
                    selected_helicities,
                    selected_colors,
                )
                .expect("resolve retained Direct global selector");
            let resolved_totals = resolved.totals();
            for point_index in 0..selector_points {
                assert_close_real(
                    direct_values[point_index],
                    legacy_values[point_index],
                    &format!("{label} direct/legacy point={point_index}"),
                );
                assert_close_real(
                    direct_values[point_index],
                    resolved_totals[point_index],
                    &format!("{label} total/resolved point={point_index}"),
                );
            }
        }

        let compiled_symmetric_group = direct
            .runtime
            .amplitude_stage
            .as_ref()
            .and_then(|amplitude| amplitude.color_contraction.as_ref())
            .is_some_and(ColorContractionRuntime::is_symmetric_group);
        if helicities.len() == 1
            && direct.metadata().color_accuracy == "full"
            && !compiled_symmetric_group
        {
            let sole_helicity = [selected_helicity.clone()];
            let explicit_total = direct
                .evaluate_f64_with_selectors(
                    &selector_momenta,
                    selector_points,
                    Some(&sole_helicity),
                    None,
                    None,
                    None,
                )
                .expect("evaluate selected FullColour explicit sole-helicity total");
            let unselected_total = direct
                .evaluate_f64(&selector_momenta, selector_points)
                .expect("evaluate selected FullColour implicit sole-helicity total");
            let resolved = direct
                .evaluate_resolved_f64(
                    &selector_momenta,
                    selector_points,
                    Some(&sole_helicity),
                    None,
                )
                .expect("resolve selected FullColour sole-helicity values");
            assert_eq!(resolved.helicity_ids.as_slice(), sole_helicity.as_slice());
            assert_eq!(resolved.color_ids.len(), 1);
            assert_eq!(resolved.point_count, selector_points);
            assert_eq!(resolved.values.len(), selector_points);
            let resolved_total = resolved.totals();
            for (point_index, ((explicit, unselected), resolved)) in explicit_total
                .iter()
                .copied()
                .zip(unselected_total.iter().copied())
                .zip(resolved_total.iter().copied())
                .enumerate()
            {
                assert_close_real(
                    explicit,
                    unselected,
                    &format!("sole-helicity explicit/unselected total point={point_index}"),
                );
                assert_close_real(
                    explicit,
                    resolved,
                    &format!("sole-helicity total/resolved point={point_index}"),
                );
            }

            let sole_helicity_index =
                u32::try_from(helicities[0].index).expect("sole public helicity index fits u32");
            let helicity_by_point = vec![sole_helicity_index; selector_points];
            let mut per_point_total = vec![f64::NAN; selector_points];
            direct
                .evaluate_f64_into_with_selectors(
                    &selector_momenta,
                    selector_points,
                    None,
                    None,
                    Some(&helicity_by_point),
                    None,
                    &mut per_point_total,
                )
                .expect("warm selected FullColour per-point sole-helicity total");
            let (result, allocations, allocated_bytes) = count_test_allocations(|| {
                direct.evaluate_f64_into_with_selectors(
                    &selector_momenta,
                    selector_points,
                    None,
                    None,
                    Some(&helicity_by_point),
                    None,
                    &mut per_point_total,
                )
            });
            result.expect("repeat selected FullColour per-point sole-helicity total");
            assert_eq!(allocations, 0, "warmed sole-helicity total allocated");
            assert_eq!(
                allocated_bytes, 0,
                "warmed sole-helicity total allocated bytes"
            );
            for (point_index, (per_point, explicit)) in per_point_total
                .iter()
                .copied()
                .zip(explicit_total.iter().copied())
                .enumerate()
            {
                assert_close_real(
                    per_point,
                    explicit,
                    &format!("sole-helicity per-point/global total point={point_index}"),
                );
            }
        }

        if direct.runtime.lc_topology_replay_enabled {
            let selector_batch = F64MomentumBatchView::from_contiguous_prevalidated(
                &selector_momenta,
                selector_points,
                direct.runtime.external_count,
                direct.input_crossing_map.as_deref(),
            )
            .expect("borrow retained selector batch");
            for (label, selected_helicities, selected_colors) in [
                ("all-components", None, None),
                (
                    "all-flows-single-helicity",
                    Some(BTreeSet::from([selected_helicity.clone()])),
                    None,
                ),
            ] {
                let mut candidate_values = vec![f64::NAN; selector_points];
                let mut oracle_values = vec![f64::NAN; selector_points];
                direct
                    .runtime
                    .run_f64_selected_into_unprofiled(
                        selector_batch,
                        selected_helicities.as_ref(),
                        selected_colors.as_ref(),
                        &mut candidate_values,
                    )
                    .expect("evaluate allocation-free replay candidate");
                direct
                    .runtime
                    .run_f64_selected_into_resolved_replay_oracle(
                        selector_batch,
                        selected_helicities.as_ref(),
                        selected_colors.as_ref(),
                        &mut oracle_values,
                    )
                    .expect("evaluate resolved replay oracle");
                for (point_index, (candidate_value, oracle_value)) in candidate_values
                    .iter()
                    .copied()
                    .zip(oracle_values.iter().copied())
                    .enumerate()
                {
                    assert_close_real(
                        candidate_value,
                        oracle_value,
                        &format!("{label} replay candidate/oracle point={point_index}"),
                    );
                }
            }
        }

        if direct.metadata().color_accuracy == "lc"
            && computed_helicities.len() >= 2
            && colors.len() >= 2
        {
            let point_count = 129usize;
            let momenta = point.repeat(point_count);
            let helicity_by_point = (0..point_count)
                .map(|point_index| {
                    computed_helicities[point_index % 2]
                        .index
                        .try_into()
                        .expect("physical helicity index fits u32")
                })
                .collect::<Vec<u32>>();
            let color_by_point = (0..point_count)
                .map(|point_index| {
                    colors[point_index % 2]
                        .index
                        .try_into()
                        .expect("physical color index fits u32")
                })
                .collect::<Vec<u32>>();
            let direct_values = direct
                .evaluate_f64_with_selectors(
                    &momenta,
                    point_count,
                    None,
                    None,
                    Some(&helicity_by_point),
                    Some(&color_by_point),
                )
                .expect("evaluate retained Direct per-point selectors");
            let legacy_values = legacy
                .evaluate_f64_with_selectors(
                    &momenta,
                    point_count,
                    None,
                    None,
                    Some(&helicity_by_point),
                    Some(&color_by_point),
                )
                .expect("evaluate retained legacy per-point selectors");
            for (point_index, (direct_value, legacy_value)) in direct_values
                .iter()
                .copied()
                .zip(legacy_values.iter().copied())
                .enumerate()
            {
                assert_close_real(
                    direct_value,
                    legacy_value,
                    &format!("per-point selector point={point_index}"),
                );
            }
        }

        let parameters = direct
            .model_parameters()
            .expect("retained model parameter metadata");
        if let Some(parameter) = parameters.iter().find(|parameter| parameter.mutable) {
            let changed_real = if parameter.default == 0.0 {
                0.125
            } else {
                parameter.default * 1.01
            };
            direct
                .set_model_parameter(&parameter.name, changed_real, parameter.default_imaginary)
                .expect("update retained Direct parameter");
            legacy
                .set_model_parameter(&parameter.name, changed_real, parameter.default_imaginary)
                .expect("update retained legacy parameter");
            let direct_changed = direct
                .evaluate_f64(&point, 1)
                .expect("evaluate retained Direct changed parameter");
            let legacy_changed = legacy
                .evaluate_f64(&point, 1)
                .expect("evaluate retained legacy changed parameter");
            assert_close_real(
                direct_changed[0],
                legacy_changed[0],
                "changed model parameter",
            );
            direct
                .set_model_parameter(
                    &parameter.name,
                    parameter.default,
                    parameter.default_imaginary,
                )
                .expect("restore retained Direct parameter");
            legacy
                .set_model_parameter(
                    &parameter.name,
                    parameter.default,
                    parameter.default_imaginary,
                )
                .expect("restore retained legacy parameter");
        }

        if !cfg!(debug_assertions) {
            let mut workloads = vec![
                ("all-components", None, None),
                (
                    "all-flows-single-helicity",
                    Some(vec![selected_helicity.clone()]),
                    None,
                ),
            ];
            if direct.metadata().color_accuracy == "lc" {
                workloads.push((
                    "single-flow-helicity-sum",
                    None,
                    Some(vec![selected_color.clone()]),
                ));
            }
            for point_count in [128usize, 1024] {
                let momenta = point.repeat(point_count);
                let repeats = if point_count == 128 { 32 } else { 8 };
                for (label, helicities, colors) in &workloads {
                    let mut direct_output = vec![f64::NAN; point_count];
                    let mut legacy_output = vec![f64::NAN; point_count];
                    direct
                        .evaluate_f64_into_with_selectors(
                            &momenta,
                            point_count,
                            helicities.as_deref(),
                            colors.as_deref(),
                            None,
                            None,
                            &mut direct_output,
                        )
                        .expect("warm retained Direct timing workload");
                    legacy
                        .evaluate_f64_into_with_selectors(
                            &momenta,
                            point_count,
                            helicities.as_deref(),
                            colors.as_deref(),
                            None,
                            None,
                            &mut legacy_output,
                        )
                        .expect("warm retained legacy timing workload");
                    let mut direct_samples = [0u128; 7];
                    let mut legacy_samples = [0u128; 7];
                    for sample in 0usize..7 {
                        let mut time_direct = || {
                            let start = Instant::now();
                            for _ in 0..repeats {
                                direct
                                    .evaluate_f64_into_with_selectors(
                                        &momenta,
                                        point_count,
                                        helicities.as_deref(),
                                        colors.as_deref(),
                                        None,
                                        None,
                                        &mut direct_output,
                                    )
                                    .expect("time retained Direct workload");
                            }
                            start.elapsed().as_nanos() / repeats
                        };
                        let mut time_legacy = || {
                            let start = Instant::now();
                            for _ in 0..repeats {
                                legacy
                                    .evaluate_f64_into_with_selectors(
                                        &momenta,
                                        point_count,
                                        helicities.as_deref(),
                                        colors.as_deref(),
                                        None,
                                        None,
                                        &mut legacy_output,
                                    )
                                    .expect("time retained legacy workload");
                            }
                            start.elapsed().as_nanos() / repeats
                        };
                        if sample.is_multiple_of(2) {
                            direct_samples[sample] = time_direct();
                            legacy_samples[sample] = time_legacy();
                        } else {
                            legacy_samples[sample] = time_legacy();
                            direct_samples[sample] = time_direct();
                        }
                        std::hint::black_box((&direct_output, &legacy_output));
                    }
                    direct_samples.sort_unstable();
                    legacy_samples.sort_unstable();
                    let direct_median = direct_samples[3];
                    let legacy_median = legacy_samples[3];
                    eprintln!(
                        "retained-compiled-direct-wall color_accuracy={} workload={label} \
                         points={point_count} samples=7 repeats={repeats} \
                         direct_samples_ns={direct_samples:?} \
                         legacy_samples_ns={legacy_samples:?} \
                         direct_median_ns={direct_median} legacy_median_ns={legacy_median} \
                         direct_ns_per_point={:.3} legacy_ns_per_point={:.3} speedup={:.6}",
                        direct.metadata().color_accuracy,
                        direct_median as f64 / point_count as f64,
                        legacy_median as f64 / point_count as f64,
                        legacy_median as f64 / direct_median as f64,
                    );
                }
            }
        }

        if !cfg!(debug_assertions)
            && std::env::var_os("RUSTICOL_TEST_BENCHMARK_COMPILED_REPLAY_TOTALS").is_some()
            && direct.metadata().color_accuracy == "lc"
        {
            let selected_helicity_set = BTreeSet::from([selected_helicity.clone()]);
            for point_count in [128usize, 1024] {
                let momenta = point.repeat(point_count);
                let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                    &momenta,
                    point_count,
                    direct.runtime.external_count,
                    direct.input_crossing_map.as_deref(),
                )
                .expect("borrow retained replay timing batch");
                let repeats = if point_count == 128 { 64 } else { 8 };
                for (label, helicities) in [
                    ("all-components", None),
                    ("all-flows-single-helicity", Some(&selected_helicity_set)),
                ] {
                    let mut candidate_output = vec![f64::NAN; point_count];
                    let mut oracle_output = vec![f64::NAN; point_count];
                    direct
                        .runtime
                        .run_f64_selected_into_unprofiled(
                            batch,
                            helicities,
                            None,
                            &mut candidate_output,
                        )
                        .expect("warm allocation-free replay candidate");
                    direct
                        .runtime
                        .run_f64_selected_into_resolved_replay_oracle(
                            batch,
                            helicities,
                            None,
                            &mut oracle_output,
                        )
                        .expect("warm resolved replay oracle");
                    let mut candidate_samples = [0u128; 7];
                    let mut oracle_samples = [0u128; 7];
                    for sample in 0usize..7 {
                        if sample.is_multiple_of(2) {
                            let start = Instant::now();
                            for _ in 0..repeats {
                                direct
                                    .runtime
                                    .run_f64_selected_into_unprofiled(
                                        batch,
                                        helicities,
                                        None,
                                        &mut candidate_output,
                                    )
                                    .expect("time allocation-free replay candidate");
                            }
                            candidate_samples[sample] = start.elapsed().as_nanos() / repeats;
                            let start = Instant::now();
                            for _ in 0..repeats {
                                direct
                                    .runtime
                                    .run_f64_selected_into_resolved_replay_oracle(
                                        batch,
                                        helicities,
                                        None,
                                        &mut oracle_output,
                                    )
                                    .expect("time resolved replay oracle");
                            }
                            oracle_samples[sample] = start.elapsed().as_nanos() / repeats;
                        } else {
                            let start = Instant::now();
                            for _ in 0..repeats {
                                direct
                                    .runtime
                                    .run_f64_selected_into_resolved_replay_oracle(
                                        batch,
                                        helicities,
                                        None,
                                        &mut oracle_output,
                                    )
                                    .expect("time resolved replay oracle");
                            }
                            oracle_samples[sample] = start.elapsed().as_nanos() / repeats;
                            let start = Instant::now();
                            for _ in 0..repeats {
                                direct
                                    .runtime
                                    .run_f64_selected_into_unprofiled(
                                        batch,
                                        helicities,
                                        None,
                                        &mut candidate_output,
                                    )
                                    .expect("time allocation-free replay candidate");
                            }
                            candidate_samples[sample] = start.elapsed().as_nanos() / repeats;
                        }
                        std::hint::black_box((&candidate_output, &oracle_output));
                    }
                    candidate_samples.sort_unstable();
                    oracle_samples.sort_unstable();
                    let candidate_median = candidate_samples[3];
                    let oracle_median = oracle_samples[3];
                    let mut candidate_deviations =
                        candidate_samples.map(|value| value.abs_diff(candidate_median));
                    let mut oracle_deviations =
                        oracle_samples.map(|value| value.abs_diff(oracle_median));
                    candidate_deviations.sort_unstable();
                    oracle_deviations.sort_unstable();
                    eprintln!(
                        "retained-compiled-replay-wall workload={label} points={point_count} \
                         samples=7 repeats={repeats} candidate_samples_ns={candidate_samples:?} \
                         oracle_samples_ns={oracle_samples:?} \
                         candidate_median_ns={candidate_median} \
                         candidate_mad_ns={} oracle_median_ns={oracle_median} oracle_mad_ns={} \
                         candidate_ns_per_point={:.3} oracle_ns_per_point={:.3} speedup={:.6}",
                        candidate_deviations[3],
                        oracle_deviations[3],
                        candidate_median as f64 / point_count as f64,
                        oracle_median as f64 / point_count as f64,
                        oracle_median as f64 / candidate_median as f64,
                    );
                }
            }
        }

        let allocation_points = 128usize;
        let allocation_momenta = point.repeat(allocation_points);
        let mut allocation_output = vec![f64::NAN; allocation_points];
        direct
            .evaluate_f64_into(
                &allocation_momenta,
                allocation_points,
                &mut allocation_output,
            )
            .expect("warm retained Direct evaluate-into");
        let (result, allocations, allocated_bytes) = count_test_allocations(|| {
            direct.evaluate_f64_into(
                &allocation_momenta,
                allocation_points,
                &mut allocation_output,
            )
        });
        result.expect("repeat retained Direct evaluate-into");
        eprintln!(
            "retained-compiled-direct-public-allocation allocations={allocations} \
             allocated_bytes={allocated_bytes}"
        );
        assert_eq!(
            allocations, 0,
            "warmed public compiled Direct evaluation allocated"
        );
        assert_eq!(
            allocated_bytes, 0,
            "warmed public compiled Direct evaluation allocated bytes"
        );
        let arena_profile = direct
            .evaluate_f64_arena_profile_repeated(
                &allocation_momenta,
                allocation_points,
                3,
                None,
                None,
            )
            .expect("profile warmed retained Direct evaluate-into boundary");
        assert_eq!(arena_profile.values.len(), allocation_points);
        for (point_index, (profiled, expected)) in arena_profile
            .values
            .iter()
            .copied()
            .zip(&allocation_output)
            .enumerate()
        {
            assert_close_real(
                profiled,
                *expected,
                &format!("warmed retained Direct profile point={point_index}"),
            );
        }
        assert_eq!(
            arena_profile
                .profile
                .native_input_container_allocation_count,
            0
        );
        assert_eq!(arena_profile.profile.native_output_allocation_count, 0);
        assert_eq!(arena_profile.profile.observed_scratch_reallocation_count, 0);
        assert!(
            arena_profile.profile.compiled_direct_arena_engine_count > 0,
            "warmed Direct profile did not authenticate an Arena engine"
        );
        assert_eq!(
            arena_profile.profile.compiled_direct_arena_call_count,
            arena_profile.profile.evaluator_backend_call_count,
        );
        assert!(
            arena_profile.profile.compiled_direct_arena_call_count > 0,
            "warmed Direct profile observed no evaluator calls"
        );
        assert_eq!(
            arena_profile
                .profile
                .compiled_direct_arena_minimum_effective_tile_capacity,
            metadata
                .compiled_direct_minimum_effective_tile_capacity
                .unwrap() as u64
        );
        assert_eq!(
            arena_profile
                .profile
                .compiled_direct_arena_maximum_physical_scalar_values_per_point,
            metadata
                .compiled_direct_maximum_physical_scalar_values_per_point
                .unwrap() as u64
        );
        assert_eq!(
            arena_profile
                .profile
                .compiled_direct_arena_maximum_hot_scalar_values_per_point,
            metadata
                .compiled_direct_maximum_hot_scalar_values_per_point
                .unwrap() as u64
        );
        assert_eq!(
            arena_profile
                .profile
                .compiled_direct_arena_maximum_source_scalar_values_per_point,
            metadata
                .compiled_direct_maximum_source_scalar_values_per_point
                .unwrap() as u64
        );
        assert_eq!(
            arena_profile
                .profile
                .compiled_direct_arena_maximum_reduction_scalar_values_per_point,
            metadata
                .compiled_direct_maximum_reduction_scalar_values_per_point
                .unwrap() as u64
        );

        let allocation_batch = F64MomentumBatchView::from_contiguous_prevalidated(
            &allocation_momenta,
            allocation_points,
            direct.runtime.external_count,
            direct.input_crossing_map.as_deref(),
        )
        .expect("borrow retained allocation batch");
        direct
            .runtime
            .run_f64_selected_into_unprofiled(allocation_batch, None, None, &mut allocation_output)
            .expect("warm retained Direct runtime");
        let (result, runtime_allocations, runtime_allocated_bytes) = count_test_allocations(|| {
            direct.runtime.run_f64_selected_into_unprofiled(
                allocation_batch,
                None,
                None,
                &mut allocation_output,
            )
        });
        result.expect("repeat retained Direct runtime");
        eprintln!(
            "retained-compiled-direct-runtime-allocation allocations={runtime_allocations} \
             allocated_bytes={runtime_allocated_bytes}"
        );
        assert_eq!(
            runtime_allocations, 0,
            "warmed compiled Direct runtime allocated"
        );
        assert_eq!(
            runtime_allocated_bytes, 0,
            "warmed compiled Direct runtime allocated bytes"
        );

        let selected_color_ids = [selected_color.clone()];
        let selected_helicity_ids = [selected_helicity.clone()];
        let selected_colors = BTreeSet::from([selected_color]);
        let selected_helicities = BTreeSet::from([selected_helicity]);
        let mut allocation_cases = vec![("selected-helicity", Some(&selected_helicities), None)];
        if direct.metadata().color_accuracy == "lc" {
            allocation_cases.push(("selected-flow", None, Some(&selected_colors)));
        }
        for (allocation_label, allocation_helicities, allocation_colors) in allocation_cases {
            direct
                .runtime
                .run_f64_selected_into_unprofiled(
                    allocation_batch,
                    allocation_helicities,
                    allocation_colors,
                    &mut allocation_output,
                )
                .expect("warm retained Direct selected runtime");
            let (result, selected_allocations, selected_allocated_bytes) =
                count_test_allocations(|| {
                    direct.runtime.run_f64_selected_into_unprofiled(
                        allocation_batch,
                        allocation_helicities,
                        allocation_colors,
                        &mut allocation_output,
                    )
                });
            result.expect("repeat retained Direct selected runtime");
            eprintln!(
                "retained-compiled-direct-{allocation_label}-allocation \
                 allocations={selected_allocations} allocated_bytes={selected_allocated_bytes}"
            );
            assert_eq!(
                selected_allocations, 0,
                "warmed selected Direct arena execution allocated"
            );
            assert_eq!(
                selected_allocated_bytes, 0,
                "warmed selected Direct arena execution allocated bytes"
            );
        }
        let mut profile_cases = vec![(
            "selected-helicity",
            Some(selected_helicity_ids.as_slice()),
            None,
        )];
        if direct.metadata().color_accuracy == "lc" {
            profile_cases.push(("selected-flow", None, Some(selected_color_ids.as_slice())));
        }
        for (label, helicities, colors) in profile_cases {
            direct
                .evaluate_f64_into_with_selectors(
                    &allocation_momenta,
                    allocation_points,
                    helicities,
                    colors,
                    None,
                    None,
                    &mut allocation_output,
                )
                .unwrap_or_else(|error| panic!("evaluate retained Direct {label} oracle: {error}"));
            let selected_profile = direct
                .evaluate_f64_arena_profile_repeated(
                    &allocation_momenta,
                    allocation_points,
                    3,
                    helicities,
                    colors,
                )
                .unwrap_or_else(|error| panic!("profile retained Direct {label}: {error}"));
            for (point_index, (profiled, expected)) in selected_profile
                .values
                .iter()
                .copied()
                .zip(&allocation_output)
                .enumerate()
            {
                assert_close_real(
                    profiled,
                    *expected,
                    &format!("profiled retained Direct {label} point={point_index}"),
                );
            }
            assert_eq!(
                selected_profile
                    .profile
                    .native_input_container_allocation_count,
                0,
                "profiled retained Direct {label} input allocations"
            );
            assert_eq!(
                selected_profile.profile.native_output_allocation_count, 0,
                "profiled retained Direct {label} output allocations"
            );
            assert_eq!(
                selected_profile.profile.observed_scratch_reallocation_count, 0,
                "profiled retained Direct {label} scratch reallocations"
            );
        }

        let (engine_count, requested_bytes, hot_calls, minimum_tile_capacity) =
            compiled_direct_runtime_summary(&direct.runtime);
        assert!(hot_calls > 0, "retained production Direct leaves executed");
        eprintln!(
            "retained-compiled-direct-production-final engines={engine_count} \
             requested_bytes={requested_bytes} hot_calls={hot_calls} \
             minimum_tile_capacity={minimum_tile_capacity}"
        );
    }

    #[test]
    fn retained_compiled_replay_totals_quick_contract() {
        let Some(root) = std::env::var_os("RUSTICOL_TEST_COMPILED_REPLAY_QUICK_ARTIFACT") else {
            return;
        };
        let mut runtime =
            NativeRuntime::load(PathBuf::from(root), None, None).expect("load replay artifact");
        assert_eq!(
            runtime.metadata().color_accuracy,
            "lc",
            "quick replay contract requires LC"
        );
        let point = retained_validation_point(&runtime);
        let helicities = runtime.helicities().expect("retained helicity metadata");
        let selected_helicity = helicities
            .iter()
            .find(|helicity| helicity.computed && !helicity.structural_zero)
            .expect("retained artifact has a nonzero computed helicity")
            .id
            .clone();
        let colors = runtime.color_components().expect("retained color metadata");
        let selected_color = colors
            .last()
            .expect("retained artifact has one physical color")
            .id
            .clone();
        let selected_helicities = BTreeSet::from([selected_helicity]);
        let selected_colors = BTreeSet::from([selected_color]);

        let parity_points = 7usize;
        let parity_momenta = point.repeat(parity_points);
        let parity_batch = F64MomentumBatchView::from_contiguous_prevalidated(
            &parity_momenta,
            parity_points,
            runtime.runtime.external_count,
            runtime.input_crossing_map.as_deref(),
        )
        .expect("borrow replay parity batch");
        for (label, helicities, colors) in [
            ("all-components", None, None),
            (
                "all-flows-single-helicity",
                Some(&selected_helicities),
                None,
            ),
            ("single-flow-helicity-sum", None, Some(&selected_colors)),
        ] {
            let mut candidate = vec![f64::NAN; parity_points];
            let mut oracle = vec![f64::NAN; parity_points];
            runtime
                .runtime
                .run_f64_selected_into_unprofiled(parity_batch, helicities, colors, &mut candidate)
                .expect("evaluate allocation-free replay candidate");
            runtime
                .runtime
                .run_f64_selected_into_resolved_replay_oracle(
                    parity_batch,
                    helicities,
                    colors,
                    &mut oracle,
                )
                .expect("evaluate resolved replay oracle");
            for (point_index, (candidate, oracle)) in
                candidate.iter().copied().zip(oracle).enumerate()
            {
                assert_close_real(
                    candidate,
                    oracle,
                    &format!("{label} quick replay point={point_index}"),
                );
            }
        }

        let allocation_points = 128usize;
        let allocation_momenta = point.repeat(allocation_points);
        let allocation_batch = F64MomentumBatchView::from_contiguous_prevalidated(
            &allocation_momenta,
            allocation_points,
            runtime.runtime.external_count,
            runtime.input_crossing_map.as_deref(),
        )
        .expect("borrow replay allocation batch");
        let mut output = vec![f64::NAN; allocation_points];
        for (label, helicities, colors) in [
            ("all-components", None, None),
            (
                "all-flows-single-helicity",
                Some(&selected_helicities),
                None,
            ),
            ("single-flow-helicity-sum", None, Some(&selected_colors)),
        ] {
            runtime
                .runtime
                .run_f64_selected_into_unprofiled(allocation_batch, helicities, colors, &mut output)
                .expect("warm replay allocation workload");
            let (result, allocations, allocated_bytes) = count_test_allocations(|| {
                runtime.runtime.run_f64_selected_into_unprofiled(
                    allocation_batch,
                    helicities,
                    colors,
                    &mut output,
                )
            });
            result.expect("repeat replay allocation workload");
            eprintln!(
                "retained-compiled-replay-quick-allocation workload={label} \
                 allocations={allocations} allocated_bytes={allocated_bytes}"
            );
            assert_eq!(allocations, 0, "warmed replay workload {label} allocated");
            assert_eq!(
                allocated_bytes, 0,
                "warmed replay workload {label} allocated bytes"
            );
        }
    }
}
