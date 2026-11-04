// SPDX-License-Identifier: 0BSD

//! Recurrence adapter for standard SymJIT direct-arena P-kernels.
//!
//! The adapter deliberately owns the recurrence policy which used to live in
//! pyAmpliCol's SymJIT fork. Proven-disjoint identity overwrites bind ordinary
//! P-kernel outputs directly to their arena destinations. Every accumulation,
//! non-identity factor, and before-write alias instead evaluates once into
//! persistent split-complex scratch planes, after which Rusticol applies the
//! exact complex factor and role-specific overwrite/add policy. Inputs which
//! are scalar at recurrence runtime are represented by persistent broadcast
//! planes and refreshed only when their value changes.

use super::symjit_plane::{
    PlaneDescriptor, SymjitPlaneKernel, SymjitPlaneLayout, SymjitPlaneTable,
};
use crate::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectClosureExecutor, DirectContributionExecutor,
    DirectExecutorHandle, DirectFactorView, DirectFinalizationExecutor, DirectMomentumView,
    DirectParameterView, record_direct_executor_error_detail,
};
use crate::recurrence::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DIRECT_NONE_U32, DirectClosureRow,
    DirectContributionRow, DirectExecutorRole, DirectFinalizationRow,
};
use crate::{RusticolError, RusticolResult};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr;
#[cfg(target_arch = "aarch64")]
use wide::f64x2;
#[cfg(target_arch = "x86_64")]
use wide::f64x4;

pub(crate) const SYMJIT_RECURRENCE_PLANE_APPLICATION_ABI: &str =
    "pyamplicol-symjit-plane-application-v2";

const STATUS_INVALID_CONTEXT: c_int = 1;
const STATUS_INVALID_ARGUMENT: c_int = 2;
const STATUS_ROLE_MISMATCH: c_int = 3;
const STATUS_EXECUTION_FAILED: c_int = 4;

/// One model-fixed arena-plane projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SymjitDirectPlaneProjection {
    ParentCurrent {
        parent: u8,
        component: u16,
        imaginary: bool,
    },
    Momentum {
        operand: u8,
        lorentz_component: u16,
    },
    DestinationCurrent {
        component: u16,
        imaginary: bool,
    },
    DestinationAmplitude {
        component: u16,
        imaginary: bool,
    },
}

/// One model-fixed scalar projection used to fill a broadcast plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SymjitDirectScalarProjection {
    Parameter { index: u32, imaginary: bool },
    ExactFactor { imaginary: bool },
    Literal(f64),
}

/// Maps one split source-input plane in the P-kernel to recurrence storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SymjitDirectParameterBinding {
    Plane { index: u32 },
    Broadcast { index: u32 },
}

/// Owns the context addressed by a [`DirectExecutorHandle`].
pub(crate) struct LoadedSymjitDirectExecutor {
    context: Box<SymjitDirectExecutorContext>,
}

struct SymjitDirectExecutorContext {
    role: DirectExecutorRole,
    kernel: SymjitPlaneKernel,
    parameter_bindings: Box<[SymjitDirectParameterBinding]>,
    input_planes: Box<[SymjitDirectPlaneProjection]>,
    scalars: Box<[SymjitDirectScalarProjection]>,
    output_destinations: Box<[u32]>,
    broadcast_slot_by_binding: Box<[Option<u32>]>,
    broadcast_scalar_by_slot: Box<[u32]>,
    exact_factor_broadcast_slots: Box<[usize]>,
    scalar_workspace: RefCell<PlaneWorkspace>,
    tiled_workspace: RefCell<PlaneWorkspace>,
    internal_scratch_bytes: Cell<u64>,
    internal_broadcast_bytes: Cell<u64>,
    #[cfg(test)]
    invariant_broadcast_resolution_count: Cell<u64>,
    #[cfg(test)]
    exact_factor_broadcast_resolution_count: Cell<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DescriptorStorageIdentity {
    current_re: usize,
    current_im: usize,
    current_scalar_len: u64,
    amplitude_re: usize,
    amplitude_im: usize,
    amplitude_scalar_len: u64,
    point_stride: u32,
    momenta: usize,
    momentum_scalar_len: u64,
    momentum_form_count: u32,
    momentum_lorentz_component_count: u16,
}

impl DescriptorStorageIdentity {
    fn new(arena: DirectArenaView, momenta: DirectMomentumView) -> Self {
        Self {
            current_re: arena.current_re.addr(),
            current_im: arena.current_im.addr(),
            current_scalar_len: arena.current_scalar_len,
            amplitude_re: arena.amplitude_re.addr(),
            amplitude_im: arena.amplitude_im.addr(),
            amplitude_scalar_len: arena.amplitude_scalar_len,
            point_stride: arena.point_stride,
            momenta: momenta.values.addr(),
            momentum_scalar_len: momenta.scalar_len,
            momentum_form_count: momenta.form_count,
            momentum_lorentz_component_count: momenta.lorentz_component_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RowTableIdentity {
    role: DirectExecutorRole,
    address: usize,
    row_count: u32,
}

impl Hash for RowTableIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.role as u16).hash(state);
        self.address.hash(state);
        self.row_count.hash(state);
    }
}

const ROW_GROUP_LINEAR_LOOKUP_LIMIT: usize = 4;

struct CachedRowGroup {
    identity: RowTableIdentity,
    rows: Vec<CachedRowTables>,
    descriptor_bytes: usize,
}

struct CachedRowTables {
    scratch: SymjitPlaneTable<'static>,
    direct: Option<SymjitPlaneTable<'static>>,
}

/// Hot storage owned by one loaded recurrence executor.
///
/// Every cached table contains only the generic raw descriptors exposed by the
/// SymJIT patch. Multiple tables may therefore name the same persistent
/// scratch/broadcast planes without constructing overlapping Rust references;
/// the executor calls them strictly sequentially.
#[derive(Default)]
struct PlaneWorkspace {
    storage: Option<DescriptorStorageIdentity>,
    row_groups: Vec<CachedRowGroup>,
    row_group_indices: HashMap<RowTableIdentity, usize>,
    descriptor_bytes: usize,
    scratch: Vec<f64>,
    broadcasts: Vec<f64>,
    broadcast_value_bits: Vec<Option<u64>>,
    point_stride: usize,
}

impl PlaneWorkspace {
    fn invalidate_row_tables(&mut self) {
        self.row_groups.clear();
        self.row_group_indices.clear();
        self.storage = None;
        self.descriptor_bytes = 0;
    }

    fn cached_row_group_index(&self, identity: RowTableIdentity) -> Option<usize> {
        // The usual full-schedule cache has only a few entries. Varied runtime
        // selections can add many row slices, which must not slow warm lookup.
        if self.row_groups.len() <= ROW_GROUP_LINEAR_LOOKUP_LIMIT {
            self.row_groups
                .iter()
                .position(|cached| cached.identity == identity)
        } else {
            self.row_group_indices.get(&identity).copied()
        }
    }

    fn cache_row_group(&mut self, cached: CachedRowGroup) -> RusticolResult<usize> {
        let total_bytes =
            checked_descriptor_cache_bytes(self.descriptor_bytes, cached.descriptor_bytes)?;
        self.row_groups.try_reserve(1).map_err(|error| {
            RusticolError::compatibility(format!(
                "could not reserve recurrence descriptor row group: {error}"
            ))
        })?;
        let index = self.row_groups.len();
        if index >= ROW_GROUP_LINEAR_LOOKUP_LIMIT {
            let additional = if index == ROW_GROUP_LINEAR_LOOKUP_LIMIT {
                index + 1
            } else {
                1
            };
            self.row_group_indices
                .try_reserve(additional)
                .map_err(|error| {
                    RusticolError::compatibility(format!(
                        "could not reserve recurrence descriptor row index: {error}"
                    ))
                })?;
            if index == ROW_GROUP_LINEAR_LOOKUP_LIMIT {
                self.row_group_indices.extend(
                    self.row_groups
                        .iter()
                        .enumerate()
                        .map(|(index, cached)| (cached.identity, index)),
                );
            }
            self.row_group_indices.insert(cached.identity, index);
        }
        self.descriptor_bytes = total_bytes;
        self.row_groups.push(cached);
        Ok(index)
    }

    fn prepare(
        &mut self,
        point_stride: usize,
        input_plane_count: usize,
        output_plane_count: usize,
        broadcast_plane_count: usize,
    ) -> RusticolResult<()> {
        let descriptor_count = input_plane_count
            .checked_add(output_plane_count)
            .ok_or_else(|| RusticolError::internal("recurrence descriptor count overflows"))?;
        let expected_scratch = output_plane_count
            .checked_mul(point_stride)
            .ok_or_else(|| RusticolError::internal("recurrence scratch size overflows"))?;
        let expected_broadcasts = broadcast_plane_count
            .checked_mul(point_stride)
            .ok_or_else(|| RusticolError::internal("recurrence broadcast size overflows"))?;
        if self.point_stride == point_stride
            && self.scratch.len() == expected_scratch
            && self.broadcasts.len() == expected_broadcasts
        {
            return Ok(());
        }

        // Drop every raw table before replacing its backing storage.
        self.invalidate_row_tables();
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(expected_scratch)
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate recurrence scratch planes: {error}"
                ))
            })?;
        scratch.resize(expected_scratch, 0.0);
        let mut broadcasts = Vec::new();
        broadcasts
            .try_reserve_exact(expected_broadcasts)
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate recurrence broadcast planes: {error}"
                ))
            })?;
        broadcasts.resize(expected_broadcasts, 0.0);
        let mut broadcast_value_bits = Vec::new();
        broadcast_value_bits
            .try_reserve_exact(broadcast_plane_count)
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate recurrence broadcast cache: {error}"
                ))
            })?;
        broadcast_value_bits.resize(broadcast_plane_count, None);
        self.scratch = scratch;
        self.broadcasts = broadcasts;
        self.broadcast_value_bits = broadcast_value_bits;
        self.point_stride = point_stride;
        debug_assert_eq!(descriptor_count, input_plane_count + output_plane_count);
        Ok(())
    }

    fn broadcast_pointer(&mut self, slot: usize, value: f64) -> RusticolResult<(*mut f64, bool)> {
        let start = slot
            .checked_mul(self.point_stride)
            .ok_or_else(|| RusticolError::internal("broadcast plane offset overflows"))?;
        let end = start
            .checked_add(self.point_stride)
            .ok_or_else(|| RusticolError::internal("broadcast plane range overflows"))?;
        let plane = self.broadcasts.get_mut(start..end).ok_or_else(|| {
            RusticolError::internal("broadcast plane is outside its persistent workspace")
        })?;
        let bits = value.to_bits();
        let cached = self.broadcast_value_bits.get_mut(slot).ok_or_else(|| {
            RusticolError::internal("broadcast cache slot is outside its workspace")
        })?;
        let refreshed = *cached != Some(bits);
        if refreshed {
            plane.fill(value);
            *cached = Some(bits);
        }
        Ok((plane.as_mut_ptr(), refreshed))
    }

    fn scratch_pointer(&mut self, plane: usize) -> RusticolResult<*mut f64> {
        let start = plane
            .checked_mul(self.point_stride)
            .ok_or_else(|| RusticolError::internal("scratch plane offset overflows"))?;
        let end = start
            .checked_add(self.point_stride)
            .ok_or_else(|| RusticolError::internal("scratch plane range overflows"))?;
        self.scratch
            .get_mut(start..end)
            .map(<[f64]>::as_mut_ptr)
            .ok_or_else(|| RusticolError::internal("scratch plane is outside its workspace"))
    }

    fn broadcast_plane_pointer(&mut self, slot: usize) -> RusticolResult<*mut f64> {
        let start = slot
            .checked_mul(self.point_stride)
            .ok_or_else(|| RusticolError::internal("broadcast plane offset overflows"))?;
        let end = start
            .checked_add(self.point_stride)
            .ok_or_else(|| RusticolError::internal("broadcast plane range overflows"))?;
        self.broadcasts
            .get_mut(start..end)
            .map(<[f64]>::as_mut_ptr)
            .ok_or_else(|| RusticolError::internal("broadcast plane is outside its workspace"))
    }
}

impl LoadedSymjitDirectExecutor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_prepared_application_bytes(
        bytes: &[u8],
        display_path: PathBuf,
        application_abi: &str,
        expected_optimization_level: u32,
        expected_compression: bool,
        role: DirectExecutorRole,
        parameter_bindings: Vec<SymjitDirectParameterBinding>,
        input_planes: Vec<SymjitDirectPlaneProjection>,
        scalars: Vec<SymjitDirectScalarProjection>,
        output_destinations: Vec<u32>,
    ) -> RusticolResult<Self> {
        if role == DirectExecutorRole::Source {
            return Err(RusticolError::compatibility(
                "recurrence source executors are Rusticol SourceIR intrinsics, not SymJIT plane applications",
            ));
        }
        if application_abi != SYMJIT_RECURRENCE_PLANE_APPLICATION_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported recurrence SymJIT plane-application ABI {application_abi:?}; regenerate the prepared model with this pyAmpliCol version"
            )));
        }
        if !parameter_bindings.len().is_multiple_of(2)
            || !output_destinations.len().is_multiple_of(2)
        {
            return Err(RusticolError::integrity(
                "recurrence SymJIT split-complex binding has an odd plane count",
            ));
        }
        let layout =
            SymjitPlaneLayout::complex(parameter_bindings.len() / 2, output_destinations.len() / 2);
        let kernel = SymjitPlaneKernel::load_bytes(bytes, display_path.clone(), layout)?;
        if u32::from(kernel.optimization_level()) != expected_optimization_level {
            return Err(RusticolError::integrity(format!(
                "recurrence SymJIT plane application {} stores optimization level {}, but its \
                 authenticated template declares optimization level \
                 {expected_optimization_level}; regenerate the prepared model",
                display_path.display(),
                kernel.optimization_level(),
            )));
        }
        if kernel.compression() != expected_compression {
            return Err(RusticolError::integrity(format!(
                "recurrence SymJIT plane application {} stores compression {}, but its \
                 authenticated prepared-kernel manifest declares compression \
                 {expected_compression}; regenerate the prepared model",
                display_path.display(),
                kernel.compression(),
            )));
        }
        validate_binding(
            &kernel,
            role,
            &parameter_bindings,
            &input_planes,
            &scalars,
            &output_destinations,
        )?;

        // A structured evaluator may bind one scalar projection to several
        // source planes (most commonly the shared structural-zero projection).
        // Intern those occurrences on the cold path so duplicate descriptors
        // address one persistent broadcast plane.
        let mut slot_by_scalar = Vec::new();
        slot_by_scalar
            .try_reserve_exact(scalars.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate recurrence scalar slot map: {error}"
                ))
            })?;
        slot_by_scalar.resize(scalars.len(), None);
        let mut broadcast_scalar_by_slot = Vec::new();
        broadcast_scalar_by_slot
            .try_reserve_exact(scalars.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate recurrence broadcast scalar catalog: {error}"
                ))
            })?;
        let mut broadcast_slot_by_binding = Vec::new();
        broadcast_slot_by_binding
            .try_reserve_exact(parameter_bindings.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate recurrence broadcast binding map: {error}"
                ))
            })?;
        for binding in &parameter_bindings {
            let SymjitDirectParameterBinding::Broadcast { index } = *binding else {
                broadcast_slot_by_binding.push(None);
                continue;
            };
            let scalar_index = index as usize;
            let slot = if let Some(slot) = slot_by_scalar[scalar_index] {
                slot
            } else {
                let slot = u32::try_from(broadcast_scalar_by_slot.len()).map_err(|_| {
                    RusticolError::integrity(
                        "recurrence persistent broadcast plane count exceeds u32",
                    )
                })?;
                slot_by_scalar[scalar_index] = Some(slot);
                broadcast_scalar_by_slot.push(index);
                slot
            };
            broadcast_slot_by_binding.push(Some(slot));
        }
        let mut exact_factor_broadcast_slots = Vec::new();
        exact_factor_broadcast_slots
            .try_reserve_exact(broadcast_scalar_by_slot.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate recurrence exact-factor broadcast catalog: {error}"
                ))
            })?;
        for (slot, &scalar_index) in broadcast_scalar_by_slot.iter().enumerate() {
            match scalars.get(scalar_index as usize).ok_or_else(|| {
                RusticolError::integrity("recurrence scalar projection is out of bounds")
            })? {
                SymjitDirectScalarProjection::Literal(_)
                | SymjitDirectScalarProjection::Parameter { .. } => {}
                SymjitDirectScalarProjection::ExactFactor { .. } => {
                    exact_factor_broadcast_slots.push(slot);
                }
            }
        }

        Ok(Self {
            context: Box::new(SymjitDirectExecutorContext {
                role,
                kernel,
                parameter_bindings: parameter_bindings.into_boxed_slice(),
                input_planes: input_planes.into_boxed_slice(),
                scalars: scalars.into_boxed_slice(),
                output_destinations: output_destinations.into_boxed_slice(),
                broadcast_slot_by_binding: broadcast_slot_by_binding.into_boxed_slice(),
                broadcast_scalar_by_slot: broadcast_scalar_by_slot.into_boxed_slice(),
                exact_factor_broadcast_slots: exact_factor_broadcast_slots.into_boxed_slice(),
                scalar_workspace: RefCell::new(PlaneWorkspace::default()),
                tiled_workspace: RefCell::new(PlaneWorkspace::default()),
                internal_scratch_bytes: Cell::new(0),
                internal_broadcast_bytes: Cell::new(0),
                #[cfg(test)]
                invariant_broadcast_resolution_count: Cell::new(0),
                #[cfg(test)]
                exact_factor_broadcast_resolution_count: Cell::new(0),
            }),
        })
    }

    pub(crate) fn handle(&self) -> DirectExecutorHandle {
        let context = ptr::from_ref(self.context.as_ref()).cast();
        match self.context.role {
            DirectExecutorRole::Contribution => DirectExecutorHandle::Contribution {
                call: execute_contribution_rows as DirectContributionExecutor,
                context,
            },
            DirectExecutorRole::Finalization => DirectExecutorHandle::Finalization {
                call: execute_finalization_rows as DirectFinalizationExecutor,
                context,
            },
            DirectExecutorRole::Closure => DirectExecutorHandle::Closure {
                call: execute_closure_rows as DirectClosureExecutor,
                context,
            },
            DirectExecutorRole::Source => {
                unreachable!("source roles are rejected while loading a SymJIT plane executor")
            }
        }
    }

    /// Drop every cached table that borrows Direct-Arena row/storage pointers.
    pub(crate) fn invalidate_row_tables(&self) -> RusticolResult<()> {
        let mut scalar = self
            .context
            .scalar_workspace
            .try_borrow_mut()
            .map_err(|_| {
                RusticolError::internal(
                    "cannot invalidate recurrence SymJIT row tables during an active call",
                )
            })?;
        let mut tiled = self.context.tiled_workspace.try_borrow_mut().map_err(|_| {
            RusticolError::internal(
                "cannot invalidate recurrence SymJIT row tables during an active call",
            )
        })?;
        scalar.invalidate_row_tables();
        tiled.invalidate_row_tables();
        Ok(())
    }

    pub(crate) fn internal_traffic_bytes(&self) -> (u64, u64) {
        (
            self.context.internal_scratch_bytes.get(),
            self.context.internal_broadcast_bytes.get(),
        )
    }

    #[cfg(test)]
    fn simd_lane_width(&self) -> usize {
        self.context.kernel.simd_lanes()
    }

    #[cfg(test)]
    fn cached_direct_table_count(&self) -> usize {
        [
            &self.context.scalar_workspace,
            &self.context.tiled_workspace,
        ]
        .into_iter()
        .map(|workspace| {
            workspace
                .borrow()
                .row_groups
                .iter()
                .flat_map(|group| &group.rows)
                .filter(|row| row.direct.is_some())
                .count()
        })
        .sum()
    }

    #[cfg(test)]
    fn broadcast_plane_count(&self) -> usize {
        self.context.broadcast_scalar_by_slot.len()
    }

    #[cfg(test)]
    fn broadcast_resolution_counts(&self) -> (u64, u64) {
        (
            self.context.invariant_broadcast_resolution_count.get(),
            self.context.exact_factor_broadcast_resolution_count.get(),
        )
    }
}

fn validate_binding(
    kernel: &SymjitPlaneKernel,
    role: DirectExecutorRole,
    parameter_bindings: &[SymjitDirectParameterBinding],
    input_planes: &[SymjitDirectPlaneProjection],
    scalars: &[SymjitDirectScalarProjection],
    output_destinations: &[u32],
) -> RusticolResult<()> {
    if parameter_bindings.len() != kernel.input_plane_count()
        || output_destinations.len() != kernel.output_plane_count()
    {
        return Err(RusticolError::integrity(
            "recurrence plane binding shape does not match its SymJIT P-kernel",
        ));
    }
    if scalars.first() != Some(&SymjitDirectScalarProjection::ExactFactor { imaginary: false })
        || scalars.get(1) != Some(&SymjitDirectScalarProjection::ExactFactor { imaginary: true })
    {
        return Err(RusticolError::integrity(
            "recurrence scalar sources 0 and 1 must be the exact factor real and imaginary values",
        ));
    }
    for binding in parameter_bindings {
        match *binding {
            SymjitDirectParameterBinding::Plane { index }
                if index as usize >= input_planes.len() =>
            {
                return Err(RusticolError::integrity(
                    "recurrence P-kernel input plane projection is out of bounds",
                ));
            }
            SymjitDirectParameterBinding::Broadcast { index }
                if index as usize >= scalars.len() =>
            {
                return Err(RusticolError::integrity(
                    "recurrence P-kernel broadcast projection is out of bounds",
                ));
            }
            _ => {}
        }
    }
    for (output, &destination) in output_destinations.iter().enumerate() {
        let projection = input_planes
            .get(destination as usize)
            .copied()
            .ok_or_else(|| {
                RusticolError::integrity(
                    "recurrence P-kernel destination projection is out of bounds",
                )
            })?;
        let expected_imaginary = output % 2 == 1;
        let valid = match (role, projection) {
            (
                DirectExecutorRole::Contribution | DirectExecutorRole::Finalization,
                SymjitDirectPlaneProjection::DestinationCurrent { imaginary, .. },
            )
            | (
                DirectExecutorRole::Closure,
                SymjitDirectPlaneProjection::DestinationAmplitude { imaginary, .. },
            ) => imaginary == expected_imaginary,
            _ => false,
        };
        if !valid {
            return Err(RusticolError::integrity(
                "recurrence P-kernel output does not map to the expected split-complex destination",
            ));
        }
    }
    for pair in output_destinations.chunks_exact(2) {
        let real = input_planes[pair[0] as usize];
        let imaginary = input_planes[pair[1] as usize];
        if !matching_split_destination(real, imaginary) {
            return Err(RusticolError::integrity(
                "recurrence P-kernel output pair maps to different destination components",
            ));
        }
    }
    Ok(())
}

fn matching_split_destination(
    real: SymjitDirectPlaneProjection,
    imaginary: SymjitDirectPlaneProjection,
) -> bool {
    match (real, imaginary) {
        (
            SymjitDirectPlaneProjection::DestinationCurrent {
                component: left,
                imaginary: false,
            },
            SymjitDirectPlaneProjection::DestinationCurrent {
                component: right,
                imaginary: true,
            },
        )
        | (
            SymjitDirectPlaneProjection::DestinationAmplitude {
                component: left,
                imaginary: false,
            },
            SymjitDirectPlaneProjection::DestinationAmplitude {
                component: right,
                imaginary: true,
            },
        ) => left == right,
        _ => false,
    }
}

unsafe extern "C" fn execute_contribution_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    invoke_typed_rows(
        context,
        arena,
        momenta,
        parameters,
        factors,
        rows,
        row_count,
        point_count,
    )
}

unsafe extern "C" fn execute_finalization_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    invoke_typed_rows(
        context,
        arena,
        momenta,
        parameters,
        factors,
        rows,
        row_count,
        point_count,
    )
}

unsafe extern "C" fn execute_closure_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectClosureRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    invoke_typed_rows(
        context,
        arena,
        momenta,
        parameters,
        factors,
        rows,
        row_count,
        point_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn invoke_typed_rows<T: DirectTypedRow>(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const T,
    row_count: u32,
    point_count: u32,
) -> c_int {
    if context.is_null() {
        return STATUS_INVALID_CONTEXT;
    }
    if rows.is_null() || row_count == 0 || point_count == 0 {
        return STATUS_INVALID_ARGUMENT;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        let context = unsafe { &*context.cast::<SymjitDirectExecutorContext>() };
        if context.role != T::ROLE {
            return Ok(STATUS_ROLE_MISMATCH);
        }
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        context.execute_rows(arena, momenta, parameters, factors, rows, point_count)?;
        Ok(DIRECT_STATUS_OK)
    })) {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            record_direct_executor_error_detail(error);
            STATUS_EXECUTION_FAILED
        }
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "non-string panic payload".to_owned()
            };
            record_direct_executor_error_detail(RusticolError::internal(format!(
                "recurrence SymJIT executor panicked: {message}"
            )));
            STATUS_EXECUTION_FAILED
        }
    }
}

enum DirectRowRef<'a> {
    Contribution(&'a DirectContributionRow),
    Finalization(&'a DirectFinalizationRow),
    Closure(&'a DirectClosureRow),
}

trait DirectTypedRow {
    const ROLE: DirectExecutorRole;
    fn direct_row_ref(&self) -> DirectRowRef<'_>;
}

impl DirectTypedRow for DirectContributionRow {
    const ROLE: DirectExecutorRole = DirectExecutorRole::Contribution;

    fn direct_row_ref(&self) -> DirectRowRef<'_> {
        DirectRowRef::Contribution(self)
    }
}

impl DirectTypedRow for DirectFinalizationRow {
    const ROLE: DirectExecutorRole = DirectExecutorRole::Finalization;

    fn direct_row_ref(&self) -> DirectRowRef<'_> {
        DirectRowRef::Finalization(self)
    }
}

impl DirectTypedRow for DirectClosureRow {
    const ROLE: DirectExecutorRole = DirectExecutorRole::Closure;

    fn direct_row_ref(&self) -> DirectRowRef<'_> {
        DirectRowRef::Closure(self)
    }
}

impl DirectRowRef<'_> {
    fn exact_factor_id(&self) -> u32 {
        match self {
            Self::Contribution(row) => row.exact_factor_id,
            Self::Finalization(row) => row.exact_factor_id,
            Self::Closure(row) => row.exact_factor_id,
        }
    }

    fn parent_component_base(&self, parent: u8) -> Option<u32> {
        match (self, parent) {
            (Self::Contribution(row), 0) => Some(row.parent0_component_base),
            (Self::Contribution(row), 1)
                if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 =>
            {
                Some(row.parent1_component_base_or_sentinel)
            }
            (Self::Finalization(row), 0) => Some(row.component_base),
            (Self::Closure(row), 0) => Some(row.parent0_component_base),
            (Self::Closure(row), 1)
                if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 =>
            {
                Some(row.parent1_component_base_or_sentinel)
            }
            _ => None,
        }
    }

    fn momentum_form_id(&self, operand: u8) -> Option<u32> {
        match (self, operand) {
            (Self::Contribution(row), 0) => Some(row.parent0_momentum_form_id),
            (Self::Contribution(row), 1)
                if row.parent1_momentum_form_id_or_sentinel != DIRECT_NONE_U32 =>
            {
                Some(row.parent1_momentum_form_id_or_sentinel)
            }
            (Self::Finalization(row), 0) => Some(row.momentum_form_id),
            (Self::Closure(row), 0) => Some(row.parent0_momentum_form_id),
            (Self::Closure(row), 1)
                if row.parent1_momentum_form_id_or_sentinel != DIRECT_NONE_U32 =>
            {
                Some(row.parent1_momentum_form_id_or_sentinel)
            }
            _ => None,
        }
    }

    fn destination_current_base(&self) -> Option<u32> {
        match self {
            Self::Contribution(row) => Some(row.destination_component_base),
            Self::Finalization(row) => Some(row.component_base),
            Self::Closure(_) => None,
        }
    }

    fn destination_amplitude_base(&self) -> Option<u32> {
        match self {
            Self::Closure(row) => Some(row.amplitude_destination_id),
            _ => None,
        }
    }

    fn accumulates_output(&self) -> bool {
        match self {
            Self::Contribution(row) => {
                row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION == 0
            }
            Self::Finalization(_) => false,
            Self::Closure(_) => true,
        }
    }
}

impl SymjitDirectExecutorContext {
    fn execute_rows<T: DirectTypedRow>(
        &self,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        factors: DirectFactorView,
        rows: &[T],
        point_count: u32,
    ) -> RusticolResult<()> {
        if arena.point_stride == 0
            || point_count > arena.point_stride
            || arena.point_stride != momenta.point_stride
        {
            return Err(RusticolError::invalid_argument(
                "recurrence SymJIT point range or momentum stride is invalid",
            ));
        }
        let point_stride = arena.point_stride as usize;
        let broadcast_count = self.broadcast_scalar_by_slot.len();
        let workspace = if point_stride == 1 {
            &self.scalar_workspace
        } else {
            &self.tiled_workspace
        };
        let mut workspace = workspace.borrow_mut();
        workspace.prepare(
            point_stride,
            self.parameter_bindings.len(),
            self.output_destinations.len(),
            broadcast_count,
        )?;

        let storage = DescriptorStorageIdentity::new(arena, momenta);
        if workspace.storage.is_some_and(|cached| cached != storage) {
            workspace.invalidate_row_tables();
        }
        workspace.storage = Some(storage);
        let row_count = u32::try_from(rows.len())
            .map_err(|_| RusticolError::integrity("recurrence row count exceeds u32"))?;
        let identity = RowTableIdentity {
            role: T::ROLE,
            address: rows.as_ptr().addr(),
            row_count,
        };
        let group_index = if let Some(index) = workspace.cached_row_group_index(identity) {
            index
        } else {
            let cached =
                self.build_cached_row_group(&mut workspace, identity, rows, arena, momenta)?;
            workspace.cache_row_group(cached)?
        };

        let mut call_scratch_bytes = 0_u64;
        let mut call_broadcast_bytes = 0_u64;
        let result = (|| {
            for (row_index, typed_row) in rows.iter().enumerate() {
                let row = typed_row.direct_row_ref();
                let refresh_bytes = if row_index == 0 {
                    // Preserve scalar-catalog order on the first row while
                    // refreshing every literal/model parameter exactly once
                    // for this executor call. Only exact factors can vary on
                    // later rows.
                    self.refresh_first_row_broadcasts(&mut workspace, &row, parameters, factors)?
                } else {
                    self.refresh_exact_factor_broadcasts(&mut workspace, &row, parameters, factors)?
                };
                call_broadcast_bytes = call_broadcast_bytes.saturating_add(refresh_bytes);
                let (factor_re, factor_im) = exact_factor(&row, factors)?;
                let use_direct = is_identity_factor(factor_re, factor_im)
                    && workspace
                        .row_groups
                        .get(group_index)
                        .and_then(|group| group.rows.get(row_index))
                        .is_some_and(|tables| tables.direct.is_some());
                {
                    let tables = workspace
                        .row_groups
                        .get_mut(group_index)
                        .and_then(|group| group.rows.get_mut(row_index))
                        .ok_or_else(|| {
                            RusticolError::internal(
                                "recurrence cached descriptor table is out of bounds",
                            )
                        })?;
                    let table = if use_direct {
                        tables.direct.as_mut().ok_or_else(|| {
                            RusticolError::internal(
                                "recurrence direct-output table disappeared after selection",
                            )
                        })?
                    } else {
                        &mut tables.scratch
                    };
                    self.kernel.execute_table(table, 0, point_count as usize)?;
                }
                if !use_direct {
                    apply_outputs(
                        &self.input_planes,
                        &self.output_destinations,
                        &workspace.scratch,
                        point_stride,
                        point_count as usize,
                        factor_re,
                        factor_im,
                        &row,
                        arena,
                        momenta,
                    )?;
                    let scratch_one_way = internal_plane_bytes(
                        self.output_destinations.len(),
                        point_count as usize,
                        "recurrence scratch traffic",
                    )?;
                    call_scratch_bytes =
                        call_scratch_bytes.saturating_add(scratch_one_way.saturating_mul(2));
                }
                call_broadcast_bytes = call_broadcast_bytes.saturating_add(internal_plane_bytes(
                    broadcast_count,
                    point_count as usize,
                    "recurrence broadcast traffic",
                )?);
            }
            Ok(())
        })();
        self.record_internal_scratch_bytes(call_scratch_bytes);
        self.record_internal_broadcast_bytes(call_broadcast_bytes);
        result
    }

    fn build_cached_row_group<T: DirectTypedRow>(
        &self,
        workspace: &mut PlaneWorkspace,
        identity: RowTableIdentity,
        rows: &[T],
        arena: DirectArenaView,
        momenta: DirectMomentumView,
    ) -> RusticolResult<CachedRowGroup> {
        let descriptor_count = self
            .parameter_bindings
            .len()
            .checked_add(self.output_destinations.len())
            .ok_or_else(|| RusticolError::integrity("recurrence descriptor count overflows"))?;
        let bytes_per_table = descriptor_count
            .checked_mul(size_of::<PlaneDescriptor<'static>>())
            .ok_or_else(|| {
                RusticolError::compatibility("recurrence descriptor cache size overflows")
            })?;
        let mut descriptor_bytes = 0_usize;
        let mut cached_rows: Vec<CachedRowTables> = Vec::new();
        cached_rows.try_reserve_exact(rows.len()).map_err(|_| {
            RusticolError::compatibility("could not reserve recurrence descriptor tables")
        })?;
        for typed_row in rows {
            let row = typed_row.direct_row_ref();
            let mut input_pointers = Vec::new();
            input_pointers
                .try_reserve_exact(self.parameter_bindings.len())
                .map_err(|_| {
                    RusticolError::compatibility(
                        "could not reserve recurrence input plane pointers",
                    )
                })?;
            for (binding_index, binding) in self.parameter_bindings.iter().enumerate() {
                let pointer = match *binding {
                    SymjitDirectParameterBinding::Plane { index } => resolve_plane_pointer(
                        *self.input_planes.get(index as usize).ok_or_else(|| {
                            RusticolError::integrity("recurrence input projection is out of bounds")
                        })?,
                        &row,
                        arena,
                        momenta,
                    )?,
                    SymjitDirectParameterBinding::Broadcast { .. } => {
                        let slot =
                            self.broadcast_slot_by_binding[binding_index].ok_or_else(|| {
                                RusticolError::internal(
                                    "recurrence broadcast binding has no persistent slot",
                                )
                            })? as usize;
                        workspace.broadcast_plane_pointer(slot)?
                    }
                };
                input_pointers.push(pointer);
            }
            let mut output_pointers = Vec::new();
            output_pointers
                .try_reserve_exact(self.output_destinations.len())
                .map_err(|_| {
                    RusticolError::compatibility(
                        "could not reserve recurrence output plane pointers",
                    )
                })?;
            for &destination in self.output_destinations.iter() {
                output_pointers.push(resolve_plane_pointer(
                    *self.input_planes.get(destination as usize).ok_or_else(|| {
                        RusticolError::integrity("recurrence output projection is out of bounds")
                    })?,
                    &row,
                    arena,
                    momenta,
                )?);
            }
            let mut scratch_pointers = Vec::new();
            scratch_pointers
                .try_reserve_exact(self.output_destinations.len())
                .map_err(|_| {
                    RusticolError::compatibility(
                        "could not reserve recurrence scratch plane pointers",
                    )
                })?;
            for output in 0..self.output_destinations.len() {
                scratch_pointers.push(workspace.scratch_pointer(output)?);
            }
            let scratch = build_plane_table(
                &self.kernel,
                &input_pointers,
                &scratch_pointers,
                workspace.point_stride,
            )?;
            descriptor_bytes = descriptor_bytes
                .checked_add(bytes_per_table)
                .ok_or_else(|| {
                    RusticolError::compatibility("recurrence descriptor cache size overflows")
                })?;
            let direct = if direct_outputs_are_structurally_safe(
                self.role,
                &input_pointers,
                &output_pointers,
                workspace.point_stride,
            )? {
                descriptor_bytes =
                    descriptor_bytes
                        .checked_add(bytes_per_table)
                        .ok_or_else(|| {
                            RusticolError::compatibility(
                                "recurrence descriptor cache size overflows",
                            )
                        })?;
                Some(build_plane_table(
                    &self.kernel,
                    &input_pointers,
                    &output_pointers,
                    workspace.point_stride,
                )?)
            } else {
                None
            };
            cached_rows.push(CachedRowTables { scratch, direct });
        }
        Ok(CachedRowGroup {
            identity,
            rows: cached_rows,
            descriptor_bytes,
        })
    }

    fn refresh_first_row_broadcasts(
        &self,
        workspace: &mut PlaneWorkspace,
        row: &DirectRowRef<'_>,
        parameters: DirectParameterView,
        factors: DirectFactorView,
    ) -> RusticolResult<u64> {
        let mut refresh_bytes = 0_u64;
        for (slot, &index) in self.broadcast_scalar_by_slot.iter().enumerate() {
            let projection = self.scalars.get(index as usize).ok_or_else(|| {
                RusticolError::integrity("recurrence scalar projection is out of bounds")
            })?;
            #[cfg(test)]
            match projection {
                SymjitDirectScalarProjection::Literal(_)
                | SymjitDirectScalarProjection::Parameter { .. } => {
                    self.invariant_broadcast_resolution_count.set(
                        self.invariant_broadcast_resolution_count
                            .get()
                            .saturating_add(1),
                    )
                }
                SymjitDirectScalarProjection::ExactFactor { .. } => {
                    self.exact_factor_broadcast_resolution_count.set(
                        self.exact_factor_broadcast_resolution_count
                            .get()
                            .saturating_add(1),
                    )
                }
            }
            let value = resolve_scalar_value(projection, row, parameters, factors)?;
            let (_, refreshed) = workspace.broadcast_pointer(slot, value)?;
            if refreshed {
                refresh_bytes = refresh_bytes.saturating_add(internal_plane_bytes(
                    1,
                    workspace.point_stride,
                    "recurrence broadcast refresh traffic",
                )?);
            }
        }
        Ok(refresh_bytes)
    }

    fn refresh_exact_factor_broadcasts(
        &self,
        workspace: &mut PlaneWorkspace,
        row: &DirectRowRef<'_>,
        parameters: DirectParameterView,
        factors: DirectFactorView,
    ) -> RusticolResult<u64> {
        let mut refresh_bytes = 0_u64;
        for &slot in self.exact_factor_broadcast_slots.iter() {
            let scalar_index = *self.broadcast_scalar_by_slot.get(slot).ok_or_else(|| {
                RusticolError::internal(
                    "recurrence exact-factor broadcast slot is outside its catalog",
                )
            })?;
            let projection = self.scalars.get(scalar_index as usize).ok_or_else(|| {
                RusticolError::integrity("recurrence scalar projection is out of bounds")
            })?;
            let SymjitDirectScalarProjection::ExactFactor { .. } = projection else {
                return Err(RusticolError::internal(
                    "recurrence row-local broadcast is not an exact factor",
                ));
            };
            #[cfg(test)]
            self.exact_factor_broadcast_resolution_count.set(
                self.exact_factor_broadcast_resolution_count
                    .get()
                    .saturating_add(1),
            );
            let value = resolve_scalar_value(projection, row, parameters, factors)?;
            let (_, refreshed) = workspace.broadcast_pointer(slot, value)?;
            if refreshed {
                refresh_bytes = refresh_bytes.saturating_add(internal_plane_bytes(
                    1,
                    workspace.point_stride,
                    "recurrence broadcast refresh traffic",
                )?);
            }
        }
        Ok(refresh_bytes)
    }

    fn record_internal_scratch_bytes(&self, bytes: u64) {
        self.internal_scratch_bytes
            .set(self.internal_scratch_bytes.get().saturating_add(bytes));
    }

    fn record_internal_broadcast_bytes(&self, bytes: u64) {
        self.internal_broadcast_bytes
            .set(self.internal_broadcast_bytes.get().saturating_add(bytes));
    }
}

fn checked_descriptor_cache_bytes(current: usize, additional: usize) -> RusticolResult<usize> {
    current.checked_add(additional).ok_or_else(|| {
        RusticolError::compatibility("recurrence SymJIT descriptor cache size overflows")
    })
}

fn build_plane_table(
    kernel: &SymjitPlaneKernel,
    input_pointers: &[*mut f64],
    output_pointers: &[*mut f64],
    point_stride: usize,
) -> RusticolResult<SymjitPlaneTable<'static>> {
    let descriptor_count = input_pointers
        .len()
        .checked_add(output_pointers.len())
        .ok_or_else(|| RusticolError::integrity("recurrence descriptor count overflows"))?;
    let mut descriptors: Vec<PlaneDescriptor<'static>> = Vec::new();
    descriptors
        .try_reserve_exact(descriptor_count)
        .map_err(|_| {
            RusticolError::compatibility("could not reserve recurrence plane descriptors")
        })?;
    for &pointer in input_pointers.iter().chain(output_pointers) {
        descriptors.push(unsafe { PlaneDescriptor::from_cached_raw_parts(pointer, point_stride)? });
    }
    kernel.build_raw_table(descriptors)
}

fn direct_outputs_are_structurally_safe(
    role: DirectExecutorRole,
    input_pointers: &[*mut f64],
    output_pointers: &[*mut f64],
    point_stride: usize,
) -> RusticolResult<bool> {
    if role != DirectExecutorRole::Finalization {
        return Ok(false);
    }
    for (output_index, &pointer) in output_pointers.iter().enumerate() {
        let output = PlaneAddressRange::new(pointer, point_stride)?;
        for &input in input_pointers {
            if output.overlaps(PlaneAddressRange::new(input, point_stride)?) {
                return Ok(false);
            }
        }
        for &prior in &output_pointers[..output_index] {
            if output.overlaps(PlaneAddressRange::new(prior, point_stride)?) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[derive(Clone, Copy)]
struct PlaneAddressRange {
    start: usize,
    end: usize,
}

impl PlaneAddressRange {
    fn new(pointer: *mut f64, point_stride: usize) -> RusticolResult<Self> {
        if pointer.is_null() || point_stride == 0 {
            return Err(RusticolError::integrity(
                "recurrence direct-output plane has a null pointer or zero stride",
            ));
        }
        let bytes = point_stride
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| RusticolError::integrity("recurrence plane byte range overflows"))?;
        let start = pointer.addr();
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| RusticolError::integrity("recurrence plane address range overflows"))?;
        Ok(Self { start, end })
    }

    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

fn is_identity_factor(real: f64, imaginary: f64) -> bool {
    real.to_bits() == 1.0_f64.to_bits() && imaginary.to_bits() == 0.0_f64.to_bits()
}

fn internal_plane_bytes(
    plane_count: usize,
    point_count: usize,
    description: &str,
) -> RusticolResult<u64> {
    let bytes = plane_count
        .checked_mul(point_count)
        .and_then(|count| count.checked_mul(size_of::<f64>()))
        .ok_or_else(|| RusticolError::integrity(format!("{description} overflows usize")))?;
    u64::try_from(bytes).map_err(|_| RusticolError::integrity(format!("{description} exceeds u64")))
}

#[allow(clippy::too_many_arguments)]
fn apply_outputs(
    projections: &[SymjitDirectPlaneProjection],
    output_destinations: &[u32],
    scratch: &[f64],
    point_stride: usize,
    point_count: usize,
    factor_re: f64,
    factor_im: f64,
    row: &DirectRowRef<'_>,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
) -> RusticolResult<()> {
    for (component, destination_pair) in output_destinations.chunks_exact(2).enumerate() {
        let scratch_re_start = component
            .checked_mul(2)
            .and_then(|plane| plane.checked_mul(point_stride))
            .ok_or_else(|| RusticolError::internal("recurrence scratch range overflows"))?;
        let scratch_im_start = scratch_re_start
            .checked_add(point_stride)
            .ok_or_else(|| RusticolError::internal("recurrence scratch range overflows"))?;
        let scratch_re = scratch
            .get(scratch_re_start..scratch_re_start + point_count)
            .ok_or_else(|| RusticolError::internal("recurrence real scratch is out of bounds"))?;
        let scratch_im = scratch
            .get(scratch_im_start..scratch_im_start + point_count)
            .ok_or_else(|| {
                RusticolError::internal("recurrence imaginary scratch is out of bounds")
            })?;
        let destination_re = resolve_plane_pointer(
            projections[destination_pair[0] as usize],
            row,
            arena,
            momenta,
        )?;
        let destination_im = resolve_plane_pointer(
            projections[destination_pair[1] as usize],
            row,
            arena,
            momenta,
        )?;
        let destination_re = unsafe { std::slice::from_raw_parts_mut(destination_re, point_count) };
        let destination_im = unsafe { std::slice::from_raw_parts_mut(destination_im, point_count) };
        apply_complex_scaled(
            destination_re,
            destination_im,
            scratch_re,
            scratch_im,
            factor_re,
            factor_im,
            row.accumulates_output(),
        );
    }
    Ok(())
}

#[inline]
fn apply_complex_scaled(
    destination_re: &mut [f64],
    destination_im: &mut [f64],
    source_re: &[f64],
    source_im: &[f64],
    factor_re: f64,
    factor_im: f64,
    accumulate: bool,
) {
    debug_assert_eq!(destination_re.len(), destination_im.len());
    debug_assert_eq!(destination_re.len(), source_re.len());
    debug_assert_eq!(destination_re.len(), source_im.len());
    #[cfg(target_arch = "x86_64")]
    let mut point = apply_complex_scaled_x86_64(
        destination_re,
        destination_im,
        source_re,
        source_im,
        factor_re,
        factor_im,
        accumulate,
    );
    #[cfg(target_arch = "aarch64")]
    let mut point = apply_complex_scaled_aarch64(
        destination_re,
        destination_im,
        source_re,
        source_im,
        factor_re,
        factor_im,
        accumulate,
    );
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let mut point = 0;
    if accumulate {
        while point < destination_re.len() {
            let re = source_re[point];
            let im = source_im[point];
            destination_re[point] += re * factor_re - im * factor_im;
            destination_im[point] += re * factor_im + im * factor_re;
            point += 1;
        }
    } else {
        while point < destination_re.len() {
            let re = source_re[point];
            let im = source_im[point];
            destination_re[point] = re * factor_re - im * factor_im;
            destination_im[point] = re * factor_im + im * factor_re;
            point += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_complex_scaled_x86_64(
    destination_re: &mut [f64],
    destination_im: &mut [f64],
    source_re: &[f64],
    source_im: &[f64],
    factor_re: f64,
    factor_im: f64,
    accumulate: bool,
) -> usize {
    let factor_re = f64x4::new([factor_re; 4]);
    let factor_im = f64x4::new([factor_im; 4]);
    let mut point = 0;
    while destination_re.len() - point >= 4 {
        let re = f64x4::new([
            source_re[point],
            source_re[point + 1],
            source_re[point + 2],
            source_re[point + 3],
        ]);
        let im = f64x4::new([
            source_im[point],
            source_im[point + 1],
            source_im[point + 2],
            source_im[point + 3],
        ]);
        let mut scaled_re = re * factor_re - im * factor_im;
        let mut scaled_im = re * factor_im + im * factor_re;
        if accumulate {
            scaled_re += f64x4::new([
                destination_re[point],
                destination_re[point + 1],
                destination_re[point + 2],
                destination_re[point + 3],
            ]);
            scaled_im += f64x4::new([
                destination_im[point],
                destination_im[point + 1],
                destination_im[point + 2],
                destination_im[point + 3],
            ]);
        }
        destination_re[point..point + 4].copy_from_slice(&scaled_re.to_array());
        destination_im[point..point + 4].copy_from_slice(&scaled_im.to_array());
        point += 4;
    }
    point
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_complex_scaled_aarch64(
    destination_re: &mut [f64],
    destination_im: &mut [f64],
    source_re: &[f64],
    source_im: &[f64],
    factor_re: f64,
    factor_im: f64,
    accumulate: bool,
) -> usize {
    let factor_re = f64x2::new([factor_re; 2]);
    let factor_im = f64x2::new([factor_im; 2]);
    let mut point = 0;
    while destination_re.len() - point >= 2 {
        let re = f64x2::new([source_re[point], source_re[point + 1]]);
        let im = f64x2::new([source_im[point], source_im[point + 1]]);
        let mut scaled_re = re * factor_re - im * factor_im;
        let mut scaled_im = re * factor_im + im * factor_re;
        if accumulate {
            scaled_re += f64x2::new([destination_re[point], destination_re[point + 1]]);
            scaled_im += f64x2::new([destination_im[point], destination_im[point + 1]]);
        }
        destination_re[point..point + 2].copy_from_slice(&scaled_re.to_array());
        destination_im[point..point + 2].copy_from_slice(&scaled_im.to_array());
        point += 2;
    }
    point
}

fn exact_factor(row: &DirectRowRef<'_>, factors: DirectFactorView) -> RusticolResult<(f64, f64)> {
    let index = row.exact_factor_id();
    if index >= factors.value_count || factors.values_re.is_null() || factors.values_im.is_null() {
        return Err(RusticolError::integrity(
            "recurrence exact factor is outside its split-complex catalog",
        ));
    }
    Ok(unsafe {
        (
            *factors.values_re.add(index as usize),
            *factors.values_im.add(index as usize),
        )
    })
}

fn resolve_scalar_value(
    projection: &SymjitDirectScalarProjection,
    row: &DirectRowRef<'_>,
    parameters: DirectParameterView,
    factors: DirectFactorView,
) -> RusticolResult<f64> {
    match *projection {
        SymjitDirectScalarProjection::Literal(value) => Ok(value),
        SymjitDirectScalarProjection::Parameter { index, imaginary } => {
            if index >= parameters.value_count {
                return Err(RusticolError::integrity(
                    "recurrence model parameter is outside its split-complex catalog",
                ));
            }
            let base = if imaginary {
                parameters.values_im
            } else {
                parameters.values_re
            };
            if base.is_null() {
                return Err(RusticolError::integrity(
                    "recurrence model parameter catalog has a null pointer",
                ));
            }
            Ok(unsafe { *base.add(index as usize) })
        }
        SymjitDirectScalarProjection::ExactFactor { imaginary } => {
            let (real, imag) = exact_factor(row, factors)?;
            Ok(if imaginary { imag } else { real })
        }
    }
}

fn resolve_plane_pointer(
    projection: SymjitDirectPlaneProjection,
    row: &DirectRowRef<'_>,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
) -> RusticolResult<*mut f64> {
    let pointer = match projection {
        SymjitDirectPlaneProjection::ParentCurrent {
            parent,
            component,
            imaginary,
        } => {
            let base = row.parent_component_base(parent).ok_or_else(|| {
                RusticolError::integrity("recurrence row has no requested parent")
            })?;
            arena_plane_pointer(
                if imaginary {
                    arena.current_im
                } else {
                    arena.current_re
                },
                arena.current_scalar_len,
                u64::from(base) + u64::from(component),
                arena.point_stride,
            )
        }
        SymjitDirectPlaneProjection::Momentum {
            operand,
            lorentz_component,
        } => {
            let form = row.momentum_form_id(operand).ok_or_else(|| {
                RusticolError::integrity("recurrence row has no requested momentum operand")
            })?;
            if form >= momenta.form_count || lorentz_component >= momenta.lorentz_component_count {
                return Err(RusticolError::integrity(
                    "recurrence momentum projection is out of bounds",
                ));
            }
            let plane = u64::from(form)
                .checked_mul(u64::from(momenta.lorentz_component_count))
                .and_then(|base| base.checked_add(u64::from(lorentz_component)))
                .ok_or_else(|| RusticolError::integrity("momentum plane index overflows"))?;
            arena_plane_pointer(
                momenta.values.cast_mut(),
                momenta.scalar_len,
                plane,
                momenta.point_stride,
            )
        }
        SymjitDirectPlaneProjection::DestinationCurrent {
            component,
            imaginary,
        } => {
            let base = row.destination_current_base().ok_or_else(|| {
                RusticolError::integrity("recurrence row has no current destination")
            })?;
            arena_plane_pointer(
                if imaginary {
                    arena.current_im
                } else {
                    arena.current_re
                },
                arena.current_scalar_len,
                u64::from(base) + u64::from(component),
                arena.point_stride,
            )
        }
        SymjitDirectPlaneProjection::DestinationAmplitude {
            component,
            imaginary,
        } => {
            let base = row.destination_amplitude_base().ok_or_else(|| {
                RusticolError::integrity("recurrence row has no amplitude destination")
            })?;
            arena_plane_pointer(
                if imaginary {
                    arena.amplitude_im
                } else {
                    arena.amplitude_re
                },
                arena.amplitude_scalar_len,
                u64::from(base) + u64::from(component),
                arena.point_stride,
            )
        }
    }?;
    Ok(pointer)
}

fn arena_plane_pointer(
    pointer: *mut f64,
    scalar_len: u64,
    plane: u64,
    point_stride: u32,
) -> RusticolResult<*mut f64> {
    if pointer.is_null() || point_stride == 0 {
        return Err(RusticolError::integrity(
            "recurrence plane storage has a null pointer or zero stride",
        ));
    }
    let offset = plane
        .checked_mul(u64::from(point_stride))
        .ok_or_else(|| RusticolError::integrity("recurrence plane offset overflows"))?;
    let end = offset
        .checked_add(u64::from(point_stride))
        .ok_or_else(|| RusticolError::integrity("recurrence plane range overflows"))?;
    if end > scalar_len {
        return Err(RusticolError::integrity(
            "recurrence plane projection is outside its arena",
        ));
    }
    let offset = usize::try_from(offset)
        .map_err(|_| RusticolError::integrity("recurrence plane offset exceeds usize"))?;
    Ok(unsafe { pointer.add(offset) })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::symjit_plane::compile_symbolica_program_to_plane_application_bytes;
    use super::*;
    use crate::recurrence::DirectSourceRow;
    use crate::recurrence::on_the_fly::{
        OnTheFlyExecutorKeyV1, OnTheFlyPreparedExecutorResolver, OnTheFlyStructuralInterpreter,
        OnTheFlyStructuralTraceV1, OnTheFlyWorkspaceV1, ResolvedOnTheFlyExecutor,
    };
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::collections::BTreeMap;

    thread_local! {
        static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
        static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
    }

    struct CountingAllocator;

    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count_allocation(layout.size());
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count_allocation(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            count_allocation(new_size);
            unsafe { System.realloc(pointer, layout, new_size) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }
    }

    fn count_allocation(bytes: usize) {
        let tracking = TRACK_ALLOCATIONS.try_with(Cell::get).unwrap_or(false);
        if tracking {
            let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
            let _ = ALLOCATED_BYTES.try_with(|total| total.set(total.get().saturating_add(bytes)));
        }
    }

    pub(crate) fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
        ALLOCATION_COUNT.with(|count| count.set(0));
        ALLOCATED_BYTES.with(|total| total.set(0));
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
        let result = operation();
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        let count = ALLOCATION_COUNT.with(Cell::get);
        let bytes = ALLOCATED_BYTES.with(Cell::get);
        (result, count, bytes)
    }

    const IDENTITY_PROGRAM: &str = "([('assign', ('out', 0), ('param', 0))], 1, [])";
    const DOUBLE_PROGRAM: &str = "
([('add', ('temp', 0), [('param', 0), ('param', 1)], 0),
  ('assign', ('out', 0), ('temp', 0))],
 2,
 [])
";
    const LENGTHS: &[usize] = &[1, 2, 3, 7, 8, 127, 128, 129, 1023, 1024, 1025];

    fn identity_executor(role: DirectExecutorRole) -> LoadedSymjitDirectExecutor {
        try_identity_executor(role, 2, false).unwrap()
    }

    fn try_identity_executor(
        role: DirectExecutorRole,
        expected_optimization_level: u32,
        expected_compression: bool,
    ) -> RusticolResult<LoadedSymjitDirectExecutor> {
        try_identity_executor_with_abi(
            role,
            expected_optimization_level,
            expected_compression,
            SYMJIT_RECURRENCE_PLANE_APPLICATION_ABI,
        )
    }

    fn try_identity_executor_with_abi(
        role: DirectExecutorRole,
        expected_optimization_level: u32,
        expected_compression: bool,
        application_abi: &str,
    ) -> RusticolResult<LoadedSymjitDirectExecutor> {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(IDENTITY_PROGRAM, 1, 1, 2, false)
                .unwrap();
        let destination = match role {
            DirectExecutorRole::Closure => [
                SymjitDirectPlaneProjection::DestinationAmplitude {
                    component: 0,
                    imaginary: false,
                },
                SymjitDirectPlaneProjection::DestinationAmplitude {
                    component: 0,
                    imaginary: true,
                },
            ],
            _ => [
                SymjitDirectPlaneProjection::DestinationCurrent {
                    component: 0,
                    imaginary: false,
                },
                SymjitDirectPlaneProjection::DestinationCurrent {
                    component: 0,
                    imaginary: true,
                },
            ],
        };
        LoadedSymjitDirectExecutor::load_prepared_application_bytes(
            &bytes,
            PathBuf::from("identity-recurrence-plane.symjit"),
            application_abi,
            expected_optimization_level,
            expected_compression,
            role,
            vec![
                SymjitDirectParameterBinding::Plane { index: 0 },
                SymjitDirectParameterBinding::Plane { index: 1 },
            ],
            vec![
                SymjitDirectPlaneProjection::ParentCurrent {
                    parent: 0,
                    component: 0,
                    imaginary: false,
                },
                SymjitDirectPlaneProjection::ParentCurrent {
                    parent: 0,
                    component: 0,
                    imaginary: true,
                },
                destination[0],
                destination[1],
            ],
            vec![
                SymjitDirectScalarProjection::ExactFactor { imaginary: false },
                SymjitDirectScalarProjection::ExactFactor { imaginary: true },
            ],
            vec![2, 3],
        )
    }

    #[test]
    fn predecessor_plane_application_abi_fails_closed_with_regeneration_message() {
        let error = try_identity_executor_with_abi(
            DirectExecutorRole::Contribution,
            2,
            false,
            "pyamplicol-symjit-plane-application-v1",
        )
        .err()
        .expect("the predecessor recurrence plane-application ABI must fail closed");
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn authenticated_template_optimization_level_mismatch_fails_closed() {
        let error = try_identity_executor(DirectExecutorRole::Contribution, 3, false)
            .err()
            .expect("a template/application optimization mismatch must fail closed");
        assert!(error.message().contains("optimization level"));
        assert!(error.message().contains("authenticated template"));
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn authenticated_manifest_compression_mismatch_fails_closed() {
        let error = try_identity_executor(DirectExecutorRole::Contribution, 2, true)
            .err()
            .expect("a manifest/application compression mismatch must fail closed");
        assert!(error.message().contains("compression"));
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn writer_valid_large_plane_and_scalar_bindings_are_not_artificially_capped() {
        const INPUT_COMPLEX_COUNT: usize = 257;
        let program =
            format!("([('assign', ('out', 0), ('param', 0))], {INPUT_COMPLEX_COUNT}, [])");
        let bytes = compile_symbolica_program_to_plane_application_bytes(
            &program,
            INPUT_COMPLEX_COUNT,
            1,
            2,
            false,
        )
        .unwrap();
        let mut parameter_bindings = Vec::new();
        let mut input_planes = Vec::new();
        for index in 0..(2 * INPUT_COMPLEX_COUNT) {
            parameter_bindings.push(SymjitDirectParameterBinding::Plane {
                index: index as u32,
            });
            input_planes.push(SymjitDirectPlaneProjection::ParentCurrent {
                parent: 0,
                component: 0,
                imaginary: index % 2 == 1,
            });
        }
        input_planes.extend([
            SymjitDirectPlaneProjection::DestinationCurrent {
                component: 0,
                imaginary: false,
            },
            SymjitDirectPlaneProjection::DestinationCurrent {
                component: 0,
                imaginary: true,
            },
        ]);
        let mut scalars = vec![
            SymjitDirectScalarProjection::ExactFactor { imaginary: false },
            SymjitDirectScalarProjection::ExactFactor { imaginary: true },
        ];
        scalars.resize(257, SymjitDirectScalarProjection::Literal(0.0));

        LoadedSymjitDirectExecutor::load_prepared_application_bytes(
            &bytes,
            PathBuf::from("large-recurrence-plane.symjit"),
            SYMJIT_RECURRENCE_PLANE_APPLICATION_ABI,
            2,
            false,
            DirectExecutorRole::Contribution,
            parameter_bindings,
            input_planes,
            scalars,
            vec![514, 515],
        )
        .unwrap();
    }

    #[test]
    fn descriptor_cache_accounting_accepts_the_observed_303_megabyte_geometry() {
        assert_eq!(
            checked_descriptor_cache_bytes(153_624_576, 149_667_840).unwrap(),
            303_292_416
        );
        assert!(checked_descriptor_cache_bytes(usize::MAX, 1).is_err());
    }

    #[test]
    fn callback_failure_reaches_the_direct_backend_with_concrete_context() {
        let loaded = identity_executor(DirectExecutorRole::Contribution);
        let DirectExecutorHandle::Contribution { call, context } = loaded.handle() else {
            unreachable!()
        };
        let mut current_re = vec![1.0; 2];
        let mut current_im = vec![0.0; 2];
        let mut amplitude_re = vec![0.0; 1];
        let mut amplitude_im = vec![0.0; 1];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            &mut amplitude_re,
            &mut amplitude_im,
            &[],
            &[],
            &[],
            &[],
            1,
        );
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 1,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };

        crate::recurrence::direct_backend::clear_direct_executor_error_detail();
        let status = unsafe { call(context, arena, momenta, parameters, factors, &row, 1, 1) };
        let error = crate::recurrence::direct_backend::check_status(
            DirectExecutorRole::Contribution,
            312,
            status,
        )
        .unwrap_err();

        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
        assert!(error.message().contains("Contribution executor 312 failed"));
        assert!(error.message().contains("exact factor"));
    }

    fn parameter_broadcast_executor(role: DirectExecutorRole) -> LoadedSymjitDirectExecutor {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(IDENTITY_PROGRAM, 1, 1, 2, false)
                .unwrap();
        let destination = match role {
            DirectExecutorRole::Closure => [
                SymjitDirectPlaneProjection::DestinationAmplitude {
                    component: 0,
                    imaginary: false,
                },
                SymjitDirectPlaneProjection::DestinationAmplitude {
                    component: 0,
                    imaginary: true,
                },
            ],
            _ => [
                SymjitDirectPlaneProjection::DestinationCurrent {
                    component: 0,
                    imaginary: false,
                },
                SymjitDirectPlaneProjection::DestinationCurrent {
                    component: 0,
                    imaginary: true,
                },
            ],
        };
        LoadedSymjitDirectExecutor::load_prepared_application_bytes(
            &bytes,
            PathBuf::from("parameter-recurrence-plane.symjit"),
            SYMJIT_RECURRENCE_PLANE_APPLICATION_ABI,
            2,
            false,
            role,
            vec![
                SymjitDirectParameterBinding::Broadcast { index: 2 },
                SymjitDirectParameterBinding::Broadcast { index: 3 },
            ],
            vec![destination[0], destination[1]],
            vec![
                SymjitDirectScalarProjection::ExactFactor { imaginary: false },
                SymjitDirectScalarProjection::ExactFactor { imaginary: true },
                SymjitDirectScalarProjection::Parameter {
                    index: 0,
                    imaginary: false,
                },
                SymjitDirectScalarProjection::Parameter {
                    index: 0,
                    imaginary: true,
                },
            ],
            vec![0, 1],
        )
        .unwrap()
    }

    fn repeated_parameter_and_zero_broadcast_executor() -> LoadedSymjitDirectExecutor {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(DOUBLE_PROGRAM, 2, 1, 2, false)
                .unwrap();
        LoadedSymjitDirectExecutor::load_prepared_application_bytes(
            &bytes,
            PathBuf::from("shared-recurrence-broadcasts.symjit"),
            SYMJIT_RECURRENCE_PLANE_APPLICATION_ABI,
            2,
            false,
            DirectExecutorRole::Contribution,
            vec![
                SymjitDirectParameterBinding::Broadcast { index: 2 },
                SymjitDirectParameterBinding::Broadcast { index: 3 },
                SymjitDirectParameterBinding::Broadcast { index: 2 },
                SymjitDirectParameterBinding::Broadcast { index: 3 },
            ],
            vec![
                SymjitDirectPlaneProjection::DestinationCurrent {
                    component: 0,
                    imaginary: false,
                },
                SymjitDirectPlaneProjection::DestinationCurrent {
                    component: 0,
                    imaginary: true,
                },
            ],
            vec![
                SymjitDirectScalarProjection::ExactFactor { imaginary: false },
                SymjitDirectScalarProjection::ExactFactor { imaginary: true },
                SymjitDirectScalarProjection::Parameter {
                    index: 0,
                    imaginary: false,
                },
                SymjitDirectScalarProjection::Literal(0.0),
            ],
            vec![0, 1],
        )
        .unwrap()
    }

    fn mixed_invariant_and_exact_broadcast_executor() -> LoadedSymjitDirectExecutor {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(DOUBLE_PROGRAM, 2, 1, 2, false)
                .unwrap();
        LoadedSymjitDirectExecutor::load_prepared_application_bytes(
            &bytes,
            PathBuf::from("mixed-recurrence-broadcasts.symjit"),
            SYMJIT_RECURRENCE_PLANE_APPLICATION_ABI,
            2,
            false,
            DirectExecutorRole::Contribution,
            vec![
                SymjitDirectParameterBinding::Broadcast { index: 2 },
                SymjitDirectParameterBinding::Broadcast { index: 3 },
                SymjitDirectParameterBinding::Broadcast { index: 0 },
                SymjitDirectParameterBinding::Broadcast { index: 1 },
            ],
            vec![
                SymjitDirectPlaneProjection::DestinationCurrent {
                    component: 0,
                    imaginary: false,
                },
                SymjitDirectPlaneProjection::DestinationCurrent {
                    component: 0,
                    imaginary: true,
                },
            ],
            vec![
                SymjitDirectScalarProjection::ExactFactor { imaginary: false },
                SymjitDirectScalarProjection::ExactFactor { imaginary: true },
                SymjitDirectScalarProjection::Parameter {
                    index: 0,
                    imaginary: false,
                },
                SymjitDirectScalarProjection::Parameter {
                    index: 0,
                    imaginary: true,
                },
            ],
            vec![0, 1],
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn views(
        current_re: &mut [f64],
        current_im: &mut [f64],
        amplitude_re: &mut [f64],
        amplitude_im: &mut [f64],
        parameters_re: &[f64],
        parameters_im: &[f64],
        factors_re: &[f64],
        factors_im: &[f64],
        stride: usize,
    ) -> (
        DirectArenaView,
        DirectMomentumView,
        DirectParameterView,
        DirectFactorView,
    ) {
        (
            DirectArenaView {
                current_re: current_re.as_mut_ptr(),
                current_im: current_im.as_mut_ptr(),
                current_scalar_len: current_re.len() as u64,
                amplitude_re: amplitude_re.as_mut_ptr(),
                amplitude_im: amplitude_im.as_mut_ptr(),
                amplitude_scalar_len: amplitude_re.len() as u64,
                point_stride: stride as u32,
            },
            DirectMomentumView {
                values: ptr::NonNull::<f64>::dangling().as_ptr(),
                scalar_len: 0,
                form_count: 0,
                lorentz_component_count: 4,
                point_stride: stride as u32,
            },
            DirectParameterView {
                values_re: parameters_re.as_ptr(),
                values_im: parameters_im.as_ptr(),
                value_count: parameters_re.len() as u32,
            },
            DirectFactorView {
                values_re: factors_re.as_ptr(),
                values_im: factors_im.as_ptr(),
                value_count: factors_re.len() as u32,
            },
        )
    }

    #[test]
    fn direct_output_classifier_requires_finalization_and_disjoint_planes() {
        let mut inputs = vec![0.0; 32];
        let mut output_re = vec![0.0; 32];
        let mut output_im = vec![0.0; 32];
        let input_pointers = [inputs.as_mut_ptr()];
        let disjoint_outputs = [output_re.as_mut_ptr(), output_im.as_mut_ptr()];
        assert!(
            direct_outputs_are_structurally_safe(
                DirectExecutorRole::Finalization,
                &input_pointers,
                &disjoint_outputs,
                32,
            )
            .unwrap()
        );
        assert!(
            !direct_outputs_are_structurally_safe(
                DirectExecutorRole::Contribution,
                &input_pointers,
                &disjoint_outputs,
                32,
            )
            .unwrap()
        );
        assert!(
            !direct_outputs_are_structurally_safe(
                DirectExecutorRole::Finalization,
                &input_pointers,
                &[inputs.as_mut_ptr()],
                32,
            )
            .unwrap()
        );
        assert!(
            !direct_outputs_are_structurally_safe(
                DirectExecutorRole::Finalization,
                &input_pointers,
                &[output_re.as_mut_ptr(), output_re.as_mut_ptr()],
                32,
            )
            .unwrap()
        );
    }

    #[test]
    fn contribution_scalar_simd_tails_apply_complex_scale_and_accumulate() {
        let loaded = identity_executor(DirectExecutorRole::Contribution);
        assert!(matches!(loaded.simd_lane_width(), 1 | 2 | 4));
        let DirectExecutorHandle::Contribution { call, context } = loaded.handle() else {
            unreachable!()
        };
        for &len in LENGTHS {
            let mut current_re = vec![0.0; len * 2];
            let mut current_im = vec![0.0; len * 2];
            for point in 0..len {
                current_re[point] = point as f64 + 0.25;
                current_im[point] = 0.5 - point as f64 * 0.125;
                current_re[len + point] = 7.0;
                current_im[len + point] = -2.0;
            }
            let mut amplitude_re = vec![0.0; len];
            let mut amplitude_im = vec![0.0; len];
            let factors_re = [2.0];
            let factors_im = [-0.5];
            let (arena, momenta, parameters, factors) = views(
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
                &[],
                &[],
                &factors_re,
                &factors_im,
                len,
            );
            let row = DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: DIRECT_NONE_U32,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                destination_component_base: 1,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: 0,
            };
            assert_eq!(
                unsafe {
                    call(
                        context, arena, momenta, parameters, factors, &row, 1, len as u32,
                    )
                },
                DIRECT_STATUS_OK
            );
            for point in 0..len {
                let re = point as f64 + 0.25;
                let im = 0.5 - point as f64 * 0.125;
                assert_eq!(current_re[len + point], 7.0 + re * 2.0 + im * 0.5);
                assert_eq!(current_im[len + point], -2.0 - re * 0.5 + im * 2.0);
            }
        }
    }

    #[test]
    fn initializing_contribution_overwrites_poisoned_complex_multipoint_destination() {
        let loaded = identity_executor(DirectExecutorRole::Contribution);
        let DirectExecutorHandle::Contribution { call, context } = loaded.handle() else {
            unreachable!()
        };
        for &len in &[3, 7, 129] {
            let mut current_re = vec![f64::NAN; len * 2];
            let mut current_im = vec![f64::NAN; len * 2];
            for point in 0..len {
                current_re[point] = point as f64 + 0.25;
                current_im[point] = 0.5 - point as f64 * 0.125;
            }
            let mut amplitude_re = vec![0.0; len];
            let mut amplitude_im = vec![0.0; len];
            let factors_re = [2.0];
            let factors_im = [-0.5];
            let (arena, momenta, parameters, factors) = views(
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
                &[],
                &[],
                &factors_re,
                &factors_im,
                len,
            );
            let row = DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: DIRECT_NONE_U32,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                destination_component_base: 1,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            };
            assert_eq!(
                unsafe {
                    call(
                        context, arena, momenta, parameters, factors, &row, 1, len as u32,
                    )
                },
                DIRECT_STATUS_OK
            );
            for point in 0..len {
                let re = point as f64 + 0.25;
                let im = 0.5 - point as f64 * 0.125;
                assert_eq!(current_re[len + point], re * 2.0 + im * 0.5);
                assert_eq!(current_im[len + point], -re * 0.5 + im * 2.0);
            }
        }
    }

    #[test]
    fn disjoint_identity_finalization_uses_direct_outputs_for_odd_tails() {
        let loaded = parameter_broadcast_executor(DirectExecutorRole::Finalization);
        assert!(matches!(loaded.simd_lane_width(), 1 | 2 | 4));
        let DirectExecutorHandle::Finalization { call, context } = loaded.handle() else {
            unreachable!()
        };
        let row = DirectFinalizationRow {
            component_base: 0,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        let parameters_re = [2.5];
        let parameters_im = [-0.75];
        let factors_re = [1.0];
        let factors_im = [0.0];
        for &len in &[127, 129, 1023, 1025] {
            let mut current_re = vec![f64::NAN; len];
            let mut current_im = vec![f64::NAN; len];
            let mut amplitude_re = vec![0.0; len];
            let mut amplitude_im = vec![0.0; len];
            let (arena, momenta, parameters, factors) = views(
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
                &parameters_re,
                &parameters_im,
                &factors_re,
                &factors_im,
                len,
            );
            assert_eq!(
                unsafe {
                    call(
                        context, arena, momenta, parameters, factors, &row, 1, len as u32,
                    )
                },
                DIRECT_STATUS_OK
            );
            assert_eq!(loaded.cached_direct_table_count(), 1);
            assert!(current_re.iter().all(|&value| value == parameters_re[0]));
            assert!(current_im.iter().all(|&value| value == parameters_im[0]));

            current_re.fill(f64::NAN);
            current_im.fill(f64::NAN);
            let (status, allocations, bytes) = count_allocations(|| unsafe {
                call(
                    context, arena, momenta, parameters, factors, &row, 1, len as u32,
                )
            });
            assert_eq!(status, DIRECT_STATUS_OK);
            assert_eq!((allocations, bytes), (0, 0));
            assert!(current_re.iter().all(|&value| value == parameters_re[0]));
            assert!(current_im.iter().all(|&value| value == parameters_im[0]));
        }
        assert_eq!(loaded.internal_traffic_bytes().0, 0);
    }

    #[test]
    fn scalar_and_tiled_row_workspaces_remain_warm_independently() {
        let loaded = parameter_broadcast_executor(DirectExecutorRole::Finalization);
        let DirectExecutorHandle::Finalization { call, context } = loaded.handle() else {
            unreachable!()
        };
        let row = DirectFinalizationRow {
            component_base: 0,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        let parameters_re = [2.5];
        let parameters_im = [-0.75];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let mut scalar_re = [f64::NAN; 1];
        let mut scalar_im = [f64::NAN; 1];
        let mut scalar_amplitude_re = [0.0; 1];
        let mut scalar_amplitude_im = [0.0; 1];
        let scalar = views(
            &mut scalar_re,
            &mut scalar_im,
            &mut scalar_amplitude_re,
            &mut scalar_amplitude_im,
            &parameters_re,
            &parameters_im,
            &factors_re,
            &factors_im,
            1,
        );
        let mut tiled_re = [f64::NAN; 8];
        let mut tiled_im = [f64::NAN; 8];
        let mut tiled_amplitude_re = [0.0; 8];
        let mut tiled_amplitude_im = [0.0; 8];
        let tiled = views(
            &mut tiled_re,
            &mut tiled_im,
            &mut tiled_amplitude_re,
            &mut tiled_amplitude_im,
            &parameters_re,
            &parameters_im,
            &factors_re,
            &factors_im,
            8,
        );
        for (views, points) in [(scalar, 1), (tiled, 8)] {
            assert_eq!(
                unsafe { call(context, views.0, views.1, views.2, views.3, &row, 1, points,) },
                DIRECT_STATUS_OK
            );
        }
        let (status, allocations, bytes) = count_allocations(|| unsafe {
            call(context, scalar.0, scalar.1, scalar.2, scalar.3, &row, 1, 1)
        });
        assert_eq!(status, DIRECT_STATUS_OK);
        assert_eq!((allocations, bytes), (0, 0));
        assert_eq!(loaded.context.scalar_workspace.borrow().row_groups.len(), 1);
        assert_eq!(loaded.context.tiled_workspace.borrow().row_groups.len(), 1);
    }

    #[test]
    fn disjoint_nonidentity_finalization_uses_scratch_and_tracks_traffic() {
        let loaded = parameter_broadcast_executor(DirectExecutorRole::Finalization);
        let DirectExecutorHandle::Finalization { call, context } = loaded.handle() else {
            unreachable!()
        };
        let len = 129;
        let mut current_re = vec![f64::NAN; len];
        let mut current_im = vec![f64::NAN; len];
        let mut amplitude_re = vec![0.0; len];
        let mut amplitude_im = vec![0.0; len];
        let parameters_re = [2.5];
        let parameters_im = [-0.75];
        let factors_re = [0.5];
        let factors_im = [-0.25];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            &mut amplitude_re,
            &mut amplitude_im,
            &parameters_re,
            &parameters_im,
            &factors_re,
            &factors_im,
            len,
        );
        let row = DirectFinalizationRow {
            component_base: 0,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert_eq!(
            unsafe {
                call(
                    context, arena, momenta, parameters, factors, &row, 1, len as u32,
                )
            },
            DIRECT_STATUS_OK
        );
        assert_eq!(loaded.cached_direct_table_count(), 1);
        let expected_re = parameters_re[0] * factors_re[0] - parameters_im[0] * factors_im[0];
        let expected_im = parameters_re[0] * factors_im[0] + parameters_im[0] * factors_re[0];
        assert!(current_re.iter().all(|&value| value == expected_re));
        assert!(current_im.iter().all(|&value| value == expected_im));
        let scratch_one_way = internal_plane_bytes(2, len, "test scratch traffic").unwrap();
        assert_eq!(
            loaded.internal_traffic_bytes().0,
            scratch_one_way.saturating_mul(2)
        );
    }

    #[test]
    fn aliased_identity_finalization_stays_on_scratch() {
        let loaded = identity_executor(DirectExecutorRole::Finalization);
        let DirectExecutorHandle::Finalization { call, context } = loaded.handle() else {
            unreachable!()
        };
        let len = 129;
        let mut current_re = (0..len).map(|point| point as f64 + 1.0).collect::<Vec<_>>();
        let mut current_im = (0..len).map(|point| 3.0 - point as f64).collect::<Vec<_>>();
        let expected_re = current_re.clone();
        let expected_im = current_im.clone();
        let mut amplitude_re = vec![0.0; len];
        let mut amplitude_im = vec![0.0; len];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            &mut amplitude_re,
            &mut amplitude_im,
            &[],
            &[],
            &factors_re,
            &factors_im,
            len,
        );
        let row = DirectFinalizationRow {
            component_base: 0,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert_eq!(
            unsafe {
                call(
                    context, arena, momenta, parameters, factors, &row, 1, len as u32,
                )
            },
            DIRECT_STATUS_OK
        );
        assert_eq!(loaded.cached_direct_table_count(), 0);
        assert_eq!(current_re, expected_re);
        assert_eq!(current_im, expected_im);
        let scratch_one_way = internal_plane_bytes(2, len, "test scratch traffic").unwrap();
        assert_eq!(
            loaded.internal_traffic_bytes().0,
            scratch_one_way.saturating_mul(2)
        );
    }

    #[test]
    fn finalization_snapshots_input_before_overwrite() {
        let loaded = identity_executor(DirectExecutorRole::Finalization);
        let DirectExecutorHandle::Finalization { call, context } = loaded.handle() else {
            unreachable!()
        };
        let len = 129;
        let mut current_re = (0..len).map(|point| point as f64 + 1.0).collect::<Vec<_>>();
        let mut current_im = (0..len).map(|point| 3.0 - point as f64).collect::<Vec<_>>();
        let original_re = current_re.clone();
        let original_im = current_im.clone();
        let mut amplitude_re = vec![0.0; len];
        let mut amplitude_im = vec![0.0; len];
        let factors_re = [0.25];
        let factors_im = [0.75];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            &mut amplitude_re,
            &mut amplitude_im,
            &[],
            &[],
            &factors_re,
            &factors_im,
            len,
        );
        let row = DirectFinalizationRow {
            component_base: 0,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert_eq!(
            unsafe {
                call(
                    context, arena, momenta, parameters, factors, &row, 1, len as u32,
                )
            },
            DIRECT_STATUS_OK
        );
        for point in 0..len {
            assert_eq!(
                current_re[point],
                original_re[point] * 0.25 - original_im[point] * 0.75
            );
            assert_eq!(
                current_im[point],
                original_re[point] * 0.75 + original_im[point] * 0.25
            );
        }
    }

    #[test]
    fn row_group_cache_preserves_exact_identity_across_lookup_modes() {
        let mut workspace = PlaneWorkspace::default();
        let mut identities = vec![
            RowTableIdentity {
                role: DirectExecutorRole::Contribution,
                address: 100,
                row_count: 1,
            },
            RowTableIdentity {
                role: DirectExecutorRole::Contribution,
                address: 100,
                row_count: 2,
            },
            RowTableIdentity {
                role: DirectExecutorRole::Closure,
                address: 100,
                row_count: 1,
            },
            RowTableIdentity {
                role: DirectExecutorRole::Contribution,
                address: 101,
                row_count: 1,
            },
        ];
        identities.extend((102..230).map(|address| RowTableIdentity {
            role: DirectExecutorRole::Finalization,
            address,
            row_count: 1,
        }));
        for (index, identity) in identities.iter().copied().enumerate() {
            assert_eq!(workspace.cached_row_group_index(identity), None);
            assert_eq!(
                workspace
                    .cache_row_group(CachedRowGroup {
                        identity,
                        rows: Vec::new(),
                        descriptor_bytes: 0,
                    })
                    .unwrap(),
                index
            );
            assert_eq!(
                workspace.row_group_indices.is_empty(),
                index < ROW_GROUP_LINEAR_LOOKUP_LIMIT
            );
            for (expected_index, cached_identity) in identities[..=index].iter().enumerate() {
                assert_eq!(
                    workspace.cached_row_group_index(*cached_identity),
                    Some(expected_index)
                );
            }
        }
        let (_, allocations, _) = count_allocations(|| {
            for (index, identity) in identities.iter().enumerate().rev() {
                assert_eq!(workspace.cached_row_group_index(*identity), Some(index));
            }
        });
        assert_eq!(allocations, 0);
    }

    #[test]
    fn row_group_cache_invalidation_restarts_small_lookup() {
        let mut workspace = PlaneWorkspace::default();
        for address in 0..32 {
            workspace
                .cache_row_group(CachedRowGroup {
                    identity: RowTableIdentity {
                        role: DirectExecutorRole::Contribution,
                        address,
                        row_count: 1,
                    },
                    rows: Vec::new(),
                    descriptor_bytes: 1,
                })
                .unwrap();
        }
        let old_identity = workspace.row_groups[31].identity;
        assert_eq!(workspace.cached_row_group_index(old_identity), Some(31));
        workspace.invalidate_row_tables();
        assert!(workspace.row_groups.is_empty());
        assert!(workspace.row_group_indices.is_empty());
        assert_eq!(workspace.descriptor_bytes, 0);
        assert_eq!(workspace.cached_row_group_index(old_identity), None);
        assert_eq!(
            workspace
                .cache_row_group(CachedRowGroup {
                    identity: old_identity,
                    rows: Vec::new(),
                    descriptor_bytes: 1,
                })
                .unwrap(),
            0
        );
        assert!(workspace.row_group_indices.is_empty());
        assert_eq!(workspace.cached_row_group_index(old_identity), Some(0));
    }

    #[test]
    fn cached_row_group_reuses_one_scratch_set_sequentially() {
        let loaded = identity_executor(DirectExecutorRole::Contribution);
        let DirectExecutorHandle::Contribution { call, context } = loaded.handle() else {
            unreachable!()
        };
        let len = 7;
        let mut current_re = vec![0.0; len * 4];
        let mut current_im = vec![0.0; len * 4];
        for point in 0..len {
            current_re[point] = point as f64 + 1.0;
            current_im[point] = 0.25;
            current_re[2 * len + point] = 10.0 + point as f64;
            current_im[2 * len + point] = -0.5;
        }
        let mut amplitude_re = vec![0.0; len];
        let mut amplitude_im = vec![0.0; len];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            &mut amplitude_re,
            &mut amplitude_im,
            &[],
            &[],
            &factors_re,
            &factors_im,
            len,
        );
        let rows = [
            DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: DIRECT_NONE_U32,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                destination_component_base: 1,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: 0,
            },
            DirectContributionRow {
                parent0_component_base: 2,
                parent1_component_base_or_sentinel: DIRECT_NONE_U32,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                destination_component_base: 3,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: 0,
            },
        ];
        assert_eq!(
            unsafe {
                call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    rows.len() as u32,
                    len as u32,
                )
            },
            DIRECT_STATUS_OK
        );
        for point in 0..len {
            assert_eq!(current_re[len + point], point as f64 + 1.0);
            assert_eq!(current_im[len + point], 0.25);
            assert_eq!(current_re[3 * len + point], 10.0 + point as f64);
            assert_eq!(current_im[3 * len + point], -0.5);
        }
        // A selected dependency run can share the same first row as a longer
        // selection. Pointer equality alone must not reject the second table.
        for row_count in [1, 2, 1] {
            assert_eq!(
                unsafe {
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        row_count,
                        len as u32,
                    )
                },
                DIRECT_STATUS_OK
            );
        }
        assert_eq!(loaded.context.tiled_workspace.borrow().row_groups.len(), 2);
    }

    unsafe extern "C" fn initialize_stable_row_probe_sources(
        _context: *const c_void,
        arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        _factors: DirectFactorView,
        rows: *const DirectSourceRow,
        row_count: u32,
        point_count: u32,
    ) -> c_int {
        if rows.is_null() || row_count == 0 {
            return STATUS_INVALID_ARGUMENT;
        }
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        for row in rows {
            for point in 0..point_count {
                let destination = row.destination_component_base as usize
                    * arena.point_stride as usize
                    + point as usize;
                unsafe {
                    *arena.current_re.add(destination) =
                        f64::from(row.destination_component_base + 1);
                    *arena.current_im.add(destination) = 0.0;
                }
            }
        }
        DIRECT_STATUS_OK
    }

    struct StableRowProbeResolver {
        source_key: OnTheFlyExecutorKeyV1,
        contribution_key: OnTheFlyExecutorKeyV1,
        contribution_handle: DirectExecutorHandle,
    }

    impl OnTheFlyPreparedExecutorResolver for StableRowProbeResolver {
        fn resolve(&self, key: OnTheFlyExecutorKeyV1) -> RusticolResult<ResolvedOnTheFlyExecutor> {
            if key == self.source_key {
                return Ok(ResolvedOnTheFlyExecutor {
                    direct_executor_id: 0,
                    handle: DirectExecutorHandle::Source {
                        call: initialize_stable_row_probe_sources,
                        context: ptr::null(),
                    },
                    parent_permutation: [0, 1],
                    packed_singleton_capable: false,
                    interaction_capability: None,
                });
            }
            if key == self.contribution_key {
                return Ok(ResolvedOnTheFlyExecutor {
                    direct_executor_id: 1,
                    handle: self.contribution_handle,
                    parent_permutation: [1, 0],
                    packed_singleton_capable: false,
                    interaction_capability: None,
                });
            }
            Err(RusticolError::integrity(
                "stable-row probe received an unknown executor key",
            ))
        }

        fn invalidate_row_tables(&self) -> RusticolResult<()> {
            Ok(())
        }
    }

    #[test]
    fn on_the_fly_uses_stable_permuted_rows_across_symjit_warm_reuse() {
        let loaded = identity_executor(DirectExecutorRole::Contribution);
        let (mut trace, source_key, contribution_key) =
            OnTheFlyStructuralTraceV1::test_two_contribution_rows();
        let semantic_digest = trace.semantic_digest();
        let parent_permutations =
            BTreeMap::from([(source_key, [0, 1]), (contribution_key, [1, 0])]);
        trace
            .bind_prepared_executor_rows(&parent_permutations)
            .unwrap();
        assert_eq!(trace.semantic_digest(), semantic_digest);
        assert!(
            trace
                .bind_prepared_executor_rows(&parent_permutations)
                .is_err()
        );
        let rows = trace.test_bound_contribution_rows();
        assert_eq!(
            rows.iter()
                .map(|row| {
                    (
                        row.parent0_component_base,
                        row.parent1_component_base_or_sentinel,
                        row.parent0_momentum_form_id,
                        row.parent1_momentum_form_id_or_sentinel,
                        row.destination_component_base,
                    )
                })
                .collect::<Vec<_>>(),
            vec![(1, 0, 1, 0, 2), (3, 0, 3, 0, 4)]
        );

        let resolver = StableRowProbeResolver {
            source_key,
            contribution_key,
            contribution_handle: loaded.handle(),
        };
        let mut workspace = OnTheFlyWorkspaceV1::new(&trace, 1).unwrap();
        for _ in 0..2 {
            OnTheFlyStructuralInterpreter::execute(&trace, &resolver, &mut workspace, 1).unwrap();
            assert_eq!(
                workspace.observed_current_components(&trace, 2, 0).unwrap(),
                vec![(2.0, 0.0)]
            );
            assert_eq!(
                workspace.observed_current_components(&trace, 4, 0).unwrap(),
                vec![(4.0, 0.0)]
            );
        }
        // The legacy OTF interpreter keeps an aligned arena stride even for a
        // one-point call, so its tables belong to the tiled cache. The newer
        // packed-singleton path uses the independent scalar cache.
        assert_eq!(loaded.context.tiled_workspace.borrow().row_groups.len(), 2);
        loaded.invalidate_row_tables().unwrap();
        for workspace in [
            &loaded.context.scalar_workspace,
            &loaded.context.tiled_workspace,
        ] {
            let workspace = workspace.borrow();
            assert!(workspace.row_groups.is_empty());
            assert!(workspace.row_group_indices.is_empty());
            assert!(workspace.storage.is_none());
            assert_eq!(workspace.descriptor_bytes, 0);
        }
    }

    #[test]
    fn model_parameter_broadcast_planes_refresh_without_reloading_kernel() {
        let loaded = parameter_broadcast_executor(DirectExecutorRole::Contribution);
        let DirectExecutorHandle::Contribution { call, context } = loaded.handle() else {
            unreachable!()
        };
        let len = 17;
        let mut current_re = vec![0.0; len];
        let mut current_im = vec![0.0; len];
        let mut amplitude_re = vec![0.0; len];
        let mut amplitude_im = vec![0.0; len];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        let mut expected_re = 0.0;
        let mut expected_im = 0.0;
        for (iteration, (parameter_re, parameter_im)) in [(2.5, -0.75), (-4.0, 1.25), (-4.0, 1.25)]
            .into_iter()
            .enumerate()
        {
            expected_re += parameter_re;
            expected_im += parameter_im;
            let parameters_re = [parameter_re];
            let parameters_im = [parameter_im];
            let (arena, momenta, parameters, factors) = views(
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
                &parameters_re,
                &parameters_im,
                &factors_re,
                &factors_im,
                len,
            );
            let invoke = || unsafe {
                call(
                    context, arena, momenta, parameters, factors, &row, 1, len as u32,
                )
            };
            let status = if iteration == 0 {
                invoke()
            } else {
                let (status, allocations, bytes) = count_allocations(invoke);
                assert_eq!((allocations, bytes), (0, 0));
                status
            };
            assert_eq!(status, DIRECT_STATUS_OK);
            for point in 0..len {
                assert_eq!(current_re[point], expected_re);
                assert_eq!(current_im[point], expected_im);
            }
        }
        let active_plane_bytes = 2 * len as u64 * size_of::<f64>() as u64;
        assert_eq!(
            loaded.internal_traffic_bytes(),
            (
                3 * 2 * active_plane_bytes,
                3 * active_plane_bytes + 2 * active_plane_bytes,
            )
        );
    }

    #[test]
    fn migration_contract_interns_repeated_parameter_and_zero_broadcasts() {
        let loaded = repeated_parameter_and_zero_broadcast_executor();
        assert_eq!(loaded.broadcast_plane_count(), 2);
        assert_eq!(
            loaded.context.broadcast_slot_by_binding.as_ref(),
            [Some(0), Some(1), Some(0), Some(1)]
        );
        let DirectExecutorHandle::Contribution { call, context } = loaded.handle() else {
            unreachable!()
        };
        let len = 129;
        let mut current_re = vec![1.0; len];
        let mut current_im = vec![-2.0; len];
        let mut amplitude_re = vec![0.0; len];
        let mut amplitude_im = vec![0.0; len];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        let mut expected_re = 1.0;
        for (iteration, parameter_re) in [2.5, 2.5, -4.0].into_iter().enumerate() {
            expected_re += 2.0 * parameter_re;
            let parameters_re = [parameter_re];
            let parameters_im = [123.0];
            let (arena, momenta, parameters, factors) = views(
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
                &parameters_re,
                &parameters_im,
                &factors_re,
                &factors_im,
                len,
            );
            let invoke = || unsafe {
                call(
                    context, arena, momenta, parameters, factors, &row, 1, len as u32,
                )
            };
            let status = if iteration == 0 {
                invoke()
            } else {
                let (status, allocations, bytes) = count_allocations(invoke);
                assert_eq!((allocations, bytes), (0, 0));
                status
            };
            assert_eq!(status, DIRECT_STATUS_OK);
            assert!(current_re.iter().all(|&value| value == expected_re));
            assert!(current_im.iter().all(|&value| value == -2.0));
        }
        assert_eq!(
            loaded.context.tiled_workspace.borrow().broadcasts.len(),
            2 * len
        );
        let plane_bytes = len as u64 * size_of::<f64>() as u64;
        assert_eq!(
            loaded.internal_traffic_bytes(),
            (
                3 * 4 * plane_bytes,
                // Two shared planes are read on each call. Both are filled
                // once, and only the parameter plane changes on the last call.
                3 * 2 * plane_bytes + 3 * plane_bytes,
            )
        );
    }

    #[test]
    fn multirow_call_refreshes_invariants_once_and_exact_factors_per_row() {
        let loaded = mixed_invariant_and_exact_broadcast_executor();
        assert_eq!(loaded.broadcast_plane_count(), 4);
        assert_eq!(loaded.context.exact_factor_broadcast_slots.as_ref(), [2, 3]);
        let DirectExecutorHandle::Contribution { call, context } = loaded.handle() else {
            unreachable!()
        };
        let len = 17;
        let mut current_re = vec![0.0; 2 * len];
        let mut current_im = vec![0.0; 2 * len];
        let mut amplitude_re = vec![0.0; len];
        let mut amplitude_im = vec![0.0; len];
        let factors_re = [2.0, -1.0];
        let factors_im = [-0.25, 0.75];
        let first_row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        let rows = [
            first_row,
            DirectContributionRow {
                destination_component_base: 1,
                exact_factor_id: 1,
                ..first_row
            },
        ];
        let expected_increment = |parameter_re: f64, parameter_im: f64, factor: usize| {
            let kernel_re = parameter_re + factors_re[factor];
            let kernel_im = parameter_im + factors_im[factor];
            (
                kernel_re * factors_re[factor] - kernel_im * factors_im[factor],
                kernel_re * factors_im[factor] + kernel_im * factors_re[factor],
            )
        };
        let invoke = |parameter_re: f64,
                      parameter_im: f64,
                      current_re: &mut [f64],
                      current_im: &mut [f64],
                      amplitude_re: &mut [f64],
                      amplitude_im: &mut [f64]| {
            let parameters_re = [parameter_re];
            let parameters_im = [parameter_im];
            let (arena, momenta, parameters, factors) = views(
                current_re,
                current_im,
                amplitude_re,
                amplitude_im,
                &parameters_re,
                &parameters_im,
                &factors_re,
                &factors_im,
                len,
            );
            unsafe {
                call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    rows.len() as u32,
                    len as u32,
                )
            }
        };

        assert_eq!(
            invoke(
                3.0,
                0.5,
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
            ),
            DIRECT_STATUS_OK
        );
        assert_eq!(loaded.broadcast_resolution_counts(), (2, 4));
        for factor in 0..2 {
            let (expected_re, expected_im) = expected_increment(3.0, 0.5, factor);
            assert!(
                current_re[factor * len..(factor + 1) * len]
                    .iter()
                    .all(|&value| value == expected_re)
            );
            assert!(
                current_im[factor * len..(factor + 1) * len]
                    .iter()
                    .all(|&value| value == expected_im)
            );
        }

        let (status, allocations, bytes) = count_allocations(|| {
            invoke(
                3.0,
                0.5,
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
            )
        });
        assert_eq!(status, DIRECT_STATUS_OK);
        assert_eq!((allocations, bytes), (0, 0));
        assert_eq!(loaded.broadcast_resolution_counts(), (4, 8));

        assert_eq!(
            invoke(
                -4.0,
                1.25,
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
            ),
            DIRECT_STATUS_OK
        );
        assert_eq!(loaded.broadcast_resolution_counts(), (6, 12));
        for factor in 0..2 {
            let (first_re, first_im) = expected_increment(3.0, 0.5, factor);
            let (mutated_re, mutated_im) = expected_increment(-4.0, 1.25, factor);
            assert!(
                current_re[factor * len..(factor + 1) * len]
                    .iter()
                    .all(|&value| value == 2.0 * first_re + mutated_re)
            );
            assert!(
                current_im[factor * len..(factor + 1) * len]
                    .iter()
                    .all(|&value| value == 2.0 * first_im + mutated_im)
            );
        }

        let plane_bytes = len as u64 * size_of::<f64>() as u64;
        assert_eq!(
            loaded.internal_traffic_bytes(),
            (
                // Two scratch output planes are written and read for each of
                // two rows on every call.
                3 * 2 * 2 * 2 * plane_bytes,
                // Four broadcast planes are read per row. Refreshes are 6 on
                // the cold call, 4 on the identical warm call, then 6 after
                // both model-parameter planes mutate.
                (3 * 2 * 4 + 6 + 4 + 6) * plane_bytes,
            )
        );
    }

    #[test]
    fn warmed_recurrence_plane_call_allocates_nothing() {
        let loaded = identity_executor(DirectExecutorRole::Contribution);
        let DirectExecutorHandle::Contribution { call, context } = loaded.handle() else {
            unreachable!()
        };
        let mut expected_scratch_bytes = 0;
        for &len in &[127, 129, 1023, 1025] {
            let mut current_re = vec![1.0; len * 2];
            let mut current_im = vec![0.5; len * 2];
            let mut amplitude_re = vec![0.0; len];
            let mut amplitude_im = vec![0.0; len];
            let factors_re = [1.0];
            let factors_im = [0.0];
            let (arena, momenta, parameters, factors) = views(
                &mut current_re,
                &mut current_im,
                &mut amplitude_re,
                &mut amplitude_im,
                &[],
                &[],
                &factors_re,
                &factors_im,
                len,
            );
            let row = DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: DIRECT_NONE_U32,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                destination_component_base: 1,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: 0,
            };
            assert_eq!(
                unsafe {
                    call(
                        context, arena, momenta, parameters, factors, &row, 1, len as u32,
                    )
                },
                DIRECT_STATUS_OK
            );
            let (status, allocations, bytes) = count_allocations(|| unsafe {
                call(
                    context, arena, momenta, parameters, factors, &row, 1, len as u32,
                )
            });
            assert_eq!(status, DIRECT_STATUS_OK);
            assert_eq!(
                (allocations, bytes),
                (0, 0),
                "warmed recurrence call allocated for {len} points"
            );
            let scratch_one_way = internal_plane_bytes(2, len, "test scratch traffic").unwrap();
            expected_scratch_bytes += scratch_one_way.saturating_mul(4);
        }
        assert_eq!(loaded.cached_direct_table_count(), 0);
        assert_eq!(loaded.internal_traffic_bytes().0, expected_scratch_bytes);
    }
}
