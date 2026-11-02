// SPDX-License-Identifier: 0BSD

//! Eager-lane orchestration for standard SymJIT direct-arena P-kernels.
//!
//! SymJIT owns only the ordinary plane kernel. Rusticol authenticates and
//! decodes immutable invocation/attachment rows, preserves their order and
//! hazards, and implements factors, overwrite/add, and fanout. A single safe
//! identity overwrite binds outputs directly; every other invocation writes
//! once to persistent split-complex scratch and fans out in allocation-free
//! Rust loops. Native C++/ASM table callables retain their public C layout.

use super::symjit_plane::{PlaneDescriptor, SymjitPlaneKernel, SymjitPlaneLayout};
use crate::artifact::PinnedNativeLibrary;
use crate::direct_arena::{DirectArenaView, DirectArenaWorkspace};
pub(crate) use crate::eager_layout::{
    EAGER_DIRECT_SOURCE_APPLICATION_ABI, EAGER_DIRECT_TABLE_BINDING_ABI,
    EAGER_DIRECT_TABLE_DESCRIPTOR_ABI, EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI,
};
use crate::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};
use std::any::Any;
#[cfg(feature = "f64-compiled")]
use std::ffi::CStr;
#[cfg(feature = "f64-compiled")]
use std::os::raw::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
#[cfg(target_arch = "aarch64")]
use wide::f64x2;
#[cfg(target_arch = "x86_64")]
use wide::f64x4;

const EAGER_PLANE_DESCRIPTOR_MAGIC: [u8; 8] = *b"PACETD01";
const EAGER_PLANE_DESCRIPTOR_VERSION: u32 = 1;
const EAGER_PLANE_DESCRIPTOR_SIZE: usize = 64;
const EAGER_DESTINATION_OVERWRITE: u32 = 0;
const EAGER_DESTINATION_ACCUMULATE: u32 = 1;
const DIRECT_STATUS_OK: i32 = 0;
const DIRECT_STATUS_INVALID_CONTEXT: i32 = 1;
const DIRECT_STATUS_INVALID_ARGUMENT: i32 = 2;
const DIRECT_STATUS_EXECUTION_FAILED: i32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EagerDirectTableMetadata {
    input_complex_count: u32,
    output_complex_count: u32,
    input_plane_count: usize,
    output_plane_count: usize,
    invocation_stride: u32,
    attachment_stride: u32,
}

impl EagerDirectTableMetadata {
    fn new(input_complex_count: u32, output_complex_count: u32) -> RusticolResult<Self> {
        if input_complex_count == 0 || output_complex_count == 0 {
            return Err(RusticolError::invalid_argument(
                "eager direct-table input and output widths must be positive",
            ));
        }
        let input_plane_count_u32 = input_complex_count.checked_mul(2).ok_or_else(|| {
            RusticolError::invalid_argument("eager direct-table input width overflows")
        })?;
        let output_plane_count_u32 = output_complex_count.checked_mul(2).ok_or_else(|| {
            RusticolError::invalid_argument("eager direct-table output width overflows")
        })?;
        let invocation_stride = input_plane_count_u32
            .checked_add(2)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| {
                RusticolError::invalid_argument("eager direct-table invocation row overflows")
            })?;
        let attachment_stride = output_plane_count_u32
            .checked_add(2)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| {
                RusticolError::invalid_argument("eager direct-table attachment row overflows")
            })?;
        Ok(Self {
            input_complex_count,
            output_complex_count,
            input_plane_count: input_plane_count_u32 as usize,
            output_plane_count: output_plane_count_u32 as usize,
            invocation_stride,
            attachment_stride,
        })
    }

    const fn layout(self) -> SymjitPlaneLayout {
        SymjitPlaneLayout::complex(
            self.input_complex_count as usize,
            self.output_complex_count as usize,
        )
    }

    const fn attachment_start_offset(self) -> usize {
        self.input_plane_count * std::mem::size_of::<u32>()
    }

    const fn attachment_count_offset(self) -> usize {
        self.attachment_start_offset() + std::mem::size_of::<u32>()
    }

    const fn scale_offset(self) -> usize {
        self.output_plane_count * std::mem::size_of::<u32>()
    }

    const fn operation_offset(self) -> usize {
        self.scale_offset() + std::mem::size_of::<u32>()
    }
}

/// Derive the authenticated Rusticol plane-table descriptor for one prepared
/// P-kernel. The function signature is retained for Python artifact generation.
pub(crate) fn eager_direct_descriptor_for_source_application_bytes(
    source_bytes: &[u8],
    input_complex_count: u32,
    output_complex_count: u32,
    display_path: &Path,
) -> RusticolResult<Vec<u8>> {
    let metadata = EagerDirectTableMetadata::new(input_complex_count, output_complex_count)?;
    // Loading here proves that generation cannot bind a B-kernel, a malformed
    // application, or mismatched split-plane dimensions.
    let _kernel = SymjitPlaneKernel::load_bytes(source_bytes, display_path, metadata.layout())?;
    Ok(encode_plane_descriptor(source_bytes, metadata))
}

fn encode_plane_descriptor(source_bytes: &[u8], metadata: EagerDirectTableMetadata) -> Vec<u8> {
    let digest = Sha256::digest(source_bytes);
    let mut bytes = Vec::with_capacity(EAGER_PLANE_DESCRIPTOR_SIZE);
    bytes.extend_from_slice(&EAGER_PLANE_DESCRIPTOR_MAGIC);
    push_descriptor_u32(&mut bytes, EAGER_PLANE_DESCRIPTOR_VERSION);
    push_descriptor_u32(&mut bytes, EAGER_PLANE_DESCRIPTOR_SIZE as u32);
    push_descriptor_u32(&mut bytes, metadata.input_complex_count);
    push_descriptor_u32(&mut bytes, metadata.output_complex_count);
    push_descriptor_u32(&mut bytes, metadata.invocation_stride);
    push_descriptor_u32(&mut bytes, metadata.attachment_stride);
    bytes.extend_from_slice(&digest);
    debug_assert_eq!(bytes.len(), EAGER_PLANE_DESCRIPTOR_SIZE);
    bytes
}

fn decode_plane_descriptor(
    source_bytes: &[u8],
    descriptor_bytes: &[u8],
    display_path: &Path,
) -> RusticolResult<EagerDirectTableMetadata> {
    let regenerate = || {
        RusticolError::compatibility(format!(
            "eager plane-table descriptor {} is not compatible with the SymJIT 2.22 P-kernel ABI; regenerate the prepared model",
            display_path.display()
        ))
    };
    if descriptor_bytes.len() != EAGER_PLANE_DESCRIPTOR_SIZE
        || descriptor_bytes.get(..8) != Some(EAGER_PLANE_DESCRIPTOR_MAGIC.as_slice())
        || read_descriptor_u32(descriptor_bytes, 8)? != EAGER_PLANE_DESCRIPTOR_VERSION
        || read_descriptor_u32(descriptor_bytes, 12)? != EAGER_PLANE_DESCRIPTOR_SIZE as u32
    {
        return Err(regenerate());
    }
    let metadata = EagerDirectTableMetadata::new(
        read_descriptor_u32(descriptor_bytes, 16)?,
        read_descriptor_u32(descriptor_bytes, 20)?,
    )?;
    if read_descriptor_u32(descriptor_bytes, 24)? != metadata.invocation_stride
        || read_descriptor_u32(descriptor_bytes, 28)? != metadata.attachment_stride
    {
        return Err(RusticolError::integrity(format!(
            "eager plane-table descriptor {} has inconsistent canonical row strides",
            display_path.display()
        )));
    }
    let digest = Sha256::digest(source_bytes);
    if descriptor_bytes[32..] != digest[..] {
        return Err(RusticolError::integrity(format!(
            "eager plane-table descriptor {} does not authenticate its P-kernel source",
            display_path.display()
        )));
    }
    Ok(metadata)
}

fn push_descriptor_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_descriptor_u32(bytes: &[u8], offset: usize) -> RusticolResult<u32> {
    let encoded = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| RusticolError::integrity("eager plane-table descriptor is truncated"))?;
    Ok(u32::from_le_bytes(encoded.try_into().map_err(|_| {
        RusticolError::internal("could not decode eager descriptor field")
    })?))
}

/// One persistent plane-catalog entry backed by the eager arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EagerDirectArenaPlaneBinding {
    CurrentReal(u32),
    CurrentImag(u32),
    AmplitudeReal(u32),
    AmplitudeImag(u32),
}

/// Rusticol-owned ABI twin retained for native C++/ASM callables.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct EagerDirectPlane {
    values: *mut f64,
    len: usize,
}

/// Rusticol-owned ABI twin retained for native C++/ASM callables.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct EagerDirectScalar {
    value: *const f64,
}

#[derive(Clone, Copy, Debug)]
struct ParsedInvocation {
    input_start: usize,
    attachment_start: u32,
    attachment_count: u32,
}

#[derive(Clone, Copy, Debug)]
struct ParsedAttachment {
    destination_start: usize,
    factor: u32,
    operation: u32,
}

/// Immutable raw rows plus their cold-decoded Rust orchestration form.
#[derive(Debug)]
pub(crate) struct EagerDirectTableRows {
    invocations_raw: Box<[u8]>,
    attachments_raw: Box<[u8]>,
    invocations: Box<[ParsedInvocation]>,
    input_planes: Box<[u32]>,
    attachments: Box<[ParsedAttachment]>,
    destination_planes: Box<[u32]>,
    input_plane_count: usize,
    output_plane_count: usize,
}

impl EagerDirectTableRows {
    pub(crate) fn invocation_count(&self) -> u32 {
        self.invocations.len() as u32
    }

    pub(crate) fn attachment_count(&self) -> u32 {
        self.attachments.len() as u32
    }
}

/// Persistent eager arena, native catalogs, P-kernel descriptors, and scratch.
pub(crate) struct EagerDirectTableWorkspace {
    arena: DirectArenaWorkspace,
    planes: Box<[EagerDirectPlane]>,
    /// Plane-catalog entry reserved as an immutable structural zero for
    /// real-only SymJIT inputs. Native tables never use this remapping.
    read_only_zero_plane_index: Option<u32>,
    // Own the stable allocations addressed by `scalars`.
    _scalar_values: Box<[f64]>,
    scalars: Box<[EagerDirectScalar]>,
    factor_re: Box<[f64]>,
    factor_im: Box<[f64]>,
    symjit_descriptors: Vec<PlaneDescriptor<'static>>,
    symjit_scratch: Vec<f64>,
    internal_scratch_bytes: u64,
}

impl EagerDirectTableWorkspace {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        current_plane_count: u32,
        amplitude_plane_count: u32,
        tile_capacity: u32,
        plane_bindings: &[EagerDirectArenaPlaneBinding],
        read_only_zero_plane_index: Option<u32>,
        scalar_values: Vec<f64>,
        factor_re: Vec<f64>,
        factor_im: Vec<f64>,
    ) -> RusticolResult<Self> {
        if plane_bindings.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager direct-table plane catalog must not be empty",
            ));
        }
        if factor_re.is_empty() || factor_re.len() != factor_im.len() {
            return Err(RusticolError::invalid_argument(
                "eager direct-table factors must be nonempty split-complex pairs",
            ));
        }
        let _ = u32::try_from(plane_bindings.len()).map_err(|_| {
            RusticolError::invalid_argument("eager direct-table plane catalog exceeds u32")
        })?;
        if let Some(zero_plane) = read_only_zero_plane_index {
            let zero_binding = plane_bindings.get(zero_plane as usize).ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "eager direct-table shared-zero plane {zero_plane} exceeds catalog size {}",
                    plane_bindings.len()
                ))
            })?;
            if !matches!(zero_binding, EagerDirectArenaPlaneBinding::CurrentImag(_)) {
                return Err(RusticolError::invalid_argument(format!(
                    "eager direct-table shared-zero plane {zero_plane} is not an imaginary current plane"
                )));
            }
            if plane_bindings
                .iter()
                .enumerate()
                .any(|(index, binding)| index != zero_plane as usize && binding == zero_binding)
            {
                return Err(RusticolError::invalid_argument(format!(
                    "eager direct-table shared-zero plane {zero_plane} has an aliased catalog binding"
                )));
            }
        }
        let _ = u32::try_from(scalar_values.len()).map_err(|_| {
            RusticolError::invalid_argument("eager direct-table scalar catalog exceeds u32")
        })?;
        let _ = u32::try_from(factor_re.len()).map_err(|_| {
            RusticolError::invalid_argument("eager direct-table factor catalog exceeds u32")
        })?;
        let mut arena =
            DirectArenaWorkspace::new(current_plane_count, amplitude_plane_count, tile_capacity)?;
        arena.begin_tile(tile_capacity)?;
        let view = arena.view()?;
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(plane_bindings.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not reserve eager direct-table plane catalog: {error}"
                ))
            })?;
        for &binding in plane_bindings {
            planes.push(bind_arena_plane(view, binding)?);
        }

        let scalar_values = scalar_values.into_boxed_slice();
        let scalars = scalar_values
            .iter()
            .map(|value| EagerDirectScalar {
                value: ptr::from_ref(value),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            arena,
            planes: planes.into_boxed_slice(),
            read_only_zero_plane_index,
            _scalar_values: scalar_values,
            scalars,
            factor_re: factor_re.into_boxed_slice(),
            factor_im: factor_im.into_boxed_slice(),
            symjit_descriptors: Vec::new(),
            symjit_scratch: Vec::new(),
            internal_scratch_bytes: 0,
        })
    }

    pub(crate) fn begin_tile(&mut self, point_count: u32) -> RusticolResult<()> {
        self.arena.begin_tile(point_count)
    }

    pub(crate) fn arena(&self) -> &DirectArenaWorkspace {
        &self.arena
    }

    pub(crate) fn split_arena_slices_mut(
        &mut self,
    ) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        self.arena.split_slices_mut()
    }

    pub(crate) fn clear_current_active(
        &mut self,
        component_base: u32,
        component_count: u32,
    ) -> RusticolResult<()> {
        self.arena
            .clear_current_active(component_base, component_count)
    }

    pub(crate) fn clear_amplitude_active(
        &mut self,
        component_base: u32,
        component_count: u32,
    ) -> RusticolResult<()> {
        self.arena
            .clear_amplitude_active(component_base, component_count)
    }

    /// Retained for cold scalar refresh paths; current eager manifests bind
    /// immutable scalar catalogs.
    #[allow(dead_code)]
    pub(crate) fn scalar_values_mut(&mut self) -> &mut [f64] {
        &mut self._scalar_values
    }

    pub(crate) fn factors_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        (&mut self.factor_re, &mut self.factor_im)
    }

    pub(crate) const fn internal_scratch_bytes(&self) -> u64 {
        self.internal_scratch_bytes
    }

    pub(crate) fn symjit_descriptor_storage_bytes(
        descriptor_count: usize,
    ) -> RusticolResult<usize> {
        descriptor_count
            .checked_mul(std::mem::size_of::<PlaneDescriptor<'static>>())
            .ok_or_else(|| RusticolError::internal("eager descriptor storage size overflows"))
    }

    pub(crate) fn symjit_workspace_bytes(&self) -> RusticolResult<usize> {
        Self::symjit_descriptor_storage_bytes(self.symjit_descriptors.capacity())?
            .checked_add(
                self.symjit_scratch
                    .capacity()
                    .checked_mul(std::mem::size_of::<f64>())
                    .ok_or_else(|| {
                        RusticolError::internal("eager scratch storage size overflows")
                    })?,
            )
            .ok_or_else(|| RusticolError::internal("eager SymJIT workspace size overflows"))
    }

    fn prepare_symjit(&mut self, metadata: EagerDirectTableMetadata) -> RusticolResult<()> {
        let descriptor_count = metadata
            .input_plane_count
            .checked_add(metadata.output_plane_count)
            .ok_or_else(|| RusticolError::internal("eager descriptor count overflows"))?;
        if self.symjit_descriptors.capacity() < descriptor_count {
            self.symjit_descriptors
                .try_reserve_exact(descriptor_count - self.symjit_descriptors.len())
                .map_err(|error| {
                    RusticolError::internal(format!(
                        "could not reserve eager P-kernel descriptors: {error}"
                    ))
                })?;
        }
        let scratch_len = metadata
            .output_plane_count
            .checked_mul(self.arena.point_stride() as usize)
            .ok_or_else(|| RusticolError::internal("eager scratch size overflows"))?;
        if self.symjit_scratch.len() < scratch_len {
            self.symjit_scratch
                .try_reserve_exact(scratch_len - self.symjit_scratch.len())
                .map_err(|error| {
                    RusticolError::internal(format!(
                        "could not reserve eager P-kernel scratch: {error}"
                    ))
                })?;
            self.symjit_scratch.resize(scratch_len, 0.0);
        }
        Ok(())
    }

    fn checked_range(&self, point_start: u32, point_count: u32) -> RusticolResult<(usize, usize)> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "eager direct-table point count must be positive",
            ));
        }
        let point_stop = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("eager direct-table point range overflows u32")
        })?;
        if point_stop > self.arena.active_point_count() {
            return Err(RusticolError::invalid_argument(format!(
                "eager direct-table point range {point_start}..{point_stop} exceeds active tile {}",
                self.arena.active_point_count()
            )));
        }
        Ok((point_start as usize, point_stop as usize))
    }

    fn native_view(
        &self,
        rows: &EagerDirectTableRows,
        metadata: EagerDirectTableMetadata,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<EagerDirectTableCallViewV1> {
        let _ = self.checked_range(point_start, point_count)?;
        Ok(EagerDirectTableCallViewV1 {
            invocations: rows.invocations_raw.as_ptr(),
            invocation_count: rows.invocation_count(),
            invocation_stride: metadata.invocation_stride,
            attachments: rows.attachments_raw.as_ptr(),
            attachment_count: rows.attachment_count(),
            attachment_stride: metadata.attachment_stride,
            planes: self.planes.as_ptr(),
            plane_count: self.planes.len() as u32,
            scalar_count: self.scalars.len() as u32,
            scalars: if self.scalars.is_empty() {
                ptr::null()
            } else {
                self.scalars.as_ptr()
            },
            scale_re: self.factor_re.as_ptr(),
            scale_im: self.factor_im.as_ptr(),
            scale_count: self.factor_re.len() as u32,
            point_start,
            point_count,
        })
    }
}

/// Loaded owner of one prepared eager callable.
pub(crate) struct LoadedSymjitEagerDirectTable {
    callable: EagerDirectTableCallable,
    display_path: PathBuf,
}

// Keep the callable inline: it is cold-built once, while boxing it would add a
// separately tracked allocation and indirection to every warmed dispatch.
#[allow(clippy::large_enum_variant)]
enum EagerDirectTableCallable {
    Symjit {
        kernel: SymjitPlaneKernel,
        metadata: EagerDirectTableMetadata,
    },
    #[cfg(feature = "f64-compiled")]
    Native(LoadedNativeEagerDirectTable),
}

#[derive(Clone, Copy)]
#[repr(C)]
struct EagerDirectTableCallViewV1 {
    invocations: *const u8,
    invocation_count: u32,
    invocation_stride: u32,
    attachments: *const u8,
    attachment_count: u32,
    attachment_stride: u32,
    planes: *const EagerDirectPlane,
    plane_count: u32,
    scalar_count: u32,
    scalars: *const EagerDirectScalar,
    scale_re: *const f64,
    scale_im: *const f64,
    scale_count: u32,
    point_start: u32,
    point_count: u32,
}

#[cfg(feature = "f64-compiled")]
type NativeEagerDirectTableCall = unsafe extern "C" fn(*const EagerDirectTableCallViewV1) -> i32;

#[cfg(feature = "f64-compiled")]
#[repr(C)]
struct NativeEagerDirectTableMetadataV1 {
    struct_size: u32,
    abi_version: u32,
    flags: u32,
    invocation_stride: u32,
    attachment_stride: u32,
    input_complex_count: u32,
    output_complex_count: u32,
    simd_lane_width: u32,
    application_abi: *const c_char,
    function_name: *const c_char,
    target_triple: *const c_char,
    evaluator_state_sha256: *const c_char,
}

#[cfg(feature = "f64-compiled")]
type NativeEagerDirectTableMetadataCall =
    unsafe extern "C" fn() -> *const NativeEagerDirectTableMetadataV1;

#[cfg(feature = "f64-compiled")]
const NATIVE_EAGER_DIRECT_TABLE_METADATA_VERSION: u32 = 1;
#[cfg(feature = "f64-compiled")]
const NATIVE_EAGER_DIRECT_TABLE_REQUIRED_FLAGS: u32 = 0x1f;

#[cfg(feature = "f64-compiled")]
struct LoadedNativeEagerDirectTable {
    _library: Arc<PinnedNativeLibrary>,
    call: NativeEagerDirectTableCall,
    metadata: EagerDirectTableMetadata,
}

impl LoadedSymjitEagerDirectTable {
    /// Whether this callable consumes SymJIT P-kernel plane descriptors.
    ///
    /// Native C++/ASM tables retain their original per-semantic plane IDs.
    pub(crate) fn is_symjit(&self) -> bool {
        matches!(&self.callable, EagerDirectTableCallable::Symjit { .. })
    }

    /// Fixed workspace shape required by this callable.
    ///
    /// The first element is the cold descriptor count. The second is the
    /// number of point-pitched scalar scratch planes. Native table callables
    /// own their workspace and therefore report no Rust-side SymJIT shape.
    pub(crate) fn symjit_workspace_shape(&self) -> Option<(usize, usize)> {
        match &self.callable {
            EagerDirectTableCallable::Symjit { metadata, .. } => Some((
                metadata.input_plane_count + metadata.output_plane_count,
                metadata.output_plane_count,
            )),
            #[cfg(feature = "f64-compiled")]
            EagerDirectTableCallable::Native(_) => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_prepared_application_bytes(
        source_bytes: &[u8],
        descriptor_bytes: &[u8],
        display_path: PathBuf,
        source_application_abi: &str,
        descriptor_abi: &str,
        binding_abi: &str,
        expected_optimization_level: u32,
        expected_compression: bool,
    ) -> RusticolResult<Self> {
        validate_portable_abis(source_application_abi, descriptor_abi, binding_abi)?;
        let metadata = decode_plane_descriptor(source_bytes, descriptor_bytes, &display_path)?;
        let kernel = SymjitPlaneKernel::load_bytes(source_bytes, &display_path, metadata.layout())?;
        if u32::from(kernel.optimization_level()) != expected_optimization_level {
            return Err(RusticolError::integrity(format!(
                "eager SymJIT plane application {} stores optimization level {}, but its \
                 authenticated prepared-kernel variant declares optimization level \
                 {expected_optimization_level}; regenerate the prepared model",
                display_path.display(),
                kernel.optimization_level(),
            )));
        }
        if kernel.compression() != expected_compression {
            return Err(RusticolError::integrity(format!(
                "eager SymJIT plane application {} stores compression {}, but its \
                 authenticated prepared-kernel manifest declares compression \
                 {expected_compression}; regenerate the prepared model",
                display_path.display(),
                kernel.compression(),
            )));
        }
        Ok(Self {
            callable: EagerDirectTableCallable::Symjit { kernel, metadata },
            display_path,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_native_application(
        library: Arc<PinnedNativeLibrary>,
        function_name: &str,
        display_path: PathBuf,
        source_application_abi: &str,
        invocation_stride: u32,
        attachment_stride: u32,
        input_complex_count: u32,
        output_complex_count: u32,
        expected_target_triple: &str,
        expected_evaluator_state_sha256: &str,
        expected_simd_lane_width: u32,
    ) -> RusticolResult<Self> {
        if source_application_abi != EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported native eager DirectTable application ABI {source_application_abi:?}; expected {EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI:?}"
            )));
        }
        validate_c_symbol(function_name)?;
        let metadata = EagerDirectTableMetadata::new(input_complex_count, output_complex_count)?;
        if metadata.invocation_stride != invocation_stride
            || metadata.attachment_stride != attachment_stride
        {
            return Err(RusticolError::integrity(format!(
                "native eager DirectTable {} row strides ({invocation_stride}, {attachment_stride}) do not match ({}, {})",
                display_path.display(),
                metadata.invocation_stride,
                metadata.attachment_stride,
            )));
        }
        #[cfg(not(feature = "f64-compiled"))]
        {
            let _ = (
                library,
                function_name,
                metadata,
                expected_target_triple,
                expected_evaluator_state_sha256,
                expected_simd_lane_width,
            );
            return Err(RusticolError::compatibility(
                "native eager DirectTable execution requires the f64-compiled feature",
            ));
        }
        #[cfg(feature = "f64-compiled")]
        {
            let library_path = library.display_path();
            let metadata_symbol_name = format!("{function_name}_metadata_v1");
            let metadata_call = unsafe {
                library
                    .get::<NativeEagerDirectTableMetadataCall>(metadata_symbol_name.as_bytes())
                    .map(|symbol| *symbol)
                    .map_err(|error| {
                        RusticolError::compatibility(format!(
                            "could not load native eager DirectTable metadata symbol {metadata_symbol_name:?} from {}: {error}",
                            library_path.display()
                        ))
                    })?
            };
            let exported = unsafe { metadata_call() };
            if exported.is_null() {
                return Err(RusticolError::integrity(format!(
                    "native eager DirectTable {} returned null ABI metadata",
                    display_path.display()
                )));
            }
            let exported = unsafe { &*exported };
            let expected_struct_size =
                u32::try_from(std::mem::size_of::<NativeEagerDirectTableMetadataV1>())
                    .expect("native eager metadata size fits u32");
            if exported.struct_size != expected_struct_size
                || exported.abi_version != NATIVE_EAGER_DIRECT_TABLE_METADATA_VERSION
                || exported.flags != NATIVE_EAGER_DIRECT_TABLE_REQUIRED_FLAGS
                || exported.invocation_stride != invocation_stride
                || exported.attachment_stride != attachment_stride
                || exported.input_complex_count != input_complex_count
                || exported.output_complex_count != output_complex_count
                || exported.simd_lane_width != expected_simd_lane_width
            {
                return Err(RusticolError::compatibility(format!(
                    "native eager DirectTable {} exports incompatible ABI metadata",
                    display_path.display()
                )));
            }
            for (label, pointer, expected) in [
                (
                    "application ABI",
                    exported.application_abi,
                    source_application_abi,
                ),
                ("function name", exported.function_name, function_name),
                (
                    "target triple",
                    exported.target_triple,
                    expected_target_triple,
                ),
                (
                    "evaluator-state digest",
                    exported.evaluator_state_sha256,
                    expected_evaluator_state_sha256,
                ),
            ] {
                let actual = native_metadata_text(pointer, label, &display_path)?;
                if actual != expected {
                    return Err(RusticolError::compatibility(format!(
                        "native eager DirectTable {} {label} is {actual:?}, expected {expected:?}",
                        display_path.display()
                    )));
                }
            }
            let call = unsafe {
                library
                    .get::<NativeEagerDirectTableCall>(function_name.as_bytes())
                    .map(|symbol| *symbol)
                    .map_err(|error| {
                        RusticolError::evaluation(format!(
                            "could not load native eager DirectTable symbol {function_name:?} from {}: {error}",
                            library_path.display()
                        ))
                    })?
            };
            Ok(Self {
                callable: EagerDirectTableCallable::Native(LoadedNativeEagerDirectTable {
                    _library: library,
                    call,
                    metadata,
                }),
                display_path,
            })
        }
    }

    fn metadata(&self) -> EagerDirectTableMetadata {
        match &self.callable {
            EagerDirectTableCallable::Symjit { metadata, .. } => *metadata,
            #[cfg(feature = "f64-compiled")]
            EagerDirectTableCallable::Native(callable) => callable.metadata,
        }
    }

    pub(crate) fn load_rows(
        &self,
        invocation_bytes: Vec<u8>,
        attachment_bytes: Vec<u8>,
    ) -> RusticolResult<EagerDirectTableRows> {
        parse_rows(self.metadata(), invocation_bytes, attachment_bytes)
    }

    /// Validate immutable rows, catalogs, factors, hazards, and maximum range.
    /// P-kernel descriptor/scratch capacity is also allocated here, never hot.
    pub(crate) fn validate_call(
        &self,
        rows: &EagerDirectTableRows,
        workspace: &mut EagerDirectTableWorkspace,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<()> {
        let metadata = self.metadata();
        validate_call(
            metadata,
            rows,
            workspace,
            point_start,
            point_count,
            &self.display_path,
        )?;
        if matches!(&self.callable, EagerDirectTableCallable::Symjit { .. }) {
            workspace.prepare_symjit(metadata)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn evaluate_checked(
        &self,
        rows: &EagerDirectTableRows,
        workspace: &mut EagerDirectTableWorkspace,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<()> {
        self.validate_call(rows, workspace, point_start, point_count)?;
        unsafe { self.evaluate_validated_unchecked(rows, workspace, point_start, point_count) }
    }

    /// Execute rows previously authenticated by [`Self::validate_call`].
    ///
    /// # Safety
    ///
    /// Rows, plane/factor catalogs, and arena allocation addresses must remain
    /// unchanged. Scalar/factor values and active point ranges may change.
    pub(crate) unsafe fn evaluate_validated_unchecked(
        &self,
        rows: &EagerDirectTableRows,
        workspace: &mut EagerDirectTableWorkspace,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<()> {
        let metadata = self.metadata();
        let _ = workspace.checked_range(point_start, point_count)?;
        match &self.callable {
            EagerDirectTableCallable::Symjit { kernel, .. } => execute_symjit_rows(
                kernel,
                metadata,
                rows,
                workspace,
                point_start as usize,
                point_count as usize,
            ),
            #[cfg(feature = "f64-compiled")]
            EagerDirectTableCallable::Native(callable) => {
                let view = workspace.native_view(rows, metadata, point_start, point_count)?;
                let status = guard_dependency_panic(
                    || unsafe { (callable.call)(ptr::from_ref(&view)) },
                    &self.display_path,
                    "execute native table call",
                )?;
                map_direct_status(status, &self.display_path)
            }
        }
    }
}

fn validate_portable_abis(
    source_application_abi: &str,
    descriptor_abi: &str,
    binding_abi: &str,
) -> RusticolResult<()> {
    if source_application_abi != EAGER_DIRECT_SOURCE_APPLICATION_ABI
        || descriptor_abi != EAGER_DIRECT_TABLE_DESCRIPTOR_ABI
        || binding_abi != EAGER_DIRECT_TABLE_BINDING_ABI
    {
        return Err(RusticolError::compatibility(format!(
            "unsupported eager plane-table artifact ABIs ({source_application_abi:?}, {descriptor_abi:?}, {binding_abi:?}); regenerate the prepared model"
        )));
    }
    Ok(())
}

fn parse_rows(
    metadata: EagerDirectTableMetadata,
    invocation_bytes: Vec<u8>,
    attachment_bytes: Vec<u8>,
) -> RusticolResult<EagerDirectTableRows> {
    let invocation_count = table_row_count(
        &invocation_bytes,
        metadata.invocation_stride,
        "invocation",
        false,
    )?;
    let attachment_count = table_row_count(
        &attachment_bytes,
        metadata.attachment_stride,
        "attachment",
        true,
    )?;
    let mut invocations = Vec::with_capacity(invocation_count as usize);
    let mut input_planes =
        Vec::with_capacity(invocation_count as usize * metadata.input_plane_count);
    for row in 0..invocation_count as usize {
        let input_start = input_planes.len();
        for input in 0..metadata.input_plane_count {
            input_planes.push(read_row_u32(
                &invocation_bytes,
                row,
                metadata.invocation_stride as usize,
                input * 4,
            )?);
        }
        let attachment_start = read_row_u32(
            &invocation_bytes,
            row,
            metadata.invocation_stride as usize,
            metadata.attachment_start_offset(),
        )?;
        let row_attachment_count = read_row_u32(
            &invocation_bytes,
            row,
            metadata.invocation_stride as usize,
            metadata.attachment_count_offset(),
        )?;
        let stop = attachment_start
            .checked_add(row_attachment_count)
            .ok_or_else(|| {
                RusticolError::integrity("eager invocation attachment range overflows")
            })?;
        if stop > attachment_count {
            return Err(RusticolError::integrity(format!(
                "eager invocation {row} attachment range {attachment_start}..{stop} exceeds {attachment_count}"
            )));
        }
        invocations.push(ParsedInvocation {
            input_start,
            attachment_start,
            attachment_count: row_attachment_count,
        });
    }

    let mut attachments = Vec::with_capacity(attachment_count as usize);
    let mut destination_planes =
        Vec::with_capacity(attachment_count as usize * metadata.output_plane_count);
    for row in 0..attachment_count as usize {
        let destination_start = destination_planes.len();
        for output in 0..metadata.output_plane_count {
            destination_planes.push(read_row_u32(
                &attachment_bytes,
                row,
                metadata.attachment_stride as usize,
                output * 4,
            )?);
        }
        attachments.push(ParsedAttachment {
            destination_start,
            factor: read_row_u32(
                &attachment_bytes,
                row,
                metadata.attachment_stride as usize,
                metadata.scale_offset(),
            )?,
            operation: read_row_u32(
                &attachment_bytes,
                row,
                metadata.attachment_stride as usize,
                metadata.operation_offset(),
            )?,
        });
    }
    Ok(EagerDirectTableRows {
        invocations_raw: invocation_bytes.into_boxed_slice(),
        attachments_raw: attachment_bytes.into_boxed_slice(),
        invocations: invocations.into_boxed_slice(),
        input_planes: input_planes.into_boxed_slice(),
        attachments: attachments.into_boxed_slice(),
        destination_planes: destination_planes.into_boxed_slice(),
        input_plane_count: metadata.input_plane_count,
        output_plane_count: metadata.output_plane_count,
    })
}

fn read_row_u32(bytes: &[u8], row: usize, stride: usize, offset: usize) -> RusticolResult<u32> {
    let start = row
        .checked_mul(stride)
        .and_then(|start| start.checked_add(offset))
        .ok_or_else(|| RusticolError::integrity("eager row field offset overflows"))?;
    let field = bytes
        .get(start..start + 4)
        .ok_or_else(|| RusticolError::integrity("eager row field is truncated"))?;
    Ok(u32::from_le_bytes(field.try_into().map_err(|_| {
        RusticolError::internal("could not decode eager row field")
    })?))
}

fn validate_call(
    metadata: EagerDirectTableMetadata,
    rows: &EagerDirectTableRows,
    workspace: &EagerDirectTableWorkspace,
    point_start: u32,
    point_count: u32,
    display_path: &Path,
) -> RusticolResult<()> {
    if rows.input_plane_count != metadata.input_plane_count
        || rows.output_plane_count != metadata.output_plane_count
    {
        return Err(RusticolError::integrity(
            "eager rows were loaded for a different P-kernel shape",
        ));
    }
    let (_, point_stop) = workspace.checked_range(point_start, point_count)?;
    if !workspace.scalars.is_empty() {
        return Err(RusticolError::integrity(
            "eager P-kernel plane tables do not accept scalar descriptors",
        ));
    }
    if workspace.factor_re.is_empty() || workspace.factor_re.len() != workspace.factor_im.len() {
        return Err(RusticolError::integrity(
            "eager factor catalog is not a nonempty split-complex pair",
        ));
    }
    for (index, plane) in workspace.planes.iter().enumerate() {
        if plane.values.is_null() || plane.len < point_stop {
            return Err(RusticolError::integrity(format!(
                "eager plane {index} does not cover the validated point range"
            )));
        }
    }
    for (row_index, invocation) in rows.invocations.iter().enumerate() {
        let inputs = invocation_inputs(rows, *invocation, metadata);
        for (binding, &plane) in inputs.iter().enumerate() {
            if plane as usize >= workspace.planes.len() {
                return Err(RusticolError::integrity(format!(
                    "eager invocation {row_index} input binding {binding} references plane {plane}, catalog has {}",
                    workspace.planes.len()
                )));
            }
        }
        for attachment_index in
            invocation.attachment_start..invocation.attachment_start + invocation.attachment_count
        {
            let attachment = rows.attachments[attachment_index as usize];
            let destinations = attachment_destinations(rows, attachment, metadata);
            for (component, &plane) in destinations.iter().enumerate() {
                if plane as usize >= workspace.planes.len() {
                    return Err(RusticolError::integrity(format!(
                        "eager attachment {attachment_index} destination component {component} references plane {plane}, catalog has {}",
                        workspace.planes.len()
                    )));
                }
                if workspace.read_only_zero_plane_index == Some(plane) {
                    return Err(RusticolError::integrity(format!(
                        "eager attachment {attachment_index} destination component {component} \
                         references read-only shared-zero plane {plane}"
                    )));
                }
            }
            if attachment.factor as usize >= workspace.factor_re.len() {
                return Err(RusticolError::integrity(format!(
                    "eager attachment {attachment_index} factor {} exceeds {}",
                    attachment.factor,
                    workspace.factor_re.len()
                )));
            }
            if !matches!(
                attachment.operation,
                EAGER_DESTINATION_OVERWRITE | EAGER_DESTINATION_ACCUMULATE
            ) {
                return Err(RusticolError::integrity(format!(
                    "eager attachment {attachment_index} operation {} is not overwrite(0) or accumulate(1)",
                    attachment.operation
                )));
            }
        }
        validate_invocation_hazards(row_index, *invocation, metadata, rows, workspace)?;
    }
    let _ = display_path;
    Ok(())
}

fn validate_invocation_hazards(
    row_index: usize,
    invocation: ParsedInvocation,
    metadata: EagerDirectTableMetadata,
    rows: &EagerDirectTableRows,
    workspace: &EagerDirectTableWorkspace,
) -> RusticolResult<()> {
    let inputs = invocation_inputs(rows, invocation, metadata);
    for (binding, &input_index) in inputs.iter().enumerate() {
        let input = plane_range(workspace.planes[input_index as usize])?;
        for attachment_index in
            invocation.attachment_start..invocation.attachment_start + invocation.attachment_count
        {
            let attachment = rows.attachments[attachment_index as usize];
            for (component, &destination_index) in
                attachment_destinations(rows, attachment, metadata)
                    .iter()
                    .enumerate()
            {
                if input.overlaps(plane_range(workspace.planes[destination_index as usize])?) {
                    return Err(RusticolError::integrity(format!(
                        "eager invocation {row_index} input binding {binding} overlaps attachment {attachment_index} destination component {component}"
                    )));
                }
            }
        }
    }

    let destination_count = invocation.attachment_count as usize * metadata.output_plane_count;
    for left in 0..destination_count {
        let left_attachment =
            invocation.attachment_start as usize + left / metadata.output_plane_count;
        let left_component = left % metadata.output_plane_count;
        let left_row = rows.attachments[left_attachment];
        let left_index = attachment_destinations(rows, left_row, metadata)[left_component] as usize;
        let left_range = plane_range(workspace.planes[left_index])?;
        for right in left + 1..destination_count {
            let right_attachment =
                invocation.attachment_start as usize + right / metadata.output_plane_count;
            let right_component = right % metadata.output_plane_count;
            let right_row = rows.attachments[right_attachment];
            let right_index =
                attachment_destinations(rows, right_row, metadata)[right_component] as usize;
            if left_range.overlaps(plane_range(workspace.planes[right_index])?) {
                return Err(RusticolError::integrity(format!(
                    "eager invocation {row_index} attachment {left_attachment} destination component {left_component} overlaps attachment {right_attachment} destination component {right_component}"
                )));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AddressRange {
    start: usize,
    end: usize,
}

impl AddressRange {
    const fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

fn plane_range(plane: EagerDirectPlane) -> RusticolResult<AddressRange> {
    let bytes = plane
        .len
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| RusticolError::integrity("eager plane byte range overflows"))?;
    let start = plane.values.addr();
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| RusticolError::integrity("eager plane address range overflows"))?;
    Ok(AddressRange { start, end })
}

fn invocation_inputs(
    rows: &EagerDirectTableRows,
    invocation: ParsedInvocation,
    metadata: EagerDirectTableMetadata,
) -> &[u32] {
    &rows.input_planes[invocation.input_start..invocation.input_start + metadata.input_plane_count]
}

fn attachment_destinations(
    rows: &EagerDirectTableRows,
    attachment: ParsedAttachment,
    metadata: EagerDirectTableMetadata,
) -> &[u32] {
    &rows.destination_planes
        [attachment.destination_start..attachment.destination_start + metadata.output_plane_count]
}

fn execute_symjit_rows(
    kernel: &SymjitPlaneKernel,
    metadata: EagerDirectTableMetadata,
    rows: &EagerDirectTableRows,
    workspace: &mut EagerDirectTableWorkspace,
    point_start: usize,
    point_count: usize,
) -> RusticolResult<()> {
    let stride = workspace.arena.point_stride() as usize;
    debug_assert!(
        workspace.symjit_descriptors.capacity()
            >= metadata.input_plane_count + metadata.output_plane_count
    );
    debug_assert!(workspace.symjit_scratch.len() >= metadata.output_plane_count * stride);

    for invocation in rows.invocations.iter().copied() {
        workspace.symjit_descriptors.clear();
        for &plane_index in invocation_inputs(rows, invocation, metadata) {
            let plane = workspace.planes[plane_index as usize];
            workspace
                .symjit_descriptors
                .push(unsafe { PlaneDescriptor::from_cached_raw_parts(plane.values, plane.len)? });
        }

        let direct_attachment = (invocation.attachment_count == 1)
            .then(|| rows.attachments[invocation.attachment_start as usize])
            .filter(|attachment| {
                attachment.operation == EAGER_DESTINATION_OVERWRITE
                    && workspace.factor_re[attachment.factor as usize].to_bits()
                        == 1.0_f64.to_bits()
                    && workspace.factor_im[attachment.factor as usize].to_bits()
                        == 0.0_f64.to_bits()
            });
        if let Some(attachment) = direct_attachment {
            for &plane_index in attachment_destinations(rows, attachment, metadata) {
                let plane = workspace.planes[plane_index as usize];
                workspace.symjit_descriptors.push(unsafe {
                    PlaneDescriptor::from_cached_raw_parts(plane.values, plane.len)?
                });
            }
        } else {
            let scratch = workspace.symjit_scratch.as_mut_ptr();
            for output in 0..metadata.output_plane_count {
                workspace.symjit_descriptors.push(unsafe {
                    PlaneDescriptor::from_cached_raw_parts(scratch.add(output * stride), stride)?
                });
            }
        }

        let execution = unsafe {
            kernel.execute_raw_descriptors_unchecked(
                &workspace.symjit_descriptors,
                point_start,
                point_count,
            )
        };
        workspace.symjit_descriptors.clear();
        execution?;

        if direct_attachment.is_none() {
            apply_scratch_fanout(
                metadata,
                invocation,
                rows,
                workspace,
                point_start,
                point_count,
            );
        }
    }
    Ok(())
}

fn apply_scratch_fanout(
    metadata: EagerDirectTableMetadata,
    invocation: ParsedInvocation,
    rows: &EagerDirectTableRows,
    workspace: &mut EagerDirectTableWorkspace,
    point_start: usize,
    point_count: usize,
) {
    let stride = workspace.arena.point_stride() as usize;
    let output_bytes = (metadata.output_plane_count as u64)
        .saturating_mul(point_count as u64)
        .saturating_mul(std::mem::size_of::<f64>() as u64);
    let read_bytes = output_bytes.saturating_mul(invocation.attachment_count as u64);
    workspace.internal_scratch_bytes = workspace
        .internal_scratch_bytes
        .saturating_add(output_bytes)
        .saturating_add(read_bytes);

    for attachment_index in
        invocation.attachment_start..invocation.attachment_start + invocation.attachment_count
    {
        let attachment = rows.attachments[attachment_index as usize];
        let factor_re = workspace.factor_re[attachment.factor as usize];
        let factor_im = workspace.factor_im[attachment.factor as usize];
        for (output, destination_indices) in attachment_destinations(rows, attachment, metadata)
            .chunks_exact(2)
            .enumerate()
        {
            let source_re = unsafe {
                std::slice::from_raw_parts(
                    workspace
                        .symjit_scratch
                        .as_ptr()
                        .add(2 * output * stride + point_start),
                    point_count,
                )
            };
            let source_im = unsafe {
                std::slice::from_raw_parts(
                    workspace
                        .symjit_scratch
                        .as_ptr()
                        .add((2 * output + 1) * stride + point_start),
                    point_count,
                )
            };
            let destination_re_plane = workspace.planes[destination_indices[0] as usize];
            let destination_im_plane = workspace.planes[destination_indices[1] as usize];
            let destination_re = unsafe {
                std::slice::from_raw_parts_mut(
                    destination_re_plane.values.add(point_start),
                    point_count,
                )
            };
            let destination_im = unsafe {
                std::slice::from_raw_parts_mut(
                    destination_im_plane.values.add(point_start),
                    point_count,
                )
            };
            apply_complex_scaled(
                destination_re,
                destination_im,
                source_re,
                source_im,
                factor_re,
                factor_im,
                attachment.operation == EAGER_DESTINATION_ACCUMULATE,
            );
        }
    }
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

#[cfg(feature = "f64-compiled")]
fn native_metadata_text(
    pointer: *const c_char,
    label: &str,
    display_path: &Path,
) -> RusticolResult<String> {
    if pointer.is_null() {
        return Err(RusticolError::integrity(format!(
            "native eager DirectTable {} has a null {label}",
            display_path.display()
        )));
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|error| {
            RusticolError::integrity(format!(
                "native eager DirectTable {} has non-UTF-8 {label}: {error}",
                display_path.display()
            ))
        })
}

fn validate_c_symbol(symbol: &str) -> RusticolResult<()> {
    let mut bytes = symbol.bytes();
    let Some(first) = bytes.next() else {
        return Err(RusticolError::artifact(
            "native eager DirectTable symbol must not be empty",
        ));
    };
    if !(first == b'_' || first.is_ascii_alphabetic())
        || bytes.any(|byte| !(byte == b'_' || byte.is_ascii_alphanumeric()))
    {
        return Err(RusticolError::artifact(format!(
            "native eager DirectTable symbol {symbol:?} is not a portable C identifier"
        )));
    }
    Ok(())
}

fn bind_arena_plane(
    arena: DirectArenaView,
    binding: EagerDirectArenaPlaneBinding,
) -> RusticolResult<EagerDirectPlane> {
    let (base, shape, component, label) = match binding {
        EagerDirectArenaPlaneBinding::CurrentReal(component) => (
            arena.current_re,
            arena.current_shape()?,
            component,
            "current real",
        ),
        EagerDirectArenaPlaneBinding::CurrentImag(component) => (
            arena.current_im,
            arena.current_shape()?,
            component,
            "current imaginary",
        ),
        EagerDirectArenaPlaneBinding::AmplitudeReal(component) => (
            arena.amplitude_re,
            arena.amplitude_shape()?,
            component,
            "amplitude real",
        ),
        EagerDirectArenaPlaneBinding::AmplitudeImag(component) => (
            arena.amplitude_im,
            arena.amplitude_shape()?,
            component,
            "amplitude imaginary",
        ),
    };
    let range = shape.checked_component_range(component, 1, label)?;
    Ok(EagerDirectPlane {
        values: unsafe { base.add(range.start) },
        len: range.len(),
    })
}

fn table_row_count(
    bytes: &[u8],
    row_stride: u32,
    label: &str,
    allow_empty: bool,
) -> RusticolResult<u32> {
    let stride = row_stride as usize;
    if stride == 0 || !bytes.len().is_multiple_of(stride) {
        return Err(RusticolError::integrity(format!(
            "eager Direct-Arena {label} table has {} bytes, not a whole number of {stride}-byte rows",
            bytes.len()
        )));
    }
    let count = u32::try_from(bytes.len() / stride).map_err(|_| {
        RusticolError::integrity(format!("eager Direct-Arena {label} row count exceeds u32"))
    })?;
    if count == 0 && !allow_empty {
        return Err(RusticolError::integrity(format!(
            "eager Direct-Arena {label} table must not be empty"
        )));
    }
    Ok(count)
}

fn map_direct_status(status: i32, path: &Path) -> RusticolResult<()> {
    match status {
        DIRECT_STATUS_OK => Ok(()),
        DIRECT_STATUS_INVALID_ARGUMENT => Err(RusticolError::integrity(format!(
            "eager Direct-Arena callable {} rejected its authenticated call view",
            path.display()
        ))),
        DIRECT_STATUS_INVALID_CONTEXT => Err(RusticolError::internal(format!(
            "eager Direct-Arena callable {} has an invalid context",
            path.display()
        ))),
        DIRECT_STATUS_EXECUTION_FAILED => Err(RusticolError::evaluation(format!(
            "eager Direct-Arena callable {} failed during generated execution",
            path.display()
        ))),
        other => Err(RusticolError::evaluation(format!(
            "eager Direct-Arena callable {} returned unknown status {other}",
            path.display()
        ))),
    }
}

fn guard_dependency_panic<T>(
    operation: impl FnOnce() -> T,
    path: &Path,
    action: &str,
) -> RusticolResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
        RusticolError::compatibility(format!(
            "dependency panicked while trying to {action} eager Direct-Arena application {}: {}",
            path.display(),
            panic_detail(payload)
        ))
    })
}

fn panic_detail(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::count_allocations;
    use std::mem;
    use symjit::{Compiler, CompilerType, Config, Storage};

    const MODEL: &str = "
([('add', ('temp', 0), [('param', 0), ('param', 1)], 0),
  ('assign', ('out', 0), ('temp', 0))],
 2,
 [])
";
    const LENGTHS: &[usize] = &[1, 2, 3, 7, 8, 127, 128, 129, 1023, 1024, 1025];
    const ODD_TAIL_LENGTHS: &[usize] = &[127, 129, 1023, 1025];

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct InvocationRow {
        x_re: u32,
        x_im: u32,
        y_re: u32,
        y_im: u32,
        attachment_start: u32,
        attachment_count: u32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct AttachmentRow {
        destination_re: u32,
        destination_im: u32,
        factor: u32,
        operation: u32,
    }

    fn source_bytes() -> Vec<u8> {
        let mut config = Config::new(CompilerType::Native, 0).unwrap();
        config.set_opt_level(2);
        config.set_cse(true);
        config.set_fastmath(true);
        config.set_simd(true);
        config.enable_simd512(false);
        config.set_simd_branch(false);
        config.set_complex(true);
        config.set_fast_complex(false);
        config.set_threads(false);
        config.set_symbolica(true);
        config.set_compact(true);
        config.set_compress(true);
        config.set_direct(false);
        config.set_huge(false);
        config.set_parallel_mul(true);
        config.set_direct_arena(true);
        config.set_direct_arena_operation(0);
        config.set_direct_arena_identity_output(true);
        let mut application = Compiler::with_config(config)
            .translate(MODEL.to_string(), 2)
            .unwrap();
        application.prepare_simd();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn table_fixture(
        point_capacity: u32,
    ) -> (
        LoadedSymjitEagerDirectTable,
        EagerDirectTableRows,
        EagerDirectTableWorkspace,
    ) {
        let source = source_bytes();
        let descriptor = eager_direct_descriptor_for_source_application_bytes(
            &source,
            2,
            1,
            Path::new("eager-table-test.symjit"),
        )
        .unwrap();
        let application = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source,
            &descriptor,
            PathBuf::from("eager-table-test.symjit"),
            EAGER_DIRECT_SOURCE_APPLICATION_ABI,
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            2,
            true,
        )
        .unwrap();
        let invocations = [
            InvocationRow {
                x_re: 0,
                x_im: 1,
                y_re: 2,
                y_im: 3,
                attachment_start: 0,
                attachment_count: 2,
            },
            InvocationRow {
                x_re: 4,
                x_im: 5,
                y_re: 6,
                y_im: 7,
                attachment_start: 2,
                attachment_count: 1,
            },
        ];
        let attachments = [
            AttachmentRow {
                destination_re: 4,
                destination_im: 5,
                factor: 0,
                operation: 0,
            },
            AttachmentRow {
                destination_re: 6,
                destination_im: 7,
                factor: 1,
                operation: 0,
            },
            AttachmentRow {
                destination_re: 8,
                destination_im: 9,
                factor: 0,
                operation: 0,
            },
        ];
        let rows = application
            .load_rows(rows_as_bytes(&invocations), rows_as_bytes(&attachments))
            .unwrap();
        let bindings = [
            EagerDirectArenaPlaneBinding::CurrentReal(0),
            EagerDirectArenaPlaneBinding::CurrentImag(0),
            EagerDirectArenaPlaneBinding::CurrentReal(1),
            EagerDirectArenaPlaneBinding::CurrentImag(1),
            EagerDirectArenaPlaneBinding::CurrentReal(2),
            EagerDirectArenaPlaneBinding::CurrentImag(2),
            EagerDirectArenaPlaneBinding::CurrentReal(3),
            EagerDirectArenaPlaneBinding::CurrentImag(3),
            EagerDirectArenaPlaneBinding::AmplitudeReal(0),
            EagerDirectArenaPlaneBinding::AmplitudeImag(0),
        ];
        let workspace = EagerDirectTableWorkspace::new(
            4,
            1,
            point_capacity,
            &bindings,
            None,
            Vec::new(),
            vec![1.0, 0.5],
            vec![0.0, 0.0],
        )
        .unwrap();
        (application, rows, workspace)
    }

    fn shared_zero_fixture(
        point_capacity: u32,
    ) -> (
        LoadedSymjitEagerDirectTable,
        EagerDirectTableRows,
        EagerDirectTableWorkspace,
    ) {
        let source = source_bytes();
        let descriptor = eager_direct_descriptor_for_source_application_bytes(
            &source,
            2,
            1,
            Path::new("eager-shared-zero-test.symjit"),
        )
        .unwrap();
        let application = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source,
            &descriptor,
            PathBuf::from("eager-shared-zero-test.symjit"),
            EAGER_DIRECT_SOURCE_APPLICATION_ABI,
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            2,
            true,
        )
        .unwrap();
        let invocations = [InvocationRow {
            x_re: 0,
            x_im: 1,
            y_re: 0,
            y_im: 1,
            attachment_start: 0,
            attachment_count: 1,
        }];
        let attachments = [AttachmentRow {
            destination_re: 2,
            destination_im: 3,
            factor: 0,
            operation: 0,
        }];
        let rows = application
            .load_rows(rows_as_bytes(&invocations), rows_as_bytes(&attachments))
            .unwrap();
        let bindings = [
            EagerDirectArenaPlaneBinding::CurrentReal(0),
            EagerDirectArenaPlaneBinding::CurrentImag(0),
            EagerDirectArenaPlaneBinding::AmplitudeReal(0),
            EagerDirectArenaPlaneBinding::AmplitudeImag(0),
        ];
        let workspace = EagerDirectTableWorkspace::new(
            1,
            1,
            point_capacity,
            &bindings,
            Some(1),
            Vec::new(),
            vec![1.0],
            vec![0.0],
        )
        .unwrap();
        (application, rows, workspace)
    }

    fn rows_as_bytes<T: Copy>(rows: &[T]) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), mem::size_of_val(rows)) }
            .to_vec()
    }

    fn initialize_inputs(workspace: &mut EagerDirectTableWorkspace, points: usize) {
        workspace.begin_tile(points as u32).unwrap();
        let stride = workspace.arena().point_stride() as usize;
        let (current_re, current_im, amplitude_re, amplitude_im) =
            workspace.split_arena_slices_mut();
        for point in 0..points {
            current_re[point] = 1.0 + point as f64;
            current_im[point] = -0.25 * point as f64;
            current_re[stride + point] = 10.0 + 2.0 * point as f64;
            current_im[stride + point] = 0.5 * point as f64;
        }
        current_re[2 * stride..4 * stride].fill(-99.0);
        current_im[2 * stride..4 * stride].fill(77.0);
        amplitude_re.fill(-55.0);
        amplitude_im.fill(44.0);
    }

    fn assert_outputs(workspace: &EagerDirectTableWorkspace, points: usize) {
        let stride = workspace.arena().point_stride() as usize;
        let (current_re, current_im) = workspace.arena().current_slices();
        let (amplitude_re, amplitude_im) = workspace.arena().amplitude_slices();
        for point in 0..points {
            let sum_re = 11.0 + 3.0 * point as f64;
            let sum_im = 0.25 * point as f64;
            assert_eq!(current_re[2 * stride + point], sum_re);
            assert_eq!(current_im[2 * stride + point], sum_im);
            assert_eq!(current_re[3 * stride + point], 0.5 * sum_re);
            assert_eq!(current_im[3 * stride + point], 0.5 * sum_im);
            assert_eq!(amplitude_re[point], 1.5 * sum_re);
            assert_eq!(amplitude_im[point], 1.5 * sum_im);
        }
    }

    #[test]
    fn ordered_fanout_cross_row_dependency_and_odd_tails() {
        let (application, rows, mut workspace) = table_fixture(1025);
        for &points in LENGTHS {
            initialize_inputs(&mut workspace, points);
            application
                .validate_call(&rows, &mut workspace, 0, points as u32)
                .unwrap();
            unsafe {
                application
                    .evaluate_validated_unchecked(&rows, &mut workspace, 0, points as u32)
                    .unwrap();
            }
            assert_outputs(&workspace, points);
            assert!(workspace.internal_scratch_bytes() > 0);
        }
    }

    #[test]
    fn complex_factors_and_accumulation_preserve_tails() {
        let (application, _, mut workspace) = table_fixture(1025);
        workspace.factor_re.copy_from_slice(&[0.5, -0.75]);
        workspace.factor_im.copy_from_slice(&[-0.25, 0.5]);
        let invocations = [InvocationRow {
            x_re: 0,
            x_im: 1,
            y_re: 2,
            y_im: 3,
            attachment_start: 0,
            attachment_count: 2,
        }];
        let attachments = [
            AttachmentRow {
                destination_re: 4,
                destination_im: 5,
                factor: 0,
                operation: EAGER_DESTINATION_OVERWRITE,
            },
            AttachmentRow {
                destination_re: 8,
                destination_im: 9,
                factor: 1,
                operation: EAGER_DESTINATION_ACCUMULATE,
            },
        ];
        let rows = application
            .load_rows(rows_as_bytes(&invocations), rows_as_bytes(&attachments))
            .unwrap();

        for &points in LENGTHS {
            initialize_inputs(&mut workspace, points);
            application
                .evaluate_checked(&rows, &mut workspace, 0, points as u32)
                .unwrap();
            let stride = workspace.arena().point_stride() as usize;
            let (current_re, current_im) = workspace.arena().current_slices();
            let (amplitude_re, amplitude_im) = workspace.arena().amplitude_slices();
            for point in 0..points {
                let source_re = 11.0 + 3.0 * point as f64;
                let source_im = 0.25 * point as f64;
                assert_eq!(
                    current_re[2 * stride + point],
                    0.5 * source_re + 0.25 * source_im
                );
                assert_eq!(
                    current_im[2 * stride + point],
                    0.5 * source_im - 0.25 * source_re
                );
                assert_eq!(
                    amplitude_re[point],
                    -55.0 - 0.75 * source_re - 0.5 * source_im
                );
                assert_eq!(
                    amplitude_im[point],
                    44.0 - 0.75 * source_im + 0.5 * source_re
                );
            }
            if points < stride {
                assert_eq!(current_re[2 * stride + points], -99.0);
                assert_eq!(current_im[2 * stride + points], 77.0);
                assert_eq!(amplitude_re[points], -55.0);
                assert_eq!(amplitude_im[points], 44.0);
            }
            assert!(workspace.internal_scratch_bytes() > 0);
        }
    }

    #[test]
    fn duplicate_real_only_inputs_share_a_read_only_zero_plane() {
        let (application, rows, mut workspace) = shared_zero_fixture(1025);
        for &points in LENGTHS {
            workspace.begin_tile(points as u32).unwrap();
            let stride = workspace.arena().point_stride() as usize;
            {
                let (current_re, current_im, amplitude_re, amplitude_im) =
                    workspace.split_arena_slices_mut();
                current_re.fill(-77.0);
                current_im.fill(0.0);
                amplitude_re.fill(-55.0);
                amplitude_im.fill(44.0);
                for (point, value) in current_re[..points].iter_mut().enumerate() {
                    *value = 1.0 + point as f64;
                }
            }
            application
                .validate_call(&rows, &mut workspace, 0, points as u32)
                .unwrap();
            unsafe {
                application
                    .evaluate_validated_unchecked(&rows, &mut workspace, 0, points as u32)
                    .unwrap();
            }
            let (_, current_im) = workspace.arena().current_slices();
            let (amplitude_re, amplitude_im) = workspace.arena().amplitude_slices();
            assert!(
                current_im.iter().all(|value| *value == 0.0),
                "shared-zero plane changed for {points} points"
            );
            for point in 0..points {
                assert_eq!(amplitude_re[point], 2.0 * (1.0 + point as f64));
                assert_eq!(amplitude_im[point], 0.0);
            }
            assert!(
                amplitude_re[points..stride]
                    .iter()
                    .all(|value| *value == -55.0),
                "real sentinel tail changed for {points} points"
            );
            assert!(
                amplitude_im[points..stride]
                    .iter()
                    .all(|value| *value == 44.0),
                "imaginary sentinel tail changed for {points} points"
            );
        }
    }

    #[test]
    fn shared_zero_destination_is_rejected_cold() {
        let (application, _, mut workspace) = shared_zero_fixture(7);
        let invocations = [InvocationRow {
            x_re: 0,
            x_im: 1,
            y_re: 0,
            y_im: 1,
            attachment_start: 0,
            attachment_count: 1,
        }];
        let attachments = [AttachmentRow {
            destination_re: 1,
            destination_im: 3,
            factor: 0,
            operation: 0,
        }];
        let rows = application
            .load_rows(rows_as_bytes(&invocations), rows_as_bytes(&attachments))
            .unwrap();
        let error = application
            .validate_call(&rows, &mut workspace, 0, 7)
            .unwrap_err();
        assert!(error.message().contains("read-only shared-zero plane"));
    }

    #[test]
    fn same_row_input_destination_hazard_is_rejected() {
        let (application, _, mut workspace) = table_fixture(7);
        let invocations = [InvocationRow {
            x_re: 0,
            x_im: 1,
            y_re: 2,
            y_im: 3,
            attachment_start: 0,
            attachment_count: 1,
        }];
        let attachments = [AttachmentRow {
            destination_re: 0,
            destination_im: 1,
            factor: 0,
            operation: 0,
        }];
        let rows = application
            .load_rows(rows_as_bytes(&invocations), rows_as_bytes(&attachments))
            .unwrap();
        let error = application
            .validate_call(&rows, &mut workspace, 0, 7)
            .unwrap_err();
        assert!(error.message().contains("overlaps"));
    }

    #[test]
    fn warmed_checked_and_validated_calls_allocate_zero() {
        let (application, rows, mut workspace) = table_fixture(1025);
        for &points in ODD_TAIL_LENGTHS {
            initialize_inputs(&mut workspace, points);
            application
                .evaluate_checked(&rows, &mut workspace, 0, points as u32)
                .unwrap();
            application
                .validate_call(&rows, &mut workspace, 0, points as u32)
                .unwrap();
            let (checked, checked_count, checked_bytes) = count_allocations(|| {
                application.evaluate_checked(&rows, &mut workspace, 0, points as u32)
            });
            checked.unwrap();
            assert_eq!(
                (checked_count, checked_bytes),
                (0, 0),
                "checked eager call allocated for {points} points"
            );
            let (unchecked, unchecked_count, unchecked_bytes) = count_allocations(|| unsafe {
                application.evaluate_validated_unchecked(&rows, &mut workspace, 0, points as u32)
            });
            unchecked.unwrap();
            assert_eq!(
                (unchecked_count, unchecked_bytes),
                (0, 0),
                "validated eager call allocated for {points} points"
            );
            assert_outputs(&workspace, points);
        }
    }

    #[test]
    fn authenticated_variant_optimization_level_mismatch_fails_closed() {
        let source = source_bytes();
        let descriptor = eager_direct_descriptor_for_source_application_bytes(
            &source,
            2,
            1,
            Path::new("descriptor.symjit"),
        )
        .unwrap();
        let error = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source,
            &descriptor,
            PathBuf::from("wrong-optimization.symjit"),
            EAGER_DIRECT_SOURCE_APPLICATION_ABI,
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            3,
            true,
        )
        .err()
        .expect("a variant/application optimization mismatch must fail closed");
        assert!(error.message().contains("optimization level"));
        assert!(
            error
                .message()
                .contains("authenticated prepared-kernel variant")
        );
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn authenticated_manifest_compression_mismatch_fails_closed() {
        let source = source_bytes();
        let descriptor = eager_direct_descriptor_for_source_application_bytes(
            &source,
            2,
            1,
            Path::new("descriptor.symjit"),
        )
        .unwrap();
        let error = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source,
            &descriptor,
            PathBuf::from("wrong-compression.symjit"),
            EAGER_DIRECT_SOURCE_APPLICATION_ABI,
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            2,
            false,
        )
        .err()
        .expect("a manifest/application compression mismatch must fail closed");
        assert!(error.message().contains("compression"));
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn descriptor_authentication_and_predecessor_abis_fail_closed() {
        let source = source_bytes();
        let descriptor = eager_direct_descriptor_for_source_application_bytes(
            &source,
            2,
            1,
            Path::new("descriptor.symjit"),
        )
        .unwrap();
        let mut tampered = descriptor.clone();
        tampered[40] ^= 1;
        let error = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source,
            &tampered,
            PathBuf::from("descriptor.symjit"),
            EAGER_DIRECT_SOURCE_APPLICATION_ABI,
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            2,
            true,
        )
        .err()
        .unwrap();
        assert!(error.message().contains("authenticate"));
        let error = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source,
            &descriptor,
            PathBuf::from("old-binding.symjit"),
            EAGER_DIRECT_SOURCE_APPLICATION_ABI,
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            concat!("pyamplicol-eager-plane-", "table-binding-v1"),
            2,
            true,
        )
        .err()
        .expect("the predecessor eager binding ABI must fail closed");
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.message().contains("regenerate"));
        let error = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source,
            &descriptor,
            PathBuf::from("old-source-application.symjit"),
            "pyamplicol-symjit-plane-application-v1",
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            2,
            true,
        )
        .err()
        .expect("the predecessor plane-application ABI must fail closed");
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(error.message().contains("regenerate"));
        let error = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source,
            b"SJTD0001",
            PathBuf::from("old.symjit"),
            EAGER_DIRECT_SOURCE_APPLICATION_ABI,
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            EAGER_DIRECT_TABLE_BINDING_ABI,
            2,
            true,
        )
        .err()
        .unwrap();
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn dependency_panics_are_contained_with_eager_application_context() {
        let error = guard_dependency_panic(
            || panic!("synthetic table panic"),
            Path::new("panic-probe.symjit"),
            "exercise the panic-containment probe",
        )
        .unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        assert!(
            error.message().contains(
                "dependency panicked while trying to exercise the panic-containment probe"
            )
        );
        assert!(error.message().contains("panic-probe.symjit"));
        assert!(error.message().contains("synthetic table panic"));
    }
}
