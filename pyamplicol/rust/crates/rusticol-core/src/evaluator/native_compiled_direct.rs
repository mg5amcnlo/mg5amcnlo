// SPDX-License-Identifier: 0BSD

//! Target-native DirectApplication adapter for one fused compiled leaf.
//!
//! C++ and inline-ASM compiled artifacts are both target-native dynamic
//! libraries, so they share this small C ABI.  Unlike the historical
//! `params/buffer/out` entry point, the native function consumes fixed
//! split-complex plane descriptors and writes canonical output planes itself.
//! It therefore does not hide a dense-row gather/call/scatter wrapper.
//!
//! This module is a bounded ABI prototype and is intentionally not wired into
//! the production engine yet.

#![allow(dead_code)]

use std::path::Path;
use std::ptr;
use std::sync::Arc;

use crate::artifact::PinnedNativeLibrary;
use crate::direct_arena::{
    AlignedF64Buffer, DirectArenaTrafficCounters, DirectArenaView, DirectFactorView,
    DirectMomentumView, DirectParameterView, validate_direct_views,
};
use crate::{RusticolError, RusticolResult};

pub(crate) const NATIVE_COMPILED_DIRECT_APPLICATION_ABI: &str =
    "pyamplicol-native-compiled-direct-application-v1";

const NATIVE_COMPILED_DIRECT_ABI_VERSION: u32 = 1;
const NATIVE_COMPILED_DIRECT_FLAG_SPLIT_COMPLEX: u32 = 1 << 0;
const NATIVE_COMPILED_DIRECT_FLAG_COMPONENT_MAJOR: u32 = 1 << 1;
const NATIVE_COMPILED_DIRECT_FLAG_POINT_CONTIGUOUS: u32 = 1 << 2;
const NATIVE_COMPILED_DIRECT_FLAG_FACTOR_FREE_OVERWRITE: u32 = 1 << 3;
const NATIVE_COMPILED_DIRECT_FLAG_NO_OUTPUT_ALIAS: u32 = 1 << 4;
const NATIVE_COMPILED_DIRECT_FLAG_NOEXCEPT: u32 = 1 << 5;
const NATIVE_COMPILED_DIRECT_REQUIRED_FLAGS: u32 = NATIVE_COMPILED_DIRECT_FLAG_SPLIT_COMPLEX
    | NATIVE_COMPILED_DIRECT_FLAG_COMPONENT_MAJOR
    | NATIVE_COMPILED_DIRECT_FLAG_POINT_CONTIGUOUS
    | NATIVE_COMPILED_DIRECT_FLAG_FACTOR_FREE_OVERWRITE
    | NATIVE_COMPILED_DIRECT_FLAG_NO_OUTPUT_ALIAS
    | NATIVE_COMPILED_DIRECT_FLAG_NOEXCEPT;
const NATIVE_COMPILED_DIRECT_KNOWN_FLAGS: u32 = NATIVE_COMPILED_DIRECT_REQUIRED_FLAGS;
const NATIVE_COMPILED_DIRECT_STATUS_OK: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NativeCompiledDirectMetadataV1 {
    abi_version: u32,
    struct_size: u32,
    flags: u32,
    input_plane_count: u32,
    scalar_input_count: u32,
    output_plane_count: u32,
    simd_lane_width: u32,
    reserved: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativeCompiledDirectDescriptorCounts {
    input_planes: u32,
    scalar_inputs: u32,
    output_planes: u32,
}

impl NativeCompiledDirectDescriptorCounts {
    fn checked_from_lengths(
        input_planes: usize,
        scalar_inputs: usize,
        output_planes: usize,
    ) -> RusticolResult<Self> {
        Ok(Self {
            input_planes: checked_descriptor_count(input_planes, "input-plane")?,
            scalar_inputs: checked_descriptor_count(scalar_inputs, "scalar-input")?,
            output_planes: checked_descriptor_count(output_planes, "output-plane")?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeCompiledDirectInputPlaneV1 {
    values: *const f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeCompiledDirectOutputPlaneV1 {
    values: *mut f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeCompiledDirectScalarV1 {
    value: *const f64,
}

type NativeCompiledDirectMetadataFunction =
    unsafe extern "C" fn() -> NativeCompiledDirectMetadataV1;
type NativeCompiledDirectEvaluateFunction = unsafe extern "C" fn(
    *const NativeCompiledDirectInputPlaneV1,
    u32,
    *const NativeCompiledDirectScalarV1,
    u32,
    *const NativeCompiledDirectOutputPlaneV1,
    u32,
    u32,
    u32,
) -> i32;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum NativeCompiledDirectArenaPlane {
    Current { component: u32, imaginary: bool },
    Amplitude { component: u32, imaginary: bool },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum NativeCompiledDirectPlaneBinding {
    Arena(NativeCompiledDirectArenaPlane),
    Momentum { form: u32, lorentz_component: u16 },
    Zero,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum NativeCompiledDirectScalarBinding {
    Parameter { index: u32, imaginary: bool },
    Literal(f64),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct NativeCompiledDirectOutputBinding(pub NativeCompiledDirectArenaPlane);

/// Loaded native function plus its authenticated logical bindings.
pub(crate) struct LoadedNativeCompiledDirectStage {
    _library: Arc<PinnedNativeLibrary>,
    evaluate: NativeCompiledDirectEvaluateFunction,
    descriptor_counts: NativeCompiledDirectDescriptorCounts,
    declared_lane_width: usize,
    input_bindings: Box<[NativeCompiledDirectPlaneBinding]>,
    scalar_bindings: Box<[NativeCompiledDirectScalarBinding]>,
    output_bindings: Box<[NativeCompiledDirectOutputBinding]>,
    display_path: Box<Path>,
}

/// Fixed descriptors borrowing persistent Direct-Arena allocations.
pub(crate) struct BoundNativeCompiledDirectStage {
    _library: Arc<PinnedNativeLibrary>,
    evaluate: NativeCompiledDirectEvaluateFunction,
    inputs: Box<[NativeCompiledDirectInputPlaneV1]>,
    scalars: Box<[NativeCompiledDirectScalarV1]>,
    outputs: Box<[NativeCompiledDirectOutputPlaneV1]>,
    _literal_values: Box<[f64]>,
    _zero_plane: AlignedF64Buffer,
    input_count: u32,
    scalar_count: u32,
    output_count: u32,
    point_stride: u32,
    declared_lane_width: usize,
    display_path: Box<Path>,
}

impl LoadedNativeCompiledDirectStage {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        library: Arc<PinnedNativeLibrary>,
        function_name: &str,
        application_abi: &str,
        input_bindings: Vec<NativeCompiledDirectPlaneBinding>,
        scalar_bindings: Vec<NativeCompiledDirectScalarBinding>,
        output_bindings: Vec<NativeCompiledDirectOutputBinding>,
    ) -> RusticolResult<Self> {
        if application_abi != NATIVE_COMPILED_DIRECT_APPLICATION_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported native compiled DirectApplication ABI {application_abi:?}; expected \
                 {NATIVE_COMPILED_DIRECT_APPLICATION_ABI:?}"
            )));
        }
        validate_c_symbol(function_name)?;
        let descriptor_counts = NativeCompiledDirectDescriptorCounts::checked_from_lengths(
            input_bindings.len(),
            scalar_bindings.len(),
            output_bindings.len(),
        )?;
        validate_static_bindings(&input_bindings, &output_bindings)?;

        let display_path = library.display_path().to_path_buf().into_boxed_path();
        let path = display_path.as_ref();
        let symbol_prefix = format!("{function_name}_direct_application_v1");
        let metadata_symbol = format!("{symbol_prefix}_metadata");
        let metadata_function = unsafe {
            load_export::<NativeCompiledDirectMetadataFunction>(&library, path, &metadata_symbol)?
        };
        let evaluate = unsafe {
            load_export::<NativeCompiledDirectEvaluateFunction>(&library, path, &symbol_prefix)?
        };
        // SAFETY: the authenticated producer contract assigns this symbol the
        // fixed metadata signature. Native exports must be `noexcept`; foreign
        // unwinding across this C boundary is not recoverable by Rust.
        let metadata = unsafe { metadata_function() };
        validate_metadata(metadata, descriptor_counts, path)?;

        Ok(Self {
            _library: library,
            evaluate,
            descriptor_counts,
            declared_lane_width: metadata.simd_lane_width as usize,
            input_bindings: input_bindings.into_boxed_slice(),
            scalar_bindings: scalar_bindings.into_boxed_slice(),
            output_bindings: output_bindings.into_boxed_slice(),
            display_path,
        })
    }

    pub(crate) const fn declared_lane_width(&self) -> usize {
        self.declared_lane_width
    }

    /// Bind the native function to persistent plane and scalar storage.
    ///
    /// # Safety
    ///
    /// Every allocation referenced by `arena`, `momenta`, and `parameters`
    /// must retain its address and declared length until the returned object
    /// drops. During `evaluate`, the caller must provide exclusive access to
    /// arena outputs and must not mutate any bound input or scalar storage.
    pub(crate) unsafe fn bind(
        self,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
    ) -> RusticolResult<BoundNativeCompiledDirectStage> {
        // SAFETY: this entry point preserves the historical identity mapping
        // and forwards the caller's complete fixed-storage contract.
        unsafe { self.bind_with_current_map(arena, momenta, parameters, &[]) }
    }

    /// Bind canonical current IDs through a cold, immutable physical map.
    pub(crate) unsafe fn bind_with_current_map(
        self,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        current_component_map: &[u32],
    ) -> RusticolResult<BoundNativeCompiledDirectStage> {
        validate_direct_views(
            arena,
            momenta,
            parameters,
            DirectFactorView {
                values_re: ptr::null(),
                values_im: ptr::null(),
                value_count: 0,
            },
        )?;
        validate_mapped_bindings(
            &self.input_bindings,
            &self.output_bindings,
            current_component_map,
        )?;

        let point_stride = arena.point_stride;
        let zero_plane =
            AlignedF64Buffer::zeroed(point_stride as usize, "native compiled zero plane")?;
        let mut literal_values = Vec::new();
        literal_values
            .try_reserve_exact(self.scalar_bindings.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate native compiled literal storage: {error}"
                ))
            })?;
        literal_values.resize(self.scalar_bindings.len(), 0.0);
        let mut literal_values = literal_values.into_boxed_slice();
        for (index, binding) in self.scalar_bindings.iter().copied().enumerate() {
            if let NativeCompiledDirectScalarBinding::Literal(value) = binding {
                if !value.is_finite() {
                    return Err(RusticolError::integrity(
                        "native compiled DirectApplication literal must be finite",
                    ));
                }
                literal_values[index] = value;
            }
        }

        let inputs = self
            .input_bindings
            .iter()
            .copied()
            .map(|binding| {
                resolve_input_plane(binding, arena, momenta, &zero_plane, current_component_map)
                    .map(|values| NativeCompiledDirectInputPlaneV1 { values })
            })
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        let outputs = self
            .output_bindings
            .iter()
            .copied()
            .map(|binding| {
                resolve_arena_plane(binding.0, arena, current_component_map)
                    .map(|values| NativeCompiledDirectOutputPlaneV1 { values })
            })
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        let scalars = self
            .scalar_bindings
            .iter()
            .copied()
            .enumerate()
            .map(|(literal_index, binding)| {
                let value = match binding {
                    NativeCompiledDirectScalarBinding::Parameter { index, imaginary } => {
                        if index >= parameters.value_count {
                            return Err(RusticolError::integrity(format!(
                                "native compiled parameter binding {index} is out of bounds"
                            )));
                        }
                        let base = if imaginary {
                            parameters.values_im
                        } else {
                            parameters.values_re
                        };
                        // SAFETY: the parameter view was validated and `index`
                        // lies within its declared storage.
                        unsafe { base.add(index as usize) }
                    }
                    NativeCompiledDirectScalarBinding::Literal(_) => {
                        ptr::from_ref(&literal_values[literal_index])
                    }
                };
                Ok(NativeCompiledDirectScalarV1 { value })
            })
            .collect::<RusticolResult<Vec<_>>>()?
            .into_boxed_slice();
        let descriptor_counts = NativeCompiledDirectDescriptorCounts::checked_from_lengths(
            inputs.len(),
            scalars.len(),
            outputs.len(),
        )?;
        if descriptor_counts != self.descriptor_counts {
            return Err(RusticolError::internal(
                "native compiled descriptor counts changed while binding persistent storage",
            ));
        }

        Ok(BoundNativeCompiledDirectStage {
            _library: self._library,
            evaluate: self.evaluate,
            inputs,
            scalars,
            outputs,
            _literal_values: literal_values,
            _zero_plane: zero_plane,
            input_count: descriptor_counts.input_planes,
            scalar_count: descriptor_counts.scalar_inputs,
            output_count: descriptor_counts.output_planes,
            point_stride,
            declared_lane_width: self.declared_lane_width,
            display_path: self.display_path,
        })
    }
}

impl BoundNativeCompiledDirectStage {
    pub(crate) const fn declared_lane_width(&self) -> usize {
        self.declared_lane_width
    }

    /// Invoke the authenticated native function without allocating or packing.
    pub(crate) fn evaluate(&self, point_start: u32, point_count: u32) -> RusticolResult<()> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "native compiled DirectApplication point count must be positive",
            ));
        }
        let point_stop = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument(
                "native compiled DirectApplication point range overflows u32",
            )
        })?;
        if point_stop > self.point_stride {
            return Err(RusticolError::invalid_argument(format!(
                "native compiled DirectApplication point range {point_start}..{point_stop} \
                 exceeds physical stride {}",
                self.point_stride
            )));
        }

        // SAFETY: load and bind authenticated the function signature and fixed
        // descriptors. The producer contract requires `noexcept` and status
        // returns; Rust cannot contain foreign unwinding across this C ABI.
        let status = unsafe {
            (self.evaluate)(
                self.inputs.as_ptr(),
                self.input_count,
                self.scalars.as_ptr(),
                self.scalar_count,
                self.outputs.as_ptr(),
                self.output_count,
                point_start,
                point_count,
            )
        };
        if status != NATIVE_COMPILED_DIRECT_STATUS_OK {
            return Err(RusticolError::evaluation(format!(
                "native compiled DirectApplication {} returned status {status}",
                self.display_path.display()
            )));
        }
        Ok(())
    }

    /// Profile adapter-side traffic without adding work to the unprofiled path.
    ///
    /// These counters describe Rust-side materialization around the call. They
    /// cannot observe allocation or copying hidden inside a native producer.
    pub(crate) fn evaluate_profiled(
        &self,
        point_start: u32,
        point_count: u32,
        traffic: &mut DirectArenaTrafficCounters,
    ) -> RusticolResult<()> {
        self.evaluate(point_start, point_count)?;
        traffic.record_call(1, point_count);
        traffic.validate_direct()
    }
}

fn validate_metadata(
    metadata: NativeCompiledDirectMetadataV1,
    descriptor_counts: NativeCompiledDirectDescriptorCounts,
    path: &Path,
) -> RusticolResult<()> {
    if metadata.abi_version != NATIVE_COMPILED_DIRECT_ABI_VERSION
        || metadata.struct_size as usize != std::mem::size_of::<NativeCompiledDirectMetadataV1>()
        || metadata.reserved != 0
    {
        return Err(RusticolError::compatibility(format!(
            "native compiled DirectApplication {} has incompatible v1 metadata",
            path.display()
        )));
    }
    if metadata.flags & NATIVE_COMPILED_DIRECT_REQUIRED_FLAGS
        != NATIVE_COMPILED_DIRECT_REQUIRED_FLAGS
        || metadata.flags & !NATIVE_COMPILED_DIRECT_KNOWN_FLAGS != 0
    {
        return Err(RusticolError::compatibility(format!(
            "native compiled DirectApplication {} does not certify split-plane, \
             point-contiguous, factor-free overwrite, no-alias and noexcept semantics",
            path.display()
        )));
    }
    for (actual, expected, label) in [
        (
            metadata.input_plane_count,
            descriptor_counts.input_planes,
            "input plane",
        ),
        (
            metadata.scalar_input_count,
            descriptor_counts.scalar_inputs,
            "scalar input",
        ),
        (
            metadata.output_plane_count,
            descriptor_counts.output_planes,
            "output plane",
        ),
    ] {
        if actual != expected {
            return Err(RusticolError::integrity(format!(
                "native compiled DirectApplication {} {label} count {actual} does not match \
                 binding count {expected}",
                path.display()
            )));
        }
    }
    if metadata.simd_lane_width == 0 || !metadata.simd_lane_width.is_power_of_two() {
        return Err(RusticolError::compatibility(format!(
            "native compiled DirectApplication {} has invalid declared lane width {}",
            path.display(),
            metadata.simd_lane_width
        )));
    }
    Ok(())
}

fn checked_descriptor_count(count: usize, label: &str) -> RusticolResult<u32> {
    u32::try_from(count).map_err(|_| {
        RusticolError::integrity(format!(
            "native compiled DirectApplication {label} count {count} exceeds u32"
        ))
    })
}

fn validate_static_bindings(
    inputs: &[NativeCompiledDirectPlaneBinding],
    outputs: &[NativeCompiledDirectOutputBinding],
) -> RusticolResult<()> {
    if outputs.is_empty() {
        return Err(RusticolError::integrity(
            "native compiled DirectApplication must expose an output",
        ));
    }
    for (index, output) in outputs.iter().copied().enumerate() {
        if outputs[..index].contains(&output) {
            return Err(RusticolError::integrity(format!(
                "native compiled output binding {index} aliases an earlier output"
            )));
        }
        if inputs.contains(&NativeCompiledDirectPlaneBinding::Arena(output.0)) {
            return Err(RusticolError::integrity(format!(
                "native compiled output binding {index} aliases an input plane"
            )));
        }
    }
    Ok(())
}

fn validate_mapped_bindings(
    inputs: &[NativeCompiledDirectPlaneBinding],
    outputs: &[NativeCompiledDirectOutputBinding],
    current_component_map: &[u32],
) -> RusticolResult<()> {
    let mapped_inputs = inputs
        .iter()
        .copied()
        .filter_map(|binding| match binding {
            NativeCompiledDirectPlaneBinding::Arena(arena) => {
                Some(map_arena_plane(arena, current_component_map))
            }
            NativeCompiledDirectPlaneBinding::Momentum { .. }
            | NativeCompiledDirectPlaneBinding::Zero => None,
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut mapped_outputs = Vec::with_capacity(outputs.len());
    for (index, output) in outputs.iter().copied().enumerate() {
        let mapped = map_arena_plane(output.0, current_component_map)?;
        if mapped_outputs.contains(&mapped) {
            return Err(RusticolError::integrity(format!(
                "native compiled mapped output binding {index} aliases an earlier output"
            )));
        }
        if mapped_inputs.contains(&mapped) {
            return Err(RusticolError::integrity(format!(
                "native compiled mapped output binding {index} aliases an input plane"
            )));
        }
        mapped_outputs.push(mapped);
    }
    Ok(())
}

fn map_arena_plane(
    binding: NativeCompiledDirectArenaPlane,
    current_component_map: &[u32],
) -> RusticolResult<NativeCompiledDirectArenaPlane> {
    match binding {
        NativeCompiledDirectArenaPlane::Current {
            component,
            imaginary,
        } if !current_component_map.is_empty() => {
            let physical = current_component_map
                .get(component as usize)
                .copied()
                .filter(|physical| *physical != u32::MAX)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "native compiled current binding {component} has no physical mapping"
                    ))
                })?;
            Ok(NativeCompiledDirectArenaPlane::Current {
                component: physical,
                imaginary,
            })
        }
        _ => Ok(binding),
    }
}

fn resolve_input_plane(
    binding: NativeCompiledDirectPlaneBinding,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    zero_plane: &AlignedF64Buffer,
    current_component_map: &[u32],
) -> RusticolResult<*const f64> {
    match binding {
        NativeCompiledDirectPlaneBinding::Arena(binding) => {
            Ok(resolve_arena_plane(binding, arena, current_component_map)?.cast_const())
        }
        NativeCompiledDirectPlaneBinding::Momentum {
            form,
            lorentz_component,
        } => {
            if form >= momenta.form_count || lorentz_component >= momenta.lorentz_component_count {
                return Err(RusticolError::integrity(format!(
                    "native compiled momentum binding ({form}, {lorentz_component}) is out of \
                     bounds"
                )));
            }
            let component = form
                .checked_mul(u32::from(momenta.lorentz_component_count))
                .and_then(|base| base.checked_add(u32::from(lorentz_component)))
                .ok_or_else(|| {
                    RusticolError::integrity("native compiled momentum component overflows u32")
                })?;
            let offset = usize::try_from(component)
                .ok()
                .and_then(|component| component.checked_mul(momenta.point_stride as usize))
                .ok_or_else(|| {
                    RusticolError::integrity("native compiled momentum offset overflows usize")
                })?;
            // SAFETY: view and component bounds were validated.
            Ok(unsafe { momenta.values.add(offset) })
        }
        NativeCompiledDirectPlaneBinding::Zero => Ok(zero_plane.as_ptr()),
    }
}

fn resolve_arena_plane(
    binding: NativeCompiledDirectArenaPlane,
    arena: DirectArenaView,
    current_component_map: &[u32],
) -> RusticolResult<*mut f64> {
    let binding = map_arena_plane(binding, current_component_map)?;
    let (base, shape, label) = match binding {
        NativeCompiledDirectArenaPlane::Current {
            component,
            imaginary,
        } => (
            if imaginary {
                arena.current_im
            } else {
                arena.current_re
            },
            arena.current_shape()?,
            ("current", component),
        ),
        NativeCompiledDirectArenaPlane::Amplitude {
            component,
            imaginary,
        } => (
            if imaginary {
                arena.amplitude_im
            } else {
                arena.amplitude_re
            },
            arena.amplitude_shape()?,
            ("amplitude", component),
        ),
    };
    let offset = shape.checked_scalar_index(label.1, 0, label.0)?;
    // SAFETY: the checked component proves a complete pitched plane exists.
    Ok(unsafe { base.add(offset) })
}

fn validate_c_symbol(symbol: &str) -> RusticolResult<()> {
    let mut bytes = symbol.bytes();
    let Some(first) = bytes.next() else {
        return Err(RusticolError::invalid_argument(
            "native compiled DirectApplication function name must not be empty",
        ));
    };
    if !(first == b'_' || first.is_ascii_alphabetic())
        || bytes.any(|byte| !(byte == b'_' || byte.is_ascii_alphanumeric()))
    {
        return Err(RusticolError::invalid_argument(format!(
            "native compiled DirectApplication function name {symbol:?} is not a portable C \
             identifier"
        )));
    }
    Ok(())
}

unsafe fn load_export<T: Copy>(
    library: &libloading::Library,
    path: &Path,
    exported_symbol: &str,
) -> RusticolResult<T> {
    // SAFETY: the producer contract assigns each fixed symbol the exact v1
    // signature selected by the caller.
    unsafe {
        library
            .get::<T>(exported_symbol.as_bytes())
            .map(|symbol| *symbol)
            .map_err(|error| {
                RusticolError::evaluation(format!(
                    "could not load native compiled DirectApplication symbol \
                     {exported_symbol:?} from {}: {error}",
                    path.display()
                ))
            })
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::super::CompiledComplexF64Evaluator;
    #[cfg(not(feature = "f64-symjit"))]
    use super::super::native_direct::tests::count_allocations;
    #[cfg(feature = "f64-symjit")]
    use super::super::symjit_direct::tests::count_allocations;
    use super::*;
    use crate::RusticolErrorKind;
    use crate::direct_arena::{DirectArenaWorkspace, DirectMomentumView};
    #[cfg(not(feature = "symbolica-runtime"))]
    use num_complex::Complex;
    use std::fs;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    #[cfg(feature = "symbolica-runtime")]
    use symbolica::prelude::Complex;

    const POINT_COUNT: usize = 129;
    const REQUIRED_FLAGS: u32 = NATIVE_COMPILED_DIRECT_REQUIRED_FLAGS;
    static FIXTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn native_fixture() -> (std::path::PathBuf, std::path::PathBuf) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "rusticol-native-compiled-direct-test-{}-{unique}-{}",
            std::process::id(),
            FIXTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(&directory).unwrap();
        let source = directory.join("fixture.cpp");
        let library = directory.join(if cfg!(target_os = "macos") {
            "libfixture.dylib"
        } else {
            "libfixture.so"
        });
        fs::write(
            &source,
            format!(
                r#"#include <complex>
#include <cstdint>
#include <cstddef>

struct Metadata {{
  std::uint32_t abi_version;
  std::uint32_t struct_size;
  std::uint32_t flags;
  std::uint32_t input_plane_count;
  std::uint32_t scalar_input_count;
  std::uint32_t output_plane_count;
  std::uint32_t simd_lane_width;
  std::uint32_t reserved;
}};
struct InputPlane {{ const double* values; }};
struct OutputPlane {{ double* values; }};
struct Scalar {{ const double* value; }};

extern "C" unsigned long rusticol_native_leaf_complexf64_get_buffer_len() {{
  return 1ul;
}}
extern "C" void rusticol_native_leaf_complexf64(
    const std::complex<double>* params,
    std::complex<double>* buffer,
    std::complex<double>* out) {{
  buffer[0] = params[0] * params[1];
  out[0] = buffer[0] + params[2];
  out[1] = params[0] + params[1];
}}

extern "C" Metadata
rusticol_native_leaf_direct_application_v1_metadata() noexcept {{
  return Metadata{{1u, sizeof(Metadata), {REQUIRED_FLAGS}u, 4u, 2u, 4u, 2u, 0u}};
}}

// This is a synthetic scalar plane-loop fixture. It addresses planes directly,
// never calls the dense function, and never constructs a params/buffer/out row.
// The declared width of 2 exercises metadata plumbing; it is not proof that
// this fixture contains SIMD or production compressed-O3 code.
extern "C" int rusticol_native_leaf_direct_application_v1(
    const InputPlane* inputs,
    std::uint32_t input_count,
    const Scalar* scalars,
    std::uint32_t scalar_count,
    const OutputPlane* outputs,
    std::uint32_t output_count,
    std::uint32_t point_start,
    std::uint32_t point_count) noexcept {{
  if (inputs == nullptr || outputs == nullptr || scalars == nullptr ||
      input_count != 4u || scalar_count != 2u || output_count != 4u ||
      point_count == 0u) return 2;
  for (std::uint32_t offset = 0; offset < point_count; ++offset) {{
    const std::uint32_t point = point_start + offset;
    const double x_re = inputs[0].values[point];
    const double x_im = inputs[1].values[point];
    const double y_re = inputs[2].values[point];
    const double y_im = inputs[3].values[point];
    const double product_re = x_re * y_re - x_im * y_im;
    const double product_im = x_re * y_im + x_im * y_re;
    outputs[0].values[point] = product_re + *scalars[0].value;
    outputs[1].values[point] = product_im + *scalars[1].value;
    outputs[2].values[point] = x_re + y_re;
    outputs[3].values[point] = x_im + y_im;
  }}
  return 0;
}}

extern "C" Metadata
rusticol_status_leaf_direct_application_v1_metadata() noexcept {{
  return Metadata{{1u, sizeof(Metadata), {REQUIRED_FLAGS}u, 4u, 2u, 4u, 1u, 0u}};
}}
extern "C" int rusticol_status_leaf_direct_application_v1(
    const InputPlane*, std::uint32_t, const Scalar*, std::uint32_t,
    const OutputPlane*, std::uint32_t, std::uint32_t, std::uint32_t) noexcept {{
  return 17;
}}
"#
            ),
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
            "could not compile native direct fixture: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        (directory, library)
    }

    fn input_bindings() -> Vec<NativeCompiledDirectPlaneBinding> {
        vec![
            NativeCompiledDirectPlaneBinding::Arena(NativeCompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: false,
            }),
            NativeCompiledDirectPlaneBinding::Arena(NativeCompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: true,
            }),
            NativeCompiledDirectPlaneBinding::Arena(NativeCompiledDirectArenaPlane::Current {
                component: 1,
                imaginary: false,
            }),
            NativeCompiledDirectPlaneBinding::Arena(NativeCompiledDirectArenaPlane::Current {
                component: 1,
                imaginary: true,
            }),
        ]
    }

    fn scalar_bindings() -> Vec<NativeCompiledDirectScalarBinding> {
        vec![
            NativeCompiledDirectScalarBinding::Parameter {
                index: 0,
                imaginary: false,
            },
            NativeCompiledDirectScalarBinding::Parameter {
                index: 0,
                imaginary: true,
            },
        ]
    }

    fn output_bindings() -> Vec<NativeCompiledDirectOutputBinding> {
        vec![
            NativeCompiledDirectOutputBinding(NativeCompiledDirectArenaPlane::Current {
                component: 2,
                imaginary: false,
            }),
            NativeCompiledDirectOutputBinding(NativeCompiledDirectArenaPlane::Current {
                component: 2,
                imaginary: true,
            }),
            NativeCompiledDirectOutputBinding(NativeCompiledDirectArenaPlane::Current {
                component: 3,
                imaginary: false,
            }),
            NativeCompiledDirectOutputBinding(NativeCompiledDirectArenaPlane::Current {
                component: 3,
                imaginary: true,
            }),
        ]
    }

    fn empty_momenta(point_stride: u32) -> AlignedF64Buffer {
        AlignedF64Buffer::zeroed(4 * point_stride as usize, "native compiled test momenta").unwrap()
    }

    fn momentum_view(values: &AlignedF64Buffer, point_stride: u32) -> DirectMomentumView {
        DirectMomentumView {
            values: values.as_ptr(),
            scalar_len: values.len() as u64,
            form_count: 1,
            lorentz_component_count: 4,
            point_stride,
        }
    }

    fn fill_inputs(workspace: &mut DirectArenaWorkspace, point_count: usize) {
        let stride = workspace.point_stride() as usize;
        let (current_re, current_im, _, _) = workspace.split_slices_mut();
        for point in 0..point_count {
            current_re[point] = 0.25 + point as f64 * 0.03125;
            current_im[point] = if point % 2 == 0 { -0.0 } else { 0.5 };
            current_re[stride + point] = 1.5 - point as f64 * 0.0078125;
            current_im[stride + point] = if point % 3 == 0 { 0.0 } else { -0.25 };
        }
    }

    fn load_and_bind(
        library: &Path,
        function_name: &str,
        workspace: &mut DirectArenaWorkspace,
        momenta: &AlignedF64Buffer,
        parameter_re: &[f64; 1],
        parameter_im: &[f64; 1],
    ) -> RusticolResult<BoundNativeCompiledDirectStage> {
        let loaded = LoadedNativeCompiledDirectStage::load(
            PinnedNativeLibrary::from_test_path(library)?,
            function_name,
            NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
            input_bindings(),
            scalar_bindings(),
            output_bindings(),
        )?;
        let arena = workspace.view()?;
        let parameters = DirectParameterView {
            values_re: parameter_re.as_ptr(),
            values_im: parameter_im.as_ptr(),
            value_count: 1,
        };
        unsafe {
            loaded.bind(
                arena,
                momentum_view(momenta, arena.point_stride),
                parameters,
            )
        }
    }

    #[test]
    fn synthetic_plane_loop_preserves_odd_tails_and_allocates_zero_when_warm() {
        let (directory, library) = native_fixture();
        let mut workspace = DirectArenaWorkspace::new(4, 0, POINT_COUNT as u32).unwrap();
        workspace.begin_tile(POINT_COUNT as u32).unwrap();
        fill_inputs(&mut workspace, POINT_COUNT);
        let stride = workspace.point_stride() as usize;
        let momenta = empty_momenta(stride as u32);
        let parameter_re = [0.375];
        let parameter_im = [-0.625];
        let bound = load_and_bind(
            &library,
            "rusticol_native_leaf",
            &mut workspace,
            &momenta,
            &parameter_re,
            &parameter_im,
        )
        .unwrap();
        assert_eq!(bound.declared_lane_width(), 2);

        for (point_start, point_count) in [
            (0_u32, 1_u32),
            (0, 2),
            (0, 3),
            (0, 127),
            (1, 127),
            (0, 128),
            (0, 129),
        ] {
            {
                let (current_re, current_im, _, _) = workspace.split_slices_mut();
                current_re[2 * stride..4 * stride].fill(f64::NAN);
                current_im[2 * stride..4 * stride].fill(f64::NAN);
            }
            bound.evaluate(point_start, point_count).unwrap();
            let point_stop = (point_start + point_count) as usize;
            let (current_re, current_im) = workspace.current_slices();
            for point in point_start as usize..point_stop {
                let x = Complex::new(current_re[point], current_im[point]);
                let y = Complex::new(current_re[stride + point], current_im[stride + point]);
                let expected_product = x * y + Complex::new(parameter_re[0], parameter_im[0]);
                let expected_sum = x + y;
                assert_eq!(
                    current_re[2 * stride + point].to_bits(),
                    expected_product.re.to_bits()
                );
                assert_eq!(
                    current_im[2 * stride + point].to_bits(),
                    expected_product.im.to_bits()
                );
                assert_eq!(
                    current_re[3 * stride + point].to_bits(),
                    expected_sum.re.to_bits()
                );
                assert_eq!(
                    current_im[3 * stride + point].to_bits(),
                    expected_sum.im.to_bits()
                );
            }
            for point in (0..point_start as usize).chain(point_stop..stride) {
                assert!(current_re[2 * stride + point].is_nan());
                assert!(current_im[2 * stride + point].is_nan());
                assert!(current_re[3 * stride + point].is_nan());
                assert!(current_im[3 * stride + point].is_nan());
            }
        }

        bound.evaluate(0, POINT_COUNT as u32).unwrap();
        let (result, allocation_count, allocated_bytes) =
            count_allocations(|| bound.evaluate(0, POINT_COUNT as u32));
        result.unwrap();
        assert_eq!(
            allocation_count, 0,
            "warmed native direct call allocated in Rust"
        );
        assert_eq!(
            allocated_bytes, 0,
            "warmed native direct call allocated bytes in Rust"
        );
        // This accounts only for Rust adapter-side traffic. The fixture source
        // is inspected separately to establish that its native loop also
        // contains no dense-row wrapper.
        let mut traffic = DirectArenaTrafficCounters::default();
        bound
            .evaluate_profiled(0, POINT_COUNT as u32, &mut traffic)
            .unwrap();
        assert_eq!(traffic.calls, 1);
        assert_eq!(traffic.rows, 1);
        assert_eq!(traffic.points, POINT_COUNT as u64);
        assert_eq!(traffic.packet_input_bytes, 0);
        assert_eq!(traffic.packet_output_bytes, 0);
        assert_eq!(traffic.gather_bytes, 0);
        assert_eq!(traffic.scatter_bytes, 0);
        assert_eq!(traffic.remap_bytes, 0);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn synthetic_plane_loop_matches_existing_dense_gather_call_scatter() {
        let (directory, library) = native_fixture();
        let mut workspace = DirectArenaWorkspace::new(4, 0, POINT_COUNT as u32).unwrap();
        workspace.begin_tile(POINT_COUNT as u32).unwrap();
        fill_inputs(&mut workspace, POINT_COUNT);
        let stride = workspace.point_stride() as usize;
        let momenta = empty_momenta(stride as u32);
        let parameter_re = [0.375];
        let parameter_im = [-0.625];

        let mut dense = CompiledComplexF64Evaluator::load(
            PinnedNativeLibrary::from_test_path(&library).unwrap(),
            "rusticol_native_leaf",
            3,
            2,
        )
        .unwrap();
        let mut packed_inputs = vec![Complex::new(0.0, 0.0); POINT_COUNT * 3];
        let mut packed_outputs = vec![Complex::new(0.0, 0.0); POINT_COUNT * 2];
        {
            let (current_re, current_im) = workspace.current_slices();
            for point in 0..POINT_COUNT {
                packed_inputs[3 * point] = Complex::new(current_re[point], current_im[point]);
                packed_inputs[3 * point + 1] =
                    Complex::new(current_re[stride + point], current_im[stride + point]);
                packed_inputs[3 * point + 2] = Complex::new(parameter_re[0], parameter_im[0]);
            }
        }
        dense
            .evaluate_batch(POINT_COUNT, &packed_inputs, &mut packed_outputs)
            .unwrap();
        let bound = load_and_bind(
            &library,
            "rusticol_native_leaf",
            &mut workspace,
            &momenta,
            &parameter_re,
            &parameter_im,
        )
        .unwrap();
        bound.evaluate(0, POINT_COUNT as u32).unwrap();

        let (current_re, current_im) = workspace.current_slices();
        for point in 0..POINT_COUNT {
            for (component, expected) in [
                (2, packed_outputs[2 * point]),
                (3, packed_outputs[2 * point + 1]),
            ] {
                assert_eq!(
                    current_re[component * stride + point].to_bits(),
                    expected.re.to_bits()
                );
                assert_eq!(
                    current_im[component * stride + point].to_bits(),
                    expected.im.to_bits()
                );
            }
        }
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn mapped_native_bindings_reject_a_physical_input_output_alias() {
        let error = validate_mapped_bindings(&input_bindings(), &output_bindings(), &[0, 1, 0, 1])
            .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);
        assert!(error.message().contains("mapped output"));
        assert!(error.message().contains("aliases an input plane"));
    }

    #[test]
    fn aliases_bounds_and_native_status_fail_closed() {
        let (directory, library) = native_fixture();
        let mut aliased_outputs = output_bindings();
        aliased_outputs[0] =
            NativeCompiledDirectOutputBinding(NativeCompiledDirectArenaPlane::Current {
                component: 0,
                imaginary: false,
            });
        let alias_error = match LoadedNativeCompiledDirectStage::load(
            PinnedNativeLibrary::from_test_path(&library).unwrap(),
            "rusticol_native_leaf",
            NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
            input_bindings(),
            scalar_bindings(),
            aliased_outputs,
        ) {
            Ok(_) => panic!("input/output alias unexpectedly loaded"),
            Err(error) => error,
        };
        assert_eq!(alias_error.kind(), RusticolErrorKind::Integrity);
        assert!(alias_error.message().contains("aliases an input plane"));

        let mut workspace = DirectArenaWorkspace::new(4, 0, POINT_COUNT as u32).unwrap();
        workspace.begin_tile(POINT_COUNT as u32).unwrap();
        fill_inputs(&mut workspace, POINT_COUNT);
        let stride = workspace.point_stride();
        let momenta = empty_momenta(stride);
        let parameter_re = [0.375];
        let parameter_im = [-0.625];
        let bound = load_and_bind(
            &library,
            "rusticol_native_leaf",
            &mut workspace,
            &momenta,
            &parameter_re,
            &parameter_im,
        )
        .unwrap();
        assert_eq!(
            bound.evaluate(stride - 1, 2).unwrap_err().kind(),
            RusticolErrorKind::InvalidArgument
        );
        assert_eq!(
            bound.evaluate(0, 0).unwrap_err().kind(),
            RusticolErrorKind::InvalidArgument
        );

        let status_bound = load_and_bind(
            &library,
            "rusticol_status_leaf",
            &mut workspace,
            &momenta,
            &parameter_re,
            &parameter_im,
        )
        .unwrap();
        let error = status_bound.evaluate(0, 1).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Evaluation);
        assert!(error.message().contains("status 17"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn descriptor_counts_fail_before_any_truncating_ffi_conversion() {
        let maximum = NativeCompiledDirectDescriptorCounts::checked_from_lengths(
            u32::MAX as usize,
            u32::MAX as usize,
            u32::MAX as usize,
        )
        .unwrap();
        assert_eq!(maximum.input_planes, u32::MAX);
        assert_eq!(maximum.scalar_inputs, u32::MAX);
        assert_eq!(maximum.output_planes, u32::MAX);

        if usize::BITS > u32::BITS {
            let too_many = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
            for error in [
                NativeCompiledDirectDescriptorCounts::checked_from_lengths(too_many, 0, 1)
                    .unwrap_err(),
                NativeCompiledDirectDescriptorCounts::checked_from_lengths(0, too_many, 1)
                    .unwrap_err(),
                NativeCompiledDirectDescriptorCounts::checked_from_lengths(0, 0, too_many)
                    .unwrap_err(),
            ] {
                assert_eq!(error.kind(), RusticolErrorKind::Integrity);
                assert!(error.message().contains("exceeds u32"));
            }
        }
    }

    fn gather_call_scatter(
        dense: &mut CompiledComplexF64Evaluator,
        workspace: &mut DirectArenaWorkspace,
        parameter: Complex<f64>,
        packed_inputs: &mut [Complex<f64>],
        packed_outputs: &mut [Complex<f64>],
    ) {
        let stride = workspace.point_stride() as usize;
        {
            let (current_re, current_im) = workspace.current_slices();
            for point in 0..POINT_COUNT {
                packed_inputs[3 * point] = Complex::new(current_re[point], current_im[point]);
                packed_inputs[3 * point + 1] =
                    Complex::new(current_re[stride + point], current_im[stride + point]);
                packed_inputs[3 * point + 2] = parameter;
            }
        }
        dense
            .evaluate_batch(POINT_COUNT, packed_inputs, packed_outputs)
            .unwrap();
        let (current_re, current_im, _, _) = workspace.split_slices_mut();
        for point in 0..POINT_COUNT {
            let product = packed_outputs[2 * point];
            let sum = packed_outputs[2 * point + 1];
            current_re[2 * stride + point] = product.re;
            current_im[2 * stride + point] = product.im;
            current_re[3 * stride + point] = sum.re;
            current_im[3 * stride + point] = sum.im;
        }
    }

    fn median(values: &mut [u128]) -> u128 {
        values.sort_unstable();
        values[values.len() / 2]
    }

    fn median_absolute_deviation(values: &[u128], center: u128) -> u128 {
        let mut deviations = values
            .iter()
            .map(|value| value.abs_diff(center))
            .collect::<Vec<_>>();
        median(&mut deviations)
    }

    #[test]
    #[ignore = "manual interleaved native DirectApplication benchmark"]
    fn benchmark_synthetic_native_plane_loop_against_gather_call_scatter() {
        const SAMPLES: usize = 9;
        const REPEATS: usize = 10_000;

        let (directory, library) = native_fixture();
        let mut workspace = DirectArenaWorkspace::new(4, 0, POINT_COUNT as u32).unwrap();
        workspace.begin_tile(POINT_COUNT as u32).unwrap();
        fill_inputs(&mut workspace, POINT_COUNT);
        let stride = workspace.point_stride();
        let momenta = empty_momenta(stride);
        let parameter_re = [0.375];
        let parameter_im = [-0.625];
        let parameter = Complex::new(parameter_re[0], parameter_im[0]);
        let bound = load_and_bind(
            &library,
            "rusticol_native_leaf",
            &mut workspace,
            &momenta,
            &parameter_re,
            &parameter_im,
        )
        .unwrap();
        let mut dense = CompiledComplexF64Evaluator::load(
            PinnedNativeLibrary::from_test_path(&library).unwrap(),
            "rusticol_native_leaf",
            3,
            2,
        )
        .unwrap();
        let mut packed_inputs = vec![Complex::new(0.0, 0.0); POINT_COUNT * 3];
        let mut packed_outputs = vec![Complex::new(0.0, 0.0); POINT_COUNT * 2];

        bound.evaluate(0, POINT_COUNT as u32).unwrap();
        gather_call_scatter(
            &mut dense,
            &mut workspace,
            parameter,
            &mut packed_inputs,
            &mut packed_outputs,
        );
        let (_, direct_allocations, direct_bytes) =
            count_allocations(|| bound.evaluate(0, POINT_COUNT as u32));
        let (_, packed_allocations, packed_bytes) = count_allocations(|| {
            gather_call_scatter(
                &mut dense,
                &mut workspace,
                parameter,
                &mut packed_inputs,
                &mut packed_outputs,
            )
        });

        let mut direct_samples = [0_u128; SAMPLES];
        let mut packed_samples = [0_u128; SAMPLES];
        for sample in 0..SAMPLES {
            let time_direct = || {
                let started = Instant::now();
                for _ in 0..REPEATS {
                    bound.evaluate(0, POINT_COUNT as u32).unwrap();
                }
                started.elapsed().as_nanos() / REPEATS as u128
            };
            let mut time_packed = || {
                let started = Instant::now();
                for _ in 0..REPEATS {
                    gather_call_scatter(
                        &mut dense,
                        &mut workspace,
                        parameter,
                        &mut packed_inputs,
                        &mut packed_outputs,
                    );
                }
                started.elapsed().as_nanos() / REPEATS as u128
            };
            if sample.is_multiple_of(2) {
                direct_samples[sample] = time_direct();
                packed_samples[sample] = time_packed();
            } else {
                packed_samples[sample] = time_packed();
                direct_samples[sample] = time_direct();
            }
            std::hint::black_box(workspace.current_slices().0[2 * stride as usize]);
        }
        let direct_ordered = direct_samples;
        let packed_ordered = packed_samples;
        let direct_median = median(&mut direct_samples);
        let packed_median = median(&mut packed_samples);
        let direct_mad = median_absolute_deviation(&direct_ordered, direct_median);
        let packed_mad = median_absolute_deviation(&packed_ordered, packed_median);
        eprintln!(
            "synthetic-native-compiled-plane-loop points={POINT_COUNT} samples={SAMPLES} \
             repeats={REPEATS} \
             direct_ns={direct_ordered:?} packed_ns={packed_ordered:?} \
             direct_median_ns={direct_median} direct_mad_ns={direct_mad} \
             packed_median_ns={packed_median} packed_mad_ns={packed_mad} \
             speedup={:.6} direct_rust_allocations={direct_allocations} \
             direct_rust_allocated_bytes={direct_bytes} \
             packed_rust_allocations={packed_allocations} \
             packed_rust_allocated_bytes={packed_bytes}",
            packed_median as f64 / direct_median as f64,
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
