// SPDX-License-Identifier: 0BSD

//! Compiled-stage adapter for standard SymJIT direct-arena P-kernels.
//!
//! pyAmpliCol owns the fused-stage binding policy. SymJIT owns only the
//! ordinary P-kernel prologue, invariant body, and epilogue. Point-dependent
//! values bind directly to persistent arena planes. Point-independent inputs
//! bind through persistent broadcast planes which are refreshed only when the
//! backing model parameter changes.

use std::path::{Path, PathBuf};
use std::ptr;

use super::symjit_plane::{
    PlaneDescriptor, SymjitPlaneKernel, SymjitPlaneLayout, SymjitPlaneTable,
};
use crate::direct_arena::{
    AlignedF64Buffer, DirectArenaAllocationCounters, DirectArenaView, DirectFactorView,
    DirectMomentumView, DirectParameterView, validate_direct_views,
};
use crate::engine::SYMJIT_PLANE_APPLICATION_ABI;
use crate::{RusticolError, RusticolResult};

/// One split-complex plane in the persistent compiled-stage arena.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompiledDirectArenaPlane {
    Current { component: u32, imaginary: bool },
    Amplitude { component: u32, imaginary: bool },
}

/// One point-dependent input consumed by a fused compiled stage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompiledDirectPlaneBinding {
    Arena(CompiledDirectArenaPlane),
    Momentum {
        form: u32,
        lorentz_component: u16,
    },
    /// The runtime-owned persistent structural-zero plane.
    Zero,
}

/// One point-independent scalar consumed by a fused compiled stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CompiledDirectScalarBinding {
    Parameter {
        index: u32,
        imaginary: bool,
    },
    // The plane ABI supports persistent literal broadcasts even though the
    // current production lowering emits parameter bindings only.
    #[allow(dead_code)]
    Literal(f64),
}

/// One distinct output plane overwritten by a fused compiled stage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CompiledDirectOutputBinding(pub CompiledDirectArenaPlane);

/// Routing from the P-kernel's split input order to runtime plane catalogs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompiledDirectSourceInputBinding {
    Plane(u32),
    Scalar(u32),
}

/// Loaded but not yet pinned compiled-stage P-kernel.
pub(crate) struct LoadedSymjitCompiledDirectStage {
    kernel: SymjitPlaneKernel,
    source_input_bindings: Box<[CompiledDirectSourceInputBinding]>,
    input_bindings: Box<[CompiledDirectPlaneBinding]>,
    scalar_bindings: Box<[CompiledDirectScalarBinding]>,
    output_bindings: Box<[CompiledDirectOutputBinding]>,
    display_path: PathBuf,
}

#[derive(Clone, Copy)]
enum BroadcastSource {
    Parameter { value: *const f64, cached_bits: u64 },
    Literal,
}

/// A cold-bound P-kernel descriptor table for one persistent compiled arena.
///
/// The unsafe bind contract guarantees that external arena, momentum,
/// parameter, and zero-plane allocations remain fixed until this object drops.
pub(crate) struct BoundSymjitCompiledDirectStage {
    kernel: SymjitPlaneKernel,
    table: SymjitPlaneTable<'static>,
    broadcast_values: AlignedF64Buffer,
    broadcast_sources: Box<[BroadcastSource]>,
    point_stride: u32,
    pending_broadcast_fill_bytes: u64,
    display_path: PathBuf,
}

impl LoadedSymjitCompiledDirectStage {
    pub(crate) const fn broadcast_plane_count(&self) -> usize {
        self.scalar_bindings.len()
    }

    /// Load one persisted standard complex P-kernel and authenticate its
    /// pyAmpliCol-owned split-plane routing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_source_bytes(
        bytes: &[u8],
        display_path: impl AsRef<Path>,
        source_application_abi: &str,
        source_optimization_level: u8,
        source_compression: bool,
        source_input_bindings: Vec<CompiledDirectSourceInputBinding>,
        input_bindings: Vec<CompiledDirectPlaneBinding>,
        scalar_bindings: Vec<CompiledDirectScalarBinding>,
        output_bindings: Vec<CompiledDirectOutputBinding>,
    ) -> RusticolResult<Self> {
        let display_path = display_path.as_ref().to_path_buf();
        if source_application_abi != SYMJIT_PLANE_APPLICATION_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported compiled plane application ABI \
                 {source_application_abi:?}; expected \
                 {SYMJIT_PLANE_APPLICATION_ABI:?}; regenerate the artifact"
            )));
        }
        validate_static_bindings(
            &source_input_bindings,
            &input_bindings,
            &scalar_bindings,
            &output_bindings,
        )?;
        let input_complex_count = source_input_bindings.len() / 2;
        let output_complex_count = output_bindings.len() / 2;
        let kernel = SymjitPlaneKernel::load_bytes(
            bytes,
            display_path.clone(),
            SymjitPlaneLayout::complex(input_complex_count, output_complex_count),
        )?;
        if kernel.optimization_level() != source_optimization_level {
            return Err(RusticolError::integrity(format!(
                "compiled plane application {} declares optimization level \
                 {source_optimization_level} but stores optimization level {}",
                display_path.display(),
                kernel.optimization_level()
            )));
        }
        if kernel.compression() != source_compression {
            return Err(RusticolError::integrity(format!(
                "compiled plane application {} declares compression \
                 {source_compression} but stores compression {}; regenerate the artifact",
                display_path.display(),
                kernel.compression()
            )));
        }
        if kernel.input_plane_count() != source_input_bindings.len()
            || kernel.output_plane_count() != output_bindings.len()
        {
            return Err(RusticolError::integrity(format!(
                "compiled plane application {} shape disagrees with its bindings",
                display_path.display()
            )));
        }
        Ok(Self {
            kernel,
            source_input_bindings: source_input_bindings.into_boxed_slice(),
            input_bindings: input_bindings.into_boxed_slice(),
            scalar_bindings: scalar_bindings.into_boxed_slice(),
            output_bindings: output_bindings.into_boxed_slice(),
            display_path,
        })
    }

    /// Pin this stage to persistent Direct-Arena storage.
    ///
    /// # Safety
    ///
    /// Every allocation referenced by `arena`, `momenta`, `parameters`, and
    /// `zero_plane` must remain allocated at the same address until the
    /// returned stage is dropped. Parameter values may change only between
    /// calls to `evaluate`. No other access may overlap a kernel invocation.
    #[cfg(test)]
    pub(crate) unsafe fn bind(
        self,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        zero_plane: &AlignedF64Buffer,
    ) -> RusticolResult<BoundSymjitCompiledDirectStage> {
        // SAFETY: this is the identity-layout specialization of the complete
        // fixed-storage contract above.
        unsafe { self.bind_with_current_map(arena, momenta, parameters, zero_plane, &[]) }
    }

    /// Pin canonical current IDs through a cold immutable physical map.
    ///
    /// # Safety
    ///
    /// Every allocation addressed by `arena`, `momenta`, `parameters`, and
    /// `zero_plane` must remain live and at a stable address until the returned
    /// bound stage drops. `current_component_map` must remain immutable for
    /// this call. No access may race with stage execution; model-parameter
    /// updates must use the runtime's synchronized update path.
    pub(crate) unsafe fn bind_with_current_map(
        self,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        zero_plane: &AlignedF64Buffer,
        current_component_map: &[u32],
    ) -> RusticolResult<BoundSymjitCompiledDirectStage> {
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
        let stride = point_stride as usize;
        if zero_plane.len() != stride {
            return Err(RusticolError::integrity(format!(
                "compiled shared zero plane has length {}, expected {point_stride}",
                zero_plane.len()
            )));
        }
        if zero_plane.as_slice().iter().any(|value| *value != 0.0) {
            return Err(RusticolError::integrity(
                "compiled shared zero plane contains a nonzero value",
            ));
        }

        let broadcast_len = self
            .scalar_bindings
            .len()
            .checked_mul(stride)
            .ok_or_else(|| {
                RusticolError::integrity("compiled scalar broadcast storage overflows usize")
            })?;
        let mut broadcast_values =
            AlignedF64Buffer::zeroed(broadcast_len, "compiled scalar broadcasts")?;
        let mut broadcast_sources = Vec::new();
        broadcast_sources
            .try_reserve_exact(self.scalar_bindings.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate compiled scalar broadcast metadata: {error}"
                ))
            })?;
        for (slot, binding) in self.scalar_bindings.iter().copied().enumerate() {
            let (value, source) = match binding {
                CompiledDirectScalarBinding::Parameter { index, imaginary } => {
                    if index >= parameters.value_count {
                        return Err(RusticolError::integrity(format!(
                            "compiled parameter binding {index} is out of bounds"
                        )));
                    }
                    let base = if imaginary {
                        parameters.values_im
                    } else {
                        parameters.values_re
                    };
                    // SAFETY: the parameter view was validated above and the
                    // checked index lies in its fixed allocation.
                    let value_pointer = unsafe { base.add(index as usize) };
                    // SAFETY: the pointer remains live by the bind contract.
                    let initial = unsafe { *value_pointer };
                    (
                        initial,
                        BroadcastSource::Parameter {
                            value: value_pointer,
                            cached_bits: initial.to_bits(),
                        },
                    )
                }
                CompiledDirectScalarBinding::Literal(value) => (value, BroadcastSource::Literal),
            };
            let start = slot * stride;
            broadcast_values.as_mut_slice()[start..start + stride].fill(value);
            broadcast_sources.push(source);
        }

        let mut resolved_inputs = Vec::new();
        resolved_inputs
            .try_reserve_exact(self.input_bindings.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate compiled input-plane catalog: {error}"
                ))
            })?;
        for binding in self.input_bindings.iter().copied() {
            resolved_inputs.push(resolve_input_plane_pointer(
                binding,
                arena,
                momenta,
                zero_plane,
                current_component_map,
            )?);
        }

        let descriptor_count = self
            .source_input_bindings
            .len()
            .checked_add(self.output_bindings.len())
            .ok_or_else(|| {
                RusticolError::integrity("compiled plane descriptor count overflows usize")
            })?;
        let mut descriptors = Vec::new();
        descriptors
            .try_reserve_exact(descriptor_count)
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate compiled P-kernel descriptors: {error}"
                ))
            })?;
        let broadcast_base = broadcast_values.as_mut_ptr();
        for source in self.source_input_bindings.iter().copied() {
            let pointer = match source {
                CompiledDirectSourceInputBinding::Plane(index) => resolved_inputs[index as usize],
                CompiledDirectSourceInputBinding::Scalar(index) => {
                    // SAFETY: static binding validation proved this scalar
                    // index and the multiplication above proved the allocation.
                    unsafe { broadcast_base.add(index as usize * stride) }
                }
            };
            // SAFETY: every resolved pointer covers one complete fixed plane
            // and remains live for the bound stage lifetime.
            descriptors.push(unsafe { PlaneDescriptor::from_cached_raw_parts(pointer, stride)? });
        }
        for output in self.output_bindings.iter().copied() {
            let pointer = resolve_arena_plane_pointer(output.0, arena, current_component_map)?;
            // SAFETY: mapped output validation proves distinct complete planes
            // which remain live for the bound stage lifetime.
            descriptors.push(unsafe { PlaneDescriptor::from_cached_raw_parts(pointer, stride)? });
        }
        let table = self.kernel.build_raw_table_from_descriptors(&descriptors)?;
        let initial_fill_bytes = u64::try_from(broadcast_len)
            .unwrap_or(u64::MAX)
            .saturating_mul(std::mem::size_of::<f64>() as u64);
        Ok(BoundSymjitCompiledDirectStage {
            kernel: self.kernel,
            table,
            broadcast_values,
            broadcast_sources: broadcast_sources.into_boxed_slice(),
            point_stride,
            pending_broadcast_fill_bytes: initial_fill_bytes,
            display_path: self.display_path,
        })
    }
}

impl BoundSymjitCompiledDirectStage {
    pub(crate) fn allocation_counters(&self) -> DirectArenaAllocationCounters {
        self.broadcast_values.allocation_counters()
    }

    /// Evaluate a point range directly into its disjoint output planes.
    ///
    /// The returned byte count is internal scalar-broadcast traffic. It is
    /// deliberately separate from boundary pack/gather/scatter counters.
    pub(crate) fn evaluate(&mut self, point_start: u32, point_count: u32) -> RusticolResult<u64> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "compiled Direct-Arena point count must be positive",
            ));
        }
        let point_stop = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("compiled Direct-Arena point range overflows u32")
        })?;
        if point_stop > self.point_stride {
            return Err(RusticolError::invalid_argument(format!(
                "compiled Direct-Arena point range {point_start}..{point_stop} \
                 exceeds physical stride {}",
                self.point_stride
            )));
        }

        let mut fill_bytes = self.pending_broadcast_fill_bytes;
        self.pending_broadcast_fill_bytes = 0;
        let stride = self.point_stride as usize;
        let values = self.broadcast_values.as_mut_slice();
        for (slot, source) in self.broadcast_sources.iter_mut().enumerate() {
            let BroadcastSource::Parameter { value, cached_bits } = source else {
                continue;
            };
            // SAFETY: the bind contract keeps the parameter allocation live,
            // and evaluation has exclusive synchronized access.
            let current = unsafe { **value };
            if current.to_bits() == *cached_bits {
                continue;
            }
            let start = slot * stride;
            values[start..start + stride].fill(current);
            *cached_bits = current.to_bits();
            fill_bytes = fill_bytes.saturating_add(
                u64::try_from(stride)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(std::mem::size_of::<f64>() as u64),
            );
        }
        self.kernel
            .execute_table(&mut self.table, point_start as usize, point_count as usize)
            .map_err(|error| {
                RusticolError::evaluation(format!(
                    "compiled plane application {} failed: {}",
                    self.display_path.display(),
                    error.message()
                ))
            })?;
        Ok(fill_bytes)
    }
}

fn validate_static_bindings(
    sources: &[CompiledDirectSourceInputBinding],
    inputs: &[CompiledDirectPlaneBinding],
    scalars: &[CompiledDirectScalarBinding],
    outputs: &[CompiledDirectOutputBinding],
) -> RusticolResult<()> {
    if sources.is_empty() || !sources.len().is_multiple_of(2) {
        return Err(RusticolError::integrity(
            "compiled P-kernel split input binding count must be positive and even",
        ));
    }
    if outputs.is_empty() || !outputs.len().is_multiple_of(2) {
        return Err(RusticolError::integrity(
            "compiled P-kernel split output binding count must be positive and even",
        ));
    }
    for (index, source) in sources.iter().copied().enumerate() {
        let valid = match source {
            CompiledDirectSourceInputBinding::Plane(slot) => (slot as usize) < inputs.len(),
            CompiledDirectSourceInputBinding::Scalar(slot) => (slot as usize) < scalars.len(),
        };
        if !valid {
            return Err(RusticolError::integrity(format!(
                "compiled P-kernel source input binding {index} is out of bounds"
            )));
        }
    }
    for (index, output) in outputs.iter().copied().enumerate() {
        if outputs[..index].contains(&output) {
            return Err(RusticolError::integrity(format!(
                "compiled P-kernel output binding {index} aliases an earlier output"
            )));
        }
        if inputs.contains(&CompiledDirectPlaneBinding::Arena(output.0)) {
            return Err(RusticolError::integrity(format!(
                "compiled P-kernel output binding {index} aliases an input plane"
            )));
        }
    }
    Ok(())
}

fn validate_mapped_bindings(
    inputs: &[CompiledDirectPlaneBinding],
    outputs: &[CompiledDirectOutputBinding],
    current_component_map: &[u32],
) -> RusticolResult<()> {
    let mapped_inputs = inputs
        .iter()
        .copied()
        .filter_map(|binding| match binding {
            CompiledDirectPlaneBinding::Arena(arena) => {
                Some(map_arena_plane(arena, current_component_map))
            }
            CompiledDirectPlaneBinding::Momentum { .. } | CompiledDirectPlaneBinding::Zero => None,
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut mapped_outputs = Vec::with_capacity(outputs.len());
    for (index, output) in outputs.iter().copied().enumerate() {
        let mapped = map_arena_plane(output.0, current_component_map)?;
        if mapped_outputs.contains(&mapped) {
            return Err(RusticolError::integrity(format!(
                "compiled P-kernel mapped output binding {index} aliases an earlier output"
            )));
        }
        if mapped_inputs.contains(&mapped) {
            return Err(RusticolError::integrity(format!(
                "compiled P-kernel mapped output binding {index} aliases an input plane"
            )));
        }
        mapped_outputs.push(mapped);
    }
    Ok(())
}

fn map_arena_plane(
    binding: CompiledDirectArenaPlane,
    current_component_map: &[u32],
) -> RusticolResult<CompiledDirectArenaPlane> {
    match binding {
        CompiledDirectArenaPlane::Current {
            component,
            imaginary,
        } if !current_component_map.is_empty() => {
            let physical = current_component_map
                .get(component as usize)
                .copied()
                .filter(|physical| *physical != u32::MAX)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "compiled current binding {component} has no physical mapping"
                    ))
                })?;
            Ok(CompiledDirectArenaPlane::Current {
                component: physical,
                imaginary,
            })
        }
        _ => Ok(binding),
    }
}

fn resolve_input_plane_pointer(
    binding: CompiledDirectPlaneBinding,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    zero_plane: &AlignedF64Buffer,
    current_component_map: &[u32],
) -> RusticolResult<*mut f64> {
    match binding {
        CompiledDirectPlaneBinding::Arena(binding) => {
            resolve_arena_plane_pointer(binding, arena, current_component_map)
        }
        CompiledDirectPlaneBinding::Momentum {
            form,
            lorentz_component,
        } => {
            if form >= momenta.form_count || lorentz_component >= momenta.lorentz_component_count {
                return Err(RusticolError::integrity(format!(
                    "compiled momentum binding ({form}, {lorentz_component}) is out of bounds"
                )));
            }
            let component = form
                .checked_mul(u32::from(momenta.lorentz_component_count))
                .and_then(|base| base.checked_add(u32::from(lorentz_component)))
                .ok_or_else(|| {
                    RusticolError::integrity("compiled momentum component overflows u32")
                })?;
            let offset = usize::try_from(component)
                .ok()
                .and_then(|component| component.checked_mul(momenta.point_stride as usize))
                .ok_or_else(|| {
                    RusticolError::integrity("compiled momentum offset overflows usize")
                })?;
            // SAFETY: the momentum view and component bounds were validated.
            Ok(unsafe { momenta.values.add(offset) }.cast_mut())
        }
        CompiledDirectPlaneBinding::Zero => Ok(zero_plane.as_ptr().cast_mut()),
    }
}

fn resolve_arena_plane_pointer(
    binding: CompiledDirectArenaPlane,
    arena: DirectArenaView,
    current_component_map: &[u32],
) -> RusticolResult<*mut f64> {
    let binding = map_arena_plane(binding, current_component_map)?;
    let (base, shape, label) = match binding {
        CompiledDirectArenaPlane::Current {
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
        CompiledDirectArenaPlane::Amplitude {
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
    // SAFETY: shape validation and the checked offset prove one complete
    // pitched plane begins at this pointer.
    Ok(unsafe { base.add(offset) })
}

#[cfg(test)]
mod tests {
    use super::super::symjit_direct::tests::count_allocations;
    use super::super::symjit_plane::compile_symbolica_program_to_plane_application_bytes;
    use super::*;
    use crate::RusticolErrorKind;
    use crate::direct_arena::{DirectArenaWorkspace, DirectMomentumView};

    const MODEL: &str = "
([('fun', ('temp', 0), 'square', [], [('param', 0)], False),
  ('add', ('temp', 1), [('temp', 0), ('param', 1)], 0),
  ('assign', ('out', 0), ('temp', 1))],
 2,
 [])
";

    fn application_bytes(level: u8) -> Vec<u8> {
        compile_symbolica_program_to_plane_application_bytes(MODEL, 2, 1, level, true).unwrap()
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

    fn loaded(level: u8) -> LoadedSymjitCompiledDirectStage {
        try_loaded(level, true).unwrap()
    }

    fn try_loaded(
        level: u8,
        expected_compression: bool,
    ) -> RusticolResult<LoadedSymjitCompiledDirectStage> {
        LoadedSymjitCompiledDirectStage::load_source_bytes(
            &application_bytes(level),
            format!("compiled-plane-o{level}.symjit"),
            SYMJIT_PLANE_APPLICATION_ABI,
            level,
            expected_compression,
            vec![
                CompiledDirectSourceInputBinding::Plane(0),
                CompiledDirectSourceInputBinding::Plane(1),
                CompiledDirectSourceInputBinding::Scalar(0),
                CompiledDirectSourceInputBinding::Scalar(1),
            ],
            vec![
                CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                    component: 0,
                    imaginary: false,
                }),
                CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                    component: 0,
                    imaginary: true,
                }),
            ],
            vec![
                CompiledDirectScalarBinding::Parameter {
                    index: 0,
                    imaginary: false,
                },
                CompiledDirectScalarBinding::Parameter {
                    index: 0,
                    imaginary: true,
                },
            ],
            vec![
                CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                    component: 1,
                    imaginary: false,
                }),
                CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                    component: 1,
                    imaginary: true,
                }),
            ],
        )
    }

    #[test]
    fn compiled_plane_stage_refreshes_parameters_and_allocates_zero_when_warm() {
        const POINTS: usize = 1025;
        const ODD_TAIL_LENGTHS: &[usize] = &[127, 129, 1023, 1025];
        let loaded = loaded(3);
        assert!(matches!(loaded.kernel.simd_lanes(), 1 | 2 | 4));
        let mut workspace = DirectArenaWorkspace::new(2, 0, POINTS as u32).unwrap();
        workspace.begin_tile(POINTS as u32).unwrap();
        let stride = workspace.point_stride() as usize;
        {
            let (current_re, current_im, _, _) = workspace.split_slices_mut();
            for point in 0..POINTS {
                current_re[point] = 0.25 + point as f64 / 32.0;
                current_im[point] = -0.5 + point as f64 / 64.0;
            }
            current_re[stride..2 * stride].fill(f64::NAN);
            current_im[stride..2 * stride].fill(f64::NAN);
        }
        let momenta = AlignedF64Buffer::zeroed(4 * stride, "compiled test momenta").unwrap();
        let zero = AlignedF64Buffer::zeroed(stride, "compiled test zero").unwrap();
        let mut parameter_re = [1.5];
        let mut parameter_im = [-0.75];
        let parameters = DirectParameterView {
            values_re: parameter_re.as_ptr(),
            values_im: parameter_im.as_ptr(),
            value_count: 1,
        };
        let arena = workspace.view().unwrap();
        let mut bound = unsafe {
            loaded.bind(
                arena,
                momentum_view(&momenta, arena.point_stride),
                parameters,
                &zero,
            )
        }
        .unwrap();

        let initial_bytes = bound.evaluate(1, 127).unwrap();
        assert_eq!(initial_bytes, (2 * stride * 8) as u64);
        {
            let (current_re, current_im) = workspace.current_slices();
            assert!(current_re[stride].is_nan());
            assert!(current_im[stride].is_nan());
            assert!(current_re[stride + 128].is_nan());
            assert!(current_im[stride + 128].is_nan());
            for point in 1..128 {
                let xr = current_re[point];
                let xi = current_im[point];
                assert_eq!(
                    current_re[stride + point],
                    xr * xr - xi * xi + parameter_re[0]
                );
                assert_eq!(current_im[stride + point], 2.0 * xr * xi + parameter_im[0]);
            }
        }

        for (case, &points) in ODD_TAIL_LENGTHS.iter().enumerate() {
            parameter_re[0] = -2.0 - case as f64;
            parameter_im[0] = 0.375 + case as f64 / 8.0;
            {
                let (current_re, current_im, _, _) = workspace.split_slices_mut();
                current_re[stride..2 * stride].fill(f64::NAN);
                current_im[stride..2 * stride].fill(f64::NAN);
            }
            let (refresh, refresh_allocations, refresh_bytes) =
                count_allocations(|| bound.evaluate(0, points as u32));
            assert_eq!(refresh.unwrap(), (2 * stride * 8) as u64);
            assert_eq!(
                (refresh_allocations, refresh_bytes),
                (0, 0),
                "compiled parameter refresh allocated for {points} points"
            );
            let (cached, cached_allocations, cached_bytes) =
                count_allocations(|| bound.evaluate(0, points as u32));
            assert_eq!(cached.unwrap(), 0);
            assert_eq!(
                (cached_allocations, cached_bytes),
                (0, 0),
                "warmed compiled call allocated for {points} points"
            );

            let (current_re, current_im) = workspace.current_slices();
            for point in 0..points {
                let xr = current_re[point];
                let xi = current_im[point];
                assert_eq!(
                    current_re[stride + point],
                    xr * xr - xi * xi + parameter_re[0]
                );
                assert_eq!(current_im[stride + point], 2.0 * xr * xi + parameter_im[0]);
            }
            if points < POINTS {
                assert!(current_re[stride + points].is_nan());
                assert!(current_im[stride + points].is_nan());
            }
        }
    }

    #[test]
    fn compiled_plane_stage_accepts_o0_through_o3_and_rejects_old_abi() {
        for level in 0..=3 {
            let stage = loaded(level);
            assert_eq!(stage.kernel.optimization_level(), level);
        }
        let error = LoadedSymjitCompiledDirectStage::load_source_bytes(
            &application_bytes(2),
            "old-plane-application.symjit",
            "pyamplicol-symjit-plane-application-v1",
            2,
            true,
            vec![
                CompiledDirectSourceInputBinding::Plane(0),
                CompiledDirectSourceInputBinding::Plane(1),
                CompiledDirectSourceInputBinding::Scalar(0),
                CompiledDirectSourceInputBinding::Scalar(1),
            ],
            vec![
                CompiledDirectPlaneBinding::Zero,
                CompiledDirectPlaneBinding::Zero,
            ],
            vec![
                CompiledDirectScalarBinding::Literal(1.0),
                CompiledDirectScalarBinding::Literal(0.0),
            ],
            vec![
                CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                    component: 0,
                    imaginary: false,
                }),
                CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                    component: 0,
                    imaginary: true,
                }),
            ],
        )
        .err()
        .expect("old ABI must fail closed");
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn authenticated_manifest_compression_mismatch_fails_closed() {
        let error = try_loaded(2, false)
            .err()
            .expect("a manifest/application compression mismatch must fail closed");
        assert!(error.message().contains("compression"));
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn static_and_mapped_output_aliases_fail_before_execution() {
        let bytes = application_bytes(2);
        let error = LoadedSymjitCompiledDirectStage::load_source_bytes(
            &bytes,
            "static-alias.symjit",
            SYMJIT_PLANE_APPLICATION_ABI,
            2,
            true,
            vec![
                CompiledDirectSourceInputBinding::Plane(0),
                CompiledDirectSourceInputBinding::Plane(1),
                CompiledDirectSourceInputBinding::Scalar(0),
                CompiledDirectSourceInputBinding::Scalar(1),
            ],
            vec![
                CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                    component: 0,
                    imaginary: false,
                }),
                CompiledDirectPlaneBinding::Arena(CompiledDirectArenaPlane::Current {
                    component: 0,
                    imaginary: true,
                }),
            ],
            vec![
                CompiledDirectScalarBinding::Literal(0.0),
                CompiledDirectScalarBinding::Literal(0.0),
            ],
            vec![
                CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                    component: 0,
                    imaginary: false,
                }),
                CompiledDirectOutputBinding(CompiledDirectArenaPlane::Current {
                    component: 1,
                    imaginary: true,
                }),
            ],
        )
        .err()
        .expect("input/output alias must fail");
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);

        let loaded = loaded(2);
        let mut workspace = DirectArenaWorkspace::new(2, 0, 8).unwrap();
        workspace.begin_tile(8).unwrap();
        let stride = workspace.point_stride() as usize;
        let momenta = AlignedF64Buffer::zeroed(4 * stride, "mapped test momenta").unwrap();
        let zero = AlignedF64Buffer::zeroed(stride, "mapped test zero").unwrap();
        let parameter_re = [0.0];
        let parameter_im = [0.0];
        let parameters = DirectParameterView {
            values_re: parameter_re.as_ptr(),
            values_im: parameter_im.as_ptr(),
            value_count: 1,
        };
        let arena = workspace.view().unwrap();
        let error = unsafe {
            loaded.bind_with_current_map(
                arena,
                momentum_view(&momenta, arena.point_stride),
                parameters,
                &zero,
                &[0, 0],
            )
        }
        .err()
        .expect("mapped alias must fail");
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    }
}
