// SPDX-License-Identifier: 0BSD

//! ABI-stable lane-neutral Direct-Arena views.

use std::mem::{align_of, size_of};
use std::ops::Range;

use crate::{RusticolError, RusticolResult};

use super::DIRECT_ARENA_ALIGNMENT;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Shape of one component-major, point-contiguous plane bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectPlaneShape {
    scalar_len: u64,
    point_stride: u32,
}

impl DirectPlaneShape {
    pub fn new(scalar_len: u64, point_stride: u32, label: &str) -> RusticolResult<Self> {
        if point_stride == 0 {
            return Err(invalid(format!(
                "direct {label} point stride must be positive"
            )));
        }
        if !scalar_len.is_multiple_of(u64::from(point_stride)) {
            return Err(invalid(format!(
                "direct {label} scalar length {scalar_len} is not a whole number of \
                 point-contiguous planes with stride {point_stride}"
            )));
        }
        Ok(Self {
            scalar_len,
            point_stride,
        })
    }

    pub const fn scalar_len(self) -> u64 {
        self.scalar_len
    }

    pub const fn point_stride(self) -> u32 {
        self.point_stride
    }

    pub fn component_count(self) -> RusticolResult<u32> {
        u32::try_from(self.scalar_len / u64::from(self.point_stride))
            .map_err(|_| invalid("direct plane component count exceeds u32"))
    }

    pub fn checked_scalar_index(
        self,
        component: u32,
        point: u32,
        label: &str,
    ) -> RusticolResult<usize> {
        if point >= self.point_stride {
            return Err(invalid(format!(
                "direct {label} point {point} exceeds stride {}",
                self.point_stride
            )));
        }
        if component >= self.component_count()? {
            return Err(invalid(format!(
                "direct {label} component {component} is out of bounds"
            )));
        }
        let index = u64::from(component)
            .checked_mul(u64::from(self.point_stride))
            .and_then(|base| base.checked_add(u64::from(point)))
            .ok_or_else(|| invalid(format!("direct {label} scalar index overflows u64")))?;
        usize::try_from(index)
            .map_err(|_| invalid(format!("direct {label} scalar index exceeds usize")))
    }

    pub fn checked_component_range(
        self,
        component_base: u32,
        component_count: u32,
        label: &str,
    ) -> RusticolResult<Range<usize>> {
        let component_stop = component_base
            .checked_add(component_count)
            .ok_or_else(|| invalid(format!("direct {label} component range overflows u32")))?;
        if component_stop > self.component_count()? {
            return Err(invalid(format!(
                "direct {label} component range {component_base}..{component_stop} is out of bounds"
            )));
        }
        let start = u64::from(component_base)
            .checked_mul(u64::from(self.point_stride))
            .ok_or_else(|| invalid(format!("direct {label} scalar range overflows u64")))?;
        let stop = u64::from(component_stop)
            .checked_mul(u64::from(self.point_stride))
            .ok_or_else(|| invalid(format!("direct {label} scalar range overflows u64")))?;
        Ok(usize::try_from(start)
            .map_err(|_| invalid(format!("direct {label} scalar range exceeds usize")))?
            ..usize::try_from(stop)
                .map_err(|_| invalid(format!("direct {label} scalar range exceeds usize")))?)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DirectArenaView {
    pub current_re: *mut f64,
    pub current_im: *mut f64,
    pub current_scalar_len: u64,
    pub amplitude_re: *mut f64,
    pub amplitude_im: *mut f64,
    pub amplitude_scalar_len: u64,
    /// Physical component-plane pitch, not the semantic active point count.
    pub point_stride: u32,
}

impl DirectArenaView {
    pub fn validate(self) -> RusticolResult<()> {
        let _ = self.current_shape()?.component_count()?;
        let _ = self.amplitude_shape()?.component_count()?;
        if self.current_scalar_len != 0 || self.amplitude_scalar_len != 0 {
            require_plane_stride_alignment(self.point_stride, "arena")?;
        }
        require_split_pair(
            self.current_re,
            self.current_im,
            self.current_scalar_len,
            "current",
        )?;
        require_split_pair(
            self.amplitude_re,
            self.amplitude_im,
            self.amplitude_scalar_len,
            "amplitude",
        )?;
        require_arena_alignment(
            self.current_re.cast_const(),
            self.current_scalar_len,
            "current real",
        )?;
        require_arena_alignment(
            self.current_im.cast_const(),
            self.current_scalar_len,
            "current imaginary",
        )?;
        require_arena_alignment(
            self.amplitude_re.cast_const(),
            self.amplitude_scalar_len,
            "amplitude real",
        )?;
        require_arena_alignment(
            self.amplitude_im.cast_const(),
            self.amplitude_scalar_len,
            "amplitude imaginary",
        )?;
        require_disjoint_ranges(&self.mutable_ranges()?)?;
        Ok(())
    }

    pub fn current_shape(self) -> RusticolResult<DirectPlaneShape> {
        DirectPlaneShape::new(self.current_scalar_len, self.point_stride, "current arena")
    }

    pub fn amplitude_shape(self) -> RusticolResult<DirectPlaneShape> {
        DirectPlaneShape::new(
            self.amplitude_scalar_len,
            self.point_stride,
            "amplitude arena",
        )
    }

    fn mutable_ranges(self) -> RusticolResult<[Option<DeclaredRange>; 4]> {
        Ok([
            checked_pointer_range(
                self.current_re.cast_const(),
                self.current_scalar_len,
                "current real",
            )?,
            checked_pointer_range(
                self.current_im.cast_const(),
                self.current_scalar_len,
                "current imaginary",
            )?,
            checked_pointer_range(
                self.amplitude_re.cast_const(),
                self.amplitude_scalar_len,
                "amplitude real",
            )?,
            checked_pointer_range(
                self.amplitude_im.cast_const(),
                self.amplitude_scalar_len,
                "amplitude imaginary",
            )?,
        ])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DirectMomentumView {
    pub values: *const f64,
    pub scalar_len: u64,
    pub form_count: u32,
    pub lorentz_component_count: u16,
    /// Physical component-plane pitch, shared with the arena view.
    pub point_stride: u32,
}

impl DirectMomentumView {
    pub fn validate(self) -> RusticolResult<()> {
        if self.form_count == 0 || self.lorentz_component_count == 0 {
            return Err(invalid("direct momentum view has an empty dimension"));
        }
        let required = u64::from(self.form_count)
            .checked_mul(u64::from(self.lorentz_component_count))
            .and_then(|planes| planes.checked_mul(u64::from(self.point_stride)))
            .ok_or_else(|| invalid("direct momentum view dimensions overflow u64"))?;
        let shape = DirectPlaneShape::new(self.scalar_len, self.point_stride, "momentum")?;
        require_plane_stride_alignment(self.point_stride, "momentum")?;
        if required != self.scalar_len {
            return Err(invalid(format!(
                "direct momentum view requires exactly {required} scalars but exposes {}",
                self.scalar_len
            )));
        }
        if self.values.is_null() {
            return Err(invalid("direct momentum view has a null base"));
        }
        require_arena_alignment(self.values, self.scalar_len, "momentum")?;
        if shape.component_count()?
            != self
                .form_count
                .checked_mul(u32::from(self.lorentz_component_count))
                .ok_or_else(|| invalid("direct momentum plane count overflows u32"))?
        {
            return Err(invalid("direct momentum plane shape is inconsistent"));
        }
        let _ = checked_pointer_range(self.values, self.scalar_len, "momentum")?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DirectParameterView {
    pub values_re: *const f64,
    pub values_im: *const f64,
    pub value_count: u32,
}

impl DirectParameterView {
    pub fn validate(self) -> RusticolResult<()> {
        require_split_pair(
            self.values_re.cast_mut(),
            self.values_im.cast_mut(),
            u64::from(self.value_count),
            "parameter",
        )?;
        require_disjoint_ranges(&[
            checked_pointer_range(
                self.values_re,
                u64::from(self.value_count),
                "parameter real",
            )?,
            checked_pointer_range(
                self.values_im,
                u64::from(self.value_count),
                "parameter imaginary",
            )?,
        ])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DirectFactorView {
    pub values_re: *const f64,
    pub values_im: *const f64,
    pub value_count: u32,
}

impl DirectFactorView {
    pub fn validate(self) -> RusticolResult<()> {
        require_split_pair(
            self.values_re.cast_mut(),
            self.values_im.cast_mut(),
            u64::from(self.value_count),
            "factor",
        )?;
        require_disjoint_ranges(&[
            checked_pointer_range(self.values_re, u64::from(self.value_count), "factor real")?,
            checked_pointer_range(
                self.values_im,
                u64::from(self.value_count),
                "factor imaginary",
            )?,
        ])
    }
}

/// Validate one complete direct-call descriptor bundle.
///
/// Read-only inputs may alias each other, but none may overlap a mutable arena.
pub fn validate_direct_views(
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
) -> RusticolResult<()> {
    arena.validate()?;
    momenta.validate()?;
    parameters.validate()?;
    factors.validate()?;
    if arena.point_stride != momenta.point_stride {
        return Err(invalid(format!(
            "direct arena pitch {} does not match momentum pitch {}",
            arena.point_stride, momenta.point_stride
        )));
    }

    let mutable_ranges = arena.mutable_ranges()?;
    let read_only_ranges = [
        checked_pointer_range(momenta.values, momenta.scalar_len, "momentum")?,
        checked_pointer_range(
            parameters.values_re,
            u64::from(parameters.value_count),
            "parameter real",
        )?,
        checked_pointer_range(
            parameters.values_im,
            u64::from(parameters.value_count),
            "parameter imaginary",
        )?,
        checked_pointer_range(
            factors.values_re,
            u64::from(factors.value_count),
            "factor real",
        )?,
        checked_pointer_range(
            factors.values_im,
            u64::from(factors.value_count),
            "factor imaginary",
        )?,
    ];
    for mutable in mutable_ranges.iter().flatten() {
        for read_only in read_only_ranges.iter().flatten() {
            if mutable.overlaps(*read_only) {
                return Err(invalid(format!(
                    "mutable direct {} range overlaps read-only direct {} range",
                    mutable.label, read_only.label
                )));
            }
        }
    }
    Ok(())
}

fn require_split_pair(
    values_re: *mut f64,
    values_im: *mut f64,
    scalar_len: u64,
    label: &str,
) -> RusticolResult<()> {
    if scalar_len != 0 && (values_re.is_null() || values_im.is_null()) {
        return Err(invalid(format!(
            "nonempty direct {label} view has a null split base"
        )));
    }
    Ok(())
}

fn require_arena_alignment(values: *const f64, scalar_len: u64, label: &str) -> RusticolResult<()> {
    if scalar_len != 0 && !(values as usize).is_multiple_of(DIRECT_ARENA_ALIGNMENT) {
        return Err(invalid(format!(
            "direct {label} base is not {DIRECT_ARENA_ALIGNMENT}-byte aligned"
        )));
    }
    Ok(())
}

fn require_plane_stride_alignment(point_stride: u32, label: &str) -> RusticolResult<()> {
    // Packed singleton views are a separate, explicit ABI layout. Every
    // other pitch must retain per-plane alignment for tiled SIMD execution.
    if point_stride == 1 {
        return Ok(());
    }
    let stride_bytes = usize::try_from(point_stride)
        .ok()
        .and_then(|stride| stride.checked_mul(size_of::<f64>()))
        .ok_or_else(|| {
            invalid(format!(
                "direct {label} point stride byte count overflows usize"
            ))
        })?;
    if !stride_bytes.is_multiple_of(DIRECT_ARENA_ALIGNMENT) {
        return Err(invalid(format!(
            "direct {label} point stride does not preserve {DIRECT_ARENA_ALIGNMENT}-byte plane alignment"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct DeclaredRange {
    start: usize,
    stop: usize,
    label: &'static str,
}

impl DeclaredRange {
    const fn overlaps(self, other: Self) -> bool {
        self.start < other.stop && other.start < self.stop
    }
}

fn checked_pointer_range(
    values: *const f64,
    scalar_len: u64,
    label: &'static str,
) -> RusticolResult<Option<DeclaredRange>> {
    if scalar_len == 0 {
        return Ok(None);
    }
    if values.is_null() {
        return Err(invalid(format!(
            "nonempty direct {label} range has a null base"
        )));
    }
    let byte_len = usize::try_from(scalar_len)
        .ok()
        .and_then(|len| len.checked_mul(size_of::<f64>()))
        .ok_or_else(|| invalid(format!("direct {label} byte length overflows usize")))?;
    if byte_len > isize::MAX as usize {
        return Err(invalid(format!(
            "direct {label} byte length exceeds isize::MAX"
        )));
    }
    let start = values as usize;
    if !start.is_multiple_of(align_of::<f64>()) {
        return Err(invalid(format!(
            "direct {label} base is not binary64 aligned"
        )));
    }
    let stop = start
        .checked_add(byte_len)
        .ok_or_else(|| invalid(format!("direct {label} address range overflows usize")))?;
    Ok(Some(DeclaredRange { start, stop, label }))
}

fn require_disjoint_ranges(ranges: &[Option<DeclaredRange>]) -> RusticolResult<()> {
    for (index, left) in ranges.iter().copied().enumerate() {
        let Some(left) = left else {
            continue;
        };
        for right in ranges[index + 1..].iter().copied().flatten() {
            if left.overlaps(right) {
                return Err(invalid(format!(
                    "direct {} and {} ranges overlap",
                    left.label, right.label
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::AlignedF64Buffer;
    use super::*;

    #[test]
    fn component_major_shape_checks_projection() {
        assert!(DirectPlaneShape::new(15, 8, "test").is_err());
        let shape = DirectPlaneShape::new(24, 8, "test").unwrap();
        assert_eq!(shape.component_count().unwrap(), 3);
        assert_eq!(shape.checked_scalar_index(2, 7, "test").unwrap(), 23);
        assert_eq!(shape.checked_component_range(1, 2, "test").unwrap(), 8..24);
        assert!(shape.checked_scalar_index(3, 0, "test").is_err());
        assert!(shape.checked_scalar_index(0, 8, "test").is_err());
    }

    #[test]
    fn views_fail_closed_on_aliases_partial_planes_and_unaligned_pitch() {
        let mut values_re = AlignedF64Buffer::zeroed(24, "test real").unwrap();
        let mut values_im = AlignedF64Buffer::zeroed(24, "test imag").unwrap();
        let alias = DirectArenaView {
            current_re: values_re.as_mut_ptr(),
            current_im: values_re.as_mut_ptr(),
            current_scalar_len: 16,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: std::ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 8,
        };
        assert!(alias.validate().is_err());
        let partial = DirectArenaView {
            current_re: values_re.as_mut_ptr(),
            current_im: values_im.as_mut_ptr(),
            current_scalar_len: 15,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: std::ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 8,
        };
        assert!(partial.validate().is_err());
        let bad_pitch = DirectArenaView {
            current_re: values_re.as_mut_ptr(),
            current_im: values_im.as_mut_ptr(),
            current_scalar_len: 6,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: std::ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 3,
        };
        assert!(bad_pitch.validate().is_err());
    }

    #[test]
    fn packed_singleton_views_are_validated_as_an_explicit_layout() {
        let mut current_re = AlignedF64Buffer::zeroed(2, "packed current real").unwrap();
        let mut current_im = AlignedF64Buffer::zeroed(2, "packed current imag").unwrap();
        DirectArenaView {
            current_re: current_re.as_mut_ptr(),
            current_im: current_im.as_mut_ptr(),
            current_scalar_len: 2,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: std::ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 1,
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn complete_bundle_rejects_mutable_read_only_overlap() {
        let mut current_re = AlignedF64Buffer::zeroed(24, "current real").unwrap();
        let mut current_im = AlignedF64Buffer::zeroed(24, "current imag").unwrap();
        let parameters_im = AlignedF64Buffer::zeroed(8, "parameter imag").unwrap();
        let arena = DirectArenaView {
            current_re: current_re.as_mut_ptr(),
            current_im: current_im.as_mut_ptr(),
            current_scalar_len: 16,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: std::ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 8,
        };
        let momenta = DirectMomentumView {
            values: unsafe { current_re.as_ptr().add(8) },
            scalar_len: 8,
            form_count: 1,
            lorentz_component_count: 1,
            point_stride: 8,
        };
        let parameters = DirectParameterView {
            values_re: std::ptr::null(),
            values_im: std::ptr::null(),
            value_count: 0,
        };
        let factors = DirectFactorView {
            values_re: parameters_im.as_ptr(),
            values_im: parameters_im.as_ptr(),
            value_count: 0,
        };
        assert!(validate_direct_views(arena, momenta, parameters, factors).is_err());
    }

    #[test]
    fn partial_split_range_overlap_is_rejected() {
        let mut values = AlignedF64Buffer::zeroed(40, "overlap").unwrap();
        let overlap = DirectArenaView {
            current_re: values.as_mut_ptr(),
            current_im: unsafe { values.as_mut_ptr().add(8) },
            current_scalar_len: 16,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: std::ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 8,
        };
        assert!(overlap.validate().is_err());
        let parameter_overlap = DirectParameterView {
            values_re: values.as_ptr(),
            values_im: unsafe { values.as_ptr().add(8) },
            value_count: 16,
        };
        assert!(parameter_overlap.validate().is_err());
    }

    #[test]
    fn read_only_inputs_may_share_authenticated_storage() {
        let mut current_re = AlignedF64Buffer::zeroed(8, "current real").unwrap();
        let mut current_im = AlignedF64Buffer::zeroed(8, "current imag").unwrap();
        let momentum = AlignedF64Buffer::zeroed(8, "momentum").unwrap();
        let shared_re = AlignedF64Buffer::zeroed(8, "shared real").unwrap();
        let shared_im = AlignedF64Buffer::zeroed(8, "shared imag").unwrap();
        let arena = DirectArenaView {
            current_re: current_re.as_mut_ptr(),
            current_im: current_im.as_mut_ptr(),
            current_scalar_len: 8,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: std::ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 8,
        };
        let momenta = DirectMomentumView {
            values: momentum.as_ptr(),
            scalar_len: 8,
            form_count: 1,
            lorentz_component_count: 1,
            point_stride: 8,
        };
        let parameters = DirectParameterView {
            values_re: shared_re.as_ptr(),
            values_im: shared_im.as_ptr(),
            value_count: 8,
        };
        let factors = DirectFactorView {
            values_re: shared_re.as_ptr(),
            values_im: shared_im.as_ptr(),
            value_count: 8,
        };
        validate_direct_views(arena, momenta, parameters, factors).unwrap();
    }

    #[test]
    fn empty_views_ignore_pointer_values_while_momentum_dimensions_are_exact() {
        let unaligned = std::ptr::without_provenance::<f64>(3);
        let empty = DirectArenaView {
            current_re: unaligned.cast_mut(),
            current_im: std::ptr::null_mut(),
            current_scalar_len: 0,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: unaligned.cast_mut(),
            amplitude_scalar_len: 0,
            point_stride: 1,
        };
        empty.validate().unwrap();

        let values = AlignedF64Buffer::zeroed(24, "momentum").unwrap();
        let extra_plane = DirectMomentumView {
            values: values.as_ptr(),
            scalar_len: 24,
            form_count: 2,
            lorentz_component_count: 1,
            point_stride: 8,
        };
        assert!(extra_plane.validate().is_err());
        let partial = DirectMomentumView {
            values: values.as_ptr(),
            scalar_len: 15,
            form_count: 2,
            lorentz_component_count: 1,
            point_stride: 8,
        };
        assert!(partial.validate().is_err());
    }

    #[test]
    fn declared_pointer_range_overflow_fails_closed() {
        let near_limit =
            std::ptr::without_provenance::<f64>(usize::MAX - (DIRECT_ARENA_ALIGNMENT - 1));
        assert!(checked_pointer_range(near_limit, 16, "overflow").is_err());
        assert!(checked_pointer_range(near_limit, u64::MAX, "overflow").is_err());
        let over_isize = (isize::MAX as usize / size_of::<f64>()) + 1;
        assert!(
            checked_pointer_range(
                std::ptr::without_provenance::<f64>(DIRECT_ARENA_ALIGNMENT),
                over_isize as u64,
                "oversized",
            )
            .is_err()
        );
    }

    #[test]
    fn split_read_only_views_require_binary64_alignment() {
        let unaligned = std::ptr::without_provenance::<f64>(1);
        let aligned = std::ptr::without_provenance::<f64>(DIRECT_ARENA_ALIGNMENT);
        let parameters = DirectParameterView {
            values_re: unaligned,
            values_im: aligned,
            value_count: 1,
        };
        assert!(parameters.validate().is_err());
        let factors = DirectFactorView {
            values_re: aligned,
            values_im: unaligned,
            value_count: 1,
        };
        assert!(factors.validate().is_err());
    }

    #[test]
    fn complete_bundle_requires_one_shared_physical_pitch() {
        let mut current_re = AlignedF64Buffer::zeroed(8, "current real").unwrap();
        let mut current_im = AlignedF64Buffer::zeroed(8, "current imag").unwrap();
        let momentum = AlignedF64Buffer::zeroed(16, "momentum").unwrap();
        let arena = DirectArenaView {
            current_re: current_re.as_mut_ptr(),
            current_im: current_im.as_mut_ptr(),
            current_scalar_len: 8,
            amplitude_re: std::ptr::null_mut(),
            amplitude_im: std::ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 8,
        };
        let momenta = DirectMomentumView {
            values: momentum.as_ptr(),
            scalar_len: 16,
            form_count: 1,
            lorentz_component_count: 1,
            point_stride: 16,
        };
        let parameters = DirectParameterView {
            values_re: std::ptr::null(),
            values_im: std::ptr::null(),
            value_count: 0,
        };
        let factors = DirectFactorView {
            values_re: std::ptr::null(),
            values_im: std::ptr::null(),
            value_count: 0,
        };
        assert!(validate_direct_views(arena, momenta, parameters, factors).is_err());
    }
}
