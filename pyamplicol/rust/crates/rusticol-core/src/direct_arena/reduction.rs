// SPDX-License-Identifier: 0BSD

//! Borrowed split-complex amplitude planes for lane-neutral reductions.

use std::mem::size_of;

use crate::{RusticolError, RusticolResult};

use super::{DIRECT_ARENA_ALIGNMENT, DirectPlaneShape};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Read-only component-major amplitude planes for one active point tile.
///
/// Each component occupies `point_stride` consecutive scalars in both halves
/// of the split-complex storage. Only `0..point_count` is semantically active;
/// the remaining physical pitch is padding retained by the persistent arena.
#[derive(Clone, Copy, Debug)]
pub struct DirectAmplitudePlanes<'a> {
    values_re: &'a [f64],
    values_im: &'a [f64],
    shape: DirectPlaneShape,
    point_count: u32,
}

impl<'a> DirectAmplitudePlanes<'a> {
    pub fn new(
        values_re: &'a [f64],
        values_im: &'a [f64],
        point_stride: u32,
        point_count: u32,
    ) -> RusticolResult<Self> {
        if values_re.len() != values_im.len() {
            return Err(invalid(
                "direct amplitude real and imaginary plane lengths disagree",
            ));
        }
        if point_count == 0 || point_count > point_stride {
            return Err(invalid(
                "direct amplitude active point count is outside its physical pitch",
            ));
        }
        let scalar_len = u64::try_from(values_re.len())
            .map_err(|_| invalid("direct amplitude scalar length exceeds u64"))?;
        let shape = DirectPlaneShape::new(scalar_len, point_stride, "amplitude reduction")?;
        let alignment_scalars = u32::try_from(DIRECT_ARENA_ALIGNMENT / size_of::<f64>())
            .map_err(|_| invalid("direct amplitude alignment exceeds u32"))?;
        if !point_stride.is_multiple_of(alignment_scalars) {
            return Err(invalid(
                "direct amplitude point stride does not preserve per-plane alignment",
            ));
        }
        for (values, label) in [(values_re, "real"), (values_im, "imaginary")] {
            if values.is_empty()
                || !(values.as_ptr() as usize).is_multiple_of(DIRECT_ARENA_ALIGNMENT)
            {
                return Err(invalid(format!(
                    "direct amplitude {label} plane base is not 64-byte aligned"
                )));
            }
        }
        let _ = shape.component_count()?;
        Ok(Self {
            values_re,
            values_im,
            shape,
            point_count,
        })
    }

    pub const fn point_count(self) -> u32 {
        self.point_count
    }

    pub const fn point_stride(self) -> u32 {
        self.shape.point_stride()
    }

    pub fn component_count(self) -> RusticolResult<u32> {
        self.shape.component_count()
    }

    /// Return one active plane without exposing its physical pitch padding.
    ///
    /// Callers validate the component catalog once before entering their hot
    /// reduction loop. Keeping points contiguous here lets reducers traverse
    /// terms/components outside and points inside without reconstructing
    /// point-major rows.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn plane_unchecked(self, component: usize) -> (&'a [f64], &'a [f64]) {
        debug_assert!(component < self.shape.component_count().unwrap_or(0) as usize);
        let start = component * self.shape.point_stride() as usize;
        let stop = start + self.point_count as usize;
        (&self.values_re[start..stop], &self.values_im[start..stop])
    }

    #[inline(always)]
    pub(crate) fn value_unchecked(self, component: usize, point: usize) -> (f64, f64) {
        debug_assert!(component < self.shape.component_count().unwrap_or(0) as usize);
        debug_assert!(point < self.point_count as usize);
        let index = component * self.shape.point_stride() as usize + point;
        (self.values_re[index], self.values_im[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_arena::DirectArenaWorkspace;

    #[test]
    fn validates_alignment_pitch_and_active_tail() {
        let mut workspace = DirectArenaWorkspace::new(0, 3, 129).unwrap();
        workspace.begin_tile(127).unwrap();
        let stride = workspace.point_stride();
        let (values_re, values_im) = workspace.amplitude_slices();
        let planes = DirectAmplitudePlanes::new(values_re, values_im, stride, 127).unwrap();
        assert_eq!(planes.component_count().unwrap(), 3);
        assert_eq!(planes.point_count(), 127);
        assert_eq!(planes.point_stride(), 136);
        let (plane_re, plane_im) = planes.plane_unchecked(1);
        assert_eq!(plane_re.len(), 127);
        assert_eq!(plane_im.len(), 127);
        assert_eq!(plane_re.as_ptr(), values_re[136..].as_ptr());
        assert_eq!(plane_im.as_ptr(), values_im[136..].as_ptr());
        assert!(DirectAmplitudePlanes::new(values_re, values_im, stride, 137).is_err());
        assert!(DirectAmplitudePlanes::new(&values_re[1..], &values_im[1..], stride, 1).is_err());
    }
}
