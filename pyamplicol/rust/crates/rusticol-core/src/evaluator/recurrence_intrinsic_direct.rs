// SPDX-License-Identifier: 0BSD

//! Allocation-free model-generic contribution intrinsics for Direct-Arena recurrence.
//!
//! Every callback consumes authenticated fixed-width contribution rows and
//! reads/writes persistent split-complex current planes directly. The loader
//! owns the immutable scale context and retains it for at least as long as the
//! returned executor handle can be called.

use crate::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectContributionExecutor, DirectContributionFanoutBatch,
    DirectContributionFanoutExecutor, DirectContributionFanoutExecutorHandle,
    DirectContributionFanoutScale, DirectExecutorHandle, DirectFactorView,
    DirectFinalizationExecutor, DirectInteractionBundleBatch, DirectInteractionExecutorContexts,
    DirectInteractionIntrinsicKind, DirectInteractionOperand, DirectInteractionOperands,
    DirectInteractionRuntimeRequirements, DirectInteractionTerminalBatch,
    DirectInteractionVectorParent, DirectMomentumView, DirectParameterView,
};
use crate::recurrence::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DIRECT_NONE_U32, DirectContributionRow,
    DirectFinalizationRow,
};
use crate::{RusticolError, RusticolResult};
use std::ffi::{c_int, c_void};
use std::ptr;
use wide::f64x2;

const STATUS_INVALID_CONTEXT: c_int = 1;
const STATUS_INVALID_ARGUMENT: c_int = 2;
const STATUS_BOUNDS: c_int = 4;

pub(crate) const WEYL_VECTOR_TO_WEYL_POSITIVE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1";
pub(crate) const WEYL_VECTOR_TO_WEYL_NEGATIVE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.weyl-vector-to-weyl-b.v1";
pub(crate) const ANTISYMMETRIC_TENSOR_VECTOR_TO_VECTOR_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.antisymmetric-tensor-vector.v1";
pub(crate) const VECTOR_WEDGE_VECTOR_TO_ANTISYMMETRIC_TENSOR_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.vector-wedge-vector.v1";
pub(crate) const COLOR_ORDERED_THREE_VECTOR_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.color-ordered-three-vector.v1";
pub(crate) const WEYL_PROPAGATOR_POSITIVE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.weyl-propagator-a.v1";
pub(crate) const WEYL_PROPAGATOR_NEGATIVE_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.weyl-propagator-b.v1";
pub(crate) const FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE: &str =
    "rusticol.recurrence-intrinsic.vector-propagator-feynman.v1";

/// Exact algebra class certified by the model compiler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecurrenceContributionIntrinsicKind {
    WeylVectorToWeylPositive,
    WeylVectorToWeylNegative,
    AntisymmetricTensorVectorToVector,
    VectorWedgeVectorToAntisymmetricTensor,
    ColorOrderedThreeVector,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RecurrenceFinalizationIntrinsicKind {
    WeylPropagatorPositive,
    WeylPropagatorNegative,
    FeynmanVectorPropagator,
}

impl RecurrenceFinalizationIntrinsicKind {
    pub(crate) fn from_runtime_template(value: &str) -> RusticolResult<Self> {
        match value {
            WEYL_PROPAGATOR_POSITIVE_TEMPLATE => Ok(Self::WeylPropagatorPositive),
            WEYL_PROPAGATOR_NEGATIVE_TEMPLATE => Ok(Self::WeylPropagatorNegative),
            FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE => Ok(Self::FeynmanVectorPropagator),
            other => Err(RusticolError::compatibility(format!(
                "unsupported Direct-Arena recurrence finalization intrinsic {other:?}"
            ))),
        }
    }
}

impl RecurrenceContributionIntrinsicKind {
    #[cfg(test)]
    pub(crate) const fn runtime_template(self) -> &'static str {
        match self {
            Self::WeylVectorToWeylPositive => WEYL_VECTOR_TO_WEYL_POSITIVE_TEMPLATE,
            Self::WeylVectorToWeylNegative => WEYL_VECTOR_TO_WEYL_NEGATIVE_TEMPLATE,
            Self::AntisymmetricTensorVectorToVector => {
                ANTISYMMETRIC_TENSOR_VECTOR_TO_VECTOR_TEMPLATE
            }
            Self::VectorWedgeVectorToAntisymmetricTensor => {
                VECTOR_WEDGE_VECTOR_TO_ANTISYMMETRIC_TENSOR_TEMPLATE
            }
            Self::ColorOrderedThreeVector => COLOR_ORDERED_THREE_VECTOR_TEMPLATE,
        }
    }

    pub(crate) fn from_runtime_template(value: &str) -> RusticolResult<Self> {
        match value {
            WEYL_VECTOR_TO_WEYL_POSITIVE_TEMPLATE => Ok(Self::WeylVectorToWeylPositive),
            WEYL_VECTOR_TO_WEYL_NEGATIVE_TEMPLATE => Ok(Self::WeylVectorToWeylNegative),
            ANTISYMMETRIC_TENSOR_VECTOR_TO_VECTOR_TEMPLATE => {
                Ok(Self::AntisymmetricTensorVectorToVector)
            }
            VECTOR_WEDGE_VECTOR_TO_ANTISYMMETRIC_TENSOR_TEMPLATE => {
                Ok(Self::VectorWedgeVectorToAntisymmetricTensor)
            }
            COLOR_ORDERED_THREE_VECTOR_TEMPLATE => Ok(Self::ColorOrderedThreeVector),
            other => Err(RusticolError::compatibility(format!(
                "unsupported Direct-Arena recurrence contribution intrinsic {other:?}"
            ))),
        }
    }
}

/// Immutable scale applied after the certified intrinsic algebra.
///
/// The effective scale is `literal * parameters[model_parameter_index]` when
/// a parameter index is present, or just `literal` otherwise. Every row then
/// contributes `effective_scale * exact_factor[row.exact_factor_id] * base`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RecurrenceIntrinsicScale {
    literal_re: f64,
    literal_im: f64,
    model_parameter_index: Option<u32>,
}

impl RecurrenceIntrinsicScale {
    pub(crate) fn new(
        literal_re: f64,
        literal_im: f64,
        model_parameter_index: Option<u32>,
    ) -> RusticolResult<Self> {
        if !literal_re.is_finite() || !literal_im.is_finite() {
            return Err(RusticolError::invalid_argument(
                "Direct-Arena recurrence intrinsic scale must be finite",
            ));
        }
        if model_parameter_index == Some(DIRECT_NONE_U32) {
            return Err(RusticolError::invalid_argument(
                "Direct-Arena recurrence intrinsic parameter index uses the reserved sentinel",
            ));
        }
        Ok(Self {
            literal_re,
            literal_im,
            model_parameter_index,
        })
    }

    #[cfg(test)]
    pub(crate) fn unit() -> Self {
        Self {
            literal_re: 1.0,
            literal_im: 0.0,
            model_parameter_index: None,
        }
    }
}

/// A context-aware contribution handle whose owner must outlive every call.
#[derive(Clone, Copy)]
pub(crate) struct ContextDirectContributionExecutorHandle {
    pub(crate) call: DirectContributionExecutor,
    pub(crate) context: *const c_void,
}

/// Owns one immutable intrinsic scale context.
pub(crate) struct LoadedRecurrenceIntrinsicDirectExecutor {
    kind: RecurrenceContributionIntrinsicKind,
    scale: Box<RecurrenceIntrinsicScale>,
}

impl LoadedRecurrenceIntrinsicDirectExecutor {
    pub(crate) fn load(
        kind: RecurrenceContributionIntrinsicKind,
        scale: RecurrenceIntrinsicScale,
    ) -> Self {
        Self {
            kind,
            scale: Box::new(scale),
        }
    }

    pub(crate) fn load_runtime_template(
        runtime_template: &str,
        scale: RecurrenceIntrinsicScale,
    ) -> RusticolResult<Self> {
        Ok(Self::load(
            RecurrenceContributionIntrinsicKind::from_runtime_template(runtime_template)?,
            scale,
        ))
    }

    pub(crate) fn contribution_handle(&self) -> ContextDirectContributionExecutorHandle {
        let call = match self.kind {
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylPositive => {
                execute_weyl_vector_to_weyl_positive_rows
            }
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylNegative => {
                execute_weyl_vector_to_weyl_negative_rows
            }
            RecurrenceContributionIntrinsicKind::AntisymmetricTensorVectorToVector => {
                execute_antisymmetric_tensor_vector_to_vector_rows
            }
            RecurrenceContributionIntrinsicKind::VectorWedgeVectorToAntisymmetricTensor => {
                execute_vector_wedge_vector_to_antisymmetric_tensor_rows
            }
            RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector => {
                execute_color_ordered_three_vector_rows
            }
        };
        ContextDirectContributionExecutorHandle {
            call,
            context: ptr::from_ref(self.scale.as_ref()).cast(),
        }
    }

    pub(crate) fn contribution_fanout_handle(
        &self,
    ) -> Option<DirectContributionFanoutExecutorHandle> {
        let (
            call,
            destination_component_count,
            parent_component_counts,
            requires_two_momenta,
            interaction_kind,
        ): (
            DirectContributionFanoutExecutor,
            u32,
            [u32; 2],
            bool,
            DirectInteractionIntrinsicKind,
        ) = match self.kind {
            RecurrenceContributionIntrinsicKind::AntisymmetricTensorVectorToVector => (
                execute_antisymmetric_tensor_vector_to_vector_fanout,
                4,
                [6, 4],
                false,
                DirectInteractionIntrinsicKind::AntisymmetricTensorVector,
            ),
            RecurrenceContributionIntrinsicKind::VectorWedgeVectorToAntisymmetricTensor => (
                execute_vector_wedge_vector_to_antisymmetric_tensor_fanout,
                6,
                [4, 4],
                false,
                DirectInteractionIntrinsicKind::VectorWedgeVector,
            ),
            RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector => (
                execute_color_ordered_three_vector_fanout,
                4,
                [4, 4],
                true,
                DirectInteractionIntrinsicKind::ColorOrderedThreeVector,
            ),
            _ => return None,
        };
        Some(DirectContributionFanoutExecutorHandle {
            call,
            context: ptr::from_ref(self.scale.as_ref()).cast(),
            destination_component_count,
            parent_component_counts,
            requires_two_momenta,
            required_parameter_count: self
                .scale
                .model_parameter_index
                .map_or(0, |parameter| parameter + 1),
            interaction_kind,
            interaction_call: (interaction_kind
                == DirectInteractionIntrinsicKind::ColorOrderedThreeVector)
                .then_some(execute_interaction_bundle),
        })
    }

    pub(crate) fn handle(&self) -> DirectExecutorHandle {
        let handle = self.contribution_handle();
        DirectExecutorHandle::Contribution {
            call: handle.call,
            context: handle.context,
        }
    }

    pub(crate) fn finalization_handle(
        runtime_template: &str,
    ) -> RusticolResult<DirectExecutorHandle> {
        let call: DirectFinalizationExecutor =
            match RecurrenceFinalizationIntrinsicKind::from_runtime_template(runtime_template)? {
                RecurrenceFinalizationIntrinsicKind::WeylPropagatorPositive => {
                    execute_weyl_propagator_positive_rows
                }
                RecurrenceFinalizationIntrinsicKind::WeylPropagatorNegative => {
                    execute_weyl_propagator_negative_rows
                }
                RecurrenceFinalizationIntrinsicKind::FeynmanVectorPropagator => {
                    execute_feynman_vector_propagator_rows
                }
            };
        Ok(DirectExecutorHandle::Finalization {
            call,
            context: ptr::null::<c_void>(),
        })
    }
}

/// One generic identity finalizer for every non-propagating state.
///
/// The prepared catalog resolves exactly this one function. Per-state
/// dimensions are deliberately read from `row.component_count`.
pub(crate) unsafe extern "C" fn execute_identity_finalization_rows(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    if rows.is_null()
        || row_count == 0
        || point_count == 0
        || arena.point_stride == 0
        || point_count > arena.point_stride
    {
        return STATUS_INVALID_ARGUMENT;
    }
    if arena.current_re.is_null()
        || arena.current_im.is_null()
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return STATUS_INVALID_ARGUMENT;
    }

    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        if row.component_count == 0 || row.exact_factor_id >= factors.value_count {
            return STATUS_BOUNDS;
        }
        let factor_re = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        let factor_im = unsafe { *factors.values_im.add(row.exact_factor_id as usize) };
        for component in 0..u64::from(row.component_count) {
            let plane = match u64::from(row.component_base).checked_add(component) {
                Some(value) => value,
                None => return STATUS_BOUNDS,
            };
            let offset = match plane.checked_mul(u64::from(arena.point_stride)) {
                Some(value) => value,
                None => return STATUS_BOUNDS,
            };
            let end = match offset.checked_add(u64::from(point_count)) {
                Some(value) => value,
                None => return STATUS_BOUNDS,
            };
            if end > arena.current_scalar_len {
                return STATUS_BOUNDS;
            }
            let Ok(offset) = usize::try_from(offset) else {
                return STATUS_BOUNDS;
            };
            for point in 0..point_count as usize {
                let index = offset + point;
                let value_re = unsafe { *arena.current_re.add(index) };
                let value_im = unsafe { *arena.current_im.add(index) };
                unsafe {
                    *arena.current_re.add(index) = factor_re * value_re - factor_im * value_im;
                    *arena.current_im.add(index) = factor_re * value_im + factor_im * value_re;
                }
            }
        }
    }
    DIRECT_STATUS_OK
}

#[derive(Clone, Copy)]
struct ComplexValue {
    re: f64,
    im: f64,
}

impl ComplexValue {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };

    #[inline(always)]
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    #[inline(always)]
    fn mul_i(self) -> Self {
        Self::new(-self.im, self.re)
    }

    #[inline(always)]
    fn mul_real(self, value: f64) -> Self {
        Self::new(self.re * value, self.im * value)
    }

    #[inline(always)]
    fn mul_real_pair(self, value: f64x2) -> SimdComplex2 {
        SimdComplex2::new(
            f64x2::new([self.re, self.re]) * value,
            f64x2::new([self.im, self.im]) * value,
        )
    }
}

trait DirectContributionFormula {
    const PARENT0_COMPONENTS: u32;
    const PARENT1_COMPONENTS: u32;
    const DESTINATION_COMPONENTS: u32;
    const BASE_SCALE: ComplexValue;

    /// Evaluate one point after all row and arena bounds have been validated.
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    );

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        unsafe {
            Self::evaluate_point(arena, row, stride, point, scale);
            Self::evaluate_point(arena, row, stride, point + 1, scale);
        }
    }
}

struct WeylVectorToWeylPositive;
struct WeylVectorToWeylNegative;
struct AntisymmetricTensorVectorToVector;
struct VectorWedgeVectorToAntisymmetricTensor;
struct ColorOrderedThreeVector;

trait DirectContributionFanoutFormula<const N: usize> {
    const BASE_SCALE: ComplexValue;

    unsafe fn evaluate(
        arena: DirectArenaView,
        momenta: Option<DirectMomentumView>,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
    ) -> [ComplexValue; N];
}

#[derive(Clone, Copy)]
struct SimdComplex2 {
    re: f64x2,
    im: f64x2,
}

impl SimdComplex2 {
    #[inline(always)]
    fn new(re: f64x2, im: f64x2) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    #[inline(always)]
    fn mul_i(self) -> Self {
        Self::new(-self.im, self.re)
    }

    #[inline(always)]
    fn mul_real(self, value: f64x2) -> Self {
        Self::new(self.re * value, self.im * value)
    }
}

impl ColorOrderedThreeVector {
    const PARENT_COMPONENTS: u32 = 4;
    const DESTINATION_COMPONENTS: u32 = 4;
    const BASE_SCALE: ComplexValue = ComplexValue::new(1.0, 0.0);

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let mut left = [SimdComplex2::new(f64x2::ZERO, f64x2::ZERO); 4];
        let mut right = [SimdComplex2::new(f64x2::ZERO, f64x2::ZERO); 4];
        let mut parent0_momentum = [f64x2::ZERO; 4];
        let mut parent1_momentum = [f64x2::ZERO; 4];
        for component in 0..4 {
            left[component] = unsafe {
                load_current_pair(
                    arena,
                    row.parent0_component_base + component as u32,
                    stride,
                    point,
                )
            };
            right[component] = unsafe {
                load_current_pair(
                    arena,
                    row.parent1_component_base_or_sentinel + component as u32,
                    stride,
                    point,
                )
            };
            parent0_momentum[component] = unsafe {
                load_momentum_pair(
                    momenta,
                    row.parent0_momentum_form_id,
                    component as u16,
                    stride,
                    point,
                )
            };
            parent1_momentum[component] = unsafe {
                load_momentum_pair(
                    momenta,
                    row.parent1_momentum_form_id_or_sentinel,
                    component as u16,
                    stride,
                    point,
                )
            };
        }

        let vector_dot = minkowski_simd_complex_dot(left, right);
        let left_dot_parent1 = minkowski_simd_complex_real_dot(left, parent1_momentum);
        let right_dot_parent0 = minkowski_simd_complex_real_dot(right, parent0_momentum);
        let two = f64x2::new([2.0, 2.0]);
        for component in 0..4 {
            let momentum_term =
                vector_dot.mul_real(parent0_momentum[component] - parent1_momentum[component]);
            let current_term = left_dot_parent1
                .mul(right[component])
                .sub(right_dot_parent0.mul(left[component]))
                .mul_real(two);
            unsafe {
                add_scaled_current_pair(
                    arena,
                    row.destination_component_base + component as u32,
                    stride,
                    point,
                    momentum_term.add(current_term),
                    scale,
                    contribution_accumulates(row),
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let output = unsafe {
            <Self as DirectContributionFanoutFormula<4>>::evaluate(
                arena,
                Some(momenta),
                row,
                stride,
                point,
            )
        };
        for (component, value) in output.into_iter().enumerate() {
            unsafe {
                add_scaled_current(
                    arena,
                    row.destination_component_base + component as u32,
                    stride,
                    point,
                    value,
                    scale,
                    contribution_accumulates(row),
                );
            }
        }
    }
}

#[inline(always)]
fn color_ordered_three_vector_loaded(
    left: [ComplexValue; 4],
    right: [ComplexValue; 4],
    parent0_momentum: [f64; 4],
    parent1_momentum: [f64; 4],
) -> [ComplexValue; 4] {
    let vector_dot = minkowski_complex_dot(left, right);
    let left_dot_parent1 = minkowski_complex_real_dot(left, parent1_momentum);
    let right_dot_parent0 = minkowski_complex_real_dot(right, parent0_momentum);
    core::array::from_fn(|component| {
        let momentum_term =
            vector_dot.mul_real(parent0_momentum[component] - parent1_momentum[component]);
        let current_term = left_dot_parent1
            .mul(right[component])
            .sub(right_dot_parent0.mul(left[component]))
            .mul_real(2.0);
        momentum_term.add(current_term)
    })
}

impl DirectContributionFanoutFormula<4> for ColorOrderedThreeVector {
    const BASE_SCALE: ComplexValue = Self::BASE_SCALE;

    #[inline(always)]
    unsafe fn evaluate(
        arena: DirectArenaView,
        momenta: Option<DirectMomentumView>,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
    ) -> [ComplexValue; 4] {
        let momenta = momenta.expect("color-ordered fanout inputs were validated");
        let mut left = [ComplexValue::ZERO; 4];
        let mut right = [ComplexValue::ZERO; 4];
        let mut parent0_momentum = [0.0; 4];
        let mut parent1_momentum = [0.0; 4];
        for component in 0..4 {
            left[component] = unsafe {
                load_current(
                    arena,
                    row.parent0_component_base + component as u32,
                    stride,
                    point,
                )
            };
            right[component] = unsafe {
                load_current(
                    arena,
                    row.parent1_component_base_or_sentinel + component as u32,
                    stride,
                    point,
                )
            };
            parent0_momentum[component] = unsafe {
                load_momentum(
                    momenta,
                    row.parent0_momentum_form_id,
                    component as u16,
                    stride,
                    point,
                )
            };
            parent1_momentum[component] = unsafe {
                load_momentum(
                    momenta,
                    row.parent1_momentum_form_id_or_sentinel,
                    component as u16,
                    stride,
                    point,
                )
            };
        }

        color_ordered_three_vector_loaded(left, right, parent0_momentum, parent1_momentum)
    }
}

impl DirectContributionFormula for WeylVectorToWeylPositive {
    const PARENT0_COMPONENTS: u32 = 2;
    const PARENT1_COMPONENTS: u32 = 4;
    const DESTINATION_COMPONENTS: u32 = 2;
    const BASE_SCALE: ComplexValue = ComplexValue::new(1.0, 0.0);

    #[inline(always)]
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let l0 = unsafe { load_current(arena, row.parent0_component_base, stride, point) };
        let l1 = unsafe { load_current(arena, row.parent0_component_base + 1, stride, point) };
        let r0 =
            unsafe { load_current(arena, row.parent1_component_base_or_sentinel, stride, point) };
        let r1 = unsafe {
            load_current(
                arena,
                row.parent1_component_base_or_sentinel + 1,
                stride,
                point,
            )
        };
        let r2 = unsafe {
            load_current(
                arena,
                row.parent1_component_base_or_sentinel + 2,
                stride,
                point,
            )
        };
        let r3 = unsafe {
            load_current(
                arena,
                row.parent1_component_base_or_sentinel + 3,
                stride,
                point,
            )
        };

        let output0 = l0
            .mul(r3)
            .mul_i()
            .neg()
            .sub(l1.mul(r1).mul_i())
            .add(l1.mul(r2))
            .add(l0.mul(r0).mul_i());
        let output1 = l0
            .mul(r2)
            .neg()
            .sub(l0.mul(r1).mul_i())
            .add(l1.mul(r0).mul_i())
            .add(l1.mul(r3).mul_i());
        unsafe {
            add_scaled_current(
                arena,
                row.destination_component_base,
                stride,
                point,
                output0,
                scale,
                contribution_accumulates(row),
            );
            add_scaled_current(
                arena,
                row.destination_component_base + 1,
                stride,
                point,
                output1,
                scale,
                contribution_accumulates(row),
            );
        }
    }

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let l0 = unsafe { load_current_pair(arena, row.parent0_component_base, stride, point) };
        let l1 = unsafe { load_current_pair(arena, row.parent0_component_base + 1, stride, point) };
        let r0 = unsafe {
            load_current_pair(arena, row.parent1_component_base_or_sentinel, stride, point)
        };
        let r1 = unsafe {
            load_current_pair(
                arena,
                row.parent1_component_base_or_sentinel + 1,
                stride,
                point,
            )
        };
        let r2 = unsafe {
            load_current_pair(
                arena,
                row.parent1_component_base_or_sentinel + 2,
                stride,
                point,
            )
        };
        let r3 = unsafe {
            load_current_pair(
                arena,
                row.parent1_component_base_or_sentinel + 3,
                stride,
                point,
            )
        };

        let output0 = l0
            .mul(r3)
            .mul_i()
            .neg()
            .sub(l1.mul(r1).mul_i())
            .add(l1.mul(r2))
            .add(l0.mul(r0).mul_i());
        let output1 = l0
            .mul(r2)
            .neg()
            .sub(l0.mul(r1).mul_i())
            .add(l1.mul(r0).mul_i())
            .add(l1.mul(r3).mul_i());
        unsafe {
            add_scaled_current_pair(
                arena,
                row.destination_component_base,
                stride,
                point,
                output0,
                scale,
                contribution_accumulates(row),
            );
            add_scaled_current_pair(
                arena,
                row.destination_component_base + 1,
                stride,
                point,
                output1,
                scale,
                contribution_accumulates(row),
            );
        }
    }
}

impl DirectContributionFormula for WeylVectorToWeylNegative {
    const PARENT0_COMPONENTS: u32 = 2;
    const PARENT1_COMPONENTS: u32 = 4;
    const DESTINATION_COMPONENTS: u32 = 2;
    const BASE_SCALE: ComplexValue = ComplexValue::new(1.0, 0.0);

    #[inline(always)]
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let l0 = unsafe { load_current(arena, row.parent0_component_base, stride, point) };
        let l1 = unsafe { load_current(arena, row.parent0_component_base + 1, stride, point) };
        let r0 =
            unsafe { load_current(arena, row.parent1_component_base_or_sentinel, stride, point) };
        let r1 = unsafe {
            load_current(
                arena,
                row.parent1_component_base_or_sentinel + 1,
                stride,
                point,
            )
        };
        let r2 = unsafe {
            load_current(
                arena,
                row.parent1_component_base_or_sentinel + 2,
                stride,
                point,
            )
        };
        let r3 = unsafe {
            load_current(
                arena,
                row.parent1_component_base_or_sentinel + 3,
                stride,
                point,
            )
        };

        let output0 = l1
            .mul(r2)
            .neg()
            .add(l0.mul(r0).mul_i())
            .add(l0.mul(r3).mul_i())
            .add(l1.mul(r1).mul_i());
        let output1 = l1
            .mul(r3)
            .mul_i()
            .neg()
            .add(l0.mul(r2))
            .add(l0.mul(r1).mul_i())
            .add(l1.mul(r0).mul_i());
        unsafe {
            add_scaled_current(
                arena,
                row.destination_component_base,
                stride,
                point,
                output0,
                scale,
                contribution_accumulates(row),
            );
            add_scaled_current(
                arena,
                row.destination_component_base + 1,
                stride,
                point,
                output1,
                scale,
                contribution_accumulates(row),
            );
        }
    }

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let l0 = unsafe { load_current_pair(arena, row.parent0_component_base, stride, point) };
        let l1 = unsafe { load_current_pair(arena, row.parent0_component_base + 1, stride, point) };
        let r0 = unsafe {
            load_current_pair(arena, row.parent1_component_base_or_sentinel, stride, point)
        };
        let r1 = unsafe {
            load_current_pair(
                arena,
                row.parent1_component_base_or_sentinel + 1,
                stride,
                point,
            )
        };
        let r2 = unsafe {
            load_current_pair(
                arena,
                row.parent1_component_base_or_sentinel + 2,
                stride,
                point,
            )
        };
        let r3 = unsafe {
            load_current_pair(
                arena,
                row.parent1_component_base_or_sentinel + 3,
                stride,
                point,
            )
        };

        let output0 = l1
            .mul(r2)
            .neg()
            .add(l0.mul(r0).mul_i())
            .add(l0.mul(r3).mul_i())
            .add(l1.mul(r1).mul_i());
        let output1 = l1
            .mul(r3)
            .mul_i()
            .neg()
            .add(l0.mul(r2))
            .add(l0.mul(r1).mul_i())
            .add(l1.mul(r0).mul_i());
        unsafe {
            add_scaled_current_pair(
                arena,
                row.destination_component_base,
                stride,
                point,
                output0,
                scale,
                contribution_accumulates(row),
            );
            add_scaled_current_pair(
                arena,
                row.destination_component_base + 1,
                stride,
                point,
                output1,
                scale,
                contribution_accumulates(row),
            );
        }
    }
}

impl DirectContributionFormula for AntisymmetricTensorVectorToVector {
    const PARENT0_COMPONENTS: u32 = 6;
    const PARENT1_COMPONENTS: u32 = 4;
    const DESTINATION_COMPONENTS: u32 = 4;
    const BASE_SCALE: ComplexValue = ComplexValue::new(1.0, 0.0);

    #[inline(always)]
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let output = unsafe {
            <Self as DirectContributionFanoutFormula<4>>::evaluate(arena, None, row, stride, point)
        };
        for (component, value) in output.into_iter().enumerate() {
            unsafe {
                add_scaled_current(
                    arena,
                    row.destination_component_base + component as u32,
                    stride,
                    point,
                    value,
                    scale,
                    contribution_accumulates(row),
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let mut left = [SimdComplex2::new(f64x2::ZERO, f64x2::ZERO); 6];
        let mut right = [SimdComplex2::new(f64x2::ZERO, f64x2::ZERO); 4];
        for (component, value) in left.iter_mut().enumerate() {
            *value = unsafe {
                load_current_pair(
                    arena,
                    row.parent0_component_base + component as u32,
                    stride,
                    point,
                )
            };
        }
        for (component, value) in right.iter_mut().enumerate() {
            *value = unsafe {
                load_current_pair(
                    arena,
                    row.parent1_component_base_or_sentinel + component as u32,
                    stride,
                    point,
                )
            };
        }
        let output = [
            left[0]
                .mul(right[1])
                .add(left[1].mul(right[2]))
                .add(left[2].mul(right[3])),
            left[0]
                .mul(right[0])
                .add(left[3].mul(right[2]))
                .add(left[4].mul(right[3])),
            left[1]
                .mul(right[0])
                .sub(left[3].mul(right[1]))
                .add(left[5].mul(right[3])),
            left[2]
                .mul(right[0])
                .sub(left[4].mul(right[1]))
                .sub(left[5].mul(right[2])),
        ];
        for (component, value) in output.into_iter().enumerate() {
            unsafe {
                add_scaled_current_pair(
                    arena,
                    row.destination_component_base + component as u32,
                    stride,
                    point,
                    value,
                    scale,
                    contribution_accumulates(row),
                );
            }
        }
    }
}

impl DirectContributionFanoutFormula<4> for AntisymmetricTensorVectorToVector {
    const BASE_SCALE: ComplexValue = <Self as DirectContributionFormula>::BASE_SCALE;

    #[inline(always)]
    unsafe fn evaluate(
        arena: DirectArenaView,
        _momenta: Option<DirectMomentumView>,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
    ) -> [ComplexValue; 4] {
        let mut left = [ComplexValue::ZERO; 6];
        let mut right = [ComplexValue::ZERO; 4];
        for (component, value) in left.iter_mut().enumerate() {
            *value = unsafe {
                load_current(
                    arena,
                    row.parent0_component_base + component as u32,
                    stride,
                    point,
                )
            };
        }
        for (component, value) in right.iter_mut().enumerate() {
            *value = unsafe {
                load_current(
                    arena,
                    row.parent1_component_base_or_sentinel + component as u32,
                    stride,
                    point,
                )
            };
        }
        antisymmetric_tensor_vector_loaded(left, right)
    }
}

#[inline(always)]
fn antisymmetric_tensor_vector_loaded(
    left: [ComplexValue; 6],
    right: [ComplexValue; 4],
) -> [ComplexValue; 4] {
    [
        left[0]
            .mul(right[1])
            .add(left[1].mul(right[2]))
            .add(left[2].mul(right[3])),
        left[0]
            .mul(right[0])
            .add(left[3].mul(right[2]))
            .add(left[4].mul(right[3])),
        left[1]
            .mul(right[0])
            .sub(left[3].mul(right[1]))
            .add(left[5].mul(right[3])),
        left[2]
            .mul(right[0])
            .sub(left[4].mul(right[1]))
            .sub(left[5].mul(right[2])),
    ]
}

impl DirectContributionFormula for VectorWedgeVectorToAntisymmetricTensor {
    const PARENT0_COMPONENTS: u32 = 4;
    const PARENT1_COMPONENTS: u32 = 4;
    const DESTINATION_COMPONENTS: u32 = 6;
    const BASE_SCALE: ComplexValue = ComplexValue::new(1.0, 0.0);

    #[inline(always)]
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let output = unsafe {
            <Self as DirectContributionFanoutFormula<6>>::evaluate(arena, None, row, stride, point)
        };
        for (component, value) in output.into_iter().enumerate() {
            unsafe {
                add_scaled_current(
                    arena,
                    row.destination_component_base + component as u32,
                    stride,
                    point,
                    value,
                    scale,
                    contribution_accumulates(row),
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
        scale: ComplexValue,
    ) {
        let mut left = [SimdComplex2::new(f64x2::ZERO, f64x2::ZERO); 4];
        let mut right = [SimdComplex2::new(f64x2::ZERO, f64x2::ZERO); 4];
        for component in 0..4 {
            left[component] = unsafe {
                load_current_pair(
                    arena,
                    row.parent0_component_base + component as u32,
                    stride,
                    point,
                )
            };
            right[component] = unsafe {
                load_current_pair(
                    arena,
                    row.parent1_component_base_or_sentinel + component as u32,
                    stride,
                    point,
                )
            };
        }
        let output = [
            left[0].mul(right[1]).sub(left[1].mul(right[0])),
            left[0].mul(right[2]).sub(left[2].mul(right[0])),
            left[0].mul(right[3]).sub(left[3].mul(right[0])),
            left[1].mul(right[2]).sub(left[2].mul(right[1])),
            left[1].mul(right[3]).sub(left[3].mul(right[1])),
            left[2].mul(right[3]).sub(left[3].mul(right[2])),
        ];
        for (component, value) in output.into_iter().enumerate() {
            unsafe {
                add_scaled_current_pair(
                    arena,
                    row.destination_component_base + component as u32,
                    stride,
                    point,
                    value,
                    scale,
                    contribution_accumulates(row),
                );
            }
        }
    }
}

impl DirectContributionFanoutFormula<6> for VectorWedgeVectorToAntisymmetricTensor {
    const BASE_SCALE: ComplexValue = <Self as DirectContributionFormula>::BASE_SCALE;

    #[inline(always)]
    unsafe fn evaluate(
        arena: DirectArenaView,
        _momenta: Option<DirectMomentumView>,
        row: DirectContributionRow,
        stride: usize,
        point: usize,
    ) -> [ComplexValue; 6] {
        let mut left = [ComplexValue::ZERO; 4];
        let mut right = [ComplexValue::ZERO; 4];
        for component in 0..4 {
            left[component] = unsafe {
                load_current(
                    arena,
                    row.parent0_component_base + component as u32,
                    stride,
                    point,
                )
            };
            right[component] = unsafe {
                load_current(
                    arena,
                    row.parent1_component_base_or_sentinel + component as u32,
                    stride,
                    point,
                )
            };
        }
        vector_wedge_vector_loaded(left, right)
    }
}

#[inline(always)]
fn vector_wedge_vector_loaded(
    left: [ComplexValue; 4],
    right: [ComplexValue; 4],
) -> [ComplexValue; 6] {
    [
        left[0].mul(right[1]).sub(left[1].mul(right[0])),
        left[0].mul(right[2]).sub(left[2].mul(right[0])),
        left[0].mul(right[3]).sub(left[3].mul(right[0])),
        left[1].mul(right[2]).sub(left[2].mul(right[1])),
        left[1].mul(right[3]).sub(left[3].mul(right[1])),
        left[2].mul(right[3]).sub(left[3].mul(right[2])),
    ]
}

#[allow(clippy::too_many_arguments)]
unsafe fn execute_contribution_fanout<F, const N: usize>(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    batch: DirectContributionFanoutBatch<'_>,
) -> RusticolResult<()>
where
    F: DirectContributionFanoutFormula<N>,
{
    // The private batch is produced only by a cold-authenticated direct or
    // family-local fanout program. Its row spans and all raw views were
    // validated before this call.
    let context = unsafe { &*context.cast::<RecurrenceIntrinsicScale>() };
    let context_scale = effective_context_scale(*context, parameters)
        .map_err(|_| RusticolError::evaluation("fused contribution fanout scale is unavailable"))?
        .mul(F::BASE_SCALE);
    let stride = arena.point_stride as usize;

    for run in batch.runs {
        let row_start = usize::try_from(run.row_start)
            .map_err(|_| RusticolError::integrity("fused fanout row start exceeds usize"))?;
        let row_end = row_start
            .checked_add(run.row_count as usize)
            .ok_or_else(|| RusticolError::integrity("fused fanout row range overflows usize"))?;
        let run_rows = batch
            .rows
            .get(row_start..row_end)
            .ok_or_else(|| RusticolError::integrity("fused fanout row range is out of bounds"))?;
        let representative = *run_rows.first().ok_or_else(|| {
            RusticolError::integrity("fused fanout run has no representative row")
        })?;
        let mut output = unsafe { F::evaluate(arena, Some(momenta), representative, stride, 0) };
        for value in &mut output {
            *value = value.mul(context_scale);
        }

        let bundle_start = usize::try_from(run.bundle_start)
            .map_err(|_| RusticolError::integrity("fused fanout bundle start exceeds usize"))?;
        let bundle_end = bundle_start
            .checked_add(run.bundle_count as usize)
            .ok_or_else(|| RusticolError::integrity("fused fanout bundle range overflows usize"))?;
        let bundles = batch.bundles.get(bundle_start..bundle_end).ok_or_else(|| {
            RusticolError::integrity("fused fanout bundle range is out of bounds")
        })?;
        for bundle in bundles {
            if bundle.effective_factor_id_or_sentinel == DIRECT_NONE_U32 {
                return Err(RusticolError::integrity(
                    "fused contribution fanout requires an explicit destination factor",
                ));
            }
            let factor = bundle.effective_factor_id_or_sentinel as usize;
            if factor >= factors.value_count as usize {
                return Err(RusticolError::integrity(
                    "fused contribution fanout factor is out of bounds",
                ));
            }
            let scale = DirectContributionFanoutScale::new(
                unsafe { *factors.values_re.add(factor) },
                unsafe { *factors.values_im.add(factor) },
            );
            let scaled_output = output.map(|value| {
                let (re, im) = scale.apply(value.re, value.im);
                ComplexValue::new(re, im)
            });
            let bundle_row_start = usize::try_from(bundle.row_start).map_err(|_| {
                RusticolError::integrity("fused fanout bundle row start exceeds usize")
            })?;
            let bundle_row_end = bundle_row_start
                .checked_add(bundle.row_count as usize)
                .ok_or_else(|| {
                    RusticolError::integrity("fused fanout bundle row range overflows usize")
                })?;
            let rows = batch
                .rows
                .get(bundle_row_start..bundle_row_end)
                .ok_or_else(|| {
                    RusticolError::integrity("fused fanout bundle row range is out of bounds")
                })?;
            for row in rows {
                for (component, value) in scaled_output.iter().enumerate() {
                    let destination =
                        (row.destination_component_base as usize + component) * stride;
                    unsafe {
                        if contribution_accumulates(*row) {
                            *arena.current_re.add(destination) += value.re;
                            *arena.current_im.add(destination) += value.im;
                        } else {
                            *arena.current_re.add(destination) = value.re;
                            *arena.current_im.add(destination) = value.im;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

unsafe fn execute_antisymmetric_tensor_vector_to_vector_fanout(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    batch: DirectContributionFanoutBatch<'_>,
) -> RusticolResult<()> {
    unsafe {
        execute_contribution_fanout::<AntisymmetricTensorVectorToVector, 4>(
            context, arena, momenta, parameters, factors, batch,
        )
    }
}

unsafe fn execute_vector_wedge_vector_to_antisymmetric_tensor_fanout(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    batch: DirectContributionFanoutBatch<'_>,
) -> RusticolResult<()> {
    unsafe {
        execute_contribution_fanout::<VectorWedgeVectorToAntisymmetricTensor, 6>(
            context, arena, momenta, parameters, factors, batch,
        )
    }
}

unsafe fn execute_color_ordered_three_vector_fanout(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    batch: DirectContributionFanoutBatch<'_>,
) -> RusticolResult<()> {
    if momenta.values.is_null()
        || momenta.lorentz_component_count != 4
        || momenta.point_stride != arena.point_stride
    {
        return Err(RusticolError::evaluation(
            "color-ordered fused fanout requires four-component momenta at the arena point stride",
        ));
    }
    unsafe {
        execute_contribution_fanout::<ColorOrderedThreeVector, 4>(
            context, arena, momenta, parameters, factors, batch,
        )
    }
}

unsafe fn execute_interaction_bundle(
    contexts: DirectInteractionExecutorContexts,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    batch: DirectInteractionBundleBatch<'_>,
) -> RusticolResult<()> {
    if contexts.color.is_null()
        || contexts.antisymmetric_tensor_vector.is_null()
        || (!batch.tensor_outputs.is_empty() && contexts.vector_wedge_vector.is_null())
    {
        return Err(RusticolError::evaluation(
            "direct interaction bundle has a null intrinsic context",
        ));
    }
    let stride = validate_interaction_runtime_views(
        arena,
        momenta,
        parameters,
        factors,
        batch.requirements,
    )?;
    let color_context = unsafe { &*contexts.color.cast::<RecurrenceIntrinsicScale>() };
    let atv_context = unsafe {
        &*contexts
            .antisymmetric_tensor_vector
            .cast::<RecurrenceIntrinsicScale>()
    };
    let color_context_scale = effective_context_scale(*color_context, parameters)
        .map_err(|_| RusticolError::evaluation("interaction color scale is unavailable"))?;
    let atv_context_scale = effective_context_scale(*atv_context, parameters)
        .map_err(|_| RusticolError::evaluation("interaction tensor-vector scale is unavailable"))?;
    if batch.tensor_outputs.is_empty() && batch.terminal.is_none() {
        return unsafe {
            execute_vector_interaction_bundle(
                arena,
                momenta,
                factors,
                batch,
                stride,
                color_context_scale,
                atv_context_scale,
            )
        };
    }
    let wedge_context_scale = if batch.tensor_outputs.is_empty() {
        None
    } else {
        let wedge_context = unsafe {
            &*contexts
                .vector_wedge_vector
                .cast::<RecurrenceIntrinsicScale>()
        };
        Some(
            effective_context_scale(*wedge_context, parameters).map_err(|_| {
                RusticolError::evaluation("interaction vector-wedge scale is unavailable")
            })?,
        )
    };
    let terminal_inputs = if let Some(terminal) = batch.terminal {
        let mut partner = [ComplexValue::ZERO; 4];
        let mut projection_weights = [ComplexValue::ZERO; 4];
        for component in 0..4 {
            partner[component] = unsafe {
                load_current(
                    arena,
                    terminal.partner_component_base + component as u32,
                    stride,
                    0,
                )
            };
            let factor_id = terminal.component_factor_start + component as u32;
            let component_factor = ComplexValue::new(
                unsafe { *factors.values_re.add(factor_id as usize) },
                unsafe { *factors.values_im.add(factor_id as usize) },
            );
            projection_weights[component] = partner[component].mul(component_factor);
        }
        Some(projection_weights)
    } else {
        None
    };

    for anchor in batch.anchors {
        let mut left = [ComplexValue::ZERO; 4];
        let mut right = [ComplexValue::ZERO; 4];
        let mut parent0_momentum = [0.0; 4];
        let mut parent1_momentum = [0.0; 4];
        for component in 0..4 {
            left[component] = unsafe {
                load_current(
                    arena,
                    anchor.parent_component_bases[0] + component as u32,
                    stride,
                    0,
                )
            };
            right[component] = unsafe {
                load_current(
                    arena,
                    anchor.parent_component_bases[1] + component as u32,
                    stride,
                    0,
                )
            };
            parent0_momentum[component] = unsafe {
                load_momentum(
                    momenta,
                    anchor.parent_momentum_form_ids[0],
                    component as u16,
                    stride,
                    0,
                )
            };
            parent1_momentum[component] = unsafe {
                load_momentum(
                    momenta,
                    anchor.parent_momentum_form_ids[1],
                    component as u16,
                    stride,
                    0,
                )
            };
        }
        let color_output =
            color_ordered_three_vector_loaded(left, right, parent0_momentum, parent1_momentum)
                .map(|value| value.mul(color_context_scale));
        let wedge_output = wedge_context_scale
            .map(|scale| vector_wedge_vector_loaded(left, right).map(|value| value.mul(scale)));

        let mut atv_outputs = [[ComplexValue::ZERO; 4]; 2];
        let atv_count = match anchor.operands {
            DirectInteractionOperands::None => 0,
            DirectInteractionOperands::One(operand) => {
                atv_outputs[0] = unsafe {
                    evaluate_interaction_operand(
                        operand,
                        left,
                        right,
                        arena,
                        stride,
                        atv_context_scale,
                    )
                };
                1
            }
            DirectInteractionOperands::Two(operands) => {
                atv_outputs[0] = unsafe {
                    evaluate_interaction_operand(
                        operands[0],
                        left,
                        right,
                        arena,
                        stride,
                        atv_context_scale,
                    )
                };
                atv_outputs[1] = unsafe {
                    evaluate_interaction_operand(
                        operands[1],
                        left,
                        right,
                        arena,
                        stride,
                        atv_context_scale,
                    )
                };
                2
            }
        };
        let outputs = batch
            .outputs
            .get(anchor.output_start as usize..anchor.output_end as usize)
            .ok_or_else(|| RusticolError::integrity("interaction output range is out of bounds"))?;

        if let Some(projection_weights) = terminal_inputs {
            let projected_color = project_terminal_vector(color_output, projection_weights);
            let mut projected_atv = [ComplexValue::ZERO; 2];
            for slot in 0..atv_count {
                projected_atv[slot] =
                    project_terminal_vector(atv_outputs[slot], projection_weights);
            }
            for output in outputs {
                let color_scale =
                    unsafe { live_interaction_factor(factors, output.exact_factor_ids[0]) };
                let scaled_color = apply_fanout_scale(projected_color, color_scale);
                let mut combined = if batch.atv_before_color {
                    ComplexValue::ZERO
                } else {
                    scaled_color
                };
                for (slot, projected) in projected_atv.iter().take(atv_count).enumerate() {
                    let scale = unsafe {
                        live_interaction_factor(factors, output.exact_factor_ids[slot + 1])
                    };
                    combined = combined.add(apply_fanout_scale(*projected, scale));
                }
                if batch.atv_before_color {
                    combined = combined.add(scaled_color);
                }
                let destination = output.destination_component_base as usize * stride;
                unsafe {
                    if output.initialize {
                        *arena.current_re.add(destination) = combined.re;
                        *arena.current_im.add(destination) = combined.im;
                    } else {
                        *arena.current_re.add(destination) += combined.re;
                        *arena.current_im.add(destination) += combined.im;
                    }
                }
            }
        } else {
            for output in outputs {
                let color_scale =
                    unsafe { live_interaction_factor(factors, output.exact_factor_ids[0]) };
                let scaled_color = color_output.map(|value| apply_fanout_scale(value, color_scale));
                let mut combined = if batch.atv_before_color {
                    [ComplexValue::ZERO; 4]
                } else {
                    scaled_color
                };
                for (slot, atv_output) in atv_outputs.iter().take(atv_count).enumerate() {
                    let scale = unsafe {
                        live_interaction_factor(factors, output.exact_factor_ids[slot + 1])
                    };
                    for component in 0..4 {
                        combined[component] = combined[component]
                            .add(apply_fanout_scale(atv_output[component], scale));
                    }
                }
                if batch.atv_before_color {
                    for component in 0..4 {
                        combined[component] = combined[component].add(scaled_color[component]);
                    }
                }
                for (component, value) in combined.into_iter().enumerate() {
                    let destination =
                        (output.destination_component_base as usize + component) * stride;
                    unsafe {
                        if output.initialize {
                            *arena.current_re.add(destination) = value.re;
                            *arena.current_im.add(destination) = value.im;
                        } else {
                            *arena.current_re.add(destination) += value.re;
                            *arena.current_im.add(destination) += value.im;
                        }
                    }
                }
            }
        }
        let tensor_outputs = batch
            .tensor_outputs
            .get(anchor.tensor_output_start as usize..anchor.tensor_output_end as usize)
            .ok_or_else(|| {
                RusticolError::integrity("interaction tensor-output range is out of bounds")
            })?;
        if !tensor_outputs.is_empty() {
            let wedge_output = wedge_output.ok_or_else(|| {
                RusticolError::integrity("interaction tensor outputs have no wedge evaluation")
            })?;
            for output in tensor_outputs {
                let scale = unsafe { live_interaction_factor(factors, output.exact_factor_id) };
                for (component, value) in wedge_output.into_iter().enumerate() {
                    let value = apply_fanout_scale(value, scale);
                    let destination =
                        (output.destination_component_base as usize + component) * stride;
                    unsafe {
                        if output.initialize {
                            *arena.current_re.add(destination) = value.re;
                            *arena.current_im.add(destination) = value.im;
                        } else {
                            *arena.current_re.add(destination) += value.re;
                            *arena.current_im.add(destination) += value.im;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn execute_vector_interaction_bundle(
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    factors: DirectFactorView,
    batch: DirectInteractionBundleBatch<'_>,
    stride: usize,
    color_context_scale: ComplexValue,
    atv_context_scale: ComplexValue,
) -> RusticolResult<()> {
    for anchor in batch.anchors {
        let mut left = [ComplexValue::ZERO; 4];
        let mut right = [ComplexValue::ZERO; 4];
        let mut parent0_momentum = [0.0; 4];
        let mut parent1_momentum = [0.0; 4];
        for component in 0..4 {
            left[component] = unsafe {
                load_current(
                    arena,
                    anchor.parent_component_bases[0] + component as u32,
                    stride,
                    0,
                )
            };
            right[component] = unsafe {
                load_current(
                    arena,
                    anchor.parent_component_bases[1] + component as u32,
                    stride,
                    0,
                )
            };
            parent0_momentum[component] = unsafe {
                load_momentum(
                    momenta,
                    anchor.parent_momentum_form_ids[0],
                    component as u16,
                    stride,
                    0,
                )
            };
            parent1_momentum[component] = unsafe {
                load_momentum(
                    momenta,
                    anchor.parent_momentum_form_ids[1],
                    component as u16,
                    stride,
                    0,
                )
            };
        }
        let color_output =
            color_ordered_three_vector_loaded(left, right, parent0_momentum, parent1_momentum)
                .map(|value| value.mul(color_context_scale));
        let mut atv_outputs = [[ComplexValue::ZERO; 4]; 2];
        let atv_count = match anchor.operands {
            DirectInteractionOperands::None => 0,
            DirectInteractionOperands::One(operand) => {
                atv_outputs[0] = unsafe {
                    evaluate_interaction_operand(
                        operand,
                        left,
                        right,
                        arena,
                        stride,
                        atv_context_scale,
                    )
                };
                1
            }
            DirectInteractionOperands::Two(operands) => {
                atv_outputs[0] = unsafe {
                    evaluate_interaction_operand(
                        operands[0],
                        left,
                        right,
                        arena,
                        stride,
                        atv_context_scale,
                    )
                };
                atv_outputs[1] = unsafe {
                    evaluate_interaction_operand(
                        operands[1],
                        left,
                        right,
                        arena,
                        stride,
                        atv_context_scale,
                    )
                };
                2
            }
        };
        let outputs = batch
            .outputs
            .get(anchor.output_start as usize..anchor.output_end as usize)
            .ok_or_else(|| RusticolError::integrity("interaction output range is out of bounds"))?;
        for output in outputs {
            let color_scale =
                unsafe { live_interaction_factor(factors, output.exact_factor_ids[0]) };
            let scaled_color = color_output.map(|value| apply_fanout_scale(value, color_scale));
            let mut combined = if batch.atv_before_color {
                [ComplexValue::ZERO; 4]
            } else {
                scaled_color
            };
            for (slot, atv_output) in atv_outputs.iter().take(atv_count).enumerate() {
                let scale =
                    unsafe { live_interaction_factor(factors, output.exact_factor_ids[slot + 1]) };
                for component in 0..4 {
                    combined[component] =
                        combined[component].add(apply_fanout_scale(atv_output[component], scale));
                }
            }
            if batch.atv_before_color {
                for component in 0..4 {
                    combined[component] = combined[component].add(scaled_color[component]);
                }
            }
            for (component, value) in combined.into_iter().enumerate() {
                let destination = (output.destination_component_base as usize + component) * stride;
                unsafe {
                    if output.initialize {
                        *arena.current_re.add(destination) = value.re;
                        *arena.current_im.add(destination) = value.im;
                    } else {
                        *arena.current_re.add(destination) += value.re;
                        *arena.current_im.add(destination) += value.im;
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) unsafe fn execute_interaction_terminal_closures(
    arena: DirectArenaView,
    factors: DirectFactorView,
    terminal: DirectInteractionTerminalBatch<'_>,
) -> RusticolResult<()> {
    if arena.point_stride == 0
        || arena.current_re.is_null()
        || arena.current_im.is_null()
        || arena.amplitude_re.is_null()
        || arena.amplitude_im.is_null()
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return Err(RusticolError::evaluation(
            "terminal interaction closure arenas are unavailable",
        ));
    }
    let stride = arena.point_stride as usize;
    let required_amplitude_scalars = u64::from(terminal.amplitude_destination_count)
        .checked_mul(u64::from(arena.point_stride))
        .ok_or_else(|| RusticolError::evaluation("terminal amplitude range overflows"))?;
    if required_amplitude_scalars > arena.amplitude_scalar_len {
        return Err(RusticolError::evaluation(
            "terminal interaction amplitude arena is too short",
        ));
    }
    for closure in terminal.closures {
        let root = unsafe { load_current(arena, closure.root_component_base, stride, 0) };
        let row_factor = unsafe { live_interaction_factor(factors, closure.exact_factor_id) };
        let value = apply_fanout_scale(root, row_factor);
        let destination = closure.amplitude_destination_id as usize * stride;
        unsafe {
            *arena.amplitude_re.add(destination) += value.re;
            *arena.amplitude_im.add(destination) += value.im;
        }
    }
    Ok(())
}

#[inline(always)]
fn project_terminal_vector(
    vector: [ComplexValue; 4],
    projection_weights: [ComplexValue; 4],
) -> ComplexValue {
    let mut projected = ComplexValue::ZERO;
    for component in 0..4 {
        projected = projected.add(vector[component].mul(projection_weights[component]));
    }
    projected
}

#[inline(always)]
unsafe fn evaluate_interaction_operand(
    operand: DirectInteractionOperand,
    left: [ComplexValue; 4],
    right: [ComplexValue; 4],
    arena: DirectArenaView,
    stride: usize,
    context_scale: ComplexValue,
) -> [ComplexValue; 4] {
    let mut tensor = [ComplexValue::ZERO; 6];
    for (component, value) in tensor.iter_mut().enumerate() {
        *value = unsafe {
            load_current(
                arena,
                operand.tensor_component_base + component as u32,
                stride,
                0,
            )
        };
    }
    let vector = match operand.vector_parent {
        DirectInteractionVectorParent::Parent0 => left,
        DirectInteractionVectorParent::Parent1 => right,
    };
    antisymmetric_tensor_vector_loaded(tensor, vector).map(|value| value.mul(context_scale))
}

fn validate_interaction_runtime_views(
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    requirements: DirectInteractionRuntimeRequirements,
) -> RusticolResult<usize> {
    if arena.point_stride == 0 || arena.current_re.is_null() || arena.current_im.is_null() {
        return Err(RusticolError::evaluation(
            "direct interaction current arena is unavailable",
        ));
    }
    let required_current_scalars = u64::from(requirements.current_component_count)
        .checked_mul(u64::from(arena.point_stride))
        .ok_or_else(|| RusticolError::evaluation("direct interaction current arena overflows"))?;
    if arena.current_scalar_len < required_current_scalars
        || usize::try_from(required_current_scalars).is_err()
    {
        return Err(RusticolError::evaluation(
            "direct interaction current arena is too short",
        ));
    }
    let required_momentum_scalars = u64::from(requirements.momentum_form_count)
        .checked_mul(4)
        .and_then(|count| count.checked_mul(u64::from(arena.point_stride)))
        .ok_or_else(|| RusticolError::evaluation("direct interaction momentum arena overflows"))?;
    if momenta.values.is_null()
        || momenta.lorentz_component_count != 4
        || momenta.point_stride != arena.point_stride
        || momenta.form_count < requirements.momentum_form_count
        || momenta.scalar_len < required_momentum_scalars
        || usize::try_from(required_momentum_scalars).is_err()
    {
        return Err(RusticolError::evaluation(
            "direct interaction momentum arena does not cover its cold program",
        ));
    }
    if requirements.parameter_count != 0
        && (parameters.values_re.is_null()
            || parameters.values_im.is_null()
            || parameters.value_count < requirements.parameter_count)
    {
        return Err(RusticolError::evaluation(
            "direct interaction parameter catalog does not cover its cold program",
        ));
    }
    if requirements.factor_count != 0
        && (factors.values_re.is_null()
            || factors.values_im.is_null()
            || factors.value_count < requirements.factor_count)
    {
        return Err(RusticolError::evaluation(
            "direct interaction factor catalog does not cover its cold program",
        ));
    }
    usize::try_from(arena.point_stride)
        .map_err(|_| RusticolError::evaluation("direct interaction point stride exceeds usize"))
}

#[inline(always)]
unsafe fn live_interaction_factor(
    factors: DirectFactorView,
    factor_id: u32,
) -> DirectContributionFanoutScale {
    DirectContributionFanoutScale::new(
        unsafe { *factors.values_re.add(factor_id as usize) },
        unsafe { *factors.values_im.add(factor_id as usize) },
    )
}

#[inline(always)]
fn apply_fanout_scale(value: ComplexValue, scale: DirectContributionFanoutScale) -> ComplexValue {
    let (re, im) = scale.apply(value.re, value.im);
    ComplexValue::new(re, im)
}

unsafe extern "C" fn execute_weyl_vector_to_weyl_positive_rows(
    context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    unsafe {
        execute_rows::<WeylVectorToWeylPositive>(
            context,
            arena,
            parameters,
            factors,
            rows,
            row_count,
            point_count,
        )
    }
}

unsafe extern "C" fn execute_weyl_vector_to_weyl_negative_rows(
    context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    unsafe {
        execute_rows::<WeylVectorToWeylNegative>(
            context,
            arena,
            parameters,
            factors,
            rows,
            row_count,
            point_count,
        )
    }
}

unsafe extern "C" fn execute_antisymmetric_tensor_vector_to_vector_rows(
    context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    unsafe {
        execute_rows::<AntisymmetricTensorVectorToVector>(
            context,
            arena,
            parameters,
            factors,
            rows,
            row_count,
            point_count,
        )
    }
}

unsafe extern "C" fn execute_vector_wedge_vector_to_antisymmetric_tensor_rows(
    context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    unsafe {
        execute_rows::<VectorWedgeVectorToAntisymmetricTensor>(
            context,
            arena,
            parameters,
            factors,
            rows,
            row_count,
            point_count,
        )
    }
}

unsafe extern "C" fn execute_color_ordered_three_vector_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    if context.is_null() {
        return STATUS_INVALID_CONTEXT;
    }
    if row_count == 0
        || rows.is_null()
        || point_count == 0
        || arena.point_stride == 0
        || point_count > arena.point_stride
        || arena.current_re.is_null()
        || arena.current_im.is_null()
        || momenta.values.is_null()
        || momenta.lorentz_component_count != 4
        || momenta.point_stride != arena.point_stride
        || point_count > momenta.point_stride
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let context = unsafe { &*context.cast::<RecurrenceIntrinsicScale>() };
    let context_scale = match effective_context_scale(*context, parameters) {
        Ok(scale) => scale.mul(ColorOrderedThreeVector::BASE_SCALE),
        Err(status) => return status,
    };
    let stride = arena.point_stride as usize;
    let points = point_count as usize;

    for row_index in 0..row_count as usize {
        let row = unsafe { *rows.add(row_index) };
        if row.parent1_component_base_or_sentinel == DIRECT_NONE_U32
            || row.parent0_momentum_form_id == DIRECT_NONE_U32
            || row.parent1_momentum_form_id_or_sentinel == DIRECT_NONE_U32
        {
            return STATUS_INVALID_ARGUMENT;
        }
        if row.exact_factor_id >= factors.value_count
            || !component_range_in_bounds(
                row.parent0_component_base,
                ColorOrderedThreeVector::PARENT_COMPONENTS,
                arena.current_scalar_len,
                arena.point_stride,
                point_count,
            )
            || !component_range_in_bounds(
                row.parent1_component_base_or_sentinel,
                ColorOrderedThreeVector::PARENT_COMPONENTS,
                arena.current_scalar_len,
                arena.point_stride,
                point_count,
            )
            || !component_range_in_bounds(
                row.destination_component_base,
                ColorOrderedThreeVector::DESTINATION_COMPONENTS,
                arena.current_scalar_len,
                arena.point_stride,
                point_count,
            )
            || !momentum_form_in_bounds(momenta, row.parent0_momentum_form_id, point_count)
            || !momentum_form_in_bounds(
                momenta,
                row.parent1_momentum_form_id_or_sentinel,
                point_count,
            )
        {
            return STATUS_BOUNDS;
        }
        let row_factor = ComplexValue::new(
            unsafe { *factors.values_re.add(row.exact_factor_id as usize) },
            unsafe { *factors.values_im.add(row.exact_factor_id as usize) },
        );
        let scale = context_scale.mul(row_factor);
        let mut point = 0usize;
        while point + 1 < points {
            unsafe {
                ColorOrderedThreeVector::evaluate_pair(arena, momenta, row, stride, point, scale);
            }
            point += 2;
        }
        if point < points {
            unsafe {
                ColorOrderedThreeVector::evaluate_point(arena, momenta, row, stride, point, scale);
            }
        }
    }
    DIRECT_STATUS_OK
}

trait DirectFinalizationFormula {
    const COMPONENTS: u16;

    unsafe fn evaluate_point(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectFinalizationRow,
        stride: usize,
        point: usize,
        row_factor: ComplexValue,
    );

    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectFinalizationRow,
        stride: usize,
        point: usize,
        row_factor: ComplexValue,
    ) {
        unsafe {
            Self::evaluate_point(arena, momenta, row, stride, point, row_factor);
            Self::evaluate_point(arena, momenta, row, stride, point + 1, row_factor);
        }
    }
}

struct WeylPropagatorPositive;
struct WeylPropagatorNegative;
struct FeynmanVectorPropagator;

impl DirectFinalizationFormula for WeylPropagatorPositive {
    const COMPONENTS: u16 = 2;

    #[inline(always)]
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectFinalizationRow,
        stride: usize,
        point: usize,
        row_factor: ComplexValue,
    ) {
        let c0 = unsafe { load_current(arena, row.component_base, stride, point) };
        let c1 = unsafe { load_current(arena, row.component_base + 1, stride, point) };
        let p0 = unsafe { load_momentum(momenta, row.momentum_form_id, 0, stride, point) };
        let p1 = unsafe { load_momentum(momenta, row.momentum_form_id, 1, stride, point) };
        let p2 = unsafe { load_momentum(momenta, row.momentum_form_id, 2, stride, point) };
        let p3 = unsafe { load_momentum(momenta, row.momentum_form_id, 3, stride, point) };
        let inv = reciprocal_minkowski_norm(p0, p1, p2, p3);
        let scale = row_factor.mul_real(inv);
        let output0 = c0
            .mul_real(p3)
            .mul_i()
            .neg()
            .sub(c1.mul_real(p1).mul_i())
            .add(c1.mul_real(p2))
            .add(c0.mul_real(p0).mul_i());
        let output1 = c0
            .mul_real(p2)
            .neg()
            .sub(c0.mul_real(p1).mul_i())
            .add(c1.mul_real(p0).mul_i())
            .add(c1.mul_real(p3).mul_i());
        unsafe {
            set_scaled_current(arena, row.component_base, stride, point, output0, scale);
            set_scaled_current(arena, row.component_base + 1, stride, point, output1, scale);
        }
    }

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectFinalizationRow,
        stride: usize,
        point: usize,
        row_factor: ComplexValue,
    ) {
        let c0 = unsafe { load_current_pair(arena, row.component_base, stride, point) };
        let c1 = unsafe { load_current_pair(arena, row.component_base + 1, stride, point) };
        let p0 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 0, stride, point) };
        let p1 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 1, stride, point) };
        let p2 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 2, stride, point) };
        let p3 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 3, stride, point) };
        let inv = reciprocal_minkowski_norm_pair(p0, p1, p2, p3);
        let scale = row_factor.mul_real_pair(inv);
        let output0 = c0
            .mul_real(p3)
            .mul_i()
            .neg()
            .sub(c1.mul_real(p1).mul_i())
            .add(c1.mul_real(p2))
            .add(c0.mul_real(p0).mul_i());
        let output1 = c0
            .mul_real(p2)
            .neg()
            .sub(c0.mul_real(p1).mul_i())
            .add(c1.mul_real(p0).mul_i())
            .add(c1.mul_real(p3).mul_i());
        unsafe {
            set_pre_scaled_current_pair(arena, row.component_base, stride, point, output0, scale);
            set_pre_scaled_current_pair(
                arena,
                row.component_base + 1,
                stride,
                point,
                output1,
                scale,
            );
        }
    }
}

impl DirectFinalizationFormula for WeylPropagatorNegative {
    const COMPONENTS: u16 = 2;

    #[inline(always)]
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectFinalizationRow,
        stride: usize,
        point: usize,
        row_factor: ComplexValue,
    ) {
        let c0 = unsafe { load_current(arena, row.component_base, stride, point) };
        let c1 = unsafe { load_current(arena, row.component_base + 1, stride, point) };
        let p0 = unsafe { load_momentum(momenta, row.momentum_form_id, 0, stride, point) };
        let p1 = unsafe { load_momentum(momenta, row.momentum_form_id, 1, stride, point) };
        let p2 = unsafe { load_momentum(momenta, row.momentum_form_id, 2, stride, point) };
        let p3 = unsafe { load_momentum(momenta, row.momentum_form_id, 3, stride, point) };
        let inv = reciprocal_minkowski_norm(p0, p1, p2, p3);
        let scale = row_factor.mul_real(inv);
        let output0 = c1
            .mul_real(p2)
            .neg()
            .add(c0.mul_real(p0).mul_i())
            .add(c0.mul_real(p3).mul_i())
            .add(c1.mul_real(p1).mul_i());
        let output1 = c1
            .mul_real(p3)
            .mul_i()
            .neg()
            .add(c0.mul_real(p2))
            .add(c0.mul_real(p1).mul_i())
            .add(c1.mul_real(p0).mul_i());
        unsafe {
            set_scaled_current(arena, row.component_base, stride, point, output0, scale);
            set_scaled_current(arena, row.component_base + 1, stride, point, output1, scale);
        }
    }

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectFinalizationRow,
        stride: usize,
        point: usize,
        row_factor: ComplexValue,
    ) {
        let c0 = unsafe { load_current_pair(arena, row.component_base, stride, point) };
        let c1 = unsafe { load_current_pair(arena, row.component_base + 1, stride, point) };
        let p0 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 0, stride, point) };
        let p1 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 1, stride, point) };
        let p2 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 2, stride, point) };
        let p3 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 3, stride, point) };
        let inv = reciprocal_minkowski_norm_pair(p0, p1, p2, p3);
        let scale = row_factor.mul_real_pair(inv);
        let output0 = c1
            .mul_real(p2)
            .neg()
            .add(c0.mul_real(p0).mul_i())
            .add(c0.mul_real(p3).mul_i())
            .add(c1.mul_real(p1).mul_i());
        let output1 = c1
            .mul_real(p3)
            .mul_i()
            .neg()
            .add(c0.mul_real(p2))
            .add(c0.mul_real(p1).mul_i())
            .add(c1.mul_real(p0).mul_i());
        unsafe {
            set_pre_scaled_current_pair(arena, row.component_base, stride, point, output0, scale);
            set_pre_scaled_current_pair(
                arena,
                row.component_base + 1,
                stride,
                point,
                output1,
                scale,
            );
        }
    }
}

impl DirectFinalizationFormula for FeynmanVectorPropagator {
    const COMPONENTS: u16 = 4;

    #[inline(always)]
    unsafe fn evaluate_point(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectFinalizationRow,
        stride: usize,
        point: usize,
        row_factor: ComplexValue,
    ) {
        let p0 = unsafe { load_momentum(momenta, row.momentum_form_id, 0, stride, point) };
        let p1 = unsafe { load_momentum(momenta, row.momentum_form_id, 1, stride, point) };
        let p2 = unsafe { load_momentum(momenta, row.momentum_form_id, 2, stride, point) };
        let p3 = unsafe { load_momentum(momenta, row.momentum_form_id, 3, stride, point) };
        let scale = row_factor
            .mul(ComplexValue::new(0.0, -1.0))
            .mul_real(reciprocal_minkowski_norm(p0, p1, p2, p3));
        for component in 0..4u32 {
            let value =
                unsafe { load_current(arena, row.component_base + component, stride, point) };
            unsafe {
                set_scaled_current(
                    arena,
                    row.component_base + component,
                    stride,
                    point,
                    value,
                    scale,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn evaluate_pair(
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        row: DirectFinalizationRow,
        stride: usize,
        point: usize,
        row_factor: ComplexValue,
    ) {
        let p0 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 0, stride, point) };
        let p1 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 1, stride, point) };
        let p2 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 2, stride, point) };
        let p3 = unsafe { load_momentum_pair(momenta, row.momentum_form_id, 3, stride, point) };
        let scale = row_factor
            .mul(ComplexValue::new(0.0, -1.0))
            .mul_real_pair(reciprocal_minkowski_norm_pair(p0, p1, p2, p3));
        for component in 0..4u32 {
            let value =
                unsafe { load_current_pair(arena, row.component_base + component, stride, point) };
            unsafe {
                set_pre_scaled_current_pair(
                    arena,
                    row.component_base + component,
                    stride,
                    point,
                    value,
                    scale,
                );
            }
        }
    }
}

unsafe extern "C" fn execute_weyl_propagator_positive_rows(
    _context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    unsafe {
        execute_finalization_rows::<WeylPropagatorPositive>(
            arena,
            momenta,
            factors,
            rows,
            row_count,
            point_count,
        )
    }
}

unsafe extern "C" fn execute_weyl_propagator_negative_rows(
    _context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    unsafe {
        execute_finalization_rows::<WeylPropagatorNegative>(
            arena,
            momenta,
            factors,
            rows,
            row_count,
            point_count,
        )
    }
}

unsafe extern "C" fn execute_feynman_vector_propagator_rows(
    _context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    unsafe {
        execute_finalization_rows::<FeynmanVectorPropagator>(
            arena,
            momenta,
            factors,
            rows,
            row_count,
            point_count,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn execute_rows<F: DirectContributionFormula>(
    context: *const c_void,
    arena: DirectArenaView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    if context.is_null() {
        return STATUS_INVALID_CONTEXT;
    }
    if row_count == 0
        || rows.is_null()
        || point_count == 0
        || arena.point_stride == 0
        || point_count > arena.point_stride
        || arena.current_re.is_null()
        || arena.current_im.is_null()
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let context = unsafe { &*context.cast::<RecurrenceIntrinsicScale>() };
    let context_scale = match effective_context_scale(*context, parameters) {
        Ok(scale) => scale.mul(F::BASE_SCALE),
        Err(status) => return status,
    };
    let stride = arena.point_stride as usize;
    let points = point_count as usize;

    for row_index in 0..row_count as usize {
        let row = unsafe { *rows.add(row_index) };
        if row.parent1_component_base_or_sentinel == DIRECT_NONE_U32 {
            return STATUS_INVALID_ARGUMENT;
        }
        if row.exact_factor_id >= factors.value_count
            || !component_range_in_bounds(
                row.parent0_component_base,
                F::PARENT0_COMPONENTS,
                arena.current_scalar_len,
                arena.point_stride,
                point_count,
            )
            || !component_range_in_bounds(
                row.parent1_component_base_or_sentinel,
                F::PARENT1_COMPONENTS,
                arena.current_scalar_len,
                arena.point_stride,
                point_count,
            )
            || !component_range_in_bounds(
                row.destination_component_base,
                F::DESTINATION_COMPONENTS,
                arena.current_scalar_len,
                arena.point_stride,
                point_count,
            )
        {
            return STATUS_BOUNDS;
        }
        let row_factor = ComplexValue::new(
            unsafe { *factors.values_re.add(row.exact_factor_id as usize) },
            unsafe { *factors.values_im.add(row.exact_factor_id as usize) },
        );
        let scale = context_scale.mul(row_factor);
        let mut point = 0usize;
        while point + 1 < points {
            unsafe { F::evaluate_pair(arena, row, stride, point, scale) };
            point += 2;
        }
        if point < points {
            unsafe { F::evaluate_point(arena, row, stride, point, scale) };
        }
    }
    DIRECT_STATUS_OK
}

unsafe fn execute_finalization_rows<F: DirectFinalizationFormula>(
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    if row_count == 0
        || rows.is_null()
        || point_count == 0
        || arena.point_stride == 0
        || point_count > arena.point_stride
        || arena.current_re.is_null()
        || arena.current_im.is_null()
        || momenta.values.is_null()
        || momenta.lorentz_component_count != 4
        || momenta.point_stride != arena.point_stride
        || point_count > momenta.point_stride
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return STATUS_INVALID_ARGUMENT;
    }
    let stride = arena.point_stride as usize;
    let points = point_count as usize;
    for row_index in 0..row_count as usize {
        let row = unsafe { *rows.add(row_index) };
        if row.component_count != F::COMPONENTS
            || row.exact_factor_id >= factors.value_count
            || !component_range_in_bounds(
                row.component_base,
                u32::from(F::COMPONENTS),
                arena.current_scalar_len,
                arena.point_stride,
                point_count,
            )
            || !momentum_form_in_bounds(momenta, row.momentum_form_id, point_count)
        {
            return STATUS_BOUNDS;
        }
        let row_factor = ComplexValue::new(
            unsafe { *factors.values_re.add(row.exact_factor_id as usize) },
            unsafe { *factors.values_im.add(row.exact_factor_id as usize) },
        );
        let mut point = 0usize;
        while point + 1 < points {
            unsafe { F::evaluate_pair(arena, momenta, row, stride, point, row_factor) };
            point += 2;
        }
        if point < points {
            unsafe { F::evaluate_point(arena, momenta, row, stride, point, row_factor) };
        }
    }
    DIRECT_STATUS_OK
}

fn effective_context_scale(
    context: RecurrenceIntrinsicScale,
    parameters: DirectParameterView,
) -> Result<ComplexValue, c_int> {
    let literal = ComplexValue::new(context.literal_re, context.literal_im);
    let Some(parameter_index) = context.model_parameter_index else {
        return Ok(literal);
    };
    if parameters.values_re.is_null()
        || parameters.values_im.is_null()
        || parameter_index >= parameters.value_count
    {
        return Err(STATUS_BOUNDS);
    }
    let parameter = ComplexValue::new(
        unsafe { *parameters.values_re.add(parameter_index as usize) },
        unsafe { *parameters.values_im.add(parameter_index as usize) },
    );
    Ok(literal.mul(parameter))
}

#[inline(always)]
fn reciprocal_minkowski_norm(p0: f64, p1: f64, p2: f64, p3: f64) -> f64 {
    1.0 / (p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3)
}

#[inline(always)]
fn reciprocal_minkowski_norm_pair(p0: f64x2, p1: f64x2, p2: f64x2, p3: f64x2) -> f64x2 {
    let one = f64x2::new([1.0, 1.0]);
    one / (p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3)
}

fn component_range_in_bounds(
    component_base: u32,
    component_count: u32,
    scalar_len: u64,
    point_stride: u32,
    point_count: u32,
) -> bool {
    let Some(last_component) = component_base.checked_add(component_count.saturating_sub(1)) else {
        return false;
    };
    u64::from(last_component)
        .checked_mul(u64::from(point_stride))
        .and_then(|base| base.checked_add(u64::from(point_count)))
        .is_some_and(|end| end <= scalar_len)
}

fn momentum_form_in_bounds(
    momenta: DirectMomentumView,
    momentum_form_id: u32,
    point_count: u32,
) -> bool {
    if momentum_form_id >= momenta.form_count || momenta.lorentz_component_count != 4 {
        return false;
    }
    u64::from(momentum_form_id)
        .checked_mul(u64::from(momenta.lorentz_component_count))
        .and_then(|plane| plane.checked_add(3))
        .and_then(|plane| plane.checked_mul(u64::from(momenta.point_stride)))
        .and_then(|base| base.checked_add(u64::from(point_count)))
        .is_some_and(|end| end <= momenta.scalar_len)
}

#[inline(always)]
fn minkowski_complex_dot(left: [ComplexValue; 4], right: [ComplexValue; 4]) -> ComplexValue {
    left[0]
        .mul(right[0])
        .sub(left[1].mul(right[1]))
        .sub(left[2].mul(right[2]))
        .sub(left[3].mul(right[3]))
}

#[inline(always)]
fn minkowski_complex_real_dot(left: [ComplexValue; 4], right: [f64; 4]) -> ComplexValue {
    left[0]
        .mul_real(right[0])
        .sub(left[1].mul_real(right[1]))
        .sub(left[2].mul_real(right[2]))
        .sub(left[3].mul_real(right[3]))
}

#[inline(always)]
fn minkowski_simd_complex_dot(left: [SimdComplex2; 4], right: [SimdComplex2; 4]) -> SimdComplex2 {
    left[0]
        .mul(right[0])
        .sub(left[1].mul(right[1]))
        .sub(left[2].mul(right[2]))
        .sub(left[3].mul(right[3]))
}

#[inline(always)]
fn minkowski_simd_complex_real_dot(left: [SimdComplex2; 4], right: [f64x2; 4]) -> SimdComplex2 {
    left[0]
        .mul_real(right[0])
        .sub(left[1].mul_real(right[1]))
        .sub(left[2].mul_real(right[2]))
        .sub(left[3].mul_real(right[3]))
}

#[inline(always)]
unsafe fn load_current(
    arena: DirectArenaView,
    component: u32,
    stride: usize,
    point: usize,
) -> ComplexValue {
    let offset = component as usize * stride + point;
    ComplexValue::new(unsafe { *arena.current_re.add(offset) }, unsafe {
        *arena.current_im.add(offset)
    })
}

#[inline(always)]
unsafe fn load_current_pair(
    arena: DirectArenaView,
    component: u32,
    stride: usize,
    point: usize,
) -> SimdComplex2 {
    let offset = component as usize * stride + point;
    SimdComplex2::new(
        f64x2::new([unsafe { *arena.current_re.add(offset) }, unsafe {
            *arena.current_re.add(offset + 1)
        }]),
        f64x2::new([unsafe { *arena.current_im.add(offset) }, unsafe {
            *arena.current_im.add(offset + 1)
        }]),
    )
}

#[inline(always)]
unsafe fn load_momentum(
    momenta: DirectMomentumView,
    momentum_form_id: u32,
    lorentz_component: u16,
    stride: usize,
    point: usize,
) -> f64 {
    let plane = momentum_form_id as usize * usize::from(momenta.lorentz_component_count)
        + usize::from(lorentz_component);
    unsafe { *momenta.values.add(plane * stride + point) }
}

#[inline(always)]
unsafe fn load_momentum_pair(
    momenta: DirectMomentumView,
    momentum_form_id: u32,
    lorentz_component: u16,
    stride: usize,
    point: usize,
) -> f64x2 {
    let plane = momentum_form_id as usize * usize::from(momenta.lorentz_component_count)
        + usize::from(lorentz_component);
    let offset = plane * stride + point;
    f64x2::new([unsafe { *momenta.values.add(offset) }, unsafe {
        *momenta.values.add(offset + 1)
    }])
}

#[inline(always)]
fn contribution_accumulates(row: DirectContributionRow) -> bool {
    row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION == 0
}

#[inline(always)]
unsafe fn add_scaled_current(
    arena: DirectArenaView,
    component: u32,
    stride: usize,
    point: usize,
    value: ComplexValue,
    scale: ComplexValue,
    accumulate: bool,
) {
    let offset = component as usize * stride + point;
    let contribution = value.mul(scale);
    unsafe {
        if accumulate {
            *arena.current_re.add(offset) += contribution.re;
            *arena.current_im.add(offset) += contribution.im;
        } else {
            *arena.current_re.add(offset) = contribution.re;
            *arena.current_im.add(offset) = contribution.im;
        }
    }
}

#[inline(always)]
unsafe fn add_scaled_current_pair(
    arena: DirectArenaView,
    component: u32,
    stride: usize,
    point: usize,
    value: SimdComplex2,
    scale: ComplexValue,
    accumulate: bool,
) {
    let scale_re = f64x2::new([scale.re, scale.re]);
    let scale_im = f64x2::new([scale.im, scale.im]);
    let contribution_re = value.re * scale_re - value.im * scale_im;
    let contribution_im = value.re * scale_im + value.im * scale_re;
    let re = unsafe { core::mem::transmute::<f64x2, [f64; 2]>(contribution_re) };
    let im = unsafe { core::mem::transmute::<f64x2, [f64; 2]>(contribution_im) };
    let offset = component as usize * stride + point;
    unsafe {
        if accumulate {
            *arena.current_re.add(offset) += re[0];
            *arena.current_re.add(offset + 1) += re[1];
            *arena.current_im.add(offset) += im[0];
            *arena.current_im.add(offset + 1) += im[1];
        } else {
            *arena.current_re.add(offset) = re[0];
            *arena.current_re.add(offset + 1) = re[1];
            *arena.current_im.add(offset) = im[0];
            *arena.current_im.add(offset + 1) = im[1];
        }
    }
}

#[inline(always)]
unsafe fn set_scaled_current(
    arena: DirectArenaView,
    component: u32,
    stride: usize,
    point: usize,
    value: ComplexValue,
    scale: ComplexValue,
) {
    let offset = component as usize * stride + point;
    let contribution = value.mul(scale);
    unsafe {
        *arena.current_re.add(offset) = contribution.re;
        *arena.current_im.add(offset) = contribution.im;
    }
}

#[inline(always)]
unsafe fn set_pre_scaled_current_pair(
    arena: DirectArenaView,
    component: u32,
    stride: usize,
    point: usize,
    value: SimdComplex2,
    scale: SimdComplex2,
) {
    let contribution = value.mul(scale);
    let re = unsafe { core::mem::transmute::<f64x2, [f64; 2]>(contribution.re) };
    let im = unsafe { core::mem::transmute::<f64x2, [f64; 2]>(contribution.im) };
    let offset = component as usize * stride + point;
    unsafe {
        *arena.current_re.add(offset) = re[0];
        *arena.current_re.add(offset + 1) = re[1];
        *arena.current_im.add(offset) = im[0];
        *arena.current_im.add(offset + 1) = im[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::DirectClosureRow;
    use crate::recurrence::direct_backend::{
        DirectContributionFanoutBundle, DirectContributionFanoutRun, DirectInteractionAnchor,
        DirectInteractionOutput, DirectInteractionTensorOutput, DirectInteractionTerminalClosure,
    };

    #[derive(Clone, Copy, Debug)]
    struct ReferenceComplex {
        re: f64,
        im: f64,
    }

    impl ReferenceComplex {
        fn new(re: f64, im: f64) -> Self {
            Self { re, im }
        }

        fn add(self, other: Self) -> Self {
            Self::new(self.re + other.re, self.im + other.im)
        }

        fn sub(self, other: Self) -> Self {
            Self::new(self.re - other.re, self.im - other.im)
        }

        fn neg(self) -> Self {
            Self::new(-self.re, -self.im)
        }

        fn mul(self, other: Self) -> Self {
            Self::new(
                self.re * other.re - self.im * other.im,
                self.re * other.im + self.im * other.re,
            )
        }

        fn mul_i(self) -> Self {
            Self::new(-self.im, self.re)
        }

        fn mul_real(self, value: f64) -> Self {
            Self::new(self.re * value, self.im * value)
        }
    }

    fn fill_component(re: &mut [f64], im: &mut [f64], stride: usize, component: usize, seed: f64) {
        for point in 0..stride {
            re[component * stride + point] = seed + point as f64 * 0.17;
            im[component * stride + point] = -0.3 * seed + point as f64 * 0.11;
        }
    }

    fn read_component(
        re: &[f64],
        im: &[f64],
        stride: usize,
        component: usize,
        point: usize,
    ) -> ReferenceComplex {
        ReferenceComplex::new(
            re[component * stride + point],
            im[component * stride + point],
        )
    }

    fn views(
        current_re: &mut [f64],
        current_im: &mut [f64],
        stride: u32,
        parameters_re: &[f64],
        parameters_im: &[f64],
        factors_re: &[f64],
        factors_im: &[f64],
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
                amplitude_re: ptr::null_mut(),
                amplitude_im: ptr::null_mut(),
                amplitude_scalar_len: 0,
                point_stride: stride,
            },
            DirectMomentumView {
                values: ptr::null(),
                scalar_len: 0,
                form_count: 0,
                lorentz_component_count: 4,
                point_stride: stride,
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

    fn effective_test_scale(
        literal: ReferenceComplex,
        parameter: ReferenceComplex,
        factor: ReferenceComplex,
    ) -> ReferenceComplex {
        literal.mul(parameter).mul(factor)
    }

    fn assert_close(actual: ReferenceComplex, expected: ReferenceComplex) {
        assert!(
            (actual.re - expected.re).abs() < 2.0e-13,
            "real mismatch: actual {}, expected {}",
            actual.re,
            expected.re
        );
        assert!(
            (actual.im - expected.im).abs() < 2.0e-13,
            "imaginary mismatch: actual {}, expected {}",
            actual.im,
            expected.im
        );
    }

    #[test]
    fn runtime_template_names_round_trip_and_unknown_names_fail_closed() {
        let kinds = [
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylPositive,
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylNegative,
            RecurrenceContributionIntrinsicKind::AntisymmetricTensorVectorToVector,
            RecurrenceContributionIntrinsicKind::VectorWedgeVectorToAntisymmetricTensor,
            RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector,
        ];
        for kind in kinds {
            assert_eq!(
                RecurrenceContributionIntrinsicKind::from_runtime_template(kind.runtime_template())
                    .unwrap(),
                kind
            );
        }
        assert!(
            RecurrenceContributionIntrinsicKind::from_runtime_template(
                "rusticol.recurrence.contribution.unknown.v1"
            )
            .is_err()
        );
    }

    #[test]
    fn fused_fanout_matches_supported_intrinsic_rows() {
        for kind in [
            RecurrenceContributionIntrinsicKind::AntisymmetricTensorVectorToVector,
            RecurrenceContributionIntrinsicKind::VectorWedgeVectorToAntisymmetricTensor,
            RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector,
        ] {
            let initial_re = (0..22)
                .map(|component| component as f64 * 0.37 - 1.2)
                .collect::<Vec<_>>();
            let initial_im = initial_re
                .iter()
                .map(|value| -0.3 * value)
                .collect::<Vec<_>>();
            let mut direct_re = initial_re.clone();
            let mut direct_im = initial_im.clone();
            let mut fused_re = initial_re;
            let mut fused_im = initial_im;
            let factors_re = [0.7, -0.3];
            let factors_im = [0.2, 0.8];
            let parameters_re = [-0.4];
            let parameters_im = [0.6];
            let momenta_values = [1.1, 0.2, -0.4, 0.7, 0.6, -0.8, 0.3, -0.5];
            let first = DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: 6,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 1,
                destination_component_base: 10,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            };
            let rows = [
                first,
                DirectContributionRow {
                    destination_component_base: 16,
                    exact_factor_id: 1,
                    flags: 0,
                    ..first
                },
            ];
            let executor = LoadedRecurrenceIntrinsicDirectExecutor::load(
                kind,
                RecurrenceIntrinsicScale::new(0.75, -0.2, Some(0)).unwrap(),
            );
            let execute_views = |re: &mut [f64], im: &mut [f64]| {
                let (arena, _, parameters, factors) = views(
                    re,
                    im,
                    1,
                    &parameters_re,
                    &parameters_im,
                    &factors_re,
                    &factors_im,
                );
                let momenta = DirectMomentumView {
                    values: momenta_values.as_ptr(),
                    scalar_len: 8,
                    form_count: 2,
                    lorentz_component_count: 4,
                    point_stride: 1,
                };
                (arena, momenta, parameters, factors)
            };
            let (arena, momenta, parameters, factors) =
                execute_views(&mut direct_re, &mut direct_im);
            let handle = executor.contribution_handle();
            assert_eq!(
                unsafe {
                    (handle.call)(
                        handle.context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        2,
                        1,
                    )
                },
                DIRECT_STATUS_OK
            );

            let (arena, momenta, parameters, factors) = execute_views(&mut fused_re, &mut fused_im);
            let fused = executor.contribution_fanout_handle().unwrap();
            let run = DirectContributionFanoutRun {
                row_start: 0,
                row_count: 2,
                scratch_component_base: 0,
                bundle_start: 0,
                bundle_count: 2,
            };
            let bundles = [
                DirectContributionFanoutBundle {
                    row_start: 0,
                    row_count: 1,
                    effective_factor_id_or_sentinel: 0,
                },
                DirectContributionFanoutBundle {
                    row_start: 1,
                    row_count: 1,
                    effective_factor_id_or_sentinel: 1,
                },
            ];
            unsafe {
                (fused.call)(
                    fused.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    DirectContributionFanoutBatch {
                        rows: &rows,
                        runs: &[run],
                        bundles: &bundles,
                    },
                )
            }
            .unwrap();
            if kind == RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector {
                assert!(
                    unsafe {
                        (fused.call)(
                            fused.context,
                            arena,
                            DirectMomentumView {
                                lorentz_component_count: 3,
                                ..momenta
                            },
                            parameters,
                            factors,
                            DirectContributionFanoutBatch {
                                rows: &rows,
                                runs: &[run],
                                bundles: &bundles,
                            },
                        )
                    }
                    .is_err()
                );
            }
            for (actual, expected) in fused_re
                .iter()
                .zip(&direct_re)
                .chain(fused_im.iter().zip(&direct_im))
            {
                assert!((actual - expected).abs() < 2.0e-13);
            }
        }
    }

    #[test]
    fn identity_finalization_is_backend_neutral_and_bounds_checked() {
        let stride = 3usize;
        let point_count = 2usize;
        let mut current_re = vec![1.0, 2.0, 99.0, 3.0, 4.0, 88.0];
        let mut current_im = vec![5.0, 6.0, 77.0, 7.0, 8.0, 66.0];
        let factors_re = [2.0];
        let factors_im = [-1.0];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            stride as u32,
            &[],
            &[],
            &factors_re,
            &factors_im,
        );
        let row = DirectFinalizationRow {
            component_base: 0,
            component_count: 2,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };

        assert_eq!(
            unsafe {
                execute_identity_finalization_rows(
                    ptr::null(),
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&row),
                    1,
                    point_count as u32,
                )
            },
            DIRECT_STATUS_OK
        );
        assert_eq!(current_re, vec![7.0, 10.0, 99.0, 13.0, 16.0, 88.0]);
        assert_eq!(current_im, vec![9.0, 10.0, 77.0, 11.0, 12.0, 66.0]);

        let out_of_bounds = DirectFinalizationRow {
            component_base: 1,
            component_count: 2,
            ..row
        };
        assert_eq!(
            unsafe {
                execute_identity_finalization_rows(
                    ptr::null(),
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&out_of_bounds),
                    1,
                    point_count as u32,
                )
            },
            STATUS_BOUNDS
        );
    }

    #[test]
    fn weyl_vector_chirality_variants_match_reference_math_and_add() {
        for kind in [
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylPositive,
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylNegative,
        ] {
            let stride = 5usize;
            let mut current_re = vec![0.0; 8 * stride];
            let mut current_im = vec![0.0; 8 * stride];
            for component in 0..6 {
                fill_component(
                    &mut current_re,
                    &mut current_im,
                    stride,
                    component,
                    0.4 + component as f64,
                );
            }
            fill_component(&mut current_re, &mut current_im, stride, 6, 7.0);
            fill_component(&mut current_re, &mut current_im, stride, 7, -3.0);
            let before_re = current_re.clone();
            let before_im = current_im.clone();
            let parameters_re = [-0.4];
            let parameters_im = [0.3];
            let factors_re = [1.25];
            let factors_im = [-0.5];
            let (arena, momenta, parameters, factors) = views(
                &mut current_re,
                &mut current_im,
                stride as u32,
                &parameters_re,
                &parameters_im,
                &factors_re,
                &factors_im,
            );
            let intrinsic_literal = ReferenceComplex::new(0.75, -0.2)
                .mul(ReferenceComplex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0));
            let scale =
                RecurrenceIntrinsicScale::new(intrinsic_literal.re, intrinsic_literal.im, Some(0))
                    .unwrap();
            let executor = LoadedRecurrenceIntrinsicDirectExecutor::load(kind, scale);
            let handle = executor.contribution_handle();
            let row = DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: 2,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 0,
                destination_component_base: 6,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: 0,
            };
            let status = unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&row),
                    1,
                    stride as u32,
                )
            };
            assert_eq!(status, DIRECT_STATUS_OK);

            let total_scale = effective_test_scale(
                intrinsic_literal,
                ReferenceComplex::new(parameters_re[0], parameters_im[0]),
                ReferenceComplex::new(factors_re[0], factors_im[0]),
            );
            for point in 0..stride {
                let l0 = read_component(&before_re, &before_im, stride, 0, point);
                let l1 = read_component(&before_re, &before_im, stride, 1, point);
                let r0 = read_component(&before_re, &before_im, stride, 2, point);
                let r1 = read_component(&before_re, &before_im, stride, 3, point);
                let r2 = read_component(&before_re, &before_im, stride, 4, point);
                let r3 = read_component(&before_re, &before_im, stride, 5, point);
                let output = match kind {
                    RecurrenceContributionIntrinsicKind::WeylVectorToWeylPositive => [
                        l0.mul(r3)
                            .mul_i()
                            .neg()
                            .sub(l1.mul(r1).mul_i())
                            .add(l1.mul(r2))
                            .add(l0.mul(r0).mul_i()),
                        l0.mul(r2)
                            .neg()
                            .sub(l0.mul(r1).mul_i())
                            .add(l1.mul(r0).mul_i())
                            .add(l1.mul(r3).mul_i()),
                    ],
                    RecurrenceContributionIntrinsicKind::WeylVectorToWeylNegative => [
                        l1.mul(r2)
                            .neg()
                            .add(l0.mul(r0).mul_i())
                            .add(l0.mul(r3).mul_i())
                            .add(l1.mul(r1).mul_i()),
                        l1.mul(r3)
                            .mul_i()
                            .neg()
                            .add(l0.mul(r2))
                            .add(l0.mul(r1).mul_i())
                            .add(l1.mul(r0).mul_i()),
                    ],
                    _ => unreachable!(),
                };
                for (component, value) in output.into_iter().enumerate() {
                    let initial =
                        read_component(&before_re, &before_im, stride, 6 + component, point);
                    let expected = initial.add(value.mul(total_scale));
                    let actual =
                        read_component(&current_re, &current_im, stride, 6 + component, point);
                    assert_close(actual, expected);
                }
            }
        }
    }

    #[test]
    fn initializing_intrinsic_contribution_overwrites_poisoned_complex_multipoint_destination() {
        let stride = 5usize;
        let mut current_re = vec![0.0; 8 * stride];
        let mut current_im = vec![0.0; 8 * stride];
        for component in 0..6 {
            fill_component(
                &mut current_re,
                &mut current_im,
                stride,
                component,
                0.4 + component as f64,
            );
        }
        let factors_re = [1.25];
        let factors_im = [-0.5];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            stride as u32,
            &[],
            &[],
            &factors_re,
            &factors_im,
        );
        let executor = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylPositive,
            RecurrenceIntrinsicScale::new(0.75, -0.2, None).unwrap(),
        );
        let handle = executor.contribution_handle();
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 2,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 6,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&row),
                    1,
                    stride as u32,
                )
            },
            DIRECT_STATUS_OK
        );
        let expected_re = current_re[6 * stride..].to_vec();
        let expected_im = current_im[6 * stride..].to_vec();
        current_re[6 * stride..].fill(f64::NAN);
        current_im[6 * stride..].fill(f64::NAN);
        let initializing = DirectContributionRow {
            flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            ..row
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&initializing),
                    1,
                    stride as u32,
                )
            },
            DIRECT_STATUS_OK
        );
        assert_eq!(&current_re[6 * stride..], expected_re.as_slice());
        assert_eq!(&current_im[6 * stride..], expected_im.as_slice());
        assert!(expected_im.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn tensor_vector_intrinsic_matches_reference_math() {
        let stride = 4usize;
        let mut current_re = vec![0.0; 14 * stride];
        let mut current_im = vec![0.0; 14 * stride];
        for component in 0..14 {
            fill_component(
                &mut current_re,
                &mut current_im,
                stride,
                component,
                0.2 + component as f64 * 0.4,
            );
        }
        let before_re = current_re.clone();
        let before_im = current_im.clone();
        let factors_re = [0.8];
        let factors_im = [0.1];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            stride as u32,
            &[],
            &[],
            &factors_re,
            &factors_im,
        );
        let executor = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::AntisymmetricTensorVectorToVector,
            RecurrenceIntrinsicScale::new(0.0, 0.5, None).unwrap(),
        );
        let handle = executor.contribution_handle();
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 6,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 10,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&row),
                    1,
                    stride as u32,
                )
            },
            DIRECT_STATUS_OK
        );

        let scale = ReferenceComplex::new(0.0, 0.5)
            .mul(ReferenceComplex::new(factors_re[0], factors_im[0]));
        for point in 0..stride {
            let left = std::array::from_fn::<_, 6, _>(|component| {
                read_component(&before_re, &before_im, stride, component, point)
            });
            let right = std::array::from_fn::<_, 4, _>(|component| {
                read_component(&before_re, &before_im, stride, 6 + component, point)
            });
            let output = [
                left[0]
                    .mul(right[1])
                    .add(left[1].mul(right[2]))
                    .add(left[2].mul(right[3])),
                left[0]
                    .mul(right[0])
                    .add(left[3].mul(right[2]))
                    .add(left[4].mul(right[3])),
                left[1]
                    .mul(right[0])
                    .sub(left[3].mul(right[1]))
                    .add(left[5].mul(right[3])),
                left[2]
                    .mul(right[0])
                    .sub(left[4].mul(right[1]))
                    .sub(left[5].mul(right[2])),
            ];
            for (component, value) in output.into_iter().enumerate() {
                let initial = read_component(&before_re, &before_im, stride, 10 + component, point);
                let actual =
                    read_component(&current_re, &current_im, stride, 10 + component, point);
                assert_close(actual, initial.add(value.mul(scale)));
            }
        }
    }

    #[test]
    fn vector_wedge_intrinsic_matches_reference_math() {
        let stride = 3usize;
        let mut current_re = vec![0.0; 14 * stride];
        let mut current_im = vec![0.0; 14 * stride];
        for component in 0..14 {
            fill_component(
                &mut current_re,
                &mut current_im,
                stride,
                component,
                -0.7 + component as f64 * 0.6,
            );
        }
        let before_re = current_re.clone();
        let before_im = current_im.clone();
        let factors_re = [1.1];
        let factors_im = [-0.35];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            stride as u32,
            &[],
            &[],
            &factors_re,
            &factors_im,
        );
        let executor = LoadedRecurrenceIntrinsicDirectExecutor::load_runtime_template(
            VECTOR_WEDGE_VECTOR_TO_ANTISYMMETRIC_TENSOR_TEMPLATE,
            RecurrenceIntrinsicScale::unit(),
        )
        .unwrap();
        let handle = executor.contribution_handle();
        assert!(
            executor
                .contribution_fanout_handle()
                .unwrap()
                .interaction_call
                .is_none(),
            "wedge remains on its ordinary/fanout dispatch path"
        );
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 4,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 8,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&row),
                    1,
                    stride as u32,
                )
            },
            DIRECT_STATUS_OK
        );

        let scale = ReferenceComplex::new(factors_re[0], factors_im[0]);
        for point in 0..stride {
            let left = std::array::from_fn::<_, 4, _>(|component| {
                read_component(&before_re, &before_im, stride, component, point)
            });
            let right = std::array::from_fn::<_, 4, _>(|component| {
                read_component(&before_re, &before_im, stride, 4 + component, point)
            });
            let output = [
                left[0].mul(right[1]).sub(left[1].mul(right[0])),
                left[0].mul(right[2]).sub(left[2].mul(right[0])),
                left[0].mul(right[3]).sub(left[3].mul(right[0])),
                left[1].mul(right[2]).sub(left[2].mul(right[1])),
                left[1].mul(right[3]).sub(left[3].mul(right[1])),
                left[2].mul(right[3]).sub(left[3].mul(right[2])),
            ];
            for (component, value) in output.into_iter().enumerate() {
                let initial = read_component(&before_re, &before_im, stride, 8 + component, point);
                let actual = read_component(&current_re, &current_im, stride, 8 + component, point);
                assert_close(actual, initial.add(value.mul(scale)));
            }
        }
    }

    #[test]
    fn interaction_bundle_matches_rows_across_order_sides_and_live_mutation() {
        let component_count = 56_usize;
        let mut seed_re = vec![0.0; component_count];
        let mut seed_im = vec![0.0; component_count];
        for component in 0..component_count {
            seed_re[component] = 0.13 + component as f64 * 0.071;
            seed_im[component] = -0.29 + component as f64 * 0.043;
        }
        for component in 40..48 {
            seed_re[component] = 91.0;
            seed_im[component] = -37.0;
        }
        let momenta_values = (0..16)
            .map(|index| -0.8 + index as f64 * 0.19)
            .collect::<Vec<_>>();
        let rows = [
            DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: 4,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 1,
                destination_component_base: 40,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            },
            DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: 4,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 1,
                destination_component_base: 44,
                exact_factor_id: 1,
                selector_domain_id: 0,
                flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            },
            DirectContributionRow {
                parent0_component_base: 8,
                parent1_component_base_or_sentinel: 12,
                parent0_momentum_form_id: 2,
                parent1_momentum_form_id_or_sentinel: 3,
                destination_component_base: 40,
                exact_factor_id: 2,
                selector_domain_id: 0,
                flags: 0,
            },
            DirectContributionRow {
                parent0_component_base: 16,
                parent1_component_base_or_sentinel: 0,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 0,
                destination_component_base: 44,
                exact_factor_id: 3,
                selector_domain_id: 0,
                flags: 0,
            },
            DirectContributionRow {
                parent0_component_base: 16,
                parent1_component_base_or_sentinel: 0,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 0,
                destination_component_base: 40,
                exact_factor_id: 4,
                selector_domain_id: 0,
                flags: 0,
            },
            DirectContributionRow {
                parent0_component_base: 24,
                parent1_component_base_or_sentinel: 12,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 3,
                destination_component_base: 40,
                exact_factor_id: 5,
                selector_domain_id: 0,
                flags: 0,
            },
        ];
        let anchors = [
            DirectInteractionAnchor {
                parent_component_bases: [0, 4],
                parent_momentum_form_ids: [0, 1],
                operands: DirectInteractionOperands::One(DirectInteractionOperand {
                    tensor_component_base: 16,
                    vector_parent: DirectInteractionVectorParent::Parent0,
                }),
                output_start: 0,
                output_end: 2,
                tensor_output_start: 0,
                tensor_output_end: 0,
            },
            DirectInteractionAnchor {
                parent_component_bases: [8, 12],
                parent_momentum_form_ids: [2, 3],
                operands: DirectInteractionOperands::One(DirectInteractionOperand {
                    tensor_component_base: 24,
                    vector_parent: DirectInteractionVectorParent::Parent1,
                }),
                output_start: 2,
                output_end: 3,
                tensor_output_start: 0,
                tensor_output_end: 0,
            },
        ];
        let outputs = [
            DirectInteractionOutput {
                destination_component_base: 40,
                exact_factor_ids: [0, 4, 0],
                initialize: true,
            },
            DirectInteractionOutput {
                destination_component_base: 44,
                exact_factor_ids: [1, 3, 0],
                initialize: true,
            },
            DirectInteractionOutput {
                destination_component_base: 40,
                exact_factor_ids: [2, 5, 0],
                initialize: false,
            },
        ];
        let color = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector,
            RecurrenceIntrinsicScale::new(0.73, -0.21, Some(0)).unwrap(),
        );
        let atv = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::AntisymmetricTensorVectorToVector,
            RecurrenceIntrinsicScale::new(-0.37, 0.42, Some(0)).unwrap(),
        );
        let color_handle = color.contribution_handle();
        let atv_handle = atv.contribution_handle();
        let contexts = DirectInteractionExecutorContexts {
            color: color_handle.context,
            vector_wedge_vector: ptr::null(),
            antisymmetric_tensor_vector: atv_handle.context,
        };

        for atv_before_color in [false, true] {
            for (parameters_re, parameters_im, factors_re, factors_im) in [
                (
                    vec![0.61],
                    vec![-0.17],
                    vec![1.2, -0.8, 0.55, -1.4, 0.72, -0.31],
                    vec![0.1, 0.35, -0.22, 0.19, -0.44, 0.27],
                ),
                (
                    vec![-0.43],
                    vec![0.29],
                    vec![-0.7, 1.6, -1.1, 0.48, -0.36, 1.23],
                    vec![0.31, -0.27, 0.08, -0.52, 0.41, -0.14],
                ),
            ] {
                let mut expected_re = seed_re.clone();
                let mut expected_im = seed_im.clone();
                let mut candidate_re = seed_re.clone();
                let mut candidate_im = seed_im.clone();
                let parameter_view = DirectParameterView {
                    values_re: parameters_re.as_ptr(),
                    values_im: parameters_im.as_ptr(),
                    value_count: 1,
                };
                let factor_view = DirectFactorView {
                    values_re: factors_re.as_ptr(),
                    values_im: factors_im.as_ptr(),
                    value_count: factors_re.len() as u32,
                };
                let momentum_view = DirectMomentumView {
                    values: momenta_values.as_ptr(),
                    scalar_len: momenta_values.len() as u64,
                    form_count: 4,
                    lorentz_component_count: 4,
                    point_stride: 1,
                };
                let arena = |re: &mut [f64], im: &mut [f64]| DirectArenaView {
                    current_re: re.as_mut_ptr(),
                    current_im: im.as_mut_ptr(),
                    current_scalar_len: re.len() as u64,
                    amplitude_re: ptr::null_mut(),
                    amplitude_im: ptr::null_mut(),
                    amplitude_scalar_len: 0,
                    point_stride: 1,
                };
                let mut oracle_rows = rows;
                if atv_before_color {
                    oracle_rows[0].flags = 0;
                    oracle_rows[1].flags = 0;
                    oracle_rows[3].flags = DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
                    oracle_rows[4].flags = DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
                    assert_eq!(
                        unsafe {
                            (atv_handle.call)(
                                atv_handle.context,
                                arena(&mut expected_re, &mut expected_im),
                                momentum_view,
                                parameter_view,
                                factor_view,
                                oracle_rows[3..].as_ptr(),
                                3,
                                1,
                            )
                        },
                        DIRECT_STATUS_OK
                    );
                    assert_eq!(
                        unsafe {
                            (color_handle.call)(
                                color_handle.context,
                                arena(&mut expected_re, &mut expected_im),
                                momentum_view,
                                parameter_view,
                                factor_view,
                                oracle_rows[..3].as_ptr(),
                                3,
                                1,
                            )
                        },
                        DIRECT_STATUS_OK
                    );
                } else {
                    assert_eq!(
                        unsafe {
                            (color_handle.call)(
                                color_handle.context,
                                arena(&mut expected_re, &mut expected_im),
                                momentum_view,
                                parameter_view,
                                factor_view,
                                oracle_rows[..3].as_ptr(),
                                3,
                                1,
                            )
                        },
                        DIRECT_STATUS_OK
                    );
                    assert_eq!(
                        unsafe {
                            (atv_handle.call)(
                                atv_handle.context,
                                arena(&mut expected_re, &mut expected_im),
                                momentum_view,
                                parameter_view,
                                factor_view,
                                oracle_rows[3..].as_ptr(),
                                3,
                                1,
                            )
                        },
                        DIRECT_STATUS_OK
                    );
                }
                let candidate_pointer = candidate_re.as_ptr();
                unsafe {
                    execute_interaction_bundle(
                        contexts,
                        arena(&mut candidate_re, &mut candidate_im),
                        momentum_view,
                        parameter_view,
                        factor_view,
                        DirectInteractionBundleBatch {
                            anchors: &anchors,
                            outputs: &outputs,
                            tensor_outputs: &[],
                            terminal: None,
                            requirements: DirectInteractionRuntimeRequirements {
                                current_component_count: 48,
                                momentum_form_count: 4,
                                parameter_count: 1,
                                factor_count: 6,
                            },
                            atv_before_color,
                        },
                    )
                }
                .unwrap();
                assert_eq!(candidate_re.as_ptr(), candidate_pointer);
                for component in 40..48 {
                    let error = (candidate_re[component] - expected_re[component])
                        .hypot(candidate_im[component] - expected_im[component]);
                    let scale = candidate_re[component]
                        .hypot(candidate_im[component])
                        .max(expected_re[component].hypot(expected_im[component]))
                        .max(1.0);
                    assert!(error <= 2.0e-12 * scale, "component {component}: {error:e}");
                }
            }
        }

        let mut rejected_re = seed_re.clone();
        let mut rejected_im = seed_im.clone();
        let before_re = rejected_re.clone();
        let before_im = rejected_im.clone();
        let parameters_re = [0.61];
        let parameters_im = [-0.17];
        let short_factors_re = [1.2, -0.8, 0.55, -1.4, 0.72];
        let short_factors_im = [0.1, 0.35, -0.22, 0.19, -0.44];
        let error = unsafe {
            execute_interaction_bundle(
                contexts,
                DirectArenaView {
                    current_re: rejected_re.as_mut_ptr(),
                    current_im: rejected_im.as_mut_ptr(),
                    current_scalar_len: rejected_re.len() as u64,
                    amplitude_re: ptr::null_mut(),
                    amplitude_im: ptr::null_mut(),
                    amplitude_scalar_len: 0,
                    point_stride: 1,
                },
                DirectMomentumView {
                    values: momenta_values.as_ptr(),
                    scalar_len: momenta_values.len() as u64,
                    form_count: 4,
                    lorentz_component_count: 4,
                    point_stride: 1,
                },
                DirectParameterView {
                    values_re: parameters_re.as_ptr(),
                    values_im: parameters_im.as_ptr(),
                    value_count: 1,
                },
                DirectFactorView {
                    values_re: short_factors_re.as_ptr(),
                    values_im: short_factors_im.as_ptr(),
                    value_count: short_factors_re.len() as u32,
                },
                DirectInteractionBundleBatch {
                    anchors: &anchors,
                    outputs: &outputs,
                    tensor_outputs: &[],
                    terminal: None,
                    requirements: DirectInteractionRuntimeRequirements {
                        current_component_count: 48,
                        momentum_form_count: 4,
                        parameter_count: 1,
                        factor_count: 6,
                    },
                    atv_before_color: false,
                },
            )
        }
        .unwrap_err();
        assert!(error.to_string().contains("factor catalog"));
        assert_eq!(rejected_re, before_re);
        assert_eq!(rejected_im, before_im);
    }

    #[test]
    fn interaction_bundle_fuses_sibling_wedge_with_live_complex_scales() {
        let mut seed_re = vec![0.0; 48];
        let mut seed_im = vec![0.0; 48];
        for component in 0..48 {
            seed_re[component] = -0.8 + component as f64 * 0.071;
            seed_im[component] = 0.3 - component as f64 * 0.043;
        }
        let momenta_values = [0.7, -1.1, 0.4, 1.6, -0.2, 0.9, -1.3, 0.5];
        let parameters_re = [0.61];
        let parameters_im = [-0.29];
        let factors_re = [1.2, -0.73, 0.44, -1.31];
        let factors_im = [-0.17, 0.36, 0.28, -0.09];
        let color_rows = [
            DirectContributionRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: 4,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 1,
                destination_component_base: 20,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            },
            DirectContributionRow {
                exact_factor_id: 1,
                flags: 0,
                ..DirectContributionRow {
                    parent0_component_base: 0,
                    parent1_component_base_or_sentinel: 4,
                    parent0_momentum_form_id: 0,
                    parent1_momentum_form_id_or_sentinel: 1,
                    destination_component_base: 20,
                    exact_factor_id: 0,
                    selector_domain_id: 0,
                    flags: 0,
                }
            },
        ];
        let atv_rows = [
            DirectContributionRow {
                parent0_component_base: 8,
                parent1_component_base_or_sentinel: 0,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 0,
                destination_component_base: 20,
                exact_factor_id: 2,
                selector_domain_id: 0,
                flags: 0,
            },
            DirectContributionRow {
                exact_factor_id: 3,
                ..DirectContributionRow {
                    parent0_component_base: 8,
                    parent1_component_base_or_sentinel: 0,
                    parent0_momentum_form_id: 0,
                    parent1_momentum_form_id_or_sentinel: 0,
                    destination_component_base: 20,
                    exact_factor_id: 2,
                    selector_domain_id: 0,
                    flags: 0,
                }
            },
        ];
        let wedge_rows = [
            DirectContributionRow {
                destination_component_base: 40,
                exact_factor_id: 0,
                flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
                ..color_rows[0]
            },
            DirectContributionRow {
                destination_component_base: 40,
                exact_factor_id: 1,
                flags: 0,
                ..color_rows[1]
            },
        ];
        let color = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector,
            RecurrenceIntrinsicScale::new(0.73, -0.21, Some(0)).unwrap(),
        );
        let atv = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::AntisymmetricTensorVectorToVector,
            RecurrenceIntrinsicScale::new(-0.37, 0.42, Some(0)).unwrap(),
        );
        let wedge = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::VectorWedgeVectorToAntisymmetricTensor,
            RecurrenceIntrinsicScale::new(0.28, 0.63, Some(0)).unwrap(),
        );
        let color_handle = color.contribution_handle();
        let atv_handle = atv.contribution_handle();
        let wedge_handle = wedge.contribution_handle();
        let momentum_view = DirectMomentumView {
            values: momenta_values.as_ptr(),
            scalar_len: momenta_values.len() as u64,
            form_count: 2,
            lorentz_component_count: 4,
            point_stride: 1,
        };
        let parameter_view = DirectParameterView {
            values_re: parameters_re.as_ptr(),
            values_im: parameters_im.as_ptr(),
            value_count: 1,
        };
        let factor_view = DirectFactorView {
            values_re: factors_re.as_ptr(),
            values_im: factors_im.as_ptr(),
            value_count: factors_re.len() as u32,
        };
        let arena = |re: &mut [f64], im: &mut [f64]| DirectArenaView {
            current_re: re.as_mut_ptr(),
            current_im: im.as_mut_ptr(),
            current_scalar_len: re.len() as u64,
            amplitude_re: ptr::null_mut(),
            amplitude_im: ptr::null_mut(),
            amplitude_scalar_len: 0,
            point_stride: 1,
        };
        let mut expected_re = seed_re.clone();
        let mut expected_im = seed_im.clone();
        for (handle, rows) in [
            (color_handle, color_rows.as_slice()),
            (atv_handle, atv_rows.as_slice()),
            (wedge_handle, wedge_rows.as_slice()),
        ] {
            assert_eq!(
                unsafe {
                    (handle.call)(
                        handle.context,
                        arena(&mut expected_re, &mut expected_im),
                        momentum_view,
                        parameter_view,
                        factor_view,
                        rows.as_ptr(),
                        rows.len() as u32,
                        1,
                    )
                },
                DIRECT_STATUS_OK
            );
        }

        let anchors = [DirectInteractionAnchor {
            parent_component_bases: [0, 4],
            parent_momentum_form_ids: [0, 1],
            operands: DirectInteractionOperands::One(DirectInteractionOperand {
                tensor_component_base: 8,
                vector_parent: DirectInteractionVectorParent::Parent0,
            }),
            output_start: 0,
            output_end: 2,
            tensor_output_start: 0,
            tensor_output_end: 2,
        }];
        let outputs = [
            DirectInteractionOutput {
                destination_component_base: 20,
                exact_factor_ids: [0, 2, 0],
                initialize: true,
            },
            DirectInteractionOutput {
                destination_component_base: 20,
                exact_factor_ids: [1, 3, 0],
                initialize: false,
            },
        ];
        let tensor_outputs = [
            DirectInteractionTensorOutput {
                destination_component_base: 40,
                exact_factor_id: 0,
                initialize: true,
            },
            DirectInteractionTensorOutput {
                destination_component_base: 40,
                exact_factor_id: 1,
                initialize: false,
            },
        ];
        let mut actual_re = seed_re.clone();
        let mut actual_im = seed_im.clone();
        unsafe {
            execute_interaction_bundle(
                DirectInteractionExecutorContexts {
                    color: color_handle.context,
                    vector_wedge_vector: wedge_handle.context,
                    antisymmetric_tensor_vector: atv_handle.context,
                },
                arena(&mut actual_re, &mut actual_im),
                momentum_view,
                parameter_view,
                factor_view,
                DirectInteractionBundleBatch {
                    anchors: &anchors,
                    outputs: &outputs,
                    tensor_outputs: &tensor_outputs,
                    terminal: None,
                    requirements: DirectInteractionRuntimeRequirements {
                        current_component_count: 46,
                        momentum_form_count: 2,
                        parameter_count: 1,
                        factor_count: 4,
                    },
                    atv_before_color: false,
                },
            )
        }
        .unwrap();
        for component in (20..24).chain(40..46) {
            let error = (actual_re[component] - expected_re[component])
                .hypot(actual_im[component] - expected_im[component]);
            let scale = actual_re[component]
                .hypot(actual_im[component])
                .max(expected_re[component].hypot(expected_im[component]))
                .max(1.0);
            assert!(error <= 2.0e-12 * scale, "component {component}: {error:e}");
        }
    }

    #[test]
    fn terminal_interaction_scalarization_matches_materialized_closure_replay() {
        let mut seed_re = vec![0.0; 40];
        let mut seed_im = vec![0.0; 40];
        for component in 0..40 {
            seed_re[component] = 0.17 - component as f64 * 0.039;
            seed_im[component] = -0.41 + component as f64 * 0.052;
        }
        let momenta_values = [0.8, -0.5, 1.2, -1.4, -0.3, 1.1, 0.6, -0.9];
        let parameters_re = [0.57];
        let parameters_im = [0.23];
        let factors_re = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let factors_im = [0.0; 10];
        let closure_rows = [
            DirectClosureRow {
                parent0_component_base: 20,
                parent1_component_base_or_sentinel: 30,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: 0,
                amplitude_destination_id: 0,
                exact_factor_id: 8,
                component_factor_start: 4,
                component_count: 4,
                selector_domain_id: 0,
                flags: 0,
            },
            DirectClosureRow {
                amplitude_destination_id: 1,
                exact_factor_id: 9,
                ..DirectClosureRow {
                    parent0_component_base: 20,
                    parent1_component_base_or_sentinel: 30,
                    parent0_momentum_form_id: 0,
                    parent1_momentum_form_id_or_sentinel: 0,
                    amplitude_destination_id: 0,
                    exact_factor_id: 8,
                    component_factor_start: 4,
                    component_count: 4,
                    selector_domain_id: 0,
                    flags: 0,
                }
            },
        ];
        let terminal_closures = [
            DirectInteractionTerminalClosure {
                root_component_base: 20,
                amplitude_destination_id: 0,
                exact_factor_id: 8,
            },
            DirectInteractionTerminalClosure {
                root_component_base: 20,
                amplitude_destination_id: 1,
                exact_factor_id: 9,
            },
        ];
        let terminal = DirectInteractionTerminalBatch {
            partner_component_base: 30,
            component_factor_start: 4,
            amplitude_destination_count: 2,
            closures: &terminal_closures,
        };
        let anchor = [DirectInteractionAnchor {
            parent_component_bases: [0, 4],
            parent_momentum_form_ids: [0, 1],
            operands: DirectInteractionOperands::One(DirectInteractionOperand {
                tensor_component_base: 8,
                vector_parent: DirectInteractionVectorParent::Parent0,
            }),
            output_start: 0,
            output_end: 2,
            tensor_output_start: 0,
            tensor_output_end: 0,
        }];
        let outputs = [
            DirectInteractionOutput {
                destination_component_base: 20,
                exact_factor_ids: [0, 2, 0],
                initialize: true,
            },
            DirectInteractionOutput {
                destination_component_base: 20,
                exact_factor_ids: [1, 3, 0],
                initialize: false,
            },
        ];
        let color = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector,
            RecurrenceIntrinsicScale::new(0.0, 0.69, None).unwrap(),
        );
        let atv = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::AntisymmetricTensorVectorToVector,
            RecurrenceIntrinsicScale::new(0.0, -0.41, None).unwrap(),
        );
        let color_handle = color.contribution_handle();
        let atv_handle = atv.contribution_handle();
        let contexts = DirectInteractionExecutorContexts {
            color: color_handle.context,
            vector_wedge_vector: ptr::null(),
            antisymmetric_tensor_vector: atv_handle.context,
        };
        let momentum_view = DirectMomentumView {
            values: momenta_values.as_ptr(),
            scalar_len: momenta_values.len() as u64,
            form_count: 2,
            lorentz_component_count: 4,
            point_stride: 1,
        };
        let parameter_view = DirectParameterView {
            values_re: parameters_re.as_ptr(),
            values_im: parameters_im.as_ptr(),
            value_count: 1,
        };
        let factor_view = DirectFactorView {
            values_re: factors_re.as_ptr(),
            values_im: factors_im.as_ptr(),
            value_count: factors_re.len() as u32,
        };
        let arena = |re: &mut [f64],
                     im: &mut [f64],
                     amplitude_re: &mut [f64],
                     amplitude_im: &mut [f64]|
         -> DirectArenaView {
            DirectArenaView {
                current_re: re.as_mut_ptr(),
                current_im: im.as_mut_ptr(),
                current_scalar_len: re.len() as u64,
                amplitude_re: amplitude_re.as_mut_ptr(),
                amplitude_im: amplitude_im.as_mut_ptr(),
                amplitude_scalar_len: amplitude_re.len() as u64,
                point_stride: 1,
            }
        };
        let requirements = DirectInteractionRuntimeRequirements {
            current_component_count: 34,
            momentum_form_count: 2,
            parameter_count: 0,
            factor_count: factors_re.len() as u32,
        };

        let mut expected_re = seed_re.clone();
        let mut expected_im = seed_im.clone();
        let mut expected_amplitude_re = vec![0.13, -0.27];
        let mut expected_amplitude_im = vec![-0.19, 0.08];
        unsafe {
            execute_interaction_bundle(
                contexts,
                arena(
                    &mut expected_re,
                    &mut expected_im,
                    &mut expected_amplitude_re,
                    &mut expected_amplitude_im,
                ),
                momentum_view,
                parameter_view,
                factor_view,
                DirectInteractionBundleBatch {
                    anchors: &anchor,
                    outputs: &outputs,
                    tensor_outputs: &[],
                    terminal: None,
                    requirements,
                    atv_before_color: false,
                },
            )
        }
        .unwrap();
        assert_eq!(
            unsafe {
                crate::engine::execute_closure_reduce_rows(
                    ptr::null(),
                    arena(
                        &mut expected_re,
                        &mut expected_im,
                        &mut expected_amplitude_re,
                        &mut expected_amplitude_im,
                    ),
                    momentum_view,
                    parameter_view,
                    factor_view,
                    closure_rows.as_ptr(),
                    closure_rows.len() as u32,
                    1,
                )
            },
            DIRECT_STATUS_OK
        );

        let mut actual_re = seed_re.clone();
        let mut actual_im = seed_im.clone();
        let mut actual_amplitude_re = vec![0.13, -0.27];
        let mut actual_amplitude_im = vec![-0.19, 0.08];
        let actual_arena = arena(
            &mut actual_re,
            &mut actual_im,
            &mut actual_amplitude_re,
            &mut actual_amplitude_im,
        );
        unsafe {
            execute_interaction_bundle(
                contexts,
                actual_arena,
                momentum_view,
                parameter_view,
                factor_view,
                DirectInteractionBundleBatch {
                    anchors: &anchor,
                    outputs: &outputs,
                    tensor_outputs: &[],
                    terminal: Some(terminal),
                    requirements,
                    atv_before_color: false,
                },
            )
        }
        .unwrap();
        unsafe { execute_interaction_terminal_closures(actual_arena, factor_view, terminal) }
            .unwrap();

        for destination in 0..2 {
            let error = (actual_amplitude_re[destination] - expected_amplitude_re[destination])
                .hypot(actual_amplitude_im[destination] - expected_amplitude_im[destination]);
            let scale = actual_amplitude_re[destination]
                .hypot(actual_amplitude_im[destination])
                .max(expected_amplitude_re[destination].hypot(expected_amplitude_im[destination]))
                .max(1.0);
            assert!(
                error <= 2.0e-12 * scale,
                "destination {destination}: {error:e}"
            );
        }
    }

    #[test]
    fn color_ordered_three_vector_matches_reference_math_and_accumulates() {
        let stride = 5usize;
        let point_count = 4usize;
        let mut current_re = vec![0.0; 12 * stride];
        let mut current_im = vec![0.0; 12 * stride];
        for component in 0..12 {
            fill_component(
                &mut current_re,
                &mut current_im,
                stride,
                component,
                -0.9 + component as f64 * 0.37,
            );
        }
        let before_re = current_re.clone();
        let before_im = current_im.clone();
        let mut momentum_values = vec![0.0; 2 * 4 * stride];
        for form in 0..2 {
            for component in 0..4 {
                for point in 0..stride {
                    momentum_values[(form * 4 + component) * stride + point] =
                        0.23 + form as f64 * 1.7 - component as f64 * 0.41 + point as f64 * 0.19;
                }
            }
        }
        let parameters_re = [-0.35];
        let parameters_im = [0.8];
        let factors_re = [1.2];
        let factors_im = [0.25];
        let (arena, _, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            stride as u32,
            &parameters_re,
            &parameters_im,
            &factors_re,
            &factors_im,
        );
        let momenta = DirectMomentumView {
            values: momentum_values.as_ptr(),
            scalar_len: momentum_values.len() as u64,
            form_count: 2,
            lorentz_component_count: 4,
            point_stride: stride as u32,
        };
        let literal = ReferenceComplex::new(0.45, -0.7);
        let executor = LoadedRecurrenceIntrinsicDirectExecutor::load_runtime_template(
            COLOR_ORDERED_THREE_VECTOR_TEMPLATE,
            RecurrenceIntrinsicScale::new(literal.re, literal.im, Some(0)).unwrap(),
        )
        .unwrap();
        let handle = executor.contribution_handle();
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
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&row),
                    1,
                    point_count as u32,
                )
            },
            DIRECT_STATUS_OK
        );

        let alpha = effective_test_scale(
            literal,
            ReferenceComplex::new(parameters_re[0], parameters_im[0]),
            ReferenceComplex::new(factors_re[0], factors_im[0]),
        );
        for point in 0..point_count {
            let left = std::array::from_fn::<_, 4, _>(|component| {
                read_component(&before_re, &before_im, stride, component, point)
            });
            let right = std::array::from_fn::<_, 4, _>(|component| {
                read_component(&before_re, &before_im, stride, 4 + component, point)
            });
            let parent0_momentum = std::array::from_fn::<_, 4, _>(|component| {
                momentum_values[component * stride + point]
            });
            let parent1_momentum = std::array::from_fn::<_, 4, _>(|component| {
                momentum_values[(4 + component) * stride + point]
            });
            let vector_dot = left[0]
                .mul(right[0])
                .sub(left[1].mul(right[1]))
                .sub(left[2].mul(right[2]))
                .sub(left[3].mul(right[3]));
            let left_dot_parent1 = left[0]
                .mul_real(parent1_momentum[0])
                .sub(left[1].mul_real(parent1_momentum[1]))
                .sub(left[2].mul_real(parent1_momentum[2]))
                .sub(left[3].mul_real(parent1_momentum[3]));
            let right_dot_parent0 = right[0]
                .mul_real(parent0_momentum[0])
                .sub(right[1].mul_real(parent0_momentum[1]))
                .sub(right[2].mul_real(parent0_momentum[2]))
                .sub(right[3].mul_real(parent0_momentum[3]));

            for component in 0..4 {
                let reference = vector_dot
                    .mul_real(parent0_momentum[component] - parent1_momentum[component])
                    .add(
                        left_dot_parent1
                            .mul(right[component])
                            .sub(right_dot_parent0.mul(left[component]))
                            .mul_real(2.0),
                    );
                let initial = read_component(&before_re, &before_im, stride, 8 + component, point);
                let actual = read_component(&current_re, &current_im, stride, 8 + component, point);
                assert_close(actual, initial.add(reference.mul(alpha)));
            }
        }
        for component in 8..12 {
            assert_close(
                read_component(&current_re, &current_im, stride, component, point_count),
                read_component(&before_re, &before_im, stride, component, point_count),
            );
        }

        let mut expected = Vec::new();
        for component in 8..12 {
            for point in 0..point_count {
                expected.push(
                    read_component(&current_re, &current_im, stride, component, point).sub(
                        read_component(&before_re, &before_im, stride, component, point),
                    ),
                );
            }
        }
        for component in 8..12 {
            let start = component * stride;
            current_re[start..start + point_count].fill(f64::NAN);
            current_im[start..start + point_count].fill(f64::NAN);
        }
        let initializing = DirectContributionRow {
            flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            ..row
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&initializing),
                    1,
                    point_count as u32,
                )
            },
            DIRECT_STATUS_OK
        );
        for (index, (component, point)) in (8..12)
            .flat_map(|component| (0..point_count).map(move |point| (component, point)))
            .enumerate()
        {
            assert_close(
                read_component(&current_re, &current_im, stride, component, point),
                expected[index],
            );
        }
    }

    #[test]
    fn color_ordered_three_vector_rejects_malformed_momentum_views_and_rows() {
        let stride = 3usize;
        let mut current_re = vec![0.0; 12 * stride];
        let mut current_im = vec![0.0; 12 * stride];
        let momentum_values = vec![0.0; 2 * 4 * stride];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let (arena, _, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            stride as u32,
            &[],
            &[],
            &factors_re,
            &factors_im,
        );
        let valid_momenta = DirectMomentumView {
            values: momentum_values.as_ptr(),
            scalar_len: momentum_values.len() as u64,
            form_count: 2,
            lorentz_component_count: 4,
            point_stride: stride as u32,
        };
        let executor = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::ColorOrderedThreeVector,
            RecurrenceIntrinsicScale::unit(),
        );
        let handle = executor.contribution_handle();
        let valid_row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 4,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 1,
            destination_component_base: 8,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        let call = |momenta: DirectMomentumView, row: DirectContributionRow, point_count: u32| unsafe {
            (handle.call)(
                handle.context,
                arena,
                momenta,
                parameters,
                factors,
                ptr::from_ref(&row),
                1,
                point_count,
            )
        };

        assert_eq!(
            call(
                DirectMomentumView {
                    values: ptr::null(),
                    ..valid_momenta
                },
                valid_row,
                stride as u32,
            ),
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            call(
                DirectMomentumView {
                    lorentz_component_count: 3,
                    ..valid_momenta
                },
                valid_row,
                stride as u32,
            ),
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            call(
                DirectMomentumView {
                    point_stride: stride as u32 + 1,
                    ..valid_momenta
                },
                valid_row,
                stride as u32,
            ),
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            call(
                valid_momenta,
                DirectContributionRow {
                    parent0_momentum_form_id: DIRECT_NONE_U32,
                    ..valid_row
                },
                stride as u32,
            ),
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            call(
                valid_momenta,
                DirectContributionRow {
                    parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                    ..valid_row
                },
                stride as u32,
            ),
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            call(
                valid_momenta,
                DirectContributionRow {
                    parent1_momentum_form_id_or_sentinel: 2,
                    ..valid_row
                },
                stride as u32,
            ),
            STATUS_BOUNDS
        );
        assert_eq!(
            call(
                DirectMomentumView {
                    scalar_len: valid_momenta.scalar_len - 1,
                    ..valid_momenta
                },
                valid_row,
                stride as u32,
            ),
            STATUS_BOUNDS
        );
        assert_eq!(
            call(valid_momenta, valid_row, stride as u32 + 1),
            STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn callback_rejects_invalid_context_arguments_and_bounds() {
        let stride = 2usize;
        let mut current_re = vec![0.0; 8 * stride];
        let mut current_im = vec![0.0; 8 * stride];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            stride as u32,
            &[],
            &[],
            &factors_re,
            &factors_im,
        );
        let executor = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylPositive,
            RecurrenceIntrinsicScale::unit(),
        );
        let handle = executor.contribution_handle();
        let valid_row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 2,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 6,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    ptr::null(),
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&valid_row),
                    1,
                    stride as u32,
                )
            },
            STATUS_INVALID_CONTEXT
        );
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&valid_row),
                    1,
                    stride as u32 + 1,
                )
            },
            STATUS_INVALID_ARGUMENT
        );

        let missing_parent = DirectContributionRow {
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            ..valid_row
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&missing_parent),
                    1,
                    stride as u32,
                )
            },
            STATUS_INVALID_ARGUMENT
        );

        let bad_factor = DirectContributionRow {
            exact_factor_id: 1,
            ..valid_row
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&bad_factor),
                    1,
                    stride as u32,
                )
            },
            STATUS_BOUNDS
        );

        let bad_destination = DirectContributionRow {
            destination_component_base: 7,
            ..valid_row
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&bad_destination),
                    1,
                    stride as u32,
                )
            },
            STATUS_BOUNDS
        );
    }

    #[test]
    fn optional_parameter_index_is_checked_before_row_execution() {
        let stride = 2usize;
        let mut current_re = vec![0.0; 8 * stride];
        let mut current_im = vec![0.0; 8 * stride];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let (arena, momenta, parameters, factors) = views(
            &mut current_re,
            &mut current_im,
            stride as u32,
            &[],
            &[],
            &factors_re,
            &factors_im,
        );
        let executor = LoadedRecurrenceIntrinsicDirectExecutor::load(
            RecurrenceContributionIntrinsicKind::WeylVectorToWeylNegative,
            RecurrenceIntrinsicScale::new(1.0, 0.0, Some(0)).unwrap(),
        );
        let handle = executor.contribution_handle();
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 2,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 6,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert_eq!(
            unsafe {
                (handle.call)(
                    handle.context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::from_ref(&row),
                    1,
                    stride as u32,
                )
            },
            STATUS_BOUNDS
        );
    }

    #[test]
    fn scale_constructor_rejects_nonfinite_values_and_reserved_parameter_index() {
        assert!(RecurrenceIntrinsicScale::new(f64::NAN, 0.0, None).is_err());
        assert!(RecurrenceIntrinsicScale::new(1.0, f64::INFINITY, None).is_err());
        assert!(RecurrenceIntrinsicScale::new(1.0, 0.0, Some(DIRECT_NONE_U32)).is_err());
    }
}
