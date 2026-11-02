// SPDX-License-Identifier: 0BSD

//! Process-owned fixed-width recurrence color-contraction payload.

use std::collections::{BTreeMap, BTreeSet};
use std::iter::FusedIterator;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::exact::{ExactComplexRational, ExactRational};
use super::symmetric_group_fft::{
    SymmetricGroupComplex64, SymmetricGroupFftPlan, SymmetricGroupFftWorkspace,
};
use crate::{RusticolError, RusticolResult};

pub const RECURRENCE_COLOR_CONTRACTION_CODEC_ABI: &str =
    "pyamplicol-recurrence-color-contraction-v3";

const MAGIC: &[u8; 8] = b"PACRCLR3";
const VERSION: u32 = 3;
const HEADER_BYTES: usize = 120;
const ENTRY_BYTES: usize = 36;
const EXACT_FACTOR_BYTES: usize = 64;
// The subgroup order must fit the codec's u32 local-group count.
const MAX_FACTOR_RANK: u32 = u32::BITS - 1;
const MAX_SYMMETRIC_GROUP_DEGREE: u32 = 10;
// A batching target, not a resource limit: a single large lane and explicitly
// requested workspace growth may exceed it.
const TARGET_SYMMETRIC_GROUP_LANE_WORKSPACE_BYTES: usize = 512 * 1024 * 1024;
const ZERO_SECTOR_OWNER: u32 = u32::MAX;
const FLAG_INCLUDES_COLOR_FACTOR: u32 = 1 << 0;
const KNOWN_FLAGS: u32 = FLAG_INCLUDES_COLOR_FACTOR;

fn malformed(message: impl Into<String>) -> RusticolError {
    RusticolError::artifact(format!(
        "recurrence color-contraction codec: {}",
        message.into()
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RecurrenceColorAccuracy {
    Nlc = 1,
    Full = 2,
}

impl RecurrenceColorAccuracy {
    fn decode(value: u32) -> RusticolResult<Self> {
        match value {
            1 => Ok(Self::Nlc),
            2 => Ok(Self::Full),
            _ => Err(malformed(format!(
                "unknown color-accuracy discriminant {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RecurrenceColorStorage {
    Expanded = 1,
    Repeated = 2,
    ConvolutionKernels = 3,
}

impl RecurrenceColorStorage {
    fn decode(value: u32) -> RusticolResult<Self> {
        match value {
            1 => Ok(Self::Expanded),
            2 => Ok(Self::Repeated),
            3 => Ok(Self::ConvolutionKernels),
            _ => Err(malformed(format!(
                "unknown color-contraction storage discriminant {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawColorContractionEntry {
    pub left_group_id: u32,
    pub right_group_id: u32,
    pub weight_re: f64,
    pub weight_im: f64,
    pub symmetry_factor: f64,
    pub exact_factor_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanonicalColorContractionEntry {
    pub left_group_id: u32,
    pub right_group_id: u32,
    pub left_destination_id: u32,
    pub right_destination_id: u32,
    pub weight_re: f64,
    pub weight_im: f64,
    pub symmetry_factor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeColorContractionEntry {
    pub left_destination_id: u32,
    pub right_destination_id: u32,
    pub coefficient_re: f64,
    pub coefficient_im: f64,
}

impl RuntimeColorContractionEntry {
    /// Evaluate `Re(coefficient * left * conj(right))` in the canonical
    /// expanded-contraction operation order.
    ///
    /// The coefficient supplied by [`RecurrenceColorContraction::runtime_entries`]
    /// already includes the upper-triangle symmetry factor.
    #[inline(always)]
    pub fn contract_real_bilinear(
        self,
        left_re: f64,
        left_im: f64,
        right_re: f64,
        right_im: f64,
    ) -> f64 {
        let product_re = left_re.mul_add(right_re, left_im * right_im);
        let product_im = left_im.mul_add(right_re, -left_re * right_im);
        self.coefficient_re
            .mul_add(product_re, -self.coefficient_im * product_im)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FactorizedColorContractionKind {
    KleinFourWalsh,
    ElementaryAbelianWalsh,
    SymmetricGroupFourier,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorizedColorContraction {
    kind: FactorizedColorContractionKind,
    rank: u32,
    coset_count: usize,
    coset_indices: Vec<u32>,
}

impl FactorizedColorContraction {
    pub fn kind(&self) -> FactorizedColorContractionKind {
        self.kind
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn subgroup_order(&self) -> usize {
        match self.kind {
            FactorizedColorContractionKind::KleinFourWalsh
            | FactorizedColorContractionKind::ElementaryAbelianWalsh => 1usize << self.rank,
            FactorizedColorContractionKind::SymmetricGroupFourier => {
                checked_factorial(self.rank).expect("validated symmetric-group degree")
            }
        }
    }

    pub fn coset_count(&self) -> usize {
        self.coset_count
    }

    pub fn coset(&self, index: usize) -> Option<&[u32]> {
        let order = self.subgroup_order();
        let start = index.checked_mul(order)?;
        self.coset_indices.get(start..start.checked_add(order)?)
    }

    pub fn coset_indices(&self) -> &[u32] {
        &self.coset_indices
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeFactorizedColorContractionEntry {
    pub left_group_index: u32,
    pub right_group_index: u32,
    pub coefficient_re: f64,
    pub coefficient_im: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeFactorizedColorContraction {
    subgroup_order: usize,
    cosets: Vec<Vec<u32>>,
    entries: Vec<RuntimeFactorizedColorContractionEntry>,
    amplitude_scale: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeSymmetricGroupKernel {
    left_channel_index: u32,
    right_channel_index: u32,
    pair_scale: f64,
    fourier_coefficients: Box<[f64]>,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeSymmetricGroupLowRankBlock {
    rank: usize,
    /// Column-major `dimension x rank` real factor in the FFT block's
    /// original Young-basis order.
    factors: Box<[f64]>,
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeSymmetricGroupLowRankKernel {
    /// Block-aligned scalar factors for the corresponding dense kernel.
    /// Empty for cross-channel kernels and diagonal kernels with no accepted
    /// factor.
    blocks: Box<[Option<RuntimeSymmetricGroupLowRankBlock>]>,
}

impl RuntimeSymmetricGroupKernel {
    pub fn left_channel_index(&self) -> u32 {
        self.left_channel_index
    }

    pub fn right_channel_index(&self) -> u32 {
        self.right_channel_index
    }

    /// Upper-triangle channel-pair scale: one on the diagonal and two for a
    /// cross-channel kernel whose Hermitian-adjoint partner is implicit.
    pub fn pair_scale(&self) -> f64 {
        self.pair_scale
    }

    /// Packed real Young blocks for the unnormalised transform of
    /// `r -> k_cd(r^-1)`.  Blocks are column-major in the shared FFT plan's
    /// canonical order.
    pub fn fourier_coefficients(&self) -> &[f64] {
        &self.fourier_coefficients
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeSymmetricGroupColorContraction {
    degree: u32,
    group_order: usize,
    channel_count: usize,
    local_group_count: usize,
    fft_plan: Arc<SymmetricGroupFftPlan>,
    kernels: Arc<[RuntimeSymmetricGroupKernel]>,
    scalar_same_channel_low_rank_kernels: Option<Arc<[RuntimeSymmetricGroupLowRankKernel]>>,
    residual_entries: Arc<[RuntimeFactorizedColorContractionEntry]>,
}

impl PartialEq for RuntimeSymmetricGroupColorContraction {
    fn eq(&self, other: &Self) -> bool {
        self.degree == other.degree
            && self.group_order == other.group_order
            && self.channel_count == other.channel_count
            && self.local_group_count == other.local_group_count
            && self.kernels.as_ref() == other.kernels.as_ref()
            && self.scalar_same_channel_low_rank_kernels
                == other.scalar_same_channel_low_rank_kernels
            && self.residual_entries.as_ref() == other.residual_entries.as_ref()
    }
}

impl RuntimeSymmetricGroupColorContraction {
    pub fn degree(&self) -> u32 {
        self.degree
    }

    pub fn group_order(&self) -> usize {
        self.group_order
    }

    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    pub fn local_group_count(&self) -> usize {
        self.local_group_count
    }

    pub fn kernels(&self) -> &[RuntimeSymmetricGroupKernel] {
        self.kernels.as_ref()
    }

    pub fn residual_entries(&self) -> &[RuntimeFactorizedColorContractionEntry] {
        self.residual_entries.as_ref()
    }

    pub(crate) fn workspace(
        &self,
        lane_capacity: usize,
    ) -> RusticolResult<RuntimeSymmetricGroupColorWorkspace> {
        RuntimeSymmetricGroupColorWorkspace::new(self, lane_capacity)
    }

    pub(crate) fn bounded_lane_capacity(&self, requested: usize) -> RusticolResult<usize> {
        if requested == 0 {
            return Err(RusticolError::invalid_argument(
                "symmetric-group requested lane capacity must be positive",
            ));
        }
        let complex_values_per_lane = self
            .local_group_count
            .checked_add(
                self.channel_count
                    .checked_mul(self.group_order)
                    .ok_or_else(|| {
                        malformed("symmetric-group transformed lane size overflows usize")
                    })?,
            )
            .and_then(|value| value.checked_add(self.group_order))
            .and_then(|value| {
                value.checked_add(
                    self.fft_plan
                        .maximum_block_dimension()
                        .checked_mul(self.fft_plan.maximum_block_dimension())?,
                )
            })
            .ok_or_else(|| malformed("symmetric-group lane workspace size overflows usize"))?;
        let bytes_per_lane = complex_values_per_lane
            .checked_mul(std::mem::size_of::<SymmetricGroupComplex64>())
            .and_then(|value| value.checked_add(std::mem::size_of::<f64>()))
            .ok_or_else(|| malformed("symmetric-group lane workspace bytes overflow usize"))?;
        let budget_lanes = (TARGET_SYMMETRIC_GROUP_LANE_WORKSPACE_BYTES / bytes_per_lane).max(1);
        Ok(requested.min(budget_lanes))
    }

    /// Reduce one component/helicity tile.  A call that exceeds the retained
    /// workspace high-water mark may grow it fallibly; repeated calls within
    /// the retained capacity allocate nothing.  The amplitude accessor
    /// receives a normalized local-group index and active lane.
    pub(crate) fn reduce_lanes(
        &self,
        workspace: &mut RuntimeSymmetricGroupColorWorkspace,
        lane_count: usize,
        mut amplitude: impl FnMut(usize, usize) -> RusticolResult<SymmetricGroupComplex64>,
    ) -> RusticolResult<()> {
        workspace.ensure_lane_capacity(self, lane_count)?;
        workspace.validate(self, lane_count)?;
        workspace.reduced[..lane_count].fill(0.0);

        for group in 0..self.local_group_count {
            let start = group * lane_count;
            for lane in 0..lane_count {
                workspace.gathered[start + lane] = amplitude(group, lane)?;
            }
        }

        // Residual and eligible/residual cross rows are contracted while the
        // gathered amplitudes are still in their authenticated local order.
        for entry in self.residual_entries.iter() {
            let left = entry.left_group_index as usize * lane_count;
            let right = entry.right_group_index as usize * lane_count;
            for lane in 0..lane_count {
                let left_value = workspace.gathered[left + lane];
                let right_value = workspace.gathered[right + lane];
                let product_re = left_value
                    .0
                    .mul_add(right_value.0, left_value.1 * right_value.1);
                let product_im = left_value
                    .1
                    .mul_add(right_value.0, -left_value.0 * right_value.1);
                workspace.reduced[lane] += entry
                    .coefficient_re
                    .mul_add(product_re, -entry.coefficient_im * product_im);
            }
        }

        let channel_stride = self.group_order * lane_count;
        for channel in 0..self.channel_count {
            let start = channel * channel_stride;
            self.fft_plan.forward_lanes(
                lane_count,
                &workspace.gathered[start..start + channel_stride],
                &mut workspace.transformed[start..start + channel_stride],
                &mut workspace.fft,
            )?;
        }

        if let Some(low_rank_kernels) = self.scalar_same_channel_low_rank_kernels.as_deref() {
            self.reduce_transformed_low_rank(
                workspace,
                lane_count,
                channel_stride,
                low_rank_kernels,
            );
        } else {
            self.reduce_transformed_dense(workspace, lane_count, channel_stride);
        }
        Ok(())
    }

    #[inline(never)]
    fn reduce_transformed_dense(
        &self,
        workspace: &mut RuntimeSymmetricGroupColorWorkspace,
        lane_count: usize,
        channel_stride: usize,
    ) {
        let normalization = 1.0 / self.group_order as f64;
        for kernel in self.kernels.iter() {
            let left_channel = kernel.left_channel_index as usize;
            let right_channel = kernel.right_channel_index as usize;
            let left = &workspace.transformed
                [left_channel * channel_stride..(left_channel + 1) * channel_stride];
            let right = &workspace.transformed
                [right_channel * channel_stride..(right_channel + 1) * channel_stride];
            for block in self.fft_plan.blocks() {
                let dimension = block.dimension();
                let offset = block.coefficient_offset();
                let coefficients =
                    &kernel.fourier_coefficients[offset..offset + block.coefficient_count()];
                let block_weight = dimension as f64 * normalization * kernel.pair_scale;
                if left_channel == right_channel {
                    contract_real_hermitian_same_channel_block(
                        left,
                        coefficients,
                        dimension,
                        offset,
                        lane_count,
                        lane_count,
                        block_weight,
                        &mut workspace.reduced,
                    );
                } else {
                    contract_general_cross_block(
                        left,
                        right,
                        coefficients,
                        dimension,
                        offset,
                        lane_count,
                        lane_count,
                        block_weight,
                        &mut workspace.reduced,
                    );
                }
            }
        }
    }

    #[inline(never)]
    fn reduce_transformed_low_rank(
        &self,
        workspace: &mut RuntimeSymmetricGroupColorWorkspace,
        lane_count: usize,
        channel_stride: usize,
        low_rank_kernels: &[RuntimeSymmetricGroupLowRankKernel],
    ) {
        debug_assert_eq!(low_rank_kernels.len(), self.kernels.len());
        let normalization = 1.0 / self.group_order as f64;
        for (kernel_index, kernel) in self.kernels.iter().enumerate() {
            let left_channel = kernel.left_channel_index as usize;
            let right_channel = kernel.right_channel_index as usize;
            let left = &workspace.transformed
                [left_channel * channel_stride..(left_channel + 1) * channel_stride];
            let right = &workspace.transformed
                [right_channel * channel_stride..(right_channel + 1) * channel_stride];
            if left_channel == right_channel {
                let scalar_blocks: &[Option<RuntimeSymmetricGroupLowRankBlock>] = low_rank_kernels
                    .get(kernel_index)
                    .map(|kernel| kernel.blocks.as_ref())
                    .unwrap_or_default();
                if scalar_blocks.is_empty() {
                    for block in self.fft_plan.blocks() {
                        let dimension = block.dimension();
                        let offset = block.coefficient_offset();
                        let coefficients = &kernel.fourier_coefficients
                            [offset..offset + block.coefficient_count()];
                        let block_weight = dimension as f64 * normalization * kernel.pair_scale;
                        contract_real_hermitian_same_channel_block(
                            left,
                            coefficients,
                            dimension,
                            offset,
                            lane_count,
                            lane_count,
                            block_weight,
                            &mut workspace.reduced,
                        );
                    }
                } else {
                    for (block_index, block) in self.fft_plan.blocks().enumerate() {
                        let dimension = block.dimension();
                        let offset = block.coefficient_offset();
                        let coefficients = &kernel.fourier_coefficients
                            [offset..offset + block.coefficient_count()];
                        let block_weight = dimension as f64 * normalization * kernel.pair_scale;
                        contract_real_hermitian_same_channel_block_opportunistic(
                            left,
                            coefficients,
                            scalar_blocks.get(block_index).and_then(Option::as_ref),
                            dimension,
                            offset,
                            lane_count,
                            lane_count,
                            block_weight,
                            &mut workspace.reduced,
                        );
                    }
                }
            } else {
                for block in self.fft_plan.blocks() {
                    let dimension = block.dimension();
                    let offset = block.coefficient_offset();
                    let coefficients =
                        &kernel.fourier_coefficients[offset..offset + block.coefficient_count()];
                    let block_weight = dimension as f64 * normalization * kernel.pair_scale;
                    contract_general_cross_block(
                        left,
                        right,
                        coefficients,
                        dimension,
                        offset,
                        lane_count,
                        lane_count,
                        block_weight,
                        &mut workspace.reduced,
                    );
                }
            }
        }
    }
}

/// Process-local mutable storage shared by the recurrence and on-the-fly
/// contraction seams. Capacity starts at the first positive reduction size,
/// grows fallibly up to the bounded point-tile high-water mark, and never
/// shrinks; every transform keeps the point lane as its innermost coordinate.
#[derive(Debug)]
pub(crate) struct RuntimeSymmetricGroupColorWorkspace {
    degree: u32,
    group_order: usize,
    channel_count: usize,
    local_group_count: usize,
    lane_capacity: usize,
    gathered: Vec<SymmetricGroupComplex64>,
    transformed: Vec<SymmetricGroupComplex64>,
    reduced: Vec<f64>,
    fft: SymmetricGroupFftWorkspace,
}

impl RuntimeSymmetricGroupColorWorkspace {
    fn new(
        contraction: &RuntimeSymmetricGroupColorContraction,
        lane_capacity: usize,
    ) -> RusticolResult<Self> {
        if lane_capacity == 0 {
            return Err(RusticolError::invalid_argument(
                "symmetric-group color workspace lane capacity must be positive",
            ));
        }
        let gathered_count = contraction
            .local_group_count
            .checked_mul(lane_capacity)
            .ok_or_else(|| malformed("symmetric-group gathered workspace size overflows usize"))?;
        let transformed_count = contraction
            .channel_count
            .checked_mul(contraction.group_order)
            .and_then(|value| value.checked_mul(lane_capacity))
            .ok_or_else(|| {
                malformed("symmetric-group transformed workspace size overflows usize")
            })?;
        let mut gathered = Vec::new();
        gathered
            .try_reserve_exact(gathered_count)
            .map_err(|error| {
                RusticolError::internal(format!(
                    "symmetric-group gathered workspace allocation failed: {error}"
                ))
            })?;
        gathered.resize(gathered_count, (0.0, 0.0));
        let mut transformed = Vec::new();
        transformed
            .try_reserve_exact(transformed_count)
            .map_err(|error| {
                RusticolError::internal(format!(
                    "symmetric-group transformed workspace allocation failed: {error}"
                ))
            })?;
        transformed.resize(transformed_count, (0.0, 0.0));
        let mut reduced = Vec::new();
        reduced.try_reserve_exact(lane_capacity).map_err(|error| {
            RusticolError::internal(format!(
                "symmetric-group reduced workspace allocation failed: {error}"
            ))
        })?;
        reduced.resize(lane_capacity, 0.0);
        let fft = contraction.fft_plan.workspace(lane_capacity)?;
        Ok(Self {
            degree: contraction.degree,
            group_order: contraction.group_order,
            channel_count: contraction.channel_count,
            local_group_count: contraction.local_group_count,
            lane_capacity,
            gathered,
            transformed,
            reduced,
            fft,
        })
    }

    fn validate(
        &self,
        contraction: &RuntimeSymmetricGroupColorContraction,
        lane_count: usize,
    ) -> RusticolResult<()> {
        if lane_count == 0 || lane_count > self.lane_capacity {
            return Err(RusticolError::invalid_argument(
                "symmetric-group color reduction exceeds its positive lane capacity",
            ));
        }
        if self.degree != contraction.degree
            || self.group_order != contraction.group_order
            || self.channel_count != contraction.channel_count
            || self.local_group_count != contraction.local_group_count
        {
            return Err(RusticolError::integrity(
                "symmetric-group color workspace does not match its contraction plan",
            ));
        }
        Ok(())
    }

    pub(crate) fn ensure_lane_capacity(
        &mut self,
        contraction: &RuntimeSymmetricGroupColorContraction,
        lane_count: usize,
    ) -> RusticolResult<()> {
        if lane_count <= self.lane_capacity {
            return Ok(());
        }
        *self = Self::new(contraction, lane_count)?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) const fn lane_capacity(&self) -> usize {
        self.lane_capacity
    }

    pub(crate) fn reduced(&self, lane_count: usize) -> RusticolResult<&[f64]> {
        if lane_count == 0 || lane_count > self.lane_capacity {
            return Err(RusticolError::invalid_argument(
                "symmetric-group reduced lane view is outside its workspace",
            ));
        }
        Ok(&self.reduced[..lane_count])
    }
}

const LOW_RANK_PIVOT_TOLERANCE_MULTIPLIER: f64 = 64.0;
const LOW_RANK_MODE_BLOCK: usize = 4;

fn compile_scalar_same_channel_low_rank_block(
    kernel: &[f64],
    dimension: usize,
) -> Option<RuntimeSymmetricGroupLowRankBlock> {
    if dimension == 0 || kernel.len() != dimension.checked_mul(dimension)? {
        return None;
    }

    // The scalar dense evaluator pairs K_ci and K_ic.  Factor exactly that
    // effective real-symmetric matrix rather than either rounded triangle.
    let mut effective = vec![0.0; kernel.len()];
    for column in 0..dimension {
        for row in 0..=column {
            let value = if row == column {
                kernel[column * dimension + row]
            } else {
                0.5 * kernel[column * dimension + row] + 0.5 * kernel[row * dimension + column]
            };
            if !value.is_finite() {
                return None;
            }
            effective[column * dimension + row] = value;
            effective[row * dimension + column] = value;
        }
    }

    let scale = (0..dimension)
        .map(|index| effective[index * dimension + index])
        .fold(f64::NEG_INFINITY, f64::max);
    if !scale.is_finite() || scale < 0.0 {
        return None;
    }
    let tolerance = LOW_RANK_PIVOT_TOLERANCE_MULTIPLIER * f64::EPSILON * dimension as f64 * scale;
    if !tolerance.is_finite() {
        return None;
    }

    let mut residual_diagonal = (0..dimension)
        .map(|index| effective[index * dimension + index])
        .collect::<Vec<_>>();
    if residual_diagonal.iter().any(|&value| value < -tolerance) {
        return None;
    }
    for value in &mut residual_diagonal {
        if *value < 0.0 {
            *value = 0.0;
        }
    }

    let mut selected = vec![false; dimension];
    let mut factors = Vec::<f64>::with_capacity(kernel.len());
    let mut rank = 0usize;
    loop {
        // Ascending traversal plus strict comparison makes equal pivots choose
        // the lowest original Young-basis index deterministically.
        let mut pivot = None;
        for index in 0..dimension {
            if selected[index] {
                continue;
            }
            if residual_diagonal[index] < -tolerance {
                return None;
            }
            if pivot.is_none_or(|best| residual_diagonal[index] > residual_diagonal[best]) {
                pivot = Some(index);
            }
        }
        let Some(pivot) = pivot else {
            break;
        };
        let pivot_value = residual_diagonal[pivot];
        if pivot_value <= tolerance {
            break;
        }
        let pivot_root = pivot_value.sqrt();
        if !pivot_root.is_finite() || pivot_root == 0.0 {
            return None;
        }

        let mut factor_column = vec![0.0; dimension];
        factor_column[pivot] = pivot_root;
        for row in 0..dimension {
            if row == pivot || selected[row] {
                continue;
            }
            let mut value = effective[pivot * dimension + row];
            for previous in 0..rank {
                value = (-factors[previous * dimension + row])
                    .mul_add(factors[previous * dimension + pivot], value);
            }
            value /= pivot_root;
            if !value.is_finite() {
                return None;
            }
            factor_column[row] = value;
        }

        selected[pivot] = true;
        residual_diagonal[pivot] = 0.0;
        for row in 0..dimension {
            if selected[row] {
                continue;
            }
            let value = factor_column[row];
            let updated = (-value).mul_add(value, residual_diagonal[row]);
            if !updated.is_finite() || updated < -tolerance {
                return None;
            }
            residual_diagonal[row] = if updated < 0.0 { 0.0 } else { updated };
        }
        factors.extend_from_slice(&factor_column);
        rank += 1;
        // Factor cost grows monotonically with rank. Once this block cannot
        // clear the conservative warm-work gate, no later pivot can recover.
        if !low_rank_scalar_cost_is_profitable(dimension, rank) {
            return None;
        }
    }

    if !low_rank_scalar_cost_is_profitable(dimension, rank)
        || !low_rank_factor_reconstructs_effective_kernel(
            &effective, dimension, rank, &factors, tolerance,
        )
    {
        return None;
    }
    Some(RuntimeSymmetricGroupLowRankBlock {
        rank,
        factors: factors.into_boxed_slice(),
    })
}

fn low_rank_scalar_cost_is_profitable(dimension: usize, rank: usize) -> bool {
    let dimension = dimension as u128;
    let rank = rank as u128;
    let factor_cost = dimension * rank * (dimension + 1);
    let dense_cost = dimension * (dimension + 1) * (dimension + 1) / 2;
    factor_cost * 10 <= dense_cost * 9
}

fn compact_scalar_same_channel_blocks(
    blocks: Vec<Option<RuntimeSymmetricGroupLowRankBlock>>,
) -> Box<[Option<RuntimeSymmetricGroupLowRankBlock>]> {
    if blocks.iter().all(Option::is_none) {
        Box::default()
    } else {
        blocks.into_boxed_slice()
    }
}

fn low_rank_factor_reconstructs_effective_kernel(
    effective: &[f64],
    dimension: usize,
    rank: usize,
    factors: &[f64],
    pivot_tolerance: f64,
) -> bool {
    if effective.len() != dimension.saturating_mul(dimension)
        || factors.len() != dimension.saturating_mul(rank)
    {
        return false;
    }
    if !pivot_tolerance.is_finite() {
        return false;
    }
    for column in 0..dimension {
        for row in 0..dimension {
            let mut reconstructed = 0.0;
            for mode in 0..rank {
                reconstructed = factors[mode * dimension + row]
                    .mul_add(factors[mode * dimension + column], reconstructed);
            }
            let residual = effective[column * dimension + row] - reconstructed;
            if !residual.is_finite() || residual.abs() > pivot_tolerance {
                return false;
            }
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn contract_real_hermitian_same_channel_block_opportunistic(
    amplitudes: &[SymmetricGroupComplex64],
    kernel: &[f64],
    low_rank: Option<&RuntimeSymmetricGroupLowRankBlock>,
    dimension: usize,
    coefficient_offset: usize,
    lane_capacity: usize,
    lane_count: usize,
    weight: f64,
    reduced: &mut [f64],
) {
    if lane_count == 1
        && let Some(low_rank) = low_rank
        && low_rank.factors.len() == dimension.saturating_mul(low_rank.rank)
    {
        contract_real_hermitian_same_channel_low_rank_block_scalar(
            amplitudes,
            low_rank,
            dimension,
            coefficient_offset,
            lane_capacity,
            weight,
            reduced,
        );
        return;
    }
    contract_real_hermitian_same_channel_block(
        amplitudes,
        kernel,
        dimension,
        coefficient_offset,
        lane_capacity,
        lane_count,
        weight,
        reduced,
    );
}

#[allow(clippy::too_many_arguments)]
fn contract_real_hermitian_same_channel_low_rank_block_scalar(
    amplitudes: &[SymmetricGroupComplex64],
    low_rank: &RuntimeSymmetricGroupLowRankBlock,
    dimension: usize,
    coefficient_offset: usize,
    lane_capacity: usize,
    weight: f64,
    reduced: &mut [f64],
) {
    let mut norm = 0.0;
    for row in 0..dimension {
        for mode_start in (0..low_rank.rank).step_by(LOW_RANK_MODE_BLOCK) {
            let mode_width = (low_rank.rank - mode_start).min(LOW_RANK_MODE_BLOCK);
            let mut projected_re = [0.0; LOW_RANK_MODE_BLOCK];
            let mut projected_im = [0.0; LOW_RANK_MODE_BLOCK];
            for column in 0..dimension {
                let amplitude =
                    amplitudes[(coefficient_offset + column * dimension + row) * lane_capacity];
                for mode_local in 0..mode_width {
                    let factor = low_rank.factors[(mode_start + mode_local) * dimension + column];
                    projected_re[mode_local] =
                        factor.mul_add(amplitude.0, projected_re[mode_local]);
                    projected_im[mode_local] =
                        factor.mul_add(amplitude.1, projected_im[mode_local]);
                }
            }
            for mode_local in 0..mode_width {
                norm = projected_re[mode_local].mul_add(
                    projected_re[mode_local],
                    projected_im[mode_local].mul_add(projected_im[mode_local], norm),
                );
            }
        }
    }
    reduced[0] = weight.mul_add(norm, reduced[0]);
}

#[allow(clippy::too_many_arguments)]
fn contract_real_hermitian_same_channel_block(
    amplitudes: &[SymmetricGroupComplex64],
    kernel: &[f64],
    dimension: usize,
    coefficient_offset: usize,
    lane_capacity: usize,
    lane_count: usize,
    weight: f64,
    reduced: &mut [f64],
) {
    // Diagonal channel kernels are authenticated at load as real and
    // inversion-Hermitian.  Their Young blocks are therefore real symmetric,
    // while Re(A_c^dagger A_i) is symmetric even in the presence of harmless
    // transform roundoff.  Pairing K_ci and K_ic preserves the previous dense
    // arithmetic for either representation and evaluates each overlap once.
    if lane_count == 1 {
        contract_real_hermitian_same_channel_block_scalar(
            amplitudes,
            kernel,
            dimension,
            coefficient_offset,
            lane_capacity,
            weight,
            reduced,
        );
        return;
    }

    for column in 0..dimension {
        let diagonal_kernel = kernel[column * dimension + column] * weight;
        if diagonal_kernel != 0.0 {
            for row in 0..dimension {
                let value = (coefficient_offset + column * dimension + row) * lane_capacity;
                for lane in 0..lane_count {
                    let amplitude = amplitudes[value + lane];
                    let norm = amplitude.0.mul_add(amplitude.0, amplitude.1 * amplitude.1);
                    reduced[lane] = diagonal_kernel.mul_add(norm, reduced[lane]);
                }
            }
        }

        for inner in column + 1..dimension {
            let paired_kernel =
                (kernel[column * dimension + inner] + kernel[inner * dimension + column]) * weight;
            if paired_kernel == 0.0 {
                continue;
            }
            for row in 0..dimension {
                let left = (coefficient_offset + column * dimension + row) * lane_capacity;
                let right = (coefficient_offset + inner * dimension + row) * lane_capacity;
                for lane in 0..lane_count {
                    let left_value = amplitudes[left + lane];
                    let right_value = amplitudes[right + lane];
                    let product_re = left_value
                        .0
                        .mul_add(right_value.0, left_value.1 * right_value.1);
                    reduced[lane] = paired_kernel.mul_add(product_re, reduced[lane]);
                }
            }
        }
    }
}

const HERMITIAN_COLUMN_BLOCK: usize = 4;

#[allow(clippy::too_many_arguments)]
fn contract_real_hermitian_same_channel_block_scalar(
    amplitudes: &[SymmetricGroupComplex64],
    kernel: &[f64],
    dimension: usize,
    coefficient_offset: usize,
    lane_capacity: usize,
    weight: f64,
    reduced: &mut [f64],
) {
    let mut result = reduced[0];
    for left_start in (0..dimension).step_by(HERMITIAN_COLUMN_BLOCK) {
        let left_width = (dimension - left_start).min(HERMITIAN_COLUMN_BLOCK);
        for right_start in (left_start..dimension).step_by(HERMITIAN_COLUMN_BLOCK) {
            let right_width = (dimension - right_start).min(HERMITIAN_COLUMN_BLOCK);
            let diagonal_block = left_start == right_start;
            let mut overlaps = [[0.0_f64; HERMITIAN_COLUMN_BLOCK]; HERMITIAN_COLUMN_BLOCK];

            // Four simultaneous unit-stride column streams reuse each loaded
            // amplitude for all overlaps in the opposing four-column block.
            for row in 0..dimension {
                let mut left_values = [(0.0, 0.0); HERMITIAN_COLUMN_BLOCK];
                for (local, value) in left_values[..left_width].iter_mut().enumerate() {
                    let column = left_start + local;
                    *value =
                        amplitudes[(coefficient_offset + column * dimension + row) * lane_capacity];
                }
                let mut right_values = [(0.0, 0.0); HERMITIAN_COLUMN_BLOCK];
                if diagonal_block {
                    right_values = left_values;
                } else {
                    for (local, value) in right_values[..right_width].iter_mut().enumerate() {
                        let column = right_start + local;
                        *value = amplitudes
                            [(coefficient_offset + column * dimension + row) * lane_capacity];
                    }
                }

                for left_local in 0..left_width {
                    let first_right = if diagonal_block { left_local } else { 0 };
                    let left_value = left_values[left_local];
                    for right_local in first_right..right_width {
                        let right_value = right_values[right_local];
                        overlaps[left_local][right_local] = left_value.0.mul_add(
                            right_value.0,
                            left_value
                                .1
                                .mul_add(right_value.1, overlaps[left_local][right_local]),
                        );
                    }
                }
            }

            for (left_local, overlap_row) in overlaps[..left_width].iter().enumerate() {
                let column = left_start + left_local;
                let first_right = if diagonal_block { left_local } else { 0 };
                for (right_local, overlap) in overlap_row[first_right..right_width]
                    .iter()
                    .copied()
                    .enumerate()
                {
                    let inner_local = first_right + right_local;
                    let inner = right_start + inner_local;
                    let kernel_value = if column == inner {
                        kernel[column * dimension + column]
                    } else {
                        kernel[column * dimension + inner] + kernel[inner * dimension + column]
                    } * weight;
                    result = kernel_value.mul_add(overlap, result);
                }
            }
        }
    }
    reduced[0] = result;
}

#[allow(clippy::too_many_arguments)]
fn contract_general_cross_block(
    left_amplitudes: &[SymmetricGroupComplex64],
    right_amplitudes: &[SymmetricGroupComplex64],
    kernel: &[f64],
    dimension: usize,
    coefficient_offset: usize,
    lane_capacity: usize,
    lane_count: usize,
    weight: f64,
    reduced: &mut [f64],
) {
    for column in 0..dimension {
        for inner in 0..dimension {
            let kernel_value = kernel[column * dimension + inner] * weight;
            for row in 0..dimension {
                let left = (coefficient_offset + column * dimension + row) * lane_capacity;
                let right = (coefficient_offset + inner * dimension + row) * lane_capacity;
                for lane in 0..lane_count {
                    let left_value = left_amplitudes[left + lane];
                    let right_value = right_amplitudes[right + lane];
                    let product_re = left_value
                        .0
                        .mul_add(right_value.0, left_value.1 * right_value.1);
                    reduced[lane] = kernel_value.mul_add(product_re, reduced[lane]);
                }
            }
        }
    }
}

/// Runtime reduction shape.  The symmetric-group variant deliberately owns
/// only authenticated raw kernels in codec phase one; lane integration adds a
/// shared immutable FFT plan without changing the payload or public C ABI.
#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeColorContractionReducer {
    Walsh(RuntimeFactorizedColorContraction),
    SymmetricGroupFourier(RuntimeSymmetricGroupColorContraction),
}

impl RuntimeFactorizedColorContraction {
    pub fn subgroup_order(&self) -> usize {
        self.subgroup_order
    }

    pub fn cosets(&self) -> &[Vec<u32>] {
        &self.cosets
    }

    pub fn entries(&self) -> &[RuntimeFactorizedColorContractionEntry] {
        &self.entries
    }

    pub fn amplitude_scale(&self) -> f64 {
        self.amplitude_scale
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecurrenceColorContraction {
    accuracy: RecurrenceColorAccuracy,
    storage: RecurrenceColorStorage,
    includes_color_factor: bool,
    group_count: u32,
    sector_count: u32,
    component_count: u32,
    local_group_count: u32,
    destination_count: u32,
    stored_entry_count: usize,
    stored_logical_entry_count: usize,
    entries: Vec<RawColorContractionEntry>,
    exact_factors: Vec<ExactComplexRational>,
    ordered_group_ids: Vec<u32>,
    destination_by_group: Vec<u32>,
    sector_by_group: Vec<u32>,
    component_by_group: Vec<u32>,
    owner_by_sector: Vec<u32>,
    ordered_destination_ids: Vec<u32>,
    factorization: Option<FactorizedColorContraction>,
    runtime_reducer: Option<RuntimeColorContractionReducer>,
}

impl RecurrenceColorContraction {
    pub fn accuracy(&self) -> RecurrenceColorAccuracy {
        self.accuracy
    }

    pub fn storage(&self) -> RecurrenceColorStorage {
        self.storage
    }

    pub fn includes_color_factor(&self) -> bool {
        self.includes_color_factor
    }

    pub fn group_count(&self) -> u32 {
        self.group_count
    }

    pub fn sector_count(&self) -> u32 {
        self.sector_count
    }

    pub fn component_count(&self) -> u32 {
        self.component_count
    }

    pub fn local_group_count(&self) -> u32 {
        self.local_group_count
    }

    pub fn destination_count(&self) -> u32 {
        self.destination_count
    }

    pub fn entries(&self) -> &[RawColorContractionEntry] {
        &self.entries
    }

    pub(crate) fn stored_entry_count(&self) -> usize {
        self.stored_entry_count
    }

    pub fn exact_factors(&self) -> &[ExactComplexRational] {
        &self.exact_factors
    }

    pub fn ordered_group_ids(&self) -> &[u32] {
        &self.ordered_group_ids
    }

    pub fn destination_by_group(&self) -> &[u32] {
        &self.destination_by_group
    }

    pub fn sector_by_group(&self) -> &[u32] {
        &self.sector_by_group
    }

    pub fn component_by_group(&self) -> &[u32] {
        &self.component_by_group
    }

    pub fn owner_by_sector(&self) -> &[u32] {
        &self.owner_by_sector
    }

    pub fn active_sector_count(&self) -> usize {
        self.owner_by_sector
            .iter()
            .enumerate()
            .filter(|(sector, owner)| *sector == **owner as usize)
            .count()
    }

    pub fn factorization(&self) -> Option<&FactorizedColorContraction> {
        self.factorization.as_ref()
    }

    pub fn runtime_factorization(&self) -> Option<&RuntimeFactorizedColorContraction> {
        match self.runtime_reducer.as_ref() {
            Some(RuntimeColorContractionReducer::Walsh(value)) => Some(value),
            _ => None,
        }
    }

    pub fn runtime_reducer(&self) -> Option<&RuntimeColorContractionReducer> {
        self.runtime_reducer.as_ref()
    }

    pub fn ordered_destination_id(
        &self,
        local_group_index: usize,
        component_index: usize,
    ) -> Option<u32> {
        let ordered_index = local_group_index
            .checked_mul(self.component_count as usize)?
            .checked_add(component_index)?;
        self.ordered_destination_ids.get(ordered_index).copied()
    }

    pub fn logical_entry_count(&self) -> usize {
        self.stored_logical_entry_count
    }

    /// Iterate canonical logical entries without allocating.
    pub fn canonical_logical_entries(&self) -> CanonicalColorContractionEntries<'_> {
        assert_ne!(
            self.storage,
            RecurrenceColorStorage::ConvolutionKernels,
            "convolution-kernel storage is not a dense logical color matrix; use its runtime reducer",
        );
        CanonicalColorContractionEntries {
            plan: self,
            next_index: 0,
            entry_count: self.logical_entry_count(),
        }
    }

    /// Iterate symmetry-folded Direct-Arena rows without allocating.
    pub fn runtime_entries(&self) -> RuntimeColorContractionEntries<'_> {
        RuntimeColorContractionEntries {
            inner: self.canonical_logical_entries(),
        }
    }

    #[cfg(test)]
    pub(crate) fn expanded_identity_for_runtime_test() -> Self {
        Self {
            accuracy: RecurrenceColorAccuracy::Full,
            storage: RecurrenceColorStorage::Expanded,
            includes_color_factor: true,
            group_count: 1,
            sector_count: 1,
            component_count: 1,
            local_group_count: 1,
            destination_count: 1,
            stored_entry_count: 1,
            stored_logical_entry_count: 1,
            entries: vec![RawColorContractionEntry {
                left_group_id: 0,
                right_group_id: 0,
                weight_re: 1.0,
                weight_im: 0.0,
                symmetry_factor: 1.0,
                exact_factor_id: 0,
            }],
            exact_factors: vec![ExactComplexRational::ONE],
            ordered_group_ids: vec![0],
            destination_by_group: vec![0],
            sector_by_group: vec![0],
            component_by_group: vec![0],
            owner_by_sector: vec![0],
            ordered_destination_ids: vec![0],
            factorization: None,
            runtime_reducer: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn symmetric_group_s3_for_runtime_test(
        destination_by_group: Vec<u32>,
        destination_count: u32,
    ) -> Self {
        assert_eq!(destination_by_group.len(), 13);
        assert!(
            destination_by_group
                .iter()
                .all(|destination| *destination < destination_count)
        );
        let diagonal_left = [6.0, 1.0, 2.0, 3.0, 3.0, 4.0];
        let cross = [1.0, 2.0, 0.0, -1.0, 0.5, 3.0];
        let diagonal_right = [7.0, 0.0, 2.0, -1.0, -1.0, 0.0];
        let mut entries = Vec::new();
        let mut push = |left_group_id, right_group_id, weight_re, symmetry_factor| {
            entries.push(RawColorContractionEntry {
                left_group_id,
                right_group_id,
                weight_re,
                weight_im: 0.0,
                symmetry_factor,
                exact_factor_id: 0,
            });
        };
        for (relative, weight) in diagonal_left.into_iter().enumerate() {
            push(0, relative as u32, weight, 1.0);
        }
        for (relative, weight) in cross.into_iter().enumerate() {
            push(0, 6 + relative as u32, weight, 2.0);
        }
        for (relative, weight) in diagonal_right.into_iter().enumerate() {
            push(6, 6 + relative as u32, weight, 1.0);
        }
        push(0, 12, 0.25, 2.0);
        for left in 1..12 {
            push(left, 12, 0.0, 2.0);
        }
        push(12, 12, 5.0, 1.0);
        let factorization = FactorizedColorContraction {
            kind: FactorizedColorContractionKind::SymmetricGroupFourier,
            rank: 3,
            coset_count: 2,
            coset_indices: (0..12).collect(),
        };
        let runtime = build_runtime_symmetric_group_convolution(&factorization, 13, &entries)
            .expect("valid symmetric-group runtime fixture");
        let stored_entry_count = entries.len();
        let ordered_group_ids = (0..13).collect::<Vec<_>>();
        let ordered_destination_ids = destination_by_group.clone();
        Self {
            accuracy: RecurrenceColorAccuracy::Full,
            storage: RecurrenceColorStorage::ConvolutionKernels,
            includes_color_factor: true,
            group_count: 13,
            sector_count: 13,
            component_count: 1,
            local_group_count: 13,
            destination_count,
            stored_entry_count,
            stored_logical_entry_count: stored_entry_count,
            entries: Vec::new(),
            exact_factors: Vec::new(),
            ordered_group_ids,
            destination_by_group,
            sector_by_group: (0..13).collect(),
            component_by_group: vec![0; 13],
            owner_by_sector: (0..13).collect(),
            ordered_destination_ids,
            factorization: Some(factorization),
            runtime_reducer: Some(RuntimeColorContractionReducer::SymmetricGroupFourier(
                runtime,
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_sparse_sector_domain_for_runtime_test(mut self) -> Self {
        self.sector_count = 27;
        self.sector_by_group = self
            .sector_by_group
            .iter()
            .map(|sector| 2 * sector + 1)
            .collect();
        self.owner_by_sector = vec![ZERO_SECTOR_OWNER; self.sector_count as usize];
        for sector in &self.sector_by_group {
            self.owner_by_sector[*sector as usize] = *sector;
        }
        self
    }

    /// Evaluate the complete dense quadratic form owned by the synthetic S3
    /// runtime fixture without entering the symmetric-group reducer.
    ///
    /// The input is group-major and lane-minor.  This deliberately retains the
    /// original convolution kernels, both cross-channel orientations, and the
    /// eligible/residual plus residual/residual rows so native lane tests have
    /// an oracle independent of the transformed execution path.
    #[cfg(test)]
    pub(crate) fn symmetric_group_s3_dense_for_runtime_test(
        amplitudes: &[(f64, f64)],
        lane_count: usize,
    ) -> Vec<f64> {
        assert!(lane_count > 0);
        assert_eq!(amplitudes.len(), 13 * lane_count);

        let diagonal_left = [6.0, 1.0, 2.0, 3.0, 3.0, 4.0];
        let cross = [1.0, 2.0, 0.0, -1.0, 0.5, 3.0];
        let diagonal_right = [7.0, 0.0, 2.0, -1.0, -1.0, 0.0];
        let kernels = [&diagonal_left[..], &cross[..], &diagonal_right[..]];
        let channel_pairs = [(0usize, 0usize), (0, 1), (1, 1)];
        let permutations = [
            [0usize, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut expected = vec![0.0; lane_count];
        for lane in 0..lane_count {
            for ((left_channel, right_channel), kernel) in channel_pairs.into_iter().zip(kernels) {
                let mut value = (0.0, 0.0);
                for left_relative in 0..6 {
                    let mut inverse_left = [0usize; 3];
                    for (position, image) in permutations[left_relative].iter().copied().enumerate()
                    {
                        inverse_left[image] = position;
                    }
                    for right_relative in 0..6 {
                        let relative =
                            permutations[right_relative].map(|image| inverse_left[image]);
                        let relative_index = permutations
                            .iter()
                            .position(|candidate| *candidate == relative)
                            .unwrap();
                        let left =
                            amplitudes[(left_channel * 6 + left_relative) * lane_count + lane];
                        let right =
                            amplitudes[(right_channel * 6 + right_relative) * lane_count + lane];
                        let product_re = left.0.mul_add(right.0, left.1 * right.1);
                        let product_im = left.0.mul_add(right.1, -left.1 * right.0);
                        value.0 += kernel[relative_index] * product_re;
                        value.1 += kernel[relative_index] * product_im;
                    }
                }
                expected[lane] += if left_channel == right_channel {
                    value.0
                } else {
                    2.0 * value.0
                };
            }

            // The fixture's residual suffix contains group 12.  Its complete
            // direct rows are 2 * 0.25 * Re(A0 conj(A12)) plus
            // 5 * |A12|^2; the intervening cross rows are exact zeros.
            let first = amplitudes[lane];
            let residual = amplitudes[12 * lane_count + lane];
            expected[lane] += 0.5 * first.0.mul_add(residual.0, first.1 * residual.1);
            expected[lane] += 5.0 * residual.0.mul_add(residual.0, residual.1 * residual.1);
        }
        expected
    }
}

pub struct CanonicalColorContractionEntries<'a> {
    plan: &'a RecurrenceColorContraction,
    next_index: usize,
    entry_count: usize,
}

impl Iterator for CanonicalColorContractionEntries<'_> {
    type Item = CanonicalColorContractionEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.entry_count {
            return None;
        }
        let logical_index = self.next_index;
        self.next_index += 1;
        let (entry, left_group_id, right_group_id) = match self.plan.storage {
            RecurrenceColorStorage::Expanded => {
                let entry = *self.plan.entries.get(logical_index)?;
                (entry, entry.left_group_id, entry.right_group_id)
            }
            RecurrenceColorStorage::Repeated | RecurrenceColorStorage::ConvolutionKernels => {
                let template_count = self.plan.entries.len();
                if template_count == 0 {
                    return None;
                }
                let component_index = logical_index / template_count;
                let entry = *self.plan.entries.get(logical_index % template_count)?;
                let component_count = self.plan.component_count as usize;
                let left_index = entry.left_group_id as usize * component_count + component_index;
                let right_index = entry.right_group_id as usize * component_count + component_index;
                (
                    entry,
                    *self.plan.ordered_group_ids.get(left_index)?,
                    *self.plan.ordered_group_ids.get(right_index)?,
                )
            }
        };
        Some(CanonicalColorContractionEntry {
            left_group_id,
            right_group_id,
            left_destination_id: self.plan.destination_by_group[left_group_id as usize],
            right_destination_id: self.plan.destination_by_group[right_group_id as usize],
            weight_re: entry.weight_re,
            weight_im: entry.weight_im,
            symmetry_factor: entry.symmetry_factor,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entry_count.saturating_sub(self.next_index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CanonicalColorContractionEntries<'_> {}
impl FusedIterator for CanonicalColorContractionEntries<'_> {}

pub struct RuntimeColorContractionEntries<'a> {
    inner: CanonicalColorContractionEntries<'a>,
}

impl Iterator for RuntimeColorContractionEntries<'_> {
    type Item = RuntimeColorContractionEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let raw = self.inner.next()?;
        Some(RuntimeColorContractionEntry {
            left_destination_id: raw.left_destination_id,
            right_destination_id: raw.right_destination_id,
            coefficient_re: raw.weight_re * raw.symmetry_factor,
            coefficient_im: raw.weight_im * raw.symmetry_factor,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for RuntimeColorContractionEntries<'_> {}
impl FusedIterator for RuntimeColorContractionEntries<'_> {}

/// Decode and fully validate a caller-authenticated process payload.
pub fn decode_recurrence_color_contraction_v3(
    bytes: &[u8],
) -> RusticolResult<RecurrenceColorContraction> {
    let mut reader = Reader::new(bytes);
    if reader.take(8, "magic")? != MAGIC {
        return Err(malformed("invalid payload magic"));
    }
    if reader.u32("version")? != VERSION {
        return Err(malformed("unsupported payload version"));
    }
    if reader.u32("header size")? as usize != HEADER_BYTES {
        return Err(malformed("header size does not match codec v3"));
    }
    let storage = RecurrenceColorStorage::decode(reader.u32("storage")?)?;
    let accuracy = RecurrenceColorAccuracy::decode(reader.u32("color accuracy")?)?;
    let flags = reader.u32("flags")?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(malformed("payload declares unknown flags"));
    }
    let group_count = reader.u32("group count")?;
    let sector_count = reader.u32("sector count")?;
    let component_count = reader.u32("component count")?;
    let local_group_count = reader.u32("local group count")?;
    let destination_count = reader.u32("destination count")?;
    let factor_kind = reader.u32("factorization kind")?;
    let factor_rank = reader.u32("factorization rank")?;
    if reader.u32("entry stride")? as usize != ENTRY_BYTES {
        return Err(malformed("entry stride does not match codec v3"));
    }
    if reader.u32("exact factor stride")? as usize != EXACT_FACTOR_BYTES {
        return Err(malformed("exact factor stride does not match codec v3"));
    }
    let entry_count = reader.count("entry count")?;
    let exact_factor_count = reader.count("exact factor count")?;
    let coset_count = reader.count("coset count")?;
    let coset_index_count = reader.count("coset index count")?;
    let declared_logical_entry_count = reader.count("logical entry count")?;
    let owner_map_count = reader.count("physical sector owner map count")?;
    let payload_bytes = reader.count("payload byte count")?;

    if group_count == 0 || sector_count == 0 || component_count == 0 || destination_count == 0 {
        return Err(malformed(
            "group, sector, component, and destination counts must be positive",
        ));
    }
    if owner_map_count != sector_count as usize {
        return Err(malformed(
            "physical sector owner map count does not match sector_count",
        ));
    }
    let expected_payload_bytes = entry_count
        .checked_mul(ENTRY_BYTES)
        .and_then(|value| value.checked_add(exact_factor_count.checked_mul(EXACT_FACTOR_BYTES)?))
        .and_then(|value| value.checked_add((group_count as usize).checked_mul(16)?))
        .and_then(|value| value.checked_add(owner_map_count.checked_mul(4)?))
        .and_then(|value| value.checked_add(coset_index_count.checked_mul(4)?))
        .ok_or_else(|| malformed("payload byte count overflows usize"))?;
    if payload_bytes != expected_payload_bytes
        || Some(bytes.len()) != HEADER_BYTES.checked_add(payload_bytes)
    {
        return Err(malformed(
            "declared payload size does not match its fixed-width sections",
        ));
    }

    let entry_domain = match storage {
        RecurrenceColorStorage::Expanded => group_count,
        RecurrenceColorStorage::Repeated | RecurrenceColorStorage::ConvolutionKernels => {
            local_group_count
        }
    };
    let mut entries = Vec::with_capacity(entry_count);
    let mut seen_entry_pairs = BTreeSet::new();
    for index in 0..entry_count {
        let entry = RawColorContractionEntry {
            left_group_id: reader.u32("entry left group ID")?,
            right_group_id: reader.u32("entry right group ID")?,
            weight_re: reader.f64("entry real weight")?,
            weight_im: reader.f64("entry imaginary weight")?,
            symmetry_factor: reader.f64("entry symmetry factor")?,
            exact_factor_id: reader.u32("entry exact factor ID")?,
        };
        validate_raw_entry(index, entry, entry_domain, &mut seen_entry_pairs)?;
        entries.push(entry);
    }
    let mut exact_factors = Vec::with_capacity(exact_factor_count);
    for index in 0..exact_factor_count {
        let factor = ExactComplexRational::new(
            ExactRational::new(
                reader.i128("exact real numerator")?,
                reader.i128("exact real denominator")?,
            )
            .map_err(|error| malformed(format!("exact color factor {index} real part: {error}")))?,
            ExactRational::new(
                reader.i128("exact imaginary numerator")?,
                reader.i128("exact imaginary denominator")?,
            )
            .map_err(|error| {
                malformed(format!(
                    "exact color factor {index} imaginary part: {error}"
                ))
            })?,
        );
        exact_factors.push(factor);
    }
    if entries
        .iter()
        .any(|entry| entry.exact_factor_id as usize >= exact_factors.len())
    {
        return Err(malformed(
            "entry references an out-of-bounds exact color factor",
        ));
    }
    for (index, entry) in entries.iter().copied().enumerate() {
        validate_exact_matches_f64(index, entry, exact_factors[entry.exact_factor_id as usize])?;
    }
    let ordered_group_ids = reader.u32_vec(group_count as usize, "ordered group ID")?;
    let destination_by_group =
        reader.u32_vec(group_count as usize, "Direct-Arena destination ID")?;
    let sector_by_group = reader.u32_vec(group_count as usize, "group physical-sector ID")?;
    let component_by_group = reader.u32_vec(group_count as usize, "group resolved-helicity ID")?;
    let owner_by_sector = reader.u32_vec(owner_map_count, "physical sector owner ID")?;
    let coset_indices = reader.u32_vec(coset_index_count, "factorization coset index")?;
    if !reader.is_finished() {
        return Err(malformed("payload contains trailing bytes"));
    }

    validate_permutation(&ordered_group_ids, group_count, "ordered group map")?;
    validate_destination_map(&destination_by_group, destination_count)?;
    validate_group_identities(
        &sector_by_group,
        &component_by_group,
        sector_count,
        component_count,
    )?;
    validate_sector_owners(&owner_by_sector, &sector_by_group, sector_count)?;

    let expected_logical_entry_count = match storage {
        RecurrenceColorStorage::Expanded => {
            if local_group_count != 0
                || factor_kind != 0
                || factor_rank != 0
                || coset_count != 0
                || coset_index_count != 0
            {
                return Err(malformed(
                    "expanded storage is mixed with repeated/factorized fields",
                ));
            }
            validate_expanded_components(&entries, &component_by_group)?;
            entry_count
        }
        RecurrenceColorStorage::Repeated => {
            if component_count < 2 {
                return Err(malformed(
                    "repeated storage requires at least two components",
                ));
            }
            if local_group_count == 0
                || local_group_count.checked_mul(component_count) != Some(group_count)
            {
                return Err(malformed(
                    "repeated local, sector, component, and group counts are inconsistent",
                ));
            }
            validate_repeated_group_identities(
                &ordered_group_ids,
                &sector_by_group,
                &component_by_group,
                local_group_count,
                component_count,
            )?;
            entry_count
                .checked_mul(component_count as usize)
                .ok_or_else(|| malformed("logical entry count overflows usize"))?
        }
        RecurrenceColorStorage::ConvolutionKernels => {
            if local_group_count == 0
                || local_group_count.checked_mul(component_count) != Some(group_count)
            {
                return Err(malformed(
                    "convolution-kernel local, component, and group counts are inconsistent",
                ));
            }
            validate_repeated_group_identities(
                &ordered_group_ids,
                &sector_by_group,
                &component_by_group,
                local_group_count,
                component_count,
            )?;
            entry_count
                .checked_mul(component_count as usize)
                .ok_or_else(|| malformed("logical entry count overflows usize"))?
        }
    };
    if declared_logical_entry_count != expected_logical_entry_count {
        return Err(malformed(
            "declared logical entry count is inconsistent with storage",
        ));
    }

    let factorization = decode_factorization(
        storage,
        factor_kind,
        factor_rank,
        coset_count,
        coset_indices,
        local_group_count,
        &entries,
    )?;
    let runtime_reducer = factorization
        .as_ref()
        .map(|factorization| match factorization.kind() {
            FactorizedColorContractionKind::KleinFourWalsh
            | FactorizedColorContractionKind::ElementaryAbelianWalsh => {
                build_runtime_factorization(factorization, local_group_count, &entries)
                    .map(RuntimeColorContractionReducer::Walsh)
            }
            FactorizedColorContractionKind::SymmetricGroupFourier => {
                build_runtime_symmetric_group_convolution(
                    factorization,
                    local_group_count,
                    &entries,
                )
                .map(RuntimeColorContractionReducer::SymmetricGroupFourier)
            }
        })
        .transpose()?;
    let ordered_destination_ids = ordered_group_ids
        .iter()
        .map(|group_id| destination_by_group[*group_id as usize])
        .collect();

    let retain_wire_catalog = storage != RecurrenceColorStorage::ConvolutionKernels;
    Ok(RecurrenceColorContraction {
        accuracy,
        storage,
        includes_color_factor: flags & FLAG_INCLUDES_COLOR_FACTOR != 0,
        group_count,
        sector_count,
        component_count,
        local_group_count,
        destination_count,
        stored_entry_count: entry_count,
        stored_logical_entry_count: declared_logical_entry_count,
        entries: if retain_wire_catalog {
            entries
        } else {
            Default::default()
        },
        exact_factors: if retain_wire_catalog {
            exact_factors
        } else {
            Default::default()
        },
        ordered_group_ids,
        destination_by_group,
        sector_by_group,
        component_by_group,
        owner_by_sector,
        ordered_destination_ids,
        factorization,
        runtime_reducer,
    })
}

/// Return the caller-owned deterministic SHA-256 digest of canonical bytes.
pub fn recurrence_color_contraction_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn validate_raw_entry(
    index: usize,
    entry: RawColorContractionEntry,
    group_count: u32,
    seen_pairs: &mut BTreeSet<(u32, u32)>,
) -> RusticolResult<()> {
    if entry.left_group_id >= group_count || entry.right_group_id >= group_count {
        return Err(malformed(format!(
            "entry {index} references an out-of-bounds group"
        )));
    }
    if entry.left_group_id > entry.right_group_id {
        return Err(malformed(format!(
            "entry {index} is not canonical upper triangular"
        )));
    }
    if !seen_pairs.insert((entry.left_group_id, entry.right_group_id)) {
        return Err(malformed(format!(
            "entry {index} duplicates a canonical group pair"
        )));
    }
    if !entry.weight_re.is_finite()
        || !entry.weight_im.is_finite()
        || !entry.symmetry_factor.is_finite()
    {
        return Err(malformed(format!(
            "entry {index} contains a non-finite f64"
        )));
    }
    if !(entry.weight_re * entry.symmetry_factor).is_finite()
        || !(entry.weight_im * entry.symmetry_factor).is_finite()
    {
        return Err(malformed(format!(
            "entry {index} overflows after symmetry folding"
        )));
    }
    Ok(())
}

fn validate_exact_matches_f64(
    index: usize,
    entry: RawColorContractionEntry,
    factor: ExactComplexRational,
) -> RusticolResult<()> {
    let actual = [
        entry.weight_re * entry.symmetry_factor,
        entry.weight_im * entry.symmetry_factor,
    ];
    let expected = [
        factor.real().numerator() as f64 / factor.real().denominator() as f64,
        factor.imag().numerator() as f64 / factor.imag().denominator() as f64,
    ];
    for (component, actual, expected) in ["real", "imaginary"]
        .into_iter()
        .zip(actual)
        .zip(expected)
        .map(|((component, actual), expected)| (component, actual, expected))
    {
        let tolerance = f64_ulp(actual).max(f64_ulp(expected));
        if (actual - expected).abs() > tolerance {
            return Err(malformed(format!(
                "entry {index} {component} f64 coefficient disagrees with its exact color factor",
            )));
        }
    }
    Ok(())
}

fn f64_ulp(value: f64) -> f64 {
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let magnitude = value.abs();
    if !magnitude.is_finite() {
        return f64::INFINITY;
    }
    f64::from_bits(magnitude.to_bits() + 1) - magnitude
}

fn validate_permutation(values: &[u32], count: u32, label: &str) -> RusticolResult<()> {
    let mut seen = vec![false; count as usize];
    for value in values {
        let Some(slot) = seen.get_mut(*value as usize) else {
            return Err(malformed(format!("{label} contains an out-of-bounds ID")));
        };
        if *slot {
            return Err(malformed(format!("{label} contains a duplicate ID")));
        }
        *slot = true;
    }
    if seen.iter().any(|value| !value) {
        return Err(malformed(format!("{label} is not a complete permutation")));
    }
    Ok(())
}

fn validate_destination_map(values: &[u32], destination_count: u32) -> RusticolResult<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        if *value >= destination_count {
            return Err(malformed(
                "destination map references an out-of-bounds Direct-Arena destination",
            ));
        }
        if !seen.insert(*value) {
            return Err(malformed(
                "destination map contains a duplicate Direct-Arena destination",
            ));
        }
    }
    Ok(())
}

fn validate_expanded_components(
    entries: &[RawColorContractionEntry],
    component_by_group: &[u32],
) -> RusticolResult<()> {
    for entry in entries {
        if component_by_group[entry.left_group_id as usize]
            != component_by_group[entry.right_group_id as usize]
        {
            return Err(malformed(
                "expanded entry couples groups from different components",
            ));
        }
    }
    Ok(())
}

fn validate_group_identities(
    sector_by_group: &[u32],
    component_by_group: &[u32],
    sector_count: u32,
    component_count: u32,
) -> RusticolResult<()> {
    if sector_by_group.len() != component_by_group.len() {
        return Err(malformed("group identity maps have different lengths"));
    }
    let mut identities = BTreeSet::new();
    for (group_id, (&sector_id, &component_id)) in
        sector_by_group.iter().zip(component_by_group).enumerate()
    {
        if sector_id >= sector_count || component_id >= component_count {
            return Err(malformed(format!(
                "group {group_id} identity references an out-of-bounds sector or component"
            )));
        }
        if !identities.insert((sector_id, component_id)) {
            return Err(malformed(
                "group identity maps repeat a sector/component pair",
            ));
        }
    }
    Ok(())
}

fn validate_sector_owners(
    owner_by_sector: &[u32],
    sector_by_group: &[u32],
    sector_count: u32,
) -> RusticolResult<()> {
    if owner_by_sector.len() != sector_count as usize {
        return Err(malformed("physical sector owner map has the wrong length"));
    }
    let mut fixed_points = BTreeSet::new();
    for (sector_id, owner_id) in owner_by_sector.iter().copied().enumerate() {
        if owner_id == ZERO_SECTOR_OWNER {
            continue;
        }
        if owner_id >= sector_count
            || owner_id as usize > sector_id
            || owner_by_sector[owner_id as usize] != owner_id
        {
            return Err(malformed(format!(
                "physical sector {sector_id} has an invalid canonical owner",
            )));
        }
        if owner_id as usize == sector_id {
            fixed_points.insert(owner_id);
        }
    }
    let active = sector_by_group.iter().copied().collect::<BTreeSet<_>>();
    if active != fixed_points {
        return Err(malformed(
            "active recurrence sectors are not exactly the authenticated owner sectors",
        ));
    }
    Ok(())
}

fn validate_repeated_group_identities(
    ordered_group_ids: &[u32],
    sector_by_group: &[u32],
    component_by_group: &[u32],
    local_group_count: u32,
    component_count: u32,
) -> RusticolResult<()> {
    for local_group in 0..local_group_count as usize {
        let start = local_group
            .checked_mul(component_count as usize)
            .ok_or_else(|| malformed("repeated group identity offset overflows usize"))?;
        let stop = start
            .checked_add(component_count as usize)
            .ok_or_else(|| malformed("repeated group identity range overflows usize"))?;
        let group_ids = ordered_group_ids
            .get(start..stop)
            .ok_or_else(|| malformed("repeated group identity range is truncated"))?;
        let first = *group_ids
            .first()
            .ok_or_else(|| malformed("repeated group identity row is empty"))?;
        let sector_id = sector_by_group[first as usize];
        for (component_id, group_id) in group_ids.iter().copied().enumerate() {
            if sector_by_group[group_id as usize] != sector_id
                || component_by_group[group_id as usize] != component_id as u32
            {
                return Err(malformed(
                    "repeated group identities are not local-color-major/component-minor",
                ));
            }
        }
    }
    Ok(())
}

fn decode_factorization(
    storage: RecurrenceColorStorage,
    factor_kind: u32,
    factor_rank: u32,
    coset_count: usize,
    coset_indices: Vec<u32>,
    local_group_count: u32,
    entries: &[RawColorContractionEntry],
) -> RusticolResult<Option<FactorizedColorContraction>> {
    if factor_kind == 0 {
        if factor_rank != 0 || coset_count != 0 || !coset_indices.is_empty() {
            return Err(malformed(
                "factorization-none carries rank or coset metadata",
            ));
        }
        if storage == RecurrenceColorStorage::ConvolutionKernels {
            return Err(malformed(
                "convolution-kernel storage requires symmetric-group Fourier metadata",
            ));
        }
        return Ok(None);
    }
    let kind = match factor_kind {
        1 => {
            if storage != RecurrenceColorStorage::Repeated {
                return Err(malformed(
                    "Walsh factorization requires repeated color storage",
                ));
            }
            if factor_rank != 2 {
                return Err(malformed("Klein-four factorization must have rank two"));
            }
            FactorizedColorContractionKind::KleinFourWalsh
        }
        2 => {
            if storage != RecurrenceColorStorage::Repeated {
                return Err(malformed(
                    "Walsh factorization requires repeated color storage",
                ));
            }
            if !(3..=MAX_FACTOR_RANK).contains(&factor_rank) {
                return Err(malformed(format!(
                    "elementary-Abelian factorization rank must be in [3, {MAX_FACTOR_RANK}]"
                )));
            }
            FactorizedColorContractionKind::ElementaryAbelianWalsh
        }
        3 => {
            if storage != RecurrenceColorStorage::ConvolutionKernels {
                return Err(malformed(
                    "symmetric-group Fourier factorization requires convolution-kernel storage",
                ));
            }
            if !(2..=MAX_SYMMETRIC_GROUP_DEGREE).contains(&factor_rank) {
                return Err(malformed(format!(
                    "symmetric-group Fourier degree must be in [2, {MAX_SYMMETRIC_GROUP_DEGREE}]"
                )));
            }
            FactorizedColorContractionKind::SymmetricGroupFourier
        }
        _ => {
            return Err(malformed(format!(
                "unknown factorization discriminant {factor_kind}"
            )));
        }
    };
    let subgroup_order = match kind {
        FactorizedColorContractionKind::KleinFourWalsh
        | FactorizedColorContractionKind::ElementaryAbelianWalsh => 1usize
            .checked_shl(factor_rank)
            .ok_or_else(|| malformed("factorization subgroup order overflows usize"))?,
        FactorizedColorContractionKind::SymmetricGroupFourier => checked_factorial(factor_rank)
            .ok_or_else(|| malformed("symmetric-group order overflows usize"))?,
    };
    if coset_count == 0 || coset_count.checked_mul(subgroup_order) != Some(coset_indices.len()) {
        return Err(malformed(
            "factorization channel shape does not match its group order",
        ));
    }
    match kind {
        FactorizedColorContractionKind::KleinFourWalsh
        | FactorizedColorContractionKind::ElementaryAbelianWalsh => {
            if coset_indices.len() != local_group_count as usize {
                return Err(malformed(
                    "Walsh cosets do not cover every local color group",
                ));
            }
            validate_permutation(&coset_indices, local_group_count, "factorization coset map")?;
            validate_walsh_invariance(&coset_indices, coset_count, subgroup_order, entries)?;
        }
        FactorizedColorContractionKind::SymmetricGroupFourier => {
            validate_symmetric_group_convolution(
                factor_rank,
                &coset_indices,
                coset_count,
                subgroup_order,
                local_group_count,
                entries,
            )?;
        }
    }
    Ok(Some(FactorizedColorContraction {
        kind,
        rank: factor_rank,
        coset_count,
        coset_indices,
    }))
}

fn validate_walsh_invariance(
    coset_indices: &[u32],
    coset_count: usize,
    subgroup_order: usize,
    entries: &[RawColorContractionEntry],
) -> RusticolResult<()> {
    let mut matrix = BTreeMap::new();
    for entry in entries {
        if entry.weight_im != 0.0 {
            return Err(malformed(
                "factorized color contraction requires real weights",
            ));
        }
        let mut coefficient = entry.weight_re * entry.symmetry_factor;
        if entry.left_group_id != entry.right_group_id {
            coefficient *= 0.5;
        }
        if !coefficient.is_finite() {
            return Err(malformed(
                "factorized color contraction matrix coefficient is not finite",
            ));
        }
        matrix.insert((entry.left_group_id, entry.right_group_id), coefficient);
    }
    let matrix_value = |left: u32, right: u32| {
        let pair = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        matrix.get(&pair).copied().unwrap_or(0.0)
    };
    for left_coset_index in 0..coset_count {
        let left_start = left_coset_index * subgroup_order;
        let left_coset = &coset_indices[left_start..left_start + subgroup_order];
        for right_coset_index in 0..coset_count {
            let right_start = right_coset_index * subgroup_order;
            let right_coset = &coset_indices[right_start..right_start + subgroup_order];
            for left_index in 0..subgroup_order {
                for right_index in 0..subgroup_order {
                    let actual = matrix_value(left_coset[left_index], right_coset[right_index]);
                    let expected =
                        matrix_value(left_coset[0], right_coset[left_index ^ right_index]);
                    if actual != expected {
                        return Err(malformed(
                            "factorization cosets are inconsistent with the canonical color matrix",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_symmetric_group_convolution(
    _degree: u32,
    channel_indices: &[u32],
    channel_count: usize,
    group_order: usize,
    local_group_count: u32,
    entries: &[RawColorContractionEntry],
) -> RusticolResult<()> {
    let eligible_group_count = channel_count
        .checked_mul(group_order)
        .ok_or_else(|| malformed("symmetric-group eligible group count overflows usize"))?;
    if eligible_group_count > local_group_count as usize {
        return Err(malformed(
            "symmetric-group channels exceed the local group domain",
        ));
    }
    if channel_indices
        .iter()
        .copied()
        .enumerate()
        .any(|(expected, actual)| actual as usize != expected)
    {
        return Err(malformed(
            "symmetric-group channels are not canonical channel-major/permutation-major indices",
        ));
    }
    let channel_pair_count = channel_count
        .checked_mul(channel_count + 1)
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| malformed("symmetric-group channel-pair count overflows usize"))?;
    let kernel_entry_count = channel_pair_count
        .checked_mul(group_order)
        .ok_or_else(|| malformed("symmetric-group kernel entry count overflows usize"))?;
    if entries.len() < kernel_entry_count {
        return Err(malformed(
            "symmetric-group kernel rows do not cover every channel pair",
        ));
    }

    let mut offset = 0usize;
    for left_channel in 0..channel_count {
        let left_identity = (left_channel * group_order) as u32;
        for right_channel in left_channel..channel_count {
            let right_start = right_channel * group_order;
            let expected_symmetry = if left_channel == right_channel {
                1.0
            } else {
                2.0
            };
            for relative_index in 0..group_order {
                let entry = entries[offset];
                if entry.left_group_id != left_identity
                    || entry.right_group_id as usize != right_start + relative_index
                    || entry.symmetry_factor != expected_symmetry
                {
                    return Err(malformed(
                        "symmetric-group kernel rows are not canonical (channel, channel, relative-permutation) records",
                    ));
                }
                if entry.weight_im != 0.0 {
                    return Err(malformed(
                        "symmetric-group color kernel contains a complex coefficient",
                    ));
                }
                offset += 1;
            }
        }
    }

    let residual_entries = &entries[kernel_entry_count..];
    let total_pair_count = (local_group_count as usize)
        .checked_mul(local_group_count as usize + 1)
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| malformed("symmetric-group residual pair count overflows usize"))?;
    let eligible_pair_count = eligible_group_count
        .checked_mul(eligible_group_count + 1)
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| malformed("symmetric-group eligible pair count overflows usize"))?;
    let expected_residual_count = total_pair_count - eligible_pair_count;
    if residual_entries.len() != expected_residual_count {
        return Err(malformed(
            "symmetric-group residual rows do not cover every pair touching the residual suffix",
        ));
    }
    let mut residual_offset = 0usize;
    for left in 0..local_group_count {
        for right in left..local_group_count {
            if right < eligible_group_count as u32 {
                continue;
            }
            let entry = residual_entries[residual_offset];
            let pair = (left, right);
            if (entry.left_group_id, entry.right_group_id) != pair {
                return Err(malformed(
                    "symmetric-group residual rows are not the exhaustive canonical pair sequence",
                ));
            }
            residual_offset += 1;
            if entry.weight_im != 0.0 {
                return Err(malformed(
                    "symmetric-group residual row contains a complex coefficient",
                ));
            }
            let expected_symmetry = if pair.0 == pair.1 { 1.0 } else { 2.0 };
            if entry.symmetry_factor != expected_symmetry {
                return Err(malformed(
                    "symmetric-group residual row has a noncanonical symmetry factor",
                ));
            }
        }
    }
    debug_assert_eq!(residual_offset, residual_entries.len());
    Ok(())
}

fn checked_factorial(value: u32) -> Option<usize> {
    (2..=value).try_fold(1usize, |result, factor| result.checked_mul(factor as usize))
}

fn build_runtime_factorization(
    factorization: &FactorizedColorContraction,
    local_group_count: u32,
    entries: &[RawColorContractionEntry],
) -> RusticolResult<RuntimeFactorizedColorContraction> {
    let subgroup_order = factorization.subgroup_order();
    let cosets = (0..factorization.coset_count())
        .map(|index| {
            factorization
                .coset(index)
                .expect("validated factorization coset")
                .to_vec()
        })
        .collect::<Vec<_>>();
    let mut matrix = BTreeMap::new();
    for entry in entries {
        let mut coefficient = entry.weight_re * entry.symmetry_factor;
        if entry.left_group_id != entry.right_group_id {
            coefficient *= 0.5;
        }
        matrix.insert((entry.left_group_id, entry.right_group_id), coefficient);
    }
    let matrix_value = |left: u32, right: u32| {
        let pair = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        matrix.get(&pair).copied().unwrap_or(0.0)
    };

    let amplitude_scale = match factorization.kind() {
        FactorizedColorContractionKind::KleinFourWalsh => 0.5,
        FactorizedColorContractionKind::ElementaryAbelianWalsh => 1.0,
        FactorizedColorContractionKind::SymmetricGroupFourier => {
            return Err(malformed(
                "symmetric-group convolution cannot use the Walsh runtime builder",
            ));
        }
    };
    let weight_scale = match factorization.kind() {
        FactorizedColorContractionKind::KleinFourWalsh => 1.0,
        FactorizedColorContractionKind::ElementaryAbelianWalsh => 1.0 / subgroup_order as f64,
        FactorizedColorContractionKind::SymmetricGroupFourier => unreachable!(),
    };
    let mut transformed_entries = Vec::new();
    for left_coset_index in 0..cosets.len() {
        for right_coset_index in left_coset_index..cosets.len() {
            let left_coset = &cosets[left_coset_index];
            let right_coset = &cosets[right_coset_index];
            let mut weights = (0..subgroup_order)
                .map(|subgroup_index| matrix_value(left_coset[0], right_coset[subgroup_index]))
                .collect::<Vec<_>>();
            walsh_butterfly_f64(&mut weights);
            for (character_index, weight) in weights.into_iter().enumerate() {
                let mut coefficient = weight * weight_scale;
                if left_coset_index != right_coset_index {
                    coefficient *= 2.0;
                }
                if coefficient == 0.0 {
                    continue;
                }
                if !coefficient.is_finite() {
                    return Err(malformed(
                        "runtime factorized color coefficient is not finite",
                    ));
                }
                transformed_entries.push(RuntimeFactorizedColorContractionEntry {
                    left_group_index: left_coset[character_index],
                    right_group_index: right_coset[character_index],
                    coefficient_re: coefficient,
                    coefficient_im: 0.0,
                });
            }
        }
    }
    if cosets
        .iter()
        .flat_map(|coset| coset.iter())
        .any(|group| *group >= local_group_count)
    {
        return Err(malformed(
            "runtime factorization references an out-of-bounds local group",
        ));
    }
    Ok(RuntimeFactorizedColorContraction {
        subgroup_order,
        cosets,
        entries: transformed_entries,
        amplitude_scale,
    })
}

fn build_runtime_symmetric_group_convolution(
    factorization: &FactorizedColorContraction,
    local_group_count: u32,
    entries: &[RawColorContractionEntry],
) -> RusticolResult<RuntimeSymmetricGroupColorContraction> {
    if factorization.kind() != FactorizedColorContractionKind::SymmetricGroupFourier {
        return Err(malformed(
            "non-symmetric factorization reached the symmetric-group runtime builder",
        ));
    }
    let fft_plan = SymmetricGroupFftPlan::new(factorization.rank() as usize).map_err(|error| {
        malformed(format!(
            "could not construct the authenticated symmetric-group FFT plan: {error}"
        ))
    })?;
    let group_order = fft_plan.order();
    if group_order != factorization.subgroup_order() {
        return Err(malformed(
            "symmetric-group FFT order disagrees with factorization metadata",
        ));
    }
    let channel_count = factorization.coset_count();
    let kernel_entry_count = channel_count
        .checked_mul(channel_count + 1)
        .and_then(|value| value.checked_div(2))
        .and_then(|value| value.checked_mul(group_order))
        .ok_or_else(|| malformed("symmetric-group runtime kernel count overflows usize"))?;
    let mut kernels = Vec::with_capacity(channel_count * (channel_count + 1) / 2);
    let mut scalar_same_channel_low_rank_kernels =
        Vec::with_capacity(channel_count * (channel_count + 1) / 2);
    let mut has_scalar_same_channel_low_rank = false;
    let mut inverse_kernel = vec![(0.0, 0.0); group_order];
    let mut transformed_kernel = vec![(0.0, 0.0); group_order];
    let inverse_relative_indices = (0..group_order)
        .map(|index| {
            fft_plan
                .inverse_lexicographic_index(index)
                .map_err(|error| {
                    malformed(format!(
                        "could not invert a symmetric-group relative index: {error}"
                    ))
                })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut fft_workspace = fft_plan.workspace(1).map_err(|error| {
        malformed(format!(
            "could not allocate the authenticated symmetric-group FFT workspace: {error}"
        ))
    })?;
    let mut offset = 0usize;
    for left_channel_index in 0..channel_count {
        for right_channel_index in left_channel_index..channel_count {
            let raw_kernel = &entries[offset..offset + group_order];
            if left_channel_index == right_channel_index {
                for (relative_index, entry) in raw_kernel.iter().enumerate() {
                    let inverse_index = inverse_relative_indices[relative_index];
                    if entry.weight_re != raw_kernel[inverse_index].weight_re {
                        return Err(malformed(
                            "symmetric-group diagonal kernel violates inverse Hermiticity",
                        ));
                    }
                }
            }
            for (relative_index, value) in inverse_kernel.iter_mut().enumerate() {
                let inverse_index = inverse_relative_indices[relative_index];
                *value = (raw_kernel[inverse_index].weight_re, 0.0);
            }
            fft_plan
                .forward(&inverse_kernel, &mut transformed_kernel, &mut fft_workspace)
                .map_err(|error| {
                    malformed(format!(
                        "could not transform an authenticated symmetric-group kernel: {error}"
                    ))
                })?;
            if transformed_kernel
                .iter()
                .any(|value| value.1 != 0.0 || !value.0.is_finite())
            {
                return Err(malformed(
                    "symmetric-group kernel transform did not produce finite real Young blocks",
                ));
            }
            // Canonical zero rows remain mandatory on wire and participate in
            // exact-factor authentication above.  Once their transform is
            // certified zero, retaining a runtime block would only add warmed
            // traffic and immutable RSS.
            if transformed_kernel.iter().any(|value| value.0 != 0.0) {
                let fourier_coefficients = transformed_kernel
                    .iter()
                    .map(|value| value.0)
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let scalar_same_channel_blocks = if left_channel_index == right_channel_index {
                    let blocks = fft_plan
                        .blocks()
                        .map(|block| {
                            let start = block.coefficient_offset();
                            let end = start + block.coefficient_count();
                            compile_scalar_same_channel_low_rank_block(
                                &fourier_coefficients[start..end],
                                block.dimension(),
                            )
                        })
                        .collect::<Vec<_>>();
                    compact_scalar_same_channel_blocks(blocks)
                } else {
                    Box::default()
                };
                has_scalar_same_channel_low_rank |=
                    scalar_same_channel_blocks.iter().any(Option::is_some);
                kernels.push(RuntimeSymmetricGroupKernel {
                    left_channel_index: left_channel_index as u32,
                    right_channel_index: right_channel_index as u32,
                    pair_scale: if left_channel_index == right_channel_index {
                        1.0
                    } else {
                        2.0
                    },
                    fourier_coefficients,
                });
                scalar_same_channel_low_rank_kernels.push(RuntimeSymmetricGroupLowRankKernel {
                    blocks: scalar_same_channel_blocks,
                });
            }
            offset += group_order;
        }
    }
    debug_assert_eq!(offset, kernel_entry_count);
    debug_assert_eq!(scalar_same_channel_low_rank_kernels.len(), kernels.len());
    let scalar_same_channel_low_rank_kernels: Option<Arc<[RuntimeSymmetricGroupLowRankKernel]>> =
        has_scalar_same_channel_low_rank.then(|| Arc::from(scalar_same_channel_low_rank_kernels));
    let residual_entries: Vec<RuntimeFactorizedColorContractionEntry> = entries
        [kernel_entry_count..]
        .iter()
        .filter_map(|entry| {
            let coefficient_re = entry.weight_re * entry.symmetry_factor;
            let coefficient_im = entry.weight_im * entry.symmetry_factor;
            (coefficient_re != 0.0 || coefficient_im != 0.0).then_some(
                RuntimeFactorizedColorContractionEntry {
                    left_group_index: entry.left_group_id,
                    right_group_index: entry.right_group_id,
                    coefficient_re,
                    coefficient_im,
                },
            )
        })
        .collect();
    let eligible_group_count = channel_count
        .checked_mul(group_order)
        .ok_or_else(|| malformed("symmetric-group eligible group count overflows usize"))?;
    if eligible_group_count > local_group_count as usize {
        return Err(malformed(
            "runtime symmetric-group channel references an out-of-bounds local group",
        ));
    }
    Ok(RuntimeSymmetricGroupColorContraction {
        degree: factorization.rank(),
        group_order,
        channel_count,
        local_group_count: local_group_count as usize,
        fft_plan: Arc::new(fft_plan),
        kernels: kernels.into(),
        scalar_same_channel_low_rank_kernels,
        residual_entries: residual_entries.into(),
    })
}

fn walsh_butterfly_f64(values: &mut [f64]) {
    debug_assert!(values.len().is_power_of_two());
    let mut stride = 1;
    while stride < values.len() {
        for start in (0..values.len()).step_by(stride * 2) {
            for offset in 0..stride {
                let left = values[start + offset];
                let right = values[start + stride + offset];
                values[start + offset] = left + right;
                values[start + stride + offset] = left - right;
            }
        }
        stride *= 2;
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize, label: &str) -> RusticolResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| malformed(format!("{label} offset overflows usize")))?;
        let result = self.bytes.get(self.offset..end).ok_or_else(|| {
            malformed(format!(
                "truncated {label} at byte {}: need {count}, have {}",
                self.offset,
                self.bytes.len().saturating_sub(self.offset)
            ))
        })?;
        self.offset = end;
        Ok(result)
    }

    fn u32(&mut self, label: &str) -> RusticolResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4, label)?.try_into().expect("checked read"),
        ))
    }

    fn u64(&mut self, label: &str) -> RusticolResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8, label)?.try_into().expect("checked read"),
        ))
    }

    fn i128(&mut self, label: &str) -> RusticolResult<i128> {
        Ok(i128::from_le_bytes(
            self.take(16, label)?.try_into().expect("checked read"),
        ))
    }

    fn f64(&mut self, label: &str) -> RusticolResult<f64> {
        Ok(f64::from_le_bytes(
            self.take(8, label)?.try_into().expect("checked read"),
        ))
    }

    fn count(&mut self, label: &str) -> RusticolResult<usize> {
        let value = self.u64(label)?;
        usize::try_from(value).map_err(|_| malformed(format!("{label} exceeds usize")))
    }

    fn u32_vec(&mut self, count: usize, label: &str) -> RusticolResult<Vec<u32>> {
        (0..count).map(|_| self.u32(label)).collect()
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestWire {
        storage: u32,
        accuracy: u32,
        flags: u32,
        group_count: u32,
        sector_count: u32,
        component_count: u32,
        local_group_count: u32,
        destination_count: u32,
        factor_kind: u32,
        factor_rank: u32,
        entries: Vec<RawColorContractionEntry>,
        exact_factors: Vec<ExactComplexRational>,
        ordered_group_ids: Vec<u32>,
        destination_by_group: Vec<u32>,
        sector_by_group: Vec<u32>,
        component_by_group: Vec<u32>,
        owner_by_sector: Vec<u32>,
        cosets: Vec<Vec<u32>>,
        logical_entry_count: u64,
    }

    impl TestWire {
        fn expanded() -> Self {
            Self {
                storage: 1,
                accuracy: 1,
                flags: FLAG_INCLUDES_COLOR_FACTOR,
                group_count: 2,
                sector_count: 2,
                component_count: 1,
                local_group_count: 0,
                destination_count: 12,
                factor_kind: 0,
                factor_rank: 0,
                entries: vec![
                    RawColorContractionEntry {
                        left_group_id: 0,
                        right_group_id: 0,
                        weight_re: 3.0,
                        weight_im: 0.0,
                        symmetry_factor: 1.0,
                        exact_factor_id: 0,
                    },
                    RawColorContractionEntry {
                        left_group_id: 0,
                        right_group_id: 1,
                        weight_re: 2.0,
                        weight_im: 0.5,
                        symmetry_factor: 2.0,
                        exact_factor_id: 1,
                    },
                ],
                exact_factors: vec![
                    ExactComplexRational::new(
                        ExactRational::new(3, 1).unwrap(),
                        ExactRational::ZERO,
                    ),
                    ExactComplexRational::new(
                        ExactRational::new(4, 1).unwrap(),
                        ExactRational::new(1, 1).unwrap(),
                    ),
                ],
                ordered_group_ids: vec![0, 1],
                destination_by_group: vec![7, 9],
                sector_by_group: vec![0, 1],
                component_by_group: vec![0, 0],
                owner_by_sector: vec![0, 1],
                cosets: Vec::new(),
                logical_entry_count: 2,
            }
        }

        fn repeated_k4() -> Self {
            Self {
                storage: 2,
                accuracy: 2,
                flags: FLAG_INCLUDES_COLOR_FACTOR,
                group_count: 8,
                sector_count: 4,
                component_count: 2,
                local_group_count: 4,
                destination_count: 20,
                factor_kind: 1,
                factor_rank: 2,
                entries: (0..4)
                    .map(|index| RawColorContractionEntry {
                        left_group_id: index,
                        right_group_id: index,
                        weight_re: 1.0,
                        weight_im: 0.0,
                        symmetry_factor: 1.0,
                        exact_factor_id: 0,
                    })
                    .collect(),
                exact_factors: vec![ExactComplexRational::ONE],
                ordered_group_ids: vec![0, 4, 1, 5, 2, 6, 3, 7],
                destination_by_group: vec![8, 9, 10, 11, 12, 13, 14, 15],
                sector_by_group: vec![0, 1, 2, 3, 0, 1, 2, 3],
                component_by_group: vec![0, 0, 0, 0, 1, 1, 1, 1],
                owner_by_sector: vec![0, 1, 2, 3],
                cosets: vec![vec![0, 1, 2, 3]],
                logical_entry_count: 8,
            }
        }

        fn repeated_k4_multicoset() -> Self {
            let diagonal_blocks = [[2.0, 1.0, 0.0, -1.0], [3.0, 0.5, -0.5, 1.5]];
            let cross_block = [1.0, 2.0, -1.0, 0.25];
            let matrix_value = |left: usize, right: usize| {
                let left_coset = left / 4;
                let right_coset = right / 4;
                let xor = (left % 4) ^ (right % 4);
                if left_coset == right_coset {
                    diagonal_blocks[left_coset][xor]
                } else {
                    cross_block[xor]
                }
            };
            let mut entries = Vec::new();
            let mut exact_factors = Vec::new();
            for left in 0..8 {
                for right in left..8 {
                    let weight = matrix_value(left, right);
                    if weight == 0.0 {
                        continue;
                    }
                    let symmetry = if left == right { 1.0 } else { 2.0 };
                    let coefficient = weight * symmetry;
                    let exact_factor_id = exact_factors.len() as u32;
                    exact_factors.push(ExactComplexRational::new(
                        ExactRational::from_f64_exact(coefficient).unwrap(),
                        ExactRational::ZERO,
                    ));
                    entries.push(RawColorContractionEntry {
                        left_group_id: left as u32,
                        right_group_id: right as u32,
                        weight_re: weight,
                        weight_im: 0.0,
                        symmetry_factor: symmetry,
                        exact_factor_id,
                    });
                }
            }
            let logical_entry_count = (2 * entries.len()) as u64;
            Self {
                storage: 2,
                accuracy: 2,
                flags: FLAG_INCLUDES_COLOR_FACTOR,
                group_count: 16,
                sector_count: 8,
                component_count: 2,
                local_group_count: 8,
                destination_count: 16,
                factor_kind: 1,
                factor_rank: 2,
                entries,
                exact_factors,
                ordered_group_ids: (0..16).collect(),
                destination_by_group: (0..16).collect(),
                sector_by_group: (0..8).flat_map(|sector| [sector, sector]).collect(),
                component_by_group: (0..8).flat_map(|_| [0, 1]).collect(),
                owner_by_sector: (0..8).collect(),
                cosets: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                logical_entry_count,
            }
        }

        fn convolution_s3_with_residual() -> Self {
            let diagonal_left = [6.0, 1.0, 2.0, 3.0, 3.0, 4.0];
            let cross = [1.0, 2.0, 0.0, -1.0, 0.5, 3.0];
            let diagonal_right = [7.0, 0.0, 2.0, -1.0, -1.0, 0.0];
            let mut entries = Vec::new();
            let mut exact_factors = Vec::new();
            let mut push = |left: u32, right: u32, weight: f64, symmetry: f64| {
                let exact_factor_id = exact_factors.len() as u32;
                exact_factors.push(ExactComplexRational::new(
                    ExactRational::from_f64_exact(weight * symmetry).unwrap(),
                    ExactRational::ZERO,
                ));
                entries.push(RawColorContractionEntry {
                    left_group_id: left,
                    right_group_id: right,
                    weight_re: weight,
                    weight_im: 0.0,
                    symmetry_factor: symmetry,
                    exact_factor_id,
                });
            };
            for (relative, weight) in diagonal_left.into_iter().enumerate() {
                push(0, relative as u32, weight, 1.0);
            }
            for (relative, weight) in cross.into_iter().enumerate() {
                push(0, 6 + relative as u32, weight, 2.0);
            }
            for (relative, weight) in diagonal_right.into_iter().enumerate() {
                push(6, 6 + relative as u32, weight, 1.0);
            }
            // Ordinary direct rows touching the residual suffix follow kernels.
            push(0, 12, 0.25, 2.0);
            for left in 1..12 {
                push(left, 12, 0.0, 2.0);
            }
            push(12, 12, 5.0, 1.0);
            let logical_entry_count = entries.len() as u64;
            Self {
                storage: 3,
                accuracy: 2,
                flags: FLAG_INCLUDES_COLOR_FACTOR,
                group_count: 13,
                sector_count: 13,
                component_count: 1,
                local_group_count: 13,
                destination_count: 20,
                factor_kind: 3,
                factor_rank: 3,
                entries,
                exact_factors,
                ordered_group_ids: (0..13).collect(),
                destination_by_group: (0..13).map(|value| value + 2).collect(),
                sector_by_group: (0..13).collect(),
                component_by_group: vec![0; 13],
                owner_by_sector: (0..13).collect(),
                cosets: vec![(0..6).collect(), (6..12).collect()],
                logical_entry_count,
            }
        }

        fn encode(&self) -> Vec<u8> {
            let flattened_cosets = self.cosets.iter().flatten().copied().collect::<Vec<_>>();
            let payload_bytes = self.entries.len() * ENTRY_BYTES
                + self.exact_factors.len() * EXACT_FACTOR_BYTES
                + self.ordered_group_ids.len() * 4
                + self.destination_by_group.len() * 4
                + self.sector_by_group.len() * 4
                + self.component_by_group.len() * 4
                + self.owner_by_sector.len() * 4
                + flattened_cosets.len() * 4;
            let mut bytes = Vec::with_capacity(HEADER_BYTES + payload_bytes);
            bytes.extend_from_slice(MAGIC);
            for value in [
                VERSION,
                HEADER_BYTES as u32,
                self.storage,
                self.accuracy,
                self.flags,
                self.group_count,
                self.sector_count,
                self.component_count,
                self.local_group_count,
                self.destination_count,
                self.factor_kind,
                self.factor_rank,
                ENTRY_BYTES as u32,
                EXACT_FACTOR_BYTES as u32,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in [
                self.entries.len() as u64,
                self.exact_factors.len() as u64,
                self.cosets.len() as u64,
                flattened_cosets.len() as u64,
                self.logical_entry_count,
                self.owner_by_sector.len() as u64,
                payload_bytes as u64,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            assert_eq!(bytes.len(), HEADER_BYTES);
            for entry in &self.entries {
                bytes.extend_from_slice(&entry.left_group_id.to_le_bytes());
                bytes.extend_from_slice(&entry.right_group_id.to_le_bytes());
                bytes.extend_from_slice(&entry.weight_re.to_le_bytes());
                bytes.extend_from_slice(&entry.weight_im.to_le_bytes());
                bytes.extend_from_slice(&entry.symmetry_factor.to_le_bytes());
                bytes.extend_from_slice(&entry.exact_factor_id.to_le_bytes());
            }
            for factor in &self.exact_factors {
                for value in [
                    factor.real().numerator(),
                    factor.real().denominator(),
                    factor.imag().numerator(),
                    factor.imag().denominator(),
                ] {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            for value in self
                .ordered_group_ids
                .iter()
                .chain(&self.destination_by_group)
                .chain(&self.sector_by_group)
                .chain(&self.component_by_group)
                .chain(&self.owner_by_sector)
                .chain(&flattened_cosets)
            {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes
        }
    }

    fn error_contains(bytes: &[u8], expected: &str) {
        let error = decode_recurrence_color_contraction_v3(bytes).unwrap_err();
        assert!(
            error.message().contains(expected),
            "expected {expected:?} in {:?}",
            error.message()
        );
    }

    #[test]
    fn expanded_payload_preserves_raw_and_derives_runtime_entries() {
        let bytes = TestWire::expanded().encode();
        let plan = decode_recurrence_color_contraction_v3(&bytes).unwrap();
        assert_eq!(plan.accuracy(), RecurrenceColorAccuracy::Nlc);
        assert_eq!(plan.storage(), RecurrenceColorStorage::Expanded);
        assert!(plan.includes_color_factor());
        assert_eq!(plan.sector_count(), 2);
        assert_eq!(plan.component_count(), 1);
        assert_eq!(plan.ordered_group_ids(), [0, 1]);
        assert_eq!(plan.destination_by_group(), [7, 9]);
        let raw = plan.canonical_logical_entries().collect::<Vec<_>>();
        assert_eq!(raw[1].weight_re, 2.0);
        assert_eq!(raw[1].weight_im, 0.5);
        assert_eq!(raw[1].symmetry_factor, 2.0);
        let runtime = plan.runtime_entries().collect::<Vec<_>>();
        assert_eq!(runtime[1].left_destination_id, 7);
        assert_eq!(runtime[1].right_destination_id, 9);
        assert_eq!(runtime[1].coefficient_re, 4.0);
        assert_eq!(runtime[1].coefficient_im, 1.0);
        assert_eq!(
            recurrence_color_contraction_digest(&bytes),
            recurrence_color_contraction_digest(&bytes)
        );
    }

    #[test]
    fn runtime_entry_contracts_complex_upper_triangle_row() {
        let plan = decode_recurrence_color_contraction_v3(&TestWire::expanded().encode()).unwrap();
        let runtime = plan.runtime_entries().collect::<Vec<_>>();

        // The diagonal row contributes 3 * |1 + 2i|^2.
        assert_eq!(runtime[0].contract_real_bilinear(1.0, 2.0, 1.0, 2.0), 15.0);
        // The stored upper-triangle row has raw coefficient 2 + 0.5i and
        // symmetry factor two, hence runtime coefficient 4 + i.
        assert_eq!(runtime[1].contract_real_bilinear(1.0, 2.0, 3.0, 4.0), 42.0);
    }

    #[test]
    fn repeated_k4_payload_expands_logical_rows_without_runtime_allocation() {
        let bytes = TestWire::repeated_k4().encode();
        let plan = decode_recurrence_color_contraction_v3(&bytes).unwrap();
        assert_eq!(plan.storage(), RecurrenceColorStorage::Repeated);
        assert_eq!(plan.logical_entry_count(), 8);
        let factor = plan.factorization().unwrap();
        assert_eq!(
            factor.kind(),
            FactorizedColorContractionKind::KleinFourWalsh
        );
        assert_eq!(factor.coset(0), Some(&[0, 1, 2, 3][..]));
        let runtime_factor = plan.runtime_factorization().unwrap();
        assert_eq!(runtime_factor.subgroup_order(), 4);
        assert_eq!(runtime_factor.cosets(), [vec![0, 1, 2, 3]]);
        assert_eq!(runtime_factor.amplitude_scale(), 0.5);
        assert_eq!(runtime_factor.entries().len(), 4);
        assert!(runtime_factor.entries().iter().all(|entry| {
            entry.left_group_index == entry.right_group_index
                && entry.coefficient_re == 1.0
                && entry.coefficient_im == 0.0
        }));
        assert_eq!(plan.ordered_destination_id(0, 0), Some(8));
        assert_eq!(plan.ordered_destination_id(0, 1), Some(12));
        assert_eq!(plan.ordered_destination_id(3, 1), Some(15));
        let mut entries = plan.canonical_logical_entries();
        assert_eq!(entries.len(), 8);
        assert_eq!(entries.next().unwrap().left_group_id, 0);
        assert_eq!(entries.next().unwrap().left_group_id, 1);
        assert_eq!(entries.next().unwrap().left_group_id, 2);
        assert_eq!(entries.next().unwrap().left_group_id, 3);
        assert_eq!(entries.next().unwrap().left_group_id, 4);
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn repeated_multicoset_k4_preserves_the_nontrivial_quadratic_form() {
        let plan =
            decode_recurrence_color_contraction_v3(&TestWire::repeated_k4_multicoset().encode())
                .unwrap();
        let amplitudes = [0.5, -1.25, 2.0, 0.75, -0.4, 1.1, 0.2, -0.9];
        let direct = plan
            .entries()
            .iter()
            .map(|entry| {
                entry.weight_re
                    * entry.symmetry_factor
                    * amplitudes[entry.left_group_id as usize]
                    * amplitudes[entry.right_group_id as usize]
            })
            .sum::<f64>();

        let factorized = plan.runtime_factorization().unwrap();
        assert_eq!(factorized.cosets().len(), 2);
        let mut transformed = [0.0; 8];
        for coset in factorized.cosets() {
            let mut values = coset
                .iter()
                .map(|index| amplitudes[*index as usize])
                .collect::<Vec<_>>();
            walsh_butterfly_f64(&mut values);
            for (index, value) in coset.iter().zip(values) {
                transformed[*index as usize] = value * factorized.amplitude_scale();
            }
        }
        let transformed_value = factorized
            .entries()
            .iter()
            .map(|entry| {
                entry.coefficient_re
                    * transformed[entry.left_group_index as usize]
                    * transformed[entry.right_group_index as usize]
            })
            .sum::<f64>();
        assert!((direct - transformed_value).abs() <= 32.0 * f64::EPSILON);
    }

    #[test]
    fn symmetric_group_payload_preserves_kernel_orientation_and_residual_rows() {
        let plan = decode_recurrence_color_contraction_v3(
            &TestWire::convolution_s3_with_residual().encode(),
        )
        .unwrap();
        assert_eq!(plan.storage(), RecurrenceColorStorage::ConvolutionKernels);
        let factorization = plan.factorization().unwrap();
        assert_eq!(
            factorization.kind(),
            FactorizedColorContractionKind::SymmetricGroupFourier
        );
        assert_eq!(factorization.rank(), 3);
        assert_eq!(factorization.subgroup_order(), 6);
        assert_eq!(factorization.coset(0), Some(&[0, 1, 2, 3, 4, 5][..]));
        assert_eq!(factorization.coset(1), Some(&[6, 7, 8, 9, 10, 11][..]));
        let RuntimeColorContractionReducer::SymmetricGroupFourier(runtime) =
            plan.runtime_reducer().unwrap()
        else {
            panic!("expected symmetric-group runtime reducer")
        };
        assert_eq!(runtime.degree(), 3);
        assert_eq!(runtime.group_order(), 6);
        assert_eq!(runtime.channel_count(), 2);
        assert_eq!(runtime.local_group_count(), 13);
        assert_eq!(runtime.kernels().len(), 3);
        assert_eq!(runtime.kernels()[1].left_channel_index(), 0);
        assert_eq!(runtime.kernels()[1].right_channel_index(), 1);
        assert_eq!(runtime.kernels()[0].pair_scale(), 1.0);
        assert_eq!(runtime.kernels()[1].pair_scale(), 2.0);
        let fft_plan = SymmetricGroupFftPlan::new(3).unwrap();
        assert!(runtime.scalar_same_channel_low_rank_kernels.is_none());
        let raw_cross = [1.0, 2.0, 0.0, -1.0, 0.5, 3.0];
        let inverse_cross = (0..fft_plan.order())
            .map(|index| {
                (
                    raw_cross[fft_plan.inverse_lexicographic_index(index).unwrap()],
                    0.0,
                )
            })
            .collect::<Vec<_>>();
        let mut expected_fourier = vec![(0.0, 0.0); fft_plan.order()];
        let mut fft_workspace = fft_plan.workspace(1).unwrap();
        fft_plan
            .forward(&inverse_cross, &mut expected_fourier, &mut fft_workspace)
            .unwrap();
        assert_eq!(
            runtime.kernels()[1].fourier_coefficients(),
            expected_fourier
                .iter()
                .map(|value| value.0)
                .collect::<Vec<_>>()
        );
        assert_eq!(runtime.residual_entries().len(), 2);
        assert_eq!(runtime.residual_entries()[0].left_group_index, 0);
        assert_eq!(runtime.residual_entries()[0].right_group_index, 12);
        assert_eq!(runtime.residual_entries()[0].coefficient_re, 0.5);
        assert_eq!(plan.stored_entry_count(), 31);
        assert_eq!(plan.logical_entry_count(), 31);
        assert!(plan.entries().is_empty());
        assert!(plan.exact_factors().is_empty());
    }

    fn test_low_rank_effective_kernel() -> (usize, Vec<f64>) {
        let dimension = 5;
        let root_two = 2.0_f64.sqrt();
        let columns = [
            [root_two, 0.0, 1.0 / root_two, 0.0, 0.0],
            [0.0, root_two, 0.0, 1.0 / root_two, 0.0],
        ];
        let mut kernel = vec![0.0; dimension * dimension];
        for column in 0..dimension {
            for row in 0..dimension {
                kernel[column * dimension + row] = columns
                    .iter()
                    .map(|factor| factor[column] * factor[row])
                    .sum();
            }
        }
        // Only the effective symmetric part is contracted. Opposite skew
        // perturbations must therefore leave the derived factor unchanged.
        kernel[2] += 0.125;
        kernel[2 * dimension] -= 0.125;
        (dimension, kernel)
    }

    #[test]
    fn scalar_low_rank_factorization_is_deterministic_scale_relative_and_certified() {
        let (dimension, kernel) = test_low_rank_effective_kernel();
        let factor = compile_scalar_same_channel_low_rank_block(&kernel, dimension).unwrap();
        assert_eq!(factor.rank, 2);
        // Equal leading residual diagonals choose original index zero first
        // and original index one second.
        assert!((factor.factors[0] - 2.0_f64.sqrt()).abs() <= 8.0 * f64::EPSILON);
        assert_eq!(factor.factors[1], 0.0);
        assert_eq!(factor.factors[dimension], 0.0);
        assert!((factor.factors[dimension + 1] - 2.0_f64.sqrt()).abs() <= 8.0 * f64::EPSILON);

        let tiny_scale = 1.0e-200;
        let mut tiny = vec![0.0; dimension * dimension];
        tiny[0] = tiny_scale;
        let tiny_factor = compile_scalar_same_channel_low_rank_block(&tiny, dimension).unwrap();
        assert_eq!(tiny_factor.rank, 1);
        assert!(tiny_factor.factors[0] > 0.0);
        assert!(tiny_factor.factors[0] < 1.0e-90);

        let mut effective = kernel.clone();
        let symmetric = 0.5 * kernel[2] + 0.5 * kernel[2 * dimension];
        effective[2] = symmetric;
        effective[2 * dimension] = symmetric;
        let scale = (0..dimension)
            .map(|index| effective[index * dimension + index])
            .fold(f64::NEG_INFINITY, f64::max);
        let tolerance =
            LOW_RANK_PIVOT_TOLERANCE_MULTIPLIER * f64::EPSILON * dimension as f64 * scale;
        assert!(low_rank_factor_reconstructs_effective_kernel(
            &effective,
            dimension,
            factor.rank,
            &factor.factors,
            tolerance,
        ));
        let mut corrupted = factor.factors.to_vec();
        corrupted[0] += 1.0e-5;
        assert!(!low_rank_factor_reconstructs_effective_kernel(
            &effective,
            dimension,
            factor.rank,
            &corrupted,
            tolerance,
        ));
    }

    #[test]
    fn scalar_low_rank_factorization_falls_back_for_invalid_or_unprofitable_blocks() {
        let dimension = 5;
        let zero = vec![0.0; dimension * dimension];
        let zero_factor = compile_scalar_same_channel_low_rank_block(&zero, dimension).unwrap();
        assert_eq!(zero_factor.rank, 0);
        assert!(zero_factor.factors.is_empty());

        let mut negative_diagonal = zero;
        negative_diagonal[0] = -1.0;
        assert!(
            compile_scalar_same_channel_low_rank_block(&negative_diagonal, dimension).is_none()
        );

        let mut indefinite = vec![0.0; dimension * dimension];
        for index in 0..dimension {
            indefinite[index * dimension + index] = 1.0;
        }
        indefinite[1] = 2.0;
        indefinite[dimension] = 2.0;
        assert!(compile_scalar_same_channel_low_rank_block(&indefinite, dimension).is_none());

        let mut nonfinite = indefinite;
        nonfinite[0] = f64::NAN;
        assert!(compile_scalar_same_channel_low_rank_block(&nonfinite, dimension).is_none());

        let mut full_rank = vec![0.0; dimension * dimension];
        for index in 0..dimension {
            full_rank[index * dimension + index] = 1.0;
        }
        assert!(compile_scalar_same_channel_low_rank_block(&full_rank, dimension).is_none());
        assert!(!low_rank_scalar_cost_is_profitable(dimension, dimension));
        assert!(low_rank_scalar_cost_is_profitable(dimension, 2));

        let compacted = compact_scalar_same_channel_blocks(vec![None, None]);
        assert!(compacted.is_empty());
        let compacted = compact_scalar_same_channel_blocks(vec![Some(zero_factor), None]);
        assert_eq!(compacted.len(), 2);
        assert!(compacted[0].is_some());
        assert!(compacted[1].is_none());
    }

    #[test]
    fn scalar_low_rank_contraction_matches_dense_and_batches_fall_back_exactly() {
        let (dimension, kernel) = test_low_rank_effective_kernel();
        let factor = compile_scalar_same_channel_low_rank_block(&kernel, dimension).unwrap();
        let coefficient_offset = 3;
        let lane_capacity = 2;
        let amplitudes = (0..(coefficient_offset + dimension * dimension) * lane_capacity)
            .map(|index| {
                (
                    ((index * 17 + 5) % 31) as f64 / 11.0 - 1.2,
                    ((index * 13 + 3) % 29) as f64 / 9.0 - 0.8,
                )
            })
            .collect::<Vec<_>>();
        let weight = 0.375;
        let mut dense = [0.25, -0.5];
        let mut low_rank = dense;
        contract_real_hermitian_same_channel_block(
            &amplitudes,
            &kernel,
            dimension,
            coefficient_offset,
            lane_capacity,
            1,
            weight,
            &mut dense,
        );
        contract_real_hermitian_same_channel_block_opportunistic(
            &amplitudes,
            &kernel,
            Some(&factor),
            dimension,
            coefficient_offset,
            lane_capacity,
            1,
            weight,
            &mut low_rank,
        );
        let scale = dense[0].abs().max(1.0);
        assert!((low_rank[0] - dense[0]).abs() <= 2.0e-12 * scale);
        assert_eq!(low_rank[1], dense[1]);

        let mut dense_batch = [0.25, -0.5];
        let mut fallback_batch = dense_batch;
        contract_real_hermitian_same_channel_block(
            &amplitudes,
            &kernel,
            dimension,
            coefficient_offset,
            lane_capacity,
            2,
            weight,
            &mut dense_batch,
        );
        contract_real_hermitian_same_channel_block_opportunistic(
            &amplitudes,
            &kernel,
            Some(&factor),
            dimension,
            coefficient_offset,
            lane_capacity,
            2,
            weight,
            &mut fallback_batch,
        );
        assert_eq!(fallback_batch, dense_batch);

        let malformed_factor = RuntimeSymmetricGroupLowRankBlock {
            rank: factor.rank,
            factors: factor.factors[..factor.factors.len() - 1]
                .to_vec()
                .into_boxed_slice(),
        };
        let mut malformed_fallback = [0.25, -0.5];
        contract_real_hermitian_same_channel_block_opportunistic(
            &amplitudes,
            &kernel,
            Some(&malformed_factor),
            dimension,
            coefficient_offset,
            lane_capacity,
            1,
            weight,
            &mut malformed_fallback,
        );
        assert_eq!(malformed_fallback, dense);
    }

    #[test]
    fn real_hermitian_same_channel_microkernel_matches_dense_complex_form() {
        for &(dimension, lane_count, lane_capacity) in &[(1, 1, 2), (5, 1, 3), (9, 3, 4)] {
            let coefficient_offset = 2;
            let weight = 0.375;
            let amplitude_count = (coefficient_offset + dimension * dimension) * lane_capacity;
            let amplitudes = (0..amplitude_count)
                .map(|index| {
                    (
                        ((index * 17 + 3) % 31) as f64 / 11.0 - 1.2,
                        ((index * 13 + 7) % 29) as f64 / 9.0 - 0.7,
                    )
                })
                .collect::<Vec<_>>();
            // The loader certifies mathematical symmetry.  Keeping a tiny
            // representational asymmetry here verifies that pairing K_ci and
            // K_ic still reproduces the previous full dense traversal.
            let kernel = (0..dimension * dimension)
                .map(|index| {
                    let row = index % dimension;
                    let column = index / dimension;
                    0.25 * (row + column + 1) as f64 + if row < column { 3.0e-14 } else { 0.0 }
                })
                .collect::<Vec<_>>();
            let initial = (0..lane_capacity)
                .map(|lane| lane as f64 * 0.125 - 0.25)
                .collect::<Vec<_>>();
            let mut expected = initial.clone();
            for column in 0..dimension {
                for inner in 0..dimension {
                    let kernel_value = kernel[column * dimension + inner] * weight;
                    for row in 0..dimension {
                        let left = (coefficient_offset + column * dimension + row) * lane_capacity;
                        let right = (coefficient_offset + inner * dimension + row) * lane_capacity;
                        for lane in 0..lane_count {
                            let left_value = amplitudes[left + lane];
                            let right_value = amplitudes[right + lane];
                            let product_re = left_value
                                .0
                                .mul_add(right_value.0, left_value.1 * right_value.1);
                            expected[lane] = kernel_value.mul_add(product_re, expected[lane]);
                        }
                    }
                }
            }

            let mut actual = initial;
            let amplitude_pointer = amplitudes.as_ptr();
            let kernel_pointer = kernel.as_ptr();
            let reduced_pointer = actual.as_ptr();
            contract_real_hermitian_same_channel_block(
                &amplitudes,
                &kernel,
                dimension,
                coefficient_offset,
                lane_capacity,
                lane_count,
                weight,
                &mut actual,
            );
            assert_eq!(amplitudes.as_ptr(), amplitude_pointer);
            assert_eq!(kernel.as_ptr(), kernel_pointer);
            assert_eq!(actual.as_ptr(), reduced_pointer);
            for lane in 0..lane_count {
                let scale = expected[lane].abs().max(1.0);
                assert!(
                    (actual[lane] - expected[lane]).abs() <= 2.0e-12 * scale,
                    "dimension={dimension}, lane={lane}: {} != {}",
                    actual[lane],
                    expected[lane]
                );
            }
            assert_eq!(&actual[lane_count..], &expected[lane_count..]);
        }
    }

    #[test]
    fn symmetric_group_reducer_matches_dense_complex_multichannel_metric_and_reuses_scratch() {
        let wire = TestWire::convolution_s3_with_residual();
        let plan = decode_recurrence_color_contraction_v3(&wire.encode()).unwrap();
        let RuntimeColorContractionReducer::SymmetricGroupFourier(reducer) =
            plan.runtime_reducer().unwrap()
        else {
            panic!("expected symmetric-group runtime reducer")
        };
        let covered_group_count = reducer.channel_count() * reducer.group_order();
        assert!(reducer.channel_count() > 0);
        assert!(reducer.local_group_count() > covered_group_count);
        assert!(reducer.residual_entries().iter().any(|entry| {
            entry.left_group_index < covered_group_count as u32
                && entry.right_group_index >= covered_group_count as u32
        }));
        let lane_count = 3;
        let amplitudes = (0..reducer.local_group_count() * lane_count)
            .map(|index| {
                let real = ((index * 17 + 5) % 29) as f64 / 7.0 - 1.5;
                let imaginary = ((index * 11 + 3) % 23) as f64 / 9.0 - 0.8;
                (real, imaginary)
            })
            .collect::<Vec<_>>();
        // Native lanes allocate their bounded full-tile capacity before warm
        // execution.  A full tile followed by an odd tail must reuse every
        // backing allocation.
        let mut workspace = reducer.workspace(lane_count).unwrap();
        assert_eq!(workspace.lane_capacity, lane_count);
        assert_eq!(
            workspace.gathered.len(),
            reducer.local_group_count() * lane_count
        );
        reducer
            .reduce_lanes(&mut workspace, lane_count, |group, lane| {
                Ok(amplitudes[group * lane_count + lane])
            })
            .unwrap();

        let expected = RecurrenceColorContraction::symmetric_group_s3_dense_for_runtime_test(
            &amplitudes,
            lane_count,
        );
        for (actual, expected) in workspace.reduced(lane_count).unwrap().iter().zip(expected) {
            let scale = expected.abs().max(1.0);
            assert!((actual - expected).abs() <= 2.0e-11 * scale);
        }

        let gathered_pointer = workspace.gathered.as_ptr();
        let transformed_pointer = workspace.transformed.as_ptr();
        let reduced_pointer = workspace.reduced.as_ptr();
        reducer
            .reduce_lanes(&mut workspace, 2, |group, lane| {
                Ok(amplitudes[group * lane_count + lane])
            })
            .unwrap();
        assert_eq!(workspace.lane_capacity, lane_count);
        assert_eq!(workspace.gathered.as_ptr(), gathered_pointer);
        assert_eq!(workspace.transformed.as_ptr(), transformed_pointer);
        assert_eq!(workspace.reduced.as_ptr(), reduced_pointer);
    }

    #[test]
    fn symmetric_group_runtime_drops_authenticated_all_zero_kernel_blocks() {
        let mut wire = TestWire::convolution_s3_with_residual();
        for entry in &mut wire.entries[6..12] {
            entry.weight_re = 0.0;
            wire.exact_factors[entry.exact_factor_id as usize] = ExactComplexRational::ZERO;
        }
        let plan = decode_recurrence_color_contraction_v3(&wire.encode()).unwrap();
        let RuntimeColorContractionReducer::SymmetricGroupFourier(reducer) =
            plan.runtime_reducer().unwrap()
        else {
            panic!("expected symmetric-group runtime reducer")
        };
        assert_eq!(reducer.kernels().len(), 2);
        assert!(
            reducer
                .kernels()
                .iter()
                .all(|kernel| kernel.left_channel_index() == kernel.right_channel_index())
        );
        assert_eq!(plan.stored_entry_count(), wire.entries.len());
    }

    #[test]
    fn symmetric_group_clones_share_immutable_data_and_keep_workspaces_isolated() {
        let plan = decode_recurrence_color_contraction_v3(
            &TestWire::convolution_s3_with_residual().encode(),
        )
        .unwrap();
        let RuntimeColorContractionReducer::SymmetricGroupFourier(reducer) =
            plan.runtime_reducer().unwrap()
        else {
            panic!("expected symmetric-group runtime reducer")
        };
        let first = reducer.clone();
        let second = reducer.clone();
        assert!(Arc::ptr_eq(&first.fft_plan, &second.fft_plan));
        assert!(Arc::ptr_eq(&first.kernels, &second.kernels));
        assert!(Arc::ptr_eq(
            &first.residual_entries,
            &second.residual_entries
        ));

        let run = |reducer: RuntimeSymmetricGroupColorContraction, scale: f64| {
            std::thread::spawn(move || {
                let mut workspace = reducer.workspace(1).unwrap();
                reducer
                    .reduce_lanes(&mut workspace, 1, |group, _| {
                        Ok((scale * (group + 1) as f64, -0.25 * scale))
                    })
                    .unwrap();
                workspace.reduced(1).unwrap()[0]
            })
        };
        let first_value = run(first, 1.0).join().unwrap();
        let second_value = run(second, 2.0).join().unwrap();
        let scale = first_value.abs().max(1.0);
        assert!((second_value - 4.0 * first_value).abs() <= 2.0e-11 * scale);
    }

    #[test]
    fn decoder_rejects_symmetric_group_kernel_and_residual_tampering() {
        let mut missing_zero = TestWire::convolution_s3_with_residual();
        missing_zero.entries.remove(8);
        missing_zero.logical_entry_count -= 1;
        error_contains(&missing_zero.encode(), "canonical");

        let mut bad_symmetry = TestWire::convolution_s3_with_residual();
        bad_symmetry.entries[6].symmetry_factor = 1.0;
        bad_symmetry.entries[6].exact_factor_id = bad_symmetry.exact_factors.len() as u32;
        bad_symmetry.exact_factors.push(ExactComplexRational::ONE);
        error_contains(&bad_symmetry.encode(), "canonical");

        let mut non_hermitian = TestWire::convolution_s3_with_residual();
        non_hermitian.entries[3].weight_re = 9.0;
        non_hermitian.entries[3].exact_factor_id = non_hermitian.exact_factors.len() as u32;
        non_hermitian.exact_factors.push(ExactComplexRational::new(
            ExactRational::new(9, 1).unwrap(),
            ExactRational::ZERO,
        ));
        error_contains(&non_hermitian.encode(), "inverse Hermiticity");

        let mut eligible_only_residual = TestWire::convolution_s3_with_residual();
        eligible_only_residual.entries[18].left_group_id = 1;
        eligible_only_residual.entries[18].right_group_id = 11;
        error_contains(&eligible_only_residual.encode(), "residual row");

        let mut complex_residual = TestWire::convolution_s3_with_residual();
        complex_residual.entries[18].weight_im = 0.5;
        complex_residual.entries[18].exact_factor_id = complex_residual.exact_factors.len() as u32;
        complex_residual
            .exact_factors
            .push(ExactComplexRational::new(
                ExactRational::new(1, 2).unwrap(),
                ExactRational::new(1, 1).unwrap(),
            ));
        error_contains(&complex_residual.encode(), "complex coefficient");

        let mut bad_residual_symmetry = TestWire::convolution_s3_with_residual();
        bad_residual_symmetry.entries[18].symmetry_factor = 1.0;
        bad_residual_symmetry.entries[18].exact_factor_id =
            bad_residual_symmetry.exact_factors.len() as u32;
        bad_residual_symmetry
            .exact_factors
            .push(ExactComplexRational::new(
                ExactRational::new(1, 4).unwrap(),
                ExactRational::ZERO,
            ));
        error_contains(&bad_residual_symmetry.encode(), "symmetry factor");
    }

    #[test]
    fn decoder_rejects_mixed_duplicate_out_of_bounds_and_nonfinite_data() {
        let mut mixed = TestWire::expanded();
        mixed.factor_kind = 1;
        mixed.factor_rank = 2;
        mixed.cosets = vec![vec![0, 1, 2, 3]];
        error_contains(&mixed.encode(), "mixed");

        let mut duplicate = TestWire::expanded();
        duplicate.entries.push(duplicate.entries[0]);
        duplicate.logical_entry_count += 1;
        error_contains(&duplicate.encode(), "duplicates");

        let mut out_of_bounds = TestWire::expanded();
        out_of_bounds.entries[0].left_group_id = 2;
        error_contains(&out_of_bounds.encode(), "out-of-bounds");

        let mut nonfinite = TestWire::expanded();
        nonfinite.entries[0].weight_re = f64::NAN;
        error_contains(&nonfinite.encode(), "non-finite");

        let mut duplicate_destination = TestWire::expanded();
        duplicate_destination.destination_by_group = vec![7, 7];
        error_contains(&duplicate_destination.encode(), "duplicate Direct-Arena");
    }

    #[test]
    fn decoder_rejects_inconsistent_factorization_map_and_matrix() {
        let mut duplicate_coset = TestWire::repeated_k4();
        duplicate_coset.cosets[0][3] = 2;
        error_contains(&duplicate_coset.encode(), "duplicate ID");

        let mut non_invariant = TestWire::repeated_k4();
        non_invariant.entries[0].weight_re = 2.0;
        non_invariant.entries[0].exact_factor_id = 1;
        non_invariant.exact_factors.push(ExactComplexRational::new(
            ExactRational::new(2, 1).unwrap(),
            ExactRational::ZERO,
        ));
        error_contains(&non_invariant.encode(), "inconsistent");
    }

    #[test]
    fn decoder_rejects_incomplete_or_noncanonical_sector_ownership() {
        let mut missing_owner = TestWire::expanded();
        missing_owner.owner_by_sector = vec![0, 0];
        error_contains(&missing_owner.encode(), "owner sectors");

        let mut forward_owner = TestWire::expanded();
        forward_owner.owner_by_sector = vec![1, 1];
        error_contains(&forward_owner.encode(), "invalid canonical owner");
    }

    #[test]
    fn decoder_rejects_cross_component_expanded_entries_and_trailing_bytes() {
        let mut cross_component = TestWire::expanded();
        cross_component.sector_count = 1;
        cross_component.component_count = 2;
        cross_component.sector_by_group = vec![0, 0];
        cross_component.component_by_group = vec![0, 1];
        cross_component.owner_by_sector = vec![0];
        error_contains(&cross_component.encode(), "different components");

        let mut trailing = TestWire::expanded().encode();
        trailing.push(0);
        error_contains(&trailing, "fixed-width sections");
    }
}
