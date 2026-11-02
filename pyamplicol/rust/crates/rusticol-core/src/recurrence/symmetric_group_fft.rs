// SPDX-License-Identifier: 0BSD

//! Fast, unnormalised Fourier transforms on symmetric groups.
//!
//! The transform follows the subgroup tower `S_0 < S_1 < ... < S_m`.  Its
//! output contains one column-major matrix for every Young diagram `lambda`,
//! packed consecutively, and uses the convention
//!
//! `F_lambda = sum_g f(g) rho_lambda(g)`.
//!
//! Complex values are represented as `(real, imaginary)` pairs so this module
//! remains independent of any evaluator backend.  In the batched interface the
//! lane is the innermost index: `value[group_element * lane_count + lane]`.

use std::collections::BTreeMap;

use crate::{RusticolError, RusticolResult};

/// Largest supported permutation degree.
///
/// `10!` still fits in `u32`, which keeps the persistent reorder map compact.
pub const MAX_SYMMETRIC_GROUP_FFT_DEGREE: usize = 10;

/// Backend-independent complex-f64 storage used by the FFT.
pub type SymmetricGroupComplex64 = (f64, f64);

type YoungTableau = Box<[(u8, u8)]>;

#[derive(Clone, Copy, Debug, PartialEq)]
struct YoungGeneratorAction {
    partner: u32,
    diagonal: f64,
    mixing: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct YoungBranch {
    child_irrep: u32,
    basis_offset: u32,
}

#[derive(Debug)]
struct YoungIrrep {
    shape: Box<[u8]>,
    dimension: usize,
    coefficient_offset: usize,
    branches: Box<[YoungBranch]>,
    tableaux: Box<[YoungTableau]>,
    generators: Box<[YoungGeneratorAction]>,
}

impl YoungIrrep {
    fn generator(&self, generator: usize, basis: usize) -> YoungGeneratorAction {
        self.generators[generator * self.dimension + basis]
    }
}

#[derive(Debug)]
struct SymmetricGroupLevel {
    degree: usize,
    order: usize,
    maximum_dimension: usize,
    irreps: Box<[YoungIrrep]>,
}

/// Public description of one packed Fourier block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymmetricGroupFftBlock<'a> {
    shape: &'a [u8],
    dimension: usize,
    coefficient_offset: usize,
}

impl<'a> SymmetricGroupFftBlock<'a> {
    /// Row lengths of the Young diagram, in non-increasing order.
    pub fn shape(self) -> &'a [u8] {
        self.shape
    }

    /// Dimension of this irreducible representation.
    pub fn dimension(self) -> usize {
        self.dimension
    }

    /// Zero-based offset of the column-major matrix in the packed transform.
    pub fn coefficient_offset(self) -> usize {
        self.coefficient_offset
    }

    /// Number of scalar complex coefficients in this block.
    pub fn coefficient_count(self) -> usize {
        self.dimension * self.dimension
    }
}

/// Immutable representation and reorder metadata for one `S_m` transform.
#[derive(Debug)]
pub struct SymmetricGroupFftPlan {
    degree: usize,
    order: usize,
    maximum_dimension: usize,
    factorials: Box<[usize]>,
    levels: Box<[SymmetricGroupLevel]>,
    recursive_to_lexicographic: Box<[u32]>,
    scalar_degree_three: Option<ScalarDegreeThreeTransform>,
}

/// Load-time compiled scalar Fourier rows for the smallest non-abelian group.
///
/// The generic subgroup-tower implementation remains the source of truth: it
/// is applied once to each basis vector while constructing an `S_3` plan.
/// Warm scalar calls can then execute the same real representation transform
/// without repeatedly walking Young-branch metadata and clearing temporary
/// blocks.  This is still the complete `S_3` Fourier transform, merely a
/// compact base case of the general recursion.
#[derive(Debug)]
struct ScalarDegreeThreeTransform {
    row_offsets: Box<[u8]>,
    terms: Box<[(u8, f64)]>,
}

impl SymmetricGroupFftPlan {
    /// Build a transform plan for `S_degree`.
    #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__TEXT,__rcl_load"))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub fn new(degree: usize) -> RusticolResult<Self> {
        if degree > MAX_SYMMETRIC_GROUP_FFT_DEGREE {
            return Err(invalid(format!(
                "symmetric-group FFT degree {degree} exceeds supported maximum {MAX_SYMMETRIC_GROUP_FFT_DEGREE}"
            )));
        }
        let factorials = checked_factorials(degree)?;
        let order = factorials[degree];
        let levels = build_levels(degree, &factorials)?;
        let maximum_dimension = levels
            .iter()
            .map(|level| level.maximum_dimension)
            .max()
            .unwrap_or(1);
        let recursive_to_lexicographic = build_recursive_to_lexicographic(degree, &factorials)?;
        let mut plan = Self {
            degree,
            order,
            maximum_dimension,
            factorials: factorials.into_boxed_slice(),
            levels: levels.into_boxed_slice(),
            recursive_to_lexicographic: recursive_to_lexicographic.into_boxed_slice(),
            scalar_degree_three: None,
        };
        if degree == 3 {
            plan.scalar_degree_three = Some(compile_scalar_degree_three_transform(&plan)?);
        }
        Ok(plan)
    }

    /// Permutation degree `m`.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Group order `m!`, equal to the number of scalar Fourier coefficients.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Largest matrix dimension needed by a reusable workspace.
    pub fn maximum_block_dimension(&self) -> usize {
        self.maximum_dimension
    }

    /// Number of Young-irrep blocks in the packed transform.
    pub fn block_count(&self) -> usize {
        self.levels[self.degree].irreps.len()
    }

    /// Describe one Young-irrep block.
    pub fn block(&self, index: usize) -> Option<SymmetricGroupFftBlock<'_>> {
        self.levels[self.degree]
            .irreps
            .get(index)
            .map(block_description)
    }

    /// Iterate over Young-irrep blocks in packed-output order.
    pub fn blocks(
        &self,
    ) -> impl ExactSizeIterator<Item = SymmetricGroupFftBlock<'_>> + DoubleEndedIterator + '_ {
        self.levels[self.degree]
            .irreps
            .iter()
            .map(block_description)
    }

    /// Map recursive coset-order indices to lexicographic image-permutation indices.
    pub fn recursive_to_lexicographic(&self) -> &[u32] {
        &self.recursive_to_lexicographic
    }

    /// Return the lexicographic index of the inverse of one indexed permutation.
    ///
    /// This is the convention boundary needed when a raw colour kernel is
    /// stored as `k(g)` but the block contraction requires the transform of
    /// `g -> k(g^-1)`.  The calculation uses fixed-size stack storage and does
    /// not allocate, so callers may build and then discard a compact reorder
    /// vector during setup.
    pub fn inverse_lexicographic_index(&self, index: usize) -> RusticolResult<usize> {
        if index >= self.order {
            return Err(invalid(format!(
                "symmetric-group permutation index {index} exceeds group order {}",
                self.order
            )));
        }
        let permutation = lexicographic_unrank(self.degree, index, &self.factorials)?;
        let mut inverse_code = 0_u64;
        for (position, value) in permutation[..self.degree].iter().copied().enumerate() {
            inverse_code |= (position as u64) << (4 * usize::from(value));
        }
        lexicographic_rank(inverse_code, self.degree, &self.factorials)
    }

    /// Allocate mutable scratch for at most `lane_capacity` innermost lanes.
    pub fn workspace(&self, lane_capacity: usize) -> RusticolResult<SymmetricGroupFftWorkspace> {
        SymmetricGroupFftWorkspace::new(self, lane_capacity)
    }

    /// Transform one complex function on `S_m`.
    pub fn forward(
        &self,
        values: &[SymmetricGroupComplex64],
        coefficients: &mut [SymmetricGroupComplex64],
        workspace: &mut SymmetricGroupFftWorkspace,
    ) -> RusticolResult<()> {
        self.forward_lanes(1, values, coefficients, workspace)
    }

    /// Transform independent lanes stored contiguously for every group element.
    pub fn forward_lanes(
        &self,
        active_lanes: usize,
        values: &[SymmetricGroupComplex64],
        coefficients: &mut [SymmetricGroupComplex64],
        workspace: &mut SymmetricGroupFftWorkspace,
    ) -> RusticolResult<()> {
        if active_lanes == 0 {
            return Err(invalid("symmetric-group FFT lane count must be positive"));
        }
        let value_count = self
            .order
            .checked_mul(active_lanes)
            .ok_or_else(|| invalid("symmetric-group FFT value count overflows usize"))?;
        if values.len() != value_count || coefficients.len() != value_count {
            return Err(invalid(format!(
                "symmetric-group FFT expected {value_count} input and output values, got {} and {}",
                values.len(),
                coefficients.len()
            )));
        }
        workspace.validate(self, active_lanes)?;

        if active_lanes == 1
            && let Some(transform) = self.scalar_degree_three.as_ref()
        {
            transform.forward(values, coefficients);
            return Ok(());
        }

        for (recursive_index, lexicographic_index) in
            self.recursive_to_lexicographic.iter().copied().enumerate()
        {
            let source = usize::try_from(lexicographic_index)
                .map_err(|_| internal("symmetric-group reorder index exceeds usize"))?
                * active_lanes;
            let target = recursive_index * active_lanes;
            workspace.buffer[target..target + active_lanes]
                .copy_from_slice(&values[source..source + active_lanes]);
        }

        let mut current_is_workspace = true;
        for current_degree in 2..=self.degree {
            let level = &self.levels[current_degree];
            let child_level = &self.levels[current_degree - 1];
            if active_lanes == 1 {
                if current_is_workspace {
                    transform_level_scalar(
                        level,
                        child_level,
                        &workspace.buffer[..value_count],
                        coefficients,
                        &mut workspace.block,
                        self.order,
                    );
                } else {
                    transform_level_scalar(
                        level,
                        child_level,
                        coefficients,
                        &mut workspace.buffer[..value_count],
                        &mut workspace.block,
                        self.order,
                    );
                }
            } else if current_is_workspace {
                transform_level_batched(
                    level,
                    child_level,
                    &workspace.buffer[..value_count],
                    coefficients,
                    &mut workspace.block,
                    active_lanes,
                    self.order,
                );
            } else {
                transform_level_batched(
                    level,
                    child_level,
                    coefficients,
                    &mut workspace.buffer[..value_count],
                    &mut workspace.block,
                    active_lanes,
                    self.order,
                );
            }
            current_is_workspace = !current_is_workspace;
        }
        if current_is_workspace {
            coefficients.copy_from_slice(&workspace.buffer[..value_count]);
        }
        Ok(())
    }
}

impl ScalarDegreeThreeTransform {
    #[inline(always)]
    fn forward(
        &self,
        values: &[SymmetricGroupComplex64],
        coefficients: &mut [SymmetricGroupComplex64],
    ) {
        debug_assert_eq!(values.len(), 6);
        debug_assert_eq!(coefficients.len(), 6);
        debug_assert_eq!(self.row_offsets.len(), 7);
        for (output, coefficient) in coefficients.iter_mut().enumerate().take(6) {
            let start = usize::from(self.row_offsets[output]);
            let end = usize::from(self.row_offsets[output + 1]);
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for &(source, factor) in &self.terms[start..end] {
                let value = values[usize::from(source)];
                real = factor.mul_add(value.0, real);
                imaginary = factor.mul_add(value.1, imaginary);
            }
            *coefficient = (real, imaginary);
        }
    }
}

fn compile_scalar_degree_three_transform(
    plan: &SymmetricGroupFftPlan,
) -> RusticolResult<ScalarDegreeThreeTransform> {
    debug_assert_eq!(plan.degree, 3);
    debug_assert_eq!(plan.order, 6);
    debug_assert!(plan.scalar_degree_three.is_none());
    let mut values = vec![(0.0, 0.0); plan.order];
    let mut coefficients = vec![(0.0, 0.0); plan.order];
    let mut workspace = plan.workspace(1)?;
    let mut columns = vec![vec![0.0; plan.order]; plan.order];
    for source in 0..plan.order {
        values[source] = (1.0, 0.0);
        plan.forward(&values, &mut coefficients, &mut workspace)?;
        values[source] = (0.0, 0.0);
        for (output, &(real, imaginary)) in coefficients.iter().enumerate() {
            if imaginary != 0.0 || !real.is_finite() {
                return Err(internal(
                    "compiled scalar S3 Fourier transform is not finite and real",
                ));
            }
            columns[source][output] = real;
        }
    }

    let mut row_offsets = Vec::with_capacity(plan.order + 1);
    let mut terms = Vec::new();
    row_offsets.push(0_u8);
    for output in 0..plan.order {
        for (source, column) in columns.iter().enumerate() {
            let factor = column[output];
            if factor != 0.0 {
                terms.push((source as u8, factor));
            }
        }
        row_offsets.push(
            u8::try_from(terms.len()).map_err(|_| {
                internal("compiled scalar S3 Fourier transform term count exceeds u8")
            })?,
        );
    }
    Ok(ScalarDegreeThreeTransform {
        row_offsets: row_offsets.into_boxed_slice(),
        terms: terms.into_boxed_slice(),
    })
}

fn block_description(irrep: &YoungIrrep) -> SymmetricGroupFftBlock<'_> {
    SymmetricGroupFftBlock {
        shape: &irrep.shape,
        dimension: irrep.dimension,
        coefficient_offset: irrep.coefficient_offset,
    }
}

/// Mutable scratch that can be reused for transforms with the same shape.
#[derive(Debug)]
pub struct SymmetricGroupFftWorkspace {
    degree: usize,
    order: usize,
    lane_capacity: usize,
    buffer: Vec<SymmetricGroupComplex64>,
    block: Vec<SymmetricGroupComplex64>,
}

impl SymmetricGroupFftWorkspace {
    /// Allocate scratch for at most `lane_capacity` independent functions.
    pub fn new(plan: &SymmetricGroupFftPlan, lane_capacity: usize) -> RusticolResult<Self> {
        if lane_capacity == 0 {
            return Err(invalid("symmetric-group FFT lane count must be positive"));
        }
        let buffer_count = plan
            .order
            .checked_mul(lane_capacity)
            .ok_or_else(|| invalid("symmetric-group FFT workspace buffer size overflows usize"))?;
        let block_count = plan
            .maximum_dimension
            .checked_mul(plan.maximum_dimension)
            .and_then(|value| value.checked_mul(lane_capacity))
            .ok_or_else(|| invalid("symmetric-group FFT block workspace size overflows usize"))?;
        Ok(Self {
            degree: plan.degree,
            order: plan.order,
            lane_capacity,
            buffer: zeroed_complex_values(buffer_count, "FFT permutation buffer")?,
            block: zeroed_complex_values(block_count, "FFT block workspace")?,
        })
    }

    /// Maximum number of independent lanes accepted by this workspace.
    pub fn lane_capacity(&self) -> usize {
        self.lane_capacity
    }

    fn validate(&self, plan: &SymmetricGroupFftPlan, active_lanes: usize) -> RusticolResult<()> {
        if self.degree != plan.degree || self.order != plan.order {
            return Err(invalid(
                "symmetric-group FFT workspace does not match the plan",
            ));
        }
        if active_lanes > self.lane_capacity {
            return Err(invalid(format!(
                "symmetric-group FFT requested {active_lanes} active lanes from workspace capacity {}",
                self.lane_capacity
            )));
        }
        Ok(())
    }
}

fn transform_level_batched(
    level: &SymmetricGroupLevel,
    child_level: &SymmetricGroupLevel,
    input: &[SymmetricGroupComplex64],
    output: &mut [SymmetricGroupComplex64],
    block: &mut [SymmetricGroupComplex64],
    lane_count: usize,
    total_order: usize,
) {
    debug_assert_eq!(input.len(), total_order * lane_count);
    debug_assert_eq!(output.len(), total_order * lane_count);
    debug_assert_eq!(level.order, level.degree * child_level.order);
    debug_assert!(block.len() >= level.maximum_dimension.pow(2) * lane_count);

    for group_start in (0..total_order).step_by(level.order) {
        let group_value_start = group_start * lane_count;
        let group_value_end = (group_start + level.order) * lane_count;
        output[group_value_start..group_value_end].fill((0.0, 0.0));

        for coset in 0..level.degree {
            let child_start = group_start + coset * child_level.order;
            for irrep in &level.irreps {
                let dimension = irrep.dimension;
                let block_value_count = dimension * dimension * lane_count;
                let block = &mut block[..block_value_count];
                block.fill((0.0, 0.0));

                for branch in &irrep.branches {
                    let child = &child_level.irreps[branch.child_irrep as usize];
                    let basis_offset = branch.basis_offset as usize;
                    for column in 0..child.dimension {
                        for row in 0..child.dimension {
                            let source_coefficient = child_start
                                + child.coefficient_offset
                                + column * child.dimension
                                + row;
                            let target_coefficient =
                                (basis_offset + column) * dimension + basis_offset + row;
                            let source = source_coefficient * lane_count;
                            let target = target_coefficient * lane_count;
                            block[target..target + lane_count]
                                .copy_from_slice(&input[source..source + lane_count]);
                        }
                    }
                }

                // The coset representative is s_i s_(i+1) ... s_(m-1).
                // Left multiplication therefore applies its rightmost factor first.
                for generator in (coset..level.degree - 1).rev() {
                    apply_generator_on_left(irrep, generator, block, lane_count);
                }

                let output_offset = (group_start + irrep.coefficient_offset) * lane_count;
                for (target, value) in output[output_offset..output_offset + block_value_count]
                    .iter_mut()
                    .zip(block.iter().copied())
                {
                    target.0 += value.0;
                    target.1 += value.1;
                }
            }
        }
    }
}

/// Scalar transforms keep each temporary matrix transposed.  A Young
/// generator then mixes two contiguous rows rather than walking two strided
/// rows of the public column-major output.  The transpose is folded into the
/// branch gather and final accumulation, so it needs no additional storage or
/// pass over the block.
fn transform_level_scalar(
    level: &SymmetricGroupLevel,
    child_level: &SymmetricGroupLevel,
    input: &[SymmetricGroupComplex64],
    output: &mut [SymmetricGroupComplex64],
    block: &mut [SymmetricGroupComplex64],
    total_order: usize,
) {
    debug_assert_eq!(input.len(), total_order);
    debug_assert_eq!(output.len(), total_order);
    debug_assert_eq!(level.order, level.degree * child_level.order);
    debug_assert!(block.len() >= level.maximum_dimension.pow(2));

    for group_start in (0..total_order).step_by(level.order) {
        output[group_start..group_start + level.order].fill((0.0, 0.0));

        for coset in 0..level.degree {
            let child_start = group_start + coset * child_level.order;
            for irrep in &level.irreps {
                let dimension = irrep.dimension;
                let block_value_count = dimension * dimension;
                let block = &mut block[..block_value_count];
                block.fill((0.0, 0.0));

                for branch in &irrep.branches {
                    let child = &child_level.irreps[branch.child_irrep as usize];
                    let basis_offset = branch.basis_offset as usize;
                    for column in 0..child.dimension {
                        for row in 0..child.dimension {
                            let source = child_start
                                + child.coefficient_offset
                                + column * child.dimension
                                + row;
                            // Fold the column-major -> row-major transpose into
                            // the sparse branch embedding.
                            let target = (basis_offset + row) * dimension + basis_offset + column;
                            block[target] = input[source];
                        }
                    }
                }

                // The coset representative is s_i s_(i+1) ... s_(m-1).
                // Left multiplication therefore applies its rightmost factor first.
                for generator in (coset..level.degree - 1).rev() {
                    apply_generator_on_left_scalar_transposed(irrep, generator, block);
                }

                let output_offset = group_start + irrep.coefficient_offset;
                for row in 0..dimension {
                    for column in 0..dimension {
                        let target = output_offset + column * dimension + row;
                        let value = block[row * dimension + column];
                        output[target].0 += value.0;
                        output[target].1 += value.1;
                    }
                }
            }
        }
    }
}

fn apply_generator_on_left(
    irrep: &YoungIrrep,
    generator: usize,
    matrix: &mut [SymmetricGroupComplex64],
    lane_count: usize,
) {
    let dimension = irrep.dimension;
    for first in 0..dimension {
        let first_action = irrep.generator(generator, first);
        let other = first_action.partner as usize;
        if other < first {
            continue;
        }
        if other == first {
            for column in 0..dimension {
                let base = (column * dimension + first) * lane_count;
                for value in &mut matrix[base..base + lane_count] {
                    value.0 *= first_action.diagonal;
                    value.1 *= first_action.diagonal;
                }
            }
            continue;
        }
        let other_action = irrep.generator(generator, other);
        for column in 0..dimension {
            let first_base = (column * dimension + first) * lane_count;
            let other_base = (column * dimension + other) * lane_count;
            for lane in 0..lane_count {
                let first_value = matrix[first_base + lane];
                let other_value = matrix[other_base + lane];
                matrix[first_base + lane] = linear_combination(
                    first_action.diagonal,
                    first_value,
                    first_action.mixing,
                    other_value,
                );
                matrix[other_base + lane] = linear_combination(
                    other_action.mixing,
                    first_value,
                    other_action.diagonal,
                    other_value,
                );
            }
        }
    }
}

fn apply_generator_on_left_scalar_transposed(
    irrep: &YoungIrrep,
    generator: usize,
    matrix: &mut [SymmetricGroupComplex64],
) {
    let dimension = irrep.dimension;
    debug_assert_eq!(matrix.len(), dimension * dimension);
    for first in 0..dimension {
        let first_action = irrep.generator(generator, first);
        let other = first_action.partner as usize;
        if other < first {
            continue;
        }
        if other == first {
            // A fixed Young-basis vector has axial distance +/-1.  The +1
            // action is the identity and is common enough to skip outright.
            if first_action.diagonal == 1.0 {
                continue;
            }
            debug_assert_eq!(first_action.diagonal, -1.0);
            let row = &mut matrix[first * dimension..(first + 1) * dimension];
            for value in row {
                value.0 = -value.0;
                value.1 = -value.1;
            }
            continue;
        }

        let other_action = irrep.generator(generator, other);
        let (before_other, from_other) = matrix.split_at_mut(other * dimension);
        let first_row = &mut before_other[first * dimension..(first + 1) * dimension];
        let other_row = &mut from_other[..dimension];
        for (first_value, other_value) in first_row.iter_mut().zip(other_row) {
            let old_first = *first_value;
            let old_other = *other_value;
            *first_value = linear_combination(
                first_action.diagonal,
                old_first,
                first_action.mixing,
                old_other,
            );
            *other_value = linear_combination(
                other_action.mixing,
                old_first,
                other_action.diagonal,
                old_other,
            );
        }
    }
}

fn linear_combination(
    first_factor: f64,
    first: SymmetricGroupComplex64,
    second_factor: f64,
    second: SymmetricGroupComplex64,
) -> SymmetricGroupComplex64 {
    (
        first_factor * first.0 + second_factor * second.0,
        first_factor * first.1 + second_factor * second.1,
    )
}

fn build_levels(degree: usize, factorials: &[usize]) -> RusticolResult<Vec<SymmetricGroupLevel>> {
    let trivial_tableau: Box<[(u8, u8)]> = Vec::new().into_boxed_slice();
    let mut levels = vec![SymmetricGroupLevel {
        degree: 0,
        order: 1,
        maximum_dimension: 1,
        irreps: vec![YoungIrrep {
            shape: Vec::new().into_boxed_slice(),
            dimension: 1,
            coefficient_offset: 0,
            branches: Vec::new().into_boxed_slice(),
            tableaux: vec![trivial_tableau].into_boxed_slice(),
            generators: Vec::new().into_boxed_slice(),
        }]
        .into_boxed_slice(),
    }];
    for current_degree in 1..=degree {
        let level = build_level(
            current_degree,
            factorials[current_degree],
            &levels[current_degree - 1],
        )?;
        levels.push(level);
    }
    Ok(levels)
}

fn build_level(
    degree: usize,
    order: usize,
    child_level: &SymmetricGroupLevel,
) -> RusticolResult<SymmetricGroupLevel> {
    debug_assert_eq!(child_level.degree + 1, degree);
    let shapes = integer_partitions(degree)?;
    let mut irreps = Vec::new();
    irreps
        .try_reserve_exact(shapes.len())
        .map_err(|error| internal(format!("Young-irrep allocation failed: {error}")))?;
    let mut coefficient_offset = 0_usize;
    let mut maximum_dimension = 1_usize;

    for shape in shapes {
        let removable_rows = (0..shape.len())
            .filter(|row| shape[*row] > shape.get(*row + 1).copied().unwrap_or(0))
            .collect::<Vec<_>>();
        let mut branches = Vec::new();
        let mut tableaux = Vec::new();
        let mut dimension = 0_usize;
        for row in removable_rows {
            let mut child_shape = shape.clone();
            child_shape[row] -= 1;
            if child_shape.last() == Some(&0) {
                child_shape.pop();
            }
            let child_irrep = child_level
                .irreps
                .iter()
                .position(|candidate| candidate.shape.as_ref() == child_shape)
                .ok_or_else(|| internal("Young branching child shape is absent"))?;
            let child = &child_level.irreps[child_irrep];
            let basis_offset = dimension;
            dimension = dimension
                .checked_add(child.dimension)
                .ok_or_else(|| internal("Young-irrep dimension overflows usize"))?;
            branches.push(YoungBranch {
                child_irrep: u32::try_from(child_irrep)
                    .map_err(|_| internal("Young child-irrep index exceeds u32"))?,
                basis_offset: u32::try_from(basis_offset)
                    .map_err(|_| internal("Young branch basis offset exceeds u32"))?,
            });
            for child_tableau in &child.tableaux {
                let mut tableau = child_tableau.to_vec();
                tableau.push((
                    u8::try_from(row).map_err(|_| internal("Young tableau row exceeds u8"))?,
                    shape[row] - 1,
                ));
                tableaux.push(tableau.into_boxed_slice());
            }
        }
        if dimension == 0 || tableaux.len() != dimension {
            return Err(internal("Young branching produced an invalid dimension"));
        }
        let generators = build_young_generators(degree, &tableaux)?;
        let coefficient_count = dimension
            .checked_mul(dimension)
            .ok_or_else(|| internal("Young Fourier block size overflows usize"))?;
        let irrep = YoungIrrep {
            shape: shape.into_boxed_slice(),
            dimension,
            coefficient_offset,
            branches: branches.into_boxed_slice(),
            tableaux: tableaux.into_boxed_slice(),
            generators: generators.into_boxed_slice(),
        };
        coefficient_offset = coefficient_offset
            .checked_add(coefficient_count)
            .ok_or_else(|| internal("Young coefficient count overflows usize"))?;
        maximum_dimension = maximum_dimension.max(dimension);
        irreps.push(irrep);
    }
    if coefficient_offset != order {
        return Err(internal(format!(
            "Young dimensions square to {coefficient_offset}, expected group order {order}"
        )));
    }
    Ok(SymmetricGroupLevel {
        degree,
        order,
        maximum_dimension,
        irreps: irreps.into_boxed_slice(),
    })
}

fn integer_partitions(degree: usize) -> RusticolResult<Vec<Vec<u8>>> {
    fn visit(
        remaining: usize,
        maximum_part: usize,
        shape: &mut Vec<u8>,
        output: &mut Vec<Vec<u8>>,
    ) -> RusticolResult<()> {
        if remaining == 0 {
            output.push(shape.clone());
            return Ok(());
        }
        for part in (1..=remaining.min(maximum_part)).rev() {
            shape.push(
                u8::try_from(part)
                    .map_err(|_| internal("Young partition row length exceeds u8"))?,
            );
            visit(remaining - part, part, shape, output)?;
            shape.pop();
        }
        Ok(())
    }

    let mut output = Vec::new();
    visit(degree, degree, &mut Vec::new(), &mut output)?;
    Ok(output)
}

fn build_young_generators(
    degree: usize,
    tableaux: &[Box<[(u8, u8)]>],
) -> RusticolResult<Vec<YoungGeneratorAction>> {
    if degree <= 1 {
        return Ok(Vec::new());
    }
    let dimension = tableaux.len();
    let action_count = (degree - 1)
        .checked_mul(dimension)
        .ok_or_else(|| internal("Young generator table size overflows usize"))?;
    let mut actions = Vec::new();
    actions
        .try_reserve_exact(action_count)
        .map_err(|error| internal(format!("Young generator allocation failed: {error}")))?;

    let tableau_index = tableaux
        .iter()
        .enumerate()
        .map(|(index, tableau)| (tableau.to_vec(), index))
        .collect::<BTreeMap<_, _>>();
    for generator in 0..degree - 1 {
        for (basis, tableau) in tableaux.iter().enumerate() {
            let first = tableau[generator];
            let second = tableau[generator + 1];
            let first_content = i16::from(first.1) - i16::from(first.0);
            let second_content = i16::from(second.1) - i16::from(second.0);
            let axial_distance = second_content - first_content;
            if axial_distance == 0 {
                return Err(internal("standard Young tableau has zero axial distance"));
            }
            let diagonal = 1.0 / f64::from(axial_distance);
            let partner = if axial_distance.unsigned_abs() == 1 {
                basis
            } else {
                let mut swapped = tableau.to_vec();
                swapped.swap(generator, generator + 1);
                *tableau_index
                    .get(&swapped)
                    .ok_or_else(|| internal("Young generator partner tableau is absent"))?
            };
            actions.push(YoungGeneratorAction {
                partner: u32::try_from(partner)
                    .map_err(|_| internal("Young generator partner exceeds u32"))?,
                diagonal,
                mixing: (1.0 - diagonal * diagonal).max(0.0).sqrt(),
            });
        }
    }
    Ok(actions)
}

fn checked_factorials(degree: usize) -> RusticolResult<Vec<usize>> {
    let mut factorials = Vec::new();
    factorials
        .try_reserve_exact(degree + 1)
        .map_err(|error| invalid(format!("factorial table allocation failed: {error}")))?;
    factorials.push(1_usize);
    for factor in 1..=degree {
        let value = factorials[factor - 1]
            .checked_mul(factor)
            .ok_or_else(|| invalid("symmetric-group order overflows usize"))?;
        if value > u32::MAX as usize {
            return Err(invalid(
                "symmetric-group order exceeds compact reorder-index capacity",
            ));
        }
        factorials.push(value);
    }
    Ok(factorials)
}

fn build_recursive_to_lexicographic(
    degree: usize,
    factorials: &[usize],
) -> RusticolResult<Vec<u32>> {
    if degree == 0 {
        return Ok(vec![0]);
    }
    let mut child_codes = vec![0_u64];
    let mut final_map = Vec::new();
    for current_degree in 1..=degree {
        let order = factorials[current_degree];
        let is_final = current_degree == degree;
        let mut next_codes = Vec::new();
        if is_final {
            final_map.try_reserve_exact(order).map_err(|error| {
                invalid(format!("permutation reorder allocation failed: {error}"))
            })?;
        } else {
            next_codes
                .try_reserve_exact(order)
                .map_err(|error| invalid(format!("permutation-code allocation failed: {error}")))?;
        }
        for coset in 0..current_degree {
            for child_code in child_codes.iter().copied() {
                let code = extend_recursive_permutation(child_code, current_degree, coset);
                if is_final {
                    let rank = lexicographic_rank(code, current_degree, factorials)?;
                    final_map.push(
                        u32::try_from(rank)
                            .map_err(|_| internal("lexicographic rank exceeds u32"))?,
                    );
                } else {
                    next_codes.push(code);
                }
            }
        }
        if !is_final {
            child_codes = next_codes;
        }
    }
    validate_reorder_map(&final_map, factorials[degree])?;
    Ok(final_map)
}

fn extend_recursive_permutation(child: u64, degree: usize, coset: usize) -> u64 {
    debug_assert!((1..=MAX_SYMMETRIC_GROUP_FFT_DEGREE).contains(&degree));
    debug_assert!(coset < degree);
    let mut result = 0_u64;
    for position in 0..degree - 1 {
        let child_value = ((child >> (4 * position)) & 0xf) as usize;
        let value = if child_value < coset {
            child_value
        } else {
            child_value + 1
        };
        result |= (value as u64) << (4 * position);
    }
    result | (coset as u64) << (4 * (degree - 1))
}

fn lexicographic_rank(code: u64, degree: usize, factorials: &[usize]) -> RusticolResult<usize> {
    let mut used = 0_u16;
    let mut rank = 0_usize;
    for position in 0..degree {
        let value = ((code >> (4 * position)) & 0xf) as usize;
        if value >= degree || used & (1_u16 << value) != 0 {
            return Err(internal("recursive permutation code is invalid"));
        }
        let smaller_mask = (1_u16 << value) - 1;
        let used_smaller = (used & smaller_mask).count_ones() as usize;
        let unused_smaller = value - used_smaller;
        rank = rank
            .checked_add(unused_smaller * factorials[degree - position - 1])
            .ok_or_else(|| internal("lexicographic permutation rank overflows usize"))?;
        used |= 1_u16 << value;
    }
    Ok(rank)
}

fn lexicographic_unrank(
    degree: usize,
    mut rank: usize,
    factorials: &[usize],
) -> RusticolResult<[u8; MAX_SYMMETRIC_GROUP_FFT_DEGREE]> {
    if factorials
        .get(degree)
        .copied()
        .is_none_or(|order| rank >= order)
    {
        return Err(invalid("lexicographic permutation rank is out of bounds"));
    }
    let mut available = [0_u8; MAX_SYMMETRIC_GROUP_FFT_DEGREE];
    for (index, value) in available[..degree].iter_mut().enumerate() {
        *value = u8::try_from(index)
            .map_err(|_| internal("permutation value exceeds compact storage"))?;
    }
    let mut available_count = degree;
    let mut permutation = [0_u8; MAX_SYMMETRIC_GROUP_FFT_DEGREE];
    for (position, value) in permutation[..degree].iter_mut().enumerate() {
        let block = factorials[degree - position - 1];
        let choice = rank / block;
        rank %= block;
        if choice >= available_count {
            return Err(internal("lexicographic unrank selected an invalid digit"));
        }
        *value = available[choice];
        available.copy_within(choice + 1..available_count, choice);
        available_count -= 1;
    }
    Ok(permutation)
}

fn validate_reorder_map(values: &[u32], order: usize) -> RusticolResult<()> {
    if values.len() != order {
        return Err(internal("permutation reorder map has the wrong length"));
    }
    let mut seen = Vec::new();
    seen.try_reserve_exact(order)
        .map_err(|error| invalid(format!("reorder validation allocation failed: {error}")))?;
    seen.resize(order, false);
    for value in values.iter().copied() {
        let index = usize::try_from(value)
            .map_err(|_| internal("permutation reorder index exceeds usize"))?;
        if index >= order || std::mem::replace(&mut seen[index], true) {
            return Err(internal("permutation reorder map is not bijective"));
        }
    }
    Ok(())
}

fn zeroed_complex_values(
    count: usize,
    label: &str,
) -> RusticolResult<Vec<SymmetricGroupComplex64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| invalid(format!("{label} allocation failed: {error}")))?;
    values.resize(count, (0.0, 0.0));
    Ok(values)
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn internal(message: impl Into<String>) -> RusticolError {
    RusticolError::internal(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 2.0e-11;

    #[test]
    fn known_young_layouts_and_dimensions() {
        let plan = SymmetricGroupFftPlan::new(4).unwrap();
        let blocks = plan
            .blocks()
            .map(|block| (block.shape().to_vec(), block.dimension()))
            .collect::<Vec<_>>();
        assert_eq!(
            blocks,
            vec![
                (vec![4], 1),
                (vec![3, 1], 3),
                (vec![2, 2], 2),
                (vec![2, 1, 1], 3),
                (vec![1, 1, 1, 1], 1),
            ]
        );
        assert_eq!(
            plan.blocks()
                .map(|block| block.coefficient_count())
                .sum::<usize>(),
            24
        );
    }

    #[test]
    fn transform_matches_direct_definition_through_six() {
        for degree in 0..=6 {
            let plan = SymmetricGroupFftPlan::new(degree).unwrap();
            let values = deterministic_values(plan.order(), 0);
            let mut fast = vec![(0.0, 0.0); plan.order()];
            let mut workspace = plan.workspace(1).unwrap();
            plan.forward(&values, &mut fast, &mut workspace).unwrap();
            let direct = direct_transform(&plan, &values);
            assert_complex_slices_close(&fast, &direct, TOLERANCE);
        }
    }

    #[test]
    fn inverse_kernel_blocks_match_direct_quadratic_form() {
        let plan = SymmetricGroupFftPlan::new(4).unwrap();
        let left = deterministic_values(plan.order(), 2);
        let right = deterministic_values(plan.order(), 11);
        // Deliberately not inversion-symmetric: this catches a silently
        // reversed raw-kernel convention that a diagonal colour kernel cannot.
        let kernel = deterministic_values(plan.order(), 23);
        let inverse_kernel = (0..plan.order())
            .map(|index| kernel[plan.inverse_lexicographic_index(index).unwrap()])
            .collect::<Vec<_>>();

        let mut workspace = plan.workspace(1).unwrap();
        let mut left_fourier = vec![(0.0, 0.0); plan.order()];
        let mut right_fourier = vec![(0.0, 0.0); plan.order()];
        let mut kernel_fourier = vec![(0.0, 0.0); plan.order()];
        plan.forward(&left, &mut left_fourier, &mut workspace)
            .unwrap();
        plan.forward(&right, &mut right_fourier, &mut workspace)
            .unwrap();
        plan.forward(&inverse_kernel, &mut kernel_fourier, &mut workspace)
            .unwrap();

        let direct = direct_relative_kernel_form(&plan, &left, &right, &kernel);
        let blocked =
            fourier_relative_kernel_form(&plan, &left_fourier, &right_fourier, &kernel_fourier);
        let scale = complex_norm(direct).max(1.0);
        assert!(
            complex_norm((blocked.0 - direct.0, blocked.1 - direct.1)) <= 5.0e-11 * scale,
            "inverse-kernel block form {blocked:?} != direct form {direct:?}"
        );
    }

    #[test]
    fn identity_impulse_transforms_to_irrep_identities() {
        for degree in 0..=7 {
            let plan = SymmetricGroupFftPlan::new(degree).unwrap();
            let mut values = vec![(0.0, 0.0); plan.order()];
            values[0] = (1.0, 0.0);
            let mut transformed = vec![(0.0, 0.0); plan.order()];
            let mut workspace = plan.workspace(1).unwrap();
            plan.forward(&values, &mut transformed, &mut workspace)
                .unwrap();
            for block in plan.blocks() {
                for column in 0..block.dimension() {
                    for row in 0..block.dimension() {
                        let value = transformed
                            [block.coefficient_offset() + column * block.dimension() + row];
                        let expected = if row == column { 1.0 } else { 0.0 };
                        assert!((value.0 - expected).abs() <= TOLERANCE);
                        assert!(value.1.abs() <= TOLERANCE);
                    }
                }
            }
        }
    }

    #[test]
    fn young_generators_obey_coxeter_relations() {
        for degree in 2..=7 {
            let plan = SymmetricGroupFftPlan::new(degree).unwrap();
            let level = &plan.levels[degree];
            for irrep in &level.irreps {
                let identity = dense_identity(irrep.dimension);
                let generators = (0..degree - 1)
                    .map(|generator| dense_generator(irrep, generator))
                    .collect::<Vec<_>>();
                for generator in &generators {
                    assert_real_matrices_close(
                        &dense_multiply(generator, generator, irrep.dimension),
                        &identity,
                        TOLERANCE,
                    );
                }
                for generator in 0..degree - 2 {
                    let left = dense_multiply(
                        &dense_multiply(
                            &generators[generator],
                            &generators[generator + 1],
                            irrep.dimension,
                        ),
                        &generators[generator],
                        irrep.dimension,
                    );
                    let right = dense_multiply(
                        &dense_multiply(
                            &generators[generator + 1],
                            &generators[generator],
                            irrep.dimension,
                        ),
                        &generators[generator + 1],
                        irrep.dimension,
                    );
                    assert_real_matrices_close(&left, &right, TOLERANCE);
                }
                for first in 0..degree - 1 {
                    for second in first + 2..degree - 1 {
                        let left = dense_multiply(
                            &generators[first],
                            &generators[second],
                            irrep.dimension,
                        );
                        let right = dense_multiply(
                            &generators[second],
                            &generators[first],
                            irrep.dimension,
                        );
                        assert_real_matrices_close(&left, &right, TOLERANCE);
                    }
                }
            }
        }
    }

    #[test]
    fn parseval_identity_holds_through_s_eight() {
        for degree in 0..=8 {
            let plan = SymmetricGroupFftPlan::new(degree).unwrap();
            let values = deterministic_values(plan.order(), degree);
            let mut transformed = vec![(0.0, 0.0); plan.order()];
            let mut workspace = plan.workspace(1).unwrap();
            plan.forward(&values, &mut transformed, &mut workspace)
                .unwrap();
            let input_norm = values
                .iter()
                .map(|value| value.0 * value.0 + value.1 * value.1)
                .sum::<f64>();
            let fourier_norm = plan
                .blocks()
                .map(|block| {
                    let start = block.coefficient_offset();
                    let end = start + block.coefficient_count();
                    block.dimension() as f64
                        * transformed[start..end]
                            .iter()
                            .map(|value| value.0 * value.0 + value.1 * value.1)
                            .sum::<f64>()
                })
                .sum::<f64>()
                / plan.order() as f64;
            let scale = input_norm.abs().max(1.0);
            assert!((fourier_norm - input_norm).abs() <= 5.0e-11 * scale);
        }
    }

    #[test]
    fn workspace_reuse_does_not_retain_transform_state() {
        let plan = SymmetricGroupFftPlan::new(7).unwrap();
        let first = deterministic_values(plan.order(), 3);
        let second = deterministic_values(plan.order(), 19);
        let mut workspace = plan.workspace(1).unwrap();
        let mut discarded = vec![(0.0, 0.0); plan.order()];
        let mut reused = vec![(0.0, 0.0); plan.order()];
        plan.forward(&first, &mut discarded, &mut workspace)
            .unwrap();
        plan.forward(&second, &mut reused, &mut workspace).unwrap();
        let mut fresh = vec![(0.0, 0.0); plan.order()];
        let mut fresh_workspace = plan.workspace(1).unwrap();
        plan.forward(&second, &mut fresh, &mut fresh_workspace)
            .unwrap();
        assert_complex_slices_close(&reused, &fresh, TOLERANCE);
    }

    #[test]
    fn batched_lanes_match_independent_transforms() {
        let plan = SymmetricGroupFftPlan::new(6).unwrap();
        let lanes = 3;
        let lane_values = (0..lanes)
            .map(|lane| deterministic_values(plan.order(), lane * 7))
            .collect::<Vec<_>>();
        let mut values = vec![(0.0, 0.0); plan.order() * lanes];
        for group_element in 0..plan.order() {
            for lane in 0..lanes {
                values[group_element * lanes + lane] = lane_values[lane][group_element];
            }
        }
        let mut batched = vec![(0.0, 0.0); values.len()];
        let mut workspace = plan.workspace(lanes).unwrap();
        plan.forward_lanes(lanes, &values, &mut batched, &mut workspace)
            .unwrap();
        for lane in 0..lanes {
            let mut expected = vec![(0.0, 0.0); plan.order()];
            let mut scalar_workspace = plan.workspace(1).unwrap();
            plan.forward(&lane_values[lane], &mut expected, &mut scalar_workspace)
                .unwrap();
            let actual = (0..plan.order())
                .map(|coefficient| batched[coefficient * lanes + lane])
                .collect::<Vec<_>>();
            assert_complex_slices_close(&actual, &expected, TOLERANCE);
        }
    }

    #[test]
    fn scalar_transposed_path_matches_batched_path_through_s_seven() {
        for degree in 0..=7 {
            let plan = SymmetricGroupFftPlan::new(degree).unwrap();
            let scalar_values = deterministic_values(plan.order(), degree * 13 + 5);
            let other_values = deterministic_values(plan.order(), degree * 17 + 9);
            let mut scalar = vec![(0.0, 0.0); plan.order()];
            let mut scalar_workspace = plan.workspace(1).unwrap();
            plan.forward(&scalar_values, &mut scalar, &mut scalar_workspace)
                .unwrap();

            let mut batched_values = vec![(0.0, 0.0); plan.order() * 2];
            for group in 0..plan.order() {
                batched_values[group * 2] = scalar_values[group];
                batched_values[group * 2 + 1] = other_values[group];
            }
            let mut batched = vec![(0.0, 0.0); batched_values.len()];
            let mut batched_workspace = plan.workspace(2).unwrap();
            plan.forward_lanes(2, &batched_values, &mut batched, &mut batched_workspace)
                .unwrap();
            let first_lane = batched
                .chunks_exact(2)
                .map(|lanes| lanes[0])
                .collect::<Vec<_>>();
            assert_complex_slices_close(&scalar, &first_lane, TOLERANCE);
        }
    }

    #[test]
    fn capacity_workspace_reuses_active_prefix_without_state_leakage() {
        let plan = SymmetricGroupFftPlan::new(5).unwrap();
        let lane_capacity = 4;
        let mut workspace = plan.workspace(lane_capacity).unwrap();
        let buffer_address = workspace.buffer.as_ptr();
        let block_address = workspace.block.as_ptr();

        assert_eq!(workspace.lane_capacity(), lane_capacity);
        for active_lanes in 1..=lane_capacity {
            let lane_values = (0..active_lanes)
                .map(|lane| deterministic_values(plan.order(), active_lanes * 11 + lane * 7))
                .collect::<Vec<_>>();
            let mut values = vec![(0.0, 0.0); plan.order() * active_lanes];
            for group_element in 0..plan.order() {
                for lane in 0..active_lanes {
                    values[group_element * active_lanes + lane] = lane_values[lane][group_element];
                }
            }

            let mut transformed = vec![(f64::NAN, f64::NAN); values.len()];
            plan.forward_lanes(active_lanes, &values, &mut transformed, &mut workspace)
                .unwrap();

            for lane in 0..active_lanes {
                let mut expected = vec![(0.0, 0.0); plan.order()];
                let mut scalar_workspace = plan.workspace(1).unwrap();
                plan.forward(&lane_values[lane], &mut expected, &mut scalar_workspace)
                    .unwrap();
                let actual = (0..plan.order())
                    .map(|coefficient| transformed[coefficient * active_lanes + lane])
                    .collect::<Vec<_>>();
                assert_complex_slices_close(&actual, &expected, TOLERANCE);
            }

            assert_eq!(workspace.buffer.as_ptr(), buffer_address);
            assert_eq!(workspace.block.as_ptr(), block_address);
            assert_eq!(workspace.lane_capacity(), lane_capacity);
        }

        let values = vec![(0.0, 0.0); plan.order() * (lane_capacity + 1)];
        let mut coefficients = values.clone();
        assert!(
            plan.forward_lanes(
                lane_capacity + 1,
                &values,
                &mut coefficients,
                &mut workspace,
            )
            .is_err()
        );
    }

    #[test]
    fn reorder_map_is_a_lexicographic_bijection() {
        for degree in 0..=8 {
            let plan = SymmetricGroupFftPlan::new(degree).unwrap();
            let mut values = plan.recursive_to_lexicographic().to_vec();
            values.sort_unstable();
            assert_eq!(
                values,
                (0..plan.order() as u32).collect::<Vec<_>>(),
                "degree {degree}"
            );
            for index in 0..plan.order() {
                let inverse = plan.inverse_lexicographic_index(index).unwrap();
                assert_eq!(
                    plan.inverse_lexicographic_index(inverse).unwrap(),
                    index,
                    "inverse-index involution failed at degree {degree}, index {index}"
                );
            }
            assert!(plan.inverse_lexicographic_index(plan.order()).is_err());
        }
    }

    #[test]
    fn degree_ten_is_supported_and_larger_degrees_are_rejected() {
        let plan = SymmetricGroupFftPlan::new(10).unwrap();
        assert_eq!(plan.order(), 3_628_800);
        assert_eq!(plan.recursive_to_lexicographic().len(), plan.order());
        assert_eq!(
            plan.blocks()
                .map(|block| block.coefficient_count())
                .sum::<usize>(),
            plan.order()
        );
        assert!(SymmetricGroupFftPlan::new(11).is_err());
    }

    fn deterministic_values(count: usize, salt: usize) -> Vec<SymmetricGroupComplex64> {
        (0..count)
            .map(|index| {
                let value = (index + salt + 1) as f64;
                (
                    ((value * 0.173).sin() + (value * 0.037).cos()) / (value + 1.0).sqrt(),
                    ((value * 0.113).cos() - (value * 0.071).sin()) / (value + 2.0).sqrt(),
                )
            })
            .collect()
    }

    fn direct_transform(
        plan: &SymmetricGroupFftPlan,
        values: &[SymmetricGroupComplex64],
    ) -> Vec<SymmetricGroupComplex64> {
        let mut result = vec![(0.0, 0.0); plan.order()];
        for (rank, value) in values.iter().copied().enumerate() {
            let permutation = lexicographic_unrank(plan.degree(), rank);
            for (block_index, block) in plan.blocks().enumerate() {
                let representation = representation_matrix(plan, block_index, &permutation);
                for (offset, coefficient) in representation.into_iter().enumerate() {
                    result[block.coefficient_offset() + offset].0 += value.0 * coefficient;
                    result[block.coefficient_offset() + offset].1 += value.1 * coefficient;
                }
            }
        }
        result
    }

    fn direct_relative_kernel_form(
        plan: &SymmetricGroupFftPlan,
        left: &[SymmetricGroupComplex64],
        right: &[SymmetricGroupComplex64],
        kernel: &[SymmetricGroupComplex64],
    ) -> SymmetricGroupComplex64 {
        let permutations = (0..plan.order())
            .map(|index| lexicographic_unrank(plan.degree(), index))
            .collect::<Vec<_>>();
        let mut result = (0.0, 0.0);
        for (left_index, left_value) in left.iter().copied().enumerate() {
            let mut inverse_left = vec![0_usize; plan.degree()];
            for (position, value) in permutations[left_index].iter().copied().enumerate() {
                inverse_left[value] = position;
            }
            for (right_index, right_value) in right.iter().copied().enumerate() {
                let relative = permutations[right_index]
                    .iter()
                    .map(|value| inverse_left[*value])
                    .collect::<Vec<_>>();
                let relative_index = permutation_rank(&relative);
                let bilinear = complex_multiply(
                    complex_multiply((left_value.0, -left_value.1), right_value),
                    kernel[relative_index],
                );
                result.0 += bilinear.0;
                result.1 += bilinear.1;
            }
        }
        result
    }

    fn fourier_relative_kernel_form(
        plan: &SymmetricGroupFftPlan,
        left: &[SymmetricGroupComplex64],
        right: &[SymmetricGroupComplex64],
        kernel: &[SymmetricGroupComplex64],
    ) -> SymmetricGroupComplex64 {
        let mut result = (0.0, 0.0);
        for block in plan.blocks() {
            let dimension = block.dimension();
            let offset = block.coefficient_offset();
            let mut product = vec![(0.0, 0.0); dimension * dimension];
            for column in 0..dimension {
                for inner in 0..dimension {
                    let kernel_value = kernel[offset + column * dimension + inner];
                    for row in 0..dimension {
                        let value =
                            complex_multiply(right[offset + inner * dimension + row], kernel_value);
                        let target = &mut product[column * dimension + row];
                        target.0 += value.0;
                        target.1 += value.1;
                    }
                }
            }
            let mut trace = (0.0, 0.0);
            for (index, right_kernel) in product.into_iter().enumerate() {
                let left_value = left[offset + index];
                let value = complex_multiply((left_value.0, -left_value.1), right_kernel);
                trace.0 += value.0;
                trace.1 += value.1;
            }
            let weight = dimension as f64 / plan.order() as f64;
            result.0 += weight * trace.0;
            result.1 += weight * trace.1;
        }
        result
    }

    fn permutation_rank(permutation: &[usize]) -> usize {
        let factorials = checked_factorials(permutation.len()).unwrap();
        permutation
            .iter()
            .enumerate()
            .map(|(position, value)| {
                let smaller = permutation[position + 1..]
                    .iter()
                    .filter(|other| *other < value)
                    .count();
                smaller * factorials[permutation.len() - position - 1]
            })
            .sum()
    }

    fn complex_multiply(
        left: SymmetricGroupComplex64,
        right: SymmetricGroupComplex64,
    ) -> SymmetricGroupComplex64 {
        (
            left.0 * right.0 - left.1 * right.1,
            left.0 * right.1 + left.1 * right.0,
        )
    }

    fn complex_norm(value: SymmetricGroupComplex64) -> f64 {
        value.0.hypot(value.1)
    }

    fn representation_matrix(
        plan: &SymmetricGroupFftPlan,
        block_index: usize,
        permutation: &[usize],
    ) -> Vec<f64> {
        let irrep = &plan.levels[plan.degree()].irreps[block_index];
        let mut current = (0..plan.degree()).collect::<Vec<_>>();
        let mut matrix = dense_identity(irrep.dimension);
        for position in 0..plan.degree() {
            let mut source = current[position..]
                .iter()
                .position(|value| *value == permutation[position])
                .unwrap()
                + position;
            while source > position {
                let generator = source - 1;
                current.swap(generator, generator + 1);
                apply_dense_generator_on_right(irrep, generator, &mut matrix);
                source -= 1;
            }
        }
        assert_eq!(current, permutation);
        matrix
    }

    fn lexicographic_unrank(degree: usize, mut rank: usize) -> Vec<usize> {
        let factorials = checked_factorials(degree).unwrap();
        let mut available = (0..degree).collect::<Vec<_>>();
        let mut permutation = Vec::with_capacity(degree);
        for position in 0..degree {
            let block = factorials[degree - position - 1];
            let choice = rank / block;
            rank %= block;
            permutation.push(available.remove(choice));
        }
        permutation
    }

    fn dense_generator(irrep: &YoungIrrep, generator: usize) -> Vec<f64> {
        let mut matrix = vec![0.0; irrep.dimension * irrep.dimension];
        for column in 0..irrep.dimension {
            let action = irrep.generator(generator, column);
            matrix[column * irrep.dimension + column] = action.diagonal;
            if action.partner as usize != column {
                matrix[column * irrep.dimension + action.partner as usize] = action.mixing;
            }
        }
        matrix
    }

    fn dense_identity(dimension: usize) -> Vec<f64> {
        let mut matrix = vec![0.0; dimension * dimension];
        for index in 0..dimension {
            matrix[index * dimension + index] = 1.0;
        }
        matrix
    }

    fn apply_dense_generator_on_right(irrep: &YoungIrrep, generator: usize, matrix: &mut [f64]) {
        for first in 0..irrep.dimension {
            let first_action = irrep.generator(generator, first);
            let other = first_action.partner as usize;
            if other < first {
                continue;
            }
            if other == first {
                for row in 0..irrep.dimension {
                    matrix[first * irrep.dimension + row] *= first_action.diagonal;
                }
                continue;
            }
            let other_action = irrep.generator(generator, other);
            for row in 0..irrep.dimension {
                let first_index = first * irrep.dimension + row;
                let other_index = other * irrep.dimension + row;
                let first_value = matrix[first_index];
                let other_value = matrix[other_index];
                matrix[first_index] =
                    first_action.diagonal * first_value + first_action.mixing * other_value;
                matrix[other_index] =
                    other_action.mixing * first_value + other_action.diagonal * other_value;
            }
        }
    }

    fn dense_multiply(first: &[f64], second: &[f64], dimension: usize) -> Vec<f64> {
        let mut product = vec![0.0; dimension * dimension];
        for column in 0..dimension {
            for inner in 0..dimension {
                let second_value = second[column * dimension + inner];
                for row in 0..dimension {
                    product[column * dimension + row] +=
                        first[inner * dimension + row] * second_value;
                }
            }
        }
        product
    }

    fn assert_complex_slices_close(
        actual: &[SymmetricGroupComplex64],
        expected: &[SymmetricGroupComplex64],
        tolerance: f64,
    ) {
        assert_eq!(actual.len(), expected.len());
        let scale = expected
            .iter()
            .map(|value| value.0.abs().max(value.1.abs()))
            .fold(1.0_f64, f64::max);
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual.0 - expected.0).abs() <= tolerance * scale
                    && (actual.1 - expected.1).abs() <= tolerance * scale,
                "complex mismatch at {index}: {actual:?} != {expected:?}, scale {scale}"
            );
        }
    }

    fn assert_real_matrices_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        let scale = expected.iter().copied().map(f64::abs).fold(1.0, f64::max);
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance * scale,
                "matrix mismatch at {index}: {actual} != {expected}"
            );
        }
    }
}
