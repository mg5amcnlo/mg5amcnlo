// SPDX-License-Identifier: 0BSD

use super::super::*;
use super::*;

/// A single-state selector needs only an equality check, not a position table
/// covering every physical helicity or colour per tile. Larger selections
/// retain the existing constant-time dense lookup, including nearly full axes.
enum SelectedAxisPositions<'a> {
    Sparse(&'a [usize]),
    Dense(&'a [Option<usize>]),
}

impl<'a> SelectedAxisPositions<'a> {
    fn new(indices: &'a [usize], complete_count: usize, dense: &'a mut Vec<Option<usize>>) -> Self {
        if indices.len() <= 1 {
            return Self::Sparse(indices);
        }
        dense.clear();
        dense.resize(complete_count, None);
        for (position, index) in indices.iter().copied().enumerate() {
            dense[index] = Some(position);
        }
        Self::Dense(dense)
    }

    #[inline(always)]
    fn position(&self, index: usize) -> Option<usize> {
        match self {
            Self::Sparse(indices) => indices
                .first()
                .filter(|&&selected| selected == index)
                .map(|_| 0),
            Self::Dense(positions) => positions[index],
        }
    }
}

trait AmplitudeSamples: Copy {
    fn value(self, row: usize, output: usize, output_length: usize) -> Complex<f64>;
}

#[derive(Clone, Copy)]
struct RowMajorAmplitudeSamples<'a> {
    values: &'a [Complex<f64>],
    point_count: usize,
    output_length: usize,
}

impl<'a> RowMajorAmplitudeSamples<'a> {
    fn new(
        values: &'a [Complex<f64>],
        point_count: usize,
        output_length: usize,
    ) -> RusticolResult<Self> {
        let expected = point_count.checked_mul(output_length).ok_or_else(|| {
            RusticolError::invalid_argument("generic amplitude output dimensions overflow")
        })?;
        if values.len() != expected {
            return Err(RusticolError::invalid_argument(format!(
                "generic amplitude output buffer has length {}, expected {expected}",
                values.len()
            )));
        }
        Ok(Self {
            values,
            point_count,
            output_length,
        })
    }
}

impl AmplitudeSamples for RowMajorAmplitudeSamples<'_> {
    #[inline(always)]
    fn value(self, row: usize, output: usize, output_length: usize) -> Complex<f64> {
        debug_assert_eq!(output_length, self.output_length);
        debug_assert!(row < self.point_count);
        debug_assert!(output < self.output_length);
        self.values[row * self.output_length + output]
    }
}

#[derive(Clone, Copy)]
struct EagerAmplitudeMajorSamples<'a> {
    values: &'a [crate::EagerComplex64],
    point_count: usize,
    output_length: usize,
}

impl<'a> EagerAmplitudeMajorSamples<'a> {
    fn new(
        values: &'a [crate::EagerComplex64],
        point_count: usize,
        output_length: usize,
    ) -> RusticolResult<Self> {
        let expected = point_count.checked_mul(output_length).ok_or_else(|| {
            RusticolError::invalid_argument("eager amplitude output dimensions overflow")
        })?;
        if values.len() != expected {
            return Err(RusticolError::invalid_argument(format!(
                "eager amplitude output buffer has length {}, expected {expected}",
                values.len()
            )));
        }
        Ok(Self {
            values,
            point_count,
            output_length,
        })
    }
}

impl AmplitudeSamples for EagerAmplitudeMajorSamples<'_> {
    #[inline(always)]
    fn value(self, row: usize, output: usize, output_length: usize) -> Complex<f64> {
        debug_assert_eq!(output_length, self.output_length);
        debug_assert!(row < self.point_count);
        debug_assert!(output < self.output_length);
        // The eager scheduler copies each amplitude's complete point plane,
        // unlike the point-major generic evaluator output.
        let value = self.values[output * self.point_count + row];
        c64(value.re, value.im)
    }
}

impl AmplitudeSamples for crate::direct_arena::DirectAmplitudePlanes<'_> {
    #[inline(always)]
    fn value(self, row: usize, output: usize, _output_length: usize) -> Complex<f64> {
        let (re, im) = self.value_unchecked(output, row);
        c64(re, im)
    }
}

#[inline(always)]
fn sum_real_hermitian_products(left: &[Complex<f64>], right: &[Complex<f64>]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    // Eight independent chains cover the wide repeated-helicity blocks while
    // the four-lane remainder preserves the smaller-block execution shape.
    let mut sums = [0.0; 8];
    let mut index = 0;
    while index + 8 <= left.len() {
        for lane in 0..8 {
            let left_value = left[index + lane];
            let right_value = right[index + lane];
            sums[lane] += left_value.re * right_value.re + left_value.im * right_value.im;
        }
        index += 8;
    }
    while index + 4 <= left.len() {
        for lane in 0..4 {
            let left_value = left[index + lane];
            let right_value = right[index + lane];
            sums[lane] += left_value.re * right_value.re + left_value.im * right_value.im;
        }
        index += 4;
    }
    while index < left.len() {
        let left_value = left[index];
        let right_value = right[index];
        sums[0] += left_value.re * right_value.re + left_value.im * right_value.im;
        index += 1;
    }
    ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]))
}

#[inline(always)]
fn sum_hermitian_products(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
    debug_assert_eq!(left.len(), right.len());
    let mut sum = c64(0.0, 0.0);
    for (left_value, right_value) in left.iter().zip(right) {
        sum.re += left_value.re * right_value.re + left_value.im * right_value.im;
        sum.im += left_value.im * right_value.re - left_value.re * right_value.im;
    }
    sum
}

fn remap_amplitude_outputs(
    batch_size: usize,
    evaluator_output_order: &[usize],
    evaluator_outputs: &[Complex<f64>],
    canonical_outputs: &mut Vec<Complex<f64>>,
) -> RusticolResult<()> {
    let output_length = evaluator_output_order.len();
    let expected = batch_size
        .checked_mul(output_length)
        .ok_or_else(|| RusticolError::invalid_argument("amplitude output length overflows"))?;
    if evaluator_outputs.len() != expected {
        return Err(RusticolError::integrity(
            "amplitude evaluator output mapping has an inconsistent buffer length",
        ));
    }
    canonical_outputs.clear();
    canonical_outputs.resize(expected, c64(0.0, 0.0));
    for row in 0..batch_size {
        let row_offset = row * output_length;
        for (evaluator_output, canonical_output) in
            evaluator_output_order.iter().copied().enumerate()
        {
            canonical_outputs[row_offset + canonical_output] =
                evaluator_outputs[row_offset + evaluator_output];
        }
    }
    Ok(())
}

impl AmplitudeRuntime {
    pub(crate) fn compiled_direct_reduction_footprint(
        &self,
        maximum_resolved_component_count: usize,
        routed: CompiledDirectRoutedReductionFootprint,
    ) -> RusticolResult<CompiledDirectReductionFootprint> {
        let maximum_group_component_count = self
            .raw_sum_groups
            .iter()
            .map(|group| group.indices.len())
            .max()
            .unwrap_or(1);
        let kind = if let Some(replay) = self.color_topology_replay.as_ref() {
            CompiledDirectReducerKind::ColorTopologyReplay {
                physical_group_count: replay.physical_groups.len(),
            }
        } else if let Some(workspace_scalars_per_lane) =
            self.compiled_symmetric_group_workspace_scalars_per_lane()?
        {
            CompiledDirectReducerKind::SymmetricGroup {
                workspace_scalars_per_lane,
            }
        } else if let Some(contraction) = self.color_contraction.as_ref() {
            CompiledDirectReducerKind::Contracted {
                group_count: contraction.group_count,
            }
        } else if self.has_coherent_groups {
            CompiledDirectReducerKind::Coherent
        } else {
            CompiledDirectReducerKind::Plain
        };
        CompiledDirectReductionFootprint::authenticated(
            kind,
            maximum_group_component_count,
            maximum_resolved_component_count,
            routed,
        )
    }

    pub(in crate::engine) fn color_topology_replay_reducer(
        output_length: usize,
        materialized_groups: &[RawSumGroup],
        replay_manifest: &GenericColorTopologyReplayAmplitudeManifest,
        contraction_manifest: &GenericColorContractionManifest,
    ) -> RusticolResult<Self> {
        let raw_sum_groups = materialized_groups
            .iter()
            .map(|group| RawSumGroup {
                id: group.id,
                indices: group.indices.clone(),
                weight: group.weight,
                all_sector_weight: group.all_sector_weight,
                sector_ids: group.sector_ids.clone(),
            })
            .collect::<Vec<_>>();
        let color_topology_replay = build_color_topology_replay_amplitude_runtime(
            Some(replay_manifest),
            Some(contraction_manifest),
            &raw_sum_groups,
        )?
        .ok_or_else(|| {
            RusticolError::integrity("color topology replay reducer lost its amplitude manifest")
        })?;
        Ok(Self {
            output_length,
            raw_sum_weights: vec![1.0; output_length],
            raw_sum_all_sector_weights: vec![1.0; output_length],
            raw_sum_color_sector_ids: vec![None; output_length],
            raw_sum_groups,
            has_coherent_groups: true,
            color_contraction: None,
            color_topology_replay: Some(color_topology_replay),
            input_components: None,
            input_spans: Vec::new(),
            parameter_scratch_f64: Vec::new(),
            evaluator_output_scratch_f64: Vec::new(),
            output_scratch_f64: Vec::new(),
            resolved_source_row_scratch_f64: Vec::new(),
            resolved_target_row_scratch_f64: Vec::new(),
            routed_reduction_scratch: RoutedReductionScratch::default(),
            materialized_helicity_direct_total_plans: Vec::new(),
            materialized_helicity_direct_total_plan_capacity: 0,
            materialized_helicity_direct_total_next_replacement: 0,
            materialized_helicity_direct_default_plan_by_selector_domain: Vec::new(),
            evaluator_output_order: None,
            evaluator: None,
        })
    }

    pub(crate) fn load(
        amplitude_stage: &GenericAmplitudeStageManifest,
        stage: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
    ) -> RusticolResult<Self> {
        let mut runtime = Self::load_reducer_only(amplitude_stage, stage)?;
        runtime.evaluator = Some(EvaluatorGroup::load_from_store(&stage.evaluator, payloads)?);
        Ok(runtime)
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn load_exact_from_plane(
        amplitude_stage: &GenericAmplitudeStageManifest,
        stage: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
    ) -> RusticolResult<Self> {
        let mut runtime = Self::load_reducer_only(amplitude_stage, stage)?;
        runtime.evaluator = Some(EvaluatorGroup::load_exact_from_plane(stage, payloads)?);
        Ok(runtime)
    }

    pub(crate) fn load_reducer_only(
        amplitude_stage: &GenericAmplitudeStageManifest,
        stage: &GenericSerializedStageEvaluatorManifest,
    ) -> RusticolResult<Self> {
        Self::load_reducer_only_with_compiled_color(amplitude_stage, stage, None)
    }

    pub(crate) fn load_reducer_only_with_compiled_color(
        amplitude_stage: &GenericAmplitudeStageManifest,
        stage: &GenericSerializedStageEvaluatorManifest,
        compiled_color: Option<crate::recurrence::RecurrenceColorContraction>,
    ) -> RusticolResult<Self> {
        if stage.stage_kind != "amplitude-roots" {
            return Err(RusticolError::invalid_argument(
                "generic amplitude runtime expected an amplitude-roots stage",
            ));
        }
        let raw_sum_weights = amplitude_stage
            .roots
            .iter()
            .map(|root| root.helicity_weight)
            .collect::<Vec<_>>();
        let raw_sum_all_sector_weights = amplitude_stage
            .roots
            .iter()
            .map(|root| root.all_sector_weight.unwrap_or(root.helicity_weight))
            .collect::<Vec<_>>();
        let raw_sum_color_sector_ids = amplitude_stage
            .roots
            .iter()
            .map(|root| root.color_sector_id)
            .collect::<Vec<_>>();
        let raw_sum_group_ids = amplitude_stage
            .roots
            .iter()
            .map(generic_root_group_id)
            .collect::<RusticolResult<Vec<_>>>()?;
        let has_coherent_groups = raw_sum_group_ids.iter().any(Option::is_some);
        let raw_sum_groups = if has_coherent_groups {
            build_raw_sum_groups(
                amplitude_stage.output_count,
                &raw_sum_weights,
                &raw_sum_all_sector_weights,
                &raw_sum_group_ids,
                &raw_sum_color_sector_ids,
            )?
        } else {
            Vec::new()
        };
        let color_topology_replay = build_color_topology_replay_amplitude_runtime(
            amplitude_stage.color_topology_replay.as_ref(),
            amplitude_stage.color_contraction.as_ref(),
            &raw_sum_groups,
        )?;
        let color_contraction = if color_topology_replay.is_some() {
            if compiled_color.is_some() {
                return Err(RusticolError::compatibility(
                    "compiled symmetric-group FFT diagnostic does not support colour-topology replay",
                ));
            }
            None
        } else if let Some(compiled_color) = compiled_color {
            Some(build_compiled_symmetric_group_color_contraction_runtime(
                amplitude_stage.color_contraction.as_ref(),
                &raw_sum_groups,
                compiled_color,
            )?)
        } else {
            build_color_contraction_runtime(
                amplitude_stage.color_contraction.as_ref(),
                &raw_sum_groups,
            )?
        };
        let (input_components, input_spans) =
            if stage.parameter_layout == "stage-local-value-momentum" {
                let mut map = vec![0usize; stage.parameter_count];
                if let Some(direct) = stage.compiled_plane_arena.as_ref() {
                    for component in &direct.input_bindings {
                        map[component.parameter_index] = component.global_component;
                    }
                } else {
                    for component in &stage.input_components {
                        map[component.parameter_index] = component.global_component;
                    }
                }
                let spans = contiguous_input_spans(&map);
                (Some(map), spans)
            } else {
                (None, Vec::new())
            };
        let mut evaluator_output_order = vec![usize::MAX; stage.output_length];
        let mut canonical_outputs = vec![false; stage.output_length];
        if let Some(direct) = stage.compiled_plane_arena.as_ref() {
            for binding in &direct.output_bindings {
                if binding.arena != "amplitude" {
                    return Err(RusticolError::integrity(
                        "compiled plane-arena amplitude binding names the wrong arena",
                    ));
                }
                let evaluator_output = binding.output_index;
                let canonical_output = binding.component;
                if evaluator_output >= stage.output_length
                    || canonical_output >= stage.output_length
                    || evaluator_output_order[evaluator_output] != usize::MAX
                    || canonical_outputs[canonical_output]
                {
                    return Err(RusticolError::artifact(
                        "amplitude evaluator output mapping is not a permutation",
                    ));
                }
                evaluator_output_order[evaluator_output] = canonical_output;
                canonical_outputs[canonical_output] = true;
            }
        } else {
            for slot in &stage.output_slots {
                let output_len =
                    slot.output_stop
                        .checked_sub(slot.output_start)
                        .ok_or_else(|| {
                            RusticolError::artifact(
                                "amplitude evaluator has an invalid output range",
                            )
                        })?;
                let component_len = slot
                    .component_stop
                    .checked_sub(slot.component_start)
                    .ok_or_else(|| {
                        RusticolError::artifact(
                            "amplitude evaluator has an invalid component range",
                        )
                    })?;
                if output_len != component_len {
                    return Err(RusticolError::artifact(
                        "amplitude evaluator output and component ranges disagree",
                    ));
                }
                for offset in 0..output_len {
                    let evaluator_output = slot.output_start + offset;
                    let canonical_output = slot.component_start + offset;
                    if evaluator_output >= stage.output_length
                        || canonical_output >= stage.output_length
                        || evaluator_output_order[evaluator_output] != usize::MAX
                        || canonical_outputs[canonical_output]
                    {
                        return Err(RusticolError::artifact(
                            "amplitude evaluator output mapping is not a permutation",
                        ));
                    }
                    evaluator_output_order[evaluator_output] = canonical_output;
                    canonical_outputs[canonical_output] = true;
                }
            }
        }
        if evaluator_output_order.contains(&usize::MAX)
            || canonical_outputs.iter().any(|covered| !covered)
        {
            return Err(RusticolError::artifact(
                "amplitude evaluator output mapping is incomplete",
            ));
        }
        let evaluator_output_order = evaluator_output_order
            .iter()
            .enumerate()
            .any(|(index, output)| index != *output)
            .then_some(evaluator_output_order);
        Ok(Self {
            output_length: amplitude_stage.output_count,
            raw_sum_weights,
            raw_sum_all_sector_weights,
            raw_sum_color_sector_ids,
            raw_sum_groups,
            has_coherent_groups,
            color_contraction,
            color_topology_replay,
            input_components,
            input_spans,
            parameter_scratch_f64: Vec::new(),
            evaluator_output_scratch_f64: Vec::new(),
            output_scratch_f64: Vec::new(),
            resolved_source_row_scratch_f64: Vec::new(),
            resolved_target_row_scratch_f64: Vec::new(),
            routed_reduction_scratch: RoutedReductionScratch::default(),
            materialized_helicity_direct_total_plans: Vec::new(),
            materialized_helicity_direct_total_plan_capacity: 0,
            materialized_helicity_direct_total_next_replacement: 0,
            materialized_helicity_direct_default_plan_by_selector_domain: Vec::new(),
            evaluator_output_order,
            evaluator: None,
        })
    }

    pub(crate) fn prepare_compiled_symmetric_group_workspace(
        &mut self,
        lane_capacity: usize,
    ) -> RusticolResult<()> {
        let Some(contraction) = self.color_contraction.as_mut() else {
            return Ok(());
        };
        let Some(symmetric_group) = contraction.symmetric_group.as_mut() else {
            return Ok(());
        };
        let reducer = match symmetric_group.plan.runtime_reducer() {
            Some(crate::recurrence::RuntimeColorContractionReducer::SymmetricGroupFourier(
                reducer,
            )) => reducer,
            _ => {
                return Err(RusticolError::integrity(
                    "compiled symmetric-group FFT payload lost its runtime reducer",
                ));
            }
        };
        symmetric_group.workspace = Some(reducer.workspace(lane_capacity)?);
        Ok(())
    }

    fn compiled_symmetric_group_workspace_scalars_per_lane(&self) -> RusticolResult<Option<usize>> {
        let Some(symmetric_group) = self
            .color_contraction
            .as_ref()
            .and_then(|contraction| contraction.symmetric_group.as_ref())
        else {
            return Ok(None);
        };
        let reducer = match symmetric_group.plan.runtime_reducer() {
            Some(crate::recurrence::RuntimeColorContractionReducer::SymmetricGroupFourier(
                reducer,
            )) => reducer,
            _ => {
                return Err(RusticolError::integrity(
                    "compiled symmetric-group FFT payload lost its runtime reducer",
                ));
            }
        };
        // The largest irrep block has d^2 <= |S_m|. Count split-complex
        // gathered, transformed, permutation, and block workspaces plus the
        // real reduced lane. This conservative authenticated bound controls
        // the reduction tile; the workspace itself is allocated exactly once.
        let complex_values = reducer
            .local_group_count()
            .checked_add(
                reducer
                    .channel_count()
                    .checked_mul(reducer.group_order())
                    .ok_or_else(|| {
                        RusticolError::integrity("compiled symmetric-group workspace overflows")
                    })?,
            )
            .and_then(|value| {
                reducer
                    .group_order()
                    .checked_mul(2)
                    .and_then(|scratch| value.checked_add(scratch))
            })
            .ok_or_else(|| {
                RusticolError::integrity("compiled symmetric-group workspace overflows")
            })?;
        Ok(Some(
            complex_values
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| {
                    RusticolError::integrity("compiled symmetric-group scalar footprint overflows")
                })?,
        ))
    }

    pub(crate) fn color_topology_replay_mappings(&self) -> Option<Vec<Vec<(usize, usize)>>> {
        self.color_topology_replay.as_ref().map(|replay| {
            replay
                .mappings
                .iter()
                .map(|mapping| mapping.label_permutation.clone())
                .collect()
        })
    }

    pub(crate) fn begin_color_topology_replay(&mut self, batch_size: usize) -> RusticolResult<()> {
        let replay = self.color_topology_replay.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "color topology replay was requested without its amplitude gather",
            )
        })?;
        let scalar_count = batch_size
            .checked_mul(replay.physical_groups.len())
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "color topology replay physical amplitude shape overflows usize",
                )
            })?;
        replay
            .physical_group_scratch_f64
            .resize(scalar_count, c64(0.0, 0.0));
        replay.physical_group_scratch_f64.fill(c64(0.0, 0.0));
        replay.covered_groups.fill(false);
        Ok(())
    }

    pub(crate) fn gather_color_topology_replay_mapping(
        &mut self,
        batch_size: usize,
        mapping_index: usize,
    ) -> RusticolResult<()> {
        let output_scratch = std::mem::take(&mut self.output_scratch_f64);
        let result = (|| {
            let samples =
                RowMajorAmplitudeSamples::new(&output_scratch, batch_size, self.output_length)?;
            self.gather_color_topology_replay_samples(samples, batch_size, 0, mapping_index)
        })();
        self.output_scratch_f64 = output_scratch;
        result
    }

    pub(in crate::engine) fn gather_color_topology_replay_eager_amplitudes(
        &mut self,
        amplitudes: &[crate::EagerComplex64],
        batch_size: usize,
        mapping_index: usize,
    ) -> RusticolResult<()> {
        let samples = EagerAmplitudeMajorSamples::new(amplitudes, batch_size, self.output_length)?;
        self.gather_color_topology_replay_samples(samples, batch_size, 0, mapping_index)
    }

    pub(crate) fn gather_color_topology_replay_planes(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        target_row_start: usize,
        mapping_index: usize,
    ) -> RusticolResult<()> {
        let plane_count = usize::try_from(amplitudes.component_count()?)
            .map_err(|_| RusticolError::invalid_argument("amplitude plane count exceeds usize"))?;
        if plane_count != self.output_length {
            return Err(RusticolError::integrity(
                "color topology replay direct amplitude width is inconsistent",
            ));
        }
        self.gather_color_topology_replay_samples(
            amplitudes,
            amplitudes.point_count() as usize,
            target_row_start,
            mapping_index,
        )
    }

    fn gather_color_topology_replay_samples<S: AmplitudeSamples>(
        &mut self,
        amplitudes: S,
        batch_size: usize,
        target_row_start: usize,
        mapping_index: usize,
    ) -> RusticolResult<()> {
        let replay = self.color_topology_replay.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "color topology replay was requested without its amplitude gather",
            )
        })?;
        let mapping = replay.mappings.get(mapping_index).ok_or_else(|| {
            RusticolError::integrity(
                "color topology replay requested an unknown external-label mapping",
            )
        })?;
        let physical_group_count = replay.physical_groups.len();
        let physical_row_count = replay
            .physical_group_scratch_f64
            .len()
            .checked_div(physical_group_count)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "color topology replay physical amplitude scratch is invalid",
                )
            })?;
        if target_row_start
            .checked_add(batch_size)
            .is_none_or(|stop| stop > physical_row_count)
        {
            return Err(RusticolError::integrity(
                "color topology replay target row range exceeds its physical scratch",
            ));
        }
        for route in &mapping.group_routes {
            if target_row_start == 0 {
                if std::mem::replace(&mut replay.covered_groups[route.target_group_index], true) {
                    return Err(RusticolError::integrity(
                        "color topology replay attempted to materialize a physical group twice",
                    ));
                }
            } else if !replay.covered_groups[route.target_group_index] {
                return Err(RusticolError::integrity(
                    "color topology replay direct tiles arrived out of order",
                ));
            }
            let source_group = &self.raw_sum_groups[route.source_group_index];
            for row in 0..batch_size {
                let mut value = c64(0.0, 0.0);
                for output_index in &source_group.indices {
                    value += amplitudes.value(row, *output_index, self.output_length);
                }
                replay.physical_group_scratch_f64
                    [(target_row_start + row) * physical_group_count + route.target_group_index] =
                    value * route.factor;
            }
        }
        Ok(())
    }

    pub(crate) fn reduce_color_topology_replay_f64_into(
        &mut self,
        batch_size: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let replay = self.color_topology_replay.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "color topology replay was requested without its amplitude gather",
            )
        })?;
        if replay.covered_groups.iter().any(|covered| !covered) {
            return Err(RusticolError::integrity(
                "color topology replay did not materialize every physical amplitude group",
            ));
        }
        let samples = RowMajorAmplitudeSamples::new(
            &replay.physical_group_scratch_f64,
            batch_size,
            replay.physical_groups.len(),
        )?;
        Self::reduce_amplitude_samples_f64_into_selected_slice(
            samples,
            batch_size,
            output,
            None,
            replay.physical_groups.len(),
            &replay.unit_weights,
            &replay.unit_weights,
            &replay.no_sector_ids,
            &replay.physical_groups,
            true,
            &mut replay.color_contraction,
        )
    }

    pub(crate) fn reduce_color_topology_replay_f64_resolved(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let replay = self.color_topology_replay.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "color topology replay was requested without its amplitude gather",
            )
        })?;
        if replay.covered_groups.iter().any(|covered| !covered) {
            return Err(RusticolError::integrity(
                "color topology replay did not materialize every physical amplitude group",
            ));
        }
        if !physics.has_contracted_color_axis() {
            return Err(RusticolError::artifact(
                "color topology replay resolved reduction requires contracted color",
            ));
        }
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        if color_indices.iter().any(|index| *index != 0) {
            return Err(RusticolError::selector(
                "color topology replay exposes one contracted color component",
            ));
        }
        let mut selected_positions = vec![None; physics.manifest.helicities.len()];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            selected_positions[index] = Some(position);
        }
        let index_by_values = physics
            .manifest
            .helicities
            .iter()
            .map(|helicity| (helicity.values.clone(), helicity.index))
            .collect::<BTreeMap<_, _>>();
        let mut members_by_group = Vec::with_capacity(replay.physical_groups.len());
        for (group, representative) in replay
            .physical_groups
            .iter()
            .zip(&replay.physical_group_helicities)
        {
            let mut members = vec![representative.clone()];
            if group.weight > 1.0 {
                if group.weight.to_bits() != 2.0f64.to_bits() {
                    return Err(RusticolError::artifact(
                        "color topology replay has an unsupported helicity reuse weight",
                    ));
                }
                let flipped = representative
                    .iter()
                    .map(|value| -*value)
                    .collect::<Vec<_>>();
                if flipped != *representative {
                    members.push(flipped);
                }
            }
            let mut indexed = Vec::with_capacity(members.len());
            let mut total_weight = 0.0;
            for member in members {
                let index = index_by_values.get(&member).copied().ok_or_else(|| {
                    RusticolError::artifact(
                        "color topology replay physical helicity is absent from public metadata",
                    )
                })?;
                let weight = physics.manifest.helicities[index].coefficient;
                total_weight += weight;
                indexed.push((index, weight));
            }
            if !total_weight.is_finite() || total_weight <= 0.0 {
                return Err(RusticolError::artifact(
                    "color topology replay physical helicity has no positive weight",
                ));
            }
            for (_, weight) in &mut indexed {
                *weight /= total_weight;
            }
            members_by_group.push(indexed);
        }
        let contraction = replay.color_contraction.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "color topology replay lost its authenticated color contraction",
            )
        })?;
        let entries = contraction.logical_entries().collect::<Vec<_>>();
        for entry in &entries {
            if members_by_group[entry.left_group_index] != members_by_group[entry.right_group_index]
            {
                return Err(RusticolError::artifact(
                    "color contraction mixes distinct replayed physical helicities",
                ));
            }
        }
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        let mut values = vec![0.0; batch_size * component_count];
        let group_count = replay.physical_groups.len();
        for row in 0..batch_size {
            let group_row = row * group_count;
            for entry in &entries {
                let left = replay.physical_group_scratch_f64[group_row + entry.left_group_index];
                let right = replay.physical_group_scratch_f64[group_row + entry.right_group_index];
                let product = left * right.conj();
                let contribution = normalization_factor
                    * entry.symmetry_factor
                    * (entry.weight_re * product.re - entry.weight_im * product.im);
                for (helicity_index, weight) in &members_by_group[entry.left_group_index] {
                    let Some(position) = selected_positions[*helicity_index] else {
                        continue;
                    };
                    values[row * component_count + position * color_indices.len()] +=
                        contribution * weight;
                }
            }
        }
        Ok(ResolvedValues {
            values,
            point_count: batch_size,
            helicity_indices,
            color_indices,
        })
    }

    pub(crate) fn evaluate_f64_into_scratch(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
    ) -> RusticolResult<()> {
        let evaluator = self.evaluator.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "compiled plane-arena reducer cannot execute a dense amplitude evaluator",
            )
        })?;
        let evaluator_params = if let Some(input_components) = self.input_components.as_ref() {
            let local_parameter_count = input_components.len();
            let global_parameter_count = state.len().checked_div(batch_size).ok_or_else(|| {
                RusticolError::invalid_argument("generic amplitude batch size is zero")
            })?;
            self.parameter_scratch_f64
                .resize(batch_size * local_parameter_count, c64(0.0, 0.0));
            for row in 0..batch_size {
                let row_state = row * global_parameter_count;
                let row_params = row * local_parameter_count;
                if self.input_spans.is_empty() {
                    for (local_index, global_index) in input_components.iter().enumerate() {
                        self.parameter_scratch_f64[row_params + local_index] =
                            state[row_state + *global_index];
                    }
                } else {
                    for (local_start, global_start, len) in &self.input_spans {
                        let target_start = row_params + *local_start;
                        let source_start = row_state + *global_start;
                        self.parameter_scratch_f64[target_start..target_start + *len]
                            .copy_from_slice(&state[source_start..source_start + *len]);
                    }
                }
            }
            self.parameter_scratch_f64.as_slice()
        } else {
            state
        };
        if let Some(order) = self.evaluator_output_order.as_deref() {
            evaluator.evaluate_batch_into(
                batch_size,
                evaluator_params,
                &mut self.evaluator_output_scratch_f64,
            )?;
            remap_amplitude_outputs(
                batch_size,
                order,
                &self.evaluator_output_scratch_f64,
                &mut self.output_scratch_f64,
            )
        } else {
            evaluator.evaluate_batch_into(
                batch_size,
                evaluator_params,
                &mut self.output_scratch_f64,
            )
        }
    }

    pub(crate) fn evaluate_f64_into_scratch_profile(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
    ) -> RusticolResult<AmplitudeEvaluationProfile> {
        let evaluator_runtime = self.evaluator.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "compiled plane-arena reducer cannot profile a dense amplitude evaluator",
            )
        })?;
        let mut input_pack_elapsed = Duration::ZERO;
        let input_copy_component_count;
        let mut scratch_reallocation_count = 0;
        let evaluator;
        let evaluator_params;
        if let Some(input_components) = self.input_components.as_ref() {
            let local_parameter_count = input_components.len();
            let global_parameter_count = state.len().checked_div(batch_size).ok_or_else(|| {
                RusticolError::invalid_argument("generic amplitude batch size is zero")
            })?;
            let pack_start = Instant::now();
            let capacity = self.parameter_scratch_f64.capacity();
            self.parameter_scratch_f64
                .resize(batch_size * local_parameter_count, c64(0.0, 0.0));
            scratch_reallocation_count +=
                u64::from(self.parameter_scratch_f64.capacity() != capacity);
            for row in 0..batch_size {
                let row_state = row * global_parameter_count;
                let row_params = row * local_parameter_count;
                if self.input_spans.is_empty() {
                    for (local_index, global_index) in input_components.iter().enumerate() {
                        self.parameter_scratch_f64[row_params + local_index] =
                            state[row_state + *global_index];
                    }
                } else {
                    for (local_start, global_start, len) in &self.input_spans {
                        let target_start = row_params + *local_start;
                        let source_start = row_state + *global_start;
                        self.parameter_scratch_f64[target_start..target_start + *len]
                            .copy_from_slice(&state[source_start..source_start + *len]);
                    }
                }
            }
            // Generated evaluators may use the platform floating-point ABI
            // aggressively. Keep timing state in integer-backed Duration
            // values until the generated call has returned.
            input_pack_elapsed = pack_start.elapsed();
            input_copy_component_count = batch_size * local_parameter_count;
            evaluator_params = self.parameter_scratch_f64.as_slice();
        } else {
            input_copy_component_count = 0;
            evaluator_params = state;
        }
        let mut output_remap_elapsed = Duration::ZERO;
        if let Some(order) = self.evaluator_output_order.as_deref() {
            evaluator = evaluator_runtime.evaluate_batch_into_profile(
                batch_size,
                evaluator_params,
                &mut self.evaluator_output_scratch_f64,
                false,
            )?;
            let remap_start = Instant::now();
            remap_amplitude_outputs(
                batch_size,
                order,
                &self.evaluator_output_scratch_f64,
                &mut self.output_scratch_f64,
            )?;
            output_remap_elapsed = remap_start.elapsed();
        } else {
            evaluator = evaluator_runtime.evaluate_batch_into_profile(
                batch_size,
                evaluator_params,
                &mut self.output_scratch_f64,
                false,
            )?;
        }
        let input_pack_s = profile_duration_seconds(input_pack_elapsed);
        Ok(AmplitudeEvaluationProfile {
            input_pack_s,
            evaluator,
            output_remap_s: profile_duration_seconds(output_remap_elapsed),
            input_copy_component_count: input_copy_component_count as u64,
            output_remap_component_count: if self.evaluator_output_order.is_some() {
                (batch_size * self.output_length) as u64
            } else {
                0
            },
            scratch_reallocation_count,
        })
    }

    pub(crate) fn evaluate_active_chunks_f64_into_scratch(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<()> {
        let evaluator = self.evaluator.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "compiled plane-arena reducer cannot execute selected dense amplitude chunks",
            )
        })?;
        if let Some(order) = self.evaluator_output_order.as_deref() {
            evaluator.evaluate_selected_chunks_f64_into_output(
                batch_size,
                state,
                self.input_components.as_deref(),
                &self.input_spans,
                &mut self.evaluator_output_scratch_f64,
                active_chunk_indices,
            )?;
            remap_amplitude_outputs(
                batch_size,
                order,
                &self.evaluator_output_scratch_f64,
                &mut self.output_scratch_f64,
            )
        } else {
            evaluator.evaluate_selected_chunks_f64_into_output(
                batch_size,
                state,
                self.input_components.as_deref(),
                &self.input_spans,
                &mut self.output_scratch_f64,
                active_chunk_indices,
            )
        }
    }

    pub(crate) fn evaluate_active_chunks_f64_into_scratch_profile(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<AmplitudeEvaluationProfile> {
        let evaluator_runtime = self.evaluator.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "compiled plane-arena reducer cannot profile selected dense amplitude chunks",
            )
        })?;
        let evaluator;
        let mut output_remap_elapsed = Duration::ZERO;
        if let Some(order) = self.evaluator_output_order.as_deref() {
            evaluator = evaluator_runtime.evaluate_selected_chunks_f64_into_output_profile(
                batch_size,
                state,
                self.input_components.as_deref(),
                &self.input_spans,
                &mut self.evaluator_output_scratch_f64,
                active_chunk_indices,
            )?;
            let remap_start = Instant::now();
            remap_amplitude_outputs(
                batch_size,
                order,
                &self.evaluator_output_scratch_f64,
                &mut self.output_scratch_f64,
            )?;
            output_remap_elapsed = remap_start.elapsed();
        } else {
            evaluator = evaluator_runtime.evaluate_selected_chunks_f64_into_output_profile(
                batch_size,
                state,
                self.input_components.as_deref(),
                &self.input_spans,
                &mut self.output_scratch_f64,
                active_chunk_indices,
            )?;
        }
        Ok(AmplitudeEvaluationProfile {
            input_pack_s: evaluator.leaf_input_pack_s,
            output_remap_s: profile_duration_seconds(output_remap_elapsed),
            // Active chunks compose the amplitude input and leaf maps, so
            // the evaluator leaf gather is the only actual input copy.
            input_copy_component_count: 0,
            output_remap_component_count: if self.evaluator_output_order.is_some() {
                (batch_size * self.output_length) as u64
            } else {
                0
            },
            scratch_reallocation_count: 0,
            evaluator,
        })
    }

    pub(crate) fn reduce_scratch_f64_into_selected(
        &mut self,
        batch_size: usize,
        raw_sums: &mut Vec<f64>,
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<()> {
        raw_sums.clear();
        raw_sums.resize(batch_size, 0.0);
        self.reduce_scratch_f64_into_selected_slice(batch_size, raw_sums, selected_color_sector_ids)
    }

    pub(crate) fn reduce_scratch_f64_into_selected_slice(
        &mut self,
        batch_size: usize,
        raw_sums: &mut [f64],
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<()> {
        let samples = RowMajorAmplitudeSamples::new(
            &self.output_scratch_f64,
            batch_size,
            self.output_length,
        )?;
        Self::reduce_amplitude_samples_f64_into_selected_slice(
            samples,
            batch_size,
            raw_sums,
            selected_color_sector_ids,
            self.output_length,
            &self.raw_sum_weights,
            &self.raw_sum_all_sector_weights,
            &self.raw_sum_color_sector_ids,
            &self.raw_sum_groups,
            self.has_coherent_groups,
            &mut self.color_contraction,
        )
    }

    /// Reduce canonical split-complex amplitude planes without packing them
    /// into the legacy point-major complex buffer.
    #[allow(dead_code)]
    pub(crate) fn reduce_planes_f64_into_selected_slice(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        raw_sums: &mut [f64],
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<()> {
        let plane_count = usize::try_from(amplitudes.component_count()?)
            .map_err(|_| RusticolError::invalid_argument("amplitude plane count exceeds usize"))?;
        if plane_count != self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "direct amplitude plane count is {plane_count}, expected {}",
                self.output_length
            )));
        }
        let batch_size = amplitudes.point_count() as usize;
        if raw_sums.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "generic reduction output has length {}, expected {batch_size}",
                raw_sums.len()
            )));
        }
        raw_sums.fill(0.0);
        let scratch = &mut self.routed_reduction_scratch;
        if let Some(contraction) = self.color_contraction.as_mut() {
            if selected_color_sector_ids.is_some() {
                return Err(RusticolError::invalid_argument(
                    "LC color-sector runtime selection is only supported for leading-colour diagonal artifacts",
                ));
            }
            if self.raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            if let Some(symmetric_group) = contraction.symmetric_group.as_mut() {
                return reduce_compiled_symmetric_group_planes(
                    amplitudes,
                    raw_sums,
                    &self.raw_sum_groups,
                    symmetric_group,
                );
            }
            let group_scalar_count =
                contraction
                    .group_count
                    .checked_mul(batch_size)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "plane-native color-contraction scratch shape overflows usize",
                        )
                    })?;
            scratch.direct_group_re.resize(group_scalar_count, 0.0);
            scratch.direct_group_re.fill(0.0);
            scratch.direct_group_im.resize(group_scalar_count, 0.0);
            scratch.direct_group_im.fill(0.0);
            for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                let group_start = group_index * batch_size;
                for output_index in &group.indices {
                    let (plane_re, plane_im) = amplitudes.plane_unchecked(*output_index);
                    for point in 0..batch_size {
                        scratch.direct_group_re[group_start + point] += plane_re[point];
                        scratch.direct_group_im[group_start + point] += plane_im[point];
                    }
                }
            }
            for entry in contraction.logical_entries() {
                let left_start = entry.left_group_index * batch_size;
                let right_start = entry.right_group_index * batch_size;
                for (point, raw_sum) in raw_sums.iter_mut().enumerate() {
                    let left_re = scratch.direct_group_re[left_start + point];
                    let left_im = scratch.direct_group_im[left_start + point];
                    let right_re = scratch.direct_group_re[right_start + point];
                    let right_im = scratch.direct_group_im[right_start + point];
                    let product_re = left_re * right_re + left_im * right_im;
                    let product_im = left_im * right_re - left_re * right_im;
                    *raw_sum += entry.symmetry_factor
                        * (entry.weight_re * product_re - entry.weight_im * product_im);
                }
            }
            return Ok(());
        }

        scratch.direct_group_re.resize(batch_size, 0.0);
        scratch.direct_group_im.resize(batch_size, 0.0);
        if self.has_coherent_groups {
            for group in &self.raw_sum_groups {
                if !raw_sum_group_is_selected(group, selected_color_sector_ids) {
                    continue;
                }
                scratch.direct_group_re.fill(0.0);
                scratch.direct_group_im.fill(0.0);
                for index in &group.indices {
                    let (plane_re, plane_im) = amplitudes.plane_unchecked(*index);
                    for point in 0..batch_size {
                        scratch.direct_group_re[point] += plane_re[point];
                        scratch.direct_group_im[point] += plane_im[point];
                    }
                }
                let weight = if selected_color_sector_ids.is_none() {
                    group.all_sector_weight
                } else {
                    group.weight
                };
                for (point, raw_sum) in raw_sums.iter_mut().enumerate() {
                    let re = scratch.direct_group_re[point];
                    let im = scratch.direct_group_im[point];
                    *raw_sum += weight * (re * re + im * im);
                }
            }
            return Ok(());
        }
        for index in 0..self.output_length {
            if !raw_sum_index_is_selected(
                self.raw_sum_color_sector_ids.get(index).copied().flatten(),
                selected_color_sector_ids,
            ) {
                continue;
            }
            let (plane_re, plane_im) = amplitudes.plane_unchecked(index);
            let weight = if selected_color_sector_ids.is_none() {
                self.raw_sum_all_sector_weights[index]
            } else {
                self.raw_sum_weights[index]
            };
            for point in 0..batch_size {
                raw_sums[point] += weight
                    * (plane_re[point] * plane_re[point] + plane_im[point] * plane_im[point]);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_amplitude_samples_f64_into_selected_slice<S: AmplitudeSamples>(
        amplitudes: S,
        batch_size: usize,
        raw_sums: &mut [f64],
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
        output_length: usize,
        raw_sum_weights: &[f64],
        raw_sum_all_sector_weights: &[f64],
        raw_sum_color_sector_ids: &[Option<i64>],
        raw_sum_groups: &[RawSumGroup],
        has_coherent_groups: bool,
        color_contraction: &mut Option<ColorContractionRuntime>,
    ) -> RusticolResult<()> {
        if raw_sums.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "generic reduction output has length {}, expected {batch_size}",
                raw_sums.len()
            )));
        }
        raw_sums.fill(0.0);
        if let Some(contraction) = color_contraction.as_mut() {
            if selected_color_sector_ids.is_some() {
                return Err(RusticolError::invalid_argument(
                    "LC color-sector runtime selection is only supported for leading-colour diagonal artifacts",
                ));
            }
            if raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            if let Some(repeated_block) = contraction.repeated_block.as_ref() {
                contraction
                    .group_scratch_f64
                    .resize(contraction.group_count, c64(0.0, 0.0));
                for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
                    if let (Some(walsh_block), Some(output_indices)) = (
                        repeated_block.walsh_block.as_ref(),
                        repeated_block.singleton_output_indices.as_deref(),
                    ) {
                        let component_count = repeated_block.component_count;
                        let coset_count = walsh_block.cosets.len();
                        for (coset_index, coset) in walsh_block.cosets.iter().enumerate() {
                            for component_index in 0..component_count {
                                let source = |local_group_index: usize| {
                                    amplitudes.value(
                                        row,
                                        output_indices
                                            [local_group_index * component_count + component_index],
                                        output_length,
                                    )
                                };
                                let x0 = source(coset[0]);
                                let x1 = source(coset[1]);
                                let x2 = source(coset[2]);
                                let x3 = source(coset[3]);
                                let sum01 = x0 + x1;
                                let difference01 = x0 - x1;
                                let sum23 = x2 + x3;
                                let difference23 = x2 - x3;
                                let targets = [
                                    (sum01 + sum23) * 0.5,
                                    (difference01 + difference23) * 0.5,
                                    (sum01 - sum23) * 0.5,
                                    (difference01 - difference23) * 0.5,
                                ];
                                for (character_index, value) in targets.into_iter().enumerate() {
                                    contraction.group_scratch_f64[((character_index
                                        * coset_count
                                        + coset_index)
                                        * component_count)
                                        + component_index] = value;
                                }
                            }
                        }
                        for entry in &walsh_block.entries {
                            let left_start = entry.left_group_index * component_count;
                            let right_start = entry.right_group_index * component_count;
                            let product_re = sum_real_hermitian_products(
                                &contraction.group_scratch_f64
                                    [left_start..left_start + component_count],
                                &contraction.group_scratch_f64
                                    [right_start..right_start + component_count],
                            );
                            *raw_sum += entry.symmetry_factor * entry.weight_re * product_re;
                        }
                        continue;
                    }
                    if let (Some(walsh_block), Some(output_indices)) = (
                        repeated_block.c2k_walsh_block.as_ref(),
                        repeated_block.singleton_output_indices.as_deref(),
                    ) {
                        let component_count = repeated_block.component_count;
                        let coset_count = walsh_block.cosets.len();
                        if walsh_block.subgroup_order == 8 {
                            for (coset_index, coset) in walsh_block.cosets.iter().enumerate() {
                                for component_index in 0..component_count {
                                    let source = |local_group_index: usize| {
                                        amplitudes.value(
                                            row,
                                            output_indices[local_group_index * component_count
                                                + component_index],
                                            output_length,
                                        )
                                    };
                                    let x0 = source(coset[0]);
                                    let x1 = source(coset[1]);
                                    let x2 = source(coset[2]);
                                    let x3 = source(coset[3]);
                                    let x4 = source(coset[4]);
                                    let x5 = source(coset[5]);
                                    let x6 = source(coset[6]);
                                    let x7 = source(coset[7]);
                                    let sum01 = x0 + x1;
                                    let difference01 = x0 - x1;
                                    let sum23 = x2 + x3;
                                    let difference23 = x2 - x3;
                                    let sum45 = x4 + x5;
                                    let difference45 = x4 - x5;
                                    let sum67 = x6 + x7;
                                    let difference67 = x6 - x7;
                                    let lower = [
                                        sum01 + sum23,
                                        difference01 + difference23,
                                        sum01 - sum23,
                                        difference01 - difference23,
                                    ];
                                    let upper = [
                                        sum45 + sum67,
                                        difference45 + difference67,
                                        sum45 - sum67,
                                        difference45 - difference67,
                                    ];
                                    let targets = [
                                        lower[0] + upper[0],
                                        lower[1] + upper[1],
                                        lower[2] + upper[2],
                                        lower[3] + upper[3],
                                        lower[0] - upper[0],
                                        lower[1] - upper[1],
                                        lower[2] - upper[2],
                                        lower[3] - upper[3],
                                    ];
                                    for (character_index, value) in targets.into_iter().enumerate()
                                    {
                                        contraction.group_scratch_f64[((character_index
                                            * coset_count
                                            + coset_index)
                                            * component_count)
                                            + component_index] = value;
                                    }
                                }
                            }
                        } else {
                            for (coset_index, coset) in walsh_block.cosets.iter().enumerate() {
                                for component_index in 0..component_count {
                                    for (subgroup_index, local_group_index) in
                                        coset.iter().copied().enumerate()
                                    {
                                        let target = ((subgroup_index * coset_count + coset_index)
                                            * component_count)
                                            + component_index;
                                        contraction.group_scratch_f64[target] = amplitudes.value(
                                            row,
                                            output_indices[local_group_index * component_count
                                                + component_index],
                                            output_length,
                                        );
                                    }
                                    let mut stride = 1;
                                    while stride < walsh_block.subgroup_order {
                                        for start in
                                            (0..walsh_block.subgroup_order).step_by(stride * 2)
                                        {
                                            for offset in 0..stride {
                                                let left_character = start + offset;
                                                let right_character = left_character + stride;
                                                let left_target = ((left_character * coset_count
                                                    + coset_index)
                                                    * component_count)
                                                    + component_index;
                                                let right_target =
                                                    ((right_character * coset_count + coset_index)
                                                        * component_count)
                                                        + component_index;
                                                let left =
                                                    contraction.group_scratch_f64[left_target];
                                                let right =
                                                    contraction.group_scratch_f64[right_target];
                                                contraction.group_scratch_f64[left_target] =
                                                    left + right;
                                                contraction.group_scratch_f64[right_target] =
                                                    left - right;
                                            }
                                        }
                                        stride *= 2;
                                    }
                                }
                            }
                        }
                        for entry in &walsh_block.entries {
                            let left_start = entry.left_group_index * component_count;
                            let right_start = entry.right_group_index * component_count;
                            let product_re = sum_real_hermitian_products(
                                &contraction.group_scratch_f64
                                    [left_start..left_start + component_count],
                                &contraction.group_scratch_f64
                                    [right_start..right_start + component_count],
                            );
                            *raw_sum += entry.symmetry_factor * entry.weight_re * product_re;
                        }
                        continue;
                    }
                    if let Some(output_indices) = repeated_block.singleton_output_indices.as_deref()
                    {
                        for (target, output_index) in
                            contraction.group_scratch_f64.iter_mut().zip(output_indices)
                        {
                            *target = amplitudes.value(row, *output_index, output_length);
                        }
                    } else {
                        for (target, group_index) in contraction
                            .group_scratch_f64
                            .iter_mut()
                            .zip(&repeated_block.component_group_indices)
                        {
                            let mut sum = c64(0.0, 0.0);
                            for output_index in &raw_sum_groups[*group_index].indices {
                                sum += amplitudes.value(row, *output_index, output_length);
                            }
                            *target = sum;
                        }
                    }

                    let component_count = repeated_block.component_count;
                    if repeated_block.all_weights_real {
                        for entry in &repeated_block.entries {
                            let left_start = entry.left_group_index * component_count;
                            let right_start = entry.right_group_index * component_count;
                            let product_re = sum_real_hermitian_products(
                                &contraction.group_scratch_f64
                                    [left_start..left_start + component_count],
                                &contraction.group_scratch_f64
                                    [right_start..right_start + component_count],
                            );
                            *raw_sum += entry.symmetry_factor * (entry.weight_re * product_re);
                        }
                    } else {
                        for entry in &repeated_block.entries {
                            let left_start = entry.left_group_index * component_count;
                            let right_start = entry.right_group_index * component_count;
                            let product = sum_hermitian_products(
                                &contraction.group_scratch_f64
                                    [left_start..left_start + component_count],
                                &contraction.group_scratch_f64
                                    [right_start..right_start + component_count],
                            );
                            *raw_sum += entry.symmetry_factor
                                * (entry.weight_re * product.re - entry.weight_im * product.im);
                        }
                    }
                }
                return Ok(());
            }
            contraction
                .group_scratch_f64
                .resize(batch_size * contraction.group_count, c64(0.0, 0.0));
            for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
                let group_row = row * contraction.group_count;
                for (group_index, group) in raw_sum_groups.iter().enumerate() {
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes.value(row, *index, output_length);
                    }
                    contraction.group_scratch_f64[group_row + group_index] = sum;
                }
                for entry in contraction.logical_entries() {
                    let left = contraction.group_scratch_f64[group_row + entry.left_group_index];
                    let right = contraction.group_scratch_f64[group_row + entry.right_group_index];
                    let product = left * right.conj();
                    *raw_sum += entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im);
                }
            }
            return Ok(());
        }
        for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
            if has_coherent_groups {
                for group in raw_sum_groups {
                    if !raw_sum_group_is_selected(group, selected_color_sector_ids) {
                        continue;
                    }
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes.value(row, *index, output_length);
                    }
                    let weight = if selected_color_sector_ids.is_none() {
                        group.all_sector_weight
                    } else {
                        group.weight
                    };
                    *raw_sum += weight * (sum.re * sum.re + sum.im * sum.im);
                }
                continue;
            }
            for index in 0..output_length {
                if !raw_sum_index_is_selected(
                    raw_sum_color_sector_ids.get(index).copied().flatten(),
                    selected_color_sector_ids,
                ) {
                    continue;
                }
                let value = amplitudes.value(row, index, output_length);
                let weight = if selected_color_sector_ids.is_none() {
                    raw_sum_all_sector_weights[index]
                } else {
                    raw_sum_weights[index]
                };
                *raw_sum += weight * (value.re * value.re + value.im * value.im);
            }
        }
        Ok(())
    }

    pub(crate) fn reduce_scratch_f64_resolved(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let amplitudes = RowMajorAmplitudeSamples::new(
            &self.output_scratch_f64,
            batch_size,
            self.output_length,
        )?;
        Self::reduce_amplitude_samples_f64_resolved(
            amplitudes,
            batch_size,
            physics,
            normalization_factor,
            selected_helicity_ids,
            selected_color_ids,
            self.output_length,
            &self.raw_sum_groups,
            self.has_coherent_groups,
            &mut self.color_contraction,
        )
    }

    /// Plane-native counterpart of the ordinary resolved reducer.
    #[allow(dead_code)]
    pub(crate) fn reduce_planes_f64_resolved(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let plane_count = usize::try_from(amplitudes.component_count()?)
            .map_err(|_| RusticolError::invalid_argument("amplitude plane count exceeds usize"))?;
        if plane_count != self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "direct amplitude plane count is {plane_count}, expected {}",
                self.output_length
            )));
        }
        Self::reduce_amplitude_samples_f64_resolved(
            amplitudes,
            amplitudes.point_count() as usize,
            physics,
            normalization_factor,
            selected_helicity_ids,
            selected_color_ids,
            self.output_length,
            &self.raw_sum_groups,
            self.has_coherent_groups,
            &mut self.color_contraction,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_amplitude_samples_f64_resolved<S: AmplitudeSamples>(
        amplitudes: S,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output_length: usize,
        raw_sum_groups: &[RawSumGroup],
        has_coherent_groups: bool,
        color_contraction: &mut Option<ColorContractionRuntime>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let helicity_count = physics.manifest.helicities.len();
        let color_count = physics.manifest.color_components.len();
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut selected_helicity_positions = vec![None; helicity_count];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            selected_helicity_positions[index] = Some(position);
        }
        let mut selected_color_positions = vec![None; color_count];
        for (position, index) in color_indices.iter().copied().enumerate() {
            selected_color_positions[index] = Some(position);
        }
        let selected_component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        let mut values = vec![0.0; batch_size * selected_component_count];

        if let Some(contraction) = color_contraction.as_mut() {
            if color_count != 1 || !physics.has_contracted_color_axis() {
                return Err(RusticolError::invalid_argument(
                    "resolved NLC/full evaluation requires one contracted color component",
                ));
            }
            if raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            let selected_contracted_color_position = selected_color_positions[0];
            let mut selected_helicity_weights_by_entry =
                Vec::with_capacity(contraction.logical_entry_count()?);
            for entry in contraction.logical_entries() {
                let left_group = &raw_sum_groups[entry.left_group_index];
                let right_group = &raw_sum_groups[entry.right_group_index];
                let left_reduction = physics
                    .numeric_reduction_by_group_id
                    .get(&left_group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            left_group.id
                        ))
                    })?;
                let right_reduction = physics
                    .numeric_reduction_by_group_id
                    .get(&right_group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            right_group.id
                        ))
                    })?;
                if left_reduction.physical_helicity_indices
                    != right_reduction.physical_helicity_indices
                {
                    return Err(RusticolError::invalid_argument(
                        "colour contraction mixed distinct physical helicities",
                    ));
                }
                let left_weights = &left_reduction.normalized_helicity_weights;
                let right_weights = &right_reduction.normalized_helicity_weights;
                if left_weights.len() != right_weights.len()
                    || left_weights.iter().zip(right_weights).any(
                        |((left_index, left_weight), (right_index, right_weight))| {
                            left_index != right_index
                                || left_weight.to_bits() != right_weight.to_bits()
                        },
                    )
                {
                    return Err(RusticolError::invalid_argument(
                        "colour contraction groups have inconsistent helicity weights",
                    ));
                }
                selected_helicity_weights_by_entry.push(
                    left_weights
                        .iter()
                        .copied()
                        .filter_map(|(helicity_index, weight)| {
                            selected_helicity_positions[helicity_index]
                                .map(|position| (position, weight))
                        })
                        .collect::<Vec<_>>(),
                );
            }
            contraction
                .group_scratch_f64
                .resize(batch_size * contraction.group_count, c64(0.0, 0.0));
            for row in 0..batch_size {
                let group_row = row * contraction.group_count;
                for (group_index, group) in raw_sum_groups.iter().enumerate() {
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes.value(row, *index, output_length);
                    }
                    contraction.group_scratch_f64[group_row + group_index] = sum;
                }
                for (entry, left_weights) in contraction
                    .logical_entries()
                    .zip(&selected_helicity_weights_by_entry)
                {
                    let left = contraction.group_scratch_f64[group_row + entry.left_group_index];
                    let right = contraction.group_scratch_f64[group_row + entry.right_group_index];
                    let product = left * right.conj();
                    let contribution = normalization_factor
                        * entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im);
                    let Some(color_position) = selected_contracted_color_position else {
                        continue;
                    };
                    for (helicity_position, weight) in left_weights {
                        values[row * selected_component_count
                            + *helicity_position * color_indices.len()
                            + color_position] += contribution * weight;
                    }
                }
            }
        } else {
            if !has_coherent_groups {
                return Err(RusticolError::invalid_argument(
                    "resolved evaluation requires coherent amplitude-group metadata",
                ));
            }
            let mut selected_member_weights_by_group = Vec::with_capacity(raw_sum_groups.len());
            for group in raw_sum_groups {
                let reduction = physics
                    .numeric_reduction_by_group_id
                    .get(&group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            group.id
                        ))
                    })?;
                selected_member_weights_by_group.push(
                    reduction
                        .normalized_member_weights
                        .iter()
                        .copied()
                        .filter_map(|(helicity_index, color_index, weight)| {
                            Some((
                                selected_helicity_positions[helicity_index]?,
                                selected_color_positions[color_index]?,
                                weight,
                            ))
                        })
                        .collect::<Vec<_>>(),
                );
            }
            for row in 0..batch_size {
                for (group, member_weights) in
                    raw_sum_groups.iter().zip(&selected_member_weights_by_group)
                {
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes.value(row, *index, output_length);
                    }
                    let contribution = normalization_factor
                        * group.all_sector_weight
                        * (sum.re * sum.re + sum.im * sum.im);
                    for (helicity_position, color_position, weight) in member_weights {
                        values[row * selected_component_count
                            + *helicity_position * color_indices.len()
                            + *color_position] += contribution * weight;
                    }
                }
            }
        }
        Ok(ResolvedValues {
            values,
            point_count: batch_size,
            helicity_indices,
            color_indices,
        })
    }

    /// Reduce one topology-replay source mapping directly into public totals.
    ///
    /// The source and target component buffers are row-sized persistent
    /// scratch.  Replaying the routes before summing the target row preserves
    /// the resolved contract's component accumulation and H-major/C-minor
    /// summation order without materializing either batch-wide tensor.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_scratch_f64_routed_totals_into(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        replay_entry: &LcResolvedReplayEntry,
        source_component_count: usize,
        target_component_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let amplitudes = RowMajorAmplitudeSamples::new(
            &self.output_scratch_f64,
            batch_size,
            self.output_length,
        )?;
        Self::reduce_amplitude_samples_f64_routed_totals_into(
            amplitudes,
            batch_size,
            physics,
            normalization_factor,
            selected_helicity_ids,
            selected_color_ids,
            replay_entry,
            source_component_count,
            target_component_count,
            output,
            self.output_length,
            &self.raw_sum_groups,
            self.has_coherent_groups,
            self.color_contraction.is_some(),
            &mut self.resolved_source_row_scratch_f64,
            &mut self.resolved_target_row_scratch_f64,
            &mut self.routed_reduction_scratch,
        )
    }

    /// Plane-native counterpart of the LC topology-replay totals reducer.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    pub(crate) fn reduce_planes_f64_routed_totals_into(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        replay_entry: &LcResolvedReplayEntry,
        source_component_count: usize,
        target_component_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let plane_count = usize::try_from(amplitudes.component_count()?)
            .map_err(|_| RusticolError::invalid_argument("amplitude plane count exceeds usize"))?;
        if plane_count != self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "direct amplitude plane count is {plane_count}, expected {}",
                self.output_length
            )));
        }
        let batch_size = amplitudes.point_count() as usize;
        if output.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "routed reduction output has length {}, expected {batch_size}",
                output.len()
            )));
        }
        if self.color_contraction.is_some() {
            return Err(RusticolError::invalid_argument(
                "LC topology replay does not support contracted color reduction",
            ));
        }
        if !self.has_coherent_groups {
            return Err(RusticolError::invalid_argument(
                "resolved evaluation requires coherent amplitude-group metadata",
            ));
        }

        let scratch = &mut self.routed_reduction_scratch;
        let helicity_count = physics.manifest.helicities.len();
        let color_count = physics.manifest.color_components.len();
        physics
            .selected_helicity_indices_into(selected_helicity_ids, &mut scratch.helicity_indices)?;
        physics.selected_color_indices_into(selected_color_ids, &mut scratch.color_indices)?;
        let selected_color_count = scratch.color_indices.len();
        let expected_source_component_count = scratch
            .helicity_indices
            .len()
            .checked_mul(selected_color_count)
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        if source_component_count != expected_source_component_count {
            return Err(RusticolError::integrity(format!(
                "topology replay source has {source_component_count} components, expected {expected_source_component_count}"
            )));
        }
        for route in &replay_entry.routes {
            if route.source_index >= source_component_count
                || route.target_index >= target_component_count
            {
                return Err(RusticolError::integrity(
                    "LC topology replay selected route is out of bounds",
                ));
            }
        }

        let helicity_positions = SelectedAxisPositions::new(
            &scratch.helicity_indices,
            helicity_count,
            &mut scratch.helicity_positions,
        );
        let color_positions = SelectedAxisPositions::new(
            &scratch.color_indices,
            color_count,
            &mut scratch.color_positions,
        );
        scratch.selected_member_weights.clear();
        scratch.selected_member_weight_ranges.clear();
        for group in &self.raw_sum_groups {
            let reduction = physics
                .numeric_reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "resolved metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            let start = scratch.selected_member_weights.len();
            for (helicity_index, color_index, weight) in
                reduction.normalized_member_weights.iter().copied()
            {
                let (Some(helicity_position), Some(color_position)) = (
                    helicity_positions.position(helicity_index),
                    color_positions.position(color_index),
                ) else {
                    continue;
                };
                scratch
                    .selected_member_weights
                    .push((helicity_position, color_position, weight));
            }
            scratch
                .selected_member_weight_ranges
                .push(start..scratch.selected_member_weights.len());
        }

        let source_scalar_count =
            source_component_count
                .checked_mul(batch_size)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("routed source scratch shape overflows usize")
                })?;
        let target_scalar_count =
            target_component_count
                .checked_mul(batch_size)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("routed target scratch shape overflows usize")
                })?;
        scratch.direct_group_re.resize(batch_size, 0.0);
        scratch.direct_group_im.resize(batch_size, 0.0);
        scratch.direct_totals.resize(batch_size, 0.0);
        scratch
            .direct_source_components
            .resize(source_scalar_count, 0.0);
        scratch.direct_source_components.fill(0.0);
        scratch
            .direct_target_components
            .resize(target_scalar_count, 0.0);
        scratch.direct_target_components.fill(0.0);

        for (group, member_weight_range) in self
            .raw_sum_groups
            .iter()
            .zip(&scratch.selected_member_weight_ranges)
        {
            if member_weight_range.is_empty() {
                continue;
            }
            scratch.direct_group_re.fill(0.0);
            scratch.direct_group_im.fill(0.0);
            for index in &group.indices {
                let (plane_re, plane_im) = amplitudes.plane_unchecked(*index);
                for point in 0..batch_size {
                    scratch.direct_group_re[point] += plane_re[point];
                    scratch.direct_group_im[point] += plane_im[point];
                }
            }
            for point in 0..batch_size {
                let re = scratch.direct_group_re[point];
                let im = scratch.direct_group_im[point];
                scratch.direct_totals[point] =
                    normalization_factor * group.all_sector_weight * (re * re + im * im);
            }
            for (helicity_position, color_position, weight) in
                &scratch.selected_member_weights[member_weight_range.clone()]
            {
                let component = *helicity_position * selected_color_count + *color_position;
                let source_start = component * batch_size;
                for point in 0..batch_size {
                    scratch.direct_source_components[source_start + point] +=
                        scratch.direct_totals[point] * *weight;
                }
            }
        }

        for route in &replay_entry.routes {
            let source_start = route.source_index * batch_size;
            let target_start = route.target_index * batch_size;
            for point in 0..batch_size {
                scratch.direct_target_components[target_start + point] +=
                    route.weight * scratch.direct_source_components[source_start + point];
            }
        }
        output.fill(0.0);
        for target_index in 0..target_component_count {
            let target_start = target_index * batch_size;
            for (point, target) in output.iter_mut().enumerate() {
                *target += scratch.direct_target_components[target_start + point];
            }
        }
        Ok(())
    }

    /// Add one plane-native topology-replay mapping to persistent point-major
    /// target components.
    ///
    /// Mappings are accumulated in caller order, while the final public fold
    /// remains H-major/C-minor. This is the allocation-free multi-mapping
    /// counterpart of `reduce_planes_f64_routed_totals_into`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_planes_f64_routed_components_add_into(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        replay_entry: &LcResolvedReplayEntry,
        source_component_count: usize,
        target_component_count: usize,
        target_point_count: usize,
        target_point_start: usize,
        target_components: &mut [f64],
    ) -> RusticolResult<()> {
        let plane_count = usize::try_from(amplitudes.component_count()?)
            .map_err(|_| RusticolError::invalid_argument("amplitude plane count exceeds usize"))?;
        if plane_count != self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "direct amplitude plane count is {plane_count}, expected {}",
                self.output_length
            )));
        }
        let batch_size = amplitudes.point_count() as usize;
        let expected_target_len = target_point_count
            .checked_mul(target_component_count)
            .ok_or_else(|| RusticolError::invalid_argument("routed target shape overflows"))?;
        if target_components.len() != expected_target_len
            || target_point_start
                .checked_add(batch_size)
                .is_none_or(|stop| stop > target_point_count)
        {
            return Err(RusticolError::invalid_argument(
                "routed target components have an inconsistent shape",
            ));
        }
        if self.color_contraction.is_some() {
            return Err(RusticolError::invalid_argument(
                "LC topology replay does not support contracted color reduction",
            ));
        }
        if !self.has_coherent_groups {
            return Err(RusticolError::invalid_argument(
                "resolved evaluation requires coherent amplitude-group metadata",
            ));
        }

        let scratch = &mut self.routed_reduction_scratch;
        let helicity_count = physics.manifest.helicities.len();
        let color_count = physics.manifest.color_components.len();
        physics
            .selected_helicity_indices_into(selected_helicity_ids, &mut scratch.helicity_indices)?;
        physics.selected_color_indices_into(selected_color_ids, &mut scratch.color_indices)?;
        let selected_color_count = scratch.color_indices.len();
        let expected_source_component_count = scratch
            .helicity_indices
            .len()
            .checked_mul(selected_color_count)
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        if source_component_count != expected_source_component_count {
            return Err(RusticolError::integrity(format!(
                "topology replay source has {source_component_count} components, expected {expected_source_component_count}"
            )));
        }
        for route in &replay_entry.routes {
            if route.source_index >= source_component_count
                || route.target_index >= target_component_count
            {
                return Err(RusticolError::integrity(
                    "LC topology replay selected route is out of bounds",
                ));
            }
        }

        let helicity_positions = SelectedAxisPositions::new(
            &scratch.helicity_indices,
            helicity_count,
            &mut scratch.helicity_positions,
        );
        let color_positions = SelectedAxisPositions::new(
            &scratch.color_indices,
            color_count,
            &mut scratch.color_positions,
        );
        scratch.selected_member_weights.clear();
        scratch.selected_member_weight_ranges.clear();
        for group in &self.raw_sum_groups {
            let reduction = physics
                .numeric_reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "resolved metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            let start = scratch.selected_member_weights.len();
            for (helicity_index, color_index, weight) in
                reduction.normalized_member_weights.iter().copied()
            {
                let (Some(helicity_position), Some(color_position)) = (
                    helicity_positions.position(helicity_index),
                    color_positions.position(color_index),
                ) else {
                    continue;
                };
                scratch
                    .selected_member_weights
                    .push((helicity_position, color_position, weight));
            }
            scratch
                .selected_member_weight_ranges
                .push(start..scratch.selected_member_weights.len());
        }

        let source_scalar_count =
            source_component_count
                .checked_mul(batch_size)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("routed source scratch shape overflows usize")
                })?;
        scratch.direct_group_re.resize(batch_size, 0.0);
        scratch.direct_group_im.resize(batch_size, 0.0);
        scratch.direct_totals.resize(batch_size, 0.0);
        scratch
            .direct_source_components
            .resize(source_scalar_count, 0.0);
        scratch.direct_source_components.fill(0.0);

        for (group, member_weight_range) in self
            .raw_sum_groups
            .iter()
            .zip(&scratch.selected_member_weight_ranges)
        {
            if member_weight_range.is_empty() {
                continue;
            }
            scratch.direct_group_re.fill(0.0);
            scratch.direct_group_im.fill(0.0);
            for index in &group.indices {
                let (plane_re, plane_im) = amplitudes.plane_unchecked(*index);
                for point in 0..batch_size {
                    scratch.direct_group_re[point] += plane_re[point];
                    scratch.direct_group_im[point] += plane_im[point];
                }
            }
            for point in 0..batch_size {
                let re = scratch.direct_group_re[point];
                let im = scratch.direct_group_im[point];
                scratch.direct_totals[point] =
                    normalization_factor * group.all_sector_weight * (re * re + im * im);
            }
            for (helicity_position, color_position, weight) in
                &scratch.selected_member_weights[member_weight_range.clone()]
            {
                let component = *helicity_position * selected_color_count + *color_position;
                let source_start = component * batch_size;
                for point in 0..batch_size {
                    scratch.direct_source_components[source_start + point] +=
                        scratch.direct_totals[point] * *weight;
                }
            }
        }

        for route in &replay_entry.routes {
            let source_start = route.source_index * batch_size;
            for point in 0..batch_size {
                let target =
                    (target_point_start + point) * target_component_count + route.target_index;
                target_components[target] +=
                    route.weight * scratch.direct_source_components[source_start + point];
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_amplitude_samples_f64_routed_totals_into<S: AmplitudeSamples>(
        amplitudes: S,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        replay_entry: &LcResolvedReplayEntry,
        source_component_count: usize,
        target_component_count: usize,
        output: &mut [f64],
        output_length: usize,
        raw_sum_groups: &[RawSumGroup],
        has_coherent_groups: bool,
        has_color_contraction: bool,
        resolved_source_row_scratch_f64: &mut Vec<f64>,
        resolved_target_row_scratch_f64: &mut Vec<f64>,
        scratch: &mut RoutedReductionScratch,
    ) -> RusticolResult<()> {
        if output.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "routed reduction output has length {}, expected {batch_size}",
                output.len()
            )));
        }
        if has_color_contraction {
            return Err(RusticolError::invalid_argument(
                "LC topology replay does not support contracted color reduction",
            ));
        }
        if !has_coherent_groups {
            return Err(RusticolError::invalid_argument(
                "resolved evaluation requires coherent amplitude-group metadata",
            ));
        }

        let helicity_count = physics.manifest.helicities.len();
        let color_count = physics.manifest.color_components.len();
        physics
            .selected_helicity_indices_into(selected_helicity_ids, &mut scratch.helicity_indices)?;
        physics.selected_color_indices_into(selected_color_ids, &mut scratch.color_indices)?;
        let expected_source_component_count = scratch
            .helicity_indices
            .len()
            .checked_mul(scratch.color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        if source_component_count != expected_source_component_count {
            return Err(RusticolError::integrity(format!(
                "topology replay source has {source_component_count} components, expected {expected_source_component_count}"
            )));
        }

        let helicity_positions = SelectedAxisPositions::new(
            &scratch.helicity_indices,
            helicity_count,
            &mut scratch.helicity_positions,
        );
        let color_positions = SelectedAxisPositions::new(
            &scratch.color_indices,
            color_count,
            &mut scratch.color_positions,
        );
        scratch.selected_member_weights.clear();
        scratch.selected_member_weight_ranges.clear();
        for group in raw_sum_groups {
            let reduction = physics
                .numeric_reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "resolved metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            let start = scratch.selected_member_weights.len();
            for (helicity_index, color_index, weight) in
                reduction.normalized_member_weights.iter().copied()
            {
                let (Some(helicity_position), Some(color_position)) = (
                    helicity_positions.position(helicity_index),
                    color_positions.position(color_index),
                ) else {
                    continue;
                };
                scratch
                    .selected_member_weights
                    .push((helicity_position, color_position, weight));
            }
            scratch
                .selected_member_weight_ranges
                .push(start..scratch.selected_member_weights.len());
        }

        resolved_source_row_scratch_f64.resize(source_component_count, 0.0);
        resolved_target_row_scratch_f64.resize(target_component_count, 0.0);
        for (row, target_total) in output.iter_mut().enumerate() {
            let source_row = &mut *resolved_source_row_scratch_f64;
            source_row.fill(0.0);
            for (group, member_weight_range) in raw_sum_groups
                .iter()
                .zip(&scratch.selected_member_weight_ranges)
            {
                let mut sum = c64(0.0, 0.0);
                for index in &group.indices {
                    sum += amplitudes.value(row, *index, output_length);
                }
                let contribution = normalization_factor
                    * group.all_sector_weight
                    * (sum.re * sum.re + sum.im * sum.im);
                for (helicity_position, color_position, weight) in
                    &scratch.selected_member_weights[member_weight_range.clone()]
                {
                    source_row
                        [*helicity_position * scratch.color_indices.len() + *color_position] +=
                        contribution * weight;
                }
            }

            let target_row = &mut *resolved_target_row_scratch_f64;
            target_row.fill(0.0);
            for route in &replay_entry.routes {
                if route.source_index >= source_component_count
                    || route.target_index >= target_component_count
                {
                    return Err(RusticolError::integrity(
                        "LC topology replay selected route is out of bounds",
                    ));
                }
                target_row[route.target_index] += route.weight * source_row[route.source_index];
            }
            *target_total = target_row.iter().sum();
        }
        Ok(())
    }

    /// Reduce amplitudes evaluated for one physical runtime-helicity domain.
    ///
    /// The ordinary resolved reducer distributes one representative amplitude
    /// over every proof-equivalent physical helicity.  A materialized
    /// recurrence artifact instead reevaluates that representative with the
    /// requested source states, so the selected helicity receives the full
    /// contribution.  Colour symmetry weights remain distributed over their
    /// physical components exactly as in the ordinary resolved contract.
    pub(crate) fn reduce_scratch_f64_for_materialized_helicity(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let amplitudes = RowMajorAmplitudeSamples::new(
            &self.output_scratch_f64,
            batch_size,
            self.output_length,
        )?;
        Self::reduce_amplitude_samples_f64_for_materialized_helicity(
            amplitudes,
            batch_size,
            physics,
            normalization_factor,
            helicity_index,
            root_factors,
            selected_color_ids,
            self.output_length,
            &self.raw_sum_groups,
            self.has_coherent_groups,
            &mut self.color_contraction,
        )
    }

    /// Plane-native counterpart of the materialized physical-helicity reducer.
    #[allow(dead_code)]
    pub(crate) fn reduce_planes_f64_for_materialized_helicity(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let plane_count = usize::try_from(amplitudes.component_count()?)
            .map_err(|_| RusticolError::invalid_argument("amplitude plane count exceeds usize"))?;
        if plane_count != self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "direct amplitude plane count is {plane_count}, expected {}",
                self.output_length
            )));
        }
        Self::reduce_amplitude_samples_f64_for_materialized_helicity(
            amplitudes,
            amplitudes.point_count() as usize,
            physics,
            normalization_factor,
            helicity_index,
            root_factors,
            selected_color_ids,
            self.output_length,
            &self.raw_sum_groups,
            self.has_coherent_groups,
            &mut self.color_contraction,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_amplitude_samples_f64_for_materialized_helicity<S: AmplitudeSamples>(
        amplitudes: S,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
        output_length: usize,
        raw_sum_groups: &[RawSumGroup],
        has_coherent_groups: bool,
        color_contraction: &mut Option<ColorContractionRuntime>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        if root_factors.len() != output_length {
            return Err(RusticolError::integrity(format!(
                "helicity recurrence route table has {} roots, expected {}",
                root_factors.len(),
                output_length
            )));
        }
        if helicity_index >= physics.manifest.helicities.len() {
            return Err(RusticolError::selector(format!(
                "runtime helicity index {helicity_index} is out of range"
            )));
        }
        let color_count = physics.manifest.color_components.len();
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut full_values = vec![0.0; batch_size * color_count];

        if let Some(contraction) = color_contraction.as_mut() {
            if color_count != 1 || !physics.has_contracted_color_axis() {
                return Err(RusticolError::invalid_argument(
                    "resolved NLC/full evaluation requires one contracted color component",
                ));
            }
            if raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            contraction
                .group_scratch_f64
                .resize(batch_size * contraction.group_count, c64(0.0, 0.0));
            let mut group_active = vec![false; contraction.group_count];
            for (group_index, group) in raw_sum_groups.iter().enumerate() {
                let reduction = physics
                    .numeric_reduction_by_group_id
                    .get(&group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            group.id
                        ))
                    })?;
                group_active[group_index] = reduction.contains_helicity(helicity_index)
                    && group
                        .indices
                        .iter()
                        .any(|index| root_factors[*index].is_some());
            }
            for (row, full_value) in full_values.iter_mut().enumerate().take(batch_size) {
                let group_row = row * contraction.group_count;
                for (group_index, group) in raw_sum_groups.iter().enumerate() {
                    let mut sum = c64(0.0, 0.0);
                    if group_active[group_index] {
                        for index in &group.indices {
                            if let Some(factor) = root_factors[*index] {
                                sum += amplitudes.value(row, *index, output_length) * factor;
                            }
                        }
                    }
                    contraction.group_scratch_f64[group_row + group_index] = sum;
                }
                for entry in contraction.logical_entries() {
                    if !group_active[entry.left_group_index]
                        || !group_active[entry.right_group_index]
                    {
                        continue;
                    }
                    let left = contraction.group_scratch_f64[group_row + entry.left_group_index];
                    let right = contraction.group_scratch_f64[group_row + entry.right_group_index];
                    let product = left * right.conj();
                    *full_value += normalization_factor
                        * entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im);
                }
            }
        } else {
            if !has_coherent_groups {
                return Err(RusticolError::invalid_argument(
                    "materialized helicity reduction requires coherent amplitude-group metadata",
                ));
            }
            let mut active_groups = Vec::new();
            for (group_index, group) in raw_sum_groups.iter().enumerate() {
                let reduction = physics
                    .numeric_reduction_by_group_id
                    .get(&group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            group.id
                        ))
                    })?;
                if !reduction.contains_helicity(helicity_index)
                    || !group
                        .indices
                        .iter()
                        .any(|index| root_factors[*index].is_some())
                {
                    continue;
                }
                active_groups.push((group_index, reduction.normalized_color_weights.clone()));
            }
            for row in 0..batch_size {
                let color_row = row * color_count;
                for (group_index, color_weights) in &active_groups {
                    let group = &raw_sum_groups[*group_index];
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        if let Some(factor) = root_factors[*index] {
                            sum += amplitudes.value(row, *index, output_length) * factor;
                        }
                    }
                    let contribution = normalization_factor
                        * group.all_sector_weight
                        * (sum.re * sum.re + sum.im * sum.im);
                    for (color_index, weight) in color_weights {
                        full_values[color_row + *color_index] += contribution * *weight;
                    }
                }
            }
        }

        let mut values = Vec::with_capacity(batch_size * color_indices.len());
        for row in 0..batch_size {
            for color_index in &color_indices {
                values.push(full_values[row * color_count + *color_index]);
            }
        }
        Ok(ResolvedValues {
            values,
            point_count: batch_size,
            helicity_indices: vec![helicity_index],
            color_indices,
        })
    }

    /// Add one materialized physical-helicity row to caller-owned totals.
    ///
    /// Colour cells are accumulated in the same group order as the resolved
    /// reducer, then folded into each point total in manifest colour order.
    /// Keeping only one colour row preserves exact summation semantics while
    /// avoiding a batch-by-colour resolved allocation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_scratch_f64_for_materialized_helicity_add_into(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let amplitudes = RowMajorAmplitudeSamples::new(
            &self.output_scratch_f64,
            batch_size,
            self.output_length,
        )?;
        Self::reduce_amplitude_samples_f64_for_materialized_helicity_add_into(
            amplitudes,
            batch_size,
            physics,
            normalization_factor,
            helicity_index,
            root_factors,
            selected_color_ids,
            output,
            self.output_length,
            &self.raw_sum_groups,
            self.has_coherent_groups,
            &mut self.color_contraction,
            &mut self.resolved_source_row_scratch_f64,
            &mut self.routed_reduction_scratch,
        )
    }

    /// Cold-bind the numeric materialized-helicity totals reducer for one
    /// concrete physics/amplitude recipient.
    ///
    /// Parent-closure lanes deliberately pass the parent's physics metadata to
    /// a child amplitude runtime.  The binding key therefore includes the
    /// actual `PhysicsRuntime`, rather than assuming the amplitude's owning
    /// execution runtime supplies the reduction metadata.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_materialized_helicity_direct_total_plan(
        &mut self,
        physics: &PhysicsRuntime,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<usize> {
        if root_factors.len() != self.output_length {
            return Err(RusticolError::integrity(format!(
                "helicity recurrence route table has {} roots, expected {}",
                root_factors.len(),
                self.output_length
            )));
        }
        if helicity_index >= physics.manifest.helicities.len() {
            return Err(RusticolError::selector(format!(
                "runtime helicity index {helicity_index} is out of range"
            )));
        }
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let root_factor_bits = root_factors
            .iter()
            .enumerate()
            .filter_map(|(index, factor)| {
                factor.map(|factor| (index, [factor.re.to_bits(), factor.im.to_bits()]))
            })
            .collect::<Vec<_>>();
        let key = MaterializedHelicityDirectTotalPlanKey {
            physics_binding_id: physics.binding_id,
            helicity_index,
            root_factor_bits,
            color_indices,
        };
        // Common 384-helicity artifacts fit every schedule; 768-helicity
        // artifacts have at most 384 non-structural/global-flip primaries and
        // therefore fit as well. Parent-closure lanes use 192 schedules for the
        // qq_Z6g artifact. Arbitrary dynamic selector/root-factor combinations
        // remain capped instead of growing the runtime cache indefinitely.
        let artifact_plan_capacity = physics.manifest.helicities.len().clamp(16, 512);
        self.materialized_helicity_direct_total_plan_capacity = self
            .materialized_helicity_direct_total_plan_capacity
            .max(artifact_plan_capacity);
        if let Some(index) = self
            .materialized_helicity_direct_total_plans
            .iter()
            .position(|plan| plan.key == key)
        {
            return Ok(index);
        }

        let mut roots = Vec::new();
        let mut groups = Vec::new();
        let mut bound_group_by_raw_group = vec![None; self.raw_sum_groups.len()];
        let mut color_weights_by_group = Vec::new();
        for (raw_group_index, group) in self.raw_sum_groups.iter().enumerate() {
            let numeric = physics
                .numeric_reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "resolved metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            if !numeric.contains_helicity(helicity_index) {
                continue;
            }
            let root_start = roots.len();
            for output_index in &group.indices {
                let factor = root_factors.get(*output_index).ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "coherent group {} references missing amplitude root {output_index}",
                        group.id
                    ))
                })?;
                if let Some(factor) = factor {
                    roots.push(MaterializedHelicityDirectTotalRoot {
                        output_index: *output_index,
                        factor: *factor,
                    });
                }
            }
            if roots.len() == root_start {
                continue;
            }
            let bound_group_index = groups.len();
            bound_group_by_raw_group[raw_group_index] = Some(bound_group_index);
            let root_range = root_start..roots.len();
            let identity_output_index = match &roots[root_range.clone()] {
                [root]
                    if root.factor.re.to_bits() == 1.0f64.to_bits()
                        && root.factor.im.to_bits() == 0.0f64.to_bits() =>
                {
                    Some(root.output_index)
                }
                _ => None,
            };
            groups.push(MaterializedHelicityDirectTotalGroup {
                root_range,
                all_sector_weight: group.all_sector_weight,
                identity_output_index,
            });
            color_weights_by_group.push(numeric.normalized_color_weights.clone());
        }

        let reduction = if let Some(contraction) = self.color_contraction.as_ref() {
            if !physics.has_contracted_color_axis() || physics.manifest.color_components.len() != 1
            {
                return Err(RusticolError::invalid_argument(
                    "materialized-helicity color contraction requires one contracted color component",
                ));
            }
            if self.raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            if selected_color_ids.is_some() && key.color_indices.as_slice() != [0] {
                return Err(RusticolError::selector(
                    "materialized-helicity color selection does not match the contracted axis",
                ));
            }
            let entries = contraction
                .logical_entries()
                .filter_map(|entry| {
                    Some(MaterializedHelicityDirectTotalContractionEntry {
                        left_group_index: bound_group_by_raw_group[entry.left_group_index]?,
                        right_group_index: bound_group_by_raw_group[entry.right_group_index]?,
                        weight_re: entry.weight_re,
                        weight_im: entry.weight_im,
                        symmetry_factor: entry.symmetry_factor,
                    })
                })
                .collect();
            MaterializedHelicityDirectTotalReduction::Contracted { entries }
        } else {
            if !self.has_coherent_groups {
                return Err(RusticolError::invalid_argument(
                    "materialized helicity reduction requires coherent amplitude-group metadata",
                ));
            }
            let mut color_group_ranges = Vec::with_capacity(key.color_indices.len());
            let mut color_groups = Vec::new();
            for color_index in &key.color_indices {
                let start = color_groups.len();
                for (group_index, color_weights) in color_weights_by_group.iter().enumerate() {
                    for (_, weight) in color_weights
                        .iter()
                        .filter(|(candidate, _)| candidate == color_index)
                    {
                        color_groups.push(MaterializedHelicityDirectTotalColorGroup {
                            group_index,
                            weight: *weight,
                        });
                    }
                }
                color_group_ranges.push(start..color_groups.len());
            }
            MaterializedHelicityDirectTotalReduction::Lc {
                color_group_ranges,
                color_groups,
            }
        };
        let plan = MaterializedHelicityDirectTotalPlan {
            key,
            roots,
            groups,
            reduction,
        };
        let index = if self.materialized_helicity_direct_total_plans.len()
            < self.materialized_helicity_direct_total_plan_capacity
        {
            let index = self.materialized_helicity_direct_total_plans.len();
            self.materialized_helicity_direct_total_plans.push(plan);
            index
        } else {
            let index = self.materialized_helicity_direct_total_next_replacement
                % self.materialized_helicity_direct_total_plans.len();
            self.materialized_helicity_direct_total_plans[index] = plan;
            for cached in &mut self.materialized_helicity_direct_default_plan_by_selector_domain {
                if cached.is_some_and(|cached| cached.plan_index == index) {
                    *cached = None;
                }
            }
            self.materialized_helicity_direct_total_next_replacement =
                (index + 1) % self.materialized_helicity_direct_total_plans.len();
            index
        };
        Ok(index)
    }

    /// Bind the immutable default contracted-colour plan for one authenticated
    /// selector domain, reusing its plan index on subsequent warm calls.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind_default_materialized_helicity_direct_total_plan(
        &mut self,
        selector_domain_id: usize,
        physics: &PhysicsRuntime,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<usize> {
        if selected_color_ids.is_some() || !physics.has_contracted_color_axis() {
            return self.bind_materialized_helicity_direct_total_plan(
                physics,
                helicity_index,
                root_factors,
                selected_color_ids,
            );
        }
        let cached = self
            .materialized_helicity_direct_default_plan_by_selector_domain
            .get(selector_domain_id)
            .copied()
            .flatten();
        if let Some(cached) = cached
            && cached.physics_binding_id == physics.binding_id
            && cached.helicity_index == helicity_index
            && self
                .materialized_helicity_direct_total_plans
                .get(cached.plan_index)
                .is_some_and(|plan| {
                    plan.key.physics_binding_id == physics.binding_id
                        && plan.key.helicity_index == helicity_index
                        && plan.key.color_indices.as_slice() == [0]
                })
        {
            return Ok(cached.plan_index);
        }

        let plan_index = self.bind_materialized_helicity_direct_total_plan(
            physics,
            helicity_index,
            root_factors,
            selected_color_ids,
        )?;
        if self
            .materialized_helicity_direct_default_plan_by_selector_domain
            .len()
            <= selector_domain_id
        {
            self.materialized_helicity_direct_default_plan_by_selector_domain
                .resize(selector_domain_id + 1, None);
        }
        self.materialized_helicity_direct_default_plan_by_selector_domain[selector_domain_id] =
            Some(MaterializedHelicityDirectDefaultPlanCacheEntry {
                physics_binding_id: physics.binding_id,
                helicity_index,
                plan_index,
            });
        Ok(plan_index)
    }

    /// Plane-native materialized-helicity totals using a cold-bound numeric
    /// reduction plan.
    pub(crate) fn reduce_planes_f64_for_materialized_helicity_plan_add_into(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        normalization_factor: f64,
        plan_index: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let plane_count = usize::try_from(amplitudes.component_count()?)
            .map_err(|_| RusticolError::invalid_argument("amplitude plane count exceeds usize"))?;
        if plane_count != self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "direct amplitude plane count is {plane_count}, expected {}",
                self.output_length
            )));
        }
        let batch_size = amplitudes.point_count() as usize;
        if output.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "materialized helicity output has length {}, expected {batch_size}",
                output.len()
            )));
        }
        let plan = self
            .materialized_helicity_direct_total_plans
            .get(plan_index)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "materialized-helicity direct-total plan {plan_index} is not bound"
                ))
            })?;
        let scratch = &mut self.routed_reduction_scratch;
        scratch.direct_group_re.resize(batch_size, 0.0);
        scratch.direct_group_im.resize(batch_size, 0.0);
        scratch.direct_totals.resize(batch_size, 0.0);

        match &plan.reduction {
            MaterializedHelicityDirectTotalReduction::Lc {
                color_group_ranges,
                color_groups,
            } => {
                for color_group_range in color_group_ranges {
                    scratch.direct_totals.fill(0.0);
                    for color_group in &color_groups[color_group_range.clone()] {
                        let group = &plan.groups[color_group.group_index];
                        if let Some(output_index) = group.identity_output_index {
                            let (plane_re, plane_im) = amplitudes.plane_unchecked(output_index);
                            for point in 0..batch_size {
                                let re = plane_re[point];
                                let im = plane_im[point];
                                let norm = if re.is_finite() && im.is_finite() {
                                    re * re + im * im
                                } else {
                                    let value = c64(re, im) * c64(1.0, 0.0);
                                    value.re * value.re + value.im * value.im
                                };
                                let contribution =
                                    normalization_factor * group.all_sector_weight * norm;
                                scratch.direct_totals[point] += contribution * color_group.weight;
                            }
                            continue;
                        }
                        scratch.direct_group_re.fill(0.0);
                        scratch.direct_group_im.fill(0.0);
                        for root in &plan.roots[group.root_range.clone()] {
                            let (plane_re, plane_im) =
                                amplitudes.plane_unchecked(root.output_index);
                            for point in 0..batch_size {
                                let value = c64(plane_re[point], plane_im[point]) * root.factor;
                                scratch.direct_group_re[point] += value.re;
                                scratch.direct_group_im[point] += value.im;
                            }
                        }
                        for point in 0..batch_size {
                            let re = scratch.direct_group_re[point];
                            let im = scratch.direct_group_im[point];
                            let contribution = normalization_factor
                                * group.all_sector_weight
                                * (re * re + im * im);
                            scratch.direct_totals[point] += contribution * color_group.weight;
                        }
                    }
                    for (target, total) in output.iter_mut().zip(&scratch.direct_totals) {
                        *target += *total;
                    }
                }
            }
            MaterializedHelicityDirectTotalReduction::Contracted { entries } => {
                let group_scalar_count =
                    plan.groups.len().checked_mul(batch_size).ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "materialized-helicity group scratch shape overflows usize",
                        )
                    })?;
                scratch.direct_group_re.resize(group_scalar_count, 0.0);
                scratch.direct_group_re.fill(0.0);
                scratch.direct_group_im.resize(group_scalar_count, 0.0);
                scratch.direct_group_im.fill(0.0);
                for (group_index, group) in plan.groups.iter().enumerate() {
                    let group_start = group_index * batch_size;
                    if let Some(output_index) = group.identity_output_index {
                        let (plane_re, plane_im) = amplitudes.plane_unchecked(output_index);
                        for point in 0..batch_size {
                            let re = plane_re[point];
                            let im = plane_im[point];
                            if re.is_finite() && im.is_finite() {
                                scratch.direct_group_re[group_start + point] = re;
                                scratch.direct_group_im[group_start + point] = im;
                            } else {
                                let value = c64(re, im) * c64(1.0, 0.0);
                                scratch.direct_group_re[group_start + point] = value.re;
                                scratch.direct_group_im[group_start + point] = value.im;
                            }
                        }
                        continue;
                    }
                    for root in &plan.roots[group.root_range.clone()] {
                        let (plane_re, plane_im) = amplitudes.plane_unchecked(root.output_index);
                        for point in 0..batch_size {
                            let value = c64(plane_re[point], plane_im[point]) * root.factor;
                            scratch.direct_group_re[group_start + point] += value.re;
                            scratch.direct_group_im[group_start + point] += value.im;
                        }
                    }
                }
                scratch.direct_totals.fill(0.0);
                for entry in entries {
                    for point in 0..batch_size {
                        let left_start = entry.left_group_index * batch_size;
                        let left = c64(
                            scratch.direct_group_re[left_start + point],
                            scratch.direct_group_im[left_start + point],
                        );
                        let right_start = entry.right_group_index * batch_size;
                        let right = c64(
                            scratch.direct_group_re[right_start + point],
                            scratch.direct_group_im[right_start + point],
                        );
                        let product = left * right.conj();
                        scratch.direct_totals[point] += normalization_factor
                            * entry.symmetry_factor
                            * (entry.weight_re * product.re - entry.weight_im * product.im);
                    }
                }
                for (target, total) in output.iter_mut().zip(&scratch.direct_totals) {
                    *target += *total;
                }
            }
        }
        Ok(())
    }

    /// Plane-native counterpart of the materialized-helicity totals accumulator.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_planes_f64_for_materialized_helicity_add_into(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let plan_index = self.bind_materialized_helicity_direct_total_plan(
            physics,
            helicity_index,
            root_factors,
            selected_color_ids,
        )?;
        self.reduce_planes_f64_for_materialized_helicity_plan_add_into(
            amplitudes,
            normalization_factor,
            plan_index,
            output,
        )
    }

    /// Route one materialized-helicity tile into persistent public replay
    /// components without constructing a resolved colour tensor.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_planes_f64_for_materialized_helicity_routed_components_add_into(
        &mut self,
        amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
        replay_entry: &LcResolvedReplayEntry,
        source_component_count: usize,
        target_component_count: usize,
        target_point_count: usize,
        target_point_start: usize,
        target_components: &mut [f64],
    ) -> RusticolResult<()> {
        let plane_count = usize::try_from(amplitudes.component_count()?)
            .map_err(|_| RusticolError::invalid_argument("amplitude plane count exceeds usize"))?;
        if plane_count != self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "direct amplitude plane count is {plane_count}, expected {}",
                self.output_length
            )));
        }
        if root_factors.len() != self.output_length {
            return Err(RusticolError::integrity(format!(
                "helicity recurrence route table has {} roots, expected {}",
                root_factors.len(),
                self.output_length
            )));
        }
        if self.color_contraction.is_some() {
            return Err(RusticolError::invalid_argument(
                "LC topology replay does not support contracted color reduction",
            ));
        }
        if !self.has_coherent_groups {
            return Err(RusticolError::invalid_argument(
                "materialized helicity reduction requires coherent amplitude-group metadata",
            ));
        }
        if helicity_index >= physics.manifest.helicities.len() {
            return Err(RusticolError::selector(format!(
                "runtime helicity index {helicity_index} is out of range"
            )));
        }
        let batch_size = amplitudes.point_count() as usize;
        let expected_target_len = target_point_count
            .checked_mul(target_component_count)
            .ok_or_else(|| RusticolError::invalid_argument("routed target shape overflows"))?;
        if target_components.len() != expected_target_len
            || target_point_start
                .checked_add(batch_size)
                .is_none_or(|stop| stop > target_point_count)
        {
            return Err(RusticolError::invalid_argument(
                "routed target components have an inconsistent shape",
            ));
        }

        let scratch = &mut self.routed_reduction_scratch;
        let color_count = physics.manifest.color_components.len();
        physics.selected_color_indices_into(selected_color_ids, &mut scratch.color_indices)?;
        if source_component_count != scratch.color_indices.len() {
            return Err(RusticolError::integrity(format!(
                "topology replay materialized-helicity source has {source_component_count} components, expected {}",
                scratch.color_indices.len()
            )));
        }
        for route in &replay_entry.routes {
            if route.source_index >= source_component_count
                || route.target_index >= target_component_count
            {
                return Err(RusticolError::integrity(
                    "LC topology replay selected route is out of bounds",
                ));
            }
        }
        let color_positions = SelectedAxisPositions::new(
            &scratch.color_indices,
            color_count,
            &mut scratch.color_positions,
        );

        scratch.selected_member_weights.clear();
        scratch.selected_member_weight_ranges.clear();
        for group in &self.raw_sum_groups {
            let start = scratch.selected_member_weights.len();
            let reduction = physics
                .numeric_reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "resolved metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            if !reduction.contains_helicity(helicity_index)
                || !group
                    .indices
                    .iter()
                    .any(|index| root_factors[*index].is_some())
            {
                scratch.selected_member_weight_ranges.push(start..start);
                continue;
            }
            for (color_index, weight) in reduction.normalized_color_weights.iter().copied() {
                if let Some(color_position) = color_positions.position(color_index) {
                    scratch
                        .selected_member_weights
                        .push((0, color_position, weight));
                }
            }
            scratch
                .selected_member_weight_ranges
                .push(start..scratch.selected_member_weights.len());
        }

        let source_scalar_count =
            source_component_count
                .checked_mul(batch_size)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("routed source scratch shape overflows usize")
                })?;
        scratch.direct_group_re.resize(batch_size, 0.0);
        scratch.direct_group_im.resize(batch_size, 0.0);
        scratch.direct_totals.resize(batch_size, 0.0);
        scratch
            .direct_source_components
            .resize(source_scalar_count, 0.0);
        scratch.direct_source_components.fill(0.0);
        for (group, color_weight_range) in self
            .raw_sum_groups
            .iter()
            .zip(&scratch.selected_member_weight_ranges)
        {
            if color_weight_range.is_empty() {
                continue;
            }
            scratch.direct_group_re.fill(0.0);
            scratch.direct_group_im.fill(0.0);
            for index in &group.indices {
                let Some(factor) = root_factors[*index] else {
                    continue;
                };
                let (plane_re, plane_im) = amplitudes.plane_unchecked(*index);
                for point in 0..batch_size {
                    let value = c64(plane_re[point], plane_im[point]) * factor;
                    scratch.direct_group_re[point] += value.re;
                    scratch.direct_group_im[point] += value.im;
                }
            }
            for point in 0..batch_size {
                let re = scratch.direct_group_re[point];
                let im = scratch.direct_group_im[point];
                scratch.direct_totals[point] =
                    normalization_factor * group.all_sector_weight * (re * re + im * im);
            }
            for (_, color_position, weight) in
                &scratch.selected_member_weights[color_weight_range.clone()]
            {
                let source_start = *color_position * batch_size;
                for point in 0..batch_size {
                    scratch.direct_source_components[source_start + point] +=
                        scratch.direct_totals[point] * *weight;
                }
            }
        }

        for route in &replay_entry.routes {
            let source_start = route.source_index * batch_size;
            for point in 0..batch_size {
                let target =
                    (target_point_start + point) * target_component_count + route.target_index;
                target_components[target] +=
                    route.weight * scratch.direct_source_components[source_start + point];
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_amplitude_samples_f64_for_materialized_helicity_add_into<S: AmplitudeSamples>(
        amplitudes: S,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
        output_length: usize,
        raw_sum_groups: &[RawSumGroup],
        has_coherent_groups: bool,
        color_contraction: &mut Option<ColorContractionRuntime>,
        resolved_source_row_scratch_f64: &mut Vec<f64>,
        scratch: &mut RoutedReductionScratch,
    ) -> RusticolResult<()> {
        if output.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "materialized helicity output has length {}, expected {batch_size}",
                output.len()
            )));
        }
        if root_factors.len() != output_length {
            return Err(RusticolError::integrity(format!(
                "helicity recurrence route table has {} roots, expected {}",
                root_factors.len(),
                output_length
            )));
        }
        if helicity_index >= physics.manifest.helicities.len() {
            return Err(RusticolError::selector(format!(
                "runtime helicity index {helicity_index} is out of range"
            )));
        }
        if let Some(contraction) = color_contraction.as_mut() {
            if !physics.has_contracted_color_axis() || physics.manifest.color_components.len() != 1
            {
                return Err(RusticolError::invalid_argument(
                    "materialized-helicity color contraction requires one contracted color component",
                ));
            }
            if raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            if let Some(ids) = selected_color_ids
                && (ids.len() != 1
                    || !ids
                        .iter()
                        .all(|id| physics.color_index_by_id.contains_key(id)))
            {
                return Err(RusticolError::selector(
                    "materialized-helicity color selection does not match the contracted axis",
                ));
            }
            contraction
                .group_scratch_f64
                .resize(contraction.group_count, c64(0.0, 0.0));
            for (row, target) in output.iter_mut().enumerate() {
                for (group_index, group) in raw_sum_groups.iter().enumerate() {
                    let reduction = physics
                        .numeric_reduction_by_group_id
                        .get(&group.id)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(format!(
                                "resolved metadata is missing coherent group {}",
                                group.id
                            ))
                        })?;
                    let active = reduction.contains_helicity(helicity_index)
                        && group
                            .indices
                            .iter()
                            .any(|index| root_factors[*index].is_some());
                    let mut sum = c64(0.0, 0.0);
                    if active {
                        for index in &group.indices {
                            if let Some(factor) = root_factors[*index] {
                                sum += amplitudes.value(row, *index, output_length) * factor;
                            }
                        }
                    }
                    contraction.group_scratch_f64[group_index] = sum;
                }
                let mut contribution = 0.0;
                for entry in contraction.logical_entries() {
                    let left = contraction.group_scratch_f64[entry.left_group_index];
                    let right = contraction.group_scratch_f64[entry.right_group_index];
                    let product = left * right.conj();
                    contribution += normalization_factor
                        * entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im);
                }
                *target += contribution;
            }
            return Ok(());
        }
        if !has_coherent_groups {
            return Err(RusticolError::invalid_argument(
                "materialized helicity reduction requires coherent amplitude-group metadata",
            ));
        }
        let color_count = physics.manifest.color_components.len();
        physics.selected_color_indices_into(selected_color_ids, &mut scratch.color_indices)?;

        scratch.selected_member_weights.clear();
        scratch.selected_member_weight_ranges.clear();
        for group in raw_sum_groups {
            let start = scratch.selected_member_weights.len();
            let reduction = physics
                .numeric_reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "resolved metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            if !reduction.contains_helicity(helicity_index)
                || !group
                    .indices
                    .iter()
                    .any(|index| root_factors[*index].is_some())
            {
                scratch.selected_member_weight_ranges.push(start..start);
                continue;
            }
            for (color_index, weight) in reduction.normalized_color_weights.iter().copied() {
                scratch
                    .selected_member_weights
                    .push((0, color_index, weight));
            }
            let stop = scratch.selected_member_weights.len();
            scratch.selected_member_weight_ranges.push(start..stop);
        }

        resolved_source_row_scratch_f64.resize(color_count, 0.0);
        for (row, target_total) in output.iter_mut().enumerate() {
            let color_row = &mut *resolved_source_row_scratch_f64;
            color_row.fill(0.0);
            for (group, color_weight_range) in raw_sum_groups
                .iter()
                .zip(&scratch.selected_member_weight_ranges)
            {
                if color_weight_range.is_empty() {
                    continue;
                }
                let mut sum = c64(0.0, 0.0);
                for index in &group.indices {
                    if let Some(factor) = root_factors[*index] {
                        sum += amplitudes.value(row, *index, output_length) * factor;
                    }
                }
                let contribution = normalization_factor
                    * group.all_sector_weight
                    * (sum.re * sum.re + sum.im * sum.im);
                for (_, color_index, weight) in
                    &scratch.selected_member_weights[color_weight_range.clone()]
                {
                    color_row[*color_index] += contribution * *weight;
                }
            }
            for color_index in &scratch.color_indices {
                *target_total += color_row[*color_index];
            }
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_materialized_helicity_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
        active_chunk_indices: &[usize],
    ) -> RusticolResult<(ResolvedValues<T>, f64, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let (amplitudes, input_pack_s, evaluator_call_s) = self.evaluate_selected_outputs_generic(
            batch_size,
            state,
            binary_precision,
            Some(active_chunk_indices),
        )?;
        let reduction_start = Instant::now();
        if root_factors.len() != self.output_length {
            return Err(RusticolError::integrity(format!(
                "helicity recurrence route table has {} roots, expected {}",
                root_factors.len(),
                self.output_length
            )));
        }
        let helicity = physics
            .manifest
            .helicities
            .get(helicity_index)
            .ok_or_else(|| {
                RusticolError::selector(format!(
                    "runtime helicity index {helicity_index} is out of range"
                ))
            })?;
        let color_count = physics.manifest.color_components.len();
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut full_values = vec![T::new_zero(); batch_size * color_count];

        if let Some(contraction) = self.color_contraction.as_ref() {
            if color_count != 1 || !physics.has_contracted_color_axis() {
                return Err(RusticolError::invalid_argument(
                    "resolved NLC/full evaluation requires one contracted color component",
                ));
            }
            if self.raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            let mut group_active = vec![false; contraction.group_count];
            for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                let reduction = physics
                    .reduction_by_group_id
                    .get(&group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            group.id
                        ))
                    })?;
                group_active[group_index] = reduction
                    .physical_helicity_ids
                    .iter()
                    .any(|id| id == &helicity.id)
                    && group
                        .indices
                        .iter()
                        .any(|index| root_factors[*index].is_some());
            }
            let mut group_values = vec![complex_zero::<T>(); batch_size * contraction.group_count];
            for (row, full_value) in full_values.iter_mut().enumerate() {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    if !group_active[group_index] {
                        continue;
                    }
                    let mut sum = complex_zero::<T>();
                    for index in &group.indices {
                        let Some(factor) = root_factors[*index] else {
                            continue;
                        };
                        sum += amplitudes[row_offset + *index].clone()
                            * c_generic(T::from(factor.re), T::from(factor.im));
                    }
                    group_values[group_row + group_index] = sum;
                }
                for entry in contraction.logical_entries() {
                    if !group_active[entry.left_group_index]
                        || !group_active[entry.right_group_index]
                    {
                        continue;
                    }
                    let left = &group_values[group_row + entry.left_group_index];
                    let right = &group_values[group_row + entry.right_group_index];
                    let product_re =
                        left.re.clone() * right.re.clone() + left.im.clone() * right.im.clone();
                    let product_im =
                        left.im.clone() * right.re.clone() - left.re.clone() * right.im.clone();
                    *full_value += T::from(normalization_factor * entry.symmetry_factor)
                        * (T::from(entry.weight_re) * product_re
                            - T::from(entry.weight_im) * product_im);
                }
            }
        } else {
            if !self.has_coherent_groups {
                return Err(RusticolError::invalid_argument(
                    "materialized helicity reduction requires coherent amplitude-group metadata",
                ));
            }
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                let color_row = row * color_count;
                for group in &self.raw_sum_groups {
                    let reduction =
                        physics
                            .reduction_by_group_id
                            .get(&group.id)
                            .ok_or_else(|| {
                                RusticolError::invalid_argument(format!(
                                    "resolved metadata is missing coherent group {}",
                                    group.id
                                ))
                            })?;
                    if !reduction
                        .physical_helicity_ids
                        .iter()
                        .any(|id| id == &helicity.id)
                    {
                        continue;
                    }
                    let mut sum = complex_zero::<T>();
                    for index in &group.indices {
                        let Some(factor) = root_factors[*index] else {
                            continue;
                        };
                        sum += amplitudes[row_offset + *index].clone()
                            * c_generic(T::from(factor.re), T::from(factor.im));
                    }
                    let color_weights = reduction
                        .physical_color_ids
                        .iter()
                        .map(|id| {
                            let index = *physics.color_index_by_id.get(id).ok_or_else(|| {
                                RusticolError::artifact(format!(
                                    "resolved reduction group {} references unknown color {id:?}",
                                    group.id
                                ))
                            })?;
                            Ok((
                                index,
                                physics.manifest.color_components[index].coefficient(),
                            ))
                        })
                        .collect::<RusticolResult<Vec<_>>>()?;
                    let total_color_weight =
                        color_weights.iter().map(|(_, weight)| *weight).sum::<f64>();
                    if !total_color_weight.is_finite() || total_color_weight <= 0.0 {
                        return Err(RusticolError::artifact(format!(
                            "resolved reduction group {} has no positive color weight",
                            group.id
                        )));
                    }
                    let contribution = T::from(normalization_factor * group.all_sector_weight)
                        * (sum.re.clone() * sum.re + sum.im.clone() * sum.im);
                    for (color_index, weight) in color_weights {
                        full_values[color_row + color_index] +=
                            contribution.clone() * T::from(weight / total_color_weight);
                    }
                }
            }
        }

        let mut values = Vec::with_capacity(batch_size * color_indices.len());
        for row in 0..batch_size {
            for color_index in &color_indices {
                values.push(full_values[row * color_count + *color_index].clone());
            }
        }
        Ok((
            ResolvedValues {
                values,
                point_count: batch_size,
                helicity_indices: vec![helicity_index],
                color_indices,
            },
            input_pack_s,
            evaluator_call_s,
            reduction_start.elapsed().as_secs_f64(),
        ))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_outputs_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(Vec<Complex<T>>, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        self.evaluate_selected_outputs_generic(batch_size, state, binary_precision, None)
    }

    #[cfg(feature = "symbolica-runtime")]
    fn evaluate_selected_outputs_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
        active_chunk_indices: Option<&[usize]>,
    ) -> RusticolResult<(Vec<Complex<T>>, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let evaluator = self.evaluator.as_mut().ok_or_else(|| {
            RusticolError::integrity(
                "compiled plane-arena reducer cannot execute an exact dense amplitude evaluator",
            )
        })?;
        let mut input_pack_elapsed = Duration::ZERO;
        let (evaluated, evaluator_call_s) = if let Some(input_components) =
            self.input_components.as_ref()
        {
            let local_parameter_count = input_components.len();
            let global_parameter_count = state.len().checked_div(batch_size).ok_or_else(|| {
                RusticolError::invalid_argument("generic amplitude batch size is zero")
            })?;
            let pack_start = Instant::now();
            let mut parameter_scratch =
                vec![complex_zero::<T>(); batch_size * local_parameter_count];
            for row in 0..batch_size {
                let row_state = row * global_parameter_count;
                let row_params = row * local_parameter_count;
                if self.input_spans.is_empty() {
                    for (local_index, global_index) in input_components.iter().enumerate() {
                        parameter_scratch[row_params + local_index] =
                            state[row_state + *global_index].clone();
                    }
                } else {
                    for (local_start, global_start, len) in &self.input_spans {
                        let target_start = row_params + *local_start;
                        let source_start = row_state + *global_start;
                        parameter_scratch[target_start..target_start + *len]
                            .clone_from_slice(&state[source_start..source_start + *len]);
                    }
                }
            }
            input_pack_elapsed = pack_start.elapsed();
            let eval_start = Instant::now();
            let evaluated = evaluator.evaluate_selected_chunks_generic(
                batch_size,
                &parameter_scratch,
                binary_precision,
                active_chunk_indices,
            )?;
            (evaluated, eval_start.elapsed().as_secs_f64())
        } else {
            let eval_start = Instant::now();
            let evaluated = evaluator.evaluate_selected_chunks_generic(
                batch_size,
                state,
                binary_precision,
                active_chunk_indices,
            )?;
            (evaluated, eval_start.elapsed().as_secs_f64())
        };
        if evaluated.len() != batch_size * self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "generic amplitude output buffer has length {}, expected {}",
                evaluated.len(),
                batch_size * self.output_length
            )));
        }
        Ok((
            evaluated,
            profile_duration_seconds(input_pack_elapsed),
            evaluator_call_s,
        ))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_raw_sums_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(Vec<T>, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let (evaluated, input_pack_s, evaluator_call_s) =
            self.evaluate_outputs_generic(batch_size, state, binary_precision)?;
        let mut raw_sums = vec![T::new_zero(); batch_size];
        if let Some(contraction) = self.color_contraction.as_ref() {
            if self.raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            let mut group_values = vec![complex_zero::<T>(); batch_size * contraction.group_count];
            for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    let mut sum = complex_zero::<T>();
                    for index in &group.indices {
                        sum.re += evaluated[row_offset + *index].re.clone();
                        sum.im += evaluated[row_offset + *index].im.clone();
                    }
                    group_values[group_row + group_index] = sum;
                }
                for entry in contraction.logical_entries() {
                    let left = &group_values[group_row + entry.left_group_index];
                    let right = &group_values[group_row + entry.right_group_index];
                    let product_re =
                        left.re.clone() * right.re.clone() + left.im.clone() * right.im.clone();
                    let product_im =
                        left.im.clone() * right.re.clone() - left.re.clone() * right.im.clone();
                    *raw_sum += T::from(entry.symmetry_factor)
                        * (T::from(entry.weight_re) * product_re
                            - T::from(entry.weight_im) * product_im);
                }
            }
            return Ok((raw_sums, input_pack_s, evaluator_call_s));
        }
        for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
            let row_offset = row * self.output_length;
            if self.has_coherent_groups {
                for group in &self.raw_sum_groups {
                    let mut sum_re = T::new_zero();
                    let mut sum_im = T::new_zero();
                    for index in &group.indices {
                        let value = &evaluated[row_offset + *index];
                        sum_re += value.re.clone();
                        sum_im += value.im.clone();
                    }
                    *raw_sum += T::from(group.all_sector_weight)
                        * (sum_re.clone() * sum_re + sum_im.clone() * sum_im);
                }
                continue;
            }
            for index in 0..self.output_length {
                let value = &evaluated[row_offset + index];
                *raw_sum += T::from(self.raw_sum_all_sector_weights[index])
                    * (value.re.clone() * value.re.clone() + value.im.clone() * value.im.clone());
            }
        }
        Ok((raw_sums, input_pack_s, evaluator_call_s))
    }

    #[cfg(feature = "symbolica-runtime")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_resolved_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<T>, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let (evaluated, input_pack_s, evaluator_call_s) =
            self.evaluate_outputs_generic(batch_size, state, binary_precision)?;
        let helicity_count = physics.manifest.helicities.len();
        let color_count = physics.manifest.color_components.len();
        let mut full_values = vec![T::new_zero(); batch_size * helicity_count * color_count];
        if let Some(contraction) = self.color_contraction.as_ref() {
            if color_count != 1 || !physics.has_contracted_color_axis() {
                return Err(RusticolError::invalid_argument(
                    "resolved NLC/full evaluation requires one contracted color component",
                ));
            }
            let mut group_values = vec![complex_zero::<T>(); batch_size * contraction.group_count];
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    let mut sum = complex_zero::<T>();
                    for index in &group.indices {
                        sum.re += evaluated[row_offset + *index].re.clone();
                        sum.im += evaluated[row_offset + *index].im.clone();
                    }
                    group_values[group_row + group_index] = sum;
                }
                for entry in contraction.logical_entries() {
                    let left_group = &self.raw_sum_groups[entry.left_group_index];
                    let right_group = &self.raw_sum_groups[entry.right_group_index];
                    let left_reduction = physics
                        .reduction_by_group_id
                        .get(&left_group.id)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(format!(
                                "resolved metadata is missing coherent group {}",
                                left_group.id
                            ))
                        })?;
                    let right_reduction = physics
                        .reduction_by_group_id
                        .get(&right_group.id)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(format!(
                                "resolved metadata is missing coherent group {}",
                                right_group.id
                            ))
                        })?;
                    if left_reduction.physical_helicity_ids != right_reduction.physical_helicity_ids
                    {
                        return Err(RusticolError::invalid_argument(
                            "colour contraction mixed distinct physical helicities",
                        ));
                    }
                    let left_weights = physics.normalized_helicity_weights(left_reduction)?;
                    let right_weights = physics.normalized_helicity_weights(right_reduction)?;
                    if left_weights.len() != right_weights.len()
                        || left_weights.iter().zip(&right_weights).any(
                            |((left_index, left_weight), (right_index, right_weight))| {
                                left_index != right_index
                                    || left_weight.to_bits() != right_weight.to_bits()
                            },
                        )
                    {
                        return Err(RusticolError::invalid_argument(
                            "colour contraction groups have inconsistent helicity weights",
                        ));
                    }
                    let left = &group_values[group_row + entry.left_group_index];
                    let right = &group_values[group_row + entry.right_group_index];
                    let product_re =
                        left.re.clone() * right.re.clone() + left.im.clone() * right.im.clone();
                    let product_im =
                        left.im.clone() * right.re.clone() - left.re.clone() * right.im.clone();
                    let coefficient = normalization_factor * entry.symmetry_factor;
                    let contribution = T::from(coefficient)
                        * (T::from(entry.weight_re) * product_re
                            - T::from(entry.weight_im) * product_im);
                    for (helicity_index, weight) in left_weights {
                        full_values[(row * helicity_count + helicity_index) * color_count] +=
                            contribution.clone() * T::from(weight);
                    }
                }
            }
        } else {
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                for group in &self.raw_sum_groups {
                    let reduction =
                        physics
                            .reduction_by_group_id
                            .get(&group.id)
                            .ok_or_else(|| {
                                RusticolError::invalid_argument(format!(
                                    "resolved metadata is missing coherent group {}",
                                    group.id
                                ))
                            })?;
                    let mut sum_re = T::new_zero();
                    let mut sum_im = T::new_zero();
                    for index in &group.indices {
                        sum_re += evaluated[row_offset + *index].re.clone();
                        sum_im += evaluated[row_offset + *index].im.clone();
                    }
                    let contribution = T::from(normalization_factor * group.all_sector_weight)
                        * (sum_re.clone() * sum_re + sum_im.clone() * sum_im);
                    for (helicity_index, color_index, weight) in
                        physics.normalized_member_weights(reduction)?
                    {
                        full_values[(row * helicity_count + helicity_index) * color_count
                            + color_index] += contribution.clone() * T::from(weight);
                    }
                }
            }
        }
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut values =
            Vec::with_capacity(batch_size * helicity_indices.len() * color_indices.len());
        for row in 0..batch_size {
            for helicity_index in &helicity_indices {
                for color_index in &color_indices {
                    values.push(
                        full_values
                            [(row * helicity_count + *helicity_index) * color_count + *color_index]
                            .clone(),
                    );
                }
            }
        }
        Ok((
            ResolvedValues {
                values,
                point_count: batch_size,
                helicity_indices,
                color_indices,
            },
            input_pack_s,
            evaluator_call_s,
        ))
    }
}

fn reduce_compiled_symmetric_group_planes(
    amplitudes: crate::direct_arena::DirectAmplitudePlanes<'_>,
    output: &mut [f64],
    groups: &[RawSumGroup],
    symmetric_group: &mut CompiledSymmetricGroupColorContraction,
) -> RusticolResult<()> {
    let point_count = amplitudes.point_count() as usize;
    if output.len() != point_count {
        return Err(RusticolError::invalid_argument(
            "compiled symmetric-group FFT output has the wrong point count",
        ));
    }
    let reducer = match symmetric_group.plan.runtime_reducer() {
        Some(crate::recurrence::RuntimeColorContractionReducer::SymmetricGroupFourier(reducer)) => {
            reducer
        }
        _ => {
            return Err(RusticolError::integrity(
                "compiled symmetric-group FFT payload lost its runtime reducer",
            ));
        }
    };
    let ordered_group_indices = &symmetric_group.ordered_group_indices;
    let workspace = symmetric_group.workspace.as_mut().ok_or_else(|| {
        RusticolError::integrity("compiled symmetric-group FFT workspace was not cold-prepared")
    })?;
    reducer.reduce_lanes(workspace, point_count, |local_group, point| {
        let group_index = *ordered_group_indices.get(local_group).ok_or_else(|| {
            RusticolError::integrity(
                "compiled symmetric-group FFT local group binding is incomplete",
            )
        })?;
        let group = groups.get(group_index).ok_or_else(|| {
            RusticolError::integrity(
                "compiled symmetric-group FFT coherent group binding is out of bounds",
            )
        })?;
        let mut real = 0.0;
        let mut imaginary = 0.0;
        for output_index in &group.indices {
            let (plane_re, plane_im) = amplitudes.plane_unchecked(*output_index);
            real += plane_re[point];
            imaginary += plane_im[point];
        }
        Ok((real, imaginary))
    })?;
    output.copy_from_slice(workspace.reduced(point_count)?);
    Ok(())
}

pub(crate) fn build_raw_sum_groups(
    output_length: usize,
    weights: &[f64],
    all_sector_weights: &[f64],
    group_ids: &[Option<i64>],
    color_sector_ids: &[Option<i64>],
) -> RusticolResult<Vec<RawSumGroup>> {
    if weights.len() != output_length
        || all_sector_weights.len() != output_length
        || group_ids.len() != output_length
        || color_sector_ids.len() != output_length
    {
        return Err(RusticolError::invalid_argument(
            "raw-sum group metadata length does not match amplitude outputs",
        ));
    }
    let mut grouped: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    let mut groups = Vec::new();
    for index in 0..output_length {
        if let Some(group_id) = group_ids[index] {
            grouped.entry(group_id).or_default().push(index);
        } else {
            groups.push(RawSumGroup {
                id: index as i64,
                indices: vec![index],
                weight: weights[index],
                all_sector_weight: all_sector_weights[index],
                sector_ids: color_sector_ids[index].into_iter().collect(),
            });
        }
    }
    for (group_id, indices) in grouped {
        let weight = weights[indices[0]];
        let all_sector_weight = all_sector_weights[indices[0]];
        if indices
            .iter()
            .any(|index| (weights[*index] - weight).abs() > 0.0)
        {
            return Err(RusticolError::invalid_argument(format!(
                "coherent amplitude group {group_id} has inconsistent raw-sum weights"
            )));
        }
        if indices
            .iter()
            .any(|index| (all_sector_weights[*index] - all_sector_weight).abs() > 0.0)
        {
            return Err(RusticolError::invalid_argument(format!(
                "coherent amplitude group {group_id} has inconsistent all-sector raw-sum weights"
            )));
        }
        groups.push(RawSumGroup {
            id: group_id,
            sector_ids: unique_color_sector_ids(&indices, color_sector_ids),
            indices,
            weight,
            all_sector_weight,
        });
    }
    Ok(groups)
}

fn build_color_topology_replay_amplitude_runtime(
    manifest: Option<&GenericColorTopologyReplayAmplitudeManifest>,
    contraction_manifest: Option<&GenericColorContractionManifest>,
    materialized_groups: &[RawSumGroup],
) -> RusticolResult<Option<ColorTopologyReplayAmplitudeRuntime>> {
    let Some(manifest) = manifest else {
        return Ok(None);
    };
    if manifest.contract_version != 1
        || manifest.physical_group_count == 0
        || manifest.physical_groups.len() != manifest.physical_group_count
        || manifest.mappings.is_empty()
    {
        return Err(RusticolError::artifact(
            "color topology replay amplitude gather has an invalid header",
        ));
    }
    let source_index_by_id = materialized_groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.id, index))
        .collect::<BTreeMap<_, _>>();
    if source_index_by_id.len() != materialized_groups.len() {
        return Err(RusticolError::artifact(
            "color topology replay materialized coherent-group ids are not unique",
        ));
    }
    let mut physical_groups = Vec::with_capacity(manifest.physical_group_count);
    let mut physical_group_helicities = Vec::with_capacity(manifest.physical_group_count);
    for (index, group) in manifest.physical_groups.iter().enumerate() {
        if group.group_id != index as i64
            || group.color_sector_id < 0
            || group.color_word.is_empty()
            || group.helicities.is_empty()
            || !group.helicity_weight.is_finite()
            || group.helicity_weight <= 0.0
        {
            return Err(RusticolError::artifact(
                "color topology replay physical coherent-group metadata is invalid",
            ));
        }
        physical_groups.push(RawSumGroup {
            id: group.group_id,
            indices: vec![index],
            weight: group.helicity_weight,
            all_sector_weight: group.helicity_weight,
            sector_ids: vec![group.color_sector_id],
        });
        physical_group_helicities.push(group.helicities.clone());
    }
    let color_contraction =
        build_color_contraction_runtime(contraction_manifest, &physical_groups)?.ok_or_else(
            || {
                RusticolError::artifact(
                    "color topology replay requires a complete NLC/full color contraction",
                )
            },
        )?;
    let mut target_coverage = vec![false; manifest.physical_group_count];
    let mut mappings = Vec::with_capacity(manifest.mappings.len());
    for mapping in &manifest.mappings {
        let label_permutation =
            normalized_color_topology_label_permutation(&mapping.label_permutation)?;
        if mapping.group_routes.is_empty() {
            return Err(RusticolError::artifact(
                "color topology replay mapping contains no amplitude routes",
            ));
        }
        let mut group_routes = Vec::with_capacity(mapping.group_routes.len());
        let mut mapping_sources = BTreeSet::new();
        for route in &mapping.group_routes {
            let source_group_index = source_index_by_id
                .get(&route.source_group_id)
                .copied()
                .ok_or_else(|| {
                    RusticolError::artifact(
                        "color topology replay route names an unknown materialized group",
                    )
                })?;
            let target_group_index = usize::try_from(route.target_group_id)
                .ok()
                .filter(|index| *index < manifest.physical_group_count)
                .ok_or_else(|| {
                    RusticolError::artifact(
                        "color topology replay route names an unknown physical group",
                    )
                })?;
            let factor = match route.factor.as_slice() {
                [real, imaginary] if real.is_finite() && imaginary.is_finite() => {
                    c64(*real, *imaginary)
                }
                _ => {
                    return Err(RusticolError::artifact(
                        "color topology replay amplitude factor must be finite complex data",
                    ));
                }
            };
            if factor == c64(0.0, 0.0)
                || !mapping_sources.insert(source_group_index)
                || std::mem::replace(&mut target_coverage[target_group_index], true)
            {
                return Err(RusticolError::artifact(
                    "color topology replay amplitude routes are not a bijection",
                ));
            }
            group_routes.push(ColorTopologyReplayAmplitudeGroupRoute {
                source_group_index,
                target_group_index,
                factor,
            });
        }
        group_routes.sort_by_key(|route| route.target_group_index);
        mappings.push(ColorTopologyReplayAmplitudeMapping {
            label_permutation,
            group_routes,
        });
    }
    if target_coverage.iter().any(|covered| !covered) {
        return Err(RusticolError::artifact(
            "color topology replay amplitude routes do not cover every physical group",
        ));
    }
    Ok(Some(ColorTopologyReplayAmplitudeRuntime {
        mappings,
        physical_groups,
        physical_group_helicities,
        color_contraction: Some(color_contraction),
        unit_weights: vec![1.0; manifest.physical_group_count],
        no_sector_ids: vec![None; manifest.physical_group_count],
        physical_group_scratch_f64: Vec::new(),
        covered_groups: vec![false; manifest.physical_group_count],
    }))
}

fn normalized_color_topology_label_permutation(
    values: &[LcTopologyReplayLabelPermutationManifest],
) -> RusticolResult<Vec<(usize, usize)>> {
    let mut representatives = BTreeSet::new();
    let mut sectors = BTreeSet::new();
    let mut mapping = Vec::new();
    for item in values {
        if item.representative_label == 0 || item.sector_label == 0 {
            return Err(RusticolError::artifact(
                "color topology replay label permutations use one-based labels",
            ));
        }
        let representative = item.representative_label - 1;
        let sector = item.sector_label - 1;
        if !representatives.insert(representative) || !sectors.insert(sector) {
            return Err(RusticolError::artifact(
                "color topology replay label permutation is not one-to-one",
            ));
        }
        if representative != sector {
            mapping.push((representative, sector));
        }
    }
    if representatives != sectors {
        return Err(RusticolError::artifact(
            "color topology replay label permutation is not a permutation",
        ));
    }
    mapping.sort_unstable();
    Ok(mapping)
}

pub(crate) fn unique_color_sector_ids(
    indices: &[usize],
    color_sector_ids: &[Option<i64>],
) -> Vec<i64> {
    indices
        .iter()
        .filter_map(|index| color_sector_ids.get(*index).copied().flatten())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn raw_sum_group_is_selected(
    group: &RawSumGroup,
    selected_color_sector_ids: Option<&BTreeSet<i64>>,
) -> bool {
    let Some(selected) = selected_color_sector_ids else {
        return true;
    };
    group
        .sector_ids
        .iter()
        .any(|sector_id| selected.contains(sector_id))
}

pub(crate) fn raw_sum_index_is_selected(
    sector_id: Option<i64>,
    selected_color_sector_ids: Option<&BTreeSet<i64>>,
) -> bool {
    let Some(selected) = selected_color_sector_ids else {
        return true;
    };
    sector_id
        .map(|value| selected.contains(&value))
        .unwrap_or(false)
}

pub(crate) fn build_color_contraction_runtime(
    manifest: Option<&GenericColorContractionManifest>,
    groups: &[RawSumGroup],
) -> RusticolResult<Option<ColorContractionRuntime>> {
    let Some(manifest) = manifest else {
        return Ok(None);
    };
    if !manifest.supported {
        return Err(RusticolError::invalid_argument(format!(
            "generic colour contraction is unsupported: {}",
            manifest
                .reason
                .as_deref()
                .unwrap_or("no diagnostic was provided")
        )));
    }
    let group_index_by_id = groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.id, index))
        .collect::<BTreeMap<_, _>>();
    if group_index_by_id.len() != manifest.group_count {
        return Err(RusticolError::invalid_argument(format!(
            "colour contraction declares {} groups but runtime has {} coherent groups",
            manifest.group_count,
            group_index_by_id.len()
        )));
    }
    if let Some(repeated) = manifest.repeated_block.as_ref() {
        if !manifest.entries.is_empty() {
            return Err(RusticolError::invalid_argument(
                "colour contraction cannot mix expanded entries with a repeated block",
            ));
        }
        if repeated.component_count < 2 {
            return Err(RusticolError::invalid_argument(
                "repeated colour contraction must contain at least two components",
            ));
        }
        if repeated.component_group_ids.len() != manifest.group_count
            || repeated.component_group_ids.len() % repeated.component_count != 0
        {
            return Err(RusticolError::invalid_argument(
                "repeated colour contraction group mapping has an inconsistent shape",
            ));
        }
        let groups_per_component = repeated.component_group_ids.len() / repeated.component_count;
        if groups_per_component == 0 {
            return Err(RusticolError::invalid_argument(
                "repeated colour contraction components cannot be empty",
            ));
        }
        let mut mapped_group_ids = BTreeSet::new();
        let mut component_group_indices = Vec::with_capacity(manifest.group_count);
        for group_id in &repeated.component_group_ids {
            if !mapped_group_ids.insert(*group_id) {
                return Err(RusticolError::invalid_argument(
                    "repeated colour contraction maps a coherent group more than once",
                ));
            }
            component_group_indices.push(*group_index_by_id.get(group_id).ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "repeated colour contraction references unknown group {group_id}"
                ))
            })?);
        }
        let mut entries = Vec::with_capacity(repeated.entries.len());
        for entry in &repeated.entries {
            if entry.weight.len() != 2 {
                return Err(RusticolError::invalid_argument(
                    "repeated colour contraction weight must have two components",
                ));
            }
            if entry.left_group_index >= groups_per_component
                || entry.right_group_index >= groups_per_component
            {
                return Err(RusticolError::invalid_argument(
                    "repeated colour contraction entry index is out of bounds",
                ));
            }
            entries.push(ColorContractionEntry {
                left_group_index: entry.left_group_index,
                right_group_index: entry.right_group_index,
                weight_re: entry.weight.first().copied().unwrap_or(0.0),
                weight_im: entry.weight.get(1).copied().unwrap_or(0.0),
                symmetry_factor: entry.symmetry_factor,
            });
        }
        let (walsh_block, c2k_walsh_block) = build_walsh_color_contraction_blocks(
            repeated.factorized_block.as_ref(),
            groups_per_component,
            &entries,
        )?;
        return Ok(Some(ColorContractionRuntime::from_repeated_block(
            groups,
            repeated.component_count,
            component_group_indices,
            entries,
            walsh_block,
            c2k_walsh_block,
        )));
    }
    let mut entries = Vec::with_capacity(manifest.entries.len());
    for entry in &manifest.entries {
        let left_group_index = *group_index_by_id.get(&entry.left_group_id).ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "colour contraction references unknown left group {}",
                entry.left_group_id
            ))
        })?;
        let right_group_index = *group_index_by_id
            .get(&entry.right_group_id)
            .ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "colour contraction references unknown right group {}",
                    entry.right_group_id
                ))
            })?;
        let weight_re = entry.weight.first().copied().unwrap_or(0.0);
        let weight_im = entry.weight.get(1).copied().unwrap_or(0.0);
        entries.push(ColorContractionEntry {
            left_group_index,
            right_group_index,
            weight_re,
            weight_im,
            symmetry_factor: entry.symmetry_factor,
        });
    }
    Ok(Some(ColorContractionRuntime::new(groups, entries)))
}

pub(crate) fn build_compiled_symmetric_group_color_contraction_runtime(
    manifest: Option<&GenericColorContractionManifest>,
    groups: &[RawSumGroup],
    plan: crate::recurrence::RecurrenceColorContraction,
) -> RusticolResult<ColorContractionRuntime> {
    let manifest = manifest.ok_or_else(|| {
        RusticolError::integrity("compiled symmetric-group FFT payload has no contraction metadata")
    })?;
    if !manifest.supported
        || manifest.group_count != groups.len()
        || !manifest.entries.is_empty()
        || manifest.repeated_block.is_some()
    {
        return Err(RusticolError::integrity(
            "compiled symmetric-group FFT metadata disagrees with coherent groups",
        ));
    }
    if plan.accuracy() != crate::recurrence::RecurrenceColorAccuracy::Full
        || plan.storage() != crate::recurrence::RecurrenceColorStorage::ConvolutionKernels
        || plan.component_count() != 1
        || plan.group_count() as usize != groups.len()
        || plan.local_group_count() as usize != groups.len()
        || plan.destination_count() as usize != groups.len()
        || plan.active_sector_count() != groups.len()
        || !matches!(
            plan.runtime_reducer(),
            Some(crate::recurrence::RuntimeColorContractionReducer::SymmetricGroupFourier(_))
        )
    {
        return Err(RusticolError::compatibility(
            "compiled symmetric-group FFT diagnostic payload is not a one-component FullColour active-sector convolution",
        ));
    }
    let group_index_by_id = groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.id, index))
        .collect::<BTreeMap<_, _>>();
    if group_index_by_id.len() != groups.len() {
        return Err(RusticolError::integrity(
            "compiled symmetric-group FFT coherent group IDs are not unique",
        ));
    }
    let mut ordered_group_indices = Vec::with_capacity(groups.len());
    let mut covered = BTreeSet::new();
    for local_group in 0..groups.len() {
        let destination_id = plan.ordered_destination_id(local_group, 0).ok_or_else(|| {
            RusticolError::integrity(
                "compiled symmetric-group FFT local group has no coherent destination",
            )
        })?;
        let group_index = *group_index_by_id
            .get(&i64::from(destination_id))
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "compiled symmetric-group FFT references unknown coherent group {destination_id}"
                ))
            })?;
        if !covered.insert(group_index) {
            return Err(RusticolError::integrity(
                "compiled symmetric-group FFT maps a coherent group more than once",
            ));
        }
        let ordered_group_id = *plan.ordered_group_ids().get(local_group).ok_or_else(|| {
            RusticolError::integrity("compiled symmetric-group FFT ordered group map is incomplete")
        })? as usize;
        let sector_id = *plan
            .sector_by_group()
            .get(ordered_group_id)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "compiled symmetric-group FFT ordered sector map is incomplete",
                )
            })?;
        if groups[group_index].sector_ids.as_slice() != [i64::from(sector_id)] {
            return Err(RusticolError::integrity(
                "compiled symmetric-group FFT coherent group sector identity disagrees with its payload",
            ));
        }
        ordered_group_indices.push(group_index);
    }
    if covered.len() != groups.len() {
        return Err(RusticolError::integrity(
            "compiled symmetric-group FFT does not exhaust coherent groups",
        ));
    }
    Ok(ColorContractionRuntime::from_symmetric_group(
        groups,
        plan,
        ordered_group_indices,
    ))
}

fn build_walsh_color_contraction_blocks(
    manifest: Option<&GenericFactorizedColorContractionBlockManifest>,
    local_group_count: usize,
    entries: &[ColorContractionEntry],
) -> RusticolResult<(
    Option<WalshColorContractionBlock>,
    Option<C2kWalshColorContractionBlock>,
)> {
    let Some(manifest) = manifest else {
        return Ok((None, None));
    };
    match manifest {
        GenericFactorizedColorContractionBlockManifest::KleinFourWalsh { cosets } => Ok((
            Some(build_klein_four_walsh_color_contraction_block(
                cosets,
                local_group_count,
                entries,
            )?),
            None,
        )),
        GenericFactorizedColorContractionBlockManifest::ElementaryAbelianWalsh { rank, cosets } => {
            Ok((
                None,
                Some(build_c2k_walsh_color_contraction_block(
                    *rank,
                    cosets,
                    local_group_count,
                    entries,
                )?),
            ))
        }
    }
}

fn build_klein_four_walsh_color_contraction_block(
    cosets: &[[usize; 4]],
    local_group_count: usize,
    entries: &[ColorContractionEntry],
) -> RusticolResult<WalshColorContractionBlock> {
    let mapped_group_count = cosets.len().checked_mul(4).ok_or_else(|| {
        RusticolError::invalid_argument("factorized color contraction coset count overflows")
    })?;
    if cosets.is_empty() || mapped_group_count != local_group_count {
        return Err(RusticolError::invalid_argument(
            "factorized color contraction cosets do not match local groups",
        ));
    }
    let mut mapped_groups = vec![false; local_group_count];
    for coset in cosets {
        for local_group_index in coset {
            let Some(mapped) = mapped_groups.get_mut(*local_group_index) else {
                return Err(RusticolError::invalid_argument(
                    "factorized color contraction coset index is out of bounds",
                ));
            };
            if *mapped {
                return Err(RusticolError::invalid_argument(
                    "factorized color contraction cosets contain a duplicate index",
                ));
            }
            *mapped = true;
        }
    }

    let matrix_size = local_group_count
        .checked_mul(local_group_count)
        .ok_or_else(|| {
            RusticolError::invalid_argument("factorized color contraction matrix size overflows")
        })?;
    let mut matrix = vec![0.0; matrix_size];
    let mut seen_pairs = BTreeSet::new();
    for entry in entries {
        if entry.weight_im != 0.0 {
            return Err(RusticolError::invalid_argument(
                "factorized color contraction requires real weights",
            ));
        }
        if !entry.weight_re.is_finite() || !entry.symmetry_factor.is_finite() {
            return Err(RusticolError::invalid_argument(
                "factorized color contraction weights must be finite",
            ));
        }
        let (left, right) = if entry.left_group_index <= entry.right_group_index {
            (entry.left_group_index, entry.right_group_index)
        } else {
            (entry.right_group_index, entry.left_group_index)
        };
        if !seen_pairs.insert((left, right)) {
            return Err(RusticolError::invalid_argument(
                "factorized color contraction has a duplicate matrix entry",
            ));
        }
        let weight = if left == right {
            entry.symmetry_factor * entry.weight_re
        } else {
            0.5 * entry.symmetry_factor * entry.weight_re
        };
        if !weight.is_finite() {
            return Err(RusticolError::invalid_argument(
                "factorized color contraction matrix weight is not finite",
            ));
        }
        matrix[left * local_group_count + right] = weight;
        matrix[right * local_group_count + left] = weight;
    }

    for left_coset in cosets {
        for right_coset in cosets {
            for left_subgroup_index in 0..4 {
                for right_subgroup_index in 0..4 {
                    let actual = matrix[left_coset[left_subgroup_index] * local_group_count
                        + right_coset[right_subgroup_index]];
                    let expected = matrix[left_coset[0] * local_group_count
                        + right_coset[left_subgroup_index ^ right_subgroup_index]];
                    if actual != expected {
                        return Err(RusticolError::invalid_argument(
                            "factorized color contraction matrix is not invariant under its Klein-four action",
                        ));
                    }
                }
            }
        }
    }

    let coset_count = cosets.len();
    let mut transformed_entries = Vec::new();
    let characters = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ];
    for (character_index, character) in characters.iter().enumerate() {
        for left_coset_index in 0..coset_count {
            for right_coset_index in left_coset_index..coset_count {
                let left_coset = cosets[left_coset_index];
                let right_coset = cosets[right_coset_index];
                let values = [
                    matrix[left_coset[0] * local_group_count + right_coset[0]],
                    matrix[left_coset[0] * local_group_count + right_coset[1]],
                    matrix[left_coset[0] * local_group_count + right_coset[2]],
                    matrix[left_coset[0] * local_group_count + right_coset[3]],
                ];
                let mut weight = 0.0;
                for (sign, value) in character.iter().zip(values) {
                    weight += sign * value;
                }
                if !weight.is_finite() {
                    return Err(RusticolError::invalid_argument(
                        "factorized color contraction Walsh weight is not finite",
                    ));
                }
                if weight == 0.0 {
                    continue;
                }
                transformed_entries.push(ColorContractionEntry {
                    left_group_index: character_index * coset_count + left_coset_index,
                    right_group_index: character_index * coset_count + right_coset_index,
                    weight_re: weight,
                    weight_im: 0.0,
                    symmetry_factor: if left_coset_index == right_coset_index {
                        1.0
                    } else {
                        2.0
                    },
                });
            }
        }
    }
    Ok(WalshColorContractionBlock {
        cosets: cosets.to_vec(),
        entries: transformed_entries,
    })
}

fn build_c2k_walsh_color_contraction_block(
    rank: usize,
    cosets: &[Vec<usize>],
    local_group_count: usize,
    entries: &[ColorContractionEntry],
) -> RusticolResult<C2kWalshColorContractionBlock> {
    if rank < 3 {
        return Err(RusticolError::invalid_argument(
            "elementary-Abelian Walsh color contraction rank must be at least three",
        ));
    }
    let shift = u32::try_from(rank).map_err(|_| {
        RusticolError::invalid_argument(
            "elementary-Abelian Walsh color contraction rank is too large",
        )
    })?;
    let subgroup_order = 1usize.checked_shl(shift).ok_or_else(|| {
        RusticolError::invalid_argument(
            "elementary-Abelian Walsh color contraction subgroup order overflows",
        )
    })?;
    let mapped_group_count = cosets.len().checked_mul(subgroup_order).ok_or_else(|| {
        RusticolError::invalid_argument("factorized color contraction coset count overflows")
    })?;
    if cosets.is_empty()
        || cosets.iter().any(|coset| coset.len() != subgroup_order)
        || mapped_group_count != local_group_count
    {
        return Err(RusticolError::invalid_argument(
            "elementary-Abelian Walsh color contraction cosets do not match rank or local groups",
        ));
    }

    let mut mapped_groups = vec![false; local_group_count];
    for coset in cosets {
        for local_group_index in coset {
            let Some(mapped) = mapped_groups.get_mut(*local_group_index) else {
                return Err(RusticolError::invalid_argument(
                    "factorized color contraction coset index is out of bounds",
                ));
            };
            if *mapped {
                return Err(RusticolError::invalid_argument(
                    "factorized color contraction cosets contain a duplicate index",
                ));
            }
            *mapped = true;
        }
    }
    if mapped_groups.iter().any(|mapped| !mapped) {
        return Err(RusticolError::invalid_argument(
            "factorized color contraction cosets do not partition local groups",
        ));
    }

    let matrix_size = local_group_count
        .checked_mul(local_group_count)
        .ok_or_else(|| {
            RusticolError::invalid_argument("factorized color contraction matrix size overflows")
        })?;
    let mut matrix = vec![0.0; matrix_size];
    let mut seen_pairs = BTreeSet::new();
    for entry in entries {
        if entry.weight_im != 0.0 {
            return Err(RusticolError::invalid_argument(
                "factorized color contraction requires real weights",
            ));
        }
        if !entry.weight_re.is_finite() || !entry.symmetry_factor.is_finite() {
            return Err(RusticolError::invalid_argument(
                "factorized color contraction weights must be finite",
            ));
        }
        let (left, right) = if entry.left_group_index <= entry.right_group_index {
            (entry.left_group_index, entry.right_group_index)
        } else {
            (entry.right_group_index, entry.left_group_index)
        };
        if !seen_pairs.insert((left, right)) {
            return Err(RusticolError::invalid_argument(
                "factorized color contraction has a duplicate matrix entry",
            ));
        }
        let weight = if left == right {
            entry.symmetry_factor * entry.weight_re
        } else {
            0.5 * entry.symmetry_factor * entry.weight_re
        };
        if !weight.is_finite() {
            return Err(RusticolError::invalid_argument(
                "factorized color contraction matrix weight is not finite",
            ));
        }
        matrix[left * local_group_count + right] = weight;
        matrix[right * local_group_count + left] = weight;
    }

    for left_coset in cosets {
        for right_coset in cosets {
            for left_subgroup_index in 0..subgroup_order {
                for right_subgroup_index in 0..subgroup_order {
                    let actual = matrix[left_coset[left_subgroup_index] * local_group_count
                        + right_coset[right_subgroup_index]];
                    let expected = matrix[left_coset[0] * local_group_count
                        + right_coset[left_subgroup_index ^ right_subgroup_index]];
                    if actual != expected {
                        return Err(RusticolError::invalid_argument(
                            "factorized color contraction matrix is not invariant under its elementary-Abelian XOR action",
                        ));
                    }
                }
            }
        }
    }

    let coset_count = cosets.len();
    let mut transformed_entries = Vec::new();
    for left_coset_index in 0..coset_count {
        for right_coset_index in left_coset_index..coset_count {
            let left_coset = &cosets[left_coset_index];
            let right_coset = &cosets[right_coset_index];
            let mut weights = (0..subgroup_order)
                .map(|subgroup_index| {
                    matrix[left_coset[0] * local_group_count + right_coset[subgroup_index]]
                })
                .collect::<Vec<_>>();
            walsh_butterfly_f64(&mut weights);
            for (character_index, weight) in weights.into_iter().enumerate() {
                if !weight.is_finite() {
                    return Err(RusticolError::invalid_argument(
                        "factorized color contraction Walsh weight is not finite",
                    ));
                }
                if weight == 0.0 {
                    continue;
                }
                transformed_entries.push(ColorContractionEntry {
                    left_group_index: character_index * coset_count + left_coset_index,
                    right_group_index: character_index * coset_count + right_coset_index,
                    // The C2^k runtime deliberately applies the unnormalized
                    // Walsh transform. H H^T = subgroup_order * I, so carry
                    // its inverse once in the transformed matrix instead of
                    // scaling every complex amplitude at every point.
                    weight_re: weight / subgroup_order as f64,
                    weight_im: 0.0,
                    symmetry_factor: if left_coset_index == right_coset_index {
                        1.0
                    } else {
                        2.0
                    },
                });
            }
        }
    }

    Ok(C2kWalshColorContractionBlock {
        subgroup_order,
        cosets: cosets.to_vec(),
        entries: transformed_entries,
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

pub(crate) fn generic_root_group_id(
    root: &GenericAmplitudeRootManifest,
) -> RusticolResult<Option<i64>> {
    let Some(value) = root.coherent_group_id.as_ref() else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(group_id) = value.as_i64() {
        return Ok(Some(group_id));
    }
    if let Some(group_id) = value.as_u64() {
        return i64::try_from(group_id)
            .map(Some)
            .map_err(|_| RusticolError::invalid_argument("generic coherent group id exceeds i64"));
    }
    if let Some(text) = value.as_str() {
        return text.parse::<i64>().map(Some).map_err(|err| {
            RusticolError::invalid_argument(format!(
                "could not parse generic coherent group id {text:?}: {err}"
            ))
        });
    }
    Err(RusticolError::invalid_argument(format!(
        "generic coherent group id for root {} must be an integer or string",
        root.root_id
    )))
}

#[cfg(test)]
mod selected_axis_position_tests {
    use super::*;

    #[test]
    fn sparse_and_complete_axis_positions_match_without_resizing_sparse_scratch() {
        let mut dense = vec![Some(usize::MAX); 17];
        for selected in [&[8][..], &[][..]] {
            let positions = SelectedAxisPositions::new(selected, 10, &mut dense);
            assert!(matches!(positions, SelectedAxisPositions::Sparse(_)));
            for index in 0..10 {
                assert_eq!(
                    positions.position(index),
                    selected.iter().position(|&i| i == index)
                );
            }
            assert_eq!(dense, vec![Some(usize::MAX); 17]);
        }
        for selected in [vec![1, 4, 8], (1..10).collect(), (0..10).collect()] {
            let positions = SelectedAxisPositions::new(&selected, 10, &mut dense);
            assert!(matches!(positions, SelectedAxisPositions::Dense(_)));
            for index in 0..10 {
                assert_eq!(
                    positions.position(index),
                    selected.iter().position(|&i| i == index)
                );
            }
            assert_eq!(dense.len(), 10);
        }
    }
}
