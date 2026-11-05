// SPDX-License-Identifier: 0BSD

use super::super::*;
use super::*;

impl EvaluatorGroup {
    pub(crate) fn load(manifest: &EvaluatorManifest, root: &Path) -> RusticolResult<Self> {
        let payloads = EvaluatorPayloadStore::directory(root);
        Self::load_from_store(manifest, &payloads)
    }

    pub(crate) fn load_from_store(
        manifest: &EvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
    ) -> RusticolResult<Self> {
        ensure_evaluator_capabilities_supported(manifest)?;
        let (input_len, _) = manifest.io_len()?;
        let leaf_layout = manifest.leaf_layout()?;
        let mut evaluators = Vec::new();
        let mut input_mappings = Vec::new();
        for leaf in &leaf_layout {
            let before = evaluators.len();
            flatten_evaluators_from_store(leaf.evaluator, payloads, &mut evaluators)?;
            if evaluators.len() != before + 1 {
                return Err(RusticolError::integrity(
                    "canonical evaluator leaf layout did not load exactly one evaluator",
                ));
            }
            let mapping = if leaf.input_indices.len() == input_len
                && leaf
                    .input_indices
                    .iter()
                    .enumerate()
                    .all(|(expected, index)| expected == *index)
            {
                None
            } else {
                Some(leaf.input_indices.clone())
            };
            input_mappings.push(mapping);
        }
        let input_mapping_spans = input_mappings
            .iter()
            .map(|mapping| {
                mapping
                    .as_deref()
                    .map(contiguous_input_spans)
                    .unwrap_or_default()
            })
            .collect();
        let output_len = leaf_layout.last().map_or(0, |leaf| leaf.output_range.end);
        Ok(Self {
            evaluators,
            input_len,
            input_mappings,
            input_mapping_spans,
            output_len,
            chunk_parameter_scratch_f64: Vec::new(),
            chunk_scratch_f64: Vec::new(),
            chunk_parameter_scratch_aosoa_f64: Vec::new(),
            chunk_scratch_aosoa_f64: Vec::new(),
            chunk_input_mapping_scratch: Vec::new(),
        })
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn load_exact_from_plane(
        stage: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
    ) -> RusticolResult<Self> {
        let manifest = &stage.evaluator;
        let direct = stage.compiled_plane_arena.as_ref().ok_or_else(|| {
            RusticolError::integrity(
                "compiled plane-arena exact evaluator has no serialized bindings",
            )
        })?;
        let (input_len, _) = manifest.io_len()?;
        let leaf_layout = manifest.leaf_layout()?;
        if leaf_layout.len() != direct.leaves.len() {
            return Err(RusticolError::integrity(
                "compiled plane-arena exact leaf bindings are incomplete",
            ));
        }
        let mut evaluators = Vec::new();
        let mut input_mappings = Vec::new();
        for (leaf, binding) in leaf_layout.iter().zip(&direct.leaves) {
            flatten_exact_evaluators_from_store(leaf.evaluator, payloads, &mut evaluators)?;
            let mapping = if binding.input_indices.len() == input_len
                && binding
                    .input_indices
                    .iter()
                    .enumerate()
                    .all(|(expected, index)| expected == *index)
            {
                None
            } else {
                Some(binding.input_indices.clone())
            };
            input_mappings.push(mapping);
        }
        let input_mapping_spans = input_mappings
            .iter()
            .map(|mapping| {
                mapping
                    .as_deref()
                    .map(contiguous_input_spans)
                    .unwrap_or_default()
            })
            .collect();
        let output_len = direct.leaves.last().map_or(0, |leaf| leaf.output_stop);
        Ok(Self {
            evaluators,
            input_len,
            input_mappings,
            input_mapping_spans,
            output_len,
            chunk_parameter_scratch_f64: Vec::new(),
            chunk_scratch_f64: Vec::new(),
            chunk_parameter_scratch_aosoa_f64: Vec::new(),
            chunk_scratch_aosoa_f64: Vec::new(),
            chunk_input_mapping_scratch: Vec::new(),
        })
    }

    pub(crate) fn uses_simd_jit(&self) -> bool {
        self.evaluators
            .iter()
            .any(|evaluator| match &evaluator.eval {
                #[cfg(feature = "symbolica-runtime")]
                F64Evaluator::ExactOnly => false,
                #[cfg(feature = "f64-symjit")]
                F64Evaluator::SymjitApplication(_) => true,
                #[cfg(feature = "symbolica-runtime")]
                F64Evaluator::Jit(_) => true,
                #[cfg(feature = "f64-compiled")]
                F64Evaluator::Compiled(_) => false,
            })
    }

    pub(crate) fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
    ) -> RusticolResult<Vec<Complex<f64>>> {
        let mut out = Vec::new();
        self.evaluate_batch_into(batch_size, params, &mut out)?;
        Ok(out)
    }

    pub(crate) fn evaluate_single_row(
        &mut self,
        params: &[Complex<f64>],
    ) -> RusticolResult<Vec<Complex<f64>>> {
        self.evaluate_batch(1, params)
    }

    pub(crate) fn output_chunk_lengths(&self) -> Vec<usize> {
        self.evaluators
            .iter()
            .map(|evaluator| evaluator.output_len)
            .collect()
    }

    pub(crate) fn is_chunked(&self) -> bool {
        self.evaluators.len() > 1
    }

    pub(crate) fn evaluate_chunks_f64_into_state(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        state_parameter_count: usize,
        state: &mut [Complex<f64>],
        chunk_outputs: &[Vec<(usize, usize)>],
        chunk_output_spans: &[Vec<(usize, usize, usize)>],
    ) -> RusticolResult<()> {
        if chunk_outputs.len() != self.evaluators.len()
            || chunk_output_spans.len() != self.evaluators.len()
        {
            return Err(RusticolError::invalid_argument(
                "stage chunk-output layout does not match evaluator chunks",
            ));
        }
        let expected_state_len = batch_size
            .checked_mul(state_parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("stage state length overflows usize"))?;
        if state.len() != expected_state_len {
            return Err(RusticolError::invalid_argument(format!(
                "stage state has length {}, expected {expected_state_len}",
                state.len()
            )));
        }
        if params.len() != batch_size * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {}",
                params.len(),
                batch_size * self.input_len
            )));
        }
        for ((((evaluator, input_mapping), input_spans), outputs), spans) in self
            .evaluators
            .iter_mut()
            .zip(&self.input_mappings)
            .zip(&self.input_mapping_spans)
            .zip(chunk_outputs)
            .zip(chunk_output_spans)
        {
            if let Some(lane_width) = evaluator
                .simd_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let mapping = input_mapping.as_deref();
                let block_count = pack_f64_parameters_aosoa(
                    params,
                    batch_size,
                    self.input_len,
                    evaluator.input_len,
                    lane_width,
                    mapping,
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                if evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )? {
                    scatter_f64_aosoa_outputs_to_state(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        state_parameter_count,
                        state,
                        outputs,
                        spans,
                    )?;
                    continue;
                }
            }
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                input_mapping.as_deref(),
                input_spans,
                &mut self.chunk_parameter_scratch_f64,
            );
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            self.chunk_scratch_f64
                .resize(batch_size * evaluator.output_len, c64(0.0, 0.0));
            evaluator.evaluate_f64_batch_unpadded(
                batch_size,
                evaluator_params,
                &mut self.chunk_scratch_f64,
            )?;
            for row in 0..batch_size {
                let row_state = row * state_parameter_count;
                let row_eval = row * evaluator.output_len;
                if spans.is_empty() {
                    for (column, state_offset) in outputs {
                        state[row_state + *state_offset] =
                            self.chunk_scratch_f64[row_eval + *column];
                    }
                } else {
                    for (column_start, state_offset_start, len) in spans {
                        let source_start = row_eval + *column_start;
                        let target_start = row_state + *state_offset_start;
                        state[target_start..target_start + *len].copy_from_slice(
                            &self.chunk_scratch_f64[source_start..source_start + *len],
                        );
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn evaluate_chunks_f64_into_state_profile(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        state_parameter_count: usize,
        state: &mut [Complex<f64>],
        chunk_outputs: &[Vec<(usize, usize)>],
        chunk_output_spans: &[Vec<(usize, usize, usize)>],
    ) -> RusticolResult<(EvaluatorBatchProfile, f64)> {
        if chunk_outputs.len() != self.evaluators.len()
            || chunk_output_spans.len() != self.evaluators.len()
        {
            return Err(RusticolError::invalid_argument(
                "stage chunk-output layout does not match evaluator chunks",
            ));
        }
        let expected_state_len = batch_size
            .checked_mul(state_parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("stage state length overflows usize"))?;
        if state.len() != expected_state_len {
            return Err(RusticolError::invalid_argument(format!(
                "stage state has length {}, expected {expected_state_len}",
                state.len()
            )));
        }

        // Keep timing state out of floating-point registers while generated
        // evaluators execute. Some native evaluator ABIs use the full SIMD
        // register file, whereas Duration remains integer-backed.
        let mut leaf_input_pack_elapsed = Duration::ZERO;
        let mut evaluator_elapsed = Duration::ZERO;
        let mut leaf_input_copy_component_count = 0u64;
        let mut backend_call_count = 0u64;
        let mut scratch_reallocation_count = 0u64;
        let mut assign_elapsed = Duration::ZERO;
        if params.len() != batch_size * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {}",
                params.len(),
                batch_size * self.input_len
            )));
        }
        for ((((evaluator, input_mapping), input_spans), outputs), spans) in self
            .evaluators
            .iter_mut()
            .zip(&self.input_mappings)
            .zip(&self.input_mapping_spans)
            .zip(chunk_outputs)
            .zip(chunk_output_spans)
        {
            if let Some(lane_width) = evaluator
                .simd_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let parameter_capacity = self.chunk_parameter_scratch_aosoa_f64.capacity();
                let leaf_start = Instant::now();
                let mapping = input_mapping.as_deref();
                let block_count = pack_f64_parameters_aosoa(
                    params,
                    batch_size,
                    self.input_len,
                    evaluator.input_len,
                    lane_width,
                    mapping,
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                leaf_input_pack_elapsed += leaf_start.elapsed();
                leaf_input_copy_component_count +=
                    (self.chunk_parameter_scratch_aosoa_f64.len() / 2) as u64;
                scratch_reallocation_count += u64::from(
                    self.chunk_parameter_scratch_aosoa_f64.capacity() != parameter_capacity,
                );

                let output_capacity = self.chunk_scratch_aosoa_f64.capacity();
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                scratch_reallocation_count +=
                    u64::from(self.chunk_scratch_aosoa_f64.capacity() != output_capacity);
                let eval_start = Instant::now();
                let evaluated = evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )?;
                evaluator_elapsed += eval_start.elapsed();
                backend_call_count += 1;
                if evaluated {
                    let assign_start = Instant::now();
                    scatter_f64_aosoa_outputs_to_state(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        state_parameter_count,
                        state,
                        outputs,
                        spans,
                    )?;
                    assign_elapsed += assign_start.elapsed();
                    continue;
                }
            }
            let leaf_capacity = self.chunk_parameter_scratch_f64.capacity();
            let leaf_start = Instant::now();
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                input_mapping.as_deref(),
                input_spans,
                &mut self.chunk_parameter_scratch_f64,
            );
            leaf_input_pack_elapsed += leaf_start.elapsed();
            if input_mapping.is_some() {
                leaf_input_copy_component_count += evaluator_params.len() as u64;
            }
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            let chunk_capacity = self.chunk_scratch_f64.capacity();
            self.chunk_scratch_f64
                .resize(batch_size * evaluator.output_len, c64(0.0, 0.0));
            let eval_start = Instant::now();
            evaluator.evaluate_f64_batch_unpadded(
                batch_size,
                evaluator_params,
                &mut self.chunk_scratch_f64,
            )?;
            evaluator_elapsed += eval_start.elapsed();
            backend_call_count += 1;
            scratch_reallocation_count +=
                u64::from(self.chunk_scratch_f64.capacity() != chunk_capacity);
            // The mapped slice is no longer live after the backend call.
            scratch_reallocation_count +=
                u64::from(self.chunk_parameter_scratch_f64.capacity() != leaf_capacity);

            let assign_start = Instant::now();
            for row in 0..batch_size {
                let row_state = row * state_parameter_count;
                let row_eval = row * evaluator.output_len;
                if spans.is_empty() {
                    for (column, state_offset) in outputs {
                        state[row_state + *state_offset] =
                            self.chunk_scratch_f64[row_eval + *column];
                    }
                } else {
                    for (column_start, state_offset_start, len) in spans {
                        let source_start = row_eval + *column_start;
                        let target_start = row_state + *state_offset_start;
                        state[target_start..target_start + *len].copy_from_slice(
                            &self.chunk_scratch_f64[source_start..source_start + *len],
                        );
                    }
                }
            }
            assign_elapsed += assign_start.elapsed();
        }
        let leaf_input_pack_s = profile_duration_seconds(leaf_input_pack_elapsed);
        let evaluator_call_s = profile_duration_seconds(evaluator_elapsed);
        let profile = EvaluatorBatchProfile {
            leaf_input_pack_s,
            legacy_evaluator_call_s: leaf_input_pack_s + evaluator_call_s,
            evaluator_call_s,
            leaf_input_copy_component_count,
            backend_call_count,
            scratch_reallocation_count,
            ..EvaluatorBatchProfile::default()
        };
        Ok((profile, assign_elapsed.as_secs_f64()))
    }

    // The explicit slices avoid constructing a descriptor in this selector hot path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_selected_chunks_f64_into_state(
        &mut self,
        batch_size: usize,
        state_parameter_count: usize,
        state: &mut [Complex<f64>],
        parent_input_components: Option<&[usize]>,
        parent_input_spans: &[(usize, usize, usize)],
        chunk_outputs: &[Vec<(usize, usize)>],
        chunk_output_spans: &[Vec<(usize, usize, usize)>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<()> {
        if chunk_outputs.len() != self.evaluators.len()
            || chunk_output_spans.len() != self.evaluators.len()
        {
            return Err(RusticolError::invalid_argument(
                "stage chunk-output layout does not match evaluator chunks",
            ));
        }
        validate_active_chunk_indices(active_chunk_indices, self.evaluators.len())?;
        let expected_state_len = batch_size
            .checked_mul(state_parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("stage state length overflows usize"))?;
        if state.len() != expected_state_len {
            return Err(RusticolError::invalid_argument(format!(
                "stage state has length {}, expected {expected_state_len}",
                state.len()
            )));
        }
        match parent_input_components {
            Some(components) => {
                if components.len() != self.input_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "stage input-component map has length {}, expected {}",
                        components.len(),
                        self.input_len
                    )));
                }
                if components
                    .iter()
                    .any(|component| *component >= state_parameter_count)
                {
                    return Err(RusticolError::invalid_argument(
                        "stage input-component map references state outside the parameter layout",
                    ));
                }
            }
            None if self.input_len != state_parameter_count => {
                return Err(RusticolError::invalid_argument(format!(
                    "evaluator input length {} does not match stage state parameter count {state_parameter_count}",
                    self.input_len
                )));
            }
            None => {}
        }
        for &chunk_index in active_chunk_indices {
            let evaluator = &mut self.evaluators[chunk_index];
            if let Some(lane_width) = evaluator
                .simd_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let input_mapping = self.input_mappings[chunk_index].as_deref();
                self.chunk_input_mapping_scratch.clear();
                self.chunk_input_mapping_scratch
                    .reserve(evaluator.input_len);
                for local_component in 0..evaluator.input_len {
                    let parent_component =
                        input_mapping.map_or(local_component, |indices| indices[local_component]);
                    self.chunk_input_mapping_scratch.push(
                        parent_input_components
                            .map_or(parent_component, |components| components[parent_component]),
                    );
                }
                let block_count = pack_f64_parameters_aosoa(
                    state,
                    batch_size,
                    state_parameter_count,
                    evaluator.input_len,
                    lane_width,
                    Some(&self.chunk_input_mapping_scratch),
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                if evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )? {
                    scatter_f64_aosoa_outputs_to_state(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        state_parameter_count,
                        state,
                        &chunk_outputs[chunk_index],
                        &chunk_output_spans[chunk_index],
                    )?;
                    continue;
                }
            }
            let evaluator_params = selected_chunk_f64_parameters(
                state,
                batch_size,
                state_parameter_count,
                self.input_len,
                parent_input_components,
                parent_input_spans,
                self.input_mappings[chunk_index].as_deref(),
                &self.input_mapping_spans[chunk_index],
                &mut self.chunk_parameter_scratch_f64,
            )?;
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            let scratch_len = batch_size
                .checked_mul(evaluator.output_len)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("stage chunk output length overflows usize")
                })?;
            self.chunk_scratch_f64.resize(scratch_len, c64(0.0, 0.0));
            evaluator.evaluate_f64_batch_unpadded(
                batch_size,
                evaluator_params,
                &mut self.chunk_scratch_f64,
            )?;
            let outputs = &chunk_outputs[chunk_index];
            let spans = &chunk_output_spans[chunk_index];
            for row in 0..batch_size {
                let row_state = row * state_parameter_count;
                let row_eval = row * evaluator.output_len;
                if spans.is_empty() {
                    for (column, state_offset) in outputs {
                        state[row_state + *state_offset] =
                            self.chunk_scratch_f64[row_eval + *column];
                    }
                } else {
                    for (column_start, state_offset_start, len) in spans {
                        let source_start = row_eval + *column_start;
                        let target_start = row_state + *state_offset_start;
                        state[target_start..target_start + *len].copy_from_slice(
                            &self.chunk_scratch_f64[source_start..source_start + *len],
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_selected_chunks_f64_into_state_profile(
        &mut self,
        batch_size: usize,
        state_parameter_count: usize,
        state: &mut [Complex<f64>],
        parent_input_components: Option<&[usize]>,
        parent_input_spans: &[(usize, usize, usize)],
        chunk_outputs: &[Vec<(usize, usize)>],
        chunk_output_spans: &[Vec<(usize, usize, usize)>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<(EvaluatorBatchProfile, f64)> {
        if chunk_outputs.len() != self.evaluators.len()
            || chunk_output_spans.len() != self.evaluators.len()
        {
            return Err(RusticolError::invalid_argument(
                "stage chunk-output layout does not match evaluator chunks",
            ));
        }
        validate_active_chunk_indices(active_chunk_indices, self.evaluators.len())?;
        let expected_state_len = batch_size
            .checked_mul(state_parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("stage state length overflows usize"))?;
        if state.len() != expected_state_len {
            return Err(RusticolError::invalid_argument(format!(
                "stage state has length {}, expected {expected_state_len}",
                state.len()
            )));
        }
        match parent_input_components {
            Some(components) => {
                if components.len() != self.input_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "stage input-component map has length {}, expected {}",
                        components.len(),
                        self.input_len
                    )));
                }
                if components
                    .iter()
                    .any(|component| *component >= state_parameter_count)
                {
                    return Err(RusticolError::invalid_argument(
                        "stage input-component map references state outside the parameter layout",
                    ));
                }
            }
            None if self.input_len != state_parameter_count => {
                return Err(RusticolError::invalid_argument(format!(
                    "evaluator input length {} does not match stage state parameter count {state_parameter_count}",
                    self.input_len
                )));
            }
            None => {}
        }

        let mut input_pack_elapsed = Duration::ZERO;
        let mut evaluator_elapsed = Duration::ZERO;
        let mut assign_elapsed = Duration::ZERO;
        let mut leaf_input_copy_component_count = 0u64;
        let mut backend_call_count = 0u64;
        let mut scratch_reallocation_count = 0u64;
        for &chunk_index in active_chunk_indices {
            let evaluator = &mut self.evaluators[chunk_index];
            if let Some(lane_width) = evaluator
                .simd_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let parameter_capacity = self.chunk_parameter_scratch_aosoa_f64.capacity();
                let input_pack_start = Instant::now();
                let input_mapping = self.input_mappings[chunk_index].as_deref();
                let mapping_capacity = self.chunk_input_mapping_scratch.capacity();
                self.chunk_input_mapping_scratch.clear();
                self.chunk_input_mapping_scratch
                    .reserve(evaluator.input_len);
                for local_component in 0..evaluator.input_len {
                    let parent_component =
                        input_mapping.map_or(local_component, |indices| indices[local_component]);
                    self.chunk_input_mapping_scratch.push(
                        parent_input_components
                            .map_or(parent_component, |components| components[parent_component]),
                    );
                }
                scratch_reallocation_count +=
                    u64::from(self.chunk_input_mapping_scratch.capacity() != mapping_capacity);
                let block_count = pack_f64_parameters_aosoa(
                    state,
                    batch_size,
                    state_parameter_count,
                    evaluator.input_len,
                    lane_width,
                    Some(&self.chunk_input_mapping_scratch),
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                input_pack_elapsed += input_pack_start.elapsed();
                leaf_input_copy_component_count +=
                    (self.chunk_parameter_scratch_aosoa_f64.len() / 2) as u64;
                scratch_reallocation_count += u64::from(
                    self.chunk_parameter_scratch_aosoa_f64.capacity() != parameter_capacity,
                );

                let output_capacity = self.chunk_scratch_aosoa_f64.capacity();
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                scratch_reallocation_count +=
                    u64::from(self.chunk_scratch_aosoa_f64.capacity() != output_capacity);
                let eval_start = Instant::now();
                let evaluated = evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )?;
                evaluator_elapsed += eval_start.elapsed();
                backend_call_count += 1;
                if evaluated {
                    let assign_start = Instant::now();
                    scatter_f64_aosoa_outputs_to_state(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        state_parameter_count,
                        state,
                        &chunk_outputs[chunk_index],
                        &chunk_output_spans[chunk_index],
                    )?;
                    assign_elapsed += assign_start.elapsed();
                    continue;
                }
            }
            let parameter_capacity = self.chunk_parameter_scratch_f64.capacity();
            let input_pack_start = Instant::now();
            let evaluator_params = selected_chunk_f64_parameters(
                state,
                batch_size,
                state_parameter_count,
                self.input_len,
                parent_input_components,
                parent_input_spans,
                self.input_mappings[chunk_index].as_deref(),
                &self.input_mapping_spans[chunk_index],
                &mut self.chunk_parameter_scratch_f64,
            )?;
            input_pack_elapsed += input_pack_start.elapsed();
            if parent_input_components.is_some() || self.input_mappings[chunk_index].is_some() {
                leaf_input_copy_component_count += evaluator_params.len() as u64;
            }

            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            let scratch_len = batch_size
                .checked_mul(evaluator.output_len)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("stage chunk output length overflows usize")
                })?;
            let output_capacity = self.chunk_scratch_f64.capacity();
            self.chunk_scratch_f64.resize(scratch_len, c64(0.0, 0.0));
            let eval_start = Instant::now();
            evaluator.evaluate_f64_batch_unpadded(
                batch_size,
                evaluator_params,
                &mut self.chunk_scratch_f64,
            )?;
            evaluator_elapsed += eval_start.elapsed();
            backend_call_count += 1;
            scratch_reallocation_count +=
                u64::from(self.chunk_parameter_scratch_f64.capacity() != parameter_capacity)
                    + u64::from(self.chunk_scratch_f64.capacity() != output_capacity);

            let assign_start = Instant::now();
            let outputs = &chunk_outputs[chunk_index];
            let spans = &chunk_output_spans[chunk_index];
            for row in 0..batch_size {
                let row_state = row * state_parameter_count;
                let row_eval = row * evaluator.output_len;
                if spans.is_empty() {
                    for (column, state_offset) in outputs {
                        state[row_state + *state_offset] =
                            self.chunk_scratch_f64[row_eval + *column];
                    }
                } else {
                    for (column_start, state_offset_start, len) in spans {
                        let source_start = row_eval + *column_start;
                        let target_start = row_state + *state_offset_start;
                        state[target_start..target_start + *len].copy_from_slice(
                            &self.chunk_scratch_f64[source_start..source_start + *len],
                        );
                    }
                }
            }
            assign_elapsed += assign_start.elapsed();
        }
        let leaf_input_pack_s = profile_duration_seconds(input_pack_elapsed);
        let evaluator_call_s = profile_duration_seconds(evaluator_elapsed);
        Ok((
            EvaluatorBatchProfile {
                leaf_input_pack_s,
                legacy_evaluator_call_s: evaluator_call_s,
                evaluator_call_s,
                leaf_input_copy_component_count,
                backend_call_count,
                scratch_reallocation_count,
                ..EvaluatorBatchProfile::default()
            },
            profile_duration_seconds(assign_elapsed),
        ))
    }

    pub(crate) fn evaluate_selected_chunks_f64_into_output(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        parent_input_components: Option<&[usize]>,
        parent_input_spans: &[(usize, usize, usize)],
        out: &mut Vec<Complex<f64>>,
        active_chunk_indices: &[usize],
    ) -> RusticolResult<()> {
        validate_active_chunk_indices(active_chunk_indices, self.evaluators.len())?;
        if batch_size == 0 || !params.len().is_multiple_of(batch_size) {
            return Err(RusticolError::invalid_argument(
                "amplitude evaluator parameter buffer has an inconsistent batch size",
            ));
        }
        let state_parameter_count = params.len() / batch_size;
        match parent_input_components {
            Some(components) => {
                if components.len() != self.input_len
                    || components
                        .iter()
                        .any(|component| *component >= state_parameter_count)
                {
                    return Err(RusticolError::invalid_argument(
                        "amplitude input-component map is inconsistent with runtime state",
                    ));
                }
            }
            None if self.input_len != state_parameter_count => {
                return Err(RusticolError::invalid_argument(format!(
                    "evaluator input length {} does not match amplitude state parameter count {state_parameter_count}",
                    self.input_len
                )));
            }
            None => {}
        }
        let output_len = batch_size
            .checked_mul(self.output_len)
            .ok_or_else(|| RusticolError::invalid_argument("amplitude output length overflows"))?;
        out.resize(output_len, c64(0.0, 0.0));
        out.fill(c64(0.0, 0.0));

        let mut output_start = 0usize;
        for chunk_index in 0..self.evaluators.len() {
            let evaluator_output_len = self.evaluators[chunk_index].output_len;
            if active_chunk_indices.binary_search(&chunk_index).is_ok() {
                let evaluator = &mut self.evaluators[chunk_index];
                let mut evaluated_aosoa = false;
                if let Some(lane_width) = evaluator
                    .amplitude_aosoa_lane_width()
                    .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
                {
                    let input_mapping = self.input_mappings[chunk_index].as_deref();
                    self.chunk_input_mapping_scratch.clear();
                    self.chunk_input_mapping_scratch
                        .reserve(evaluator.input_len);
                    for local_component in 0..evaluator.input_len {
                        let parent_component = input_mapping
                            .map_or(local_component, |indices| indices[local_component]);
                        self.chunk_input_mapping_scratch.push(
                            parent_input_components.map_or(parent_component, |components| {
                                components[parent_component]
                            }),
                        );
                    }
                    let block_count = pack_f64_parameters_aosoa(
                        params,
                        batch_size,
                        state_parameter_count,
                        evaluator.input_len,
                        lane_width,
                        Some(&self.chunk_input_mapping_scratch),
                        &mut self.chunk_parameter_scratch_aosoa_f64,
                    )?;
                    self.chunk_scratch_aosoa_f64.resize(
                        aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                        0.0,
                    );
                    if evaluator.evaluate_f64_aosoa_blocks(
                        block_count,
                        &self.chunk_parameter_scratch_aosoa_f64,
                        &mut self.chunk_scratch_aosoa_f64,
                    )? {
                        scatter_f64_aosoa_outputs_to_rows(
                            &self.chunk_scratch_aosoa_f64,
                            batch_size,
                            evaluator.output_len,
                            lane_width,
                            self.output_len,
                            output_start,
                            out,
                        )?;
                        evaluated_aosoa = true;
                    }
                }
                if !evaluated_aosoa {
                    let evaluator_params = selected_chunk_f64_parameters(
                        params,
                        batch_size,
                        state_parameter_count,
                        self.input_len,
                        parent_input_components,
                        parent_input_spans,
                        self.input_mappings[chunk_index].as_deref(),
                        &self.input_mapping_spans[chunk_index],
                        &mut self.chunk_parameter_scratch_f64,
                    )?;
                    validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
                    self.chunk_scratch_f64.resize(
                        batch_size
                            .checked_mul(evaluator_output_len)
                            .ok_or_else(|| {
                                RusticolError::invalid_argument(
                                    "amplitude chunk output length overflows usize",
                                )
                            })?,
                        c64(0.0, 0.0),
                    );
                    evaluator.evaluate_f64_batch_unpadded(
                        batch_size,
                        evaluator_params,
                        &mut self.chunk_scratch_f64,
                    )?;
                    for row in 0..batch_size {
                        let source_start = row * evaluator_output_len;
                        let target_start = row * self.output_len + output_start;
                        out[target_start..target_start + evaluator_output_len].copy_from_slice(
                            &self.chunk_scratch_f64
                                [source_start..source_start + evaluator_output_len],
                        );
                    }
                }
            }
            output_start = output_start
                .checked_add(evaluator_output_len)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("amplitude chunk output range overflows")
                })?;
        }
        if output_start != self.output_len {
            return Err(RusticolError::integrity(
                "amplitude evaluator chunk outputs do not cover the declared output length",
            ));
        }
        Ok(())
    }

    pub(crate) fn evaluate_selected_chunks_f64_into_output_profile(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        parent_input_components: Option<&[usize]>,
        parent_input_spans: &[(usize, usize, usize)],
        out: &mut Vec<Complex<f64>>,
        active_chunk_indices: &[usize],
    ) -> RusticolResult<EvaluatorBatchProfile> {
        validate_active_chunk_indices(active_chunk_indices, self.evaluators.len())?;
        if batch_size == 0 || !params.len().is_multiple_of(batch_size) {
            return Err(RusticolError::invalid_argument(
                "amplitude evaluator parameter buffer has an inconsistent batch size",
            ));
        }
        let state_parameter_count = params.len() / batch_size;
        match parent_input_components {
            Some(components) => {
                if components.len() != self.input_len
                    || components
                        .iter()
                        .any(|component| *component >= state_parameter_count)
                {
                    return Err(RusticolError::invalid_argument(
                        "amplitude input-component map is inconsistent with runtime state",
                    ));
                }
            }
            None if self.input_len != state_parameter_count => {
                return Err(RusticolError::invalid_argument(format!(
                    "evaluator input length {} does not match amplitude state parameter count {state_parameter_count}",
                    self.input_len
                )));
            }
            None => {}
        }
        let output_len = batch_size
            .checked_mul(self.output_len)
            .ok_or_else(|| RusticolError::invalid_argument("amplitude output length overflows"))?;
        let mut output_gather_elapsed = Duration::ZERO;
        let output_prepare_start = Instant::now();
        let output_capacity = out.capacity();
        out.resize(output_len, c64(0.0, 0.0));
        out.fill(c64(0.0, 0.0));
        output_gather_elapsed += output_prepare_start.elapsed();
        let mut output_gather_component_count = output_len as u64;
        let mut scratch_reallocation_count = u64::from(out.capacity() != output_capacity);

        let mut chunk_output_starts = Vec::with_capacity(self.evaluators.len());
        let mut output_start = 0usize;
        for evaluator in &self.evaluators {
            chunk_output_starts.push(output_start);
            output_start = output_start
                .checked_add(evaluator.output_len)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("amplitude chunk output range overflows")
                })?;
        }
        if output_start != self.output_len {
            return Err(RusticolError::integrity(
                "amplitude evaluator chunk outputs do not cover the declared output length",
            ));
        }

        let mut input_pack_elapsed = Duration::ZERO;
        let mut evaluator_elapsed = Duration::ZERO;
        let mut leaf_input_copy_component_count = 0u64;
        let mut backend_call_count = 0u64;
        for &chunk_index in active_chunk_indices {
            let evaluator = &mut self.evaluators[chunk_index];
            if let Some(lane_width) = evaluator
                .amplitude_aosoa_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let parameter_capacity = self.chunk_parameter_scratch_aosoa_f64.capacity();
                let mapping_capacity = self.chunk_input_mapping_scratch.capacity();
                let input_pack_start = Instant::now();
                let input_mapping = self.input_mappings[chunk_index].as_deref();
                self.chunk_input_mapping_scratch.clear();
                self.chunk_input_mapping_scratch
                    .reserve(evaluator.input_len);
                for local_component in 0..evaluator.input_len {
                    let parent_component =
                        input_mapping.map_or(local_component, |indices| indices[local_component]);
                    self.chunk_input_mapping_scratch.push(
                        parent_input_components
                            .map_or(parent_component, |components| components[parent_component]),
                    );
                }
                let block_count = pack_f64_parameters_aosoa(
                    params,
                    batch_size,
                    state_parameter_count,
                    evaluator.input_len,
                    lane_width,
                    Some(&self.chunk_input_mapping_scratch),
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                input_pack_elapsed += input_pack_start.elapsed();
                leaf_input_copy_component_count +=
                    (self.chunk_parameter_scratch_aosoa_f64.len() / 2) as u64;
                scratch_reallocation_count +=
                    u64::from(
                        self.chunk_parameter_scratch_aosoa_f64.capacity() != parameter_capacity,
                    ) + u64::from(self.chunk_input_mapping_scratch.capacity() != mapping_capacity);

                let chunk_capacity = self.chunk_scratch_aosoa_f64.capacity();
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                scratch_reallocation_count +=
                    u64::from(self.chunk_scratch_aosoa_f64.capacity() != chunk_capacity);
                let eval_start = Instant::now();
                let evaluated = evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )?;
                evaluator_elapsed += eval_start.elapsed();
                backend_call_count += 1;
                if evaluated {
                    let output_gather_start = Instant::now();
                    scatter_f64_aosoa_outputs_to_rows(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        self.output_len,
                        chunk_output_starts[chunk_index],
                        out,
                    )?;
                    output_gather_elapsed += output_gather_start.elapsed();
                    output_gather_component_count += (batch_size * evaluator.output_len) as u64;
                    continue;
                }
            }
            let parameter_capacity = self.chunk_parameter_scratch_f64.capacity();
            let input_pack_start = Instant::now();
            let evaluator_params = selected_chunk_f64_parameters(
                params,
                batch_size,
                state_parameter_count,
                self.input_len,
                parent_input_components,
                parent_input_spans,
                self.input_mappings[chunk_index].as_deref(),
                &self.input_mapping_spans[chunk_index],
                &mut self.chunk_parameter_scratch_f64,
            )?;
            input_pack_elapsed += input_pack_start.elapsed();
            if parent_input_components.is_some() || self.input_mappings[chunk_index].is_some() {
                leaf_input_copy_component_count += evaluator_params.len() as u64;
            }

            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            let chunk_capacity = self.chunk_scratch_f64.capacity();
            self.chunk_scratch_f64.resize(
                batch_size
                    .checked_mul(evaluator.output_len)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "amplitude chunk output length overflows usize",
                        )
                    })?,
                c64(0.0, 0.0),
            );
            let eval_start = Instant::now();
            evaluator.evaluate_f64_batch_unpadded(
                batch_size,
                evaluator_params,
                &mut self.chunk_scratch_f64,
            )?;
            evaluator_elapsed += eval_start.elapsed();
            backend_call_count += 1;
            scratch_reallocation_count +=
                u64::from(self.chunk_parameter_scratch_f64.capacity() != parameter_capacity)
                    + u64::from(self.chunk_scratch_f64.capacity() != chunk_capacity);
            let output_gather_start = Instant::now();
            let chunk_output_start = chunk_output_starts[chunk_index];
            for row in 0..batch_size {
                let source_start = row * evaluator.output_len;
                let target_start = row * self.output_len + chunk_output_start;
                out[target_start..target_start + evaluator.output_len].copy_from_slice(
                    &self.chunk_scratch_f64[source_start..source_start + evaluator.output_len],
                );
            }
            output_gather_elapsed += output_gather_start.elapsed();
            output_gather_component_count += (batch_size * evaluator.output_len) as u64;
        }
        let leaf_input_pack_s = profile_duration_seconds(input_pack_elapsed);
        let evaluator_call_s = profile_duration_seconds(evaluator_elapsed);
        let output_gather_s = profile_duration_seconds(output_gather_elapsed);
        Ok(EvaluatorBatchProfile {
            leaf_input_pack_s,
            legacy_evaluator_call_s: evaluator_call_s + output_gather_s,
            evaluator_call_s,
            output_gather_s,
            leaf_input_copy_component_count,
            output_gather_component_count,
            backend_call_count,
            scratch_reallocation_count,
        })
    }

    pub(crate) fn evaluate_batch_into(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut Vec<Complex<f64>>,
    ) -> RusticolResult<()> {
        self.evaluate_batch_into_unprofiled(batch_size, params, out, false)
    }

    pub(crate) fn evaluate_batch_into_padded_simd_tail(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut Vec<Complex<f64>>,
    ) -> RusticolResult<()> {
        self.evaluate_batch_into_unprofiled(batch_size, params, out, true)
    }

    fn evaluate_batch_into_unprofiled(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut Vec<Complex<f64>>,
        pad_incomplete_simd_tail: bool,
    ) -> RusticolResult<()> {
        let expected_output_len = batch_size * self.output_len;
        if out.len() != expected_output_len {
            out.resize(expected_output_len, c64(0.0, 0.0));
        }
        if params.len() != batch_size * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {}",
                params.len(),
                batch_size * self.input_len
            )));
        }
        if self.evaluators.len() == 1 {
            let evaluator = &mut self.evaluators[0];
            if let Some(lane_width) = evaluator
                .amplitude_aosoa_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let block_count = pack_f64_parameters_aosoa(
                    params,
                    batch_size,
                    self.input_len,
                    evaluator.input_len,
                    lane_width,
                    self.input_mappings[0].as_deref(),
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                if evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )? {
                    scatter_f64_aosoa_outputs_to_rows(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        self.output_len,
                        0,
                        out,
                    )?;
                    return Ok(());
                }
            }
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                self.input_mappings[0].as_deref(),
                &self.input_mapping_spans[0],
                &mut self.chunk_parameter_scratch_f64,
            );
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            if pad_incomplete_simd_tail {
                evaluator.evaluate_f64_batch(batch_size, evaluator_params, out)?;
            } else {
                evaluator.evaluate_f64_batch_unpadded(batch_size, evaluator_params, out)?;
            }
            return Ok(());
        }
        let mut output_offset = 0;
        for ((evaluator, input_mapping), input_spans) in self
            .evaluators
            .iter_mut()
            .zip(&self.input_mappings)
            .zip(&self.input_mapping_spans)
        {
            if let Some(lane_width) = evaluator
                .amplitude_aosoa_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let block_count = pack_f64_parameters_aosoa(
                    params,
                    batch_size,
                    self.input_len,
                    evaluator.input_len,
                    lane_width,
                    input_mapping.as_deref(),
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                if evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )? {
                    scatter_f64_aosoa_outputs_to_rows(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        self.output_len,
                        output_offset,
                        out,
                    )?;
                    output_offset += evaluator.output_len;
                    continue;
                }
            }
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                input_mapping.as_deref(),
                input_spans,
                &mut self.chunk_parameter_scratch_f64,
            );
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            self.chunk_scratch_f64
                .resize(batch_size * evaluator.output_len, c64(0.0, 0.0));
            if pad_incomplete_simd_tail {
                evaluator.evaluate_f64_batch(
                    batch_size,
                    evaluator_params,
                    &mut self.chunk_scratch_f64,
                )?;
            } else {
                evaluator.evaluate_f64_batch_unpadded(
                    batch_size,
                    evaluator_params,
                    &mut self.chunk_scratch_f64,
                )?;
            }
            for row in 0..batch_size {
                let src = row * evaluator.output_len;
                let dst = row * self.output_len + output_offset;
                out[dst..dst + evaluator.output_len]
                    .copy_from_slice(&self.chunk_scratch_f64[src..src + evaluator.output_len]);
            }
            output_offset += evaluator.output_len;
        }
        Ok(())
    }

    pub(crate) fn evaluate_batch_into_profile(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut Vec<Complex<f64>>,
        pad_incomplete_simd_tail: bool,
    ) -> RusticolResult<EvaluatorBatchProfile> {
        let mut leaf_input_pack_elapsed = Duration::ZERO;
        let mut evaluator_elapsed = Duration::ZERO;
        let mut output_gather_elapsed = Duration::ZERO;
        let mut leaf_input_copy_component_count = 0u64;
        let mut output_gather_component_count = 0u64;
        let mut backend_call_count = 0u64;
        let mut scratch_reallocation_count = 0u64;
        let expected_output_len = batch_size * self.output_len;
        let output_capacity = out.capacity();
        if out.len() != expected_output_len {
            out.resize(expected_output_len, c64(0.0, 0.0));
        }
        scratch_reallocation_count += u64::from(out.capacity() != output_capacity);
        if params.len() != batch_size * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {}",
                params.len(),
                batch_size * self.input_len
            )));
        }
        if self.evaluators.len() == 1 {
            let evaluator = &mut self.evaluators[0];
            if let Some(lane_width) = evaluator
                .amplitude_aosoa_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let leaf_capacity = self.chunk_parameter_scratch_aosoa_f64.capacity();
                let leaf_start = Instant::now();
                let block_count = pack_f64_parameters_aosoa(
                    params,
                    batch_size,
                    self.input_len,
                    evaluator.input_len,
                    lane_width,
                    self.input_mappings[0].as_deref(),
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                leaf_input_pack_elapsed += leaf_start.elapsed();
                leaf_input_copy_component_count +=
                    (self.chunk_parameter_scratch_aosoa_f64.len() / 2) as u64;
                scratch_reallocation_count +=
                    u64::from(self.chunk_parameter_scratch_aosoa_f64.capacity() != leaf_capacity);
                let chunk_capacity = self.chunk_scratch_aosoa_f64.capacity();
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                scratch_reallocation_count +=
                    u64::from(self.chunk_scratch_aosoa_f64.capacity() != chunk_capacity);
                let backend_start = Instant::now();
                let evaluated = evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )?;
                evaluator_elapsed += backend_start.elapsed();
                backend_call_count += 1;
                if evaluated {
                    let gather_start = Instant::now();
                    scatter_f64_aosoa_outputs_to_rows(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        self.output_len,
                        0,
                        out,
                    )?;
                    output_gather_elapsed += gather_start.elapsed();
                    output_gather_component_count += (batch_size * evaluator.output_len) as u64;
                    let leaf_input_pack_s = profile_duration_seconds(leaf_input_pack_elapsed);
                    let evaluator_call_s = profile_duration_seconds(evaluator_elapsed);
                    let output_gather_s = profile_duration_seconds(output_gather_elapsed);
                    return Ok(EvaluatorBatchProfile {
                        leaf_input_pack_s,
                        legacy_evaluator_call_s: leaf_input_pack_s
                            + evaluator_call_s
                            + output_gather_s,
                        evaluator_call_s,
                        output_gather_s,
                        leaf_input_copy_component_count,
                        output_gather_component_count,
                        backend_call_count,
                        scratch_reallocation_count,
                    });
                }
            }
            let leaf_capacity = self.chunk_parameter_scratch_f64.capacity();
            let leaf_start = Instant::now();
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                self.input_mappings[0].as_deref(),
                &self.input_mapping_spans[0],
                &mut self.chunk_parameter_scratch_f64,
            );
            leaf_input_pack_elapsed += leaf_start.elapsed();
            if self.input_mappings[0].is_some() {
                leaf_input_copy_component_count += evaluator_params.len() as u64;
            }
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            let backend_start = Instant::now();
            if pad_incomplete_simd_tail {
                evaluator.evaluate_f64_batch(batch_size, evaluator_params, out)?;
            } else {
                evaluator.evaluate_f64_batch_unpadded(batch_size, evaluator_params, out)?;
            }
            evaluator_elapsed += backend_start.elapsed();
            backend_call_count += 1;
            scratch_reallocation_count +=
                u64::from(self.chunk_parameter_scratch_f64.capacity() != leaf_capacity);
            let leaf_input_pack_s = profile_duration_seconds(leaf_input_pack_elapsed);
            let evaluator_call_s = profile_duration_seconds(evaluator_elapsed);
            return Ok(EvaluatorBatchProfile {
                leaf_input_pack_s,
                legacy_evaluator_call_s: leaf_input_pack_s + evaluator_call_s,
                evaluator_call_s,
                leaf_input_copy_component_count,
                backend_call_count,
                scratch_reallocation_count,
                ..EvaluatorBatchProfile::default()
            });
        }
        let mut output_offset = 0;
        for ((evaluator, input_mapping), input_spans) in self
            .evaluators
            .iter_mut()
            .zip(&self.input_mappings)
            .zip(&self.input_mapping_spans)
        {
            if let Some(lane_width) = evaluator
                .amplitude_aosoa_lane_width()
                .filter(|lane_width| *lane_width > 1 && batch_size >= *lane_width)
            {
                let leaf_capacity = self.chunk_parameter_scratch_aosoa_f64.capacity();
                let leaf_start = Instant::now();
                let block_count = pack_f64_parameters_aosoa(
                    params,
                    batch_size,
                    self.input_len,
                    evaluator.input_len,
                    lane_width,
                    input_mapping.as_deref(),
                    &mut self.chunk_parameter_scratch_aosoa_f64,
                )?;
                leaf_input_pack_elapsed += leaf_start.elapsed();
                leaf_input_copy_component_count +=
                    (self.chunk_parameter_scratch_aosoa_f64.len() / 2) as u64;
                scratch_reallocation_count +=
                    u64::from(self.chunk_parameter_scratch_aosoa_f64.capacity() != leaf_capacity);
                let chunk_capacity = self.chunk_scratch_aosoa_f64.capacity();
                self.chunk_scratch_aosoa_f64.resize(
                    aosoa_scalar_len(block_count, evaluator.output_len, lane_width, "output")?,
                    0.0,
                );
                scratch_reallocation_count +=
                    u64::from(self.chunk_scratch_aosoa_f64.capacity() != chunk_capacity);
                let backend_start = Instant::now();
                let evaluated = evaluator.evaluate_f64_aosoa_blocks(
                    block_count,
                    &self.chunk_parameter_scratch_aosoa_f64,
                    &mut self.chunk_scratch_aosoa_f64,
                )?;
                evaluator_elapsed += backend_start.elapsed();
                backend_call_count += 1;
                if evaluated {
                    let gather_start = Instant::now();
                    scatter_f64_aosoa_outputs_to_rows(
                        &self.chunk_scratch_aosoa_f64,
                        batch_size,
                        evaluator.output_len,
                        lane_width,
                        self.output_len,
                        output_offset,
                        out,
                    )?;
                    output_gather_elapsed += gather_start.elapsed();
                    output_gather_component_count += (batch_size * evaluator.output_len) as u64;
                    output_offset += evaluator.output_len;
                    continue;
                }
            }
            let leaf_capacity = self.chunk_parameter_scratch_f64.capacity();
            let leaf_start = Instant::now();
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                input_mapping.as_deref(),
                input_spans,
                &mut self.chunk_parameter_scratch_f64,
            );
            leaf_input_pack_elapsed += leaf_start.elapsed();
            if input_mapping.is_some() {
                leaf_input_copy_component_count += evaluator_params.len() as u64;
            }
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            let chunk_capacity = self.chunk_scratch_f64.capacity();
            self.chunk_scratch_f64
                .resize(batch_size * evaluator.output_len, c64(0.0, 0.0));
            scratch_reallocation_count +=
                u64::from(self.chunk_scratch_f64.capacity() != chunk_capacity);
            let backend_start = Instant::now();
            if pad_incomplete_simd_tail {
                evaluator.evaluate_f64_batch(
                    batch_size,
                    evaluator_params,
                    &mut self.chunk_scratch_f64,
                )?;
            } else {
                evaluator.evaluate_f64_batch_unpadded(
                    batch_size,
                    evaluator_params,
                    &mut self.chunk_scratch_f64,
                )?;
            }
            evaluator_elapsed += backend_start.elapsed();
            backend_call_count += 1;
            scratch_reallocation_count +=
                u64::from(self.chunk_parameter_scratch_f64.capacity() != leaf_capacity);
            let gather_start = Instant::now();
            for row in 0..batch_size {
                let src = row * evaluator.output_len;
                let dst = row * self.output_len + output_offset;
                out[dst..dst + evaluator.output_len]
                    .copy_from_slice(&self.chunk_scratch_f64[src..src + evaluator.output_len]);
            }
            output_gather_elapsed += gather_start.elapsed();
            output_gather_component_count += (batch_size * evaluator.output_len) as u64;
            output_offset += evaluator.output_len;
        }
        let leaf_input_pack_s = profile_duration_seconds(leaf_input_pack_elapsed);
        let evaluator_call_s = profile_duration_seconds(evaluator_elapsed);
        let output_gather_s = profile_duration_seconds(output_gather_elapsed);
        Ok(EvaluatorBatchProfile {
            leaf_input_pack_s,
            legacy_evaluator_call_s: leaf_input_pack_s + evaluator_call_s + output_gather_s,
            evaluator_call_s,
            output_gather_s,
            leaf_input_copy_component_count,
            output_gather_component_count,
            backend_call_count,
            scratch_reallocation_count,
        })
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_selected_chunks_generic<T>(
        &mut self,
        batch_size: usize,
        params: &[Complex<T>],
        binary_precision: Option<u32>,
        active_chunk_indices: Option<&[usize]>,
    ) -> RusticolResult<Vec<Complex<T>>>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if let Some(indices) = active_chunk_indices {
            validate_active_chunk_indices(indices, self.evaluators.len())?;
        }
        let mut out = vec![complex_zero::<T>(); batch_size * self.output_len];
        if params.len() != batch_size * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {}",
                params.len(),
                batch_size * self.input_len
            )));
        }
        let mut output_offset = 0;
        let mut active_position = 0;
        for (chunk_index, (evaluator, input_mapping)) in self
            .evaluators
            .iter_mut()
            .zip(&self.input_mappings)
            .enumerate()
        {
            if let Some(indices) = active_chunk_indices {
                if indices.get(active_position) != Some(&chunk_index) {
                    output_offset += evaluator.output_len;
                    continue;
                }
                active_position += 1;
            }
            let mapped_params = input_mapping.as_ref().map(|indices| {
                let mut mapped = Vec::with_capacity(batch_size * indices.len());
                for row in 0..batch_size {
                    let row_start = row * self.input_len;
                    mapped.extend(
                        indices
                            .iter()
                            .map(|index| params[row_start + *index].clone()),
                    );
                }
                mapped
            });
            let evaluator_params = mapped_params.as_deref().unwrap_or(params);
            let mut chunk_out = vec![complex_zero::<T>(); batch_size * evaluator.output_len];
            for row in 0..batch_size {
                let in_start = row * evaluator.input_len;
                let out_start = row * evaluator.output_len;
                T::evaluate_loaded(
                    evaluator,
                    &evaluator_params[in_start..in_start + evaluator.input_len],
                    &mut chunk_out[out_start..out_start + evaluator.output_len],
                    binary_precision,
                )?;
            }
            for row in 0..batch_size {
                let src = row * evaluator.output_len;
                let dst = row * self.output_len + output_offset;
                out[dst..dst + evaluator.output_len]
                    .clone_from_slice(&chunk_out[src..src + evaluator.output_len]);
            }
            output_offset += evaluator.output_len;
        }
        Ok(out)
    }
}

fn aosoa_scalar_len(
    block_count: usize,
    component_count: usize,
    lane_width: usize,
    label: &str,
) -> RusticolResult<usize> {
    block_count
        .checked_mul(component_count)
        .and_then(|len| len.checked_mul(2))
        .and_then(|len| len.checked_mul(lane_width))
        .ok_or_else(|| {
            RusticolError::invalid_argument(format!("{label} AoSoA buffer length overflows usize"))
        })
}

fn pack_f64_parameters_aosoa(
    source: &[Complex<f64>],
    batch_size: usize,
    source_parameter_count: usize,
    leaf_input_len: usize,
    lane_width: usize,
    input_components: Option<&[usize]>,
    scratch: &mut Vec<f64>,
) -> RusticolResult<usize> {
    if batch_size == 0 || lane_width <= 1 {
        return Err(RusticolError::invalid_argument(
            "AoSoA parameter packing requires a non-empty SIMD batch",
        ));
    }
    let expected_source_len = batch_size
        .checked_mul(source_parameter_count)
        .ok_or_else(|| {
            RusticolError::invalid_argument("source parameter length overflows usize")
        })?;
    if source.len() != expected_source_len {
        return Err(RusticolError::invalid_argument(format!(
            "source parameter buffer has length {}, expected {expected_source_len}",
            source.len()
        )));
    }
    match input_components {
        Some(components)
            if components.len() != leaf_input_len
                || components
                    .iter()
                    .any(|component| *component >= source_parameter_count) =>
        {
            return Err(RusticolError::integrity(
                "AoSoA leaf input mapping is inconsistent with source parameters",
            ));
        }
        None if leaf_input_len != source_parameter_count => {
            return Err(RusticolError::integrity(
                "AoSoA identity leaf input length does not match source parameters",
            ));
        }
        Some(_) | None => {}
    }
    let block_count = batch_size.div_ceil(lane_width);
    scratch.resize(
        aosoa_scalar_len(block_count, leaf_input_len, lane_width, "parameter")?,
        0.0,
    );
    match lane_width {
        // SAFETY: the checks above establish every source component and the
        // complete destination length required by the fixed-lane packers.
        2 => unsafe {
            pack_f64_parameters_aosoa_lanes::<2>(
                source,
                batch_size,
                source_parameter_count,
                leaf_input_len,
                input_components,
                scratch,
            )
        },
        // SAFETY: see the lane-2 branch.
        4 => unsafe {
            pack_f64_parameters_aosoa_lanes::<4>(
                source,
                batch_size,
                source_parameter_count,
                leaf_input_len,
                input_components,
                scratch,
            )
        },
        // SAFETY: see the lane-2 branch.
        8 => unsafe {
            pack_f64_parameters_aosoa_lanes::<8>(
                source,
                batch_size,
                source_parameter_count,
                leaf_input_len,
                input_components,
                scratch,
            )
        },
        _ => pack_f64_parameters_aosoa_dynamic(
            source,
            batch_size,
            source_parameter_count,
            leaf_input_len,
            lane_width,
            input_components,
            scratch,
        ),
    }
    Ok(block_count)
}

/// # Safety
///
/// `source` must contain `batch_size * source_parameter_count` elements,
/// `input_components` must contain `leaf_input_len` in-range indices when
/// present, and `scratch` must contain the complete padded AoSoA destination.
unsafe fn pack_f64_parameters_aosoa_lanes<const LANES: usize>(
    source: &[Complex<f64>],
    batch_size: usize,
    source_parameter_count: usize,
    leaf_input_len: usize,
    input_components: Option<&[usize]>,
    scratch: &mut [f64],
) {
    let block_count = batch_size.div_ceil(LANES);
    let source_ptr = source.as_ptr();
    let target_ptr = scratch.as_mut_ptr();
    for block in 0..block_count {
        let first_row = block * LANES;
        for local_component in 0..leaf_input_len {
            let component =
                input_components.map_or(local_component, |components| components[local_component]);
            let target = (block * leaf_input_len + local_component) * 2 * LANES;
            for lane in 0..LANES {
                let row = (first_row + lane).min(batch_size - 1);
                // SAFETY: the caller validated the complete source and target
                // lengths plus every mapped component before entering this
                // fixed-lane hot loop.
                unsafe {
                    let value = *source_ptr.add(row * source_parameter_count + component);
                    *target_ptr.add(target + lane) = value.re;
                    *target_ptr.add(target + LANES + lane) = value.im;
                }
            }
        }
    }
}

fn pack_f64_parameters_aosoa_dynamic(
    source: &[Complex<f64>],
    batch_size: usize,
    source_parameter_count: usize,
    leaf_input_len: usize,
    lane_width: usize,
    input_components: Option<&[usize]>,
    scratch: &mut [f64],
) {
    let block_count = batch_size.div_ceil(lane_width);
    let scalar_stride = lane_width * 2;
    for block in 0..block_count {
        let first_row = block * lane_width;
        for local_component in 0..leaf_input_len {
            let component =
                input_components.map_or(local_component, |components| components[local_component]);
            let target = (block * leaf_input_len + local_component) * scalar_stride;
            for lane in 0..lane_width {
                let row = (first_row + lane).min(batch_size - 1);
                let value = source[row * source_parameter_count + component];
                scratch[target + lane] = value.re;
                scratch[target + lane_width + lane] = value.im;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_f64_aosoa_outputs_to_state(
    outputs_aosoa: &[f64],
    batch_size: usize,
    evaluator_output_len: usize,
    lane_width: usize,
    state_parameter_count: usize,
    state: &mut [Complex<f64>],
    outputs: &[(usize, usize)],
    spans: &[(usize, usize, usize)],
) -> RusticolResult<()> {
    let expected_state_len = batch_size
        .checked_mul(state_parameter_count)
        .ok_or_else(|| RusticolError::invalid_argument("stage state length overflows usize"))?;
    if state.len() != expected_state_len {
        return Err(RusticolError::invalid_argument(format!(
            "stage state has length {}, expected {expected_state_len}",
            state.len()
        )));
    }
    let block_count = batch_size.div_ceil(lane_width);
    let expected_output_len =
        aosoa_scalar_len(block_count, evaluator_output_len, lane_width, "output")?;
    if outputs_aosoa.len() != expected_output_len {
        return Err(RusticolError::integrity(format!(
            "AoSoA evaluator output has length {}, expected {expected_output_len}",
            outputs_aosoa.len()
        )));
    }
    let scalar_stride = lane_width * 2;
    for block in 0..block_count {
        let first_row = block * lane_width;
        let valid_lanes = (batch_size - first_row).min(lane_width);
        for lane in 0..valid_lanes {
            let row_state = (first_row + lane) * state_parameter_count;
            if spans.is_empty() {
                for (column, state_offset) in outputs {
                    let source = (block * evaluator_output_len + *column) * scalar_stride;
                    state[row_state + *state_offset] = Complex::new(
                        outputs_aosoa[source + lane],
                        outputs_aosoa[source + lane_width + lane],
                    );
                }
            } else {
                for (column_start, state_offset_start, len) in spans {
                    for offset in 0..*len {
                        let source =
                            (block * evaluator_output_len + column_start + offset) * scalar_stride;
                        state[row_state + state_offset_start + offset] = Complex::new(
                            outputs_aosoa[source + lane],
                            outputs_aosoa[source + lane_width + lane],
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scatter_f64_aosoa_outputs_to_rows(
    outputs_aosoa: &[f64],
    batch_size: usize,
    evaluator_output_len: usize,
    lane_width: usize,
    row_output_len: usize,
    output_offset: usize,
    out: &mut [Complex<f64>],
) -> RusticolResult<()> {
    let expected_output_rows = batch_size.checked_mul(row_output_len).ok_or_else(|| {
        RusticolError::invalid_argument("amplitude output length overflows usize")
    })?;
    if out.len() != expected_output_rows {
        return Err(RusticolError::invalid_argument(format!(
            "amplitude output has length {}, expected {expected_output_rows}",
            out.len()
        )));
    }
    let output_end = output_offset
        .checked_add(evaluator_output_len)
        .ok_or_else(|| RusticolError::invalid_argument("amplitude output range overflows usize"))?;
    if output_end > row_output_len {
        return Err(RusticolError::integrity(
            "AoSoA evaluator output range exceeds the amplitude row",
        ));
    }
    let block_count = batch_size.div_ceil(lane_width);
    let expected_aosoa_len =
        aosoa_scalar_len(block_count, evaluator_output_len, lane_width, "output")?;
    if outputs_aosoa.len() != expected_aosoa_len {
        return Err(RusticolError::integrity(format!(
            "AoSoA evaluator output has length {}, expected {expected_aosoa_len}",
            outputs_aosoa.len()
        )));
    }
    let scalar_stride = lane_width * 2;
    for block in 0..block_count {
        let first_row = block * lane_width;
        let valid_lanes = (batch_size - first_row).min(lane_width);
        for lane in 0..valid_lanes {
            let row_output = (first_row + lane) * row_output_len + output_offset;
            for column in 0..evaluator_output_len {
                let source = (block * evaluator_output_len + column) * scalar_stride;
                out[row_output + column] = Complex::new(
                    outputs_aosoa[source + lane],
                    outputs_aosoa[source + lane_width + lane],
                );
            }
        }
    }
    Ok(())
}

fn mapped_f64_parameters<'a>(
    params: &'a [Complex<f64>],
    batch_size: usize,
    parent_input_len: usize,
    input_mapping: Option<&[usize]>,
    input_spans: &[(usize, usize, usize)],
    scratch: &'a mut Vec<Complex<f64>>,
) -> &'a [Complex<f64>] {
    let Some(indices) = input_mapping else {
        return params;
    };
    scratch.resize(batch_size * indices.len(), c64(0.0, 0.0));
    for row in 0..batch_size {
        let source_start = row * parent_input_len;
        let target_start = row * indices.len();
        if input_spans.is_empty() {
            for (local_index, parent_index) in indices.iter().enumerate() {
                scratch[target_start + local_index] = params[source_start + *parent_index];
            }
        } else {
            for (local_start, parent_start, len) in input_spans {
                let target = target_start + *local_start;
                let source = source_start + *parent_start;
                scratch[target..target + *len].copy_from_slice(&params[source..source + *len]);
            }
        }
    }
    scratch
}

#[allow(clippy::too_many_arguments, dead_code)]
fn selected_chunk_f64_parameters<'a>(
    state: &'a [Complex<f64>],
    batch_size: usize,
    state_parameter_count: usize,
    parent_input_len: usize,
    parent_input_components: Option<&[usize]>,
    parent_input_spans: &[(usize, usize, usize)],
    input_mapping: Option<&[usize]>,
    input_mapping_spans: &[(usize, usize, usize)],
    scratch: &'a mut Vec<Complex<f64>>,
) -> RusticolResult<&'a [Complex<f64>]> {
    let Some(parent_components) = parent_input_components else {
        return Ok(mapped_f64_parameters(
            state,
            batch_size,
            state_parameter_count,
            input_mapping,
            input_mapping_spans,
            scratch,
        ));
    };
    let mapped_input_len = input_mapping.map_or(parent_input_len, <[usize]>::len);
    let scratch_len = batch_size.checked_mul(mapped_input_len).ok_or_else(|| {
        RusticolError::invalid_argument("selected stage input length overflows usize")
    })?;
    scratch.resize(scratch_len, c64(0.0, 0.0));
    for row in 0..batch_size {
        let source_start = row * state_parameter_count;
        let target_start = row * mapped_input_len;
        if let Some(indices) = input_mapping {
            for (local_index, parent_index) in indices.iter().copied().enumerate() {
                scratch[target_start + local_index] =
                    state[source_start + parent_components[parent_index]];
            }
        } else if parent_input_spans.is_empty() {
            for (local_index, global_index) in parent_components.iter().copied().enumerate() {
                scratch[target_start + local_index] = state[source_start + global_index];
            }
        } else {
            for (local_start, global_start, len) in parent_input_spans {
                let target = target_start + *local_start;
                let source = source_start + *global_start;
                scratch[target..target + *len].copy_from_slice(&state[source..source + *len]);
            }
        }
    }
    Ok(scratch)
}

#[allow(dead_code)]
fn validate_active_chunk_indices(
    active_chunk_indices: &[usize],
    chunk_count: usize,
) -> RusticolResult<()> {
    let mut previous = None;
    for &chunk_index in active_chunk_indices {
        if chunk_index >= chunk_count {
            return Err(RusticolError::invalid_argument(format!(
                "active evaluator chunk index {chunk_index} is outside chunk count {chunk_count}",
            )));
        }
        if previous.is_some_and(|previous| previous >= chunk_index) {
            return Err(RusticolError::invalid_argument(
                "active evaluator chunk indices must be strictly increasing",
            ));
        }
        previous = Some(chunk_index);
    }
    Ok(())
}

fn validate_leaf_parameter_length(
    evaluator: &LoadedEvaluator,
    batch_size: usize,
    params: &[Complex<f64>],
) -> RusticolResult<()> {
    if params.len() != batch_size * evaluator.input_len {
        return Err(RusticolError::artifact(format!(
            "mapped evaluator parameter buffer has length {}, expected {}",
            params.len(),
            batch_size * evaluator.input_len
        )));
    }
    Ok(())
}

impl LoadedEvaluator {
    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn exact_evaluator(
        &mut self,
    ) -> RusticolResult<&ExpressionEvaluator<Complex<Rational>>> {
        if self.exact_eval.is_none() {
            let source = self.exact_eval_source.as_ref().ok_or_else(|| {
                RusticolError::invalid_argument(
                    "high-precision evaluation requires an evaluator-state artifact, but this process artifact has no evaluator_state_path for one or more chunks",
                )
            })?;
            self.exact_eval = Some(load_evaluator_state_source(source)?.1);
        }
        Ok(self
            .exact_eval
            .as_ref()
            .expect("exact evaluator initialized from its state path"))
    }

    fn simd_lane_width(&self) -> Option<usize> {
        match &self.eval {
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::ExactOnly => None,
            #[cfg(feature = "f64-symjit")]
            F64Evaluator::SymjitApplication(eval) => eval.simd_lane_width(),
            #[cfg(feature = "f64-compiled")]
            F64Evaluator::Compiled(_) => None,
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Jit(_) => None,
        }
    }

    fn amplitude_aosoa_lane_width(&self) -> Option<usize> {
        // Tiny amplitude leaves (notably LC's common 4 -> 1 kernels) do not
        // amortize direct packing and scattering. This static shape gate keeps
        // the portable path for substantive kernels without runtime tuning or
        // architecture-specific thresholds.
        const MINIMUM_COMPONENT_FOOTPRINT: usize = 32;
        if self.input_len.saturating_add(self.output_len) < MINIMUM_COMPONENT_FOOTPRINT {
            None
        } else {
            self.simd_lane_width()
        }
    }

    fn evaluate_f64_aosoa_blocks(
        &self,
        block_count: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> RusticolResult<bool> {
        match &self.eval {
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::ExactOnly => Ok(false),
            #[cfg(feature = "f64-symjit")]
            F64Evaluator::SymjitApplication(eval) => {
                eval.evaluate_aosoa_blocks(block_count, params, out)
            }
            #[cfg(feature = "f64-compiled")]
            F64Evaluator::Compiled(_) => Ok(false),
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Jit(_) => Ok(false),
        }
    }

    pub(crate) fn evaluate_f64_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        match &mut self.eval {
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::ExactOnly => Err(RusticolError::integrity(
                "compiled plane-arena exact evaluator cannot execute f64",
            )),
            #[cfg(feature = "f64-symjit")]
            F64Evaluator::SymjitApplication(eval) => eval.evaluate_batch(batch_size, params, out),
            #[cfg(feature = "f64-compiled")]
            F64Evaluator::Compiled(eval) => eval.evaluate_batch(batch_size, params, out),
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Jit(eval) => {
                if params.len() != batch_size * self.input_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "parameter buffer has length {}, expected {}",
                        params.len(),
                        batch_size * self.input_len
                    )));
                }
                if out.len() != batch_size * self.output_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "output buffer has length {}, expected {}",
                        out.len(),
                        batch_size * self.output_len
                    )));
                }
                eval.evaluate_batch(batch_size, params, out)
                    .map_err(RusticolError::evaluation)
            }
        }
    }

    fn evaluate_f64_batch_unpadded(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        match &mut self.eval {
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::ExactOnly => Err(RusticolError::integrity(
                "compiled plane-arena exact evaluator cannot execute f64",
            )),
            #[cfg(feature = "f64-symjit")]
            F64Evaluator::SymjitApplication(eval) => {
                eval.evaluate_batch_unpadded(batch_size, params, out)
            }
            #[cfg(feature = "f64-compiled")]
            F64Evaluator::Compiled(eval) => eval.evaluate_batch(batch_size, params, out),
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Jit(eval) => {
                if params.len() != batch_size * self.input_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "parameter buffer has length {}, expected {}",
                        params.len(),
                        batch_size * self.input_len
                    )));
                }
                if out.len() != batch_size * self.output_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "output buffer has length {}, expected {}",
                        out.len(),
                        batch_size * self.output_len
                    )));
                }
                eval.evaluate_batch(batch_size, params, out)
                    .map_err(RusticolError::evaluation)
            }
        }
    }
}

#[cfg(feature = "symbolica-runtime")]
fn flatten_exact_evaluators_from_store(
    manifest: &EvaluatorManifest,
    payloads: &EvaluatorPayloadStore,
    output: &mut Vec<LoadedEvaluator>,
) -> RusticolResult<()> {
    match manifest {
        EvaluatorManifest::SymjitApplication {
            input_len,
            output_len,
            evaluator_state_path,
            evaluator_state_runtime_capability,
            ..
        } => {
            let state_path = evaluator_state_path.as_ref().ok_or_else(|| {
                RusticolError::compatibility(
                    "compiled plane-arena exact execution requires evaluator-state payloads",
                )
            })?;
            if evaluator_state_runtime_capability.as_deref()
                != Some(SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY)
            {
                return Err(RusticolError::compatibility(format!(
                    "SymJIT evaluator state {state_path:?} does not declare capability {:?}",
                    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
                )));
            }
            output.push(LoadedEvaluator {
                eval: F64Evaluator::ExactOnly,
                exact_eval: None,
                exact_eval_source: Some(payloads.source(state_path)?),
                double_eval: None,
                arb_eval: None,
                input_len: *input_len,
                output_len: *output_len,
            });
            Ok(())
        }
        EvaluatorManifest::CompiledComplex {
            input_len,
            output_len,
            evaluator_state_path,
            native_direct_application: Some(_),
            ..
        } => {
            let state_path = evaluator_state_path.as_ref().ok_or_else(|| {
                RusticolError::compatibility(
                    "compiled plane-arena exact execution requires evaluator-state payloads",
                )
            })?;
            output.push(LoadedEvaluator {
                eval: F64Evaluator::ExactOnly,
                exact_eval: None,
                exact_eval_source: Some(payloads.source(state_path)?),
                double_eval: None,
                arb_eval: None,
                input_len: *input_len,
                output_len: *output_len,
            });
            Ok(())
        }
        EvaluatorManifest::Chunked { chunks, .. } => {
            for chunk in chunks {
                flatten_exact_evaluators_from_store(chunk, payloads, output)?;
            }
            Ok(())
        }
        _ => Err(RusticolError::compatibility(
            "compiled plane-arena exact execution requires direct leaves with evaluator state",
        )),
    }
}

pub(crate) fn flatten_evaluators_from_store(
    manifest: &EvaluatorManifest,
    payloads: &EvaluatorPayloadStore,
    output: &mut Vec<LoadedEvaluator>,
) -> RusticolResult<()> {
    match manifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability,
            application_path,
            application_abi,
            input_len,
            output_len,
            element_layout,
            batch_layout,
            compiler_type,
            translation_mode,
            optimization_level,
            word_bits,
            endianness,
            required_defuns,
            evaluator_state_path,
            evaluator_state_runtime_capability,
            ..
        } => {
            #[cfg(feature = "f64-symjit")]
            {
                let application_source = payloads.source(application_path)?;
                let application_bytes = application_source.read()?;
                let eval = SymjitApplicationEvaluator::load_bytes(
                    application_bytes.as_ref(),
                    PathBuf::from(application_source.display_name()),
                    SymjitApplicationMetadata {
                        runtime_capability,
                        application_abi,
                        input_len: *input_len,
                        output_len: *output_len,
                        element_layout,
                        batch_layout,
                        compiler_type,
                        translation_mode,
                        optimization_level: *optimization_level,
                        word_bits: *word_bits,
                        endianness,
                        required_defuns,
                    },
                )?;
                #[cfg(feature = "symbolica-runtime")]
                let exact_eval_source = match evaluator_state_path {
                    Some(state_path) => {
                        if evaluator_state_runtime_capability.as_deref()
                            != Some(SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY)
                        {
                            return Err(RusticolError::compatibility(format!(
                                "SymJIT evaluator state {:?} does not declare capability {:?}",
                                state_path, SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
                            )));
                        }
                        Some(payloads.source(state_path)?)
                    }
                    None => None,
                };
                #[cfg(not(feature = "symbolica-runtime"))]
                let _ = (evaluator_state_path, evaluator_state_runtime_capability);
                output.push(LoadedEvaluator {
                    eval: F64Evaluator::SymjitApplication(eval),
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval_source,
                    #[cfg(feature = "symbolica-runtime")]
                    double_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "f64-symjit"))]
            {
                let _ = (
                    application_path,
                    application_abi,
                    input_len,
                    output_len,
                    element_layout,
                    batch_layout,
                    compiler_type,
                    translation_mode,
                    optimization_level,
                    word_bits,
                    endianness,
                    required_defuns,
                    evaluator_state_path,
                    evaluator_state_runtime_capability,
                );
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::Jit {
            runtime_capability,
            input_len,
            output_len,
            evaluator_state_path,
        } => {
            #[cfg(feature = "symbolica-runtime")]
            {
                let _ = runtime_capability;
                let (jit_settings, exact_eval, jit_eval) =
                    load_evaluator_state_source(&payloads.source(evaluator_state_path)?)?;
                let eval = F64Evaluator::Jit(match jit_eval {
                    Some(eval) => eval,
                    None => exact_eval
                        .jit_compile::<Complex<f64>>(jit_settings)
                        .map_err(|err| {
                            RusticolError::evaluation(format!(
                                "could not compile scalar JIT evaluator from {}: {err}",
                                evaluator_state_path
                            ))
                        })?,
                });
                output.push(LoadedEvaluator {
                    eval,
                    exact_eval: Some(exact_eval),
                    exact_eval_source: None,
                    double_eval: None,
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "symbolica-runtime"))]
            {
                let _ = (input_len, output_len, evaluator_state_path);
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::CompiledComplex {
            runtime_capability,
            function_name,
            input_len,
            output_len,
            library_path,
            evaluator_state_path,
            number_type,
            native_direct_application: _,
        } => {
            #[cfg(feature = "f64-compiled")]
            {
                let _ = runtime_capability;
                if number_type != "complex" {
                    return Err(RusticolError::invalid_argument(format!(
                        "rusticol currently supports compiled complex evaluators, got {number_type}"
                    )));
                }
                let library = payloads.load_native_library(library_path)?;
                let eval = CompiledComplexF64Evaluator::load(
                    library,
                    function_name,
                    *input_len,
                    *output_len,
                )?;
                #[cfg(feature = "symbolica-runtime")]
                let exact_eval_source = evaluator_state_path
                    .as_deref()
                    .map(|state_path| payloads.source(state_path))
                    .transpose()?;
                #[cfg(not(feature = "symbolica-runtime"))]
                let _ = evaluator_state_path;
                output.push(LoadedEvaluator {
                    eval: F64Evaluator::Compiled(eval),
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval_source,
                    #[cfg(feature = "symbolica-runtime")]
                    double_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "f64-compiled"))]
            {
                let _ = (
                    function_name,
                    input_len,
                    output_len,
                    library_path,
                    evaluator_state_path,
                    number_type,
                );
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::Chunked { chunks, .. } => {
            for chunk in chunks {
                flatten_evaluators_from_store(chunk, payloads, output)?;
            }
            Ok(())
        }
    }
}

pub(crate) fn ensure_evaluator_capabilities_supported(
    manifest: &EvaluatorManifest,
) -> RusticolResult<()> {
    let declared = evaluator_runtime_capabilities(manifest)?;
    ensure_runtime_capabilities_supported(declared.iter().map(String::as_str))
}

pub(crate) fn evaluator_runtime_capabilities(
    manifest: &EvaluatorManifest,
) -> RusticolResult<BTreeSet<String>> {
    let mut capabilities = BTreeSet::new();
    collect_evaluator_capabilities(manifest, &mut capabilities)?;
    Ok(capabilities)
}

fn collect_evaluator_capabilities(
    manifest: &EvaluatorManifest,
    output: &mut BTreeSet<String>,
) -> RusticolResult<()> {
    match manifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[SYMJIT_APPLICATION_RUNTIME_CAPABILITY],
            output,
        ),
        EvaluatorManifest::Jit {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY],
            output,
        ),
        EvaluatorManifest::CompiledComplex {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[
                SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
                SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
            ],
            output,
        ),
        EvaluatorManifest::Chunked {
            required_runtime_capabilities,
            chunks,
            ..
        } => {
            let mut actual = BTreeSet::new();
            for chunk in chunks {
                collect_evaluator_capabilities(chunk, &mut actual)?;
            }
            let declared = required_runtime_capabilities
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            if declared.len() != required_runtime_capabilities.len() || declared != actual {
                return Err(RusticolError::integrity(format!(
                    "chunked evaluator capabilities {:?} do not match child capabilities {:?}",
                    required_runtime_capabilities, actual
                )));
            }
            output.extend(actual);
            Ok(())
        }
    }
}

fn validate_and_insert_capability(
    capability: &str,
    expected: &[&str],
    output: &mut BTreeSet<String>,
) -> RusticolResult<()> {
    if !expected.contains(&capability) {
        return Err(RusticolError::compatibility(format!(
            "evaluator declares runtime capability {capability:?}, expected one of {expected:?}"
        )));
    }
    output.insert(capability.to_string());
    Ok(())
}

#[cfg(any(
    not(feature = "f64-compiled"),
    not(feature = "f64-symjit"),
    not(feature = "symbolica-runtime")
))]
fn unsupported_capability(capability: &str) -> RusticolError {
    RusticolError::unsupported_runtime_capability(
        capability,
        format!(
            "this Rusticol build supports {:?}",
            supported_runtime_capabilities()
        ),
    )
}

#[cfg(feature = "symbolica-runtime")]
type LoadedEvaluatorState = (
    JITCompilationSettings,
    ExpressionEvaluator<Complex<Rational>>,
    Option<JITCompiledEvaluator<Complex<f64>>>,
);

#[cfg(feature = "symbolica-runtime")]
fn load_evaluator_state_source(
    source: &EvaluatorPayloadSource,
) -> RusticolResult<LoadedEvaluatorState> {
    let bytes = source.read()?;
    load_evaluator_state_bytes(bytes.as_ref(), &source.display_name())
}

#[cfg(feature = "symbolica-runtime")]
fn load_evaluator_state_bytes(
    bytes: &[u8],
    display_name: &str,
) -> RusticolResult<LoadedEvaluatorState> {
    type SavedEvaluator = (
        bool,
        JITCompilationSettings,
        ExpressionEvaluator<Complex<Rational>>,
        Option<JITCompiledEvaluator<f64>>,
        Option<JITCompiledEvaluator<Complex<f64>>>,
    );

    let ((_, settings, evaluator, _, jit_complex), consumed) =
        bincode::decode_from_slice::<SavedEvaluator, _>(bytes, bincode::config::standard())
            .map_err(|error| {
                RusticolError::compatibility(format!(
                    "evaluator state {} does not match Symbolica serialization ABI {}: {error}; regenerate the schema-v3 artifact",
                    display_name,
                    crate::SYMBOLICA_SERIALIZATION_ABI
                ))
            })?;
    ensure_evaluator_state_consumed(display_name, bytes.len(), consumed)?;
    Ok((settings, evaluator, jit_complex))
}

#[cfg(feature = "symbolica-runtime")]
fn ensure_evaluator_state_consumed(
    display_name: &str,
    encoded_len: usize,
    consumed: usize,
) -> RusticolResult<()> {
    if consumed == encoded_len {
        return Ok(());
    }
    Err(RusticolError::integrity(format!(
        "evaluator state {} contains {} trailing bytes",
        display_name,
        encoded_len - consumed
    )))
}

#[cfg(test)]
mod aosoa_tests {
    use super::*;

    fn source_value(row: usize, component: usize) -> Complex<f64> {
        Complex::new(
            (100 * row + component) as f64,
            -((100 * row + component) as f64) - 0.5,
        )
    }

    fn check_pack_and_scatter(lane_width: usize) {
        let batch_size = lane_width + 1;
        let source_parameter_count = 4;
        let input_components = [2, 0, 3];
        let source = (0..batch_size)
            .flat_map(|row| {
                (0..source_parameter_count).map(move |component| source_value(row, component))
            })
            .collect::<Vec<_>>();
        let mut packed = Vec::new();
        let block_count = pack_f64_parameters_aosoa(
            &source,
            batch_size,
            source_parameter_count,
            input_components.len(),
            lane_width,
            Some(&input_components),
            &mut packed,
        )
        .unwrap();
        assert_eq!(block_count, 2);
        for block in 0..block_count {
            for (local_component, source_component) in input_components.iter().copied().enumerate()
            {
                let scalar_start =
                    (block * input_components.len() + local_component) * 2 * lane_width;
                for lane in 0..lane_width {
                    let row = (block * lane_width + lane).min(batch_size - 1);
                    let expected = source_value(row, source_component);
                    assert_eq!(packed[scalar_start + lane], expected.re);
                    assert_eq!(packed[scalar_start + lane_width + lane], expected.im);
                }
            }
        }

        let evaluator_output_len = 4;
        let mut evaluator_outputs = vec![0.0; block_count * evaluator_output_len * 2 * lane_width];
        for block in 0..block_count {
            for column in 0..evaluator_output_len {
                let scalar_start = (block * evaluator_output_len + column) * 2 * lane_width;
                for lane in 0..lane_width {
                    let row = block * lane_width + lane;
                    evaluator_outputs[scalar_start + lane] = (1000 * row + column) as f64;
                    evaluator_outputs[scalar_start + lane_width + lane] =
                        -((1000 * row + column) as f64);
                }
            }
        }

        let state_parameter_count = 7;
        let sentinel = Complex::new(12345.0, -12345.0);
        let mut mapped_state = vec![sentinel; batch_size * state_parameter_count];
        scatter_f64_aosoa_outputs_to_state(
            &evaluator_outputs,
            batch_size,
            evaluator_output_len,
            lane_width,
            state_parameter_count,
            &mut mapped_state,
            &[(0, 1), (2, 5)],
            &[],
        )
        .unwrap();
        for row in 0..batch_size {
            assert_eq!(
                mapped_state[row * state_parameter_count + 1],
                Complex::new((1000 * row) as f64, -((1000 * row) as f64))
            );
            assert_eq!(
                mapped_state[row * state_parameter_count + 5],
                Complex::new((1000 * row + 2) as f64, -((1000 * row + 2) as f64))
            );
            assert_eq!(mapped_state[row * state_parameter_count], sentinel);
        }

        let mut spanned_state = vec![sentinel; batch_size * state_parameter_count];
        scatter_f64_aosoa_outputs_to_state(
            &evaluator_outputs,
            batch_size,
            evaluator_output_len,
            lane_width,
            state_parameter_count,
            &mut spanned_state,
            &[],
            &[(1, 3, 2)],
        )
        .unwrap();
        for row in 0..batch_size {
            for offset in 0..2 {
                assert_eq!(
                    spanned_state[row * state_parameter_count + 3 + offset],
                    Complex::new(
                        (1000 * row + 1 + offset) as f64,
                        -((1000 * row + 1 + offset) as f64)
                    )
                );
            }
            assert_eq!(spanned_state[row * state_parameter_count], sentinel);
        }

        let row_output_len = evaluator_output_len + 2;
        let mut row_outputs = vec![sentinel; batch_size * row_output_len];
        scatter_f64_aosoa_outputs_to_rows(
            &evaluator_outputs,
            batch_size,
            evaluator_output_len,
            lane_width,
            row_output_len,
            1,
            &mut row_outputs,
        )
        .unwrap();
        for row in 0..batch_size {
            assert_eq!(row_outputs[row * row_output_len], sentinel);
            assert_eq!(
                row_outputs[row * row_output_len + row_output_len - 1],
                sentinel
            );
            for column in 0..evaluator_output_len {
                assert_eq!(
                    row_outputs[row * row_output_len + 1 + column],
                    Complex::new(
                        (1000 * row + column) as f64,
                        -((1000 * row + column) as f64)
                    )
                );
            }
        }
    }

    #[test]
    fn portable_aosoa_pack_and_scatter_cover_fixed_and_dynamic_widths() {
        for lane_width in [2, 4, 8, 3] {
            check_pack_and_scatter(lane_width);
        }
    }

    #[test]
    fn aosoa_pack_rejects_inconsistent_mappings_and_lengths() {
        let mut scratch = Vec::new();
        let source = vec![c64(0.0, 0.0); 8];
        assert!(pack_f64_parameters_aosoa(&source, 2, 4, 2, 4, Some(&[0]), &mut scratch).is_err());
        assert!(
            pack_f64_parameters_aosoa(&source, 2, 4, 2, 4, Some(&[0, 4]), &mut scratch).is_err()
        );
        assert!(
            pack_f64_parameters_aosoa(&source[..7], 2, 4, 2, 4, Some(&[0, 1]), &mut scratch)
                .is_err()
        );
    }
}
