// SPDX-License-Identifier: 0BSD

use super::super::*;

impl StageRuntime {
    pub(crate) fn load(
        stage: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
    ) -> RusticolResult<Self> {
        let evaluator = EvaluatorGroup::load_from_store(&stage.evaluator, payloads)?;
        Self::load_with_evaluator(stage, evaluator, false)
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn load_exact_from_plane(
        stage: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
    ) -> RusticolResult<Self> {
        let evaluator = EvaluatorGroup::load_exact_from_plane(stage, payloads)?;
        Self::load_with_evaluator(stage, evaluator, true)
    }

    fn load_with_evaluator(
        stage: &GenericSerializedStageEvaluatorManifest,
        evaluator: EvaluatorGroup,
        plane_bindings: bool,
    ) -> RusticolResult<Self> {
        let mut outputs = Vec::new();
        if plane_bindings {
            let direct = stage.compiled_plane_arena.as_ref().ok_or_else(|| {
                RusticolError::integrity(
                    "compiled plane-arena exact stage has no serialized bindings",
                )
            })?;
            for binding in &direct.output_bindings {
                if binding.arena != "current" {
                    return Err(RusticolError::integrity(
                        "compiled plane-arena exact current stage names the wrong output arena",
                    ));
                }
                outputs.push((binding.output_index, binding.component));
            }
        } else {
            for slot in &stage.output_slots {
                let output_len =
                    slot.output_stop
                        .checked_sub(slot.output_start)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(format!(
                                "generic stage {} has an invalid output slot range",
                                stage.evaluator_label
                            ))
                        })?;
                let component_len = slot
                    .component_stop
                    .checked_sub(slot.component_start)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "generic stage {} has an invalid component slot range",
                            stage.evaluator_label
                        ))
                    })?;
                if output_len != component_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "generic stage {} output slot length does not match component length",
                        stage.evaluator_label
                    )));
                }
                for component in 0..component_len {
                    outputs.push((
                        slot.output_start + component,
                        slot.component_start + component,
                    ));
                }
            }
        }
        outputs.sort_unstable_by_key(|(column, _state_offset)| *column);
        let (input_components, input_spans) =
            if stage.parameter_layout == "stage-local-value-momentum" {
                let mut map = vec![0usize; stage.parameter_count];
                if plane_bindings {
                    let direct = stage.compiled_plane_arena.as_ref().ok_or_else(|| {
                        RusticolError::integrity(
                            "compiled plane-arena exact stage has no serialized bindings",
                        )
                    })?;
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
        let output_spans = contiguous_output_spans(&outputs);
        let (chunk_outputs, chunk_output_spans) =
            localized_chunk_output_layouts(&evaluator.output_chunk_lengths(), &outputs)?;
        Ok(Self {
            outputs,
            output_spans,
            chunk_outputs,
            chunk_output_spans,
            input_components,
            input_spans,
            parameter_scratch_f64: Vec::new(),
            output_scratch_f64: Vec::new(),
            evaluator,
        })
    }

    pub(crate) fn evaluate_f64_into_state(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        let evaluator_params = if let Some(input_components) = self.input_components.as_ref() {
            let local_parameter_count = input_components.len();
            self.parameter_scratch_f64
                .resize(batch_size * local_parameter_count, c64(0.0, 0.0));
            for row in 0..batch_size {
                let row_state = row * parameter_count;
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
            if self.evaluator.is_chunked() {
                return self.evaluator.evaluate_chunks_f64_into_state(
                    batch_size,
                    &self.parameter_scratch_f64,
                    parameter_count,
                    state,
                    &self.chunk_outputs,
                    &self.chunk_output_spans,
                );
            }
            self.parameter_scratch_f64.as_slice()
        } else {
            &*state
        };
        self.evaluator.evaluate_batch_into(
            batch_size,
            evaluator_params,
            &mut self.output_scratch_f64,
        )?;
        for row in 0..batch_size {
            let row_state = row * parameter_count;
            let row_eval = row * self.evaluator.output_len;
            if self.output_spans.is_empty() {
                for (column, state_offset) in &self.outputs {
                    state[row_state + *state_offset] = self.output_scratch_f64[row_eval + *column];
                }
            } else {
                for (column_start, state_offset_start, len) in &self.output_spans {
                    let source_start = row_eval + *column_start;
                    let target_start = row_state + *state_offset_start;
                    state[target_start..target_start + *len].copy_from_slice(
                        &self.output_scratch_f64[source_start..source_start + *len],
                    );
                }
            }
        }
        Ok(())
    }

    pub(crate) fn evaluate_f64_into_state_profile(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<f64>],
    ) -> RusticolResult<StageEvaluationProfile> {
        let mut input_pack_elapsed = Duration::ZERO;
        let evaluator;
        let mut scratch_reallocation_count = 0;
        if let Some(input_components) = self.input_components.as_ref() {
            let local_parameter_count = input_components.len();
            let pack_start = Instant::now();
            let capacity = self.parameter_scratch_f64.capacity();
            self.parameter_scratch_f64
                .resize(batch_size * local_parameter_count, c64(0.0, 0.0));
            scratch_reallocation_count +=
                u64::from(self.parameter_scratch_f64.capacity() != capacity);
            for row in 0..batch_size {
                let row_state = row * parameter_count;
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
            input_pack_elapsed = pack_start.elapsed();
            if self.evaluator.is_chunked() {
                let (evaluator, assign_s) = self.evaluator.evaluate_chunks_f64_into_state_profile(
                    batch_size,
                    &self.parameter_scratch_f64,
                    parameter_count,
                    state,
                    &self.chunk_outputs,
                    &self.chunk_output_spans,
                )?;
                return Ok(StageEvaluationProfile {
                    input_pack_s: input_pack_elapsed.as_secs_f64(),
                    evaluator,
                    output_assign_s: assign_s,
                    input_copy_component_count: (batch_size * local_parameter_count) as u64,
                    output_assign_component_count: (batch_size * self.outputs.len()) as u64,
                    scratch_reallocation_count,
                });
            }
            evaluator = self.evaluator.evaluate_batch_into_profile(
                batch_size,
                &self.parameter_scratch_f64,
                &mut self.output_scratch_f64,
                false,
            )?;
        } else {
            evaluator = self.evaluator.evaluate_batch_into_profile(
                batch_size,
                state,
                &mut self.output_scratch_f64,
                false,
            )?;
        }

        let assign_start = Instant::now();
        for row in 0..batch_size {
            let row_state = row * parameter_count;
            let row_eval = row * self.evaluator.output_len;
            if self.output_spans.is_empty() {
                for (column, state_offset) in &self.outputs {
                    state[row_state + *state_offset] = self.output_scratch_f64[row_eval + *column];
                }
            } else {
                for (column_start, state_offset_start, len) in &self.output_spans {
                    let source_start = row_eval + *column_start;
                    let target_start = row_state + *state_offset_start;
                    state[target_start..target_start + *len].copy_from_slice(
                        &self.output_scratch_f64[source_start..source_start + *len],
                    );
                }
            }
        }
        Ok(StageEvaluationProfile {
            input_pack_s: input_pack_elapsed.as_secs_f64(),
            evaluator,
            output_assign_s: assign_start.elapsed().as_secs_f64(),
            input_copy_component_count: self
                .input_components
                .as_ref()
                .map_or(0, |components| batch_size * components.len())
                as u64,
            output_assign_component_count: (batch_size * self.outputs.len()) as u64,
            scratch_reallocation_count,
        })
    }

    pub(crate) fn evaluate_active_chunks_f64_into_state(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<f64>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<()> {
        self.evaluator.evaluate_selected_chunks_f64_into_state(
            batch_size,
            parameter_count,
            state,
            self.input_components.as_deref(),
            &self.input_spans,
            &self.chunk_outputs,
            &self.chunk_output_spans,
            active_chunk_indices,
        )
    }

    pub(crate) fn evaluate_active_chunks_f64_into_state_profile(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<f64>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<StageEvaluationProfile> {
        let (evaluator, output_assign_s) = self
            .evaluator
            .evaluate_selected_chunks_f64_into_state_profile(
                batch_size,
                parameter_count,
                state,
                self.input_components.as_deref(),
                &self.input_spans,
                &self.chunk_outputs,
                &self.chunk_output_spans,
                active_chunk_indices,
            )?;
        Ok(StageEvaluationProfile {
            input_pack_s: evaluator.leaf_input_pack_s,
            output_assign_s,
            // The selected-chunk evaluator gathers directly from global
            // state into leaf inputs. There is no distinct parent stage-input
            // copy to count in addition to the evaluator's leaf gather.
            input_copy_component_count: 0,
            output_assign_component_count: active_chunk_indices
                .iter()
                .map(|index| self.chunk_outputs[*index].len() as u64 * batch_size as u64)
                .sum(),
            scratch_reallocation_count: 0,
            evaluator,
        })
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_generic_into_state<T>(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<T>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(f64, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        self.evaluate_selected_chunks_generic_into_state(
            batch_size,
            parameter_count,
            state,
            binary_precision,
            None,
        )
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_selected_chunks_generic_into_state<T>(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<T>],
        binary_precision: Option<u32>,
        active_chunk_indices: Option<&[usize]>,
    ) -> RusticolResult<(f64, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let mut input_pack_s = 0.0;
        let (evaluated, evaluator_s) =
            if let Some(input_components) = self.input_components.as_ref() {
                let local_parameter_count = input_components.len();
                let pack_start = Instant::now();
                let mut parameter_scratch =
                    vec![complex_zero::<T>(); batch_size * local_parameter_count];
                for row in 0..batch_size {
                    let row_state = row * parameter_count;
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
                input_pack_s = pack_start.elapsed().as_secs_f64();
                let eval_start = Instant::now();
                let evaluated = self.evaluator.evaluate_selected_chunks_generic(
                    batch_size,
                    &parameter_scratch,
                    binary_precision,
                    active_chunk_indices,
                )?;
                (evaluated, eval_start.elapsed().as_secs_f64())
            } else {
                let eval_start = Instant::now();
                let evaluated = self.evaluator.evaluate_selected_chunks_generic(
                    batch_size,
                    state,
                    binary_precision,
                    active_chunk_indices,
                )?;
                (evaluated, eval_start.elapsed().as_secs_f64())
            };

        if let Some(indices) = active_chunk_indices {
            // Inactive chunks may share storage with still-live currents.
            // Assign only the selected leaves, never their zero placeholders.
            let assign_start = Instant::now();
            let mut offset = 0;
            let chunk_offsets = self
                .evaluator
                .output_chunk_lengths()
                .into_iter()
                .map(|length| {
                    let start = offset;
                    offset += length;
                    start
                })
                .collect::<Vec<_>>();
            for row in 0..batch_size {
                let row_state = row * parameter_count;
                let row_eval = row * self.evaluator.output_len;
                for &index in indices {
                    for &(column, state_offset) in &self.chunk_outputs[index] {
                        state[row_state + state_offset] =
                            evaluated[row_eval + chunk_offsets[index] + column].clone();
                    }
                }
            }
            return Ok((
                input_pack_s,
                evaluator_s,
                assign_start.elapsed().as_secs_f64(),
            ));
        }
        self.assign_generic_outputs(
            batch_size,
            parameter_count,
            state,
            evaluated,
            input_pack_s,
            evaluator_s,
        )
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn assign_generic_outputs<T>(
        &self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<T>],
        evaluated: Vec<Complex<T>>,
        input_pack_s: f64,
        evaluator_s: f64,
    ) -> RusticolResult<(f64, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let assign_start = Instant::now();
        for row in 0..batch_size {
            let row_state = row * parameter_count;
            let row_eval = row * self.evaluator.output_len;
            if self.output_spans.is_empty() {
                for (column, state_offset) in &self.outputs {
                    state[row_state + *state_offset] = evaluated[row_eval + *column].clone();
                }
            } else {
                for (column_start, state_offset_start, len) in &self.output_spans {
                    let source_start = row_eval + *column_start;
                    let target_start = row_state + *state_offset_start;
                    state[target_start..target_start + *len]
                        .clone_from_slice(&evaluated[source_start..source_start + *len]);
                }
            }
        }
        Ok((
            input_pack_s,
            evaluator_s,
            assign_start.elapsed().as_secs_f64(),
        ))
    }
}

type ChunkOutputs = Vec<Vec<(usize, usize)>>;
type ChunkOutputSpans = Vec<Vec<(usize, usize, usize)>>;

fn localized_chunk_output_layouts(
    chunk_lengths: &[usize],
    outputs: &[(usize, usize)],
) -> RusticolResult<(ChunkOutputs, ChunkOutputSpans)> {
    if chunk_lengths.iter().sum::<usize>() != outputs.len() {
        return Err(RusticolError::invalid_argument(
            "stage output layout length does not match evaluator output length",
        ));
    }
    let mut chunk_outputs = Vec::with_capacity(chunk_lengths.len());
    let mut chunk_spans = Vec::with_capacity(chunk_lengths.len());
    let mut output_offset = 0usize;
    for chunk_len in chunk_lengths {
        let chunk_stop = output_offset + *chunk_len;
        let mut localized = Vec::with_capacity(*chunk_len);
        for (expected_column, (column, state_offset)) in outputs[output_offset..chunk_stop]
            .iter()
            .copied()
            .enumerate()
        {
            let expected_global_column = output_offset + expected_column;
            if column != expected_global_column {
                return Err(RusticolError::invalid_argument(
                    "stage output columns are not contiguous in evaluator order",
                ));
            }
            localized.push((expected_column, state_offset));
        }
        chunk_spans.push(contiguous_output_spans(&localized));
        chunk_outputs.push(localized);
        output_offset = chunk_stop;
    }
    Ok((chunk_outputs, chunk_spans))
}

pub(crate) fn contiguous_input_spans(input_components: &[usize]) -> Vec<(usize, usize, usize)> {
    if input_components.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut local_start = 0usize;
    let mut global_start = input_components[0];
    let mut previous_local = local_start;
    let mut previous_global = global_start;
    let mut len = 1usize;
    for (local, global) in input_components.iter().copied().enumerate().skip(1) {
        if local == previous_local + 1 && global == previous_global + 1 {
            previous_local = local;
            previous_global = global;
            len += 1;
            continue;
        }
        spans.push((local_start, global_start, len));
        local_start = local;
        global_start = global;
        previous_local = local;
        previous_global = global;
        len = 1;
    }
    spans.push((local_start, global_start, len));
    if spans.len() >= input_components.len() {
        Vec::new()
    } else {
        spans
    }
}

pub(crate) fn contiguous_output_spans(outputs: &[(usize, usize)]) -> Vec<(usize, usize, usize)> {
    if outputs.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let (mut output_start, mut state_start) = outputs[0];
    let mut previous_output = output_start;
    let mut previous_state = state_start;
    let mut len = 1usize;
    for (output, state) in outputs.iter().copied().skip(1) {
        if output == previous_output + 1 && state == previous_state + 1 {
            previous_output = output;
            previous_state = state;
            len += 1;
            continue;
        }
        spans.push((output_start, state_start, len));
        output_start = output;
        state_start = state;
        previous_output = output;
        previous_state = state;
        len = 1;
    }
    spans.push((output_start, state_start, len));
    if spans.len() >= outputs.len() {
        Vec::new()
    } else {
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(
        feature = "f64-compiled",
        any(target_os = "linux", target_os = "macos")
    ))]
    use std::process::Command;
    #[cfg(all(
        feature = "f64-compiled",
        any(target_os = "linux", target_os = "macos")
    ))]
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn chunk_output_layouts_are_localized_without_losing_state_offsets() {
        let outputs = vec![(0, 10), (1, 11), (2, 20), (3, 22), (4, 23)];
        let (chunks, spans) = localized_chunk_output_layouts(&[2, 3], &outputs).unwrap();

        assert_eq!(chunks[0], vec![(0, 10), (1, 11)]);
        assert_eq!(chunks[1], vec![(0, 20), (1, 22), (2, 23)]);
        assert_eq!(spans[0], vec![(0, 10, 2)]);
        assert_eq!(spans[1], vec![(0, 20, 1), (1, 22, 2)]);
    }

    #[test]
    fn chunk_output_layouts_reject_inconsistent_evaluator_metadata() {
        let length_error = localized_chunk_output_layouts(&[1], &[(0, 10), (1, 11)])
            .expect_err("chunk lengths must cover every output");
        assert!(
            length_error
                .to_string()
                .contains("layout length does not match")
        );

        let order_error = localized_chunk_output_layouts(&[2], &[(0, 10), (2, 11)])
            .expect_err("output columns must be contiguous");
        assert!(
            order_error
                .to_string()
                .contains("columns are not contiguous")
        );
    }

    #[cfg(feature = "symbolica-runtime")]
    #[test]
    fn generic_selected_chunks_preserve_inactive_live_current_slots() {
        use symbolica::{atom::AtomCore, parse};

        let leaf = |expression: &str| LoadedEvaluator {
            eval: F64Evaluator::ExactOnly,
            exact_eval: Some(
                parse!(expression)
                    .evaluator(&[parse!("selected_chunk_x")])
                    .build()
                    .unwrap(),
            ),
            exact_eval_source: None,
            double_eval: None,
            arb_eval: None,
            input_len: 1,
            output_len: 1,
        };
        let evaluator = EvaluatorGroup {
            evaluators: vec![
                leaf("selected_chunk_x + 100"),
                leaf("selected_chunk_x + 200"),
            ],
            input_len: 2,
            input_mappings: vec![Some(vec![0]), Some(vec![1])],
            input_mapping_spans: vec![Vec::new(), Vec::new()],
            output_len: 2,
            chunk_parameter_scratch_f64: Vec::new(),
            chunk_scratch_f64: Vec::new(),
            chunk_parameter_scratch_aosoa_f64: Vec::new(),
            chunk_scratch_aosoa_f64: Vec::new(),
            chunk_input_mapping_scratch: Vec::new(),
        };
        let mut stage = StageRuntime {
            outputs: vec![(0, 1), (1, 3)],
            output_spans: Vec::new(),
            chunk_outputs: vec![vec![(0, 1)], vec![(0, 3)]],
            chunk_output_spans: vec![Vec::new(), Vec::new()],
            input_components: Some(vec![2, 0]),
            input_spans: Vec::new(),
            parameter_scratch_f64: Vec::new(),
            output_scratch_f64: Vec::new(),
            evaluator,
        };
        let number = |value| c_generic(DoubleFloat::from(value), DoubleFloat::from(0.0));
        let initial = [3.0, -99.0, 7.0, -99.0, 11.0, -99.0, 13.0, -99.0]
            .map(number)
            .to_vec();
        let mut state = initial.clone();
        stage
            .evaluate_selected_chunks_generic_into_state(2, 4, &mut state, None, Some(&[1]))
            .unwrap();
        assert!(stage.evaluator.evaluators[0].double_eval.is_none());
        assert!(stage.evaluator.evaluators[1].double_eval.is_some());
        assert_eq!(state[1], number(-99.0));
        assert_eq!(state[5], number(-99.0));
        assert_eq!(state[3], number(203.0));
        assert_eq!(state[7], number(211.0));

        stage
            .evaluate_selected_chunks_generic_into_state(2, 4, &mut state, None, Some(&[0]))
            .unwrap();
        assert_eq!(state[1], number(107.0));
        assert_eq!(state[5], number(113.0));
        assert_eq!(state[3], number(203.0));
        assert_eq!(state[7], number(211.0));
        let mut complete = initial;
        stage
            .evaluate_generic_into_state(2, 4, &mut complete, None)
            .unwrap();
        assert_eq!(state, complete);
        for invalid in [&[1, 1][..], &[1, 0][..], &[2][..]] {
            assert!(
                stage
                    .evaluate_selected_chunks_generic_into_state(
                        2,
                        4,
                        &mut state,
                        None,
                        Some(invalid),
                    )
                    .is_err()
            );
            assert_eq!(state, complete);
        }
    }

    #[cfg(all(
        feature = "f64-compiled",
        any(target_os = "linux", target_os = "macos")
    ))]
    #[test]
    fn alternating_selected_stage_chunks_cannot_leak_inactive_prior_state() {
        let (directory, library, marker) = selective_stage_fixture();
        let library_name = library.file_name().unwrap().to_str().unwrap().to_string();
        let leaf = |function_name: &str| EvaluatorManifest::CompiledComplex {
            runtime_capability: SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY.to_string(),
            function_name: function_name.to_string(),
            input_len: 1,
            output_len: 1,
            library_path: library_name.clone(),
            evaluator_state_path: None,
            number_type: "complex".to_string(),
            native_direct_application: None,
        };
        let evaluator = EvaluatorGroup::load(
            &EvaluatorManifest::Chunked {
                required_runtime_capabilities: vec![
                    SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY.to_string(),
                ],
                input_len: Some(2),
                chunk_input_indices: Some(vec![vec![0], vec![1]]),
                chunks: vec![leaf("stage_chunk_0"), leaf("stage_chunk_1")],
            },
            &directory,
        )
        .unwrap();
        let outputs = vec![(0, 1), (1, 3)];
        let mut stage = StageRuntime {
            output_spans: contiguous_output_spans(&outputs),
            outputs,
            chunk_outputs: vec![vec![(0, 1)], vec![(0, 3)]],
            chunk_output_spans: vec![Vec::new(), Vec::new()],
            input_components: Some(vec![2, 0]),
            input_spans: Vec::new(),
            parameter_scratch_f64: Vec::new(),
            output_scratch_f64: Vec::new(),
            evaluator,
        };

        let sentinel = c64(-99.0, 0.0);
        let mut selected_state = vec![c64(3.0, 0.0), sentinel, c64(7.0, 0.0), sentinel];
        let selected_profile = stage
            .evaluate_active_chunks_f64_into_state_profile(1, 4, &mut selected_state, &[1])
            .unwrap();
        assert_eq!(selected_state[1], sentinel, "inactive output was assigned");
        assert_eq!(selected_state[3], c64(203.0, 0.0));
        assert_eq!(
            selected_profile.input_copy_component_count, 0,
            "composed selected-chunk gather is not a separate parent input copy"
        );
        assert_eq!(
            selected_profile.evaluator.leaf_input_copy_component_count, 1,
            "the composed selected-chunk gather must be counted once as a leaf input"
        );
        assert_eq!(fs::read_to_string(&marker).unwrap(), "1");

        let parameter_capacity = stage.evaluator.chunk_parameter_scratch_f64.capacity();
        let output_capacity = stage.evaluator.chunk_scratch_f64.capacity();
        selected_state[3] = sentinel;
        stage
            .evaluate_active_chunks_f64_into_state(1, 4, &mut selected_state, &[1])
            .unwrap();
        assert_eq!(
            stage.evaluator.chunk_parameter_scratch_f64.capacity(),
            parameter_capacity
        );
        assert_eq!(
            stage.evaluator.chunk_scratch_f64.capacity(),
            output_capacity
        );

        let marker_before_invalid = fs::read_to_string(&marker).unwrap();
        for invalid in [&[1, 1][..], &[1, 0][..], &[2][..]] {
            assert!(
                stage
                    .evaluate_active_chunks_f64_into_state(1, 4, &mut selected_state, invalid)
                    .is_err()
            );
        }
        assert_eq!(fs::read_to_string(&marker).unwrap(), marker_before_invalid);

        fs::write(&marker, "").unwrap();
        let mut full_state = vec![c64(3.0, 0.0), sentinel, c64(7.0, 0.0), sentinel];
        stage
            .evaluate_f64_into_state(1, 4, &mut full_state)
            .unwrap();
        assert_eq!(full_state[1], c64(107.0, 0.0));
        assert_eq!(full_state[3], c64(203.0, 0.0));
        assert_eq!(fs::read_to_string(&marker).unwrap(), "01");

        // The execution runtime clears its reusable state before every
        // selector-pruned schedule. Alternate schedules on the same backing
        // allocation to ensure inactive outputs cannot retain a prior value.
        for chunk_index in [0, 1, 0, 1] {
            selected_state.fill(c64(0.0, 0.0));
            selected_state[0] = c64(3.0, 0.0);
            selected_state[2] = c64(7.0, 0.0);
            stage
                .evaluate_active_chunks_f64_into_state(1, 4, &mut selected_state, &[chunk_index])
                .unwrap();
            if chunk_index == 0 {
                assert_eq!(selected_state[1], c64(107.0, 0.0));
                assert_eq!(selected_state[3], c64(0.0, 0.0));
            } else {
                assert_eq!(selected_state[1], c64(0.0, 0.0));
                assert_eq!(selected_state[3], c64(203.0, 0.0));
            }
        }

        let parent_inputs = [c64(7.0, 0.0), c64(3.0, 0.0)];
        let mut selected_outputs = vec![sentinel; 2];
        for chunk_index in [0, 1, 0, 1] {
            stage
                .evaluator
                .evaluate_selected_chunks_f64_into_output(
                    1,
                    &parent_inputs,
                    None,
                    &[],
                    &mut selected_outputs,
                    &[chunk_index],
                )
                .unwrap();
            if chunk_index == 0 {
                assert_eq!(selected_outputs, vec![c64(107.0, 0.0), c64(0.0, 0.0)]);
            } else {
                assert_eq!(selected_outputs, vec![c64(0.0, 0.0), c64(203.0, 0.0)]);
            }
        }

        drop(stage);
        fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(all(
        feature = "f64-compiled",
        any(target_os = "linux", target_os = "macos")
    ))]
    fn selective_stage_fixture() -> (PathBuf, PathBuf, PathBuf) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "rusticol-selective-stage-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let source = directory.join("fixture.cpp");
        let marker = directory.join("calls.txt");
        fs::write(&marker, "").unwrap();
        let library = directory.join(if cfg!(target_os = "macos") {
            "libfixture.dylib"
        } else {
            "libfixture.so"
        });
        let marker_literal = marker.to_string_lossy().replace('\\', "\\\\");
        fs::write(
            &source,
            format!(
                r#"#include <complex>
#include <cstdio>
static void mark(const char* value) {{
    std::FILE* file = std::fopen("{marker_literal}", "a");
    std::fputs(value, file);
    std::fclose(file);
}}
extern "C" unsigned long stage_chunk_0_complexf64_get_buffer_len() {{ return 1; }}
extern "C" void stage_chunk_0_complexf64(
    std::complex<double>* params,
    std::complex<double>*,
    std::complex<double>* out) {{
    mark("0");
    out[0] = params[0] + std::complex<double>(100.0, 0.0);
}}
extern "C" unsigned long stage_chunk_1_complexf64_get_buffer_len() {{ return 1; }}
extern "C" void stage_chunk_1_complexf64(
    std::complex<double>* params,
    std::complex<double>*,
    std::complex<double>* out) {{
    mark("1");
    out[0] = params[0] + std::complex<double>(200.0, 0.0);
}}
"#
            ),
        )
        .unwrap();
        let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
        let mut command = Command::new(compiler);
        command.arg("-std=c++17");
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
            "could not compile selective stage fixture: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        (directory, library, marker)
    }
}
