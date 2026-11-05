// SPDX-License-Identifier: 0BSD

use super::*;

struct RuntimeParameterStateSnapshot {
    values: Vec<f64>,
    particle_masses: BTreeMap<i32, f64>,
    normalization_factor: f64,
    helicity_sum: Option<Box<RuntimeParameterStateSnapshot>>,
    helicity_selectors: Vec<RuntimeParameterStateSnapshot>,
    color_selectors: BTreeMap<i64, RuntimeParameterStateSnapshot>,
}

pub(super) fn parse_complex_parameter_overrides(
    text: &str,
    path: &Path,
) -> RusticolResult<BTreeMap<String, (f64, f64)>> {
    let value = serde_json::from_str::<Value>(text).map_err(|err| {
        RusticolError::invalid_argument(format!(
            "could not parse model-parameter JSON {}: {err}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        RusticolError::invalid_argument(format!(
            "model-parameter JSON {} must contain an object",
            path.display()
        ))
    })?;
    let mut overrides = BTreeMap::new();
    for (name, value) in object {
        let (real, imaginary) = if let Some(real) = value.as_f64() {
            (real, 0.0)
        } else {
            let components = value.as_array().ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "model-parameter JSON entry {name:?} must be a real number or [real, imaginary]"
                ))
            })?;
            if components.len() != 2 {
                return Err(RusticolError::invalid_argument(format!(
                    "model-parameter JSON entry {name:?} must have exactly two components"
                )));
            }
            let real = components[0].as_f64().ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "model-parameter JSON entry {name:?} has a non-numeric real component"
                ))
            })?;
            let imaginary = components[1].as_f64().ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "model-parameter JSON entry {name:?} has a non-numeric imaginary component"
                ))
            })?;
            (real, imaginary)
        };
        if !real.is_finite() || !imaginary.is_finite() {
            return Err(RusticolError::invalid_argument(format!(
                "model-parameter JSON entry {name:?} must contain finite values"
            )));
        }
        overrides.insert(name.clone(), (real, imaginary));
    }
    Ok(overrides)
}

impl ExecutionRuntime {
    pub(super) fn current_model_parameter_value(&self, name: &str) -> RusticolResult<(f64, f64)> {
        let mut real = None;
        let mut imaginary = 0.0;
        for parameter in &self.model_parameters {
            if parameter.runtime_name.as_deref().unwrap_or(&parameter.name) != name {
                continue;
            }
            let value = self
                .model_parameter_values_f64
                .get(parameter.parameter_index)
                .copied()
                .ok_or_else(|| RusticolError::integrity("model-parameter value slot is absent"))?;
            match parameter.complex_component.as_deref() {
                Some("imag") => imaginary = value,
                Some("real") | None => real = Some(value),
                Some(_) => return Err(RusticolError::integrity("invalid parameter component")),
            }
        }
        real.map(|value| (value, imaginary)).ok_or_else(|| {
            RusticolError::model_parameter(format!("model parameter {name:?} has no runtime value"))
        })
    }

    pub(super) fn apply_model_parameter_overrides(
        &mut self,
        overrides: &BTreeMap<String, (f64, f64)>,
    ) -> RusticolResult<()> {
        let snapshot = self.parameter_state_snapshot();
        if let Err(error) = self.apply_model_parameter_overrides_recursive(overrides) {
            self.restore_parameter_state(snapshot);
            return Err(error);
        }
        Ok(())
    }

    fn apply_model_parameter_overrides_recursive(
        &mut self,
        overrides: &BTreeMap<String, (f64, f64)>,
    ) -> RusticolResult<()> {
        self.apply_model_parameter_overrides_local(overrides)?;
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            sum_runtime.apply_model_parameter_overrides_recursive(overrides)?;
        }
        for selector_runtime in &mut self.helicity_selector_runtimes {
            selector_runtime.apply_model_parameter_overrides_recursive(overrides)?;
        }
        for selector_runtime in self.color_selector_runtimes.values_mut() {
            selector_runtime.apply_model_parameter_overrides_recursive(overrides)?;
        }
        Ok(())
    }

    fn parameter_state_snapshot(&self) -> RuntimeParameterStateSnapshot {
        RuntimeParameterStateSnapshot {
            values: self.model_parameter_values_f64.clone(),
            particle_masses: self.particle_masses.clone(),
            normalization_factor: self.normalization_factor,
            helicity_sum: self
                .helicity_sum_runtime
                .as_deref()
                .map(ExecutionRuntime::parameter_state_snapshot)
                .map(Box::new),
            helicity_selectors: self
                .helicity_selector_runtimes
                .iter()
                .map(|runtime| runtime.parameter_state_snapshot())
                .collect(),
            color_selectors: self
                .color_selector_runtimes
                .iter()
                .map(|(sector_id, runtime)| (*sector_id, runtime.parameter_state_snapshot()))
                .collect(),
        }
    }

    fn restore_parameter_state(&mut self, snapshot: RuntimeParameterStateSnapshot) {
        self.model_parameter_values_f64 = snapshot.values;
        self.particle_masses = snapshot.particle_masses;
        self.normalization_factor = snapshot.normalization_factor;
        if let (Some(runtime), Some(snapshot)) =
            (self.helicity_sum_runtime.as_mut(), snapshot.helicity_sum)
        {
            runtime.restore_parameter_state(*snapshot);
        }
        for (runtime, lane_snapshot) in self
            .helicity_selector_runtimes
            .iter_mut()
            .zip(snapshot.helicity_selectors)
        {
            runtime.restore_parameter_state(lane_snapshot);
        }
        for (sector_id, lane_snapshot) in snapshot.color_selectors {
            if let Some(runtime) = self.color_selector_runtimes.get_mut(&sector_id) {
                runtime.restore_parameter_state(lane_snapshot);
            }
        }
    }

    fn apply_model_parameter_overrides_local(
        &mut self,
        overrides: &BTreeMap<String, (f64, f64)>,
    ) -> RusticolResult<()> {
        let mut proposed = self.model_parameter_values_f64.clone();
        for (name, (real, imaginary)) in overrides {
            if self.model_parameters.iter().any(|parameter| {
                parameter.kind == "derived_parameter_component"
                    && parameter.runtime_name.as_deref().unwrap_or(&parameter.name) == name
            }) {
                return Err(RusticolError::invalid_argument(format!(
                    "derived model parameter {name:?} is immutable",
                )));
            }
            let Some(slots) = self.model_parameter_runtime_slots.get(name).copied() else {
                return Err(RusticolError::invalid_argument(format!(
                    "model-parameter override {name:?} is not used by process {}",
                    self.process
                )));
            };
            if slots.real >= self.model_parameter_values_f64.len()
                || !real.is_finite()
                || !imaginary.is_finite()
            {
                return Err(RusticolError::invalid_argument(format!(
                    "model-parameter override {name:?} has invalid value [{real}, {imaginary}]",
                )));
            }
            proposed[slots.real] = *real;
            if let Some(index) = slots.imaginary {
                if index >= self.model_parameter_values_f64.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "model-parameter override {name:?} has an invalid imaginary slot"
                    )));
                }
                proposed[index] = *imaginary;
            } else if *imaginary != 0.0 {
                return Err(RusticolError::invalid_argument(format!(
                    "real model parameter {name:?} cannot receive a nonzero imaginary component"
                )));
            }
        }
        refresh_derived_model_parameter_values(
            self.model_parameter_evaluator.as_mut(),
            &mut proposed,
        )?;
        let proposed_masses = self.particle_mass_parameters_for(&proposed);
        validate_particle_mass_class_stability(&self.particle_masses, &proposed_masses)?;
        self.model_parameter_values_f64 = proposed;
        self.particle_masses = proposed_masses;
        self.refresh_normalization_factor();
        Ok(())
    }

    pub(super) fn refresh_derived_model_parameters(&mut self) -> RusticolResult<()> {
        refresh_derived_model_parameter_values(
            self.model_parameter_evaluator.as_mut(),
            &mut self.model_parameter_values_f64,
        )
    }

    fn particle_mass_parameters_for(&self, values: &[f64]) -> BTreeMap<i32, f64> {
        let mut particle_masses = self.particle_masses.clone();
        for parameter in &self.model_parameters {
            if parameter.kind == "particle_mass"
                && let Some(pdg) = parameter.pdg
                && let Some(value) = values.get(parameter.parameter_index)
            {
                particle_masses.insert(pdg, *value);
            }
        }
        for (pdg, name) in &self.particle_mass_parameter_names {
            let Some(slots) = self.model_parameter_runtime_slots.get(name) else {
                continue;
            };
            if let Some(value) = values.get(slots.real) {
                particle_masses.insert(*pdg, *value);
            }
        }
        particle_masses
    }

    pub(super) fn refresh_normalization_factor(&mut self) {
        let alpha_s = self
            .model_parameter_name_to_index
            .get("normalization.alpha_s_me_check")
            .and_then(|index| self.model_parameter_values_f64.get(*index))
            .copied();
        let alpha_ew = self
            .model_parameter_name_to_index
            .get("normalization.alpha_ew")
            .and_then(|index| self.model_parameter_values_f64.get(*index))
            .copied();
        let mut global_coupling_factor = 1.0;
        if self.normalization_qcd_coupling_power > 0 {
            let Some(alpha_s) = alpha_s else {
                return;
            };
            global_coupling_factor *= (4.0 * std::f64::consts::PI * alpha_s)
                .powi(self.normalization_qcd_coupling_power as i32);
        }
        if self.normalization_electroweak_coupling_power > 0 {
            let Some(alpha_ew) = alpha_ew else {
                return;
            };
            global_coupling_factor *= (2.0 * 4.0 * std::f64::consts::PI * alpha_ew)
                .powi(self.normalization_electroweak_coupling_power as i32);
        }
        self.normalization_factor = self.normalization_color_factor * global_coupling_factor
            / (self.normalization_average_factor * self.normalization_identical_factor);
    }
}

pub(super) fn refresh_derived_model_parameter_values(
    runtime: Option<&mut ModelParameterEvaluatorRuntime>,
    values: &mut [f64],
) -> RusticolResult<()> {
    let Some(runtime) = runtime else {
        return Ok(());
    };
    let parameters = runtime
        .input_parameter_indices
        .iter()
        .map(|index| {
            values
                .get(*index)
                .copied()
                .map(|value| c64(value, 0.0))
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "model-parameter evaluator input index {index} is out of range"
                    ))
                })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let evaluated = runtime.evaluator.evaluate_single_row(&parameters)?;
    for output in &runtime.outputs {
        let value = evaluated.get(output.output_index).ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "model-parameter evaluator output {} for {:?} is absent",
                output.output_index, output.runtime_name
            ))
        })?;
        let Some(real) = values.get_mut(output.real_parameter_index) else {
            return Err(RusticolError::invalid_argument(format!(
                "derived model-parameter real slot {} is out of range",
                output.real_parameter_index
            )));
        };
        *real = value.re;
        if let Some(index) = output.imag_parameter_index {
            let imaginary = values.get_mut(index).ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "derived model-parameter imaginary slot {index} is out of range"
                ))
            })?;
            *imaginary = value.im;
        }
    }
    Ok(())
}

fn validate_particle_mass_class_stability(
    previous: &BTreeMap<i32, f64>,
    proposed: &BTreeMap<i32, f64>,
) -> RusticolResult<()> {
    for (pdg, previous_mass) in previous {
        let Some(proposed_mass) = proposed.get(pdg) else {
            continue;
        };
        if (*previous_mass == 0.0) != (*proposed_mass == 0.0) {
            return Err(RusticolError::invalid_argument(format!(
                "model-parameter update changes particle {pdg} from a {} to a {} mass class; regenerate the process artifact for that mass class",
                if *previous_mass == 0.0 {
                    "massless"
                } else {
                    "massive"
                },
                if *proposed_mass == 0.0 {
                    "massless"
                } else {
                    "massive"
                },
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod parameter_json_tests {
    use super::parse_complex_parameter_overrides;
    use std::path::Path;

    #[test]
    fn parameter_json_accepts_real_scalars_and_complex_pairs() {
        let parsed = parse_complex_parameter_overrides(
            r#"{"aS": 0.117, "complex": [2.5, -0.25]}"#,
            Path::new("parameters.json"),
        )
        .expect("parameter card should parse");

        assert_eq!(parsed.get("aS"), Some(&(0.117, 0.0)));
        assert_eq!(parsed.get("complex"), Some(&(2.5, -0.25)));
    }

    #[test]
    fn parameter_json_rejects_non_numeric_values() {
        let error =
            parse_complex_parameter_overrides(r#"{"aS": "0.117"}"#, Path::new("parameters.json"))
                .expect_err("string parameter must fail");

        assert!(
            error
                .to_string()
                .contains("must be a real number or [real, imaginary]")
        );
    }
}
