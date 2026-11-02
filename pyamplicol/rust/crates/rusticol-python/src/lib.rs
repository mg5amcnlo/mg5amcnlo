// SPDX-License-Identifier: 0BSD

#[cfg(feature = "numpy")]
mod eager_lowering;
#[cfg(feature = "numpy")]
mod recurrence;
#[cfg(feature = "numpy")]
mod recurrence_numerical_evidence;
#[cfg(feature = "numpy")]
mod recurrence_template;

#[cfg(feature = "numpy")]
use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyInt, PyList, PyTuple};
use rusticol_core::__private::{
    build_recurrence_bootstrap_image_v1, compile_symbolica_program_to_plane_application_bytes,
};
use rusticol_core::{
    ColorAccuracy, ColorComponent as CoreColorComponent, ModelParameter as CoreModelParameter,
    NativeOnTheFlyWarmUpEvent, NativeOnTheFlyWarmUpEventKind, NativeOnTheFlyWarmUpObserver,
    NativeOnTheFlyWarmUpStage, NativeResolvedEvaluation, NativeRuntime, NativeRuntimeProfile,
    ParameterKind, ParticleRole, ProcessPhysics as CoreProcessPhysics, ReductionKind,
    RusticolError as CoreError, RusticolErrorKind,
    eager_direct_descriptor_for_source_application_bytes, preflight_prepared_kernel_pack,
    runtime_target_info,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

create_exception!(_rusticol, RusticolError, PyException);
create_exception!(_rusticol, ArtifactError, RusticolError);
create_exception!(_rusticol, CompatibilityError, RusticolError);
create_exception!(_rusticol, EvaluationError, RusticolError);
create_exception!(_rusticol, SelectorError, EvaluationError);
create_exception!(_rusticol, ModelParameterError, EvaluationError);

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct TargetInfo {
    triple: String,
    cpu_features: Vec<String>,
}

fn python_error(error: CoreError) -> PyErr {
    let message = error.to_string();
    match error.kind() {
        RusticolErrorKind::InvalidArgument => PyValueError::new_err(message),
        RusticolErrorKind::Artifact
        | RusticolErrorKind::Security
        | RusticolErrorKind::Integrity
        | RusticolErrorKind::Serialization => ArtifactError::new_err(message),
        RusticolErrorKind::Compatibility
        | RusticolErrorKind::UnsupportedPrecision
        | RusticolErrorKind::UnsupportedRuntimeCapability => CompatibilityError::new_err(message),
        RusticolErrorKind::Selector => SelectorError::new_err(message),
        RusticolErrorKind::ModelParameter => ModelParameterError::new_err(message),
        RusticolErrorKind::Evaluation | RusticolErrorKind::Internal => {
            EvaluationError::new_err(message)
        }
        _ => EvaluationError::new_err(message),
    }
}

fn require_raw_f64_precision(precision: u32) -> Result<(), CoreError> {
    if precision == 16 {
        return Ok(());
    }
    Err(CoreError::compatibility(format!(
        "raw pyamplicol._rusticol supports only precision=16 for Symbolica-free direct SymJIT evaluation; precision={precision} must be routed through the public Python Symbolica executor"
    )))
}

fn report_on_the_fly_warm_up_progress(
    callback: &Py<PyAny>,
    callback_error: &Mutex<Option<PyErr>>,
    event: &NativeOnTheFlyWarmUpEvent,
) -> Result<bool, CoreError> {
    let payload = serde_json::to_string(event).map_err(|error| {
        CoreError::serialization(format!(
            "could not serialize on-the-fly warm-up progress event: {error}"
        ))
    })?;
    let terminal_notification = event.kind == NativeOnTheFlyWarmUpEventKind::End
        && event.stage == NativeOnTheFlyWarmUpStage::FirstEvaluation;
    Python::attach(|py| {
        let decision = callback.bind(py).call1((payload,));
        let decision = match decision {
            Ok(decision) => decision,
            Err(error) if terminal_notification => {
                error.write_unraisable(py, Some(callback.bind(py)));
                return Ok(true);
            }
            Err(error) => {
                let message = error.to_string();
                let mut slot = callback_error.lock().map_err(|_| {
                    CoreError::internal("on-the-fly warm-up callback error lock was poisoned")
                })?;
                *slot = Some(error);
                return Err(CoreError::evaluation(format!(
                    "on-the-fly warm-up progress callback failed: {message}"
                )));
            }
        };
        if decision.is_none() {
            return Ok(true);
        }
        if !decision.is_instance_of::<PyBool>() {
            let error = PyTypeError::new_err(
                "on-the-fly warm-up progress callback must return bool or None",
            );
            if terminal_notification {
                error.write_unraisable(py, Some(callback.bind(py)));
                return Ok(true);
            }
            let mut slot = callback_error.lock().map_err(|_| {
                CoreError::internal("on-the-fly warm-up callback error lock was poisoned")
            })?;
            *slot = Some(error);
            return Err(CoreError::invalid_argument(
                "on-the-fly warm-up progress callback must return bool or None",
            ));
        }
        decision.extract::<bool>().map_err(|error| {
            CoreError::invalid_argument(format!(
                "could not read on-the-fly warm-up progress decision: {error}"
            ))
        })
    })
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ExternalParticle {
    index: usize,
    label: usize,
    name: String,
    pdg_id: i32,
    state: String,
    momentum_slot: usize,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct HelicityConfiguration {
    id: String,
    index: usize,
    values: Vec<i32>,
    computed: bool,
    structural_zero: bool,
    representative_id: String,
    coefficient: f64,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ColorFlow {
    id: String,
    index: usize,
    word: Vec<usize>,
    computed: bool,
    representative_id: String,
    coefficient: f64,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ContractedColorComponent {
    id: String,
    index: usize,
    description: String,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ModelParameter {
    name: String,
    kind: String,
    default_real: f64,
    default_imaginary: f64,
    mutable: bool,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ReductionGroup {
    id: String,
    representative_helicity_id: String,
    representative_color_id: String,
    physical_helicity_ids: Vec<String>,
    physical_color_ids: Vec<String>,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct PhysicsReduction {
    kind: String,
    groups: Vec<ReductionGroup>,
}

impl From<&CoreModelParameter> for ModelParameter {
    fn from(parameter: &CoreModelParameter) -> Self {
        Self {
            name: parameter.name.clone(),
            kind: parameter_kind_name(parameter.kind).to_string(),
            default_real: parameter.default_real,
            default_imaginary: parameter.default_imaginary,
            mutable: parameter.mutable,
        }
    }
}

#[pyclass(module = "pyamplicol._rusticol", frozen, skip_from_py_object)]
#[derive(Clone)]
struct ProcessPhysics {
    process_id: String,
    process: String,
    color_accuracy: String,
    helicity_coverage: String,
    color_coverage: String,
    color_kind: String,
    structural_zero_helicity_count: usize,
    external_particles: Vec<ExternalParticle>,
    helicities: Vec<HelicityConfiguration>,
    color_flows: Vec<ColorFlow>,
    contracted_color_components: Vec<ContractedColorComponent>,
    reduction: PhysicsReduction,
    model_parameters: Vec<ModelParameter>,
    selector_capabilities: Vec<String>,
}

impl ProcessPhysics {
    fn from_core(physics: &CoreProcessPhysics) -> Self {
        let external_particles = physics
            .external_particles
            .iter()
            .map(|particle| ExternalParticle {
                index: particle.index,
                label: particle.label,
                name: particle.particle.clone(),
                pdg_id: particle.pdg,
                state: match particle.role {
                    ParticleRole::Initial => "incoming",
                    ParticleRole::Final => "outgoing",
                }
                .to_string(),
                momentum_slot: particle.momentum_slot,
            })
            .collect();
        let helicities = physics
            .helicities
            .iter()
            .map(|helicity| HelicityConfiguration {
                id: helicity.id.clone(),
                index: helicity.index,
                values: helicity.values.clone(),
                computed: helicity.computed,
                structural_zero: helicity.structural_zero,
                representative_id: helicity.representative_id.clone(),
                coefficient: helicity.coefficient,
            })
            .collect();
        let color_flows = physics
            .color_components
            .iter()
            .filter_map(|component| match component {
                CoreColorComponent::LcFlow(flow) => Some(ColorFlow {
                    id: flow.id.clone(),
                    index: flow.index,
                    word: flow.word.clone(),
                    computed: flow.computed,
                    representative_id: flow.representative_id.clone(),
                    coefficient: flow.coefficient,
                }),
                CoreColorComponent::ContractedColor(_) => None,
            })
            .collect();
        let contracted_color_components = physics
            .color_components
            .iter()
            .filter_map(|component| match component {
                CoreColorComponent::ContractedColor(color) => Some(ContractedColorComponent {
                    id: color.id.clone(),
                    index: color.index,
                    description: color.description.clone(),
                }),
                CoreColorComponent::LcFlow(_) => None,
            })
            .collect();
        let mut selector_capabilities = Vec::new();
        if physics.selectors.helicity {
            selector_capabilities.push("helicity".to_string());
        }
        if physics.selectors.color_flow {
            selector_capabilities.push("color_flow".to_string());
        }
        if physics.selectors.contracted_color {
            selector_capabilities.push("contracted_color".to_string());
        }
        let reduction = PhysicsReduction {
            kind: reduction_kind_name(physics.reduction.kind).to_string(),
            groups: physics
                .reduction
                .groups
                .iter()
                .map(|group| ReductionGroup {
                    id: group.id.clone(),
                    representative_helicity_id: group.representative_helicity_id.clone(),
                    representative_color_id: group.representative_color_id.clone(),
                    physical_helicity_ids: group.physical_helicity_ids.clone(),
                    physical_color_ids: group.physical_color_ids.clone(),
                })
                .collect(),
        };
        Self {
            process_id: physics.process_id.clone(),
            process: physics.process.clone(),
            color_accuracy: color_accuracy_name(physics.color_accuracy).to_string(),
            helicity_coverage: physics.coverage.helicities.clone(),
            color_coverage: physics.coverage.color.clone(),
            color_kind: physics.coverage.color_kind.clone(),
            structural_zero_helicity_count: physics.coverage.structural_zero_helicity_count,
            external_particles,
            helicities,
            color_flows,
            contracted_color_components,
            reduction,
            model_parameters: physics.model_parameters.iter().map(Into::into).collect(),
            selector_capabilities,
        }
    }
}

#[pymethods]
impl ProcessPhysics {
    #[getter]
    fn process_id(&self) -> &str {
        &self.process_id
    }

    #[getter]
    fn process(&self) -> &str {
        &self.process
    }

    #[getter]
    fn color_accuracy(&self) -> &str {
        &self.color_accuracy
    }

    #[getter]
    fn helicity_coverage(&self) -> &str {
        &self.helicity_coverage
    }

    #[getter]
    fn color_coverage(&self) -> &str {
        &self.color_coverage
    }

    #[getter]
    fn color_kind(&self) -> &str {
        &self.color_kind
    }

    #[getter]
    fn structural_zero_helicity_count(&self) -> usize {
        self.structural_zero_helicity_count
    }

    #[getter]
    fn external_particles(&self) -> Vec<ExternalParticle> {
        self.external_particles.clone()
    }

    #[getter]
    fn helicities(&self) -> Vec<HelicityConfiguration> {
        self.helicities.clone()
    }

    #[getter]
    fn color_flows(&self) -> Vec<ColorFlow> {
        self.color_flows.clone()
    }

    #[getter]
    fn contracted_color_components(&self) -> Vec<ContractedColorComponent> {
        self.contracted_color_components.clone()
    }

    #[getter]
    fn reduction(&self) -> PhysicsReduction {
        self.reduction.clone()
    }

    #[getter]
    fn model_parameters(&self) -> Vec<ModelParameter> {
        self.model_parameters.clone()
    }

    #[getter]
    fn selector_capabilities(&self) -> Vec<String> {
        self.selector_capabilities.clone()
    }
}

#[pyclass(module = "pyamplicol._rusticol", unsendable)]
struct ResolvedEvaluation {
    values: Py<PyAny>,
    totals: Py<PyAny>,
    shape: (usize, usize, usize),
    helicity_ids: Vec<String>,
    color_ids: Vec<String>,
    color_accuracy: String,
    precision: u32,
}

#[pymethods]
impl ResolvedEvaluation {
    #[getter]
    fn values(&self, py: Python<'_>) -> Py<PyAny> {
        self.values.clone_ref(py)
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    #[getter]
    fn helicity_ids(&self) -> Vec<String> {
        self.helicity_ids.clone()
    }

    #[getter]
    fn color_ids(&self) -> Vec<String> {
        self.color_ids.clone()
    }

    #[getter]
    fn color_flow_ids(&self) -> Vec<String> {
        if self.color_accuracy == "lc" {
            self.color_ids.clone()
        } else {
            Vec::new()
        }
    }

    #[getter]
    fn color_accuracy(&self) -> &str {
        &self.color_accuracy
    }

    #[getter]
    fn precision(&self) -> u32 {
        self.precision
    }

    fn total(&self, py: Python<'_>) -> Py<PyAny> {
        self.totals.clone_ref(py)
    }
}

#[pyclass(module = "pyamplicol._rusticol", unsendable)]
struct Runtime {
    runtime: NativeRuntime,
}

/// An exclusive `NativeRuntime` borrow which stable PyO3 can carry through
/// [`Python::detach`].
///
/// Stable PyO3 conservatively uses `Send` as its `Ungil` bound even though
/// `detach` invokes its closure synchronously on the calling thread. The
/// native runtime is deliberately `!Send`, so this adapter retains its Rust
/// borrow for the whole detached call and permits access only on the thread
/// which created it.
struct SameThreadNativeRuntimeBorrow<'a> {
    runtime: NonNull<NativeRuntime>,
    origin: ThreadId,
    _exclusive: PhantomData<&'a mut NativeRuntime>,
}

fn require_origin_thread(origin: ThreadId) -> Result<(), CoreError> {
    if thread::current().id() == origin {
        return Ok(());
    }
    Err(CoreError::internal(
        "detached NativeRuntime operation moved away from its owning thread",
    ))
}

impl<'a> SameThreadNativeRuntimeBorrow<'a> {
    fn new(runtime: &'a mut NativeRuntime) -> Self {
        Self {
            runtime: NonNull::from(runtime),
            origin: thread::current().id(),
            _exclusive: PhantomData,
        }
    }

    fn run<R>(mut self, operation: impl FnOnce(&mut NativeRuntime) -> R) -> Result<R, CoreError> {
        require_origin_thread(self.origin)?;
        // SAFETY: `new` receives an exclusive borrow whose lifetime is retained
        // by `_exclusive`. This consuming method can create only one mutable
        // reborrow, and the thread check above prevents dereferencing the
        // deliberately thread-affine runtime from any other thread.
        Ok(operation(unsafe { self.runtime.as_mut() }))
    }
}

// SAFETY: transferring the adapter cannot transfer, move, drop, or access the
// borrowed runtime. `run` is its only accessor and checks the creating thread
// before dereferencing; dropping the adapter never touches the pointee. The
// lifetime marker continues to enforce exclusive access on the creating stack.
unsafe impl Send for SameThreadNativeRuntimeBorrow<'_> {}

// These signatures intentionally mirror the keyword-rich public Python ABI.
#[allow(clippy::too_many_arguments)]
#[pymethods]
impl Runtime {
    #[new]
    #[pyo3(signature=(artifact, *, process=None, model_parameters=None, mute_warnings=false))]
    fn new(
        artifact: &Bound<'_, PyAny>,
        process: Option<&str>,
        model_parameters: Option<&Bound<'_, PyAny>>,
        mute_warnings: bool,
    ) -> PyResult<Self> {
        let path = path_from_python(artifact)?;
        let mut runtime = NativeRuntime::load(path, process, None).map_err(python_error)?;
        if let Some(mapping) = model_parameters {
            let values = parse_model_parameters(mapping)?;
            runtime
                .set_model_parameters(&values)
                .map_err(python_error)?;
        }
        if mute_warnings {
            runtime.mute_warnings();
        }
        Ok(Self { runtime })
    }

    #[staticmethod]
    #[pyo3(signature=(artifact, *, process=None, model_parameters=None, mute_warnings=false))]
    fn load(
        artifact: &Bound<'_, PyAny>,
        process: Option<&str>,
        model_parameters: Option<&Bound<'_, PyAny>>,
        mute_warnings: bool,
    ) -> PyResult<Self> {
        Self::new(artifact, process, model_parameters, mute_warnings)
    }

    #[getter]
    fn process(&self) -> String {
        self.runtime.metadata().process
    }

    #[getter]
    fn artifact_id(&self) -> &str {
        self.runtime.artifact_id()
    }

    #[getter]
    fn process_id(&self) -> String {
        self.runtime.metadata().process_key
    }

    #[getter]
    fn color_accuracy(&self) -> String {
        self.runtime.metadata().color_accuracy
    }

    #[getter]
    fn physics(&self) -> PyResult<ProcessPhysics> {
        Ok(ProcessPhysics::from_core(
            self.runtime.process_physics().map_err(python_error)?,
        ))
    }

    fn metadata_json(&self) -> PyResult<String> {
        self.runtime.metadata_json().map_err(python_error)
    }

    fn save(&mut self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = path_from_python(path)?;
        let runtime = SameThreadNativeRuntimeBorrow::new(&mut self.runtime);
        py.detach(move || runtime.run(|runtime| runtime.save(path)))
            .and_then(|result| result)
            .map_err(python_error)
    }

    fn load_cache(&mut self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = path_from_python(path)?;
        let runtime = SameThreadNativeRuntimeBorrow::new(&mut self.runtime);
        py.detach(move || runtime.run(|runtime| runtime.load_cache(path)))
            .and_then(|result| result)
            .map_err(python_error)
    }

    #[pyo3(signature=(color_flow_ids=None))]
    fn _on_the_fly_benchmark_context_json(
        &self,
        color_flow_ids: Option<Vec<String>>,
    ) -> PyResult<Option<String>> {
        self.runtime
            .on_the_fly_benchmark_context_json(color_flow_ids.as_deref())
            .map_err(python_error)
    }

    #[pyo3(signature=(helicity_ids=None, color_flow_ids=None))]
    fn _on_the_fly_selector_ordinals_json(
        &self,
        helicity_ids: Option<Vec<String>>,
        color_flow_ids: Option<Vec<String>>,
    ) -> PyResult<Option<String>> {
        self.runtime
            .on_the_fly_selector_ordinals_json(helicity_ids.as_deref(), color_flow_ids.as_deref())
            .map_err(python_error)
    }

    /// Private read-only production introspection of retained compact OTF state.
    fn _on_the_fly_runtime_state_census_json(&self) -> PyResult<Option<String>> {
        self.runtime
            .on_the_fly_runtime_state_census_json()
            .map_err(python_error)
    }

    #[pyo3(signature=(momenta, helicity_ids=None, color_flow_ids=None, progress_callback=None, *, n_cores=None))]
    fn _on_the_fly_warm_up_f64_json(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicity_ids: Option<Vec<String>>,
        color_flow_ids: Option<Vec<String>>,
        progress_callback: Option<Py<PyAny>>,
        n_cores: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let n_cores = n_cores
            .map(|value| {
                if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
                    return Err(PyTypeError::new_err(
                        "n_cores must be a positive integer or None",
                    ));
                }
                match value.extract::<usize>() {
                    Ok(count) if count > 0 => Ok(count),
                    _ => Err(PyValueError::new_err(
                        "n_cores must be a positive integer fitting the platform's usize",
                    )),
                }
            })
            .transpose()?;
        let momenta = parse_f64_momenta(momenta, self.runtime.external_count())?;
        if momenta.point_count() != 1 {
            return Err(PyValueError::new_err(format!(
                "on-the-fly warm_up requires exactly one phase-space point, received {}",
                momenta.point_count()
            )));
        }
        let momenta = momenta.as_slice().to_vec();
        if let Some(callback) = progress_callback.as_ref()
            && !callback.bind(py).is_callable()
        {
            return Err(PyTypeError::new_err(
                "on-the-fly warm_up progress_callback must be callable or None",
            ));
        }
        let callback_error = Arc::new(Mutex::new(None));
        let detached_callback_error = Arc::clone(&callback_error);
        let detached_runtime = SameThreadNativeRuntimeBorrow::new(&mut self.runtime);
        let result = py
            .detach(move || {
                detached_runtime.run(|runtime| {
                    if let Some(callback) = progress_callback.as_ref() {
                        let mut observer = |event: &NativeOnTheFlyWarmUpEvent| {
                            report_on_the_fly_warm_up_progress(
                                callback,
                                detached_callback_error.as_ref(),
                                event,
                            )
                        };
                        let observer: &mut NativeOnTheFlyWarmUpObserver<'_> = &mut observer;
                        runtime.warm_up_on_the_fly_f64_with_selectors(
                            &momenta,
                            helicity_ids.as_deref(),
                            color_flow_ids.as_deref(),
                            n_cores,
                            Some(observer),
                        )
                    } else {
                        runtime.warm_up_on_the_fly_f64_with_selectors(
                            &momenta,
                            helicity_ids.as_deref(),
                            color_flow_ids.as_deref(),
                            n_cores,
                            None,
                        )
                    }
                })
            })
            .and_then(|result| result);
        if let Some(error) = callback_error
            .lock()
            .map_err(|_| {
                PyException::new_err("on-the-fly warm-up callback error lock was poisoned")
            })?
            .take()
        {
            return Err(error);
        }
        let result = result.map_err(python_error)?;
        serde_json::to_string(&result).map_err(|error| {
            python_error(CoreError::serialization(format!(
                "could not serialize on-the-fly warm-up result: {error}"
            )))
        })
    }

    fn physics_json(&self) -> PyResult<String> {
        self.runtime.physics_json().map_err(python_error)
    }

    fn _exact_runtime_state_json(&self) -> PyResult<String> {
        self.runtime
            .exact_runtime_state_json()
            .map_err(python_error)
    }

    #[pyo3(signature=(momenta, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16))]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
    ) -> PyResult<Py<PyAny>> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let momenta = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let point_count = momenta.point_count();
        let mut values = vec![0.0; point_count];
        self.runtime
            .evaluate_f64_into_with_selectors(
                momenta.as_slice(),
                point_count,
                helicities.as_deref(),
                color_flows.as_deref(),
                helicity_by_point.as_deref(),
                color_flow_by_point.as_deref(),
                &mut values,
            )
            .map_err(python_error)?;
        let result = PyList::new(py, values)?.into_any().unbind();
        self.emit_warnings(py)?;
        Ok(result)
    }

    #[pyo3(signature=(momenta, repetitions, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16))]
    fn _benchmark_f64_wall_time(
        &mut self,
        momenta: &Bound<'_, PyAny>,
        repetitions: usize,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
    ) -> PyResult<f64> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let momenta = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let point_count = momenta.point_count();
        self.runtime
            .benchmark_f64_wall_time_with_selectors(
                momenta.as_slice(),
                point_count,
                repetitions,
                helicities.as_deref(),
                color_flows.as_deref(),
                helicity_by_point.as_deref(),
                color_flow_by_point.as_deref(),
            )
            .map_err(python_error)
    }

    #[pyo3(signature=(momenta, repetitions, *, helicities=None, color_flows=None, precision=16, include_values=false))]
    fn _profile_arena_repeated(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        repetitions: usize,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        precision: u32,
        include_values: bool,
    ) -> PyResult<Py<PyAny>> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let momenta = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let point_count = momenta.point_count();
        let profiled = self
            .runtime
            .evaluate_f64_arena_profile_repeated(
                momenta.as_slice(),
                point_count,
                repetitions,
                helicities.as_deref(),
                color_flows.as_deref(),
            )
            .map_err(python_error)?;
        let measured_points = point_count
            .checked_mul(repetitions)
            .ok_or_else(|| PyValueError::new_err("arena profile point count overflowed"))?;
        let result = runtime_profile_to_python(py, &profiled.profile, measured_points)?;
        result.set_item("execution_mode", self.runtime.metadata().execution_mode)?;
        result.set_item(
            "profile_boundary",
            "warmed-direct-arena-borrowed-input-preallocated-output-v1",
        )?;
        result.set_item("borrowed_flat_input", true)?;
        result.set_item("preallocated_output", true)?;
        result.set_item("phase_timing_scope", "coarse-arena-boundary-only-v1")?;
        result.set_item("evaluator_timing_available", false)?;
        if include_values {
            result.set_item("values", PyList::new(py, profiled.values)?)?;
        }
        self.emit_warnings(py)?;
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature=(momenta, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16, include_values=false))]
    fn profile(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
        include_values: bool,
    ) -> PyResult<Py<PyAny>> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let momenta = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let point_count = momenta.point_count();
        let profiled = self
            .runtime
            .evaluate_f64_profile_with_selectors(
                momenta.as_slice(),
                point_count,
                helicities.as_deref(),
                color_flows.as_deref(),
                helicity_by_point.as_deref(),
                color_flow_by_point.as_deref(),
            )
            .map_err(python_error)?;
        let result = runtime_profile_to_python(py, &profiled.profile, point_count)?;
        result.set_item("execution_mode", self.runtime.metadata().execution_mode)?;
        if include_values {
            result.set_item("values", PyList::new(py, profiled.values)?)?;
        }
        self.emit_warnings(py)?;
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature=(momenta, repetitions, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16, include_values=false))]
    fn profile_repeated(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        repetitions: usize,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
        include_values: bool,
    ) -> PyResult<Py<PyAny>> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let momenta = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let point_count = momenta.point_count();
        let profiled = self
            .runtime
            .evaluate_f64_profile_repeated_with_selectors(
                momenta.as_slice(),
                point_count,
                repetitions,
                helicities.as_deref(),
                color_flows.as_deref(),
                helicity_by_point.as_deref(),
                color_flow_by_point.as_deref(),
            )
            .map_err(python_error)?;
        let measured_points = point_count
            .checked_mul(repetitions)
            .ok_or_else(|| PyValueError::new_err("profile point count overflowed"))?;
        let result = runtime_profile_to_python(py, &profiled.profile, measured_points)?;
        result.set_item("execution_mode", self.runtime.metadata().execution_mode)?;
        if include_values {
            result.set_item("values", PyList::new(py, profiled.values)?)?;
        }
        self.emit_warnings(py)?;
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature=(momenta, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16, include_values=false))]
    fn evaluate_profile(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
        include_values: bool,
    ) -> PyResult<Py<PyAny>> {
        self.profile(
            py,
            momenta,
            helicities,
            color_flows,
            helicity_by_point,
            color_flow_by_point,
            precision,
            include_values,
        )
    }

    #[pyo3(signature=(momenta, *, helicities=None, color_flows=None, precision=16))]
    fn evaluate_resolved(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        precision: u32,
    ) -> PyResult<ResolvedEvaluation> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let accuracy = self.runtime.metadata().color_accuracy;
        let momenta = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let point_count = momenta.point_count();
        let resolved = self
            .runtime
            .evaluate_resolved_f64(
                momenta.as_slice(),
                point_count,
                helicities.as_deref(),
                color_flows.as_deref(),
            )
            .map_err(python_error)?;
        let result = resolved_f64_to_python(py, resolved, accuracy, precision)?;
        self.emit_warnings(py)?;
        Ok(result)
    }

    fn evaluate_with_prec(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        decimal_digit_precision: u32,
    ) -> PyResult<Py<PyAny>> {
        self.evaluate(py, momenta, None, None, None, None, decimal_digit_precision)
    }

    #[pyo3(signature=(momenta, decimal_digit_precision, helicities=None, color_flows=None))]
    fn evaluate_resolved_with_prec(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        decimal_digit_precision: u32,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
    ) -> PyResult<ResolvedEvaluation> {
        self.evaluate_resolved(
            py,
            momenta,
            helicities,
            color_flows,
            decimal_digit_precision,
        )
    }

    fn set_model_parameters(&mut self, mapping: &Bound<'_, PyAny>) -> PyResult<()> {
        let values = parse_model_parameters(mapping)?;
        self.runtime
            .set_model_parameters(&values)
            .map_err(python_error)
    }

    fn clear(&mut self) -> PyResult<()> {
        self.runtime.clear().map_err(python_error)
    }

    fn set_model_parameter(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = parse_complex_value(value, name)?;
        self.runtime
            .set_model_parameter(name, value.0, value.1)
            .map_err(python_error)
    }

    fn mute_warnings(&mut self) {
        self.runtime.mute_warnings();
    }

    fn unmute_warnings(&mut self) {
        self.runtime.unmute_warnings();
    }

    fn take_warnings(&mut self) -> Vec<String> {
        self.runtime.take_warnings()
    }
}

impl Runtime {
    fn emit_warnings(&mut self, py: Python<'_>) -> PyResult<()> {
        let pending = self.runtime.take_warnings();
        if pending.is_empty() {
            return Ok(());
        }
        let warnings = py.import("warnings")?;
        for message in pending {
            warnings.call_method1("warn", (message,))?;
        }
        Ok(())
    }
}

fn color_accuracy_name(value: ColorAccuracy) -> &'static str {
    value.as_str()
}

fn reduction_kind_name(value: ReductionKind) -> &'static str {
    match value {
        ReductionKind::LcDiagonal => "lc-diagonal",
        ReductionKind::ContractedColor => "contracted-color",
    }
}

fn parameter_kind_name(value: ParameterKind) -> &'static str {
    match value {
        ParameterKind::Normalization => "normalization",
        ParameterKind::Mass => "mass",
        ParameterKind::Width => "width",
        ParameterKind::Coupling => "coupling",
        ParameterKind::External => "external",
        ParameterKind::Derived => "derived",
    }
}

fn path_from_python(value: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    value.extract::<PathBuf>().map_err(|error| {
        PyTypeError::new_err(format!(
            "artifact must be a string or path-like object: {error}"
        ))
    })
}

fn parse_model_parameters(mapping: &Bound<'_, PyAny>) -> PyResult<BTreeMap<String, (f64, f64)>> {
    let items = mapping.call_method0("items").map_err(|_| {
        PyTypeError::new_err("model_parameters must be a mapping from names to numbers")
    })?;
    let mut result = BTreeMap::new();
    for item in items.try_iter()? {
        let item = item?;
        let name = item.get_item(0)?.extract::<String>()?;
        if name.is_empty() {
            return Err(PyValueError::new_err(
                "model parameter names must not be empty",
            ));
        }
        let value = item.get_item(1)?;
        if result
            .insert(name.clone(), parse_complex_value(&value, &name)?)
            .is_some()
        {
            return Err(PyValueError::new_err(format!(
                "duplicate model parameter {name:?}"
            )));
        }
    }
    Ok(result)
}

fn parse_complex_value(value: &Bound<'_, PyAny>, name: &str) -> PyResult<(f64, f64)> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "model parameter {name:?} must be numeric, not bool"
        )));
    }
    if let Ok(real) = value.extract::<f64>() {
        return Ok((real, 0.0));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        if value.len()? != 2 {
            return Err(PyValueError::new_err(format!(
                "model parameter {name:?} array must contain [real, imaginary]"
            )));
        }
        let component = |index: usize, label: &str| -> PyResult<f64> {
            let item = value.get_item(index)?;
            if item.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(format!(
                    "model parameter {name:?} {label} component must be numeric, not bool"
                )));
            }
            item.extract::<f64>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "model parameter {name:?} {label} component must be numeric"
                ))
            })
        };
        return Ok((component(0, "real")?, component(1, "imaginary")?));
    }
    let real = value.getattr("real")?.extract::<f64>().map_err(|_| {
        PyTypeError::new_err(format!(
            "model parameter {name:?} must be a number or [real, imaginary]"
        ))
    })?;
    let imaginary = value.getattr("imag")?.extract::<f64>().map_err(|_| {
        PyTypeError::new_err(format!(
            "model parameter {name:?} must be a number or [real, imaginary]"
        ))
    })?;
    Ok((real, imaginary))
}

enum ParsedF64Momenta<'py> {
    #[cfg(feature = "numpy")]
    Borrowed {
        array: PyReadonlyArray3<'py, f64>,
        point_count: usize,
    },
    Owned {
        values: Vec<f64>,
        point_count: usize,
        _python: PhantomData<&'py ()>,
    },
}

impl ParsedF64Momenta<'_> {
    fn as_slice(&self) -> &[f64] {
        match self {
            #[cfg(feature = "numpy")]
            Self::Borrowed { array, .. } => array
                .as_slice()
                .expect("contiguous NumPy momenta were validated while parsing"),
            Self::Owned { values, .. } => values,
        }
    }

    fn point_count(&self) -> usize {
        match self {
            #[cfg(feature = "numpy")]
            Self::Borrowed { point_count, .. } => *point_count,
            Self::Owned { point_count, .. } => *point_count,
        }
    }
}

fn parse_f64_momenta<'py>(
    momenta: &Bound<'py, PyAny>,
    expected_legs: usize,
) -> PyResult<ParsedF64Momenta<'py>> {
    #[cfg(feature = "numpy")]
    if let Ok(array) = momenta.extract::<PyReadonlyArray3<'py, f64>>() {
        let shape = array.shape();
        let point_count = shape[0];
        if shape[1] != expected_legs || shape[2] != 4 {
            return Err(PyValueError::new_err(format!(
                "momenta array has shape ({}, {}, {}), expected (points, {expected_legs}, 4)",
                point_count, shape[1], shape[2]
            )));
        }
        if point_count == 0 {
            return Err(PyValueError::new_err(
                "momenta must contain at least one point",
            ));
        }
        if array.as_slice().is_ok() {
            return Ok(ParsedF64Momenta::Borrowed { point_count, array });
        }
    }

    let mut values = Vec::new();
    let mut point_count = 0;
    for (point_index, point) in momenta.try_iter()?.enumerate() {
        let point = point?;
        let mut leg_count = 0;
        for (leg_index, leg) in point.try_iter()?.enumerate() {
            let leg = leg?;
            let components = leg
                .try_iter()?
                .map(|component| component?.extract::<f64>())
                .collect::<PyResult<Vec<_>>>()?;
            if components.len() != 4 {
                return Err(PyValueError::new_err(format!(
                    "momenta point {point_index} leg {leg_index} has {} components, expected 4",
                    components.len()
                )));
            }
            values.extend(components);
            leg_count += 1;
        }
        if leg_count != expected_legs {
            return Err(PyValueError::new_err(format!(
                "momenta point {point_index} has {leg_count} external legs, expected {expected_legs}"
            )));
        }
        point_count += 1;
    }
    if point_count == 0 {
        return Err(PyValueError::new_err(
            "momenta must contain at least one point",
        ));
    }
    Ok(ParsedF64Momenta::Owned {
        values,
        point_count,
        _python: PhantomData,
    })
}

fn resolved_f64_to_python(
    py: Python<'_>,
    resolved: NativeResolvedEvaluation,
    color_accuracy: String,
    precision: u32,
) -> PyResult<ResolvedEvaluation> {
    let shape = resolved.shape();
    let values = nested_f64_values(py, &resolved.values, shape.1, shape.2)?;
    let totals = PyList::new(py, resolved.totals())?.into_any().unbind();
    Ok(ResolvedEvaluation {
        values,
        totals,
        shape,
        helicity_ids: resolved.helicity_ids,
        color_ids: resolved.color_ids,
        color_accuracy,
        precision,
    })
}

fn nested_f64_values(
    py: Python<'_>,
    values: &[f64],
    helicity_count: usize,
    color_count: usize,
) -> PyResult<Py<PyAny>> {
    let point_width = helicity_count
        .checked_mul(color_count)
        .ok_or_else(|| EvaluationError::new_err("resolved shape overflow"))?;
    if point_width == 0 || !values.len().is_multiple_of(point_width) {
        return Err(EvaluationError::new_err(
            "resolved value buffer does not match its shape",
        ));
    }
    let mut points = Vec::with_capacity(values.len() / point_width);
    for point in values.chunks(point_width) {
        let mut helicities = Vec::with_capacity(helicity_count);
        for colors in point.chunks(color_count) {
            helicities.push(PyList::new(py, colors.iter().copied())?.into_any().unbind());
        }
        points.push(PyList::new(py, helicities)?.into_any().unbind());
    }
    Ok(PyList::new(py, points)?.into_any().unbind())
}

fn runtime_profile_to_python<'py>(
    py: Python<'py>,
    profile: &NativeRuntimeProfile,
    point_count: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("points", point_count)?;
    payload.set_item("wall_time_s", profile.total_s)?;
    payload.set_item("native_input_pack_time_s", profile.native_input_pack_s)?;
    payload.set_item(
        "native_input_crossing_time_s",
        profile.native_input_crossing_s,
    )?;
    payload.set_item("orchestration_time_s", profile.orchestration_s)?;
    payload.set_item("state_prepare_time_s", profile.state_prepare_s)?;
    payload.set_item("state_clear_time_s", profile.state_clear_s)?;
    payload.set_item("source_fill_time_s", profile.source_fill_s)?;
    payload.set_item(
        "momentum_input_setup_time_s",
        profile.momentum_input_setup_s,
    )?;
    payload.set_item("momentum_setup_time_s", profile.momentum_setup_s)?;
    payload.set_item(
        "model_parameter_setup_time_s",
        profile.model_parameter_setup_s,
    )?;
    payload.set_item("stage_input_pack_time_s", profile.stage_input_pack_s)?;
    payload.set_item(
        "stage_leaf_input_pack_time_s",
        profile.stage_leaf_input_pack_s,
    )?;
    payload.set_item(
        "stage_evaluator_call_time_s",
        profile.stage_evaluator_call_s,
    )?;
    payload.set_item("stage_evaluator_time_s", profile.stage_evaluator_s)?;
    payload.set_item("stage_backend_call_time_s", profile.stage_backend_call_s)?;
    payload.set_item(
        "stage_evaluator_output_gather_time_s",
        profile.stage_evaluator_output_gather_s,
    )?;
    payload.set_item("output_assign_time_s", profile.output_assign_s)?;
    payload.set_item(
        "amplitude_input_pack_time_s",
        profile.amplitude_input_pack_s,
    )?;
    payload.set_item(
        "amplitude_evaluator_call_time_s",
        profile.amplitude_evaluator_call_s,
    )?;
    payload.set_item(
        "amplitude_leaf_input_pack_time_s",
        profile.amplitude_leaf_input_pack_s,
    )?;
    payload.set_item(
        "amplitude_backend_call_time_s",
        profile.amplitude_backend_call_s,
    )?;
    payload.set_item(
        "amplitude_evaluator_output_gather_time_s",
        profile.amplitude_evaluator_output_gather_s,
    )?;
    payload.set_item(
        "amplitude_output_remap_time_s",
        profile.amplitude_output_remap_s,
    )?;
    payload.set_item("amplitude_evaluator_time_s", profile.amplitude_evaluator_s)?;
    payload.set_item("reduction_time_s", profile.reduction_s)?;
    payload.set_item(
        "resolved_reduction_materialization_inclusive_time_s",
        profile.resolved_reduction_materialization_s,
    )?;
    payload.set_item(
        "total_materialization_time_s",
        profile.total_materialization_s,
    )?;
    payload.set_item("final_output_copy_time_s", profile.final_output_copy_s)?;
    payload.set_item(
        "stage_input_pack_by_stage_time_s",
        profile.stage_input_pack_by_stage_s.clone(),
    )?;
    payload.set_item(
        "stage_evaluator_call_by_stage_time_s",
        profile.stage_evaluator_call_by_stage_s.clone(),
    )?;
    payload.set_item(
        "stage_leaf_input_pack_by_stage_time_s",
        profile.stage_leaf_input_pack_by_stage_s.clone(),
    )?;
    payload.set_item(
        "stage_backend_call_by_stage_time_s",
        profile.stage_backend_call_by_stage_s.clone(),
    )?;
    payload.set_item(
        "stage_evaluator_output_gather_by_stage_time_s",
        profile.stage_evaluator_output_gather_by_stage_s.clone(),
    )?;
    payload.set_item(
        "stage_output_assign_by_stage_time_s",
        profile.stage_output_assign_by_stage_s.clone(),
    )?;
    payload.set_item("eager_initialize_time_s", profile.eager_initialize_s)?;
    payload.set_item("eager_gather_time_s", profile.eager_gather_s)?;
    payload.set_item("eager_kernel_call_time_s", profile.eager_kernel_call_s)?;
    payload.set_item(
        "eager_invocation_scatter_time_s",
        profile.eager_invocation_scatter_s,
    )?;
    payload.set_item("eager_finalization_time_s", profile.eager_finalization_s)?;
    payload.set_item(
        "eager_scatter_finalization_time_s",
        profile.eager_scatter_finalization_s,
    )?;
    payload.set_item("eager_closure_time_s", profile.eager_closure_s)?;
    payload.set_item("eager_reduction_time_s", profile.eager_reduction_s)?;
    payload.set_item("eager_copy_out_time_s", profile.eager_copy_out_s)?;
    payload.set_item(
        "recurrence_momentum_fill_time_s",
        profile.recurrence_momentum_fill_s,
    )?;
    payload.set_item(
        "recurrence_union_source_fill_time_s",
        profile.recurrence_union_source_fill_s,
    )?;
    payload.set_item("recurrence_schedule_time_s", profile.recurrence_schedule_s)?;
    payload.set_item(
        "recurrence_source_kernel_time_s",
        profile.recurrence_source_kernel_s,
    )?;
    payload.set_item(
        "recurrence_contribution_kernel_time_s",
        profile.recurrence_contribution_kernel_s,
    )?;
    payload.set_item(
        "recurrence_finalization_time_s",
        profile.recurrence_finalization_s,
    )?;
    payload.set_item("recurrence_closure_time_s", profile.recurrence_closure_s)?;
    payload.set_item(
        "recurrence_replay_output_mapping_time_s",
        profile.recurrence_replay_output_mapping_s,
    )?;
    payload.set_item(
        "recurrence_momentum_scalar_value_count",
        profile.recurrence_momentum_scalar_value_count,
    )?;
    payload.set_item(
        "recurrence_schedule_execution_count",
        profile.recurrence_schedule_execution_count,
    )?;
    payload.set_item(
        "recurrence_replay_schedule_execution_count",
        profile.recurrence_replay_schedule_execution_count,
    )?;
    payload.set_item(
        "recurrence_union_schedule_execution_count",
        profile.recurrence_union_schedule_execution_count,
    )?;
    payload.set_item(
        "recurrence_union_source_row_count",
        profile.recurrence_union_source_row_count,
    )?;
    payload.set_item(
        "recurrence_replay_output_value_count",
        profile.recurrence_replay_output_value_count,
    )?;
    payload.set_item(
        "recurrence_source_call_count",
        profile.recurrence_source_call_count,
    )?;
    payload.set_item(
        "recurrence_source_row_count",
        profile.recurrence_source_row_count,
    )?;
    payload.set_item(
        "recurrence_contribution_call_count",
        profile.recurrence_contribution_call_count,
    )?;
    payload.set_item(
        "recurrence_contribution_row_count",
        profile.recurrence_contribution_row_count,
    )?;
    payload.set_item(
        "recurrence_finalization_call_count",
        profile.recurrence_finalization_call_count,
    )?;
    payload.set_item(
        "recurrence_finalization_row_count",
        profile.recurrence_finalization_row_count,
    )?;
    payload.set_item(
        "recurrence_closure_call_count",
        profile.recurrence_closure_call_count,
    )?;
    payload.set_item(
        "recurrence_closure_row_count",
        profile.recurrence_closure_row_count,
    )?;
    payload.set_item("selector_planner_time_s", profile.selector_planner_s)?;
    payload.set_item("selector_gather_time_s", profile.selector_gather_s)?;
    payload.set_item("selector_scatter_time_s", profile.selector_scatter_s)?;
    payload.set_item("selector_plan_kind", &profile.selector_plan_kind)?;
    payload.set_item("selector_group_sizes", profile.selector_group_sizes.clone())?;
    payload.set_item(
        "selector_reordered_point_count",
        profile.selector_reordered_point_count,
    )?;
    payload.set_item("selector_simd_lane_width", profile.selector_simd_lane_width)?;
    payload.set_item("selector_simd_occupancy", profile.selector_simd_occupancy)?;
    payload.set_item(
        "native_input_component_count",
        profile.native_input_component_count,
    )?;
    payload.set_item("native_input_pack_bytes", profile.native_input_pack_bytes)?;
    payload.set_item(
        "native_input_crossing_bytes",
        profile.native_input_crossing_bytes,
    )?;
    payload.set_item(
        "native_input_container_allocation_count",
        profile.native_input_container_allocation_count,
    )?;
    payload.set_item("state_component_count", profile.state_component_count)?;
    payload.set_item(
        "state_clear_component_count",
        profile.state_clear_component_count,
    )?;
    payload.set_item("source_component_count", profile.source_component_count)?;
    payload.set_item("momentum_component_count", profile.momentum_component_count)?;
    payload.set_item(
        "model_parameter_component_count",
        profile.model_parameter_component_count,
    )?;
    payload.set_item(
        "stage_input_copy_component_count",
        profile.stage_input_copy_component_count,
    )?;
    payload.set_item(
        "stage_leaf_input_copy_component_count",
        profile.stage_leaf_input_copy_component_count,
    )?;
    payload.set_item(
        "stage_evaluator_output_gather_component_count",
        profile.stage_evaluator_output_gather_component_count,
    )?;
    payload.set_item(
        "stage_output_assign_component_count",
        profile.stage_output_assign_component_count,
    )?;
    payload.set_item(
        "amplitude_input_copy_component_count",
        profile.amplitude_input_copy_component_count,
    )?;
    payload.set_item(
        "amplitude_leaf_input_copy_component_count",
        profile.amplitude_leaf_input_copy_component_count,
    )?;
    payload.set_item(
        "amplitude_evaluator_output_gather_component_count",
        profile.amplitude_evaluator_output_gather_component_count,
    )?;
    payload.set_item(
        "amplitude_output_remap_component_count",
        profile.amplitude_output_remap_component_count,
    )?;
    payload.set_item(
        "evaluator_backend_call_count",
        profile.evaluator_backend_call_count,
    )?;
    payload.set_item(
        "compiled_direct_arena_engine_count",
        profile.compiled_direct_arena_engine_count,
    )?;
    payload.set_item(
        "compiled_direct_arena_call_count",
        profile.compiled_direct_arena_call_count,
    )?;
    payload.set_item(
        "compiled_direct_arena_minimum_effective_tile_capacity",
        profile.compiled_direct_arena_minimum_effective_tile_capacity,
    )?;
    payload.set_item(
        "compiled_direct_arena_maximum_physical_scalar_values_per_point",
        profile.compiled_direct_arena_maximum_physical_scalar_values_per_point,
    )?;
    payload.set_item(
        "compiled_direct_arena_maximum_hot_scalar_values_per_point",
        profile.compiled_direct_arena_maximum_hot_scalar_values_per_point,
    )?;
    payload.set_item(
        "compiled_direct_arena_maximum_source_scalar_values_per_point",
        profile.compiled_direct_arena_maximum_source_scalar_values_per_point,
    )?;
    payload.set_item(
        "compiled_direct_arena_maximum_reduction_scalar_values_per_point",
        profile.compiled_direct_arena_maximum_reduction_scalar_values_per_point,
    )?;
    payload.set_item(
        "compiled_direct_arena_boundary_input_bytes",
        profile.compiled_direct_arena_boundary_input_bytes,
    )?;
    payload.set_item(
        "compiled_direct_arena_boundary_current_output_bytes",
        profile.compiled_direct_arena_boundary_current_output_bytes,
    )?;
    payload.set_item(
        "compiled_direct_arena_boundary_amplitude_output_bytes",
        profile.compiled_direct_arena_boundary_amplitude_output_bytes,
    )?;
    payload.set_item(
        "reduction_input_component_count",
        profile.reduction_input_component_count,
    )?;
    payload.set_item(
        "selector_gather_point_count",
        profile.selector_gather_point_count,
    )?;
    payload.set_item("selector_gather_bytes", profile.selector_gather_bytes)?;
    payload.set_item(
        "selector_scatter_value_count",
        profile.selector_scatter_value_count,
    )?;
    payload.set_item(
        "resolved_materialized_component_count",
        profile.resolved_materialized_component_count,
    )?;
    payload.set_item(
        "total_materialized_value_count",
        profile.total_materialized_value_count,
    )?;
    payload.set_item(
        "final_output_copy_value_count",
        profile.final_output_copy_value_count,
    )?;
    payload.set_item(
        "observed_scratch_reallocation_count",
        profile.observed_scratch_reallocation_count,
    )?;
    payload.set_item(
        "native_output_allocation_count",
        profile.native_output_allocation_count,
    )?;
    Ok(payload)
}

#[pyfunction]
fn abi_version() -> u32 {
    NativeRuntime::ABI_VERSION
}

#[pyfunction]
fn package_version() -> &'static str {
    option_env!("PYAMPLICOL_PACKAGE_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))
}

#[pyfunction]
fn native_build_inputs_sha256() -> &'static str {
    option_env!("PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256").unwrap_or("")
}

#[pyfunction]
fn target_info() -> TargetInfo {
    let target = runtime_target_info();
    TargetInfo {
        triple: target.triple,
        cpu_features: target.cpu_features,
    }
}

#[pyfunction]
fn _preflight_eager_kernel_pack(manifest_path: PathBuf, payload_root: PathBuf) -> PyResult<usize> {
    preflight_prepared_kernel_pack(&manifest_path, &payload_root).map_err(python_error)
}

#[pyfunction]
fn _eager_direct_descriptor_v1(
    py: Python<'_>,
    source_application: &[u8],
    input_complex_count: u32,
    output_complex_count: u32,
) -> PyResult<Py<PyBytes>> {
    let descriptor = eager_direct_descriptor_for_source_application_bytes(
        source_application,
        input_complex_count,
        output_complex_count,
        std::path::Path::new("<prepared eager application>"),
    )
    .map_err(python_error)?;
    Ok(PyBytes::new(py, &descriptor).unbind())
}

#[pyfunction]
fn _compile_symjit_plane_application_v2(
    py: Python<'_>,
    program_repr: &str,
    input_complex_count: usize,
    output_complex_count: usize,
    optimization_level: u8,
    compress: bool,
) -> PyResult<Py<PyBytes>> {
    let application = compile_symbolica_program_to_plane_application_bytes(
        program_repr,
        input_complex_count,
        output_complex_count,
        optimization_level,
        compress,
    )
    .map_err(python_error)?;
    Ok(PyBytes::new(py, &application).unbind())
}

#[pyfunction]
fn _build_recurrence_bootstrap_v1(
    py: Python<'_>,
    context_json: &[u8],
    execution_json: &[u8],
    physics_json: &[u8],
) -> PyResult<Py<PyBytes>> {
    let image = build_recurrence_bootstrap_image_v1(context_json, execution_json, physics_json)
        .map_err(python_error)?;
    Ok(PyBytes::new(py, &image).unbind())
}

#[pyfunction]
fn _load_eager_reduction_groups_v1(
    py: Python<'_>,
    artifact_root: PathBuf,
    process_id: String,
) -> PyResult<Py<PyAny>> {
    let result_process_id = process_id.clone();
    let reduction_groups = py
        .detach(move || NativeRuntime::load_eager_reduction_groups(artifact_root, &process_id))
        .map_err(python_error)?;
    let result = PyDict::new(py);
    result.set_item("abi", "pyamplicol-eager-reduction-groups-v1")?;
    result.set_item("runtime_layout_abi", "pyamplicol-eager-runtime-layout-v1")?;
    result.set_item("process_id", result_process_id)?;
    result.set_item(
        "reduction_groups",
        json_value_to_python(py, &reduction_groups)?,
    )?;
    Ok(result.into_any().unbind())
}

#[pyfunction]
fn _load_eager_exact_sections_v1(
    py: Python<'_>,
    artifact_root: PathBuf,
    process_id: String,
) -> PyResult<Py<PyAny>> {
    let sections = py
        .detach(move || NativeRuntime::load_eager_exact_sections(artifact_root, &process_id))
        .map_err(python_error)?;
    eager_exact_sections_to_python(py, sections)
}

#[pyfunction]
fn _load_recurrence_exact_sections_v1(
    py: Python<'_>,
    artifact_root: PathBuf,
    process_id: String,
) -> PyResult<Py<PyAny>> {
    let sections = py
        .detach(move || NativeRuntime::load_recurrence_exact_sections(artifact_root, &process_id))
        .map_err(python_error)?;
    recurrence_exact_sections_to_python(py, sections)
}

fn eager_exact_sections_to_python(
    py: Python<'_>,
    sections: rusticol_core::NativeEagerExactSections,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("abi", "pyamplicol-eager-exact-sections-v1")?;
    result.set_item("runtime_layout_abi", "pyamplicol-eager-runtime-layout-v1")?;
    result.set_item("process_id", sections.process_id)?;
    result.set_item(
        "exact_schema",
        json_value_to_python(py, &sections.exact_schema)?,
    )?;
    result.set_item(
        "reduction_groups",
        json_value_to_python(py, &sections.reduction_groups)?,
    )?;
    result.set_item("selector_group_ids", sections.selector_group_ids)?;
    result.set_item("selector_domains", sections.selector_domains)?;
    result.set_item(
        "couplings",
        sections
            .couplings
            .into_iter()
            .map(|row| {
                (
                    row.real_parameter_id,
                    row.imaginary_parameter_id,
                    row.constant_real,
                    row.constant_imaginary,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "stages",
        sections
            .stages
            .into_iter()
            .map(|row| {
                (
                    row.stage_index,
                    row.invocation_start,
                    row.invocation_count,
                    row.attachment_start,
                    row.attachment_count,
                    row.finalization_start,
                    row.finalization_count,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "invocations",
        sections
            .invocations
            .into_iter()
            .map(|row| {
                (
                    row.kernel_id,
                    row.left_value_slot_id,
                    row.right_value_slot_id,
                    row.left_momentum_slot_id,
                    row.right_momentum_slot_id,
                    row.coupling_slot_id,
                    row.output_factor_source,
                    row.attachment_start,
                    row.attachment_count,
                    row.selector_domain_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "attachments",
        sections
            .attachments
            .into_iter()
            .map(|row| {
                (
                    row.result_current_id,
                    row.factor_numerators
                        .into_iter()
                        .map(|[real, imaginary]| (real, imaginary))
                        .collect::<Vec<_>>(),
                    row.factor_denominator
                        .map(|[real, imaginary]| (real, imaginary)),
                    row.selector_domain_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "finalizations",
        sections
            .finalizations
            .into_iter()
            .map(|row| {
                (
                    row.kernel_id,
                    row.current_id,
                    row.unpropagated_value_slot_id,
                    row.propagated_value_slot_id,
                    row.momentum_slot_id,
                    row.unpropagated_selector_domain_id,
                    row.propagated_selector_domain_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "closures",
        sections
            .closures
            .into_iter()
            .map(|row| {
                (
                    row.kernel_id,
                    row.left_value_slot_id,
                    row.right_value_slot_id,
                    row.amplitude_index,
                    row.coupling_slot_id,
                    row.output_factor_source,
                    row.factor_numerators
                        .into_iter()
                        .map(|[real, imaginary]| (real, imaginary))
                        .collect::<Vec<_>>(),
                    row.factor_denominator
                        .map(|[real, imaginary]| (real, imaginary)),
                    row.direct_coefficients.map(|values| {
                        values
                            .into_iter()
                            .map(|[real, imaginary]| (real, imaginary))
                            .collect::<Vec<_>>()
                    }),
                    row.coherent_group_id,
                    row.selector_domain_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    Ok(result.into_any().unbind())
}

fn recurrence_exact_sections_to_python(
    py: Python<'_>,
    sections: rusticol_core::NativeRecurrenceExactSections,
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    result.set_item("abi", "pyamplicol-recurrence-exact-sections-v1")?;
    result.set_item(
        "runtime_layout_abi",
        "pyamplicol-recurrence-runtime-layout-v2",
    )?;
    result.set_item("process_id", sections.process_id)?;
    result.set_item("strategy", sections.strategy)?;
    result.set_item("semantic_digest", sections.semantic_digest)?;
    result.set_item("runtime_layout_digest", sections.runtime_layout_digest)?;
    result.set_item(
        "counts",
        (
            sections.current_arena_components,
            sections.amplitude_destination_count,
            sections.parameter_value_count,
            sections.external_source_count,
        ),
    )?;
    result.set_item(
        "currents",
        sections
            .currents
            .into_iter()
            .map(|row| {
                (
                    row.semantic_current_id,
                    row.node_kind as u16,
                    row.state_template_id,
                    row.component_base,
                    row.component_count,
                    row.momentum_form_id,
                    row.stage,
                    row.selector_domain_id,
                    row.first_use,
                    row.last_use,
                    row.source_row_or_sentinel,
                    row.finalization_row_or_sentinel,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "sources",
        sections
            .sources
            .into_iter()
            .map(|row| {
                (
                    row.source_slot,
                    row.destination_component_base,
                    row.momentum_form_id,
                    row.source_template_or_dispatch_domain,
                    row.spin_state_class,
                    row.exact_factor_id,
                    row.selector_domain_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "contributions",
        sections
            .contributions
            .into_iter()
            .map(|row| {
                (
                    row.parent0_component_base,
                    row.parent1_component_base_or_sentinel,
                    row.parent0_momentum_form_id,
                    row.parent1_momentum_form_id_or_sentinel,
                    row.destination_component_base,
                    row.exact_factor_id,
                    row.selector_domain_id,
                    row.flags,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "finalizations",
        sections
            .finalizations
            .into_iter()
            .map(|row| {
                (
                    row.component_base,
                    row.component_count,
                    row.momentum_form_id,
                    row.exact_factor_id,
                    row.selector_domain_id,
                    row.flags,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "closures",
        sections
            .closures
            .into_iter()
            .map(|row| {
                (
                    row.parent0_component_base,
                    row.parent1_component_base_or_sentinel,
                    row.parent0_momentum_form_id,
                    row.parent1_momentum_form_id_or_sentinel,
                    row.amplitude_destination_id,
                    row.exact_factor_id,
                    row.component_factor_start,
                    row.component_count,
                    row.selector_domain_id,
                    row.flags,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "row_groups",
        sections
            .row_groups
            .into_iter()
            .map(|row| {
                (
                    row.stage,
                    row.role as u16,
                    row.destination_operation as u16,
                    row.direct_executor_id,
                    row.row_start,
                    row.row_count,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "momentum_forms",
        sections
            .momentum_forms
            .into_iter()
            .map(|row| (row.term_start, row.term_count))
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "momentum_terms",
        sections
            .momentum_terms
            .into_iter()
            .map(|row| (row.source_slot, row.coefficient))
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "replay_targets",
        sections
            .replay_targets
            .into_iter()
            .map(|row| {
                (
                    row.public_flow_id,
                    row.representative_id,
                    row.source_permutation_start,
                    row.source_permutation_count,
                    row.helicity_map_start,
                    row.helicity_map_count,
                    row.phase_exact_factor_id,
                    row.multiplicity,
                    row.selector_domain_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item("source_permutations", sections.source_permutations)?;
    result.set_item("replay_momentum_signs", sections.replay_momentum_signs)?;
    result.set_item("replay_helicity_map", sections.replay_helicity_map)?;
    result.set_item(
        "source_state_assignments",
        sections
            .source_state_assignments
            .into_iter()
            .map(|row| (row.source_slot, row.state_index))
            .collect::<Vec<_>>(),
    )?;
    let source_dispatch_variants = sections
        .source_dispatch_variants
        .into_iter()
        .map(|row| {
            Ok(vec![
                i64::try_from(row.embedding_start).map_err(|_| {
                    PyValueError::new_err("recurrence source embedding offset exceeds i64")
                })?,
                i64::try_from(row.projection_start).map_err(|_| {
                    PyValueError::new_err("recurrence source projection offset exceeds i64")
                })?,
                i64::from(row.source_row_id),
                i64::from(row.dispatch_domain_id),
                i64::from(row.runtime_variant_id),
                i64::from(row.source_state_index),
                i64::from(row.source_template_id),
                i64::from(row.source_state_template_id),
                i64::from(row.crossed_state_template_id),
                i64::from(row.crossed_spin_state_class),
                i64::from(row.direct_executor_id),
                i64::from(row.crossing_exact_factor_id),
                i64::from(row.embedding_count),
                i64::from(row.projection_count),
            ])
        })
        .collect::<PyResult<Vec<_>>>()?;
    result.set_item("source_dispatch_variants", source_dispatch_variants)?;
    result.set_item(
        "source_embeddings",
        sections
            .source_embeddings
            .into_iter()
            .map(|row| {
                (
                    row.full_component,
                    row.source_component_or_sentinel,
                    row.exact_factor_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "source_projections",
        sections
            .source_projections
            .into_iter()
            .map(|row| (row.source_component, row.full_component))
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "resolved_source_selections",
        sections
            .resolved_source_selections
            .into_iter()
            .map(|row| (row.source_slot, row.dispatch_variant_id))
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "amplitude_destinations",
        sections
            .amplitude_destinations
            .into_iter()
            .map(|row| {
                (
                    row.closure_row_start,
                    row.id,
                    row.target_sector_id,
                    row.target_helicity_id_or_sentinel,
                    row.closure_row_count,
                    row.selector_domain_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item(
        "resolved_helicities",
        sections
            .resolved_helicities
            .into_iter()
            .map(|row| {
                (
                    row.source_state_start,
                    row.source_selection_start,
                    row.public_helicity_start,
                    row.id,
                    row.source_state_count,
                    row.source_selection_count,
                    row.public_helicity_count,
                    row.selector_domain_id,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item("public_helicities", sections.public_helicities)?;
    result.set_item(
        "exact_factors",
        sections
            .exact_factors
            .into_iter()
            .map(|factor| {
                (
                    factor.real_numerator,
                    factor.real_denominator,
                    factor.imaginary_numerator,
                    factor.imaginary_denominator,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    result.set_item("public_flow_ids", sections.public_flow_ids)?;
    result.set_item(
        "executors",
        sections
            .executors
            .into_iter()
            .map(|executor| {
                (
                    executor.direct_executor_id,
                    executor.role,
                    executor.destination_operation,
                    executor.parent_component_counts,
                    executor.destination_component_count,
                    executor.momentum_operand_count,
                    executor.prepared_kernel_id,
                    executor.runtime_template,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    Ok(result.into_any().unbind())
}

fn json_value_to_python(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => value.into_py_any(py),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                value.into_py_any(py)
            } else if let Some(value) = value.as_u64() {
                value.into_py_any(py)
            } else if let Some(value) = value.as_f64() {
                value.into_py_any(py)
            } else {
                Err(ArtifactError::new_err(
                    "compact eager exact schema contains an invalid number",
                ))
            }
        }
        Value::String(value) => value.as_str().into_py_any(py),
        Value::Array(values) => {
            let values = values
                .iter()
                .map(|value| json_value_to_python(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, values)?.into_any().unbind())
        }
        Value::Object(values) => {
            let result = PyDict::new(py);
            for (key, value) in values {
                result.set_item(key, json_value_to_python(py, value)?)?;
            }
            Ok(result.into_any().unbind())
        }
    }
}

#[pymodule]
fn _rusticol(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RusticolError", module.py().get_type::<RusticolError>())?;
    module.add("ArtifactError", module.py().get_type::<ArtifactError>())?;
    module.add(
        "CompatibilityError",
        module.py().get_type::<CompatibilityError>(),
    )?;
    module.add("EvaluationError", module.py().get_type::<EvaluationError>())?;
    module.add("SelectorError", module.py().get_type::<SelectorError>())?;
    module.add(
        "ModelParameterError",
        module.py().get_type::<ModelParameterError>(),
    )?;
    module.add_class::<Runtime>()?;
    module.add_class::<ProcessPhysics>()?;
    module.add_class::<ExternalParticle>()?;
    module.add_class::<HelicityConfiguration>()?;
    module.add_class::<ColorFlow>()?;
    module.add_class::<ContractedColorComponent>()?;
    module.add_class::<ReductionGroup>()?;
    module.add_class::<PhysicsReduction>()?;
    module.add_class::<ModelParameter>()?;
    module.add_class::<ResolvedEvaluation>()?;
    module.add_class::<TargetInfo>()?;
    module.add_function(wrap_pyfunction!(abi_version, module)?)?;
    module.add_function(wrap_pyfunction!(package_version, module)?)?;
    module.add_function(wrap_pyfunction!(native_build_inputs_sha256, module)?)?;
    module.add_function(wrap_pyfunction!(target_info, module)?)?;
    module.add_function(wrap_pyfunction!(_preflight_eager_kernel_pack, module)?)?;
    module.add_function(wrap_pyfunction!(
        _compile_symjit_plane_application_v2,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_build_recurrence_bootstrap_v1, module)?)?;
    module.add_function(wrap_pyfunction!(_eager_direct_descriptor_v1, module)?)?;
    module.add_function(wrap_pyfunction!(_load_eager_reduction_groups_v1, module)?)?;
    module.add_function(wrap_pyfunction!(_load_eager_exact_sections_v1, module)?)?;
    module.add_function(wrap_pyfunction!(
        _load_recurrence_exact_sections_v1,
        module
    )?)?;
    #[cfg(feature = "numpy")]
    module.add_function(wrap_pyfunction!(
        eager_lowering::_lower_eager_runtime_v1,
        module
    )?)?;
    #[cfg(feature = "numpy")]
    module.add_function(wrap_pyfunction!(
        recurrence::_lower_recurrence_direct_v2,
        module
    )?)?;
    #[cfg(all(feature = "numpy", feature = "on-the-fly-test-support"))]
    module.add_function(wrap_pyfunction!(
        recurrence::_on_the_fly_test_support_probe_v1,
        module
    )?)?;
    #[cfg(all(feature = "numpy", feature = "on-the-fly-test-support"))]
    module.add_function(wrap_pyfunction!(
        recurrence::_on_the_fly_query_family_census_v1,
        module
    )?)?;
    #[cfg(all(feature = "numpy", feature = "on-the-fly-test-support"))]
    module.add_function(wrap_pyfunction!(
        recurrence::_on_the_fly_artifact_probe_v1,
        module
    )?)?;
    #[cfg(all(feature = "numpy", feature = "on-the-fly-test-support"))]
    module.add_function(wrap_pyfunction!(
        recurrence::_on_the_fly_execution_diagnostic_v1,
        module
    )?)?;
    #[cfg(feature = "numpy")]
    module.add_function(wrap_pyfunction!(
        recurrence_template::_validate_recurrence_template_input_v1,
        module
    )?)?;
    #[cfg(feature = "numpy")]
    module.add_function(wrap_pyfunction!(
        recurrence::_build_on_the_fly_process_seeds_v1,
        module
    )?)?;
    #[cfg(feature = "numpy")]
    module.add_function(wrap_pyfunction!(
        recurrence::_build_on_the_fly_process_seed_v1,
        module
    )?)?;
    #[cfg(feature = "numpy")]
    module.add_function(wrap_pyfunction!(
        recurrence::_inspect_on_the_fly_process_seed_v1,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusticol_core::{
        ContractedColor as CoreContractedColor, Coverage, ExternalParticle as CoreExternalParticle,
        Helicity, LcColorFlow, Reduction, ReductionGroup as CoreReductionGroup, ReductionKind,
        SelectorCapabilities,
    };

    #[cfg(not(feature = "on-the-fly-test-support"))]
    #[test]
    fn default_build_exposes_loaded_on_the_fly_state_census_binding() {
        let binding: fn(&Runtime) -> PyResult<Option<String>> =
            Runtime::_on_the_fly_runtime_state_census_json;
        let _ = binding;
    }

    #[test]
    fn raw_runtime_accepts_only_f64_precision() {
        assert!(require_raw_f64_precision(16).is_ok());
        let error = require_raw_f64_precision(32).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(error.to_string().contains("only precision=16"));
        assert!(
            error
                .to_string()
                .contains("public Python Symbolica executor")
        );
    }

    #[test]
    fn detached_runtime_guard_allows_only_its_origin_thread() {
        let origin = thread::current().id();
        assert!(require_origin_thread(origin).is_ok());

        let error = thread::spawn(move || require_origin_thread(origin).unwrap_err())
            .join()
            .expect("thread guard check must not panic");
        assert_eq!(error.kind(), RusticolErrorKind::Internal);
        assert!(error.to_string().contains("owning thread"));
    }

    #[test]
    fn process_physics_maps_the_canonical_python_metadata_fields() {
        let mut physics = CoreProcessPhysics {
            schema_version: 1,
            kind: "pyamplicol-resolved-physics".to_string(),
            process_id: "p0".to_string(),
            process: "d d~ > z".to_string(),
            color_accuracy: ColorAccuracy::Lc,
            coverage: Coverage {
                helicities: "complete".to_string(),
                color: "complete".to_string(),
                color_kind: "physical-lc-flows".to_string(),
                structural_zero_helicity_count: 0,
            },
            external_particles: vec![
                core_particle(0, "d", 1, ParticleRole::Initial),
                core_particle(1, "d~", -1, ParticleRole::Initial),
                core_particle(2, "z", 23, ParticleRole::Final),
            ],
            helicities: vec![Helicity {
                id: "h0".to_string(),
                index: 0,
                values: vec![-1, 1, 0],
                computed: true,
                structural_zero: false,
                representative_id: "h0".to_string(),
                coefficient: 2.0,
            }],
            color_components: vec![CoreColorComponent::LcFlow(LcColorFlow {
                id: "c0".to_string(),
                index: 0,
                word: vec![1, 2],
                computed: true,
                representative_id: "c0".to_string(),
                coefficient: 3.0,
            })],
            reduction: Reduction {
                kind: ReductionKind::LcDiagonal,
                groups: vec![CoreReductionGroup {
                    id: "group:0".to_string(),
                    representative_helicity_id: "h0".to_string(),
                    representative_color_id: "c0".to_string(),
                    physical_helicity_ids: vec!["h0".to_string()],
                    physical_color_ids: vec!["c0".to_string()],
                }],
            },
            model_parameters: vec![CoreModelParameter {
                name: "alpha_s".to_string(),
                kind: ParameterKind::Coupling,
                default_real: 0.118,
                default_imaginary: 0.0,
                mutable: true,
            }],
            selectors: SelectorCapabilities {
                helicity: true,
                color_flow: true,
                contracted_color: false,
            },
            extensions: BTreeMap::new(),
        };

        physics.validate().unwrap();
        let mapped = ProcessPhysics::from_core(&physics);

        assert_eq!(mapped.process_id, "p0");
        assert_eq!(mapped.process, "d d~ > z");
        assert_eq!(mapped.color_accuracy, "lc");
        assert_eq!(mapped.helicity_coverage, "complete");
        assert_eq!(mapped.color_coverage, "complete");
        assert_eq!(mapped.color_kind, "physical-lc-flows");
        assert_eq!(mapped.structural_zero_helicity_count, 0);
        assert_eq!(mapped.external_particles[0].name, "d");
        assert_eq!(mapped.external_particles[0].pdg_id, 1);
        assert_eq!(mapped.external_particles[0].state, "incoming");
        assert_eq!(mapped.external_particles[2].state, "outgoing");
        assert_eq!(mapped.helicities[0].representative_id, "h0");
        assert_eq!(mapped.helicities[0].coefficient, 2.0);
        assert_eq!(mapped.color_flows[0].word, vec![1, 2]);
        assert_eq!(mapped.color_flows[0].coefficient, 3.0);
        assert!(mapped.contracted_color_components.is_empty());
        assert_eq!(mapped.reduction.kind, "lc-diagonal");
        assert_eq!(mapped.reduction.groups[0].id, "group:0");
        assert_eq!(mapped.reduction.groups[0].physical_helicity_ids, vec!["h0"]);
        assert_eq!(mapped.model_parameters[0].kind, "coupling");
        assert_eq!(mapped.model_parameters[0].default_real, 0.118);
        assert!(mapped.model_parameters[0].mutable);
        assert_eq!(mapped.selector_capabilities, vec!["helicity", "color_flow"]);

        physics.color_accuracy = ColorAccuracy::Full;
        physics.coverage.color = "contracted".to_string();
        physics.coverage.color_kind = "contracted-color".to_string();
        physics.color_components = vec![CoreColorComponent::ContractedColor(CoreContractedColor {
            id: "contracted".to_string(),
            index: 0,
            description: "fully contracted color".to_string(),
        })];
        physics.reduction.kind = ReductionKind::ContractedColor;
        physics.reduction.groups[0].representative_color_id = "contracted".to_string();
        physics.reduction.groups[0].physical_color_ids = vec!["contracted".to_string()];
        physics.selectors.color_flow = false;
        physics.validate().unwrap();

        let mapped = ProcessPhysics::from_core(&physics);
        assert!(mapped.color_flows.is_empty());
        assert_eq!(mapped.contracted_color_components[0].id, "contracted");
        assert_eq!(mapped.selector_capabilities, vec!["helicity"]);
    }

    #[test]
    fn target_info_is_canonical_and_matches_the_core() {
        let mapped = target_info();
        let core = runtime_target_info();
        assert_eq!(mapped.triple, core.triple);
        assert_eq!(mapped.cpu_features, core.cpu_features);
        assert!(mapped.cpu_features.windows(2).all(|pair| pair[0] < pair[1]));
    }

    fn core_particle(
        index: usize,
        name: &str,
        pdg: i32,
        role: ParticleRole,
    ) -> CoreExternalParticle {
        CoreExternalParticle {
            index,
            label: index + 1,
            particle: name.to_string(),
            pdg,
            role,
            momentum_slot: index,
            momentum_components: ["E", "px", "py", "pz"].map(str::to_string),
        }
    }
}
