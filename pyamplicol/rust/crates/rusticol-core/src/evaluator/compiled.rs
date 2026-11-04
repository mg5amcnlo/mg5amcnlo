// SPDX-License-Identifier: 0BSD

use super::super::*;
use crate::artifact::PinnedNativeLibrary;
use std::sync::Arc;

type EvaluateFunction =
    unsafe extern "C" fn(*const Complex<f64>, *mut Complex<f64>, *mut Complex<f64>);
type BufferLengthFunction = unsafe extern "C" fn() -> std::ffi::c_ulong;

pub(crate) struct CompiledComplexF64Evaluator {
    _library: Arc<PinnedNativeLibrary>,
    evaluate: EvaluateFunction,
    scratch: Vec<Complex<f64>>,
    input_len: usize,
    output_len: usize,
}

impl CompiledComplexF64Evaluator {
    pub(crate) fn load(
        library: Arc<PinnedNativeLibrary>,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> RusticolResult<Self> {
        if function_name.is_empty() || input_len == 0 || output_len == 0 {
            return Err(RusticolError::invalid_argument(
                "compiled evaluator function_name, input_len, and output_len must be non-empty",
            ));
        }
        if std::mem::size_of::<Complex<f64>>() != 2 * std::mem::size_of::<f64>()
            || std::mem::align_of::<Complex<f64>>() != std::mem::align_of::<f64>()
        {
            return Err(RusticolError::compatibility(
                "Rusticol complex-f64 storage is incompatible with the compiled evaluator ABI",
            ));
        }

        let path = library.display_path();
        let exported_name = format!("{function_name}_complexf64");
        let evaluate = unsafe {
            *library
                .get::<EvaluateFunction>(exported_name.as_bytes())
                .map_err(|error| {
                    RusticolError::evaluation(format!(
                        "could not load compiled evaluator symbol {exported_name:?} from {}: {error}",
                        path.display()
                    ))
                })?
        };
        let buffer_name = format!("{exported_name}_get_buffer_len");
        let buffer_length = unsafe {
            *library
                .get::<BufferLengthFunction>(buffer_name.as_bytes())
                .map_err(|error| {
                    RusticolError::evaluation(format!(
                        "could not load compiled evaluator symbol {buffer_name:?} from {}: {error}",
                        path.display()
                    ))
                })?
        };
        let scratch_len = usize::try_from(unsafe { buffer_length() }).map_err(|_| {
            RusticolError::evaluation("compiled evaluator scratch-buffer length exceeds usize")
        })?;
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(scratch_len).map_err(|error| {
            RusticolError::evaluation(format!(
                "could not allocate compiled evaluator scratch buffer of length {scratch_len}: {error}"
            ))
        })?;
        scratch.resize(scratch_len, Complex { re: 0.0, im: 0.0 });

        Ok(Self {
            _library: library,
            evaluate,
            scratch,
            input_len,
            output_len,
        })
    }

    pub(crate) fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        if batch_size == 0 {
            return Err(RusticolError::invalid_argument(
                "compiled evaluator batch_size must be positive",
            ));
        }
        let expected_params = batch_size.checked_mul(self.input_len).ok_or_else(|| {
            RusticolError::invalid_argument("compiled evaluator input length overflows usize")
        })?;
        let expected_outputs = batch_size.checked_mul(self.output_len).ok_or_else(|| {
            RusticolError::invalid_argument("compiled evaluator output length overflows usize")
        })?;
        if params.len() != expected_params {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {expected_params}",
                params.len()
            )));
        }
        if out.len() != expected_outputs {
            return Err(RusticolError::invalid_argument(format!(
                "output buffer has length {}, expected {expected_outputs}",
                out.len()
            )));
        }

        for (input_row, output_row) in params
            .chunks_exact(self.input_len)
            .zip(out.chunks_exact_mut(self.output_len))
        {
            // SAFETY: Length checks above guarantee complete input/output rows. The
            // scratch buffer has the size reported by the loaded evaluator itself.
            unsafe {
                (self.evaluate)(
                    input_row.as_ptr(),
                    self.scratch.as_mut_ptr(),
                    output_row.as_mut_ptr(),
                );
            }
        }
        Ok(())
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn compiled_fixture() -> (PathBuf, PathBuf) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "rusticol-compiled-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let source = directory.join("fixture.cpp");
        let library = directory.join(if cfg!(target_os = "macos") {
            "libfixture.dylib"
        } else {
            "libfixture.so"
        });
        fs::write(
            &source,
            r#"#include <complex>
extern "C" unsigned long rusticol_test_complexf64_get_buffer_len() { return 1; }
extern "C" void rusticol_test_complexf64(
    std::complex<double>* params,
    std::complex<double>* buffer,
    std::complex<double>* out) {
    buffer[0] = params[0] + params[1];
    out[0] = buffer[0];
    out[1] = params[0] * params[1];
}
"#,
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
            "could not compile test evaluator: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        (directory, library)
    }

    #[test]
    fn symbolica_compiled_complex_abi_loads_and_evaluates_batches() {
        let (directory, library) = compiled_fixture();
        let mut evaluator = CompiledComplexF64Evaluator::load(
            PinnedNativeLibrary::from_test_path(&library).unwrap(),
            "rusticol_test",
            2,
            2,
        )
        .unwrap();
        let params = [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, -4.0),
            Complex::new(-2.0, 0.5),
            Complex::new(4.0, 1.5),
        ];
        let mut output = [Complex::new(0.0, 0.0); 4];

        evaluator.evaluate_batch(2, &params, &mut output).unwrap();

        assert_eq!(output[0], Complex::new(4.0, -2.0));
        assert_eq!(output[1], Complex::new(11.0, 2.0));
        assert_eq!(output[2], Complex::new(2.0, 2.0));
        assert_eq!(output[3], Complex::new(-8.75, -1.0));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cpp_and_asm_manifests_load_through_the_runtime_group() {
        let (directory, library) = compiled_fixture();
        let library_name = library.file_name().unwrap().to_str().unwrap().to_string();
        for capability in [
            SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
            SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
        ] {
            let manifest = EvaluatorManifest::CompiledComplex {
                runtime_capability: capability.to_string(),
                function_name: "rusticol_test".to_string(),
                input_len: 2,
                output_len: 2,
                library_path: library_name.clone(),
                evaluator_state_path: Some("unused-symbolica-state.bin".to_string()),
                number_type: "complex".to_string(),
                native_direct_application: None,
            };
            let mut group = EvaluatorGroup::load(&manifest, &directory).unwrap();
            let output = group
                .evaluate_batch(1, &[Complex::new(1.0, 2.0), Complex::new(3.0, -4.0)])
                .unwrap();

            assert_eq!(output, [Complex::new(4.0, -2.0), Complex::new(11.0, 2.0)]);
        }
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn compiled_evaluator_validates_batch_dimensions() {
        let (directory, library) = compiled_fixture();
        let mut evaluator = CompiledComplexF64Evaluator::load(
            PinnedNativeLibrary::from_test_path(&library).unwrap(),
            "rusticol_test",
            2,
            2,
        )
        .unwrap();
        let mut output = [Complex::new(0.0, 0.0); 2];

        let error = evaluator
            .evaluate_batch(1, &[Complex::new(1.0, 0.0)], &mut output)
            .unwrap_err();

        assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
        fs::remove_dir_all(directory).unwrap();
    }
}
