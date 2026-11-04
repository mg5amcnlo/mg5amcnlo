// SPDX-License-Identifier: 0BSD

use super::super::*;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use symjit::{Applet, Application, Compiled, Config, Defuns, Storage};

pub(crate) struct SymjitApplicationEvaluator {
    applet: Applet,
    application_path: PathBuf,
    input_len: usize,
    output_len: usize,
    input_tail_scratch: Vec<Complex<f64>>,
    output_tail_scratch: Vec<Complex<f64>>,
}

impl std::fmt::Debug for SymjitApplicationEvaluator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymjitApplicationEvaluator")
            .field("application_path", &self.application_path)
            .field("input_len", &self.input_len)
            .field("output_len", &self.output_len)
            .finish_non_exhaustive()
    }
}

pub(crate) struct SymjitApplicationMetadata<'a> {
    pub(crate) runtime_capability: &'a str,
    pub(crate) application_abi: &'a str,
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,
    pub(crate) element_layout: &'a str,
    pub(crate) batch_layout: &'a str,
    pub(crate) compiler_type: &'a str,
    pub(crate) translation_mode: &'a str,
    pub(crate) optimization_level: u8,
    pub(crate) word_bits: u8,
    pub(crate) endianness: &'a str,
    pub(crate) required_defuns: &'a [String],
}

impl SymjitApplicationEvaluator {
    #[cfg(test)]
    pub(crate) fn load(
        path: &Path,
        metadata: SymjitApplicationMetadata<'_>,
    ) -> RusticolResult<Self> {
        let bytes = fs::read(path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not read SymJIT application {}: {error}",
                path.display()
            ))
        })?;
        Self::load_bytes(&bytes, path.to_path_buf(), metadata)
    }

    pub(crate) fn load_bytes(
        bytes: &[u8],
        display_path: PathBuf,
        metadata: SymjitApplicationMetadata<'_>,
    ) -> RusticolResult<Self> {
        validate_manifest_metadata(&metadata)?;

        // This is the same trusted-input path used by Symbolica's
        // JITCompiledEvaluator::load.
        let mut loader_config = Config::default();
        loader_config.set_defuns(Defuns::new());
        let mut input = bytes;
        let application = guard_symjit_panic(
            || Application::load(&mut input, &loader_config),
            |detail| {
                RusticolError::compatibility(format!(
                    "SymJIT panicked while loading application {} with ABI {}: {detail}",
                    display_path.display(),
                    SYMJIT_APPLICATION_STORAGE_ABI
                ))
            },
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not load SymJIT application {} with ABI {}: {error}",
                display_path.display(),
                SYMJIT_APPLICATION_STORAGE_ABI
            ))
        })?;
        validate_loaded_application(&display_path, &application, &metadata)?;
        let applet = guard_symjit_panic(
            || application.seal(),
            |detail| {
                RusticolError::compatibility(format!(
                    "SymJIT panicked while sealing application {}: {detail}",
                    display_path.display()
                ))
            },
        )?
        .map_err(|error| {
            RusticolError::evaluation(format!(
                "could not seal SymJIT application {}: {error}",
                display_path.display()
            ))
        })?;
        Ok(Self {
            applet,
            application_path: display_path,
            input_len: metadata.input_len,
            output_len: metadata.output_len,
            input_tail_scratch: Vec::new(),
            output_tail_scratch: Vec::new(),
        })
    }

    pub(crate) fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        validate_batch_lengths(
            batch_size,
            self.input_len,
            self.output_len,
            params.len(),
            out.len(),
        )?;
        if batch_size == 0 {
            return Ok(());
        }
        let lane_width = super::native_f64_simd_lane_width();
        let remainder = batch_size % lane_width;
        if remainder == 0 {
            return evaluate_matrix(
                &self.applet,
                &self.application_path,
                batch_size,
                params,
                out,
            );
        }

        let aligned_batch_size = batch_size - remainder;
        if aligned_batch_size != 0 {
            evaluate_matrix(
                &self.applet,
                &self.application_path,
                aligned_batch_size,
                &params[..aligned_batch_size * self.input_len],
                &mut out[..aligned_batch_size * self.output_len],
            )?;
        }

        let padded_input_len = lane_width
            .checked_mul(self.input_len)
            .ok_or_else(|| RusticolError::invalid_argument("SymJIT SIMD input batch overflows"))?;
        self.input_tail_scratch
            .resize(padded_input_len, Complex::new(0.0, 0.0));
        let source_start = aligned_batch_size * self.input_len;
        let source_end = source_start + remainder * self.input_len;
        self.input_tail_scratch[..remainder * self.input_len]
            .copy_from_slice(&params[source_start..source_end]);
        let tail_row_start = (remainder - 1) * self.input_len;
        for row in remainder..lane_width {
            let target_start = row * self.input_len;
            self.input_tail_scratch.copy_within(
                tail_row_start..tail_row_start + self.input_len,
                target_start,
            );
        }
        let padded_output_len = lane_width
            .checked_mul(self.output_len)
            .ok_or_else(|| RusticolError::invalid_argument("SymJIT SIMD output batch overflows"))?;
        self.output_tail_scratch
            .resize(padded_output_len, Complex::new(0.0, 0.0));
        evaluate_matrix(
            &self.applet,
            &self.application_path,
            lane_width,
            &self.input_tail_scratch,
            &mut self.output_tail_scratch,
        )?;
        let output_start = aligned_batch_size * self.output_len;
        let output_end = output_start + remainder * self.output_len;
        out[output_start..output_end]
            .copy_from_slice(&self.output_tail_scratch[..remainder * self.output_len]);
        Ok(())
    }

    pub(crate) fn evaluate_batch_unpadded(
        &self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        validate_batch_lengths(
            batch_size,
            self.input_len,
            self.output_len,
            params.len(),
            out.len(),
        )?;
        evaluate_matrix(
            &self.applet,
            &self.application_path,
            batch_size,
            params,
            out,
        )
    }

    pub(crate) fn simd_lane_width(&self) -> Option<usize> {
        if self.applet.use_threads {
            // The direct AoSoA entry point below executes blocks serially.
            // Preserve SymJIT's threaded matrix dispatch for applications that
            // explicitly requested it.
            return None;
        }
        self.applet
            .compiled_simd
            .as_ref()
            .map(Compiled::count_lanes)
            .filter(|lane_width| *lane_width != 0)
    }

    /// Evaluates already-transposed SIMD blocks without a row-major staging copy.
    ///
    /// Each block contains `input_len` complex SIMD values in AoSoA order: all
    /// real lanes followed by all imaginary lanes for one input, then the next
    /// input. Outputs use the analogous `output_len` layout. `false` asks the
    /// caller to fall back to the ordinary evaluator because no SIMD kernel is
    /// available or the loaded kernel rejected a block.
    pub(crate) fn evaluate_aosoa_blocks(
        &self,
        block_count: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> RusticolResult<bool> {
        let Some(compiled) = self.applet.compiled_simd.as_ref() else {
            return Ok(false);
        };
        let lane_width = compiled.count_lanes();
        if lane_width == 0 {
            return Ok(false);
        }
        validate_aosoa_lengths(
            block_count,
            lane_width,
            self.input_len,
            self.output_len,
            params.len(),
            out.len(),
        )?;
        if block_count == 0 {
            return Ok(true);
        }

        let input_block_len = lane_width * self.input_len * 2;
        let output_block_len = lane_width * self.output_len * 2;
        let function = compiled.func();
        guard_symjit_panic(
            || {
                for block in 0..block_count {
                    let status = function(
                        out[block * output_block_len..].as_mut_ptr(),
                        std::ptr::null(),
                        0,
                        params[block * input_block_len..].as_ptr(),
                    );
                    if status != 0 {
                        return false;
                    }
                }
                true
            },
            |detail| {
                RusticolError::evaluation(format!(
                    "SymJIT panicked while evaluating AoSoA blocks for application {}: {detail}",
                    self.application_path.display()
                ))
            },
        )
    }
}

fn evaluate_matrix(
    applet: &Applet,
    application_path: &Path,
    batch_size: usize,
    params: &[Complex<f64>],
    out: &mut [Complex<f64>],
) -> RusticolResult<()> {
    let params = complex_slice_as_scalars(params);
    let out = complex_slice_as_scalars_mut(out);
    guard_symjit_panic(
        || applet.evaluate_matrix(params, out, batch_size),
        |detail| {
            RusticolError::evaluation(format!(
                "SymJIT panicked while evaluating application {}: {detail}",
                application_path.display()
            ))
        },
    )?;
    Ok(())
}

fn guard_symjit_panic<T>(
    operation: impl FnOnce() -> T,
    error: impl FnOnce(String) -> RusticolError,
) -> RusticolResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| error(panic_detail(payload)))
}

fn panic_detail(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

pub(crate) fn validate_manifest_metadata(
    metadata: &SymjitApplicationMetadata<'_>,
) -> RusticolResult<()> {
    if metadata.runtime_capability != SYMJIT_APPLICATION_RUNTIME_CAPABILITY {
        return Err(RusticolError::unsupported_runtime_capability(
            metadata.runtime_capability,
            format!("a direct SymJIT evaluator requires {SYMJIT_APPLICATION_RUNTIME_CAPABILITY:?}"),
        ));
    }
    if metadata.application_abi != SYMJIT_APPLICATION_STORAGE_ABI {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application ABI {:?} is unsupported; this runtime requires {:?}",
            metadata.application_abi, SYMJIT_APPLICATION_STORAGE_ABI
        )));
    }
    if metadata.input_len == 0 || metadata.output_len == 0 {
        return Err(RusticolError::invalid_argument(
            "SymJIT application input_len and output_len must both be positive",
        ));
    }
    if metadata.element_layout != "complex-f64" || metadata.batch_layout != "row-major" {
        return Err(RusticolError::compatibility(format!(
            "unsupported SymJIT application layout element={:?}, batch={:?}",
            metadata.element_layout, metadata.batch_layout
        )));
    }
    if metadata.compiler_type != "native" {
        return Err(RusticolError::compatibility(format!(
            "unsupported SymJIT compiler type {:?}; expected \"native\"",
            metadata.compiler_type
        )));
    }
    if metadata.translation_mode != "indirect" {
        return Err(RusticolError::compatibility(format!(
            "unsupported SymJIT translation {:?}; direct translation is not a stable artifact ABI",
            metadata.translation_mode
        )));
    }
    if metadata.word_bits != usize::BITS as u8 || metadata.word_bits != 64 {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application requires a {}-bit runtime, but this host uses {} bits",
            metadata.word_bits,
            usize::BITS
        )));
    }
    let host_endianness = if cfg!(target_endian = "little") {
        "little"
    } else {
        "big"
    };
    if metadata.endianness != host_endianness || metadata.endianness != "little" {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application endianness {:?} is incompatible with host endianness {host_endianness:?}",
            metadata.endianness
        )));
    }
    if !metadata.required_defuns.is_empty() {
        return Err(RusticolError::unsupported_runtime_capability(
            SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            format!(
                "external functions are not self-contained: {}",
                metadata.required_defuns.join(", ")
            ),
        ));
    }
    Ok(())
}

fn validate_loaded_application(
    path: &Path,
    application: &Application,
    metadata: &SymjitApplicationMetadata<'_>,
) -> RusticolResult<()> {
    let expected_params = metadata.input_len.checked_mul(2).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT complex input count overflows usize")
    })?;
    let expected_outputs = metadata.output_len.checked_mul(2).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT complex output count overflows usize")
    })?;
    if application.count_states != 0
        || application.count_diffs != 0
        || application.count_params != expected_params
        || application.count_obs != expected_outputs
    {
        return Err(RusticolError::integrity(format!(
            "SymJIT application {} has counts states={}, params={}, outputs={}, diffs={}; expected 0, {}, {}, 0",
            path.display(),
            application.count_states,
            application.count_params,
            application.count_obs,
            application.count_diffs,
            expected_params,
            expected_outputs
        )));
    }
    validate_application_config(path, &application.config, metadata)
}

fn validate_application_config(
    path: &Path,
    config: &Config,
    metadata: &SymjitApplicationMetadata<'_>,
) -> RusticolResult<()> {
    if !config.is_complex() || !config.symbolica() {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application {} is not a complex Symbolica evaluator",
            path.display()
        )));
    }
    if config.direct() {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application {} uses direct translation; regenerate it with indirect translation",
            path.display()
        )));
    }
    let host_native = if cfg!(target_arch = "aarch64") {
        config.is_arm64()
    } else if cfg!(target_arch = "x86_64") {
        config.is_amd64()
    } else if cfg!(target_arch = "riscv64") {
        config.is_riscv64()
    } else {
        false
    };
    if !host_native && !matches!(metadata.optimization_level, 1 | 2) {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application {} uses compiler type {:?}, which is not native for this host; only optimization level 1 or 2 applications are cross-architecture portable",
            path.display(),
            config.compiler_type()
        )));
    }
    if config.opt_level() != metadata.optimization_level {
        return Err(RusticolError::integrity(format!(
            "SymJIT application {} declares optimization level {} but stores optimization level {}",
            path.display(),
            metadata.optimization_level,
            config.opt_level()
        )));
    }
    Ok(())
}

fn validate_batch_lengths(
    rows: usize,
    input_len: usize,
    output_len: usize,
    params_len: usize,
    output_buffer_len: usize,
) -> RusticolResult<()> {
    let expected_params = rows.checked_mul(input_len).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT parameter buffer length overflows usize")
    })?;
    let expected_outputs = rows.checked_mul(output_len).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT output buffer length overflows usize")
    })?;
    if params_len != expected_params {
        return Err(RusticolError::invalid_argument(format!(
            "parameter buffer has length {params_len}, expected {expected_params}"
        )));
    }
    if output_buffer_len != expected_outputs {
        return Err(RusticolError::invalid_argument(format!(
            "output buffer has length {output_buffer_len}, expected {expected_outputs}"
        )));
    }
    Ok(())
}

fn validate_aosoa_lengths(
    block_count: usize,
    lane_width: usize,
    input_len: usize,
    output_len: usize,
    params_len: usize,
    output_buffer_len: usize,
) -> RusticolResult<()> {
    let scalar_lanes = lane_width.checked_mul(2).ok_or_else(|| {
        RusticolError::invalid_argument("SymJIT AoSoA complex lane count overflows usize")
    })?;
    let expected_params = block_count
        .checked_mul(input_len)
        .and_then(|count| count.checked_mul(scalar_lanes))
        .ok_or_else(|| {
            RusticolError::invalid_argument("SymJIT AoSoA parameter buffer length overflows usize")
        })?;
    let expected_outputs = block_count
        .checked_mul(output_len)
        .and_then(|count| count.checked_mul(scalar_lanes))
        .ok_or_else(|| {
            RusticolError::invalid_argument("SymJIT AoSoA output buffer length overflows usize")
        })?;
    if params_len != expected_params {
        return Err(RusticolError::invalid_argument(format!(
            "AoSoA parameter buffer has length {params_len}, expected {expected_params}"
        )));
    }
    if output_buffer_len != expected_outputs {
        return Err(RusticolError::invalid_argument(format!(
            "AoSoA output buffer has length {output_buffer_len}, expected {expected_outputs}"
        )));
    }
    Ok(())
}

fn complex_slice_as_scalars<T>(values: &[Complex<T>]) -> &[T] {
    assert_eq!(
        std::mem::size_of::<Complex<T>>(),
        2 * std::mem::size_of::<T>(),
        "complex storage must be two adjacent scalars"
    );
    assert_eq!(
        std::mem::align_of::<Complex<T>>(),
        std::mem::align_of::<T>(),
        "complex storage must use scalar alignment"
    );
    // SAFETY: Both supported Complex implementations are repr(C) structs containing exactly
    // adjacent `re` and `im` fields. The size/alignment assertions guard this runtime contract.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len() * 2) }
}

fn complex_slice_as_scalars_mut<T>(values: &mut [Complex<T>]) -> &mut [T] {
    assert_eq!(
        std::mem::size_of::<Complex<T>>(),
        2 * std::mem::size_of::<T>(),
        "complex storage must be two adjacent scalars"
    );
    assert_eq!(
        std::mem::align_of::<Complex<T>>(),
        std::mem::align_of::<T>(),
        "complex storage must use scalar alignment"
    );
    // SAFETY: See complex_slice_as_scalars. The mutable borrow guarantees exclusive access.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<T>(), values.len() * 2) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use symjit::{Compiler, CompilerType, Storage};

    fn application_bytes() -> Vec<u8> {
        let mut config = Config::new(CompilerType::Native, 0).unwrap();
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_opt_level(3);
        config.set_simd(true);
        let mut compiler = Compiler::with_config(config);
        let instructions = r#"[[{"Add":[{"Out":0},[{"Param":0},{"Param":1}],0]}],1,[]]"#;
        let application = compiler.translate(instructions.to_string(), 2).unwrap();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn metadata<'a>(required_defuns: &'a [String]) -> SymjitApplicationMetadata<'a> {
        SymjitApplicationMetadata {
            runtime_capability: SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
            application_abi: SYMJIT_APPLICATION_STORAGE_ABI,
            input_len: 2,
            output_len: 1,
            element_layout: "complex-f64",
            batch_layout: "row-major",
            compiler_type: "native",
            translation_mode: "indirect",
            optimization_level: 3,
            word_bits: 64,
            endianness: "little",
            required_defuns,
        }
    }

    fn nonnative_config(optimization_level: u8) -> Config {
        let compiler_type = if cfg!(target_arch = "aarch64") {
            CompilerType::Amd
        } else {
            CompilerType::Arm
        };
        let mut config = Config::new(compiler_type, 0).unwrap();
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_direct(false);
        config.set_opt_level(optimization_level);
        config
    }

    fn write_application(bytes: &[u8], suffix: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "rusticol-symjit-{}-{suffix}.symjit",
            std::process::id()
        ));
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    fn direct_manifest(application_path: String) -> EvaluatorManifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability: SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string(),
            application_path,
            application_abi: SYMJIT_APPLICATION_STORAGE_ABI.to_string(),
            input_len: 2,
            output_len: 1,
            element_layout: "complex-f64".to_string(),
            batch_layout: "row-major".to_string(),
            compiler_type: "native".to_string(),
            translation_mode: "indirect".to_string(),
            optimization_level: 3,
            word_bits: 64,
            endianness: "little".to_string(),
            required_defuns: Vec::new(),
            plane_application: None,
            evaluator_state_path: Some("absent-exact-evaluator-state.bin".to_string()),
            evaluator_state_runtime_capability: Some(
                SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY.to_string(),
            ),
        }
    }

    fn pack_aosoa(
        params: &[Complex<f64>],
        batch_size: usize,
        input_len: usize,
        lane_width: usize,
    ) -> Vec<f64> {
        assert!(batch_size != 0);
        let block_count = batch_size.div_ceil(lane_width);
        let mut packed = vec![0.0; block_count * input_len * 2 * lane_width];
        for block in 0..block_count {
            for lane in 0..lane_width {
                let row = (block * lane_width + lane).min(batch_size - 1);
                for input in 0..input_len {
                    let value = params[row * input_len + input];
                    let start = (block * input_len + input) * 2 * lane_width;
                    packed[start + lane] = value.re;
                    packed[start + lane_width + lane] = value.im;
                }
            }
        }
        packed
    }

    fn unpack_aosoa(
        packed: &[f64],
        batch_size: usize,
        output_len: usize,
        lane_width: usize,
    ) -> Vec<Complex<f64>> {
        let mut outputs = Vec::with_capacity(batch_size * output_len);
        for row in 0..batch_size {
            let block = row / lane_width;
            let lane = row % lane_width;
            for output in 0..output_len {
                let start = (block * output_len + output) * 2 * lane_width;
                outputs.push(Complex::new(
                    packed[start + lane],
                    packed[start + lane_width + lane],
                ));
            }
        }
        outputs
    }

    #[test]
    fn direct_application_loads_and_evaluates_complex_batches() {
        let path = write_application(&application_bytes(), "batch");
        let mut evaluator = SymjitApplicationEvaluator::load(&path, metadata(&[])).unwrap();
        let params = [
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(-2.0, 1.5),
            Complex::new(5.0, -0.5),
        ];
        let mut outputs = [Complex::new(0.0, 0.0); 2];
        evaluator.evaluate_batch(2, &params, &mut outputs).unwrap();
        assert_eq!(outputs[0], Complex::new(4.0, 6.0));
        assert_eq!(outputs[1], Complex::new(3.0, 1.0));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn direct_aosoa_blocks_match_complete_row_major_batches() {
        let path = write_application(&application_bytes(), "aosoa-complete");
        let mut evaluator = SymjitApplicationEvaluator::load(&path, metadata(&[])).unwrap();
        let Some(lane_width) = evaluator.simd_lane_width() else {
            let _ = fs::remove_file(path);
            return;
        };
        let batch_size = 2 * lane_width;
        let mut params = Vec::with_capacity(batch_size * 2);
        for row in 0..batch_size {
            params.extend([
                Complex::new(row as f64 + 0.25, -(row as f64) - 0.5),
                Complex::new(2.0 * row as f64 + 1.0, row as f64 + 0.75),
            ]);
        }
        let mut expected = vec![Complex::new(0.0, 0.0); batch_size];
        evaluator
            .evaluate_batch(batch_size, &params, &mut expected)
            .unwrap();

        let packed = pack_aosoa(&params, batch_size, 2, lane_width);
        let mut packed_outputs = vec![0.0; 2 * lane_width * 2];
        assert!(
            evaluator
                .evaluate_aosoa_blocks(2, &packed, &mut packed_outputs)
                .unwrap()
        );
        assert_eq!(
            unpack_aosoa(&packed_outputs, batch_size, 1, lane_width),
            expected
        );

        assert!(
            evaluator
                .evaluate_aosoa_blocks(2, &packed[..packed.len() - 1], &mut packed_outputs)
                .is_err()
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn direct_aosoa_blocks_match_padded_row_major_tails() {
        let path = write_application(&application_bytes(), "aosoa-tail");
        let mut evaluator = SymjitApplicationEvaluator::load(&path, metadata(&[])).unwrap();
        let Some(lane_width) = evaluator.simd_lane_width() else {
            let _ = fs::remove_file(path);
            return;
        };
        let batch_size = lane_width + 1;
        let mut params = Vec::with_capacity(batch_size * 2);
        for row in 0..batch_size {
            params.extend([
                Complex::new(row as f64 + 1.0, row as f64 + 2.0),
                Complex::new(-(row as f64) - 3.0, 0.5 * row as f64),
            ]);
        }
        let mut expected = vec![Complex::new(0.0, 0.0); batch_size];
        evaluator
            .evaluate_batch(batch_size, &params, &mut expected)
            .unwrap();

        let block_count = batch_size.div_ceil(lane_width);
        let packed = pack_aosoa(&params, batch_size, 2, lane_width);
        let mut packed_outputs = vec![0.0; block_count * lane_width * 2];
        assert!(
            evaluator
                .evaluate_aosoa_blocks(block_count, &packed, &mut packed_outputs)
                .unwrap()
        );
        assert_eq!(
            unpack_aosoa(&packed_outputs, batch_size, 1, lane_width),
            expected
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn direct_application_pads_incomplete_simd_tails() {
        let path = write_application(&application_bytes(), "simd-tail");
        let mut evaluator = SymjitApplicationEvaluator::load(&path, metadata(&[])).unwrap();
        let lane_width = super::super::native_f64_simd_lane_width();
        let batch_size = lane_width.saturating_add(1);
        let mut params = Vec::with_capacity(batch_size * 2);
        let mut expected = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let left = Complex::new(row as f64 + 1.0, 2.0);
            let right = Complex::new(3.0, -(row as f64));
            params.extend([left, right]);
            expected.push(left + right);
        }
        let mut outputs = vec![Complex::new(0.0, 0.0); batch_size];
        evaluator
            .evaluate_batch(batch_size, &params, &mut outputs)
            .unwrap();
        assert_eq!(outputs, expected);
        if lane_width > 1 {
            assert_eq!(evaluator.input_tail_scratch.len(), lane_width * 2);
            assert_eq!(evaluator.output_tail_scratch.len(), lane_width);
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn direct_f64_group_load_does_not_read_exact_fallback_state() {
        let path = write_application(&application_bytes(), "lazy-exact");
        let root = path.parent().unwrap();
        let manifest = direct_manifest(path.file_name().unwrap().to_str().unwrap().to_string());

        let capabilities = evaluator_runtime_capabilities(&manifest).unwrap();
        assert_eq!(
            capabilities,
            BTreeSet::from([SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string()])
        );

        let mut group = EvaluatorGroup::load(&manifest, root).unwrap();
        let output = group
            .evaluate_batch(1, &[Complex::new(2.0, 3.0), Complex::new(5.0, 7.0)])
            .unwrap();
        assert_eq!(output, vec![Complex::new(7.0, 10.0)]);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn chunked_group_gathers_mapped_parent_inputs() {
        let path = write_application(&application_bytes(), "mapped-inputs");
        let root = path.parent().unwrap();
        let application_path = path.file_name().unwrap().to_str().unwrap().to_string();
        let manifest = EvaluatorManifest::Chunked {
            required_runtime_capabilities: vec![SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string()],
            input_len: Some(3),
            chunk_input_indices: Some(vec![vec![0, 2], vec![1, 2]]),
            chunks: vec![
                direct_manifest(application_path.clone()),
                direct_manifest(application_path),
            ],
        };

        let mut group = EvaluatorGroup::load(&manifest, root).unwrap();
        let output = group
            .evaluate_batch(
                2,
                &[
                    Complex::new(1.0, 0.0),
                    Complex::new(10.0, 0.0),
                    Complex::new(3.0, 0.0),
                    Complex::new(2.0, 0.0),
                    Complex::new(20.0, 0.0),
                    Complex::new(5.0, 0.0),
                ],
            )
            .unwrap();

        assert_eq!(
            output,
            vec![
                Complex::new(4.0, 0.0),
                Complex::new(13.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(25.0, 0.0),
            ]
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn chunked_group_copies_contiguous_mapped_input_spans() {
        let path = write_application(&application_bytes(), "mapped-input-spans");
        let root = path.parent().unwrap();
        let application_path = path.file_name().unwrap().to_str().unwrap().to_string();
        let manifest = EvaluatorManifest::Chunked {
            required_runtime_capabilities: vec![SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string()],
            input_len: Some(4),
            chunk_input_indices: Some(vec![vec![0, 1], vec![2, 3]]),
            chunks: vec![
                direct_manifest(application_path.clone()),
                direct_manifest(application_path),
            ],
        };

        let mut group = EvaluatorGroup::load(&manifest, root).unwrap();
        assert_eq!(
            group.input_mapping_spans,
            vec![vec![(0, 0, 2)], vec![(0, 2, 2)]]
        );
        let output = group
            .evaluate_batch(
                2,
                &[
                    Complex::new(1.0, 0.0),
                    Complex::new(3.0, 0.0),
                    Complex::new(10.0, 0.0),
                    Complex::new(30.0, 0.0),
                    Complex::new(2.0, 0.0),
                    Complex::new(5.0, 0.0),
                    Complex::new(20.0, 0.0),
                    Complex::new(50.0, 0.0),
                ],
            )
            .unwrap();

        assert_eq!(
            output,
            vec![
                Complex::new(4.0, 0.0),
                Complex::new(40.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(70.0, 0.0),
            ]
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn legacy_chunked_group_reuses_full_parent_inputs() {
        let path = write_application(&application_bytes(), "legacy-chunks");
        let root = path.parent().unwrap();
        let application_path = path.file_name().unwrap().to_str().unwrap().to_string();
        let manifest = EvaluatorManifest::Chunked {
            required_runtime_capabilities: vec![SYMJIT_APPLICATION_RUNTIME_CAPABILITY.to_string()],
            input_len: None,
            chunk_input_indices: None,
            chunks: vec![
                direct_manifest(application_path.clone()),
                direct_manifest(application_path),
            ],
        };

        let mut group = EvaluatorGroup::load(&manifest, root).unwrap();
        let output = group
            .evaluate_batch(1, &[Complex::new(2.0, 3.0), Complex::new(5.0, 7.0)])
            .unwrap();

        assert_eq!(
            output,
            vec![Complex::new(7.0, 10.0), Complex::new(7.0, 10.0)]
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn direct_f64_group_loads_without_exact_fallback_metadata() {
        let path = write_application(&application_bytes(), "no-exact-fallback");
        let root = path.parent().unwrap();
        let mut manifest = direct_manifest(path.file_name().unwrap().to_str().unwrap().to_string());
        let EvaluatorManifest::SymjitApplication {
            evaluator_state_path,
            evaluator_state_runtime_capability,
            ..
        } = &mut manifest
        else {
            unreachable!()
        };
        *evaluator_state_path = None;
        *evaluator_state_runtime_capability = None;

        let mut group = EvaluatorGroup::load(&manifest, root).unwrap();
        let output = group
            .evaluate_batch(1, &[Complex::new(2.0, 3.0), Complex::new(5.0, 7.0)])
            .unwrap();

        assert_eq!(output, vec![Complex::new(7.0, 10.0)]);
        let _ = fs::remove_file(path);
    }

    #[cfg(not(feature = "symbolica-runtime"))]
    #[test]
    fn unsupported_legacy_jit_evaluator_is_rejected_before_payload_access() {
        let manifest = EvaluatorManifest::Jit {
            runtime_capability: SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY.to_string(),
            input_len: 2,
            output_len: 1,
            evaluator_state_path: "absent-legacy-jit-state.bin".to_string(),
        };

        let error = match EvaluatorGroup::load(&manifest, Path::new("/absent-artifact-root")) {
            Ok(_) => panic!("unsupported capability must win over absent payload"),
            Err(error) => error,
        };
        assert_eq!(
            error.kind(),
            crate::RusticolErrorKind::UnsupportedRuntimeCapability
        );
    }

    #[cfg(not(feature = "f64-compiled"))]
    #[test]
    fn unsupported_compiled_evaluators_are_rejected_before_payload_access() {
        let manifests = [
            EvaluatorManifest::CompiledComplex {
                runtime_capability: SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY.to_string(),
                function_name: "evaluate".to_string(),
                input_len: 2,
                output_len: 1,
                library_path: "absent-cpp-library.so".to_string(),
                evaluator_state_path: None,
                number_type: "complex".to_string(),
                native_direct_application: None,
            },
            EvaluatorManifest::CompiledComplex {
                runtime_capability: SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY.to_string(),
                function_name: "evaluate".to_string(),
                input_len: 2,
                output_len: 1,
                library_path: "absent-asm-library.so".to_string(),
                evaluator_state_path: None,
                number_type: "complex".to_string(),
                native_direct_application: None,
            },
        ];

        for manifest in manifests {
            let error = match EvaluatorGroup::load(&manifest, Path::new("/absent-artifact-root")) {
                Ok(_) => panic!("unsupported capability must win over absent payload"),
                Err(error) => error,
            };
            assert_eq!(
                error.kind(),
                crate::RusticolErrorKind::UnsupportedRuntimeCapability
            );
        }
    }

    #[test]
    fn malformed_application_bytes_are_rejected() {
        let path = write_application(b"not-a-symjit-application", "malformed");
        let error = SymjitApplicationEvaluator::load(&path, metadata(&[])).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn cross_architecture_application_accepts_o1_o2_only() {
        for optimization_level in [1, 2] {
            let mut metadata = metadata(&[]);
            metadata.optimization_level = optimization_level;
            validate_application_config(
                Path::new("portable.symjit"),
                &nonnative_config(optimization_level),
                &metadata,
            )
            .unwrap();
        }

        for optimization_level in [0, 3] {
            let mut metadata = metadata(&[]);
            metadata.optimization_level = optimization_level;
            let error = validate_application_config(
                Path::new("target-specific.symjit"),
                &nonnative_config(optimization_level),
                &metadata,
            )
            .unwrap_err();
            assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
            assert!(error.to_string().contains("optimization level 1 or 2"));
        }
    }

    #[test]
    fn symjit_panic_payloads_retain_dependency_context() {
        assert_eq!(
            panic_detail(Box::new(String::from(
                "architecture-specific register allocation"
            ))),
            "architecture-specific register allocation"
        );
    }

    #[test]
    fn direct_application_requires_complex_f64_row_major_layout() {
        let path = write_application(&application_bytes(), "layout");

        let mut wrong_element_layout = metadata(&[]);
        wrong_element_layout.element_layout = "complex-f32";
        let error = SymjitApplicationEvaluator::load(&path, wrong_element_layout).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);

        let mut wrong_batch_layout = metadata(&[]);
        wrong_batch_layout.batch_layout = "column-major";
        let error = SymjitApplicationEvaluator::load(&path, wrong_batch_layout).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn capability_counts_and_external_functions_are_validated() {
        let bytes = application_bytes();
        let path = write_application(&bytes, "metadata");

        let mut wrong_counts = metadata(&[]);
        wrong_counts.output_len = 2;
        let error = SymjitApplicationEvaluator::load(&path, wrong_counts).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);

        let functions = vec!["model_function".to_string()];
        let error = SymjitApplicationEvaluator::load(&path, metadata(&functions)).unwrap_err();
        assert_eq!(
            error.kind(),
            crate::RusticolErrorKind::UnsupportedRuntimeCapability
        );

        let mut wrong_capability = metadata(&[]);
        wrong_capability.runtime_capability = SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY;
        let error = SymjitApplicationEvaluator::load(&path, wrong_capability).unwrap_err();
        assert_eq!(
            error.kind(),
            crate::RusticolErrorKind::UnsupportedRuntimeCapability
        );
        let _ = fs::remove_file(path);
    }
}
