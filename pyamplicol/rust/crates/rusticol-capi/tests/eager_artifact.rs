// SPDX-License-Identifier: 0BSD

use rusticol_capi::{
    RUSTICOL_STATUS_OK, RusticolRuntimeHandle, rusticol_last_error_message,
    rusticol_runtime_artifact_id, rusticol_runtime_color_id, rusticol_runtime_evaluate_f64,
    rusticol_runtime_evaluate_resolved_f64, rusticol_runtime_execution_mode, rusticol_runtime_free,
    rusticol_runtime_get_model_parameter, rusticol_runtime_helicity_id, rusticol_runtime_load,
    rusticol_runtime_resolved_shape, rusticol_runtime_set_model_parameter,
};
use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::ptr;

fn last_error() -> String {
    let mut required = 0;
    // SAFETY: This is a zero-capacity query with writable required-size storage.
    let status = unsafe { rusticol_last_error_message(ptr::null_mut(), 0, &mut required) };
    if status != RUSTICOL_STATUS_OK || required == 0 {
        return format!("could not query Rusticol error (status {status})");
    }
    let mut buffer = vec![0 as c_char; required];
    // SAFETY: The output buffer has the queried capacity.
    let status =
        unsafe { rusticol_last_error_message(buffer.as_mut_ptr(), buffer.len(), &mut required) };
    if status != RUSTICOL_STATUS_OK {
        return format!("could not copy Rusticol error (status {status})");
    }
    // SAFETY: A successful Rusticol string getter always writes a trailing NUL.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn load_artifact(root: &Path) -> *mut RusticolRuntimeHandle {
    let root = CString::new(root.to_string_lossy().as_bytes()).expect("artifact C path");
    let mut handle: *mut RusticolRuntimeHandle = ptr::null_mut();
    // SAFETY: The path and output handle storage remain valid for the call.
    let status =
        unsafe { rusticol_runtime_load(root.as_ptr(), ptr::null(), ptr::null(), &mut handle) };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert!(!handle.is_null());
    handle
}

fn execution_mode(handle: *mut RusticolRuntimeHandle) -> String {
    let mut required = 0;
    // SAFETY: This is a zero-capacity query against a live handle.
    let status =
        unsafe { rusticol_runtime_execution_mode(handle, ptr::null_mut(), 0, &mut required) };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    let mut mode = vec![0 as c_char; required];
    // SAFETY: The output buffer has the queried capacity and the handle remains live.
    let status = unsafe {
        rusticol_runtime_execution_mode(handle, mode.as_mut_ptr(), mode.len(), &mut required)
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    // SAFETY: A successful Rusticol string getter always writes a trailing NUL.
    unsafe { CStr::from_ptr(mode.as_ptr()) }
        .to_str()
        .unwrap()
        .to_owned()
}

fn indexed_string(
    handle: *mut RusticolRuntimeHandle,
    index: usize,
    getter: unsafe extern "C" fn(
        *const RusticolRuntimeHandle,
        usize,
        *mut c_char,
        usize,
        *mut usize,
    ) -> i32,
) -> CString {
    let mut required = 0;
    // SAFETY: This is a zero-capacity query against a live handle.
    let status = unsafe { getter(handle, index, ptr::null_mut(), 0, &mut required) };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    let mut buffer = vec![0 as c_char; required];
    // SAFETY: The output buffer has the queried capacity and the handle remains live.
    let status = unsafe {
        getter(
            handle,
            index,
            buffer.as_mut_ptr(),
            buffer.len(),
            &mut required,
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    // SAFETY: A successful Rusticol string getter always writes a trailing NUL.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_owned()
}

fn free_runtime(handle: *mut RusticolRuntimeHandle) {
    // SAFETY: The handle is live and is consumed exactly once here.
    assert_eq!(
        unsafe { rusticol_runtime_free(handle) },
        RUSTICOL_STATUS_OK,
        "{}",
        last_error()
    );
}

#[test]
fn generated_compiled_artifact_reports_compiled_execution_mode() {
    let Some(root) = std::env::var_os("RUSTICOL_COMPILED_ARTIFACT") else {
        return;
    };
    let handle = load_artifact(Path::new(&root));
    assert_eq!(execution_mode(handle), "compiled");
    free_runtime(handle);
}

#[test]
fn generated_eager_artifact_loads_and_evaluates_through_the_c_abi() {
    let Some(root) = std::env::var_os("RUSTICOL_EAGER_ARTIFACT") else {
        return;
    };
    let root = Path::new(&root);
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("artifact.json")).expect("read eager artifact manifest"),
    )
    .expect("parse eager artifact manifest");
    let process_id = manifest["default_process_id"]
        .as_str()
        .expect("eager artifact default process");
    let validation: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            root.join("processes")
                .join(process_id)
                .join("validation-momenta.json"),
        )
        .expect("read eager validation momenta"),
    )
    .expect("parse eager validation momenta");
    let momenta = validation["points"][0]
        .as_array()
        .expect("validation point")
        .iter()
        .flat_map(|leg| {
            leg["momentum"]
                .as_array()
                .expect("validation momentum")
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .expect("string momentum component")
                        .parse::<f64>()
                        .expect("f64 momentum component")
                })
        })
        .collect::<Vec<_>>();

    let handle = load_artifact(root);
    assert_eq!(execution_mode(handle), "eager");

    let mut required = 0;
    // SAFETY: The handle is live and required-size storage is writable.
    assert_eq!(
        unsafe { rusticol_runtime_artifact_id(handle, ptr::null_mut(), 0, &mut required) },
        RUSTICOL_STATUS_OK
    );
    let mut identity = vec![0 as c_char; required];
    // SAFETY: The output buffer has exactly the queried capacity.
    assert_eq!(
        unsafe {
            rusticol_runtime_artifact_id(
                handle,
                identity.as_mut_ptr(),
                identity.len(),
                &mut required,
            )
        },
        RUSTICOL_STATUS_OK
    );
    // SAFETY: A successful string getter includes a trailing NUL.
    let identity = unsafe { CStr::from_ptr(identity.as_ptr()) }
        .to_str()
        .unwrap();
    assert_eq!(identity, manifest["artifact_id"].as_str().unwrap());

    let mut output = [f64::NAN];
    // SAFETY: Every input and output buffer remains live and has the declared length.
    let status = unsafe {
        rusticol_runtime_evaluate_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            1,
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert!(output[0].is_finite());

    let mut helicity_count = 0;
    let mut color_count = 0;
    // SAFETY: Null selector arrays request complete coverage and shape outputs are writable.
    let status = unsafe {
        rusticol_runtime_resolved_shape(
            handle,
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut helicity_count,
            &mut color_count,
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert!(helicity_count > 0);
    assert!(color_count > 0);
    let mut resolved = vec![f64::NAN; helicity_count * color_count];
    // SAFETY: Every input, output, and shape buffer remains live and has the declared length.
    let status = unsafe {
        rusticol_runtime_evaluate_resolved_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            1,
            ptr::null(),
            0,
            ptr::null(),
            0,
            resolved.as_mut_ptr(),
            resolved.len(),
            &mut helicity_count,
            &mut color_count,
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert!(resolved.iter().all(|value| value.is_finite()));
    let resolved_sum = resolved.iter().sum::<f64>();
    let tolerance = 1.0e-15 + 1.0e-12 * output[0].abs();
    assert!(
        (resolved_sum - output[0]).abs() <= tolerance,
        "resolved sum {resolved_sum:.17e} does not reproduce total {:.17e}",
        output[0],
    );

    let helicity = indexed_string(handle, 0, rusticol_runtime_helicity_id);
    let color = indexed_string(handle, 0, rusticol_runtime_color_id);
    let helicity_ids = [helicity.as_ptr()];
    let color_ids = [color.as_ptr()];
    let mut selected_helicity_count = 0;
    let mut selected_color_count = 0;
    // SAFETY: Selector arrays and shape outputs remain valid for the call.
    let status = unsafe {
        rusticol_runtime_resolved_shape(
            handle,
            helicity_ids.as_ptr(),
            helicity_ids.len(),
            color_ids.as_ptr(),
            color_ids.len(),
            &mut selected_helicity_count,
            &mut selected_color_count,
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert_eq!((selected_helicity_count, selected_color_count), (1, 1));
    let mut selected = [f64::NAN];
    // SAFETY: Every input, selector, output, and shape buffer has the declared length.
    let status = unsafe {
        rusticol_runtime_evaluate_resolved_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            1,
            helicity_ids.as_ptr(),
            helicity_ids.len(),
            color_ids.as_ptr(),
            color_ids.len(),
            selected.as_mut_ptr(),
            selected.len(),
            &mut selected_helicity_count,
            &mut selected_color_count,
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert!(selected[0].is_finite());

    let baseline = output[0];
    let parameter = CString::new("normalization.alpha_ew").unwrap();
    // SAFETY: The parameter name and live runtime handle remain valid for the call.
    let status =
        unsafe { rusticol_runtime_set_model_parameter(handle, parameter.as_ptr(), 0.006, 0.0) };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    let mut current_real = f64::NAN;
    let mut current_imaginary = f64::NAN;
    // SAFETY: Both outputs and the parameter name remain live for this call.
    let status = unsafe {
        rusticol_runtime_get_model_parameter(
            handle,
            parameter.as_ptr(),
            &mut current_real,
            &mut current_imaginary,
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert_eq!((current_real, current_imaginary), (0.006, 0.0));
    // SAFETY: Every input and output buffer remains live and has the declared length.
    let status = unsafe {
        rusticol_runtime_evaluate_f64(
            handle,
            momenta.as_ptr(),
            momenta.len(),
            1,
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert_eq!(status, RUSTICOL_STATUS_OK, "{}", last_error());
    assert!(output[0].is_finite());
    assert_ne!(output[0], baseline);

    free_runtime(handle);
}
