// SPDX-License-Identifier: 0BSD

use rusticol_core::{
    NativeCorrelatedRequest, NativeOnTheFlyWarmUpEvent, NativeOnTheFlyWarmUpEventKind,
    NativeOnTheFlyWarmUpResult, NativeOnTheFlyWarmUpStage, NativeRuntime,
    NativeSpinCorrelationVectors, RusticolError, RusticolErrorKind, supported_runtime_capabilities,
};
use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_double, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::slice;

mod correlated;

#[allow(non_camel_case_types)]
type size_t = usize;

pub const RUSTICOL_STATUS_OK: c_int = 0;
pub const RUSTICOL_STATUS_INVALID_ARGUMENT: c_int = 1;
pub const RUSTICOL_STATUS_BUFFER_TOO_SMALL: c_int = 2;
pub const RUSTICOL_STATUS_RUNTIME_ERROR: c_int = 3;
pub const RUSTICOL_STATUS_PANIC: c_int = 4;

pub const RUSTICOL_WARM_UP_EVENT_START: u32 = 0;
pub const RUSTICOL_WARM_UP_EVENT_UPDATE: u32 = 1;
pub const RUSTICOL_WARM_UP_EVENT_END: u32 = 2;

pub const RUSTICOL_WARM_UP_STAGE_PROCESS_PREPARATION: u32 = 0;
pub const RUSTICOL_WARM_UP_STAGE_QUERY_FAMILY: u32 = 1;
pub const RUSTICOL_WARM_UP_STAGE_FAMILY_FINALIZATION: u32 = 2;
pub const RUSTICOL_WARM_UP_STAGE_FIRST_EVALUATION: u32 = 3;

/// Fixed-layout progress snapshot passed to native warm-up observers.
///
/// `current_rss_bytes` and `peak_rss_bytes` are meaningful only when the
/// corresponding `*_available` field is nonzero. The event is borrowed and is
/// valid only for the duration of the callback.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RusticolWarmUpProgressEvent {
    pub schema_version: u32,
    pub kind: u32,
    pub stage: u32,
    pub reserved: u32,
    pub completed: u64,
    pub total: u64,
    pub elapsed_seconds: c_double,
    pub current_rss_bytes: u64,
    pub peak_rss_bytes: u64,
    pub workers: u64,
    pub current_rss_available: u32,
    pub peak_rss_available: u32,
}

/// Fixed-layout result from one explicit binary64 on-the-fly warm-up.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RusticolWarmUpResult {
    pub schema_version: u32,
    pub reserved: u32,
    pub elapsed_seconds: c_double,
    pub query_count: u64,
    pub warmed_query_count: u64,
    pub current_rss_bytes: u64,
    pub peak_rss_bytes: u64,
    pub current_rss_available: u32,
    pub peak_rss_available: u32,
    pub already_warm: u32,
    pub first_evaluation_completed: u32,
}

/// Return nonzero to continue or zero to request cancellation.
///
/// The callback runs on the coordinating caller thread. The terminal
/// first-evaluation `END` event is an informational post-commit notification;
/// returning zero for that one event does not undo a completed warm-up.
pub type RusticolWarmUpProgressCallback = Option<
    unsafe extern "C" fn(
        event: *const RusticolWarmUpProgressEvent,
        user_data: *mut c_void,
    ) -> c_int,
>;

pub struct RusticolRuntimeHandle {
    runtime: NativeRuntime,
}

/// One borrowed literal complex spin-vector batch; see rusticol.h for layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RusticolSpinCorrelationVector {
    pub leg: size_t,
    pub point_count: size_t,
    pub components: *const c_double,
    pub component_count: size_t,
}

/// One ordered colour insertion and either explicit or inherited spin vectors.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RusticolCorrelatedRequest {
    pub color_correlation: *const c_char,
    pub spin_vectors: *const RusticolSpinCorrelationVector,
    pub spin_vector_count: size_t,
    pub use_default_spin_vectors: u32,
}

struct AbiError {
    status: c_int,
    message: String,
}

type AbiResult<T> = Result<T, AbiError>;

impl From<RusticolError> for AbiError {
    fn from(error: RusticolError) -> Self {
        let status = match error.kind() {
            RusticolErrorKind::InvalidArgument
            | RusticolErrorKind::Selector
            | RusticolErrorKind::ModelParameter
            | RusticolErrorKind::UnsupportedPrecision => RUSTICOL_STATUS_INVALID_ARGUMENT,
            _ => RUSTICOL_STATUS_RUNTIME_ERROR,
        };
        Self {
            status,
            message: error.to_string(),
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty CString"));
}

fn sanitize_c_string(message: impl AsRef<str>) -> CString {
    let sanitized = message.as_ref().replace('\0', "\\0");
    CString::new(sanitized)
        .unwrap_or_else(|_| CString::new("Rusticol error").expect("literal CString"))
}

fn set_last_error(message: impl AsRef<str>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = sanitize_c_string(message));
}

fn guard(operation: impl FnOnce() -> AbiResult<()>) -> c_int {
    finish_guard(catch_unwind(AssertUnwindSafe(operation)))
}

fn finish_guard(result: Result<AbiResult<()>, Box<dyn Any + Send>>) -> c_int {
    match result {
        Ok(Ok(())) => RUSTICOL_STATUS_OK,
        Ok(Err(error)) => {
            set_last_error(&error.message);
            error.status
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown Rust panic");
            set_last_error(format!("Rusticol panic: {message}"));
            RUSTICOL_STATUS_PANIC
        }
    }
}

fn abi_error(status: c_int, message: impl Into<String>) -> AbiError {
    AbiError {
        status,
        message: message.into(),
    }
}

fn invalid(message: impl Into<String>) -> AbiError {
    abi_error(RUSTICOL_STATUS_INVALID_ARGUMENT, message)
}

fn buffer_too_small(message: impl Into<String>) -> AbiError {
    abi_error(RUSTICOL_STATUS_BUFFER_TOO_SMALL, message)
}

unsafe fn required_handle<'a>(
    handle: *const RusticolRuntimeHandle,
) -> AbiResult<&'a RusticolRuntimeHandle> {
    if handle.is_null() {
        return Err(invalid("Rusticol runtime handle is null"));
    }
    // SAFETY: The caller promises that a non-null handle was returned by rusticol_runtime_load
    // and remains alive for the duration of this call.
    Ok(unsafe { &*handle })
}

unsafe fn required_handle_mut<'a>(
    handle: *mut RusticolRuntimeHandle,
) -> AbiResult<&'a mut RusticolRuntimeHandle> {
    if handle.is_null() {
        return Err(invalid("Rusticol runtime handle is null"));
    }
    // SAFETY: The ABI documents handles as mutable and non-concurrently callable.
    Ok(unsafe { &mut *handle })
}

unsafe fn optional_c_string<'a>(value: *const c_char) -> AbiResult<Option<&'a str>> {
    if value.is_null() {
        return Ok(None);
    }
    // SAFETY: The caller supplies a NUL-terminated string for non-null pointers.
    let value = unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|error| invalid(format!("C string is not valid UTF-8: {error}")))?;
    Ok(Some(value))
}

unsafe fn required_c_string<'a>(value: *const c_char, name: &str) -> AbiResult<&'a str> {
    unsafe { optional_c_string(value) }?.ok_or_else(|| invalid(format!("{name} is null")))
}

unsafe fn write_size(value: usize, output: *mut size_t, name: &str) -> AbiResult<()> {
    if output.is_null() {
        return Err(invalid(format!("{name} is null")));
    }
    // SAFETY: The caller supplies writable storage for one size_t.
    unsafe { *output = value as size_t };
    Ok(())
}

unsafe fn write_i32(value: i32, output: *mut i32, name: &str) -> AbiResult<()> {
    if output.is_null() {
        return Err(invalid(format!("{name} is null")));
    }
    // SAFETY: The caller supplies writable storage for one i32.
    unsafe { *output = value };
    Ok(())
}

unsafe fn write_string(
    value: &str,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> AbiResult<()> {
    let bytes = value.as_bytes();
    let required_capacity = bytes
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid("string length overflow"))?;
    unsafe { write_size(required_capacity, required, "required string capacity")? };
    if buffer.is_null() {
        if capacity == 0 {
            return Ok(());
        }
        return Err(invalid("string output buffer is null"));
    }
    if capacity < required_capacity {
        return Err(buffer_too_small(format!(
            "string output buffer has capacity {capacity}, requires {required_capacity}"
        )));
    }
    // SAFETY: Capacity was checked and the source does not overlap the caller's buffer.
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), bytes.len());
        *buffer.add(bytes.len()) = 0;
    }
    Ok(())
}

unsafe fn read_selector_ids(
    values: *const *const c_char,
    count: size_t,
    name: &str,
) -> AbiResult<Option<Vec<String>>> {
    if count == 0 {
        return Ok(None);
    }
    if values.is_null() {
        return Err(invalid(format!("{name} array is null")));
    }
    // SAFETY: The caller supplies count string pointers.
    let values = unsafe { slice::from_raw_parts(values, count) };
    let mut result = Vec::with_capacity(count);
    for (index, value) in values.iter().enumerate() {
        result.push(unsafe { required_c_string(*value, &format!("{name}[{index}]")) }?.to_string());
    }
    Ok(Some(result))
}

unsafe fn read_f64_slice<'a>(
    values: *const c_double,
    count: size_t,
    name: &str,
) -> AbiResult<&'a [f64]> {
    if count == 0 {
        return Err(invalid(format!("{name} must not be empty")));
    }
    if values.is_null() {
        return Err(invalid(format!("{name} is null")));
    }
    // SAFETY: The caller supplies count readable f64 values.
    Ok(unsafe { slice::from_raw_parts(values, count) })
}

unsafe fn read_optional_u32_slice<'a>(
    values: *const u32,
    count: size_t,
    expected_count: usize,
    name: &str,
) -> AbiResult<Option<&'a [u32]>> {
    if count == 0 {
        return Ok(None);
    }
    if values.is_null() {
        return Err(invalid(format!("{name} is null")));
    }
    if count != expected_count {
        return Err(invalid(format!(
            "{name} has length {count}, expected {expected_count} (one selector per point)"
        )));
    }
    // SAFETY: The caller supplies count readable u32 values.
    Ok(Some(unsafe { slice::from_raw_parts(values, count) }))
}

unsafe fn write_f64_slice(
    values: &[f64],
    output: *mut c_double,
    capacity: size_t,
    name: &str,
) -> AbiResult<()> {
    if output.is_null() {
        return Err(invalid(format!("{name} is null")));
    }
    if capacity < values.len() {
        return Err(buffer_too_small(format!(
            "{name} has capacity {capacity}, requires {}",
            values.len()
        )));
    }
    // SAFETY: Capacity was checked and the source cannot overlap the caller-owned output.
    unsafe { ptr::copy_nonoverlapping(values.as_ptr(), output, values.len()) };
    Ok(())
}

fn validate_f64_output(
    output: *mut c_double,
    capacity: size_t,
    required: usize,
    name: &str,
) -> AbiResult<()> {
    if output.is_null() {
        return Err(invalid(format!("{name} is null")));
    }
    if capacity < required {
        return Err(buffer_too_small(format!(
            "{name} has capacity {capacity}, requires {required}"
        )));
    }
    Ok(())
}

fn warm_up_event_kind(kind: NativeOnTheFlyWarmUpEventKind) -> u32 {
    match kind {
        NativeOnTheFlyWarmUpEventKind::Start => RUSTICOL_WARM_UP_EVENT_START,
        NativeOnTheFlyWarmUpEventKind::Update => RUSTICOL_WARM_UP_EVENT_UPDATE,
        NativeOnTheFlyWarmUpEventKind::End => RUSTICOL_WARM_UP_EVENT_END,
    }
}

fn warm_up_stage(stage: NativeOnTheFlyWarmUpStage) -> u32 {
    match stage {
        NativeOnTheFlyWarmUpStage::ProcessPreparation => RUSTICOL_WARM_UP_STAGE_PROCESS_PREPARATION,
        NativeOnTheFlyWarmUpStage::QueryFamily => RUSTICOL_WARM_UP_STAGE_QUERY_FAMILY,
        NativeOnTheFlyWarmUpStage::FamilyFinalization => RUSTICOL_WARM_UP_STAGE_FAMILY_FINALIZATION,
        NativeOnTheFlyWarmUpStage::FirstEvaluation => RUSTICOL_WARM_UP_STAGE_FIRST_EVALUATION,
    }
}

fn warm_up_progress_event(event: &NativeOnTheFlyWarmUpEvent) -> RusticolWarmUpProgressEvent {
    RusticolWarmUpProgressEvent {
        schema_version: event.schema_version,
        kind: warm_up_event_kind(event.kind),
        stage: warm_up_stage(event.stage),
        reserved: 0,
        completed: event.completed,
        total: event.total,
        elapsed_seconds: event.elapsed_seconds,
        current_rss_bytes: event.current_rss_bytes.unwrap_or_default(),
        peak_rss_bytes: event.peak_rss_bytes.unwrap_or_default(),
        workers: event.workers,
        current_rss_available: u32::from(event.current_rss_bytes.is_some()),
        peak_rss_available: u32::from(event.peak_rss_bytes.is_some()),
    }
}

fn warm_up_result(result: NativeOnTheFlyWarmUpResult) -> RusticolWarmUpResult {
    RusticolWarmUpResult {
        schema_version: result.schema_version,
        reserved: 0,
        elapsed_seconds: result.elapsed_seconds,
        query_count: result.query_count,
        warmed_query_count: result.warmed_query_count,
        current_rss_bytes: result.current_rss_bytes.unwrap_or_default(),
        peak_rss_bytes: result.peak_rss_bytes.unwrap_or_default(),
        current_rss_available: u32::from(result.current_rss_bytes.is_some()),
        peak_rss_available: u32::from(result.peak_rss_bytes.is_some()),
        already_warm: u32::from(result.already_warm),
        first_evaluation_completed: u32::from(result.first_evaluation_completed),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rusticol_abi_version() -> u32 {
    catch_unwind(|| NativeRuntime::ABI_VERSION).unwrap_or(0)
}

/// Returns the supported runtime capabilities as JSON.
///
/// # Safety
///
/// If non-null, `required` must be writable for one `size_t`. If non-null, `buffer` must be
/// writable for `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_supported_runtime_capabilities_json(
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        let json = serde_json::to_string(&supported_runtime_capabilities()).map_err(|error| {
            AbiError::from(RusticolError::serialization(format!(
                "could not serialize supported runtime capabilities: {error}"
            )))
        })?;
        // SAFETY: Pointer validation and copying are performed by write_string.
        unsafe { write_string(&json, buffer, capacity, required) }
    })
}

/// Copies the calling thread's last Rusticol error message.
///
/// # Safety
///
/// If non-null, `required` must be writable for one `size_t`. If non-null, `buffer` must be
/// writable for `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_last_error_message(
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    match catch_unwind(AssertUnwindSafe(|| {
        LAST_ERROR.with(|slot| {
            let value = slot.borrow();
            // SAFETY: Pointer validation and copying are performed by write_string.
            unsafe {
                write_string(
                    value.to_str().unwrap_or("Rusticol error"),
                    buffer,
                    capacity,
                    required,
                )
            }
        })
    })) {
        Ok(Ok(())) => RUSTICOL_STATUS_OK,
        Ok(Err(error)) => {
            set_last_error(&error.message);
            error.status
        }
        Err(_) => {
            set_last_error("Rusticol panic while retrieving the last error message");
            RUSTICOL_STATUS_PANIC
        }
    }
}

/// Loads a Rusticol runtime and returns an owning opaque handle.
///
/// # Safety
///
/// Every non-null string pointer must reference a readable NUL-terminated string for the duration
/// of the call. If non-null, `output` must be writable for one handle pointer. A handle returned
/// through `output` must eventually be released exactly once with [`rusticol_runtime_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_load(
    process_dir: *const c_char,
    process_key: *const c_char,
    model_parameters_path: *const c_char,
    output: *mut *mut RusticolRuntimeHandle,
) -> c_int {
    guard(|| {
        if output.is_null() {
            return Err(invalid("runtime output handle is null"));
        }
        // SAFETY: The caller supplies writable handle storage.
        unsafe { *output = ptr::null_mut() };
        // SAFETY: String pointers follow the ABI contract; output is checked above.
        let process_dir = unsafe { required_c_string(process_dir, "process_dir") }?;
        let process_key = unsafe { optional_c_string(process_key) }?;
        let model_parameters = unsafe { optional_c_string(model_parameters_path) }?;
        let runtime =
            NativeRuntime::load(process_dir, process_key, model_parameters.map(Path::new))?;
        let boxed = Box::new(RusticolRuntimeHandle { runtime });
        // SAFETY: output points to writable handle storage supplied by the caller.
        unsafe { *output = Box::into_raw(boxed) };
        Ok(())
    })
}

/// Saves the completed, retained OTF recursion cache to a file.
///
/// # Safety
///
/// `handle` must be a live runtime handle, and `path` must reference a readable
/// NUL-terminated UTF-8 string. No other call on this handle may overlap this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_save(
    handle: *const RusticolRuntimeHandle,
    path: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: The handle and string follow the ABI contract above.
        let runtime = unsafe { required_handle(handle) }?;
        let path = unsafe { required_c_string(path, "cache path") }?;
        runtime.runtime.save(path)?;
        Ok(())
    })
}

/// Restores a saved OTF recursion cache into an already-loaded matching process.
///
/// # Safety
///
/// `handle` must be a live runtime handle, and `path` must reference a readable
/// NUL-terminated UTF-8 string. No other call on this handle may overlap this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_load_cache(
    handle: *mut RusticolRuntimeHandle,
    path: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: The handle and string follow the ABI contract above.
        let runtime = unsafe { required_handle_mut(handle) }?;
        let path = unsafe { required_c_string(path, "cache path") }?;
        runtime.runtime.load_cache(path)?;
        Ok(())
    })
}

/// Releases a Rusticol runtime handle.
///
/// # Safety
///
/// `handle` must be null or a live handle returned by [`rusticol_runtime_load`] that has not
/// already been freed. No other access to a non-null handle may overlap this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_free(handle: *mut RusticolRuntimeHandle) -> c_int {
    guard(|| {
        if !handle.is_null() {
            // SAFETY: The caller must free a live handle exactly once.
            unsafe { drop(Box::from_raw(handle)) };
        }
        Ok(())
    })
}

unsafe fn runtime_string(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
    get: impl FnOnce(&NativeRuntime) -> AbiResult<String>,
) -> c_int {
    guard(|| {
        // SAFETY: Handle and output pointers are validated by helpers.
        let handle = unsafe { required_handle(handle) }?;
        let value = get(&handle.runtime)?;
        unsafe { write_string(&value, buffer, capacity, required) }
    })
}

/// Copies the existing content identity of the authenticated artifact manifest.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_artifact_id(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            Ok(runtime.artifact_id().to_owned())
        })
    }
}

/// Copies the runtime metadata as JSON.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_metadata_json(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            Ok(runtime.metadata_json()?)
        })
    }
}

/// Copies the runtime physics metadata as JSON.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_physics_json(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            Ok(runtime.physics_json()?)
        })
    }
}

/// Copies the runtime process name.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_process(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            Ok(runtime.metadata().process)
        })
    }
}

/// Copies the runtime process key.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_process_key(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            Ok(runtime.metadata().process_key)
        })
    }
}

/// Copies the stable representative process key implementing this selection.
///
/// # Safety
///
/// Pointer requirements are identical to [`rusticol_runtime_process_key`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_representative_process_key(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            Ok(runtime.metadata().representative_process_key)
        })
    }
}

/// Copies the runtime color-accuracy label.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_color_accuracy(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            Ok(runtime.metadata().color_accuracy)
        })
    }
}

/// Writes the number of external particles.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `output` must be writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_external_count(
    handle: *const RusticolRuntimeHandle,
    output: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        unsafe { write_size(handle.runtime.external_count(), output, "external count") }
    })
}

/// Copies the runtime execution mode (`compiled`, `eager`, or `recurrence`).
///
/// # Safety
///
/// `handle` must be a live handle returned by [`rusticol_runtime_load`]. If non-null,
/// `required` must be writable for one `size_t`. If non-null, `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_execution_mode(
    handle: *const RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: The ABI requires a live handle for this call.
        let handle = unsafe { required_handle(handle) }?;
        let execution_mode = handle.runtime.metadata().execution_mode;
        // SAFETY: Pointer validation and copying are performed by write_string.
        unsafe { write_string(&execution_mode, buffer, capacity, required) }
    })
}

/// Writes the PDG identifier for one external particle.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `output` must be writable for one `i32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_external_pdg(
    handle: *const RusticolRuntimeHandle,
    index: size_t,
    output: *mut i32,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        let particles = handle.runtime.external_particles()?;
        let particle = particles
            .get(index)
            .ok_or_else(|| invalid(format!("external particle index {index} is out of range")))?;
        unsafe { write_i32(particle.pdg, output, "external PDG output") }
    })
}

/// Copies the representative-index to public-index external permutation.
///
/// # Safety
///
/// A non-null `handle` must remain live. If non-null, `required` must be writable for one
/// `size_t`; `output` must be writable for `capacity` `size_t` values. A null output is valid
/// only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_external_permutation(
    handle: *const RusticolRuntimeHandle,
    output: *mut size_t,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        let permutation = handle.runtime.metadata().external_permutation;
        unsafe { write_size(permutation.len(), required, "external permutation length")? };
        if output.is_null() {
            if capacity == 0 {
                return Ok(());
            }
            return Err(invalid("external permutation output is null"));
        }
        if capacity < permutation.len() {
            return Err(buffer_too_small(format!(
                "external permutation capacity {capacity} is smaller than {}",
                permutation.len()
            )));
        }
        // SAFETY: Capacity was checked and usize matches C size_t.
        unsafe { ptr::copy_nonoverlapping(permutation.as_ptr(), output, permutation.len()) };
        Ok(())
    })
}

/// Writes the exact factor and charge conjugation relating the selected process to the
/// representative that evaluates it: `ME = numerator / denominator * ME_representative`.
/// Selections other than non-generated process-group members report `1/1` and `0`.
///
/// # Safety
///
/// A non-null `handle` must remain live. Each output must be writable for one value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_process_group_relation(
    handle: *const RusticolRuntimeHandle,
    numerator: *mut u64,
    denominator: *mut u64,
    conjugated: *mut i32,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        if numerator.is_null() || denominator.is_null() {
            return Err(invalid("process-group factor output is null"));
        }
        let factor = handle.runtime.process_group_factor();
        // SAFETY: The caller supplies writable storage for one u64 each.
        unsafe {
            *numerator = factor.numerator;
            *denominator = factor.denominator;
        }
        unsafe {
            write_i32(
                i32::from(handle.runtime.charge_conjugated()),
                conjugated,
                "charge-conjugation output",
            )
        }
    })
}

/// Loads one JSON kinematic point in public/requested external order.
///
/// # Safety
///
/// `path` must reference a readable NUL-terminated string. Other pointer requirements match
/// [`rusticol_runtime_external_permutation`], with `output` containing f64 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_load_kinematics_json(
    handle: *const RusticolRuntimeHandle,
    path: *const c_char,
    output: *mut c_double,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        let value_count = handle
            .runtime
            .external_count()
            .checked_mul(4)
            .ok_or_else(|| invalid("kinematics value count overflows size_t"))?;
        unsafe { write_size(value_count, required, "kinematics value count")? };
        if output.is_null() {
            if capacity == 0 {
                return Ok(());
            }
            return Err(invalid("kinematics output is null"));
        }
        if capacity < value_count {
            return Err(buffer_too_small(format!(
                "kinematics capacity {capacity} is smaller than {}",
                value_count
            )));
        }
        let path = unsafe { required_c_string(path, "kinematics path") }?;
        let values = handle.runtime.load_kinematics_json(Path::new(path))?;
        if values.len() != value_count {
            return Err(invalid(format!(
                "kinematics JSON contains {} values, expected {value_count}",
                values.len()
            )));
        }
        // SAFETY: Capacity was checked.
        unsafe { ptr::copy_nonoverlapping(values.as_ptr(), output, value_count) };
        Ok(())
    })
}

/// Writes the number of helicity configurations.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `output` must be writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_helicity_count(
    handle: *const RusticolRuntimeHandle,
    output: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        unsafe { write_size(handle.runtime.helicities()?.len(), output, "helicity count") }
    })
}

/// Copies a helicity identifier.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_helicity_id(
    handle: *const RusticolRuntimeHandle,
    index: size_t,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            runtime
                .helicities()?
                .get(index)
                .map(|item| item.id.clone())
                .ok_or_else(|| invalid(format!("helicity index {index} is out of range")))
        })
    }
}

/// Copies one helicity vector.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `output` must be writable for
/// `capacity` `i32` values. A null `output` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_helicity_vector(
    handle: *const RusticolRuntimeHandle,
    index: size_t,
    output: *mut i32,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        let helicities = handle.runtime.helicities()?;
        let item = helicities
            .get(index)
            .ok_or_else(|| invalid(format!("helicity index {index} is out of range")))?;
        unsafe { write_size(item.helicities.len(), required, "helicity vector length")? };
        if output.is_null() {
            if capacity == 0 {
                return Ok(());
            }
            return Err(invalid("helicity vector output is null"));
        }
        if capacity < item.helicities.len() {
            return Err(buffer_too_small(format!(
                "helicity vector capacity {capacity} is smaller than {}",
                item.helicities.len()
            )));
        }
        // SAFETY: Capacity was checked.
        unsafe {
            ptr::copy_nonoverlapping(item.helicities.as_ptr(), output, item.helicities.len())
        };
        Ok(())
    })
}

/// Writes the number of color components.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `output` must be writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_color_count(
    handle: *const RusticolRuntimeHandle,
    output: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        unsafe {
            write_size(
                handle.runtime.color_components()?.len(),
                output,
                "color count",
            )
        }
    })
}

/// Copies a color-component identifier.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_color_id(
    handle: *const RusticolRuntimeHandle,
    index: size_t,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            runtime
                .color_components()?
                .get(index)
                .map(|item| item.id.clone())
                .ok_or_else(|| invalid(format!("color index {index} is out of range")))
        })
    }
}

/// Copies a color-component kind.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_color_kind(
    handle: *const RusticolRuntimeHandle,
    index: size_t,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            runtime
                .color_components()?
                .get(index)
                .map(|item| item.kind.clone())
                .ok_or_else(|| invalid(format!("color index {index} is out of range")))
        })
    }
}

/// Copies one color word.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `output` must be writable for
/// `capacity` `size_t` values. A null `output` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_color_word(
    handle: *const RusticolRuntimeHandle,
    index: size_t,
    output: *mut size_t,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        let colors = handle.runtime.color_components()?;
        let item = colors
            .get(index)
            .ok_or_else(|| invalid(format!("color index {index} is out of range")))?;
        unsafe { write_size(item.word.len(), required, "color word length")? };
        if output.is_null() {
            if capacity == 0 {
                return Ok(());
            }
            return Err(invalid("color word output is null"));
        }
        if capacity < item.word.len() {
            return Err(buffer_too_small(format!(
                "color word capacity {capacity} is smaller than {}",
                item.word.len()
            )));
        }
        // SAFETY: Capacity was checked and usize matches C size_t.
        unsafe { ptr::copy_nonoverlapping(item.word.as_ptr(), output, item.word.len()) };
        Ok(())
    })
}

/// Writes the number of model parameters.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `output` must be writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_model_parameter_count(
    handle: *const RusticolRuntimeHandle,
    output: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle(handle) }?;
        unsafe {
            write_size(
                handle.runtime.model_parameters()?.len(),
                output,
                "model parameter count",
            )
        }
    })
}

/// Copies a model-parameter name.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. If
/// non-null, `required` must be writable for one `size_t`, and `buffer` must be writable for
/// `capacity` bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_model_parameter_name(
    handle: *const RusticolRuntimeHandle,
    index: size_t,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    // SAFETY: The caller upholds this function's pointer contract.
    unsafe {
        runtime_string(handle, buffer, capacity, required, |runtime| {
            runtime
                .model_parameters()?
                .get(index)
                .map(|item| item.name.clone())
                .ok_or_else(|| invalid(format!("model parameter index {index} is out of range")))
        })
    }
}

/// Writes the resolved helicity and color dimensions for a selector set.
///
/// # Safety
///
/// A non-null `handle` must remain live and available for shared access during the call. For each
/// nonzero selector count, the corresponding pointer must reference that many readable string
/// pointers, and every non-null string pointer must reference a readable NUL-terminated string.
/// Non-null output pointers must each be writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_resolved_shape(
    handle: *const RusticolRuntimeHandle,
    helicity_ids: *const *const c_char,
    helicity_count: size_t,
    color_ids: *const *const c_char,
    color_count: size_t,
    output_helicity_count: *mut size_t,
    output_color_count: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers and arrays.
        let handle = unsafe { required_handle(handle) }?;
        let helicities =
            unsafe { read_selector_ids(helicity_ids, helicity_count, "helicity ids") }?;
        let colors = unsafe { read_selector_ids(color_ids, color_count, "color ids") }?;
        let (helicity_count, color_count) = handle
            .runtime
            .resolved_shape(helicities.as_deref(), colors.as_deref())?;
        unsafe {
            write_size(
                helicity_count,
                output_helicity_count,
                "resolved helicity count",
            )?;
            write_size(color_count, output_color_count, "resolved color count")
        }
    })
}

/// Warms one binary64 point using the process-output construction default.
///
/// This preserves the original C ABI and forwards to
/// [`rusticol_runtime_warm_up_f64_with_cores`] with `n_cores=0`.
///
/// # Safety
///
/// The same pointer, storage, and callback requirements as
/// [`rusticol_runtime_warm_up_f64_with_cores`] apply.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_warm_up_f64(
    handle: *mut RusticolRuntimeHandle,
    momenta: *const c_double,
    momentum_count: size_t,
    helicity_ids: *const *const c_char,
    helicity_count: size_t,
    color_ids: *const *const c_char,
    color_count: size_t,
    progress_callback: RusticolWarmUpProgressCallback,
    progress_user_data: *mut c_void,
    output: *mut RusticolWarmUpResult,
) -> c_int {
    // SAFETY: This forwards the caller's storage and callback unchanged.
    unsafe {
        rusticol_runtime_warm_up_f64_with_cores(
            handle,
            momenta,
            momentum_count,
            helicity_ids,
            helicity_count,
            color_ids,
            color_count,
            0,
            progress_callback,
            progress_user_data,
            output,
        )
    }
}

/// Explicitly warms one binary64 point for an on-the-fly runtime selection.
///
/// This constructs and retains the selected OTF query family, performs exactly one evaluation,
/// and returns a fixed-layout summary. Global selector arrays contain physical string IDs; an
/// omitted selector axis is summed over every component retained by the artifact. Progress is
/// reported only from the coordinating caller thread and is throttled by the core runtime.
/// Returning zero from the callback requests cancellation at a cancellable cold-path boundary.
/// The terminal first-evaluation `END` event is post-commit and informational.
/// A zero `n_cores` uses the process-output construction default; a positive value overrides
/// construction workers for this warm-up only, without changing evaluation or cache identity.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call. `momenta`
/// must reference `momentum_count` readable `f64` values containing exactly one point in
/// `[external particle][E, px, py, pz]` order. For each nonzero selector count, the corresponding
/// pointer must reference that many readable NUL-terminated string pointers. `output` must be
/// writable for one [`RusticolWarmUpResult`]. If supplied, `progress_callback` must remain callable
/// for the duration of this function and may use `progress_user_data`; it must not call the same
/// runtime handle recursively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_warm_up_f64_with_cores(
    handle: *mut RusticolRuntimeHandle,
    momenta: *const c_double,
    momentum_count: size_t,
    helicity_ids: *const *const c_char,
    helicity_count: size_t,
    color_ids: *const *const c_char,
    color_count: size_t,
    n_cores: size_t,
    progress_callback: RusticolWarmUpProgressCallback,
    progress_user_data: *mut c_void,
    output: *mut RusticolWarmUpResult,
) -> c_int {
    guard(|| {
        if output.is_null() {
            return Err(invalid("warm-up result output is null"));
        }
        // SAFETY: The caller supplies writable result storage. Initialize it before any
        // fallible operation so failures never leave stale success data behind.
        unsafe { *output = RusticolWarmUpResult::default() };

        // SAFETY: Helpers validate the handle and all input arrays.
        let handle = unsafe { required_handle_mut(handle) }?;
        let momenta = unsafe { read_f64_slice(momenta, momentum_count, "warm-up momenta") }?;
        let helicities =
            unsafe { read_selector_ids(helicity_ids, helicity_count, "helicity ids") }?;
        let colors = unsafe { read_selector_ids(color_ids, color_count, "color ids") }?;
        let n_cores = (n_cores != 0).then_some(n_cores);

        let result = if let Some(callback) = progress_callback {
            let mut observer = |event: &NativeOnTheFlyWarmUpEvent| {
                let event = warm_up_progress_event(event);
                // SAFETY: The caller guarantees that the callback and user-data pointer remain
                // valid for this call. The borrowed event lives through the callback only.
                Ok(unsafe { callback(&event, progress_user_data) } != 0)
            };
            handle.runtime.warm_up_f64(
                momenta,
                helicities.as_deref(),
                colors.as_deref(),
                n_cores,
                Some(&mut observer),
            )
        } else {
            handle.runtime.warm_up_f64(
                momenta,
                helicities.as_deref(),
                colors.as_deref(),
                n_cores,
                None,
            )
        }
        .map_err(AbiError::from)?;

        // SAFETY: The non-null output was validated before invoking the runtime.
        unsafe { *output = warm_up_result(result) };
        Ok(())
    })
}

/// Evaluates total f64 matrix elements for a batch of momentum points.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call. If non-null,
/// `momenta` must reference `momentum_count` readable `f64` values and `output` must reference
/// `output_capacity` writable `f64` values. The input and output regions must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_evaluate_f64(
    handle: *mut RusticolRuntimeHandle,
    momenta: *const c_double,
    momentum_count: size_t,
    point_count: size_t,
    output: *mut c_double,
    output_capacity: size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate all pointers.
        let handle = unsafe { required_handle_mut(handle) }?;
        let momenta = unsafe { read_f64_slice(momenta, momentum_count, "momenta") }?;
        validate_f64_output(output, output_capacity, point_count, "total output")?;
        // SAFETY: The output pointer is non-null and has at least point_count writable values.
        // Restrict the mutable slice to the exact result prefix so excess caller capacity and
        // trailing canaries remain untouched.
        let output = unsafe { slice::from_raw_parts_mut(output, point_count) };
        handle
            .runtime
            .evaluate_f64_into(momenta, point_count, output)
            .map_err(AbiError::from)
    })
}

/// Evaluates total f64 matrix elements with optional global or per-point selectors.
///
/// Global selector arrays contain physical helicity/color string IDs and retain subset/sum
/// semantics. Per-point selector arrays contain zero-based physical-axis indices and must contain
/// exactly one selector for every input point. Global and per-point selectors are mutually
/// exclusive on the same axis. An omitted axis is summed over every component retained by the
/// artifact.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call. `momenta` must
/// reference `momentum_count` readable `f64` values and `output` must reference `output_capacity`
/// writable `f64` values; those regions must not overlap. For each nonzero global selector count,
/// the corresponding pointer must reference that many readable string pointers and each string
/// must be NUL-terminated. For each nonzero per-point selector count, the corresponding pointer
/// must reference exactly `point_count` readable `u32` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_evaluate_selected_f64(
    handle: *mut RusticolRuntimeHandle,
    momenta: *const c_double,
    momentum_count: size_t,
    point_count: size_t,
    helicity_ids: *const *const c_char,
    helicity_count: size_t,
    color_ids: *const *const c_char,
    color_count: size_t,
    helicity_by_point: *const u32,
    helicity_by_point_count: size_t,
    color_flow_by_point: *const u32,
    color_flow_by_point_count: size_t,
    output: *mut c_double,
    output_capacity: size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate all pointers and arrays.
        let handle = unsafe { required_handle_mut(handle) }?;
        if point_count == 0 {
            return Err(invalid("point_count must be positive"));
        }
        let momenta = unsafe { read_f64_slice(momenta, momentum_count, "momenta") }?;
        let helicities =
            unsafe { read_selector_ids(helicity_ids, helicity_count, "helicity ids") }?;
        let colors = unsafe { read_selector_ids(color_ids, color_count, "color ids") }?;
        let helicity_by_point = unsafe {
            read_optional_u32_slice(
                helicity_by_point,
                helicity_by_point_count,
                point_count,
                "helicity_by_point",
            )
        }?;
        let color_flow_by_point = unsafe {
            read_optional_u32_slice(
                color_flow_by_point,
                color_flow_by_point_count,
                point_count,
                "color_flow_by_point",
            )
        }?;
        if helicities.is_some() && helicity_by_point.is_some() {
            return Err(invalid(
                "helicity ids and helicity_by_point are mutually exclusive",
            ));
        }
        if colors.is_some() && color_flow_by_point.is_some() {
            return Err(invalid(
                "color ids and color_flow_by_point are mutually exclusive",
            ));
        }
        validate_f64_output(output, output_capacity, point_count, "total output")?;
        // SAFETY: The output pointer is non-null and has at least point_count writable values.
        // Restrict the mutable slice to the exact result prefix so excess caller capacity and
        // trailing canaries remain untouched.
        let output = unsafe { slice::from_raw_parts_mut(output, point_count) };
        handle
            .runtime
            .evaluate_f64_into_with_selectors(
                momenta,
                point_count,
                helicities.as_deref(),
                colors.as_deref(),
                helicity_by_point,
                color_flow_by_point,
                output,
            )
            .map_err(AbiError::from)
    })
}

/// Evaluates resolved f64 matrix elements for a batch of momentum points.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call. If non-null,
/// `momenta` must reference `momentum_count` readable `f64` values and `output` must reference
/// `output_capacity` writable `f64` values; those regions must not overlap. For each nonzero
/// selector count, the corresponding pointer must reference that many readable string pointers,
/// and every non-null string pointer must reference a readable NUL-terminated string. Non-null
/// shape outputs must each be writable for one `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_evaluate_resolved_f64(
    handle: *mut RusticolRuntimeHandle,
    momenta: *const c_double,
    momentum_count: size_t,
    point_count: size_t,
    helicity_ids: *const *const c_char,
    helicity_count: size_t,
    color_ids: *const *const c_char,
    color_count: size_t,
    output: *mut c_double,
    output_capacity: size_t,
    output_helicity_count: *mut size_t,
    output_color_count: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate all pointers and arrays.
        let handle = unsafe { required_handle_mut(handle) }?;
        let momenta = unsafe { read_f64_slice(momenta, momentum_count, "momenta") }?;
        let helicities =
            unsafe { read_selector_ids(helicity_ids, helicity_count, "helicity ids") }?;
        let colors = unsafe { read_selector_ids(color_ids, color_count, "color ids") }?;
        if output_helicity_count.is_null() {
            return Err(invalid("resolved helicity count output is null"));
        }
        if output_color_count.is_null() {
            return Err(invalid("resolved color count output is null"));
        }
        if point_count == 0 {
            return Err(invalid("point_count must be positive"));
        }
        let (expected_helicity_count, expected_color_count) = handle
            .runtime
            .resolved_shape(helicities.as_deref(), colors.as_deref())?;
        let required = point_count
            .checked_mul(expected_helicity_count)
            .and_then(|value| value.checked_mul(expected_color_count))
            .ok_or_else(|| invalid("resolved output shape overflow"))?;
        validate_f64_output(output, output_capacity, required, "resolved output")?;
        let resolved = handle.runtime.evaluate_resolved_f64(
            momenta,
            point_count,
            helicities.as_deref(),
            colors.as_deref(),
        )?;
        let (_, helicity_count, color_count) = resolved.shape();
        if helicity_count != expected_helicity_count || color_count != expected_color_count {
            return Err(abi_error(
                RUSTICOL_STATUS_RUNTIME_ERROR,
                "resolved evaluation shape changed after buffer validation",
            ));
        }
        unsafe {
            write_size(
                helicity_count,
                output_helicity_count,
                "resolved helicity count",
            )?;
            write_size(color_count, output_color_count, "resolved color count")?;
            write_f64_slice(&resolved.values, output, output_capacity, "resolved output")
        }
    })
}

/// Updates multiple runtime model parameters atomically.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call. For nonzero
/// `count`, `names` must reference `count` readable string pointers, every non-null string pointer
/// must reference a readable NUL-terminated string, and `real` and `imaginary` must each reference
/// `count` readable `f64` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_set_model_parameters(
    handle: *mut RusticolRuntimeHandle,
    names: *const *const c_char,
    real: *const c_double,
    imaginary: *const c_double,
    count: size_t,
) -> c_int {
    guard(|| {
        if count == 0 {
            return Err(invalid("model parameter update must not be empty"));
        }
        // SAFETY: Helpers and explicit checks validate all pointers.
        let handle = unsafe { required_handle_mut(handle) }?;
        let names = unsafe { read_selector_ids(names, count, "model parameter names") }?
            .expect("positive count returns names");
        let real = unsafe { read_f64_slice(real, count, "model parameter real values") }?;
        let imaginary =
            unsafe { read_f64_slice(imaginary, count, "model parameter imaginary values") }?;
        let mut values = BTreeMap::new();
        for index in 0..count {
            if values
                .insert(names[index].clone(), (real[index], imaginary[index]))
                .is_some()
            {
                return Err(invalid(format!(
                    "duplicate model parameter update {:?}",
                    names[index]
                )));
            }
        }
        Ok(handle.runtime.set_model_parameters(&values)?)
    })
}

/// Updates one runtime model parameter.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call. A non-null
/// `name` must reference a readable NUL-terminated string for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_set_model_parameter(
    handle: *mut RusticolRuntimeHandle,
    name: *const c_char,
    real: c_double,
    imaginary: c_double,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle_mut(handle) }?;
        let name = unsafe { required_c_string(name, "model parameter name") }?;
        Ok(handle.runtime.set_model_parameter(name, real, imaginary)?)
    })
}

/// Reads a current runtime model parameter, including refreshed derived values.
///
/// # Safety
///
/// The handle must remain live and shared-accessible; name must be a readable
/// NUL-terminated string. Both outputs must be writable for one double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_get_model_parameter(
    handle: *const RusticolRuntimeHandle,
    name: *const c_char,
    real: *mut c_double,
    imaginary: *mut c_double,
) -> c_int {
    guard(|| {
        if real.is_null() || imaginary.is_null() {
            return Err(invalid("model parameter outputs must be non-null"));
        }
        // SAFETY: The helpers validate the pointer arguments.
        let handle = unsafe { required_handle(handle) }?;
        let name = unsafe { required_c_string(name, "model parameter name") }?;
        let value = handle.runtime.model_parameter_value(name)?;
        // SAFETY: Non-null output storage is guaranteed by the caller contract.
        unsafe {
            *real = value.0;
            *imaginary = value.1;
        }
        Ok(())
    })
}

/// Updates runtime model parameters from a JSON file.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call. A non-null
/// `path` must reference a readable NUL-terminated string for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_set_model_parameters_json(
    handle: *mut RusticolRuntimeHandle,
    path: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle_mut(handle) }?;
        let path = unsafe { required_c_string(path, "model parameter JSON path") }?;
        Ok(handle.runtime.set_model_parameters_json(Path::new(path))?)
    })
}

/// Mutes or unmutes runtime warnings.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_mute_warnings(
    handle: *mut RusticolRuntimeHandle,
    muted: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: Helper validates the handle.
        let handle = unsafe { required_handle_mut(handle) }?;
        if muted == 0 {
            handle.runtime.unmute_warnings();
        } else {
            handle.runtime.mute_warnings();
        }
        Ok(())
    })
}

/// Copies and consumes the runtime's pending warnings as JSON.
///
/// # Safety
///
/// A non-null `handle` must remain live and exclusively accessible during the call. If non-null,
/// `required` must be writable for one `size_t`, and `buffer` must be writable for `capacity`
/// bytes. A null `buffer` is valid only for a zero-capacity query.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_take_warnings_json(
    handle: *mut RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        // SAFETY: Helpers validate pointers.
        let handle = unsafe { required_handle_mut(handle) }?;
        let json = handle.runtime.pending_warnings_json()?;
        unsafe { write_string(&json, buffer, capacity, required)? };
        if !buffer.is_null() {
            handle.runtime.clear_pending_warnings();
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_abi_error_is_thread_local_and_nul_terminated() {
        set_last_error("one\0two");
        LAST_ERROR.with(|slot| {
            assert_eq!(slot.borrow().to_str().unwrap(), "one\\0two");
        });
    }

    #[test]
    fn null_handle_is_reported_without_panicking() {
        let mut count = 0;
        // SAFETY: A null handle is explicitly accepted and reported as an ABI error.
        let status = unsafe { rusticol_runtime_external_count(ptr::null(), &mut count) };
        assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn current_parameter_getter_rejects_null_outputs_without_writing() {
        let name = CString::new("aS").unwrap();
        let mut value = 42.0;
        // SAFETY: Null arguments are explicitly validated, the other output is live.
        let status = unsafe {
            rusticol_runtime_get_model_parameter(
                ptr::null(),
                name.as_ptr(),
                &mut value,
                ptr::null_mut(),
            )
        };
        assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
        assert_eq!(value, 42.0);
    }

    #[test]
    fn short_string_buffer_has_a_distinct_status() {
        set_last_error("a message larger than one byte");
        let mut byte = 0_i8;
        let mut required = 0;
        // SAFETY: Both outputs point to writable storage of the declared capacity.
        let status = unsafe { rusticol_last_error_message(&mut byte, 1, &mut required) };
        assert_eq!(status, RUSTICOL_STATUS_BUFFER_TOO_SMALL);
        assert!(required > 1);
    }

    #[test]
    fn invalid_last_error_output_reports_invalid_argument() {
        // SAFETY: Null outputs are explicitly accepted and reported as an ABI error.
        let status = unsafe { rusticol_last_error_message(ptr::null_mut(), 0, ptr::null_mut()) };
        assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn null_string_buffer_is_only_valid_for_a_zero_capacity_query() {
        let mut required = 0;
        let query = unsafe { write_string("value", ptr::null_mut(), 0, &mut required) };
        assert!(query.is_ok());
        assert_eq!(required, 6);

        let error = unsafe { write_string("value", ptr::null_mut(), 1, &mut required) }
            .expect_err("nonzero null buffer must fail");
        assert_eq!(error.status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn f64_output_capacity_is_validated_before_copying() {
        let values = [1.0, 2.0];
        let mut output = [99.0];

        let error = unsafe { write_f64_slice(&values, output.as_mut_ptr(), 1, "test output") }
            .expect_err("short output must fail");

        assert_eq!(error.status, RUSTICOL_STATUS_BUFFER_TOO_SMALL);
        assert_eq!(output, [99.0]);
    }

    #[test]
    fn failed_load_clears_the_output_handle() {
        let path = CString::new("/path/that/does/not/exist").unwrap();
        let mut output = usize::MAX as *mut RusticolRuntimeHandle;

        // SAFETY: The path is NUL-terminated and output points to writable handle storage.
        let status =
            unsafe { rusticol_runtime_load(path.as_ptr(), ptr::null(), ptr::null(), &mut output) };

        assert_eq!(status, RUSTICOL_STATUS_RUNTIME_ERROR);
        assert!(output.is_null());
    }

    #[test]
    fn cache_calls_reject_null_handles() {
        let path = CString::new("cache.bin").unwrap();
        // SAFETY: A null handle is explicitly validated by both entry points.
        assert_eq!(
            unsafe { rusticol_runtime_save(ptr::null(), path.as_ptr()) },
            RUSTICOL_STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { rusticol_runtime_load_cache(ptr::null_mut(), path.as_ptr()) },
            RUSTICOL_STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn static_c_api_advertises_all_symbolica_free_f64_capabilities() {
        let mut required = 0;
        // SAFETY: This is a zero-capacity query with writable required-size storage.
        let status = unsafe {
            rusticol_supported_runtime_capabilities_json(ptr::null_mut(), 0, &mut required)
        };
        assert_eq!(status, RUSTICOL_STATUS_OK);
        assert!(required > 1);

        let mut buffer = vec![0_i8; required];
        // SAFETY: The output buffer and required-size storage are valid for the declared sizes.
        let status = unsafe {
            rusticol_supported_runtime_capabilities_json(
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut required,
            )
        };
        assert_eq!(status, RUSTICOL_STATUS_OK);
        let json = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_str().unwrap();
        assert_eq!(
            serde_json::from_str::<Vec<String>>(json).unwrap(),
            vec![
                "compiled-plane-arena-v1".to_string(),
                "eager-direct-arena-v1".to_string(),
                "rusticol.color-contraction.symmetric-group-fft.v1".to_string(),
                "rusticol.compiled.color-contraction-walsh-c2k.v1".to_string(),
                "rusticol.compiled.color-contraction-walsh.v1".to_string(),
                "rusticol.compiled.color-topology-lanes.v1".to_string(),
                "rusticol.compiled.helicity-dual-lane.v1".to_string(),
                "rusticol.compiled.helicity-primary-recurrence.v1".to_string(),
                "rusticol.compiled.helicity-selector-union.v1".to_string(),
                "rusticol.compiled.runtime-selectors.v1".to_string(),
                "rusticol.eager-runtime-layout.complex-f64.v1".to_string(),
                "rusticol.on-the-fly.complex-f64.v1".to_string(),
                "rusticol.on-the-fly.contracted-color.v1".to_string(),
                "rusticol.on-the-fly.lc-color.v1".to_string(),
                "rusticol.recurrence-color.contracted.v1".to_string(),
                "rusticol.recurrence-color.lc.v1".to_string(),
                "rusticol.recurrence-direct-arena.complex-f64.v1".to_string(),
                "rusticol.recurrence-helicity-selector-companion.v2".to_string(),
                "symbolica.compiled-asm.complex-f64.v1".to_string(),
                "symbolica.compiled-cpp.complex-f64.v1".to_string(),
                "symjit.application.complex-f64.v1".to_string(),
            ]
        );
    }

    #[test]
    fn native_argument_errors_map_to_the_c_argument_status() {
        let error = AbiError::from(RusticolError::selector("bad selector"));
        assert_eq!(error.status, RUSTICOL_STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn warm_up_pods_preserve_schema_stage_memory_and_completion_state() {
        assert_eq!(std::mem::size_of::<RusticolWarmUpProgressEvent>(), 72);
        assert_eq!(std::mem::size_of::<RusticolWarmUpResult>(), 64);
        let event = NativeOnTheFlyWarmUpEvent {
            schema_version: 1,
            kind: NativeOnTheFlyWarmUpEventKind::Update,
            stage: NativeOnTheFlyWarmUpStage::QueryFamily,
            completed: 7,
            total: 11,
            elapsed_seconds: 0.25,
            current_rss_bytes: Some(1024),
            peak_rss_bytes: None,
            workers: 3,
            message: Some("not copied across the fixed C ABI".to_string()),
        };
        let pod = warm_up_progress_event(&event);
        assert_eq!(pod.schema_version, 1);
        assert_eq!(pod.kind, RUSTICOL_WARM_UP_EVENT_UPDATE);
        assert_eq!(pod.stage, RUSTICOL_WARM_UP_STAGE_QUERY_FAMILY);
        assert_eq!((pod.completed, pod.total), (7, 11));
        assert_eq!(pod.current_rss_bytes, 1024);
        assert_eq!(pod.current_rss_available, 1);
        assert_eq!(pod.peak_rss_bytes, 0);
        assert_eq!(pod.peak_rss_available, 0);
        assert_eq!(pod.workers, 3);

        let result = warm_up_result(NativeOnTheFlyWarmUpResult {
            schema_version: 1,
            elapsed_seconds: 0.5,
            query_count: 19,
            warmed_query_count: 17,
            current_rss_bytes: None,
            peak_rss_bytes: Some(4096),
            already_warm: false,
            first_evaluation_completed: true,
        });
        assert_eq!((result.query_count, result.warmed_query_count), (19, 17));
        assert_eq!(result.current_rss_available, 0);
        assert_eq!(result.peak_rss_available, 1);
        assert_eq!(result.peak_rss_bytes, 4096);
        assert_eq!(result.already_warm, 0);
        assert_eq!(result.first_evaluation_completed, 1);
    }

    #[test]
    fn failed_warm_up_clears_result_before_validating_the_handle() {
        let point = [0.0; 8];
        let mut result = RusticolWarmUpResult {
            schema_version: u32::MAX,
            query_count: u64::MAX,
            first_evaluation_completed: 1,
            ..RusticolWarmUpResult::default()
        };
        // SAFETY: The point and result are valid; a null handle is deliberately rejected.
        let status = unsafe {
            rusticol_runtime_warm_up_f64(
                ptr::null_mut(),
                point.as_ptr(),
                point.len(),
                ptr::null(),
                0,
                ptr::null(),
                0,
                None,
                ptr::null_mut(),
                &mut result,
            )
        };
        assert_eq!(status, RUSTICOL_STATUS_INVALID_ARGUMENT);
        assert_eq!(result, RusticolWarmUpResult::default());
    }

    #[test]
    fn panic_payloads_map_to_the_panic_status() {
        let payload: Box<dyn Any + Send> = Box::new(String::from("contained test panic"));
        let status = finish_guard(Err(payload));
        assert_eq!(status, RUSTICOL_STATUS_PANIC);
        LAST_ERROR.with(|slot| {
            assert!(
                slot.borrow()
                    .to_str()
                    .unwrap()
                    .contains("contained test panic")
            );
        });
    }

    #[test]
    fn guard_contains_panics_before_they_cross_the_c_boundary() {
        let status = guard(|| -> AbiResult<()> { panic!("contained ABI panic") });

        assert_eq!(status, RUSTICOL_STATUS_PANIC);
        LAST_ERROR.with(|slot| {
            assert!(
                slot.borrow()
                    .to_str()
                    .unwrap()
                    .contains("contained ABI panic")
            );
        });
    }
}
