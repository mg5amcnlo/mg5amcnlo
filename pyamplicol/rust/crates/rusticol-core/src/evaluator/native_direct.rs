// SPDX-License-Identifier: 0BSD

//! Native dynamic-library adapter for Direct-Arena recurrence executors.
//!
//! This module only owns and authenticates native prepared callables. Source
//! filling and intrinsic closure reduction remain Rusticol-owned. The returned
//! handle calls directly into the loaded library and therefore performs no
//! packing, scattering, or allocation. Callers must retain this owner for as
//! long as a copied [`DirectExecutorHandle`] can be invoked.

use crate::artifact::PinnedNativeLibrary;
use crate::recurrence::direct_backend::{
    DirectClosureExecutor, DirectContributionExecutor, DirectExecutorHandle,
    DirectFinalizationExecutor,
};
use crate::recurrence::{DIRECT_NONE_U32, DirectExecutorRole};
use crate::{RusticolError, RusticolResult};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

const NATIVE_DIRECT_SYMBOL_PREFIX: &str = "pyamplicol_recurrence_direct";

#[repr(C)]
struct NativeDirectBindingContextV1 {
    coupling_re: f64,
    coupling_im: f64,
}

/// Owns one loaded native Direct-Arena executor and its dynamic library.
///
/// Moving this value is safe because the copied function pointer addresses the
/// loaded library rather than this Rust object. Dropping it invalidates any
/// copied handle, so runtime integration must retain the owner beside the
/// `DirectExecutorCatalog`.
pub(crate) struct LoadedNativeDirectExecutor {
    _library: Arc<PinnedNativeLibrary>,
    _binding_context: Option<Box<NativeDirectBindingContextV1>>,
    handle: DirectExecutorHandle,
    #[cfg(test)]
    role: DirectExecutorRole,
    #[cfg(test)]
    prepared_kernel_id: u32,
}

impl LoadedNativeDirectExecutor {
    /// Load one authenticated role-specific native Direct-Arena export.
    ///
    /// `exported_symbol` is carried by the prepared-kernel producer contract.
    /// It must exactly equal [`native_direct_symbol_name`] for the requested
    /// role and prepared-kernel ID. This makes role confusion fail before the
    /// dynamic library is opened.
    pub(crate) fn load(
        library: Arc<PinnedNativeLibrary>,
        role: DirectExecutorRole,
        prepared_kernel_id: u32,
        exported_symbol: &str,
        binding_coupling: Option<(f64, f64)>,
    ) -> RusticolResult<Self> {
        validate_c_symbol(exported_symbol)?;
        let expected_symbol = native_direct_symbol_name(role, prepared_kernel_id)?;
        if exported_symbol != expected_symbol {
            return Err(RusticolError::integrity(format!(
                "native Direct-Arena symbol {exported_symbol:?} does not authenticate role \
                 {role:?} and prepared kernel {prepared_kernel_id}; expected \
                 {expected_symbol:?}"
            )));
        }
        if binding_coupling.is_some_and(|(re, im)| !re.is_finite() || !im.is_finite()) {
            return Err(RusticolError::integrity(
                "native Direct-Arena binding coupling must be finite",
            ));
        }

        let path = library.display_path();
        let binding_context = binding_coupling.map(|(coupling_re, coupling_im)| {
            Box::new(NativeDirectBindingContextV1 {
                coupling_re,
                coupling_im,
            })
        });
        let context = binding_context
            .as_deref()
            .map_or(ptr::null(), |value| std::ptr::from_ref(value).cast());
        let handle = match role {
            DirectExecutorRole::Contribution => {
                let call = unsafe {
                    load_export::<DirectContributionExecutor>(&library, path, exported_symbol)?
                };
                DirectExecutorHandle::Contribution { call, context }
            }
            DirectExecutorRole::Finalization => {
                let call = unsafe {
                    load_export::<DirectFinalizationExecutor>(&library, path, exported_symbol)?
                };
                DirectExecutorHandle::Finalization { call, context }
            }
            DirectExecutorRole::Closure => {
                let call = unsafe {
                    load_export::<DirectClosureExecutor>(&library, path, exported_symbol)?
                };
                DirectExecutorHandle::Closure { call, context }
            }
            DirectExecutorRole::Source => {
                return Err(RusticolError::invalid_argument(
                    "native Direct-Arena libraries cannot provide source executors; source filling is Rusticol-owned",
                ));
            }
        };

        Ok(Self {
            _library: library,
            _binding_context: binding_context,
            handle,
            #[cfg(test)]
            role,
            #[cfg(test)]
            prepared_kernel_id,
        })
    }

    /// Return the direct callable. The owner must outlive every invocation.
    pub(crate) const fn handle(&self) -> DirectExecutorHandle {
        self.handle
    }

    #[cfg(test)]
    pub(crate) const fn role(&self) -> DirectExecutorRole {
        self.role
    }

    #[cfg(test)]
    pub(crate) const fn prepared_kernel_id(&self) -> u32 {
        self.prepared_kernel_id
    }
}

/// Deterministic C export for one prepared Direct-Arena kernel.
///
/// The convention is
/// `pyamplicol_recurrence_direct_<role>_k<8-lowercase-hex-id>_v1`.
pub(crate) fn native_direct_symbol_name(
    role: DirectExecutorRole,
    prepared_kernel_id: u32,
) -> RusticolResult<String> {
    if prepared_kernel_id == DIRECT_NONE_U32 {
        return Err(RusticolError::invalid_argument(
            "native Direct-Arena prepared-kernel ID must not be the missing-ID sentinel",
        ));
    }
    let role_name = match role {
        DirectExecutorRole::Contribution => "contribution",
        DirectExecutorRole::Finalization => "finalization",
        DirectExecutorRole::Closure => "closure",
        DirectExecutorRole::Source => {
            return Err(RusticolError::invalid_argument(
                "native Direct-Arena symbol names are not defined for Rusticol-owned source executors",
            ));
        }
    };
    Ok(format!(
        "{NATIVE_DIRECT_SYMBOL_PREFIX}_{role_name}_k{prepared_kernel_id:08x}_v1"
    ))
}

fn validate_c_symbol(symbol: &str) -> RusticolResult<()> {
    let mut bytes = symbol.bytes();
    let Some(first) = bytes.next() else {
        return Err(RusticolError::invalid_argument(
            "native Direct-Arena exported symbol must not be empty",
        ));
    };
    if !(first == b'_' || first.is_ascii_alphabetic())
        || bytes.any(|byte| !(byte == b'_' || byte.is_ascii_alphanumeric()))
    {
        return Err(RusticolError::invalid_argument(format!(
            "native Direct-Arena exported symbol {symbol:?} is not a portable C identifier"
        )));
    }
    Ok(())
}

unsafe fn load_export<T: Copy>(
    library: &libloading::Library,
    path: &Path,
    exported_symbol: &str,
) -> RusticolResult<T> {
    // SAFETY: The producer contract assigns this authenticated role-specific
    // symbol exactly the `T` function signature selected by the caller.
    unsafe {
        library
            .get::<T>(exported_symbol.as_bytes())
            .map(|symbol| *symbol)
            .map_err(|error| {
                RusticolError::evaluation(format!(
                    "could not load native Direct-Arena symbol {exported_symbol:?} from {}: {error}",
                    path.display()
                ))
            })
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
pub(crate) mod tests {
    use super::*;
    use crate::RusticolErrorKind;
    use crate::artifact::EvaluatorPayloadStore;
    use crate::recurrence::direct_backend::{
        DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
    };
    #[cfg(not(feature = "f64-symjit"))]
    use std::alloc::{GlobalAlloc, Layout, System};
    #[cfg(not(feature = "f64-symjit"))]
    use std::cell::Cell;
    use std::fs;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(not(feature = "f64-symjit"))]
    thread_local! {
        static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
        static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
    }

    #[cfg(not(feature = "f64-symjit"))]
    struct CountingAllocator;

    #[cfg(not(feature = "f64-symjit"))]
    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

    #[cfg(not(feature = "f64-symjit"))]
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count_allocation(layout.size());
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count_allocation(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            count_allocation(new_size);
            unsafe { System.realloc(pointer, layout, new_size) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }
    }

    #[cfg(not(feature = "f64-symjit"))]
    fn count_allocation(bytes: usize) {
        let tracking = TRACK_ALLOCATIONS.try_with(Cell::get).unwrap_or(false);
        if tracking {
            let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
            let _ = ALLOCATED_BYTES.try_with(|total| total.set(total.get().saturating_add(bytes)));
        }
    }

    #[cfg(not(feature = "f64-symjit"))]
    pub(crate) fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
        ALLOCATION_COUNT.with(|count| count.set(0));
        ALLOCATED_BYTES.with(|total| total.set(0));
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
        let result = operation();
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        let count = ALLOCATION_COUNT.with(Cell::get);
        let bytes = ALLOCATED_BYTES.with(Cell::get);
        (result, count, bytes)
    }

    const CONTRIBUTION_ID: u32 = 7;
    const FINALIZATION_ID: u32 = 11;
    const CLOSURE_ID: u32 = 13;

    fn expect_load_error(result: RusticolResult<LoadedNativeDirectExecutor>) -> RusticolError {
        match result {
            Ok(_) => panic!("native Direct-Arena executor unexpectedly loaded"),
            Err(error) => error,
        }
    }

    fn native_fixture() -> (std::path::PathBuf, std::path::PathBuf) {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "rusticol-native-direct-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        let source = directory.join("fixture.c");
        let library = directory.join(if cfg!(target_os = "macos") {
            "libnative_direct_fixture.dylib"
        } else {
            "libnative_direct_fixture.so"
        });
        let contribution_symbol =
            native_direct_symbol_name(DirectExecutorRole::Contribution, CONTRIBUTION_ID).unwrap();
        let finalization_symbol =
            native_direct_symbol_name(DirectExecutorRole::Finalization, FINALIZATION_ID).unwrap();
        let closure_symbol =
            native_direct_symbol_name(DirectExecutorRole::Closure, CLOSURE_ID).unwrap();
        let fixture = r#"#include <stdint.h>

typedef struct {
    double *current_re;
    double *current_im;
    uint64_t current_scalar_len;
    double *amplitude_re;
    double *amplitude_im;
    uint64_t amplitude_scalar_len;
    uint32_t point_stride;
} DirectArenaView;

typedef struct {
    const double *values;
    uint64_t scalar_len;
    uint32_t form_count;
    uint16_t lorentz_component_count;
    uint32_t point_stride;
} DirectMomentumView;

typedef struct {
    const double *values_re;
    const double *values_im;
    uint32_t value_count;
} DirectParameterView;

typedef DirectParameterView DirectFactorView;

typedef struct {
    double coupling_re;
    double coupling_im;
} DirectNativeBindingContextV1;

#define DIRECT_ARGS \
    const void *context, \
    DirectArenaView arena, \
    DirectMomentumView momenta, \
    DirectParameterView parameters, \
    DirectFactorView factors, \
    const void *rows, \
    uint32_t row_count, \
    uint32_t point_count

int __CONTRIBUTION_SYMBOL__(DIRECT_ARGS) {
    (void)arena; (void)momenta; (void)parameters; (void)factors; (void)rows;
    int binding_offset = 0;
    if (context != 0) {
        const DirectNativeBindingContextV1 *binding =
            (const DirectNativeBindingContextV1 *)context;
        binding_offset =
            (int)(100.0 * binding->coupling_re + 10.0 * binding->coupling_im);
    }
    return 1000 + binding_offset + (int)row_count + (int)point_count;
}

int __FINALIZATION_SYMBOL__(DIRECT_ARGS) {
    (void)context; (void)arena; (void)momenta; (void)parameters; (void)factors; (void)rows;
    return 2000 + (int)row_count + (int)point_count;
}

int __CLOSURE_SYMBOL__(DIRECT_ARGS) {
    (void)context; (void)arena; (void)momenta; (void)parameters; (void)factors; (void)rows;
    return 3000 + (int)row_count + (int)point_count;
}
"#
        .replace("__CONTRIBUTION_SYMBOL__", &contribution_symbol)
        .replace("__FINALIZATION_SYMBOL__", &finalization_symbol)
        .replace("__CLOSURE_SYMBOL__", &closure_symbol);
        fs::write(&source, fixture).unwrap();
        let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let mut command = Command::new(compiler);
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
            "could not compile native Direct-Arena fixture: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        (directory, library)
    }

    fn pinned(path: &Path) -> Arc<PinnedNativeLibrary> {
        PinnedNativeLibrary::from_test_path(path).unwrap()
    }

    fn empty_views() -> (
        DirectArenaView,
        DirectMomentumView,
        DirectParameterView,
        DirectFactorView,
    ) {
        (
            DirectArenaView {
                current_re: ptr::null_mut(),
                current_im: ptr::null_mut(),
                current_scalar_len: 0,
                amplitude_re: ptr::null_mut(),
                amplitude_im: ptr::null_mut(),
                amplitude_scalar_len: 0,
                point_stride: 0,
            },
            DirectMomentumView {
                values: ptr::null(),
                scalar_len: 0,
                form_count: 0,
                lorentz_component_count: 0,
                point_stride: 0,
            },
            DirectParameterView {
                values_re: ptr::null(),
                values_im: ptr::null(),
                value_count: 0,
            },
            DirectFactorView {
                values_re: ptr::null(),
                values_im: ptr::null(),
                value_count: 0,
            },
        )
    }

    #[test]
    fn deterministic_symbols_encode_role_and_prepared_kernel_id() {
        assert_eq!(
            native_direct_symbol_name(DirectExecutorRole::Contribution, CONTRIBUTION_ID,).unwrap(),
            "pyamplicol_recurrence_direct_contribution_k00000007_v1"
        );
        assert_eq!(
            native_direct_symbol_name(DirectExecutorRole::Finalization, FINALIZATION_ID,).unwrap(),
            "pyamplicol_recurrence_direct_finalization_k0000000b_v1"
        );
        assert_eq!(
            native_direct_symbol_name(DirectExecutorRole::Closure, CLOSURE_ID).unwrap(),
            "pyamplicol_recurrence_direct_closure_k0000000d_v1"
        );
        assert_eq!(
            native_direct_symbol_name(DirectExecutorRole::Source, 0)
                .unwrap_err()
                .kind(),
            RusticolErrorKind::InvalidArgument
        );
        assert_eq!(
            native_direct_symbol_name(DirectExecutorRole::Contribution, DIRECT_NONE_U32,)
                .unwrap_err()
                .kind(),
            RusticolErrorKind::InvalidArgument
        );
    }

    #[test]
    fn symbol_authentication_rejects_empty_invalid_and_wrong_role_names() {
        let (directory, library_path) = native_fixture();
        let library = pinned(&library_path);
        for symbol in ["", "not-a-c-symbol", "contains\0nul"] {
            let error = expect_load_error(LoadedNativeDirectExecutor::load(
                Arc::clone(&library),
                DirectExecutorRole::Contribution,
                CONTRIBUTION_ID,
                symbol,
                None,
            ));
            assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
        }

        let contribution_symbol =
            native_direct_symbol_name(DirectExecutorRole::Contribution, CONTRIBUTION_ID).unwrap();
        let error = expect_load_error(LoadedNativeDirectExecutor::load(
            Arc::clone(&library),
            DirectExecutorRole::Closure,
            CONTRIBUTION_ID,
            &contribution_symbol,
            None,
        ));
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);
        assert!(error.message().contains("expected"));

        let error = expect_load_error(LoadedNativeDirectExecutor::load(
            Arc::clone(&library),
            DirectExecutorRole::Contribution,
            CONTRIBUTION_ID,
            &contribution_symbol,
            Some((f64::NAN, 0.0)),
        ));
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);

        let error = expect_load_error(LoadedNativeDirectExecutor::load(
            library,
            DirectExecutorRole::Source,
            0,
            "pyamplicol_recurrence_direct_source_k00000000_v1",
            None,
        ));
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn native_role_exports_load_and_remain_callable_while_owners_move() {
        let (directory, library_path) = native_fixture();
        let store = EvaluatorPayloadStore::directory(&directory);
        let relative_library = library_path.file_name().unwrap().to_str().unwrap();
        let library = store.load_native_library(relative_library).unwrap();
        let cached_library = store.load_native_library(relative_library).unwrap();
        assert!(Arc::ptr_eq(&library, &cached_library));
        let specifications = [
            (DirectExecutorRole::Contribution, CONTRIBUTION_ID, 1008),
            (DirectExecutorRole::Finalization, FINALIZATION_ID, 2008),
            (DirectExecutorRole::Closure, CLOSURE_ID, 3008),
        ];
        let mut owners = Vec::new();
        let mut handles = Vec::new();
        for (role, prepared_kernel_id, _) in specifications {
            let symbol = native_direct_symbol_name(role, prepared_kernel_id).unwrap();
            let owner = LoadedNativeDirectExecutor::load(
                Arc::clone(&library),
                role,
                prepared_kernel_id,
                &symbol,
                None,
            )
            .unwrap();
            assert_eq!(owner.role(), role);
            assert_eq!(owner.prepared_kernel_id(), prepared_kernel_id);
            handles.push(owner.handle());
            owners.push(owner);
        }

        fs::write(&library_path, b"replaced-after-authenticated-load").unwrap();

        let (arena, momenta, parameters, factors) = empty_views();
        for ((role, _, expected), handle) in specifications.into_iter().zip(handles.iter().copied())
        {
            let status = unsafe {
                match (role, handle) {
                    (
                        DirectExecutorRole::Contribution,
                        DirectExecutorHandle::Contribution { call, context },
                    ) => call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        ptr::null(),
                        3,
                        5,
                    ),
                    (
                        DirectExecutorRole::Finalization,
                        DirectExecutorHandle::Finalization { call, context },
                    ) => call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        ptr::null(),
                        3,
                        5,
                    ),
                    (
                        DirectExecutorRole::Closure,
                        DirectExecutorHandle::Closure { call, context },
                    ) => call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        ptr::null(),
                        3,
                        5,
                    ),
                    _ => panic!("native Direct-Arena handle has the wrong role"),
                }
            };
            assert_eq!(status, expected);
        }

        let contribution_symbol =
            native_direct_symbol_name(DirectExecutorRole::Contribution, CONTRIBUTION_ID).unwrap();
        let first_context = LoadedNativeDirectExecutor::load(
            Arc::clone(&library),
            DirectExecutorRole::Contribution,
            CONTRIBUTION_ID,
            &contribution_symbol,
            Some((2.0, 3.0)),
        )
        .unwrap();
        let first_handle = first_context.handle();
        let second_context = LoadedNativeDirectExecutor::load(
            Arc::clone(&library),
            DirectExecutorRole::Contribution,
            CONTRIBUTION_ID,
            &contribution_symbol,
            Some((-1.0, 4.0)),
        )
        .unwrap();
        let second_handle = second_context.handle();
        owners.push(first_context);
        owners.push(second_context);
        for (handle, expected) in [(first_handle, 1_238), (second_handle, 948)] {
            let DirectExecutorHandle::Contribution { call, context } = handle else {
                panic!("context fixture must load a contribution");
            };
            let status = unsafe {
                call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    ptr::null(),
                    3,
                    5,
                )
            };
            assert_eq!(status, expected);
        }

        #[cfg(not(feature = "f64-symjit"))]
        {
            let DirectExecutorHandle::Contribution { call, context } = handles[0] else {
                panic!("first native Direct-Arena handle must be a contribution");
            };
            let (status_sum, allocation_count, allocated_bytes) = count_allocations(|| {
                let mut status_sum = 0;
                for _ in 0..1_024 {
                    status_sum += unsafe {
                        call(
                            context,
                            arena,
                            momenta,
                            parameters,
                            factors,
                            ptr::null(),
                            3,
                            5,
                        )
                    };
                }
                status_sum
            });
            assert_eq!(status_sum, 1_024 * 1_008);
            assert_eq!(allocation_count, 0, "warmed native direct calls allocated");
            assert_eq!(allocated_bytes, 0, "warmed native direct calls allocated");
        }

        let missing_symbol =
            native_direct_symbol_name(DirectExecutorRole::Contribution, 23).unwrap();
        let error = expect_load_error(LoadedNativeDirectExecutor::load(
            library,
            DirectExecutorRole::Contribution,
            23,
            &missing_symbol,
            None,
        ));
        assert_eq!(error.kind(), RusticolErrorKind::Evaluation);

        drop(owners);
        fs::remove_dir_all(directory).unwrap();
    }
}
