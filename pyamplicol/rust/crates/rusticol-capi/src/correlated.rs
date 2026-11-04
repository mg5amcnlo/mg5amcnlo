// SPDX-License-Identifier: 0BSD
//! Additive correlated evaluation ABI. Ordinary calls do not enter this module.

use super::*;

unsafe fn read_spin_vectors(
    vectors: *const RusticolSpinCorrelationVector,
    count: usize,
) -> AbiResult<NativeSpinCorrelationVectors> {
    let mut result = NativeSpinCorrelationVectors::new();
    if count == 0 {
        return Ok(result);
    }
    if vectors.is_null() {
        return Err(invalid("spin-vector array is null"));
    }
    // SAFETY: The caller supplies count readable records.
    for vector in unsafe { slice::from_raw_parts(vectors, count) } {
        if vector.leg == 0 || vector.point_count == 0 {
            return Err(invalid("spin-vector leg and point count must be positive"));
        }
        let required = vector
            .point_count
            .checked_mul(8)
            .ok_or_else(|| invalid("spin-vector size overflow"))?;
        if vector.component_count != required {
            return Err(invalid(
                "spin-vector components must contain eight doubles per point",
            ));
        }
        // SAFETY: This record's component buffer is readable for component_count doubles.
        let flat = unsafe {
            read_f64_slice(vector.components, vector.component_count, "spin components")
        }?;
        if flat.iter().any(|value| !value.is_finite()) {
            return Err(invalid("spin-vector components must be finite"));
        }
        let points = flat
            .chunks_exact(8)
            .map(|point| std::array::from_fn(|mu| [point[2 * mu], point[2 * mu + 1]]))
            .collect();
        if result.insert(vector.leg, points).is_some() {
            return Err(invalid("a spin-vector array repeats an external leg"));
        }
    }
    Ok(result)
}

unsafe fn read_requests(
    requests: *const RusticolCorrelatedRequest,
    count: usize,
) -> AbiResult<Vec<NativeCorrelatedRequest>> {
    if count == 0 || requests.is_null() {
        return Err(invalid(
            "a correlated evaluation requires a nonempty request array",
        ));
    }
    // SAFETY: The caller supplies count readable records and their nested buffers.
    unsafe { slice::from_raw_parts(requests, count) }
        .iter()
        .map(|request| {
            let identifier =
                unsafe { required_c_string(request.color_correlation, "colour correlation ID") }?;
            if identifier.is_empty() {
                return Err(invalid("colour correlation ID must not be empty"));
            }
            let spin_vectors = match request.use_default_spin_vectors {
                0 => Some(unsafe {
                    read_spin_vectors(request.spin_vectors, request.spin_vector_count)
                }?),
                1 if request.spin_vector_count == 0 => None,
                1 => {
                    return Err(invalid(
                        "inherited spin vectors require an empty explicit array",
                    ));
                }
                _ => return Err(invalid("use_default_spin_vectors must be zero or one")),
            };
            Ok(NativeCorrelatedRequest {
                color_correlation: identifier.to_owned(),
                spin_vectors,
            })
        })
        .collect()
}

/// Enumerate the selected process's generation-time correlation IDs.
///
/// # Safety
/// handle must be live and exclusively accessible; output must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_color_correlation_count(
    handle: *mut RusticolRuntimeHandle,
    output: *mut size_t,
) -> c_int {
    guard(|| {
        let handle = unsafe { required_handle_mut(handle) }?;
        let ids = handle.runtime.color_correlation_ids()?;
        unsafe { write_size(ids.len(), output, "colour correlation count") }
    })
}

/// Copy one correlation ID. NULL/zero buffer queries its NUL-inclusive size.
///
/// # Safety
/// handle must be live/exclusive; required and buffer follow write_string's contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_color_correlation_id(
    handle: *mut RusticolRuntimeHandle,
    index: size_t,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        let handle = unsafe { required_handle_mut(handle) }?;
        let ids = handle.runtime.color_correlation_ids()?;
        let value = ids
            .get(index)
            .ok_or_else(|| invalid("colour correlation index out of range"))?;
        unsafe { write_string(value, buffer, capacity, required) }
    })
}

/// Copy the full ordered catalogue as JSON, including generation-time histories.
///
/// # Safety
/// Pointer requirements match rusticol_runtime_color_correlation_id.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_color_correlation_catalogue_json(
    handle: *mut RusticolRuntimeHandle,
    buffer: *mut c_char,
    capacity: size_t,
    required: *mut size_t,
) -> c_int {
    guard(|| {
        let handle = unsafe { required_handle_mut(handle) }?;
        let value = handle.runtime.available_color_correlations_json()?;
        unsafe { write_string(&value, buffer, capacity, required) }
    })
}

/// Copy literal spin-vector defaults. An empty array clears them.
///
/// # Safety
/// handle must be live/exclusive; vectors and nested component buffers must be readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_set_spin_correlation_vectors_f64(
    handle: *mut RusticolRuntimeHandle,
    vectors: *const RusticolSpinCorrelationVector,
    vector_count: size_t,
) -> c_int {
    guard(|| {
        let handle = unsafe { required_handle_mut(handle) }?;
        let vectors = unsafe { read_spin_vectors(vectors, vector_count) }?;
        handle.runtime.set_spin_correlation_vectors_f64(vectors)?;
        Ok(())
    })
}

/// Evaluate grouped colour/spin requests; result doubles are [request][point][real,imag].
///
/// # Safety
/// handle must be live/exclusive. All arrays must be readable for their stated counts;
/// nested request/vector/string buffers must remain live. output must be writable for
/// output_capacity doubles and must not overlap input buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_runtime_evaluate_correlated_many_f64(
    handle: *mut RusticolRuntimeHandle,
    momenta: *const c_double,
    momentum_count: size_t,
    point_count: size_t,
    requests: *const RusticolCorrelatedRequest,
    request_count: size_t,
    helicity_ids: *const *const c_char,
    helicity_count: size_t,
    output: *mut c_double,
    output_capacity: size_t,
) -> c_int {
    guard(|| {
        let required = request_count
            .checked_mul(point_count)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("correlated result size overflow"))?;
        if point_count == 0 || request_count == 0 {
            return Err(invalid(
                "correlated request and point counts must be positive",
            ));
        }
        validate_f64_output(output, output_capacity, required, "correlated output")?;
        let handle = unsafe { required_handle_mut(handle) }?;
        let momenta = unsafe { read_f64_slice(momenta, momentum_count, "momenta") }?;
        let requests = unsafe { read_requests(requests, request_count) }?;
        let helicities =
            unsafe { read_selector_ids(helicity_ids, helicity_count, "helicity IDs") }?;
        let values = handle.runtime.evaluate_correlated_many_f64(
            momenta,
            point_count,
            &requests,
            helicities.as_deref(),
        )?;
        if values.point_count != point_count
            || values.request_count != request_count
            || values.values.len().checked_mul(2) != Some(required)
        {
            return Err(abi_error(
                RUSTICOL_STATUS_RUNTIME_ERROR,
                "correlated runtime returned inconsistent dimensions",
            ));
        }
        let flat: Vec<f64> = values.values.into_iter().flatten().collect();
        unsafe { write_f64_slice(&flat, output, output_capacity, "correlated output") }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_buffers_preserve_complex_layout_and_reject_duplicate_legs() {
        let components = [1., 2., 3., 4., 5., 6., 7., 8.];
        let vector = RusticolSpinCorrelationVector {
            leg: 3,
            point_count: 1,
            components: components.as_ptr(),
            component_count: 8,
        };
        let result = unsafe { read_spin_vectors(&vector, 1) }.ok().unwrap();
        assert_eq!(result[&3], vec![[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]]);
        let duplicated = [vector, vector];
        assert!(unsafe { read_spin_vectors(duplicated.as_ptr(), 2) }.is_err());
        assert!(
            unsafe { read_spin_vectors(ptr::null(), 0) }
                .ok()
                .unwrap()
                .is_empty()
        );
        assert!(unsafe { read_spin_vectors(ptr::null(), 1) }.is_err());
        let wrong = RusticolSpinCorrelationVector {
            component_count: 7,
            ..vector
        };
        assert!(unsafe { read_spin_vectors(&wrong, 1) }.is_err());
    }

    #[test]
    fn requests_distinguish_defaults_from_physical_helicities() {
        let request = RusticolCorrelatedRequest {
            color_correlation: c"born".as_ptr(),
            spin_vectors: ptr::null(),
            spin_vector_count: 0,
            use_default_spin_vectors: 1,
        };
        assert!(
            unsafe { read_requests(&request, 1) }.ok().unwrap()[0]
                .spin_vectors
                .is_none()
        );
        let explicit = RusticolCorrelatedRequest {
            use_default_spin_vectors: 0,
            ..request
        };
        assert!(
            unsafe { read_requests(&explicit, 1) }.ok().unwrap()[0]
                .spin_vectors
                .as_ref()
                .unwrap()
                .is_empty()
        );
        let invalid = RusticolCorrelatedRequest {
            use_default_spin_vectors: 2,
            ..request
        };
        assert!(unsafe { read_requests(&invalid, 1) }.is_err());
        let mixed = RusticolCorrelatedRequest {
            spin_vector_count: 1,
            ..request
        };
        assert!(unsafe { read_requests(&mixed, 1) }.is_err());
        assert!(unsafe { read_requests(ptr::null(), 0) }.is_err());
    }
}
