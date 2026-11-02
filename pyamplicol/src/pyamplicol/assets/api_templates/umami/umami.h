/*
 *                                   _
 *                                  (_)
 *   _   _ _ __ ___   __ _ _ __ ___  _
 *  | | | | '_ ` _ \ / _` | '_ ` _ \| |
 *  | |_| | | | | | | (_| | | | | | | |
 *   \__,_|_| |_| |_|\__,_|_| |_| |_|_|
 *
 *  Unified  MAtrix  eleMent  Interface
 *
 *
 */

#ifndef UMAMI_HEADER
#define UMAMI_HEADER 1

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Major version number of the UMAMI interface. If the major version is the same
 * between caller and implementation, binary compatibility is ensured.
 */
#define UMAMI_MAJOR_VERSION 1
/**
 * Minor version number of the UMAMI interface. Between minor versions, new keys for
 * errors, devices, metadata, inputs and outputs can be added.
 */
#define UMAMI_MINOR_VERSION 2

typedef enum {
    /** operation was executed successfully */
    UMAMI_SUCCESS,
    /** an unspecified error */
    UMAMI_ERROR,
    /** operation not implemented */
    UMAMI_ERROR_NOT_IMPLEMENTED,
    /** the provided input key is not supported by this matrix element implementation */
    UMAMI_ERROR_UNSUPPORTED_INPUT,
    /** the provided output key is not supported by this matrix element implementation
     */
    UMAMI_ERROR_UNSUPPORTED_OUTPUT,
    /** the meta_key was not yet initialized */
    UMAMI_ERROR_UNINITIALIZED_META,
    /** the provided metadata key is not supported by this matrix element implementation
     */
    UMAMI_ERROR_UNSUPPORTED_META,
    /** a mandatory matrix element input was not provided */
    UMAMI_ERROR_MISSING_INPUT,
} UmamiStatus;

typedef enum {
    UMAMI_DEVICE_CPU,
    UMAMI_DEVICE_CUDA,
    UMAMI_DEVICE_HIP,
} UmamiDevice;

typedef enum {
    /** `UmamiDevice` specifying the type of device */
    UMAMI_META_DEVICE,
    /** `int` specifying the number of external particles */
    UMAMI_META_PARTICLE_COUNT,
    /** `int` specifying the number of Feynman diagrams */
    UMAMI_META_DIAGRAM_COUNT,
    /** `int` specifying the number of helicities */
    UMAMI_META_HELICITY_COUNT,
    /** `int` specifying the number of colors */
    UMAMI_META_COLOR_COUNT,
    /** `double` array of length particle count: the external-leg masses (GeV) */
    UMAMI_META_MASSES,
} UmamiMetaKey;

typedef enum {
    /** momenta of the external legs, type: `double`, shape: `(particle count, 4)` */
    UMAMI_IN_MOMENTA,
    /** value for the strong coupling, type: `double`, shape: `()` */
    UMAMI_IN_ALPHA_S,
    /** flavor index, type: `int`, shape: `()` */
    UMAMI_IN_FLAVOR_INDEX,
    /** random number for color selection, type: `double`, shape: `()` */
    UMAMI_IN_RANDOM_COLOR,
    /** random number for helicity selection, type: `double`, shape: `()` */
    UMAMI_IN_RANDOM_HELICITY,
    /** random number for diagram selection, type: `double`, shape: `()` */
    UMAMI_IN_RANDOM_DIAGRAM,
    /** externally selected helicity index, type: `int`, shape: `()` */
    UMAMI_IN_HELICITY_INDEX,
    /** externally selected diagram index, type: `unsigned int`, shape: `()` */
    UMAMI_IN_DIAGRAM_INDEX,
    /** externally selected channel index, type: `unsigned int`, shape: `()` */
    UMAMI_IN_CHANNEL_INDEX,
    /** number of external legs, type: `int`, shape: `()` */
    UMAMI_IN_PARTICLE_COUNT,
    /** PDG ids of the external legs, type: `int`, shape: `(particle count)` */
    UMAMI_IN_PDG_IDS,
    /** momentum fraction of an incoming parton, type: `double`, shape: `()` */
    UMAMI_IN_MOMENTUM_FRACTION,
    /** energy scale, e.g. the factorization scale, type: `double`, shape: `()` */
    UMAMI_IN_SCALE,
    /** PDG id of a single parton, type: `int`, shape: `()` */
    UMAMI_IN_PDG_ID,
    /** index of the beam (0 or 1), type: `int`, shape: `()` */
    UMAMI_IN_BEAM_INDEX,
} UmamiInputKey;

/** Number of values in `UmamiInputKey` */
#define UMAMI_INPUT_KEY_COUNT (UMAMI_IN_BEAM_INDEX + 1)

typedef enum {
    /** value of the matrix element, type: `double`, shape: `()` */
    UMAMI_OUT_MATRIX_ELEMENT,
    /** squared amplitudes of the diagrams, type: `double`, shape: `(diagram count)` */
    UMAMI_OUT_DIAGRAM_AMP2,
    /** selected color index, type: `int`, shape: `()` */
    UMAMI_OUT_COLOR_INDEX,
    /** selected helicity index, type: `int`, shape: `()` */
    UMAMI_OUT_HELICITY_INDEX,
    /** selected diagram index, type: `int`, shape: `()` */
    UMAMI_OUT_DIAGRAM_INDEX,
    /** CUDA or HIP stream for asynchronous execution. Listed as an output as it is a
     * mutable pointer */
    UMAMI_OUT_GPU_STREAM,
    /** value of a custom cut, 0 rejects the event, type: `double`, shape: `()` */
    UMAMI_OUT_CUT,
    /** value of a custom energy scale, type: `double`, shape: `()` */
    UMAMI_OUT_SCALE,
    /** value x f(x, Q) of a custom parton density, type: `double`, shape: `()` */
    UMAMI_OUT_PDF,
    /** value of a custom event bias, type: `double`, shape: `()` */
    UMAMI_OUT_BIAS,
    /** ratio of the matrix element to a reference matrix element defined by the
     * caller, type: `double`, shape: `()` */
    UMAMI_OUT_MATRIX_ELEMENT_RATIO,
} UmamiOutputKey;

/** Number of values in `UmamiOutputKey` */
#define UMAMI_OUTPUT_KEY_COUNT (UMAMI_OUT_MATRIX_ELEMENT_RATIO + 1)

/** Implementation-defined pointer to a matrix element instance */
typedef void* UmamiHandle;

/**
 * Retrieve metadata about the implemented matrix element
 *
 * @param meta_key
 *     key specifying the type of metadata to be retrieved
 * @param result
 *     pointer to store the result. It's type depends on the metadata key
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_get_meta(UmamiMetaKey meta_key, void* result);

/**
 * Reports which input keys can be passed to this matrix element implementation, to
 * be written by whoever implements the UMAMI interface. Implementing this function
 * is optional; a caller that fails to resolve it should treat it as unavailable.
 *
 * @param supported
 *     pointer set on return to an implementation-owned array of booleans, where
 *     element `i` indicates whether the input key with numeric value `i`
 *     (see `UmamiInputKey`) is accepted by this implementation.
 * @param count
 *     pointer set on return to the number of elements in `*supported`, reflecting
 *     the number of `UmamiInputKey` values known to this implementation. A caller
 *     built against a newer header should treat any key index beyond `*count` as
 *     not supported.
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_supported_inputs(bool const** supported, int* count);

/**
 * Reports which input keys must be passed to this matrix element implementation for
 * `umami_matrix_element` to succeed, to be written by whoever implements the UMAMI
 * interface. This is a subset of the keys reported by `umami_supported_inputs`.
 * Implementing this function is optional; a caller that fails to resolve it should
 * treat it as unavailable.
 *
 * @param required
 *     pointer set on return to an implementation-owned array of booleans, where
 *     element `i` indicates whether the input key with numeric value `i`
 *     (see `UmamiInputKey`) is mandatory for this implementation.
 * @param count
 *     pointer set on return to the number of elements in `*required`. A caller
 *     built against a newer header should treat any key index beyond `*count` as
 *     not required.
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_required_inputs(bool const** required, int* count);

/**
 * Reports which output keys can be requested from this matrix element
 * implementation, to be written by whoever implements the UMAMI interface.
 * Implementing this function is optional; a caller that fails to resolve it should
 * treat it as unavailable.
 *
 * @param supported
 *     pointer set on return to an implementation-owned array of booleans, where
 *     element `i` indicates whether the output key with numeric value `i`
 *     (see `UmamiOutputKey`) can be produced by this implementation.
 * @param count
 *     pointer set on return to the number of elements in `*supported`. A caller
 *     built against a newer header should treat any key index beyond `*count` as
 *     not supported.
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_supported_outputs(bool const** supported, int* count);

/**
 * Creates an instance of the matrix element. Each instance is independent, so thread
 * safety can be achieved by creating a separate one for every thread.
 *
 * @param handle
 *     pointer to an instance of the subprocess. Has to be cleaned up by
 *     the caller with `free_subprocess`.
 * @param param_card_path
 *     path to the parameter file
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_initialize(UmamiHandle* handle, char const* param_card_path);

/**
 * Sets the value of a model parameter
 *
 * @param handle
 *     handle of a matrix element instance
 * @param name
 *     name of the parameter
 * @param parameter_real
 *     real part of the parameter value
 * @param parameter_imag
 *     imaginary part of the parameter value. Ignored for real valued parameters.
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_set_parameter(
    UmamiHandle handle, char const* name, double parameter_real, double parameter_imag
);

/**
 * Retrieves the value of a model parameter
 *
 * @param handle
 *     handle of a matrix element instance
 * @param name
 *     name of the parameter
 * @param parameter_real
 *     pointer to double to return real part of the parameter value
 * @param parameter_imag
 *     pointer to double to return imaginary part of the parameter value. Ignored
 *     for real-valued parameters (i.e. you may pass a null pointer)
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_get_parameter(
    UmamiHandle handle, char const* name, double* parameter_real, double* parameter_imag
);

/**
 * Evaluates the matrix element as a function of the given inputs, filling the
 * requested outputs. Unless otherwise specified, all inputs and outputs have a
 * column-major memory layout and have a batch dimension that is contiguous in memory.
 *
 * @param handle
 *     handle of a matrix element instance
 * @param count
 *     number of events to evaluate the matrix element for
 * @param stride
 *     stride of the batch dimension of the input and output arrays to simplify
 *     parallel execution on CPUs, see memory layout
 * @param offset
 *     offset of the event index
 * @param input_count
 *     number of inputs to the matrix element
 * @param input_keys
 *     pointer to an array of input keys, length `input_count`
 * @param inputs
 *     pointer to an array of void pointers to the inputs. The type of the inputs
 *     depends on the input key
 * @param output_count
 *     number of outputs to the matrix element
 * @param output_keys
 *     pointer to an array of output keys, length `output_count`
 * @param outputs
 *     pointer to an array of void pointers to the outputs. The type of the outputs
 *     depends on the output key. The caller is responsible for allocating memory for
 *     the outputs.
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_matrix_element(
    UmamiHandle handle,
    size_t count,
    size_t stride,
    size_t offset,
    size_t input_count,
    UmamiInputKey const* input_keys,
    void const* const* inputs,
    size_t output_count,
    UmamiOutputKey const* output_keys,
    void* const* outputs
);

/**
 * Frees matrix element instance
 *
 * @param handle
 *     handle of a matrix element instance
 */
UmamiStatus umami_free(UmamiHandle handle);

#ifdef __cplusplus
}
#endif

#endif // UMAMI_HEADER
