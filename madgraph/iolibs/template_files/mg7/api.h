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

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Major version number of the UMAMI interface. If the major version is the same
 * between caller and implementation, binary compatibility is ensured.
 */
const int UMAMI_MAJOR_VERSION = 1;
/**
 * Minor version number of the UMAMI interface. Between minor versions, new keys for
 * errors, devices, metadata, inputs and outputs can be added.
 */
const int UMAMI_MINOR_VERSION = 0;

typedef enum {
    UMAMI_SUCCESS,
    UMAMI_ERROR,
    UMAMI_ERROR_NOT_IMPLEMENTED,
    UMAMI_ERROR_UNSUPPORTED_INPUT,
    UMAMI_ERROR_UNSUPPORTED_OUTPUT,
    UMAMI_ERROR_UNSUPPORTED_META,
    UMAMI_ERROR_MISSING_INPUT,
} UmamiStatus;

typedef enum {
    UMAMI_DEVICE_CPU,
    UMAMI_DEVICE_CUDA,
    UMAMI_DEVICE_HIP,
} UmamiDevice;

typedef enum {
    UMAMI_META_DEVICE,
    UMAMI_META_PARTICLE_COUNT,
    UMAMI_META_DIAGRAM_COUNT,
    UMAMI_META_HELICITY_COUNT,
    UMAMI_META_COLOR_COUNT,
} UmamiMetaKey;

typedef enum {
    UMAMI_IN_MOMENTA,
    UMAMI_IN_ALPHA_S,
    UMAMI_IN_FLAVOR_INDEX,
    UMAMI_IN_RANDOM_COLOR,
    UMAMI_IN_RANDOM_HELICITY,
    UMAMI_IN_RANDOM_DIAGRAM,
    UMAMI_IN_HELICITY_INDEX,
    UMAMI_IN_DIAGRAM_INDEX,
} UmamiInputKey;

typedef enum {
    UMAMI_OUT_MATRIX_ELEMENT,
    UMAMI_OUT_DIAGRAM_AMP2,
    UMAMI_OUT_COLOR_INDEX,
    UMAMI_OUT_HELICITY_INDEX,
    UMAMI_OUT_DIAGRAM_INDEX,
    UMAMI_OUT_GPU_STREAM,
    // NLO: born, virtual, poles, counterterms
    // color: LC-ME, FC-ME
} UmamiOutputKey;

typedef void* UmamiHandle;


/**
 * Creates an instance of the matrix element. Each instance is independent, so thread
 * safety can be achieved by creating a separate one for every thread.
 *
 * @param meta_key
 *     path to the parameter file
 * @param handle
 *     pointer to an instance of the subprocess. Has to be cleaned up by
 *     the caller with `free_subprocess`.
 * @return
 *     UMAMI_SUCCESS on success, error code otherwise
 */
UmamiStatus umami_get_meta(UmamiMetaKey meta_key, void* result);

/**
 * Creates an instance of the matrix element. Each instance is independent, so thread
 * safety can be achieved by creating a separate one for every thread.
 *
 * @param param_card_path
 *     path to the parameter file
 * @param handle
 *     pointer to an instance of the subprocess. Has to be cleaned up by
 *     the caller with `free_subprocess`.
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
    UmamiHandle handle,
    char const* name,
    double parameter_real,
    double parameter_imag
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
    UmamiHandle handle,
    char const* name,
    double* parameter_real,
    double* parameter_imag
);

/**
 * Evaluates the matrix element as a function of the given inputs, filling the
 * requested outputs.
 *
 * @param handle
 *     handle of a matrix element instance
 * @param count
 *     number of events to evaluate the matrix element for
 * @param stride
 *     stride of the batch dimension of the input and output arrays, see memory layout
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
