/* SPDX-License-Identifier: 0BSD */
/* UMAMI provider of the ratio of a target-accuracy matrix element (e.g. full colour)
 * to the leading-colour one summed over all colour flows, at a fixed helicity. */
#include "umami.h"
#include "reweight_provider.h"
#include <rusticol.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef UMAMI_REWEIGHT_HEADER
#define UMAMI_REWEIGHT_HEADER "reweight_p0.h"
#endif
#include UMAMI_REWEIGHT_HEADER

typedef struct {
    char *name;
    double real, imaginary;
} Override;

typedef struct {
    char *base;
    RusticolRuntimeHandle **reference, **target;
    double *momenta;
    /* Parameters set so far, applied to the processes loaded later. */
    Override *overrides;
    size_t override_count;
} Instance;

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = UMAMI_HAS_ALPHA_S,
    [UMAMI_IN_FLAVOR_INDEX] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
    [UMAMI_IN_CHANNEL_INDEX] = true,
};
static bool const required_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
};
static bool const supported_outputs[UMAMI_OUTPUT_KEY_COUNT] = {
    [UMAMI_OUT_MATRIX_ELEMENT_RATIO] = true,
};

static UmamiStatus native_error(int status) {
    if (status == RUSTICOL_STATUS_OK) return UMAMI_SUCCESS;
    char message[1024];
    size_t required = 0;
    if (rusticol_last_error_message(message, sizeof(message), &required) ==
        RUSTICOL_STATUS_OK) fprintf(stderr, "umami: %s\n", message);
    return UMAMI_ERROR;
}

static UmamiStatus load(char const *base, char const *relative, char const *identity,
                        ReweightProcess const *process, RusticolRuntimeHandle **result) {
    size_t length = strlen(base) + strlen(relative) + 2;
    char *path = malloc(length);
    if (path == NULL) return UMAMI_ERROR;
    snprintf(path, length, "%s/%s", base, relative);
    RusticolRuntimeHandle *runtime = NULL;
    int status = rusticol_runtime_load(path, process->key, NULL, &runtime);
    free(path);
    if (status != RUSTICOL_STATUS_OK) return native_error(status);
    size_t count = 0, required = 0;
    status = rusticol_runtime_external_count(runtime, &count);
    bool compatible = status == RUSTICOL_STATUS_OK && count == reweight_provider.particle_count;
    for (size_t j = 0; compatible && j < count; ++j) {
        int32_t pdg = 0;
        status = rusticol_runtime_external_pdg(runtime, j, &pdg);
        compatible = status == RUSTICOL_STATUS_OK && pdg == process->pdgs[j];
    }
    if (compatible) {
        status = rusticol_runtime_helicity_count(runtime, &count);
        compatible = status == RUSTICOL_STATUS_OK && count == process->helicity_count;
    }
    if (compatible && identity != NULL) {
        char *loaded = malloc(strlen(identity) + 1);
        compatible = loaded != NULL && rusticol_runtime_artifact_id(
            runtime, loaded, strlen(identity) + 1, &required) == RUSTICOL_STATUS_OK &&
            strcmp(loaded, identity) == 0;
        free(loaded);
    }
    if (!compatible) {
        (void)rusticol_runtime_free(runtime);
        fprintf(stderr, "umami: artifact %s is incompatible with reweight provider %s\n",
                relative, reweight_provider.id);
        return UMAMI_ERROR;
    }
    *result = runtime;
    return UMAMI_SUCCESS;
}

/* Loads a reference (side 0) or target process on first use, with the parameters set. */
static UmamiStatus ensure_loaded(Instance *instance, int side, size_t index) {
    RusticolRuntimeHandle **slot =
        side == 0 ? &instance->reference[index] : &instance->target[index];
    if (*slot != NULL) return UMAMI_SUCCESS;
    RusticolRuntimeHandle *runtime = NULL;
    UmamiStatus status = side == 0
        ? load(instance->base, reweight_provider.reference_artifact,
               reweight_provider.reference_artifact_id,
               &reweight_provider.reference_processes[index], &runtime)
        : load(instance->base, reweight_provider.target_artifact,
               reweight_provider.target_artifact_id,
               &reweight_provider.target_processes[index], &runtime);
    if (status != UMAMI_SUCCESS) return status;
    for (size_t i = 0; i < instance->override_count; ++i) {
        Override const *override = &instance->overrides[i];
        double re = 0.0, im = 0.0;
        if (rusticol_runtime_get_model_parameter(runtime, override->name, &re, &im) !=
            RUSTICOL_STATUS_OK) continue;
        status = native_error(rusticol_runtime_set_model_parameter(
            runtime, override->name, override->real, override->imaginary));
        if (status != UMAMI_SUCCESS) { (void)rusticol_runtime_free(runtime); return status; }
    }
    *slot = runtime;
    return UMAMI_SUCCESS;
}

static UmamiStatus record_override(Instance *instance, char const *name,
                                   double real, double imaginary) {
    for (size_t i = 0; i < instance->override_count; ++i)
        if (strcmp(instance->overrides[i].name, name) == 0) {
            instance->overrides[i].real = real;
            instance->overrides[i].imaginary = imaginary;
            return UMAMI_SUCCESS;
        }
    Override *overrides = realloc(instance->overrides,
                                  (instance->override_count + 1) * sizeof(*overrides));
    if (overrides == NULL) return UMAMI_ERROR;
    instance->overrides = overrides;
    char *copy = malloc(strlen(name) + 1);
    if (copy == NULL) return UMAMI_ERROR;
    strcpy(copy, name);
    overrides[instance->override_count++] = (Override){copy, real, imaginary};
    return UMAMI_SUCCESS;
}

UmamiStatus umami_get_meta(UmamiMetaKey key, void *result) {
    if (result == NULL) return UMAMI_ERROR;
    switch (key) {
    case UMAMI_META_DEVICE: *(UmamiDevice *)result = UMAMI_DEVICE_CPU; break;
    case UMAMI_META_PARTICLE_COUNT: *(int *)result = (int)reweight_provider.particle_count; break;
    case UMAMI_META_HELICITY_COUNT:
        *(int *)result = (int)reweight_provider.exported_helicity_count; break;
    default: return UMAMI_ERROR_UNSUPPORTED_META;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_supported_inputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_required_inputs(bool const **required, int *count) {
    if (required == NULL || count == NULL) return UMAMI_ERROR;
    *required = required_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_supported_outputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_outputs; *count = UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

UmamiStatus umami_initialize(UmamiHandle *handle, char const *param_card_path) {
    if (handle == NULL) return UMAMI_ERROR;
    *handle = NULL;
    struct stat info;
    if (param_card_path == NULL || param_card_path[0] != '/' ||
        stat(param_card_path, &info) != 0 || !S_ISDIR(info.st_mode) ||
        reweight_provider.particle_count > INT_MAX) return UMAMI_ERROR;
    Instance *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) return UMAMI_ERROR;
    instance->reference = calloc(reweight_provider.reference_process_count + 1,
                                 sizeof(*instance->reference));
    instance->target = calloc(reweight_provider.target_process_count + 1,
                              sizeof(*instance->target));
    instance->momenta = calloc(4 * reweight_provider.particle_count, sizeof(double));
    instance->base = malloc(strlen(param_card_path) + 1);
    if (instance->reference == NULL || instance->target == NULL || instance->momenta == NULL ||
        instance->base == NULL || reweight_provider.reference_process_count == 0 ||
        reweight_provider.target_process_count == 0) {
        umami_free(instance); return UMAMI_ERROR;
    }
    strcpy(instance->base, param_card_path);
    /* Validate both artifacts now; other processes are loaded lazily on first use. */
    for (int side = 0; side < 2; ++side) {
        UmamiStatus status = ensure_loaded(instance, side, 0);
        if (status != UMAMI_SUCCESS) { umami_free(instance); return status; }
    }
    *handle = instance;
    return UMAMI_SUCCESS;
}

/* Runs body for every loaded runtime, reference ones first. */
#define FOR_EACH_RUNTIME(instance, runtime, body) \
    for (size_t side = 0; side < 2; ++side) { \
        size_t count = side == 0 ? reweight_provider.reference_process_count \
                                 : reweight_provider.target_process_count; \
        for (size_t i = 0; i < count; ++i) { \
            RusticolRuntimeHandle *runtime = \
                side == 0 ? (instance)->reference[i] : (instance)->target[i]; \
            if (runtime == NULL) continue; \
            body \
        } \
    }

UmamiStatus umami_get_parameter(UmamiHandle handle, char const *name,
                              double *real, double *imaginary) {
    if (handle == NULL || name == NULL || real == NULL) return UMAMI_ERROR;
    Instance *instance = handle;
    double re = 0.0, im = 0.0;
    FOR_EACH_RUNTIME(instance, runtime, {
        if (rusticol_runtime_get_model_parameter(runtime, name, &re, &im) ==
            RUSTICOL_STATUS_OK) {
            *real = re;
            if (imaginary != NULL) *imaginary = im;
            return UMAMI_SUCCESS;
        }
    })
    return UMAMI_ERROR;
}

UmamiStatus umami_set_parameter(UmamiHandle handle, char const *name,
                              double real, double imaginary) {
    if (handle == NULL || name == NULL || !isfinite(real)) return UMAMI_ERROR;
    Instance *instance = handle;
    bool found = false;
    double re = 0.0, im = 0.0;
    FOR_EACH_RUNTIME(instance, runtime, {
        if (rusticol_runtime_get_model_parameter(runtime, name, &re, &im) !=
            RUSTICOL_STATUS_OK) continue;
        found = true;
        if (native_error(rusticol_runtime_set_model_parameter(runtime, name, real, imaginary)) !=
            UMAMI_SUCCESS) return UMAMI_ERROR;
    })
    if (!found) return UMAMI_ERROR;
    return record_override(instance, name, real, imaginary);
}

/* Value of one process at a fixed runtime helicity, with an optional alpha_s. */
static UmamiStatus evaluate(Instance *instance, RusticolRuntimeHandle *runtime,
                            ReweightProcess const *process, uint32_t helicity,
                            double const *alpha_s, double *result) {
    size_t n = reweight_provider.particle_count;
    double previous_real = 0.0, previous_imaginary = 0.0;
    bool changed = false;
    UmamiStatus status;
    if (alpha_s != NULL) {
        if (process->alpha_s_name == NULL) return UMAMI_ERROR_UNSUPPORTED_INPUT;
        status = native_error(rusticol_runtime_get_model_parameter(
            runtime, process->alpha_s_name, &previous_real, &previous_imaginary));
        if (status != UMAMI_SUCCESS) return status;
        changed = previous_real != *alpha_s || previous_imaginary != 0.0;
        if (changed) {
            status = native_error(rusticol_runtime_set_model_parameter(
                runtime, process->alpha_s_name, *alpha_s, 0.0));
            if (status != UMAMI_SUCCESS) return status;
        }
    }
    status = native_error(rusticol_runtime_evaluate_selected_f64(
        runtime, instance->momenta, 4*n, 1, NULL, 0, NULL, 0,
        &helicity, 1, NULL, 0, result, 1));
    if (changed) {
        UmamiStatus restored = native_error(rusticol_runtime_set_model_parameter(
            runtime, process->alpha_s_name, previous_real, previous_imaginary));
        if (restored != UMAMI_SUCCESS) return restored;
    }
    return status;
}

UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t input_count, UmamiInputKey const *input_keys, void const *const *inputs,
    size_t output_count, UmamiOutputKey const *output_keys, void *const *outputs) {
    if (handle == NULL || offset > stride || count > stride-offset ||
        (input_count && (input_keys == NULL || inputs == NULL)) ||
        (output_count && (output_keys == NULL || outputs == NULL)) ||
        (stride && reweight_provider.particle_count > SIZE_MAX / stride / 4)) return UMAMI_ERROR;
    void const *in[UMAMI_INPUT_KEY_COUNT] = {0};
    void *out[UMAMI_OUTPUT_KEY_COUNT] = {0};
    for (size_t i = 0; i < input_count; ++i) {
        int key = input_keys[i];
        if (key < 0 || key >= UMAMI_INPUT_KEY_COUNT || !supported_inputs[key])
            return UMAMI_ERROR_UNSUPPORTED_INPUT;
        if (inputs[i] == NULL || in[key] != NULL) return UMAMI_ERROR;
        in[key] = inputs[i];
    }
    for (size_t i = 0; i < output_count; ++i) {
        int key = output_keys[i];
        if (key < 0 || key >= UMAMI_OUTPUT_KEY_COUNT || !supported_outputs[key])
            return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
        if (outputs[i] == NULL || out[key] != NULL) return UMAMI_ERROR;
        out[key] = outputs[i];
    }
    if (count == 0) return UMAMI_SUCCESS;
    if (in[UMAMI_IN_MOMENTA] == NULL || in[UMAMI_IN_HELICITY_INDEX] == NULL)
        return UMAMI_ERROR_MISSING_INPUT;
    Instance *instance = handle;
    double const *momenta = in[UMAMI_IN_MOMENTA];
    size_t n = reweight_provider.particle_count;
    for (size_t event = offset; event < offset+count; ++event) {
        size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ?
            ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
        int flavour = in[UMAMI_IN_FLAVOR_INDEX] ?
            ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
        int helicity = ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event];
        if (channel >= reweight_provider.channel_count || flavour < 0 ||
            (size_t)flavour >= reweight_provider.channels[channel].flavour_count ||
            helicity < 0 || (size_t)helicity >= reweight_provider.exported_helicity_count)
            return UMAMI_ERROR;
        ReweightFlavour const *entry = &reweight_provider.channels[channel].flavours[flavour];
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < 4; ++k) {
                double value = momenta[event + stride * (n*k + entry->permutation[j])];
                if (!isfinite(value)) return UMAMI_ERROR;
                instance->momenta[4*j+k] = value;
            }
        double const *alpha_s =
            in[UMAMI_IN_ALPHA_S] ? &((double const *)in[UMAMI_IN_ALPHA_S])[event] : NULL;
        int reference_helicity = entry->reference_helicity[helicity];
        int target_helicity = entry->target_helicity[helicity];
        double reference = 0.0, target = 0.0;
        if ((reference_helicity >= 0 &&
             ensure_loaded(instance, 0, entry->reference_process) != UMAMI_SUCCESS) ||
            (target_helicity >= 0 &&
             ensure_loaded(instance, 1, entry->target_process) != UMAMI_SUCCESS))
            return UMAMI_ERROR;
        if (reference_helicity >= 0) {
            UmamiStatus status = evaluate(instance, instance->reference[entry->reference_process],
                &reweight_provider.reference_processes[entry->reference_process],
                (uint32_t)reference_helicity, alpha_s, &reference);
            if (status != UMAMI_SUCCESS) return status;
        }
        if (target_helicity >= 0) {
            UmamiStatus status = evaluate(instance, instance->target[entry->target_process],
                &reweight_provider.target_processes[entry->target_process],
                (uint32_t)target_helicity, alpha_s, &target);
            if (status != UMAMI_SUCCESS) return status;
        }
        double ratio = reference != 0.0 ? target / reference : 0.0;
        if (!isfinite(ratio)) return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT_RATIO])
            ((double *)out[UMAMI_OUT_MATRIX_ELEMENT_RATIO])[event] = ratio;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
    if (instance->reference != NULL)
        for (size_t i = 0; i < reweight_provider.reference_process_count; ++i)
            if (instance->reference[i] != NULL) (void)rusticol_runtime_free(instance->reference[i]);
    if (instance->target != NULL)
        for (size_t i = 0; i < reweight_provider.target_process_count; ++i)
            if (instance->target[i] != NULL) (void)rusticol_runtime_free(instance->target[i]);
    for (size_t i = 0; i < instance->override_count; ++i) free(instance->overrides[i].name);
    free(instance->overrides); free(instance->base);
    free(instance->reference); free(instance->target); free(instance->momenta); free(instance);
    return UMAMI_SUCCESS;
}
