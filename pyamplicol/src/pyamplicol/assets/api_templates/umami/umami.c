/* SPDX-License-Identifier: 0BSD */
#include "umami.h"
#include "umami_provider.h"
#include <rusticol.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef UMAMI_PROVIDER_HEADER
#define UMAMI_PROVIDER_HEADER "provider_p0.h"
#endif
#include UMAMI_PROVIDER_HEADER

typedef struct {
    char *name;
    double real, imaginary;
} Override;

typedef struct {
    char *artifact;
    RusticolRuntimeHandle **runtimes;
    double *momenta;
    double *helicity_values;
    /* Parameters set so far, applied to the processes loaded later. */
    Override *overrides;
    size_t override_count;
} Instance;

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = UMAMI_HAS_ALPHA_S,
    [UMAMI_IN_FLAVOR_INDEX] = true,
    [UMAMI_IN_RANDOM_HELICITY] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
    [UMAMI_IN_CHANNEL_INDEX] = true,
};
static bool const required_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
};
static bool const supported_outputs[UMAMI_OUTPUT_KEY_COUNT] = {
    [UMAMI_OUT_MATRIX_ELEMENT] = true,
    [UMAMI_OUT_COLOR_INDEX] = UMAMI_HAS_COLOR_FLOW,
    [UMAMI_OUT_HELICITY_INDEX] = true,
};

static UmamiStatus native_error(int status) {
    if (status == RUSTICOL_STATUS_OK) return UMAMI_SUCCESS;
    char message[1024];
    size_t required = 0;
    if (rusticol_last_error_message(message, sizeof(message), &required) ==
        RUSTICOL_STATUS_OK) fprintf(stderr, "umami: %s\n", message);
    return UMAMI_ERROR;
}

static UmamiParameter const *parameter(size_t process, char const *name) {
    UmamiProcess const *entry = &umami_provider.processes[process];
    for (size_t i = 0; i < entry->parameter_count; ++i)
        if (strcmp(entry->parameters[i].name, name) == 0) return &entry->parameters[i];
    return NULL;
}

static UmamiStatus load_process(Instance *instance, size_t process) {
    if (instance->runtimes[process] != NULL) return UMAMI_SUCCESS;
    UmamiProcess const *entry = &umami_provider.processes[process];
    RusticolRuntimeHandle *runtime = NULL;
    int status = rusticol_runtime_load(instance->artifact, entry->key, NULL, &runtime);
    if (status != RUSTICOL_STATUS_OK) return native_error(status);
    size_t count = 0, required = 0;
    char accuracy[16];
    status = rusticol_runtime_external_count(runtime, &count);
    bool compatible = status == RUSTICOL_STATUS_OK && count == umami_provider.particle_count;
    char const *expected_identity = UMAMI_ARTIFACT_ID;
    if (compatible && expected_identity != NULL) {
        size_t capacity = strlen(expected_identity) + 1;
        char *identity = malloc(capacity);
        if (identity == NULL) compatible = false;
        else {
            status = rusticol_runtime_artifact_id(runtime, identity, capacity, &required);
            compatible = status == RUSTICOL_STATUS_OK && strcmp(identity, expected_identity) == 0;
            free(identity);
        }
    }
    for (size_t j = 0; compatible && j < count; ++j) {
        int32_t pdg = 0;
        status = rusticol_runtime_external_pdg(runtime, j, &pdg);
        compatible = status == RUSTICOL_STATUS_OK && pdg == entry->pdgs[j];
    }
    if (compatible) {
        status = rusticol_runtime_color_accuracy(runtime, accuracy, sizeof(accuracy), &required);
        compatible = status == RUSTICOL_STATUS_OK && strcmp(accuracy, entry->color_accuracy) == 0;
    }
    if (compatible) {
        status = rusticol_runtime_helicity_count(runtime, &count);
        compatible = status == RUSTICOL_STATUS_OK && count == entry->helicity_count;
    }
    for (size_t j = 0; compatible && j < entry->helicity_count; ++j) {
        size_t capacity = strlen(entry->helicity_ids[j]) + 1;
        char *identifier = malloc(capacity);
        if (identifier == NULL) { compatible = false; break; }
        status = rusticol_runtime_helicity_id(runtime, j, identifier, capacity, &required);
        compatible = status == RUSTICOL_STATUS_OK && strcmp(identifier, entry->helicity_ids[j]) == 0;
        free(identifier);
    }
    if (!compatible) {
        (void)rusticol_runtime_free(runtime);
        fprintf(stderr, "umami: artifact is incompatible with provider %s, process %s\n",
                umami_provider.id, entry->key);
        return UMAMI_ERROR;
    }
    for (size_t i = 0; i < instance->override_count; ++i) {
        Override const *override = &instance->overrides[i];
        UmamiParameter const *known = parameter(process, override->name);
        if (known == NULL) continue;
        status = rusticol_runtime_set_model_parameter(runtime, override->name,
            override->real, known->is_complex ? override->imaginary : 0.0);
        if (status != RUSTICOL_STATUS_OK) {
            (void)rusticol_runtime_free(runtime);
            return native_error(status);
        }
    }
    instance->runtimes[process] = runtime;
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
    case UMAMI_META_PARTICLE_COUNT: *(int *)result = (int)umami_provider.particle_count; break;
    case UMAMI_META_HELICITY_COUNT: *(int *)result = (int)umami_provider.helicity_count; break;
    case UMAMI_META_COLOR_COUNT:
        if (!UMAMI_HAS_COLOR_FLOW) return UMAMI_ERROR_UNSUPPORTED_META;
        *(int *)result = (int)umami_provider.color_count; break;
    case UMAMI_META_MASSES:
        memcpy(result, umami_provider.masses, umami_provider.particle_count * sizeof(double));
        break;
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
        umami_provider.process_count == 0 || umami_provider.particle_count > INT_MAX ||
        umami_provider.particle_count > SIZE_MAX / (4 * sizeof(double))) return UMAMI_ERROR;
    Instance *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) return UMAMI_ERROR;
    size_t length = strlen(param_card_path) + 1;
    instance->artifact = malloc(length);
    instance->runtimes = calloc(umami_provider.process_count, sizeof(*instance->runtimes));
    instance->momenta = calloc(4 * umami_provider.particle_count, sizeof(double));
    instance->helicity_values = calloc(umami_provider.helicity_count + 1, sizeof(double));
    if (instance->artifact == NULL || instance->runtimes == NULL || instance->momenta == NULL ||
        instance->helicity_values == NULL) {
        umami_free(instance); return UMAMI_ERROR;
    }
    memcpy(instance->artifact, param_card_path, length);
    /* Validate the artifact now; other processes are loaded lazily on first use. */
    UmamiStatus status = load_process(instance, 0);
    if (status != UMAMI_SUCCESS) { umami_free(instance); return status; }
    *handle = instance;
    return UMAMI_SUCCESS;
}

UmamiStatus umami_get_parameter(UmamiHandle handle, char const *name,
                              double *real, double *imaginary) {
    if (handle == NULL || name == NULL || real == NULL) return UMAMI_ERROR;
    Instance *instance = handle;
    for (size_t i = 0; i < umami_provider.process_count; ++i) {
        UmamiParameter const *entry = parameter(i, name);
        if (entry == NULL) continue;
        if (entry->is_complex && imaginary == NULL) return UMAMI_ERROR;
        UmamiStatus status = load_process(instance, i);
        if (status != UMAMI_SUCCESS) return status;
        double re, im;
        status = native_error(rusticol_runtime_get_model_parameter(instance->runtimes[i], name, &re, &im));
        if (status != UMAMI_SUCCESS) return status;
        *real = re;
        if (imaginary != NULL) *imaginary = im;
        return UMAMI_SUCCESS;
    }
    return UMAMI_ERROR;
}

UmamiStatus umami_set_parameter(UmamiHandle handle, char const *name,
                              double real, double imaginary) {
    if (handle == NULL || name == NULL || !isfinite(real)) return UMAMI_ERROR;
    Instance *instance = handle;
    size_t count = umami_provider.process_count;
    double *previous = calloc(count * 2, sizeof(double));
    bool *changed = calloc(count, sizeof(bool));
    if (previous == NULL || changed == NULL) { free(previous); free(changed); return UMAMI_ERROR; }
    UmamiStatus status = UMAMI_ERROR;
    bool found = false;
    for (size_t i = 0; i < count; ++i) {
        UmamiParameter const *entry = parameter(i, name);
        if (entry == NULL) continue;
        /* Load the first process using it, so that the value is validated now. */
        if (!found && load_process(instance, i) != UMAMI_SUCCESS) goto rollback;
        found = true;
        if (!entry->mutable || (entry->is_complex && !isfinite(imaginary))) goto rollback;
        if (instance->runtimes[i] == NULL) continue;
        status = native_error(rusticol_runtime_get_model_parameter(
            instance->runtimes[i], name, &previous[2*i], &previous[2*i+1]));
        if (status != UMAMI_SUCCESS) goto rollback;
        status = native_error(rusticol_runtime_set_model_parameter(
            instance->runtimes[i], name, real, entry->is_complex ? imaginary : 0.0));
        if (status != UMAMI_SUCCESS) goto rollback;
        changed[i] = true;
    }
    if (found && record_override(instance, name, real, imaginary) != UMAMI_SUCCESS)
        goto rollback;
    free(previous); free(changed);
    return found ? UMAMI_SUCCESS : UMAMI_ERROR;
rollback:
    for (size_t i = 0; i < count; ++i)
        if (changed[i]) (void)rusticol_runtime_set_model_parameter(
            instance->runtimes[i], name, previous[2*i], previous[2*i+1]);
    free(previous); free(changed);
    return UMAMI_ERROR;
}

/* Total of one contribution, at a fixed runtime helicity if helicity is not NULL. With
 * values, the values of all runtime helicities are written there and summed. */
static UmamiStatus evaluate_contribution(Instance *instance, UmamiContribution const *entry,
                                        double const *momenta, size_t event, size_t stride,
                                        double const *alpha_s, uint32_t const *helicity,
                                        double *values, double *result) {
    size_t process = entry->process_index;
    UmamiProcess const *description = &umami_provider.processes[process];
    UmamiStatus status = load_process(instance, process);
    if (status != UMAMI_SUCCESS) return status;
    RusticolRuntimeHandle *runtime = instance->runtimes[process];
    size_t n = umami_provider.particle_count;
    for (size_t j = 0; j < n; ++j)
        for (size_t k = 0; k < 4; ++k) {
            double value = momenta[event + stride * (n*k + entry->permutation[j])];
            if (!isfinite(value)) return UMAMI_ERROR;
            instance->momenta[4*j+k] = value;
        }
    double previous_real = 0.0, previous_imaginary = 0.0;
    bool changed = false;
    if (alpha_s != NULL) {
        if (description->alpha_s_name == NULL) return UMAMI_ERROR_UNSUPPORTED_INPUT;
        if (!isfinite(*alpha_s) || *alpha_s < 0.0) return UMAMI_ERROR;
        status = native_error(rusticol_runtime_get_model_parameter(
            runtime, description->alpha_s_name, &previous_real, &previous_imaginary));
        if (status != UMAMI_SUCCESS) return status;
        changed = previous_real != *alpha_s || previous_imaginary != 0.0;
        if (changed) {
            status = native_error(rusticol_runtime_set_model_parameter(
                runtime, description->alpha_s_name, *alpha_s, 0.0));
            if (status != UMAMI_SUCCESS) return status;
        }
    }
    int native_status;
    if (values != NULL) {
        size_t helicity_count = 0, color_count = 0;
        native_status = rusticol_runtime_evaluate_resolved_f64(
            runtime, instance->momenta, 4*n, 1, NULL, 0,
            entry->color_id == NULL ? NULL : &entry->color_id,
            entry->color_id == NULL ? 0 : 1,
            values, description->helicity_count, &helicity_count, &color_count);
        if (native_status == RUSTICOL_STATUS_OK &&
            (helicity_count != description->helicity_count || color_count != 1)) {
            fprintf(stderr, "umami: unexpected resolved shape of process %s\n", description->key);
            native_status = RUSTICOL_STATUS_RUNTIME_ERROR;
        }
        *result = 0.0;
        for (size_t h = 0; native_status == RUSTICOL_STATUS_OK && h < helicity_count; ++h)
            *result += values[h];
    } else {
        native_status = rusticol_runtime_evaluate_selected_f64(
            runtime, instance->momenta, 4*n, 1,
            NULL, 0,
            entry->color_id == NULL ? NULL : &entry->color_id,
            entry->color_id == NULL ? 0 : 1,
            helicity, helicity == NULL ? 0 : 1, NULL, 0, result, 1);
    }
    status = native_error(native_status);
    if (changed) {
        UmamiStatus restored = native_error(rusticol_runtime_set_model_parameter(
            runtime, description->alpha_s_name, previous_real, previous_imaginary));
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
        (stride && umami_provider.particle_count > SIZE_MAX / stride / 4)) return UMAMI_ERROR;
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
    if (in[UMAMI_IN_MOMENTA] == NULL) return UMAMI_ERROR_MISSING_INPUT;
    Instance *instance = handle;
    for (size_t event = offset; event < offset+count; ++event) {
        size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ?
            ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
        int flavour = in[UMAMI_IN_FLAVOR_INDEX] ?
            ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
        if (channel >= umami_provider.channel_count || flavour < 0 ||
            (size_t)flavour >= umami_provider.channels[channel].flavour_count) return UMAMI_ERROR;
        UmamiFlavour const *selection = &umami_provider.channels[channel].flavours[flavour];
        if (selection->contribution_count == 0) return UMAMI_ERROR;
        double random = in[UMAMI_IN_RANDOM_HELICITY] ?
            ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event] : 0.5;
        if (!isfinite(random) || random < 0.0 || random >= 1.0) return UMAMI_ERROR;
        double const *alpha_s =
            in[UMAMI_IN_ALPHA_S] ? &((double const *)in[UMAMI_IN_ALPHA_S])[event] : NULL;
        UmamiContribution const *first = &selection->contributions[0];
        int label = 0;
        double total = 0.0;
        if (in[UMAMI_IN_HELICITY_INDEX]) {
            /* a fixed exported helicity, absent ones do not contribute */
            label = ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event];
            if (label < 0 || (size_t)label >= umami_provider.exported_helicity_count)
                return UMAMI_ERROR;
            for (size_t j = 0; j < selection->contribution_count; ++j) {
                UmamiContribution const *entry = &selection->contributions[j];
                if (entry->helicity_runtime[label] < 0) continue;
                uint32_t helicity = (uint32_t)entry->helicity_runtime[label];
                double value = 0.0;
                UmamiStatus status = evaluate_contribution(instance, entry,
                    in[UMAMI_IN_MOMENTA], event, stride, alpha_s, &helicity, NULL, &value);
                if (status != UMAMI_SUCCESS) return status;
                total += entry->factor * value;
            }
        } else if (out[UMAMI_OUT_HELICITY_INDEX] && selection->contribution_count == 1) {
            /* sample the helicity in proportion to its contribution */
            double *values = instance->helicity_values;
            UmamiStatus status = evaluate_contribution(instance, first,
                in[UMAMI_IN_MOMENTA], event, stride, alpha_s, NULL, values, &total);
            if (status != UMAMI_SUCCESS) return status;
            size_t count = umami_provider.processes[first->process_index].helicity_count;
            double target = random * total, cumulative = 0.0;
            size_t chosen = 0;
            for (size_t h = 0; h < count; ++h) {
                if (values[h] <= 0.0) continue;
                chosen = h;
                cumulative += values[h];
                if (cumulative > target) break;
            }
            label = (int)first->helicity_export[chosen];
            total *= first->factor;
        } else {
            if (out[UMAMI_OUT_HELICITY_INDEX]) return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
            for (size_t j = 0; j < selection->contribution_count; ++j) {
                double value = 0.0;
                UmamiStatus status = evaluate_contribution(instance, &selection->contributions[j],
                    in[UMAMI_IN_MOMENTA], event, stride, alpha_s, NULL, NULL, &value);
                if (status != UMAMI_SUCCESS) return status;
                total += selection->contributions[j].factor * value;
            }
        }
        if (!isfinite(total)) return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT]) ((double *)out[UMAMI_OUT_MATRIX_ELEMENT])[event] = total;
        if (out[UMAMI_OUT_HELICITY_INDEX]) ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = label;
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] =
            (int)first->color_index;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
    if (instance->runtimes != NULL)
        for (size_t i = 0; i < umami_provider.process_count; ++i)
            if (instance->runtimes[i] != NULL) (void)rusticol_runtime_free(instance->runtimes[i]);
    for (size_t i = 0; i < instance->override_count; ++i) free(instance->overrides[i].name);
    free(instance->overrides);
    free(instance->runtimes); free(instance->momenta); free(instance->helicity_values);
    free(instance->artifact); free(instance);
    return UMAMI_SUCCESS;
}
