/* SPDX-License-Identifier: 0BSD */
#ifndef PYAMPLICOL_UMAMI_PROVIDER_H
#define PYAMPLICOL_UMAMI_PROVIDER_H
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    const char *name;
    double real, imaginary;
    bool mutable, is_complex;
} UmamiParameter;

typedef struct {
    const char *key;
    const int *pdgs;
    const char *alpha_s_name;
    const char *color_accuracy;
    size_t helicity_count;
    const char *const *helicity_ids;
    size_t parameter_count;
    const UmamiParameter *parameters;
} UmamiProcess;

typedef struct {
    size_t process_index;
    const char *color_id;
    size_t color_index;
    /* runtime[j] = input[permutation[j]], including both incoming legs. */
    const size_t *permutation;
    double factor;
    /* Exported helicity index (input leg order) -> runtime index, -1 if absent. */
    const int *helicity_runtime;
    /* Runtime helicity index -> exported helicity index. */
    const size_t *helicity_export;
} UmamiContribution;

typedef struct {
    size_t contribution_count;
    const UmamiContribution *contributions;
} UmamiFlavour;

typedef struct {
    size_t flavour_count;
    const UmamiFlavour *flavours;
} UmamiChannel;

typedef struct {
    const char *id;
    size_t particle_count, incoming_count, helicity_count, color_count;
    /* Number of exported helicities, the metadata "helicities" table. */
    size_t exported_helicity_count;
    const double *masses;
    size_t process_count;
    const UmamiProcess *processes;
    size_t channel_count;
    const UmamiChannel *channels;
    bool supports_alpha_s;
} UmamiProvider;
#endif
