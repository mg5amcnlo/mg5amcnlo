/* SPDX-License-Identifier: 0BSD */
#ifndef PYAMPLICOL_UMAMI_REWEIGHT_PROVIDER_H
#define PYAMPLICOL_UMAMI_REWEIGHT_PROVIDER_H
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    const char *key;
    const int *pdgs;
    const char *alpha_s_name;
    size_t helicity_count;
} ReweightProcess;

typedef struct {
    size_t reference_process, target_process;
    /* runtime[j] = input[permutation[j]], including both incoming legs. */
    const size_t *permutation;
    /* Exported helicity index -> runtime index of each process, -1 if absent. */
    const int *reference_helicity, *target_helicity;
} ReweightFlavour;

typedef struct {
    size_t flavour_count;
    const ReweightFlavour *flavours;
} ReweightChannel;

typedef struct {
    const char *id;
    size_t particle_count, exported_helicity_count;
    /* Artifact directories relative to the directory passed to umami_initialize. */
    const char *reference_artifact, *target_artifact;
    const char *reference_artifact_id, *target_artifact_id;
    size_t reference_process_count;
    const ReweightProcess *reference_processes;
    size_t target_process_count;
    const ReweightProcess *target_processes;
    size_t channel_count;
    const ReweightChannel *channels;
    bool supports_alpha_s;
} ReweightProvider;
#endif
