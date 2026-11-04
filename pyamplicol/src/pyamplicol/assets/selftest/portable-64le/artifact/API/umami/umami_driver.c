/* SPDX-License-Identifier: 0BSD */
/* A normal linked executable: no dlopen, Python, or direct Rusticol linkage. */
#include "umami.h"
#include "umami_provider.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef UMAMI_PROVIDER_HEADER
#define UMAMI_PROVIDER_HEADER "provider_p0.h"
#endif
#include UMAMI_PROVIDER_HEADER

static void check(UmamiStatus status, char const *operation) {
    if (status != UMAMI_SUCCESS) {
        fprintf(stderr, "umami_driver: %s failed (status %d)\n", operation, status);
        exit(EXIT_FAILURE);
    }
}

static unsigned int index_argument(char const *text) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || text == end || *end || text[0] == '-' || value > INT_MAX) {
        fprintf(stderr, "invalid channel/flavour index: %s\n", text);
        exit(EXIT_FAILURE);
    }
    return (unsigned int)value;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s ABSOLUTE_ARTIFACT [CHANNEL [FLAVOUR]]\n", argv[0]);
        return EXIT_FAILURE;
    }
    unsigned int channel = argc > 2 ? index_argument(argv[2]) : 0;
    int flavour = argc > 3 ? (int)index_argument(argv[3]) : 0;
    if (channel >= umami_provider.channel_count ||
        (size_t)flavour >= umami_provider.channels[channel].flavour_count) {
        fputs("channel/flavour out of range\n", stderr); return EXIT_FAILURE;
    }
    UmamiContribution const *entry = &umami_provider.channels[channel].flavours[flavour].contributions[0];
    char const *process = umami_provider.processes[entry->process_index].key;
    int n = 0;
    check(umami_get_meta(UMAMI_META_PARTICLE_COUNT, &n), "particle count");
    double *point = calloc((size_t)n * 4, sizeof(double));
    double *momenta = calloc((size_t)n * 4, sizeof(double));
    size_t path_length = strlen(argv[1]) + sizeof("/API/validation_points.dat");
    char *path = malloc(path_length);
    if (point == NULL || momenta == NULL || path == NULL) return EXIT_FAILURE;
    snprintf(path, path_length, "%s/API/validation_points.dat", argv[1]);
    FILE *file = fopen(path, "r");
    if (file == NULL) { perror(path); return EXIT_FAILURE; }
    char name[256];
    if (fscanf(file, "%255s", name) != 1 || strcmp(name, "RUSTICOL_VALIDATION_POINTS_V1")) {
        fputs("invalid validation-point file\n", stderr); return EXIT_FAILURE;
    }
    bool found = false;
    size_t particles;
    while (fscanf(file, "%255s %zu", name, &particles) == 2) {
        bool selected = strcmp(name, process) == 0 && particles == (size_t)n;
        for (size_t j = 0; j < particles * 4; ++j) {
            double value;
            if (fscanf(file, "%lf", &value) != 1) return EXIT_FAILURE;
            if (selected) point[j] = value;
        }
        if (selected) { found = true; break; }
    }
    fclose(file); free(path);
    if (!found) { fprintf(stderr, "no saved point for %s\n", process); return EXIT_FAILURE; }
    for (size_t j = 0; j < (size_t)n; ++j)
        for (size_t k = 0; k < 4; ++k)
            momenta[(size_t)n*k + entry->permutation[j]] = point[4*j+k];
    UmamiHandle handle = NULL;
    check(umami_initialize(&handle, argv[1]), "initialize");
    double value = NAN, random = 0.5;
    int helicity = -1;
    UmamiInputKey input_keys[] = {UMAMI_IN_MOMENTA, UMAMI_IN_CHANNEL_INDEX,
                                  UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_RANDOM_HELICITY};
    void const *inputs[] = {momenta, &channel, &flavour, &random};
    UmamiOutputKey output_keys[] = {UMAMI_OUT_MATRIX_ELEMENT, UMAMI_OUT_HELICITY_INDEX};
    void *outputs[] = {&value, &helicity};
    check(umami_matrix_element(handle, 1, 1, 0, 4, input_keys, inputs, 2, output_keys, outputs), "evaluate");
    check(umami_free(handle), "free");
    free(point); free(momenta);
    if (!isfinite(value)) return EXIT_FAILURE;
    printf("{\"provider\":\"%s\",\"channel\":%u,\"flavour\":%d,\"matrix_element\":%.17g,\"helicity_label\":%d}\n",
           umami_provider.id, channel, flavour, value, helicity);
    return EXIT_SUCCESS;
}
