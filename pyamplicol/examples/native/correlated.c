/* SPDX-License-Identifier: 0BSD */
/* Generate the d d~ > d d~ g example in the correlator guide first. */
#include <rusticol.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int status) {
    if (status != RUSTICOL_STATUS_OK) {
        char error[2048];
        size_t required = 0;
        rusticol_last_error_message(error, sizeof error, &required);
        fprintf(stderr, "Rusticol status %d: %s\n", status,
                required <= sizeof error ? error : "error message too long");
        exit(1);
    }
}

static void same(double actual, double expected) {
    if (fabs(actual - expected) > 1.e-12 * fmax(fabs(expected), 1.e-12)) {
        fprintf(stderr, "mismatch: %.17g versus %.17g\n", actual, expected);
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: correlated_c ARTIFACT PROCESS\n");
        return 2;
    }
    RusticolRuntimeHandle *runtime = NULL;
    check(rusticol_runtime_load(argv[1], argv[2], NULL, &runtime));
    size_t count = 0, required = 0;
    check(rusticol_runtime_color_correlation_count(runtime, &count));
    int found_born = 0, found_interference = 0;
    for (size_t index = 0; index < count; ++index) {
        check(rusticol_runtime_color_correlation_id(runtime, index, NULL, 0, &required));
        char *id = malloc(required);
        if (!id) return 1;
        check(rusticol_runtime_color_correlation_id(runtime, index, id, required, &required));
        found_born |= strcmp(id, "born") == 0;
        found_interference |= strcmp(id, "cascade-interference") == 0;
        free(id);
    }
    if (!found_born || !found_interference) return 1;
    check(rusticol_runtime_color_correlation_catalogue_json(runtime, NULL, 0, &required));
    if (required < 2) return 1;

    const double momenta[] = {
        400,0,0,400, 400,0,0,-400,
        300,300,0,0, 250,-150,200,0, 250,-150,-200,0,
        400,0,0,400, 400,0,0,-400,
        300,0,300,0, 250,200,-150,0, 250,-200,-150,0,
    };
    const double spin[] = {0,0, 0,0, 0,0, 1,0};
    const RusticolSpinCorrelationVector vector = {5, 1, spin, 8};
    const RusticolCorrelatedRequest requests[] = {
        {"born", NULL, 0, 0},
        {"cascade-interference", NULL, 0, 0},
        {"cascade-interference", &vector, 1, 0},
    };
    double values[12], ordinary_before[2], ordinary_after[2];
    check(rusticol_runtime_evaluate_f64(runtime, momenta, 40, 2, ordinary_before, 2));
    check(rusticol_runtime_evaluate_correlated_many_f64(
        runtime, momenta, 40, 2, requests, 3, NULL, 0, values, 12));
    for (size_t request = 0; request < 3; ++request) {
        for (size_t point = 0; point < 2; ++point) {
            size_t index = 2 * (2 * request + point);
            printf("VALUE %zu %zu %.17g %.17g\n", request, point,
                   values[index], values[index + 1]);
        }
    }
    check(rusticol_runtime_set_spin_correlation_vectors_f64(runtime, &vector, 1));
    RusticolCorrelatedRequest inherited = {"cascade-interference", NULL, 0, 1};
    double selected[4];
    check(rusticol_runtime_evaluate_correlated_many_f64(
        runtime, momenta, 40, 2, &inherited, 1, NULL, 0, selected, 4));
    for (size_t index = 0; index < 4; ++index) same(selected[index], values[8 + index]);
    check(rusticol_runtime_evaluate_f64(runtime, momenta, 40, 2, ordinary_after, 2));
    for (size_t point = 0; point < 2; ++point) same(ordinary_after[point], ordinary_before[point]);
    check(rusticol_runtime_set_spin_correlation_vectors_f64(runtime, NULL, 0));
    check(rusticol_runtime_evaluate_correlated_many_f64(
        runtime, momenta, 40, 2, &inherited, 1, NULL, 0, selected, 4));
    for (size_t index = 0; index < 4; ++index) same(selected[index], values[4 + index]);
    /* Per-point complex scaling: |i|^2=1 and |2|^2=4. */
    const double scaled[] = {0,0, 0,0, 0,0, 0,1, 0,0, 0,0, 0,0, 2,0};
    RusticolSpinCorrelationVector per_point = {5, 2, scaled, 16};
    RusticolCorrelatedRequest varied = {"cascade-interference", &per_point, 1, 0};
    check(rusticol_runtime_evaluate_correlated_many_f64(
        runtime, momenta, 40, 2, &varied, 1, NULL, 0, selected, 4));
    same(selected[0], values[8]);
    same(selected[2], 4 * values[10]);
    /* Ward identity: substitute the on-shell momentum on gluon leg 5. */
    double longitudinal[16];
    for (size_t point = 0; point < 2; ++point) {
        for (size_t mu = 0; mu < 4; ++mu) {
            longitudinal[8 * point + 2 * mu] = momenta[20 * point + 16 + mu];
            longitudinal[8 * point + 2 * mu + 1] = 0;
        }
    }
    per_point.components = longitudinal;
    check(rusticol_runtime_evaluate_correlated_many_f64(
        runtime, momenta, 40, 2, &varied, 1, NULL, 0, selected, 4));
    for (size_t point = 0; point < 2; ++point) {
        if (hypot(selected[2 * point], selected[2 * point + 1])
            > 1.e-15 * 800 * 800 * fabs(values[2 * point])) return 1;
    }
    /* Errors do not partially overwrite the caller's output. */
    selected[0] = 123;
    if (rusticol_runtime_evaluate_correlated_many_f64(
        runtime, momenta, 40, 2, requests, 3, NULL, 0, selected, 4)
        != RUSTICOL_STATUS_BUFFER_TOO_SMALL || selected[0] != 123) return 1;
    RusticolCorrelatedRequest missing = {"missing", NULL, 0, 0};
    if (rusticol_runtime_evaluate_correlated_many_f64(
        runtime, momenta, 40, 2, &missing, 1, NULL, 0, selected, 4)
        == RUSTICOL_STATUS_OK || selected[0] != 123) return 1;
    check(rusticol_runtime_free(runtime));
    return 0;
}
