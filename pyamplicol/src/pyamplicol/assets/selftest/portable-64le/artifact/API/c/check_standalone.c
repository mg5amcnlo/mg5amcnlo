/* SPDX-License-Identifier: 0BSD */

#include <rusticol.h>

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_OVERRIDES 128

typedef struct {
    const char *process;
    const char *model_parameters;
    const char *kinematics;
    const char *parameter_names[MAX_OVERRIDES];
    double parameter_real[MAX_OVERRIDES];
    double parameter_imaginary[MAX_OVERRIDES];
    size_t parameter_count;
    int precision;
    bool json;
} Options;

typedef struct {
    char *id;
    int32_t *values;
    size_t value_count;
} HelicityMetadata;

typedef struct {
    char *id;
    char *kind;
    size_t *word;
    size_t word_count;
} ColorMetadata;

typedef int (*RuntimeStringGetter)(
    const RusticolRuntimeHandle *, char *, size_t, size_t *
);
typedef int (*IndexedStringGetter)(
    const RusticolRuntimeHandle *, size_t, char *, size_t, size_t *
);

static void fail(const char *message) {
    fprintf(stderr, "check_standalone: %s\n", message);
    exit(EXIT_FAILURE);
}

static void fail_errno(const char *operation) {
    fprintf(stderr, "check_standalone: %s: %s\n", operation, strerror(errno));
    exit(EXIT_FAILURE);
}

static void *allocate(size_t count, size_t size) {
    void *output;
    if (count != 0 && size > SIZE_MAX / count) {
        fail("allocation size overflow");
    }
    output = calloc(count == 0 ? 1 : count, size == 0 ? 1 : size);
    if (output == NULL) {
        fail("out of memory");
    }
    return output;
}

static void *resize(void *pointer, size_t size) {
    void *output = realloc(pointer, size == 0 ? 1 : size);
    if (output == NULL) {
        free(pointer);
        fail("out of memory");
    }
    return output;
}

static void check_rusticol(int status, const char *operation) {
    size_t required = 0;
    char *message = NULL;
    if (status == RUSTICOL_STATUS_OK) {
        return;
    }
    if (rusticol_last_error_message(NULL, 0, &required) == RUSTICOL_STATUS_OK &&
        required > 0) {
        message = allocate(required, sizeof(*message));
        if (rusticol_last_error_message(message, required, &required) !=
            RUSTICOL_STATUS_OK) {
            free(message);
            message = NULL;
        }
    }
    fprintf(
        stderr,
        "check_standalone: %s failed (status %d)%s%s\n",
        operation,
        status,
        message == NULL ? "" : ": ",
        message == NULL ? "" : message
    );
    free(message);
    exit(EXIT_FAILURE);
}

static double parse_double(const char *text, const char *description) {
    char *end = NULL;
    double value;
    errno = 0;
    value = strtod(text, &end);
    if (text == end || end == NULL || *end != '\0' || errno == ERANGE ||
        !isfinite(value)) {
        fprintf(stderr, "check_standalone: invalid %s: %s\n", description, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static size_t parse_size(const char *text, const char *description) {
    char *end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (text == end || end == NULL || *end != '\0' || errno == ERANGE ||
        value > SIZE_MAX) {
        fprintf(stderr, "check_standalone: invalid %s: %s\n", description, text);
        exit(EXIT_FAILURE);
    }
    return (size_t)value;
}

static Options parse_options(int argc, char **argv) {
    Options options = {0};
    int index;
    options.precision = 16;
    for (index = 1; index < argc; ++index) {
        const char *argument = argv[index];
        if (strcmp(argument, "--") == 0) {
            continue;
        }
        if (strcmp(argument, "--process") == 0) {
            if (++index >= argc) {
                fail("missing value after --process");
            }
            options.process = argv[index];
        } else if (strcmp(argument, "--model-parameters") == 0) {
            if (++index >= argc) {
                fail("missing value after --model-parameters");
            }
            options.model_parameters = argv[index];
        } else if (strcmp(argument, "--kinematics") == 0) {
            if (++index >= argc) {
                fail("missing value after --kinematics");
            }
            options.kinematics = argv[index];
        } else if (strcmp(argument, "--set-parameter") == 0) {
            size_t slot = options.parameter_count;
            if (index + 3 >= argc) {
                fail("--set-parameter requires NAME REAL IMAG");
            }
            if (slot >= MAX_OVERRIDES) {
                fail("too many --set-parameter options");
            }
            options.parameter_names[slot] = argv[++index];
            options.parameter_real[slot] =
                parse_double(argv[++index], "real model-parameter component");
            options.parameter_imaginary[slot] =
                parse_double(argv[++index], "imaginary model-parameter component");
            options.parameter_count += 1;
        } else if (strcmp(argument, "--precision") == 0) {
            size_t precision;
            if (++index >= argc) {
                fail("missing value after --precision");
            }
            precision = parse_size(argv[index], "--precision value");
            if (precision > INT32_MAX) {
                fail("invalid --precision value");
            }
            options.precision = (int)precision;
        } else if (strcmp(argument, "--json") == 0) {
            options.json = true;
        } else if (strcmp(argument, "--help") == 0 || strcmp(argument, "-h") == 0) {
            puts(
                "usage: check_standalone [--process ID|EXPRESSION] "
                "[--model-parameters PATH] "
                "[--kinematics PATH] "
                "[--set-parameter NAME REAL IMAG] "
                "[--precision 16] [--json]"
            );
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "check_standalone: unknown option: %s\n", argument);
            exit(EXIT_FAILURE);
        }
    }
    if (options.precision != 16) {
        fail("the C Rusticol API supports only double precision (--precision 16)");
    }
    return options;
}

static char *runtime_string(
    const RusticolRuntimeHandle *runtime,
    RuntimeStringGetter getter,
    const char *operation
) {
    size_t required = 0;
    char *output;
    check_rusticol(getter(runtime, NULL, 0, &required), operation);
    if (required == 0) {
        fail("Rusticol returned an invalid string size");
    }
    output = allocate(required, sizeof(*output));
    check_rusticol(getter(runtime, output, required, &required), operation);
    return output;
}

static char *indexed_string(
    const RusticolRuntimeHandle *runtime,
    size_t index,
    IndexedStringGetter getter,
    const char *operation
) {
    size_t required = 0;
    char *output;
    check_rusticol(getter(runtime, index, NULL, 0, &required), operation);
    if (required == 0) {
        fail("Rusticol returned an invalid indexed string size");
    }
    output = allocate(required, sizeof(*output));
    check_rusticol(getter(runtime, index, output, required, &required), operation);
    return output;
}

static char *read_line(FILE *stream) {
    size_t capacity = 256;
    size_t length = 0;
    char *line = allocate(capacity, sizeof(*line));
    int character;
    while ((character = fgetc(stream)) != EOF && character != '\n') {
        if (length + 1 >= capacity) {
            if (capacity > SIZE_MAX / 2) {
                free(line);
                fail("validation-point row is too large");
            }
            capacity *= 2;
            line = resize(line, capacity);
        }
        line[length++] = (char)character;
    }
    if (ferror(stream)) {
        free(line);
        fail_errno("read API/validation_points.dat");
    }
    if (character == EOF && length == 0) {
        free(line);
        return NULL;
    }
    if (length > 0 && line[length - 1] == '\r') {
        length -= 1;
    }
    line[length] = '\0';
    return line;
}

static char *take_field(char **cursor) {
    char *field;
    char *separator;
    if (*cursor == NULL) {
        return NULL;
    }
    field = *cursor;
    separator = strchr(field, '\t');
    if (separator == NULL) {
        *cursor = NULL;
    } else {
        *separator = '\0';
        *cursor = separator + 1;
    }
    return field;
}

static double *load_validation_point(
    const char *path,
    const char *process_key,
    size_t external_count
) {
    FILE *input = fopen(path, "r");
    char *line;
    if (input == NULL) {
        if (errno == ENOENT) {
            return NULL;
        }
        fail_errno("open API/validation_points.dat");
    }
    line = read_line(input);
    if (line == NULL || strcmp(line, "RUSTICOL_VALIDATION_POINTS_V1") != 0) {
        free(line);
        fclose(input);
        fail("unsupported validation_points.dat format");
    }
    free(line);

    while ((line = read_line(input)) != NULL) {
        char *cursor = line;
        char *row_process;
        char *row_count_text;
        size_t row_count;
        size_t momentum_count;
        size_t index;
        double *momenta;
        if (line[0] == '\0' || line[0] == '#') {
            free(line);
            continue;
        }
        row_process = take_field(&cursor);
        row_count_text = take_field(&cursor);
        if (row_process == NULL || row_count_text == NULL) {
            free(line);
            fclose(input);
            fail("invalid validation point row");
        }
        if (strcmp(row_process, process_key) != 0) {
            free(line);
            continue;
        }
        row_count = parse_size(row_count_text, "validation external-particle count");
        if (row_count != external_count || external_count > SIZE_MAX / 4) {
            free(line);
            fclose(input);
            fail("validation point has an incompatible external-particle count");
        }
        momentum_count = 4 * external_count;
        momenta = allocate(momentum_count, sizeof(*momenta));
        for (index = 0; index < momentum_count; ++index) {
            char *component = take_field(&cursor);
            if (component == NULL) {
                free(momenta);
                free(line);
                fclose(input);
                fail("validation point has too few momentum components");
            }
            momenta[index] =
                parse_double(component, "validation-point momentum component");
        }
        if (cursor != NULL) {
            free(momenta);
            free(line);
            fclose(input);
            fail("validation point has too many momentum components");
        }
        free(line);
        if (fclose(input) != 0) {
            free(momenta);
            fail_errno("close API/validation_points.dat");
        }
        return momenta;
    }
    if (fclose(input) != 0) {
        fail_errno("close API/validation_points.dat");
    }
    return NULL;
}

static size_t *load_external_permutation(
    const RusticolRuntimeHandle *runtime,
    size_t expected_count
) {
    size_t required = 0;
    size_t *permutation;
    check_rusticol(
        rusticol_runtime_external_permutation(runtime, NULL, 0, &required),
        "query external permutation size"
    );
    if (required != expected_count) {
        fail("external permutation has an incompatible size");
    }
    permutation = allocate(required, sizeof(*permutation));
    check_rusticol(
        rusticol_runtime_external_permutation(
            runtime, permutation, required, &required
        ),
        "query external permutation"
    );
    return permutation;
}

static double *reorder_validation_point(
    double *representative,
    const size_t *permutation,
    size_t external_count
) {
    double *public_point = allocate(4 * external_count, sizeof(*public_point));
    bool *seen = allocate(external_count, sizeof(*seen));
    size_t representative_index;
    for (representative_index = 0; representative_index < external_count;
         ++representative_index) {
        size_t public_index = permutation[representative_index];
        if (public_index >= external_count || seen[public_index]) {
            free(seen);
            free(public_point);
            free(representative);
            fail("runtime returned an invalid external permutation");
        }
        seen[public_index] = true;
        memcpy(
            &public_point[4 * public_index],
            &representative[4 * representative_index],
            4 * sizeof(*public_point)
        );
    }
    free(seen);
    free(representative);
    return public_point;
}

static double *load_kinematics_json(
    const RusticolRuntimeHandle *runtime,
    const char *path,
    size_t expected_count
) {
    size_t required = 0;
    double *momenta;
    check_rusticol(
        rusticol_runtime_load_kinematics_json(
            runtime, path, NULL, 0, &required
        ),
        "query kinematics JSON size"
    );
    if (required != 4 * expected_count) {
        fail("kinematics JSON has an incompatible external-particle count");
    }
    momenta = allocate(required, sizeof(*momenta));
    check_rusticol(
        rusticol_runtime_load_kinematics_json(
            runtime, path, momenta, required, &required
        ),
        "load kinematics JSON"
    );
    return momenta;
}

static int32_t *load_external_particles(
    const RusticolRuntimeHandle *runtime,
    size_t *count
) {
    int32_t *particles;
    size_t index;
    check_rusticol(
        rusticol_runtime_external_count(runtime, count),
        "query external-particle count"
    );
    particles = allocate(*count, sizeof(*particles));
    for (index = 0; index < *count; ++index) {
        check_rusticol(
            rusticol_runtime_external_pdg(runtime, index, &particles[index]),
            "query external-particle PDG"
        );
    }
    return particles;
}

static HelicityMetadata *load_helicities(
    const RusticolRuntimeHandle *runtime,
    size_t *count
) {
    HelicityMetadata *items;
    size_t index;
    check_rusticol(
        rusticol_runtime_helicity_count(runtime, count),
        "query helicity count"
    );
    items = allocate(*count, sizeof(*items));
    for (index = 0; index < *count; ++index) {
        items[index].id = indexed_string(
            runtime,
            index,
            rusticol_runtime_helicity_id,
            "query helicity ID"
        );
        check_rusticol(
            rusticol_runtime_helicity_vector(
                runtime, index, NULL, 0, &items[index].value_count
            ),
            "query helicity-vector size"
        );
        items[index].values =
            allocate(items[index].value_count, sizeof(*items[index].values));
        if (items[index].value_count > 0) {
            check_rusticol(
                rusticol_runtime_helicity_vector(
                    runtime,
                    index,
                    items[index].values,
                    items[index].value_count,
                    &items[index].value_count
                ),
                "query helicity vector"
            );
        }
    }
    return items;
}

static ColorMetadata *load_colors(
    const RusticolRuntimeHandle *runtime,
    size_t *count
) {
    ColorMetadata *items;
    size_t index;
    check_rusticol(rusticol_runtime_color_count(runtime, count), "query color count");
    items = allocate(*count, sizeof(*items));
    for (index = 0; index < *count; ++index) {
        items[index].id = indexed_string(
            runtime, index, rusticol_runtime_color_id, "query color ID"
        );
        items[index].kind = indexed_string(
            runtime, index, rusticol_runtime_color_kind, "query color kind"
        );
        check_rusticol(
            rusticol_runtime_color_word(
                runtime, index, NULL, 0, &items[index].word_count
            ),
            "query color-word size"
        );
        items[index].word =
            allocate(items[index].word_count, sizeof(*items[index].word));
        if (items[index].word_count > 0) {
            check_rusticol(
                rusticol_runtime_color_word(
                    runtime,
                    index,
                    items[index].word,
                    items[index].word_count,
                    &items[index].word_count
                ),
                "query color word"
            );
        }
    }
    return items;
}

static void write_json_string(const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    putchar('"');
    while (*cursor != '\0') {
        unsigned char character = *cursor++;
        switch (character) {
        case '"': fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\b': fputs("\\b", stdout); break;
        case '\f': fputs("\\f", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if (character < 0x20) {
                printf("\\u%04x", (unsigned int)character);
            } else {
                putchar((int)character);
            }
        }
    }
    putchar('"');
}

static void write_common_json(
    const char *process,
    const char *process_key,
    const char *color_accuracy,
    const int32_t *particles,
    size_t particle_count,
    const HelicityMetadata *helicities,
    size_t helicity_count,
    const ColorMetadata *colors,
    size_t color_count
) {
    size_t index;
    fputs("\"process\":", stdout);
    write_json_string(process);
    fputs(",\"process_key\":", stdout);
    write_json_string(process_key);
    fputs(",\"color_accuracy\":", stdout);
    write_json_string(color_accuracy);
    fputs(",\"external_particles\":[", stdout);
    for (index = 0; index < particle_count; ++index) {
        if (index != 0) {
            putchar(',');
        }
        printf("{\"index\":%zu,\"pdg\":%" PRId32 "}", index, particles[index]);
    }
    fputs("],\"helicities\":[", stdout);
    for (index = 0; index < helicity_count; ++index) {
        size_t component;
        if (index != 0) {
            putchar(',');
        }
        fputs("{\"id\":", stdout);
        write_json_string(helicities[index].id);
        fputs(",\"helicities\":[", stdout);
        for (component = 0; component < helicities[index].value_count; ++component) {
            if (component != 0) {
                putchar(',');
            }
            printf("%" PRId32, helicities[index].values[component]);
        }
        fputs("]}", stdout);
    }
    fputs("],\"colors\":[", stdout);
    for (index = 0; index < color_count; ++index) {
        size_t component;
        if (index != 0) {
            putchar(',');
        }
        fputs("{\"id\":", stdout);
        write_json_string(colors[index].id);
        fputs(",\"kind\":", stdout);
        write_json_string(colors[index].kind);
        fputs(",\"word\":[", stdout);
        for (component = 0; component < colors[index].word_count; ++component) {
            if (component != 0) {
                putchar(',');
            }
            printf("%zu", colors[index].word[component]);
        }
        fputs("]}", stdout);
    }
    putchar(']');
}

static void free_helicities(HelicityMetadata *items, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) {
        free(items[index].id);
        free(items[index].values);
    }
    free(items);
}

static void free_colors(ColorMetadata *items, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) {
        free(items[index].id);
        free(items[index].kind);
        free(items[index].word);
    }
    free(items);
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    RusticolRuntimeHandle *runtime = NULL;
    FILE *manifest;
    char *process;
    char *process_key;
    char *representative_process_key;
    char *color_accuracy;
    int32_t *particles;
    HelicityMetadata *helicities;
    ColorMetadata *colors;
    size_t particle_count = 0;
    size_t helicity_count = 0;
    size_t color_count = 0;
    double *momenta;
    size_t *external_permutation = NULL;
    double total = NAN;
    double *resolved = NULL;
    size_t resolved_helicity_count = 0;
    size_t resolved_color_count = 0;
    size_t resolved_count;
    double explicit_total = 0.0;
    size_t index;

    manifest = fopen("artifact.json", "r");
    if (manifest == NULL) {
        fail("run check_standalone from a generated artifact directory");
    }
    if (fclose(manifest) != 0) {
        fail_errno("close artifact.json");
    }

    check_rusticol(
        rusticol_runtime_load(
            ".", options.process, options.model_parameters, &runtime
        ),
        "load artifact"
    );
    if (options.parameter_count > 0) {
        check_rusticol(
            rusticol_runtime_set_model_parameters(
                runtime,
                options.parameter_names,
                options.parameter_real,
                options.parameter_imaginary,
                options.parameter_count
            ),
            "set model parameters"
        );
    }

    process = runtime_string(runtime, rusticol_runtime_process, "query process");
    process_key =
        runtime_string(runtime, rusticol_runtime_process_key, "query process key");
    representative_process_key = runtime_string(
        runtime,
        rusticol_runtime_representative_process_key,
        "query representative process key"
    );
    color_accuracy = runtime_string(
        runtime, rusticol_runtime_color_accuracy, "query color accuracy"
    );
    particles = load_external_particles(runtime, &particle_count);
    helicities = load_helicities(runtime, &helicity_count);
    colors = load_colors(runtime, &color_count);
    if (options.kinematics != NULL) {
        momenta = load_kinematics_json(runtime, options.kinematics, particle_count);
    } else {
        external_permutation = load_external_permutation(runtime, particle_count);
        momenta = load_validation_point(
            "API/validation_points.dat", representative_process_key, particle_count
        );
        if (momenta != NULL) {
            momenta = reorder_validation_point(
                momenta, external_permutation, particle_count
            );
        }
    }

    if (momenta == NULL) {
        if (options.json) {
            fputs("{\"language\":\"c\",\"available\":false,", stdout);
            write_common_json(
                process,
                process_key,
                color_accuracy,
                particles,
                particle_count,
                helicities,
                helicity_count,
                colors,
                color_count
            );
            fputs(
                ",\"diagnostic\":\"no bundled validation point is available\"}\n",
                stdout
            );
        } else {
            printf("process: %s\n", process);
            puts("no bundled validation point is available; metadata load succeeded");
        }
    } else {
        check_rusticol(
            rusticol_runtime_evaluate_f64(
                runtime, momenta, 4 * particle_count, 1, &total, 1
            ),
            "evaluate compatibility total"
        );
        check_rusticol(
            rusticol_runtime_resolved_shape(
                runtime,
                NULL,
                0,
                NULL,
                0,
                &resolved_helicity_count,
                &resolved_color_count
            ),
            "query resolved shape"
        );
        if (resolved_helicity_count != helicity_count ||
            resolved_color_count != color_count) {
            fail("resolved shape does not match runtime metadata");
        }
        if (resolved_helicity_count != 0 &&
            resolved_color_count > SIZE_MAX / resolved_helicity_count) {
            fail("resolved output size overflow");
        }
        resolved_count = resolved_helicity_count * resolved_color_count;
        resolved = allocate(resolved_count, sizeof(*resolved));
        check_rusticol(
            rusticol_runtime_evaluate_resolved_f64(
                runtime,
                momenta,
                4 * particle_count,
                1,
                NULL,
                0,
                NULL,
                0,
                resolved,
                resolved_count,
                &resolved_helicity_count,
                &resolved_color_count
            ),
            "evaluate resolved components"
        );
        if (!isfinite(total)) {
            fail("matrix-element total is not finite");
        }
        for (index = 0; index < resolved_count; ++index) {
            if (!isfinite(resolved[index])) {
                fail("resolved matrix-element output is not finite");
            }
            explicit_total += resolved[index];
        }

        if (options.json) {
            fputs(
                "{\"language\":\"c\",\"available\":true,\"precision\":16,",
                stdout
            );
            write_common_json(
                process,
                process_key,
                color_accuracy,
                particles,
                particle_count,
                helicities,
                helicity_count,
                colors,
                color_count
            );
            printf(
                ",\"shape\":[1,%zu,%zu],\"values\":[",
                resolved_helicity_count,
                resolved_color_count
            );
            for (index = 0; index < resolved_count; ++index) {
                if (index != 0) {
                    putchar(',');
                }
                printf("%.17g", resolved[index]);
            }
            printf(
                "],\"resolved_sum\":[%.17g],\"compatibility_total\":[%.17g]}\n",
                explicit_total,
                total
            );
        } else {
            size_t helicity_index;
            size_t color_index;
            printf("process: %s [%s]\n", process, process_key);
            printf(
                "resolved shape: (1, %zu, %zu)\n",
                resolved_helicity_count,
                resolved_color_count
            );
            for (helicity_index = 0;
                 helicity_index < resolved_helicity_count;
                 ++helicity_index) {
                for (color_index = 0; color_index < resolved_color_count; ++color_index) {
                    size_t offset =
                        helicity_index * resolved_color_count + color_index;
                    printf(
                        "  %s  %s  %.17g\n",
                        helicities[helicity_index].id,
                        colors[color_index].id,
                        resolved[offset]
                    );
                }
            }
            printf("explicit resolved sum: %.17g\n", explicit_total);
            printf("compatibility total:   %.17g\n", total);
        }
        if (!isfinite(explicit_total) ||
            fabs(explicit_total - total) > 1.0e-12 * fmax(fabs(total), 1.0)) {
            fail("resolved components do not reproduce the compatibility total");
        }
    }

    check_rusticol(rusticol_runtime_free(runtime), "free runtime");
    free(momenta);
    free(resolved);
    free(process);
    free(process_key);
    free(representative_process_key);
    free(external_permutation);
    free(color_accuracy);
    free(particles);
    free_helicities(helicities, helicity_count);
    free_colors(colors, color_count);
    return EXIT_SUCCESS;
}
