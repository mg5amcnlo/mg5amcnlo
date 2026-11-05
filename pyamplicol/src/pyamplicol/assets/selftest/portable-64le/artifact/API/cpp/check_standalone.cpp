// SPDX-License-Identifier: 0BSD

#include <rusticol.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string process;
    std::string model_parameters;
    std::string kinematics;
    std::vector<std::string> parameter_names;
    std::vector<double> parameter_real;
    std::vector<double> parameter_imaginary;
    int precision = 16;
    bool json = false;
};

std::string json_string(const std::string &value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(character) << std::dec << std::setfill(' ');
            } else {
                output << character;
            }
        }
    }
    output << '"';
    return output.str();
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto require = [&](int count) {
            if (index + count >= argc) {
                throw std::invalid_argument("missing value after " + argument);
            }
        };
        if (argument == "--process") {
            require(1);
            options.process = argv[++index];
        } else if (argument == "--model-parameters") {
            require(1);
            options.model_parameters = argv[++index];
        } else if (argument == "--kinematics") {
            require(1);
            options.kinematics = argv[++index];
        } else if (argument == "--set-parameter") {
            require(3);
            options.parameter_names.emplace_back(argv[++index]);
            options.parameter_real.push_back(std::stod(argv[++index]));
            options.parameter_imaginary.push_back(std::stod(argv[++index]));
        } else if (argument == "--precision") {
            require(1);
            options.precision = std::stoi(argv[++index]);
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "usage: check_standalone [--process ID|EXPRESSION] "
                         "[--model-parameters PATH] "
                         "[--kinematics PATH] "
                         "[--set-parameter NAME REAL IMAG] "
                         "[--precision 16] [--json]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.precision != 16) {
        throw std::invalid_argument(
            "the C++ Rusticol API supports only double precision (--precision 16)");
    }
    return options;
}

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const auto separator = line.find('\t', start);
        fields.push_back(line.substr(start, separator - start));
        if (separator == std::string::npos) break;
        start = separator + 1;
    }
    return fields;
}

std::vector<double> load_validation_point(
    const fs::path &path, const std::string &process_id, std::size_t external_count) {
    std::ifstream input(path);
    if (!input) return {};
    std::string line;
    std::getline(input, line);
    if (line != "RUSTICOL_VALIDATION_POINTS_V1") {
        throw std::runtime_error("unsupported validation_points.dat format");
    }
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto fields = split_tabs(line);
        if (fields.size() < 2 || fields[0] != process_id) continue;
        const auto row_count = static_cast<std::size_t>(std::stoull(fields[1]));
        if (row_count != external_count || fields.size() != 2 + 4 * external_count) {
            throw std::runtime_error(
                "validation point has an incompatible external-particle count");
        }
        std::vector<double> values;
        values.reserve(4 * external_count);
        for (std::size_t index = 2; index < fields.size(); ++index) {
            values.push_back(std::stod(fields[index]));
        }
        return values;
    }
    return {};
}

std::vector<double> reorder_validation_point(
    const std::vector<double> &representative,
    const std::vector<std::size_t> &permutation) {
    if (representative.empty()) return {};
    if (representative.size() != 4 * permutation.size()) {
        throw std::runtime_error("validation point does not match external permutation");
    }
    std::vector<double> public_point(representative.size());
    std::vector<bool> seen(permutation.size(), false);
    for (std::size_t representative_index = 0;
         representative_index < permutation.size(); ++representative_index) {
        const auto public_index = permutation[representative_index];
        if (public_index >= permutation.size() || seen[public_index]) {
            throw std::runtime_error("runtime returned an invalid external permutation");
        }
        seen[public_index] = true;
        std::copy_n(
            representative.begin() + 4 * representative_index,
            4,
            public_point.begin() + 4 * public_index);
    }
    return public_point;
}

template <typename Integer>
void write_integer_array(std::ostream &output, const std::vector<Integer> &values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_common_json(
    std::ostream &output,
    const rusticol::Runtime &runtime,
    const std::vector<rusticol::ExternalParticle> &particles,
    const std::vector<rusticol::HelicityConfiguration> &helicities,
    const std::vector<rusticol::ColorComponent> &colors) {
    output << "\"process\":" << json_string(runtime.process())
           << ",\"process_key\":" << json_string(runtime.process_key())
           << ",\"color_accuracy\":" << json_string(runtime.color_accuracy());
    output << ",\"external_particles\":[";
    for (std::size_t index = 0; index < particles.size(); ++index) {
        if (index) output << ',';
        output << "{\"index\":" << particles[index].index
               << ",\"pdg\":" << particles[index].pdg << '}';
    }
    output << "],\"helicities\":[";
    for (std::size_t index = 0; index < helicities.size(); ++index) {
        if (index) output << ',';
        output << "{\"id\":" << json_string(helicities[index].id)
               << ",\"helicities\":";
        write_integer_array(output, helicities[index].helicities);
        output << '}';
    }
    output << "],\"colors\":[";
    for (std::size_t index = 0; index < colors.size(); ++index) {
        if (index) output << ',';
        output << "{\"id\":" << json_string(colors[index].id)
               << ",\"kind\":" << json_string(colors[index].kind)
               << ",\"word\":";
        write_integer_array(output, colors[index].word);
        output << '}';
    }
    output << ']';
}

int main(int argc, char **argv) {
    try {
        const auto options = parse_options(argc, argv);
        const fs::path executable = fs::weakly_canonical(fs::absolute(argv[0]));
        fs::path root = fs::current_path();
        if (!fs::is_regular_file(root / "artifact.json")) {
            root = executable.parent_path().parent_path().parent_path();
        }
        rusticol::Runtime runtime(root.string(), options.process, options.model_parameters);
        if (!options.parameter_names.empty()) {
            runtime.set_model_parameters(
                options.parameter_names,
                options.parameter_real,
                options.parameter_imaginary);
        }
        const auto particles = runtime.external_particles();
        const auto helicities = runtime.helicities();
        const auto colors = runtime.colors();
        const auto momenta = options.kinematics.empty()
            ? reorder_validation_point(
                  load_validation_point(
                      root / "API" / "validation_points.dat",
                      runtime.representative_process_key(),
                      particles.size()),
                  runtime.external_permutation())
            : runtime.load_kinematics_json(options.kinematics);
        if (momenta.empty()) {
            if (options.json) {
                std::cout << "{\"language\":\"cpp\",\"available\":false,";
                write_common_json(std::cout, runtime, particles, helicities, colors);
                std::cout << ",\"diagnostic\":"
                             "\"no bundled validation point is available\"}\n";
            } else {
                std::cout << "process: " << runtime.process() << "\n"
                          << "no bundled validation point is available; "
                             "metadata load succeeded\n";
            }
            return 0;
        }
        const auto total = runtime.evaluate(momenta, 1);
        const auto resolved = runtime.evaluate_resolved(momenta, 1);
        const auto explicit_total = resolved.total();

        if (options.json) {
            std::cout << std::setprecision(17)
                      << "{\"language\":\"cpp\",\"available\":true,"
                         "\"precision\":16,";
            write_common_json(std::cout, runtime, particles, helicities, colors);
            std::cout << ",\"shape\":[1," << resolved.helicities.size() << ','
                      << resolved.colors.size() << "],\"values\":[";
            for (std::size_t index = 0; index < resolved.values.size(); ++index) {
                if (index) std::cout << ',';
                std::cout << resolved.values[index];
            }
            std::cout << "],\"resolved_sum\":[" << explicit_total[0]
                      << "],\"compatibility_total\":[" << total[0] << "]}\n";
        } else {
            std::cout << std::setprecision(17)
                      << "process: " << runtime.process() << " ["
                      << runtime.process_key() << "]\n"
                      << "resolved shape: (1, " << resolved.helicities.size() << ", "
                      << resolved.colors.size() << ")\n";
            for (std::size_t helicity = 0;
                 helicity < resolved.helicities.size();
                 ++helicity) {
                for (std::size_t color = 0; color < resolved.colors.size(); ++color) {
                    std::cout << "  " << resolved.helicities[helicity].id << "  "
                              << resolved.colors[color].id << "  "
                              << resolved(0, helicity, color) << '\n';
                }
            }
            std::cout << "explicit resolved sum: " << explicit_total[0] << '\n'
                      << "compatibility total:   " << total[0] << '\n';
        }
        const double scale = std::max(std::abs(total[0]), 1.0);
        if (std::abs(explicit_total[0] - total[0]) > 1.0e-12 * scale) {
            throw std::runtime_error(
                "resolved components do not reproduce the compatibility total");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "check_standalone: " << error.what() << '\n';
        return 1;
    }
}
