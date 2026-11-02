// SPDX-License-Identifier: 0BSD

#include <rusticol.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: runtime_cpp ARTIFACT PROCESS [PARAMETERS.json]\n";
        return 2;
    }

    try {
        rusticol::Runtime runtime(argv[1], argv[2]);
        if (argc == 4) {
            runtime.set_model_parameters_json(argv[3]);
        }
        runtime.set_model_parameter("aS", 0.117);

        // The external-SM d d~ > Z g g subprocess p_p_to_z_j_j_3,
        // flattened as
        // [point][particle][E,px,py,pz].
        const std::vector<double> momenta{
            500.0, 0.0, 0.0, 500.0,
            500.0, 0.0, 0.0, -500.0,
            462.6501613061637, 14.340107538562991, 155.76435943335707,
            -425.7484539710246,
            369.7738416261408, -17.479290785282917, 2.0064955613504103,
            369.3550355960509,
            167.57599706769557, 3.1391832467199254, -157.77085499470743,
            56.3934183749737,
        };
        constexpr std::size_t point_count = 1;
        const auto totals = runtime.evaluate(momenta, point_count);
        const auto resolved = runtime.evaluate_resolved(momenta, point_count);
        const auto checked_totals = resolved.total();
        const double scale = std::max(1.0, std::abs(totals.at(0)));
        if (std::abs(totals.at(0) - checked_totals.at(0)) > 1.0e-12 * scale) {
            std::cerr << "resolved components do not reproduce the total\n";
            return 1;
        }

        std::cout << std::setprecision(17)
                  << "process=" << runtime.process_key() << "\n"
                  << "color_accuracy=" << runtime.color_accuracy() << "\n"
                  << "total=" << totals.at(0) << "\n";
    } catch (const std::exception &error) {
        std::cerr << "Rusticol error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
