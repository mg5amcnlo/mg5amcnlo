// SPDX-License-Identifier: 0BSD
// Generate d d~ > d d~ g with the E.3 "cascade-interference" insertion first.
#include <rusticol.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: correlated_cpp ARTIFACT PROCESS\n";
        return 2;
    }
    try {
        rusticol::Runtime runtime(argv[1], argv[2]);
        const auto ids = runtime.color_correlation_ids();
        for (const auto &id : {"born", "cascade-interference"}) {
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
                throw std::runtime_error("required colour correlation is not registered");
            }
        }
        if (runtime.color_correlation_catalogue_json().empty()) {
            throw std::runtime_error("missing colour histories");
        }
        const std::vector<double> momenta{
            400,0,0,400, 400,0,0,-400, 300,300,0,0, 250,-150,200,0, 250,-150,-200,0,
            400,0,0,400, 400,0,0,-400, 300,0,300,0, 250,200,-150,0, 250,-200,-150,0,
        };
        const std::vector<rusticol::SpinCorrelationVector> spins{
            {5, {{{0., 0., 0., 1.}}}}}, physical{};
        const std::vector<rusticol::CorrelatedRequest> requests{
            {"born", physical}, {"cascade-interference", physical},
            {"cascade-interference", spins},
        };
        const auto ordinary = runtime.evaluate(momenta, 2);
        const auto result = runtime.evaluate_correlated_many(momenta, 2, requests);
        runtime.set_spin_correlation_vectors(spins);
        const auto inherited = runtime.evaluate_correlated(momenta, 2, "cascade-interference");
        const auto after = runtime.evaluate(momenta, 2);
        runtime.set_spin_correlation_vectors();
        const auto cleared = runtime.evaluate_correlated(momenta, 2, "cascade-interference");
        const auto close = [](std::complex<double> a, std::complex<double> b) {
            return std::abs(a - b) <= 1e-11 * std::max(1e-100, std::max(std::abs(a), std::abs(b)));
        };
        for (std::size_t point = 0; point < 2; ++point) {
            if (!close(inherited[point], result(2, point)) ||
                !close(cleared[point], result(1, point)) ||
                ordinary[point] != after[point]) {
                throw std::runtime_error("setter or ordinary-evaluation isolation check failed");
            }
        }
        std::cout << std::setprecision(17);
        for (std::size_t request = 0; request < requests.size(); ++request) {
            for (std::size_t point = 0; point < 2; ++point) {
                const auto value = result(request, point);
                std::cout << "VALUE " << request << ' ' << point << ' '
                          << value.real() << ' ' << value.imag() << '\n';
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
