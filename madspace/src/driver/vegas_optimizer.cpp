#include "madspace/driver/vegas_optimizer.hpp"

using namespace madspace;

void VegasGridOptimizer::add_data(Tensor values, Tensor counts) {
    auto values_cpu = values.cpu();
    auto counts_cpu = counts.cpu();
    // TODO: check all the shapes here
    auto values_view = values_cpu.view<double, 3>()[0];
    auto counts_view = counts_cpu.view<me_int_t, 3>()[0];
    std::size_t n_dims = values_view.size(0);
    std::size_t n_bins = values_view.size(1);

    while (_data.size() < n_dims) {
        _data.push_back(
            {std::vector<std::size_t>(n_bins), std::vector<double>(n_bins)}
        );
    }

    for (std::size_t i_dim = 0; i_dim < n_dims; ++i_dim) {
        auto& [bin_counts, bin_values] = _data[i_dim];
        for (std::size_t i_bin = 0; i_bin < n_bins; ++i_bin) {
            bin_counts[i_bin] += counts_view[i_dim][i_bin];
            bin_values[i_bin] += values_view[i_dim][i_bin];
        }
    }
}

void VegasGridOptimizer::optimize() {
    auto grid_cpu = _contexts.at(0)->global(_grid_name).cpu();
    std::size_t n_dims = grid_cpu.size(1);
    std::size_t n_bins = grid_cpu.size(2) - 1;
    auto new_grid = grid_cpu.copy();
    auto grid_view = grid_cpu.view<double, 3>()[0];
    auto new_grid_view = new_grid.view<double, 3>()[0];

    for (std::size_t i_dim = 0; i_dim < n_dims; ++i_dim) {
        auto& [bin_counts, bin_values] = _data.at(i_dim);
        if (bin_counts.size() != n_bins || bin_values.size() != n_bins) {
            throw std::runtime_error("no data to run optimization");
        }

        // compute averages
        std::size_t count_tot = 0;
        for (std::size_t i_bin = 0; i_bin < n_bins; ++i_bin) {
            count_tot += bin_counts[i_bin];
            if (bin_counts[i_bin] > 0) {
                bin_values[i_bin] /= bin_counts[i_bin];
            }
        }

        // apply smoothing
        double prev_value = bin_values[0];
        double current_value = bin_values[1];
        double sum = 0.;
        // bin_values[0] = (7. * prev_value + current_value) / 8.;
        bin_values[0] = (prev_value + current_value) / 2.;
        for (std::size_t i_bin = 1; i_bin < n_bins - 1; ++i_bin) {
            double next_value = bin_values[i_bin + 1];
            // double new_value = (prev_value + 6. * current_value + next_value) / 8.;
            double new_value = (prev_value + current_value + next_value) / 3.;
            bin_values[i_bin] = new_value;
            sum += new_value;
            prev_value = current_value;
            current_value = next_value;
        }
        // bin_values[n_bins - 1] = (prev_value + 7. * current_value) / 8.;
        bin_values[n_bins - 1] = (prev_value + current_value) / 2.;

        // normalize and apply damping
        constexpr double tiny = 1e-10;
        double damped_avg = 0.;
        if (sum == 0) {
            std::fill(
                bin_values.begin(),
                bin_values.end(),
                std::pow(-(1 - tiny) / std::log(tiny), _damping)
            );
            damped_avg = tiny;
        } else {
            for (std::size_t i_bin = 0; i_bin < n_bins; ++i_bin) {
                double val_norm = std::max(bin_values[i_bin] / sum, tiny);
                double new_val = val_norm <= 0.99999999
                    ? std::pow(-(1 - val_norm) / std::log(val_norm), _damping)
                    : val_norm;
                bin_values[i_bin] = new_val;
                damped_avg += new_val;
            }
            damped_avg /= n_bins;
        }

        // update grid
        double accumulator = 0.;
        me_int_t j_bin = -1;
        for (std::size_t i_bin = 1; i_bin < n_bins; ++i_bin) {
            while (accumulator < damped_avg) {
                ++j_bin;
                if (j_bin == n_bins) {
                    break;
                }
                accumulator += bin_values[j_bin];
            }
            if (j_bin == n_bins) {
                break;
            }
            double grid_j = grid_view[i_dim][j_bin];
            double grid_j_next = grid_view[i_dim][j_bin + 1];
            double bin_width = grid_j_next - grid_j;
            accumulator -= damped_avg;
            new_grid_view[i_dim][i_bin] =
                grid_j_next - accumulator / bin_values[j_bin] * bin_width;
        }

        std::fill(bin_counts.begin(), bin_counts.end(), 0);
        std::fill(bin_values.begin(), bin_values.end(), 0.);
    }

    for (auto& context : _contexts) {
        context->global(_grid_name).copy_from(new_grid);
    }
}

std::size_t VegasGridOptimizer::input_dim() const {
    return _contexts.at(0)->global(_grid_name).size(1);
}
