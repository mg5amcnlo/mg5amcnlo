#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

template <typename T>
KERNELSPEC void kernel_vegas_forward(
    FIn<T, 0> input, FIn<T, 1> grid, FOut<T, 0> output, FOut<T, 0> det
) {
    FVal<T> bin_count_f(grid.size() - 1);
    FVal<T> bin_index_f(input * bin_count_f);
    IVal<T> bin_index(bin_index_f);
    FVal<T> bin_index_rounded(bin_index);
    auto left_edge = grid.gather(bin_index);
    auto right_edge = grid.gather(bin_index + 1);
    auto bin_size = right_edge - left_edge;
    output = left_edge + (input * bin_count_f - bin_index_rounded) * bin_size;
    det = bin_count_f * bin_size;
}

template <typename T>
KERNELSPEC void kernel_vegas_inverse(
    FIn<T, 0> input, FIn<T, 1> grid, FOut<T, 0> output, FOut<T, 0> det
) {
    std::size_t bin_count = grid.size() - 1;
    FVal<T> bin_count_f(bin_count);
    IVal<T> bin_index(0);
    for (std::size_t i = 0; i < bin_count; ++i) {
        bin_index = where(input < grid[i], i, bin_index);
    }
    auto left_edge = grid.gather(bin_index);
    auto right_edge = grid.gather(bin_index + 1);
    auto bin_size = right_edge - left_edge;
    output = (FVal<T>(bin_index) + (input - left_edge) / bin_size) / bin_count_f;
    det = 1 / (bin_count_f * bin_size);
}

} // namespace kernels
} // namespace madspace
