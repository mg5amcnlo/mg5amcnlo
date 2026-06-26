#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

template <typename T>
KERNELSPEC void kernel_collect_channel_weights(
    FIn<T, 1> amp2,
    IIn<T, 1> channel_indices,
    IIn<T, 0> channel_count,
    FOut<T, 1> channel_weights
) {
    FVal<T> norm(0.);
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = 0.;
    }
    for (std::size_t i = 0; i < amp2.size(); ++i) {
        std::size_t chan_index = single_index(channel_indices[i]);
        if (chan_index >= channel_weights.size()) {
            continue;
        }
        auto amp2_val = amp2[i];
        norm = norm + amp2_val;
        channel_weights[chan_index] += amp2_val;
    }
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = channel_weights[i] / norm;
    }
}

template <typename T>
KERNELSPEC void kernel_sde2_channel_weights(
    FIn<T, 1> invariants,
    FIn<T, 2> masses,
    FIn<T, 2> widths,
    IIn<T, 2> indices,
    FOut<T, 1> channel_weights
) {
    // TODO: MG has a special case here if tprid != 0. check what that is and if we need
    // it...
    FVal<T> channel_weights_norm(0.);
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        auto masses_i = masses[i];
        auto widths_i = widths[i];
        auto indices_i = indices[i];
        FVal<T> prop_product(1.);
        for (std::size_t j = 0; j < indices_i.size(); ++j) {
            auto indices_ij = indices_i[j];
            auto mask = indices_ij == -1;
            auto invar = invariants.gather(where(mask, 0, indices_ij));
            auto mass = masses_i[j];
            auto width = widths_i[j];
            auto tmp = invar - mass * mass;
            auto tmp2 = mass * width;
            prop_product = prop_product * where(mask, 1., tmp * tmp + tmp2 * tmp2);
        }
        auto channel_weight = 1. / prop_product;
        channel_weights[i] = channel_weight;
        channel_weights_norm = channel_weights_norm + channel_weight;
    }
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = channel_weights[i] / channel_weights_norm;
    }
}

template <typename T>
KERNELSPEC void kernel_subchannel_weights(
    FIn<T, 1> invariants,
    FIn<T, 2> masses,
    FIn<T, 2> widths,
    IIn<T, 2> indices,
    IIn<T, 2> on_shell,
    IIn<T, 1> group_sizes,
    FOut<T, 1> channel_weights
) {
    FVal<T> channel_weights_norm(0.);
    std::size_t group_size = single_index(group_sizes[0]);
    std::size_t group_index = 0;
    for (std::size_t i = 0, i_group = 1; i < channel_weights.size(); ++i, ++i_group) {
        FVal<T> channel_weight(1.);
        for (std::size_t j = 0; j < on_shell.size(1); ++j) {
            auto mass = masses[i][j];
            auto width = widths[i][j];
            auto index = indices[i][j];
            auto is_off_shell = on_shell[i][j] == 0;
            auto mask = (index == -1) | is_off_shell;
            auto invar = invariants.gather(where(mask, 0, index));
            auto mass2 = mass * mass;
            auto tmp = invar - mass2;
            auto tmp2 = mass * width;
            channel_weight = channel_weight *
                where(mask,
                      1.,
                      (mass2 * mass2 + tmp2 * tmp2) / (tmp * tmp + tmp2 * tmp2));
        }
        channel_weights[i] = channel_weight;
        channel_weights_norm = channel_weights_norm + channel_weight;
        if (i_group == group_size) {
            for (std::size_t j = 0; j < group_size; ++j) {
                channel_weights[i - j] = channel_weights[i - j] / channel_weights_norm;
            }
            ++group_index;
            group_size = single_index(group_sizes[group_index]);
            i_group = 0;
            channel_weights_norm = 0;
        }
    }
}

template <typename T>
KERNELSPEC void kernel_apply_subchannel_weights(
    FIn<T, 1> channel_weights_in,
    FIn<T, 1> subchannel_weights,
    IIn<T, 1> channel_indices,
    IIn<T, 1> subchannel_indices,
    FOut<T, 1> channel_weights_out
) {
    for (std::size_t i = 0; i < channel_indices.size(); ++i) {
        auto mask = subchannel_indices[i] == -1;
        auto chan_weight = channel_weights_in.gather(channel_indices[i]);
        auto subchan_weight =
            subchannel_weights.gather(where(mask, 0, subchannel_indices[i]));
        channel_weights_out[i] = chan_weight * where(mask, 1., subchan_weight);
    }
}

template <typename T>
KERNELSPEC void kernel_permute_momenta(
    FIn<T, 2> momenta, IIn<T, 2> permutations, IIn<T, 0> index, FOut<T, 2> output
) {
    auto perm = permutations[single_index(index)];
    for (std::size_t i = 0; i < perm.size(); ++i) {
        auto input_i = momenta[single_index(perm[i])];
        auto output_i = output[i];
        for (std::size_t j = 0; j < 4; ++j) {
            output_i[j] = input_i[j];
        }
    }
}

} // namespace kernels
} // namespace madspace
