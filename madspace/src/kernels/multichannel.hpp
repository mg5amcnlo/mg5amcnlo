#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

template <typename T>
KERNELSPEC void kernel_collect_channel_weights(
    FIn<T, 1> amp2,
    ISIn<T, 1> channel_indices,
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
    // Every amplitude can vanish at once -- a channel whose matrix element is
    // identically zero (an FCNC subprocess whose Wilson coefficients are all zero,
    // say) has amp2 == 0 for every diagram, so norm == 0. Dividing by it would give
    // 0/0 = nan for every channel weight, and that nan reaches the event weight and
    // aborts the whole integration instead of contributing nothing. Fall back to the
    // uniform distribution: the channel weights have to stay a normalised
    // distribution over the channels, and with no amplitude to prefer one, none is
    // preferred. The event weight is zero either way, since the amplitudes are.
    // norm is substituted before the division rather than the result being repaired
    // after it, so that the degenerate 0/0 is never formed in the first place.
    auto degenerate = norm == 0.;
    auto safe_norm = where(degenerate, FVal<T>(1.), norm);
    FVal<T> uniform_weight(1. / static_cast<double>(channel_weights.size()));
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] =
            where(degenerate, uniform_weight, channel_weights[i] / safe_norm);
    }
}

template <typename T>
KERNELSPEC void kernel_density_channel_weights(
    FIn<T, 1> densities, FIn<T, 1> mask, FOut<T, 1> channel_weights
) {
    // A density that is not a finite positive number (an inverse mapping that
    // failed) counts as zero: that channel cannot have produced the event.
    FVal<T> norm(0.), mask_count(0.);
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        auto density = densities[i];
        auto masked = mask[i] > 0.;
        auto valid = (density > 0.) & (density < 1e300) & masked;
        channel_weights[i] = where(valid, density, FVal<T>(0.));
        norm = norm + channel_weights[i];
        mask_count = mask_count + where(masked, FVal<T>(1.), FVal<T>(0.));
    }
    // if all inverses failed numerically, uniform over the masked channels, so
    // that the weights stay a function of the event alone
    auto degenerate = norm == 0.;
    auto safe_norm = where(degenerate, FVal<T>(1.), norm);
    auto uniform = 1. / where(mask_count > 0., mask_count, FVal<T>(1.));
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = where(
            degenerate,
            where(mask[i] > 0., uniform, FVal<T>(0.)),
            channel_weights[i] / safe_norm
        );
    }
}

template <typename T>
KERNELSPEC void kernel_sde2_channel_weights(
    FIn<T, 1> invariants,
    FSIn<T, 2> masses,
    FSIn<T, 2> widths,
    ISIn<T, 2> indices,
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
    FSIn<T, 2> masses,
    FSIn<T, 2> widths,
    ISIn<T, 2> indices,
    ISIn<T, 2> on_shell,
    ISIn<T, 1> group_sizes,
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
    ISIn<T, 1> channel_indices,
    ISIn<T, 1> subchannel_indices,
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
KERNELSPEC void kernel_compress_channel_weights(
    IIn<T, 0> channel_index,
    FIn<T, 1> channel_weights,
    IIn<T, 0> keep_count,
    FOut<T, 1> chan_weight_values,
    IOut<T, 1> chan_weight_indices
) {
    IVal<T> keep_index = channel_index;
    std::size_t n_keep = chan_weight_values.size();
    for (std::size_t i = 0; i < n_keep - 1; i++) {
        chan_weight_values[i] = 0.0;
        chan_weight_indices[i] = -1;
    }
    chan_weight_values[n_keep - 1] = channel_weights.gather(keep_index);
    chan_weight_indices[n_keep - 1] = keep_index;
    if (n_keep == 1) {
        return;
    }
    for (std::size_t i = 0; i < channel_weights.size(); i++) {
        FVal<T> value = where(i == keep_index, -1.0, channel_weights[i]);
        BVal<T> low_mask = value > chan_weight_values[0];
        chan_weight_values[0] = where(low_mask, value, chan_weight_values[0]);
        chan_weight_indices[0] = where(low_mask, i, chan_weight_indices[0]);

        for (std::size_t j = 0; j < n_keep - 2; j++) {
            FVal<T> val1 = chan_weight_values[j], val2 = chan_weight_values[j + 1];
            IVal<T> idx1 = chan_weight_indices[j], idx2 = chan_weight_indices[j + 1];
            BVal<T> mask = val1 > val2;
            chan_weight_values[j] = where(mask, val2, val1);
            chan_weight_indices[j] = where(mask, idx2, idx1);
            chan_weight_values[j + 1] = where(mask, val1, val2);
            chan_weight_indices[j + 1] = where(mask, idx1, idx2);
        }
    }
}

template <typename T>
KERNELSPEC void kernel_restore_channel_weights(
    FIn<T, 1> chan_weight_values,
    IIn<T, 1> chan_weight_indices,
    IIn<T, 0> full_count,
    FOut<T, 1> channel_weights
) {
    FVal<T> epsilon = 1e-12;
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = epsilon;
    }
    FVal<T> sum = channel_weights.size() * epsilon;
    for (std::size_t i = 0; i < chan_weight_values.size(); ++i) {
        FVal<T> value = chan_weight_values[i];
        IVal<T> index = chan_weight_indices[i];
        channel_weights.scatter_add(where(index == -1, 0, index), value);
        sum += value;
    }
    for (std::size_t i = 0; i < channel_weights.size(); ++i) {
        channel_weights[i] = channel_weights[i] / sum;
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
