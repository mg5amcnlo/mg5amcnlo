#include "madspace/phasespace/flow.hpp"

#include "madspace/constants.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <format>
#include <numeric>

using namespace madspace;

namespace {

std::tuple<Value, Value> build_block(
    FunctionBuilder& fb,
    const MLP& subnet,
    Value input,
    Value condition,
    me_int_t bin_count,
    bool inverse
) {
    auto subnet_out = subnet.build_function(fb, {condition});
    auto [widths_unnorm, heights_unnorm, derivatives] =
        fb.rqs_reshape(subnet_out.at(0), bin_count);
    auto widths = fb.softmax(widths_unnorm);
    auto heights = fb.softmax(heights_unnorm);
    auto rqs_condition = fb.rqs_find_bin(
        input, inverse ? heights : widths, inverse ? widths : heights, derivatives
    );
    auto [out, det] = inverse
        ? fb.rqs_inverse(input, rqs_condition)
        : fb.rqs_forward(input, rqs_condition);
    return {out, fb.reduce_product(det)};
}

void vegas_init(
    ContextPtr context,
    const std::string& grid_name,
    const std::string& bias_name,
    const std::vector<me_int_t>& indices,
    bool inverse_spline
) {
    // TODO: check shapes
    auto bias_global = context->global(bias_name);
    auto grid_tensor = context->global(grid_name).cpu();
    bool is_cpu = false; // context->device() == cpu_device();
    Tensor bias_tensor =
        is_cpu ? bias_global : Tensor(DataType::dt_float, bias_global.shape());
    auto bias_view = bias_tensor.view<double, 2>()[0];
    auto grid_view = grid_tensor.view<double, 3>()[0];
    std::size_t n_dims = grid_tensor.size(1);
    std::size_t n_bins_vegas = grid_tensor.size(2) - 1;
    std::size_t n_bins_flow = (bias_tensor.size(1) / indices.size() - 1) / 3;

    if (n_bins_flow > n_bins_vegas) {
        throw std::runtime_error("The flow must have less bins than VEGAS");
    }

    std::vector<double> w, h, d;
    std::size_t bias_index = 0;

    for (auto index : indices) {
        // Initialize width, heights and derivatives from VEGAS grid
        w.clear();
        h.clear();
        d.clear();
        auto dim_grid = grid_view[index];
        for (std::size_t i = 0; i < n_bins_vegas; ++i) {
            w.push_back(dim_grid[i + 1] - dim_grid[i]);
            h.push_back(1. / n_bins_vegas);
        }
        d.push_back(w.at(0) * n_bins_vegas);
        for (std::size_t i = 0; i < n_bins_vegas - 1; ++i) {
            d.push_back((w.at(i) + w.at(i + 1)) / 2. * n_bins_vegas);
        }
        d.push_back(w.at(n_bins_vegas - 1) * n_bins_vegas);

        // Run bin reduction algorithm
        double size_lim = 2. / n_bins_flow;
        while (w.size() > n_bins_flow) {
            std::size_t min_diff_index = 0;
            double min_diff = std::numeric_limits<int>::max();
            bool min_is_large = true;
            for (std::size_t i = 0; i < w.size() - 1; ++i) {
                double wl = w.at(i), wh = w.at(i + 1), hl = h.at(i), hh = h.at(i + 1);
                double diff = std::abs(wl / hl - wh / hh);
                bool is_large = wl + wh > size_lim || hl + hh > size_lim;
                if (is_large < min_is_large ||
                    (is_large == min_is_large && diff < min_diff)) {
                    min_diff = diff;
                    min_is_large = is_large;
                    min_diff_index = i;
                }
            }
            w.at(min_diff_index) += w.at(min_diff_index + 1);
            h.at(min_diff_index) += h.at(min_diff_index + 1);
            w.erase(w.begin() + min_diff_index + 1);
            h.erase(h.begin() + min_diff_index + 1);
            d.erase(d.begin() + min_diff_index + 1);
        }

        // Invert softmax and softplus functions applied to subnet outputs
        double w_sum = 0.;
        for (double& wi : w) {
            wi = std::log(
                std::max(wi - MIN_BIN_SIZE, MIN_BIN_SIZE * 1e-5) /
                (1. - n_bins_flow * MIN_BIN_SIZE)
            );
            w_sum += wi;
        }
        double h_sum = 0.;
        for (double& hi : h) {
            hi = std::log(
                std::max(hi - MIN_BIN_SIZE, MIN_BIN_SIZE * 1e-5) /
                (1. - n_bins_flow * MIN_BIN_SIZE)
            );
            h_sum += hi;
        }
        std::size_t stride = indices.size();
        for (std::size_t i = 0; i < n_bins_flow; ++i) {
            if (inverse_spline) {
                bias_view[bias_index + stride * i] = w.at(i) - w_sum / n_bins_flow;
                bias_view[bias_index + stride * (n_bins_flow + i)] =
                    h.at(i) - h_sum / n_bins_flow;
            } else {
                bias_view[bias_index + stride * i] = h.at(i) - h_sum / n_bins_flow;
                bias_view[bias_index + stride * (n_bins_flow + i)] =
                    w.at(i) - w_sum / n_bins_flow;
            }
        }
        for (std::size_t i = 0; i < n_bins_flow + 1; ++i) {
            double di = inverse_spline ? 1. / d.at(i) : d.at(i);
            bias_view[bias_index + stride * (2 * n_bins_flow + i)] = std::log(
                std::max(
                    std::exp((MIN_DERIVATIVE + LOG_TWO) * di - MIN_DERIVATIVE) - 1.,
                    MIN_DERIVATIVE * 1e-5
                )
            );
        }
        ++bias_index;
    }

    if (!is_cpu) {
        bias_global.copy_from(bias_tensor);
    }
}

} // namespace

Flow::Flow(
    std::size_t input_dim,
    std::size_t condition_dim,
    const std::string& prefix,
    std::size_t bin_count,
    std::size_t subnet_hidden_dim,
    std::size_t subnet_layers,
    MLP::Activation subnet_activation,
    bool invert_spline
) :
    Mapping(
        "Flow",
        {{"latent", batch_float_array(input_dim)}},
        {{"data", batch_float_array(input_dim)}},
        condition_dim == 0 ? NamedVector<Type>{}
                           : NamedVector<Type>{{"c", batch_float_array(condition_dim)}}
    ),
    _input_dim(input_dim),
    _condition_dim(condition_dim),
    _bin_count(bin_count),
    _invert_spline(invert_spline) {
    if (input_dim == 0) {
        throw std::invalid_argument("Flow input dimension must be at least 2");
    }
    std::size_t block_count = 0;
    for (std::size_t dim = input_dim - 1; dim > 0; dim /= 2) {
        ++block_count;
    }
    std::vector<std::bitset<32>> masks;
    for (std::size_t i = 0; i < input_dim; ++i) {
        masks.push_back(i);
    }

    for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
        std::vector<me_int_t> indices1, indices2;
        std::size_t dim_index = 0;
        for (auto& mask : masks) {
            if (mask.test(block_index)) {
                indices1.push_back(dim_index);
            } else {
                indices2.push_back(dim_index);
            }
            ++dim_index;
        }
        _coupling_blocks.push_back(
            {MLP(indices2.size() + condition_dim,
                 indices1.size() * (3 * bin_count + 1),
                 subnet_hidden_dim,
                 subnet_layers,
                 subnet_activation,
                 prefixed_name(prefix, std::format("subnet{}a", block_index + 1))),
             MLP(indices1.size() + condition_dim,
                 indices2.size() * (3 * bin_count + 1),
                 subnet_hidden_dim,
                 subnet_layers,
                 subnet_activation,
                 prefixed_name(prefix, std::format("subnet{}b", block_index + 1))),
             indices1,
             indices2}
        );
    }
}

void Flow::initialize_globals(ContextPtr context) const {
    for (auto& block : _coupling_blocks) {
        block.subnet1.initialize_globals(context);
        block.subnet2.initialize_globals(context);
    }
}

void Flow::initialize_from_vegas(
    ContextPtr context, const std::string& grid_name
) const {
    initialize_globals(context);
    auto& last_block = _coupling_blocks.at(_coupling_blocks.size() - 1);
    vegas_init(
        context,
        grid_name,
        last_block.subnet1.last_layer_bias_name(),
        last_block.indices1,
        _invert_spline
    );
    vegas_init(
        context,
        grid_name,
        last_block.subnet2.last_layer_bias_name(),
        last_block.indices2,
        _invert_spline
    );
}

Mapping::Result Flow::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return build_transform(fb, inputs.values(), conditions.values(), false);
}

Mapping::Result Flow::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return build_transform(fb, inputs.values(), conditions.values(), true);
}

Mapping::Result Flow::build_transform(
    FunctionBuilder& fb,
    const ValueVec& inputs,
    const ValueVec& conditions,
    bool inverse
) const {
    Value x = inputs.at(0);
    Value cond;
    bool has_cond = _condition_dim != 0;
    if (has_cond) {
        cond = conditions.at(0);
    }
    ValueVec dets;
    std::vector<me_int_t> dim_positions(_input_dim);
    std::iota(dim_positions.begin(), dim_positions.end(), 0);

    auto loop_body = [&](const CouplingBlock& block) {
        std::vector<me_int_t> half1_indices, half2_indices;
        for (auto index : block.indices1) {
            half1_indices.push_back(dim_positions[index]);
        }
        for (auto index : block.indices2) {
            half2_indices.push_back(dim_positions[index]);
        }
        auto half1 = fb.select(x, half1_indices);
        auto half2 = fb.select(x, half2_indices);
        Value det1, det2;
        bool spline_inv = _invert_spline ^ inverse;
        if (inverse) {
            auto cond1 = has_cond ? fb.cat({half2, cond}) : half2;
            std::tie(half1, det1) =
                build_block(fb, block.subnet1, half1, cond1, _bin_count, spline_inv);
            auto cond2 = has_cond ? fb.cat({half1, cond}) : half1;
            std::tie(half2, det2) =
                build_block(fb, block.subnet2, half2, cond2, _bin_count, spline_inv);
        } else {
            auto cond2 = has_cond ? fb.cat({half1, cond}) : half1;
            std::tie(half2, det2) =
                build_block(fb, block.subnet2, half2, cond2, _bin_count, spline_inv);
            auto cond1 = has_cond ? fb.cat({half2, cond}) : half2;
            std::tie(half1, det1) =
                build_block(fb, block.subnet1, half1, cond1, _bin_count, spline_inv);
        }
        dets.push_back(det1);
        dets.push_back(det2);
        x = fb.cat({half1, half2});

        std::size_t pos = 0;
        for (std::size_t index : block.indices1) {
            dim_positions[index] = pos;
            ++pos;
        }
        for (std::size_t index : block.indices2) {
            dim_positions[index] = pos;
            ++pos;
        }
    };
    if (inverse) {
        std::for_each(_coupling_blocks.rbegin(), _coupling_blocks.rend(), loop_body);
    } else {
        std::for_each(_coupling_blocks.begin(), _coupling_blocks.end(), loop_body);
    }

    return {
        {{inverse ? "latent" : "data", fb.select(x, dim_positions)}}, fb.product(dets)
    };
}
