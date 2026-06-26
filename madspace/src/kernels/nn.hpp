#pragma once

#include "definitions.hpp"
#include "madspace/constants.hpp"

namespace madspace {
namespace kernels {

// Kernels

template <typename T>
KERNELSPEC void kernel_relu(FIn<T, 0> input, FOut<T, 0> output) {
    FVal<T> x(input);
    output = where(x < 0., 0., x);
}

template <typename T>
KERNELSPEC void
backward_kernel_relu(FIn<T, 0> input, FIn<T, 0> output_grad, FOut<T, 0> input_grad) {
    FVal<T> x(input), g(output_grad);
    input_grad = where(x < 0., 0., g);
}

template <typename T>
KERNELSPEC void kernel_leaky_relu(FIn<T, 0> input, FOut<T, 0> output) {
    FVal<T> x(input);
    output = where(x < 0., x * 0.01, x);
}

template <typename T>
KERNELSPEC void backward_kernel_leaky_relu(
    FIn<T, 0> input, FIn<T, 0> output_grad, FOut<T, 0> input_grad
) {
    FVal<T> x(input), g(output_grad);
    input_grad = where(x < 0., g * 0.01, g);
}

template <typename T>
KERNELSPEC void kernel_elu(FIn<T, 0> input, FOut<T, 0> output) {
    FVal<T> x(input);
    output = where(x < 0., expm1(x), x);
}

template <typename T>
KERNELSPEC void
backward_kernel_elu(FIn<T, 0> input, FIn<T, 0> output_grad, FOut<T, 0> input_grad) {
    FVal<T> x(input), g(output_grad);
    input_grad = where(x < 0., g * exp(x), g);
}

template <typename T>
KERNELSPEC void kernel_gelu(FIn<T, 0> input, FOut<T, 0> output) {
    output = 0.5 * input * (1. + erf(SQRT_HALF * input));
}

template <typename T>
KERNELSPEC void
backward_kernel_gelu(FIn<T, 0> input, FIn<T, 0> output_grad, FOut<T, 0> input_grad) {
    FVal<T> x(input), g(output_grad);
    auto cdf = 0.5 * (1. + erf(SQRT_HALF * x));
    auto pdf = 0.5 * SQRT_HALF * TWO_DIV_SQRT_PI * exp(-0.5 * x * x);
    input_grad = g * (cdf + x * pdf);
}

template <typename T>
KERNELSPEC void kernel_sigmoid(FIn<T, 0> input, FOut<T, 0> output) {
    FVal<T> x = input;
    output = 1. / (1. + exp(-x));
}

template <typename T>
KERNELSPEC void backward_kernel_sigmoid(
    FIn<T, 0> output, FIn<T, 0> output_grad, FOut<T, 0> input_grad
) {
    FVal<T> y(output), g(output_grad);
    input_grad = g * y * (1. - y);
}

template <typename T>
KERNELSPEC void kernel_softplus(FIn<T, 0> input, FOut<T, 0> output) {
    FVal<T> x(input);
    output = where(x < 20., log1p(exp(x)), x);
}

template <typename T>
KERNELSPEC void backward_kernel_softplus(
    FIn<T, 0> input, FIn<T, 0> output_grad, FOut<T, 0> input_grad
) {
    FVal<T> x(input), g(output_grad);
    auto z = exp(x);
    input_grad = where(x < 20., g * z / (1. + z), g);
}

template <typename T>
KERNELSPEC FVal<T> fastexp(FVal<T> x) {
    auto xx = x / 16;
    auto y = 0.5 * (xx + sqrt(xx * xx + 4));
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    y = y * y;
    return y;
}

template <typename T>
KERNELSPEC void kernel_rqs_find_bin(
    FIn<T, 0> input,
    FIn<T, 1> in_sizes,
    FIn<T, 1> out_sizes,
    FIn<T, 1> derivatives,
    FOut<T, 1> condition
) {
    auto n_bins = in_sizes.size();
    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));
    auto bin_factor = 1. - MIN_BIN_SIZE * n_bins;

    FVal<T> loop_cumwidth(0.), loop_cumheight(0.);
    FVal<T> width(0.), height(0.), cumwidth(0.), cumheight(0.);
    FVal<T> derivative_unorm(0.), derivative_plus_one_unorm(0.);
    for (std::size_t bin = 0; bin < n_bins; ++bin) {
        auto w = MIN_BIN_SIZE + bin_factor * in_sizes[bin];
        auto h = MIN_BIN_SIZE + bin_factor * out_sizes[bin];
        auto d = derivatives[bin];
        auto dp1 = derivatives[bin + 1];

        auto mask = input01 < loop_cumwidth;
        width = where(mask, width, w);
        height = where(mask, height, h);
        derivative_unorm = where(mask, derivative_unorm, d);
        derivative_plus_one_unorm = where(mask, derivative_plus_one_unorm, dp1);
        cumwidth = where(mask, cumwidth, loop_cumwidth);
        cumheight = where(mask, cumheight, loop_cumheight);
        loop_cumwidth = loop_cumwidth + w;
        loop_cumheight = loop_cumheight + h;
    }
    condition[0] = width;
    condition[1] = height;
    condition[2] = cumwidth;
    condition[3] = cumheight;
    condition[4] = derivative_unorm;
    condition[5] = derivative_plus_one_unorm;
}

template <typename T>
KERNELSPEC void backward_kernel_rqs_find_bin(
    FIn<T, 0> input,
    FIn<T, 1> in_sizes,
    FIn<T, 1> out_sizes,
    FIn<T, 1> derivatives,
    FIn<T, 1> condition_grad,
    FOut<T, 0> input_grad,
    FOut<T, 1> in_sizes_grad,
    FOut<T, 1> out_sizes_grad,
    FOut<T, 1> derivatives_grad
) {
    FVal<T> width_grad = condition_grad[0];
    FVal<T> height_grad = condition_grad[1];
    FVal<T> cumwidth_grad = condition_grad[2];
    FVal<T> cumheight_grad = condition_grad[3];
    FVal<T> derivative_unorm_grad = condition_grad[4];
    FVal<T> derivative_plus_one_unorm_grad = condition_grad[5];
    auto n_bins = in_sizes.size();
    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));
    auto bin_factor = 1. - MIN_BIN_SIZE * n_bins;

    FVal<T> loop_cumwidth(0.), loop_cumheight(0.);
    IVal<T> selected_bin(0);
    BVal<T> mask(false);
    for (std::size_t bin = 0; bin < n_bins; ++bin) {
        auto w = MIN_BIN_SIZE + bin_factor * in_sizes[bin];
        selected_bin = where(mask, selected_bin, bin);
        loop_cumwidth = loop_cumwidth + w;
        mask = input01 < loop_cumwidth;
        in_sizes_grad[bin] += where(mask, 0., bin_factor * cumwidth_grad);
        out_sizes_grad[bin] += where(mask, 0., bin_factor * cumheight_grad);
    }
    in_sizes_grad.scatter_add(selected_bin, bin_factor * width_grad);
    out_sizes_grad.scatter_add(selected_bin, bin_factor * height_grad);
    derivatives_grad.scatter_add(selected_bin, derivative_unorm_grad);
    derivatives_grad.scatter_add(selected_bin + 1, derivative_plus_one_unorm_grad);
}

template <typename T>
KERNELSPEC void kernel_rqs_forward(
    FIn<T, 0> input, FIn<T, 1> condition, FOut<T, 0> output, FOut<T, 0> det
) {
    FVal<T> width(condition[0]), height(condition[1]);
    FVal<T> cumwidth(condition[2]), cumheight(condition[3]);
    FVal<T> derivative_unorm(condition[4]), derivative_plus_one_unorm(condition[5]);
    auto softplus_scale = LOG_TWO + MIN_DERIVATIVE;
    auto derivative =
        (where(derivative_unorm > 20., derivative_unorm, log1p(exp(derivative_unorm))) +
         MIN_DERIVATIVE) /
        softplus_scale;
    auto derivative_plus_one =
        (where(
             derivative_plus_one_unorm > 20.,
             derivative_plus_one_unorm,
             log1p(exp(derivative_plus_one_unorm))
         ) +
         MIN_DERIVATIVE) /
        softplus_scale;

    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));

    auto delta = height / width;
    auto input_diff = input01 - cumwidth;
    auto theta = input_diff / width;
    auto one_minus_theta = 1. - theta;
    auto theta_theta = theta * theta;
    auto theta_one_minus_theta = theta * one_minus_theta;
    auto numerator =
        height * (delta * theta_theta + derivative * theta_one_minus_theta);
    auto two_delta = 2. * delta;
    auto denominator =
        delta + (derivative + derivative_plus_one - two_delta) * theta_one_minus_theta;
    auto out = cumheight + numerator / denominator;
    output = where(clamp, input, out);

    auto derivative_numerator = delta * delta *
        (derivative_plus_one * theta_theta + two_delta * theta_one_minus_theta +
         derivative * one_minus_theta * one_minus_theta);
    det = where(clamp, 1., derivative_numerator / (denominator * denominator));
}

template <typename T>
KERNELSPEC void backward_kernel_rqs_forward(
    FIn<T, 0> input,
    FIn<T, 1> condition,
    FIn<T, 0> output_grad,
    FIn<T, 0> det_grad,
    FOut<T, 0> input_grad,
    FOut<T, 1> condition_grad
) {
    FVal<T> width(condition[0]), height(condition[1]);
    FVal<T> cumwidth(condition[2]), cumheight(condition[3]);
    FVal<T> derivative_unorm(condition[4]), derivative_plus_one_unorm(condition[5]);

    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));

    auto expd = exp(derivative_unorm);
    auto expdpo = exp(derivative_plus_one_unorm);
    auto log1pd = log1p(expd);
    auto log1pdpo = log1p(expdpo);
    auto spd = where(derivative_unorm > 20., derivative_unorm, log1pd);
    auto spdpo =
        where(derivative_plus_one_unorm > 20., derivative_plus_one_unorm, log1pdpo);
    auto spd_md = spd + MIN_DERIVATIVE;
    auto spdpo_md = spdpo + MIN_DERIVATIVE;
    auto softplus_scale = LOG_TWO + MIN_DERIVATIVE;
    auto derivative = spd_md / softplus_scale;
    auto derivative_plus_one = spdpo_md / softplus_scale;

    auto delta = height / width;
    auto input_diff = input01 - cumwidth;
    auto theta = input_diff / width;
    auto one_minus_theta = 1. - theta;
    auto theta_theta = theta * theta;
    auto theta_one_minus_theta = theta * one_minus_theta;
    auto tmp1 = delta * theta_theta;
    auto tmp2 = derivative * theta_one_minus_theta;
    auto tmp3 = tmp1 + tmp2;
    auto numerator = height * tmp3;
    auto two_delta = 2. * delta;
    auto tmp4 = derivative + derivative_plus_one;
    auto tmp5 = tmp4 - two_delta;
    auto tmp6 = tmp5 * theta_one_minus_theta;
    auto denominator = delta + tmp6;
    auto tmp7 = numerator / denominator;
    auto out = cumheight + tmp7;

    auto tmp8 = delta * delta;
    auto tmp9 = derivative_plus_one * theta_theta;
    auto tmp10 = two_delta * theta_one_minus_theta;
    auto tmp11 = derivative * one_minus_theta;
    auto tmp12 = tmp11 * one_minus_theta;
    auto tmp13 = tmp9 + tmp10;
    auto tmp14 = tmp13 + tmp12;
    auto derivative_numerator = tmp8 * tmp14;
    auto tmp15 = denominator * denominator;
    auto tmp16 = derivative_numerator / tmp15;

    auto grad_tmp16 = det_grad;
    auto grad_tmp15 = -grad_tmp16 * derivative_numerator / (tmp15 * tmp15);
    auto grad_derivative_numerator = grad_tmp16 / tmp15;
    auto grad_denominator = 2. * grad_tmp15 * denominator;
    auto grad_tmp8 = tmp14 * grad_derivative_numerator;
    auto grad_tmp14 = tmp8 * grad_derivative_numerator;
    auto grad_tmp13 = grad_tmp14;
    auto grad_tmp12 = grad_tmp14;
    auto grad_tmp9 = grad_tmp13;
    auto grad_tmp10 = grad_tmp13;
    auto grad_tmp11 = one_minus_theta * grad_tmp12;
    auto grad_one_minus_theta = tmp11 * grad_tmp12;
    auto grad_derivative = one_minus_theta * grad_tmp11;
    grad_one_minus_theta += derivative * grad_tmp11;
    auto grad_two_delta = theta_one_minus_theta * grad_tmp10;
    auto grad_theta_one_minus_theta = two_delta * grad_tmp10;
    auto grad_derivative_plus_one = theta_theta * grad_tmp9;
    auto grad_theta_theta = derivative_plus_one * grad_tmp9;
    auto grad_delta = 2. * delta * grad_tmp8;

    auto grad_cumheight = output_grad;
    auto grad_tmp7 = output_grad;
    auto grad_numerator = grad_tmp7 / denominator;
    grad_denominator += -numerator * grad_tmp7 / (denominator * denominator);
    grad_delta += grad_denominator;
    auto grad_tmp6 = grad_denominator;
    auto grad_tmp5 = theta_one_minus_theta * grad_tmp6;
    grad_theta_one_minus_theta += tmp5 * grad_tmp6;
    auto grad_tmp4 = grad_tmp5;
    grad_two_delta += -grad_tmp5;
    grad_derivative += grad_tmp4;
    grad_derivative_plus_one += grad_tmp4;
    grad_delta += 2. * grad_two_delta;
    auto grad_height = tmp3 * grad_numerator;
    auto grad_tmp3 = height * grad_numerator;
    auto grad_tmp1 = grad_tmp3;
    auto grad_tmp2 = grad_tmp3;
    grad_derivative += theta_one_minus_theta * grad_tmp2;
    grad_theta_one_minus_theta += derivative * grad_tmp2;
    grad_delta += theta_theta * grad_tmp1;
    grad_theta_theta += delta * grad_tmp1;
    auto grad_theta = one_minus_theta * grad_theta_one_minus_theta;
    grad_one_minus_theta += theta * grad_theta_one_minus_theta;
    grad_theta += 2. * theta * grad_theta_theta;
    grad_theta += -grad_one_minus_theta;
    auto grad_input_diff = grad_theta / width;
    auto grad_width = -input_diff * grad_theta / (width * width);
    auto grad_input = grad_input_diff;
    auto grad_cumwidth = -grad_input_diff;

    grad_height += grad_delta / width;
    grad_width += -height * grad_delta / (width * width);

    auto grad_spdpo_md = grad_derivative_plus_one / softplus_scale;
    auto grad_spd_md = grad_derivative / softplus_scale;
    auto grad_spdpo = grad_spdpo_md;
    auto grad_spd = grad_spd_md;

    auto grad_log1pdpo = grad_spdpo;
    auto grad_expdpo = grad_log1pdpo / (1. + expdpo);
    auto grad_derivative_plus_one_unorm =
        where(derivative_plus_one_unorm > 20., grad_spdpo, expdpo * grad_expdpo);

    auto grad_log1pd = grad_spd;
    auto grad_expd = grad_log1pd / (1. + expd);
    auto grad_derivative_unorm =
        where(derivative_unorm > 20., grad_spd, expd * grad_expd);

    input_grad = where(clamp, output_grad, grad_input);
    condition_grad[0] = where(clamp, 0., grad_width);
    condition_grad[1] = where(clamp, 0., grad_height);
    condition_grad[2] = where(clamp, 0., grad_cumwidth);
    condition_grad[3] = where(clamp, 0., grad_cumheight);
    condition_grad[4] = where(clamp, 0., grad_derivative_unorm);
    condition_grad[5] = where(clamp, 0., grad_derivative_plus_one_unorm);
}

template <typename T>
KERNELSPEC void kernel_rqs_inverse(
    FIn<T, 0> input, FIn<T, 1> condition, FOut<T, 0> output, FOut<T, 0> det
) {
    FVal<T> height(condition[0]), width(condition[1]);
    FVal<T> cumheight(condition[2]), cumwidth(condition[3]);
    FVal<T> derivative_unorm(condition[4]), derivative_plus_one_unorm(condition[5]);
    auto softplus_scale = LOG_TWO + MIN_DERIVATIVE;
    auto derivative =
        (where(derivative_unorm > 20., derivative_unorm, log1p(exp(derivative_unorm))) +
         MIN_DERIVATIVE) /
        softplus_scale;
    auto derivative_plus_one =
        (where(
             derivative_plus_one_unorm > 20.,
             derivative_plus_one_unorm,
             log1p(exp(derivative_plus_one_unorm))
         ) +
         MIN_DERIVATIVE) /
        softplus_scale;

    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));

    auto delta = height / width;
    auto two_delta = 2. * delta;
    auto input_diff = input01 - cumheight;
    auto d_sum = derivative + derivative_plus_one - two_delta;
    auto tmp = input_diff * d_sum;
    auto a = tmp + height * (delta - derivative);
    auto b = height * derivative - tmp;
    auto c = delta * input_diff;
    auto discriminant = b * b + 4. * a * c;
    auto theta = 2. * c / (b + sqrt(discriminant));
    auto out = cumwidth + theta * width;
    output = where(clamp, input, out);

    auto one_minus_theta = 1. - theta;
    auto theta_one_minus_theta = theta * one_minus_theta;
    auto denominator = delta + d_sum * theta_one_minus_theta;
    auto derivative_numerator = delta * delta *
        (derivative_plus_one * theta * theta + two_delta * theta_one_minus_theta +
         derivative * one_minus_theta * one_minus_theta);
    det = where(clamp, 1., denominator * denominator / derivative_numerator);
}

template <typename T>
KERNELSPEC void backward_kernel_rqs_inverse(
    FIn<T, 0> input,
    FIn<T, 1> condition,
    FIn<T, 0> output_grad,
    FIn<T, 0> det_grad,
    FOut<T, 0> input_grad,
    FOut<T, 1> condition_grad
) {
    FVal<T> height(condition[0]), width(condition[1]);
    FVal<T> cumheight(condition[2]), cumwidth(condition[3]);
    FVal<T> derivative_unorm(condition[4]), derivative_plus_one_unorm(condition[5]);

    auto low_mask = input < 0.;
    auto high_mask = input > 1.;
    auto clamp = low_mask | high_mask;
    auto input01 = where(high_mask, 1., where(low_mask, 0., input));

    auto expd = exp(derivative_unorm);
    auto expdpo = exp(derivative_plus_one_unorm);
    auto log1pd = log1p(expd);
    auto log1pdpo = log1p(expdpo);
    auto spd = where(derivative_unorm > 20., derivative_unorm, log1pd);
    auto spdpo =
        where(derivative_plus_one_unorm > 20., derivative_plus_one_unorm, log1pdpo);
    auto spd_md = spd + MIN_DERIVATIVE;
    auto spdpo_md = spdpo + MIN_DERIVATIVE;
    auto softplus_scale = LOG_TWO + MIN_DERIVATIVE;
    auto derivative = spd_md / softplus_scale;
    auto derivative_plus_one = spdpo_md / softplus_scale;

    auto delta = height / width;

    auto tmp1 = derivative + derivative_plus_one;
    auto two_delta = 2. * delta;
    auto tmp2 = tmp1 - two_delta;
    auto input_diff = input01 - cumheight;
    auto tmp3 = input_diff * tmp2;
    auto tmp4 = delta - derivative;
    auto tmp5 = height * tmp4;
    auto a = tmp3 + tmp5;
    auto tmp6 = height * derivative;
    auto b = tmp6 - tmp3;
    auto c = delta * input_diff;
    auto tmp7 = b * b;
    auto tmp8 = 4. * a;
    auto tmp9 = tmp8 * c;
    auto discriminant = tmp7 + tmp9;
    auto tmp10 = 2. * c;
    auto tmp11 = sqrt(discriminant);
    auto tmp12 = b + tmp11;
    auto theta = tmp10 / tmp12;
    auto tmp13 = theta * width;
    auto out = tmp13 + cumwidth;

    auto one_minus_theta = 1. - theta;
    auto theta_one_minus_theta = theta * one_minus_theta;
    auto tmp14 = tmp2 * theta_one_minus_theta;
    auto denominator = delta + tmp14;
    auto tmp15 = delta * delta;
    auto tmp16 = derivative_plus_one * theta;
    auto tmp17 = tmp16 * theta;
    auto tmp18 = two_delta * theta_one_minus_theta;
    auto tmp19 = tmp17 + tmp18;
    auto tmp20 = derivative * one_minus_theta;
    auto tmp21 = tmp20 * one_minus_theta;
    auto tmp22 = tmp19 + tmp21;
    auto derivative_numerator = tmp15 * tmp22;
    auto tmp23 = denominator * denominator;
    auto tmp24 = tmp23 / derivative_numerator;

    auto grad_tmp24 = det_grad;
    auto grad_tmp23 = grad_tmp24 / derivative_numerator;
    auto grad_derivative_numerator =
        -grad_tmp24 * tmp23 / (derivative_numerator * derivative_numerator);
    auto grad_denominator = 2. * grad_tmp23 * denominator;

    auto grad_tmp15 = tmp22 * grad_derivative_numerator;
    auto grad_tmp22 = tmp15 * grad_derivative_numerator;
    auto grad_tmp19 = grad_tmp22;
    auto grad_tmp21 = grad_tmp22;
    auto grad_tmp20 = one_minus_theta * grad_tmp21;
    auto grad_one_minus_theta = tmp20 * grad_tmp21;
    auto grad_derivative = one_minus_theta * grad_tmp20;
    grad_one_minus_theta += derivative * grad_tmp20;
    auto grad_tmp17 = grad_tmp19;
    auto grad_tmp18 = grad_tmp19;
    auto grad_two_delta = theta_one_minus_theta * grad_tmp18;
    auto grad_theta_one_minus_theta = two_delta * grad_tmp18;
    auto grad_tmp16 = theta * grad_tmp17;
    auto grad_theta = tmp16 * grad_tmp17;
    auto grad_derivative_plus_one = theta * grad_tmp16;
    grad_theta += derivative_plus_one * grad_tmp16;
    auto grad_delta = 2. * delta * grad_tmp15;
    grad_delta += grad_denominator;
    auto grad_tmp14 = grad_denominator;
    auto grad_tmp2 = theta_one_minus_theta * grad_tmp14;
    grad_theta_one_minus_theta += tmp2 * grad_tmp14;
    grad_theta += one_minus_theta * grad_theta_one_minus_theta;
    grad_one_minus_theta += theta * grad_theta_one_minus_theta;
    grad_theta += -grad_one_minus_theta;

    auto grad_tmp13 = output_grad;
    auto grad_cumwidth = output_grad;
    auto grad_width = theta * grad_tmp13;
    grad_theta += width * grad_tmp13;
    auto grad_tmp10 = grad_theta / tmp12;
    auto grad_tmp12 = -tmp10 * grad_theta / (tmp12 * tmp12);
    auto grad_b = grad_tmp12;
    auto grad_tmp11 = grad_tmp12;
    auto grad_discriminant = 0.5 * grad_tmp11 / tmp11;
    auto grad_c = 2. * grad_tmp10;
    auto grad_tmp7 = grad_discriminant;
    auto grad_tmp9 = grad_discriminant;
    grad_c += tmp8 * grad_tmp9;
    auto grad_tmp8 = c * grad_tmp9;
    auto grad_a = 4. * grad_tmp8;
    grad_b += 2. * b * grad_tmp7;
    grad_delta += input_diff * grad_c;
    auto grad_input_diff = delta * grad_c;
    auto grad_tmp6 = grad_b;
    auto grad_tmp3 = -grad_b;
    auto grad_height = derivative * grad_tmp6;
    grad_derivative += height * grad_tmp6;
    grad_tmp3 += grad_a;
    auto grad_tmp5 = grad_a;
    grad_height += tmp4 * grad_tmp5;
    auto grad_tmp4 = height * grad_tmp5;
    grad_delta += grad_tmp4;
    grad_derivative += -grad_tmp4;
    grad_input_diff += tmp2 * grad_tmp3;
    grad_tmp2 += input_diff * grad_tmp3;
    auto grad_input = grad_input_diff;
    auto grad_cumheight = -grad_input_diff;
    auto grad_tmp1 = grad_tmp2;
    grad_two_delta += -grad_tmp2;
    grad_delta += 2. * grad_two_delta;
    grad_derivative += grad_tmp1;
    grad_derivative_plus_one += grad_tmp1;

    grad_height += grad_delta / width;
    grad_width += -height * grad_delta / (width * width);

    auto grad_spdpo_md = grad_derivative_plus_one / softplus_scale;
    auto grad_spd_md = grad_derivative / softplus_scale;
    auto grad_spdpo = grad_spdpo_md;
    auto grad_spd = grad_spd_md;

    auto grad_log1pdpo = grad_spdpo;
    auto grad_expdpo = grad_log1pdpo / (1. + expdpo);
    auto grad_derivative_plus_one_unorm =
        where(derivative_plus_one_unorm > 20., grad_spdpo, expdpo * grad_expdpo);

    auto grad_log1pd = grad_spd;
    auto grad_expd = grad_log1pd / (1. + expd);
    auto grad_derivative_unorm =
        where(derivative_unorm > 20., grad_spd, expd * grad_expd);

    input_grad = where(clamp, output_grad, grad_input);
    condition_grad[0] = where(clamp, 0., grad_height);
    condition_grad[1] = where(clamp, 0., grad_width);
    condition_grad[2] = where(clamp, 0., grad_cumheight);
    condition_grad[3] = where(clamp, 0., grad_cumwidth);
    condition_grad[4] = where(clamp, 0., grad_derivative_unorm);
    condition_grad[5] = where(clamp, 0., grad_derivative_plus_one_unorm);
}

template <typename T>
KERNELSPEC void kernel_softmax(FIn<T, 1> input, FOut<T, 1> output) {
    FVal<T> norm(0.), in_max(0.);
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto in = input[i];
        in_max = max(in, in_max);
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto exp_in = exp(input[i] - in_max);
        norm = norm + exp_in;
        output[i] = exp_in;
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = output[i] / norm;
    }
}

template <typename T>
KERNELSPEC void backward_kernel_softmax(
    FIn<T, 1> output, FIn<T, 1> output_grad, FOut<T, 1> input_grad
) {
    std::size_t dim = output.size();
    FVal<T> grad_sum(0.);
    for (std::size_t i = 0; i < dim; ++i) {
        grad_sum = grad_sum + output[i] * output_grad[i];
    }
    for (std::size_t i = 0; i < dim; ++i) {
        input_grad[i] += output[i] * (output_grad[i] - grad_sum);
    }
}

template <typename T>
KERNELSPEC void
kernel_softmax_prior(FIn<T, 1> input, FIn<T, 1> prior, FOut<T, 1> output) {
    FVal<T> norm(0.);
    // TODO: solve exp->inf issue
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto unnorm_prob = exp(input[i]) * prior[i];
        norm = norm + unnorm_prob;
        output[i] = unnorm_prob;
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = output[i] / norm;
    }
}

template <typename T>
KERNELSPEC void backward_kernel_softmax_prior(
    FIn<T, 1> output,
    FIn<T, 1> output_grad,
    FOut<T, 1> input_grad,
    FOut<T, 1> prior_grad
) {
    // TODO: also gradient for prior?
    backward_kernel_softmax<T>(output, output_grad, input_grad);
}

template <typename T>
KERNELSPEC void
kernel_one_hot(IIn<T, 0> index, IIn<T, 0> option_count, FOut<T, 1> output) {
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = where(i == index, FVal<T>(1.), 0.);
    }
}

template <typename T>
KERNELSPEC void kernel_madnis_abs_weight(FIn<T, 0> f, FIn<T, 0> q, FOut<T, 0> w) {
    w = fabs(f) / q;
}

template <typename T>
KERNELSPEC void backward_kernel_madnis_abs_weight(
    FIn<T, 0> f, FIn<T, 0> q, FIn<T, 0> out_grad, FOut<T, 0> f_grad, FOut<T, 0> q_grad
) {
    auto f_abs = fabs(f);
    auto f_abs_grad = out_grad / q;
    q_grad += -out_grad * f_abs / (q * q);
    f_grad += where(f < 0., -f_abs_grad, f_abs_grad);
}

template <typename T>
KERNELSPEC void kernel_madnis_softclip(
    FIn<T, 0> f, FIn<T, 0> q, FIn<T, 0> norm, FIn<T, 0> threshold, FOut<T, 0> f_out
) {
    auto factor = q * norm * threshold;
    f_out = factor * asinh(f / factor);
}

template <typename T>
KERNELSPEC void backward_kernel_madnis_softclip(
    FIn<T, 0> f,
    FIn<T, 0> q,
    FIn<T, 0> norm,
    FIn<T, 0> threshold,
    FIn<T, 0> f_out_grad,
    FOut<T, 0> f_grad,
    FOut<T, 0> q_grad,
    FOut<T, 0> norm_grad,
    FOut<T, 0> threshold_grad
) {
    auto factor = q * norm * threshold;
    f_grad += f_out_grad / sqrt(f * f / (factor * factor) + 1);
}

template <typename T>
KERNELSPEC void kernel_madnis_variance(
    FIn<T, 0> f, FIn<T, 0> g, FIn<T, 0> q, FIn<T, 0> mean, FOut<T, 0> var
) {
    auto diff = f / g - mean;
    var = g * diff * diff / q;
}

template <typename T>
KERNELSPEC void backward_kernel_madnis_variance(
    FIn<T, 0> f,
    FIn<T, 0> g,
    FIn<T, 0> q,
    FIn<T, 0> mean,
    FIn<T, 0> var_grad,
    FOut<T, 0> f_grad,
    FOut<T, 0> g_grad,
    FOut<T, 0> q_grad,
    FOut<T, 0> mean_grad
) {
    auto diff = f / g - mean;
    auto diff2 = diff * diff;
    auto diff_grad = 2. * var_grad * g / q * diff;
    g_grad += var_grad * diff2 / q - diff_grad * f / (g * g);
    mean_grad += -diff_grad;
    f_grad += diff_grad / g;
    q_grad += -var_grad * g * diff2 / (q * q);
}

template <typename T>
KERNELSPEC void kernel_madnis_single_channel_variance(
    FIn<T, 0> var, FIn<T, 0> abs_mean, FOut<T, 0> loss
) {
    loss = var / (abs_mean * abs_mean);
}

template <typename T>
KERNELSPEC void backward_kernel_madnis_single_channel_variance(
    FIn<T, 0> abs_mean,
    FIn<T, 0> loss_grad,
    FOut<T, 0> vars_grad,
    FOut<T, 0> abs_means_grad
) {
    vars_grad += loss_grad / (abs_mean * abs_mean);
}

template <typename T>
KERNELSPEC void kernel_madnis_multi_channel_variance(
    FIn<T, 1> vars, FIn<T, 1> abs_means, FOut<T, 0> loss
) {
    FVal<T> std_sum(0.), abs_mean_sum(0.);
    for (std::size_t i = 0; i < vars.size(); ++i) {
        std_sum = std_sum + sqrt(vars[i] + 1e-15);
        abs_mean_sum = abs_mean_sum + abs_means[i];
    }
    auto std_sum_norm = std_sum / abs_mean_sum;
    loss = std_sum_norm * std_sum_norm;
}

template <typename T>
KERNELSPEC void backward_kernel_madnis_multi_channel_variance(
    FIn<T, 1> vars,
    FIn<T, 1> abs_means,
    FIn<T, 0> loss_grad,
    FOut<T, 1> vars_grad,
    FOut<T, 1> abs_means_grad
) {
    FVal<T> std_sum(0.), abs_mean_sum(0.);
    for (std::size_t i = 0; i < vars.size(); ++i) {
        std_sum = std_sum + sqrt(vars[i] + 1e-15);
        abs_mean_sum = abs_mean_sum + abs_means[i];
    }
    auto std_sum_norm = std_sum / abs_mean_sum;
    auto std_sum_grad = 2. * loss_grad * std_sum_norm / abs_mean_sum;
    for (std::size_t i = 0; i < vars.size(); ++i) {
        vars_grad[i] += 0.5 * std_sum_grad / sqrt(vars[i] + 1e-15);
    }
}

template <typename T>
KERNELSPEC void kernel_adam_step(
    FIn<T, 0> gradient,
    FOut<T, 0> parameter,
    FOut<T, 0> exp_avg,
    FOut<T, 0> exp_avg_sq,
    FVal<T> step_size,
    FVal<T> beta1,
    FVal<T> beta2,
    FVal<T> eps,
    FVal<T> bias_corr2_sqrt
) {
    auto gradient2 = gradient * gradient;
    exp_avg = fma(beta1, exp_avg, fma(-beta1, gradient, gradient));
    exp_avg_sq = fma(beta2, exp_avg_sq, fma(-beta2, gradient2, gradient2));
    auto denom = sqrt(exp_avg_sq) / bias_corr2_sqrt + eps;
    parameter = parameter - step_size * exp_avg / denom;
}

} // namespace kernels
} // namespace madspace
