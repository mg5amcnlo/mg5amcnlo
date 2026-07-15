"""Implementation of multi-channel loss functions"""

from collections.abc import Callable
from functools import wraps

import torch


def dtype_epsilon(tensor: torch.Tensor) -> float:
    return torch.finfo(tensor.dtype).eps


def softclip(x: torch.Tensor, threshold: torch.Tensor = 30.0):
    return threshold * torch.arcsinh(x / threshold)


SingleChannelLoss = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
MultiChannelLoss = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor
]


def multi_channel_loss(loss: SingleChannelLoss) -> MultiChannelLoss:
    """
    Turns a single-channel loss function into a multi-channel loss function by evaluating it for
    each channel separately and then adding them weighted by TODO weighted by what?

    Args:
        loss: single-channel loss function, that expects the integrand value, test probability and
            sampling probability as arguments
    Returns:
        multi-channel loss function, that expects the integrand value, test probability and,
        optionally, sampling probability and channel indices as arguments.
    """

    # TODO: this unfortunately does not yield the correct signature (with the extra channels argument),
    # so it does not show up in the documentation
    @wraps(loss)
    def wrapped_multi(
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor | None = None,
        channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_sample is None:
            q_sample = q_test
        if channels is None:
            return loss(f_true, q_test, q_sample)

        loss_tot = 0
        for channel in channels.unique():
            mask = channels == channel
            fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
            ni = mask.count_nonzero()
            # loss_tot += ni / q_sample.shape[0] * loss(fi, qti, qsi) if ni > 0 else 0.0
            loss_tot += loss(fi, qti, qsi) if ni > 0 else 0.0
        return loss_tot

    return wrapped_multi


def stratified_variance(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor | None = None,
    channels: torch.Tensor | None = None,
):
    """
    Computes the stratified variance as introduced in [2311.01548] for two given sets of
    probabilities, ``f_true`` and ``q_test``. It uses importance sampling with a sampling
    probability specified by ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
    Returns:
        computed stratified variance
    """
    if q_sample is None:
        q_sample = q_test
    if channels is None:
        abs_integral = torch.mean(f_true.detach().abs() / q_sample)
        return _variance(f_true, q_test, q_sample) / abs_integral.square()

    stddev_sum = 0
    abs_integral = 0
    for i in channels.unique():
        mask = channels == i
        fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
        stddev_sum += torch.sqrt(_variance(fi, qti, qsi) + dtype_epsilon(f_true))
        abs_integral += torch.mean(fi.detach().abs() / qsi)
    return (stddev_sum / abs_integral) ** 2


def stratified_variance_softclip(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor | None = None,
    channels: torch.Tensor | None = None,
    threshold: torch.Tensor = 30.0,
):
    """
    Computes the stratified variance as introduced in [2311.01548] for two given sets of
    probabilities, ``f_true`` and ``q_test``. It uses importance sampling with a sampling
    probability specified by ``q_sample``. A soft clipping function is applied to the
    sample weights.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
        threshold: approximate point of transition between linear and logarithmic behavior
    Returns:
        computed stratified variance
    """
    if q_sample is None:
        q_sample = q_test
    if channels is None:
        norm = torch.mean(f_true.detach().abs() / q_sample)
        f_true = softclip(f_true / q_sample / norm) * q_sample * norm
        abs_integral = torch.mean(f_true.detach().abs() / q_sample)
        return _variance(f_true, q_test, q_sample) / abs_integral.square()

    stddev_sum = 0
    abs_integral = 0
    for i in channels.unique():
        mask = channels == i
        fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
        norm = torch.mean(fi.detach().abs() / qsi)
        fi = softclip(fi / qsi / norm) * qsi * norm
        stddev_sum += torch.sqrt(_variance(fi, qti, qsi) + dtype_epsilon(f_true))
        abs_integral += torch.mean(fi.detach().abs() / qsi)
    return (stddev_sum / abs_integral) ** 2


@multi_channel_loss
def variance(
    f_true: torch.Tensor, q_test: torch.Tensor, q_sample: torch.Tensor
) -> torch.Tensor:
    abs_integral = torch.mean(f_true.detach().abs() / q_sample) + dtype_epsilon(f_true)
    return _variance(f_true, q_test, q_sample) / abs_integral.square()


def _variance(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the variance for two given sets of probabilities, ``f_true`` and ``q_test``. It uses
    importance sampling with a sampling probability specified by ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
    Returns:
        computed variance
    """
    ratio = q_test / q_sample
    mean = torch.mean(f_true / q_sample)
    sq = (f_true / q_test - mean) ** 2
    return (
        torch.mean(sq * ratio)
        if len(f_true) > 0
        else torch.tensor(0.0, device=f_true.device, dtype=f_true.dtype)
    )


@multi_channel_loss
def kl_divergence(
    f_true: torch.Tensor, q_test: torch.Tensor, q_sample: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence for two given sets of probabilities, ``f_true`` and
    ``q_test``. It uses importance sampling, i.e. the estimator is divided by an additional factor
    of ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
    Returns:
        computed KL divergence
    """
    f_true = f_true.detach().abs()
    f_true /= torch.mean(f_true / q_sample)
    log_q = torch.log(q_test)
    log_f = torch.log(f_true + dtype_epsilon(f_true))
    return torch.mean(f_true / q_sample * (log_f - log_q))


@multi_channel_loss
def kl_divergence_softclip(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor,
    threshold: torch.Tensor = 30.0,
) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence for two given sets of probabilities, ``f_true`` and
    ``q_test``. It uses importance sampling, i.e. the estimator is divided by an additional factor
    of ``q_sample``. A soft clipping function is applied to the sample weights.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
        threshold: approximate point of transition between linear and logarithmic behavior
    Returns:
        computed KL divergence
    """
    f_true = f_true.detach().abs()
    weight = f_true / q_sample
    weight /= weight.abs().mean()
    clipped_weight = softclip(weight, threshold)
    log_q = torch.log(q_test)
    log_f = torch.log(clipped_weight * q_sample + dtype_epsilon(f_true))
    return torch.mean(clipped_weight * (log_f - log_q))
