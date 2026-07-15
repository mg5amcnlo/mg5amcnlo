import math
from collections.abc import Callable
from typing import Literal, Protocol

import numpy as np
import torch
import torch.nn as nn

L2PI = -0.5 * math.log(2 * math.pi)

Mapping = Callable[[torch.Tensor, bool], tuple[torch.Tensor, torch.Tensor]]


class Distribution(Protocol):
    """
    Protocol for a (potentially learnable) distribution that can be used for sampling and
    density estimation, like a normalizing flow.
    """

    def sample(
        self,
        n: int,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Draws samples following the distribution

        Args:
            n: number of samples
            c: condition, shape (n, dims_c) or None for an unconditional flow
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
            return_log_prob: if True, also return the log-probabilities
            return_prob: if True, also return the probabilities
            device: device of the returned tensor. Only required if no condition is given.
            dtype: dtype of the returned tensor. Only required if no condition is given.
        Returns:
            samples with shape (n, dims_in). Depending on the arguments ``return_log_prob``,
            ``return_prob``, this function should also return the log-probabilities with shape (n, ),
            the probabilities with shape (n, ).
        """
        ...

    def log_prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> torch.Tensor:
        """
        Computes the log-probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            c: condition, shape (n, dims_c) or None for an unconditional flow
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
        Returns:
            log-probabilities with shape (n, )
        """
        return self.prob(x, c, channel).log()

    def prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> torch.Tensor:
        """
        Computes the probabilities of the input data.

        Args:
            x: input data, shape (n, dims_in)
            c: condition, shape (n, dims_c) or None for an unconditional flow
            channel: encodes the channel of the samples. It must have one of the following types:

                - ``Tensor``: integer tensor of shape (n, ), containing the channel index for every
                  input sample;
                - ``list``: list of integers, specifying the number of samples in each channel;
                - ``int``: integer specifying a single channel containing all the samples;
                - ``None``: used in the single-channel case or to indicate that all channels contain
                  the same number of samples in the multi-channel case.
        Returns:
            probabilities with shape (n, )
        """
        return self.log_prob(x, c, channel).exp()
