from collections.abc import Callable

import torch
import torch.nn as nn

from .. import _madspace_py_loader as ms
from ..torch import FunctionModule
from .channel_grouping import ChannelGrouping
from .distribution import Distribution
from .integrand import Integrand


class IntegrandDistribution(nn.Module, Distribution):
    def __init__(
        self,
        channels: list[ms.Integrand],
        channel_remap_function: Callable[[torch.Tensor], torch.Tensor],
        context: ms.Context,
    ):
        super().__init__()
        self.channel_count = len(channels)
        self.channels = channels
        self.context = context
        self.channel_remap_function = channel_remap_function
        self.latent_dims, self.latent_float = channels[0].latent_dims()
        self.integrand_prob = None
        self.update_channel_mask(torch.ones(self.channel_count, dtype=torch.bool))

    def update_channel_mask(self, mask: torch.Tensor) -> None:
        self.channel_mask = mask
        multi_prob = ms.MultiChannelFunction(
            [
                ms.IntegrandProbability(chan)
                for chan, active in zip(self.channels, mask)
                if active
            ]
        )
        func = multi_prob.function()
        if self.integrand_prob is None:
            self.integrand_prob = FunctionModule(func, self.context)
        else:
            self.integrand_prob.runtime = ms.FunctionRuntime(func, self.context)

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
        raise NotImplementedError(
            "IntegrandDistribution does not support sampling directly. "
            "Use the underlying ms.Integrand object instead."
        )

    def prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> torch.Tensor:
        channel_perm = None
        if isinstance(channel, torch.Tensor):
            channel = self.channel_remap_function(channel)
            channel_perm = torch.argsort(channel)
            x = x[channel_perm]
            channel = channel.bincount(minlength=self.channel_count).to(torch.int32)
        elif channel is None:
            channel = torch.tensor([len(x)], dtype=torch.int32)
        else:
            raise NotImplementedError("channel argument type not supported")
        channel = channel[self.channel_mask]

        prob_args = [
            xi if is_float else xi[:, 0].to(torch.int32)
            for xi, is_float in zip(x.split(self.latent_dims, dim=1), self.latent_float)
        ]
        prob = self.integrand_prob(*prob_args, channel.cpu())
        if channel_perm is None:
            return prob
        else:
            channel_perm_inv = torch.argsort(channel_perm)
            return prob[channel_perm_inv]


class IntegrandFunction:
    def __init__(self, channels: list[ms.Integrand], context: ms.Context):
        self.channel_count = len(channels)
        self.channels = channels
        self.context = context
        self.update_channel_mask(torch.ones(self.channel_count, dtype=torch.bool))

    def update_channel_mask(self, mask: torch.Tensor) -> None:
        self.channel_mask = mask.cpu()
        multi_integrand = ms.MultiChannelIntegrand(
            [chan for chan, active in zip(self.channels, mask) if active]
        )
        self.multi_runtime = ms.FunctionRuntime(
            multi_integrand.function(), self.context
        )

    def __call__(self, channels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        channel_perm = torch.argsort(channels)
        channels = channels.bincount(minlength=self.channel_count).cpu().to(torch.int32)
        channels = channels[self.channel_mask]
        (
            weight,
            latent,
            prob,
            chan_index,
            alphas_prior,
            y,
            *rest,
        ) = self.multi_runtime(channels)

        x_parts = [latent, *rest]
        x = torch.cat(
            [xi.double().reshape(latent.shape[0], -1) for xi in x_parts], dim=1
        )
        channel_perm_inv = torch.argsort(channel_perm)
        return (
            x[channel_perm_inv],
            prob[channel_perm_inv],
            weight[channel_perm_inv],
            y[channel_perm_inv],
            alphas_prior[channel_perm_inv],
            chan_index[channel_perm_inv].to(torch.int64),
        )


def build_madnis_integrand(
    channels: list[ms.Integrand],
    cwnet: ms.ChannelWeightNetwork | None = None,
    channel_grouping: ChannelGrouping | None = None,
    context: ms.Context = ms.default_context(),
) -> tuple[Integrand, Distribution, nn.Module | None]:
    device = torch.device("cpu" if context.device() == ms.cpu_device() else "cuda:0")
    if channel_grouping is None:
        remap_channels = lambda channels: channels
        group_indices = torch.arange(len(channels))
    else:
        channel_id_map = torch.tensor(
            [channel.group.group_index for channel in channel_grouping.channels],
            device=device,
        )
        remap_channels = lambda channels: channel_id_map[channels]
        group_indices = torch.tensor(
            [group.target_index for group in channel_grouping.groups], device=device
        )

    integrand_function = IntegrandFunction(channels, context)
    flow = IntegrandDistribution(channels, remap_channels, context)

    def update_mask(mask: torch.Tensor) -> None:
        context.get_global(cwnet.mask_name()).torch()[0, :] = mask.double()
        group_mask = mask[group_indices]
        if torch.any(group_mask.cpu() != integrand_function.channel_mask):
            integrand_function.update_channel_mask(group_mask)
            flow.update_channel_mask(group_mask)

    integrand = Integrand(
        function=integrand_function,
        input_dim=sum(channels[0].latent_dims()[0]),
        channel_count=len(channel_grouping.channels),
        remapped_dim=cwnet.preprocessing().output_dim(),
        has_channel_weight_prior=cwnet is not None,
        channel_grouping=channel_grouping,
        function_includes_sampling=True,
        update_active_channels_mask=update_mask,
    )
    cwnet_module = (
        None if cwnet is None else FunctionModule(cwnet.mlp().function(), context)
    )
    return integrand, flow, cwnet_module
