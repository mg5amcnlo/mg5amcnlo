import torch
import numpy as np
import madspace as ms
from madspace.madnis import (
    ChannelGrouping,
    Integrator,
    stratified_variance,
    stratified_variance_softclip,
    kl_divergence,
    kl_divergence_softclip,
    build_madnis_integrand,
)


def train_madnis(
    integrands: list[ms.Integrand],
    phasespace,
    madnis_args: dict,
    context: ms.Context,
    log_callback,
) -> None:
    channel_grouping = (
        None if phasespace.symfact is None else ChannelGrouping(phasespace.symfact)
    )
    madnis_integrand, flow, cwnet = build_madnis_integrand(
        integrands, phasespace.cwnet, channel_grouping, context
    )

    loss = {
        "stratified_variance": stratified_variance,
        "stratified_variance_softclip": stratified_variance,
        "kl_divergence": kl_divergence,
        "kl_divergence_softclip": kl_divergence,
    }[madnis_args["loss"]]

    def build_scheduler(optimizer):
        if madnis_args["lr_scheduler"] == "exponential":
            decay_rate = madnis_args["lr_decay"] ** (
                1 / max(madnis_args["train_batches"], 1)
            )
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=decay_rate
            )
        elif madnis_args["lr_scheduler"] == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=madnis_args["lr_max"],
                total_steps=madnis_args["train_batches"],
            )
        elif madnis_args["lr_scheduler"] == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=madnis_args["train_batches"]
            )
        else:
            return None

    madevent_device = context.device()
    if madevent_device == ms.cpu_device():
        device = torch.device("cpu")
    elif madevent_device == ms.cuda_device():
        device = torch.device("cpu")
    elif madevent_device == ms.hip_device():
        device = torch.device("rocm")

    integrator = Integrator(
        integrand=madnis_integrand,
        flow=flow,
        train_channel_weights=cwnet is not None,
        cwnet=cwnet,
        loss=loss,
        batch_size=madnis_args["batch_size_offset"],
        batch_size_per_channel=madnis_args["batch_size_per_channel"],
        learning_rate=madnis_args["lr"],
        scheduler=build_scheduler,
        uniform_channel_ratio=madnis_args["uniform_channel_ratio"],
        integration_history_length=madnis_args["integration_history_length"],
        drop_zero_integrands=madnis_args["drop_zero_integrands"],
        batch_size_threshold=madnis_args["batch_size_threshold"],
        buffer_capacity=madnis_args["buffer_capacity"],
        minimum_buffer_size=madnis_args["minimum_buffer_size"],
        buffered_steps=madnis_args["buffered_steps"],
        max_stored_channel_weights=madnis_args["max_stored_channel_weights"],
        channel_dropping_threshold=madnis_args["channel_dropping_threshold"],
        channel_dropping_interval=madnis_args["channel_dropping_interval"],
        channel_grouping_mode="uniform",
        freeze_cwnet_iteration=int(
            madnis_args["train_batches"] * (1 - madnis_args["fixed_cwnet_fraction"])
        ),
        device=torch.device("cpu" if context.device() == ms.cpu_device() else "cuda:0"),
        dtype=torch.float64,
    )

    log_interval = madnis_args["log_interval"]
    batch_target = madnis_args["train_batches"]
    online_losses = np.full(log_interval, np.nan)
    buffered_losses = np.full(log_interval, np.nan)
    channel_count = len(phasespace.channels)
    def callback(status):
        nonlocal channel_count
        if status.buffered:
            buffered_losses[status.step % log_interval] = status.loss
        else:
            online_losses[status.step % log_interval] = status.loss
        channel_count -= status.dropped_channels
        log_callback(
            status.step,
            batch_target,
            np.nanmean(online_losses),
            status.learning_rate,
            channel_count
        )
        #online_loss = np.mean(online_losses)
        #info = [f"Batch {batch:6d}: loss={online_loss:.6f}"]
        #if len(buffered_losses) > 0:
        #    buffered_loss = np.mean(buffered_losses)
        #    info.append(f"buf={buffered_loss:.6f}")
        #if status.learning_rate is not None:
        #    info.append(f"lr={status.learning_rate:.4e}")
        #if status.dropped_channels > 0:
        #    info.append(f"drop={status.dropped_channels}")

        #print(", ".join(info))
        #online_losses.clear()
        #buffered_losses.clear()

    integrator.train(madnis_args["train_batches"], callback)

    phasespace.channels = [
        channel
        for channel, active in zip(
            phasespace.channels, integrator.active_channels_mask
        )
        if active
    ]
