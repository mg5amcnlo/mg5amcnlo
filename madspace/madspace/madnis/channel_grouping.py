from dataclasses import dataclass


@dataclass
class ChannelGroup:
    """
    A group of channels

    Args:
        group_index: index of the group in the list of groups
        target_index: index of the channel that all other channels in the group are mapped to
        channel_indices: indices of the channels in the group
    """

    group_index: int
    target_index: int
    channel_indices: list[int]


@dataclass
class ChannelData:
    """
    Information about a single channel

    Args:
        channel_index: index of the channel
        target_index: index of the channel that it is mapped to
        group: channel group that the channel belongs to
        remapped: True if the channel is remapped to another channel
        position_in_group: index of the channel within its group
    """

    channel_index: int
    target_index: int
    group: ChannelGroup
    remapped: bool
    position_in_group: int


class ChannelGrouping:
    """
    Class that encodes how channels are grouped together for a multi-channel integrand
    """

    def __init__(self, channel_assignment: list[int | None]):
        """
        Args:
            channel_assignment: list with an entry for each channel. If None, the channel is not
                remapped. Otherwise, the index of the channel to which it is mapped.
        """
        group_dict = {}
        for source_channel, target_channel in enumerate(channel_assignment):
            if target_channel is None:
                group_dict[source_channel] = ChannelGroup(
                    group_index=len(group_dict),
                    target_index=source_channel,
                    channel_indices=[source_channel],
                )

        self.channels: list[ChannelData] = []
        self.groups: list[ChannelGroup] = list(group_dict.values())

        for source_channel, target_channel in enumerate(channel_assignment):
            if target_channel is None:
                self.channels.append(
                    ChannelData(
                        channel_index=source_channel,
                        target_index=source_channel,
                        group=group_dict[source_channel],
                        remapped=False,
                        position_in_group=0,
                    )
                )
            else:
                group = group_dict[target_channel]
                self.channels.append(
                    ChannelData(
                        channel_index=source_channel,
                        target_index=target_channel,
                        group=group,
                        remapped=True,
                        position_in_group=len(group.channel_indices),
                    )
                )
                group.channel_indices.append(source_channel)
