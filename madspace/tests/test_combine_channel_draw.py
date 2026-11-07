"""Unit tests for select_combine_channel, the rule that maps a random index onto
the channel that owns it when combining events (see generator_data.hpp and
EventGenerator::read_and_combine).

The draw is over per-channel cumulative counts of the events each channel still
owes, with the random index uniform on [0, total). Every index must resolve to a
channel that still has a share left: read_and_combine decrements the drawn
channel's count, so handing out a channel with a zero share wraps an unsigned
counter, and the channel is then drawn far beyond the events its file holds.
"""

import madspace as ms


def cumulative(counts):
    total = 0
    cum_counts = []
    for count in counts:
        total += count
        cum_counts.append(total)
    return cum_counts


def draw_counts(counts):
    """How often each channel is drawn over the whole index range [0, total)."""
    cum_counts = cumulative(counts)
    drawn = [0] * len(counts)
    for random_index in range(cum_counts[-1]):
        drawn[ms.select_combine_channel_index(cum_counts, random_index)] += 1
    return drawn


def test_every_channel_gets_exactly_its_share():
    # Over the full index range each channel is drawn as often as its count.
    counts = [5, 1, 12, 3, 40]
    assert draw_counts(counts) == counts


def test_zero_share_channel_is_never_drawn():
    # A channel with a zero share is unreachable, including at index 0, which a
    # lower-bound draw would hand to it.
    counts = [0, 7, 0, 0, 4, 0]
    assert draw_counts(counts) == counts


def test_index_zero_goes_to_the_first_channel_with_a_share():
    assert ms.select_combine_channel_index(cumulative([3, 2]), 0) == 0
    assert ms.select_combine_channel_index(cumulative([0, 2]), 0) == 1
    assert ms.select_combine_channel_index(cumulative([0, 0, 2]), 0) == 2


def test_boundaries_are_exact():
    # Channel 0 owns [0, 3), channel 1 owns [3, 4), channel 2 owns [4, 9).
    cum_counts = cumulative([3, 1, 5])
    assert [ms.select_combine_channel_index(cum_counts, i) for i in range(9)] == [
        0,
        0,
        0,
        1,
        2,
        2,
        2,
        2,
        2,
    ]


def test_single_channel_takes_every_index():
    cum_counts = cumulative([4])
    assert [ms.select_combine_channel_index(cum_counts, i) for i in range(4)] == [
        0,
        0,
        0,
        0,
    ]


def test_out_of_range_index_yields_no_channel():
    # Not reachable from read_and_combine (the index is drawn below the total),
    # but the rule must not silently return a channel that doesn't own it.
    cum_counts = cumulative([2, 3])
    assert ms.select_combine_channel_index(cum_counts, 5) == len(cum_counts)
    assert ms.select_combine_channel_index(cumulative([0]), 0) == 1
