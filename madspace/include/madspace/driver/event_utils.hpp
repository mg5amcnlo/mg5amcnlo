#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "madspace/driver/channel_generator.hpp"
#include "madspace/driver/generator_data.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/driver/lhe_output.hpp"
#include "madspace/driver/random.hpp"

namespace madspace {

/// Weight of an unweighted event in units of its channel's maximum weight:
/// +-1, or larger in magnitude for an overweight event.
inline double unweighted_event_weight(double weight, double max_weight) {
    return std::copysign(std::max(1., std::abs(weight) / max_weight), weight);
}

/// Layout of combined events: weight, subprocess index and event data, plus
/// the extra fields (`extra_event_flags`, `extra_particle_flags`) the channels
/// carry, except for the bias, which is divided out of the weight.
DataLayout combined_event_layout(int extra_event_flags, int extra_particle_flags);

/// The MLM alpha_s vertex scale and pdf ladder particle flags of any of
/// `channels`: a merged run mixes 2 -> 1 samples without vertices with the jet
/// samples, and channel 0 may be either.
int mlm_particle_flags(
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels
);

/**
 * Copy event `in_index` of a channel's event buffer to event `out_index` of a
 * combined buffer (layout @ref combined_event_layout), padding the momenta of
 * missing particles with zeros.
 *
 * @param weight            Weight to store for the event, divided by the bias
 *                          if the channel records one.
 * @param subprocess_index  Stored if the channel records no per-event
 *                          subprocess index.
 * @param extra_flags       The channel's extra `EventRecord` layout flags.
 * @param in_particle_flags The channel's extra `ParticleRecord` layout flags.
 * @param out_particle_flags The extra `ParticleRecord` layout flags of the
 *                          combined buffer.
 */
void copy_channel_event(
    EventBuffer& in_buffer,
    std::size_t in_index,
    EventBuffer& out_buffer,
    std::size_t out_index,
    double weight,
    int subprocess_index,
    int extra_flags,
    int in_particle_flags,
    int out_particle_flags
);

/// Build `lhe_event` from event `event_index` of a combined buffer and
/// complete it with `lhe_completer`. `extra_event_flags` and
/// `extra_particle_flags` are the extra layout flags of the buffer's channels.
void fill_lhe_event(
    LHECompleter& lhe_completer,
    LHEEvent& lhe_event,
    EventBuffer& buffer,
    std::size_t event_index,
    MixMaxRandom& rand_gen,
    int extra_event_flags,
    int extra_particle_flags
);

/// Write the particles of `lhe_event` to event `event_index` of a buffer with
/// LHE particle layout, padding the remaining slots with empty particles.
void write_lhe_particles(
    const LHEEvent& lhe_event, EventBuffer& buffer, std::size_t event_index
);

/// Sum the integrals and event counts of `channels` into `status`. Throws if
/// an integral is not finite.
void sum_channel_status(
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
    GeneratorStatus& status
);

} // namespace madspace
