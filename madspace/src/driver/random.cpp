#include "madspace/driver/random.hpp"

#include <format>
#include <random>

namespace madspace {

DerivedSeed::DerivedSeed(
    const std::optional<std::uint64_t>& seed,
    SeedType seed_type,
    std::size_t job_index,
    std::size_t channel_index,
    std::size_t stream_index
) {
    std::uint64_t main_seed;
    if (seed) {
        main_seed = seed.value();
    } else {
        std::random_device rand_device;
        main_seed = rand_device();
    }
    if (channel_index >= max_channel_count) {
        throw std::invalid_argument(
            std::format(
                "channel index {} exceeded {}", channel_index, max_channel_count - 1
            )
        );
    }
    if (job_index >= max_job_count) {
        throw std::invalid_argument(
            std::format("job index {} exceeded {}", job_index, max_job_count - 1)
        );
    }
    if (stream_index >= max_stream_count) {
        throw std::invalid_argument(
            std::format(
                "stream index {} exceeded {}", stream_index, max_stream_count - 1
            )
        );
    }
    seed_parts[0] = static_cast<std::uint32_t>(main_seed >> 32);
    seed_parts[1] = static_cast<std::uint32_t>(main_seed & 0xFFFFFFFFULL);
    seed_parts[2] = job_index;
    seed_parts[3] =
        ((static_cast<std::uint32_t>(seed_type) << 28) |
         (static_cast<std::uint32_t>(channel_index) << 16) |
         static_cast<std::uint32_t>(stream_index));
}

} // namespace madspace
