#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/channel_weight_network.hpp"

namespace madspace {

class MadnisLoss : public FunctionGenerator {
public:
    MadnisLoss(
        const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
        const std::optional<ChannelWeightNetwork>& cwnet,
        double softclip_threshold = 0.0
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::shared_ptr<FunctionGenerator>> _functions;
    std::optional<ChannelWeightNetwork> _cwnet;
    double _softclip_threshold;
};

} // namespace madspace
