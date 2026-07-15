#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/observable.hpp"

namespace madspace {

class ObservableHistograms : public FunctionGenerator {
public:
    struct HistItem {
        Observable observable;
        double min;
        double max;
        std::size_t bin_count;
    };
    ObservableHistograms(const std::vector<HistItem>& observables);
    const std::vector<HistItem>& observables() const { return _observables; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<HistItem> _observables;
};

} // namespace madspace
