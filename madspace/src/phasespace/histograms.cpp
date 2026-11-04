#include "madspace/phasespace/histograms.hpp"

using namespace madspace;

ObservableHistograms::ObservableHistograms(const std::vector<HistItem>& observables) :
    FunctionGenerator(
        "ObservablesHistogram",
        {{"weight", batch_float},
         {"momenta", observables.at(0).observable.arg_types().at(0)}},
        [&]() {
            NamedVector<Type> ret_types;
            for (std::size_t i = 0; auto& obs : observables) {
                ret_types.push_back(
                    std::format("values{}", i), single_float_array(obs.bin_count + 2)
                );
                ret_types.push_back(
                    std::format("square_values{}", i),
                    single_float_array(obs.bin_count + 2)
                );
                ++i;
            }
            return ret_types;
        }()
    ),
    _observables(observables) {}

NamedVector<Value> ObservableHistograms::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value weight = args.at(0), momenta = args.at(1);
    ValueVec histograms;
    for (auto& obs : _observables) {
        Value obs_result = obs.observable.build_function(fb, {momenta}).at(0);
        auto [values, square_values] = fb.histogram(
            obs_result, weight, obs.min, obs.max, static_cast<me_int_t>(obs.bin_count)
        );
        histograms.push_back(values);
        histograms.push_back(square_values);
    }
    return {return_types().keys(), histograms};
}

ObservableValues::ObservableValues(const std::vector<Observable>& observables) :
    FunctionGenerator(
        "ObservableValues",
        {{"momenta", observables.at(0).arg_types().at(0)}},
        [&]() {
            NamedVector<Type> ret_types;
            for (std::size_t i = 0; i < observables.size(); ++i) {
                // event-level observables can return a vector per event
                ret_types.push_back(
                    std::format("value{}", i), observables.at(i).return_types().at(0)
                );
            }
            return ret_types;
        }()
    ),
    _observables(observables) {}

NamedVector<Value> ObservableValues::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value momenta = args.at(0);
    ValueVec values;
    for (auto& obs : _observables) {
        values.push_back(obs.build_function(fb, {momenta}).at(0));
    }
    return {return_types().keys(), values};
}
