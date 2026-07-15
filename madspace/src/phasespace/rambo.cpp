#include "madspace/phasespace/rambo.hpp"

#include <cmath>

#include "madspace/constants.hpp"

using namespace madspace;

FastRamboMapping::FastRamboMapping(std::size_t n_particles, bool massless, bool com) :
    Mapping(
        "FastRamboMapping",
        [&] {
            NamedVector<Type> input_types;
            for (std::size_t i = 0; i < 3 * n_particles - 4; ++i) {
                input_types.push_back(std::format("random_{}", i), batch_float);
            }
            if (!com) {
                input_types.push_back("com_momentum", batch_four_vec);
            }
            return input_types;
        }(),
        [&] {
            NamedVector<Type> output_types;
            for (std::size_t i = 0; i < n_particles; ++i) {
                output_types.push_back(std::format("momentum_{}", i), batch_four_vec);
            }
            return output_types;
        }(),
        [&] {
            NamedVector<Type> cond_types{{"com_energy", batch_float}};
            if (!massless) {
                for (std::size_t i = 0; i < n_particles; ++i) {
                    cond_types.push_back(std::format("mass_{}", i), batch_float);
                }
            }
            return cond_types;
        }()
    ),
    _n_particles(n_particles),
    _massless(massless),
    _com(com) {
    if (n_particles < 3 || n_particles > 12) {
        throw std::invalid_argument("The number of particles must be between 3 and 12");
    }
}

Mapping::Result FastRamboMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value r = fb.stack(ValueVec(inputs.begin(), inputs.begin() + random_dim()));
    Value e_cm = conditions.at(0);

    std::array<Value, 2> output;
    if (_massless) {
        if (_com) {
            output = fb.fast_rambo_massless_com(r, e_cm);
        } else {
            Value p0 = inputs.back();
            output = fb.fast_rambo_massless(r, e_cm, p0);
        }
    } else {
        if (_com) {
            ValueVec masses(conditions.begin() + 1, conditions.end());
            output = fb.fast_rambo_massive_com(r, e_cm, fb.stack(masses));
        } else {
            ValueVec masses(conditions.begin() + 1, conditions.end());
            Value p0 = inputs.back();
            output = fb.fast_rambo_massive(r, e_cm, fb.stack(masses), p0);
        }
    }
    return {{output_types().keys(), fb.unstack(output[0])}, output[1]};
}

Mapping::Result FastRamboMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value p_out = fb.stack(inputs.values());
    Value e_cm = conditions.at(0);

    auto [r, p0, det] = _massless
        ? fb.fast_rambo_massless_inverse(p_out, e_cm)
        : fb.fast_rambo_massive_inverse(
              p_out, e_cm, fb.stack(ValueVec(conditions.begin() + 1, conditions.end()))
          );
    ValueVec inv_inputs = fb.unstack(r);
    if (!_com) {
        inv_inputs.push_back(p0);
    }
    return {{input_types().keys(), inv_inputs}, det};
}
