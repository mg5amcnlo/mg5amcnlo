#include "madspace/phasespace/custom_function.hpp"

#include "madspace/phasespace/matrix_element.hpp"

#include <format>

using namespace madspace;

namespace {

std::vector<MatrixElement::MatrixElementOutput>
output_keys(CustomFunction::CustomFunctionType type) {
    switch (type) {
    case CustomFunction::cut:
        return {MatrixElement::cut_out};
    case CustomFunction::scale:
        return {
            MatrixElement::ren_scale_out,
            MatrixElement::fact_scale1_out,
            MatrixElement::fact_scale2_out
        };
    case CustomFunction::bias:
        return {MatrixElement::bias_out};
    }
    throw std::invalid_argument("unknown custom function type");
}

NamedVector<Type> output_types(CustomFunction::CustomFunctionType type) {
    switch (type) {
    case CustomFunction::cut:
        return {{"cut", batch_float}};
    case CustomFunction::scale:
        return {
            {"ren_scale", batch_float},
            {"fact_scale1", batch_float},
            {"fact_scale2", batch_float}
        };
    case CustomFunction::bias:
        return {{"bias", batch_float}};
    }
    throw std::invalid_argument("unknown custom function type");
}

} // namespace

CustomFunction::CustomFunction(
    std::size_t function_index,
    CustomFunctionType type,
    std::size_t particle_count,
    const nested_vector2<me_int_t>& pdg_id_options
) :
    FunctionGenerator(
        "CustomFunction",
        [&] {
            NamedVector<Type> arg_types{
                {"momenta", batch_four_vec_array(particle_count)}
            };
            if (pdg_id_options.size() > 1) {
                arg_types.push_back("flavor_index", batch_int);
            }
            return arg_types;
        }(),
        output_types(type)
    ),
    _function_index(function_index),
    _type(type),
    _particle_count(particle_count),
    _pdg_id_options(pdg_id_options) {
    for (auto& option : pdg_id_options) {
        if (option.size() != particle_count) {
            throw std::invalid_argument(
                std::format(
                    "custom function: expected {} PDG ids per option, got {}",
                    particle_count,
                    option.size()
                )
            );
        }
    }
}

NamedVector<Value> CustomFunction::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto momenta = args.at(0);
    auto batch_size = fb.batch_size({momenta});
    ValueVec pdg_ids;
    for (std::size_t i = 0; i < _particle_count; ++i) {
        if (_pdg_id_options.empty()) {
            pdg_ids.push_back(fb.full({static_cast<me_int_t>(0), batch_size}));
        } else if (_pdg_id_options.size() == 1) {
            pdg_ids.push_back(fb.full({_pdg_id_options.at(0).at(i), batch_size}));
        } else {
            std::vector<me_int_t> column;
            for (auto& option : _pdg_id_options) {
                column.push_back(option.at(i));
            }
            pdg_ids.push_back(fb.gather_int(args.at(1), column));
        }
    }
    MatrixElement function(
        _function_index,
        _particle_count,
        {MatrixElement::momenta_in,
         MatrixElement::particle_count_in,
         MatrixElement::pdg_ids_in},
        output_keys(_type)
    );
    auto particle_count = fb.full({static_cast<me_int_t>(_particle_count), batch_size});
    return function.build_function(fb, {momenta, particle_count, fb.stack(pdg_ids)});
}
