#include "madspace/phasespace/weighting_step.hpp"

#include "madspace/util.hpp"

using namespace madspace;

namespace {

const std::array<MatrixElement::MatrixElementOutput, 3> index_outputs{
    MatrixElement::color_index_out,
    MatrixElement::helicity_index_out,
    MatrixElement::diagram_index_out,
};

int output_position(
    const MatrixElement& matrix_element, MatrixElement::MatrixElementOutput output
) {
    auto& outputs = matrix_element.outputs();
    auto search = std::find(outputs.begin(), outputs.end(), output);
    return search == outputs.end() ? -1 : static_cast<int>(search - outputs.begin());
}

} // namespace

bool WeightingStep::returns_ratio() const {
    return !matrix_elements.empty() &&
        output_position(
            matrix_elements.at(0), MatrixElement::matrix_element_ratio_out
        ) != -1;
}

WeightingStepFunction::WeightingStepFunction(
    const WeightingStep& step,
    std::size_t particle_count,
    const std::array<bool, 3>& write_indices,
    std::size_t step_index,
    std::size_t step_count
) :
    FunctionGenerator(
        "WeightingStepFunction",
        [&] {
            NamedVector<Type> arg_types{
                {"momenta", batch_four_vec_array(particle_count)},
                {"alpha_s", batch_float},
                {"flavor", batch_int},
            };
            if (step.matrix_elements.size() > 1) {
                arg_types.push_back("matrix_element_index", batch_int);
            }
            arg_types.push_back("color_index", batch_int);
            arg_types.push_back("helicity_index", batch_int);
            arg_types.push_back("diagram_index", batch_int);
            arg_types.push_back("random", batch_float_array(random_count * step_count));
            return arg_types;
        }(),
        {
            {"matrix_element", batch_float},
            {"color_index", batch_int},
            {"helicity_index", batch_int},
            {"diagram_index", batch_int},
            {"unweight_random", batch_float},
        }
    ),
    _matrix_elements(step.matrix_elements),
    _write_indices(write_indices),
    _step_index(step_index) {
    if (step_index >= step_count) {
        throw std::invalid_argument("step index out of range");
    }
    if (_matrix_elements.empty()) {
        throw std::invalid_argument("a weighting step needs a matrix element");
    }
    for (auto& matrix_element : _matrix_elements) {
        if (matrix_element.outputs() != _matrix_elements.at(0).outputs()) {
            throw std::invalid_argument(
                "the matrix elements of a weighting step need the same outputs"
            );
        }
        if (matrix_element.particle_count() != particle_count) {
            throw std::invalid_argument(
                "the matrix elements of a weighting step need the particle count "
                "of the events"
            );
        }
        if (output_position(matrix_element, MatrixElement::matrix_element_out) == -1 &&
            output_position(matrix_element, MatrixElement::matrix_element_ratio_out) ==
                -1) {
            throw std::invalid_argument(
                "the matrix elements of a weighting step must return the matrix "
                "element or its ratio"
            );
        }
        for (auto input : matrix_element.external_inputs()) {
            switch (input) {
            case MatrixElement::momenta_in:
            case MatrixElement::alpha_s_in:
            case MatrixElement::flavor_in:
            case MatrixElement::random_color_in:
            case MatrixElement::random_helicity_in:
            case MatrixElement::random_diagram_in:
            case MatrixElement::helicity_in:
            case MatrixElement::diagram_in:
                break;
            default:
                throw std::invalid_argument(
                    "unsupported input of a weighting step matrix element"
                );
            }
        }
        bool has_random_inputs = std::any_of(
            matrix_element.inputs().begin(),
            matrix_element.inputs().end(),
            [](auto input) {
                return input == MatrixElement::random_color_in ||
                    input == MatrixElement::random_helicity_in ||
                    input == MatrixElement::random_diagram_in;
            }
        );
        if (has_random_inputs && matrix_element.sample_random_inputs()) {
            throw std::invalid_argument(
                "the random inputs of a weighting step matrix element are passed in "
                "and must not be sampled internally"
            );
        }
    }
}

NamedVector<Value> WeightingStepFunction::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec all_random = fb.unstack(args.at("random"));
    ValueVec random(
        all_random.begin() + random_count * _step_index,
        all_random.begin() + random_count * (_step_index + 1)
    );
    ValueVec indices{
        args.at("color_index"), args.at("helicity_index"), args.at("diagram_index")
    };
    auto me_args = [&](const MatrixElement& matrix_element, auto&& arg) {
        ValueVec ret;
        for (auto input : matrix_element.inputs()) {
            switch (input) {
            case MatrixElement::momenta_in:
                ret.push_back(arg(args.at("momenta")));
                break;
            case MatrixElement::alpha_s_in:
                ret.push_back(arg(args.at("alpha_s")));
                break;
            case MatrixElement::flavor_in:
                ret.push_back(arg(args.at("flavor")));
                break;
            case MatrixElement::random_color_in:
                ret.push_back(arg(random.at(0)));
                break;
            case MatrixElement::random_helicity_in:
                ret.push_back(arg(random.at(1)));
                break;
            case MatrixElement::random_diagram_in:
                ret.push_back(arg(random.at(2)));
                break;
            case MatrixElement::helicity_in:
                ret.push_back(arg(indices.at(1)));
                break;
            case MatrixElement::diagram_in:
                ret.push_back(arg(indices.at(2)));
                break;
            default:
                break;
            }
        }
        return ret;
    };
    // matrix element plus the index outputs, missing ones are empty
    auto select_outputs =
        [&](const MatrixElement& matrix_element, const ValueVec& outputs) {
            int value_position =
                output_position(matrix_element, MatrixElement::matrix_element_out);
            if (value_position == -1) {
                value_position = output_position(
                    matrix_element, MatrixElement::matrix_element_ratio_out
                );
            }
            ValueVec ret{outputs.at(value_position)};
            for (auto output : index_outputs) {
                int position = output_position(matrix_element, output);
                ret.push_back(position == -1 ? Value() : outputs.at(position));
            }
            return ret;
        };

    ValueVec results;
    if (_matrix_elements.size() == 1) {
        auto& matrix_element = _matrix_elements.at(0);
        results = select_outputs(
            matrix_element,
            matrix_element
                .build_function(fb, me_args(matrix_element, [](Value v) { return v; }))
                .values()
        );
    } else {
        ValueVec split_indices = fb.batch_split_by_index(
            args.at("matrix_element_index"),
            static_cast<me_int_t>(_matrix_elements.size())
        );
        std::vector<ValueVec> split_results(1 + index_outputs.size());
        for (auto [matrix_element, split_index] :
             zip(_matrix_elements, split_indices)) {
            ValueVec outputs = select_outputs(
                matrix_element,
                matrix_element
                    .build_function(
                        fb,
                        me_args(
                            matrix_element,
                            [&](Value v) { return fb.batch_gather(split_index, v); }
                        )
                    )
                    .values()
            );
            for (auto [output, split] : zip(outputs, split_results)) {
                if (output) {
                    split.push_back(output);
                    split.push_back(split_index);
                }
            }
        }
        for (auto& split : split_results) {
            results.push_back(split.empty() ? Value() : fb.batch_merge_by_index(split));
        }
    }

    ValueVec outputs{results.at(0)};
    for (std::size_t i = 0; i < index_outputs.size(); ++i) {
        Value result = results.at(i + 1);
        outputs.push_back(result && _write_indices.at(i) ? result : indices.at(i));
    }
    outputs.push_back(random.at(3));
    return {return_types().keys(), outputs};
}
