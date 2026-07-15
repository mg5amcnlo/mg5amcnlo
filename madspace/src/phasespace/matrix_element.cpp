#include "madspace/phasespace/matrix_element.hpp"

using namespace madspace;

MatrixElement::MatrixElement(
    std::size_t matrix_element_index,
    std::size_t particle_count,
    const std::vector<MatrixElementInput>& inputs,
    const std::vector<MatrixElementOutput>& outputs,
    std::size_t diagram_count,
    bool sample_random_inputs
) :
    FunctionGenerator(
        "MatrixElement",
        [&] {
            NamedVector<Type> arg_types;
            bool found_momenta = false;
            for (auto input : inputs) {
                switch (input) {
                case momenta_in:
                    arg_types.push_back(
                        "momenta", batch_four_vec_array(particle_count)
                    );
                    found_momenta = true;
                    break;
                case alpha_s_in:
                    arg_types.push_back("alpha_s", batch_float);
                    break;
                case random_color_in:
                    if (!sample_random_inputs) {
                        arg_types.push_back("random_color", batch_float);
                    }
                    break;
                case random_helicity_in:
                    if (!sample_random_inputs) {
                        arg_types.push_back("random_helicity", batch_float);
                    }
                    break;
                case random_diagram_in:
                    if (!sample_random_inputs) {
                        arg_types.push_back("random_diagram", batch_float);
                    }
                    break;
                case flavor_in:
                    arg_types.push_back("flavor", batch_int);
                    break;
                case helicity_in:
                    arg_types.push_back("helicity", batch_int);
                    break;
                case diagram_in:
                    arg_types.push_back("diagram", batch_int);
                    break;
                case channel_in:
                    arg_types.push_back("channel", batch_int);
                    break;
                default:
                    throw std::invalid_argument("unknown input type");
                }
            }
            if (!found_momenta) {
                throw std::invalid_argument("missing momentum input");
            }
            return arg_types;
        }(),
        [&] {
            NamedVector<Type> ret_types;
            for (auto output : outputs) {
                switch (output) {
                case matrix_element_out:
                    ret_types.push_back("matrix_element", batch_float);
                    break;
                case diagram_amp2_out:
                    ret_types.push_back(
                        "diagram_amp2", batch_float_array(diagram_count)
                    );
                    break;
                case color_index_out:
                    ret_types.push_back("color_index", batch_int);
                    break;
                case helicity_index_out:
                    ret_types.push_back("helicity_index", batch_int);
                    break;
                case diagram_index_out:
                    ret_types.push_back("diagram_index", batch_int);
                    break;
                default:
                    throw std::invalid_argument("unknown output type");
                }
            }
            return ret_types;
        }()
    ),
    _matrix_element_index(matrix_element_index),
    _inputs(inputs),
    _outputs(outputs),
    _particle_count(particle_count),
    _diagram_count(diagram_count),
    _sample_random_inputs(sample_random_inputs) {}

NamedVector<Value> MatrixElement::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec matrix_args{
        static_cast<me_int_t>(_matrix_element_index),
        static_cast<me_int_t>(_inputs.size()),
        static_cast<me_int_t>(_outputs.size())
    };
    ValueVec random;
    if (_sample_random_inputs) {
        std::size_t random_count = 0;
        for (auto& input : _inputs) {
            if (input == random_color_in || input == random_helicity_in ||
                input == random_diagram_in) {
                ++random_count;
            }
        }
        random = fb.unstack(
            fb.random(fb.batch_size(args.values()), static_cast<me_int_t>(random_count))
        );
    }
    for (std::size_t arg_index = 0, random_index = 0; auto& input : _inputs) {
        UmamiInputKey input_key;
        switch (input) {
        case momenta_in:
            input_key = UMAMI_IN_MOMENTA;
            break;
        case alpha_s_in:
            input_key = UMAMI_IN_ALPHA_S;
            break;
        case flavor_in:
            input_key = UMAMI_IN_FLAVOR_INDEX;
            break;
        case random_color_in:
            input_key = UMAMI_IN_RANDOM_COLOR;
            break;
        case random_helicity_in:
            input_key = UMAMI_IN_RANDOM_HELICITY;
            break;
        case random_diagram_in:
            input_key = UMAMI_IN_RANDOM_DIAGRAM;
            break;
        case helicity_in:
            input_key = UMAMI_IN_HELICITY_INDEX;
            break;
        case diagram_in:
            input_key = UMAMI_IN_DIAGRAM_INDEX;
            break;
        case channel_in:
            input_key = UMAMI_IN_CHANNEL_INDEX;
            break;
        }
        matrix_args.push_back(static_cast<me_int_t>(input_key));
        if (_sample_random_inputs &&
            (input == random_color_in || input == random_helicity_in ||
             input == random_diagram_in)) {
            matrix_args.push_back(random.at(random_index));
            ++random_index;
        } else {
            matrix_args.push_back(args.at(arg_index));
            ++arg_index;
        }
    }

    for (auto output : _outputs) {
        UmamiOutputKey output_key;
        me_int_t size_arg = 0;
        switch (output) {
        case matrix_element_out:
            output_key = UMAMI_OUT_MATRIX_ELEMENT;
            break;
        case diagram_amp2_out:
            output_key = UMAMI_OUT_DIAGRAM_AMP2;
            size_arg = _diagram_count;
            break;
        case color_index_out:
            output_key = UMAMI_OUT_COLOR_INDEX;
            break;
        case helicity_index_out:
            output_key = UMAMI_OUT_HELICITY_INDEX;
            break;
        case diagram_index_out:
            output_key = UMAMI_OUT_DIAGRAM_INDEX;
            break;
        }
        matrix_args.push_back(static_cast<me_int_t>(output_key));
        matrix_args.push_back(size_arg);
    }
    return {return_types().keys(), fb.matrix_element(matrix_args)};
}

std::vector<MatrixElement::MatrixElementInput> MatrixElement::external_inputs() const {
    std::vector<MatrixElement::MatrixElementInput> ret;
    for (auto input : _inputs) {
        if (!_sample_random_inputs ||
            (input != random_color_in && input != random_helicity_in &&
             input != random_diagram_in)) {
            ret.push_back(input);
        }
    }
    return ret;
}
