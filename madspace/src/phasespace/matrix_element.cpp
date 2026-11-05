#include "madspace/phasespace/matrix_element.hpp"

#include <format>
#include <utility>

using namespace madspace;

namespace {

/// A per-particle 0/1 selector, 1 for every one-based particle number in
/// `indices`. `what` names the setting the numbers came from, for the error.
std::vector<me_int_t> particle_mask(
    std::size_t particle_count, const std::vector<me_int_t>& indices, const char* what
) {
    std::vector<me_int_t> mask(particle_count, 0);
    for (me_int_t index : indices) {
        if (index < 1 || std::cmp_greater(index, particle_count)) {
            throw std::invalid_argument(
                std::format(
                    "{} particle {} out of range, this process has {} external "
                    "particles",
                    what,
                    index,
                    particle_count
                )
            );
        }
        mask.at(index - 1) = 1;
    }
    return mask;
}

} // namespace

MatrixElement::MatrixElement(
    std::size_t matrix_element_index,
    std::size_t particle_count,
    const std::vector<MatrixElementInput>& inputs,
    const std::vector<MatrixElementOutput>& outputs,
    std::size_t diagram_count,
    bool sample_random_inputs,
    const std::vector<me_int_t>& me_frame,
    std::size_t incoming_count,
    me_int_t channel
) :
    FunctionGenerator(
        "MatrixElement",
        [&] {
            NamedVector<Type> arg_types;
            for (auto input : inputs) {
                switch (input) {
                case momenta_in:
                    arg_types.push_back(
                        "momenta", batch_four_vec_array(particle_count)
                    );
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
                    if (channel < 0) {
                        arg_types.push_back("channel", batch_int);
                    }
                    break;
                case particle_count_in:
                    arg_types.push_back("particle_count", batch_int);
                    break;
                case pdg_ids_in:
                    arg_types.push_back("pdg_ids", batch_int_array(particle_count));
                    break;
                case momentum_fraction_in:
                    arg_types.push_back("momentum_fraction", batch_float);
                    break;
                case scale_in:
                    arg_types.push_back("scale", batch_float);
                    break;
                case pdg_id_in:
                    arg_types.push_back("pdg_id", batch_int);
                    break;
                case beam_index_in:
                    arg_types.push_back("beam_index", batch_int);
                    break;
                default:
                    throw std::invalid_argument("unknown input type");
                }
            }
            if (arg_types.size() == 0) {
                throw std::invalid_argument("matrix element needs at least one input");
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
                case cut_out:
                    ret_types.push_back("cut", batch_float);
                    break;
                case ren_scale_out:
                    ret_types.push_back("ren_scale", batch_float);
                    break;
                case fact_scale1_out:
                    ret_types.push_back("fact_scale1", batch_float);
                    break;
                case fact_scale2_out:
                    ret_types.push_back("fact_scale2", batch_float);
                    break;
                case pdf_out:
                    ret_types.push_back("pdf", batch_float);
                    break;
                case bias_out:
                    ret_types.push_back("bias", batch_float);
                    break;
                case matrix_element_ratio_out:
                    ret_types.push_back("matrix_element_ratio", batch_float);
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
    _sample_random_inputs(sample_random_inputs),
    _channel(channel),
    _frame_mask(
        me_frame.empty() ? std::vector<me_int_t>()
                         : particle_mask(particle_count, me_frame, "me_frame")
    ),
    _reference_mask([&] {
        if (_frame_mask.empty()) {
            return std::vector<me_int_t>();
        }
        if (incoming_count == 0) {
            // an all-zero mask would be read by the kernel as "no frame"
            throw std::invalid_argument(
                "a process needs at least one incoming particle to define the "
                "frame me_frame is reached from"
            );
        }
        std::vector<me_int_t> incoming;
        for (std::size_t i = 1; i <= incoming_count; ++i) {
            incoming.push_back(static_cast<me_int_t>(i));
        }
        auto mask = particle_mask(particle_count, incoming, "incoming_count");
        // asking for the rest frame of the incoming system is the reference
        // frame itself: one boost gets there. A collision does not go through
        // it at all, see _along_beam.
        if (mask == _frame_mask || incoming_count == 2) {
            mask.clear();
        }
        return mask;
    }()),
    _along_beam([&] {
        if (_frame_mask.empty() || incoming_count != 2) {
            return false;
        }
        std::vector<me_int_t> incoming{1, 2};
        return particle_mask(particle_count, incoming, "incoming_count") != _frame_mask;
    }()) {
    auto has_output = [&](MatrixElementOutput output) {
        return std::find(outputs.begin(), outputs.end(), output) != outputs.end();
    };
    if (has_output(matrix_element_out) && has_output(matrix_element_ratio_out)) {
        throw std::invalid_argument(
            "a matrix element returns either matrix_element or matrix_element_ratio, "
            "not both"
        );
    }
}

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
        case particle_count_in:
            input_key = UMAMI_IN_PARTICLE_COUNT;
            break;
        case pdg_ids_in:
            input_key = UMAMI_IN_PDG_IDS;
            break;
        case momentum_fraction_in:
            input_key = UMAMI_IN_MOMENTUM_FRACTION;
            break;
        case scale_in:
            input_key = UMAMI_IN_SCALE;
            break;
        case pdg_id_in:
            input_key = UMAMI_IN_PDG_ID;
            break;
        case beam_index_in:
            input_key = UMAMI_IN_BEAM_INDEX;
            break;
        }
        matrix_args.push_back(static_cast<me_int_t>(input_key));
        if (_sample_random_inputs &&
            (input == random_color_in || input == random_helicity_in ||
             input == random_diagram_in)) {
            matrix_args.push_back(random.at(random_index));
            ++random_index;
        } else if (input == channel_in && _channel >= 0) {
            matrix_args.push_back(fb.full({_channel, fb.batch_size(args.values())}));
        } else if (input == momenta_in && !_frame_mask.empty()) {
            // Only the matrix element sees the me_frame-boosted momenta; cuts,
            // scales and the written event use the generated ones.
            // A collision first boosts along the beam to zero rapidity of the
            // selection, then into its rest frame, so the result does not depend
            // on the starting frame (no Wigner rotation). A decay starts from
            // the rest frame of the decaying particle.
            Value momenta = args.at(arg_index);
            if (_along_beam) {
                momenta = fb.boost_to_zero_rapidity(momenta, _frame_mask);
            } else if (!_reference_mask.empty()) {
                momenta = fb.boost_to_frame(momenta, _reference_mask);
            }
            matrix_args.push_back(fb.boost_to_frame(momenta, _frame_mask));
            ++arg_index;
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
        case cut_out:
            output_key = UMAMI_OUT_CUT;
            break;
        case ren_scale_out:
            output_key = UMAMI_OUT_REN_SCALE;
            break;
        case fact_scale1_out:
            output_key = UMAMI_OUT_FACT_SCALE_1;
            break;
        case fact_scale2_out:
            output_key = UMAMI_OUT_FACT_SCALE_2;
            break;
        case pdf_out:
            output_key = UMAMI_OUT_PDF;
            break;
        case bias_out:
            output_key = UMAMI_OUT_BIAS;
            break;
        case matrix_element_ratio_out:
            output_key = UMAMI_OUT_MATRIX_ELEMENT_RATIO;
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
        if (input == channel_in && _channel >= 0) {
            continue;
        }
        if (!_sample_random_inputs ||
            (input != random_color_in && input != random_helicity_in &&
             input != random_diagram_in)) {
            ret.push_back(input);
        }
    }
    return ret;
}
