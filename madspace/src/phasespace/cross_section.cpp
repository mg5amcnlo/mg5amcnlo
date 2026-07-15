#include "madspace/phasespace/cross_section.hpp"

using namespace madspace;

DifferentialCrossSection::DifferentialCrossSection(
    const MatrixElement& matrix_element,
    double cm_energy,
    const std::optional<RunningCoupling>& running_coupling,
    const std::variant<std::monostate, EnergyScale, CachedScale>& energy_scale,
    const nested_vector2<me_int_t>& pid_options,
    const std::variant<std::monostate, PdfGrid, CachedPdf>& pdf1,
    const std::variant<std::monostate, PdfGrid, CachedPdf>& pdf2,
    bool input_momentum_fraction
) :
    FunctionGenerator(
        "DifferentialCrossSection",
        [&] {
            NamedVector<Type> arg_types;
            for (auto [arg_type, arg_name, input] :
                 zip(matrix_element.arg_types(),
                     matrix_element.arg_types().keys(),
                     matrix_element.external_inputs())) {
                if (input != MatrixElement::alpha_s_in) {
                    arg_types.push_back(arg_name, arg_type);
                }
            }
            if (input_momentum_fraction) {
                arg_types.push_back("x1", batch_float);
                arg_types.push_back("x2", batch_float);
            }
            arg_types.push_back("pdf_id", batch_int);
            bool uses_cached_pdf = false;
            auto add_pdf_args = [&](auto& pdf, int index) {
                if (std::holds_alternative<CachedPdf>(pdf)) {
                    arg_types.push_back(std::format("pdf{}", index + 1), batch_float);
                    uses_cached_pdf = true;
                }
            };
            add_pdf_args(pdf1, 0);
            add_pdf_args(pdf2, 1);
            if (std::holds_alternative<CachedScale>(energy_scale)) {
                if (std::holds_alternative<PdfGrid>(pdf1) ||
                    std::holds_alternative<PdfGrid>(pdf2)) {
                    throw std::invalid_argument(
                        "PDFs cannot be computed in cached scale mode"
                    );
                }
                arg_types.push_back("alpha_s", batch_float);
            }
            return arg_types;
        }(),
        matrix_element.return_types()
    ),
    _pid_options(pid_options),
    _matrix_element(matrix_element),
    _has_pdf(
        {!std::holds_alternative<std::monostate>(pdf1),
         !std::holds_alternative<std::monostate>(pdf2)}
    ),
    _running_coupling(running_coupling),
    _e_cm(cm_energy),
    _energy_scale(energy_scale),
    _input_momentum_fraction(input_momentum_fraction) {
    auto init_pdf = [&](auto& pdf, int index) {
        if (std::holds_alternative<PdfGrid>(pdf)) {
            std::vector<int> pids;
            for (auto& option : pid_options) {
                pids.push_back(option.at(index));
            }
            _pdfs.at(index) = PartonDensity(std::get<PdfGrid>(pdf), pids, true);
        }
    };
    init_pdf(pdf1, 0);
    init_pdf(pdf2, 1);
}

NamedVector<Value> DifferentialCrossSection::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    std::size_t arg_index = 0;
    int alpha_s_index = -1;
    Value momenta;
    ValueVec matrix_args;
    for (auto input : _matrix_element.external_inputs()) {
        if (input == MatrixElement::alpha_s_in) {
            alpha_s_index = arg_index;
        } else {
            Value arg = args.at(arg_index++);
            if (input == MatrixElement::momenta_in) {
                momenta = arg;
            }
            matrix_args.push_back(arg);
        }
    }

    std::array<Value, 2> x1x2;
    if (_input_momentum_fraction) {
        x1x2 = {args.at(arg_index), args.at(arg_index + 1)};
        arg_index += 2;
    } else {
        x1x2 = fb.momenta_to_x1x2(momenta, _e_cm);
    }
    auto pdf_flavor_id = args.at(arg_index++);
    NamedVector<Value> scales;
    if (std::holds_alternative<EnergyScale>(_energy_scale)) {
        scales = std::get<EnergyScale>(_energy_scale).build_function(fb, {momenta});
    }

    std::array<Value, 2> pdf_outputs{1., 1.};
    for (std::size_t i = 0;
         auto [pdf_output, pdf, x, has_pdf] : zip(pdf_outputs, _pdfs, x1x2, _has_pdf)) {
        if (pdf) {
            pdf_output =
                pdf.value()
                    .build_function(fb, {x, scales.at(i + 1), pdf_flavor_id})
                    .at(0);
        } else if (has_pdf) {
            pdf_output = args.at(arg_index++);
        }
    }

    if (alpha_s_index != -1) {
        matrix_args.insert(
            matrix_args.begin() + alpha_s_index,
            std::holds_alternative<CachedScale>(_energy_scale)
                ? args.at(arg_index)
                : _running_coupling.value().build_function(fb, {scales.at(0)}).at(0)
        );
    }

    auto me_result = _matrix_element.build_function(fb, matrix_args);
    auto search = std::find(
        _matrix_element.outputs().begin(),
        _matrix_element.outputs().end(),
        MatrixElement::matrix_element_out
    );
    if (search == _matrix_element.outputs().end()) {
        throw std::runtime_error("matrix element missing in return values");
    }
    std::size_t me_index = search - _matrix_element.outputs().begin();
    me_result.at(me_index) = fb.diff_cross_section(
        x1x2.at(0),
        x1x2.at(1),
        pdf_outputs.at(0),
        pdf_outputs.at(1),
        me_result.at(me_index),
        _e_cm * _e_cm
    );
    return me_result;
}
