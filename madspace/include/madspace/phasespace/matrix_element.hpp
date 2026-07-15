#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

class MatrixElement : public FunctionGenerator {
public:
    enum MatrixElementInput {
        momenta_in,
        alpha_s_in,
        flavor_in,
        random_color_in,
        random_helicity_in,
        random_diagram_in,
        helicity_in,
        channel_in,
        diagram_in
    };

    enum MatrixElementOutput {
        matrix_element_out,
        diagram_amp2_out,
        color_index_out,
        helicity_index_out,
        diagram_index_out
    };

    MatrixElement(
        std::size_t matrix_element_index,
        std::size_t particle_count,
        const std::vector<MatrixElementInput>& inputs = {momenta_in},
        const std::vector<MatrixElementOutput>& outputs = {matrix_element_out},
        std::size_t diagram_count = 1,
        bool sample_random_inputs = false
    );
    MatrixElement(
        const MatrixElementApi& matrix_element_api,
        const std::vector<MatrixElementInput>& inputs = {momenta_in},
        const std::vector<MatrixElementOutput>& outputs = {matrix_element_out},
        bool sample_random_inputs = false
    ) :
        MatrixElement(
            matrix_element_api.index(),
            matrix_element_api.particle_count(),
            inputs,
            outputs,
            matrix_element_api.diagram_count(),
            sample_random_inputs
        ) {};
    std::size_t matrix_element_index() const { return _matrix_element_index; }
    std::size_t diagram_count() const { return _diagram_count; }
    std::size_t particle_count() const { return _particle_count; }
    const std::vector<MatrixElementInput>& inputs() const { return _inputs; }
    const std::vector<MatrixElementOutput>& outputs() const { return _outputs; }
    std::vector<MatrixElementInput> external_inputs() const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _matrix_element_index;
    std::size_t _particle_count;
    std::size_t _diagram_count;
    std::vector<MatrixElementInput> _inputs;
    std::vector<MatrixElementOutput> _outputs;
    bool _sample_random_inputs;
};

} // namespace madspace
