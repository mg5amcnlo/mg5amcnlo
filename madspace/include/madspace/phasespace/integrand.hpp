#pragma once

#include "madspace/phasespace/channel_weight_network.hpp"
#include "madspace/phasespace/channel_weights.hpp"
#include "madspace/phasespace/cross_section.hpp"
#include "madspace/phasespace/discrete_flow.hpp"
#include "madspace/phasespace/discrete_sampler.hpp"
#include "madspace/phasespace/flow.hpp"
#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/phasespace/phasespace.hpp"
#include "madspace/phasespace/unweighter.hpp"
#include "madspace/phasespace/vegas.hpp"
#include "madspace/util.hpp"

namespace madspace {

class Integrand : public FunctionGenerator {
public:
    using AdaptiveMapping = std::variant<std::monostate, VegasMapping, Flow>;
    using AdaptiveDiscrete =
        std::variant<std::monostate, DiscreteSampler, DiscreteFlow>;

    inline static const std::vector<MatrixElement::MatrixElementInput>
        matrix_element_inputs = {
            MatrixElement::momenta_in,
            MatrixElement::alpha_s_in,
            MatrixElement::flavor_in,
            MatrixElement::random_color_in,
            MatrixElement::random_helicity_in,
            MatrixElement::random_diagram_in,
        };
    inline static const std::vector<MatrixElement::MatrixElementOutput>
        matrix_element_outputs = {
            MatrixElement::matrix_element_out,
            MatrixElement::diagram_amp2_out,
            MatrixElement::color_index_out,
            MatrixElement::helicity_index_out,
            MatrixElement::diagram_index_out,
        };

    Integrand(
        const PhaseSpaceMapping& mapping,
        const DifferentialCrossSection& diff_xs,
        const AdaptiveMapping& adaptive_map = std::monostate{},
        const AdaptiveDiscrete& discrete_before = std::monostate{},
        const AdaptiveDiscrete& discrete_after = std::monostate{},
        const std::optional<PdfGrid>& pdf_grid = std::nullopt,
        const std::optional<RunningCoupling>& running_coupling = std::nullopt,
        const std::optional<EnergyScale>& energy_scale = std::nullopt,
        const std::optional<PropagatorChannelWeights>& prop_chan_weights = std::nullopt,
        const std::optional<SubchannelWeights>& subchan_weights = std::nullopt,
        const std::optional<ChannelWeightNetwork>& chan_weight_net = std::nullopt,
        const std::vector<me_int_t>& chan_weight_remap = {},
        std::size_t remapped_chan_count = 0,
        bool madnis_training = false,
        bool drop_cuts_and_rescale = false,
        bool partial_weights = false,
        const std::vector<std::size_t>& channel_indices = {},
        const std::vector<std::size_t>& active_flavors = {},
        const std::vector<std::size_t>& flavor_remap = {},
        const std::vector<double>& flavor_factors = {},
        const std::vector<bool>& flavor_mirror = {}
    );
    std::size_t particle_count() const { return _mapping.particle_count(); }
    bool madnis_training() const { return _madnis_training; }
    std::optional<std::string> vegas_grid_name() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->grid_name();
        } else {
            return std::nullopt;
        }
    }
    std::size_t vegas_dimension() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->dimension();
        } else {
            return 0;
        }
    }
    std::size_t vegas_bin_count() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->bin_count();
        } else {
            return 0;
        }
    }
    const PhaseSpaceMapping& mapping() const { return _mapping; }
    const DifferentialCrossSection& diff_xs() const { return _diff_xs; }
    const AdaptiveMapping& adaptive_map() const { return _adaptive_map; }
    const AdaptiveDiscrete& discrete_before() const { return _discrete_before; }
    const AdaptiveDiscrete& discrete_after() const { return _discrete_after; }
    const std::optional<EnergyScale>& energy_scale() const { return _energy_scale; }
    const std::optional<PropagatorChannelWeights>& prop_chan_weights() const {
        return _prop_chan_weights;
    }
    const std::optional<ChannelWeightNetwork>& chan_weight_net() const {
        return _chan_weight_net;
    }
    const std::size_t random_dim() const { return _random_dim; }
    std::tuple<std::vector<std::size_t>, std::vector<bool>> latent_dims() const;
    const std::vector<me_int_t>& channel_indices() const { return _channel_indices; }
    const std::vector<std::size_t>& active_flavors() const { return _active_flavors; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
    NamedVector<Type> compute_channel_part_ret_types() const;
    NamedVector<Value>
    build_channel_part(FunctionBuilder& fb, const NamedVector<Value>& args) const;
    NamedVector<Value>
    build_common_part(FunctionBuilder& fb, const NamedVector<Value>& channel_out) const;

    PhaseSpaceMapping _mapping;
    DifferentialCrossSection _diff_xs;
    AdaptiveMapping _adaptive_map;
    AdaptiveDiscrete _discrete_before;
    AdaptiveDiscrete _discrete_after;
    std::array<std::optional<PartonDensity>, 2> _pdfs;
    std::array<std::vector<me_int_t>, 2> _pdf_indices;
    std::optional<RunningCoupling> _running_coupling;
    std::optional<EnergyScale> _energy_scale;
    std::optional<PropagatorChannelWeights> _prop_chan_weights;
    std::optional<SubchannelWeights> _subchan_weights;
    std::optional<ChannelWeightNetwork> _chan_weight_net;
    std::vector<me_int_t> _chan_weight_remap;
    me_int_t _remapped_chan_count;
    bool _madnis_training;
    bool _drop_cuts_and_rescale;
    bool _partial_weights;
    std::vector<me_int_t> _channel_indices;
    me_int_t _random_dim;
    std::size_t _latent_dim;
    std::vector<std::size_t> _active_flavors;
    std::vector<double> _active_flavors_mask;
    std::vector<me_int_t> _flavor_remap;
    std::vector<double> _flavor_factors;
    std::vector<me_int_t> _flavor_mirror;
    bool _has_mirror;
    NamedVector<Type> _channel_part_ret_types;

    friend class IntegrandProbability;
    friend class IntegrandChannelPart;
    friend class IntegrandCommonPart;
    friend class IntegrandConcatenator;
    friend class MultiChannelIntegrand;
};

class IntegrandChannelPart : public FunctionGenerator {
public:
    IntegrandChannelPart(const Integrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    const Integrand& _integrand;
};

class IntegrandCommonPart : public FunctionGenerator {
public:
    IntegrandCommonPart(const Integrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    const Integrand& _integrand;
};

class IntegrandConcatenator : public FunctionGenerator {
public:
    IntegrandConcatenator(const Integrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    const Integrand& _integrand;
};

class MultiChannelIntegrand : public FunctionGenerator {
public:
    MultiChannelIntegrand(
        const std::vector<std::shared_ptr<Integrand>>& integrands,
        bool return_sizes = false
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::shared_ptr<Integrand>> _integrands;
    bool _return_sizes;
};

class IntegrandProbability : public FunctionGenerator {
public:
    IntegrandProbability(const Integrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    Integrand::AdaptiveMapping _adaptive_map;
    Integrand::AdaptiveDiscrete _discrete_before;
    Integrand::AdaptiveDiscrete _discrete_after;
    std::size_t _permutation_count;
    std::size_t _flavor_count;
    bool _has_pdf_prior;
};

} // namespace madspace
