#include "madspace/phasespace/integrand.hpp"

#include "madspace/util.hpp"

#include <set>

using namespace madspace;

Integrand::Integrand(
    const PhaseSpaceMapping& mapping,
    const DifferentialCrossSection& diff_xs,
    const AdaptiveMapping& adaptive_map,
    const AdaptiveDiscrete& discrete_before,
    const AdaptiveDiscrete& discrete_after,
    const std::optional<PdfGrid>& pdf_grid,
    const std::optional<EnergyScale>& energy_scale,
    const std::optional<PropagatorChannelWeights>& prop_chan_weights,
    const std::optional<SubchannelWeights>& subchan_weights,
    const std::optional<ChannelWeightNetwork>& chan_weight_net,
    const std::vector<me_int_t>& chan_weight_remap,
    std::size_t remapped_chan_count,
    int flags,
    const std::vector<std::size_t>& channel_indices,
    const std::vector<std::size_t>& active_flavors,
    const std::vector<std::size_t>& flavor_remap,
    const std::vector<double>& flavor_factors
) :
    FunctionGenerator(
        "Integrand",
        [&] {
            NamedVector<Type> arg_types;
            if (flags & sample) {
                arg_types.push_back("batch_size", Type({batch_size}));
            } else {
                arg_types.push_back(
                    "random",
                    batch_float_array(
                        mapping.random_dim() +               // phasespace
                        (mapping.channel_count() > 1) +      // symmetric channel
                        (diff_xs.pid_options().size() > 1) + // flavor
                        diff_xs.has_mirror()                 // flipped initial state
                    )
                );
            }
            if (flags & unweight) {
                arg_types.push_back("max_weight", single_float);
            }
            return arg_types;
        }(),
        [&] {
            NamedVector<Type> ret_types;
            if (flags & exclude_adaptive_and_chan_weight) {
                ret_types.push_back("full_weight", batch_float);
                ret_types.push_back("weight", batch_float);
            } else {
                ret_types.push_back("weight", batch_float);
            }
            if (flags & return_momenta) {
                ret_types.push_back(
                    "momenta", batch_four_vec_array(mapping.particle_count())
                );
            }
            if (flags & return_x1_x2) {
                ret_types.push_back("x1", batch_float);
                ret_types.push_back("x2", batch_float);
            }
            if (flags & return_indices) {
                ret_types.push_back("color_index", batch_int);
                ret_types.push_back("helicity_index", batch_int);
                ret_types.push_back("diagram_index", batch_int);
                ret_types.push_back("flavor_index", batch_int);
            }
            if (flags & return_random) {
                ret_types.push_back("random", batch_float_array(mapping.random_dim()));
            }
            if (flags & return_latent) {
                if (std::holds_alternative<std::monostate>(adaptive_map)) {
                    throw std::invalid_argument(
                        "return_latent flag can only be set if adaptive mapping is "
                        "present"
                    );
                }
                ret_types.push_back("latent", batch_float_array(mapping.random_dim()));
                ret_types.push_back("adaptive_prob", batch_float);
            }
            if (flags & return_channel) {
                ret_types.push_back("channel_index", batch_int);
            }
            if (flags & return_chan_weights) {
                ret_types.push_back(
                    "channel_weights",
                    batch_float_array(
                        chan_weight_remap.size() > 0 ? remapped_chan_count
                            : subchan_weights        ? subchan_weights->channel_count()
                                              : diff_xs.matrix_element().diagram_count()
                    )
                );
            }
            if (flags & return_cwnet_input) {
                ret_types.push_back(
                    "cwnet_input",
                    batch_float_array(
                        chan_weight_net.value().preprocessing().output_dim()
                    )
                );
            }
            auto flav_count = diff_xs.pid_options().size();
            if (flags & return_discrete) {
                if (mapping.channel_count() > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_before)) {
                    ret_types.push_back("channel_index_in_group", batch_int);
                }
                if (flav_count > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_after)) {
                    ret_types.push_back("discrete_flavor_index", batch_int);
                }
            }
            if (flags & return_discrete_latent) {
                ret_types.push_back("channel_index_in_group", batch_int);
                if (flav_count > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_after)) {
                    ret_types.push_back("discrete_flavor_index", batch_int);
                    if (pdf_grid && energy_scale) {
                        ret_types.push_back("pdf_prior", batch_float_array(flav_count));
                    }
                }
            }
            return ret_types;
        }()
    ),
    _mapping(mapping),
    _diff_xs(diff_xs),
    _adaptive_map(adaptive_map),
    _discrete_before(discrete_before),
    _discrete_after(discrete_after),
    _energy_scale(energy_scale),
    _prop_chan_weights(prop_chan_weights),
    _subchan_weights(subchan_weights),
    _chan_weight_net(chan_weight_net),
    _chan_weight_remap(chan_weight_remap),
    _remapped_chan_count(remapped_chan_count),
    _flags(flags),
    _channel_indices(channel_indices.begin(), channel_indices.end()),
    _random_dim(
        mapping.random_dim() +               // phasespace
        (mapping.channel_count() > 1) +      // symmetric channel
        (diff_xs.pid_options().size() > 1) + // flavor
        diff_xs.has_mirror()                 // flipped initial state
    ),
    _flavor_remap(flavor_remap.begin(), flavor_remap.end()),
    _flavor_factors(flavor_factors),
    _active_flavors(active_flavors) {
    if (pdf_grid) {
        for (std::size_t i = 0; i < 2; ++i) {
            std::set<int> pids;
            for (auto& option : diff_xs.pid_options()) {
                pids.insert(option.at(i));
            }
            for (auto& option : diff_xs.pid_options()) {
                _pdf_indices.at(i).push_back(
                    std::distance(pids.begin(), pids.find(option.at(i)))
                );
            }
            _pdfs.at(i) = PartonDensity(pdf_grid.value(), {pids.begin(), pids.end()});
        }
        if (active_flavors.size() > 0 &&
            active_flavors.size() < diff_xs.pid_options().size()) {
            _active_flavors_mask.resize(diff_xs.pid_options().size());
            for (auto index : active_flavors) {
                _active_flavors_mask.at(index) = 1.;
            }
        }
    }
}

std::tuple<std::vector<std::size_t>, std::vector<bool>> Integrand::latent_dims() const {
    std::vector<std::size_t> dims{_mapping.random_dim(), 1};
    std::vector<bool> is_float{true, false};

    auto flav_count = _diff_xs.pid_options().size();
    if (flav_count > 1 && !std::holds_alternative<std::monostate>(_discrete_after)) {
        dims.push_back(1);
        is_float.push_back(false);
        if ((_pdfs.at(0) || _pdfs.at(1)) && _energy_scale) {
            dims.push_back(flav_count);
            is_float.push_back(true);
        }
    }

    return {dims, is_float};
}

NamedVector<Value> Integrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    bool has_multi_flavor = _diff_xs.pid_options().size() > 1;
    ChannelArgs channel_args{
        .r = _flags & sample ? fb.random(args.at(0), _random_dim) : args.at(0),
        .batch_size = _flags & sample ? args.at(0) : fb.batch_size({args.at(0)}),
        .has_permutations = _mapping.channel_count() > 1,
        .has_multi_flavor = has_multi_flavor,
        .has_mirror = _diff_xs.has_mirror(),
        .has_pdf_prior =
            (_pdfs.at(0) || _pdfs.at(1)) && _energy_scale && has_multi_flavor,
    };
    if (_flags & unweight) {
        channel_args.max_weight = args.at(1);
    }
    ChannelResult result = build_channel_part(fb, channel_args);
    return build_common_part(fb, channel_args, result);
}

Integrand::ChannelResult Integrand::build_channel_part(
    FunctionBuilder& fb, const Integrand::ChannelArgs& args
) const {
    ChannelResult result;
    ValueVec mapping_conditions;
    ValueVec weights_before_cuts, weights_after_cuts, adaptive_probs;
    ValueVec extra_weights_before_cuts, extra_weights_after_cuts;

    // split up random numbers depending on discrete degrees of freedom
    Value chan_random, flavor_random, mirror_random;
    result.r() = args.r;
    if (args.has_permutations) {
        auto [r_rest, r_val] = fb.pop(result.r());
        result.r() = r_rest;
        chan_random = r_val;
    }
    if (args.has_multi_flavor) {
        auto [r_rest, r_val] = fb.pop(result.r());
        result.r() = r_rest;
        flavor_random = r_val;
    }
    if (args.has_mirror) {
        auto [r_rest, r_val] = fb.pop(result.r());
        result.r() = r_rest;
        mirror_random = r_val;
    }

    // if the channel contains multiple symmetry permutations, sample them either
    // uniformly, adaptively or NN-based depending on _discrete_before
    ValueVec flow_conditions;
    if (args.has_permutations) {
        me_int_t opt_count = _channel_indices.size();
        std::visit(
            Overloaded{
                [&](std::monostate) {
                    auto [index, chan_det] = fb.sample_discrete(chan_random, opt_count);
                    result.chan_index_in_group() = index;
                    weights_before_cuts.push_back(chan_det);
                },
                [&](auto discrete_before) {
                    auto discrete_result =
                        discrete_before.build_forward(fb, {chan_random}, {});
                    result.chan_index_in_group() = discrete_result.at(0);
                    if (_flags & exclude_adaptive_and_chan_weight) {
                        extra_weights_before_cuts.push_back(discrete_result["det"]);
                    } else {
                        weights_before_cuts.push_back(discrete_result["det"]);
                    }
                    adaptive_probs.push_back(discrete_result["det"]);
                }
            },
            _discrete_before
        );
        result.chan_index() =
            fb.gather_int(result.chan_index_in_group(), _channel_indices);
        mapping_conditions.push_back(result.chan_index_in_group());
        // flow_conditions.push_back(fb.one_hot(chan_index_in_group, opt_count));
    } else {
        result.chan_index() =
            fb.full({static_cast<me_int_t>(_channel_indices.at(0)), args.batch_size});
        result.chan_index_in_group() =
            fb.full({static_cast<me_int_t>(0), args.batch_size});
    }

    // apply VEGAS or MadNIS if given in _adaptive_map
    result.latent() = result.r();
    std::visit(
        Overloaded{
            [&](std::monostate) {},
            [&](auto& admap) {
                ValueVec cond;
                using TAdaptive = std::decay_t<decltype(admap)>;
                if constexpr (std::is_same_v<TAdaptive, Flow>) {
                    if (flow_conditions.size() == 1) {
                        cond.push_back(flow_conditions.at(0));
                    } else if (flow_conditions.size() > 1) {
                        cond.push_back(fb.cat(flow_conditions));
                    }
                }
                auto admap_result = admap.build_forward(fb, {result.r()}, cond);
                result.latent() = admap_result["data"];
                adaptive_probs.push_back(admap_result["det"]);
                if (_flags & exclude_adaptive_and_chan_weight) {
                    extra_weights_before_cuts.push_back(admap_result["det"]);
                } else {
                    weights_before_cuts.push_back(admap_result["det"]);
                }
                flow_conditions.push_back(result.latent());
            }
        },
        _adaptive_map
    );

    // apply phase space mapping
    auto mapping_result =
        _mapping.build_forward(fb, {result.latent()}, mapping_conditions);
    weights_before_cuts.push_back(mapping_result["det"]);
    result.momenta() = mapping_result["momenta"];
    result.x(0) = mapping_result["x1"];
    result.x(1) = mapping_result["x2"];

    if (args.has_mirror) {
        auto [index, mirror_det] =
            fb.sample_discrete(mirror_random, static_cast<me_int_t>(2));
        result.mirror_id() = index;
        result.momenta_mirror() =
            fb.mirror_momenta(result.momenta(), result.mirror_id());
        weights_before_cuts.push_back(mirror_det);
    }

    // filter out events that did not pass cuts
    result.weight_before_cuts() = fb.product(weights_before_cuts);
    if (extra_weights_before_cuts.size() > 0) {
        result.extra_weight_before_cuts() = fb.product(extra_weights_before_cuts);
    }
    result.indices_acc() = fb.nonzero(result.weight_before_cuts());
    result.momenta_acc() = fb.batch_gather(result.indices_acc(), result.momenta());
    for (std::size_t i = 0; i < 2; ++i) {
        result.x_acc(i) = fb.batch_gather(result.indices_acc(), result.x(i));
    }
    for (auto& cond : flow_conditions) {
        cond = fb.batch_gather(result.indices_acc(), cond);
    }

    // if PDF grid and energy scale were given and the channel has more than one flavor,
    // evaluate energy scale and PDF for all flavors to use it as prior for flavor
    // sampling
    if (args.has_pdf_prior) {
        auto scales = _energy_scale.value().build_function(fb, {result.momenta_acc()});
        ValueVec pdf_priors;
        for (std::size_t i = 0; i < 2; ++i) {
            if (_diff_xs.has_pdf(i)) {
                auto pdf =
                    _pdfs.at(i)
                        .value()
                        .build_function(fb, {result.x_acc(i), scales.at(i + 1)})
                        .at(0);
                result.pdf_cache(i) = pdf;
                pdf_priors.push_back(fb.select(pdf, _pdf_indices.at(i)));
            }
        }
        result.pdf_prior() = fb.product(pdf_priors);
        if (_active_flavors_mask.size() > 0) {
            result.pdf_prior() = fb.mul(result.pdf_prior(), _active_flavors_mask);
        }
        result.scale_cache() = scales.at(0);
    }

    // if the channel has more than one flavor, sample them either uniformly,
    // adaptively or NN-based depending on _discrete_after
    auto batch_size_acc = fb.batch_size({result.momenta_acc()});
    result.flavor_id() = fb.full({static_cast<me_int_t>(0), batch_size_acc});
    if (args.has_multi_flavor) {
        auto flavor_random_acc = fb.batch_gather(result.indices_acc(), flavor_random);
        std::visit(
            Overloaded{
                [&](std::monostate) {
                    if (args.has_pdf_prior) {
                        auto [index, flavor_det] = fb.sample_discrete_probs(
                            flavor_random_acc, result.pdf_prior()
                        );
                        result.flavor_id() = index;
                        weights_after_cuts.push_back(flavor_det);
                    } else {
                        auto [index, flavor_det] = fb.sample_discrete(
                            flavor_random_acc,
                            static_cast<me_int_t>(_diff_xs.pid_options().size())
                        );
                        result.flavor_id() = index;
                        weights_after_cuts.push_back(flavor_det);
                    }
                },
                [&](auto discrete_after) {
                    ValueVec discrete_condition;
                    using TDiscrete = std::decay_t<decltype(discrete_after)>;
                    if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                        if (flow_conditions.size() == 1) {
                            discrete_condition.push_back(flow_conditions.at(0));
                        } else if (flow_conditions.size() > 1) {
                            discrete_condition.push_back(fb.cat(flow_conditions));
                        }
                    }
                    if (args.has_pdf_prior) {
                        discrete_condition.push_back(result.pdf_prior());
                    }
                    auto discrete_result = discrete_after.build_forward(
                        fb, {flavor_random_acc}, discrete_condition
                    );
                    result.flavor_id() = discrete_result.at(0);
                    if (_flags & exclude_adaptive_and_chan_weight) {
                        extra_weights_after_cuts.push_back(discrete_result["det"]);
                    } else {
                        weights_after_cuts.push_back(discrete_result["det"]);
                    }
                    auto ones = fb.full({1., args.batch_size});
                    adaptive_probs.push_back(fb.batch_scatter(
                        result.indices_acc(), ones, discrete_result["det"]
                    ));
                }
            },
            _discrete_after
        );
    }
    result.weight_after_cuts() = weights_after_cuts.size() == 0
        ? fb.full({1., batch_size_acc})
        : fb.product(weights_after_cuts);
    result.adaptive_prob() = adaptive_probs.size() == 0
        ? fb.full({1., args.batch_size})
        : fb.product(adaptive_probs);
    return result;
}

Value Integrand::scatter_or_drop(
    FunctionBuilder& fb, ChannelResult& result, Value default_value, Value value
) const {
    if (_flags & drop_cuts_and_rescale) {
        return value;
    } else {
        return fb.batch_scatter(result.indices_acc(), default_value, value);
    }
}

Value Integrand::optional_cut(
    FunctionBuilder& fb, ChannelResult& result, Value value
) const {
    if (_flags & drop_cuts_and_rescale) {
        return fb.batch_gather(result.indices_acc(), value);
    } else {
        return value;
    }
}

NamedVector<Value> Integrand::build_common_part(
    FunctionBuilder& fb,
    const Integrand::ChannelArgs& args,
    Integrand::ChannelResult& result
) const {
    // if _prop_chan_weights is given, compute channel weight based
    // on denominators of propagators
    Value chan_weights_acc;
    std::size_t channel_count = _chan_weight_remap.size() > 0 ? _remapped_chan_count
        : _subchan_weights
        ? _subchan_weights->channel_count()
        : _diff_xs.matrix_element().diagram_count();
    if (channel_count > 1 && _prop_chan_weights) {
        chan_weights_acc =
            _prop_chan_weights->build_function(fb, {result.momenta_acc()}).at(0);
    }

    // evaluate differential cross section
    ValueVec xs_args{
        result.momenta_acc(),
        _flavor_remap.size() > 0
            ? fb.gather_int(result.flavor_id(), _flavor_remap)
            : result.flavor_id(),
    };
    for (std::size_t i = 0; i < 2; ++i) {
        xs_args.push_back(result.x_acc(i));
    }
    xs_args.push_back(result.flavor_id());
    if (args.has_mirror) {
        xs_args.push_back(result.mirror_id());
    }
    if (args.has_pdf_prior) {
        for (std::size_t i = 0; i < 2; ++i) {
            if (_diff_xs.has_pdf(i)) {
                xs_args.push_back(result.pdf_cache(i));
            }
        }
        xs_args.push_back(result.scale_cache());
    }
    auto dxs_vec = _diff_xs.build_function(fb, xs_args);
    auto diff_xs_acc = dxs_vec.at(0);
    if (_flavor_factors.size() > 0) {
        diff_xs_acc =
            fb.mul(diff_xs_acc, fb.gather(result.flavor_id(), _flavor_factors));
    }
    ValueVec weights_after_cuts{result.weight_after_cuts(), diff_xs_acc};
    ValueVec extra_weights_after_cuts;
    if (result.extra_weight_after_cuts()) {
        extra_weights_after_cuts.push_back(result.extra_weight_after_cuts());
    }
    if (!_prop_chan_weights) {
        chan_weights_acc = dxs_vec.at(1);
    }
    if (channel_count > 1 && _subchan_weights) {
        chan_weights_acc =
            _subchan_weights
                ->build_function(fb, {result.momenta_acc(), chan_weights_acc})
                .at(0);
    }
    if (_chan_weight_remap.size() > 0) {
        chan_weights_acc = fb.collect_channel_weights(
            chan_weights_acc, _chan_weight_remap, _remapped_chan_count
        );
    }

    // if given, apply channel weight network
    auto prior_chan_weights_acc = chan_weights_acc;
    if (_chan_weight_net) {
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc =
            preproc
                .build_function(
                    fb, {result.momenta_acc(), result.x_acc(0), result.x_acc(1)}
                )
                .at(0);
        chan_weights_acc =
            _chan_weight_net.value()
                .build_function(fb, {cw_preproc_acc, chan_weights_acc})
                .at(0);
    }

    // compute full phase-space weight
    if (channel_count > 1 && !(_flags & exclude_adaptive_and_chan_weight)) {
        Value chan_index_acc =
            fb.batch_gather(result.indices_acc(), result.chan_index());
        Value w = fb.gather(chan_index_acc, chan_weights_acc);
        if (_flags & exclude_adaptive_and_chan_weight) {
            extra_weights_after_cuts.push_back(w);
        } else {
            weights_after_cuts.push_back(w);
        }
    }
    auto weight = fb.mul(
        result.weight_before_cuts(),
        fb.batch_scatter(
            result.indices_acc(),
            result.weight_before_cuts(),
            fb.product(weights_after_cuts)
        )
    );

    // return results based on _flags
    NamedVector<Value> outputs;
    if (_flags & exclude_adaptive_and_chan_weight) {
        Value full_weight = weight;
        if (extra_weights_after_cuts.size() > 0) {
            full_weight = fb.mul(
                weight,
                fb.batch_scatter(
                    result.indices_acc(),
                    full_weight,
                    fb.product(extra_weights_after_cuts)
                )
            );
        }
        if (result.extra_weight_before_cuts()) {
            full_weight = fb.mul(full_weight, result.extra_weight_before_cuts());
        }
        outputs.push_back("full_weight", optional_cut(fb, result, full_weight));
        outputs.push_back("weight", optional_cut(fb, result, weight));
    } else {
        outputs.push_back("weight", optional_cut(fb, result, weight));
    }
    if (_flags & return_momenta) {
        outputs.push_back(
            "momenta",
            optional_cut(
                fb, result, args.has_mirror ? result.momenta_mirror() : result.momenta()
            )
        );
    }
    if (_flags & return_x1_x2) {
        if (_flags & drop_cuts_and_rescale) {
            outputs.push_back("x1", result.x_acc(0));
            outputs.push_back("x2", result.x_acc(1));
        } else {
            outputs.push_back("x1", result.x(0));
            outputs.push_back("x2", result.x(1));
        }
    }
    if (_flags & return_indices) {
        auto zeros = fb.full({static_cast<me_int_t>(0), args.batch_size});
        outputs.push_back(
            "color_index", scatter_or_drop(fb, result, zeros, dxs_vec.at(2))
        );
        outputs.push_back(
            "helicity_index", scatter_or_drop(fb, result, zeros, dxs_vec.at(3))
        );
        outputs.push_back(
            "diagram_index", scatter_or_drop(fb, result, zeros, dxs_vec.at(4))
        );
        outputs.push_back(
            "flavor_index", scatter_or_drop(fb, result, zeros, result.flavor_id())
        );
    }
    if (_flags & return_random) {
        outputs.push_back("random", optional_cut(fb, result, result.r()));
    }
    if (_flags & return_latent) {
        outputs.push_back("latent", optional_cut(fb, result, result.latent()));
        Value norm = _flags & drop_cuts_and_rescale
            ? fb.accept_norm(result.indices_acc(), result.adaptive_prob())
            : Value(1.);
        outputs.push_back(
            "adaptive_prob",
            fb.div(norm, optional_cut(fb, result, result.adaptive_prob()))
        );
    }
    if (_flags & return_channel) {
        outputs.push_back(
            "channel_index", optional_cut(fb, result, result.chan_index())
        );
    }
    if (_flags & return_chan_weights) {
        if (channel_count > 1) {
            auto cw_flat = fb.full(
                {1. / channel_count,
                 args.batch_size,
                 static_cast<me_int_t>(channel_count)}
            );
            outputs.push_back(
                "channel_weights",
                scatter_or_drop(fb, result, cw_flat, prior_chan_weights_acc)
            );
        } else {
            auto cw_flat = fb.full(
                {1. / channel_count,
                 fb.batch_size({outputs.at(0)}),
                 static_cast<me_int_t>(channel_count)}
            );
            outputs.push_back("channel_weights", cw_flat);
        }
    }
    if (_flags & return_cwnet_input) {
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc =
            preproc
                .build_function(
                    fb, {result.momenta_acc(), result.x_acc(0), result.x_acc(1)}
                )
                .at(0);
        auto zeros =
            fb.full({0., args.batch_size, static_cast<me_int_t>(preproc.output_dim())});
        outputs.push_back(
            "cwnet_inputs", scatter_or_drop(fb, result, zeros, cw_preproc_acc)
        );
    }
    if (_flags & return_discrete) {
        if (args.has_permutations &&
            !std::holds_alternative<std::monostate>(_discrete_before)) {
            // TODO: probably doesn't work for multichannel integrand
            outputs.push_back(
                "channel_index_in_group",
                optional_cut(fb, result, result.chan_index_in_group())
            );
        }
        if (args.has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_after)) {
            auto zeros = fb.full({static_cast<me_int_t>(0), args.batch_size});
            outputs.push_back(
                "discrete_flavor_index",
                scatter_or_drop(fb, result, zeros, result.flavor_id())
            );
        }
    }
    if (_flags & return_discrete_latent) {
        outputs.push_back(
            "channel_index_in_group",
            optional_cut(fb, result, result.chan_index_in_group())
        );
        if (args.has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_after)) {
            auto zeros = fb.full({static_cast<me_int_t>(0), args.batch_size});
            outputs.push_back(
                "discrete_flavor_index",
                scatter_or_drop(fb, result, zeros, result.flavor_id())
            );
            if (args.has_pdf_prior) {
                auto flav_count = _diff_xs.pid_options().size();
                auto norm = fb.full(
                    {1. / flav_count,
                     args.batch_size,
                     static_cast<me_int_t>(flav_count)}
                );
                outputs.push_back(
                    "pdf_prior", scatter_or_drop(fb, result, norm, result.pdf_prior())
                );
            }
        }
    }

    if (_flags & unweight) {
        Unweighter unweighter(return_types());
        NamedVector<Value> unweighter_args = outputs;
        unweighter_args.push_back("max_weight", args.max_weight);
        outputs = unweighter.build_function(fb, unweighter_args);
    }

    return outputs;
}

MultiChannelIntegrand::MultiChannelIntegrand(
    const std::vector<std::shared_ptr<Integrand>>& integrands, bool return_sizes
) :
    FunctionGenerator(
        "MultiChannelIntegrand",
        [&] {
            NamedVector<Type> arg_types{
                {"batch_sizes", multichannel_batch_size(integrands.size())}
            };
            if (integrands.at(0)->_flags & Integrand::unweight) {
                arg_types.push_back("max_weight", single_float);
            }
            return arg_types;
        }(),
        [&] {
            NamedVector<Type> ret_types = integrands.at(0)->return_types();
            if (return_sizes) {
                ret_types.push_back(
                    "return_batch_sizes", multichannel_batch_size(integrands.size())
                );
            }
            return ret_types;
        }()
    ),
    _integrands(integrands),
    _return_sizes(return_sizes) {
    auto& first_function = integrands.at(0);
    std::size_t arg_count = first_function->arg_types().size();
    std::size_t return_count = first_function->return_types().size();
    for (auto& integrand : integrands) {
        if (integrand->arg_types().size() != arg_count ||
            integrand->return_types().size() != return_count) {
            throw std::invalid_argument(
                "All integrands must have the same number of inputs and outputs"
            );
        }
    }
}

NamedVector<Value> MultiChannelIntegrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto& first_integrand = _integrands.at(0);
    bool has_multi_flavor = first_integrand->_diff_xs.pid_options().size() > 1;
    auto batch_sizes = args.at(0);
    auto all_batch_sizes = fb.unstack_sizes(batch_sizes);
    Integrand::ChannelArgs common_args{
        .has_permutations = first_integrand->_mapping.channel_count() > 1,
        .has_multi_flavor = has_multi_flavor,
        .has_mirror = first_integrand->_diff_xs.has_mirror(),
        .has_pdf_prior =
            (first_integrand->_pdfs.at(0) || first_integrand->_pdfs.at(1)) &&
            first_integrand->_energy_scale && has_multi_flavor,
    };
    if (first_integrand->_flags & Integrand::unweight) {
        common_args.max_weight = args.at(1);
    }
    std::vector<Integrand::ChannelResult> results;
    ValueVec ret_batch_sizes;
    for (std::size_t index = 0;
         auto [integrand, chan_size] : zip(_integrands, all_batch_sizes)) {
        fb.set_current_stream(index + 1);
        auto channel_args = common_args;
        channel_args.r = fb.random(chan_size, integrand->_random_dim);
        channel_args.batch_size = chan_size;
        channel_args.has_permutations = integrand->_mapping.channel_count() > 1;
        results.push_back(integrand->build_channel_part(fb, channel_args));
        if (_return_sizes) {
            ret_batch_sizes.push_back(fb.batch_size({results.back().indices_acc()}));
        }
        ++index;
    }
    fb.set_current_stream(0);

    Integrand::ChannelResult common_results;
    for (std::size_t i = 0; i < common_results.values.size(); ++i) {
        bool is_set = true;
        ValueVec values;
        values.reserve(results.size());
        for (auto& result : results) {
            auto& value = result.values[i];
            if (!value) {
                is_set = false;
                break;
            }
            values.push_back(value);
        }
        if (is_set) {
            auto [cat, cat_sizes] = fb.batch_cat(values);
            auto& value = common_results.values[i];
            if (&value == &common_results.indices_acc()) {
                value = fb.add_int(fb.offset_indices(batch_sizes, cat_sizes), cat);
            } else {
                value = cat;
            }
        }
    }
    common_args.batch_size = fb.batch_size({common_results.momenta()});
    auto output = _integrands.at(0)->build_common_part(fb, common_args, common_results);
    if (_return_sizes) {
        output.push_back(
            "return_batch_sizes",
            _integrands.at(0)->_flags & Integrand::drop_cuts_and_rescale
                ? fb.stack_sizes(ret_batch_sizes)
                : batch_sizes
        );
    }
    return output;
}

IntegrandProbability::IntegrandProbability(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandProbability",
        [&] {
            NamedVector<Type> arg_types{
                {"latent", batch_float_array(integrand._mapping.random_dim())},
                {"channel_index_in_group", batch_int}
            };
            auto flavor_count = integrand._diff_xs.pid_options().size();
            if (flavor_count > 1 &&
                !std::holds_alternative<std::monostate>(integrand._discrete_after)) {
                arg_types.push_back("discrete_flavor_index", batch_int);
                if ((integrand._pdfs.at(0) || integrand._pdfs.at(1)) &&
                    integrand._energy_scale) {
                    arg_types.push_back("pdf_prior", batch_float_array(flavor_count));
                }
            }
            return arg_types;
        }(),
        {{"prob", batch_float}}
    ),
    _adaptive_map(integrand._adaptive_map),
    _discrete_before(integrand._discrete_before),
    _discrete_after(integrand._discrete_after),
    _permutation_count(integrand._mapping.channel_count()),
    _flavor_count(integrand._diff_xs.pid_options().size()),
    _has_pdf_prior(
        (integrand._pdfs.at(0) || integrand._pdfs.at(1)) && integrand._energy_scale
    ) {}

NamedVector<Value> IntegrandProbability::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec probs, flow_conditions;
    if (_permutation_count > 1) {
        auto chan_index = args.at(1);
        /*flow_conditions.push_back(
            fb.one_hot(chan_index, static_cast<me_int_t>(_permutation_count))
        );*/
        std::visit(
            Overloaded{
                [](std::monostate) {},
                [&](auto discrete_before) {
                    auto discrete_result =
                        discrete_before.build_inverse(fb, {chan_index}, {});
                    probs.push_back(discrete_result["det"]);
                }
            },
            _discrete_before
        );
    }

    auto latent = args.at(0);
    std::visit(
        Overloaded{
            [&](std::monostate) {},
            [&](auto& admap) {
                ValueVec cond;
                using TAdaptive = std::decay_t<decltype(admap)>;
                if constexpr (std::is_same_v<TAdaptive, Flow>) {
                    if (flow_conditions.size() == 1) {
                        cond.push_back(flow_conditions.at(0));
                    } else if (flow_conditions.size() > 1) {
                        cond.push_back(fb.cat(flow_conditions));
                    }
                }
                auto admap_result = admap.build_inverse(fb, {latent}, cond);
                probs.push_back(admap_result["det"]);
                flow_conditions.push_back(latent);
            }
        },
        _adaptive_map
    );

    std::size_t arg_index = 2;
    if (_flavor_count > 1) {
        std::visit(
            Overloaded{
                [&](std::monostate) {},
                [&](auto discrete_after) {
                    auto flavor = args.at(arg_index);
                    ++arg_index;
                    ValueVec discrete_condition;
                    using TDiscrete = std::decay_t<decltype(discrete_after)>;
                    if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                        if (flow_conditions.size() == 1) {
                            discrete_condition.push_back(flow_conditions.at(0));
                        } else if (flow_conditions.size() > 1) {
                            discrete_condition.push_back(fb.cat(flow_conditions));
                        }
                    }
                    if (_has_pdf_prior) {
                        auto pdf_prior = args.at(arg_index);
                        ++arg_index;
                        discrete_condition.push_back(pdf_prior);
                    }
                    auto discrete_result =
                        discrete_after.build_inverse(fb, {flavor}, discrete_condition);
                    probs.push_back(discrete_result["det"]);
                }
            },
            _discrete_after
        );
    }

    return {{"prob", fb.product(probs)}};
}
