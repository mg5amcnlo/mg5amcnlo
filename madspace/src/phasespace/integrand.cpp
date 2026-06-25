#include "madspace/phasespace/integrand.hpp"

#include "madspace/util.hpp"

#include <set>

using namespace madspace;

static const BatchSize acc_batch_size("acc_batch_size");

Integrand::Integrand(
    const PhaseSpaceMapping& mapping,
    const DifferentialCrossSection& diff_xs,
    const AdaptiveMapping& adaptive_map,
    const AdaptiveDiscrete& discrete_before,
    const AdaptiveDiscrete& discrete_after,
    const std::optional<PdfGrid>& pdf_grid,
    const std::optional<RunningCoupling>& running_coupling,
    const std::optional<EnergyScale>& energy_scale,
    const std::optional<PropagatorChannelWeights>& prop_chan_weights,
    const std::optional<SubchannelWeights>& subchan_weights,
    const std::optional<ChannelWeightNetwork>& chan_weight_net,
    const std::vector<me_int_t>& chan_weight_remap,
    std::size_t remapped_chan_count,
    bool madnis_training,
    bool drop_cuts_and_rescale,
    bool partial_weights,
    const std::vector<std::size_t>& channel_indices,
    const std::vector<std::size_t>& active_flavors,
    const std::vector<std::size_t>& flavor_remap,
    const std::vector<double>& flavor_factors
) :
    FunctionGenerator(
        "Integrand",
        {{"batch_size", Type({batch_size})}},
        [&] {
            NamedVector<Type> ret_types;
            auto flav_count = diff_xs.pid_options().size();
            if (madnis_training) {
                if (std::holds_alternative<std::monostate>(adaptive_map)) {
                    throw std::invalid_argument(
                        "madnis_training requires an adaptive mapping"
                    );
                }
                ret_types.push_back("full_weight", batch_float);
                ret_types.push_back("weight", batch_float);
                ret_types.push_back("latent", batch_float_array(mapping.random_dim()));
                ret_types.push_back("adaptive_prob", batch_float);
                ret_types.push_back("channel_index", batch_int);
                ret_types.push_back(
                    "channel_weights",
                    batch_float_array(
                        chan_weight_remap.size() > 0 ? remapped_chan_count
                            : subchan_weights        ? subchan_weights->channel_count()
                                              : diff_xs.matrix_element().diagram_count()
                    )
                );
                ret_types.push_back(
                    "cwnet_input",
                    batch_float_array(
                        chan_weight_net.value().preprocessing().output_dim()
                    )
                );
                ret_types.push_back("channel_index_in_group", batch_int);
                if (flav_count > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_after)) {
                    ret_types.push_back("discrete_flavor_index", batch_int);
                    if (pdf_grid && energy_scale) {
                        ret_types.push_back("pdf_prior", batch_float_array(flav_count));
                    }
                }
            } else {
                ret_types.push_back("weight", batch_float);
                ret_types.push_back(
                    "momenta", batch_four_vec_array(mapping.particle_count())
                );
                ret_types.push_back("color_index", batch_int);
                ret_types.push_back("helicity_index", batch_int);
                ret_types.push_back("diagram_index", batch_int);
                ret_types.push_back("flavor_index", batch_int);
                ret_types.push_back("ren_scale", batch_float);
                ret_types.push_back("alpha_qcd", batch_float);
                if (partial_weights) {
                    if (diff_xs.has_pdf(0)) {
                        ret_types.push_back("x1", batch_float);
                        ret_types.push_back("fact_scale1", batch_float);
                    }
                    if (diff_xs.has_pdf(1)) {
                        ret_types.push_back("x2", batch_float);
                        ret_types.push_back("fact_scale2", batch_float);
                    }
                    if (diff_xs.has_pdf(0) || diff_xs.has_pdf(1)) {
                        ret_types.push_back("partial_weight_product", batch_float);
                    }
                }
                ret_types.push_back("random", batch_float_array(mapping.random_dim()));
                if (mapping.channel_count() > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_before)) {
                    ret_types.push_back("channel_index_in_group", batch_int);
                }
                if (flav_count > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_after)) {
                    ret_types.push_back("discrete_flavor_index", batch_int);
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
    _running_coupling(running_coupling),
    _energy_scale(energy_scale),
    _prop_chan_weights(prop_chan_weights),
    _subchan_weights(subchan_weights),
    _chan_weight_net(chan_weight_net),
    _chan_weight_remap(chan_weight_remap),
    _remapped_chan_count(remapped_chan_count),
    _madnis_training(madnis_training),
    _drop_cuts_and_rescale(drop_cuts_and_rescale),
    _partial_weights(partial_weights),
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

    _channel_part_ret_types = compute_channel_part_ret_types();
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
    auto channel_out = build_channel_part(fb, args);
    return build_common_part(fb, channel_out);
}

NamedVector<Type> Integrand::compute_channel_part_ret_types() const {
    auto acc_float = Type(DataType::dt_float, acc_batch_size, {});
    auto acc_int = Type(DataType::dt_int, acc_batch_size, {});
    auto acc_float_array = [](int n) {
        return Type(DataType::dt_float, acc_batch_size, {n});
    };
    auto acc_four_vec_array = [](int n) {
        return Type(DataType::dt_float, acc_batch_size, {n, 4});
    };

    bool has_multi_flavor = _diff_xs.pid_options().size() > 1;
    bool has_mirror = _diff_xs.has_mirror();
    int particle_count = static_cast<int>(_mapping.particle_count());
    int random_dim = static_cast<int>(_mapping.random_dim());

    NamedVector<Type> ret;

    // outputs before cuts
    ret.push_back("r", batch_float_array(random_dim));
    ret.push_back("latent", batch_float_array(random_dim));
    ret.push_back("weight_before_cuts", batch_float);
    ret.push_back("adaptive_prob", batch_float);
    ret.push_back("chan_index", batch_int);
    ret.push_back("chan_index_in_group", batch_int);
    if (!_madnis_training) {
        ret.push_back("momenta", batch_four_vec_array(particle_count));
        if (has_mirror) {
            ret.push_back("momenta_mirror", batch_four_vec_array(particle_count));
        }
    }
    if (has_mirror) {
        ret.push_back("mirror_id", batch_int);
    }
    if (_madnis_training) {
        ret.push_back("extra_weight_before_cuts", batch_float);
    }

    // outputs after cuts
    ret.push_back("indices_acc", Type(DataType::dt_int, acc_batch_size, {}));
    ret.push_back("momenta_acc", acc_four_vec_array(particle_count));
    ret.push_back("x1_acc", acc_float);
    ret.push_back("x2_acc", acc_float);
    ret.push_back("flavor_id", acc_int);
    ret.push_back("weight_after_cuts", acc_float);
    if (_madnis_training && !std::holds_alternative<std::monostate>(_discrete_after)) {
        ret.push_back("extra_weight_after_cuts", acc_float);
    }
    ret.push_back("ren_scale", acc_float);
    if ((_pdfs.at(0) || _pdfs.at(1)) && _energy_scale) {
        auto flav_count = static_cast<int>(_diff_xs.pid_options().size());
        if (has_multi_flavor) {
            ret.push_back("pdf_prior", acc_float_array(flav_count));
        }
        for (std::size_t i = 0; i < 2; ++i) {
            if (_diff_xs.has_pdf(i)) {
                ret.push_back(std::format("pdf{}", i + 1), acc_float);
                ret.push_back(std::format("fact_scale{}", i + 1), acc_float);
            }
        }
    }

    return ret;
}

NamedVector<Value> Integrand::build_channel_part(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    bool has_multi_flavor = _diff_xs.pid_options().size() > 1;
    bool has_permutations = _mapping.channel_count() > 1;
    bool has_mirror = _diff_xs.has_mirror();
    auto batch_size_val = args.at("batch_size");

    Value r = fb.random(batch_size_val, _random_dim);
    ValueVec weights_before_cuts, weights_after_cuts, adaptive_probs;
    ValueVec extra_weights_before_cuts;

    // Split off discrete random numbers
    Value chan_random, flavor_random, mirror_random;
    if (has_permutations) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        chan_random = r_val;
    }
    if (has_multi_flavor) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        flavor_random = r_val;
    }
    if (has_mirror) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        mirror_random = r_val;
    }

    // Sample channel permutation
    Value chan_index, chan_index_in_group;
    ValueVec mapping_conditions, flow_conditions;
    if (has_permutations) {
        me_int_t opt_count = _channel_indices.size();
        std::visit(
            Overloaded{
                [&](std::monostate) {
                    auto [index, chan_det] = fb.sample_discrete(chan_random, opt_count);
                    chan_index_in_group = index;
                    weights_before_cuts.push_back(chan_det);
                },
                [&](const auto& discrete_before) {
                    auto discrete_result =
                        discrete_before.build_forward(fb, {chan_random}, {});
                    chan_index_in_group = discrete_result.at(0);
                    if (_madnis_training) {
                        extra_weights_before_cuts.push_back(discrete_result["det"]);
                    } else {
                        weights_before_cuts.push_back(discrete_result["det"]);
                    }
                    adaptive_probs.push_back(discrete_result["det"]);
                }
            },
            _discrete_before
        );
        chan_index = fb.gather_int(chan_index_in_group, _channel_indices);
        mapping_conditions.push_back(chan_index_in_group);
        // flow_conditions.push_back(fb.one_hot(chan_index_in_group, opt_count));
    } else {
        chan_index =
            fb.full({static_cast<me_int_t>(_channel_indices.at(0)), batch_size_val});
        chan_index_in_group = fb.full({static_cast<me_int_t>(0), batch_size_val});
    }

    // Apply adaptive map (VEGAS or MadNIS flow)
    Value latent = r;
    std::visit(
        Overloaded{
            [&](std::monostate) {},
            [&](const auto& admap) {
                ValueVec cond;
                using TAdaptive = std::decay_t<decltype(admap)>;
                if constexpr (std::is_same_v<TAdaptive, Flow>) {
                    if (flow_conditions.size() == 1) {
                        cond.push_back(flow_conditions.at(0));
                    } else if (flow_conditions.size() > 1) {
                        cond.push_back(fb.cat(flow_conditions));
                    }
                }
                auto admap_result = admap.build_forward(fb, {r}, cond);
                latent = admap_result["data"];
                adaptive_probs.push_back(admap_result["det"]);
                if (_madnis_training) {
                    extra_weights_before_cuts.push_back(admap_result["det"]);
                } else {
                    weights_before_cuts.push_back(admap_result["det"]);
                }
                flow_conditions.push_back(latent);
            }
        },
        _adaptive_map
    );

    // Apply phase space mapping
    auto mapping_result = _mapping.build_forward(fb, {latent}, mapping_conditions);
    weights_before_cuts.push_back(mapping_result["det"]);
    Value momenta = mapping_result["momenta"];
    Value x0 = mapping_result["x1"];
    Value x1 = mapping_result["x2"];

    Value momenta_mirror, mirror_id;
    if (has_mirror) {
        auto [index, mirror_det] =
            fb.sample_discrete(mirror_random, static_cast<me_int_t>(2));
        mirror_id = index;
        momenta_mirror = fb.mirror_momenta(momenta, mirror_id);
        weights_before_cuts.push_back(mirror_det);
    }

    // Filter events that pass cuts
    Value weight_before_cuts = fb.product(weights_before_cuts);
    Value extra_weight_before_cuts;
    if (!extra_weights_before_cuts.empty()) {
        extra_weight_before_cuts = fb.product(extra_weights_before_cuts);
    }
    Value indices_acc = fb.nonzero(weight_before_cuts);
    Value momenta_acc = fb.batch_gather(indices_acc, momenta);
    std::array<Value, 2> x_acc{
        {fb.batch_gather(indices_acc, x0), fb.batch_gather(indices_acc, x1)}
    };
    for (auto& cond : flow_conditions) {
        cond = fb.batch_gather(indices_acc, cond);
    }

    // Evaluate PDF prior for adaptive flavor sampling
    Value pdf_prior;
    auto scales = _energy_scale.value().build_function(fb, {momenta_acc});
    std::array<Value, 2> pdf_results;
    bool has_pdf_prior = false;
    if ((_pdfs.at(0) || _pdfs.at(1)) && _energy_scale) {
        ValueVec pdf_priors;
        for (std::size_t i = 0; i < 2; ++i) {
            if (_diff_xs.has_pdf(i)) {
                auto pdf =
                    _pdfs.at(i)
                        .value()
                        .build_function(fb, {x_acc.at(i), scales.at(i + 1)})
                        .at(0);
                pdf_results.at(i) = pdf;
                pdf_priors.push_back(fb.select(pdf, _pdf_indices.at(i)));
            }
        }
        if (has_multi_flavor) {
            pdf_prior = fb.product(pdf_priors);
            if (_active_flavors_mask.size() > 0) {
                pdf_prior = fb.mul(pdf_prior, _active_flavors_mask);
            }
            has_pdf_prior = true;
        }
    }

    // Flavor sampling
    auto batch_size_acc = fb.batch_size({momenta_acc});
    Value flavor_id = fb.full({static_cast<me_int_t>(0), batch_size_acc});
    Value extra_weight_after_cuts;
    if (has_multi_flavor) {
        auto flavor_random_acc = fb.batch_gather(indices_acc, flavor_random);
        std::visit(
            Overloaded{
                [&](std::monostate) {
                    if (has_pdf_prior) {
                        auto [index, flavor_det] =
                            fb.sample_discrete_probs(flavor_random_acc, pdf_prior);
                        flavor_id = index;
                        weights_after_cuts.push_back(flavor_det);
                    } else {
                        auto [index, flavor_det] = fb.sample_discrete(
                            flavor_random_acc,
                            static_cast<me_int_t>(_diff_xs.pid_options().size())
                        );
                        flavor_id = index;
                        weights_after_cuts.push_back(flavor_det);
                    }
                },
                [&](const auto& discrete_after) {
                    ValueVec discrete_condition;
                    using TDiscrete = std::decay_t<decltype(discrete_after)>;
                    if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                        if (flow_conditions.size() == 1) {
                            discrete_condition.push_back(flow_conditions.at(0));
                        } else if (flow_conditions.size() > 1) {
                            discrete_condition.push_back(fb.cat(flow_conditions));
                        }
                    }
                    if (has_pdf_prior) {
                        discrete_condition.push_back(pdf_prior);
                    }
                    auto discrete_result = discrete_after.build_forward(
                        fb, {flavor_random_acc}, discrete_condition
                    );
                    flavor_id = discrete_result.at(0);
                    if (_madnis_training) {
                        extra_weight_after_cuts = discrete_result["det"];
                    } else {
                        weights_after_cuts.push_back(discrete_result["det"]);
                    }
                    auto ones = fb.full({1., batch_size_val});
                    adaptive_probs.push_back(
                        fb.batch_scatter(indices_acc, ones, discrete_result["det"])
                    );
                }
            },
            _discrete_after
        );
        for (auto [pdf, indices] : zip(pdf_results, _pdf_indices)) {
            if (pdf) {
                pdf = fb.gather(fb.gather_int(flavor_id, indices), pdf);
            }
        }
    } else {
        for (auto& pdf : pdf_results) {
            if (pdf) {
                pdf = fb.squeeze(pdf);
            }
        }
    }

    Value weight_after_cuts = weights_after_cuts.empty()
        ? fb.full({1., batch_size_acc})
        : fb.product(weights_after_cuts);
    Value adaptive_prob = adaptive_probs.empty()
        ? fb.full({1., batch_size_val})
        : fb.product(adaptive_probs);

    NamedVector<Value> out;

    // outputs before cuts
    out.push_back("r", r);
    out.push_back("latent", latent);
    out.push_back("weight_before_cuts", weight_before_cuts);
    out.push_back("adaptive_prob", adaptive_prob);
    out.push_back("chan_index", chan_index);
    out.push_back("chan_index_in_group", chan_index_in_group);
    if (!_madnis_training) {
        out.push_back("momenta", momenta);
        if (has_mirror) {
            out.push_back("momenta_mirror", momenta_mirror);
        }
    }
    if (has_mirror) {
        out.push_back("mirror_id", mirror_id);
    }
    if (_madnis_training) {
        out.push_back("extra_weight_before_cuts", extra_weight_before_cuts);
    }

    // outputs after cuts
    out.push_back("indices_acc", indices_acc);
    out.push_back("momenta_acc", momenta_acc);
    out.push_back("x1_acc", x_acc.at(0));
    out.push_back("x2_acc", x_acc.at(1));
    out.push_back("flavor_id", flavor_id);
    out.push_back("weight_after_cuts", weight_after_cuts);
    if (_madnis_training && extra_weight_after_cuts) {
        out.push_back("extra_weight_after_cuts", extra_weight_after_cuts);
    }
    out.push_back("ren_scale", scales.at(0));
    if (has_pdf_prior) {
        out.push_back("pdf_prior", pdf_prior);
    }
    if (_diff_xs.has_pdf(0)) {
        out.push_back("pdf1", pdf_results.at(0));
        out.push_back("fact_scale1", scales.at(1));
    }
    if (_diff_xs.has_pdf(1)) {
        out.push_back("pdf2", pdf_results.at(1));
        out.push_back("fact_scale2", scales.at(2));
    }

    return out;
}

NamedVector<Value> Integrand::build_common_part(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    bool has_multi_flavor = _diff_xs.pid_options().size() > 1;
    bool has_permutations = _mapping.channel_count() > 1;
    bool has_mirror = _diff_xs.has_mirror();
    bool has_pdf_prior =
        (_pdfs.at(0) || _pdfs.at(1)) && _energy_scale && has_multi_flavor;

    auto indices_acc = args.at("indices_acc");
    auto momenta_acc = args.at("momenta_acc");
    auto x1_acc = args.at("x1_acc");
    auto x2_acc = args.at("x2_acc");
    auto flavor_id = args.at("flavor_id");
    auto batch_size_val = fb.batch_size({args.at("weight_before_cuts")});

    auto scatter_or_drop = [&](Value default_value, Value value) -> Value {
        if (_drop_cuts_and_rescale) {
            return value;
        }
        return fb.batch_scatter(indices_acc, default_value, value);
    };
    auto optional_cut = [&](Value value) -> Value {
        if (_drop_cuts_and_rescale) {
            return fb.batch_gather(indices_acc, value);
        }
        return value;
    };

    // Channel weight computation
    std::size_t channel_count = _chan_weight_remap.size() > 0 ? _remapped_chan_count
        : _subchan_weights
        ? _subchan_weights->channel_count()
        : _diff_xs.matrix_element().diagram_count();
    Value chan_weights_acc;
    if (channel_count > 1 && _prop_chan_weights) {
        chan_weights_acc = _prop_chan_weights->build_function(fb, {momenta_acc}).at(0);
    }

    // Compute running coupling
    Value alpha_qcd_acc =
        _running_coupling.value().build_function(fb, {args.at("ren_scale")}).at(0);

    // Evaluate differential cross section
    ValueVec xs_args{
        momenta_acc,
        _flavor_remap.size() > 0 ? fb.gather_int(flavor_id, _flavor_remap) : flavor_id,
    };
    xs_args.push_back(x1_acc);
    xs_args.push_back(x2_acc);
    xs_args.push_back(flavor_id);
    if (has_mirror) {
        xs_args.push_back(args.at("mirror_id"));
    }
    if (_diff_xs.has_pdf(0)) {
        xs_args.push_back(args.at("pdf1"));
    }
    if (_diff_xs.has_pdf(1)) {
        xs_args.push_back(args.at("pdf2"));
    }
    xs_args.push_back(alpha_qcd_acc);
    auto dxs_vec = _diff_xs.build_function(fb, xs_args);
    auto diff_xs_acc = dxs_vec.at(0);
    if (_flavor_factors.size() > 0) {
        diff_xs_acc = fb.mul(diff_xs_acc, fb.gather(flavor_id, _flavor_factors));
    }
    ValueVec weights_after_cuts{args.at("weight_after_cuts"), diff_xs_acc};
    ValueVec extra_weights_after_cuts;
    if (args.index_map().contains("extra_weight_after_cuts")) {
        extra_weights_after_cuts.push_back(args.at("extra_weight_after_cuts"));
    }
    if (!_prop_chan_weights) {
        chan_weights_acc = dxs_vec.at(1);
    }
    if (channel_count > 1 && _subchan_weights) {
        chan_weights_acc =
            _subchan_weights->build_function(fb, {momenta_acc, chan_weights_acc}).at(0);
    }
    if (_chan_weight_remap.size() > 0) {
        chan_weights_acc = fb.collect_channel_weights(
            chan_weights_acc, _chan_weight_remap, _remapped_chan_count
        );
    }

    // Apply channel weight network
    auto prior_chan_weights_acc = chan_weights_acc;
    if (_chan_weight_net) {
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc =
            preproc.build_function(fb, {momenta_acc, x1_acc, x2_acc}).at(0);
        chan_weights_acc =
            _chan_weight_net.value()
                .build_function(fb, {cw_preproc_acc, chan_weights_acc})
                .at(0);
    }

    // Compute full phase-space weight
    if (channel_count > 1 && !_madnis_training) {
        Value chan_index_acc = fb.batch_gather(indices_acc, args.at("chan_index"));
        weights_after_cuts.push_back(fb.gather(chan_index_acc, chan_weights_acc));
    }
    auto weight = fb.mul(
        args.at("weight_before_cuts"),
        fb.batch_scatter(
            indices_acc, args.at("weight_before_cuts"), fb.product(weights_after_cuts)
        )
    );

    NamedVector<Value> outputs;
    if (_madnis_training) {
        Value full_weight = weight;
        if (!extra_weights_after_cuts.empty()) {
            full_weight = fb.mul(
                weight,
                fb.batch_scatter(
                    indices_acc, full_weight, fb.product(extra_weights_after_cuts)
                )
            );
        }
        if (args.index_map().contains("extra_weight_before_cuts")) {
            full_weight = fb.mul(full_weight, args.at("extra_weight_before_cuts"));
        }
        outputs.push_back("full_weight", optional_cut(full_weight));
        outputs.push_back("weight", optional_cut(weight));
        outputs.push_back("latent", optional_cut(args.at("latent")));
        Value norm = _drop_cuts_and_rescale
            ? fb.accept_norm(indices_acc, args.at("adaptive_prob"))
            : Value(1.);
        outputs.push_back(
            "adaptive_prob", fb.div(norm, optional_cut(args.at("adaptive_prob")))
        );
        outputs.push_back("channel_index", optional_cut(args.at("chan_index")));
        if (channel_count > 1) {
            auto cw_flat = fb.full(
                {1. / channel_count,
                 batch_size_val,
                 static_cast<me_int_t>(channel_count)}
            );
            outputs.push_back(
                "channel_weights", scatter_or_drop(cw_flat, prior_chan_weights_acc)
            );
        } else {
            outputs.push_back(
                "channel_weights",
                fb.full(
                    {1. / channel_count,
                     fb.batch_size({outputs.at(0)}),
                     static_cast<me_int_t>(channel_count)}
                )
            );
        }
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc =
            preproc.build_function(fb, {momenta_acc, x1_acc, x2_acc}).at(0);
        outputs.push_back(
            "cwnet_input",
            scatter_or_drop(
                fb.full(
                    {0., batch_size_val, static_cast<me_int_t>(preproc.output_dim())}
                ),
                cw_preproc_acc
            )
        );
        outputs.push_back(
            "channel_index_in_group", optional_cut(args.at("chan_index_in_group"))
        );
        if (has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_after)) {
            auto zeros = fb.full({static_cast<me_int_t>(0), batch_size_val});
            outputs.push_back(
                "discrete_flavor_index", scatter_or_drop(zeros, flavor_id)
            );
            if (has_pdf_prior) {
                auto flav_count = static_cast<me_int_t>(_diff_xs.pid_options().size());
                outputs.push_back(
                    "pdf_prior",
                    scatter_or_drop(
                        fb.full({1. / flav_count, batch_size_val, flav_count}),
                        args.at("pdf_prior")
                    )
                );
            }
        }
    } else {
        outputs.push_back("weight", optional_cut(weight));
        outputs.push_back(
            "momenta",
            optional_cut(has_mirror ? args.at("momenta_mirror") : args.at("momenta"))
        );
        auto zeros_int = fb.full({static_cast<me_int_t>(0), batch_size_val});
        auto zeros_float = fb.full({0., batch_size_val});
        outputs.push_back("color_index", scatter_or_drop(zeros_int, dxs_vec.at(2)));
        outputs.push_back("helicity_index", scatter_or_drop(zeros_int, dxs_vec.at(3)));
        outputs.push_back("diagram_index", scatter_or_drop(zeros_int, dxs_vec.at(4)));
        outputs.push_back("flavor_index", scatter_or_drop(zeros_int, flavor_id));

        outputs.push_back(
            "ren_scale", scatter_or_drop(zeros_float, args.at("ren_scale"))
        );
        outputs.push_back("alpha_qcd", scatter_or_drop(zeros_float, alpha_qcd_acc));
        if (_partial_weights) {
            ValueVec pdf_vals;
            if (_diff_xs.has_pdf(0)) {
                outputs.push_back(
                    "x1", scatter_or_drop(zeros_float, args.at("x1_acc"))
                );
                outputs.push_back(
                    "fact_scale1", scatter_or_drop(zeros_float, args.at("fact_scale1"))
                );
                pdf_vals.push_back(args.at("pdf1"));
            }
            if (_diff_xs.has_pdf(1)) {
                outputs.push_back(
                    "x2", scatter_or_drop(zeros_float, args.at("x2_acc"))
                );
                outputs.push_back(
                    "fact_scale2", scatter_or_drop(zeros_float, args.at("fact_scale2"))
                );
                pdf_vals.push_back(args.at("pdf2"));
            }
            if (_diff_xs.has_pdf(0) || _diff_xs.has_pdf(1)) {
                outputs.push_back(
                    "partial_weight_product",
                    scatter_or_drop(zeros_float, fb.product(pdf_vals))
                );
            }
        }
        outputs.push_back("random", optional_cut(args.at("r")));
        if (has_permutations &&
            !std::holds_alternative<std::monostate>(_discrete_before)) {
            outputs.push_back(
                "channel_index_in_group", optional_cut(args.at("chan_index_in_group"))
            );
        }
        if (has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_after)) {
            outputs.push_back(
                "discrete_flavor_index", scatter_or_drop(zeros_int, flavor_id)
            );
        }
    }

    return outputs;
}

IntegrandChannelPart::IntegrandChannelPart(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandChannelPart",
        {{"batch_size", Type({batch_size})}},
        integrand._channel_part_ret_types
    ),
    _integrand(integrand) {}

NamedVector<Value> IntegrandChannelPart::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return _integrand.build_channel_part(fb, args);
}

IntegrandCommonPart::IntegrandCommonPart(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandCommonPart",
        integrand._channel_part_ret_types,
        integrand.return_types()
    ),
    _integrand(integrand) {}

NamedVector<Value> IntegrandCommonPart::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return _integrand.build_common_part(fb, args);
}

IntegrandConcatenator::IntegrandConcatenator(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandConcatenator",
        [&] {
            NamedVector<Type> arg_types;
            arg_types.reserve(2 * integrand._channel_part_ret_types.size());
            auto keys = integrand._channel_part_ret_types.keys();
            for (auto [key, type] : zip(keys, integrand._channel_part_ret_types)) {
                arg_types.push_back(std::format("arg1_{}", key), type);
            }
            for (auto [key, type] : zip(keys, integrand._channel_part_ret_types)) {
                arg_types.push_back(std::format("arg2_{}", key), type);
            }
            return arg_types;
        }(),
        integrand._channel_part_ret_types
    ),
    _integrand(integrand) {}

NamedVector<Value> IntegrandConcatenator::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    // Combine per-channel results into a single NamedVector
    ValueVec outputs;
    auto keys = return_types().keys();
    std::size_t half_count = args.size() / 2;
    Value batch_sizes = fb.stack_sizes(
        {fb.batch_size({args.at("arg1_momenta")}),
         fb.batch_size({args.at("arg2_momenta")})}
    );
    for (auto [key, val1, val2] :
         zip(std::span(keys.begin(), keys.begin() + half_count),
             std::span(args.begin(), args.begin() + half_count),
             std::span(args.begin() + half_count, args.end()))) {
        auto [cat, cat_sizes] = fb.batch_cat({val1, val2});
        if (key == "indices_acc" && !_integrand._drop_cuts_and_rescale) {
            outputs.push_back(
                fb.add_int(fb.offset_indices(batch_sizes, cat_sizes), cat)
            );
        } else {
            outputs.push_back(cat);
        }
    }
    return {keys, outputs};
}

MultiChannelIntegrand::MultiChannelIntegrand(
    const std::vector<std::shared_ptr<Integrand>>& integrands, bool return_sizes
) :
    FunctionGenerator(
        "MultiChannelIntegrand",
        {{"batch_sizes", multichannel_batch_size(integrands.size())}},
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
    auto batch_sizes = args.at(0);
    auto all_batch_sizes = fb.unstack_sizes(batch_sizes);

    std::vector<NamedVector<Value>> results;
    ValueVec ret_batch_sizes;

    for (std::size_t index = 0;
         auto [integrand, chan_size] : zip(_integrands, all_batch_sizes)) {
        fb.set_current_stream(index + 1);
        results.push_back(
            integrand->build_channel_part(fb, {{"batch_size", chan_size}})
        );
        if (_return_sizes) {
            ret_batch_sizes.push_back(
                fb.batch_size({results.back().at("indices_acc")})
            );
        }
        ++index;
    }
    fb.set_current_stream(0);

    // Combine per-channel results into a single NamedVector
    NamedVector<Value> common_results;
    for (const auto& key : results.at(0).keys()) {
        ValueVec values;
        for (auto& result : results) {
            values.push_back(result.at(key));
        }
        auto [cat, cat_sizes] = fb.batch_cat(values);
        if (key == "indices_acc") {
            common_results.push_back(
                key, fb.add_int(fb.offset_indices(batch_sizes, cat_sizes), cat)
            );
        } else {
            common_results.push_back(key, cat);
        }
    }

    auto output = _integrands.at(0)->build_common_part(fb, common_results);
    if (_return_sizes) {
        output.push_back(
            "return_batch_sizes",
            _integrands.at(0)->_drop_cuts_and_rescale
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
        std::visit(
            Overloaded{
                [](std::monostate) {},
                [&](const auto& discrete_before) {
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
            [&](const auto& admap) {
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
                [&](const auto& discrete_after) {
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
