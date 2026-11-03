#include "madspace/driver/lhe_output.hpp"

#include "madspace/util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>

using namespace madspace;
using json = nlohmann::json;

namespace {

std::size_t cantor_pairing(std::size_t i, std::size_t j) {
    return (i + j) * (i + j + 1) / 2 + i;
}

std::size_t cantor_pairing(std::size_t i, std::size_t j, std::size_t k) {
    return cantor_pairing(cantor_pairing(i, j), k);
}

int pdg_color_type(int pdg_id, const std::unordered_map<int, int>& pdg_color_types) {
    if (auto search = pdg_color_types.find(pdg_id); search != pdg_color_types.end()) {
        return search->second;
    }
    return 1;
}

std::tuple<int, int> compute_decay_color(
    const Topology::Decay& decay,
    std::size_t color_slot,
    std::size_t colors_size,
    int color_type,
    const std::vector<std::tuple<int, int>>& prop_colors,
    int& sign_flip
) {
    std::vector<int> decay_colors, decay_anti_colors;
    for (std::size_t child_index : decay.child_indices) {
        auto [color, anti_color] =
            prop_colors.at(colors_size * child_index + color_slot);
        decay_colors.push_back(color);
        decay_anti_colors.push_back(anti_color);
    }
    for (int& color : decay_colors) {
        for (int& anti_color : decay_anti_colors) {
            if (color == anti_color) {
                color = 0;
                anti_color = 0;
            }
        }
    }
    decay_colors.erase(
        std::remove_if(
            decay_colors.begin(),
            decay_colors.end(),
            [](int color) { return color == 0; }
        ),
        decay_colors.end()
    );
    decay_anti_colors.erase(
        std::remove_if(
            decay_anti_colors.begin(),
            decay_anti_colors.end(),
            [](int color) { return color == 0; }
        ),
        decay_anti_colors.end()
    );

    // TODO: sign_flip is a workaround for the situation where madgraph gets the sign
    // of the propagator wrong. In this case, the sign is inferred from the color flow
    if (color_type == 1) {
        if (decay_colors.size() > 0 || decay_anti_colors.size() > 0) {
            throw std::runtime_error("Incompatible with color singlet");
        }
        return {0, 0};
    } else if (color_type == 3) {
        if (decay_colors.size() == 0 && decay_anti_colors.size() == 1) {
            sign_flip = -1;
            return {0, decay_anti_colors.at(0)};
        }
        if (decay_colors.size() != 1 || decay_anti_colors.size() > 0) {
            throw std::runtime_error("Incompatible with color triplet");
        }
        return {decay_colors.at(0), 0};
    } else if (color_type == -3) {
        if (decay_colors.size() == 1 && decay_anti_colors.size() == 0) {
            sign_flip = -1;
            return {decay_colors.at(0), 0};
        }
        if (decay_colors.size() > 0 || decay_anti_colors.size() != 1) {
            throw std::runtime_error("Incompatible with anti-color triplet");
        }
        return {0, decay_anti_colors.at(0)};
    } else if (color_type == 8) {
        if (decay_colors.size() == 0 && decay_anti_colors.size() == 0) {
            return {0, 0};
        }
        if (decay_colors.size() != 1 || decay_anti_colors.size() != 1) {
            throw std::runtime_error("Incompatible with color octet");
        }
        return {decay_colors.at(0), decay_anti_colors.at(0)};
    } else {
        throw std::runtime_error("Invalid color type");
    }
}

} // namespace

void LHEEvent::format_to(std::string& buffer) const {
    auto insert_iter = std::back_inserter(buffer);
    std::format_to(
        insert_iter,
        "<event>\n{:4} {:4} {:+.10e} {:.10e} {:.10e} {:.10e}\n",
        particles.size(),
        process_id,
        weight,
        scale,
        alpha_qed,
        alpha_qcd
    );
    for (auto particle : particles) {
        std::format_to(
            insert_iter,
            "{:4} {:4} {:4} {:4} {:4} {:4} {:+.16e} {:+.16e} {:+.16e} {:.16e} {:.16e} "
            "{:.4e} {:+.4e}\n",
            particle.pdg_id,
            particle.status_code,
            particle.mother1,
            particle.mother2,
            particle.color,
            particle.anti_color,
            particle.px,
            particle.py,
            particle.pz,
            particle.energy,
            particle.mass,
            particle.lifetime,
            particle.spin
        );
    }
    if (has_cluster_scales || has_scales) {
        std::format_to(insert_iter, "<scales");
    }
    if (has_scales) {
        std::format_to(
            insert_iter,
            " mur=\"{:.6g}\" muf=\"{:.6g}\" muf1=\"{:.6g}\" muf2=\"{:.6g}\"",
            mu_r,
            std::max(mu_f1, mu_f2),
            mu_f1,
            mu_f2
        );
    }
    if (has_cluster_scales) {
        for (std::size_t i = 1; auto& particle : particles) {
            if (particle.status_code == 1) {
                std::format_to(
                    insert_iter, " pt_clust_{}=\"{:.5f}\"", i, particle.cluster_scale
                );
            }
            ++i;
        }
    }
    if (has_cluster_scales || has_scales) {
        std::format_to(insert_iter, "></scales>\n");
    }
    if (lo_info && lo_info->qcd_power >= 0 && lo_info->has_beam1 &&
        lo_info->has_beam2) {
        // as madevent's unwgt.f writes it: the alpha_s power left at mu_R and
        // the vertex scales, then per beam n, the n pdgs, x and scales
        int n_vertices = static_cast<int>(lo_info->asrwt.size());
        std::format_to(
            insert_iter,
            "<mgrwt>\n<rscale> {} {:.8e}</rscale>\n<asrwt> {}",
            lo_info->qcd_power - n_vertices,
            lo_info->ren_scale,
            n_vertices
        );
        for (double scale : lo_info->asrwt) {
            std::format_to(insert_iter, " {:.8e}", scale);
        }
        buffer += "</asrwt>\n";
        std::array<int, 2> pdg{lo_info->pdg1, lo_info->pdg2};
        std::array<double, 2> x{lo_info->x1, lo_info->x2};
        std::array<double, 2> q{lo_info->fact_scale1, lo_info->fact_scale2};
        for (std::size_t b = 0; b < 2; ++b) {
            bool ladder = !lo_info->ladder_pdgs[b].empty();
            std::size_t n = ladder ? lo_info->ladder_pdgs[b].size() : 1;
            std::format_to(insert_iter, "<pdfrwt beam=\"{}\"> {}", b + 1, n);
            for (std::size_t e = 0; e < n; ++e) {
                std::format_to(
                    insert_iter, " {}", ladder ? lo_info->ladder_pdgs[b][e] : pdg[b]
                );
            }
            for (std::size_t e = 0; e < n; ++e) {
                std::format_to(
                    insert_iter, " {:.8e}", ladder ? lo_info->ladder_x[b][e] : x[b]
                );
            }
            for (std::size_t e = 0; e < n; ++e) {
                std::format_to(
                    insert_iter, " {:.8e}", ladder ? lo_info->ladder_q[b][e] : q[b]
                );
            }
            buffer += "</pdfrwt>\n";
        }
        buffer += "<totfact> 1.0</totfact>\n</mgrwt>\n";
    }
    if (!rwgt.empty()) {
        buffer += "<rwgt>\n";
        for (auto [id, value] : zip(rwgt_ids, rwgt)) {
            std::format_to(insert_iter, "<wgt id='{}'> {:+13.7e} </wgt>\n", id, value);
        }
        buffer += "</rwgt>\n";
    }
    buffer += "</event>\n";
}

std::size_t LHECompleter::append_helicities(const SubprocArgs& args) {
    std::size_t particle_count = args.helicities.at(0).size();
    for (auto& helicities : args.helicities) {
        if (helicities.size() != particle_count) {
            throw std::invalid_argument("Invalid number of helicities");
        }
        _helicities.insert(_helicities.end(), helicities.begin(), helicities.end());
    }
    return particle_count;
}

std::size_t
LHECompleter::append_colors(const SubprocArgs& args, std::size_t particle_count) {
    std::size_t color_count = args.color_flows.size();
    for (auto& color_flows : args.color_flows) {
        if (color_flows.size() != particle_count) {
            throw std::invalid_argument("Invalid number of particles per color");
        }
        _colors.insert(_colors.end(), color_flows.begin(), color_flows.end());
    }
    return color_count;
}

void LHECompleter::append_pdg_ids(const SubprocArgs& args, std::size_t particle_count) {
    if (args.pdg_id_weights.size() != 0 &&
        args.pdg_id_weights.size() != args.pdg_ids.size()) {
        throw std::invalid_argument("Need PDG-id option weights for every flavor");
    }
    for (std::size_t flavor = 0; auto& pdg_id_options : args.pdg_ids) {
        if (pdg_id_options.size() == 0) {
            throw std::invalid_argument(
                "Must provide at least one option per flavor index"
            );
        }
        std::span<const double> weights;
        if (args.pdg_id_weights.size() != 0) {
            weights = args.pdg_id_weights.at(flavor);
        }
        ++flavor;
        if (weights.size() != 0) {
            if (weights.size() != pdg_id_options.size()) {
                throw std::invalid_argument("Need one weight per PDG-id option");
            }
            double total = 0.;
            for (double weight : weights) {
                if (!(weight > 0.) || !std::isfinite(weight)) {
                    throw std::invalid_argument(
                        "PDG-id option weights must be positive and finite"
                    );
                }
                total += weight;
            }
            // equal weights keep the uniform draw
            if (std::any_of(weights.begin(), weights.end(), [&](double weight) {
                    return weight != weights[0];
                })) {
                _pdg_id_weight_offsets[_pdg_id_and_count.size()] =
                    _pdg_id_cumulative.size();
                double sum = 0.;
                for (double weight : weights) {
                    sum += weight;
                    _pdg_id_cumulative.push_back(sum / total);
                }
            }
        }
        _pdg_id_and_count.push_back({_pdg_ids.size(), pdg_id_options.size()});
        for (auto& pdg_ids : pdg_id_options) {
            if (pdg_ids.size() != particle_count) {
                throw std::invalid_argument("Invalid number of particles ids");
            }
            _pdg_ids.insert(_pdg_ids.end(), pdg_ids.begin(), pdg_ids.end());
        }
    }
}

void LHECompleter::append_masses(const Topology& first_topo) {
    _masses.insert(
        _masses.end(),
        first_topo.incoming_masses().begin(),
        first_topo.incoming_masses().end()
    );
    _masses.insert(
        _masses.end(),
        first_topo.outgoing_masses().begin(),
        first_topo.outgoing_masses().end()
    );
}

void LHECompleter::init_propagator_data(
    const Topology& topo,
    const SubprocArgs& args,
    const std::vector<std::size_t>& colors,
    const std::vector<std::size_t>& permutation,
    std::vector<double>& e_min,
    std::vector<int>& momentum_masks,
    std::vector<std::tuple<int, int>>& prop_colors,
    std::vector<int>& resonant_prop_indices
) const {
    std::size_t decay_count = topo.decays().size();
    e_min.clear();
    e_min.resize(decay_count);
    momentum_masks.clear();
    momentum_masks.resize(decay_count);
    prop_colors.clear();
    prop_colors.resize(decay_count * colors.size());
    resonant_prop_indices.clear();
    resonant_prop_indices.resize(decay_count, -1);

    // permutation maps external leg -> topology slot; invert it here, since
    // using it directly mismatches whenever it isn't self-inverse.
    std::vector<std::size_t> inv_permutation(permutation.size());
    for (std::size_t leg = 0; leg < permutation.size(); ++leg) {
        inv_permutation.at(permutation.at(leg)) = leg;
    }

    // The permutation runs over all external legs, incoming first, so the
    // outgoing ones start at the incoming count -- 2 for a collision, 1 for a
    // decay. Getting this wrong silently reads another particle's color flow.
    for (auto [index, mass, perm_index] :
         zip(topo.outgoing_indices(),
             topo.outgoing_masses(),
             std::span(
                 inv_permutation.begin() + topo.incoming_masses().size(),
                 inv_permutation.end()
             ))) {
        e_min.at(index) = mass;
        momentum_masks.at(index) = 1 << perm_index;
        for (std::size_t i = 0; std::size_t color_index : colors) {
            prop_colors.at(colors.size() * index + i) =
                args.color_flows.at(color_index).at(perm_index);
            ++i;
        }
    }
}

void LHECompleter::find_resonant_propagators(
    const Topology& topo,
    const SubprocArgs& args,
    const std::vector<std::size_t>& colors,
    const std::vector<int>& propagator_pdgs,
    std::size_t prop_offset,
    std::vector<double>& e_min,
    std::vector<int>& momentum_masks,
    std::vector<std::tuple<int, int>>& prop_colors,
    std::vector<int>& resonant_prop_indices
) {
    // For each decay, the propagators hanging below it with no propagator in
    // between. A decay that never becomes a propagator is not written to the
    // LHE, so what sits under it belongs to the nearest ancestor that is: its
    // mask passes straight through. Without this a resonance separated from
    // its parent resonance by a plain internal line keeps mother (1,2) -- the
    // W of a q q~ > Z > l l(-> l W) diagram, say.
    std::vector<int> subtree_prop_masks(topo.decays().size(), 0);

    // Pass 1: resonance status from mass/width alone, no color involved.
    for (auto& decay : std::views::reverse(topo.decays())) {
        if (decay.child_indices.size() == 0) {
            continue;
        }
        if (decay.index == 0 && topo.t_integration_order().size() > 0) {
            continue;
        }

        double& e_min_item = e_min.at(decay.index);
        int& momentum_mask = momentum_masks.at(decay.index);
        int child_prop_mask = 0;
        for (std::size_t child_index : decay.child_indices) {
            e_min_item += e_min.at(child_index);
            momentum_mask |= momentum_masks.at(child_index);
            int child_prop_index = resonant_prop_indices.at(child_index);
            if (child_prop_index != -1) {
                child_prop_mask |= 1 << child_prop_index;
            } else {
                child_prop_mask |= subtree_prop_masks.at(child_index);
            }
        }
        if (e_min_item >= decay.mass) {
            subtree_prop_masks.at(decay.index) = child_prop_mask;
            continue;
        }

        resonant_prop_indices.at(decay.index) = _propagators.size() - prop_offset;
        _propagators.push_back({
            .pdg_id = decay.pdg_id,
            .momentum_mask = momentum_mask,
            .child_prop_mask = child_prop_mask,
            .mass = decay.mass,
            .width = decay.width,
        });
    }

    if (_propagators.size() == prop_offset) {
        // Nothing resonant: no color is needed, and skipping it sidesteps
        // color flows that never get validated for non-resonant parts.
        return;
    }

    // Pass 2: a decay needs its color computed only if it (or something
    // further down its own decay chain) is resonant.
    std::vector<bool> needed(topo.decays().size(), false);
    for (auto& decay : topo.decays()) {
        if (decay.child_indices.size() == 0) {
            continue;
        }
        bool is_needed = resonant_prop_indices.at(decay.index) != -1;
        if (!is_needed && decay.parent_index != decay.index) {
            is_needed = needed.at(decay.parent_index);
        }
        needed.at(decay.index) = is_needed;
    }

    // Pass 3: compute colors, only for decays marked needed above.
    for (auto& decay : std::views::reverse(topo.decays())) {
        if (decay.child_indices.size() == 0 || !needed.at(decay.index)) {
            continue;
        }
        if (decay.index == 0 && topo.t_integration_order().size() > 0) {
            continue;
        }

        // Prefer this diagram's own pdg (merge_same_topologies can share a
        // topology across diagrams with different particles per slot).
        int propagator_pdg = decay.pdg_id;
        if (decay.flat_propagator_index != Topology::no_propagator &&
            decay.flat_propagator_index < propagator_pdgs.size()) {
            propagator_pdg = propagator_pdgs.at(decay.flat_propagator_index);
        }

        int color_type = pdg_color_type(propagator_pdg, args.pdg_color_types);
        int sign_flip = 1;
        for (std::size_t i = 0; std::size_t color_index : colors) {
            prop_colors.at(colors.size() * decay.index + i) = compute_decay_color(
                decay, i, colors.size(), color_type, prop_colors, sign_flip
            );
            ++i;
        }

        int prop_index = resonant_prop_indices.at(decay.index);
        if (prop_index != -1) {
            _propagators.at(prop_offset + prop_index).pdg_id =
                sign_flip * propagator_pdg;
        }
    }
}

void LHECompleter::record_propagator_colors(
    std::size_t subproc_index,
    std::size_t diag_index,
    const std::vector<std::size_t>& colors,
    std::size_t prop_offset,
    const std::vector<std::tuple<int, int>>& prop_colors,
    const std::vector<int>& resonant_prop_indices
) {
    std::size_t prop_count = _propagators.size() - prop_offset;
    if (prop_count == 0) {
        return;
    }

    for (std::size_t i = 0; std::size_t color : colors) {
        std::size_t prop_color_offset = _propagator_colors.size();
        for (std::size_t j = resonant_prop_indices.size();
             int prop_index : std::views::reverse(resonant_prop_indices)) {
            --j;
            if (prop_index != -1) {
                _propagator_colors.push_back(prop_colors.at(colors.size() * j + i));
            }
        }
        _propagator_index_and_count[cantor_pairing(subproc_index, diag_index, color)] =
            {prop_offset, prop_color_offset, prop_count};
        ++i;
    }
}

std::pair<std::size_t, std::size_t>
LHECompleter::build_propagators(std::size_t subproc_index, const SubprocArgs& args) {
    std::vector<double> e_min;
    std::vector<int> momentum_masks;
    std::vector<std::tuple<int, int>> prop_colors;
    std::vector<int> resonant_prop_indices;

    static const nested_vector2<int> no_channel_propagator_pdgs;
    static const std::vector<int> no_diagram_propagator_pdgs;

    std::size_t diagram_count = 0;
    std::size_t max_prop_count = 0;
    for (std::size_t channel_index = 0;
         auto [topo, permutations, diag_indices, diag_colors] :
         zip(args.topologies,
             args.permutations,
             args.diagram_indices,
             args.diagram_color_indices)) {
        // diagram_propagator_pdgs is optional; fall back to decay.pdg_id if absent.
        const nested_vector2<int>& channel_propagator_pdgs =
            channel_index < args.diagram_propagator_pdgs.size()
            ? args.diagram_propagator_pdgs.at(channel_index)
            : no_channel_propagator_pdgs;
        for (std::size_t diag_in_channel = 0;
             auto [permutation, diag_index, colors] :
             zip(permutations, diag_indices, diag_colors)) {
            const std::vector<int>& propagator_pdgs =
                diag_in_channel < channel_propagator_pdgs.size()
                ? channel_propagator_pdgs.at(diag_in_channel)
                : no_diagram_propagator_pdgs;
            if (diag_index >= diagram_count) {
                diagram_count = diag_index + 1;
            }
            init_propagator_data(
                topo,
                args,
                colors,
                permutation,
                e_min,
                momentum_masks,
                prop_colors,
                resonant_prop_indices
            );
            std::size_t prop_offset = _propagators.size();
            find_resonant_propagators(
                topo,
                args,
                colors,
                propagator_pdgs,
                prop_offset,
                e_min,
                momentum_masks,
                prop_colors,
                resonant_prop_indices
            );
            record_propagator_colors(
                subproc_index,
                diag_index,
                colors,
                prop_offset,
                prop_colors,
                resonant_prop_indices
            );
            std::size_t prop_count = _propagators.size() - prop_offset;
            if (prop_count > max_prop_count) {
                max_prop_count = prop_count;
            }
            ++diag_in_channel;
        }
        ++channel_index;
    }
    return {diagram_count, max_prop_count};
}

LHECompleter::LHECompleter(
    const std::vector<SubprocArgs>& subproc_args, double bw_cutoff
) :
    _bw_cutoff(bw_cutoff), _max_particle_count(0) {
    std::size_t color_offset = 0, pdg_id_offset = 0, helicity_offset = 0,
                mass_offset = 0;
    for (std::size_t subproc_index = 0; auto& args : subproc_args) {
        std::size_t particle_count = append_helicities(args);
        std::size_t color_count = append_colors(args, particle_count);
        append_pdg_ids(args, particle_count);
        append_masses(args.topologies.at(0));
        auto [diagram_count, max_prop_count] = build_propagators(subproc_index, args);
        if (_max_particle_count < particle_count + max_prop_count) {
            _max_particle_count = particle_count + max_prop_count;
        }

        _subproc_data.push_back({
            .process_id = args.process_id,
            .color_offset = color_offset,
            .pdg_id_offset = pdg_id_offset,
            .helicity_offset = helicity_offset,
            .mass_offset = mass_offset,
            .particle_count = particle_count,
            .color_count = color_count,
            .flavor_count = args.pdg_ids.size(),
            .diagram_count = diagram_count,
            .helicity_count = args.helicities.size(),
            .incoming_count = args.topologies.at(0).incoming_masses().size(),
        });

        helicity_offset += particle_count * args.helicities.size();
        color_offset += particle_count * color_count;
        pdg_id_offset += args.pdg_ids.size();
        mass_offset += particle_count;
        ++subproc_index;
    }
}

void LHECompleter::complete_event_data(
    LHEEvent& event,
    int subprocess_index,
    int diagram_index,
    int color_index,
    int flavor_index,
    int helicity_index,
    MixMaxRandom& rand_gen
) {
    auto& subproc_data = _subproc_data.at(subprocess_index);
    if (event.particles.size() != subproc_data.particle_count) {
        throw std::runtime_error("Invalid particle number for subprocess");
    }
    // a subprocess without diagrams has no propagators to insert
    bool has_diagrams = subproc_data.diagram_count > 0;
    if (has_diagrams ? diagram_index < 0 || diagram_index >= subproc_data.diagram_count
                     : diagram_index != -1) {
        throw std::runtime_error("Diagram index out of range");
    }
    if (color_index < 0 || color_index >= subproc_data.color_count) {
        throw std::runtime_error("Color index out of range");
    }
    if (flavor_index < 0 || flavor_index >= subproc_data.flavor_count) {
        throw std::runtime_error("Flavor index out of range");
    }
    if (helicity_index < 0 || helicity_index >= subproc_data.helicity_count) {
        throw std::runtime_error("Helicity index out of range");
    }

    event.process_id = subproc_data.process_id;

    // Number of leading entries in the event record that are initial state:
    // 2 for a collision, 1 for a decay. Everything that indexes past the
    // initial state -- where resonances get inserted, what the outgoing
    // particles' mothers are, how far the momentum masks are shifted -- is
    // offset by this rather than by a hard-coded 2.
    const std::size_t n_in = subproc_data.incoming_count;

    std::size_t color_offset =
        subproc_data.color_offset + subproc_data.particle_count * color_index;
    std::size_t helicity_offset =
        subproc_data.helicity_offset + subproc_data.particle_count * helicity_index;
    std::size_t mass_offset = subproc_data.mass_offset;

    std::size_t flavor_slot = subproc_data.pdg_id_offset + flavor_index;
    auto [pdg_index, pdg_count] = _pdg_id_and_count.at(flavor_slot);
    std::size_t pdg_random;
    if (auto weights = _pdg_id_weight_offsets.find(flavor_slot);
        weights != _pdg_id_weight_offsets.end()) {
        auto begin = _pdg_id_cumulative.begin() + weights->second;
        pdg_random = std::min<std::size_t>(
            std::upper_bound(begin, begin + pdg_count, rand_gen.generate_double()) -
                begin,
            pdg_count - 1
        );
    } else {
        pdg_random = rand_gen.generate_int(pdg_count);
    }
    std::size_t pdg_offset = pdg_index + subproc_data.particle_count * pdg_random;

    for (std::size_t particle_index = 0; auto& particle : event.particles) {
        std::tie(particle.color, particle.anti_color) =
            _colors.at(color_offset + particle_index);
        particle.pdg_id = _pdg_ids.at(pdg_offset + particle_index);
        if (particle_index < n_in) {
            particle.status_code = -1;
            particle.mother1 = 0;
            particle.mother2 = 0;
        } else {
            particle.status_code = 1;
            particle.mother1 = 1;
            // A decay has a single mother, which LHE spells as
            // mother1 == mother2 rather than a (1, 2) range.
            particle.mother2 = static_cast<int>(n_in);
        }
        particle.mass = _masses.at(mass_offset + particle_index);
        particle.lifetime = 0;
        particle.spin = _helicities.at(helicity_offset + particle_index);
        ++particle_index;
    }

    // The flavor tables list the incoming legs in leg order, and a mirrored
    // event has them the other way round on the beams. LHE wants particle 1 to
    // come from beam 1, moving along +z.
    if (n_in == 2 && event.particles.at(0).pz < 0.) {
        std::swap(event.particles.at(0), event.particles.at(1));
    }

    if (!has_diagrams) {
        return;
    }
    auto find_propagators = _propagator_index_and_count.find(
        cantor_pairing(subprocess_index, diagram_index, color_index)
    );
    if (find_propagators == _propagator_index_and_count.end()) {
        return;
    }
    auto [prop_offset, prop_color_offset, prop_count] = find_propagators->second;
    std::vector<LHEParticle> new_particles;
    int resonant_prop_mask = 0;
    for (std::size_t prop_index = 0;
         auto [propagator, prop_color] :
         zip(std::span(
                 _propagators.begin() + prop_offset,
                 _propagators.begin() + prop_offset + prop_count
             ),
             std::span(
                 _propagator_colors.begin() + prop_color_offset,
                 _propagator_colors.begin() + prop_color_offset + prop_count
             ))) {
        int momentum_mask = propagator.momentum_mask;
        double e = 0, px = 0, py = 0, pz = 0;
        for (auto& particle : event.particles) {
            if (momentum_mask & 1) {
                e += particle.energy;
                px += particle.px;
                py += particle.py;
                pz += particle.pz;
            }
            momentum_mask >>= 1;
        }
        double m2 = e * e - px * px - py * py - pz * pz;
        // Once bw_cutoff exceeds mass/width the window reaches below zero,
        // where there is no invariant mass left to exclude. Squaring a
        // negative m_min would instead turn it into a large positive floor and
        // reject nearly everything, so a wider window would write *fewer*
        // resonances than a narrow one.
        double m_min = std::max(0., propagator.mass - _bw_cutoff * propagator.width);
        double m_max = propagator.mass + _bw_cutoff * propagator.width;
        if (m2 > m_min * m_min && m2 < m_max * m_max) {
            auto [color, anti_color] = prop_color;
            resonant_prop_mask |= 1 << prop_index;
            new_particles.push_back({
                .pdg_id = propagator.pdg_id,
                .status_code = 2,
                .mother1 = 1,
                .mother2 = static_cast<int>(n_in),
                .color = color,
                .anti_color = anti_color,
                .px = px,
                .py = py,
                .pz = pz,
                .energy = e,
                .mass = std::sqrt(m2),
                .lifetime = 0,
                .spin = 9,
            });
        }
        ++prop_index;
    }
    event.particles.insert(
        event.particles.begin() + n_in, new_particles.rbegin(), new_particles.rend()
    );
    // Where each propagator ended up among the ones this event actually wrote,
    // which is the order they were just inserted in: outermost first. Only the
    // propagators inside resonant_prop_mask have a row, and child_prop_mask
    // names them by prop_index, so the two orders have to be related
    // explicitly -- counting set mask bits instead lands on the wrong row as
    // soon as a propagator in between is resonant without being a child.
    std::vector<int> res_index_of_prop(prop_count, -1);
    for (std::size_t prop_index = prop_count, next_res_index = 0; prop_index-- > 0;) {
        if (resonant_prop_mask & (1 << prop_index)) {
            res_index_of_prop.at(prop_index) = static_cast<int>(next_res_index++);
        }
    }

    for (std::size_t prop_index = prop_count, res_index = 0;
         auto& propagator : std::views::reverse(
             std::span(
                 _propagators.begin() + prop_offset,
                 _propagators.begin() + prop_offset + prop_count
             )
         )) {
        --prop_index;
        if (resonant_prop_mask & (1 << prop_index)) {
            // Descend through the propagators this event left off shell: they
            // have no row of their own, so their children attach here.
            int pending_prop_mask = propagator.child_prop_mask;
            while (pending_prop_mask != 0) {
                int child_prop_index = 0;
                while ((pending_prop_mask & (1 << child_prop_index)) == 0) {
                    ++child_prop_index;
                }
                pending_prop_mask &= ~(1 << child_prop_index);
                int child_res_index = res_index_of_prop.at(child_prop_index);
                if (child_res_index == -1) {
                    pending_prop_mask |=
                        _propagators.at(prop_offset + child_prop_index).child_prop_mask;
                    continue;
                }
                auto& child_particle = event.particles.at(child_res_index + n_in);
                child_particle.mother1 = res_index + n_in + 1;
                child_particle.mother2 = res_index + n_in + 1;
            }

            int momentum_mask = propagator.momentum_mask >> n_in;
            for (auto& particle : std::span(
                     event.particles.begin() + n_in + new_particles.size(),
                     event.particles.end()
                 )) {
                if (momentum_mask & 1) {
                    particle.mother1 = res_index + n_in + 1;
                    particle.mother2 = res_index + n_in + 1;
                }
                momentum_mask >>= 1;
            }

            ++res_index;
        }
    }
}

void LHECompleter::save(const std::string& file) const {
    std::ofstream f(file);
    json j;
    j = *this;
    f << j.dump();
}

LHECompleter LHECompleter::load(const std::string& file) {
    std::ifstream f(file);
    LHECompleter lhe_completer;
    from_json(json::parse(f), lhe_completer);
    return lhe_completer;
}

void madspace::to_json(nlohmann::json& j, const LHECompleter& lhe_completer) {
    json propagator_index_and_count = json::array();
    for (auto& [key, value] : lhe_completer._propagator_index_and_count) {
        propagator_index_and_count.push_back(json::array({key, value}));
    }
    j = json{
        {"subproc_data", lhe_completer._subproc_data},
        {"process_indices", lhe_completer._process_indices},
        {"masses", lhe_completer._masses},
        {"colors", lhe_completer._colors},
        {"helicities", lhe_completer._helicities},
        {"pdg_id_and_count", lhe_completer._pdg_id_and_count},
        {"pdg_ids", lhe_completer._pdg_ids},
        {"pdg_id_weight_offsets", lhe_completer._pdg_id_weight_offsets},
        {"pdg_id_cumulative", lhe_completer._pdg_id_cumulative},
        {"propagator_index_and_count", propagator_index_and_count},
        {"propagators", lhe_completer._propagators},
        {"propagator_colors", lhe_completer._propagator_colors},
        {"bw_cutoff", lhe_completer._bw_cutoff},
        {"max_particle_count", lhe_completer._max_particle_count},
    };
}

void madspace::from_json(const nlohmann::json& j, LHECompleter& lhe_completer) {
    lhe_completer._subproc_data =
        j.at("subproc_data").get<std::vector<LHECompleter::SubprocData>>();
    lhe_completer._process_indices = j.at("process_indices").get<std::vector<int>>();
    lhe_completer._masses = j.at("masses").get<std::vector<double>>();
    lhe_completer._colors = j.at("colors").get<std::vector<std::tuple<int, int>>>();
    lhe_completer._helicities = j.at("helicities").get<std::vector<double>>();
    lhe_completer._pdg_id_and_count =
        j.at("pdg_id_and_count").get<std::vector<std::array<std::size_t, 2>>>();
    lhe_completer._pdg_ids = j.at("pdg_ids").get<std::vector<int>>();
    // absent from completers saved before weighted options: all uniform
    lhe_completer._pdg_id_weight_offsets = j.value(
        "pdg_id_weight_offsets", std::unordered_map<std::size_t, std::size_t>{}
    );
    lhe_completer._pdg_id_cumulative =
        j.value("pdg_id_cumulative", std::vector<double>{});
    lhe_completer._propagator_index_and_count = {};
    for (auto& item : j.at("propagator_index_and_count")) {
        lhe_completer._propagator_index_and_count[item.at(0).get<std::size_t>()] =
            item.at(1).get<std::array<std::size_t, 3>>();
    }
    lhe_completer._propagators =
        j.at("propagators").get<std::vector<LHECompleter::PropagatorData>>();
    lhe_completer._propagator_colors =
        j.at("propagator_colors").get<std::vector<std::tuple<int, int>>>();
    lhe_completer._bw_cutoff = j.at("bw_cutoff").get<double>();
    lhe_completer._max_particle_count = j.at("max_particle_count").get<std::size_t>();
}

void madspace::to_json(
    nlohmann::json& j, const LHECompleter::SubprocData& subproc_data
) {
    j = json{
        subproc_data.process_id,
        subproc_data.color_offset,
        subproc_data.pdg_id_offset,
        subproc_data.helicity_offset,
        subproc_data.mass_offset,
        subproc_data.particle_count,
        subproc_data.color_count,
        subproc_data.flavor_count,
        subproc_data.diagram_count,
        subproc_data.helicity_count,
        subproc_data.incoming_count,
    };
}

void madspace::from_json(
    const nlohmann::json& j, LHECompleter::SubprocData& subproc_data
) {
    subproc_data = {
        .process_id = j.at(0).get<int>(),
        .color_offset = j.at(1).get<std::size_t>(),
        .pdg_id_offset = j.at(2).get<std::size_t>(),
        .helicity_offset = j.at(3).get<std::size_t>(),
        .mass_offset = j.at(4).get<std::size_t>(),
        .particle_count = j.at(5).get<std::size_t>(),
        .color_count = j.at(6).get<std::size_t>(),
        .flavor_count = j.at(7).get<std::size_t>(),
        .diagram_count = j.at(8).get<std::size_t>(),
        .helicity_count = j.at(9).get<std::size_t>(),
        // Gridpacks written before decays were supported carry no entry here
        // and are always collisions.
        .incoming_count = j.size() > 10 ? j.at(10).get<std::size_t>() : 2,
    };
}

void madspace::to_json(
    nlohmann::json& j, const LHECompleter::PropagatorData& prop_data
) {
    j = json{
        prop_data.pdg_id,
        prop_data.momentum_mask,
        prop_data.child_prop_mask,
        prop_data.mass,
        prop_data.width,
    };
}

void madspace::from_json(
    const nlohmann::json& j, LHECompleter::PropagatorData& prop_data
) {
    prop_data = {
        .pdg_id = j.at(0).get<int>(),
        .momentum_mask = j.at(1).get<int>(),
        .child_prop_mask = j.at(2).get<int>(),
        .mass = j.at(3).get<double>(),
        .width = j.at(4).get<double>(),
    };
}

LHEFileWriter::LHEFileWriter(const std::string& file_name, const LHEMeta& meta) :
    _file_stream(file_name) {
    _file_stream << "<LesHouchesEvents version=\"3.0\">\n<header>\n";
    for (auto [name, content, escape_content] : meta.headers) {
        _file_stream
            << (escape_content
                    ? std::format("<{0}>\n<![CDATA[\n{1}\n]]>\n</{0}>\n", name, content)
                    : std::format("<{0}>\n{1}\n</{0}>\n", name, content));
    }
    _file_stream << std::format(
        "</header>\n<init>\n{} {} {:.10e} {:.10e} {} {} {} {} {} {}\n",
        meta.beam1_pdg_id,
        meta.beam2_pdg_id,
        meta.beam1_energy,
        meta.beam2_energy,
        meta.beam1_pdf_authors,
        meta.beam2_pdf_authors,
        meta.beam1_pdf_id,
        meta.beam2_pdf_id,
        meta.weight_mode,
        meta.processes.size()
    );
    for (auto process : meta.processes) {
        _file_stream << std::format(
            "{:.10e} {:.10e} {:.10e} {}\n",
            process.cross_section,
            process.cross_section_error,
            process.max_weight,
            process.process_id
        );
    }
    _file_stream << "</init>\n";
}

void LHEFileWriter::write(const LHEEvent& event) {
    std::string buffer;
    event.format_to(buffer);
    _file_stream << buffer;
}

void LHEFileWriter::write_string(const std::string& str) { _file_stream << str; }

LHEFileWriter::~LHEFileWriter() { _file_stream << "</LesHouchesEvents>\n"; }
